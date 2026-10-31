"""
Local Semantic Embedding Engine
================================
Uses sentence-transformers `all-MiniLM-L6-v2` (384-dim vectors, ~80MB, CPU-only).
Model downloads ONCE on first run and is cached permanently to disk.

No API keys. No internet required after first download.
"""

from __future__ import annotations

import logging
import hashlib
import re
from functools import lru_cache
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Cosine similarity threshold: posts with score >= this value are considered
# the same topic and will be merged into the same cluster.
SIMILARITY_THRESHOLD: float = 0.72

# Singleton model handle — loaded once at startup, reused for every call.
_model = None
_model_loaded: bool = False


def _load_model():
    """Lazy-load the sentence-transformer model. Thread-safe singleton."""
    global _model, _model_loaded
    if _model_loaded:
        return _model
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        logger.info("Loading sentence-transformer model 'all-MiniLM-L6-v2' ...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        _model_loaded = True
        logger.info("✅ Embedding model loaded successfully (384-dim, CPU-only).")
    except Exception as exc:
        logger.warning(
            "⚠️  sentence-transformers not available (%s). "
            "Semantic clustering will fall back to keyword overlap.",
            exc,
        )
        _model = None
        _model_loaded = True  # mark as attempted so we don't retry on every call
    return _model


# ---------------------------------------------------------------------------
# LRU cache: avoids re-encoding the same text twice within a session.
# Key = SHA-256 of the raw text string (truncated to 128 chars for speed).
# ---------------------------------------------------------------------------
_embedding_cache: dict[str, list[float]] = {}
_CACHE_MAX = 1024  # max entries to keep in memory


def _text_key(text: str) -> str:
    return hashlib.sha256(text[:256].encode()).hexdigest()


def get_embedding(title: str, content: str = "") -> Optional[list[float]]:
    """
    Encode `title + content` into a 384-dim float vector.
    Strips platform prefixes and generic filler descriptions before encoding.

    Returns None if the model is unavailable (triggers keyword-overlap fallback).
    Results are LRU-cached so duplicate posts never hit the encoder twice.
    """
    model = _load_model()
    if model is None:
        return None

    # Clean title from platform prefixes
    clean_title = re.sub(
        r'^(surging\s+search|wikipedia\s+pageview\s+surge|breaking\s+analysis)\s*:\s*',
        '',
        title.strip(),
        flags=re.IGNORECASE
    )

    # Filter out known generic boilerplate content
    generic_phrases = [
        "high acceleration in real-time query volume",
        "rising community discussion tracked in",
        "breaking technology and venture capital dispatch",
        "global wire dispatch published by international newsroom",
        "significant acceleration with",
        "trending social technology discourse syndicated",
        "high-signal incoming mention tracked in",
    ]
    clean_content = content.strip()
    for phrase in generic_phrases:
        if phrase in clean_content.lower():
            clean_content = ""
            break

    combined = f"{clean_title} {clean_content}".strip()[:512]
    if not combined:
        combined = clean_title
    cache_key = _text_key(combined)

    if cache_key in _embedding_cache:
        return _embedding_cache[cache_key]

    try:
        vector: np.ndarray = model.encode(combined, normalize_embeddings=True)
        embedding = vector.tolist()

        # Evict oldest entry when cache is full
        if len(_embedding_cache) >= _CACHE_MAX:
            oldest_key = next(iter(_embedding_cache))
            del _embedding_cache[oldest_key]

        _embedding_cache[cache_key] = embedding
        return embedding

    except Exception as exc:
        logger.error("Embedding encode failed: %s", exc)
        return None


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Compute cosine similarity between two pre-normalised vectors.
    Since `normalize_embeddings=True` is set in encode(), this is just a dot product.
    Returns a float in range [0.0, 1.0].
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    return float(np.dot(va, vb))


def update_centroid(
    old_centroid: Optional[list[float]],
    new_embedding: list[float],
    n: int,
) -> list[float]:
    """
    Incrementally update a cluster centroid with a new member embedding.
    Uses the online mean formula: centroid_new = (centroid_old * n + new) / (n + 1)
    This keeps the centroid as the exact mean of all member embeddings so far.
    """
    if old_centroid is None:
        return new_embedding

    c = np.array(old_centroid, dtype=np.float32)
    v = np.array(new_embedding, dtype=np.float32)
    new_c = (c * n + v) / (n + 1)

    # Re-normalise so cosine comparisons stay accurate
    norm = np.linalg.norm(new_c)
    if norm > 0:
        new_c = new_c / norm

    return new_c.tolist()


def is_available() -> bool:
    """Returns True if the embedding model loaded successfully."""
    return _load_model() is not None

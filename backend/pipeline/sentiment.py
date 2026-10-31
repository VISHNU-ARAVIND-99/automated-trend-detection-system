"""
Offline Sentiment Classifier
==============================
Uses NLTK VADER (Valence Aware Dictionary and sEntiment Reasoner) — a rule-based
lexicon that runs entirely in-process with zero API calls and zero GPU requirements.

VADER compound score → label:
  >= +0.30  →  📈 bullish   (positive news, growth, achievements)
  <= -0.30  →  🚨 crisis    (disaster, scandal, crash, emergency)
  viral heuristics hit  →  ⚡ viral  (entertainment, memes, cultural moments)
  otherwise  →  ℹ️ informational
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.models import TopicCluster, SentimentLabel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# VADER initialisation — downloads lexicon (~2MB) once, then cached to disk
# ---------------------------------------------------------------------------
_vader = None
_vader_loaded = False


def _get_vader():
    global _vader, _vader_loaded
    if _vader_loaded:
        return _vader
    try:
        import nltk  # type: ignore
        nltk.download("vader_lexicon", quiet=True)
        from nltk.sentiment.vader import SentimentIntensityAnalyzer  # type: ignore
        _vader = SentimentIntensityAnalyzer()
        _vader_loaded = True
        logger.info("✅ VADER sentiment analyser loaded.")
    except Exception as exc:
        logger.warning("⚠️  VADER unavailable (%s). Using keyword heuristics only.", exc)
        _vader = None
        _vader_loaded = True
    return _vader


# ---------------------------------------------------------------------------
# Domain keyword sets for heuristic boosting
# ---------------------------------------------------------------------------
_CRISIS_KEYWORDS = {
    "crash", "explosion", "dead", "deaths", "killed", "emergency", "disaster",
    "collapse", "recall", "breach", "hack", "scandal", "fraud", "arrest",
    "resign", "bankrupt", "layoffs", "fired", "war", "attack", "shooting",
    "earthquake", "flood", "fire", "pandemic", "outbreak", "contamination",
}

_BULLISH_KEYWORDS = {
    "record", "breakthrough", "profit", "growth", "soar", "rise", "win",
    "award", "launch", "approved", "success", "milestone", "revenue", "ipo",
    "acquisition", "deal", "partnership", "expansion", "investment",
}

_VIRAL_KEYWORDS = {
    "meme", "viral", "celebrity", "album", "movie", "trailer", "trending",
    "tiktok", "twitter", "reddit", "challenge", "drama", "rumour", "rumor",
    "gossip", "breakup", "wedding", "beef", "cancelled", "cancel",
}

_MULTI_PLATFORM_THRESHOLD = 3  # cluster must span >= 3 sources to qualify as viral


def _keyword_boost(text: str) -> dict[str, float]:
    """Return boost scores per label based on keyword matches."""
    words = set(re.sub(r"[^\w\s]", " ", text.lower()).split())
    return {
        "crisis":        len(words & _CRISIS_KEYWORDS) * 0.15,
        "bullish":       len(words & _BULLISH_KEYWORDS) * 0.12,
        "viral":         len(words & _VIRAL_KEYWORDS) * 0.18,
        "informational": 0.0,
    }


def _count_unique_sources(cluster: "TopicCluster") -> int:
    """Count how many distinct platforms contributed to this cluster."""
    sources = {p.source for p in cluster.topPosts}
    return len(sources)


def classify_sentiment(cluster: "TopicCluster") -> "SentimentLabel":
    """
    Classify the sentiment of a topic cluster using VADER + keyword heuristics.

    Completely offline — no API calls, no GPU, runs in < 1ms.
    """
    from backend.models import SentimentLabel  # avoid circular import at module level

    vader = _get_vader()

    # Gather top post titles for analysis (up to 5)
    titles = [p.title for p in cluster.topPosts[:5]]
    if not titles:
        titles = [cluster.canonicalName]

    combined_text = " ".join(titles)

    # --- VADER scoring ---
    compound: float = 0.0
    if vader:
        scores = vader.polarity_scores(combined_text)
        compound = scores["compound"]  # range: -1.0 to +1.0

    # --- Keyword boost ---
    boosts = _keyword_boost(combined_text + " " + cluster.canonicalName)

    # --- Platform diversity for viral detection ---
    unique_sources = _count_unique_sources(cluster)
    is_cross_platform = unique_sources >= _MULTI_PLATFORM_THRESHOLD

    # --- Final classification ---
    label: str
    confidence: float
    reasoning: str

    # Viral check first (cross-platform + viral keywords beat VADER)
    viral_score = boosts["viral"] + (0.25 if is_cross_platform else 0.0)
    if viral_score >= 0.25:
        label = "viral"
        confidence = min(0.95, 0.50 + viral_score)
        reasoning = (
            f"Viral keyword signals detected across {unique_sources} platforms "
            f"(viral boost: {viral_score:.2f})"
        )

    # Crisis check
    elif compound <= -0.30 or boosts["crisis"] >= 0.30:
        label = "crisis"
        raw_conf = 0.55 + abs(compound) * 0.30 + boosts["crisis"]
        confidence = min(0.97, raw_conf)
        reasoning = (
            f"Negative VADER compound score ({compound:.2f}) with "
            f"crisis keyword presence ({boosts['crisis']:.2f} boost)"
        )

    # Bullish check
    elif compound >= 0.30 or boosts["bullish"] >= 0.25:
        label = "bullish"
        raw_conf = 0.50 + compound * 0.30 + boosts["bullish"]
        confidence = min(0.95, raw_conf)
        reasoning = (
            f"Positive VADER compound score ({compound:.2f}) with "
            f"growth keyword signals ({boosts['bullish']:.2f} boost)"
        )

    # Informational default
    else:
        label = "informational"
        confidence = 0.60 + abs(compound) * 0.10
        reasoning = (
            f"Neutral sentiment profile (VADER compound: {compound:.2f}); "
            "predominantly factual or analytical content"
        )

    return SentimentLabel(
        label=label,
        confidence=round(confidence, 3),
        reasoning=reasoning,
    )


# ---------------------------------------------------------------------------
# Icon and colour helpers — used by the API response serialiser
# ---------------------------------------------------------------------------
SENTIMENT_ICONS: dict[str, str] = {
    "bullish":       "📈",
    "crisis":        "🚨",
    "viral":         "⚡",
    "informational": "ℹ️",
}

SENTIMENT_COLORS: dict[str, str] = {
    "bullish":       "#22c55e",   # green-500
    "crisis":        "#ef4444",   # red-500
    "viral":         "#f59e0b",   # amber-500
    "informational": "#3b82f6",   # blue-500
}

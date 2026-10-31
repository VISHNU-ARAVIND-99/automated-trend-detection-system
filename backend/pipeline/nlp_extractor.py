"""
NLP Extractor & Cluster Assignment
=====================================
Two-stage clustering strategy:

  Stage 1 (Semantic): Compute a 384-dim embedding for the post (all-MiniLM-L6-v2).
    Compare cosine similarity against each existing cluster centroid.
    If best_sim >= SIMILARITY_THRESHOLD (0.72) → assign to that cluster.
    Update cluster centroid with an online incremental mean.

  Stage 2 (Keyword Fallback): If embedding model is unavailable, OR no cluster
    has a centroid yet, fall back to the original keyword/entity overlap scoring.

Both stages are always active during a mixed state (some clusters have centroids,
some don't) — the best of either method wins.
"""

import re
import time
import random
from datetime import datetime, timezone
from backend.models import EntityItem, RawPost, TopicCluster, HourlyDataPoint
from backend import database
from backend.pipeline.embeddings import (
    get_embedding,
    cosine_similarity,
    update_centroid,
    is_available as embedding_available,
    SIMILARITY_THRESHOLD,
)

STOP_WORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'as', 'at',
    'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'could', 'did', 'do',
    'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'has', 'have', 'having',
    'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'it',
    'its', 'itself', 'just', 'me', 'more', 'most', 'my', 'myself', 'no', 'nor', 'not', 'now', 'of', 'off', 'on',
    'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', 'should', 'so', 'some',
    'such', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', 'this',
    'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'we', 'were', 'what', 'when', 'where',
    'which', 'while', 'who', 'whom', 'why', 'with', 'would', 'you', 'your', 'yours', 'yourself', 'yourselves',
    'says', 'said', 'new', 'update', 'breaking', 'live', 'post', 'first', 'day', 'week', 'month', 'year', 'watch',
    'video', 'news', 'world', 'report', 'reports', 'amid', 'via', 'show', 'shows', 'look', 'looks', 'see', 'sees',
    'eight', 'nine', 'ten', 'two', 'three', 'four', 'five', 'six', 'seven', 'one', 'many', 'chief', 'plans', 'plan',
    'inside', 'reveals', 'revealed', 'latest', 'today', 'among', 'across', 'high', 'last', 'part', 'make', 'makes',
    'amidst', 'aftermath', 'back', 'backs', 'face', 'faces', 'take', 'takes', 'call', 'calls', 'tell', 'tells',
    'everything', 'everyone', 'something', 'someone', 'things', 'thing', 'details', 'detail', 'reaction', 'source',
    'wikipedia', 'pageview', 'pageviews', 'surge', 'surges', 'surging', 'thread', 'threads', 'discussion', 'ongoing',
    'rising', 'dispatch', 'syndication', 'reddit', 'wire', 'trends', 'google', 'hacker', 'newsroom', 'coverage', 'developments'
}


def extract_entities(text: str) -> list[EntityItem]:
    """
    High-Signal Entity & Term Extractor:
    1. Multi-word Capitalized Named Entities (e.g., 'Federal Reserve', 'Saudi Arabia', 'OpenAI')
    2. Capitalized Proper Nouns
    3. Topical Keywords
    """
    clean = re.sub(r'https?://\S+', '', text)
    clean = re.sub(r'^(wikipedia\s+pageview\s+surge|surging\s+search|breaking\s+analysis)\s*:\s*', '', clean, flags=re.IGNORECASE)
    clean = re.sub(r'["\'""]', ' ', clean)
    clean = re.sub(r'[^\w\s-]', ' ', clean)

    entities: list[EntityItem] = []
    seen_terms: set[str] = set()

    # 1. Multi-word Capitalized Phrases
    multi_cap_pattern = r'\b[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)+\b'
    for match in re.finditer(multi_cap_pattern, clean):
        phrase = match.group(0).strip()
        words = phrase.split()
        if all(w.lower() in STOP_WORDS for w in words):
            continue
        p_lower = phrase.lower()
        if len(phrase) > 3 and p_lower not in seen_terms:
            seen_terms.add(p_lower)
            for w in words:
                seen_terms.add(w.lower())
            entities.append(EntityItem(text=phrase, type='ORG'))

    # 2. Prominent Single Capitalized Proper Nouns
    single_cap_pattern = r'\b[A-Z][a-z0-9]{2,}\b'
    for match in re.finditer(single_cap_pattern, clean):
        word = match.group(0).strip()
        w_lower = word.lower()
        if w_lower not in STOP_WORDS and w_lower not in seen_terms:
            seen_terms.add(w_lower)
            entities.append(EntityItem(text=word, type='PRODUCT'))

    # 3. Frequency-based technical keywords fallback
    words = [
        w.lower() for w in clean.split()
        if len(w) >= 4 and w.lower() not in STOP_WORDS and not w.isdigit()
    ]
    for w in words:
        if w not in seen_terms:
            seen_terms.add(w)
            entities.append(EntityItem(text=w.capitalize(), type='TECH'))
        if len(entities) >= 6:
            break

    return entities[:6]


def _keyword_affinity(post: RawPost, cluster: TopicCluster) -> float:
    """
    Original keyword/entity overlap scorer. Used as fallback when a cluster
    has no centroid embedding yet.
    """
    post_entities = [e.text.lower() for e in post.entities if e.text.lower() not in STOP_WORDS]
    post_tokens = {
        w.lower() for w in re.sub(r'[^\w\s]', ' ', post.title).split()
        if len(w) >= 4 and w.lower() not in STOP_WORDS and not w.isdigit()
    }

    score: float = 0.0
    for ent in cluster.entities:
        ent_lower = ent.lower()
        if ent_lower in STOP_WORDS:
            continue
        if ent_lower in post_entities:
            score += 6.0 if ' ' in ent else 4.0

    cluster_tokens = {
        w.lower() for w in re.sub(r'[^\w\s]', ' ', cluster.canonicalName).split()
        if len(w) >= 4 and w.lower() not in STOP_WORDS and not w.isdigit()
    }
    score += len(post_tokens.intersection(cluster_tokens)) * 2.5
    return score


def assign_to_cluster(post: RawPost) -> TopicCluster:
    """
    Assigns a post to the best matching cluster using a two-stage strategy:

    Stage 1 — Semantic (all-MiniLM-L6-v2):
        Compute post embedding, compare cosine similarity against all cluster centroids.
        Assign if best >= SIMILARITY_THRESHOLD (0.72).

    Stage 2 — Keyword Fallback:
        For any cluster without a centroid, run keyword/entity overlap scoring.
        Assign if best >= 6.0.

    If no match found: create a new cluster and store the post embedding as its centroid.
    """
    clusters = database.get_all_clusters()

    # Clean title from platform prefixes before generating embedding
    clean_title = re.sub(r'^(wikipedia\s+pageview\s+surge|surging\s+search|breaking\s+analysis)\s*:\s*', '', post.title, flags=re.IGNORECASE)
    clean_content = re.sub(r'Significant\s+acceleration\s+with\s+[\d,]+\s+visits\s+in\s+recent\s+window\.?', '', post.content, flags=re.IGNORECASE)

    post_embedding = get_embedding(clean_title, clean_content)

    best_cluster: TopicCluster | None = None
    best_score: float = 0.0
    best_is_semantic: bool = False

    for cluster in clusters:
        # --- Stage 1: Semantic similarity ---
        if post_embedding and cluster.centroidEmbedding:
            sim = cosine_similarity(post_embedding, cluster.centroidEmbedding)
            if sim > best_score and sim >= SIMILARITY_THRESHOLD:
                best_score = sim
                best_cluster = cluster
                best_is_semantic = True
            continue  # Don't also run keyword check when centroid exists

        # --- Stage 2: Keyword fallback (cluster has no centroid yet) ---
        kw_score = _keyword_affinity(post, cluster)
        if kw_score > best_score and kw_score >= 6.0:
            best_score = kw_score
            best_cluster = cluster
            best_is_semantic = False

    if best_cluster:
        best_cluster.postCountTotal += 1
        best_cluster.volumeLast1h += 1
        best_cluster.lastUpdated = post.timestamp

        # Update centroid: if embedding available, fold into running mean
        if post_embedding:
            best_cluster.centroidEmbedding = update_centroid(
                best_cluster.centroidEmbedding,
                post_embedding,
                n=best_cluster.postCountTotal - 1,  # count before this post
            )

        # Add new novel entities (keep top 8)
        existing_ents_lower = {e.lower() for e in best_cluster.entities}
        for ent in post.entities:
            if ent.text.lower() not in existing_ents_lower:
                best_cluster.entities.append(ent.text)
                existing_ents_lower.add(ent.text.lower())
        best_cluster.entities = best_cluster.entities[:8]

        post.clusterId = best_cluster.id
        database.save_cluster(best_cluster)
        return best_cluster

    # -----------------------------------------------------------------------
    # No match — create a brand new cluster
    # -----------------------------------------------------------------------
    multi_word_ent = next((e for e in post.entities if ' ' in e.text and len(e.text) > 5), None)
    if multi_word_ent:
        secondary = next((e for e in post.entities if e.text != multi_word_ent.text and e.text not in multi_word_ent.text), None)
        topic_name = f"{multi_word_ent.text}: {secondary.text} Developments" if secondary else f"{multi_word_ent.text} Live Coverage"
    elif len(post.entities) >= 2:
        topic_name = f"{post.entities[0].text} & {post.entities[1].text} Ongoing Discussion"
    elif len(post.entities) == 1:
        topic_name = f"{post.entities[0].text} Topic Thread"
    else:
        headline_tokens = [w for w in post.title.split() if w.lower() not in STOP_WORDS][:4]
        topic_name = " ".join(headline_tokens) if len(headline_tokens) >= 2 else "Emerging News Story"

    cluster_id = f"cluster-{int(time.time() * 1000)}-{random.randint(100, 999)}"
    new_cluster = TopicCluster(
        id=cluster_id,
        canonicalName=topic_name,
        entities=[e.text for e in post.entities],
        postCountTotal=1,
        volumeLast1h=1,
        baselineHourlyRate=2.0,
        baselineStdDev=1.0,
        velocityMultiplier=0.5,
        zScore=-0.5,
        status='normal',
        firstSeen=post.timestamp,
        lastUpdated=post.timestamp,
        centroidEmbedding=post_embedding,  # seed centroid with this first post
        hourlyHistory=[
            HourlyDataPoint(hourLabel='-3h', mentions=2),
            HourlyDataPoint(hourLabel='-2h', mentions=2),
            HourlyDataPoint(hourLabel='-1h', mentions=2),
            HourlyDataPoint(hourLabel='Now', mentions=1),
        ]
    )

    post.clusterId = cluster_id
    database.save_cluster(new_cluster)
    return new_cluster

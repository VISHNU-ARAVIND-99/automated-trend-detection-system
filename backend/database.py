import sqlite3
import json
import os
from datetime import datetime, timezone
from typing import Optional
from backend.models import (
    RawPost,
    TopicCluster,
    TrendAlert,
    PipelineConfig,
    EntityItem,
    HourlyDataPoint,
    AISummary,
    SourceLink,
    SentimentLabel,
)

DB_PATH = os.path.join(os.getcwd(), "trends.db")

def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()

    # Raw posts table
    c.execute("""
        CREATE TABLE IF NOT EXISTS raw_posts (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            external_id TEXT UNIQUE NOT NULL,
            title TEXT NOT NULL,
            content TEXT,
            url TEXT,
            author TEXT,
            subreddit TEXT,
            timestamp TEXT NOT NULL,
            entities_json TEXT,
            cluster_id TEXT
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_posts_timestamp ON raw_posts(timestamp)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_posts_cluster_id ON raw_posts(cluster_id)")

    # Topic clusters table
    c.execute("""
        CREATE TABLE IF NOT EXISTS topic_clusters (
            id TEXT PRIMARY KEY,
            canonical_name TEXT NOT NULL,
            entities_json TEXT,
            post_count_total INTEGER DEFAULT 0,
            volume_last_1h INTEGER DEFAULT 0,
            baseline_hourly_rate REAL DEFAULT 2.0,
            baseline_std_dev REAL DEFAULT 1.0,
            velocity_multiplier REAL DEFAULT 0.5,
            z_score REAL DEFAULT -0.5,
            status TEXT DEFAULT 'normal',
            first_seen TEXT NOT NULL,
            last_updated TEXT NOT NULL,
            hourly_history_json TEXT,
            ai_summary_json TEXT
        )
    """)

    # Alerts history table
    c.execute("""
        CREATE TABLE IF NOT EXISTS alerts_history (
            id TEXT PRIMARY KEY,
            cluster_id TEXT NOT NULL,
            topic_name TEXT NOT NULL,
            velocity_multiplier REAL,
            volume_1h INTEGER,
            baseline_rate REAL,
            z_score REAL,
            timestamp TEXT NOT NULL,
            dispatched_to_webhook INTEGER DEFAULT 0,
            webhook_target TEXT,
            payload_preview_json TEXT,
            ai_summary_text TEXT,
            source_links_json TEXT
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_alerts_cluster ON alerts_history(cluster_id, timestamp)")

    # Pipeline config table (single row key-value)
    c.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_config (
            key TEXT PRIMARY KEY,
            value_json TEXT
        )
    """)

    conn.commit()

    # Non-breaking schema migrations — safe to run on existing databases
    migrations = [
        "ALTER TABLE topic_clusters ADD COLUMN centroid_embedding_json TEXT",
        "ALTER TABLE topic_clusters ADD COLUMN sentiment_json TEXT",
    ]
    for sql in migrations:
        try:
            c.execute(sql)
            conn.commit()
        except Exception:
            pass  # Column already exists — skip silently

    # Prune any duplicate posts keeping only the most recent instance per title
    try:
        c.execute("""
            DELETE FROM raw_posts
            WHERE id NOT IN (
                SELECT id FROM (
                    SELECT id, ROW_NUMBER() OVER (PARTITION BY LOWER(TRIM(title)) ORDER BY timestamp DESC) as rn
                    FROM raw_posts
                ) WHERE rn = 1
            )
        """)
        conn.commit()
    except Exception:
        pass

    conn.close()

# --- Post Operations ---

def post_exists(external_id: str) -> bool:
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT 1 FROM raw_posts WHERE external_id = ?", (external_id,))
    exists = c.fetchone() is not None
    conn.close()
    return exists

def title_exists(title: str) -> bool:
    """Checks if a post with the same normalized title already exists in the database."""
    if not title or not title.strip():
        return False
    conn = get_db_connection()
    c = conn.cursor()
    clean = title.strip().lower()
    c.execute("SELECT 1 FROM raw_posts WHERE LOWER(TRIM(title)) = ? LIMIT 1", (clean,))
    exists = c.fetchone() is not None
    conn.close()
    return exists

def cleanup_duplicate_posts() -> int:
    """Removes duplicate rows with identical normalized titles, keeping the latest one."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("""
        DELETE FROM raw_posts
        WHERE id NOT IN (
            SELECT id FROM (
                SELECT id, ROW_NUMBER() OVER (PARTITION BY LOWER(TRIM(title)) ORDER BY timestamp DESC) as rn
                FROM raw_posts
            ) WHERE rn = 1
        )
    """)
    deleted = c.rowcount
    conn.commit()
    conn.close()
    return deleted

def sync_cluster_metrics():
    """Synchronizes cluster post counts and 1h volumes with real database records."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("""
        UPDATE topic_clusters
        SET post_count_total = (
            SELECT COUNT(*) FROM raw_posts WHERE raw_posts.cluster_id = topic_clusters.id
        ),
        volume_last_1h = (
            SELECT COUNT(*) FROM raw_posts WHERE raw_posts.cluster_id = topic_clusters.id
        )
    """)
    conn.commit()
    conn.close()

def save_raw_post(post: RawPost) -> bool:
    conn = get_db_connection()
    c = conn.cursor()
    try:
        entities_json = json.dumps([e.model_dump() for e in post.entities])
        c.execute("""
            INSERT INTO raw_posts (
                id, source, external_id, title, content, url, author, subreddit, timestamp, entities_json, cluster_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                cluster_id = excluded.cluster_id,
                entities_json = excluded.entities_json
        """, (
            post.id,
            post.source,
            post.externalId,
            post.title,
            post.content,
            post.url,
            post.author,
            post.subreddit,
            post.timestamp,
            entities_json,
            post.clusterId
        ))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def get_recent_posts(limit: int = 50) -> list[RawPost]:
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM raw_posts ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()

    posts: list[RawPost] = []
    for r in rows:
        entities_raw = json.loads(r["entities_json"] or "[]")
        entities = [EntityItem(**item) for item in entities_raw]
        posts.append(RawPost(
            id=r["id"],
            source=r["source"],
            externalId=r["external_id"],
            title=r["title"],
            content=r["content"] or "",
            url=r["url"] or "",
            author=r["author"],
            subreddit=r["subreddit"],
            timestamp=r["timestamp"],
            entities=entities,
            clusterId=r["cluster_id"]
        ))
    return posts

def get_posts_for_cluster(cluster_id: str, limit: int = 15) -> list[RawPost]:
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM raw_posts WHERE cluster_id = ? ORDER BY timestamp DESC LIMIT ?", (cluster_id, limit))
    rows = c.fetchall()
    conn.close()

    posts: list[RawPost] = []
    for r in rows:
        entities_raw = json.loads(r["entities_json"] or "[]")
        entities = [EntityItem(**item) for item in entities_raw]
        posts.append(RawPost(
            id=r["id"],
            source=r["source"],
            externalId=r["external_id"],
            title=r["title"],
            content=r["content"] or "",
            url=r["url"] or "",
            author=r["author"],
            subreddit=r["subreddit"],
            timestamp=r["timestamp"],
            entities=entities,
            clusterId=r["cluster_id"]
        ))
    return posts

def get_total_posts_count() -> int:
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM raw_posts")
    count = c.fetchone()[0]
    conn.close()
    return count

# --- Cluster Operations ---

def save_cluster(cluster: TopicCluster):
    conn = get_db_connection()
    c = conn.cursor()
    entities_json = json.dumps(cluster.entities)
    history_json = json.dumps([h.model_dump() for h in cluster.hourlyHistory])
    ai_json = json.dumps(cluster.aiSummary.model_dump()) if cluster.aiSummary else None
    centroid_json = json.dumps(cluster.centroidEmbedding) if cluster.centroidEmbedding else None
    sentiment_json = json.dumps(cluster.sentiment.model_dump()) if cluster.sentiment else None

    c.execute("""
        INSERT INTO topic_clusters (
            id, canonical_name, entities_json, post_count_total, volume_last_1h,
            baseline_hourly_rate, baseline_std_dev, velocity_multiplier, z_score,
            status, first_seen, last_updated, hourly_history_json, ai_summary_json,
            centroid_embedding_json, sentiment_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            canonical_name = excluded.canonical_name,
            entities_json = excluded.entities_json,
            post_count_total = excluded.post_count_total,
            volume_last_1h = excluded.volume_last_1h,
            baseline_hourly_rate = excluded.baseline_hourly_rate,
            baseline_std_dev = excluded.baseline_std_dev,
            velocity_multiplier = excluded.velocity_multiplier,
            z_score = excluded.z_score,
            status = excluded.status,
            last_updated = excluded.last_updated,
            hourly_history_json = excluded.hourly_history_json,
            ai_summary_json = excluded.ai_summary_json,
            centroid_embedding_json = excluded.centroid_embedding_json,
            sentiment_json = excluded.sentiment_json
    """, (
        cluster.id,
        cluster.canonicalName,
        entities_json,
        cluster.postCountTotal,
        cluster.volumeLast1h,
        cluster.baselineHourlyRate,
        cluster.baselineStdDev,
        cluster.velocityMultiplier,
        cluster.zScore,
        cluster.status,
        cluster.firstSeen,
        cluster.lastUpdated,
        history_json,
        ai_json,
        centroid_json,
        sentiment_json,
    ))
    conn.commit()
    conn.close()

def get_all_clusters() -> list[TopicCluster]:
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM topic_clusters ORDER BY velocity_multiplier DESC")
    rows = c.fetchall()
    conn.close()

    clusters: list[TopicCluster] = []
    for r in rows:
        entities = json.loads(r["entities_json"] or "[]")
        history_raw = json.loads(r["hourly_history_json"] or "[]")
        history = [HourlyDataPoint(**h) for h in history_raw]
        ai_summary = AISummary(**json.loads(r["ai_summary_json"])) if r["ai_summary_json"] else None
        top_posts = get_posts_for_cluster(r["id"], limit=8)

        centroid = json.loads(r["centroid_embedding_json"]) if r["centroid_embedding_json"] else None
        sentiment_raw = r["sentiment_json"] if r["sentiment_json"] else None
        sentiment = SentimentLabel(**json.loads(sentiment_raw)) if sentiment_raw else None

        clusters.append(TopicCluster(
            id=r["id"],
            canonicalName=r["canonical_name"],
            entities=entities,
            postCountTotal=r["post_count_total"],
            volumeLast1h=r["volume_last_1h"],
            baselineHourlyRate=r["baseline_hourly_rate"],
            baselineStdDev=r["baseline_std_dev"],
            velocityMultiplier=r["velocity_multiplier"],
            zScore=r["z_score"],
            status=r["status"],
            firstSeen=r["first_seen"],
            lastUpdated=r["last_updated"],
            topPosts=top_posts,
            hourlyHistory=history,
            aiSummary=ai_summary,
            centroidEmbedding=centroid,
            sentiment=sentiment,
        ))
    return clusters

def get_cluster(cluster_id: str) -> Optional[TopicCluster]:
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM topic_clusters WHERE id = ?", (cluster_id,))
    r = c.fetchone()
    conn.close()

    if not r:
        return None

    entities = json.loads(r["entities_json"] or "[]")
    history_raw = json.loads(r["hourly_history_json"] or "[]")
    history = [HourlyDataPoint(**h) for h in history_raw]
    ai_summary = AISummary(**json.loads(r["ai_summary_json"])) if r["ai_summary_json"] else None
    top_posts = get_posts_for_cluster(r["id"], limit=15)

    centroid = json.loads(r["centroid_embedding_json"]) if r["centroid_embedding_json"] else None
    sentiment_raw = r["sentiment_json"] if r["sentiment_json"] else None
    sentiment = SentimentLabel(**json.loads(sentiment_raw)) if sentiment_raw else None

    return TopicCluster(
        id=r["id"],
        canonicalName=r["canonical_name"],
        entities=entities,
        postCountTotal=r["post_count_total"],
        volumeLast1h=r["volume_last_1h"],
        baselineHourlyRate=r["baseline_hourly_rate"],
        baselineStdDev=r["baseline_std_dev"],
        velocityMultiplier=r["velocity_multiplier"],
        zScore=r["z_score"],
        status=r["status"],
        firstSeen=r["first_seen"],
        lastUpdated=r["last_updated"],
        topPosts=top_posts,
        hourlyHistory=history,
        aiSummary=ai_summary,
        centroidEmbedding=centroid,
        sentiment=sentiment,
    )

# --- Alert Operations ---

def save_alert(alert: TrendAlert):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO alerts_history (
            id, cluster_id, topic_name, velocity_multiplier, volume_1h, baseline_rate,
            z_score, timestamp, dispatched_to_webhook, webhook_target, payload_preview_json,
            ai_summary_text, source_links_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            dispatched_to_webhook = excluded.dispatched_to_webhook,
            webhook_target = excluded.webhook_target
    """, (
        alert.id,
        alert.clusterId,
        alert.topicName,
        alert.velocityMultiplier,
        alert.volume1h,
        alert.baselineRate,
        alert.zScore,
        alert.timestamp,
        1 if alert.dispatchedToWebhook else 0,
        alert.webhookTarget,
        json.dumps(alert.webhookPayloadPreview),
        alert.aiSummaryText,
        json.dumps([s.model_dump() for s in alert.sourceLinks])
    ))
    conn.commit()
    conn.close()

def clear_alerts_history():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("DELETE FROM alerts_history")
    conn.commit()
    conn.close()

def get_recent_alerts(limit: int = 30) -> list[TrendAlert]:
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM alerts_history ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()

    alerts: list[TrendAlert] = []
    for r in rows:
        payload = json.loads(r["payload_preview_json"] or "{}")
        sources_raw = json.loads(r["source_links_json"] or "[]")
        sources = [SourceLink(**s) for s in sources_raw]

        alerts.append(TrendAlert(
            id=r["id"],
            clusterId=r["cluster_id"],
            topicName=r["topic_name"],
            velocityMultiplier=r["velocity_multiplier"],
            volume1h=r["volume_1h"],
            baselineRate=r["baseline_rate"],
            zScore=r["z_score"],
            timestamp=r["timestamp"],
            dispatchedToWebhook=bool(r["dispatched_to_webhook"]),
            webhookTarget=r["webhook_target"],
            webhookPayloadPreview=payload,
            aiSummaryText=r["ai_summary_text"],
            sourceLinks=sources
        ))
    return alerts

def get_last_alert_for_cluster(cluster_id: str) -> Optional[TrendAlert]:
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        "SELECT * FROM alerts_history WHERE cluster_id = ? ORDER BY timestamp DESC LIMIT 1",
        (cluster_id,)
    )
    r = c.fetchone()
    conn.close()
    if not r:
        return None
    return TrendAlert(
        id=r["id"],
        clusterId=r["cluster_id"],
        topicName=r["topic_name"],
        velocityMultiplier=r["velocity_multiplier"],
        volume1h=r["volume_1h"],
        baselineRate=r["baseline_rate"],
        zScore=r["z_score"],
        timestamp=r["timestamp"],
        dispatchedToWebhook=bool(r["dispatched_to_webhook"]),
        webhookTarget=r["webhook_target"]
    )

# --- Config Operations ---

def load_pipeline_config() -> PipelineConfig:
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT value_json FROM pipeline_config WHERE key = 'main_config'")
    row = c.fetchone()
    conn.close()

    if row and row["value_json"]:
        try:
            data = json.loads(row["value_json"])
            return PipelineConfig(**data)
        except Exception:
            pass
    return PipelineConfig()

def save_pipeline_config(config: PipelineConfig):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO pipeline_config (key, value_json)
        VALUES ('main_config', ?)
        ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json
    """, (json.dumps(config.model_dump()),))
    conn.commit()
    conn.close()

def sync_cluster_metrics():
    """Recalculates postCountTotal and volumeLast1h for all clusters based on actual rows in raw_posts."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("""
        SELECT cluster_id, COUNT(*) as cnt
        FROM raw_posts
        WHERE cluster_id IS NOT NULL
        GROUP BY cluster_id
    """)
    counts = dict(c.fetchall())

    c.execute("SELECT id, canonical_name FROM topic_clusters")
    all_clusters = c.fetchall()

    for row in all_clusters:
        cid = row[0]
        actual_cnt = counts.get(cid, 0)
        c.execute("""
            UPDATE topic_clusters
            SET post_count_total = ?, volume_last_1h = ?
            WHERE id = ?
        """, (actual_cnt, actual_cnt, cid))

    # Delete non-seed clusters that have 0 posts attached
    c.execute("""
        DELETE FROM topic_clusters
        WHERE post_count_total = 0
        AND id NOT IN ('cluster-culture-records', 'cluster-fed-interest-rates', 'cluster-ai-chip-race')
    """)

    conn.commit()
    conn.close()


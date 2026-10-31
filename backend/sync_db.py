import sqlite3
import json
from datetime import datetime, timezone
from backend import database
from backend.models import TopicCluster, HourlyDataPoint, RawPost
from backend.pipeline import nlp_extractor, velocity_engine

def run_sync():
    conn = database.get_db_connection()
    c = conn.cursor()

    # 1. Create dedicated Nvidia Quantum Core cluster if it doesn't exist
    now_iso = datetime.now(timezone.utc).isoformat()
    c.execute("""
        INSERT OR REPLACE INTO topic_clusters (
            id, canonical_name, entities_json, post_count_total, volume_last_1h,
            baseline_hourly_rate, baseline_std_dev, velocity_multiplier, z_score,
            status, first_seen, last_updated, hourly_history_json, ai_summary_json
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
    """, (
        "cluster-nvidia-quantum-core",
        "Nvidia Quantum Core",
        json.dumps(["Nvidia", "Quantum Core", "GPU", "Semiconductors"]),
        5, 5, 1.5, 0.8, 3.33, 4.38, "spike_alert",
        now_iso, now_iso,
        json.dumps([
            {"hourLabel": "-1h", "mentions": 1, "isSpike": False},
            {"hourLabel": "Now", "mentions": 5, "isSpike": True}
        ]),
        None
    ))

    # 2. Re-assign the Nvidia posts to the Nvidia cluster
    c.execute("""
        UPDATE raw_posts 
        SET cluster_id = 'cluster-nvidia-quantum-core'
        WHERE title LIKE '%Nvidia Quantum Core%'
    """)
    print(f"Re-assigned {c.rowcount} Nvidia posts to cluster-nvidia-quantum-core")

    # 3. Clean up OpenAI GPT-5 cluster
    c.execute("""
        UPDATE topic_clusters
        SET canonical_name = 'OpenAI GPT-5',
            entities_json = ?,
            post_count_total = 5,
            volume_last_1h = 5,
            baseline_hourly_rate = 1.5,
            velocity_multiplier = 3.33,
            z_score = 4.38,
            status = 'spike_alert'
        WHERE id = 'cluster-1789627555550'
    """, (json.dumps(["OpenAI", "GPT-5", "LLM", "Benchmark"]),))
    print("Cleaned up OpenAI GPT-5 cluster")

    # 4. For any raw_posts where cluster_id is NULL, assign them to clusters
    c.execute("SELECT * FROM raw_posts WHERE cluster_id IS NULL")
    rows = c.fetchall()
    print(f"Found {len(rows)} raw_posts with cluster_id IS NULL")
    conn.commit()
    conn.close()

    for r in rows:
        entities_raw = json.loads(r["entities_json"] or "[]")
        post = RawPost(
            id=r["id"],
            source=r["source"],
            externalId=r["external_id"],
            title=r["title"],
            content=r["content"] or "",
            url=r["url"] or "",
            author=r["author"],
            subreddit=r["subreddit"],
            timestamp=r["timestamp"],
            entities=entities_raw,
            clusterId=None
        )
        assigned_cluster = nlp_extractor.assign_to_cluster(post)
        # Update raw_posts table with assigned clusterId
        conn2 = database.get_db_connection()
        c2 = conn2.cursor()
        c2.execute("UPDATE raw_posts SET cluster_id = ? WHERE id = ?", (assigned_cluster.id, post.id))
        conn2.commit()
        conn2.close()

    # 5. Synchronize every cluster's post_count_total and volume_last_1h to match raw_posts
    conn3 = database.get_db_connection()
    c3 = conn3.cursor()
    c3.execute("""
        UPDATE topic_clusters
        SET post_count_total = (
            SELECT COUNT(*) FROM raw_posts WHERE raw_posts.cluster_id = topic_clusters.id
        )
        WHERE id NOT IN ('cluster-culture-records', 'cluster-fed-interest-rates', 'cluster-space-telescope')
    """)
    # For seed clusters that have baseline history, ensure volume matches
    c3.execute("""
        UPDATE topic_clusters
        SET volume_last_1h = post_count_total
        WHERE id NOT IN ('cluster-culture-records', 'cluster-fed-interest-rates', 'cluster-space-telescope')
    """)
    conn3.commit()
    conn3.close()

    # 6. Recalculate velocity scoring & alerts
    velocity_engine.recalculate_velocity_and_alerts()
    print("Synchronization complete!")

if __name__ == "__main__":
    run_sync()

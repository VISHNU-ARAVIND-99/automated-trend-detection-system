import asyncio
import os
import time
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import csv
import io
import json
from fastapi.responses import FileResponse, Response

load_dotenv()

from backend.models import (
    PipelineStats,
    PipelineConfig,
    PipelineOverviewResponse,
    TopicCluster,
    RawPost,
    TrendAlert,
    ConfigUpdateRequest,
    SpikeInjectRequest,
    SummarizeRequest,
    WebhookTestRequest,
    HourlyDataPoint,
    EntityItem
)
from backend import database
from backend.pipeline import ingestion, velocity_engine, ai_summary, webhooks

# Global in-memory flags
pipeline_running = True
last_run_timestamp: str | None = None
background_task = None

async def background_ingestion_loop():
    global last_run_timestamp
    while True:
        try:
            if pipeline_running:
                await ingestion.run_ingestion_cycle()
                last_run_timestamp = datetime.now(timezone.utc).isoformat()
        except Exception as e:
            print(f"[Loop] Ingestion cycle error: {e}")
        await asyncio.sleep(20)

def seed_initial_data_if_empty():
    """Populates realistic seed clusters on fresh DB startup if empty."""
    clusters = database.get_all_clusters()
    if len(clusters) > 0:
        return

    now_iso = datetime.now(timezone.utc).isoformat()
    seeds = [
        TopicCluster(
            id="cluster-culture-records",
            canonicalName="Global Box Office & Streaming Audience Records",
            entities=["Warner Bros", "Netflix", "Box Office", "Cinema", "Hollywood", "Christopher Nolan"],
            postCountTotal=412,
            volumeLast1h=84,
            baselineHourlyRate=18.0,
            baselineStdDev=3.5,
            velocityMultiplier=4.67,
            zScore=18.8,
            status="accelerating",
            firstSeen=now_iso,
            lastUpdated=now_iso,
            hourlyHistory=[
                HourlyDataPoint(hourLabel="-6h", mentions=16),
                HourlyDataPoint(hourLabel="-5h", mentions=18),
                HourlyDataPoint(hourLabel="-4h", mentions=21),
                HourlyDataPoint(hourLabel="-3h", mentions=34),
                HourlyDataPoint(hourLabel="-2h", mentions=52),
                HourlyDataPoint(hourLabel="-1h", mentions=69),
                HourlyDataPoint(hourLabel="Now", mentions=84, isSpike=True),
            ]
        ),
        TopicCluster(
            id="cluster-fed-interest-rates",
            canonicalName="Federal Reserve Interest Rate & Treasury Yield Updates",
            entities=["Federal Reserve", "Jerome Powell", "Treasury", "Inflation", "Wall Street", "Yields"],
            postCountTotal=289,
            volumeLast1h=56,
            baselineHourlyRate=14.0,
            baselineStdDev=2.8,
            velocityMultiplier=4.0,
            zScore=15.0,
            status="accelerating",
            firstSeen=now_iso,
            lastUpdated=now_iso,
            hourlyHistory=[
                HourlyDataPoint(hourLabel="-6h", mentions=12),
                HourlyDataPoint(hourLabel="-5h", mentions=14),
                HourlyDataPoint(hourLabel="-4h", mentions=15),
                HourlyDataPoint(hourLabel="-3h", mentions=22),
                HourlyDataPoint(hourLabel="-2h", mentions=38),
                HourlyDataPoint(hourLabel="-1h", mentions=49),
                HourlyDataPoint(hourLabel="Now", mentions=56),
            ]
        ),
        TopicCluster(
            id="cluster-space-telescope",
            canonicalName="NASA James Webb Deep Space Atmospheric Discovery",
            entities=["NASA", "James Webb", "Exoplanet", "Astronomy", "ESA", "Spectrometry"],
            postCountTotal=195,
            volumeLast1h=42,
            baselineHourlyRate=9.0,
            baselineStdDev=2.1,
            velocityMultiplier=4.67,
            zScore=15.7,
            status="accelerating",
            firstSeen=now_iso,
            lastUpdated=now_iso,
            hourlyHistory=[
                HourlyDataPoint(hourLabel="-6h", mentions=8),
                HourlyDataPoint(hourLabel="-5h", mentions=9),
                HourlyDataPoint(hourLabel="-4h", mentions=11),
                HourlyDataPoint(hourLabel="-3h", mentions=18),
                HourlyDataPoint(hourLabel="-2h", mentions=28),
                HourlyDataPoint(hourLabel="-1h", mentions=35),
                HourlyDataPoint(hourLabel="Now", mentions=42),
            ]
        )
    ]

    for s in seeds:
        database.save_cluster(s)

@asynccontextmanager
async def lifespan(app: FastAPI):
    database.init_db()
    seed_initial_data_if_empty()
    database.sync_cluster_metrics()
    task = asyncio.create_task(background_ingestion_loop())
    yield
    task.cancel()

app = FastAPI(title="Automated Trend Detection System", version="2.0.0", lifespan=lifespan)

# Permissive CORS for local Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def serve_index():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "TrendPulse API running"}

def build_pipeline_stats() -> PipelineStats:
    cfg = database.load_pipeline_config()
    clusters = database.get_all_clusters()
    spikes_count = sum(1 for c in clusters if c.status == "spike_alert")
    total_posts = database.get_total_posts_count()
    alerts = database.get_recent_alerts(limit=50)

    return PipelineStats(
        isRunning=pipeline_running,
        totalPostsIngested=total_posts,
        activeClustersCount=len(clusters),
        spikesDetectedCount=spikes_count,
        alertsFiredCount=len(alerts),
        lastRunTimestamp=last_run_timestamp or datetime.now(timezone.utc).isoformat(),
        config=cfg
    )

# --- API Endpoints ---

@app.get("/api/pipeline/overview", response_model=PipelineOverviewResponse)
def get_pipeline_overview():
    stats = build_pipeline_stats()
    clusters = database.get_all_clusters()
    feed = database.get_recent_posts(limit=50)
    alerts = database.get_recent_alerts(limit=30)

    return PipelineOverviewResponse(
        status=stats,
        clusters=clusters,
        feed=feed,
        alerts=alerts
    )

@app.get("/api/pipeline/status", response_model=PipelineStats)
def get_pipeline_status():
    return build_pipeline_stats()

@app.post("/api/pipeline/toggle")
def toggle_pipeline():
    global pipeline_running
    pipeline_running = not pipeline_running
    return {"isRunning": pipeline_running}

@app.post("/api/pipeline/run-step")
async def run_pipeline_step(body: dict | None = None):
    sub = body.get("subreddit") if body else None
    result = await ingestion.run_ingestion_cycle(subreddit=sub)
    return result

@app.post("/api/pipeline/inject-spike")
def inject_breaking_spike(req: SpikeInjectRequest):
    topic_name = (req.topicName or "OpenAI GPT-5 Autonomous Release").strip()
    entity_name = (req.entityName or "OpenAI").strip()
    headline = req.headline or f"{topic_name}: Worldwide release initiates massive API surge and developer benchmark debate"

    clusters = database.get_all_clusters()
    # Match ONLY if topic_name strongly matches the canonical name (prevent cross-topic collisions)
    target_cluster = next((
        c for c in clusters
        if topic_name.lower() == c.canonicalName.lower()
        or (len(topic_name) >= 6 and topic_name.lower() in c.canonicalName.lower())
        or (len(c.canonicalName) >= 6 and c.canonicalName.lower() in topic_name.lower())
    ), None)

    now_iso = datetime.now(timezone.utc).isoformat()
    if not target_cluster:
        cid = f"cluster-{int(time.time() * 1000)}"
        target_cluster = TopicCluster(
            id=cid,
            canonicalName=topic_name,
            entities=[entity_name, "Breaking", "Trending"],
            postCountTotal=0,
            volumeLast1h=0,
            baselineHourlyRate=1.5,
            baselineStdDev=0.8,
            velocityMultiplier=1.0,
            zScore=0.0,
            status="normal",
            firstSeen=now_iso,
            lastUpdated=now_iso,
            hourlyHistory=[
                HourlyDataPoint(hourLabel="-3h", mentions=1),
                HourlyDataPoint(hourLabel="-2h", mentions=1),
                HourlyDataPoint(hourLabel="-1h", mentions=1),
                HourlyDataPoint(hourLabel="Now", mentions=1),
            ]
        )
        database.save_cluster(target_cluster)

    # Add synthetic breaking posts with varied, realistic headlines
    _SPIKE_TEMPLATES = [
        headline,
        f"Experts weigh in: What's driving the {topic_name} surge",
        f"{topic_name} — live updates and community reaction thread",
        f"Why {topic_name} is dominating every platform right now",
        f"Analysis: The data behind the {topic_name} breakout",
        f"{topic_name}: Social media volume hits 24-hour high",
        f"Newsrooms scramble to cover {topic_name} developments",
        f"Markets and media respond to {topic_name} momentum",
    ]
    _SPIKE_SOURCES = ["reddit", "x_twitter", "hackernews", "bluesky", "wire_news"]
    _SPIKE_AUTHORS = ["trend_watch", "breakingdesk", "velocity_bot", "newsradar", "live_pulse"]

    for i in range(len(_SPIKE_TEMPLATES)):
        title_text = _SPIKE_TEMPLATES[i]
        ext_id = f"ext-spike-{int(time.time() * 1000)}-{i}"
        if database.title_exists(title_text):
            continue
        post = RawPost(
            id=f"spike-post-{int(time.time() * 1000)}-{i}",
            source=_SPIKE_SOURCES[i % len(_SPIKE_SOURCES)],
            externalId=ext_id,
            title=title_text,
            content=f"Velocity spike detected: {topic_name} is seeing a sharp increase in cross-platform coverage.",
            url=f"https://reddit.com/r/technology/spike_{i}",
            author=_SPIKE_AUTHORS[i % len(_SPIKE_AUTHORS)],
            subreddit="technology",
            timestamp=now_iso,
            entities=[EntityItem(text=entity_name, type="ORG")],
            clusterId=target_cluster.id
        )
        database.save_raw_post(post)

    # Synchronize cluster metrics directly with actual posts in the database
    cluster_posts = database.get_posts_for_cluster(target_cluster.id, limit=100)
    actual_count = len(cluster_posts)
    target_cluster.postCountTotal = actual_count
    target_cluster.volumeLast1h = actual_count
    target_cluster.baselineHourlyRate = max(round(actual_count / 5.0, 1), 1.2)
    target_cluster.lastUpdated = now_iso
    database.save_cluster(target_cluster)

    triggered_alerts = velocity_engine.recalculate_velocity_and_alerts()
    updated_cluster = database.get_cluster(target_cluster.id)

    return {
        "success": True,
        "cluster": updated_cluster.model_dump() if updated_cluster else None,
        "multiplier": updated_cluster.velocityMultiplier if updated_cluster else 0,
        "alerts": [a.model_dump() for a in triggered_alerts],
    }

@app.get("/api/clusters", response_model=list[TopicCluster])
def get_clusters():
    return database.get_all_clusters()

@app.get("/api/feed", response_model=list[RawPost])
def get_feed():
    return database.get_recent_posts(limit=50)

@app.get("/api/alerts", response_model=list[TrendAlert])
def get_alerts():
    return database.get_recent_alerts(limit=30)

@app.post("/api/alert/summarize")
async def summarize_trend(req: SummarizeRequest):
    cluster = database.get_cluster(req.clusterId)
    if not cluster:
        raise HTTPException(status_code=404, detail="Cluster not found")

    summary = await ai_summary.generate_trend_brief(cluster)
    database.save_cluster(cluster)
    return summary

@app.post("/api/config")
def update_config(req: ConfigUpdateRequest):
    cfg = database.load_pipeline_config()
    if req.velocityThresholdMultiplier is not None:
        cfg.velocityThresholdMultiplier = req.velocityThresholdMultiplier
    if req.min1hVolume is not None:
        cfg.min1hVolume = req.min1hVolume
    if req.cooldownMinutes is not None:
        cfg.cooldownMinutes = req.cooldownMinutes
    if req.discordWebhookUrl is not None:
        cfg.discordWebhookUrl = req.discordWebhookUrl
    if req.slackWebhookUrl is not None:
        cfg.slackWebhookUrl = req.slackWebhookUrl
    if req.selectedSubreddits is not None:
        cfg.selectedSubreddits = req.selectedSubreddits

    database.save_pipeline_config(cfg)
    velocity_engine.recalculate_velocity_and_alerts(cfg)
    return {"success": True, "config": cfg}

@app.post("/api/webhook/test")
async def test_webhook(req: WebhookTestRequest):
    if not req.webhookUrl:
        raise HTTPException(status_code=400, detail="Webhook URL required")

    success, msg = await webhooks.test_webhook_connection(req.webhookUrl, req.platform or "discord")
    if success:
        return {"success": True, "status": 200, "message": msg}
    raise HTTPException(status_code=400, detail=msg)

@app.get("/api/export/clusters.csv")
def export_clusters_csv():
    clusters = database.get_all_clusters()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "id", "canonical_name", "status", "velocity_multiplier",
        "volume_last_1h", "baseline_hourly_rate", "z_score",
        "sentiment_label", "sentiment_confidence", "post_count_total",
        "first_seen", "last_updated"
    ])
    for c in clusters:
        sentiment_label = c.sentiment.label if c.sentiment else "unclassified"
        sentiment_conf = c.sentiment.confidence if c.sentiment else 0.0
        writer.writerow([
            c.id, c.canonicalName, c.status, c.velocityMultiplier,
            c.volumeLast1h, c.baselineHourlyRate, c.zScore,
            sentiment_label, sentiment_conf, c.postCountTotal,
            c.firstSeen, c.lastUpdated
        ])
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=trendpulse_clusters.csv"}
    )

@app.get("/api/export/alerts.json")
def export_alerts_json():
    alerts = database.get_recent_alerts(limit=500)
    data = [a.model_dump() for a in alerts]
    return Response(
        content=json.dumps(data, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=trendpulse_alerts.json"}
    )

@app.post("/api/alerts/clear")
def clear_alerts():
    database.clear_alerts_history()
    return {"success": True, "message": "Dispatched alert history cleared."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)

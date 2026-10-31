import time
import math
from datetime import datetime, timezone, timedelta
from typing import Optional
from backend.models import TopicCluster, TrendAlert, PipelineConfig, SourceLink
from backend import database
from backend.pipeline import webhooks
from backend.pipeline.sentiment import classify_sentiment, SENTIMENT_ICONS

def recalculate_velocity_and_alerts(config: Optional[PipelineConfig] = None) -> list[TrendAlert]:
    """
    Step 3: Velocity Scoring Engine & Lifecycle Manager
    Calculates rate of change and manages temporal decay for each cluster:
    - Temporal decay: Exponential decay if cluster has been inactive >1 hour
    - Baseline (mu_7d): Average hourly mentions
    - Variance (sigma_7d): Standard deviation of hourly baseline
    - Recent Volume (V_1h): Mentions in the last 60 minutes
    - Velocity Multiplier (M): V_1h / max(mu_7d, 1.5)
    - Z-Score: (V_1h - mu_7d) / max(sigma_7d, 1.0)
    - Lifecycle State: 'emerging' -> 'accelerating' -> 'spike_alert' -> 'decaying' -> 'normal'
    """
    if config is None:
        config = database.load_pipeline_config()

    clusters = database.get_all_clusters()
    triggered_alerts: list[TrendAlert] = []
    now_time = datetime.now(timezone.utc)

    for cluster in clusters:
        # Calculate cluster inactivity age
        age_hours = 0.0
        try:
            last_dt = datetime.fromisoformat(cluster.lastUpdated.replace('Z', '+00:00'))
            age_hours = max(0.0, (now_time - last_dt).total_seconds() / 3600.0)
        except Exception:
            pass

        baseline = max(cluster.baselineHourlyRate, 1.2)
        volume = cluster.volumeLast1h
        std_dev = max(cluster.baselineStdDev, 0.8)

        # Apply gentle temporal decay if inactive for more than 1.5 hours
        previous_status = cluster.status
        if age_hours > 1.5 and volume > 1:
            decay_factor = math.exp(-0.25 * (age_hours - 1.0))
            volume = max(1, int(round(volume * decay_factor)))
            cluster.volumeLast1h = volume

        multiplier = round(volume / baseline, 2)
        z_score = round((volume - baseline) / std_dev, 2)

        cluster.velocityMultiplier = multiplier
        cluster.zScore = z_score

        # Determine full lifecycle status
        if multiplier >= config.velocityThresholdMultiplier and volume >= config.min1hVolume:
            cluster.status = 'spike_alert'
        elif multiplier >= 2.5 and volume >= 4:
            cluster.status = 'accelerating'
        elif volume <= 3 and multiplier >= 1.5 and cluster.postCountTotal <= 3:
            cluster.status = 'emerging'
        elif previous_status in ('spike_alert', 'accelerating') and (multiplier < 2.0 or age_hours > 1.0):
            cluster.status = 'decaying'
        elif multiplier >= 1.2:
            cluster.status = 'monitoring'
        else:
            cluster.status = 'normal'

        # Update hourly history latest point
        if cluster.hourlyHistory:
            cluster.hourlyHistory[-1].mentions = volume
            cluster.hourlyHistory[-1].isSpike = (cluster.status == 'spike_alert')

        # Classify sentiment when cluster is actively trending
        if cluster.status in ('spike_alert', 'accelerating', 'emerging') and cluster.sentiment is None:
            try:
                cluster.sentiment = classify_sentiment(cluster)
            except Exception:
                pass  # Never block velocity logic on sentiment failure

        database.save_cluster(cluster)

        # Step 4: Alert Condition
        if cluster.status == 'spike_alert':
            last_alert = database.get_last_alert_for_cluster(cluster.id)
            is_cooldown_over = True

            if last_alert:
                try:
                    last_time = datetime.fromisoformat(last_alert.timestamp.replace('Z', '+00:00'))
                    now_time = datetime.now(timezone.utc)
                    diff_mins = (now_time - last_time).total_seconds() / 60.0
                    if diff_mins < config.cooldownMinutes:
                        is_cooldown_over = False
                except Exception:
                    pass

            if is_cooldown_over:
                alert_id = f"alert-{int(time.time() * 1000)}-{cluster.id}"
                now_iso = datetime.now(timezone.utc).isoformat()

                top_posts = database.get_posts_for_cluster(cluster.id, limit=3)
                source_links = [
                    SourceLink(title=p.title, url=p.url, source=p.source)
                    for p in top_posts
                ]

                sources_text = "\n".join([f"• [{p.title[:60]}]({p.url})" for p in top_posts[:2]]) or "None"

                sentiment_icon = SENTIMENT_ICONS.get(
                    cluster.sentiment.label if cluster.sentiment else "informational",
                    "ℹ️"
                )

                payload_preview = {
                    "content": f"{sentiment_icon} **TREND SPIKE**: {cluster.canonicalName}",
                    "embeds": [
                        {
                            "title": f"Velocity Multiplier: {multiplier}x Normal Volume",
                            "description": f"Volume in last 1 hour reached **{volume} mentions** against baseline of **{cluster.baselineHourlyRate}/hr** (Z-Score: +{z_score}σ).",
                            "color": 16724530,
                            "fields": [
                                {"name": "Core Topic", "value": cluster.canonicalName, "inline": True},
                                {"name": "Acceleration", "value": f"{multiplier}x baseline", "inline": True},
                                {"name": "Entities Detected", "value": ", ".join(cluster.entities) or "N/A", "inline": False},
                                {"name": "Top Source Posts", "value": sources_text, "inline": False},
                            ],
                            "footer": {"text": "Automated Real-Time Trend Detection Pipeline (Python FastAPI)"},
                            "timestamp": now_iso,
                        }
                    ],
                }

                alert = TrendAlert(
                    id=alert_id,
                    clusterId=cluster.id,
                    topicName=cluster.canonicalName,
                    velocityMultiplier=multiplier,
                    volume1h=volume,
                    baselineRate=cluster.baselineHourlyRate,
                    zScore=z_score,
                    timestamp=now_iso,
                    dispatchedToWebhook=False,
                    webhookPayloadPreview=payload_preview,
                    sourceLinks=source_links
                )

                # Webhook dispatch
                if config.discordWebhookUrl or config.slackWebhookUrl:
                    webhooks.dispatch_webhook(alert, config)

                database.save_alert(alert)
                triggered_alerts.append(alert)

    return triggered_alerts

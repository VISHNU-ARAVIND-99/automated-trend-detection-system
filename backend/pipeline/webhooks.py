import httpx
from typing import Optional
from backend.models import TrendAlert, PipelineConfig

def dispatch_webhook(alert: TrendAlert, config: PipelineConfig):
    """
    Synchronous or background dispatcher for Discord and Slack webhooks.
    """
    target_url = config.discordWebhookUrl or config.slackWebhookUrl
    if not target_url:
        return

    is_slack = "slack.com" in target_url
    if is_slack:
        payload = {
            "text": f"🚨 *Trend Spike Alert:* {alert.topicName} ({alert.velocityMultiplier}x Velocity)",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": (
                            f"*🚨 BREAKING TREND SPIKE DETECTED*\n"
                            f"*Topic:* {alert.topicName}\n"
                            f"*Velocity:* `{alert.velocityMultiplier}x` (1h Vol: {alert.volume1h} vs Baseline: {alert.baselineRate}/h)\n"
                            f"*Statistical Significance:* Z-Score `+{alert.zScore}σ`"
                        ),
                    },
                }
            ],
        }
    else:
        payload = alert.webhookPayloadPreview

    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.post(target_url, json=payload)
            if resp.status_code in (200, 204):
                alert.dispatchedToWebhook = True
                alert.webhookTarget = "Slack" if is_slack else "Discord"
    except Exception as e:
        print(f"[Webhook] Dispatch error: {e}")

async def test_webhook_connection(webhook_url: str, platform: str = "discord") -> tuple[bool, str]:
    """
    Tests pinging a user-provided webhook URL.
    """
    is_slack = platform == "slack" or "slack.com" in webhook_url
    if is_slack:
        payload = {
            "text": "🔔 [Trend Engine] Test webhook ping from Automated Trend Detection System. Connection verified!"
        }
    else:
        payload = {
            "content": "🔔 **Trend Engine Connection Test**\nVerified communication channel with Automated Trend Detection System."
        }

    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
            resp = await client.post(webhook_url, json=payload)
            if resp.status_code in (200, 204):
                return True, "Webhook ping sent successfully!"
            return False, f"Webhook returned HTTP {resp.status_code}"
    except Exception as e:
        return False, str(e)

import os
import re
import json
import math
from datetime import datetime, timezone
from typing import Optional
from backend.models import TopicCluster, AISummary

_client = None

def get_genai_client():
    global _client
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    if _client is None:
        try:
            from google import genai
            _client = genai.Client(api_key=api_key)
        except Exception as e:
            print(f"[AI] GenAI initialization notice: {e}")
            return None
    return _client


def _generate_offline_intelligence(cluster: TopicCluster) -> AISummary:
    """
    100% Free, Offline Intelligence Engine:
    Performs deterministic NLP extraction, catalyst reasoning, virality scoring,
    and sentiment-tailored actionable recommendations with zero external API calls.
    """
    posts = cluster.topPosts or []
    titles = [p.title.strip() for p in posts if p.title]
    sources = list({p.source for p in posts if p.source})
    entities = cluster.entities or []
    sentiment_label = cluster.sentiment.label if cluster.sentiment else "informational"
    sentiment_conf = cluster.sentiment.confidence if cluster.sentiment else 0.7

    # 1. Synthesize Executive Summary
    if titles:
        # Pick the most informative lead title (longest, non-generic)
        lead_title = max(titles[:4], key=lambda t: len(t.split()))
        if len(titles) > 1:
            sources_str = ", ".join(sources[:3]) if sources else "monitored feeds"
            summary_text = (
                f"Rapid acceleration across {sources_str} centered on '{lead_title}'. "
                f"Discussion volume reached {cluster.volumeLast1h} mentions in the last hour ({cluster.velocityMultiplier}x normal baseline)."
            )
        else:
            summary_text = f"Discussion surge detected for '{cluster.canonicalName}': {lead_title}."
    else:
        summary_text = (
            f"Sudden velocity spike detected for '{cluster.canonicalName}' with {cluster.volumeLast1h} mentions in the past hour "
            f"({cluster.velocityMultiplier}x baseline rate)."
        )

    # 2. Extract Root Cause Catalyst
    CATALYST_PATTERNS = [
        (r"(record|surges|breaks record|peak)", "Historic volume surge and record-setting milestones"),
        (r"(launch|releases|unveil|reveals|announces)", "Major product release, official unveiling, or commercial debut"),
        (r"(investigation|lawsuit|antitrust|hack|breach|fine)", "Regulatory scrutiny, legal action, or critical security event"),
        (r"(rate|inflation|yield|treasury|fed|central bank)", "Macroeconomic policy shifts and financial market recalibration"),
        (r"(discovery|telescope|observatory|space|exoplanet)", "Breakthrough scientific observation and empirical research release"),
        (r"(championship|cup|finals|goals|trade|transfers)", "High-stakes competitive outcome and sports milestone"),
        (r"(funding|raises|valuation|acquisition|merger)", "High-profile capital investment, funding round, or acquisition activity"),
    ]

    detected_catalyst = None
    all_text = " ".join(titles + [cluster.canonicalName]).lower()
    for pattern, description in CATALYST_PATTERNS:
        if re.search(pattern, all_text):
            detected_catalyst = description
            break

    entity_highlight = f" involving {', '.join(entities[:3])}" if entities else ""
    if detected_catalyst:
        why_spiking = f"{detected_catalyst}{entity_highlight}, triggering widespread commentary across {len(sources) or 1} distinct news channels."
    else:
        why_spiking = (
            f"Correlated cross-channel mentions{entity_highlight} exceeded the expected baseline rate of "
            f"{cluster.baselineHourlyRate}/hr with high statistical deviation (Z-score: +{cluster.zScore} std dev)."
        )

    # 3. Calculate Dynamic Virality Score (0–100)
    # Multiplier contributes up to 45 pts, source diversity up to 25 pts, sentiment confidence up to 15 pts, base 15 pts
    mult_points = min(45, cluster.velocityMultiplier * 4.5)
    source_points = min(25, len(sources) * 7.0)
    conf_points = min(15, sentiment_conf * 15.0)
    virality_score = min(99, max(25, int(round(15 + mult_points + source_points + conf_points))))

    # 4. Contextual Playbook Recommendations
    if sentiment_label == "crisis":
        recommended_action = (
            "Crisis containment: Prepare holding statements, monitor negative sentiment trajectory, "
            "verify factual citations against primary newsroom wire, and evaluate brand exposure."
        )
    elif sentiment_label == "bullish":
        recommended_action = (
            "Bullish momentum: Capitalize on positive audience reception, amplify key growth indicators, "
            "and prepare follow-up editorial commentary while reader interest remains elevated."
        )
    elif sentiment_label == "viral":
        recommended_action = (
            "Viral breakout: Distribute short-form clips and key takeaways immediately, join community discussions, "
            "and target high-engagement keywords for trending search algorithms."
        )
    else:
        recommended_action = (
            "Developing story: Monitor incoming feed iterations over the next 2 hours, cross-check secondary sources, "
            f"and evaluate whether velocity multiplier sustains above {cluster.velocityMultiplier}x before initiating long-form coverage."
        )

    return AISummary(
        summary=summary_text,
        whySpiking=why_spiking,
        viralityScore=virality_score,
        recommendedAction=recommended_action,
        generatedAt=datetime.now(timezone.utc).isoformat()
    )


async def generate_trend_brief(cluster: TopicCluster) -> AISummary:
    """
    Step 4 AI Enrichment:
    Uses Google Gemini if configured with a valid GEMINI_API_KEY, otherwise
    executes the 100% free, offline deterministic intelligence synthesizer.
    """
    client = get_genai_client()
    headlines_text = "\n".join([f"- {p.title} ({p.source})" for p in (cluster.topPosts or [])[:5]])

    if client:
        try:
            prompt = f"""You are a real-time trend intelligence engine.
A sudden velocity spike has been detected:
Topic: "{cluster.canonicalName}"
Current Velocity Multiplier: {cluster.velocityMultiplier}x normal volume
Mentions in last hour: {cluster.volumeLast1h} (baseline: {cluster.baselineHourlyRate}/hr)
Recent Top Headlines:
{headlines_text or 'No specific headlines provided.'}

Provide a concise, JSON-formatted trend brief with these exact keys:
{{
  "summary": "1-2 sentence objective summary of what is happening",
  "whySpiking": "The root cause driving this rapid acceleration",
  "viralityScore": a number from 1 to 100 assessing staying power,
  "recommendedAction": "Actionable advice for a social media strategist, analyst, or content team"
}}"""

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={"response_mime_type": "application/json"}
            )

            text = response.text or "{}"
            parsed = json.loads(text)

            summary = AISummary(
                summary=parsed.get("summary", "Spike in discussion volume detected across monitored channels."),
                whySpiking=parsed.get("whySpiking", "High rate of change in community posts."),
                viralityScore=int(parsed.get("viralityScore", 85)),
                recommendedAction=parsed.get("recommendedAction", "Monitor subsequent feed iterations for follow-up verification."),
                generatedAt=datetime.now(timezone.utc).isoformat()
            )
            cluster.aiSummary = summary
            return summary
        except Exception as e:
            print(f"[AI] Gemini generation notice: {e}, falling back to offline intelligence engine.")

    # High-performance offline intelligence synthesizer (0 API keys required)
    summary = _generate_offline_intelligence(cluster)
    cluster.aiSummary = summary
    return summary

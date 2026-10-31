from typing import Literal, Optional, Any
from pydantic import BaseModel, Field


PlatformSource = Literal[
    'reddit',
    'google_trends',
    'bluesky',
    'hackernews',
    'wikipedia',
    'wire_news',
    'x_twitter',
    'custom_feed'
]

ClusterStatus = Literal['normal', 'monitoring', 'accelerating', 'spike_alert', 'emerging', 'decaying']
SentimentLabelType = Literal['bullish', 'crisis', 'informational', 'viral']

class SentimentLabel(BaseModel):
    label: SentimentLabelType
    confidence: float = 0.0       # 0.0 – 1.0
    reasoning: str = ''           # human-readable 1-sentence explanation

class EntityItem(BaseModel):
    text: str
    type: Literal['ORG', 'PERSON', 'PRODUCT', 'TECH', 'MISC'] = 'MISC'

class RawPost(BaseModel):
    id: str
    source: PlatformSource
    externalId: str
    title: str
    content: str
    url: str
    author: Optional[str] = None
    subreddit: Optional[str] = None
    timestamp: str
    entities: list[EntityItem] = Field(default_factory=list)
    clusterId: Optional[str] = None

class HourlyDataPoint(BaseModel):
    hourLabel: str
    mentions: int
    isSpike: Optional[bool] = False

class AISummary(BaseModel):
    summary: str
    whySpiking: str
    viralityScore: int
    recommendedAction: str
    generatedAt: str

class TopicCluster(BaseModel):
    id: str
    canonicalName: str
    entities: list[str] = Field(default_factory=list)
    postCountTotal: int = 0
    volumeLast1h: int = 0
    baselineHourlyRate: float = 2.0
    baselineStdDev: float = 1.0
    velocityMultiplier: float = 0.5
    zScore: float = -0.5
    status: ClusterStatus = 'normal'
    firstSeen: str
    lastUpdated: str
    topPosts: list[RawPost] = Field(default_factory=list)
    hourlyHistory: list[HourlyDataPoint] = Field(default_factory=list)
    aiSummary: Optional[AISummary] = None
    centroidEmbedding: Optional[list[float]] = None   # mean semantic vector of all member posts
    sentiment: Optional[SentimentLabel] = None         # classified when status reaches spike_alert

class SourceLink(BaseModel):
    title: str
    url: str
    source: str

class TrendAlert(BaseModel):
    id: str
    clusterId: str
    topicName: str
    velocityMultiplier: float
    volume1h: int
    baselineRate: float
    zScore: float
    timestamp: str
    dispatchedToWebhook: bool = False
    webhookTarget: Optional[str] = None
    webhookPayloadPreview: dict[str, Any] = Field(default_factory=dict)
    aiSummaryText: Optional[str] = None
    sourceLinks: list[SourceLink] = Field(default_factory=list)

class PipelineConfig(BaseModel):
    velocityThresholdMultiplier: float = 5.0
    min1hVolume: int = 8
    cooldownMinutes: int = 30
    selectedSubreddits: list[str] = Field(
        default_factory=lambda: [
            'news', 'worldnews', 'sports', 'movies', 'entertainment',
            'economics', 'science', 'technology', 'todayilearned'
        ]
    )
    discordWebhookUrl: str = ''
    slackWebhookUrl: str = ''

class PipelineStats(BaseModel):
    isRunning: bool
    totalPostsIngested: int
    activeClustersCount: int
    spikesDetectedCount: int
    alertsFiredCount: int
    lastRunTimestamp: Optional[str] = None
    sourcesActive: dict[str, bool] = Field(
        default_factory=lambda: {
            'reddit': True,
            'google_trends': True,
            'bluesky': True,
            'hackernews': True,
            'wikipedia': True,
            'wire_news': True,
            'x_twitter': True,
        }
    )
    config: PipelineConfig

class PipelineOverviewResponse(BaseModel):
    status: PipelineStats
    clusters: list[TopicCluster]
    feed: list[RawPost]
    alerts: list[TrendAlert]

class ConfigUpdateRequest(BaseModel):
    velocityThresholdMultiplier: Optional[float] = None
    min1hVolume: Optional[int] = None
    cooldownMinutes: Optional[int] = None
    discordWebhookUrl: Optional[str] = None
    slackWebhookUrl: Optional[str] = None
    selectedSubreddits: Optional[list[str]] = None

class SpikeInjectRequest(BaseModel):
    topicName: Optional[str] = None
    entityName: Optional[str] = None
    volumeSpike: Optional[int] = 48
    headline: Optional[str] = None

class SummarizeRequest(BaseModel):
    clusterId: str

class WebhookTestRequest(BaseModel):
    webhookUrl: str
    platform: Optional[str] = 'discord'

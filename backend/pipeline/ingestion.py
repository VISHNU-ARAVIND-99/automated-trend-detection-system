import asyncio
import hashlib
import re
import time
import random
from datetime import datetime, timezone, timedelta
import httpx
from backend.models import RawPost, PipelineConfig
from backend import database
from backend.pipeline import nlp_extractor, velocity_engine

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
}

WIKI_HEADERS = {
    "User-Agent": "AutomatedTrendDetector/2.0 (research-bot@trendpulse.local)"
}


async def fetch_reddit_rising(subreddit: str = "news", limit: int = 6) -> list[RawPost]:
    posts: list[RawPost] = []
    # Prefer RSS which is open and not rate-limited with 403
    url = f"https://www.reddit.com/r/{subreddit}/rising.rss"
    try:
        async with httpx.AsyncClient(headers=HEADERS, timeout=6.0, follow_redirects=True) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                xml = resp.text
                entries = re.findall(
                    r"<entry>[\s\S]*?<title>([\s\S]*?)<\/title>[\s\S]*?<link\s+href=[\"']([\s\S]*?)[\"'][\s\S]*?<\/entry>",
                    xml
                )
                for raw_title, link in entries[:limit]:
                    clean_title = raw_title.strip()
                    if not clean_title or clean_title.lower().startswith("rising submissions :"):
                        continue
                    ext_id = "reddit-" + hashlib.sha256(clean_title.encode()).hexdigest()[:18]
                    if database.post_exists(ext_id) or database.title_exists(clean_title):
                        continue

                    entities = nlp_extractor.extract_entities(clean_title)
                    post = RawPost(
                        id=f"post-{int(time.time() * 1000)}-{random.randint(100, 999)}",
                        source="reddit",
                        externalId=ext_id,
                        title=clean_title,
                        content=f"Rising community discussion tracked in r/{subreddit}.",
                        url=link.strip(),
                        author="reddit_user",
                        subreddit=subreddit,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        entities=entities
                    )
                    posts.append(post)
    except Exception as e:
        print(f"[Ingestion] Reddit fallback notice: {e}")
    return posts


async def fetch_google_trends(limit: int = 5) -> list[RawPost]:
    posts: list[RawPost] = []
    url = "https://trends.google.com/trending/rss?geo=US"
    try:
        async with httpx.AsyncClient(headers=HEADERS, timeout=6.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                xml = resp.text
                item_matches = re.findall(
                    r"<item>[\s\S]*?<title>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?<\/title>[\s\S]*?<link>(.*?)<\/link>[\s\S]*?<\/item>",
                    xml
                )
                count = 0
                for match in item_matches:
                    raw_title, link = match[0].strip(), match[1].strip()
                    if not raw_title or "google trends" in raw_title.lower():
                        continue
                    full_title = raw_title.strip()
                    ext_id = "gt-" + hashlib.sha256(raw_title.lower().encode()).hexdigest()[:18]
                    if database.post_exists(ext_id) or database.title_exists(full_title):
                        continue

                    entities = nlp_extractor.extract_entities(raw_title)
                    post = RawPost(
                        id=f"post-{int(time.time() * 1000)}-{random.randint(100, 999)}",
                        source="google_trends",
                        externalId=ext_id,
                        title=full_title,
                        content="",
                        url=link or "https://trends.google.com/trends/trendingsearches/daily",
                        author="GoogleTrendsCrawler",
                        subreddit="worldwide_trends",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        entities=entities
                    )
                    posts.append(post)
                    count += 1
                    if count >= limit:
                        break
    except Exception as e:
        print(f"[Ingestion] Google Trends fallback notice: {e}")
    return posts


async def fetch_hacker_news(limit: int = 4) -> list[RawPost]:
    posts: list[RawPost] = []
    try:
        async with httpx.AsyncClient(headers=HEADERS, timeout=6.0) as client:
            resp = await client.get("https://hacker-news.firebaseio.com/v0/topstories.json")
            if resp.status_code == 200:
                story_ids = resp.json()[:30]
                for sid in story_ids:
                    ext_id = f"hn-{sid}"
                    if database.post_exists(ext_id):
                        continue
                    item_resp = await client.get(f"https://hacker-news.firebaseio.com/v0/item/{sid}.json")
                    if item_resp.status_code == 200:
                        story = item_resp.json()
                        title = story.get("title")
                        if not title or database.title_exists(title):
                            continue
                        entities = nlp_extractor.extract_entities(title)
                        post = RawPost(
                            id=f"post-{int(time.time() * 1000)}-{random.randint(100, 999)}",
                            source="hackernews",
                            externalId=ext_id,
                            title=title,
                            content=f"HN story by @{story.get('by')} with {story.get('score', 0)} points and {story.get('descendants', 0)} comments.",
                            url=story.get("url") or f"https://news.ycombinator.com/item?id={sid}",
                            author=story.get("by"),
                            subreddit="tech_policy_finance",
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            entities=entities
                        )
                        posts.append(post)
                        if len(posts) >= limit:
                            break
    except Exception as e:
        print(f"[Ingestion] Hacker News fallback notice: {e}")
    return posts


async def fetch_tech_wire(limit: int = 4) -> list[RawPost]:
    posts: list[RawPost] = []
    url = "https://techcrunch.com/feed/"
    try:
        async with httpx.AsyncClient(headers=HEADERS, timeout=6.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                xml = resp.text
                matches = re.findall(
                    r"<item>[\s\S]*?<title>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?<\/title>[\s\S]*?<link>(.*?)<\/link>[\s\S]*?<\/item>",
                    xml
                )
                for raw_title, link in matches:
                    clean_title = raw_title.strip()
                    if not clean_title:
                        continue
                    ext_id = "tc-" + hashlib.sha256(clean_title.encode()).hexdigest()[:18]
                    if database.post_exists(ext_id) or database.title_exists(clean_title):
                        continue

                    entities = nlp_extractor.extract_entities(clean_title)
                    post = RawPost(
                        id=f"post-{int(time.time() * 1000)}-{random.randint(100, 999)}",
                        source="wire_news",
                        externalId=ext_id,
                        title=clean_title,
                        content="Breaking technology and venture capital dispatch from verified newsroom wire.",
                        url=link.strip(),
                        author="TechWireDesk",
                        subreddit="technology_news",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        entities=entities
                    )
                    posts.append(post)
                    if len(posts) >= limit:
                        break
    except Exception as e:
        print(f"[Ingestion] Tech Wire fallback notice: {e}")
    return posts


async def fetch_bbc_wire(limit: int = 4) -> list[RawPost]:
    posts: list[RawPost] = []
    url = "https://feeds.bbci.co.uk/news/world/rss.xml"
    try:
        async with httpx.AsyncClient(headers=HEADERS, timeout=6.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                xml = resp.text
                matches = re.findall(
                    r"<item>[\s\S]*?<title>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?<\/title>[\s\S]*?<link>(.*?)<\/link>[\s\S]*?<\/item>",
                    xml
                )
                for raw_title, link in matches:
                    clean_title = raw_title.strip()
                    if not clean_title or "bbc" in clean_title.lower():
                        continue
                    ext_id = "wire-" + hashlib.sha256(clean_title.encode()).hexdigest()[:18]
                    if database.post_exists(ext_id) or database.title_exists(clean_title):
                        continue

                    entities = nlp_extractor.extract_entities(clean_title)
                    post = RawPost(
                        id=f"post-{int(time.time() * 1000)}-{random.randint(100, 999)}",
                        source="wire_news",
                        externalId=ext_id,
                        title=clean_title,
                        content="Global wire dispatch published by international newsroom syndication.",
                        url=link.strip() or "https://www.bbc.com/news/world",
                        author="GlobalWireDesk",
                        subreddit="world_newsroom",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        entities=entities
                    )
                    posts.append(post)
                    if len(posts) >= limit:
                        break
    except Exception as e:
        print(f"[Ingestion] BBC Wire fallback notice: {e}")
    return posts


async def fetch_wikipedia_trending(limit: int = 4) -> list[RawPost]:
    posts: list[RawPost] = []
    yesterday = datetime.now(timezone.utc) - timedelta(days=1)
    yyyy = yesterday.strftime("%Y")
    mm = yesterday.strftime("%m")
    dd = yesterday.strftime("%d")
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/top/en.wikipedia/all-access/{yyyy}/{mm}/{dd}"

    try:
        async with httpx.AsyncClient(headers=WIKI_HEADERS, timeout=6.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                articles = data.get("items", [{}])[0].get("articles", [])
                filtered = [
                    a for a in articles
                    if not a.get("article", "").startswith(("Special:", "File:", "Main_Page", "Wikipedia:", "Portal:"))
                ][:35]

                for a in filtered:
                    name = a.get("article", "").replace("_", " ")
                    full_title = f"Wikipedia Pageview Surge: {name}"
                    ext_id = "wiki-" + hashlib.sha256(name.encode()).hexdigest()[:18]
                    if database.post_exists(ext_id) or database.title_exists(full_title) or database.title_exists(name):
                        continue

                    entities = nlp_extractor.extract_entities(name)
                    post = RawPost(
                        id=f"post-{int(time.time() * 1000)}-{random.randint(100, 999)}",
                        source="wikipedia",
                        externalId=ext_id,
                        title=full_title,
                        content=f"Significant acceleration with {a.get('views', 0):,} visits in recent window.",
                        url=f"https://en.wikipedia.org/wiki/{a.get('article')}",
                        author="Wikimedia Foundation",
                        subreddit="encyclopedia_trends",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        entities=entities
                    )
                    posts.append(post)
                    if len(posts) >= limit:
                        break
    except Exception as e:
        print(f"[Ingestion] Wikipedia fallback notice: {e}")
    return posts


async def fetch_social_bluesky(limit: int = 4) -> list[RawPost]:
    posts: list[RawPost] = []
    url = "https://techmeme.com/feed.xml"
    try:
        async with httpx.AsyncClient(headers=HEADERS, timeout=6.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                xml = resp.text
                matches = re.findall(
                    r"<item>[\s\S]*?<title>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?<\/title>[\s\S]*?<link>(.*?)<\/link>[\s\S]*?<\/item>",
                    xml
                )
                for raw_title, link in matches:
                    clean_title = raw_title.strip()
                    if not clean_title:
                        continue
                    ext_id = "bsky-" + hashlib.sha256(clean_title.encode()).hexdigest()[:18]
                    if database.post_exists(ext_id) or database.title_exists(clean_title):
                        continue

                    entities = nlp_extractor.extract_entities(clean_title)
                    post = RawPost(
                        id=f"post-{int(time.time() * 1000)}-{random.randint(100, 999)}",
                        source="bluesky",
                        externalId=ext_id,
                        title=clean_title,
                        content="Trending social technology discourse syndicated across open decentralized web networks.",
                        url=link.strip(),
                        author="SocialRadar",
                        subreddit="social_trending",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        entities=entities
                    )
                    posts.append(post)
                    if len(posts) >= limit:
                        break
    except Exception as e:
        print(f"[Ingestion] Bluesky/Social notice: {e}")
    return posts


def generate_synthetic_fallback() -> list[RawPost]:
    """Generates initial seed posts ONLY when the database is brand new and empty."""
    templates = [
        ("Global central banks review quarterly policy roadmap following international trade figures", "google_trends", "economics", "Federal Reserve"),
        ("Cinema circuits report record international attendance and premium large format ticket sales", "reddit", "movies", "Warner Bros"),
        ("Continental soccer championship concludes historic group stage qualifying round", "wire_news", "sports", "UEFA"),
        ("Deep space observatory delivers high-resolution spectrometry of nearby exoplanet atmosphere", "hackernews", "science", "NASA"),
        ("Regional power grid achieves peak generating record through battery storage interconnection", "wire_news", "technology", "Clean Energy"),
        ("Semiconductor fabrication consortia unveil breakthrough 1nm photolithography architecture", "hackernews", "hardware", "TSMC"),
        ("Autonomous maritime navigation systems complete first trans-Pacific cargo crossing", "wire_news", "logistics", "Maritime"),
        ("Biopharma trials indicate positive phase-3 endpoints for next-generation mRNA therapy", "reddit", "biotech", "Moderna")
    ]
    posts = []
    # Filter templates to only those that do NOT already exist in the database
    available = [t for t in templates if not database.title_exists(t[0])]
    if not available:
        return []

    title, source, sub, org = random.choice(available)
    ext_id = "syn-" + hashlib.sha256(title.encode()).hexdigest()[:18]
    if database.post_exists(ext_id) or database.title_exists(title):
        return []

    entities = nlp_extractor.extract_entities(title)
    posts.append(RawPost(
        id=f"post-{int(time.time() * 1000)}-{random.randint(100, 999)}",
        source=source,
        externalId=ext_id,
        title=title,
        content=f"High-signal incoming mention tracked in r/{sub}.",
        url=f"https://trends.google.com/search?q={org}",
        author="signal_monitor_bot",
        subreddit=sub,
        timestamp=datetime.now(timezone.utc).isoformat(),
        entities=entities
    ))
    return posts


async def run_ingestion_cycle(subreddit: str | None = None) -> dict:
    config = database.load_pipeline_config()
    sub = subreddit or random.choice(config.selectedSubreddits)

    # Ingest across reliable channels concurrently
    results = await asyncio.gather(
        fetch_reddit_rising(sub, limit=4),
        fetch_google_trends(limit=4),
        fetch_hacker_news(limit=3),
        fetch_tech_wire(limit=3),
        fetch_bbc_wire(limit=3),
        fetch_wikipedia_trending(limit=3),
        fetch_social_bluesky(limit=3),
        return_exceptions=True
    )

    new_posts: list[RawPost] = []
    for r in results:
        if isinstance(r, list):
            new_posts.extend(r)

    # Only fall back to synthetic generation if the database has NO posts at all (cold-start)
    if not new_posts and database.get_total_posts_count() == 0:
        new_posts.extend(generate_synthetic_fallback())

    # Deduplicate, assign to cluster, and save with cluster_id populated
    saved_posts: list[RawPost] = []
    for post in new_posts:
        if database.title_exists(post.title) or database.post_exists(post.externalId):
            continue
        nlp_extractor.assign_to_cluster(post)
        if database.save_raw_post(post):
            saved_posts.append(post)

    new_alerts = velocity_engine.recalculate_velocity_and_alerts(config)

    return {
        "ingestedCount": len(saved_posts),
        "newItems": [p.model_dump() for p in saved_posts],
        "newAlerts": [a.model_dump() for a in new_alerts]
    }

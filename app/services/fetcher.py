import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

import feedparser

from app.config import FeedSource, MAX_ENTRIES_PER_FEED, RSS_FETCH_TIMEOUT
from app.models import Article
from app.normalizer import normalize_entry

logger = logging.getLogger(__name__)


def fetch_feed(source: FeedSource) -> list[Article]:
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                feedparser.parse, source.url, request_headers={"User-Agent": "NewsBot/1.0"}
            )
            feed = future.result(timeout=RSS_FETCH_TIMEOUT)
    except FuturesTimeoutError:
        logger.warning("Feed %s timed out after %ds", source.name, RSS_FETCH_TIMEOUT)
        return []
    except Exception as e:
        logger.warning("Failed to fetch feed %s: %s", source.name, e)
        return []

    if feed.bozo and not feed.entries:
        logger.warning("Feed %s returned no entries (bozo=%s)", source.name, feed.bozo_exception)
        return []

    articles: list[Article] = []
    for entry in feed.entries[:MAX_ENTRIES_PER_FEED]:
        try:
            article = normalize_entry(entry, source.name)
            if article is not None:
                articles.append(article)
        except Exception as e:
            logger.debug("Error normalizing entry in %s: %s", source.name, e)

    logger.info("Fetched %d articles from %s", len(articles), source.name)
    return articles

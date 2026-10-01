import logging
from datetime import datetime, timezone

from app.config import SOURCES
from app.services.fetcher import fetch_feed
from app.deduplicator import deduplicate
from app.storage import initialize_database, insert_articles, count_articles

logger = logging.getLogger(__name__)


def ingest_feeds() -> int:
    logger.info("Starting RSS ingestion...")
    initialize_database()

    all_articles = []
    feed_failures = 0
    per_source_stats: list[tuple[str, int]] = []

    for source in SOURCES:
        try:
            articles = fetch_feed(source)
            per_source_stats.append((source.name, len(articles)))
            all_articles.extend(articles)
        except Exception as e:
            logger.warning("Error fetching %s: %s — continuing with other sources", source.name, e)
            feed_failures += 1
            per_source_stats.append((source.name, 0))

    for name, count in per_source_stats:
        logger.info("  %s: fetched %d", name, count)

    total_fetched = len(all_articles)
    logger.info("Total fetched: %d", total_fetched)

    if not all_articles:
        logger.info("No articles fetched from any source")
        return 0

    deduped = deduplicate(all_articles)
    logger.info("After deduplication: %d", len(deduped))

    before_count = count_articles()
    inserted = insert_articles(deduped)
    after_count = count_articles()
    new_articles = after_count - before_count
    duplicates_skipped = len(deduped) - new_articles

    logger.info("New articles: %d", new_articles)
    logger.info("Duplicates skipped: %d", duplicates_skipped)
    logger.info("Feed failures: %d", feed_failures)
    logger.info("Ingestion complete. Total articles in database: %d", after_count)

    return new_articles

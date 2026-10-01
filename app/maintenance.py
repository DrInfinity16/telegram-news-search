"""Intentional development-database maintenance utilities.

These helpers never run automatically. Use them only when you deliberately
want to clean synthetic rows or inspect the local development corpus.

Examples:
    python -m app.maintenance --dry-run
    python -m app.maintenance --apply
    python -m app.maintenance --apply --reingest
"""

from __future__ import annotations

import argparse
import logging
from typing import Iterable

from app.config import SOURCES
from app import storage

logger = logging.getLogger(__name__)

KNOWN_FEED_SOURCES = {s.name for s in SOURCES}


def iter_synthetic_rows(db_path: str | None = None) -> list[dict]:
    """Return article rows whose source is not a configured RSS feed."""
    if db_path is not None:
        import os

        os.environ["DATABASE_PATH"] = db_path
        storage.close_database()

    storage.initialize_database()
    conn = storage._get_conn()
    rows = conn.execute(
        "SELECT id, title, url, source, published_at FROM articles ORDER BY source, title"
    ).fetchall()
    synthetic = []
    for row in rows:
        source = row["source"] or ""
        if source not in KNOWN_FEED_SOURCES:
            synthetic.append(
                {
                    "id": row["id"],
                    "title": row["title"],
                    "url": row["url"],
                    "source": source,
                    "published_at": row["published_at"],
                }
            )
    return synthetic


def remove_synthetic_rows(db_path: str | None = None) -> int:
    """Delete articles whose source is not in the configured RSS list.

    Returns the number of deleted rows. Does not touch configured-feed rows.
    """
    if db_path is not None:
        import os

        os.environ["DATABASE_PATH"] = db_path
        storage.close_database()

    storage.initialize_database()
    synthetic = iter_synthetic_rows()
    if not synthetic:
        return 0
    conn = storage._get_conn()
    ids = [row["id"] for row in synthetic]
    placeholders = ",".join("?" for _ in ids)
    cursor = conn.execute(f"DELETE FROM articles WHERE id IN ({placeholders})", ids)
    conn.commit()
    deleted = cursor.rowcount
    logger.info("Removed %d synthetic article rows", deleted)
    return deleted


def reingest_from_feeds() -> int:
    """Fetch RSS feeds into the current DATABASE_PATH (safe upsert/ignore)."""
    from app.services.ingest import ingest_feeds

    return ingest_feeds()


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Project 10 development DB maintenance")
    parser.add_argument("--dry-run", action="store_true", help="List synthetic rows only")
    parser.add_argument("--apply", action="store_true", help="Delete synthetic rows")
    parser.add_argument(
        "--reingest",
        action="store_true",
        help="After apply, re-fetch RSS feeds into the database",
    )
    parser.add_argument("--db", default=None, help="Optional explicit database path")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.dry_run and not args.apply:
        parser.error("Specify --dry-run or --apply")

    synthetic = iter_synthetic_rows(args.db)
    print(f"Synthetic rows found: {len(synthetic)}")
    for row in synthetic[:50]:
        print(f"  [{row['source']}] {row['title'][:80]}")
    if len(synthetic) > 50:
        print(f"  ... and {len(synthetic) - 50} more")

    if args.dry_run:
        return 0

    deleted = remove_synthetic_rows(args.db)
    print(f"Deleted synthetic rows: {deleted}")

    if args.reingest:
        inserted = reingest_from_feeds()
        print(f"New articles ingested from RSS: {inserted}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

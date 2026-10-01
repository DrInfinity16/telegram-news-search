import logging
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.evidence import (
    concept_match_display,
    evaluate_concept_coverage,
    filter_meaningful_matches,
    has_minimum_concept_coverage,
    has_minimum_thematic_evidence,
    phrase_has_informative_token,
)
from app.models import Article
from app.query_intent import QueryIntent
from app.text import MIN_SINGLE_TOKEN_CHARS, normalize_text, norm_for_search, token_in_normalized_text

logger = logging.getLogger(__name__)

_conn: sqlite3.Connection | None = None


@dataclass
class SearchStats:
    raw_lexical_matches: int = 0
    after_token_filter: int = 0
    after_evidence_filter: int = 0
    after_coverage_filter: int = 0
    candidate_cap_applied: int = 0
    ranked_count: int = 0
    returned_count: int = 0


def _get_database_path() -> str:
    return os.getenv("DATABASE_PATH", "news.db")


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        db_path = _get_database_path()
        _conn = sqlite3.connect(db_path, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
    return _conn


def initialize_database() -> None:
    conn = _get_conn()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS articles (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            url TEXT NOT NULL,
            source TEXT NOT NULL,
            summary TEXT DEFAULT '',
            author TEXT,
            published_at TEXT,
            fetched_at TEXT NOT NULL,
            ingested_at TEXT NOT NULL,
            title_norm TEXT NOT NULL DEFAULT '',
            summary_norm TEXT NOT NULL DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_articles_published ON articles(published_at);
        CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source);
        """
    )
    _migrate_normalized_columns(conn)
    conn.commit()
    logger.info("Database initialized at %s", _get_database_path())


def _migrate_normalized_columns(conn: sqlite3.Connection) -> None:
    """Add/backfill accent-normalized search columns on existing databases."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(articles)")}
    if "title_norm" not in cols:
        conn.execute("ALTER TABLE articles ADD COLUMN title_norm TEXT NOT NULL DEFAULT ''")
        logger.info("Migrated articles table: added title_norm")
    if "summary_norm" not in cols:
        conn.execute("ALTER TABLE articles ADD COLUMN summary_norm TEXT NOT NULL DEFAULT ''")
        logger.info("Migrated articles table: added summary_norm")

    rows = conn.execute(
        """
        SELECT id, title, summary, title_norm, summary_norm
        FROM articles
        """
    ).fetchall()
    updated = 0
    for row in rows:
        expected_title = norm_for_search(row["title"])
        expected_summary = norm_for_search(row["summary"] or "")
        if row["title_norm"] != expected_title or (row["summary_norm"] or " ") != expected_summary:
            conn.execute(
                "UPDATE articles SET title_norm = ?, summary_norm = ? WHERE id = ?",
                (expected_title, expected_summary, row["id"]),
            )
            updated += 1
    if updated:
        logger.info("Backfilled normalized search columns for %d articles", updated)


def insert_article(article: Article) -> bool:
    now = datetime.now(timezone.utc).isoformat()
    try:
        conn = _get_conn()
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO articles
                (id, title, url, source, summary, author, published_at, fetched_at, ingested_at,
                 title_norm, summary_norm)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                article.id,
                article.title,
                article.url,
                article.source,
                article.summary,
                article.author,
                article.published_at,
                article.fetched_at,
                now,
                norm_for_search(article.title),
                norm_for_search(article.summary),
            ),
        )
        conn.commit()
        return cursor.rowcount > 0
    except sqlite3.Error as e:
        logger.error("Error inserting article %s: %s", article.id, e)
        return False


def insert_articles(articles: list[Article]) -> int:
    now = datetime.now(timezone.utc).isoformat()
    if not articles:
        return 0
    try:
        conn = _get_conn()
        before = conn.total_changes
        conn.executemany(
            """
            INSERT OR IGNORE INTO articles
                (id, title, url, source, summary, author, published_at, fetched_at, ingested_at,
                 title_norm, summary_norm)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    a.id,
                    a.title,
                    a.url,
                    a.source,
                    a.summary,
                    a.author,
                    a.published_at,
                    a.fetched_at,
                    now,
                    norm_for_search(a.title),
                    norm_for_search(a.summary),
                )
                for a in articles
            ],
        )
        conn.commit()
        return conn.total_changes - before
    except sqlite3.Error as e:
        logger.error("Error batch inserting articles: %s", e)
        return 0


def get_article_by_id(article_id: str) -> Article | None:
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM articles WHERE id = ?", (article_id,)
    ).fetchone()
    if row is None:
        return None
    return _row_to_article(row)


def get_all_articles() -> list[Article]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM articles ORDER BY ingested_at DESC"
    ).fetchall()
    return [_row_to_article(row) for row in rows]


def search_articles_tiered(
    intent: QueryIntent,
    limit: int = 50,
    stats: SearchStats | None = None,
) -> list[Article]:
    """Lexical candidate retrieval with high recall.

    Uses accent-normalized columns for SQL, retrieves informative phrase +
    token evidence (stopwords excluded), applies no recency pre-filter,
    then filters minimum thematic evidence and caps candidates by match
    strength.
    """
    conn = _get_conn()
    out_stats = stats if stats is not None else SearchStats()

    multi_phrases = [
        p for p in intent.multi_word_phrases() if phrase_has_informative_token(p)
    ]
    retrieval_tokens = intent.retrieval_tokens()

    if not multi_phrases and not retrieval_tokens:
        out_stats.raw_lexical_matches = 0
        out_stats.after_token_filter = 0
        out_stats.after_evidence_filter = 0
        out_stats.after_coverage_filter = 0
        out_stats.candidate_cap_applied = 0
        return []

    like_conditions: list[str] = []
    like_params: list[str] = []

    for phrase in multi_phrases:
        nphrase = normalize_text(phrase)
        if not nphrase:
            continue
        like_conditions.append("title_norm LIKE ?")
        like_conditions.append("summary_norm LIKE ?")
        like_params.append(f"%{nphrase}%")
        like_params.append(f"%{nphrase}%")

    for token in retrieval_tokens:
        ntoken = normalize_text(token)
        if not ntoken:
            continue
        like_conditions.append("title_norm LIKE ?")
        like_conditions.append("summary_norm LIKE ?")
        if len(ntoken) < MIN_SINGLE_TOKEN_CHARS:
            like_params.append(f"% {ntoken} %")
            like_params.append(f"% {ntoken} %")
        else:
            like_params.append(f"%{ntoken}%")
            like_params.append(f"%{ntoken}%")

    if not like_conditions:
        out_stats.raw_lexical_matches = 0
        return []

    where_clause = " OR ".join(like_conditions)
    query = f"SELECT DISTINCT * FROM articles WHERE {where_clause}"

    try:
        rows = conn.execute(query, like_params).fetchall()
    except sqlite3.Error as e:
        logger.error("Search error: %s", e)
        return []

    candidates = [_row_to_article(row) for row in rows]
    out_stats.raw_lexical_matches = len(candidates)

    all_phrases = intent.all_phrases()
    token_filter_count = 0
    matched: list[Article] = []
    for article in candidates:
        norm_title = normalize_text(article.title)
        norm_summary = normalize_text(article.summary)
        article_matched: list[str] = []

        for phrase in all_phrases:
            if not phrase_has_informative_token(phrase):
                continue
            if _phrase_matches(phrase, norm_title, norm_summary):
                article_matched.append(phrase)

        for token in retrieval_tokens:
            if token in article_matched:
                continue
            if token_in_normalized_text(token, norm_title) or token_in_normalized_text(
                token, norm_summary
            ):
                article_matched.append(token)

        meaningful = filter_meaningful_matches(article_matched)
        if not meaningful:
            continue
        article.matched_keywords = meaningful
        matched.append(article)
        token_filter_count += 1

    out_stats.after_token_filter = token_filter_count

    evidence_matched = [
        article
        for article in matched
        if has_minimum_thematic_evidence(intent, article.matched_keywords, article)
    ]
    out_stats.after_evidence_filter = len(evidence_matched)

    coverage_matched: list[Article] = []
    for article in evidence_matched:
        if not has_minimum_concept_coverage(intent, article.matched_keywords, article):
            continue
        cov = evaluate_concept_coverage(intent, article.matched_keywords, article)
        display = concept_match_display(cov)
        if display:
            article.matched_keywords = display
        coverage_matched.append(article)
    out_stats.after_coverage_filter = len(coverage_matched)

    capped = _cap_candidates(coverage_matched, intent, limit)
    out_stats.candidate_cap_applied = len(capped)

    logger.debug(
        "Tiered search raw=%d token_filter=%d evidence=%d coverage=%d capped=%d query='%s'",
        out_stats.raw_lexical_matches,
        out_stats.after_token_filter,
        out_stats.after_evidence_filter,
        out_stats.after_coverage_filter,
        out_stats.candidate_cap_applied,
        intent.original_query,
    )
    return capped


def _phrase_matches(phrase: str, norm_title: str, norm_summary: str) -> bool:
    if not phrase:
        return False
    if not phrase_has_informative_token(phrase):
        return False
    nphrase = normalize_text(phrase)
    if not nphrase:
        return False
    if nphrase in norm_title or nphrase in norm_summary:
        return True
    words = [normalize_text(w) for w in phrase.split() if w]
    words = [w for w in words if w and len(w) >= MIN_SINGLE_TOKEN_CHARS]
    # Drop stopwords from non-contiguous multi-word evidence.
    from app.evidence import is_stopword

    words = [w for w in words if not is_stopword(w)]
    if len(words) > 1:
        return all(
            token_in_normalized_text(w, norm_title) or token_in_normalized_text(w, norm_summary)
            for w in words
        )
    return False


def _cap_candidates(articles: list[Article], intent: QueryIntent, limit: int) -> list[Article]:
    """Cap candidate count by lexical match strength, not recency alone."""
    if limit <= 0 or len(articles) <= limit:
        return articles

    orig_norm = normalize_text(intent.original_query)

    def priority(article: Article) -> tuple:
        matched_norms = [normalize_text(kw) for kw in article.matched_keywords]
        has_original = orig_norm in matched_norms
        multi_score = sum(1 for kw in article.matched_keywords if len(kw.split()) > 1)
        token_score = sum(1 for kw in article.matched_keywords if len(kw.split()) == 1)
        return (1 if has_original else 0, multi_score, token_score, article.published_at or "")

    ranked = sorted(articles, key=priority, reverse=True)
    return ranked[:limit]


def count_articles() -> int:
    conn = _get_conn()
    row = conn.execute("SELECT COUNT(*) FROM articles").fetchone()
    return row[0] if row else 0


def article_exists(article_id: str) -> bool:
    conn = _get_conn()
    row = conn.execute(
        "SELECT 1 FROM articles WHERE id = ?", (article_id,)
    ).fetchone()
    return row is not None


def close_database() -> None:
    global _conn
    if _conn is not None:
        _conn.close()
        _conn = None


def _row_to_article(row: sqlite3.Row) -> Article:
    return Article(
        title=row["title"],
        url=row["url"],
        source=row["source"],
        summary=row["summary"] or "",
        author=row["author"],
        published_at=row["published_at"],
        fetched_at=row["fetched_at"],
    )

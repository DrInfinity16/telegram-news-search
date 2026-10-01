import logging

from app.config import MAX_ARTICLES_PER_SEARCH, SEARCH_CANDIDATE_LIMIT
from app.evidence import (
    concept_match_display,
    evaluate_concept_coverage,
    filter_meaningful_matches,
    has_minimum_concept_coverage,
    has_minimum_thematic_evidence,
)
from app.models import Article
from app.query_intent import QueryIntent
from app.services.query_expander import expand_query
from app.storage import SearchStats, search_articles_tiered
from app.ranker import score_article_tiered, rank_articles

logger = logging.getLogger(__name__)


def execute_search(raw_query: str, user_keywords: list[str]) -> list[Article]:
    logger.info('Search query: "%s"', raw_query)

    intent = expand_query(raw_query)
    total_concepts = len(intent.all_phrases())

    logger.info(
        "Query interpretation: core=%d, equiv=%d, related=%d, total=%d",
        len(intent.core_concepts),
        len(intent.equivalent_phrases),
        len(intent.related_concepts),
        total_concepts,
    )

    stats = SearchStats()
    candidates = search_articles_tiered(intent, limit=SEARCH_CANDIDATE_LIMIT, stats=stats)

    # Defense in depth: re-apply evidence + coverage filters and clean match explanations.
    qualified: list[Article] = []
    for article in candidates:
        article.matched_keywords = filter_meaningful_matches(article.matched_keywords)
        if not article.matched_keywords:
            continue
        if not has_minimum_thematic_evidence(intent, article.matched_keywords, article):
            continue
        if not has_minimum_concept_coverage(intent, article.matched_keywords, article):
            continue
        cov = evaluate_concept_coverage(intent, article.matched_keywords, article)
        display = concept_match_display(cov)
        if display:
            article.matched_keywords = display
        qualified.append(article)
    stats.after_evidence_filter = len(qualified)
    stats.after_coverage_filter = len(qualified)

    if not qualified:
        logger.info(
            "Search pipeline: raw=%d token_filter=%d evidence=%d coverage=0 returning=0",
            stats.raw_lexical_matches,
            stats.after_token_filter,
            stats.after_evidence_filter,
        )
        return []

    for article in qualified:
        article.relevance_score = score_article_tiered(
            article, intent, article.matched_keywords
        )

    ranked = rank_articles(qualified)
    stats.ranked_count = len(ranked)
    top = ranked[:MAX_ARTICLES_PER_SEARCH]
    stats.returned_count = len(top)

    logger.info(
        "Search pipeline: raw=%d token_filter=%d evidence=%d coverage=%d cap=%d ranked=%d returning=%d",
        stats.raw_lexical_matches,
        stats.after_token_filter,
        stats.after_evidence_filter,
        stats.after_coverage_filter,
        stats.candidate_cap_applied,
        stats.ranked_count,
        stats.returned_count,
    )

    for i, art in enumerate(top):
        logger.debug(
            "  %d. score=%.4f title='%s' matched=%s",
            i + 1,
            art.relevance_score,
            art.title[:60],
            art.matched_keywords,
        )

    return top

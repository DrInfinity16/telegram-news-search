from datetime import datetime, timezone, timedelta

from app.evidence import (
    evaluate_concept_coverage,
    filter_meaningful_matches,
    has_minimum_concept_coverage,
    has_minimum_thematic_evidence,
    is_informative_token,
    is_weak_context,
    phrase_has_informative_token,
)
from app.models import Article
from app.query_intent import QueryIntent
from app.text import normalize_text, token_in_normalized_text


def score_article_tiered(
    article: Article,
    intent: QueryIntent,
    matched_keywords: list[str],
) -> float:
    matched = filter_meaningful_matches(matched_keywords)
    if not matched:
        return 0.0
    if not has_minimum_thematic_evidence(intent, matched, article):
        return 0.0
    if not has_minimum_concept_coverage(intent, matched, article):
        return 0.0

    norm_title = normalize_text(article.title)
    norm_summary = normalize_text(article.summary)
    norm_text = norm_title + " " + norm_summary

    original_query_score = _phrase_in_text(intent.original_query, norm_text, in_title=norm_title)
    core_score = _phrase_list_score(intent.core_concepts, norm_text, norm_title)
    equiv_score = _phrase_list_score(intent.equivalent_phrases, norm_text, norm_title)
    related_score = _phrase_list_score(intent.related_concepts, norm_text, norm_title)

    multi_word_signals = _count_multi_word_signals(intent, matched)
    single_token_support = _single_token_support_count(intent, matched)
    ambiguous_only = _is_ambiguous_only_match(intent, matched)
    weak_only = _is_weak_only_match(intent, matched)
    recency = _compute_recency(article.published_at)
    coverage = _coverage_score_factor(intent, matched, article)

    has_strong_evidence = (
        original_query_score > 0
        or core_score > 0.3
        or equiv_score > 0.3
        or multi_word_signals >= 2
    )

    score = (
        original_query_score * 0.30
        + core_score * 0.25
        + equiv_score * 0.15
        + related_score * 0.08
        + min(multi_word_signals / 3.0, 1.0) * 0.10
        + min(single_token_support / 3.0, 1.0) * 0.04
        + recency * 0.08
    )

    if ambiguous_only or weak_only:
        score *= 0.05
    elif not has_strong_evidence and multi_word_signals == 0:
        score *= 0.3

    score *= coverage
    return round(score, 4)


def _coverage_score_factor(
    intent: QueryIntent,
    matched_keywords: list[str],
    article: Article | None = None,
) -> float:
    """Scale score by concept coverage quality.

    - Full original multi-word phrase: 1.0
    - Single-concept atomic intent: 1.0
    - Multi-concept with both original components: 1.0
    - Multi-concept with one original + related support: 0.85
    - Any other passing case: 0.75
    """
    cov = evaluate_concept_coverage(intent, matched_keywords, article)
    if cov.has_full_intent_phrase:
        return 1.0
    if cov.total_original <= 1:
        return 1.0
    if len(cov.matched_original_labels) >= 2:
        return 1.0
    if len(cov.matched_original_labels) == 1 and cov.matched_related_labels:
        return 0.85
    return 0.75


def _count_multi_word_signals(intent: QueryIntent, matched_keywords: list[str]) -> int:
    count = 0
    for kw in matched_keywords:
        if len(kw.split()) > 1:
            if kw in intent.all_phrases():
                count += 1
    return count


def _phrase_in_text(phrase: str, norm_text: str, in_title: str = "") -> float:
    if not phrase or not phrase_has_informative_token(phrase):
        return 0.0
    nphrase = normalize_text(phrase)
    if not nphrase:
        return 0.0
    words = [normalize_text(w) for w in phrase.split() if w]
    words = [w for w in words if w and is_informative_token(w)]
    if len(words) > 1:
        if nphrase in in_title:
            return 1.0
        if nphrase in norm_text:
            return 0.7
        if words and all(token_in_normalized_text(w, in_title) for w in words):
            return 0.55
        if words and all(
            token_in_normalized_text(w, in_title) or token_in_normalized_text(w, norm_text)
            for w in words
        ):
            return 0.40
        return 0.0
    if not words:
        return 0.0
    if token_in_normalized_text(words[0], in_title):
        return 0.5
    if token_in_normalized_text(words[0], norm_text):
        return 0.3
    return 0.0


def _phrase_list_score(
    phrases: list[str], norm_text: str, norm_title: str
) -> float:
    if not phrases:
        return 0.0
    total = 0.0
    scored = 0
    for phrase in phrases:
        if not phrase_has_informative_token(phrase):
            continue
        total += _phrase_in_text(phrase, norm_text, norm_title)
        scored += 1
    if scored == 0:
        return 0.0
    return min(total / scored, 1.0)


def _single_token_support_count(intent: QueryIntent, matched_keywords: list[str]) -> int:
    count = 0
    for kw in matched_keywords:
        words = kw.split()
        if len(words) == 1 and is_informative_token(kw) and not is_weak_context(kw):
            if normalize_text(kw) in {normalize_text(t) for t in intent.all_phrases()}:
                count += 1
    return count


_GENERIC_SCORE_TOKENS = frozenset({
    "personas", "mexico", "gobierno", "país", "pais", "mundo",
    "apoyo", "política", "politica", "sistema", "mercado",
    "innovación", "innovacion", "tecnología", "tecnologia", "datos",
    "computación", "computacion", "desarrollo", "proceso", "proyectos",
    "proyecto", "caso", "días", "dia", "nuevo", "nueva", "años", "ano",
    "casos", "forma", "parte", "vez", "bien", "mal", "mayor", "menor",
})


def _is_ambiguous_only_match(intent: QueryIntent, matched_keywords: list[str]) -> bool:
    meaningful = filter_meaningful_matches(matched_keywords)
    if not meaningful:
        return False
    for kw in meaningful:
        if len(kw.split()) > 1:
            return False
        if not intent.token_is_ambiguous(kw) and kw not in _GENERIC_SCORE_TOKENS:
            if is_informative_token(kw) and not is_weak_context(kw):
                return False
    all_norms = {normalize_text(p) for p in intent.all_phrases()}
    for kw in meaningful:
        if normalize_text(kw) in all_norms and is_informative_token(kw) and not is_weak_context(kw):
            return False
    return True


def _is_weak_only_match(intent: QueryIntent, matched_keywords: list[str]) -> bool:
    meaningful = filter_meaningful_matches(matched_keywords)
    if not meaningful:
        return True
    for kw in meaningful:
        tokens = kw.split()
        if len(tokens) == 1:
            if is_informative_token(tokens[0]) and not is_weak_context(tokens[0]):
                return False
        else:
            if any(
                is_informative_token(t) and not is_weak_context(t) for t in tokens
            ):
                return False
    return True


def rank_articles(articles: list[Article]) -> list[Article]:
    return sorted(articles, key=lambda a: (-a.relevance_score, a.fetched_at or ""), reverse=False)


def _compute_recency(published_at: str | None) -> float:
    if not published_at:
        return 0.5

    try:
        dt = datetime.fromisoformat(published_at)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        delta = now - dt
    except (ValueError, TypeError):
        return 0.5

    if delta < timedelta(hours=1):
        return 1.0
    if delta < timedelta(hours=6):
        return 0.8
    if delta < timedelta(hours=24):
        return 0.6
    if delta < timedelta(hours=48):
        return 0.4
    return 0.2

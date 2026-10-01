from datetime import datetime, timezone, timedelta

from app.models import Article
from app.query_intent import QueryIntent
from app.ranker import (
    score_article_tiered,
    rank_articles,
    _compute_recency,
    _phrase_in_text,
    _phrase_list_score,
    _is_ambiguous_only_match,
    _count_multi_word_signals,
)


class TestComputeRecency:
    def test_recent_article_scores_high(self):
        now = datetime.now(timezone.utc)
        published = (now - timedelta(minutes=30)).isoformat()
        score = _compute_recency(published)
        assert score == 1.0

    def test_six_hours_ago(self):
        now = datetime.now(timezone.utc)
        published = (now - timedelta(hours=5)).isoformat()
        score = _compute_recency(published)
        assert score == 0.8

    def test_one_day_ago(self):
        now = datetime.now(timezone.utc)
        published = (now - timedelta(hours=20)).isoformat()
        score = _compute_recency(published)
        assert score == 0.6

    def test_two_days_ago(self):
        now = datetime.now(timezone.utc)
        published = (now - timedelta(hours=36)).isoformat()
        score = _compute_recency(published)
        assert score == 0.4

    def test_old_article(self):
        now = datetime.now(timezone.utc)
        published = (now - timedelta(days=5)).isoformat()
        score = _compute_recency(published)
        assert score == 0.2

    def test_none_returns_default(self):
        score = _compute_recency(None)
        assert score == 0.5

    def test_invalid_date_returns_default(self):
        score = _compute_recency("not-a-date")
        assert score == 0.5


class TestRankArticles:
    def test_sorts_by_score_descending(self):
        a1 = Article(title="Low", url="https://a.com", source="S", relevance_score=0.3)
        a2 = Article(title="High", url="https://b.com", source="S", relevance_score=0.9)
        a3 = Article(title="Mid", url="https://c.com", source="S", relevance_score=0.6)

        ranked = rank_articles([a1, a2, a3])
        assert [a.title for a in ranked] == ["High", "Mid", "Low"]

    def test_empty_list(self):
        assert rank_articles([]) == []

    def test_single_article(self):
        a = Article(title="Only", url="https://a.com", source="S")
        ranked = rank_articles([a])
        assert len(ranked) == 1


class TestPhraseInText:
    def test_exact_phrase_in_title(self):
        score = _phrase_in_text("madres buscadoras", "las madres buscadoras buscan", "las madres buscadoras buscan")
        assert score == 1.0

    def test_exact_phrase_in_text_not_title(self):
        score = _phrase_in_text("madres buscadoras", "hoy las madres buscadoras encontraron restos", "")
        assert score == 0.7

    def test_single_token_in_title(self):
        score = _phrase_in_text("terremoto", "hoy terremoto sacude", "hoy terremoto sacude")
        assert score == 0.5

    def test_single_token_in_text(self):
        score = _phrase_in_text("terremoto", "hoy se reporto un terremoto", "")
        assert score == 0.3

    def test_no_match(self):
        score = _phrase_in_text("madres buscadoras", "noticia deportiva", "")
        assert score == 0.0

    def test_empty_phrase(self):
        score = _phrase_in_text("", "some text", "some text")
        assert score == 0.0


class TestPhraseListScore:
    def test_empty_list(self):
        score = _phrase_list_score([], "some text", "some text")
        assert score == 0.0

    def test_all_match(self):
        phrases = ["terremoto", "sismo"]
        text = "un terremoto sacude la region y el sismo se siente"
        score = _phrase_list_score(phrases, text, "")
        assert score > 0.0

    def test_partial_match(self):
        phrases = ["terremoto", "huracan"]
        text = "un terremoto sacude la region"
        score = _phrase_list_score(phrases, text, "")
        assert 0.0 < score < 1.0


class TestGenericOnlyMatch:
    def test_generic_only_returns_true(self):
        intent = QueryIntent(
            original_query="madres buscadoras",
            core_concepts=["madres buscadoras"],
            equivalent_phrases=[],
            related_concepts=[],
        )
        matched = ["personas"]
        assert _is_ambiguous_only_match(intent, matched) is True

    def test_specific_match_returns_false(self):
        intent = QueryIntent(
            original_query="madres buscadoras",
            core_concepts=["madres buscadoras", "personas desaparecidas"],
            equivalent_phrases=[],
            related_concepts=[],
        )
        matched = ["madres buscadoras"]
        assert _is_ambiguous_only_match(intent, matched) is False

    def test_no_generic_tokens_returns_false(self):
        intent = QueryIntent(
            original_query="inteligencia artificial",
            core_concepts=["inteligencia artificial"],
            equivalent_phrases=["machine learning"],
            related_concepts=[],
        )
        matched = ["inteligencia artificial"]
        assert _is_ambiguous_only_match(intent, matched) is False

    def test_empty_matched_returns_false(self):
        intent = QueryIntent(
            original_query="test",
            core_concepts=["test"],
        )
        assert _is_ambiguous_only_match(intent, []) is False

    def test_token_in_intent_returns_false(self):
        intent = QueryIntent(
            original_query="terremoto",
            core_concepts=["terremoto", "sismo"],
            equivalent_phrases=[],
            related_concepts=[],
        )
        matched = ["terremoto"]
        assert _is_ambiguous_only_match(intent, matched) is False


class TestScoreArticleTiered:
    def test_exact_original_query_in_title_highest(self):
        a1 = Article(title="Madres buscadoras encuentran restos", url="https://a.com", source="S")
        a1.matched_keywords = ["madres buscadoras"]
        a2 = Article(title="Personas desaparecidas en Mexico", url="https://b.com", source="S")
        a2.matched_keywords = ["personas desaparecidas"]

        intent = QueryIntent(
            original_query="madres buscadoras",
            core_concepts=["madres buscadoras", "colectivos de busqueda"],
            equivalent_phrases=["madre buscadora"],
            related_concepts=["personas desaparecidas"],
        )

        s1 = score_article_tiered(a1, intent, a1.matched_keywords)
        s2 = score_article_tiered(a2, intent, a2.matched_keywords)

        assert s1 > s2

    def test_generic_only_match_penalized(self):
        intent = QueryIntent(
            original_query="madres buscadoras",
            core_concepts=["madres buscadoras"],
            equivalent_phrases=[],
            related_concepts=[],
        )

        article_generic = Article(title="Personas en peligro", url="https://a.com", source="S")
        article_generic.matched_keywords = ["personas"]

        article_specific = Article(title="Madres buscadoras en accion", url="https://b.com", source="S")
        article_specific.matched_keywords = ["madres buscadoras"]

        s_generic = score_article_tiered(article_generic, intent, article_generic.matched_keywords)
        s_specific = score_article_tiered(article_specific, intent, article_specific.matched_keywords)

        assert s_specific > s_generic

    def test_multi_word_concept_beats_single_token(self):
        intent = QueryIntent(
            original_query="economia mexicana",
            core_concepts=["economia mexicana", "crecimiento economico"],
            equivalent_phrases=["PIB mexicano"],
            related_concepts=[],
        )

        a1 = Article(title="Economia mexicana crece", url="https://a.com", source="S")
        a1.matched_keywords = ["economia mexicana"]

        a2 = Article(title="Mexico crece", url="https://b.com", source="S")
        a2.matched_keywords = ["economia", "mexicana"]

        s1 = score_article_tiered(a1, intent, a1.matched_keywords)
        s2 = score_article_tiered(a2, intent, a2.matched_keywords)

        assert s1 > s2

    def test_score_between_zero_and_one(self):
        intent = QueryIntent(
            original_query="terremoto",
            core_concepts=["terremoto", "sismo"],
            equivalent_phrases=["temblor"],
            related_concepts=["desastre natural"],
        )
        article = Article(title="Terremoto sacude Oaxaca", url="https://a.com", source="S")
        article.matched_keywords = ["terremoto"]
        score = score_article_tiered(article, intent, article.matched_keywords)
        assert 0.0 <= score <= 1.0

    def test_deterministic(self):
        intent = QueryIntent(
            original_query="test",
            core_concepts=["test"],
            equivalent_phrases=[],
            related_concepts=[],
        )
        article = Article(title="Test article", url="https://a.com", source="S")
        article.matched_keywords = ["test"]
        s1 = score_article_tiered(article, intent, article.matched_keywords)
        s2 = score_article_tiered(article, intent, article.matched_keywords)
        assert s1 == s2

    def test_no_match_scores_very_low(self):
        intent = QueryIntent(
            original_query="inteligencia artificial",
            core_concepts=["inteligencia artificial"],
            equivalent_phrases=["machine learning"],
            related_concepts=["IA"],
        )
        article = Article(title="Deportes hoy", url="https://a.com", source="S", summary="futbol")
        article.matched_keywords = []
        score = score_article_tiered(article, intent, article.matched_keywords)
        assert score <= 0.15

    def test_accent_insensitive_scoring(self):
        intent = QueryIntent(
            original_query="economía",
            core_concepts=["economía", "economia mexicana"],
            equivalent_phrases=[],
            related_concepts=[],
        )
        article = Article(title="La economia mexicana crece", url="https://a.com", source="S")
        article.matched_keywords = ["economía"]
        score = score_article_tiered(article, intent, article.matched_keywords)
        assert score > 0.0

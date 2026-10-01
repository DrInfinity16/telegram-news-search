"""Regression tests for stopword / weak-evidence precision filtering.

These tests encode the general failure classes from the precision pass:
stopwords must never create relevance; weak contextual tokens may support
but not create it; minimum thematic evidence is required; match
explanations must stay meaningful.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from app import storage
from app.bot.formatter import format_articles
from app.evidence import (
    filter_meaningful_matches,
    has_minimum_thematic_evidence,
    is_stopword,
    is_weak_context,
)
from app.models import Article
from app.query_intent import QueryIntent
from app.services.pipeline import execute_search


def _iso(days_ago: float = 0.0) -> str:
    dt = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return dt.isoformat()


class TestStopwordFiltering:
    def test_stopword_set_is_conservative(self):
        for token in ["de", "la", "el", "en", "y", "a", "del", "las", "los", "por", "con", "para"]:
            assert is_stopword(token), token
        for token in ["Mexico", "gobierno", "personas", "familias", "nacional", "mercado"]:
            assert not is_stopword(token), token

    def test_retrieval_tokens_exclude_stopwords(self):
        intent = QueryIntent(
            original_query="guerra Rusia Ucrania",
            core_concepts=["guerra en Rusia", "guerra Ucrania"],
            related_concepts=["en el frente", "contra Rusia"],
        )
        tokens = {t.lower() for t in intent.retrieval_tokens()}
        assert "en" not in tokens
        assert "contra" not in tokens
        assert "guerra" in tokens
        assert "rusia" in tokens
        assert "ucrania" in tokens

    def test_filter_meaningful_matches_removes_stopwords(self):
        raw = ["en", "a", "de", "contra", "el", "del", "la", "guerra Rusia Ucrania"]
        meaningful = filter_meaningful_matches(raw)
        assert meaningful == ["guerra Rusia Ucrania"]

    def test_stopword_only_article_rejected(self):
        intent = QueryIntent(
            original_query="guerra Rusia Ucrania",
            core_concepts=["guerra Rusia Ucrania", "guerra en Rusia"],
            equivalent_phrases=["en el frente", "contra Rusia"],
            related_concepts=["la guerra", "el conflicto"],
        )
        article = Article(
            title="Reporte de la mañana en la ciudad",
            url="https://example.com/stop-1",
            source="Test",
            summary="El y a de por con para del las los un una",
        )
        storage.insert_article(article)
        results = storage.search_articles_tiered(intent, limit=10)
        titles = [r.title for r in results]
        assert "Reporte de la mañana en la ciudad" not in titles
        assert not has_minimum_thematic_evidence(
            intent, ["en", "a", "de", "la", "el"], article
        )


class TestWeakContextTerms:
    def test_weak_terms_are_not_stopwords(self):
        for token in ["Mexico", "nacional", "personas", "gobierno", "familias"]:
            assert not is_stopword(token)
            assert is_weak_context(token)

    def test_weak_geographic_only_rejected(self):
        intent = QueryIntent(
            original_query="economia mexicana",
            core_concepts=["economia mexicana", "inflacion", "Banxico"],
            equivalent_phrases=["peso mexicano"],
            related_concepts=["mercado"],
        )
        weak_article = Article(
            title="Mexico y nacional en foco",
            url="https://example.com/weak-geo",
            source="Test",
            summary="Mexico nacional gobierno personas",
        )
        strong_article = Article(
            title="Inflacion y peso mexicano presionan a Banxico",
            url="https://example.com/strong-econ",
            source="Test",
            summary="Banxico eleva tasas frente a inflacion y tipo de cambio",
        )
        storage.insert_article(weak_article)
        storage.insert_article(strong_article)

        results = storage.search_articles_tiered(intent, limit=10)
        titles = [r.title for r in results]
        assert "Mexico y nacional en foco" not in titles
        assert any("Inflacion" in t for t in titles)

        assert not has_minimum_thematic_evidence(
            intent, ["Mexico", "nacional"], weak_article
        )
        assert has_minimum_thematic_evidence(
            intent, ["inflacion", "Mexico"], strong_article
        )

    def test_weak_term_can_support_but_not_create(self):
        intent = QueryIntent(
            original_query="desapariciones Mexico",
            core_concepts=["desapariciones Mexico", "personas desaparecidas"],
            equivalent_phrases=["desaparicion forzada"],
            related_concepts=["familias"],
        )
        weak_only = Article(
            title="Familias y personas en Mexico",
            url="https://example.com/desap-weak",
            source="Test",
            summary="personas familias Mexico",
        )
        strong = Article(
            title="Personas desaparecidas y buscadoras en Mexico",
            url="https://example.com/desap-strong",
            source="Test",
            summary="personas desaparecidas madres buscadoras",
        )
        storage.insert_article(weak_only)
        storage.insert_article(strong)

        results = storage.search_articles_tiered(intent, limit=10)
        titles = [r.title for r in results]
        assert "Familias y personas en Mexico" not in titles
        assert any("desaparecidas" in t.lower() for t in titles)


class TestMinimumEvidence:
    def test_stopword_plus_weak_not_enough(self):
        intent = QueryIntent(
            original_query="economia mexicana",
            core_concepts=["economia mexicana"],
            equivalent_phrases=[],
            related_concepts=["nacional", "mercado"],
        )
        article = Article(
            title="De en la del nacional",
            url="https://example.com/econ-noise",
            source="Test",
            summary="en a de la el y del nacional mercado",
        )
        assert not has_minimum_thematic_evidence(
            intent, ["en", "a", "de", "nacional", "mercado"], article
        )

    def test_original_phrase_is_strong(self):
        intent = QueryIntent(
            original_query="economia mexicana",
            core_concepts=["economia mexicana"],
        )
        article = Article(
            title="Economia mexicana crece",
            url="https://example.com/econ-yes",
            source="S",
            summary="ok",
        )
        assert has_minimum_thematic_evidence(intent, ["economia mexicana"], article)

    def test_informative_original_token_is_strong(self):
        intent = QueryIntent(
            original_query="inflacion Mexico",
            core_concepts=["inflacion Mexico"],
        )
        article = Article(
            title="Inflacion sube en el pais",
            url="https://example.com/inf",
            source="S",
            summary="ok",
        )
        assert has_minimum_thematic_evidence(intent, ["inflacion"], article)

    def test_generic_core_fragment_alone_is_not_enough(self):
        intent = QueryIntent(
            original_query="madres buscadoras",
            core_concepts=["madres buscadoras", "colectivos de busqueda"],
            equivalent_phrases=["madre buscadora"],
            related_concepts=["personas desaparecidas"],
        )
        article = Article(
            title="Ficha de busqueda policial",
            url="https://example.com/busqueda-only",
            source="S",
            summary="se difunde ficha de busqueda",
        )
        assert not has_minimum_thematic_evidence(intent, ["busqueda"], article)
        assert has_minimum_thematic_evidence(
            intent, ["madres buscadoras", "busqueda"], article
        )

    def test_short_acronym_core_token_alone_is_enough(self):
        intent = QueryIntent(
            original_query="inteligencia artificial",
            core_concepts=["inteligencia artificial", "IA"],
            equivalent_phrases=["machine learning"],
        )
        article = Article(
            title="IA y fuga de datos",
            url="https://example.com/ia-alone",
            source="S",
            summary="IA en produccion",
        )
        assert has_minimum_thematic_evidence(intent, ["IA"], article)

    def test_two_informative_related_tokens_qualify(self):
        intent = QueryIntent(
            original_query="terremoto",
            core_concepts=["terremoto"],
            equivalent_phrases=["sismo"],
            related_concepts=["rescate", "victimas", "escombros"],
        )
        article = Article(
            title="Rescate tras sismo",
            url="https://example.com/rescate",
            source="S",
            summary="equipos de rescate buscan victimas en escombros",
        )
        assert has_minimum_thematic_evidence(
            intent, ["rescate", "victimas"], article
        )


class TestMadresBuscadorasPrecision:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        db_path = str(tmp_path / "precision_madres.db")
        storage.close_database()
        monkeypatch.setenv("DATABASE_PATH", db_path)
        storage.initialize_database()
        yield
        storage.close_database()
        monkeypatch.delenv("DATABASE_PATH", raising=False)

    @patch("app.services.pipeline.expand_query")
    def test_strong_disappearance_article_ranks_above_unrelated_women_article(
        self, mock_expand
    ):
        strong = Article(
            title="Madres buscadoras encuentran restos",
            url="https://example.com/madres-1",
            source="Test",
            summary="Mujeres buscadoras y personas desaparecidas en fosas",
            published_at=_iso(1),
        )
        unrelated = Article(
            title="Violencia de genero y familias afectadas",
            url="https://example.com/madres-2",
            source="Test",
            summary="mujeres familias violencia general apoyo",
            published_at=_iso(0.1),
        )
        storage.insert_article(strong)
        storage.insert_article(unrelated)

        mock_expand.return_value = QueryIntent(
            original_query="madres buscadoras",
            core_concepts=["madres buscadoras", "colectivos de busqueda"],
            equivalent_phrases=["madre buscadora"],
            related_concepts=["personas desaparecidas", "desaparicion forzada"],
        )
        results = execute_search("madres buscadoras", ["madres", "buscadoras"])
        assert results, "expected at least the strong article"
        assert "Madres buscadoras" in results[0].title
        titles = [r.title for r in results]
        assert "Violencia de genero y familias afectadas" not in titles or titles.index(
            "Madres buscadoras encuentran restos"
        ) < titles.index("Violencia de genero y familias afectadas")

        assert all(
            not any(is_stopword(k) for k in r.matched_keywords) for r in results
        )


class TestAIPrecision:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        db_path = str(tmp_path / "precision_ai.db")
        storage.close_database()
        monkeypatch.setenv("DATABASE_PATH", db_path)
        storage.initialize_database()
        yield
        storage.close_database()
        monkeypatch.delenv("DATABASE_PATH", raising=False)

    @patch("app.services.pipeline.expand_query")
    def test_ai_true_positive_retained_and_false_positive_rejected(self, mock_expand):
        true_pos = Article(
            title="Inteligencia artificial transforma la industria",
            url="https://example.com/ai-true",
            source="Test",
            summary="Empresas adoptan IA y modelos de lenguaje",
            published_at=_iso(1),
        )
        false_pos = Article(
            title="Gobierno y sistema de datos generales",
            url="https://example.com/ai-false",
            source="Test",
            summary="personas gobierno sistema datos apoyo general",
            published_at=_iso(0.2),
        )
        storage.insert_article(true_pos)
        storage.insert_article(false_pos)

        mock_expand.return_value = QueryIntent(
            original_query="inteligencia artificial",
            core_concepts=["inteligencia artificial", "IA"],
            equivalent_phrases=["machine learning", "modelos de lenguaje"],
            related_concepts=["gobierno", "personas", "sistema", "datos", "apoyo"],
        )
        results = execute_search(
            "inteligencia artificial", ["inteligencia artificial"]
        )
        titles = [r.title for r in results]
        assert any("inteligencia artificial" in t.lower() for t in titles)
        assert "Gobierno y sistema de datos generales" not in titles
        top = results[0]
        assert any(
            is_stopword(k) is False for k in top.matched_keywords
        ), top.matched_keywords
        assert "en" not in {k.lower() for k in top.matched_keywords}
        assert "de" not in {k.lower() for k in top.matched_keywords}

    def test_short_acronym_boundary(self):
        from app.text import token_in_normalized_text, normalize_text

        assert token_in_normalized_text("IA", normalize_text("usa IA en produccion"))
        assert not token_in_normalized_text("IA", normalize_text("paises aislados"))

        intent = QueryIntent(
            original_query="IA",
            core_concepts=["IA", "modelos de lenguaje"],
        )
        storage.insert_article(
            Article(
                title="Usa IA en produccion",
                url="https://example.com/ia-yes",
                source="Test",
                summary="equipos de IA",
            )
        )
        storage.insert_article(
            Article(
                title="Paises aislados",
                url="https://example.com/ia-no",
                source="Test",
                summary="noticias generales",
            )
        )
        results = storage.search_articles_tiered(intent, limit=10)
        titles = [r.title for r in results]
        assert "Usa IA en produccion" in titles
        assert "Paises aislados" not in titles


class TestMatchExplanations:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        db_path = str(tmp_path / "precision_match.db")
        storage.close_database()
        monkeypatch.setenv("DATABASE_PATH", db_path)
        storage.initialize_database()
        yield
        storage.close_database()
        monkeypatch.delenv("DATABASE_PATH", raising=False)

    @patch("app.services.pipeline.expand_query")
    def test_stopwords_never_appear_in_match_output(self, mock_expand):
        storage.insert_article(
            Article(
                title="Guerra en Rusia y Ucrania",
                url="https://example.com/guerra-1",
                source="Test",
                summary="Cobertura del conflicto entre Rusia y Ucrania",
                published_at=_iso(1),
            )
        )
        storage.insert_article(
            Article(
                title="El de la y a noticia",
                url="https://example.com/noise",
                source="Test",
                summary="en con por para del las los",
                published_at=_iso(0.1),
            )
        )
        mock_expand.return_value = QueryIntent(
            original_query="guerra Rusia Ucrania",
            core_concepts=["guerra Rusia Ucrania", "guerra en Rusia"],
            equivalent_phrases=["en el frente", "contra Rusia"],
            related_concepts=["la guerra", "el conflicto"],
        )
        results = execute_search(
            "guerra Rusia Ucrania", ["guerra", "Rusia", "Ucrania"]
        )
        assert results
        stopword_hits = set()
        for r in results:
            for kw in r.matched_keywords:
                for token in kw.split():
                    if is_stopword(token):
                        stopword_hits.add(token)
        assert not stopword_hits
        message = format_articles(results)
        assert "Match:" in message
        for bad in ["de", "en", "la", "el", "y", "a", "del"]:
            assert f"Match: {bad}" not in message
            assert f", {bad}," not in message.split("Match:")[1][:200]


class TestCandidateCapAndRanking:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        db_path = str(tmp_path / "precision_cap.db")
        storage.close_database()
        monkeypatch.setenv("DATABASE_PATH", db_path)
        storage.initialize_database()
        yield
        storage.close_database()
        monkeypatch.delenv("DATABASE_PATH", raising=False)

    @patch("app.services.pipeline.expand_query")
    def test_strong_matches_survive_many_weak_candidates(self, mock_expand):
        for i in range(40):
            storage.insert_article(
                Article(
                    title=f"Mexico nacional general noticia {i}",
                    url=f"https://example.com/weak/{i}",
                    source="Test",
                    summary="Mexico nacional gobierno personas apoyo general",
                    published_at=_iso(0.01 * i),
                )
            )
        storage.insert_article(
            Article(
                title="Inflacion dispara precio del peso",
                url="https://example.com/strong/1",
                source="Test",
                summary="Banxico y inflacion golpean al peso mexicano",
                published_at=_iso(2),
            )
        )
        storage.insert_article(
            Article(
                title="Economia mexicana bajo presion",
                url="https://example.com/strong/2",
                source="Test",
                summary="economia mexicana y mercado financiero",
                published_at=_iso(2.5),
            )
        )

        mock_expand.return_value = QueryIntent(
            original_query="economia mexicana",
            core_concepts=["economia mexicana", "inflacion", "Banxico"],
            equivalent_phrases=["peso mexicano"],
            related_concepts=["mercado"],
        )
        results = execute_search("economia mexicana", ["economia", "mexicana"])
        assert results
        titles = [r.title for r in results]
        assert any("Economia mexicana" in t for t in titles)
        assert any("Inflacion" in t or "inflacion" in t.lower() for t in titles)
        # Weak geography-only articles must not dominate the head.
        head = titles[:2]
        assert any("Economia" in t or "Inflacion" in t or "inflacion" in t.lower() for t in head)
        assert all(
            not is_weak_context(r.matched_keywords[0]) if len(r.matched_keywords) == 1 else True
            for r in results
            if r.matched_keywords
        )

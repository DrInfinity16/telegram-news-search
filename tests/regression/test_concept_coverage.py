"""Regression tests for concept coverage / co-occurrence evidence.

These tests encode the precision failures of pass 3: multi-concept queries
must require multiple meaningful components, not one vaguely related word.
No query-specific hardcoding; coverage is derived from QueryIntent only.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from app import storage
from app.bot.formatter import format_articles
from app.evidence import (
    concept_match_display,
    derive_concept_groups,
    evaluate_concept_coverage,
    has_minimum_concept_coverage,
)
from app.models import Article
from app.query_intent import QueryIntent
from app.ranker import score_article_tiered
from app.services.pipeline import execute_search


def _iso(days_ago: float = 0.0) -> str:
    dt = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return dt.isoformat()


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "coverage.db")
    storage.close_database()
    monkeypatch.setenv("DATABASE_PATH", db_path)
    storage.initialize_database()
    yield
    storage.close_database()
    monkeypatch.delenv("DATABASE_PATH", raising=False)


class TestMultiConceptConflictRejection:
    """A: Iraq and weather articles must not satisfy guerra Rusia Ucrania."""

    def _guerra_intent(self) -> QueryIntent:
        return QueryIntent(
            original_query="guerra Rusia Ucrania",
            core_concepts=["guerra Rusia Ucrania", "guerra en Rusia"],
            equivalent_phrases=["conflicto Rusia Ucrania", "invasión de Ucrania"],
            related_concepts=["frente oriental", "zona de combate"],
        )

    def test_iraq_invasion_rejected(self):
        intent = self._guerra_intent()
        article = Article(
            title="Invasión militar a Irak",
            url="https://example.com/iraq",
            source="Test",
            summary="Operación militar en el oriente medio",
        )
        matched = ["invasión", "militar", "zona"]
        assert not has_minimum_concept_coverage(intent, matched, article)
        cov = evaluate_concept_coverage(intent, matched, article)
        assert cov.passes is False
        assert score_article_tiered(article, intent, matched) == 0.0

    def test_weather_frente_rejected(self):
        intent = self._guerra_intent()
        article = Article(
            title="Frente oriental trae lluvias",
            url="https://example.com/weather",
            source="Test",
            summary="Zona de lluvias en el oriente",
        )
        matched = ["frente", "oriental", "zona"]
        assert not has_minimum_concept_coverage(intent, matched, article)
        cov = evaluate_concept_coverage(intent, matched, article)
        assert cov.passes is False

    def test_both_war_components_pass(self):
        intent = self._guerra_intent()
        article = Article(
            title="Guerra entre Rusia y Ucrania",
            url="https://example.com/war",
            source="Test",
            summary="Cobertura del conflicto entre Rusia y Ucrania",
        )
        matched = ["Rusia", "Ucrania", "guerra"]
        assert has_minimum_concept_coverage(intent, matched, article)
        cov = evaluate_concept_coverage(intent, matched, article)
        assert cov.passes is True
        assert len(cov.matched_original_labels) >= 2


class TestEconomyGeographyCoverage:
    """B: economy + geography needs both; crime/debt Mexico alone rejected."""

    def _econ_intent(self) -> QueryIntent:
        return QueryIntent(
            original_query="economía mexicana",
            core_concepts=["economía mexicana", "inflación", "Banxico"],
            equivalent_phrases=["peso mexicano"],
            related_concepts=["mercado financiero"],
        )

    def test_crime_debt_mexico_rejected(self):
        intent = self._econ_intent()
        article = Article(
            title="Crimen y deuda en México",
            url="https://example.com/crime",
            source="Test",
            summary="México peso deuda criminalidad",
        )
        matched = ["México", "peso", "deuda"]
        assert not has_minimum_concept_coverage(intent, matched, article)

    def test_economy_topic_plus_geo_passes(self):
        intent = self._econ_intent()
        article = Article(
            title="Economía mexicana bajo presión",
            url="https://example.com/econ",
            source="Test",
            summary="Economía mexicana y mercado financiero",
        )
        matched = ["economía mexicana", "México"]
        assert has_minimum_concept_coverage(intent, matched, article)
        cov = evaluate_concept_coverage(intent, matched, article)
        assert cov.passes is True
        assert any("economía" in l.lower() or "economia" in l.lower() for l in cov.matched_labels)


class TestMadresBuscadorasTailRejection:
    """C: weak-tail single-fragment articles must not pass."""

    def _intent(self) -> QueryIntent:
        return QueryIntent(
            original_query="madres buscadoras",
            core_concepts=["madres buscadoras", "colectivos de búsqueda"],
            equivalent_phrases=["madres desaparecidas"],
            related_concepts=["búsqueda de desaparecidos"],
        )

    def test_busqueda_only_rejected(self):
        intent = self._intent()
        article = Article(
            title="Comisión revisa búsqueda",
            url="https://example.com/tail1",
            source="Test",
            summary="El movimiento busca respuestas",
        )
        matched = ["búsqueda", "comisión", "verdad"]
        assert not has_minimum_concept_coverage(intent, matched, article)

    def test_madres_plus_busqueda_passes(self):
        intent = self._intent()
        article = Article(
            title="Madres buscadoras exigen respuestas",
            url="https://example.com/tail2",
            source="Test",
            summary="Colectivos de búsqueda de desaparecidos",
        )
        matched = ["madres", "búsqueda"]
        assert has_minimum_concept_coverage(intent, matched, article)
        cov = evaluate_concept_coverage(intent, matched, article)
        assert cov.passes is True


class TestSingleConceptPreservation:
    """D: single-concept queries still need only one component."""

    def test_terremoto_single_token_passes(self):
        intent = QueryIntent(
            original_query="terremoto",
            core_concepts=["terremoto", "sismo"],
            equivalent_phrases=["temblor"],
            related_concepts=["daños materiales"],
        )
        article = Article(
            title="Terremoto sacude región",
            url="https://example.com/quake",
            source="Test",
            summary="Sismo de magnitud 6.2",
        )
        matched = ["terremoto"]
        assert has_minimum_concept_coverage(intent, matched, article)
        cov = evaluate_concept_coverage(intent, matched, article)
        assert cov.passes is True
        assert cov.total_original == 1
        assert score_article_tiered(article, intent, matched) > 0.0


class TestCompoundPhraseIntegrity:
    """E: fixed compounds like inteligencia artificial stay atomic."""

    def test_ia_compound_decomposes_but_ia_covers_both(self):
        intent = QueryIntent(
            original_query="inteligencia artificial",
            core_concepts=["inteligencia artificial", "IA"],
            equivalent_phrases=["machine learning"],
            related_concepts=["tecnología"],
        )
        groups = derive_concept_groups(intent)
        original_groups = [g for g in groups if g.is_original_component]
        # Multi-word original decomposes into components; IA is a single-token alt.
        assert len(original_groups) == 2
        labels = {g.label.lower() for g in original_groups}
        assert labels == {"inteligencia", "artificial"}
        for g in original_groups:
            assert "ia" in g.match_tokens
        article = Article(
            title="Usa IA en produccion",
            url="https://example.com/ia-atomic",
            source="Test",
            summary="equipos de IA",
        )
        assert has_minimum_concept_coverage(intent, ["IA"], article)
        cov = evaluate_concept_coverage(intent, ["IA"], article)
        assert cov.passes is True
        assert len(cov.matched_original_labels) >= 2

    def test_only_generic_tech_rejected(self):
        intent = QueryIntent(
            original_query="inteligencia artificial",
            core_concepts=["inteligencia artificial", "IA"],
            equivalent_phrases=["machine learning"],
            related_concepts=["tecnología"],
        )
        article = Article(
            title="Noticia de tecnología",
            url="https://example.com/tech",
            source="Test",
            summary="Avances tecnológicos y datos",
        )
        matched = ["tecnología", "datos"]
        assert not has_minimum_concept_coverage(intent, matched, article)

    def test_ia_token_passes(self):
        intent = QueryIntent(
            original_query="inteligencia artificial",
            core_concepts=["inteligencia artificial", "IA"],
            equivalent_phrases=["machine learning"],
            related_concepts=["tecnología"],
        )
        article = Article(
            title="Usa IA en producción",
            url="https://example.com/ia",
            source="Test",
            summary="Equipos de inteligencia artificial",
        )
        matched = ["IA", "inteligencia artificial"]
        assert has_minimum_concept_coverage(intent, matched, article)


class TestEntityTopicCoverage:
    """F: entity + topic (inflación México) needs both sides."""

    def test_geo_only_rejected(self):
        intent = QueryIntent(
            original_query="inflación México",
            core_concepts=["inflación", "México", "inflación mexicana"],
            equivalent_phrases=["precio al consumidor"],
            related_concepts=["Banxico"],
        )
        article = Article(
            title="México y su gente",
            url="https://example.com/geo",
            source="Test",
            summary="México nacional personas",
        )
        matched = ["México"]
        assert not has_minimum_concept_coverage(intent, matched, article)

    def test_topic_plus_entity_passes(self):
        intent = QueryIntent(
            original_query="inflación México",
            core_concepts=["inflación", "México", "inflación mexicana"],
            equivalent_phrases=["precio al consumidor"],
            related_concepts=["Banxico"],
        )
        article = Article(
            title="Inflación golpea a México",
            url="https://example.com/inflacion",
            source="Test",
            summary="La inflación en México supera expectativas",
        )
        matched = ["inflación", "México"]
        assert has_minimum_concept_coverage(intent, matched, article)
        cov = evaluate_concept_coverage(intent, matched, article)
        assert cov.passes is True


class TestExactPhraseOverride:
    """G: full original multi-word phrase is a full-intent override."""

    def test_full_phrase_passes_even_with_sparse_tokens(self):
        intent = QueryIntent(
            original_query="guerra Rusia Ucrania",
            core_concepts=["guerra Rusia Ucrania"],
            equivalent_phrases=["conflicto Rusia Ucrania"],
            related_concepts=["frente oriental"],
        )
        article = Article(
            title="Situación de guerra Rusia Ucrania",
            url="https://example.com/full",
            source="Test",
            summary="Cobertura de guerra Rusia Ucrania",
        )
        matched = ["guerra Rusia Ucrania"]
        cov = evaluate_concept_coverage(intent, matched, article)
        assert cov.has_full_intent_phrase is True
        assert cov.passes is True
        display = concept_match_display(cov)
        assert display == ["guerra Rusia Ucrania"]
        assert score_article_tiered(article, intent, matched) > 0.5


class TestRelatedOnlyRejection:
    """H: related-only support cannot pass multi-concept queries."""

    def test_related_only_rejected(self):
        intent = QueryIntent(
            original_query="madres buscadoras",
            core_concepts=["madres buscadoras"],
            equivalent_phrases=["colectivos de búsqueda"],
            related_concepts=["desaparecidos", "búsqueda de familiares"],
        )
        article = Article(
            title="Familias buscan desaparecidos",
            url="https://example.com/related",
            source="Test",
            summary="Búsqueda de familiares y personas",
        )
        matched = ["desaparecidos", "familias"]
        assert not has_minimum_concept_coverage(intent, matched, article)
        cov = evaluate_concept_coverage(intent, matched, article)
        assert cov.passes is False
        assert score_article_tiered(article, intent, matched) == 0.0

    def test_original_plus_related_support_passes(self):
        intent = QueryIntent(
            original_query="madres buscadoras",
            core_concepts=["madres buscadoras"],
            equivalent_phrases=["colectivos de búsqueda"],
            related_concepts=["desaparecidos", "búsqueda de familiares"],
        )
        article = Article(
            title="Madres buscadoras de desaparecidos",
            url="https://example.com/orig-rel",
            source="Test",
            summary="Colectivos de búsqueda y familiares",
        )
        matched = ["madres", "desaparecidos"]
        assert has_minimum_concept_coverage(intent, matched, article)
        cov = evaluate_concept_coverage(intent, matched, article)
        assert cov.passes is True
        assert cov.matched_original_labels
        assert cov.matched_related_labels
        assert "madres" in [l.lower() for l in cov.matched_original_labels]
        assert any("desaparecido" in l.lower() for l in cov.matched_related_labels)


class TestPipelineCoverageStage:
    """End-to-end: coverage filter + concept-level Match display."""

    @patch("app.services.pipeline.expand_query")
    def test_pipeline_rejects_partial_and_shows_concepts(self, mock_expand):
        storage.insert_article(
            Article(
                title="Invasión militar a Irak",
                url="https://example.com/iraq-full",
                source="Test",
                summary="Operación militar en el oriente medio",
                published_at=_iso(0.2),
            )
        )
        storage.insert_article(
            Article(
                title="Guerra entre Rusia y Ucrania",
                url="https://example.com/war-full",
                source="Test",
                summary="Cobertura del conflicto entre Rusia y Ucrania",
                published_at=_iso(0.5),
            )
        )
        mock_expand.return_value = QueryIntent(
            original_query="guerra Rusia Ucrania",
            core_concepts=["guerra Rusia Ucrania", "guerra en Rusia"],
            equivalent_phrases=["conflicto Rusia Ucrania", "invasión de Ucrania"],
            related_concepts=["frente oriental", "zona de combate"],
        )
        results = execute_search("guerra Rusia Ucrania", ["guerra", "Rusia", "Ucrania"])
        titles = [r.title for r in results]
        assert titles
        assert any("Rusia" in t and "Ucrania" in t for t in titles)
        assert not any("Irak" in t for t in titles)
        for r in results:
            assert r.matched_keywords
            joined = " ".join(r.matched_keywords).lower()
            assert "rusia" in joined or "ucrania" in joined or "guerra" in joined
        message = format_articles(results)
        assert "Match:" in message

    @patch("app.services.pipeline.expand_query")
    def test_search_stats_include_coverage_stage(self, mock_expand):
        from app.storage import SearchStats, search_articles_tiered

        storage.insert_article(
            Article(
                title="Inflación en México",
                url="https://example.com/infl-mx",
                source="Test",
                summary="Inflación mexicana y peso",
            )
        )
        storage.insert_article(
            Article(
                title="México y su gente",
                url="https://example.com/geo-only",
                source="Test",
                summary="México nacional personas",
            )
        )
        mock_expand.return_value = QueryIntent(
            original_query="inflación México",
            core_concepts=["inflación", "México"],
            equivalent_phrases=["inflación mexicana"],
            related_concepts=["Banxico"],
        )
        intent = mock_expand.return_value
        stats = SearchStats()
        results = search_articles_tiered(intent, limit=10, stats=stats)
        assert stats.after_coverage_filter == len(results)
        assert stats.after_coverage_filter <= stats.after_evidence_filter
        assert any("Inflación" in r.title for r in results)
        assert not any("México y su gente" in r.title for r in results)

"""Regression tests for verified retrieval/persistence defects.

These tests target production storage behavior with realistic fixtures.
Gemini is not required; intents are constructed deterministically.
"""

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app import storage
from app.models import Article
from app.query_intent import QueryIntent
from app.ranker import score_article_tiered, rank_articles
from app.services.pipeline import execute_search
from app.text import normalize_text
from unittest.mock import patch


def _iso(days_ago: float = 0.0, hours_ago: float = 0.0) -> str:
    dt = datetime.now(timezone.utc) - timedelta(days=days_ago, hours=hours_ago)
    return dt.isoformat()


class TestAccentRetrieval:
    def test_unaccented_query_matches_accented_title_without_fallback(self):
        storage.insert_article(
            Article(
                title="La economía mexicana muestra señales de crecimiento",
                url="https://example.com/economia-1",
                source="Test",
                summary="El PIB mejora",
            )
        )
        # Many newer unrelated articles must not hide the accented match
        for i in range(60):
            storage.insert_article(
                Article(
                    title=f"Noticia general numero {i}",
                    url=f"https://example.com/news/{i}",
                    source="Test",
                    summary="contenido general",
                    published_at=_iso(days_ago=0.01 * i),
                )
            )

        intent = QueryIntent(
            original_query="economia mexicana",
            core_concepts=["economia mexicana"],
        )
        results = storage.search_articles_tiered(intent, limit=50)
        titles = [r.title for r in results]
        assert any("economía mexicana" in t for t in titles), titles

    def test_accented_query_matches_unaccented_title(self):
        storage.insert_article(
            Article(
                title="Politica monetaria del Banxico",
                url="https://example.com/politica-1",
                source="Test",
                summary="tasas de interes",
            )
        )
        intent = QueryIntent(
            original_query="política monetaria",
            core_concepts=["política monetaria"],
        )
        results = storage.search_articles_tiered(intent, limit=20)
        assert any("Politica monetaria" in r.title for r in results)

    def test_sql_path_uses_normalized_columns(self):
        storage.insert_article(
            Article(
                title="La economía mexicana crece",
                url="https://example.com/econ-2",
                source="Test",
                summary="ok",
            )
        )
        conn = storage._get_conn()
        row = conn.execute(
            "SELECT title_norm, summary_norm FROM articles WHERE url LIKE '%econ-2%'"
        ).fetchone()
        assert row is not None
        assert "economia mexicana" in row["title_norm"]
        # padding present for boundary-safe short-token matching
        assert row["title_norm"].startswith(" ")
        assert row["title_norm"].endswith(" ")


class TestMultiWordDecomposition:
    def test_non_contiguous_terms_enter_candidates(self):
        storage.insert_article(
            Article(
                title="Rusia intensifica ataques mientras Ucrania refuerza sus defensas",
                url="https://example.com/ru-ua-1",
                source="Test",
                summary="El frente del este se intensifica",
            )
        )
        # Irrelevant newer articles using one generic token only
        for i in range(20):
            storage.insert_article(
                Article(
                    title=f"Gobierno anuncia programa {i}",
                    url=f"https://example.com/gov/{i}",
                    source="Test",
                    summary="politica nacional",
                    published_at=_iso(days_ago=0.01 * i),
                )
            )

        intent = QueryIntent(
            original_query="guerra Rusia Ucrania",
            core_concepts=["guerra Rusia Ucrania"],
        )
        results = storage.search_articles_tiered(intent, limit=50)
        titles = [r.title for r in results]
        assert any("Rusia" in t and "Ucrania" in t for t in titles), titles

    def test_retrieval_tokens_include_multiword_parts(self):
        intent = QueryIntent(
            original_query="guerra Rusia Ucrania",
            core_concepts=["guerra Rusia Ucrania"],
        )
        tokens = {normalize_text(t) for t in intent.retrieval_tokens()}
        assert "guerra" in tokens
        assert "rusia" in tokens
        assert "ucrania" in tokens

    def test_retrieval_tokens_include_short_tokens_for_boundary_matching(self):
        intent = QueryIntent(original_query="IA", core_concepts=["IA"])
        tokens = {normalize_text(t) for t in intent.retrieval_tokens()}
        assert "ia" in tokens

    def test_multiword_not_equal_to_or_without_ranking_signals(self):
        storage.insert_article(
            Article(
                title="Rusia intensifica ataques mientras Ucrania refuerza sus defensas",
                url="https://example.com/ru-ua-2",
                source="Test",
                summary="conflicto",
            )
        )
        storage.insert_article(
            Article(
                title="Guerra comercial entre socios",
                url="https://example.com/guerra-solo",
                source="Test",
                summary="comercio",
            )
        )
        intent = QueryIntent(
            original_query="guerra Rusia Ucrania",
            core_concepts=["guerra Rusia Ucrania"],
        )
        candidates = storage.search_articles_tiered(intent, limit=50)
        for article in candidates:
            article.relevance_score = score_article_tiered(
                article, intent, article.matched_keywords
            )
        ranked = rank_articles(candidates)
        assert ranked
        assert "Rusia" in ranked[0].title


class TestCombinationRelevance:
    def test_multi_term_article_ranks_above_single_generic(self):
        storage.insert_article(
            Article(
                title="Banxico mantiene política monetaria y controla inflación",
                url="https://example.com/banxico",
                source="Test",
                summary="La institución revisa tasas y tipo de cambio",
            )
        )
        storage.insert_article(
            Article(
                title="Sistema educativo enfrenta retos",
                url="https://example.com/edu",
                source="Test",
                summary="mexico y politica en un solo contexto generico",
            )
        )
        intent = QueryIntent(
            original_query="economía mexicana",
            core_concepts=["economía mexicana", "política monetaria"],
            equivalent_phrases=["Banxico"],
            related_concepts=["inflación"],
        )
        candidates = storage.search_articles_tiered(intent, limit=50)
        for article in candidates:
            article.relevance_score = score_article_tiered(
                article, intent, article.matched_keywords
            )
        ranked = rank_articles(candidates)
        assert ranked
        assert "Banxico" in ranked[0].title


class TestCandidateAge:
    def test_old_relevant_article_survives_newer_noise(self):
        older = Article(
            title="Madres buscadoras localizan fosa en Tamaulipas",
            url="https://example.com/madres-old",
            source="Test",
            summary="colectivos de busqueda de desaparecidos",
            published_at=_iso(days_ago=40),
        )
        storage.insert_article(older)

        for i in range(80):
            storage.insert_article(
                Article(
                    title=f"Deportes y espectaculos {i}",
                    url=f"https://example.com/sports/{i}",
                    source="Test",
                    summary="contenido sin relacion",
                    published_at=_iso(days_ago=0.001 * i),
                )
            )

        intent = QueryIntent(
            original_query="madres buscadoras",
            core_concepts=["madres buscadoras", "colectivos de busqueda"],
        )
        candidates = storage.search_articles_tiered(intent, limit=50)
        titles = [r.title for r in candidates]
        assert any("madres buscadoras" in t.lower() for t in titles), titles

        for article in candidates:
            article.relevance_score = score_article_tiered(
                article, intent, article.matched_keywords
            )
        ranked = rank_articles(candidates)
        assert ranked
        assert "madres buscadoras" in ranked[0].title.lower()


class TestShortTokenSafety:
    def test_two_letter_token_does_not_match_arbitrary_substrings(self):
        storage.insert_article(
            Article(
                title="Noticias del día en justicia y provincias",
                url="https://example.com/noticias-ia",
                source="Test",
                summary="resumen general",
            )
        )
        storage.insert_article(
            Article(
                title="La IA transforma la industria",
                url="https://example.com/ia-real",
                source="Test",
                summary="modelos de inteligencia artificial",
            )
        )
        intent = QueryIntent(original_query="IA", core_concepts=["IA"])
        results = storage.search_articles_tiered(intent, limit=20)
        titles = [r.title for r in results]
        assert titles, "expected the real IA article to match"
        assert all("IA" in t or "ia" in normalize_text(t) for t in titles)
        assert not any("Noticias del día" in t for t in titles)

    def test_token_in_normalized_text_rejects_substring(self):
        from app.text import token_in_normalized_text

        norm = normalize_text("Las noticias de justicia provincial")
        assert token_in_normalized_text("IA", norm) is False
        assert token_in_normalized_text("ia", norm) is False
        assert token_in_normalized_text("IA", normalize_text("La IA es útil")) is True


class TestDatabaseIsolation:
    def test_tests_use_temp_database_not_production(self, tmp_path):
        production = Path("news.db")
        production_existed = production.exists()
        production_stat = production.stat().st_mtime_ns if production_existed else None
        production_size = production.stat().st_size if production_existed else None

        configured = os.environ.get("DATABASE_PATH", "")
        assert configured, "conftest must set DATABASE_PATH"
        assert "news.db" not in Path(configured).name or tmp_path in Path(configured).parents
        assert Path(configured).resolve() != production.resolve() if production_existed else True

        storage.insert_article(
            Article(
                title="Isolation probe article",
                url="https://example.com/isolation-probe",
                source="Test",
                summary="should only exist in temp db",
            )
        )
        assert storage.count_articles() >= 1

        if production_existed:
            after = production.stat()
            assert after.st_mtime_ns == production_stat
            assert after.st_size == production_size

    def test_conftest_path_is_under_tmp(self, tmp_path):
        assert str(tmp_path) in os.environ["DATABASE_PATH"]


class TestInsertMetrics:
    def test_first_insert_reports_true_duplicate_reports_false(self):
        article = Article(title="Only once", url="https://example.com/once", source="T")
        assert storage.insert_article(article) is True
        assert storage.insert_article(article) is False

    def test_batch_inserted_count_is_exact(self):
        articles = [
            Article(title=f"A{i}", url=f"https://example.com/batch/{i}", source="T")
            for i in range(5)
        ]
        articles.append(articles[0])  # duplicate within batch
        inserted = storage.insert_articles(articles)
        assert inserted == 5
        assert storage.count_articles() == 5

        inserted_again = storage.insert_articles(articles)
        assert inserted_again == 0

    def test_insert_articles_empty_batch_returns_zero(self):
        assert storage.insert_articles([]) == 0


class TestProductionRetrievalPath:
    def test_execute_search_uses_storage_retrieval_without_live_gemini(self):
        storage.insert_article(
            Article(
                title="La economía mexicana muestra señales de crecimiento",
                url="https://example.com/econ-path",
                source="Test",
                summary="PIB y empleo mejoran",
            )
        )
        intent = QueryIntent(
            original_query="economia mexicana",
            core_concepts=["economia mexicana"],
            equivalent_phrases=["PIB"],
            related_concepts=["inflacion"],
        )
        with patch("app.services.pipeline.expand_query", return_value=intent):
            results = execute_search("economia mexicana", ["economia", "mexicana"])
        assert results
        assert "economía mexicana" in results[0].title

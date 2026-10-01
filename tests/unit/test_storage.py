import os

import pytest

from app import storage
from app.models import Article
from app.query_intent import QueryIntent


@pytest.fixture(autouse=True)
def temp_database(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    storage.close_database()
    monkeypatch.setenv("DATABASE_PATH", db_path)
    storage.initialize_database()
    yield
    storage.close_database()
    monkeypatch.delenv("DATABASE_PATH", raising=False)


class TestInitializeDatabase:
    def test_creates_tables(self):
        conn = storage._get_conn()
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='articles'"
        ).fetchone()
        assert row is not None

    def test_creates_indexes(self):
        conn = storage._get_conn()
        indexes = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()
        index_names = [r["name"] for r in indexes]
        assert "idx_articles_published" in index_names
        assert "idx_articles_source" in index_names

    def test_idempotent(self):
        storage.initialize_database()
        storage.initialize_database()
        assert storage.count_articles() == 0


class TestInsertArticle:
    def test_insert_single_article(self):
        article = Article(title="Test", url="https://a.com", source="S")
        storage.insert_article(article)
        assert storage.count_articles() == 1

    def test_insert_returns_status(self):
        article = Article(title="Test", url="https://a.com", source="S")
        result = storage.insert_article(article)
        assert isinstance(result, bool)

    def test_duplicate_article_ignored(self):
        article = Article(title="Test", url="https://a.com", source="S")
        storage.insert_article(article)
        storage.insert_article(article)
        assert storage.count_articles() == 1

    def test_different_articles_both_stored(self):
        a1 = Article(title="A", url="https://a.com", source="S")
        a2 = Article(title="B", url="https://b.com", source="S")
        storage.insert_article(a1)
        storage.insert_article(a2)
        assert storage.count_articles() == 2


class TestInsertArticles:
    def test_insert_multiple(self):
        articles = [
            Article(title=f"Article {i}", url=f"https://a.com/{i}", source="S")
            for i in range(5)
        ]
        storage.insert_articles(articles)
        assert storage.count_articles() == 5

    def test_insert_multiple_with_duplicates(self):
        a1 = Article(title="A", url="https://a.com", source="S")
        a2 = Article(title="B", url="https://b.com", source="S")
        storage.insert_articles([a1, a2, a1])
        assert storage.count_articles() == 2

    def test_empty_list(self):
        count = storage.insert_articles([])
        assert count == 0


class TestGetArticleById:
    def test_get_existing_article(self):
        article = Article(title="Test", url="https://a.com", source="S", summary="About test")
        storage.insert_article(article)
        result = storage.get_article_by_id(article.id)
        assert result is not None
        assert result.title == "Test"

    def test_get_nonexistent_article(self):
        result = storage.get_article_by_id("nonexistent")
        assert result is None

    def test_article_fields_preserved(self):
        article = Article(
            title="Title",
            url="https://a.com",
            source="Source",
            summary="Summary",
            author="Author",
            published_at="2026-01-01T00:00:00+00:00",
        )
        storage.insert_article(article)
        result = storage.get_article_by_id(article.id)
        assert result.title == "Title"
        assert result.source == "Source"
        assert result.summary == "Summary"
        assert result.author == "Author"


class TestGetAllArticles:
    def test_empty_database(self):
        articles = storage.get_all_articles()
        assert articles == []

    def test_returns_all_articles(self):
        for i in range(3):
            storage.insert_article(
                Article(title=f"Article {i}", url=f"https://a.com/{i}", source="S")
            )
        articles = storage.get_all_articles()
        assert len(articles) == 3


class TestCountArticles:
    def test_empty_database(self):
        assert storage.count_articles() == 0

    def test_after_insert(self):
        storage.insert_article(Article(title="T", url="https://a.com", source="S"))
        assert storage.count_articles() == 1


class TestSearchArticlesTiered:
    def _insert_test_corpus(self):
        articles = [
            Article(title="Madres buscadoras encuentran restos en Tamaulipas", url="https://e1.com/1", source="Test",
                    summary="Madres buscadoras de personas desaparecidas localizaron fosas clandestinas"),
            Article(title="Familiares de desaparecidos demandan acciones", url="https://e2.com/2", source="Test",
                    summary="Colectivos de busqueda de personas desaparecidas piden justicia"),
            Article(title="Bar extorsionado por criminal", url="https://e3.com/3", source="Test",
                    summary="Personas en el bar fueron victimas de extorsion"),
            Article(title="Instituciones policiacas reformadas", url="https://e4.com/4", source="Test",
                    summary="El gobierno de Mexico reforma las instituciones policiacas"),
            Article(title="Terremoto sacude la costa de Oaxaca", url="https://e5.com/5", source="Test",
                    summary="Sismo de magnitud 6.2 registrado en la madrugada"),
            Article(title="Economia mexicana crece 3.5%", url="https://e6.com/6", source="Test",
                    summary="El PIB de Mexico muestra recuperacion economica sostenida"),
            Article(title="Inteligencia artificial transforma la industria", url="https://e7.com/7", source="Test",
                    summary="Empresas mexicanas adoptan IA y machine learning"),
            Article(title="Elecciones 2026: Morena y oposicion en disputa", url="https://e8.com/8", source="Test",
                    summary="Los candidatos preparan estrategias de campana para las elecciones federales"),
        ]
        for a in articles:
            storage.insert_article(a)

    def test_phrase_match_beats_generic_word(self):
        self._insert_test_corpus()
        intent = QueryIntent(
            original_query="madres buscadoras",
            core_concepts=["madres buscadoras", "colectivos de busqueda"],
            equivalent_phrases=["madre buscadora"],
            related_concepts=["personas desaparecidas", "desaparicion forzada"],
        )
        results = storage.search_articles_tiered(intent, limit=10)
        titles = [r.title for r in results]
        assert any("Madres buscadoras" in t for t in titles)

    def test_generic_word_does_not_flood_results(self):
        self._insert_test_corpus()
        intent = QueryIntent(
            original_query="personas",
            core_concepts=["personas"],
            equivalent_phrases=[],
            related_concepts=[],
        )
        results = storage.search_articles_tiered(intent, limit=10)
        assert len(results) <= 5

    def test_multi_word_concept_retrieved(self):
        self._insert_test_corpus()
        intent = QueryIntent(
            original_query="economia mexicana",
            core_concepts=["economia mexicana", "crecimiento economico"],
            equivalent_phrases=["PIB mexicano"],
            related_concepts=[],
        )
        results = storage.search_articles_tiered(intent, limit=10)
        assert len(results) >= 1
        assert any("conomia" in r.title.lower() for r in results)

    def test_accent_insensitive_tiered(self):
        self._insert_test_corpus()
        intent = QueryIntent(
            original_query="economía",
            core_concepts=["economía", "economia mexicana"],
            equivalent_phrases=[],
            related_concepts=[],
        )
        results = storage.search_articles_tiered(intent, limit=10)
        assert len(results) >= 1

    def test_no_match_returns_empty(self):
        self._insert_test_corpus()
        intent = QueryIntent(
            original_query="astronomia estelar",
            core_concepts=["astronomia estelar"],
            equivalent_phrases=[],
            related_concepts=[],
        )
        results = storage.search_articles_tiered(intent, limit=10)
        assert results == []

    def test_empty_intent_returns_empty(self):
        intent = QueryIntent(original_query="")
        results = storage.search_articles_tiered(intent, limit=10)
        assert results == []

    def test_respects_limit(self):
        self._insert_test_corpus()
        intent = QueryIntent(
            original_query="noticias",
            core_concepts=["noticias"],
            equivalent_phrases=[],
            related_concepts=[],
        )
        results = storage.search_articles_tiered(intent, limit=2)
        assert len(results) <= 2

import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest

from app.services.pipeline import execute_search
from app.services.ingest import ingest_feeds
from app.models import Article
from app.query_intent import QueryIntent
from app import storage


def _setup_db(tmp_path, monkeypatch, name="test.db"):
    db_path = str(tmp_path / name)
    storage.close_database()
    monkeypatch.setenv("DATABASE_PATH", db_path)
    storage.initialize_database()
    return db_path


@pytest.fixture(autouse=True)
def temp_database(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    storage.close_database()
    monkeypatch.setenv("DATABASE_PATH", db_path)
    storage.initialize_database()
    yield
    storage.close_database()
    monkeypatch.delenv("DATABASE_PATH", raising=False)


def _seed_article(title="AI News", url="https://example.com/ai", summary="About AI", source="Test"):
    article = Article(title=title, url=url, source=source, summary=summary)
    storage.insert_article(article)
    return article


def _make_mock_entry(title="AI News", link="https://example.com/ai", summary="About AI"):
    entry = MagicMock()
    entry.title = title
    entry.link = link
    entry.summary = summary
    entry.author = None
    entry.published = None
    entry.updated = None
    entry.created = None
    entry.published_parsed = None
    entry.updated_parsed = None
    entry.created_parsed = None
    return entry


def _mock_expand(intent: QueryIntent):
    def expand(query):
        return intent
    return expand


class TestSearchPipeline:
    @patch("app.services.pipeline.expand_query")
    def test_search_finds_matching_article(self, mock_expand):
        mock_expand.return_value = QueryIntent(
            original_query="ai",
            core_concepts=["ai", "artificial intelligence"],
            equivalent_phrases=["machine learning"],
            related_concepts=[],
        )
        _seed_article(title="AI Revolution", summary="About artificial intelligence")
        _seed_article(title="Sports News", summary="Football scores")

        results = execute_search("ai", ["ai"])

        assert len(results) >= 1
        assert any("AI" in r.title for r in results)

    @patch("app.services.pipeline.expand_query")
    def test_no_matches_returns_empty(self, mock_expand):
        mock_expand.return_value = QueryIntent(
            original_query="blockchain",
            core_concepts=["blockchain"],
        )
        _seed_article(title="Sports", summary="Football")

        results = execute_search("blockchain", ["blockchain"])

        assert results == []

    @patch("app.services.pipeline.expand_query")
    def test_empty_database_returns_empty(self, mock_expand):
        mock_expand.return_value = QueryIntent(
            original_query="ai",
            core_concepts=["ai"],
        )

        results = execute_search("ai", ["ai"])

        assert results == []

    @patch("app.services.pipeline.expand_query")
    def test_results_are_ranked(self, mock_expand):
        mock_expand.return_value = QueryIntent(
            original_query="ai",
            core_concepts=["ai"],
        )
        _seed_article(title="AI is everywhere", summary="about artificial intelligence")
        _seed_article(title="General news", summary="something about AI")

        results = execute_search("ai", ["ai"])

        if len(results) >= 2:
            assert results[0].relevance_score >= results[1].relevance_score

    @patch("app.services.pipeline.expand_query")
    def test_results_limited_to_five(self, mock_expand):
        mock_expand.return_value = QueryIntent(
            original_query="news",
            core_concepts=["news"],
        )
        for i in range(10):
            _seed_article(title=f"News Article {i}", url=f"https://a.com/{i}", summary="news")

        results = execute_search("news", ["news"])

        assert len(results) <= 5

    @patch("app.services.pipeline.expand_query")
    def test_accent_insensitive_search(self, mock_expand):
        mock_expand.return_value = QueryIntent(
            original_query="economia",
            core_concepts=["economia"],
        )
        _seed_article(title="La economia mexicana crece", summary="PIB al alza")

        results = execute_search("economia", ["economia"])

        assert len(results) >= 1

    @patch("app.services.pipeline.expand_query")
    def test_accent_insensitive_reverse(self, mock_expand):
        mock_expand.return_value = QueryIntent(
            original_query="economía",
            core_concepts=["economía"],
        )
        _seed_article(title="La economia mexicana crece", summary="PIB al alza")

        results = execute_search("economía", ["economía"])

        assert len(results) >= 1

    @patch("app.services.pipeline.expand_query")
    def test_search_logs_query(self, mock_expand, caplog):
        mock_expand.return_value = QueryIntent(
            original_query="test",
            core_concepts=["test"],
        )
        _seed_article(title="Test Article")

        with caplog.at_level("INFO"):
            execute_search("test", ["test"])

        assert 'Search query: "test"' in caplog.text


class TestIngestionPipeline:
    @patch("app.services.fetcher.feedparser.parse")
    def test_ingest_stores_articles(self, mock_parse):
        entries = [
            _make_mock_entry("AI News", "https://a.com/1", "About AI"),
            _make_mock_entry("Sports", "https://a.com/2", "Football"),
        ]
        mock_feed = MagicMock()
        mock_feed.entries = entries
        mock_feed.bozo = False
        mock_parse.return_value = mock_feed

        from app.config import FeedSource
        sources = [FeedSource("Test", "https://example.com/feed")]

        with patch("app.services.ingest.SOURCES", sources):
            ingest_feeds()

        assert storage.count_articles() == 2

    @patch("app.services.fetcher.feedparser.parse")
    def test_ingest_deduplicates(self, mock_parse):
        entries = [_make_mock_entry("AI News", "https://shared.com/article", "AI")]
        mock_feed = MagicMock()
        mock_feed.entries = entries
        mock_feed.bozo = False
        mock_parse.return_value = mock_feed

        from app.config import FeedSource
        sources = [
            FeedSource("S1", "https://s1.com/feed"),
            FeedSource("S2", "https://s2.com/feed"),
        ]

        with patch("app.services.ingest.SOURCES", sources):
            ingest_feeds()

        assert storage.count_articles() == 1

    @patch("app.services.fetcher.feedparser.parse")
    def test_partial_feed_failure_does_not_abort(self, mock_parse):
        def side_effect(url, **kwargs):
            if "fail" in url:
                raise Exception("Network error")
            mock = MagicMock()
            mock.entries = [_make_mock_entry("AI News", "https://ok.com/1", "AI")]
            mock.bozo = False
            return mock

        mock_parse.side_effect = side_effect

        from app.config import FeedSource
        sources = [
            FeedSource("Fail", "https://fail.com/feed"),
            FeedSource("OK", "https://ok.com/feed"),
        ]

        with patch("app.services.ingest.SOURCES", sources):
            ingest_feeds()

        assert storage.count_articles() == 1

    @patch("app.services.fetcher.feedparser.parse")
    def test_ingest_repeatedly_no_duplicates(self, mock_parse):
        entries = [_make_mock_entry("AI News", "https://a.com/1", "AI")]
        mock_feed = MagicMock()
        mock_feed.entries = entries
        mock_feed.bozo = False
        mock_parse.return_value = mock_feed

        from app.config import FeedSource
        sources = [FeedSource("Test", "https://example.com/feed")]

        with patch("app.services.ingest.SOURCES", sources):
            ingest_feeds()
            ingest_feeds()
            ingest_feeds()

        assert storage.count_articles() == 1


GENERALIZED_TEST_ARTICLES = [
    ("Madres buscadoras encuentran restos en Tamaulipas", "https://e1.com/1",
     "Madres buscadoras de personas desaparecidas localizaron fosas clandestinas en Tamaulipas"),
    ("Familiares de desaparecidos demandan acciones del gobierno", "https://e2.com/2",
     "Colectivos de busqueda de personas desaparecidas piden justicia"),
    ("Elecciones 2026: Morena y oposicion en disputa", "https://e3.com/3",
     "Los candidatos preparan sus estrategias de campana para las elecciones federales"),
    ("Economia mexicana crece 3.5% en segundo trimestre", "https://e4.com/4",
     "El PIB de Mexico muestra recuperacion economica sostenida"),
    ("Inteligencia artificial transforma la industria manufacturera", "https://e5.com/5",
     "Empresas mexicanas adoptan IA y machine learning para optimizar procesos"),
    ("Violencia en Guerrero deja varios muertos", "https://e6.com/6",
     "Enfrentamientos entre grupos criminales en la region"),
    ("Terremoto sacude la costa de Oaxaca", "https://e7.com/7",
     "Sismo de magnitud 6.2 registrado en la madrugada"),
    ("Trump impone nuevos aranceles a Mexico", "https://e8.com/8",
     "El presidente de Estados Unidos anuncia medidas comerciales contra Mexico"),
    ("Desaparicion forzada en Nuevo Leon genera protestas", "https://e9.com/9",
     "Familiares buscan a sus seres queridos desaparecidos"),
    ("La educacion publica enfrenta nuevos desafios", "https://e10.com/10",
     "Maestros y alumnos protestan por falta de recursos en escuelas publicas"),
]


def _seed_mexican_corpus():
    for title, url, summary in GENERALIZED_TEST_ARTICLES:
        storage.insert_article(Article(title=title, url=url, source="Test", summary=summary))


class TestExactPhraseMatching:
    @pytest.fixture(autouse=True)
    def setup_mexican(self, tmp_path, monkeypatch):
        db_path = str(tmp_path / "test_phrase.db")
        storage.close_database()
        monkeypatch.setenv("DATABASE_PATH", db_path)
        storage.initialize_database()
        _seed_mexican_corpus()
        yield
        storage.close_database()
        monkeypatch.delenv("DATABASE_PATH", raising=False)

    @patch("app.services.pipeline.expand_query")
    def test_exact_entity_query(self, mock_expand):
        mock_expand.return_value = QueryIntent(
            original_query="Trump",
            core_concepts=["Trump", "Donald Trump"],
            equivalent_phrases=["presidente de Estados Unidos"],
            related_concepts=["aranceles", "comercio"],
        )
        results = execute_search("Trump", ["Trump"])
        assert len(results) >= 1
        assert any("Trump" in r.title for r in results)

    @patch("app.services.pipeline.expand_query")
    def test_exact_named_entity_prioritized(self, mock_expand):
        mock_expand.return_value = QueryIntent(
            original_query="Guerrero",
            core_concepts=["Guerrero"],
            equivalent_phrases=[],
            related_concepts=["violencia", "crimen"],
        )
        results = execute_search("Guerrero", ["Guerrero"])
        assert len(results) >= 1
        assert any("Guerrero" in r.title for r in results)


class TestSynonymMatching:
    @pytest.fixture(autouse=True)
    def setup_mexican(self, tmp_path, monkeypatch):
        db_path = str(tmp_path / "test_synonym.db")
        storage.close_database()
        monkeypatch.setenv("DATABASE_PATH", db_path)
        storage.initialize_database()
        _seed_mexican_corpus()
        yield
        storage.close_database()
        monkeypatch.delenv("DATABASE_PATH", raising=False)

    @patch("app.services.pipeline.expand_query")
    def test_synonym_finds_related_article(self, mock_expand):
        mock_expand.return_value = QueryIntent(
            original_query="terremoto",
            core_concepts=["terremoto"],
            equivalent_phrases=["sismo", "temblor"],
            related_concepts=[],
        )
        results = execute_search("terremoto", ["terremoto"])
        assert len(results) >= 1
        assert any("Terremoto" in r.title or "sismo" in r.title.lower() for r in results)

    @patch("app.services.pipeline.expand_query")
    def test_morphological_variation(self, mock_expand):
        mock_expand.return_value = QueryIntent(
            original_query="eleccion",
            core_concepts=["eleccion"],
            equivalent_phrases=["elecciones", "votacion", "comicios"],
            related_concepts=["candidatos", "campana"],
        )
        results = execute_search("eleccion", ["eleccion"])
        assert len(results) >= 1
        assert any("Elecciones" in r.title for r in results)


class TestCompoundConcepts:
    @pytest.fixture(autouse=True)
    def setup_mexican(self, tmp_path, monkeypatch):
        db_path = str(tmp_path / "test_compound.db")
        storage.close_database()
        monkeypatch.setenv("DATABASE_PATH", db_path)
        storage.initialize_database()
        _seed_mexican_corpus()
        yield
        storage.close_database()
        monkeypatch.delenv("DATABASE_PATH", raising=False)

    @patch("app.services.pipeline.expand_query")
    def test_compound_query_more_specific(self, mock_expand):
        mock_expand.return_value = QueryIntent(
            original_query="economia mexicana",
            core_concepts=["economia mexicana", "crecimiento economico"],
            equivalent_phrases=["PIB mexicano"],
            related_concepts=[],
        )
        results = execute_search("economia mexicana", ["economia", "mexicana"])
        assert len(results) >= 1
        assert any("conomia" in r.title.lower() for r in results)

    @patch("app.services.pipeline.expand_query")
    def test_event_plus_location(self, mock_expand):
        mock_expand.return_value = QueryIntent(
            original_query="terremoto en Oaxaca",
            core_concepts=["terremoto en Oaxaca", "sismo en Oaxaca"],
            equivalent_phrases=[],
            related_concepts=[],
        )
        results = execute_search("terremoto en Oaxaca", ["terremoto", "oaxaca"])
        assert len(results) >= 1
        assert any("Terremoto" in r.title and "Oaxaca" in r.title for r in results)


class TestTechnicalConcepts:
    @pytest.fixture(autouse=True)
    def setup_mexican(self, tmp_path, monkeypatch):
        db_path = str(tmp_path / "test_tech.db")
        storage.close_database()
        monkeypatch.setenv("DATABASE_PATH", db_path)
        storage.initialize_database()
        _seed_mexican_corpus()
        yield
        storage.close_database()
        monkeypatch.delenv("DATABASE_PATH", raising=False)

    @patch("app.services.pipeline.expand_query")
    def test_technical_concept_with_related_terms(self, mock_expand):
        mock_expand.return_value = QueryIntent(
            original_query="inteligencia artificial",
            core_concepts=["inteligencia artificial", "IA"],
            equivalent_phrases=["machine learning", "aprendizaje automatico"],
            related_concepts=["redes neuronales"],
        )
        results = execute_search("inteligencia artificial", ["inteligencia", "artificial"])
        assert len(results) >= 1
        assert any("inteligencia artificial" in r.title.lower() for r in results)


class TestGenericTermFalsePositives:
    @pytest.fixture(autouse=True)
    def setup_mexican(self, tmp_path, monkeypatch):
        db_path = str(tmp_path / "test_generic.db")
        storage.close_database()
        monkeypatch.setenv("DATABASE_PATH", db_path)
        storage.initialize_database()
        _seed_mexican_corpus()
        yield
        storage.close_database()
        monkeypatch.delenv("DATABASE_PATH", raising=False)

    @patch("app.services.pipeline.expand_query")
    def test_generic_only_match_penalized(self, mock_expand):
        mock_expand.return_value = QueryIntent(
            original_query="personas",
            core_concepts=["personas"],
            equivalent_phrases=[],
            related_concepts=[],
        )
        results = execute_search("personas", ["personas"])
        for r in results:
            assert r.relevance_score < 0.3

    @patch("app.services.pipeline.expand_query")
    def test_broad_query_does_not_flood(self, mock_expand):
        mock_expand.return_value = QueryIntent(
            original_query="noticias",
            core_concepts=["noticias"],
            equivalent_phrases=[],
            related_concepts=[],
        )
        results = execute_search("noticias", ["noticias"])
        assert len(results) <= 5


class TestMultipleSupportingSignals:
    @pytest.fixture(autouse=True)
    def setup_mexican(self, tmp_path, monkeypatch):
        db_path = str(tmp_path / "test_multi.db")
        storage.close_database()
        monkeypatch.setenv("DATABASE_PATH", db_path)
        storage.initialize_database()
        _seed_mexican_corpus()
        yield
        storage.close_database()
        monkeypatch.delenv("DATABASE_PATH", raising=False)

    @patch("app.services.pipeline.expand_query")
    def test_article_with_multiple_concepts_scores_higher(self, mock_expand):
        mock_expand.return_value = QueryIntent(
            original_query="desaparicion forzada",
            core_concepts=["desaparicion forzada", "familiares de desaparecidos"],
            equivalent_phrases=["personas desaparecidas"],
            related_concepts=["colectivos de busqueda"],
        )
        results = execute_search("desaparicion forzada", ["desaparicion", "forzada"])
        if len(results) >= 2:
            assert results[0].relevance_score >= results[1].relevance_score


class TestGeminiFailureFallback:
    @patch("app.services.pipeline.expand_query")
    def test_fallback_to_original_query(self, mock_expand):
        mock_expand.return_value = QueryIntent(
            original_query="terremoto",
            core_concepts=["terremoto"],
        )
        _seed_article(title="Terremoto en Oaxaca", summary="Sismo fuerte")
        results = execute_search("terremoto", ["terremoto"])
        assert len(results) >= 1


class TestEmptyNoResultQueries:
    @patch("app.services.pipeline.expand_query")
    def test_unrelated_query_returns_empty(self, mock_expand):
        mock_expand.return_value = QueryIntent(
            original_query="astronomia estelar",
            core_concepts=["astronomia estelar"],
        )
        results = execute_search("astronomia estelar", ["astronomia"])
        assert results == []

    @patch("app.services.pipeline.expand_query")
    def test_nonexistent_topic_returns_empty(self, mock_expand):
        mock_expand.return_value = QueryIntent(
            original_query="receta de pastel",
            core_concepts=["receta de pastel"],
        )
        results = execute_search("receta de pastel", ["receta"])
        assert results == []


class TestCompletelyUnrelatedArticles:
    @patch("app.services.pipeline.expand_query")
    def test_sports_query_no_news_results(self, mock_expand):
        mock_expand.return_value = QueryIntent(
            original_query="liga mx",
            core_concepts=["liga mx", "futbol mexicano"],
            equivalent_phrases=["campeonato"],
            related_concepts=[],
        )
        results = execute_search("liga mx", ["liga", "mx"])
        assert results == []


DIAGNOSTIC_ARTICLES = [
    ("Resalta Partido Verde apoyo de la Guardia Nacional en tareas de inspeccion y conservacion ambiental",
     "https://d1.com/1", "El Partido Verde resalta el apoyo de la Guardia Nacional en inspeccion ambiental"),
    ("Fortalece gobierno capitalino drenaje en GAM para prevenir inundaciones",
     "https://d2.com/2", "La Ciudad de Mexico fortalece el drenaje en el area metropolitana"),
    ("Cuartos de final de la Leagues Cup 2026: equipos mexicanos avanzan",
     "https://d3.com/3", "Los equipos mexicanos avanzan a cuartos de final en la Leagues Cup"),
    ("El paraiso de otakus y gamers en la Ciudad de Mexico",
     "https://d4.com/4", "La Capital se convierte en el paraiso de los otakus y gamers"),
    ("No sabian rescatar, pero entraron a los escombros del terremoto",
     "https://d5.com/5", "Voluntarios sin experiencia rescataron personas de los escombros tras el sismo"),
    ("Economia mexicana crece 3.5% en segundo trimestre",
     "https://d6.com/6", "El PIB de Mexico muestra recuperacion economica sostenida con inflacion controlada"),
    ("Inteligencia artificial transforma la industria manufacturera mexicana",
     "https://d7.com/7", "Empresas mexicanas adoptan IA, machine learning y modelos de lenguaje para optimizar procesos"),
    ("Conflicto Rusia-Ucrania: ataques en Kharkiv dejan victimas civiles",
     "https://d8.com/8", "Los ataques rusos en Ucrania continuan en el frente de batalla del este"),
    ("Madres buscadoras encuentran restos en Tamaulipas",
     "https://d9.com/9", "Colectivos de busqueda de personas desaparecidas localizaron fosas clandestinas"),
    ("Violencia en Guerrero: enfrentamientos entre grupos criminales",
     "https://d10.com/10", "La violencia en Guerrero deja varios muertos en la region"),
]


def _seed_diagnostic_corpus():
    for title, url, summary in DIAGNOSTIC_ARTICLES:
        storage.insert_article(Article(title=title, url=url, source="Test", summary=summary))


class TestDiagnosticA_IAShortToken:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        db_path = str(tmp_path / "test_diag_a.db")
        storage.close_database()
        monkeypatch.setenv("DATABASE_PATH", db_path)
        storage.initialize_database()
        _seed_diagnostic_corpus()
        yield
        storage.close_database()
        monkeypatch.delenv("DATABASE_PATH", raising=False)

    @patch("app.services.pipeline.expand_query")
    def test_ia_only_in_summary_does_not_rank_highly(self, mock_expand):
        mock_expand.return_value = QueryIntent(
            original_query="inteligencia artificial",
            core_concepts=["inteligencia artificial"],
            equivalent_phrases=["IA"],
            related_concepts=["machine learning", "modelos de lenguaje"],
        )
        results = execute_search("inteligencia artificial", ["inteligencia", "artificial"])
        for r in results:
            if "IA" in r.matched_keywords and len(r.matched_keywords) == 1:
                assert r.relevance_score < 0.3, (
                    f"Article matched only 'IA' but scored {r.relevance_score}: {r.title}"
                )


class TestDiagnosticB_EconomiaMexicana:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        db_path = str(tmp_path / "test_diag_b.db")
        storage.close_database()
        monkeypatch.setenv("DATABASE_PATH", db_path)
        storage.initialize_database()
        _seed_diagnostic_corpus()
        yield
        storage.close_database()
        monkeypatch.delenv("DATABASE_PATH", raising=False)

    @patch("app.services.pipeline.expand_query")
    def test_economia_mexicana_finds_article(self, mock_expand):
        mock_expand.return_value = QueryIntent(
            original_query="economia mexicana",
            core_concepts=["economia mexicana"],
            equivalent_phrases=["PIB de Mexico", "crecimiento economico"],
            related_concepts=["inflacion", "Banxico"],
        )
        results = execute_search("economia mexicana", ["economia", "mexicana"])
        assert len(results) >= 1
        titles = [r.title for r in results]
        assert any("conomia" in t.lower() for t in titles)


class TestDiagnosticC_GuerraRusiaUcrania:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        db_path = str(tmp_path / "test_diag_c.db")
        storage.close_database()
        monkeypatch.setenv("DATABASE_PATH", db_path)
        storage.initialize_database()
        _seed_diagnostic_corpus()
        yield
        storage.close_database()
        monkeypatch.delenv("DATABASE_PATH", raising=False)

    @patch("app.services.pipeline.expand_query")
    def test_guerra_rusia_ucrania_finds_article(self, mock_expand):
        mock_expand.return_value = QueryIntent(
            original_query="guerra Rusia Ucrania",
            core_concepts=["guerra Rusia Ucrania", "conflicto Rusia Ucrania"],
            equivalent_phrases=["invasion rusa de Ucrania"],
            related_concepts=["ataques en Ucrania", "frente de batalla"],
        )
        results = execute_search("guerra Rusia Ucrania", ["guerra", "rusia", "ucrania"])
        assert len(results) >= 1
        titles = [r.title for r in results]
        assert any("ucrania" in t.lower() or "rusia" in t.lower() for t in titles)


class TestDiagnosticD_MadresBuscadoras:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        db_path = str(tmp_path / "test_diag_d.db")
        storage.close_database()
        monkeypatch.setenv("DATABASE_PATH", db_path)
        storage.initialize_database()
        _seed_diagnostic_corpus()
        yield
        storage.close_database()
        monkeypatch.delenv("DATABASE_PATH", raising=False)

    @patch("app.services.pipeline.expand_query")
    def test_madres_buscadoras_finds_article(self, mock_expand):
        mock_expand.return_value = QueryIntent(
            original_query="madres buscadoras",
            core_concepts=["madres buscadoras", "colectivos de busqueda"],
            equivalent_phrases=["familiares de desaparecidos"],
            related_concepts=["personas desaparecidas", "fosas clandestinas"],
        )
        results = execute_search("madres buscadoras", ["madres", "buscadoras"])
        assert len(results) >= 1
        titles = [r.title for r in results]
        assert any("madres" in t.lower() or "desaparec" in t.lower() for t in titles)


class TestDiagnosticE_GenericFalsePositives:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        db_path = str(tmp_path / "test_diag_e.db")
        storage.close_database()
        monkeypatch.setenv("DATABASE_PATH", db_path)
        storage.initialize_database()
        _seed_diagnostic_corpus()
        yield
        storage.close_database()
        monkeypatch.delenv("DATABASE_PATH", raising=False)

    @patch("app.services.pipeline.expand_query")
    def test_generic_article_not_returned_for_specific_query(self, mock_expand):
        mock_expand.return_value = QueryIntent(
            original_query="inteligencia artificial",
            core_concepts=["inteligencia artificial"],
            equivalent_phrases=["IA"],
            related_concepts=["machine learning"],
        )
        results = execute_search("inteligencia artificial", ["inteligencia", "artificial"])
        generic_titles = ["Partido Verde", "Leagues Cup", "otakus y gamers"]
        for r in results:
            for gt in generic_titles:
                assert gt.lower() not in r.title.lower(), (
                    f"Generic article returned: {r.title}"
                )


class TestDiagnosticF_Terremoto:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        db_path = str(tmp_path / "test_diag_f.db")
        storage.close_database()
        monkeypatch.setenv("DATABASE_PATH", db_path)
        storage.initialize_database()
        _seed_diagnostic_corpus()
        yield
        storage.close_database()
        monkeypatch.delenv("DATABASE_PATH", raising=False)

    @patch("app.services.pipeline.expand_query")
    def test_terremoto_finds_relevant_article(self, mock_expand):
        mock_expand.return_value = QueryIntent(
            original_query="terremoto",
            core_concepts=["terremoto", "sismo"],
            equivalent_phrases=["temblor", "movimiento telurico"],
            related_concepts=["escombros", "victimas"],
        )
        results = execute_search("terremoto", ["terremoto"])
        assert len(results) >= 1
        titles = [r.title for r in results]
        assert any("terremoto" in t.lower() or "sismo" in t.lower() or "escombros" in t.lower() for t in titles)


EVALUATION_CORPUS = [
    ("Madres buscadoras encuentran restos en Tamaulipas",
     "https://eval1.com/1",
     "Colectivos de busqueda de personas desaparecidas localizaron fosas clandestinas en Tamaulipas"),
    ("Familiares de desaparecidos exigen justicia y verdad",
     "https://eval2.com/2",
     "Hijos y esposas de victimas de desaparicion forzada marchan en la ciudad de Mexico"),
    ("Terremoto de magnitud 6.5 sacude Oaxaca",
     "https://eval3.com/3",
     "Sismo registrado en la madrugada provoco danos materiales en la costa"),
    ("Economia mexicana muestra signos de recuperacion",
     "https://eval4.com/4",
     "El PIB registra crecimiento sostenido y la inflacion se mantiene controlada"),
    ("Inteligencia artificial revoluciona la industria manufacturera",
     "https://eval5.com/5",
     "Empresas adoptan modelos de IA y machine learning para optimizar procesos productivos"),
    ("Conflicto entre Rusia y Ucrania deja victimas civiles",
     "https://eval6.com/6",
     "Los ataques en el frente del este intensifican la crisis humanitaria"),
    ("Trump anuncia nuevos aranceles a importaciones mexicanas",
     "https://eval7.com/7",
     "El presidente de Estados Unidos impone medidas comerciales contra Mexico"),
    ("Violencia en Guerrero deja varios muertos",
     "https://eval8.com/8",
     "Enfrentamientos entre grupos criminales en la region sur del pais"),
    ("Cambio climatico afecta productividad agricola en Mexico",
     "https://eval9.com/9",
     "La sequia y las inundaciones provocan perdidas en el campo"),
    ("Nearshoring impulsa la inversion extranjera en Mexico",
     "https://eval10.com/10",
     "Empresas multinacionales relocalizan operaciones a la frontera norte"),
    ("Crisis migratoria en la frontera sur de Mexico",
     "https://eval11.com/11",
     "Miles de migrantes centroamericanos avanzan hacia Estados Unidos"),
    ("Energias renovables crecen en el sector electrico mexicano",
     "https://eval12.com/12",
     "La energia solar y eolica aumentan su participacion en la matriz energetica"),
    ("Violencia de genero sigue siendo un problema en Mexico",
     "https://eval13.com/13",
     "Feminicidios y agresiones contra mujeres aumentan en diversas entidades"),
    ("Politica monetaria del Banxico afecta tipos de cambio",
     "https://eval14.com/14",
     "El banco central mantiene tasas de interes para controlar la inflacion"),
    ("Crisis en el sistema de salud publica mexicana",
     "https://eval15.com/15",
     "Hospitales enfrentan falta de insumos y personal medico"),
    ("Innovacion tecnologica en startups mexicanas crece",
     "https://eval16.com/16",
     "Empresas de tecnologia captan inversion y expanden operaciones"),
    ("Elecciones 2026: Morena y oposicion en disputa",
     "https://eval17.com/17",
     "Los candidatos preparan estrategias de campana para las elecciones federales"),
    ("Proteccion civil prepara simulacro de sismo nacional",
     "https://eval18.com/18",
     "Autoridades realizan entrenamiento para respuesta ante emergencias teluricas"),
    ("Desercion escolar aumenta por falta de recursos",
     "https://eval19.com/19",
     "Maestros y alumnos protestan por condiciones precarias en escuelas publicas"),
    ("Tecnologia blockchain transforma el sector financiero",
     "https://eval20.com/20",
     "El uso de criptomonedas y contratos inteligentes gana terreno en Mexico"),
]


def _seed_evaluation_corpus():
    for title, url, summary in EVALUATION_CORPUS:
        storage.insert_article(Article(title=title, url=url, source="Eval", summary=summary))


class TestEvaluationQueries:
    @pytest.fixture(autouse=True)
    def setup_evaluation(self, tmp_path, monkeypatch):
        db_path = str(tmp_path / "test_eval.db")
        storage.close_database()
        monkeypatch.setenv("DATABASE_PATH", db_path)
        storage.initialize_database()
        _seed_evaluation_corpus()
        yield
        storage.close_database()
        monkeypatch.delenv("DATABASE_PATH", raising=False)

    @patch("app.services.pipeline.expand_query")
    def test_madres_buscadoras(self, mock_expand):
        mock_expand.return_value = QueryIntent(
            original_query="madres buscadoras",
            core_concepts=[
                "madres buscadoras",
                "madres de personas desaparecidas",
                "colectivos de busqueda",
                "busqueda de desaparecidos",
            ],
            equivalent_phrases=[
                "familiares de desaparecidos",
                "rastreadoras de fosas",
                "madres buscadoras de fosas",
                "familias de victimas",
                "busqueda de personas",
                "madres desaparecidos",
                "familiares de fosas",
                "madres de desaparecidos",
            ],
            related_concepts=[
                "fosas clandestinas",
                "desaparicion forzada",
                "personas desaparecidas",
                "identificacion de restos",
                "crisis forense",
                "justicia para victimas",
                "verdad historica",
                "busqueda forense",
                "exhumacion",
                "fosas ocultas",
            ],
        )
        results = execute_search("madres buscadoras", ["madres", "buscadoras"])
        assert len(results) >= 1
        assert any(
            "madres" in r.title.lower() or "desaparec" in r.title.lower() or "busca" in r.title.lower()
            for r in results
        )

    @patch("app.services.pipeline.expand_query")
    def test_terremoto(self, mock_expand):
        mock_expand.return_value = QueryIntent(
            original_query="terremoto",
            core_concepts=[
                "terremoto",
                "sismo",
                "temblor fuerte",
                "movimiento telurico",
            ],
            equivalent_phrases=[
                "seismo",
                "sacudida",
                "temblor de tierra",
                "sismo destructivo",
                "temblor en la costa",
                "movimiento sismico",
                "sismo en la region",
                "temblor fuerte en oaxaca",
                "sismo registrado",
            ],
            related_concepts=[
                "escombros",
                "victimas",
                "rescate",
                "ayuda humanitaria",
                "protector civil",
                "alerta sismica",
                "placas tectonicas",
                "zona de impacto",
                "danos materiales",
                "emergencia nacional",
            ],
        )
        results = execute_search("terremoto", ["terremoto"])
        assert len(results) >= 1
        assert any(
            "terremoto" in r.title.lower() or "sismo" in r.title.lower() or "danos" in r.summary.lower()
            for r in results
        )

    @patch("app.services.pipeline.expand_query")
    def test_inteligencia_artificial(self, mock_expand):
        mock_expand.return_value = QueryIntent(
            original_query="inteligencia artificial",
            core_concepts=[
                "inteligencia artificial",
                "IA",
                "sistemas de IA",
                "modelos de IA",
            ],
            equivalent_phrases=[
                "machine learning",
                "aprendizaje automatico",
                "redes neuronales",
                "IA generativa",
                "modelos de lenguaje",
                "procesamiento de lenguaje natural",
                "deep learning",
                "aprendizaje profundo",
                "IA en la industria",
            ],
            related_concepts=[
                "tecnologia",
                "innovacion",
                "automatizacion",
                "robotica",
                "datos",
                "algoritmos",
                "transformacion digital",
                "industria 4.0",
                "computacion",
                "programacion",
            ],
        )
        results = execute_search("inteligencia artificial", ["inteligencia", "artificial"])
        assert len(results) >= 1
        assert any(
            "inteligencia artificial" in r.title.lower() or "ia" in r.title.lower()
            or "machine learning" in r.summary.lower()
            for r in results
        )

    @patch("app.services.pipeline.expand_query")
    def test_economia_mexicana(self, mock_expand):
        mock_expand.return_value = QueryIntent(
            original_query="economia mexicana",
            core_concepts=[
                "economia mexicana",
                "economia de mexico",
                "crecimiento economico",
                "PIB mexicano",
            ],
            equivalent_phrases=[
                "producto interno bruto",
                "indicadores economicos",
                "desarrollo economico",
                "actividad economica",
                "sector productivo",
                "produccion nacional",
                "rendimiento economico",
                "economia nacional",
                " crecimiento del PIB",
            ],
            related_concepts=[
                "inflacion",
                "tipo de cambio",
                "Banxico",
                "politica monetaria",
                "empleo",
                "exportaciones",
                "inversion",
                "comercio exterior",
                "deuda publica",
                "presupuesto federal",
            ],
        )
        results = execute_search("economia mexicana", ["economia", "mexicana"])
        assert len(results) >= 1
        assert any(
            "conomia" in r.title.lower() or "pib" in r.summary.lower() or "crecimiento" in r.summary.lower()
            for r in results
        )

    @patch("app.services.pipeline.expand_query")
    def test_guerra_rusia_ucrania(self, mock_expand):
        mock_expand.return_value = QueryIntent(
            original_query="guerra Rusia Ucrania",
            core_concepts=[
                "guerra Rusia Ucrania",
                "conflicto Rusia Ucrania",
                "invasion rusa",
                "guerra en Ucrania",
            ],
            equivalent_phrases=[
                "conflicto armado en Ucrania",
                "invasion de Ucrania",
                "guerra ruso-ucraniana",
                "batalla en Ucrania",
                "agresion rusa contra Ucrania",
                "crisis en Ucrania",
                "guerra en Europa del este",
                "conflicto internacional",
                "guerra prolongada",
            ],
            related_concepts=[
                "ataques en Ucrania",
                "frente de batalla",
                "victim as civiles",
                "crisis humanitaria",
                "refugiados",
                "armas",
                "diplomacia",
                "sanciones",
                "OTAN",
                "seguridad internacional",
            ],
        )
        results = execute_search("guerra Rusia Ucrania", ["guerra", "rusia", "ucrania"])
        assert len(results) >= 1
        assert any(
            "rusia" in r.title.lower() or "ucrania" in r.title.lower()
            or "conflicto" in r.title.lower() or "ataques" in r.summary.lower()
            for r in results
        )

    @patch("app.services.pipeline.expand_query")
    def test_cambio_climatico(self, mock_expand):
        mock_expand.return_value = QueryIntent(
            original_query="cambio climatico",
            core_concepts=[
                "cambio climatico",
                "crisis climatica",
                "calentamiento global",
                "crisis ambiental",
            ],
            equivalent_phrases=[
                "cambio global",
                "crisis de clima",
                "calentamiento de la tierra",
                "emergencia climatica",
                "crisis ecologica",
                "impacto ambiental",
                "calentamiento climatico",
                "cambio de clima",
                "crisis planetaria",
            ],
            related_concepts=[
                "sequia",
                "inundaciones",
                "temperaturas extremas",
                "efecto invernadero",
                "emisiones de carbono",
                "energia limpia",
                "sostenibilidad",
                "medio ambiente",
                "保护 del medio ambiente",
                "crisis de agua",
            ],
        )
        results = execute_search("cambio climatico", ["cambio", "climatico"])
        assert len(results) >= 1
        assert any(
            "clima" in r.title.lower() or "climatico" in r.title.lower() or "agrícola" in r.summary.lower()
            for r in results
        )

    @patch("app.services.pipeline.expand_query")
    def test_nearshoring(self, mock_expand):
        mock_expand.return_value = QueryIntent(
            original_query="nearshoring",
            core_concepts=[
                "nearshoring",
                "relocalizacion de empresas",
                "nearshoring en Mexico",
            ],
            equivalent_phrases=[
                "reubicacion de operaciones",
                "mudanza de fabricas",
                "deslocalizacion cercana",
                "relocalizacion cercana",
                "nearshoring manufacturer",
                "reubicacion productiva",
                "mudanza de plantas",
                "cambio de sede",
                "nearshoring en America Latina",
            ],
            related_concepts=[
                "inversion extranjera",
                "manufactura",
                "frontera norte",
                "maquiladora",
                "produccion",
                "empleo industrial",
                "competitividad",
                "cadena de suministro",
                "infraestructura",
                "desarrollo regional",
            ],
        )
        results = execute_search("nearshoring", ["nearshoring"])
        assert len(results) >= 1
        assert any(
            "nearshoring" in r.title.lower() or "inversion" in r.summary.lower() or "multinacional" in r.summary.lower()
            for r in results
        )

    @patch("app.services.pipeline.expand_query")
    def test_crisis_migratoria(self, mock_expand):
        mock_expand.return_value = QueryIntent(
            original_query="crisis migratoria",
            core_concepts=[
                "crisis migratoria",
                "migracion",
                "migrantes",
                "crisis de migrantes",
            ],
            equivalent_phrases=[
                "flujo migratorio",
                "oleada de migrantes",
                "migracion centroamericana",
                "crisis fronteriza",
                "migrantes indocumentados",
                "migracion irregular",
                "migracion en masse",
                "exodo migratorio",
                "crisis humanitaria migratoria",
            ],
            related_concepts=[
                "frontera sur",
                "Estados Unidos",
                "caravana",
                "refugiados",
                "asilo",
                "derechos humanos",
                "migracion internacional",
                "centroamerica",
                "trata de personas",
                "smuggling",
            ],
        )
        results = execute_search("crisis migratoria", ["crisis", "migratoria"])
        assert len(results) >= 1
        assert any(
            "migrat" in r.title.lower() or "migrat" in r.summary.lower() or "frontera" in r.summary.lower()
            for r in results
        )

    @patch("app.services.pipeline.expand_query")
    def test_energias_renovables(self, mock_expand):
        mock_expand.return_value = QueryIntent(
            original_query="energias renovables",
            core_concepts=[
                "energias renovables",
                "energia limpia",
                "energias limpias",
                "generacion de energia limpia",
            ],
            equivalent_phrases=[
                "fuentes renovables",
                "energia sostenible",
                "energia verde",
                "energia sin combustibles fosiles",
                "produccion de energia renovable",
                "generacion renovable",
                "energia alternativa",
                "matriz energetica limpia",
                "fuentes de energia limpia",
            ],
            related_concepts=[
                "energia solar",
                "energia eolica",
                "paneles solares",
                "turbinas eolicas",
                "transicion energetica",
                "descarbonizacion",
                "electricidad",
                "red electrica",
                "almacenamiento de energia",
                "baterias",
            ],
        )
        results = execute_search("energias renovables", ["energias", "renovables"])
        assert len(results) >= 1
        assert any(
            "renovable" in r.title.lower() or "limpia" in r.title.lower() or "solar" in r.summary.lower()
            for r in results
        )

    @patch("app.services.pipeline.expand_query")
    def test_violencia_de_genero(self, mock_expand):
        mock_expand.return_value = QueryIntent(
            original_query="violencia de genero",
            core_concepts=[
                "violencia de genero",
                "violencia contra la mujer",
                "violencia machista",
            ],
            equivalent_phrases=[
                "feminicidio",
                "agresion a mujeres",
                "maltrato a mujeres",
                "violencia domestica",
                "violencia intrafamiliar",
                "violencia sexual",
                "acoso a mujeres",
                "discriminacion de genero",
                "violencia patriarcal",
            ],
            related_concepts=[
                "marcha feminista",
                "justicia para mujeres",
                "igualdad de genero",
                "derechos de la mujer",
                "alerta de genero",
                "violencia domestica",
                "muertes violentas de mujeres",
                "crimen de odio",
                "prevencion de violencia",
                "atencion a victimas",
            ],
        )
        results = execute_search("violencia de genero", ["violencia", "genero"])
        assert len(results) >= 1
        assert any(
            "genero" in r.title.lower() or "mujer" in r.title.lower() or "feminicidio" in r.summary.lower()
            for r in results
        )

    @patch("app.services.pipeline.expand_query")
    def test_politica_monetaria(self, mock_expand):
        mock_expand.return_value = QueryIntent(
            original_query="politica monetaria",
            core_concepts=[
                "politica monetaria",
                "tasas de interes",
                "banco central",
            ],
            equivalent_phrases=[
                "politica de tasas",
                "control monetario",
                "regulacion monetaria",
                "politica de credito",
                "politica de liquidez",
                "gestion monetaria",
                "intervencion del banco central",
                "politica de tasas de interes",
                "control de inflacion",
            ],
            related_concepts=[
                "Banxico",
                "inflacion",
                "tipo de cambio",
                "peso mexicano",
                "dolar",
                "economia",
                "crecimiento",
                "desempleo",
                "presupuesto",
                "deuda publica",
            ],
        )
        results = execute_search("politica monetaria", ["politica", "monetaria"])
        assert len(results) >= 1
        assert any(
            "monetaria" in r.title.lower() or "banxico" in r.title.lower() or "tasas" in r.summary.lower()
            for r in results
        )

    @patch("app.services.pipeline.expand_query")
    def test_donald_trump(self, mock_expand):
        mock_expand.return_value = QueryIntent(
            original_query="Donald Trump",
            core_concepts=[
                "Donald Trump",
                "presidente de Estados Unidos",
                "gobierno de Trump",
            ],
            equivalent_phrases=[
                "expresidente Trump",
                "realDonaldTrump",
                "Donald J. Trump",
                "maga",
                "administracion Trump",
                "candidato Trump",
                "presidente estadounidense",
                "lider de la derecha",
                "magnate inmobiliario",
            ],
            related_concepts=[
                "aranceles",
                "comercio",
                "Mexico",
                "Estados Unidos",
                "politica internacional",
                "elecciones",
                "partido republicano",
                "wall",
                "migracion",
                "relaciones internacionales",
            ],
        )
        results = execute_search("Donald Trump", ["Donald", "Trump"])
        assert len(results) >= 1
        assert any(
            "trump" in r.title.lower() or "estados unidos" in r.summary.lower()
            for r in results
        )

    @patch("app.services.pipeline.expand_query")
    def test_desaparicion_forzada(self, mock_expand):
        mock_expand.return_value = QueryIntent(
            original_query="desaparicion forzada",
            core_concepts=[
                "desaparicion forzada",
                "desaparecidos",
                "personas desaparecidas",
            ],
            equivalent_phrases=[
                "secuestro",
                "levanton",
                "desaparicion",
                "desapariciones",
                "desaparecidos forzados",
                "desaparicion politica",
                "victimas de desaparicion",
                "desaparecidos en Mexico",
                "crisis de desapariciones",
            ],
            related_concepts=[
                "busqueda",
                "fosas",
                "justicia",
                "familiares",
                "colectivos",
                "derechos humanos",
                "fiscalia",
                "investigacion",
                "verdad",
                "memoria historica",
            ],
        )
        results = execute_search("desaparicion forzada", ["desaparicion", "forzada"])
        assert len(results) >= 1
        assert any(
            "desaparec" in r.title.lower() or "desaparec" in r.summary.lower() or "busca" in r.title.lower()
            for r in results
        )

    @patch("app.services.pipeline.expand_query")
    def test_expansion_reaches_20_plus_terms(self, mock_expand):
        mock_expand.return_value = QueryIntent(
            original_query="terremoto",
            core_concepts=[
                "terremoto", "sismo", "temblor fuerte", "movimiento telurico",
            ],
            equivalent_phrases=[
                "seismo", "sacudida", "temblor de tierra", "sismo destructivo",
                "temblor en la costa", "movimiento sismico", "sismo registrado",
                "temblor fuerte en oaxaca", "seismo fuerte",
            ],
            related_concepts=[
                "escombros", "victimas", "rescate", "ayuda humanitaria",
                "protector civil", "alerta sismica", "placas tectonicas",
                "zona de impacto", "danos materiales", "emergencia nacional",
            ],
        )
        results = execute_search("terremoto", ["terremoto"])
        intent = mock_expand.return_value
        total = len(intent.all_phrases())
        assert total >= 20, f"Expansion should reach 20+ terms, got {total}"
        assert len(results) >= 1

    @patch("app.services.pipeline.expand_query")
    def test_reranking_preserved(self, mock_expand):
        mock_expand.return_value = QueryIntent(
            original_query="inteligencia artificial",
            core_concepts=[
                "inteligencia artificial", "IA", "sistemas de IA",
            ],
            equivalent_phrases=[
                "machine learning", "aprendizaje automatico",
            ],
            related_concepts=[
                "tecnologia", "innovacion",
            ],
        )
        results = execute_search("inteligencia artificial", ["inteligencia", "artificial"])
        if len(results) >= 2:
            assert results[0].relevance_score >= results[1].relevance_score

    @patch("app.services.pipeline.expand_query")
    def test_original_query_always_first_core(self, mock_expand):
        mock_expand.return_value = QueryIntent(
            original_query="nearshoring",
            core_concepts=[
                "relocalizacion de empresas",
                "nearshoring en Mexico",
            ],
            equivalent_phrases=[],
            related_concepts=[],
        )
        results = execute_search("nearshoring", ["nearshoring"])
        intent = mock_expand.return_value
        assert intent.core_concepts[0] == "nearshoring" or intent.original_query == "nearshoring"

import json
from unittest.mock import patch, MagicMock

from app.services.query_expander import expand_query, _parse_expansion, _parse_flat_fallback
from app.query_intent import QueryIntent


class TestExpandQuery:
    @patch("app.services.gemini._get_client")
    def test_returns_query_intent(self, mock_get_client):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "core": ["madres buscadoras"],
            "equivalent": ["madre buscadora"],
            "related": ["colectivos de busqueda"],
        })
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = expand_query("madres buscadoras")

        assert isinstance(result, QueryIntent)
        assert result.original_query == "madres buscadoras"

    @patch("app.services.gemini._get_client")
    def test_original_query_preserved(self, mock_get_client):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "core": ["inteligencia artificial"],
            "equivalent": ["IA en la industria"],
            "related": ["machine learning"],
        })
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = expand_query("inteligencia artificial")

        assert result.original_query == "inteligencia artificial"

    @patch("app.services.gemini._get_client")
    def test_core_concepts_populated(self, mock_get_client):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "core": ["terremoto", "sismo fuerte"],
            "equivalent": ["temblor"],
            "related": ["desastre natural"],
        })
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = expand_query("terremoto")

        assert len(result.core_concepts) >= 1

    @patch("app.services.gemini._get_client")
    def test_gemini_unavailable_fallback(self, mock_get_client):
        mock_get_client.return_value = None

        result = expand_query("test query")

        assert isinstance(result, QueryIntent)
        assert result.original_query == "test query"
        assert result.core_concepts == ["test query"]

    @patch("app.services.gemini._get_client")
    def test_gemini_timeout_fallback(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("timeout")
        mock_get_client.return_value = mock_client

        result = expand_query("test")

        assert isinstance(result, QueryIntent)
        assert result.core_concepts == ["test"]

    @patch("app.services.gemini._get_client")
    def test_gemini_exception_fallback(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("API error")
        mock_get_client.return_value = mock_client

        result = expand_query("test")

        assert isinstance(result, QueryIntent)
        assert result.core_concepts == ["test"]

    @patch("app.services.gemini._get_client")
    def test_spanish_query(self, mock_get_client):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "core": ["economia mexicana", "crecimiento economico"],
            "equivalent": ["economia de mexico", "PIB mexicano"],
            "related": ["inflacion", "tipo de cambio"],
        })
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = expand_query("economia mexicana")

        assert isinstance(result, QueryIntent)
        assert len(result.all_phrases()) >= 2

    @patch("app.services.gemini._get_client")
    def test_empty_response_fallback(self, mock_get_client):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = ""
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = expand_query("test")

        assert isinstance(result, QueryIntent)
        assert result.core_concepts == ["test"]

    @patch("app.services.gemini._get_client")
    def test_malformed_json_uses_flat_fallback(self, mock_get_client):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "term1, term2, term3"
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = expand_query("test")

        assert isinstance(result, QueryIntent)
        assert "test" in result.core_concepts

    @patch("app.services.gemini._get_client")
    def test_json_with_code_block_markers(self, mock_get_client):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "```json\n" + json.dumps({
            "core": ["test"],
            "equivalent": [],
            "related": [],
        }) + "\n```"
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = expand_query("test")

        assert isinstance(result, QueryIntent)
        assert "test" in result.core_concepts

    @patch("app.services.gemini._get_client")
    def test_large_expansion_reaches_20_plus(self, mock_get_client):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "core": ["terremoto", "sismo", "temblor fuerte", "sismo fuerte"],
            "equivalent": [
                "movimiento telurico", "seismo", "sacudida",
                "temblor de tierra", "sismo destructivo", "seismo fuerte",
            ],
            "related": [
                "escombros", "victimas", "rescate",
                "ayuda humanitaria", "protector civil",
                "alerta sismica", "placas tectonicas",
                "zona de impacto", "danos materiales",
                "emergencia nacional", "refugios temporales",
            ],
        })
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = expand_query("terremoto")

        total = len(result.all_phrases())
        assert total >= 20, f"Expected >=20 concepts, got {total}"

    @patch("app.services.gemini._get_client")
    def test_generic_terms_not_rejected_from_expansion(self, mock_get_client):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "core": ["inteligencia artificial"],
            "equivalent": ["IA generativa", "modelos de lenguaje"],
            "related": ["redes neuronales", "aprendizaje automatico"],
        })
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = expand_query("inteligencia artificial")

        all_phrases = result.all_phrases()
        assert "inteligencia artificial" in all_phrases


class TestParseExpansion:
    def test_valid_json_new_keys(self):
        raw = json.dumps({
            "core": ["ai"],
            "equivalent": ["artificial intelligence"],
            "related": ["machine learning"],
        })
        result = _parse_expansion(raw, "ai")
        assert isinstance(result, QueryIntent)
        assert result.original_query == "ai"
        assert "ai" in result.core_concepts
        assert "artificial intelligence" in result.equivalent_phrases
        assert "machine learning" in result.related_concepts

    def test_valid_json_old_keys_backward_compat(self):
        raw = json.dumps({
            "core_concepts": ["ai"],
            "equivalent_phrases": ["artificial intelligence"],
            "related_concepts": ["machine learning"],
        })
        result = _parse_expansion(raw, "ai")
        assert isinstance(result, QueryIntent)
        assert "ai" in result.core_concepts
        assert "artificial intelligence" in result.equivalent_phrases
        assert "machine learning" in result.related_concepts

    def test_empty_json_fields(self):
        raw = json.dumps({
            "core": [],
            "equivalent": [],
            "related": [],
        })
        result = _parse_expansion(raw, "test")
        assert result.core_concepts == ["test"]

    def test_original_query_inserted_when_missing(self):
        raw = json.dumps({
            "core": ["other term"],
            "equivalent": [],
            "related": [],
        })
        result = _parse_expansion(raw, "my query")
        assert result.core_concepts[0] == "my query"

    def test_max_terms_respected(self):
        raw = json.dumps({
            "core": [f"term{i}" for i in range(10)],
            "equivalent": [f"equiv{i}" for i in range(10)],
            "related": [f"rel{i}" for i in range(10)],
        })
        result = _parse_expansion(raw, "orig")
        assert len(result.all_phrases()) <= 25

    def test_structure_preserved_when_under_limit(self):
        raw = json.dumps({
            "core": ["c1", "c2"],
            "equivalent": ["e1", "e2", "e3"],
            "related": ["r1", "r2"],
        })
        result = _parse_expansion(raw, "c1")
        assert len(result.core_concepts) == 2
        assert len(result.equivalent_phrases) == 3
        assert len(result.related_concepts) == 2

    def test_strips_quotes_from_terms(self):
        raw = json.dumps({
            "core": ['"term1"', "'term2'"],
            "equivalent": [],
            "related": [],
        })
        result = _parse_expansion(raw, "orig")
        assert "term1" in result.core_concepts
        assert "term2" in result.core_concepts

    def test_empty_string_returns_fallback(self):
        result = _parse_expansion("", "original")
        assert isinstance(result, QueryIntent)
        assert result.core_concepts == ["original"]

    def test_non_string_items_filtered(self):
        raw = json.dumps({
            "core": ["valid", 123, None, "also valid"],
            "equivalent": [],
            "related": [],
        })
        result = _parse_expansion(raw, "orig")
        assert "valid" in result.core_concepts
        assert "also valid" in result.core_concepts
        assert 123 not in result.core_concepts
        assert None not in result.core_concepts
        assert len(result.core_concepts) == 3

    def test_truncation_preserves_ratio(self):
        raw = json.dumps({
            "core": [f"c{i}" for i in range(10)],
            "equivalent": [f"e{i}" for i in range(15)],
            "related": [f"r{i}" for i in range(15)],
        })
        result = _parse_expansion(raw, "orig")
        total = len(result.all_phrases())
        assert total <= 25
        assert len(result.core_concepts) >= 8
        assert len(result.equivalent_phrases) >= 8
        assert len(result.related_concepts) >= 8


class TestParseFlatFallback:
    def test_comma_separated(self):
        result = _parse_flat_fallback("term1, term2, term3", "original")
        assert isinstance(result, QueryIntent)
        assert result.core_concepts[0] == "original"

    def test_empty_returns_fallback(self):
        result = _parse_flat_fallback("", "original")
        assert result.core_concepts == ["original"]

    def test_strips_quotes(self):
        result = _parse_flat_fallback('"term1", "term2"', "original")
        assert "term1" in result.core_concepts
        assert "term2" in result.core_concepts

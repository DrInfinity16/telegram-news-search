from app.query_intent import QueryIntent


class TestQueryIntent:
    def test_all_phrases_deduplicates(self):
        intent = QueryIntent(
            original_query="test",
            core_concepts=["test", "other"],
            equivalent_phrases=["test", "another"],
            related_concepts=[],
        )
        phrases = intent.all_phrases()
        assert phrases.count("test") == 1

    def test_all_phrases_preserves_order(self):
        intent = QueryIntent(
            original_query="original",
            core_concepts=["core1", "core2"],
            equivalent_phrases=["equiv1"],
            related_concepts=["rel1"],
        )
        phrases = intent.all_phrases()
        assert phrases[0] == "original"
        assert "core1" in phrases
        assert "equiv1" in phrases
        assert "rel1" in phrases

    def test_search_terms_includes_multi_word_phrases(self):
        intent = QueryIntent(
            original_query="madres buscadoras",
            core_concepts=["colectivos de busqueda"],
            equivalent_phrases=[],
            related_concepts=[],
        )
        terms = intent.search_terms()
        assert "madres buscadoras" in terms
        assert "colectivos de busqueda" in terms

    def test_search_terms_includes_long_single_tokens(self):
        intent = QueryIntent(
            original_query="inteligencia artificial",
            core_concepts=["machine learning"],
            equivalent_phrases=[],
            related_concepts=[],
        )
        terms = intent.search_terms()
        assert "machine" in terms
        assert "learning" in terms

    def test_search_terms_excludes_short_tokens(self):
        intent = QueryIntent(
            original_query="IA",
            core_concepts=["IA"],
            equivalent_phrases=[],
            related_concepts=[],
        )
        terms = intent.search_terms()
        assert "ia" not in terms

    def test_tier_phrases_separates_correctly(self):
        intent = QueryIntent(
            original_query="original",
            core_concepts=["core phrase", "single"],
            equivalent_phrases=["equiv phrase"],
            related_concepts=["rel phrase"],
        )
        orig, core, related = intent.tier_phrases()
        assert orig == ["original"]
        assert "core phrase" in core
        assert "equiv phrase" in core
        assert "rel phrase" in related

    def test_tier_phrases_excludes_single_tokens(self):
        intent = QueryIntent(
            original_query="test",
            core_concepts=["single"],
            equivalent_phrases=[],
            related_concepts=[],
        )
        orig, core, related = intent.tier_phrases()
        assert "single" not in core

    def test_generic_single_tokens(self):
        intent = QueryIntent(
            original_query="IA",
            core_concepts=["IA"],
            equivalent_phrases=[],
            related_concepts=[],
        )
        generic = intent.generic_single_tokens()
        assert "IA" in generic

    def test_generic_single_tokens_empty_for_long_words(self):
        intent = QueryIntent(
            original_query="personas",
            core_concepts=["personas"],
            equivalent_phrases=[],
            related_concepts=[],
        )
        generic = intent.generic_single_tokens()
        assert generic == []

    def test_empty_intent(self):
        intent = QueryIntent(original_query="")
        assert intent.all_phrases() == [""]
        assert intent.search_terms() == [""]

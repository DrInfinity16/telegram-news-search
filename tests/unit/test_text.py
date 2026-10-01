from app.text import normalize_text


class TestNormalizeText:
    def test_lowercase(self):
        assert normalize_text("HELLO") == "hello"

    def test_strip_accents(self):
        assert normalize_text("economía") == "economia"

    def test_strip_accents_mixed(self):
        assert normalize_text("inteligencia artificial") == "inteligencia artificial"

    def test_strip_tilde(self):
        assert normalize_text("niños") == "ninos"

    def test_strip_dieresis(self):
        assert normalize_text("pingüino") == "pinguino"

    def test_already_normalized(self):
        assert normalize_text("hello") == "hello"

    def test_empty_string(self):
        assert normalize_text("") == ""

    def test_numbers_unchanged(self):
        assert normalize_text("123") == "123"

    def test_special_characters_unchanged(self):
        assert normalize_text("hello-world") == "hello-world"

    def test_spanish_sentence(self):
        text = "La economía mexicana creció un 3.5%"
        result = normalize_text(text)
        assert "economia" in result
        assert "3.5%" in result

    def test_uppercase_accents(self):
        assert normalize_text("ECONOMÍA") == "economia"

    def test_accented_query_matches_unaccented_text(self):
        query = normalize_text("economía")
        text = normalize_text("La economia mexicana")
        assert query in text

    def test_unaccented_query_matches_accented_text(self):
        query = normalize_text("economia")
        text = normalize_text("La economía mexicana")
        assert query in text

import pytest

from app import config


class TestConfigValidation:
    def test_missing_telegram_token_exits(self):
        monkeypatch_token = config.TELEGRAM_BOT_TOKEN
        config.TELEGRAM_BOT_TOKEN = ""
        try:
            with pytest.raises(SystemExit):
                config.validate_config()
        finally:
            config.TELEGRAM_BOT_TOKEN = monkeypatch_token

    def test_valid_token_does_not_exit(self):
        original = config.TELEGRAM_BOT_TOKEN
        config.TELEGRAM_BOT_TOKEN = "valid-token-123"
        try:
            config.validate_config()
        finally:
            config.TELEGRAM_BOT_TOKEN = original


class TestFeedSources:
    def test_sources_is_a_list(self):
        assert isinstance(config.SOURCES, list)
        assert len(config.SOURCES) >= 10

    def test_sources_have_names(self):
        for source in config.SOURCES:
            assert source.name
            assert isinstance(source.name, str)

    def test_sources_have_valid_https_urls(self):
        for source in config.SOURCES:
            assert source.url.startswith("https://"), f"{source.name} URL is not HTTPS: {source.url}"

    def test_sources_have_country(self):
        for source in config.SOURCES:
            assert source.country, f"{source.name} missing country"
            assert isinstance(source.country, str)

    def test_sources_have_category(self):
        for source in config.SOURCES:
            assert source.category, f"{source.name} missing category"
            assert isinstance(source.category, str)

    def test_mexican_sources_majority(self):
        mexican = [s for s in config.SOURCES if s.country == "Mexico"]
        assert len(mexican) >= 6, f"Expected >=6 Mexican sources, got {len(mexican)}"

    def test_international_sources_present(self):
        international = [s for s in config.SOURCES if s.country != "Mexico"]
        assert len(international) >= 3, f"Expected >=3 international sources, got {len(international)}"

    def test_source_repr(self):
        source = config.SOURCES[0]
        r = repr(source)
        assert "FeedSource" in r
        assert source.name in r
        assert source.url in r
        assert source.country in r
        assert source.category in r


class TestConstants:
    def test_max_articles_per_search(self):
        assert config.MAX_ARTICLES_PER_SEARCH == 5

    def test_max_entries_per_feed(self):
        assert config.MAX_ENTRIES_PER_FEED == 15

    def test_rss_fetch_timeout(self):
        assert config.RSS_FETCH_TIMEOUT == 10

    def test_gemini_timeout(self):
        assert config.GEMINI_TIMEOUT == 30

    def test_gemini_model(self):
        assert isinstance(config.GEMINI_MODEL, str)
        assert len(config.GEMINI_MODEL) > 0


class TestGeminiApiKey:
    def test_gemini_api_key_is_string(self):
        assert isinstance(config.GEMINI_API_KEY, str)

    def test_gemini_api_key_empty_when_not_set(self):
        original = config.GEMINI_API_KEY
        config.GEMINI_API_KEY = ""
        try:
            assert not config.GEMINI_API_KEY
        finally:
            config.GEMINI_API_KEY = original

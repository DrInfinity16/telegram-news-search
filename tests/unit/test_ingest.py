from unittest.mock import patch, MagicMock

import pytest

from app.services.ingest import ingest_feeds
from app.models import Article
from app import storage


@pytest.fixture(autouse=True)
def temp_database(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    storage.close_database()
    monkeypatch.setenv("DATABASE_PATH", db_path)
    storage.initialize_database()
    yield
    storage.close_database()
    monkeypatch.delenv("DATABASE_PATH", raising=False)


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


class TestIngestFeeds:
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
    def test_duplicate_articles_ignored(self, mock_parse):
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
    def test_normalization_is_applied(self, mock_parse):
        entry = _make_mock_entry(
            "<![CDATA[AI News]]>",
            "https://a.com/1",
            "<p>About <b>AI</b></p>",
        )
        mock_feed = MagicMock()
        mock_feed.entries = [entry]
        mock_feed.bozo = False
        mock_parse.return_value = mock_feed

        from app.config import FeedSource
        sources = [FeedSource("Test", "https://example.com/feed")]

        with patch("app.services.ingest.SOURCES", sources):
            ingest_feeds()

        articles = storage.get_all_articles()
        assert len(articles) == 1
        assert articles[0].title == "AI News"

    @patch("app.services.fetcher.feedparser.parse")
    def test_repeated_ingestion_no_duplicates(self, mock_parse):
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

    @patch("app.services.fetcher.feedparser.parse")
    def test_returns_article_count(self, mock_parse):
        entries = [_make_mock_entry("AI News", "https://a.com/1", "AI")]
        mock_feed = MagicMock()
        mock_feed.entries = entries
        mock_feed.bozo = False
        mock_parse.return_value = mock_feed

        from app.config import FeedSource
        sources = [FeedSource("Test", "https://example.com/feed")]

        with patch("app.services.ingest.SOURCES", sources):
            count = ingest_feeds()

        assert isinstance(count, int)

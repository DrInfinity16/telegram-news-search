from unittest.mock import MagicMock, patch
import time

from app.services.fetcher import fetch_feed
from app.config import FeedSource


def _mock_feedparser_return(entries=None):
    mock_feed = MagicMock()
    mock_feed.entries = entries or []
    mock_feed.bozo = False
    return mock_feed


def _make_mock_entry(title="Title", link="https://example.com/article", summary="Summary"):
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


class TestFetchFeed:
    @patch("app.services.fetcher.feedparser.parse")
    def test_successful_fetch(self, mock_parse):
        entries = [_make_mock_entry() for _ in range(3)]
        mock_parse.return_value = _mock_feedparser_return(entries)

        source = FeedSource("Test", "https://example.com/feed")
        articles = fetch_feed(source)

        assert len(articles) == 3
        assert articles[0].title == "Title"

    @patch("app.services.fetcher.feedparser.parse")
    def test_empty_feed(self, mock_parse):
        mock_parse.return_value = _mock_feedparser_return([])

        source = FeedSource("Test", "https://example.com/feed")
        articles = fetch_feed(source)

        assert articles == []

    @patch("app.services.fetcher.feedparser.parse")
    def test_bozo_feed_with_no_entries(self, mock_parse):
        mock_feed = MagicMock()
        mock_feed.entries = []
        mock_feed.bozo = True
        mock_feed.bozo_exception = Exception("bad feed")
        mock_parse.return_value = mock_feed

        source = FeedSource("Test", "https://example.com/feed")
        articles = fetch_feed(source)

        assert articles == []

    @patch("app.services.fetcher.feedparser.parse")
    def test_entry_with_missing_title_skipped(self, mock_parse):
        entry = _make_mock_entry()
        entry.title = None
        mock_parse.return_value = _mock_feedparser_return([entry])

        source = FeedSource("Test", "https://example.com/feed")
        articles = fetch_feed(source)

        assert articles == []

    @patch("app.services.fetcher.feedparser.parse")
    def test_entry_with_missing_link_skipped(self, mock_parse):
        entry = _make_mock_entry()
        entry.link = None
        mock_parse.return_value = _mock_feedparser_return([entry])

        source = FeedSource("Test", "https://example.com/feed")
        articles = fetch_feed(source)

        assert articles == []

    @patch("app.services.fetcher.feedparser.parse")
    def test_respects_max_entries(self, mock_parse):
        entries = [_make_mock_entry(title=f"Article {i}") for i in range(20)]
        mock_parse.return_value = _mock_feedparser_return(entries)

        source = FeedSource("Test", "https://example.com/feed")
        articles = fetch_feed(source)

        assert len(articles) <= 15

    @patch("app.services.fetcher.feedparser.parse")
    def test_fetch_error_returns_empty(self, mock_parse):
        mock_parse.side_effect = Exception("Network error")

        source = FeedSource("Test", "https://example.com/feed")
        articles = fetch_feed(source)

        assert articles == []

    @patch("app.services.fetcher.feedparser.parse")
    def test_mixed_valid_and_invalid_entries(self, mock_parse):
        good = _make_mock_entry(title="Good Article")
        bad = _make_mock_entry(title=None)
        mock_parse.return_value = _mock_feedparser_return([good, bad])

        source = FeedSource("Test", "https://example.com/feed")
        articles = fetch_feed(source)

        assert len(articles) == 1
        assert articles[0].title == "Good Article"

    @patch("app.services.fetcher.RSS_FETCH_TIMEOUT", 1)
    @patch("app.services.fetcher.feedparser.parse")
    def test_timeout_returns_empty(self, mock_parse):
        def slow_parse(*args, **kwargs):
            time.sleep(5)
            return _mock_feedparser_return([])

        mock_parse.side_effect = slow_parse

        source = FeedSource("Test", "https://example.com/feed")
        articles = fetch_feed(source)

        assert articles == []

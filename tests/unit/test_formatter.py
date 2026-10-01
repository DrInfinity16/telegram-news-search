from app.bot.formatter import format_articles, format_summary, split_message, _relative_time, TELEGRAM_MAX_MESSAGE_LENGTH
from app.models import Article


class TestFormatArticles:
    def test_empty_results(self):
        result = format_articles([])
        assert result == "No articles found matching your keywords."

    def test_single_article(self):
        articles = [
            Article(
                title="Test Article",
                url="https://example.com/article",
                source="TestSource",
                matched_keywords=["ai"],
            )
        ]
        result = format_articles(articles)
        assert "1. Test Article" in result
        assert "TestSource" in result
        assert "https://example.com/article" in result
        assert "ai" in result

    def test_multiple_articles(self):
        articles = [
            Article(title="A1", url="https://a.com", source="S", matched_keywords=["a"]),
            Article(title="A2", url="https://b.com", source="S", matched_keywords=["b"]),
        ]
        result = format_articles(articles)
        assert "1. A1" in result
        assert "2. A2" in result

    def test_article_without_matched_keywords(self):
        articles = [
            Article(title="A1", url="https://a.com", source="S", matched_keywords=[])
        ]
        result = format_articles(articles)
        assert "Match:" not in result

    def test_article_with_published_at(self):
        from datetime import datetime, timezone, timedelta

        now = datetime.now(timezone.utc)
        published = (now - timedelta(hours=2)).isoformat()
        articles = [
            Article(title="A1", url="https://a.com", source="S", published_at=published)
        ]
        result = format_articles(articles)
        assert "ago" in result


class TestFormatSummary:
    def test_format_summary(self):
        result = format_summary("Article Title", "Summary text here.", "https://example.com")
        assert 'Article Title' in result
        assert "Summary text here." in result
        assert "https://example.com" in result


class TestRelativeTime:
    def test_none_returns_empty(self):
        assert _relative_time(None) == ""

    def test_invalid_returns_empty(self):
        assert _relative_time("not-a-date") == ""

    def test_just_now(self):
        from datetime import datetime, timezone, timedelta

        now = datetime.now(timezone.utc)
        published = (now - timedelta(seconds=30)).isoformat()
        result = _relative_time(published)
        assert result == "just now"

    def test_minutes_ago(self):
        from datetime import datetime, timezone, timedelta

        now = datetime.now(timezone.utc)
        published = (now - timedelta(minutes=5)).isoformat()
        result = _relative_time(published)
        assert "m ago" in result

    def test_hours_ago(self):
        from datetime import datetime, timezone, timedelta

        now = datetime.now(timezone.utc)
        published = (now - timedelta(hours=3)).isoformat()
        result = _relative_time(published)
        assert "h ago" in result

    def test_days_ago(self):
        from datetime import datetime, timezone, timedelta

        now = datetime.now(timezone.utc)
        published = (now - timedelta(days=2)).isoformat()
        result = _relative_time(published)
        assert "d ago" in result


class TestSplitMessage:
    def test_short_message_not_split(self):
        msg = "Short message"
        result = split_message(msg)
        assert result == ["Short message"]

    def test_empty_message(self):
        result = split_message("")
        assert result == [""]

    def test_exact_limit(self):
        msg = "x" * TELEGRAM_MAX_MESSAGE_LENGTH
        result = split_message(msg)
        assert len(result) == 1
        assert len(result[0]) <= TELEGRAM_MAX_MESSAGE_LENGTH

    def test_over_limit_splits(self):
        lines = [f"Line {i}: {'x' * 100}" for i in range(100)]
        msg = "\n".join(lines)
        result = split_message(msg, max_length=500)
        assert len(result) > 1
        for chunk in result:
            assert len(chunk) <= 500

    def test_preserves_content(self):
        lines = [f"Line {i}" for i in range(10)]
        msg = "\n".join(lines)
        result = split_message(msg, max_length=30)
        combined = "\n".join(result)
        for line in lines:
            assert line in combined

    def test_splits_at_line_boundaries(self):
        msg = "Line1\nLine2\nLine3"
        result = split_message(msg, max_length=12)
        for chunk in result:
            assert len(chunk) <= 12

    def test_single_very_long_line(self):
        msg = "x" * 5000
        result = split_message(msg, max_length=100)
        assert len(result) > 1
        for chunk in result:
            assert len(chunk) <= 100

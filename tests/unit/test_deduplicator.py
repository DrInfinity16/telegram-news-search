from app.models import Article
from app.deduplicator import deduplicate


class TestDeduplicate:
    def test_no_duplicates(self):
        articles = [
            Article(title="A", url="https://a.com/article1", source="S1"),
            Article(title="B", url="https://a.com/article2", source="S1"),
        ]
        result = deduplicate(articles)
        assert len(result) == 2

    def test_exact_duplicates(self):
        articles = [
            Article(title="A", url="https://example.com/article", source="S1"),
            Article(title="B", url="https://example.com/article", source="S2"),
        ]
        result = deduplicate(articles)
        assert len(result) == 1
        assert result[0].title == "A"

    def test_www_vs_non_www(self):
        articles = [
            Article(title="A", url="https://www.example.com/article", source="S1"),
            Article(title="B", url="https://example.com/article", source="S2"),
        ]
        result = deduplicate(articles)
        assert len(result) == 1

    def test_trailing_slash(self):
        articles = [
            Article(title="A", url="https://example.com/article/", source="S1"),
            Article(title="B", url="https://example.com/article", source="S2"),
        ]
        result = deduplicate(articles)
        assert len(result) == 1

    def test_query_params(self):
        articles = [
            Article(title="A", url="https://example.com/article?ref=twitter", source="S1"),
            Article(title="B", url="https://example.com/article?ref=facebook", source="S2"),
        ]
        result = deduplicate(articles)
        assert len(result) == 1

    def test_fragment(self):
        articles = [
            Article(title="A", url="https://example.com/article#top", source="S1"),
            Article(title="B", url="https://example.com/article#bottom", source="S2"),
        ]
        result = deduplicate(articles)
        assert len(result) == 1

    def test_different_articles_preserved(self):
        articles = [
            Article(title="A", url="https://example.com/article1", source="S1"),
            Article(title="B", url="https://example.com/article2", source="S2"),
            Article(title="C", url="https://example.com/article3", source="S3"),
        ]
        result = deduplicate(articles)
        assert len(result) == 3

    def test_empty_list(self):
        assert deduplicate([]) == []

    def test_single_article(self):
        articles = [Article(title="A", url="https://example.com/a", source="S")]
        result = deduplicate(articles)
        assert len(result) == 1

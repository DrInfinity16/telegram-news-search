from app.models import Article


class TestArticleModel:
    def test_create_article_with_all_fields(self):
        article = Article(
            title="Test Title",
            url="https://example.com/article",
            source="TestSource",
            summary="Test summary",
            author="Author",
            published_at="2025-01-15T10:00:00+00:00",
            relevance_score=0.8,
            matched_keywords=["test"],
        )
        assert article.title == "Test Title"
        assert article.url == "https://example.com/article"
        assert article.source == "TestSource"
        assert article.summary == "Test summary"
        assert article.author == "Author"
        assert article.relevance_score == 0.8
        assert article.matched_keywords == ["test"]

    def test_create_article_with_minimal_fields(self):
        article = Article(
            title="Title",
            url="https://example.com",
            source="Source",
        )
        assert article.title == "Title"
        assert article.summary == ""
        assert article.author is None
        assert article.published_at is None
        assert article.relevance_score == 0.0
        assert article.matched_keywords == []

    def test_article_id_is_consistent(self):
        a1 = Article(title="T", url="https://example.com/article", source="S")
        a2 = Article(title="Different", url="https://example.com/article", source="Other")
        assert a1.id == a2.id

    def test_article_id_differs_for_different_urls(self):
        a1 = Article(title="T", url="https://example.com/article1", source="S")
        a2 = Article(title="T", url="https://example.com/article2", source="S")
        assert a1.id != a2.id

    def test_url_normalization_www(self):
        a1 = Article(title="T", url="https://www.example.com/article", source="S")
        a2 = Article(title="T", url="https://example.com/article", source="S")
        assert a1.id == a2.id

    def test_url_normalization_trailing_slash(self):
        a1 = Article(title="T", url="https://example.com/article/", source="S")
        a2 = Article(title="T", url="https://example.com/article", source="S")
        assert a1.id == a2.id

    def test_url_normalization_query_params(self):
        a1 = Article(title="T", url="https://example.com/article?utm_source=twitter", source="S")
        a2 = Article(title="T", url="https://example.com/article", source="S")
        assert a1.id == a2.id

    def test_url_normalization_fragment(self):
        a1 = Article(title="T", url="https://example.com/article#comments", source="S")
        a2 = Article(title="T", url="https://example.com/article", source="S")
        assert a1.id == a2.id

    def test_fetched_at_is_set(self):
        article = Article(title="T", url="https://example.com", source="S")
        assert article.fetched_at is not None
        assert "T" in article.fetched_at

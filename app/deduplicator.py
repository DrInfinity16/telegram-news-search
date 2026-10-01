from app.models import Article


def deduplicate(articles: list[Article]) -> list[Article]:
    seen: set[str] = set()
    result: list[Article] = []

    for article in articles:
        article_id = article.id
        if article_id not in seen:
            seen.add(article_id)
            result.append(article)

    return result

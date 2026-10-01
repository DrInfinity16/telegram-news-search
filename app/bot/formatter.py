from app.models import Article

TELEGRAM_MAX_MESSAGE_LENGTH = 4096


def format_articles(articles: list[Article]) -> str:
    if not articles:
        return "No articles found matching your keywords."

    lines: list[str] = []
    for i, article in enumerate(articles, 1):
        time_str = _relative_time(article.published_at)
        source_part = article.source
        if time_str:
            source_part += f" | {time_str}"

        lines.append(f"{i}. {article.title}")
        lines.append(f"   {source_part}")
        lines.append(f"   {article.url}")
        if article.matched_keywords:
            lines.append(f"   Match: {', '.join(article.matched_keywords)}")
        lines.append("")

    return "\n".join(lines).strip()


def split_message(message: str, max_length: int = TELEGRAM_MAX_MESSAGE_LENGTH) -> list[str]:
    if len(message) <= max_length:
        return [message]

    chunks: list[str] = []
    current = ""

    for line in message.split("\n"):
        if len(line) > max_length:
            if current:
                chunks.append(current)
                current = ""
            while len(line) > max_length:
                chunks.append(line[:max_length])
                line = line[max_length:]
            current = line
        else:
            candidate = f"{current}\n{line}" if current else line
            if len(candidate) > max_length:
                if current:
                    chunks.append(current)
                current = line
            else:
                current = candidate

    if current:
        chunks.append(current)

    return chunks


def format_summary(title: str, summary: str, url: str) -> str:
    return f'Summary of "{title}":\n\n{summary}\n\nSource: {url}'


def _relative_time(published_at: str | None) -> str:
    if not published_at:
        return ""

    try:
        from datetime import datetime, timezone

        dt = datetime.fromisoformat(published_at)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        delta = now - dt

        total_seconds = int(delta.total_seconds())
        if total_seconds < 60:
            return "just now"
        if total_seconds < 3600:
            minutes = total_seconds // 60
            return f"{minutes}m ago"
        if total_seconds < 86400:
            hours = total_seconds // 3600
            return f"{hours}h ago"
        days = total_seconds // 86400
        return f"{days}d ago"
    except (ValueError, TypeError):
        return ""

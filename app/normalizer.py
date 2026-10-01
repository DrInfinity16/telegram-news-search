import html as html_module
import logging
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import feedparser

from app.models import Article

logger = logging.getLogger(__name__)

_CDATA_RE = re.compile(r"<!\[CDATA\[(.*?)\]\]>", re.DOTALL)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTIPLE_WS_RE = re.compile(r"[ \t]+")
_NEWLINE_WS_RE = re.compile(r"[ \t]*\n[ \t]*")
_MULTIPLE_NEWLINES_RE = re.compile(r"\n{3,}")
_NOEMA_WRAPPER_RE = re.compile(
    r"^The post .+ appeared first on NOEMA\s*\.?$",
)


def _strip_cdata(text: str) -> str:
    match = _CDATA_RE.fullmatch(text.strip())
    if match:
        return match.group(1).strip()
    return text


def _strip_html(text: str) -> str:
    text = _HTML_TAG_RE.sub(" ", text)
    text = html_module.unescape(text)
    text = _NEWLINE_WS_RE.sub("\n", text)
    text = _MULTIPLE_WS_RE.sub(" ", text)
    text = _MULTIPLE_NEWLINES_RE.sub("\n\n", text)
    return text.strip()


def _is_noema_wrapper(text: str) -> bool:
    return bool(_NOEMA_WRAPPER_RE.match(text))


def normalize_entry(entry: feedparser.FeedParserDict, source_name: str) -> Article | None:
    title = getattr(entry, "title", None)
    link = getattr(entry, "link", None)

    if not title or not link:
        logger.debug("Skipping entry with missing title or link in %s", source_name)
        return None

    title = _strip_cdata(title.strip())

    summary = getattr(entry, "summary", "") or ""
    summary = summary.strip()
    if summary:
        summary = _strip_html(summary)
        if _is_noema_wrapper(summary):
            summary = ""

    author = getattr(entry, "author", None)
    published_at = _parse_date(entry)

    return Article(
        title=title,
        url=link.strip(),
        source=source_name,
        summary=summary,
        author=author,
        published_at=published_at,
    )


def _parse_date(entry: feedparser.FeedParserDict) -> str | None:
    for attr in ("published", "updated", "created"):
        raw = getattr(entry, attr + "_parsed", None)
        if raw:
            try:
                dt = datetime(*raw[:6], tzinfo=timezone.utc)
                return dt.isoformat()
            except Exception:
                pass
        raw_str = getattr(entry, attr, None)
        if raw_str:
            try:
                dt = parsedate_to_datetime(raw_str)
                return dt.isoformat()
            except Exception:
                pass
    return None

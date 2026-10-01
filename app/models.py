from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class Article:
    title: str
    url: str
    source: str
    summary: str = ""
    author: str | None = None
    published_at: str | None = None
    fetched_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    relevance_score: float = 0.0
    matched_keywords: list[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        normalized = self._normalize_url(self.url)
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    @staticmethod
    def _normalize_url(url: str) -> str:
        from urllib.parse import urlparse, urlunparse

        parsed = urlparse(url)
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        path = parsed.path.rstrip("/")
        return urlunparse((scheme, netloc, path, "", "", ""))

import os
import sys
import logging

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


class FeedSource:
    def __init__(self, name: str, url: str, country: str = "", category: str = ""):
        self.name = name
        self.url = url
        self.country = country
        self.category = category

    def __repr__(self) -> str:
        return (
            f"FeedSource(name={self.name!r}, url={self.url!r}, "
            f"country={self.country!r}, category={self.category!r})"
        )


SOURCES: list[FeedSource] = [
    FeedSource("Excelsior", "https://www.excelsior.com.mx/rss/nacion", "Mexico", "general"),
    FeedSource("La Jornada", "https://www.jornada.com.mx/rss/edicion.xml", "Mexico", "general"),
    FeedSource("El Financiero", "https://www.elfinanciero.com.mx/arc/outboundfeeds/rss/?outputType=xml", "Mexico", "business"),
    FeedSource("Contralinea", "https://www.contralinea.com.mx/feed/", "Mexico", "investigative"),
    FeedSource("Tercera Via", "https://www.terceravia.mx/feed/", "Mexico", "opinion"),
    FeedSource("Mexico News Daily", "https://mexiconewsdaily.com/feed/", "Mexico", "general"),
    FeedSource("El País", "https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/portada", "Spain", "general"),
    FeedSource("BBC Mundo", "https://feeds.bbci.co.uk/mundo/rss.xml", "UK", "world"),
    FeedSource("BBC World", "https://feeds.bbci.co.uk/news/world/rss.xml", "UK", "world"),
    FeedSource("The Guardian World", "https://www.theguardian.com/world/rss", "UK", "world"),
    FeedSource("Noema", "https://www.noemamag.com/feed/", "International", "science"),
]

MAX_ARTICLES_PER_SEARCH = 5
MAX_ENTRIES_PER_FEED = 15
RSS_FETCH_TIMEOUT = 10
GEMINI_TIMEOUT = 30
GEMINI_MODEL = "gemini-3.1-flash-lite"

DATABASE_PATH = os.getenv("DATABASE_PATH", "news.db")
INGEST_INTERVAL_MINUTES = int(os.getenv("INGEST_INTERVAL_MINUTES", "30"))
EXPANSION_MODEL = os.getenv("EXPANSION_MODEL", GEMINI_MODEL)
MAX_EXPANDED_TERMS = int(os.getenv("MAX_EXPANDED_TERMS", "25"))
SEARCH_CANDIDATE_LIMIT = int(os.getenv("SEARCH_CANDIDATE_LIMIT", "50"))
GENERIC_TOKEN_MIN_CHARS = int(os.getenv("GENERIC_TOKEN_MIN_CHARS", "4"))


def validate_config() -> None:
    if not TELEGRAM_BOT_TOKEN:
        logger.critical("Missing required environment variable: TELEGRAM_BOT_TOKEN")
        sys.exit("TELEGRAM_BOT_TOKEN is required. Set it in .env or environment.")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.WARNING)

import logging
import time

from app.config import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_TIMEOUT

logger = logging.getLogger(__name__)

_client = None


def _get_client():
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            return None
        try:
            from google import genai
            from google.genai import types

            _client = genai.Client(
                api_key=GEMINI_API_KEY,
                http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT * 1000),
            )
        except Exception as e:
            logger.error("Failed to initialize Gemini client: %s", e)
            _client = False
    if _client is False:
        return None
    return _client


def is_available() -> bool:
    return _get_client() is not None


def summarize(title: str, content: str) -> str:
    client = _get_client()
    if client is None:
        return "AI summarization is not configured. Set GEMINI_API_KEY to enable it."

    truncated = content[:2000]
    prompt = f"Resume este articulo en 2-3 parrafos: {title}. {truncated}"

    max_retries = 1
    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            return response.text or "No summary generated."
        except Exception as e:
            logger.warning("Gemini API error (attempt %d/%d): %s", attempt + 1, max_retries + 1, e)
            if attempt < max_retries:
                time.sleep(2)

    return "Summary unavailable due to a service error. Please try again later."

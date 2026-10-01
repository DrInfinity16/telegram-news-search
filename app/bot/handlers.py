import logging
import re
import time

from telegram import Update
from telegram.ext import ContextTypes

from app.services.pipeline import execute_search
from app.services import gemini
from app.bot.formatter import format_articles, format_summary, split_message
from app.models import Article

logger = logging.getLogger(__name__)

_last_search: dict[int, list[Article]] = {}
_last_command_time: dict[int, float] = {}

RATE_LIMIT_SECONDS = 5
SUMMARY_RATE_LIMIT_SECONDS = 60


def _check_rate_limit(chat_id: int, limit: float) -> bool:
    now = time.time()
    last = _last_command_time.get(chat_id, 0)
    if now - last < limit:
        return False
    _last_command_time[chat_id] = now
    return True


async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Welcome to the News Bot!\n\n"
        "Commands:\n"
        "/search keyword1, keyword2 — Search news (1-5 keywords)\n"
        "/summary N — Summarize article N from last search\n"
        "/help — Show this message"
    )


async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Usage:\n\n"
        "/search inteligencia artificial, machine learning\n"
        "  Search news articles with AI-powered query expansion.\n"
        "  Provide 1 to 5 comma-separated keywords.\n\n"
        "/summary 3\n"
        "  Get an AI summary of article #3 from your last search."
    )


async def search_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id

    if not _check_rate_limit(chat_id, RATE_LIMIT_SECONDS):
        await update.message.reply_text("Please wait a few seconds before searching again.")
        return

    raw_input = " ".join(context.args) if context.args else ""

    if not raw_input.strip():
        await update.message.reply_text(
            "Please provide keywords.\n"
            "Example: /search inteligencia artificial, machine learning"
        )
        return

    keywords = [kw.strip().lower() for kw in re.split(r"[,;]", raw_input) if kw.strip()]

    if len(keywords) > 5:
        await update.message.reply_text("Maximum 5 keywords. Please try again with fewer keywords.")
        return

    if any(len(kw) > 50 for kw in keywords):
        await update.message.reply_text("Each keyword must be 50 characters or fewer.")
        return

    await update.message.reply_text("Searching...")

    try:
        raw_query = " ".join(keywords)
        results = execute_search(raw_query, keywords)
    except Exception as e:
        logger.error("Search pipeline error: %s", e)
        await update.message.reply_text("An error occurred while searching. Please try again.")
        return

    _last_search[chat_id] = results

    message = format_articles(results)
    for chunk in split_message(message):
        await update.message.reply_text(chunk)


async def summary_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id

    if not _check_rate_limit(chat_id, SUMMARY_RATE_LIMIT_SECONDS):
        await update.message.reply_text("Please wait before requesting another summary.")
        return

    if not context.args:
        await update.message.reply_text(
            "Please provide an article number.\n"
            "Example: /summary 3"
        )
        return

    try:
        index = int(context.args[0])
    except ValueError:
        await update.message.reply_text("Please provide a valid number.")
        return

    results = _last_search.get(chat_id, [])
    if not results:
        await update.message.reply_text("No search results. Use /search first.")
        return

    if index < 1 or index > len(results):
        await update.message.reply_text(f"Invalid number. Choose between 1 and {len(results)}.")
        return

    article = results[index - 1]

    if not gemini.is_available():
        await update.message.reply_text("AI summarization is not configured. Set GEMINI_API_KEY to enable it.")
        return

    await update.message.reply_text("Generating summary...")

    try:
        summary_text = gemini.summarize(article.title, article.summary)
    except Exception as e:
        logger.error("Gemini summarization error: %s", e)
        await update.message.reply_text("Summary unavailable due to a service error.")
        return

    message = format_summary(article.title, summary_text, article.url)
    await update.message.reply_text(message)

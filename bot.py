import logging

from telegram.ext import ApplicationBuilder, CommandHandler

from app.config import TELEGRAM_BOT_TOKEN, INGEST_INTERVAL_MINUTES, setup_logging, validate_config
from app.bot.handlers import start_handler, help_handler, search_handler, summary_handler
from app.services.ingest import ingest_feeds
from app.storage import initialize_database

logger = logging.getLogger(__name__)


async def periodic_ingest(context) -> None:
    try:
        ingest_feeds()
    except Exception as e:
        logger.error("Periodic ingestion error: %s", e)


def main() -> None:
    setup_logging()
    validate_config()

    logger.info("Starting News Bot...")
    initialize_database()

    try:
        ingest_feeds()
    except Exception as e:
        logger.error("Initial ingestion failed: %s — bot will start with existing data", e)

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("help", help_handler))
    app.add_handler(CommandHandler("search", search_handler))
    app.add_handler(CommandHandler("summary", summary_handler))

    if app.job_queue is not None:
        app.job_queue.run_repeating(
            periodic_ingest,
            interval=INGEST_INTERVAL_MINUTES * 60,
            first=INGEST_INTERVAL_MINUTES * 60,
        )
        logger.info("Periodic ingestion scheduled every %d minutes", INGEST_INTERVAL_MINUTES)
    else:
        logger.warning("JobQueue not available; periodic ingestion disabled")

    logger.info("Bot is running. Press Ctrl+C to stop.")
    app.run_polling()


if __name__ == "__main__":
    main()

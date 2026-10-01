import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.bot import handlers
from app.bot.handlers import (
    start_handler,
    help_handler,
    search_handler,
    summary_handler,
    _last_search,
    _last_command_time,
)
from app.models import Article

_chat_counter = 0


def _make_update(text="", args=None):
    global _chat_counter
    _chat_counter += 1
    update = MagicMock()
    update.message = AsyncMock()
    update.message.text = text
    update.effective_chat.id = 100000 + _chat_counter
    return update


def _make_context(args=None):
    context = MagicMock()
    context.args = args or []
    return context


@pytest.fixture(autouse=True)
def clean_state():
    _last_search.clear()
    _last_command_time.clear()
    yield
    _last_search.clear()
    _last_command_time.clear()


class TestStartHandler:
    @pytest.mark.asyncio
    async def test_start_returns_welcome(self):
        update = _make_update()
        context = _make_context()
        await start_handler(update, context)
        update.message.reply_text.assert_called_once()
        text = update.message.reply_text.call_args[0][0]
        assert "Welcome" in text
        assert "/search" in text
        assert "/summary" in text


class TestHelpHandler:
    @pytest.mark.asyncio
    async def test_help_returns_usage(self):
        update = _make_update()
        context = _make_context()
        await help_handler(update, context)
        update.message.reply_text.assert_called_once()
        text = update.message.reply_text.call_args[0][0]
        assert "/search" in text
        assert "/summary" in text


class TestSearchHandler:
    @pytest.mark.asyncio
    async def test_no_keywords_shows_error(self):
        update = _make_update()
        context = _make_context(args=[])
        await search_handler(update, context)
        update.message.reply_text.assert_called_once()
        text = update.message.reply_text.call_args[0][0]
        assert "keywords" in text.lower()

    @pytest.mark.asyncio
    async def test_too_many_keywords_shows_error(self):
        update = _make_update()
        context = _make_context(args=["a, b, c, d, e, f"])
        await search_handler(update, context)
        text = update.message.reply_text.call_args[0][0]
        assert "5" in text

    @pytest.mark.asyncio
    @patch("app.bot.handlers.execute_search")
    async def test_search_calls_pipeline(self, mock_pipeline):
        mock_pipeline.return_value = [
            Article(title="AI News", url="https://example.com", source="Test")
        ]
        update = _make_update()
        context = _make_context(args=["ai"])
        await search_handler(update, context)
        mock_pipeline.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.bot.handlers.execute_search")
    async def test_search_stores_results(self, mock_pipeline):
        articles = [Article(title="AI News", url="https://example.com", source="Test")]
        mock_pipeline.return_value = articles
        update = _make_update()
        context = _make_context(args=["ai"])
        await search_handler(update, context)
        chat_id = update.effective_chat.id
        assert chat_id in _last_search

    @pytest.mark.asyncio
    @patch("app.bot.handlers.execute_search")
    async def test_search_sends_searching_then_results(self, mock_pipeline):
        mock_pipeline.return_value = [
            Article(title="AI News", url="https://example.com", source="Test")
        ]
        update = _make_update()
        context = _make_context(args=["ai"])
        await search_handler(update, context)
        calls = update.message.reply_text.call_args_list
        assert calls[0][0][0] == "Searching..."
        assert "AI News" in calls[1][0][0]

    @pytest.mark.asyncio
    @patch("app.bot.handlers.execute_search")
    async def test_search_handles_pipeline_error(self, mock_pipeline):
        mock_pipeline.side_effect = Exception("Pipeline error")
        update = _make_update()
        context = _make_context(args=["ai"])
        await search_handler(update, context)
        calls = update.message.reply_text.call_args_list
        assert calls[0][0][0] == "Searching..."
        last_call_text = calls[-1][0][0]
        assert "error" in last_call_text.lower()


class TestSummaryHandler:
    @pytest.mark.asyncio
    async def test_no_args_shows_error(self):
        update = _make_update()
        context = _make_context(args=[])
        await summary_handler(update, context)
        text = update.message.reply_text.call_args[0][0]
        assert "number" in text.lower()

    @pytest.mark.asyncio
    async def test_invalid_number_shows_error(self):
        update = _make_update()
        context = _make_context(args=["abc"])
        await summary_handler(update, context)
        text = update.message.reply_text.call_args[0][0]
        assert "valid" in text.lower()

    @pytest.mark.asyncio
    async def test_no_previous_search_shows_error(self):
        update = _make_update()
        context = _make_context(args=["1"])
        await summary_handler(update, context)
        text = update.message.reply_text.call_args[0][0]
        assert "search" in text.lower()

    @pytest.mark.asyncio
    async def test_out_of_range_shows_error(self):
        update = _make_update()
        chat_id = update.effective_chat.id
        _last_search[chat_id] = [Article(title="A", url="https://a.com", source="S")]
        context = _make_context(args=["5"])
        await summary_handler(update, context)
        text = update.message.reply_text.call_args_list[-1][0][0]
        assert "invalid" in text.lower() or "1" in text

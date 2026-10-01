import os

import pytest

from app import storage


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Ensure clean environment for each test."""
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)


@pytest.fixture(autouse=True)
def isolated_database(tmp_path, monkeypatch):
    """Point every test at a temporary SQLite database.

    Tests must never write to the production DATABASE_PATH configured in
    the environment. This fixture makes isolation automatic.
    """
    db_path = str(tmp_path / "test_news.db")
    monkeypatch.setenv("DATABASE_PATH", db_path)
    storage.close_database()
    storage.initialize_database()
    yield db_path
    storage.close_database()
    monkeypatch.delenv("DATABASE_PATH", raising=False)

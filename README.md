# Telegram News Search Bot

A Telegram bot that ingests Mexican-majority RSS news feeds into SQLite and answers searches with **LLM-assisted query understanding**, lexical retrieval, evidence filtering, concept coverage, and deterministic ranking. Gemini is optional: when it is unavailable, search falls back to original-query lexical retrieval.

## What it does

From Telegram you can:

- Aggregate news from 11 RSS sources (Mexico-majority, plus international outlets)
- Search articles with `/search`
- Request an AI summary of a result with `/summary`
- Keep a local article corpus that refreshes periodically via JobQueue

Search is **not** embedding/vector search. It is structured query expansion + lexical matching + precision filters + deterministic scoring.

## Key Features

- Telegram interface (`/start`, `/help`, `/search`, `/summary`)
- RSS aggregation with per-feed error isolation
- SQLite persistence with accent-normalized search columns
- Duplicate prevention by normalized URL
- Periodic ingestion (python-telegram-bot JobQueue)
- Gemini-assisted structured query expansion (`QueryIntent`)
- Lexical candidate retrieval (phrase + informative tokens)
- Stopword / weak-evidence filtering
- Concept-coverage filtering (multi-part intent support)
- Deterministic multi-signal ranking
- Graceful fallback when Gemini is unavailable
- Article summaries via Gemini when configured
- Automated test suite with temporary-database isolation

## Architecture

```
RSS sources (11)
      ↓
Fetch / Normalize
      ↓
Deduplicate (URL)
      ↓
SQLite (title_norm / summary_norm)
      ↓
Periodic refresh (JobQueue)

Telegram /search
      ↓
Gemini QueryIntent  (or fallback: original query)
      ↓
Lexical Retrieval
      ↓
Evidence Filter
      ↓
Concept Coverage
      ↓
Deterministic Ranking
      ↓
Top results (default 5)

Telegram /summary
      ↓
Gemini Summary  (or clear fallback message)
```

**Fallback:** Gemini unavailable → original-query deterministic retrieval still works; `/summary` explains that AI summarization is not configured.

## Search Design

1. **QueryIntent** — Gemini interprets the user query into:
   - original query
   - **core** concepts
   - **equivalent** phrases (synonyms / paraphrases)
   - **related** concepts (context, not synonyms)
2. **Normalization** — accents/case folded for SQL and matching (`economía` ≈ `economia`).
3. **Lexical retrieval** — informative phrases and tokens against normalized columns; short tokens use boundary-aware matching.
4. **Evidence filtering** — stopwords never create relevance; weak contextual tokens may support but not create it; minimum thematic evidence is required.
5. **Concept coverage** — multi-concept queries must support multiple intent parts (e.g. `guerra Rusia Ucrania` needs more than one vague token). Single-concept queries need only one component. Full original phrases act as an override.
6. **Ranking** — deterministic multi-signal score (original/core/equiv/related, multi-word support, recency) with coverage scaling and weak/ambiguous penalties.

## Tech Stack

- Python (tested on 3.14; requires 3.10+)
- [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot) `[job-queue]`
- feedparser
- SQLite (stdlib)
- Google Gemini (`google-genai`)
- pytest / pytest-cov / pytest-asyncio

## Setup

```bash
git clone <repo-url>
cd <repo-directory>

python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt

copy .env.example .env
# Windows
# notepad .env
# macOS / Linux
# cp .env.example .env && $EDITOR .env
```

Edit `.env` and set at least `TELEGRAM_BOT_TOKEN`. Set `GEMINI_API_KEY` if you want query expansion and summaries.

The public repository does **not** ship a database. On first run the app creates/migrates `news.db` (or `DATABASE_PATH`) automatically.

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `TELEGRAM_BOT_TOKEN` | Yes | — | Bot token from [@BotFather](https://t.me/BotFather) |
| `GEMINI_API_KEY` | No | — | Gemini API key for query expansion and summaries |
| `DATABASE_PATH` | No | `news.db` | SQLite database path |
| `INGEST_INTERVAL_MINUTES` | No | `30` | RSS refresh interval |
| `EXPANSION_MODEL` | No | `gemini-3.1-flash-lite` | Gemini model for query expansion |
| `MAX_EXPANDED_TERMS` | No | `25` | Max concepts per expansion |
| `SEARCH_CANDIDATE_LIMIT` | No | `50` | Max lexical candidates before ranking |

Never commit a real `.env`. It is gitignored.

## Running

```bash
python bot.py
```

On startup the bot:

1. Initializes/migrates SQLite
2. Ingests configured RSS feeds
3. Starts Telegram polling
4. Schedules periodic ingestion

`Ctrl+C` stops the bot.

## Telegram Commands

| Command | Description |
|---|---|
| `/start` | Welcome + command list |
| `/help` | Usage help |
| `/search keyword1, keyword2` | Search with optional Gemini expansion (1–5 keywords, max 50 chars each) |
| `/summary N` | AI summary of result N from your last search |

### Example

```
User: /search madres buscadoras
Bot:  Searching...
Bot:  1. Madres buscadoras encuentran restos en Tamaulipas
      Excelsior | 2h ago
      https://excelsior.com.mx/...
      Match: madres, buscadoras

User: /summary 1
Bot:  Summary of "Madres buscadoras encuentran restos...":
      ...
```

## Tests

```bash
pytest
# with coverage
pytest --cov=app --cov-report=term-missing
```

Tests automatically use a temporary SQLite database and never write to production `news.db`.

At publication time: **368 automated tests**, approximately **85% statement coverage**.

Optional development DB maintenance (never automatic):

```bash
python -m app.maintenance --dry-run
python -m app.maintenance --apply
```

## RSS Sources

| Source | Region |
|---|---|
| Excelsior | Mexico |
| La Jornada | Mexico |
| El Financiero | Mexico |
| Contralinea | Mexico |
| Tercera Vía | Mexico |
| Mexico News Daily | Mexico |
| El País | Spain |
| BBC Mundo | UK |
| BBC World | UK |
| The Guardian World | UK |
| Noema | International |

Sources are hard-coded in `app/config.py` (not user-supplied URLs).

## Project Structure

```
proj10/
├── app/
│   ├── bot/                 Telegram handlers + formatter
│   ├── services/            fetch, ingest, pipeline, Gemini, query expansion
│   ├── config.py            env, feeds, constants
│   ├── evidence.py          stopwords, evidence, concept coverage
│   ├── ranker.py            deterministic scoring
│   ├── storage.py           SQLite + tiered retrieval
│   ├── query_intent.py      structured intent model
│   ├── text.py              accent/normalization helpers
│   ├── models.py            Article dataclass
│   ├── normalizer.py        RSS entry → Article
│   ├── deduplicator.py      URL dedup
│   └── maintenance.py       optional synthetic-row cleanup CLI
├── tests/
│   ├── unit/
│   ├── integration/
│   └── regression/
├── bot.py                   entry point
├── requirements.txt
├── .env.example
├── pytest.ini
└── LICENSE
```

## Project Scope / Limitations

- Corpus is limited to configured RSS feeds and the recent entries those feeds expose
- Gemini availability affects expansion quality and summaries; fallback remains available
- Retrieval is lexical/LLM-assisted, **not** embedding-based vector search
- Search state (`/summary` context) is in-memory per process
- Ranking is deterministic for a fixed corpus and intent; no personalization

## Future Improvements

- Morphology/stemming for Spanish
- Gemini retry/backoff and response caching
- Persistent per-user search history
- Optional hybrid embedding retrieval if the corpus grows

## License

MIT — see [LICENSE](LICENSE).

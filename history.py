"""Query-response history storage using SQLite.

Automatically records every /query call and its agent responses
for later review and evaluation.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import aiosqlite

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS queries (
    query_id    TEXT PRIMARY KEY,
    query_text  TEXT NOT NULL,
    format      TEXT NOT NULL,
    agent_ids   TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS responses (
    response_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id      TEXT NOT NULL REFERENCES queries(query_id),
    agent_id      TEXT NOT NULL,
    agent_name    TEXT NOT NULL,
    response_text TEXT,
    status        TEXT NOT NULL,
    error_message TEXT,
    model         TEXT NOT NULL,
    latency_ms    INTEGER,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    cached_tokens INTEGER,
    created_at    TEXT NOT NULL,
    UNIQUE(query_id, agent_id)
);

CREATE TABLE IF NOT EXISTS evaluations (
    eval_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    response_id INTEGER NOT NULL REFERENCES responses(response_id),
    pass_fail   INTEGER,
    score       REAL,
    notes       TEXT,
    evaluator   TEXT,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_responses_query_id ON responses(query_id);
CREATE INDEX IF NOT EXISTS idx_responses_agent_id ON responses(agent_id);
CREATE INDEX IF NOT EXISTS idx_responses_created_at ON responses(created_at);
CREATE INDEX IF NOT EXISTS idx_evaluations_response_id ON evaluations(response_id);
"""


async def init_db(path: str = "history.db") -> aiosqlite.Connection:
    db = await aiosqlite.connect(path)
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA synchronous=NORMAL")
    await db.execute("PRAGMA busy_timeout=5000")
    await db.executescript(SCHEMA)
    await db.commit()
    log.info(f"History DB ready: {path}")
    return db


async def save_query(
    db: aiosqlite.Connection,
    query_id: str,
    query_text: str,
    fmt: str,
    agent_ids: list[str] | None,
) -> None:
    await db.execute(
        "INSERT INTO queries (query_id, query_text, format, agent_ids, created_at) VALUES (?, ?, ?, ?, ?)",
        (query_id, query_text, fmt, json.dumps(agent_ids) if agent_ids else None, _now()),
    )
    await db.commit()


async def save_responses(
    db: aiosqlite.Connection,
    query_id: str,
    results: list[dict],
    model: str,
) -> None:
    await db.executemany(
        """INSERT OR IGNORE INTO responses
           (query_id, agent_id, agent_name, response_text, status, error_message,
            model, latency_ms, input_tokens, output_tokens, cached_tokens, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                query_id,
                r["agent_id"],
                r["name"],
                r.get("response"),
                r["status"],
                r.get("error"),
                model,
                r.get("latency_ms"),
                r.get("input_tokens"),
                r.get("output_tokens"),
                r.get("cached_tokens"),
                _now(),
            )
            for r in results
        ],
    )
    await db.commit()
    log.info(f"Saved {len(results)} responses for query {query_id[:8]}...")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

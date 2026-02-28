"""Per-session stats tracking and SQLite persistence."""

import dataclasses
import hashlib
import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional

log = logging.getLogger("buster-ripper")

# session_id → SessionStats (in-memory, loaded from DB on startup)
_stats: dict[str, "SessionStats"] = {}
_db_conn: Optional[sqlite3.Connection] = None


@dataclasses.dataclass
class SessionStats:
    """Per-session tracking: latest input token count and EMA of TTFT latency."""
    input_tokens: int = 0
    avg_latency_ms: float = 0.0
    request_count: int = 0

    def update(self, input_tokens: int, latency_ms: float, ema_alpha: float = 0.3) -> None:
        """Update stats with a new observation. Uses EMA for latency smoothing."""
        self.input_tokens = input_tokens
        self.request_count += 1
        if self.request_count == 1:
            self.avg_latency_ms = latency_ms
        else:
            self.avg_latency_ms = (1 - ema_alpha) * self.avg_latency_ms + ema_alpha * latency_ms


def db_init(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS session_stats (
            session_id     TEXT PRIMARY KEY,
            input_tokens   INTEGER NOT NULL DEFAULT 0,
            avg_latency_ms REAL    NOT NULL DEFAULT 0.0,
            request_count  INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.commit()
    return conn


def db_load(conn: sqlite3.Connection) -> dict[str, SessionStats]:
    cur = conn.execute(
        "SELECT session_id, input_tokens, avg_latency_ms, request_count FROM session_stats"
    )
    return {
        row[0]: SessionStats(input_tokens=row[1], avg_latency_ms=row[2], request_count=row[3])
        for row in cur.fetchall()
    }


def db_upsert(conn: sqlite3.Connection, sid: str, stats: SessionStats) -> None:
    conn.execute(
        """
        INSERT INTO session_stats (session_id, input_tokens, avg_latency_ms, request_count)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            input_tokens   = excluded.input_tokens,
            avg_latency_ms = excluded.avg_latency_ms,
            request_count  = excluded.request_count
        """,
        (sid, stats.input_tokens, stats.avg_latency_ms, stats.request_count),
    )
    conn.commit()


def update_session_stats(sid: str, input_tokens: int, latency_ms: float) -> None:
    """Update in-memory stats and persist to SQLite if configured."""
    stats = _stats.get(sid)
    if stats is None:
        stats = SessionStats()
        _stats[sid] = stats
    stats.update(input_tokens, latency_ms)
    if _db_conn is not None:
        try:
            db_upsert(_db_conn, sid, stats)
        except Exception:
            log.warning("db upsert failed for session %s", sid)


def estimate_tokens(body: dict) -> int:
    """Rough token estimate from request body text (4 chars ≈ 1 token)."""
    return max(1, len(json.dumps(body, ensure_ascii=False)) // 4)


def session_id(body: dict) -> str:
    """Stable session ID from the system prompt content."""
    system = body.get("system", "")
    if isinstance(system, list):
        system = " ".join(b.get("text", "") for b in system if isinstance(b, dict))
    return hashlib.sha256(system.encode()).hexdigest()[:12]

#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fastapi",
#   "uvicorn[standard]",
#   "httpx",
#   "click",
# ]
# ///
# ruff: noqa: E501
"""
buster-ripper — Anthropic /v1/messages normalizing proxy for vLLM prefix-cache stability.

Rips out the cache busters that Claude Code injects into every request,
keeping the KV cache prefix stable across turns.

Normalizations applied before forwarding to vLLM:

  1. Sort tool definitions alphabetically by name.
     GLM-4.7's chat template injects tools at position 0 of the formatted
     prompt (before the system message). Any ordering change invalidates the
     entire KV cache. MCP servers reconnect and return tools in arbitrary
     order — sorting makes the tool block byte-identical across turns.

  2. (Optional, --strip-date) Remove the Claude Code framework-injected
     "Today's date is YYYY-MM-DD." line from user message content.
     This is appended to MEMORY.md content inside <system-reminder> blocks
     and changes daily, busting the cache at midnight. Pass --strip-date to
     strip it; omit to leave it (model may use date awareness).

  3. (Optional, --dump-dir PATH) Session-aware prompt diffing.
     Saves the full normalized request body for turn 0 of each session, then
     a unified diff vs the previous turn for all subsequent turns.  Useful for
     identifying remaining cache busters after normalization.

     Session ID = first 12 hex chars of SHA-256(system_prompt_text).
     Stable within a Claude Code session; changes between sessions.

     Output layout:
       PATH/<session_id>/turn_000.json   — full body (turn 0)
       PATH/<session_id>/turn_001.diff   — unified diff vs turn 000
       PATH/<session_id>/turn_002.diff   — unified diff vs turn 001
       ...
       PATH/<session_id>/index.txt       — one-line summary per turn

  4. /v1/messages/count_tokens — Claude Code calls this to decide when to
     auto-compact the context. vLLM returns 404, so compaction never fires.
     We synthesize the response from tracked session usage. When both token
     fill AND TTFT latency exceed thresholds, we inflate the reported count
     to the nudge value, triggering Claude Code to compact and restore speed.

Usage:
  uv run buster_ripper.py [--upstream URL] [--port PORT] [--strip-date] [--verbose]
                          [--dump-dir PATH] [--max-model-len N]
                          [--compact-token-ratio F] [--compact-latency-ms F]
                          [--stats-db PATH]

  # default: listen on 0.0.0.0:30001, forward to localhost:30000
  uv run buster_ripper.py

  # enable prompt diffing
  uv run buster_ripper.py --dump-dir ~/mllm/prompt-diffs

Point Claude Code at buster-ripper:
  ANTHROPIC_BASE_URL=http://localhost:30001 claude ...
"""

import asyncio
import click
import dataclasses
import difflib
import hashlib
import json
import logging
import re
import sqlite3
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Optional

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

# ── Config (overridden by CLI args) ──────────────────────────────────────────
UPSTREAM = "http://localhost:30000"
STRIP_DATE = False
VERBOSE = False
DUMP_DIR: Path | None = None

# eval-mode: inject chat_template_kwargs into /v1/chat/completions requests
# so GLM-4.7 returns answers in content (not reasoning) for lm-eval scoring.
EVAL_MODE = False
EVAL_MAX_TOKENS: int = 0           # 0 = no limit (let vLLM use model default)
EVAL_THINKING_BUDGET: int = 0      # 0 = no budget (unlimited thinking)

# ── Compaction policy config ──────────────────────────────────────────────────
# Max context length of the served model. count_tokens will return a nudged
# value (COMPACT_NUDGE_RATIO × MAX_MODEL_LEN) when both thresholds are exceeded,
# tricking Claude Code into triggering auto-compaction to restore speed.
MAX_MODEL_LEN: int = 200_000
COMPACT_TOKEN_RATIO: float = 0.75   # nudge when tokens > this fraction of max
COMPACT_LATENCY_MS: float = 15_000  # nudge when TTFT > this (ms)
COMPACT_NUDGE_RATIO: float = 0.95   # nudged count = this × MAX_MODEL_LEN
COMPACT_KV_RATIO: float = 0.75      # nudge when server KV cache usage > this fraction
KV_POLL_INTERVAL: int = 10          # seconds between /metrics polls
STATS_DB: Optional[Path] = None     # SQLite path; None = in-memory only

# Matches the framework-injected currentDate line, e.g.:
#   Today's date is 2026-02-21.\n
_DATE_RE = re.compile(r"Today's date is \d{4}-\d{2}-\d{2}\.\n?")

# Matches the per-request billing nonce block injected by Claude Code, e.g.:
#   x-anthropic-billing-header: cc_version=2.1.50.f15; cc_entrypoint=cli; cch=27acd;
# This changes on every request and lives at position 0 of the system prompt,
# busting the entire KV cache. Meaningless to a local model — always stripped.
_BILLING_RE = re.compile(r"x-anthropic-billing-header:[^\n]*\n?")

# Hop-by-hop headers that must not be forwarded from upstream response
_HOP_BY_HOP = frozenset(
    [
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
    ]
)

log = logging.getLogger("buster-ripper")

# ── KV cache utilization (updated by background poller) ──────────────────────

_kv_cache_usage: float = 0.0  # 0.0–1.0, EMA-smoothed server-wide value

# Matches vLLM Prometheus metric lines, e.g.:
#   vllm:kv_cache_usage_perc{engine="0",...} 0.714
_KV_METRIC_RE = re.compile(r'^vllm:kv_cache_usage_perc\b[^\n]*\s+([\d.]+)', re.MULTILINE)

# EMA alpha for KV cache smoothing. vLLM reports per-interval averages that can
# bounce (75% → 0% → 23%) when the idle interval resets the counter. α=0.3
# damps the noise while still reacting within a few poll cycles to real pressure.
_KV_EMA_ALPHA: float = 0.3


async def _poll_kv_cache() -> None:
    """Background task: poll vLLM /metrics every KV_POLL_INTERVAL seconds."""
    global _kv_cache_usage
    first = True
    while True:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{UPSTREAM}/metrics")
            if resp.status_code == 200:
                m = _KV_METRIC_RE.search(resp.text)
                if m:
                    raw = float(m.group(1))
                    if first:
                        _kv_cache_usage = raw
                        first = False
                    else:
                        _kv_cache_usage = (1 - _KV_EMA_ALPHA) * _kv_cache_usage + _KV_EMA_ALPHA * raw
                    log.debug("kv_cache_usage=%.1f%% (raw=%.1f%%)", _kv_cache_usage * 100, raw * 100)
        except Exception:
            pass  # upstream may be starting up; ignore silently
        await asyncio.sleep(KV_POLL_INTERVAL)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    task = asyncio.create_task(_poll_kv_cache())
    yield
    task.cancel()


app = FastAPI(lifespan=_lifespan)


# ── Session stats & compaction policy ────────────────────────────────────────


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


# session_id → SessionStats (in-memory, loaded from DB on startup)
_stats: dict[str, SessionStats] = {}
_db_conn: sqlite3.Connection | None = None


def _db_init(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS session_stats (
            session_id    TEXT PRIMARY KEY,
            input_tokens  INTEGER NOT NULL DEFAULT 0,
            avg_latency_ms REAL   NOT NULL DEFAULT 0.0,
            request_count INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.commit()
    return conn


def _db_load(conn: sqlite3.Connection) -> dict[str, SessionStats]:
    cur = conn.execute(
        "SELECT session_id, input_tokens, avg_latency_ms, request_count FROM session_stats"
    )
    return {
        row[0]: SessionStats(input_tokens=row[1], avg_latency_ms=row[2], request_count=row[3])
        for row in cur.fetchall()
    }


def _db_upsert(conn: sqlite3.Connection, sid: str, stats: SessionStats) -> None:
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


def _update_session_stats(sid: str, input_tokens: int, latency_ms: float) -> None:
    """Update in-memory stats and persist to SQLite if configured."""
    stats = _stats.get(sid)
    if stats is None:
        stats = SessionStats()
        _stats[sid] = stats
    stats.update(input_tokens, latency_ms)
    if _db_conn is not None:
        try:
            _db_upsert(_db_conn, sid, stats)
        except Exception:
            log.warning("db upsert failed for session %s", sid)


def _extract_input_tokens(response_body: bytes) -> int | None:
    """Extract input_tokens from a non-streaming JSON response body."""
    try:
        return json.loads(response_body).get("usage", {}).get("input_tokens")
    except Exception:
        return None


def _find_message_start(chunk: bytes) -> dict | None:
    """Scan an SSE chunk for a 'message_start' event. Returns parsed dict or None.

    The message_start event carries usage.input_tokens and is always the first
    SSE event in an Anthropic streaming response — it arrives in the first
    network chunk, making it a reliable TTFT signal.
    """
    text = chunk.decode(errors="replace")
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[6:].strip()
        if payload == "[DONE]":
            continue
        try:
            evt = json.loads(payload)
            if evt.get("type") == "message_start":
                return evt
        except Exception:
            pass
    return None


def should_nudge_compact(
    input_tokens: int,
    avg_latency_ms: float,
    max_model_len: int,
    token_ratio: float,
    latency_threshold_ms: float,
    kv_cache_usage: float = 0.0,
    kv_ratio: float = 1.0,
) -> bool:
    """Return True when compaction should be nudged.

    Two independent triggers — either is sufficient:
      1. Per-session: token fill AND latency both exceed thresholds.
         Catches sessions that are large AND slow simultaneously.
      2. Server-wide: KV cache utilization exceeds kv_ratio.
         Catches pressure from multiple concurrent sessions even if
         individual sessions haven't hit the per-session thresholds.
    """
    token_full = input_tokens >= int(max_model_len * token_ratio)
    latency_high = avg_latency_ms >= latency_threshold_ms
    per_session = token_full and latency_high
    kv_pressure = kv_cache_usage >= kv_ratio
    return per_session or kv_pressure


def compact_token_count(max_model_len: int, nudge_ratio: float) -> int:
    """Return the inflated token count that triggers Claude Code auto-compaction."""
    return int(max_model_len * nudge_ratio)


def _estimate_tokens(body: dict) -> int:
    """Rough token estimate from request body text (4 chars ≈ 1 token).

    Fallback for count_tokens when we have no tracked stats and the upstream
    also returns a non-200.
    """
    return max(1, len(json.dumps(body, ensure_ascii=False)) // 4)


# ── Normalization ─────────────────────────────────────────────────────────────


def normalize(body: dict) -> tuple[dict, list[str]]:
    """Apply cache-stabilizing normalizations. Returns (body, list of changes)."""
    changes: list[str] = []

    # 0. Strip per-request billing nonce from system prompt.
    #    Claude Code injects "x-anthropic-billing-header: ...cch=<nonce>;" as
    #    the first system block. It changes every request and lives at token
    #    position 0 — busts the entire KV cache unconditionally. Strip it.
    system = body.get("system")
    if isinstance(system, list):
        filtered = []
        for block in system:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                cleaned = _BILLING_RE.sub("", block["text"]).strip()
                if cleaned:
                    if cleaned != block["text"].strip():
                        changes.append("stripped billing-header from system block")
                    block = {**block, "text": cleaned}
                    filtered.append(block)
                else:
                    changes.append("stripped billing-header system block")
            else:
                filtered.append(block)
        body["system"] = filtered
    elif isinstance(system, str) and _BILLING_RE.search(system):
        body["system"] = _BILLING_RE.sub("", system).strip()
        changes.append("stripped billing-header from system string")

    # 1. Sort tools by name; drop empty tools array.
    #    vLLM's Anthropic endpoint only short-circuits on tools=None, not [].
    #    An empty list leaks through to the OpenAI serving layer which then
    #    can't determine tool extraction mode → ERROR log on every such request.
    if isinstance(body.get("tools"), list):
        if not body["tools"]:
            del body["tools"]
            changes.append("dropped empty tools array")
        else:
            original_names = [t.get("name", "") for t in body["tools"]]
            body["tools"] = sorted(body["tools"], key=lambda t: t.get("name", ""))
            sorted_names = [t.get("name", "") for t in body["tools"]]
            if original_names != sorted_names:
                changes.append(
                    f"tools reordered: [{', '.join(original_names[:4])}{'...' if len(original_names) > 4 else ''}]"
                    f" → [{', '.join(sorted_names[:4])}{'...' if len(sorted_names) > 4 else ''}]"
                )

    # 2. Strip cache_control fields from messages and system blocks.
    #    Claude Code uses Anthropic's prompt caching API — it attaches
    #    cache_control: {type: ephemeral} to the latest messages, then removes
    #    it from those messages on the next turn (the marker moves forward).
    #    For vLLM these fields are meaningless, but the changed JSON content
    #    modifies the hash of every affected message → cache miss for everything
    #    from that point onward. Strip them so message content is stable.
    if isinstance(body.get("system"), list):
        stripped = 0
        for block in body["system"]:
            if isinstance(block, dict) and "cache_control" in block:
                del block["cache_control"]
                stripped += 1
        if stripped:
            changes.append(f"stripped cache_control from {stripped} system block(s)")

    if isinstance(body.get("messages"), list):
        stripped = 0
        for msg in body["messages"]:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "cache_control" in block:
                        del block["cache_control"]
                        stripped += 1
            if "cache_control" in msg:
                del msg["cache_control"]
                stripped += 1
        if stripped:
            changes.append(f"stripped cache_control from {stripped} message field(s)")

    # 4. Strip currentDate injection from user message content
    if STRIP_DATE and isinstance(body.get("messages"), list):
        for msg in body["messages"]:
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, str) and _DATE_RE.search(content):
                msg["content"] = _DATE_RE.sub("", content)
                changes.append("stripped currentDate from user message")
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and isinstance(block.get("text"), str):
                        if _DATE_RE.search(block["text"]):
                            block["text"] = _DATE_RE.sub("", block["text"])
                            changes.append("stripped currentDate from user content block")

    return body, changes


# ── Session-aware prompt diffing ──────────────────────────────────────────────


@dataclasses.dataclass
class SessionState:
    turn: int = 0
    prev_text: str = ""          # rendered text of previous turn (for diffing)
    session_dir: Path = None     # type: ignore[assignment]


# session_id → SessionState
_sessions: dict[str, SessionState] = {}


def _session_id(body: dict) -> str:
    """Stable session ID from the system prompt content."""
    system = body.get("system", "")
    if isinstance(system, list):
        # Anthropic multi-block system
        system = " ".join(
            b.get("text", "") for b in system if isinstance(b, dict)
        )
    return hashlib.sha256(system.encode()).hexdigest()[:12]


def _render_body(body: dict) -> str:
    """Human-readable rendering of a request body for diffing.

    Renders as labelled sections so diffs are easy to read:
      [tools]       sorted list of tool names + full defs
      [system]      system prompt text
      [messages]    each message as role: content
    """
    parts: list[str] = []

    # Tools section — names first for a quick overview, then full defs
    tools = body.get("tools") or []
    if tools:
        names_line = "# " + ", ".join(t.get("name", "?") for t in tools)
        defs = json.dumps(tools, indent=2, ensure_ascii=False)
        parts.append(f"[tools] ({len(tools)} total)\n{names_line}\n{defs}")

    # System section
    system = body.get("system", "")
    if isinstance(system, list):
        system_text = "\n".join(
            b.get("text", "") for b in system if isinstance(b, dict)
        )
    else:
        system_text = system or ""
    if system_text:
        parts.append(f"[system]\n{system_text}")

    # Messages section — one block per message
    for i, msg in enumerate(body.get("messages") or []):
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if isinstance(content, list):
            content_text = "\n".join(
                b.get("text", json.dumps(b, ensure_ascii=False))
                if isinstance(b, dict)
                else str(b)
                for b in content
            )
        else:
            content_text = str(content)
        parts.append(f"[message {i} / {role}]\n{content_text}")

    return "\n\n" + ("\n\n" + "─" * 80 + "\n\n").join(parts) + "\n"


def _dump_turn(body: dict) -> None:
    """Save full body (turn 0) or unified diff (turn N) for this session."""
    assert DUMP_DIR is not None
    sid = _session_id(body)
    state = _sessions.get(sid)
    now = datetime.now(timezone.utc).strftime("%H:%M:%S")
    n_msgs = len(body.get("messages") or [])
    n_tools = len(body.get("tools") or [])

    if state is None:
        session_dir = DUMP_DIR / sid
        session_dir.mkdir(parents=True, exist_ok=True)
        state = SessionState(turn=0, prev_text="", session_dir=session_dir)
        _sessions[sid] = state
        log.info("dump: new session %s → %s", sid, session_dir)

    current_text = _render_body(body)
    turn = state.turn

    if turn == 0:
        # Full save
        out = state.session_dir / "turn_000.json"
        out.write_text(
            json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        size_kb = len(current_text) / 1024
        summary = f"turn 000  {now}  FULL  {n_msgs} msgs  {n_tools} tools  {size_kb:.1f} KB"
        log.info("dump: %s", summary)
    else:
        # Diff vs previous turn
        prev_lines = state.prev_text.splitlines(keepends=True)
        curr_lines = current_text.splitlines(keepends=True)
        diff_lines = list(
            difflib.unified_diff(
                prev_lines,
                curr_lines,
                fromfile=f"turn_{turn - 1:03d}",
                tofile=f"turn_{turn:03d}",
                lineterm="",
            )
        )
        out = state.session_dir / f"turn_{turn:03d}.diff"
        out.write_text("".join(diff_lines), encoding="utf-8")
        added = sum(1 for l in diff_lines if l.startswith("+") and not l.startswith("+++"))
        removed = sum(1 for l in diff_lines if l.startswith("-") and not l.startswith("---"))
        summary = (
            f"turn {turn:03d}  {now}  +{added}/-{removed} lines  "
            f"{n_msgs} msgs  {n_tools} tools"
        )
        log.info("dump: %s", summary)

    # Append to index
    index = state.session_dir / "index.txt"
    with index.open("a", encoding="utf-8") as f:
        f.write(summary + "\n")

    state.prev_text = current_text
    state.turn += 1


# ── Request forwarding ────────────────────────────────────────────────────────


def _forward_headers(request: Request) -> dict[str, str]:
    """Build headers to send upstream, dropping hop-by-hop and host."""
    return {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length", *_HOP_BY_HOP)
    }


def _response_headers(upstream_headers: httpx.Headers) -> dict[str, str]:
    """Strip hop-by-hop headers from upstream response."""
    return {
        k: v
        for k, v in upstream_headers.items()
        if k.lower() not in _HOP_BY_HOP
    }


@app.post("/v1/messages")
async def proxy_messages(request: Request) -> Response:
    body = await request.json()
    body, changes = normalize(body)

    if changes and VERBOSE:
        log.info("normalized: %s", "; ".join(changes))
    elif changes:
        log.debug("normalized: %s", "; ".join(changes))

    if DUMP_DIR is not None:
        try:
            _dump_turn(body)
        except Exception:
            log.exception("dump failed")

    sid = _session_id(body)
    headers = _forward_headers(request)
    stream = body.get("stream", False)

    if stream:
        t_start = time.monotonic()
        capture: dict = {"done": False}  # mutable state for the generator closure

        async def generate() -> AsyncIterator[bytes]:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    f"{UPSTREAM}/v1/messages",
                    json=body,
                    headers=headers,
                ) as resp:
                    async for chunk in resp.aiter_bytes():
                        # Parse TTFT + input_tokens from the first message_start event.
                        # This is always the very first SSE event, so we only need to
                        # scan until we find it.
                        if not capture["done"]:
                            evt = _find_message_start(chunk)
                            if evt is not None:
                                ttft_ms = (time.monotonic() - t_start) * 1000
                                input_tokens = (
                                    evt.get("message", {})
                                    .get("usage", {})
                                    .get("input_tokens", 0)
                                )
                                capture["done"] = True
                                _update_session_stats(sid, input_tokens, ttft_ms)
                                if VERBOSE:
                                    log.info(
                                        "session %s: tokens=%d ttft=%.0f ms",
                                        sid, input_tokens, ttft_ms,
                                    )
                        yield chunk

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    else:
        t_start = time.monotonic()
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(
                f"{UPSTREAM}/v1/messages",
                json=body,
                headers=headers,
            )
        elapsed_ms = (time.monotonic() - t_start) * 1000
        input_tokens = _extract_input_tokens(resp.content)
        if input_tokens is not None:
            _update_session_stats(sid, input_tokens, elapsed_ms)
            if VERBOSE:
                log.info(
                    "session %s: tokens=%d latency=%.0f ms",
                    sid, input_tokens, elapsed_ms,
                )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=_response_headers(resp.headers),
        )


# ── count_tokens endpoint ─────────────────────────────────────────────────────


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: Request) -> Response:
    """Respond to Claude Code's context-size probe.

    Claude Code calls POST /v1/messages/count_tokens?beta=true periodically
    to decide when to auto-compact the conversation context. vLLM returns 404,
    so without this endpoint compaction never fires and the context grows until
    TTFT becomes unbearable.

    Strategy:
      - Return the input_tokens we observed in the last response for this
        session (exact, from the model's own usage report).
      - If both token fill AND average TTFT exceed their thresholds, inflate
        the reported count to COMPACT_NUDGE_RATIO × MAX_MODEL_LEN, which is
        above Claude Code's compaction threshold and triggers a context reset.
      - If we have no stats yet (first turn), fall back to a character-count
        estimate. vLLM always returns 404 for this endpoint so forwarding
        upstream is pointless.
    """
    raw_body = await request.body()
    try:
        body = json.loads(raw_body)
    except Exception:
        body = {}

    sid = _session_id(body)
    stats = _stats.get(sid)

    if stats and stats.input_tokens > 0:
        real_tokens = stats.input_tokens
    else:
        # No stats yet (first turn) — estimate from body size.
        # vLLM always returns 404 for this endpoint so there's no point
        # forwarding upstream; just estimate and respond immediately.
        real_tokens = _estimate_tokens(body)
        log.debug("session %s: count_tokens no-stats estimate=%d", sid, real_tokens)

    # Apply compaction nudge if per-session OR server KV thresholds are exceeded
    avg_latency = stats.avg_latency_ms if stats else 0.0
    if should_nudge_compact(
        real_tokens,
        avg_latency,
        MAX_MODEL_LEN,
        COMPACT_TOKEN_RATIO,
        COMPACT_LATENCY_MS,
        kv_cache_usage=_kv_cache_usage,
        kv_ratio=COMPACT_KV_RATIO,
    ):
        reported = compact_token_count(MAX_MODEL_LEN, COMPACT_NUDGE_RATIO)
        kv_trigger = _kv_cache_usage >= COMPACT_KV_RATIO
        session_trigger = real_tokens >= int(MAX_MODEL_LEN * COMPACT_TOKEN_RATIO) and avg_latency >= COMPACT_LATENCY_MS
        trigger = "kv_pressure" if kv_trigger and not session_trigger else "session" if session_trigger and not kv_trigger else "both"
        log.info(
            "session %s: nudging compaction tokens=%d→%d avg_latency=%.0f ms kv=%.1f%% trigger=%s",
            sid, real_tokens, reported, avg_latency, _kv_cache_usage * 100, trigger,
        )
    else:
        reported = real_tokens

    return Response(
        content=json.dumps({"input_tokens": reported}).encode(),
        status_code=200,
        media_type="application/json",
    )


# ── /v1/chat/completions — eval-mode thinking injection ───────────────────────


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    """Forward chat completions to vLLM, optionally injecting eval params.

    When --eval-mode is active:
      - Sets chat_template_kwargs.enable_thinking = true so GLM-4.7 uses
        chain-of-thought reasoning for better code/math scores.
      - Drops max_gen_toks (lm-eval internal field that vLLM ignores with a warning).
      - Sets max_tokens cap if --eval-max-tokens was specified.
      - Post-processes the response: if content is null (thinking model puts the
        answer in reasoning_content), copies reasoning_content → content so
        lm-eval can score it.
    """
    body = await request.body()
    headers = _forward_headers(request)

    if EVAL_MODE and body:
        try:
            data = json.loads(body)
            kwargs = data.setdefault("chat_template_kwargs", {})
            kwargs.setdefault("enable_thinking", True)
            if EVAL_THINKING_BUDGET > 0:
                kwargs.setdefault("thinking_budget", EVAL_THINKING_BUDGET)
            data.pop("max_gen_toks", None)  # lm-eval internal field, not an OpenAI field
            if EVAL_MAX_TOKENS > 0:
                data.setdefault("max_tokens", EVAL_MAX_TOKENS)
            body = json.dumps(data).encode()
            headers["content-length"] = str(len(body))
            if VERBOSE:
                log.info("eval-mode: injected enable_thinking=true, dropped max_gen_toks")
        except Exception:
            pass  # leave body untouched on parse error

    # Strip empty/invalid Authorization headers (lm-eval sends "Bearer " with no token)
    auth = headers.get("authorization", "")
    if auth.strip() in ("Bearer", "Bearer "):
        headers.pop("authorization", None)

    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(
            f"{UPSTREAM}/v1/chat/completions",
            content=body,
            headers=headers,
        )

    # In eval-mode with thinking enabled, GLM-4.7 may return the answer in
    # reasoning_content with content=null. Copy it over so lm-eval can score it.
    if EVAL_MODE:
        try:
            resp_data = resp.json()
            fixed = False
            for choice in resp_data.get("choices", []):
                msg = choice.get("message", {})
                if not msg.get("content"):
                    thinking = msg.get("reasoning_content") or msg.get("reasoning") or ""
                    if thinking:
                        msg["content"] = thinking
                        fixed = True
            if fixed:
                resp_body = json.dumps(resp_data).encode()
                resp_headers = dict(_response_headers(resp.headers))
                resp_headers["content-length"] = str(len(resp_body))
                if VERBOSE:
                    log.info("eval-mode: copied reasoning_content → content")
                return Response(content=resp_body, status_code=resp.status_code, headers=resp_headers)
        except Exception:
            pass  # fall through to normal response on parse error

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=_response_headers(resp.headers),
    )


# ── Passthrough for all other endpoints (health, /v1/models, etc.) ────────────


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "HEAD"])
async def passthrough(request: Request, path: str) -> Response:
    body = await request.body()
    headers = _forward_headers(request)
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.request(
            method=request.method,
            url=f"{UPSTREAM}/{path}",
            content=body,
            headers=headers,
            params=dict(request.query_params),
        )
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=_response_headers(resp.headers),
    )


# ── Entrypoint ────────────────────────────────────────────────────────────────


@click.command(help=__doc__)
@click.option("--upstream", default="http://localhost:30000", show_default=True,
              help="vLLM base URL.")
@click.option("--host", default="0.0.0.0", show_default=True,
              help="Bind address.")
@click.option("--port", default=30001, show_default=True, type=int,
              help="Listen port.")
@click.option("--strip-date", is_flag=True,
              help="Strip framework-injected 'Today's date is ...' from user messages.")
@click.option("--verbose", is_flag=True,
              help="Log every normalization applied and per-turn token/latency stats.")
@click.option("--dump-dir", metavar="PATH", default=None,
              help="Enable session prompt diffing under PATH/<session_id>/.")
@click.option("--max-model-len", default=200_000, show_default=True, type=int,
              help="Max context length of the served model (used by compaction policy).")
@click.option("--compact-token-ratio", default=0.75, show_default=True, type=float,
              help="Nudge compaction when tokens exceed this fraction of --max-model-len.")
@click.option("--compact-latency-ms", default=15_000.0, show_default=True, type=float,
              help="Nudge compaction when average TTFT exceeds this value in ms.")
@click.option("--compact-kv-ratio", default=0.75, show_default=True, type=float,
              help="Nudge compaction when server KV cache usage exceeds this fraction. "
                   "Set to 1.0 to disable.")
@click.option("--stats-db", metavar="PATH", default=None,
              help="SQLite file for session stats persistence across restarts.")
@click.option("--eval-mode", is_flag=True,
              help="Eval proxy mode for lm-evaluation-harness: enables thinking, drops "
                   "max_gen_toks, and copies reasoning_content→content so lm-eval can score it.")
@click.option("--eval-max-tokens", default=0, show_default=True, type=int,
              help="max_tokens injected into chat completions when --eval-mode is active.")
@click.option("--eval-thinking-budget", default=0, show_default=True, type=int,
              help="thinking_budget injected into chat_template_kwargs when --eval-mode is active. "
                   "Caps the think block to prevent runaway reasoning loops. 0 = no budget.")
def main(
    upstream, host, port, strip_date, verbose, dump_dir,
    max_model_len, compact_token_ratio, compact_latency_ms, compact_kv_ratio,
    stats_db, eval_mode, eval_max_tokens, eval_thinking_budget,
):
    global UPSTREAM, STRIP_DATE, VERBOSE, DUMP_DIR
    global MAX_MODEL_LEN, COMPACT_TOKEN_RATIO, COMPACT_LATENCY_MS, COMPACT_KV_RATIO
    global EVAL_MODE, EVAL_MAX_TOKENS, EVAL_THINKING_BUDGET

    UPSTREAM = upstream
    STRIP_DATE = strip_date
    VERBOSE = verbose
    MAX_MODEL_LEN = max_model_len
    COMPACT_TOKEN_RATIO = compact_token_ratio
    COMPACT_LATENCY_MS = compact_latency_ms
    COMPACT_KV_RATIO = compact_kv_ratio
    EVAL_MODE = eval_mode
    EVAL_MAX_TOKENS = eval_max_tokens
    EVAL_THINKING_BUDGET = eval_thinking_budget

    if dump_dir:
        DUMP_DIR = Path(dump_dir).expanduser().resolve()
        DUMP_DIR.mkdir(parents=True, exist_ok=True)

    if stats_db:
        db_path = Path(stats_db).expanduser().resolve()
        global _db_conn
        _db_conn = _db_init(db_path)
        _stats.update(_db_load(_db_conn))
        log.info("stats db: %s (%d sessions loaded)", db_path, len(_stats))

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [buster-ripper] %(message)s",
        datefmt="%H:%M:%S",
    )

    log.info("Listening on %s:%d → upstream %s", host, port, UPSTREAM)
    log.info(
        "Tool sorting: enabled | Date stripping: %s | Dump dir: %s | Eval mode: %s",
        "enabled" if STRIP_DATE else "disabled",
        DUMP_DIR or "disabled",
        f"enabled (max_tokens={EVAL_MAX_TOKENS})" if EVAL_MODE else "disabled",
    )
    log.info(
        "Compaction policy: max_len=%d token_ratio=%.2f latency_ms=%.0f nudge_ratio=%.2f kv_ratio=%.2f",
        MAX_MODEL_LEN, COMPACT_TOKEN_RATIO, COMPACT_LATENCY_MS, COMPACT_NUDGE_RATIO, COMPACT_KV_RATIO,
    )

    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()

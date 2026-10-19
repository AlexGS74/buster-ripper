"""Session-aware prompt dumping.

``documented`` (default): per-turn files — ``turn_000.json`` (full body),
``turn_{N:03d}.diff`` (unified diff of the rendered bodies) and one summary
line per turn in ``index.txt``.

``change-focused`` / ``self-contained`` (collapsed): each turn's record is
buffered in memory and, on flush, appended in arrival order to one artifact
per session — ``DUMP_DIR/<sid>/<format>.log`` — in a single write. Flushes
fire on the volume threshold, the timer, and shutdown. Records
carry the same summary lines as the documented format; change-focused stores
turn 0's JSON body and rendering then per-turn diffs, while self-contained
stores every turn's complete rendering (turn 0 also the JSON body).
"""

import asyncio
import dataclasses
import difflib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from . import config
from .session import session_id

log = logging.getLogger("buster-ripper")

_ARTIFACT_SUFFIX = ".log"


@dataclasses.dataclass
class SessionState:
    turn: int = 0
    prev_text: str = ""
    session_dir: Path = None  # type: ignore[assignment]
    # Buffered records awaiting a bulk flush. pending_bytes tracks rendered-text
    # UTF-8 bytes of _render_body(body), the volume measure — not record bytes.
    pending: list[str] = dataclasses.field(default_factory=list)
    pending_bytes: int = 0
    # Format the pending records were built under; the flush files under it.
    fmt: str = "documented"


# session_id → SessionState
_sessions: dict[str, SessionState] = {}


def _render_tools(tools: list) -> str:
    """The [tools] section: count, one names line, then the full definitions."""
    names_line = "# " + ", ".join(t.get("name", "?") for t in tools)
    defs = json.dumps(tools, indent=2, ensure_ascii=False)
    return f"[tools] ({len(tools)} total)\n{names_line}\n{defs}"


def _system_text(system) -> str:
    """The system prompt as plain text, whether given as a string or block list."""
    if isinstance(system, list):
        return "\n".join(b.get("text", "") for b in system if isinstance(b, dict))
    return system or ""


def _content_text(content) -> str:
    """A message's content as plain text; dict blocks keep non-text payloads as JSON."""
    if isinstance(content, list):
        return "\n".join(
            b.get("text", json.dumps(b, ensure_ascii=False))
            if isinstance(b, dict) else str(b)
            for b in content
        )
    return str(content)


def _render_body(body: dict) -> str:
    """Human-readable rendering of a request body for diffing."""
    parts: list[str] = []

    tools = body.get("tools") or []
    if tools:
        parts.append(_render_tools(tools))

    system_text = _system_text(body.get("system", ""))
    if system_text:
        parts.append(f"[system]\n{system_text}")

    for i, msg in enumerate(body.get("messages") or []):
        role = msg.get("role", "?")
        content_text = _content_text(msg.get("content", ""))
        parts.append(f"[message {i} / {role}]\n{content_text}")

    return "\n\n" + ("\n\n" + "─" * 80 + "\n\n").join(parts) + "\n"


def dump_turn(body: dict) -> None:
    """Record one intercepted turn under the configured dump format."""
    if config.DUMP_DIR is None:
        raise ValueError("dump_turn requires config.DUMP_DIR to be set")
    fmt = _resolve_format()
    state = _lookup_or_create_state(body)
    state.fmt = fmt
    current_text = _render_body(body)
    if fmt == "documented":
        _write_documented(state, body, current_text)
    elif fmt == "change-focused":
        _append_record(state, _change_focused_record(state, body, current_text), current_text)
    else:
        _append_record(state, _self_contained_record(state, body, current_text), current_text)
    state.prev_text = current_text
    state.turn += 1


def flush_all() -> None:
    """Append every session's pending records to its artifact in ONE write.

    Collapsed-path drain: no-op when dumping is off or nothing is pending.
    Never raises (per-session failures are logged and the batch retained);
    never touches turn/prev_text.
    """
    if config.DUMP_DIR is None:
        return
    for state in _sessions.values():
        _flush_session(state)


async def poll_flush() -> None:
    """Flush pending records every DUMP_FLUSH_INTERVAL seconds.

    Per-iteration failures are logged and the loop continues; cancellation
    (from on_cleanup) lands at the sleep await.
    """
    while True:
        try:
            flush_all()
        except Exception:
            log.exception("dump: periodic flush failed")
        await asyncio.sleep(config.DUMP_FLUSH_INTERVAL)


def _resolve_format() -> str:
    """Return the configured dump format; unknown values are a hard error."""
    fmt = config.DUMP_FORMAT
    if fmt not in config.DUMP_FORMATS:
        raise ValueError(f"unknown dump format: {fmt!r} not in {config.DUMP_FORMATS}")
    return fmt


def _lookup_or_create_state(body: dict) -> SessionState:
    """Per-session state lookup/creation, keyed by session id."""
    sid = session_id(body)
    state = _sessions.get(sid)
    if state is not None:
        return state
    session_dir = config.DUMP_DIR / sid
    # A pre-existing symlink here would make every artifact land outside DUMP_DIR.
    if session_dir.is_symlink() or (session_dir.exists() and not session_dir.is_dir()):
        raise ValueError(f"dump: refusing symlinked or non-directory session dir {session_dir}")
    session_dir.mkdir(parents=True, exist_ok=True)
    state = SessionState(turn=0, prev_text="", session_dir=session_dir)
    _sessions[sid] = state
    log.info("dump: new session %s → %s", sid, session_dir)
    return state


def _write_documented(state: SessionState, body: dict, current_text: str) -> None:
    """Documented writer: today's per-turn files and index line, unchanged."""
    if state.turn == 0:
        summary = _write_full_turn(state, body, current_text)
    else:
        summary = _write_diff_turn(state, body, current_text)
    _append_index_line(state.session_dir, summary)


def _write_full_turn(state: SessionState, body: dict, current_text: str) -> str:
    """Write turn_000.json; return the FULL summary line."""
    out = state.session_dir / "turn_000.json"
    _require_writable(out)
    out.write_text(json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = _full_summary(body, current_text)
    log.info("dump: %s", summary)
    return summary


def _write_diff_turn(state: SessionState, body: dict, current_text: str) -> str:
    """Write turn_{N:03d}.diff; return the +added/-removed summary line."""
    turn = state.turn
    diff_lines = _unified_diff(turn, state.prev_text, current_text)
    out = state.session_dir / f"turn_{turn:03d}.diff"
    _require_writable(out)
    out.write_text("".join(diff_lines), encoding="utf-8")
    summary = _diff_summary(turn, _diff_counts(diff_lines), body)
    log.info("dump: %s", summary)
    return summary


def _append_index_line(session_dir: Path, summary: str) -> None:
    """Append one summary line to the session's index.txt."""
    index = session_dir / "index.txt"
    _require_writable(index)
    with index.open("a", encoding="utf-8") as f:
        f.write(summary + "\n")


def _append_record(state: SessionState, record: str, current_text: str) -> None:
    """Collapsed path: buffer the record; flush when the volume threshold is reached."""
    rendered_bytes = len(current_text.encode("utf-8"))
    state.pending.append(record)
    state.pending_bytes += rendered_bytes
    if state.pending_bytes >= config.DUMP_FLUSH_BYTES:
        flush_all()


def _change_focused_record(state: SessionState, body: dict, current_text: str) -> str:
    """Change-focused record: turn 0's body and rendering, later turns' diffs."""
    turn = state.turn
    if turn == 0:
        return _turn_zero_block(body, current_text)
    diff_lines = _unified_diff(turn, state.prev_text, current_text)
    summary = _diff_summary(turn, _diff_counts(diff_lines), body)
    return f"{summary}\n--- turn {turn:03d} diff ---\n{''.join(diff_lines)}"


def _self_contained_record(state: SessionState, body: dict, current_text: str) -> str:
    """Self-contained record: every turn's full rendering; turn 0 also the body."""
    turn = state.turn
    if turn == 0:
        return _turn_zero_block(body, current_text)
    counts = _diff_counts(_unified_diff(turn, state.prev_text, current_text))
    summary = _diff_summary(turn, counts, body)
    return f"{summary}\n--- turn {turn:03d} rendered ---\n{current_text}"


def _turn_zero_block(body: dict, current_text: str) -> str:
    """Turn-0 record shared by both collapsed formats: body plus rendering."""
    summary = _full_summary(body, current_text)
    json_body = json.dumps(body, indent=2, ensure_ascii=False)
    return (f"{summary}\n--- turn 000 body ---\n{json_body}\n"
            f"--- turn 000 rendered ---\n{current_text}")


def _full_summary(body: dict, text: str) -> str:
    """Turn-0 summary line, the FULL form (format-independent)."""
    n_msgs, n_tools = _msg_tool_counts(body)
    return (f"turn 000  {_now()}  FULL  {n_msgs} msgs  {n_tools} tools  "
            f"{len(text) / 1024:.1f} KB")


def _diff_summary(turn: int, counts: tuple[int, int], body: dict) -> str:
    """Turn-≥1 summary line, the +added/-removed form (format-independent)."""
    added, removed = counts
    n_msgs, n_tools = _msg_tool_counts(body)
    return (f"turn {turn:03d}  {_now()}  +{added}/-{removed} lines  "
            f"{n_msgs} msgs  {n_tools} tools")


def _msg_tool_counts(body: dict) -> tuple[int, int]:
    """(message count, tool count) used by every summary line."""
    return len(body.get("messages") or []), len(body.get("tools") or [])


def _now() -> str:
    """UTC wall-clock timestamp for summary lines."""
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def _unified_diff(turn: int, prev_text: str, current_text: str) -> list[str]:
    """Unified diff lines for turn ≥ 1; ``lineterm`` terminates only the control lines."""
    prev = prev_text.splitlines(keepends=True)
    curr = current_text.splitlines(keepends=True)
    return list(difflib.unified_diff(
        prev, curr,
        fromfile=f"turn_{turn - 1:03d}", tofile=f"turn_{turn:03d}",
        lineterm="\n",
    ))


def _diff_counts(diff_lines: list[str]) -> tuple[int, int]:
    """(+added, -removed) line counts, excluding the +++/--- file headers."""
    added = sum(1 for l in diff_lines if l.startswith("+") and not l.startswith("+++"))
    removed = sum(1 for l in diff_lines if l.startswith("-") and not l.startswith("---"))
    return added, removed


def _artifact_path(state: SessionState, fmt: str) -> Path:
    """The collapsed artifact's path: DUMP_DIR/<sid>/<format>.log."""
    return state.session_dir / f"{fmt}{_ARTIFACT_SUFFIX}"


def _flush_session(state: SessionState) -> None:
    """Flush one session's pending batch; failures are logged, never raised."""
    if not state.pending:
        return
    try:
        _flush_state(state)
    except Exception:
        log.exception("dump: flush failed for session %s", state.session_dir.name)


def _flush_state(state: SessionState) -> None:
    """Append the pending batch as one write, then clear the buffer."""
    if state.fmt == "documented":
        return
    path = _artifact_path(state, state.fmt)
    if not _writable_target(path):
        raise ValueError(f"dump: refusing non-regular artifact {path}")
    with path.open("a", encoding="utf-8") as f:
        f.write("\n\n".join(state.pending) + "\n")
    state.pending.clear()
    state.pending_bytes = 0


def _require_writable(path: Path) -> None:
    """Refuse symlinked/non-regular write targets (durable-writer guard)."""
    if not _writable_target(path):
        raise ValueError(f"dump: refusing non-regular target {path}")


def _writable_target(path: Path) -> bool:
    """A write target must be absent or a regular, non-symlink file."""
    return not path.is_symlink() and (not path.exists() or path.is_file())

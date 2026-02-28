"""Session-aware prompt diffing — saves full body (turn 0) or diff (turn N)."""

import dataclasses
import difflib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from . import config
from .session import session_id

log = logging.getLogger("buster-ripper")


@dataclasses.dataclass
class SessionState:
    turn: int = 0
    prev_text: str = ""
    session_dir: Path = None  # type: ignore[assignment]


# session_id → SessionState
_sessions: dict[str, SessionState] = {}


def _render_body(body: dict) -> str:
    """Human-readable rendering of a request body for diffing."""
    parts: list[str] = []

    tools = body.get("tools") or []
    if tools:
        names_line = "# " + ", ".join(t.get("name", "?") for t in tools)
        defs = json.dumps(tools, indent=2, ensure_ascii=False)
        parts.append(f"[tools] ({len(tools)} total)\n{names_line}\n{defs}")

    system = body.get("system", "")
    if isinstance(system, list):
        system_text = "\n".join(b.get("text", "") for b in system if isinstance(b, dict))
    else:
        system_text = system or ""
    if system_text:
        parts.append(f"[system]\n{system_text}")

    for i, msg in enumerate(body.get("messages") or []):
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if isinstance(content, list):
            content_text = "\n".join(
                b.get("text", json.dumps(b, ensure_ascii=False))
                if isinstance(b, dict) else str(b)
                for b in content
            )
        else:
            content_text = str(content)
        parts.append(f"[message {i} / {role}]\n{content_text}")

    return "\n\n" + ("\n\n" + "─" * 80 + "\n\n").join(parts) + "\n"


def dump_turn(body: dict) -> None:
    """Save full body (turn 0) or unified diff (turn N) for this session."""
    assert config.DUMP_DIR is not None
    sid = session_id(body)
    state = _sessions.get(sid)
    now = datetime.now(timezone.utc).strftime("%H:%M:%S")
    n_msgs = len(body.get("messages") or [])
    n_tools = len(body.get("tools") or [])

    if state is None:
        session_dir = config.DUMP_DIR / sid
        session_dir.mkdir(parents=True, exist_ok=True)
        state = SessionState(turn=0, prev_text="", session_dir=session_dir)
        _sessions[sid] = state
        log.info("dump: new session %s → %s", sid, session_dir)

    current_text = _render_body(body)
    turn = state.turn

    if turn == 0:
        out = state.session_dir / "turn_000.json"
        out.write_text(json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8")
        size_kb = len(current_text) / 1024
        summary = f"turn 000  {now}  FULL  {n_msgs} msgs  {n_tools} tools  {size_kb:.1f} KB"
        log.info("dump: %s", summary)
    else:
        prev_lines = state.prev_text.splitlines(keepends=True)
        curr_lines = current_text.splitlines(keepends=True)
        diff_lines = list(difflib.unified_diff(
            prev_lines, curr_lines,
            fromfile=f"turn_{turn - 1:03d}", tofile=f"turn_{turn:03d}",
            lineterm="",
        ))
        out = state.session_dir / f"turn_{turn:03d}.diff"
        out.write_text("".join(diff_lines), encoding="utf-8")
        added = sum(1 for l in diff_lines if l.startswith("+") and not l.startswith("+++"))
        removed = sum(1 for l in diff_lines if l.startswith("-") and not l.startswith("---"))
        summary = (
            f"turn {turn:03d}  {now}  +{added}/-{removed} lines  "
            f"{n_msgs} msgs  {n_tools} tools"
        )
        log.info("dump: %s", summary)

    index = state.session_dir / "index.txt"
    with index.open("a", encoding="utf-8") as f:
        f.write(summary + "\n")

    state.prev_text = current_text
    state.turn += 1

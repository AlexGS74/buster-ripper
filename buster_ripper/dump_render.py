"""Body rendering for the dump: tools, system and message sections as diffable text."""

import json


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


def _render_message(index: int, msg: dict) -> str:
    """One message section: the [message i / role] header and its content text."""
    role = msg.get("role", "?")
    content_text = _content_text(msg.get("content", ""))
    return f"[message {index} / {role}]\n{content_text}"


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
        parts.append(_render_message(i, msg))

    return "\n\n" + ("\n\n" + "─" * 80 + "\n\n").join(parts) + "\n"

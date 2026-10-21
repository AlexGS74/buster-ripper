"""Shared process-level QA harness for the buster-ripper gates.

Drives the real proxy as a subprocess over real HTTP with real signals, per
``.agents/skills/qa-project/SKILL.md``. These gates need the remote upstream
(``http://hyperipper.local:30000``) and are NOT part of the hermetic default
``uv run pytest`` run; select them explicitly::

    .venv/bin/python -m pytest tests/qa -o addopts=

The endpoint is the model server; the ``:30001`` correction is recorded in
``decisions.md`` §US-009. Synchronization is by observation — readiness by polling
the listening port, quiescence by waiting for process exit or polling the
artifact/stderr — never a fixed sleep.

``Proxy(..., upstream=..., require_live_upstream=False)`` is the harness hook for
the ``MS-004`` upstream-TLS gate's own loopback peer, the one amendment to the
upstream requirement (``.agents/skills/qa-project/SKILL.md`` §Upstream).
"""

from __future__ import annotations

import hashlib
import os
import re


def session_id(system_prompt: str) -> str:
    """First 12 hex chars of SHA-256(system prompt text)."""
    return hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:12]


def billing_system(system: str, nonce: str) -> str:
    """The framework's billing-header form of a system prompt.

    ``README.md`` §Normalizations 1: Claude Code injects
    ``x-anthropic-billing-header: ... cch=<nonce>`` at position 0 of the system
    prompt on every request and buster-ripper strips it. A gate that must observe
    that stripping sends this form; a nonce-free body is identical before and
    after normalization and cannot distinguish the ordering.
    """
    return (
        "x-anthropic-billing-header: cc_version=2.1.50.f15; "
        f"cc_entrypoint=cli; cch={nonce};\n{system}"
    )


def turn_body(
    system: str, user_text: str, model: str = "sloppus", nonce: str | None = None
) -> dict:
    """One /v1/messages request body carrying a single user message.

    ``nonce`` selects the framework's billing-header form of ``system`` (the
    normalization gate); omitting it keeps every existing caller's body bytes and
    ``session_id`` derivation unchanged.
    """
    return {
        "model": model,
        "max_tokens": 16,
        "system": billing_system(system, nonce) if nonce else system,
        "messages": [{"role": "user", "content": user_text}],
    }


def record_section(artifact_text: str, header: str) -> str:
    """One record's body from a collapsed artifact, bounded by its public headers.

    ``README.md`` §Reading a collapsed artifact publishes the section headers
    (``--- turn NNN body ---``, ``--- turn NNN rendered ---``,
    ``--- turn NNN diff ---``) and the one-line-per-turn summary lines; a record
    runs from its header to the next summary line or section header.
    """
    i = artifact_text.find(header)
    assert i >= 0, f"artifact is missing section header {header!r}"
    start = i + len(header)
    m = re.search(r"(?m)^(?:--- turn |turn \d{3}\b)", artifact_text[start:])
    return artifact_text[start:start + m.start()] if m else artifact_text[start:]


def read_text(path: str, newline: str | None = None) -> str:
    """Read a dump artifact as text, tolerating a hostile byte in a captured body.

    ``newline`` is passed to ``open`` unchanged. The default (``None``) is the
    universal-newline mode the JSON, index and nonce reads want. A comparison over
    the writer's own line identity passes ``newline=""``: the documented diff is
    computed over the renderings *before* the write, so the expected side must see
    the same ``\\r\\n``/``\\r`` bytes production diffed (``S21`` CE-2).
    """
    with open(path, encoding="utf-8", errors="replace", newline=newline) as f:
        return f.read()


def rglob_files(root: str) -> list[str]:
    """Every file under ``root``, whatever the layout."""
    out = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            out.append(os.path.join(dirpath, name))
    return out


def log_artifacts(dump_dir: str) -> list[str]:
    """The collapsed ``*.log`` artifacts anywhere under a dump root.

    The documented path must produce none of these; the collapsed path must
    produce exactly one per configured format.
    """
    return sorted(str(p) for p in rglob_files(dump_dir) if p.endswith(".log"))


def listing(path: str) -> list[str]:
    return sorted(os.listdir(path)) if os.path.isdir(path) else []


def assert_upstream_response(status: int, body: bytes) -> None:
    """Valid requests traverse the live peer without a proxy transport failure.

    The status class is the whole oracle: a proxy-local failure to reach the peer is answered
    ``502``, so ``< 500`` separates "the peer answered" from "the proxy could not reach it".
    The body is deliberately **not** inspected — a relayed answer is the peer's own bytes — and
    the dropped ``"Cannot connect"`` limb is recorded in ``chronicle.md``. Whether an error body
    leaks the proxy's configuration is witnessed at ``tests/test_upstream_answer_relay.py``'s
    ``TestLeakFreeErrorBodies``.
    """
    assert 200 <= status < 500, f"valid request failed with HTTP {status}: {body[:300]!r}"

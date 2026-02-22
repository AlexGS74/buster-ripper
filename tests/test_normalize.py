"""Tests for the normalize() function and its sub-normalizations."""

import pytest

import buster_ripper as br


# ── Billing nonce stripping ───────────────────────────────────────────────────


def test_billing_nonce_stripped_from_system_list():
    body = {
        "system": [
            {"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.50; cch=abc123;\nReal system prompt."},
        ],
        "messages": [],
    }
    out, changes = br.normalize(body)
    assert out["system"][0]["text"] == "Real system prompt."
    assert any("billing" in c for c in changes)


def test_billing_nonce_only_block_removed():
    """A system block containing only the billing nonce is dropped entirely."""
    body = {
        "system": [
            {"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.50; cch=abc123;\n"},
            {"type": "text", "text": "Keep this."},
        ],
        "messages": [],
    }
    out, changes = br.normalize(body)
    assert len(out["system"]) == 1
    assert out["system"][0]["text"] == "Keep this."
    assert any("billing" in c for c in changes)


def test_billing_nonce_stripped_from_system_string():
    body = {
        "system": "x-anthropic-billing-header: cch=xyz;\nSystem prompt here.",
        "messages": [],
    }
    out, changes = br.normalize(body)
    assert out["system"] == "System prompt here."
    assert any("billing" in c for c in changes)


def test_no_billing_nonce_no_change():
    body = {"system": [{"type": "text", "text": "Clean system prompt."}], "messages": []}
    out, changes = br.normalize(body)
    assert out["system"][0]["text"] == "Clean system prompt."
    assert not any("billing" in c for c in changes)


# ── cache_control stripping ───────────────────────────────────────────────────


def test_cache_control_stripped_from_system():
    body = {
        "system": [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}],
        "messages": [],
    }
    out, changes = br.normalize(body)
    assert "cache_control" not in out["system"][0]
    assert any("cache_control" in c for c in changes)


def test_cache_control_stripped_from_message_content_block():
    body = {
        "system": [],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}}
                ],
            }
        ],
    }
    out, changes = br.normalize(body)
    assert "cache_control" not in out["messages"][0]["content"][0]
    assert any("cache_control" in c for c in changes)


def test_cache_control_stripped_from_message_top_level():
    body = {
        "system": [],
        "messages": [
            {"role": "user", "content": "hi", "cache_control": {"type": "ephemeral"}}
        ],
    }
    out, changes = br.normalize(body)
    assert "cache_control" not in out["messages"][0]
    assert any("cache_control" in c for c in changes)


# ── Tool sorting ──────────────────────────────────────────────────────────────


def test_tools_sorted_by_name():
    body = {
        "tools": [
            {"name": "Zebra", "description": "z"},
            {"name": "Alpha", "description": "a"},
            {"name": "Mango", "description": "m"},
        ],
        "messages": [],
    }
    out, changes = br.normalize(body)
    assert [t["name"] for t in out["tools"]] == ["Alpha", "Mango", "Zebra"]
    assert any("reordered" in c for c in changes)


def test_tools_already_sorted_no_change_logged():
    body = {
        "tools": [
            {"name": "Alpha", "description": "a"},
            {"name": "Beta", "description": "b"},
        ],
        "messages": [],
    }
    out, changes = br.normalize(body)
    assert [t["name"] for t in out["tools"]] == ["Alpha", "Beta"]
    assert not any("reordered" in c for c in changes)


def test_empty_tools_array_dropped():
    body = {"tools": [], "messages": []}
    out, changes = br.normalize(body)
    assert "tools" not in out
    assert any("empty tools" in c for c in changes)


def test_no_tools_key_unchanged():
    body = {"messages": [{"role": "user", "content": "hi"}]}
    out, changes = br.normalize(body)
    assert "tools" not in out
    assert not any("tools" in c for c in changes)


# ── Date stripping ────────────────────────────────────────────────────────────


def test_date_stripped_from_user_message_string(monkeypatch):
    monkeypatch.setattr(br, "STRIP_DATE", True)
    body = {
        "messages": [
            {"role": "user", "content": "Today's date is 2026-02-21.\nDo something."}
        ]
    }
    out, changes = br.normalize(body)
    assert "Today's date" not in out["messages"][0]["content"]
    assert any("currentDate" in c for c in changes)


def test_date_stripped_from_user_content_block(monkeypatch):
    monkeypatch.setattr(br, "STRIP_DATE", True)
    body = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Today's date is 2026-02-21.\nRemember this."}],
            }
        ]
    }
    out, changes = br.normalize(body)
    assert "Today's date" not in out["messages"][0]["content"][0]["text"]
    assert any("currentDate" in c for c in changes)


def test_date_not_stripped_when_disabled(monkeypatch):
    monkeypatch.setattr(br, "STRIP_DATE", False)
    body = {
        "messages": [
            {"role": "user", "content": "Today's date is 2026-02-21.\nDo something."}
        ]
    }
    out, changes = br.normalize(body)
    assert "Today's date is 2026-02-21." in out["messages"][0]["content"]
    assert not any("currentDate" in c for c in changes)


def test_date_not_stripped_from_assistant_message(monkeypatch):
    monkeypatch.setattr(br, "STRIP_DATE", True)
    body = {
        "messages": [
            {"role": "assistant", "content": "Today's date is 2026-02-21."}
        ]
    }
    out, changes = br.normalize(body)
    assert "Today's date is 2026-02-21." in out["messages"][0]["content"]
    assert not any("currentDate" in c for c in changes)

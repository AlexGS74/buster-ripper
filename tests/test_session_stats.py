"""Tests for SessionStats, usage extraction, and session tracking."""

import json

import pytest

import buster_ripper.session as session_mod
from buster_ripper.session import (
    SessionStats,
    session_id,
    update_session_stats,
    _stats,
)
from buster_ripper.utils import extract_input_tokens, find_message_start


# ── SessionStats.update ───────────────────────────────────────────────────────


def test_first_update_sets_latency_directly():
    s = SessionStats()
    s.update(input_tokens=1000, latency_ms=5000.0)
    assert s.input_tokens == 1000
    assert s.avg_latency_ms == 5000.0
    assert s.request_count == 1


def test_second_update_applies_ema():
    s = SessionStats()
    s.update(1000, 5000.0)
    s.update(2000, 10000.0, ema_alpha=0.3)
    assert s.input_tokens == 2000
    assert s.avg_latency_ms == pytest.approx(6500.0)
    assert s.request_count == 2


def test_ema_smooths_spike():
    s = SessionStats()
    for _ in range(5):
        s.update(50_000, 3000.0)
    assert s.avg_latency_ms < 5000

    s.update(50_000, 30_000.0)
    assert s.avg_latency_ms < 15_000


def test_input_tokens_always_latest():
    s = SessionStats()
    s.update(100_000, 1000.0)
    s.update(120_000, 2000.0)
    s.update(50_000, 1000.0)
    assert s.input_tokens == 50_000


# ── extract_input_tokens ──────────────────────────────────────────────────────


def test_extract_input_tokens_from_response():
    response = json.dumps({
        "type": "message",
        "usage": {"input_tokens": 12345, "output_tokens": 50},
    }).encode()
    assert extract_input_tokens(response) == 12345


def test_extract_input_tokens_missing_usage():
    response = json.dumps({"type": "message"}).encode()
    assert extract_input_tokens(response) is None


def test_extract_input_tokens_invalid_json():
    assert extract_input_tokens(b"not json at all") is None


# ── find_message_start ────────────────────────────────────────────────────────


def _make_sse_chunk(event: dict) -> bytes:
    return f"data: {json.dumps(event)}\n\n".encode()


def test_find_message_start_basic():
    evt = {
        "type": "message_start",
        "message": {"id": "msg_abc", "usage": {"input_tokens": 9876, "output_tokens": 0}},
    }
    result = find_message_start(_make_sse_chunk(evt))
    assert result is not None
    assert result["type"] == "message_start"
    assert result["message"]["usage"]["input_tokens"] == 9876


def test_find_message_start_not_present():
    assert find_message_start(_make_sse_chunk({"type": "content_block_start", "index": 0})) is None


def test_find_message_start_done_line():
    assert find_message_start(b"data: [DONE]\n\n") is None


def test_find_message_start_multi_event_chunk():
    ping = json.dumps({"type": "ping"})
    start = json.dumps({"type": "message_start", "message": {"usage": {"input_tokens": 42}}})
    chunk = f"data: {ping}\n\ndata: {start}\n\n".encode()
    result = find_message_start(chunk)
    assert result is not None
    assert result["message"]["usage"]["input_tokens"] == 42


def test_find_message_start_garbled_json():
    assert find_message_start(b"data: {not valid json}\n\n") is None


# ── session_id ────────────────────────────────────────────────────────────────


def test_session_id_stable_for_same_system():
    body = {"system": [{"type": "text", "text": "You are a helpful assistant."}], "messages": []}
    assert session_id(body) == session_id(body)


def test_session_id_differs_for_different_system():
    assert session_id({"system": "A", "messages": []}) != session_id({"system": "B", "messages": []})


def test_session_id_ignores_messages():
    body1 = {"system": "Same system", "messages": [{"role": "user", "content": "turn 1"}]}
    body2 = {"system": "Same system", "messages": [{"role": "user", "content": "turn 2"}]}
    assert session_id(body1) == session_id(body2)


def test_session_id_is_12_hex_chars():
    sid = session_id({"system": "test", "messages": []})
    assert len(sid) == 12
    assert all(c in "0123456789abcdef" for c in sid)


# ── update_session_stats ──────────────────────────────────────────────────────


def test_update_session_stats_creates_entry(monkeypatch):
    monkeypatch.setattr(session_mod, "_stats", {})
    monkeypatch.setattr(session_mod, "_db_conn", None)
    update_session_stats("test_sid", 5000, 2000.0)
    assert "test_sid" in session_mod._stats
    assert session_mod._stats["test_sid"].input_tokens == 5000


def test_update_session_stats_accumulates(monkeypatch):
    monkeypatch.setattr(session_mod, "_stats", {})
    monkeypatch.setattr(session_mod, "_db_conn", None)
    update_session_stats("sid2", 5000, 2000.0)
    update_session_stats("sid2", 7000, 4000.0)
    stats = session_mod._stats["sid2"]
    assert stats.input_tokens == 7000
    assert stats.request_count == 2

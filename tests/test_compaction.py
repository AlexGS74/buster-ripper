"""Tests for the compaction policy and the /v1/messages/count_tokens endpoint."""

import json

import pytest
from fastapi.testclient import TestClient

import buster_ripper.config as cfg
import buster_ripper.compaction as compaction_mod
import buster_ripper.session as session_mod
from buster_ripper.app import app
from buster_ripper.compaction import compact_token_count, should_nudge_compact
from buster_ripper.session import SessionStats


# ── should_nudge_compact ──────────────────────────────────────────────────────


def test_nudge_when_both_thresholds_exceeded():
    assert should_nudge_compact(
        input_tokens=160_000,
        avg_latency_ms=20_000,
        max_model_len=200_000,
        token_ratio=0.75,
        latency_threshold_ms=15_000,
    )


def test_no_nudge_when_only_tokens_high():
    assert not should_nudge_compact(
        input_tokens=160_000,
        avg_latency_ms=5_000,
        max_model_len=200_000,
        token_ratio=0.75,
        latency_threshold_ms=15_000,
    )


def test_no_nudge_when_only_latency_high():
    assert not should_nudge_compact(
        input_tokens=50_000,
        avg_latency_ms=20_000,
        max_model_len=200_000,
        token_ratio=0.75,
        latency_threshold_ms=15_000,
    )


def test_no_nudge_when_both_below():
    assert not should_nudge_compact(
        input_tokens=50_000,
        avg_latency_ms=5_000,
        max_model_len=200_000,
        token_ratio=0.75,
        latency_threshold_ms=15_000,
    )


def test_nudge_at_exact_token_threshold():
    assert should_nudge_compact(
        input_tokens=150_000,
        avg_latency_ms=20_000,
        max_model_len=200_000,
        token_ratio=0.75,
        latency_threshold_ms=15_000,
    )


def test_no_nudge_one_token_below_threshold():
    assert not should_nudge_compact(
        input_tokens=149_999,
        avg_latency_ms=20_000,
        max_model_len=200_000,
        token_ratio=0.75,
        latency_threshold_ms=15_000,
    )


def test_nudge_on_kv_pressure_alone():
    assert should_nudge_compact(
        input_tokens=10_000,
        avg_latency_ms=100,
        max_model_len=200_000,
        token_ratio=0.75,
        latency_threshold_ms=15_000,
        kv_cache_usage=0.80,
        kv_ratio=0.75,
    )


def test_no_nudge_kv_below_threshold():
    assert not should_nudge_compact(
        input_tokens=10_000,
        avg_latency_ms=100,
        max_model_len=200_000,
        token_ratio=0.75,
        latency_threshold_ms=15_000,
        kv_cache_usage=0.74,
        kv_ratio=0.75,
    )


def test_nudge_kv_at_exact_threshold():
    assert should_nudge_compact(
        input_tokens=10_000,
        avg_latency_ms=100,
        max_model_len=200_000,
        token_ratio=0.75,
        latency_threshold_ms=15_000,
        kv_cache_usage=0.75,
        kv_ratio=0.75,
    )


def test_kv_ratio_1_0_disables_kv_trigger():
    assert not should_nudge_compact(
        input_tokens=10_000,
        avg_latency_ms=100,
        max_model_len=200_000,
        token_ratio=0.75,
        latency_threshold_ms=15_000,
        kv_cache_usage=0.99,
        kv_ratio=1.0,
    )


# ── compact_token_count ───────────────────────────────────────────────────────


def test_compact_token_count_value():
    assert compact_token_count(200_000, 0.95) == 190_000


def test_compact_token_count_is_int():
    assert isinstance(compact_token_count(200_000, 0.95), int)


# ── /v1/messages/count_tokens HTTP endpoint ───────────────────────────────────


@pytest.fixture()
def client(monkeypatch):
    """TestClient with patched config and clean stats."""
    monkeypatch.setattr(session_mod, "_stats", {})
    monkeypatch.setattr(cfg, "UPSTREAM", "http://test-upstream")
    monkeypatch.setattr(cfg, "MAX_MODEL_LEN", 200_000)
    monkeypatch.setattr(cfg, "COMPACT_TOKEN_RATIO", 0.75)
    monkeypatch.setattr(cfg, "COMPACT_LATENCY_MS", 15_000)
    monkeypatch.setattr(cfg, "COMPACT_NUDGE_RATIO", 0.95)
    monkeypatch.setattr(cfg, "COMPACT_KV_RATIO", 0.75)
    monkeypatch.setattr(compaction_mod, "_kv_cache_usage", 0.0)
    return TestClient(app, raise_server_exceptions=True)


def _body(system: str = "sys") -> dict:
    return {"system": system, "messages": [{"role": "user", "content": "hi"}]}


def test_count_tokens_returns_tracked_tokens(client, monkeypatch):
    body = _body(system="stable_system")
    sid = session_mod.session_id(body)
    session_mod._stats[sid] = SessionStats(input_tokens=80_000, avg_latency_ms=3_000, request_count=3)

    resp = client.post("/v1/messages/count_tokens", json=body, params={"beta": "true"})
    assert resp.status_code == 200
    assert resp.json()["input_tokens"] == 80_000


def test_count_tokens_nudges_when_thresholds_exceeded(client):
    body = _body(system="large_context_system")
    sid = session_mod.session_id(body)
    session_mod._stats[sid] = SessionStats(input_tokens=160_000, avg_latency_ms=20_000, request_count=10)

    resp = client.post("/v1/messages/count_tokens", json=body)
    assert resp.status_code == 200
    reported = resp.json()["input_tokens"]
    assert reported == 190_000
    assert reported > 160_000


def test_count_tokens_no_nudge_fast_session(client):
    body = _body(system="fast_large_system")
    sid = session_mod.session_id(body)
    session_mod._stats[sid] = SessionStats(input_tokens=160_000, avg_latency_ms=2_000, request_count=5)

    resp = client.post("/v1/messages/count_tokens", json=body)
    assert resp.status_code == 200
    assert resp.json()["input_tokens"] == 160_000


def test_count_tokens_nudges_on_kv_pressure(client, monkeypatch):
    monkeypatch.setattr(compaction_mod, "_kv_cache_usage", 0.80)

    body = _body(system="small_fast_session")
    sid = session_mod.session_id(body)
    session_mod._stats[sid] = SessionStats(input_tokens=10_000, avg_latency_ms=500, request_count=2)

    resp = client.post("/v1/messages/count_tokens", json=body)
    assert resp.status_code == 200
    assert resp.json()["input_tokens"] == 190_000


def test_count_tokens_no_nudge_kv_below_threshold(client, monkeypatch):
    monkeypatch.setattr(compaction_mod, "_kv_cache_usage", 0.50)

    body = _body(system="small_fast_low_kv")
    sid = session_mod.session_id(body)
    session_mod._stats[sid] = SessionStats(input_tokens=10_000, avg_latency_ms=500, request_count=2)

    resp = client.post("/v1/messages/count_tokens", json=body)
    assert resp.status_code == 200
    assert resp.json()["input_tokens"] == 10_000


def test_count_tokens_fallback_estimate_on_upstream_404(client, monkeypatch):
    import httpx as _httpx

    async def mock_post(*args, **kwargs):
        return _httpx.Response(404, text="Not Found")

    monkeypatch.setattr(_httpx.AsyncClient, "post", mock_post)

    body = _body(system="no_stats_system")
    assert session_mod.session_id(body) not in session_mod._stats

    resp = client.post("/v1/messages/count_tokens", json=body)
    assert resp.status_code == 200
    assert resp.json()["input_tokens"] > 0

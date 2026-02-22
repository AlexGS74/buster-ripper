"""Tests for the compaction policy and the /v1/messages/count_tokens endpoint."""

import json

import pytest
from fastapi.testclient import TestClient

import buster_ripper as br
from buster_ripper import (
    SessionStats,
    app,
    compact_token_count,
    should_nudge_compact,
)


# ── should_nudge_compact ──────────────────────────────────────────────────────


def test_nudge_when_both_thresholds_exceeded():
    assert should_nudge_compact(
        input_tokens=160_000,    # 80% of 200k → above 75% threshold
        avg_latency_ms=20_000,   # 20 s → above 15 s threshold
        max_model_len=200_000,
        token_ratio=0.75,
        latency_threshold_ms=15_000,
    )


def test_no_nudge_when_only_tokens_high():
    assert not should_nudge_compact(
        input_tokens=160_000,
        avg_latency_ms=5_000,    # fast → below threshold
        max_model_len=200_000,
        token_ratio=0.75,
        latency_threshold_ms=15_000,
    )


def test_no_nudge_when_only_latency_high():
    assert not should_nudge_compact(
        input_tokens=50_000,     # small context → below threshold
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
    # Exactly at the threshold — should nudge
    assert should_nudge_compact(
        input_tokens=150_000,    # 75% of 200k exactly
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
    """KV cache pressure alone (small session, fast) should trigger nudge."""
    assert should_nudge_compact(
        input_tokens=10_000,     # small — would NOT trigger per-session
        avg_latency_ms=100,      # fast — would NOT trigger per-session
        max_model_len=200_000,
        token_ratio=0.75,
        latency_threshold_ms=15_000,
        kv_cache_usage=0.80,     # above 0.75 threshold
        kv_ratio=0.75,
    )


def test_no_nudge_kv_below_threshold():
    """KV just under threshold + small session → no nudge."""
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
    """KV exactly at threshold → nudge."""
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
    """kv_ratio=1.0 means KV trigger is disabled (can never reach 100%)."""
    assert not should_nudge_compact(
        input_tokens=10_000,
        avg_latency_ms=100,
        max_model_len=200_000,
        token_ratio=0.75,
        latency_threshold_ms=15_000,
        kv_cache_usage=0.99,     # very high KV but kv_ratio disabled
        kv_ratio=1.0,
    )


# ── compact_token_count ───────────────────────────────────────────────────────


def test_compact_token_count_value():
    # 200_000 × 0.95 = 190_000
    assert compact_token_count(200_000, 0.95) == 190_000


def test_compact_token_count_is_int():
    result = compact_token_count(200_000, 0.95)
    assert isinstance(result, int)


# ── /v1/messages/count_tokens HTTP endpoint ───────────────────────────────────


@pytest.fixture()
def client(monkeypatch):
    """TestClient with patched upstream and clean stats."""
    monkeypatch.setattr(br, "_stats", {})
    monkeypatch.setattr(br, "UPSTREAM", "http://test-upstream")
    monkeypatch.setattr(br, "MAX_MODEL_LEN", 200_000)
    monkeypatch.setattr(br, "COMPACT_TOKEN_RATIO", 0.75)
    monkeypatch.setattr(br, "COMPACT_LATENCY_MS", 15_000)
    monkeypatch.setattr(br, "COMPACT_NUDGE_RATIO", 0.95)
    monkeypatch.setattr(br, "COMPACT_KV_RATIO", 0.75)
    monkeypatch.setattr(br, "_kv_cache_usage", 0.0)
    return TestClient(app, raise_server_exceptions=True)


def _body(system: str = "sys", n_tokens: int = 0) -> dict:
    """Minimal request body for count_tokens."""
    return {"system": system, "messages": [{"role": "user", "content": "hi"}]}


def test_count_tokens_returns_tracked_tokens(client, monkeypatch):
    """When we have stats, return them without hitting upstream."""
    body = _body(system="stable_system")
    sid = br._session_id(body)
    br._stats[sid] = SessionStats(input_tokens=80_000, avg_latency_ms=3_000, request_count=3)

    resp = client.post(
        "/v1/messages/count_tokens",
        json=body,
        params={"beta": "true"},
    )
    assert resp.status_code == 200
    assert resp.json()["input_tokens"] == 80_000


def test_count_tokens_nudges_when_thresholds_exceeded(client, monkeypatch):
    """When both thresholds exceeded, returned count is nudged to trigger compaction."""
    body = _body(system="large_context_system")
    sid = br._session_id(body)
    # 160k tokens (80%) + 20s avg latency → both thresholds exceeded
    br._stats[sid] = SessionStats(input_tokens=160_000, avg_latency_ms=20_000, request_count=10)

    resp = client.post("/v1/messages/count_tokens", json=body)
    assert resp.status_code == 200
    reported = resp.json()["input_tokens"]
    # Should be nudged to 200_000 × 0.95 = 190_000
    assert reported == 190_000
    assert reported > 160_000


def test_count_tokens_no_nudge_fast_session(client, monkeypatch):
    """High token count but fast session → no nudge."""
    body = _body(system="fast_large_system")
    sid = br._session_id(body)
    br._stats[sid] = SessionStats(input_tokens=160_000, avg_latency_ms=2_000, request_count=5)

    resp = client.post("/v1/messages/count_tokens", json=body)
    assert resp.status_code == 200
    assert resp.json()["input_tokens"] == 160_000


def test_count_tokens_nudges_on_kv_pressure(client, monkeypatch):
    """Small/fast session gets nudged when server KV cache is above threshold."""
    monkeypatch.setattr(br, "_kv_cache_usage", 0.80)  # above 0.75 COMPACT_KV_RATIO

    body = _body(system="small_fast_session")
    sid = br._session_id(body)
    # Small, fast session — would NOT trigger per-session compaction
    br._stats[sid] = SessionStats(input_tokens=10_000, avg_latency_ms=500, request_count=2)

    resp = client.post("/v1/messages/count_tokens", json=body)
    assert resp.status_code == 200
    # Should be nudged due to KV pressure: 200_000 × 0.95 = 190_000
    assert resp.json()["input_tokens"] == 190_000


def test_count_tokens_no_nudge_kv_below_threshold(client, monkeypatch):
    """Small/fast session NOT nudged when KV cache is below threshold."""
    monkeypatch.setattr(br, "_kv_cache_usage", 0.50)  # below threshold

    body = _body(system="small_fast_low_kv")
    sid = br._session_id(body)
    br._stats[sid] = SessionStats(input_tokens=10_000, avg_latency_ms=500, request_count=2)

    resp = client.post("/v1/messages/count_tokens", json=body)
    assert resp.status_code == 200
    assert resp.json()["input_tokens"] == 10_000


def test_count_tokens_fallback_estimate_on_upstream_404(client, monkeypatch):
    """No stats + upstream 404 → fall back to character-count estimate."""
    import httpx as _httpx

    async def mock_post(*args, **kwargs):
        return _httpx.Response(404, text="Not Found")

    monkeypatch.setattr(_httpx.AsyncClient, "post", mock_post)

    body = _body(system="no_stats_system")
    # Ensure no stats exist for this session
    assert br._session_id(body) not in br._stats

    resp = client.post("/v1/messages/count_tokens", json=body)
    assert resp.status_code == 200
    # Estimate should be > 0 and reasonable (not nudged, no stats to compare against)
    tokens = resp.json()["input_tokens"]
    assert tokens > 0

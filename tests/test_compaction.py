"""Tests for compaction module with behavioral session testing."""

from buster_ripper.compaction import compact_token_count, should_nudge_compact, CompactionConfig


# ── should_nudge_compact ──────────────────────────────────────────────────────

def test_nudge_when_both_thresholds_exceeded():
    """Nudge when both token and latency thresholds exceeded."""
    cfg = CompactionConfig(
        max_model_len=200_000,
        token_ratio=0.75,
        latency_threshold_ms=15_000,
    )
    assert should_nudge_compact(
        input_tokens=160_000,
        avg_latency_ms=20_000,
        kv_cache_usage=0.0,
        cfg=cfg,
    )


def test_no_nudge_when_only_tokens_high():
    """No nudge when only token threshold is high."""
    cfg = CompactionConfig(
        max_model_len=200_000,
        token_ratio=0.75,
        latency_threshold_ms=15_000,
    )
    assert not should_nudge_compact(
        input_tokens=160_000,
        avg_latency_ms=5_000,
        kv_cache_usage=0.0,
        cfg=cfg,
    )


def test_no_nudge_when_only_latency_high():
    """No nudge when only latency threshold is high."""
    cfg = CompactionConfig(
        max_model_len=200_000,
        token_ratio=0.75,
        latency_threshold_ms=15_000,
    )
    assert not should_nudge_compact(
        input_tokens=50_000,
        avg_latency_ms=20_000,
        kv_cache_usage=0.0,
        cfg=cfg,
    )


def test_no_nudge_when_both_below():
    """No nudge when both below thresholds."""
    cfg = CompactionConfig(
        max_model_len=200_000,
        token_ratio=0.75,
        latency_threshold_ms=15_000,
    )
    assert not should_nudge_compact(
        input_tokens=50_000,
        avg_latency_ms=5_000,
        kv_cache_usage=0.0,
        cfg=cfg,
    )


def test_nudge_at_exact_token_threshold():
    """Nudge at exact token threshold."""
    cfg = CompactionConfig(
        max_model_len=200_000,
        token_ratio=0.75,
        latency_threshold_ms=15_000,
    )
    assert should_nudge_compact(
        input_tokens=150_000,
        avg_latency_ms=20_000,
        kv_cache_usage=0.0,
        cfg=cfg,
    )


def test_no_nudge_one_token_below_threshold():
    """No nudge one token below threshold."""
    cfg = CompactionConfig(
        max_model_len=200_000,
        token_ratio=0.75,
        latency_threshold_ms=15_000,
    )
    assert not should_nudge_compact(
        input_tokens=149_999,
        avg_latency_ms=20_000,
        kv_cache_usage=0.0,
        cfg=cfg,
    )


def test_nudge_on_kv_pressure_alone():
    """Nudge on KV cache pressure alone."""
    cfg = CompactionConfig(
        max_model_len=200_000,
        token_ratio=0.75,
        latency_threshold_ms=15_000,
        kv_ratio=0.75,
    )
    assert should_nudge_compact(
        input_tokens=10_000,
        avg_latency_ms=100,
        kv_cache_usage=0.80,
        cfg=cfg,
    )


def test_no_nudge_kv_below_threshold():
    """No nudge when KV below threshold."""
    cfg = CompactionConfig(
        max_model_len=200_000,
        token_ratio=0.75,
        latency_threshold_ms=15_000,
        kv_ratio=0.75,
    )
    assert not should_nudge_compact(
        input_tokens=10_000,
        avg_latency_ms=100,
        kv_cache_usage=0.74,
        cfg=cfg,
    )


def test_nudge_kv_at_exact_threshold():
    """Nudge when KV at exact threshold."""
    cfg = CompactionConfig(
        max_model_len=200_000,
        token_ratio=0.75,
        latency_threshold_ms=15_000,
        kv_ratio=0.75,
    )
    assert should_nudge_compact(
        input_tokens=10_000,
        avg_latency_ms=100,
        kv_cache_usage=0.75,
        cfg=cfg,
    )


def test_kv_ratio_1_0_disables_kv_trigger():
    """KV ratio 1.0 disables KV trigger."""
    cfg = CompactionConfig(
        max_model_len=200_000,
        token_ratio=0.75,
        latency_threshold_ms=15_000,
        kv_ratio=1.0,
    )
    assert not should_nudge_compact(
        input_tokens=10_000,
        avg_latency_ms=100,
        kv_cache_usage=0.99,
        cfg=cfg,
    )


# ── compact_token_count ───────────────────────────────────────────────────────

def test_compact_token_count_value():
    """Verify compact_token_count returns correct value."""
    assert compact_token_count(200_000, 0.95) == 190_000


def test_compact_token_count_is_int():
    """Verify compact_token_count returns int."""
    assert isinstance(compact_token_count(200_000, 0.95), int)

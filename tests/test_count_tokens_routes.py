"""Integration tests for the /v1/messages/count_tokens endpoint."""

import json

import pytest

from buster_ripper import config
from buster_ripper.normalize import normalize
from buster_ripper.session import session_id
from tests.upstream_stub import StubUpstream


def _sid(body: dict) -> str:
    """The digest count_tokens must key by: the one /v1/messages records stats under,
    which is ``session_id()`` of the *normalized* body (billing nonce already gone)."""
    return session_id(normalize(dict(body), session_id="")[0])


class TestCountTokensRoute:
    """Integration tests for /v1/messages/count_tokens endpoint."""

    async def test_count_tokens_returns_token_count(self, proxy, mocker):
        """The seeded stats are what the answer reports, keyed by the digest the route
        computes."""
        from buster_ripper import session as session_mod

        body = {"model": "claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "Test"}]}
        mocker.patch.dict(session_mod._stats,
                          {_sid(body): mocker.Mock(input_tokens=1000, avg_latency_ms=500)})

        response = await proxy.post("/v1/messages/count_tokens", json=body)

        assert response.status == 200
        assert await response.json() == {"input_tokens": 1000}

    async def test_a_billing_nonce_body_finds_the_stats_messages_recorded(self, proxy,
                                                                          mocker):
        """count_tokens must look up the post-normalize digest that ``/v1/messages``
        recorded stats under, so the per-request nonce cannot make the lookup miss."""
        from buster_ripper import session as session_mod

        mocker.patch.dict(session_mod._stats, {}, clear=True)
        answer = json.dumps({"id": "m", "usage": {"input_tokens": 1234}}).encode()
        mocker.patch.object(proxy.server.app["session"], "post", return_value=StubUpstream(
            body=answer, headers={"Content-Type": "application/json"}))
        body = {
            "model": "test", "max_tokens": 100,
            "system": [{"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.50\nsys"}],
            "messages": [{"role": "user", "content": "Hello"}],
        }

        assert (await proxy.post("/v1/messages", json=body)).status == 200
        response = await proxy.post("/v1/messages/count_tokens", json=body)

        assert await response.json() == {"input_tokens": 1234}, (
            "count_tokens answered its own estimate: the stats lookup missed")

    async def test_count_tokens_handles_empty_request(self, proxy):
        """Test that count_tokens handles empty request."""
        response = await proxy.post(
            "/v1/messages/count_tokens",
            json={},
        )

        assert response.status == 200
        data = await response.json()
        assert "input_tokens" in data

    @pytest.mark.parametrize("payload", [b"\xff", b"\xc3", b"\xef\xbb\xbf\xff"])
    async def test_count_tokens_rejects_undecodable_json(self, proxy, payload):
        response = await proxy.post(
            "/v1/messages/count_tokens", data=payload,
            headers={"Content-Type": "application/json"},
        )
        assert response.status == 400
        assert await response.json() == {"error": "Invalid JSON"}

    async def test_count_tokens_syntax_error_keeps_fallback(self, proxy):
        response = await proxy.post(
            "/v1/messages/count_tokens", data=b"{",
            headers={"Content-Type": "application/json"},
        )
        assert response.status == 200
        assert "input_tokens" in await response.json()

    @pytest.mark.parametrize("kv, latency", [
        pytest.param(0.85, 100, id="kv-pressure"),
        pytest.param(0.0, 20_000, id="per-session"),
    ])
    async def test_count_tokens_inflates_on_compaction_trigger(self, proxy, mocker, kv,
                                                               latency):
        """Either arm alone inflates the count, and each case rules the other out:
        ``per-session`` runs with KV at 0.0, so a green cannot be a kv-forced pass."""
        from buster_ripper import session as session_mod, compaction

        body = {"model": "claude-3-5-sonnet-20241022",
                "system": [{"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.50\nsys"}],
                "messages": [{"role": "user", "content": "Test"}]}
        mocker.patch.dict(session_mod._stats,
                          {_sid(body): mocker.Mock(input_tokens=180_000,
                                                   avg_latency_ms=latency)})
        original_kv = compaction._kv_cache_usage
        try:
            compaction._kv_cache_usage = kv

            response = await proxy.post("/v1/messages/count_tokens", json=body)

            assert response.status == 200
            nudged = int(config.MAX_MODEL_LEN * config.COMPACT_NUDGE_RATIO)
            assert await response.json() == {"input_tokens": nudged}
        finally:
            compaction._kv_cache_usage = original_kv

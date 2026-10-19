"""Integration tests for route handlers using aiohttp test client."""

import json

import pytest
from aiohttp.test_utils import TestClient, TestServer

from buster_ripper import backend, config
from buster_ripper.normalize import normalize
from buster_ripper.server import create_app
from buster_ripper.session import session_id
from tests.upstream_stub import StubUpstream


def _sid(body: dict) -> str:
    """The digest count_tokens must key by: the one /v1/messages records stats under,
    which is ``session_id()`` of the *normalized* body (billing nonce already gone)."""
    return session_id(normalize(dict(body), session_id="")[0])


def _create_mock_response(mocker, status=200, json_data=None, text_data=None, headers=None):
    """Create a mock aiohttp response with async context manager support.

    ``read()`` answers with the bytes of whatever else was declared, because a route
    that reads its payload once gets bytes and parses them itself: an upstream declared
    with ``json_data`` is a JSON answer, one declared with ``text_data`` is that text.
    Declaring only ``json_data`` used to be enough — nothing read the raw body.
    """
    mock_response = mocker.AsyncMock()
    mock_response.status = status
    mock_response.headers = headers or {}

    payload = b"test"
    if json_data is not None:
        mock_response.json = mocker.AsyncMock(return_value=json_data)
        payload = json.dumps(json_data).encode()
    if text_data is not None:
        mock_response.text = mocker.AsyncMock(return_value=text_data)
        payload = text_data.encode()

    mock_response.read = mocker.AsyncMock(return_value=payload)
    mock_response.__aenter__ = mocker.AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = mocker.AsyncMock(return_value=None)

    return mock_response


_MESSAGES_BODY = {
    "model": "test",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "Hello"}],
}


@pytest.fixture
async def client(mocker):
    """Create test client with mocked upstream backend."""
    app = create_app()

    # Mock backend detection
    mocker.patch.object(
        backend,
        "detect_backend",
        return_value=None,
    )
    mocker.patch.object(
        backend,
        "get_backend_state",
        return_value="sglang",
    )

    # Create test client
    async with TestServer(app) as server:
        async with TestClient(server) as client:
            yield client


class TestMessagesRoute:
    """Integration tests for /v1/messages endpoint."""

    async def test_messages_endpoint_accepts_json(self, client, mocker):
        """Test that /v1/messages accepts valid JSON."""
        # Mock upstream to return success
        mock_response = _create_mock_response(
            mocker,
            json_data={"id": "msg_test", "content": [{"type": "text", "text": "Hello"}]},
        )

        mocker.patch.object(
            client.server.app["session"],
            "post",
            return_value=mock_response,
        )

        response = await client.post(
            "/v1/messages",
            json={
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status == 200

    async def test_messages_endpoint_rejects_invalid_json(self, client):
        """Test that /v1/messages rejects invalid JSON."""
        response = await client.post(
            "/v1/messages",
            data="not json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status == 400
        data = await response.json()
        assert "error" in data

    @pytest.mark.parametrize("payload", [b"\xff", b"\xc3", b"\xef\xbb\xbf\xff"])
    async def test_messages_endpoint_rejects_undecodable_json(self, client, mocker, payload):
        upstream = mocker.patch.object(client.server.app["session"], "post")
        response = await client.post(
            "/v1/messages", data=payload, headers={"Content-Type": "application/json"},
        )
        assert response.status == 400
        assert await response.json() == {"error": "Invalid JSON"}
        upstream.assert_not_called()

    async def test_messages_endpoint_normalizes_request(self, client, mocker):
        """Test that /v1/messages normalizes the request (strips billing header)."""
        mock_response = _create_mock_response(
            mocker,
            json_data={"id": "msg_test", "content": [{"type": "text", "text": "Response"}]},
        )

        mocker.patch.object(
            client.server.app["session"],
            "post",
            return_value=mock_response,
        )

        # Send request with billing header in system prompt
        response = await client.post(
            "/v1/messages",
            json={
                "model": "test",
                "max_tokens": 100,
                "system": [{"type": "text", "text": "x-anthropic-billing-header: test\nReal system"}],
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        # Should succeed despite billing header
        assert response.status == 200


class TestChatRoute:
    """Integration tests for /v1/chat/completions endpoint."""

    @pytest.mark.parametrize("payload", [b"data: {\"x\":1}\n\n", b"data: [DONE]\n\n"])
    @pytest.mark.parametrize("content_type", ["text/event-stream", "application/problem+json"])
    async def test_chat_stream_relays_exact_sse_bytes(self, client, mocker, payload, content_type):
        upstream = _create_mock_response(mocker, headers={"Content-Type": content_type})

        async def chunks():
            yield payload[:4]
            yield payload[4:]

        upstream.content.iter_any = chunks
        mocker.patch.object(client.server.app["session"], "post", return_value=upstream)
        response = await client.post("/v1/chat/completions", json={"stream": True, "messages": []})
        assert response.status == 200
        assert response.headers["Content-Type"] == "text/event-stream"
        assert await response.read() == payload
        upstream.json.assert_not_awaited()

    @pytest.mark.parametrize("content_type, payload", [
        ("text/plain; charset=utf-8", b"rate limit"),
        ("application/problem+json", b'{"error":"busy"}'),
        (None, b"opaque error"),
    ])
    async def test_chat_stream_upstream_error_retains_representation(
        self, client, mocker, content_type, payload,
    ):
        """The stream error arm forwarded ``Content-Type`` alone, so a caller could not
        honor a ``Retry-After`` hint or correlate an ``X-Request-Id``. Every other
        end-to-end name travels with the body, and a body-less error still answers."""
        headers = {"Retry-After": "7", "X-Request-Id": "req_7f3a"}
        if content_type:
            headers["Content-Type"] = content_type
        upstream = _create_mock_response(mocker, status=429, headers=headers)
        upstream.read.return_value = payload
        mocker.patch.object(client.server.app["session"], "post", return_value=upstream)
        response = await client.post("/v1/chat/completions", json={"stream": True, "messages": []})
        assert response.status == 429
        assert await response.read() == payload
        assert response.headers["Content-Type"] == (content_type or "application/octet-stream")
        assert response.headers["Retry-After"] == "7"
        assert response.headers["X-Request-Id"] == "req_7f3a"
        upstream.json.assert_not_awaited()

    async def test_chat_nonstream_compatible_json_transforms(self, client, mocker):
        upstream = _create_mock_response(
            mocker, json_data={"choices": [{"message": {"content": "<think>reason</think>answer"}}]},
            headers={"Content-Type": "application/problem+json"},
        )
        mocker.patch.object(client.server.app["session"], "post", return_value=upstream)
        response = await client.post("/v1/chat/completions", json={"stream": False, "messages": []})
        assert response.status == 200
        assert json.loads(await response.read())["choices"][0]["message"]["content"] == "answer"
        # The mechanism pin, re-aimed at the read-once shape: the non-stream arm reads the
        # payload once and parses those bytes itself, so nothing asks for ``json()``. It
        # used to be the only read, and that is why an unparseable answer could only ever
        # become a proxy error. Transform coverage above, wiring here.
        upstream.read.assert_awaited_once()
        upstream.json.assert_not_awaited()

    async def test_chat_endpoint_accepts_json(self, client, mocker):
        """Test that /v1/chat/completions accepts valid JSON."""
        mock_response = _create_mock_response(
            mocker,
            json_data={
                "id": "chat_test",
                "choices": [{"message": {"content": "Hello"}, "finish_reason": "stop"}],
            },
        )

        mocker.patch.object(
            client.server.app["session"],
            "post",
            return_value=mock_response,
        )

        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status == 200

    async def test_chat_endpoint_rejects_invalid_json(self, client):
        """Test that /v1/chat/completions rejects invalid JSON."""
        response = await client.post(
            "/v1/chat/completions",
            data="not json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status == 400
        data = await response.json()
        assert "error" in data

    @pytest.mark.parametrize("payload", [b"\xff", b"\xc3", b"\xef\xbb\xbf\xff"])
    async def test_chat_endpoint_rejects_undecodable_json(self, client, mocker, payload):
        upstream = mocker.patch.object(client.server.app["session"], "post")
        response = await client.post(
            "/v1/chat/completions", data=payload,
            headers={"Content-Type": "application/json"},
        )
        assert response.status == 400
        assert await response.json() == {"error": "Invalid JSON"}
        upstream.assert_not_called()

    @pytest.mark.parametrize("backend_state", ["vllm", "sglang"])
    @pytest.mark.parametrize("payload", [b"null", b"[1]", b"5"])
    async def test_chat_endpoint_rejects_a_non_object_json_root(
        self, client, mocker, payload, backend_state
    ):
        """A parsed root that is not a mapping reached ``dict(body)`` or
        ``body.get``: a 500 where the route's own invalid-JSON envelope belongs."""
        mocker.patch.object(backend, "get_backend_state", return_value=backend_state)
        upstream = mocker.patch.object(client.server.app["session"], "post")

        response = await client.post(
            "/v1/chat/completions", data=payload,
            headers={"Content-Type": "application/json"},
        )

        assert response.status == 400
        assert await response.json() == {"error": "Invalid JSON"}
        upstream.assert_not_called()


class TestCountTokensRoute:
    """Integration tests for /v1/messages/count_tokens endpoint."""

    async def test_count_tokens_returns_token_count(self, client, mocker):
        """The seeded stats are what the answer reports, keyed by the digest the route
        computes. The seed used to be a literal that matches nothing, so this asserted
        only that *some* number came back (``CR3-F-02``'s test-validity half)."""
        from buster_ripper import session as session_mod

        body = {"model": "claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "Test"}]}
        mocker.patch.dict(session_mod._stats,
                          {_sid(body): mocker.Mock(input_tokens=1000, avg_latency_ms=500)})

        response = await client.post("/v1/messages/count_tokens", json=body)

        assert response.status == 200
        assert await response.json() == {"input_tokens": 1000}

    async def test_a_billing_nonce_body_finds_the_stats_messages_recorded(self, client,
                                                                          mocker):
        """``CR3-F-02``: ``/v1/messages`` records stats under the post-normalize digest
        while count_tokens looked the raw one up, so the nonce that changes on every
        request made the lookup miss on every documented-client call."""
        from buster_ripper import session as session_mod

        mocker.patch.dict(session_mod._stats, {}, clear=True)
        answer = json.dumps({"id": "m", "usage": {"input_tokens": 1234}}).encode()
        mocker.patch.object(client.server.app["session"], "post", return_value=StubUpstream(
            body=answer, headers={"Content-Type": "application/json"}))
        body = {
            "model": "test", "max_tokens": 100,
            "system": [{"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.50\nsys"}],
            "messages": [{"role": "user", "content": "Hello"}],
        }

        assert (await client.post("/v1/messages", json=body)).status == 200
        response = await client.post("/v1/messages/count_tokens", json=body)

        assert await response.json() == {"input_tokens": 1234}, (
            "count_tokens answered its own estimate: the stats lookup missed")

    async def test_count_tokens_handles_empty_request(self, client):
        """Test that count_tokens handles empty request."""
        response = await client.post(
            "/v1/messages/count_tokens",
            json={},
        )

        assert response.status == 200
        data = await response.json()
        assert "input_tokens" in data

    @pytest.mark.parametrize("payload", [b"\xff", b"\xc3", b"\xef\xbb\xbf\xff"])
    async def test_count_tokens_rejects_undecodable_json(self, client, payload):
        response = await client.post(
            "/v1/messages/count_tokens", data=payload,
            headers={"Content-Type": "application/json"},
        )
        assert response.status == 400
        assert await response.json() == {"error": "Invalid JSON"}

    async def test_count_tokens_syntax_error_keeps_fallback(self, client):
        response = await client.post(
            "/v1/messages/count_tokens", data=b"{",
            headers={"Content-Type": "application/json"},
        )
        assert response.status == 200
        assert "input_tokens" in await response.json()

    @pytest.mark.parametrize("kv, latency", [
        pytest.param(0.85, 100, id="kv-pressure"),
        pytest.param(0.0, 20_000, id="per-session"),
    ])
    async def test_count_tokens_inflates_on_compaction_trigger(self, client, mocker, kv,
                                                               latency):
        """Either arm alone inflates the count, and each case rules the other out.

        The seed is a real digest now — the literal that was here matched nothing, so the
        stats path never ran and the row passed on the kv arm alone. ``per-session`` is the
        branch that ordering defect made unreachable: with the lookup missing,
        ``avg_latency`` stayed 0.0 no matter what the session had done. KV is at 0.0 there,
        so a green cannot be another kv-forced pass.
        """
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

            response = await client.post("/v1/messages/count_tokens", json=body)

            assert response.status == 200
            nudged = int(config.MAX_MODEL_LEN * config.COMPACT_NUDGE_RATIO)
            assert await response.json() == {"input_tokens": nudged}
        finally:
            compaction._kv_cache_usage = original_kv


class TestPassthroughRoute:
    """Integration tests for passthrough endpoint."""

    async def test_passthrough_forwards_requests(self, client, mocker):
        """Test that passthrough forwards requests to upstream."""
        mock_response = _create_mock_response(
            mocker,
            json_data={"models": []},
            headers={"Content-Type": "application/json"},
        )

        mocker.patch.object(
            client.server.app["session"],
            "request",
            return_value=mock_response,
        )

        response = await client.get("/v1/models")

        assert response.status == 200


class TestSessionPooling:
    """Integration tests for session pooling in route handlers."""

    async def test_routes_use_shared_session(self, client, mocker):
        """Test that route handlers use the shared session from app state."""
        # Verify the session exists in app state
        assert "session" in client.server.app

        # Mock a response
        mock_response = _create_mock_response(
            mocker,
            json_data={"content": "test"},
        )

        # Track if session.post was called
        session = client.server.app["session"]
        mocker.patch.object(session, "post", return_value=mock_response)

        # Make a request
        await client.post(
            "/v1/messages",
            json={
                "model": "test",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "test"}],
            },
        )

        # Verify the shared session was used
        session.post.assert_called()

    async def test_session_exists_in_app_state(self, client):
        """Test that a shared session exists in app state."""
        assert "session" in client.server.app
        assert client.server.app["session"] is not None


class TestDumpFailureIsolated:
    """VP-T001-dump-failure-isolated: a raising dump cannot fail the request."""

    async def test_raising_dump_still_yields_upstream_response(
        self, client, mocker, tmp_path
    ):
        """A dump_turn that raises is logged and swallowed; the proxy answers."""
        mocker.patch.object(config, "DUMP_DIR", tmp_path)
        mocker.patch(
            "buster_ripper.routes.messages.dump_turn",
            side_effect=RuntimeError("boom"),
        )
        upstream = _create_mock_response(
            mocker, json_data={"id": "m", "content": [{"type": "text", "text": "Hi"}]},
        )
        mocker.patch.object(client.server.app["session"], "post", return_value=upstream)

        response = await client.post("/v1/messages", json=_MESSAGES_BODY)
        assert response.status == 200


class TestErrorHandling:
    """Integration tests for error handling."""

    async def test_messages_handles_upstream_error(self, client, mocker):
        """Test that /v1/messages handles upstream errors gracefully."""
        mock_response = _create_mock_response(mocker, status=500)

        mocker.patch.object(
            client.server.app["session"],
            "post",
            return_value=mock_response,
        )

        response = await client.post(
            "/v1/messages",
            json={
                "model": "test",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "test"}],
            },
        )

        # Should propagate upstream error
        assert response.status == 500

    async def test_chat_handles_upstream_error(self, client, mocker):
        """Test that /v1/chat/completions handles upstream errors gracefully."""
        mock_response = _create_mock_response(
            mocker,
            status=503,
            json_data={"error": "Service unavailable"},
        )

        mocker.patch.object(
            client.server.app["session"],
            "post",
            return_value=mock_response,
        )

        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "test"}],
            },
        )

        # Should propagate upstream error
        assert response.status == 503

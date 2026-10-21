"""Integration tests for the /v1/chat/completions endpoint."""

import json

import pytest

from buster_ripper import backend
from tests.upstream_stub import StubStream, StubUpstream


class TestChatRoute:
    """Integration tests for /v1/chat/completions endpoint."""

    @pytest.mark.parametrize("payload", [b"data: {\"x\":1}\n\n", b"data: [DONE]\n\n"])
    @pytest.mark.parametrize("content_type", ["text/event-stream", "application/problem+json"])
    async def test_chat_stream_relays_exact_sse_bytes(self, proxy, mocker, payload, content_type):
        upstream = StubStream(
            headers={"Content-Type": content_type},
            chunks=(payload[:4], payload[4:]),
        )
        json_spy = mocker.spy(upstream, "json")
        mocker.patch.object(proxy.server.app["session"], "post", return_value=upstream)
        response = await proxy.post("/v1/chat/completions", json={"stream": True, "messages": []})
        assert response.status == 200
        assert response.headers["Content-Type"] == "text/event-stream"
        assert await response.read() == payload
        assert json_spy.await_count == 0

    @pytest.mark.parametrize("content_type, payload", [
        ("text/plain; charset=utf-8", b"rate limit"),
        ("application/problem+json", b'{"error":"busy"}'),
        (None, b"opaque error"),
    ])
    async def test_chat_stream_upstream_error_retains_representation(
        self, proxy, mocker, content_type, payload,
    ):
        """The stream error arm forwarded ``Content-Type`` alone, so a caller could not
        honor a ``Retry-After`` hint or correlate an ``X-Request-Id``. Every other
        end-to-end name travels with the body, and a body-less error still answers."""
        headers = {"Retry-After": "7", "X-Request-Id": "req_7f3a"}
        if content_type:
            headers["Content-Type"] = content_type
        upstream = StubUpstream(status=429, headers=headers, body=payload)
        json_spy = mocker.spy(upstream, "json")
        mocker.patch.object(proxy.server.app["session"], "post", return_value=upstream)
        response = await proxy.post("/v1/chat/completions", json={"stream": True, "messages": []})
        assert response.status == 429
        assert await response.read() == payload
        assert response.headers["Content-Type"] == (content_type or "application/octet-stream")
        assert response.headers["Retry-After"] == "7"
        assert response.headers["X-Request-Id"] == "req_7f3a"
        assert json_spy.await_count == 0

    async def test_chat_nonstream_compatible_json_transforms(self, proxy, mocker):
        upstream = StubUpstream(
            body=json.dumps(
                {"choices": [{"message": {"content": "<think>reason</think>answer"}}]}
            ).encode(),
            headers={"Content-Type": "application/problem+json"},
        )
        read_spy = mocker.spy(upstream, "read")
        json_spy = mocker.spy(upstream, "json")
        mocker.patch.object(proxy.server.app["session"], "post", return_value=upstream)
        response = await proxy.post("/v1/chat/completions", json={"stream": False, "messages": []})
        assert response.status == 200
        assert json.loads(await response.read())["choices"][0]["message"]["content"] == "answer"
        # The mechanism pin: the non-stream arm reads the payload once and parses those
        # bytes itself, so nothing asks for ``json()``. Transform coverage above,
        # wiring here.
        assert read_spy.await_count == 1
        assert json_spy.await_count == 0

    async def test_chat_endpoint_accepts_json(self, proxy, mocker):
        """Test that /v1/chat/completions accepts valid JSON."""
        mock_response = StubUpstream(
            body=json.dumps(
                {
                    "id": "chat_test",
                    "choices": [{"message": {"content": "Hello"}, "finish_reason": "stop"}],
                }
            ).encode(),
        )

        mocker.patch.object(
            proxy.server.app["session"],
            "post",
            return_value=mock_response,
        )

        response = await proxy.post(
            "/v1/chat/completions",
            json={
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status == 200

    async def test_chat_endpoint_rejects_invalid_json(self, proxy):
        """Test that /v1/chat/completions rejects invalid JSON."""
        response = await proxy.post(
            "/v1/chat/completions",
            data="not json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status == 400
        data = await response.json()
        assert "error" in data

    @pytest.mark.parametrize("payload", [b"\xff", b"\xc3", b"\xef\xbb\xbf\xff"])
    async def test_chat_endpoint_rejects_undecodable_json(self, proxy, mocker, payload):
        upstream = mocker.patch.object(proxy.server.app["session"], "post")
        response = await proxy.post(
            "/v1/chat/completions", data=payload,
            headers={"Content-Type": "application/json"},
        )
        assert response.status == 400
        assert await response.json() == {"error": "Invalid JSON"}
        upstream.assert_not_called()

    @pytest.mark.parametrize("backend_state", ["vllm", "sglang"])
    @pytest.mark.parametrize("payload", [b"null", b"[1]", b"5"])
    async def test_chat_endpoint_rejects_a_non_object_json_root(
        self, proxy, mocker, payload, backend_state
    ):
        """A parsed root that is not a mapping reached ``dict(body)`` or
        ``body.get``: a 500 where the route's own invalid-JSON envelope belongs."""
        mocker.patch.object(backend, "get_backend_state", return_value=backend_state)
        upstream = mocker.patch.object(proxy.server.app["session"], "post")

        response = await proxy.post(
            "/v1/chat/completions", data=payload,
            headers={"Content-Type": "application/json"},
        )

        assert response.status == 400
        assert await response.json() == {"error": "Invalid JSON"}
        upstream.assert_not_called()


    async def test_chat_handles_upstream_error(self, proxy, mocker):
        """Test that /v1/chat/completions handles upstream errors gracefully."""
        mock_response = StubUpstream(
            status=503,
            body=json.dumps({"error": "Service unavailable"}).encode(),
        )

        mocker.patch.object(
            proxy.server.app["session"],
            "post",
            return_value=mock_response,
        )

        response = await proxy.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "test"}],
            },
        )

        # Should propagate upstream error
        assert response.status == 503

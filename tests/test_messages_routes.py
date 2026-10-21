"""Integration tests for the /v1/messages endpoint."""

import json

import pytest

from buster_ripper import config
from tests.upstream_stub import StubUpstream


_MESSAGES_BODY = {
    "model": "test",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "Hello"}],
}


class TestMessagesRoute:
    """Integration tests for /v1/messages endpoint."""

    async def test_messages_endpoint_accepts_json(self, proxy, mocker):
        """Test that /v1/messages accepts valid JSON."""
        # Mock upstream to return success
        mock_response = StubUpstream(
            body=json.dumps(
                {"id": "msg_test", "content": [{"type": "text", "text": "Hello"}]}
            ).encode(),
        )

        mocker.patch.object(
            proxy.server.app["session"],
            "post",
            return_value=mock_response,
        )

        response = await proxy.post(
            "/v1/messages",
            json={
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status == 200

    async def test_messages_endpoint_rejects_invalid_json(self, proxy):
        """Test that /v1/messages rejects invalid JSON."""
        response = await proxy.post(
            "/v1/messages",
            data="not json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status == 400
        data = await response.json()
        assert "error" in data

    @pytest.mark.parametrize("payload", [b"\xff", b"\xc3", b"\xef\xbb\xbf\xff"])
    async def test_messages_endpoint_rejects_undecodable_json(self, proxy, mocker, payload):
        upstream = mocker.patch.object(proxy.server.app["session"], "post")
        response = await proxy.post(
            "/v1/messages", data=payload, headers={"Content-Type": "application/json"},
        )
        assert response.status == 400
        assert await response.json() == {"error": "Invalid JSON"}
        upstream.assert_not_called()

    async def test_messages_endpoint_normalizes_request(self, proxy, mocker):
        """Test that /v1/messages normalizes the request (strips billing header)."""
        mock_response = StubUpstream(
            body=json.dumps(
                {"id": "msg_test", "content": [{"type": "text", "text": "Response"}]}
            ).encode(),
        )

        mocker.patch.object(
            proxy.server.app["session"],
            "post",
            return_value=mock_response,
        )

        # Send request with billing header in system prompt
        response = await proxy.post(
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


class TestSessionPooling:
    """Integration tests for session pooling in route handlers."""

    async def test_routes_use_shared_session(self, proxy, mocker):
        """Test that route handlers use the shared session from app state."""
        # Verify the session exists in app state
        assert "session" in proxy.server.app

        # Mock a response
        mock_response = StubUpstream(
            body=json.dumps({"content": "test"}).encode(),
        )

        # Track if session.post was called
        session = proxy.server.app["session"]
        mocker.patch.object(session, "post", return_value=mock_response)

        # Make a request
        await proxy.post(
            "/v1/messages",
            json={
                "model": "test",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "test"}],
            },
        )

        # Verify the shared session was used
        session.post.assert_called()

    async def test_session_exists_in_app_state(self, proxy):
        """Test that a shared session exists in app state."""
        assert "session" in proxy.server.app
        assert proxy.server.app["session"] is not None


class TestDumpFailureIsolated:
    """VP-T001-dump-failure-isolated: a raising dump cannot fail the request."""

    async def test_raising_dump_still_yields_upstream_response(
        self, proxy, mocker, tmp_path
    ):
        """A dump_turn that raises is logged and swallowed; the proxy answers."""
        mocker.patch.object(config, "DUMP_DIR", tmp_path)
        mocker.patch(
            "buster_ripper.routes.messages.dump_turn",
            side_effect=RuntimeError("boom"),
        )
        upstream = StubUpstream(
            body=json.dumps(
                {"id": "m", "content": [{"type": "text", "text": "Hi"}]}
            ).encode(),
        )
        mocker.patch.object(proxy.server.app["session"], "post", return_value=upstream)

        response = await proxy.post("/v1/messages", json=_MESSAGES_BODY)
        assert response.status == 200


    async def test_messages_handles_upstream_error(self, proxy, mocker):
        """Test that /v1/messages handles upstream errors gracefully."""
        mock_response = StubUpstream(status=500)

        mocker.patch.object(
            proxy.server.app["session"],
            "post",
            return_value=mock_response,
        )

        response = await proxy.post(
            "/v1/messages",
            json={
                "model": "test",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "test"}],
            },
        )

        # Should propagate upstream error
        assert response.status == 500

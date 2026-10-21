"""Integration tests for the passthrough endpoint."""

import json

from tests.upstream_stub import StubUpstream


class TestPassthroughRoute:
    """Integration tests for passthrough endpoint."""

    async def test_passthrough_forwards_requests(self, proxy, mocker):
        """Test that passthrough forwards requests to upstream."""
        mock_response = StubUpstream(
            body=json.dumps({"models": []}).encode(),
            headers={"Content-Type": "application/json"},
        )

        mocker.patch.object(
            proxy.server.app["session"],
            "request",
            return_value=mock_response,
        )

        response = await proxy.get("/v1/models")

        assert response.status == 200

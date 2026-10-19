"""PR1-R4-E1 — a messages request is the client's own bytes unless normalization changed it.

Base forwarded the body it read and re-serialized only inside its transforms. Both
``/v1/messages`` request call sites sent ``json=body``, so a body the client encoded
under a non-UTF-8 charset reached the upstream re-encoded as UTF-8 under the client's own
``Content-Type`` label.
"""

import json

import pytest

from tests.upstream_stub import StubUpstream

_BODY = {"model": "m", "max_tokens": 10,
         "messages": [{"role": "user", "content": "hi"}]}
_JSON = {"Content-Type": "application/json"}
# The per-request block Claude Code puts at system position 0: normalize() removes it.
_NONCE = "x-anthropic-billing-header: cc_version=2.1.50.f15; cch=27acd;\nsys"

_STREAM = [
    pytest.param(False, id="non-stream"),
    pytest.param(True, id="stream"),
]


class TestMessagesRequestBodyBytes:
    @pytest.mark.parametrize("stream", _STREAM)
    async def test_a_utf16_body_forwards_byte_identical_with_its_content_type(
        self, proxy, mocker, stream
    ):
        """The parsed body equals the client's bytes, so no transform ran and the
        client's own encoding is what the upstream must receive."""
        raw = json.dumps({**_BODY, "stream": stream}).encode("utf-16")
        post = mocker.patch.object(proxy.server.app["session"], "post",
                                    return_value=StubUpstream(headers=_JSON))

        response = await proxy.post(
            "/v1/messages", data=raw,
            headers={"Content-Type": "application/json; charset=utf-16"})

        assert response.status == 200
        assert post.call_args.kwargs["data"] == raw
        assert post.call_args.kwargs["headers"]["Content-Type"] == (
            "application/json; charset=utf-16")
        assert "json" not in post.call_args.kwargs, "the body was re-serialized"

    @pytest.mark.parametrize("stream", _STREAM)
    async def test_a_normalized_body_reserializes_with_json_content_type(
        self, proxy, mocker, stream
    ):
        """The other half: once normalization strips the nonce the client's bytes no
        longer describe the body, so it is re-encoded and relabelled."""
        raw = json.dumps({**_BODY, "stream": stream, "system": _NONCE}).encode()
        post = mocker.patch.object(proxy.server.app["session"], "post",
                                    return_value=StubUpstream(headers=_JSON))

        response = await proxy.post(
            "/v1/messages", data=raw,
            headers={"Content-Type": "application/json; charset=utf-8"})

        assert response.status == 200
        sent = json.loads(post.call_args.kwargs["data"])
        assert sent["system"] == "sys", "the normalized body was relayed raw"
        names = [k.lower() for k in post.call_args.kwargs["headers"]]
        assert names.count("content-type") == 1, "the stale label survived beside the new one"
        assert post.call_args.kwargs["headers"]["Content-Type"] == "application/json"

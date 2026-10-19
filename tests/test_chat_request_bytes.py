"""PR1-R4-6 — a chat request is the client's own bytes unless a transform changed it.

Base forwarded the request body it read (``buster_ripper_legacy.py:812-847``) and
re-serialized only inside the eval transform. The route sent ``json=body`` on every
call, so a body the client encoded under a non-UTF-8 charset reached the upstream
re-encoded as UTF-8 under the client's own ``Content-Type`` label.
"""

import json

from buster_ripper import config
from tests.upstream_stub import StubUpstream

_BODY = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
_JSON = {"Content-Type": "application/json"}


class TestChatRequestBodyBytes:
    async def test_a_utf16_body_forwards_byte_identical_with_its_content_type(
        self, proxy, mocker
    ):
        """The parsed body equals the client's bytes, so no transform ran and the
        client's own encoding is what the upstream must receive."""
        raw = json.dumps(_BODY).encode("utf-16")
        post = mocker.patch.object(proxy.server.app["session"], "post",
                                    return_value=StubUpstream(headers=_JSON))

        response = await proxy.post(
            "/v1/chat/completions", data=raw,
            headers={"Content-Type": "application/json; charset=utf-16"})

        assert response.status == 200
        assert post.call_args.kwargs["data"] == raw
        assert post.call_args.kwargs["headers"]["Content-Type"] == (
            "application/json; charset=utf-16")
        assert "json" not in post.call_args.kwargs, "the body was re-serialized"

    async def test_a_transformed_body_reserializes_with_json_content_type(
        self, proxy, mocker
    ):
        """The other half: once the eval transform removes a field the client's bytes
        no longer describe the body, so it is re-encoded and relabelled."""
        mocker.patch.object(config, "EVAL_MODE", True)
        raw = json.dumps({**_BODY, "max_gen_toks": 5}).encode()
        post = mocker.patch.object(proxy.server.app["session"], "post",
                                    return_value=StubUpstream(headers=_JSON))

        response = await proxy.post(
            "/v1/chat/completions", data=raw,
            headers={"Content-Type": "application/json; charset=utf-8"})

        assert response.status == 200
        sent = json.loads(post.call_args.kwargs["data"])
        assert sent == _BODY
        names = [k.lower() for k in post.call_args.kwargs["headers"]]
        assert names.count("content-type") == 1, "the stale label survived beside the new one"
        assert post.call_args.kwargs["headers"]["Content-Type"] == "application/json"

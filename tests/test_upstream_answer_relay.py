"""A proxied answer is the upstream's answer: its status, bytes and headers.

``CR2-F-1`` every non-stream ``/v1/chat/completions`` answer carries the upstream's
end-to-end headers on both arms — the body a transform rewrote and the body it left alone —
and only the rewritten arm owes a re-encode. ``CR2-F-2``'s relay half: a well-formed answer
that is not JSON is relayed as upstream sent it. ``TP-1``'s redirect half: a 3xx reaches the
caller with its ``Location`` and body at every boundary, never as a hop to follow.

Every claim is driven through the real app, so each assertion reads bytes and headers that
reached a socket.
"""

import json

import pytest

from buster_ripper import backend, config
from tests.relay_bodies import (
    _CHAT,
    _JSON_HEADERS,
    _MESSAGES,
    _OVERFLOW,
    _answer,
)
from tests.upstream_stub import StubUpstream


class TestTransformPathHeaderParity:
    """CR2-F-1: both return paths of the response transform relay the upstream's headers."""

    @pytest.mark.parametrize("content, rewritten", [
        pytest.param("plain answer", False, id="unfixed-body"),
        pytest.param("<think>why</think>answer", True, id="fixed-body"),
    ])
    async def test_upstream_end_to_end_names_reach_the_caller(self, proxy, mocker, content,
                                                             rewritten):
        """Both paths rebuilt the answer with ``Content-Type`` alone, dropping the
        correlation id and retry hint; ``rewritten`` is derived from the body.
        """
        mocker.patch.object(config, "EVAL_MODE", False)
        mocker.patch.object(proxy.server.app["session"], "post",
                            return_value=StubUpstream(headers=_JSON_HEADERS, body=_answer(content)))

        response = await proxy.post("/v1/chat/completions", json=_CHAT)
        payload = json.loads(await response.read())

        expected = "answer" if rewritten else content
        assert response.status == 200
        assert payload["choices"][0]["message"]["content"] == expected
        for name, value in _JSON_HEADERS.items():
            assert response.headers[name] == value, f"{name} was dropped from the answer"

    async def test_a_rewritten_body_carries_the_length_of_what_is_sent(self, proxy, mocker):
        """The rewritten body is re-serialized, so the relayed ``content-length`` must be
        the derived one: a lying upstream length would leave the caller waiting. The indented
        upstream body is the second witness — the answer comes back compact.
        """
        mocker.patch.object(config, "EVAL_MODE", False)
        payload = {"id": "c1", "choices": [{"message": {"content": "<think>why</think>answer"}}]}
        body = json.dumps(payload, indent=2).encode()
        mocker.patch.object(proxy.server.app["session"], "post",
                            return_value=StubUpstream(headers={**_JSON_HEADERS,
                                                               "Content-Length": "999"},
                                                      body=body))

        response = await proxy.post("/v1/chat/completions", json=_CHAT)
        received = await response.read()

        assert json.loads(received)["choices"][0]["message"]["content"] == "answer"
        assert received != body, "the rewritten answer was relayed as upstream sent it"
        assert len(received) != 999, "the fixture would not catch a relayed length"
        assert int(response.headers["Content-Length"]) == len(received)


class TestChatAnswerBytes:
    """CR2-F-1: an answer no transform rewrote is the upstream's own bytes.

    The non-stream relay parsed the answer and then always re-encoded it with
    ``json.dumps``, so every caller read the proxy's spacing and key order instead
    of the upstream's — and the length changed with them. Base re-serialized only
    the arm it had rewritten and relayed the rest verbatim
    (``buster_ripper_legacy.py:882-892``).
    """

    async def test_an_untransformed_answer_is_byte_identical(self, proxy, mocker):
        """Non-canonical spacing is the witness: a re-encode normalizes it away."""
        mocker.patch.object(config, "EVAL_MODE", False)
        body = b'{"id": "c1",  "choices": [{"message": {"content": "plain answer"}}]}'
        mocker.patch.object(proxy.server.app["session"], "post",
                            return_value=StubUpstream(headers=_JSON_HEADERS, body=body))

        response = await proxy.post("/v1/chat/completions", json=_CHAT)

        assert response.status == 200
        assert await response.read() == body, "the answer was re-encoded with nothing changed"


class TestNonJsonRelay:
    """CR2-F-2: an answer that is not JSON is the upstream's to relay (R1 hostile corpus).

    These replace the witness that used to pin the synthetic 502 as the contract
    (``test_routes_integration.py``'s
    ``test_chat_upstream_undecodable_json_is_upstream_error``). Base fell through to the
    raw relay instead (``buster_ripper_legacy.py:885-892``), so a gateway's HTML 502 and
    a plaintext 429 keep their status, their bytes and their headers here.
    """

    @pytest.mark.parametrize("case", [
        pytest.param((502, "text/html; charset=utf-8",
                      b"<html><body>502 Bad Gateway</body></html>"), id="gateway-html-502"),
        pytest.param((429, "text/plain", b"rate limit exceeded, retry later\n"),
                     id="plaintext-429"),
        pytest.param((200, "application/json", b'{"choices": ['), id="truncated-json"),
        pytest.param((200, "application/json", b""), id="empty-body"),
        pytest.param((503, "application/json", b"\xff\xfe"), id="undecodable-bytes"),
    ])
    async def test_upstream_status_bytes_and_headers_all_travel(self, proxy, mocker, case):
        """The synthetic 502 erased the upstream's status, bytes and headers at once."""
        status, content_type, payload = case
        mocker.patch.object(proxy.server.app["session"], "post",
                            return_value=StubUpstream(status=status,
                                                      headers={"Content-Type": content_type,
                                                               "Retry-After": "7"},
                                                      body=payload))

        response = await proxy.post("/v1/chat/completions", json=_CHAT)

        assert response.status == status, "the proxy answered for the upstream"
        assert await response.read() == payload, "the upstream's bytes were replaced"
        assert response.headers["Content-Type"] == content_type
        assert response.headers["Retry-After"] == "7"

    @pytest.mark.parametrize("payload", [b"null", b"[1, 2]", b'"a string"'])
    async def test_json_that_is_not_an_object_relayed_verbatim(self, proxy, mocker, payload):
        """The boundary the corpus ends at: these parse, so they take the transform path,
        whose ``resp_data.get`` raises and is swallowed as it always did.
        """
        mocker.patch.object(proxy.server.app["session"], "post",
                            return_value=StubUpstream(headers=_JSON_HEADERS, body=payload))

        response = await proxy.post("/v1/chat/completions", json=_CHAT)

        assert response.status == 200
        assert json.loads(await response.read()) == json.loads(payload)
        assert response.headers["X-Request-Id"] == "req_7f3a"


class TestOverflowArmPreserved:
    """The read-once mechanism must not sever the retry contract (behavior unchanged)."""

    async def test_a_parseable_4xx_overflow_still_drives_the_retry_loop(self, proxy, mocker):
        """``_raise_chat_overflow`` now takes the product of one ``read()`` plus a parse;
        if that wiring broke, the overflow would go upstream-only — three upstream calls is
        what the retry loop owes (``max_retries = 2``).
        """
        mocker.patch.object(backend, "get_backend_state", return_value="sglang")
        post = mocker.patch.object(proxy.server.app["session"], "post",
                                   return_value=StubUpstream(status=400,
                                                             headers=_JSON_HEADERS,
                                                             body=_OVERFLOW))

        response = await proxy.post("/v1/chat/completions", json=_CHAT)

        assert post.call_count == 3, "the overflow retry loop did not run"
        assert response.status == 400


class TestNonObjectOverflowAnswer:
    """PR1-R4-A1: a non-object upstream answer is relayed, not a proxy 500.

    ``_raise_chat_overflow`` read ``.get`` on whatever ``json.loads`` produced, so a
    JSON string or list carrying the overflow text raised ``AttributeError`` out of the
    handler instead of relaying the upstream's error.
    """

    @pytest.mark.parametrize("payload", [
        b'"Context overflow: 5000 > 4096"',
        b'[{"error": {"message": "Context overflow: 5000 > 4096"}}]',
    ], ids=["string-root", "list-root"])
    async def test_a_non_object_overflow_answer_relays(self, proxy, mocker, payload):
        mocker.patch.object(backend, "get_backend_state", return_value="sglang")
        mocker.patch.object(proxy.server.app["session"], "post",
                            return_value=StubUpstream(status=400, headers=_JSON_HEADERS,
                                                      body=payload))

        response = await proxy.post("/v1/chat/completions", json=_CHAT)

        assert response.status == 400, "the proxy answered the upstream's payload with a 500"
        assert await response.read() == payload


_REDIRECT = {"Location": "https://elsewhere.example/next", "Content-Type": "text/plain"}
_REDIRECT_BODY = b"moved"


class TestUpstreamRedirect:
    """TP-1: a 3xx is the upstream's answer at every boundary, not a hop to follow.

    Each call asks for base's posture, and the streaming arms relay the redirect instead
    of preparing an SSE head over it.
    """

    @pytest.mark.parametrize("arm", [
        pytest.param(("request", "GET", "/v1/models", None), id="passthrough"),
        pytest.param(("post", "POST", "/v1/messages", _MESSAGES), id="messages-non-stream"),
        pytest.param(("post", "POST", "/v1/messages", {**_MESSAGES, "stream": True}),
                     id="messages-stream"),
        pytest.param(("post", "POST", "/v1/chat/completions", _CHAT), id="chat-non-stream"),
        pytest.param(("post", "POST", "/v1/chat/completions", {**_CHAT, "stream": True}),
                     id="chat-stream"),
    ])
    async def test_the_redirect_reaches_the_caller_as_sent(self, proxy, mocker, arm):
        attr, method, path, body = arm
        call = mocker.patch.object(proxy.server.app["session"], attr,
                                   return_value=StubUpstream(status=302, headers=_REDIRECT,
                                                             body=_REDIRECT_BODY))
        kwargs = {"json": body} if body is not None else {}

        response = await proxy.request(method, path, allow_redirects=False, **kwargs)

        assert response.status == 302, "the proxy answered a redirect with a 2xx"
        assert response.headers["Location"] == _REDIRECT["Location"]
        assert await response.read() == _REDIRECT_BODY
        assert call.call_args.kwargs["allow_redirects"] is False, (
            "the upstream call would follow the redirect before the handler saw it")

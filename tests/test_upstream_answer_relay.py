"""CR2 wave — a proxied answer is the upstream's answer, and a proxy error says nothing back.

Three claims, one per routed finding, all driven through the real app so every
assertion reads bytes and headers that actually reached a socket:

``CR2-F-1`` every non-stream ``/v1/chat/completions`` answer carries the upstream's
end-to-end response headers on **both** arms — the body a transform rewrote, and the
body it left alone — and only the rewritten arm owes a re-encode: the untouched
answer is the upstream's own bytes. The hop-by-hop census for this boundary lives in
``test_header_parity.py``'s ingress sites; what is unique here is the two-arms split,
which only a parseable JSON answer can show.

``CR2-F-2`` a well-formed answer that is not JSON — a gateway's HTML 502, a plaintext
429 — is relayed as upstream sent it: its status, its bytes, its headers. Base fell
through to that relay (``buster_ripper_legacy.py:885-892``); the branch's
``except (json.JSONDecodeError, UnicodeDecodeError)`` arm replaced it with a synthetic
502 that erased the upstream's answer. The same finding's other half: a genuine
transport failure keeps its 502, but its body is static, because ``str(e)`` named the
internal upstream host and port to anyone who probed us. That arm exists on three
routes, so all three are answered here — the leak is the class, not the line.

``CR2-F-3`` ``--preserved-thinking`` is on in the shipped launcher
(``serve_buster_ripper.sh:30``), and its contract is that every forwarded
``/v1/messages`` request to a backend that honors the pair carries
``enable_thinking=true``/``clear_thinking=false``.
``dict.setdefault`` returns the *existing* value, so a client that sent
``"chat_template_kwargs": null`` — or a list, or a scalar — made the nested
``setdefault`` raise ``AttributeError``: HTTP 500, no upstream call, no injected pair.
The guard replaces the non-mapping instead, which is the only reading that keeps the
guarantee; skipping would forward the ``null`` with nothing injected, and no settled
authority rejects a null-valued optional field. The flag-off arm is pinned untouched
for the same reason: a guard that wrote on the off arm would be a behavior change the
finding never asked for. PR1-R5 narrows the guarantee to the backends that honor it:
SGLang v0.5.9's ``/v1/messages`` discards ``chat_template_kwargs``, so nothing is
injected there, while vLLM's Anthropic endpoint supports the pair.

``_relay_after_commit``'s post-commit contract (T013 accepted) is not re-witnessed
here — ``test_stream_relay_commit.py`` owns it, and the static-body rule stops at the
pre-commit arms.

``CR3-F-04`` (the CR3 wave) extends the same rule to the decode step at every boundary
that has one, in both directions, because the two directions are asymmetric and the code
must not be written as though they matched. Response side: a payload that is not valid
UTF-8 raises ``UnicodeDecodeError``, which is not a ``ClientError``, so the strict
``text()`` at the two ``/v1/messages`` arms escaped their single arm and the caller got
aiohttp's own 500 with a traceback instead of the upstream's status and bytes — an
unknown *response* charset is not on that path at all, since ``get_encoding()``
suppresses it and falls back to UTF-8. Request side: an unknown *client* charset raises
``LookupError`` from ``bytes.decode()``, which no arm named either.

``TP-1`` the transport itself is part of the answer: a fired upstream deadline reaches
the route's own 502 envelope (``asyncio.TimeoutError`` is not a ``ClientError``, so the
old arms let aiohttp generate its own 504), and a redirect is relayed with its
``Location`` and body at every boundary (base's ``httpx`` client followed none, while
aiohttp follows them by default).
"""

import asyncio
import json
import logging

import aiohttp
import pytest

from buster_ripper import backend, config
from tests.upstream_stub import StubUpstream

_CHAT = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
_MESSAGES = {"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]}

# The names a caller needs back and the branch dropped on the floor: a correlation id,
# a retry hint, and a rate-limit budget. All end-to-end, none hop-by-hop.
_RELAY = {"X-Request-Id": "req_7f3a", "Retry-After": "7", "X-RateLimit-Remaining": "3"}
_JSON_HEADERS = {"Content-Type": "application/json", **_RELAY}

# SGLang's overflow shape, matching config.CONTEXT_OVERFLOW_RE ("Context overflow: 5 > 4").
_OVERFLOW = json.dumps({"error": {"message": "Context overflow: 50000 > 4096"}}).encode()


def _upstream_authority():
    """What ``str(e)`` leaked: the configured upstream's host:port, without its scheme."""
    return config.UPSTREAM.split("//", 1)[1]


def _sent_body(post_call):
    """The JSON body that actually went out to the upstream stand-in."""
    kwargs = post_call.kwargs
    return json.loads(kwargs["data"]) if "data" in kwargs else kwargs["json"]


def _answer(content):
    return json.dumps({"id": "c1", "choices": [{"message": {"content": content}}]}).encode()


class TestTransformPathHeaderParity:
    """CR2-F-1: both return paths of the response transform relay the upstream's headers."""

    @pytest.mark.parametrize("content, rewritten", [
        pytest.param("plain answer", False, id="unfixed-body"),
        pytest.param("<think>why</think>answer", True, id="fixed-body"),
    ])
    async def test_upstream_end_to_end_names_reach_the_caller(self, proxy, mocker, content,
                                                             rewritten):
        """Both paths rebuilt the answer with ``Content-Type`` alone, so a caller could
        not correlate a request id or honor a retry hint. ``rewritten`` records which
        path the answer takes, derived from the body rather than asserted blind."""
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
        """The transform path's body is not upstream's — it is re-serialized, so the
        relayed ``content-length`` must be the dropped-and-derived one: a lying upstream
        length would leave the caller waiting for bytes that never come. The indented
        upstream body is the second witness: the answer comes back compact, which only the
        re-encode arm can produce."""
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


class TestEvalChatTemplateKwargsGuard:
    """PR1-R4-4: a non-mapping ``chat_template_kwargs`` is replaced, not skipped.

    The eval transform's ``setdefault`` returned the client's ``null``/list/scalar and
    the nested ``setdefault`` raised ``AttributeError``, which the transform's broad
    handler swallowed: the request went upstream with no pair and no budget.
    """

    @pytest.mark.parametrize("value", [None, [], "x"], ids=["null", "list", "scalar"])
    async def test_a_non_mapping_is_replaced_and_the_pair_goes_out(self, proxy, mocker, value):
        mocker.patch.object(config, "EVAL_MODE", True)
        mocker.patch.object(config, "EVAL_PROFILE", config.PROFILES["glm47"])
        mocker.patch.object(config, "EVAL_THINKING", True)
        mocker.patch.object(config, "EVAL_THINKING_BUDGET", 128)
        post = mocker.patch.object(proxy.server.app["session"], "post",
                                    return_value=StubUpstream())

        response = await proxy.post("/v1/chat/completions",
                                    json={**_CHAT, "chat_template_kwargs": value})

        assert response.status == 200, "the route 500ed on a caller-supplied field"
        sent = _sent_body(post.call_args)
        assert sent["chat_template_kwargs"] == {
            "enable_thinking": True, "thinking_budget": 128}

    async def test_a_mapping_from_the_client_is_merged_not_flattened(self, proxy, mocker):
        """Replacement is for non-mappings only: ``setdefault`` semantics on a real
        mapping are unchanged, including a client's own budget."""
        mocker.patch.object(config, "EVAL_MODE", True)
        mocker.patch.object(config, "EVAL_PROFILE", config.PROFILES["glm47"])
        mocker.patch.object(config, "EVAL_THINKING", True)
        mocker.patch.object(config, "EVAL_THINKING_BUDGET", 128)
        post = mocker.patch.object(proxy.server.app["session"], "post",
                                    return_value=StubUpstream())

        await proxy.post("/v1/chat/completions", json={
            **_CHAT, "chat_template_kwargs": {"thinking_budget": 64}})

        sent = _sent_body(post.call_args)
        assert sent["chat_template_kwargs"] == {
            "thinking_budget": 64, "enable_thinking": True}


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
        """The synthetic 502 erased all three at once: upstream's gateway status became
        the proxy's, its HTML became a JSON error, and its headers vanished."""
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
        whose ``resp_data.get`` raises and is swallowed as it always did — and a JSON
        ``null`` parses to ``None``, which shares the relay rather than inventing a body.
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
        """``_raise_chat_overflow`` used to be handed ``resp.json()``'s product; it is now
        handed the product of one ``read()`` plus a parse. If that wiring broke, the
        overflow would never be seen and the answer would go upstream-only: three
        upstream calls is what the retry loop owes (``max_retries = 2``)."""
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


class TestLeakFreeErrorBodies:
    """CR2-F-2: no client-facing error body on any proxy route embeds exception text."""

    @staticmethod
    def _refuse_the_upstream(proxy, mocker, attr):
        """One connection failure whose message carries the upstream, as aiohttp's own
        ClientConnectorError text does."""
        mocker.patch.object(proxy.server.app["session"], attr, side_effect=aiohttp.ClientConnectionError(
            f"Cannot connect to host {_upstream_authority()} ssl:default "
            "[Errno 111] Connection refused"))

    @pytest.mark.parametrize("arm", [
        pytest.param(("post", "POST", "/v1/chat/completions", _CHAT), id="chat-non-stream"),
        pytest.param(("post", "POST", "/v1/messages", {**_MESSAGES, "stream": True}),
                     id="messages-stream"),
        pytest.param(("post", "POST", "/v1/messages", _MESSAGES), id="messages-non-stream"),
        pytest.param(("request", "GET", "/v1/models", None), id="passthrough"),
    ])
    async def test_the_502_names_no_upstream_topology(self, proxy, mocker, arm):
        attr, method, path, body = arm
        self._refuse_the_upstream(proxy, mocker, attr)
        kwargs = {"json": body} if body is not None else {}

        response = await proxy.request(method, path, **kwargs)
        text = (await response.read()).decode()

        assert response.status == 502
        assert _upstream_authority() not in text, f"upstream host:port reached the client: {text}"
        assert "Cannot connect" not in text, f"exception text reached the client: {text}"
        assert json.loads(text) == {"error": "Upstream request failed"}

    async def test_the_exception_is_still_for_the_operator(self, proxy, mocker, caplog):
        """Taking the text out of the body must not take it out of the record: a 502
        with no cause anywhere is undebuggable, and four arms share this one contract."""
        self._refuse_the_upstream(proxy, mocker, "post")

        with caplog.at_level(logging.DEBUG, logger="buster-ripper"):
            response = await proxy.post("/v1/chat/completions", json=_CHAT)

        assert response.status == 502
        lines = [r.getMessage() for r in caplog.records if "Upstream error" in r.getMessage()]
        assert len(lines) == 1
        assert _upstream_authority() in lines[0], "the cause never reached the log"


class TestPreservedThinkingGuard:
    """CR2-F-3: the flag's pair goes out on every forwarded body a backend honors it for."""

    @pytest.mark.parametrize("value", [
        pytest.param(None, id="null"),
        pytest.param([], id="empty-list"),
        pytest.param([{"enable_thinking": False}], id="list-of-mappings"),
        pytest.param("enable", id="string-scalar"),
        pytest.param(7, id="number-scalar"),
        pytest.param(True, id="boolean-scalar"),
    ])
    async def test_a_non_mapping_is_replaced_and_the_pair_goes_out(self, proxy, mocker, value):
        """The shipped launcher default, at the route level: pre-guard this is an
        ``AttributeError`` out of ``handle_messages`` — HTTP 500 and no upstream call."""
        mocker.patch.object(config, "PRESERVED_THINKING", True)
        post = mocker.patch.object(proxy.server.app["session"], "post",
                                   return_value=StubUpstream())

        response = await proxy.post("/v1/messages", json={**_MESSAGES,
                                                          "chat_template_kwargs": value})

        assert response.status == 200, "the route 500ed on a caller-supplied field"
        assert post.call_count == 1, "the request never reached the upstream"
        assert _sent_body(post.call_args)["chat_template_kwargs"] == {
            "enable_thinking": True, "clear_thinking": False}

    @pytest.mark.parametrize("value, expected", [
        pytest.param({"thinking_budget": 128},
                     {"thinking_budget": 128, "enable_thinking": True, "clear_thinking": False},
                     id="client-kwargs-kept"),
        pytest.param({"enable_thinking": False},
                     {"enable_thinking": False, "clear_thinking": False},
                     id="client-override-still-wins"),
    ])
    async def test_a_mapping_from_the_client_is_merged_not_flattened(self, proxy, mocker, value,
                                                                     expected):
        """The replacement is for non-mappings only: ``setdefault`` semantics on a real
        mapping are what they always were, including a client that deliberately turns
        thinking off. A build that overwrote the pair instead of defaulting it fails
        here, and that would be a behavior change no finding in this batch asked for."""
        mocker.patch.object(config, "PRESERVED_THINKING", True)
        post = mocker.patch.object(proxy.server.app["session"], "post",
                                   return_value=StubUpstream())

        await proxy.post("/v1/messages", json={**_MESSAGES, "chat_template_kwargs": value})

        assert _sent_body(post.call_args)["chat_template_kwargs"] == expected

    @pytest.mark.parametrize("backend_type", ["vllm", "unknown"])
    async def test_an_absent_field_still_gets_the_pair(self, proxy, mocker, backend_type):
        """The guarantee the flag's own help text states, for the ordinary request, on both
        backends whose Anthropic endpoint honors the pair."""
        mocker.patch.object(config, "PRESERVED_THINKING", True)
        mocker.patch.object(backend, "get_backend_state", return_value=backend_type)
        post = mocker.patch.object(proxy.server.app["session"], "post",
                                   return_value=StubUpstream())

        await proxy.post("/v1/messages", json=_MESSAGES)

        assert _sent_body(post.call_args)["chat_template_kwargs"] == {
            "enable_thinking": True, "clear_thinking": False}

    @pytest.mark.parametrize("value", [
        pytest.param(None, id="null"),
        pytest.param("enable", id="scalar"),
        pytest.param({"thinking_budget": 1}, id="mapping"),
    ])
    async def test_the_flag_off_arm_forwards_the_body_untouched(self, proxy, mocker, value):
        """No cost on the off arm: with the flag off this route must not write the key,
        so a build that moves the injection outside the guard fails here."""
        mocker.patch.object(config, "PRESERVED_THINKING", False)
        post = mocker.patch.object(proxy.server.app["session"], "post",
                                   return_value=StubUpstream())

        await proxy.post("/v1/messages", json={**_MESSAGES, "chat_template_kwargs": value})

        assert _sent_body(post.call_args)["chat_template_kwargs"] == value


class TestPreservedThinkingSglangGate:
    """PR1-R5: SGLang v0.5.9's ``/v1/messages`` discards ``chat_template_kwargs``.

    The flag's pair was injected into that body anyway, advertising an effect the endpoint
    never applies. The gate skips SGLang only: a client-sent mapping still travels as sent,
    and the vLLM/unknown arms above keep their pair.
    """

    async def test_sglang_gets_no_injected_pair(self, proxy, mocker):
        mocker.patch.object(config, "PRESERVED_THINKING", True)
        mocker.patch.object(backend, "get_backend_state", return_value="sglang")
        post = mocker.patch.object(proxy.server.app["session"], "post",
                                   return_value=StubUpstream())

        response = await proxy.post("/v1/messages", json=_MESSAGES)

        assert response.status == 200
        assert "chat_template_kwargs" not in _sent_body(post.call_args)

    async def test_a_client_sent_mapping_is_forwarded_unchanged(self, proxy, mocker):
        mocker.patch.object(config, "PRESERVED_THINKING", True)
        mocker.patch.object(backend, "get_backend_state", return_value="sglang")
        value = {"enable_thinking": False}
        post = mocker.patch.object(proxy.server.app["session"], "post",
                                   return_value=StubUpstream())

        await proxy.post("/v1/messages",
                         json={**_MESSAGES, "chat_template_kwargs": value})

        assert _sent_body(post.call_args)["chat_template_kwargs"] == value


_BAD_CODEC = "application/json; charset=x-not-a-real-codec"
_UNDECODABLE = b"\xff\xfe\x00garbage"
_JUNK = {"Content-Type": "application/json"}


class TestUnknownClientCodecArm:
    """CR3-F-04(c): a client naming a codec Python lacks gets the route's own 400."""

    @pytest.mark.parametrize("path", [
        pytest.param("/v1/messages", id="messages"),
        pytest.param("/v1/chat/completions", id="chat-completions"),
        pytest.param("/v1/messages/count_tokens", id="count-tokens"),
    ])
    async def test_the_unknown_codec_never_reaches_the_caller_as_a_500(self, proxy, mocker,
                                                                       path):
        """``web_request.py:689`` hands ``charset`` to ``bytes.decode()`` and that raises
        ``LookupError`` — not a ``UnicodeDecodeError``, so no arm named it, the handler
        never returned, and the caller got aiohttp's generated 500 with the traceback.
        The route's body is well-formed here: what is broken is the name it arrived under,
        which is why the two neighbouring arms (a syntax error, undecodable bytes) settle
        it and this one escaped."""
        upstream = mocker.patch.object(proxy.server.app["session"], "post",
                                       return_value=StubUpstream(headers=_JUNK))

        response = await proxy.post(path, data=json.dumps(_MESSAGES).encode(),
                                    headers={"Content-Type": _BAD_CODEC})

        assert response.status == 400, f"{path} answered {response.status} for a codec name"
        assert json.loads(await response.read()) == {"error": "Invalid JSON"}
        upstream.assert_not_called()


class TestMessagesStreamErrorRelay:
    """PR1-R4-2: the streaming error arm is the upstream's answer, headers and all.

    The arm decoded the body and answered ``application/json`` with its own text, so
    a 429's ``Retry-After`` and correlation id never reached the caller and a
    plaintext error lost its label.
    """

    async def test_a_429_relays_its_status_body_and_end_to_end_headers(self, proxy, mocker):
        headers = {"Retry-After": "7", "X-Request-Id": "req_7f3a",
                   "Content-Type": "text/plain; charset=utf-8"}
        mocker.patch.object(proxy.server.app["session"], "post",
                            return_value=StubUpstream(status=429, headers=headers,
                                                      body=b"slow down"))

        response = await proxy.post("/v1/messages", json={**_MESSAGES, "stream": True})

        assert response.status == 429
        for name, value in headers.items():
            assert response.headers[name] == value, f"{name} was dropped from the error"
        assert await response.read() == b"slow down", "the upstream's bytes were replaced"


class TestUndecodableUpstreamAnswer:
    """CR3-F-04(a)/(b): a payload the response cannot decode is still the upstream's answer."""

    async def test_the_non_stream_probe_decodes_the_body_it_is_only_checking(self, proxy,
                                                                            mocker):
        """The probe parses text out of a body the relay then sends as bytes, so a strict
        ``decode()`` here made undecodable bytes an argument the handler raised on its way
        to forwarding them. The overflow text is ASCII, so it survives replacement."""
        mocker.patch.object(proxy.server.app["session"], "post",
                            return_value=StubUpstream(status=429, headers=_JUNK,
                                                      body=_UNDECODABLE))

        response = await proxy.post("/v1/messages", json=_MESSAGES)

        assert response.status == 429, "the proxy answered the upstream's payload with a 500"
        assert await response.read() == _UNDECODABLE

    async def test_a_stream_error_body_that_no_charset_can_decode_still_answers_itself(
        self, proxy, mocker
    ):
        """The arm relays the upstream's status, headers and bytes; the probe decodes
        those bytes with replacement, so no charset can raise out of the loop."""
        mocker.patch.object(proxy.server.app["session"], "post",
                            return_value=StubUpstream(status=503, headers=_JUNK,
                                                      body=_UNDECODABLE))

        response = await proxy.post("/v1/messages", json={**_MESSAGES, "stream": True})

        assert response.status == 503, "the proxy answered the upstream's payload with a 500"
        assert await response.read() == _UNDECODABLE

    async def test_an_unknown_response_charset_over_decodable_bytes_is_not_a_raise(
        self, proxy, mocker
    ):
        """The probe decodes bytes with replacement, so the upstream's unknown charset
        label cannot raise out of the loop now that the label is relayed: a ``.text``
        read here would be the ``LookupError`` the relay's own headers introduce."""
        mocker.patch.object(proxy.server.app["session"], "post",
                            return_value=StubUpstream(
                                status=503,
                                headers={"Content-Type": _BAD_CODEC},
                                body=b'{"error":"boom"}'))

        response = await proxy.post("/v1/messages", json={**_MESSAGES, "stream": True})

        assert response.status == 503, "the proxy answered the upstream's payload with a 500"
        assert json.loads(await response.read()) == {"error": "boom"}


class TestUpstreamTimeout:
    """TP-1: a fired upstream deadline is the route's own 502, never aiohttp's 504.

    ``asyncio.TimeoutError`` is a ``TimeoutError``, not a ``ClientError``: the arm on
    each of these routes named only the latter, so the exception left the handler and
    aiohttp's protocol generated its own 504 in place of the route's envelope.
    """

    @pytest.mark.parametrize("arm", [
        pytest.param(("request", "GET", "/v1/models", None), id="passthrough"),
        pytest.param(("post", "POST", "/v1/messages", _MESSAGES), id="messages-non-stream"),
        pytest.param(("post", "POST", "/v1/chat/completions", _CHAT), id="chat"),
    ])
    async def test_a_fired_deadline_answers_the_routes_own_502(self, proxy, mocker, arm):
        attr, method, path, body = arm
        mocker.patch.object(proxy.server.app["session"], attr,
                            side_effect=asyncio.TimeoutError())
        kwargs = {"json": body} if body is not None else {}

        response = await proxy.request(method, path, **kwargs)

        assert response.status == 502, "aiohttp answered for the route with its own 504"
        assert json.loads(await response.read()) == {"error": "Upstream request failed"}


_REDIRECT = {"Location": "https://elsewhere.example/next", "Content-Type": "text/plain"}
_REDIRECT_BODY = b"moved"


class TestUpstreamRedirect:
    """TP-1: a 3xx is the upstream's answer at every boundary, not a hop to follow.

    Base's ``httpx`` client followed none; aiohttp follows them by default, so each
    call now asks for the base posture. The streaming arms also relay the redirect
    instead of preparing an SSE head over it.
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

"""CR3 wave (T018) — the transport contract the base stack had: no body cap, no stream deadline.

Two divergences the Starlette→aiohttp migration introduced, both witnessable only
through the real app on a real socket with a real client session:

``CR3-F-01`` ``web.Application`` caps a request body at 1 MiB by default, and the cap
fires at the handler's *first* read — ahead of ``normalize()`` and ``--strip-images``.
Base imposed no cap, so a body base accepted has to be accepted here too, at every route
class. The ``--strip-images`` case is the sharpest: the feature that exists to move image
bytes out of the forwarded body never saw a request over 1 MiB. The repair deletes the cap
rather than choosing a number, so neither reading of what the number counts (the ``>=``
boundary, or pre- or post-normalization size) has to be ruled on.

``CR3-F-03`` the shared session's ``total=600`` armed the *streaming* relay's whole
lifetime, so a stream that outlived it was cut mid-body: the deadline raises a bare
``asyncio.TimeoutError`` from the payload read, which is not a ``ClientError``, so
``_stream_single_request``'s arm does not match and ``_relay_after_commit``'s
``except Exception`` swallows it — a 200 SSE stream that simply stops, with no error event
and nothing in the caller's way to tell it. Base ran that path with no deadline
(``buster_ripper_legacy.py:664``), and this removes the *event* rather than arming an arm
to catch the class: the request-local ``ClientTimeout`` replaces the session's rather than
merging into it, so every component is unset on that one call.

Neither witness widens an arm. ``messages.py``'s ``except aiohttp.ClientError`` and the
post-commit swallow are the ones ``T013`` accepted, and this task leaves them byte-exact:
the deadline is gone, so the class that could escape the arm has no producer here.
"""

import asyncio
import json

import aiohttp
import pytest
from aiohttp import web

from buster_ripper import config
from buster_ripper import image as image_handler
from tests.upstream_stub import StubStream, StubUpstream

MIB = 1024 * 1024
_HUGE = "x" * (2 * MIB)
_MESSAGES = {"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": _HUGE}]}
_CHAT = {"model": "m", "messages": [{"role": "user", "content": _HUGE}]}
_PASSTHROUGH = b"payload=" + _HUGE.encode()
_RELAY = {"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]}
_STREAM = dict(_RELAY, stream=True)

_JSON = {"Content-Type": "application/json"}
_ANSWER = b'{"id": "msg_1", "content": [{"type": "text", "text": "ok"}]}'

# The SSE shapes the relay is judged on: one that starts the stream, and the terminal
# that only arrives if nothing cut the call short.
_SSE_START = b'data: {"type":"message_start","message":{"usage":{"input_tokens":7}}}\n\n'
_SSE_DONE = b"data: [DONE]\n\n"
_TERMINATED = b"\r\n0\r\n\r\n"
_HEADERS = (("Accept", "text/event-stream"), ("Anthropic-Version", "2023-06-01"))


def _upstream_app(stall: float) -> web.Application:
    """An SSE upstream: answers at once, sends one chunk, stalls, then finishes."""
    app = web.Application()

    async def handle(request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        await resp.write(_SSE_START)
        await asyncio.sleep(stall)
        await resp.write(_SSE_DONE)
        await resp.write_eof()
        return resp

    app.router.add_post("/v1/messages", handle)
    return app


async def _serve(app: web.Application) -> tuple:
    """Run `app` on loopback; returns (its base url, its teardown coroutine)."""
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    host, port = runner.addresses[0][:2]
    return f"http://{host}:{port}", runner.cleanup


def _arm_shared_session(mocker, raw_proxy, deadline: float) -> aiohttp.ClientSession:
    """Arm the app's own shared session to `deadline`, as base armed it to 600.

    The number comes from the construction this task must leave alone (``server.py:
    66-68``) and ``ClientSession`` has no setter for it, so the witness changes the
    number on the object already wired — one connector, one posture, still the one the
    app built. A short deadline is what lets a test run watch a stream straddle a
    timeout that would otherwise need ten minutes of upstream to reach.
    """
    session = raw_proxy.app["session"]
    mocker.patch.object(session, "_timeout", aiohttp.ClientTimeout(total=deadline))
    return session


def _status_lines(raw: bytes) -> list:
    return [line for line in raw.split(b"\r\n") if line.startswith(b"HTTP/")]


class TestNoRequestBodyCap:
    """CR3-F-01: no route answers 413 for body size, at any size the base stack accepted."""

    @pytest.mark.parametrize("case", [
        pytest.param(("/v1/messages", {"json": _MESSAGES}, "post"), id="messages"),
        pytest.param(("/v1/chat/completions", {"json": _CHAT}, "post"), id="chat-completions"),
        pytest.param(("/v1/models", {"data": _PASSTHROUGH}, "request"),
                     id="passthrough-catch-all"),
    ])
    async def test_a_two_mebibyte_body_reaches_the_upstream(self, proxy, mocker, case):
        """Red at pre-fix bytes as a 413 out of aiohttp's reader, before any handler
        ran: the cap is the application's, so it is one construction site to change."""
        path, kwargs, method = case
        sent = mocker.patch.object(
            proxy.server.app["session"], method,
            return_value=StubUpstream(body=_ANSWER, headers=_JSON))

        response = await proxy.request("POST", path, **kwargs)

        assert response.status == 200, "the app answered for the body size"
        assert sent.called, "the read never happened, so nothing was forwarded"

    async def test_count_tokens_reads_a_two_mebibyte_body_it_never_forwards(self, proxy):
        """The fourth route class answers from its own body, so its read path shows up as
        a token estimate rather than as an upstream call."""
        response = await proxy.post("/v1/messages/count_tokens",
                                    json=dict(_CHAT, max_tokens=10))

        assert response.status == 200, "the app answered for the body size"
        assert "input_tokens" in await response.json()

    async def test_a_strip_images_body_over_the_cap_reaches_the_image_handler(
        self, proxy, mocker
    ):
        """The cap sat *ahead* of the feature that shrinks this very body: with
        ``--strip-images`` on, an image-bearing request over 1 MiB was refused before one
        image byte could be moved out of what the proxy forwards."""
        mocker.patch.object(config, "STRIP_IMAGES", True)
        stripped = mocker.patch.object(image_handler, "strip_and_cache_images")
        mocker.patch.object(proxy.server.app["session"], "post",
                            return_value=StubUpstream(body=_ANSWER, headers=_JSON))
        body = dict(_MESSAGES, messages=[{
            "role": "user",
            "content": [{"type": "image", "source": {
                "type": "base64", "media_type": "image/png", "data": _HUGE}}],
        }])

        response = await proxy.post("/v1/messages", json=body)

        assert response.status == 200, "the cap still answers before the handler reads"
        assert stripped.called, "the image never reached the writer that strips it"


class TestStreamCarriesNoDeadline:
    """CR3-F-03: the /v1/messages relay lives as long as the upstream streams."""

    _CASES = {
        "outlives": (0.1, 0.4, False),
        "operand": (0.1, 0.05, True),
    }

    async def _relay(self, mocker, raw_proxy, case):
        """One streaming request through the real app against an upstream that stalls.

        The body here is deliberately small: this class is about the *deadline*, and a
        multi-megabyte body would put the cap witness's subject into it too.
        """
        deadline, stall, spy = self._CASES[case]
        url, teardown = await _serve(_upstream_app(stall))
        session = _arm_shared_session(mocker, raw_proxy, deadline)
        post = mocker.patch.object(session, "post", wraps=session.post) if spy else None
        mocker.patch.object(config, "UPSTREAM", url)
        try:
            response = await raw_proxy.send(
                "/v1/messages", headers=_HEADERS, body=json.dumps(_STREAM).encode(),
            )
            return response, post
        finally:
            await teardown()

    async def test_a_stream_outliving_the_shared_deadline_arrives_whole(self, mocker,
                                                                        raw_proxy):
        """Before: the deadline fired inside the payload read, the swallow ate it, and the
        caller was left with a 200 whose body simply stopped. After: the terminal is there,
        because nothing is armed on this call."""
        response, _ = await self._relay(mocker, raw_proxy, "outlives")
        raw = response.raw

        assert _status_lines(raw) == [b"HTTP/1.1 200 OK"]
        assert _SSE_START in raw, "the stream never even started"
        assert _SSE_DONE in raw, "the shared session's deadline cut the relay mid-body"
        assert raw.endswith(_TERMINATED)
        assert not response.closed, "the connection was cut instead of finished"

    async def test_the_streaming_call_passes_a_timeout_with_nothing_armed(self, mocker,
                                                                         raw_proxy):
        """All four components unset, not merely a ``total`` that looks unlimited: the
        request-local object replaces the session's wholesale, so a spelling that
        re-inherited ``total`` — or armed ``sock_read`` "to be safe" — would pass the
        truncation witness above while leaving the class the arm cannot catch still armed.
        """
        response, post = await self._relay(mocker, raw_proxy, "operand")

        assert _SSE_DONE in response.raw, "the operand under test did not relay to the end"
        operand = post.call_args.kwargs["timeout"]
        assert isinstance(operand, aiohttp.ClientTimeout)
        assert (operand.total, operand.sock_connect, operand.sock_read,
                operand.connect) == (None, None, None, None), f"{operand!r} arms a deadline"

    async def test_the_non_stream_arm_still_inherits_the_shared_deadline(self, proxy, mocker):
        """The companion this repair must not silently widen: one call loses its deadline,
        so the other consumer of the shared session keeps both what it inherits from and
        the number on it (``server.py:66-68``)."""
        post = mocker.patch.object(proxy.server.app["session"], "post",
                                   return_value=StubUpstream(body=_ANSWER, headers=_JSON))

        response = await proxy.post("/v1/messages", json=dict(_RELAY, stream=False))

        assert response.status == 200
        assert "timeout" not in post.call_args.kwargs, (
            "the non-stream arm stopped inheriting the session's deadline")
        assert proxy.server.app["session"].timeout.total == 600


class TestChatStreamCarriesNoDeadline:
    """PR1-R4-3: the chat relay's deadline is the request's shape, not the session's.

    The chat route builds its own request-local ``ClientTimeout``, so the shared
    session's number never reached it. A streaming relay must arm nothing at all;
    a non-streaming request keeps base's 300 s bound.
    """

    async def test_a_streaming_chat_arms_no_total(self, proxy, mocker):
        """Before: ``total=300`` was armed on the streaming call, so a relay longer
        than five minutes was cut mid-body and the post-commit swallow hid it."""
        post = mocker.patch.object(
            proxy.server.app["session"], "post",
            return_value=StubStream(headers=_JSON, chunks=(b"data: {}\n\n",)))

        response = await proxy.post("/v1/chat/completions", json=dict(_RELAY, stream=True))

        assert response.status == 200
        timeout = post.call_args.kwargs["timeout"]
        assert isinstance(timeout, aiohttp.ClientTimeout)
        assert timeout.total is None, f"{timeout!r} arms a deadline on the stream"

    async def test_a_non_stream_chat_keeps_the_300s_bound(self, proxy, mocker):
        """The other half: dropping the stream's deadline must not drop the bound
        base had on a non-streaming chat request."""
        post = mocker.patch.object(
            proxy.server.app["session"], "post",
            return_value=StubUpstream(body=_ANSWER, headers=_JSON))

        response = await proxy.post("/v1/chat/completions",
                                    json=dict(_RELAY, stream=False))

        assert response.status == 200
        assert post.call_args.kwargs["timeout"].total == 300

"""T013 G-1 — a failure after the response head is committed abandons the stream.

``_stream_single_request`` prepares its SSE response inside the handler, so once
``prepare()`` has run the status line is already on the client's wire. Returning a
different response object from there — a fresh 502, or nothing at all, which
aiohttp turns into a 500 — hands the client a second head *inside* the first
response's body, where no conforming client reads it as a status. Required
behavior: log, abandon, and return the committed object, so ``prepare()``
short-circuits on its own writer and ``finish_response()``'s ``write_eof()`` ends
the stream.

The fault cases read the raw downstream bytes, because "the 502 is absent" is not a
pass — a dropped turn has no 502 either; the terminal is what separates them. The
cancellation case is a unit call for the same reason it needs no server: the claim
is which base class the arm stops at. The chat route carries the same two witnesses
at its own boundary, beside these.
"""

import asyncio
import inspect
import json
import logging
import textwrap

import aiohttp
import pytest
from aiohttp import web

from buster_ripper.routes import chat as chat_routes
from buster_ripper.routes import messages as messages_routes
from tests.upstream_stub import StubStream

_MESSAGE_START = (
    b'data: {"type":"message_start","message":{"usage":{"input_tokens":42}}}\n\n'
)
_PLAIN = b"data: hello\n\n"

BODY_JSON = json.dumps(
    {"model": "m", "max_tokens": 10, "stream": True,
     "messages": [{"role": "user", "content": "hi"}]}
).encode()
CHAT_BODY_JSON = json.dumps({"stream": True, "messages": []}).encode()

_HEADERS = (("Accept", "text/event-stream"), ("Anthropic-Version", "2023-06-01"))
_CLOSE = _HEADERS + (("Connection", "close"),)
_STREAM_HEADERS = {"Content-Type": "text/event-stream"}

# One param per case: (position, fault raised there, chunk relayed before it).
# The write-position case arms the downstream writer instead of the upstream, so it
# needs two chunks: the first write is swallowed and the second raises. That is the one
# position the iteration faults cannot reach, and a relay whose try wrapped only
# the iteration would pass every other case in this table.
POST_COMMIT_FAULTS = [
    pytest.param(("iteration", aiohttp.ClientPayloadError("chunk died"), _PLAIN),
                 id="client-payload-error"),
    pytest.param(("iteration", asyncio.TimeoutError(), _PLAIN),
                 id="bare-asyncio-timeout-error"),
    pytest.param(("iteration", aiohttp.ClientConnectionResetError("client left"), _PLAIN),
                 id="client-connection-reset"),
    pytest.param(("write", OSError("downstream gone"), None),
                 id="write-position-os-error"),
]


def _arm_fault(mocker, case):
    """Point the failure at the case's position and return the upstream answer."""
    position, fault, chunk = case
    if position == "write":
        mocker.patch.object(web.StreamResponse, "write", side_effect=[None, fault])
        return StubStream(headers=_STREAM_HEADERS, chunks=[_MESSAGE_START, _PLAIN])
    return StubStream(headers=_STREAM_HEADERS, chunks=[chunk], fault=fault)


async def _drive_fault(mocker, raw_proxy, case):
    """One streaming request whose relay dies after the first chunk."""
    upstream = _arm_fault(mocker, case)
    mocker.patch.object(raw_proxy.app["session"], "post", return_value=upstream)
    return await raw_proxy.send("/v1/messages", headers=_HEADERS, body=BODY_JSON)


def _status_lines(raw: bytes) -> list:
    return [line for line in raw.split(b"\r\n") if line.startswith(b"HTTP/")]


def _relay_lines(caplog) -> list:
    return [r.getMessage() for r in caplog.records if "relay error" in r.getMessage().lower()]


def _server_error_turns(caplog) -> list:
    return [r for r in caplog.records if "Error handling request" in r.getMessage()]


class TestPostCommitFaults:
    """Case table rows 1-3: every post-commit failure gets one identical terminal."""

    @pytest.mark.parametrize("case", POST_COMMIT_FAULTS)
    async def test_one_head_one_terminal_and_no_error_turn(self, mocker, raw_proxy,
                                                          caplog, case):
        with caplog.at_level(logging.DEBUG):
            response = await _drive_fault(mocker, raw_proxy, case)
        raw = response.raw

        assert _status_lines(raw) == [b"HTTP/1.1 200 OK"], _status_lines(raw)
        assert raw.endswith(b"\r\n0\r\n\r\n"), f"stream never terminated: {raw[-60:]!r}"
        assert not response.closed, "the connection was cut instead of finished"
        assert b" 502 " not in raw and b" 500 " not in raw
        assert _server_error_turns(caplog) == [], "aiohttp's own error path ran"
        assert len(_relay_lines(caplog)) == 1, _relay_lines(caplog)

    @pytest.mark.parametrize("case", POST_COMMIT_FAULTS)
    async def test_the_log_names_position_not_cause(self, mocker, raw_proxy, caplog, case):
        """One arm covers upstream payload errors, both timeout classes and a client
        leaving, so it may not attribute the loss to the upstream (DC-S11-D8-r1)."""
        with caplog.at_level(logging.DEBUG):
            await _drive_fault(mocker, raw_proxy, case)
        lines = _relay_lines(caplog)

        assert len(lines) == 1, f"expected one post-commit log line, got {lines!r}"
        assert "post-commit" in lines[0].lower()
        assert "upstream error" not in lines[0].lower()


class TestWrongShapedMessageStart:
    """PR1-R4-A2: a wrong-shaped message_start cannot end the committed relay.

    ``_note_message_start`` read ``.get`` on whatever ``message`` and ``usage`` held,
    so an upstream event with a non-mapping value raised ``AttributeError``, the
    post-commit catch swallowed it, and the rest of the turn was never relayed.
    """

    @pytest.mark.parametrize("event", [
        pytest.param({"type": "message_start", "message": None}, id="null-message"),
        pytest.param({"type": "message_start", "message": {"usage": []}},
                     id="list-usage"),
    ])
    async def test_the_following_chunk_still_reaches_the_client(self, mocker, raw_proxy,
                                                                event):
        chunk = b"data: " + json.dumps(event).encode() + b"\n\n"
        stats = mocker.patch.object(messages_routes, "update_session_stats")
        upstream = StubStream(headers=_STREAM_HEADERS, chunks=[chunk, _PLAIN])
        mocker.patch.object(raw_proxy.app["session"], "post", return_value=upstream)

        response = await raw_proxy.send("/v1/messages", headers=_HEADERS, body=BODY_JSON)
        raw = response.raw

        assert _status_lines(raw) == [b"HTTP/1.1 200 OK"]
        assert chunk in raw and _PLAIN in raw, "the wrong-shaped event ended the relay"
        assert raw.endswith(b"\r\n0\r\n\r\n")
        stats.assert_called_once()
        assert stats.call_args.args[1] == 0, "the guarded read did not fall back to 0"


class TestChatPostCommit:
    """The chat route's own arm: one head and one terminal after a post-commit loss.

    The messages cases above close the same defect at their boundary; this is the
    sibling the roadmap tracked, and it reads the raw bytes for the same reason — a
    dropped turn and a relayed 502 both end without a usable answer, so the second
    status line inside the first body is what separates them.
    """

    async def test_chat_payload_error_after_commit_gets_one_head_and_a_terminal(
        self, mocker, raw_proxy, caplog
    ):
        upstream = StubStream(
            headers=_STREAM_HEADERS,
            chunks=[_PLAIN],
            fault=aiohttp.ClientPayloadError("chunk died"),
        )
        mocker.patch.object(raw_proxy.app["session"], "post", return_value=upstream)

        with caplog.at_level(logging.DEBUG):
            response = await raw_proxy.send(
                "/v1/chat/completions", headers=_HEADERS, body=CHAT_BODY_JSON
            )
        raw = response.raw

        assert _status_lines(raw) == [b"HTTP/1.1 200 OK"], _status_lines(raw)
        assert raw.endswith(b"\r\n0\r\n\r\n"), f"stream never terminated: {raw[-60:]!r}"
        assert b" 502 " not in raw and b" 500 " not in raw
        assert _server_error_turns(caplog) == [], "aiohttp's own error path ran"
        lines = _relay_lines(caplog)
        assert len(lines) == 1, f"expected one post-commit log line, got {lines!r}"
        assert "post-commit" in lines[0].lower()
        assert "upstream error" not in lines[0].lower()

    async def test_chat_cancellation_still_propagates(self, mocker):
        """except Exception, never BaseException: a client leaving still unwinds."""
        response = mocker.Mock()
        response.prepare = mocker.AsyncMock()
        response.write = mocker.AsyncMock()
        response.write_eof = mocker.AsyncMock()
        mocker.patch.object(chat_routes.web, "StreamResponse", return_value=response)
        resp = mocker.Mock()
        resp.status = 200

        async def iter_any():
            raise asyncio.CancelledError
            yield b""  # pragma: no cover - only makes this an async generator

        resp.content.iter_any = iter_any

        with pytest.raises(asyncio.CancelledError):
            await chat_routes._relay_chat_stream(_fake_request(mocker), resp, None)

        response.write_eof.assert_not_awaited()


class TestArmShape:
    """The catch-all has to stay one catch-all."""

    def test_the_relay_is_one_catch_all_not_a_class_split(self):
        """Falsifiable only structurally: a build with separate ClientError /
        AttributeError / disconnect arms produces the same wire bytes per case, and
        DC-S11-D8-r1 forbids adding the disconnect-specific arm anyway."""
        clauses = [line.strip() for line in _source(messages_routes._relay_after_commit).splitlines()
                   if line.strip().startswith("except")]
        assert len(clauses) == 1 and clauses[0].startswith("except Exception")

    async def test_cancellation_still_propagates(self, mocker):
        """Case table row 4: except Exception, never BaseException."""
        resp = mocker.Mock()

        async def iter_any():
            raise asyncio.CancelledError
            yield b""  # pragma: no cover - only makes this an async generator

        resp.content.iter_any = iter_any
        response = mocker.AsyncMock()
        state = {"done": False, "input_tokens": 0, "sid": "s", "started": 0.0}

        with pytest.raises(asyncio.CancelledError):
            await messages_routes._relay_after_commit(_fake_request(mocker), resp, response, state)

        response.write_eof.assert_not_awaited()


def _source(func) -> str:
    return textwrap.dedent(inspect.getsource(func))


def _fake_request(mocker):
    request = mocker.Mock()
    request.app = {"log": logging.getLogger("buster-ripper")}
    return request


class TestRelayBehaviour:
    """The halves the split must leave alone."""

    async def test_chunks_are_relayed_verbatim_and_usage_recorded(self, mocker, raw_proxy):
        stats = mocker.patch.object(messages_routes, "update_session_stats")
        upstream = StubStream(headers={"Content-Type": "text/event-stream"},
                              chunks=[_MESSAGE_START, _PLAIN])
        mocker.patch.object(raw_proxy.app["session"], "post", return_value=upstream)

        response = await raw_proxy.send("/v1/messages", headers=_HEADERS, body=BODY_JSON)
        raw = response.raw

        assert _status_lines(raw) == [b"HTTP/1.1 200 OK"]
        assert _MESSAGE_START in raw and _PLAIN in raw
        assert raw.endswith(b"\r\n0\r\n\r\n")
        stats.assert_called_once()
        assert stats.call_args.args[1] == 42

    async def test_a_pre_commit_client_error_still_answers_502(self, mocker, raw_proxy, caplog):
        """The half of the family that was already closed stays closed: nothing is
        committed yet, so a fresh error response is correct there."""
        mocker.patch.object(raw_proxy.app["session"], "post",
                            side_effect=aiohttp.ClientConnectionError("refused"))
        with caplog.at_level(logging.DEBUG):
            response = await raw_proxy.send("/v1/messages", headers=_CLOSE, body=BODY_JSON)

        assert _status_lines(response.raw) == [b"HTTP/1.1 502 Bad Gateway"]
        assert any("Upstream error" in r.getMessage() for r in caplog.records)
        assert _relay_lines(caplog) == []

"""Proxy-local failures: the route's own envelopes, never a leak or aiohttp's answer.

``CR2-F-2``'s leak half: a genuine transport failure keeps its 502 but its body is static,
because ``str(e)`` named the internal upstream host and port, while the cause still reaches the
operator's log. ``TP-1``'s deadline half: a fired upstream deadline reaches the route's own 502
envelope (``asyncio.TimeoutError`` is not a ``ClientError``). ``CR3-F-04``(c): a client naming
a codec Python lacks gets the route's own 400.
"""

import asyncio
import json
import logging

import aiohttp
import pytest

from tests.relay_bodies import _BAD_CODEC, _CHAT, _JUNK, _MESSAGES, _upstream_authority
from tests.upstream_stub import StubUpstream


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
        """Taking the text out of the body must not take it out of the record."""
        self._refuse_the_upstream(proxy, mocker, "post")

        with caplog.at_level(logging.DEBUG, logger="buster-ripper"):
            response = await proxy.post("/v1/chat/completions", json=_CHAT)

        assert response.status == 502
        lines = [r.getMessage() for r in caplog.records if "Upstream error" in r.getMessage()]
        assert len(lines) == 1
        assert _upstream_authority() in lines[0], "the cause never reached the log"


class TestUnknownClientCodecArm:
    """CR3-F-04(c): a client naming a codec Python lacks gets the route's own 400."""

    @pytest.mark.parametrize("path", [
        pytest.param("/v1/messages", id="messages"),
        pytest.param("/v1/chat/completions", id="chat-completions"),
        pytest.param("/v1/messages/count_tokens", id="count-tokens"),
    ])
    async def test_the_unknown_codec_never_reaches_the_caller_as_a_500(self, proxy, mocker,
                                                                       path):
        """``web_request.py:689`` hands ``charset`` to ``bytes.decode()``, which raises
        ``LookupError`` — not a ``UnicodeDecodeError``, so no arm named it and the caller got
        aiohttp's 500. The body is well-formed; the name it arrived under is not.
        """
        upstream = mocker.patch.object(proxy.server.app["session"], "post",
                                       return_value=StubUpstream(headers=_JUNK))

        response = await proxy.post(path, data=json.dumps(_MESSAGES).encode(),
                                    headers={"Content-Type": _BAD_CODEC})

        assert response.status == 400, f"{path} answered {response.status} for a codec name"
        assert json.loads(await response.read()) == {"error": "Invalid JSON"}
        upstream.assert_not_called()


class TestUpstreamTimeout:
    """TP-1: a fired upstream deadline is the route's own 502, never aiohttp's 504.

    ``asyncio.TimeoutError`` is not a ``ClientError``, so the arm named only the latter
    and the exception left the handler.
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

"""A streaming error arm and an undecodable payload relay as upstream sent them.

``CR2-F-2``'s decode half: the streaming error arm answers with the upstream's status,
bytes and headers rather than its own text. ``CR3-F-04``: the decode step at every boundary
that has one answers rather than raising, in both directions, because the two directions are
asymmetric.
"""

import json

from tests.relay_bodies import _BAD_CODEC, _JUNK, _MESSAGES, _UNDECODABLE
from tests.upstream_stub import StubUpstream


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
        ``decode()`` made undecodable bytes an argument the handler raised on. The overflow
        text is ASCII, so it survives replacement.
        """
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
        those bytes with replacement, so no charset can raise out of the loop.
        """
        mocker.patch.object(proxy.server.app["session"], "post",
                            return_value=StubUpstream(status=503, headers=_JUNK,
                                                      body=_UNDECODABLE))

        response = await proxy.post("/v1/messages", json={**_MESSAGES, "stream": True})

        assert response.status == 503, "the proxy answered the upstream's payload with a 500"
        assert await response.read() == _UNDECODABLE

    async def test_an_unknown_response_charset_over_decodable_bytes_is_not_a_raise(
        self, proxy, mocker
    ):
        """The probe decodes bytes with replacement, so the relayed unknown charset
        label cannot raise out of the loop as a ``.text`` read would."""
        mocker.patch.object(proxy.server.app["session"], "post",
                            return_value=StubUpstream(
                                status=503,
                                headers={"Content-Type": _BAD_CODEC},
                                body=b'{"error":"boom"}'))

        response = await proxy.post("/v1/messages", json={**_MESSAGES, "stream": True})

        assert response.status == 503, "the proxy answered the upstream's payload with a 500"
        assert json.loads(await response.read()) == {"error": "boom"}

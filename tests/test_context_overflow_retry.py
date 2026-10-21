"""The three route-level context overflow retry loops, and which backends reach them.

``TestStreamingOverflowRetry`` and ``TestNonStreamingOverflowRetry`` enter the loops at
the route level, ``TestChatStreamingOverflowRetry`` covers the chat stream's error arm, and
``TestBackendAwareRetry`` only checks which backends reach them.
"""

import json

import aiohttp
import pytest

from buster_ripper import backend
from tests.backend_probe import reset_backend_state
from tests.overflow_bodies import (
    _ERROR_HEADERS,
    _JSON,
    _MESSAGES,
    _OVERFLOW,
    _REDUCED_TO,
    _sglang_stream,
)
from tests.upstream_stub import StubUpstream


class TestBackendAwareRetry:
    """Test that retry logic only applies to SGLang backends."""

    def test_sglang_backend_enables_retry(self, reset_backend_state):
        """SGLang backend should enable retry logic."""
        backend._backend_type = "sglang"
        backend._backend_state_cached = True

        assert backend.get_backend_state() == "sglang"

    def test_vllm_backend_skips_retry(self, reset_backend_state):
        """vLLM backend should skip retry logic."""
        backend._backend_type = "vllm"
        backend._backend_state_cached = True

        assert backend.get_backend_state() == "vllm"

    def test_unknown_backend_skips_retry(self, reset_backend_state):
        """Unknown backend should skip retry logic."""
        backend._backend_type = "unknown"
        backend._backend_state_cached = True

        assert backend.get_backend_state() == "unknown"

    def test_none_backend_skips_retry(self, reset_backend_state):
        """None backend should skip retry logic."""
        backend._backend_type = None
        backend._backend_state_cached = False

        assert backend.get_backend_state() is None


class TestStreamingOverflowRetry:
    """The streaming probe is handed the upstream's error *text*, so the loop can fire.

    The old read took ``.decode()`` if bytes else ``str(...)``, which handed the matcher a
    ``StringPayload`` repr no pattern matches, so the SGLang streaming retry could never
    ``continue``. The read is now the response's own text surface.

    Route-level on purpose: the claim is about the object that crosses
    ``_stream_single_request`` into ``handle_streaming``.
    """

    async def test_an_overflow_reduces_the_window_and_reissues(self, proxy, mocker):
        """Three upstream requests, the second and third carrying the reduced window. Red at
        pre-fix bytes with ``call_count == 1`` and no ``max_tokens`` change on any body."""
        post, sent = _sglang_stream(proxy, mocker,
                                    StubUpstream(status=400, headers=_JSON, body=_OVERFLOW))

        await proxy.post("/v1/messages", json={**_MESSAGES, "stream": True})

        assert post.call_count == 3, "the overflow retry loop did not run"
        assert sent[0]["max_tokens"] == _MESSAGES["max_tokens"], "the first request was pre-reduced"
        assert sent[1]["max_tokens"] == _REDUCED_TO
        assert sent[2]["max_tokens"] == _REDUCED_TO
        assert all("max_completion_tokens" not in body for body in sent), (
            "the messages retry wrote the chat-only field")

    async def test_the_loop_answers_for_itself_when_the_retries_run_out(self, proxy, mocker):
        """The terminal arm — the loop's own answer at ``attempt == max_retries`` — needs a
        match to reach it, so the inert read left it dead too.
        """
        _sglang_stream(proxy, mocker, StubUpstream(status=400, headers=_JSON, body=_OVERFLOW))

        response = await proxy.post("/v1/messages", json={**_MESSAGES, "stream": True})

        assert response.status == 400
        assert json.loads(await response.read()) == {
            "error": "Context overflow after 2 retries"}

    @pytest.mark.parametrize("case", [
        pytest.param((400, b'{"error": "quota exceeded, retry later"}'), id="plain-4xx"),
        pytest.param((503, b"\xff\xfe\x00garbage"), id="undecodable-payload"),
    ])
    async def test_an_error_that_is_not_an_overflow_relays_once_and_unchanged(self, proxy,
                                                                             mocker, case):
        """The other half: reading the relayed bytes must not turn every 4xx into a
        re-dispatch, and the caller gets the upstream's status, bytes and names.
        """
        status, payload = case
        post, _ = _sglang_stream(
            proxy, mocker,
            StubUpstream(status=status, headers=_ERROR_HEADERS, body=payload))

        response = await proxy.post("/v1/messages", json={**_MESSAGES, "stream": True})

        assert post.call_count == 1, "a non-overflow error became a re-dispatch"
        assert response.status == status
        for name, value in _ERROR_HEADERS.items():
            assert response.headers[name] == value, f"{name} was dropped from the error"
        assert await response.read() == payload, "the upstream's bytes were replaced"

    async def test_the_transport_502_crosses_the_probe_without_raising(self, proxy, mocker):
        """The static ``json_response`` 502 is the loop's other input, whose body *is*
        bytes; the text surface reads it as well, so nothing new can raise.
        """
        mocker.patch.object(backend, "get_backend_state", return_value="sglang")
        post = mocker.patch.object(proxy.server.app["session"], "post",
                                   side_effect=aiohttp.ClientConnectionError("refused"))

        response = await proxy.post("/v1/messages", json={**_MESSAGES, "stream": True})

        assert post.call_count == 1, "a transport failure was retried as an overflow"
        assert response.status == 502, "the probe raised past the handler's caller"
        assert json.loads(await response.read()) == {"error": "Upstream request failed"}


class TestNonStreamingOverflowRetry:
    """The non-streaming probe raises the parsed overflow, so the loop reduces and
    reissues ``max_tokens``, never the chat-only ``max_completion_tokens``.
    """

    async def test_an_overflow_reduces_max_tokens_and_reissues(self, proxy, mocker):
        """Three upstream requests, the second and third carrying the reduced window."""
        post, sent = _sglang_stream(proxy, mocker,
                                    StubUpstream(status=400, headers=_JSON, body=_OVERFLOW))

        response = await proxy.post("/v1/messages", json=_MESSAGES)

        assert post.call_count == 3, "the overflow retry loop did not run"
        assert sent[0]["max_tokens"] == _MESSAGES["max_tokens"], "the first request was pre-reduced"
        assert sent[1]["max_tokens"] == _REDUCED_TO
        assert sent[2]["max_tokens"] == _REDUCED_TO
        assert all("max_completion_tokens" not in body for body in sent), (
            "the messages retry wrote the chat-only field")
        assert response.status == 400
        assert json.loads(await response.read())["error"].startswith(
            "Context overflow after 2 retries")


class TestChatStreamingOverflowRetry:
    """The chat stream's error arm is that route's only overflow probe: it parses
    the error bytes it relays rather than ``resp.json()``.
    """

    async def test_an_overflow_reduces_the_window_and_reissues(self, proxy, mocker):
        """Three upstream requests, the second and third carrying the reduced window.
        Red at pre-fix bytes with ``call_count == 1``."""
        post, sent = _sglang_stream(proxy, mocker,
                                     StubUpstream(status=400, headers=_JSON, body=_OVERFLOW))

        response = await proxy.post("/v1/chat/completions",
                                   json={**_MESSAGES, "stream": True})

        assert post.call_count == 3, "the overflow retry loop did not run"
        assert "max_completion_tokens" not in sent[0], "the first request was pre-reduced"
        assert sent[1]["max_completion_tokens"] == _REDUCED_TO
        assert sent[2]["max_completion_tokens"] == _REDUCED_TO
        assert response.status == 400
        assert json.loads(await response.read())["error"].startswith(
            "Context overflow after 2 retries")

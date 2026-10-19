"""Tests for context overflow retry functionality.

Tests cover:
- Context overflow error parsing
- Token calculation
- Retry behavior in route handlers — the streaming and non-streaming route loops are
  entered at the route level by `TestStreamingOverflowRetry` and
  `TestNonStreamingOverflowRetry` below; `TestBackendAwareRetry` checks only which
  backends reach those loops, never what happens inside them.
- Backend-aware retry (SGLang only)
"""

import json

import aiohttp
import pytest

from buster_ripper import config, backend, utils
from tests.upstream_stub import StubUpstream


class TestContextOverflowParsing:
    """Test context overflow error parsing."""

    def test_parse_overflow_with_context_overflow_pattern(self):
        """Should parse Context overflow error message."""
        error_text = "Context overflow: 50000 > 4096"
        result = utils.parse_context_overflow_error(error_text)

        assert result is not None
        assert result.input_tokens == 50000
        assert result.ctx_limit == 4096

    def test_parse_overflow_with_different_numbers(self):
        """Should parse different number combinations."""
        error_text = "Context overflow detected: 100000 > 8192"
        result = utils.parse_context_overflow_error(error_text)

        assert result is not None
        assert result.input_tokens == 100000
        assert result.ctx_limit == 8192

    def test_parse_overflow_no_match(self):
        """Should return None when pattern doesn't match."""
        error_text = "Some other error message"
        result = utils.parse_context_overflow_error(error_text)

        assert result is None

    def test_parse_overflow_with_max_completion_tokens(self):
        """Should prefer CTX_LIMIT_RE for context limit if present."""
        error_text = "Context overflow: 50000 > 4096 max_completion_tokens 8192"
        result = utils.parse_context_overflow_error(error_text)

        assert result is not None
        assert result.input_tokens == 50000
        assert result.ctx_limit == 8192  # Uses CTX_LIMIT_RE value

    def test_context_overflow_error_attributes(self):
        """Should have correct attributes on error."""
        error = utils.ContextOverflowError("test error", 1000, 4096)

        assert str(error) == "test error"
        assert error.input_tokens == 1000
        assert error.ctx_limit == 4096


class TestTokenCalculation:
    """Test new max_completion_tokens calculation."""

    def test_calculate_normal_case(self):
        """Should calculate correct value with safety margin."""
        # ctx_limit=10000, input_tokens=5000, margin=100
        # new_max = max(256, 10000 - 5000 - 100) = 4900
        result = utils.calculate_new_max_tokens(10000, 5000)
        assert result == 4900

    def test_calculate_uses_min_tokens(self):
        """Should use MIN_COMPLETION_TOKENS when calculation falls below."""
        # ctx_limit=1000, input_tokens=900, margin=100
        # new_max = max(256, 1000 - 900 - 100) = max(256, 0) = 256
        result = utils.calculate_new_max_tokens(1000, 900)
        assert result == config.MIN_COMPLETION_TOKENS

    def test_calculate_exact_boundary(self):
        """Should handle boundary case correctly."""
        # ctx_limit=5000, input_tokens=4644, margin=100
        # new_max = max(256, 5000 - 4644 - 100) = max(256, 256) = 256
        result = utils.calculate_new_max_tokens(5000, 4644)
        assert result == 256

    def test_min_completion_tokens_constant(self):
        """Should have correct MIN_COMPLETION_TOKENS value."""
        assert config.MIN_COMPLETION_TOKENS == 256


class TestRegexPatterns:
    """Test regex patterns compile and match correctly."""

    def test_context_overflow_re_pattern(self):
        """CONTEXT_OVERFLOW_RE should match context overflow errors."""
        pattern = config.CONTEXT_OVERFLOW_RE

        # Should match
        match = pattern.search("Context overflow: 50000 > 4096")
        assert match is not None
        assert match.group(1) == "50000"
        assert match.group(2) == "4096"

    def test_context_overflow_re_various_formats(self):
        """CONTEXT_OVERFLOW_RE should handle various formats."""
        pattern = config.CONTEXT_OVERFLOW_RE

        # Various formats
        texts = [
            "Context overflow 1000>500",
            "Context overflow: 1000 > 500 tokens",
            "Error: Context overflow detected 99999 > 128000",
        ]

        for text in texts:
            match = pattern.search(text)
            assert match is not None, f"Should match: {text}"

    def test_ctx_limit_re_pattern(self):
        """CTX_LIMIT_RE should match max_completion_tokens values."""
        pattern = config.CTX_LIMIT_RE

        match = pattern.search("max_completion_tokens 4096")
        assert match is not None
        assert match.group(1) == "4096"

    def test_ctx_limit_re_various_formats(self):
        """CTX_LIMIT_RE should handle various formats."""
        pattern = config.CTX_LIMIT_RE

        texts = [
            "max_completion_tokens=8192",
            "max_completion_tokens: 4096",
            "set max_completion_tokens 16384",
        ]

        for text in texts:
            match = pattern.search(text)
            assert match is not None, f"Should match: {text}"


class TestBackendAwareRetry:
    """Test that retry logic only applies to SGLang backends."""

    @pytest.fixture
    def reset_backend_state(self):
        """Reset backend module state before each test."""
        backend._backend_type = None
        backend._backend_state_cached = False
        yield
        backend._backend_type = None
        backend._backend_state_cached = False

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


_MESSAGES = {"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]}
_JSON = {"Content-Type": "application/json"}
# The names the streaming error arm used to replace with `Content-Type` alone.
_ERROR_HEADERS = {"Content-Type": "application/json", "X-Request-Id": "req_7f3a",
                  "Retry-After": "7"}

# The routed counterexample, verbatim. It parses to input 5000 over a 4096 window, so the
# reduced request carries 256 — written as the value, not recomputed from the formula, so a
# change to `calculate_new_max_tokens` shows up here as a broken retry rather than a pass.
_OVERFLOW_TEXT = json.dumps(
    {"error": "Context overflow: input 5000 > 4096 tokens, max_completion_tokens 4096"}
)
_OVERFLOW = _OVERFLOW_TEXT.encode()
_REDUCED_TO = 256


def _sglang_stream(proxy, mocker, answer):
    """A backend that runs the retry loop, answering every request with `answer`.

    Returns the mock and each request body as that call saw it. The copy is what makes the
    per-attempt claim checkable: the loop rewrites one dict in place, so the mock's own
    call history hands back the *final* body for every attempt — and an assertion that
    reissue 2 and 3 carry the reduced window would stay green on a build that reduced the
    window before the first request ever went out.
    """
    mocker.patch.object(backend, "get_backend_state", return_value="sglang")
    sent: list = []

    def record(url, **kwargs):
        sent.append(json.loads(kwargs["data"]) if "data" in kwargs else dict(kwargs["json"]))
        return answer

    post = mocker.patch.object(proxy.server.app["session"], "post", side_effect=record)
    return post, sent


class TestStreamingOverflowRetry:
    """The streaming probe is handed the upstream's error *text*, so the loop can fire.

    ``_stream_single_request`` answers an upstream 4xx with ``web.Response(body=<str>,
    content_type="application/json")`` on its ``status >= 400`` arm, and aiohttp stores a
    ``str`` given to ``body=`` as a ``StringPayload`` instead of as its bytes. So the old
    read — ``.decode()`` if bytes else ``str(...)`` — took the ``else`` branch
    and handed the matcher ``<aiohttp.payload.StringPayload object at 0x…>``. No pattern in
    ``config.py:122-123`` matches a memory address, ``parse_context_overflow_error``
    returned ``None``, and the loop returned the upstream's 400 after exactly one request.
    Both answers the loop can be handed were inert — the other is the static 502, whose
    body cannot say "Context overflow" either — so the SGLang streaming retry could never
    ``continue``. The read is now the response's own text surface, which hands back the
    string that went in.

    Route-level on purpose: the claim is about the object that crosses
    ``_stream_single_request`` into ``handle_streaming``, and a unit call would have to
    construct it by hand instead of receiving the one production constructs.
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
        """The terminal arm — the loop's own answer, taken when ``overflow_err`` is truthy at
        ``attempt == max_retries`` — needs a match to reach it, so the inert read left it dead
        too. Pre-fix the caller receives the upstream's own body here; post-fix it receives
        the loop's."""
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
        """The other half of the contract: reading the relayed bytes must not turn every
        4xx into a re-dispatch, and what the caller gets is the upstream's own answer —
        its status, its bytes and every end-to-end name. The undecodable case is the
        hostile one at this boundary: the probe replaces what it cannot decode, so no
        charset can raise out of the loop.
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
        """The second of the two objects the loop can be handed is the static ``json_response``
        502 on ``_stream_single_request``'s ``ClientError`` arm, whose body *is* bytes — the one
        shape the old ternary handled. The text surface reads it as well, which is why nothing
        new can raise out of ``handle_streaming`` from that read: 502 rather than aiohttp's 500
        is the observable form of the claim."""
        mocker.patch.object(backend, "get_backend_state", return_value="sglang")
        post = mocker.patch.object(proxy.server.app["session"], "post",
                                   side_effect=aiohttp.ClientConnectionError("refused"))

        response = await proxy.post("/v1/messages", json={**_MESSAGES, "stream": True})

        assert post.call_count == 1, "a transport failure was retried as an overflow"
        assert response.status == 502, "the probe raised past the handler's caller"
        assert json.loads(await response.read()) == {"error": "Upstream request failed"}


class TestNonStreamingOverflowRetry:
    """The non-streaming probe raises the parsed overflow out of the single-request arm,
    so the loop reduces and reissues. The budget it writes is ``max_tokens`` — the field
    the Anthropic endpoint reads — and never the chat-only ``max_completion_tokens``.
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
    """The chat stream's error arm is that route's only overflow probe.

    It relayed the upstream's 400 verbatim, so ``handle_chat``'s retry loop never
    saw an overflow: one request, and the reduced ``max_completion_tokens`` was never
    reissued. The probe parses the error bytes it relays rather than ``resp.json()``, so
    an error that is not JSON keeps answering as the upstream sent it.
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

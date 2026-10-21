"""Overflow bodies and a sent-request recorder for the context overflow retry tests."""

import json

from buster_ripper import backend

_MESSAGES = {"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]}
_JSON = {"Content-Type": "application/json"}
# The names the streaming error arm used to replace with `Content-Type` alone.
_ERROR_HEADERS = {"Content-Type": "application/json", "X-Request-Id": "req_7f3a",
                  "Retry-After": "7"}

# The routed counterexample, verbatim: 5000 over a 4096 window, so the reduced request
# carries 256 — the value, not the formula, so a changed calculation shows up as a broken retry.
_OVERFLOW_TEXT = json.dumps(
    {"error": "Context overflow: input 5000 > 4096 tokens, max_completion_tokens 4096"}
)
_OVERFLOW = _OVERFLOW_TEXT.encode()
_REDUCED_TO = 256


def _sglang_stream(proxy, mocker, answer):
    """A backend that runs the retry loop, answering every request with `answer`.

    Returns the mock and a copy of each request body, because the loop rewrites one dict in
    place and the mock's call history would hand back the final body for every attempt.
    """
    mocker.patch.object(backend, "get_backend_state", return_value="sglang")
    sent: list = []

    def record(url, **kwargs):
        sent.append(json.loads(kwargs["data"]) if "data" in kwargs else dict(kwargs["json"]))
        return answer

    post = mocker.patch.object(proxy.server.app["session"], "post", side_effect=record)
    return post, sent

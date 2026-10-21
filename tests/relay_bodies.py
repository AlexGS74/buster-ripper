"""Bodies, relay headers and sent-request helpers for the upstream relay tests."""

import json

from buster_ripper import config

_CHAT = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
_MESSAGES = {"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]}

# The end-to-end names the branch dropped: a correlation id, a retry hint, a budget.
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


_BAD_CODEC = "application/json; charset=x-not-a-real-codec"
_UNDECODABLE = b"\xff\xfe\x00garbage"
_JUNK = {"Content-Type": "application/json"}

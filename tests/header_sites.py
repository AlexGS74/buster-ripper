"""Site tables for the header-parity tests."""

import json

import pytest

from buster_ripper.routes import chat as chat_routes
from buster_ripper.routes import messages as messages_routes
from buster_ripper.routes import passthrough as passthrough_routes
from tests.upstream_stub import StubUpstream

SIX_NAMES = ("TE", "Upgrade", "Keep-Alive", "Proxy-Authorization",
             "Proxy-Authenticate", "Trailers")
ALL_NAMES = SIX_NAMES + ("Connection", "Transfer-Encoding")
END_TO_END = {"Anthropic-Version": "2023-06-01", "X-Custom-End-To-End": "keep-me"}

_CHAT_BODY = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
_MESSAGES_BODY = {"model": "m", "max_tokens": 10,
                  "messages": [{"role": "user", "content": "hi"}]}
_CHAT_JSON = json.dumps(_CHAT_BODY).encode()

# Which mapping a helper was handed is discriminated by a name that exists on
# exactly one side of the boundary.
INBOUND_MARKER = "X-Inbound-Only"
UPSTREAM_MARKER = "X-Upstream-Only"

CHAT_SITE = {"path": "/v1/chat/completions", "method": "POST", "attr": "post",
             "module": chat_routes, "body": _CHAT_BODY}
MESSAGES_SITE = {"path": "/v1/messages", "method": "POST", "attr": "post",
                 "module": messages_routes, "body": _MESSAGES_BODY}
PASSTHROUGH_SITE = {"path": "/v1/models", "method": "GET", "attr": "request",
                    "module": passthrough_routes, "body": None}

EGRESS_SITES = [pytest.param(CHAT_SITE, id="chat"),
                pytest.param(MESSAGES_SITE, id="messages"),
                pytest.param(PASSTHROUGH_SITE, id="passthrough")]
# The chat boundary joined at the CR2 wave: its non-stream arm now builds its answer
# through utils.response_headers too, so the same kept/drop census has to hold there.
INGRESS_SITES = [pytest.param(CHAT_SITE, id="chat"),
                 pytest.param(MESSAGES_SITE, id="messages"),
                 pytest.param(PASSTHROUGH_SITE, id="passthrough")]
NON_CHAT_SITES = [pytest.param(MESSAGES_SITE, id="messages"),
                  pytest.param(PASSTHROUGH_SITE, id="passthrough")]

_REAL_TOKEN = "Bearer sk-real"
_EMPTY_SPELLINGS = ("Bearer", "Bearer ")
# `Authorization:` with nothing after the colon. Not a bearer form at all, and base
# forwarded it — so it is neither a credential nor something the rule deletes.
_BLANK = ""

# seam-S15 CE-2: the name an upstream sends twice. A custom end-to-end name, because
# the claim is about occurrences, not about any one name's semantics.
_REPEAT = "X-Repeat"
_REPEAT_PAIR = [(_REPEAT, "first"), (_REPEAT, "second")]


def _repeated_head() -> list:
    """An upstream answer that says one end-to-end name twice with *different* values,
    beside a kept name and a hop-by-hop name sent twice too."""
    return _REPEAT_PAIR + [(UPSTREAM_MARKER, "upstream"),
                           ("Content-Type", "application/json"),
                           ("Connection", "one"), ("Connection", "two")]


def _values(mapping, name: str) -> list:
    """Every value held for a name, case-insensitively, from any header mapping."""
    return [v for k, v in mapping.items() if k.lower() == name.lower()]


def _wire_headers(extra=None) -> dict:
    """What a client sends: hop-by-hop names, end-to-end names, a boundary marker."""
    headers = {"TE": "trailers", "Upgrade": "h2c", "Keep-Alive": "timeout=5, max=99",
               "Proxy-Authorization": "Basic YWxhZGRpbjpvcGVuc2VzYW1l",
               "Proxy-Authenticate": 'Basic realm="proxy"', "Trailers": "X-Total-Count",
               "Connection": "keep-alive", INBOUND_MARKER: "inbound", **END_TO_END}
    headers.update(extra or {})
    return headers


def _spy_helpers(mocker, module):
    """Record which helper a site calls, and with which header mapping."""
    spies = {"forward_headers": mocker.spy(module, "forward_headers")}
    if hasattr(module, "response_headers"):
        spies["response_headers"] = mocker.spy(module, "response_headers")
    return spies


async def _egress_capture(proxy, mocker, site, extra_headers=None):
    """Send one request; return the header mapping that reached the upstream."""
    called = mocker.patch.object(
        proxy.server.app["session"], site["attr"],
        return_value=StubUpstream(headers={UPSTREAM_MARKER: "upstream"}))
    kwargs = {"json": site["body"]} if site["body"] is not None else {}
    response = await proxy.request(site["method"], site["path"],
                                   headers=_wire_headers(extra_headers), **kwargs)
    assert response.status == 200
    assert called.call_count == 1
    return called.call_args.kwargs["headers"]


async def _ingress_relay(proxy, mocker, site, relay):
    """Relay an upstream answer; return the client-side response.

    `relay` carries what the upstream sends and how to ask for it: `headers`,
    optional `body`, and any `client_*` keyword for the caller itself (a timeout,
    for the case whose wrong answer waits instead of answering).
    """
    client_kw = {k[len("client_"):]: v for k, v in relay.items() if k.startswith("client_")}
    upstream = StubUpstream(headers=relay.get("headers"), body=relay.get("body", b"upstream-body"))
    mocker.patch.object(proxy.server.app["session"], site["attr"], return_value=upstream)
    kwargs = {"json": site["body"]} if site["body"] is not None else {}
    kwargs.update(client_kw)
    return await proxy.request(site["method"], site["path"], headers=_wire_headers(), **kwargs)

"""T013 F-4 — hop-by-hop filtering at every proxy boundary goes through utils.

Base proxy semantics are the contract: the eight names in ``utils._HOP_BY_HOP`` are
dropped at all three request and all three response boundaries from ONE shared
implementation, and ``content-length`` is dropped on the response side so aiohttp
derives the length of the body actually sent. The empty-``Bearer`` rule stays
chat-route-local: ``/v1/messages`` and the catch-all forward an empty bearer.

Header *names* are never matched against an expected wire spelling, since a name's
spelling depends on the parser backend; every absence is asserted case-insensitively
on the mapping. The response side is a census over *occurrences* as well as names
(``TestRepeatedHeaderOccurrences``).
"""

import aiohttp
import pytest
from aiohttp import http_parser, web_protocol
from multidict import CIMultiDict

from buster_ripper.routes import chat as chat_routes
from buster_ripper.utils import response_headers
from tests.header_sites import (
    ALL_NAMES,
    CHAT_SITE,
    EGRESS_SITES,
    END_TO_END,
    INBOUND_MARKER,
    INGRESS_SITES,
    MESSAGES_SITE,
    NON_CHAT_SITES,
    UPSTREAM_MARKER,
    _BLANK,
    _CHAT_JSON,
    _EMPTY_SPELLINGS,
    _REAL_TOKEN,
    _REPEAT,
    _REPEAT_PAIR,
    _egress_capture,
    _ingress_relay,
    _repeated_head,
    _spy_helpers,
    _values,
)
from tests.upstream_stub import StubUpstream


class TestEgressSites:
    """The three request boundaries: no hop-by-hop name reaches the upstream."""

    @pytest.mark.parametrize("site", EGRESS_SITES)
    async def test_hop_by_hop_names_never_reach_upstream(self, proxy, mocker, site):
        captured = await _egress_capture(proxy, mocker, site)
        for name in ALL_NAMES:
            assert _values(captured, name) == [], f"{name} was forwarded upstream"

    @pytest.mark.parametrize("site", EGRESS_SITES)
    async def test_end_to_end_names_survive(self, proxy, mocker, site):
        captured = await _egress_capture(proxy, mocker, site)
        for name, value in END_TO_END.items():
            assert _values(captured, name) == [value]
        assert _values(captured, INBOUND_MARKER) == ["inbound"]

    @pytest.mark.parametrize("site", EGRESS_SITES)
    async def test_site_calls_the_egress_helper_on_the_request_mapping(
        self, proxy, mocker, site
    ):
        """Direction pairing: wiring response_headers here would keep the client's
        own Host and forward a stale Content-Length, so both are asserted absent and
        each helper is asserted to have received its own direction's mapping."""
        spies = _spy_helpers(mocker, site["module"])
        captured = await _egress_capture(proxy, mocker, site)
        assert spies["forward_headers"].call_count == 1
        handed = spies["forward_headers"].call_args.args[0]
        assert _values(handed, INBOUND_MARKER) == ["inbound"], "helper got the wrong mapping"
        if "response_headers" in spies and spies["response_headers"].call_count:
            relayed = spies["response_headers"].call_args.args[0]
            assert _values(relayed, UPSTREAM_MARKER) == ["upstream"]
        assert _values(captured, "Host") == []
        assert _values(captured, "Content-Length") == []

    @pytest.mark.parametrize("spelling", ["tE", "te"])
    async def test_mis_cased_name_is_still_dropped(self, proxy, mocker, spelling):
        """The drop matches the lowercased name, whichever spelling the peer used."""
        captured = await _egress_capture(proxy, mocker, MESSAGES_SITE,
                                         extra_headers={spelling: "trailers"})
        assert _values(captured, "TE") == []


class TestIngressSites:
    """The three response boundaries: relayed hop-by-hop names stop at the proxy."""

    @pytest.mark.parametrize("site", INGRESS_SITES)
    async def test_hop_by_hop_names_never_reach_the_client(self, proxy, mocker, site):
        headers = {UPSTREAM_MARKER: "upstream", "Content-Type": "application/json"}
        headers.update({name: "relayed" for name in ALL_NAMES})
        response = await _ingress_relay(proxy, mocker, site, {"headers": headers})
        for name in ALL_NAMES:
            assert _values(response.headers, name) == [], f"{name} reached the client"

    @pytest.mark.parametrize("site", INGRESS_SITES)
    async def test_end_to_end_names_survive(self, proxy, mocker, site):
        headers = {UPSTREAM_MARKER: "upstream", "Content-Type": "application/json",
                   "Etag": '"abc"', "X-Custom-End-To-End": "keep-me"}
        response = await _ingress_relay(proxy, mocker, site, {"headers": headers})
        assert _values(response.headers, UPSTREAM_MARKER) == ["upstream"]
        assert _values(response.headers, "X-Custom-End-To-End") == ["keep-me"]

    @pytest.mark.parametrize("site", INGRESS_SITES)
    async def test_site_calls_the_ingress_helper_on_the_upstream_mapping(
        self, proxy, mocker, site
    ):
        """Direction pairing, response side: forward_headers would drop the relayed
        Host too, and is the wrong helper for this boundary."""
        spies = _spy_helpers(mocker, site["module"])
        headers = {UPSTREAM_MARKER: "upstream", "Content-Type": "application/json", "Host": "up"}
        response = await _ingress_relay(proxy, mocker, site, {"headers": headers})
        assert response.status == 200
        assert spies["response_headers"].call_count == 1
        handed = spies["response_headers"].call_args.args[0]
        assert _values(handed, UPSTREAM_MARKER) == ["upstream"], "helper got the wrong mapping"
        assert _values(spies["forward_headers"].call_args.args[0], INBOUND_MARKER) == ["inbound"]

    @pytest.mark.parametrize("site", INGRESS_SITES)
    async def test_lying_upstream_content_length_never_reaches_the_client(
        self, proxy, mocker, site
    ):
        """The content-length extension is load-bearing, not cosmetic: an upstream
        that sends both framing headers must not have its length relayed, because
        Response._start derives a length only when the header is absent. Without the
        extension the client waits on 999 bytes and this fails on a timeout."""
        body = b"0123456789ab"
        headers = {UPSTREAM_MARKER: "upstream", "Content-Type": "application/json",
                   "Transfer-Encoding": "chunked", "Content-Length": "999"}
        response = await _ingress_relay(proxy, mocker, site, {
            "headers": headers, "body": body,
            "client_timeout": aiohttp.ClientTimeout(total=5)})
        received = await response.read()
        assert received == body
        assert _values(response.headers, "Transfer-Encoding") == []
        assert int(response.headers["Content-Length"]) == len(body)


class TestRepeatedHeaderOccurrences:
    """seam-S15 CE-2 — one name sent twice reaches the client twice, on its own line.

    Base parity: base *merged* occurrences into one comma-joined value, so every
    value travelled; dropping all but the last is branch-introduced. The drop census
    runs over occurrences too — a removed name is removed every time it appears.
    """

    def test_response_headers_keeps_every_occurrence_of_one_name(self):
        """Unit tier: the fold to a plain ``dict`` is the defect, so this is red while the
        filter collapses keys, whatever the boundary it is reached from does later."""
        relayed = response_headers(CIMultiDict(_REPEAT_PAIR))
        assert _values(relayed, _REPEAT) == ["first", "second"]

    @pytest.mark.parametrize("name", ALL_NAMES + ("Content-Length",))
    def test_every_occurrence_of_a_dropped_name_is_dropped(self, name):
        """Kept-set completeness, the other direction: preserving occurrences must not
        turn into preserving them *before* the drop runs."""
        relayed = response_headers(CIMultiDict([(name, "one"), (name, "two"),
                                                ("X-Keep", "y")]))
        assert _values(relayed, name) == []
        assert _values(relayed, "X-Keep") == ["y"]

    def test_two_spellings_of_one_dropped_name_are_both_dropped(self):
        """The pair the occurrence rule must still merge: `Connection` arriving as two
        field-lines under different spellings is one hop-by-hop name twice, not one
        kept occurrence. Lowercasing decides the drop, per-occurrence storage does not."""
        relayed = response_headers(CIMultiDict([("Connection", "close"),
                                                ("connection", "keep-alive"),
                                                ("X-Keep", "y")]))
        assert _values(relayed, "Connection") == []
        assert _values(relayed, "X-Keep") == ["y"]

    def test_two_spellings_of_one_kept_name_stay_two_lines_with_their_own_spelling(self):
        """And its converse: a kept name sent under two spellings keeps both lines and
        both spellings — the filter canonicalizes for the *drop*, never for the relay."""
        relayed = response_headers(CIMultiDict([("date", "T1"), ("Date", "T2")]))
        assert [k for k, _ in relayed.items()] == ["date", "Date"]
        assert _values(relayed, "date") == ["T1", "T2"]

    @pytest.mark.parametrize("site", INGRESS_SITES)
    async def test_both_occurrences_reach_the_client(self, proxy, mocker, site):
        """One witness per response boundary — the same filter at three sites is only one
        place to get it wrong if all three hand its result to ``web.Response``."""
        response = await _ingress_relay(proxy, mocker, site, {"headers": _repeated_head()})
        assert _values(response.headers, _REPEAT) == ["first", "second"]
        assert _values(response.headers, "Connection") == []

    async def test_the_two_values_arrive_as_two_field_lines(self, raw_proxy, mocker):
        """The wire, not the parsed view: a build that comma-joined the pair would satisfy
        every assertion above one client-side value short of two field-lines."""
        mocker.patch.object(raw_proxy.app["session"], "post",
                            return_value=StubUpstream(headers=_repeated_head()))
        response = await raw_proxy.send("/v1/chat/completions",
                                        headers=[("Connection", "close")], body=_CHAT_JSON)
        head = response.raw.split(b"\r\n\r\n")[0]
        assert head.lower().count(_REPEAT.lower().encode() + b":") == 2
        assert b"X-Repeat: first\r\n" in head and b"X-Repeat: second\r\n" in head


class TestEmptyBearerRule:
    """Done-when (3)'s four cases: the rule is chat-route-local and value-complete."""

    @pytest.mark.parametrize("value", _EMPTY_SPELLINGS, ids=["trailing-space", "bare"])
    async def test_chat_deletes_a_bearer_that_carries_no_token(self, proxy, mocker, value):
        captured = await _egress_capture(proxy, mocker, CHAT_SITE,
                                         extra_headers={"Authorization": value})
        assert _values(captured, "Authorization") == []

    async def test_chat_keeps_a_real_token(self, proxy, mocker):
        captured = await _egress_capture(proxy, mocker, CHAT_SITE,
                                         extra_headers={"Authorization": _REAL_TOKEN})
        assert _values(captured, "Authorization") == [_REAL_TOKEN]

    @pytest.mark.parametrize("site", NON_CHAT_SITES)
    async def test_the_other_boundaries_forward_it_verbatim(self, proxy, mocker, site):
        """Base parity: the strip lives only in the chat handler, so a build that
        promotes it into forward_headers fails here."""
        captured = await _egress_capture(proxy, mocker, site,
                                         extra_headers={"Authorization": "Bearer "})
        assert _values(captured, "Authorization") != []

    def test_a_value_less_authorization_line_is_not_touched(self):
        """The rule's subject is the value-less bearer form, not every blank-ish
        Authorization line: base forwarded `Authorization:` and so does this. Beside an
        empty bearer the blank line is still what survives — neither carries a
        credential, so only the bearer form is dropped."""
        headers = chat_routes._prepare_headers(CIMultiDict([("Authorization", "")]))
        assert _values(headers, "Authorization") == [""]

        mixed = chat_routes._prepare_headers(
            CIMultiDict([("Authorization", _BLANK), ("authorization", "Bearer ")])
        )
        assert _values(mixed, "Authorization") == [_BLANK]

    @pytest.mark.parametrize("order", ["real-first", "empty-first"])
    @pytest.mark.parametrize("parser", ["c", "py"], ids=["c-parser", "pure-python-parser"])
    async def test_duplicate_spellings_never_lose_the_credential(
        self, raw_proxy, mocker, order, parser
    ):
        """The rule reads every value the client sent, not the one entry the
        collapsed copy of the filtered dict keeps: on the C backend two spellings of
        one known name collapse last-wins, so an empty bearer arriving second deletes
        a real credential. On the pure-Python backend both survive, so the empty
        bearer is forwarded beside the token. Both are wrong, and only a raw socket
        puts two Authorization lines on the wire — the aiohttp client merges them.
        """
        if parser == "c" and not hasattr(http_parser, "HttpRequestParserC"):
            pytest.skip("aiohttp C parser extension is not installed")
        cls = getattr(http_parser, {"c": "HttpRequestParserC", "py": "HttpRequestParserPy"}[parser])
        mocker.patch.object(web_protocol, "HttpRequestParser", cls)
        pairs = [("Authorization", _REAL_TOKEN), ("authorization", "Bearer ")]
        if order == "empty-first":
            pairs.reverse()

        called = mocker.patch.object(raw_proxy.app["session"], "post",
                                     return_value=StubUpstream(headers={UPSTREAM_MARKER: "u"}))
        spies = _spy_helpers(mocker, CHAT_SITE["module"])
        response = await raw_proxy.send("/v1/chat/completions",
                                        headers=pairs + [("Connection", "close")],
                                        body=_CHAT_JSON)

        assert response.raw.startswith(b"HTTP/1.1 200"), response.raw[:80]
        assert response.closed, "the socket should have finished at EOF, not at the window"
        arrived = _values(spies["forward_headers"].call_args.args[0], "Authorization")
        assert len(arrived) == 2, f"both lines must reach the handler, got {arrived!r}"
        assert _REAL_TOKEN in arrived and len([v for v in arrived if v.strip() == "Bearer"]) == 1
        assert _values(called.call_args.kwargs["headers"], "Authorization") == [_REAL_TOKEN]

    @pytest.mark.parametrize("pairs", [
        [("Authorization", _REAL_TOKEN), ("Authorization", "Bearer ")],
        [("Authorization", "Bearer "), ("Authorization", _REAL_TOKEN)],
        [("Authorization", _REAL_TOKEN), ("authorization", "Bearer ")],
        [("authorization", "Bearer "), ("Authorization", _REAL_TOKEN)],
        # 066-01-yield P1: the blank line is not a credential. Arriving first it must
        # not be the value chosen (base forwarded both spellings, and collapsed the
        # same-spelling pair to the token — either way the token reached the upstream);
        # arriving last it must not displace one (base collapsed to [""] there).
        [("Authorization", _BLANK), ("Authorization", _REAL_TOKEN)],
        [("Authorization", _REAL_TOKEN), ("Authorization", _BLANK)],
        [("Authorization", _BLANK), ("authorization", _REAL_TOKEN)],
        [("authorization", _REAL_TOKEN), ("Authorization", _BLANK)],
    ], ids=["c-collapse-real-first", "c-collapse-empty-first",
            "py-split-real-first", "py-split-empty-first",
            "c-collapse-blank-first", "c-collapse-real-first-blank-last",
            "py-split-blank-first", "py-split-real-first-blank-last"])
    def test_the_rule_reads_the_request_mapping_not_the_collapsed_copy(self, pairs):
        assert _values(chat_routes._prepare_headers(CIMultiDict(pairs)),
                       "Authorization") == [_REAL_TOKEN]

    @pytest.mark.parametrize("value", _EMPTY_SPELLINGS)
    def test_a_lone_empty_bearer_is_still_deleted(self, value):
        for name in ("Authorization", "authorization"):
            headers = chat_routes._prepare_headers(CIMultiDict([(name, value)]))
            assert _values(headers, "Authorization") == []

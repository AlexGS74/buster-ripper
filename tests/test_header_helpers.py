"""Helper-level tests for the shared header filters."""

from pathlib import Path

import pytest

from buster_ripper.routes import messages as messages_routes
from buster_ripper.utils import _HOP_BY_HOP, forward_headers, response_headers
from tests.header_sites import ALL_NAMES, _values


class TestSharedHelpers:
    """One policy owner: the kept/drop sets each boundary is allowed to have."""

    @pytest.mark.parametrize("name", ALL_NAMES)
    def test_forward_headers_drops_every_hop_name(self, name):
        dropped = forward_headers({name: "x", "X-Keep": "y"})
        assert _values(dropped, name) == []
        assert _values(dropped, "X-Keep") == ["y"]

    def test_forward_headers_drops_host_and_content_length(self):
        dropped = forward_headers({"Host": "h", "Content-Length": "9", "Accept": "a"})
        assert {k.lower() for k in dropped} == {"accept"}

    @pytest.mark.parametrize("name", ALL_NAMES + ("Content-Length",))
    def test_response_headers_drops_hop_names_and_content_length(self, name):
        dropped = response_headers({name: "x", "X-Keep": "y"})
        assert _values(dropped, name) == []
        assert _values(dropped, "X-Keep") == ["y"]

    def test_response_headers_keeps_host(self):
        """Kept-set boundary: response_headers is not forward_headers."""
        assert _values(response_headers({"Host": "up", "Connection": "keep-alive"}),
                       "Host") == ["up"]

    def test_hop_by_hop_set_is_bases_eight_names(self):
        """The set's authority is base parity (buster_ripper_legacy.py:178), not a
        standard's enumeration: no name is added and none is respelled."""
        assert _HOP_BY_HOP == frozenset({"connection", "keep-alive", "proxy-authenticate",
                                         "proxy-authorization", "te", "trailers",
                                         "transfer-encoding", "upgrade"})


class TestSiteCensus:
    """Done-when (1): the five comprehensions are gone and the helpers own the policy."""

    def test_no_header_comprehension_is_left_in_a_route(self):
        routes = Path(messages_routes.__file__).parent
        offenders = sorted(
            path.name for path in routes.glob("*.py") if "headers.items()" in path.read_text()
        )
        assert offenders == [], f"route-local filter comprehensions remain in {offenders}"

    def test_utils_holds_the_only_two_filters(self):
        source = Path(forward_headers.__code__.co_filename).read_text()
        assert source.count("for k, v in") == 2, "the shared helpers are the only filters"

"""CR4-F-01 — `/v1/messages/count_tokens` keeps base's tolerance for a malformed body member.

The `T019` step 1 repair made this route normalize the body before keying the stats
lookup, which is what the derived-after-normalize session digest requires. It did so by
calling `normalize()` — the whole of it, all four steps — and `normalize()` walks
`tools` and `messages` assuming every element is a mapping. So the route acquired a
500 that base never had: `{"tools":[null]}`, `{"messages":[null]}` and
`{"messages":["x"]}` raise `AttributeError` out of the element loops in `_normalize_tools`
and `_strip_cache_control_from_messages`, aiohttp authors the answer, and the client gets
a generated 500 where base answered an estimate.

Measured, not reasoned about — `evidence/cr4-probe-matrix.txt`:
  §1 the trio answers 200 at `6508941` (pre-range) and 500 at HEAD, on both axes;
  §2 base answers 200 to all three (`buster_ripper.py:690` reads the body, `:696` keys
     it, and no `normalize` call is on that path at all);
  §5 rebinding this route's normalization to the system-strip step alone restores
     base's column exactly — the trio 200, the non-dict body still 500.

`session_id()` reads `system` only (`session.py:93-98`) and `_normalize_system` is the
one step of the four that writes `system`'s text, so the narrowing keeps the digest the
invariant asks for. What the trio must *not* do is move `/v1/messages`: there the same
three bodies raise the same `AttributeError` in base itself (`buster_ripper.py:374`,
`:402` — evidence §3), so its 500 is migrated behavior that `G-003` holds still. The
class below that answers 500 without reaching the relay is a drift guard for the
rejected candidate (hardening `normalize()`'s element loops), and `roadmap.md` §RM-012
owns any change to that posture.
"""

import json

import aiohttp
import pytest

from buster_ripper import backend, config
from buster_ripper import session as session_mod
from buster_ripper.normalize import normalize, strip_billing_nonce
from buster_ripper.session import session_id
from tests.upstream_stub import StubUpstream

_COUNT = "/v1/messages/count_tokens"
_MESSAGES = "/v1/messages"

_ANSWER = b'{"id": "msg_1", "content": [{"type": "text", "text": "ok"}]}'
_JSON = {"Content-Type": "application/json"}

# The block Claude Code puts at system position 0, verbatim in shape: it changes on every
# request, which is the whole reason the digest is keyed after it is removed.
_NONCE_SYSTEM = "x-anthropic-billing-header: cc_version=2.1.50.f15; cch=27acd;"

# One user turn, so a body is never only a system prompt.
_TURN = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]


def _copy(body: dict) -> dict:
    """A private deep copy — both call shapes mutate the dict they are handed."""
    return json.loads(json.dumps(body))


# The three shapes the finding names: a non-mapping element in `tools`, a non-mapping
# element in `messages`, and a string element in `messages`. All three are malformed per
# the Anthropic schema, so the documented client never sends them — which is why the
# defect survived a review of the route and surfaced only in the external one.
_MALFORMED_MEMBERS = [
    pytest.param({"tools": [None]}, id="tools-null-element"),
    pytest.param({"messages": [None]}, id="messages-null-element"),
    pytest.param({"messages": ["x"]}, id="messages-string-element"),
]


class TestCountTokensToleratesMalformedMembers:
    """The route answers an estimate for every JSON object body, as base does."""

    @pytest.mark.parametrize("body", _MALFORMED_MEMBERS)
    async def test_a_malformed_member_still_answers_an_estimate(self, proxy, body):
        """Red at pre-fix bytes as a 500 out of `normalize()`'s element loops.

        The authority is base's own column at `buster_ripper.py:672-696` and this route's
        column at `6508941` — both answer 200 (evidence §1, §2). The body is the only
        thing under test here: no stats are seeded, so a 200 can only be the estimate.
        """
        response = await proxy.post(_COUNT, json=body)

        assert response.status == 200, "the route answered 5xx for a member base tolerates"
        reported = (await response.json())["input_tokens"]
        assert type(reported) is int, f"{reported!r} is not a token count"

    @pytest.mark.parametrize("kwargs", [
        pytest.param({"json": {"system": [None]}}, id="system-null-block"),
        pytest.param({"json": {"messages": [{"content": [None]}]}}, id="content-null-block"),
        pytest.param({"json": {}}, id="empty-object"),
        pytest.param({"data": b"{", "headers": {"Content-Type": "application/json"}},
                     id="unparseable-body"),
    ])
    async def test_a_shape_base_already_tolerated_stays_tolerated(self, proxy, kwargs):
        """Green in both states, deliberately: this is the over-reach guard.

        `{"system":[null]}` and `{"messages":[{"content":[null]}]}` survive
        `_normalize_system`'s and `_strip_cache_control_from_content`'s own isinstance
        guards, and `{}`/an unparseable body reach the route as an empty dict. All four
        answer 200 at base, at `6508941` and at HEAD (evidence §1, §2).

        What reddens each direction, measured three ways on `6a2069b`: a key narrowed
        past the billing nonce reddens `TestCountTokensKeysOnTheNormalizedDigest`;
        hardened element loops redden `TestMessagesMalformedMemberPosture`; routing the
        estimate through `normalize(body, session_id="")` reddens
        `test_a_malformed_member_still_answers_an_estimate`. This class's own
        over-reach direction is a stricter `_read_body` that rejects the unparseable
        body with 400 — not a narrower key or a hardened loop.
        """
        response = await proxy.post(_COUNT, **kwargs)

        assert response.status == 200, "the repair reached a body base already tolerated"
        assert "input_tokens" in await response.json()

    @pytest.mark.parametrize("body", [
        pytest.param(b"null", id="null-root"),
        pytest.param(b"[1, 2]", id="list-root"),
        pytest.param(b"5", id="number-root"),
    ])
    async def test_a_non_mapping_body_answers_the_routes_400(self, proxy, body):
        """PR1-R4-A3 supersedes the 500 this row used to pin: base crashed on a
        non-object root too, so the route's own authored 400 is the ranked answer
        rather than a crash to preserve. Red at pre-fix bytes as a 500."""
        response = await proxy.post(_COUNT, data=body,
                                   headers={"Content-Type": "application/json"})

        assert response.status == 400, "the route answered 500 for a non-object body"
        assert await response.json() == {"error": "Invalid JSON"}


class TestMessagesMalformedMemberPosture:
    """`/v1/messages` still answers base's 500 — the drift guard for §RM-012.

    The rejected candidate for `CR4-F-01` was to harden `normalize()`'s element loops
    instead of narrowing this one call. That would fix the count_tokens column by moving
    `/v1/messages` from base's generated 500 to a proxy-authored 502 (evidence §7) — the
    route `G-003` exists to hold still, with an unauthorized ruling on what a proxy owes
    schema-invalid input doing the work. Any witness in this class reddens exactly then.
    """

    @pytest.mark.parametrize("body", _MALFORMED_MEMBERS)
    async def test_a_malformed_member_still_answers_bas_own_500(self, proxy, mocker, body):
        """Red first in the direction of the candidate, not of the shipped code.

        Green today because the raise happens in `normalize()` before the relay is
        reached. Hardening the loops lets the request through, so the upstream call is
        made and the answer becomes a 502 — base's authority for this route is
        `buster_ripper.py:374`/`:402`, which raise the identical `AttributeError`, and
        `roadmap.md` §RM-012 owns any change to that posture.

        The upstream call is stubbed to refuse rather than pointed at a socket, so the
        502 arm this witness describes is the same one `T013`'s review accepted.
        """
        mocker.patch.dict(session_mod._stats, {}, clear=True)
        sent = mocker.patch.object(proxy.server.app["session"], "post",
                                   side_effect=aiohttp.ClientConnectionError("stub refused"))

        response = await proxy.post(_MESSAGES, json=body)

        assert response.status == 500, (
            "/v1/messages stopped answering base's 500 — see roadmap.md §RM-012")
        assert not sent.called, "the request survived normalization and reached the relay"

    async def test_a_well_formed_body_still_reaches_the_relay(self, proxy, mocker):
        """The pair that keeps the witness above from passing on a dead route.

        Same stub, opposite expectation: a body normalization accepts must reach the
        upstream call and relay its answer, so a 500 in the class above is attributable
        to `normalize()` rather than to a route that never forwards anything.
        """
        sent = mocker.patch.object(
            proxy.server.app["session"], "post",
            return_value=StubUpstream(body=_ANSWER, headers=_JSON))

        response = await proxy.post(_MESSAGES, json={
            "model": "m", "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}],
        })

        assert response.status == 200
        assert sent.called, "the control body never reached the relay"


class TestCountTokensKeysOnTheNormalizedDigest:
    """The digest this route keys on stays the one `/v1/messages` records under.

    Narrowing the call is only safe while the system-strip step is the only step that can
    change what `session_id()` hashes. These two witnesses are why that is a checked
    condition rather than an assumption: the seeded key is always computed with the whole
    of `normalize()`, so the route is required to agree with it body for body.
    """

    # The documented Claude Code shape, in the three system spellings evidence §4
    # measured digest-equal over the full battery.
    @pytest.mark.parametrize("system", [
        pytest.param(_NONCE_SYSTEM + "\nsys", id="string-system"),
        pytest.param([{"type": "text", "text": _NONCE_SYSTEM + "\nsys"}], id="block-system"),
        pytest.param([{"type": "text", "text": _NONCE_SYSTEM + "\nsys", "cache_control":
                       {"type": "ephemeral"}}], id="block-system-with-cache-control"),
    ])
    async def test_a_nonce_body_finds_the_stats_under_the_full_normalize_digest(
        self, proxy, mocker, system
    ):
        """Red under the pre-`T019` spelling (keying the raw body), and red under any
        future `normalize()` step that starts rewriting `system`'s text — the regression
        this narrowing otherwise makes possible.

        The seeded key is `session_id()` of the *fully* normalized body, which is what
        `/v1/messages` records under. `input_tokens` is a number no estimate of these
        small bodies can produce, so a green cannot be the fallback answering instead.
        """
        body = {"model": "m", "max_tokens": 10, "system": system, "messages": _TURN}
        seeded = session_id(normalize(_copy(body), session_id="")[0])
        mocker.patch.dict(session_mod._stats, {seeded: mocker.Mock(
            input_tokens=1234, avg_latency_ms=0.0)}, clear=True)

        response = await proxy.post(_COUNT, json=body)

        assert await response.json() == {"input_tokens": 1234}, (
            "count_tokens answered its own estimate: the narrowed key missed")


def _battery(system=None, **extra) -> dict:
    """One battery body: the documented turn, the case's own system, plus its extras."""
    body = {"messages": _TURN, **extra}
    if system is not None:
        body["system"] = system
    return body


# The battery evidence/cr4-probe-matrix.txt §4 measured digest-equal over 18 cases per
# axis — the documented Claude Code shape in both of its system spellings, the nonce-only
# block `_strip_billing_from_blocks` deletes outright, the shapes whose members the other
# three steps touch, and an image body.
_BATTERY = [
    pytest.param(_battery(_NONCE_SYSTEM + "\nsys"), id="string-system-nonce"),
    pytest.param(_battery([{"type": "text", "text": _NONCE_SYSTEM + "\nsys"}]),
                 id="block-system-nonce"),
    pytest.param(_battery([{"type": "text", "text": _NONCE_SYSTEM + "\nsys",
                            "cache_control": {"type": "ephemeral"}}]),
                 id="block-system-nonce-and-cache-control"),
    pytest.param(_battery([{"type": "text", "text": _NONCE_SYSTEM}]),
                 id="nonce-only-block-deleted-wholesale"),
    pytest.param(_battery(), id="no-system-key"),
    pytest.param(_battery([None]), id="system-non-mapping-block"),
    pytest.param(_battery([None, {"type": "text", "text": _NONCE_SYSTEM + "\nkeep"}]),
                 id="system-mixed-list"),
    pytest.param(_battery("plain system", tools=[{"name": "b"}, {"name": "a"}]),
                 id="tools-reordered"),
    pytest.param(_battery([{"type": "text", "text": "keep",
                            "cache_control": {"type": "ephemeral"}}]),
                 id="cache-control-on-system-only"),
    pytest.param(_battery(_NONCE_SYSTEM + "\nsys", messages=[
        {"role": "user", "content": [{"type": "image", "source": {
            "type": "base64", "media_type": "image/png", "data": "AAAA"}}]}]),
        id="image-body"),
]

# The two configurations the probe matrix ran both routes under: the shipped default
# posture, and the one the fourth step (`_normalize_date`) needs to be live at all.
_AXES = [
    pytest.param(False, "vllm", id="default"),
    pytest.param(True, "sglang", id="strip-date-sglang"),
]


class TestTheSystemStripIsTheWholeOfWhatTheDigestCanSee:
    """Narrowing the call cannot move a digest, on any body or under any posture.

    Red first in the direction of the future, not of the shipped code: the day a
    `normalize()` step starts rewriting `system`'s text, the route's key and the one
    `/v1/messages` records under diverge, and this witness is what says so. It is a pure
    claim about the two call shapes, so nothing here boots the app — the route's side of
    the same equality is `TestCountTokensKeysOnTheNormalizedDigest` above.
    """

    @pytest.mark.parametrize("strip_date, backend_type", _AXES)
    @pytest.mark.parametrize("body", _BATTERY)
    def test_the_narrowed_key_matches_the_full_normalize_one(self, mocker, body,
                                                            strip_date, backend_type):
        mocker.patch.object(config, "STRIP_IMAGES", True)
        mocker.patch.object(config, "STRIP_DATE", strip_date)
        mocker.patch.object(backend, "get_backend_state", return_value=backend_type)
        narrowed, full = _copy(body), _copy(body)

        strip_billing_nonce(narrowed)

        assert session_id(narrowed) == session_id(normalize(full, session_id="")[0]), (
            "a normalize() step beyond the system strip changed the digest")

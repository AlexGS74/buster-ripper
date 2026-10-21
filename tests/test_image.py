"""Tests for the `--strip-images` scope contract.

Which messages the flag acts on, and what the module does with a body it has no
business touching. The detector and the writer must agree on the scope (`057` F-1), and
neither may raise at a shape the flag-off path forwards for upstream to answer
(`DC-CE-1-r1`). Placeholder identity and cache retention are the sibling module,
`tests/test_image_stability.py`.
"""

import json

import pytest

from buster_ripper import config, image
from buster_ripper.normalize import normalize
from tests.image_bodies import (
    assistant,
    body,
    image_block,
    images_left,
    placeholders,
    reset_cache,
    say,
    sources,
    user,
)
from tests.upstream_stub import StubUpstream


def _sent_body(post_call):
    """The JSON body that actually went out, whichever mechanism the route used."""
    kwargs = post_call.kwargs
    return json.loads(kwargs["data"]) if "data" in kwargs else kwargs["json"]


@pytest.fixture(autouse=True)
def reset_image_cache():
    """Reset image cache before and after each test."""
    reset_cache()
    yield
    reset_cache()


class TestImageDetection:
    """Test image detection in messages."""

    def test_has_images_no_messages(self):
        """Should return False when no messages."""
        assert image.has_images({}) is False
        assert image.has_images(body()) is False

    def test_has_images_no_user_message(self):
        """Should return False when no user message."""
        assert image.has_images(body(assistant("hi"))) is False

    def test_has_images_user_text_only(self):
        """Should return False when user message has no images."""
        assert image.has_images(body(user("hello"))) is False

    def test_has_images_with_image(self):
        """Should return True when user message has image."""
        assert image.has_images(body(user([
            {"type": "text", "text": "look at this"},
            image_block({"url": "data:image/png;base64,abc"}),
        ]))) is True


# The position table `063 R-D` binds, every row a witness; rows 6-8 are `DC-CE-1-r1`'s
# shape-safety class. Each body is a lambda because the writer mutates what it is handed.
_POSITION_TABLE = [
    pytest.param(
        lambda: body(user([image_block({"id": "earliest"})]), assistant("ok"),
                     user(say("text only"))),
        True, id="earliest-user-image-latest-text-only"),
    pytest.param(
        lambda: body(user(say("first, no image")), assistant("ok"),
                     user([image_block({"id": "later"})])),
        True, id="text-only-first-image-on-a-later-turn"),
    pytest.param(
        lambda: body(user(say("first")), assistant("ok"), user([image_block({"id": "middle"})]),
                     assistant("ok"), user(say("latest, text only"))),
        True, id="image-only-in-a-middle-user-message"),
    pytest.param(
        lambda: body(user(say("a")), assistant("b"), user(say("c"))),
        False, id="no-image-in-any-user-message"),
    pytest.param(
        lambda: body(user(say("a")), assistant([image_block({"id": "assistant"})])),
        False, id="image-only-in-a-non-user-message"),
    pytest.param(lambda: {"system": "s", "messages": {"x": 1}}, False, id="messages-is-a-mapping"),
    pytest.param(lambda: {"system": "s", "messages": "hi"}, False, id="messages-is-a-string"),
    pytest.param(lambda: {"system": "s", "messages": None}, False, id="messages-is-none"),
]


class TestDetectorWriterScope:
    """F-1: one detector scope, one writer scope — all of a request's user messages."""

    @pytest.mark.parametrize("build, expected", _POSITION_TABLE)
    def test_position_table(self, build, expected):
        """`has_images` answers for every user message without raising, and the writer agrees."""
        request = build()

        assert image.has_images(request) is expected

        result = image.strip_and_cache_images(request, "session-1")

        assert result is request
        if expected:
            assert images_left(result) == []
            assert placeholders(result)
        else:
            assert sources("session-1") == []
            if isinstance(result.get("messages"), list):
                assert result == build(), "the writer changed a body it had nothing to strip"

    def test_a_message_that_is_not_a_mapping_is_skipped(self):
        """The rest of the shape class: `messages` entries that are not mappings.

        Neither stripped nor raised at, and a real user message beside them still strips.
        """
        real = user([image_block({"id": "kept"})])

        assert image.has_images(body("hi", 7, real)) is True
        assert image.has_images(body("hi", 7, user("text"))) is False

        image.strip_and_cache_images(body("hi", 7, real), "session-1")

        assert images_left(body(real)) == []
        assert len(placeholders(body(real))) == 1
        assert sources("session-1") == [{"id": "kept"}]

    def test_no_strip_string_when_there_is_nothing_to_strip(self, mocker):
        """The guarded call site claims a strip only when the detector found one."""
        mocker.patch.object(config, "STRIP_IMAGES", True)
        request = body(user(say("text only")), assistant("ok"), user(say("more text")))

        _, changes = normalize(request, session_id="session-1")

        assert "stripped and cached images" not in changes
        assert sources("session-1") == []

    def test_the_strip_string_appears_once_the_gate_opens_for_a_history(self, mocker):
        """The same call site, the row-1 shape: the gate opens and the claim is true."""
        mocker.patch.object(config, "STRIP_IMAGES", True)
        request = body(user([image_block({"id": "historical"})]), assistant("ok"),
                       user(say("text only")))

        _, changes = normalize(request, session_id="session-1")

        assert "stripped and cached images" in changes
        assert images_left(request) == []

    async def test_a_historical_image_is_stripped_through_the_route(self, proxy, mocker):
        """The route-level witness: a text-only latest turn still strips an earlier image."""
        mocker.patch.object(config, "STRIP_IMAGES", True)
        sent = mocker.patch.object(proxy.server.app["session"], "post",
                                   return_value=StubUpstream())
        base64 = "QUJD" * 40
        request = {"system": "s", "max_tokens": 16, "messages": [
            user([{"type": "text", "text": "here"},
                  image_block({"type": "base64", "media_type": "image/png", "data": base64})]),
            assistant("ok"),
            user(say("and now?")),
        ]}

        response = await proxy.request("POST", "/v1/messages", json=request)

        assert response.status == 200
        forwarded = {"messages": _sent_body(sent.call_args)["messages"]}
        assert images_left(forwarded) == []
        assert len(placeholders(forwarded)) == 1
        assert base64 not in repr(forwarded), "a base64 payload reached the upstream"

    @pytest.mark.parametrize("strip_images", [True, False], ids=["flag-on", "flag-off"])
    async def test_a_malformed_body_gets_the_same_answer_whatever_the_flag_says(
        self, proxy, mocker, strip_images
    ):
        """`DC-CE-1-r1`'s two-run comparison: the flag does not change the status class.

        A `messages` that is not a list is upstream's to answer, in both runs.
        """
        mocker.patch.object(config, "STRIP_IMAGES", strip_images)
        sent = mocker.patch.object(proxy.server.app["session"], "post",
                                   return_value=StubUpstream())

        response = await proxy.request("POST", "/v1/messages",
                                       json={"system": "s", "messages": {"x": 1}})

        assert response.status == 200, "the proxy answered for a body it should have forwarded"
        assert _sent_body(sent.call_args)["messages"] == {"x": 1}

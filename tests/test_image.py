"""Tests for image handling with caching and eviction.

The `--strip-images` scope contract: which messages the flag acts on, and what the
module does with a body it has no business touching.

The detector and the writer must agree on the scope (`057` F-1), and neither may raise
at a shape the flag-off path forwards for upstream to answer (`DC-CE-1-r1`) — enabling a
content-transform flag must not change the status class of a request the proxy would
otherwise forward. Placeholder identity and cache retention are the sibling module,
`tests/test_image_stability.py`.

Stamps come from the module's monotonic source, injected through `image._now`, and no
witness sleeps on a clock (`DC-CE-4-r1`).
"""

import asyncio
import contextlib
import json
import logging

import pytest

from buster_ripper import config, image
from buster_ripper.normalize import normalize
from tests.image_bodies import (
    assistant,
    body,
    image_block,
    images_left,
    one_image,
    placeholders,
    reset_cache,
    say,
    sources,
    stamp_source,
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


# The position table `063 R-D` binds, every row a witness rather than an idea. Rows 1-3
# retire the latest-only abbreviation, the earliest-only one, and both at once; row 5
# keeps the writer's user-only scope; rows 6-8 are `DC-CE-1-r1`'s shape-safety class.
# Each body comes from the lambda, because the writer mutates the one it is handed, and
# the text-only turns carry list content so an out-of-scope turn cannot hide an
# abbreviation from a detector that stopped early.
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
        """The rest of the shape class: entries of `messages` that are not mappings.

        A list holding a string and a number is a body upstream rejects. The flag neither
        strips it nor raises at it, and a real user message sitting beside them is still
        handled — the detector and the writer have to agree about that list too.
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
        """`DC-CE-1-r1`'s two-run comparison, in process: the flag does not change the status class.

        A `messages` that is not a list is not this flag's business to validate, and
        nothing between the handler and aiohttp catches — so before the fix the flag-on
        run raised inside `reversed(messages)` and aiohttp answered its own 500 with
        `force_close()`, while the flag-off run proxied the body and let upstream reply to
        it. Syntax is already answered at the parse boundary; shape is upstream's.
        """
        mocker.patch.object(config, "STRIP_IMAGES", strip_images)
        sent = mocker.patch.object(proxy.server.app["session"], "post",
                                   return_value=StubUpstream())

        response = await proxy.request("POST", "/v1/messages",
                                       json={"system": "s", "messages": {"x": 1}})

        assert response.status == 200, "the proxy answered for a body it should have forwarded"
        assert _sent_body(sent.call_args)["messages"] == {"x": 1}


class TestImageCaching:
    """Test image stripping and caching."""

    def test_strip_images_replaces_with_placeholder(self):
        """Should replace images with text placeholders, in place."""
        request = one_image({"url": "data:image/png;base64,abc"})

        result = image.strip_and_cache_images(request, "session-1")

        assert result["messages"][0]["content"][0]["type"] == "text"
        assert images_left(result) == []
        assert placeholders(result) == ["img_04b45e9ae5ea"]

    def test_strips_multiple_images(self):
        """Should handle multiple images in one message."""
        request = body(user([image_block({"url": "data:1"}),
                             {"type": "text", "text": "and"},
                             image_block({"url": "data:2"})]))

        result = image.strip_and_cache_images(request, "session-1")

        assert len(result["messages"][0]["content"]) == 3
        assert result["messages"][0]["content"][1]["type"] == "text"
        assert result["messages"][0]["content"][1]["text"] == "and"

    def test_cache_isolated_by_session(self):
        """Should isolate cache by session ID — distinct sources, so neither write dedupes."""
        image.strip_and_cache_images(one_image({"url": "data:1"}), "session-1")
        image.strip_and_cache_images(one_image({"url": "data:2"}), "session-2")

        assert "session-1" in image._image_cache
        assert "session-2" in image._image_cache

    def test_get_cached_image_retrieves_source(self):
        """Should retrieve cached image source, under the key the placeholder names."""
        request = one_image({"url": "data:1"})
        image.strip_and_cache_images(request, "session-1")

        assert image.get_cached_image("session-1", placeholders(request)[0]) == {"url": "data:1"}

    def test_get_cached_image_returns_none_for_missing(self):
        """Should return None for non-existent key."""
        assert image.get_cached_image("session-1", "img_999") is None


class TestImageCacheTTL:
    """Test TTL-based expiration on the read path. `IMAGE_CACHE_TTL` is absolute age
    since first cached, so a re-send does not extend retention (`063 R-C`)."""

    def test_cleanup_removes_expired(self, mocker):
        """Should remove expired entries during cleanup."""
        mocker.patch.object(config, "IMAGE_CACHE_TTL", 100)
        clock = stamp_source(mocker)
        request = one_image({"url": "data:1"})
        image.strip_and_cache_images(request, "session-1")
        key = placeholders(request)[0]
        assert len(image._image_cache["session-1"]) == 1

        clock.advance(101)

        assert image.get_cached_image("session-1", key) is None

    def test_non_expired_images_remain(self, mocker):
        """Should keep non-expired entries during cleanup."""
        mocker.patch.object(config, "IMAGE_CACHE_TTL", 10_000)
        clock = stamp_source(mocker)
        request = one_image({"url": "data:1"})
        image.strip_and_cache_images(request, "session-1")

        clock.advance(1)

        assert image.get_cached_image("session-1", placeholders(request)[0]) == {"url": "data:1"}


class TestImageCacheLRUEviction:
    """Test LRU-based cache eviction by count. Each case names its own pre-state; a size
    assertion inside a loop is a statement about accumulation, not about the key rule."""

    def test_evicts_when_over_limit(self, mocker):
        """Should evict the least recently touched entries when over max size. Pre-state: empty."""
        mocker.patch.object(config, "IMAGE_CACHE_MAX_SIZE", 2)
        clock = stamp_source(mocker)
        reset_cache()

        keys = []
        for i in range(3):
            request = one_image({"id": i})
            image.strip_and_cache_images(request, "session-1")
            keys.append(placeholders(request)[0])
            clock.advance(1)

        # Should have evicted oldest to stay at max
        assert len(image._image_cache["session-1"]) == 2

        # First image should be evicted (oldest)
        assert image.get_cached_image("session-1", keys[0]) is None
        assert image.get_cached_image("session-1", keys[1]) is not None
        assert image.get_cached_image("session-1", keys[2]) is not None

    def test_no_eviction_when_under_limit(self, mocker):
        """Should not evict when under max size. Pre-state: empty."""
        mocker.patch.object(config, "IMAGE_CACHE_MAX_SIZE", 10)
        reset_cache()

        for i in range(3):
            image.strip_and_cache_images(one_image({"id": i}), "session-1")

        # All should remain
        assert len(image._image_cache["session-1"]) == 3

    def test_lru_eviction_respects_sessions(self, mocker):
        """Should evict per-session, not globally.

        Pre-state: session-1 accumulates across the loop — three distinct sources against
        a bound of 2 — which is the point; session-2's single entry is untouched by it.
        """
        mocker.patch.object(config, "IMAGE_CACHE_MAX_SIZE", 2)
        clock = stamp_source(mocker)
        reset_cache()

        for i in range(3):
            image.strip_and_cache_images(one_image({"session": 1, "id": i}), "session-1")
            clock.advance(1)
        image.strip_and_cache_images(one_image({"session": 2, "id": 0}), "session-2")

        assert len(image._image_cache["session-1"]) == 2
        assert len(image._image_cache["session-2"]) == 1


class TestClearSessionCache:
    """Test session cache clearing."""

    def test_clear_removes_session(self):
        """Should remove entire session cache."""
        image.strip_and_cache_images(one_image({"url": "data:1"}), "session-1")
        assert "session-1" in image._image_cache

        image.clear_session_cache("session-1")
        assert "session-1" not in image._image_cache

    def test_clear_nonexistent_session(self):
        """Should not error when clearing nonexistent session."""
        image.clear_session_cache("nonexistent")


async def _one_tick(poll) -> None:
    """Run a periodic loop's first sweep, then cancel it at its sleep await."""
    task = asyncio.create_task(poll())
    await asyncio.sleep(0)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


def _stop_at_sleep(mocker, intervals: list) -> None:
    """Record the loop's next interval and end it there, so no clock is waited on."""
    async def _stop(seconds):
        intervals.append(seconds)
        raise asyncio.CancelledError()

    mocker.patch.object(image.asyncio, "sleep", _stop)


class TestPollExpiry:
    """The periodic sweep expires the cache while no request arrives at all."""

    async def test_sweep_drops_an_expired_image_with_no_request(self, mocker):
        """An idle process drops a source past its TTL on one sweep tick."""
        mocker.patch.object(config, "IMAGE_CACHE_TTL", 100)
        clock = stamp_source(mocker)
        request = one_image({"url": "data:1"})
        image.strip_and_cache_images(request, "session-1")
        key = placeholders(request)[0]
        clock.advance(101)

        await _one_tick(image.poll_expiry)

        assert key not in image._image_cache.get("session-1", {})

    async def test_sweep_keeps_a_source_inside_the_ttl(self, mocker):
        """A source younger than the TTL still resolves after a sweep tick."""
        mocker.patch.object(config, "IMAGE_CACHE_TTL", 100)
        clock = stamp_source(mocker)
        request = one_image({"url": "data:1"})
        image.strip_and_cache_images(request, "session-1")
        key = placeholders(request)[0]
        clock.advance(99)

        await _one_tick(image.poll_expiry)

        assert image.get_cached_image("session-1", key) == {"url": "data:1"}

    async def test_a_non_positive_ttl_sleeps_a_second(self, mocker):
        """A TTL of 0 cannot make the loop spin: the interval floors at 1 s."""
        mocker.patch.object(config, "IMAGE_CACHE_TTL", 0)
        intervals = []
        _stop_at_sleep(mocker, intervals)

        with pytest.raises(asyncio.CancelledError):
            await image.poll_expiry()

        assert intervals == [1]

    async def test_a_raising_sweep_is_logged_and_the_loop_continues(
        self, mocker, caplog
    ):
        """A sweep that raises is logged, and the loop still reaches its sleep."""
        mocker.patch.object(config, "IMAGE_CACHE_TTL", 100)
        sweeps = {"n": 0}

        def flaky_sweep():
            sweeps["n"] += 1
            if sweeps["n"] == 1:
                raise RuntimeError("boom")

        mocker.patch.object(image, "cleanup_expired_images", flaky_sweep)
        intervals = []
        _stop_at_sleep(mocker, intervals)
        caplog.set_level(logging.ERROR, logger="buster-ripper")

        with pytest.raises(asyncio.CancelledError):
            await image.poll_expiry()

        assert sweeps["n"] == 1
        assert intervals == [100]
        assert any("periodic expiry failed" in r.message for r in caplog.records)


class TestMakeDataUrl:
    """Test data URL construction."""

    def test_make_data_url_adds_prefix(self):
        """Should add data URL prefix to raw base64."""
        assert image.make_data_url("abc123", "image/png") == "data:image/png;base64,abc123"

    def test_make_data_url_strips_existing_prefix(self):
        """Should strip existing prefix and re-add."""
        result = image.make_data_url("data:image/jpeg;base64,xyz789", "image/jpeg")
        assert result == "data:image/jpeg;base64,xyz789"

    def test_make_data_url_handles_base64_in_data(self):
        """Should handle data that contains 'base64' substring."""
        result = image.make_data_url("base64encodeddata", "image/png")
        # Should strip everything before and including the last 'base64'
        assert "base64encodeddata" not in result

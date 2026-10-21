"""Tests for the image cache itself: stripping, TTL expiry, LRU eviction and data URLs.

The route and scope witnesses are `tests/test_image.py`; placeholder identity and write-path
retention are `tests/test_image_stability.py`. Stamps come from `image._now`, and no witness
sleeps on a clock (`DC-CE-4-r1`).
"""

import asyncio
import contextlib
import logging

import pytest

from buster_ripper import config, image
from tests.image_bodies import (
    body,
    image_block,
    images_left,
    one_image,
    placeholders,
    reset_cache,
    stamp_source,
    user,
)


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
        reset_cache()
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
        reset_cache()
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

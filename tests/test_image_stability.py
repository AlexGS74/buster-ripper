"""Tests for placeholder identity and cache retention — the two stability contracts.

Sibling of `tests/test_image.py`, which carries the scope contract. Two things a
forwarded request depends on that the module did not previously give it:

* **Identity** (`057` F-2, `063 R-A`) — a placeholder key is a deterministic function of
  one image `source`, so the same source carries the same marker text on every request
  that carries it. That byte stability in the forwarded body is the marker's whole
  contract; the cached source has no production reader and nothing restores an image
  from its placeholder (`DC-S13-GAP-r1`).
* **Retention** (`057` F-3, `063 R-B`/`R-C`, `DC-CE-2-r1`…`DC-CE-5-r1`) — both advertised
  bounds, TTL and size, bind on the write path; retention is absolute age; an emptied
  session is reclaimed; and every stamp comes from a non-decreasing clock.

Every case names its own starting state, because the reset is per test and not per call
(`DC-claim-289-r1`), and no witness sleeps on a clock.
"""

import time

import pytest

from buster_ripper import config, image
from buster_ripper.normalize import normalize
from tests.image_bodies import (
    body,
    image_block,
    ids,
    one_image,
    placeholders,
    reset_cache,
    sources,
    stamp_source,
    user,
)

# `063 R-A`'s formula pinned on bytes: the slice comes off the digest of the canonical
# serialization, never off the serialization itself. Realistic base64 sources share their
# first characters, so slicing the operand would hand every image in a session the same
# key — the collision this member exists to remove.
_PIN_SOURCE = {"url": "data:image/png;base64,abc"}
_PIN_KEY = "img_04b45e9ae5ea"


@pytest.fixture(autouse=True)
def reset_image_cache():
    """Reset image cache before and after each test."""
    reset_cache()
    yield
    reset_cache()


class TestPlaceholderIdentity:
    """One image source, one placeholder key, on every request that carries it."""

    def test_the_key_is_the_digest_of_the_canonical_source(self):
        """The formula, pinned on the bytes the flag puts in the forwarded body."""
        request = one_image(_PIN_SOURCE)

        image.strip_and_cache_images(request, "session-1")

        assert placeholders(request) == [_PIN_KEY]

    def test_one_source_two_requests_one_text(self):
        """A replayed history keeps its placeholder byte-identical."""
        first, second = one_image({"url": "data:1"}), one_image({"url": "data:1"})

        image.strip_and_cache_images(first, "session-1")
        image.strip_and_cache_images(second, "session-1")

        assert placeholders(first) == placeholders(second)

    def test_key_order_in_the_source_does_not_move_the_key(self):
        """Equal sources built in different insertion orders agree — kills `repr()`."""
        left, right = one_image({"a": 1, "b": 2}), one_image({"b": 2, "a": 1})

        image.strip_and_cache_images(left, "session-1")
        image.strip_and_cache_images(right, "session-2")

        assert placeholders(left) == placeholders(right)

    def test_the_other_call_site_sees_the_same_key_through_a_cache_control(self, mocker):
        """The referent discriminator: `normalize`'s step 0 strips while `cache_control` is still on the block.

        `normalize.py:212` runs the image step before `_normalize_cache_control` at `:223`,
        while `routes/messages.py:34` has already deleted it by the time that site's `:38`
        guard opens. A key over the whole block would therefore differ per call site for
        one unchanged image — F-2's defect class, relocated. The step at `:212` is
        unreachable from production (`normalize` is only ever called with an empty
        `session_id`), which is why this is authored as the direct unit call it has to be
        and not as an integration path (`DC-S13-CE4-r1`).
        """
        mocker.patch.object(config, "STRIP_IMAGES", True)
        at_normalize = body(user([{"type": "image", "source": {"url": "data:1"},
                                   "cache_control": {"type": "ephemeral"}}]))
        at_messages = one_image({"url": "data:1"})

        normalize(at_normalize, session_id="session-A")
        image.strip_and_cache_images(at_messages, "session-B")

        assert placeholders(at_normalize) == placeholders(at_messages)

    def test_two_sources_two_keys(self):
        """Distinct sources never share an identity, however alike their bytes."""
        left, right = one_image({"url": "data:1"}), one_image({"url": "data:2"})

        image.strip_and_cache_images(left, "session-1")
        image.strip_and_cache_images(right, "session-1")

        assert len(set(placeholders(left) + placeholders(right))) == 2

    def test_one_source_repeated_in_one_request_is_one_entry(self):
        """Pre-state: an empty cache, then two blocks carrying one source in a single call."""
        reset_cache()
        request = body(user([image_block({"url": "same"}), image_block({"url": "same"})]))

        image.strip_and_cache_images(request, "session-1")

        assert len(placeholders(request)) == 2
        assert len(set(placeholders(request))) == 1
        assert len(image._image_cache["session-1"]) == 1

    def test_two_sources_in_one_message_at_the_cap_do_not_overwrite(self, mocker):
        """The converse of the row above: distinct sources at the cap get distinct keys and evict.

        Pre-state built here, not inherited: this session's map holds exactly
        `IMAGE_CACHE_MAX_SIZE` entries from two earlier writes. Under the count-derived key
        both blocks recomputed `img_2`, the second replaced the first, and both forwarded
        one byte-identical token for two different client images while the first block's
        source was referenced by nothing (`DC-S13-CE1-r1`).
        """
        mocker.patch.object(config, "IMAGE_CACHE_MAX_SIZE", 2)
        reset_cache()
        image.strip_and_cache_images(one_image({"id": "at-cap-1"}), "session-1")
        image.strip_and_cache_images(one_image({"id": "at-cap-2"}), "session-1")
        assert len(image._image_cache["session-1"]) == 2

        request = body(user([image_block({"id": "P"}), image_block({"id": "Q"})]))
        image.strip_and_cache_images(request, "session-1")

        assert len(set(placeholders(request))) == 2, "two sources forwarded one placeholder"
        assert len(image._image_cache["session-1"]) == 2
        assert ids("session-1") == {"P", "Q"}

    def test_one_source_two_sessions_one_text(self):
        """`session_id` is not an operand: the cache is already keyed by it."""
        in_a, in_b = one_image({"url": "data:1"}), one_image({"url": "data:1"})

        image.strip_and_cache_images(in_a, "session-1")
        image.strip_and_cache_images(in_b, "session-2")

        assert placeholders(in_a) == placeholders(in_b)

    def test_source_less_blocks_share_the_empty_source_key(self):
        """No special case for a block with no `source`: `{}` is one source, so one key.

        `:80`'s `.get("source", {})` default means every source-less block hashes the empty
        mapping, and under content addressing that shared key is the correct reading of one
        source, one placeholder — not a collision to fix (`DC-S13-CE1-r1`).
        """
        first, second = one_image({}), one_image({})

        image.strip_and_cache_images(first, "session-1")
        image.strip_and_cache_images(second, "session-2")

        assert len(set(placeholders(first) + placeholders(second))) == 1

    def test_eviction_keeps_running_and_never_reuses_a_live_key(self, mocker):
        """Eviction liveness: the bound is enforced on every insert, and no key lands on a live entry.

        Pre-state: an empty cache, then eight distinct sources through one session — the
        101st-image shape at the default bound, at 3 for the cost of the loop. Under the
        count-derived key the fourth image hit the live `img_3`, the replace kept its
        position, the size never exceeded the bound again, and nothing was ever evicted
        again (`DC-CE-5-r1`).
        """
        mocker.patch.object(config, "IMAGE_CACHE_MAX_SIZE", 3)
        mocker.patch.object(config, "IMAGE_CACHE_TTL", 10_000)
        reset_cache()

        for i in range(8):
            before = set(image._image_cache.get("session-1", {}))
            request = one_image({"id": i})
            image.strip_and_cache_images(request, "session-1")
            after = set(image._image_cache["session-1"])

            assert placeholders(request)[0] not in before, "a placeholder named a live entry"
            assert len(after) == min(len(before) + 1, 3)
            if len(before) == 3:
                assert len(before - after) == 1, "the bound was exceeded with nothing removed"
            assert {"id": i} in sources("session-1")


class TestWritePathReclamation:
    """The write path's tail sweep prunes every session and reclaims the emptied ones."""

    def test_w1_same_session_expiry(self, mocker):
        """A write sweeps its own session's expired payload and keeps the new one."""
        mocker.patch.object(config, "IMAGE_CACHE_TTL", 100)
        clock = stamp_source(mocker)
        image.strip_and_cache_images(one_image({"url": "old"}), "A")

        clock.advance(101)
        image.strip_and_cache_images(one_image({"url": "new"}), "A")

        assert "A" in image._image_cache
        assert sources("A") == [{"url": "new"}]

    def test_w2_abandoned_session_is_dropped(self, mocker):
        """Another session's write reclaims an abandoned session's dict entirely.

        The global axis, and the witness that the outer-key drop is new code inside the
        janitor rather than a new call to it (`DC-CE-3-r1`).
        """
        mocker.patch.object(config, "IMAGE_CACHE_TTL", 100)
        clock = stamp_source(mocker)
        image.strip_and_cache_images(one_image({"url": "A"}), "A")
        image.strip_and_cache_images(one_image({"url": "B"}), "B")

        clock.advance(101)
        image.strip_and_cache_images(one_image({"url": "B2"}), "B")

        assert "A" not in image._image_cache
        assert "B" in image._image_cache

    def test_w3_a_write_after_its_session_aged_out_does_not_raise(self, mocker):
        """What the positional prohibition prevents: no delete between the ensure and the write.

        A drop placed there would surface to the client as an HTTP 500, because neither
        guarded call site has a `try` and the one middleware does not catch.
        """
        mocker.patch.object(config, "IMAGE_CACHE_TTL", 100)
        clock = stamp_source(mocker)
        image.strip_and_cache_images(one_image({"url": "A"}), "A")

        clock.advance(101)
        request = one_image({"url": "A2"})
        image.strip_and_cache_images(request, "A")

        assert len(placeholders(request)) == 1
        assert sources("A") == [{"url": "A2"}]

    def test_w4_the_read_path_expires_but_keeps_the_session(self, mocker):
        """`_cleanup_expired_images` stays expiry-only: the read path indexes right after it."""
        mocker.patch.object(config, "IMAGE_CACHE_TTL", 100)
        clock = stamp_source(mocker)
        request = one_image({"url": "A"})
        image.strip_and_cache_images(request, "A")
        key = placeholders(request)[0]

        clock.advance(101)

        assert image.get_cached_image("A", key) is None
        assert "A" in image._image_cache

    def test_w5_a_write_onto_an_aged_entry_carries_its_age_forward(self, mocker):
        """A re-send past the TTL is not a fresh entry, and this call's own sweep drops it.

        Pre-state: one entry, aged out with no sweep in between. A build that re-created
        the stamp on a same-key write passes W1-W4 and fails here, because that sweep would
        find an age-zero entry (`DC-CE-5-r1`).
        """
        mocker.patch.object(config, "IMAGE_CACHE_TTL", 100)
        clock = stamp_source(mocker)
        first = one_image({"url": "same"})
        image.strip_and_cache_images(first, "A")
        key = placeholders(first)[0]

        clock.advance(101)
        again = one_image({"url": "same"})
        image.strip_and_cache_images(again, "A")

        assert placeholders(again) == [key], "the placeholder text moved on a re-send"
        assert key not in image._image_cache.get("A", {}), "the aged entry survived its own sweep"

    def test_w6_eviction_orders_on_recency_not_on_age(self, mocker):
        """The sort key is the moved slot: leaving it on `first_seen` keeps every size green and is FIFO."""
        mocker.patch.object(config, "IMAGE_CACHE_MAX_SIZE", 2)
        mocker.patch.object(config, "IMAGE_CACHE_TTL", 10_000)
        clock = stamp_source(mocker)
        reset_cache()

        image.strip_and_cache_images(one_image({"id": "X"}), "A")
        clock.advance(1)
        image.strip_and_cache_images(one_image({"id": "Y"}), "A")
        clock.advance(1)
        image.strip_and_cache_images(one_image({"id": "X"}), "A")
        clock.advance(1)
        image.strip_and_cache_images(one_image({"id": "Z"}), "A")

        assert ids("A") == {"X", "Z"}, "the touched entry was evicted on its age, not its recency"

    def test_the_stamp_source_is_a_non_decreasing_clock(self):
        """`DC-CE-4-r1`: both stamps come from `time.monotonic`, which cannot step backwards.

        The pin the injection cannot supply on its own: a build that rebound `_now` to
        `time.time()` keeps every injected witness green while re-opening the entry a clock
        adjustment makes immune to both advertised bounds.
        """
        assert image._now is time.monotonic

    def test_w7_a_wall_clock_adjustment_cannot_pin_an_entry(self, mocker):
        """Stamps and every comparison come from one source, not from the wall clock.

        No stamp injection here, deliberately: this row's wrong build reads the wall clock,
        so the adjustment has to land on a name that build actually resolves at call time.
        The wall clock is pinned at 1000 s and then at 400 s — one step back of 600 s — and
        three writes go in past the size bound. A wall-clock module leaves the first entry
        holding the largest stamp: last in every later deletion slice, and negatively aged,
        so neither advertised bound can ever remove it (`DC-CE-4-r1`).
        """
        mocker.patch.object(config, "IMAGE_CACHE_MAX_SIZE", 2)
        mocker.patch.object(config, "IMAGE_CACHE_TTL", 100)
        wall = mocker.patch("time.time", side_effect=[1000.0, 400.0, 401.0])
        reset_cache()

        image.strip_and_cache_images(one_image({"id": "X"}), "A")
        image.strip_and_cache_images(one_image({"id": "Y"}), "A")
        image.strip_and_cache_images(one_image({"id": "Z"}), "A")

        assert len(image._image_cache["A"]) == 2, "the bound was exceeded with nothing removed"
        assert ids("A") == {"Y", "Z"}, "the entry stamped before the adjustment pinned the slice"
        now = image._now()
        assert all(now - first_seen >= 0 for _, first_seen, _ in image._image_cache["A"].values())
        assert wall.call_count == 0, "a stamp or an age came off the wall clock"

    def test_an_empty_messages_body_is_returned_before_the_sweep(self):
        """Recorded boundary, not a requirement: this path never sweeps, by construction.

        Both guarded call sites gate on `has_images`, which is False for an absent or empty
        list, so no request reaches this branch; the early return stays and the tail sweep
        sits after the loop it skips (`065` footnote F3).
        """
        request = {"messages": []}

        assert image.strip_and_cache_images(request, "A") is request
        assert image._image_cache == {"A": {}}

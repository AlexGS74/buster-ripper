VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`tests/test_image.py` asserts TTL expiry only through a `get_cached_image` call

## What it actually does
TTL behaviour is exercised only in `TestImageCacheTTL` (`config.IMAGE_CACHE_TTL` is patched only at lines 139 and 160). The expiry itself is asserted only in `test_cleanup_removes_expired`: after `time.sleep(1.1)`, the sole post-expiry assertion is `result = image.get_cached_image("session-1", "img_0"); assert result is None` (lines 155–156). No assertion after the sleep inspects `image._image_cache` directly — the `len(image._image_cache["session-1"]) == 1` on line 149 is a pre-sleep precondition check, not an expiry assertion. The companion test (`test_non_expired_images_remain`) likewise observes cache state only through a `get_cached_image` call (line 170).

## Source
/Users/alexhubitski/Projects/buster-ripper/tests/test_image.py, lines 134–172 (claim subject is the test suite, so the test file is the production surface per the P10 exception)

## Source Version
Mutable worktree file; no snapshot supplied. Content fingerprint below.

## Invalidates On
- Any edit to tests/test_image.py adding a post-expiry assertion that inspects image._image_cache (or another helper) instead of calling get_cached_image
- Any edit adding a new IMAGE_CACHE_TTL fixture/test that asserts expiry through a path other than get_cached_image

## Checked At
2026-10-05T18:52:36Z

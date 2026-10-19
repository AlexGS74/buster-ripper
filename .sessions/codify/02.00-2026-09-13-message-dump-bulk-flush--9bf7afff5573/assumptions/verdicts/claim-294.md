VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
An autouse `tests/test_image.py` fixture clears `_image_cache` before and after each test

## What it actually does
`reset_image_cache` (lines 10–15) is declared `@pytest.fixture(autouse=True)`, so it applies to every test in the module without being requested. Its body calls `image._image_cache.clear()` before the `yield` (setup, i.e. before each test) and `image._image_cache.clear()` again after the `yield` (teardown, i.e. after each test). It is the only fixture defined in the file — `@pytest.fixture` appears exactly once (line 10).

## Source
/Users/alexhubitski/Projects/buster-ripper/tests/test_image.py, lines 10–15 (claim subject is the test suite, so the test file is the production surface per the P10 exception)

## Source Version
Mutable worktree file; no snapshot supplied. Content fingerprint below.

## Invalidates On
- Any edit removing autouse=True or dropping either the pre-yield or post-yield image._image_cache.clear() call
- Any edit adding a second fixture that also clears (or stops clearing) _image_cache, changing the "an autouse fixture clears before and after each test" pattern

## Checked At
2026-10-05T18:52:36Z

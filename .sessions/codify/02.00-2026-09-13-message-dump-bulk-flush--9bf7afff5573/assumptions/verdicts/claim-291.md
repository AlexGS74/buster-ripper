VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`tests/test_image.py`'s eviction cases cache a distinct `source` mapping per image

## What it actually does
Every image in `TestImageCacheLRUEviction` (L175-242) is given its own distinct `source` dict,
and each is a freshly-constructed literal, so no two images share content or object identity:

- `test_evicts_when_over_limit` — L187 `{"id": i}` inside `for i in range(3)` (the whole `body`
  literal is inside the loop, L184-189) → `{"id":0}`, `{"id":1}`, `{"id":2}` cached under
  `img_0`/`img_1`/`img_2`.
- `test_no_eviction_when_under_limit` — L208 `{"id": i}`, same three distinct mappings.
- `test_lru_eviction_respects_sessions` — L225 `{"session": 1, "id": i}` → three distinct
  mappings for `session-1`, and L234 `{"session": 2, "id": 0}`, distinct from those by the
  `session` key, for `session-2`.

The claim holds whether "eviction cases" is read as only the two tests that actually evict
(L178, L216) or as the whole LRU class including the under-limit control (L200).

One thing to record for whoever reads this: the distinctness is constructed but never asserted.
The eviction tests check only presence/absence — L196 `img_0 is None`, L197-198
`img_1/img_2 is not None` — and per-session counts (L193, L214, L240, L242). No eviction
assertion compares the returned `source` against the expected mapping, so a cache that keyed
content incorrectly while still evicting the right count would pass. Source identity is
asserted only outside this class, at L126 and L172.

## Source
/Users/alexhubitski/Projects/buster-ripper/tests/test_image.py L175-242 (whole file read; the
claim's subject is the test suite, so the test file is the production surface for it — P10
exception)

## Source Version
tests/test_image.py, 286 lines, content identified by the fingerprint below; repository
revision not consulted (Git metadata is out of scope for this role)

## Source Fingerprint
sha256:b603bfd1624e30eab1c6d57390194a71cfd3716525abc33c742292bc5a3dfda0

## Invalidates On
- Replacing a loop's `source` literal with a hoisted constant or a body built outside the loop
  (e.g. reusing one `body` object across iterations at L184-189 or L205-210), which would cache
  the same mapping per image.
- Changing L187/L208 to a loop-invariant value such as `{"id": 0}`, or editing L225/L234 so
  `session-1` and `session-2` sources collide.
- Moving an eviction case into a parametrised test that reuses one fixture body for all images.

## Checked At
2026-10-05T18:52:58Z

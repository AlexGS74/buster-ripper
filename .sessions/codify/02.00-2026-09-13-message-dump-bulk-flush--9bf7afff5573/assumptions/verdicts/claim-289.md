VERDICT: DOES-NOT-HOLD
PROMOTION: ELIGIBLE

## Claim
Every `strip_and_cache_images` call in `tests/test_image.py` adds one image to a freshly reset cache

## What it actually does
The cache is reset once **per test**, not once per call, and several calls in the file are not
single-image calls either.

`tests/test_image.py` L10-15 defines an autouse function-scoped fixture that clears
`image._image_cache` before the test function runs and again after it yields. Nothing inside a
test resets the cache between statements, so any second-or-later call in the same test sees a
cache already populated by that test's own earlier calls.

The 12 call sites, with freshness:

- Fresh (first call of their test): L75, L93, L108, L123, L148, L169, L257, plus the first
  iteration of each of the three loops at L190, L211, L228.
- **Not fresh**: L109 (`session-2` call runs after the `session-1` call at L108 in the same
  test), L190 iterations 2 and 3, L211 iterations 2 and 3, L228 iterations 2 and 3, and L237
  (the `session-2` call runs after three `session-1` calls earlier in the same test).
  That is 8 of the 15 actual call executions in the file.

The "one image" half is also contradicted at L93 (`test_strips_multiple_images`), whose body
carries two image blocks (L86 and L88) in a single message — one call, two images in. The
per-call cache-entry count is the implementation's behaviour in the image module, which is
outside the bounded source root supplied for this claim; the file's own assertions corroborate
one-entry-per-single-image call (L149 `len(...) == 1` after one call; L214 `len(...) == 3`
after three one-image calls), and the L93 body is sufficient to refute "adds one image" as a
universal over calls.

Either conjunct alone refutes the claim as written.

## Source
/Users/alexhubitski/Projects/buster-ripper/tests/test_image.py (whole file read; the claim's
subject is the test suite, so the test file is the production surface for it — P10 exception)

## Source Version
tests/test_image.py, 286 lines, content identified by the fingerprint below; repository
revision not consulted (Git metadata is out of scope for this role)

## Source Fingerprint
sha256:b603bfd1624e30eab1c6d57390194a71cfd3716525abc33c742292bc5a3dfda0

## Invalidates On
- A change to the `reset_image_cache` fixture (L10-15) from function scope to a per-call reset,
  or the introduction of an explicit `image._image_cache.clear()` before a call site, which
  would make the freshness conjunct true for those calls.
- Splitting L109, L237, or the loop bodies at L190/L211/L228 into separate test functions.
- Rewriting L93's body to a single image block (removes the multi-image counterexample; the
  freshness counterexamples above would still stand).

## Checked At
2026-10-05T18:52:58Z

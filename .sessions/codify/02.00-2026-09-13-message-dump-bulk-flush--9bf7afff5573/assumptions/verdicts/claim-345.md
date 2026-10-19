VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
Every whole-mapping emptying statement for `_image_cache` in `tests/test_image.py` is inside the autouse fixture, not inside a test body

## What it actually does
The file contains exactly two whole-mapping emptying statements for `_image_cache`, at lines 13 and 15, and both are inside `reset_image_cache` — the single fixture declared `@pytest.fixture(autouse=True)` at line 10 (line 13 before the `yield`, line 15 after it). No test body empties the whole mapping: every other reference to `_image_cache` in a test body is a read-only assertion (`"x" in image._image_cache` / `len(image._image_cache[...])`), at lines 111, 112, 149, 193, 214, 240, 242, 258, 261.

Exhaustive check over the whole 286-line file for alternative emptying forms found none: no `_image_cache = {}` rebinding, no `del image._image_cache[...]`, no `popitem()`, and no `.clear()` call on any object other than lines 13 and 15.

## Scope note (term boundary)
`image.clear_session_cache(...)` is called from two test bodies (lines 260, 266). Those are **not** whole-mapping emptying statements for `_image_cache`: the call passes an explicit session-id argument, so its scope is one session's entries, not the mapping as a whole — and the assertion at 261 confirms the mapping object itself survives the call. Under that plain reading the claim holds. Note that a reader who broadened "whole-mapping emptying" to "any call that removes cache entries" would sweep those two scoped calls in and read the claim as false; that reading contradicts both the argument at the call site and the wording of the claim itself.

## Source
`/Users/alexhubitski/Projects/buster-ripper/tests/test_image.py` — supplied `Source root`; read in full (lines 1–286).

Per the P10 exception, the claim's subject IS the test suite: it asserts where statements are *written* in a test file, not what the production module does at runtime. The test file is therefore the primary and correct surface for this claim, not corroborating evidence. Nothing was run (P1); this is a static-membership claim settled by reading the file's contents.

## Source Version
Unavailable as a revision identifier — the Git-metadata boundary in this role forbids resolving repository state. Content is instead pinned by the fingerprint below, which is the stability requirement for promotion.

## Source Fingerprint
sha256:b603bfd1624e30eab1c6d57390194a71cfd3716525abc33c742292bc5a3dfda0

## Invalidates On
- Any edit to `tests/test_image.py` that adds a whole-mapping emptying statement (`image._image_cache.clear()`, an `_image_cache = {}` rebinding, or a `del`/`popitem` sweep) into a test body.
- Removal, renaming, or loss of `autouse=True` on `reset_image_cache` (line 10), or relocation of line 13/15 outside that fixture.
- Introduction of a second fixture that empties `_image_cache`, which would break the "inside the autouse fixture" locator.
- Any change to the fingerprinted bytes above.

## Checked At
2026-10-05

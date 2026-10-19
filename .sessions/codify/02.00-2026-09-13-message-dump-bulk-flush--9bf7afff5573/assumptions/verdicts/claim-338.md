VERDICT: UNCLEAR
PROMOTION: INELIGIBLE — verdict is UNCLEAR (P9: had to choose between two meanings of "clears `_image_cache`" to make the claim false)

## Claim
Nothing in `tests/test_image.py` clears `_image_cache` between statements inside one test function

## What it actually does
The file contains exactly two `image._image_cache.clear()` statements — lines 13 and 15 — and both are inside the autouse fixture `reset_image_cache`, which runs before and after each test, i.e. *between* test functions rather than between statements inside one. Every other `_image_cache` reference in a test body is read-only (`"session-1" in image._image_cache`, `len(image._image_cache["session-1"])`), and no test body contains `del`/`pop`/`reset` against it. The one statement that could break the claim is line 260, `image.clear_session_cache("session-1")`, which sits inside `test_clear_removes_session` between the assertion at line 258 and the assertion at line 261; line 261 asserts `"session-1" not in image._image_cache`, so within the test file's own expectations that call removes entries from `_image_cache` mid-function.

## Source
`/Users/alexhubitski/Projects/buster-ripper/tests/test_image.py` — lines 10–15 (fixture); 111–112, 149, 193, 214, 240–242, 258, 261 (read-only references in test bodies); 260 and 266 (`clear_session_cache` calls). Exhaustive over `_image_cache`/`clear`/`del`/`pop`/`reset` in the file.

The claim's subject IS the test suite, so the test file is the production surface for it (P10 exception).

## UNCLEAR — both readings, and which would settle it
- **Narrow reading — "clears `_image_cache`" = empties the dict (`_image_cache.clear()`):** the claim **HOLDS**. The only two such statements are fixture-level, outside every test function body.
- **Broad reading — "clears `_image_cache`" = removes entries from `_image_cache` during a test function:** the claim **DOES-NOT-HOLD**. Line 260 (`image.clear_session_cache("session-1")`) mutates `_image_cache` between statements inside `test_clear_removes_session`, and that test's own assertions (258 → 261) treat it as doing so.

I had to pick a definition, so per P9 the word is the finding, not the behaviour.

What would settle it, in order of directness:
1. The extractor naming which reading is meant (empty the whole dict vs. remove any entries).
2. If the broad reading is meant: `buster_ripper/image.py`'s `clear_session_cache` implementation — **this is outside the supplied bounded source (`Source root: tests/test_image.py`) and was not read.** To confirm the broad reading the caller must supply the `buster_ripper/image.py` source root; the test file alone cannot establish what that function mutates.

## Source Version
unavailable — no revision supplied; the consulted surface is the file bytes fingerprinted below.

## Source Fingerprint
sha256:b603bfd1624e30eab1c6d57390194a71cfd3716525abc33c742292bc5a3dfda0

## Invalidates On
- Any edit to `tests/test_image.py` that moves or adds an `image._image_cache.clear()` statement inside a test function body — settles the narrow reading as DOES-NOT-HOLD.
- Any edit to `tests/test_image.py` that removes the `clear_session_cache` call at line 260, or rewrites `test_clear_removes_session`'s 258–261 assertion pair — settles the broad reading as HOLDS.
- Supplying a source root for `buster_ripper/image.py` and reading `clear_session_cache` — settles the broad reading either way.
- Any change to the fingerprint above.

## Checked At
2026-10-05T21:20:55Z

VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
One test body in `tests/test_image.py` calls `image.clear_session_cache` between its own assertions

## What it actually does
Exactly one test body satisfies this. `image.clear_session_cache` has two call sites in the file, both inside test bodies, and they differ in precisely the way the claim describes:

- `test_clear_removes_session` (lines 248–261) — the call at line 260 sits **between two assertions belonging to that same test body**: line 258 `assert "session-1" in image._image_cache` precedes it and line 261 `assert "session-1" not in image._image_cache` follows it. This is the one.
- `test_clear_nonexistent_session` (lines 263–266) — the call at line 266 is the body's only statement (preceded by a `# Should not raise` comment) and that test contains **no assertions at all**, so the call there is not between anything.

Searched the full 286-line file for every occurrence of `clear_session_cache` and for every `assert` line to establish the boundaries; the two sites above are the complete set, and no other body places a cache-mutating call between its own assertions.

## Independence
Checked from the file's contents alone, without relying on the sibling claim in this bundle (P4). The two claims address different statements — `_image_cache.clear()` at file level versus `clear_session_cache(...)` at call level — and neither was used to establish the other.

## Source
`/Users/alexhubitski/Projects/buster-ripper/tests/test_image.py` — supplied `Source root`; read in full (lines 1–286), with per-line `assert` and call-site extraction.

Per the P10 exception, the claim's subject IS the test suite: it asserts the *arrangement* of statements within a test body — a call placed between assertions — not runtime behaviour of the production module. Statement ordering is directly readable from the source text, which is the primary surface here. Nothing was run (P1).

## Source Version
Unavailable as a revision identifier — the Git-metadata boundary in this role forbids resolving repository state. Content is instead pinned by the fingerprint below, which is the stability requirement for promotion.

## Source Fingerprint
sha256:b603bfd1624e30eab1c6d57390194a71cfd3716525abc33c742292bc5a3dfda0

## Invalidates On
- Any edit to `test_clear_nonexistent_session` that adds assertions around its line-266 call (which would make the count two, not one).
- Any edit that moves the line-260 call out from between lines 258 and 261, or removes either of those two assertions.
- Addition of a third `clear_session_cache` call site placed between assertions in some other test body.
- Any change to the fingerprinted bytes above.

## Checked At
2026-10-05

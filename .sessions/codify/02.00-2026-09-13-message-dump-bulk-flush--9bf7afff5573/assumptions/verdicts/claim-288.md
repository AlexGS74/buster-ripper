VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
Every positive placeholder assertion in `tests/test_image.py` pins a literal `img_` key

## What it actually does
Every assertion in the file that positively pins a placeholder/cache key uses a hard-coded
`img_` string literal — there is no computed, formatted, or variable-derived placeholder key
anywhere in the suite. The full membership of the class, from the whole 286-line file:

- L78 `assert "[image:img_0]" in result["messages"][0]["content"][0]["text"]` — the only
  assertion on placeholder *text*; the `img_0` token is inside the literal.
- L124+L126 `source = image.get_cached_image("session-1", "img_0")` then
  `assert source == {"url": "data:1"}` — positive, key pinned literally one statement earlier.
- L170+L172 same shape, `"img_0"`, positive `== {"url": "data:1"}`.
- L197 `assert image.get_cached_image("session-1", "img_1") is not None` and
  L198 `"img_2") is not None` — positive, keys pinned literally in the assert itself.

Excluded as negative (the claim says *positive*): L131 (`img_999` → `is None`),
L156 (`img_0` → `is None`), L196 (`img_0` → `is None`), and L286
(`assert "base64encodeddata" not in result`, a data-URL assertion, not a placeholder one).

An `rg` for `img_`, `f"` and `f'` across the file returns only the literals listed above — no
f-string or concatenation ever builds a placeholder key, so the positive set cannot drift off
the `img_` prefix without an edit to these lines.

## Source
/Users/alexhubitski/Projects/buster-ripper/tests/test_image.py (whole file read; the claim's
subject is the test suite, so the test file is the production surface for it — P10 exception)

## Source Version
tests/test_image.py, 286 lines, content identified by the fingerprint below; repository
revision not consulted (Git metadata is out of scope for this role)

## Source Fingerprint
sha256:b603bfd1624e30eab1c6d57390194a71cfd3716525abc33c742292bc5a3dfda0

## Invalidates On
- Any edit to L78, L124, L170, L197, or L198 that introduces a computed or formatted key
  (e.g. `f"img_{i}"`, a constant, or a value read from the returned body) into a positive
  placeholder assertion.
- A change to the placeholder key scheme in the image module's strip/cache path such that
  keys no longer begin with `img_` — new positive assertions would then pin a different
  literal prefix.
- Adding a new test to `tests/test_image.py` that asserts placeholder presence via a
  non-`img_` key.

## Checked At
2026-10-05T18:52:58Z

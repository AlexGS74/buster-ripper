VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
At least one case in `tests/test_image.py` passes a message containing two image blocks in a single `strip_and_cache_images` call

## What it actually does
`TestImageCaching::test_strips_multiple_images` builds a body whose single user message has three content blocks — `{"type": "image", ...}` at line 86, `{"type": "text", "text": "and"}` at line 87, and a second `{"type": "image", ...}` at line 88 — and passes that whole body to `image.strip_and_cache_images(body, "session-1")` once, at line 93. So one message with two image blocks does reach a single call. The claim's own assertions at lines 95–97 (content length still 3, middle block unchanged) corroborate that the two image blocks and the interleaved text block are all present in the one call's input.

Sweep of every `strip_and_cache_images` call site in the file (lines 75, 93, 108, 109, 123, 148, 169, 190, 211, 228, 237, 257): only line 93's message carries two image blocks. The calls at 190, 211 and 228 are inside `for i in range(3)` loops, so each iteration is one message with one image block — repeated calls, not a multi-image message. `test_has_images_with_image` (lines 36–47) does pair one text and one image block but calls `has_images`, not `strip_and_cache_images`.

## Source
`/Users/alexhubitski/Projects/buster-ripper/tests/test_image.py` — lines 80–97, specifically the two `{"type": "image"}` blocks at 86 and 88 and the single call at line 93.

The claim's subject IS the test suite, so the test file is the production surface for it (P10 exception).

## Source Version
unavailable — no revision supplied; the consulted surface is the file bytes fingerprinted below.

## Source Fingerprint
sha256:b603bfd1624e30eab1c6d57390194a71cfd3716525abc33c742292bc5a3dfda0

## Invalidates On
- `test_strips_multiple_images` (lines 80–97) being reduced to a single image block, split into two `strip_and_cache_images` calls, or moved so its two image blocks land in separate messages.
- Any rename of the blocks at lines 86/88 away from `"type": "image"` (e.g. to `"type": "image_url"`), which would remove the second image block under this claim's reading of "image blocks".
- Any change to the fingerprint above.

## Checked At
2026-10-05T21:20:55Z

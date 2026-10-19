VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
No `try` block in `handle_messages` covers its guarded image-strip call

## What it actually does
`handle_messages` spans lines 26–71 (the next definition starts at line 74). Inside
that span there are exactly two `try` statements: line 28 with
`except (json.JSONDecodeError, UnicodeDecodeError)` at line 30, whose only protected
statement is `body = await request.json()` at line 29; and line 58 with
`except Exception` at line 60, whose only protected statement is `dump_turn(body)` at
line 59. The guarded image-strip call — `image_handler.strip_and_cache_images(body, sid)`
at line 41, inside the `if config.STRIP_IMAGES and image_handler.has_images(body):`
block at line 40 — lies at function-level indentation between the two handlers and is
textually inside neither block. The function has no whole-body `try`, and the append at
line 42 is equally uncovered. The remaining `try` statements in the file (lines 129,
180, 230, 258) belong to `_stream_single_request`, `_relay_after_commit`,
`handle_non_streaming`, and `_non_stream_single_request` — not to `handle_messages`.
Consequence within this source: an exception raised by `strip_and_cache_images`
propagates out of `handle_messages` uncaught by this function (what the framework does
with it is outside the supplied source root).

## Source
`/Users/alexhubitski/Projects/buster-ripper/buster_ripper/routes/messages.py`
lines 26–71, specifically line 41 against the `try` blocks at lines 28–31 and 58–61;
`try`/`except`/`def` positions enumerated across the full file (293 lines), and
indentation of lines 36–62 confirmed byte-for-byte via `sed -n '36,62p' | cat -et`.

## Source Version
Repo source file read from the supplied source root; no package version string was
consulted, and git metadata/revision state is out of scope for this role. The
fingerprint below pins the consulted bytes.

## Invalidates On
- Any edit wrapping line 41 (or the line-40 block) in a `try`, extending an existing
  `try` outward to include it, or adding a whole-function `try`/`except` around
  `handle_messages`'s body.
- Any edit relocating `strip_and_cache_images` into the line-58 `try` block or into
  the line-28 `try` block.
- Any change to `buster_ripper/routes/messages.py` whose bytes differ from the
  fingerprint above.

## Checked At
2026-10-06

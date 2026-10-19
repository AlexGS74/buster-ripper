VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`handle_messages` calls `dump_turn` only when `config.DUMP_DIR` is not None

## What it actually does
In `handle_messages` there is exactly one call to `dump_turn` — `dump_turn(body)` at
line 59 — and it sits directly under the guard `if config.DUMP_DIR is not None:`
(line 57), at that block's indentation. Nothing else in the function reaches
`dump_turn`: across the whole source root the name occurs only at the import
(line 12, `from ..dump import dump_turn`) and at line 59. The guard's inner
`try/except Exception` (lines 58–61) only handles failure of the call; it does not
widen when the call happens, and there is no `else` branch or restructured form
that would call `dump_turn` on the `DUMP_DIR is None` path.

## Source
`/Users/alexhubitski/Projects/buster-ripper/buster_ripper/routes/messages.py`
lines 26–71 (`handle_messages` body), specifically lines 57–61; name occurrences
verified over the full file (293 lines).

## Source Version
Repo source file read from the supplied source root; no package version string was
consulted, and git metadata/revision state is out of scope for this role. The
fingerprint below pins the consulted bytes.

## Source Fingerprint
sha256:560a81be69d42552204400a43e2fc5ddec9a983a0b67f165fbe8d7ffee713709

## Invalidates On
- Any edit that moves `dump_turn(body)` out from under `if config.DUMP_DIR is not None:`,
  adds a second `dump_turn` call site in `handle_messages`, or changes the guard
  predicate (e.g. to truthiness `if config.DUMP_DIR:` or a different config field).
- Any change to `buster_ripper/routes/messages.py` whose bytes differ from the
  fingerprint above, or to the `dump_turn` definition in `..dump` if the claim is
  later read as covering indirect calls (this check reads the direct call site in
  `handle_messages`; callees `normalize`, `session_id`, `image_handler` live outside
  the supplied source root and were not opened).

## Checked At
2026-10-06

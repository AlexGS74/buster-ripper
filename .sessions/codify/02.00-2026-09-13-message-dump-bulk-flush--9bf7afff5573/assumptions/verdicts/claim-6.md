VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`dump.py` writes each turn ≥ 1's record as the `difflib.unified_diff` of `_render_body(body)` against the previous turn's rendering to `turn_{turn:03d}.diff`

## What it actually does
For every turn ≥ 1, `dump_turn` renders the current body with `_render_body(body)` (line 77), splits both it and `state.prev_text` (the previous turn's `_render_body` result, assigned at line 108 after each turn) into lines, computes `difflib.unified_diff(prev_lines, curr_lines, fromfile=f"turn_{turn - 1:03d}", tofile=f"turn_{turn:03d}", lineterm="")`, and writes the joined diff lines to `{session_dir}/turn_{turn:03d}.diff` via `write_text`.

## Source
`buster_ripper/dump.py`, `dump_turn` — rendering at line 77, `difflib.unified_diff` at lines 89–93, output path and write at lines 94–95, `state.prev_text = current_text` at line 108

## Source Version
unavailable (git metadata excluded per protocol boundary)

## Source Fingerprint
sha256:243b21434c14e6ea8144329fe1d0ac4e0c499ed9a7e6e09bdcac5e2417a56b1d

## Invalidates On
- any change to the `difflib.unified_diff` call, its input renderings, or the `turn_{turn:03d}.diff` filename in `buster_ripper/dump.py`
- `state.prev_text` updated from something other than the previous turn's `_render_body` output (e.g., raw JSON or a compact form)

## Checked At
2026-09-19

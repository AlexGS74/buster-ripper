VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`dump.py` writes turn 0's record as `json.dumps(body, indent=2, ensure_ascii=False)` to `turn_000.json`

## What it actually does
On turn 0 (when `state.turn == 0`), `dump_turn` writes `json.dumps(body, indent=2, ensure_ascii=False)` — verbatim call, same arguments — to `{session_dir}/turn_000.json` via `out.write_text(..., encoding="utf-8")`. The raw request body is serialized as pretty-printed JSON; no rendering or diffing is applied to the stored bytes.

## Source
`buster_ripper/dump.py`, `dump_turn`, lines 80–82: `out = state.session_dir / "turn_000.json"` followed by `out.write_text(json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8")`

## Source Version
unavailable (git metadata excluded per protocol boundary)

## Source Fingerprint
sha256:243b21434c14e6ea8144329fe1d0ac4e0c499ed9a7e6e09bdcac5e2417a56b1d

## Invalidates On
- any change to the `json.dumps` call arguments (indent, ensure_ascii) or to the `turn_000.json` filename in `buster_ripper/dump.py`
- turn 0 record switched to the `_render_body` / unified-diff path used for later turns

## Checked At
2026-09-19

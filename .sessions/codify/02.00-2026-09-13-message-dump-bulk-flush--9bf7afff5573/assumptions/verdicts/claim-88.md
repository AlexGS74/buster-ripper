VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`_write_full_turn` writes turn 0's body as indented, non-ASCII-preserving JSON to `turn_000.json`

## What it actually does
`_write_full_turn` targets `state.session_dir / "turn_000.json"` and writes `json.dumps(body, indent=2, ensure_ascii=False)` as UTF-8. Its caller `_write_documented` invokes it when `state.turn == 0`.

## Source
`buster_ripper/dump.py:150-165`

## Source Version
unavailable

## Source Fingerprint
sha256:c8f093c89b61c355ac182feb070cd90b84924ca0d187332117d76b5c766ad9b9

## Invalidates On
Change to `buster_ripper/dump.py`'s `_write_documented` turn-0 dispatch or `_write_full_turn` serialization and target path.

## Checked At
2026-09-27

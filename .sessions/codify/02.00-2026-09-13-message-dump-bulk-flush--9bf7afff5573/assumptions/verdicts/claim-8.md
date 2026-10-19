VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
dump.py advances `state.prev_text` and `state.turn` after processing each turn

## What it actually does
After the selected writer or record-appender returns, `dump_turn` assigns the rendered current body to `state.prev_text` and increments `state.turn`. If processing raises, those lines are not reached.

## Source
buster_ripper/dump.py:83-98

## Source Version
unavailable

## Source Fingerprint
sha256:c8f093c89b61c355ac182feb070cd90b84924ca0d187332117d76b5c766ad9b9

## Invalidates On
Changes to `dump_turn`'s state update or processing order in `buster_ripper/dump.py`.

## Checked At
2026-09-27

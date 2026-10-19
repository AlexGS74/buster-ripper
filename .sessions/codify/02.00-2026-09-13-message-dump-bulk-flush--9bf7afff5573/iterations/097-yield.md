# Yield 097 — `T021 · harden the MS-001 documented-format gate`

## Orchestrator Prompt (verbatim)

> You are a subagent spawned by another session.
> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
> Task: T021

## Result

`ACCEPT` — the increment is test-only, its strengthened gate passes on the reverted bytes, reds on both
production drifts the done-when names (plus a third that isolates the complete-body arm), and touched zero
production, config or registry bytes. Two defer-tier findings recorded.

## Reviewed scope

- Increment: Checkpoint 217 `92d3d4d` plus its docs-only follow-ups `aa95d4b`, `6b0c60f`, `05b993a`
  (the last three change only `iterations/096-forward-qa.md`).
- Declared files (`096` §Files Changed): `tests/qa/test_qap_ms001.py`, `tests/qa/harness.py`,
  `tests/qa/test_qap_ms002.py`, `qa-plan.md`, plus the session artifacts.
- `git diff --name-only 92d3d4d^ 05b993a`: zero `buster_ripper/`, `invariants/`, `decisions.md`,
  `pyproject.toml` or `contracts/` bytes. No `Outside plan` file.

## Required checks

| Check | Result | Evidence |
|---|---|---|
| Affected tests (the three QA gates that import the changed `tests/qa/harness.py`) | PASS — `QAP-MS001` **12 passed in 1.98s**, `QAP-MS002` **15 passed in 13.85s**, `QAP-MS004` **6 passed in 44.29s** | `runs/097-qap-ms001.txt`, `runs/097-qap-ms002.txt`, `runs/097-qap-ms004.txt`; tree clean on `buster_ripper/` and `tests/` at run time |
| Hermetic suite (the report's claim re-measured) | PASS — **511 passed @ 96.71%** | `runs/097-hermetic.txt`; `tests/qa` is in `norecursedirs`, so the harness edit cannot reach this suite |
| Type check | N/A — no type checker on this tree (`uv run mypy` → `Failed to spawn`; no `[tool.mypy]` in `pyproject.toml`) | — |
| LNCP / FL-LNCP scan | Recorded, defer-tier — `test_qap_ms001.py` **775** lines (Red band; 617 pre-edit, +158), `assert_documented_contents` L:34→**49** C:10→**12** P:2→**4**, `test_tc108…` L:**78** N:**3** C:6; `harness.py` 350 (review band, 312 pre-edit) | `codify scan path` on the current and `92d3d4d^` bytes |
| Capacity ratchet | N/A — no additive production surface; the file-size growth is the defer-tier row below | — |
| Scope / test cost / adversarial surface | PASS — every changed file is declared in `096` §Files Changed; the gates reuse `tests/qa/harness.py` and `record_section` is promoted rather than duplicated; no production parser, writer or payload changed | — |
| Backlog pickup | N/A — the `forward-qa` contract defines no `Backlog pickup:` line; the two open `tests/qa/harness.py` rows were already resolved at `077` | `quality-backlog.md:42`, `:77` |

## Mutation replays (the done-when's arms)

Every run on `HEAD 05b993a` with a clean production tree; each mutation reverted byte-exact
(`git status --short -- buster_ripper/ tests/` empty and `git diff --stat -- buster_ripper/` empty afterwards).

| Arm | Mutation | Result |
|---|---|---|
| (a) unmutated | none | **12 passed** — the strengthened gate is green |
| (b) route hands `dump_turn` the pre-normalization body | `routes/messages.py`: keep `sid` on the normalized body, call `dump_turn(raw_body)` | **2 failed, 10 passed** — `TC-107` and `TC-108` red at `test_qap_ms001.py:376` (`today's layout is N + 1 files: []`): `dump_turn` re-derives `sid` from the body it is handed, so the artifacts land under the nonce-bearing system's directory. The produced artifacts still keep every filename, the `N + 1` count, the index shapes and `2N` |
| (b′) writer serializes the pre-normalization system, `sid` stable | `dump.py:_write_full_turn` re-attaches the billing header to the written body | **2 failed, 10 passed** — `TC-107`/`TC-108` red at `test_qap_ms001.py:166` (`turn_000.json is not the normalized request body: missing=[] unexpected=[] differing=['system']`): the complete-body arm alone catches the drift, with layout, count, index shapes and `2N` unchanged |
| (c) diff source is `json.dumps(body)` | `dump.py`: `SessionState.prev_body` added, `_write_diff_turn` diffs consecutive raw JSON bodies | **1 failed, 11 passed** — `TC-108` red at `test_qap_ms001.py:208` (`turn_001.diff is not computed over the turn's rendering: observed=['-      "content": "documented-t0-…"', '+      "content": "documented-t1-…"']`); `TC-107` has no diff arm; `2N`, `N + 1` and index shapes unchanged |
| (d) scope | `git diff --name-only 92d3d4d^ 05b993a` | only `tests/qa/` plus session artifacts — zero production, config, plan, contract or registry bytes |

The done-when's parenthetical allows arm (b) to move `sid`; replay (b′) is added because it keeps `sid`
stable and therefore isolates the complete-body arm the strengthened gate added.

## Findings

### [P3] The sensitivity witness's body-drift arm cannot fail
- **Priority**: P3 — the gate itself is proven falsifiable by the replays above; only this witness line is inert
- **Location**: `tests/qa/test_qap_ms001.py:460-469`
- **What is wrong**: the arm ends with `assert drifted != clean` and two substring checks over two
  locally built dicts. No change to `assert_documented_contents` — deleting its `expected_body` comparison
  entirely — turns those lines red, so they do not witness what the docstring claims ("the body comparison must
  reject a nonce-bearing body"). The diff arm above them does redden on a helper change, so the test as a whole
  can fail. `096`'s statement that the witness "proves the comparisons discriminate the drift shapes" holds
  for the diff comparison and is overstated for the body comparison.
- **Evidence**: replay (b′) reds `assert_documented_contents`'s body comparison at `test_qap_ms001.py:166`;
  the sensitivity arm still passes with that comparison removed.
- **Disposition**: defer-tier → `QB:qa-gate/body-drift-witness-inert`. Not reject-tier: no product behaviour,
  no false pass (the gate itself reds in replays (b), (b′), (c)), and the fix is a test-shape change owned by
  the next `tests/qa/` touch.

### [P3] `test_qap_ms001.py` crossed the FL-LNCP red band and its shared helper worsened
- **Priority**: P3 — test-file debt; no behaviour at risk
- **Location**: `tests/qa/test_qap_ms001.py:1`; `assert_documented_contents:143-211`; `test_tc108…:338-426`
- **What is wrong**: the file went **617 → 775** raw lines, crossing the 700 red line that `T021`'s planned
  capacity answer said it would not cross ("additions keep it in the same band and cross none" — it was in the
  501–700 band and is now above 700). `assert_documented_contents` moved L:34→49, C:10→12 (entering a hard
  C violation) and P:2→4; the TC-108 witness is L:78 N:3. The increment is test-only, so the capacity
  ratchet is N/A and this is the defer-tier file-size class.
- **Evidence**: `codify scan path tests/qa/test_qap_ms001.py --detailed`; the same scan on the
  `92d3d4d^` bytes; `wc -l` at both ends.
- **Disposition**: defer-tier → `QB:test-qap-ms001/file-size`. Remedy at the next touch that is not a
  gate repair: split the drift-observation helpers and their witnesses into a focused module, leaving the
  scenario-driven gates here — the split the planned context already admitted as "the implementation's choice".

## Manual verification

- **Type**: CLI / E2E QA gates
- **What was tested**: the three QA gates that import the changed harness, plus the hermetic suite, each
  once on the reverted bytes; the two named production drifts were replayed by mutation and reverted.
- **Result**: Pass — 12/15/6 gates green, hermetic 511 @ 96.71%, both drifts red.
- **Evidence**: `runs/097-*.txt`; the replay rows above.

## Gap resolved

None — the plan and the code agree on the contract; no reading was taken on the user's behalf.

## Open

- **Assumption extraction for the new gate surface is still owed** — owner: the dispatcher's
  `forward-collect` route, on these bytes, before the closing Yield Red check (`096` §Open).
- **`resolved` for `T015`–`T021`/`MS-001`, phase acceptance, the `Delivered` entry and parent
  completion** — owner: the closing Yield Red check.
- `095`'s artifacts and `runs/` remain uncommitted — owner: the dispatcher.

## Next

Closing Yield Red check on the strengthened bytes; then `phase-boundary-finalize`.

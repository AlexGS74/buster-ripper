# Yield 109 — `T021.003` local Yield (`S21` CE-2 residual gate repair)

## Orchestrator Prompt (verbatim)

> Execute the yield-interactive skill for this session. Arguments: $ARGUMENTS
>
> You are a subagent spawned by another session.
> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
> Task: T021.003

## Result

`ACCEPT` — `T021.003` · raw artifact reads: the documented-format gate now derives its expected diff
over the same `\r\n`/`\r` line identity production diffed before the write, and the focused artifact
round-trip regression is red on the pre-repair read with the exact CE-2 numbers and green after.
`T021.003` is `accepted-local`, not `resolved`.

## Reviewed scope

Commit `eea5086` (checkpoint 251), full increment `git diff HEAD~1 HEAD`:
`tests/qa/harness.py`, `tests/qa/test_qap_ms001.py`, plus the forward-qa pass's own status record
(`qa-plan.md`, `plan-summary.md`, `handoff-context.md`, `handoff-archive.md`, `plan-completed.md`,
`chronicle.md`, `iterations/108-forward-qa.md`). Zero `buster_ripper/`, `pyproject.toml`, `invariants/`
or `decisions.md` bytes. Pass 1 read the diff and the gate source before Pass 1.5.

## Required checks

| Check | Result | Evidence |
|---|---|---|
| Affected scope, run once at `HEAD eea5086` | **PASS** — `QAP-MS001` **17 passed** (10.70 s), `QAP-MS002` **15 passed** (29.42 s), `QAP-MS004` **6 passed** (44.36 s) | `.venv/bin/python -m pytest tests/qa/test_qap_ms00{1,2,4}.py -o addopts= -q` |
| Hermetic suite | **PASS** — **513 passed @ 96.71%** (9.55 s) | `.venv/bin/python -m pytest` |
| Type check | N/A — no type checker on this tree | — |
| Plan fulfillment | **PASS** — every done-when clause re-derived below | — |
| Capacity ratchet | N/A — no additive production surface | — |
| Scope | **PASS** — every changed file is declared: test bytes plus the forward-qa pass's own status record | `git diff --name-only HEAD~1 HEAD` |
| Test cost | **PASS** — one focused test, `tmp_path`, no new fixture/helper duplicate; `_synthetic_self_contained` is the writer's published record shape, not a second harness | `tests/qa/test_qap_ms001.py:628-637` |
| Adversarial surface | **PASS** — no production parser, writer, payload, identity key or boundary set changed; R2's writer guard is untouched | — |

## Done-when re-derivation

| Clause | Result |
|---|---|
| Focused regression round-trips a CRLF rendering through `"\n\n".join(records) + "\n"`, reads with the gate's reader, extracts with `rendering_section`, asserts the raw renderings and `changed_lines == diff_changed_lines` | **PASS** — `test_sensitivity_s21_ce2_artifact_read_preserves_line_endings`, `tests/qa/test_qap_ms001.py:688-738` |
| Red on the pre-repair read with `expected=[]` / `observed=['-line1\n', '+line1\n']` | **PASS** — `read_text` forced to universal mode; red at `:719` with exactly `([], ['-line1\n', '+line1\n'])`; harness restored sha256 `58f70198…` |
| The dropped-line-ending writer arm rejected | **PASS** — `:732-738` (`diff_changed_lines(mutated) != expected`) |
| CE-1/CE-2/CE-3 in-memory regressions green | **PASS** — `QAP-MS001` 17/17 |
| The two `T021` mutation replays still red | **PASS** — (b) route hands `dump_turn` the pre-normalization body: **2 failed** (`TC-107`, `TC-108`); (c) diff over `json.dumps(body)`: **1 failed** (`TC-108`); both reverted byte-exact (`dump.py` `536ac105…`, `routes/messages.py` `64be411b…`) |
| Unmutated gates | **PASS** — 17/17, 15/15, 6/6; hermetic 513 @ 96.71% |
| Zero production/config/plan-contract/registry bytes | **PASS** — `buster_ripper/` clean; `qa-plan.md` gained a re-entry note only, no scenario semantics |

## Parallel reconciliation

`PARALLEL-RECONCILED: assumptions run 1 (claims ledger) + 107-assumption-decide | findings 1 (`S21` CE-2 residual) | repaired 1 (`T021.003`) | covered: `S21` recomposition and the `DC-S21-r2` ledger row | blocked: none.`

- `S21` · unified-diff seam: current composition is `COUNTEREXAMPLE`; the CE-2 residual is the only
  load-bearing finding and this pass repairs it. `S21`'s single post-repair recomposition allowance is
  consumed; a further non-`SAFE` composition is capped by rule and its remainder goes to the user in one message.
- `S22`/`S23` remain `SAFE`; rows 426–431/438–440 remain `HOLD`/`absorbed`; no verdict or consequence
  is unresolved. `FK-011` is the only open Fork and this task does not touch it.

## Findings

None reject-tier.

### [P3] The focused regression pins the reader, not the gate's two call sites
- **Priority**: P3 — should fix only when the gate is next touched
- **Location**: `tests/qa/test_qap_ms001.py:688-738`; call sites `:276`, `:465`
- **What is wrong**: the regression drives `read_text`'s raw mode directly, so a revert of either
  `assert_documented_contents`/`TC-108` call site back to the universal default would leave it green and
  restore the CE-2 false-pass surface. The plan scoped the proof to the reader plus the artifact round trip
  and `107` recorded a CRLF `TC-108` body as additive, not a precondition, so this is a coverage limit, not
  an unmet clause.
- **Evidence**: the mutation replay (`read_text` forced universal) reddens the regression; a call-site revert
  does not.
- **Reach**: `latent` — no `\r` byte exists under `tests/qa/`, so today the gate cannot meet the input.

## Defer-tier observations

- `tests/qa/test_qap_ms001.py` grew **963 → 1027** lines (+64) and the new
  `test_sensitivity_s21_ce2_artifact_read_preserves_line_endings` is **L:31 N:0 C:1 P:1** (hard
  length zone). Test bytes only; still defer-tier. `QB:test-qap-ms001/file-size` extended.
- `tests/qa/harness.py` grew **358 → 365** lines (+7) in the yellow review band. Test-support only;
  `QB:qa-harness/file-size-yellow` recorded.
- `QB:qa-gate/ce2-call-site-unpinned` recorded for the finding above.

## Outside plan

None — every changed file is either the task's declared test surface (`tests/qa/harness.py`,
`tests/qa/test_qap_ms001.py`) or the forward-qa pass's own status record.

## Noted, not pursued

- `iterations/108-forward-qa.md` appended a `chronicle.md` entry before local acceptance. The chronicle's
  acceptance-only trigger is a process rule; the entry is accurate and this pass appends its own acceptance
  entry. No repair.
- `_synthetic_self_contained` writes `FULL` summaries on later turns where the writer uses `+A/-R`; only the
  record separators and renderings are consumed by the regression, so the summary text is inert here.
- The 59 untracked `assumptions/**` files from `107-assumption-decide` are that run's output; left untouched.

## Open

- Capped post-fix `S21` recomposition — owner: assumption gate.
- Closing Yield Red check and `phase-boundary-finalize` — owner: dispatcher.

## Next

`T021.003` is `accepted-local`; run the capped `S21` recomposition, then the closing Yield Red check.

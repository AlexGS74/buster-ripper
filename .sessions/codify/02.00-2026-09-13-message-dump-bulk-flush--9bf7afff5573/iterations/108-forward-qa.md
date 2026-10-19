# Forward QA 108 — `QAP-MS001` gate repair `T021.003` (`S21` CE-2 residual)

## Orchestrator Prompt (verbatim)

> You are a subagent spawned by another session.
> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
> QA Pass: QAP-MS001

## Result

`READY-PASSING` — `T021.003` is authored in test bytes only: the documented-format gate now
reads both diff-comparison surfaces in raw text mode, and the focused artifact round-trip
regression is red on the pre-repair universal read (`expected=[]`,
`observed=['-line1\n', '+line1\n']`) and green after. `QAP-MS001` observes its four contracts at
17/17 on `HEAD 65c298c`.

## QA Pass

| Field | Value |
|---|---|
| QA pass | `QAP-MS001` |
| Validation points | `VP-T001-writes-deferred`, `VP-T001-dump-failure-isolated`, `VP-T002-documented-format-preserved`, `VP-T002-default-encoding` |
| Scenarios | `TC-100`, `TC-001` (both write paths), `TC-107`, `TC-108` |
| Gate | `MS-001` (`T001`/`T002`/`T005` accepted-local; `T021.003` nested under the accepted-local `T021`, local Yield pending) |
| Plan authority | [`qa-plan.md`](../qa-plan.md) §QA Gate Matrix / §Scenario Contracts; [`handoff-context.md`](../handoff-context.md) §Planned Task Context → `T021.003`; `DC-S21-r2` |

## Contract Draft

Scenario semantics are unchanged from [070](070-forward-qa.md), [096](096-forward-qa.md) and
[099](099-forward-qa.md); this pass repairs the gate's reader for the same contract. The draft is
restated so the repair cannot widen it.

| Scenario | Real trigger | Observable assertions | Forbidden assertions | Lifecycle proof | Telemetry proof | Harness entry |
|---|---|---|---|---|---|---|
| TC-100 | collapsed format + N turns, no flush parameter (`022`) | no artifact before a flush; one `<format>.log` after shutdown | per-turn file / `index.txt` / one artifact per pending turn | process healthy across N turns | none declared | `test_tc100_collapsed_format_defers_every_write_until_a_flush` |
| TC-001 | documented write path and collapsed flush path raise | upstream status/body unchanged; exception logged | a dump-caused 5xx or a silent log | process keeps serving | none declared | `test_tc001_documented_write_failure_still_returns_the_upstream_response`, `..._collapsed_flush_failure_...` |
| TC-107 | no `--dump-format` | documented layout at `2N` writes, no `.log` | a collapsed artifact under the default | unchanged from today | none declared | `test_tc107_no_format_configured_produces_the_documented_layout` |
| TC-108 | documented format at `N=3`, renderings from a `self-contained` artifact | three files, README content/line format, `2N` writes | drift in the body, diff source, index columns, or write count | unchanged from today | none declared | `test_tc108_documented_format_preserves_contents_at_2n_writes`; `test_sensitivity_s21_ce2_artifact_read_preserves_line_endings` |

## Repair

- **`T021.003` · artifact-read line identity (`S21` CE-2 residual).** `read_text` gained an
  optional `newline` parameter forwarded to `open`; the default stays universal-newline mode, so the
  JSON/index/nonce reads in `QAP-MS002`/`QAP-MS004` are unchanged (`DC-S21-r2` rejected widening
  the shared default). `assert_documented_contents` reads the observed `turn_NNN.diff` and `TC-108`
  reads the `self-contained` artifact with `newline=""`, so the expected diff is computed over the same
  `\r\n`/`\r` bytes production diffed before the write.
- **Focused regression** `test_sensitivity_s21_ce2_artifact_read_preserves_line_endings` round-trips the
  CE-2 renderings through the writer's `"\n\n".join(records) + "\n"` artifact shape, extracts them with
  `rendering_section`, asserts the recovered renderings are the raw ones, and asserts
  `changed_lines == diff_changed_lines == ['-line1\r\n', '+line1\n']`. It also reproduces the pre-repair
  shape (`expected=[]` / `observed=['-line1\n', '+line1\n']`) through the default read mode, and rejects
  a writer that drops the line-ending change.
- **Mutation replay (local, reverted byte-exact).** With `read_text` mutated to ignore `newline` (the
  pre-repair universal read), the regression fails at the exact CE-2 numbers: `AssertionError:
  ([], ['-line1\n', '+line1\n'])`. Restored harness sha256
  `58f7019871ea308544146dcce3e3bb7a35cbf2b2db37ae1ad68a7f1d826f0461`; the test passes again.
- **No E2E CRLF body added.** `107`'s disposition says a CRLF `TC-108` body is additive coverage, not
  a precondition of the repair; the oracle defect is proven by the artifact round trip, and every gate body
  stays LF, so the scenario set is unchanged.

## Boundary Inspection Exceptions

None. No production source was read. The artifact shape is `README.md` §Dump mode's published
`self-contained` record contract (already mirrored by `_synthetic_self_contained`), and the writer's
`"\n\n".join(records) + "\n"` shape is the same public contract. The only inspected surface beyond the
gate was the session's `S21` composition record, which is a plan artifact, not implementation internals.

## Files Changed

- `tests/qa/harness.py` — `read_text` gained the optional `newline` parameter (default unchanged)
  and its docstring states the CE-2 reason for raw mode.
- `tests/qa/test_qap_ms001.py` — raw reads at `assert_documented_contents`'s observed diff and at
  `TC-108`'s `self-contained` artifact; one focused regression; module docstring re-entry note.
- `qa-plan.md`, `plan-summary.md`, `handoff-context.md` — status and re-entry record only.
- Zero production, config, `invariants/`, `decisions.md`, plan-contract or registry bytes.

## Harness Execution

Commands run at `HEAD 65c298c`, clean tracked tree (59 untracked assumption files from
`107-assumption-decide` left untouched):

| Command | Result |
|---|---|
| `.venv/bin/python -m pytest tests/qa/test_qap_ms001.py -o addopts= -q` | **17 passed** in 12.11s (`runs/108-qap-ms001.txt`) |
| `.venv/bin/python -m pytest tests/qa/test_qap_ms002.py -o addopts= -q` | **15 passed** in 46.02s (`runs/108-qap-ms002.txt`) |
| `.venv/bin/python -m pytest tests/qa/test_qap_ms004.py -o addopts= -q` | **6 passed** in 44.24s (`runs/108-qap-ms004.txt`) |
| `.venv/bin/python -m pytest` (hermetic default) | **513 passed @ 96.71%** (`runs/108-hermetic.txt`) |
| `grep -n "dump-flush" tests/qa/test_qap_ms001.py` | docstring only; no `extra_args` passes a flush flag (`022` honored) |
| mutation replay: `read_text` forced to universal mode, focused regression | **red** with `expected=[]` / `observed=['-line1\n', '+line1\n']`; restored byte-exact |

`ready` is not a verdict: the local Yield and the `MS-001` boundary records re-run and write the
authoritative result.

## Open

- `T021.003` local Yield — owner: dispatcher's `yield` route.
- Capped post-fix `S21` recomposition, then the closing Yield Red check — owner: assumption gate / Yield Red.

## Next

Local Yield for `T021.003`, then the capped `S21` recomposition, then Yield Red.

# Forward QA 099 — `QAP-MS001` run-49 gate repair (`T021.001`/`T021.002`)

## Orchestrator Prompt (verbatim)

> You are a subagent spawned by another session.
> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
> QA Pass: QAP-MS001

## Result

`READY-PASSING` — the two run-49 gate repairs are authored in test-side bytes only and the
`QAP-MS001` gate observes its four contracts at 16/16 on `HEAD f3468c7`.

## QA Pass

| Field | Value |
|---|---|
| QA pass | `QAP-MS001` |
| Validation points | `VP-T001-writes-deferred`, `VP-T001-dump-failure-isolated`, `VP-T002-documented-format-preserved`, `VP-T002-default-encoding` |
| Scenarios | `TC-100`, `TC-001` (both write paths), `TC-107`, `TC-108` |
| Gate | `MS-001` (`T001`/`T002`/`T005` accepted-local); repairs anchored to `T021`, local Yield pending |
| Plan authority | [`qa-plan.md`](../qa-plan.md) §QA Gate Matrix / §Scenario Contracts; [`handoff-context.md`](../handoff-context.md) §Planned Task Context → `T021.001`/`T021.002` |

## Contract Draft

Scenario semantics are unchanged from [070](070-forward-qa.md) and [096](096-forward-qa.md); this pass
repairs the gate's implementation of the same contract. The draft is restated so the repair cannot widen it.

| Scenario | Real trigger | Observable assertions | Forbidden assertions | Lifecycle proof | Telemetry proof | Harness entry |
|---|---|---|---|---|---|---|
| TC-100 | collapsed format + N turns, no flush parameter (`022`) | no artifact before a flush; one `<format>.log` after shutdown | per-turn file / `index.txt` / one artifact per pending turn | process healthy across N turns | none declared | `test_tc100_collapsed_format_defers_every_write_until_a_flush` |
| TC-001 | documented write path and collapsed flush path raise | upstream status/body unchanged; exception logged | a dump-caused 5xx or a silent log | process keeps serving | none declared | `test_tc001_documented_write_failure_still_returns_the_upstream_response`, `..._collapsed_flush_failure_...` |
| TC-107 | no `--dump-format` | documented layout at `2N` writes, no `.log` | a collapsed artifact under the default | unchanged from today | none declared | `test_tc107_no_format_configured_produces_the_documented_layout` |
| TC-108 | documented format at `N=3` | three files, README content/line format, `2N` writes | drift in the body, diff source, index columns, or write count | unchanged from today | none declared | `test_tc108_documented_format_preserves_contents_at_2n_writes` |

## Findings Repaired

- **`T021.001` · diff-line identity (`S21` CE-1/CE-2/CE-3).** `changed_lines` now mirrors the
  published writer's shape — `splitlines(keepends=True)`, `lineterm=""`, `"".join` — and both sides are
  read by one structural reader, `diff_changed_lines`, which consumes the diff's own leading
  file-header pair and each glued `@@` control line and recovers the content line that follows;
  a mid-diff removed/added line whose text begins with `--`/`++` stays content, so the reader
  cannot drop the rest of the diff at a false header. `rendering_section` replaces
  `record_section(...).strip()`
  and removes only the artifact's record-separator newlines. Evidence: the three new focused regressions are
  green on the repaired gate and red on the pre-repair helpers (CE-1 8-vs-7, CE-2 `[]`-vs-`-line1/+line1`,
  CE-3 recovered-renderings mismatch), and the two `T021` mutation replays still differ.
- **`T021.002` · ambient-proxy-free client hop (`S22`).** `Proxy.post` builds
  `urllib.request.build_opener(urllib.request.ProxyHandler({}))`, matching `require_upstream`, instead of
  `urlopen`. Evidence: `test_sensitivity_s22_proxy_post_ignores_ambient_proxy_settings` passes on the repaired
  harness and fails with `URLError: [Errno 61] Connection refused` on the pre-repair harness (stashed and re-run).
  The live upstream was reachable (`:30000` preflight HTTP 200) and the dump artifact proves the local proxy answered.

## Boundary Inspection Exceptions

### Boundary Inspection Exception
- **Need**: place the artifact's record-separator newlines without duplicating the shared reader's regex.
- **Path/symbol**: `buster_ripper/dump.py:295-305` (`_flush_state`, the `"\n\n"` join and closing `"\n"`).
- **Question**: how many newlines does the artifact append after a rendering, and does the last record differ?
- **Assertion impact**: none; the approved contract is unchanged. Answer: two between records, one at EOF;
  `rendering_section` removes exactly those and keeps the rendering's own trailing `"\n"`.

## Files Changed

- `tests/qa/test_qap_ms001.py` — `production_diff_text`, repaired `changed_lines`/`diff_changed_lines`
  (`_content_lines`), `rendering_section`, the renderings extraction, three `S21` regressions, one `S22`
  regression, and the existing sensitivity expectations updated to the writer's line identity.
- `tests/qa/harness.py` — `Proxy.post` proxy-free opener plus its docstring.
- `qa-plan.md`, `plan-summary.md`, `handoff-context.md` — status only.
- Zero production, config, `invariants/`, `decisions.md`, or approved-contract bytes.

## Harness Execution

Commands run at `HEAD f3468c7`, clean tree apart from these two test-side files:

| Command | Result |
|---|---|
| `.venv/bin/python -m pytest tests/qa/test_qap_ms001.py -o addopts= -q` | **16 passed** (`runs/099-qap-ms001.txt`) |
| `.venv/bin/python -m pytest tests/qa/test_qap_ms002.py -o addopts= -q` | **15 passed** (`runs/099-qap-ms002.txt`) |
| `.venv/bin/python -m pytest tests/qa/test_qap_ms004.py -o addopts= -q` | **6 passed** (`runs/099-qap-ms004.txt`) |
| `.venv/bin/python -m pytest` (hermetic default) | **511 passed @ 96.71%** (`runs/099-hermetic.txt`) |
| `grep -n "dump-flush" tests/qa/test_qap_ms001.py` | docstring only; no `extra_args` passes a flush flag (`022` honored) |

`ready` is not a verdict: the `MS-001` Yield Phase re-runs and writes the authoritative `pass`/`fail`.

## Result

`READY-PASSING`

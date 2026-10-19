# Forward QA 105 — `QAP-MS001` re-proof after the gate-model alias adoption

## Orchestrator Prompt (verbatim)

> You are a subagent spawned by another session.
> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
> QA Pass: QAP-MS001

## Result

`READY-PASSING` — the four `MS-001` scenario gates observe their approved contracts at 16/16 on the current bytes, and the five uncommitted QA/harness edits they run through are model-alias substitutions that weaken no assertion.

## QA Pass

| Field | Value |
|---|---|
| QA pass | `QAP-MS001` |
| Validation points | `VP-T001-writes-deferred`, `VP-T001-dump-failure-isolated`, `VP-T002-documented-format-preserved`, `VP-T002-default-encoding` |
| Scenarios | `TC-100`, `TC-001` (documented and collapsed), `TC-107`, `TC-108` (+ its three process-free sensitivity witnesses) |
| Gate | `MS-001` — `T001`/`T002`/`T005` accepted-local; boundary `SEMANTIC-APPROVE` at [093-yield-phase](093-yield-phase.md) |
| Plan authority | [`qa-plan.md`](../qa-plan.md) §QA Gate Matrix / §Scenario Contracts / §QA Author Passes |

Milestone completeness: every `MS-001` constituent task is accepted by local Yield — `T001`/`T002` at [031-01](031-01-yield.md)/[031-02](031-02-yield.md), `T005` at [049-01](049-01-yield.md). The pass is scheduled after those acceptances and before the milestone verdict; the milestone's own Yield Phase has already ruled `SEMANTIC-APPROVE` on this gate at [093](093-yield-phase.md), so this re-entry re-proves the authored gate on moved bytes and authors nothing.

## Contract Draft

Semantics are unchanged from [070](070-forward-qa.md), [096](096-forward-qa.md) and [099](099-forward-qa.md); this pass authors no scenario and changes no assertion. The draft is restated so the alias edits cannot be read as a contract change.

| Scenario | Real trigger | Observable assertions | Forbidden assertions | Lifecycle proof | Telemetry proof | Harness entry |
|---|---|---|---|---|---|---|
| TC-100 | collapsed format + 3 turns, no flush parameter (`022`) | no artifact before a flush; one `<format>.log` after shutdown carrying all three turns; one write-bearing op | per-turn file / `index.txt` / one artifact per pending turn | process healthy across the turns, rc 0 | none declared | `test_tc100_collapsed_format_defers_every_write_until_a_flush` |
| TC-001 | documented write path and collapsed flush path raise (`PermissionError` on the session dir) | upstream status/body unchanged; exception logged; process keeps serving | a dump-caused 5xx or a silent log absence | rc 0 after the raising flush | none declared | `test_tc001_documented_write_failure_still_returns_the_upstream_response`, `..._collapsed_flush_failure_...` |
| TC-107 | no `--dump-format`, no config override | documented layout at `2N` writes; no `.log`; complete normalized body, nonce stripped | a collapsed artifact under the default | unchanged from today | none declared | `test_tc107_no_format_configured_produces_the_documented_layout` |
| TC-108 | `--dump-format documented` at `N=3`, nonce-bearing bodies | three files, README content and line format, `2N = 6` writes, `N + 1 = 4` files | drift in the body, diff source, index columns or write count | unchanged from today | none declared | `test_tc108_documented_format_preserves_contents_at_2n_writes` (+ `test_tc108_sensitivity_documented_line_pattern_distinguishes_drift`) |

## Adopted Concurrent Edits

Five tracked QA/harness edits were uncommitted when this pass started (mtime 2026-10-17 21:14:27; `runs/100-yield-qa.txt` at 21:13:46 predates them). They are QA artifacts and the harness declaration this pass operates under, so the pass validated and committed them with the gate rather than leaving a validated gate unreproducible from `HEAD`.

| File | Edit | Verdict |
|---|---|---|
| `.agents/skills/qa-project/SKILL.md` | §Upstream declares `model: sloppus` as the shared alias and any other name as a 404 before generation | **adopted** — verified by direct probes: `sloppus` → HTTP 200, `qa-gate-model` → HTTP 404 `NotFoundError`; this is the harness contract the gate runs under |
| `tests/qa/harness.py` | `turn_body` default model `qa-gate-model` → `sloppus` | **adopted** — the default is the declared alias; no assertion reads the default |
| `tests/qa/test_qap_ms001.py` | duplicate-key body's last value `m-last` → `sloppus`; assertion follows | **adopted** — `m-first` still differs, so last-wins discrimination is intact; a first-wins writer still fails |
| `tests/qa/test_qap_ms002.py` | three `qa-gate-model` literals → `sloppus` | **adopted** — each asserted value is the body the test itself sends; no `VP-T002-*` semantics change |
| `tests/qa/tls_peer.py` | the loopback peer's model list/message and its answered-request oracle now name `sloppus` | **adopted** — the peer answers whatever the gate sends; the `MARKER` limb is untouched |

`test_sensitivity_upstream_oracle_distinguishes_transport_failure_from_a_relayed_body` still samples the literal `{"error":"The model \`qa-gate-model\` does not exist"}`. That is a sample *body*, not a request; the direct probe now shows the live service really returns exactly that shape for a non-declared name, so the sample stays accurate.

## Boundary Inspection Exceptions

None. The pass read only the approved contracts, `qa-project`, the existing gate modules and the shared harness; no production internals were inspected.

## Files Changed

- `iterations/105-forward-qa.md` (this report)
- `qa-plan.md`, `plan-summary.md`, `handoff-context.md` — status and evidence references only
- Adopted with this pass, zero bytes authored by it: `.agents/skills/qa-project/SKILL.md`, `tests/qa/harness.py`, `tests/qa/test_qap_ms001.py`, `tests/qa/test_qap_ms002.py`, `tests/qa/tls_peer.py`
- Run evidence: `runs/105-qap-ms001.txt`, `runs/105-qap-ms001-measures.txt`, `runs/105-qap-ms002.txt`, `runs/105-qap-ms004.txt`, `runs/105-hermetic.txt`
- Zero production, config, `invariants/`, `decisions.md`, or approved `TC-*`/`VP-*` bytes.

## Harness Execution

At `HEAD 8d27f6e`, with the five adopted edits in the tree (the run's test/harness bytes are exactly the bytes the checkpoint commits):

| Command | Result |
|---|---|
| `curl -m 8 http://hyperipper.local:30000/v1/models` | HTTP 200; served ids include `sloppus` |
| direct `POST /v1/messages` `model: sloppus` | HTTP 200 |
| direct `POST /v1/messages` `model: qa-gate-model` | HTTP 404 `NotFoundError: The model \`qa-gate-model\` does not exist.` |
| `.venv/bin/python -m pytest tests/qa/test_qap_ms001.py -o addopts= -q` | **16 passed** in 13.64s (`runs/105-qap-ms001.txt`) |
| `.venv/bin/python -m pytest tests/qa/test_qap_ms002.py -o addopts= -q` | **15 passed** in 49.21s (`runs/105-qap-ms002.txt`) |
| `.venv/bin/python -m pytest tests/qa/test_qap_ms004.py -o addopts= -q` | **6 passed** in 44.29s (`runs/105-qap-ms004.txt`) |
| `.venv/bin/python -m pytest` (hermetic default) | **513 passed @ 96.71%** (`runs/105-hermetic.txt`) |
| scoped `-s` run of TC-100 + TC-108 | TC-100 `write_ops=1`, `write_bytes=1418`, three empty pre-flush snapshots; TC-108 documented half `write_ops=6=2N`, `write_bytes=1041`, `files=4=N+1` (`runs/105-qap-ms001-measures.txt`) |
| `grep -n "dump-flush" tests/qa/test_qap_ms001.py` | docstring only; no `extra_args` passes a flush flag (`022` honored) |

The scoped `-s` run is an output-capture run, not a confidence rerun: the full-module run passed but pytest discarded the two plan-required measures. Production bytes are unchanged since [099](099-forward-qa.md) (`git diff f3468c7..HEAD -- buster_ripper/` is empty), so the measures match 099's recorded `2N = 6` / `N + 1 = 4` and `write_ops = 1`.

Required checks: (1)–(3) are the scenario bodies re-executed above; (4) is the module's existing sensitivity witnesses — the three process-free `S21`/`S22` witnesses and the README line-pattern witness; (5) is the R1/R2 hostile tier re-executed (`UNKNOWN_FORMATS`, body parse boundaries, symlink containment); (6) N/A — no payload field changed; (7) is the R5 near-miss `sid` pair re-executed.

`ready` is not a verdict: the `MS-001` Yield Phase re-runs and writes the authoritative `pass`/`fail`.

## Next

`MS-001` Yield Phase: read the runs above if `HEAD` is unchanged and the tracked tree is clean, otherwise re-run `tests/qa/test_qap_ms001.py` and write the authoritative verdicts.

# Iteration 048 — Forward QA (QAP-MS002 assertion fidelity repair)

## Orchestrator Prompt (verbatim)

```text
You are a subagent spawned by another session.
Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
Milestone: MS-002
QA Pass: QAP-MS002
Mode: QA-only fidelity repair
Source: iterations/047-forward-qa.md and its absorbed QA assertion closure packet. Preserve real-upstream requirement; do not count an upstream-down baseline as milestone PASS.
```

## QA Pass

- `QAP-MS002` after accepted-local `T003` and `T003.001`, before `MS-002` Yield Phase.
- `VP-T001-session-isolation`, `VP-T002-collapsed-artifact`, `VP-T002-change-readable`, `VP-T002-turn0-available`, `VP-T002-write-count-reduced`, `VP-T002-volume-flush`, `VP-T002-file-count-reduced`, `VP-T003-shutdown-flush`, `VP-T003-interval-flush`, `VP-T003-timer-resilience`; `TC-101`–`TC-106`, `TC-109`–`TC-112`.
- Existing `tests/qa/test_qap_ms002.py` and `tests/qa/harness.py` are retained; no intermediate QA helper needs promotion.

## Contract Draft

| Scenario | Real trigger | Observable assertions | Forbidden assertions | Lifecycle proof | Telemetry proof | Harness entry |
|---|---|---|---|---|---|---|
| TC-101 | Two near-miss prompts, timer flush, live upstream | Two isolated session artifacts; successful proxy transit independent of model output | Merged records, proxy connection error | Healthy process and both artifacts | none | real subprocess + POST + listing |
| TC-102/103/104 | Three turns per collapsed format, volume flush | Single named artifact, ordered records, readable change and full turn 0 | Documented files, lost change/body, proxy connection error | Append-only | none | POST + artifact |
| TC-105/109 | Same N in collapsed and documented formats | F vs 2N ops, pending-batch bytes, 1 vs N+1 files | Per-turn/rewrite writes, both writers, proxy connection error | Per-session counts | none | instrumented subprocess |
| TC-106 | Cross byte threshold | Batch on disk while process lives | Early/absent flush, proxy connection error | Live process | none | POST + listing |
| TC-110 | Real SIGINT and SIGTERM | Buffered turns, clean exit | Lost turns, traceback, proxy connection error | Wait for exit | none | subprocess + signal |
| TC-111/112 | Timer; failing tick then recovery | Live flush, logged failure and later success | Dead timer, fatal cleanup, proxy connection error | Poll artifact, clean cancellation | none | subprocess + artifact/stderr |
| R1/R2/R5 | Invalid bodies, symlinks, identity near misses | Typed no-write for invalid bytes; separate valid identities | Server error, escaped write, proxy connection error on valid probes | Process healthy | none | raw POST + filesystem |

The upstream is mandatory for every gate. A reachable upstream may return a non-model-specific 2xx or 4xx response for these test requests; no assertion may require the previous DNS-failure 502 baseline or inspect model output. Upstream-down must stop the gate as unavailable, not pass it.

## Boundary Inspection Exceptions

None. This repairs only QA-side harness and assertions using the public process contract, without reading implementation internals.

## Files Changed

- `tests/qa/harness.py`: directly probe the declared remote endpoint before spawning the proxy; DNS, connection, timeout, or server failure stops the gate instead of permitting an upstream-down 502 as evidence.
- `tests/qa/test_qap_ms002.py`: replace the hard-coded 502/`Cannot connect` oracle with an upstream-response discriminator (2xx/4xx and no proxy connection marker), applied to every valid POST; no model-output assertion and no changed artifact/typed-error contracts.
- `qa-plan.md`: readiness/evidence reference only.
- This report.

## Harness Execution

- `.venv/bin/python -m pytest tests/qa/test_qap_ms002.py -o addopts= -q -x` → expected **prerequisite failure before any proxy is started**: `hyperipper.local:30001` DNS `gaierror [Errno 8]`, `RuntimeError: required live upstream ... unavailable`. No gate scenario completed. This cannot be recorded as milestone PASS.
- Direct QA discriminator smoke check: 200/400 allowed; 502/500 and a 400 carrying `Cannot connect` rejected. `py_compile` for both QA files succeeded. These are not substitutes for a live process gate.
- After reachability is restored, rerun the full `QAP-MS002` suite without `-x` against the declared upstream and let Yield Phase inspect the independent result.

## Result

**BLOCKED-HARNESS** — persistent QA retainer repaired, but the mandatory real upstream is not resolvable in this environment. The previous upstream-down green run is not a milestone gate result. No production behavior or approved scenario semantics changed; the planning QA Capability Gate did not miss an application testability seam.

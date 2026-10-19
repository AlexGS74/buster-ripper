# Forward QA 096 — `T021 · harden the MS-001 documented-format gate`

## Orchestrator Prompt (verbatim)

> You are a subagent spawned by another session.
> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
> Task: T021

## Result

`READY-PASSING` — the strengthened `TC-107`/`TC-108` observe the approved contract on the current bytes, and the two content-drift shapes are rejected by the authored comparisons. Not an approval verdict.

## QA Pass

- **QAP**: `QAP-MS001` (re-entry) — `qa-plan.md` §QA Author Passes, re-entered by `T021`'s planned-task
  context (`plan-summary.md:386`, `handoff-context.md:129-219`). This is not a fresh milestone pass:
  `T021` gates the existing `VP-T002-documented-format-preserved` and is **not** an `MS-001` constituent,
  so the `MS-001` constituent set (`T001`, `T002`, `T005`) is already `accepted-local` and the
  milestone-incomplete block does not apply.
- **VP**: `VP-T002-documented-format-preserved` (`plan-summary.md:228`) — "today's content and line
  format", `Evidence needed`: "file contents".
- **TC**: `TC-108` (`qa-plan.md:53`, §Scenario Contracts `:210-217`), and the shared helper it shares with
  `TC-107` (`qa-plan.md:52`).
- **Gate handles**: `G-008`(a) (`VP-T002-documented-format-preserved`'s only witness), `US-012` first
  clause, `OI:buster_ripper.dump/dump_turn/output_layout`.
- **Adjudication**: `YR-QA-1 · documented-format gate cannot fail on content drift` (`095-yield-red`
  §Adjudication, run `YR-MS001-r2`); blind evidence `yr-p002a.md` DOES-NOT-HOLD, `yr-p002b.md` HOLDS,
  `yr-p001.md` HOLDS, `yr-p020.md` HOLDS.
- **Result state**: `READY-PASSING` (see §Harness Execution). This is not an acceptance verdict.

## Contract Draft

Persisted before any production-source reading, from the approved `VP-*`/`TC-*`/`T021` records, the
public `README.md` dump contract, the existing `tests/qa/` fixtures and the adjudicated loophole measurement.

| Scenario | Real trigger | Observable assertions | Forbidden assertions | Lifecycle proof | Telemetry proof | Harness entry |
|---|---|---|---|---|---|---|
| TC-108 · documented format preserved | `--dump-format documented`, N=3 turns, each body carrying a **fresh billing nonce** in the system prompt (`README.md:105`), through `POST /v1/messages` to the live upstream | (1) `turn_000.json` parses to **exactly** the sent body with the nonce-bearing `system` replaced by the nonce-free system text — a complete expected value set (`README.md:153` "full normalized request body"); (2) the nonce token appears in **no** artifact in the session dir; (3) each `turn_NNN.diff`'s changed `+`/`-` lines are **exactly**, in order, `difflib.unified_diff` over the **self-contained artifact's** consecutive `--- turn NNN rendered ---` sections — the documented contract's own rendering (`README.md:187` "every later turn stores its unified diff", `:188` self-contained stores "its complete rendering"); (4) layout `N + 1`, index line shapes and `2N` decomposition unchanged | a pre-normalization body in `turn_000.json`; the nonce in any artifact; a diff computed over raw request JSON (`json.dumps(body)`); a diff whose source is anything but the per-turn rendering; a changed filename, count, index shape or write decomposition | process exits 0; the documented run is read only after exit; the self-contained run's rendering is read only after its own exit (shutdown flush, no flush parameter per `022`) | none — the project declares no telemetry capability (`qa-plan.md:22-25`) | `Proxy(..., extra_args=["--dump-format","documented"], counted_report=...)` for the documented half; a second `Proxy(..., extra_args=["--dump-format","self-contained"])` in its own dump root for the rendering oracle |
| TC-107 · default encoding (shared helper) | no `--dump-format`, no config override, same nonce-bearing bodies | `turn_000.json` parses to exactly the sent body with the nonce-free `system`; the nonce appears nowhere; documented layout at `2N`, no `.log` | a collapsed `.log` with nothing configured; any pre-normalization body; any content drift in the default path | process exits 0; post-exit inspection | none | `Proxy(...)` with no format argument |

**Forbidden during authoring**: calling `_render_body`, `normalize`, `dump_turn` or any module helper to
build the expectation; asserting the rendering's exact text as a golden string; widening or narrowing
`VP-T002-documented-format-preserved`; touching production bytes.

## Boundary Inspection Exceptions

### Boundary Inspection Exception
- **Need**: `turn_000.json`'s expected value must be "the normalized body", and a nonce-free body is
  identical before and after normalization, so the gate needs the exact framework form Claude Code injects.
- **Path/symbol**: `buster_ripper/normalize.py:11-15` (`_BILLING_RE`).
- **Question**: what byte form does the billing-header block take, and what does stripping leave behind?
- **Assertion impact**: none — the approved contract remains "the normalized request body, no billing nonce".

### Boundary Inspection Exception
- **Need**: the documented diff's source must be pinned to the per-turn rendering without calling
  `_render_body`; the only public artifact that stores a rendering is the `self-contained` collapsed
  format, so the gate must know where its per-turn rendering begins.
- **Path/symbol**: `README.md:186-211` (§Collapsed dump formats, §Reading a collapsed artifact) and the
  existing QA fixture `tests/qa/test_qap_ms002.py:71-76` (`record_section`), which already reads
  `--- turn NNN rendered ---` sections from that artifact.
- **Question**: which public section delimiter bounds one turn's rendering?
- **Assertion impact**: none — the expected diff is computed with `difflib` over that public rendering,
  never by calling the implementation's renderer.

## Files Changed

- `tests/qa/test_qap_ms001.py` — `assert_documented_contents` now compares `turn_000.json` against a
  complete expected body and each documented diff's changed lines against the rendering-derived diff;
  `TC-107`/`TC-108` drive nonce-bearing bodies and the nonce-free session id; `TC-108` adds the
  self-contained rendering run; one process-free sensitivity witness added.
- `tests/qa/harness.py` — `turn_body` gains an optional `nonce` parameter; `billing_system` added;
  `record_section` promoted out of `test_qap_ms002.py` so both gates share one public-artifact reader.
- `tests/qa/test_qap_ms002.py` — imports the promoted `record_section`, its local copy removed (no
  assertion changed).
- `qa-plan.md` — `QAP-MS001` re-entry note and `TC-107`/`TC-108` evidence pointers.

## Harness Execution

Commands, all at `HEAD c9b4751` with only this pass's uncommitted `tests/qa/` bytes in the tree — these
runs prove the authored gate executes; a later agent reruns if those bytes move:

| Run | Command | Result |
|---|---|---|
| `QAP-MS001` | `.venv/bin/python -m pytest tests/qa/test_qap_ms001.py -o addopts=` | **12 passed in 2.90s** — the eleven pre-existing tests plus the new sensitivity witness |
| `QAP-MS002` | `.venv/bin/python -m pytest tests/qa/test_qap_ms002.py -o addopts=` | **15 passed in 14.66s** — the promoted `record_section` reader is byte-compatible with the intermediate it replaced |
| `QAP-MS004` | `.venv/bin/python -m pytest tests/qa/test_qap_ms004.py -o addopts=` | **6 passed in 44.27s** (own loopback peer) |
| hermetic | `uv run pytest` | **511 passed @ 96.71%** |

Output files: `runs/096-qap-ms001.txt`, `runs/096-qap-ms002.txt`, `runs/096-qap-ms004.txt`,
`runs/096-hermetic.txt`. The `:30000` preflight passed on every QAP-MS001/QAP-MS002 run.

Observed on the strengthened gate (printed by the run, not a verdict):

- `TC-108`: documented `turn_000.json` **equals** `{**sent, "system": nonce-free}` — the
  billing nonce appears in no artifact of either root.
- `TC-108`: `turn_001.diff`'s changed lines are exactly `-documented-t0…`, `+documented-t1…`
  and `turn_002.diff`'s exactly `-documented-t1…`, `+documented-t2…` — the rendering-derived
  diff matches at the changed-line level.
- `TC-108`: `write_ops = 6 = 2N`, `files = 4 = N + 1`; `TC-100` unchanged at `write_ops = 1`
  for three buffered turns.
- `022` honored: `grep -n "dump-flush" tests/qa/test_qap_ms001.py` finds only the docstring line, no
  `extra_args` occurrence.

The two production mutations the done-when names (`YR-QA-1`'s arms) are **local Yield's** replay; this pass
did not modify production bytes, and `git diff --name-only` shows only `tests/qa/` plus session artifacts.

## Open

- **`095-yield-red`'s artifacts are still uncommitted** — `iterations/095-yield-red.md`, the
  `assumptions/verdicts/yr-p0*.md` set and `assumptions/compositions/S-YR-*.md` — owner: the dispatcher,
  which commits the Yield Red chain's output when the chain ends. This pass did not stage them (one run, one
  checkpoint).
- **The two mutation replays are owed** — owner: local Yield on `T021`. The process-free sensitivity witness in
  this pass proves the comparisons discriminate the drift shapes; only a production mutation replay proves the real gate reds.
- **Assumption extraction for the new gate surface is owed** — owner: the orchestrator's `forward-collect` route.
  `T021` carries `required:new-surface` (the complete-body and rendering-source propositions have no existing seam),
  so the extract/check/compose sequence runs on the bytes this pass wrote, before the closing Yield Red check.

## Next

Local Yield on `T021` (replay both production mutations, revert byte-exact, then accept), then the closing
Yield Red check on the strengthened bytes.

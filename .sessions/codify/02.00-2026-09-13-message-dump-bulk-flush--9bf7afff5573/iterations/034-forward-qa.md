# Iteration 034 — Forward QA (QAP-MS002)

## Orchestrator Prompt (verbatim)

```text
Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
Milestone: MS-002
QA Pass: QAP-MS002
Target base: main (72e3a1c6861bf0d8de6317adf52289e5f6cfa4e1)
```

## QA Pass

- **Pass**: `QAP-MS002` (qa-plan.md §QA Author Passes) — runs after T003 accepted by
  local Yield ([032-01-yield](032-01-yield.md), Checkpoint 67), before the MS-002 Yield Phase.
  Scheduling verified: MS-002's constituent set is T003 only; MS-002's Yield Phase has not run.
- **Gate handles**: TC-101, TC-102, TC-103, TC-104, TC-105, TC-106, TC-109, TC-110, TC-111, TC-112 →
  VP-T001-session-isolation, VP-T002-collapsed-artifact, VP-T002-change-readable, VP-T002-turn0-available,
  VP-T002-write-count-reduced, VP-T002-volume-flush, VP-T002-file-count-reduced, VP-T003-shutdown-flush,
  VP-T003-interval-flush, VP-T003-timer-resilience.
- **Milestone**: MS-002 (collapsed path observable: volume/timer/shutdown flush).
- **Target base**: `main @ 72e3a1c` verified ancestor of HEAD (`git merge-base --is-ancestor` OK;
  matches [033-01-forward](033-01-forward.md)'s base-sync record — no merge was owed).
- **Adversarial tiers** (skill Required Checks 5–7): R1 hostile parse corpus and R2
  symlinked/dangling writer targets → mapped below to VP-T002-collapsed-artifact (record/parse
  boundary) and VP-T002-volume-flush (flush writer into a pre-existing `DUMP_DIR` tree);
  R5 identity near-misses (sid 1-char pair; `(sid, format)` pair) → VP-T001-session-isolation and
  VP-T002-collapsed-artifact. R4 payload census: **not applicable** — the milestone adds a
  human-read artifact with no in-code consumer and no cross-language decoder
  (plan RB-003: "no reader in code"); no payload field is added.
- **Environment disclosure (load-bearing)**: the declared live remote upstream
  `http://hyperipper.local:30001` was **unreachable for this entire pass** (DNS resolution
  failure; no `/etc/hosts` entry, no Bonjour answer). `qa-project` declares "a gate run fails if
  that host is unreachable", and `US-009` settles *never a stub* — no substitution was made.
  A probe run established the executable surface: dump capture and session-dir creation happen at
  request interception (stderr shows `dump: new session …` at request time, the upstream
  connection error ~5 s later), the shutdown flush fires on SIGINT, and exit is clean — so every
  QAP-MS002 assertion surface (dump artifacts, flushes, signals, exit codes) is exercisable with
  every proxied request returning the **502 upstream-connection baseline** (~5 s each). No
  QAP-MS002 scenario asserts a 2xx; request-isolation clauses are asserted as outcome-equivalence
  with that 502 baseline (same status, same upstream-attributed error body, process keeps
  serving). Yield Phase should re-run the gates when the upstream is reachable again.
- **Scope note**: `tests/qa/` did not exist — **QAP-MS001 has not run** (TC-100/TC-001/TC-107/TC-108
  unauthored; MS-001's Yield Phase also not run). That is outside this pass's write boundary.
  TC-105/TC-109's scenario trigger itself names the documented run ("two runs at the same N — one
  per collapsed format, one documented … the pair is completed here"), so the documented half is
  authored **here**, inside QAP-MS002, exactly as the approved scenario text directs.

## Contract Draft

Independent interpretation of the approved `VP-*`/`TC-*`/`MS-*` records, `qa-project`, and
public runtime output only (probe artifact + `--help` + README). No implementation source,
diff, iteration report of Forward, or implementation unit test was read (see Boundary
Inspection Exceptions — none).

Calibration facts used (public observables): sid = `sha256(system prompt text)[:12]` (verified
against the probe run); collapsed artifact = `DUMP_DIR/<sid>/<format>.log`; record sections
`--- turn NNN body ---` / `--- turn NNN rendered ---` with one summary line per record in the
documented index form (`turn 000 … FULL …` / `turn 001 … +A/-R lines …`); rendered text
includes the system prompt, so per-turn rendered bytes ≈ system + user content.

| Scenario | Real trigger | Observable assertions | Forbidden assertions | Lifecycle proof | Telemetry proof | Harness entry |
|---|---|---|---|---|---|---|
| TC-101 (+R5 sid near-miss) | Proxy `--dump-format change-focused --dump-flush-interval 2 --dump-flush-bytes <unreachable>`; two system prompts differing by exactly one character, one turn each; the timer tick is the driven flush | Two session dirs at the two computed sids, each holding its own `<format>.log`; each artifact contains only its own prompt's sentinel (and not the other's); artifacts appear with no shutdown | One directory for both prompts; one session's flush writing the other's records | Process healthy across all turns; both artifacts appear while the process runs | none declared | subprocess + real HTTP POSTs + directory listings |
| TC-102 (× each collapsed format) | Fresh dump dir, `--dump-format <value>`, turns 0..2 of ~40 KB rendered each, `--dump-flush-bytes 100000` → one volume flush carries all three; interval 3600 | Listing is exactly `{<format>.log}` (filename states the configured value); no `turn_000.json`/`turn_NNN.diff`/`index.txt`; `turn 000/001/002` tokens at strictly increasing offsets (arrival order); computed sid == the created directory; artifact absent after turn 1 (sensitivity/pre-trigger state) | Two artifacts for one configured format; any documented file on the collapsed path; records out of arrival order | Append-only artifact extended by the flush; no per-turn writes before the threshold | none | subprocess + listing snapshots between responses + artifact text |
| TC-103 (× each collapsed format) | Same run as TC-102; turn 1 content carries `old-marker`, turn 2 the same message changed to `new-marker` | change-focused: a unified-diff `-` line containing `old-marker` and a `+` line containing `new-marker` inside turn 2's record; self-contained: turn 1's rendering contains `old-marker`, turn 2's rendering contains `new-marker` (reader-performed comparison is possible); every turn ≥1 summary line carries `+N/-M` counts and its turn number; turn 0's line carries `FULL` | A summary line that lost the turn number or the change counts; the change unlocatable from the directory alone | n/a (content assertions) | none | artifact text parsing |
| TC-104 (× each collapsed format) | Same run as TC-102 | The `--- turn 000 body ---` block parses as JSON and carries the sent model/system/message sentinels (raw normalized body recoverable); the `--- turn 000 rendered ---` block carries the bare sentinels and message-section markers (rendering recoverable) | Turn 0 stored only as a diff; body or rendering unrecoverable from the artifact | n/a | none | artifact text parsing |
| TC-105 + TC-109 (paired runs, the G-008(b) measurement) | Three instrumented runs at N=6, turns ~40 KB rendered, threshold 100000 (flush after turn 2 and turn 5 → F=2): change-focused, self-contained, documented | Collapsed (per format): instrumented write-bearing ops on dump artifacts == 2 == F; every op path is `<sid>/<format>.log`; Σ op bytes == final artifact size (append-only); each op's bytes < final size (rewrite-per-flush excluded); listing == 1 file. Documented: ops == 12 == 2N; listing == {`turn_000.json`} ∪ 5 diffs ∪ {`index.txt`} == 7 == N+1; `index.txt` has N lines; `turn_000.json` parses. The three measurements are printed as the paired comparison evidence for the MS-002 Yield Phase's comparison record | `2N + F` in any configuration (witness C); a flush that writes once per pending turn (witness A); a rewrite-per-flush whose bytes grow O(N²) (witness B — excluded by both byte clauses); a collapsed run that also writes documented files | Counts are per session; F deterministic (injected threshold; timer disabled); nothing pending at shutdown | none | subprocess via `counted_proxy_entry.py` (io.open/builtins.open counter scoped to the dump root; mkdir and SQLite outside the measured unit) + JSON report at exit |
| TC-106 | Interval 3600 (timer cannot fire), threshold 100000, turns ~40 KB | After turn 1's response: no `.log` artifact anywhere under the dump dir; after turn 2's response: artifact exists with exactly turns 0,1,2 in arrival order; no shutdown, process still running | Turns accumulating past the threshold with nothing on disk; a check evaluated before the append or off by one (flush at turn 1, or four records) | Collapsed-path only; the check reads the same field the append wrote | none | listing snapshots between responses |
| TC-110 (SIGINT and SIGTERM, separate runs) | 3 small turns pending (interval 3600, threshold unreachable); artifact absent pre-signal (sensitivity); real signal delivered to the subprocess | Exit code 0; no `Traceback` in stderr; artifact present post-exit with the 3 records in arrival order | Buffered turns lost at shutdown; a traceback replacing GracefulExit; non-zero exit | Quiescence = wait for process exit before inspecting the directory | none | subprocess + `os.kill` + post-exit listing + exit code |
| TC-111 | Interval 3 (small), threshold unreachable; one appended turn | Artifact absent immediately after the response; then appears (contents gain the turn) while the process keeps running, observed by polling (no fixed sleep); process alive throughout | The artifact never gains the turn for a live session (dead or wrong-state flush task) | Staleness bounded by the injected interval, observed via artifact poll | none | live artifact polling with deadline |
| TC-112 | Interval 2, threshold unreachable; turn 0 flushed by a tick; artifact `chmod 0444` (flush now raises); turns 1,2 appended; permissions restored | `PermissionError` appears in stderr (the failing tick is logged — polled, not slept); after `chmod` back, a later tick flushes turns 1,2 (artifact gains them — polled); SIGTERM then exits 0 with no traceback (cancelled timer is clean) | A timer task that dies silently on its first exception; a flush that raises out of `on_cleanup`; turns lost | The loop catches per-iteration exceptions and continues; cancel is clean | none (log record is process stderr, not telemetry) | chmod injection + stderr/artifact polling + exit code |
| Hostile H1/H2 (R1 → VP-T002-collapsed-artifact) | Raw non-UTF-8 bytes body; empty body — both `application/json` | Typed stop: HTTP 400 for each; no session dir and no artifact created (zero writes); the process still serves afterwards (a following valid request reaches the dump and the 502 baseline) | A 5xx or crash caused by the hostile body; a dump record written for an unparsable body | Process healthy after the hostile inputs | none | raw-byte HTTP via urllib + listing |
| Hostile H3 (R1 duplicate JSON keys) | Body with duplicate `"model"` keys (`m-first`, then `m-last`), threshold 1 (immediate flush) | Request is forwarded (502 baseline — duplicate keys are valid JSON, not a 400); the flushed turn-0 body JSON has `model == "m-last"`; `m-first` appears nowhere in the artifact | The record carrying the first value or both values (duplicate-key corruption); the body rejected as malformed | n/a | none | raw-byte HTTP + artifact JSON parse |
| Hostile H4 (R1 trailing newline) | Valid JSON body followed by `\n\n` | Request forwarded (502 baseline); the turn's record is created (parse boundary classifies trailing whitespace as valid) | The body misclassified as malformed; no record | n/a | none | raw-byte HTTP + artifact |
| Hostile H5 (R2 dangling session-dir symlink → VP-T002-volume-flush) | `DUMP_DIR/<sid>` pre-created as a symlink to a dangling path outside the dump root; one valid turn, threshold 1 → volume flush attempts to write | No-write stop: nothing is created at or through the dangling target; the dump-dir listing still contains only the symlink; the request returns the 502 baseline (dump failure isolated); process serves a second turn; SIGTERM exits 0, no traceback | A write escaping the dump root; the request failing because of dump behavior; a traceback | Contained failure; clean shutdown despite the failing flush | none | symlink fixture + listing + baseline equivalence |
| Hostile H6 (R2 artifact symlink escape → VP-T002-volume-flush) | Real `DUMP_DIR/<sid>/` with `<format>.log` pre-created as a symlink to an outside canary path; one valid turn, threshold 1 | No-write stop: the canary is not created and not modified; the session dir listing is unchanged; the request returns the 502 baseline; process healthy; clean exit | The flush writing through the symlink into the outside tree (containment breach) | Contained; clean shutdown | none | symlink fixture + canary + listing |
| Identity I2 (R5 `(sid, format)` near-miss → VP-T002-collapsed-artifact) | Same dump dir, same system prompt: run 1 `--dump-format change-focused` (1 turn, threshold 1), then run 2 `--dump-format self-contained` (fresh process, 1 turn, threshold 1) | The directory ends with exactly two artifacts — `change-focused.log` and `self-contained.log`; `change-focused.log` is byte-identical before/after run 2 (the key did not merge the second format into the first artifact); `self-contained.log` carries run 2's sentinel | One artifact receiving both formats' records (key too narrow — merged identities); the first artifact rewritten | `(sid, format)` identity per the VP's scope note (restart with a different format is legitimate) | none | two subprocess runs + byte comparison |

Intermediate-reuse note: `tests/qa/` did not exist, so there are no intermediates to promote or
subsume; this pass creates the shared fixture location itself (`tests/qa/harness.py`), which
later passes (QAP-MS001 if dispatched, QAP-MS004) are expected to import rather than duplicate.

## Boundary Inspection Exceptions

None. No implementation source, implementation diff, Forward iteration report, or
implementation-authored unit test was read. All calibration facts came from public runtime
observables: `--help`, the probe run's stderr/artifact, the README, and Python-stdlib behavior
(`io.open is builtins.open`; `Path.open` → `io.open`) verified in the project venv.

## Files Changed

- `tests/qa/__init__.py` (new — package marker so pytest imports the gate module cleanly)
- `tests/qa/harness.py` (new — shared process-level QA harness helpers)
- `tests/qa/counted_proxy_entry.py` (new — instrumented write-counter entry, the declared
  TC-105 evidence mechanism)
- `tests/qa/test_qap_ms002.py` (new — the QAP-MS002 gate tests)
- `qa-plan.md` (test paths + `ready` status only)
- `plan-summary.md` / `handoff-context.md` (Forward QA status pointer)
- this report

## Harness Execution

Command (explicit selection; `-o addopts=` strips the hermetic suite's coverage flags, which
do not apply to subprocess-only gates):

```
.venv/bin/python -m pytest tests/qa/test_qap_ms002.py -o addopts="" -v -s
```

Collection verified both ways: explicit path collects all 15 gates; a default
`pytest --collect-only` run collects **zero** `tests/qa` items (`norecursedirs` holds — the
hermetic default `uv run pytest` run is untouched).

### Environment during execution

`http://hyperipper.local:30001` unreachable (DNS failure) for the whole pass. Every proxied
request therefore returned the **502 upstream-connection baseline** (~5 s each; ~40 requests,
total suite wall time 5 m 34 s). Dump capture happens at request interception (probe evidence:
session-dir log line at request time, upstream error later), so every assertion surface the
scenarios name was exercised for real. Isolation clauses were asserted as outcome-equivalence
with the baseline (same 502, upstream-attributed body, process keeps serving).

### Observed results (final run: 1 failed, 14 passed, 334.13 s)

| Gate | Result | Observed |
|---|---|---|
| TC-101 (+sid near-miss) | pass | two dirs at the two computed sids; each artifact only its own prompt's sentinel |
| TC-102/103/104 `[change-focused]` | pass | exactly `change-focused.log`; 3 records in order; `-old/+new` diff lines; turn-0 body parses + rendering present; `FULL` / `+N/-M` summary forms |
| TC-102/103/104 `[self-contained]` | pass | exactly `self-contained.log`; renderings per turn enable the reader comparison; turn-0 body + rendering |
| Identity `(sid, format)` near-miss | pass | two artifacts in one dir; `change-focused.log` byte-unchanged by the second format's run |
| TC-105/TC-109 paired runs | pass | see comparison below — `F=2` collapsed (both formats) vs `2N=12` documented; bytes track batches (`Σbytes == final size`, each op `< final size`); files `1` vs `7` |
| TC-106 volume flush | pass | no artifact below threshold (after turn 1); turns 0..2 on disk after the threshold turn, process running |
| TC-110 `[sigint]` / `[sigterm]` | pass / pass | exit 0, no traceback, 3 buffered turns on disk in order after each real signal |
| TC-111 interval flush | pass | turn reaches the artifact while running (polled, no fixed sleep) |
| TC-112 timer resilience | pass | `PermissionError` tick logged; later tick flushes the retained batch; clean exit |
| Hostile non-UTF-8 + empty body | **FAIL** | **R1 finding** — see below |
| Hostile duplicate JSON keys | pass | forwarded (baseline, not 400); record carries `m-last`; `m-first` absent from the artifact |
| Hostile trailing newline | pass | forwarded; record created |
| Hostile dangling session-dir symlink | pass | no write at/through the dangling target; listing unchanged; request isolated; clean exit |
| Hostile artifact symlink escape | pass | canary untouched; no write through the symlinked artifact; clean exit |

### TC-105/TC-109 comparison record (the G-008(b) measurement, for the MS-002 Yield Phase)

```
[TC-105 comparison] N=6 format=change-focused write_ops=2 write_bytes=482762 files=1
[TC-105 comparison] N=6 format=self-contained  write_ops=2 write_bytes=282647 files=1
[TC-105 comparison] N=6 format=documented      write_ops=12 write_bytes=442217 files=7
```

- **M1 (write-bearing operations)**: `F = 2` per collapsed format vs `2N = 12` documented →
  reduction `1 − F/(2N) = 1 − 2/12 ≈ 83.3%`, identical for both collapsed formats
  (`PI:T002/dump_format/format_does_not_change_write_count` witnessed).
- **M2 (files)**: `1` vs `N + 1 = 7`.
- **Witness B excluded by measurement**: each flush's write carried only its batch
  (each op's bytes strictly below the final artifact size) and `Σbytes == final artifact size`
  for both collapsed runs — bytes grow with `F`, not `N²`.
- **Witness C excluded**: the collapsed runs wrote no documented files (listing exactly
  `{<format>.log}`); the documented run wrote no `.log`.
- Caveat carried from the plan: M1 is a **proxy** proof (M3/physical WAF unmeasurable in-repo);
  M4 logical bytes are not the proof and are recorded above only as context.

### R1 hostile finding (the one failure)

`test_hostile_non_utf8_and_empty_body_no_write` fails faithfully:

- **Observed**: a non-UTF-8 `application/json` body to `POST /v1/messages` returns **500**
  with an unhandled `UnicodeDecodeError` escaping `routes/messages.py:27` (`await
  request.json()`), logged as `Error handling request` + server traceback. The **empty** body
  correctly returns **400**. **Neither writes anything** (dump dir stays empty — the no-write
  stop holds), the process stays healthy, and shutdown is clean (exit 0).
- **Contract**: R1 (codify-adversarial-surface-discipline) requires the hostile corpus at every
  parse boundary with a *typed* stop; an unparseable client body is a client error (the empty
  body's own 400 is the in-repo precedent), not a 5xx that pollutes the server-error signal.
- The test is kept exactly as authored. Per this skill's rules a failing faithful test is
  routed by Yield Phase: test fidelity confirmed here against the R1 tier and the scenario's
  no-write clause; the behavioral fix belongs to implementation Forward.

### Assertion-fidelity correction (recorded for transparency)

The draft rows for TC-112/H5/H6 said "no `Traceback` in stderr". First execution failed those
three on exactly that string — but the tracebacks came from **logged** exceptions (the
injected `PermissionError` / refused symlink writes), which the scenario contracts themselves
require ("the raise is logged"); verbose log formatting includes traceback text by design.
The fatal-vs-clean discriminator the contract names ("a traceback replacing `GracefulExit`",
"a flush that raises out of `on_cleanup`") is the **exit status**. The three tests were
corrected to rely on `rc == 0` (already asserted) and the no-write/containment assertions,
which all hold. TC-110 — where no failure is injected — still asserts no traceback text and
passes on both signals. The R1 finding was NOT touched by this correction.

## Result

**READY-FAILING** — the persistent QAP-MS002 gate suite is authored under `tests/qa/`
(15 tests covering TC-101…TC-112 with the R1/R2/R5 tiers) and currently exposes one behavior
failure: the `/v1/messages` parse boundary returns an unhandled 500 (`UnicodeDecodeError`)
for non-UTF-8 bodies instead of the typed 4xx client-error stop (R1). All ten approved
QAP-MS002 scenarios pass, including the paired write-count/file-count measurement
`G-008`(b) consumes. This is not an acceptance verdict; Yield Phase must independently
inspect contract fidelity and re-run — ideally with the declared upstream reachable, since
this execution ran entirely against the 502 upstream-down baseline disclosed above.

# 070 — Forward QA: QAP-MS001 (MS-001 integration gates)

## Orchestrator Prompt (verbatim)

```
You are a subagent spawned by another session.
Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
QA Pass: QAP-MS001
```

Orchestrator instruction line:

```
Execute the forward-qa-interactive skill for this session. Arguments: $ARGUMENTS
```

## QA Pass

- **QAP**: `QAP-MS001` (`qa-plan.md` §QA Author Passes, line 63)
- **Runs after**: all `MS-001` tasks accepted by local Yield — **confirmed**: `T001`
  ([031-01-yield](031-01-yield.md)), `T002` ([031-02-yield](031-02-yield.md)), `T005`
  ([049-01-yield](049-01-yield.md)); all three archived in `plan-completed.md` rows 7, 8, 10.
- **Runs before**: `MS-001` Yield Phase. `MS-001` carries no authored gate yet
  (`qa-plan.md` line 95: "QAP-MS001 — not yet authored"), so the pass is still owed and is not
  scheduled after implementation Forward ahead of local Yield.
- **Validation points**: `VP-T001-writes-deferred`, `VP-T001-dump-failure-isolated`,
  `VP-T002-documented-format-preserved`, `VP-T002-default-encoding`
- **Scenarios**: `TC-100`, `TC-001`, `TC-108`, `TC-107`
- **Session identity**: `session-identity.json` → `session_uid 9bf7afff-5573-402a-b102-827423a1dded`
  matches the received handle suffix `…--9bf7afff5573`. No mismatch.
- **Harness**: `.agents/skills/qa-project/SKILL.md` — process-level E2E, real subprocess, real
  signals, live remote upstream `http://hyperipper.local:30001` (preflight answered `HTTP 200` at
  authoring time), gates under `tests/qa/`, selected explicitly.
- **Contract directive honored**: `qa-plan.md` §Phase Requirements line 193 and `TC-100`'s trigger
  note — **no `QAP-MS001` scenario passes `--dump-flush-interval` or `--dump-flush-bytes`.** Every
  gate below starts the proxy with `--dump-dir` and, where the scenario names one, `--dump-format`.
  Nothing else.

## Contract Draft

Stage 1 only. Sources: `plan-summary.md` §Validation Point Matrix (lines 194-203), `qa-plan.md`
§Scenario Contracts and §QA Gate Matrix, `.agents/skills/qa-project/SKILL.md`, `README.md` §Dump mode
(the public layout/line-format and reading-workflow contract, lines 139-220), `buster-ripper --help`
(the public selector surface: `--dump-format [documented|change-focused|self-contained]`,
`[default: documented]`), and the existing QA fixtures `tests/qa/harness.py`,
`tests/qa/counted_proxy_entry.py`, `tests/qa/test_qap_ms002.py`, `tests/qa/test_qap_ms004.py`.
No implementation source body, implementation iteration report, or implementation unit test was read.

| Scenario | Real trigger | Observable assertions | Forbidden assertions | Lifecycle proof | Telemetry proof | Harness entry |
|---|---|---|---|---|---|---|
| **TC-100** `VP-T001-writes-deferred` | Start the real proxy with `--dump-dir <fresh tmp> --dump-format change-focused` and **no flush parameter**; POST N=3 turns to `/v1/messages`; snapshot `DUMP_DIR/<sid>/` immediately after each request acknowledgement | Every snapshot taken while **no** artifact exists shows the session dir holds **nothing**: no `<fmt>.log`, no `turn_000.json`, no `turn_NNN.diff`, no `index.txt` (absent dir or empty dir both satisfy the VP). At least one such pre-flush sample must exist. After shutdown: the dir holds exactly `["change-focused.log"]`, and the counted report shows **every** write-bearing op targets that one artifact | A documented-path file appearing on the collapsed path (the deferral broken); one artifact per pending turn (multiple `.log` files); an assertion that no artifact can ever exist | Process healthy across all N turns (`rc == 0` at SIGTERM); the deferral **ends** at a flush: the post-run artifact contains all three turns' markers, in the same single file | none required (`qa-plan`: "Telemetry: none"; project declares no telemetry capability) | `Proxy(dump, logs, extra_args=["--dump-format", "change-focused"], counted_report=…)` |
| **TC-001** `VP-T001-dump-failure-isolated` — documented path | Pre-create `DUMP_DIR/<sid>` **read-only** (`0o500`), start the proxy with `--dump-dir` only (documented is the default and is also pinned explicitly in the second half), POST a valid turn | The response is the upstream's own: `200 <= status < 500`, no proxy transport failure in the body; the injected `PermissionError` **is** logged on stderr; a following valid session still captures | A 5xx caused by dump behavior; a silent dump failure with no log record | The process keeps serving after the failing write path; clean `rc == 0` at SIGTERM | none | `Proxy(dump, logs)` with a pre-seeded read-only session dir |
| **TC-001** `VP-T001-dump-failure-isolated` — collapsed path (the append/flush code T001+T002 added) | Same fault injected into a `--dump-format change-focused` run; the raise therefore surfaces at the **flush** (shutdown, since no flush parameter is passed) | The intercepted request still returns the upstream body; the failing flush is logged (`PermissionError` on stderr); nothing is written to the unwritable session dir | The raise escaping `on_cleanup` as a non-zero exit or a fatal traceback | Clean `rc == 0`; process served the request before the fault | none | `Proxy(dump, logs, extra_args=["--dump-format", "change-focused"])` |
| **TC-107** `VP-T002-default-encoding` | Start with `--dump-dir` **and no `--dump-format`**, every `*DUMP*` environment variable scrubbed (the "no config override" clause); POST N=3 turns | Sensitivity: no session dir before the first request. After: `DUMP_DIR/<sid>/` holds exactly `turn_000.json`, `turn_001.diff`, `turn_002.diff`, `index.txt`; `index.txt` carries one line per turn in the README-documented shape; `turn_000.json` is the full normalized body carrying the turn's text | A collapsed `.log` anywhere under `DUMP_DIR` with nothing configured — witness **W1** of `G-008`, the shape `US-012` forbids; a missing/renamed documented file | `rc == 0`; the directory layout is created by the run and left consistent | none | `Proxy(dump, logs)` + module-local env scrub |
| **TC-108** `VP-T002-documented-format-preserved` | Same run driven through the **instrumented** entry: `--dump-dir … --dump-format documented`, **no flush parameter**; POST N=3 turns | Write-bearing ops `= 2N` (`= 6`), decomposed per path: `index.txt` written once per turn (`N`) and each turn file written exactly once; every op inside `DUMP_DIR/<sid>/`. Contents match the README-documented contract: `turn_000.json` parses as the full body; `turn_00N.diff` are unified diffs (`--- / +++ / @@`) carrying the new turn's line as `+` and the replaced line as `-`; `index.txt` lines match the README's own reader pattern `^turn [0-9]{3}  [0-9]{2}:[0-9]{2}`, turn 000's line carries `FULL` and a `KB` field, later lines carry `+A/-R lines`, `<n> msgs`, `<n> tools` | A drifted `index.txt` line format; a missing or renamed documented file; a `.log` artifact on the documented path; a write count that is not `2N` | Counts are per session (all ops under this session's dir); `rc == 0` | none | `Proxy(..., counted_report=…)` via `harness.COUNTED_ENTRY` |
| **TC-108 sensitivity** (Check 4, pure) | The documented line-format matcher applied to the README's own published example lines and to drift forms | The matcher accepts every line of the README §Reading the index example block and the collapsed-summary identity clause's equivalent, and rejects: turn 000 without `FULL`, a line missing the timestamp column, `+A/-R lines` absent from a later turn's line, and a collapsed `.log` filename | Accepting a drifted form (a matcher that cannot fail is not an assertion) | n/a | none | in-module, no process |
| **H-R1** selector parse boundary (Check 5) | Start the real binary with `--dump-format` set to `documentedd`, `change_focused`, `DOCUMENTED`, `" documented"`, `""` | Each is refused at startup with a non-zero exit and a usage error naming the allowed set; `--dump-dir` is **never created** and holds no file | A mistyped value silently **falling back to `documented`** — the closed value set merging with the default identity, which would make TC-107's default clause unfalsifiable | The refusals are pre-serve: no port is bound, no directory is produced | none | direct `subprocess.run` on `.venv/bin/buster-ripper` (a startup-refusal helper has no existing QA equivalent) |
| **H-R1** request-body parse boundary, **documented writer** (Check 5) | POST, in order: non-UTF-8 bytes, an empty body, a duplicate-`model`-key body, a trailing-newline body | The two unparsable bodies are typed client errors (`400 <= status < 500`) and create **no** session directory; the duplicate-key body's `turn_000.json` carries the last key's value and no duplicate corruption; the trailing-newline body is captured as exactly one turn | A 5xx or a crash on hostile input; a session dir or documented file produced for a body that was never parsed; a duplicated turn record | Process still serves a valid sentinel turn after the hostile inputs and exits `rc == 0` | none | `Proxy(dump, logs)` (documented default) |
| **H-R2** documented writer into pre-existing trees (Check 5 — R2's sibling audit of the writer `MS-002` did not audit) | Two sids in one dump root: (a) `DUMP_DIR/<sidA>` pre-created as a **dangling symlink** to an outside target; (b) `DUMP_DIR/<sidB>/index.txt` pre-created as a symlink to an outside canary. Drive one turn per sid | No bytes land at either outside target; the symlink entries are still symlinks and no real directory replaced them; both requests still return the upstream body; the failing write is logged | Writing **through** a symlinked/non-regular target outside `DUMP_DIR`; the writer replacing the symlink with a real path to escape the fault | Clean `rc == 0` (the fault must not become fatal); process serves a later sentinel turn | none | `Proxy(dump, logs)` with pre-seeded symlink targets |
| **R5** identity-key near-miss (Check 7) | One run, no `--dump-format`; two system prompts differing in the **last character only**; `DUMP_DIR/<sidA>/change-focused.log` pre-planted with foreign bytes before the run | Two **distinct** session directories, each with its own full documented layout and its own `turn_000.json`; each `index.txt` carries only its own prompt's marker; the pre-planted `change-focused.log` survives byte-identical and is **not** counted as a documented artifact | Merging the two near-miss identities into one directory (the collision `R5` exists for); the documented writer removing, migrating or rewriting the pre-existing collapsed `.log` (`README.md` §Pre-existing session directories) | Per-session turn counters are independent; `rc == 0` | none | `Proxy(dump, logs)` |

**Payload census (Check 6, R4) disposition.** `MS-001` adds no wire payload field and no second
compiled decoder. The artifact's declared consumers are (1) the Python writer itself and (2) the
**shell reader documented in `README.md` §Reading the index / Reading a collapsed artifact**, whose
published pattern is `^turn [0-9]{3}  [0-9]{2}:[0-9]{2}` and whose column contract is
`turn NNN  HH:MM:SS  FULL|+A/-R lines  <n> msgs  <n> tools  [KB]`. That second consumer language is
exercised: TC-108 asserts the README's own pattern and column set against `index.txt`, so a dropped
or renamed column fails the gate somewhere.

**Promoted / subsumed intermediates.** `tests/qa/test_qap_ms002.py` already carries
`assert_upstream_response`, `read_text` and `log_artifacts` as module-local helpers, and this pass
needs all three for the same reason (a valid request must not be judged on model output; directory
listings must ignore non-dump files). Per the black-box-first rule they are **promoted** into the
shared fixture location `tests/qa/harness.py` and imported by both passes; `MS-002`'s local copies
are deleted rather than left duplicated. `session_id`, `Proxy`, `poll_until`, `turn_body`,
`COUNTED_ENTRY` and the counted-report instrument are **reused as-is**. Nothing in
`tests/qa/test_qap_ms004.py` overlaps `QAP-MS001`'s scenarios (that gate is the `MS-004` peer).

**For Yield Phase Check 6 (deletion decision).** Promoted by this pass and therefore no longer
duplicated anywhere: `read_text`, `log_artifacts` (+ its `rglob_files`), `assert_upstream_response`
— now single-defined in `harness.py`. Newly added to the shared harness, used by this pass only:
`startup_refused()` (the negative counterpart of `Proxy`, for inputs the CLI must reject outright).
`MS-002`'s remaining module-local helpers (`big_text`, `summary_lines`, `record_section`,
`TIMER_OFF_INTERVAL`, `TURN_SIZE`, `THRESHOLD_AT_TURN_2`, `UNREACHABLE_BYTES`) are **not** subsumed
by this pass — they exist for the collapsed-path thresholds `QAP-MS002` controls and this pass has
no use for them. Nothing here is a candidate for deletion.

## Boundary Inspection Exceptions

**None opened.** No implementation source body, implementation iteration report, implementation
diff, or implementation-authored unit test was read at any point in this pass. Every expected
outcome in the table above was fixed in Stage 1 from the approved `VP-*`/`TC-*` records,
`qa-project`, `README.md` §Dump mode (lines 139-220) and `buster-ripper --help`.

Two harness-connection questions arose that the public contracts did not answer, and both were
settled without reading source:

1. *"What does the app's own stderr look like when a dump write fails?"* — answered by running the
   gate and reading the captured `proxy.err` file, which is the evidence surface `TC-001` itself
   names ("the logged exception in stderr"). `codify-test-cost` rule 6 / the harness question is
   about connecting, not about expected values.
2. *"Is the session directory created at append time or at flush time?"* — not needed: the VP
   declares either satisfactory, and the assertions are written to hold under both (a pre-flush
   snapshot asserts the directory is absent **or** empty).

One assertion was **removed** during execution, and the removal went against the implementation,
not with it: `test_hostile_r2_...` initially also required the refusal to name a specific exception
class in stderr (`FileNotFoundError`/`PermissionError`). The run showed the documented writer does
contain the fault (`ValueError: dump: refusing non-regular target …`, `FileExistsError` for the
dangling session dir, both logged under `dump failed`) and writes nothing outside the root — but
"the refusal must log class X" is not in `R2`'s contract or in any `VP-*`, so the assertion was
replaced by what the contract does require (containment: no byte at either outside target, no
symlink replaced, the healthy session still captures) rather than by the observed wording. Had it
been left as-is the gate would have been a wording snapshot; it is now a behavior gate that can
still fail.

## Files Changed

Production source, implementation unit tests, and the approved `VP-*`/`TC-*` semantics: **untouched.**

| File | Change |
|---|---|
| `tests/qa/test_qap_ms001.py` | **NEW** — the `QAP-MS001` gate module: 10 tests (`TC-100`, `TC-001` ×2 paths, `TC-107`, `TC-108`, `TC-108` sensitivity, R1 selector refusals, R1 body parse boundary, R2 documented-writer containment, R5 identity near-miss) |
| `tests/qa/harness.py` | Shared fixture location extended: `read_text`, `rglob_files`, `log_artifacts`, `assert_upstream_response` promoted in from `MS-002`; `startup_refused()` added (negative startup check). `Proxy`, `session_id`, `poll_until`, `turn_body`, `COUNTED_ENTRY` unchanged |
| `tests/qa/test_qap_ms002.py` | Import block now takes the three promoted helpers from `harness`; its local duplicates deleted. No assertion in that file changed |
| `.sessions/.../iterations/070-forward-qa.md` | This report |
| `.sessions/.../qa-plan.md` | Test paths + `ready` status only |
| `.sessions/.../plan-summary.md`, `handoff-context.md` | Forward QA author-pass status |

## Harness Execution

Environment: `hyperipper.local:30001` reachable — the harness preflight (`require_upstream`)
answered `HTTP 200`; the gates therefore ran against the real remote upstream, not on a
DNS/connection-error baseline. The live service rejects the gate's invented model name with
`404 NotFoundError`; `assert_upstream_response` (promoted unchanged from `MS-002`) treats any
`2xx-4xx` as "the proxy reached its upstream and the fault is not ours", and **no gate asserts on
model output**. The dump path runs on request receipt, so every artifact assertion below is
exercised for real — visible in the observed numbers (`turn_000.json` carries the sent body,
`index.txt` carries one line per turn).

```
.venv/bin/python -m pytest tests/qa/test_qap_ms001.py -o addopts= -q
10 passed in 2.59s
```

Observed evidence, scenario by scenario:

| Scenario | Observed |
|---|---|
| `TC-100` | **passed** — `[TC-100] pre-flush snapshots=[(0.01, []), (0.02, []), (0.03, [])] turns=3 write_ops=1 write_bytes=1424`. Three acknowledged turns, three snapshots, each an empty/absent session directory; after the flush exactly one artifact, written by **one** write-bearing operation, containing all three turns. No documented-layout file on the collapsed path. |
| `TC-001` (documented write path) | **passed** — request answered with the upstream's status, identical to a healthy control session's status (the fault changed no response); `PermissionError` logged on stderr; the process kept serving and the control session captured. |
| `TC-001` (collapsed flush path) | **passed** — the buffered turn was acknowledged, nothing was on disk at append time, the injected read-only session dir made the *flush* raise, the raise was logged (`dump: flush failed for session … PermissionError`), and the process exited `rc == 0`. |
| `TC-107` | **passed** — no session dir before the first request; after 3 turns: exactly `turn_000.json`, `turn_001.diff`, `turn_002.diff`, `index.txt`; **zero** `.log` files anywhere under the dump root with nothing configured (witness W1 of `G-008` absent). Env scrubbed by an autouse fixture, so "no config override" is enforced, not assumed. |
| `TC-108` | **passed** — `[TC-108 documented half] N=3 format=documented write_ops=6 write_bytes=1047 files=4 expected_ops=2N=6 expected_files=N+1=4`. Per-path decomposition: `index.txt` 3 ops (one per turn), `turn_000.json` 1, `turn_001.diff` 1, `turn_002.diff` 1; every op inside the session dir. Contents: `turn_000.json` is the full body carrying the turn's text; each `turn_NNN.diff` is a unified diff (`---/+++/@@`) adding its own turn's content and removing the previous turn's; every `index.txt` line matches the README's own reader pattern `^turn [0-9]{3}  [0-9]{2}:[0-9]{2}` and keeps the documented `msgs`/`tools` columns, with `FULL`+`KB` on turn 000 and `+A/-R lines` on later turns. |
| `TC-108` sensitivity | **passed** — the matcher accepts all three README-published example lines and rejects five drift forms (lost `FULL`, unpadded turn number, lost timestamp, lost change column, lost tools column). The content assertions above can fail. |
| `H-R1` selector | **passed** — `documentedd`, `change_focused`, `DOCUMENTED`, `" documented"`, `""` each refused before serving (`rc != 0`), each naming the allowed set, and each leaving the `--dump-dir` root **uncreated**: no file written for a refused value. The closed value set does not merge with the default identity. |
| `H-R1` body parse | **passed** — non-UTF-8 and empty bodies are `4xx` with no traceback leaked, and **no** session directory was created for either; duplicate `model` keys land as last-wins in `turn_000.json` with no `m-first` corruption; a trailing-newline body is exactly one `index.txt` line, not two. |
| `H-R2` containment | **passed** — with `DUMP_DIR/<sidA>` a dangling symlink and `DUMP_DIR/<sidB>/index.txt` a symlink to an outside canary: nothing appeared at either outside target, neither symlink was replaced by a real path, the process kept serving, a healthy session still captured, `rc == 0`. This is R2's sibling audit of the writer `MS-002` did not cover, and the documented writer holds. |
| `R5` identity | **passed** — `…alpha-0001` and `…alpha-0002` produced two distinct session directories, each with its own `turn_000.json` + `turn_001.diff` + `index.txt` (interleaved turns, so per-session counters are proven, not a global counter), each carrying only its own markers; the pre-planted `change-focused.log` survived byte-identical — the documented writer neither migrated nor removed a file belonging to the other half of the `(sid, format)` identity. |

Regression proof for the harness promotion (the shared file is consumed by the other two passes):

```
.venv/bin/python -m pytest tests/qa/test_qap_ms004.py -o addopts= -q              → 6 passed in 44.41s
.venv/bin/python -m pytest tests/qa/test_qap_ms002.py::test_hostile_non_utf8_and_empty_body_no_write
                        tests/qa/test_qap_ms002.py::test_tc105_tc109_write_and_file_counts_paired_runs
                                                                                  → 2 passed in 1.06s
uv run pytest (hermetic default, network-free)                                    → 407 passed, 95.75%
```

`MS-002`'s paired measurement re-printed intact: `documented N=6 write_ops=12 files=7` against
`change-focused … write_ops=2 files=1` — the `G-008`(b) pair, with `TC-108` now supplying the
documented half at `MS-001` as well.

## Note for Yield Phase: the 022 directive, honored under changed conditions

`qa-plan.md` line 193 and `TC-100`'s trigger note both say **no `QAP-MS001` scenario may pass
`--dump-flush-interval` / `--dump-flush-bytes`**, and give the reason: *"those options do not exist
until `T003`"*, with 022's related premise that at `MS-001` **"no trigger exists at all"**. Both
premises are historical: `T003` is accepted, so the options exist and their defaults (10 s interval,
4 MiB volume — `README.md:183-184`) *can* fire during this milestone's own gate.

The directive is honored literally: `grep -- "--dump-flush" tests/qa/test_qap_ms001.py` matches
only the docstring that states the rule. No gate passes either flag.

What changed is how `TC-100` establishes "before any flush". It does not race the timer and it does
not sleep. Because the artifact is append-only and never removed, a snapshot that holds no artifact
**is** a pre-flush observation — the absence is a direct reading of disk state, and the test's
`≥1 pre-flush snapshot` clause fails loudly naming the reason if the environment is ever slow enough
that a flush beats the first request acknowledgement. The deferral's decisive, window-free witness
(no documented-layout file on the collapsed path, one artifact file, all buffered turns present in
it, every write op targeting that one file) is asserted from the final state and the instrumented
write report. If Yield Phase judges that a milestone gate should control its own flush cadence
instead — i.e. that 022's *reason* has expired along with its premise — that is a plan change for
this `QAP-*` handle, not a test edit to make here; the current test needs no relaxation either way.

## Result

**`READY-PASSING`**

`QAP-MS001` is authored and, as executed here, observes the approved contract on all four linked
validation points (`VP-T001-writes-deferred`, `VP-T001-dump-failure-isolated`,
`VP-T002-documented-format-preserved`, `VP-T002-default-encoding`) across `TC-100`, `TC-001`,
`TC-108`, `TC-107`, with the R1/R2 hostile tier and the R5 identity tier attached, and a
discriminating sensitivity check. `MS-004`'s `G-008`(b) documented-half numbers are reproducible at
this boundary too (`write_ops=2N`, `files=N+1`).

`READY-PASSING` is not approval. `MS-001` Yield Phase must independently inspect contract fidelity
and re-run `tests/qa/test_qap_ms001.py` against the live upstream before writing the authoritative
`pass`/`fail` on the four `VP-*` points, and must confirm the 022 directive reading above.

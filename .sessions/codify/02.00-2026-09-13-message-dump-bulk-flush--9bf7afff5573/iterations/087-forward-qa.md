# 087 — FORWARD QA (author pass, re-entry on current bytes): QAP-MS001

## Orchestrator Prompt (verbatim)

```text
Execute the forward-qa-interactive skill for this session. Arguments: $ARGUMENTS
# Your Model
- Name: sloppus (shared alias)
- Provider ID: hyperipper
- Model ID: sloppus

---

You are a subagent spawned by another session.
Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
Milestone: MS-001
Mode: re-entry on current bytes — the gate tests already exist and are approved; this pass re-proves them on the bytes the CR3 wave and T019.001 landed
```

## QA Pass

- **QA pass**: `QAP-MS001` — resolved from the dispatch's `Milestone: MS-001` against
  `qa-plan.md` §QA Author Passes, where `MS-001` has exactly one author pass (`QAP-MS001`).
  Not ambiguous, not orphaned, scheduled after all constituent tasks pass local Yield and before
  Yield Phase. Not `BLOCKED-PLAN` on identity/scheduling.
- **Session**: `02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573` — verified against
  `session-identity.json` (`session_id` `02.00-2026-09-13-message-dump-bulk-flush`,
  `session_uid` `9bf7afff-5573-402a-b102-827423a1dded` → handle suffix `9bf7afff5573`).
  Handle matches; not `BLOCKED-SESSION-IDENTITY`.
- **Validation points in scope (only these)**: `VP-T001-writes-deferred`,
  `VP-T001-dump-failure-isolated`, `VP-T002-documented-format-preserved`,
  `VP-T002-default-encoding`
- **Scenarios in scope (only these)**: `TC-100`, `TC-001`, `TC-107`, `TC-108`
- **Gate handles**: `tests/qa/test_qap_ms001.py` (authored at
  [070-forward-qa](070-forward-qa.md); SEMANTIC-APPROVED at
  [071-yield-phase](071-yield-phase.md); re-issued on current bytes at
  [078-yield-phase](078-yield-phase.md); last re-entered at
  [077-forward-qa](077-forward-qa.md)), shared fixture location `tests/qa/harness.py`,
  instrumented write-count entry `tests/qa/counted_proxy_entry.py`.
- **Repository state under test**: `8dfa5afad50045a0a5b0085aacbc706f376bac27` (Checkpoint 201),
  branch `refactor/package-structure`, clean worktree at pass start.

### Why this pass is a re-entry and not a first authoring

The four gates exist, are approved, and were green at 077 on Checkpoint 183. Since then the
**CR3 wave** (`T018`+`T019`+`T020`, accepted-local at
[083-01-yield](083-01-yield.md)) and the run-45 remediation (`T019.001`, accepted-local at
[085-01-yield](085-01-yield.md)) landed production bytes on the very path these gates trigger
through. File-level census (`git diff --stat 6508941..HEAD -- buster_ripper/ tests/ README.md
pyproject.toml`), taken as boundary metadata and not as implementation reading:

| Path | Moved since 077? | Relevance to this pass |
|---|---|---|
| `buster_ripper/server.py` | yes | the `web.Application` construction — `TC-100`/`TC-107`/`TC-108`/`TC-001` all start the real binary |
| `buster_ripper/routes/messages.py` | yes | the interception path every scenario here drives, its caller-side dump swallow, and its client-error arms |
| `buster_ripper/routes/chat.py`, `routes/count_tokens.py` | yes | sibling route modules — no scenario in this pass drives them |
| `buster_ripper/dump.py`, `config.py`, `main.py`, `session.py`, `utils.py` | **no** | the writer, the selector's default, the CLI surface, the session key |
| `tests/qa/**` | **no** | `git log -1 -- tests/qa/` → `6508941` (077's own checkpoint): the gate bytes are byte-identical to the approved set |
| `README.md`, `pyproject.toml` | **no** | the public contract and the exclusion of `tests/qa/` from the hermetic run |

So: **nothing in this pass's contract was re-authored, and nothing in the gate bytes changed.**
The work this pass owes is Required Check 8 — re-prove the whole gate executable and observing
the contract on the new bytes — plus the two consequences of that specific delta:

1. `TC-001`'s approved visible result ("the upstream response body is returned unchanged") is
   stated through a request path whose error arms the CR3 wave rewrote. The gate must still
   witness the same status equality and the same logged raise.
2. The hostile R1 tier posts unparsable bodies through the request-body parse boundary the wave
   also changed (decode totality, and the application's body-size cap). Absence-of-write and
   typed-client-error are both assertions that a transport-layer change can move.

### Preconditions checked

- All `MS-001` constituent tasks accepted by local Yield — `T001` (accepted 2026-09-22,
  Checkpoint 61; `plan-summary.md:187`), `T002` (accepted 2026-09-22,
  [031-02-yield](031-02-yield.md); `plan-completed.md:8`), `T005` (accepted 2026-09-28,
  [049-01-yield](049-01-yield.md), archived Checkpoint 109; `plan-completed.md:10`).
  Not `BLOCKED-PLAN — milestone incomplete`.
- `qa-plan.md` present; `.agents/skills/qa-project/SKILL.md` present → not `BLOCKED-HARNESS`.
- Quiescence contract and deterministic mode both **declared and used** (`wait_tcp` readiness,
  `poll_until` / process-exit quiescence, fresh `--dump-dir` per gate, unique system prompts,
  zero timed sleeps in the gate) → no Capability Gate miss to report.
- `022` directive re-checked by grep rather than memory: `grep -n "dump-flush"
  tests/qa/test_qap_ms001.py` → **line 27 only, the docstring stating the rule**; the three
  `extra_args=` sites (172, 268, 435) carry `--dump-format` and nothing else. No gate in this
  pass passes `--dump-flush-interval` / `--dump-flush-bytes`.
- Declared live upstream reachable at pass start: `GET http://hyperipper.local:30001/v1/models`
  → HTTP 200 (no ambient proxy).
- **Backlog drain check**: every open `quality-backlog.md` row was searched for this pass or
  `tests/qa/` as its named trigger. The two `forward-qa`-owned rows that named this pass
  (`QB:qa-harness/unexercised-path-keyword`, `QB:qa-harness/static-body-cannot-connect-limb`)
  are **resolved at 077** — that trigger has fired and there is no second debt in the lane.
  No open row routes to `forward-qa`. The two `tests/qa`-adjacent open rows belong to other
  owners (`QB:test-tls-posture/private-attr-pinning` → `roadmap.md` §RM-007;
  `QB:qa-plan/tc-tls-lifecycle-clause` → Authority Registrar) and are not this pass's to take.

## Contract Draft

Stage 1, re-derived before any implementation byte was read, from the approved records only:
`qa-plan.md` §Scenario Contracts / §Phase Requirements, `plan-summary.md` §Validation Point
Matrix rows for the four `VP-*`, the `qa-project` harness declaration, the public user-facing
contract `README.md` §Dump mode / §Collapsed dump formats / §Reading the index /
§Pre-existing session directories, and the existing QA fixtures (`tests/qa/harness.py`,
`tests/qa/counted_proxy_entry.py`, `tests/qa/test_qap_ms001.py`).

| Scenario | Real trigger | Observable assertions | Forbidden assertions | Lifecycle proof | Telemetry proof | Harness entry |
|---|---|---|---|---|---|---|
| **TC-100** `VP-T001-writes-deferred` | Start the real proxy with `--dump-dir <fresh tmp> --dump-format change-focused` and **no flush parameter**; POST `N=3` `POST /v1/messages` turns; then SIGTERM | At every turn acknowledged while **no artifact exists anywhere under the dump root**, that session dir's listing is empty; post-run, exactly one `change-focused.log` under `DUMP_DIR/<sid>/` holding all three buffered turns; every write-bearing operation in the instrumented report targets that one file | Never the *absence of the session directory* as the assertion (the VP names that a forward detail); never model output; never a flush parameter used to manufacture the pre-flush window | `rc == 0` across all turns; the deferral demonstrably ends at a flush — the same directory is non-empty post-run, so the emptiness above is not a dead-recorder reading | none — the project declares no telemetry-verification capability | `Proxy(..., extra_args=["--dump-format", COLLAPSED], counted_report=…)` |
| **TC-001** `VP-T001-dump-failure-isolated` | (a) documented path, session dir made unwritable (`chmod 0o500`) **before** start, one faulted turn plus a healthy control session; (b) collapsed path, same fault, the raise surfacing in the **shutdown** flush | The proxied request still answers with what the peer returned — the status class is identical between the faulted session and a healthy control, and `< 500`; `PermissionError` appears on the proxy's own stderr; a healthy sibling session still captures after the fault | Never the fault message's *wording*; never a 5xx attributable to dump behavior; never a body substring as the connectivity oracle (077's drain made the helper a pure status predicate and the approved assertion says "the upstream response body", not its content) | The process keeps serving (`proc.poll() is None`) and exits `rc == 0` — including when the raise happens inside `on_cleanup`'s flush | none | `Proxy(...)` with a pre-created unwritable `DUMP_DIR/<sid>/` |
| **TC-107** `VP-T002-default-encoding` | Start with **no `--dump-format`** and no `DUMP*` env override (autouse fixture clears them); POST `N=3` turns | `DUMP_DIR/<sid>/` holds exactly `turn_000.json`, `turn_001.diff`, `turn_002.diff`, `index.txt` (`N + 1`) with README §Dump mode's contents and §Reading the index's line format; no `*.log` anywhere under the dump root | Must NOT assert the collapsed layout — the R2 candidate clause is recorded forbidden to author (`US-012`, `G-008` witness W1) | `rc == 0`; layout unchanged from today; sensitivity: no session dir before any request | none | `Proxy(dump, logs)` — defaults only |
| **TC-108** `VP-T002-documented-format-preserved` | Same run driven with `--dump-format documented` through the **instrumented** entry; `N=3` turns, no flush parameter | Layout `N + 1` files; `turn_000.json` parses as the full normalized body and carries the turn's content; every `index.txt` line matches README `:205`'s own published reader pattern `^turn [0-9]{3}  [0-9]{2}:[0-9]{2}`, carries its turn number, `N msgs`, `N tools`, plus `FULL`/`KB` on turn 000 and `+A/-R lines` on later turns; each `turn_NNN.diff` is a unified diff (`---`/`+++`/`@@`) adding its own turn and removing the previous one; **write-bearing ops `= 2N`**, decomposed per path (`index.txt` ×N, each turn file ×1); all ops inside this session's dir | Never a drifted `index.txt` line format, a missing/renamed documented file, a `.log` on this path, or a count that is not `2N` | `rc == 0`; counts are per session | none | `Proxy(..., counted_report=…)` via `harness.COUNTED_ENTRY` |

**The tiers this pass also carries** (Required Checks 5-7, established at 070 and unchanged in
bytes): R1 hostile ×2 — unknown selector values refused at startup with the allowed set named
and **no write into the dump root**, and the request-body parse boundary on the documented
writer (non-UTF-8, empty, duplicate JSON keys, trailing newline); R2 — the **documented**
writer's symlink containment into pre-existing trees (dangling session-dir symlink, symlinked
`index.txt`), asserting typed no-write containment rather than the happy path; R5 — the
near-miss `sid` pair that the session key must not merge, plus the `(sid, format)` identity
against a pre-existing collapsed artifact.

### Fidelity questions the current bytes raise (to be answered by execution, not by reading)

1. `TC-001`: does a raising dump write still leave the client-facing answer the peer's, after the
   wave rewrote the route's error arms? The gate's witness is the status-equality limb — a
   regression that turned the dump fault into a 5xx shows up there, not in prose.
2. R1 parse boundary: `non-UTF-8` / `empty` bodies must still be typed client errors **and**
   must still create no session directory. Both are properties of the arm the wave's
   decode-totality work sits on.
3. `TC-108`: `2N` write-bearing ops and the per-path decomposition are the documented half of
   the `G-008`(b) pair. 070 recorded `write_ops=6`, `files=4`; 077 re-measured the same. This
   pass re-measures against the wave's bytes and reports the figure rather than inheriting it.
4. `TC-100`: `turns=3 write_ops=1 write_bytes=1424` at 077 — one bulk write for three buffered
   turns. A byte-count change here is expected drift (the marker text differs per run), a
   **write-count** change is a contract event.

## Boundary Inspection Exceptions

### Boundary Inspection Exception
- **Need**: `T018` deleted the application's 1 MiB request-body cap, so an input the documented
  writer never used to see now reaches it. Before deciding whether this pass owes a new hostile
  variant at that parse boundary, I had to know whether the property already has an owner at a
  cheaper tier — `codify-test-cost` rule 3 forbids a parallel gate, and the skill forbids widening
  an approved scenario.
- **Path/symbol**: the owner search only — `grep -rn "client_max_size" buster_ripper/ tests/`
  (surfacing `server.py:119`/`:125` and their comment) and
  `tests/test_transport_parity.py::TestNoRequestBodyCap` (class + witness names and the
  `/v1/messages` over-cap case at `:104-152`).
- **Question**: Is "a request body over the former cap reaches the proxy's read path instead of
  being answered for its size" already witnessed somewhere, and on which surface?
- **Assertion impact**: none; approved contract remains unchanged. No assertion in
  `tests/qa/test_qap_ms001.py` was derived from, weakened by, or added because of this reading.

**Answer obtained**: yes, owned at the hermetic tier by
`tests/test_transport_parity.py::TestNoRequestBodyCap` — `test_a_two_mebibyte_body_reaches_the_upstream`
(parametrized over the route classes) and `test_a_strip_images_body_over_the_cap_reaches_the_image_handler`
assert `response.status == 200` plus "the read actually happened". This pass therefore authors
nothing at that boundary and records the owner below.

## Files Changed

| File | Change |
|---|---|
| `.sessions/…/iterations/087-forward-qa.md` | This report — the re-entry's contract draft, the byte-delta census, the execution record, and the dispositions |
| `.sessions/…/qa-plan.md` | `QAP-MS001` §Authored Gate Tests: `ready` status retained, this re-entry recorded (gate bytes unchanged, re-proved on the CR3 + `T019.001` bytes); no test path added, none removed |
| `.sessions/…/plan-summary.md` | §Forward QA status row: the `QAP-MS001` re-entry recorded on current bytes |
| `.sessions/…/handoff-context.md` | §Current Status: one bullet for this pass, so the MS-001 close does not have to rediscover that the gate was re-proven after the wave |

**Zero test bytes and zero production bytes changed.** `git diff --stat` at the close of this
pass names only the four session artifacts above. `tests/qa/test_qap_ms001.py`,
`tests/qa/harness.py` and `tests/qa/counted_proxy_entry.py` are byte-identical to the approved set
at Checkpoint 183 (`git log -1 -- tests/qa/` → `6508941`). No production source, no
implementation unit test, no `VP-*`/`TC-*` semantics, no invariant record, and no implementation
Forward report was touched — and none needed to be, because nothing failed.

## Harness Execution

`qa-project` invocation form (the `tests/qa/` gates are excluded from the hermetic default run by
`norecursedirs` — `pyproject.toml:23` names `qa`):

```
.venv/bin/python -m pytest tests/qa/test_qap_ms001.py -o addopts=
```

Environment: `HEAD = 8dfa5af` (Checkpoint 201), clean worktree at start and at close;
`.venv/bin/python` 3.13.8; `hyperipper.local:30001` → HTTP 200 before the pass and the harness
`require_upstream()` preflight passes inside every gate.

| Run | Result |
|---|---|
| Cheap tier first — the two process-free sensitivity witnesses (070's line-pattern drift witness, 077's upstream-oracle witness) | **2 passed** in 0.03 s |
| **`QAP-MS001` full gate** (11 tests: `TC-100`, `TC-001` ×2, `TC-107`, `TC-108` + sensitivity, oracle sensitivity, R1 ×2, R2, R5) | **11 passed** in 2.26 s |
| `QAP-MS002` — shared-harness consumer on the same interception path and writer (regression witness only, not this pass's verdict) | **15 passed** in 14.65 s |
| `QAP-MS004` — shared-harness consumer against its own loopback peer (regression witness only) | **6 passed** in 44.34 s |
| Hermetic default `uv run pytest` | **476 passed @ 95.91%**, coverage floor 80% reached — the figures `085-01-yield` recorded at `T019.001`'s acceptance, unchanged |
| Combined process-level tier | **32/32 gates green** (11 + 15 + 6) |

Scenario diagnostics re-observed on the current bytes, against 077's record:

| Gate | 077 on Checkpoint 183 | 087 on Checkpoint 201 | Delta |
|---|---|---|---|
| `TC-100` | `pre-flush snapshots=[(0.01, []), (0.02, []), (0.03, [])]`, `turns=3 write_ops=1 write_bytes=1424` | `pre-flush snapshots=[(0.01, []), (0.02, []), (0.02, [])]`, `turns=3 write_ops=1 write_bytes=1424` | none — three acknowledged turns each witnessed an empty session dir, one bulk write for all three |
| `TC-108` (documented half of `G-008`(b)) | `N=3 write_ops=6 write_bytes=1047 files=4 (2N=6, N+1=4)` | `N=3 write_ops=6 write_bytes=1047 files=4 (2N=6, N+1=4)` | none — **071's and 078's approved figures still describe the tree after the CR3 wave and `T019.001`** |
| `TC-107` | documented layout, no `.log` | documented layout, no `.log` | none |
| `TC-001` ×2 | same-status faulted vs control, `PermissionError` logged, `rc == 0` incl. the raising shutdown flush | same | none |
| R1 ×2 / R2 / R5 | refusals name the allowed set with no write; unparsable bodies are typed client errors with no session dir; symlinked targets untouched, neither symlink replaced; near-miss `sid`s separate with a foreign `.log` preserved | same | none |

`022` re-checked mechanically rather than from memory: `grep -n "dump-flush"
tests/qa/test_qap_ms001.py` → the docstring at `:27` stating the rule; the three `extra_args=`
sites (`:172`, `:268`, `:435`) pass `--dump-format` only. No gate in this pass passes a flush
parameter, and `TC-100` still bounds its pre-flush window by observation, as ruled at
[071-yield-phase](071-yield-phase.md) §Check 6.

### Intermediate / helper disposition (Yield Phase Check 6)

Nothing new is deletable from this pass and nothing was promoted, duplicated, or subsumed: no
helper moved, no intermediate QA test exists to absorb, and the three `tests/qa/` gate modules
still share the single definitions of `read_text`, `log_artifacts`, `assert_upstream_response`,
`startup_refused`, `Proxy` and `counted_proxy_entry.py` that 070 promoted and 077 drained. The
only cross-lane note for Check 6 is the ownership record below — an observation, not a deletion
candidate.

### Dispositions this pass closes without a test edit

1. **The oversize-body boundary `T018` moved.** `client_max_size=0` widened the input domain that
   `TC-107`/`TC-108`'s documented writer sits behind: pre-wave, a body over 1 MiB was answered by
   aiohttp before `normalize()` and before `--strip-images` saw it; now it is read. Nothing in
   `VP-T002-documented-format-preserved`, `VP-T002-default-encoding`,
   `VP-T001-writes-deferred` or `VP-T001-dump-failure-isolated` speaks to body size — their
   contracts are layout, content, line format, write count, deferral, and fault isolation, all
   re-proven above. Authoring a QA-tier over-cap gate here would (a) widen an approved scenario,
   which this role may not do, and (b) duplicate the hermetic witness that already owns it:
   `tests/test_transport_parity.py::TestNoRequestBodyCap` (`test_a_two_mebibyte_body_reaches_the_upstream`,
   `test_a_strip_images_body_over_the_cap_reaches_the_image_handler`), which asserts both the
   status and that the read happened. Disposition: **owned elsewhere, nothing owed here**; if the
   MS-001 boundary later wants a *process-level* witness that a >1 MiB body lands in
   `turn_000.json` intact, that is a new `VP-*` for the plan, not a test edit by this lane.
2. **No backlog row was owed to this pass.** The two `forward-qa`-owned `tests/qa/harness.py` rows
   named 077 as their drain and are resolved there; re-touching `harness.py` to satisfy nothing
   would be churn against a shared interface every gate imports. Recorded so the close does not
   read this pass as a partial drain.
3. **`tests/qa/`'s bytes did not move, so 071's and 078's semantic approval still addresses the
   same test source.** What changed under it is the application, and that is exactly what
   Required Check 8 is for.

## Result

**`READY-PASSING`**

- All four linked scenarios — `TC-100`, `TC-001` (both the documented write path and the collapsed
  flush path), `TC-107`, `TC-108` — are gated by persistent tests under `tests/qa/`, driven through
  the real proxy subprocess against the declared live upstream, and the module is green on the
  current bytes: **11/11 in 2.26 s** at `HEAD = 8dfa5af`, the CR3 wave (`T018`+`T019`+`T020`) and
  `T019.001` landed.
- The documented half of the `G-008`(b) comparison re-measures identically to the approved record
  (`write_ops = 2N = 6`, `files = N + 1 = 4`), and `TC-100` still witnesses one bulk write for three
  buffered turns with an empty session directory at each acknowledged turn — so the numbers
  `071-yield-phase` and `078-yield-phase` ruled on still describe this tree.
- The shared fixture location is proven still connected for its other two consumers on the same
  bytes (`MS-002` 15/15, `MS-004` 6/6), and the hermetic default run is unmoved (476 @ 95.91%).
- Nothing failed an authored test, so no production behavior is being reported as broken; no
  scenario's approved contract was weakened, reinterpreted, or widened; no gate passes a flush
  parameter. Nothing is `BLOCKED`: `qa-plan.md`, `qa-project`, the quiescence contract and the
  deterministic mode are all declared and used, so there is no Capability Gate miss to report.
- **This is not an approval.** `READY-PASSING` records that the persistent test observes the
  contract on these bytes. Yield Phase independently inspects contract fidelity and re-runs for the
  authoritative `pass`/`fail` on `VP-T001-writes-deferred`, `VP-T001-dump-failure-isolated`,
  `VP-T002-documented-format-preserved` and `VP-T002-default-encoding`.

### For Yield Phase's attention

1. **The re-proof is a re-run, not a re-authoring.** If Check 1 wants the gate's independence
   demonstrated rather than asserted, the discriminating witnesses are the two process-free
   sensitivity tests: the line-pattern witness rejects five named drift forms, and the
   upstream-oracle witness refuses `500`/`502`/`503` and accepts the peer's own `400`/`404`/`429`
   — it goes red the moment the dropped `"Cannot connect"` limb is restored, which is the shape a
   later harness edit would take by accident.
2. **A `READY-PASSING` on unchanged test bytes transfers responsibility to the application.**
   Because `tests/qa/` is identical to 077's and the production path is not, a future divergence at
   this boundary is a production event, and 071's SEMANTIC-APPROVAL should be read as re-issued on
   `8dfa5af`-class bytes only if Yield Phase re-runs it here.
3. **`TC-001`'s oracle is a pure status predicate** (077's drain, still in force). Its
   anti-vacuity rests on `require_upstream()`'s fail-closed preflight and on the artifact-content
   assertions. If the boundary wants a gate-tier witness that the preflight itself is load-bearing,
   that is a new claim outside `qa-plan.md` and belongs to the plan, not to a test edit in this lane.
4. **One input-domain change has no scenario here by design** — the over-cap body, dispositioned
   above with its cheaper-tier owner named. Flagged so the phase record shows it was considered and
   routed, rather than overlooked at the milestone whose whole subject is the dump writer.

## Checkpoint

**Checkpoint 202** (`fd1d9d2`) carries exactly the four files this pass wrote — this report,
`qa-plan.md`, `plan-summary.md`, `handoff-context.md` — and nothing else: `git show --stat` lists
4 files, `+317/−1`. The single deleted line is `plan-summary.md`'s Forward QA status row, extended
in place. `tests/qa/` and `buster_ripper/` are absent from it, which is the pass's claim stated as
a diff. The worktree is clean at close; the checkpoint is not compacted — squashing is the
reviewer's act, and 077's two-checkpoint precedent (a completing follow-up rather than an amend)
is the convention this section follows by being committed as Checkpoint 203.

One precision about the runs above: 202 and 203 stage session artifacts only, so the code bytes at
`d9632cb` are the code bytes at `8dfa5af` — the table's "at close" re-run of the gate (**11 passed in
2.52 s**) is a re-run on identical application source, not a re-run against a moved tree.

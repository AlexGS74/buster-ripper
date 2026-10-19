# 077 — FORWARD QA (author pass, re-entry): QAP-MS001

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
QA Pass: QAP-MS001
```

## QA Pass

- **QA pass**: `QAP-MS001` (`qa-plan.md` §QA Author Passes)
- **Session**: `02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573` — verified against
  `session-identity.json` (`session_id` `02.00-2026-09-13-message-dump-bulk-flush`,
  `session_uid` `9bf7afff-5573-402a-b102-827423a1dded`). Handle matches; not
  `BLOCKED-SESSION-IDENTITY`.
- **Validation points in scope (only these)**: `VP-T001-writes-deferred`,
  `VP-T001-dump-failure-isolated`, `VP-T002-documented-format-preserved`,
  `VP-T002-default-encoding`
- **Scenarios in scope (only these)**: `TC-100`, `TC-001`, `TC-107`, `TC-108`
- **Gate handles**: `tests/qa/test_qap_ms001.py` (existing gate, authored at
  [070-forward-qa](070-forward-qa.md), SEMANTIC-APPROVED at
  [071-yield-phase](071-yield-phase.md)); shared fixture location `tests/qa/harness.py`;
  instrumented write-count entry `tests/qa/counted_proxy_entry.py`.

### Why this pass is a re-entry and not a first authoring

`QAP-MS001` was authored at 070 and its gate was SEMANTIC-APPROVED at 071. This dispatch
re-enters the same pass because **two open `quality-backlog.md` rows name this pass as their
own trigger and name `forward-qa` as their owning route**:

| Row | File | Recorded trigger |
|---|---|---|
| `QB:qa-harness/unexercised-path-keyword` | `tests/qa/harness.py:209-212` | "next touch of `tests/qa/harness.py`, **the owed `QAP-MS001` author pass being the next such touch**" |
| `QB:qa-harness/static-body-cannot-connect-limb` | `tests/qa/harness.py:135` | "next touch of `tests/qa/harness.py` — the row `QB:qa-harness/unexercised-path-keyword` names the same file and the same trigger, **so drain them together**" |

070 did touch `harness.py` (the helper promotion) but drained neither row; 073-01-yield then
re-routed the second row to this lane with the joint-drain instruction. Both are inside this
pass's write boundary (QA test artifacts in the location `qa-project` authorizes, shared fixture
location) and outside any production surface.

**No new scenario gate is authored.** Re-authoring `TC-100`/`TC-001`/`TC-107`/`TC-108` gates that
already exist and are approved would duplicate coverage (`codify-test-cost` rule 3) and collide with
the approved verdict. This pass (a) drains the two owed rows in the shared harness, (b) adds the
one sensitivity witness the drain requires, and (c) re-proves the whole gate is executable against
current bytes (Required Check 8), since every gated file moved after 070.

### Preconditions checked

- All `MS-001` constituent tasks accepted by local Yield: `T001` (accepted 2026-09-22, Checkpoint
  61), `T002` (accepted 2026-09-22, 031-02-yield), `T005` (accepted 2026-10-04, Checkpoint 109) —
  `plan-summary.md` §Task Ledger archive notes. Not `BLOCKED-PLAN — milestone incomplete`.
- `qa-plan.md` present; `.agents/skills/qa-project/SKILL.md` present → not `BLOCKED-HARNESS`.
- `022` directive honored as ruled at 071 §Check 6: **no gate in this pass passes
  `--dump-flush-interval` or `--dump-flush-bytes`.**
- Declared live upstream `http://hyperipper.local:30001` reachable at pass start (`GET /v1/models`
  → HTTP 200), so the gate can be executed rather than reported blocked.

## Contract Draft

Stage 1, authored from the approved records (`qa-plan.md` §Scenario Contracts,
`plan-summary.md` §Validation Point Matrix), the `qa-project` harness declaration, the public
user-facing contract `README.md` §Dump mode / §Collapsed dump formats / §Pre-existing session
directories, and the existing QA fixtures only. No implementation report, diff, source body, or
implementation-authored unit test was read before this table.

| Scenario | Real trigger | Observable assertions | Forbidden assertions | Lifecycle proof | Telemetry proof | Harness entry |
|---|---|---|---|---|---|---|
| **TC-100** `VP-T001-writes-deferred` | Start the real proxy with `--dump-dir <fresh tmp> --dump-format change-focused` and **no flush parameter**; POST `N=3` `/v1/messages` turns; SIGTERM | At every turn acknowledged **while no artifact exists anywhere under the dump root**, that session dir's listing is empty; after the run, exactly one `change-focused.log` under `DUMP_DIR/<sid>/` holding all three buffered turns; the instrumented report names only that one file | Never assert the session *directory* is absent (the VP explicitly calls that a forward detail); never assert on model output | `rc == 0` across all turns; the deferral ends at the first flush, so the same directory is non-empty post-run | none required — project declares no telemetry capability | `Proxy(..., extra_args=["--dump-format", COLLAPSED], counted_report=…)` |
| **TC-001** `VP-T001-dump-failure-isolated` | (a) documented path with the session directory made unwritable (`chmod 0o500`) before start, one faulted turn + a healthy control session; (b) collapsed path with the same fault, flushed by a real SIGTERM | The proxied request still answers with the upstream's own response (status class identical between the faulted session and a healthy control, `< 500`); `PermissionError` appears on the proxy's own stderr; a healthy sibling session still captures after the fault | Never assert on the fault message's *wording*; never accept a 5xx attributable to dump behavior | Process keeps serving (`proc.poll() is None`) and exits `rc == 0` — including when the raise happens inside the shutdown flush | none | `Proxy(...)` with a pre-created unwritable `DUMP_DIR/<sid>/` |
| **TC-107** `VP-T002-default-encoding` | Start with **no `--dump-format`** and no `DUMP*` env override; POST `N=3` turns | `DUMP_DIR/<sid>/` holds exactly `turn_000.json`, `turn_001.diff`, `turn_002.diff`, `index.txt` (`N + 1` files) with the README's documented contents and line format; no `*.log` anywhere | Must NOT assert the collapsed layout — the R2 candidate clause is recorded as forbidden to author (`US-012`, `G-008` witness W1) | `rc == 0`; layout unchanged from today | none | `Proxy(dump, logs)` — defaults only |
| **TC-108** `VP-T002-documented-format-preserved` | Same run driven with `--dump-format documented` through the **instrumented** entry; `N=3` turns, no flush parameter | Layout `N + 1` files; `turn_000.json` parses as the full normalized body and carries the turn's content; every `index.txt` line matches `README.md:205`'s own published reader pattern `^turn [0-9]{3}  [0-9]{2}:[0-9]{2}`, carries its turn number, `N msgs`, `N tools`, plus `FULL`/`KB` on turn 000 and `+A/-R lines` on later turns; each `turn_NNN.diff` is a unified diff (`---`/`+++`/`@@`) adding its own turn and removing the previous one; **write-bearing ops `= 2N`** decomposed per path (`index.txt` ×N, each turn file ×1); all ops inside this session's dir | Never a drifted `index.txt` line format, a missing/renamed documented file, a `.log` on this path, or a count that is not `2N` | `rc == 0`; counts are per session | none | `Proxy(..., counted_report=…)` via `harness.COUNTED_ENTRY` |

**Contract fidelity for the two harness rows being drained.** `assert_upstream_response` is the
helper through which every scenario above states "the request reached the peer". The approved
visible results are `the upstream response body is returned unchanged` (TC-001), and for
TC-100/TC-107/TC-108 the real assertion is on the artifact and the write count — the response
check exists only to stop a gate passing against a dead transport. So the helper's contract is:
**distinguish a proxy-local transport failure from an answer that came from the peer.** Nothing in
the four scenarios asks the helper to inspect the *content* of a legitimately relayed answer. That
is the seam the `static-body-cannot-connect-limb` row says the current body test no longer sits on.

**Sensitivity (Required Check 4) for the helper change**: the drain is only witnessed if a cheap,
in-process check can show (a) the surviving limb still refuses a proxy-local transport failure and
accepts the peer's own client-error answer, and (b) the removed limb's false-positive shape — a
2xx answer whose body legitimately says `Cannot connect` — now passes. With the limb restored the
second condition is red, which names the code change that turns the new test red
(`codify-test-cost` rule 1).

## Boundary Inspection Exceptions

### Boundary Inspection Exception
- **Need**: The public contracts name the peer's answer only as "the upstream response body". To
  drop `assert_upstream_response`'s body limb I must know that a proxy-local transport failure is
  still refused by the *surviving* limb — otherwise the drain could let a dead-transport gate pass
  vacuously, which is the exact property the limb was written to protect. `qa-project` does not
  state the proxy's transport-failure status class, and no QA fixture records it.
- **Path/symbol**: `buster_ripper/routes/messages.py` — the `ClientError` handler arm of
  `handle_messages` (its response status only).
- **Question**: When the proxy cannot reach its upstream, does the client-facing answer carry a
  status `>= 500`?
- **Assertion impact**: none; approved contract remains unchanged.

**Answer obtained** (`messages.py:165-167`, four lines read): the `aiohttp.ClientError` arm returns
`web.json_response({"error": "Upstream request failed"}, status=502)`. So a proxy-local transport
failure is answered `>= 500` and the surviving `200 <= status < 500` limb refuses it — the drain
cannot let a dead-transport gate pass vacuously. Read: that arm only. No control flow was mirrored
into a test, and no expected outcome was derived from the implementation.

## Files Changed

| File | Change |
|---|---|
| `tests/qa/harness.py` | `QB:qa-harness/unexercised-path-keyword` drained: `Proxy.url(path="/v1/messages")` → `Proxy.url()`, `Proxy.post(body, timeout, path="/v1/messages")` → `Proxy.post(body, timeout=90.0)`, restored to the pre-055 signatures (`git show f803b04:tests/qa/harness.py` §126-132); `post`'s internal call site now `self.url()`. `QB:qa-harness/static-body-cannot-connect-limb` drained: `assert_upstream_response`'s `b"Cannot connect" not in body` limb dropped, and its docstring rewritten to state what the helper proves (status class is the whole oracle), why the body is deliberately not inspected (a relayed body is the peer's bytes, so no substring is attributable to the proxy), and which owner the leak property now names (`TestLeakFreeErrorBodies`) |
| `tests/qa/test_qap_ms001.py` | Module docstring records the re-entry and its scope; one process-free test added: `test_sensitivity_upstream_oracle_distinguishes_transport_failure_from_a_relayed_body`. **No existing gate's assertions were altered** — `TC-100`, `TC-001` ×2, `TC-107`, `TC-108` + its pattern sensitivity check, the R1 hostile tier, R2 containment and the R5 near-miss are byte-identical to 070's |
| `.sessions/…/quality-backlog.md` | Both `QB:qa-harness/*` rows marked resolved with the remedy taken, the rejected alternative and its rule, the evidence re-measured, and the regression runs |
| `.sessions/…/iterations/055-forward-qa.md` | Appended dated correction to its §Files Changed claim about the two `path=` keywords. Original text left verbatim (this session's report-accuracy convention: correct in a follow-up checkpoint, never rewrite a committed record) |
| `.sessions/…/qa-plan.md` | `QAP-MS001` §Authored Gate Tests entry: `ready` status retained, re-entry recorded, new test path named |
| `.sessions/…/iterations/077-forward-qa.md` | This report |

**Not changed, and out of this pass's boundary**: production source, implementation unit tests,
`VP-*`/`TC-*` semantics, invariant records, implementation Forward reports. `git diff --stat` at the
close of authoring is two files, both under `tests/qa/`.

## Harness Execution

`qa-project` invocation form (gates are excluded from the hermetic default run by
`norecursedirs` — `pyproject.toml:23` names `qa`):

```
.venv/bin/python -m pytest tests/qa/test_qap_ms001.py -o addopts=
```

Environment at run time: `hyperipper.local:30001` reachable (`GET /v1/models` → HTTP 200 before the
pass started, and the harness `require_upstream()` preflight passes inside every gate).

| Run | Result |
|---|---|
| Cheap tier first — the new sensitivity witness + 070's pattern witness | **2 passed** in 0.03 s (in-process, no subprocess) |
| Fail-first proof of the new witness — limb temporarily restored, then reverted | **1 failed** (`AssertionError: proxy could not connect to its upstream` at `harness.py:149`) → restored → **1 passed**. Tree returned to the intended two-file diff, verified by `git diff --stat` |
| `QAP-MS001` full gate (11 tests) | **11 passed** in 2.5 s |
| `QAP-MS002` (23 lines reference the changed helper; 22 call sites) | **15 passed** in 14.7 s |
| `QAP-MS004` (consumes `Proxy.post` / `Proxy.url` against its own peer) | **6 passed** in 44.2 s |
| Hermetic default `uv run pytest` | **455 passed @ 95.75%**, coverage floor 80% reached — unchanged from `T017`'s accepted figures, as expected since `tests/qa/` is excluded from that run |
| Combined process-level tier | **32/32 gates green** |

Scenario diagnostics re-observed on current bytes:

- `TC-100`: `pre-flush snapshots=[(0.01, []), (0.02, []), (0.03, [])]` — an empty listing witnessed
  at all three acknowledged turns — `turns=3 write_ops=1 write_bytes=1424`. One bulk write for three
  buffered turns; the deferral is real, not a timing artifact.
- `TC-108`: `N=3 format=documented write_ops=6 write_bytes=1047 files=4 expected_ops=2N=6
  expected_files=N+1=4`. **The documented half of the `G-008`(b) pair re-measures identically to
  070's record**, so 071's approved numbers still describe the tree after `T013`–`T017` landed.
- The `022` directive re-checked by grep rather than memory: `grep -n "dump-flush"
  tests/qa/test_qap_ms001.py` → the docstring stating the rule; nothing in any `extra_args`.

### Intermediate / helper disposition (Yield Phase Check 6)

No intermediate QA test was created or subsumed by this pass. `tests/qa/harness.py` is the shared
fixture location and remains the single definition of `read_text`, `log_artifacts`,
`assert_upstream_response` and `startup_refused`; 070's promotion of the first three out of
`test_qap_ms002.py` still stands and nothing was re-duplicated here. Nothing is newly deletable: the
one helper the drain touched is consumed by all three passes.

### Post-checkpoint correction (Checkpoint 183, completing 182)

182's `tests/qa/test_qap_ms001.py` carried two session-report citations written as relative markdown
links (`../../iterations/077-forward-qa.md`, `073-01-yield.md`). Neither resolves from `tests/qa/`, and
no other file in that tree cites session reports in that form — the kind of inert, unresolvable wording
that itself becomes a backlog line. Both are now plain report names. Docstrings only, no assertion or
signature moved; the module was re-run rather than argued about: **11 passed** in 2.6 s, and both
changed files re-parsed. Checkpoint 182 is left intact rather than amended.

## Result

**`READY-PASSING`**

- All four linked scenarios are gated by persistent tests under `tests/qa/`, driven through the real
  proxy subprocess against the declared live upstream, and the whole module is green on current
  bytes (11/11), with the shared-helper consumers green (15/15, 6/6) and the hermetic default run
  unmoved (455 @ 95.75%).
- The two `forward-qa`-owned backlog rows whose named trigger is this pass are drained, and the
  drain is witnessed by a fail-first test rather than by assertion in prose.
- No production behavior failed an authored test; no scenario's approved contract was weakened,
  reinterpreted, or widened. Nothing is `BLOCKED`: `qa-plan.md`, `qa-project`, the quiescence
  contract (`wait_tcp` / `poll_until` / process-exit quiescence) and the deterministic mode (fresh
  `--dump-dir` per gate, unique system prompts, no timed sleeps) are all declared and used, so there
  is no Capability Gate miss to report.
- **This is not an approval.** `READY-PASSING` records that the persistent test observes the
  contract. Yield Phase independently inspects contract fidelity and re-runs for the authoritative
  `pass`/`fail` on `VP-T001-writes-deferred`, `VP-T001-dump-failure-isolated`,
  `VP-T002-documented-format-preserved` and `VP-T002-default-encoding`. 071's SEMANTIC-APPROVAL of
  these four points stands as issued on 070's bytes; Yield Phase decides whether the harness drain
  recorded here reopens any of them.

### For Yield Phase's attention

1. The `path=`-keyword removal changes a **shared signature** (`Proxy.post`, `Proxy.url`). Every
   consumer in `tests/qa/` was grepped (zero pass `path`) and every gate re-run, but it is the kind
   of change Check 1 should confirm independently rather than take from this report.
2. Dropping the body limb makes `assert_upstream_response` a **pure status predicate**. The
   anti-vacuity property it used to (perhaps accidentally) carry is now carried in two places the
   gates can see: the `require_upstream()` fail-closed preflight inside `Proxy.__init__`, and the
   artifact-content assertions in `TC-100`/`TC-107`/`TC-108` which cannot pass unless interception
   actually happened. If Yield Phase wants a gate-tier witness that the preflight itself is load-
   bearing, that is a new claim not in `qa-plan.md` and belongs to the plan, not to a test edit here.
3. Two other `forward-qa`-owned rows remain open and were **not** in scope for this pass, since
   neither names `tests/qa/harness.py` or this pass as its trigger: `QB:qa-tls-peer/stale-localhost-claim`
   is already resolved (Checkpoint 156), while `QB:qa-plan/tc-tls-lifecycle-clause` is owned by the
   Authority Registrar's finalization pass. Listed so the close does not read this pass as draining
   the lane wholesale.

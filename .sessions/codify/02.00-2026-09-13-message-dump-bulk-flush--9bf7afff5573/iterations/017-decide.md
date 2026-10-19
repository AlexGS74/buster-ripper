# 017 — Decide (REPLAN-INCREMENT: upstream TLS cluster — `US-011`)

## Orchestrator Prompt (verbatim)

> Execute the decide-interactive skill for this session. Arguments: $ARGUMENTS
>
> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
> Mode: REPLAN-INCREMENT
> Cluster: upstream TLS cluster (US-011 — skip-verify flag for the upstream HTTP client)
> Reason (from Interview, verbatim): upstream TLS cluster — every agent-ready criterion fails. No done-when criteria, no bound `## Anchor Map` row, no validation point, no milestone mapping; the skip-verify flag is `Mode: NEW` and so owes a complete `RB-*` Reuse Evidence row.

(The remainder of the dispatch was the `decide-interactive`, `codify-agents`, `critical-thinker`,
`codify-user-prompt`, and `codify-deterministic-closure` skill bodies. `$ARGUMENTS` was empty.)

## Session Type / Mode

- **`REPLAN-INCREMENT`**, scoped to the **upstream TLS cluster ONLY**. Dump cluster untouched.
- **Operating mode within `decide-interactive`: `PLAN-SELECTED`.** The pick is not open — the cluster is
  `US-011`'s, placement was settled in-session (round 9), and organize 016 landed its half. What this pass owes
  is the *finalization* 016 deferred to decide: the task handle and done-when, the proof route, the milestone,
  the goal row, the flag token, the documentation boundary, and the two `OPEN` goal-shape rows.
- **Not a child session.** `session-identity.json` has **no `parent_session_uid`** → the parent coordination
  contract does not apply; no parent path to resolve, no `coordination status` to copy.
- **No new `US-*` reservation is owed.** The direct user context in scope is `US-011`, which already exists as
  the canonical record. The orchestrator prompt carries no new user utterance.
- **Question Mode: deferred** — no `AskUserQuestion` on this dispatch path. Both forks authored below are routed to
  `interview` for presentation; neither is presented here.

## Context Re-established

- Session identity re-verified: handle `…--9bf7afff5573`, UID `9bf7afff5573` → first 12 hex of
  `9bf7afff-5573-402a-b102-827423a1dded`; display id `02.00-2026-09-13-message-dump-bulk-flush`. Matches.
- Read before acting: `decisions.md` (`US-011` in full, `US-009`, `US-010`, `FK-007`), `plan-summary.md`
  (all sections), `handoff-context.md`, `iterations/016-organize.md` (in full), `iterations/009-interview.md`
  §Round 8–9, `quality-backlog.md`, `.agents/skills/qa-project/SKILL.md`, and the source:
  `buster_ripper/{server,main,config}.py`, `pyproject.toml`, `README.md` §Quick start / §Serve script options,
  `serve_buster_ripper.sh`, and installed `aiohttp`.
- **No context conflict.** Every surface agrees: PLAN-SELECTED landed; no FK open; the `qa-project` gap is
  closed; the outstanding items are this replan and the three `OI:` stamps.

## What This Pass Finalizes (and what it must not touch)

| Row | 016 state | 017 state |
|---|---|---|
| `T006` handle / done-when / split | candidate | **finalized** — one task, no split |
| Proof route (`016` §Where the Testable Decision Lives A–E) | open, decide's | **finalized** — route D, tier per `FK-009` |
| Milestone | candidate `MS-004` | **finalized** — `MS-004`, new, not a join |
| `G-007` | candidate, 3 halves, third `OPEN` | **finalized** — two halves settled, third `READY-BLOCKED` on `FK-008` |
| Flag token | 3 candidates, not settled | **finalized** — `--skip-tls-verify` |
| Documentation boundary | decide's call | **finalized** — `--help` + README + `serve_buster_ripper.sh` |
| `VP-T006-optout-is-explicit` | candidate, do-not-author | **`READY-BLOCKED`** on `FK-008` |
| `VP-T006-dump-cluster-unaffected` | candidate | **dropped** (deliberate — see §Dropped-Clause Sweep) |
| `RB-005` | complete | **verified complete, unchanged** |
| Two `OPEN` goal-shape rows | routed to decide | **two Forks authored** — `FK-008` (preference), `FK-009` (ratification) |
| Capacity finding (`main.py:main`) | reported, decide's to route | **CLOSURE: absorbed** — see §Deterministic Closure Packets |

## Context Conflict Detected

**None.** The orchestrator prompt carried no technical detail beyond the session ID, mode, cluster and
the interview's reason string. `016`'s plan-level rows, `US-011`'s normative choice, and the source all agree.

## Fork Authoring

Two Forks, both reserved through the allocator and filled in place. **No ID was calculated, reused, or
overwritten**; `FK-008` and `FK-009` are the allocator's.

| FK | Type | One-line question | Why it is the user's |
|---|---|---|---|
| `FK-008` | `preference` | When `--skip-tls-verify` is passed, must the process say so at runtime? | No data settles it: `US-011`'s words name the flag's shape and the posture, and "explicit" modifies the flag, not the output. Both readings are consistent with every word the user wrote. `Why-yours` quotes `it is irreversible or expensive to reverse (data loss, public API surface, security)`, binding `security` to the app's public observable surface. |
| `FK-009` | `ratification` | May the TLS gate start its own local self-signed HTTPS peer, amending the approved harness contract? | The act is authority-bound: option 1 edits `.agents/skills/qa-project/SKILL.md`, which `US-010` approved as presented. `Why-yours` quotes `the choice would contradict a recorded user decision or a protected invariant`, binding that decision to `US-010` and its Upstream row. |

`FK-009` is a `ratification`, not a preference: the closure ranks option 1 first (§Deterministic Closure
Packets below), so the answer is known — what is not the agent's is the **act** of amending a user-approved
document. Stated as a preference it would invent a choice that no longer exists; stated as a ratification it lets
the user agree in one word. `Recommendation: option 1` is present, as a ratification requires.

`FK-008` carries **no recommendation** — `Recommendation: none`, as a preference requires. A recommendation here
would be pre-selection of user-visible product policy.

**Neither fork was authored on a preference for its own convenience.** `FK-009` exists because the closure's own ranking
selects a route that contradicts `US-010`; the alternative (hermetic tier) is admissible and would dissolve the question,
which is exactly why the question is real rather than rhetorical — see §Why `FK-009` Is Not Dissolved By Choosing The Other Route.

## Deterministic Closure Packets

### Packet 1 — the proof route / gate tier (`C-TLS-1`)

```text
CLOSURE: question
Finding: C-TLS-1 — iterations/016-organize.md §Goal Shape Analysis (row C-TLS-1, Authority: OPEN — user)
Required outcome: both upstream-TLS postures (verification ON with no opt-out; verification OFF with the opt-out) are
  proven against a peer whose certificate does not chain to a trusted root, at MS-004
Authority: US-010 (the user's approval of the QA harness contract, decisions.md#us-010);
  .agents/skills/qa-project/SKILL.md §Harness Declarations row "Upstream" + §Conventions "Hermetic default suite" +
  §Capability Declarations row "Harness tier"; US-011 (decisions.md#us-011); codify-agents §Validation Point Design
  ("Implementation Forward does not author persistent gate tests ... tests produced by forward-qa")
Admissible candidates:
  A — a process-level tests/qa/ gate that starts the real console script with --upstream https://127.0.0.1:<peer-port>,
      against a gate-local self-signed HTTPS peer; authored by forward-qa
  B — a hermetic tests/ posture proof against a loopback self-signed peer, with no tests/qa/ TLS gate; the
      flag→global link proven by a CLI-invocation unit test with web.run_app patched
Rejected candidates:
  C (patch aiohttp.TCPConnector in on_startup and assert the ssl= kwarg) -> the mock mirrors the implementation;
      proves the intent, not the effect. 016 §Where the Testable Decision Lives already flags this.
  D (read session.connector._ssl) -> asserts on a private attribute; aiohttp exposes no public accessor
      (connector.py:1003 assigns it, connector.py:1243 reads it, both private — verified below).
  E (point at the settled plain-HTTP upstream) -> no TLS is negotiated, so neither posture is observable
Selected closure: none — A and B are unranked by settled authority on the axis that decides
Risk shape: unchanged — neither candidate weakens or skips an existing gate; A amends an approved document, B leaves
  the stronger evidence unbuilt
Deferred additive work: none
Owning route: decide authors the fork (FK-009); interview presents it
Question required: yes
```

**Why the ranking does not close this.** The closure rules rank candidates by correctness first and
architectural integrity second. Both A and B prove the required outcome, so correctness is equivalent. A then wins on
architectural integrity — its gate is authored by `forward-qa` rather than by the agent that implemented the connector, and
it exercises the shipped console script. But A's admission is conditioned on `US-010`'s Upstream row, which the caller cannot
amend; B is the authority-preserving alternative. Rule 8's `question` applies: two materially different admissible outcomes
remain, and no settled authority the caller holds ranks them — one of the two authorities that would rank them is the very
record the user owns.

### Packet 2 — the capacity ratchet on `main.py:main`

```text
CLOSURE: absorbed
Finding: 016 §Capacity Gate / §Gates Found Unmet item 1 — main.py:main is at P=21 and L≈31, hard violations on both
  axes, and the cluster adds a parameter and a statement to it
Required outcome: --skip-tls-verify (and the three dump-cluster options) exist on the CLI and assign their config
  globals, while main.py:main's already-violated parameter count does not grow
Authority: codify-agents §Preventive Change-Surface Capacity Gate step 5 ("An already-violated function metric may
  not increase"); plan-summary.md §Task List T002/T003/T006 (the options must be declared and must assign their globals
  — the assignment is the requirement, the parameter is not)
Admissible candidates:
  A — each new option declares a callback that assigns its own config global, so main()'s parameter list does not grow;
      T006 additionally performs the smallest coherent L extraction inside its touch set
  B — main() gains one parameter per option (P 21 -> 24/25)
  C — restructure the CLI (option groups, or a **kwargs main)
Rejected candidates:
  B -> increases an already-violated function metric, which step 5 forbids absolutely
  C -> broadens scope beyond a smaller complete remediation, and its touch set is the 20 settled option declarations
Selected closure: A
Risk shape: unchanged — same flags, same defaults, same --help text, same globals; only the location of the
  assignment statement differs
Deferred additive work: none
Owning route: decide (plan owner) — recorded in plan-summary.md §Task List and §Goals and Constraints
Question required: no
```

## The Flag Token (replan input 10 — settled, not a fork)

**`--skip-tls-verify`**, config global `SKIP_TLS_VERIFY: bool = False`.

`US-011` delegated by shape — *"have explicit skip verify, like usual, skip TLS verify flag"* — and named the
words. Read off convention: the user's own phrase in repo kebab-case is `--skip-tls-verify`; every option in
`main.py` is kebab-case (`--strip-date`, `--preserved-thinking`, `--eval-thinking-budget`); the repo's one
domain-prefixed option (`--dump-dir`, `--dump-format`) prefixes the *subsystem*, and there is no second TLS surface
to disambiguate from (the proxy listens on plain HTTP — `--host`/`--port`, no inbound TLS anywhere), so
`--upstream-skip-tls-verify` disambiguates against nothing. `--insecure-skip-tls-verify` is kubectl's spelling, not
this repo's, and it is longer than the user's own phrase.

**Why this is not a fork.** The user delegated the name and supplied the words; the repo convention supplies the form.
The read-off is deterministic, so there is no preference to express. It is recorded as a plan decision in `T006`'s
done-when with its source, and it is reversible-cheap while unshipped (see §Fix Decision Principle note below) — the
contrary argument (that a shipped CLI token is a public surface) is recorded in §Residual Risk, not converted into a question.

## Documentation Boundary (settled)

| Surface | Decision | Why |
|---|---|---|
| `--help` text | **in `T006`** | The `help=` string is part of the flag's own declaration; the six `is_flag=True` precedents all carry one |
| `README.md` §Serve script options | **in `T006`** | The section is the documented local-deployment surface (`README.md:71-93`) and already documents `STRIP_DATE`/`VERBOSE`/`DUMP_DIR`/`UPSTREAM`/`PORT` |
| `serve_buster_ripper.sh` `SKIP_TLS_VERIFY` env var | **in `T006`** | The script is the local-deployment entry point the user's rationale names, and it is the only surface where the flag is otherwise **unreachable**: `exec buster-ripper --upstream … --host … --port … ${EXTRA_FLAGS}` (`serve_buster_ripper.sh:37-41`) forwards no `"$@"`, so a script user cannot pass `--skip-tls-verify` without editing the script. Default `SKIP_TLS_VERIFY=${SKIP_TLS_VERIFY:-0}` keeps the safe posture |
| `--upstream` default | **no change owed** | A scheme change is a value change (`main.py:13`). Recorded as a `G-007` boundary and restated in `T006`'s done-when REMOVED clause — the double declaration (`config.py:13` vs `main.py:13`) is the quality-backlog finding, not this task's |

**Fix Decision Principle note.** The env-var addition is *above* the MVP floor only if the floor is read as "the flag
exists". It is included because the user's stated rationale is the local-deployment path and that path cannot reach the
flag today. It is a three-line, trivially reversible change with no new behavior when unset, so it is not a
hard-to-reverse boundary crossing and is not a Deferred Ruling.

## Why `FK-009` Is Not Dissolved By Choosing The Other Route

The tempting shortcut is to pick the hermetic tier, notice that `US-010` is then untouched, and declare `C-TLS-1`
dissolved — no fork, one fewer user question. That reasoning is sound on its own terms and is **rejected**, for a reason
that is worth stating because it is a general trap:

> **Declining to act leaves nothing to cite.** An action taken once leaves an artifact that the next round can point at;
> choosing *not* to build the process-level gate leaves nothing, so a later round can never cite this decision as precedent
> for having avoided a question. The asymmetry runs one way.

Concretely: the hermetic route is admissible, so the closure does not *force* a question. But selecting it because it
avoids the question — rather than because it is better — would be selecting the weaker evidence to dodge a touchpoint.
The two routes are not equal on the axis the closure ranks second: A's gate is authored independently by `forward-qa`
(codify-agents §Validation Point Design: *"Implementation Forward does not author persistent gate tests"*) and exercises the
shipped console script; B's is authored by the implementing increment or by `forward-qa` inside the hermetic tree the
contract reserves for the mocked suite. `FK-009` therefore exists, carries the recommendation, and states the fallback in
its `Impact` so option 2 is a named landing rather than a re-plan.

## Task Finalization

**`T006` — Upstream TLS verification opt-out** (finalized; **not split**)

*Why one task and not two.* The seam is a single statement pair (`server.py:47` becomes a connector + session), and
every criterion below is a facet of that one seam: the global, the flag, the connector, the proof. Splitting the proof
into its own task would separate a gate from the change it gates without adding a boundary — `MS-003` is the precedent for a
milestone whose gate is the suite itself, and `T004` is the precedent for a task that carries its own tests.

- **ADDED**: `config.SKIP_TLS_VERIFY: bool = False` in `config.py`, following `KV_POLL_INTERVAL`'s module-global
  shape (`config.py:44`) and the `False`-default shape of `STRIP_DATE`/`VERBOSE` (`config.py:14-15`).
- **ADDED**: `main.py`'s `--skip-tls-verify` option — `is_flag=True`, `help=` stating that it disables upstream
  certificate verification and is for self-signed certificates, following the six existing `is_flag=True` precedents'
  shape (`main.py:19,21,23,38,42,51`). It assigns `config.SKIP_TLS_VERIFY` **through the option's own `callback`**,
  so `main()`'s parameter list does not grow (closure packet 2).
- **ADDED**: `server.py:47` builds an explicit `aiohttp.TCPConnector(ssl=not config.SKIP_TLS_VERIFY)` and passes it as
  `connector=` to the `ClientSession` — **inside `on_startup`**, because `ClientSession` raises
  `RuntimeError("Session and connector has to use same event loop")` unless `connector._loop is loop`
  (`client.py:377`; the default connector is built at `client.py:370-372`). With the opt-out unset the posture is
  `ssl=True`, which `_merge_ssl_params` resolves to `_SSL_CONTEXT_VERIFIED` (`connector.py:1003`, `:924`) — the same
  context aiohttp's own default connector already yields — so the unset path is **behaviour-preserving**. The explicit
  connector is the **only** seam: `ClientSession.__init__` accepts `connector=` and no `ssl` parameter
  (`client.py:295-315`).
- **ADDED**: the smallest coherent extraction inside `main.py:main`'s touch set — the config-application block
  (`main.py:70-86`) moves into a module-private helper — reducing **L** (closure packet 2, step 2).
- **ADDED**: `README.md`'s serve-script section documents `SKIP_TLS_VERIFY` and the flag, and states that it disables
  upstream certificate verification.
- **ADDED**: `serve_buster_ripper.sh`'s `SKIP_TLS_VERIFY` env var, defaulting off, mapped to `--skip-tls-verify`.
- **ADDED**: the proof material the selected route needs — a self-signed certificate/key fixture for `127.0.0.1`
  committed under `tests/` (the repo has none: `cryptography` and `trustme` are **absent** from
  `.venv/lib/python3.13/site-packages`; `openssl` 3.6.3 is present at `/opt/homebrew/bin/openssl` but a
  test-time CLI dependency is avoidable and a committed fixture is deterministic). The peer must be a **self-signed**
  certificate for `127.0.0.1`, so the failure mode is the untrusted issuer and **not** a hostname mismatch.
- **ADDED**: unit coverage for the default (`SKIP_TLS_VERIFY is False`), the flag→global link, and — per `FK-009`'s
  answer — either the process-level gate under `tests/qa/` (`QAP-MS004`) or the hermetic posture proof in `tests/`.
- **REMOVED**: **nothing.** The connector's default posture is unchanged when the opt-out is unset; `--upstream`'s
  default is unchanged; no dump-cluster behavior changes.
- **Depends on**: **—**. Independent of `T001`–`T005`: `on_startup` and `on_cleanup` are separate functions and the
  `main.py` option block is additive. The only interaction is adjacent-line merge friction in `main.py`.

**Not a task.** The capacity extraction rides `T006`'s increment. `RB-005` is complete and unchanged. The certificate
fixture is test material authorized by this task's done-when, not an Anchor Map row (the `T002` precedent: `tests/test_dump.py`
was authorized by `T004`'s done-when with no new row).

## Anchor Map — Stance Finalization

`016`'s two rows stand **as written**; this pass finalizes their stance tokens and verifies `RB-005`'s completeness.
No existing row is re-stanced and no settled binding is re-opened.

| Row | Mode | Final? | RB cited | Check |
|---|---|---|---|---|
| upstream TLS verification opt-out (`config.py` global + `main.py` option) | **NEW** | yes | `RB-005` | Complete: name search (zero `ssl\|tls\|insecure\|noverify\|cafile\|truststore` matches package-wide, `"--no-` → zero), role/peer (six `is_flag=True` shape peers, all opt-in, none relaxing a safety default; `--eval-profile`/`--backend-type` a different shape; no second client-construction site), native capability (aiohttp 3.13.3 `TCPConnector(ssl=…)` verified in the installed source, with the load-bearing negative fact that `ClientSession.__init__` takes no `ssl` kwarg). The `Notes` cell cites `RB-005` |
| shared client construction (`server.py:47`) | **extend** | yes | `US-011`'s own wording (*"if it's already implemented on the positive path"*). The referent exists and is modified in place; no new artifact |

**Read path — still not an anchor row.** Unchanged from `016`: the cluster has no reader, no artifact and no file.

## Goal Definition and Coverage — `G-007` finalized

`G-007` is finalized as **one** goal with two settled halves and one `READY-BLOCKED` third. The third half is
**not** written into `Met when`: `FK-008` is unresolved, and finalizing the row around either reading would settle a
user question by writing. `016`'s third `G-007` row made "nothing records that verification is disabled" a *failure
witness*, which presupposes reading (ii); that framing is corrected here to a fork reference.

| Field | Final content |
|---|---|
| Outcome | The shared client verifies the upstream's certificate when the upstream is HTTPS, and an explicit opt-out turns verification off for self-signed certificates |
| Observation surface | the outbound TLS handshake the shared `aiohttp.ClientSession` (`server.py:47`) performs against `config.UPSTREAM`; the CLI surface (`--help`, `serve_buster_ripper.sh`); the default run's process output |
| `Met when` | With **no** opt-out and `--upstream https://<untrusted-cert host>`, a request through the proxy does not complete against that upstream and the failure is a certificate-verification error; with the **same** peer and the opt-out passed, the request completes. The opt-out defaults OFF and is never inferred from the URL scheme |
| Failure witness 1 | The proxy starts, forwards and returns 200 against a **self-signed HTTPS upstream with no opt-out set** — the client was built with `ssl=False` unconditionally (or the connector was dropped), so the posture is inverted. Every task completed, the flag exists and works, `--help` is accurate, and the security property the user asked for is absent |
| Failure witness 2 | The flag is accepted and sets its global, but the connector is built **without** it (global read at import time, or the session constructed before the assignment lands), so both postures behave identically — with and without the flag, the same self-signed HTTPS upstream yields the same outcome |
| Boundary / non-goals | No mTLS / client certificates; **no custom CA-bundle or truststore option** (the user asked for the skip flag by name; a CA option is new scope the plan does not hold); no change to `--upstream`'s default; no inbound (client-facing) TLS; process-wide, not per-request |
| Alternate reading (excluded) | Verification-on is a *deliverable* → excluded by the user's own words (*"if it's already implemented on the positive path"*) and by `connector.py:978` (`ssl: … = True`) |
| Alternate reading (not excluded, boundary) | Verification-on needs a positive affordance the app controls (a CA-bundle path) → not excluded by code, but additional scope; recorded as a boundary, not a goal |
| Third half — `READY-BLOCKED` on `FK-008` | Whether *explicit* requires the skip to be **visible at runtime**. Settled halves: the flag exists and defaults off. Open half: whether the process emits a record. **Does not enter `Met when` until `FK-008` resolves** |
| Authority | `code-grounded` (the default posture; the global is structurally forced — `on_startup` reads `config.<NAME>` only, `server.py:52`) / `user` (the CA boundary, the warning, the public CLI surface) |

**Goal Coverage row**: `G-007` → required capabilities/flows = `config.SKIP_TLS_VERIFY` global → the CLI flag assigns
it → `server.py:47` builds the shared `TCPConnector` carrying the posture → one connector governs every outbound call.
Tasks: `T006`. Validation points: `VP-T006-verify-on-by-default`, `VP-T006-optout-honored`,
`VP-T006-posture-is-process-wide`. Disposition: **mapped**.

## Invariant Reconciliation — TLS cluster finalized

### Planned invariants (final)

- `PI:T006/upstream_tls/verification_on_unless_explicitly_disabled` — **hard_requirement**: the shared client
  verifies the upstream's certificate unless the opt-out is explicitly configured. The opt-out is never inferred from the URL
  scheme, the host, or the environment. Grounds `G-007` and the `shared_client_verifies_by_default` refine.
- `PI:T006/upstream_tls/optout_is_process_wide` — **hard_requirement**: one connector, one posture, for every outbound
  call — the proxied request, the `/metrics` poll (`compaction.poll_kv_cache`) and the `/v1/models` probe
  (`backend.detect_backend`) alike. **Coupling this names**: splitting the client construction into a connector + session
  pair at `server.py:47` does not create per-call-site postures, and this invariant is what keeps that true. Where it is
  checked: `VP-T006-posture-is-process-wide`, paired with a black-box VP so it cannot mirror the implementation.

*(`FK-008`'s answer may add a third: the skip is announced. It is not authored now — under reading (i) no such
behavior exists, and authoring it would settle the fork by writing.)*

### Conflict matrix (TLS rows; the dump rows are unchanged from 015)

| Planned invariant | Existing invariant | Relationship | Decision | Why |
|---|---|---|---|---|
| `PI:T006/upstream_tls/verification_on_unless_explicitly_disabled` | `EI:buster_ripper.server/on_startup/shared_client_verifies_by_default` | refines | **preserve** | The default survives; the change adds the named way off and a witness where none existed. The old statement ("built with no `ssl`/`connector=` argument") becomes literally false, which is why it needs a register entry, not because its property changes |
| `PI:T006/upstream_tls/optout_is_process_wide` | *(none)* | unknown | **preserve** | No existing invariant constrains outbound-client posture; new surface from `US-011` |

### Existing invariants in scope — `none`, re-verified

`016` narrowed the scope to the outbound client construction, the `config`/CLI seam, and the request path, and found no
`EI:` intersecting it. **Re-verified this pass** against the six collected `EI:` entries (`output_layout`,
`sequential_diff_chain`, `turn_zero_is_full_body`, `dump_failure_isolated`, `derived_after_normalize`,
`graceful_shutdown_hook`): the TLS cluster writes no file, diffs no text, derives no session id, and touches `on_startup`,
not `on_cleanup`. `dump_failure_isolated` is the closest and still does not intersect: a TLS change makes the
**upstream** call fail, which is the pre-existing request-error path, not the dump path.

### Override register — a fourth entry is owed

`016` drafted `EI:buster_ripper.server/on_startup/shared_client_verifies_by_default` and dispositioned it **refine**. The
register holds every `OI:` disposition, refinements included (`OI:.../sequential_diff_chain` and `OI:.../turn_zero_is_full_body`
are both refinements). The TLS refine therefore lands as `OI:` entry 4, and — like the other three — it needs the user's
stamp before final approval. **Topic 3 grows from three stamps to four.**

## Validation Point Matrix — TLS rows finalized

| Validation point | Observable behavior | Observable after | Validate at | QA author pass | Scope reason | Evidence needed |
|---|---|---|---|---|---|---|
| `VP-T006-verify-on-by-default` | With **no** opt-out passed and an untrusted-certificate HTTPS upstream, a request through the proxy does **not** complete against that upstream, and the failure is a certificate-verification error | `T006` | `MS-004` | `QAP-MS004` | The default is currently implicit (an aiohttp default, `connector.py:978`) and the change is what can invert it, so it needs a witness for the first time — `VP-T002-default-encoding` is the precedent. `T006` is the earliest boundary: no TLS VP is observable before the connector exists, and the connector is the whole of the increment | connector construction + the failed request's error class |
| `VP-T006-optout-honored` | With the opt-out passed and the **same** untrusted-certificate HTTPS upstream, a request through the proxy completes | `T006` | `MS-004` | `QAP-MS004` | Only observable when **both** postures are exercised against one peer; a single-posture test cannot distinguish "flag works" from "flag ignored" | the completed request + the connector construction |
| `VP-T006-posture-is-process-wide` | One opt-out governs every outbound call — the `/metrics` poll and the `/v1/models` probe use the same posture as the proxied request | `T006` | `MS-004` | `QAP-MS004` | Structural/instrumented, protecting `PI:T006/upstream_tls/optout_is_process_wide`. Exactly one `ClientSession` exists (`server.py:47`, consumed at `:21,51,72` and `routes/{chat:37,passthrough:25,messages:127,228}`). **Must be paired with a black-box VP so it cannot mirror the implementation** | instrumented call sites / one-connector assertion |
| `VP-T006-optout-is-explicit` | **`READY-BLOCKED` on `FK-008`.** Under reading (ii): with the opt-out passed the process emits a record stating certificate verification is disabled; with it absent, no such record. **Do not author before `FK-008` resolves** — under reading (i) there is no such behavior and this VP must not exist | `T006` | `MS-004` | `QAP-MS004` | Conditional on the row-3 reading. Authoring it now would settle a user question by writing | log record presence/absence |
| ~~`VP-T006-dump-cluster-unaffected`~~ | **DROPPED** — see §Dropped-Clause Sweep | — | — | — | The handle is retired and must not be reused. The dump cluster's own gates plus `VP-T003-passthrough-regression` are the regression evidence, and on a plain-HTTP upstream the flag has no effect, so the VP would be a tautology — it could not fail for a reason no other test fails for | — |

`Observable after` for all four is `T006`, and no earlier boundary is sufficient: no TLS VP is observable before the
connector is built, and the connector is the whole of the increment. The two settled VPs and the structural one are
`READY-BLOCKED` on `FK-009` for their **evidence tier**, not on implementation; `VP-T006-optout-is-explicit` is
`READY-BLOCKED` on `FK-008` for its **existence**.

## Milestone Finalization

| Milestone | Constituent tasks | Complete when | Validation points | QA author pass |
|---|---|---|---|---|
| `MS-004` (**new**, not a join) | `T006` | With the opt-out absent an untrusted-certificate HTTPS upstream fails; with it present the same upstream succeeds; both postures come from one connector; the existing suite is green | `VP-T006-verify-on-by-default`, `VP-T006-optout-honored`, `VP-T006-posture-is-process-wide` | `QAP-MS004` *(per `FK-009`; fallback `— (the hermetic suite is the gate)`)* |

`MS-001`…`MS-003` are **not** joined: their completion criteria are dump-shaped and the TLS cluster composes with
none of them. `MS-003` is the coverage gate and the TLS cluster is **coverage-neutral** — `K = 0`, because
`pyproject.toml:24-28` omits `buster_ripper/routes/*`, `buster_ripper/main.py` **and** `buster_ripper/server.py`, so no
line of this cluster earns coverage credit wherever it is written and the `U ≤ 3.2 + 0.2·K` gate is untouched by it.

## Behavior Dispositions — TLS rows

| Behavior | Disposition | Validation point or reason |
|---|---|---|
| (new) the upstream's certificate is verified with no opt-out | mapped | `VP-T006-verify-on-by-default` |
| (new) the opt-out turns verification off | mapped | `VP-T006-optout-honored` |
| (new) the opt-out is off by default | mapped | `VP-T006-verify-on-by-default` |
| (new) the opt-out is process-wide | mapped | `VP-T006-posture-is-process-wide` |
| (new) skipping is visible at runtime | **deferred** | `FK-008`; author only if reading (ii) is taken |
| (new) the flag appears in `--help`, the README and the serve script | **mapped — structural** | `T006`'s done-when. No VP: the dump cluster's `T005` (README) carries none either, and a help-string assertion proves the string, not the posture |

## Dropped-Clause Sweep

`016` is the direct predecessor for this cluster, and this pass supersedes its candidate `T006`, candidate `VP-T006-*`,
candidate `MS-004` and candidate `G-007` with final versions. Every load-bearing clause is dispositioned:

| Clause (from 016) | Disposition | Why / where it now lives |
|---|---|---|
| `T006` — `config.py` opt-out global, default off | carried | `T006` ADDED clause 1; token and shape named |
| `T006` — `main.py` `is_flag=True` option assigning the global | carried | `T006` ADDED clause 2; token settled; callback mechanism added by closure packet 2 |
| `T006` — `server.py:47` explicit `TCPConnector`, inside `on_startup`, behaviour-preserving when unset | carried verbatim | `T006` ADDED clause 3 |
| `T006` — the proof route from A–E | carried, narrowed | Route D selected; C/D/E rejected with reasons in closure packet 1 |
| `T006` — cert material if route D (repo has none; `openssl` present) | carried, sharpened | `T006` ADDED clause 7 — committed fixture, self-signed for `127.0.0.1` so the failure is the untrusted issuer and not a hostname mismatch |
| `T006` — unit coverage for both postures and the default | carried | `T006` ADDED clause 8; tier per `FK-009` |
| `T006` — README and/or serve script documentation (decide's call) | carried, decided | Both surfaces; §Documentation Boundary |
| `T006` — `REMOVED: nothing` | carried | `T006` REMOVED clause |
| `T006` — `Depends on: —` | carried | `T006` Depends clause, with the reason |
| `T006` — the capacity extraction at `main.py:main` rides this increment, not a task | carried | `T006` §Not a task; mechanism decided by closure packet 2 |
| `VP-T006-verify-on-by-default` | carried | VP matrix, row 1 |
| `VP-T006-optout-honored` | carried | VP matrix, row 2 |
| `VP-T006-posture-is-process-wide` | carried | VP matrix, row 3 |
| `VP-T006-optout-is-explicit` (conditional, do not author) | carried | VP matrix, row 4, now `READY-BLOCKED` on `FK-008` with both landings named |
| `VP-T006-dump-cluster-unaffected` | **dropped — deliberate** | Not authored. `VP-T003-passthrough-regression` plus the dump cluster's own gates already fire if the connector change breaks the shared session, and on a plain-HTTP upstream the flag has no effect, so the VP could not fail for a reason no other test fails for (`codify-agents` §Test Set Design) |
| `MS-004` candidate | carried | Milestone matrix, finalized |
| `G-007` rows 1 and 2 | carried | Goal Definition Matrix |
| `G-007` row 3 (explicitness) | carried, **corrected** | Its drafted failure witness ("nothing in stdout/stderr records that verification is disabled") presupposes reading (ii). Replaced by a `READY-BLOCKED` clause so the row no longer settles `FK-008` by writing |
| `016` §Notes for Decide — two `OPEN` rows | carried, routed | `FK-008`, `FK-009` |
| `016` §Notes for Decide — flag spelling | carried, settled | §The Flag Token |
| `016` §Notes for Decide — `serve_buster_ripper.sh` boundary | carried, settled | §Documentation Boundary |
| `016` §Notes for Decide — `--upstream` default must not change | carried | `G-007` boundary + `T006` REMOVED clause |
| `016` §Capacity Gate correction (replan input 8) | carried, routed | Closure packet 2; `T006` ADDED clause 4 |
| `016` §Replan Input Disposition table (12 items) | carried — not superseded | That table is `016`'s own, and this pass changes no disposition in it. Not re-swept: only the definitions this pass replaces are swept (one hop) |
| `016` §Invariant Reconciliation Prep, `EI:` draft | carried, finalized | §Invariant Reconciliation; the draft becomes `OI:` entry 4 |

All other `016` clauses (its §Validation Performed, §Compaction Decision, §Gates Found Unmet, and the dump-cluster sections it did
not touch) carry verbatim.

## Capacity Gate — `T006` touch surfaces

Baseline at the pre-edit worktree, measured by `016` and re-read this pass:

| Surface | L | N | C | P | Verdict |
|---|---|---|---|---|---|
| `config.py` (104 lines) | — | — | — | — | Green; `+1` global is free |
| `server.py:on_startup` (`:44-57`) | 4 | 0 | 1 | 1 | Green; `+2` statements stays Green |
| `server.py` (112 lines) | — | — | — | — | Green → ~114, Green |
| **`main.py:main` (`:62-126`)** | **≈31** | 1 | ≈4 | **21** | **Hard L violation AND hard P violation** |
| `main.py` (130 lines) | — | — | — | — | Green by line count — which is why the plan missed it |

`Capacity gate: TRIGGERED — main.py:main`. Trigger test: *"a touched function already has a hard LNCP violation … and the
task would add statements, branches, nesting, or parameters to it"*. `main.py:main` is touched and `T006` would add a parameter
and a statement. **Closure packet 2 applies**: the option's `callback` assigns the global, so the parameter count does not grow,
and `T006` extracts the config-application block to reduce L. Focused pre/post evidence is the existing suite
(`tests/test_*`, 98 tests) run before and after the extraction, inside `T006`'s increment.

**`T002`/`T003` are the same finding and are *reported*, not cured.** Their scope is settled and this pass does not reopen it;
the closure that applies to `T006` applies identically to their three options, and the plan records it as a known open item below.

## Gates Found Unmet (reported, not cured)

**Count reported to the orchestrator: `gates found unmet 3`.**

1. **The plan's capacity paragraph still describes only the `T006` instance of the ratchet.** `T002` and `T003` add three options to
   the same already-red `main.py:main` (`P=21`, `L≈31`), and their done-when text is settled. The same closure applies to them; applying
   it is a mechanism amendment that preserves every settled observable (same flags, same defaults, same `--help`), so it is recorded here
   and in `plan-summary.md` §Goals and Constraints for the plan owner rather than silently rewritten into two settled tasks.
   **Owner: the plan owner (Interview), on the user's word.** Not curable by this agent without touching settled task text.
2. **`US-011`'s `Impact.Plan` still reads as a future obligation** — *"those land with the scoped `organize` → `decide` replan"* —
   while both halves have now landed. Owner: `interview` (`US-011`'s `Owner`). The destinations are landed; only the record's own wording lags.
3. **`FK-009` blocks `MS-004`'s gate tier and `FK-008` blocks one VP's existence.** Both are routed as forks; neither can be
   cured by an agent. **Owner: the user, via `interview`.** Until they resolve, `MS-004` cannot be accepted and `VP-T006-optout-is-explicit`
   must not be authored.

## Validation Performed

| Claim | How verified | Result |
|---|---|---|
| Session identity / handle suffix | `session-identity.json` vs handle | matches (`9bf7afff5573` = first 12 hex of UID) |
| No parent coordination contract | `session-identity.json` has no `parent_session_uid` | confirmed — contract N/A |
| `ClientSession.__init__` takes no `ssl` kwarg | `.venv/…/aiohttp/client.py:295-315` read in full | confirmed — `ssl` is absent from the parameter list; `connector=` present |
| The default connector is built with `ssl=True` | `.venv/…/aiohttp/client.py:370-372` | `if connector is None: connector = TCPConnector(loop=loop, ssl_shutdown_timeout=…)` — `ssl` unset → default `True` |
| The connector must share the loop | `.venv/…/aiohttp/client.py:377` | `raise RuntimeError("Session and connector has to use same event loop")` |
| aiohttp's default posture is verification ON | `.venv/…/aiohttp/connector.py:978` | `ssl: Union[bool, Fingerprint, SSLContext] = True` |
| `ssl=False` disables verification | `.venv/…/aiohttp/connector.py:897-917,924-925,1003` | `_make_ssl_context(False)` → `check_hostname=False`, `verify_mode=ssl.CERT_NONE`; `_SSL_CONTEXT_UNVERIFIED`; merged at `:1003` |
| No public accessor for the posture | `016`'s grep of `connector.py` for `def .*ssl` | only `_close`/`_get_ssl_context`; `self._ssl` private — route D/B confirmed, route C rejected as a relay |
| No cert material in the repo | `ls .venv/…/site-packages` filtered for `cryptography|trustme|openssl` | **absent**; `openssl` 3.6.3 present at `/opt/homebrew/bin/openssl` — a committed fixture avoids the test-time CLI dependency |
| `main.py:main` is a hard violation | `main.py:62-67` (21 params), `:70-126` (≈31 stmts) | P=21 hard, L≈31 hard |
| `main.py:70-86` is the config-application block | `main.py` read | 17 statements; the smallest coherent extraction inside `main()`'s own touch set |
| `main.py:107-115` already prints a startup mode summary | `main.py` read | confirmed — `log.info("Tool sorting: enabled | Date stripping: %s | …")`; the adjacent evidence for `FK-008` |
| The serve script cannot forward the flag today | `serve_buster_ripper.sh:37-41` | `exec buster-ripper --upstream … --host … --port … ${EXTRA_FLAGS}` — no `"$@"` pass-through; `EXTRA_FLAGS` is built internally |
| `--help`/README document the flag surfaces | `README.md:71-93` §Serve script options | confirmed — five env vars documented; the section is the local-deployment surface |
| `RB-005` is complete | `plan-summary.md` §Reuse Evidence row `RB-005` | name search, role/peer search, native capability, and conclusion all present — the `Mode: NEW` row is not malformed |
| No settled surface was edited | diff of this pass's edits | `T001`–`T005`, the four `PI:`, `MS-001`–`MS-003`, every dump `VP-*`, `G-001`–`G-006` unchanged; the three `OI:` entries' replacement statements unchanged |
| The `OI:` entries are still unstamped | read of plan §Approved Override Register | all three read `pending — interview final approval`; a fourth is added by this pass, also pending |
| No `FK-`/`US-` ID was calculated or reused | `codify codify reserve-decision-record` ×2 | `FK-008`, `FK-009` returned by the allocator; no ID written by hand |

## Compaction Decision

- Prior same-phase files reviewed: `iterations/003-decide.md`, `008-decide.md`, `010-decide.md`,
  `011-decide.md`, `012-decide.md`, `014-decide.md`, `015-decide.md`.
- **Superseded**: none.
- **Not superseded**: all seven.
- **Compaction performed: no.**
- **If no**: different slice. `003` is the initial decision analysis; `008`/`010`/`011`/`012` are the flush-policy,
  payload and encoding slices; `014` is the collapsed-format closure; `015` is the dump cluster's PLAN-SELECTED task-out.
  This pass is the **TLS cluster**, which changes no decision for any of their slices — it touches no dump task, no dump
  VP, no dump invariant and no dump milestone. Nothing to absorb, and nothing to compact.

## Deferred Questions

- **Does "explicit skip verify" require a runtime record that verification is disabled?**
  - Status: **DEFERRED → `FK-008`** (authored, `State: open`, `Type: preference`, `Recommendation: none`).
  - Impact: decides whether `VP-T006-optout-is-explicit` exists at all.
  - Agent analysis: `none` — no data settles it; see the FK record.
- **Does a TLS gate's local peer amend the contract the user approved?**
  - Status: **DEFERRED → `FK-009`** (authored, `State: open`, `Type: ratification`, `Recommendation: option 1`).
  - Impact: decides `MS-004`'s gate tier and the TLS VPs' evidence.
  - Agent analysis: option 1 recommended by closure packet 1, with option 2's full landing named in the record.

## Questions Asked

**This pass: 0.** Both user-authority rows are authored as Forks and routed to `interview`; Question Mode is
**deferred**, so no `AskUserQuestion` is available on this dispatch path. No `FK-*` or `US-*` ID was calculated by hand;
both were allocated by `codify codify reserve-decision-record`.

## Residual Risk (recorded, not a question)

- **The flag token is settled but reversible-cheap until it ships.** `--skip-tls-verify` is the literal read-off of the user's
  own phrase and matches repo kebab-case exactly, so the read-off is deterministic; but once documented in `README.md` and
  `serve_buster_ripper.sh` it becomes a public CLI surface and renaming it is a breaking change. If the user wants a
  different token, the cheapest moment is before `T006` lands. Not raised as a fork because no data ranks the candidates and the
  user delegated the name by shape — see §The Flag Token.
- **The self-contained encoding's readability clause is untouched by this cluster.** `F-11` and `G-006` are dump-cluster
  matters; `US-011` names no artifact and no reader.

## Next Step

1. **`interview`** presents `FK-008` and `FK-009` (one question per turn), re-runs the agent-ready check on the real `T006`,
   then presents **Topic 3** — now **four** `OI:` stamps — then the final-approval gate.
2. On `FK-009` = option 1, `interview` amends `.agents/skills/qa-project/SKILL.md`'s Upstream row; on option 2 it applies the
   named fallback in `MS-004` and the VP matrix. On `FK-008` = reading (ii), it authors `VP-T006-optout-is-explicit` and the
   `T006` log clause; on reading (i) it leaves both unauthored.
3. Final approval stays blocked until both forks resolve and the four `OI:` stamps land.

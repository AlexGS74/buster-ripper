# Iteration 091 — Yield Phase (MS-001 / QAP-MS001, boundary re-verification on final bytes)

## Orchestrator Prompt (verbatim)

```text
You are a subagent spawned by another session.
Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
Milestone: MS-001
Mode: boundary re-verification on final bytes
```

Orchestrator instruction line:

```
Execute the yield-phase-interactive skill for this session. Arguments: $ARGUMENTS
```

Dispatch context carried into this pass (recorded, not inferred): last `SEMANTIC-APPROVE` at
[078-yield-phase](078-yield-phase.md); three accepted remediation acts landed production bytes on the
interception path since (`T018`+`T019`+`T020` at [083-01-yield](083-01-yield.md), `T019.001` at
[085-01-yield](085-01-yield.md), `T019.002` at [090-01-yield](090-01-yield.md)/Checkpoint 205);
`QAP-MS001` re-entered at [087-forward-qa](087-forward-qa.md) as `READY-PASSING`, byte-scoped to
`8dfa5af`, which predates `T019.002`; `READY-PASSING` is not a verdict — this pass owns `pass`/`fail`
on the four `VP-*`.

## Invocation Contract Verification

- **Session handle**: `02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573` verified against
  `session-identity.json` (`session_id: 02.00-2026-09-13-message-dump-bulk-flush`,
  `session_uid: 9bf7afff-5573-402a-b102-827423a1dded`) — **MATCH** (`9bf7afff5573` = uid groups 1+2).
  Files read directly from `.sessions/codify/<handle>/`; nothing globbed or searched for.
  Not `BLOCKED-SESSION-IDENTITY`.
- **Full milestone gate fields**: `Milestone: MS-001` present. `QA Pass` is not restated in this
  dispatch but is **unambiguous by the plan**: `qa-plan.md` §QA Author Passes gives `MS-001` exactly one
  author pass, `QAP-MS001`, and the dispatch names it (`QAP-MS001` re-entered at 087) — resolved, not
  guessed. `Target base: main` / `Target object: 72e3a1c6861bf0d8de6317adf52289e5f6cfa4e1` likewise are
  not restated in the dispatch but are recorded by the latest base-sync report
  ([054-01-forward](054-01-forward.md)), which names **that same object**, and 071/078 both verified it
  at their own HEADs. Contract satisfied on resolved values; recorded as such rather than silently.
- **BASE-SYNC gate: verified here** — see §Base-Sync Re-Verification. **PASS.**
- **Constituent tasks**: `MS-001 = T001, T002, T005` (Milestone Matrix, `plan-summary.md:204`), all
  accepted by local Yield (031-01 / 031-02 / 049-01) and archived in `plan-completed.md`.
- **Trigger level: full cycle/phase boundary** — all 7 checks run. No Codex findings were supplied with
  this dispatch; the last external review of this line of work is the CR4 packet whose finding became
  `T019.002`, closed at [090-01-yield](090-01-yield.md). §Additional review findings records none.

## OUTCOME UP FRONT — `BLOCKED-UPSTREAM`, no integration verdict issued

The declared live upstream `http://hyperipper.local:30001` — the endpoint `US-009`/`US-010` settled as
the only acceptable one for process-level gates, "never a stub" — **is not reachable from this host at
this boundary**. `192.168.50.132` answers `icmp` in 0.25 ms and its ports `22` and `30000` accept
connections; **port `30001` refuses** (`ConnectionRefusedError [Errno 61]`) on 3 immediate probes, on all
43 periodic probes over the pass's window, and on a final direct re-probe at the close. The harness
therefore refuses every gate that needs it, fail-closed and by design:
**8 of `QAP-MS001`'s 11 gates fail at `require_upstream()` with zero assertion failures**, which is
the harness proving it cannot pass against a dead transport rather than a product defect.

Because `READY-PASSING` is not a verdict and this pass owns `pass`/`fail` on the four `VP-*`, those four
handles are recorded **`unknown`** — not inherited from 087's run at `8dfa5af`, which no longer covers
these bytes. `unknown` on a required scenario is a P1 under Check 6, but its root cause is
environmental, so the honest verdict is **`BLOCKED-UPSTREAM` → re-run this gate when the endpoint is
listening**, not a rejection of the implementation. See §Deterministic Closure and §Routing.

Everything a source-reading check can establish **is** established here, on final bytes: Checks 1–5 PASS
with evidence, Check 7 step 0 (the full hermetic suite) **PASSED at 511 @ 96.71 %**, and the two
non-rerun items the dispatch named (T018's cap deletion, T019.002's key narrowing) are adjudicated
rather than deferred: the milestone's contract is **silent on both**, and where each went is recorded.

## Scope Of This Pass (what moved since 078, and therefore what is not inherited)

078 gave this boundary `SEMANTIC-APPROVE` on HEAD `6508941`. Cumulative production delta since, taken
from the tree rather than from any report:

```
$ git diff --stat 6508941...HEAD -- buster_ripper/ tests/
 buster_ripper/normalize.py           |  13 ++     (T019.002 — new public strip_billing_nonce)
 buster_ripper/routes/chat.py         |   2 +-     (T019 — client arm + LookupError)
 buster_ripper/routes/count_tokens.py |  16 +-     (T019 _read_body arm; T019.001 key; T019.002 narrowed key)
 buster_ripper/routes/messages.py     |  57 +++--  (T018 stream timeout, T019 decode arms, T019.001 result.text)
 buster_ripper/server.py              |   8 +-     (T018 client_max_size=0)
 tests/… 6 modules, +894/-64          (test_transport_parity, test_count_tokens_parity, test_compaction,
                                       test_context_overflow, test_routes_integration, test_upstream_answer_relay)
```

Three commits carry it: `9c90675` (CR3 wave `T018`–`T020`), `e645ff0` (`T019.001`), `f00216c`
(`T019.002` = Checkpoint 205).

**What did NOT move — and this is why the pass is a re-execution, not a re-derivation:**

```
$ git diff --stat 6508941...HEAD -- buster_ripper/dump.py buster_ripper/config.py buster_ripper/main.py \
      buster_ripper/session.py buster_ripper/utils.py README.md serve_buster_ripper.sh pyproject.toml
(empty)                                                   # MS-001's owned surface is byte-identical to 078's
$ git log -1 --format=%h -- tests/qa/
6508941                                                   # the approved gate bytes are 078's own HEAD
```

- The **dump call block** in `messages.py` is byte-identical (content-anchored `diff` of
  `if config.DUMP_DIR is not None: … dump_turn(body)` old vs new → no difference), so the milestone's
  producer site is the one 078 traced.
- Exactly **two definitions** appeared in production since 078 and **none disappeared**:
  `normalize.strip_billing_nonce` (public) and `messages._error_body` (module-private). Both are wired —
  §Check 1.
- Delta **since the QAP re-entry** (`8dfa5af`, Checkpoint 201) is narrower still — `normalize.py`,
  `count_tokens.py`, `test_count_tokens_parity.py`, i.e. `T019.002` alone. 087's `READY-PASSING` is
  byte-scoped to before that, which is exactly why the dispatch forbids inheriting it.

Cumulative review scope for the milestone remains `git diff 72e3a1c6…HEAD`.

## Base-Sync Re-Verification

Latest base-sync report is `BASE-SYNC: unchanged` ([054-01-forward](054-01-forward.md)), target base
`72e3a1c6861bf0d8de6317adf52289e5f6cfa4e1`. Re-verified at HEAD `83de78c`:

```
$ git rev-parse main origin/main            (after git fetch origin main)
72e3a1c6861bf0d8de6317adf52289e5f6cfa4e1   # both — base has not moved
72e3a1c6861bf0d8de6317adf52289e5f6cfa4e1
$ git merge-base --is-ancestor 72e3a1c6… HEAD && echo true
true
$ git rev-list --left-right --count main...HEAD
0       238                                  # 238 ahead / 0 behind
$ git status --porcelain | wc -l
0                                            # clean; nothing moved under this pass
```

- **No newer base-sync report exists**: `grep -rn "BASE-SYNC"` over the session directory returns
  carriers at 033, 035, 054, 056, 058, 069, 071, 072, 078 — 054 is still the newest *report*; the later
  ones re-verify it and restate no target.
- **Every correctness-required integration repair passed local Yield**: `T001`/`T002`/`T005`
  ([031-01](031-01-yield.md)/[031-02](031-02-yield.md)/[049-01](049-01-yield.md)), and since 078 the
  wave members `T018`+`T019`+`T020` ([083-01-yield](083-01-yield.md)), `T019.001`
  ([085-01-yield](085-01-yield.md)), `T019.002` ([090-01-yield](090-01-yield.md)) — all accepted-local,
  none outstanding.
- **No final-phase preventive merge work deferred**: `MS-004` closed at
  [056-yield-phase](056-yield-phase.md); [054-01-forward](054-01-forward.md) reviewed every then-open
  backlog item for merge exposure and emitted no prework packet; `main` has not moved since.
- **`BASE-SYNC: unchanged` holds → the integration verdict is permitted to run.** What stopped it is
  not this gate; it is §Check 6/7's environment precondition.

## Check 1: Caller Verification — PASS

Scope: the two definitions the cycle added, every production entry point on the dump path, and every
removal/replacement in the delta.

**New definitions → production callers (re-derived at HEAD, not inherited)**

| Symbol | Added by | Visibility | Production caller(s) | Verdict |
|---|---|---|---|---|
| `normalize.strip_billing_nonce` | T019.002 | public | `routes/count_tokens.py:10` (import) + `:87` (call, on the live `handle_count_tokens` path behind `server.py`'s `/v1/messages/count_tokens` route) | ✔ one caller, and it is the caller the docstring names — "one caller needs exactly this step and no other" |
| `messages._error_body` | T018/T019 | module-private entry point of the stream arm | `routes/messages.py:161`, inside `_stream_single_request`'s `status >= 400` arm | ✔ reached from `handle_streaming` → `handle_messages` → the route |

`strip_billing_nonce` has **zero** test-only status: the census is
`grep -rn "strip_billing_nonce" buster_ripper/ tests/` → production import+call, plus
`tests/test_count_tokens_parity.py:36,:278` as witnesses. A new public symbol with exactly one
production caller is fine; the failure criterion is *zero* production callers.

**Removed / replaced wiring in the delta, and its replacement**

| Removed or narrowed | By | Replacement + its caller | Verdict |
|---|---|---|---|
| `count_tokens`'s `normalize(body, session_id="")` call and its import | T019.002 | `strip_billing_nonce(body)` at `:87`, imported at `:10`; `grep -n "normalize(" buster_ripper/routes/count_tokens.py` → **zero hits**, so no orphan import survives | ✔ replacement wired on the same statement position |
| aiohttp's 1 MiB request-body cap (implicit default) | T018 | `client_max_size=0` at the app's **only** construction site (`server.py:125`, with the parity reason at `:119-124`); `grep -rn client_max_size buster_ripper/ tests/` → that single assignment and its comment, zero occurrences in `tests/`, so no second limit channel exists | ✔ deleted, not re-set — no number left to disagree with |
| the streaming relay's inherited session `total` | T018 | request-local `timeout=aiohttp.ClientTimeout(total=None)` on that one `session.post` (`messages.py:157`); the non-stream arm keeps inheriting (`server.py` session construction byte-exact, witnessed by `test_the_non_stream_arm_still_inherits_the_shared_deadline`) | ✔ scoped, not global |
| `result.body.decode()/str(result.body)` in the retry loop | T019.001 | `result.text` (`messages.py:117`), same statement position, same consumer | ✔ |
| strict `await resp.text()` / `content.decode()` | T019 | `_error_body(resp)` / `content.decode(errors="replace")` | ✔ |

No definition was deleted anywhere in the delta (`grep -E "^- *(async )?def " ` over the production diff →
zero). The milestone's own entry points are unchanged from 078's census and re-checked here at HEAD:
`dump_turn` ← `messages.py:12,:58` (only production caller, the only route that dumps); `flush_all` ←
`server.py:112`, `dump.py:133`, `dump.py:206`; `poll_flush` ← `server.py:45` inside the task
`create_task`'d at startup; `config.DUMP_FORMATS`/`DUMP_FORMAT` ← `main.py:80` writes via the option
callback, read at `dump.py:141-142`. **Zero severed paths, zero test-only production symbols.**

## Check 2: Migration Parity — PASS

The cycle contains three replaced paths. Parity was taken against the **base monolith** (`buster_ripper_legacy.py`),
which is what the CR3 wave's own authority is, and against the pre-wave bytes.

**Pair 1 — T019.002's narrowed key vs T019's full `normalize()` vs base.** Three-way, because T019.002
replaces something the wave itself had just added:

| Path | What it did before keying `session_id` | Malformed-member body | Digest |
|---|---|---|---|
| base (`buster_ripper_legacy.py:725-750`) | `json.loads(request.body())`, **no normalize at all** | tolerated (`except Exception: body = {}`) | un-normalized → the key the messages path never writes |
| T019 (`8dfa5af`-era `count_tokens.py:87`) | `body, _ = normalize(body, session_id="")` | **500** — the tools/messages steps assume every element is a mapping | normalized |
| T019.002 (HEAD) | `strip_billing_nonce(body)` | tolerated again | **equal to T019's**, witnessed |

So the omission of steps 0, 2, 3, 4 of `normalize()` is **intentional, documented, and parity-restoring
toward base**, and it is not justified by prose alone:
- `_normalize_system` (`normalize.py:50-62`) is the only step that writes `system`'s **text**; step 3's
  `_normalize_cache_control` rewrites block *keys* (`{**block, "text": cleaned}` is step 1's shape, and
  cache_control removal does not touch `text`), and `session_id` (`session.py:93-98`) reads `text`
  fields of dict blocks only. The claim "the narrowed key is the full key" is therefore a structural
  fact about which fields the digest reads — and it is **pinned, not argued**, by
  `tests/test_count_tokens_parity.py::TestTheSystemStripIsTheWholeOfWhatTheDigestCanSee::test_the_narrowed_key_matches_the_full_normalize_one`,
  parametrized over a body battery × `STRIP_DATE` × backend axes, with `STRIP_IMAGES` forced **True** so
  the image step is live in the comparison. That is the witness that reddens the day a `normalize()`
  step starts rewriting `system`.
- The route's own side of the equality — that the stats `/v1/messages` records are findable under the
  narrowed key — is `TestCountTokensKeysOnTheNormalizedDigest::test_a_nonce_body_finds_the_stats_under_the_full_normalize_digest`.
- Both ran green in this pass's hermetic execution (§Check 7 step 0).
- The discarded return value is not a lost side effect: `normalize()` mutated `body` in place and
  returned the same object, and T019 already discarded `changes` (`body, _ = …`). Nothing downstream read it.

**Pair 2 — T018's cap deletion vs base.** Base was Starlette/uvicorn and imposed **no** body cap, so
the migration's 1 MiB default was the divergence and `client_max_size=0` is the parity side; the check
sat at the handler's first read, ahead of `normalize()` and `--strip-images`, so the comparison includes
the strip-images route, which base reached with image-bearing bodies over the cap. Side effects on the
parity side: none of the four handlers' arms changed, so nothing else the capped path did is lost.
`git diff` over the four body-reading handlers shows only the added `LookupError` names and the stream
timeout — no arm narrowed, no swallow removed.

**Pair 3 — T019.001's `result.text` vs the old two-branch decode.** Old side effect was computing
`error_text` for `parse_context_overflow_error`; it is the only one, it is retained, and the old form
was *broken* on this producer (`str(result.body)` on a `StringPayload` yields the repr, i.e. an address
no pattern matches). The change makes the SGLang retry actually fire — the behavior the base stack had
through a different container. No side effect dropped; a latent one gained. **Zero accidental omissions
found across the three pairs.**

## Check 3: Data Flow Completeness — PASS

**Path A — the dump write path (this milestone's producer → artifact).** Byte-identical to 078's trace
at every step the milestone owns: `server.py:133` route registration → `messages.py:29` parse → `:38`
`normalize` → `:39` `sid` → `:42-50` images/preserved-thinking → **`:56-60` `dump_turn(body)` under the
swallow-all** → `dump.py:98 _resolve_format()` → `:99 _lookup_or_create_state` → exclusive three-arm
dispatch → documented: `turn_000.json`/`turn_NNN.diff`/`index.txt`; collapsed: `state.pending` →
`flush_all` (`:112`) → one append to `<format>.log` → the human/`grep` reader documented at
`README.md:151-167,:197-211`. 078 cited the route as `server.py:127`; **the number moved, the wiring did
not** — `T018`'s six-line comment/kwarg insertion sits above it. The two flow facts 078 established still
hold on this tree: `session.py:93-98` hashes only `system`, so the pre-injection `sid` and `dump.py:149`'s
post-injection `sid` cannot diverge; and `counted_proxy_entry.py` patches both `io.open` and
`builtins.open` before importing the app, so the counted ops are the process's own writes.

**Path B — the flush triggers** — unchanged (`dump.py:206`, `server.py:45`, `server.py:112` re-grepped at
§Check 1); order protected in `handoff-context.md` §Notes is not touched by any of the three commits.

**Path C — the request path the gates drive, across the post-078 delta.** This is the check that has to
be re-run rather than inherited, because the delta is *on* it:
- `T018`'s `client_max_size=0` sits **ahead** of `messages.py:29`, so the only way it changes the dump
  path is by letting larger bodies reach it — no new branch, no new drop. ✔ (its consequences for the
  milestone are §T018/T019.002-adjudication.)
- `T019`'s added `LookupError` name is in the **client-parse arm that returns before the dump site**
  (`messages.py:28-33`), so a client naming a codec Python lacks now answers 400 without ever reaching
  `dump_turn` — the same shape it already had for `JSONDecodeError`. No silent drop: nothing that
  reaches the writer is refused earlier than before, only strictly less. ✔
- `T019`/`T018`'s relay-side changes (`_error_body`, `errors="replace"`, `ClientTimeout(total=None)`) are
  all **after** the dump call and cannot affect the artifact. ✔
- The proxy-local branch (`ClientError` → `502` at all four arms) that the 077 harness drain keys its
  liveness oracle on is unchanged by this cycle: the delta adds no fifth arm and removes none, so
  `200 <= status < 500` still refuses a dead transport. ✔
- **Path D — the new seam** (`/v1/messages/count_tokens` → stats): producer `messages.py`'s
  `update_session_stats(sid, …)` with `sid` from the **fully normalized** body; consumer
  `count_tokens.py:88-89` (`sid = session_id(body)` → `session_mod._stats.get(sid)`) with `sid` from the
  **system-stripped** body. The two keys are equal by the structural argument in §Check 2 pair 1 *and* by
  two witnesses, one per side of the seam. The seam does not touch the dump path
  (`grep -n dump buster_ripper/routes/count_tokens.py` → no dump import), which is why this is recorded as
  verified-and-off-milestone rather than as a gate owed here. ✔

Nobody writes and nobody reads a channel the other end ignores; no transformation in the delta loses
information needed downstream.

## Check 4: Consumer Contract Tracing — PASS

Structures created or modified by the cycle, and their consumers:

| Producer | Field / value | Consumers | Handled? |
|---|---|---|---|
| `count_tokens.py:87` `strip_billing_nonce` | returns `list[str]` (the change labels) | **no consumer** — the call is made for its in-place mutation; the return is discarded exactly as T019 discarded `changes` | ✔ vacuous by construction, not by oversight (`_normalize_system` return is read only by `normalize()` itself) |
| `_stream_single_request` | the `status >= 400` answer: `web.Response(body=<str>, status=…, headers=…)` | `handle_streaming`'s `isinstance(result, web.Response) and result.status >= 400` → `result.text` → `parse_context_overflow_error` | ✔ `.text` is defined on every value that can reach the guard |
| `_stream_single_request` | its success answer: a committed `web.StreamResponse` | same guard | ✔ excluded — in aiohttp `web.Response` **subclasses** `StreamResponse`, so a bare `StreamResponse` fails the `isinstance` test and never enters the text branch. The dispatch field is narrower than the producer set in the safe direction |
| `_stream_single_request` | its `ClientError` answer: `web.json_response(…, status=502)` | same guard → `.text` → no overflow match → returned to the client | ✔ (this is the arm `harness.assert_upstream_response`'s `< 500` refuses — re-confirmed: all four arms answer 502, unchanged by this cycle) |
| `messages.py:311` probe | `content.decode(errors="replace")` text | `parse_context_overflow_error`'s two patterns (ASCII) | ✔ replacement cannot corrupt the ASCII digits the patterns match; `CL:python/errors-replace-valid-decode-parity` in the registry carries that premise |
| `main.py:14`/`config.DUMP_FORMAT` and `dump.dump_turn`'s `state.fmt` | unchanged since 078 | `_resolve_format`, the three-arm dispatch, `_flush_state`, `_artifact_path` | ✔ re-checked as unchanged: the delta touches neither `config.py` nor `main.py` nor `dump.py` |

**Merged/coalesced data**: nothing in the cycle touches the flush merge; `pending` batching and
`"\n\n".join` are byte-identical, so 078's per-record `turn NNN` homogeneity finding and its `R5`
near-miss witness stand on the same bytes.

**No new dispatch value can reach a consumer that the consumer does not handle.** The one widening is
in the *input* domain, not the dispatch domain (T018) — §T018/T019.002-adjudication.

## Check 5: Plan Fulfillment — PASS

`MS-001`'s tasks are not being re-implemented at this boundary; the question is whether the cycle's
remediation left the plan's architectural intent intact, including its **inverted** removal test
(`US-012`: `G-008`(a)'s writer must be *retained*, so "addition without removal" is the *required*
posture until `MS-002`'s comparison persuades the user).

- **Owned surface unchanged** — `dump.py`, `config.py`, `main.py`, `README.md`,
  `serve_buster_ripper.sh`, `pyproject.toml`: empty diff vs 078 (§Scope). Every `T001`/`T002`/`T005`
  done-when clause 078 verified against source therefore still verifies: `git diff --exit-code` on those
  paths is the proof, and no clause was re-derived from 078's prose.
- **The protected properties of the milestone's own invariant cluster are not touched by the delta**:
  the four write sites, `flush_all`'s no-`await`/cannot-raise posture, `turn`/`prev_text` advancing at
  append only, and the `on_cleanup` order all live in files the three commits never opened.
- **Does the cycle regress the plan's `EI:` carries?** `dump_failure_isolated` — the swallow-all is
  byte-identical (§Check 1's content-anchored diff), and `T019`'s new `LookupError` name is in the arm
  *above* it, which returns rather than raises. `derived_after_normalize` — `messages.py` still
  normalizes before keying (`:38` then `:39`), byte-identical; `count_tokens` now applies the one step
  that the digest input depends on, which satisfies the property's *reason* and is pinned by
  `test_the_narrowed_key_matches_the_full_normalize_one` (adjudicated Aligned; §Invariant
  Cross-Reference). `graceful_shutdown_hook` — `server.py`'s delta is the `web.Application(...)`
  kwarg at `:125`, and `on_startup`/`on_cleanup` are untouched.
- **Deferred switch still deferred**: `config.DUMP_FORMAT = "documented"` with `show_default` — file
  unchanged since 078. No relay masquerading as a transfer, and no forbidden early removal.
- **`022` scoping intact**: `MS-001` declares no flush; nothing in the cycle added a flush gate —
  `grep -n "dump-flush" tests/qa/test_qap_ms001.py` → the docstring stating the rule, no `extra_args`
  carries either flag (the gate module is byte-identical to the approved set, so this re-check is cheap
  and it is a re-check, not an assumption).

**Verdict on this check: the milestone's plan is still fulfilled on final bytes. What is NOT fulfilled
is Check 6's evidence obligation — an execution requirement, not a plan requirement.**

## T018 / T019.002 adjudication — the two things that are not a rerun

The dispatch names these as the only non-inherited questions at this boundary. Both are answered the
same way: **the milestone's own contract is silent on them, and each already has an owner at a cheaper
tier. Neither gets a gate invented for it here.**

**`T018` — `client_max_size=0` widened the body-size domain reaching the documented dump writer.**
- Contract census: `MS-001`'s `Complete when` (`plan-summary.md:204`) and the four `VP-*` rows
  (`:219`, `:221`, `:227`, `:228`) speak to write counts, layout, deferral and failure isolation. **No
  `VP-*` and no scenario names a request-body size, a cap, or 413.** The word does not occur in
  `qa-plan.md` §Scenario Contracts. Silence, stated plainly — not a gap this milestone owns.
- Where it went, recorded rather than overlooked: `tests/test_transport_parity.py::TestNoRequestBodyCap`
  — 4 hermetic witnesses over the four body-reading route classes (2 MiB through `/v1/messages`,
  `/v1/chat/completions`, the `/v1/models` catch-all, and `/v1/messages/count_tokens`) plus the
  `--strip-images` case. All green in this pass's 511-run.
- What that owner does **not** prove, stated so the next reader does not over-read it: those witnesses
  patch the upstream call and never set `config.DUMP_DIR` (`grep -n "DUMP" tests/test_transport_parity.py`
  → the single hit is a request body, not a dump flag), so **no witness at any tier shows a >1 MiB body
  reaching the writer and being laid down**. The residual is *unwitnessed domain*, not a known defect:
  every branch behind the widened domain is size-independent by reading (`_write_full_turn` serializes
  whatever it is given; the index summary's `KB` column is a float; `difflib` and the `pending_bytes`
  volume trigger are all size-parameterized, not size-gated), so a size-dependent failure has no
  producer today. Recorded as a **note with an owner**, not a `VP-*`: if the session ever wants a
  positive statement that a large body dumps whole, that is `MS-002`-tier work on the collapsed artifact
  (`pending_bytes`/`DUMP_FLUSH_BYTES` is where size actually *does* select behavior) and belongs to the
  plan, not to a boundary pass. 087 made the same disposition in `qa-plan.md` §QA Author Passes; this
  pass confirms it against the tree instead of citing it.
- **`T019.002` — what `count_tokens` normalizes before keying its stats digest.** `count_tokens` is not
  on the dump path and holds no `VP-*` at this milestone (the route's nearest `VP-*` are `MS-003`'s and
  `MS-004`'s and none covers keying). Where it went: 35 witnesses in
  `tests/test_count_tokens_parity.py` (hermetic, green here), the registry premise
  `CL:buster_ripper.session#session_id/key-derivation-input`, `quality-backlog.md:106` (the row that
  recorded T019's prescribed `normalize` call as a consequence worth stating) — now `[x]`, closed by
  T019.002 — and `DC-CR4-F-01`'s ledger entry. Two open rows remain in the backlog from this
  neighborhood and are **not** new findings here: `quality-backlog.md:105` (the accepted-risk
  over-broad `LookupError` name in three client-input arms) and `:112` (a record-accuracy correction to
  `:105`'s resolution text). Both carry owners and triggers; neither is `MS-001` closure material.

## Check 6: QA Milestone Closure — **BLOCKED (`unknown` ×4), not a pass**

**Steps 1–5 (handles, authorship, contract fidelity): all satisfied, on final bytes.**

- `MS-001` → `QAP-MS001` (matches `qa-plan.md` §QA Author Passes); scenarios whose `Required at` is
  exactly `MS-001`: `TC-100`, `TC-001`, `TC-107`, `TC-108` — and only those. Each names a `VP-*` whose
  `Observable after`/`Validate at` agree with `plan-summary.md` §Validation Point Matrix; all four
  milestone VPs have a selected scenario, so no orphan in either direction (table reproduced in
  §Handle Ledger).
- **Authorship**: the persistent gate module `tests/qa/test_qap_ms001.py` was supplied by
  [070-forward-qa](070-forward-qa.md), re-issued by 077, and is **byte-identical to that approved set at
  this HEAD** (`git log -1 -- tests/qa/` → `6508941`). Contract draft and inspection log present in
  070/077; 087 added no test byte. This pass authored **no** gate test. No rejection to `forward-qa` on
  authorship grounds.
- **Fidelity**: because the gate module and `harness.py` are byte-identical to the set 078 reviewed
  clause by clause, and because `git diff 6508941...HEAD -- qa-plan.md` shows only §QA Author Passes
  amendment text (no §Scenario Contracts, no §QA Gate Matrix change), the approved trigger/owner/
  visible-result/forbidden-result/lifecycle/telemetry contracts are implemented unchanged. What the
  cycle *did* change under them — the interception path each scenario drives through — is what step 6
  exists to re-execute, and step 6 is what the environment refuses.
- **Intermediate sweep (step 8)**: `ls tests/qa/` → `__init__.py`, `harness.py`,
  `counted_proxy_entry.py`, `tls_peer.py` + committed cert/key, `tls_probe_entry.py`, and exactly the
  three `test_qap_ms00{1,2,4}.py` gates. No `MS-001` intermediate lingers; the cycle added only
  permanent hermetic modules (`test_transport_parity.py`, `test_count_tokens_parity.py`) under `tests/`,
  which are suite members, not intermediates. Nothing was created by this pass, so nothing is newly
  deletable.

**Step 6 (execution): FAILED CLOSED — the required evidence cannot be produced on this host right now.**

Declared precondition (`qa-project` §Harness Declarations, Upstream): *"the live remote vLLM-compatible
endpoint… There is no local stub: **a gate run fails if that host is unreachable**."* Measured, not
assumed:

```
$ curl -s -o /dev/null -w "%{http_code}" http://hyperipper.local:30001/v1/models
000     → curl: (7) Failed to connect to hyperipper.local port 30001: Couldn't connect to server
$ dscacheutil -q host -a name hyperipper.local     → 192.168.50.132 (+2 fd11:… addresses)
$ ping -c1 hyperipper.local                          → 64 bytes from 192.168.50.132: time=0.251 ms
$ nc -z 192.168.50.132 30001   (3 immediate, then 43 periodic probes at 20 s from 20:02:59 to 20:17:08,
  plus a final direct re-probe at the close)                    → refused, every single time
$ nc -z 192.168.50.132 30000 / 22                     → OPEN        # host is up, the 30001 listener is not
$ this host's own address: 192.168.50.152            # the upstream is remote, so this is not a localhost mix-up
```

The 43 periodic probes are preserved verbatim at
[`../evidence/ms001-reboundary-091-upstream-probes.txt`](../evidence/ms001-reboundary-091-upstream-probes.txt),
so the count and the window do not rest on this report's prose.

The gate run over the full `QAP-MS001` module (`evidence/ms001-reboundary-091-gate-attempt.txt`):

```
8 failed, 3 passed in 17.79s
E   urllib.error.URLError <urlopen error [Errno 61] Connection refused>
E   RuntimeError: required live upstream http://hyperipper.local:30001 unavailable: …
```

- **All 8 failures are the harness's own fail-closed preflight; the transcript contains zero
  `assert`-level failures.** So nothing here indicts the product, and equally nothing here clears it:
  the gates never reached a request. The failure mode is the correct one — a dead endpoint cannot
  produce a pass, which is the property `US-009`'s arbitration and the 077 drain were built to preserve.
- The 3 that passed do **not** need the endpoint: `test_tc108_sensitivity_documented_line_pattern_distinguishes_drift`
  and `test_sensitivity_upstream_oracle_distinguishes_transport_failure_from_a_relayed_body` are the
  two process-free sensitivity witnesses, and `test_hostile_r1_unknown_dump_format_is_refused_and_writes_nothing`
  drives `startup_refused()`, where the CLI rejects the selector at parse time before anything binds a
  port. Recording them as passes would be accurate for *those* witnesses and useless for the VPs:
  **none of the four `VP-*` handles is carried by any of the three.**
- **No stub fallback was taken, and none is available to this pass**: `decisions.md` records explicitly
  that "whether the harness may fall back to a stub when the remote host is unreachable" is
  **deliberately NOT carried as authority** — "The user did not say; recorded as a pending factual
  claim, not resolved by inference." Substituting `:30000` (which answers 200) or any local peer would
  be exactly the laundering this boundary exists to prevent, and `UPSTREAM` is a hard constant at
  `tests/qa/harness.py:39` with no environment override — the harness is built so this cannot be
  quietly re-pointed.

### Handle Ledger (Check 6 step 7)

| Scenario | VP | Recorded |
|---|---|---|
| `TC-100` | `VP-T001-writes-deferred` | **unknown** — gate FAILED at `require_upstream()` before any request, so the trigger never ran |
| `TC-001` (documented write path) | `VP-T001-dump-failure-isolated` | **unknown** — same |
| `TC-001` (collapsed flush path) | `VP-T001-dump-failure-isolated` | **unknown** — same |
| `TC-107` | `VP-T002-default-encoding` | **unknown** — same |
| `TC-108` | `VP-T002-documented-format-preserved` | **unknown** — same |
| `TC-108` pattern-sensitivity witness | (supports `VP-T002-documented-format-preserved`) | **pass** — ran, proves the matcher can fail; carries no VP by itself |
| 077 drain witness | (supports `TC-001`'s oracle) | **pass** — ran, process-free |
| `R1` selector refusal | hostile tier | **pass** — ran, no endpoint needed |
| `R1` body-parse, `R2` containment, `R5` identity | hostile/identity tiers | **unknown** — all three failed at the same preflight |
| `TC-101…TC-112`, `TC-002`, `TC-003`, `TC-113`, `TC-114` | other milestones | `not_applicable` at `MS-001` |

**Every one of the eight is a pytest `FAILED` raised inside the test body (not a collection or fixture
`ERROR`)** — the transcript's summary line is `8 failed, 3 passed`. That distinction matters for the
disposition: a fixture error could hide a broken gate, whereas this is the harness's own deliberate
`raise RuntimeError(...)` firing on the first statement of each gate's `Proxy(...)`.

**Not inherited**: 087's `11/11` is scoped to `8dfa5af`; `T019.002` moved `normalize.py` and
`count_tokens.py` after it, so this pass re-ran rather than reused — and the re-run is the one that
could not complete.

## Check 7: Integration Suite Execution — step 0 PASS; steps 1–2 BLOCKED

**Step 0 — the full hermetic unit suite once on this boundary's snapshot** (the run that can catch a
regression in an unrelated module, which per-task Yield does not):

```
$ .venv/bin/python -m pytest -q            # HEAD 83de78c, clean tree
511 passed, 22 warnings in 9.73s
Required test coverage of 80% reached. Total coverage: 96.71%
```

Transcript: [`../evidence/ms001-reboundary-091-hermetic.txt`](../evidence/ms001-reboundary-091-hermetic.txt).
**511 @ 96.71 %** — the same figure and the same suite the CR3/T019.001/T019.002 passes each re-ran
(476 @ 95.91 % at 087 grew to 511 when `T019.002` added `test_count_tokens_parity.py`). Nothing to
bisect: zero failures, so no unrelated-module regression exists at this snapshot. Hermetic-ness proved,
not assumed: the default run collects zero `tests/qa` items and `tests/test_pytest_config.py` gates that
`norecursedirs` carries `qa` beside all ten pytest defaults.

**Scope derived FROM THE DIFF before running anything**: the cycle changed code in exactly one package,
`buster_ripper/` (5 of its modules), whose downstream dependents are its own consumers — the hermetic
integration layer (`tests/test_routes_integration.py`, `test_context_overflow.py`,
`test_upstream_answer_relay.py`, `test_transport_parity.py`, `test_count_tokens_parity.py`) and every
gate in `tests/qa/`, all of which read `buster_ripper.config`'s globals and drive the real binary.
Because `tests/qa/harness.py` and `messages.py`/`server.py` moved in this cycle, **all three** `tests/qa`
gate modules are in scope — the dispatch's reason for asking for `MS-002`/`MS-004` alongside, which this
pass also cannot satisfy for the same environmental cause (`MS-002`'s 15 and `MS-004`'s 6 gates all start
the real proxy against the declared endpoint; `MS-004`'s TLS gate is the one amendment-scoped exception
and it still needs the proxy to boot against its own loopback peer).
**Excluded, with the reason**: `buster_ripper_legacy.py` (tracked pre-refactor monolith, referenced by
nothing under `buster_ripper/`, out of scope by the plan's own §Task List "Not tasks"); `docs/`,
`invariants/`, `.agents/`, `.sessions/` (non-code). **Nothing downstream of a changed module was dropped,
and no scope was reduced after a failure** — the reduction here is not a choice, it is the endpoint.

**Steps 1–2 — the process-level integration layer: BLOCKED.** Not attempted blind: the `MS-001` module
was run first and its 8 errors are all `require_upstream()`, so `MS-002`/`MS-004` would fail identically
(their gates call the same `Proxy.__init__`), and running 21 more guaranteed-red gates to produce a
prettier transcript would burn two minutes of a shared resource to learn nothing. When the endpoint is
listening, the owed re-run is: `tests/qa/test_qap_ms001.py` (11), `test_qap_ms002.py` (15),
`test_qap_ms004.py` (6) — 32 gates, one snapshot.

**Shared-run rule**: cannot be applied to close Checks 6 and 7 together at this boundary, because the
one invocation that ran covers neither required scope completely. The session-close full suite remains
separately owed.

## Invariant Cross-Reference (scope: the cycle's affected packages)

Narrowed to what this cycle touched and what the milestone owns:
`invariants/buster_ripper/dump/invariants.md`, `invariants/buster_ripper/server/invariants.md`,
`invariants/buster_ripper/assumption-claims.md`'s `session_id`/`count_tokens`/`messages` rows, and the
registry header. No session-level `invariants.md` exists; all registry `active`/`wip` entries are zero —
the dump cluster is `planned`, per `invariants/index.md`.

- **Classification: Aligned, with one wording observation.** `EI:buster_ripper.session/session_id/derived_after_normalize`
  is a **session-carried `preserve`** (plan-summary.md:277), deliberately **not** a registry-active row.
  On `messages.py` it is untouched (byte-identical `normalize`-then-`session_id` order). On
  `count_tokens.py` it is **satisfied rather than regressed**: the digest input is `system`'s text
  (`CL:buster_ripper.session#session_id/key-derivation-input`) and the step the route now runs is the
  one step that writes it. Noted for the close rather than invented here: the registry row that will
  eventually carry this `EI:` should state the property as *"the step the digest input depends on runs
  before keying"*, because T019.002 makes the literal phrase "derived after `normalize()`" false of
  `/v1/messages/count_tokens` while keeping the property true. That is Authority Registrar projection
  work at the close — **this pass moves zero `invariants/` bytes**, per the triad rule.
- **`Override:` entries — all still fully applied, none exceeded**: `OI:…/output_layout`
  (`dump.py` unchanged), `OI:…/sequential_diff_chain` (`turn`/`prev_text` advance at append — file
  unchanged), `OI:…/turn_zero_is_full_body` (unchanged). The removal half of `OI:…/output_layout` is
  still absent, as `US-012` requires.
- 078's two registry observations stand and were **not** re-litigated: the stale
  `E:T002/dump/bulk-flush` gloss ("No caller until T003's triggers land") and
  `PI:…/format_set_is_closed`'s unwitnessed "exactly three values" clause. Both are rowed with owners
  (`QB:assumption-registry/t006-tls-referent-elevation`, and the format-set pin under `Forward`), both
  are records this cycle did not touch (`dump.py`/`config.py` unchanged), and `T019.002` neither opens
  nor closes either.

## Deterministic Closure (mandatory pre-question gate)

Applied to the one condition this pass would otherwise escalate — the blocked execution.

```text
CLOSURE: absorbed
Finding: 091-B-01 — `QAP-MS001`'s four `VP-*` are `unknown` at this boundary because the endpoint the
  project's QA contract declares as the only acceptable upstream for process-level gates is refusing
  connections, so the required trigger-to-outcome evidence cannot be produced on final bytes.
Required outcome: the milestone's closure evidence exists — the 11 `QAP-MS001` gates (plus the 21
  neighbor gates Check 7 asked for) run green on HEAD `f00216c`-or-later bytes against a listening
  `hyperipper.local:30001`, and `pass`/`fail` is recorded per `VP-*` handle by this gate.
Authority: (1) `US-009`/`US-010` — settled, "do not re-litigate or re-ask" in `handoff-context.md`
  §Notes: the QA harness runs against the live remote `http://hyperipper.local:30001`, never a stub;
  (2) `.agents/skills/qa-project/SKILL.md` §Harness Declarations: "a gate run fails if that host is
  unreachable"; (3) `decisions.md` §US-009 `Factual claims` annotation: stub fallback is deliberately
  NOT carried as authority ("The user did not say"), and its own stated consequence for this claim
  failing is "every MS-001 gate must re-scope to a stub or the milestone cannot be gated at all";
  (4) `codify-scope` — an environment outage is not a product defect, and repairing another host's
  service is not this session's scope; (5) precedent, in this session's own chronicle: `US-016`
  (2026-09-28) handled the identical outage ("blocked: DNS gaierror, Checkpoints 98/99") by **deferring
  the re-verdict until the host is reachable**, changing no scope and taking no fallback.
Admissible candidates: exactly one — hold the verdict and re-run this same gate when the endpoint
  answers. The remediation is operational (start/restore the listener on `192.168.50.132:30001`, the
  service the LAN operator owns), it is uniquely determined by the two authorities above, and the check
  that must be re-executed is already identified, scoped and scripted by this report.
Rejected candidates:
  - inherit 087's `READY-PASSING` 11/11 -> the dispatch forbids it, and it is factually byte-scoped to
    `8dfa5af`, which predates `T019.002`'s `normalize.py`/`count_tokens.py` bytes. `READY-PASSING` is an
    author-pass state, not a verdict; an author pass certifies the tests, not these bytes.
  - substitute `hyperipper.local:30000` (it answers 200 on `/v1/models`) or the committed loopback TLS
    peer -> replaces the declared endpoint with one no authority in this session permits; `US-009`
    names 30001 specifically and the peer exists only as the `MS-004` TLS-gate amendment. Also
    unbuildable: `UPSTREAM` is a module constant with no override, so it would mean editing the
    approved harness — which is the gate's own subject.
  - mark the four VPs `pass` on the strength of Checks 1–5 + the 511-test hermetic run -> the milestone
    is a process-level integration gate by contract (`qa-project` §Capability Declarations: "No
    in-process or mocked tier can prove…"); unit green is not trigger-to-outcome evidence.
  - `REJECT → implementation Forward` -> nothing in the diff failed. Zero assertion failures, zero
    hermetic failures, production surface byte-identical on everything the milestone owns. Rejecting
    would send Forward to fix an outage it cannot fix and would re-enter the retry cap for a root task
    that has no defect.
  - open a `QB:` row and re-approve 078's state -> re-approval of a boundary whose bytes moved after it
    is exactly what this pass was dispatched to prevent.
Risk shape: unchanged for production — zero code bytes, zero test bytes, zero invariant bytes moved by
  this pass. The only state that changes is that the boundary is explicitly not certified.
Deferred additive work: none. (The two "not a rerun" items are disposed above with their owners named;
  neither is closure material for `MS-001`.)
Owning route: `Forward` at the next dispatch, but the blocking action is **environmental, not code**:
  restore/confirm the `hyperipper.local:30001` listener (the LAN box at 192.168.50.132 — host is up, the
  port is not), then re-run Check 6 step 6 + Check 7 steps 1–2 on the current HEAD. No re-derivation of
  Checks 1–5 is owed unless production bytes move again; if they do, this pass is the baseline.
Question required: no — and specifically no question about the stub posture, which `US-009` already
  settled against fallback. The dispatcher needs the fact (endpoint down, gate unclosed), not a choice.
```

## Routing: owned factual claims

Enumerated every atomic entry under `Factual claims` in `decisions.md` carrying
`Verification: yield-phase:MS-001`: **exactly one** — `US-009` claim 2 (`decisions.md:791-792`),
currently recorded `HOLDS (2026-10-06, HEAD c7420ce)`. No `yield-phase:MS-001 PENDING` row exists
(`grep -n PENDING decisions.md` → two matches, neither a claim row: `:888` records a proposition as
deliberately *not* promoted, `:1246` is a governance note). Legacy-record inspection: `US-009` carries
the canonical field and every `handoff-archive.md`/`assumptions/claims.md` reference to it names this
same arbiter row, so no `MIGRATION-REQUIRED` return is owed.

**Adjudicated first-party on this snapshot, and it does not hold now:**

| Element | This pass's observation |
|---|---|
| "`http://hyperipper.local:30001` is reachable from the host that runs the QA gates" | **does not hold** at HEAD `83de78c`, 2026-10-08: `ConnectionRefusedError [Errno 61]` on 3 immediate probes, on all 43 periodic ones (20 s apart, 20:02:59–20:17:08) and on a final direct re-probe, while the same host answers `icmp` and accepts `22`/`30000` |
| "…and serves the vLLM-compatible API the proxy expects" | **unverifiable at this boundary** — the endpoint refuses connections, so no answer was ever observed. Not contradicted: `:30000` on that box answers `200` with a vLLM-shaped model list, which is consistent with the 30001 listener being down rather than the service being gone |
| the prior `HOLDS` (071, re-established 078) | **not overwritten**: it is a dated, HEAD-scoped observation of transient host state, and `decisions.md:794` says so in its own annotation. This pass adds a re-check note under it rather than destroying an arbitration that was true when made |

Ledger edit made by this pass (record-only, dated, non-destructive): a re-check annotation under
`decisions.md` §US-009 claim 2 stating the HEAD, the timestamp, the refusal, the `unknown` ×4, and the
evidence locator — so the next reader of the ledger does not find a bare `HOLDS` where a boundary pass
just recorded the opposite. `Verification: yield-phase:MS-001 HOLDS` is left intact, with this pass's
disposition named beside it. `Consequence` is unchanged: the row's own annotation already names the
consequence of non-holding, and this pass did not act on it (no stub, no re-scope).

This is the load-bearing claim for the milestone's gates. A non-holding load-bearing claim is a finding
under the User Authority Boundary, not an authoritative exception — hence **no `SEMANTIC-APPROVE`**, and
hence the closure packet above, which routes the outage to its operational owner rather than treating a
prior pass's green as credit.

## Findings

### [P1] The four `MS-001` validation points have no execution evidence at this boundary
- **Priority**: P1 (blocked-tier: harness precondition unmet, not a product or test defect)
- **Check**: 6 QA Milestone Closure (step 6) — propagated to Check 7 steps 1–2
- **Location**: `.agents/skills/qa-project/SKILL.md:19` (declared precondition); `tests/qa/harness.py:41-58`
  (`require_upstream`, fail-closed); consumed at `tests/qa/test_qap_ms001.py` via `Proxy.__init__:182`
- **What is wrong**: nothing in the tree. The required evidence is missing because the only endpoint the
  project permits for process-level gates is refusing connections, so `VP-T001-writes-deferred`,
  `VP-T001-dump-failure-isolated`, `VP-T002-default-encoding` and `VP-T002-documented-format-preserved`
  are all `unknown` on final bytes. 078's approval does not extend here (bytes moved), and 087's
  `READY-PASSING` is byte-scoped to `8dfa5af`, which predates `T019.002`.
- **Evidence**: `evidence/ms001-reboundary-091-gate-attempt.txt` — `8 failed, 3 passed`, every failure
  `RuntimeError: required live upstream http://hyperipper.local:30001 unavailable: <urlopen error [Errno
  61] Connection refused>`, **zero `assert`-level failures in the transcript**; the three passes are the
  process-free witnesses (`TC-108` pattern sensitivity, the 077 drain oracle, `R1` CLI refusal), none of
  which carries a `VP-*`. Connectivity: refused on 3 immediate probes, on all 43 periodic probes, and on a final direct re-probe while the
  same host pinged at 0.25 ms and accepted ports 22/30000.
- **Data path affected**: every gate's trigger leg — `pytest → Proxy.__init__ → require_upstream()`,
  and downstream `client → proxy :<free> → upstream :30001 → dump writer`
- **Reach**: **reached** — this is not a latent path; the run was executed and the outage is the
  observed state at this timestamp. (Entry path: `Proxy.__init__` called by all 8 endpoint-dependent
  gates.)
- **Disposition**: `CLOSURE: absorbed` — one remediation (restore the listener, re-run this gate),
  uniquely ranked by `US-009` + `qa-project` + the `US-016` precedent. No user question required about
  the posture; the fact is escalated because the dispatcher owns scheduling, not because an outcome is
  unranked.

### Notes (recorded, not findings — no owner owes them at this milestone)
1. **`T018`'s widened body-size domain reaching the dump writer is unwitnessed at every tier**, and the
   milestone's contract is silent on it by text (§T018/T019.002 adjudication). Its disposition is
   `tests/test_transport_parity.py::TestNoRequestBodyCap`, green here; the residual is a *positive*
   statement no `VP-*` asks for, with a size-dependent failure mode that has no producer in the source.
   If the session ever wants it, it is plan work at `MS-002`'s collapsed-artifact tier.
2. **`T019.002` satisfies a session-carried `preserve` invariant while making its literal wording false
   of one route** (§Invariant Cross-Reference). Projection fix for the close; zero `invariants/` bytes
   moved here.
3. Two open backlog rows neighbor this cycle and are neither new nor `MS-001` material:
   `quality-backlog.md:105` (+ its `:112` record-accuracy correction) — accepted-risk over-broad
   `LookupError` names in three client-input arms, rowed with an owner.
4. 078's two P3s (unwitnessed `format_set_is_closed` size clause; stale `E:T002/dump/bulk-flush` gloss)
   are unchanged by this cycle and remain with their owners.

## Verdict

**`BLOCKED-UPSTREAM` — no integration verdict issued for `MS-001` / `QAP-MS001` at HEAD `83de78c`.**

**Chronicle is deliberately not written by this pass.** Per this skill's own rule, a Yield Phase pass
persists its gate result and appends no `Delivered` entry — publication belongs to the pass that closes
the boundary. The precedent inside this session points the same way: the 2026-09-28 outage was recorded
in `chronicle.md` as a **`US-016` consequence** (a user decision plus a re-order), i.e. by the pass that
acted on it, not by the pass that first hit the dead port. `handoff-context.md` §Current Status — the
surface box-pulse and project-pulse actually read — now leads with this blocked state, the four
`unknown` handles, and the exact 32-gate re-run owed, so nothing is lost by leaving the narrative entry
to the close. If the dispatcher wants a chronicle line for the outage itself, it should say *"MS-001
boundary blocked on the declared upstream at `83de78c`; re-run owed"*, not anything resembling Delivered.

- Checks **1, 2, 3, 4, 5: PASS**, re-derived from source on final bytes, not inherited — with the
  milestone's owned surface (`dump.py`, `config.py`, `main.py`, `session.py`, `utils.py`, `README.md`,
  `serve_buster_ripper.sh`, `pyproject.toml`) and the whole of `tests/qa/**` shown byte-identical to
  078's approved state.
- Check **7 step 0: PASS** — hermetic 511 @ 96.71 %, zero failures, nothing to bisect.
- Check **6: BLOCKED**, all four `VP-*` `unknown`; Check **7 steps 1–2: BLOCKED**, so the owed re-run is
  32 process-level gates (`MS-001` 11 + `MS-002` 15 + `MS-004` 6) on the current HEAD against a
  listening `hyperipper.local:30001`.
- **Routing**: not a rejection to implementation Forward (zero failures attributable to the tree); not a
  `forward-qa` rejection (the gate module is the approved one, byte-for-byte); it is the environment
  precondition the QA contract declares, failing closed exactly as designed. The ledger's one owned
  factual claim for this milestone is adjudicated **does not hold now** and is annotated as such.
- **Not permitted by this verdict**: phase acceptance, child publication, assumptions/Yield Red dispatch
  as "post-boundary", or any `SEMANTIC-APPROVE` inheritance. 078's approval is superseded by bytes;
  nothing replaces it until the gates run.

## Artifacts written by this pass

| Path | What | Bytes |
|---|---|---|
| `iterations/091-yield-phase.md` | this report | new |
| `evidence/ms001-reboundary-091-gate-attempt.txt` | the 11-gate run on `83de78c` + upstream census (1891 lines; the `8 failed, 3 passed` summary is `:1891`, and the zero-assertion-failure claim is checkable by grepping the whole file for `^E  +assert` → no match) | new |
| `evidence/ms001-reboundary-091-hermetic.txt` | the 511-test hermetic run + coverage (45 lines; summary `:45`, `dump.py` 180/0/**100 %** at `:37`) | new |
| `evidence/ms001-reboundary-091-upstream-probes.txt` | the 43-probe connectivity window, verbatim (48 lines) | new |
| `evidence/INDEX.md` | two rows for the gate-attempt and hermetic artifacts | append |
| `decisions.md` | one dated re-check annotation under §US-009 claim 2 (`Verification:` cell left intact) | append |
| `handoff-context.md` | this boundary's status, the owed re-run, and the outage | update |

Zero production bytes, zero test bytes, zero invariant bytes, zero `qa-plan.md`/`plan-summary.md`
semantics moved. Nothing committed, pushed, squashed or rewritten by this pass.

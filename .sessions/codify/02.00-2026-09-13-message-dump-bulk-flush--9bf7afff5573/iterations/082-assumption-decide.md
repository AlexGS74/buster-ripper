# 082 · Assumption Decide — CR3 wave run-42 composition routing (`S16`–`S19`)

## Orchestrator Prompt (verbatim)

> You are a subagent spawned by another session.
> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
> Worktree root: /Users/alexhubitski/Projects/buster-ripper
> Mode: composition routing (CR3 wave run 42)
> Assumptions directory: /Users/alexhubitski/Projects/buster-ripper/.sessions/codify/02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/
> Registry root: /Users/alexhubitski/Projects/buster-ripper/invariants/
>
> Composition ids to disposition: `S16-body-limit.md` (`UNCLEAR`, no counterexample, two named missing contracts), `S17-stream-deadline.md` (`COUNTEREXAMPLE` CE-1 and CE-2), `S18-decode-failure-surface.md` (`COUNTEREXAMPLE` CE-1, plus MC-1/MC-2 missing contracts), `S19-async-witness.md` (`COUNTEREXAMPLE` CE-1, CE-2, CE-3). Apply the Reach Gate to each before deciding anything, then one disposition each, with the task/ledger row, counterexample, reach basis, evidence reference, and required proof named.
>
> `S18-CE-1` and `S19-CE-1/2/3` most likely land as `absorbed` against existing ledger entries rather than new work: `T019`'s blueprint already names the unknown-codec `LookupError` arm after row 401, and `T020`'s done-when was already rewritten after row 409 to forbid the config-only-wires shape and demand a negative control. Confirm that against what each amended task now actually asserts and say so plainly if it holds; if a composition shows something the amended blueprint still does not cover, amend the blueprint rather than opening a new task, and if it shows a third-party atom the ledger does not carry, hand that one back to the collector.
>
> `S17-CE-1` and `S17-CE-2` are the ones to think hardest about, because they are migration-introduced and `T018` already owns the stream deadline. The facts the composer established on current bytes: `server.py:66-68` arms only `ClientTimeout(total=600)`; when that budget fires during the header wait, aiohttp raises bare `asyncio.TimeoutError` (`client.py:622-624`, passed through unwrapped at `:793-798`), and the stream path's only handler at `messages.py:165` catches `aiohttp.ClientError`, whose exhaustive set does not contain it; nothing above catches it either, so the site's declared upstream-transport answer (502 + JSON + the `"Upstream error"` log) is never produced. The same number firing after the commit at `messages.py:161` is swallowed by `_relay_after_commit`'s `except Exception` and reaches the client as a 200 with a truncated SSE body. And the same number firing during a TLS handshake on an `https://` upstream is re-raised at `client.py:737-740` as `ConnectionTimeoutError`, which *is* a `ClientError`, so that one does get the 502. The pre-migration base is the authority for what is owed here: base `messages.py` caught `httpx.HTTPError` at the equivalent arm and httpx's `TimeoutException` derives from `TransportError` derives from `HTTPError`, so a header-wait timeout was answered as 502 + JSON error body + `"Upstream error"` log before the aiohttp migration moved it. That makes the escape a migration-introduced divergence with production reach on a reachable input (a slow upstream on a streaming request), not a theoretical one. `T018`'s done-when currently covers `--stream-timeout-seconds 0` and per-request override propagation — decide whether this belongs inside `T018` or is a separate owning task, and say what the repair must actually prove. Consider honestly whether a fix that makes the stream path's timeout handling match base is a small, contained change to the error-class arms or a substantial one that should split from this wave; the wave is already three members and `chat.py` is at its line ceiling, so an unbounded expansion here is a real cost, and a follow-up task with the evidence pinned is a legitimate disposition if the wave cannot carry it.
>
> `S16` is `UNCLEAR` rather than `COUNTEREXAMPLE`, with two named missing contracts: (a) `>=` makes the configured value the smallest rejected size, and the gate sits at the handler's *first* read, ahead of `normalize()` and `strip_and_cache_images()`, so under `--strip-images` an image-bearing body over 1 MiB gets 413 from aiohttp even though the forwarded body would be far smaller; (b) the 413 arrives as `text/plain` with an English sentence, against the repo's own `{"error": …}` JSON at every other client-facing failure site. Also disposition the composer's inert note that server-side auto-decompression means the cap bounds *decompressed* bytes, and `DEFAULT_MAX_DECOMPRESS_SIZE = 2**25` raises a non-`HTTPException` into the generic handler. Decide intended behavior against the base-parity authority (base aiohttp imposed no cap at all, and CR3-F-01 already established the unbounded-body path as base-authorized), and route what is genuinely owed rather than what is merely possible.
>
> Two record-keeping blockers the composers named, which are not findings and must not be settled by re-checking sound evidence: rows `230`, `330`, `331`, `68`–`72` and `137` carry no permanent `CL:` handle because they are repo-subject rows under this session's third-party-only scope ruling, and rows `402`/`403` plus six older verdicts carry an annotated line-2 header that the promoter cannot parse, already registered at `QB:assumption-registry/annotated-promotion-headers`. State plainly for the record which promotion each composition is therefore ineligible for and why that is a record condition rather than an evidence one.
>
> Constraint: third-party subjects only, `S1`–`S14` stay terminal. `FK-011` remains the only open Fork; a question is admissible only where materially different outcomes remain unranked after the base-parity and `G-003` authorities are applied, and the routing brief for `S17` and `S16` states the authority that settles each — if it does, no question belongs in this pass.

Result: **ASSUMPTIONS: 14 routed | 8 absorbed | 0 questions | 0 breaks | 1 malformed | 5 accepted | 0 capped**

Counting unit is the finding, not the seam file — fourteen adjudicated:

| # | Finding | Disposition | Durable record |
|---|---|---|---|
| 1 | `S16`-(a) what the number counts, where the cap measures | absorbed | `DC-S16-a-r1` → `T018` step 1 |
| 2 | `S16`-(b) the error envelope for a size rejection | accepted | §Accepted risks → §RM-010 |
| 3 | `S16` decompress note (`2**25`, non-`HTTPException`, decompressed-bytes cap) | accepted (1%) | §Accepted risks + §Elevated to user → §RM-011 |
| 4 | `S16` no-production-caller arms census (zero/negative cap, `post()`/`multipart()`, `content`, `clone()`, factory-less `Server`) | accepted | §Accepted risks |
| 5 | `S16` "four POST route classes" wording | accepted (inert wording) | §Accepted risks → `QB:assumption-ledger/row-s16-dimension-route-count` |
| 6 | `S17-CE-1` the armed `total` escapes the only arm, pre- vs post-commit | absorbed | `DC-S17-CE-1-r1` → `T018` step 2 (+done-when clause) |
| 7 | `S17-CE-2` one number, two semantic kinds, decided by network phase | absorbed | `DC-S17-CE-2-r1` → `T018` step 2 (+same clause) |
| 8 | `S17` sibling-arm residual + `N3`/`N4` (mine, from the reach check) | accepted (out of scope, blocks nothing) | §Accepted risks + §Elevated to user → §RM-010 |
| 9 | `S18-CE-1` unknown-codec `LookupError` escapes all three client parses | absorbed | `DC-S18-CE-1-r1` → `T019` step 2(c), zero change |
| 10 | `S18-MC-1` fate of a response-direction decode raise | absorbed | `DC-S18-MC-1-r1` → `T019` step 2(b), zero change |
| 11 | `S18-MC-2` the union's fourth arm has no class named | **malformed** | → collector, `QB:assumption-ledger/s18-s19-unrowed-class-atoms` |
| 12 | `S19-CE-1` `async with` succeeds, payload discarded | absorbed | `DC-S19-CE-1-r1` → `T020`, zero change |
| 13 | `S19-CE-2` sync arm and awaited arm need different kinds | absorbed | `DC-S19-CE-2-r1` → `T020`, zero change |
| 14 | `S19-CE-3` no member grants a second tick | absorbed | `DC-S19-CE-3-r1` → `T020` **done-when (6) amended** |

`8 absorbed` ↔ eight `DC-*-r1` ledger rows ↔ eight `## Planned Task Context` blocks (three under `T018`, two under
`T019`, three under `T020`). `1 malformed` is a re-extraction handoff, not a task, and no code waits on it.

## The two questions the brief asked, answered up front

**1. Do `S18-CE-1` and `S19-CE-1/2` land as absorbed against the amended blueprints? Yes — confirmed against what
those blueprints now actually assert, not against the brief's description of them.** `T019` step 2(c) is verbatim
"the three client parses (`messages.py:30`, `chat.py:31`, `count_tokens.py:22-25`) add `LookupError` to their
existing decode catches and answer each site's existing 400 shape", with counterexample (3) "client POSTs
`charset=bogus` to each of the three POST routes → 400 'Invalid JSON', never 500" and the run-40 amendment leaving
2(c) untouched ("steps 2(a)/2(c) unchanged (401 HOLDS for the request side)"). That is `S18-CE-1` exactly — same
three sites, same class, same settlement — so it is absorbed with **zero** blueprint change. `T020`'s amended
done-when items (1) per-tick value assertions, (2) one awaited-channel precondition, (3) the forbidden double list
and (4) the sibling idiom as the permitted construction cover `CE-1` (payload silently discarded) and `CE-2`
(sync `text` on the awaited arm): under both, `text()` is never awaited and `_kv_cache_usage` never moves, so the
tick-2 value assertion and the awaited-channel precondition both fail. Also absorbed with **zero** change.

**`S19-CE-3` does not land that way.** The amended done-when item (6) says the witness must "reach tick 2 without
a real `KV_POLL_INTERVAL` sleep (410's yield, the cancel-and-await arm 126 licenses)". CE-3 establishes that 410's
guarantee is *a yield, not a timer* — its own verdict limits it to "whether anything else executes depends on the
loop having other ready work at that iteration" — and the loop is parked on
`await asyncio.sleep(config.KV_POLL_INTERVAL)` at `compaction.py:57` with the interval `10`
(`config.py:64`). So the clause T020 relies on to license tick 2 is unsatisfiable **as written**. That is a
blueprint defect, not a new task: amended in place, one clause, naming the mechanism that actually makes the
interval elapse inside a test (see `DC-S19-CE-3-r1`).

**2. Does `S17` belong inside `T018`, and is an arm-level fix small or substantial? It belongs inside `T018`, and
the repair is not an arm change at all — which is also why the arm change must *not* be made here.** `T018`
step 2 puts `timeout=aiohttp.ClientTimeout(total=None)` on the streaming `session.post`. Per row 57 a request-local
`timeout=` **replaces** the session object rather than merging into it, and per row 396 a `None` component arms
nothing — so all four components are `None` and no deadline instance exists for that call. With nothing armed,
`CE-1`'s header-wait escape has no producer, `CE-1`'s post-commit truncation has no producer, and `CE-2`'s
handshake-scoped `ConnectionTimeoutError` has no producer either (`ssl_handshake_timeout` is forwarded from
`timeout.total`, row 396). The escape is deleted by removing the event, which is precisely what base did
(`buster_ripper_legacy.py:664` runs the stream path `timeout=None`). Adding `asyncio.TimeoutError` to the arm at
`messages.py:165` instead would restyle the symptom of a deadline the parity authority says should not exist on
that path, and would be dead code the moment step 2 lands. Rejected on that ground, not on convenience.

The genuinely substantial part — widening the arms at the three sites that *keep* an armed `total`
(`messages.py:297` non-stream, `chat.py:150` at 300 s, `passthrough.py:48` at 60 s) — is out of this session's
scope by `codify-scope` rule 2 and blocks no done-when, so it goes to `roadmap.md` §RM-010 rather than into a wave
that is already three members with `chat.py` at **300/300** raw lines (measured this pass; `roadmap.md:172`,
`handoff-context.md` §T015 capacity note).

## Three premises in the brief that do not survive the bytes — stated plainly

These are corrections to the routing input, not to the composers' verdicts. None of them changes a disposition;
two of them change the *severity* of what is owed, and all three are recorded so the next pass does not re-import
them.

1. **Base never answered an upstream timeout with 502, anywhere.** The brief's parity premise — "base
   `messages.py` caught `httpx.HTTPError` at the equivalent arm … so a header-wait timeout was answered as
   502 + JSON + `Upstream error` log before the aiohttp migration moved it" — is not what the base file contains.
   `buster_ripper_legacy.py` has **no** `except httpx.HTTPError` (grep: zero matches for `httpx.HTTPError`,
   `TransportError` or `"Upstream error"` in the file or anywhere in the repo outside `.venv`); its only `except`
   arms are `except Exception` at `:225, :314, :322, :344, :652, :747, :835, :885`, none of them around an upstream
   call. `proxy_messages` (`:639`), the chat handler (`:811`) and the passthrough (`:901`) issue their httpx call
   with **no** enclosing handler, so an `httpx.ReadTimeout` propagated to Starlette and became a 500. The
   502 + `{"error": "Upstream request failed"}` + `"Upstream error: %s"` arm is **branch-authored** (`T013`–`T017`
   work, witnessed by `tests/test_upstream_answer_relay.py:229`). Consequence: `S17-CE-1` is a gap between the
   branch's own arm and the class set aiohttp hands it, **not** a migration regression against base. That lowers
   what parity obliges here — it is exactly why the residual sibling-arm widening is roadmap work, and it is also
   why `T018` step 2 stays correct: parity obliges *no deadline on the stream path*, which is what step 2 does.
2. **A handler that raises out is not left unanswered — it is answered 504.** `S17-N3` says "the terminal status
   for CE-1's escape is unnamed by the verified set" and `S18-CE-1` says the member set "does not establish the
   status code the client finally sees". It does — by row **230**, an `S16` member with `VERDICT: HOLDS`:
   `verdicts/claim-230.md` records the dispatch at `web_protocol.py:513-523`, including
   `:518 except asyncio.TimeoutError` → `handle_error(request, 504)` and
   `:522 except Exception` → `handle_error(request, 500, exc)`, both sent through `finish_response` with
   `force_close()`; re-read against the installed bytes this pass (identical at `eacf5d33…`). So the honest
   observable for `S17-CE-1` is *504 with aiohttp's generated `text/plain` body and a traceback log* where the
   site declares *502 with `{"error": …}`*, and for `S18-CE-1` it is *generated 500* where the site declares 400.
   A wrong-status error turn, not an abandoned request. That is the difference between a P1-class parity break and
   an envelope-consistency gap.
3. **The brief's description of `T018`'s done-when does not match `T018`.** There is no `--stream-timeout-seconds`
   option, in `T018`'s blueprint or anywhere in the repo (`grep -rn "stream-timeout\|stream_timeout"` across
   `buster_ripper/`, `tests/` and this session's directory: **zero** hits; `main.py`'s option set has no timeout
   option at all). `T018` step 2's actual criteria are the `total=None` kwarg, the short-`total` slow-stream
   completion witness (red before / green after), the companion witness that the **non-stream** arm's inherited
   deadline is unchanged, the must-not-touch on `_relay_after_commit`'s logic and `POST_COMMIT_FAULTS`, and the
   permitted **docstring-clause-only** correction at `messages.py:180-185`. Dispositioning against a flag that
   does not exist would have produced the wrong answer, so the routed clauses above are the ones I checked
   against. `T018`'s step 2 does **not** today pin that the request-local object carries *no* armed component —
   which is the mechanism `CE-1`/`CE-2` turn on — so one done-when clause is added (see §Amendments).

## Reach Gate — per finding

Reach is taken from the production callers and consumers the composition cites, plus the sites I re-read
(`server.py`, `routes/messages.py`, `routes/chat.py`, `routes/passthrough.py`). No test or fixture was treated as a
caller. The Reach Gate does not apply to `S19`'s three counterexamples (false evidence — the gate's own exclusion
list, and `DC-CR3-F-05-r1`'s basis already says so).

| Finding | Reach | Basis |
|---|---|---|
| `S16`-(a) cap semantics + `--strip-images` ordering | **reached** today, **deleted** by the approved remediation | `server.py:119` constructs the app with no `client_max_size`; `web_request.py:670-680` gates the first read at `messages.py:29` / `chat.py:30` / `count_tokens.py:19` / `passthrough.py:14`, all four ahead of any shrink step. `T018` step 1 sets `client_max_size=0` and row 393 (`CL:aiohttp/client-max-size-falsy-skip`, HOLDS) skips the comparison entirely, so there is no configured number whose meaning is in question |
| `S16`-(b) 413 envelope | **unreachable** once `T018` step 1 lands | the only producer of the size rejection is the gate the falsy value removes; what remains (`400` for malformed framing, generated `500`/`504` for raised-out handlers) is a protocol-level answer base also served as plain text (§RM-010) |
| `S16` decompress note (`2**25`, non-`HTTPException`) | real but rare, **no obvious fix where it shows** → 1% category | needs a *request* carrying `Content-Encoding` whose decompressed output exceeds 32 MiB in one feed; the cause is aiohttp's server-side parser posture plus this proxy's header forwarding (`utils.py:20-26` keeps `Content-Encoding`), not the seam's cap. Base never decompressed a request body (`buster_ripper_legacy.py`: zero `gzip`/`decompress`/`Content-Encoding` references) |
| `S16` zero/negative cap, `post()`/`multipart()`, `content` bypass, `clone()`, factory-less `Server` | **unreachable** — "no production caller found" | the composer's own Notes census; `server.py:119` passes no value and no repo symbol writes `_client_max_size` (`web_app.py:116` is a read-only `ATTRS` property) |
| `S16` "four POST route classes" wording | **inert wording** | nothing reads the count; `server.py:127-136` is three `add_post` + one any-method catch-all |
| `S17-CE-1` | **reached** on the stream path today, **unreachable** after `T018` step 2 | `server.py:67` arms `total` only; `messages.py:137-141` passes no local `timeout=`; a >600 s header wait on an exposed route is real use. With `total=None` and 57's replace-not-merge, no `TimeoutHandle` exists for the call |
| `S17-CE-2` | **latent** → unreachable on the same landing | needs `--upstream https://…` (`main.py:62-63`, `config.py:13`) with the handshake as the binding constraint; the handshake scope is fed by `timeout.total` (row 396), so a `None` total disarms it too |
| `S17` sibling-arm residual (mine, from the reach check) | **reached**, out of scope, blocks nothing | `messages.py:265-297` (600 s inherited), `chat.py:139/150` (300 s), `passthrough.py:27/48` (60 s) each arm a `total` and catch only `ClientError`; base answered the same event with a 500, so parity is silent |
| `S18-CE-1` | **reached** — exposed surface, client-supplied header | `Content-Type: application/json; charset=<unregistered>` on `POST /v1/messages` (`server.py:127`), `/v1/chat/completions`, `/v1/messages/count_tokens`; nothing upstream of the read validates it |
| `S18-MC-1` | **reached**, and already settled by session evidence | an upstream ≥ 400 with an invalid-UTF-8 body at `messages.py:144`; `DC-claim-399-r1` names the fate ("`UnicodeDecodeError` (not an `aiohttp.ClientError`) → escapes … → proxy 500 with a traceback") and `T019` step 2(b) repairs it |
| `S18-MC-2` | **reached** input class, **unknown** outcome | `charset=` naming a registered non-text codec (`base64`, `hex`, `rot_13`) at the same three sites; no member claim, and no ledger row, states which class `bytes.decode(<that name>)` raises |
| `S19-CE-1`/`CE-2`/`CE-3` | Reach Gate n/a (false evidence), production reach of the *witnessed branch* noted anyway | `_kv_cache_usage` is read at `count_tokens.py:30`/`:51` and the loop is spawned at `server.py:22`/`:25-34`, so an inert witness misstates live behavior |

## Invariant Cross-Reference (all findings)

Sources: `invariants/index.md` (`## Handle Index`, `## Claim Index`, `## Composition Index`, §Source Sessions),
`invariants/buster_ripper/dump/invariants.md`, `invariants/buster_ripper/server/invariants.md`,
`invariants/buster_ripper/assumption-claims.md`. No session-level `invariants.md` exists (verified: no such file in
the session directory), so the carried `EI:` drafts are read from `handoff-context.md` §Notes as the registry's
§Source Sessions says to.

- Active/wip entries in the registry: **zero**. The `## Handle Index` holds 4 `OI:` provenance rows (dump layout,
  sequential diff chain, turn-0 body, shared client verifies by default) and **8 `planned` `PI:`** rows
  (`PI:T002/dump_format/*` ×6, `PI:T006/upstream_tls/*` ×2); all eight `Status: planned` and are ignored by the
  skill's Step 2 filter.
- Findings checked: 14 (plus the three record conditions, which are not findings). **Neutral: 14. Aligned: 0.
  Contradiction: 0. Ambiguous: 0. Overrides matched: 0.** Nothing in the registry governs a request-body cap, a client deadline, a charset catch
  list, a generated-error envelope or a test double, so no finding is an invariant contradiction and none binds an
  existing handle. `EI:buster_ripper.session/session_id/derived_after_normalize` binds `T019` **step 1** only
  (080's `Aligned` classification) and none of this pass's findings touch the ordering rule — recorded so the
  classification is neither extended nor weakened, exactly as 081 put it for `claim-399`.
- Permanent records used as **evidence inputs only**, never as authority: the `CL:` handles behind rows
  390/392/393/394/395/412 (`app-client-max-size-keyword`, `client-max-size-default`, `client-max-size-falsy-skip`,
  `client-max-size-enforcement`, `client-max-size-*`, plus `client_max_size` propagation), 57
  (`per-request-timeout-override`), 396/397 (`client-timeout-unset-total`, `client-timeout-payload-span`), 401/413/
  400/398/414/405/406/407/51/70 (the decode family) and S19's five members. Rows 230/330/331/68–72/137/402/403
  carry no handle; their *verdicts* are on file with `State: HOLD` and were treated as this session's settled
  evidence, which is what the ledger is for.
- Non-authoritative invariant **candidates** for Interview, recorded here and written nowhere else: (i) from
  `S17`/`S18` — *"a catch list names classes, not a semantic kind: an arm written as `except <library error>` does
  not own 'the upstream failed' until the deadline classes that library raises outside its subtree are accounted
  for"*; (ii) from `S19-CE-3` — *"a witness of a timer-driven loop is entitled to a second iteration only by a
  stated mechanism for the interval; a yield guarantee is not a clock"*. Both are candidates: this pass created no
  invariant and promoted nothing.

## Deterministic Closure Gate — packets

`CLOSURE: absorbed` ×7 — one admissible remediation each, all bound to an already-approved task step.
`CLOSURE: malformed` ×1 (`S18-MC-2` — the closure space is not establishable without the unrowed codec atom).
`CLOSURE: accepted` ×3 (Reach First: `S16`-(b), the `S16` decompress/1% family, the `S17` sibling-arm residual).
**`question` ×0**: after base parity (`buster_ripper_legacy.py:664` for the stream path; no 502-for-timeout anywhere
in base), `G-003` (proxy behavior on `/v1/messages` — the request fails *because of dump/buffer/flush behavior*, which
none of these findings is), `DC-CR3-F-01-r1`/`-r3`/`-r4`, `DC-CR3-F-05-r1`, `DC-claim-399-r1`, `DC-claim-409-r1`,
the approved blueprints and `codify-test-cost`, every finding left exactly one authority-ranked outcome.
`reserve-decision-record` was **not** invoked; `FK-011` remains the only open Fork.

Full packets are the seven `DC-*-r1` rows in `handoff-context.md` §Deterministic Closure Ledger (run-42 section);
the rejected candidates there are the ones the brief invited — arm-widening on the stream path, larger-constant
deadlines, config dials for the cap or the interval, real-sleep witnesses, stdlib clock patching, and
"route-out" vs "absorb" for `S16`-(a) (absorbed, following 080's own F-01 ruling and the 057/CR2 one-merge-unit
precedent, so the branch does not carry a reproduced 413 into review while its boundary is open).

## Amendments to pending blueprints (criteria only — no new task, no new wave)

1. **`T018` done-when**, one clause added: the stream-path witness must pin that the request-local object carries
   **every** component `None` (`total`, `connect`, `sock_connect`, `sock_read`), because 57's replace-not-merge is
   what makes `total=None` mean "no deadline at all", and a spelling that inherits the session's `total` — or that
   arms `sock_read` "to be safe" — leaves `S17-CE-1`'s escape live and `CE-2`'s two-kind split intact. This also
   discharges `S17-N2`: `None`/`0`/omitted-`total` was recorded there as "no production caller found", and after
   step 2 this call site is the one that spells it deliberately, so the witness is what keeps the deliberate
   spelling from degrading into an accidental one.
2. **`T020` done-when item (6)**, one clause corrected: tick 2 is reached by driving the interval the loop reads at
   `compaction.py:57` (`config.KV_POLL_INTERVAL`, `config.py:64` — a module attribute read per iteration), set to
   `0` for the duration of the driven task and restored with it; the parenthetical's claim that 410's yield
   licenses reaching tick 2 *without* a real interval change is struck, because 410's own verdict limits it to "the
   guarantee is 'allows other tasks to run', not 'a particular task will run'". Rejected instead: patching
   `asyncio.sleep` (a stdlib-wide patch where one attribute read suffices), a real 10 s sleep (test-cost, on a
   hermetic suite this boundary certifies at 455 tests), making the interval a parameter (`compaction.py` is
   `T020`'s must-not-touch), and a frozen-clock harness (a new mechanism no authority asks for).

`T019` receives **no** criteria change. `S18-MC-2`'s atom is contingent for its *totality claim*, not for its
repairs, and a `malformed` routing is not implemented by Forward — so the contingency is recorded in the task
block and in `QB:assumption-ledger/s18-s19-unrowed-class-atoms`, where the collector lane owns it, rather than as
an instruction to add a fourth name to a catch list on an unverified class.

## Record conditions (not findings, not evidence)

Which promotion each composition is ineligible for, and why the reason is a record condition:

- **`S16`** — ineligible for a `Composition:` (`CO:`) record on **two** counts: its verdict is `UNCLEAR`
  (eligibility needs `SAFE` or `COUNTEREXAMPLE`), and member row **230** has `Permanent: —`. Both survive this
  pass unchanged: routing an `UNCLEAR` seam's two missing contracts does not convert the verdict, and row 230's
  handle is the promoter's to create.
- **`S17`** — ineligible for `CO:` publication: members **330** and **331** carry `Permanent: —`, so "permanent
  handles for all member claims" fails. Its `COUNTEREXAMPLE` verdict would otherwise be promotable (as `S15`'s
  was). The composer says it in one line and I adopt it: *the blocker is handle elevation, not composition.*
- **`S18`** — ineligible for `CO:` publication on **three** counts: seven members without handles (68, 69, 71, 72,
  137, 402, 403), four members recording `Source Fingerprint: unavailable` (68, 69, 70, 71), and the annotated
  line-2 headers on 402/403 already registered at `QB:assumption-registry/annotated-promotion-headers`.
- **`S19`** — **eligible**, and not blocked by either condition: its header reads `PROMOTION: ELIGIBLE`, all five
  members (`126, 408, 415, 410, 411`) carry permanent `CL:` handles, and none of the six annotated-header verdicts
  is a member. Publication remains the promoter's act, and `Invalidates On` 7 already requires a recomposition over
  the landed `tests/test_compaction.py` bytes.

**Why this is a record condition and not an evidence one — including where the brief's reason is wrong.** The
propositions in all of those rows are settled: every one is `State: HOLD` with a verdict artifact on file, and the
compositions re-verified their digests against the working tree. Nothing awaits a re-check, and this pass re-checked
no sound evidence on unchanged bytes. The gap is *identity*: no promotion pass has ever covered those runs. Run 40's
promotion covered rows 390–411 and wrote "Promoted: 18 new `CL:` records … Not promoted: 4"; the older rows were
never in an in-scope set. The precedent for exactly this state is in `claims.md:577` — S3's and S5's compositions
recorded `PROMOTION: INELIGIBLE — no permanent member handles` for the same reason, and the seams later published
once handles existed. One correction, stated plainly because the brief asserts it: rows **230**, **330**, **331** and
**137** are **not** repo-subject rows — their `Subject` cells are `pkg:aiohttp@3.13.3`; rows **68–72**, **402**,
**403** are `lang:python`. The third-party-only scope ruling therefore cannot be the reason they lack handles (the
same ruling is why rows 405/406/407/414 *do* have `CL:python/…` handles, and why 51/390–395 have `CL:aiohttp/…`).
The reason is promotion coverage, which is the collector/promoter lane's bookkeeping, and it makes the dispositions
above *safer*, not weaker: they rest on HOLD verdicts that were simply never given a permanent name.

## Seam state handed on

- `S16`–`S19` Routing cells: `unverified` → **`absorbed`** (one token per seam, following `S15`'s precedent where
  two findings of different dispositions shared one cell). No `State` cell moved — none of them is composed
  post-implementation yet, and `resolved` requires the affected seam's post-repair composition to be `SAFE` or all
  its findings `accepted` (routing does not resolve).
- All four stay **open** until their post-task recompositions: `S16` and `S17` on `T018` (`Invalidates On` 3 names
  the config channel as `T018` step 1's own surface), `S18` on `T019` (plus the `MC-2` re-extraction it cannot be
  certified `SAFE` without), `S19` on `T020` (`Invalidates On` 7).
- **No seam is capped.** None of the four has been composed *after* a repair; these are their first compositions
  (each record says so: "no prior `S1x` record exists"). The recomposition allowance for each is still unspent and
  belongs to the post-`T018`/`T019`/`T020` bytes.
- **The wave's shape is unchanged**: anchor `T018`, members `T018` → `T019` → `T020`, one local Yield, per-member
  retry counters starting at 0, `T020` file-disjoint. Both amendments are criteria text on pending members, so
  `plan-adhoc-task` was not invoked and no wave was opened. `S1`–`S14` terminal, `S15` unmoved, and no `repo:` row
  or seam was created, rechecked or recomposed — the standing third-party-subjects-only ruling held.

## What this pass did not do

Routed only. Zero production or test bytes — every path this pass wrote is under
`.sessions/codify/02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/`. No working-tree query was issued: this
lane's Git boundary permits exactly one operation, `rev-parse HEAD` for the ledger's repository-state field, and no
`git status` claim is made anywhere in these records. Zero `invariants/` bytes, no `decisions.md` byte, no verdict file written or edited, no State cell moved, no
composition performed or judged `SAFE`, nothing promoted, no `CL:`/`CO:` handle allocated, no Fork reserved, no task
created, no claim re-extracted (the ledger carried **zero** `Routing: pending` rows entering this pass — verified by
grep — so there was no atomic row to route). Writes: this file; four `Routing` cells plus one changelog section in
`assumptions/seams.md`; `handoff-context.md` (one status bullet, three task-block amendments, seven ledger rows, one
routing section, five accepted-risk lines, two elevated lines); three `plan-summary.md` task-row pointers; two
`roadmap.md` rows; three `quality-backlog.md` rows.

Context budget: this pass adds ~230 lines to `handoff-context.md`, which 079 §Observation 4 already records as
over cap (~1050 against the 200-line cap) with the lossless trim into `handoff-archive.md` reserved for
`phase-boundary-finalize`/`session-squash`. Not this lane's act; disclosed rather than silently grown.

## Receipt (post-write hashes; repository state at routing time was `git-commit:5f7f9b9d43a50f571c497c0c52bf18406e77a567`)

| Artifact | Hash after this pass's writes | Note |
|---|---|---|
| `assumptions/claims.md` | `sha256:988bc3f9a8f096338c9cb3e181093db5838f3280c3b0fb7bd7b7c1b4f2b833ca` | **not written** — zero `Routing: pending` rows entering this pass (verified by grep), so there was no atomic row to route and no cell to move |
| `assumptions/seams.md` | `sha256:706cc1d3bdb31fd97aac16a812b67a44dcf997afe43bd6e106e90c46c58334eb` | four `Routing` cells `unverified` → `absorbed` (S16–S19); no `State`, `Permanent`, member or dimension cell touched; hash after the four cells and before the changelog section was `2aed2ea8…`, recorded in that section |
| `handoff-context.md` | `sha256:47c896782b84da3e371c580e6d85aad2e2a8b50335ffa760ce819a09215118f6` | one status bullet, three task-block amendments (eight blocks + two corrected clauses), eight ledger rows, one routing section, five accepted-risk lines, two elevated lines |
| `plan-summary.md` | `sha256:79435bf08dbe27df054fa187bfc224eefe454e37576e09da6918b97a0ee5703a` | three task-row pointers (`T018`, `T019`, `T020`); no task added, no queue order moved |
| `roadmap.md` | `sha256:a2c5c8975af11d1765f55ce0febd6c52b829e285d4a2da69a6ca0c1421a66547` | §RM-010, §RM-011; §RM-008/§RM-009 untouched and not widened |
| `quality-backlog.md` | `sha256:370b9977a86650fb4bae794da5a0c9102981029e3b6fe81244cc8a50d1b546de` | three rows: `s18-s19-unrowed-class-atoms`, `row-57-version-pin`, `row-s16-dimension-route-count` |

Every absorbed finding has exactly one ledger row, one task-step landing and one `Planned Task Context` block; every
accepted finding has one §Accepted risks line with its Reach verdict and citation; the two roadmap-routed items each
carry an §Elevated to user line with `Reported: —`. Nothing exists only in this response.

```text
ASSUMPTIONS: 14 routed | 8 absorbed | 0 questions | 0 breaks | 1 malformed | 5 accepted | 0 capped
```

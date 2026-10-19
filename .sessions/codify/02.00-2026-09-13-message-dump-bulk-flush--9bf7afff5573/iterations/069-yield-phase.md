# Iteration 069 — Yield Phase (MS-002 / QAP-MS002 — RE-VERDICT)

## Orchestrator Prompt (verbatim)

```text
Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
Milestone: MS-002
QA Pass: QAP-MS002
Target base: main
Target object: 72e3a1c6861bf0d8de6317adf52289e5f6cfa4e1

This is the owed RE-VERDICT of MS-002's integration gate, not a first run. MS-002 was rejected at
iteration 035 and its remediations (T003.001, T007, T008, T008.001) have since passed local Yield,
so the milestone's constituents are all accepted and the gate needs an independent verdict at
current HEAD.

The rejection's root cause was environmental, and it is why this pass is only now possible:
QAP-MS002's gates require the real upstream at http://hyperipper.local:30001 and a run is not a
pass while that host is unreachable (US-009/US-010 — no stub may substitute). The host was down for
days; a very recent drain pass observed all 15 gates passing against it. Independently confirm
reachability yourself and record what you observe — if the host is unreachable now, say so plainly
and return the gate as blocked on reachability rather than crediting a baseline of connection
errors as a pass.

Re-derive the milestone's validation points from qa-plan.md on this branch state, and note for the
record what has moved since 035: the T013 header-parity wave rewired all five filter sites through
the shared utils helpers and split _stream_single_request at the prepare() commit, and the T014
image-slice wave rewrote image.py's detector scope, placeholder keying and cache retention.
```

## Invocation Contract Verification

- Session handle `02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573` verified against
  `session-identity.json` (`session_id: 02.00-2026-09-13-message-dump-bulk-flush`,
  `session_uid: 9bf7afff-5573-402a-b102-827423a1dded`) — **MATCH**. Files read directly; nothing globbed.
- Full milestone gate: `Milestone: MS-002` + `QA Pass: QAP-MS002` + `Target base: main` + full
  `Target object` — all present, contract satisfied.
- **BASE-SYNC gate: PASSES.** Latest base-sync report is `BASE-SYNC: unchanged`
  ([054-01-forward](054-01-forward.md), target `main @ 72e3a1c6…`); no newer one exists (grep over
  `iterations/*.md`), and [056](056-yield-phase.md) §Invocation Contract and
  [058-01-forward](058-01-forward.md) §Evidence-Mode Step 1 each re-asserted it. Re-verified here at
  HEAD `bd167958e0e4ba321dd080eb5703960d126f281d`: `git rev-parse main` = the target object exactly;
  `git merge-base --is-ancestor 72e3a1c6… HEAD` → **true**; worktree clean before and during both runs.
  Every correctness-required integration repair passed local Yield (T003.001
  [043-01](043-01-yield.md), T007 [044-01](044-01-yield.md), T008 [045-01](045-01-yield.md),
  T008.001 [046-01](046-01-yield.md); then T009/T010/T010.001, T011, T012, T013, T014 all
  `accepted-local`). No final-phase preventive merge work is deferred (MS-004 is the final phase and
  closed at 056).
- Constituent tasks: MS-002 = **T003** (Milestone Matrix, `plan-completed.md`), accepted local Yield
  ([032-01-yield](032-01-yield.md)), with its four remediation subtasks `accepted-local`.
  `QAP-MS002` authored ([034-forward-qa](034-forward-qa.md)) and repaired
  ([048-forward-qa](048-forward-qa.md)) — the invocation's `QAP-*` matches this milestone's row in
  the §QA Author Passes matrix.
- **Upstream reachability — the condition that blocked this re-verdict: REACHABLE, verified
  independently (evidence in §Check 6).** This run is not a 502-baseline credit.
- Review scope: cumulative diff `72e3a1c6…HEAD`, with the milestone's owned surface re-derived
  because **every file it gates has moved since the rejected run.** `git diff --stat f803b04..HEAD`
  (035's HEAD → here) over `buster_ripper/ tests/ serve_buster_ripper.sh pyproject.toml README.md`
  = 38 files, +4419/−512. Per-file: `dump.py` ±50, `image.py` ±154, `routes/messages.py` ±87,
  `routes/chat.py` ±87, `routes/count_tokens.py` ±78, `main.py` ±67, `backend.py` ±39,
  `server.py` ±23, `utils.py` ±11, `config.py` ±4, `routes/passthrough.py` ±10, plus the gate module
  `tests/qa/test_qap_ms002.py` ±71 and `tests/qa/harness.py` ±77. Checks 1-5 below are re-derived
  against this tree, not replayed from 035.
- Codex findings: none supplied at dispatch.

### Method note recorded for the corpus

This repo sets `diff.external = ~/.dotfiles/bin/git-colordiff`, which emits ANSI escapes **before**
the `+`/`-` gutter marker. A first-pass enumeration of `git diff … | grep -E '^\+def '` therefore
returned **zero added definitions** — a false negative that would have made Check 1 vacuous. Re-run
with the escapes stripped (`sed 's/\x1b\[[0-9;]*m//g'`), the real set is 21 added / 2 removed
production definitions. Any future pass that anchors a gutter grep on `^` must strip first.

## Check 1: Caller Verification — PASS

Production entry points added/removed since 035 (`buster_ripper/`), with call sites grepped in
production and in `tests/`:

| Symbol | Home | Production caller | Verdict |
|---|---|---|---|
| `_read_body` | `routes/count_tokens.py:16` | `:78` in `handle_count_tokens` | WIRED |
| `_prepare_headers` (rewritten) | `routes/chat.py:98` | `:36` in `handle_chat` | WIRED |
| `_carries_token` | `routes/chat.py:124` | `:110` inside `_prepare_headers` | WIRED |
| `_relay_after_commit` | `routes/messages.py:163` | `:155` in `_stream_single_request`, immediately after `await response.prepare(request)` (`:154`) | WIRED |
| `_note_message_start` | `routes/messages.py:193` | `:182` in the relay loop | WIRED |
| `_relay_chat_stream` | `routes/chat.py:174` | `:151` dispatch arm on `body.get("stream")` | WIRED |
| `_raise_chat_overflow` | `routes/chat.py:164` | `:154` | WIRED |
| `_classify_models` | `backend.py:74` | `:60`, inside `detect_backend` | WIRED |
| `_nudge_tokens` / `_nudge_trigger_name` | `routes/count_tokens.py` | `:96` / `:59` | WIRED |
| `_scoped_contents`, `_user_content`, `_image_key`, `_cache_source` | `image.py` | `:69`+`:192`, `:58`, `:140`, `:142` | WIRED |
| `_render_tools`, `_system_text`, `_content_text` | `dump.py:49/56/63` | `:80`, `:82`, `:88` in `_render_body` | WIRED |
| `_apply_cli_config`, `_set_skip_tls_verify` | `main.py:36/30` | `main.py:139` / `@click.option` callback `:65` (T006, also certified at 056) | WIRED |
| `has_images` (rewritten) | `image.py:61` | `routes/messages.py:40` (live), `normalize.py:212` (guarded off, see Check 3) | WIRED |

The milestone's own surface, re-derived rather than inherited:

- `dump.dump_turn` ← `routes/messages.py:59`, inside `if config.DUMP_DIR is not None:` (`:56`) with the
  try/except swallow-all (`:57-61`) — the `E:T001/dump/dump-caller` referent's contract intact.
- `dump.flush_all` ← **three** producers, all in production: `dump.py:206` (append-time volume
  check), `dump.py:133` (`poll_flush` loop body), `server.py:112` (`_final_dump_flush`).
- `dump.poll_flush` ← `server.py:45` `asyncio.create_task(dump.poll_flush())` in `spawn_dump_flush`,
  itself called at `server.py:81` inside `on_startup`; registered at `server.py:120` from
  `create_app()`, called at `main.py:188` before `web.run_app` (`:189`).
- `_cancel_background_task` ← `server.py:86-87`; `_final_dump_flush` ← `server.py:88`.
- CLI: `--dump-format` (`main.py:80-82`), `--dump-flush-interval` (`:86-88`), `--dump-flush-bytes`
  (`:91-93`) — each `expose_value=False` + its own callback, so `main()` gains no parameter.

**Removed callbacks/hooks with no replacement:** none. The two removed definitions
(`_prepare_headers`'s old body, `has_images`) were rewritten in place with callers intact.

**Zero-production-caller surfaces examined and dismissed as already owned (not re-litigated):**
`get_cached_image` (`image.py:203`) and `clear_session_cache` / `make_data_url` (`:252` / `:226`)
have test-only or no callers. `get_cached_image`'s absent reader is a **settled** disposition —
`RB-003` / `FK-007` option 1 / the Anchor Map's "Read path — deliberately NOT an anchor row", and
T014's own yield recorded the census (`handoff-context.md` §T014: "the placeholder text has no
production reader today (`get_cached_image` has zero production callers, settled deliberately by
RB-003/FK-007 option 1/the Anchor Map)"). `clear_session_cache`/`make_data_url` are unchanged since
035 and fall under the pre-existing dead-surface backlog row (`QB` line at `quality-backlog.md:40`).

## Check 2: Migration Parity — PASS

Three replaced paths named by the dispatch, compared against the code at 035's HEAD (`f803b04`),
not against the forward reports.

**(a) T013 — five header-filter sites rewired through `utils`.** Independent census at current HEAD:

```
grep -rn "forward_headers(\|response_headers(" buster_ripper/ --include='*.py' | grep -v utils.py
  routes/chat.py:103        forward_headers   (client request headers -> upstream)
  routes/messages.py:64     forward_headers   (client request headers -> upstream)
  routes/passthrough.py:17  forward_headers   (client request headers -> upstream)
  routes/messages.py:285    response_headers  (upstream response headers -> client)
  routes/passthrough.py:41  response_headers  (upstream response headers -> client)
grep -rn "headers.items()" buster_ripper/routes/          → (empty)
grep -rn "skip_keys\|_HOP_BY_HOP" buster_ripper/          → utils.py:12/23/39 only
```

Direction pairing: **three `forward_headers` sites** (outbound-to-upstream, built from the client's
request headers — `chat.py:103`, `messages.py:64`, `passthrough.py:17`) and **two
`response_headers` sites** (outbound-to-client, built from the upstream's response headers —
`messages.py:285`, `passthrough.py:41`); this is T013's "3-egress / 2-ingress" census, named by
helper here because "ingress/egress" is used for opposite ends in different reports. No inline
skip-set survives; one policy source. Parity of
each replaced site: the request-header sites formerly dropped `{host, content-length, connection,
transfer-encoding}` and now drop that set ∪ `_HOP_BY_HOP` — a strict superset, which is the base
(`buster_ripper.py:178`) behavior the wave restores, not a loss. The response-header sites formerly
dropped `[connection, transfer-encoding, content-length]`, now `content-length` ∪ `_HOP_BY_HOP` —
same direction. `chat.py`'s route-local empty-`Bearer` rule keeps its old side effect (delete the
value-less bearer line) and adds the credential-preserving selection that 066-01 required; 067-01
verified zero credential regressions against base.

**(b) T013 — `_stream_single_request` split at the `prepare()` commit.** Side-effect inventory of
the pre-split inline relay loop vs `_relay_after_commit` + `_note_message_start`:

| Old side effect | New counterpart | Verdict |
|---|---|---|
| `async for chunk in resp.content.iter_any()` | `:181` | ✓ |
| first-`message_start` detection (`find_message_start`) | `_note_message_start` def `:193`, called `:182` | ✓ |
| `capture["done"]`/`capture["input_tokens"]` bookkeeping | `state["done"]`/`state["input_tokens"]` `:201-202` | ✓ |
| `update_session_stats(sid, input_tokens, ttft_ms)` | `:204` (same arguments, same order) | ✓ |
| verbose `session %s: tokens=%d ttft=%.0f ms` log | `:206-208` (same format string) | ✓ |
| `await response.write(chunk)` per chunk | `:183` | ✓ |
| `await response.write_eof()` | `:184` | ✓ |
| `except aiohttp.ClientError → web.json_response(502)` covering the **whole** body | kept at `:158` for the pre-commit arm only; post-commit failures now log and return | **intentional omission**, documented in `_relay_after_commit`'s docstring (returning a second `Response` after `prepare()` was the defect G-1 closes — a second head inside a committed body) |

`state["input_tokens"]` is written and never read — verified to have been equally unread at
`f803b04` (`capture["input_tokens"]` at `:166`, no read). Carried dead state, not a loss introduced
by the split; recorded as an observation (§Observations O-1), not a finding.

**(c) T014 — `image.py` detector scope, placeholder keying, cache retention.** Scope widened
(`has_images` now scans every user message via `_scoped_contents`), keying changed to content-derived
(`_image_key`: `sha256(json.dumps(source, sort_keys=True, separators=(",",":")))[:12]` → `img_<12>`),
retention added (`_cache_source` keeps `first_seen` on a write onto an already-expired entry; the
tail sweep lives inside `cleanup_expired_images`). Parity direction: the old behavior (first-user-message
only, slot-named key, TTL as idle-timeout) is **deliberately displaced** by the wave's three members,
each accepted-local at [068-01-yield](068-01-yield.md) with its own witnesses (W1-W7, the recomputed
`img_04b45e9ae5ea` pin, the at-cap converse). Nothing in the old side-effect set was dropped
silently: the writer, the LRU eviction call and the placeholder block shape
(`{"type":"text","text":"[image:img_…]"}`) all remain.

**(d) The milestone's own migration, re-checked because `dump.py` moved.** Two deltas only since
035: (i) the `_render_body` 3-gray extraction (`_render_tools`/`_system_text`/`_content_text`) —
statement-for-statement equivalent, and byte-identical output is **measured**, not assumed: this
run's TC-105 produces `write_bytes` 482762 / 282647 / 442217, the same three figures 034 and 035
recorded; (ii) T009's `pending_bytes` now counts `len(current_text.encode("utf-8"))` rather than
`len(current_text)` — the volume measure moved from code points to UTF-8 bytes, which is the plan's
own settled discretion (`E:T003/dump/flush-trigger-parameters`: "pending rendered-text bytes";
`plan-summary` §Notes: "rendered-text UTF-8 bytes"). No documented-writer side effect was lost:
`_write_documented` still drives `_write_full_turn` → `_write_diff_turn` → `_append_index_line`, and
`turn/prev_text` still advance at **append** time (`dump.py:108-109`), never at flush.

## Check 3: Data Flow Completeness — PASS

Traced by reading code at current HEAD, producer → consumer:

`POST /v1/messages` → `handle_messages` (`messages.py:26`) → `request.json()` typed-stop (`:28-30`)
→ `normalize(body, session_id="")` (`:36`) → `sid` (`:37`) → optional image strip (`:40-42`)
→ optional preserved-thinking (`:44-52`) → **`dump_turn(body)`** (`:57-61`) → `_resolve_format`
(`dump.py:139`) → `_lookup_or_create_state` → format branch → collapsed: `_append_record`
(`:200-207`: append, `pending_bytes += utf-8 bytes`, threshold → `flush_all`) / documented:
`_write_documented` → three producers into **one** consumer write path
`flush_all` (`:112`) → `_flush_session` (`:285`) → `_flush_state` (`:295`, one
`path.open("a")` + one `write`) → artifact `DUMP_DIR/<sid>/<fmt>.log` → human reader (RB-003:
deliberately no in-code reader).

- **The T013/T014 restructures sit outside this chain's data path and do not touch it.** The dump
  call happens at `:59`, before `forward_headers` (`:64`) and long before `_stream_single_request`'s
  `prepare()`/relay split (`:154-155`), so the split cannot drop, reorder or corrupt a record. The
  only image-path interaction is ordering: stripping precedes dumping, so a placeholder record is
  what lands — correct by construction, and gated only with `STRIP_IMAGES` off (its default).
- Image-path single-consumer check: `normalize()` is called from exactly one site in production
  (`messages.py:36`) and it passes `session_id=""`, so `normalize.py:212`'s image branch is dead
  there; the live image path is `messages.py:40`. No double-strip, no second writer of the body the
  dump reads.
- Volume trigger reads `state.pending_bytes` **after** the append (`:204-206`) — no off-by-one, no
  pre-append evaluation. Collapsed-path only: `_append_record` is unreachable for `documented`, so
  nothing accumulates and the check never fires there.
- Timer: `poll_flush` flushes then sleeps, per-iteration `except Exception` + `log.exception`
  (`:128-135`); the retained batch is what a later tick writes (`_flush_state` clears only on
  success). Cancellation lands at the `await asyncio.sleep` and is handled by
  `server.py:87`'s await.
- Shutdown: `on_cleanup` order is cancel(kv) → cancel(dump) → `_final_dump_flush()` →
  `session.close()` (`server.py:84-89`); `_final_dump_flush` cannot raise (`:109-114`).
  The T010.001 rewrite of `_cancel_background_task` isolates an inner task's ordinary failure
  (`except Exception` → log) so a dead metrics poll can no longer skip the flush, while still
  propagating a *fresh outer* cancellation (the `cleanup.cancelling()` delta). That propagation
  means `_final_dump_flush()` is skipped when the cleanup task itself is cancelled mid-await — the
  deliberate choice T010.001's arbiter made (042-01-yield H2: "fresh parent cancellation during
  child await conflicts with swallowing child cancellation, because count delta must propagate").
  **Not re-opened here**; the process-level consequence is exactly what TC-110 measures under real
  SIGINT and real SIGTERM, and both pass.
- No queue or channel has an unread end; the only spawned tasks are `dump_flush_task` and
  `kv_monitor_task`, both created (`server.py:45`, `:33`) and awaited at cleanup (`:86-87`).
- Write-site inventory unchanged in kind: `dump.py` `write_text` ×2 + `open("a")` ×2 (index,
  artifact); `mkdir` sites are not write-bearing; SQLite is outside the settled unit. Measured by
  the instrumented counter, not assumed (TC-105: 2 / 2 / 12).

## Check 4: Consumer Contract Tracing — PASS

- **`SessionState.fmt`** — assigned only from `_resolve_format()` (`dump.py:98-100`), which raises
  `ValueError` for anything outside `config.DUMP_FORMATS` (`:139-144`). Consumers `_flush_state`
  (`:297` documented → early return; `:299` `_artifact_path(state, state.fmt)`) and
  `dump_turn`'s three-arm branch cover the closed set; `click.Choice(list(config.DUMP_FORMATS))`
  (`main.py:80`) constrains the CLI arm. A future fourth value would take the self-contained shape —
  the same P3-tier extensibility note 035 recorded, already carried by the F3 family in
  `quality-backlog.md`, unchanged and not re-opened.
- **Coalesced data (the flush batch)** — one `flush_all` iteration drains one `_sessions` entry per
  session (`:121-122`); `_flush_state` joins only that session's `pending`, whose records were all
  built under one `state.fmt`, so dispatch values are homogeneous within the merge and arrival order
  is list-append order. Witnessed end-to-end by TC-102 (`ids == [0,1,2]`) and by the identity
  near-miss gate, which proves `(sid, format)` does not merge formats.
- **`state` dict in `_stream_single_request`** (`{"done","input_tokens","sid","started"}`) — keys
  are set at creation and read only by `_note_message_start`; no consumer branches on a value the
  producer can fail to supply. `sid`/`started` are new relative to the pre-split `capture` and are
  populated at `:126`, before any handoff.
- **Placeholder blocks (`T014`)** — `_replace_image_with_placeholder` emits
  `{"type":"text","text":"[image:img_…]"}`. The one downstream surface in this milestone is
  `dump._content_text`, whose dict arm is `b.get("text", json.dumps(b))` — a `type: "text"` block
  always carries `text`, so the renderer never falls through to a JSON dump of a placeholder, and no
  dispatch on `type` exists in the dump path. Latent only under `STRIP_IMAGES=1`, which no MS-002
  scenario turns on and whose read path is the settled `FK-007` gap.
- **Non-UTF-8 / unparsable bodies** — the typed stop at `:28-30` returns before `dump_turn`, so the
  producer never creates state for a body it could not parse. Asserted at process level by the R1
  hostile gate (no session directory appears: `os.listdir(dump) == []`).

## Check 5: Plan Fulfillment — PASS

T003's seven ADDED done-when items, re-verified present at current HEAD:

| Item | Evidence at `bd16795` |
|---|---|
| `config` globals | `DUMP_FORMATS`/`DUMP_FORMAT="documented"` (`config.py:27-28`), `DUMP_FLUSH_INTERVAL=10` (`:36`), `DUMP_FLUSH_BYTES=4 MiB` (`:37`) |
| CLI options through their own callbacks | `main.py:80-93`, all `expose_value=False`; `main()`'s parameter list unchanged |
| append-time volume check, collapsed-only | `dump.py:202-207` |
| `poll_flush` loop body (flush-then-sleep, per-iteration catch) | `dump.py:125-135` |
| spawner + app key | `server.py:37-45`, called `:81`, key `dump_flush_task` |
| `on_cleanup` cancel → flush → close | `server.py:84-89` |
| tests | hermetic suite **407 passed @ 95.75%**; `dump.py` at **100%** coverage |

**REMOVAL criteria** (the axis that catches relays):
- T003 owes no removal — it is the milestone that gives a path which previously had **no trigger**
  its three callers. The pre-T003 callerlessness of `flush_all` (MS-001's declared intermediate, ⚠022)
  is closed: three production callers, verified in Check 1.
- **T003.001 owed removal of the missed decode class**: `grep -rn "except json.JSONDecodeError"
  buster_ripper/` returns exactly one line, `count_tokens.py:20`, and that is the *deliberate*
  syntax-fallback arm inside `_read_body`, paired with `except UnicodeDecodeError` → typed 400 at
  `:22-25`. `messages.py:30` and `chat.py:30` both catch the union. No client-body parse site is
  left catching only the syntax class. The rejected P1 is closed at the source, not papered over.
- Ownership per plan: buffer + flush + artifact identity live in `dump.py`; process lifecycle lives
  in `server.py`; the route stays a pure consumer behind its swallow-all guard. No producer logic
  leaked into `routes/`, and the T013/T014 waves added none (`dump.py` imports nothing new).
- Registry cross-check (`invariants/buster_ripper/dump/invariants.md`): all six `PI:T002/dump_format/*`
  statements hold on this snapshot — `format_is_per_artifact` (TC-102 + identity gate),
  `format_does_not_change_write_count` (F=2 for **both** collapsed formats vs 2N=12),
  `default_format_is_documented` (TC-105's documented half ran with **no** `--dump-format` and
  produced the documented layout — a live default-path observation), `format_set_is_closed`
  (`_resolve_format` hard error + `click.Choice`), `one_write_path_per_turn` (never `2N+F`: witness C
  excluded by the listing assertions), `documented_format_is_retained` (three files, `index.txt` at
  N lines, `turn_000.json` parses). The three `Override:` blocks are all `Decision: preserve` with
  their MS-002-side verification satisfied here (TC-102/103/104); their MS-001-side clauses
  (TC-107/TC-108) remain owed at MS-001 — see §Adjacent debt.
- `server` cluster: `E:T003/server/shutdown-flush-seam` intact; the two `PI:T006/*` TLS invariants
  and `OI:.../shared_client_verifies_by_default` are MS-004's, certified at 056, and this pass
  re-ran their gate green (6/6) rather than assuming it.

## Check 6: QA Milestone Closure — PASS

**Validation points re-derived from `qa-plan.md` on this branch state** (§QA Gate Matrix + §QA
Author Passes + §Scenario Contracts; cross-checked against `plan-summary.md` §Milestone Matrix and
§Validation Point Matrix). Scenarios with `Required at = MS-002`, all with `Earliest task = T003`
and matching `Observable after`/`Validate at` in the VP matrix (the 022 corrections included):
TC-101, TC-102, TC-103, TC-104, TC-105, TC-106, TC-109, TC-110, TC-111, TC-112 — **ten scenarios,
ten VPs, no orphan in either direction**. The author-pass row for `QAP-MS002` names exactly this set.

**Preflight observation — recorded independently, as dispatched (2026-10-06):**
- `dscacheutil -q host -a name hyperipper.local` → `192.168.50.132` (+ two `fd11:…` addrs).
- `ping -c 2 hyperipper.local` → 2/2 received, 0.0% loss, 0.203/0.243/0.284 ms.
- `curl --noproxy '*' …/v1/models` → **HTTP 200**, 488 bytes, a real model list.
- `harness.require_upstream()` executed from the committed harness → **PASS in 0.02 s**.
- Through the proxy, a gate-shaped request gets **HTTP 404**
  `{"type":"error","error":{"type":"NotFoundError","message":"The model \`qa-gate-model\` does not exist."}}`
  — an *upstream-authored* error. The blocked baseline produced a **proxy-local 502** carrying
  `Cannot connect`. The two are distinguishable, and the gate's oracle distinguishes them.

**Gate fidelity inspection (all 15 tests read against the approved contracts).** Triggers: real
subprocess, real HTTP, real `SIGINT`/`SIGTERM`, both flush parameters injected per `qa-project`
§Deterministic triggers, timer disabled wherever the volume gate owns the flush. Visible results,
forbidden results (witnesses A/B/C on TC-105/109; the no-documented-file clause on TC-102),
lifecycle (poll-based quiescence, no fixed sleeps), telemetry (none declared). **FIDELITY: faithful.**
The one post-035 change to the gate module is 048's, and it *tightened* the gate rather than
loosening it: the hard-coded `BASELINE_STATUS = 502` / `BASELINE_MARKER = b"Cannot connect"` oracle
and the "outcome-equivalence with that baseline" clause were deleted and replaced by
`assert_upstream_response` + a fail-closed preflight. I verified that discriminator adversarially,
in-process, without touching the tree:

```
FAIL-CLOSED  http://127.0.0.1:1              -> RuntimeError (Connection refused)
FAIL-CLOSED  http://no-such-host.invalid     -> RuntimeError (DNS)
ORACLE       proxy-local 502 "Cannot connect" -> REJECTED
ORACLE       HTTP 500                         -> REJECTED
ORACLE       400 carrying "Cannot connect"    -> REJECTED
ORACLE       live upstream's 404 model reject -> ACCEPTED   (the case 048 §Contract Draft and
                                                              harness.py's docstring anticipate)
```

Every one of the 15 tests constructs a `Proxy`, whose `__init__` calls `require_upstream()` unless
`require_live_upstream=False` — a keyword **no** MS-002 test passes (only `test_qap_ms004.py` uses
it, under the `qa-project` §Upstream amendment). So an upstream-down run errors out before any
assertion; it cannot produce a pass, and no stub is available to substitute. US-009/US-010 hold.

**Execution (authoritative, this invocation):**
`.venv/bin/python -m pytest tests/qa/test_qap_ms002.py -o addopts= -v -s` → **15 passed in 14.62 s**
at HEAD `bd16795`. Re-run for flakiness: **15 passed in 15.20 s**. (035's identical-scope run took
333.70 s because every proxied request waited ~5 s on a dead upstream; the duration drop is the
reachability change, not a scope change.)

| VP / scenario handle | Gate test | Verdict |
|---|---|---|
| VP-T001-session-isolation / TC-101 | `test_tc101_session_isolation_near_miss_prompts` | **pass** |
| VP-T001-session-isolation (R5 near-miss identity) | `test_identity_sid_format_near_miss_two_artifacts` | **pass** |
| VP-T002-collapsed-artifact / TC-102 | `test_tc102_103_104_…[change-focused]`, `[…self-contained]` | **pass / pass** |
| VP-T002-change-readable / TC-103 | same two cases | **pass / pass** |
| VP-T002-turn0-available / TC-104 | same two cases | **pass / pass** |
| VP-T002-write-count-reduced / TC-105 (paired, + documented half) | `test_tc105_tc109_write_and_file_counts_paired_runs` | **pass** |
| VP-T002-file-count-reduced / TC-109 | same test | **pass** |
| VP-T002-volume-flush / TC-106 | `test_tc106_volume_flush` | **pass** |
| VP-T003-shutdown-flush / TC-110 | `test_tc110_shutdown_flush[sigint]`, `[sigterm]` | **pass / pass** |
| VP-T003-interval-flush / TC-111 | `test_tc111_interval_flush` | **pass** |
| VP-T003-timer-resilience / TC-112 | `test_tc112_timer_resilience` | **pass** |
| VP-T002-collapsed-artifact (R1 parse boundary — 035's P1) | `test_hostile_non_utf8_and_empty_body_no_write` | **pass** |
| VP-T002-collapsed-artifact (dup keys, trailing newline) | `test_hostile_duplicate_json_keys_last_wins_single_record`, `test_hostile_trailing_newline_body_is_valid` | **pass / pass** |
| VP-T002-volume-flush (R2 dangling symlink, symlink escape) | `test_hostile_dangling_session_dir_symlink_no_write`, `test_hostile_artifact_symlink_escape_no_write` | **pass / pass** |

No scenario is `unknown`, none is `not_applicable`, none is orphaned, and no evidence bypasses a
declared trigger. The forbidden-ownership clause is respected: every gate asserts on the artifact,
the listing, the exit code and the log — never on model output.

**Durable G-008(b) comparison record — re-issued at this snapshot.** `evidence/ms002-comparison-record.txt`
was written by 035 (a rejected pass) and carried a live-upstream-unreachable disclosure plus a
"milestone gate REJECTED" footer. Its numbers reproduce **exactly** here (F=2/2/12 ops; files
1/1/7; bytes 482762/282647/442217), so the record has been updated in place: the environment
disclosure is now resolved, the R1 footer replaced by this verdict, and a re-execution block added.
This is the artifact the roadmap default-switch trigger consumes, and it now attests to a run with
the upstream reachable. `evidence/INDEX.md` gained the 069 row.

**Intermediate QA sweep (Check 6 step 8).** `git log f803b04..HEAD -- tests/qa/` → four commits:
`58ef966` (048's fidelity repair of this milestone's own gate), `98391ac`/`25c2b77` (MS-004's
artifacts), `c4e0200` (a docstring retraction in the TLS peer). No intermediate QA test exists for
T003 or for any of T003.001/T007/T008/T008.001 — the persistent gate module and the shared retainers
(`tests/qa/harness.py`, `tests/qa/counted_proxy_entry.py`) are the declared promotion targets, and
`counted_proxy_entry.py` is byte-unchanged since 035. Nothing to delete; nothing lingering. The only
untracked path under `tests/qa/` is `__pycache__` (ignored).

## Check 7: Integration Suite Execution — PASS

**Scope derived from the diff before running:** changed package = `buster_ripper` (the sole
production package: `dump.py`, `image.py`, `backend.py`, `config.py`, `main.py`, `server.py`,
`utils.py`, and all four `routes/*`), plus the shell entry `serve_buster_ripper.sh` and the README.
**Downstream dependents in the dependency chain: none** — nothing else in the repo imports the
package; the only external consumers are the `[project.scripts]` console entry
(`pyproject.toml:11` → `buster_ripper.main:main`) and the `tests/` trees.
**Excluded, and why:** `buster_ripper_legacy.py` — the tracked pre-refactor monolith, excluded from
scope by the approved plan (`plan-summary.md` §Task List "Not tasks" note); it is referenced by
nothing under `buster_ripper/`. `REFACTORING_PLAN.md` and `docs/` — prose. No package downstream of
a changed one was dropped, and nothing was excluded after a failure (there was none).

Runs at HEAD `bd16795`, clean tree:

1. **Full hermetic unit/integration suite** (the run that catches a regression in an unrelated
   module — per-task Yield runs only affected tests): `pytest` with the project's default
   `addopts` → **407 passed**, coverage **95.75%** against the `--cov-fail-under=80` floor, 9.61 s.
   `norecursedirs` (`pyproject.toml:23`) carries `"qa"`, and the run's output contains **zero**
   references to `tests/qa` — the network/signal gates are genuinely excluded from the default run.
   `buster_ripper/dump.py` 100%, `buster_ripper/image.py` 98%.
2. **Process-level E2E QA gates, this milestone** (`tests/qa/test_qap_ms002.py`): **15/15**, twice.
3. **Combined `tests/qa`** (MS-002 + MS-004's already-approved gate on the same snapshot):
   **21 passed in 59.92 s** — this milestone's gates neither interfere with nor depend on the TLS
   gate's loopback peer, which re-confirms 056's verdict at the current bytes.

**Shared-run rule applied:** one independently executed invocation covers Check 6 and Check 7 — the
same 15 tests, the same harness configuration, the same snapshot, with the hermetic suite and the
combined QA tree adding the coverage the milestone's own gate does not. Results are mapped to each
gate's tests and handles above; nothing was run twice merely to produce two reports.

## Findings

**None.** No P1, P2 or P3. The P1 that rejected MS-002 at 035 is closed at the source (typed 400 for
undecodable client bodies at every client-body parse site, T003.001 accepted-local) and its gate test
is green in two independent runs at this snapshot.

## Observations (recorded, not findings)

- **O-1 — carried dead state.** `messages.py:201` writes `state["input_tokens"]`, which nothing
  reads; verified equally unread at 035's HEAD (`capture["input_tokens"]`, `f803b04:166`). Not a
  regression from the `prepare()` split and not a severed data path (the value reaches its real
  consumer via `update_session_stats`). Routed to `quality-backlog.md` as a hygiene line so it is
  not lost; no action owed by this gate.
- **O-2 — `chat.py`'s post-commit second head** (`quality-backlog.md:51`, `roadmap.md` §Chat route)
  is a **live defect of the class T013/G-1 closed in `messages.py`**, already disposed out of scope
  by 067-01-yield. Verified non-interfering with this gate: no MS-002 scenario posts to
  `/v1/chat/completions`, and the chat route carries no dump call (dump is `/v1/messages`-only by
  plan). This approval does not certify it, and it must not be folded into this milestone.
- **O-3 — the 14.6 s vs 333.7 s duration gap** between this run and 035's is fully explained by
  upstream reachability (each proxied request formerly waited ~5 s for a connection failure).
  Recorded because a 23× speed-up on an unchanged test list is the kind of signal that should be
  explained rather than enjoyed.
- **O-4 — `qa-project` §Verified Against This Repository** still narrates the 2026-09-14 state
  ("`tests/qa/` does not exist yet", the three dump options "arrive with the buffered-dump work").
  It reads as a dated verification record, not a present-tense claim, and every row of
  §Harness Declarations is current and correct — so no gate is misdirected. Noted, not routed.

## Adjacent debt (not this gate's, listed so the orchestrator does not lose it)

- **MS-001's `QAP-MS001` has never been authored** (`qa-plan.md` §Authored Gate Tests). TC-100,
  TC-001, TC-107 and TC-108 have no committed gate, and `decisions.md:792` carries the session's
  only remaining PENDING verification row (`US-009` c2 → `yield-phase:MS-001 PENDING`). This pass
  observed the *default* documented path live (TC-105's documented half, no `--dump-format`,
  2N=12 writes, 7 files, no `.log`), which is corroboration for `OI:buster_ripper.dump/dump_turn/output_layout`'s
  MS-001-side clause — but the authoritative gate for TC-107/TC-108 is MS-001's own, still owed.
  Now that the upstream is reachable, nothing blocks authoring it.

## Assumption Routing (skill Routing section)

- `decisions.md` sweep for `Factual claims` entries carrying `Verification: yield-phase:MS-002 … PENDING`:
  **zero rows** (grep for `yield-phase:MS-002` → no match in `decisions.md`, `handoff-context.md` or
  `handoff-archive.md`).
- Whole-ledger sweep: the only PENDING verification row in `decisions.md` is
  `yield-phase:MS-001 PENDING` at `:792`, held by MS-001's arbiter; `:963` and `:972` read
  `yield-phase:MS-004 HOLDS (2026-10-05, HEAD 25c2b77)`. `:879` records a claim deliberately **not**
  promoted into a PENDING row.
- Legacy records: `handoff-context.md` §Canonical Verification Reconciliation — residue confirms the
  same, with a 2026-10-05 update stating the residue is now **1 row** (`US-009` c2, MS-001's). No
  legacy handoff or interview record names a `yield-phase:MS-002` arbiter → **no MIGRATION-REQUIRED**,
  and **no owned claim left `PENDING` or `UNCLEAR` by this pass**.

## Rejection Retry Cap

Root task MS-002 (constituent T003) stands at **rejection #1** in `handoff-context.md`
§Forward->Yield Retries — recorded by 035. This pass issues **no rejection**, so the counter is
**unchanged at 1** and was not incremented. Had this re-verdict rejected again, the cap would have
fired and required the pre-question gate over both causes; it is not needed.

## Deterministic Closure (pre-question gate)

No escalation, plan gap, permission request or user-facing remediation option arose in this pass, so
no closure packet is owed by it. The one thing this gate could have turned into a question — the
`_cancel_background_task` propagation trade-off at O/Check 3 — is already settled authority
(T010.001, 042-01-yield H2) and is gated at process level by TC-110, which passes. No uniquely
determined remediation was needed, and no material choice is being smuggled past the user.

## Verdict

**SEMANTIC-APPROVE — MS-002's integration gate closes at HEAD `bd16795`.**

- All seven checks PASS. Zero findings. Ten validation points and ten scenarios recorded `pass`
  against real trigger→outcome evidence, with the upstream reachable and the fail-closed preflight
  proven to fail when it is not.
- Not phase acceptance: per the identity contract this pass persists the gate result only. No
  Delivered chronicle entry, no parent-completion publication, and no phase-accepted mark was
  written. **Next: assumptions recheck and Yield Red**, which owns those actions after a CLEAN
  verdict. `HEAD` was not advanced, nothing was pushed, squashed or rewritten; the worktree is as
  found apart from this report, the re-issued comparison record, its index row, the backlog line
  (O-1), and the `handoff-context.md` status update.
- **`plan-summary.md` deliberately untouched**, following the 035 and 056 precedent for this agent.
  Its §Progress Overview line still reads "Still owed when `hyperipper.local:30001` is reachable:
  QAP-MS002 re-run → MS-002 re-verdict; QAP-MS001 never authored" — the first clause is now
  discharged by this report and the second still stands. That line is finalization's to rewrite, not
  this gate's; it is named here so the next reader knows it is stale rather than unaddressed, and so
  the orchestrator does not re-dispatch this milestone's gate on its account.
- A parallel Codex review (dispatched by the orchestrator) supersedes nothing here, but any source
  re-threading invalidates this verdict and requires a fresh applicable execution.

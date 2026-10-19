# 124 · Assumption Decide — PR #1 review round 1 (nine root findings)

## Orchestrator Prompt (verbatim)

> You are a subagent spawned by another session.
> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
> Worktree root: /Users/alexhubitski/Projects/buster-ripper
> Mode: external-review routing (PR #1, codex round 1)
> Assumptions directory: /Users/alexhubitski/Projects/buster-ripper/.sessions/codify/02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/
> Registry root: /Users/alexhubitski/Projects/buster-ripper/invariants/
>
> Nine root findings were posted on PR #1 at commit `0d465ec` (2026-10-19T01:52:19Z), one P1 and eight P2. Cross-reference each against `plan-summary.md`, the session roadmap (RM-001…RM-021) and the active invariants first, then classify and route. The user's instruction for this round: **fix every finding that is not in the 1% or 0.01% category** — rare-and-unreachable findings may stay recorded, everything else must be repaired.
>
> The findings, as posted:
>
> 1. **P1 chat.py:197 — post-commit chat-stream failure returns a fresh 502.** If the upstream drops a streaming chat response after `prepare()`, e.g. `iter_any()` raises `ClientPayloadError`, the exception escapes to `_make_chat_request`, whose `aiohttp.ClientError` handler returns a fresh 502 even though the head is committed — a second HTTP head inside the first response. This is the same class `T013` fixed for `/v1/messages` and is already recorded as `roadmap.md` §Chat route's post-commit second head (`RM-016`), with `QB:routes-chat/post-commit-second-head`, and `chat.py` is at 300/300 with `RM-001` as the prerequisite for in-place growth.
>
> 2. **P2 passthrough.py:52 — timeout class.** A passthrough request exceeding the 60 s total raises `asyncio.TimeoutError`, not `aiohttp.ClientError`, so the handler is skipped and aiohttp emits a generated error response instead of the route's JSON 502. The same mismatch exists in the timeout-bearing non-stream messages and chat relays (recorded as `RM-010`).
>
> 3. **P2 chat.py:221 — unfixed re-serialization.** Every valid JSON chat response where no transform sets `fixed` is re-serialized (`json.dumps`) instead of relaying the original bytes, changing whitespace/escapes/numeric form. Recorded as `QB:routes-chat/raw-bytes-passthrough-when-unfixed`, deferred by `DC-CR2-F-1-r1`.
>
> 4. **P2 utils.py:27 — `Content-Encoding` after decoded request body.** The server decompresses request bodies before handlers read them, but the filter retains the original `Content-Encoding`, so passthrough forwards plain bytes labeled compressed and JSON routes reserialize under a stale label. Recorded as `RM-008` (`latent`, base parity).
>
> 5. **P2 dump.py:155 — symlinked session directory.** A pre-existing `DUMP_DIR/<sid>` that is a symlink is accepted by `mkdir(exist_ok=True)` and later artifacts resolve outside `DUMP_DIR`. `QAP-MS001`'s R2 covered target-file symlinks but not an ancestor symlink.
>
> 6. **P2 dump.py:156 — collapsed numbering across restart.** A restart with the same dump dir, session id and collapsed format restarts at turn 0 with no previous rendering, so the next flush appends a duplicate `turn 000` and a broken cross-restart diff chain.
>
> 7. **P2 chat.py:190 — Retry-After dropped on streaming chat errors.** The streaming error branch forwards only `Content-Type` and discards every other end-to-end header, losing `Retry-After` and rate-limit headers.
>
> 8. **P2 chat.py:149 — streaming chat skips overflow detection.** For `stream: true`, the branch returns `_relay_chat_stream` directly, so an upstream 400 carrying the recognized context-overflow message never reaches `_chat_answer`/`_raise_chat_overflow`, and the retry loop performs zero retries for streaming chats.
>
> 9. **P2 passthrough.py:34 — redirects followed.** aiohttp follows redirects by default, so a 3xx passthrough fetches the redirect target and returns that target's response instead of relaying the original status/`Location`/body; base httpx did not follow by default. The messages and chat POST calls share the default.
>
> For each: apply the Reach Gate, say whether it is 1%/0.01% (and why), and disposition it. Then allocate the remediation as the fewest safe waves: `RM-016` (finding 1) is a recorded out-of-scope item whose owning route was "its own branch off main"; the user's instruction now overrides that deferral, so decide whether it rides this wave or needs its own isolation. `chat.py` is at 300/300 and `RM-001` is the extraction prerequisite for in-place growth — say plainly which findings can land without breaching the ceiling and which need `RM-001` first.
>
> Constraint: no user question unless materially different outcomes remain unranked after base parity and the recorded authorities are applied. The PR is open and every finding must end with a fix, a recorded accepted risk, or a recorded 1% item.

## Review round

`Review round: 1 (late: no — first hosted round; the round contains normal-use repairs, so the late-round 1% re-rating does not apply)`.

Enumeration through the frozen `pr-enumerate.py` script: `total=9 root_findings=9 replies=0 unreplied=9`, all created `2026-10-19T01:52:19Z` at `0d465ec873ab4b01daf936a37f99c15b51fdbf31`. The nine comment ids are carried per finding below; the parent owns the inline replies after the fixing commits exist.

## Recheck against the bytes (`HEAD 0d465ec`, clean tree)

| Finding | Claim | Measured on this tree | Verdict |
|---|---|---|---|
| 1 | `_relay_chat_stream` prepares, then relays; a `ClientPayloadError` escapes to `_make_chat_request`'s `except aiohttp.ClientError` | `chat.py:195` `await response.prepare(request)`; `:196-197` the unguarded relay; `:150` `except aiohttp.ClientError` returns a fresh `web.json_response(..., status=502)` | holds |
| 2 | a 60 s total raises `asyncio.TimeoutError`, not `ClientError` | `passthrough.py:27` `ClientTimeout(total=60)`; `:48` `except aiohttp.ClientError`; `asyncio.TimeoutError` is the built-in `TimeoutError` on Python ≥3.11, not an `aiohttp.ClientError` | holds |
| 3 | `fixed=False` re-serializes | `chat.py:221` `return _chat_response(resp, json.dumps(resp_data).encode())` after the `:216-217` fixed arm | holds |
| 4 | `forward_headers` keeps `Content-Encoding` over a body the server already inflated | `utils.py:21-27` keeps every name but `host`/`content-length`/`connection`/`transfer-encoding`/hop-by-hop; `passthrough.py:14` reads inflated bytes, `:31` forwards them under the peer's header; `chat.py:141`/`messages.py:139` send `json=` | holds — but this is the request-direction row `RM-011`, not `RM-008` (the response-direction sibling) |
| 5 | a live symlink at `DUMP_DIR/<sid>` is accepted | `dump.py:153-154` `session_dir.mkdir(parents=True, exist_ok=True)`; `Path.mkdir(exist_ok=True)` follows a symlink to a directory; `_require_writable`/`_writable_target` check only the leaf | holds — R2's dangling-symlink case is at `tests/qa/test_qap_ms001.py:948`; no live-ancestor case exists |
| 6 | a same-format restart appends a duplicate `turn 000` | `dump.py:147-158` always builds `SessionState(turn=0, prev_text="")`; `:200-206` appends; `:295-305` appends to the same artifact with no restart read | holds |
| 7 | the streaming error branch forwards only `Content-Type` | `chat.py:187-190` builds `headers={"Content-Type": ...}` only; `Retry-After` and `X-RateLimit-*` are dropped | holds — and `test_chat_stream_upstream_error_retains_representation` (`tests/test_routes_integration.py:181-192`) pins the narrow shape |
| 8 | `stream: true` bypasses `_chat_answer`/`_raise_chat_overflow` | `chat.py:147-148` returns `_relay_chat_stream` directly; `:187-190` returns the error without parsing it; `handle_chat`'s `:50-65` loop therefore never sees `ContextOverflowError` for a streaming chat | holds |
| 9 | aiohttp follows redirects; httpx did not | installed `aiohttp.ClientSession._request` has `allow_redirects: bool = True`; `passthrough.py:30-37`, `messages.py:155`/`:268`, `chat.py:141` pass no operand; base used `httpx.AsyncClient` with its default `follow_redirects=False` | holds |

## Invariant cross-reference

- Sources scanned: `invariants/index.md`, `invariants/buster_ripper/dump/invariants.md`, `invariants/buster_ripper/server/invariants.md`, `invariants/buster_ripper/assumption-claims.md`; no session `invariants.md` exists.
- Active/wip entries: **0**. Every registry entry is `Status: planned` (`OI:` provenance blocks have no `Status:` field and are not active entries). The skill therefore classifies every finding **Neutral**; the plan's protected invariants are used below as closure basis, never as an aligned-invariant claim.
- Findings checked: 9 — Neutral 9, Aligned 0, Contradictions 0, Ambiguous 0, Overrides matched 0.
- No permanent `Claim:`/`Composition:` record is cited as authority. The relevant descriptive handles are `CO:client-response-to-relay-consumer/payload-and-header-pairing` (response direction) and the planned `PI:T002/dump_format/*` family; none is treated as an invariant.

## Findings and dispositions

| # | Comment id | Location | Reach | Category | Disposition | Owning record |
|---|---|---|---|---|---|---|
| 1 | 4309720759 | `chat.py:197` | `latent` — a streaming chat whose upstream dies after the head; no code path prevents it, and the recorded roadmap already carries the byte-level repro | not 1%: P1, known R-E shape, client-visible protocol corruption, cheap arm; not security/data-loss | **`absorbed`** | lite brief `2026-10-18-pr1-chat-post-commit`; ledger `DC-PR1-R1-1-r1` |
| 2 | 4309720765 | `passthrough.py:52` | `reached` — every passthrough call arms the 60 s total; RM-010 records it reachable at all three arms | not 1%: class-list widening, one line per arm, existing 502 envelope; the whole-arm scope is recorded in RM-010 | **`absorbed`** | lite brief `2026-10-18-pr1-transport-parity`; ledger `DC-PR1-R1-2-r1` |
| 3 | 4309720772 | `chat.py:221` | `reached` — every valid JSON non-stream chat answer with no transform (normal use) | not 1%: normal use + cosmetic + cheap (net-negative with the in-place bool return); the deferral's own condition was capacity, which the same fix removes | **`absorbed`** | lite brief `2026-10-18-pr1-chat-relay-fidelity`; ledger `DC-PR1-R1-3-r1` |
| 4 | 4309720775 | `utils.py:27` | `latent` — needs a client that compresses its request body; no documented client does and the deployed launcher never sends one | **1%**: rare + base parity + no cheap fix where it shows (a request-direction content-coding posture, three admissible shapes, no settled authority ranks them) | **`accepted`** | existing `RM-011` + `handoff-context.md` §Elevated to user (`S16-decompress`); the reviewer's `RM-008` citation is the response-direction sibling |
| 5 | 4309720778 | `dump.py:155` | `latent` — needs a pre-existing live symlink at `DUMP_DIR/<sid>`; no production path creates one | not 1%: the fix is one guard at state creation, the containment contract is already the writer's own (`_require_writable`), and R2 already covers the sibling cases; high consequence (write outside the dump root) is stated first | **`absorbed`** | lite brief `2026-10-18-pr1-dump-containment`; ledger `DC-PR1-R1-5-r1` |
| 6 | 4309720779 | `dump.py:156` | `latent` — needs a restart with the same dump dir, session id and collapsed format | **1%**: unusual-but-plausible restart, wrong result that persists (duplicate `turn 000`, broken cross-restart chain), no cheap fix (the artifact is append-only and `(sid, format)`-keyed; continuing needs a reader/reconstruction path that does not exist, while truncating or per-run files violate settled identity/append-only) | **`accepted`** | new `RM-022` + `handoff-context.md` §Elevated to user |
| 7 | 4309720784 | `chat.py:190` | `reached` — any upstream 429/503 on a streaming chat request | not 1%: normal error path, cheap (net −1 line), base relayed the same headers | **`absorbed`** | lite brief `2026-10-18-pr1-chat-relay-fidelity`; ledger `DC-PR1-R1-7-r1` |
| 8 | 4309720787 | `chat.py:149` | `reached` — SGLang + streaming + overflow, the exact condition the route's own retry advertises | not 1%: cheap via reuse of `_chat_answer`/`_raise_chat_overflow`, and the branch's own docstring promises the retry | **`absorbed`** | lite brief `2026-10-18-pr1-chat-relay-fidelity`; ledger `DC-PR1-R1-8-r1` |
| 9 | 4309720790 | `passthrough.py:34` | `latent` — the deployed upstream answers no 3xx on the relayed paths; the branch's own default is what would follow one | not 1%: base-parity drift on all four relay calls, one kwarg each, cheap | **`absorbed`** | lite brief `2026-10-18-pr1-transport-parity`; ledger `DC-PR1-R1-9-r1` |

### Why 2 is not the 1% category

The fix changes the class list at three arms and keeps each arm's existing JSON 502. It is a few
lines, no new mechanism, and no import (`TimeoutError` is the built-in alias on the declared ≥3.11
range). The 1% row needs *not cheap*; this is cheap. The posture question RM-010 recorded (502 vs 504)
is answered by the repo's own authored envelope: every repo-authored client-facing failure is
`{"error": …}`, and 504 is a generated `text/plain` body that no route authors. The whole-arm scope is
RM-010's, and it is satisfied by the same one-line change at all three arms.

### Why 5 is not the 1% category

The defect is real and the consequence is a write outside `DUMP_DIR`, but the fix is one guard at state
creation and the writer already owns the containment contract (`_require_writable` refuses symlinked/non-regular
*leaves*; the session directory is the missing ancestor). The 1% row requires *not cheap*; this is cheap.
The reviewer's R2 citation is exact: `test_hostile_r2_documented_writer_does_not_escape_the_dump_root`
covers a **dangling** ancestor symlink (`:948`) and a symlinked **leaf** (`:951`), not a live ancestor.

### Why 6 is the 1% category

`VP-T002-collapsed-artifact` already acknowledges a restart with a **different** `--dump-format` may
legitimately leave one artifact per format; it says nothing about the same format. The artifact is
append-only and `(sid, format)`-keyed, so the candidate remedies are materially different products:
(a) continue numbering and seed `prev_text` from the existing artifact — needs a production reader and a
reconstruction path (change-focused stores diffs, not renderings), a real design decision;
(b) rotate per process run — breaks `PI:T002/dump_format/format_is_per_artifact`;
(c) truncate and restart — breaks append-only and loses the prior run;
(d) refuse the second run — breaks `G-002` (buffered dump reaches disk on shutdown).
No settled authority ranks them, and the fix is not cheap. The finding is real, its result persists, and the
trigger is an unusual-but-plausible restart rather than the common path. Recorded once; no repair here.

### Why 4 is the 1% category

The trigger needs a client that compresses its request body. Nothing in the package's documented surface
asks for it, the deployed launcher never sends one, and base could not produce the state at all (Starlette never
decompressed a request body; zero `gzip`/`decompress`/`Content-Encoding` references in
`buster_ripper_legacy.py`). The cause is a content-coding **posture**, not a defect elimination, and the three
admissible shapes (drop the label, reject encoded ingress, re-encode) are not ranked by any settled authority.
The row is `RM-011`; `RM-008` is the response-direction sibling. The reviewer's `RM-008` citation is corrected
here so a later reader does not file the request-direction defect under the response-direction row.

## Reach Gate — citations

| # | Reach | Citation |
|---|---|---|
| 1 | `latent` | the deployed upstream (`hyperipper.local:30000`, vLLM/uvicorn) does not drop a stream mid-body; the recorded RM-016 repro is a fault injection/worker death, not a normal answer |
| 2 | `reached` | `passthrough.py:27` arms `ClientTimeout(total=60)` on every passthrough call; RM-010 records the class at all three arms |
| 3 | `reached` | `chat.py:162-167` parses every valid JSON non-stream answer; `fixed=False` is the common no-transform path |
| 4 | `latent` | no documented client compresses a request body; `serve_buster_ripper.sh` sends none; base never decompressed one |
| 5 | `latent` | no production path creates a symlink at `DUMP_DIR/<sid>`; the R2 hostile tier creates one by hand |
| 6 | `latent` | needs a restart with the same `DUMP_DIR`, session id and collapsed format; the collapsed format is opt-in |
| 7 | `reached` | `chat.py:187` handles every upstream `status >= 400` on a streaming chat; a 429 is a normal upstream answer |
| 8 | `reached` | `chat.py:147` branches on `body.get("stream")`; `handle_chat:50-65` is the advertised SGLang retry |
| 9 | `latent` | the deployed upstream answers 2xx on the relayed paths; no production path produces a 3xx today (same guard shape as RM-008's `latent`) |

## Deterministic Closure Gate — packets

### 1 · Post-commit chat stream

```text
CLOSURE: absorbed
Finding: PR1-R1-1 (comment 4309720759, chat.py:197)
Required outcome: a failure after `prepare()` leaves one HTTP head and a terminated stream, never a fresh 502 inside the committed body; cancellation still propagates.
Authority: RM-016's "What the fix owes" (R-E's shape, log position-not-cause, return the committed object, CancelledError propagating); `messages.py:_relay_after_commit` accepted precedent (T013); G-003; the user's round-1 instruction overriding the branch deferral.
Admissible candidates: (a) catch `Exception` inside `_relay_chat_stream`, log, and return the committed `response`; (b) return a fresh 502 from `_make_chat_request`; (c) catch only `aiohttp.ClientError`.
Rejected candidates: (b) -> preserves the verified second head, the exact defect; (c) -> misses `AttributeError`, `TimeoutError` and downstream `OSError` classes the committed relay can raise.
Selected closure: (a), with the terminal left to `finish_response`'s `write_eof`.
Risk shape: unchanged — one private arm in an existing route function; the documented path, error branch, non-stream relay and request transforms are byte-unchanged.
Deferred additive work: a `seam-S11` reading for the chat relay (RM-016 obligation), owned by the seam-registry lane; trigger = the next `S11` composition dispatch on these bytes.
Owning route: lite brief `2026-10-18-pr1-chat-post-commit`.
Question required: no
```

### 2 · Timeout class

```text
CLOSURE: absorbed
Finding: PR1-R1-2 (comment 4309720765, passthrough.py:52)
Required outcome: a fired upstream deadline reaches the route's existing JSON 502 envelope, not aiohttp's generated response; the whole timeout-bearing arm family answers the same way.
Authority: every repo-authored client-facing failure uses `{"error": …}`; RM-010's whole-arm scope; G-003; the user's round-1 instruction overriding RM-010's posture deferral; base settled the class worse (Starlette 500), so no parity constraint forbids the 502.
Admissible candidates: (a) widen the class list to `(aiohttp.ClientError, TimeoutError)` at the three timeout-bearing arms; (b) leave the arms and pin aiohttp's generated 504; (c) widen only passthrough.
Rejected candidates: (b) -> preserves the generated `text/plain` body and traceback log the finding names; no settled authority adopts 504; (c) -> leaves the same mismatch at messages and chat, contrary to RM-010's whole-arm scope.
Selected closure: (a), with one fail-first witness per arm that names `asyncio.TimeoutError` as the escaped class.
Risk shape: unchanged — the only observable change is the timeout class reaching the site's existing authored envelope; no new mechanism and no new import.
Deferred additive work: `S17-N4`'s per-request ceiling remains with RM-010's owner; trigger = a new upstream-timeout config channel.
Owning route: lite brief `2026-10-18-pr1-transport-parity`.
Question required: no
```

### 3 · Unfixed re-serialization

```text
CLOSURE: absorbed
Finding: PR1-R1-3 (comment 4309720772, chat.py:221)
Required outcome: an unchanged JSON chat answer relays the upstream's bytes; only a transformed answer is re-serialized.
Authority: base `buster_ripper_legacy.py:882-892` re-serialized only the arm it rewrote; `DC-CR2-F-1-r1` deferred half; the finding's own byte-integrity evidence; G-003.
Admissible candidates: (a) make `_apply_response_transforms` mutate in place and return `bool`, with `_chat_answer` selecting `json.dumps` only when fixed; (b) add `content` as a fourth parameter; (c) keep the re-serialization and record it.
Rejected candidates: (c) -> normal-use byte drift, and the fix is net-negative; the deferral's capacity condition is removed by the same fix; (b) -> takes the helper from two gray metrics to three and triggers the preventive capacity gate; the in-place bool return removes metrics instead.
Selected closure: (a), net −2 raw lines.
Risk shape: unchanged — one private helper's return contract and one call site; the fixed path still re-serializes; the non-JSON fall-through and the overflow probe are byte-unchanged.
Deferred additive work: none.
Owning route: lite brief `2026-10-18-pr1-chat-relay-fidelity`.
Question required: no
```

### 5 · Symlinked session directory

```text
CLOSURE: absorbed
Finding: PR1-R1-5 (comment 4309720778, dump.py:155)
Required outcome: a pre-existing live symlink at `DUMP_DIR/<sid>` cannot make any artifact resolve outside `DUMP_DIR`.
Authority: `_require_writable`/`_writable_target`'s own "absent or a regular, non-symlink file" containment rule; QAP-MS001 R2's documented-writer containment contract; G-006/G-008; the caller's swallow-all contract at `messages.py:55-59`.
Admissible candidates: (a) refuse when the session directory itself is a symlink before `mkdir`; (b) check only the artifact leaf; (c) replace/delete the symlink; (d) resolve and verify containment.
Rejected candidates: (b) -> the verified defect is the ancestor; `_require_writable` already checks the leaf; (c) -> R2 asserts the symlink survives and rewriting it routes around the fault; (d) -> `Path.resolve()` follows the symlink and turns the escape into a legitimate-looking path; the writer must refuse, not normalize.
Selected closure: (a), raising `ValueError` so the caller's swallow-all arm keeps the request alive.
Risk shape: unchanged — one guard at state creation; both writers inherit it; no format, record or write-count change.
Deferred additive work: none.
Owning route: lite brief `2026-10-18-pr1-dump-containment`.
Question required: no
```

### 7 · Streaming chat error headers

```text
CLOSURE: absorbed
Finding: PR1-R1-7 (comment 4309720784, chat.py:190)
Required outcome: a streaming chat error relays the upstream's status, bytes and every end-to-end header, including `Retry-After` and rate-limit names.
Authority: base `buster_ripper_legacy.py` relayed `_response_headers(resp.headers)` on the same answer; `DC-CR2-F-2-r1`'s end-to-end-header rule; `T008.001`'s status/bytes/representation scope was a floor, not an exclusion; the user's round-1 instruction.
Admissible candidates: (a) use the shared `utils.response_headers(resp.headers)`; (b) hand-pick `Retry-After` and rate-limit names; (c) keep the `Content-Type`-only arm and record the loss.
Rejected candidates: (c) -> normal-use header loss and base drift, with a net-negative fix; (b) -> a hand-kept name list drifts from the upstream and duplicates the shared filter's job.
Selected closure: (a), net −1 raw line.
Risk shape: unchanged — status and bytes already travel; only which end-to-end names travel changes, and the shared hop-by-hop drop is untouched.
Deferred additive work: none.
Owning route: lite brief `2026-10-18-pr1-chat-relay-fidelity`.
Question required: no
```

### 8 · Streaming chat overflow

```text
CLOSURE: absorbed
Finding: PR1-R1-8 (comment 4309720787, chat.py:149)
Required outcome: a streaming SGLang chat whose upstream answers 400 with the recognized overflow message re-issues with a reduced `max_completion_tokens`, exactly as the non-stream chat arm and the messages streaming arm already do.
Authority: `handle_chat`'s own docstring ("Includes context overflow retry for SGLang backends"); `messages.py:handle_streaming`'s accepted shape (T008.001); `TestStreamingOverflowRetry` as the messages precedent; G-003.
Admissible candidates: (a) route the streaming `status >= 400` arm through the existing `_chat_answer(resp, await resp.read(), profile)`; (b) leave streaming chat without retry and document it; (c) parse the error with `resp.json()`.
Rejected candidates: (b) -> contradicts the route's own advertised backend-aware retry and the messages precedent; the fix reuses existing helpers and adds no mechanism; (c) -> a non-JSON error body would raise a new exception and turn a relayed error into a proxy 502; `_chat_answer`'s read-once JSON guard is the existing correct shape.
Selected closure: (a), which also carries finding 7's header relay and finding 3's byte relay.
Risk shape: unchanged — the error arm now uses the same read-once parse and overflow probe as the non-stream arm; the success stream path and the status/bytes relay are byte-unchanged.
Deferred additive work: none.
Owning route: lite brief `2026-10-18-pr1-chat-relay-fidelity`.
Question required: no
```

### 9 · Redirects

```text
CLOSURE: absorbed
Finding: PR1-R1-9 (comment 4309720790, passthrough.py:34)
Required outcome: a 3xx from the upstream reaches the client as the upstream sent it — status, `Location` and body — on every relay call, as base did.
Authority: base `httpx` default `follow_redirects=False` (base parity); G-003 holds `/v1/messages` behavior; the relay contract at every route; the user's round-1 instruction.
Admissible candidates: (a) pass `allow_redirects=False` at all four relay calls, and make the two streaming arms relay a 3xx instead of treating it as success; (b) fix only passthrough; (c) set the operand on the shared session; (d) follow and relay the final response.
Rejected candidates: (b) -> leaves the same drift on the messages and chat POST calls the finding names; (c) -> `ClientSession` has no such constructor parameter; the per-request operand is the native mechanism; (d) -> changes the status and drops `Location`, the exact defect.
Selected closure: (a) — four one-line operands; the chat streaming arm widens its existing `status >= 400` test to `>= 300` (net 0, `_chat_answer` relays status/headers/bytes for a 3xx), and the messages streaming arm gains a `300 <= status < 400` relay branch (+3).
Risk shape: unchanged — one per-request operand per call site and one status bound in each streaming arm; the shared session and connector are untouched.
Deferred additive work: none.
Owning route: lite brief `2026-10-18-pr1-transport-parity`.
Question required: no
```

## Remediation waves

The chat relay-fidelity brief is the **first** wave because it is the only net-negative
change in `chat.py`: findings 3 and 7+8 together remove **four** raw lines (300 → 296) and create
the headroom the P1 needs. The P1 cannot be the first `chat.py` edit at 300/300: alone it adds
two lines (300 → 302). With the relay-fidelity brief first, the P1 lands at 298 in its own wave;
without it, the P1 needs `RM-001`'s extraction. **No finding needs `RM-001` under the recorded order.**

| Wave | Brief | Findings | Why this wave / order | chat.py after |
|---|---|---|---|---|
| 1 | `2026-10-18-pr1-chat-relay-fidelity` | 3, 7, 8 | Net −4 raw lines; the only chat.py change that creates room for the P1; shares `_chat_answer`/`_relay_chat_stream` with the P1 but changes the error and body arms, not the post-commit success arm; no interference with any other brief | **296** |
| 1 | `2026-10-18-pr1-dump-containment` | 5 | Different component (`dump.py`), no shared assumption with the chat brief; rides Wave 1 as an independent lite run | 316 → ~318 (`dump.py`, yellow) |
| 2 | `2026-10-18-pr1-chat-post-commit` | 1 | P1, isolated: own brief, own commit and own lite-yield for independent attribution; +2 lines fits only after Wave 1 | **298** |
| 3 | `2026-10-18-pr1-transport-parity` | 2, 9 | Chat arm is net-zero (finding 2) and +1 (finding 9); messages and passthrough are under 500; needs Wave 1's chat headroom only | **299** |

Measured on a scratch copy of the current bytes, applying the six chat fixes one finding at a time:
finding 1 alone `300 → 302`; finding 2 alone `300 → 300`; finding 3 alone `300 → 298`;
findings 7+8 together `300 → 298`; finding 9 alone `300 → 301`; all six `300 → 299`.
So the first `chat.py` edit can only be finding 2 (net-zero) or the finding 3 / 7+8 pair (net-negative);
finding 1 cannot be first at 300/300, and finding 9 cannot be first. After Wave 1 the rest fit.
`RM-001` is **not** needed for any of them under this order.

`RM-016` **rides this round, not its own branch.** The user's round-1 instruction overrides the
`067-01-yield` deferral. It gets its own brief and its own lite-yield (independent attribution for a P1),
but it does not need a separate pull request: the same route and the same file are already under repair.
`RM-001` is **not** a prerequisite for this round under the order above. It becomes the prerequisite only
if the P1 must be the *first* `chat.py` edit; then `RM-001`'s extraction is required before the P1, and the
same is true for finding 9's +1 chat arm. Findings 3 and 7+8 can land first (net-negative), and finding 2's
chat arm is net-zero.

## Files that carry the durable record

- `handoff-context.md` §Deterministic Closure Ledger: `DC-PR1-R1-1-r1` … `DC-PR1-R1-9-r1` (seven absorbed rows; the two accepted findings get §Accepted risks / §Elevated to user instead).
- `handoff-context.md` §Planned Task Context: the three waves above with their brief paths, member findings, prerequisites and separation reasons.
- `handoff-context.md` §Accepted risks: finding 4 (`RM-011`) and finding 6 (`RM-022`).
- `handoff-context.md` §Elevated to user: finding 6 (`RM-022`), with `Ask: yes` (the user asked for a fix; the answer is a design decision about restart semantics).
- `roadmap.md` + `.sessions/codify/roadmap/2026-10-18-message-dump-bulk-flush.md`: new `RM-022`; notes on `RM-010`, `RM-016`, `RM-020` that PR #1 round 1 took them up.
- Lite briefs under `00bab65:.sessions/lite/2026-10-18-pr1-*/brief.md`.

## Notes

- Finding 4's reviewer handle (`RM-008`) is corrected to `RM-011` for the request direction. `RM-008` is the response-direction sibling; both are 1% records and neither is repaired here.
- Finding 6 is the only unrecorded finding; it gets `RM-022`.
- No user question was emitted. Every finding had either one admissible remediation or an already-recorded 1% posture. The only genuinely unranked design space is finding 6's restart contract, and it is recorded as a 1% item with `Ask: yes` rather than asked mid-routing, per the user's constraint.

```text
ASSUMPTIONS: 9 routed | 7 absorbed | 0 questions | 0 breaks | 0 malformed | 2 accepted | 0 capped
```

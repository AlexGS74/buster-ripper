# 066-01 Forward — T013 header-parity wave (057 F-4 → G-1 → G-2)

Session: `02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573`
Base at start: `41d3fc6` (Checkpoint 130) · Checkpoints created: **131** `c31d1ac` (F-4), **132** `d0b1317` (G-1), **133** `96fe429` (G-2), **134** `bdd04c5` and **137** `50413d7` (test-side: a further witness, then the full `codify-test-cost` pass — see §Test-cost audit), **135** `35666d3` + **136** `3352d73` (this report and the plan/handoff/backlog/evidence record). Production behaviour is final at 133; 134 and 137 move tests only.
Blueprint: `handoff-context.md` §Planned Task Context → `T013`, with the run-32 and run-35(second-pass) finding rows.

## Orchestrator Prompt (verbatim)

> You are a subagent spawned by another session.
> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
> Worktree root: /Users/alexhubitski/Projects/buster-ripper
> Task: T013 (header-parity wave: F-4, G-1, G-2)
>
> Read the `T013` block under §Planned Task Context in handoff-context.md together with BOTH routed-finding sections that amend it — "Assumption findings routed into `T013` (claims run 32)" and "(composition pass over `seam-S10`/`seam-S11` … second pass)" — and the DC ledger entries they cite. The second-pass section carries obligations the first did not: the parser-backend-independent empty-`Bearer` requirement (`DC-S10-C2-r1`, a fourth case on the same authoring instruction), the response-direction prohibition and direction-pairing witness (`DC-S10-C4-r1`, `DC-S10-LAT-r1`), the `content-length` framing witness, and G-1's corrected shape — the committed object is returned rather than a fresh one or `None`, with the third post-commit injection and the log-attribution requirement (`DC-S11-CE1/CE2/CE3-r1`, `DC-S11-D8-r1`). `062 R-A`…`R-E` are the single blueprint for the wave; the second-pass findings are narrower obligations inside it, not reopened bindings.

## Increment Brief

- **Task / goal:** `T013` — restore base hop-by-hop filtering at all five proxy boundaries from one shared owner (F-4), stop the messages relay from replacing a committed response (G-1), and make `REFACTORING_PLAN.md` describe the tree that exists (G-2).
- **Anchors touched:** none of the Anchor Map's rows. This wave edits the aiohttp-migration surface outside `T001–T006`; `utils.forward_headers`/`response_headers` are **extend** on existing referents, the five filter sites are **replace** in place, `REFACTORING_PLAN.md` is documentation.
- **Will NOT create:** no `@web.middleware`, no new module/class/registry, no sixth call site, no central filter (`062 R-A`); no authorization semantics in `utils.forward_headers` (`062 R-B`); no `buster_ripper/convert/` package (G-2).
- **Reuse evidence:** n/a — this increment creates no `Mode: NEW` referent, so the creation gate is not engaged. The two helpers it wires already exist at `utils.py:18-33` and had zero production callers.
- **Inventions:** none. Three module-private helpers were added inside the route pair purely to keep the case-table split inside LNCP bounds — `_relay_after_commit` and `_note_message_start` in `routes/messages.py`, and the chat-route empty-`Bearer` predicate, `_is_empty_bearer` at 066-01, re-shaped by retry 1's checkpoint 141 and now `_carries_token` in `routes/chat.py:124` — "Private helpers are not rows", `plan-summary.md` §Anchor Map closing note. (Corrected at drain from "Two … inside `routes/messages.py`"; the third function was disclosed in §What was implemented and §L→N→C→P, so nothing was hidden from review — `QB:report-accuracy/066-01-inventions-count`.)

## What was implemented

### F-4 — one shared filter at every boundary (Checkpoint 131)

| Site | Before | After |
|---|---|---|
| `chat.py:102` (`_prepare_headers`) | inline 4-name skip set | `forward_headers(request_headers)` |
| `messages.py:63` (`handle_messages`) | inline 4-name skip set | `forward_headers(request.headers)` |
| `passthrough.py:16` | inline 4-name skip set | `forward_headers(request.headers)` |
| `messages.py:258` (`_non_stream_single_request`) | inline 3-name list | `response_headers(resp.headers)` |
| `passthrough.py:41` | inline 3-name list | `response_headers(resp.headers)` |

- `utils.response_headers` extended to drop `content-length` alongside the eight names (`DC-F-4-r1`'s bound extension), with the reason in its docstring: `Response._start` derives a length **only when the header is absent**, so a relayed lying length is honored.
- `chat.py`'s empty-`Bearer` rule rewritten per `DC-S10-C2-r1`: it now reads **every** value under that name from the request's own mapping (`request_headers.getall("authorization", ())`), removes the name only when no non-empty bearer is among them, and re-adds the surviving credential when one is. It stays in `_prepare_headers`; `utils.forward_headers` gained no authorization semantics (`062 R-B`).
- `/v1/messages` and the catch-all still forward an empty `Bearer ` verbatim (base asymmetry, `062 R-D`'s third case).
- `DC-S10-C4-r1` honored as a prohibition: none of `chat.py:159-164`, `:191`, `:200` or `messages.py`'s SSE name fields is routed through `response_headers` or edited — proof in §Evidence.

### G-1 — the committed response is the only response (Checkpoint 132)

`_stream_single_request`'s single `try` is split at `await response.prepare(request)`:

- **pre-commit** keeps `except aiohttp.ClientError` → the 502 (unchanged half);
- **post-commit** (`_relay_after_commit`) gets its own `except Exception` that logs, does not re-raise, and lets the caller `return` **the already-committed `response` object** — not a fresh one (that is CE-1's second head inside the 200 body) and not `None` (which `web_protocol.finish_response` turns into an `HTTPInternalServerError` head). `prepare()`'s re-entry guard is per-object while the writer is per-request, so returning the committed instance short-circuits re-preparation and `finish_response()`'s `write_eof()` terminates the stream.
- One arm covers the whole class, per `DC-S11-CE3-r1`: `ClientPayloadError` and the `sock_*` timeout classes (`ClientError`-derived, and therefore *inside* the old arm — that is what produced the second head), the bare `asyncio.TimeoutError` that `ClientTimeout.total` raises (not a `ClientError`, so *outside* the old arm — that route today drops the turn with no status at all, per the corrected `DC-G-1-r2` cell), an upstream-controlled payload that raises `AttributeError`, and a client that leaves (`ClientConnectionResetError`, `ClientError`-derived). `asyncio.CancelledError` is a `BaseException` and still propagates.
- `DC-S11-D8-r1`: the arm's log is `"Post-commit relay error (%s): %s"` with the exception type and message — position plus exception, never upstream causation — and no disconnect-specific arm was added.
- `DC-S11-SGL-r1` is foreclosed rather than witnessed, as its row directs: the committed object is a `web.StreamResponse` with status 200, so `handle_streaming:95`'s `isinstance(result, web.Response) and result.status >= 400` is false on both clauses and the retry loop cannot re-dispatch onto a committed request.

### G-2 — `REFACTORING_PLAN.md` describes the tree that exists (Checkpoint 133)

The `Status:` heading no longer claims a complete Phase 1 that included a conversion layer; `buster_ripper/convert/` now appears only inside §Remaining Work under an explicit "nothing in this section is implemented" marker; the file-structure block matches `buster_ripper/` (`convert/` subtree removed, `backend.py` listed); the two §Source Project Reference rows that listed `convert/` files as shipped equivalents now say `— none yet` with the phase that owns them; Phase 6's image-test row reflects the `tests/test_image.py` that ships. One file touched, zero edits under `.sessions/` by the fix itself.

## Assumption findings handled (16 rows, all `implemented`, none self-marked `resolved`)

| Finding | Row | What it cost this increment |
|---|---|---|
| `claim-208` | wording condition | No name entered `_HOP_BY_HOP`; `trailers` not respelled; the set is pinned to base parity in a test, and no test/comment cites §7.6.1 |
| `claim-223` | scope | Exactly the five sites changed; no filter added to the metrics-poll path or any non-route outbound call |
| `claim-237` / `claim-238` | assertion form | Every absence is asserted case-insensitively on the captured mapping; no witness pins an output spelling; a mis-cased `tE` case proves the lowercased-name path |
| `claim-251` | scope | G-2 stays one file; the structural check is the single-file grep, not a repo-wide `convert` census |
| `S10-C1` | confirmation | The helpers are now live at five call sites — the seam's "single policy owner" premise became true instead of restated |
| `S10-C2` | **new requirement** | Clause (3)'s fourth case: duplicate `Authorization` spellings, both orders, **both parser backends** |
| `S10-C3` | witness | `Transfer-Encoding: chunked` + lying `Content-Length` → client length equals the body |
| `S10-C4` | correction + prohibition | Census reading corrected in the tests' own docstring; proxy-authored literals untouched, verified by diff |
| `S10-LAT` | witness | Each of the five witnesses pins which helper its site calls *and* which mapping it was handed |
| `S10-D6D8` | record | No obligation; nothing re-litigated — `forward_headers` gained no authorization semantics, no RFC name added |
| `S11-CE-1` | **requirement** | Return the committed object; termination (not "no 502") is asserted on raw bytes |
| `S11-CE-2` | correction | No message/comment/test calls the raise-out a second head; it is described as a dropped turn |
| `S11-CE-3` | witness + discharge | Bare `asyncio.TimeoutError` is the third injection; the timeout split is stated by class, never "caught by the pre-`prepare()` arm" |
| `S11-D8` | requirement | Position-not-cause log line, asserted per fault case; `CancelledError` still propagates |
| `S11-SGL` | record | Foreclosed by CE-1's return value; no witness spent, per the row |

## Failure-Path Evidence

| Operation / condition | Expected outcome and authority | Handling / observation evidence | Disposition |
|---|---|---|---|
| Egress: each of the 8 names, each of the 3 request sites | dropped before upstream — base parity `legacy.py:178` | `TestEgressSites` (parametrized 8×3 + spy pairing) | safe |
| Egress: end-to-end names (`Authorization`, `Anthropic-Version`, `X-*`) | survive | `test_end_to_end_names_survive` | safe |
| Egress: peer name spelling varies by backend (`tE`/`TE`/`te`) | drop keys on the lowercased name | `test_mis_cased_name_is_still_dropped`; filter lowercases before membership | safe |
| Egress: duplicate `Authorization` spellings, both orders, C + pure-Python parser | exactly one line upstream, the real token | `test_duplicate_spellings_never_lose_the_credential` over a raw socket; pre-fix build produced `[]` | safe |
| Egress: `Authorization: ""` (no bearer text at all) | untouched, forwarded — base forwarded it and the rule's subject is the value-less *bearer* form | `test_a_value_less_authorization_line_is_not_touched` | safe |
| Egress: no `Authorization` at all | dict returned unchanged | covered by every site witness (no auth header sent) | safe |
| Ingress: each of the 8 names on an upstream answer | absent at the client | `TestIngressSites` | safe |
| Ingress: `Transfer-Encoding: chunked` + lying `Content-Length` | client length equals the body sent | `test_lying_upstream_content_length_never_reaches_the_client`; mutation without the extension hangs the client → TimeoutError | safe (falsified) |
| Ingress: proxy-authored SSE/error header sets | not relayed, not filtered (`DC-S10-C4-r1`) | diff shows no changed line naming those fields | safe |
| Post-commit: `ClientPayloadError` / payload `AttributeError` / bare `asyncio.TimeoutError` / `ClientConnectionResetError` | one head, one terminal, no raise-out, no aiohttp error turn | `TestPostCommitFaults` over raw bytes; 10 of the 12 relay witnesses fail on the pre-G-1 bytes (M8) | safe |
| Post-commit: `asyncio.CancelledError` (client gone / shutdown) | propagates untouched | `test_cancellation_still_propagates`; arm is `except Exception` | safe |
| Post-commit: bytes already relayed before the fault | stay delivered | head/terminal shape: `test_one_head_one_terminal_and_no_error_turn` (200-only status lines, terminal `0\r\n\r\n`, connection not cut). The named witness was retired at checkpoint 137; verbatim relay is pinned fault-free by `test_chunks_are_relayed_verbatim_and_usage_recorded`, but no post-fault witness re-asserts the pre-fault chunk bytes — restoration belongs with the `write_faults=` knob of `QB:test-stream-relay/write-position-injection` (corrected at drain, `QB:report-accuracy/066-01-inventions-count`) | safe (shape); coverage narrowed at 137, recorded |
| Pre-commit: `ClientError` | still a 502 with the "Upstream error" log | `test_a_pre_commit_client_error_still_answers_502` | safe |
| sglang retry loop after a post-commit fault | cannot re-dispatch onto a committed request | foreclosed by construction (`DC-S11-SGL-r1`): committed `StreamResponse` status 200 fails both clauses of `handle_streaming:95` | not-applicable (unreachable by the closure; no witness spent, per the row) |
| A future build narrows the arm or adds a class-specific arm | inadmissible (`R-E`, `DC-S11-D8-r1`) | `test_the_relay_is_one_catch_all_not_a_class_split` + the third injection | safe |
| Cancellation-cleanup / partial completion of the upstream call | unchanged: the `async with` still closes the upstream response | success-path relay test + unchanged context-manager shape | safe |
| Hostile wire input at the parse boundary (R1) | aiohttp rejects malformed names/values itself (`InvalidHeader` for leading/trailing OWS in the name, `\n`/`\r`/NUL in the value); the proxy adds no parser of its own | excluded by family: this wave removes route-side parsing rather than adding any | not-applicable (evidence: `http_parser.py:164-167,232-235`) |
| Durable writes into pre-existing trees (R2), capture→act windows (R3), cross-language decoders (R4), join-key injectivity (R5), comparator parity (R7), generated artifacts (R8), text-matching over parsed structure (R9) | — | no writer, no capture window, no payload field, no identity key, no ordering, no regenerated artifact and no hand-rolled matcher is introduced; G-2 edits one prose file and its own check is a literal grep over that file | not-applicable |
| Documentation claim about the tree | no line asserts a path absent from `buster_ripper/` | `grep -n "convert/"` returns only not-implemented rows; every other claimed path verified to exist | safe |

## Repair delta — sensitive-data questions (touches `Authorization` handling)

| Question | Answer, with the file:line that settles it |
|---|---|
| Is a sensitive value used as a key, file name, or identifier anywhere? | No. `routes/chat.py:112-115` selects which `Authorization` value survives; the value is only ever written back into the outgoing header mapping. Nothing hashes, keys or logs it. |
| Is anything derived from it returned or logged, enabling guess-testing? | No. No log line in this wave includes a credential: `chat.py` never logs the header, and `messages.py`'s post-commit log carries the exception type/message only (`messages.py:200-202`). |
| Is a value trusted because of its shape rather than its origin? | The rule decides on shape (`value.strip() == "Bearer"`), which is the pre-existing lm-eval contract (`config.py:91`, `legacy.py:838` "Model-agnostic"). What changed is *which* shape wins: a non-empty value now outranks an empty one instead of last-wins-by-backend. Origin is not a thing the proxy asserts. |
| After a delete or mode switch, can the original still be read? | Yes, from the inbound `web.Request` object, unchanged and unowned by this wave; the delete is of the outgoing copy (`forward_headers` output), which is what base did too. |
| Was an identifier shortened until two values can collide? | No identifier introduced. |

## L→N→C→P compliance (final tree)

| Function | Before | After |
|---|---|---|
| `utils.forward_headers` / `response_headers` | clean | clean (`codify scan --git` → "No refactoring issues found") |
| `chat._prepare_headers` | L:8 N:2 C:4 (one gray: Medium Nesting) | not reported by the scanner — every metric at or below ideal (`chat.py --git` → "No refactoring issues found") |
| `chat._is_empty_bearer` (new) | — | clean |
| `messages._stream_single_request` | **L:50 N:5 C:7** P:4 | L:30 N:2 C:3 P:4 — three metrics improved, none worsened; two gray + P:4 pre-existing |
| `messages._relay_after_commit` (new) | — | L:9 N:1 C:3 P:4 → 1 gray |
| `messages._note_message_start` (new) | — | clean |
| `messages._non_stream_single_request` | L:36 N:4 C:8 | L:33 N:4 C:8 — improved, pre-existing violation untouched |
| `messages.handle_messages` | L:31 N:2 C:11 | L:30 N:2 C:11 — improved |
| `passthrough.handle_passthrough` | L:29 N:1 C:2 | L:25 N:1 C:2 → 1 gray |

File sizes at handoff: `utils.py` 133 · `chat.py` 295 · `messages.py` 293 · `passthrough.py` 53 · `conftest.py` 118 · `upstream_stub.py` 75 · `test_header_parity.py` 335 · `test_stream_relay_commit.py` 175 · `REFACTORING_PLAN.md` 164. Every production file stays under the 300-line Green ceiling (`messages.py` went 266 → 293 while its worst function went L:50 → L:30); `test_header_parity.py` entered the 301-500 review band, recorded as defer-tier debt like the suite's other test modules.

**Capacity gate:** not triggered as an extraction mandate — no additive production surface was created (the five sites are replacing edits, and G-1 is a split of existing code); the pre-edit baselines are in the table above and every touched function's metrics are equal or better. No file crossed 300 (Green) or 501 (must-extract), and none grew while already 501+, so no trigger condition applies; `messages.py` grew 266 → 293 raw lines because the case-table split needs two functions where one 50-line function was.

**Backlog pickup:** `QB:routes-messages/handler-lncp` and `QB:routes-chat/handler-lncp`/`request-lncp` are in this wave's touch-set and stay **unchecked** — `handle_messages`/`_handle_chat`/`_make_chat_request` each improved or stayed flat, and the block binds the mechanism to in-place substitution at five enumerated sites (`062 R-A`) with a prohibition on reaching the proxy-authored response literals (`DC-S10-C4-r1`); a handler restructure is the quality pass's design call, not this finding's. `QB:test-routes/file-size` — respected rather than skipped: the new witnesses went into two new modules instead of growing `tests/test_routes_integration.py` past 461 lines. New defer-tier items filed by this pass: `QB:test-header-parity/file-size` (the 335-line witness module) and `QB:test-routes/duplicate-upstream-stub` (the accepted route-integration module's local `_create_mock_response`, which the shared `tests/upstream_stub.py` now supersedes; migrating it is a next-touch job, not this wave's). `QB:deps/aiohttp-unpinned` — a note was added to the item: this wave's witnesses read `aiohttp.http_parser.HttpRequestParserC/Py` and `aiohttp.web_protocol.HttpRequestParser`, so they join `test_tls_posture.py` in being 3.13-internals-pinned (the C-parser case skips cleanly if the extension is absent).

## Tests and coverage

- `tests/test_header_parity.py` — **57** witnesses (F-4 clauses 1-3: per-site absence and survival, direction pairing by helper *and* by mapping handed, mis-cased names, the framing case, the four empty-`Bearer` cases, both parser backends, base-parity scope of the name set, the census).
- `tests/test_stream_relay_commit.py` — **12** witnesses (G-1's case table over raw downstream bytes, position-not-cause log, cancellation, one-catch-all shape, unchanged success relay, unchanged pre-commit 502).
- `tests/conftest.py` — `proxy` (real app behind an aiohttp client) and `raw_proxy` (real app on a loopback socket), shared by both modules. The socket is needed twice: the aiohttp **client** collapses two `Authorization` lines before they reach the wire, and G-1's terminal is only observable as bytes.
- `tests/upstream_stub.py` — one `StubUpstream`/`StubStream`, imported the way the QA gates import `tests.qa.harness`.
- Affected scope + suite: `uv run pytest -q` → **370 passed** in 10.2 s, coverage **94.20%** (was 301 tests @ 93.88% before this wave; the 69 new witnesses add 0.13 s) (`--cov-fail-under=80` reached; `utils.py` 98%, `buster_ripper/routes/*` omitted per `pyproject.toml:30-34`, which `DC-F-4-r2` forbids re-opening).
- `QAP-MS004` re-run (F-4 done-when (4)): `uv run pytest tests/qa/test_qap_ms004.py` → **6 passed**, run once after F-4 and once after G-1.
- No test asserts LNCP metrics; the source-shape test in `test_stream_relay_commit.py` witnesses a plan prohibition (one catch-all arm), not a metric.

## Test-cost audit

The `codify-test-cost` rules arrived mid-wave (after checkpoints 131-133). Applied to
this wave's own tests and landed at **137**, with the full record in
`evidence/t013-header-parity-wave.txt`:

- **Rule 6** — the wave had grown three stand-ins for the same upstream answer and a
  proxy fixture that reimplemented `tests/test_routes_integration.py`'s accepted
  `client`. Consolidated into `tests/conftest.py` (one `proxy`, one `raw_proxy`) and
  `tests/upstream_stub.py` (one stub, imported rather than fixed-injected, so no test
  pays a parameter for it). The accepted module was left alone; migrating it is filed
  as `QB:test-routes/duplicate-upstream-stub` rather than dragged into this diff.
- **Rules 4/5** — `RawProxy.send` ended every read on a 1.0-2.0 s idle window, so eleven
  tests each paid a second for a response that had already completed. It now ends at the
  chunked terminator, at EOF, or on an explicit `Connection: close`; the window only
  bounds a build that never terminates, which is itself what G-1 witnesses.
- **Rule 3** — three witnesses removed: a third header-spelling variant, a duplicate of
  the bare-timeout case, and an already-relayed-bytes case whose falsifier was the same
  wrong build M4 already catches.
- **Rule 1** — worth stating plainly: the *first* consolidation pass gave the stub an
  `iter_any()` on the response object instead of on its `.content`, so the relay died with
  `AttributeError` inside the arm and four terminal witnesses passed **for the wrong
  reason** — one head, a terminator, no error turn, none of them caused by the fix. The
  pre-fix baseline run (M8) is what caught it; that run is now part of the evidence record
  rather than authoring scaffolding.
- **Arity** — no test or helper in this wave takes more than five parameters. The accepted
  suite reports zero "Too Many Parameters"; so do these four files.

Cost: the two witness modules **9.19 s → 0.13 s**, the hermetic suite **19.5 s → 10.2 s**,
72 → 69 witnesses, with M1-M8 all still turning the set red.

## Evidence

`evidence/t013-header-parity-wave.txt` is the full record: the rename-proof census, the
3/2 direction-pairing call sites, the `DC-S10-C4-r1` byte-unchanged proof (plus
`chat.py` not importing `response_headers` at all), the name set shown equal to
`buster_ripper_legacy.py:178` with zero RFC citations anywhere in the diff or the seven
commit messages, the change-set scope check for `DC-claim-223-r1`, the `convert/` grep
with every claimed path verified against the tree, the two-parser wire probe, the
mutation matrix below, the test-cost audit, and the measured LNCP table.

Fail-first matrix — re-run against the final consolidated fixtures, each mutation applied,
the suite run, then reverted:

| # | Wrong build | Fired | What it proves |
|---|---|---|---|
| M1 | `response_headers` without the `content-length` extension | 3 / 57 | both ingress framing witnesses hang out the client's 5 s timeout on the relayed lying 999 (`DC-S10-C3-r1`) |
| M2 | messages' egress site wired to `response_headers` | 2 / 57 | the direction pairing, by spy and by `Host`/`Content-Length` behaviour (`DC-S10-LAT-r1`) |
| M3 | `chat.py` at `41d3fc6` (rule decides from the collapsed dict) | 9 / 57 | the fourth `Bearer` case on both backends and both orders (`DC-S10-C2-r1`) |
| M4 | post-commit arm returns a fresh 502 — today's defect | 8 / 12 | second status line or missing terminator at every terminal (`DC-S11-CE1-r1`) |
| M5 | arm narrowed to `except aiohttp.ClientError` | 5 / 12 | the non-`ClientError` half escapes again, and the one-catch-all shape witness fires (`DC-S11-CE3-r1`) |
| M6 | post-commit log back to `"Upstream error: %s"` | 8 / 12 | position-not-cause attribution (`DC-S11-D8-r1`) |
| M7 | abandonment path returns `None` | 4 / 12 | aiohttp turns the missing return into a 500 head (`DC-S11-CE1-r1`) |
| M8 | `messages.py` at Checkpoint 131 (pre-G-1) vs the relay module | 10 / 12 | the regression baseline: the two survivors are the success relay and the pre-commit 502, the halves the split must not change |

## Concerns for Yield

1. **`_prepare_headers` now emits the canonical `Authorization` spelling** when it re-adds a surviving credential. On the C backend that is what the peer's line already became, so nothing observable moves; on `AIOHTTP_NO_EXTENSIONS=1` a client that sent `authorization:` now goes upstream as `Authorization:`. Field names are case-insensitive, `DC-claim-238-r1` forbids pinning an output spelling, and base itself was spelling-blind here (it never rewrote the key at all, but the C parser had already canonicalized it). Flagging because it is a *deliberate* consequence of deciding from the request mapping rather than the collapsed copy — not an oversight to be "fixed" back.
2. **`test_header_parity.py` is a 335-line new module, in the 301-500 FL-LNCP review band.** Splitting it by boundary direction was the alternative; it was left whole because the five-site census and the four-case `Bearer` table are one acceptance set and reading them apart costs more than it buys. Filed as defer-tier debt, not resolved by rearranging witnesses. `messages.py` itself stayed Green at 293 raw lines while every function inside it improved.
3. **`_stream_single_request` still carries two gray metrics (L:30, N:2) plus pre-existing P:4**, all three improved from L:50/N:5/C:7. The residual length is the `web.StreamResponse` literal that `DC-S10-C4-r1` puts out of bounds — extracting it is the natural cure and is exactly what that prohibition forbids touching. `handle_messages`' hard C:11 and the other route-handler debts stay with the quality pass, listed under §Backlog pickup.
4. **One G-1 fact is not independently falsifiable:** deleting the inner `await response.write_eof()` changes no byte, because `finish_response()` performs the terminating `write_eof()` for the returned committed object — which *is* the mechanism `DC-S11-CE1-r1` relies on. The termination witness does discriminate every wrong answer the ledger names (fresh object, `None`, raise-out). Recorded rather than papered over.
5. **The C parser does not strip trailing OWS from a header value; the pure-Python one does** (`"Bearer "` vs `"Bearer"`). That is why the contract's two spellings both matter and why the witnesses accept either; it is a new library fact this pass observed, not one the block recorded. It does not change any Required cell.
6. **G-3-adjacent staleness left alone on purpose:** `REFACTORING_PLAN.md:112` still describes `session.py` as "Session stats + SQLite persistence" (true: `sqlite3` is imported there). Only `convert/` rows and the `Status:` line were in this member's evidence, and `QB:cli-surface/…` owns the remaining doc debt.

## Invariant Contradictions Raised

None. No invariant was created, edited, superseded or deleted (triad rule respected; the registry stays 0/0/0/0 for this wave). `CL:buster_ripper.routes.messages#handle_streaming/upstream-body-relay` and `CL:buster_ripper.routes.chat#_relay_chat_stream/sse-response-preparation` were read as descriptive evidence only.

## Assumption / seam obligations this wave owes on return

- `seam-S9` rider (from the block): the G-1 edit landed inside its §Sources range `routes/messages.py:121-176`, and member row 67 pins the old `messages.py` digest — S9 must be re-verified on the new bytes and recomposed before it is citable as current. Its `State` was already `recheck`.
- `seam-S10` and `seam-S11` both read `COUNTEREXAMPLE` on the pre-wave bytes; all 11 findings landed inside F-4/G-1, so recomposition is owed at this wave's accepted-local recheck. `S10-D6D8` and `S11-SGL` retire with that `SAFE` read.
- `QB:assumption-registry/run-31-verdict-fingerprints`: rows 246/247 (`seam-S10`'s test-surface negatives) are named for re-issue at *this* wave's accepted-local recheck — `tests/` bytes did move (two new modules + `conftest.py`), so that recheck is now the right moment.

## Next

Yield on the wave (`c31d1ac`, `d0b1317`, `96fe429`, plus the test-only `bdd04c5`): one local Yield, no milestone gate. `T014` (image-slice) is independent and untouched by this branch's bytes.

## Session artifacts this pass wrote

- `plan-summary.md`: the `T013` row's status and Notes cell (paragraph → pointer + one outcome sentence), the
  §Progress Overview Forward/Yield row, and the §Current Phase remediation-wave paragraph — all three prior cells
  preserved verbatim in §Superseded plan wording below. File went 465 → 455 lines.
- `handoff-context.md`: §Current Status **replaced** (241 lines → 34) per the Forward contract; the routing detail it
  summarized still lives in that file's §Deterministic Closure Ledger, §Composition Routing and §Planned Task Context,
  which this pass did not touch. `Last Updated:` stamp rewritten (old text preserved below). All 16 `T013`
  assumption rows moved `pending` → `implemented`; `T014`'s rows untouched.
- `quality-backlog.md`: `QB:deps/aiohttp-unpinned` widened with the new parser-internals dependents (still open);
  the three route-handler LNCP items annotated with their re-measured metrics and why they stay open; new
  defer-tier item `QB:test-header-parity/file-size`. Nothing was marked resolved.
- `evidence/t013-header-parity-wave.txt` + one newest-first row in `evidence/INDEX.md` (Disposition left empty —
  the user's ruling).
- Two test-only additions landed **after** the three production checkpoints and were committed separately
  (`test(proxy): …`, checkpoints 134 and 137), never by amending 131/132/133: the value-less-`Authorization`
  witness, and then the full `codify-test-cost` pass described in §Test-cost audit above. Both re-ran M1-M8
  afterwards to prove the reshaping did not blunt a single witness.

## Superseded plan wording (preserved verbatim per the rewrite rule)

Three  cells were rewritten to current truth by this increment. Their prior contents:

### §Task List row  (Notes cell)

```text
| T013 | Restore base hop-by-hop header filtering at every proxy boundary, close the post-`prepare()` error return, and correct `REFACTORING_PLAN.md`'s phantom `convert/` rows | pending | Assumption: required:new-surface (no seam named by the packet; `buster_ripper/utils.py` and the header-filter sites sit outside every recorded seam) — **plus the recorded S9 rider**: the G-1 edit lands inside `assumptions/compositions/seam-S9.md`'s §Sources range `routes/messages.py:121-176`, so the wave owes the ordinary post-implementation member re-verification and recomposition before `seam-S9` is citable as current. Closure packets: 057 F-4 (`DC-F-4-r1` → `DC-F-4-r2`), G-1 (`DC-G-1-r1` → `DC-G-1-r2`), G-2 ride-along (`DC-G-2-r1`, no repair row). **Durable block repaired 2026-10-05 by Authority Registrar per [062-decide](iterations/062-decide.md) `R-A`…`R-E` (0 user forks); it is the single blueprint — this row is a pointer. Awaits a `forward-red` re-run returning `0 AMBIGUOUS`; Forward NOT authorized yet.** Owning route: Forward; wave anchor for the **header-parity** wave. Gates no milestone — outside `02.00`'s approved plan (§Task List T001–T006), so no accepted verdict moves. **Assumption gate status (claims run 32, 2026-10-05 assumption-decide): the wave's four `DOES-NOT-HOLD` premises are routed `absorbed` into this block (`DC-claim-208-r1`, `DC-claim-223-r1`, `DC-claim-237-r1`, `DC-claim-238-r1`, `DC-claim-251-r1` — premise corrections and one assertion-form requirement, no `062` binding reopened) and row 216 is `malformed` → `forward-collect`, which blocks `seam-S10`'s composition but no implementation criterion; Forward still NOT authorized**. **Composition pass (2026-10-05, `assumptions/seams.md` §Run-35): the chain is closed for this wave — `seam-S10` and `seam-S11` are composed, both `COUNTEREXAMPLE`, and all eleven of their findings route `absorbed` into `T013` steps 1/2 (`DC-S10-C1-r1`…`DC-S10-LAT-r1`, `DC-S10-D6D8-r1`, `DC-S11-CE1-r1`…`DC-S11-SGL-r1`), 0 questions / 0 breaks / 0 malformed. One genuinely new behavior requirement (comp `S10-C2`: the parser-backend collapse lets `chat.py`'s empty-`Bearer` rule delete a *real* credential — the rule must read every value under that name, case-insensitively, and stays route-local per `062 R-B`), plus four witness/record obligations inside F-4 and G-1 — direction pairing on the five sites, the `chunked`+lying-`Content-Length` framing case, the response-direction census correction (the five proxy-authored literals stay out of scope and must not be routed through `response_headers`), and G-1's abandonment path returning the *committed* response rather than a fresh object or `None`. One landed ledger row's evidence cell corrected: `DC-G-1-r2`'s Today column claimed the post-commit non-`ClientError` raise-out writes a second head; it does not — `handle_error` refuses once `output_size > 0`, so that route drops the turn (`DC-S11-CE2-r1`). No `062` binding reopened, no new task or file. `T013`'s block is unchanged as the single blueprint; Forward's authorization remains the orchestrator's call** |```

### §Current Phase — remediation-wave paragraph

```text
⚠ **Two remediation waves are open and neither is dispatchable yet** (registered 2026-10-05 by Authority
Registrar). `T013` · restore base hop-by-hop header filtering at every proxy boundary (header-parity: F-4 → G-1 →
G-2) and `T014` · make `--strip-images` cover the whole request (image-slice: F-1 → F-2 → F-3) carry the durable
blocks in [handoff-context.md](handoff-context.md) §Planned Task Context — **those blocks are the single blueprints**
(both `plan-summary.md` restatements were reduced to pointers by the same batch). Their `red-findings` repairs have
all landed: `T013` per [062-decide](iterations/062-decide.md) `R-A`…`R-E`, `T014` per
[063-decide](iterations/063-decide.md) `R-A`…`R-E`, ledger `DC-F-4-r2`, `DC-G-1-r2`, `DC-F-1-r2`, `DC-F-2-r2`,
`DC-F-3-r2` appended (the `r1` dispositions refined, not reversed). **Zero user forks came from either pass**, so
`decisions.md` gained no FK and no user question is owed by this batch. **Next: `forward-red` re-dispatched per wave,
each returning `0 AMBIGUOUS` before its Forward may run** (the 029 chain). Both waves gate **no** milestone:
`MS-004`'s APPROVED verdict and every accepted task's scope stand untouched, and `FK-011` remains open and
Interview's to present.```

### §Progress Overview — Forward / Yield row

```text
| Forward / Yield | **MS-004 boundary base-sync PASSED — `unchanged`; next milestone work is the MS-004/QAP-MS004 author pass** | [054-01-forward](iterations/054-01-forward.md): target `main @ 72e3a1c…` already integrated (ancestor true, 154 ahead / 0 behind, clean HEAD `ea0b40b…`) — no merge, no re-threading, no prework packet; suite evidence inherited from [053-01-yield](iterations/053-01-yield.md) at the bit-identical production tree (T012 ACCEPT, zero reject-tier findings, 3 P3 deferred, 301 @ 93.88%). Post-acceptance recheck stands: 25-member handle-bearing `seam-S6` `SAFE` on `c58256fb…`, published in place to `CO:serve-script-env-to-argv/env-var-flag-crossing` (`assumptions/seams.md` run-29). Still owed when `hyperipper.local:30001` is reachable: QAP-MS002 re-run → MS-002 re-verdict. |```

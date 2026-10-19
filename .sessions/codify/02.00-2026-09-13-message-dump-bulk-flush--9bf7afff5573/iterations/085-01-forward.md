# 085-01 Forward — `T019.001` (run-45 remediation, one-member wave)

Session: `02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573` · Worktree: `/Users/alexhubitski/Projects/buster-ripper`
Branch: `refactor/package-structure` · Wave base: `c96ca3e2` (Checkpoint 194) — no capacity extraction was owed (see §Capacity gate)
Task: `T019.001` (adhoc remediation nested under `T019`; **not** a `T019` retry — `T019`'s counter stays 0, its landed range `d5df8b06…9c90675` untouched)
Blueprint: `handoff-context.md` §Planned Task Context → `T019.001`
Ledgers: `DC-claim-418-r1` (primary), `DC-claim-422-r1`, `DC-claim-423-r1`
Origin: `assumptions/compositions/S18-decode-failure-surface.md` §Notes N6, adjudicated by the run-45 atomic routing pass [084-assumption-decide](084-assumption-decide.md)
Evidence: [t019-001-fail-first-run.txt](../evidence/t019-001-fail-first-run.txt) · reach/base: [n6-reach-and-base-comparison.txt](../evidence/n6-reach-and-base-comparison.txt)

## Orchestrator Prompt (verbatim)

> You are a subagent spawned by another session.
> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
> Task: T019.001

Result: **DONE** — one production expression replaced (`error_text = result.text`), no constructor touched; 5 route-level witnesses, **2 red at pre-fix bytes** and green at final bytes; hermetic **476 passed @ 95.91%** (baseline 471 @ 95.91%); `handle_streaming` re-measured **L:25 N:3 C:8 P:4 — unchanged from baseline**; the five must-not-move surfaces proven identical by AST, not by grep. **Pointer discipline:** this pass's first draft cited the blueprint's pre-shift line numbers *inside code it was adding*, which those very edits invalidated — corrected in place, now by role rather than by number (§Pointer discipline).

## Increment Brief

- **Task and goal**: `T019.001` — make `handle_streaming`'s context-overflow probe read the upstream's error **text**, so the SGLang streaming retry can fire; today it is handed a payload object's `repr` and never matches.
- **Anchors touched** (plan §Anchor Map): dump caller → `routes/messages.py:handle_messages` → **reuse** (untouched); the probe itself is `handle_streaming`, an existing bound referent named by the blueprint at `:106`, mode **extend** — the plan carries no row for it because §Anchor Map binds this session's *dump/TLS* cluster, and `T019.001` is a remediation routed from `claims.md`, whose route text names the site.
- **Will NOT create**: no second error-text surface beside `_error_body`; no second overflow matcher beside `utils.parse_context_overflow_error`; no duplicate of `tests/upstream_stub.py`'s `StubUpstream`; no new module for either the fix or its witnesses.
- **Creation gate**: not exercised — zero `Mode: NEW` anchors in this increment, so no `RB-*` replay was owed. `Inventions: none` beyond module-private test scaffolding (`_sglang_stream` and its inner `record`).
- **Protected invariants**: none contradicted. The blueprint classifies the cross-reference **Neutral** — `invariants/` has no `buster_ripper/routes/*` file, every entry reaching this session is `Status: planned` (`PI:T002/dump_format/*`, `PI:T006/upstream_tls/*`), and there is no session `invariants.md`. Zero `invariants/` bytes moved — the triad rule held.
- **QA contract**: `qa-plan.md` present; `T019.001` carries **no `VP-*`/`TC-*` handle** — its gate is this increment's local Yield. No `QAP-*` gate test was inspected or authored here.
- **Assumption finding**: `claim-418` / `claim-422` / `claim-423`, all `DOES-NOT-HOLD`, disposition `absorbed` (`Assumption impact: required:new-surface`). Implemented for Yield to rule on; not closed by this pass.

## Failure-Path Evidence

| Operation / condition | Expected outcome and authority | Handling / observation evidence | Disposition |
|---|---|---|---|
| SGLang streaming upstream `400` carrying the overflow text | reduce `max_completion_tokens` and re-issue, as the non-stream sibling already does (`DC-claim-418-r1`) | `test_an_overflow_reduces_the_window_and_reissues` — red pre-fix (`assert 1 == 3`), green final with `sent[1]`/`sent[2]` both `256` | safe (repaired) |
| The same overflow, retries exhausted | the loop's own `400 Context overflow after 2 retries`, not the upstream's body relayed as if it were the answer (`:119-123`) | `test_the_loop_answers_for_itself_when_the_retries_run_out` — red pre-fix on the body diff (upstream text where the loop's belongs), green final | safe (repaired; the arm was dead on **both** paths pre-fix) |
| A `>= 400` that is **not** an overflow | relay **once**, byte-identical — status, body, `Content-Type: application/json` with no charset (done-when 3) | `[…][plain-4xx]` — `post.call_count == 1` and header/body equality; red on any build that retries every 4xx | safe (pin, green in both states by design) |
| Upstream `>= 400` whose bytes no charset can decode, **inside** the retry loop | still the upstream's answer; the text the probe now reads is strict-decodable by construction because `_error_body` replaced what it could not decode | `[…][undecodable-payload]` — the `\xff\xfe\x00garbage` 503 relays once with `\ufffd\ufffd\x00garbage` on the wire | safe (pin; R1 hostile input at the parse boundary) |
| The transport arm's static `502` crossing the probe | nothing raises out of `handle_streaming` from that read — both objects the loop can be handed (`:151-155` StringPayload, `:171-173` json_response bytes) round-trip through `.text` | `test_the_transport_502_crosses_the_probe_without_raising` → `502`, not aiohttp's 500; plus the in-source note at `:106` naming both shapes (done-when 5) | safe (pin) |
| A build that repairs the arm by constructing with `text=` instead of reading `.text` | rejected candidate (`DC-claim-418-r1`) — it adds `; charset=utf-8` to a relayed error, a client-visible change on the arm the accepted relay-fidelity work governs | pinned by the `Content-Type` equality in the non-overflow witness, which is red for exactly that build | safe (foreclosed by a witness, not by prose) |
| `.text` returning `None` out of the probe | impossible on the shipped line: `_stream_single_request` returns a `str`-bodied `Response`, a bytes-bodied `json_response`, or a prepared `StreamResponse` the `isinstance` guard at `:105` excludes | read of both constructors (`:151-155`, `:173`) + `web_response.py:693-730`; `_error_body` returns a `str` on every arm | not-applicable (no branch, per R10) |
| A non-SGLang backend receiving the same overflow | no retry at all, unchanged | existing gate at `:94-95`; `TestBackendAwareRetry` + every `proxy`-fixture test runs `vllm`, all green unedited | safe (untouched) |
| Client names a codec Python lacks / body over a cap / stream outliving a deadline | those arms are `T018`/`T019`'s accepted work | files byte-identical (`git status --porcelain buster_ripper/ tests/` → 2 paths, both mine) | safe (out of this increment) |

## Changes

### `T019.001` — `DC-claim-418-r1`, `DC-claim-422-r1`, `DC-claim-423-r1`

**`buster_ripper/routes/messages.py`** (+12/−1 — 11 comment lines, one statement for one statement — in one hunk, inside `handle_streaming` only):

```python
-            error_text = result.body.decode() if isinstance(result.body, bytes) else str(result.body)
+            # The response's own text surface. A str handed to web.Response(body=) is
+            # … (10 further lines: the totality note done-when (5) prescribes) …
+            error_text = result.text
```

- **What was wrong, in the dependency's own words**: `Response.body`'s getter is `return self._body` (`web_response.py:693-695`); its setter stores a `str` as `payload.PAYLOAD_REGISTRY.get(body)` → `StringPayload` (`:705`, `payload.py:1111`). So `isinstance(result.body, bytes)` was **false** for the only producer that reaches the arm, and `str(StringPayload)` fell through to `object.__repr__` — an address. Neither `CONTEXT_OVERFLOW_RE` nor `CTX_LIMIT_RE` (`config.py:122-123`) can match an address, so `parse_context_overflow_error` returned `None` at `:107` and the loop returned the upstream's `400` at `:125` after exactly one request. Rows 418 and 423 record the two halves; row 422 certifies `.text` as the only accessor on `Response` that returns the string's content.
- **Why `.text` and not the alternatives**: `Response.text` is `self._body.decode(self.charset or "utf-8")` (`web_response.py:723-730`). `StringPayload.__init__` encodes at `utf-8` (`payload.py:430-458`); the constructor at `:151-155` sets `content_type="application/json"` with **no** charset parameter, so `.charset` is `None` and `.text`'s fallback is that same codec — a strict round-trip. `json_response`'s 502 puts `charset=utf-8` on the header and stores encoded bytes, so `.text` reads it too. Verified by execution against the installed 3.13.3 (transcript in §Verification, and independently in `evidence/n6-reach-and-base-comparison.txt` §4).
- **No constructor changed** — done-when (1) held by the diff itself: +12/−1 in a single hunk, all of it inside the probe's statement block, and zero net executable lines.

**`tests/test_context_overflow.py`** (179 → 309 lines, +131/−1): module docstring's "Retry behavior in route handlers" claim made true; `TestStreamingOverflowRetry` added — 5 cases, one private recorder helper. Placed here rather than in a new module because this is the behaviour's home (test-cost rule 3): the file is `context overflow retry functionality`, and `TestBackendAwareRetry` is the class that has been asserting the *gate* into the loop while nothing entered it.

- **Why the bodies are snapshotted per call**: `handle_streaming` mutates **one** `current_body` dict in place (`:99`, `:111`), so `post.call_args_list` hands back the *final* body for every attempt. Asserting `sent[1]/sent[2] == 256` off the raw history is unfalsifiable — it stays green on a build that reduces the window **before** the first request, and it made my first draft red on `sent[0]` for that reason alone. `_sglang_stream` therefore copies `kwargs["json"]` inside a `side_effect`, which is what makes the counterexample's "reissue 2 and 3 carry the recomputed value" claim checkable. That is a correction the plan did not specify and no ledger covers; recorded here for Yield.
- **Value, not formula**: `256` is written as the literal (`_REDUCED_TO`), per done-when (2). It is `max(256, 4096 − 5000 − 100)` for the routed counterexample — the clamp, reached because the subtraction is negative. A change to `calculate_new_max_tokens` shows up here as a broken retry, not a pass. No witness asserts on the retry log line.

## Verification

- **Fail-first, replayed at pre-fix bytes with the *final* test code**: the expression restored to `result.body.decode() if isinstance(result.body, bytes) else str(result.body)` (comment block left in place, nothing else moved) → **2 failed / 3 passed**, `-v` transcript in [t019-001-fail-first-run.txt](../evidence/t019-001-fail-first-run.txt), pytest exit 1:
  - `[…][reduces_the_window_and_reissues]` — `AssertionError: the overflow retry loop did not run` / `assert 1 == 3`. This is the counterexample's own signature: one upstream request, not three.
  - `[…][answers_for_itself_when_the_retries_run_out]` — full diff printed: caller received `'Context overflow: input 5000 > 4096 tokens, max_completion_tokens 4096'` where the loop's `'Context overflow after 2 retries'` belongs. The terminal arm was dead, and this witness is what shows it now runs.
  - The three **pins** (`[plain-4xx]`, `[undecodable-payload]`, `[transport_502]`) stayed green in **both** states — that is their whole point, and it is the same disposition 083-01 recorded for its own class-guards.
  - Restored `result.text` → **22 passed** over the module, and the file is byte-identical to what the affected-scope run below measured. The mutation was applied and reverted by scripted replace with an assertion on each side, so the committed production diff contains none of it; `git status --porcelain buster_ripper/ tests/` shows only the two files this increment owns.
  - **Order note for Yield**: this transcript was captured **twice more** after the first capture, and the file in `evidence/` is the last one — at test bytes that already contain the per-call recorder and the role-named docstrings. The first capture (pre-recorder) would not have evidenced the snapshot claim, and the second would have printed source lines that no longer exist. The outcomes were identical in all three runs — 2 red / 3 green, same two ids — which is the honest way to state it: the *evidence* was re-cut, the *result* was not.
- **Yield's assigned replay** (blueprint §Yield replay) is executable on these bytes as written: `call_count == 3` + the literal `256`, the non-overflow single-relay witness, and the restore-the-ternary mutation → overflow witness red. Nothing in my witnesses is carried by the mock rather than by production.
- **Byte-exactness, by AST rather than grep** (done-when 7): `_error_body`, `_stream_single_request` (holds the `:151-155` constructor), `handle_non_streaming` and `_non_stream_single_request` (holds the non-stream probe), `_relay_after_commit` — all **IDENTICAL** between `HEAD:c96ca3e2` and the working tree, compared on `ast.unparse` of each module-level function. `buster_ripper/routes/chat.py` whole module **IDENTICAL**, still **300/300** raw lines against the Green ceiling.
- **Affected scope, once, at the checkpoint boundary**: `uv run pytest -q` (the project's canonical runner; `tests/qa/` excluded by `norecursedirs`) → **476 passed**, coverage **95.91%**, `--cov-fail-under=80` reached. Baseline 471 @ 95.91%: **+5 tests, none removed, coverage unchanged**. `routes/*` stays coverage-omitted by `pyproject.toml`, so the repaired line is covered by the witnesses above rather than by the coverage gate — the same reading 083-01 recorded for this file.
- **Warnings**: the run's 23 are the pre-existing `NotAppKeyWarning`s `create_app()` emits at `server.py:34/45/66` for every `proxy`-fixture test. No new warning class; confirmed by a single-module run (20 warnings, all three of those sites).
- **Lint**: `uvx ruff@0.16.10 check --select F buster_ripper/routes/messages.py tests/test_context_overflow.py` → **All checks passed**.

## L→N→C→P

| Surface | Baseline (`c96ca3e2`) | Final | Ratchet |
|---|---|---|---|
| `handle_streaming` | L:25 N:3 C:8 P:4 | **L:25 N:3 C:8 P:4** | unchanged — done-when (6) met exactly |
| `_stream_single_request` | L:29 N:2 C:3 | unchanged (AST-identical) | `QB:routes-messages/stream-single-request-gray-pair` untouched |
| `handle_messages` | L:28 N:2 **C:11** | unchanged | §RM-001's own decision held |
| `_error_body` / `handle_non_streaming` / `_non_stream_single_request` / `_relay_after_commit` | L:5 N:1 C:2 / L:21 N:2 C:5 / L:33 N:4 C:8 / L:9 N:1 C:3 | all unchanged (AST-identical) | pass |
| `tests/test_context_overflow.py` (file) | 179 (Green) | **309** — review band | recorded as `QB:test-context-overflow/file-size-yellow` |
| `_sglang_stream` (new test helper) | — | L:5 N:0 C:1 P:3 | all ideal |
| `record` (inner) | — | L:2 N:0 C:1 P:2 | all ideal |
| `[…][relays_once_and_unchanged]` | — | L:8 N:0 C:1 **P:4** | **1 gray** — the house shape of a parametrized route witness (`test_upstream_answer_relay.py:146`) |
| the four other new witnesses | — | L:5–7 N:0 C:1, P:3 | all ideal |

No new function exceeds the ideal targets on more than one metric; no already-violated metric grew. Scans run at the **final** bytes: `codify scan path buster_ripper/routes/messages.py --git --detailed` → 4 issues, all of them in `handle_streaming`, identical to the pre-edit scan; `codify scan path tests/test_context_overflow.py --git --detailed` → the 309-line file-band note and no function finding, because after Checkpoint 195 the only diff in that file is docstring prose — so the honest final figure is the whole-file pass, `codify scan path tests/test_context_overflow.py --detailed` → **1 issue in 1 function**, the P:4 witness above.

## Capacity gate

**Not applicable — no additive production surface.** Classified from the planned touch set before the edit: the single change at `:106` is **replacing** (one expression for one expression); nothing in this increment adds a statement, an arm, a function, a module or a configuration surface to production; nothing is **deleted**. The nine added lines are the comment block done-when (5) prescribes, and comments are not counted by the ratchet.

Evidence rather than assertion: `git diff --numstat c96ca3e2 -- buster_ripper/` → **12 added / 1 removed**, one hunk; `handle_streaming` re-measured at L:25 (identical to baseline, so zero executable statements entered it — the one statement added replaces the one removed); `wc -l` 323 → 334 with the +11 all prose. The gate's trigger — an additive surface extending an oversized file — has no instance to fire on.

The **misplacement check** `QB:routes-messages/file-size-yellow` owes at next touch was nevertheless **run**, because the file is at 334 inside the 301-500 band and this was its next touch. Declined, with the reasons recorded in that row: the candidate split moves `handle_streaming` itself, this task's own done-when (7) rules four surfaces on that relay path byte-exact (and AST proves they are), `S17`'s `Invalidates On` re-bases on contact with that surface, and the restructure is §RM-001's design decision under `062 R-A` / `DC-S10-C4-r1` — a one-member repair wave is not its drain. **No new `roadmap.md` row**: the deferred completeness (the handler/relay restructure) is already §RM-001's, and this increment ships the correct minimal core rather than a shortcut, so there is nothing above the MVP floor to park that §RM-001 does not already name.

## Backlog pickup

`Backlog pickup:` read against this increment's touch-set (`messages.py`, `tests/test_context_overflow.py`).

- **Absorbed by performing what the row owed**: `QB:routes-messages/file-size-yellow` — the re-measure (323 → 334) and the owed misplacement check, both written back into the row.
- **Advanced, not closed**: `QB:report-accuracy/001-collect-b005-retry-as-tested` — its own text says it is "closed by `T019.001`'s acceptance", which is Yield's act. Left `[ ]`, annotated with what now exists (`TestStreamingOverflowRetry`, 2 cases red pre-fix) and the transcript locator.
- **New row added from this increment's own footprint**: `QB:test-context-overflow/file-size-yellow` (the module crossed into the review band on my +131; every new function inside the ideal targets but the one P:4 house-shape witness).
- **Deliberately NOT absorbed**, each with its reason in the row:
  - `QB:routes-messages/handler-lncp` (`handle_messages` C:11) and `QB:routes-messages/deep-violations-unenumerated` (`handle_streaming` N:3, `_non_stream_single_request` L:33 N:4) — both §RM-001's design decision; the second row's own remedy is "add both to §RM-001's enumeration **at the next drain**", and its owning route is §RM-001's owner, not a repair wave's Forward. Not edited here; flagged in §Open so the next drain does not have to re-derive that the trigger has fired twice now.
  - `QB:routes-messages/stream-single-request-gray-pair` — its function is AST-identical; a row about a surface this increment did not move stays open unchanged.
  - `QB:routes-client-parse/lookup-error-admits-index-key-error`, `QB:routes-count-tokens/normalize-side-effects-on-estimate-and-malformed-bodies`, `QB:test-transport-parity/*`, `QB:test-compaction/file-size-yellow`, `QB:assumption-ledger/*` — outside this touch-set, each with a named owner.

## Dependency claims — what was checked, what moved

| Claim encoded in this increment's line | The dependency does | Read at | Changed anything? |
|---|---|---|---|
| `Response.text` returns the string that was passed to `body=` (row 422, `DOES-NOT-HOLD` — the accessor exists) | `if self._body is None: return None` / `return self._body.decode(self.charset or "utf-8")` | `web_response.py:723-730` | no |
| `str=` given to `body=` is stored as a payload, never as its bytes (rows 418/423) | `self._body = body = payload.PAYLOAD_REGISTRY.get(body)`; `PAYLOAD_REGISTRY.register(StringPayload, str)`; getter is a bare `return self._body` | `web_response.py:693-721`, `payload.py:1111` | no |
| The encode and decode codecs agree at this site, so `.text` cannot raise here | `StringPayload.__init__` picks `real_encoding = "utf-8"` when neither `encoding` nor a charset-bearing `content_type` is given; `HeadersMixin.charset` returns the Content-Type's `charset` parameter, absent for `application/json`, so `.text` falls back to `"utf-8"` | `payload.py:430-458`, `helpers.py:769-776`, `web_response.py:647-683` (`__init__`'s header construction) | no |
| `json_response`'s 502 is the *other* shape, and reads strictly too | `text=dumps(data)` → `real_headers[CONTENT_TYPE] = "application/json; charset=utf-8"`, `body = text.encode("utf-8")` → the setter's `isinstance(body, (bytes, bytearray))` branch stores raw bytes | `web_response.py:833-856`, `:654-668`, `:701-702` | no |
| `str(StringPayload)` is an object repr (row 423 — why the old `else` arm could not work) | no `__str__`/`__repr__` on `StringPayload` → `BytesPayload` → `Payload(ABC)` → `object` | `payload.py:151,350,430-458` + grep for `def __str__`/`def __repr__` over `payload.py` | no |
| `calculate_new_max_tokens(4096, 5000)` is 256 — the literal the witness asserts | `max(config.MIN_COMPLETION_TOKENS, ctx_limit - input_tokens - 100)`, `MIN_COMPLETION_TOKENS = 256` | `utils.py:143-149`, `config.py:124` | no |

Reach check (`assumptions/claims.md`): rows **418**, **422**, **423** are each `absorbed` against **`T019.001` only** — read from the Routing column at `claims.md:2993-2996`, not inferred. None is listed against another task, so nothing here is a shared claim a local fix could strand. Row **419** is `malformed` and back with the collector; its successor row gates `S20`'s recomposition, and this pass neither wrote nor closed it. `claims.md` exists for this session, so no "assumption pass has not run" caveat applies. Checks that **confirmed** a claim (all six rows above) are recorded here because this is the increment's whole evidence base, not because any moved: **no dependency reading contradicted the plan**, and nothing was corrected silently.

## Pointer discipline

The blueprint, the three ledgers and `claims.md` all cite this site by its coordinates at wave base `c96ca3e2`: the read at `:106`, the terminal arm at `:119-123`, the constructor at `:151-155`, the static 502 at `:171-173`. **This increment moved those numbers**, because the note done-when (5) requires is 12 lines of prose inserted at `:106`: every coordinate at or below the old `:107` is now **+11**, and everything at or above `:105` is unchanged. Verified by measurement, not arithmetic — the guard is still `:105`, the read is now `:117`, the terminal arm's body is `:130-134`, the constructor `:162-166`, the 502 arm `:182-184`, the non-stream probe `:307`/`:311`.

This report and the plan-side records **keep the blueprint's numbers when quoting the plan** — that is what a quotation is, and Yield's replay instructions address those coordinates. What must not happen is the reverse: code this pass *authored* naming the neighbours it had just renumbered. My first draft did exactly that — the in-source note named `:151-155` and `:171-173`, and two witness docstrings named `:119-123`, `:171-173` and `:106`, all of which the insertion invalidated. Corrected at Checkpoint 196: **the note and the witness docstrings now name those two arms by role** (`_stream_single_request`'s `status >= 400` arm, its `ClientError` arm), because the sites live below the probe in the same function and this file renumbers on every edit. Same defect class 081 was corrected for at Checkpoint 189, caught here by re-reading my own citations after the fact rather than by a reviewer.

## Concerns for Yield

1. **The per-call body snapshot is my addition, not the blueprint's.** Done-when (2) asked for `post.call_count == 3` and for the 2nd/3rd bodies to carry `256`. Read literally against `call_args_list` that second clause cannot fail, because the loop mutates one dict — so the witness as specified would credit a build that pre-reduces the first request. I made it checkable at the cost of one nested function. If Yield rules the literal reading was intended, the change is one line to revert and the `call_count == 3` half still carries the repair.
2. **`sent[0]` is asserted to be *unedited*, which the counterexample does not require.** It is the free half of the snapshot: once bodies are recorded per call, the assertion that the first request still carries no `max_completion_tokens` costs one line and forecloses the pre-reduction build named above. Say so plainly rather than leaving it as an unexplained extra.
3. **`_REDUCED_TO = 256` sits at module scope next to the fixture text** rather than inline in the two assertions. It is the value the routed counterexample computes to, and the comment says so; if Yield prefers the literal inlined twice, that is a style call on the witness, not on the repair.
4. **Two backlog triggers fired without absorption, both on purpose** — `QB:routes-messages/deep-violations-unenumerated` ("any touch of `messages.py`'s two relay arms") and the file-size row's "next touch". The first's remedy is a `roadmap.md` enumeration edit owned by §RM-001's owner at a drain; I did not edit another route's record. Yield can rule otherwise and it is a one-line amendment.
5. **`chat.py`'s sibling probe was re-read and is fine.** `chat.py:178` feeds the matcher `json.dumps(resp_data)` — real text, never a repr — so the same defect class does not live on the chat boundary (`evidence/n6-reach-and-base-comparison.txt` §2 census: one `str` enters `body=`, one site reads `.body` back, and they are both this pair). Recorded so the next reader does not re-open the census from scratch. Nothing was changed there.
6. **My first draft carried a citation defect of this session's own known class, and it is fixed at Checkpoint 196, not at 195.** Checkpoint 195's in-source note and two witness docstrings named neighbouring sites by the blueprint's pre-shift line numbers, which the note's own 12 inserted lines invalidated. §Pointer discipline states the shift rule (+11 at or below the old `:107`), the measured positions, and that the authored prose now names those arms by role. Yield's replay reads the plan's coordinates, so nothing in §Yield replay changes; if Yield prefers numbers in the note instead, that is a one-line edit back.

## Questions asked

**Zero.** No `AskUserQuestion`, no `BLOCKED-APPROVAL`, no `ESCALATION`, no `PLAN-GAP`. The pre-question gate had nothing to route: every choice this increment faced was settled by the blueprint's own ranking (the rejected-candidates list forecloses `text=`, the bytes-returning `_error_body`, deleting the loop, accepting-as-risk, and roadmap deferral), and the one implementation addition (§Concerns 1) is the smallest spelling that makes a required assertion able to fail — an admissible outcome with no materially different alternative left unranked. `FK-011` remains the session's only open Fork and was neither touched nor consulted.

## Invariant Contradictions Raised

**None.** No invariant was found wrong, stale, incomplete or blocking; no `invariants/` or `decisions.md` byte written. The blueprint's **Invariant candidate** ("Reading a body back off an `aiohttp.web.Response` the code constructed goes through `.text`; `body=` given a `str` stores a payload object, so `.body` is never that string's bytes") is left exactly where `084` placed it — a candidate in `T019.001`'s block for Interview to rule on. It is *now evidenced* by this increment's dependency table and by `claim-422`'s accessor enumeration, which is the record Interview needs; recording and promotion remain that route's act.

## Open

- `S20` cannot recompose until row 425 lands (`084`); this pass closes no verdict and recomposes no seam. `S16`–`S19`'s first recompositions remain the MS-001 boundary's assumptions recheck, which the chronicle places **after** this task.
- `T019.001` is **implemented, not resolved**: `resolved` for `T018`/`T019`/`T020` and the phase acceptance are Yield Red's, and this increment's own acceptance is the local Yield's.
- **Next**: local Yield for this one-member wave (replays: `call_count == 3` + literal `256`, the non-overflow single relay, and the restore-the-ternary mutation → overflow witness red), then the MS-001 boundary's assumptions recheck on these bytes → Yield Red → `phase-boundary-finalize`.

DEMO-EVIDENCE: [.sessions/codify/02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/evidence/INDEX.md](../evidence/INDEX.md) — no pause

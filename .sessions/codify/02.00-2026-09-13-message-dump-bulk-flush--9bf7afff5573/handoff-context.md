# Session Context

## Session Metadata
- Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573 · UID: 9bf7afff-5573-402a-b102-827423a1dded
- Topic: Buffer captured messages in memory, bulk-flush dump to disk, flush on Ctrl-C/SIGTERM · Created: 2026-09-13
- Last Updated: **PR #1 review round 5 routed at [127](iterations/127-assumption-decide.md)** (2026-10-19, `713d3af`) — session closed at `BOUNDARY-FINALIZED` [123-01](iterations/123-01-forward.md): Review object `4250ae6`; base `main @ 72e3a1c` an ancestor; evidence `INDEX.md` **39/39 ruled** ([`US-017`](decisions.md#us-017): 10 keep / 29 remove); roadmap **21/21 promoted and landed**; decisions **11 FK + 17 US, 0 reserved**; final code bytes green at `793a5f0` (**551 passed**: 513 @ 96.71% + 17/15/6 gate). `T024`/`T024.001` `resolved`. Checkpoint 274.

- Planning CLOSED, plan implementable (final-approval answer [`US-015`](decisions.md#us-015), round 18 gate record [020](iterations/020-interview.md) §Round 18; registry `invariants/` elevated, `CHECK OK`); no fork owed.
- **Interview Finalize approval — historical (2026-09-17), Registrar-projected (2026-09-27):** round-18 "approve, begin" (`US-015`) is recorded historical implementation authorization, not a newly resolved final-approval FK or fresh `FINALIZED` gate result. The workflow-contract blocker stays with its owner ([040](iterations/040-decide.md)); `US-009` c2 → `yield-phase:MS-001` was adjudicated **HOLDS** at [071-yield-phase](iterations/071-yield-phase.md) (2026-10-06, `Consequence: none`) — **the session now has no PENDING verification row in `decisions.md`**; `FK-008` c1/c2 → `yield-phase:MS-004` were adjudicated **HOLDS** at [056-yield-phase](iterations/056-yield-phase.md) (2026-10-05). Registry: 0 new/updated/superseded/upheld; no new Forward permission inferred beyond round-18 authorization.


## Current Status

- **PR #1 review round 5 routed at [127-assumption-decide](iterations/127-assumption-decide.md) (2026-10-19, `713d3af`)**:
  three new P2 root findings (4318096152/59/66). **Two `absorbed`** into **one wave, one brief** — the messages
  retry control field (#2, both arms) and the preserved-thinking SGLang skip (#3, plus the README/help/config-comment
  corrections) — and **one `accepted`**: the non-atomic append + retry (#1, with the `_append_index_line` sibling) is the
  1% `RM-023` record, so `dump.py`'s two append sites change no code this round. No user question; the `Ask: yes` line
  waits for the summary. `chat.py` untouched at 299/300; `messages.py` 339 → ≤342 of 344; `dump.py` untouched at 319.
- **⚑ Gate upstream `http://hyperipper.local:30000`** (user correction 2026-10-17; `:30001` is a peer Buster Ripper proxy).
- **Context trimmed to this cap**: superseded blocks are verbatim in [handoff-archive.md](handoff-archive.md)
  §Trimmed 2026-10-19 (`127-assumption-decide`).



## Open Forks / Retries
- **NONE OPEN.** `FK-011 · preserved-thinking serve default` — resolved 2026-10-18, **Option A** (keep the default on, document it), recorded at [115](iterations/115-interview-reconcile.md) and registered 2026-10-18 as **T023 · document `--preserved-thinking` and `PRESERVED_THINKING` in `README.md`'s serve-script section**; the user's objection to the Fork's admission is preserved verbatim there as a routing-quality note for `assumption-decide`. **`T023` is accepted and `resolved` at [116-01-yield](iterations/116-01-yield.md)** — zero reject-tier findings.
- None else open, none else owed. **`FK-010` · serve-script disable form** — resolved 2026-10-03, **option 1**, delivered by `T011` (`accepted-local`, [052-01-yield](iterations/052-01-yield.md)); all `FK-001`…`FK-011` resolved or superseded — canonical records in [decisions.md](decisions.md).


## Forward->Yield Retries
- Every counter on this boundary is 0 except the two `T024.001` rows below; the completed waves' counters and closures are verbatim in [handoff-archive.md](handoff-archive.md) §Trimmed 2026-10-17 (`097-yield`); the run-49 gate-repair wave (`T021.001`/`T021.002`) closed at 0/0; the `T024` census-count row closed at Yield (1 rejection, corrected, no route).
- T024.001 / `quality-backlog.md` Refactoring marker form: rejections 1 ([119-01-yield](iterations/119-01-yield.md)), `absorbed` → `DC-T024-Y-1-r1`
- T024.001 / `handoff-context.md` stray `<details>`: rejections 1 ([119-01-yield](iterations/119-01-yield.md)), `absorbed` → `DC-T024-Y-3-r1`

## Notes (load-bearing for remaining tasks)
- Moved verbatim to [handoff-archive.md](handoff-archive.md) §Trimmed 2026-10-18 (`098-assumption-decide`); the settled decisions and protected invariants remain canonical in `decisions.md`/`plan-summary.md`.

## Planned Task Context
- **PR #1 review round 5 (2026-10-19, `713d3af`) — repair lane, ONE wave (one brief); the two `dump.py` append sites are the 1% record `PR1-R5-1`, no task.** The CODIFY session stays closed; the vehicle is a lite brief under `00bab65:.sessions/lite/`. Round line: **round 5, late**. The brief covers the messages retry control field and the preserved-thinking backend gate, plus the three doc corrections; `dump.py` stays untouched.

### Wave 1 — `2026-10-19-pr1-r5-messages` (`messages.py`, `config.py`, `main.py`, `README.md`)

- **Anchor**: lite brief `2026-10-19-pr1-r5-messages`.
- **Members**: finding 2 (4318096159, both retry arms); finding 3 (4318096166); family siblings C1 (`config.py:45-49`), C2 (`main.py:107-110`), C3 (`README.md:94-104`).
- **Evidence**: [127](iterations/127-assumption-decide.md) §Family B / §Family C; [comment 4318096159](https://github.com/AlexGS74/buster-ripper/pull/1#discussion_r4318096159); [comment 4318096166](https://github.com/AlexGS74/buster-ripper/pull/1#discussion_r4318096166).
- **Prerequisite**: none. One brief because all five edits are the same `messages.py` request-body concern; editing the same file alone is not interference.
- **Owning route**: `lite-forward` → `lite-yield`, one brief.
- **Separation reason**: none — the retry field and the backend gate are independent edits in the same file, and the doc lines are the same claim.
- **Capacity shape**: `handle_messages` (`L:28 N:2 C:11`) is a hard C violation, so the log and call swap replace two statements and add none; `_inject_preserved_thinking` is Green and gains the guard (+3 raw lines). `_stream_single_request` / `_non_stream_single_request` are hard Length violations, so #2 changes one field name per site and adds no statement. `messages.py` ≤ **344** and does not cross 500.
- **Done when**: (a) on a mocked `"sglang"` backend with `PRESERVED_THINKING` on, a `/v1/messages` body is forwarded without an injected `chat_template_kwargs` (a client-sent one is forwarded unchanged), and on `"vllm"`/`"unknown"` the pair is still injected exactly as today; (b) an SGLang streaming overflow reissues attempts 2 and 3 with `max_tokens` reduced (and no `max_completion_tokens` added), and the non-streaming arm does the same; (c) the chat retry still carries `max_completion_tokens` and the chat `chat_template_kwargs` still carries `enable_thinking`/`thinking_budget`; (d) the README, CLI help and config comment state the SGLang v0.5.9 `/v1/messages` limitation and the vLLM Anthropic support; each test red before the fix.
- **Family sweep**: every durable append audited in [127](iterations/127-assumption-decide.md) §Family A (the two `dump.py` append sites are the 1% record, not repaired here); every retry control field audited in §Family B (the chat arm is correct and unchanged); every injected option audited in §Family C (the chat arm's `chat_template_kwargs` is supported on both chat endpoints and unchanged).
- **Invariant cross-reference**: Neutral — zero active/wip registry invariants; `PI:T002/dump_format/*` and `PI:T006/upstream_tls/*` are `planned` and untouched; no invariant candidate.
- **Closure ledger**: `DC-PR1-R5-1-r1` (#2), `DC-PR1-R5-2-r1` (#3), `DC-PR1-R5-3-r1` (C1–C3).

- **PR #1 review round 4 Planned Task Context** (two waves, `6a089d8`) and its ten closure rows are verbatim in [handoff-archive.md](handoff-archive.md) §Trimmed 2026-10-19 (`127-assumption-decide`).

- **PR #1 review round 3 Planned Task Context** (three briefs, `f37d475`) is verbatim in [handoff-archive.md](handoff-archive.md) §Trimmed 2026-10-19 (`126-assumption-decide`).
- **PR #1 review round 1 Planned Task Context** (three waves, `0d465ec`) is verbatim in [handoff-archive.md](handoff-archive.md) §Trimmed 2026-10-19 (`126-assumption-decide`).
## Deterministic Closure Ledger
- Live rows: the three round-5 rows below. **Every earlier row** — `DC-CR4-F-01-r1`; run-45 `DC-claim-418/422/423-r1`;
  run-42 `DC-S16-a-r1`, `DC-S17-CE-1/2-r1`, `DC-S18-CE-1-r1`/`-MC-1-r1`, `DC-S19-CE-1/2/3-r1`; CR3 `DC-CR3-F-01…05-r1`;
  run-40 `DC-claim-399/409-r1`; CR2 `DC-CR2-F-1…4-r1`; run-49 `DC-S21-r1`/`DC-S22-r1`; run-50 `DC-S21-r2`;
  round-1/3/4 `DC-PR1-R1-*`/`-R3-*`/`-R4-*`; `DC-T022-Y-1-r1` — is verbatim in
  [handoff-archive.md](handoff-archive.md) §Trimmed 2026-10-17/18/19. `DC-S21-r2` is **accepted-local** (`T021.003`, [109](iterations/109-yield.md));
  every other row is implemented and accepted-local; `resolved` is published by the closing Yield Red `YR-MS001-r4`
  ([095](iterations/095-yield-red.md) §Adjudication).

### DC-PR1-R5-1-r1
Finding: PR1-R5-2 (PR #1 review comment 4318096159, P2 — `messages.py:113` / `:283` write `max_completion_tokens` on an Anthropic body whose only declared budget field is `max_tokens`)
Evidence: https://github.com/AlexGS74/buster-ripper/pull/1#discussion_r4318096159; iterations/127-assumption-decide.md §Recheck against the bytes; §Family B
Evidence fingerprint: unavailable
Repository state: git-commit:713d3af14da26ee52bc1e7ff70e9cbc147a5f0ac
Selected closure: assign `current_body["max_tokens"] = new_max` in both retry loops (`handle_streaming:113`, `handle_non_streaming:283`) and say `max_tokens` in both log lines, so the reduced generation budget is the field SGLang's Anthropic request declares and copies unchanged.
Closure basis: SGLang v0.5.9 `AnthropicMessagesRequest` declares `max_tokens` and `serving.py:197` copies it unchanged; `calculate_new_max_tokens` and both log lines name the generation budget; the chat sibling's `max_completion_tokens` is the correct field on the chat endpoint, not on this one; base parity has no retry at all (`e96b6c9` added it at Checkpoint 05).
Rejected candidates:
- write both `max_tokens` and `max_completion_tokens`: the extra field is dropped by the Anthropic schema and the log then names two fields for one budget.
- move the retry to the chat endpoint: changes the client's protocol and cannot answer `/v1/messages`.
- leave it and record: preserves a retry that can never recover.
Risk shape: unchanged - two field names and two log strings; the probe, the dispatch and every response arm are byte-unchanged.
Deferred additive work: none.
Task: lite brief `00bab65:.sessions/lite/2026-10-19-pr1-r5-messages/brief.md`, step 1

### DC-PR1-R5-2-r1
Finding: PR1-R5-3 (PR #1 review comment 4318096166, P2 — `messages.py:78` injects `chat_template_kwargs` on `/v1/messages`, a field SGLang v0.5.9's Anthropic schema drops)
Evidence: https://github.com/AlexGS74/buster-ripper/pull/1#discussion_r4318096166; iterations/127-assumption-decide.md §Recheck against the bytes; §Family C
Evidence fingerprint: unavailable
Repository state: git-commit:713d3af14da26ee52bc1e7ff70e9cbc147a5f0ac
Selected closure: skip `_inject_preserved_thinking` when `backend.get_backend_state() == "sglang"` and return whether the pair was injected, so the `handle_messages` log never claims a skipped injection; keep injecting on `"vllm"` and `"unknown"`.
Closure basis: `FK-011` Option A ("keep the default on and document it") is preserved - the default stays on and the pair still goes out where the endpoint declares it; SGLang v0.5.9's Anthropic `AnthropicMessagesRequest` declares no `chat_template_kwargs` and no thinking field, and its conversion uses declared fields only; vLLM 0.26.1rc0 declares `chat_template_kwargs` (`protocol.py:146`) and forwards it (`serving.py:477`); the Reach Gate's "refuse before handling" ranks the fail-closed skip over an ignored extension.
Rejected candidates:
- keep injecting on SGLang and correct only the docs: leaves the ignored extension the reviewer names and a branch that does nothing.
- translate to a SGLang Anthropic field: v0.5.9 declares no thinking field at all, so there is nothing to translate to.
- rewrite to `/v1/chat/completions`: changes the response protocol the client parses.
- require a newer SGLang: the repo does not control the upstream and no verified version supports the field.
Risk shape: unchanged - one guard and a return value in a small function, two statement swaps in `handle_messages`; the chat route, the dump path and every response arm are byte-unchanged.
Deferred additive work: none.
Task: lite brief `00bab65:.sessions/lite/2026-10-19-pr1-r5-messages/brief.md`, step 1

### DC-PR1-R5-3-r1
Finding: PR1-R5-C1/C2/C3 (family sweep of 4318096166 — `config.py:45-49`, `main.py:107-110`, `README.md:94-104` advertise the pair without the SGLang limitation)
Evidence: iterations/127-assumption-decide.md §Family C; §Recheck against the bytes
Evidence fingerprint: unavailable
Repository state: git-commit:713d3af14da26ee52bc1e7ff70e9cbc147a5f0ac
Selected closure: correct the three advertised surfaces in the same brief - the `config.PRESERVED_THINKING` comment, the `--preserved-thinking` help string and the README's §Serve script options - to name the SGLang v0.5.9 `/v1/messages` limitation and the vLLM Anthropic support, without moving the default-on posture or the injection where it is honored.
Closure basis: `T023`'s accepted README section is the documented choice under `FK-011` Option A; the code now skips only on a confirmed SGLang backend, so the advertised claim must match it; the CLI help and the config comment are the same claim on the other two surfaces a user reads first.
Rejected candidates:
- leave the wording and rely on the finding's reply: leaves the advertised default false on two user-facing surfaces.
- delete the flag or its README section: reverses `T023` and `FK-011` Option A.
Risk shape: unchanged - three doc strings; no code, test or record byte moves.
Deferred additive work: none.
Task: lite brief `00bab65:.sessions/lite/2026-10-19-pr1-r5-messages/brief.md`, step 1


## Capped seams
- **`S21` · diff-text seam — recomposition owed, capped in outcome, deferred with a named trigger.** Its composition `assumptions/compositions/S21.md` (`sha256:7f4c55b7…`, checked 2026-10-18T02:55:16Z, `COUNTEREXAMPLE`) is stale on these bytes: `Invalidates On` 2/3 fired (`test_qap_ms001.py` `0b3b052c…` → `477b603c…`, `harness.py` `0ac81b37…` → `58f70198…`) and its Transformation 3 names `read_text`'s universal mode, which no longer exists. `YR-P021` finds the two comparison reads (`:276`, `:465`) and the regression's own two (`:713`, `:718`) raw; `YR-P022`/`YR-P023`/`YR-P024` HOLD. The single post-repair allowance was spent at 107 (run 51), so a further non-`SAFE` outcome is **capped**: accept what it permits, route nothing, open no task/Fork/wave. Ledger row stays `recheck`/`absorbed`/`—`. Reach: **latent** (no `\r` byte under `tests/qa/`). Deferred-work handle `QB:assumption-registry/s21-diff-text-identity`; trigger = the next `S21` composition, the next promotion of member rows 426–431/438–440, or any later byte change to `test_qap_ms001.py`/`harness.py`/`dump.py`. **PR #1 round 3 (2026-10-19) fires the byte-change trigger**: `dump.py`'s `_unified_diff` and `test_qap_ms001.py`'s expected side move under `DC-PR1-R3-3-r1`. The seam stays **capped** — the single recomposition allowance was spent at 107 (run 51) — so the owed reading is deferred to the named trigger and **no task, Fork or wave opens from it**; the repair brief's Done-when carries the re-pointed gate so the reading is not silently skipped.
- **`S22` · upstream/preflight seam — `SAFE`, delta re-trace owed as a record.** `Invalidates On` 1/2 fired (`harness.py` `ed105893…` → `0ac81b37…` → `58f7019…`); substance unchanged (`require_upstream` `ProxyHandler({})` at `harness.py:57-58`, `Proxy.post` at `:291`). Reach: **latent** (ambient proxy settings are environmental). Trigger = the next composition dispatch on current bytes, or any byte change to those two sites.
- **`S23` · environment/spawn seam — `SAFE`, delta re-trace owed as a record.** `Invalidates On` 2/3 fired; cited lines moved (fixture `:95-99` → `:112-117`, `Popen :259` → `:266`); substance unchanged (fixture deletes every `DUMP*` name, spawn inherits the environment with no `env=`). Reach: **latent**. Trigger = the next composition dispatch on current bytes, or any byte change to the fixture or spawn.
- Nothing is opened from this list. The three rows above are the capped/deferred material from [095](iterations/095-yield-red.md) §Capped and deferred seam records, presented to the user in one message; no task, Fork, wave or repair may start from them before the named trigger fires.

## Accepted risks
- Every accepted risk (`CR4-F-01` second half, `CR3-F-11`, `S17-sibling-arms`, `S16-decompress`, `claim-425`, `CR5-F-01`) is verbatim in [handoff-archive.md](handoff-archive.md) §Trimmed 2026-10-17 (`097-yield`); each keeps its own `Reported:` state. No new accepted risk was added by run 49 or run 50 — `S21` CE-2 was `absorbed` into `T021.003`, not accepted.
- `PR1-R1-4` — PR #1 review comment [4309720775](https://github.com/AlexGS74/buster-ripper/pull/1#discussion_r4309720775), `utils.py:27`: a request body the server already inflated is forwarded under the client's `Content-Encoding` label. Reach **`latent`** — no documented client compresses a request body and the deployed launcher sends none; base never decompressed one. **1% category**, so no repair: the row is `RM-011` (`RM-008` is the response-direction sibling; the reviewer's citation is corrected), and the §Elevated to user `S16-decompress` line already carries the risk. Evidence: [124-assumption-decide](iterations/124-assumption-decide.md) §Findings and dispositions.
- `PR1-R1-6` — PR #1 review comment [4309720779](https://github.com/AlexGS74/buster-ripper/pull/1#discussion_r4309720779), `dump.py:156`: a same-format restart appends a duplicate `turn 000` and breaks the cross-restart diff chain. Reach **`latent`** — needs a restart with the same `DUMP_DIR`, session id and collapsed format. **1% category**, so no repair: `RM-022`; see §Elevated to user. Evidence: [124-assumption-decide](iterations/124-assumption-decide.md) §Why 6 is the 1% category.
- `PR1-R4-MSG` — family sweep of PR #1 review comment [4317334042](https://github.com/AlexGS74/buster-ripper/pull/1#discussion_r4317334042), `routes/messages.py:29-33` + `normalize.py:88/89/90/130/131/205/209`: a syntactically valid non-object `/v1/messages` body, and a `tools`/`messages` array carrying a non-mapping element, reach a mapping access that raises — measured **500** on `{"tools":[null]}`, `{"messages":[null]}`, `{"messages":["x"]}` and on `null`. Reach **`reached`** (exposed route; measured). **Not this wave's**, because `RM-012` records three admissible answers (400 with the repo's `{"error": …}` envelope, tolerate-and-forward, validate-and-drop) that no settled authority ranks; `G-003` holds `/v1/messages` behavior still; and `handle_messages` carries a hard **C:11** violation (`L:28 N:2 C:11`), so an additive guard triggers the capacity gate and requires `RM-001`'s extraction first. Owner: `roadmap.md` §RM-012 (root shape added to its owed work). Evidence: [126-assumption-decide](iterations/126-assumption-decide.md) §Findings and dispositions (MSG row), §Family A.
- `PR1-R5-1` — PR #1 review comment [4318096152](https://github.com/AlexGS74/buster-ripper/pull/1#discussion_r4318096152), `dump.py:300-307` (collapsed batch append) and `:195-200` (`index.txt` append): a failed append that wrote a prefix is retried with the whole batch, so the artifact carries a torn prefix followed by the batch (measured 7 bytes + `pending=1` → 483 bytes, head `turn 00turn 000 …`) and `index.txt` joins the torn line to the retried summary (measured `turn 000 turn 000 …`). Reach **`latent`** — the collapsed artifact is opt-in, the documented index append is on the default path, and a mid-write `ENOSPC`/`EIO`/`EDQUOT` is an environmental condition no guard prevents. **1% category** — the trigger is unusual but plausible, the result persists in an append-only artifact, and the round is late, so the late-round rule records it even though the fix is small; no normal-use finding sends a repair into `dump.py` this round. `RM-023`. `Ask: yes — whether to take the eight-line rollback now or leave it on the roadmap with the risk accepted.` `Reported: —`

## Elevated to user

- `S15-CE-1` — a proxied answer can arrive labelled with an encoding the proxy already undid, and a
  conforming client then fails on bytes that are plain (`curl: (61) Error while processing content
  unencoding: incorrect header check` — reproduced end-to-end through the real routes against a
  compressing peer, `evidence/s15-reach-and-base-comparison.txt` §1e). Trigger is narrow: it needs an
  upstream or gateway in front of it that *accepts* compression, and the deployed vLLM/uvicorn does not,
  at any size, on any offer. Costs one broken proxied request per affected call when it does fire; the
  upstream having succeeded is what makes it look like the proxy's fault. Proper fix lives in
  `buster_ripper/utils.py`'s shared filter or in the proxy's egress `Accept-Encoding` posture — a product
  decision, not a defect fix: **relay decoded and stop labelling**, **ask for `identity` and relay
  verbatim**, or **relay the wire bytes and decode only for the proxy's own parsing** (the last collides
  with `messages.py:272/276` and `chat.py:163`, which need the parsed form). Roadmap handle:
  `roadmap.md` §RM-008. `Reported: —`
- `CR3-F-11` — a `message_start` SSE event split across network chunks is silently missed, and the turn's
  session stats (TTFT + input tokens) never record; what happens: one turn's compaction inputs are missing and
  the session-trigger nudge rides the previous turn's numbers; how narrow the trigger: the chunk boundary must
  land inside the first SSE event, on a self-healing path (the next turn re-records). Cost when it fires: one
  turn's telemetry, no client-visible harm. Proper fix lives in `buster_ripper/utils.py:91`'s per-chunk scan
  (or a relay-side carry buffer) — a small change on the accepted relay surface, deliberately not made mid-wave:
  the behavior is **base parity** (`buster_ripper_legacy.py:674-676`, re-verified), so nothing in this batch
  regressed it. Roadmap handle: `roadmap.md` §RM-009. `Reported: —`
- `S17-sibling-arms` — an upstream **timeout** on three of the four proxy arms is answered by aiohttp, not by this
  proxy: the deadline the shared session arms (`total=600` inherited by `messages.py:265-297`; 300 s at
  `chat.py:139/150`; 60 s at `passthrough.py:27/48`) raises bare `asyncio.TimeoutError`, which those arms'
  `except aiohttp.ClientError` does not name (rows 330/331), so the protocol's `:518` arm answers with a generated
  **504** + `text/plain` + a traceback log instead of the site's own 502 + `{"error": "Upstream request failed"}` +
  `"Upstream error: %s"`. What happens and how narrow it is: any upstream that takes longer than the armed budget
  on a non-streaming call — ordinary network time at 60 s, less ordinary at 600 s, but reachable at all three; the
  `/v1/messages` **streaming** path is not in this set (closed by `T018` step 2). Cost when it fires: a client that
  parses the proxy's error envelope gets a body it did not expect, and the operator's log says "Error handling
  request" with a traceback rather than naming the upstream. Proper fix lives in the three arms' class lists (or in
  one shared wrapper) — small in lines, not in cost: `chat.py` sits at **300/300** raw lines, so it drags §RM-001's
  route restructure with it, and it changes a client-visible status (504 → 502) that **base settled worse** (base
  had no arm at all: an `httpx` timeout propagated to Starlette and became a 500 — `buster_ripper_legacy.py`, no
  `httpx.HTTPError` anywhere). Needs no decision to *route*: it is out of this session's scope and blocks no
  done-when. Roadmap handle: `roadmap.md` §RM-010. `Reported: —`
  — **PR #1 review round 1 (2026-10-19) takes the whole arm family up**: the user's round-1 instruction
  overrides `RM-010`'s posture deferral, and the 502 is uniquely ranked by the repo's own authored envelope
  (every repo-authored client-facing failure is `{"error": …}`; 504 is aiohttp's generated `text/plain` body).
  Repaired in the `2026-10-18-pr1-transport-parity` lite brief; `Reported: —` stands until the fixing commit exists.
- `S16-decompress` — the aiohttp server decompresses **request** bodies transparently, and nothing in the relay says
  so. What happens: a client sending `Content-Encoding: gzip` has its body decoded before the handler sees it, so the
  body forwarded upstream is plain bytes travelling under the peer's own `Content-Encoding` label
  (`passthrough.py:14`/`:31`; the two `json=` paths re-serialize a body the peer labelled encoded), and a body whose
  *decompressed* output exceeds `2**25` in one feed raises `DecompressSizeError` — not an `HTTPException` — so the
  protocol's generic arm answers a generated **500** where a size limit would want a 413. How narrow the trigger: it
  needs a compressing client, which nothing in this package's documented surface asks for and the deployed launcher
  never sends; the 32 MiB half needs a body of that size after inflation. Cost when it fires: one failed or
  mislabelled upstream request, no data loss, nothing silent on the common path. Proper fix lives in the request
  direction's content-coding posture — the shared filter (`utils.py:20-26`), an egress-side re-encode, or a
  deliberate `Content-Encoding` drop — which is the same **design** decision §RM-008 already holds for the *response*
  direction; base could not produce this state because it never decompressed a request body (zero
  `gzip`/`decompress`/`Content-Encoding` references in `buster_ripper_legacy.py`). Needs no decision to route; it
  blocks nothing in this wave. Roadmap handle: `roadmap.md` §RM-011. `Reported: —`
  — **PR #1 review round 1 (2026-10-19) re-confirms the row and corrects the reviewer's handle**: the comment
  cites `RM-008`, but the request-direction half is `RM-011`; `RM-008` is the response-direction sibling. Both stay
  1% records; no repair is taken. `Reported: —` stands.
- `CR4-F-01` (second half) — **a schema-invalid request body is answered with a generated 500 on `/v1/messages`,
  and this session deliberately leaves it that way because base does.** What happens: `{"tools":[null]}`,
  `{"messages":[null]}` or `{"messages":["x"]}` at `/v1/messages` raises `AttributeError` inside `normalize()`'s
  element loops, the traceback escapes the handler, and aiohttp authors the answer. How narrow the trigger: malformed
  per the Anthropic schema, so the documented client never sends it; exposed route, so nothing *prevents* it.
  Measured, not reasoned: base raises the same exception at the same point in its own `normalize()`
  (`evidence/cr4-probe-matrix.txt` §3), and the package did so before the range and after it (§1) — so this is not a
  500 this migration brought, unlike `count_tokens`' new one, which is being repaired in `T019.002`. Why it is not
  fixed here: hardening those loops changes `/v1/messages` from 500 to relay traffic (502 with the upstream
  unreachable; §7 measured), which is the proxy answering differently than base on the one route `G-003` holds
  still, whose constraint column says "no normalization changes" — and a two-loop hardening would not even be
  complete (`normalize.py:192` inherits the crash the moment `:117` is hardened, §6). Cost if it is never taken up: a
  malformed request gets a 500 and a server-side traceback instead of a 400 the proxy authors; no wrong data, no
  leak, nothing silent on a valid request. Needs no decision to *route* — it opens no task and blocks no done-when —
  but it is a real posture choice with an owner: `roadmap.md` **§RM-012**, the route-input posture design pass
  (co-owned with §RM-001), which owes the answer shape (`400 {"error": …}` versus tolerate-and-forward) plus all
  three loops in one motion. `Reported: —`
- `PR1-R1-6` — **a restart with the same `DUMP_DIR`, session id and collapsed format appends a duplicate
  `turn 000` to the existing append-only artifact and breaks the cross-restart diff chain.** What happens: `dump.py:147-158`
  always builds `SessionState(turn=0, prev_text="")`; `_append_record` buffers the next turn as turn 0 and `_flush_state`
  appends it to the same `<format>.log`, so the artifact carries two `turn 000` records and a diff numbered from the
  second one. How narrow the trigger: it needs a process restart with the same dump dir and the same session (the
  collapsed format is opt-in); `VP-T002-collapsed-artifact` already anticipates a restart with a **different**
  `--dump-format` but says nothing about the same format. Cost when it fires: one permanently malformed session artifact;
  no request fails, nothing is silent on the common path. Proper fix is a design decision with four materially different
  outcomes — continue numbering by seeding `turn`/`prev_text` from the artifact (needs a production reader and a
  change-focused diff-replay path), rotate a distinctly identified run (breaks `format_is_per_artifact`), truncate
  (breaks append-only), or refuse the second run (breaks `G-002`). No settled authority ranks them. Roadmap handle:
  `RM-022`. `Ask: yes — whether the collapsed artifact continues across a restart, rotates per run, or requires a
  fresh dump dir decides what a reader sees and what code is built.` `Reported: —`
- `121-P2` — **the `[tool.ruff.lint]` block's own comment calls the F-family selection a "Minimal gate" while nothing in the declared dev environment runs it.** What happens: a reader of `pyproject.toml:25-28` believes an automated F gate exists, while `ruff` is absent from `[dependency-groups] dev` and `uv.lock` (`grep -c 'name = "ruff"'` → 0), `.venv/bin/ruff` does not exist, `uv run --frozen ruff --version` → `Failed to spawn: 'ruff'`, and no repository automation invokes it (no `.github/workflows`, no Makefile/justfile/`scripts/`, `addopts` carries only the coverage gate). How narrow the trigger: nothing reads the claim; the policy is real and runnable by the repo's recorded ad-hoc invocation (`uvx --offline ruff@0.16.10 check buster_ripper/ tests/ buster_ripper_legacy.py` → `All checks passed!`, `[119-01-yield](iterations/119-01-yield.md):18`), so the false byte is one word, not a severed path. Cost when it fires: a developer trusts a gate that never ran and ships an F finding; the settled record already calls this "hygiene, not behavior" (`[079-codex-review](iterations/079-codex-review.md):225`, `[080-assumption-decide](iterations/080-assumption-decide.md):133`). Proper remedy lives in the doc line `QB:lint/ruff-config-gate-wording` (reword the comment at the next touch); the enforcement decision lives with `RM-021` (the lint-policy pass: whether `ruff` joins `[dependency-groups] dev` and the verification entrypoint enforces `select = ["F"]`). Source: step-13 stability round, `runs/121-closing-check-codex.txt` P2; disposition `accepted` (INTENDED + inert wording), no task opens, `T024`'s accepted record repairs stand. `Reported: —`
- `PR1-R5-1` — **a failed append is retried with the whole batch, leaving a torn prefix and duplicated records in an append-only dump artifact.** What happens: `dump._flush_state` (`:300-307`) writes the joined pending batch with one `f.write` and clears the buffer only after it returns; `_flush_session` (`:288-294`) catches and retains the batch. A mid-write `OSError` leaves the prefix, and the next periodic or shutdown flush appends the whole batch after it (measured 7 bytes + `pending=1` → 483 bytes, head `turn 00turn 000 …`). `_append_index_line` (`:195-200`) has the same shape: a torn line, then the same turn is retried (the exception skips `state.turn += 1`) and the new summary is joined to it (measured `turn 000 turn 000 …` on one line). How narrow the trigger: it needs a write that fails partway (a full disk, a quota, or an I/O error); the collapsed path is opt-in, but the documented index append runs every turn. Cost when it fires: one permanently malformed session artifact — no request fails, and the records are duplicated and torn rather than lost. Proper fix lives in one shared append helper that records the pre-write size and truncates back to it on failure, used by both append sites — about eight lines plus two failure-path tests; small, but the round is late and no normal-use finding sends a repair into `dump.py` this round. Roadmap handle: `roadmap.md` §RM-023. `Ask: yes — whether to take the rollback now or leave it on the roadmap with the risk accepted.` `Reported: —`

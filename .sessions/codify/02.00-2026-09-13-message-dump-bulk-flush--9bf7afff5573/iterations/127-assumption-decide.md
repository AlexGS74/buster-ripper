# 127 · Assumption Decide — PR #1 review round 5 (three root findings, three family sweeps)

## Orchestrator Prompt (verbatim)

> You are a subagent spawned by another session.
> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
> Worktree root: /Users/alexhubitski/Projects/buster-ripper
> Mode: external-review routing (PR #1, codex round 5) with family sweep
> Assumptions directory: /Users/alexhubitski/Projects/buster-ripper/.sessions/codify/02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/
> Registry root: /Users/alexhubitski/Projects/buster-ripper/invariants/
>
> Three new P2 root findings on PR #1 at commit `713d3af` (posted 2026-10-19T19:45:14Z):
>
> 1. **dump.py:306 — roll back partial appends before retrying a batch.** If `_flush_state` writes a prefix then raises (ENOSPC mid-write), `state.pending` retains the whole batch; the next flush appends it again, leaving a truncated prefix followed by duplicate records and corrupting arrival order. Make the append recoverable (restore the file to its pre-write offset on failure) or commit each batch atomically before clearing/retrying.
>
> 2. **messages.py:113 — reduce `max_tokens` on Messages retry.** On an SGLang `/v1/messages` overflow the code sets `current_body["max_completion_tokens"]`, but the Anthropic body's field is `max_tokens`; SGLang's `AnthropicMessagesRequest` declares `max_tokens` and its conversion copies it unchanged, so attempts two and three keep the same generation budget and repeat the overflow. The non-streaming retry at line 283 has the same defect.
>
> 3. **messages.py:78 — preserved thinking uses a field SGLang drops.** With `--preserved-thinking` against SGLang, `_inject_preserved_thinking` adds `chat_template_kwargs` to the Anthropic `/v1/messages` body. SGLang v0.5.9's `AnthropicMessagesRequest` does not declare that field, and its conversion uses only declared fields, so Pydantic discards `enable_thinking`/`clear_thinking`; the advertised default has no effect. Translate the option into a supported field or apply it at a compatible endpoint/version rather than injecting an ignored extension.
>
> **Verified third-party facts for this routing pass (read from pinned upstream sources, not assumed):**
> - SGLang **v0.5.9** `python/sglang/srt/entrypoints/anthropic/protocol.py`: `AnthropicMessagesRequest` fields are `model`, `messages`, `max_tokens`, `metadata`, `stop_sequences`, `stream`, `system`, `temperature`, `tool_choice`, `tools`, `top_k`, `top_p`. No `chat_template_kwargs`, no `thinking`, no `max_completion_tokens`. Pydantic's default extra behavior drops unknown keys.
> - SGLang **v0.5.9** `python/sglang/srt/entrypoints/anthropic/serving.py` conversion sets `"max_tokens": anthropic_request.max_tokens` unchanged.
> - SGLang **v0.5.9** `python/sglang/srt/entrypoints/openai/protocol.py`: `ChatCompletionRequest` declares both `max_completion_tokens` (line 546, preferred) and `chat_template_kwargs` (line 593); `max_new_tokens = max_completion_tokens or max_tokens` (line 727).
> - **vLLM 0.26.1rc0** (the live upstream at `http://hyperipper.local:30000`, `owned_by: vllm`): its Anthropic `AnthropicMessagesRequest` declares `max_tokens`, `thinking` and `chat_template_kwargs`; its chat endpoint supports `max_completion_tokens`. So finding 3 is SGLang-specific: on vLLM the injection is honored.
> - `_inject_preserved_thinking` is pre-existing behavior added by `847ebbe`, and `FK-011` (Option A) keeps the default on and documented; the Fork explicitly says the injection itself is not in question, so any fix must preserve the default-on posture where the backend honors it.
>
> **Family sweep requirements** (the user's standing instruction: use each finding's type to find similar cases and close them together):
>
> - **Family A — non-atomic append followed by retry.** #1 is the collapsed writer. The second append site is `dump._append_index_line` (`index.txt`), which can also leave a torn line with no rollback; decide whether the same rollback applies there. The documented writer's `write_text` files truncate on retry, so they are not the same shape. Check `session.py` / any other durable append or state file for the same defect.
> - **Family B — retry control field not honored by the target endpoint schema.** #2 is the messages route. Sweep the chat route's retry (`max_completion_tokens` on the chat endpoint, which SGLang and vLLM both honor) and any other retry/control field written to a schema-bound body. Confirm the chat arm is correct and say so.
> - **Family C — advertised option injected into a body whose schema discards it.** #3 is messages on SGLang. Sweep the chat route's `chat_template_kwargs` (supported on both chat endpoints) and any other injected option (`thinking_budget`, eval options) for the same schema mismatch. The fix must keep `FK-011` Option A intact: the default stays on, and it must still work on backends whose Anthropic endpoint declares `chat_template_kwargs` (vLLM). For a confirmed SGLang backend, decide between (a) skipping the dead injection and documenting the limitation for `/v1/messages` on SGLang v0.5.9, or (b) translating to a supported mechanism — and note that v0.5.9's Anthropic schema declares no thinking field at all, so (b) may be impossible on that endpoint. If (a), the README claim must be corrected and the chat route remains the preserved-thinking path on SGLang.
>
> Apply the Reach Gate per finding, classify 1%/0.01% where warranted, and allocate the remediation as the fewest safe waves. Note `chat.py` is at 299/300 and must stay ≤300; `messages.py` is at 339/344; `dump.py` is at 319. Do not reopen accepted work without naming the record it contradicts.
>
> Then the loop continues: fix → local Yield → push → reply inline to every new root → request the next review. Report the wave structure and any 1%/0.01% classification with its reason.

## Review round

`Review round: 5 (late: yes — the hosted rounds are 1, 2, 3, 4 and 5; round 2 raised no root. From round 5 the count rule makes the round late, so an unusual-or-rare finding is 1%/0.01% even when its fix is cheap; a finding normal use reaches is fixed in every round)`.

Enumeration through the frozen `pr-review-comments` pattern (`--paginate --slurp`, reviewer `chatgpt-codex-connector[bot]`, operator-authored replies only): `total=39 root_findings=21 replied=18 unreplied=3`. The three unreplied roots are the only new work; all 18 earlier roots carry operator replies and their repairs are at `e26ab28`, `8d6bb0d`, `394fc9c`, `f37d475`, `e4a7a56`, `457ba8e`, `6a089d8`, `a5b3c0a`, `31fe304`, `e8459c2`, `713d3af`. No root is a repost.

| # | Comment id | Created | Path | Priority |
|---|---|---|---|---|
| 1 | 4318096152 | 2026-10-19T19:45:14Z | `buster_ripper/dump.py:306` | P2 |
| 2 | 4318096159 | 2026-10-19T19:45:14Z | `buster_ripper/routes/messages.py:113` | P2 |
| 3 | 4318096166 | 2026-10-19T19:45:14Z | `buster_ripper/routes/messages.py:78` | P2 |

The issue-channel summary at `6257894978` names the run `Completed` at `713d3af` with `has_unresolved_findings: true`; no new root lies outside the three above.

## Recheck against the bytes (`HEAD 713d3af`, clean production tree)

Every claim was re-measured on this tree with a stubbed upstream and a patched `Path.open`/`session.post` (no repository byte written). The measurements are the pass's own, not the reviewer's.

| # | Claim | Measured on this tree | Verdict |
|---|---|---|---|
| 1 | a partial append is retried without rollback | `dump.py:300-307` opens the artifact in `"a"` and makes one `f.write("\n\n".join(state.pending) + "\n")`; `state.pending.clear()` and `pending_bytes = 0` sit **after** the write, and `_flush_session` (`:288-294`) catches and retains the batch. Measured with a `Path.open` proxy whose `write` lands 7 bytes then raises `OSError(ENOSPC)`: after the failed flush **7 bytes, pending=1**; after the next `flush_all()` **483 bytes, pending=0**, head `'turn 00turn 000  19:58:46  FULL  1 msgs  0 tools  0.1 KB\\n--- turn 000 body ---…'` — the torn prefix is followed by the whole batch | holds |
| 1-A1 | the same shape at `_append_index_line` | `dump.py:195-200` opens `index.txt` in `"a"` and writes `summary + "\\n"` with no rollback; a failure propagates out of `dump_turn` **before** `state.turn += 1`, so the next request retries the same turn number. Measured with a 9-byte partial write then a retry of the same sid: `index.txt` = `'turn 000 turn 000  19:58:46  FULL  2 msgs  0 tools  0.2 KB\\n'` — the torn prefix and the retried summary on one line | holds |
| 2 | the messages retry writes a field the Anthropic schema drops | `messages.py:113` and `:283` assign `current_body["max_completion_tokens"] = new_max`; the log at `:115`/`:285` says the same. Measured through the real app with `get_backend_state() == "sglang"` and an overflow 400: streaming **3 attempts**, `max_tokens` **10 / 10 / 10** and `max_completion_tokens` **None / 256 / 256**; non-streaming the same, ending `400 {"error": "Context overflow after 2 retries: …"}`. The reduced value never reaches `max_tokens` | holds |
| 3 | the preserved-thinking pair goes out on a body whose schema drops it | `messages.py:46-47` calls `_inject_preserved_thinking(body)` before dispatch; `:73-80` writes `chat_template_kwargs`. Measured through the real app with `PRESERVED_THINKING=True`: the forwarded `/v1/messages` body carries `chat_template_kwargs={'enable_thinking': True, 'clear_thinking': False}`. SGLang v0.5.9's `AnthropicMessagesRequest` declares only the twelve fields the dispatch lists (fetched from the `v0.5.9` tag: `model`, `messages`, `max_tokens`, `metadata`, `stop_sequences`, `stream`, `system`, `temperature`, `tool_choice`, `tools`, `top_k`, `top_p`), and its conversion (`serving.py:193-197`) builds `request_data` from declared fields only and copies `max_tokens` unchanged. vLLM 0.26.1rc0 declares `chat_template_kwargs` (`protocol.py:146`) and forwards it (`serving.py:477`) | holds |

## Invariant cross-reference

- Sources scanned: `invariants/index.md`, `invariants/buster_ripper/dump/invariants.md`, `invariants/buster_ripper/server/invariants.md`, `invariants/buster_ripper/assumption-claims.md`, `invariants/generated-index.md`; no session `invariants.md` exists.
- Active/wip entries: **0**. Every registry entry is `Status: planned` (the eight `PI:*` entries across the two module files); the `OI:` provenance blocks are `preserve` overrides, not active invariants; the skill therefore classifies every finding **Neutral**; no permanent `Claim:`/`Composition:` record is cited as authority and none is classified as an aligned invariant.
- Findings checked: **3 roots + 2 sweep siblings = 5** — Neutral 5, Aligned 0, Contradictions 0, Ambiguous 0, Overrides matched 0.
- `PI:T002/dump_format/documented_format_is_retained` (planned, compatibility, protected) is not touched: finding 1 changes no writer, no record shape and no default; it adds a failure-path rollback around the same append. `PI:T002/dump_format/format_is_per_artifact` is not touched: no artifact path or format key moves.
- No invariant candidate is created by any closure. The two closure packets add a failure path to an existing append and a backend gate to an existing injection; no invariant statement ranks either shape.

## Findings and dispositions

| # | Comment id | Location | Reach | Category | Disposition | Owning record |
|---|---|---|---|---|---|---|
| 1 | 4318096152 | `dump.py:300-307` (+ sibling `:195-200`) | **`latent`** — the collapsed artifact is opt-in (`config.DUMP_FORMAT` defaults to `documented`; `_flush_state` returns on that path), and a mid-write failure is an environmental condition no guard prevents; the documented index append is on the default path, but the failure input is the same. Measured 7 bytes + pending=1 → 483 bytes with `turn 00turn 000 …`, and `turn 000 turn 000 …` in `index.txt` | **1%** — the trigger is unusual but plausible (ENOSPC/EIO/EDQUOT mid-write), the result persists in an append-only artifact (a torn prefix plus duplicated records, broken arrival order), the fix is small but the round is late, and no normal-use finding sends a repair into `dump.py` this round | **`accepted`** (1% record) | `roadmap.md` §RM-023; `handoff-context.md` §Accepted risks / §Elevated to user `PR1-R5-1`; no task, wave or Fork |
| 1-A1 | family sweep of 1 | `dump.py:195-200` | **`reached`** — the documented writer is the default format and `_append_index_line` runs every turn; measured `turn 000 turn 000 …` on one line after a 9-byte partial write and a retry of the same turn | same 1% finding as #1: the same failure shape, the same rollback helper, and the same late round. Recorded in `PR1-R5-1`; no task | `roadmap.md` §RM-023; `handoff-context.md` §Accepted risks / §Elevated to user `PR1-R5-1` |
| 2 | 4318096159 | `messages.py:111-118`, `:282-287` | **`reached`** — SGLang is a supported backend (`--backend-type` / auto-detect) and an overflow is the exact condition the retry exists for; measured 3 attempts through the real app | not 1%: normal use on SGLang, and the fix is one field name at two sites. The retry was introduced at Checkpoint 05 (`e96b6c9`, `feat(retry): add context overflow retry for SGLang backends`) and has never had a working control field on this route; `T019.001` only repaired the error-text probe (`error_text = result.text`), which stays byte-unchanged | **`absorbed`** | lite brief `2026-10-19-pr1-r5-messages`; ledger `DC-PR1-R5-1-r1` |
| 3 | 4318096166 | `messages.py:46-51`, `:73-80` | **`reached`** — `PRESERVED_THINKING` is ON by default in the shipped launcher (`serve_buster_ripper.sh:30` `:-1`, `:42` appends `--preserved-thinking`); every `/v1/messages` request on an SGLang backend reaches the dead injection. Measured the forwarded body carrying the pair | not 1%: normal use on a supported backend with the default-on flag; the fix is a backend guard plus the advertised doc lines. `FK-011` Option A is preserved — the default stays on and the pair still goes out on vLLM/unknown | **`absorbed`** | lite brief `2026-10-19-pr1-r5-messages`; ledger `DC-PR1-R5-2-r1` |
| C | family sweep of 3 | `config.py:45-49`, `main.py:107-110`, `README.md:94-104` | **`reached`** — the config comment is read by developers, `--help` is user-facing and the README is the documented run path | same class as #3 in the inert-wording form: all three surfaces claim the pair without the SGLang limitation; the config comment is inverted about the endpoint that declares the field. The README is `T023`'s accepted deliverable and is **corrected, not reverted** (the default-on posture and `FK-011` stand). Doc-only correction in the same brief | **`absorbed`** | lite brief `2026-10-19-pr1-r5-messages`; ledger `DC-PR1-R5-3-r1` |

### Why #1 is the 1% category and #2/#3/C are not

- **#1** is real, its result persists, and its fix is small — but the round is late. The late-round rule reads an unusual-or-rare finding's Fix as `not cheap`, and the finding is unusual: a mid-write `ENOSPC`/`EIO`/`EDQUOT` is an environmental condition, not a normal-use input. The result is a permanently malformed append-only artifact (the torn prefix plus duplicated records cannot be undone by a rerun; the session is over and the artifact is the record). No normal-use finding sends a repair into `dump.py` this round, so it opens no brief; the row is `RM-023` and the `Ask: yes` line lets the user take the small rollback now. This is the same posture as `PR1-R1-6` (`RM-022`), whose trigger is also unusual-but-plausible and whose artifact also persists.
- **#2** is not 1%: the path is normal use on a supported backend (an overflow is why the retry exists), and the fix is one field name at two sites with no new mechanism. The late rule does not re-rate a normal-use finding.
- **#3** is not 1%: the flag is ON by default in the shipped launcher, so every SGLang `/v1/messages` request reaches the dead injection; the fix is one backend guard plus doc lines. The late rule does not re-rate a normal-use finding.
- **C** is not 1%: it is the same false advertisement as #3 and is corrected in the same brief.

### Why #1 is `accepted` and not `question`

Settled authority ranks the repair mechanism (a rollback around the existing append), so the closure space is deterministic. What the late rule changes is **whether** the repair is taken this round, not **which** repair is correct: it records the finding once and leaves the small, uniquely ranked rollback to the user's `Ask: yes` choice. No Fork is created; `RM-023` carries the decision and the trigger.

## Family sweep — the three types, every candidate audited

### Family A — non-atomic append followed by retry

| Site | Append shape | Failure shape | Verdict |
|---|---|---|---|
| `dump._flush_state:300-307` | one `f.write` of the joined batch, `"a"` mode; clear after the write | a partial write leaves a prefix; the batch is retained by `_flush_session` and re-appended whole | **defect — finding #1, 1% `RM-023`** |
| `dump._append_index_line:195-200` | one `f.write` of `summary + "\n"`, `"a"` mode | a partial write leaves a torn line; the exception skips `state.turn += 1`, so the next request retries the same turn and appends the new summary at the torn offset | **defect — A1, same 1% `RM-023`** (the same rollback applies; both sites share one helper) |
| `dump._write_full_turn:174-179`, `_write_diff_turn:186-191` | `Path.write_text` (`"w"` mode, truncating) | a retry rewrites the whole file from offset 0; a failed turn leaves no state advance and the next turn overwrites the same turn file | safe — truncating write, not an append; no prefix can survive a retry |
| `session.db_upsert:59-71` | SQLite `INSERT … ON CONFLICT … DO UPDATE` + `commit` | a failed commit leaves the transaction open; the next upsert commits the same row's latest values | safe — transactional and idempotent per row; no append list, no torn prefix |
| `chat._relay_chat_stream:202`, `messages._relay_after_commit:224` | `await response.write(chunk)` | a failed chunk write raises to the post-commit arm; the same chunk is not retried | safe — network stream, not a durable append; the post-commit contract already owns the arm |
| `image.py`, `compaction.py`, `backend.py`, `count_tokens.py`, `passthrough.py` | no durable append | n/a | safe |

**Family A closes with two sites in one 1% record.** Both are the same defect class (a partial append followed by a retry of the same record), the same rollback shape fixes both, and the round is late, so neither opens a brief. The documented `write_text` sites are not the same shape: they truncate on retry.

### Family B — retry control field not honored by the target endpoint schema

| Retry site | Field written | Endpoint schema | Verdict |
|---|---|---|---|
| `messages.handle_streaming:113` | `max_completion_tokens` | SGLang Anthropic `AnthropicMessagesRequest` declares `max_tokens` only; `serving.py:197` copies it unchanged; Pydantic drops the unknown key | **defect — finding #2, fixed** |
| `messages.handle_non_streaming:283` | `max_completion_tokens` | same | **defect — finding #2, fixed at the same time** |
| `chat.handle_chat:64` | `max_completion_tokens` | SGLang chat `ChatCompletionRequest` declares it (`protocol.py:546`, preferred) and computes `max_new_tokens = max_completion_tokens or max_tokens` (`:727`); vLLM chat supports it | **safe — the correct field on the chat endpoint** |
| `chat._apply_request_transforms:88` | `max_tokens` | chat endpoint declares `max_tokens` | safe |
| `passthrough`, `count_tokens`, `compaction`, `backend` | no retry or control field | n/a | safe |

**Family B closes with two assignments.** The chat arm was audited and is correct: its `max_completion_tokens` is the preferred field on an OpenAI chat body, and SGLang's own `max_new_tokens` preference reads it first.

### Family C — advertised option injected into a body whose schema discards it

| Injection site | Field | Endpoint schema | Verdict |
|---|---|---|---|
| `messages._inject_preserved_thinking:73-80` (called at `:46-47`) | `chat_template_kwargs` (`enable_thinking`, `clear_thinking`) on `/v1/messages` | SGLang v0.5.9 Anthropic declares no such field and no thinking field at all; conversion uses declared fields only → dropped. vLLM 0.26.1rc0 declares it (`protocol.py:146`) and forwards it (`serving.py:477`) → honored | **defect — finding #3, fixed by skipping the dead injection on a confirmed SGLang backend** |
| `chat._apply_request_transforms:91-94` | `chat_template_kwargs` (`enable_thinking`, `thinking_budget`) on `/v1/chat/completions` | SGLang chat declares it (`protocol.py:593`); vLLM chat supports it | safe — supported on both chat endpoints |
| `chat._apply_request_transforms:85` | `pop("max_gen_toks")` | removal, not injection | safe |
| `chat._apply_request_transforms:88` | `max_tokens` on `/v1/chat/completions` | declared | safe |
| `config.py:45-49` | module comment: "SGLang only; vLLM ignores `clear_thinking` but the flag is harmless to send" | inverted about which endpoint declares the field; SGLang v0.5.9's Anthropic endpoint drops the whole object, vLLM's Anthropic endpoint declares it | **defect — C1, doc correction** |
| `main.py:107-110` | `--preserved-thinking` help: "Inject … into /v1/messages requests" | claims the effect with no backend caveat | **defect — C2, doc correction** |
| `README.md:94-104` | §Serve script options: "forwarded /v1/messages requests carry … in chat_template_kwargs" | claims the effect with no backend caveat; it is `T023`'s accepted deliverable | **defect — C3, doc correction (the default-on posture and `FK-011` stand)** |

**Family C closes with one code guard and three doc lines.** The chat route's `chat_template_kwargs` is the schema-supported path on SGLang and is untouched; the messages route's injection is gated on a confirmed SGLang backend, and the advertised surfaces are corrected.

## Reach Gate — citations

| # | Reach | Citation |
|---|---|---|
| 1 | `latent` | the collapsed artifact is opt-in (`config.DUMP_FORMAT` defaults to `documented`; `_flush_state` returns on the documented path) and a mid-write failure needs an environmental condition no guard prevents; the documented index append is on the default path but the failing input is the same environmental condition. Measured 7 bytes + pending=1 → 483 bytes with `turn 00turn 000 …`, and `turn 000 turn 000 …` in `index.txt` |
| 2 | `reached` | `--backend-type sglang` / `/v1/models` auto-detect is configuration-supplied input; the retry loop is the exact path for an SGLang overflow; measured 3 attempts with `max_tokens` unchanged |
| 3 | `reached` | `PRESERVED_THINKING` is ON by default in the shipped launcher (`serve_buster_ripper.sh:30,42`) and every `/v1/messages` request on an SGLang backend reaches `_inject_preserved_thinking`; measured the forwarded body carrying the pair |
| C | `reached` | the config comment is read by developers, `--help` is user-facing, and `README.md` §Serve script options is the documented run path; same class as #3 |

## Deterministic Closure Gate — packets

All packets are `CLOSURE: absorbed`, `Question required: no`. The required outcome, authority, rejected candidates, risk shape and deferral are recorded per finding in `handoff-context.md` §Deterministic Closure Ledger (`DC-PR1-R5-1-r1` … `-3-r1`).

- **#1** — required outcome: a failed append must leave the artifact at its pre-write bytes and the retained batch must be the only copy re-appended. Authority: the artifact's append-only arrival-order contract (`PI:T002/dump_format/format_is_per_artifact`, `VP-T002-collapsed-artifact`); the reviewer's own two remedies (restore the offset or commit atomically) both produce that outcome; `_flush_session`'s retained batch is the existing retry contract. Rejected: keep the batch and accept the duplicate (preserves the defect); truncate the artifact and re-dump the whole session (loses earlier records); drop the batch on failure (loses the turn silently). The closure is deterministic but the late-round rule routes it to the 1% record.
- **#2** — required outcome: an SGLang overflow retry reduces the generation budget the Anthropic endpoint reads. Authority: SGLang v0.5.9's declared `max_tokens` and unchanged conversion; the retry's own name and log (`reducing …`); the chat sibling's `max_completion_tokens` is the correct field *on its own endpoint*, not on this one. Rejected: write both fields (the extra field is dropped and the log becomes ambiguous); move the retry to the chat endpoint (changes the client's protocol); leave it (preserves the dead retry).
- **#3** — required outcome: the flag's advertised effect is either honored or truthfully documented as not honored on the target endpoint, with the default-on posture and the vLLM/unknown effect preserved. Authority: `FK-011` Option A ("keep the default on and document it"); the Reach Gate's "refuse before handling" for a class outside what the mechanism supports; the verified SGLang v0.5.9 field set (no thinking field exists to translate to). Rejected: keep injecting and document only (leaves the ignored extension the reviewer names); translate to a SGLang field (none exists on v0.5.9's Anthropic schema); rewrite to `/v1/chat/completions` (changes the response protocol the client parses); require a newer SGLang (the repo does not control the upstream and no verified version supports it).
- **C** — required outcome: the advertised claim matches the effect. Authority: same as #3. Rejected: leave the wording (preserves the false advertisement); delete the flag (breaks `FK-011` Option A and `T023`).

## Wave structure — fewest safe waves

The two absorbed roots and the three doc siblings form **one wave with one brief**. `messages.py` is at **339/344**, so the #3 guard (+3 raw lines) and the docstring swap stay under the cap; #2 is net-zero (two field names). `chat.py` is untouched at **299/300**; `dump.py` is untouched at **319** because finding 1 is the 1% record.

### Wave 1 — `2026-10-19-pr1-r5-messages` (`messages.py`, `config.py`, `main.py`, `README.md`)

- **Anchor**: lite brief `2026-10-19-pr1-r5-messages`.
- **Members**: finding **2** (4318096159, both retry arms); finding **3** (4318096166); family siblings **C1** (`config.py:45-49`), **C2** (`main.py:107-110`), **C3** (`README.md:94-104`).
- **Prerequisite**: none. One brief because all five edits are the same `messages.py` request-body concern; editing the same file is not interference.
- **Owning route**: `lite-forward` → `lite-yield`.
- **Capacity shape**: `handle_messages` (`L:28 N:2 C:11`) is a hard C violation, so the log and call swap **replace two statements and add none**; `_inject_preserved_thinking` is Green and gains the guard (+3 raw lines). `_stream_single_request` / `_non_stream_single_request` are hard Length violations, so #2 changes one field name per site and adds no statement. `messages.py` ≤ **344** and does not cross 500.
- **Done when**: (a) on a mocked `"sglang"` backend with `PRESERVED_THINKING` on, a `/v1/messages` body is forwarded **without** an injected `chat_template_kwargs` (a client-sent one is forwarded unchanged), and on `"vllm"`/`"unknown"` the pair is still injected exactly as today; (b) an SGLang streaming overflow reissues attempts 2 and 3 with `max_tokens` reduced (and no `max_completion_tokens` added), and the non-streaming arm does the same; (c) the chat retry still carries `max_completion_tokens` and the chat `chat_template_kwargs` still carries `enable_thinking`/`thinking_budget`; (d) the README, CLI help and config comment state the SGLang v0.5.9 `/v1/messages` limitation and the vLLM Anthropic support; each test red before the fix.
- **Family sweep**: every durable append audited in §Family A; every retry control field audited in §Family B; every injected option audited in §Family C.
- **Invariant cross-reference**: Neutral — zero active/wip registry invariants; `PI:T002/dump_format/*` and `PI:T006/upstream_tls/*` are `planned` and untouched; no invariant candidate.
- **Closure ledger**: `DC-PR1-R5-1-r1` (#2), `DC-PR1-R5-2-r1` (#3), `DC-PR1-R5-3-r1` (C1–C3).

## Consumed records named, not reopened

- `T019.001` — its accepted error-text probe (`error_text = result.text`) is byte-unchanged; finding 2 changes only the retry's control field, which predates `T019.001` at Checkpoint 05 (`e96b6c9`). No done-when clause moves.
- `T023` — its accepted README section is **corrected** (a limitation sentence and a backend note added); the default-on posture, `PRESERVED_THINKING=1` and `FK-011` Option A stand. The record it touches is `T023`'s own deliverable; the change is additive, not a reversal.
- `FK-011` Option A — the default stays on and the pair still goes out on vLLM/unknown; the SGLang skip is the finding's own remedy, not a reopening of the default.
- `CL:buster_ripper.dump#_flush_state/conditional-batch-append` — its "one `f.write`" statement stays true under the rollback; its `Invalidates-On` fires on any `_flush_state` change, so its fingerprint will be stale for the promotion lane. Named, not reopened.
- `CL:buster_ripper.dump#_flush_state/pending-clear-after-append` — "an exception before that point skips both resets" stays true; the rollback is before the clear.
- `CL:buster_ripper.dump#_append_index_line/index-line-append` — the one-line append stays; the rollback is around it.
- `RM-012` — untouched: the `/v1/messages` root and element shapes stay with it; finding 3 is a request-body field, not a normalization shape.
- `RM-011` — untouched: the content-coding posture is not this finding.

## Handoff

- Iteration report written; `handoff-context.md` §Current Status, §Planned Task Context, §Deterministic Closure Ledger, §Accepted risks and §Elevated to user updated; `roadmap.md` §RM-023 added; `.sessions/codify/roadmap/2026-10-18-message-dump-bulk-flush.md` `RM-023` added; `quality-backlog.md` round-5 registry-staleness lines added.
- No production byte, test byte, plan task, invariant or decision record is written by this pass. The absorbed fixes run in the repair lane as one lite brief; the orchestrator owns the wave checkpoint.
- Reply plan: one inline reply per new root (4318096152, 4318096159, 4318096166) with its fixing commit or its 1% record, plus the `@codex review` comment listing the accepted `PR1-R5-1` row and its `RM-023` handle so it is not re-raised.

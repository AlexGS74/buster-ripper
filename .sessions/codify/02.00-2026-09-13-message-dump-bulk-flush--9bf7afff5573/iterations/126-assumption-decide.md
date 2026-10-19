# 126 · Assumption Decide — PR #1 review round 4 (six root findings, five family sweeps)

## Orchestrator Prompt (verbatim)

> You are a subagent spawned by another session.
> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
> Worktree root: /Users/alexhubitski/Projects/buster-ripper
> Mode: external-review routing (PR #1, codex round 4) with family sweep
> Assumptions directory: /Users/alexhubitski/Projects/buster-ripper/.sessions/codify/02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/
> Registry root: /Users/alexhubitski/Projects/buster-ripper/invariants/
>
> Six new P2 root findings on PR #1 at commit `6a089d8` (posted 2026-10-19T18:26:04Z):
>
> 1. **chat.py:35 — non-object chat JSON → 500.** Valid JSON whose root is `null`, an array or a scalar parses, then `dict(body)` (SGLang path) or `body.get("stream")` raises uncaught. Validate the decoded body is a mapping and return a client error.
> 2. **messages.py:174 — streaming message errors drop upstream headers.** A streaming `/v1/messages` upstream error keeps the status but discards every end-to-end header and forces `application/json`, so `Retry-After`, rate-limit metadata, request IDs and a non-JSON representation do not reach the caller. Relay the upstream headers and body on this pre-commit path, as the redirect, non-streaming and chat error paths now do.
> 3. **chat.py:138 — 300 s total deadline stays active for streamed chat.** A healthy SSE generation longer than five minutes is terminated; the new post-commit catch returns the committed response, so truncation looks like a normal EOF. Use an unbounded relay timeout as the streaming messages path does.
> 4. **chat.py:84 — non-mapping eval `chat_template_kwargs`.** With eval mode and an injecting profile such as `glm47`, a request carrying `chat_template_kwargs: null`, a list or a scalar makes `setdefault` return that non-mapping value and the nested `setdefault` raises; the broad handler silently forwards without `enable_thinking` or the budget, potentially after removing `max_gen_toks`. Normalize non-mapping values before injection.
> 5. **image.py:198 — image cache expiry only on image-bearing requests.** If traffic turns text-only, cached base64 payloads remain in memory indefinitely past `IMAGE_CACHE_TTL`; session keys have no global bound. Invoke expiry from regular request processing or a periodic task.
> 6. **chat.py:143 — always re-serializes the request body.** Even with no transform, `json=body` re-encodes; UTF-16 JSON becomes UTF-8 while `Content-Type` still says `charset=utf-16`. Keep the raw request body on the unchanged path; reserialize and reconcile representation headers only when a transform modifies it.
>
> The user's standing instruction: **on each pass, use the type of each finding to find more similar cases and close them all together.** Group the six into families and sweep each across the package:
>
> - **Family A — syntactically valid but wrong-shaped input reaches mapping/`setdefault` access and raises (or is silently swallowed).** #1 and #4 are two shapes of one class. Sweep every client-body parse site (`messages`, `chat`, `count_tokens`, `passthrough`) and every nested mapping access (`chat_template_kwargs`, `tools`, `messages`, `system`, `max_tokens`/`max_gen_toks`, eval options) for the same defect, including the `normalize()` element loops already tracked as `RM-012` — decide whether this wave closes them or whether `RM-012` remains the owner for the `/v1/messages` half.
> - **Family B — upstream error responses lose headers or body on one path.** #2 is the messages streaming sibling of the chat fix in `e26ab28`. Sweep every error path in all four routes for header/body fidelity.
> - **Family C — streaming relays keep an armed total deadline.** #3 is the chat sibling of the messages fix in `T018`. Sweep every streaming call for `ClientTimeout` composition.
> - **Family D — cache expiry triggered only by the path that writes the cache.** #5 is the image cache; sweep every cache (`image`, `compaction`, `dump._sessions`, `session._stats`) for expiry or growth that depends on the same path that populates it.
> - **Family E — request bytes re-serialized when nothing changed.** #6 is the chat request sibling of the chat *response* fix in `e26ab28`; sweep the request direction of every route (`messages`, `chat`, `passthrough`) for unconditional re-serialization and representation-header reconciliation.
>
> Apply the Reach Gate per finding, classify 1%/0.01% where warranted, and allocate the remediation as the fewest safe waves. Note `chat.py` is at 300/300 and `messages.py` at 344; the router must say which fixes need headroom and in what order. Do not reopen accepted work without naming the record it contradicts.
>
> Then the loop continues: fix → local Yield → push → reply inline to every new root → request the next review. Report the wave structure and any 1%/0.01% classification with its reason.

## Review round

`Review round: 4 (late: no — hosted rounds are 1, 3 and 4; round 2 raised no root. Rounds 1 and 3 both carried normal-use repairs, and this round's six roots are cheap repairs on normal or unusual-but-plausible paths, so the from-round-5 / all-rare late rule has not fired)`. Late-round re-rating starts at round 5; nothing here is classified by that rule.

Enumeration through the frozen `pr-review-comments` pattern (`--paginate --slurp`, reviewer `chatgpt-codex-connector[bot]`, operator-authored replies only): `total=30 root_findings=18 replied=12 unreplied=6`. The six unreplied roots are the only new work; all 12 earlier roots carry operator replies and their repairs are at `e26ab28`, `8d6bb0d`, `394fc9c`, `f37d475`, `e4a7a56`, `457ba8e`, `6a089d8`. No root is a repost.

| # | Comment id | Created | Path | Priority |
|---|---|---|---|---|
| 1 | 4317334042 | 2026-10-19T18:26:04Z | `buster_ripper/routes/chat.py:35` | P2 |
| 2 | 4317334048 | 2026-10-19T18:26:04Z | `buster_ripper/routes/messages.py:174` | P2 |
| 3 | 4317334057 | 2026-10-19T18:26:04Z | `buster_ripper/routes/chat.py:138` | P2 |
| 4 | 4317334067 | 2026-10-19T18:26:04Z | `buster_ripper/routes/chat.py:84` | P2 |
| 5 | 4317334078 | 2026-10-19T18:26:04Z | `buster_ripper/image.py:198` | P2 |
| 6 | 4317334102 | 2026-10-19T18:26:04Z | `buster_ripper/routes/chat.py:143` | P2 |

## Recheck against the bytes (`HEAD 6a089d8`, clean production tree)

Every claim was re-measured on this tree with a stubbed upstream and a mocked `session.post` (no repository byte written). The measurements are the pass's own, not the reviewer's.

| # | Claim | Measured on this tree | Verdict |
|---|---|---|---|
| 1 | a non-object chat body 500s | `chat.py:29-32` accepts any JSON root; `chat.py:48` `dict(body)` raises `TypeError` on `null`/`[1]` on the SGLang path, `chat.py:147` `body.get("stream")` raises `AttributeError` on the other; no arm catches either (`:150` names `ClientError`/`TimeoutError`). Measured with a reachable stub upstream: `POST /v1/chat/completions` body `null` → **500**, body `[1]` → **500**, SGLang state `null` → **500** | holds |
| 2 | the streaming messages error arm loses headers and forces JSON | `messages.py:169-175` returns `web.Response(body=await _error_body(resp), status=resp.status, content_type="application/json")`; `_error_body` (`:196-207`) decodes and replaces undecodable bytes. Measured: upstream 429 with `Content-Type: text/plain; charset=utf-8`, `Retry-After: 7`, `X-Request-Id: abc`, body `slow down` → client gets 429, `Content-Type: application/json`, **no** `Retry-After`, **no** `X-Request-Id`, body `slow down` re-encoded UTF-8 | holds |
| 3 | the chat relay keeps the 300 s total armed | `chat.py:138` `timeout = aiohttp.ClientTimeout(total=300)`; `:141-148` holds the `async with session.post(...)` open across `_relay_chat_stream`; `messages.py:150-157` documents the same trap and disarms it with `total=None`; round 1's `chat.py:200-202` catches `Exception` and returns the committed response, so the client sees a normal EOF | holds |
| 4 | a non-mapping eval `chat_template_kwargs` silently loses the injection | `chat.py:83-84` `kwargs = body.setdefault("chat_template_kwargs", {})` then `kwargs.setdefault(...)`; `:93-94` swallows the `AttributeError`. Measured: EVAL_MODE, profile `glm47`, `EVAL_MAX_TOKENS=512`, body `{"max_gen_toks": 7, "chat_template_kwargs": null}` → forwarded body has `max_gen_toks` **removed**, `max_tokens` **512**, `chat_template_kwargs` still **null**, no `enable_thinking` | holds |
| 5 | the image cache only expires on an image-bearing request | `image.py:198` is the only production call to `cleanup_expired_images()`; it sits inside `strip_and_cache_images`, whose only production caller is gated by `has_images(body)` (`messages.py:42-44`). `get_cached_image` (`:203-223`) has **no production caller** (`grep` → tests only). Measured through the route with `--strip-images` on: an image turn caches 1 source; after `IMAGE_CACHE_TTL + 1`, a text-only `/v1/messages` request leaves **1** source cached | holds |
| 6 | the chat request body is always re-serialized | `chat.py:143` `json=body`; `chat.py:35` `_apply_request_transforms` is a no-op when `EVAL_MODE` is off (`:73-74`). Measured: a UTF-16 body with `Content-Type: application/json; charset=utf-16` reaches `session.post` as `json=<dict>`, `data=None`, and the forwarded body has **no UTF-16 byte** while the header still names `charset=utf-16`. The same measurement on `/v1/messages` (`messages.py:156`, `:308`) reproduces it | holds |

## Invariant cross-reference

- Sources scanned: `invariants/index.md`, `invariants/buster_ripper/dump/invariants.md`, `invariants/buster_ripper/server/invariants.md`, `invariants/buster_ripper/assumption-claims.md`, `invariants/generated-index.md`; no session `invariants.md` exists.
- Active/wip entries: **0**. Every registry entry is `Status: planned` (eight entries across the two module files); the skill therefore classifies every finding **Neutral**; no permanent `Claim:`/`Composition:` record is cited as authority and none is classified as an aligned invariant.
- Findings checked: **11** — Neutral 11, Aligned 0, Contradictions 0, Ambiguous 0, Overrides matched 0.
- `PI:T002/dump_format/documented_format_is_retained` (planned, compatibility, protected) is not touched: finding 5 changes no writer, no record shape and no default; it only sweeps a cache whose TTL is already advertised. `PI:T006/upstream_tls/optout_is_process_wide` is not touched: no finding adds a session or a connector.
- No invariant candidate is created by any closure. Finding 5's periodic sweep is a new background task, but no invariant statement ranks a task shape; `G-004`'s timer pattern is the existing convention, not a protected invariant.

## Findings and dispositions

| # | Comment id | Location | Reach | Category | Disposition | Owning record |
|---|---|---|---|---|---|---|
| 1 | 4317334042 | `routes/chat.py:29-35` | **`reached`** — `POST /v1/chat/completions` is an exposed route and the body is network input; measured 500 in both backend states | not 1%: the repo's own `{"error": "Invalid JSON"}` 400 is the uniquely ranked answer (it is the route's existing shape at `chat.py:32`), the fix is an entry gate, and base **forwarded** a non-object chat body raw (`buster_ripper_legacy.py:812-847`, the `except Exception: pass` leaves `body` as the raw bytes), so the current 500 is a migration regression | **`absorbed`** | lite brief `2026-10-19-pr1-r4-chat-relay`; ledger `DC-PR1-R4-1-r1` |
| 2 | 4317334048 | `routes/messages.py:169-175` | **`reached`** — an upstream 429 is ordinary rate limiting; measured through the streaming route | not 1%: `DC-PR1-R1-7-r1` already settled the identical rule for the chat streaming arm ("relay the upstream's read bytes under the shared `utils.response_headers`"); `DC-CR3-F-04-r1`'s rejected candidate explicitly assigned this arm's relabeling to "the §RM-005/relay-fidelity lane", which this round takes up; the fix is net-negative lines | **`absorbed`** | lite brief `2026-10-19-pr1-r4-messages-relay`; ledger `DC-PR1-R4-2-r1` |
| 3 | 4317334057 | `routes/chat.py:138` | **`reached`** — a >5 min SSE generation is normal for long reasoning and eval runs; measured code path | not 1%: `T018` step 2 settled the identical rule on the messages sibling ("a spelling that re-inherited the session's total would truncate a long stream in silence"); the fix is one operand, net 0 lines | **`absorbed`** | lite brief `2026-10-19-pr1-r4-chat-relay`; ledger `DC-PR1-R4-3-r1` |
| 4 | 4317334067 | `routes/chat.py:82-86` | **`reached`** — `EVAL_MODE` and `--eval-profile glm47` are configuration values and the field is client input; measured | not 1%: `messages.py:73-80` already implements the exact rule ("a client-sent non-mapping is replaced, not skipped"); the fix is one guarded assignment, and the broad handler currently also removes `max_gen_toks` before failing | **`absorbed`** | lite brief `2026-10-19-pr1-r4-chat-relay`; ledger `DC-PR1-R4-4-r1` |
| 5 | 4317334078 | `image.py:195-198` | **`reached`** — `--strip-images` is a configuration value and image-then-text traffic is normal use; measured 1 → 1 past the TTL | not 1%: `IMAGE_CACHE_TTL` advertises an absolute bound the code does not enforce off the write path, and the package already runs two background sweepers (`dump.poll_flush`, `compaction.poll_kv_cache`); the fix is a third instance of an existing pattern | **`absorbed`** | lite brief `2026-10-19-pr1-r4-image-expiry`; ledger `DC-PR1-R4-5-r1` |
| 6 | 4317334102 | `routes/chat.py:141-146` | **`reached`** — every chat request; measured UTF-16 → UTF-8 under a stale `charset=utf-16` | not 1%: base forwarded the client's raw bytes outside eval mode (`buster_ripper_legacy.py:812-847`), and `DC-PR1-R1-3-r1` already settled the response-direction rule ("relay the original bytes when no transform changed the body"); the fix reuses that rule | **`absorbed`** | lite brief `2026-10-19-pr1-r4-chat-relay`; ledger `DC-PR1-R4-6-r1` |
| A1 | family sweep | `routes/chat.py:177-184` | **`reached`** — an upstream error answer is network input | same class as #1 on the response side: `_raise_chat_overflow(None, 400)` / `usage: []` raises `AttributeError` and answers 500 where the upstream's answer should relay; measured 500 for `"Context overflow: 5000 > 4096"` and for `{"error": …, "usage": []}`. Fix is one guarded line | **`absorbed`** | lite brief `2026-10-19-pr1-r4-chat-relay`; ledger `DC-PR1-R4-A1-r1` |
| A2 | family sweep | `routes/messages.py:241-251` | **`reached`** — upstream SSE is network input | same class as #1 in the silent-swallow form: a `message_start` event whose `message` is `null` raises `AttributeError`, `_relay_after_commit` (`:233-238`) swallows it and ends the stream. Measured: 200 with an **empty body** after a first chunk that carried the rest of the turn. Fix is two guarded reads | **`absorbed`** | lite brief `2026-10-19-pr1-r4-messages-relay`; ledger `DC-PR1-R4-A2-r1` |
| A3 | family sweep | `routes/count_tokens.py:17-26` | **`reached`** — exposed route, measured 500 | same class as #1: `_read_body` returns the parsed root, `strip_billing_nonce(body)` (`:87`) raises `AttributeError` on `null`/`[1]`; the route's own 400 `{"error": "Invalid JSON"}` shape (`:23-26`) is the ranked answer. Fix is the same entry gate | **`absorbed`** | lite brief `2026-10-19-pr1-r4-messages-relay`; ledger `DC-PR1-R4-A3-r1` |
| E1 | family sweep | `routes/messages.py:156`, `:308` | **`reached`** — every messages request | same class as #6: measured UTF-16 → UTF-8 under a stale `charset=utf-16`; base also re-serialized here (`buster_ripper_legacy.py:645`), so the raw-bytes rule is the same rule, not a new posture. Fix shares #6's helper shape | **`absorbed`** | lite brief `2026-10-19-pr1-r4-messages-relay`; ledger `DC-PR1-R4-E1-r1` |
| MSG | family sweep | `routes/messages.py:29-33`, `normalize.py:88/89/90/130/131/205/209` | **`reached`** — exposed route; measured 500 on `{"tools":[null]}`, `{"messages":[null]}`, `{"messages":["x"]}` and on a non-object root | **not this wave's**: the root shape and the three element classes are one posture question — `RM-012` records three admissible answers (400 with `{"error": …}`, tolerate-and-forward, validate-and-drop) that no settled authority ranks; `G-003` holds `/v1/messages` behavior still; and `handle_messages` carries a hard **C:11** violation (`L:28 N:2 C:11`), so adding the guard triggers the capacity gate and requires `RM-001`'s extraction first. Splitting the root half out would answer part of `RM-012` inconsistently | **`accepted`** (accepted risk, owner `RM-012`) | `roadmap.md` §RM-012 (root shape added to its owed work); no task, wave or Fork |

### Why 1, 2, 3, 4, 5, 6 and the sweep siblings are not 1%

Each has a **cheap complete remediation**, so the "any other | any | cheap → fix it" row decides before the rare-and-not-cheap row can apply. Stated individually:

- **#1** is one entry gate at the parse site, and base **did not** 500 on this input — it forwarded the raw bytes — so this is a regression, not a rare-and-unfixable case.
- **#2** has an accepted sibling fix in `e26ab28` with the identical mechanism; the repair is net-negative in lines.
- **#3** is one `ClientTimeout` operand, and `T018` settled the identical shape.
- **#4** is one guarded assignment mirroring `_inject_preserved_thinking`, which is already in the same package.
- **#5** is a third instance of the package's own periodic-task pattern (`spawn_dump_flush`, `spawn_kv_cache_monitor`); it adds no new mechanism and no CLI surface.
- **#6** is base parity (`buster_ripper_legacy.py` forwarded raw bytes outside eval mode) and one helper plus one call-site operand.
- **A1** is one guarded line; **A2** is two guarded reads; **A3** is the same entry gate as #1; **E1** is the same helper as #6.

### No 1% and no 0.01% finding this round

No supplied finding meets the 1% row: each is real, reachable, and has a cheap complete remediation, so the "any other | any | cheap" row applies first. No finding meets the 0.01% row: none is cosmetic-only with a non-cheap fix. No new roadmap row is minted and `§Elevated to user` is unchanged.

## Family sweep — the five types, every candidate audited

### Family A — syntactically valid but wrong-shaped input reaches mapping access

Client-body parse sites, all four routes:

| Site | Parse | Mapping access | Today's outcome | Verdict |
|---|---|---|---|---|
| `routes/messages.py:29` | `request.json()` | `normalize(body)` → `body.get("system")`; `session_id(body)`; `body.get("stream")` | 500 on a non-object root; 500 on a non-mapping `tools`/`messages` element | **root and element shapes stay with `RM-012`** (see MSG row above). Root measured 500, element classes measured 500 |
| `routes/chat.py:29` | `request.json()` | `_apply_request_transforms` (swallowed); `dict(body)` (`:48`); `body.get("stream")` (`:147`) | 500 on `null`, `[1]`, `5` in both backend states | **defect — finding #1, fixed** |
| `routes/count_tokens.py:20` | `request.json()` | `strip_billing_nonce(body)` → `body.get("system")` (`:87`) | 500 on `null`/`[1]` | **defect — A3, fixed by the same gate** |
| `routes/passthrough.py:14` | `request.read()` | none — raw bytes relayed | unaffected | safe — no parse |

Nested mapping accesses:

| Site | Access | Guard today | Outcome | Verdict |
|---|---|---|---|---|
| `chat.py:83-84` | `chat_template_kwargs` → `setdefault` | none; broad `except Exception` at `:93-94` | silently forwards without `enable_thinking`/budget after removing `max_gen_toks` | **defect — finding #4, fixed** |
| `chat.py:80` | `max_tokens` via `setdefault` | body is a dict after #1's gate | n/a | safe |
| `chat.py:77` | `max_gen_toks` via `pop` | dict `pop` on a dict | n/a | safe |
| `chat.py:180-183` | upstream `resp_data.get("usage", {})` | none; `(ClientError, TimeoutError)` arm at `:150` | 500 on a non-object overflow answer, and on `usage: []` | **defect — A1, fixed** |
| `chat.py:210-217` | `choice.get`, `msg.get` | `except Exception: return False` | relays unfixed | safe — deliberate fallback |
| `messages.py:75-80` | `chat_template_kwargs` | `isinstance(kwargs, dict)` | replaced, not skipped | safe — the precedent #4 mirrors |
| `messages.py:248` | `evt.get("message", {}).get("usage", {})` | none; `_relay_after_commit` swallows `Exception` | stream ends after the first chunk — silent truncation | **defect — A2, fixed** |
| `messages.py:117` | `result.text` | probe over `_stream_single_request` answers | total today because `_error_body` replaced undecodable bytes; would raise `LookupError` on an unknown charset once the arm relays raw headers | **must stay total — pinned in #2's fix** (measured: `web.Response.text` raises `LookupError` on `charset=x-bogus`; `result.body.decode(errors="replace")` does not) |
| `normalize.py:88/89/90` | `t.get("name")` ×3 | none | 500 | **`RM-012`** — element shape |
| `normalize.py:130/131` | `msg.get("content")`, `"cache_control" in msg` | none | 500 | **`RM-012`** — element shape |
| `normalize.py:205/208-209` | `msg.get("role")`, `msg["content"] = …` | none | 500 (masked until `:130` is guarded) | **`RM-012`** — element shape |
| `normalize.py:32/107/115/168` | `block.get`, `"cache_control" in block` | `isinstance(block, dict)` | safe | safe |
| `dump.py:51/59/67/86-89` | `t.get`, `b.get`, `msg.get` | none; `handle_messages`'s `except Exception` at `:59-60` swallows | the turn is not dumped | safe — dump is best-effort by contract (`G-003` failure isolation) |
| `image.py:34/41/56` | `block.get`, `msg.get` | `isinstance` guards | safe | safe |
| `session.py:95-97` | `body.get("system")`, `b.get("text")` | root gate at the route + `isinstance` | safe | safe |
| `backend.py:85/89` | `data.get`, `models[0].get` | `except Exception` → retry/`unknown` | safe — T007/R3-2 design | safe |
| `utils.py:63` | `json.loads(...).get` | `except Exception` | safe | safe |

**Family A closes with four edits**: the chat entry gate (#1), the same gate in `count_tokens` (A3), the guarded `_raise_chat_overflow` (A1), and the guarded `_note_message_start` (A2). The `/v1/messages` root and element shapes stay with `RM-012` for the reasons in the MSG row.

### Family B — upstream error responses lose headers or body

| Route / arm | Status relayed | Body | Headers | Verdict |
|---|---|---|---|---|
| messages streaming `>= 400` (`:169-175`) | yes | `_error_body` — decoded and re-encoded UTF-8 | **`Content-Type` only** | **defect — finding #2, fixed** |
| messages streaming `3xx` (`:161-166`) | yes | `resp.read()` | `response_headers` | safe |
| messages non-stream (`:336-341`) | yes | `resp.read()` | `response_headers` | safe |
| chat streaming `>= 300` (`:189-190`) | yes | `resp.read()` via `_chat_answer` | `response_headers` | safe — round 1 (`e26ab28`) |
| chat non-stream (`:149`) | yes | `resp.read()` via `_chat_answer` | `response_headers` | safe |
| passthrough (`:39-48`) | yes | `resp.read()` | `response_headers` | safe |
| count_tokens | authored, no upstream | authored JSON | `application/json` | safe |
| all `ClientError`/`Timeout` arms | authored 502 | authored JSON | `application/json` | safe — the proxy's own answer |

**Family B closes with one arm.** The messages streaming error arm was the only pre-commit path that both decoded the upstream's body and replaced its headers. `_error_body`'s only caller is that arm, so the helper is deleted and the probe reads bytes with replacement (`result.body.decode(errors="replace")`), which is the one shape that stays total for an unknown charset.

### Family C — streaming relays keep an armed total deadline

| Call | Timeout | Streams? | Verdict |
|---|---|---|---|
| `messages._stream_single_request:155-158` | `ClientTimeout(total=None)` | yes | safe — `T018` step 2 |
| `chat._make_chat_request:138-146` | `ClientTimeout(total=300)` | yes and no | **defect — finding #3, fixed**; the operand becomes `total=None if body.get("stream") else 300` |
| `passthrough:27-38` | `ClientTimeout(total=60)` | no — `resp.read()` finishes the body | safe |
| `backend.detect_backend:63-65` | `ClientTimeout(total=5)` | no | safe |
| `compaction.poll_kv_cache:32-40` | `ClientTimeout(total=5)` | no | safe |
| shared session (`server.py:69-72`) | `total=600` | per-request overrides | safe — no request inherits it unarmed |

**Family C closes with one operand.** No other relay reads a stream under an armed total.

### Family D — cache expiry triggered only by the path that writes the cache

| Cache | Populated by | Expiry / growth | Verdict |
|---|---|---|---|
| `image._image_cache` | `strip_and_cache_images` (route path, gated on `has_images`) | `cleanup_expired_images()` only inside that writer; `get_cached_image` has no production caller | **defect — finding #5, fixed with a periodic sweep** |
| `compaction._kv_cache_usage` | `poll_kv_cache` background loop | scalar, refreshed every `KV_POLL_INTERVAL`; no per-key growth | safe |
| `dump._sessions` | `_lookup_or_create_state` per new session | no eviction; holds `prev_text` + at most one flush interval of `pending` per session | **not the same defect and not cheap**: no advertised bound exists, and evicting would restart turn numbering — `RM-022`'s open design. Backlog row minted (`QB:dump-sessions/no-eviction`) with the `RM-022` trigger |
| `session._stats` | `update_session_stats` per session | one small record per session, persisted to SQLite by design (`STATS_DB`) and reloaded on restart | safe — intentional lifetime state |
| `config.*` | startup assignment | no growth | safe |

**Family D closes with the image sweep.** `dump._sessions` is recorded, not repaired: its remedy is the same undecided restart-numbering question `RM-022` owns.

### Family E — request bytes re-serialized when nothing changed

| Route | Today | Raw path available? | Verdict |
|---|---|---|---|
| chat `:141-146` | `json=body` always | `request.read()` after `request.json()` returns the cached bytes | **defect — finding #6, fixed** |
| messages `:155-158`, `:305-311` | `json=body` always | same cached bytes; `normalize()` reports its own changes in `changes` | **defect — E1, fixed by the same rule** |
| passthrough `:30-38` | `data=body` — the client's bytes | n/a | safe |

Representation-header reconciliation is required on both fixed routes: `forward_headers` keeps the client's `Content-Type`, and aiohttp copies a payload's own header only when the name is absent (`client_reqrep.py`, `update_body_from_data`), so a re-serialization must set `Content-Type: application/json` explicitly. Measured on `HEAD`: UTF-16 body → forwarded body has no UTF-16 byte while the header still names `charset=utf-16`, on chat and messages.

**Family E closes with one helper shape used at both routes.** The chat and messages arms share the cause; the passthrough was already raw.

## Reach Gate — citations

| # | Reach | Citation |
|---|---|---|
| 1 | `reached` | `POST /v1/chat/completions` is an exposed route; the body is client-supplied; measured 500 with a reachable stub upstream in both backend states |
| 2 | `reached` | an upstream 429 is ordinary rate limiting on `POST /v1/messages {stream:true}`; measured header loss through the route |
| 3 | `reached` | `POST /v1/chat/completions {stream:true}` with a healthy upstream generation past 300 s; the relay holds `_relay_chat_stream` inside the `ClientTimeout(total=300)` scope |
| 4 | `reached` | `EVAL_MODE` is a configuration value and the field is client input (`main.py:51`, `config.py:53-56`); measured injection loss |
| 5 | `reached` | `--strip-images` is a configuration value (`main.py:74-75`); an image turn followed by text-only traffic is normal use; measured 1 → 1 |
| 6 | `reached` | every chat request; measured UTF-16 → UTF-8 on `HEAD` |
| A1 | `reached` | upstream error answers on `/v1/chat/completions`; measured 500 |
| A2 | `reached` | upstream SSE on `/v1/messages`; measured an empty relayed stream |
| A3 | `reached` | `POST /v1/messages/count_tokens` is an exposed route; measured 500 |
| E1 | `reached` | every `/v1/messages` request; measured UTF-16 → UTF-8 |
| MSG | `reached` | `/v1/messages` is exposed; measured 500 on all three element classes and on a non-object root |

## Deterministic Closure Gate — packets

All packets are `CLOSURE: absorbed`, `Question required: no`. The required outcome, authority, rejected candidates, risk shape and deferral are recorded per finding in `handoff-context.md` §Deterministic Closure Ledger (`DC-PR1-R4-1-r1` … `-6-r1`, `-A1-r1` … `-A3-r1`, `-E1-r1`).

- **#1** — required outcome: a non-object chat body must not crash the route. Authority: the repo's authored 400 envelope (`chat.py:32`, `messages.py:33`, `count_tokens.py:23-26`), the Reach Gate's "gate invalid input at the entry", base parity (base forwarded the raw body, so 500 is a regression). Rejected: tolerate-and-forward (the reviewer asks for a client error; base parity alone does not rank a silent forward above an authored refusal), leave-and-record (defect preserved).
- **#2** — required outcome: the streaming messages error arm relays the upstream's status, body and end-to-end headers. Authority: `DC-PR1-R1-7-r1` (the chat sibling's identical rule), `T008.001`'s media-type scope, `DC-CR3-F-04-r1`'s explicit deferral to "the §RM-005/relay-fidelity lane". Rejected: keep `application/json` (preserves the defect), hand-pick names (duplicates the shared filter).
- **#3** — required outcome: a streaming chat relay is not terminated by the proxy's own deadline. Authority: `T018` step 2's messages sibling and its recorded reasoning; base's 300 s bound applies to a non-streaming request, which still gets it. Rejected: two call sites (duplicate body), no timeout at all (loses the non-stream bound).
- **#4** — required outcome: an injecting profile still injects after a non-mapping client value. Authority: `messages.py:73-80`'s accepted "replaced, not skipped" shape. Rejected: leave-and-swallow (preserves the defect), reject the request with 400 (a new rejection no authority imposes).
- **#5** — required outcome: a cached image source cannot outlive `IMAGE_CACHE_TTL` regardless of later traffic. Authority: `config.IMAGE_CACHE_TTL`'s advertised bound, `cleanup_expired_images`'s own docstring, the package's two periodic-task precedents. Rejected: a per-request sweep only (does not cover traffic stopping entirely), a sweep in the logging middleware (couples logging to the image cache), leave-and-record (leaves an advertised bound broken for a sensitive payload).
- **#6 / E1** — required outcome: an unchanged request is forwarded byte-for-byte with its representation headers; a changed one is reserialized and relabelled. Authority: base parity (`buster_ripper_legacy.py` forwarded raw bytes outside eval mode), `DC-PR1-R1-3-r1`'s response-direction rule, `RM-011`'s note that the `json=` paths reserialize under a peer label. Rejected: only outside eval mode (incomplete), leave-and-record (defect preserved, base parity ranks raw).
- **A1** — required outcome: a non-object upstream error answer relays instead of 500ing. Authority: the route's own relay contract (`chat.py:172-174`), the `_chat_answer` non-JSON fall-through. Rejected: none; one guard.
- **A2** — required outcome: a wrong-shaped `message_start` does not end the stream. Authority: `_relay_after_commit`'s post-commit contract ("one head, one terminal") and `T013`. Rejected: none; two guarded reads.
- **A3** — required outcome: a non-object count_tokens body is a client error. Authority: the same as #1 plus the route's own authored 400. Rejected: none; the same gate.
- **MSG** — `CLOSURE: question` is not reached: settled authority ranks `RM-012` as the owner and the capacity gate independently blocks an additive edit to `handle_messages`. Disposition `accepted` per the scope rule.

## Wave structure — fewest safe waves

The six roots and the five sweep siblings form **two waves**: one capacity prerequisite, then three briefs in parallel. `chat.py` is at **300/300** raw lines; `messages.py` at **344** (301-500 band); `image.py` at **255** (Green); `count_tokens.py` at **106** (Green).

### Wave 1 — `chat.py` capacity headroom (prerequisite, no finding)

- **Anchor**: lite brief `2026-10-19-pr1-r4-chat-headroom`.
- **Members**: none. It is the capacity prerequisite for Wave 2's chat brief, not a finding repair.
- **Why it is first and why it is separate**: `chat.py` is at exactly 300 raw lines and Wave 2's four chat fixes add statements and raw lines. The quality gate's preventive capacity clause says "a touched file is over 300 raw lines, or the task's additions would take it over 300" triggers extraction, and a comment shrink is the gate's own first move ("shrink first, and split only if the gate still triggers"). Keeping it as its own brief and commit also satisfies "the preparatory extraction rides … as its own commit before the behavior commit". It is a net-negative, behavior-free edit: **shrink over-long comments/docstrings to their point** — no code statement, no branch, no test byte, no import moves.
- **Named shrink set** (measured on `HEAD`): `_prepare_headers`' 4-line docstring and 4-line comment (`:100-109`), the 3-line comment at `:113-115`, `_carries_token`'s 4-line docstring (`:126-129`), `_chat_answer`'s 6-line docstring (`:156-161`), `_relay_chat_stream`'s 16 lines of docstring and comments (`:188-203`), and the one-line docstrings at `:172-173`, `:205-206`, `:224-225`, `:237-238`, `:252-253`, `:266-267`, `:288-289`. Load-bearing explanations that cannot be shortened move to `README.md` or the knowledge catalog, not back into the file.
- **Done when**: `wc -l buster_ripper/routes/chat.py` ≤ **272**; `git diff -U0` names no executable line (comments/docstrings only); the existing chat suite is green.
- **Owning route**: `lite-forward` → `lite-yield`. No plan task, no closure ledger row (not a finding).

### Wave 2 — three briefs in parallel (after Wave 1)

#### Brief A — `2026-10-19-pr1-r4-chat-relay` (`chat.py`)

- **Members**: findings **1, 3, 4, 6**; family sibling **A1** (`_raise_chat_overflow`).
- **Prerequisite**: Wave 1 (capacity headroom). Runs after the headroom brief, before or beside Briefs B and C.
- **Done when**: (a) `POST /v1/chat/completions` with body `null`, `[1]` or `5` answers the route's own `400 {"error": "Invalid JSON"}` in both backend states, with a reachable stub upstream; (b) a streaming chat whose stub sends a 200 head and healthy chunks for more than 300 s is not terminated — the relay arms no total deadline while `stream` is true, and a non-streaming request still arms 300 s; (c) `EVAL_MODE` + `glm47` + `EVAL_MAX_TOKENS=512` + `chat_template_kwargs: null` forwards `chat_template_kwargs` as a dict carrying `enable_thinking` and the budget, and the same for `[]` and `"x"`; (d) a UTF-16 request body is forwarded byte-identical with `Content-Type` unchanged, while a transformed body is reserialized and its `Content-Type` reconciled to `application/json`; (e) an upstream 400 whose body matches the overflow pattern but is a JSON string, or carries `usage: []`, relays the upstream's 400 instead of a proxy 500; each test is red before the fix.
- **Capacity shape (must be in the brief)**: `handle_chat` (`L:29 N:2 C:6`) and `_apply_request_transforms` (`L:20 N:2 C:8`) already carry three gray metrics; adding statements triggers the capacity gate. Carry the parse, the root gate and the raw-byte selection in **one private helper** and the kwargs normalization in a **second small helper**, so neither touched function gains a statement. `_make_chat_request` is "near tolerance" on Nesting and Parameters — **do not add a sixth parameter**; carry the payload decision through a helper that reads `request.read()` (cached after `request.json()`), not through a new argument.
- **Final**: `chat.py` ≤ 300 raw lines. If the helpers overshoot Wave 1's headroom, the brief's first commit is a further comment shrink, not a line-limit deletion.
- **Owning route**: `lite-forward` → `lite-yield`. Ledger `DC-PR1-R4-1-r1`, `-3-r1`, `-4-r1`, `-6-r1`, `-A1-r1`.

#### Brief B — `2026-10-19-pr1-r4-messages-relay` (`messages.py`, `count_tokens.py`)

- **Members**: finding **2**; family sibling **E1** (`messages` request bytes); family sibling **A2** (`_note_message_start`); family sibling **A3** (`count_tokens` root).
- **Prerequisite**: none; independent of Brief A and C.
- **Done when**: (a) a streaming `/v1/messages` upstream 429 relays the upstream's status, body bytes and end-to-end headers — `Retry-After`, `X-Request-Id`, `Content-Type: text/plain; charset=utf-8` all reach the client, and the overflow probe still reads the error text without raising on an unknown charset (`result.body.decode(errors="replace")`); (b) a non-overflow error still relays once and unchanged, with the re-pointed expectations (upstream headers and bytes, not `application/json`); (c) a UTF-16 `/v1/messages` body is forwarded byte-identical with its `Content-Type` unchanged, and a normalized body is reserialized with `Content-Type: application/json`; (d) an upstream `message_start` whose `message` is `null` does not end the relay — the following chunk still reaches the client; (e) a non-object `count_tokens` body answers the route's 400; each test is red before the fix.
- **Capacity shape**: `_stream_single_request` (`L:35`) and `handle_streaming` (`L:25 N:3`) are hard violations but each edit is **net-negative or net-zero statements** (the error arm loses a statement, the probe swaps one expression); no statement is added. Delete `_error_body` (dead after the arm changes) and shrink its 11-line comment. `_note_message_start` is within limits; the guard adds statements there only. Carry the raw-byte selection in one helper used by both request call sites — `_stream_single_request` and `_non_stream_single_request` are already hard violations, so adding a parameter is not available.
- **Final**: `messages.py` ≤ 344 and does not cross 500. **Re-point** `tests/test_context_overflow.py::test_an_error_that_is_not_an_overflow_relays_once_and_unchanged` (its `Content-Type` and re-encoded-body assertions) and `test_a_stream_error_body_that_no_charset_can_decode_still_answers_itself` (`await response.text()` → `await response.read()`); add the unknown-charset probe witness.
- **Owning route**: `lite-forward` → `lite-yield`. Ledger `DC-PR1-R4-2-r1`, `-E1-r1`, `-A2-r1`, `-A3-r1`.

#### Brief C — `2026-10-19-pr1-r4-image-expiry` (`image.py`, `server.py`)

- **Members**: finding **5**.
- **Prerequisite**: none; independent of Briefs A and B.
- **Done when**: with `--strip-images` on, a session caches an image, `IMAGE_CACHE_TTL` passes with **no further request at all**, and the cached source is gone — driven by the periodic sweep, not by an image-bearing write; an image-bearing request still caches and a live source still resolves inside the TTL; `on_cleanup` cancels and awaits the sweep before closing the session; a non-positive `IMAGE_CACHE_TTL` does not spin; the test is red before the sweep is spawned.
- **Shape**: mirror `dump.poll_flush` + `spawn_dump_flush` + `_cancel_background_task` exactly (a third instance of the existing pattern); sweep interval `max(IMAGE_CACHE_TTL, 1)` so the TTL is the only clock. `image.py` is Green; `server.py` has room.
- **Owning route**: `lite-forward` → `lite-yield`. Ledger `DC-PR1-R4-5-r1`.

### Consumed records named, not reopened

- `DC-CR3-F-04-r1` — its rejected candidate **explicitly deferred this arm's relabeling to "the §RM-005/relay-fidelity lane"**; finding 2 takes up that lane, so nothing accepted is contradicted. The `_error_body` helper it built is deleted only because its sole caller stops using it (dead code), and its decode-totality requirement is preserved through `result.body.decode(errors="replace")`.
- `DC-PR1-R1-7-r1` — the chat streaming error rule is the same rule applied to the messages sibling; no chat byte moves.
- `DC-PR1-R1-3-r1` — the chat response rule is the same rule applied to the request direction; its rejected candidate ("Add a shared helper outside `chat.py`: the capacity gate forbids an arbitrary helper solely to move lines") still stands: the Wave 2 helpers hold a new concern, and the shrink is a separate brief.
- `T018` step 2 — the messages stream deadline rule is the same rule applied to chat.
- `RM-012` — remains the owner of the `/v1/messages` root and element shapes; the root case is added to its owed work, not answered here. The decision not to close it is the capacity gate on `handle_messages` (`C:11`) plus the unranked posture.
- `RM-020` — its response-direction fix landed in round 1 (`DC-PR1-R1-3-r1`); the request direction is a new row in the same lane, fixed here.
- `RM-011` — untouched: its content-coding posture remains a design decision; finding 6 is the representation/charset half, not the `Content-Encoding` half.

## Handoff

- Iteration report written; `handoff-context.md` §Current Status, §Planned Task Context, §Deterministic Closure Ledger and §Accepted risks updated; `roadmap.md` §RM-012 / §RM-020 and `quality-backlog.md` updated with the round-4 note.
- No production byte, test byte, plan task, invariant or decision record is written by this pass. The six fixes run in the repair lane as lite briefs; the orchestrator owns the wave checkpoint.
- Reply plan: one inline reply per new root (4317334042, 4317334048, 4317334057, 4317334067, 4317334078, 4317334102) with its fixing commit, plus the `@codex review` comment listing the accepted `MSG` row and its `RM-012` handle so it is not re-raised.

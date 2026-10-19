# 125 · Assumption Decide — PR #1 review round 3 (three root findings, three family sweeps)

## Orchestrator Prompt (verbatim)

> You are a subagent spawned by another session.
> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
> Worktree root: /Users/alexhubitski/Projects/buster-ripper
> Mode: external-review routing (PR #1, codex round 3) with family sweep
> Assumptions directory: /Users/alexhubitski/Projects/buster-ripper/.sessions/codify/02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/
> Registry root: /Users/alexhubitski/Projects/buster-ripper/invariants/
>
> Three new root findings on PR #1 at commit `f37d475` (posted 2026-10-19T17:22:08Z):
>
> 1. **P1 server.py:68 — cookie persistence in the shared proxy session.** When an upstream response includes `Set-Cookie`, the default `ClientSession` cookie jar stores it and automatically attaches it to later requests to that upstream, including requests from unrelated downstream clients that never supplied the cookie. The proxy shares this session process-wide and binds `0.0.0.0` by default, so an upstream auth or affinity cookie leaks state across callers. Fix: construct the session with `aiohttp.DummyCookieJar()` so only each request's forwarded `Cookie` header is sent.
>
> 2. **P2 backend.py:69 — backend detection caches `unknown` permanently.** If `/v1/models` is unavailable or times out during the single startup probe (proxy and upstream starting concurrently), the exception path sets the backend to `unknown` and caches it permanently; SGLang-only behavior (date stripping, context-overflow retries) stays disabled until restart.
>
> 3. **P2 dump.py:272 — collapsed diff control lines unterminated.** `splitlines(keepends=True)` supplies endings only for content lines while `lineterm=""` leaves `---`, `+++`, `@@` unterminated; the caller joins with no delimiter, producing `--- turn_000+++ turn_001@@ ...` rather than a valid unified diff, so standard diff readers and patch tools cannot consume the advertised artifact. This is the production-side sibling of the `S21` gate composition's N5 note ("malformed-artifact/published-contract gap") — the gate was taught to mirror the malformed shape; the artifact itself was never fixed.
>
> The user's standing instruction for this loop: **on each pass, use the type of each finding to find more similar cases and close them all together** — a family sweep, not point fixes. So for each of the three types, the remediation must search the codebase for siblings and close them in the same wave:
>
> - **Family A — shared-session cross-caller state.** Cookie jar is the named one; also inspect the shared `ClientSession` construction for any other per-caller state the session carries across requests (default headers, auth, proxy, `trust_env`, `auto_decompress`, connector state) and say which are caller-visible.
> - **Family B — negative/transient results cached as permanent.** `unknown` backend is the named one; sweep for other caches of a failure or of an absence (backend cache, compaction state, image cache, any `lru_cache`/module global set on an exception path) and decide which are genuinely transient.
> - **Family C — artifact/output shape that is malformed or not consumable by its advertised reader.** The collapsed diff is the named one; check the documented `turn_NNN.diff`, the collapsed artifact records and `index.txt` line shapes for the same defect (missing terminators, wrong join, non-parseable records), since the milestone's contract is "today's content and line format" and `G-006` promises a reader can use the directory without tooling.
>
> Apply the Reach Gate per finding, classify 1%/0.01% where warranted, and allocate the remediation as the fewest safe waves. Note `chat.py` is at 300/300; `server.py` and `backend.py` have room; `dump.py` is in the 301–500 band. Do not reopen accepted work without naming the record it contradicts.
>
> Then the loop continues: fix → local Yield → push → reply inline to every new root → request the next review. Report the wave structure and any finding you classify as 1%/0.01%, with the reason.

## Review round

`Review round: 3 (late: no — hosted rounds are 1, 2 and 3; round 1 carried normal-use repairs, round 2 raised no repair-bearing root, and this round's three roots are all cheap repairs on normal or unusual-but-plausible paths, so the from-round-5 / all-rare late rule has not fired)`.

Enumeration through the frozen `pr-review-comments` pattern (`--paginate --slurp`, reviewer `chatgpt-codex-connector[bot]`, operator-authored replies only): `total=21 root_findings=12 replied=9 unreplied=3`. The three unreplied roots are the only new work:

| # | Comment id | Created | Path | Priority |
|---|---|---|---|---|
| 1 | 4316745948 | 2026-10-19T17:22:08Z | `buster_ripper/server.py:66-68` | P1 |
| 2 | 4316745957 | 2026-10-19T17:22:08Z | `buster_ripper/backend.py:69` | P2 |
| 3 | 4316745969 | 2026-10-19T17:22:08Z | `buster_ripper/dump.py:272` | P2 |

The nine round-1 roots (4309720759, -65, -72, -75, -78, -79, -84, -87, -90) all carry operator replies and their repairs are at `e26ab28`, `8d6bb0d`, `394fc9c`, `f37d475`. No round-2 root exists. Every finding below is a new root; none is a repost.

## Recheck against the bytes (`HEAD f37d475`, clean tree)

| # | Claim | Measured on this tree | Verdict |
|---|---|---|---|
| 1 | the shared `ClientSession` keeps a live cookie jar and replays upstream cookies to later callers | `server.py:65` builds one `TCPConnector`; `:66-68` one `ClientSession(timeout=…, connector=…)` with no `cookie_jar`; installed aiohttp `ClientSession.__init__` defaults `cookie_jar=CookieJar()`; `_request` takes `all_cookies = self._cookie_jar.filter_cookies(url)` (`client.py`, verified in `.venv`), and `ClientRequest.update_cookies` merges those into any caller-sent `Cookie` header, deleting and rebuilding it; the same session is `app["session"]` for every route (`chat.py:38`, `messages.py:147/303`, `passthrough.py:26`) and for the probe and KV poll | holds |
| 2 | the exception path caches `unknown` permanently | `backend.py:53-60` one probe; `:62-67` sets `_backend_type = "unknown"` on any exception; `:69` sets `_backend_state_cached = True` unconditionally; `:49-50` short-circuits every later `detect_backend`; nothing else calls it (`server.py:71` is the only call), so no later probe exists; `get_backend_state()` (`:89-95`) keeps returning `unknown`; the three consumers gate on `== "sglang"` (`normalize.py:195-196`, `chat.py:40-43`, `messages.py:91-94`) | holds |
| 3 | the control lines are unterminated and the join glues them | `dump.py:267-268` `splitlines(keepends=True)`; `:272` `lineterm=""`; `:269-273` yields the list; `:186` `out.write_text("".join(diff_lines))`; `:219` `f"{summary}\n--- turn {turn:03d} diff ---\n{''.join(diff_lines)}"`; measured on this tree: `'--- turn_000+++ turn_001@@ -1,3 +1,3 @@ a\n-b\n+B\n c\n'` | holds |

## Invariant cross-reference

- Sources scanned: `invariants/index.md`, `invariants/buster_ripper/dump/invariants.md`, `invariants/buster_ripper/server/invariants.md`, `invariants/buster_ripper/assumption-claims.md`; no session `invariants.md` exists.
- Active/wip entries: **0**. Every registry entry is `Status: planned` (eight entries across the two module files). The skill therefore classifies every finding **Neutral**; no permanent `Claim:`/`Composition:` record is cited as authority.
- Findings checked: 3 — Neutral 3, Aligned 0, Contradictions 0, Ambiguous 0, Overrides matched 0.
- `PI:T002/dump_format/documented_format_is_retained` (planned, compatibility, protected) is the one entry the Family C repair comes near. It is used as **closure basis**, not as an aligned invariant: its `Statement` protects the documented writer's *selection* and *the value produced when nothing is configured*; its `Why` says the user gated **displacement**, not byte-identity of a malformed control-line join. The binding and the rejected strict reading are recorded in the `DC-PR1-R3-3-r1` basis below.
- `PI:T006/upstream_tls/optout_is_process_wide` (planned, structural) is not touched: the cookie operand adds no session and no connector.
- No invariant candidate is created by any of the three closures.

## Findings and dispositions

| # | Comment id | Location | Reach | Category | Disposition | Owning record |
|---|---|---|---|---|---|---|
| 1 | 4316745948 | `server.py:66-68` | **`latent`** — the deployed upstream (`hyperipper.local:30000`, vLLM/uvicorn) answers no `Set-Cookie`; an auth or affinity upstream, or a gateway in front, delivers one, and `--upstream` can name one. Exposed surface: the proxy binds `0.0.0.0` by default and serves many callers from one session | not 1%: one constructor operand; consequence is **security** (cross-caller credential/affinity leak) stated first; the fix is cheap and adds no mechanism. The cheap row of the table decides it even in the rare band | **`absorbed`** | lite brief `2026-10-19-pr1-r3-session-cookie`; ledger `DC-PR1-R3-1-r1` |
| 2 | 4316745957 | `backend.py:62-69` | **`latent`** — needs the startup probe to fail (proxy and upstream starting concurrently, or an upstream restart); the deployed launcher starts against a long-running server | not 1%: the cause is this session's own `backend.py`, the fix is a bounded retry coroutine on the existing background-task pattern, no new CLI surface. Consequence is a wrong result that persists (SGLang-only behavior off until restart) | **`absorbed`** | lite brief `2026-10-19-pr1-r3-backend-retry`; ledger `DC-PR1-R3-2-r1` |
| 3 | 4316745969 | `dump.py:265-273` | **`reached`** — the documented writer is the default and writes `turn_NNN.diff` on every turn ≥ 1; the `change-focused` writer writes its diff record on every turn ≥ 1 | not 1%: normal use, wrong result that persists (a permanently malformed artifact), one operand in one shared helper fixes both writers; the gate change strengthens rather than weakens | **`absorbed`** | lite brief `2026-10-19-pr1-r3-diff-control-lines`; ledger `DC-PR1-R3-3-r1` |

### Why 1 is not the 1% category

The finding is real and its consequence is **security** — a cookie stored from one caller's upstream answer is replayed to every later caller, and `ClientRequest.update_cookies` lets it *override* a same-named caller cookie. The 1% row needs *not cheap*: this fix is one constructor operand (`cookie_jar=aiohttp.DummyCookieJar()`) with no new mechanism, no new dependency, and no request-path change. The cheap row decides it (`any other | any | cheap → fix it`), and a high-consequence finding is stated first rather than used to change the band.

### Why 2 is not the 1% category

The trigger is unusual-but-plausible and the consequence is a wrong result that persists. The 1% row's third clause is *no obvious fix where it shows*: the cause lives in this session's own `backend.py`, and there **is** a small fix — the package already runs two background pollers through `spawn_*` / `_cancel_background_task`, so a retry task is the established shape rather than a new mechanism. The fix adds no CLI option (so `RM-005`'s surface does not move), no dependency, and no request-path latency. Cheap → fix.

### Why 3 is not the 1% category

Normal use reaches it in the default configuration (`documented` writes `turn_NNN.diff` on every turn ≥ 1), and the artifact is permanently malformed (a wrong result that persists). The fix is one operand in the single helper both writers share, and the test-side mirror is re-pointed with it. Cheap → fix.

### No 1% and no 0.01% finding this round

No supplied finding meets the 1% row: each has a cheap complete remediation, so the "any other | any | cheap" row applies before the rare-and-not-cheap row. No finding meets the 0.01% row. No new roadmap row is minted, and `§Elevated to user` is unchanged.

## Family sweep — the three types, every candidate audited

### Family A — shared-session cross-caller state

`ClientSession` construction: `server.py:66-68` — `timeout=aiohttp.ClientTimeout(total=600)`, `connector=TCPConnector(ssl=not SKIP_TLS_VERIFY)`. No `headers=`, no `auth=`, no `proxy=`/`proxy_auth=`, no `trust_env=`, no `skip_auto_headers=`, no `cookie_jar=`, no `raise_for_status=`, no `auto_decompress=` operand.

| Carrier | Default in this session | Caller-visible across requests? |
|---|---|---|
| `cookie_jar` | `CookieJar()` (implicit) | **Yes — the finding.** Stores every upstream `Set-Cookie` and replays it on later requests to that host, and `ClientRequest.update_cookies` merges it into (and overrides) the caller's own `Cookie` header. Base built a per-request `httpx.AsyncClient` (`buster_ripper_legacy.py:668/704/847/906`), so no cookie outlived a request; this is a migration regression, not base parity |
| `headers` | not set (aiohttp's per-request defaults only) | No — the session holds no name and every request supplies its own `headers=` from `forward_headers(request.headers)` |
| `auth` | not set | No — no session-level credential exists; the caller's `Authorization` travels as a forwarded header |
| `proxy`, `proxy_auth` | not set; `trust_env=False` (aiohttp default) | No — no ambient proxy is consulted; the only proxy operand would be session- or request-level and neither is set |
| `connector` | one `TCPConnector(ssl=…)`, `force_close=False` | No — the pool is per-host and carries no caller identity; `PI:T006/upstream_tls/optout_is_process_wide` deliberately makes the TLS posture process-wide |
| `timeout` | `total=600` | No — a shared deadline, not caller state; three sites override it per request (`passthrough.py:27` 60 s, `backend.py:55` 5 s, `messages.py:265` `total=None`) |
| `auto_decompress` | `True` (aiohttp default) | No — per-response decoding; the request-direction posture is `RM-011` (1%) and the response-direction sibling is `RM-008` (1%), both already recorded |
| `raise_for_status` | `False` (aiohttp default) | No — each site inspects `resp.status` itself |

Module-level carriers outside the session, audited and cleared: `image._image_cache` and `session._stats` are keyed by the derived `session_id` (same system prompt ⇒ same conversation, by design) and neither is read on the request path for a *different* caller's data (`image.get_cached_image` has no production reader); `dump._sessions` is keyed by the same id and is the dump feature's own contract; `compaction._kv_cache_usage` is a server-wide gauge by design. **Family A closes with the one cookie-jar operand; no other cross-caller carrier exists.**

### Family B — negative/transient results cached as permanent

| Carrier | Set where | Genuinely transient? | Verdict |
|---|---|---|---|
| `backend._backend_state_cached` + `_backend_type` | `backend.py:69` after the `except Exception` at `:62-67`; also `_classify_models`'s empty-list `unknown` at `:78-79` | **Yes** — `/v1/models` answers `data: []` while the model list is still loading, and the probe fails while the upstream is not listening. A successful non-empty answer is definitive (sglang, vllm, or an unrecognized non-empty `owned_by`) | **defect — fixed by this wave** |
| `compaction._kv_cache_usage` | only on a 200 with a matched metric (`compaction.py:40-52`); failure is swallowed (`:55-56`) | No — the last good value is retained and the next poll (10 s) re-attempts; `0.0` is the safe direction for the `>=` threshold | safe |
| `image._image_cache` | only on a positive `strip_and_cache_images` write (`image.py:124-134`); TTL + LRU | No — no failure is cached; an expired entry is simply absent | safe |
| `dump._sessions` | `_lookup_or_create_state` only after the guard passes; a failed `_flush_state` keeps `pending` for the next flush | No — no negative state; a refusal caches nothing, so the next turn re-checks | safe |
| `session._stats` / `_db_conn` | positive updates only; a DB failure logs a warning and keeps memory (`session.py:81-85`) | No — no failure is cached as a value | safe |
| `config.*` | startup assignment only; never on an exception path | n/a | safe |

No `functools.lru_cache` or `functools.cache` exists in the package (`grep` → 0). **Family B closes with the backend cache; its two entry paths (exception and empty list) are the same cache and the same closure.**

### Family C — artifact/output shape not consumable by its advertised reader

| Artifact | Advertised reader | Measured shape | Verdict |
|---|---|---|---|
| `turn_NNN.diff` (documented, default) | `README.md:165-166` "unified diff"; `G-006` "well-formed … without additional tooling" | `_write_diff_turn` (`dump.py:183-192`) writes `"".join(diff_lines)`; `_unified_diff`'s `lineterm=""` leaves `---`/`+++`/`@@` unterminated, so the file is `--- turn_000+++ turn_001@@ …` | **defect — same root as the finding, fixed by the same operand** |
| `change-focused` record diff | `README.md:198` "every later turn stores its unified diff" | `_change_focused_record` (`dump.py:212-219`) writes `''.join(diff_lines)` with the same unterminated control lines | **defect — the finding** |
| `self-contained` record | `README.md:199-200` full rendering | `_self_contained_record` (`:222-229`) writes `current_text`, which `_render_body` always terminates with `"\n"` (`:91`) | safe — no diff text |
| record separators (`_flush_state`) | `README.md:189-196` one artifact, records in arrival order | `"\n\n".join(state.pending) + "\n"` (`:306`); every record ends with `"\n"`, so each record starts on its own line | safe — the separator is unambiguous even though a new batch starts one newline after the previous batch |
| `index.txt` line shape | `README.md:170-176` one line per turn; `QAP-MS001` `README_LINE_PATTERN` | `_full_summary` / `_diff_summary` + `"\n"` (`:195-200`); both README clauses matched | safe |
| `turn_000.json` | `README.md:164` full normalized body | `json.dumps(body, indent=2)` (`:177`) | safe |
| `QAP-MS001` gate's expected side | mirrors the *public* shape (its own docstring) | `production_diff_text` (`tests/qa/test_qap_ms001.py:137-157`) reproduces `lineterm=""` and `""`-join, and the coarse check at `:277` is a substring test — `S21` N5 | **defect — the gate was taught to mirror the malformed shape; re-pointed and strengthened in the same brief** |
| record-header collision with user content | `S21` N4 | a user line matching the record-header pattern (a line starting with `--- turn `, or a `turn NNN` word) can truncate `harness.record_section`'s extraction | **not the named class**: the artifact's own boundary is the summary line (`turn NNN  HH:MM:SS …`), which user content cannot impersonate without the timestamp grammar; the truncation is the gate extractor's limitation (`S21` N4), recorded, not repaired here |

**Family C closes with the one `lineterm` operand plus the gate re-point.** Both production writers and all three test-side mirrors share the single helper's shape.

## Reach Gate — citations

| # | Reach | Citation |
|---|---|---|
| 1 | `latent` | the deployed upstream (`hyperipper.local:30000`, vLLM/uvicorn) answers no `Set-Cookie` on `/v1/messages`, `/v1/chat/completions`, `/v1/models` or `/metrics`; a gateway or auth-enabled upstream would, and `--upstream` is an exposed configuration value; the proxy binds `0.0.0.0` (`main.py:68`) and shares one session across all callers |
| 2 | `latent` | `server.py:71` is the only `detect_backend` call; the probe fails only while the upstream is not answering `/v1/models` at startup (concurrent start, restart); the deployed launcher starts against a long-running server |
| 3 | `reached` | `dump.py:102-103` is the default writer, so every turn ≥ 1 writes `turn_NNN.diff`; `dump.py:104-105` writes the `change-focused` record for every turn ≥ 1 when that format is selected; `README.md:165-166` publishes both as unified diffs |

## Deterministic Closure Gate — packets

### 1 · Shared-session cookie jar

```text
CLOSURE: absorbed
Finding: PR1-R3-1 (comment 4316745948, server.py:66-68)
Required outcome: an upstream `Set-Cookie` reaches the caller that earned it and never a later, unrelated caller; each request sends only the `Cookie` header its own caller forwarded.
Authority: base parity (base built a per-request `httpx.AsyncClient` at `buster_ripper_legacy.py:668/704/847/906`, so no cookie outlived a request); the relay contract (`utils.forward_headers` already forwards `Cookie`, `utils.response_headers` already relays `Set-Cookie` downstream); `G-003`; the user's round-3 instruction; the installed aiohttp source shows the jar both stores and overrides.
Admissible candidates: (a) construct the session with `cookie_jar=aiohttp.DummyCookieJar()`; (b) build a fresh `ClientSession` per request; (c) keep the jar and clear it between requests.
Rejected candidates: (b) -> abandons the one-connector posture `PI:T006/upstream_tls/optout_is_process_wide` and the whole point of T006; (c) -> a timer/middleware, a new mechanism that races concurrent requests; (a) is one operand and preserves both.
Selected closure: (a), with the request's forwarded `Cookie` header and the response's relayed `Set-Cookie` byte-unchanged.
Risk shape: unchanged — one constructor operand; the connector, the TLS posture, every route and every relay are byte-unchanged.
Deferred additive work: none.
Owning route: lite brief `2026-10-19-pr1-r3-session-cookie`.
Question required: no
```

### 2 · Backend detection retry

```text
CLOSURE: absorbed
Finding: PR1-R3-2 (comment 4316745957, backend.py:62-69)
Required outcome: a transient `/v1/models` failure at startup cannot leave the process in `unknown` for its lifetime; once the upstream answers, the backend is classified without a restart, and a definitive answer is still cached once.
Authority: T007's done-when (`bound the probe`, `ordinary/timeout fallback unknown`) fixes the *fallback value* and the 5 s bound, not the permanence; the existing `spawn_kv_cache_monitor`/`spawn_dump_flush` + `_cancel_background_task` lifecycle is the package's own background-poll pattern (`server.py:25-45`, `:86-106`); `compaction.poll_kv_cache`'s "upstream may be starting up; ignore silently" is the same problem shape; `RM-005`'s no-new-CLI-surface posture; `G-002`'s clean shutdown; the user's round-3 instruction.
Admissible candidates: (a) a background re-probe task, cancelled and awaited in `on_cleanup`, that re-probes until a definitive answer; (b) leave `_backend_state_cached` False and add a lazy async re-probe at the three `get_backend_state()` consumers; (c) bounded retries inside `detect_backend`; (d) treat `unknown` as `sglang`; (e) keep the permanent cache and record it.
Rejected candidates: (b) -> adds an await to the request path at three sites and cannot fix the sync `normalize._normalize_date` read; (c) -> still gives up permanently for a longer outage, so it does not meet the required outcome; (d) -> wrong for vLLM (date stripping and overflow retry would fire on a non-SGLang backend); (e) -> leaves the verified defect in place, contrary to the user's instruction.
Selected closure: (a); the retry stops on a definitive answer (`sglang`, `vllm`, or a successful non-empty model list), never caches the empty-list or exception result, and reuses the existing spawn/cancel helpers.
Risk shape: unchanged — one private coroutine in `backend.py` plus one spawn and one cancel line in `server.py`; the probe, its 5 s bound, the fallback value and all three consumers are byte-unchanged; no new CLI surface.
Deferred additive work: none.
Owning route: lite brief `2026-10-19-pr1-r3-backend-retry`.
Question required: no
```

### 3 · Diff control-line terminators

```text
CLOSURE: absorbed
Finding: PR1-R3-3 (comment 4316745969, dump.py:265-273)
Required outcome: every diff the dump writes is a valid unified diff — the `---`, `+++` and `@@` control lines stand on their own terminated lines — and the QA gate rejects the glued shape instead of mirroring it.
Authority: `README.md:165-166` publishes `turn_NNN.diff` as a unified diff; `README.md:198` publishes the `change-focused` record as a unified diff; `G-006`'s failure column makes "the artifact is not well-formed" a goal failure and requires reading without tooling; `G-008`(a)'s "today's content and line format" is bound below; `S21` N5 records the gate's mirroring as the malformed-artifact/published-contract gap.
Admissible candidates: (a) `lineterm="\n"` in `_unified_diff` (one operand, both writers, both formats) plus re-point the three test-side mirrors and strengthen the gate's coarse check; (b) join with `"\n"` while leaving `lineterm=""`; (c) correct the README to stop saying "unified diff"; (d) leave the artifact and record it.
Rejected candidates: (b) -> content lines already carry their own endings, so the join double-terminates every content line and produces blank lines between them; (c) -> weakens the published contract and leaves `G-006`'s well-formedness failure in place; (d) -> leaves the artifact not consumable by its advertised reader and the fix is one operand.
Selected closure: (a).
Risk shape: unchanged — one operand in the single helper both writers and all mirrors share; the counts, the summaries, the record separators, the self-contained rendering and `turn_000.json` are byte-unchanged; `dump.py` stays net 0 lines in its 301–500 band.
Deferred additive work: `S21`'s recomposition is owed on the new bytes; the seam is already capped (`§Capped seams`), so no task opens from it; trigger = the next composition dispatch on these bytes.
Owning route: lite brief `2026-10-19-pr1-r3-diff-control-lines`.
Question required: no
```

### The `G-008`(a) binding, stated inspectably

Verbatim: *"With **no format configured**, N turns produce exactly `turn_000.json`, `turn_{N:03d}.diff`, `index.txt` with today's content and line format, at `2N` write-bearing operations."*

- I am reading `today's content and line format` as: the three files exist; `turn_000.json` is the full normalized body; `turn_NNN.diff` is the unified diff between consecutive renderings; `index.txt` carries the README's one-line-per-turn shape. I am **not** reading it as byte-identity with base's malformed control-line join, because `G-006`'s failure column makes a not-well-formed artifact a goal failure and the README publishes the file as a unified diff. A byte-strict reading would put two approved goals in conflict on the same bytes; the layout/reader reading is the only one compatible with both (the method 019 used for `US-012`).
- What a competent author could have intended: the join came from the port of `buster_ripper_legacy.py:595-603` verbatim; nothing in the plan chose it, and `S21` N5 recorded it as an artifact property rather than a decision. The base-parity argument therefore does not defend it, and `G-008`(a)'s protection is the *selection* of the documented writer (`PI:T002/dump_format/documented_format_is_retained`'s `Why`), which this fix leaves intact.
- Consequence: no user question. The finding does not contradict an *active* invariant (the registry holds 0 active/wip entries), and the planned entry's `Statement` is satisfied on the selection reading. `G-008`(a)'s witness (`TC-108`) reads the *README*, not the implementation (its own docstring says so), so the witness is re-pointed with the fix rather than contradicted by it.

## Remediation waves

**One wave, three briefs, no prerequisite and no capacity ordering.** No brief touches `chat.py` (300/300), so no extraction or ordering is owed. `server.py` (144) and `backend.py` (95) are green; `dump.py` (319) stays net 0 in its 301–500 band.

| Wave | Brief | Findings | Why this wave / order |
|---|---|---|---|
| 1 | `2026-10-19-pr1-r3-session-cookie` | 1 (P1) | One operand in `server.py`. Independent of the other two; first because it is the P1 and the smallest change |
| 1 | `2026-10-19-pr1-r3-backend-retry` | 2 | `backend.py` + two `server.py` lifecycle lines. Independent of the other two. Runs after the cookie brief so the two `server.py` edits never interleave in one lite run |
| 1 | `2026-10-19-pr1-r3-diff-control-lines` | 3 | `dump.py` + its three test-side mirrors. Independent of the other two; can run in any position |

Separation reasons: none from interference (three different causes in three different modules); the two `server.py` briefs are ordered inside the wave only so their commits are individually attributable. No `RM-001` prerequisite exists for any of them.

## Files that carry the durable record

- `handoff-context.md` §Deterministic Closure Ledger: `DC-PR1-R3-1-r1`, `DC-PR1-R3-2-r1`, `DC-PR1-R3-3-r1`.
- `handoff-context.md` §Planned Task Context: the one wave with its three briefs, member findings, ordering and separation reasons.
- `handoff-context.md` §Accepted risks: unchanged (no new accepted risk).
- `handoff-context.md` §Elevated to user: unchanged (no new 1%/0.01% item).
- Lite briefs under `00bab65:.sessions/lite/2026-10-19-pr1-r3-*/brief.md`.
- No `roadmap.md` or chunk-row change: every finding is repaired in this wave.

## Notes

- The two `server.py` briefs are **sequential inside the wave**, not concurrent: each lite run commits on `ACCEPT`, and the second reads the first's committed bytes. That is the only ordering the wave carries.
- Family A's audit is the part the reviewer could not see: `headers`, `auth`, `proxy`, `trust_env`, `auto_decompress`, `raise_for_status` and the connector are all caller-neutral in this session, and the cookie jar is the only per-caller carrier. Base parity makes it a migration regression, not a posture choice.
- Family B's second entry path (empty `data: []`) is covered by the same retry stop condition; no second finding or second closure is minted for it.
- Family C's documented `turn_NNN.diff` is the default writer and the same helper feeds it; the family sweep therefore repairs the default path, not only the reviewer's opt-in one.
- `S21`'s composition is stale on the new bytes after this wave. The seam is already capped (`§Capped seams`) and the single recomposition allowance is spent, so the repair records the owed reading as deferred work and opens nothing from it.
- No user question was emitted. Every finding had one uniquely ranked remediation.

```text
ASSUMPTIONS: 3 routed | 3 absorbed | 0 questions | 0 breaks | 0 malformed | 0 accepted | 0 capped
```

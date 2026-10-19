# 073-01 Forward — T015 + T016 CR2 wave (landing pass)

Session: `02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573` · Worktree: `/Users/alexhubitski/Projects/buster-ripper`
Tasks: `T015` (wave anchor) + `T016` (member 2), each on its own counter, both starting at 0 — no prior attempt of this wave reached Yield, so this is `{NNN}-01`, not a retry
Wave base: `1c84ebd` (Checkpoint 163) → capacity extraction `a4d1d26` (Checkpoint 164) → **this landing commit `c8df363` (Checkpoint 165)**, production + witnesses
Checkpoints this pass: **165** `c8df363` (production + tests) · **166** (this record, the plan/handoff/backlog updates, and the two evidence files)
Blueprint: `handoff-context.md` §Planned Task Context → `T015`, `T016`; ledgers `DC-CR2-F-1-r1`, `DC-CR2-F-2-r1`, `DC-CR2-F-3-r1`, `DC-CR2-F-4-r1`
Evidence: [cr2-fail-first-run.txt](../evidence/cr2-fail-first-run.txt) · [cr2-wave-surface-and-scan.txt](../evidence/cr2-wave-surface-and-scan.txt) (both authored by the implementing pass, committed here)

## Orchestrator Prompt (verbatim)

> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
> Task: T015 (CR2 wave anchor — members T015 and T016)
> Scope of THIS dispatch: land the increment another Forward pass already implemented but did not commit. Do not re-implement, restructure, or restyle any production or test content — the closure exists in the worktree on top of Checkpoint 164 (`a4d1d26`), whose own base is Checkpoint 163 (`1c84ebd`).
>
> Read the uncommitted diff, the `T015`/`T016` blueprint blocks and their `DC-CR2-F-1-r1`…`F-4-r1` ledger entries, and the two evidence files the implementing pass left: `evidence/cr2-fail-first-run.txt` (26 red / 100 green-to-red against both the wave base and the extraction commit, 126 passed after closure) and `evidence/cr2-wave-surface-and-scan.txt` (AST foreclosures against `1c84ebd`, file sizes, LNCP before/after). Then:
>
> 1. Confirm the tree independently: run the hermetic suite and the affected scope, and record the figures you observe rather than inheriting the claim of 439 @ 95.75%.
> 2. Write `iterations/073-01-forward.md` — the CLI-resolved id is taken. It must carry the verbatim orchestrator prompt, the increment brief with an `Inventions:` line naming the three module-private helpers (private helpers are not Anchor Map rows), a Failure-Path Evidence table, the `Capacity gate:` record (triggered → the behavior-preserving extraction at Checkpoint 164 → final metrics, with `chat.py` landing at exactly 300/300 and `messages.py` at 299), and a `Backlog pickup:` line that marks `QB:routes-messages/write-only-state-input-tokens` absorbed and records `QB:routes-chat/raw-bytes-passthrough-when-unfixed` as **skipped with its reason** (chat.py at the ceiling and `DC-CR2-F-1-r1` classed it non-competing). Include the aiohttp dependency note the implementing pass surfaced: `ClientResponse.json()` accepts `application/problem+json`, so the witnesses read bytes rather than trusting `.json()`.
> 3. Update `plan-summary.md`'s T015/T016 rows and Progress Overview, replace `handoff-context.md`'s `## Current Status` (not append), and mark the absorbed backlog line `[x]` with its iteration id and evidence.
> 4. Commit through `codify git create-checkpoint` in two steps — production plus tests first, then the session artifacts — never raw git.
>
> Three items to flag in the report for the single local Yield that follows, without ruling on them yourself: the test surface spans three files because extending `tests/test_routes_integration.py` would cross 500 lines (T014's three-file surface was accepted at 068-01-yield); `tests/qa/harness.py:135`'s `assert b"Cannot connect" not in body` can no longer fire against a proxy-local 502 now that the bodies are static, though its `200 <= status < 500` gate still holds, and that harness is outside this wave's touch-set; and `chat.py:55` plus `messages.py`'s 400 overflow bodies still embed `{e}`, which belongs to `CR2-X-1`'s retry lane rather than this wave.

*(The one line omitted above is the harness's own framing sentence, "You are a subagent spawned by another session."; everything from `Session ID:` onward is the dispatch as received.)*

**Mode note, recorded for the trail:** this is an ordinary Forward pass over an already-written increment, not a `phase-boundary-*` mode. The instruction "do not re-implement, restructure, or restyle" governs content; it does not remove the obligation to verify the tree and document the whole scope, so every figure below was re-measured here rather than inherited (§Independent confirmation).

## Increment Brief

T015 + T016 · the CR2 remediation wave, one local Yield: stop a proxied answer from
being rebuilt as the proxy's own, and stop a proxy error from saying the upstream back
to the client.

Anchors touched: none of the Anchor Map's rows. Its rows bind the dump and TLS clusters
(`plan-summary.md` §Anchor Map); `chat.py`/`messages.py`/`passthrough.py` response relay
is named by no row, and the wave reuses the existing `utils.response_headers` referent
rather than binding a new one. Reuse evidence: no `Mode: NEW` anchor was consumed or
created, so no `RB-*` replay was owed. **Will NOT create:** a second header filter, a
response-relay class/registry, a client-error type, a read path, or a `config`/`main`
surface; `FK-011` is untouched (no `decisions.md` byte moved).
`Inventions:` three module-private helpers — `_chat_answer` and `_chat_response` in
`buster_ripper/routes/chat.py`, `_inject_preserved_thinking` in
`buster_ripper/routes/messages.py`. Private helpers are not Anchor Map rows
(`plan-summary.md` §Anchor Map, closing sentence); they decompose existing concerns for
LNCP and authorize no new artifact. Two of the three were created by the capacity
extraction at `a4d1d26`, one by this closure; none is a new load-bearing referent.

## Failure-Path Evidence

Cases enumerated from the operations the two blueprints name, before judging any result.
Rows marked *excluded* name the authority that excludes them — an omitted row is not an
exclusion.

| Operation / condition | Expected outcome and authority | Handling / observation evidence | Disposition |
|---|---|---|---|
| `_make_chat_request` non-stream arm / upstream answers well-formed **non-JSON** (gateway HTML 502, plaintext 429) | relay upstream's status, its bytes verbatim, its filtered headers — never a synthetic 502 (`DC-CR2-F-2-r1`; base `buster_ripper_legacy.py:885-892`) | `_chat_answer` parses inside a `try` and falls through to `_chat_response`; `TestNonJsonRelay::test_upstream_status_bytes_and_headers_all_travel` × 5 params (502 HTML, 429 plaintext, truncated JSON, empty body, undecodable bytes), each asserting status + bytes + `Content-Type` + `Retry-After` on the wire. Red at `1c84ebd` and `a4d1d26` | safe |
| same arm / body parses to a **non-object** (`null`, `[1, 2]`, `"a string"`) | parse succeeds, so the transform path is taken; its `.get` raise is swallowed as it always was and the body is relayed — no invented body (same ledger's rejected candidate "swallow parse failure and re-serialize `{}`") | `test_json_that_is_not_an_object_relayed_verbatim` × 3, which additionally pins `X-Request-Id` surviving the swallow | safe |
| same arm / **empty body** | an empty answer is still upstream's answer: relay 200 + zero bytes, do not error | `test_upstream_status_bytes_and_headers_all_travel[empty-body]` | safe |
| same arm / bytes **undecodable under any charset** (`\xff\xfe`, 503) | relay verbatim; `UnicodeDecodeError` is a parse failure, not a transport failure (`DC-CR2-F-2-r1` — the branch's `except (JSONDecodeError, UnicodeDecodeError)` → 502 shape is what this closure removes) | `[…][undecodable-bytes]`; and `json.loads(bytes)` decoding failure is caught by the same arm that produced the red at base bytes | safe |
| same arm / **parseable 4xx overflow** (SGLang `Context overflow: 50000 > 4096`, backend `sglang`) | `_raise_chat_overflow` still fires and the retry loop still runs `max_retries = 2` → three upstream calls, final 400 (T013-accepted contract, "behavior unchanged" in `T015`'s done-when) | `TestOverflowArmPreserved::test_a_parseable_4xx_overflow_still_drives_the_retry_loop` asserts `post.call_count == 3` and the terminal 400; red at both base commits, which is the wiring proof that the read-once rewrite kept the parsed form reachable | safe |
| `_chat_answer` / upstream body read **once**, inside the response context | payload drained before the connection returns; no second read, no unclosed payload | `_chat_answer(resp, await resp.read(), profile)` sits inside `async with session.post(...) as resp`; the re-aimed mechanism pin `upstream.read.assert_awaited_once()` + `upstream.json.assert_not_awaited()` (`tests/test_routes_integration.py::TestChatRoute::test_chat_nonstream_compatible_json_transforms`) fails if a second read or a `.json()` call is reintroduced | safe |
| `_chat_response` / upstream declares a **lying `content-length`** on a body the proxy rewrote | the relayed length is the dropped-and-derived one, or the caller waits for bytes that never come (`utils.response_headers` docstring; `T013`'s settled rule) | `test_a_rewritten_body_carries_the_length_of_what_is_sent` — fixture upstream sends `Content-Length: 999`, witness asserts the received length ≠ 999 **and** equals the header the client got; plus `test_header_parity.py`'s `[…][chat]` row of `test_lying_upstream_content_length_never_reaches_the_client` | safe |
| `_apply_response_transforms` / **both** return paths (rewritten body and untouched body) | identical header source, so end-to-end names (`X-Request-Id`, `Retry-After`, `X-RateLimit-Remaining`) survive whichever path a body takes (`DC-CR2-F-1-r1`) | `test_upstream_end_to_end_names_reach_the_caller[fixed-body]` / `[unfixed-body]`, with `rewritten` derived from the body rather than asserted blind; four `test_header_parity.py` ingress classes gained the `chat` param | safe |
| `_make_chat_request`, `_stream_single_request`, `_non_stream_single_request`, `handle_passthrough` / genuine **transport failure** (`ClientError` before anything is sent) | keep a 502, but a client-facing body that names **no** upstream topology; the exception stays in `log.error` (`DC-CR2-F-2-r1` leak half; the leak was witnessed live by the 069/071 gates) | `TestLeakFreeErrorBodies::test_the_502_names_no_upstream_topology` × 4 arms — asserts `config.UPSTREAM`'s host:port absent, `"Cannot connect"` absent, and the exact static body; `test_the_exception_is_still_for_the_operator` asserts one `Upstream error` line carrying the host:port, i.e. the cause moved rather than vanished. `passthrough.py` had no log line before; it has one now | safe |
| `_inject_preserved_thinking` / client sends a present-but-**non-mapping** `chat_template_kwargs` (`null`, `[]`, `[{...}]`, `"enable"`, `7`, `True`) with the shipped default `PRESERVED_THINKING` on | replace with a fresh dict and inject the pair — the flag's documented guarantee holds for every forwarded body; the route must not 500 (`main.py:107-110`, `DC-CR2-F-3-r1`) | `test_a_non_mapping_is_replaced_and_the_pair_goes_out` × 6, each asserting `status == 200`, `post.call_count == 1`, and the exact injected pair in the body that actually left. At base bytes these fail with `AttributeError: 'bool' object has no attribute 'setdefault'` → HTTP 500 and no upstream call | safe |
| same / client sends a real **mapping**, including a deliberate `enable_thinking: False` | merge, do not flatten; `setdefault` semantics unchanged, so a client that turns thinking off still does | `test_a_mapping_from_the_client_is_merged_not_flattened` × 2 (`thinking_budget` preserved; client override still wins) — the witness that fails a build which overwrote rather than defaulted | safe |
| same / field **absent** | still gets the pair (the flag's own help text) | `test_an_absent_field_still_gets_the_pair` | safe |
| `handle_messages` / flag **off**, any value of the field | body forwarded byte-identical; no key written — the guard must not cost anything on the off arm (`DC-CR2-F-3-r1`'s risk shape) | `test_the_flag_off_arm_forwards_the_body_untouched` × 3 (`null`, scalar, mapping), asserting the outgoing body carries the value unchanged | safe |
| `_stream_single_request` state mapping / **dead key removed** (`QB:routes-messages/write-only-state-input-tokens`) | session stats still receive `input_tokens`; only the write-only field goes | `grep -n "input_tokens" buster_ripper/routes/messages.py` → the value reaches its consumer as a local at `:210`/`:214`, never through `state`; existing `tests/test_stream_relay_commit.py` and the stats witnesses stay green | safe |
| **Partial completion / post-commit failure** on the streaming relays (`_relay_after_commit`, `_relay_chat_stream`) | position-not-cause, no second response — the contract `T013` accepted | Excluded from this touch-set by `T015`'s blueprint and `DC-CR2-F-2-r1` ("`_relay_after_commit`'s post-commit contract … not in this touch-set"). AST foreclosure: `_relay_chat_stream` among chat's held 11/11; `_relay_after_commit` among messages' held set | not-applicable (excluded) |
| **Second HTTP head after a post-commit failure** (`CR2-F-4`) | no code change in this session — record-lane only | Excluded by `DC-CR2-F-4-r1` and the codify-scope ruling rechecked there; tracked at `roadmap.md` §Chat route + `QB:routes-chat/post-commit-second-head` | not-applicable (excluded) |
| **Retry / duplicate delivery** — `handle_chat`'s retry arm re-issuing after overflow; what its terminal 400 body says | retry count is owed and witnessed; the terminal overflow body's *content* is `CR2-X-1`'s, which the routing pass returned to the collector lane as `malformed` | Excluded by both blueprints ("the `handle_chat` retry arm" / "`CR2-X-1` is `malformed`, stays with the collect lane"); the call-count half is covered by the overflow row above. Flagged for Yield at §Flagged, item 3, without a ruling | not-applicable (excluded) |
| **Cancellation / cleanup** in the touched arms | no new await point is introduced while holding a resource — the only `await` added is the single `resp.read()` inside the existing context manager | Source inspection of the diff: `_chat_answer` and `_chat_response` are synchronous; no task, lock, or file handle is in scope on any touched path | not-applicable (with evidence) |
| **Competing / out-of-order updates** | none introduced: every touched path is request-scoped, and the wave *removes* one write to a per-request mapping | Same inspection; no shared mutable state is added anywhere in the diff | not-applicable (with evidence) |
| **Restore/recovery** after a relayed upstream failure | not a proxy concern; nothing is cached, retried, or re-driven by any touched arm, and the static 502 is terminal | Excluded by the absence of state in the touched surface (grep: no cache/queue symbol added) | not-applicable (with evidence) |

## Capacity gate: TRIGGERED → extracted at Checkpoint 164 → closed at or under the ceiling

| Stage | `chat.py` | `messages.py` | `passthrough.py` |
|---|---|---|---|
| Wave base `1c84ebd` (raw lines) | **299** / 300 — Green ceiling, zero headroom | 293 | 53 |
| Trigger | additive production surface required (header threading + read-once + raw relay) on a file already at the FL-LNCP ceiling → gate fires; `T015`'s blueprint names the same condition and forbids a silent breach | additive guard on the injection block | additive log line + static body (no trigger: headroom, single concern) |
| Extraction `a4d1d26` (its own commit, behavior-preserving) | `_chat_answer` takes the read → overflow-check → transform sequence out of `_make_chat_request` (302 during the move) | `_inject_preserved_thinking` takes the two kwargs writes out of `handle_messages` (298) | — |
| Focused characterization re-run after the extraction | `a4d1d26`'s own record: **163 affected tests pass unchanged** — the move, not the closure, was what the re-run covered | | |
| Final, at `c8df363` | **300 / 300** — at the ceiling, not over it; no capacity review item owed | **299** | **56** |

The closure lands at or under the cap because it is largely *subtraction on the same
lines*: the two-arm `except (json.JSONDecodeError, UnicodeDecodeError)` body and the four
inline `web.Response(...)` literals are replaced by one `_chat_response` call per site.
`T015`'s blueprint asked for exactly that test ("can land at or under the cap; if it
cannot, file the capacity review item **with** the change — do not silently breach"). It
did. `chat.py`'s net movement across the whole wave is **+1 line** (299 → 300) for three
behavioral closures plus one new helper.

Function metrics, re-scanned here with `codify scan path <file> --git --detailed` (one
invocation per file, matching the recorded census in §3 of the surface evidence):

| Function | Before (`1c84ebd`) | After | Note |
|---|---|---|---|
| `_make_chat_request` | L:18 N:2 C:4 (P:5) | **L:14 N:2 C:3 (P:5)** | improved — the parse/relay sequence moved out; its five parameters are unchanged in number, only re-flowed onto one signature line, which is part of how `chat.py` held at 300 |
| `_apply_response_transforms` | L:24 N:2 C:6 | **L:14 N:2 C:6** | improved by 10 lines; C:6 and N:2 unchanged, one gray-zone metric |
| `_chat_answer` (new) | — | unflagged by the scanner (L:6 N:1 C:2, P:3) | module-private helper; hand-counted on final bytes, and the scanner reports nothing against it |
| `_chat_response` (new) | — | unflagged (L:1, P:2) | single-expression answer builder |
| `_inject_preserved_thinking` (new) | — | unflagged (L:6 N:1 C:2, P:1) | the guard adds two executable lines to the extracted block |
| `handle_chat` | L:29 N:2 C:6 | **L:29 N:2 C:6 — identical**, zero bytes changed | re-measured at base bytes and at final bytes; the AST foreclosure and the numeric census agree |
| `_non_stream_single_request` | L:33 N:4 C:8 | **L:33 N:4 C:8 — unchanged** | pre-existing hard violation inside a touched surface, baseline preserved: the arm edited is one line and one line only. Recorded as debt, not fixed here (§Concerns) |
| `_stream_single_request` | L:30 N:2 C:3 | **L:30 N:2 C:3** | unchanged length; the absorbed backlog item removes a dict entry, adds none |
| `handle_passthrough` | L:25 N:1 C:2 | **L:26 N:1 C:2** | +1 gray-zone line for the operator's log entry; single metric, no other violation |

Test-surface scan, recorded because Yield will run it and the numbers are not zero: the new
module reports **"8 total issues found in 8 functions — Medium Parameters"** and
`tests/test_header_parity.py` / `tests/test_routes_integration.py` report
**360** / **467** raw lines, review band. Every one of the 8 parameter reports is a pytest
fixture parameter on a test function (`self, proxy, mocker, case` = P:4, gray-zone) — inherent
to the framework, not a behavior violation — and both file sizes were already open backlog lines
before this wave (`QB:test-header-parity/file-size`, `QB:test-routes/file-size`), with the new
module recorded the same way (`QB:test-upstream-answer-relay/file-size`) so the three-file
surface's size is visible at the next touch instead of re-discovered.

Whole-file scanner census, re-derived by this pass from the base bytes checked out to a scratch
path (not quoted from the surface evidence): `chat.py` **18 issues in 8 functions at `1c84ebd` →
14 issues in 8 functions now** — the wave removes four reports while adding no new function to the
count; `messages.py` **18 in 6 → 18 in 6** (unchanged, with `handle_messages` slipping L:30→L:28
while staying in the same Medium-Length band); `passthrough.py` **1 in 1 → 1 in 1**. `handle_chat`
measures **L:29 N:2 C:6 at base and L:29 N:2 C:6 now**, which is the numeric form of the
blueprint's "do not reopen the retry arm".

No new hard violation, no worsened baseline, no touched surface crossed 500 raw lines, and
the 501-700 "must extract" band is untouched — the wave's largest file is
`tests/test_routes_integration.py` at 467, which was already at 461 (review band, not a
new breach) and whose +6 is a docstring and a re-aimed pin.

## Backlog pickup:

Read at the start of the task, cross-checked against the wave's actual touch-set
(`chat.py`, `messages.py`, `passthrough.py`, `tests/test_routes_integration.py`,
`tests/test_header_parity.py`, `tests/test_upstream_answer_relay.py`). **Seven open lines name a
touched file.** One is absorbed, six are recorded skips with reasons (one by its own trigger's
condition, five because the remedy is the item's own design or a multi-file refactor plan that the
pickup bounds leave to the quality pass); every one is annotated in `quality-backlog.md` itself so
the next touch inherits this wave's measurement rather than re-deriving it.

`QB:routes-messages/write-only-state-input-tokens` — **absorbed**, rides this increment with no
separate report or checkpoint. `_stream_single_request`'s state literal no longer carries
`"input_tokens": 0` and `_note_message_start` no longer writes it; the value still reaches
`update_session_stats` through the local at `messages.py:207`. Evidence: the grep in the
Failure-Path table's last row, plus the item's own remedy text ("drop the key — the local
`input_tokens` already carries the value") being exactly the change made. Marked `[x]` with this
iteration id.

`QB:routes-chat/raw-bytes-passthrough-when-unfixed` — **skipped, with its reason.** Two
independent grounds, both already on the record rather than invented here: (1) `chat.py` lands at
**exactly 300/300**, and the item is purely additive — passing the upstream's original bytes
through when `fixed=False` instead of re-serializing cannot be done in negative lines, so it would
breach the ceiling the blueprint forbids breaching silently; (2) its own trigger reads "lands with
T015 **if it fits**, else next touch of the chat non-stream relay", and `DC-CR2-F-1-r1`'s Rejected
candidates classify it as "correct but additive to the header closure — deferred below, **not
competing**". It does not fit, so it stays open, handle and history preserved. Per the pickup rule
this is a recorded skip (defer-tier at worst), never a rejection cause.

Five lines whose trigger this wave satisfied but whose remedy is not this wave's to take — each is
"its own design or a multi-file refactor plan", which the pickup bounds leave to the quality pass:

| Line | Touch-set relation | Why not absorbed here | What this wave measured |
|---|---|---|---|
| `QB:routes-messages/handler-lncp` | `handle_messages` lost its comment line in this wave, its kwargs block in the Checkpoint 164 extraction | **routed** to `roadmap.md` §RM-001; the row's own text says the restructure "is this item's own design decision under `062 R-A` and `DC-S10-C4-r1`" | `handle_messages` **L:28 N:2 C:11** (was L:30 N:2 C:11) — L improved 2, the hard C:11 untouched |
| `QB:routes-chat/request-lncp` | `_make_chat_request` is in the wave's touch-set | same §RM-001 routing; the row calls its own restructure "blocked the same 299/300 way", and the ceiling is now fully consumed | `_make_chat_request` **L:14 N:2 C:3 (P:5)** (was L:18 N:2 C:4) — one gray removed, none added |
| `QB:test-header-parity/file-size` | module grew 355 → **360** | the remedy is moving `TestSharedHelpers`/`TestSiteCensus` into a helper-level module — a file-shape reorganization of an accepted suite module, which would put an unrelated diff under CR2's review; the row itself is kept as "next-touch job" | +5 = two docstring lines recording that the chat boundary joined the census, three comment lines above `INGRESS_SITES`; the four `[…][chat]` params ride existing lines |
| `QB:test-routes/file-size` | module grew 461 → **467** | same open-ended reorganization the row already deferred at 050-01; **and** the alternative — putting the 29 witnesses here — crosses the 500 must-extract boundary, which is §Concerns 1 | +6 net = +9 (the `_create_mock_response` docstring and its derived payload), +5 (the re-aimed mechanism pin), −8 (the removed defect-pinning witness); ratchet not crossed |
| `QB:test-routes/duplicate-upstream-stub` | the wave edited `_create_mock_response`, the row's subject | migrating the module to `StubUpstream`/`proxy` is a 13-mention rewrite of an accepted module's fixtures — a multi-file plan, not a CR2 side effect. **Its precondition list is now stale in the wave's favor**, recorded on the row: the `:239` case that needed a `json()` that RAISES no longer exists (that witness was replaced), and `:168`'s post-construction `read.return_value` mutation is now a derived payload inside the helper | `tests/upstream_stub.py` verified to still answer `json()` unconditionally (no failure knob), so the row stays genuinely blocked on one knob, not two |

The new module follows the pickup rule's reuse side rather than its refactor side:
`tests/test_upstream_answer_relay.py` imports `StubUpstream` and uses `conftest.py`'s `proxy`
fixture directly (`codify-test-cost` rule 6), and adds no second mock, fixture, or builder of its
own — its three module-level helpers (`_upstream_authority`, `_sent_body`, `_answer`) are body
builders for its own parametrized tables, each built once per test in memory.

No other unchecked line names a touched file. `QB:routes-chat/post-commit-second-head`
(`chat.py:174-188`) and `QB:test-stream-relay/write-position-injection`
(`tests/test_stream_relay_commit.py:48-54`) are open and were not absorbed because neither surface
is in this wave's touch-set — `_relay_chat_stream` holds at 11/11 unchanged and
`test_stream_relay_commit.py` carries zero bytes of diff; the first is `DC-CR2-F-4-r1`'s record-lane
item and the second is that module's own witness hole.

## Dependency claim — checked, and already on the ledger

**The note the implementing pass surfaced, re-verified.** `aiohttp.ClientResponse.json()`
with its default expected content type **accepts** `application/problem+json`: the default
path matches with `json_re = re.compile(r"^application/(?:[\w.+-]+?\+)?json")`
(`client_reqrep.py:102`) via `_is_expected_content_type` (`:260-265`), and the default
`content_type: Optional[str] = "application/json"` is declared at `:747`. Consequence for
this increment's **witnesses, not its production**: an RFC-7807-shaped
(`application/problem+json`) upstream answer sails through `.json()` without complaint, so
a test that reads its result through the client's own `.json()` cannot distinguish the
relay of the upstream's bytes from a body the proxy authored. The wave's witnesses
therefore assert on `await response.read()` bytes and on the headers that arrived, and the
re-aimed mechanism pin asserts `upstream.json.assert_not_awaited()` rather than inferring
the shape from a decoded result. The fixture that exercises it carries
`Content-Type: application/problem+json` for exactly this reason
(`tests/test_routes_integration.py::TestChatRoute::test_chat_nonstream_compatible_json_transforms`).

**Reach check, done rather than assumed.** `assumptions/claims.md` already carries this
proposition: **row 84**, `pkg:aiohttp@3.13.3`, Tasks `T008, T008.001`, Scope **shared**,
Bundle `B113`, State `HOLD`, handle `CL:aiohttp/structured-suffix-json-content-type-validation`.
Its verdict record (`assumptions/verdicts/claim-84.md`, checked 2026-09-27) cites the same three
sites, and row **83** is its sibling (`text/event-stream` → `ContentTypeError`). A shared row is
this pass's to *rely on*, never to re-resolve locally: had the premise failed on current bytes the
correct move would have been to raise it — T008/T008.001's accepted gates cite the same fact — and
it did not. So the claim was re-read and its freshness re-established rather than inherited:
installed aiohttp is still **3.13.3** and
`sha256(client_reqrep.py)` = `7844440eb67433c6454edd708c776e47e3fcfec9b8d0aeb980dcfe88c1986ac9`,
byte-equal to the fingerprint that verdict pinned, so row 84 is current on this tree.

One symmetry worth naming: the witness style the note explains is not new to this file —
`test_chat_stream_upstream_error_retains_representation` has pinned `upstream.json.assert_not_awaited()`
for `application/problem+json` among its three cases since `T008.001`
(`tests/test_routes_integration.py:167-183`). What the wave did is make the **non-stream** arm
witnessable the same way, once the production read stopped being a `.json()` call.

Two adjacent rows are worth naming for Yield because this wave moves the code they
described: **row 65** (`chat.py`'s `_make_chat_request` parses every upstream response
with `await resp.json()`) is already marked `superseded`, and the read-once arm is the
second move away from it; **row 372** (`_note_message_start` takes its timing origin and
session id from the state mapping `_stream_single_request` builds before its `try`, T013,
Scope `single`, State `retired`) is the only recorded claim whose subject the absorbed
backlog item edited — both operands it names, `state["started"]` and `state["sid"]`,
survive unchanged, so the retired row is not contradicted by the key removal.

## Independent confirmation (this pass's own runs)

Nothing below is inherited from the two evidence files; each figure was re-measured
against the working tree before Checkpoint 165.

| What | Command | Observed | Claim being checked |
|---|---|---|---|
| Affected scope — the three changed route modules and everything that drives them | `uv run pytest tests/test_routes_integration.py tests/test_header_parity.py tests/test_upstream_answer_relay.py tests/test_stream_relay_commit.py tests/test_context_overflow.py tests/test_image.py tests/test_image_stability.py tests/test_backend.py tests/test_server_hooks.py -q -p no:cacheprovider --no-cov` | **244 passed**, 23 warnings, 5.79 s — zero failures | (not previously claimed; the wave's own gate) |
| Full hermetic suite | `uv run pytest -q -p no:cacheprovider` | **439 passed**, 22 warnings, 9.14 s; coverage `TOTAL 635 stmts / 27 miss / 96%`, **95.75%**, `--cov-fail-under=80` satisfied | the claimed "439 @ 95.75%" — **confirmed as measured, not accepted on assertion** |
| File sizes | `wc -l` | `chat.py` **300**, `messages.py` **299**, `passthrough.py` 56; tests 467 / 360 / 307 | §2 of the surface evidence |
| LNCP | `codify scan path <file> --git --detailed`, one file per invocation | chat 2 issues / 2 fns; messages 7 / 2; passthrough 1 / 1 — identical to §3 | §3 of the surface evidence |
| LNCP **before**-column | `git show 1c84ebd:<file>` to a scratch path, then `codify scan path <scratch> --detailed` | `handle_chat` L:29 N:2 C:6 · `_apply_response_transforms` L:24 N:2 C:6 · `_make_chat_request` L:18 N:2 C:4 · `handle_messages` L:30 N:2 C:11 · `_non_stream_single_request` L:33 N:4 C:8 · `_stream_single_request` L:30 N:2 C:3 · `handle_passthrough` L:25 N:1 C:2 — every figure matches §4 of the surface evidence, re-measured rather than quoted | §4 of the surface evidence |
| Suite arithmetic | 407 (`dbd3072` baseline, [071](071-yield-phase.md)) **+29** new file **+4** chat ingress params **−1** removed defect-pinning witness = **439** | exact | the 439 is not a coincidence of a re-run but the expected delta |
| Wave test count | 29 + 65 + 32 | **126** | "126 passed after closure" — the three files collect exactly 126 today |

The fail-first claim (26 red / 100 passed against `1c84ebd`, same against `a4d1d26`, 126
green after closure) is recorded in [cr2-fail-first-run.txt](../evidence/cr2-fail-first-run.txt)
and is **not re-derived by this pass**: reproducing it means swapping base production bytes
into a tree holding the only copy of uncommitted work, which no instruction asked for.
Yield's replay is the place it gets re-established. One completeness gap is disclosed in
§Concerns, item 4.

## Concerns — raised, not ruled

**Flagged for the single local Yield, as the dispatch named them, without a ruling here.**

1. **The test surface spans three files, not two.** `tests/test_upstream_answer_relay.py`
   is new. The reason is arithmetic, not taste: `test_routes_integration.py` sits at **467**
   and the wave's 29 witnesses would push it past **500**, the boundary where FL-LNCP says a
   file must be extracted rather than reviewed. The two touched files were already in the
   review band (461 and 355). Precedent the wave relies on: `T014`'s three-file test surface
   was **accepted** at [068-01-yield](068-01-yield.md) on the same grounds, with production
   surface held to the declared files — which is what happened here (three route modules,
   zero production files outside them; `utils.py` was consumed, not edited). If Yield rules
   the split wrong, the witnesses move back into an over-limit file; the bodies do not change.
2. **`tests/qa/harness.py:135` can no longer fire as its text reads.**
   `assert_upstream_response` carries both `assert 200 <= status < 500` and
   `assert b"Cannot connect" not in body`. The first still holds and still discriminates a
   proxy-local 502 from an upstream answer. The second was written for bodies that embedded
   `str(e)`; with the four arms now answering `{"error": "Upstream request failed"}`, that
   byte sequence cannot arrive from a proxy-local 502 at all. Whether the assertion is now a
   dead limb, a still-useful belt-and-braces check against an upstream that says so in its
   own body, or something to re-point is Yield's call: the harness lives in `tests/qa/`,
   outside this wave's touch-set and owned by the milestone QA lane, so **no byte of it was
   changed here**, and the wave's own witnesses test the same property positively
   (`TestLeakFreeErrorBodies`).
3. **Two client-facing bodies still embed `{e}`.** `chat.py:55-58` and `messages.py:242`
   (`"Context overflow after {max_retries} retries: {e}"`) interpolate the exception. Both
   sit inside the retry arm `T015`'s blueprint puts on its must-not-touch list, and the
   egress facts for those bodies are what `CR2-X-1` asserted — the item the routing pass
   returned to the collector lane as `malformed`, with `DC-CR2-F-2-r1`'s static-body rule
   scoped to the `ClientError` arms. Whether the two `{e}` bodies are in or out of CR2's
   intended closure is therefore not this wave's to settle, and it is recorded here so
   nobody discovers it as an untouched finding after the fact.

**Additional observations, same standing (none is a defect claim against the increment):**

4. **Evidence completeness.** `cr2-fail-first-run.txt` enumerates all 26 failing ids for
   the run against `1c84ebd` but only **two** ids under the §"counterpart run" heading for
   `a4d1d26`, immediately above a `26 failed, 100 passed` summary line. The count and the
   green run are consistent with everything else in the file; what is missing is the
   enumeration. Flagged so Yield knows the excerpt is short before it treats the section as
   an itemized list.
5. **The AST foreclosure in the surface evidence reports one intentional violation:**
   `messages.py: must-not-move held=3/4 VIOLATIONS=['_note_message_start']`. The function is
   in the wave not because a finding named it but because the absorbed
   `QB:routes-messages/write-only-state-input-tokens` line points at its dead write. The
   blueprint's must-not-touch lists (T015: `_relay_chat_stream`, `_prepare_headers`,
   `_carries_token`, `_apply_request_transforms`, the `handle_chat` retry arm — all held
   11/11 in `chat.py`) are intact; `T016` never listed `_note_message_start`.
6. **`_non_stream_single_request` remains a hard violation (L:33 N:4 C:8)** inside a touched
   surface, byte-identical to its baseline at `1c84ebd`. The capacity ratchet permits
   checkpointing a touched legacy violation whose baseline is preserved, so this is recorded
   as debt rather than fixed: the closure in that function is one line for one arm, and
   fixing L:33/N:4 there is a refactor the CR2 batch never asked for. It is the same function
   `roadmap.md` §Chat route and the messages-side capacity notes already watch.
7. **`passthrough.py`'s 502 arm grew a log line.** `handle_passthrough` had no `log.error`
   for `ClientError`; taking the exception text out of the body without putting it in the log
   would have left an operator with a bare 502 and no cause. That is the smallest change that
   keeps the pair of obligations in `DC-CR2-F-2-r1` ("the exception is the operator's, the
   body is the client's"), and it is why the file is 56 rather than 53+2 lines of body alone.

## What landed, per routed finding

| Ledger | Finding | Where | Witness set |
|---|---|---|---|
| `DC-CR2-F-1-r1` | CR2-F-1 — non-stream answers rebuilt with `Content-Type` alone | `chat.py`: `_chat_response` + `_apply_response_transforms(resp_data, profile, resp)`; both return paths now call `utils.response_headers(resp.headers)` | `TestTransformPathHeaderParity` (3) + 4 × `[…][chat]` ingress params |
| `DC-CR2-F-2-r1` | CR2-F-2 — unconditional `resp.json()` erases well-formed non-JSON answers; `str(e)` leaks upstream topology | `chat.py`: read-once + parse-failure fall-through to the raw relay. All four `ClientError` arms on three routes answer a static body | `TestNonJsonRelay` (8), `TestOverflowArmPreserved` (1), `TestLeakFreeErrorBodies` (5), the re-aimed mechanism pin, and the replaced `test_chat_upstream_undecodable_json_is_upstream_error` |
| `DC-CR2-F-3-r1` | CR2-F-3 — `chat_template_kwargs: null` → 500 with no upstream call under the shipped default | `messages.py`: `_inject_preserved_thinking` isinstance-guard, replace-not-skip | `TestPreservedThinkingGuard` (12), incl. the merge and flag-off pins |
| `DC-CR2-F-4-r1` | CR2-F-2/F-4 sibling — the post-commit second head | **no code change**, per the ledger's own selected closure (record-lane only) | none owed here |

Test-surface bookkeeping: `tests/test_header_parity.py`'s module docstring and
`TestIngressSites` docstring moved from "two response boundaries" to "three", which is the
census the wave changed rather than a restyle; `tests/test_routes_integration.py`'s
`_create_mock_response` now answers `read()` with the bytes of whatever was declared (a
fixture honesty fix — declaring `json_data` alone used to be sufficient because nothing read
the raw body).

## Durable signal for later compaction

- **Override entries exercised: none.** No protected invariant was touched; `invariants/`
  holds only `dump/`, `server/` and `assumption-claims.md`, and no active EI/OI governs
  response relay or client-error body shape (both blueprints' invariant cross-reference is
  **Neutral** against the registry, **Aligned** with `T013`'s settled boundary-parity
  direction). Zero invariant records created, edited, superseded or deleted — the triad rule
  holds; the code and tests are the only sides that moved.
- **`FK-011` stays open and untouched** (`DC-CR2-F-3-r1`'s explicit constraint: the guard is
  posture-independent). No `decisions.md` byte moved; no `US-*` was reserved because the
  dispatch carried no new user-provided normative content — it transmitted a routing pass's
  existing dispositions.
- **No plan decision was overridden.** Every selected closure in the four ledgers is
  implemented as written; the only recorded deviation from the blueprints' *file* lists is
  the third test file, escalated at §Concerns 1 rather than silently kept.
- **Superseded wording, preserved here rather than stacked in the plan:** the pre-wave
  witness removed at `test_routes_integration.py` read `assert await response.json() ==
  {"error": "Invalid JSON from upstream"}` under the name
  `test_chat_upstream_undecodable_json_is_upstream_error` — it pinned the defect shape the
  blueprint directed be replaced, not a behavior worth keeping.
- **Retry counters:** `T015` and `T016` each remain at 0 — no Yield pass has yet ruled on
  this wave.

## Questions asked

None. No pre-question gate fired: every choice the tree presented had a single
authority-ranked answer already on the record (the four ledgers, the two blueprints, the
backlog lines' own triggers), so nothing rose to `CLOSURE: question`, `BLOCKED-APPROVAL`, or
`PLAN-GAP`. The three dispatch-flagged items are findings for Yield, not questions.

## For Yield

- Replay per both blueprints' `Yield replay` clauses: (a) 200 JSON + end-to-end headers,
  (b) `text/html` 502, (c) an injected `ClientError`, at `/v1/chat/completions` non-stream;
  the null-kwargs 500 before/after; connection failures against all three routes; and the
  no-bytes-moved check on `_relay_chat_stream`, `_prepare_headers`,
  `_apply_request_transforms`, `handle_chat`.
- Rule on §Concerns 1, 2 and 3. Nothing in them blocks a pass on the increment's own
  done-when clauses.
- The assumption lane owes its own bookkeeping independent of this wave: `T015`/`T016`'s
  plan rows carry `Assumption: required:new-surface` in the external-review closure-packet
  lane, and no row of `assumptions/claims.md` was extracted for the CR2 batch (the last
  extraction predates it). `DC-CR2-F-1-r1`/`F-2-r1`/`F-3-r1`'s factual premises are
  therefore re-derivable but not blind-checked as a set; the two aiohttp facts this report
  relies on are the exception — rows 83/84, `HOLD`, bundle `B113`, fresh on these bytes.

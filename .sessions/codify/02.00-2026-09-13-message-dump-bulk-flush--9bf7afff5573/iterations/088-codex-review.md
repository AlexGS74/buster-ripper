# 088 — External review, fourth pass (CR4)

Dispatched by the orchestrator on the post-wave bytes, **before** MS-001's Yield Phase, so a fresh
finding would not force another boundary restart. Verbatim report from the `codex-review` subagent.

## Provenance

- **Range.** Base resolved from `iterations/079-codex-review.md:22` → Review object `650894111c…` =
  **Checkpoint 183** (the dispatch's parenthetical said 186; 186 is `e204b18`, the Assumption-Decide
  routing checkpoint, one commit past the 185 that persists CR3 — the Review object as named was used).
  Through `HEAD` = `c5e2be4`: 28 commits, of which code bytes are **4 production files + 5 test files**;
  everything else in the range is `.sessions/` and `invariants/` bookkeeping.
- **Pass A (broad):** `codex review --base <review-object> --title "CR4…"` — `gpt-5.6-terra` @ medium,
  `codex-cli 0.160.0`, session `01a11480-350e-7e40-b74f-e2732417d04e`, exit 0. Verdict as returned:
  *"No actionable correctness regressions were identified in the changed production paths. The full test
  suite passes (476 tests)."*
- **Pass B (scoped, answers the three questions):** `codex review --uncommitted`, session
  `01a11483-377a-7a10-9adf-56158d7cd185`, exit 0, against a scratch worktree holding exactly the
  code+test delta. **A skill/CLI defect surfaced:** on 0.160.0 the positional PROMPT is rejected with
  `--base`, `--commit` **and** `--uncommitted` (exit 2,
  `error: the argument '--base <BRANCH>' cannot be used with '[PROMPT]'`), so the skill's Step-2
  invocation as written cannot run. Workaround was to pass the settled-authority context as `AGENTS.md`
  in a throwaway worktree — the reviewer read it (it appears in its own transcript as `ROOT INSTRUCTIONS`).
  Worth a fix to `~/.dotfiles/.agents/skills/codex-review`.
- Independently: full suite re-run → **476 passed**, zero skips across the new witnesses (49 selected by
  name); no repo byte touched, `git status` empty at exit.

## Findings

**[P2] The `T019` normalize-before-keying repair gives `/v1/messages/count_tokens` a 500 it did not have
— and base never had.** `buster_ripper/routes/count_tokens.py:87`.

Measured, same probe, only the range differing:

| body (valid JSON) | pre-range `6508941` | `HEAD` |
|---|---|---|
| `{"tools":[null]}` | 200 | **500** |
| `{"messages":[null]}` | 200 | **500** |
| `{"messages":["x"]}` | 200 | **500** |
| `{"system":[null]}`, `{"messages":[{"content":[null]}]}` | 200 | 200 |

`AttributeError` escapes `handle_count_tokens` from `normalize()` at `buster_ripper/normalize.py:75`
(`t.get("name","")` on a non-mapping `tools` element) and `:117` (`msg.get("content")` on a non-mapping
`messages` element); a third site, `:192`, joins the same class under `STRIP_DATE` + sglang. Base's
`count_tokens` (`buster_ripper_legacy.py:725-760`) never calls `normalize` — it parses to `{}` on any
failure and keys off `_session_id(body)` — so it returns an estimate, and the pre-range branch matched it.
This is exactly the class `CR3-F-04` was rated **P2** and repaired for (a network-supplied payload
producing a proxy 500 with a traceback instead of the route's own answer), with the same `latent` reach
verdict: exposed route, no guard prevents the input. Not P1: malformed per the Anthropic schema, no wrong
data served, no leak, no leak of handles.

**No regression-of-disposition:** `DC-CR3-F-02-r1`'s requirement (digest computed after normalize) is
still met — the defect is a side effect of *how*, not a failure of the fix.

**Closure:** `absorbed`, owner Forward, same lane as `T019`. The deterministic part is that the new 500
must go; `session_id()` reads `system` only (`buster_ripper/session.py:93-98`) and of normalize's four
steps only `_normalize_system` writes `system`'s text (`_normalize_cache_control` deletes `cache_control`
keys, which the digest never reads), so the invariant is satisfied by narrowing the call to the
system-strip step — that restores base's tolerance on this route and leaves `/v1/messages` byte-identical.
**One half is a genuine ruling, not a repair:** hardening `normalize()`'s two element loops instead would
also change `/v1/messages`, where base 500s on the same input today (verified at both `6508941` and
`HEAD`) — parity protects that crash until someone decides otherwise.

**[P3] Blank line introduced at EOF.** `tests/test_upstream_answer_relay.py:406` — the file ended with a
single `\n` at `6508941`, and the appended classes left `\n\n`. Cosmetic; no lint gate runs on this tree
(`CR3-F-15`'s `QB:lint/ruff-f-select-findings` covers `F` rules only), and `tests/test_dump.py` already
carries the same shape.

Nothing else in the range is a bug. Not manufactured further.

## Answers to the three questions

**2. Is `T019.001` complete? Yes — verified on the installed library, not on the diff's wording.**
`web.Response(body=<str>)` really does store a `StringPayload` (`aiohttp/web_response.py:697-721`),
`str()` of it is the object repr, and `Response.text` (`:723-730`) returns the string that went in. Both
objects the loop can be handed read strictly through `.text`: the `status>=400` arm's `StringPayload` at
utf-8 (`content_type="application/json"` names no charset → `charset is None` → utf-8 fallback — confirmed
empirically), and the `json_response` 502 arm's bytes. `body=""` yields `''`, not `None`, so
`parse_context_overflow_error` never receives a non-str; and `isinstance(result, web.Response)` cannot
catch the committed `StreamResponse`. **The retry fires:** three upstream calls, reissue 2 and 3 carrying
`max_completion_tokens == 256`, and the terminal arm at `messages.py:130-134` is live (it was dead too).
The re-issued request carries the loop's intent — one `dict(body)` copy rewritten in place, sent as `json=`
with unchanged headers, and the test snapshots the body **per call** rather than reading `call_args_list`,
which is the aliasing trap that would otherwise credit a pre-reduced build. The one sub-question about the
reissue — whether an Anthropic-schema upstream honors `max_completion_tokens` at all, when the client sent
`max_tokens` — is already owed as a `CR2-X-1` egress fact (`quality-backlog.md:78`), and the same field
name is what the already-firing non-stream arm uses (`:279`) and `chat.py:61`; not new here.

**3. Are any repairs superficial? No, and the evidence is unusually strong.** `client_max_size=0` is a
real deletion of the cap, not a bigger number — `web_request.py:675` is `if self._client_max_size:`, so 0
skips the check outright. The stream deadline is genuinely unarmed: `client.py:584-590` shows a
request-local `ClientTimeout` **replaces** the session's wholesale rather than merging, and
`test_the_streaming_call_passes_a_timeout_with_nothing_armed` asserts all four components are `None`
instead of trusting a `total` that looks unlimited, while pinning that the non-stream arm still inherits
600. The `CR3-F-02` witness is end-to-end behavior, not fix-assertion — it drives `/v1/messages` to record
stats, then asks `count_tokens` the same body (pre-range it answered `45` where `1234` belongs). `T020`'s
EMA pair drives production's `poll_kv_cache` across ticks and asserts literals for α=0.3, with a mutation
replay that reddens on swapped operands (`[0.75, 0.225, 0.2285]`) and controls that redden when the body
never arrives — the old pair could not fail at all. `tests/conftest.py` builds the real `create_app()`, so
the cap witness exercises the real construction site. Fail-first transcripts carry real 413s, a real
truncated stream, and real `LookupError`/`UnicodeDecodeError` tracebacks in each direction.

## Checked and cleared (so this isn't re-opened)

`_error_body`'s fallback is not a double read — `ClientResponse.text()` (`client_reqrep.py:732-740`)
populates `self._body` before decoding, so the subsequent `read()` returns cached bytes inside the live
`async with`; a `ClientPayloadError` from either call lands on the function's own `except aiohttp.ClientError`
→ 502. The `get_encoding()` suppression claim (`:709-727`) is accurate, so response-side `LookupError`
really is unreachable while request-side is not — the asymmetry the new witnesses pin rather than argue.
`result.text` returning `Optional[str]` is not a live hole (both arms always carry a body, empty included).
Every `aiohttp` claim row added to `invariants/index.md` that the range depends on
(`client-max-size-falsy-skip`, `per-request-timeout-override`, `web-response-str-body-text-property`,
`string-payload-str-representation` as DOES-NOT-HOLD) matches the installed 3.13.3 source. `count_tokens`'
non-dict body (`[1,2]`) still 500s — but `session_id()` raised there before the range too, so that is
unchanged, not moved. One deliberate consequence to note rather than fix: the estimate at
`count_tokens.py:94` is now computed over the normalized body, which is what the proxy forwards and is
documented at `:86`; with `STRIP_IMAGES` on it still counts image bytes because the empty `session_id`
keeps the call image-inert, exactly as pre-range.

Already-owned neighbours checked rather than re-reported: the `{e}` interpolation in both terminal overflow
bodies (`QB:routes-overflow-body/exception-text-interpolated`), split `message_start` (RM-009), the three
armed-`total` arms (RM-010), `chat.py`'s 300-line capacity rule.

## Disposition status at write time

Both findings are **un-routed** at this writing. `CR4-F-01` (P2) is a production defect introduced by an
accepted task and must be dispositioned through the owning route before MS-001's Yield Phase can carry
the boundary. `CR4-F-02` (P3, cosmetic) awaits the same pass.

## Dispositions (filled in by the routing pass — [089-assumption-decide](089-assumption-decide.md), 2026-10-07)

Both findings re-established on current bytes **by execution at three points** — base `72e3a1c6`, the pre-range
branch `6508941`, and `HEAD` `c5e2be4` — before either was dispositioned:
`evidence/cr4-probe-matrix.txt`. The matrix reproduces as reported. Two corrections, both measurement rather than
argument: the `:192` site is a member of the class but is **masked by `:117` today** (it becomes the raising frame
only once `:117` is hardened — §6), and base's `/v1/messages` column needed a live upstream stub to be attributable
at all, because with a dead one it 500s on every body including the well-formed control (§3).

| Finding | Disposition | Reach | Route |
|---|---|---|---|
| `CR4-F-01` (P2) | **absorbed** | `latent` — exposed route, no guard prevents a non-mapping element, nothing pins it, so fix-or-pin resolves to fix | **`T019.002`**, nested under `T019` on `T019`'s owning surface; one-member wave; ledger `DC-CR4-F-01-r1`; blueprint in `handoff-context.md` §Planned Task Context |
| `CR4-F-02` (P3) | **accepted** | inert — nothing reads a file's final blank line, and no gate on this tree can | `QB:tests/blank-line-at-eof`; owner = the next Forward pass that edits either file. One byte, no task, no wave |

**The proposed closure is adopted; the open half is ruled, not deferred.** Narrowing the `count_tokens` call to the
system-strip step is the selected closure, and the digest equality it depends on is now measured (18 cases × 2 axes,
zero differing digests — §4), as is the restored route behavior (§5). On the second half: **`/v1/messages`' 500 is
protected by parity** — base raises the same `AttributeError` in its own `normalize()` on these same bodies (§3), and
the package matched base before the range and after it (§1) — and hardening the loops turns those 500s into relay
traffic (500 → 502 measured, §7), which is a change to the one route `G-003` holds still, whose constraint column says
"no normalization changes". So it is **not absorbed here**, and it is **not a `question`**: once base parity and
`G-003` are applied, no materially different admissible outcome is left unranked, which is the bar the pre-question
gate sets. The session's pattern of absorbing what its own migration introduced (`T018` and the 1 MiB cap) is
followed, not departed — that pattern keys on *who introduced the rejection*, and here base introduces it. Owner of
the posture, named: **`roadmap.md` §RM-012**, the route-input posture design pass (co-owned with §RM-001), with the
evidence pinned, the three-way answer shape recorded, the completeness obligation (all three loops, not two) and the
`T019.002` clause-5 drift guard that reddens 500 → 502 if anyone hardens `normalize()` under cover of this repair.

**Capacity, answered plainly:** the fix lands with **zero bytes** in `messages.py` (334, its split is §RM-001's
decision) or `chat.py` (300/300), and none in `normalize.py` or `session.py` unless the implementation names a public
entry for the system step (three lines; both files are byte-unchanged across the whole range today).

No relayed sentence, severity label, finding or "Checked and cleared" item was altered by this routing; §Dispositions
is the router's own text and supersedes the packet's §Disposition status at write time. The two corrections named above
are the router's measurements, not edits to the report: the reviewer's table was right about every status code, and
wrong about nothing except the independence of the `:192` site.

## Orchestrator Prompt (verbatim)

> You are a subagent spawned by another session.
> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
> Mode: scoped adhoc misread gate
> Tasks: T013 (header-parity wave: F-4, G-1, G-2)
>
> Execute the forward-red skill for this session.

---

## Forward-Red — scoped adhoc misread gate: `T013` (header-parity wave: 057 F-4 → G-1 → G-2)

Date: 2026-10-05 · Pass: full depth on `F-4` and `G-1`, compact depth on `G-2` · HEAD at read: `25c2b77`

> **⚠ Iteration-ID collision (recorded, not hidden).** `next-iteration-id` returned **059** to this pass at
> ~10:33 and again to the concurrently-dispatched **`T014` image-slice** forward-red pass. That pass wrote its
> report to `059-forward-red.md` first (its baton stamp is already in `handoff-context.md` §Current Status, citing
> `059`), and this pass's write at 10:37 overwrote it before the collision was visible. This report therefore
> lives at **060**; `059-forward-red.md` is replaced by a notice rather than a report. **The `T014` pass's own
> findings are NOT lost**: its §Current Status stamp carries its verdict (RED 14 | 5 AMBIGUOUS | 8 footnotes) and
> the substance of rows A-1…A-5 with file:line evidence, which is enough for Decide to route. What is unrecoverable
> is its full attack table. Both passes' stamps now cite distinct files. No action for this pass's owner; the
> orchestrator may re-dispatch `T014`'s red pass if it wants the table itself, or route from the stamp.

### Clean-room compliance

Read only what Forward receives: `plan-summary.md` (§Task List `T013` row + its `<details>` restatement, §Goal
Definition Matrix, §Anchor Map, §Reuse Evidence), `handoff-context.md` (§Current Status, §Planned Task Context →
`T013`), `qa-plan.md`, and the code (`buster_ripper/utils.py`, `routes/{messages,chat,passthrough}.py`,
`server.py`, `session.py`, `buster_ripper_legacy.py`, `REFACTORING_PLAN.md`, `tests/`, `pyproject.toml`,
`.venv/.../aiohttp/{client_exceptions,web_protocol,web_response}.py`).
No interview / invite / decide / organize / collect iteration report was opened. `iterations/` was listed by name
only, never read.

### Verification profile — recorded by the owner? No

`grep -rn "Verification profile" handoff-context.md plan-summary.md` → **zero matches**. The §Planned Task Context
block carries the packet field set (finding id, evidence, required outcome, selected closure, done-when, owning
route, invariant cross-reference, closure-ledger key, execution order) but no risk/recovery/uncertainty profile.
Per proportional verification I did not silently classify that as low; bounded assessment made here:

- **Risk: high.** Concrete wrong outcome: a `Proxy-Authorization`/`TE` value still reaches `--upstream` on one of
  three request boundaries, or a second response head still lands on a committed SSE stream — both are
  upstream/client-visible and neither is observable through any gate the session owns.
- **Recovery: easy.** Call-site substitution and one `except` clause; no data, no migration, no external effect.
- **Uncertainty:** the wave's referents (`utils.forward_headers`, `utils.response_headers`, `_HOP_BY_HOP`,
  `_prepare_headers`, the five filter sites, `REFACTORING_PLAN.md`) sit **outside the Anchor Map**, which binds
  only the dump/TLS cluster. `required:new-surface` is recorded, so the premises go to the assumption gate; what
  is *not* covered there is intended-behavior binding, which is this pass's remit.
- **Route chosen:** full on `F-4` (multi-site, credential exposure, no coverage gate watching — `pyproject.toml:30-34`
  omits `buster_ripper/routes/*`) and on `G-1` (wire-level defect with an unenumerated exception surface);
  compact on `G-2` (documentation).
- **Budget:** 17 attacks attempted, ceiling 8 task-misreadings + mandatory goal-shape attempts; expanded twice
  (rows 3 and 6) because the first attempt exposed a load-bearing decision rather than a wording nit. No budget
  exhaustion — see §Coverage.

### Goal-shape attack status

`T013` gates no milestone, and the wave edits none of the observation surfaces in §Goal Definition Matrix
(`DUMP_DIR` write operations; the outbound TLS handshake), so `G-001`…`G-008` are not in scope — verified against
the matrix's own rows. The adhoc analogue is each member's **Required behavior ↔ Done when** pair, attacked for
"acceptance passes while purpose fails". That produced rows **4** and **6** (both goal-shape class, not counted
against the task-misreading ceiling).

### Attack table

| # | Surface (`G-*` or task) | Plausible misreading | Excluded by | Verdict |
|---|---|---|---|---|
| 1 | `F-4` site boundary | "**Every** header set forwarded across a proxy boundary… at **every route**" (the block's own invariant candidate) + "at every proxy boundary" (task title) reads the proxy-**authored** SSE header block at `routes/messages.py:145-153` as a fourth response-filter site to route through `response_headers`. Consequence is not cosmetic: `connection` is in `_HOP_BY_HOP`, so `Connection: keep-alive` is stripped from the committed SSE response, and a relay-upstream-headers variant also drops `X-Accel-Buffering: no` / overrides `Content-Type`. | **Partially.** `§Member F-4` states "The filter sites are **five** (three request, two response)" and the carve-out reason is given — but **only for `chat.py`**: "do not 'fix' `chat.py:158-164` into a filter… it is the proxy's own response, not a relayed upstream header set". Nothing in the forward-visible artifacts says the same word-for-word about the identically-shaped block in `messages.py`, which is the route the finding is about. | **AMBIGUOUS** |
| 2 | `F-4` mechanism | "no route carries its own copy of the policy" + done-when (1)'s token grep read as *centralise the policy*: a `@web.middleware` in `server.py` filtering every response, deleting the two response sites. The repo has live precedent (`server.py:48` `@web.middleware`, `:119` `web.Application(middlewares=[logging_middleware])`), and a middleware satisfies done-when (1), (2), (3) and (4) while changing the SSE response header set (row 1's harm) via a new, unauthorized artifact. | **Partially.** "Selected closure… route every filter site **through** `utils.forward_headers` / `utils.response_headers`" and "preserve `_prepare_headers`' empty-`Bearer` deletion" both assume per-site substitution; `§Anchor Map` adds "The `Mode: NEW` rows above remain the only authorization" — but that sentence scopes the *approved* plan's rows and `T013` is recorded as outside it, so a reader can discount it. The block never states "no new central mechanism". | **AMBIGUOUS** |
| 3 | `F-4` `_prepare_headers` | To reach "one shared implementation", the empty-`Bearer` deletion is moved **into** `utils.forward_headers`. `/v1/messages` and the catch-all then silently start dropping `Authorization: Bearer ` too — a behavior change on a route the finding never touches, and one that is invisible to done-when (2) ("`Authorization`… end-to-end headers survive" is tested with a *real* token, not an empty `Bearer `). | **No.** "preserve `_prepare_headers`' empty-`Bearer` deletion" does not say it must stay route-local. The plan's own required semantics ("pre-refactor proxy semantics") in fact settle it — `buster_ripper_legacy.py:621` (`_forward_headers`, shared, used at `:656`/`:812`/`:901`) vs `:840`, where the `Bearer` strip lives **inside one route's handler** — but the `T013` block never cites that asymmetry, and the row-6 hazard below means the file may not get read. | **AMBIGUOUS** |
| 4 | `F-4` acceptance quantifier (**goal-shape**) | Done-when (2) is "**per direction**" = **two** tests, and the Yield replay is "drive **one** real proxy request… and **one** upstream response". The Required behavior is "**every** boundary… at **every route**". A wave can pass all four done-when clauses while `chat.py` or `passthrough.py` keeps its own abbreviated list: done-when (1) is a *token* grep (`skip_keys\|_HOP_BY_HOP`), so a copy renamed `DROP_KEYS`/an inline literal tuple is invisible to it, and neither routes nor utils is coverage-gated (`pyproject.toml:30-34` omits `buster_ripper/routes/*`). The finding's own counterexample (a credential forwarded to whatever `--upstream` names) can remain true on two of the three request boundaries. | **No.** Nothing requires the witness set to cover all five sites or all three routes, and no clause forbids a differently-named filter copy. | **AMBIGUOUS** |
| 5 | `F-4` done-when (3) | "the empty-`Bearer` **witness still passes**" presupposes an existing test. There is none: `grep -rni "bearer\|authorization" tests/` → **zero matches**; the only in-code statement of the contract is the comment at `config.py:91` / `routes/chat.py:104`. So the clause is either vacuously satisfied (Forward confirms by reading and marks done) or it silently becomes "author a new test" — and the packet's stated *purpose* for the clause ("so the eval contract survives") ends up with no gate either way, since the route that carries the rule is coverage-omitted. | **No** — the premise itself is false, and it is a done-when clause, so Forward cannot satisfy it as written. | **AMBIGUOUS** |
| 6 | `G-1` exception surface (**goal-shape**) | Done-when reads "the `except` is **scoped to** the section before `await response.prepare(request)`" — most naturally the *existing* `except aiohttp.ClientError` clause moved up, with a post-commit log-and-return. Then every **non-**`ClientError` post-commit failure still raises out, and raising out reproduces the same defect through aiohttp itself: `web_protocol._handle_request:518-522` catches it → `handle_error(request, 500, exc)` builds a **fresh** `Response` → `finish_response` calls `resp.prepare(request)`, which the packet's own verified mechanism (`web_response.py:455-461`) does **not** short-circuit. Witness, upstream-controlled: `messages.py:161-165` does `evt.get("message", {}).get("usage", {}).get("input_tokens", 0)` on a payload `find_message_start` only type-checked for `type == "message_start"` (`utils.py:84`); `data: {"type":"message_start","message":"x"}` → `AttributeError` **after** `prepare()` at `:154` → a 500 head written onto the committed SSE body. The wave's test injects only "a payload error", so acceptance passes with the hole open, while the block's invariant candidate claims the whole class is closed ("a failure after `prepare()` is logged and the stream abandoned"). | **No.** Required behavior says "**a failure** after commit", unqualified; the done-when's witness is `ClientPayloadError`-specific. `ClientPayloadError ⊂ ClientError` is verified (`client_exceptions.py:297`), and aiohttp's own timeouts are `ServerTimeoutError(ServerConnectionError, asyncio.TimeoutError)` → `ClientError`, so *those* are covered — the gap is the non-`ClientError` remainder, which the plan never enumerates either way. | **AMBIGUOUS** |
| 7 | `F-4` referent: the dead helpers are the fix | "build a new shared filter helper in `utils`" (creation bias) — the packet's own §Evidence line forecloses it. | §Member F-4: "`forward_headers`/`response_headers` have zero production callers (only the stale `chat.py:14` import) — **the dead helpers *are* the fix**"; verified in-tree (`utils.py:18-33`, zero callers outside the stale import). | Excluded |
| 8 | `F-4` referent: `chat.py:155-164` | Reading the error-branch `Content-Type` copy + the SSE header block as a missed response filter. | §Correction carried to Forward: "`chat.py:155-164` is **not** a filter… The filter sites are **five**… do not 'fix' `chat.py:158-164` into a filter." (Same reasoning, uncited for `messages.py` — see row 1.) | Excluded |
| 9 | `T013` title noun "base" | "base" = the git base branch: Forward diffs against `origin/main` and restores *that* code. | §Member F-4: "Base contract confirmed **without git**, from the tracked pre-refactor copy in the tree: `buster_ripper_legacy.py:178`…`:626`…`:635`". Independently verified: `git ls-tree --name-only origin/main` → `buster_ripper.py` only, no package at all, so the main-branch reading has nothing to restore. | Excluded |
| 10 | `F-4` closure: `response_headers` extended to drop `content-length` | Read as a behavior change needing its own witness, or as a change to `forward_headers` too. | §Selected closure states the reason — "aiohttp recomputes it; this **preserves** the branch's current observable behavior at those two sites" — and both sites already drop it (`messages.py:259`, `passthrough.py:42`), verified. | Excluded |
| 11 | Wave sequencing `F-4 → G-1 → G-2` | Reading the order as advisory and doing `G-1` first, drifting `F-4`'s cited line numbers. | §Wave: "**ordered members**: F-4 → G-1 → G-2, **one commit each** so each restored behavior is independently attributable"; the line-number hazard is itself answered by §Assumption impact ("prefer in-place replacements **or** re-verify that citation"). | Excluded |
| 12 | `G-2` treatment choice | Sensing a forced design call: delete the rows vs relabel them. | §Member G-2 Done when: "**Either** treatment satisfies the same observable outcome, so the choice of wording is **Forward's**; a still-false 'Complete' is not admissible." | Excluded |
| 13 | `G-2` replay blind spot | Passing `grep -n "convert/" REFACTORING_PLAN.md` while the document still asserts the conversion *shipped*: `:5` "Phase 1 (aiohttp migration + image handling + **Anthropic→OpenAI conversion**) committed and documented" and `:7` "…including image handling and **request conversion utilities**" carry no `convert/` token. | Partly: Done-when clause 1 covers "**the `Status:` line**" (which `:5` sits directly under) and Required behavior is "the repository ships **no durable claim** that a module exists when it does not". Verified repo-wide: the only shipped doc asserting a phantom module is this file (`README.md`, `docs/` → no `convert` hits; the other matches are `.sessions/codify/01.00…` archives, not shipped). | Discretion footnote |
| 14 | `G-1` post-commit terminal action | "`returns`/`aborts`" is a slash over two behaviors: hand back the committed `StreamResponse` (clean truncated stream) vs force-close the connection. Both satisfy every written clause and the raw-bytes replay. | Not excluded, but not material: one line, no invariant touched, and "does not raise out" already rules out the third option (propagate, which is what `chat.py:165-169` still does). | Discretion footnote |
| 15 | `G-1` route coverage | Fixing `chat.py:_relay_chat_stream` "as part of the family" (it has the same post-commit raise-out), widening the wave beyond 057 G-1's cited site; or the reverse — treating the wave as closing the *class* repo-wide, which the invariant candidate's universal wording ("**No handler** returns a fresh response object…") invites. | The packet cites one site (`messages.py:179`) and the family's prior half is recorded as already closed. Coverage question, not a binding conflict; no test observes chat's post-commit path either way. | Discretion footnote |
| 16 | `T013` handle namespace | `G-1`/`G-2` collide with the Goal Definition Matrix's `G-001`…`G-008`, in a file where both are live (`T013` row: "Closure packets: 057 F-4…, G-1…, G-2"; §Task List "Gates no milestone"). A reader could try to reconcile "member G-1" with goal `G-001` (bulk-flush write count) and look for a dump-side edit. | Mostly: every wave row prefixes **057** ("057 F-4 + G-1 + G-2"), and `T013` is stated as outside the approved plan. | Discretion footnote |
| 17 | `T013` duplicate blueprint | `plan-summary.md` §Implementation Plan Summary's `<details>` restatement of the `T013` done-when has **drifted** from the `handoff-context.md` blueprint it summarises: it says "`TE`/`Upgrade`/`Keep-Alive`/`Proxy-Authorization` are each witnessed absent **upstream** in both directions" — wrong for the response half, where the observation surface is the client — and it drops the five-site count and the `chat.py` carve-out. Forward receives both renderings and must reconcile them; either can win. | Not a misreading to exclude — it is a redundant rendering of one blueprint element. Per the restroking rule the repair is to keep the handoff block as the single blueprint and point the summary prose at it, **not** to reword both into agreement. | Discretion footnote (structural) |

### Materiality notes on the six blocking rows

Rows 1, 2 and 3 all fork **user-visible downstream/upstream behavior** (the SSE response header set; whether
`/v1/messages` strips an empty `Bearer`), none of it observed by any gate the session owns: no test drives
`/v1/messages` with `"stream": true` (the only two stream tests hit `/v1/chat/completions`,
`tests/test_routes_integration.py:152,:170`), nothing pins `Connection` or `X-Accel-Buffering` anywhere in
`tests/`, and `QAP-MS004` posts `turn_body(...)` with **no** `stream` key (`tests/qa/harness.py:94-102`), so it
exercises request filtering and the *non-streaming* response filter only. Row 4 and row 6 are the goal-shape
witnesses — both let every `Done when` clause pass with the required behavior unmet. Row 5 cites an artifact that
does not exist. Rows 1-4 and 6 need a binding; row 5 needs a correction, not a choice.

### Coverage and reuse

- Attacked all three members' Required-behavior + Done-when clauses, the wave-level ordering/rider text, both
  closure-packet corrections, the Anchor Map's silence on these referents, and `qa-plan.md`'s whole gate matrix for
  header coverage (there is none — consistent with §Current Status's "no red pass, no invariant and no coverage
  gate watched them").
- Prior attack coverage reused: none. `iterations/039-forward-red.md` (an earlier, differently-scoped red pass) was
  listed by name but not opened — the
  recorded §Task List entry for `T013` is `pending` and no earlier pass covered this surface, so nothing here is a
  re-attack of a repaired row.
- **Not attacked, deliberately:** `T014`'s image-slice wave (out of this dispatch's scope), and the factual
  premises of the packets themselves (aiohttp subclassing, `prepare()` short-circuit, the `pyproject` omit list) —
  those are the assumption gate's, dispatched next; I re-verified only the premises a blocking row leans on.
- Two rows (3, 6) were expansions: the first attempt at each exposed a load-bearing decision, not a wording nit.
  Budget was not exhausted, so no row here is a HOLDS-by-starvation.

### Routing

6 AMBIGUOUS → **do not authorize Forward.** Every row goes to **Decide in `red-findings` mode** for disposition
(reject / force a repair / frame a fork / declare a plan break), then Interview applies Decide's dispositions.
Each row is one-line repairable in the `handoff-context.md` `T013` block — none requires replanning the wave, and
none touches `MS-004`'s APPROVED verdict or an accepted scope.

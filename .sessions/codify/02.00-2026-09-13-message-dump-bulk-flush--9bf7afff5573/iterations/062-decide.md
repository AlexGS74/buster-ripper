# 062 — decide (RED-FINDINGS mode — 060's six AMBIGUOUS rows, `T013` header-parity wave)

## Orchestrator Prompt (verbatim)

> You are a subagent spawned by another session.
> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
> Mode: red-findings
> Reports: .sessions/codify/02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/iterations/060-forward-red.md
>
> Execute the decide-interactive skill for this session. Arguments: $ARGUMENTS

## Mode and inputs read

`Mode: red-findings`. Read the named Red report (`060-forward-red.md`), the forward-visible artifacts it attacks
(`handoff-context.md` §Planned Task Context → `T013` block `:242-369`, §Current Status; `plan-summary.md` §Task List
`T013` row `:366` + its `<details>` restatement `:389-407`, §Anchor Map `:108-134`, §Reuse Evidence), and only the
source needed to verify its rows: `buster_ripper/utils.py`, `routes/{messages,chat,passthrough}.py`,
`buster_ripper_legacy.py`, `server.py`, `tests/`, and the installed aiohttp
(`web_protocol.py`, `web_response.py`, `client_exceptions.py`). `codify-deterministic-closure` applied to every row
before its disposition. No unrelated plan choice reopened; no settled decision re-decided.

**Ownership:** nothing below is *applied* by this pass. Repairs are authorized exactly and handed to **Authority
Registrar** (the sole writer for plan repairs and invariant-registry changes). Only this report was written; no
`plan-summary.md` edit, no invariant edit, no Fork reserved (zero `user-fork` rows → `decisions.md` and its handoff
index untouched).

## Disposition table

| # | Surface | Disposition | Closure | Citation / repair | Reason (one line) |
|---|---|---|---|---|---|
| 1 | `F-4` site boundary — `messages.py:145-153` read as a 4th response filter site | **invalid** | n/a | §Member F-4 Evidence lists the response sites **by line** (`messages.py:257-260`, `passthrough.py:39-43`) and §Correction fixes the set at "five (three request, two response)"; verified: the streaming path never reads `resp.headers` at all (`messages.py:144-177` — the four keys at `:147-152` are literals) | The alternate reading is falsified by source, not merely unmentioned — there is no relayed header set there to filter, so the `chat.py` carve-out reason ("the proxy's own response, not a relayed upstream header set") applies to it by its own terms |
| 2 | `F-4` mechanism — central `@web.middleware` instead of per-site substitution | **deterministic-repair** | absorbed | R-A | The selected closure already names per-site substitution through the two existing helpers, and a middleware is both unauthorized-new-artifact and insufficient (it cannot touch a header set committed inside the handler at `messages.py:154` / `web_response.py:454-461`, while it does reach every other response in the app) |
| 3 | `F-4` `_prepare_headers` — empty-`Bearer` strip promoted into shared `utils.forward_headers` | **deterministic-repair** | absorbed | R-B | "Pre-refactor proxy semantics" is the required outcome and base is provably asymmetric — shared `_forward_headers` (`buster_ripper_legacy.py:621`, called at `:656`/`:812`/`:901`) has **no** Bearer rule while the strip lives inside the chat handler only (`:838-841`) — so promotion is a behavior change at two routes the finding never touches |
| 4 | `F-4` acceptance quantifier — done-when (1) token grep, (2) "per direction" = 2 tests vs "every boundary" | **deterministic-repair** | absorbed | R-C | Required behavior is site-universal and no clause makes the witness set or the check rename-proof; the smallest gate that cannot pass with a surviving copy is a receiver-level grep plus one witness per enumerated site |
| 5 | `F-4` done-when (3) — "the empty-`Bearer` witness still passes" | **deterministic-repair** | absorbed | R-D | The premise is false (`grep -rni --include='*.py' "bearer\|authorization" tests/` → zero matches; the only hit is a stale `test_chat_routes` `.pyc` from the file deleted in `1737663`), and a vacuous clause leaves the eval contract ungated — a correction with one admissible form, not a choice |
| 6 | `G-1` exception surface — non-`ClientError` post-commit failures still raise out | **deterministic-repair** | absorbed | R-E | Required behavior is unqualified ("a failure after commit") and the raise-out path provably reproduces the same defect through aiohttp (`web_protocol.py:522-524` → `handle_error(…, 500, exc)` → `finish_response` → `:698` `resp.prepare`), reachable from an upstream-controlled payload (`utils.py:84` type-checks only `type == "message_start"`, so `messages.py:161-165` raises `AttributeError` after `prepare()` at `:154`); a catch-all after commit is the only remediation that closes the class |

`DECIDE-RED` arithmetic: **6 rows | 1 invalid | 5 deterministic | 0 user forks | 0 plan breaks.**

## Closure packets

### Packet 1 — Row 2 (`F-4` mechanism)

```text
CLOSURE: absorbed
Finding: 060 row 2 (handoff-context.md:308-311 §Selected closure × :312-317 done-when)
Required outcome: the full RFC 9110 §7.6.1 set is dropped in both directions at the five cited relay sites,
  from one shared implementation, with no route-local copy — and no change to any header set the proxy authors.
Authority:
  - §Selected closure names the mechanism: "route every filter site through utils.forward_headers /
    utils.response_headers" (handoff-context.md:308-311) — approved closure packet, disposition absorbed, DC-F-4-r1
  - plan-summary.md:134 — "The `Mode: NEW` rows above remain the only authorization"; §Anchor Map :112-131 binds
    only the dump/TLS cluster, so `T013` creates no new artifact
  - Verified: _stream_single_request commits headers inside the handler (routes/messages.py:154) and
    prepare() short-circuits once _payload_writer is set (web_response.py:454-461)
Admissible candidates:
  (a) in-place call-site substitution at the five cited sites, reusing the two existing helpers
      (utils.py:18-33 — zero production callers today, verified)
Rejected candidates:
  - a new @web.middleware in server.py filtering every response -> (i) creates an artifact no Mode: NEW row
    authorizes; (ii) insufficient: for the streaming path its header edits land after prepare() committed them,
    so it does not perform the required outcome there, while (iii) it does alter every non-streaming and error
    response app-wide (a wider blast radius than the five sites, incl. handle_messages' 400 path at messages.py:29);
    (iv) it deletes the two response sites rather than routing them, which is exactly the reading row 1 shows to be
    harmful. Strictly larger, concretely riskier, and it does not accomplish the outcome — closure steps 3 and 4
  - user-fork (middleware vs per-site to the user) -> not two materially different admissible outcomes: one is
    inadmissible. A wiring-shape difference alone is never a different user-visible outcome (closure step 8 /
    mechanism-collapse rule)
Selected closure: (a), with the mechanism stated as a prohibition so it cannot be re-forked (R-A).
Risk shape: unchanged — fewer moving parts than the rejected variant; no new referent, no gate moves.
Deferred additive work: none
Owning route: Authority Registrar (plan repair R-A) -> Forward T013
Question required: no
```

### Packet 2 — Row 3 (`F-4` empty-`Bearer` locality)

```text
CLOSURE: absorbed
Finding: 060 row 3 (handoff-context.md:310-311 "preserve _prepare_headers' empty-Bearer deletion")
Required outcome: the lm-eval empty-"Bearer " contract keeps working at /v1/chat/completions and nothing changes
  at the boundaries the finding does not touch.
Authority:
  - §Member F-4 Required behavior: "pre-refactor proxy semantics" (handoff-context.md:306-307)
  - Verified base asymmetry: buster_ripper_legacy.py:621 defines the shared _forward_headers with no Bearer rule
    and all three request boundaries call it (:656 /v1/messages, :812 chat, :901 catch-all), while the strip lives
    only inside the chat handler (:838-841). Current tree mirrors that locality (routes/chat.py:104-109).
  - §Selected closure scopes the change to the five filter sites, none of which is an auth rule
Admissible candidates:
  (a) utils.forward_headers gains hop-by-hop semantics only; the empty-Bearer deletion stays route-local in
      routes/chat.py:_prepare_headers
Rejected candidates:
  - promote the deletion into utils.forward_headers -> forwards a new behavior to /v1/messages and the catch-all
    (they would start dropping an empty Authorization), contradicting the required "pre-refactor" semantics and
    broadening scope beyond the smaller complete remediation (closure steps 3 and 4)
  - delete the route-local strip and rely on upstream tolerance -> knowingly breaks the eval contract
  - user-fork (shared vs route-local) -> the required outcome is settled by the base contract the packet names;
    only (a) preserves it, so no unranked pair survives (CLOSURE is not `question`)
Selected closure: (a), bound explicitly (R-B) and gated by the witness R-D authors.
Risk shape: unchanged — the repair states where an existing rule stays; it moves no behavior.
Deferred additive work: none
Owning route: Authority Registrar (R-B, R-D) -> Forward T013
Question required: no
```

### Packet 3 — Row 4 (`F-4` acceptance quantifier)

```text
CLOSURE: absorbed
Finding: 060 row 4 (handoff-context.md:312-314 done-when (1) and (2))
Required outcome: the gate is falsifiable for the claim actually made — "every header set forwarded across a
  proxy boundary … at every route", with no route-local copy of the policy.
Authority:
  - §Member F-4 Required behavior + invariant candidate (handoff-context.md:297-307) — site-universal
  - The five sites are enumerated by line in §Evidence (:283-285), so a per-site witness is bounded and known
  - Verified receiver-level exhaustiveness: `grep -rn "headers.items()" buster_ripper/` today returns exactly the
    five route comprehensions (chat.py:102, messages.py:63, messages.py:258, passthrough.py:16, passthrough.py:41)
    plus the two helpers (utils.py:22, :31) — no sixth site, and any renamed or inline-literal copy still reads a
    `.headers` mapping, so the receiver grep survives renaming in a way `skip_keys|_HOP_BY_HOP` does not
  - No coverage gate watches either file (pyproject.toml:30-34 omits buster_ripper/routes/*), so the witness set is
    the only defense
Admissible candidates:
  (a) bind the witness set to the five enumerated sites and make the structural check receiver-level (R-C)
Rejected candidates:
  - leave "per direction" as written -> two tests can pass with an abbreviated copy surviving at chat.py or
    passthrough.py, i.e. the verified counterexample (a credential reaching --upstream) stays true on two of three
    request boundaries: preserves the defect class (closure step 3)
  - add a coverage gate for buster_ripper/routes/* -> re-opens the session's settled pyproject omit list and
    broadens cumulative risk well beyond the smaller complete remediation
  - user-fork (2 tests vs 5) -> test count is not a user-visible outcome; one admissible remediation remains
Selected closure: (a).
Risk shape: unchanged for production code; the wave's test authoring grows from 2 cases to 5 site witnesses —
  still within one member's increment, and required to make the done-when satisfiable at all.
Deferred additive work: none
Owning route: Authority Registrar (R-C) -> Forward T013
Question required: no
```

### Packet 4 — Row 5 (`F-4` done-when (3) false premise)

```text
CLOSURE: absorbed
Finding: 060 row 5 (handoff-context.md:315 "the empty-Bearer witness still passes")
Required outcome: the eval contract ("strip empty 'Bearer '" for lm-eval) is witnessed by a gate, not by a comment.
Authority: verified absence — `grep -rni --include='*.py' "bearer|authorization" tests/` → zero matches; the sole
  match anywhere under tests/ is `tests/__pycache__/test_chat_routes.cpython-314-pytest-9.0.2.pyc`, whose source was
  deleted in 1737663 ("test: remove over-engineered tests"). The only live statements of the contract are comments
  (config.py:91, routes/chat.py:104).
Admissible candidates:
  (a) replace the clause with an explicit authoring instruction: a focused test with the three cases in R-D
Rejected candidates:
  - keep the clause as written -> unsatisfiable as stated (there is no witness to "still pass"), so it resolves to
    Forward confirming by reading and marking done: a required gate left vacuous (closure step 3)
  - drop the clause -> knowingly leaves the eval contract ungated on a coverage-omitted route
  - point the clause at config.py:91 / chat.py:104 comments -> a comment is not a witness
Selected closure: (a), merged with R-B's base-parity half so one test gates both directions of the auth rule.
Risk shape: unchanged — additive test authoring inside a member that already owes tests; no production behavior moves.
Deferred additive work: none
Owning route: Authority Registrar (R-D) -> Forward T013
Question required: no
```

### Packet 5 — Row 6 (`G-1` exception surface)

```text
CLOSURE: absorbed
Finding: 060 row 6 (handoff-context.md:344-349 Required behavior vs done-when)
Required outcome: "a failure after commit aborts the stream; the client is never handed a second response head"
  — the sentence is unqualified, and the invariant candidate claims the whole class is closed.
Authority:
  - §Member G-1 Required behavior (handoff-context.md:344) and invariant candidate (:338-339) — both class-wide
  - Verified raise-out mechanism: web_protocol.py:522-524 `except Exception` -> handle_error(request, 500, exc)
    -> finish_response -> :698 `resp.prepare(request)`; web_response.py:454-461 short-circuits only when
    _payload_writer is already set, so the fresh object writes a second head onto the committed stream
  - Verified upstream-controlled trigger: utils.py:84 gates only `evt.get("type") == "message_start"`, so
    `data: {"type":"message_start","message":"x"}` makes messages.py:161-165 raise AttributeError after
    prepare() at :154 — aiohttp then performs the exact defect the finding is about
  - Verified coverage of the ClientError half: client_exceptions.py:297 ClientPayloadError(ClientError)
  - asyncio.CancelledError must keep propagating (web_protocol.py:516-517 re-raises it)
Admissible candidates:
  (a) split the single try by commit boundary: the pre-commit section keeps `except aiohttp.ClientError`
      (returns the 502), the post-commit relay section gets its own `except Exception` that logs and abandons
      without constructing any replacement response; witness set covers both failure classes
Rejected candidates:
  - move the existing `except aiohttp.ClientError` above prepare() and stop there -> the verified AttributeError
    path still raises out and still gets a second head: preserves the defect (closure step 3)
  - guard the payload shape at messages.py:161-165 instead of a catch-all -> closes one instance, not the class
    (e.g. OSError/ConnectionResetError from response.write at :174); a required gate stays red in principle
  - catch BaseException -> swallows CancelledError, breaking client-disconnect and shutdown propagation
  - user-fork (how wide to catch) -> the plan's own Required behavior already ranks this: any scope narrower
    than "a failure" leaves an admissible-outcome-free remainder, so no unranked pair exists
Selected closure: (a), written as a case table (R-E) per R10.
Risk shape: unchanged in kind (one handler's exception structure), increased only in completeness — the
  post-commit path already returns nothing today on the ClientError half it wrongly handles.
Deferred additive work: none. chat.py's post-commit raise-out (:165-169) stays out of scope (Red row 15's
  discretion footnote) — this repair bounds `messages.py`, not the class repo-wide.
Owning route: Authority Registrar (R-E) -> Forward T013
Question required: no
```

## Repairs authorized for Authority Registrar (apply exactly these, nothing wider)

All five land in `handoff-context.md` §Planned Task Context → the `T013` `<details>` block (`:242-369`). Line
numbers below are the pre-repair positions.

**R-A — §Member F-4, after §Selected closure (`:308-311`).** Append:

> **Mechanism is fixed, not open.** The five cited sites are edited **in place** to call the two existing helpers
> (`utils.py:18-33`); `T013` creates no new referent. No new central filter — no `@web.middleware`, no new module,
> class or registry — because `plan-summary.md` §Anchor Map :134 ("The `Mode: NEW` rows above remain the only
> authorization") is not widened by this task, and a response middleware could not perform the required outcome
> anyway: `_stream_single_request` commits its headers inside the handler (`routes/messages.py:154`) and
> `prepare()` short-circuits once `_payload_writer` exists (`web_response.py:454-461`), so a middleware's edits are
> inert for the committed SSE response while they *do* reach every non-streaming and error response app-wide.

**R-B — §Member F-4, §Selected closure's last clause (`:310-311`).** Replace "and preserve `_prepare_headers`'
empty-`Bearer` deletion so the eval contract (`strip empty "Bearer "` for lm-eval) survives." with:

> and preserve `_prepare_headers`' empty-`Bearer` deletion so the eval contract (`strip empty "Bearer "` for
> lm-eval) survives — **route-local, by base precedent**: `utils.forward_headers` gains **no** authorization
> semantics and the deletion stays inside `routes/chat.py:_prepare_headers` (`:104-109`). Base is asymmetric here,
> verified: `buster_ripper_legacy.py:621` defines the shared `_forward_headers` with no Bearer rule and every
> request boundary calls it (`:656` `/v1/messages`, `:812` chat, `:901` catch-all), while the strip lives only
> inside the chat handler (`:838-841`). Consequence to preserve: `/v1/messages` and the catch-all keep forwarding an
> empty `Authorization: Bearer ` verbatim.

**R-C — §Member F-4, done-when clauses (1) and (2) (`:312-314`).** Replace with:

> (1) **rename-proof site check** — `grep -rn "headers.items()" buster_ripper/routes/` returns **zero** matches
> (all five comprehensions — `chat.py:102`, `messages.py:63`, `messages.py:258`, `passthrough.py:16`,
> `passthrough.py:41` — become calls into `utils`; the helpers' own `utils.py:22`/`:31` are the only `.items()`
> filters left in the package, and this receiver-level pattern catches a copy renamed `DROP_KEYS` or inlined as a
> literal tuple, which a `skip_keys|_HOP_BY_HOP` token grep does not); (2) a focused test **per enumerated site**
> (five witnesses, parametrized over the three request and the two response sites) asserts `TE`, `Upgrade`,
> `Keep-Alive`, `Proxy-Authorization`, `Proxy-Authenticate`, `Trailers` are absent from what reaches upstream and
> from what reaches the client, while `Authorization` and `Anthropic-Version`-style end-to-end headers survive;

**R-D — §Member F-4, done-when clause (3) (`:315`).** Replace "the empty-`Bearer` witness still passes;" with:

> a **new** empty-`Bearer` test passes — none exists today (`grep -rni --include='*.py' "bearer|authorization"
> tests/` → zero matches; the only hit is the stale `tests/__pycache__/test_chat_routes.cpython-314-pytest-9.0.2.pyc`
> of a file deleted in `1737663`, and the contract's only live statement is comments at `config.py:91` /
> `routes/chat.py:104`), so this clause was vacuous and is now an authoring instruction with three cases: chat
> deletes `Authorization: "Bearer "` and `Authorization: "Bearer"`; chat keeps a real token; `/v1/messages` and the
> catch-all do **not** delete an empty `Bearer` (the `R-B` base-parity half);

**R-E — §Member G-1, done-when (`:347-349`).** Replace the whole clause with:

> **The section boundary is `await response.prepare(request)` at `messages.py:154`, and both sides of it need a
> handler today** — the single `except aiohttp.ClientError` at `:179` spans pre-commit (`:129-142`) and post-commit
> (`:144-177`), so splitting it means nesting a second `try` around the relay loop inside the `async with`. Case
> table for what may follow commit:
>
> | Post-commit failure | Today | Required |
> |---|---|---|
> | `aiohttp.ClientError` (incl. `ClientPayloadError`, `client_exceptions.py:297`; aiohttp timeouts resolve to `ClientError`) | returns a fresh `web.json_response(502)` → `prepare()` does not short-circuit → second head | log + abandon; **no replacement response object** |
> | non-`ClientError` from upstream-controlled payload — `data: {"type":"message_start","message":"x"}` → `AttributeError` at `:161-165` (`utils.py:84` type-checks only `type == "message_start"`) | raises out → `web_protocol.py:522-524` `except Exception` → `handle_error(…, 500, exc)` → `finish_response` → `:698` `resp.prepare(request)` → **the same second-head defect** | same as above |
> | other post-commit `Exception` (e.g. `OSError`/`ConnectionResetError` from `response.write` at `:174`) | raises out | same as above |
> | `asyncio.CancelledError` (client disconnect, shutdown) | propagates | **must keep propagating** — `except Exception` only, never `BaseException` (`web_protocol.py:516-517`) |
>
> Done when: the pre-commit section keeps `except aiohttp.ClientError` returning the 502, the post-commit relay
> section is wrapped in its own `except Exception` that logs and abandons the stream without constructing a
> replacement response and does not re-raise, and the focused test injects **both** a post-commit
> `ClientPayloadError` **and** a post-commit non-`ClientError` (the `message_start` payload above is
> upstream-controlled and needs no monkeypatch) and asserts exactly one status line reaches the client and that the
> handler does not raise out. A payload-shape guard at `:161-165` is not required and must not be substituted for
> the catch-all: it closes one instance, not the class.

**One structural note to carry with these (Red row 17, not an AMBIGUOUS row, so no disposition here):** once
R-C/R-D/R-E land, `plan-summary.md` §Implementation Plan Summary's `<details>` restatement (`:396-404`) drifts
further from the block it summarises. Row 17's restroking rule applies — keep the handoff block as the single
blueprint and point the summary prose at it; do **not** reword both into agreement.

Rows 13-16 of Red's table are discretion footnotes and need no round, per its own §Routing and §Current Status.

## Deterministic Closure Ledger entries for this pass

| # | Finding | Closure | Selected remediation | Plan-surface change | Question |
|---|---|---|---|---|---|
| 1 | 060 row 1 (`messages.py:145-153` as a 4th response site) | — (no remediation owed) | disposition `invalid`; citation recorded above | no | no |
| 2 | 060 row 2 (middleware vs per-site) | absorbed | R-A | no | no |
| 3 | 060 row 3 (Bearer locality) | absorbed | R-B (+ gated by R-D) | no | no |
| 4 | 060 row 4 (witness quantifier) | absorbed | R-C | yes — `T013`'s own done-when grows its test set; no milestone, no `MS-*`/`QAP-*` gate moves | no |
| 5 | 060 row 5 (false premise: no Bearer witness) | absorbed | R-D | same as above | no |
| 6 | 060 row 6 (`G-1` exception class) | absorbed | R-E | same as above | no |

Suggested keys for Registrar: `DC-F-4-r2` (rows 2-5, one member) and `DC-G-1-r2` (row 6), superseded-by-nothing —
both append to §Deterministic Closure Ledger — 2026-10-05 routing without editing `DC-F-4-r1`/`DC-G-1-r1`, whose
`absorbed` dispositions stand.

No invariant-registry change is owed: the two invariant candidates (`:297-300`, `:338-339`) remain
non-authoritative Interview material, exactly as the block records them, and these repairs make them *harder* to
violate rather than restating them.

## Compaction Decision

- Prior same-phase files reviewed: `022-decide.md`, `025-decide.md`, `027-decide.md`, `029-decide.md`,
  `040-decide.md` (decide reports of the current planning cycle; earlier ones belong to closed cycles).
- Superseded: **none.**
- Not superseded: **all six.** Each dispositions a different slice — 022 the completeness repair, 025/027 the
  024/026 goal and R16 rows, 029 028's T006 guard shape, 040 the scoped-plan-blocker pass. None covers `T013`'s
  header-parity wave, which has never had a Decide pass.
- Compaction performed: **no.**
- If no: `red-findings` mode forbids compacting prior Decide reports, and no prior report holds an unresolved option
  or uncertainty that this pass now represents.

## After R-A…R-E land

`T013` Forward is still **not** authorized until `forward-red` is re-dispatched against the repaired block and
returns `0 AMBIGUOUS` (the 029 chain). `T014`'s five rows (`061-forward-red.md`) are a separate Decide dispatch;
nothing here disposes them, and the two waves share no file behavior (`routes/messages.py` sites are disjoint).

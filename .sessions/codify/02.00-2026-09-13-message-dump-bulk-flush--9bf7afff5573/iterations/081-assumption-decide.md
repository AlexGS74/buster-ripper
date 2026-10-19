# 081 · Assumption Decide — CR3 wave run-40 atomic non-holding routing (`S16`–`S19` members)

## Orchestrator Prompt (verbatim)

> You are a subagent spawned by another session.
> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
> Worktree root: /Users/alexhubitski/Projects/buster-ripper
> Mode: atomic non-holding routing (CR3 wave run 40)
> Assumptions directory: /Users/alexhubitski/Projects/buster-ripper/.sessions/codify/02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/
> Registry root: /Users/alexhubitski/Projects/buster-ripper/invariants/
>
> Four rows sit at `Routing: pending` after the atomic promotion, all `pkg:`/`lang:` subjects belonging to the CR3 wave (`T018` body cap + stream deadline, `T019` normalize-before-keying + decode-total arms, `T020` the EMA witness rewrite). Route each exactly once; apply the Reach Gate where a finding depends on an input or state.
>
> Row **391** (`UNCLEAR`, aiohttp's request enforcing the limit configured on "the application serving the request") — the checker established the mechanism unambiguously and named the fork: the app that constructed the request, versus `request.app` = `match_info.current_app`, whose own value is inert under `add_subapp`. Its verdict says the repair is a rewording, not more source.
>
> Row **404** (`UNCLEAR`, `bytes.decode(errors="replace")` leaves valid bytes unchanged) — the fork is "same characters as strict would yield" versus "same bytes", plus a per-byte-versus-in-context reading of "valid under the codec"; the verdict supplies rewording the same three pages settle.
>
> Row **399** (`DOES-NOT-HOLD`, published as `CL:aiohttp/client-response-text-unknown-charset-error`) — aiohttp suppresses its own `LookupError` inside `get_encoding()` and falls back to UTF-8 or the session's `fallback_charset_resolver`, so an unknown response charset is never loud. Read against row **401** (`HOLDS`: the **request** side, `web.Request.json()`, genuinely raises a bare `LookupError`), which means the two directions of CR3-F-04's tolerance are not symmetric, and the task blueprint as written assumed they were.
>
> Row **409** (`DOES-NOT-HOLD`, published as `CL:aiohttp/magicmock-async-context-manager` — subject `lang:python`) — a plain `MagicMock` **does** supply an awaitable `__aenter__`, because `_async_method_magics` is folded into the magic set and `_get_child_mock` routes async magic to `AsyncMock` children; the real failure cases are bare `Mock()`/`NonCallableMock()` and a `spec=` lacking the dunder. The promotion pass flagged explicitly that `T020`'s witness rewrite was drafted against this falsified premise, so decide what `T020`'s done-when must now assert for the witness to be capable of going red for the right reason.
>
> Also disposition this record defect, which the promotion pass could not clear: rows **402** and **403** returned `VERDICT: HOLDS` with fingerprinted locators but line 2 reads `PROMOTION: ELIGIBLE — {annotation}` rather than the exact required token, so both stayed ineligible while their neighbors published. Six verdicts across this session carry an annotated header and none has ever been promoted. Route the header re-issue to its owning lane and name whether that lane is the checker, the collector, or the promoter — do not have it settled by re-checking sound evidence on unchanged bytes.
>
> Seam state to expect when you finish: `S16`–`S19` are `new` and cannot be composed while a member is non-`HOLD`, so the wave's Forward act waits on your dispositions. Constraint from this session's standing scope ruling: third-party subjects only — no repo row, no repo seam, and `S1`–`S14` stay terminal. `FK-011` remains the only open Fork; a question is admissible only where materially different outcomes remain unranked after the base-parity and `G-003` authorities are applied.

Result: **ASSUMPTIONS: 5 routed | 2 absorbed | 0 questions | 0 breaks | 1 malformed | 2 accepted | 0 capped**

The four atomic rows plus the one record defect the brief asked me to disposition. Answer to the two questions the
brief posed: **`T020`'s done-when must now assert the loop's computed value per tick, one awaited-channel
precondition, and a named-failing-assertion mutation replay, and must forbid the double shapes that pass on an inert
peer** (§T020 below); **the header re-issue belongs to the checker** — the verdict lane, the only role that writes a
verdict file's line 2 — not the collector and not the promoter, and it is a re-issue, not a re-check (§402/403).

Reads: `claims.md`'s four `pending` rows + their verdict artifacts, `seams.md` rows `S16`–`S19`, the four sibling
verdicts the routed rows are read against (398/400/401/402/403 and 408/410/411/126), `../handoff-context.md`
(§Planned Task Context `T018`–`T020`, `DC-CR3-F-01-r1`…`-r5`, §Accepted risks, §Current Status), `../plan-summary.md`'s
three wave rows, `../quality-backlog.md`'s `QB:assumption-registry/*` class, the registry `## Claim Index`/`## Handle
Index`/§Source Sessions and both `invariants/buster_ripper/*/invariants.md`, and the cited production sites
(`routes/messages.py:28-31,136-164,264-297`, `routes/count_tokens.py`, `compaction.py:34-57`,
`tests/test_compaction.py`, `server.py:119`). Repository state before the first write, through Git from the worktree
root: `git-commit:e204b18e4f82f618e03c808743361637e3b149d2`. No `.git/` path read; nothing under `invariants/` written.

---

## Row rows

### claim-391 — the body limit's owning application is ambiguous — **accepted** (inert wording)

The artifact is the rare good `UNCLEAR`: the mechanism is settled without residue (`_client_max_size` set from the
constructing app's `_make_request`, `web_request.py:161,198` → `web_app.py:510`, read only by `read()`/`post()`), and
what is unresolved is a **word in the claim** — "the application serving the request" — which could name the root app
(HOLDS) or `request.app` = `match_info.current_app` (DOES-NOT-HOLD for a sub-app, whose own value is inert because
routing happens after construction).

Reach First: **inert wording**. The fork cannot split an outcome on this tree — the package builds exactly one flat
`web.Application` (`server.py:119`, re-read) and `add_subapp` greps **zero** under `buster_ripper/` and `tests/`, so
under either reading the value that governs every request is the one `T018` sets to `0`; and the propagation the seam's
boundary value actually depends on is carried by rows 390/392–395, all `HOLD`, all published. The gap lives in a ledger
sentence, the repair is a rewording the verdict already supplies, and per Reach First that is `accepted` plus one doc
line — `QB:assumption-ledger/row-391-serving-app-wording` — with **no task, no Fork, no wave, no re-extraction, and
`S16` not held open**. Cross-reference **Neutral**. Routing `accepted-consequence`; State stays `UNCLEAR` — that cell is
the record of which wording was ambiguous, and no lane of this pass may copy a verdict token into it.

*Why not `absorbed` on a rewording task:* the closure gate's own ranking has nothing to select — both wordings are
true statements of different facts, and neither changes `T018`'s outcome. Routing it `absorbed` would invent a
dispatch for a sentence and put a ledger-wording edit into a code wave's touch-set.

### claim-404 — what `errors="replace"` preserves — **accepted** (inert wording)

Same class, one axis further: "unchanged" reads as *same characters as `errors='strict'`* or as *same bytes*, and
"valid under the codec" reads per byte or per complete sequence in context. The artifact pins the pair that the three
fingerprinted pages settle, and supplies the reworded sentence verbatim.

Reach First: **inert wording**, with an evidence-backed reason rather than a convenience: the character-reading is the
one `T019` step 2(a) stands on (the ASCII overflow text survives replacement at `messages.py:276`), and the byte-reading
is false of **every** `decode()` call — `decode()` returns `str`; no byte survives "unchanged" under any `errors` value
— so it is not a competing reading any consumer could be relying on, and the crossing axis (a lone UTF-8 lead byte *is*
an error and *is* replaced) is exactly what the pinned wording removes the ambiguity about. Rows 402/403 (`HOLD`) carry
the strict default and the U+FFFD marker the arm depends on. Disposition: `accepted`, doc line
`QB:assumption-ledger/row-404-replace-preservation-wording`, no task, no seam held open; `S18`'s open state is owed to
`claim-399`. Cross-reference **Neutral**. Routing `accepted-consequence`; State stays `UNCLEAR`.

### claim-399 — the response side suppresses its own `LookupError` — **absorbed** (`DC-claim-399-r1`)

Read against 401 the wave's decode story is **asymmetric**, and the blueprint did not know it:

| direction | site | unknown charset in `Content-Type` | authority |
|---|---|---|---|
| client → proxy | `messages.py:29`, `chat.py:30`, `count_tokens.py:19` (`Request.json()`) | **raises a bare `LookupError`** — real, uncaught today → proxy 500 | row 401 `HOLDS` / `CL:aiohttp/request-json-unknown-charset-error` |
| upstream → proxy | `messages.py:144` (`ClientResponse.text()`) | **never raises** — `get_encoding()` wraps `codecs.lookup` in `contextlib.suppress(LookupError, ValueError)` and falls back to `utf-8` (or `fallback_charset_resolver`, default `utf-8`); a raise appears only as `UnicodeDecodeError` when the body is undecodable **as the fallback** | row 399 `DOES-NOT-HOLD` / `CL:aiohttp/client-response-text-unknown-charset-error` |

The consequence is not that `T019`'s fix is wrong — it is that one of its proofs cannot exist. `CR3-F-04`'s
counterexample (2) offers "`Content-Type: application/json; charset=bogus`" as an alternative trigger for the
proxy-500; with a UTF-8-decodable body that input **does not raise at all**, so a fail-first witness written on it is
green before the fix and the plan's own done-when ("each red at base bytes") is unattainable at that site. Verified
against the shipped shape: `messages.py:143-148`'s `await resp.text()` sits inside `try:` at `:136` whose only arm is
`except aiohttp.ClientError` at `:164`, and `UnicodeDecodeError` is not a `ClientError` (`DC-CR3-F-04-r1`'s own re-derivation: the two are disjoint `ValueError`
subtrees; rows 68–72 carry the hierarchy the new catch lists stand on) — so the live
class at that site is the fallback decode's `UnicodeDecodeError`, exactly as 399 states.

```text
CLOSURE: absorbed
Finding: claim-399 (assumptions/verdicts/claim-399.md, read against claim-401.md)
Required outcome: no upstream answer 500s the proxy through a decode failure at messages.py:143-149, and every T019
  witness is capable of going red at pre-fix bytes
Authority: DC-CR3-F-04-r1's required outcome (unchanged) · row 401 HOLDS (request side) · row 400 HOLDS (read() escape
  hatch) · G-003 (proxy behavior on /v1/messages unchanged) · T019's own done-when fail-first clause · plan approval
  of the catch list
Admissible candidates: one
Rejected candidates: drop LookupError from the response arm (edits an approved catch list for a defensive clause 399
  itself names a live path to — the caller's explicit encoding= — with no behavior gain); keep counterexample (2) as
  written (defect-preserving in the proof lane: an unfalsifiable fail-first); make the response side re-raise on
  unknown charset (a new raising behavior no finding names, opposite to F-04's tolerance, and a G-003 deviation);
  open a new task/wave (the repair is criteria text on pending T019 inside an already-scheduled wave)
Selected closure: amend T019 step 2(b) — keep except (UnicodeDecodeError, LookupError) + the read()/errors="replace"
  fallback, re-aim the response-side fail-first at bytes undecodable under the fallback charset, strike the
  charset=bogus-raises-LookupError alternative, and add one witness pinning charset=bogus over a UTF-8-decodable error
  body as a NO-RAISE RELAY so the asymmetry is tested, not prose
Risk shape: unchanged — zero production bytes added here; arms still only become total where they 500'd
Deferred additive work: S18's CO: promotion waits on QB:assumption-registry/annotated-promotion-headers (checker
  lane; trigger: next legitimate touch of those verdict artifacts)
Owning route: Forward, T019 step 2, same wave and same single local Yield
Question required: no
```

Cross-reference **Neutral**: zero `active`/`wip` registry entries reach a charset arm (eight `planned` PIs are
dump-format/TLS-posture); the session-carried `EI:.../derived_after_normalize` binds step **1**, is untouched by this
amendment, and 080's **Aligned** classification for `CR3-F-02` neither extends to `claim-399` nor weakens. Non-authoritative
candidate for Interview: *"a proxy's decode arms are written per boundary direction, because aiohttp's client and
server halves of one charset failure raise different classes."* Not written, not promoted.

### claim-409 — the falsified mock premise and what the witness must now prove — **absorbed** (`DC-claim-409-r1`)

Reach First does not apply: this is false evidence, which the gate's own exclusion list covers (as `DC-CR3-F-05-r1`
recorded at 080). What the falsification changes is the **mechanics of the silence**. With `_async_method_magics`
folded into `MagicMock`'s magic set and `_get_child_mock` routing an async magic to an `AsyncMock` child,
`async with session.get(...) as resp` **succeeds** on a plain `MagicMock` — and binds `resp` to a newly auto-created
child, not the object the test configured. So `resp.status == 200` is False, `compaction.py:40-52` never runs,
`_kv_cache_usage` never moves, and `except Exception: pass` at `:55-56` absorbs every other fixture failure too —
including the two shapes that *do* raise (`async with Mock()`, and a `spec=`/`autospec` response double whose spec
prunes the dunder). "The double supports `async with`" has stopped being evidence of anything; the branch-skipping
path is now silent rather than loud, which is worse for a witness whose job is to go red.

That is the answer to the brief's question. `T020`'s done-when must now assert, and the block transmits it:

1. **per-tick value assertions** on `compaction._kv_cache_usage` — tick 1 = raw1 exactly (only reachable if `resp.status
   == 200` compared True against an `int` *and* the regex matched a `str` the double supplied, so it doubles as the
   branch-reachability proof), tick 2 = `(1-α)·raw1 + α·raw2` with the expectation written independently of the source
   line's operand order, tick 3 the three-distinct-raws bounce claim the test's name promises;
2. **one awaited-channel precondition** (`text()` awaited once per tick) so a wrong double reads as "the peer never
   delivered a 200 body" instead of masquerading as a production result;
3. **forbidden constructions**: bare `Mock()`/`NonCallableMock()` as the context manager, any `spec=`/`autospec`
   response double, and an unconfigured `MagicMock` where the response belongs;
4. **permitted construction**: the file's own sibling idiom (`tests/test_compaction.py:182-195`, `:229-245` —
   `mocker.AsyncMock()` response with `status`/`text`/`__aenter__`/`__aexit__` set, `mocker.Mock(spec=aiohttp.ClientSession)`
   session with `get` a plain `Mock` returning it), or a `MagicMock` whose `__aenter__.return_value` **is** the
   configured response;
5. **the mutation replay must go red on a value assertion, with the failing assertion named** — plus Yield's own
   negative control: forcing the double's `status` to a non-200 must also go red, which is the direct proof the witness
   observes production rather than its fixture;
6. 411 (`CL:pytest-asyncio/auto-mode-async-test-collection`, HOLDS) licenses `async def` witnesses with no marker;
   410 and 126 license reaching tick 2 without a real `KV_POLL_INTERVAL` sleep, inside the cancel-and-await arm;
7. 080's criteria unchanged: no missing line at `:49`, `-k ema` + coverage delta, `:373`'s F841 consumed, globals
   restored, 455+ @ ≥95.75%, LNCP not worsened, `compaction.py` bytes must-not-touch.

```text
CLOSURE: absorbed
Finding: claim-409 (assumptions/verdicts/claim-409.md)
Required outcome: the two test_ema_* witnesses fail when production's EMA arithmetic changes, and cannot pass on a
  double that never reached the branch
Authority: DC-CR3-F-05-r1's required outcome (unchanged) · codify-test-cost (a test must prove something) and its
  rejected-candidate list (no self-proving recomputation, no deletion, no doc line) · rows 408/410/411/126 HOLDS ·
  MS-003's coverage commitment on the certified 95.75% figure · T020 is tests-only per plan
Admissible candidates: one
Rejected candidates: assert call counts / "two ticks without raising" (defect-preserving — green under exactly the
  inert-double shape 409 makes possible); configure the double and cite the old premise as the reason (keeps a
  falsified assumption as the design rationale and hides the auto-child trap); a real aiohttp metrics peer (scope
  growth on a file-disjoint tests-only member, and it moves the witness off the branch it exists to cover); delete
  or document (rejected verbatim at DC-CR3-F-05-r1)
Selected closure: amend T020's done-when to the seven criteria above
Risk shape: unchanged — tests/test_compaction.py only, zero production bytes
Deferred additive work: none
Owning route: Forward, T020 step 1, same wave and same single local Yield
Question required: no
```

Cross-reference **Neutral**. Non-authoritative candidate for Interview: *"a witness of an async loop is evidence only
when it asserts the loop's computed state, never when it asserts that a double was usable."* Not written, not promoted.

### rows 402/403 (+ 129, 226, 227, 228) — annotated `PROMOTION:` headers — **malformed** (record defect; no ledger row by rule)

The propositions are sound, the locators are exact, the fingerprints are current, and the bytes have not moved: what is
malformed is one **machine-readable field**, not the claim. `assumption-promote` §Eligibility requires "line 2 is
exactly `PROMOTION: ELIGIBLE`" and states the promoter "never infers eligibility from prose"; `assumption-check`'s own
output template (`PROMOTION: ELIGIBLE | INELIGIBLE — {reason when ineligible}`) sanctions the trailing annotation only
on the ineligible arm, and its warning at :275-276 says the promotion pass reads that line mechanically. Six artifacts
carry the annotated form — `claim-129`, `claim-226`, `claim-227`, `claim-228`, `claim-402`, `claim-403` (`grep -l
"^PROMOTION: ELIGIBLE —" verdicts/*.md` → exactly those six) — and none has ever been promoted, which is one component
of why the seams that carry them have no `CO:` row either (`S7`/`S11` both `Permanent: —`) and it is the whole reason
`S18`'s next composition will be unpublishable until this is fixed: composition eligibility requires **every** member
claim to resolve to a permanent `CL:` handle.

**The owning lane is the checker** (`assumption-check`, the verdict lane). It is the only role that writes a verdict
file's line 2, and the promotion pass pointed at the same lane itself ("The one-line remedy sits with the verdict lane,
not here"). Not the **collector**: `forward-collect` owns `claims.md`/`seams.md` cells and, on its own run-40 record,
"wrote no verdict file" — it cannot re-issue a field it does not author. Not the **promoter**: its read is
mechanical by design and manufacturing eligibility from prose is the one act it is barred from; the run-40 pass wrote
no handle, no registry row and no `Permanent` cell for 402/403, which is the gate working, not a gap to close.

**Remedy is a re-issue, explicitly not a re-check.** Per file: bare token on line 2, annotation moved into the body,
no source re-consultation, no fingerprint recomputation, no blind re-dispatch — the consulted bytes are unchanged and
re-deriving sound evidence to fix a header would spend a check to make a typo. Rows 402/403 keep `HOLD`/`none` and
**neither Routing cell was written by this pass** (they were never `pending`; the defect is in the artifacts, which are
the checkers' lane and are committed by the dispatcher at the assumption gate). Durable at
`QB:assumption-registry/annotated-promotion-headers`, sibling of `QB:assumption-registry/run-31-verdict-fingerprints`
(same class: record thin, truth unmoved, verdict-lane remedy) — trigger: the next dispatch that legitimately touches
those artifacts, or immediately as a one-line edit per file. Nothing here blocks `T019`: the code lands with or
without the handle; only `S18`'s `CO:` publication waits. (Six rows carrying these artifacts hold no `CL:` handle,
which is one component of why the seams that carry them have no `CO:` row either — `S7` and `S11` both sit at
`Permanent: —` — but `S18` is the live case because its composition is the next one owed.)

---

## Wave, seams, and what this pass did not do

- **Wave state handed to Forward**: anchor `T018`, members `T018` → `T019` → `T020`, one local Yield, per-member retry
  counters 0 — all of 080's scheduling **unchanged**. Both absorbed rows are criteria amendments on tasks that already
  exist, so `plan-adhoc-task` was **not invoked** and no new task id was invented. `T019`/`T020`'s blocks in
  §Planned Task Context now carry the transmitted sub-blocks (counterexample, required behavior, done-when, Yield
  replay, assumption impact, cross-reference, candidate, ledger key) and `../plan-summary.md`'s rows name the
  amendment, so a path is not the handoff.
- **Assumption impact moved** from `required:new-surface` (what 080 recorded before the seams existed) to
  `scoped:S18` / `scoped:S19`; those two seams therefore **hold open** until their post-implementation composition is
  `SAFE`. `S16`/`S17` hold nothing: 391 is accepted and 391's State is not a blocker for composition. No seam is
  capped — none of `S16`–`S19` has been composed once, so the one-recomposition allowance is intact.
- **Seam state at hand-off, as the brief asked**: `S16`–`S19` remain `new`/`unverified`, `Permanent: —`, and every
  member row is now routed (`claims.md` `| pending |` count: **0**). The wave's Forward act is unblocked; the
  compositions run after the members land.
- **Questions**: zero. After base parity and `G-003`, no row left two materially different admissible outcomes —
  which is why `reserve-decision-record` was not invoked and **`FK-011` remains the only open Fork**, untouched.
- **Scope**: third-party subjects only. No `repo:` row created, rechecked or recomposed; `S1`–`S14` terminal and
  unopened; `S15`'s cells unmoved at `registry-counterexample`/`accepted-consequence`. The record-defect routing names
  lanes, not files this pass may edit.
- **Lanes not entered**: no claim checked, no fingerprint recomputed, no verdict file written or edited, no
  composition performed or judged safe, nothing promoted, no `CL:`/`CO:` handle allocated, no `Invariant:`/`Override:`/
  `Entity:`/`Scenario:` byte touched, nothing under `invariants/` opened for write, no `decisions.md` byte moved, no
  project code run. `git status --porcelain -- buster_ripper tests` is empty before and after.
- **Checkpoint**: **187** (`14d46a6`), `codify git create-checkpoint`, staging exactly the six files this pass wrote —
  `assumptions/claims.md`, `assumptions/seams.md`, `handoff-context.md`, `plan-summary.md`, `quality-backlog.md` and
  this report. **Disclosure:** the run-40 checkers' 22 untracked `assumptions/verdicts/claim-39x/4xx.md` artifacts and
  the two modified `invariants/` records (`assumption-claims.md`, `index.md`) are **not** in it and stay uncommitted —
  those are the `assumption-check` and `assumption-promote` lanes' writes, which the dispatcher commits at the
  assumption gate (`codify-agents` §Checkpoint Protocol), and routing them required reading them, not changing them.
  This is the state `075-01-yield` §Goalpost check certified as expected at this boundary: a non-empty `git status`
  outside the Forward lane, with `invariants/` untouched by the implementation loop. (A first draft of this paragraph
  stated that the checkpoint also carried those artifacts; Checkpoint 187's `--stat` says otherwise and the sentence is
  corrected here rather than left standing.)
- **Hashes.** Written from: `claims.md` `sha256:2011b68229e2f12c3e6c6365c87b25e769419b1b55c9823f70b50a585f0bea82`,
  `seams.md` `sha256:999d807102eaf32565ae3fff7ad0da12cb249bbada416e6f668e55813b24e425`. Re-read after the writes:
  `claims.md` `sha256:dc6fc0db73c3d498233c39ee185ea69b4f8f5cd1119186b9f89be8c0380b5d7f` (2100 → 2157 lines),
  `seams.md` `sha256:41aa8f0081af9b8ee2530182ff58efb5917004903204811532ec87e6a37cd7f5` (989 → 1032).
  `handoff-context.md` 1164 → 1410 and `plan-summary.md` 441 → 451 — both over cap by their own standing disclosure;
  the deep verbatim trim stays `phase-boundary-finalize`/`session-squash`'s act, and this pass offset nothing by
  deleting history because it wrote none.

Next: **Forward on `T018`** (wave anchor) → `T019` → `T020` → one local Yield, with the two amended done-whens binding
members 2 and 3 → `S16`–`S19` composition → assumptions recheck on the wave's bytes → Yield Red →
`phase-boundary-finalize`. The checker-lane header re-issue rides whichever lane reaches those artifacts first; it
gates `S18`'s publication, not the code.

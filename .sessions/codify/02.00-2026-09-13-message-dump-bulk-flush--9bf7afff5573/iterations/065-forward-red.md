# 065 — forward-red (scoped adhoc **misread RECHECK** — `T014` image-slice wave, bindings `063 R-A`…`R-E`)

## Orchestrator Prompt (verbatim)

> Execute the forward-red skill for this session. Arguments: $ARGUMENTS
>
> You are a subagent spawned by another session.
> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
> Mode: scoped adhoc misread recheck
> Tasks: T014
> Recheck: rows repaired by iterations/063-decide.md as projected by the registrar (T014 block bindings R-A…R-E)

---

## Verdict

**RED: 17 readings attacked | 0 AMBIGUOUS | 9 discretion footnotes** (+ 6 excluded-with-citation — one of them the
reused group of 061's eight prior exclusions, three of which the repairs **strengthened inline** — and 2 prior notes
re-checked and **discharged**: the row 14 + 19 restroking pair, and row 18's `G-003` converse half).

All five of 061's blocking rows are **foreclosed** — `A-1` by `R-A`, `A-2` by `R-B`, `A-3` by `R-C`, `A-4` by `R-D`,
`A-5` by `R-B` + `R-E` — and each now has at least one witness that cannot pass with the wrong artifact, which is
what the repairs were for. `T014` Forward is authorized **to the extent this gate owns: the assumption gate is
authorized next** (`required:new-surface` on `buster_ripper/image.py`, which no prior seam covers).

Pass depth: **compact** — one independent bounded attempt at a competing observable interpretation per landed binding
— with **one expansion** (row 6, the `R-A` × `R-B` × `R-C` write/sweep ordering), admitted because that interaction is
the only place the three repairs compose rather than stand alone. Route justification per
[proportional-verification](../../../.dotfiles/.agents/skills/run-codify/references/proportional-verification.md):
settled outcome with zero user forks (`R-A`…`R-E` are all deterministic closures), contained consequences
(`STRIP_IMAGES` defaults `False`, `config.py:40`, so no shipped default run moves), direct validation (the landed
case tables are executable witnesses), easy recovery (one module + its tests). Full depth would be owed only for a
new consequential decision; there is none left in this block.

**HEAD at read: `bcf596e`** (the registrar's Checkpoint 129 — the commit that *landed* these repairs, so unlike 064
this pass reads the registrations at their committed head, not beside them). Freshness: `git diff --name-only
51530da..bcf596e` returns **only session files** (0 non-`.sessions/` paths), so every code citation in the repaired
block is current at the head Forward will be dispatched against; `image.py` blob `9dab3461…` and `tests/test_image.py`
blob `4c990f64…` re-computed with `git hash-object` and equal to the block's stamp and to the evidence head
`25c2b77`. Working tree clean.

## Clean-room record

Read only what Forward receives: `plan-summary.md` (§Task List `T014` row `:392`, its `<details>` pointer `:394-411`,
§Goal Definition Matrix `G-003` `:98`, §Goal Coverage Matrix `:115`, §Anchor Map `:133-159` — including the closing
"Private helpers are not rows" sentence at `:158-159` that `R-A` cites, and the "Read path — deliberately NOT an anchor
row" note at `:154-156`), `handoff-context.md` (§Current Status, §Planned Task Context → the whole `T014` block
`:485-680` including the registration stamp, §Deterministic Closure Ledger → `DC-F-1-r2` `:914-928`, `DC-F-2-r2`
`:930-945`, `DC-F-3-r2` `:947-965`, §Question/note-disposal/wave schedule `:967-1008`), `qa-plan.md` (grepped: **zero**
hits for `image`/`strip`/`ttl`/`placeholder`/`T014` — there is no QA gate on this surface, as the block itself records;
one hit for the string `F-3` at `:225`, in the plan's *internal* findings namespace — see F8), and the code:
`buster_ripper/image.py` (full), `tests/test_image.py` (full), `normalize.py:202-228`, `routes/messages.py:24-60` +
the exception structure to `:138`, `config.py:39-42`, `main.py:70-78`/`:125-128`, `session.py:88-98`,
`server.py:40-60`/`:119`, `pyproject.toml:28-34`.

**Not opened:** any interview / invite / decide / organize / collect report — including `062-decide.md` and
`063-decide.md`, which named this scope but were not read. Repair *content* was taken only from the landed inline
`*(063 R-…)*` markings in the block and from the `r2` ledger rows the registrar projected, both forward-visible in
`handoff-context.md`.

**Disclosed exceptions (row identity, not intent):** `iterations/061-forward-red.md` — the prior instance of *this
same gate* on *this same wave* — read in full, because a recheck that does not know which readings the repairs were
meant to foreclose cannot test whether they do; it carries no planning-conversation context, and where its rows are
re-attacked I adjudicate from the repaired artifacts and the code, not from its conclusions. Also the first 40 lines of
`iterations/064-forward-red.md` (the sibling `T013` recheck) for pass shape and counting convention only — it states
on its face that it wrote no `T014` row. `iterations/059-forward-red.md` appeared as a NOTICE in the directory listing
(process history, no plan content).

## Attack table

| # | Surface (`G-*` or task) | Plausible misreading | Excluded by | Verdict |
|---|---|---|---|---|
| 1 | **`R-A`** (F-2) — the digest formula `img_<sha256(json.dumps(source, sort_keys=True, separators=(",", ":")))[:12]>` | **What `[:12]` slices.** Read as slicing the *serialization* (`sha256(dumps(source)[:12])`) rather than the digest — a reading that is not academic here: every realistic base64 source canonicalizes to `{"type":"bas…`, so a 12-char prefix is shared by **all** images in a session and every placeholder collapses to one key. The expression is also not executable as written (`.hexdigest()` is absent), so the reader must complete it, and `[:12]` can be inserted in either position. | The nesting puts `[:12]` outside `sha256(…)`, and the six-case table's own row **"two different `source` dicts → different keys"** fails under the truncated-serialization reading — including on the fixtures already in the file the member must migrate (`tests/test_image.py:86`/`:88`: `{"url":"data:1"}` vs `{"url":"data:2"}` share exactly the first 12 chars). One line to write correctly, no binding touched. | **Footnote F1** — recommend `hashlib.sha256(canonical).hexdigest()[:12]`, digest sliced, never the operand |
| 2 | **`R-A`** core (A-1 re-attack) — "one stable placeholder key per image source" | The original fork survives: hash the **whole image `block`** (which is what `plan-summary.md:392`'s title phrase "per image source" could still reach), or fold `session_id` into the digest. | `R-A` binds the referent to `block["source"]` and states the prohibition in the blueprint ("**The whole image `block` is not the input**"), with the falsifying counterexample in the artifact (`normalize.py:212` runs the image step **before** `_normalize_cache_control` at `:223`, so a block-level digest differs per call site — verified: `messages.py:34` calls `normalize(body, session_id="")`, so `:223` has already run before `messages.py:38`), plus the case-table row that exists specifically to discriminate it (stripped-at-`:38` vs stripped-at-`normalize.py:213` → byte-identical) and "same `source` in session A and in session B → identical" for the `session_id` axis. The task row now also binds its own title wording: `plan-summary.md:392` "the row's 'image source' wording is bound by `R-A` to `block['source']`, not to the whole image block". | **Excluded (cited)** — **A-1 foreclosed** |
| 3 | **`R-B`** core (A-2 re-attack) — where the prune/drop sits | Any of A-2's three placements still constructible: (a) between the `image.py:121-122` ensure and the writer's `:84` index → `KeyError` out of `messages.py:39` → HTTP 500 on the second request from a session idle past the TTL; (b) before the ensure → the ensure re-creates the dict, so the drop is unobservable; (c) after the loop but **session-scoped** → unreachable once F-1 aligns detector and writer. | (a) is now excluded by a clause written as **binding, not advisory** with its reason and its consequence in the same sentence, and witnessed by **W3** ("the writer does not raise, and no `KeyError` reaches the handler"); (b) by the positive position ("at the **tail** of `strip_and_cache_images`, **after the message loop**") and re-listed in `DC-F-3-r2`'s rejected candidates; (c) by "prune expired entries for **every** session in `_image_cache`" + **W2** (`"A" not in _image_cache`, an axis no session-scoped variant can produce). Verified in code: the read path really does index again after cleanup (`:155` → `:157`), so the "delete inside `_cleanup_expired_images`" variant is excluded by name and by `tests/test_image.py:137-156`. | **Excluded (cited)** — **A-2 foreclosed** |
| 4 | **`R-B`** rationale — "The primitive already exists and is entirely caller-less — `cleanup_expired_images()` (`image.py:176-179`) — so this is **a wiring fix, not a new mechanism**" | Read the rationale as the scope: wire `cleanup_expired_images()` **unchanged** and consider the member done, because the drop is already implemented somewhere in that primitive. It is not: `:176-179` loops sessions calling `_cleanup_expired_images(sid)`, and `:47-59` deletes expired **keys** only — neither ever removes an emptied session dict, so under that reading the leak W2 witnesses survives the fix. `DC-F-3-r2`'s Closure basis repeats the claim. | Not excluded by the rationale sentence; excluded **in effect** by the imperative it sits under ("prune expired entries for **every** session … **and delete any session dict that has emptied**"), by the Done-when clause, and by **W2** being a required gate rather than a suggestion. Gate: Q1 no (cannot reach a shipped run — W2 is mandated), Q2 no (no invariant or Anchor Map row), Q3 no (a few lines in the same function), Q4 no (same artifact). | **Footnote F2** + **evidence-tier observation** for the assumption route: correct the clause to "the global *prune* already exists and is caller-less; the emptied-dict drop is the addition" |
| 5 | **`R-B`** — "at the tail … after the message loop" | Position read against the function's *last statement* rather than the loop: `image.py:125-126` returns early when `messages` is empty, so a "tail of the function" placement never sweeps for that body — and a reader could conclude either that the sweep must move above `:125` (into the prohibited window's neighbourhood) or that a message-less request is a reclamation hole the wave must close. | The clause says "after the message **loop**", which is unambiguous once read as control flow; and the branch is unreachable from either guarded call site, because `has_images` returns `False` for an empty `messages` list (`image.py:31-32` → `:44`) and both sites guard on it (`messages.py:38`, `normalize.py:212`). No W-row depends on it. | **Footnote F3** — recommend: place after the loop, leave the `:125-126` early return untouched, and note in the commit message that the empty-`messages` path never sweeps by construction |
| 6 | **`R-C` × `R-B` × `R-A`** (the expansion) — "a repeat of an already-cached source refreshes `last_touched` only" + "`first_seen`, which TTL compares" + the sweep's tail position | The write that lands on an entry that is **present but already expired** is unwritten territory: treat it as a miss (fresh `first_seen`) rather than a refresh, and the tail sweep will not prune it — so an image resent on every turn is retained indefinitely and `--image-cache-ttl` silently becomes an *idle* timeout, which is the very shape the row's rejected-candidate list calls "retention unbounded for a busy session — re-enacts 'advertised controls that bind nothing'". `R-C`'s own narrative ("an entry **removed at expiry** and re-cached on the next send") presupposes a removal that, at a tail position, happens *after* the write that may have just refreshed it. | Partially excluded, by wording rather than by witness: "refreshes `last_touched` **only**" states it for any present entry, and "so retention is **absolute**" plus the rejected list ranks the alternative. Nothing in W1-W4 discriminates it — **W1 deliberately uses a *different* image**, and W2's `sid-B` request asserts only about `"A"`. Gate: Q1 no observable move in the forwarded body (the key is content-derived, so the text is identical either way) and no shipped-default change; Q2 no — the rejected list is block text, not an invariant or Anchor Map binding; Q3 no (one line); Q4 no. A wrong reading needs the implementer to invent a write-time expiry check no clause asks for, so the two readings are not comparably natural. | **Footnote F4** — recommend: on a key hit, carry `first_seen` forward unconditionally (no expiry test at the write site), and add **W5** = *resend the same `source` after expiry → that request's tail sweep drops the entry and the placeholder text is unchanged*, which is the one witness that makes the residue falsifiable |
| 7 | **`R-C`** — "Tuple shape at `:14` becomes `(source, first_seen, last_touched)`" against "`_evict_by_lru`'s order … must stay as it behaves today" and the read path's "no observable move" | Read the read-path immobility as a prohibition on *touching* `get_cached_image`, whose `source, _ = _image_cache[session_id][image_key]` at `:160-161` raises `ValueError` on a 3-tuple — so the member either skips the unpack edit or "preserves" the read path by leaving the module broken. | `R-A` itself cites `:160-161` as part of the referent's evidence trail (not as an untouchable), and the requirement is *behavioral* — the contract stays "source or `None`, never 'may delete your session'" (061 FN-1, honored by `R-B` keeping `_cleanup_expired_images` expiry-only). Five existing tests call `get_cached_image` (`:124, :155, :170, :196-198`), so a skipped unpack is red immediately. Sites to update: `:14` annotation, `:55` unpack, `:72` sort key, `:84` write, `:160` unpack. | **Footnote F5** — recommend: update the unpacks, change no read-path contract |
| 8 | **`R-D`** core (A-4 re-attack) — the five-row position table | An earliest-only detector still passes, because "scans all user messages" is wording and the witnesses are what Yield can observe. | Now foreclosed by witnesses rather than wording: row 2 (text-only **first** user message, image on a later turn) and row 3 (image only in a **middle** user message, first and latest text-only) each return `False` for an earliest-only and a latest-only abbreviation respectively. Verified against the code being replaced: the `reversed(messages)` + `break` shape at `image.py:35-42` returns `False` on row 2's body. Row 1's parenthetical also fixes the disposition of the old characterization test (`tests/test_image.py:49-58` — verified: it is exactly the earliest-image/latest-text-only shape asserting `False`) — "inverted to expect `True` … it is not exempted". | **Excluded (cited)** — **A-4 foreclosed** |
| 9 | **`R-D`** row 5 — "image only in a **non-user** (assistant) message → `False`" | Scope read narrower than intended (user-only is a bug: an assistant turn can carry base64 too) or wider (scan every role) — the sibling wave's rule ("no route carries its own abbreviated copy of the policy") invites the generalization. | The boundary is stated on both sides: the writer's scope is user-only (`image.py:128-130`, cited in the row), the flag's own help text says "Strip images from **user messages**" (`main.py:74-75`), the invariant candidate is phrased per user message, and the adjacent axes are recorded-with-owner rather than silently dropped (`tool_result`-nested images: "owner = this task; trigger = when … come into scope"). A mixed body (user image **and** assistant image) therefore strips one and forwards the other **by design**, and `has_images` must still return `True` for it — which row 5 answers only in the assistant-only case; the mixed case follows from row 1 (earliest/latest are both user) and is not a second referent. | **Excluded (cited)** |
| 10 | **`R-E`** — "Leak trigger gained (063 `R-E`) … the background janitor remains available as additive work" | Read the newly-added trigger as owing the background janitor **in this wave** (mode: replace the write-path sweep with a task, or add one alongside), which is the shape `DC-F-3-r1` rejected. | Same sentence's own tail: "**Recorded, not owed now**"; the wave surface (`image.py` + `tests/test_image.py` only); `R-B`'s explicit limit — "what 063 `R-B` authorizes is the existing janitor gaining a write-path caller, **not a second lifecycle task**"; and `DC-F-3-r2`'s rejected candidate "installing a background task as well". | **Excluded (cited)** — **A-5 foreclosed** (axis closed by `R-B`'s global sweep; the residual — a session that receives *no further* traffic — has owner + trigger) |
| 11 | **Structural** — 061 rows 14 + 19 (`T014` as three drifted renderings) | Re-attacked from the *repaired* artifacts: bind "the write path" to the dump cluster, or take "image source" as the whole block, working from the task list alone. | **Discharged as designed.** `plan-summary.md:394-411` no longer restates Required behavior, done-when, or any case table — it is a pointer that names both hazards in its own body ("this rendering is where `T014`'s 'write path' bound to the **dump cluster** … Read the block; do not reword both renderings into agreement"), and the task row binds its own title phrase (`:392`). Verified: the only `T014` semantics-bearing text in `plan-summary.md` is the row title + Notes; `IMAGE_CACHE_TTL` is not a dump symbol (`config.py:42`), so the one-hop misbinding no longer has a foothold even before following the pointer. The repair took 061's caution (i) — no double rewording. | **Prior note — DISCHARGED** |
| 12 | **Goal shape** — `G-003` boundary cell "No new endpoints, **no normalization changes**", whose failure witness is scoped to "dump/buffer/flush behavior" (`plan-summary.md:98`), and 061 FN-4's converse half ("nothing in the plan bounds an image-path **request failure**") | Two attempts: (i) `G-003` forbids the flag-on body delta F-1 deliberately creates (base64 → placeholder for historical images); (ii) `G-003` says nothing about the image surface, so a `KeyError`-→-500 regression sits outside every goal's witness — and `routes/*` is coverage-omitted (`pyproject.toml:31`), so nothing measures it either. | (i) excluded for the shipped default by the wave's own guard clause (`STRIP_IMAGES` defaults `False`, `config.py:40`), and the flag-on delta *is* the finding's intent (`057` §A.1). (ii) is now closed inside the blueprint rather than by a goal row: `R-B`'s positional prohibition is written "binding, not advisory" precisely because no `try` absorbs a misplacement, and **W3** is the observable gate. `G-003`'s witness text remains dump-scoped — recorded honestly: the authority for request-failure isolation on this surface is the block clause + W3, not the goal matrix. | **Footnote's prior note — DISCHARGED** (FN-4's converse half closed; no goal row owes a change) |
| 13 | **Prior exclusions reused** — 061 rows 6-13 (drop-the-guard; fix-the-route; consolidate the two call sites; scan-all-roles; build a key abstraction; Codex's monotonic counter; background task; delete `get_cached_image`) | Re-attacked each against the repaired wording: does a repair *re-open* a closed reading? | All eight still excluded, and three are now stated **in the blueprint itself** rather than only in the ledger: the key-abstraction reading by `R-A` ("no new module/class/registry — the `Mode: NEW` rows remain the only authorization", with the Anchor Map's closing sentence at `plan-summary.md:158-159` re-pointed after the line shift — verified: that is the sentence), the reviewer-proposal reading by `R-A`'s new "A monotonic per-session counter (Codex's own proposal) and first-free-index scanning are inadmissible" plus its reasons, and the second-lifecycle-task reading by `R-B`'s "not a second lifecycle task". Rows 8 and 13 also now have `R-D`'s by-construction clause ("it is not a directive to consolidate the two sites") and `R-B`'s "stays expiry-only". | **Excluded (cited; reused unchanged ×5, strengthened ×3)** |
| 14 | **`R-D`/`R-B` Done-when** — "advances the clock past `IMAGE_CACHE_TTL` with **no `get_cached_image` call**" | `time.sleep(TTL+ε)` following the module's own precedent at `tests/test_image.py:152`, versus patching the `time.time` seam the module actually calls (`:53`, `:72`, `:84`). | Not excluded; **non-material** — test-mechanics only, and the sweep now runs inside the same call as the write, so a sleep-based W-row pays wall time per session pair for no added witness. Recommendation unchanged from 061 FN-2 (patch the seam). Adjacent, and worth one line while the reader is here: the comparator is strict (`now - ts > config.IMAGE_CACHE_TTL`, `:56`), so "advance **past**" is literal and there is **no** witness at the exact boundary — `>=` vs `>` at `t == TTL` has no mandated test and no required behavior depends on it. | **Footnote F6** (carried, extended) |
| 15 | **`R-A`** Done-when — "every `img_N` literal in `tests/test_image.py` moves to the derived key — **all of them**: … `:78, :124, :155, :170` … `:196-198`" | Read "all of them" as the enumeration. Re-verified: the file holds **eight** `img_` sites — `:78, :124, :130, :155, :170, :196, :197, :198` — and the repaired list names seven, omitting `:130`, which is `get_cached_image("session-1", "img_999")`: a deliberate **unresolvable-key negative** with no derived form. Forcing it either fabricates a key or retargets the test's meaning; the repair's own emphasis makes the omission more visible than the packet's did, not less. | Not excluded; **non-material** — one line, self-correcting when the suite is run. | **Footnote F7** (carried, sharpened) — recommend: `grep -n "img_" tests/test_image.py` at F-2 time, migrate the positive cases, keep `:130` |
| 16 | **Naming** — the bare `F-n` namespace | "F-1/F-2/F-3" read as the plan's internal planning-findings series instead of this wave's members (`057` §A.1–A.3). Now with a second live instance: `qa-plan.md:225` says "their dispositions are closed in the plan (`F-3`, `FK-007` option 1, …)" — a *different* `F-3`, and this wave's `plan-summary.md:144` Anchor Map row for `F-1` is about numeric `config` globals with CLI precedent, a set that literally contains `IMAGE_CACHE_TTL`. | Not excluded; **non-material** — label only; every member's `Evidence:` line carries the disambiguating `057` §A.n path, and the repaired block qualifies each `R-` citation by report (`R-A`…`R-E` exist in both 062 and 063, which the registration stamp flags at `:493-494`). | **Footnote F8** (carried) — keep citing the producer, never the bare id |
| 17 | **Wave shape** — "ordered members: **F-1 → F-2 → F-3**" vs "F-2 and F-3 both edit … so they are **one increment**" vs "dispatch one local Yield per **wave**, not per member" (`:997`) | Cut as three sequential commits each with its own verification (the sibling wave's shape), or as one three-member increment. | Not excluded; **non-material** — commit granularity, cheap to re-cut, and unchanged by 063's batch. The load-bearing half is stated: `F-3` lands on the writer `F-2` rewrote, so that pair is designed together. | **Footnote F9** (carried) — recommend: `F-1`, then `F-2`+`F-3`, one local Yield for the wave |

**Counting convention** (comparable to 061/064): 17 readings attacked → **0 AMBIGUOUS**, **6 excluded-with-citation**
(rows 2, 3, 8, 9, 10, 13 — row 13 is one row standing for eight reused prior exclusions), **9 discretion footnotes**
(F1-F9: five new — F1, F2, F3, F4, F5; four carried — F6, F7, F8, F9), **2 prior notes re-checked and discharged**
(rows 11, 12). The 8-row ceiling applies to findings carried *out* of a pass; zero are carried.

## Materiality-gate record

The two calls a reader would most likely make differently, shown rather than asserted:

- **Row 4 stays a footnote, not AMBIGUOUS**, even though its premise is *false as written* and repeated in
  `DC-F-3-r2`'s basis. It would have blocked if the wrong reading could reach a shipped run; it cannot, because
  **W2** is a required gate and the instruction sentence is imperative ("prune … **and delete** any session dict that
  has emptied"). The gate asks whether a wrong guess is expensive or user-visible — here it is neither, so long as the
  acceptance set survives intact. The premise error is nonetheless real and cheap to fix, so it is handed to the
  assumption route rather than to Decide: this wave's recorded impact is already `required:new-surface` on exactly this
  module, so Forward Collect checks the claim against `image.py:176-179` at extraction time.
- **Row 6 stays a footnote, not AMBIGUOUS**, even though a wrong reading defeats F-3's own purpose for the dominant
  traffic pattern (an agent loop that resends one image every turn). Two reasons, both from the gate rather than
  instinct: Q1 — the forwarded body is identical either way, since the key is content-derived (`R-A`), so nothing
  user-observable moves and the shipped default is unchanged; Q2 — the rejected list the block itself carries is
  *plan text*, not an invariant or Anchor Map binding, so a conflict with it is a strong hint, not a gate hit. And
  unlike 061 row 3 (A-3), the two readings here are **not** comparably natural: the misreading requires adding a
  write-time expiry test that no clause asks for, whereas "refresh `last_touched` only" is what any straight
  key-hit branch does. It is recorded as a *witness* residue (recommend W5), which is exactly the tier 064 used for
  its three evidence-tier observations, and not a reopened A-3.
- **Row 12's disposition is narrower than "clean".** `G-003` still does not bound an image-path request failure — that
  hole is real and the prior pass named it. What changed is that the blueprint now closes it from inside (`R-B`'s
  prohibition + W3), so no plan artifact owes an edit. Stating it as discharged-without-caveat would hide which
  document carries the authority.

## Premises checked and found sound (recorded so a later pass does not re-open them)

- **`R-B`'s 500-path claim holds; its parenthetical does not.** Verified: nothing wraps `messages.py:38-40` — the
  handler's `try` blocks are `:26-29` (JSON) and `:56-59` (the dump call), with a third at `:129` inside a helper, so
  "the only handler-level `try` is `:26-29`" is inexact **while the conclusion is correct**. `logging_middleware`
  (`server.py:48-52`) only logs and awaits, and it is the sole middleware (`:119`). No behavior turns on the slip; it
  is recorded so a reader who goes looking finds one stated inaccuracy rather than a false premise.
- **`R-A`'s ordering counterexample is real**: `normalize.py:210-213` is step 0 and `_normalize_cache_control` is step
  3 at `:223`; `messages.py:31-34` calls `normalize(body, session_id="")` **before** computing `sid`, so on the
  production path the cache_control strip happens before `messages.py:38` ever runs, while a `normalize`-driven strip
  would see the field. A block-level digest therefore differs per call site, exactly as `R-A` says. (FN-5/row 20 from
  061 also re-verified: `:212`'s guard requires a truthy `session_id`, so that branch is production-dead today — the
  `:213` discriminator row is a unit witness, not a route witness.)
- **`session_id` exclusion is right for the stated reason**: the dict is already keyed by it at `image.py:84`, and
  `session.py:93-98` derives it from the system prompt, so folding it into the digest buys nothing and breaks the
  cross-session identity row.
- **`_evict_by_lru`'s "stay as it behaves today" is a true description of today**: the only stamp is written at insert
  **and on every overwrite** (`:84`), so sorting on `x[1][1]` (`:72`) is already least-recently-*touched*, and
  `R-C`'s split preserves the observable order while making TTL absolute.
- **Source-less image blocks do not break `R-A`**: `block.get("source", {})` (`:80`) maps two source-less blocks to
  one digest, and they hold the *same* value (`{}`), so "one source ⇄ one key" is satisfied, not violated — reported
  nowhere, because there is no competing reading of the plan here, only a behavior worth knowing.
- **`R-A`'s re-pointed Anchor Map citation lands**: `plan-summary.md:158-159` is the "Private helpers are not rows"
  closing sentence whose final clause the repair quotes — correct after the batch's line shifts, which is the
  non-trivial half (the pre-registration `:133-134` it replaced is now the section's header table).
- **Every test-site citation in the repaired block is exact**: `:49-58` (the earliest-image/latest-text-only
  characterization), `:10-15` (the autouse reset that hides F-2), `:137-156` (the read-path TTL witness), `:152`
  (the `sleep` precedent), `:196-198` (`get_cached_image` on `img_0/1/2` — the block's addition to the packet's list;
  its class name is `TestImageCacheLRUEviction`, so the cited method `test_evicts_when_over_limit` is the
  unambiguous handle). `DC-F-1-r2`'s claim that no existing shape pairs a text-only first user message with a later
  image is true on the whole file.
- **Coverage arithmetic as cited**: `image.py` is measured (`pyproject.toml:29-30`), `routes/*`, `main.py`, `server.py`
  are omitted (`:31-33`) — so the wave's witnesses must live in `tests/test_image.py` plus a route-level check, which
  is what `R-D`'s "Plus: a route-level witness" line is for. `DC-F-1-r2`'s rejected candidate "adding a coverage gate
  for `routes/*`" correctly cites the settled omit list (`:30-34`) instead of re-litigating it.

## Coverage / trade-off observed (per proportional-verification §Observe the trade-off)

- **Second recheck-generation pass on this wave** (061 → 063 → this): cost was one compact clean-room re-read of
  `image.py`, `tests/test_image.py`, both call sites, the three `r2` ledger rows, and the repaired block. Findings
  causing rework: **zero blocking**. One expansion was spent (row 6) and produced a witness recommendation, not a
  reopened reading.
- **Evidence reused, with its basis**: F6-F9 are 061's FN-2/FN-3/FN-6/FN-7 carried under
  [evidence-reuse](../../../.dotfiles/.agents/skills/run-codify/references/evidence-reuse.md) — same subject, same
  aspect, same domain, and the *freshness basis verified*: `image.py` and `tests/test_image.py` are byte-identical
  (blob `9dab3461…` / `4c990f64…`) to the head 061 checked, and 063's batch changed no code. Row 13 reuses 061
  rows 6-13 the same way, re-testing each against the new wording rather than trusting it. No prior *verdict* was
  copied: A-1…A-5 were re-adjudicated from the repaired text and the tree.
- **What the repairs did not need**: no new mechanism, no new artifact, no registry write, no user fork — and the
  batch resisted the temptation to reword both renderings into agreement (row 11). The five `r2` rows are
  refinements of the `r1` dispositions, which is what the ledger says and what the block's content matches.
- **Handed to the assumption route** (files and lines named, no Required cell or binding changed): F2's premise
  (`buster_ripper/image.py:176-179` does not drop emptied session dicts, while `R-B`'s rationale and
  `DC-F-3-r2`'s basis say the primitive already covers it) and F4's residue (no witness over the write-onto-expired-
  entry case at `image.py:84` under `R-C`'s two-stamp tuple). Forward Collect is owed fresh extraction on this module
  anyway (`required:new-surface`), so neither costs a round.

## Routing

- **0 AMBIGUOUS** → the **assumption gate is authorized** for the `T014` image-slice wave. **Forward is not
  authorized by this pass** — it runs after the assumption gate clears, then normal plan order **F-1 → F-2 → F-3** with
  `F-2`+`F-3` as one increment, one local Yield for the wave, no milestone gate (wave table `:989-997`).
- **No Decide or Interview round is owed.** The five footnotes with recommended readings are yield-reviewable
  discretion as usual; the two evidence-tier observations go to the assumption route.
- **Untouched by this pass**: `MS-004`'s APPROVED verdict, every accepted task's scope, the invariants registry
  (the two `T014` invariant candidates remain non-authoritative Interview material), `FK-011`, and the still-owed
  `QAP-MS001` / reachability-gated `QAP-MS002` items. No `T014` member carries a preserved-thinking clause.
- ⚠ This pass's report and its baton stamp are **uncommitted** session-file writes, consistent with 059-064; the
  repairs they check are committed at `bcf596e`.

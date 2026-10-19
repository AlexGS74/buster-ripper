# Iteration 061 — forward-red (scoped adhoc misread gate, **RE-RUN**) — `T014` image-slice wave (057 F-1, F-2, F-3)

## Orchestrator Prompt (verbatim)

> Execute the forward-red skill for this session. Arguments: $ARGUMENTS
>
> You are a subagent spawned by another session.
> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
> Mode: scoped adhoc misread gate (re-run)
> Tasks: T014 (image-slice wave: F-1, F-2, F-3)
> Context for re-run: the prior pass's report was destroyed by a concurrent iteration-id collision (see
> iterations/059-forward-red.md NOTICE); its five AMBIGUOUS rows survive in §Current Status but its discretion
> footnotes are lost. Re-run the pass and write the complete durable report.

## Why this file exists

[059-forward-red.md](059-forward-red.md) is a NOTICE, not a report: the `T014` pass and the `T013` pass were both
issued iteration id 059 and the `T013` report overwrote this pass's subject at that path. Surviving: the verdict and
the five blocking rows, quoted in `handoff-context.md` §Current Status. Lost: the attack table, the
excluded-with-citation rows, and the 8 discretion footnotes as written. This is a **fresh clean-room re-run over the
same scope**, not a reconstruction from memory — every row below is re-derived from the forward-visible artifacts and
the code, and each carries its own citation so a later reader does not have to trust the surviving stamp.

**Convergence note (reported, not leaned on):** this pass independently produced five blocking rows whose subjects
match the five in the §Current Status stamp, and the numbering is kept aligned (A-1…A-5 ↔ A-1…A-5) so Decide's routing
does not fork on handles. The match is evidence the surface is stable, not evidence either pass checked the other.

**Collision handling:** this pass allocated its number at write time, per 059's process note. `ls iterations` at
2026-10-05T14:56:25Z showed 77 files with newest `060-forward-red.md`, and `next-iteration-id` returned `061`; no
`061-*` file existed when this file was written.

## Verification profile and depth

The recorded profile for this surface (forward-visible fields): `Risk shape` per finding (`DC-F-1-r1`, `DC-F-2-r1`,
`DC-F-3-r1`), scope (`§Question, note-disposal, and wave schedule` table rows for `image-slice`), validation (each
member's `Done when` + `Yield replay`), and the `required:new-surface` assumption classification. **Absent from the
packet: an explicit `compact | parallel | full` route and reason.** This pass therefore selects **full depth on the
affected surface only** — reason: the wave edits a live request path (`/v1/messages`) whose failure mode is an
uncaught exception with no goal-level witness (row 18), and three members share two functions, so an individual
member's wording cannot be checked without its neighbours'. Scope is still bounded: no dump-cluster goal row, no TLS
row, no `MS-*` row is attacked here, and `G-003` is touched only where its boundary text reaches this path.

## Inputs read (clean-room record)

- `plan-summary.md` — §Implementation Plan Summary rows `T013`/`T014` + both `<details>` restatements (read as
  forward-visible artifacts, which is what makes them attackable — see row 19), §Anchor Map (including the "Read path —
  deliberately NOT an anchor row" note and the "`Mode: NEW` rows … remain the only authorization" preamble), §Goal
  Definition Matrix `G-003`, §Goals and Constraints, §Notes-adjacent constraints.
- `handoff-context.md` — §Current Status, §Open Forks (`FK-011`), §Notes, §Planned Task Context → `T014` block (the
  durable blueprint), §Deterministic Closure Ledger — 2026-10-05 routing (`DC-F-1-r1`, `DC-F-2-r1`, `DC-F-3-r1`),
  §Question, note-disposal, and wave schedule.
- `qa-plan.md` — grepped for the image surface: **zero** matches (consistent with `T014` gating no milestone).
- Code: `buster_ripper/image.py` (whole), `tests/test_image.py` (whole), `routes/messages.py:24-62`,
  `normalize.py:100-112, 202-227`, `session.py:93-98`, `config.py:40-42`, `main.py:74-75, 125-128`,
  `server.py:48-53, 119`, `pyproject.toml:28-34`, `README.md:1-20`.
- `iterations/057-codex-review.md` §A.1–§A.3 — read because the forward-visible packets cite it as their evidence path
  (`Evidence: iterations/057-codex-review.md §A.1`), i.e. Forward receives the pointer; §A.4+ not read (out of scope).
- `iterations/059-forward-red.md` — read as the dispatcher's stated reason for this re-run (a process record).

**Not read:** any interview / invite / decide / organize / collect iteration report; `060-forward-red.md`'s findings
(the sibling `T013` pass) were not used as input to any row below. `decisions.md`, `chronicle.md`,
`assumptions/**`, `quality-backlog.md` and `handoff-archive.md` were not opened.

---

## Attack table

`#` 1–5 are the blocking rows; 6–14 are readings the artifacts exclude (each still reported with the citation, per
step 4); 15–21 are non-excluded readings that fail the materiality gate → discretion footnotes; 19 is the structural
restroking note. Handles `A-*` match the §Current Status stamp.

| # | Surface (`G-*` or task) | Plausible misreading | Excluded by | Verdict |
|---|---|---|---|---|
| 1 | **F-2** — "the key is a pure function of the source **block**" / "deterministic function of the **image source**" / `img_<sha256(source)[:12]>` | **Two hash inputs.** (i) `block["source"]` — the dict named `source` at `image.py:80`; (ii) the whole image `block`. Both are "the source" in the artifacts' own words (`handoff-context.md:420` says *source block*; `:413` says *the image source*; `plan-summary.md:378` says *image source*). No canonical byte-form is stated for a dict input (`json.dumps(sort_keys=…)` vs `repr()` vs payload-only), and whether `session_id` participates is unstated. | **Not excluded.** `DC-F-2-r1`'s "e.g." (`handoff-context.md:518`, `057` §A.2 packet) marks the example non-binding; no Anchor Map row reaches `image.py`; the invariant candidate is `non-authoritative — Interview's` (`:408`). | **AMBIGUOUS — A-1** |
| 2 | **F-3** — "`strip_and_cache_images` removes that session's expired entries and **deletes `_image_cache[session_id]` once empty**" (`:456-458`) | **The prune's position relative to the placeholder writes is not stated**, and the three straightforward placements differ observably: (a) *after* the ensure at `image.py:121-122` → prune empties the dict, delete fires, then the loop's write at `:84` (`image.py:136` → `:84`) indexes a deleted key → `KeyError` propagates through `messages.py:39` → **HTTP 500 on the second request from any session idle past the TTL**; (b) *before* `:121-122` → the ensure re-creates the dict, so the drop is unobservable for any image-carrying request; (c) *after* the loop → the request's own writes make the dict non-empty, so the drop clause is dead code. | **Not excluded.** Nothing in `DC-F-3-r1` (`:533`), the member block (`:452-458`) or 057's packet ("call `_cleanup_expired_images(session_id)` from `strip_and_cache_images`") fixes the order, and no `Yield replay` line names it. | **AMBIGUOUS — A-2** |
| 3 | **F-2 × F-3** — `_image_cache[sid][key] = (source, time.time())` at `:84` under a content-addressed key | **Refresh-on-resend vs insert-on-miss is undecided**, so `IMAGE_CACHE_TTL` means either *idle since last send* (today's shape: an image resent every turn is retained for hours, and `--image-cache-ttl 300` never bounds it) or *absolute age since first cached* (base64 can vanish while its placeholder is still in the forwarded body). The same timestamp is what `_evict_by_lru` sorts on (`:72`), so the choice also rewrites "LRU" from last-used to first-seen order. | **Not excluded.** F-2's only statement about repeats is "dedupes … to **one cache entry**" (`:421`) — a count claim, silent on the timestamp. `config.py:42` and `main.py:127-128`'s help ("Image cache TTL in seconds") disambiguate neither. | **AMBIGUOUS — A-3** |
| 4 | **F-1** — "`has_images` scans **all** user messages" (`:390`), witnessed only by shapes with the image in the **earliest** user message | **An earliest-only detector passes every mandated F-1 witness.** `for msg in messages: if user: check; break` returns `True` for the inverted test at `tests/test_image.py:49-58` (image in message 0, latest user message text-only) and for the route-level witness as worded ("a text-only latest turn still strips a historical image", `:392`) and for the packet's counterexample (`:381-382`). It still returns `False` for the mirror shape — text-only first user message, image attached on a **later** turn — which forwards the base64 payload, the exact harm 057 §A.1 reports. | **Not excluded by any observable.** The wording says "every user message", but no witness in `Done when`, `Yield replay`, or `DC-F-1-r1` discriminates "all" from "first". | **AMBIGUOUS — A-4** |
| 5 | **F-3 / `T014` row** — "**retained base64 is bounded by** `config.IMAGE_CACHE_TTL` with empty per-session dicts reclaimed on the write path" (`plan-summary.md:379`, `:450-451`) against a closure that is **session-scoped** (`:456`) | **A global claim discharged by a per-session mechanism.** 057 §A.3's harm is precisely sessions that go away: `session_id = sha256(system)[:12]` (`session.py:93-98`), so each distinct system prompt adds an `_image_cache` key, and a write path that only prunes *the calling session's* entries never revisits an abandoned session — its expired payloads and its dict survive the process lifetime. Compounding row 2: after F-1 aligns detector and writer, `strip_and_cache_images` always writes ≥ 1 entry when it runs, so "delete once empty" is unreachable for every production-shaped request — the global janitor `cleanup_expired_images()` (`image.py:176-179`) stays caller-less. | **Not excluded.** `DC-F-3-r1` rejects a background task "**alone**" (`:536`), which leaves add-on open, and `:460-461` defers the janitor with a trigger ("if write-time pruning ever shows on the request path") that is a *cost* condition, not a *leak* condition — so the two artifacts do not settle whether the reported harm is closed. | **AMBIGUOUS — A-5** |
| 6 | **F-1** — "Required behavior: … one detector scope, one writer scope" | Read as 057's own alternative: **drop the `has_images` guard** and always call the writer when the flag is on. | Excluded: `DC-F-1-r1` Rejected candidates (`:507`) and 057 §A.1 packet candidate (b) — "leaves `normalize.py:212`'s guarded call carrying the identical defect, and makes `changes.append("stripped and cached images")` fire unconditionally" (`:387-389`). | Excluded (cited) |
| 7 | **F-1** — 057 §A.1's heading names `buster_ripper/routes/messages.py:38-40` | Read as **fix the route** (edit or wrap the guard at the call site). | Excluded: `DC-F-1-r1` Rejected candidates — "leave the detector and fix only the route: keeps two divergent scope definitions, which is the defect's mechanism" (`:508`); plus the wave-surface line "All three live in `buster_ripper/image.py` and `tests/test_image.py`" (`:357-358`). | Excluded (cited) |
| 8 | **F-1** — "both guarded call sites (`messages.py:38`, `normalize.py:212`) are covered by the same detector" (`:393`) | Read as a directive to **consolidate the two guarded call sites** into one shared entry point (mode: replace, not extend) — the sibling wave's rule ("no route carries its own abbreviated copy of the policy", `plan-summary.md:400`) invites the generalisation. | Excluded by the wave-surface line (`:357-358`: `image.py` + `tests/test_image.py` only) and by the clause's own wording — it constrains the *detector*, which both sites already call (`normalize.py:212`, `messages.py:38`). | Excluded (cited) |
| 9 | **F-1** — "every image block in **every user message**" | Read wider: scan **all messages of any role** (assistant/tool content lists can carry blocks). | Excluded: `:379` / `plan-summary.md:377` both say *user message*, and the writer's own scope is user-only (`image.py:129-130`); the `tool_result`-nested case is explicitly "recorded, not owed here" (`:396-397`). | Excluded (cited) |
| 10 | **F-2** — "derive one stable placeholder key per image source" | **Creation bias**: build a new key/registry abstraction (a `keys.py` module, a cache-key class, or a name→key registry) rather than a one-line derivation inside `_replace_image_with_placeholder`. | Excluded: `plan-summary.md:133-134` — "`Mode: NEW` rows above remain the **only** authorization", and no Anchor Map row reaches the image surface (`:365-368` records that); the packet's own example is an inline expression (`:518`). | Excluded (cited) |
| 11 | **F-2** — Codex's proposed remedy (`057` §A.2 concern text: "retain a monotonic per-session sequence") | Read the **reviewer's** suggestion as the plan's instruction, since 057 is a named evidence path in the block. | Excluded: `DC-F-2-r1` Rejected candidates name it and its reason (`:521`), and the member block repeats the rejection verbatim (`:417-419`) — the wave artifact overrides the producer's proposal, which 057 §A.2 itself flags as "understated". | Excluded (cited) |
| 12 | **F-3** — "enforce `IMAGE_CACHE_TTL`" | **Background task** (or wiring the janitor into the existing periodic machinery — this session's vocabulary already contains a flush timer and an `on_cleanup`), rather than a write-path prune. | Excluded: `DC-F-3-r1` Rejected candidates (`:536`), the member block's "A background task **alone** is inadmissible" (`:453-455`), and the `Deferred additive work` line keeping it available rather than owing it (`:460-461`); `server.py`'s timer belongs to the collapsed dump path (`plan-summary.md:118`). | Excluded (cited) |
| 13 | **F-3 / F-2** — "the placeholder text has no production reader today" (`:429-432`) | Read as license to **delete `get_cached_image` / the read-side API** as dead, or to change what the placeholder text resolves to. | Excluded: `DC-F-3-r1` Rejected candidates (`:538`) — "crosses settled discretion"; and the Anchor Map's "Read path — deliberately NOT an anchor row" (`plan-summary.md:129-131`) with `FK-007` option 1 (no tool ships). | Excluded (cited) |
| 14 | **F-3** — "empty per-session dicts reclaimed **on the write path**" (`plan-summary.md:379`, `:385`) — read **from `plan-summary.md` alone** | **Referent confusion with the dump cluster.** In this session's vocabulary "the write path", "per-session" and "session dict" are dump terms (`bulk flush` → `dump.py:flush_all`, per-`sid` buffer state, `DUMP_DIR/<sid>/`), and the `plan-summary.md` `<details>` block for `T014` names **neither** `image.py` **nor** `strip_and_cache_images` — it says only "pruned by the write path" and "the session dict dropped when it empties" (`:379-384`). A reader working from the task list can bind F-3 to `buster_ripper/dump.py`. | Excluded — but only one hop away: the `<details>` opens with "Full durable block in `handoff-context.md` §Planned Task Context" (`:372`), and that block names `strip_and_cache_images` (`:456`) and the wave surface (`:357`). The exclusion lives in the artifact the restatement points at, not in the restatement. | Excluded (cited; drives row 19) |
| 15 | **F-3** — "prune expired entries" vs 057's "call `_cleanup_expired_images(session_id)`" | **Which function carries the prune.** Extending the shared helper (`image.py:47-59`) to also drop the emptied dict gives the **read** path a delete: `get_cached_image` checks membership at `:151`, calls cleanup at `:155`, then indexes `_image_cache[session_id]` again at `:157` → `KeyError` after a drop, and `tests/test_image.py:137-156` errors instead of passing. | Not excluded; **non-material** — the read path has zero production callers (`:430-431`), the suite catches the variant, and the correction is one function boundary. | Footnote FN-1 |
| 16 | **F-3** — "a focused test **advances the clock** past `IMAGE_CACHE_TTL` through the write path" (`:457-458`) | `time.sleep(TTL+ε)`, following the module's own precedent at `tests/test_image.py:152`, versus patching the clock at the seam the module uses (`image.py` calls `time.time()` at `:53`, `:72`, `:84`). | Not excluded; **non-material** — test timing detail, no behavioral fork. | Footnote FN-2 |
| 17 | **F-2** — "every `img_N` literal in `tests/test_image.py` moves to the derived key — **all of them**: … `:78, :124, :155, :170` … `:196-198`" (`:421-423`) | Read the enumeration as exhaustive. It is not: `tests/test_image.py:130` also holds an `img_N` literal (`get_cached_image("session-1", "img_999")`), and it is a deliberate **negative** case with no derived form — forcing it to a digest either fabricates a key or retargets the test's meaning. | Not excluded; **non-material** — one line, self-correcting at run time. | Footnote FN-3 |
| 18 | **G-003** boundary — "No new endpoints, **no normalization changes**" (`plan-summary.md:73`), whose failure witness is scoped to "dump/buffer/flush behavior" | Two readings reach this wave: (i) `G-003` forbids the flag-on body delta F-1 deliberately creates (historical images now become placeholders where they were forwarded as base64); (ii) `G-003` is the dump cluster's goal and says nothing about the image surface — so nothing in the plan bounds an image-path **request failure**, and the row-2 `KeyError` → 500 sits outside every goal's witness (`image.py` is measured for coverage, `pyproject.toml:29-30`; `routes/*` is omitted, `:31`). | (i) excluded for the **default** path by the wave's own clause "`STRIP_IMAGES` defaults `False` … so no accepted `02.00` behavior and no shipped default run changes" (`:360-362`); the flag-on delta is the finding's intent (`057` §A.1). (ii) is a real hole but is dispositioned through row 2, not independently. | Footnote FN-4 |
| 19 | **Structural — restroking (double rendering)** | `T014` exists as **three** renderings that have drifted: `plan-summary.md:367` (table row), `plan-summary.md:376-385` (`<details>` "Required behavior / Done when"), and the `handoff-context.md:353-463` durable block (+ `DC-F-1/2/3-r1` at `:499-541`). The drift is not cosmetic: the compressed rendering **omits** F-1's two-call-site clause and F-2's test-literal migration, and it says "deterministic function of the **image source**" where the blueprint says "pure function of the **source block**" — i.e. the two renderings disagree **inside the phrase A-1 is about**. Row 14's mis-binding is available only through the compressed rendering. | n/a — this is a blueprint-element redundancy, not a plan gap to reword into agreement. | **Structural note (non-blocking)** |
| 20 | **F-1** — "both guarded call sites … are covered by the same detector" (`:393`), read as "one witness per call site" | Author a `normalize()`-path witness by passing a non-empty `session_id`, believing that site is live. It is not: `normalize()`'s only production caller is `messages.py:34`, which passes `session_id=""`, and `normalize.py:212`'s guard is `… and session_id and …` → the branch is unreachable in production today. That also weakens `DC-F-1-r1`'s stated reason for rejecting candidate (b) (`:507`). | Not excluded; **non-material** — costs one test, changes no shipped behavior; but see FN-5, because that test is where A-1's hash-input choice becomes visible. | Footnote FN-5 |
| 21 | **Naming — finding-id collision** | "F-1/F-2/F-3" is overloaded in the artifacts Forward reads: the members of this wave (`057` §A.1–A.3) and the plan's own internal planning-finding series (`plan-summary.md:28` "Twelve findings (F-1…F-12)"; `:119` "⚠ **`F-1` corrected** …"; §Notes last bullet "Findings `F-1`…`F-24`"). The Anchor Map's `F-1` is about **numeric `config` globals with CLI precedent** — a set that literally lists `IMAGE_CACHE_TTL` — so the two series collide on a row about this wave's own flag. | Not excluded; **non-material** — label only, and every member's `Evidence:` line carries the disambiguating `057 §A.n` path. | Footnote FN-6 |
| 22 | **Wave shape — "ordered members: F-1 → F-2 → F-3"** (`:357`, wave table `:670`) vs "**they are one increment**" (`:358-359`) vs "dispatch one local Yield per wave, **not per member**" (`:675`) | Read as three sequential commits-with-own-verification (the sibling wave's shape, "one commit each", `:231`) rather than one `F-2+F-3` increment after `F-1`. Load-bearing only in one direction: `F-3` lands on the writer `F-2` rewrote, so the pair must be designed together. | Not excluded; **non-material** — commit granularity, cheap to re-cut; the block is explicit that `F-2`+`F-3` are one increment. | Footnote FN-7 |

**Counting convention (so this is comparable to the lost pass's numbers):** 22 readings attacked in total →
**5 AMBIGUOUS** (rows 1–5), **9 excluded-with-citation** (rows 6–14), **7 discretion footnotes** (rows 15–18, 20–22),
**1 structural restroking note** (row 19). The 8-row ceiling applies to the findings carried out of this pass, and 5
are carried; the excluded rows are step-4 evidence that the gate ran, not additional findings.

---

## Footnotes (Forward discretion, yield-reviewable)

Non-blocking; each has a recommended default so the reader is never choosing blind.

- **FN-1 (row 15) — do not give the read path a delete.** Recommendation: keep `_cleanup_expired_images`
  (`image.py:47-59`) as the *expiry-only* helper and put the drop in the write path's own tail, or make it a separate
  module-private `_prune_session(sid)` used by the writer alone. `get_cached_image`'s contract stays "returns the
  source or `None`" — never "may delete your session".
- **FN-2 (row 16) — patch the clock, don't sleep on it.** Recommendation: patch the `time.time` seam the module
  actually calls (`image.py:53/72/84`); `sleep(1.1)` at `:152` is precedent for a *read*-path test, and the write-path
  witness needs two writes plus a prune, so sleeping multiplies suite time for no added witness.
- **FN-3 (row 17) — the literal list is as-of-writing.** Recommendation: `grep -n "img_" tests/test_image.py` at F-2
  time and migrate the positive-case assertions; **keep `:130`'s `"img_999"`** as an unresolvable key, which is what
  that test is for.
- **FN-4 (row 18) — `G-003` is not this wave's contract.** Recommendation: read `G-003`'s "no normalization changes"
  as the dump cluster's plan-scope boundary, and take the wave's own guard clause (`:360-362`) as the operative one:
  the shipped default run must not move. Converse, and worth one line in the repaired plan: nothing currently forbids
  `strip_and_cache_images` from **raising out of the request path** — `messages.py:38-40` has no `try` (the only
  handler-level `try` is `:24-27` for JSON parsing), `logging_middleware` (`server.py:48-53`) does not catch, and
  `G-003`'s witness is dump-scoped. Row 2 is where that becomes reachable.
- **FN-5 (row 20) — read "both guarded call sites" as scope, not as a per-site witness.** The clause is satisfied by
  construction (one `has_images`). If a `normalize`-path test is authored anyway, note it exercises a production-dead
  branch (`messages.py:34` passes `session_id=""`) — and that on that branch the block still carries `cache_control`,
  because `normalize` does the image step at `:212` **before** `_normalize_cache_control` at `:223`. That is A-1's
  counterexample; do not let the test pick the hash input implicitly.
- **FN-6 (row 21) — cite the producer, not the bare id.** Recommendation: keep every member reference as
  "`057` §A.n" (the `Evidence:` lines already do); a bare "F-1" in this session's files is ambiguous by construction.
- **FN-7 (row 22) — cut the wave as two commits.** Recommendation: `F-1` alone, then `F-2`+`F-3` together (the block's
  own "one increment" wording, `:358-359`), with one local Yield for the wave (`:675`).

**Structural note (row 19), for the repair owner — not a wording tidy-up.** Keep the `handoff-context.md` `T014` block
as the single blueprint and reduce `plan-summary.md:376-385` to the pointer it already opens with (`:372`), or delete
the restatement. Two cautions: (i) the fix must **not** be to reword both renderings into agreement — that doubles the
next drift; (ii) when the compressed rendering's "deterministic function of the image source" is reconciled with the
blueprint's "pure function of the source block", the **referent still has to be chosen** — that is A-1, and Decide
should close it as one binding, not as a synonym.

---

## Materiality-gate record

Every non-excluded row above was put through the four questions; the two decisions a reader would most likely make
differently are shown rather than asserted.

- **Row 4 (A-4) is AMBIGUOUS, not a footnote**, even though "all user messages" is plainly written: the gate asks
  whether a *wrong guess is expensive or user-visible*, and here the guess ships a detector that forwards base64 for
  the most common real shape (text first, image attached on a later turn) while every mandated witness is green — Q1
  yes (payload forwarded upstream, KV prefix changes per turn), Q4 yes (the artifact that "fixes" F-1 does not). The
  plan-wording exclusion that *feels* sufficient is exactly the thing this pass exists to distrust, because the
  witnesses are what Yield can observe.
- **Row 3 (A-3) is AMBIGUOUS, not a footnote**, even though the correction is one line: the two readings make an
  advertised operator control mean different things, and F-3's own defect *is* "advertised controls that bind
  nothing" (`:440`). It would have been a footnote if the fork were only retention volume; it is a fork because the
  same timestamp also defines `_evict_by_lru`'s order (`image.py:72`), so the choice decides whether "LRU" means
  last-used or first-seen.
- **Row 15 and row 20 stayed footnotes** on purpose: both differ only in internal placement, both are caught by the
  existing suite (`tests/test_image.py:137-156`, `:128-131`), and neither touches an invariant or an Anchor Map
  binding.

## One premise checked and found sound (recorded so a later pass does not re-open it)

`DC-F-2-r1`'s basis attributes "the KV cache prefix [stays] stable across turns" to "README/module docstring". Checked:
`README.md:3-5` says the proxy rips out cache busters "**so the KV cache prefix stays stable across turns**" — the
quote is real where the packet points, so F-2's direction has a live authority and is **not** a false premise. It
still does not define the *derivation* (A-1), which is a different question from the direction.

## Coverage / trade-off observed (per proportional-verification §Observe the trade-off)

- **This is a repeated check caused by a process defect, not by a finding.** Cost: one full pass re-run (clean-room
  re-read of `image.py`, `tests/test_image.py`, both call sites, and the three closure packets). Findings caused by
  the re-run itself: **zero new AMBIGUOUS rows** — the five converge with the surviving stamp — but the re-run
  recovered 9 excluded rows, 7 footnotes, and the row-19 restroking warning that the lost report held.
- **Named process gap for the runner** (not a plan finding, and not a repeat of 059's suggestion verbatim): a scoped
  pass that allocates its iteration id at dispatch can be overwritten by a sibling allocated at the same moment. The
  race is closed by taking the id at *write* time — `next-iteration-id` re-run immediately before the write, as done
  here — or by namespacing scoped passes per task (`NNN-forward-red-T014.md`). This pass did the former; the latter is
  the runner owner's call.

## Routing

- **5 AMBIGUOUS** → rows 1–5 (`A-1`…`A-5`) go to **Decide in `red-findings` mode**. `T014` Forward is **not**
  authorized. Each row needs one binding clause, and none needs a new mechanism: A-1 = the hash input and its
  canonical form (and whether `session_id` participates); A-2 = the prune/drop's position relative to the writer's
  writes, plus what "once empty" is measured against; A-3 = whether a re-send refreshes the entry's timestamp (idle vs
  absolute TTL, and what "LRU" then means); A-4 = one discriminating F-1 witness (image in a **later** user message,
  earliest user message text-only); A-5 = whether the wave owns global reclamation, or the required-behavior sentence
  is narrowed to active sessions and the idle-session axis gets a named owner.
- **Rows 14 + 19 travel with A-1** rather than as separate items: the restatement is where the wrong module and the
  wrong referent are both reachable.
- **Footnotes and the structural note need no round** — Yield reviews the discretion calls as usual.
- Nothing here moves `MS-004`'s APPROVED verdict, any accepted task's scope, or `FK-011`'s ownership (no
  `T014` member carries a preserved-thinking clause).

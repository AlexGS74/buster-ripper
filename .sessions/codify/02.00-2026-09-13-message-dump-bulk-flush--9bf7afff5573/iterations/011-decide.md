# 011 — Decide (OPTIONS repair — `FK-004` → `FK-005`)

## Orchestrator Prompt (verbatim)

> Execute the decide-interactive skill for this session. Arguments: $ARGUMENTS
>
> [The remainder of the prompt was the `decide-interactive` skill body plus the
> `codify-agents`, `critical-thinker`, `codify-user-prompt`, and
> `codify-deterministic-closure` skill bodies. `$ARGUMENTS` was empty. No `Mode:`
> line was present, so this is OPTIONS mode. Session ID was supplied out-of-band
> as `02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573`; Question Mode is
> `deferred` per session metadata.]
>
> Task block, verbatim:
>
> > Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
> > User Context: "Donald Dispatch, is it possible for us to make actual conversation be rebuilt, not full of it, but this turn of actual conversation being rebuilt in the log?"
> > FK-004 is open and owned by decide. Repair or extend its presented option set as the owner judges necessary, then re-present.

## Session Type / Mode

- Continuing session `02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573`.
  Identity verified against `session-identity.json`: UID
  `9bf7afff-5573-402a-b102-827423a1dded`; handle suffix `9bf7afff5573` = first 12
  hex of the UID; display id `02.00-2026-09-13-message-dump-bulk-flush` matches the
  handle prefix. **No `parent_session_uid`** → the parent coordination contract does
  not apply; there is no parent path to resolve or reconcile.
- **Mode: OPTIONS** (no `Mode:` line in the prompt), narrowed to the one slice the
  prompt names: `FK-004`'s option set. `FK-001` and `FK-002` are resolved and are
  treated as settled; no other slice is re-derived.
- **Question Mode: deferred.** Decide does not ask. The repaired option set is
  authored as `FK-005` for invite/interview to present.
- **`codify-user-prompt` is not invoked**: it presents one already-recorded FK via
  `AskUserQuestion`, which is not available on this dispatch path, and the FK it
  would present does not exist until this pass authors it.

## Context Conflict Detected

None between session-local sources. One **premise in the orchestrator prompt** was
checked rather than adopted: the prompt calls `FK-004` "its presented option set".
`FK-004`'s own `Presented.Report` field records "**not yet presented** — authored
this pass for presentation", and no invite iteration exists between 010 and this
pass, so the user has never seen `FK-004`. The word "presented" is therefore read as
"the option set `FK-004` carries", not as "the text the user saw". This matters for
one decision only — whether the repair can be made in place — and is resolved below
under §Fork Authoring Record.

## Changes Since Iteration 010

- NEW: `US-006` reserved by the allocator, filled in place, set `recorded` — the
  user's fourth statement, with its normative projection and three falsifiable
  claims split and each routed to a verifier.
- NEW: `FK-005` reserved, filled, set `open` — the repaired presented set.
- UPDATED: `FK-004` `State: open` → `superseded`, `Superseded-by: FK-005`, with a
  supersession note. Its `Presented` text is **untouched**.
- NEW: **Answer 4** — is it possible to have the actual conversation rebuilt in the
  log, not the full request? (yes; and for the change-focused option it is already
  what the artifact holds).
- NEW: Finding **F-8** — the artifact's turns are not one linear conversation; 42 of
  the real dump's 137 transitions go backwards.
- UPDATED: `plan-summary.md` §Current Phase, §Options, §Goal Definition Matrix
  (`G-006` row), §Open Questions 2b.
- UPDATED: `handoff-context.md` §Current Status, §Open Forks, §Resolved Forks, §Notes.
- UNCHANGED: `FK-001`, `FK-002`, `FK-003`, `US-003`/`US-004`/`US-005`, every `EI:`
  disposition, `FK-A`/`FK-B`, the flush-policy cluster, the milestone skeleton, the
  Anchor Map's pre-existing bindings, `RB-001`–`RB-003`, findings F-1…F-7.

## What the user actually faces

The user has seen `FK-003`'s three options (per-turn diffs / per-turn full bodies /
machine record + viewer). Their fourth statement does not pick among them; it asks
whether the log can hold **the actual conversation** instead of the request. That is
answerable, and the answer changes what the options mean — so it is answered and the
presented set is repaired, not extended with a new alternative.

| Item | Status after this pass |
|---|---|
| **What the artifact holds and how a human reads it** | **`FK-005` — open, authored, ready to present** |
| `FK-004` | **superseded** (never presented) — its presented text is preserved |
| Is "the actual conversation rebuilt in the log" possible? | **answered** (yes; §Answer 4) |
| `FK-003`, `US-003`, `US-004`, `US-005` | unchanged |
| `FK-001`, `FK-002`, `FK-A`, `FK-B`, the `EI:` dispositions | unchanged |
| QA infrastructure setup | carried — still a presentation **topic**, not an FK |

## Answer 4 — "is it possible for us to make actual conversation be rebuilt, not full of it, but this turn of actual conversation being rebuilt in the log?"

**Yes — and for the change-focused option it is already what the artifact holds.** The
answer has three parts: what "the actual conversation" is in this artifact, why the
change-focused log already carries it, and the one thing it does not do.

**1. The conversation is a named part of the rendering, not a new capability.** The
artifact's readable text is produced by `_render_body` (`dump.py:27-58`), which emits
exactly three kinds of section: `[tools]`, `[system]`, and `[message i / role]`. The
conversation is the message sections (`dump.py:45-56`) — the user turns, the
assistant's replies with their thinking and tool calls, and the tool results. It is
already in the artifact; no new rendering or reconstruction is needed to have it.

**2. The request's constant scaffolding never appears in a change-focused record.**
This is the part the user's question could not have known, and it is measured, not
argued:

- A session is keyed on the **system prompt** — `session_id` is
  `sha256(system)[:12]` (`session.py:93-98`). The `[system]` section therefore
  *cannot* differ between two records of the same artifact; it is constant by
  construction.
- The `[tools]` section is constant in practice: across all **137** diffs of the real
  dump, `grep -lE '^[-+].*\[(tools|system)\]' turn_*.diff` matches **0** files. The
  index line for every one of the 137 turns reads "7 tools".
- A unified diff emits only changed lines plus context, so a constant section produces
  no diff lines at all.

So a change-focused record is, in substance, **a conversation record**: what it shows
turn to turn is the conversation's changes, and the request's scaffolding enters it
only when it changes — which is itself a cache buster, and exactly the thing the tool
exists to find (`README.md:134-135`). Nothing has to be built for this.

**3. The one thing "rebuilt" does not get for free is a single thread.** The
artifact's turns are **not one linear conversation**:

- `grep -o '[0-9]* msgs' index.txt` → **42 of 137** turns have fewer messages than the
  previous turn.
- The early diffs show whole message blocks added and removed: `turn_003.diff` adds
  messages 2–7, `turn_004.diff` removes 5–7, `turn_005.diff` re-adds 5–10.

Consecutive turns therefore belong to interleaved request histories that share the
system prompt (and hence the artifact). Rebuilding *the* conversation as a merged
thread would need a grouping rule the artifact does not currently carry; rebuilding
*each turn's* conversation — the per-turn message list, which is what the records
already hold — needs nothing. The user's phrasing ("this turn of actual conversation")
is the per-turn reading, which is the available one.

**On "not full of it":** the clause is informative only against `FK-003`'s "per-turn
full bodies" option, which the user had just seen. It reads as a lean toward the
change-focused form — the form in which the conversation arrives as changes rather
than as a repeated whole. Decide does not treat it as a pick; the option set presents
both and the user chooses.

**Cost of scoping a record to the conversation *alone* (dropping the constant tool
definitions and system prompt):** measured on turn 0 — tools 8,851 bytes, system
13,223 bytes, messages 68,113 bytes, against 88.2 KB of rendered text. So the
scaffolding is about **a quarter** of a rendered record. In the change-focused options
that quarter never appears anyway (point 2), so scoping changes almost nothing; in the
self-contained option it is a real saving, because that option repeats the scaffolding
in every record. This is why the scope is offered as a **named modification** on the
options rather than as a fifth alternative — see §Why four options and not five.

## Options Analysis

The option set is **repaired, not extended**. All four of `FK-004`'s options survive
with their identities, their costs and their `Recommendation: none` unchanged; the
question gains the constant-scaffolding clause, the preamble answers the conversation
question, and each option now states what it holds of the conversation versus the
request. No option was added, removed, or re-ranked.

`G-006` is the goal being arbitrated. It is OPEN with four alternate readings, and no
repository evidence excludes any of them. The choice remains a **genuine preference**:
the constant-scaffolding fact removes a false cost from the change-focused options
(they were never carrying the request's bulk), but it ranks nothing — the
bytes-versus-self-containment trade-off between options 1 and 2 is a real product
choice, and the viewer options trade scope for navigation.

### Option 1 — Change-focused log, plain text tools

- **What it is**: one append-only artifact per session holding a summary line per
  turn, turn 0's baseline (raw body + rendered rendering), and for every later turn
  the change against the previous turn.
- **What it changes**: three files become one; the information content is what the
  tool produces today.
- **What stays the same**: the `grep`/`less` reading habit; the summary-line-per-turn
  index; the change-focused view that makes a cache buster visible at a glance.
- **Pros**: smallest of the four (≈5.8 MB on the real dump); closest to today's
  reading experience; no new component; the `+N/-M` counts stay free by-products
  (`dump.py:96-97`); and it is already, in substance, the conversation log — the
  constant scaffolding produces no diff lines (0 of 137).
- **Cons**: turn K's full text needs a replay from turn 0; the artifact is a text log,
  so a scripted consumer must parse diffs.
- **Complexity**: low. **Risk**: low.
- **Anchor stance delta**: none. No `Mode: NEW` row.

### Option 2 — Self-contained log, plain text tools

- **What it is**: the same one file and the same tools, but every turn's record
  carries its complete rendered text.
- **What it changes**: no changes are stored; every record stands alone.
- **What stays the same**: one write per flush, one file per session, append-only,
  arrival order, one summary line per turn.
- **Pros**: any turn opens and reads directly; robust to damage in an earlier record;
  a `grep` for a cache-buster string hits every turn it appears in.
- **Cons**: ≈2.3× the bytes (≈13 MB vs ≈5.8 MB), which is the quantity the user's own
  `FK-002` rationale was about; the change between two turns is no longer shown; the
  `+N/-M` counts become a computed comparison; **and each record repeats the constant
  tool definitions and system prompt** (22 KB of a 88.2 KB turn-0 rendering) — the one
  option where scoping to the conversation alone is a real byte saving.
- **Complexity**: low. **Risk**: medium — the risk is the byte axis, and the research
  (iteration 007, F1/F2) is explicit that bytes written, not write-call count, is the
  physical wear quantity.
- **Anchor stance delta**: none. No `Mode: NEW` row. A conversation-scoped variant is
  a clause on this option, not a new anchor.

### Option 3 — Change-focused log + small terminal viewer  *(the user's earlier proposal)*

- **What it is**: option 1's artifact, unchanged, plus a small terminal viewer that
  lists a session's turns, shows a turn's reconstructed full text, and highlights what
  changed against the previous turn.
- **What it changes**: the plan gains a read-side command; the artifact's bytes and
  content are option 1's.
- **What stays the same**: everything option 1 keeps, including `grep`/`less` working
  on the artifact; the artifact is still the primary record.
- **Pros**: fewest bytes *and* the best readability — it is the only option that
  improves navigation without paying bytes; the dependency cost is zero (stdlib
  `curses`; `difflib` already imported at `dump.py:5`); it answers the user's stated
  interest directly.
- **Cons**: a component this plan has never scoped and must authorize and evidence;
  `README.md:146-162` documents a tool instead of a format; `G-006`'s "without
  additional tooling" clause must be re-derived (the artifact alone no longer meets
  it); more test surface than options 1 or 2.
- **Complexity**: medium. **Risk**: medium — the risk is scope, not correctness: a
  read-side component arrives in a session whose purpose is a write-side change, and
  `RB-003`'s read-path search found no code reader to build on, so it is a new
  component from zero.
- **Anchor stance delta**: **adds** a `Mode: NEW` row (viewer command) requiring its
  own complete `RB-*` row, and turns `T005` into tool documentation.

### Option 4 — Machine record + viewer  *(retained from `FK-003`)*

- **What it is**: one JSON object per turn, appended one per line, read with `jq` or
  the viewer.
- **What it changes**: the artifact stops being human-readable; the documented
  `cat`/`grep`/`less` workflow stops working.
- **What stays the same**: one write per flush, one file per session, append-only,
  arrival order.
- **Pros**: the most compact and the most scriptable; trivial to aggregate across
  sessions; the record shape can change later without breaking a text format.
- **Cons**: the viewer becomes **required** rather than additive, so it carries option
  3's new-component cost *and* loses what option 3 keeps; the tool's documented purpose
  moves behind a tool; the smallest records are the least FTL-friendly shape per the
  research (iteration 007, F2); and it holds the conversation only as a field that must
  be extracted before it can be read.
- **Complexity**: medium. **Risk**: medium-high — it is the only option that removes
  human readability from the artifact itself, which is in tension with `US-005`.
- **Anchor stance delta**: same `Mode: NEW` viewer row as option 3, plus a change to
  `T005`'s content.

### Why four options and not five

The user's statement names a direction — the conversation rather than the request —
and the first instinct is to add it as a fifth option. It is not added, because once
measured it is not a materially different outcome in the form the user asked for:

- In the **change-focused** form the scaffolding never appears in a record (0 of 137
  diffs touch `[tools]` or `[system]`, and `[system]` cannot change at all). A
  "conversation-scoped change log" would therefore differ from option 1 only in
  *suppressing* the signal when the tool definitions or the system prompt do change —
  a pure loss, because unchanged sections already cost nothing. Presenting it as an
  option would be manufacturing an alternative, which the skill forbids.
- In the **self-contained** form the scope is a real saving (≈25% of the bytes) but it
  is a *variant of one option*, not a different approach. The skill's own rule applies:
  a combination not packaged in the list is expressible by the user without an extra
  option. `FK-005`'s preamble names it explicitly so the user can attach it in one
  clause.

Option 3 is retained for the same reason it was retained in 010 — it is the user's own
earlier proposal, it is a genuinely distinct combination (change-focused **and** a
viewer), and the user has already engaged with the option set it contrasts with.
Dropping it would hide a real alternative.

### Rejected — "write a blob" (no readability commitment)

Considered and rejected in iterations 008 and 010, and **still rejected**. The cheapest
way to one write per flush is to write the least. That satisfies `G-001` completely and
destroys the tool's documented purpose while every implementation task completes — it
is precisely `G-006`'s failure witness. Not a product choice; the failure mode. Not
presented.

## Findings

### F-8 — the artifact's turns are not one linear conversation

The artifact's records are ordered by arrival, and its diffs are taken against the
*previous record*, not against the previous turn of the same conversation. In the real
dump, **42 of 137** turns have fewer messages than the previous turn, and whole message
blocks are added and removed (`turn_003.diff` adds messages 2–7; `turn_004.diff`
removes 5–7; `turn_005.diff` re-adds 5–10). The artifact's `session_id` keys on the
system prompt (`session.py:93-98`), so several interleaved request histories share one
artifact. **Cause not determined** — this is an observation about the artifact's
contents, not a verdict about the client.

**Consequence for PLAN-SELECTED**: no option in `FK-005` requires a conversation
thread, so this changes nothing about the task list. It becomes load-bearing only if
someone proposes reconstructing *a* conversation from the artifact (a merged thread, a
per-thread view in the viewer, or a "conversation rebuilt" artifact) — that needs a
grouping rule the plan does not have, and it is out of scope for this session's
write-side change. If a viewer option is chosen, the viewer lists **turns** in arrival
order, not threads. Record the limitation in the README text `T005` writes.

### F-9 — `G-006`'s "without additional tooling" clause (carried, F-7, restated)

`G-006`'s `Met when` reads "…without additional tooling". Options 1 and 2 satisfy it as
written. Options 3 and 4 do not: the artifact alone no longer answers "where did this
change", so the clause must be re-derived to "…or, for the viewer options, with the
viewer this plan ships". **PLAN-SELECTED must** re-derive `G-006` on the chosen option,
and the `OI:` register must carry the replacement statement for
`EI:.../output_layout` accordingly. Not a user question — it follows from the pick.

### Carried from iterations 008 and 010 (unchanged, restated so PLAN-SELECTED does not lose them)

**F-6** — "full body" is load-bearing in `VP-T002-turn0-available` and
`EI:.../turn_zero_is_full_body` and defined in neither; define it in the `OI:` statement
and `T002` as turn 0's raw normalized JSON body **and** its rendered rendering (all four
options keep both). **F-1** — the Anchor Map's *flush trigger parameters* row claims
every `config` global has a CLI flag in `main.py`; **false** (`KV_POLL_INTERVAL`,
`config.py:44`, has none), so the flag is required by QA injectability, not convention.
**F-2** — that row names `DUMP_FLUSH_TURNS` against `FK-001`'s recorded **data-volume**
basis; it is `DUMP_FLUSH_BYTES`. **F-3** — keep one write-count VP, per-flush form
folded in. **F-4** — keep `G-001` witness B (the byte axis) live. **F-5** —
`dump.py:63`'s bare `assert`; coverage headroom `U ≤ 3.2 + 0.2·K`.

## Planned Invariants In Scope

**No change from iterations 008 and 010.** The three `EI:` entries in the contradiction
surface keep their dispositions (`output_layout` **override**; `sequential_diff_chain`
**refine**; `turn_zero_is_full_body` **refine**), and this repair moves none of them:

- `output_layout` **override** — the replacement statement's *record content* is written
  by PLAN-SELECTED from `FK-005`'s answer. F-6 adds the definition of "full" to it.
- `sequential_diff_chain` **refine** — under option 2 or 4 there is no diff chain in the
  artifact at all; the load-bearing clause is the `turn`/`prev_text` sequencing at append
  time, which `T001` already settles. Unchanged. **F-8 does not touch it**: the diff is
  against the previous *record*, which is what the invariant says, and arrival order is
  the artifact's order.
- `turn_zero_is_full_body` **refine** — `FK-005`'s preamble keeps turn 0's raw body in
  every option, so the refine holds for all four. F-6 sharpens its wording.

`G-006` ↔ `G-001` remains a goal-level tension, not an invariant one. `FK-005`
arbitrates it; options 3 and 4 additionally move `G-006`'s own text (F-9).

## Fork Authoring Record

- **`US-006`** — reserved via `codify codify reserve-decision-record --kind
  user-statement --owner decide`, filled in place, set `recorded`. It preserves the
  verbatim words, the source timestamp and locator, the normative projection, and three
  falsifiable claims each with its own `Verification`/`Evidence`/`Consequence`, all
  routed to `assumptions PENDING`. Decide's own readings are recorded as `#`
  annotations inside the claim blocks and expanded in §Answer 4 — they are not promoted
  to verdicts.
- **`FK-005`** — reserved via `codify codify reserve-decision-record --kind fork --owner
  decide`, filled in place, `State: open`. `Type: preference`; `Why-yours` quotes the
  escalation-floor clause *"it is irreversible or expensive to reverse (data loss,
  public API surface, security)"* verbatim (interview → "Escalation floor"); `Presented`
  carries the exact question, the preamble, all four option labels and descriptions,
  `Recommendation: none`, the timestamp, and the report path.
- **`FK-004`** — `State: open` → `superseded`, `Superseded-by: FK-005`, with a
  supersession note. `Supersedes: FK-003` and the whole `Presented` block are untouched.
  **Why supersede rather than repair in place**: the skill's rule is to never overwrite
  prior `Presented` text, and `Presented` is written *before* showing precisely so the
  record of what would be shown survives a repair. `FK-004` was never presented, so
  superseding loses nothing the user saw; overwriting in place would leave its
  `Report: iterations/010-decide.md` pointer describing text that no longer exists.
- **No reservation was left abandoned.** Two allocator calls this pass (`US-006`,
  `FK-005`); both were filled.
- **`FK-A` / `FK-B`** remain **not askable** (iterations 008 and 010's escalation-floor
  re-tests; unchanged by this pass). Not reserved, not filled.
- **Legacy migration**: not applicable — `decisions.md` is canonical and the handoff's
  Fork sections are indexes only.

## Dropped-Clause Sweep

**Triggered** — this pass wholesale replaces a prior definition of the same scope
(`FK-004`'s presented set is superseded by `FK-005`'s). One hop only: the direct
predecessor is `FK-004`'s `Presented` block, which has no invariant lineage to walk
(it cites `G-006`, whose text is unchanged by this pass).

| `FK-004` `Presented` clause | Disposition | Where it now lives |
|---|---|---|
| "(1) one append-only file per session, written once per bulk flush — your `FK-002` choice" | `carried` | `FK-005` preamble (1) |
| "(2) 'Full' is now defined … raw JSON body **and** its rendered rendering" | `carried` | `FK-005` preamble (2) |
| "(3) the numbers … 137 turns, 93,886 bytes … 5,785,285 bytes of change text" | `carried` + extended | `FK-005` preamble (3) — now also names the 88.2 KB / 8.8 KB tools / 13.2 KB system split |
| Option 1 label and text | `carried` | `FK-005` option 1 — plus the constant-scaffolding clause |
| "Smallest of the four: about 5.8 MB" | `carried` | `FK-005` option 1 |
| "reconstruct it by replaying the chain from turn 0" | `carried` | `FK-005` option 1 |
| "Nothing new is built" | `carried` | `FK-005` option 1 |
| Option 2 label and text | `carried` | `FK-005` option 2 — plus the repeated-scaffolding clause and the named scope modification |
| "roughly 2.3× the bytes, about 13 MB instead of 5.8 MB" | `carried` | `FK-005` option 2 |
| Option 3 label and text (the viewer, `curses`, the README/G-006 costs, the one-shot subset) | `carried` | `FK-005` option 3 |
| Option 4 label and text (JSONL, `jq`, the lost `cat`/`grep`/`less` workflow) | `carried` | `FK-005` option 4 — plus the "conversation as a field to extract" clause |
| "Recommendation: none" | `carried` | `FK-005` `Recommendation: none` |
| The question | `carried` + extended | `FK-005` question — now carries the constant-scaffolding clause |
| The four-option structure and each option's identity | `carried` | unchanged; no option added, removed, or re-ranked |

**Nothing was dropped.** Every clause of `FK-004`'s `Presented` block is present in
`FK-005` or is deliberately superseded by a stated replacement (the question's
enumeration, which the added clause extends). No clause could not be dispositioned.

## Compaction Decision

- Prior same-phase files reviewed: `iterations/003-decide.md`, `iterations/008-decide.md`,
  `iterations/010-decide.md`.
- **Superseded**: none.
- **Not superseded**: `iterations/003-decide.md`, `iterations/008-decide.md`,
  `iterations/010-decide.md`.
- **Compaction performed: no.**
- **If no**: 010 is the provenance for `FK-004`'s authored-but-unpresented option set,
  for the `US-003`/`US-004`/`US-005` analysis, and for Answer 1–3 — and `FK-004` is
  preserved in `decisions.md` with its full `Presented` text, so deleting 010 would
  remove the *reasoning* behind a set this pass repairs. Its findings F-6/F-7 are carried
  live into this pass. 008 is the provenance for the original option space, the
  `FK-A`/`FK-B` escalation-floor re-tests, the `EI:` dispositions, and F-1…F-5 — none
  re-derived here. 003 covers the resolved `FK-001`/`FK-002` slice. Different slices in
  all three cases.

## Deferred Questions

- **Which device/filesystem hosts `DUMP_DIR` in production** (Linux ext4/XFS beside
  vLLM, or macOS APFS on the dev box)?
  - Status: DEFERRED for /codify-review
  - Impact: the exact write-amplification factor differs; the directional verdict does not.
  - Agent analysis: `none` — a factual premise, not a user preference.
- **The concrete default of the volume threshold** — DEFERRED to PLAN-SELECTED;
  decide's recommended reading is a single-digit MiB of rendered-text bytes.
- **Whether the viewer, if not chosen now, belongs on `roadmap.md`** — DEFERRED to
  PLAN-SELECTED; it follows from the `FK-005` answer, not from a separate question.
- **Why the real dump's message counts are non-monotonic** (parallel requests sharing a
  system-prompt-derived session key, or client-side context variation) — DEFERRED;
  F-8 records the artifact-visible fact and explicitly does not assert a cause. It
  changes no option and no task.

## Questions Asked

No questions asked — deferred mode. `FK-005` is authored for invite/interview to
present; the user's statement was answered by decide (§Answer 4) rather than reflected
back as a question, because it is answerable from source and from the real dump the user
supplied.

## Validation Performed

| Claim | How verified | Result |
|---|---|---|
| Session identity / handle suffix | `session-identity.json` vs handle | matches (`9bf7afff5573` = first 12 hex of UID) |
| No parent coordination contract | `session-identity.json` has no `parent_session_uid` | confirmed |
| `FK-004` was never presented | `decisions.md` §`FK-004` → `Presented.Report` ("not yet presented"); no invite iteration between 010 and 011 | confirmed |
| The conversation is a named section of the rendering | `dump.py:27-58`, `:45-56` | confirmed — `[message i / role]` |
| The session is keyed on the system prompt | `session.py:93-98` | confirmed — `sha256(system)[:12]` |
| No diff changes `[tools]` or `[system]` | `grep -lE '^[-+].*\[(tools|system)\]' turn_*.diff` over 137 files | **0 of 137** |
| The artifact's turns are not one linear conversation | `grep -o '[0-9]* msgs' index.txt`; block add/remove in `turn_003/004/005.diff` | **42 of 137** transitions go backwards |
| Scaffolding share of a rendered record | `jq -c` over `turn_000.json` vs the 88.2 KB rendered size | tools 8,851 B + system 13,223 B of 88.2 KB ≈ a quarter |
| Messages dominate the rendering | `jq -c '.messages'` → 68,113 B | confirmed |
| No TUI dependency is installed | `pyproject.toml:5-8` | `aiohttp`, `click` only |
| `curses` is stdlib, `difflib` already imported | Python stdlib; `dump.py:5` | confirmed |
| `FK-003`/`FK-004` presented text untouched | `decisions.md` §`FK-003`, §`FK-004` | confirmed |
| No orphaned reservation | `decisions.md` §`US-006`, §`FK-005` | both filled; no `State: reserved` remains |

## Next Step

1. **invite** presents `FK-005` — its question, preamble and four options, verbatim
   from `decisions.md` §`FK-005` → `Presented` — together with §Answer 4 above as the
   landscape. One FK per turn.
2. **interview** resolves it; the one-FK-per-turn contract carries it.
3. **decide** then runs **PLAN-SELECTED** on the chosen option: the `OI:` register
   (three entries, `output_layout`'s replacement statement carrying F-6's definition of
   "full"), `G-006` re-derived on the pick (F-9), the Anchor Map finalization (F-1/F-2
   repairs, plus a `Mode: NEW` viewer row with its own `RB-*` row for options 3 and 4),
   the VP/milestone matrices (F-3/F-4), and the full task-out.
4. **Final approval stays blocked** until `FK-005` resolves, PLAN-SELECTED lands with
   `User Approved: yes` on each `OI:` entry, and the `qa-project` skill exists for
   `QAP-MS001`/`QAP-MS002`.

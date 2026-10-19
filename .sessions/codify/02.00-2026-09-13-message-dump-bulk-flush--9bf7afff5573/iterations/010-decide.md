# 010 — Decide (OPTIONS repair — `FK-003` → `FK-004`)

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
> > User Context: "Would it be practical to keep current D format and just create a viewer in small tui?"
> > Additional user-provided context from this conversation, verbatim: the real dump the user pointed at is reachable at smb://HYPERIPPER._smb._tcp.local/data/buster-dumps/c451c754ae4f (mounted at /Volumes/data/buster-dumps/c451c754ae4f); the user also said "Could it be it depends on the definition of what full is?" and "if it's possible, I would be interested in improving readability for a human."
> > FK-003 is open and is owned by decide. Repair or extend its presented option set as the owner judges necessary, then re-present.

## Session Type / Mode

- Continuing session `02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573`.
  Identity verified against `session-identity.json`: UID
  `9bf7afff-5573-402a-b102-827423a1dded`; handle suffix `9bf7afff5573` = first 12
  hex of the UID; display id `02.00-2026-09-13-message-dump-bulk-flush` matches the
  handle prefix. **No `parent_session_uid`** → the parent coordination contract
  does not apply; there is no parent path to resolve or reconcile.
- **Mode: OPTIONS** (no `Mode:` line in the prompt), narrowed to the one slice the
  prompt names: `FK-003`'s option set. This pass does not re-derive the rest of the
  option space — `FK-001` and `FK-002` are resolved and are treated as settled.
- **Question Mode: deferred.** Decide does not ask. The repaired option set is
  authored as `FK-004` for invite/interview to present.
- **`codify-user-prompt` is not invoked**: it presents one already-recorded FK via
  `AskUserQuestion`, which is not available on this dispatch path, and the FK it
  would present does not exist until this pass authors it.

## Context Conflict Detected

None between session-local sources. One **premise in the orchestrator prompt** was
checked rather than adopted: "keep current D format" reads, on its face, as a
challenge to `FK-002`. It is resolved below (§Answer 1) as a statement about the
artifact's *content* format, not its *file layout* — the file layout is `FK-002`'s
recorded decision and is not re-opened. The reading is stated explicitly in
`FK-004`'s presented preamble so the user can correct it in one word if wrong.

## Changes Since Iteration 008

- NEW: `US-003`, `US-004`, `US-005` reserved by the allocator, filled in place, set
  `recorded` — the three user statements this pass acts on, with the normative and
  factual parts split and each falsifiable claim routed to a verifier.
- NEW: `FK-004` reserved, filled, set `open` — the repaired and extended option set.
- UPDATED: `FK-003` `State: open` → `superseded`, `Superseded-by: FK-004`, with a
  supersession note. Its `Presented` text is **untouched** (it is what the user
  actually saw).
- NEW: **Answer 1** — is "keep the current dump format + a small TUI viewer"
  practical? (yes, with a stated cost, and with a distinction the question elides).
- NEW: **Answer 2** — does the trade-off depend on the definition of "full"? (yes,
  strongly on readability, weakly on bytes; and the plan had already half-resolved it).
- NEW: **Answer 3** — what improving readability costs.
- NEW: Finding **F-6** — `VP-T002-turn0-available` and `EI:.../turn_zero_is_full_body`
  both say "full body" without defining it, and the definition is load-bearing.
- UPDATED: `plan-summary.md` §Current Phase, §Options, §Goal Definition Matrix
  (`G-006` row), §Open Questions 2b, §Quality Backlog pointer.
- UPDATED: `handoff-context.md` §Current Status, §Open Forks, §Resolved Forks, §Notes.
- UNCHANGED: `FK-001`, `FK-002`, every `EI:` disposition, `FK-A`/`FK-B` discretion
  footnotes, the flush-policy cluster (iteration 005), `T001`/`T003`'s trigger half,
  the milestone skeleton, the Anchor Map's pre-existing bindings.

## What the user actually faces

`FK-003` asked *what a record contains*. The user answered with three statements
rather than a pick, and between them they move the question: they add a **reader**
to it ("create a viewer", "improving readability for a human") and they expose an
**undefined term** in it ("the definition of what full is"). Neither is answerable
by re-presenting the same three options, because the set contains no option that
keeps the human-readable artifact *and* adds a viewer — its only viewer option
replaces the readable artifact with a machine record.

| Item | Status after this pass |
|---|---|
| **What the artifact holds and how a human reads it** | **`FK-004` — open, authored, ready to present** |
| `FK-003` | **superseded** (unresolved; the user did not answer it) — its presented text is preserved |
| The definition of "full" | **resolved from the plan surface**, not asked: see §Answer 2 |
| Is "keep current format + TUI viewer" practical? | **answered** (yes; §Answer 1 and §Answer 3) |
| `FK-001`, `FK-002`, `FK-A`, `FK-B`, the `EI:` dispositions | unchanged |
| QA infrastructure setup | carried — still a presentation **topic**, not an FK |

## Answer 1 — "Would it be practical to keep current D format and just create a viewer in small tui?"

**Yes — with one distinction the question elides, and one cost it does not mention.**

**The distinction: "format" is two things, and only one of them can be kept.** The
dump's format is (a) its *file layout* — `turn_000.json` + `turn_NNN.diff` +
`index.txt`, one file per turn — and (b) its *content format* — the summary line per
turn, the full baseline, and the per-turn change text. `FK-002` displaced (a): the
user chose it knowing "our bulks would be single write, not a bunch of writes", and a
flush that writes N per-turn files is the ~50% ceiling they declined. Keeping (a) is
therefore not available. Keeping (b) is fully available: the rendered-text sections,
the per-turn diffs and the summary lines can be concatenated into the one append-only
artifact and appended once per flush. `FK-004`'s option 1 and option 3 are exactly
that.

**The practicality: yes.** The viewer needs no new dependency — Python's standard
library ships `curses`, and `difflib` is already imported at `dump.py:5` and already
used to produce the diffs at `dump.py:87-93`. Reconstruction from the artifact is
mechanical: apply the recorded changes in order from turn 0's baseline. The
repository's dependency set is `aiohttp` + `click` only (`pyproject.toml:5-8`), so
this is the difference between a new *command* in this repository and a new *stack* —
and it is the former.

**The cost it does not mention** is in §Answer 3.

## Answer 2 — "Could it be it depends on the definition of what full is?"

**Yes, and the axis it moves is readability, not bytes — and the plan had already
half-resolved it in a way its own text never says out loud.**

Today the word denotes two different things at once, and this is visible in the real
dump:

| What today calls "full" | Where it comes from | What it is |
|---|---|---|
| `turn_000.json` | `dump.py:82` — `json.dumps(body, indent=2)` | the **raw normalized request body**, every top-level key included (`model`, `max_tokens`, `temperature`, thinking config, …) |
| the thing the diffs are taken against | `dump.py:77` — `_render_body(body)` | the **rendered text** — only `[tools]`, `[system]` and `[message i / role]` sections (`dump.py:27-58`) |

So the artifact's baseline and its change chain are **already in two different
representations**, and they are not interchangeable in either direction:

- `_render_body` is **lossy** — every top-level body key outside `tools`/`system`/
  `messages` is absent from the rendered text (`dump.py:27-58`). So a record holding
  only the rendered text cannot answer "what was actually sent".
- The rendered text is **not** recoverable by hand from the raw body — you need the
  tool's own code to produce it. So a baseline that is only the raw body does not
  satisfy "a reader inspecting only the dump directory, with no additional tooling".

**On bytes the dependence is weak.** On the real dump, turn 0 is 93,886 bytes as raw
JSON and 88.2 KB as rendered text — about a 4% difference. The byte axis the user's
`FK-002` reasoning targets barely notices the definition.

**On readability the dependence is total**, and that is why the question was worth
asking.

**Resolution, and it is not decide's invention: the plan already requires the raw
body.** `plan-summary.md` §Validation Point Matrix `VP-T002-turn0-available` reads
"Turn 0's **full normalized request body** is present in the artifact and recoverable
from it", and `EI:buster_ripper.dump/dump_turn/turn_zero_is_full_body` says "turn 0 is
always written as a full JSON body, never as a diff". Both already mean the raw
body. What neither says is that the **rendered baseline** must also be present — and
without it the first change cannot be read by hand. `FK-004`'s preamble therefore
fixes both: turn 0's raw JSON body **and** its rendered rendering. The cost is one
extra ~88 KB record once per session (≈1.5% of a 5.8 MB session).

This is Finding **F-6** below.

## Answer 3 — what improving readability for a human costs

The user's stated interest (`US-005`, "I would be interested in improving readability
for a human") is an outcome, not a pick, and it is satisfiable at three different
prices. Stated so the choice is priced, not argued:

1. **Free** — make the artifact self-consistent (the §Answer 2 resolution) and keep
   the change-focused content. Reading turn K's full text still means replaying the
   chain, but the replay is now by hand on one representation instead of needing the
   tool's rendering code.
2. **~2.3× the bytes** — every turn's record carries its full text, so no replay is
   ever needed. Measured basis: on the real dump the rendered text stays ≈1,770–2,160
   lines across the session (hunk headers: `turn_001.diff` `@@ -1766,3 +1766,31 @@`,
   `turn_137.diff` `@@ -2141,6 +2141,19 @@`) at ≈51 bytes/line (88.2 KB / 1,768
   lines at turn 0), i.e. ≈100 KB per turn × 137 turns ≈ 13 MB against the 5.79 MB of
   change text actually written. **Estimate, not a measurement** — the bytes/line
   ratio is taken from turn 0 and applied across the session.
3. **A new component** — the viewer, which is what the user asked about. It buys
   navigation and highlighting that neither of the above gives, and it costs a
   component this plan has never scoped, a README that documents a tool rather than a
   format, and a restatement of `G-006`'s "without additional tooling" clause. The
   interactive TUI the user named is the larger form; a one-shot "print turn N" form
   is a strictly cheaper subset of the same option.

**The write-reduction goal is untouched by all three.** `G-001` is about flush
frequency, not record content: for N turns flushed F times the artifact costs F
writes whether it holds diffs or full bodies. Only the *bytes* axis moves, which is
why option 2's cost is stated in bytes rather than in writes.

## Options Analysis

The option set is **extended and repaired**, not replaced. All three of `FK-003`'s
options survive; a fourth is added — the direction the user themselves proposed — and
every option's text now states what "full" means.

`G-006` is the goal being arbitrated. It is OPEN with four alternate readings, and no
repository evidence excludes any of them. The choice remains a **genuine preference**:
the user's stated interest in readability narrows the space but ranks nothing, and the
byte/self-containment trade-off between options 1 and 2 is a real product choice.

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
  (`dump.py:96-97` already computes them).
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
  `FK-002` rationale was about; the change between two turns is no longer shown;
  the `+N/-M` counts become a computed comparison.
- **Complexity**: low. **Risk**: medium — the risk is the byte axis, and the research
  (iteration 007, F1/F2) is explicit that bytes written, not write-call count, is the
  physical wear quantity.
- **Anchor stance delta**: none. No `Mode: NEW` row.

### Option 3 — Change-focused log + small terminal viewer  *(the user's proposal)*

- **What it is**: option 1's artifact, unchanged, plus a small terminal viewer that
  lists a session's turns, shows a turn's reconstructed full text, and highlights
  what changed against the previous turn.
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
  3's new-component cost *and* loses what option 3 keeps; the tool's documented
  purpose moves behind a tool; the smallest records are the least FTL-friendly shape
  per the research (iteration 007, F2).
- **Complexity**: medium. **Risk**: medium-high — it is the only option that removes
  human readability from the artifact itself, which is in tension with `US-005`.
- **Anchor stance delta**: same `Mode: NEW` viewer row as option 3, plus a change to
  `T005`'s content.

### Why four options and not three

The skill's "up to 3 most viable" guidance guards against padding. This set is not
padded: option 4 is the previous set's third option, which the user has already seen
and engaged with — their question ("keep current D format and just create a viewer")
is a direct response to its machine-record framing — and they did not reject it.
Dropping an option a user has seen and questioned would be hiding a genuine
alternative; the alternative is real (it is the only one that is scriptable and
compact). Option 3 is the user's own proposal and is not padding by definition.

Options 3 and 4 share the viewer axis; they differ in what the artifact is without the
viewer. Options 1 and 2 share the plain-text axis; they differ in bytes versus
self-containment. The question names both axes, so a combination not packaged here
(e.g. self-contained records *and* a viewer) is expressible by the user without a
fifth option.

### Rejected — "write a blob" (no readability commitment)

Considered and rejected in iteration 008 and **still rejected**. The cheapest way to
one write per flush is to write the least. That satisfies `G-001` completely and
destroys the tool's documented purpose while every implementation task completes — it
is precisely `G-006`'s failure witness. Not a product choice; the failure mode. Not
presented.

## Findings

### F-6 — "full body" is load-bearing in two plan artifacts and defined in neither

`VP-T002-turn0-available` requires "Turn 0's **full normalized request body** …
present in the artifact and recoverable from it", and
`EI:buster_ripper.dump/dump_turn/turn_zero_is_full_body` states "turn 0 is always
written as a full JSON body, never as a diff". Neither defines "full", and the two
candidate meanings are not interchangeable in either direction (§Answer 2):
`_render_body` is lossy (`dump.py:27-58`), and the rendered text is not hand-derivable
from the raw body.

**PLAN-SELECTED must** (a) write the definition into the `OI:` replacement statement
for `turn_zero_is_full_body` — turn 0's raw normalized JSON body **and** its rendered
rendering are both present — and (b) carry the same definition into `T002`'s done-when
and `VP-T002-turn0-available`'s scope reason. Independent of which option is chosen;
options 1–4 all keep both.

### F-7 — `G-006`'s "without additional tooling" clause is option-dependent

`G-006`'s `Met when` reads "…without additional tooling". Options 1 and 2 satisfy it
as written. Options 3 and 4 do not: the artifact alone no longer answers "where did
this change", so the clause must be re-derived to "…or, for the viewer options, with
the viewer this plan ships". **PLAN-SELECTED must** re-derive `G-006` on the chosen
option, and the `OI:` register must carry the replacement statement for
`EI:.../output_layout` accordingly. Not a user question — it follows from the pick.

### Carried from iteration 008 (unchanged, restated so PLAN-SELECTED does not lose them)

F-1 (`KV_POLL_INTERVAL` has no CLI flag — the flag is a new precedent required by QA
injectability, not a convention), F-2 (`DUMP_FLUSH_BYTES`, not `DUMP_FLUSH_TURNS`),
F-3 (one write-count VP, per-flush form folded in), F-4 (keep `G-001` witness B, the
byte axis), F-5 (`dump.py:63`'s bare `assert`; coverage headroom `U ≤ 3.2 + 0.2·K`).

## Planned Invariants In Scope

**No change from iteration 008.** The three `EI:` entries in the contradiction surface
keep their dispositions (`output_layout` **override**; `sequential_diff_chain`
**refine**; `turn_zero_is_full_body` **refine**), and the pick does not move any of
them:

- `output_layout` **override** — the replacement statement's *record content* is
  written by PLAN-SELECTED from `FK-004`'s answer. F-6 adds the definition of "full"
  to that statement.
- `sequential_diff_chain` **refine** — under option 2 or 4 there is no diff chain in
  the artifact at all; the load-bearing clause is the `turn`/`prev_text` sequencing at
  append time, which `T001` already settles. Unchanged.
- `turn_zero_is_full_body` **refine** — `FK-004`'s preamble keeps turn 0's raw body in
  every option, so the refine holds for all four. F-6 sharpens its wording.

`G-006` ↔ `G-001` remains a goal-level tension, not an invariant one. `FK-004`
arbitrates it; options 3 and 4 additionally move `G-006`'s own text (F-7).

## Fork Authoring Record

- **`US-003`, `US-004`, `US-005`** — reserved via
  `codify codify reserve-decision-record --kind user-statement --owner decide`, filled
  in place, set `recorded`. Each preserves the verbatim words, the source timestamp and
  locator, the normative projection (`none` for the two questions; the conditional
  readability outcome for `US-005`), and each falsifiable claim with its own
  `Verification`/`Evidence`/`Consequence`. All factual claims are routed to
  `assumptions PENDING`; decide's own readings are recorded as `#` annotations inside
  the claim blocks and expanded in §Answer 1 and §Answer 2 — they are not promoted to
  verdicts.
- **`FK-004`** — reserved via `codify codify reserve-decision-record --kind fork
  --owner decide`, filled in place, `State: open`. `Type: preference`;
  `Why-yours` quotes the escalation-floor clause *"it is irreversible or expensive to
  reverse (data loss, public API surface, security)"* verbatim; `Presented` carries the
  exact question, the preamble, all four option labels and descriptions,
  `Recommendation: none`, the timestamp, and the report path.
- **`FK-003`** — `State: open` → `superseded`, `Superseded-by: FK-004`, with a
  supersession note. `Supersedes: —` and the whole `Presented` block are untouched.
- **No reservation was left abandoned.** Four allocator calls this pass
  (`US-003`, `US-004`, `US-005`, `FK-004`); all four were filled.
- **`FK-A` / `FK-B`** remain **not askable** (iteration 008's escalation-floor
  re-tests; unchanged by this pass). Not reserved, not filled.
- **Legacy migration**: not applicable — `decisions.md` is canonical and the handoff's
  Fork sections are indexes only.

## Dropped-Clause Sweep

**Triggered** — this pass wholesale replaces a prior definition of the same scope
(`FK-003`'s presented option set is superseded by `FK-004`'s). One hop only: the direct
predecessor is `FK-003`'s `Presented` block. It has no invariant lineage to walk —
`FK-003` cites `G-006`, whose text is unchanged by this pass.

| `FK-003` `Presented` clause | Disposition | Where it now lives |
|---|---|---|
| "turn 0 embeds the full request body" | `carried` + `restored` | `FK-004` preamble — now explicit that "full" means the raw JSON body **and** the rendered rendering (F-6). The old text left it undefined; that is the restoration |
| "every later turn embeds the unified diff of the rendered request against the previous turn" | `carried` | `FK-004` option 1, option 3 |
| "each turn contributes one summary line" | `carried` | `FK-004` option 1 |
| "Closest to today's reading experience and the fewest bytes written of the three" | `carried` (re-scoped) | `FK-004` option 1 — "smallest of the four", ≈5.8 MB measured on the real dump |
| "A turn's full text is recoverable only by replaying the diff chain from turn 0" | `carried` | `FK-004` option 1 |
| "Every turn's record embeds the complete rendered request" | `carried` | `FK-004` option 2 |
| "each record stands alone and any turn can be read directly with no replay" | `carried` | `FK-004` option 2 |
| "Writes substantially more bytes per session" | `carried` + strengthened | `FK-004` option 2 — now quantified (≈2.3×, ≈13 MB vs ≈5.8 MB) with its measured basis |
| "the change between two turns is visible only by comparing their records" | `carried` | `FK-004` option 2 |
| "One JSON object per turn, appended one per line, read with `jq`" | `carried` | `FK-004` option 4 |
| "The most compact and the most scriptable" | `carried` | `FK-004` option 4 |
| "the documented `cat`/`grep`/`less` workflow stops working" | `carried` | `FK-004` option 4 |
| "a small viewer/reconstruction command must be built" | `carried` | `FK-004` option 4 (and option 3, where it is additive rather than required) |
| "the README documents a tool instead of a format" | `carried` | `FK-004` options 3 and 4 |
| "Adds a new component to the codebase that this plan has not scoped" | `carried` | `FK-004` options 3 and 4 |
| "Recommendation: none" | `carried` | `FK-004` `Recommendation: none` — evidence still does not rank the options |
| The question's readability commitment ("a reader can still locate a changed turn and see what changed about it") | `carried` | `FK-004`'s question ("how should a human read it") and every option's text; the substance is `G-006` |

**Nothing was dropped.** Every clause of `FK-003`'s `Presented` block is present in
`FK-004` or is deliberately superseded by a stated replacement (the "of the three"
cardinality, which the fourth option makes false). No clause could not be dispositioned.

## Compaction Decision

- Prior same-phase files reviewed: `iterations/003-decide.md`, `iterations/008-decide.md`.
- **Superseded**: none.
- **Not superseded**: `iterations/003-decide.md`, `iterations/008-decide.md`.
- **Compaction performed: no.**
- **If no**: 008 is the provenance for `FK-003`'s option analysis, the `FK-A`/`FK-B`
  escalation-floor re-tests, the `EI:` dispositions, and findings F-1…F-5 — all still
  live and carried forward, none re-derived here. Its analysis of the three original
  options is what `FK-004`'s repair builds on, so it is not superseded, and no FK it
  originated is unresolved in a way that would make it unsafe to keep. `FK-003` itself
  is preserved in `decisions.md` with its full `Presented` text, so deleting 008 would
  remove the *reasoning* behind a still-open option set. 003 covers the resolved
  `FK-001`/`FK-002` slice. Different slices in all three cases.

## Deferred Questions

- **Which device/filesystem hosts `DUMP_DIR` in production** (Linux ext4/XFS beside
  vLLM, or macOS APFS on the dev box)?
  - Status: DEFERRED for /codify-review
  - Impact: the exact write-amplification factor differs; the directional verdict does not.
  - Agent analysis: `none` — a factual premise, not a user preference.
- **The concrete default of the volume threshold** — DEFERRED to PLAN-SELECTED;
  decide's recommended reading is a single-digit MiB of rendered-text bytes.
- **Whether the viewer, if not chosen now, belongs on `roadmap.md`** — DEFERRED to
  PLAN-SELECTED; it follows from the `FK-004` answer, not from a separate question.

## Questions Asked

No questions asked — deferred mode. `FK-004` is authored for invite/interview to
present; the user's three statements were answered by decide (§Answer 1–3) rather than
reflected back as questions, because each is answerable from source and from the real
dump the user supplied.

## Validation Performed

| Claim | How verified | Result |
|---|---|---|
| Session identity / handle suffix | `session-identity.json` vs handle | matches (`9bf7afff5573` = first 12 hex of UID) |
| No parent coordination contract | `session-identity.json` has no `parent_session_uid` | confirmed |
| The real dump is reachable and is a real session | `ls` + `wc -c` over `/Volumes/data/buster-dumps/c451c754ae4f` | confirmed — 137 turns |
| `turn_000.json` is the **raw** body, not the rendering | `dump.py:82` (`json.dumps(body, indent=2)`) + `turn_000.json` head | confirmed |
| The diffs are taken on `_render_body` output | `dump.py:77`, `:87-95` | confirmed |
| `_render_body` is lossy for top-level keys | read `dump.py:27-58` — only `tools`, `system`, `messages` are emitted | confirmed |
| Turn 0's "full" as rendered text ≈88 KB vs 93,886 bytes raw | `index.txt` line 1 ("FULL … 88.2 KB") vs `wc -c turn_000.json` | confirmed — ≈4% apart |
| The rendered text stays bounded across the session | hunk headers of `turn_001/050/100/136/137.diff` | ≈1,768 → ≈2,159 lines; bounded |
| Change text totals 5.79 MB over 137 turns | `cat turn_*.diff \| wc -c` | 5,785,285 bytes |
| No TUI dependency is installed | `pyproject.toml:5-8` | `aiohttp`, `click` only |
| `curses` is stdlib, `difflib` already imported | Python stdlib; `dump.py:5` | confirmed |
| The plan already requires the raw body in the artifact | `plan-summary.md` §Validation Point Matrix `VP-T002-turn0-available`; §Invariant Reconciliation `EI:.../turn_zero_is_full_body` | confirmed — neither defines "full" (F-6) |
| `G-006`'s "without additional tooling" clause | `plan-summary.md` §Goal Definition Matrix `G-006` | confirmed — option-dependent (F-7) |
| `FK-003` is unresolved | `decisions.md` §`FK-003` → `Resolution` all `—` | confirmed — superseded, not answered |
| No orphaned reservation | `decisions.md` §`US-003`/`US-004`/`US-005`/`FK-004` | all four filled; no `State: reserved` remains |

## Next Step

1. **invite** presents `FK-004` — its question, preamble and four options, verbatim
   from `decisions.md` §`FK-004` → `Presented` — together with the three answers above
   as the landscape. One FK per turn.
2. **interview** resolves it; the one-FK-per-turn contract carries it.
3. **decide** then runs **PLAN-SELECTED** on the chosen option: the `OI:` register
   (three entries, `output_layout`'s replacement statement carrying F-6's definition of
   "full"), `G-006` re-derived on the pick (F-7), the Anchor Map finalization
   (F-1/F-2 repairs, plus a `Mode: NEW` viewer row with its own `RB-*` row for options
   3 and 4), the VP/milestone matrices (F-3/F-4), and the full task-out.
4. **Final approval stays blocked** until `FK-004` resolves, PLAN-SELECTED lands with
   `User Approved: yes` on each `OI:` entry, and the `qa-project` skill exists for
   `QAP-MS001`/`QAP-MS002`.

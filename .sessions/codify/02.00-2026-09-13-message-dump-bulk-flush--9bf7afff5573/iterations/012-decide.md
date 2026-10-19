# 012 — Decide (OPTIONS repair — `FK-005` → `FK-006`, switchable encoding)

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
> > User Context: "can we implment additional formats and keep teh current?" — and, asked which reading was meant, the user replied: "I'm for switchable option". The user's intent is that the artifact's record format be selectable (one artifact per session, one write per flush, format chosen by configuration), with the current change-focused format remaining available — not several formats written simultaneously.
> > FK-005 is open and owned by decide. Repair or extend its presented option set as the owner judges necessary, then re-present.

## Session Type / Mode

- Continuing session `02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573`.
  Identity verified against `session-identity.json`: UID
  `9bf7afff-5573-402a-b102-827423a1dded`; handle suffix `9bf7afff5573` = first 12
  hex of the UID; display id `02.00-2026-09-13-message-dump-bulk-flush` matches the
  handle prefix. **No `parent_session_uid`** → the parent coordination contract does
  not apply; there is no parent path to resolve or reconcile.
- **Mode: OPTIONS** (no `Mode:` line in the prompt), narrowed to the one slice the
  prompt names: the presented option set. `FK-001` and `FK-002` are resolved and
  treated as settled; no other slice is re-derived.
- **Question Mode: deferred.** Decide does not ask. The repaired set is authored as
  `FK-006` for invite/interview to present.
- **`codify-user-prompt` is not invoked**: it presents one already-recorded FK via
  `AskUserQuestion`, which is not available on this dispatch path, and the FK it
  would present (`FK-006`) is authored by this pass.

## Context Conflict Detected

**One conflict, between the prompt and the skill — resolved for the skill.**

- **Source A (prompt, priority 5)**: "FK-005 is open and owned by decide. **Repair
  or extend its presented option set** as the owner judges necessary, then re-present."
  Read literally, this says the repair lands inside `FK-005`.
- **Source B (skill, `decide-interactive` → Fork Authoring / `codify-agents` →
  Fork Records)**: "never overwrite prior `Presented` text. A materially changed
  question gets a newly reserved FK, or the old record becomes `State: superseded`
  with `Superseded-by: FK-NNN` and the successor records `Supersedes: FK-NNN`."
- **Resolution**: following the skill. `FK-005`'s `Presented` block is written and
  timestamped (2026-09-13T21:46:34Z) even though it was never shown, and the
  question changes materially (mutual exclusivity → switchability). The repair
  therefore lands as **`FK-006`**, and `FK-005` becomes `superseded`. The prompt's
  intent — a repaired, re-presentable option set for this slice — is satisfied; only
  the handle the repair lands on differs. This is the same handling iteration 011
  applied to `FK-004`, and the reason is the same: overwriting in place would leave
  `FK-005`'s `Report: iterations/011-decide.md` pointer describing text that no
  longer exists.

**One premise in the prompt was checked rather than adopted.** The prompt's
user-context field asserts "The user's intent is that the artifact's record format be
selectable (one artifact per session, one write per flush, format chosen by
configuration), with the current change-focused format remaining available — not
several formats written simultaneously." Only the two quoted strings are user words
(`US-007`). The gloss is orchestrator-worded (priority 5). Its clauses are
dispositioned in `US-007`: "one artifact per session, one write per flush" is
`FK-002`, already settled and independently recorded; "not several formats written
simultaneously" is the one clause that is neither user words nor prior settled state,
and it is treated as a normative design constraint **only** because it is the reading
under which "switchable option" carries information — a switch selects one. If the
user meant simultaneous output, that is a different and much larger change, and the
`FK-006` preamble states the single-encoding constraint in the first position so the
user can correct it in one clause.

**One staleness check, as required.** `FK-005`'s `Trigger.Source` citations were
re-verified this pass rather than carried: `dump.py:72-101` (the whole record path),
`config.py:16` and `:44`, `main.py:23-24`, `README.md:146-162`, `pyproject.toml:5-8`,
and the real dump's measurements all resolve as cited. The `FK-005` citation
`pyproject.toml:5-8` → dependencies `aiohttp` + `click` only is confirmed.

## Changes Since Iteration 011

- NEW: `US-007` reserved by the allocator, filled in place, set `recorded` — the
  user's third planning-round statement, with its normative projection, its
  orchestrator-gloss provenance, and three falsifiable claims each routed to a
  verifier.
- NEW: `FK-006` reserved, filled, set `open` — the repaired presented set.
- UPDATED: `FK-005` `State: open` → `superseded`, `Superseded-by: FK-006`, with a
  supersession note. Its `Presented` text is **untouched**.
- NEW: **Answer 5** — can additional formats be implemented while keeping the
  current one? (yes; and what it costs).
- NEW: Finding **F-10** — `FK-005` option 4's "most compact" is **false**: a
  JSON-encoded change record measures **1.047×** the plain-text change log.
- NEW: Finding **F-11** — `F-9` generalizes: with the encoding configurable,
  `G-006`'s "without additional tooling" clause and `EI:.../sequential_diff_chain`'s
  refine both become **encoding-dependent**, not merely option-dependent.
- NEW: Finding **F-12** — the encoding switch **breaks the artifact's
  self-description**: a reader can no longer tell from `DUMP_DIR/<sid>/` what the
  artifact holds. This is the coupling the decomposition hides, and it needs a
  checked home.
- NEW: Planned invariant **`PI:T002/artifact_encoding/encoding_is_per_artifact`**.
- UPDATED: `plan-summary.md` §Current Phase, §Options, §Goal Definition Matrix
  (`G-006` row), §Open Questions 2b, §Approved Override Register (note only).
- UPDATED: `handoff-context.md` §Current Status, §Open Forks, §Resolved Forks, §Notes.
- UNCHANGED: `FK-001`, `FK-002`, `FK-003`, `FK-004`, `US-003`/`US-004`/`US-005`/
  `US-006`, every `EI:` disposition, `FK-A`/`FK-B`, the flush-policy cluster, the
  milestone skeleton, the Anchor Map's pre-existing bindings, `RB-001`–`RB-003`,
  findings F-1…F-9.

## What the user actually faces

| Item | Status after this pass |
|---|---|
| **What the switchable format offers** | **`FK-006` — open, authored, ready to present** |
| `FK-005` | **superseded** (never presented) — its presented text is preserved |
| Is the format a configuration choice? | **answered** (yes; §Answer 5) |
| `FK-003`, `US-003`, `US-004`, `US-005`, `US-006` | unchanged |
| `FK-001`, `FK-002`, `FK-A`, `FK-B`, the `EI:` dispositions | unchanged |
| QA infrastructure setup | carried — still a presentation **topic**, not an FK |

**Why this is still a question and not a work item.** `US-007` settles the *shape*
(the encoding is a configuration choice; the current one is kept) and not the
*contents* (which encodings exist). "Additional formats" is plural and names no set.
The three encodings the user has been shown — change-focused, self-contained, machine
record — are three different amounts of shipped, documented, tested surface, and the
viewer is a component the plan has never scoped. That is a genuine preference with
materially different outcomes, so it is asked rather than defaulted. What is *not*
asked is the default encoding: `US-007` says "keep teh current", which fixes it, and
the preamble states it in the second position so the user can correct it in one clause.

## Answer 5 — "can we implment additional formats and keep teh current?"

**Yes, and it is a branch on a configured value rather than a new mechanism.** Four
parts: what the current format actually is, what "additional formats" costs, what it
does not cost, and the one thing it breaks.

**1. The whole record path is already one function with one branch.** `dump_turn`
(`dump.py:72-101`) builds the record: at `state.turn == 0` it writes the full
normalized JSON body; at `turn ≥ 1` it writes a unified diff of `_render_body(body)`
against the previous turn's rendering. The summary line is computed in both branches
and appended. There is no other write site on this path. An encoding parameter
therefore selects which record this function *builds* — it does not add a write, a
file, or a trigger.

**2. "Keep the current" is exactly preservable.** The change-focused record is the
`turn ≥ 1` branch and the summary line; making it the default means the no-argument
behaviour is byte-for-byte what it is today (modulo the settled collapse of the three
files into one, which is `FK-002`'s change, not this one). Nothing about the current
reading workflow changes for a user who configures nothing.

**3. What additional encodings cost — measured, not argued.** On the real dump the
user pointed at (`/Volumes/data/buster-dumps/c451c754ae4f`, 137 turns):

| Encoding | Bytes over 137 turns | Ratio to the change log | Human-readable with `cat`/`grep`/`less` |
|---|---|---|---|
| change-focused (current, default) | 5,785,285 (raw diff text) | 1.000× | yes |
| self-contained full text | ≈13 MB (`FK-005`'s recorded estimate; carried) | ≈2.3× | yes |
| machine record (JSON, one object per turn) | **6,061,579** (measured this pass) | **1.047×** | **no** |

The machine-record figure is the sum of `jq -Rs .` over every `turn_*.diff`, i.e. the
same change content JSON-escaped, plus the per-line framing. **It is larger than the
plain-text change log, not smaller** — see F-10.

**4. What it does not cost.** The write-count goal (`G-001`) is untouched: one
artifact per session, one write per flush, `F` writes for N turns, `F` versus `2N`
today, reduction `1 − F/(2N)` with the same 50% floor. The encoding is a property of
what a record *contains*, and the flush already writes "the pending records" as one
operation. Nor does it cost a dependency: every candidate encoding is `json` +
`difflib`, both already imported at `dump.py:5-7`, and `pyproject.toml:5-8` lists
`aiohttp` + `click` only.

**5. The one thing it breaks — and it is not obvious.** Today the artifact's *name and
layout* announce its content: a directory holding `turn_000.json` and `turn_NNN.diff`
tells a reader what it holds. Once the encoding is a configuration choice, that
self-description is gone — the artifact can be a change log, a full-text log, or a
line-delimited JSON stream, and `DUMP_DIR/<sid>/` looks the same in all three. This
is finding **F-12**, and it is the concrete price of the switch: the plan must give the
artifact a way to say what it is (the natural candidate is the filename/extension,
which `VP-T002-collapsed-artifact` already inspects).

## Options Analysis

The set is **repaired, not extended**: three options, each a *shipped set* of
encodings plus the viewer question, because the two are the two live user questions
and the plan must answer both. `FK-005`'s four mutually exclusive positions become:
the change-focused encoding is no longer an option — it is the fixed default
(`US-007`); the self-contained encoding is in every option; the viewer and the machine
record are the two things that vary. `Recommendation: none` is carried.

`G-006` is still the goal being arbitrated, and `US-007` narrows it without ranking
it: the user has said the encoding should be choosable and the current one kept, which
excludes only encodings that would *replace* the current one. `US-005`'s interest in
human readability argues against the machine record as the *only* format and not
against it existing. No repository evidence ranks the three options.

### Option 1 — Two text formats, no viewer

- **What it is**: the encoding becomes a configuration choice between the current
  change-focused record (default) and a self-contained record carrying every turn's
  complete rendered text.
- **What it changes**: `dump_turn` gains an encoding branch; `config` gains an
  encoding global; `main.py` gains its flag; the README documents two encodings.
- **What stays the same**: one artifact per session, one write per flush, arrival
  order, the summary line, turn 0's full body, the default reading workflow,
  `cat`/`grep`/`less` on both encodings.
- **Pros**: smallest realization of what the user asked for; both encodings stay
  human-readable, so `US-005`'s interest is met without a new component; `G-006`'s
  "without additional tooling" clause survives as written; no `Mode: NEW` anchor
  beyond the encoding selector.
- **Cons**: the self-contained encoding costs ≈2.3× the bytes when selected; the
  artifact no longer announces its own encoding (F-12) — a filename or header
  decision becomes load-bearing; two serializers and two test surfaces.
- **Complexity**: low-medium. **Risk**: low — the only irreversible part is shipping
  a documented encoding, which is what the user asked for.
- **Anchor stance delta**: `collapsed dump artifact` row's serializer becomes
  encoding-parameterized (still `NEW`); **adds** a `Mode: NEW` config row for the
  encoding selector + its CLI flag. No viewer row.

### Option 2 — Two text formats + small terminal viewer

- **What it is**: option 1's two encodings and default, plus the viewer `US-003`
  asked about: lists a session's turns, shows a turn's reconstructed full text,
  highlights the change against the previous turn.
- **What it changes**: the plan gains a read-side command; `README.md:146-162` is
  rewritten to document a tool as well as the encodings.
- **What stays the same**: everything option 1 keeps, including `grep`/`less`
  working on the artifact; the artifact remains the primary record.
- **Pros**: navigation neither encoding gives on its own — the change-focused one
  makes you replay the chain, the self-contained one makes you find a turn by
  scrolling; zero dependency cost (stdlib `curses`; `difflib` already imported at
  `dump.py:5`); it is the direction the user raised in `US-003` and `US-005` asks for
  readability.
- **Cons**: a component this plan has never scoped, needing authorization and its own
  complete `RB-*` row; `G-006`'s "without additional tooling" clause must be re-derived
  (F-9/F-11); more test surface; the README documents a tool.
- **Complexity**: medium. **Risk**: medium — the risk is scope, not correctness: a
  read-side component arrives in a session whose purpose is a write-side change, and
  `RB-003`'s read-path search found no code reader to build on, so it is a component
  from zero.
- **Anchor stance delta**: option 1's rows, **plus** a `Mode: NEW` viewer row
  requiring its own complete `RB-*` row; `T005` becomes tool documentation.

### Option 3 — Two text formats + machine record + viewer

- **What it is**: option 2 plus a third encoding — one JSON object per turn, one per
  line, for `jq` and cross-session aggregation.
- **What it changes**: a third serializer; the README documents three encodings;
  `G-006`'s clause becomes **encoding-dependent** rather than restated once.
- **What stays the same**: the default and the reading workflow; the write count.
- **Pros**: the only encoding a script reads without parsing diffs; the cheapest form
  for aggregating across sessions; its incremental implementation cost is small now
  that the encoding is already a parameter.
- **Cons**: **it does not save bytes** — measured 1.047× the change log (F-10); for
  that encoding the artifact stops being human-readable, so `G-006`'s clause must be
  qualified per encoding (F-11); a third README section and a third test surface.
- **Complexity**: medium. **Risk**: medium — the risk is that a documented,
  non-human-readable encoding invites sessions recorded in a form the documented
  reading workflow cannot read.
- **Anchor stance delta**: option 2's rows, plus the encoding selector's enum grows;
  no additional `Mode: NEW` row.

### Why three options and not four

The uncovered combination — three encodings without the viewer — is a real
combination and is deliberately **not** packaged as a fourth option, because the
viewer is orthogonal to the encoding set and the skill's own rule applies: a
combination not packaged in the list is expressible by the user without an extra
option. The `FK-006` preamble names the two axes explicitly so the user can attach
either choice in one clause ("option 3 without the viewer").

### Rejected — "write a blob" (no readability commitment)

Considered and rejected in iterations 008, 010 and 011, and **still rejected**. The
cheapest way to one write per flush is to write the least. That satisfies `G-001`
completely and destroys the tool's documented purpose while every implementation task
completes — it is precisely `G-006`'s failure witness. `US-007` adds nothing to it:
"switchable" is a choice among readable encodings, and a switch whose positions
include "write nothing readable" is the failure mode with a flag. Not presented.

### Rejected — "make a new encoding the default"

The prompt's gloss offered this as a reading of the user's intent. It is not
presented as an option because `US-007`'s own words settle it: "keep teh current"
fixes the current encoding's position, and the second position of the preamble states
the default and invites a one-clause correction. Presenting it as an option would
manufacture a choice the user's words already made.

## Findings

### F-10 — `FK-005` option 4's "most compact" is false

`FK-005` option 4 described the machine record as "The most compact and the most
scriptable". Measured on the real dump, a JSON-encoded change record is
**6,061,579 bytes against 5,785,285** for the plain-text change log — **1.047×**, i.e.
marginally *larger*, before any per-line framing. The claim is corrected in
`FK-006`'s third option and in its preamble (4). The encoding's real advantage is
that a script reads it without parsing diffs, not that it saves space. **Consequence
for PLAN-SELECTED**: do not carry the superlative into `T002`'s or `T005`'s text, and
do not let a "compactness" argument decide the encoding set.

### F-11 — `G-006`'s clause and `EI:.../sequential_diff_chain` become encoding-dependent (generalizes F-9)

`F-9` established that `G-006`'s "a reader inspecting only `DUMP_DIR/<sid>/` … without
additional tooling" is *option*-dependent. `US-007` makes it *encoding*-dependent:
under the change-focused encoding it holds as written; under the self-contained one
the reader can still locate a turn and read it, but the "what changed" half becomes a
comparison the reader performs; under the machine record it fails entirely (`cat`,
`grep`, `less` do not work); and if a viewer ships, the clause must admit the tool.
`EI:.../sequential_diff_chain`'s refine has the same shape: "each turn's record derives
from the previous turn's rendered text, in arrival order" is true of the change-focused
encoding only. **PLAN-SELECTED must** write `G-006`'s `Met when` per encoding (or
qualify it), and must qualify the `OI:` replacement statement for `output_layout`
accordingly. Not a user question — it follows from the pick.

### F-12 — the encoding switch breaks the artifact's self-description

Today `DUMP_DIR/<sid>/` is self-describing: `turn_000.json` plus `turn_NNN.diff` plus
`index.txt` *is* the layout, and `README.md:146-162` documents that reading. Once the
encoding is a configured choice, three different artifacts produce the same directory
listing and a reader cannot tell which they are holding. This is exactly the coupling
a decomposition hides: the encoding decision and the reader are one relation, and
splitting them into "a config global" and "a README section" makes the relation
invisible — no single piece is malformed, and nothing reports a defect. **Where it
must be checked**: the artifact must carry its encoding in something the reader sees
without a tool — the natural candidate is the artifact's filename/extension, which
`VP-T002-collapsed-artifact` already inspects for "exactly one artifact file".
**PLAN-SELECTED must** name the encoding in the artifact's name (or an equivalent
in-band marker) and extend that VP's "exactly one artifact" clause to "exactly one
artifact, whose name states its encoding". Recommended reading for forward: the
artifact's extension carries the encoding (`.log` for the text encodings,
distinguished, and `.jsonl` for the machine record). User-visible but cheap and
reversible; recorded here rather than asked.

### Carried from iterations 008, 010 and 011 (unchanged, restated so PLAN-SELECTED does not lose them)

**F-8** — the artifact's turns are **not one linear conversation** (42 of 137
transitions go backwards); no encoding needs a thread, but any merged-thread view
would need a grouping rule the plan lacks, and `T005`'s README text must record the
limitation. **F-6** — "full body" is load-bearing in `VP-T002-turn0-available` and
`EI:.../turn_zero_is_full_body` and defined in neither; define it in the `OI:`
statement and `T002` as turn 0's raw normalized JSON body **and** its rendered
rendering (all three `FK-006` options keep both). **F-1** — the Anchor Map's *flush
trigger parameters* row claims every `config` global has a CLI flag in `main.py`;
**false** (`KV_POLL_INTERVAL`, `config.py:44`, has none), so the flag is required by
QA injectability, not convention — and the **encoding selector's** flag is new
surface for the same reason. **F-2** — that row names `DUMP_FLUSH_TURNS` against
`FK-001`'s **data-volume** basis; it is `DUMP_FLUSH_BYTES`. **F-3** — keep one
write-count VP, per-flush form folded in. **F-4** — keep `G-001` witness B (the byte
axis) live. **F-5** — `dump.py:63`'s bare `assert`; coverage headroom `U ≤ 3.2 + 0.2·K`.
**F-9** — superseded in scope by F-11 (it is the same clause, now encoding-dependent).

## Planned Invariants In Scope

### PI:T002/artifact_encoding/encoding_is_per_artifact

- **Statement**: The record encoding is chosen once per process and is constant for
  the life of a session's artifact; one artifact carries exactly one encoding, and the
  artifact states which one it is.
- **Reason**: A directory that can hold three encodings is no longer readable by a
  reader who does not already know which one it holds (F-12). The flush, the summary
  line, and every reader are coupled through this; making it a rule gives the coupling
  a checked home.
- **Strength**: hard_requirement.

### PI:T002/artifact_encoding/encoding_does_not_change_write_count

- **Statement**: The selected encoding changes what a record contains and not how many
  writes a session performs — one artifact, one write per non-empty flush.
- **Reason**: `US-007` makes the encoding a knob; the session's whole purpose
  (`G-001`) is the write count. Without this rule a future encoding could reintroduce
  a per-turn write and no gate would notice, because the write-count VPs are written
  against the change-focused encoding.
- **Strength**: hard_requirement.

### Conflict matrix

| Planned invariant | Existing invariant | Relationship | Decision | Why |
|---|---|---|---|---|
| `PI:T002/artifact_encoding/encoding_is_per_artifact` | `EI:buster_ripper.dump/dump_turn/output_layout` | contradicts | override | `output_layout` names one layout; the replacement now names a *family* of layouts selected by configuration, and `FK-002`'s "one collapsed artifact" survives inside it |
| `PI:T002/artifact_encoding/encoding_does_not_change_write_count` | `EI:buster_ripper.dump/dump_turn/sequential_diff_chain` | refines | refine | the diff chain is a property of the change-focused encoding only; the append-time `turn`/`prev_text` sequencing it protects is what must not move, and that is encoding-independent |
| `PI:T002/artifact_encoding/encoding_is_per_artifact` | `EI:buster_ripper.dump/dump_turn/turn_zero_is_full_body` | supports | preserve | every encoding keeps turn 0's raw body and rendering; the invariant is encoding-independent (F-6 sharpens its wording) |

**No new `OI:` entry is created by this pass.** `FK-002` already supplies the override
for `output_layout`; `US-007` changes the *replacement statement's* shape from one
layout to a configured family, which PLAN-SELECTED writes with `User Approved: yes`.
The three `EI:` dispositions are otherwise unchanged. No invariant registry exists in
this repository, so the `Registry` impact of every record is `none`.

## Fork Authoring Record

- **`US-007`** — reserved via `codify codify reserve-decision-record --kind
  user-statement --owner decide`, filled in place, set `recorded`. It preserves the
  verbatim words (both strings, typos included), the source timestamp and locator, the
  provenance of the orchestrator-worded gloss as distinct from the user's words, the
  normative projection, and three falsifiable claims each with its own
  `Verification`/`Evidence`/`Consequence`, all routed to `assumptions PENDING`.
- **`FK-006`** — reserved via `codify codify reserve-decision-record --kind fork
  --owner decide`, filled in place, `State: open`. `Type: preference`;
  `Why-yours` quotes the escalation-floor clause *"it is irreversible or expensive to
  reverse (data loss, public API surface, security)"* verbatim (interview →
  "Escalation floor"); `Presented` carries the exact question, the four-position
  preamble, all three option labels and descriptions, `Recommendation: none`, the
  timestamp, and the report path.
- **`FK-005`** — `State: open` → `superseded`, `Superseded-by: FK-006`, with a
  supersession note. Its `Presented` block is untouched. **Why supersede rather than
  repair in place**: the skill forbids overwriting prior `Presented` text, and
  `Presented` is written *before* showing precisely so the record of what would be
  shown survives a repair. `FK-005` was never presented, so superseding loses nothing
  the user saw; overwriting in place would leave its
  `Report: iterations/011-decide.md` pointer describing text that no longer exists.
- **No reservation was left abandoned.** Two allocator calls this pass (`US-007`,
  `FK-006`); both were filled.
- **`FK-A` / `FK-B`** remain **not askable** (iterations 008 and 010's escalation-floor
  re-tests; unchanged by this pass). Not reserved, not filled.
- **Legacy migration**: not applicable — `decisions.md` is canonical and the handoff's
  Fork sections are indexes only.

## Dropped-Clause Sweep

**Triggered** — this pass wholesale replaces a prior definition of the same scope
(`FK-005`'s presented set is superseded by `FK-006`'s). One hop only: the direct
predecessor is `FK-005`'s `Presented` block, whose invariant lineage is
`EI:.../output_layout` (still in the contradiction surface, still an override target —
no `Superseded-by` predecessor to walk).

| `FK-005` `Presented` clause | Disposition | Where it now lives |
|---|---|---|
| "(1) one append-only file per session, written once per bulk flush" | `carried` | `FK-006` preamble (1), (3) — now also stated as "exactly one encoding" |
| "(2) 'Full' is defined … raw JSON body **and** its rendered rendering" | `carried` | `FK-006` §Findings F-6 (restated); PLAN-SELECTED writes it into the `OI:` statement and `T002` |
| "(3) the numbers … 137 turns, 93,886 bytes raw body, 5,785,285 bytes of change text" | `carried` + extended | `FK-006` preamble (3) — now also the measured machine-record figure 6,061,579 bytes |
| "88.2 KB rendered turn 0 / 8.8 KB tools / 13.2 KB system" | `carried` | `FK-006` option 1's "≈2.3× / ≈13 MB" (the figure it grounds); F-11's reasoning |
| Option 1 "Change-focused log, plain text tools" (label, text, ≈5.8 MB, replay cost, "nothing new is built") | `carried` as the **fixed default**, not an option | `FK-006` preamble (2) + every option's "stays the default"; `US-007` moved it out of the choice set |
| Option 2 "Self-contained log, plain text tools" (label, text, ≈2.3×, no-diff-shown cost, conversation-scope modification) | `carried` | `FK-006` option 1 (present in all three options); the conversation-scope modification is **dropped as a named modification** — see below |
| Option 3 "Change-focused log + small terminal viewer" (the viewer, `curses`, `difflib`, the README/`G-006` costs, the one-shot subset) | `carried` | `FK-006` option 2, verbatim in substance |
| Option 4 "Machine record + viewer" (JSONL, `jq`, the lost `cat`/`grep`/`less` workflow, the viewer-required cost) | `carried` **and corrected** | `FK-006` option 3 — "most compact" is **removed** and replaced with the measured 1.047× (F-10); "viewer required" becomes "`G-006`'s clause becomes encoding-dependent" (F-11) |
| "Recommendation: none" | `carried` | `FK-006` `Recommendation: none` |
| The question (pick exactly one of four) | **dropped** — deliberate | Replaced by "what should the switchable format offer"; `US-007` is the user's answer that no single choice exists. This is the one clause whose *form* does not survive, and it is the point of the repair |
| The constant-scaffolding clause and §Answer 4 (the conversation is the `[message i / role]` sections) | `carried` | `FK-006`'s options all carry the conversation; §Answer 4 stands in `iterations/011-decide.md` and its claims stay `assumptions PENDING` |

**One deliberate drop, named.** `FK-005`'s preamble offered "scope the record to the
conversation alone (lift the constant tool definitions and system prompt out)" as a
**named modification** the user could attach to either encoding. It is **not** carried
into `FK-006`'s preamble. Reason: `US-007` makes the encoding a first-class configured
choice, and a conversation-scoped variant is a *fourth* encoding axis (scope), not a
clause on an existing one — carrying it would add a dimension to a question the user
has already declined to answer once per round. It is recorded here and in
`handoff-context.md` §Notes as a **PLAN-SELECTED candidate** (the ≈25% saving is real
only for the self-contained encoding, and it is the one encoding where the scaffolding
repeats). Not lost, not presented.

**Nothing else was dropped.** Every other clause of `FK-005`'s `Presented` block is
present in `FK-006` or is deliberately replaced with the stated successor. No clause
could not be dispositioned.

## Compaction Decision

- Prior same-phase files reviewed: `iterations/003-decide.md`, `iterations/008-decide.md`,
  `iterations/010-decide.md`, `iterations/011-decide.md`.
- **Superseded**: none.
- **Not superseded**: all four.
- **Compaction performed: no.**
- **If no**: each is a different slice with live content. `003` covers the resolved
  `FK-001`/`FK-002` slice; `008` is the provenance for the original option space, the
  `FK-A`/`FK-B` escalation-floor re-tests, the `EI:` dispositions and F-1…F-5; `010`
  is the provenance for `US-003`/`US-004`/`US-005` and Answer 1–3; `011` is the
  provenance for `US-006`, Answer 4 and F-6/F-8/F-9. `FK-004` and `FK-005` are
  preserved in full in `decisions.md`, so the *records* survive — but their
  *reasoning* (why the viewer is practical, why "full" needed defining, why the
  conversation question was answered rather than asked) lives only in 010 and 011,
  and `FK-006`'s options restate conclusions from it. Deleting any of them would
  remove reasoning a live option still rests on.

## Deferred Questions

- **Whether the encoding belongs in the artifact's filename or an in-band marker**
  (F-12) — DEFERRED to PLAN-SELECTED; decide's recommended reading is the
  filename/extension, because `VP-T002-collapsed-artifact` already inspects the
  directory listing and a marker line would be a second thing a reader must know to
  skip.
- **Whether the conversation-scoped encoding variant is wanted** — DEFERRED to
  PLAN-SELECTED as a candidate, not an option (see §Dropped-Clause Sweep).
- **Which device/filesystem hosts `DUMP_DIR` in production** — unchanged, DEFERRED
  for /codify-review; a factual premise, `Agent analysis: none`.
- **The concrete default of the volume threshold** and **the interval value** —
  unchanged, DEFERRED to PLAN-SELECTED as Forward discretion footnotes (`FK-A`,
  `FK-B`).
- **Why the real dump's message counts are non-monotonic** — unchanged; F-8 records
  the artifact-visible fact and explicitly does not assert a cause.
- **QA infrastructure setup** — unchanged; a presentation **topic**, not an FK.

## Questions Asked

No questions asked — deferred mode. `FK-006` is authored for invite/interview to
present; the user's feasibility question was answered by decide (§Answer 5) rather
than reflected back as a question, because it is answerable from source and from the
real dump the user supplied.

## Validation Performed

| Claim | How verified | Result |
|---|---|---|
| Session identity / handle suffix | `session-identity.json` vs handle | matches (`9bf7afff5573` = first 12 hex of UID) |
| No parent coordination contract | `session-identity.json` has no `parent_session_uid` | confirmed |
| `FK-005` was never presented | `decisions.md` §`FK-005` → `Presented.Report` ("not yet presented"); no invite/interview iteration between 011 and 012 | confirmed |
| The whole record path is one function with one branch | `dump.py:72-101` | confirmed — `turn == 0` writes the JSON body, `turn ≥ 1` writes the diff; one summary append |
| `config.py` is the home for tunables and the shape a format global follows | `config.py:16` (`DUMP_DIR`), `:44` (`KV_POLL_INTERVAL`) | confirmed |
| No dump format flag exists today | `main.py:23-24` (`--dump-dir` is the only dump option) | confirmed — the flag is new surface |
| No serialization dependency is installed | `pyproject.toml:5-8` | `aiohttp`, `click` only |
| The machine record is not the most compact | `jq -Rs .` over every `turn_*.diff`, summed → 6,061,579 bytes vs 5,785,285 raw | **1.047×** — F-10 confirmed |
| The real dump's change text total | `cat turn_*.diff \| wc -c` over 137 files | 5,785,285 bytes |
| Turn 0's raw body size | `stat -f%z turn_000.json` | 93,886 bytes |
| `README.md:146-162` is the documented read surface | `README.md` | confirmed — layout block + "Reading the index" |
| `FK-005`'s citations still resolve | re-read `dump.py`, `config.py`, `main.py`, `README.md`, `pyproject.toml` | all resolve as cited |
| `FK-003`/`FK-004`/`FK-005` presented text untouched | `decisions.md` §`FK-003`, §`FK-004`, §`FK-005` | confirmed — only `FK-005`'s `State`/`Superseded-by` changed |
| No orphaned reservation | `decisions.md` §`US-007`, §`FK-006` | both filled; no `State: reserved` remains |

## Next Step

1. **invite** presents `FK-006` — its question, preamble and three options, verbatim
   from `decisions.md` §`FK-006` → `Presented` — together with §Answer 5 above as the
   landscape. One FK per turn.
2. **interview** resolves it; the one-FK-per-turn contract carries it.
3. **decide** then runs **PLAN-SELECTED** on the chosen set: the `OI:` register (three
   entries, `output_layout`'s replacement statement now per-encoding and carrying
   F-6's definition of "full"), `G-006` re-derived on the pick (F-11), the Anchor Map
   finalization (F-1/F-2 repairs, the encoding selector as a `Mode: NEW` config row,
   F-12's encoding-in-the-name decision, plus a `Mode: NEW` viewer row with its own
   `RB-*` row if a viewer option is picked), the VP/milestone matrices (F-3/F-4, plus
   `VP-T002-collapsed-artifact`'s extended clause), and the full task-out.
4. **Final approval stays blocked** until `FK-006` resolves, PLAN-SELECTED lands with
   `User Approved: yes` on each `OI:` entry, and the `qa-project` skill exists for
   `QAP-MS001`/`QAP-MS002`.

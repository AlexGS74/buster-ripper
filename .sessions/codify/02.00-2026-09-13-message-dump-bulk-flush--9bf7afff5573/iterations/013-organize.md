# 013 — Organize (encoding/format cluster — `FK-006`, `G-006`)

## Orchestrator Prompt (verbatim)

> Execute the organize-interactive skill for this session. Arguments: $ARGUMENTS
>
> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
> Replan cluster: encoding/format cluster (FK-006, G-006)
> Interview's stated reason: the user's FK-006 answer selects none of the three presented options; it keeps the current change-focused record as one option and asks for something more human-visible/human-friendly than the plain change log, but names no encoding, so FK-006's Resolution cannot be written. The repair moves the Anchor Map (T002's record builder, plus a Mode: NEW viewer row with its own RB-* row if a viewer ships), T005, the VP-T002-* set, and the OI: replacement statement.
>
> [The remainder of the dispatch was the `organize-interactive` skill body plus the
> `codify-agents`, `critical-thinker`, and `codify-user-prompt` skill bodies. `$ARGUMENTS`
> was empty. Question Mode is `deferred` per session metadata; no `Mode:` line was present.]

## Session Type / Mode

- Continuing session `02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573`. Identity
  verified against `session-identity.json`: UID `9bf7afff-5573-402a-b102-827423a1dded`;
  handle suffix `9bf7afff5573` = first 12 hex of the UID; display id
  `02.00-2026-09-13-message-dump-bulk-flush` matches the handle prefix. **No
  `parent_session_uid`** → the parent coordination contract does not apply; there is no
  parent path to resolve, no sibling to inspect, and no `coordination status` to copy.
- **Mode: REPLAN-INCREMENT on the encoding/format cluster.** This is the third organize
  pass of the session (`002` initial, `005` flush policy, `006` payload/format). This pass
  narrows to the **encoding sub-slice** of `006`'s cluster: `FK-006` (the switchable
  encoding set) and the `G-006` goal it arbitrates.
- **Question Mode: deferred.** No `AskUserQuestion` on this path. The one live user
  decision (`FK-006`'s repair) is **decide's** to author as a successor FK; organize does
  not author, refine, or present it.
- **Input**: `iterations/009-interview.md` §Round 3 (the interview report — the primary
  collect input for a reroute), plus `decisions.md` §`FK-006`/§`US-008`,
  `iterations/012-decide.md`, `plan-summary.md`, `handoff-context.md`, and the source.
- **Nothing was asked, nothing was authored.** No FK/US reservation, no `Presented` text
  touched, no task done-when rewritten, no code or test written.

## Context Conflict Detected

None. Session-local sources agree: `decisions.md` §`FK-006` (`State: open`, `Owner: decide`,
`Type: preference`, `Recommendation: none`, `Presented` complete and unaltered),
`plan-summary.md` §Current Phase / §Open Questions 2b, and `handoff-context.md` §Current
Status / §Open Forks all name `FK-006` as the single open FK and the encoding set as the
single open question. No orchestrator-prompt technical detail conflicts with session files.
The prompt's "Replan cluster" and "Interview's stated reason" fields **match** the interview
report verbatim in substance; both are treated as orchestrator-supplied scope, and every
factual clause in them was re-derived from session files and code before use (below).

## Changes Since Iteration 006 (same phase, earlier cycle)

- NEW: **Goal Shape Analysis** for `G-006` and for the two clauses of `US-008` — three rows
  are `OPEN` (below).
- NEW: **Anchor Map re-verification of every row in the cluster** against source, including
  rows previously marked Clear. Two corrections and one open boundary found.
- CORRECTED: `dump summary line` row — its Binding named a *new symbol* while its Mode token
  was `extend` (contradictory for the `Mode: NEW` creation gate, and the role already exists
  inline in code).
- CORRECTED: `RB-002`'s row title in §Reuse Evidence still named `DUMP_FLUSH_TURNS`; `F-2`
  had already corrected the Anchor Map row to `DUMP_FLUSH_BYTES`. The two surfaces
  disagreed.
- CORRECTED: the `difflib` import citation — cited as `dump.py:5` in `FK-005`, `FK-006`,
  `US-003`, `US-005`, `US-007` and iterations 010–012; it is **`dump.py:4`** (`:5` is
  `import json`). The claim it supports HOLDS; only the citation was wrong.
- NEW: **prospective `RB-*` evidence** for a viewer row (name search, entry-point role
  search, native capability), so decide's row is grounded rather than asserted.
- NEW: the **`G-006` ↔ `US-008` goal-level tension** — `G-006`'s "without additional
  tooling" clause and `US-008`'s "more human-visible or human-friendly" instruction point
  opposite ways; decide arbitrates.
- NEW: the **reachability finding** — the real dump `/Volumes/data/buster-dumps/c451c754ae4f`
  is **not mounted** in this agent context, so the `F-10` measurements are carried, not
  re-verified.
- UPDATED: `plan-summary.md` §Current Phase, §Anchor Map (four rows + the read-path note),
  §Reuse Evidence (`RB-002` title), §Open Questions 2b, §Validation Point Matrix (candidate
  VPs).
- UPDATED: `handoff-context.md` §Current Status, §Notes, QA Planning Context.
- UNCHANGED: `FK-001`, `FK-002`, every `Presented` text, the three `EI:` dispositions,
  `FK-A`/`FK-B`, the flush-policy cluster, `T001`/`T003`'s trigger half, the milestone
  skeleton, `RB-001`/`RB-003`.

## Interview Report Absorption

The interview report (`009-interview.md` §Round 3) is the primary input for this reroute.
Origin is tracked per finding so decide can weigh it.

| Finding | Origin | Weight |
|---|---|---|
| `FK-006` was presented and answered; the answer selects **none** of the three options | interview (round 3) | authoritative — it is a record of what was shown and received |
| The answer is recorded losslessly as `US-008`; `FK-006` stays `open`; the repair is decide's | interview (round 3) | authoritative |
| The repair's scope: Anchor Map (`T002`'s record builder + a possible `Mode: NEW` viewer row), `T005`, the `VP-T002-*` set, the `OI:` replacement statement | interview (round 3) — **adopted and re-derived** below | scope, verified against the plan's own Impact destinations |
| The nearest candidate in the presented set is the small terminal viewer (options 2/3) | interview (round 3) | an observation, not a resolution — carried as one of three readings |
| The default question ("as one option" may demote the current encoding from the default) | `US-008` (interview) — explicitly recorded as **not resolved** | authoritative as an open question |
| Four things are settled and must not be re-asked: `index.txt` cannot survive separately; append-only; the encoding is a per-process configuration choice; the current change-focused encoding is the default | prior analysis (006, 012) + `US-007` | carried; the fourth is now **qualified** by `US-008`(a) |
| `G-006` is the goal `FK-006` arbitrates | prior analysis (006, 008) | carried, re-derived in the Goal Shape table |

**Correction to the prompt's framing, recorded.** The prompt says the answer "keeps the
current change-focused record as one option and asks for something more human-visible/
human-friendly than the plain change log, but names no encoding". Verified against
`US-008`: the user's words name no element at all — neither an encoding nor a tool — so
"names no encoding" is accurate but incomplete; the answer names **no element**, which is
why the reading is open in three directions rather than two.

## Goal Shape Analysis

Every explicit user goal, MVP-floor outcome, and hard constraint in the cluster, tested as
an outcome rather than as work.

| Goal candidate | Source wording | Observation surface | Met when | Failure witness | Boundary / non-goals | Alternate readings | Authority |
|---|---|---|---|---|---|---|---|
| **`G-006` — the dump stays readable (function, not layout)** | `README.md:134-135` ("useful for identifying remaining cache busters after normalization") + `README.md:146-162` (the documented reading workflow) + `FK-002`'s option text ("diffs become a read-time concern") | a reader inspecting only `DUMP_DIR/<sid>/` after a session — the directory listing and the artifact's contents | given a session in which one known message changed at turn K, a reader inspecting only `DUMP_DIR/<sid>/` can locate turn K and identify what changed against turn K−1 — **for the encoding that session used** (`F-11`) | the artifact is written once per flush, is well-formed, carries one encoding in arrival order — and a reader holding only the directory cannot find the changed turn or see what changed (the selected encoding is the machine record with no viewer; or the change-focused encoding lost the per-turn diff). Every implementation task completed; the tool's documented purpose is gone | not a requirement that the *layout* survive; **not a requirement that a viewer be built** (see the tension row below); not a formatting requirement; does not require the machine record to be human-readable | (a) satisfied by per-turn diffs; (b) by per-turn rendered bodies (change visible only by comparing); (c) only by shipping a viewer/reconstruction tool; (d) not a requirement — the artifact is a machine record; **(e) new this pass** — satisfied by the change-focused record *plus* a human-facing addition (`US-008`) | **user** — `FK-006` is open; no repository evidence excludes (a)–(e). **OPEN** |
| **`US-008`(b) — the offered set carries something more human-visible or human-friendly** | `US-008`: "Keep the existing one as one option and **add something more human-visible or human-friendly**." | the shipped, selectable set: the encoding selector's accepted values (the `--dump-format` choices) **and** any shipped read-side command — as documented in `README.md` and reachable by a user | at least one element of the offered set affords a human reader something the plain change-focused log does not: either (i) a record that opens and reads on its own without replaying the chain, or (ii) a shipped tool that lists a session's turns, shows a turn's reconstructed text, and shows the change against the previous turn | every task completes, the artifact is well-formed, `README` documents "the formats", and the selector offers **exactly the encodings the user was already shown** (change-focused + self-contained + machine record) with no navigable or visible element added — the instruction "**add** something more…" produced no addition, so the offered set is materially what the user declined one round earlier | not a requirement to change the *default* (separate row); not a requirement that the machine record be human-readable (`F-11` says it is not); not a license for a web UI or a long-running service (`US-003`'s "small tui" bounds it); not a requirement to remove the change-focused encoding | (1) "more human-**visible**" = the small terminal viewer (`US-003`'s direction); (2) "more human-**friendly**" = the self-contained encoding — **which the presented set already contains in all three options**, so this reading makes the instruction a *clarification*, not an addition; (3) a human-facing element the presented set does **not** contain, which is what "**add**" most literally implies | **user** — the wording is the user's and selects among none of the readings. **OPEN** → decide's repair |
| **`US-008`(a) — the current encoding stays *available as one option*; does it stay the *default*?** | `US-008`: "Keep the existing one **as one option**…" | the artifact produced with **no** encoding configured — the selector's default value | reading A: with no configuration the change-focused record is what you get, so today's behaviour and today's reading workflow are unchanged for anyone who configures nothing; reading B: the change-focused record is one selectable value and the default may be another | the default is silently changed to a new encoding while the user only said "keep it as one option" — a user who configures nothing gets a different artifact and a different reading workflow than today, and no plan line records that this changed | does not decide *which* encodings exist (row 2's scope); does not require the change-focused encoding to be first in a list; does not by itself re-open `US-007`'s "the current one is retained" | A (stays the default) vs B (demoted to one option, default open) | **user** — `US-008` records the ambiguity as deliberately **not** resolved. **OPEN** → decide's repair |
| **`G-001` — the write-count reduction (hard constraint in this cluster)** | `FK-002`'s resolution + `plan-summary.md` §Goals and Constraints (`1 − F/(2N)`, 50% floor) | instrumented write-bearing file operations against the dump artifact over a session of N turns | for N turns flushed F times, total writes `= F` versus `2N` today — **for every shipped encoding** (`PI:T002/artifact_encoding/encoding_does_not_change_write_count`) | an encoding (or a viewer that maintains an index) reintroduces a per-turn write, so a session recorded in that encoding writes `N + F`: the goal holds for the default encoding only, and the encoding selector is the thing that broke it | durability waived (`US-001`); not a byte-volume cap; the encoding changes a record's *content*, not the write pattern | none material — the arithmetic is settled and no reading of "reducing writes" is left unmet | **code-grounded** — not open |

**Summary of `OPEN` rows** (mirrored into `handoff-context.md` so decide cannot build options
around an implicit reading): `G-006` reading selection; the identity of `US-008`(b)'s
human-facing element; whether `US-008`(a) demotes the default.

**Structural observation (for decide, not authority).** `US-008`'s sentence addresses two
things in sequence, and the presented set **fused** them: "keep the existing one as one
option" speaks to the **encoding axis**, and "add something more human-visible or
human-friendly" speaks to the **human-facing axis**. Iteration 012 §"Why three options and
not four" already observed that "the viewer is orthogonal to the encoding set" — and then
packaged the viewer *inside* options 2 and 3. The user's answer is consistent with a reader
who did not experience the viewer as a thing being offered to them. Whether that is what
they meant is decide's to settle; it is recorded here because a repair that re-presents a
fused set repeats the presentation the user just answered past.

## Anchor Map Verification and Refinement

Every row in the cluster was re-read against source this pass — **including rows collect and
organize previously marked Clear**. Results:

| Row | Binding as planned | Code check | Verdict |
|---|---|---|---|
| dump entry point | `dump.py:dump_turn` | `dump.py:61-109` — the entire record path in one function: `turn == 0` → `json.dumps(body)` (`:80-85`); `turn ≥ 1` → `difflib.unified_diff` over `_render_body` (`:86-102`); summary appended (`:104-106`) | **CONFIRMED referent**; description covers the change-focused encoding only → qualifier added |
| per-session dump state | `dump.py:SessionState` | `dump.py:16-20` (`turn`, `prev_text`, `session_dir`) | CONFIRMED |
| dump write buffer | `dump.py:SessionState` | same symbol; the Notes' justification (no separate buffer type) holds | CONFIRMED |
| bulk flush | `dump.py` (new symbol) | no such symbol; `RB-003` | CONFIRMED `NEW` |
| flush timer | `dump.py` (new symbol) | no such symbol; `compaction.py:28-57` is the pattern peer, `server.py:25-34` the spawner | CONFIRMED `NEW` |
| flush trigger parameters | `config.py:DUMP_FLUSH_INTERVAL`, `DUMP_FLUSH_BYTES` (new globals) | `config.py:16` (`DUMP_DIR`), `:44` (`KV_POLL_INTERVAL: int = 10`) | CONFIRMED `NEW`; `F-1`/`F-2` stand |
| dump dir config | `config.py:DUMP_DIR` | `config.py:16` | CONFIRMED `reuse` |
| shutdown flush seam | `server.py:on_cleanup` | `server.py:60-72` — cancel `kv_monitor_task` → `await app["session"].close()` | CONFIRMED `extend`; the planned ordering (cancel **two** tasks → flush → close) inserts cleanly |
| signal handling | aiohttp `web_runner` | not re-run (native aiohttp path) | CONFIRMED `reuse`, unchanged |
| dump caller | `routes/messages.py:handle_messages` | `routes/messages.py:12` (import), `:24` (def), `:55-57` (call under `config.DUMP_DIR is not None`) | CONFIRMED `reuse` |
| **dump summary line** | `dump.py` (**new symbol**: the per-turn summary-string builder) — Mode: `extend` | the summary construction exists **inline** at `dump.py:84` and `:98-101`; **no symbol exists** | **MALFORMED → CORRECTED** (below) |
| collapsed dump artifact | `dump.py` (new symbol: artifact filename constant + record serializer) — Mode: `NEW` | no such symbol; `RB-003` | CONFIRMED `NEW`; Notes gain `F-12` + the open boundary |

### Correction 1 — `dump summary line` row (headline)

The row's Binding named a **new symbol** while its Mode token was **`extend`**. That is
contradictory for the mechanical `Mode: NEW` creation gate: a row whose referent does not
exist yet cannot hold `extend`, and a row that holds `extend` cannot require an `RB-*` row
it does not have. Verified against source, the role **already exists**: the summary string
is built inline inside `dump_turn` at `dump.py:84` (the turn-0 `FULL` line) and `:98-101`
(the `+added/-removed` line). Extracting a two-line f-string into a standalone symbol purely
to give it a name is the wrapper/relay case the skill forbids. **Corrected in place**: the
Binding now names the existing construction site; Mode stays `extend`; the Notes record that
if the encoding parameterization turns the construction into a distinct concept, decide
re-stances it to `NEW` with an `RB-*` row.

### Correction 2 — `RB-002` row title named the wrong global

§Reuse Evidence's `RB-002` row is titled "flush timer + its trigger parameters (`dump.py`
periodic coroutine; `config.py:DUMP_FLUSH_INTERVAL`, **`DUMP_FLUSH_TURNS`**)". `F-2` had
already corrected the Anchor Map row to `DUMP_FLUSH_BYTES` against `FK-001`'s recorded
**data-volume** basis, and `RB-002`'s own name search cites `KV_POLL_INTERVAL`. Two
surfaces in the same plan named two different globals for one parameter. **Corrected in
place** to `DUMP_FLUSH_BYTES`.

### Correction 3 — the `difflib` line citation (not a codebase defect)

`FK-005`, `FK-006`, `US-003`, `US-005`, `US-007`, and iterations 010–012 all cite
"`difflib` is already imported at `dump.py:5`". The import is at **`dump.py:4`**; `:5` is
`import json`. The claim those records rest on — the viewer needs no new dependency — still
**HOLDS**; only the citation is off by one. The frozen FK text is **not** edited (the Fork
rules forbid it); recorded here and in the plan's read-path note so forward does not chase
the wrong line.

### Open boundary flagged, NOT corrected (decide's)

**Three rows describe record construction and no row states the boundary between them:**

- `dump entry point` (`dump.py:dump_turn`, `extend`) — "**builds** the turn's record at
  append time (summary line; turn 0 → full JSON body; turn ≥ 1 → unified diff …)"
- `collapsed dump artifact` (`dump.py`, `NEW`) — "artifact filename constant + **record
  serializer**"
- `dump summary line` (`dump.py:84`,`:98-101`, `extend`) — the summary-string construction

`US-007`'s encoding parameter is a **branch on what a record contains**, and `F-12`'s
"the name states the encoding" is a property of the artifact symbol. Both must land on
exactly one of these three. Organize does **not** settle it: the choice depends on the
design decide settles (one serializer vs. `dump_turn` + a serializer), so it is **not
deterministically resolvable** and guessing would foreclose decide's design. Flagged for
the repair; it is the reason `T002`'s record-builder row is named in the repair scope.

### Prospective `RB-*` evidence for a viewer row (gathered, not stanced)

The viewer row is **not** added — the plan already records that the anchor "is created by
that decision, not pre-authorized here", and the pick is unmade. But decide's `RB-*` row
must be grounded in an operation, not asserted, so the evidence is gathered now:

- **Name search**: `grep -rniE "tui|viewer|curses|display|reconstruct|pager|navigat|print
  turn|show turn"` over `*.py`/`*.md`/`*.sh` (excluding `.venv/`, `.sessions/`,
  `__pycache__/`) → **zero matches**. No viewer symbol, module, class, or command exists.
  (Consistent with `RB-003`'s read-path search: no code reader of `turn_000.json`, `*.diff`,
  or `index.txt`.)
- **Role/peer search**: the only read-side artifact today is prose (`README.md:146-162`).
  Entry points are `main.py:62:main` (the console script, `pyproject.toml:10-11`) and
  `server.py:99:main` — the latter **already a `quality-backlog.md` confusability finding**
  (duplicate `main`, `server.py:99` vs `main.py:62`). A viewer command is therefore a
  **new entry point** that lands on that existing hazard: decide must say whether it is a
  second console script, a `--dump-*` subcommand on the existing one, or a separate module
  with its own `__main__`.
- **Native capability (attempted operation, not self-assessment)**: the venv interpreter is
  uv CPython 3.13.8 (`/Users/alexhubitski/.local/share/uv/python/cpython-3.13.8-macos-aarch64-none/bin/python3.13`;
  `.venv/bin/python` symlinks to it). Its `lib/libpython3.13.dylib` exports
  `_PyInit__curses` and `_PyInit__curses_panel` (verified with `nm -g`), i.e. the `_curses`
  C extension is **statically linked into the interpreter** rather than living in
  `lib-dynload` (which holds only `_ctypes_test` and `_tkinter`). `lib/python3.13/curses/`
  is present. So `import curses` resolves with **no new package** — the "no new dependency"
  claim in `US-003`/`US-005`/`FK-006` option 2 **HOLDS**. `difflib` is imported at
  `dump.py:4`. Dependencies remain `aiohttp` + `click` (`pyproject.toml:5-8`).
- **Conclusion**: no existing symbol owns a viewer role; a viewer is genuinely `NEW` and
  needs its own complete `RB-*` row. Its nearest reusable primitives are `curses`
  (available) and `difflib` (imported) — neither supplies turn semantics, a listing, or an
  ordering rule.

## Confusability Findings

**No new backlog entry this pass.** Reasoning, per the reliability rule (record only what
actually misled an agent):

- Correction 3 (`difflib` at `:5` vs `:4`) is a **session-record** inaccuracy, not a
  codebase or registry artifact that misleads readers — the code is unambiguous and the
  claim it supports holds. It is corrected in the report and the plan, not backlogged.
- Correction 1 (the malformed `dump summary line` row) is a **plan** artifact, not the
  codebase.
- The viewer entry-point hazard lands on the **existing** backlog entry
  (`server.py:99` duplicate `main`); it sharpens that finding's consequence rather than
  creating a new one. Noted in the plan's read-path note.
- The `buster_ripper_legacy.py:41-45` docstring documents the displaced layout, and
  `RB-002` warns readers not to bind to the legacy timer copy — both were **recognized and
  guarded** in prior passes; no agent was misled. Not a new finding.

`quality-backlog.md` therefore still holds its single entry. No speculative sweep was run.

## Invariant Reconciliation Prep

**No invariant registry exists in this repository** (`.invariants/` absent — re-verified).
All `EI:` entries are session-local drafts.

### Existing invariants in contradiction scope

- `EI:buster_ripper.dump/dump_turn/output_layout` — the session dir holds `turn_000.json` /
  `turn_{N:03d}.diff` / `index.txt` | Status: **intended** | **override target** (displaced
  by `FK-002`; the replacement statement now describes a configured *family* of encodings
  and needs `FK-006` for the membership of the family)
- `EI:buster_ripper.dump/dump_turn/sequential_diff_chain` — turn N diffs turn N−1's rendered
  text; `prev_text` advances only when a turn is written | Status: **intended** | **refine**
  — encoding-qualified (`F-11`); the encoding-independent part is the append-time
  `turn`/`prev_text` sequencing
- `EI:buster_ripper.dump/dump_turn/turn_zero_is_full_body` — turn 0 is written as a full
  JSON body, never a diff | Status: **intended** | **refine** — `F-6` sharpens "full body";
  encoding-independent (every option keeps it)
- `EI:buster_ripper.routes.messages/handle_messages/dump_failure_isolated` | Status:
  **intended** | **preserve** (the append path builds records, so the swallow-all contract
  covers more code)

### Likely planned invariant themes

- `PI:T002/artifact_encoding/encoding_is_per_artifact` (hard) — one encoding per artifact,
  chosen once per process, **and the artifact states which** (`F-12`). Grounds the
  `output_layout` override.
- `PI:T002/artifact_encoding/encoding_does_not_change_write_count` (hard) — protects `G-001`
  against a future encoding that reintroduces a per-turn write.
- **NEW candidate this pass** — a human-facing read-surface invariant, contingent on the
  viewer shipping: *the viewer lists a session's turns in **arrival order** (not as threads —
  `F-8`) and shows each turn's reconstructed text with its change against the previous
  turn.* Grounded in `US-008`(b) + `US-003`. Not yet a planned invariant; it becomes one only
  if decide's repair ships the viewer.
- **NEW candidate this pass** — if `US-008`(a) resolves to reading A: *with no encoding
  configured, the change-focused encoding is produced.* Makes the default an observable
  contract rather than a config detail.

### Suspected contradiction pairs

- `EI:.../output_layout` ↔ the encoding-family theme — direct displacement. **Override**
  (already the target).
- `EI:.../turn_zero_is_full_body` ↔ the collapsed/encoded artifact — **no contradiction**;
  every encoding keeps turn 0's raw body and rendering. **Refine.**
- `EI:.../sequential_diff_chain` ↔ the encoding theme — the per-turn-diff-against-the-previous-
  rendering property holds for the change-focused encoding only. **Refine**, encoding-qualified.
- **NEW (goal-level, not invariant-level)**: `G-006`'s "**without additional tooling**" ↔
  `US-008`'s "**more human-visible or human-friendly**". The user's instruction most
  plausibly asks for a tool; the goal's clause as written excludes one. Decide arbitrates,
  and PLAN-SELECTED writes `Met when` per encoding (`F-11`).
- **NEW**: `PI:.../encoding_is_per_artifact` ↔ `EI:.../output_layout` — the replacement names
  a family, and each member must announce itself (`F-12`). **Override** (the same target).

### Out-of-scope invariants intentionally dropped

- `EI:buster_ripper.session/session_id/derived_after_normalize` — untouched by the encoding
  cluster; it remains in the plan's register but is not in this cluster's contradiction
  surface.
- `EI:buster_ripper.server/on_cleanup/graceful_shutdown_hook` — likewise; the flush ordering
  it carries is `T003`'s, not this cluster's.

**Critical rule honoured**: none of the three dump invariants is `known_bug` or strongly
`likely_accidental` — all three are `intended`. `output_layout` is the only override target;
the other two are refinements, and neither is framed as something to preserve unchanged.

## QA Planning Context

### Regression Test Candidates

The existing suite has **no dump tests** (`tests/` holds 8 modules — `test_backend`,
`test_compaction`, `test_context_overflow`, `test_image`, `test_normalize`,
`test_routes_integration`, `test_server_hooks`, `test_session_stats` — and `tests/conftest.py`
is a 3-line stub). `dump.py` carries the bulk of the suite's uncovered statements.

| Behavior ID | Description | Risk | Tier | Priority |
|---|---|---|---|---|
| B-003 | dump failure isolated from the request path | High | `tests/test_routes_integration.py` | P0 |
| B-005 | proxy response passthrough unaffected | High | `tests/test_routes_integration.py` | P0 |
| B-004 | `on_cleanup` shutdown behavior | Medium | `tests/test_server_hooks.py` | P1 |
| B-006 | session directory isolation | Medium | new (`tests/test_dump.py`) | P1 |
| B-001 | dump output layout / index format | — | **replaced** by the collapsed artifact | re-dispositioned |
| B-002 | diff chain correctness | — | **re-dispositioned** per encoding | re-dispositioned |
| B-007 | turn 0's full body preserved and recoverable | High | new (`tests/test_dump.py`) | P0 |
| (new) | the human-facing element (`US-008`(b)) | High if it ships | `tests/qa/` | P0 **if shipped** |

### New Feature Test Areas

| Feature | Expected Behaviors | Tier | Dependencies |
|---|---|---|---|
| collapsed per-session artifact | one artifact file, one write per flush, arrival order, **name states the encoding** (`F-12`) | unit (`tests/test_dump.py`) | T001, T002 |
| encoding selector | the configured encoding determines the record built; the default is the change-focused one (per `US-008`(a)'s reading) | unit + integration | T002 |
| human-facing addition (if a viewer ships) | lists turns in arrival order; shows a turn's reconstructed text; shows the change against the previous turn | integration (`tests/qa/`) | T002 + the new viewer task |
| flush triggers | interval, volume, shutdown | unit + `tests/qa/` | T003 |

### Validation Point Candidates

| Source behavior | Validation point | Observable behavior | Candidate observable after | Candidate validate at | QA author pass | Dependencies | Scope reason |
|---|---|---|---|---|---|---|---|
| B-001 | `VP-T002-collapsed-artifact` | after a flush of turns 0..2, `DUMP_DIR/<sid>/` holds **exactly one** artifact and no `turn_000.json`/`turn_NNN.diff`/`index.txt`; **its name states its encoding**; the three records are in arrival order | T002 | MS-001 | QAP-MS001 | T001, T002 | on-disk shape needs a multi-turn flush; the listing is all a reader sees before opening the file |
| B-002 | `VP-T002-change-readable` **(must be qualified per encoding — `F-11`)** | given a session where turn 2 changes one known message, a reader can locate turn 2 and identify the differing content against turn 1 | T002 | MS-001 | QAP-MS001 | T002 | ordering and per-turn identity need a multi-turn flush; the clause holds as written for change-focused, becomes a comparison for self-contained, fails for the machine record |
| B-007 | `VP-T002-turn0-available` | turn 0's full normalized JSON body is present and recoverable | T002 | MS-001 | QAP-MS001 | T002 | fails for any encoding storing only diffs |
| (new, `G-001`) | `VP-T002-write-count-reduced` | instrumented writes `= F` for N turns, **for each shipped encoding** | T002 | MS-001 | QAP-MS001 | T002 | needs a deterministic injected `F`; `F-3` keeps this as the single write-count VP |
| (new, `US-008`(b)) | **`VP-T002-human-facing` (candidate)** | if a viewer ships: it lists a session's turns in **arrival order** and shows a turn's reconstructed text plus its change against the previous turn | the viewer task | MS-001 (or a new `MS-*` — decide) | QAP-MS001 | the viewer task | the only observable form of `US-008`'s "more human-visible or human-friendly"; without it the instruction has no gate |
| (new, `US-008`(a)) | **`VP-T002-default-encoding` (candidate)** | with no encoding configured, the artifact is the change-focused encoding | T002 | MS-001 | QAP-MS001 | T002 | makes the default an observable contract rather than a config detail — only if reading A holds |
| (new, `G-005`) | `VP-T002-volume-flush` | reaching the configured volume threshold puts the pending turns on disk with no shutdown and no timer | T002 | MS-001 | QAP-MS001 | T002 | basis-agnostic wording survives `FK-B` |
| B-004 | `VP-T003-shutdown-flush` | after SIGINT/SIGTERM, buffered turns are on disk and the process exits without a traceback | T003 | MS-002 | QAP-MS002 | T003 | requires process-level signal delivery |
| (new, `G-004`) | `VP-T003-interval-flush` | with a small injected interval and an unreachable volume threshold, one appended turn appears in the artifact while the process runs | T003 | MS-002 | QAP-MS002 | T003 | the timer only exists once wired |
| (new, `G-004`) | `VP-T003-timer-resilience` | a raising flush is logged and the loop keeps flushing; a cancelled timer stops without a traceback | T003 | MS-002 | QAP-MS002 | T003 | a dead trigger in a healthy process is only visible in a live loop |
| B-005 | `VP-T003-passthrough-regression` | existing integration suite green | T003 | MS-003 | — (existing suite) | — | regression gate |
| B-003 | `VP-T001-dump-failure-isolated` | a raising dump/flush still yields the upstream response body | T001 | MS-001 | QAP-MS001 | T001 | request path unchanged |

**Behavior dispositions**: B-001, B-002 re-dispositioned; B-003…B-007 mapped; plus the new
human-facing behavior (mapped only if the viewer ships) and the new default-encoding
behavior (mapped only if reading A holds).

### Candidate Milestones

| Milestone | Constituent tasks | Complete when | Validation points | QA author pass |
|---|---|---|---|---|
| MS-001 | T001, T002, T005 (**+ the viewer task if it ships — decide's call whether it joins MS-001 or forms a new `MS-*`**) | a buffered session's turns reach disk only after a flush, in the collapsed format, in arrival order, with turn 0's full body present, the write count `F` controlled, and the encoding stated by the artifact's name; the README documents the new artifact | `VP-T001-*`, `VP-T002-*` (incl. the two candidates above) | QAP-MS001 |
| MS-002 | T003 | after a real SIGINT/SIGTERM, buffered turns are in the artifact and the process exits cleanly; **and** with the timer running, a pending turn reaches the artifact without shutdown | `VP-T003-shutdown-flush`, `VP-T003-interval-flush`, `VP-T003-timer-resilience` | QAP-MS002 |
| MS-003 | T004 | existing suite green with new `dump.py` lines covered and total coverage ≥ 80% | `VP-T003-passthrough-regression` | — (existing suite is the gate) |

**Redundancy still declared**: `VP-T002-single-write-per-flush` is derivable from
`VP-T002-write-count-reduced` + `VP-T001-writes-deferred` (`F-3`). Keep the tighter contract,
not both.

### Notes for Interview / Decide

- `T004` must cover **each shipped encoding**, not just the default: `PI:.../encoding_does_not_change_write_count`
  has no gate unless the write-count test runs per encoding.
- If the viewer ships, `T005` changes kind — it documents a **tool** as well as a format
  (`US-008`(b) makes this likely).
- The `qa-project` skill is still absent, so `QAP-MS001`/`QAP-MS002` remain blocked on
  Topic 2 (QA infrastructure setup) — a presentation topic, not an FK.

## Dropped-Clause Sweep

**Not triggered.** This pass supersedes no prior definition of the same scope: it re-verifies
Anchor Map rows, corrects two plan-artifact inaccuracies in place, adds candidate validation
points, and states the goal shape. No task done-when, no contract, and no invariant statement
is replaced. The Anchor Map edits are refinements of rows whose referents are unchanged
(`dump_turn`, `SessionState`, `config.DUMP_DIR`, `server.on_cleanup`,
`routes/messages.handle_messages`) or corrections of a row that named a nonexistent symbol.
`FK-006`'s `Presented` text is untouched; the wholesale replacement of that scope belongs to
decide's repair, which will owe its own sweep.

## Compaction Decision

- Prior same-phase files reviewed: `iterations/002-organize.md`, `iterations/005-organize.md`,
  `iterations/006-organize.md`.
- **Superseded**: none.
- **Not superseded**: all three.
- **Compaction performed: no.**
- **If no**: different slice, and the prior files still carry live uncertainty. `002` is the
  whole-session initial synthesis; `005` is the flush-policy cluster (`RB-002`, the timer
  rows, `G-004`/`G-005`, the coverage headroom) — untouched by this pass; `006` is the
  payload/format cluster's **layout** decision, and this pass is the narrower **encoding**
  sub-slice inside it. `006`'s reasoning (why the documented layout is displaced, why
  append-only, why `index.txt` cannot survive separately, why `RB-003`'s read path has no
  code reader) is the provenance every encoding option still rests on, and none of it is
  restated here in full. Deleting it would remove reasoning live options depend on.

## Deferred Questions

- **Which element is the "more human-visible or human-friendly" one** (the small terminal
  viewer / the self-contained encoding already in every option / an element the set lacks)?
  - Status: **routed to `decide`** as the `FK-006` repair (successor FK from `US-008`) — not
    deferred to /codify-review, and not an organize question.
  - Impact: decides the option set, the Anchor Map viewer row, `T005`'s kind, the
    `VP-T002-*` set, and the `OI:` replacement statement.
  - Agent analysis: `none` — evidence does not rank the three readings; a preference.
- **Whether `US-008`(a) demotes the current encoding from the default.**
  - Status: **routed to `decide`** as part of the same repair (`US-008` records it as
    deliberately unresolved).
  - Impact: the artifact a user gets with no configuration; the `VP-T002-default-encoding`
    candidate.
  - Agent analysis: `none` — the user's words do not settle it.
- **Whether the viewer is interactive or the one-shot "print turn N" subset.**
  - Status: **routed to `decide`**; `US-003`'s "small tui" is the nearest ground.
  - Impact: the viewer task's size and its entry-point form.
  - Agent analysis: `none` — a preference with a stated cheaper subset.
- **Which device/filesystem hosts `DUMP_DIR` in production** (Linux ext4/XFS beside vLLM, or
  macOS APFS on the dev box)?
  - Status: DEFERRED for /codify-review (unchanged).
  - Impact: the exact write-amplification factor differs; the directional verdict does not.
  - Agent analysis: `none` — a factual premise, not a user preference.
- **QA infrastructure setup** (`.agents/skills/qa-project/SKILL.md` absent) — unchanged; a
  presentation **topic**, not an FK.

## Questions Asked

**No questions** — deferred mode, and the one live user decision belongs to `decide` (it must
reserve and fill a successor FK). No `AskUserQuestion` call was made; no FK was authored,
refined, presented, or resolved by this agent.

## Gates Found Unmet (reported, not cured)

1. **`FK-006` is unresolved.** The answer selects none of the three presented options; the
   successor FK is owed by its owner (`decide`). `FK-006` must then be marked `superseded`.
2. **`.agents/skills/qa-project/SKILL.md` does not exist** while the plan declares
   `QAP-MS001`/`QAP-MS002`. Blocked on Topic 2's answers; not written here.
3. **Final approval remains blocked** on (1), (2), and PLAN-SELECTED landing with
   `User Approved: yes` on each `OI:` entry.
4. **The real dump is unreachable from this context.** `/Volumes/data` is **not mounted**
   (`ls /Volumes` → Blackmagic, Grok Bot Installer, Macintosh HD, URX, users, Videos; no
   `data`), so `/Volumes/data/buster-dumps/c451c754ae4f` cannot be read and the `F-10`
   measurements — 5,785,285 bytes of change text, 6,061,579 bytes for the JSON-encoded
   record (1.047×), 93,886 bytes for `turn_000.json`, 137 turns — are **carried, not
   re-verified** this pass. `US-007`'s third factual claim and `FK-006`'s `F-10` both rest
   on them; the assumptions pass must either re-measure on a host with the volume mounted or
   record the provenance limit. Not cured here.
5. `FK-006`'s `Presented.Report` still names `iterations/012-decide.md`; the *presentation*
   is recorded in `009-interview.md` §Round 3. Advisory field only.

## Validation Performed

| Claim | How verified | Result |
|---|---|---|
| Session identity / handle suffix | `session-identity.json` vs handle | matches (`9bf7afff5573` = first 12 hex of UID) |
| No parent coordination contract | `session-identity.json` has no `parent_session_uid` | confirmed — nothing to resolve or reconcile |
| Exactly one open FK | `grep` of `State:` in `decisions.md` | `FK-006` open; `FK-001`/`FK-002` resolved; `FK-003`/`FK-004`/`FK-005` superseded |
| `dump_turn` is the whole record path | `dump.py:61-109` | confirmed — one function, two branches, one summary append |
| `SessionState` is the buffer key's holder | `dump.py:16-24` | confirmed |
| The summary construction exists inline | `dump.py:84`, `:98-101` | confirmed — no symbol exists (correction 1) |
| `config.DUMP_DIR` is the mode switch | `config.py:16` | confirmed |
| `KV_POLL_INTERVAL` has no CLI flag | `main.py:12-67` option list | confirmed (`F-1` stands) |
| No `--dump-format` flag today | `main.py:25-26` (`--dump-dir` only) | confirmed — new surface |
| `on_cleanup` ordering seam | `server.py:60-72` | confirmed |
| `handle_messages` is the only dump caller | `routes/messages.py:12,24,55-57` | confirmed |
| No reader of dump artifacts in code | `grep` over `buster_ripper/` + `tests/` | confirmed — writers and prose only (`RB-003` stands) |
| No viewer role in code | `grep -rniE "tui\|viewer\|curses\|display\|reconstruct\|pager\|navigat"` | **zero** matches |
| stdlib `curses` is available | `nm -g` on the venv's `libpython3.13.dylib` | `_PyInit__curses`, `_PyInit__curses_panel` present (statically linked); `lib/python3.13/curses/` present |
| `difflib` import line | `dump.py` | **line 4**, not `:5` (correction 3) |
| Dependencies unchanged | `pyproject.toml:5-8` | `aiohttp`, `click` |
| Coverage omits `server.py` / `routes/*` / `main.py` | `pyproject.toml` `[tool.coverage.run].omit` | confirmed |
| No `tests/test_dump.py` | `ls tests/` | confirmed — 8 modules, no dump tests |
| `buster_ripper_legacy.py:41-45` documents the displaced layout | `sed -n '38,50p'` | confirmed — excluded from scope |
| The real dump is unreachable | `ls /Volumes`, `ls /Volumes/data` | not mounted → `F-10` figures carried, not re-verified |
| `FK-006` `Presented` text unaltered | `decisions.md` §`FK-006` | confirmed; only `US-008` and the presentation-outcome note were added by interview |
| No invariant registry | `.invariants/` | absent |

## Next Step

1. **`decide` REPLAN-INCREMENT on `FK-006`** — reserve and fill a successor FK from `US-008`:
   repair the option set so the human-facing element is explicit and separable, settle the
   three `OPEN` rows (the element's identity, whether the default moves, and whether the
   viewer is interactive or one-shot), state which Anchor Map row owns the encoding branch
   and which owns the artifact's name (`F-12`), then mark `FK-006` `superseded`. Owe a
   Dropped-Clause Sweep against `FK-006`'s `Presented` block.
2. **`invite`** re-presents the repaired set; **`interview`** resolves it.
3. **`decide` PLAN-SELECTED** on the pick — the `OI:` register (`output_layout`'s replacement
   statement per encoding, carrying `F-6`'s definition of "full"), `G-006` re-derived
   (`F-11` + the `US-008` tension), the Anchor Map finalization (`F-1`/`F-2` repairs, the
   encoding selector as a `Mode: NEW` config row, the boundary above, plus a `Mode: NEW`
   viewer row with its own `RB-*` row if the viewer ships), the VP/milestone matrices
   (`F-3`/`F-4` + the two candidate VPs), and the full task-out.
4. **Final approval stays blocked** until `FK-006` resolves, PLAN-SELECTED lands with
   `User Approved: yes` on each `OI:` entry, and the `qa-project` skill exists.

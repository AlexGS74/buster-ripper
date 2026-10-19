# 015 — Decide (PLAN-SELECTED on `FK-007` option 1)

## Orchestrator Prompt (verbatim)

> Execute the decide-interactive skill for this session. Arguments: $ARGUMENTS
>
> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
> Mode: plan-selected
> Fork: FK-007
> Pick (verbatim user response): 1
> Selected: FK-007 option 1 — "The full-text encoding is the addition — no new tool" — with no attached modifications (no preamble opt-in taken, no element named outside the three options).
>
> [The remainder of the dispatch was the `decide-interactive` skill body plus the
> `codify-agents`, `critical-thinker`, `codify-user-prompt`, and
> `codify-deterministic-closure` skill bodies. `$ARGUMENTS` was empty. Question Mode is
> `deferred` per session metadata.]

## Session Type / Mode

- Continuing session `02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573`. Identity verified
  against `session-identity.json`: UID `9bf7afff-5573-402a-b102-827423a1dded`; handle suffix
  `9bf7afff5573` = first 12 hex of the UID; display id `02.00-2026-09-13-message-dump-bulk-flush`
  matches the handle prefix. **No `parent_session_uid`** → the parent coordination contract does
  not apply: no parent path to resolve, no sibling to inspect, no `coordination status` to copy.
  Organize's Parent Contract Reconciliation is therefore **N/A** (recorded, not skipped).
- **Mode: PLAN-SELECTED.** The dispatch carries `Mode: plan-selected`, `Fork: FK-007`, the
  verbatim pick `1`, and the `Selected` line. Per the skill, this pass reads **narrowly**:
  organize's latest output, the options skeleton, and the selection. The option set is **not**
  re-derived and the pick is **not** re-litigated.
- **Question Mode: deferred.** No `AskUserQuestion`. `decide` asks nothing; no FK was reserved
  (the pick left nothing open, so no successor FK and no `US-*` are owed — contrast `FK-006`).
- **Input read**: `iterations/014-decide.md` (the options skeleton + closure packet),
  `iterations/013-organize.md` (Goal Shape Analysis, Anchor Map verification, prospective `RB-*`
  evidence), `decisions.md` §`FK-007`/§`US-008`/§`US-007`/§`US-005`/§`US-003`/§`US-006`,
  `plan-summary.md`, `handoff-context.md`, and the source.

## Context Conflict Detected

None material. `decisions.md` §`FK-007` (`State: resolved`, complete `Presented`, `Resolution`
= the pick), `plan-summary.md` §Current Phase, and `handoff-context.md` §Resolved Forks agree:
`FK-007` is resolved with option 1, no FK is open, and PLAN-SELECTED is the next stroke. The
prompt's `Mode`/`Fork`/`Pick`/`Selected` fields match the session files in substance.

**One factual refinement, recorded rather than treated as a conflict.** `plan-summary.md` §Options
and `FK-006`'s `Trigger` both carried `F-1`'s framing that the `--dump-format` flag is "new
precedent required by QA injectability, not an existing convention" because `KV_POLL_INTERVAL`
has no CLI flag. Re-read against source this pass: the repository **does** have an established
convention for a *named-value selector* flag backed by a config-declared value set —
`main.py:45-48` (`--eval-profile`, `type=click.Choice(list(config.PROFILES.keys()))`) and
`main.py:60` (`--backend-type`, `click.Choice(["sglang","vllm"])`). `F-1`'s claim is **true for
numeric tunables** (`KV_POLL_INTERVAL` has no flag) and **incomplete as a general statement**.
`--dump-format` therefore follows an existing in-repo pattern; the QA-injectability requirement is
still the *reason* the flag must exist, but it is not the only precedent. Recorded in §Anchor Map
and `RB-004`; `F-1` is not deleted (it remains correct about the numeric globals) — it is
**narrowed**.

## Changes Since Iteration 014

- NEW: **PLAN-SELECTED finalization of the whole plan** — done-when per task, finalized Anchor Map
  stances (one new `Mode: NEW` row), four planned invariants, the conflict matrix, the override
  register, the finalized Goal Definition / Goal Coverage / VP / milestone matrices, the Behavior
  Disposition Matrix, the capacity-gate baseline, the adversarial-surface census, and a Dropped-Clause
  Sweep against the predecessor `OI:`/`G-006`/`VP` definitions.
- NEW: **`RB-004`** — the complete Reuse Evidence row the encoding selector requires (it is new surface).
- NEW: **the artifact's identity is `(sid, encoding)`**, not `sid` (`R5` identity-key completeness):
  a session directory may legitimately hold one artifact per encoding after a process restart with a different
  `--dump-format`. This is the resolution of `F-12`'s "the listing no longer announces the encoding".
- NEW: **an `R2` durable-writer finding** — the artifact is written into a possibly pre-existing
  `DUMP_DIR/<sid>/` that may already hold a previous version's `turn_000.json` / `turn_NNN.diff` /
  `index.txt`. Those are neither migrated nor deleted, and the VP is scoped to a fresh session dir.
- NEW: **`VP-T001-writes-deferred` and `VP-T001-session-isolation` timing corrected** T001 → T002 (the
  behaviors do not exist until T002 removes the write sites). A plan defect found and repaired in place.
- NEW: **`VP-T002-single-write-per-flush` retired** — folded into `VP-T002-write-count-reduced` per `F-3`.
- UPDATED: `T002`'s scope now includes the encoding selector's config surface and CLI flag, so the branch
  is configurable in the increment that first reads it (dependency ordering: the encoding row is T002's, the
  flush-trigger rows stay T003's).
- UNCHANGED: `FK-001`, `FK-002`, every `Presented` text, `FK-A`/`FK-B`, the flush-policy cluster,
  `RB-001`/`RB-002`/`RB-003`, `T001`/`T003`'s flush-trigger half, the milestone skeleton, `G-001`…`G-005`.
- NOT DONE, deliberately: no invariant registry was created (`.invariants/` is absent — the `EI:`/`PI:` entries
  are session-local); no `roadmap.md` was seeded (the recorded plan seeds it at final approval); no FK authored.

## Selected Option Consumption

`FK-007` → `Resolution.Normative choice` is consumed as binding, not re-derived:

- The offered set is **exactly two encodings**: the change-focused record (**kept, and still the `default`**) and
  the **self-contained** record.
- **No tool ships.** No terminal viewer, no one-shot `print turn N` command.
- **The machine record is not added** (preamble (3)'s opt-in was not taken).
- The ≈2.3× byte cost is accepted **for the self-contained encoding when selected**; the default path's written
  volume is unchanged, so `FK-002`'s write-reduction outcome is untouched.

Everything below is PLAN-SELECTED output for that pick. No user modification was attached to the pick, so the task
structure is decide's to finalize; no sizing principle or merge/split preference is imposed.

## Deterministic Closure Gate — `F-12`, how the artifact states its encoding

Applied per `codify-deterministic-closure` before treating the artifact-name scheme as a user question (it is
user-visible, so it is not obviously an internal mechanism).

```text
CLOSURE: absorbed
Finding: F-12 (iterations/012-decide.md; carried into plan-summary.md §Anchor Map, the `collapsed dump
  artifact` row) — with the encoding configurable, the directory listing no longer announces what the artifact holds
Required outcome: a reader inspecting only DUMP_DIR/<sid>/ can tell which encoding the artifact holds, without
  opening the file
Authority: F-12's own required outcome ("the artifact's name must carry it"), decide 014 §Anchor Map
  ("The artifact's name/extension carrying the encoding stays Forward discretion per F-12" — the channel is settled,
  the token is not); VP-T002-collapsed-artifact ("the artifact's name states its encoding (F-12)"); US-007
  Normative choice ("one encoding per artifact"); buster_ripper/dump.py:61-109 (the artifact is produced by one
  process with one configured value, so exactly one token is needed)
Admissible candidates: (a) the filename carries the encoding's configured value — DUMP_DIR/<sid>/<encoding>.log;
  (b) the extension alone distinguishes them — session.diff / session.full; (c) an in-band marker on the first line
Rejected candidates: (c) — 014's recorded reading ("an in-band marker would be a second thing a reader must know
  to skip") and the VP's wording (the *name* states it); (b) — admissible but does not state the encoding: the token
  differs from the configured value, so a reader must learn a mapping, which is the lookup F-12 exists to remove
Selected closure: (a) — DUMP_DIR/<sid>/<encoding>.log, where <encoding> is exactly config.DUMP_FORMAT's value
Risk shape: unchanged — the artifact is one file per (sid, encoding) either way; (a) adds no new mechanism
Deferred additive work: none
Owning route: decide (this pass) → plan-summary.md §Task List T002/T005 + §Validation Point Matrix
  VP-T002-collapsed-artifact, which asserts the *property* (the name states the encoding), not the string
Question required: no
```

**Not papered over.** The chosen token is written into the plan text (`T002`'s done-when, `T005`'s README
scope) and the VP asserts the *property*, so the user sees the concrete name at final approval and can overturn it
in one clause. The rejected alternative is recorded above and in §Decide-Settled Design Details.

**Second, smaller closure — the `VP-T001-*` timing correction.** Not user-facing; recorded here for the ledger:

```text
CLOSURE: absorbed
Finding: plan-summary.md §Validation Point Matrix — VP-T001-writes-deferred and VP-T001-session-isolation
  carry `Observable after: T001`
Required outcome: the validation points must be observable at the task boundary they name
Authority: plan-summary.md §Task List (T001 is the buffer/state task; T002 is the task that removes the write
  sites and adds flush_all — "dump_turn builds the turn's record *instead of writing files*"); buster_ripper/dump.py:81-82,
  :94-95, :104-106 (the three write sites exist until T002 removes them)
Admissible candidates: (a) move both VPs to `Observable after: T002`; (b) re-scope T001 to remove the write sites
  and add a minimal flush
Rejected candidates: (b) — leaves a commit in which buffered turns can never reach disk (no flush trigger exists
  until T003), i.e. strictly worse than the state before it; it also restructures a settled task list without need
Selected closure: (a) — both VPs become observable at T002; handles stay stable, only `Observable after` moves
Risk shape: unchanged — no task scope moves
Deferred additive work: none
Owning route: decide (this pass) → plan-summary.md §Validation Point Matrix
Question required: no
```

## Goal Shape Consumption (organize 013's three `OPEN` rows)

Consumed, not re-derived. Every `OPEN` row is now dispositioned.

| Organize row | Disposition after the pick |
|---|---|
| (i) the identity of the "more human-visible or human-friendly" element | **CLOSED by the pick** — `FK-007` option 1: the element is the **self-contained encoding itself**, and no tool ships. Reading (2) of organize's table ("the self-contained encoding, which makes the instruction a *clarification* rather than an addition") is the one the user chose. Organize's failure witness for this row ("the selector offers exactly the encodings the user was already shown … with no navigable or visible element added") **does not fire**: the self-contained encoding was *not* in this plan's shipped set — it is new work in `T002` — so the plan's offered set genuinely gains an element. `VP-T002-change-readable`'s self-contained clause and `VP-T002-collapsed-artifact` are its gates. |
| (ii) whether `US-008`(a) demotes the current encoding from the default | **CLOSED deterministically** — `CLOSURE: absorbed` (014 §Deterministic Closure Gate). `FK-007` preamble (2) restated the default **with its ground and a one-clause correction path**; the user did not take it, so the default stands. `PI:T002/artifact_encoding/default_encoding_is_change_focused` + `VP-T002-default-encoding` make it an observable contract. |
| (iii) interactive viewer vs one-shot | **MOOT** — no tool ships. |

`G-006`'s alternate-reading column is resolved to reading **(b)**, `G-006`'s `Met when` is written **per
encoding** (`F-11`), and `G-006`'s "without additional tooling" clause **holds as written for both shipped
encodings** — no restatement is needed, because no tool ships.

## Planned Invariants In Scope

### PI:T002/artifact_encoding/encoding_is_per_artifact
- **Statement**: A dump artifact carries exactly **one** encoding. The encoding is selected once per process
  (`config.DUMP_FORMAT`, set by `main()` at startup, never mutated during a run), and the artifact's filename
  states which encoding it holds (`DUMP_DIR/<sid>/<encoding>.log`).
- **Reason**: `US-007` makes the encoding a configuration choice, so without this rule two encodings could be interleaved
  into one artifact — an unreadable record with no single decoder. Grounds the `output_layout` override.
- **Strength**: hard_requirement

### PI:T002/artifact_encoding/encoding_does_not_change_write_count
- **Statement**: The encoding changes what a record *contains*, never how many write-bearing file operations a session
  performs: one artifact, one write per non-empty flush, `F` writes for N turns.
- **Reason**: Protects `G-001` (the 50%-floor write reduction) against a future encoding that reintroduces a per-turn
  write, or a flush that rewrites the artifact.
- **Strength**: hard_requirement

### PI:T002/artifact_encoding/default_encoding_is_change_focused
- **Statement**: With no encoding configured, the artifact produced is the **change-focused** encoding, and today's reading
  workflow is unchanged for anyone who configures nothing.
- **Reason**: `US-008`(a) left the default open; decide 014 closed it deterministically and `FK-007`'s preamble (2)
  offered a correction path the user did not take. This makes the default an observable contract rather than a config detail.
- **Strength**: hard_requirement

### PI:T002/artifact_encoding/encoding_set_is_closed
- **Statement**: The accepted encoding set is exactly the two values declared in `config.DUMP_FORMATS`. An unknown configured
  value is a hard failure, never a silent fallback to another encoding.
- **Reason**: `FK-006`'s `If-wrong` records the irreversible part as the *number* of encodings shipped, because every one
  becomes a documented format. The closed set is what keeps the machine record a roadmap entry rather than a plan element.
- **Strength**: preferred

## Invariant Conflict Matrix

| Planned Invariant | Existing Invariant | Relationship | Decision | Why |
|---|---|---|---|---|
| `PI:T002/artifact_encoding/encoding_is_per_artifact` | `EI:buster_ripper.dump/dump_turn/output_layout` | contradicts | **override** | The session dir holds one collapsed artifact whose encoding is configured, not the three fixed per-turn files. The replacement statement names a configured *family* and requires each member to announce itself. |
| `PI:T002/artifact_encoding/encoding_is_per_artifact` | `EI:buster_ripper.dump/dump_turn/turn_zero_is_full_body` | refines | **preserve** | Every shipped encoding keeps turn 0's raw body **and** its rendering; the rule survives in both. `F-6` sharpens "full body". |
| `PI:T002/artifact_encoding/encoding_does_not_change_write_count` | `EI:buster_ripper.dump/dump_turn/sequential_diff_chain` | refines | **preserve** | The append-time `turn`/`prev_text` sequencing is encoding-independent and is what the refine protects. The per-turn-diff-against-the-previous-rendering property holds for the change-focused encoding only (`F-11`). |
| `PI:T002/artifact_encoding/default_encoding_is_change_focused` | `EI:buster_ripper.dump/dump_turn/output_layout` | refines | **override** (same target) | Part of the replacement contract: the configured family has a stated default member. |
| `PI:T002/artifact_encoding/encoding_set_is_closed` | *(none)* | unknown | **preserve** | No existing invariant constrains the size of the encoding set; this is new surface created by `US-007`/`FK-007`. |

**Unchanged in this cluster** (verified against source, dispositions carried from 006/008/013, not re-opened):
`EI:buster_ripper.routes.messages/handle_messages/dump_failure_isolated` (**preserve**),
`EI:buster_ripper.session/session_id/derived_after_normalize` (**preserve**),
`EI:buster_ripper.server/on_cleanup/graceful_shutdown_hook` (**preserve**).

**Critical rule honoured**: none of the three dump invariants is `known_bug` or `likely_accidental` — all three
are `intended`, so `output_layout` is the only override target and the other two are refinements, not
preserve-for-compatibility entries.

## Approved Override Register

Each entry's **direction** is settled (`FK-002`), each **replacement statement** is finalized here, and each is
**pending the user's `User Approved: yes` stamp**, which decide cannot write (the User Authority Boundary and the
Triad write authority both forbid an agent self-certifying user approval).

### OI:buster_ripper.dump/dump_turn/output_layout
- **Old invariant**: `EI:buster_ripper.dump/dump_turn/output_layout` — the session dir holds `turn_000.json`,
  `turn_{N:03d}.diff`, `index.txt`.
- **New invariant**: `PI:T002/artifact_encoding/encoding_is_per_artifact` — the session dir holds one collapsed
  per-session artifact, appended once per non-empty flush. Its filename states its encoding
  (`DUMP_DIR/<sid>/<encoding>.log`, `<encoding>` ∈ `{"change-focused", "self-contained"}`). One encoding per artifact,
  chosen once per process; the default is `change-focused`. The **change-focused** records are: one summary line per turn;
  turn 0 carries the full normalized JSON body (`json.dumps(body, indent=2, ensure_ascii=False)`) **and** its
  `_render_body` rendering; turn ≥ 1 carries the unified diff of `_render_body` output against the previous turn. The
  **self-contained** records are: one summary line per turn; every turn carries its complete `_render_body` rendering;
  turn 0 additionally carries the full normalized JSON body. Turn 0's "full body" therefore means **both** the raw
  normalized JSON body and its rendering (`F-6`), in every shipped encoding.
- **Decision**: override
- **Implementation consequence**: forward deletes the three write sites (`dump.py:81-82`, `:94-95`, `:104-106`) and the
  `turn_000.json` / `turn_{N:03d}.diff` / `index.txt` name literals; adds `config.DUMP_FORMATS`/`config.DUMP_FORMAT`,
  `main.py`'s `--dump-format`, the encoding branch, the artifact filename constant + path builder, and `flush_all()`.
- **Yield proof**: `VP-T002-collapsed-artifact` (one file for the configured encoding, no `turn_000.json` /
  `turn_NNN.diff` / `index.txt`, the name states the encoding), `VP-T002-default-encoding`,
  `VP-T002-write-count-reduced` (run per encoding).
- **User approved**: **pending — interview final approval.** The direction is `FK-002`'s; the replacement statement above
  has not been shown to the user.

### OI:buster_ripper.dump/dump_turn/sequential_diff_chain
- **Old invariant**: `EI:buster_ripper.dump/dump_turn/sequential_diff_chain` — turn N diffs turn N−1's rendered text;
  `prev_text` advances only when a turn is written.
- **New invariant**: `prev_text` and `turn` advance at **append** time, never at flush time, so a flush never resets the
  chain. For the **change-focused** encoding, each turn's record derives from the previous turn's rendered text, in
  arrival order. The append-time sequencing is encoding-independent (`F-11`).
- **Decision**: refine
- **Implementation consequence**: forward moves the `state.prev_text = current_text` / `state.turn += 1` advancement
  (`dump.py:108-109`) so it is unreachable from `flush_all()`; `flush_all()` must not touch `turn`/`prev_text`.
- **Yield proof**: `VP-T002-change-readable` (change-focused clause) plus `VP-T002-write-count-reduced` (a flush
  that carried K turns leaves the chain intact for the next turn).
- **User approved**: **pending — interview final approval.**

### OI:buster_ripper.dump/dump_turn/turn_zero_is_full_body
- **Old invariant**: `EI:buster_ripper.dump/dump_turn/turn_zero_is_full_body` — turn 0 is always written as a full
  JSON body, never as a diff.
- **New invariant**: turn 0's full normalized JSON body **and** its rendered rendering are present in the session artifact
  once flushed, never only a diff — in **every** shipped encoding (`F-6`).
- **Decision**: refine
- **Implementation consequence**: the turn-0 record carries both representations; the self-contained encoding must not
  drop the raw body, and the change-focused encoding must not drop the rendering (it is the first diff's baseline).
- **Yield proof**: `VP-T002-turn0-available`, run for both encodings.
- **User approved**: **pending — interview final approval.**

## Anchor Map (finalized)

`extend` = modify an existing referent · `reuse` = consume unchanged · `replace` = displaces a referent ·
`NEW` = the creation gate authorizes this new entity.

| Term | Binding (file:symbol) | Mode | Invariants | Notes |
|---|---|---|---|---|
| dump entry point | `buster_ripper/dump.py:dump_turn` | extend | `EI:.../output_layout` (override target), `EI:.../turn_zero_is_full_body`, `EI:.../sequential_diff_chain` | Becomes buffer-append. **Owns the encoding branch**: the record's content is selected by `config.DUMP_FORMAT` inside this concept. Also hosts the append-time volume check (`G-005`) and the `DUMP_DIR is None` guard that replaces `dump.py:63`'s bare assert (`F-5`). Keeps signature `dump_turn(body) -> None` and the swallow-all caller contract. **LNCP capacity gate**: this function is at `L≈34` executable statements (hard violation) and this task adds branches, so `T002` includes the smallest coherent extraction — module-private helpers inside this same concept (state lookup, record construction, the two encodings, the summary line, the artifact path). No new Anchor Map row is created by that extraction: it relocates this row's own concern, it does not add a referent |
| per-session dump state | `buster_ripper/dump.py:SessionState` | extend | `EI:.../sequential_diff_chain` | Gains `pending: list[str]` (encoded records) and `pending_bytes: int`. `turn`/`prev_text` advance at **append** time, not flush time |
| dump write buffer | `buster_ripper/dump.py:SessionState` | extend | `EI:.../sequential_diff_chain` | `RB-003`. The pending payload is a list of **format records**, not a list of `(path, text)` writes, keyed per `sid`. `SessionState` already holds that key, so no separate buffer type |
| bulk flush | `buster_ripper/dump.py` (new symbol `flush_all`) | NEW | `EI:.../sequential_diff_chain`, `EI:.../turn_zero_is_full_body`, `EI:.../graceful_shutdown_hook`, `PI:.../encoding_does_not_change_write_count` | `RB-003`. Appends each session's pending records as **ONE** write to that session's artifact, clears pending, no-ops when `DUMP_DIR is None` or nothing is pending, must not raise, contains **no `await`**. Called from all three triggers. Never writes per-turn files or `index.txt`. Never touches `turn`/`prev_text` |
| collapsed dump artifact | `buster_ripper/dump.py` (new symbol: artifact filename constant + path builder + append target) | NEW | `EI:.../output_layout` (override target), `EI:.../turn_zero_is_full_body`, `EI:.../sequential_diff_chain`, `PI:.../encoding_is_per_artifact` | `RB-003`. One file per `(sid, encoding)` under `DUMP_DIR/<sid>/`, named `<encoding>.log` where `<encoding>` is exactly the configured value (`F-12`). **Append-only** — a rewrite-per-flush variant is excluded by `FK-002`'s presented option text and by `G-001` witness B. **Boundary settled (decide 014)**: this row owns the artifact's **name** and the append target; the **encoding branch** belongs to `dump entry point`; `dump summary line` is encoding-independent |
| dump encoding selector | `buster_ripper/config.py:DUMP_FORMAT`, `DUMP_FORMATS` (new globals) + `buster_ripper/main.py:--dump-format` | NEW | `PI:.../encoding_is_per_artifact`, `PI:.../default_encoding_is_change_focused`, `PI:.../encoding_set_is_closed` | **`RB-004`**. The injectability seam for the encoding. `config.DUMP_FORMAT` follows `config.KV_POLL_INTERVAL`'s module-global shape; the flag follows the **`config.PROFILES` + `--eval-profile` shape** (`main.py:45-48`) — a config-declared value set exposed as `click.Choice` with a `default=`. ⚠ **`F-1` narrowed**: `KV_POLL_INTERVAL` has no flag (true for numeric tunables), but a named-value selector flag is an **existing** convention here, not new precedent. `--dump-format` is new surface because no dump-encoding selector exists |
| dump summary line | `buster_ripper/dump.py` (the per-turn summary-string construction, relocated by the capacity extraction) | extend | `EI:.../output_layout` (override target) | **Encoding-independent.** One summary line per record in every encoding: turn 0 the `FULL` form, turn ≥ 1 the `+added/-removed` form. The counts are computed in **both** encodings (the diff is computed either way); only whether the diff *text* is stored differs. Not re-stanced to `NEW`: the role exists, and the encoding does not make the construction a distinct concept |
| flush timer | `buster_ripper/dump.py` (new symbol) | NEW | `EI:.../sequential_diff_chain`, `EI:.../graceful_shutdown_hook` | `RB-002`. The periodic loop body, following the in-repo split where the domain loop lives in the domain module (`compaction.poll_kv_cache`) and the app wiring lives in `server.py`. **Must catch per-iteration exceptions and continue** — a raise kills the trigger silently (`G-004` failure witness) |
| flush trigger parameters | `buster_ripper/config.py:DUMP_FLUSH_INTERVAL`, `DUMP_FLUSH_BYTES` (new globals) | NEW | — | `RB-002`. The injectability seam for the **flush triggers** (not the encoding). `KV_POLL_INTERVAL` (`:44`) supplies the shape; `F-2` stands (`DUMP_FLUSH_BYTES`, a data-volume basis) |
| dump dir config | `buster_ripper/config.py:DUMP_DIR` | reuse | — | Already the mode switch; `dump_turn` must guard it, `flush_all` must tolerate `None` |
| shutdown flush seam | `buster_ripper/server.py:on_cleanup` | extend | `EI:.../graceful_shutdown_hook` | Runs on aiohttp SIGINT/SIGTERM graceful exit; a raise here replaces `GracefulExit` (`web.py:511`) and the process exits with a traceback, so the flush is wrapped. Now cancels **two** tasks, then flushes, then closes the shared session — in that order |
| signal handling | aiohttp `web_runner` | reuse | `EI:.../graceful_shutdown_hook` | Native; verified `web_runner.py:283-284` → `web.py:511` → `web.py:425` → app cleanup hooks. No custom handler |
| dump caller | `buster_ripper/routes/messages.py:handle_messages` | reuse | `EI:.../dump_failure_isolated` | Only interception site today; no edit needed — the signature is unchanged. The append path now builds records, so the swallow-all contract covers more code |

**Read path — deliberately NOT an anchor row.** `RB-003`'s read-path search found **no reader in code**:
nothing under `buster_ripper/` or `tests/` reads `index.txt`, `turn_000.json`, or a `turn_NNN.diff`. The read path
today is the human workflow at `README.md:153-162`. `FK-007` selected **no tool**, so **no `Mode: NEW` read-side row is
added** and the read path stays the human workflow.

**Private helpers are not rows.** `T002`'s capacity extraction creates module-private functions inside `dump.py`
(state lookup, record construction, the two encodings, the artifact path builder). They are the same concepts decomposed for
LNCP, not new referents, and they do not authorize new artifacts — the creation gate's `Mode: NEW` rows above remain the
only authorization.

## Reuse Evidence

| Evidence | NEW anchor | Name search | Role/peer search | Native capability | Conclusion |
|---|---|---|---|---|---|
| `RB-004` | dump encoding selector (`config.py:DUMP_FORMAT`, `DUMP_FORMATS`; `main.py:--dump-format`) | `grep -rniE "format\|encoding\|encoder\|serializ\|dump_format"` over `*.py`/`*.md`/`*.sh`/`*.toml` (excluding `.sessions/`, `.venv/`, `__pycache__/`): **zero** encoding-selector matches. Every hit is unrelated — `encoding="utf-8"` (an I/O kwarg, `dump.py:82,95,105`), `transfer-encoding` (an HTTP hop-by-hop header, `routes/messages.py:62`), `format=` (the `logging.basicConfig` format, `main.py:102`), and prose. `buster_ripper_legacy.py:581,600,611` is the tracked pre-refactor monolith's copy of the same write sites, referenced by nothing under `buster_ripper/`, **excluded from scope**. No dump-encoding symbol, module, class, or command exists | No registries or Protocol/ABC contracts exist in this repository. **Nearest structural peer**: `config.PROFILES` (`config.py:87-96`) + `--eval-profile` (`main.py:45-48`) — a config-declared set of named values exposed as `click.Choice(list(config.PROFILES.keys()))` with a `default=` and the global assigned in `main()`. **Second instance**: `--backend-type` (`main.py:60`, a literal `click.Choice(["sglang","vllm"])` behind `config.BACKEND_TYPE_OVERRIDE`). Neither is reusable — one selects model-quirk strategy flags, the other a backend adapter — but the **pattern is established in-repo**, so `--dump-format` follows it rather than inventing one. `config.DUMP_DIR` (`:16`) is the mode switch this switch parallels. **No** existing global selects a record encoding | `click.Choice` is native and already used at `main.py:46` and `:60`; the value set is a module-level tuple; the branch is a string comparison. Python has no built-in encoding-registry facility and none is warranted for two values. Deps stay `aiohttp` + `click` (`pyproject.toml:5-8`); `json`/`difflib` already imported (`dump.py:4-5`) | No existing symbol owns the encoding-selector role. `config.DUMP_FORMAT`/`config.DUMP_FORMATS` + `--dump-format` are genuinely `Mode: NEW`; the nearest reusable thing is the established `config.PROFILES` + `--eval-profile` **shape**, not code |
| `RB-003` | *(unchanged from organize 013 — re-verified in scope this pass)* | The write-path and read-path searches stand: no buffering/batching/flush/artifact symbol exists, and **no reader** of `index.txt`/`turn_000.json`/`*.diff` exists in code | Stands | Stands | Unchanged. The encoding selector adds no second artifact, so `RB-003`'s artifact search still covers the write path |

`RB-001`/`RB-002` are unchanged and not restated here (see `plan-summary.md` §Reuse Evidence).

## Goal Definition Matrix (finalized)

`G-001`…`G-005` are carried from iteration 005's flush-policy cluster, with the `FK-002` name repairs
already landed. `G-001` gains the per-encoding clause. `G-006` is finalized below.

| Goal | Outcome or constraint | Observation surface | Met when | Failure witness | Boundary / non-goals |
|---|---|---|---|---|---|
| G-001 | Captured messages are held in memory and written in bulk instead of once per request | write-bearing file operations issued against `DUMP_DIR` for a session of N turns; on-disk state of `DUMP_DIR/<sid>/` | For N turns flushed `F` times, total file-write operations `= F` versus `2N` today; reduction `= 1 − F/(2N)`, i.e. **50% ≤ reduction < 100%** for every achievable `F` (an empty flush writes nothing, so `F ≤ N`) — **for every shipped encoding** | **A**: the flush writes one artifact per pending turn, so a flush of K turns costs K writes and the session total is `N + F` — buffering changed only *when* writes happen, never *how many*. **B**: the flush is one write, but it **rewrites** the whole artifact each time, so bytes written grow with the session (`O(N²)`) — the write *count* fell while the SSD write *volume* rose. Both leave the process healthy and the artifact correct | `/v1/messages` responses unchanged; durability still waived (`US-001`). The reduction is a count, and witness B is why the count alone is not the goal. **`F` is a per-`(sid, encoding)` count** — a flush that carries no pending record for a session writes nothing for it, which is what keeps `F ≤ N` in a multi-session process |
| G-002 | Buffered dump reaches disk when the process is asked to stop | `DUMP_DIR/<sid>/` contents after SIGINT/SIGTERM | After SIGINT or SIGTERM, buffered turns are present on disk, in order, **in the configured encoding**, and the process exits | Process exits cleanly (no traceback, exit 0) but `DUMP_DIR/<sid>/` is missing the turns that were buffered | SIGKILL/hard power loss out of scope; no custom signal handler |
| G-003 | Proxy behavior on `/v1/messages` is unchanged | HTTP responses + existing integration suite | Existing suite green; dump failure still cannot fail a request | A request fails or is delayed because of dump/buffer/flush behavior | No new endpoints, no normalization changes |
| G-004 | Buffered dump reaches disk periodically, without a shutdown, bounding staleness | the `DUMP_DIR/<sid>/` collapsed artifact (existence, contents, mtime) **while the process runs** | With the interval configured to `I`, a turn appended at `t` appears in the artifact no later than `t + I + ε`, with the process still running | The timer task exists, is spawned, and runs, yet the artifact never gains the turn for a live session — the loop's flush is a no-op against the wrong state, or an exception kills the task on the first tick and nothing logs it. Process healthy, staleness unbounded | No durability guarantee (`US-001`); no fsync; not a request-latency bound. Interval **value** is a `config` default (10 s), CLI-exposed |
| G-005 | Buffered dump reaches disk when the pending volume reaches the configured threshold, without a shutdown or a timer tick | `DUMP_DIR/<sid>/` contents after appending turns with the timer effectively disabled | Reaching the configured volume threshold puts the pending turns on disk with no shutdown and no timer | The threshold and its comparison exist but turns accumulate past it with nothing on disk until shutdown — the check reads the wrong field, or is evaluated before the append, or is off by one | No memory *cap* is promised; the basis is rendered-text bytes (`FK-B`) and the value is a `config` default, CLI-exposed |
| **G-006 — the dump stays readable (function, not layout)** | `README.md:134-135` ("useful for identifying remaining cache busters after normalization") + `README.md:146-162` (the documented reading workflow) + `FK-002`'s option text ("diffs become a read-time concern") | a reader inspecting only `DUMP_DIR/<sid>/` after a session | Given a session in which one known message changed at turn K, a reader inspecting only `DUMP_DIR/<sid>/` can locate turn K and identify what changed against turn K−1, **without additional tooling** — **per shipped encoding** (`F-11`). *change-focused* (the default): the clause holds **as written** — turn K's record carries the change against turn K−1 and the summary line names turn K, so `grep`/`less` on the one file locates it and shows the differing content, with no tooling. *self-contained*: the reader locates turn K by its summary line and record header, and identifies what changed by **comparing** turn K's record against turn K−1's — a reader-performed comparison, still with no tooling | The artifact is written once per flush, is well-formed, carries one encoding in arrival order, turn 0's body is present — and a reader holding only the directory cannot find the changed turn: the summary line lost the turn number or the change counts, or the self-contained records lost their per-turn identity. Every implementation task completed; the tool's documented purpose is gone | Not a requirement that the *layout* survive (`FK-002` displaced it); **not a requirement that a viewer tool be built** — `FK-007` option 1 ships none; not a formatting requirement; does not require the machine record (not added). Not a requirement that a previous version's files be migrated or removed | **RESOLVED → `FK-007` option 1**: reading **(b)** — the self-contained full-text encoding is the addition, **no tool**. Predecessors `FK-003` → `FK-004` → `FK-005` → `FK-006` (all superseded). ⚠ **`F-11`** resolved by the pick: the "without additional tooling" clause **holds as written** for the change-focused encoding and becomes a reader-performed comparison for the self-contained one — **no restatement is needed, because no tool ships**. ⚠ **`F-12`**: the artifact's filename states its encoding. ⚠ **`F-6`**: "full body" = turn 0's raw normalized JSON body **and** its rendering |

## Goal Coverage Matrix (finalized)

| Goal | Required capabilities / flows | Tasks | Validation points | Disposition |
|---|---|---|---|---|
| G-001 | capture on `/v1/messages` → build a record → append it to the per-session in-memory buffer → ONE write per flush to the collapsed artifact, in arrival order, **for every shipped encoding** | T001, T002 | VP-T001-writes-deferred, VP-T001-session-isolation, VP-T002-collapsed-artifact, VP-T002-write-count-reduced | mapped |
| G-006 | the collapsed artifact carries enough per-turn information for a reader to locate and read a change, in whichever encoding the session used | T002, T005 | VP-T002-change-readable, VP-T002-turn0-available, VP-T002-collapsed-artifact, VP-T002-default-encoding | mapped |
| G-002 | aiohttp SIGINT/SIGTERM graceful exit → `on_cleanup` → cancel both tasks → flush → close session → exit | T003 | VP-T003-shutdown-flush | mapped |
| G-003 | dump/flush call path stays failure-isolated and non-blocking for the request path | T001, T003 | VP-T001-dump-failure-isolated, VP-T003-passthrough-regression | mapped |
| G-004 | background timer coroutine → periodic `flush_all` → pending turns on disk while the process runs | T003 | VP-T003-interval-flush, VP-T003-timer-resilience | mapped |
| G-005 | append-time volume check inside `dump_turn` → flush when the threshold is reached | T002 | VP-T002-volume-flush | mapped |

Every task maps to a goal: `T001`/`T002` → G-001/G-005/G-006; `T003` → G-002/G-003/G-004; `T005` →
G-006; `T004` is an **enabling step** for the coverage gate that verifies G-001…G-006 (MS-003). No orphan task.

## Task List (finalized — done-when)

**T001 — Add the per-session pending-record buffer**
- ADDED: `SessionState` gains `pending: list[str]` (encoded records) and `pending_bytes: int` (the rendered-text byte
  measure — `FK-B`, `len(_render_body(body))`, what `dump.py:83` already computes).
- ADDED: `dump_turn` appends the turn's encoded record to `state.pending` and adds `len(current_text)` to
  `state.pending_bytes`.
- ADDED: `turn`/`prev_text` keep advancing at append time, unconditionally, exactly as today (`dump.py:108-109`).
- ADDED: unit coverage in `tests/test_dump.py` for the append and the byte accumulation.
- REMOVED: **nothing.** This increment is purely additive and green: the three write sites and the `index.txt` append stay
  live, so a turn still reaches disk the way it does today. The buffer is populated but not yet drained.
- **Depends on**: —.

**T002 — Collapsed artifact, encoding branch, single-write flush**
- ADDED: `config.DUMP_FORMATS: tuple[str, ...] = ("change-focused", "self-contained")` and
  `config.DUMP_FORMAT: str = "change-focused"` in `config.py`.
- ADDED: `main.py`'s `--dump-format` option — `type=click.Choice(list(config.DUMP_FORMATS))`,
  `default="change-focused"`, `show_default=True` — assigning `config.DUMP_FORMAT` (`RB-004`; the `--eval-profile` shape).
- ADDED: the record builder, extracted from `dump_turn` by the **LNCP capacity gate** into single-concern
  module-private helpers in `dump.py`: session state lookup/creation; the encoding dispatch; the **change-focused**
  record; the **self-contained** record; the artifact path builder. The encoding dispatch resolves
  `config.DUMP_FORMAT`; an unknown value raises `ValueError` (`PI:.../encoding_set_is_closed`) and is caught by the
  caller's swallow-all contract, so a request never fails.
- ADDED: the **change-focused** record — one summary line; turn 0 carries `json.dumps(body, indent=2, ensure_ascii=False)`
  **and** its `_render_body` rendering; turn ≥ 1 carries the `difflib.unified_diff` of `_render_body` output against the
  previous turn. **Behavior preserved byte-for-byte** from `dump.py:77-102`.
- ADDED: the **self-contained** record — one summary line; every turn carries its complete `_render_body` rendering;
  turn 0 additionally carries the full normalized JSON body (`EI:.../turn_zero_is_full_body`, `F-6`).
- ADDED: the summary line is built in **both** encodings and is encoding-independent: turn 0 the `FULL` form, turn ≥ 1 the
  `+added/-removed` form. The `+`/`-` counts are computed in both encodings; only whether the diff *text* is stored differs.
- ADDED: the artifact filename constant + path builder — `DUMP_DIR/<sid>/<encoding>.log`, where `<encoding>` is exactly the
  configured value (`F-12`; the closure packet in §Deterministic Closure Gate).
- ADDED: `flush_all()` — for each session with pending records, appends them to that session's artifact in **one**
  write-bearing file operation, then clears `pending`/`pending_bytes`. No-ops when `config.DUMP_DIR is None` or nothing is
  pending. **Must not raise. Contains no `await`.** Never touches `turn`/`prev_text`.
- ADDED: `dump.py`'s module docstring describes the collapsed artifact and both encodings.
- ADDED: `dump_turn`'s bare `assert config.DUMP_DIR is not None` (`dump.py:63`) is replaced by a real guard that raises
  `ValueError` (`F-5`).
- ADDED: unit coverage for both encodings, the artifact shape and name, the encoding guard, and the `DUMP_DIR` guard.
- REMOVED: the `turn_000.json` write (`dump.py:81-82`), the `turn_{N:03d}.diff` write (`:94-95`), and the
  `index.txt` append (`:104-106`).
- REMOVED: `dump_turn`'s inline record construction (relocated into the extracted helpers, not duplicated).
- **Depends on**: T001.

**T003 — Wire the flush: volume trigger, timer, shutdown**
- ADDED: `config.DUMP_FLUSH_INTERVAL: int = 10` (seconds — matching `config.KV_POLL_INTERVAL`) and
  `config.DUMP_FLUSH_BYTES: int = 4 * 1024 * 1024` (a single-digit MiB, `FK-B`) in `config.py`.
- ADDED: `main.py`'s `--dump-flush-interval` and `--dump-flush-bytes` options assigning those globals.
- ADDED: the append-time volume check in `dump_turn` — after the append, flush when
  `state.pending_bytes >= config.DUMP_FLUSH_BYTES` (`G-005`; the check must read the same field the append wrote).
- ADDED: the periodic flush coroutine's loop body in `dump.py` — `flush_all()`, then
  `await asyncio.sleep(config.DUMP_FLUSH_INTERVAL)`; **per-iteration exceptions are caught and logged and the loop
  continues** (`G-004` failure witness).
- ADDED: `server.py`'s spawner + `app["dump_flush_task"]` key, following `spawn_kv_cache_monitor`.
- ADDED: `server.on_cleanup` cancels the flush task (wrapped, `asyncio.CancelledError` swallowed), then calls
  `flush_all()` wrapped so it cannot raise, then closes the shared session — **in that order**.
- ADDED: unit/integration coverage for the volume trigger, the timer loop, timer resilience, and `on_cleanup`.
- REMOVED: **nothing.**
- **Depends on**: T002.

**T004 — Tests and coverage**
- ADDED: `tests/test_dump.py`: buffer append and byte accumulation; exactly one write-bearing operation per flush
  (with K ≥ 2 pending turns); the collapsed-artifact shape (**exactly one file for the configured encoding**, no
  `turn_000.json` / `turn_NNN.diff` / `index.txt`); the filename states the encoding; arrival order; change readability
  **per encoding**; turn-0 availability **per encoding**; session isolation; dump/flush failure isolation; the volume trigger;
  the timer loop and its resilience; `on_cleanup`; real-signal (subprocess) cases; the **write-count test run per shipped
  encoding** (`PI:.../encoding_does_not_change_write_count` has no gate otherwise); the default-encoding test; the unknown-encoding
  and `DUMP_DIR is None` guards.
- ADDED: coverage of every new `dump.py` statement, holding `U ≤ 3.2 + 0.2·K`.
- REMOVED: nothing — no dump tests exist today.
- **Depends on**: T001–T003.

**T005 — Rewrite the dump section of the README**
- ADDED: `README.md`'s dump section documents the artifact's name and shape (`<encoding>.log` under
  `DUMP_DIR/<sid>/`), **each shipped encoding and how it is selected** (`--dump-format`, default `change-focused`), and
  the reading workflow that replaces "Reading the index" — the `grep`/`less` commands that spot a cache buster in each encoding.
- ADDED: `F-8`'s limitation — the artifact's turns are not one linear conversation (42 of 137 transitions in the real
  dump have fewer messages than the previous turn), so a reader must read turns in arrival order.
- ADDED: a note that files from a previous version's layout (`turn_000.json`, `turn_NNN.diff`, `index.txt`) in a
  pre-existing session directory are neither migrated nor removed.
- REMOVED: the `turn_000.json` / `turn_NNN.diff` / `index.txt` layout block and the "Reading the index" sample output.
- REMOVED: **no tool section** — `FK-007` option 1 ships no tool.
- **Depends on**: T002.

**Not tasks.** The `Approved Override Register` entries are `plan-summary.md` artifacts. `buster_ripper_legacy.py:41-45`
documents the same displaced layout; it is the tracked pre-refactor monolith referenced by nothing under `buster_ripper/`,
**excluded from scope**. The `qa-project` skill is a capability gap, not a task (see §Gates Found Unmet).

## Milestone Matrix (finalized)

| Milestone | Constituent tasks | Complete when | Validation points | QA author pass |
|---|---|---|---|---|
| MS-001 | T001, T002, T005 | A buffered session's turns reach disk only after a flush — a volume-triggered flush or a shutdown flush — **in the collapsed format**, in arrival order, with turn 0's body present, the write count `F` with `F` controlled, and the artifact's name stating its encoding; the README documents the new artifact | VP-T001-writes-deferred, VP-T001-session-isolation, VP-T001-dump-failure-isolated, VP-T002-collapsed-artifact, VP-T002-change-readable, VP-T002-turn0-available, VP-T002-write-count-reduced, VP-T002-volume-flush, VP-T002-default-encoding | QAP-MS001 |
| MS-002 | T003 | After a real SIGINT/SIGTERM, buffered turns are in the collapsed artifact and the process exits cleanly; **and** with the timer running, a pending turn reaches the artifact without shutdown | VP-T003-shutdown-flush, VP-T003-interval-flush, VP-T003-timer-resilience | QAP-MS002 |
| MS-003 | T004 | Existing suite green with the new `dump.py` lines covered and total coverage ≥ 80% | VP-T003-passthrough-regression | — (existing suite is the gate) |

## Validation Point Matrix (finalized)

| Validation point | Observable behavior | Observable after | Validate at | QA author pass | Scope reason | Evidence needed |
|---|---|---|---|---|---|---|
| VP-T001-writes-deferred | After N intercepted requests below **both** thresholds and with the timer effectively disabled, no artifact containing those turns exists on disk | **T002** *(corrected from T001)* | MS-001 | QAP-MS001 | The deferral is the black-box form of "buffering". It becomes observable only once the per-turn write sites are gone — that is `T002`, not `T001`. Validating it at `T001` would be **artificial**: at `T001` the writes are still live by design, so the VP could only pass by asserting something the increment does not do. Whether the session *directory* is created at append or at flush is a forward detail and either satisfies this | `DUMP_DIR/<sid>/` listing |
| VP-T001-session-isolation | Two distinct system prompts produce two `DUMP_DIR/<sid>/` artifacts, each flushed independently | **T002** *(corrected from T001)* | MS-001 | QAP-MS001 | Keying is unchanged but now gates the buffer; independent *flushing* needs `flush_all` to exist, which is `T002` | two dirs + contents |
| VP-T001-dump-failure-isolated | A raising dump/flush still yields the upstream response body | T001 | MS-001 | QAP-MS001 | The request path is unchanged and the append path exists from `T001`; observable as soon as the buffer exists. The append path now builds records, so the swallow-all contract covers more code | HTTP status/body + logged exception |
| **VP-T002-collapsed-artifact** | After a flush of turns 0..2, `DUMP_DIR/<sid>/` contains **exactly one artifact for the configured encoding** and **no** `turn_000.json`, `turn_NNN.diff`, or `index.txt`; **the artifact's filename states its encoding (F-12)**; and the artifact carries the three turns' records in arrival order | T002 | MS-001 | QAP-MS001 | The on-disk shape is only observable once a multi-turn flush exists; the "exactly one file" clause is `FK-002`'s requirement made visible, and the name clause is `F-12`'s. **Identity is `(sid, encoding)`**: a session directory may legitimately hold one artifact per encoding after a process restart with a different `--dump-format`, so this VP is scoped to a **single** configured encoding and does **not** assert that the directory holds only one file in total | `DUMP_DIR/<sid>/` listing + artifact contents |
| **VP-T002-change-readable** | **Per encoding.** *change-focused*: given a session where turn 2 changes one known message, a reader inspecting only `DUMP_DIR/<sid>/` can locate turn 2 **and see the differing content** against turn 1 in turn 2's record. *self-contained*: the reader locates turn 2 by its summary line and record header, and **identifies the differing content by comparing** turn 2's record against turn 1's. Neither requires a tool | T002 | MS-001 | QAP-MS001 | Ordering and per-turn identity are only observable across a multi-turn flush. **Qualified per encoding (`F-11`)**: the clause holds as written for change-focused and becomes a reader-performed comparison for self-contained. Validating this only against the default encoding would leave the self-contained clause with no gate | artifact contents, per encoding |
| **VP-T002-turn0-available** | Turn 0's full normalized JSON body **and** its `_render_body` rendering are present in the artifact and recoverable from it — **in both encodings** (`F-6`) | T002 | MS-001 | QAP-MS001 | Carries `EI:.../turn_zero_is_full_body`, which had no VP. Fails for any encoding that stores only diffs; the self-contained encoding is the one at risk, since its records are renderings | artifact contents, per encoding |
| VP-T002-write-count-reduced | For N turns flushed `F` times, instrumented write-bearing file operations `= F`, versus `2N` today; reduction `= 1 − F/(2N)`; **run per shipped encoding**. The per-flush form is folded in: a flush carrying K ≥ 2 pending turns performs **exactly one** write-bearing operation | T002 | MS-001 | QAP-MS001 | Structural/instrumented proof — must be paired with the black-box VPs so a mock cannot mirror the implementation. **`F` must be deterministic**, so the test injects the parameters and disables the timer. Unit: one write-bearing file operation on a dump artifact (`write_text`, or `open(...,"a"/"w")`+`write`); directory creation is not counted (once per session). **Per encoding**, or `PI:.../encoding_does_not_change_write_count` has no gate | instrumented write-call count + deferred-state observation + a known, injected `F` |
| VP-T002-volume-flush | With the interval injected large (timer cannot fire), reaching the configured volume threshold puts the pending turns in the artifact with no shutdown | T002 | MS-001 | QAP-MS001 | Needs the flush + the append-time threshold check; wording is basis-agnostic so it survives the `FK-B` default | artifact contents before/after the threshold turn |
| **VP-T002-default-encoding** | With **no** `--dump-format` passed and no config override, the artifact produced is the **change-focused** encoding — filename `<change-focused>.log`, records are the summary line + turn 0's full body and rendering + per-turn diffs | T002 | MS-001 | QAP-MS001 | Makes the default an observable contract rather than a config detail. `US-008`(a) left it open; decide 014 closed it deterministically and `FK-007`'s preamble (2) stated it with a one-clause correction path the user did not take | `DUMP_DIR/<sid>/` listing + artifact, with no `--dump-format` passed |
| ~~VP-T002-human-facing~~ | **NOT CREATED** — `FK-007` option 1 ships no tool; it was a candidate only for options 2/3 | — | — | — | **Not created.** `US-008`'s "more human-visible or human-friendly" is satisfied by the self-contained encoding, whose observable contract is `VP-T002-change-readable` + `VP-T002-collapsed-artifact`; no tool ships, so there is no tool output to gate | — |
| ~~VP-T002-single-write-per-flush~~ | **RETIRED** — folded into `VP-T002-write-count-reduced` (`F-3`: keep one write-count VP) | — | — | — | The handle is retired and must not be reused. Its clause is carried verbatim in `VP-T002-write-count-reduced`'s "per-flush form" text | — |
| VP-T003-interval-flush | With the interval injected small and the volume threshold set unreachable, a single appended turn appears in the `DUMP_DIR/<sid>/` artifact while the process keeps running | T003 | MS-002 | QAP-MS002 | The timer only exists once wired; the volume trigger is disabled so only the timer can satisfy it | live artifact contents/mtime with no shutdown |
| VP-T003-timer-resilience | A flush that raises inside the timer loop is logged and the loop keeps flushing on later ticks; a cancelled timer stops without a traceback | T003 | MS-002 | QAP-MS002 | The `G-004` failure witness is a dead trigger in a healthy process; only a live loop can show it | log record + a later successful tick; exit code |
| VP-T003-shutdown-flush | After SIGINT or SIGTERM, buffered turns are on disk and the process exits without a traceback | T003 | MS-002 | QAP-MS002 | Requires process-level signal delivery; no in-process mock can prove it | subprocess + real signal + post-exit dir listing |
| VP-T003-passthrough-regression | Existing `tests/test_routes_integration.py` suite green | T003 | MS-003 | — (existing suite) | Regression gate | green suite |

## Behavior Disposition Matrix (finalized)

| Behavior | Disposition | Validation point |
|---|---|---|
| B-001 dump output layout and index format | mapped (re-dispositioned) — the layout is *replaced*, the readable-layout **function** is preserved | VP-T002-collapsed-artifact |
| B-002 diff chain correctness | mapped (re-dispositioned) — per-turn change information in arrival order, per encoding | VP-T002-change-readable |
| B-007 (new) turn 0's full body preserved and recoverable | mapped — carries `EI:.../turn_zero_is_full_body` | VP-T002-turn0-available |
| B-003 dump failure isolated from request | mapped | VP-T001-dump-failure-isolated |
| B-004 `on_cleanup` shutdown behavior | mapped | VP-T003-shutdown-flush |
| B-005 proxy response passthrough unaffected | mapped | VP-T003-passthrough-regression |
| B-006 session directory isolation | mapped | VP-T001-session-isolation |
| (new) buffering deferral | mapped | VP-T001-writes-deferred |
| (new) write-count reduction, per encoding | mapped | VP-T002-write-count-reduced |
| (new) volume-triggered flush | mapped | VP-T002-volume-flush |
| (new) time-triggered flush | mapped | VP-T003-interval-flush |
| (new) timer resilience | mapped | VP-T003-timer-resilience |
| (new) the configured encoding determines the record built | mapped | VP-T002-default-encoding, VP-T002-collapsed-artifact |
| (new) the artifact's name states its encoding | mapped | VP-T002-collapsed-artifact |

## Capacity Gate Baseline (`dump.py`)

| Surface | Raw lines | Level | Function | L (executable stmts) | Trigger |
|---|---|---|---|---|---|
| `buster_ripper/dump.py` | 109 | Green (≤300) | `dump_turn` (`:61-109`) | **≈34 → hard violation** (>30) | **TRIGGERED for T002** |
| `buster_ripper/config.py` | 104 | Green | — (new globals only) | — | not triggered |
| `buster_ripper/main.py` | 130 | Green | — (new options only) | — | not triggered |

`dump_turn` already carries a hard LNCP length violation and `T002` adds branches and statements to it, so the
gate fires. `T002` therefore performs the smallest coherent extraction **inside its own touch set, before adding
behavior**: the encoding dispatch, the summary-line construction, the turn-0 baseline, and the two encodings become
single-concern module-private helpers in `dump.py`; `dump_turn` becomes session lookup + append + volume check. This
is a relocation of this row's own concern, not a second serializer — `dump_turn` no longer builds the record itself, so no
wrapper or relay is created. The extraction rides `T002`'s increment; it is not a separate task, checkpoint, or Yield cycle.
File-level: `dump.py` is projected at roughly 150–170 lines after the change, still Green. Forward records
`Capacity gate:` with the post-extraction metrics; no post-change metric may increase or enter a new violation zone.

## Adversarial Surface Discipline (design-time guards, encoded as fail-first tests in `T004`)

| Rule | Surface in this plan | Guard encoded |
|---|---|---|
| **R1** hostile inputs at every parse boundary | `--dump-format` (CLI) and `config.DUMP_FORMAT` (programmatic). The record path is not a network parse boundary — `_render_body` already tolerates missing/odd keys with `.get()`/`or []`/`or ""` (`dump.py:31-56`) | `click.Choice` rejects an unknown CLI value with a usage error; the encoding dispatch raises `ValueError` for an unknown configured value, caught by the caller's swallow-all contract. Both are fail-first tests |
| **R2** durable-writer discipline into pre-existing trees | `DUMP_DIR/<sid>/` may already hold a previous version's `turn_000.json` / `turn_NNN.diff` / `index.txt` (a session spanning the upgrade), and may already hold an artifact from a run with a different `--dump-format` | The artifact is **append-only** and is never rewritten or truncated; the encoding in the filename means a new artifact never collides with an old one or with the other encoding; the old-layout files are **neither migrated nor removed** (scope exclusion, documented in `T005`). `VP-T002-collapsed-artifact` is scoped to a **fresh** session directory so it is not falsified by unrelated pre-existing files |
| **R3** capture→act windows | The flush reads `pending`, writes the artifact, then clears `pending`. On asyncio's single thread a concurrent append cannot interleave **only because the flush contains no `await`** — this is the protected invariant that makes all three triggers safe | The protected invariant is asserted by `T004`; `flush_all()`'s no-`await` clause is in `T002`'s done-when |
| **R4** payload decoder census | The artifact's only decoder is the human reading workflow. `RB-003`'s read-path search found **no** code reader of `turn_000.json` / `*.diff` / `index.txt`, and the encoding adds none (no tool ships). The census is therefore `{README.md:146-162}` — which is exactly what `T005` rewrites | `T005`'s done-when names the reading workflow per encoding; the census has one member |
| **R5** identity-key and boundary-value completeness | The artifact's identity key is **`(sid, encoding)`**, not `sid`: after a restart with a different `--dump-format`, the same session directory legitimately holds a second artifact. The encoding set's boundary values are the two declared values, and an unknown one is a hard failure. The volume threshold's boundary is `pending_bytes >= DUMP_FLUSH_BYTES` evaluated **after** the append | `VP-T002-collapsed-artifact` scopes "exactly one artifact" to the configured encoding; `PI:.../encoding_set_is_closed` and `PI:.../encoding_is_per_artifact` state the key; the `VP-T002-volume-flush` clause "the check reads the same field the append wrote" is the boundary guard |

## Decide-Settled Design Details (visible, overturnable at final approval)

| Detail | Reading | Alternatives considered | Ground |
|---|---|---|---|
| The artifact's filename | `DUMP_DIR/<sid>/<encoding>.log`, where `<encoding>` is exactly the configured value (`change-focused.log`, `self-contained.log`) | `session.diff` / `session.full` (the extension alone distinguishes them); an in-band first-line marker | The closure packet above; `F-12`; `VP-T002-collapsed-artifact`. The extension-only alternative is rejected because its token would differ from the configured value, reintroducing the lookup `F-12` exists to remove |
| The summary line is encoding-independent | Both encodings build the same summary line; the `+added/-removed` counts are computed in both (the diff is computed either way) and only the diff *text* is stored in the change-focused encoding | Omit the counts from the self-contained summary line | The `dump summary line` row is `extend` and encoding-independent; the counts are what let a reader spot a cache buster at a glance in either encoding |
| Flush scope | `flush_all()` flushes every session with pending records; the volume trigger calls the same no-arg function | A per-session `flush(sid)` | `_sessions` stays private to `dump.py` (no new accessor), and `G-001`'s `F ≤ N` bound is unaffected: a flush that carries no pending record for a session writes nothing for it |
| Unknown encoding | The dispatch raises `ValueError` | Silently fall back to the default encoding; silently no-op | `PI:.../encoding_set_is_closed`. The caller's swallow-all contract makes a raise safe for the request path |
| Previous-version files in a session directory | Neither migrated nor removed | Migrate on first write; delete on first write | Out of scope (`FK-002` displaces the layout; no user instruction covers migration). Deleting would be a destructive act with no authorization |

## Dropped-Clause Sweep

Triggered: this pass **replaces prior definitions of the same scope** — `G-006`'s single `Met when` clause, the
three `OI:` replacement statements (which stood as "pending"), `T002`/`T004`/`T005`'s descriptions, and three
`VP` rows. Direct predecessor only, one hop. The `FK-006`→`FK-007` sweep is 014's and is not repeated.

| Predecessor clause | Disposition | Why / where it now lives |
|---|---|---|
| `G-006` `Met when` — single clause, "without additional tooling" | **replaced** | Now written **per encoding** (`F-11`): holds as written for change-focused, becomes a reader-performed comparison for self-contained. §Goal Definition Matrix |
| `G-006` `Met when` — "a reader can locate turn K and identify what changed against turn K−1" | **carried** | Same clause, now qualified per encoding |
| `G-006` failure witness — "a reader cannot find the changed turn or see what changed" | **carried + sharpened** | The change-focused encoding lost its per-turn diff; the self-contained encoding lost per-turn identity. §Goal Definition Matrix |
| `G-006` boundary — "not a requirement that a viewer tool be built" | **carried, now decided** | `FK-007` option 1 ships none; the clause is no longer contingent. §Goal Definition Matrix |
| `G-006` boundary — readings (a)–(d), with (c) "only by shipping a viewer" | **replaced** | Resolved to reading (b) by the pick; the reading list is no longer open. §Goal Definition Matrix |
| `OI:output_layout` replacement statement — "whose record encoding is the configured one (`US-007`) and whose name states that encoding" | **carried + expanded** | Now names both encodings' record shapes, the `<encoding>.log` name, and `F-6`'s definition of "full body". §Approved Override Register |
| `OI:sequential_diff_chain` replacement — "…`prev_text` advances at append time … for the change-focused encoding" | **carried** | Unchanged; `F-11` qualification retained. §Approved Override Register |
| `OI:turn_zero_is_full_body` replacement — "turn 0's full normalized JSON body **and** its rendered rendering …" | **carried** | Unchanged. §Approved Override Register |
| `T002` description — "the artifact's name states the encoding (F-12)" | **carried + made concrete** | The name scheme is now stated (`<encoding>.log`) and the encoding selector is in `T002`'s scope. §Task List |
| `T002` description — "A `flush_all` appends all pending records to the per-session collapsed artifact in **one** write" | **carried** | Unchanged in substance; the no-`await`/must-not-raise/no-op clauses are now in done-when form |
| `T005` description — "If a tool is picked, this section documents the tool as well" | **dropped — deliberate** | No tool ships (`FK-007` option 1). Recorded in §Task List `T005` REMOVED |
| `VP-T002-collapsed-artifact` — "exactly one artifact file" | **carried + scoped** | Scoped to "for the configured encoding" per `R5`'s `(sid, encoding)` key |
| `VP-T002-change-readable` — format-agnostic single clause | **carried + qualified** | Per encoding per `F-11`. §Validation Point Matrix |
| `VP-T002-single-write-per-flush` row | **dropped — deliberate** | Folded into `VP-T002-write-count-reduced` per `F-3`; the handle is retired and not reused. §Validation Point Matrix |
| `VP-T002-human-facing` row | **dropped — deliberate** | `FK-007` option 1 ships no tool. §Validation Point Matrix |
| `VP-T001-writes-deferred` / `VP-T001-session-isolation` — `Observable after: T001` | **replaced** | Corrected to `T002`; the behavior does not exist until the write sites are removed. §Validation Point Matrix |
| `G-001` `Met when` — "reduction `= 1 − F/(2N)`, 50% ≤ reduction < 100%" | **carried + scoped** | Now "for every shipped encoding", and `F` is stated as a per-`(sid, encoding)` count. §Goal Definition Matrix |
| Milestone / behavior-disposition / anchor rows not listed | **carried verbatim** | All other clauses carried verbatim. |

**One candidate NOT adopted, recorded so it is not silently lost.** The **conversation-scoped encoding variant**
(lifting the constant tool definitions and system prompt out of a record; ≈25% saving, real only for the self-contained
encoding) was carried by 014 as a PLAN-SELECTED candidate. `FK-007`'s pick took no named modification, and the encoding set
is closed at two, so it is **not adopted**. It is not added to the roadmap either — `FK-007`'s `Impact` names the two roadmap
entries, and inventing a third would exceed what the user approved.

## Compaction Decision

- Prior same-phase files reviewed: `iterations/003-decide.md`, `008-decide.md`, `010-decide.md`, `011-decide.md`,
  `012-decide.md`, `014-decide.md`.
- **Superseded**: none.
- **Not superseded**: all six.
- **Compaction performed: no.**
- **If no**: the slice differs. `003` is the initial options pass (`FK-001`/`FK-002`); `008` the flush-policy pass plus the
  `FK-A`/`FK-B` escalation-floor re-tests; `010`/`011`/`012` the three prior payload repairs; `014` the `FK-006` repair and
  the `FK-007` authoring. This pass is **PLAN-SELECTED finalization** — it supersedes no FK and re-derives no option. `014`
  in particular holds the fusion diagnosis, the closure packet, the measured byte figures, and the construction-boundary settlement
  that this pass consumes; `012` holds the F-10/F-11/F-12 derivations; `011`/`010` hold the F-6/F-8/F-9 derivations.
  Deleting any would remove reasoning the live plan depends on. `decisions.md` §`FK-006`/§`FK-007` are self-contained.
- **`plan-summary.md` was rewritten (not a report)**: it is the living plan, its ≤400-line cap requires the process
  narrative to move out, and no load-bearing clause was dropped — see the sweep above. The displaced narrative lives in this
  report and in `014`/`013`/`012`.

## Deferred Questions

- **Which device/filesystem hosts `DUMP_DIR` in production** (Linux ext4/XFS beside vLLM, or macOS APFS on the dev box)?
  - Status: DEFERRED for /codify-review (unchanged since 008).
  - Impact: the exact write-amplification factor differs; the directional verdict does not.
  - Agent analysis: `none` — a factual premise, not a user preference.
- **QA infrastructure setup** (`.agents/skills/qa-project/SKILL.md` absent).
  - Status: DEFERRED — a presentation **topic** for interview, not an FK. Code-natural reading: stub upstream
    (`tests/test_routes_integration.py:32-60` mocks it), `tests/qa/`, subprocess + real signal.
  - Impact: `QAP-MS001`/`QAP-MS002` cannot run without it; the harness must inject both flush-trigger parameters at
    process start or `VP-T002-write-count-reduced`, `VP-T003-interval-flush`, and `VP-T002-volume-flush` cannot be deterministic.
  - Agent analysis: `none` — infrastructure availability, not a user preference.
- **`F-5`'s guard shape** (`raise ValueError` vs a silent `return`).
  - Status: NOT deferred — decide settled it (§Decide-Settled Design Details) and it is recorded in `T002`'s done-when.
  - Impact: `T002`'s done-when and `T004`'s guard test.
  - Agent analysis: **ratification-flavoured but not a fork** — the escalation floor does not fit (a bare
    `assert` stripped under `-O` versus a raise is not "irreversible or expensive to reverse"), so it is a Forward
    discretion item that decide settles to keep the plan unambiguous.

## Questions Asked

**No questions** — deferred mode. The one live user decision was `FK-007`, already resolved by the pick. No
`AskUserQuestion` call was made; decide does not ask, and PLAN-SELECTED re-derives nothing the pick settled.

## Validation Performed

| Claim | How verified | Result |
|---|---|---|
| Session identity / handle suffix | `session-identity.json` vs handle | matches (`9bf7afff5573` = first 12 hex of UID) |
| No parent coordination contract | `session-identity.json` has no `parent_session_uid` | confirmed — Parent Contract Reconciliation is N/A |
| `FK-007` is the only resolved fork and nothing is open | `grep` of `State:` in `decisions.md` | `FK-001`/`FK-002`/`FK-007` resolved; `FK-003`/`FK-004`/`FK-005`/`FK-006` superseded; **none open** |
| `FK-007`'s pick is option 1 and nothing else | `decisions.md` §`FK-007` → `Resolution.Normative choice` | confirmed: two encodings, no tool, machine record not added |
| The pick left nothing open | `FK-007` → `Resolution` sub-comment | confirmed — no successor FK and no `US-*` owed |
| No new FK/US reserved this pass | no allocator call made | confirmed — no id written by hand |
| `dump_turn` is the whole record path | `dump.py:61-109` | confirmed — one function, two branches, one summary append |
| `dump_turn`'s executable-statement count | counted `dump.py:63-109` | **≈34 → hard LNCP length violation**; capacity gate TRIGGERS for T002 |
| The three write sites | `dump.py:81-82`, `:94-95`, `:104-106` | confirmed — `turn_000.json`, `turn_{N:03d}.diff`, `index.txt` |
| The bare assert | `dump.py:63` | confirmed — `assert config.DUMP_DIR is not None` |
| `difflib` import line | `dump.py` | **line 4** (`:5` is `import json`) — 013 Correction 3 stands |
| The summary construction exists inline | `dump.py:84`, `:98-101` | confirmed — no symbol owns it |
| `config.DUMP_DIR` / `KV_POLL_INTERVAL` | `config.py:16`, `:44` | confirmed |
| A config-declared named-value selector with a CLI flag exists | `main.py:45-48` (`--eval-profile`, `click.Choice(list(config.PROFILES.keys()))`), `main.py:60` (`--backend-type`) | **confirmed — `F-1` is narrowed, not contradicted** |
| `--dump-dir` is the only dump CLI option | `main.py:25-26` | confirmed — `--dump-format` is new surface |
| No dump-encoding selector exists | `grep -rniE "format\|encoding\|encoder\|serializ\|dump_format"` over `*.py`/`*.md`/`*.sh`/`*.toml` | confirmed — only `encoding="utf-8"`, `transfer-encoding`, `format=`, and prose |
| No reader of dump artifacts in code | `grep` over `buster_ripper/` + `tests/` | confirmed — writers and prose only (`RB-003` stands) |
| No `.invariants/` registry | `ls .invariants` | absent — `EI:`/`PI:` are session-local |
| Coverage omit list excludes `server.py`, `main.py`, `routes/*` | `pyproject.toml` `[tool.coverage.run].omit` | confirmed — the timer loop body and record builder must live in `dump.py` to be testable |
| `on_cleanup` ordering seam | `server.py:60-72` | confirmed — cancel, then close |
| The real dump is unreachable | `ls /Volumes` | `data` **not** mounted → every byte figure is carried, not re-measured |

## Gates Found Unmet (reported, not cured)

1. **`.agents/skills/qa-project/SKILL.md` does not exist** while the plan declares `QAP-MS001`/`QAP-MS002`.
   Blocked on interview Topic 2; not written here.
2. **Final approval remains blocked** on (1) and on the user's `User Approved: yes` stamp on each `OI:` entry,
   which decide cannot write.
3. **`FK-007`'s `Impact.Plan` was `pending`** — updated by this pass to name the landed destinations.
4. **The real dump is unreachable** (`/Volumes/data` unmounted), so every byte figure in the plan is carried,
   not re-measured. `US-007` claim 3 and `FK-006`'s `F-10` rest on them.
5. **`F-1` is narrowed, not deleted** — it remains true that no *numeric* `config` global has a CLI flag.
6. **`FK-006`'s `Impact` still names the artifact row's "serializer"** as encoding-parameterized. The
   superseded record is preserved as review evidence; 014 corrected the boundary in place.

## Next Step

1. **`invite`** presents the finalized plan with the `FK-007` outcome applied and asks the first alignment question.
   There is **no open FK** — the remaining decision points are the `OI:` approval and Topic 2.
2. **`interview`** walks Topic 2 (QA infrastructure setup), records the `User Approved: yes` stamp on each `OI:`
   entry, and seeds `roadmap.md` at final approval.
3. Final approval stays blocked until the `OI:` stamps land and the `qa-project` skill exists.

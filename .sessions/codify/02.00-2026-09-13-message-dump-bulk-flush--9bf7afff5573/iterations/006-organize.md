# 006 — Organize (REPLAN-INCREMENT: payload/format cluster)

## Orchestrator Prompt (verbatim)

> Execute the organize-interactive skill for this session. Arguments: $ARGUMENTS
>
> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
> Mode: replan
> Cluster: payload/format cluster
> Reason (from Interview): REPLAN-INCREMENT — the user answered the payload question
> by rejecting "keep the documented layout" and choosing to change the flush format so
> one bulk flush is a single write, on the stated grounds that a write-per-turn flush
> would kill the purpose and any benefit of saving SSD. This cluster was never planned
> to agent-ready depth. Concretely: VP-T002-layout-preserved and VP-T002-diff-chain are
> dead with no replacements; G-001's criterion and VP-T002-write-count-reduced must move
> from N + F to F; MS-001's completion condition still names the displaced layout;
> RB-001 must be re-run for the new write AND read paths; T002/T004 done-when text and
> the README.md:146-162 rewrite have no tasks; and EI:.../output_layout and
> EI:.../sequential_diff_chain now require an Approved Override Register entry rather
> than a conditional one. The flush-policy cluster from iteration 005 (RB-002, the timer
> rows, G-004/G-005, coverage headroom U ≤ 3.2 + 0.2·K) is untouched and stays settled.
> Additionally, FK-002 fixes the requirement (one write per flush) but not the format
> itself — whether index.txt survives, whether the per-turn diffs remain readable and
> who reconstructs them at read time, append-only vs rewritten per flush, and what
> replaces the documented reading workflow. Decide owns that as a successor FK. Also
> flagged: FK-A's recorded If-wrong figure (½·(1 − F/N), zero at F = N) is the
> documented-layout arithmetic and is now wrong — under the collapsed payload a flush
> costs F writes for N turns, so a short interval adds a linear cost instead of erasing
> the benefit; that premise must be repaired before decide authors FK-A.
>
> [Remainder of the prompt was the `organize-interactive` skill body plus the
> `codify-agents`, `critical-thinker`, and `codify-user-prompt` skill bodies.
> `$ARGUMENTS` was empty. Question Mode is `deferred` per session metadata; no
> active-mode phrase was present.]

## Session Type / Mode

- New parent session (`02.00`), no `parent_session_uid` → the parent coordination
  contract does not apply. No parent path to read or reconcile.
- **Input is the interview report** (`iterations/004-interview.md` §Round 2) plus the
  prior collect/organize/decide reports. Per the interview-as-collect-input rule the
  interview findings are the primary source; they override the stale assumptions they
  correct (`FK-002`'s successor-question list, the displaced `N + F` arithmetic).
- Scope is the **payload/format cluster only**. `T001`, the flush-policy cluster's
  decisions (`RB-002`, the timer rows, `G-004`/`G-005`, the coverage formula), and the
  milestone skeleton are treated as settled and are not re-litigated.

## Context Conflict Detected

One, and it is a naming conflict rather than a decision conflict.

- **Source A** (priority 1 — own session canonical surfaces, `FK-002.Impact.Plan` and
  `plan-summary.md` §Goals and Constraints): `G-001`'s criterion and
  `VP-T002-write-count-reduced` must move from `N + F` to `F`.
- **Source B** (priority 2 — `iterations/005-organize.md` §Goal Shape Analysis,
  §Successor Question Scope, and `plan-summary.md` §Goal Definition Matrix as of that
  pass): the criterion is `N + F` with reduction `½·(1 − F/N)`.
- **Resolution**: following Source A. Iteration 005 authored the `N + F` figure for the
  *documented* layout, which was still the open reading at the time; `FK-002` resolved
  the other way afterwards. `FK-002`'s own `Impact.Plan` clause names this exact
  replacement, so 005's figure is superseded by the FK it predates. 005's *decisions*
  (timer mechanism, `RB-002`, `G-004`/`G-005`, coverage formula) are untouched — only
  the two arithmetic figures it authored against the documented layout are replaced.

No other conflict: `decisions.md`, `plan-summary.md`, `handoff-context.md`, and
`iterations/004-interview.md` agree with each other and with source on the format
choice.

## Changes Since Iteration 005

- NEW: **`RB-003`** authored — the re-run of `RB-001` for the collapsed format, covering
  the write path *and* the read path. Headline result: **no code read path exists and
  none is required**; the read path is documented prose.
- RE-DERIVED: `G-001`'s criterion and failure witnesses. Reduction is `1 − F/(2N)`,
  whose **floor is 50%** — the interval can no longer erase the goal.
- RE-DERIVED: `VP-T002-write-count-reduced` target (`= F` versus `2N`).
- NEW: `G-006` (dump readability preserved) — the boundary the collapse must not
  silently discard, and the goal the successor FK actually arbitrates.
- NEW: replacements for the two invalidated VPs — `VP-T002-collapsed-artifact` and
  `VP-T002-change-readable` — plus `VP-T002-turn0-available`, which preserves
  `EI:.../turn_zero_is_full_body` (a third dump invariant the plan currently never
  dispositions).
- **FINDING — the payload change leaks artifact names into the "settled" flush-policy
  cluster.** `G-004`'s observation surface, `VP-T003-interval-flush`, and
  `VP-T001-writes-deferred` all name `index.txt` / "turn file", which `FK-002` removes.
  The *decisions* stand; the *observation surfaces* must be repaired or those VPs assert
  against a file that will not exist.
- **FINDING — `EI:.../turn_zero_is_full_body` has no disposition.** The plan names two
  overrides (`output_layout`, `sequential_diff_chain`); a third dump invariant is in the
  contradiction surface and is currently undecided.
- **FINDING — `FK-A` may no longer clear the escalation floor.** The same arithmetic
  repair that fixes its `If-wrong` also removes the cost of a wrong answer. Decide must
  re-test it before authoring (analysis in §Successor Question Scope).
- **FINDING — `FK-B`'s basis is already named in a recorded user choice** ("data
  volume-based"). The residual question is narrower than 005 recorded.
- RESOLVED-FROM-RECORD (not a user question): append-only (not rewritten per flush),
  and `index.txt` cannot survive as a separate file.
- NEW: the collapsed format's filename is a `Mode: NEW` anchor (`RB-003`); the read path
  is deliberately **not** an anchor row.
- RE-CHECKED: coverage headroom unchanged; the rewrite is coverage-neutral to favourable.

## Anchor Map Verification (re-verified against source)

Every affected row was re-read at its bound symbol in `buster_ripper/dump.py`. **All
bindings still resolve; no wrong-binding correction this round.** What changed is the
*contract* three rows carry, not their referents.

| Term | Binding | Re-verified against | Result |
|---|---|---|---|
| dump entry point | `dump.py:dump_turn` | `dump.py:61-109` | CONFIRMED — still the single append point; sync; called only from `routes/messages.py:57`. **Notes updated**: it now *builds a record* instead of writing files |
| per-session dump state | `dump.py:SessionState` | `dump.py:16-24` | CONFIRMED — `turn`/`prev_text`/`session_dir`; module-global `_sessions` at `:24`. **Notes updated**: gains the pending-record field |
| dump write buffer | `dump.py:SessionState` (extend) | `dump.py:16-24` | CONFIRMED — unchanged stance. **Notes updated**: the pending payload is now a list of *format records*, not a list of `(path, text)` file writes |
| bulk flush | `dump.py` (new symbol) | `grep -rn "write_text\|\.open("` → three write sites, all `dump.py:82,95,105` | CONFIRMED NEW — **RB-003** (re-run of `RB-001`). **Notes rewritten**: it no longer writes `index.txt` or per-turn files; it appends the pending records as ONE write |
| dump dir config | `config.py:DUMP_DIR` | `config.py:16` | CONFIRMED `reuse` — unchanged |
| shutdown flush seam | `server.py:on_cleanup` | `server.py:60-72` | CONFIRMED — `:60` cancel KV task, `:72` close session; registered `:79`. Unaffected by the payload |
| signal handling | aiohttp web_runner | `web_runner.py:283-284` → `web.py:511` → `web.py:425` → `server.py:60` | CONFIRMED `reuse` — no custom handler |
| dump caller | `routes/messages.py:handle_messages` | `routes/messages.py:55-59` | CONFIRMED `reuse` |
| flush timer | `dump.py` (new symbol) | `compaction.py:28-57`, `server.py:25-34` | CONFIRMED `NEW` / `RB-002` — unaffected by the payload |
| flush trigger parameters | `config.py:DUMP_FLUSH_INTERVAL`, `DUMP_FLUSH_TURNS` | `config.py:44` | CONFIRMED `NEW` / `RB-002` — unaffected |
| ~~dump index~~ → **dump summary line** | `buster_ripper/dump.py` (new symbol: the per-turn summary-string builder) | `dump.py:84`, `dump.py:98-101`, `dump.py:104-106` | **REBOUND.** The row's old binding was "the `index.txt` write", and that symbol no longer exists. The summary *line* survives as a record inside the artifact; its construction (the `summary = f"..."` expressions at `:84` and `:98`) is what the row now names. Mode stays `extend` |

### Rows added this pass

| Term | Binding (file:symbol) | Mode | Invariants | Notes |
|---|---|---|---|---|
| collapsed dump artifact | buster_ripper/dump.py (new symbol: artifact filename constant + record serializer) | NEW | EI:.../output_layout (override target), EI:.../turn_zero_is_full_body, EI:.../sequential_diff_chain | **RB-003**; one file per session under `DUMP_DIR/<sid>/`, appended once per non-empty flush. Turn 0's full JSON body is embedded; arrival order preserved. **Append-only** — a rewrite-per-flush variant is excluded by the presented option text and by the write-volume arithmetic (§Successor Question Scope) |

`bulk flush` Notes (rewritten): writes the pending format records as **one** append to
the collapsed artifact; no-op when `DUMP_DIR is None` or nothing is pending; must not
raise; **must contain no `await`**; called from three triggers (volume, timer,
shutdown). **No longer writes per-turn files or `index.txt`.**

`dump entry point` Notes (rewritten): becomes buffer-append; **builds** the turn's
record at append time (summary line; turn 0 → full JSON body; turn ≥ 1 → unified diff of
`_render_body` output against the previous turn's), preserving the settled
`turn`/`prev_text`-at-append-time sequencing; keeps signature `dump_turn(body) -> None`
and the swallow-all caller contract; also hosts the append-time volume check (`G-005`).

### Deliberately NOT added as an anchor row: the read path

The prompt requires `RB-001` to be re-run for "the new write **and read** paths". The
re-run's finding (§Reuse Evidence `RB-003`) is that **there is no code read path to
bind**: nothing in `buster_ripper/` or `tests/` reads `index.txt`, `turn_000.json`, or a
`turn_NNN.diff`. The read path today is the human workflow documented at
`README.md:153-162` (`cat index.txt`, then open the named diff). Inventing a `Mode: NEW`
"reader" row would authorize code the plan has not decided to build. If decide's
successor FK selects a viewer/reconstructor, that becomes a new `Mode: NEW` row with its
own `RB-*` row — the anchor is created by that decision, not pre-authorized here.

## Reuse Evidence — RB-003

| Evidence | NEW anchor | Name search | Role/peer search | Native capability | Conclusion |
|---|---|---|---|---|---|
| RB-003 (re-run of RB-001 for the collapsed format, write **and** read paths) | collapsed dump artifact (`dump.py` filename constant + record serializer) | Write side: `grep -rniE "buffer\|batch\|pending\|flush\|queue\|drain\|accumulat\|writer\|defer\|debounce\|artifact\|jsonl\|ndjson\|logfile\|session\.log"` over `buster_ripper/` + `tests/` → the only pre-existing hit remains `X-Accel-Buffering` (`routes/messages.py:150`) plus prose. Collapsed-format-specific terms (`artifact`, `jsonl`, `ndjson`, `logfile`) return **zero** matches anywhere in the package or tests. Read side: `grep -rn "index\.txt\|turn_000\|turn_\|session_dir\|DUMP_DIR"` over `*.py`/`*.sh`/`*.toml`/`*.md` (excluding `.sessions/` and `.venv/`) → writers only (`dump.py:81,94,104`; `main.py:89-90`; `config.py:16`), prose only (`README.md:146-162`, `buster_ripper_legacy.py:41-45`), and **no reader**. No parser, no viewer, no `--dump-*` read subcommand (`main.py:12-90` has no read-side flag) | No registries or Protocol/ABC contracts exist in this repository. Sibling implementations inspected for the write side: the same three call paths as `RB-001` (`routes/messages.py:55-59` → `dump.py:61`; `server.py:44-72`; `compaction.py:28-57`). **Nearest structural peer for the collapsed write is `logging.FileHandler`-style append-to-file**: considered and rejected — `logging` owns its own formatting, timestamps and handler lifecycle, is not session-keyed, and the repo's logger is configured once at `main.py` with no per-session channel. The *mechanism* it shares (`open(path, "a")`) is already the repo's own (`dump.py:105`), so what is new is the artifact symbol, not the append technique. Read side: nothing to reuse, nothing to extend — the role does not exist in code. `difflib.unified_diff` (already imported at `dump.py:5`, used at `:89`) is the reusable primitive **if** decide chooses a reconstruction tool; it is not a reader | Verified: `Path.open("a")` append mode is the native mechanism and is what `dump.py:105` already uses for the index — reusable as-is, but it does not supply a filename, a record shape, or ordering semantics. `logging.FileHandler`/`logging.handlers` verified present in stdlib and rejected (above). No compression/rotation facility is warranted (`US-001` waives durability; no size cap is promised). Dependencies are `aiohttp` + `click` only (`pyproject.toml:5-8`) — no `jsonlines`/`ndjson` package is installed | No existing symbol owns the collapsed-artifact role; a new filename constant + record serializer is required, and it is genuinely `Mode: NEW`. The **read** path has no candidate at all because it has no code existence — its replacement is documentation (`README.md:146-162`), which is why it is not an anchor row |

## Coverage Headroom Re-check

Re-verified against the live `.coverage` this pass:

```
Name                          Stmts   Miss  Cover
buster_ripper/dump.py            71     54    24%
TOTAL                           506     98    81%     (gate: --cov-fail-under=80, pyproject.toml:19)
```

The formula is unchanged and still binding:

```
(408 + K − U) / (506 + K) ≥ 0.80   →   U ≤ 3.2 + 0.2·K
```

**The payload rewrite does not tighten the gate, and probably loosens it.** `dump.py` is
coverage-visible (`pyproject.toml:26-30` omits `routes/*`, `main.py`, `server.py` only).
The three per-turn write sites at `:82`, `:95`, `:105` are currently *uncovered* — they
are inside the 54 missed statements. Replacing them with a record builder plus one flush
that the new `tests/test_dump.py` exercises moves statements from the missed column to
the covered column. `K` here is **net** new statements; the `server.py` spawner and
`on_cleanup` wiring stay coverage-free (omitted at `pyproject.toml:29`). Iteration 005's
conclusion stands unchanged: the binding fact is **placement**, and the flush body and
the record builder both live in `dump.py` and must be tested.

## Goal Shape Analysis

Revised from iteration 005. `G-001`'s criterion, "Met when" and failure witnesses are
**rebuilt** for the collapsed payload; `G-006` is new. `G-002`/`G-003`/`G-005` are
unchanged in substance (only `G-004`'s observation surface is repaired).

| Goal candidate | Source wording | Observation surface | Met when | Failure witness | Boundary / non-goals | Alternate readings | Authority |
|---|---|---|---|---|---|---|---|
| G-001 — bulk write | "do collection into memory and dump to the disk in bulk reducing number of writes" | file-write operations issued against `DUMP_DIR` for a session of N turns, and the on-disk state of `DUMP_DIR/<sid>/` | For a session of N intercepted turns flushed F times, total file-write operations `= F`, versus `2N` today; reduction `= 1 − F/(2N)`, i.e. `50% ≤ reduction < 100%` for every achievable `F` (an empty flush writes nothing, so `F ≤ N + 1`) | **A**: the flush writes one artifact per pending turn, so a flush of K turns costs K writes and the session total is `N + F` — buffering changed *when* writes happen, never *how many*. **B**: the flush is one write, but it **rewrites** the whole artifact each time, so bytes written grow with the session (`O(N²)`) — the write *count* fell while the SSD write *volume* rose. Both leave the process healthy and the artifact correct | `/v1/messages` responses unchanged; durability still waived (`US-001`); the reduction is a count, and witness B is why the count alone is not the goal | (a) "reducing writes" = write-operation count → met by an append-only collapsed artifact; (b) "reducing writes" = bytes written to the SSD → met *only* by an append-only artifact (witness B); (c) "reducing writes" is the user's proxy for removing blocking IO from the request path | **user** — `FK-002` settles the format; the arithmetic is now code-grounded (`dump.py:82,95,104-106`) and no reading of "reducing writes" is left unmet. The interval value no longer changes this classification |
| G-002 — flush on Ctrl-C / SIGTERM | "properly handle ctrl c seek term … do a flush of that dump on a disk before terminating the process" | `DUMP_DIR/<sid>/` contents after SIGINT/SIGTERM | After SIGINT or SIGTERM the process exits and every turn buffered before the signal is on disk, in order, in the collapsed format | Process exits cleanly (no traceback, exit 0) but `DUMP_DIR/<sid>/` is missing the turns buffered before the signal — the hook ran and wrote nothing | SIGKILL / hard power loss out of scope; no custom signal handler | none material | code-grounded (`web_runner.py:283-284` → `web.py:511` → `web.py:425` → `server.py:60`) |
| G-003 — proxy unchanged | "make this repository … to do collection into memory" (implicit) | HTTP responses on `/v1/messages` + the existing suite | Existing suite green; a raising dump/flush still yields the upstream response | A request returns non-200, or its latency is dominated by dump/flush work, because the buffer moved IO onto the request path | No new endpoints; no normalization changes; `dump_turn` stays failure-isolated | none material | code-grounded (`routes/messages.py:55-59`) |
| G-004 — periodic flush (bounded staleness) | "We need to have both time-based and data volume-based flash frequencies" | **repaired**: the `DUMP_DIR/<sid>/` collapsed artifact (existence, contents, mtime) **while the process runs** — the former surface named `index.txt`, which `FK-002` removes | With the interval configured to `I`, a turn appended at `t` appears in the artifact no later than `t + I + ε`, with the process still running | The timer task exists, is spawned, and runs, but the artifact never gains the turn for a live session — the loop's flush is a no-op against the wrong state, or an exception kills the task on the first tick and nothing logs it. Process healthy, staleness unbounded | No durability guarantee (`US-001`); no fsync; not a request-latency bound. Interval **value** is an open decision (`FK-A`, escalation-floor re-test pending) | unchanged from 005 | code-grounded on the mechanism; interval value OPEN → decide |
| G-005 — volume-triggered flush | "We need to have both time-based and data volume-based flash frequencies" | `DUMP_DIR/<sid>/` contents after appending turns with the timer effectively disabled | Reaching the configured volume threshold puts the pending turns on disk with no shutdown and no timer | The threshold and its comparison exist but turns accumulate past it with nothing on disk until shutdown — the check reads the wrong field, is evaluated before the append, or is off by one | No memory *cap* is promised unless the user picks a byte basis. Basis and value are policy (`FK-B`) | unchanged from 005 | **OPEN** → decide |
| **G-006 — the dump stays readable (function, not layout)** | `README.md:134-135` ("useful for identifying remaining cache busters after normalization") + `README.md:146-162` (the documented reading workflow) + `FK-002`'s option text ("diffs become a read-time concern") | a reader inspecting only `DUMP_DIR/<sid>/` after a session | Given a session in which one known message changed at turn K, a reader inspecting only `DUMP_DIR/<sid>/` can locate turn K and identify what changed against turn K−1, without additional tooling | The artifact is written once per flush and is well-formed, but a reader cannot find the changed turn or see what changed — the tool's documented purpose (spotting cache busters) is gone while every implementation task completed | Not a requirement that the *layout* survive (`FK-002` displaced it); not a requirement that a viewer tool be built; not a formatting requirement | (a) readability is satisfied by embedding the per-turn diffs (change-focused, closest to today); (b) satisfied by embedding the per-turn rendered bodies (self-contained, change visible only by comparing); (c) satisfied only by shipping a viewer/reconstruction tool; (d) not a requirement at all — the artifact is a machine record | **OPEN** → decide (successor FK). `README.md:146-162` and `FK-002`'s option text ground the *purpose*; no repository evidence excludes readings (a)–(d) |
| Constraint — durability is a non-goal | "we don't care about making this super robust and if for example shutdown will kill the in memory in flight messages it's okay" | n/a — normative constraint on design | The implementation ships no durability machinery (no fsync, no journal, no crash recovery, no write-ahead log) | The implementation adds fsync/journaling/recovery to satisfy a durability requirement the user explicitly waived | SIGKILL loss acceptable; do not build for it | none | code-grounded (`US-001`) |
| Constraint — write-reduction floor | `US-001` "reducing number of writes" + `FK-002`'s collapsed format | measured write operations per session | **rebuilt**: reduction `= 1 − F/(2N)`; the floor is **50%** (at `F = N`), approached 100% as `F → 1`. The reduction cannot reach zero while flushes are non-empty | The plan states a fixed reduction percentage the interval can drive to zero — no longer possible; or states "~100%" as the guaranteed outcome — wrong, the guarantee is 50% | The former `½(1 − F/N)` ceiling belonged to the displaced documented layout and must not be cited | none | code-grounded (`dump.py:82,95,104-106`) |
| Constraint — coverage gate | plan constraint | `--cov-fail-under=80` | Uncovered new statements `≤ 3.2 + 0.2·K`; all new `dump.py` lines covered | New `dump.py` statements ship uncovered and the suite fails | Not a user question | none | code-grounded (`pyproject.toml:19,26-30`) |

### Goal-shape findings (for decide)

1. **The interval no longer arbitrates `G-001`.** This is the headline repair. Under the
   collapsed payload every achievable `F` leaves at least a 50% reduction, so a short
   interval cannot erase the user's other stated goal. `FK-A` must not be presented with
   the old figure, and its escalation basis must be re-tested (§Successor Question
   Scope).
2. **Witness B moved from "too many flushes" to "rewritten instead of appended".** The
   old `G-001` witness B ("the timer fires so often that `F → N`, so the reduction is
   ~0") is now **false** — at `F = N` the reduction is 50%. The genuine remaining way to
   lose the goal is a flush that rewrites the artifact, which lowers the write count
   while raising the write volume. This is why append-only is load-bearing rather than a
   stylistic detail, and it is the reason `FK-002`'s stated rationale ("saving SSD") is
   not fully served by a count-only criterion.
3. **`G-006` is the goal the successor FK actually arbitrates.** `FK-002` settled the
   *requirement* (one write per flush) and displaced the *layout*. What is unresolved is
   whether the artifact carries the change information at all, and in what form. Without
   `G-006`, "collapse" can be read as "write a blob", which satisfies `G-001` and
   destroys the tool.
4. **`G-004`'s observation surface was broken by `FK-002`.** The decision stands; the
   surface named a file that will not exist. Same for `VP-T003-interval-flush` and the
   "turn file" wording in `VP-T001-writes-deferred`. These are repairs, not re-opens.

## Invariant Reconciliation Prep

**No invariant registry exists in this repository** (`.invariants/` absent; re-confirmed
this pass). Collect's six `EI:` entries remain session-local drafts. This pass narrows to
the contradiction surface for the payload cluster.

### Existing invariants in contradiction scope

- `EI:buster_ripper.dump/dump_turn/output_layout` — session dir holds `turn_000.json`,
  `turn_{N:03d}.diff`, `index.txt` with one summary line per turn | Status: **intended**,
  now **displaced** → **override target**
- `EI:buster_ripper.dump/dump_turn/turn_zero_is_full_body` — turn 0 is always written as
  a full JSON body, never as a diff | Status: **intended**, **at risk and currently
  undispositioned** → the load-bearing property (a full baseline exists) survives iff the
  artifact embeds turn 0's body; the *mechanism* (a separate `turn_000.json` file) does
  not → **refine, with the statement restated** (see Finding below)
- `EI:buster_ripper.dump/dump_turn/sequential_diff_chain` — turn N diffs turn N-1's
  rendered text; `prev_text` advances only when a turn is written | Status: **intended**,
  **partially displaced** → the property (per-turn diff against the previous turn's
  rendered text, in arrival order) **survives** because `T001` keeps `turn`/`prev_text`
  advancing at append time; the clause "advances only when a turn is written" is what the
  plan changes → **refine** (restate as "advances at append time"), not a wholesale
  override

### ⚠ Finding: the plan names two overrides; three invariants are in the surface

`plan-summary.md`, `handoff-context.md` and `iterations/004-interview.md` all record that
`EI:.../output_layout` and `EI:.../sequential_diff_chain` need an override entry.
`EI:.../turn_zero_is_full_body` is bound to the same Anchor Map rows (`dump entry point`,
`bulk flush`) and is displaced by the same change, but appears in **no** disposition.
This is the classic dropped-clause shape: the successor text reads complete on its own.
Decide must give it an explicit disposition before PLAN-SELECTED. My reading, offered as
input rather than as a ruling:

- If the collapsed artifact embeds turn 0's full JSON body (which `VP-T002-turn0-available`
  requires), the invariant is **refined**: restate as "turn 0's full body is present in
  the session artifact, never only a diff", keeping the load-bearing clause ("a missing
  full baseline makes the whole session directory unreadable").
- If the artifact stores only diffs, the invariant is **overridden** and the baseline is
  recoverable only as the removed side of turn 1's diff — an implicit, fragile encoding
  that `G-006`'s reading (a) does not require but reading (c) would.

### Likely planned invariant themes

- **One flush is one write** — `FK-002`'s requirement, and the new hard requirement on the
  flush. Load-bearing across all three triggers.
- **The artifact is append-only** — the invariant that keeps write *volume* (not just
  count) falling. This is the one that carries the user's stated SSD rationale.
- **Buffered append preserves turn order** — unchanged; sequencing stays at append time,
  now load-bearing across three flush callers.
- **The flush contains no `await`** — unchanged; still the reason three triggers are safe
  on asyncio's single thread.
- **Flush cannot raise out of `on_cleanup`** — unchanged.
- **Flush is idempotent per pending batch** — unchanged; also why an idle timer tick is a
  no-op and why `F ≤ N + 1`.
- **The timer loop survives a raising flush** — unchanged (`RB-002`, `G-004`).
- **NEW: the artifact is self-sufficient for the documented reading purpose** — the
  `G-006` theme, pending the successor FK.

### Suspected contradiction pairs

- `EI:.../output_layout` ↔ *one-flush-one-write theme* — direct displacement. **Override.**
- `EI:.../turn_zero_is_full_body` ↔ *collapsed artifact theme* — contradiction **only if**
  the artifact drops the baseline. Pending the successor FK.
- `EI:.../sequential_diff_chain` ↔ *buffered append theme* — unchanged from 002/005 and
  still the sharpest one: a flush must not advance `turn`/`prev_text`, or the diff chain
  resets whenever the interval fires. **Refine.**
- `EI:.../dump_failure_isolated` ↔ *collapsed artifact theme* — not a conflict, but the
  new format moves more work into the append path (record building), so the swallow-all
  contract at `routes/messages.py:55-59` now covers more code. Unchanged disposition.
- `G-006` ↔ *G-001* — a goal-level tension, not an invariant one: the cheapest way to
  reach one write per flush is to write the least, and the cheapest way to stay readable
  is to write the most. The successor FK arbitrates.

### Out-of-scope invariants intentionally dropped

- None. All six `EI:` entries still intersect the planned change.

## Approved Override Register (prep — decide finalizes)

`plan-summary.md` has no `## Approved Override Register` section yet; `FK-002`'s
resolution requires one. Organize does not write the approved entry (it needs the
successor FK's new statement and the user's approval of the *specific* replacement), so
this is the prepared skeleton with the decisions that are already settled:

| OI | Old invariant | Decision | New invariant (pending) | User approved |
|---|---|---|---|---|
| `OI:buster_ripper.dump/dump_turn/output_layout` | session dir holds `turn_000.json` / `turn_{N:03d}.diff` / `index.txt` | **override** | the session dir holds one collapsed per-session artifact, appended once per non-empty flush | **direction yes** (`FK-002`); the replacement statement needs the successor FK |
| `OI:buster_ripper.dump/dump_turn/sequential_diff_chain` | …`prev_text` advances only when a turn is written | **refine** (recommended over override) | …`prev_text` advances at append time; each turn's record diffs the previous turn's rendered text, in arrival order | pending decide's disposition |
| `OI:buster_ripper.dump/dump_turn/turn_zero_is_full_body` | turn 0 is written as a full JSON body | **refine** (recommended) or override | turn 0's full body is present in the session artifact | **not yet raised** — see Finding |

The `OI:` entries belong in `plan-summary.md` under `## Approved Override Register` with
`User Approved: yes` only after the successor FK resolves. The template is in
`interview-interactive` → "Add this section to `plan-summary.md`".

## Dropped-Clause Sweep

Triggered — this pass **rebuilds** `G-001`'s criterion and witnesses,
`VP-T002-write-count-reduced`'s target, the write-reduction constraint, and the two
invalidated VPs (wholesale replacement of definitions). One hop: the direct predecessors
are the corresponding rows in `plan-summary.md` as of iteration 005, plus `FK-002`'s
`Impact: Plan` clause list in `decisions.md`. No invariant lineage is cited by any of
them (`Superseded-by`/`Refined-by` do not exist — the registry is absent), so there is no
second hop.

| Clause (predecessor) | Disposition | Why / where it now lives |
|---|---|---|
| G-001 outcome: "held in memory and written in bulk instead of once per request" | carried | `plan-summary.md` §Goal Definition Matrix, `G-001` |
| G-001 observation surface: "write operations against `DUMP_DIR`; on-disk state of `DUMP_DIR/<sid>/`" | carried | same row |
| G-001 met-when: "total file-write operations `= N + F`, and `F < N`" | **superseded** | rebuilt as `= F` versus `2N`; reduction `1 − F/(2N)`, floor 50% |
| G-001 failure witness A: "`dump_turn` still calls `Path.write_text` and re-opens `index.txt` on every turn (`F = N`, so `N + F = 2N`)" | **superseded** | rebuilt as "the flush writes one artifact per pending turn" — the old witness named the *displaced* write sites, so it could no longer fire |
| G-001 failure witness B: "the timer fires so often that `F → N`, so the measured reduction is ~0" | **superseded** | **now false** — at `F = N` the reduction is 50%. Replaced by the rewrite-instead-of-append witness (write *volume*), which is the only remaining way to lose the goal |
| G-001 boundary: "Layout unchanged; no new persistence format" | **superseded** | replaced by "`/v1/messages` responses unchanged; durability still waived" — `FK-002` explicitly introduces a new persistence format |
| G-001 alternate readings (a) "~50% reduction" | **superseded** | the ~50% figure is now the *floor*, not the ceiling; reading (a) as a ceiling is the displaced claim |
| G-001 alternate readings (b) "collapse → ~100%" | carried | now the selected reading; lives in `FK-002` |
| G-001 alternate readings (c) "real goal is IO off the request path" | carried | unchanged |
| G-001 Authority: "**OPEN** on the interval value, which sets `F` → decide" | **superseded** | now `user` — the interval no longer changes whether `G-001` is met |
| `VP-T002-write-count-reduced` observable: "`= N + F`, versus `2N`; reduction `= N − F`" | **superseded** | rebuilt as `= F`, versus `2N`, reduction `1 − F/(2N)` |
| `VP-T002-write-count-reduced` scope reason / evidence (`F` deterministic, inject the parameters, disable the timer) | carried | unchanged |
| `VP-T002-layout-preserved` (whole row) | **superseded** | replaced by `VP-T002-collapsed-artifact` |
| `VP-T002-diff-chain` (whole row) | **superseded** | replaced by `VP-T002-change-readable` |
| Constraint "write-reduction ceiling … reduction `= ½(1 − F/N)`; ~50% only as `F → 1`" | **superseded** | rebuilt as the write-reduction *floor*: `1 − F/(2N)`, ≥ 50% |
| Constraint "(organize replan, now superseded by `FK-002` — kept only so the dropped arithmetic is visible; do not cite)" | **superseded** | the placeholder's purpose is served: the displaced figure is now visibly replaced, so the clause is deleted rather than kept |
| `MS-001` complete-when: "…in the documented layout and arrival order, and the write count is `N + F` with `F` controlled" | **superseded** | rebuilt as the collapsed format, write count `F` |
| `T002` contract: "write pending turn files in order, then `index.txt` once" | **superseded** | rebuilt as record-building + one-append flush |
| `T004` contract: tests for "layout, diff chain" | **superseded** | rebuilt around the collapsed artifact and the reading behavior |
| `RB-001` conclusion: "a new flush symbol in `dump.py` … writes pending turn files in order then `index.txt` once" | **superseded** | `RB-003` re-runs the evidence for the collapsed format and reaches a new conclusion |
| `B-001` / `B-002` dispositions | **superseded** | re-dispositioned to the replacement VPs; `B-007` added for turn-0 availability |
| Anchor Map `dump index` row (binding `index.txt` write) | **superseded** | rebound as `dump summary line`; the `index.txt` symbol no longer exists |
| `FK-002` Impact:Plan clauses 1–10 | carried | every clause is executed by this pass or is explicitly decide's; see §Scope Disposition of FK-002's Impact Clauses below |
| `FK-001` Impact:Plan clauses (trigger, `T003`, `MS-002`, coverage) | carried | untouched by this cluster |
| `FK-001` resolution: "both a time interval and a data-volume threshold, plus the shutdown flush" | carried | unchanged |

Nothing was found lost beyond the three items surfaced as Findings.

### Scope Disposition of `FK-002`'s Impact Clauses

| `FK-002` Impact:Plan clause | Disposition |
|---|---|
| §Options — Option 3 is the pick; "recommend preserve" framing retired | carried (already in `plan-summary.md`) |
| §Goals and Constraints — arithmetic becomes `2N → F` | **executed** — re-derived with the 50% floor |
| Goal Definition `G-001` (outcome, Met when, witnesses A/B) | **executed** — rebuilt; `G-006` added for the readability boundary |
| Anchor Map — `dump index` contract, `bulk flush` payload, new rows for the collapsed writer and its read path, each `Mode: NEW` with a complete `RB-*` | **executed with one deviation**: the collapsed writer row + `RB-003` are added; the **read path is deliberately not a row** (no code referent exists — see §Deliberately NOT added) |
| §Reuse Evidence — `RB-001` must be re-run | **executed** — `RB-003` |
| §Task List — `T002`/`T003` re-derived; a README rewrite becomes a task | **executed** — `T002`/`T004` re-derived, `T005` added for the README; `T003`'s flush reference updated |
| Validation Point Matrix — the two invalidated VPs need replacements | **executed** — two replacements plus one new VP |
| Behavior Disposition Matrix — `B-001`, `B-002` | **executed** |
| §Open Questions item 2 | **executed** — item 2b narrowed (see §Successor Question Scope) |
| `FK-A`'s `If-wrong` framing must be revised | **executed as a finding, stronger than requested** — the premise repair dissolves the cost of a wrong answer (§Successor Question Scope) |

## QA Planning Context

### Regression Test Candidates

| Behavior ID | Description | Risk | Tier | Priority |
|-------------|-------------|------|------|----------|
| B-001 | Dump output layout and index format | High | integration (tmp `DUMP_DIR`) | P0 — **re-dispositioned**: the layout is replaced, the readable-layout *function* is preserved |
| B-002 | Diff chain correctness across turns | High | integration | P0 — **re-dispositioned**: per-turn change information in arrival order |
| B-003 | Dump failure does not break the proxied request | High | integration | P0 |
| B-004 | `on_cleanup` shutdown behavior | Medium | unit (`tests/test_server_hooks.py`) | P1 |
| B-005 | Proxy response passthrough unaffected | High | existing suite | P0 |
| B-006 | Session directory isolation by `session_id` | High | integration | P1 |
| **B-007 (new)** | Turn 0's full request body is preserved and recoverable | High | integration | P0 — carries `EI:.../turn_zero_is_full_body`, which had no behavior handle |

### New Feature Test Areas

| Feature | Expected Behaviors | Tier | Dependencies |
|---------|-------------------|------|--------------|
| In-memory pending-record buffer | Turns append a record to per-session pending state; nothing appears on disk below both triggers | unit (`tests/test_dump.py`, new) | T001 |
| Collapsed bulk flush | One flush of K ≥ 2 pending turns performs exactly ONE file-write operation and the session directory gains the records in arrival order | unit | T001, T002 |
| Collapsed artifact readability | A known change at turn K is locatable and its content identifiable from the artifact alone | unit | T001, T002 |
| Turn-0 baseline | Turn 0's full body is present in the artifact | unit | T001, T002 |
| Volume-triggered flush | Reaching the configured volume threshold flushes with no timer and no shutdown | unit | T001, T002 |
| Time-triggered flush | With the interval injected small, a pending turn reaches the artifact with no shutdown | integration (running loop) + unit (loop body) | T001-T003 |
| Timer resilience | A raising flush inside the timer loop is logged and the loop continues; a cancelled timer stops cleanly | unit | T003 |
| Shutdown flush | SIGINT/SIGTERM → `on_cleanup` → cancel timer → flush → all buffered turns in the artifact | integration (subprocess + real signal) | T001-T003 |
| Flush failure isolation | A raising flush neither fails a request nor changes process exit | unit + integration | T001, T003 |

### Test Scope Recommendations

- **Supporting local proof**: append order; one write per flush; the artifact carries the
  records in arrival order; turn 0's body present; flush no-ops when `DUMP_DIR is None`
  or nothing is pending; flush clears pending without resetting `turn`/`prev_text`; flush
  idempotence across a volume-then-shutdown sequence; a raising flush inside the timer
  loop is swallowed and the loop continues; write-operation count for a known, injected `F`.
- **Phase/milestone**: `MS-001` (buffer + volume trigger + collapsed artifact observable in
  the filesystem), `MS-002` (time trigger observable on a live process; shutdown flush
  observable across a real signal).
- **Final integration**: existing suite green plus a real subprocess SIGTERM/SIGINT test —
  no in-process mock can prove the signal path.

### Validation Point Candidates

| Source behavior | Validation point | Observable behavior | Candidate observable after | Candidate validate at | QA author pass | Dependencies | Scope reason |
|---|---|---|---|---|---|---|---|
| B-001 | **VP-T002-collapsed-artifact** *(replaces `VP-T002-layout-preserved`)* | After a flush of turns 0..2, `DUMP_DIR/<sid>/` contains exactly one artifact file and **no** `turn_000.json`, `turn_NNN.diff`, or `index.txt`; the artifact carries the three turns' records in arrival order | T002 | MS-001 | QAP-MS001 | T001, T002 | the on-disk shape is only observable once a multi-turn flush exists; the "exactly one file" clause is `FK-002`'s requirement made visible |
| B-002 | **VP-T002-change-readable** *(replaces `VP-T002-diff-chain`)* | Given a session where turn 2 changes one known message, a reader inspecting only `DUMP_DIR/<sid>/` can locate turn 2 and identify the differing content against turn 1, without additional tooling | T002 | MS-001 | QAP-MS001 | T001, T002 | ordering and per-turn identity are only observable across a multi-turn flush. Deliberately format-agnostic so it survives the successor FK — it asserts the *behavior*, not the encoding |
| B-007 | **VP-T002-turn0-available** *(new)* | Turn 0's full normalized request body is present in the artifact and recoverable from it | T002 | MS-001 | QAP-MS001 | T001, T002 | carries `EI:.../turn_zero_is_full_body`, which had no VP. Fails for any encoding that stores only diffs |
| (new) write count | **VP-T002-write-count-reduced** *(re-derived)* | For N turns flushed F times, instrumented file-write operations `= F`, versus `2N` today; reduction `= 1 − F/(2N)` | T002 | MS-001 | QAP-MS001 | T001, T002 | structural/instrumented proof — must be paired with the black-box VPs so a mock cannot mirror the implementation. **`F` must be deterministic** — the test injects the parameters and disables the timer. Unit: one write-bearing file operation on a dump artifact (`write_text`, or `open(...,"a"/"w")`+`write`); directory creation is not counted (once per session) |
| B-001 / B-006 | **VP-T002-single-write-per-flush** | A flush carrying K ≥ 2 pending turns performs exactly one write-bearing file operation | T002 | MS-001 | QAP-MS001 | T001, T002 | the per-flush form of `FK-002`'s requirement. **Redundancy declared**: this is derivable from `VP-T002-write-count-reduced` plus `VP-T001-writes-deferred`; it is listed because it is the invariant in its sharpest form. Decide should keep whichever is the tighter contract, not both by default |
| B-003 | VP-T001-dump-failure-isolated | A raising dump/flush still yields the upstream response body | T001 | MS-001 | QAP-MS001 | T001 | observable as soon as the buffer path exists |
| B-006 | VP-T001-session-isolation | Two distinct system prompts produce two `DUMP_DIR/<sid>/` artifacts, each flushed independently | T001 | MS-001 | QAP-MS001 | T001 | keying is unchanged but now gates the buffer |
| (new) buffering | **VP-T001-writes-deferred** *(re-worded)* | After N turns below **both** thresholds and with the timer effectively disabled, no artifact containing those turns exists on disk | T001 | MS-001 | QAP-MS001 | T001 | the deferral is the black-box form of "buffering". Wording repaired: "turn file" named the displaced layout. Whether the session *directory* is created at append or at flush is a forward detail and either satisfies this |
| (new) volume trigger | VP-T002-volume-flush | With the interval injected large (timer cannot fire), reaching the configured volume threshold puts the pending turns in the artifact with no shutdown | T002 | MS-001 | QAP-MS001 | T001, T002 | needs the flush + the append-time threshold check; basis-agnostic wording, so it survives the `FK-B` answer |
| B-004 | VP-T003-shutdown-flush | After SIGINT or SIGTERM, buffered turns are in the artifact and the process exits without a traceback | T003 | MS-002 | QAP-MS002 | T001-T003 | requires process-level signal delivery; no in-process mock can prove it |
| (new) time trigger | **VP-T003-interval-flush** *(surface repaired)* | With the interval injected small and the volume threshold set unreachable, a single appended turn appears in the `DUMP_DIR/<sid>/` artifact while the process keeps running | T003 | MS-002 | QAP-MS002 | T001-T003 | the timer only exists once wired; the volume trigger is disabled so only the timer can satisfy it. **Repair, not a re-open**: the former text named `index.txt`, which `FK-002` removes |
| (new) timer resilience | VP-T003-timer-resilience | A flush that raises inside the timer loop is logged and the loop keeps flushing on later ticks; a cancelled timer stops without a traceback | T003 | MS-002 | QAP-MS002 | T003 | the `G-004` failure witness is a dead trigger in a healthy process; only a live loop can show it |
| B-005 | VP-T003-passthrough-regression | Existing `tests/test_routes_integration.py` suite green | T003 | MS-003 | — (existing suite) | T001-T003 | regression gate |

**Risk note.** `VP-T002-write-count-reduced` and `VP-T002-single-write-per-flush` are
white-box instrumented assertions and must stay paired with the black-box VPs. Their
determinism depends on the trigger parameters being injectable. The unit definition of a
"file-write operation" must be pinned in the test, or the assertion is not repeatable.

**Test Set Report input.** Net new test functions: not yet written (planning phase). No
invariant handle is claimed by two tests at the same tier. The one redundancy declared
above (`VP-T002-single-write-per-flush` vs `VP-T002-write-count-reduced`) is stated
explicitly rather than hidden; the "what already-failing test was checked first" question
is not yet answerable and is recorded as unanswered rather than as satisfied.

### Behavior Dispositions

| Behavior | Disposition | Validation point or reason |
|---|---|---|
| B-001 | mapped (re-dispositioned) | VP-T002-collapsed-artifact |
| B-002 | mapped (re-dispositioned) | VP-T002-change-readable |
| B-003 | mapped | VP-T001-dump-failure-isolated |
| B-004 | mapped | VP-T003-shutdown-flush |
| B-005 | mapped | VP-T003-passthrough-regression (existing suite) |
| B-006 | mapped | VP-T001-session-isolation |
| B-007 (new) | mapped | VP-T002-turn0-available |
| (new) buffering deferral | mapped | VP-T001-writes-deferred |
| (new) write-count reduction | mapped | VP-T002-write-count-reduced |
| (new) volume-triggered flush | mapped | VP-T002-volume-flush |
| (new) time-triggered flush | mapped | VP-T003-interval-flush |
| (new) timer resilience | mapped | VP-T003-timer-resilience |

### Candidate Milestones

| Milestone | Constituent tasks | Complete when | Validation points | QA author pass |
|---|---|---|---|---|
| MS-001 | T001, T002, T005 | A buffered session's turns reach disk only after a flush — a volume-triggered flush or a shutdown flush — **in the collapsed format**, in arrival order, with turn 0's full body present and the write count `F` with `F` controlled; the README documents the new artifact | VP-T001-*, VP-T002-* | QAP-MS001 |
| MS-002 | T003 | After a real SIGINT/SIGTERM, buffered turns are in the artifact and the process exits cleanly; **and** with the timer running, a pending turn reaches the artifact without shutdown | VP-T003-shutdown-flush, VP-T003-interval-flush, VP-T003-timer-resilience | QAP-MS002 |
| MS-003 | T004 | Existing suite green with the new `dump.py` lines covered and total coverage ≥ 80% | VP-T003-passthrough-regression | — (existing suite is the gate) |

`T005` (README) is placed in `MS-001` because the format is not delivered until its
documented contract matches — the README *is* the read path (`RB-003`). Decide may move
it; it is a one-task dependency, not a milestone of its own.

### Notes for Interview Agent

- **One successor FK is genuinely open** (`G-006` / the artifact's record content). See
  §Successor Question Scope.
- **`FK-A` and `FK-B` must be re-tested against the escalation floor before authoring** —
  the arithmetic repair removed `FK-A`'s cost of a wrong answer, and `FK-B`'s basis is
  already named in a recorded user choice.
- **QA setup (Topic 3) is unchanged and still unasked**: the harness must inject both
  trigger parameters at process start, and `.agents/skills/qa-project/SKILL.md` still does
  not exist. The payload change adds no new QA capability requirement.

## Successor Question Scope (for decide — decide authors the FKs)

The prompt asks whether `index.txt` survives, whether the per-turn diffs remain readable
and who reconstructs them, and append-only vs rewritten per flush. Applying the
scope-first rule to the record answers three of the four; one remains genuinely open.

| Candidate question | Verdict | Basis |
|---|---|---|
| **What does one flush's record contain — the per-turn unified diffs, the per-turn rendered bodies, or a machine record?** | **USER DECISION** (preference) → the successor FK | `FK-002` fixed the requirement (one write per flush) and not the content. Materially different products: diffs preserve today's change-focused reading experience but make any turn's full text recoverable only by replay; bodies are self-contained but larger and show no change; a machine record (`jsonl`) is parseable but not readable. The user's stated purpose (`README.md:134-135`, "identifying remaining cache busters") is served differently by each → `G-006`'s readings (a)/(b)/(c)/(d) |
| **Does `index.txt` survive as a separate file?** | **RESOLVED — NO**, from the record | A per-flush `index.txt` would be a **second** write per flush, contradicting `FK-002`'s requirement. A shutdown-only `index.txt` would not exist during the session, contradicting `FK-001`'s recorded basis for the time trigger ("`index.txt` is documented as the reading interface for a live session") — and would duplicate information the artifact already carries. Therefore the summary line is folded into the artifact. **Not a user question** |
| **Append-only or rewritten per flush?** | **RESOLVED — append-only**, from the record | `FK-002`'s presented option text, which the user chose, reads "**append** every pending turn record to a single per-session file: one write per flush regardless of turn count". Corroborated by arithmetic: a rewrite is a read plus a write of a file that grows with the session, so bytes written become `O(N²)` and the user's stated rationale (sparing the SSD) is defeated even though the write *count* stays `F`. That is `G-001` witness B. **Not a user question** |
| **Who reconstructs the per-turn diffs at read time?** | **RESOLVED — the reader, from the artifact; no tool required** *if* the artifact embeds the change information | `RB-003`'s read-path search found no code reader to reuse and no basis to authorize one. Building a viewer is a `Mode: NEW` addition the plan has not scoped. Reading the artifact with `cat`/`less`/`grep` is what the documented workflow does today (`README.md:153-162`). **If** decide picks an encoding where the change is not directly visible (bodies-only or machine record), a viewer becomes necessary — and that is a *consequence of the content question*, not a separate fork. Decide should surface it inside the successor FK's option text rather than as its own FK |
| **What is the artifact named / what extension?** | **Forward discretion, presented in the FK's option text** | User-visible (the README names it) but cheap and reversible, and it follows mechanically from the content choice. Naming it in the option text gives the user the information without spending a fork on it |
| **`FK-A` — interval value** | **RE-TEST REQUIRED against the escalation floor before authoring.** Analysis below suggests it no longer clears | See below |
| **`FK-B` — volume-threshold basis** | **NARROWED — the basis is already named in a recorded user choice.** Residual ambiguity is "bytes of what" | `FK-001`'s `Resolution` records the normative choice as "BOTH a time interval and a **data-volume** threshold". The user's own words are "data volume-based", and `FK-001`'s presented option text (which said "pending-turn threshold") is superseded by that answer. A literal reading of the user's words is **bytes**, so "turns vs bytes" is largely settled against the option text. What remains is bytes of the *rendered text* (what the diffs are computed from, already produced each turn at `dump.py:77`) vs bytes of the raw JSON body (only available for turn 0 today) — a materially smaller question. Decide should re-test the floor on that residual, and note that turns-basis additionally leaves buffer memory unbounded (a proxy-killing OOM risk, which is a `G-003` concern rather than a durability one) |

### `FK-A` escalation-floor re-test (decide's first act on this FK)

`FK-A`'s `Why-yours` cites "it is irreversible or expensive to reverse (data loss, public
API surface, security)". With the premise repaired, test the three clauses:

| Clause | Verdict |
|---|---|
| contradicts a recorded user decision or a protected invariant | **no** — any interval is consistent with `FK-001` |
| irreversible or expensive to reverse (data loss, public API surface, security) | **no** — it is one `config` global plus a CLI flag; the flag's *existence* is already resolved from code, so the *value* is a default |
| plan surface gives no basis at all **and** a wrong guess is expensive | **fails the second conjunct** — the plan surface supplies a direct in-repo peer (`config.KV_POLL_INTERVAL: int = 10  # seconds`, `config.py:44`), and the repaired arithmetic bounds the cost of any wrong guess: every achievable `F` leaves ≥ 50% of the reduction, and a short interval costs only extra small appends |

**Recommendation to decide: do not author `FK-A` as a user question.** Under the Fork
rules, a `Why-yours` matching no escalation-floor clause means DO NOT ASK — record a
Forward discretion footnote with a recommended reading instead. The recommended reading:
default the interval to `KV_POLL_INTERVAL`'s value (10 s), matching the in-repo peer, and
expose it as a CLI flag so the QA harness can inject it. That keeps the injectability
requirement (which is a QA capability, not a user choice) intact.

**Caveat, stated explicitly:** this is organize's reading of the floor, and the earlier
rounds recorded `FK-A` as a genuine user decision. If decide disagrees — for instance by
arguing the interval is a user-visible operating default users will come to rely on, which
is a public-surface argument — decide owns that call and authors the FK. What must not
happen is authoring `FK-A` with the old `If-wrong` figure.

### The premise repair itself (decide must use this text, not the old figure)

> Under the collapsed payload a flush costs **one** write regardless of how many turns it
> carries. For N turns flushed F times, writes are `F` versus `2N` today, so the reduction
> is `1 − F/(2N)`. A short interval therefore does **not** erase the benefit: at `F = N` —
> a flush on every turn — the reduction is still **50%**, and it reaches zero only at
> `F = 2N`, which is unreachable because an empty flush writes nothing (`F ≤ N + 1`). The
> old figure, `½·(1 − F/N)` reaching zero at `F = N`, is the displaced documented-layout
> arithmetic. The interval is a staleness/IO-frequency choice, not a goal-survival choice.

## Initial Task Identification

`T001` and `T003`'s trigger half are settled. `T002` and `T004` are re-derived; `T005` is
new. The milestone skeleton is preserved (`T003` remains the sole constituent of `MS-002`).

| Task | Description | Depends on | Status |
|---|---|---|---|
| T001 | Per-session pending-record buffer on `SessionState`; appended in `dump_turn`; `turn`/`prev_text` still advance at append time | — | settled |
| T002 | **Re-derived.** In `dump.py`: `dump_turn` builds the turn's record instead of writing files — summary line; turn 0 embeds the full JSON body; turn ≥ 1 embeds the unified diff of `_render_body` output against the previous turn. A `flush_all` appends all pending records to the per-session collapsed artifact in **one** write, clears pending, no-ops when `DUMP_DIR is None` or nothing is pending, must not raise, contains no `await`. Called from three triggers. `dump.py`'s module docstring is updated | T001 | **re-derived — decide finalizes** |
| T003 | Wire the flush into `server.on_cleanup` wrapped so it cannot raise; add the volume-threshold check in the append path, the periodic timer coroutine (loop body in `dump.py`, spawner + `app`-state key + cancel branch in `server.py`), the injectable `config`/CLI parameters, and the timer's per-iteration raise guard | T002 | settled (trigger half); its flush reference now means the collapsed flush |
| T004 | **Re-derived.** `tests/test_dump.py`: buffer, one-write-per-flush, collapsed-artifact shape (exactly one file, no per-turn file set), arrival order, change readability, turn-0 availability, session isolation, failure isolation, volume trigger, timer loop + resilience; plus `on_cleanup` and real-signal cases; keep all new `dump.py` lines covered | T001-T003 | **re-derived — decide finalizes** |
| T005 | **New.** Rewrite `README.md:146-162`: the artifact's name and shape, and the reading workflow that replaces "Reading the index" (the `cat`/`grep`/`less` commands that spot a cache buster). Also update `dump.py`'s module docstring if T002 has not | T002 | new |

**Deliberately not tasks.** (a) The `Approved Override Register` entry is a
`plan-summary.md` artifact produced by PLAN-SELECTED, not a code task. (b)
`buster_ripper_legacy.py:41-45` documents the same displaced layout in its module
docstring; it is the tracked pre-refactor monolith referenced by nothing under
`buster_ripper/` and is **excluded from scope** — decide should record that exclusion
rather than leave it unmentioned, so no reader binds the old layout to live code.

**Verdict for the REPLAN-INCREMENT:** the payload/format cluster is now structured to
agent-ready depth. What remains is decide's authorship of the successor FK (the artifact's
record content), the escalation-floor re-tests on `FK-A`/`FK-B`, the `OI:` entries, and
the PLAN-SELECTED task-out. Final approval stays blocked until those land.

## Compaction Decision

- Prior same-phase files reviewed: `iterations/002-organize.md`, `iterations/005-organize.md`.
- Superseded: **none.**
- Not superseded: both.
  - `002-organize.md` covers the *initial* planning slice — Anchor Map verification, the
    two-dial option input, the original `G-001`–`G-003` rows, the `EI:` inventory, and the
    initial task identification. This run rebuilds one `G-001` row and rebinds one Anchor
    Map row; it does not replace the analysis that produced them.
  - `005-organize.md` covers the *flush-policy* slice — trigger parameters, timer
    mechanism, `RB-002`, `G-004`/`G-005`, the coverage formula. This run touches that
    slice only to repair two artifact names its observation surfaces carry (`G-004`'s
    surface, `VP-T003-interval-flush`'s wording); every decision in it stands.
- Compaction performed: **no**.
- If no: slice-based reason. Three organize reports now cover three disjoint slices of one
  planning cycle (`002` initial, `005` flush policy, `006` payload format). Compacting any
  of them would delete the record of the analysis that produced `FK-001`/`FK-002` and the
  flush-policy decisions, neither of which this run supersedes. The Dropped-Clause Sweep
  above carries the specific clauses this run rebuilds, which is the intended mechanism for
  the delta. **Note for a future full reroute**: the cycle now has three organize reports;
  a genuine full-scope reroute should compact all three at once, and the slice-based reason
  here does not apply to that case.

## Confusability Findings

**No new finding.** Two near-misses were examined and both were rejected under the
observed-confusion reliability rule (record a finding only when an agent actually misread
the artifact this session):

- `buster_ripper_legacy.py:41-45` — the tracked monolith's module docstring documents the
  same displaced output layout as live behavior. I read it during the read-path search and
  identified it as legacy on first read; it is self-disambiguating by filename and is
  referenced by nothing under `buster_ripper/`. **Recorded in `RB-003`'s name search as
  explicitly excluded scope**, which is where a future reader's search will land. No
  backlog entry.
- `README.md:143-144` — says the session ID is the hash of "system prompt text" while
  `EI:.../derived_after_normalize` requires it be computed *after* normalization
  (`routes/messages.py:31-35`). A reader could take the README literally. Not a misread
  this session, not in this cluster's scope, and the phrasing is defensible. Noted for
  decide as adjacent, not raised.

The existing backlog finding (duplicate `main` entry points, `server.py:99` vs
`main.py:62`) is unchanged and remains triaged as not endangering this session.

## Deferred Questions

Deferred mode (`Question Mode: deferred`; no active-mode phrase in the orchestrator
prompt). No `AskUserQuestion` used. The genuine user decision is routed to decide (the FK
owner) to author; organize does not present it.

- **What does one flush's record contain (diffs / rendered bodies / machine record)?**
  - Status: DEFERRED for /codify-review — routed to decide (successor FK owner)
  - Impact: determines whether the documented cache-buster workflow survives, whether a
    viewer tool becomes necessary, and the artifact's size. `G-006`'s four readings produce
    materially different products.
  - Agent analysis: **none — genuine preference.** No source settles it. `RB-003` found no
    code reader to constrain the answer; `README.md:134-135` and `:146-162` ground the
    purpose but not the encoding.

## Questions Asked

No questions - deferred mode (`Question Mode: deferred` in session metadata; no
active-mode phrase in the orchestrator prompt). Organize does not present FKs; the
successor question is structured above and routed to its owner.

## Validation Performed

- Re-read `plan-summary.md`, `handoff-context.md`, `decisions.md`,
  `iterations/001-collect.md`, `iterations/002-organize.md`, `iterations/004-interview.md`
  (the primary input), and `iterations/005-organize.md`.
- Read `buster_ripper/dump.py:1-109` in full — the three write sites (`:82`, `:95`,
  `:104-106`), `SessionState` (`:16-20`), `_sessions` (`:24`), `_render_body` (`:27-58`),
  the bare `assert` (`:63`), the `summary` f-strings (`:84`, `:98-101`), the diff
  construction (`:87-95`), and the append-time advancement of `prev_text`/`turn`
  (`:108-109`).
- Read `buster_ripper/routes/messages.py:1-70` — normalize-then-`session_id` ordering
  (`:31-35`), the swallow-all dump call (`:55-59`).
- Read `buster_ripper/server.py:55-80` — `on_cleanup` (`:60-72`), registration (`:79`).
- Read `buster_ripper/config.py` — `DUMP_DIR` (`:16`), `KV_POLL_INTERVAL` (`:44`).
- Read `README.md:134-162` — the dump-mode section, the documented output layout
  (`:146-151`), and the "Reading the index" workflow (`:153-162`).
- Read `pyproject.toml:1-35` — `--cov-fail-under=80` (`:19`); the coverage omissions
  (`:26-30`); deps `aiohttp` + `click` only (`:5-8`).
- `grep -rn "index\.txt\|turn_000\|turn_\|session_dir\|DUMP_DIR"` over
  `*.py`/`*.sh`/`*.toml`/`*.md` excluding `.sessions/` and `.venv/` — **established the
  central `RB-003` result: every hit is a writer or prose; there is no reader in code.**
- `grep -rniE "artifact\|jsonl\|ndjson\|logfile\|session\.log"` over `buster_ripper/` +
  `tests/` — zero matches.
- Read `buster_ripper_legacy.py:35-50` — the legacy docstring documenting the same
  displaced layout.
- Read `buster_ripper/session.py:93-98` — `session_id`.
- `.venv/bin/python -m coverage report` — re-ran: `dump.py` 71 stmts / 54 miss / 24%;
  TOTAL 506 / 98 / 81%.
- `grep -n "Approved Override Register"` over the skills tree — located the canonical
  section format in `interview-interactive` (the `OI:` block template).
- `grep -n "Escalation floor"` over `interview-interactive` — read the three clauses used
  in the `FK-A` re-test.

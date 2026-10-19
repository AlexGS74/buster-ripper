# Iteration 049-01 — Forward (T005)

## Orchestrator Prompt (verbatim)

> You are a subagent spawned by another session.
> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
> Task ID: T005
>
> Execute the forward-interactive skill for this session. Arguments: $ARGUMENTS

*(No task-specific arguments were expanded; the session files supplied the binding — T005's finalized task entry in [plan-summary.md](../plan-summary.md) §Task List and the `T005 Planned Task Context — claim-32` block in [handoff-context.md](../handoff-context.md).)*

## Increment Brief

- **Task**: T005 — extend the dump section of the README (addition-only) and make the collapsed formats selectable on the documented run path.
- **Goal**: document the collapsed artifact (name, shape, selection, per-format reading workflow, arrival-order caveat, old-layout note) beside the unchanged documented workflow, and wire `DUMP_FORMAT` through `serve_buster_ripper.sh`.
- **Anchors touched**: dump format selector `config.py:DUMP_FORMAT/DUMP_FORMATS` + `main.py:--dump-format` (Mode: NEW, RB-004 — landed in T002) → **reuse, unchanged**; README dump section (read-path documentation, deliberately not an Anchor Map row; plan-authorized **addition-only extension**); serve-script option channels (verified two-channel inventory, claim-32/`DC-claim-32-r1` — `DUMP_FORMAT` rides `EXTRA_FLAGS` beside the `DUMP_DIR` peer).
- **Will NOT create**: any new symbol, module, artifact, or read-side code; **no tool** (`FK-007` option 1 ships none).
- **Reuse evidence**: no `Mode: NEW` anchor is created by T005 → no `RB-*` replay owed. `Inventions: none`.
- **Pre-flight**: Interview Finalize approval present (historical 2026-09-17, round-18 "approve, begin", `US-015`); `qa-plan.md` exists; **T005 carries no VP** (plan §Behavior Disposition Matrix: structural, like the T006 help/README/script row); its milestone gate is `MS-001`/`QAP-MS001`, not run here.
- **Clustering**: none — T004 precedes T005 in task order and is pending (gated behind the MS-002 re-verdict per chronicle), so the no-gaps rule forbids pulling T006; T005 runs alone.

## Plan Adherence Record

Implemented exactly the T005 clauses:

1. **README documents the collapsed artifact's name and shape** — `DUMP_DIR/<session_id>/<format>.log`, one append-only file per session, records buffered and bulk-appended in a single write, arrival order, same summary lines as `index.txt`, and the flush triggers (volume `--dump-flush-bytes` default 4 MiB rendered text, `--dump-flush-interval` default 10 s, shutdown) — all read from the accepted T002/T003 source, not restated from the plan prose.
2. **Each shipped format and how it is selected** — the `--dump-format [documented|change-focused|self-contained]` selector with default `documented`, stating explicitly that the documented layout is itself one value of the selector; per-format record contents (change-focused: turn 0 body+rendering then per-turn diffs; self-contained: every turn's rendering, turn 0 also the body).
3. **Reading workflow per collapsed format** — per-format `grep`/`less` command pairs that spot a cache buster (summary-line `-` count, then the `--- turn NNN diff ---` / `--- turn NNN rendered ---` block).
4. **Run-path example** — `DUMP_FORMAT=change-focused DUMP_DIR=~/mllm/prompt-diffs bash serve_buster_ripper.sh` beside the existing no-format example (022 clause), plus the direct-CLI form of the same selector.
5. **`F-8` limitation** — turns are not one linear conversation (42 of 137 transitions in the real dump had fewer messages than the previous turn); reader must read in arrival order.
6. **Previous-layout note** — `turn_000.json` / `turn_NNN.diff` / `index.txt` in a pre-existing session dir are neither migrated nor removed.
7. **Documented block untouched** — `README.md`'s original dump section (former lines 134–162) is byte-identical; the diff is **+53 lines, 0 deletions**. The dropped 018/019 removal clause was **not** authored.
8. **Serve script** — `DUMP_FORMAT=${DUMP_FORMAT:-}` (empty default) and `[[ -n "${DUMP_FORMAT}" ]] && EXTRA_FLAGS="${EXTRA_FLAGS} --dump-format ${DUMP_FORMAT}"`, mirroring the `DUMP_DIR` peer's shape exactly per the 022 clause and claim-32; +3 lines (one usage-comment line, one default, one guard). The `SKIP_TLS_VERIFY` boolean shape is **not** used (that is T006's clause, not mine).
9. **No tool section** authored (FK-007 option 1).

## Assumption Finding (Planned Task Context)

- **Finding**: claim-32 — serve script's verified channel inventory. **Disposition**: absorbed (`DC-claim-32-r1`). **Assumption impact**: `scoped:S6`.
- **Implemented required behavior**: `DUMP_FORMAT` → `--dump-format` rides `EXTRA_FLAGS` beside the `DUMP_DIR` peer, empty-default/`-n` guard, one flag per env var, no positional forwarding (verified below). **Not marked resolved** — Yield owes the replay.
- **Done-when evidence**: see Verification.

## Verification

**claim-32 done-when (argv recording against a PATH shim)** — [evidence/t005-serve-wiring-readme.txt](../evidence/t005-serve-wiring-readme.txt):

| Run | Exec line observation |
|---|---|
| `DUMP_FORMAT` unset | no `--dump-format`; defaults + `DUMP_DIR` peer unchanged |
| `DUMP_FORMAT=` (empty) | no `--dump-format` |
| `DUMP_FORMAT=change-focused` | **exactly one** added flag: `--dump-format change-focused` |
| `DUMP_FORMAT=self-contained DUMP_DIR=/tmp/peer-check` | peer mapping untouched (`--dump-dir /tmp/peer-check` forwarded); flag beside it |

All runs: every argv token is a flag or its value — **no positionals forwarded**. `bash -n` passes.

**README workflow truthfulness** — the documented grep/less commands were executed against a real 3-turn `change-focused` artifact produced by the actual dump code (fresh tmp `DUMP_DIR`): the summary-line grep yields index-style lines only, the `--- turn NNN diff ---`/`--- turn NNN rendered ---` anchors exist verbatim, and a tool-rename cache buster surfaces as `+7/-2 lines` exactly as the section documents (evidence file §"README workflow validated").

**Full suite (checkpoint gate)** — `uv run pytest`: **250 passed**, 25 pre-existing warnings, coverage **93.87%** (threshold 80%). No production Python code changed; nothing else touched.

## Failure-Path Evidence

| Operation / condition | Expected outcome and authority | Handling / observation evidence | Disposition |
|---|---|---|---|
| serve script, `DUMP_FORMAT` unset/empty | no flag; CLI default `documented` (022 clause; keeps `VP-T002-default-encoding`'s "no `--dump-format` passed" clause true on the script path) | argv runs 1–2 | safe |
| serve script, `DUMP_FORMAT=<valid>` | exactly one `--dump-format <v>`, no positionals (claim-32 required behavior) | argv runs 3–4 | safe |
| serve script, `DUMP_FORMAT=bogus` | forwarded as `--dump-format bogus`; `click.Choice` rejects at parse time before the server starts (`PI:T002/dump_format/format_set_is_closed`) | real-CLI probe in evidence file | safe |
| serve script, value containing whitespace | unquoted `EXTRA_FLAGS` expansion would split the value (extra token → click rejects); identical exposure to the `DUMP_DIR` peer today | peer-consistent by design (mirroring the peer's shape is the plan requirement); not changed | safe (peer-consistent) |
| README, pre-existing session dir from old layout | old files neither migrated nor removed; `.log` written alongside (plan clause 3; Anchor Map "a session dir may hold both shapes") | documented; T002-accepted behavior | safe |
| README grep over `self-contained` over-matching rendered content | cosmetic risk only: the documented pattern requires `turn NNN␣␣HH:MM` shape; validated on a real artifact to match summary lines only | validation section in evidence file | safe |

Exclusions: no Python code paths added or changed (dump/flush failure isolation, thresholds, shutdown ordering are T001–T003's accepted surface, untouched).

## Quality gates

- **Backlog pickup**: none match touch-set — T005 touches `README.md` and `serve_buster_ripper.sh`; every open [quality-backlog.md](../quality-backlog.md) item targets Python modules or test files.
- **Capacity gate**: not applicable - no additive production surface — no Python file is touched; the shell script gains 3 lines mirroring an existing peer shape; `codify scan path` has no Python surface to scan (recorded in lieu of a scan).
- **LNCP**: no new functions authored; N/A (no Python touched).
- **Tests/coverage**: no persistent test authored — the plan records this surface as **structural with no VP** ("a test of it would prove the wrapper, not the format branch"); T004 owns the dump test suite. Verification is the argv/CLI/artifact evidence above; full suite stays green.

## Durable signal

- `DC-claim-32-r1`'s required behavior is implemented on the exact peer shape it specifies; the finding remains for Yield's replay (set/unset argv observation) — **do not mark resolved**.
- No override entries exercised; no decision changes; no invariant entries created or edited (triad rule held: code + docs only).

## Concerns

None. (Minor, non-blocking: the README's `grep -E '^turn [0-9]{3}  [0-9]{2}:[0-9]{2}'` pattern relies on the summary line's two-space separator — it matched a real artifact cleanly, and the workflow is guidance, not a parser.)

## Next

Yield: verify T005 against the task entry + claim-32 replay; then T004 remains gated behind the MS-002 re-verdict (chronicle), T006 pending.

DEMO-EVIDENCE: .sessions/codify/02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/evidence/INDEX.md — no pause

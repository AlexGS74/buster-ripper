# Iteration 052-01 — Forward — T011: Honor the serve script's documented `DUMP_DIR=""` disable form

## Orchestrator Prompt (verbatim)

> You are a subagent spawned by another session.
> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
> Task: T011

## Increment Brief

- **Task and goal (my words):** make `serve_buster_ripper.sh` actually honor the `DUMP_DIR=""` disable its own usage header promises — line 28's default must apply only when `DUMP_DIR` is *unset* — and pin the three states on the script's assembled argv with a hermetic witness in the default suite.
- **Anchors touched:** no Anchor Map row moves. The edit is inside `serve_buster_ripper.sh` line 28 (an existing referent the map never bound — T011 is post-planning adhoc remediation registered by the 2026-10-04 Registrar batch); the behavior it terminates in — `buster_ripper/config.py:DUMP_DIR` — is mapped as **reuse** and is untouched.
- **Will NOT create:** no QA gate test (`tests/qa/` is milestone-owned; QAP-MS004 is `forward-qa`'s), no new `buster_ripper/` symbol or module, no argv-assembly restructure (lines 32–38/46 = **T012**, the next wave), no README change, no new value protocol.
- **Reuse evidence:** no `Mode: NEW` production anchor consumed. The one new artifact is the witness module, which the task's own `Done when` authorizes ("one new hermetic test module … the test module's name is Forward's, within `tests/`"); named `tests/test_serve_script_argv.py` so T012's argv witness rides the same shim helper instead of duplicating it. Pre-flight: `grep -rln "serve_buster_ripper\|subprocess" tests/` → only `tests/qa/harness.py` spawns a process (claim-199's basis); no existing argv witness existed to reuse.
- **Inventions:** none beyond that authorized test module.
- **Capacity gate: not applicable — no additive production surface.** The only production change replaces one word on one existing line (46 lines before, 46 after); the test module is not a production surface and is Green at 137 lines.

## Pre-flight gates

- **Plan approval:** session-level Interview approval recorded (`US-015`, round 18 "approve, begin"; plan-summary §Open Questions "STATUS: CLOSED"). T011 is an adhoc task added later and inherits it; the 2026-10-04 Registrar batch records it `dispatchable now` with "no further user decision is owed for this task". **PASS.**
- **QA contract:** `qa-plan.md` exists; T011 declares **no `VP-*`/`TC-*`** handle (FK-010's Impact: "no milestone gate moved"), and the witness is a supporting local test, not a gate test. Nothing to record against a validation point. **PASS.**
- **Protected invariants:** read scope narrowed to the touched path (`serve_buster_ripper.sh`) + the `DUMP_DIR` slice of `buster_ripper/config.py`/`main.py`/`dump.py`. No `EI:`/`OI:` covers the serve-script env→argv slice (the task block's cross-reference reads **Neutral**; registry has no session `invariants.md` for it). `PI:T002/dump_format/*` and `PI:T006/upstream_tls/*` neither created nor displaced. No contradiction → no stop. **PASS.**
- **Planned Task Context consumed:** `handoff-context.md` §Planned Task Context → `T011 Planned Task Context` block (authority `FK-010` option 1; finding `seam-S6-disable-form`, disposition `absorbed`, ledger `DC-seam-S6-disable-form-r1`). Its required behavior, `Done when`, Yield replay and verification profile are treated as mandatory criteria and reproduced below.
- **Wave membership:** anchor T011 alone; lines 32–38/46 untouched, so T012's targets stay exactly as its `Done when` pins them.

## What was implemented

1. **`serve_buster_ripper.sh:28`** — `DUMP_DIR=${DUMP_DIR:-${HOME}/mllm/prompt-diffs}` → `DUMP_DIR=${DUMP_DIR-${HOME}/mllm/prompt-diffs}`. The colon dropped is the whole fix: `:-` substitutes the default for *unset or empty*, which made line 36's `[[ -n "${DUMP_DIR}" ]]` guard true for the documented disable form; `-` substitutes only when **unset**, so an explicitly empty value stays empty, the guard is false, `--dump-dir` is never appended, and the proxy runs at its off-mode. One-character change; no quoting added, no guard rewritten, header line 14 untouched, `DUMP_FORMAT` peer (line 29) untouched, lines 32–46 untouched.
   - Script fingerprint moves `f4076d1ceae359f2…` → `d5db2144a2452840e7a5284c524c3fbfeef60dc117a55d61fab57a5aeafc5c90` (46 lines both before and after). This is the event `assumptions/claims.md` run-24 anticipated: *"the re-verification T011's Done when names … fires when the line-28 edit lands and moves that fingerprint"*.
2. **`tests/test_serve_script_argv.py`** (new, 5 tests, hermetic, in the default suite) — the argv witness:
   - **Mechanism:** a PATH shim named `buster-ripper` (written into `tmp_path/bin`, `chmod 0o755`) records the argv the wrapper would have exec'd, NUL-separated, into `$_ARGV_OUT`; the wrapper is invoked as `bash serve_buster_ripper.sh`, so **no server starts, no port binds, no upstream is contacted**. `_serve_env` copies the inherited environment **minus every variable the wrapper reads** (`_CONSUMED`: `UPSTREAM HOST PORT STRIP_DATE VERBOSE SKIP_TLS_VERIFY DUMP_DIR DUMP_FORMAT PRESERVED_THINKING`), so an omitted override is genuinely unset and a developer's own shell cannot leak into an expected argv; `HOME` is pinned to `tmp_path/home`, a path that is never created, so the default case asserts without touching the real home directory.
   - **Three-state witness (`TestDumpDirDisableForm`)** — each asserts the *complete* assembled list, which pins "exactly one", "none", the always-on `--upstream/--host/--port` trio, the `--dump-format` peer, guard ordering, and "no positional forwarding" (an orphan or split word breaks equality) in one readable diff:
     | case | assembled argv witness |
     |---|---|
     | `DUMP_DIR` unset | trio + `--dump-dir <pinned-home>/mllm/prompt-diffs` + `--preserved-thinking` (today's default **preserved** — the risk named in the verification profile) |
     | `DUMP_DIR=""` | **no `--dump-dir` token at all**; `--preserved-thinking` still present |
     | `DUMP_DIR=/tmp/served-diffs` | trio + `--dump-dir /tmp/served-diffs` + `--preserved-thinking` |
     | `DUMP_DIR="" DUMP_FORMAT=change-focused` | the format peer still appends while the dump flag is suppressed (lines 36/37 independence) |
   - **Documentation-agreement witness (`TestDocumentationAgreement`)** — asserts the header's exact `DUMP_DIR="" … # disable prompt diffing` line is still present, so the suite holds both halves of the user's answer together: the promise stands (FK-010: the documentation is not retracted) and the behavior witness above is what makes it true. If someone later "fixes" the bug by deleting the doc line, this test fails.
   - `_read_argv` drops only the trailing NUL field, so a legitimately empty argv word survives — deliberate, because T012's witness must be able to see a whitespace-only value as its own word and must not inherit a helper that silently erases it.

## Verification against the task's `Done when`

| clause | state |
|---|---|
| line 28 carries the unset-only default | ✅ landed |
| hermetic witness in the default suite observes the three states on the assembled argv, `--dump-format` peer + always-on trio intact, no positional forwarding | ✅ 5 tests, `tests/test_serve_script_argv.py`, no server started |
| full hermetic suite passes with no regression | ⏳ Yield's independent run — per role boundary I ran the **affected scope**, not the full suite (see Tests); nothing under `buster_ripper/` changed, so no test can regress except by the script's own behavior |
| `seam-S6` script-pinned members re-verified fresh against the new bytes | ⏳ **owed, and now unblocked** — the landing is the trigger; see §Assumption state |
| fresh `seam-S6` recomposition lands | ⏳ **owed** — the fix fires that composition's own `Invalidates-On` ("line 28's default form changing in a way that alters guard reachability"); the seam must not be cited as current until it does |

Independent behavior probe (also in the evidence file, raw argv as recorded by the shim): unset → `--dump-dir …/mllm/prompt-diffs`; `DUMP_DIR=` → no `--dump-dir`; explicit path → that path; `DUMP_DIR= DUMP_FORMAT=change-focused` → `--dump-format change-focused` and no dump dir; `SKIP_TLS_VERIFY=1 DUMP_DIR=` → `--skip-tls-verify` still appends beside a disabled dump; `DUMP_DIR= DUMP_FORMAT= PRESERVED_THINKING=0` → rc=0 with exactly the trio, i.e. line 36's guard is now reachable-false and the previously dead false arm runs clean under `set -euo pipefail` with no `unbound variable` abort (claims 188/189's zero-word arm). Observed on `/bin/bash` 3.2.57 — the oldest interpreter on this machine — so the `-` form is not relying on newer-bash behavior.

## Backlog pickup

`Backlog pickup: none match touch-set`. Read `quality-backlog.md` (9 unchecked items) against the touch-set (`serve_buster_ripper.sh`, new `tests/test_serve_script_argv.py`): every item targets `plan-summary.md`, `buster_ripper/dump.py`, `buster_ripper/routes/*`, `tests/test_backend.py`, `tests/test_routes_integration.py`, `tests/test_tls_posture.py` or `pyproject.toml` — none is in this increment's files or module. Explicit zero; no item silently skipped.

## Capacity gate

- Additive production surfaces: **none** (`not applicable` — the only production edit replaces one word in place; nothing added, deleted, or restructured). Pre-edit raw sizes: `serve_buster_ripper.sh` 46 lines; post-edit 46 lines. New non-production file: 137 lines (Green).
- No ratchet to hold, no extraction owed, no characterization coverage required beyond the witness itself.

## L→N→C→P compliance (new functions)

| Function | L | N | C | P | Verdict |
|---|---|---|---|---|---|
| `_pinned_home` / `_argv_file` | 1 | 0 | 1 | 1 | green |
| `_install_shim` | 4 | 0 | 1 | 1 | green |
| `_serve_env` | 5 | 0 | 1 | 2 | green |
| `_read_argv` | 4 | 1 | 2 | 1 | green |
| `serve_argv` (fixture) | 3 + inner `run` 3 | 1 | 1 | 0 | green |
| test methods (5) | 2–3 | 0 | 1 | 1–2 | green |

Zero gray-zone metrics. `codify scan path tests/test_serve_script_argv.py --git --detailed` → **No refactoring issues found**; `codify scan path serve_buster_ripper.sh --git --detailed` → no Python functions (shell is outside the scanner's language). No `# ESSENTIAL COMPLEXITY` markers added.

## Tests and coverage

- **Affected scope (run ONCE before checkpoint):** `uv run pytest tests/test_serve_script_argv.py tests/test_main_cli.py tests/test_pytest_config.py -q --no-cov` → **20 passed** (5 new + the two neighboring modules that assert the same `--dump-dir`/`--dump-format` CLI surface and the pytest-collection config).
  - `--no-cov` is deliberate: `pyproject.toml` `addopts` sets a *suite-wide* `--cov-fail-under=80`, which any partial run reports as 0% by construction. Not a failure signal, so it is switched off for a scoped run rather than answered by widening into the full suite (which the Forward role never runs).
- **Coverage of new code paths:** the changed production statement is in a shell script and the new module is test code; `[tool.coverage.run] source = ["buster_ripper"]` instruments neither, so the suite's coverage figure is unchanged **by construction** from T006's accepted **93.88%**. The behavior the fix enables (`config.DUMP_DIR is None` off-mode) was already covered by the T002–T004 dump-off tests.
- No existing test reads `serve_buster_ripper.sh` (verified by grep before writing, so the witness is the first — see the claim-199 note below), and no `buster_ripper/` source or `tests/qa/` file is touched: `git status` = 1 modified script + 1 untracked test module + 1 evidence file.

## Dependency-claim checks (recorded per contract)

Both rows the extraction pass left as **write-time checks for Forward** (`assumptions/claims.md` run-24: rows 195 and 198, "affirmative single-task claims outside every seam … stay write-time checks for Forward per §3") were checked against source while writing, and both **HOLD** — no correction to the plan was needed:

- **claim-195** — `main.py`'s `--dump-dir` declares `default=None`: HOLDS at `buster_ripper/main.py:79` (`@click.option("--dump-dir", metavar="PATH", default=None, …)`). Load-bearing: it is why *omitting* the flag is the off-switch rather than an error.
- **claim-198** — `flush_all` returns without writing when `config.DUMP_DIR is None`: HOLDS at `buster_ripper/dump.py:108-109`. With it, `main.py:151` (`if dump_dir:` → assignment only inside the truthy branch) and `config.py:20` (`DUMP_DIR: Optional[Path] = None`), the disabled path writes nothing at any trigger.
- **claim-196 / claim-197** (the two `absence` rows for the same off-mode chain) re-read incidentally and unchanged — their subject files are not in this diff.
- **⚠ claim-199 is now FALSE by design** — "No module under `tests/` invokes `serve_buster_ripper.sh`" (`verdicts/claim-199.md`, `HOLDS` at 2026-10-05T02:23:43Z). Its own `Invalidates On` names exactly what landed ("Introduction of `serve_buster_ripper.sh` plus a subprocess/exec reference to it anywhere under `tests/`", plus any new module under `tests/`). The task's `Done when` *requires* that witness, so this is a planned invalidation, not a contradiction: the row must be re-extracted (the proposition becomes "one module invokes it, hermetically via a PATH shim") before the seam is cited again. I did **not** edit `assumptions/claims.md` — those cells belong to the collect/checker passes. Rows 195/198 likewise still read `new`/`unverified` in the ledger; the verification above is the artifact the checker/yield pass should fold in.

## Assumption state (finding `seam-S6-disable-form`, ledger `DC-seam-S6-disable-form-r1`)

- **Implemented for Yield.** Status stays `pending` in the ledger — Forward does not resolve its own finding. Nothing here needs a `BLOCKED-APPROVAL` or `PLAN-GAP` route: the fix is the user's own option, picked unmodified.
- **The landing fired `seam-S6`'s Invalidates-On.** Script-pinned members whose verdicts pin `f4076d1c…` — **159, 160, 161, 162, 163, 179, 180, 191, 192, 193** — are all stale against `d5db2144…` and need fresh re-verification; plus **claim-199** above. The `lang:bash` rows **183–190** keep their documentation digests (bash's rules did not move — and `-`/`:-` semantics were confirmed empirically on bash 3.2.57 as a side-witness), and the README-pinned rows **33, 164, 181, 194** stay fresh: `README.md` is untouched at `fe80cb9acabfe699…` (re-verified this pass). Per the task block: "Do not treat the seam as current on post-fix bytes until that recomposition is recorded."
- **Not touched deliberately:** members 187/189's C2 half (value-word splitting) — that is `seam-S6-value-word-split` → **T012**, next wave, and its `Done when` pins lines 32–38/46 as still carrying the pre-T012 assembly. T012's witness needs the zero-word arm this task made reachable, which is exactly why the waves are ordered this way.

## Durable signal

- **Settled authority exercised:** `FK-010` · serve-script disable form, **option 1**, the user's words verbatim — *"fix the bug, make it to not lie. I do not need the dumps always."* Implemented with no clause added or waived. Its recorded Scope guard held: the dump-on default when `DUMP_DIR` is unset, the formats, the flush policy, the TLS posture and every accepted `T001`–`T006` behavior are unchanged ("I do not need the dumps always" is the ground for wanting the off-switch, not an instruction to flip the default).
- **Override entries:** none exercised; no protected invariant displaced; no `PI:` created or weakened. **Invariant Contradictions Raised: none.** Invariant write ban honored — `.invariants/` untouched.
- **Decision changes/reversals:** none. No plan text needed supersession.
- **Bug-fix rationale that still matters after acceptance:** the defect was `:-` versus `-`, a one character that made line 36's guard *unreachable-false* from the environment. The distinguishing witness is the **unset** case: a "fix" that changed both states (e.g. dropping the default entirely, or rewriting the guard) would silently stop ordinary documented runs from dumping. That is why the witness pins all three states as whole-list equality rather than only testing the disable.

## Concerns (for Yield)

1. **Suite-wide coverage gate vs scoped runs** (`pyproject.toml` `addopts`): any partial `uv run pytest` reports `Required test coverage of 80% not reached. Total coverage: 0.00%` and exits non-zero even when every test passes. That is a trap for exactly the scoped runs the Forward protocol requires; a `-m 'not qa'`-style split or a `--no-cov` convention documented in the project notes would make affected-scope evidence unambiguous. Not this increment's surface to change (no backlog item, and pyproject is outside the touch-set), so it is raised, not fixed.
2. **Evidence capture writes under `$TMPDIR`** — the witness creates a throwaway `bin/` + argv file per test inside pytest's `tmp_path`; no repo or `$HOME` state is touched. Flagged only so Yield can confirm hermeticity rather than re-derive it.
3. **The `Disposition` column** of this task's new `evidence/INDEX.md` row is left empty — that ruling is the user's.

## Questions Asked

**None.** The pre-question gate is not engaged: no `ESCALATION`, `BLOCKED-APPROVAL`, or `PLAN-GAP` surfaced — the outcome, the mechanism, and the exact edit were all settled by `FK-010` option 1, the two write-time dependency claims held, and no escalation-floor action was needed (no `--snapshot-update`, no baseline regeneration, no push, no history rewrite). `CLOSURE: absorbed` would have had nothing left to ask.

## Superseded wording (preserved here per the rewrite-in-place rule)

`plan-summary.md` §Progress Overview, `Forward / Yield` row, rewritten to the current truth by this iteration. Previous wording, verbatim:

> **T006 accepted-local — next: MS-004 QA author pass** | [051-01-yield](iterations/051-01-yield.md): ACCEPT, zero reject-tier findings (2 P3 deferred: `QB:test-tls-posture/private-attr-pinning`, `QB:deps/aiohttp-unpinned`); all 8 clauses verified, claim-38/claim-34 replays PASS/PASS+CONFIRMED (statuses accepted-local), capacity baseline-replayed (L:52→46, P:21 held). Implementation: [051-01-forward](iterations/051-01-forward.md) — `SKIP_TLS_VERIFY` global, callback-wired `--skip-tls-verify`, explicit `TCPConnector` inside `on_startup` (posture process-wide, unset path behaviour-preserving), serve-script boolean guard, README wiring, committed self-signed `127.0.0.1` fixture for QAP-MS004, 6 posture unit tests + absorbed `QB:dump-flush/trigger-value-validation`; suite **278/278 @ 93.88%** (reproduced). Before it: T004 accepted-local ([050-01](iterations/050-01-yield.md)). Still owed when `hyperipper.local:30001` is reachable: QAP-MS002 re-run → MS-002 re-verdict; now: MS-004/QAP-MS004 (forward-qa authors the gate against the fixture) + S6 recomposition / S4 refresh (Forward Collect). **Registered 2026-10-04:** [`FK-010`](decisions.md#fk-010) · serve-script disable form is resolved (option 1) and its remediation is dispatchable as **T011 · Honor the serve script's documented DUMP_DIR="" disable form** — it gates no milestone and displaces no task; MS-004/QAP-MS004 remains the next milestone work.

Nothing in that record is contradicted — T006's acceptance, the 278/278 @ 93.88% suite figure, the deferred P3 pair and the owed MS-002 re-verdict all stand; only "what is current" moved, and `S6 recomposition / S4 refresh (Forward Collect)` is now superseded by the concrete recheck this landing fired.

## Return

- **Checkpoint:** 114 (`3c01cde`) — worktree clean after it; HEAD before it was `901f7ed` (Checkpoint 113), so this increment's diff is exactly `serve_buster_ripper.sh` (1 line) + the new witness module + the session artifacts.
- **Session ID:** `02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573`
- **Iteration:** `052-01-forward`
- **Tasks covered:** T011 alone (anchor; T012 deliberately not pulled in — wave membership per §Planned Task Context).
- **Next:** Yield (local correctness: three-state argv witness, one-line diff scope, header/doc agreement), then the `seam-S6` member re-verification + recomposition the landing fired, then `MS-004`'s QAP-MS004 author pass, which this task neither gates nor displaces.

DEMO-EVIDENCE: `.sessions/codify/02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/evidence/INDEX.md` — no pause

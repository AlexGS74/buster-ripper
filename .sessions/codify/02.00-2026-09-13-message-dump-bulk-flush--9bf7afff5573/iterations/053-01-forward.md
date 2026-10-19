# Iteration 053-01 — Forward — T012: Preserve environment values as single argv words in the serve script

## Orchestrator Prompt (verbatim)

> You are a subagent spawned by another session.
> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
> Task: T012

## Increment Brief

- **Task and goal (my words):** stop the wrapper from silently re-cutting the caller's own value at whitespace — assemble `EXTRA_FLAGS` so each forwarded value reaches `buster-ripper` as exactly one argv word, verbatim, and pin that at the word level on the script's assembled argv.
- **Anchors touched:** no Anchor Map row moves. The edit is inside `serve_buster_ripper.sh` lines 32–42 and the exec expansion (existing referents the map never bound — T012 is post-planning adhoc remediation routed from the `seam-S6` recomposition). The consumer, `buster_ripper/main.py:--dump-dir/--dump-format`, is mapped as **reuse** and is untouched; no `buster_ripper/` file is in this diff.
- **Will NOT create:** no new module, class, command, handler, protocol or config surface; no QA gate test (`tests/qa/` is milestone-owned; QAP-MS004 stays `forward-qa`'s); no new value protocol (no quoting rules, no escaping, no `eval`); no README or header change; line 28's default form (T011's) untouched.
- **Reuse evidence:** no `Mode: NEW` production anchor consumed. The only artifact added is the extension of the **existing** witness module `tests/test_serve_script_argv.py`, which the task's own Scope names ("the existing argv-witness module extended with the value-shape cases") — so no `RB-*` row is owed. Pre-flight: `grep -rln "serve_buster_ripper\|test_serve_script_argv" tests/ buster_ripper/ pyproject.toml` → the witness module is the script's only consumer in the test surface (row 200's census re-verified this pass).
- **Inventions:** none. The array + guarded-expansion mechanism is the plan's own named option (§Verification profile → Uncertainty: "array versus an equivalent that yields the same observable is Forward's"), and `Done when`'s own Invalidates-On wording anticipates "`${EXTRA_FLAGS}` becoming quoted or array-based at line 46".
- **Capacity gate: not applicable — no additive production surface.** The change replaces the six appends and one expansion in place; `serve_buster_ripper.sh` 46 → 54 lines, of which 8 are comments (0 executable statements added).

## Pre-flight gates

- **Plan approval:** session-level Interview approval recorded (`US-015`, round-18 "approve, begin"; plan-summary §Open Questions "STATUS: CLOSED"). T012 is an adhoc row added later and inherits it; the 2026-10-05 recomposition pass records it **dispatchable** with "no user decision is owed". **PASS.**
- **QA contract:** `qa-plan.md` exists; T012 declares **no `VP-*`/`TC-*`/`MS-*`** handle ("Gates no milestone; `MS-004`'s QA author pass remains the next milestone work"). The witness cases are supporting local tests, not gate tests. Nothing to record against a validation point. **PASS.**
- **Protected invariants:** narrowed to the touched path (`serve_buster_ripper.sh`) + the `--dump-dir`/`--dump-format` slice of `buster_ripper/main.py`. The task block's cross-reference reads **Neutral** and the registry was re-read for that narrowed scope: zero `Status: active`/`wip` entries exist — only the eight `planned` `PI:` rows, the preserve-provenance `OI:` blocks and descriptive `CL:`/`CO:`/`E:` rows. Nothing contradicted, displaced or upheld; no override needed; nothing created or edited under `invariants/`. **PASS.**
- **Planned Task Context consumed:** `handoff-context.md` §Planned Task Context → `T012 Planned Task Context` block (finding `seam-S6-value-word-split`, recomposition C2, disposition `absorbed`, ledger `DC-seam-S6-value-word-split-r1`). Its Required behavior, `Done when`, Yield replay and verification profile are mandatory criteria and are reproduced clause-by-clause below. Finding **not** marked resolved here — statuses move at Yield/recomposition.
- **Wave membership:** anchor T012 alone. T011 is `accepted-local` and its `seam-S6` recomposition landed `SAFE`, so the recorded ordering gate is met; no other task is pulled in (nothing follows T012 in the plan).
- **Creation gate:** nothing new is created, so no `Mode: NEW` binding is required. **PASS.**

## Failure-path evidence

| Unhappy path | Approved outcome (authority) | Handling / observation evidence | Disposition |
|---|---|---|---|
| Value contains `$IFS` whitespace (`DUMP_DIR="/tmp/my docs"`) | argv carries the flag + **one** word equal to the value (consumer contract: `main.py:78` `metavar="PATH"`, `:152` `Path(...).expanduser().resolve()` — an opaque path string) | array element quoted at append; whole-list equality witness `test_path_containing_a_space_arrives_intact`; before/after in `evidence/t012-value-word-integrity.txt` §counterexamples | **safe** (was a live defect: 10 words → 9) |
| Value is whitespace-only (`" "`, `"\t"`, `" \t "`) — guard true, field zero | one word containing that whitespace, **and** the next guard's flag survives (no neighbor swallowing) | `[[ -n " " ]]` is true, so the append happens; 3 parametrized cases + the `[ ]` row of the evidence file (before: `--preserved-thinking` eaten, argc 8) | **safe** (was a live defect) |
| Value is a glob pattern that matches files in the cwd | verbatim one word — the wrapper must not pathname-expand | expansion quoted, so neither splitting nor generation runs; `test_glob_metacharacters_are_not_expanded` plants `a.md`/`b.md` and chdirs, making it fail-first (before: one word per file) | **safe** (was a live defect) |
| Value looks like a flag (`--verbose`, `--dump-format`) or carries quotes/backslash/newline | forwarded verbatim — "no new value protocol" | 4 parametrized verbatim cases + the tab-and-newline row of the evidence file (before: `x y z` from `x\ty\nz`) | **safe** |
| Every append suppressed (`DUMP_DIR="" DUMP_FORMAT="" PRESERVED_THINKING=0`) | exec still runs with a well-formed argv; **no `set -u` abort** | `${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}`; `TestNothingAppended` + the evidence file's mutation section (naive bare `"${EXTRA_FLAGS[@]}"` on bash 3.2.57 → `EXTRA_FLAGS[@]: unbound variable`, exit 1, exec never reached) | **safe** |
| Wholesale-quoting `${EXTRA_FLAGS}` (the tempting wrong fix) | every appended flag would collapse into **one** argv word | rejected by construction: the all-suppressed witness (`argc=6`) and the documented-form witnesses each enumerate words individually | **not-applicable** — outcome unreachable under the shipped mechanism |
| Documented word sets drift (regression risk named in the verification profile) | README 71–93/192 and header line 14 keep byte-identical word sets | `TestDocumentedForms` (6 cases) + T011's three-state witnesses, unchanged and still green; before/after argv pairs side by side in the evidence file | **safe** |
| Value is empty (`DUMP_DIR=""`) | guard false → nothing appended (T011's settled behavior) | guard conditions untouched; `test_empty_value_suppresses_the_flag` still green on the new bytes | **safe** |
| Positional arguments to the script itself | still never forwarded (row 163's negative) | the exec line gained no positional expansion; `test_positional_arguments_are_not_forwarded`-shaped coverage stays the whole-list equality (an extra positional breaks it) | **safe** — unchanged by this increment |
| Signal/cancellation, partial completion, retry/duplicate delivery, cleanup, competing updates | — | no such surface: this increment is a one-shot argv assembly in a launcher that `exec`s; no loop, no IO, no state, no write | **not-applicable** (by family) |

## What was implemented

1. **`serve_buster_ripper.sh` — the assembly (lines 32–42) is an array, not a string.**

   ```bash
   EXTRA_FLAGS=()
   [[ "${STRIP_DATE}"          == "1" ]] && EXTRA_FLAGS+=(--strip-date)
   [[ "${VERBOSE}"             == "1" ]] && EXTRA_FLAGS+=(--verbose)
   [[ "${SKIP_TLS_VERIFY}"     == "1" ]] && EXTRA_FLAGS+=(--skip-tls-verify)
   [[ -n "${DUMP_DIR}"               ]] && EXTRA_FLAGS+=(--dump-dir "${DUMP_DIR}")
   [[ -n "${DUMP_FORMAT}"            ]] && EXTRA_FLAGS+=(--dump-format "${DUMP_FORMAT}")
   [[ "${PRESERVED_THINKING}"  == "1" ]] && EXTRA_FLAGS+=(--preserved-thinking)
   ```

   Each append now contributes whole words: the flag literal and the value are separate array elements, and the value element is quoted so `$IFS` splitting and pathname generation never see it. Everything the task pins as unchanged stayed unchanged — the six **guard conditions** are byte-identical (`== "1"` for the toggles, `-n` for the two value guards), line 28's `${DUMP_DIR-…}` unset-only default is T011's, `--preserved-thinking` is still the last append, header line 14 and `README.md` are untouched, and no `buster_ripper/` file is in the diff. Only the right-hand sides and the initialiser moved (plus two explanatory comment blocks, which is the 8-line growth).

2. **`serve_buster_ripper.sh` — the exec expansion (line 54) is the nounset-safe guarded array form.**

   ```bash
   exec buster-ripper \
     --upstream "${UPSTREAM}" \
     --host     "${HOST}" \
     --port     "${PORT}" \
     ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}
   ```

   The bare `"${EXTRA_FLAGS[@]}"` is **not** available here: on bash 3.2 — which is what this script's `/usr/bin/env bash` shebang resolves to on this machine (verified: `GNU bash, version 3.2.57(1)-release (arm64-apple-darwin25)`) — expanding an empty array under `set -u` aborts with `EXTRA_FLAGS[@]: unbound variable`. That is exactly the all-appends-suppressed run T011 made reachable and T012's `Done when` requires to exec cleanly, so the `${arr[@]+"${arr[@]}"}` guard is used: zero words when nothing was appended, individually-quoted words otherwise. The reason is recorded as a comment at the site so the next reader does not "simplify" it into the failing form, and the witness pins it (see §Tests — the mutation run proves that witness is live, not decorative).

   - Script fingerprint moves `d5db2144a2452840e7a5284c524c3fbfeef60dc117a55d61fab57a5aeafc5c90` → **`c58256fbec793c520720c9e86fee8740653939275b508f50b212ab60bd0908af`** (46 → 54 lines). This is the event `seam-S6`'s own `Invalidates-On` names ("quotes added to line 36/37's value interpolation, `${EXTRA_FLAGS}` becoming quoted or array-based at line 46"), so the script-pinned members must be re-verified and the seam recomposed before S6 is citable again — §Assumption state.
   - `README.md` re-hashed this pass: `fe80cb9acabfe699fac070a69b43ac425436606d627b243825d4a028d75a1c15` — byte-identical to the fingerprint rows 33/164/181/194 pin, so those four stay fresh and are not fired by this increment.

3. **`tests/test_serve_script_argv.py` — extended from 5 to 23 tests** (240 lines, Green), reusing the existing PATH-shim fixture (`serve_argv`) and the whole-list-equality shape claim-201 records. Module docstring rewritten to state both duties (supersession-by-rewrite; the old wording is in §Superseded wording).

   - **`TestValueWordIntegrity`** (11 cases) — the C2 class at the word level: space-containing path; whitespace-only value parametrized over `" "`, `"\t"`, `" \t "` (each also proves the *following* flag survives); `DUMP_FORMAT="a b"` through line 37's guard; both values whitespace-bearing in one run (word order preserved, "neither split borrows from the other"); verbatim pass-through parametrized over `--verbose`, `--dump-format`, `q"uote`, `x\ty\nz`; and the glob case.
   - **`TestNothingAppended`** (1 case) — the zero-word arm as a persistent witness rather than an evidence-only probe: `DUMP_DIR="" DUMP_FORMAT="" PRESERVED_THINKING=0` → `argv == _TRIO`, with the fixture's `assert result.returncode == 0` surfacing the interpreter's stderr on abort. This is the absorbed backlog item.
   - **`TestDocumentedForms`** (6 cases — corrected at drain from the "7" stated twice: 4 parametrized toggles + the upstream/port case + the format-with-a-directory case; 6 is also the figure consistent with this report's own "18 new" = 11 + 1 + 6) — the regression half the verification profile names as the concrete wrong outcome: each README toggle form (individually and all three together) contributes exactly its own flag word ahead of the dump pair; `UPSTREAM=http://10.0.0.1:30000 PORT=30002` keeps the always-on trio quoted; the README:192 `DUMP_FORMAT`+`DUMP_DIR` pair keeps two flag/value pairs.
   - **New fixture `matched_pattern`** — plants `a.md`/`b.md` in `tmp_path` and chdirs there, so the glob witness is fail-first (a pattern matching nothing would pass under the defective expansion too). Kept as a fixture so the test signature stays within L→N→C→P's P≤3.
   - Every new case asserts the **complete** argv list, never membership or a slice — required both by claim-201's `Invalidates-On` ("a new environment-state test that asserts only part of the argv") and because whole-list equality is what makes "no orphan word", "no swallowed neighbour flag" and "no positional forwarding" observable in one readable diff.
   - The three T011 witnesses and the documentation-agreement witness are **unedited** and still pass — that is the "byte-identical word sets" clause enforced by construction rather than by restatement.

## Verification against the task's `Done when`

| clause | state |
|---|---|
| assembly (lines 32–38 and 46) yields one argv word per forwarded value | ✅ landed — array append at 36/37, guarded array expansion at the exec |
| line 28's default form (T011's), the guard **conditions**, header line 14 and `README.md` unchanged | ✅ line 28 byte-identical; six guard conditions byte-identical; header and README untouched (`README.md` still `fe80cb9a…`) |
| witness covers `DUMP_DIR="/tmp/my docs"` → exactly one `--dump-dir` whose value word is `/tmp/my docs` | ✅ `test_path_containing_a_space_arrives_intact` |
| `DUMP_DIR=" "` → `--dump-dir` with the whitespace value as its own single word and a separate `--preserved-thinking` word still present | ✅ `test_whitespace_only_value_is_its_own_word[ ]` (plus `\t` and ` \t `) |
| `DUMP_FORMAT="a b"` → one `--dump-format` word | ✅ `test_format_value_containing_a_space_is_one_word` (+ the both-values case) |
| all-appends-suppressed case → exec runs with no `unbound variable` abort on the interpreter `env bash` resolves | ✅ `TestNothingAppended`, run on bash 3.2.57; proven live by the naive-array mutation (exit 1, `unbound variable`, exec never reached) |
| the three T011 states and every documented README form keep byte-identical word sets | ✅ T011's three witnesses green unedited; `TestDocumentedForms` covers README 71–93/192; before/after argv pairs captured per row in the evidence file |
| full hermetic suite passes with no regression | ⏳ **Yield's independent run** — per role boundary I ran the affected scope, not the full suite (see §Tests). Nothing under `buster_ripper/` changed and the script's only test consumer is the witness module (row 200), so no other test can regress except through the script's own behavior. |
| script-pinned members **159, 160, 161, 162, 163, 179, 180, 191, 192, 193** re-verified fresh against the new bytes | ⏳ **owed and now unblocked** — this landing is the trigger; see §Assumption state |
| `lang:bash` claim-189 and script-pinned claim-162 re-checked on the current digests | ⏳ **owed** — 189's premise is what this fix answers and 162's own `Invalidates-On` names "replaced by … an array expansion"; both need the fresh digests before S6 can be cited |
| a fresh `seam-S6` recomposition lands recording C2's class as closed | ⏳ **owed** — dispatch it **with the 22 `CL:` handles in its arguments** (rider 1) |

## Backlog pickup

`Backlog pickup: QB:serve-script-argv/zero-word-arm` — absorbed, marked `[x]` with this iteration's evidence. Read `quality-backlog.md` (12 unchecked items) against the touch-set (`serve_buster_ripper.sh`, `tests/test_serve_script_argv.py`):

- **Absorbed (1):** `QB:serve-script-argv/zero-word-arm` (`tests/test_serve_script_argv.py:105-127`) — its own trigger text is "pick up when `serve_buster_ripper.sh`/this module are next touched … T012's `Done when` already requires witnessing it on the post-restructure bytes". Now pinned persistently by `TestNothingAppended` on the post-restructure bytes, and the item's "not by a persistent test" gap is closed. No standalone increment, no extra checkpoint.
- **Rides the wave, not Forward's to write (1):** `QB:assumption-registry/co-serve-script-basis-republish` — the remedy is one **handle-bearing S6 recomposition dispatch** into `invariants/`, which the triad rule forbids this role from editing (see §Assumption state). Recorded so the dispatcher carries it; left unchecked.
- **Explicit zero (12 — corrected at drain from "10"; 14 unchecked at read time, 13 by the pickup including the added `QB:assumption-registry/co-serve-script-basis-republish`, minus the one absorbed row — `QB:report-accuracy/t012-counts`):** the remainder target `plan-summary.md`, `buster_ripper/dump.py`, `buster_ripper/routes/*`, `tests/test_backend.py`, `tests/test_routes_integration.py`, `tests/test_tls_posture.py` and `pyproject.toml` — none is in this increment's files or module. No matching item silently skipped.

## Capacity gate

- Classification: **not applicable — no additive production surface.** The six appends and the one expansion are *replacing* in place; nothing added, deleted, or restructured elsewhere. Pre-edit `serve_buster_ripper.sh` 46 lines / post-edit 54, of which the delta is 8 comment lines (0 added executable statements, 0 added branches).
- No trigger condition applies: no oversized legacy file grew (54 lines, Green), no function exceeded any metric (see §L→N→C→P), no new module was created.
- Non-production file: `tests/test_serve_script_argv.py` 137 → 240 lines, still Green (<300) and no ratchet crossed.

## L→N→C→P compliance (new/changed functions)

`codify scan path serve_buster_ripper.sh --git --detailed` → **No refactoring issues found**. `codify scan path tests/test_serve_script_argv.py --git --detailed` → **No refactoring issues found** (re-run after two first-draft signatures exceeded P: the glob case's `tmp_path, serve_argv, monkeypatch` and the boolean case's `serve_argv, toggles, expected` were both brought to ≤3 by extracting the `matched_pattern` fixture and collapsing the parametrize into one `case` parameter). Whole-file scan is also clean.

| unit | L | N | C | P | note |
|---|---|---|---|---|---|
| `serve_buster_ripper.sh` top level (the six appends + exec) | not scored by the scanner (shell) | — | — | — | six single-line guards, no nesting, no new branch: the array gained no `if` |
| `TestValueWordIntegrity::test_path_containing_a_space_arrives_intact` | 2 | 0 | 1 | 2 | |
| `…::test_whitespace_only_value_is_its_own_word` | 2 | 0 | 1 | 3 | 3 parametrized cases |
| `…::test_format_value_containing_a_space_is_one_word` | 2 | 0 | 1 | 2 | |
| `…::test_both_values_keep_their_own_words` | 3 | 0 | 1 | 2 | |
| `…::test_value_is_forwarded_verbatim` | 2 | 0 | 1 | 3 | 4 parametrized cases |
| `…::test_glob_metacharacters_are_not_expanded` | 2 | 0 | 1 | 2 | |
| `TestNothingAppended::test_all_appends_suppressed_still_reaches_the_binary` | 1 | 0 | 1 | 2 | |
| `TestDocumentedForms::test_boolean_forms_contribute_their_own_flag` | 3 | 0 | 1 | 3 | 4 parametrized cases |
| `TestDocumentedForms::test_custom_upstream_and_port` | 2 | 0 | 1 | 2 | |
| `TestDocumentedForms::test_format_with_a_directory` | 3 | 0 | 1 | 2 | |
| `matched_pattern` (fixture) | 4 | 0 | 1 | 2 | |

Zero metrics in the gray zone; no function needs extraction. No `# ESSENTIAL COMPLEXITY` marker added.

## Tests and coverage

- **New tests:** 18 (module total 5 → 23), all hermetic — PATH shim, no server, no port bound, no upstream contacted.
- **Fail-first proof (required for a behavior-defect fix), run on the pre-task bytes** (`git show HEAD:serve_buster_ripper.sh` swapped in, then restored): **8 failed, 15 passed** — the space-path case, all three whitespace-only cases, the format case, the both-values case, the tab/newline verbatim case, and the glob case. Each failure is the finding's own counterexample (`At index 7 diff: 'a.md' != '*.md'`, `argc` 8 instead of 9 where a flag was swallowed).
- **Second mutation (proves the absorbed zero-word witness is live):** the guarded expansion replaced by bare `"${EXTRA_FLAGS[@]}"` → **1 failed, 22 passed**, failing with `line 50: EXTRA_FLAGS[@]: unbound variable`. So the bash-3.2 guard is pinned by a test that fails if someone removes it.
- **Narrow runs during implementation:** `uv run pytest tests/test_serve_script_argv.py -q --no-cov` → 23 passed (after the signature fixes; run before the mutations above, which restored the file to `c58256fb…` each time).
- **Affected scope (run ONCE before checkpoint):** `uv run pytest tests/test_serve_script_argv.py tests/test_main_cli.py tests/test_pytest_config.py -q --no-cov` → **38 passed**. Same scope T011 used for the same touch-set: the script has no Python importer, `test_main_cli.py` asserts the `--dump-dir`/`--dump-format` CLI surface the wrapper feeds, and `test_pytest_config.py` guards the collection config the new witness relies on. `--no-cov` for the recorded reason in §Concerns 1.
- **Coverage:** the increment's new code is shell (outside `[tool.coverage.run] source = ["buster_ripper"]`) and test code (likewise omitted); no `buster_ripper/` line was added or changed, so the ≥80% requirement is unaddressed by construction — the suite-wide figure is Yield's independent measurement (T011's baseline: 283 @ 93.88%).

## Dependency-claim checks (recorded per contract)

| claim the plan implied | what the dependency does | read | recorded correction |
|---|---|---|---|
| "bash-version behavior for empty-array expansion under `nounset` must be checked on the interpreter the script actually resolves … rather than assumed" (§Verification profile → Uncertainty) | `bash` **3.2.57** (what `/usr/bin/env bash` resolves to here) aborts on `"${arr[@]}"` for an empty array under `set -u`: `arr[@]: unbound variable`; `${arr[@]+"${arr[@]}"}` expands to zero words and to individually-quoted words otherwise | `env bash --version`; two probes (`unbound variable` on the naive form, `argc` correct on the guarded form) and the mutation run in §Tests | **Implemented the guarded form** — the plan left the mechanism to Forward precisely because it declined to assume this. No plan decision reversed. |
| an unquoted expansion word-splits and an empty value contributes zero words (row 189, `HOLD`, `lang:bash`) | confirmed: `DUMP_DIR=" "` under the old assembly produced `--dump-dir` followed directly by `--preserved-thinking` (argc 8) | probe + the before/after rows in `evidence/t012-value-word-integrity.txt` | none — confirms, so nothing recorded as a change |
| the value is an opaque path string with no character restriction (rows 195/196 and `main.py:78/:152`) | confirmed: `--dump-dir` declares `metavar="PATH"`, `default=None`, and `main()` does `Path(dump_dir).expanduser().resolve()` — no wrapper-level grammar to add | `buster_ripper/main.py:78`, `:151-152` re-read this pass | none — confirms; no `buster_ripper/` file touched |
| a bash array `+=` append and the `${arr[@]+…}` guard exist on the target interpreter (never stated by the plan; implied by "array versus an equivalent") | `+=` on arrays is bash 3.1+; both used forms verified on 3.2.57 | probes above | none — verified before encoding |

Row 200's census (`no module under tests/ other than tests/test_serve_script_argv.py` invokes the script) was re-read this pass: `grep -rln "serve_buster_ripper" tests/ buster_ripper/ pyproject.toml` → only the witness module. The census did **not** grow, which is what that row exists to bound.

**Shared-claim note (per the reach table):** rows **187**, **189**, **200** and **201** are listed against `T011, T012`. T011 is `accepted-local`, so no sibling task is left built on a different reading; none of the four moved — 187/189 confirm as stated, and 200/201's subjects are the witness module's shape, which this increment preserved deliberately: whole-list equality everywhere, `_TRIO`/`_THINKING` kept as plain constants (the documented-form override case spells its expected list out literally rather than making `_TRIO` variadic, because that is one of claim-201's `Invalidates-On` clauses). Nothing to raise; recorded so Yield can check it rather than re-derive it.

## Assumption state (finding `seam-S6-value-word-split`, ledger `DC-seam-S6-value-word-split-r1`)

- **Finding id:** `seam-S6-value-word-split` (the 2026-10-05 `seam-S6` recomposition's C2). **`Assumption impact` disposition: `absorbed`** — routed with `CLOSURE: absorbed` / Question required: no, so no user question was asked or owed. Implemented-for, **not** marked resolved here: statuses move at Yield and at the recomposition, per role boundary.
- **Nothing under `assumptions/` was written by this role** — `claims.md`, `seams.md` and `verdicts/` are absent from this increment's diff (verified against `git status`). The claims ledger rows this landing fires are re-verification work for the assumption pass, not for Forward.
- **What this landing fires**, so the dispatcher can carry it in the same wave: script-pinned members **159, 160, 161, 162, 163, 179, 180, 191, 192, 193** re-verified fresh against `c58256fb…` (their own `## Source Fingerprint` pins `d5db2144…`); `lang:bash` claim-189 and script-pinned claim-162 re-checked on the current digests (162's `Invalidates-On` names "replaced by per-flag conditionals or an array expansion" — that is precisely this edit); and a fresh `seam-S6` recomposition that records C2's class as closed. **Rider 1:** dispatch that recomposition **with the 22 `CL:` handles supplied in its arguments**, which is what lets `composition-promote` upsert `CO:serve-script-env-to-argv/env-var-flag-crossing` in place and retire the fired `fb247496…` basis (`QB:assumption-registry/co-serve-script-basis-republish`). **Rider 2:** **Forward Collect still owes one atomic premise row** for the value-grammar contract — every forwarded value reaches argv as exactly one word, and `$IFS` whitespace, glob metacharacters and a leading dash carry no wrapper-level restriction. That row is why the post-T012 composition can trace C2's class as a *dimension* instead of declining it for want of a premise; this increment ships the behavior and the witness, but the premise row is the collect role's to write. Until it lands, `SAFE` on S6 is not C2's closure, per the task block's own warning.
- `README.md` unchanged (`fe80cb9a…`), so README-pinned members 33/164/181/194 stay fresh and are not part of the re-verification set — as the `Done when` states.

## Invariant Contradictions Raised

None. No invariant was read as authority, created, edited, superseded, or deleted; the registry's two dirty files at dispatch start (`invariants/buster_ripper/assumption-claims.md`, `invariants/index.md`) are the 2026-10-05 promotion pass's own pending artifacts and are **not** in this increment's diff — they ride the checkpoint as pre-existing session evidence, untouched by this role. The `CL:serve_buster_ripper.sh/*` records those files pin now sit on expired bytes (`d5db2144…` → `c58256fb…`), which is the expected consequence of the landing and is the assumption pass's to republish, not mine.

## Concerns (for Yield, non-blocking)

1. **Suite-wide coverage gate vs scoped runs** (carried from [052-01-forward](052-01-forward.md) §Concerns, unchanged): `pyproject.toml`'s `addopts` sets `--cov-fail-under=80` suite-wide, so any partial run reports `Total coverage: 0.00%` and exits non-zero by construction. Not this increment's surface — `pyproject.toml` is outside the touch-set and no backlog item names it — so it is raised again rather than fixed.
2. **Mechanism choice is mine, the observable is the plan's.** The array is the plan's named option and `Done when`'s Invalidates-On wording anticipates it, so this is a report note, not a deviation. The alternative that would also satisfy the observable — moving the always-on `--upstream/--host/--port` trio into the same array so the empty-array case cannot arise — was **rejected** because it edits lines 43–45, which the task's Scope names as out of scope, and it rewrites the shape claim-162 pins ("always-on trio … and the assembled `EXTRA_FLAGS`"). The guarded expansion keeps the diff inside the named lines at the cost of one dense idiom, which is documented at the site and pinned by a test.
3. **Not gold-plated, deliberately:** no value validation, no `--dump-dir=`-style protocol, no warning on whitespace-bearing paths. `Required behavior` says "faithful; no new value protocol", and the consumer contract treats the value as an opaque path, so the wrapper stays a transparent forwarder. If the user ever wants the wrapper to *reject* a leading-dash value, that is a new decision for Interview, not a Forward convenience.

## Durable signal

- **Override entries exercised: none.** No approved override was needed or used; no protected invariant was contradicted; nothing in `invariants/` was written by this role.
- **Decision change or reversal: none.** `FK-010` option 1's scope guard (line 28's default form, the guard conditions, header line 14, README) is intact — T012's own `Done when` re-states those as unchanged, and this increment honors them.
- **Bug-fix rationale that still matters after acceptance:** the wrapper's argv assembly must be whole-word by construction, and the reason the exec line uses `${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}` instead of the obvious `"${EXTRA_FLAGS[@]}"` is **bash 3.2's empty-array-under-`set -u` abort on this machine's interpreter**. A future reader who "simplifies" that idiom re-breaks the documented `DUMP_DIR=""` disable path. The guard is pinned by `TestNothingAppended` and the mutation is recorded in `evidence/t012-value-word-integrity.txt`, so the constraint travels with the code.
- **Authority this increment implemented from:** finding `seam-S6-value-word-split` (C2) with disposition `absorbed`, ledger `DC-seam-S6-value-word-split-r1`; the consumer contract at `main.py:78`/`:151-152`; the wrapper's documented role (header 11–16, README 71–93/192 — neither documents any restriction the wrapper may add).
- **Invariant entries changed because of this implementation: none** (registry rows on `serve_buster_ripper.sh` bytes are evidence owned by the promotion pass).
- **Wave accounting:** two waves, anchor T011 (accepted, `seam-S6` recomposed `SAFE`) then T012 (this increment). One Yield for this wave. Gates no milestone; `MS-004`'s QAP-MS004 `forward-qa` author pass remains the next milestone work and is neither gated nor displaced by T012. Still owed independently of this wave: the QAP-MS002 re-run → MS-002 re-verdict when `hyperipper.local:30001` is reachable.

## Roadmap

No `future-scope` row added. This increment ships the correct minimal core with nothing above the MVP floor deferred: the value-grammar contract the finding named is implemented **in full** (one word per value, verbatim, no protocol of the wrapper's own), the zero-word arm is closed rather than parked, and the two things still outstanding are not deferred scope — they are this wave's own verification riders (a recomposition dispatch and a premise row) plus an assumption-pass re-verification, all recorded in §Assumption state and on the backlog.

## Demo evidence

`evidence/t012-value-word-integrity.txt` — raw argv as recorded by the PATH shim for every `Done when` state, **both byte states side by side** (pre-task `d5db2144…` vs post-task `c58256fb…`), plus the naive-array mutation with the interpreter's stderr. Rendered with Python repr so whitespace and control characters are visible inside one line: the space path (`argc=10` → `argc=9`), the swallowed `--preserved-thinking` (argc 8 → 9), the tab, the format value, both-values, the glob against planted `a.md`/`b.md`, the flag-shaped and punctuation-bearing values, the all-suppressed trio, T011's three states, and the six README-documented forms. One row appended to `evidence/INDEX.md` (Disposition left empty — the user's ruling).

DEMO-EVIDENCE: .sessions/codify/02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/evidence/INDEX.md — no pause

## Questions Asked

**None.** The pre-question gate was applied at the one real fork (the empty-array mechanism) and closed deterministically: the interpreter is fixed and observable (`env bash` → 3.2.57), the `Done when` states the required outcome (no `unbound variable` abort), and the plan explicitly assigns the mechanism to Forward — a single authority-ranked remediation, so `absorbed`, not a question. No materially different admissible outcome remained unranked, nothing was deferred to Yield as a question, and no `BLOCKED-APPROVAL` or `PLAN-GAP` was hit: no escalation-floor action was required (no snapshot baseline, no project-declared must-ask step), and no unplanned load-bearing referent needed creating.

## Superseded wording (preserved here per the rewrite-in-place rule)

`tests/test_serve_script_argv.py` module docstring, first line:

> `"""Hermetic argv witness for serve_buster_ripper.sh (T011).`

rewritten in place to `(T011, T012)`, with T011's FK-010 paragraph retained verbatim below it and a new paragraph added recording the C2 mechanism and its fix (quoted in §What was implemented, item 3). The task-specific T011 framing of the file is unchanged in substance; only the scope statement moved, because the module now witnesses the shape of every forwarded value and not just the presence of one flag.

## Post-checkpoint correction (Checkpoint 119 follow-up)

Checkpoint 119's code and evidence are exactly as reported; one measured figure in this report and in
`handoff-context.md` was stale by 8 lines. `tests/test_serve_script_argv.py` is **240** lines, not 232 —
the earlier figure was read before the `matched_pattern` fixture (8 lines) was added when the two
parameter-count violations were fixed after the first LNCP scan. All four mentions corrected. Nothing
about the conclusion changes: 240 is still Green (<300), no FL-LNCP ratchet crossed, and the reported
metrics are `codify scan` output taken after the fixture existed. The `137 → 240` growth is the
witness-scope expansion the task's Scope asked for ("the existing argv-witness module extended with the
value-shape cases"), not drift.

## Return

- **Session ID:** `02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573`
- **Iteration:** [053-01-forward](053-01-forward.md)
- **Tasks covered:** T012 (anchor, alone — no clustering: nothing follows it in the plan)
- **Files changed:** `serve_buster_ripper.sh` (`d5db2144…` → `c58256fb…`, 46 → 54 lines), `tests/test_serve_script_argv.py` (5 → 23 tests, 137 → 240 lines), `quality-backlog.md` (1 item absorbed), session files
- **Affected scope:** 38 passed · **Fail-first:** 8 of 18 new tests fail on the pre-task bytes · **Mutation:** 1 fails on a naive-array expansion
- **Ready for:** Yield (local correctness: L→N→C→P, fail-first witnesses, backlog pickup, the `Done when` clause table). Yield also owns the two verification riders' next step: the fresh `seam-S6` recomposition dispatched **with the 22 `CL:` handles** (which retires the fired `CO:` basis), and the Forward Collect premise row for the value-grammar contract that turns C2's class into a traced dimension. S6 must not be cited as current until that recomposition lands on `c58256fb…`.
- **Next increment planned:** `MS-004`'s QAP-MS004 author pass (`forward-qa`) — the next milestone work, independent of this wave's acceptance.

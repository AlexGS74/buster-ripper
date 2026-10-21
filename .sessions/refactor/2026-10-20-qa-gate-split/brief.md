# Split the MS-001 QA gate module

- Goal: split `tests/qa/test_qap_ms001.py`'s drift-observation helpers and witnesses out of the scenario gates, and split `tests/qa/harness.py`'s process driver from the artifact readers, with behavior unchanged.
- Target: roadmap `RM-018`.
- In scope: `tests/qa/test_qap_ms001.py`, `tests/qa/harness.py`
- Importers: `tests/qa/test_qap_ms002.py`, `tests/qa/test_qap_ms004.py`
- Out of scope: `buster_ripper/dump.py` — `RM-002` is taken by `origin/fix/tool-reference-scrub` (stale since 2026-09-17)
- Verify: `.venv/bin/python -m pytest tests/qa/test_qap_ms001.py tests/qa/test_qap_ms002.py tests/qa/test_qap_ms004.py -o addopts=`
- Final verify: `uv run pytest`
- Worktree: /Users/alexhubitski/Worktrees/buster-ripper/refactor-qa-gate-split

## Collected
| # | Candidate | Why | Recorded as | State |
|---|---|---|---|---|
| 2 | tests/qa/test_qap_ms001.py | 1035 lines (threshold: 700); `test_tc108_documented_format_preserves_contents_at_2n_writes` L:78 N:3 C:6 (Long Function, Deep Nesting); `assert_documented_contents` L:47 N:2 C:10 P:4 (Long Function, Medium Nesting, Medium Complexity, Medium Parameters); ahead of RM-018 · drift-helper/witness split and CE-2 call-site pin | RM-018 | free |
| 3 | tests/qa/harness.py | 365 lines (review: <=300); `__init__` L:41 N:1 C:7 P:8 (Long Function, Medium Complexity, Too Many Parameters); ahead of RM-018 · process-driver/artifact-reader split | RM-018 | free |

- Tool gaps: `quality-backlog.md`, cited by RM-002/RM-018, is absent from the tree after the 02.00 squash, so the backlog rows could not be read and `Recorded as` names the roadmap handle only.

## Safety net
- Ran: `uv run python -m pytest tests/qa/test_qap_ms001.py tests/qa/test_qap_ms002.py tests/qa/test_qap_ms004.py -o addopts=` (the brief's `.venv/bin/python` is absent in the worktree; `uv run` resolves the same interpreter) -> 38 passed, 0 skipped
- Scan: `tests/qa/test_qap_ms001.py` — file 1035 lines (threshold 700); 23 issues in 16 functions; top: `test_tc108_documented_format_preserves_contents_at_2n_writes` L:78 N:3 C:6 P:1 (Long Function, Deep Nesting, Medium Complexity), `test_tc100_collapsed_format_defers_every_write_until_a_flush` L:53 N:2 C:4 P:1, `assert_documented_contents` L:47 N:2 C:10 P:4, `test_hostile_r2_documented_writer_does_not_escape_the_dump_root` L:41 N:0 C:1, `test_r5_near_miss_session_ids_are_not_merged` L:39 N:2 C:4. `tests/qa/harness.py` — file 365 lines (review <=300); 9 issues in 7 functions; top: `__init__` L:41 N:1 C:7 P:8 (Long Function, Medium Complexity, Too Many Parameters), `startup_refused` L:23 N:1 C:2 P:4 (Medium Length and Parameters), `require_upstream` L:11 N:2 C:5, `wait_tcp` L:8 N:2 C:3, `poll_until` L:6 N:2 C:3
- Sizes: tests/qa/test_qap_ms001.py: 1035, tests/qa/harness.py: 365
- Public paths kept: none (neither module defines `__all__`; nothing outside the repository imports `tests.qa.harness`)

## Map

### U0 · comments
| # | Kind | What | From | To |
|---|---|---|---|---|
| 1 | shrink | module docstring (61 lines of re-entry history) and the eight witness docstrings' pre-repair narratives | tests/qa/test_qap_ms001.py | kept reasons: `.sessions/codify/02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/chronicle.md` (077/096/099/108 entries) and `qa-plan.md` §QA Gate Matrix / §Scenario Contracts |
| 2 | shrink | module docstring (26 lines) and `assert_upstream_response`'s docstring (20 lines) | tests/qa/harness.py | kept reasons: `.sessions/codify/02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/decisions.md` §US-009 (endpoint correction), `.agents/skills/qa-project/SKILL.md` §Upstream (MS-004 amendment), same session's `chronicle.md` (dropped body limb) |
- Narrow tests: tests/qa/test_qap_ms001.py, tests/qa/test_qap_ms002.py, tests/qa/test_qap_ms004.py

### U1 · tests/qa/harness.py: split the process driver from the artifact readers
| # | Kind | What | From | To |
|---|---|---|---|---|
| 1 | move | `require_upstream`, `free_port`, `wait_tcp`, `poll_until`, `Proxy`, `startup_refused`; `REPO_ROOT`, `PROXY_BIN`, `COUNTED_ENTRY`, `TLS_PROBE_ENTRY`, `UPSTREAM` | tests/qa/harness.py | tests/qa/harness_process.py |
| 2 | reshape | `Proxy.__init__` (called by `test_tc100_collapsed_format_defers_every_write_until_a_flush`, and by every gate in test_qap_ms002/ms004): extract `_build_command` (the counted/probe/plain command selection) | tests/qa/harness.py | tests/qa/harness_process.py |
- Importers to update: tests/qa/test_qap_ms001.py, tests/qa/test_qap_ms002.py, tests/qa/test_qap_ms004.py
- Witnesses to re-point: invariants/buster_ripper/assumption-claims.md:2655,2656,2660 -> tests/qa/harness_process.py (`Proxy`, `PROXY_BIN`, `COUNTED_ENTRY`, `cmd`); invariants/index.md:289 names `qa/harness.py` only inside the aggregate-fingerprint cell, which any tests edit already invalidates and the registry lane regenerates — no path-only re-point there
- Tests with import lines or patch targets to update: tests/qa/test_qap_ms001.py:74-87, tests/qa/test_qap_ms002.py:40-49, tests/qa/test_qap_ms004.py:50
- Narrow tests: tests/qa/test_qap_ms001.py tests/qa/test_qap_ms002.py tests/qa/test_qap_ms004.py

### U2 · tests/qa/test_qap_ms001.py: extract the documented-format observation helpers into a support module
| # | Kind | What | From | To |
|---|---|---|---|---|
| 1 | move | `documented_files`, `README_LINE_PATTERN`, `FULL_CLAUSE`, `KB_CLAUSE`, `CHANGE_CLAUSE`, `MSGS_CLAUSE`, `TOOLS_CLAUSE`, `index_lines`, `production_diff_text`, `changed_lines`, `diff_changed_lines`, `_content_lines`, `assert_wellformed_diff`, `rendering_section`, `assert_documented_contents`, `count_report`, `ops_by_name`, `marker`, `nonce`, `no_dump_config_override` (autouse fixture), `COLLAPSED` | tests/qa/test_qap_ms001.py | tests/qa/qap_ms001_support.py |
| 2 | move | `listing` | tests/qa/test_qap_ms001.py | tests/qa/harness.py |
- Importers to update: tests/qa/test_qap_ms001.py
- Witnesses to re-point: none
- Tests with import lines or patch targets to update: tests/qa/test_qap_ms001.py (import block lines 63-87)
- Narrow tests: tests/qa/test_qap_ms001.py

### U3 · tests/qa/test_qap_ms001.py: move the sensitivity witnesses into their own module
| # | Kind | What | From | To |
|---|---|---|---|---|
| 1 | move | `test_tc108_sensitivity_diff_source_and_normalized_body_distinguish_drift`, `test_tc108_sensitivity_documented_line_pattern_distinguishes_drift`, `test_sensitivity_upstream_oracle_distinguishes_transport_failure_from_a_relayed_body`, `_synthetic_self_contained`, `test_sensitivity_s21_ce1_glued_control_line_is_rejected`, `test_sensitivity_s21_ce2_line_endings_are_line_identity`, `test_sensitivity_s21_ce2_artifact_read_preserves_line_endings`, `test_sensitivity_s21_ce3_renderings_keep_their_trailing_newline`, `test_sensitivity_s22_proxy_post_ignores_ambient_proxy_settings` | tests/qa/test_qap_ms001.py | tests/qa/test_qap_ms001_witnesses.py |
- Importers to update: none (new module)
- Witnesses to re-point: none
- Tests with import lines or patch targets to update: tests/qa/test_qap_ms001_witnesses.py (new import block; imports `no_dump_config_override` by name so the autouse fixture applies to S22, the one process-driving witness)
- Narrow tests: tests/qa/test_qap_ms001_witnesses.py (not selected by the brief's Verify command — the yield must run it explicitly, because `pytest tests/qa/test_qap_ms001.py` no longer reaches the moved witnesses)

### U4 · tests/qa/test_qap_ms001.py: move the fault-isolation scenario and the R1/R2/R5 tiers into their own module
| # | Kind | What | From | To |
|---|---|---|---|---|
| 1 | move | `_read_only_session_dir`, `test_tc001_documented_write_failure_still_returns_the_upstream_response`, `test_tc001_collapsed_flush_failure_still_returns_the_upstream_response`, `UNKNOWN_FORMATS`, `test_hostile_r1_unknown_dump_format_is_refused_and_writes_nothing`, `test_hostile_r1_body_parse_boundaries_on_the_documented_writer`, `test_hostile_r2_documented_writer_does_not_escape_the_dump_root`, `test_r5_near_miss_session_ids_are_not_merged` | tests/qa/test_qap_ms001.py | tests/qa/test_qap_ms001_tiers.py |
- Importers to update: none (new module)
- Witnesses to re-point: none
- Tests with import lines or patch targets to update: tests/qa/test_qap_ms001_tiers.py (new import block; imports `no_dump_config_override` by name)
- Narrow tests: tests/qa/test_qap_ms001_tiers.py (not selected by the brief's Verify command — run it explicitly)

### U5 · tests/qa/qap_ms001_support.py: split the content assertion into its clauses
| # | Kind | What | From | To |
|---|---|---|---|---|
| 1 | reshape | `assert_documented_contents` (called by `test_tc107_no_format_configured_produces_the_documented_layout` and `test_tc108_documented_format_preserves_contents_at_2n_writes`): extract `_assert_turn_zero_body` (the `turn_000.json` clause) and `_assert_index_lines` (the `index.txt` clause); the diff clause stays | tests/qa/qap_ms001_support.py | tests/qa/qap_ms001_support.py |
- Importers to update: none
- Witnesses to re-point: none
- Tests with import lines or patch targets to update: none
- Narrow tests: tests/qa/test_qap_ms001.py (TC-107, TC-108 call it)

### U6 · tests/qa/test_qap_ms001.py: split the documented-format gate's rendering read and write-report assertion
| # | Kind | What | From | To |
|---|---|---|---|---|
| 1 | reshape | `test_tc108_documented_format_preserves_contents_at_2n_writes` (a pytest-collected gate): extract `_read_renderings` and `_assert_2n_write_report` | tests/qa/test_qap_ms001.py | tests/qa/test_qap_ms001.py |
- Importers to update: none
- Witnesses to re-point: none
- Tests with import lines or patch targets to update: none
- Narrow tests: tests/qa/test_qap_ms001.py
- **Dropped: green after U4** — `tests/qa/test_qap_ms001.py` is 267 lines (≤300), so its split unit no longer runs.

## U0 Forward
- Moved: nothing
- Reshaped: nothing
- Files: tests/qa/test_qap_ms001.py, tests/qa/harness.py
- Tests edited: none
- Ran: `.venv/bin/python -m pytest tests/qa/test_qap_ms001.py tests/qa/test_qap_ms002.py tests/qa/test_qap_ms004.py -o addopts=` after each shrink step and once as the brief's `Verify` -> 38 passed, 0 skipped, each time
- Scan: `tests/qa/test_qap_ms001.py` — file 976 lines (threshold 700); 23 issues in 16 functions; top: `test_tc108_documented_format_preserves_contents_at_2n_writes` L:78 N:3 C:6 P:1, `test_tc100_collapsed_format_defers_every_write_until_a_flush` L:53 N:2 C:4 P:1, `assert_documented_contents` L:47 N:2 C:10 P:4, `test_hostile_r2_documented_writer_does_not_escape_the_dump_root` L:41 N:0 C:1 P:1, `test_r5_near_miss_session_ids_are_not_merged` L:39 N:2 C:4. `tests/qa/harness.py` — file 347 lines (review <=300); 9 issues in 7 functions; top: `__init__` L:41 N:1 C:7 P:8, `startup_refused` L:23 N:1 C:2 P:4, `require_upstream` L:11 N:2 C:5, `wait_tcp` L:8 N:2 C:3, `poll_until` L:6 N:2 C:3
- Sizes: tests/qa/test_qap_ms001.py: 976, tests/qa/harness.py: 347
- Noticed: `test_tc108_sensitivity_documented_line_pattern_distinguishes_drift` carried no pre-repair narrative, so seven of the eight named witness docstrings changed and the eighth is unchanged.
- Noticed: the remaining size and scan debt in both files is what the map's later units (U1, U2, U5, U6) already own. Ask: no — the splits are mapped, not missed here.
- Noticed: `tests/qa/harness.py`'s `UPSTREAM` comment restates the §US-009 correction the module docstring now cites. Ask: no — a duplicated comment with no behavioural effect.
- Noticed: the brief's Safety net says `.venv/bin/python` is absent in this worktree, but it exists and ran every command above. Ask: no — environment note only.

## U0 Yield
- Verdict: ACCEPT
- Ran: `.venv/bin/python -m pytest tests/qa/test_qap_ms001.py tests/qa/test_qap_ms002.py tests/qa/test_qap_ms004.py -o addopts=` -> 38 passed, 0 skipped (Safety net: 38 passed)
- Scan: `scan clean` for new findings — `tests/qa/test_qap_ms001.py` 976 lines, 23 issues in 16 functions, same printed functions and metrics as the Safety net (`test_tc108_documented_format_preserves_contents_at_2n_writes` L:78 N:3 C:6 P:1, `test_tc100_collapsed_format_defers_every_write_until_a_flush` L:53 N:2 C:4 P:1, `assert_documented_contents` L:47 N:2 C:10 P:4, `test_hostile_r2_documented_writer_does_not_escape_the_dump_root` L:41 N:0 C:1, `test_r5_near_miss_session_ids_are_not_merged` L:39 N:2 C:4); `tests/qa/harness.py` 347 lines, 9 issues in 7 functions, same printed functions and metrics as the Safety net (`__init__` L:41 N:1 C:7 P:8, `startup_refused` L:23 N:1 C:2 P:4, `require_upstream` L:11 N:2 C:5, `wait_tcp` L:8 N:2 C:3, `poll_until` L:6 N:2 C:3)
- Sizes: tests/qa/test_qap_ms001.py: 976, tests/qa/harness.py: 347
- Findings: none
- Noticed: the new `tests/qa/test_qap_ms001.py` docstring points at the chronicle for "077/096/099/108", but the chronicle records those repairs under `077-forward-qa`, `097-yield` (T021), `100-yield` (T021.001/.002) and `108-forward-qa`; the path holds the substance, only the iteration labels are the brief's shorthand. Ask: no — a doc pointer with no behavioural effect.
- Noticed: `.venv/bin/python` exists in this worktree and ran the Verify command, contradicting the Safety net's note that it is absent. Ask: no — environment note only.

## U1 Forward
- Moved: `require_upstream`, `free_port`, `wait_tcp`, `poll_until`, `Proxy`, `startup_refused`, `REPO_ROOT`, `PROXY_BIN`, `COUNTED_ENTRY`, `TLS_PROBE_ENTRY`, `UPSTREAM` -> tests/qa/harness_process.py (text byte-identical; the new module imports `read_text` and `session_id` from tests.qa.harness)
- Reshaped: `Proxy.__init__`: the counted/probe/plain command selection moved into `_build_command` and `__init__` calls it; L 41 -> 21, C 7 -> 5, N 1, P 8 unchanged. Fix round: `_build_command` had 4 parameters and 21 body lines — two medium factors — so the scan printed it as a new Near Tolerance Length Parameters issue; the entry selection is now `_instrumented_entry(counted_report, probe_report, dump_dir)` (module level, P:3) returning the instrumented entry's argv prefix or None, and `_build_command(extra_args, entry_prefix)` (P:3, L 11) wraps the base args with that prefix. The argv is byte-identical in all three cases, and the scan no longer prints `_build_command`.
- Files: tests/qa/harness_process.py (new), tests/qa/harness.py, tests/qa/test_qap_ms001.py, tests/qa/test_qap_ms002.py, tests/qa/test_qap_ms004.py, invariants/buster_ripper/assumption-claims.md
- Tests edited: tests/qa/test_qap_ms001.py: import block only (`Proxy`, `free_port`, `poll_until`, `startup_refused` now from tests.qa.harness_process); tests/qa/test_qap_ms002.py: import block only (`Proxy`, `poll_until`); tests/qa/test_qap_ms004.py: import line only (`Proxy`, `poll_until`)
- Ran: `.venv/bin/python -m pytest tests/qa/test_qap_ms001.py tests/qa/test_qap_ms002.py tests/qa/test_qap_ms004.py -o addopts=` after the move, after the first reshape, and after the fix round's reshape and `json` removal (which is also the brief's `Verify`) -> 38 passed, 0 skipped, each time
- Scan: `tests/qa/harness.py` — 2 issues in 2 functions: `turn_body` L:6 N:0 C:1 P:4 (Medium Parameters), `rglob_files` L:5 N:2 C:3 P:1 (Medium Nesting); both pre-existing and unmoved. `tests/qa/harness_process.py` — 6 issues in 5 functions: `__init__` L:21 N:1 C:5 P:8 (Medium Length, Too Many Parameters), `startup_refused` L:23 N:1 C:2 P:4 (Near Tolerance Length Parameters), `require_upstream` L:11 N:2 C:5 (Medium Nesting), `wait_tcp` L:8 N:2 C:3 (Medium Nesting), `poll_until` L:6 N:2 C:3 (Medium Nesting); `_instrumented_entry` and `_build_command` are clean. `tests/qa/test_qap_ms001.py` — 978 lines, 23 issues in 16 functions, unchanged. `tests/qa/test_qap_ms002.py` — 634 lines, 15 issues in 14 functions, unchanged. `tests/qa/test_qap_ms004.py` — 477 lines, 7 issues in 7 functions, unchanged
- Sizes: tests/qa/harness.py: 120, tests/qa/harness_process.py: 243, tests/qa/test_qap_ms001.py: 978, tests/qa/test_qap_ms002.py: 634, tests/qa/test_qap_ms004.py: 477
- Noticed: `Proxy.__init__` still prints P:8 (Too Many Parameters); the unit's reshape extracts the command selection only, and the signature is pinned by the map and by every caller. Ask: no — changing the signature would change callers, which the map forbids.
- Noticed: `startup_refused` still prints Near Tolerance Length Parameters (L:23 P:4); it moved byte-identically from `tests/qa/harness.py`, where the Safety net's scan already printed the same issue, and the unit has no reshape row for it. Ask: no — pre-existing debt, not created here.
- Noticed: `tests/qa/harness.py`'s module docstring still describes `Proxy(..., upstream=..., require_live_upstream=False)` and the subprocess contract, though `Proxy` now lives in `tests/qa/harness_process.py`. Ask: no — a doc pointer with no behavioural effect, and U0 already shrank that docstring.
- Noticed: the witness re-point changed only the path on `invariants/buster_ripper/assumption-claims.md:2655,2656,2660`, so their cited line numbers ("line 129", "lines 29-31", "111-129") now describe the pre-move layout of `tests/qa/harness.py`. Ask: no — stale citation numbers inside a witness cell, no behavioural effect; the map said change the path, nothing else.
- Noticed: the remaining scan debt in `tests/qa/test_qap_ms001.py`, `tests/qa/test_qap_ms002.py` and `tests/qa/test_qap_ms004.py` is what U2-U6 already own. Ask: no — the splits are mapped, not missed here.

## U1 Yield
- Verdict: ACCEPT
- Ran: `.venv/bin/python -m pytest tests/qa/test_qap_ms001.py tests/qa/test_qap_ms002.py tests/qa/test_qap_ms004.py -o addopts=` -> 38 passed, 0 skipped (Safety net: 38 passed)
- Scan: `tests/qa/harness.py` — 2 issues in 2 functions: `turn_body` L:6 N:0 C:1 P:4 (Medium Parameters), `rglob_files` L:5 N:2 C:3 P:1 (Medium Nesting); both in the Safety net's 7-function set, unmoved, metrics unchanged. `tests/qa/harness_process.py` — 6 issues in 5 functions: `__init__` L:21 N:1 C:5 P:8 (Medium Length, Too Many Parameters; was L:41 C:7 plus Too Many Parameters in the Safety net, so L and C improved), `startup_refused` L:23 N:1 C:2 P:4, `require_upstream` L:11 N:2 C:5, `wait_tcp` L:8 N:2 C:3, `poll_until` L:6 N:2 C:3; `_instrumented_entry` and `_build_command` are clean, so no new function is printed and the prior round's `_build_command` finding is gone. `tests/qa/test_qap_ms001.py` — 978 lines, 23 issues in 16 functions, unchanged. `tests/qa/test_qap_ms002.py` — 634 lines, 15 issues in 14 functions, unchanged. `tests/qa/test_qap_ms004.py` — 477 lines, 7 issues in 7 functions, unchanged
- Sizes: tests/qa/harness.py: 120, tests/qa/harness_process.py: 243, tests/qa/test_qap_ms001.py: 978, tests/qa/test_qap_ms002.py: 634, tests/qa/test_qap_ms004.py: 477
- Findings: none
- Noticed: the reshape extracted `_instrumented_entry` beside the map's `_build_command`; both sit inside the row's "counted/probe/plain command selection", and the scan prints neither. Ask: no — structure-only, no behaviour change.
- Noticed: `Proxy`'s argv is byte-identical to the pre-move `Proxy` for plain, extra-args, counted, counted-plus-extra and probe selection (executed both versions with `Popen` and `wait_tcp` stubbed and compared the lists); `__init__`'s signature and every caller are unchanged. Ask: no — verified equivalence, nothing to decide.
- Noticed: `tests/qa/harness.py:15` still describes `Proxy(..., upstream=..., require_live_upstream=False)` and the subprocess contract after the move; a doc pointer with no behavioural effect, and U0 already shrank that docstring. Ask: no.
- Noticed: the witness re-point changed only the path on `invariants/buster_ripper/assumption-claims.md:2655,2656,2660`, so "all 22 Python modules" is now 23 and the cited line numbers ("line 129", "lines 29-31", "111-129") still describe the pre-move layout; the map said path only and the aggregate fingerprint is the registry lane's. Ask: no — stale counts inside a witness cell, no behavioural effect.
- Noticed: `tests/qa/test_qap_ms001.py` is 978 lines against the brief's 700 threshold; U2-U6 own the split. Ask: no — mapped, not missed here.

## U2 Forward
- Moved: `documented_files`, `README_LINE_PATTERN`, `FULL_CLAUSE`, `KB_CLAUSE`, `CHANGE_CLAUSE`, `MSGS_CLAUSE`, `TOOLS_CLAUSE`, `index_lines`, `production_diff_text`, `changed_lines`, `diff_changed_lines`, `_content_lines`, `assert_wellformed_diff`, `rendering_section`, `assert_documented_contents`, `count_report`, `ops_by_name`, `marker`, `nonce`, `no_dump_config_override`, `COLLAPSED` -> tests/qa/qap_ms001_support.py (byte-identical; compared the new module body against the source line ranges)
- Moved: `listing` -> tests/qa/harness.py (byte-identical, placed after `log_artifacts`)
- Reshaped: nothing
- Files: tests/qa/qap_ms001_support.py (new), tests/qa/test_qap_ms001.py, tests/qa/harness.py
- Tests edited: tests/qa/test_qap_ms001.py: import block only — `re` dropped (its last use moved), `record_section` dropped, `listing` added from tests.qa.harness, and `CHANGE_CLAUSE`, `COLLAPSED`, `FULL_CLAUSE`, `KB_CLAUSE`, `MSGS_CLAUSE`, `README_LINE_PATTERN`, `TOOLS_CLAUSE`, `assert_documented_contents`, `assert_wellformed_diff`, `changed_lines`, `count_report`, `diff_changed_lines`, `documented_files`, `index_lines`, `marker`, `no_dump_config_override`, `nonce`, `ops_by_name`, `production_diff_text`, `rendering_section` added from tests.qa.qap_ms001_support (the six line-pattern names stay imported because the sensitivity witnesses U3 moves still read them)
- Ran: `.venv/bin/python -m pytest tests/qa/test_qap_ms001.py -o addopts=` after the helper move -> 17 passed; after the `listing` move -> 17 passed; then the brief's `Verify` once -> 38 passed, 0 skipped
- Scan: `tests/qa/test_qap_ms001.py` — file 793 lines (threshold 700); 19 issues in 15 functions, top `test_tc108_documented_format_preserves_contents_at_2n_writes` L:78 N:3 C:6 P:1 (Long Function, Deep Nesting, Medium Complexity), `test_tc100_collapsed_format_defers_every_write_until_a_flush` L:53 N:2 C:4 P:1, `test_hostile_r2_documented_writer_does_not_escape_the_dump_root` L:41 N:0 C:1, `test_r5_near_miss_session_ids_are_not_merged` L:39 N:2 C:4, `test_sensitivity_s21_ce2_artifact_read_preserves_line_endings` L:31 N:0 C:1 — 19+4 = the Safety net's 23 issues and 15+1 = its 16 functions, same printed functions and metrics. `tests/qa/qap_ms001_support.py` — 4 issues in 1 function: `assert_documented_contents` L:47 N:2 C:10 P:4 (Long Function, Medium Nesting, Medium Complexity, Medium Parameters), identical to the Safety net. `tests/qa/harness.py` — 2 issues in 2 functions: `turn_body` L:6 N:0 C:1 P:4 (Medium Parameters), `rglob_files` L:5 N:2 C:3 (Medium Nesting); both in the Safety net's set, unmoved. `listing` is not printed
- Sizes: tests/qa/test_qap_ms001.py: 793, tests/qa/qap_ms001_support.py: 213, tests/qa/harness.py: 124
- Noticed: `assert_documented_contents` still prints Long Function, Medium Nesting, Medium Complexity and Medium Parameters with the Safety net's exact metrics; the unit has no reshape row and U5 owns it. Ask: no — pre-existing debt, mapped, not created here.
- Noticed: `tests/qa/test_qap_ms001.py` is 793 lines against the brief's 700 threshold; U3, U4 and U6 own the remaining split. Ask: no — mapped, not missed here.
- Noticed: the autouse fixture now lives in a non-conftest module, so it applies to `test_qap_ms001.py` only because that module imports it by name; `--setup-plan` lists `no_dump_config_override` for all 17 collected tests. Ask: no — verified, no behavioural change.
- Noticed: `tests/qa/harness.py`'s module docstring still describes only the process driver and never mentions `listing`, which now lives there. Ask: no — a doc pointer with no behavioural effect, and no doc change was mapped.
- Noticed: the `# helpers` section separator at the old helper block's head was dropped with the block; every scenario in `tests/qa/test_qap_ms001.py` keeps its own `# ----` separator. Ask: no — layout only.

## U2 Yield
- Verdict: ACCEPT
- Ran: `.venv/bin/python -m pytest tests/qa/test_qap_ms001.py tests/qa/test_qap_ms002.py tests/qa/test_qap_ms004.py -o addopts=` -> 38 passed, 0 skipped (Safety net: 38 passed)
- Scan: `tests/qa/test_qap_ms001.py` — 793 lines, 19 issues in 15 functions: `test_tc108_documented_format_preserves_contents_at_2n_writes` L:78 N:3 C:6 P:1, `test_tc100_collapsed_format_defers_every_write_until_a_flush` L:53 N:2 C:4 P:1, `test_hostile_r2_documented_writer_does_not_escape_the_dump_root` L:41 N:0 C:1, `test_r5_near_miss_session_ids_are_not_merged` L:39 N:2 C:4, `test_sensitivity_s21_ce2_artifact_read_preserves_line_endings` L:31 N:0 C:1; every printed function and metric is the Safety net's, whose remaining 11 functions have byte-identical bodies (AST-compared against HEAD). `tests/qa/qap_ms001_support.py` — 4 issues in 1 function: `assert_documented_contents` L:47 N:2 C:10 P:4, the Safety net's own metrics; no other function printed. `tests/qa/harness.py` — 2 issues in 2 functions: `turn_body` L:6 N:0 C:1 P:4, `rglob_files` L:5 N:2 C:3; both in the Safety net's 7-function set, `listing` not printed. 19+4 = the Safety net's 23 issues, 15+1 = its 16 functions
- Sizes: tests/qa/test_qap_ms001.py: 793, tests/qa/qap_ms001_support.py: 213, tests/qa/harness.py: 124
- Findings: none
- Noticed: the old helper block's three-line `# helpers` separator does not reappear in the support module; layout only, and every scenario keeps its own `# ----` separator. Ask: no — no behavioural effect.
- Noticed: `tests/qa/test_qap_ms001.py` is 793 lines against the brief's 700 threshold; U3, U4 and U6 own the remaining split. Ask: no — mapped, not missed here.
- Noticed: `assert_documented_contents` still prints Long Function, Medium Nesting, Medium Complexity and Medium Parameters with the Safety net's exact metrics; U5 owns the reshape. Ask: no — pre-existing debt, mapped, not created here.
- Noticed: the autouse `no_dump_config_override` now lives in a non-conftest module and reaches the file's tests only through the explicit import; `--setup-plan` lists it for all 17 collected tests. Ask: no — verified, no behavioural change.
- Noticed: `tests/qa/harness.py`'s module docstring still describes only the process driver and never mentions `listing`, which now lives there. Ask: no — a doc pointer with no behavioural effect, and no doc change was mapped.

## U3 Forward
- Moved: `test_tc108_sensitivity_diff_source_and_normalized_body_distinguish_drift`, `test_tc108_sensitivity_documented_line_pattern_distinguishes_drift`, `test_sensitivity_upstream_oracle_distinguishes_transport_failure_from_a_relayed_body`, `_synthetic_self_contained`, `test_sensitivity_s21_ce1_glued_control_line_is_rejected`, `test_sensitivity_s21_ce2_line_endings_are_line_identity`, `test_sensitivity_s21_ce2_artifact_read_preserves_line_endings`, `test_sensitivity_s21_ce3_renderings_keep_their_trailing_newline`, `test_sensitivity_s22_proxy_post_ignores_ambient_proxy_settings` -> tests/qa/test_qap_ms001_witnesses.py (byte-identical; diffed the extracted block against the source range)
- Reshaped: nothing
- Files: tests/qa/test_qap_ms001_witnesses.py (new), tests/qa/test_qap_ms001.py
- Tests edited: tests/qa/test_qap_ms001.py: import block only (`difflib`, `urllib.request`, `pytest`, `billing_system`, `free_port` and `CHANGE_CLAUSE`, `FULL_CLAUSE`, `KB_CLAUSE`, `MSGS_CLAUSE`, `README_LINE_PATTERN`, `TOOLS_CLAUSE`, `assert_wellformed_diff`, `changed_lines`, `diff_changed_lines`, `production_diff_text` dropped — each was the moved block's last use); tests/qa/test_qap_ms001_witnesses.py: new import block only, including `no_dump_config_override` by name and `N_TURNS` from `tests.qa.test_qap_ms001`
- Ran: `.venv/bin/python -m pytest tests/qa/test_qap_ms001_witnesses.py -o addopts=` after the move -> 8 passed; `.venv/bin/python -m pytest tests/qa/test_qap_ms001.py -o addopts=` -> 9 passed; the two files together -> 17 passed, no warnings (no double collection, gate module rewritten once); the brief's `Verify` once -> 30 passed, 0 skipped (Safety net: 38 — the 8 moved witnesses are no longer selected by `tests/qa/test_qap_ms001.py`)
- Scan: `tests/qa/test_qap_ms001.py` — file 517 lines (ideal <=500); 13 issues in 9 functions: `test_tc108_documented_format_preserves_contents_at_2n_writes` L:78 N:3 C:6 P:1 (Long Function, Deep Nesting, Medium Complexity), `test_tc100_collapsed_format_defers_every_write_until_a_flush` L:53 N:2 C:4 P:1, `test_hostile_r2_documented_writer_does_not_escape_the_dump_root` L:41 N:0 C:1 P:1, `test_r5_near_miss_session_ids_are_not_merged` L:39 N:2 C:4 P:1, `test_hostile_r1_body_parse_boundaries_on_the_documented_writer` L:30 N:1 C:2 P:1. `tests/qa/test_qap_ms001_witnesses.py` — 6 issues in 6 functions: `test_sensitivity_s21_ce2_artifact_read_preserves_line_endings` L:31 N:0 C:1 P:1 (Long Function), `test_tc108_sensitivity_diff_source_and_normalized_body_distinguish_drift` L:26 N:0 C:4 P:0 (Medium Length), `test_tc108_sensitivity_documented_line_pattern_distinguishes_drift` L:25 N:1 C:9 P:0 (Near Tolerance Complexity Length), `test_sensitivity_s21_ce1_glued_control_line_is_rejected` L:19 N:1 C:1 P:0 (Medium Length), `test_sensitivity_s22_proxy_post_ignores_ambient_proxy_settings` L:18 N:0 C:1 P:2 (Medium Length). 13+6 = the Safety net's 19 issues and 9+6 = its 15 functions, every printed metric the moved functions already carried
- Sizes: tests/qa/test_qap_ms001.py: 517, tests/qa/test_qap_ms001_witnesses.py: 297
- Noticed: `N_TURNS` has no home outside `tests/qa/test_qap_ms001.py`, so the new module imports it from there; the moved function text is unchanged and the constant stays single-sourced. Ask: no — a later unit that moves the constant changes one import line.
- Noticed: `tests/qa/test_qap_ms001.py` is 517 lines, over the scan's 500-line ideal but under the brief's 700 threshold; U4 and U6 own the remaining split. Ask: no — mapped, not missed here.
- Noticed: `import pytest` was dropped from `tests/qa/test_qap_ms001.py` because the moved witnesses held its last use; import line only, no test body touched. Ask: no — no behavioural effect.
- Noticed: the new module imports the gate module for `N_TURNS`, so `pytest tests/qa/test_qap_ms001.py tests/qa/test_qap_ms001_witnesses.py` imports it once; both files together collect 17 tests with no rewrite warning. Ask: no — verified, no behavioural change.
- Noticed: the moved witnesses print the same six scan issues with the Safety net's metrics; the move created none. Ask: no — pre-existing debt, moved text.
- Noticed: `tests/qa/test_qap_ms001_witnesses.py`'s one-line docstring names no re-entry repairs or `qa-plan.md` sections the witnesses rely on; the map says one line, and the shrunk docstrings of the moved functions still cite their sources. Ask: no — a doc pointer with no behavioural effect.

## U3 Yield
- Verdict: ACCEPT
- Ran: `.venv/bin/python -m pytest tests/qa/test_qap_ms001.py tests/qa/test_qap_ms002.py tests/qa/test_qap_ms004.py -o addopts=` -> 30 passed, 0 skipped; `.venv/bin/python -m pytest tests/qa/test_qap_ms001_witnesses.py -o addopts=` -> 8 passed; 30 + 8 = the Safety net's 38 passed (the 8 moved witnesses are no longer selected by the Verify command, as the unit's Narrow tests row says)
- Scan: `tests/qa/test_qap_ms001.py` — 517 lines; 13 issues in 9 functions: `test_tc108_documented_format_preserves_contents_at_2n_writes` L:78 N:3 C:6 P:1 (Long Function, Deep Nesting, Medium Complexity), `test_tc100_collapsed_format_defers_every_write_until_a_flush` L:53 N:2 C:4 P:1, `test_hostile_r2_documented_writer_does_not_escape_the_dump_root` L:41 N:0 C:1 P:1, `test_r5_near_miss_session_ids_are_not_merged` L:39 N:2 C:4 P:1, `test_hostile_r1_body_parse_boundaries_on_the_documented_writer` L:30 N:1 C:2 P:1. `tests/qa/test_qap_ms001_witnesses.py` — 297 lines; 6 issues in 6 functions: `test_sensitivity_s21_ce2_artifact_read_preserves_line_endings` L:31 N:0 C:1 P:1 (Long Function), `test_tc108_sensitivity_diff_source_and_normalized_body_distinguish_drift` L:26 N:0 C:4 P:0 (Medium Length), `test_tc108_sensitivity_documented_line_pattern_distinguishes_drift` L:25 N:1 C:9 P:0 (Near Tolerance Complexity Length), `test_sensitivity_s21_ce1_glued_control_line_is_rejected` L:19 N:1 C:1 P:0 (Medium Length), `test_sensitivity_s22_proxy_post_ignores_ambient_proxy_settings` L:18 N:0 C:1 P:2 (Medium Length); every printed function is a moved one, and the nine moved bodies are AST-identical to HEAD, so no metric moved. 13+6+the untouched `tests/qa/qap_ms001_support.py` 4 issues in 1 function = the Safety net's 23 issues in 16 functions; no function is printed that its scan did not print
- Sizes: tests/qa/test_qap_ms001.py: 517, tests/qa/test_qap_ms001_witnesses.py: 297
- Findings: none
- Noticed: the new module imports `N_TURNS` from `tests.qa.test_qap_ms001` because the constant has no other home; the moved text is unchanged and both files together collect 17 tests with no double collection. Ask: no — a later unit that moves the constant changes one import line.
- Noticed: `tests/qa/test_qap_ms001.py` is 517 lines, over the scan's 500 ideal but under the brief's 700 threshold; U4 and U6 own the remaining split. Ask: no — mapped, not missed here.
- Noticed: `import pytest` was dropped from `tests/qa/test_qap_ms001.py` because the moved witnesses held its last use; the diff adds no line at all to that file. Ask: no — import line only, no behavioural effect.
- Noticed: `no_dump_config_override` is imported by name but used as an autouse fixture, not as a symbol, in both modules; `--setup-plan` lists it for all 8 witness tests including the process-driving S22. Ask: no — verified, no behavioural change.
- Noticed: the moved witnesses print the same six scan issues with the Safety net's metrics; the move created none. Ask: no — pre-existing debt, moved text.
- Noticed: `tests/qa/test_qap_ms001_witnesses.py`'s one-line docstring names no re-entry repairs or `qa-plan.md` sections the witnesses rely on; the map says one line, and the shrunk docstrings of the moved functions still cite their sources. Ask: no — a doc pointer with no behavioural effect.

## U4 Forward
- Moved: `_read_only_session_dir`, `test_tc001_documented_write_failure_still_returns_the_upstream_response`, `test_tc001_collapsed_flush_failure_still_returns_the_upstream_response`, `UNKNOWN_FORMATS`, `test_hostile_r1_unknown_dump_format_is_refused_and_writes_nothing`, `test_hostile_r1_body_parse_boundaries_on_the_documented_writer`, `test_hostile_r2_documented_writer_does_not_escape_the_dump_root`, `test_r5_near_miss_session_ids_are_not_merged` and the four `# ----` section headers that head them -> tests/qa/test_qap_ms001_tiers.py (byte-identical; compared the new module's body against the source line range 274-517 of HEAD)
- Reshaped: nothing
- Files: tests/qa/test_qap_ms001_tiers.py (new), tests/qa/test_qap_ms001.py
- Tests edited: tests/qa/test_qap_ms001_tiers.py: new import block only, including `no_dump_config_override` by name; tests/qa/test_qap_ms001.py: import block only (`json`, `poll_until`, `startup_refused`, `index_lines` dropped — each was the moved block's last use)
- Ran: `.venv/bin/python -m pytest tests/qa/test_qap_ms001_tiers.py -o addopts=` after the move -> 1 collection error (`documented_files` and `index_lines` were imported from tests.qa.harness instead of tests.qa.qap_ms001_support); after the import fix -> 6 passed; `.venv/bin/python -m pytest tests/qa/test_qap_ms001.py -o addopts=` -> 3 passed; the three ms001 modules together -> 17 passed, no warnings; `--setup-plan` on the tiers module lists `no_dump_config_override` for all 6 tests; the brief's `Verify` once -> 24 passed, 0 skipped (Safety net: 38; U3's 8 witnesses and this unit's 6 tiers are no longer selected by the Verify command)
- Scan: `tests/qa/test_qap_ms001.py` — 267 lines; 6 issues in 3 functions: `test_tc108_documented_format_preserves_contents_at_2n_writes` L:78 N:3 C:6 P:1 (Long Function, Deep Nesting, Medium Complexity), `test_tc100_collapsed_format_defers_every_write_until_a_flush` L:53 N:2 C:4 P:1 (Long Function, Medium Nesting), `test_tc107_no_format_configured_produces_the_documented_layout` L:29 N:2 C:4 P:1 (Near Tolerance Nesting Length). `tests/qa/test_qap_ms001_tiers.py` — 274 lines; 7 issues in 6 functions: `test_hostile_r2_documented_writer_does_not_escape_the_dump_root` L:41 N:0 C:1 P:1 (Long Function), `test_r5_near_miss_session_ids_are_not_merged` L:39 N:2 C:4 P:1 (Long Function, Medium Nesting), `test_hostile_r1_body_parse_boundaries_on_the_documented_writer` L:30 N:1 C:2 P:1 (Medium Length), `test_tc001_documented_write_failure_still_returns_the_upstream_response` L:23 N:0 C:1 P:1 (Medium Length), `test_tc001_collapsed_flush_failure_still_returns_the_upstream_response` L:16 N:0 C:1 P:1 (Medium Length), `test_hostile_r1_unknown_dump_format_is_refused_and_writes_nothing` L:12 N:2 C:7 P:1 (Near Tolerance Complexity Nesting). 6+7 = the U3 scan's 13 issues in 3+6 = its 9 functions, every printed function a moved one with its pre-move metrics; `_read_only_session_dir` is not printed
- Sizes: tests/qa/test_qap_ms001.py: 267, tests/qa/test_qap_ms001_tiers.py: 274
- Noticed: `tests/qa/test_qap_ms001.py`'s module docstring still lists TC-001, the hostile tier and the identity tier, which now live in `tests/qa/test_qap_ms001_tiers.py`. Ask: no — a doc pointer with no behavioural effect, and no doc change is mapped.
- Noticed: `test_hostile_r1_unknown_dump_format_is_refused_and_writes_nothing` prints Near Tolerance Complexity Nesting (L:12 N:2 C:7 P:1); it moved byte-identically and the unit has no reshape row. Ask: no — pre-existing debt, moved text.
- Noticed: `no_dump_config_override` is imported by name but used as an autouse fixture, not as a symbol; `--setup-plan` lists it for all 6 tier tests, including the five that drive a process. Ask: no — verified, no behavioural change.
- Noticed: the four `# ----` section headers moved with their functions, so no orphan header is left at the end of `tests/qa/test_qap_ms001.py`. Ask: no — layout only.

## U4 Yield
- Verdict: ACCEPT
- Ran: `.venv/bin/python -m pytest tests/qa/test_qap_ms001.py tests/qa/test_qap_ms002.py tests/qa/test_qap_ms004.py -o addopts=` -> 24 passed, 0 skipped; `.venv/bin/python -m pytest tests/qa/test_qap_ms001_tiers.py -o addopts=` -> 6 passed (the unit's Narrow tests row); `.venv/bin/python -m pytest tests/qa/test_qap_ms001_witnesses.py -o addopts=` -> 8 passed; 24 + 6 + 8 = the Safety net's 38 passed. The three ms001 modules together -> 17 passed, no double collection
- Scan: `tests/qa/test_qap_ms001.py` — 6 issues in 3 functions: `test_tc108_documented_format_preserves_contents_at_2n_writes` L:78 N:3 C:6 P:1 (Long Function, Deep Nesting, Medium Complexity), `test_tc100_collapsed_format_defers_every_write_until_a_flush` L:53 N:2 C:4 P:1 (Long Function, Medium Nesting), `test_tc107_no_format_configured_produces_the_documented_layout` L:29 N:2 C:4 P:1 (Near Tolerance Nesting Length). `tests/qa/test_qap_ms001_tiers.py` — 7 issues in 6 functions: `test_hostile_r2_documented_writer_does_not_escape_the_dump_root` L:41 N:0 C:1 P:1 (Long Function), `test_r5_near_miss_session_ids_are_not_merged` L:39 N:2 C:4 P:1 (Long Function, Medium Nesting), `test_hostile_r1_body_parse_boundaries_on_the_documented_writer` L:30 N:1 C:2 P:1 (Medium Length), `test_tc001_documented_write_failure_still_returns_the_upstream_response` L:23 N:0 C:1 P:1 (Medium Length), `test_tc001_collapsed_flush_failure_still_returns_the_upstream_response` L:16 N:0 C:1 P:1 (Medium Length), `test_hostile_r1_unknown_dump_format_is_refused_and_writes_nothing` L:12 N:2 C:7 P:1 (Near Tolerance Complexity Nesting); `_read_only_session_dir` is not printed. 6+7 = the U3 scan's 13 issues in 3+6 = its 9 functions, every printed function a moved one with its pre-move metrics (all nine bodies byte-identical to HEAD)
- Sizes: tests/qa/test_qap_ms001.py: 267, tests/qa/test_qap_ms001_tiers.py: 274
- Findings: none
- Noticed: `tests/qa/test_qap_ms001.py`'s module docstring still lists TC-001, the hostile tier and the identity tier, which now live in `tests/qa/test_qap_ms001_tiers.py`. Ask: no — a doc pointer with no behavioural effect, and no doc change is mapped.
- Noticed: `test_hostile_r1_unknown_dump_format_is_refused_and_writes_nothing` prints Near Tolerance Complexity Nesting (L:12 N:2 C:7 P:1); it moved byte-identically and the unit has no reshape row. Ask: no — pre-existing debt, moved text.
- Noticed: `no_dump_config_override` is imported by name but used as an autouse fixture, not as a symbol; `--setup-plan` lists it for all 6 tier tests, including the five that drive a process. Ask: no — verified, no behavioural change.
- Noticed: `tests/qa/test_qap_ms001.py` keeps a single-name parenthesized import block for `Proxy` after `poll_until`/`startup_refused` were dropped; style only. Ask: no — import line only, no behavioural effect.
- Noticed: the four `# ----` section headers moved with their functions, so no orphan header is left at the end of `tests/qa/test_qap_ms001.py`. Ask: no — layout only.

## U5 Forward
- Moved: nothing
- Reshaped: `assert_documented_contents`: its `turn_000.json` clause became `_assert_turn_zero_body(sid_dir, markers, expected_body)` and its `index.txt` clause became `_assert_index_lines(sid_dir, markers)`, called from the parent in the original order; the `turn_NNN.diff` clause and the signature stay. The index clause taken whole printed as a new Near Tolerance Nesting Length issue (L:20 N:2), so it is itself split — `_assert_index_line_shape(i, line)` (reader pattern, turn number, msgs/tools columns) and `_assert_turn_columns(i, line)` (turn 000 FULL + size, later +A/-R) are called in that order per line — until every extracted function is clean. Assertion text, messages and order are byte-identical; `assert_documented_contents` L 47 -> 18, C 10 -> 5, N 2 and P 4 unchanged
- Files: tests/qa/qap_ms001_support.py
- Tests edited: none
- Ran: `.venv/bin/python -m pytest tests/qa/test_qap_ms001.py -o addopts=` before the step -> 3 passed; after the two named extractions -> 3 passed; after the index-clause split -> 3 passed; the brief's `Verify` once -> 24 passed, 0 skipped (Safety net: 38; U3's 8 witnesses and U4's 6 tiers are not selected by the Verify command); `tests/qa/test_qap_ms001_witnesses.py` + `tests/qa/test_qap_ms001_tiers.py` -> 14 passed
- Scan: `tests/qa/qap_ms001_support.py` — 3 issues in 1 function: `assert_documented_contents` L:18 N:2 C:5 P:4 (Medium Length, Medium Nesting, Medium Parameters), the Safety net's own function with L 47 -> 18 and C 10 -> 5, N and P unchanged; `_assert_turn_zero_body`, `_assert_index_line_shape`, `_assert_turn_columns` and `_assert_index_lines` are clean, so no function is printed that the Safety net did not print
- Sizes: tests/qa/qap_ms001_support.py: 230
- Noticed: `assert_documented_contents` still prints Medium Length, Medium Nesting and Medium Parameters; N:2 (the diff loop plus `renderings is not None`) and P:4 (its signature, pinned by TC-107 and TC-108) are the row's pinned shape, and the row says the diff clause stays. Ask: no — the row names the two extractions, and both factors it could move did move.
- Noticed: `_assert_index_line_shape` and `_assert_turn_columns` are two helpers the row does not name; they are the index clause split so the extracted helper does not carry the parent's near-tolerance length and nesting. Ask: no — structure only, no behaviour change.
- Noticed: the three extracted helpers carry their clause comments but no docstrings, like the module's existing bare helpers (`index_lines`, `marker`, `count_report`). Ask: no — no behavioural effect, and no doc change is mapped.
- Noticed: the Safety net records `assert_documented_contents` in `tests/qa/test_qap_ms001.py` (L:47 N:2 C:10 P:4); U2 moved it to `tests/qa/qap_ms001_support.py` and this unit reshaped it there, so the file in that row is its pre-move home. Ask: no — the same function, already re-homed.

## U5 Yield
- Verdict: ACCEPT
- Ran: `.venv/bin/python -m pytest tests/qa/test_qap_ms001.py tests/qa/test_qap_ms002.py tests/qa/test_qap_ms004.py -o addopts=` -> 24 passed, 0 skipped; `.venv/bin/python -m pytest tests/qa/test_qap_ms001_witnesses.py tests/qa/test_qap_ms001_tiers.py -o addopts=` -> 14 passed; 24 + 14 = the Safety net's 38 passed (the 14 moved tests are no longer selected by the Verify command, as the U3/U4 rows say)
- Scan: `tests/qa/qap_ms001_support.py` — 3 issues in 1 function: `assert_documented_contents` L:18 N:2 C:5 P:4 (Medium Length, Medium Nesting, Medium Parameters), the Safety net's own function with L 47 -> 18 and C 10 -> 5, N and P unchanged; `_assert_turn_zero_body`, `_assert_index_line_shape`, `_assert_turn_columns` and `_assert_index_lines` are not printed, so no function is printed that the Safety net's scan did not print. The scan printed no file-size line, so the file is under its thresholds
- Sizes: tests/qa/qap_ms001_support.py: 230
- Findings: none
- Noticed: `_assert_index_line_shape` and `_assert_turn_columns` are two helpers the row does not name; they hold the index clause's own statements, unchanged, and are what keeps the named `_assert_index_lines` off the scan. Ask: no — structure only, no behaviour change.
- Noticed: `assert_documented_contents` still prints Medium Length, Medium Nesting and Medium Parameters; N:2 (the diff loop plus `renderings is not None`) and P:4 (its signature, pinned by TC-107 and TC-108) are the row's pinned shape, and the row says the diff clause stays. Ask: no — pre-existing debt the row leaves in place.
- Noticed: the public entry point now sits below its four helpers, so the file opens with private clauses; layout only. Ask: no — no behavioural effect.
- Noticed: the old function's clause comments moved with their clauses and the parent docstring stayed with the parent, so no comment or docstring line was lost. Ask: no — verified, no behavioural effect.

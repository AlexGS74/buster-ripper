# Split the two free QA gate modules

- Goal: split `tests/qa/test_qap_ms002.py` and `tests/qa/test_qap_ms004.py` along their scenario/helper boundaries, with behavior unchanged.
- Target: open (global roadmap); chosen candidates #4/#5; no roadmap row.
- In scope: `tests/qa/test_qap_ms002.py`, `tests/qa/test_qap_ms004.py`
- Importers: `tests/qa/test_qap_ms002.py`, `tests/qa/test_qap_ms002_hostile.py`, `tests/qa/test_qap_ms002_flush.py`, `tests/qa/qap_ms002_support.py`, `tests/qa/test_qap_ms004.py`, `tests/qa/test_qap_ms004_adversarial.py`, `tests/qa/qap_ms004_support.py`
- Out of scope: the taken roadmap rows `RM-001`, `RM-002` and `RM-025` (`origin/fix/tool-reference-scrub`, stale since 2026-09-17); no other QA gate.
- Verify: `.venv/bin/python -m pytest tests/qa/test_qap_ms002.py tests/qa/test_qap_ms004.py -o addopts=`
- Final verify: `uv run pytest`
- Worktree: /Users/alexhubitski/Worktrees/buster-ripper/refactor-qa-gates

## Collected
| # | Candidate | Why | Recorded as | State |
|---|---|---|---|---|
| 1 | buster_ripper/routes/messages.py + buster_ripper/routes/chat.py | `messages.py — File is 341 lines (review: <=300)`; `handle_messages` L:28 N:2 C:12 (High Complexity), `_non_stream_single_request` L:34 N:4 C:8 P:4 (Long Function, Deep Nesting), `_stream_single_request` L:35 N:2 C:4 P:4 (Long Function); `chat.py` 14 issues in 8 functions, `handle_chat` L:26 N:2 C:6; ahead of RM-010 (`messages.py:289-334`, `chat.py:134-152`), RM-011 (`:156,299` / `:143`), RM-012 | RM-001 (`QB:routes-messages/handler-lncp`) | taken by origin/fix/tool-reference-scrub (stale since 2026-09-17) |
| 2 | buster_ripper/dump.py | `File is 319 lines (review: <=300)`; `_render_body` L:12 N:1 C:6 (Medium Complexity); ahead of RM-013 (`:174,186,196`), RM-014, RM-015, RM-022 (`:147-158`, `:295-305`), RM-023 (`:195-200`, `:300-307`) | RM-002 (`QB:dump/file-size-yellow`) | taken by origin/fix/tool-reference-scrub (stale since 2026-09-17) |
| 3 | tests/test_compaction.py | `File is 451 lines (review: <=300)`; `Summary: 2 total issues found in 2 functions` — Medium Length: 2; `test_poll_kv_cache_updates_global_cache` L:20 N:0 C:2 P:1; the row's own drift-helper split remedy | RM-025 (`QB:test-compaction/file-size-yellow`) | taken by origin/fix/tool-reference-scrub (stale since 2026-09-17) |
| 4 | tests/qa/test_qap_ms002.py | `File is 634 lines (ideal: <=500)`; `Summary: 15 total issues found in 14 functions` — Long Function: 4, Medium Length: 10, Medium Complexity: 1; top `test_tc102_103_104_collapsed_artifact_shape_readability_turn0` L:46 N:1 C:5 P:2 | none | free |
| 5 | tests/qa/test_qap_ms004.py | `File is 477 lines (review: <=300)`; `Noise decomposition: 42 trimmable full-line comments`; `Summary: 7 total issues found in 7 functions` — Long Function: 4, Medium Length: 3; `test_tc114_one_posture_governs_every_outbound_call` L:71 N:1 C:4 | none | free |
- Tool gaps: `quality-backlog.md` is absent from the tree (deleted by the 02.00 squash `3323551`), so its `## Refactoring` section was read from `git show 3323551^:.sessions/codify/02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/quality-backlog.md`; every open refactoring line there already routes to an `RM-*` handle. No CODIFY session at HEAD whose Progress Overview shows implementation not started carries an Anchor Map, so no `planned:` files could be derived. The repo-root scan also flags `buster_ripper_legacy.py` (1017 lines, threshold 700), the tracked base-parity reference copy outside the `buster_ripper/` source root; it is taken by the same stale branch and is not offered.

## Safety net
- Ran: `.venv/bin/python -m pytest tests/qa/test_qap_ms002.py tests/qa/test_qap_ms004.py -o addopts=` -> 21 passed, 0 skipped (the worktree had no `.venv`; `uv run python -c ...` built it once, then the brief's exact command ran unchanged)
- Scan: `codify scan path tests/qa/test_qap_ms002.py tests/qa/test_qap_ms004.py` -> `tests/qa/test_qap_ms002.py` — file 634 lines (ideal <=500); 15 issues in 14 functions: Long Function 4, Medium Length 10, Medium Complexity 1; top: `test_tc102_103_104_collapsed_artifact_shape_readability_turn0` L:46 N:1 C:5 P:2 (Long Function), `test_tc112_timer_resilience` L:44 N:1 C:4 P:1 (Long Function), `test_tc101_session_isolation_near_miss_prompts` L:38 N:0 C:6 P:1 (Long Function, Medium Complexity), `test_tc105_tc109_write_and_file_counts_paired_runs` L:37 N:1 C:5 P:1 (Long Function), `test_hostile_artifact_symlink_escape_no_write` L:30 N:0 C:1 P:1 (Medium Length); `_counted_run` L:20 N:1 C:3 P:3 (Medium Length). `tests/qa/test_qap_ms004.py` — file 477 lines (review <=300); noise decomposition 42 trimmable full-line comments; 7 issues in 7 functions: Long Function 4, Medium Length 3; top: `test_tc114_one_posture_governs_every_outbound_call` L:71 N:1 C:4 P:1, `test_r1_upstream_url_shape_does_not_change_posture` L:45 N:0 C:2 P:1, `test_tc003_untrusted_peer_refused_by_default` L:36 N:0 C:2 P:1, `test_tc113_same_peer_accepted_with_optout` L:34 N:0 C:1 P:1, `test_r1_duplicate_flag_occurrence_honored_once` L:21 N:0 C:2 P:1; `assert_single_client` L:18 N:0 C:1 P:1 (Medium Length)
- Sizes: tests/qa/test_qap_ms002.py: 634, tests/qa/test_qap_ms004.py: 477
- Public paths kept: none (neither module defines `__all__`, and nothing outside the repository imports `tests.qa.test_qap_ms002` or `tests.qa.test_qap_ms004`)

## Map

### U0 · comments
| # | Kind | What | From | To |
|---|---|---|---|---|
| 1 | shrink | module docstring (29 lines: the TC↔VP map, the hostile-tier mapping, the run line, the environment note), the ten 3-line `# ---` banners to their one-line titles, and the two 4-line comments (`:55-58` threshold rationale, `:464-467` clean-exit discriminator) | tests/qa/test_qap_ms002.py | kept reasons: `qa-plan.md` §QA Gate Matrix + §QA Author Passes (the TC↔VP map), `.agents/skills/qa-project/SKILL.md` §Upstream (live upstream, fail closed) and §Capability Declarations (deterministic triggers); the clean-exit discriminator stays in the test, one line |
| 2 | shrink | module docstring (41 lines: the TC map, the W1/W2 witnesses, the adversarial-tier dispositions, the harness notes), the five 3-line `# ---` banners to their one-line titles, and the per-test narratives (`PATH_DEADLINE` 3 lines, TC-114 8 lines, R5 7 lines) | tests/qa/test_qap_ms004.py | kept reasons: `qa-plan.md` §QA Gate Matrix + §Scenario Contracts (TC-003/113/114, W1/W2), `.agents/skills/qa-project/SKILL.md` §Upstream + §Capability Declarations (loopback peer, readiness/quiescence, no fixed sleeps, no telemetry); the R2/R4 no-surface disposition has no other home in the tree (its iteration report is squashed away), so it stays as one line in the module docstring |
- Narrow tests: tests/qa/test_qap_ms002.py, tests/qa/test_qap_ms004.py
- Green after U0: neither file drops a unit — both stay far over 300 (634 -> ~586 and 477 -> ~420), so every split unit below still runs.

### U1 · tests/qa/test_qap_ms002.py: extract the shared constants and text helpers into a support module
| # | Kind | What | From | To |
|---|---|---|---|---|
| 1 | move | `COLLAPSED_FORMATS`, `TURN_SIZE`, `THRESHOLD_AT_TURN_2`, `UNREACHABLE_BYTES`, `TIMER_OFF_INTERVAL` | tests/qa/test_qap_ms002.py | tests/qa/qap_ms002_support.py |
| 2 | move | `big_text`, `summary_lines` | tests/qa/test_qap_ms002.py | tests/qa/qap_ms002_support.py |
- Importers to update: tests/qa/test_qap_ms002.py (import block only; `json`, `os`, `re`, `pytest`, `Proxy`, `poll_until` stay)
- Witnesses to re-point: none (no `invariants/` record names either path)
- Tests with import lines or patch targets to update: tests/qa/test_qap_ms002.py import block (currently `:31-51`, before U0 shifts it)
- Narrow tests: tests/qa/test_qap_ms002.py

### U2 · tests/qa/test_qap_ms002.py: move the hostile tier into its own module
| # | Kind | What | From | To |
|---|---|---|---|---|
| 1 | move | `test_hostile_non_utf8_and_empty_body_no_write`, `test_hostile_duplicate_json_keys_last_wins_single_record`, `test_hostile_trailing_newline_body_is_valid`, `test_hostile_dangling_session_dir_symlink_no_write`, `test_hostile_artifact_symlink_escape_no_write` | tests/qa/test_qap_ms002.py | tests/qa/test_qap_ms002_hostile.py |
- Importers to update: none (new module; its import block pulls `json`, `os`, `assert_upstream_response`, `read_text`, `record_section`, `session_id`, `turn_body`, `Proxy`, `TIMER_OFF_INTERVAL`, `UNREACHABLE_BYTES`)
- Witnesses to re-point: `qa-plan.md:77` (the QAP-MS002 `Tests:` list) and `qa-plan.md:80` (its run command) gain `tests/qa/test_qap_ms002_hostile.py`; the re-proof lines at `:132`, `:149-150`, `:162` record past runs at fixed HEADs and stay as written
- Tests with import lines or patch targets to update: tests/qa/test_qap_ms002_hostile.py (new import block); no change to tests/qa/test_qap_ms002.py's import block
- Narrow tests: tests/qa/test_qap_ms002_hostile.py (5 tests, not selected by the brief's Verify command — the yield must run it explicitly), tests/qa/test_qap_ms002.py (10 remaining)

### U3 · tests/qa/test_qap_ms002.py: move the flush-trigger scenarios and the counted measurement into their own module
| # | Kind | What | From | To |
|---|---|---|---|---|
| 1 | move | `N_TURNS`, `_counted_run`, `test_tc105_tc109_write_and_file_counts_paired_runs`, `test_tc106_volume_flush`, `test_tc110_shutdown_flush`, `test_tc111_interval_flush`, `test_tc112_timer_resilience` | tests/qa/test_qap_ms002.py | tests/qa/test_qap_ms002_flush.py |
- Importers to update: none (new module; its import block pulls `json`, `os`, `signal`, `pytest`, `assert_upstream_response`, `log_artifacts`, `read_text`, `session_id`, `turn_body`, `Proxy`, `poll_until`, `THRESHOLD_AT_TURN_2`, `TIMER_OFF_INTERVAL`, `UNREACHABLE_BYTES`, `big_text`, `summary_lines`)
- Witnesses to re-point: `qa-plan.md:77`, `qa-plan.md:80` (same path-only update as U2)
- Tests with import lines or patch targets to update: tests/qa/test_qap_ms002_flush.py (new import block); tests/qa/test_qap_ms002.py import block (currently `:31-51`; drop `signal`, now unused)
- Narrow tests: tests/qa/test_qap_ms002_flush.py (6 collected tests, not selected by the brief's Verify command — run it explicitly), tests/qa/test_qap_ms002.py (4 collected tests remaining)

### U4 · tests/qa/test_qap_ms002.py: split the long gate bodies
| # | Kind | What | From | To |
|---|---|---|---|---|
| 1 | reshape | `test_tc102_103_104_collapsed_artifact_shape_readability_turn0` (parametrized gate): extract `_assert_collapsed_artifact(sid_dir, fmt)` (TC-102: one `<fmt>.log`, arrival order, summary-line forms; returns the text) and `_assert_change_readable(text, fmt)` (TC-103) and `_assert_turn0_recoverable(text, system)` (TC-104) | tests/qa/test_qap_ms002.py | tests/qa/test_qap_ms002.py |
| 2 | reshape | `test_tc105_tc109_write_and_file_counts_paired_runs`: extract `_assert_counted_collapsed(rep, sid_dir, fmt)` (F=2, per-flush bytes, one file, `N` records) and `_assert_documented_half(rep, sid_dir)` (the `N + 1` files and `2N` writes) | tests/qa/test_qap_ms002_flush.py | tests/qa/test_qap_ms002_flush.py |
| 3 | reshape | `test_tc101_session_isolation_near_miss_prompts`: extract `_await_both_artifacts(a1, a2, s1, s2)` (the bounded poll) and `_assert_sessions_isolated(dump, s1, s2, t1, t2)` (distinct sids, listing, no cross-records) | tests/qa/test_qap_ms002.py | tests/qa/test_qap_ms002.py |
| 4 | reshape | `test_tc112_timer_resilience`: extract `_drive_raising_ticks(proxy, artifact, system)` (chmod 444, two turns, poll the logged `PermissionError`, restore 644 in `finally`) | tests/qa/test_qap_ms002_flush.py | tests/qa/test_qap_ms002_flush.py |
- Importers to update: none
- Witnesses to re-point: none
- Tests with import lines or patch targets to update: none (the reshaped helpers stay in the modules that call them)
- Narrow tests: tests/qa/test_qap_ms002.py, tests/qa/test_qap_ms002_flush.py

### U5 · tests/qa/test_qap_ms004.py: extract the shared constants and assertion helpers into a support module
| # | Kind | What | From | To |
|---|---|---|---|---|
| 1 | move | `PATH_DEADLINE`, `CERT_MARKER`, `CERT_REASON`, `MODEL_PROBE`, `METRICS_POLL`, `PROXIED`, `OUTBOUND_PATHS` | tests/qa/test_qap_ms004.py | tests/qa/qap_ms004_support.py |
| 2 | move | `start_peer`, `read_report`, `call_paths`, `assert_single_client`, `assert_all_calls_through`, `assert_cert_verification_failure`, `assert_clean_shutdown`, `messages_body` | tests/qa/test_qap_ms004.py | tests/qa/qap_ms004_support.py |
- Importers to update: tests/qa/test_qap_ms004.py (import block only; `json` and `os` leave with the helpers, while `signal` stays and `urlparse` is needed by both the helpers and the gate bodies)
- Witnesses to re-point: none (`invariants/` names neither path)
- Tests with import lines or patch targets to update: tests/qa/test_qap_ms004.py import block (currently `:43-52`, before U0 shifts it)
- Narrow tests: tests/qa/test_qap_ms004.py

### U6 · tests/qa/test_qap_ms004.py: move the adversarial tier into its own module
| # | Kind | What | From | To |
|---|---|---|---|---|
| 1 | move | `test_r5_trust_identity_pair_is_not_merged`, `test_r1_duplicate_flag_occurrence_honored_once`, `test_r1_upstream_url_shape_does_not_change_posture` | tests/qa/test_qap_ms004.py | tests/qa/test_qap_ms004_adversarial.py |
- Importers to update: none (new module; its import block pulls `signal`, `urlparse`, `MARKER`, `TlsPeer`, `Proxy`, `poll_until`, `PATH_DEADLINE`, `MODEL_PROBE`, `assert_cert_verification_failure`, `assert_clean_shutdown`, `assert_single_client`, `messages_body`, `read_report`, `start_peer`)
- Witnesses to re-point: `qa-plan.md:87` (the QAP-MS004 `Tests:` list) and `qa-plan.md:93` (its run command) gain `tests/qa/test_qap_ms004_adversarial.py`; `.sessions/codify/roadmap/2026-10-18-message-dump-bulk-flush.md:45` names the gate file as the carrier of both postures and stays true
- Tests with import lines or patch targets to update: tests/qa/test_qap_ms004_adversarial.py (new import block); tests/qa/test_qap_ms004.py import block (currently `:43-52` — nothing it still uses drops out)
- Narrow tests: tests/qa/test_qap_ms004_adversarial.py (3 tests, not selected by the brief's Verify command — run it explicitly), tests/qa/test_qap_ms004.py (3 remaining)

### U7 · tests/qa/test_qap_ms004.py: split the paired-run gates and the one-client assertion
| # | Kind | What | From | To |
|---|---|---|---|---|
| 1 | reshape | `test_tc114_one_posture_governs_every_outbound_call`: extract `_run_arm(tmp_path, tag, extra_args)` (peer, proxy, drive every call site, stop, read the report, `assert_single_client`; returns `(report, connector)`) and `_assert_optout_everywhere(result, peer)` and `_assert_refused_everywhere(result, peer, proxy)` | tests/qa/test_qap_ms004.py | tests/qa/test_qap_ms004.py |
| 2 | reshape | `test_r1_upstream_url_shape_does_not_change_posture`: extract `_run_url_shape_arm(tmp_path, tag, extra_args)` (the same driver with `upstream=peer.url() + hostile_suffix`; returns `(report, connector)`) | tests/qa/test_qap_ms004_adversarial.py | tests/qa/test_qap_ms004_adversarial.py |
| 3 | reshape | `test_tc003_untrusted_peer_refused_by_default`: extract `_assert_refused_before_the_peer(proxy, peer)` (status, error class, abort witness, empty peer log) | tests/qa/test_qap_ms004.py | tests/qa/test_qap_ms004.py |
| 4 | reshape | `test_tc113_same_peer_accepted_with_optout`: extract `_assert_optout_reaches_every_site(peer)` (the `{/v1/models, /metrics}` poll, the proxied hit, no aborts) | tests/qa/test_qap_ms004.py | tests/qa/test_qap_ms004.py |
| 5 | reshape | `assert_single_client` (called by TC-003, TC-113, TC-114, `test_r1_duplicate_flag_occurrence_honored_once`, `test_r1_upstream_url_shape_does_not_change_posture`): extract `_assert_one_connector(report)` and `_assert_one_session(report, connector)`; the public signature and return value stay | tests/qa/qap_ms004_support.py | tests/qa/qap_ms004_support.py |
- Importers to update: none
- Witnesses to re-point: none
- Tests with import lines or patch targets to update: none (helpers stay in the modules that call them)
- Narrow tests: tests/qa/test_qap_ms004.py, tests/qa/test_qap_ms004_adversarial.py

## U0 Forward
- Moved: nothing
- Reshaped: nothing
- Files: tests/qa/test_qap_ms002.py, tests/qa/test_qap_ms004.py
- Tests edited: none
- Ran: `.venv/bin/python -m pytest tests/qa/test_qap_ms002.py tests/qa/test_qap_ms004.py -o addopts=` -> 21 passed (after row 1); 21 passed (after row 2); 21 passed (Verify)
- Scan: `tests/qa/test_qap_ms002.py` — file 598 lines (ideal <=500); 15 issues in 14 functions: Long Function 4, Medium Length 10, Medium Complexity 1; top: `test_tc102_103_104_collapsed_artifact_shape_readability_turn0` L:46 N:1 C:5 P:2, `test_tc112_timer_resilience` L:44 N:1 C:4 P:1, `test_tc101_session_isolation_near_miss_prompts` L:38 N:0 C:6 P:1, `test_tc105_tc109_write_and_file_counts_paired_runs` L:37 N:1 C:5 P:1, `test_hostile_artifact_symlink_escape_no_write` L:30 N:0 C:1 P:1. `tests/qa/test_qap_ms004.py` — file 437 lines (review <=300); noise decomposition 30 trimmable full-line comments; 7 issues in 7 functions: Long Function 4, Medium Length 3; top: `test_tc114_one_posture_governs_every_outbound_call` L:71 N:1 C:4 P:1, `test_r1_upstream_url_shape_does_not_change_posture` L:45 N:0 C:2 P:1, `test_tc003_untrusted_peer_refused_by_default` L:36 N:0 C:2 P:1, `test_tc113_same_peer_accepted_with_optout` L:34 N:0 C:1 P:1, `test_r1_duplicate_flag_occurrence_honored_once` L:21 N:0 C:2 P:1
- Sizes: tests/qa/test_qap_ms002.py: 598, tests/qa/test_qap_ms004.py: 437
- Noticed: the scan's function-level findings are pre-existing debt in code this unit only re-commented; U4/U7 carry the reshape rows for them.
  Ask: no — U4/U7 already own those reshapes.

## U0 Yield
- Verdict: ACCEPT
- Ran: `.venv/bin/python -m pytest tests/qa/test_qap_ms002.py tests/qa/test_qap_ms004.py -o addopts=` -> 21 passed in 75.59s (Safety net: 21 passed, 0 skipped)
- Scan: `codify scan path tests/qa/test_qap_ms002.py tests/qa/test_qap_ms004.py` -> `tests/qa/test_qap_ms002.py` 598 lines (ideal <=500), 15 issues in 14 functions: Long Function 4, Medium Length 10, Medium Complexity 1, top `test_tc102_103_104_collapsed_artifact_shape_readability_turn0` L:46 N:1 C:5 P:2; `tests/qa/test_qap_ms004.py` 437 lines (review <=300), noise decomposition 30 trimmable full-line comments, 7 issues in 7 functions: Long Function 4, Medium Length 3, top `test_tc114_one_posture_governs_every_outbound_call` L:71 N:1 C:4 P:1. Every function and metric matches the Safety net's scan; none added, none worse.
- Sizes: tests/qa/test_qap_ms002.py: 598, tests/qa/test_qap_ms004.py: 437
- Findings: none
- Noticed: both files end 12 and 17 lines above the map's ~586 / ~420 projection (598 / 437 actual) while staying far over 300, so every split unit below still runs.
  Ask: no — the projection is a planning estimate; the files are still over 300 and no downstream unit depends on the exact count.

## U1 Forward
- Moved: `COLLAPSED_FORMATS`, `TURN_SIZE`, `THRESHOLD_AT_TURN_2`, `UNREACHABLE_BYTES`, `TIMER_OFF_INTERVAL` (with the two-line `~TURN_SIZE` comment above them) -> tests/qa/qap_ms002_support.py
- Moved: `big_text`, `summary_lines` -> tests/qa/qap_ms002_support.py
- Reshaped: nothing
- Files: tests/qa/test_qap_ms002.py, tests/qa/qap_ms002_support.py (new)
- Tests edited: tests/qa/test_qap_ms002.py: import block only (added `from tests.qa.qap_ms002_support import ...`); `json`, `os`, `re`, `pytest`, `Proxy`, `poll_until` stay and are still used
- Ran: `.venv/bin/python -m pytest tests/qa/test_qap_ms002.py -o addopts=` -> 15 passed (after row 1); 15 passed (after row 2); `.venv/bin/python -m pytest tests/qa/test_qap_ms002.py tests/qa/test_qap_ms004.py -o addopts=` (Verify) -> 21 passed
- Scan: `tests/qa/test_qap_ms002.py` — file 589 lines (ideal <=500); 15 issues in 14 functions: Long Function 4, Medium Length 10, Medium Complexity 1; top: `test_tc102_103_104_collapsed_artifact_shape_readability_turn0` L:46 N:1 C:5 P:2, `test_tc112_timer_resilience` L:44 N:1 C:4 P:1, `test_tc101_session_isolation_near_miss_prompts` L:38 N:0 C:6 P:1, `test_tc105_tc109_write_and_file_counts_paired_runs` L:37 N:1 C:5 P:1, `test_hostile_artifact_symlink_escape_no_write` L:30 N:0 C:1 P:1. `tests/qa/qap_ms002_support.py` — `scan clean` (No refactoring issues found)
- Sizes: tests/qa/test_qap_ms002.py: 589, tests/qa/qap_ms002_support.py: 23
- Noticed: the scan's 15 function-level findings are the same set U0 recorded, in code this unit did not touch; U4 carries their reshape rows.
  Ask: no — U4 already owns those reshapes and this unit changed no function body.

## U1 Yield
- Verdict: ACCEPT
- Ran: `.venv/bin/python -m pytest tests/qa/test_qap_ms002.py tests/qa/test_qap_ms004.py -o addopts=` -> 21 passed in 71.68s (Safety net: 21 passed, 0 skipped)
- Scan: `codify scan path tests/qa/test_qap_ms002.py tests/qa/qap_ms002_support.py` -> `tests/qa/test_qap_ms002.py` 589 lines (ideal <=500), 15 issues in 14 functions: Long Function 4, Medium Length 10, Medium Complexity 1, top `test_tc102_103_104_collapsed_artifact_shape_readability_turn0` L:46 N:1 C:5 P:2, `test_tc112_timer_resilience` L:44 N:1 C:4 P:1, `test_tc101_session_isolation_near_miss_prompts` L:38 N:0 C:6 P:1, `test_tc105_tc109_write_and_file_counts_paired_runs` L:37 N:1 C:5 P:1, `test_hostile_artifact_symlink_escape_no_write` L:30 N:0 C:1 P:1; `tests/qa/qap_ms002_support.py` `scan clean` (No refactoring issues found). Every function and metric matches the Safety net's scan; none added, none worse.
- Sizes: tests/qa/test_qap_ms002.py: 589, tests/qa/qap_ms002_support.py: 23
- Findings: none
- Noticed: `COLLAPSED_FORMATS` is independently re-declared in five other `tests/` dump modules (`test_dump.py`, `test_dump_records.py`, `test_dump_triggers.py`, `test_dump_write_count.py`, `test_dump_flush.py`), a dedup candidate outside this unit's scope.
  Ask: no — each declaration is self-consistent today and nothing depends on the QA copy.

## U2 Forward
- Moved: `test_hostile_non_utf8_and_empty_body_no_write`, `test_hostile_duplicate_json_keys_last_wins_single_record`, `test_hostile_trailing_newline_body_is_valid`, `test_hostile_dangling_session_dir_symlink_no_write`, `test_hostile_artifact_symlink_escape_no_write` (with their two section banners, byte-identical) -> tests/qa/test_qap_ms002_hostile.py
- Reshaped: nothing
- Files: tests/qa/test_qap_ms002_hostile.py (new), tests/qa/test_qap_ms002.py, .sessions/codify/02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/qa-plan.md
- Tests edited: tests/qa/test_qap_ms002_hostile.py: new import block only; tests/qa/test_qap_ms002.py import block unchanged (still uses every name it imports)
- Ran: `.venv/bin/python -m pytest tests/qa/test_qap_ms002_hostile.py tests/qa/test_qap_ms002.py -o addopts=` -> 15 passed (5 hostile + 10 remaining); `.venv/bin/python -m pytest tests/qa/test_qap_ms002.py tests/qa/test_qap_ms004.py -o addopts=` (Verify) -> 16 passed
- Scan: `tests/qa/test_qap_ms002.py` — file 429 lines (review <=300), noise decomposition 32 trimmable full-line comments; 10 issues in 9 functions: Long Function 4, Medium Complexity 1, Medium Length 5; top: `test_tc102_103_104_collapsed_artifact_shape_readability_turn0` L:46 N:1 C:5 P:2, `test_tc112_timer_resilience` L:44 N:1 C:4 P:1, `test_tc101_session_isolation_near_miss_prompts` L:38 N:0 C:6 P:1, `test_tc105_tc109_write_and_file_counts_paired_runs` L:37 N:1 C:5 P:1, `test_identity_sid_format_near_miss_two_artifacts` L:29 N:0 C:2 P:1. `tests/qa/test_qap_ms002_hostile.py` — file 179 lines (no file-size finding); 5 issues in 5 functions: Medium Length 5; top: `test_hostile_artifact_symlink_escape_no_write` L:30 N:0 C:1 P:1, `test_hostile_dangling_session_dir_symlink_no_write` L:27 N:0 C:1 P:1, `test_hostile_duplicate_json_keys_last_wins_single_record` L:26 N:0 C:1 P:1, `test_hostile_non_utf8_and_empty_body_no_write` L:25 N:0 C:1 P:1, `test_hostile_trailing_newline_body_is_valid` L:19 N:0 C:1 P:1
- Sizes: tests/qa/test_qap_ms002.py: 429, tests/qa/test_qap_ms002_hostile.py: 179
- Noticed: the 5 Medium Length findings in the new hostile module are the same functions the Safety net scan reported inside test_qap_ms002.py, moved verbatim; the map has no reshape row for the hostile tier.
  Ask: no — they are pre-existing debt that this unit only relocated, and U4/U7 do not name them.
- Noticed: `tests/qa/test_qap_ms002.py`'s module docstring still says it covers "the hostile tier (R1/R2)", which now lives in the new module; no map row authorizes a docstring edit here.
  Ask: no — a stale docstring misleads no runtime path and breaks nothing.
- Noticed: `codify knowledge sync-registry --apply` reported `already-current` and two pre-existing `curated-index-generated-section` errors in `invariants/index.md:24` and `:43`; it wrote no file.
  Ask: no — the registry paths are unchanged by this unit and the errors predate it.

## U2 Yield
- Verdict: ACCEPT
- Ran: `.venv/bin/python -m pytest tests/qa/test_qap_ms002.py tests/qa/test_qap_ms004.py -o addopts=` -> 16 passed in 66.63s (the 5 moved hostile tests are no longer collected by this command, as the map row says); `.venv/bin/python -m pytest tests/qa/test_qap_ms002_hostile.py -o addopts=` -> 5 passed in 3.31s (21 of 21 tests green across the three modules, equal to the Safety net's 21 passed, 0 skipped)
- Scan: `codify scan path tests/qa/test_qap_ms002.py tests/qa/test_qap_ms002_hostile.py` -> `tests/qa/test_qap_ms002.py` 429 lines (review <=300), noise decomposition 32 trimmable full-line comments, 10 issues in 9 functions: Long Function 4, Medium Complexity 1, Medium Length 5, top `test_tc102_103_104_collapsed_artifact_shape_readability_turn0` L:46 N:1 C:5 P:2; `tests/qa/test_qap_ms002_hostile.py` 179 lines (no file-size finding), 5 issues in 5 functions: Medium Length 5, top `test_hostile_artifact_symlink_escape_no_write` L:30 N:0 C:1 P:1. Together 15 issues in 14 functions, the same set the Safety net printed; every function and metric matches, none added, none worse.
- Sizes: tests/qa/test_qap_ms002.py: 429, tests/qa/test_qap_ms002_hostile.py: 179
- Findings: none
- Noticed: the moved block is byte-identical to the removed one (banner included); the new file adds only its one-line docstring, `from __future__`, and the import block the map row lists, and every remaining import in `tests/qa/test_qap_ms002.py` is still used.
  Ask: no — the only diff line is the module separator blank line.
- Noticed: `tests/qa/test_qap_ms002.py`'s module docstring still claims "the hostile tier (R1/R2)" and its run line still names only the old path, while the authoritative `qa-plan.md:77,80` now names both modules.
  Ask: no — a stale docstring changes no test collection and misleads nobody who follows the updated qa-plan; the map authorizes no docstring edit in this unit.
- Noticed: `codify knowledge sync-registry` preview reports `current` (no pending sync); `codify knowledge check --profile strict` reports 5 diagnostics, all pre-existing at HEAD in files this unit never touched: `missing-version` in `buster_ripper/dump/invariants.md:1`, `buster_ripper/server/invariants.md:1` and `index.md:1`, plus the two `curated-index-generated-section` errors in `invariants/index.md:24` and `:43`.
  Ask: no — `invariants/` is unchanged by this unit (`git status` clean there) and every diagnostic exists at HEAD; the re-pointed witness is a session `qa-plan.md`, which the registry does not track.

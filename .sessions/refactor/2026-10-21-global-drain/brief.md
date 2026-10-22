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

## U3 Forward
- Moved: `N_TURNS`, `_counted_run`, `test_tc105_tc109_write_and_file_counts_paired_runs`, `test_tc106_volume_flush`, `test_tc110_shutdown_flush`, `test_tc111_interval_flush`, `test_tc112_timer_resilience` (with their four section banners and the `N_TURNS` comment, byte-identical: `diff` of the old `:202-429` against the new body is empty) -> tests/qa/test_qap_ms002_flush.py
- Reshaped: nothing
- Files: tests/qa/test_qap_ms002_flush.py (new), tests/qa/test_qap_ms002.py, .sessions/codify/02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/qa-plan.md
- Tests edited: tests/qa/test_qap_ms002_flush.py: new import block only (the row's list plus `COLLAPSED_FORMATS`, which TC-105's `(*COLLAPSED_FORMATS, None)` loop needs); tests/qa/test_qap_ms002.py: import block only, dropped `signal` (only the moved TC-110 used it)
- Ran: `.venv/bin/python -m pytest tests/qa/test_qap_ms002_flush.py tests/qa/test_qap_ms002.py -o addopts=` -> 10 passed (6 flush + 4 remaining); `.venv/bin/python -m pytest tests/qa/test_qap_ms002.py tests/qa/test_qap_ms004.py -o addopts=` (Verify) -> 10 passed
- Scan: `tests/qa/test_qap_ms002.py` — file 198 lines (no file-size finding); 4 issues in 3 functions: Long Function 2, Medium Complexity 1, Medium Length 1; top: `test_tc102_103_104_collapsed_artifact_shape_readability_turn0` L:46 N:1 C:5 P:2, `test_tc101_session_isolation_near_miss_prompts` L:38 N:0 C:6 P:1, `test_identity_sid_format_near_miss_two_artifacts` L:29 N:0 C:2 P:1. `tests/qa/test_qap_ms002_flush.py` — file 259 lines (no file-size finding); 6 issues in 6 functions: Long Function 2, Medium Length 4; top: `test_tc112_timer_resilience` L:44 N:1 C:4 P:1, `test_tc105_tc109_write_and_file_counts_paired_runs` L:37 N:1 C:5 P:1, `test_tc110_shutdown_flush` L:26 N:1 C:2 P:2, `test_tc111_interval_flush` L:26 N:0 C:2 P:1, `test_tc106_volume_flush` L:25 N:0 C:1 P:1, `_counted_run` L:20 N:1 C:3 P:3
- Sizes: tests/qa/test_qap_ms002.py: 198, tests/qa/test_qap_ms002_flush.py: 259
- Noticed: the 6 flush-module findings are the same functions the Safety net scan reported inside test_qap_ms002.py, moved verbatim; U4 carries the reshape rows for `test_tc105_tc109_write_and_file_counts_paired_runs` and `test_tc112_timer_resilience`, and the other four (three tests plus `_counted_run`, all Medium Length) have no reshape row.
  Ask: no — pre-existing debt that this unit only relocated.
- Noticed: `tests/qa/test_qap_ms002.py`'s module docstring still lists TC-105…TC-112 and its run line still names only the old path, both now in the flush module; no map row authorizes a docstring edit here.
  Ask: no — a stale docstring changes no test collection and misleads nobody who follows the updated qa-plan.
- Noticed: `codify knowledge sync-registry --apply` reported `already-current` and the same two pre-existing `curated-index-generated-section` errors in `invariants/index.md:24` and `:43`; it wrote no file.
  Ask: no — the re-pointed witness is a session `qa-plan.md`, which the registry does not track, and every diagnostic exists at HEAD.

## U3 Yield
- Verdict: ACCEPT
- Ran: `.venv/bin/python -m pytest tests/qa/test_qap_ms002.py tests/qa/test_qap_ms004.py -o addopts=` -> 10 passed in 49.33s (the 6 moved tests are no longer collected by this command, as the map row says); `.venv/bin/python -m pytest tests/qa/test_qap_ms002_flush.py -o addopts=` -> 6 passed in 17.63s (with U2's verified 5 hostile passes, 21 of 21 tests green, equal to the Safety net's 21 passed, 0 skipped)
- Scan: `codify scan path tests/qa/test_qap_ms002.py tests/qa/test_qap_ms002_flush.py` -> `tests/qa/test_qap_ms002.py` 198 lines (no file-size finding), 4 issues in 3 functions: Long Function 2, Medium Complexity 1, Medium Length 1, top `test_tc102_103_104_collapsed_artifact_shape_readability_turn0` L:46 N:1 C:5 P:2, `test_tc101_session_isolation_near_miss_prompts` L:38 N:0 C:6 P:1, `test_identity_sid_format_near_miss_two_artifacts` L:29 N:0 C:2 P:1; `tests/qa/test_qap_ms002_flush.py` 259 lines (no file-size finding), 6 issues in 6 functions: Long Function 2, Medium Length 4, top `test_tc112_timer_resilience` L:44 N:1 C:4 P:1, `test_tc105_tc109_write_and_file_counts_paired_runs` L:37 N:1 C:5 P:1, `test_tc110_shutdown_flush` L:26 N:1 C:2 P:2, `test_tc111_interval_flush` L:26 N:0 C:2 P:1, `test_tc106_volume_flush` L:25 N:0 C:1 P:1, `_counted_run` L:20 N:1 C:3 P:3. Together 10 issues in 9 functions — exactly the set the Safety net printed inside `tests/qa/test_qap_ms002.py`; every function and metric matches, none added, none worse.
- Sizes: tests/qa/test_qap_ms002.py: 198, tests/qa/test_qap_ms002_flush.py: 259
- Findings: none
- Noticed: `diff` of the removed `tests/qa/test_qap_ms002.py:202-429` against `tests/qa/test_qap_ms002_flush.py:32-259` is empty, banners and the `N_TURNS` comment included; the new file adds only its one-line docstring, `from __future__`, and the import block (the map's list plus `COLLAPSED_FORMATS`, which TC-105's `(*COLLAPSED_FORMATS, None)` loop needs); `signal` was dropped from `tests/qa/test_qap_ms002.py` and is used nowhere else there.
  Ask: no — the import set is complete and every remaining name is still used.
- Noticed: `tests/qa/test_qap_ms002.py`'s module docstring still lists TC-105…TC-112 and its run line still names only the old path, both now in the flush module; no map row authorizes a docstring edit here.
  Ask: no — a stale docstring changes no test collection and misleads nobody who follows the updated `qa-plan.md:77,80`.
- Noticed: `codify knowledge sync-registry` preview reports `current` (no pending sync); `codify knowledge check --profile strict` reports 5 diagnostics, all pre-existing at HEAD in files this unit never touched: `missing-version` in `buster_ripper/dump/invariants.md:1`, `buster_ripper/server/invariants.md:1` and `index.md:1`, plus the two `curated-index-generated-section` errors in `invariants/index.md:24` and `:43`.
  Ask: no — `invariants/` is unchanged by this unit (`git status` clean there) and every diagnostic exists at HEAD; the re-pointed witness is a session `qa-plan.md`, which the registry does not track.

## U4 Forward
- Moved: nothing
- Reshaped: `test_tc102_103_104_collapsed_artifact_shape_readability_turn0`: extracted `_assert_collapsed_artifact(sid_dir, fmt)` (one `<fmt>.log`, arrival order `[0, 1, 2]`, FULL / +A-R summary forms; returns the text), `_assert_change_readable(text, fmt)` (TC-103 diff / rendered-section checks) and `_assert_turn0_recoverable(text, system)` (TC-104 body and rendering); the test keeps its setup, the sensitivity posts and the three calls
- Reshaped: `test_tc105_tc109_write_and_file_counts_paired_runs`: extracted `_assert_counted_collapsed(rep, sid_dir, fmt)` (F=2, per-flush bytes, one file, `N` records) and `_assert_documented_half(rep, sid_dir)` (the `N + 1` files, `2N` writes, index line count, turn-000 model, no `.log` artifact)
- Reshaped: `test_tc101_session_isolation_near_miss_prompts`: extracted `_await_both_artifacts(a1, a2, s1, s2)` (the bounded poll; `s1`/`s2` are the two posted sentinels) and `_assert_sessions_isolated(dump, s1, s2, t1, t2)` (distinct sids, listing, no cross-records)
- Reshaped: `test_tc112_timer_resilience`: extracted `_drive_raising_ticks(proxy, artifact, system)` (chmod 444, two turns, poll the logged `PermissionError`, restore 644 in `finally`, then the later-tick flush poll); the test keeps its setup, turn 0 and the healthy-process assert
- Files: tests/qa/test_qap_ms002.py, tests/qa/test_qap_ms002_flush.py
- Tests edited: none
- Ran: `.venv/bin/python -m pytest tests/qa/test_qap_ms002.py tests/qa/test_qap_ms002_flush.py -o addopts=` -> 10 passed (after row 1); 10 passed (after row 2); 10 passed (after row 3); 10 passed (after row 4); `.venv/bin/python -m pytest tests/qa/test_qap_ms002.py tests/qa/test_qap_ms004.py -o addopts=` (Verify) -> 10 passed
- Scan: `tests/qa/test_qap_ms002.py` — 5 issues in 5 functions: Medium Length 3, Medium Parameters 2; `test_identity_sid_format_near_miss_two_artifacts` L:29 N:0 C:2 P:1, `test_tc101_session_isolation_near_miss_prompts` L:27 N:0 C:1 P:1, `test_tc102_103_104_collapsed_artifact_shape_readability_turn0` L:25 N:0 C:1 P:2 (all Medium Length), `_await_both_artifacts` L:8 N:0 C:4 P:4 and `_assert_sessions_isolated` L:5 N:0 C:3 P:5 (Medium Parameters). `tests/qa/test_qap_ms002_flush.py` — 7 issues in 7 functions: Medium Length 7; `test_tc112_timer_resilience` L:27 N:0 C:2 P:1, `test_tc110_shutdown_flush` L:26 N:1 C:2 P:2, `test_tc111_interval_flush` L:26 N:0 C:2 P:1, `test_tc106_volume_flush` L:25 N:0 C:1 P:1, `_counted_run` L:20 N:1 C:3 P:3, `test_tc105_tc109_write_and_file_counts_paired_runs` L:19 N:1 C:5 P:1, `_drive_raising_ticks` L:18 N:1 C:3 P:3. Every Long Function and Medium Complexity finding the Safety net reported in these two functions is gone; the remaining findings are pre-existing Medium Length debt or map-mandated helper signatures.
- Sizes: tests/qa/test_qap_ms002.py: 217, tests/qa/test_qap_ms002_flush.py: 271
- Noticed: the two new Medium Parameters findings (`_await_both_artifacts` P:4, `_assert_sessions_isolated` P:5) and the new `_drive_raising_ticks` Medium Length (L:18) come from the map's own signatures and from the extracted body verbatim; shrinking them would mean deviating from the map.
  Ask: no — the map names those signatures explicitly and no size target is set for these modules.
- Noticed: `_assert_documented_half` derives the dump root as `os.path.dirname(sid_dir)`, equal to the old `str(tmp_path / "dump-documented")`, because the map's signature omits `tmp_path`; the no-`.log` assertion tests the same path.
  Ask: no — the derived string is identical, so nothing observable changed.
- Noticed: `_await_both_artifacts` takes the two sentinels (`"sentinel-one"`, `"sentinel-two"`) as its `s1`/`s2` arguments, matching the map's parameter names; passing the session prompts instead would have changed the poll predicate, so the sentinels keep the predicate byte-identical.
  Ask: no — the helper is the same bounded poll with the same inputs.

## U4 Yield
- Verdict: ACCEPT
- Ran: `.venv/bin/python -m pytest tests/qa/test_qap_ms002.py tests/qa/test_qap_ms004.py -o addopts=` -> 10 passed in 49.45s; `.venv/bin/python -m pytest tests/qa/test_qap_ms002_flush.py -o addopts=` -> 6 passed in 17.66s (with U2's verified 5 hostile passes, 21 of 21 tests green, equal to the Safety net's 21 passed, 0 skipped)
- Scan: `codify scan path tests/qa/test_qap_ms002.py tests/qa/test_qap_ms002_flush.py` -> `tests/qa/test_qap_ms002.py` 5 issues in 5 functions: Medium Length 3, Medium Parameters 2 — `test_identity_sid_format_near_miss_two_artifacts` L:29 N:0 C:2 P:1, `test_tc101_session_isolation_near_miss_prompts` L:27 N:0 C:1 P:1, `test_tc102_103_104_collapsed_artifact_shape_readability_turn0` L:25 N:0 C:1 P:2, `_await_both_artifacts` L:8 N:0 C:4 P:4, `_assert_sessions_isolated` L:5 N:0 C:3 P:5; `tests/qa/test_qap_ms002_flush.py` 7 issues in 7 functions: Medium Length 7 — `test_tc112_timer_resilience` L:27 N:0 C:2 P:1, `test_tc110_shutdown_flush` L:26 N:1 C:2 P:2, `test_tc111_interval_flush` L:26 N:0 C:2 P:1, `test_tc106_volume_flush` L:25 N:0 C:1 P:1, `_counted_run` L:20 N:1 C:3 P:3, `test_tc105_tc109_write_and_file_counts_paired_runs` L:19 N:1 C:5 P:1, `_drive_raising_ticks` L:18 N:1 C:3 P:3. Every Long Function and Medium Complexity finding the Safety net printed in the four reshaped tests is gone; the only names the Safety net did not print are the map's three new helpers.
- Sizes: tests/qa/test_qap_ms002.py: 217, tests/qa/test_qap_ms002_flush.py: 271
- Findings: none
- Noticed: the assertion multiset is identical to HEAD across both files; the only changed production line is `_assert_documented_half`'s `assert not log_artifacts(os.path.dirname(sid_dir))`, equal to the old `str(tmp_path / "dump-documented")` because `_counted_run` builds `dump = tmp_path / f"dump-{tag}"` with `tag = "documented"`.
  Ask: no — the two expressions name the same path for every input `_counted_run` can produce.
- Noticed: the scan's new names (`_await_both_artifacts` Medium Parameters P:4, `_assert_sessions_isolated` Medium Parameters P:5, `_drive_raising_ticks` Medium Length L:18) are the map's own helper signatures and their extracted bodies verbatim, not regressions in pre-existing functions.
  Ask: no — shrinking them would mean deviating from the signatures the map names.
- Noticed: `_await_both_artifacts` receives the sentinels (`"sentinel-one"`, `"sentinel-two"`) in its `s1`/`s2` parameters, so the poll predicate is byte-identical to HEAD's inline literals.
  Ask: no — passing the session prompts instead would change what the poll waits for.

## U5 Forward
- Moved: `PATH_DEADLINE`, `CERT_MARKER`, `CERT_REASON`, `MODEL_PROBE`, `METRICS_POLL`, `PROXIED`, `OUTBOUND_PATHS` (with their two `#:` comments) -> tests/qa/qap_ms004_support.py
- Moved: `start_peer`, `read_report`, `call_paths`, `assert_single_client`, `assert_all_calls_through`, `assert_cert_verification_failure`, `assert_clean_shutdown`, `messages_body` (with the `# helpers` banner, byte-identical: `diff` of the old `:33-132` against the new module's `:13-112` is empty) -> tests/qa/qap_ms004_support.py
- Reshaped: nothing
- Files: tests/qa/qap_ms004_support.py (new), tests/qa/test_qap_ms004.py
- Tests edited: tests/qa/test_qap_ms004.py: import block only (dropped `os` and `turn_body`, added `from tests.qa.qap_ms004_support import (...)`); `json`, `signal`, `urlparse`, `Proxy`, `poll_until`, `MARKER` and `TlsPeer` stay and are still used
- Ran: `.venv/bin/python -m pytest tests/qa/test_qap_ms004.py -o addopts=` -> 6 passed (after row 1); 6 passed (after row 2); `.venv/bin/python -m pytest tests/qa/test_qap_ms002.py tests/qa/test_qap_ms004.py -o addopts=` (Verify) -> 10 passed
- Scan: `tests/qa/test_qap_ms004.py` — file 347 lines (review <=300), noise decomposition 22 trimmable full-line comments; 6 issues in 6 functions: Long Function 4, Medium Length 2; top: `test_tc114_one_posture_governs_every_outbound_call` L:71 N:1 C:4 P:1, `test_r1_upstream_url_shape_does_not_change_posture` L:45 N:0 C:2 P:1, `test_tc003_untrusted_peer_refused_by_default` L:36 N:0 C:2 P:1, `test_tc113_same_peer_accepted_with_optout` L:34 N:0 C:1 P:1, `test_r1_duplicate_flag_occurrence_honored_once` L:21 N:0 C:2 P:1. `tests/qa/qap_ms004_support.py` — 1 issue in 1 function: Medium Length 1; `assert_single_client` L:18 N:0 C:1 P:1. Together 7 issues in 7 functions — exactly the set the Safety net printed in `tests/qa/test_qap_ms004.py`; every function and metric matches, none added, none worse
- Sizes: tests/qa/test_qap_ms004.py: 347, tests/qa/qap_ms004_support.py: 115
- Noticed: the map's import note says `json` leaves with the helpers, but `test_tc114_one_posture_governs_every_outbound_call` still formats `json.dumps(report['calls'], indent=2)` in its final "both postures from one connector" block, which U7's extraction rows do not name, so `json` stays in the test module.
  Ask: no — dropping it would only surface as a NameError while that assertion is already failing, and no passing run changes.
- Noticed: `os` and `turn_body` did leave, as the row says; `urlparse`, `Proxy`, `poll_until`, `MARKER` and `TlsPeer` are needed by both the helpers and the gate bodies (or the remaining R5 test) and stay in the test module.
  Ask: no — every remaining import is used and every moved helper's dependency is imported in the support module.
- Noticed: the support module imports `Proxy` for its two parameter annotations only; with `from __future__ import annotations` that is a typing import, kept for the documented signature, and pyflakes counts annotation use as use, so the repo's `F` gate stays clean (ruff is not installed in this worktree, so the gate itself was not run).
  Ask: no — it changes no runtime path.
- Noticed: no witness line names either path (`invariants/` and the roadmap never mention `tests/qa/test_qap_ms004.py` helpers), so `codify knowledge sync-registry --apply` was not run; no other module imports names from `tests.qa.test_qap_ms004`.
  Ask: no — the registry tracks neither module and the re-point list is empty.

## U5 Yield
- Verdict: ACCEPT
- Ran: `.venv/bin/python -m pytest tests/qa/test_qap_ms004.py -o addopts=` -> 6 passed in 44.79s (the module's narrow tests); `.venv/bin/python -m pytest tests/qa/test_qap_ms002.py tests/qa/test_qap_ms004.py -o addopts=` -> 10 passed in 49.34s (the 11 MS-002 tests moved by U2/U3 are no longer collected by this command, as their map rows say); `.venv/bin/python -m pytest tests/qa/test_qap_ms002_hostile.py tests/qa/test_qap_ms002_flush.py -o addopts=` -> 11 passed in 22.55s. 21 of 21 tests green across the four modules, equal to the Safety net's 21 passed, 0 skipped.
- Scan: `codify scan path tests/qa/test_qap_ms004.py tests/qa/qap_ms004_support.py` -> `tests/qa/test_qap_ms004.py` 347 lines (review <=300), noise decomposition 22 trimmable full-line comments, 6 issues in 6 functions: Long Function 4, Medium Length 2, top `test_tc114_one_posture_governs_every_outbound_call` L:71 N:1 C:4 P:1, `test_r1_upstream_url_shape_does_not_change_posture` L:45 N:0 C:2 P:1, `test_tc003_untrusted_peer_refused_by_default` L:36 N:0 C:2 P:1, `test_tc113_same_peer_accepted_with_optout` L:34 N:0 C:1 P:1, `test_r1_duplicate_flag_occurrence_honored_once` L:21 N:0 C:2 P:1; `tests/qa/qap_ms004_support.py` 1 issue in 1 function: Medium Length 1, `assert_single_client` L:18 N:0 C:1 P:1. Together 7 issues in 7 functions — exactly the set the Safety net printed in `tests/qa/test_qap_ms004.py`; every function and metric matches, none added, none worse.
- Sizes: tests/qa/test_qap_ms004.py: 347, tests/qa/qap_ms004_support.py: 115
- Findings: none
- Noticed: `diff` of HEAD's `tests/qa/test_qap_ms004.py:33-132` against `tests/qa/qap_ms004_support.py:13-112` is empty, `#:` comments and the `# helpers` banner included; the new file adds only its one-line docstring, `from __future__`, and the import block, and the test file's diff holds only import lines plus that removed block, with every test body byte-identical.
  Ask: no — the move is byte-for-byte and the tests collect and pass unchanged.
- Noticed: the map's import note says `json` leaves with the helpers, but `test_tc114_one_posture_governs_every_outbound_call` still formats `json.dumps(report['calls'], indent=2)` at `:235`, a line U7's rows do not name, so `json` stays; `os` and `turn_body` did leave, and `CERT_REASON` is not imported because the test file no longer uses it.
  Ask: no — dropping `json` would only surface as a NameError while that assertion is already failing, and every remaining import is used.
- Noticed: the 6 remaining findings in `tests/qa/test_qap_ms004.py` are the same pre-existing functions the Safety net printed (the file stays 347 lines, over the 300 review bound), and `assert_single_client`'s L:18 Medium Length moved verbatim; U6/U7 carry the map rows for all of them.
  Ask: no — pre-existing debt this unit only relocated, and U6/U7 already own it.
- Noticed: no witness line names either path (`invariants/` and the roadmap mention neither module's helpers; roadmap line 45 only says `tests/qa/test_qap_ms004.py` carries both postures, still true), so no registry command was needed and no other module imports names from `tests.qa.test_qap_ms004`.
  Ask: no — the re-point list is empty and the registry tracks neither module.

## U6 Forward
- Moved: `test_r5_trust_identity_pair_is_not_merged`, `test_r1_duplicate_flag_occurrence_honored_once`, `test_r1_upstream_url_shape_does_not_change_posture` (with the `# Adversarial tier` banner, byte-identical: `diff` of the old `:239-347` against the new module's `:22-130` is empty) -> tests/qa/test_qap_ms004_adversarial.py
- Reshaped: nothing
- Files: tests/qa/test_qap_ms004_adversarial.py (new), tests/qa/test_qap_ms004.py, .sessions/codify/02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/qa-plan.md
- Tests edited: tests/qa/test_qap_ms004_adversarial.py: new import block only; tests/qa/test_qap_ms004.py: import block only (dropped `CERT_MARKER`, used only by the moved R5 test)
- Ran: `.venv/bin/python -m pytest tests/qa/test_qap_ms004_adversarial.py tests/qa/test_qap_ms004.py -o addopts=` -> 6 passed (3 adversarial + 3 remaining); `.venv/bin/python -m pytest tests/qa/test_qap_ms002.py tests/qa/test_qap_ms004.py -o addopts=` (Verify) -> 7 passed
- Scan: `tests/qa/test_qap_ms004.py` — 235 lines (no file-size finding); 3 issues in 3 functions: Long Function 3; `test_tc114_one_posture_governs_every_outbound_call` L:71 N:1 C:4 P:1, `test_tc003_untrusted_peer_refused_by_default` L:36 N:0 C:2 P:1, `test_tc113_same_peer_accepted_with_optout` L:34 N:0 C:1 P:1. `tests/qa/test_qap_ms004_adversarial.py` — 130 lines (no file-size finding); 3 issues in 3 functions: Long Function 1, Medium Length 2; `test_r1_upstream_url_shape_does_not_change_posture` L:45 N:0 C:2 P:1 (Long Function), `test_r1_duplicate_flag_occurrence_honored_once` L:21 N:0 C:2 P:1 and `test_r5_trust_identity_pair_is_not_merged` L:16 N:0 C:1 P:1 (Medium Length). Together 6 issues in 6 functions — exactly the set the Safety net printed in `tests/qa/test_qap_ms004.py`; every function and metric matches, none added, none worse
- Sizes: tests/qa/test_qap_ms004.py: 235, tests/qa/test_qap_ms004_adversarial.py: 130
- Noticed: the map row's import list for the new module names `urlparse`, but none of the three moved tests uses it, so it was omitted to keep the `F` gate clean; conversely `CERT_MARKER` is absent from that list yet the R5 test asserts on it, so it was imported.
  Ask: no — the import set is exactly what the moved bodies reference, and every remaining name in `tests/qa/test_qap_ms004.py` is still used.
- Noticed: `tests/qa/test_qap_ms004.py`'s module docstring still says "the adversarial tier adds the R5 trust-identity near-miss pair and the R1 flag-repeat and upstream-URL-shape boundaries" and its run line names only the old path, both now in the new module; no map row authorizes a docstring edit here.
  Ask: no — a stale docstring changes no test collection and misleads nobody who follows the updated `qa-plan.md:87,93`.
- Noticed: the 6 remaining findings are the same pre-existing functions the Safety net printed, moved verbatim for the three adversarial ones; U7 carries the reshape rows for all of them (`assert_single_client` included).
  Ask: no — pre-existing debt this unit only relocated, and U7 already owns it.
- Noticed: `codify knowledge sync-registry --apply` reported `already-current` and the same two pre-existing `curated-index-generated-section` errors in `invariants/index.md:24` and `:43`; it wrote no file.
  Ask: no — the re-pointed witness is a session `qa-plan.md`, which the registry does not track, and every diagnostic exists at HEAD.

## U6 Yield
- Verdict: ACCEPT
- Ran: `.venv/bin/python -m pytest tests/qa/test_qap_ms002.py tests/qa/test_qap_ms004.py -o addopts=` -> 7 passed in 47.65s (the 3 moved tests are no longer collected by this command, as the map row says); `.venv/bin/python -m pytest tests/qa/test_qap_ms004_adversarial.py -o addopts=` -> 3 passed in 1.85s (with U2's verified 5 hostile and U3's verified 6 flush passes, 21 of 21 tests green, equal to the Safety net's 21 passed, 0 skipped); `codify knowledge sync-registry` -> `Mode: preview — current` (no pending sync); `codify knowledge check --profile strict` -> 5 diagnostics, all pre-existing at HEAD in files this unit never touched
- Scan: `codify scan path tests/qa/test_qap_ms004.py tests/qa/test_qap_ms004_adversarial.py` -> `tests/qa/test_qap_ms004.py` 3 issues in 3 functions: Long Function 3 — `test_tc114_one_posture_governs_every_outbound_call` L:71 N:1 C:4 P:1, `test_tc003_untrusted_peer_refused_by_default` L:36 N:0 C:2 P:1, `test_tc113_same_peer_accepted_with_optout` L:34 N:0 C:1 P:1; `tests/qa/test_qap_ms004_adversarial.py` 3 issues in 3 functions: Long Function 1, Medium Length 2 — `test_r1_upstream_url_shape_does_not_change_posture` L:45 N:0 C:2 P:1, `test_r1_duplicate_flag_occurrence_honored_once` L:21 N:0 C:2 P:1, `test_r5_trust_identity_pair_is_not_merged` L:16 N:0 C:1 P:1. Together 6 issues in 6 functions — exactly the Safety net's 7 less `assert_single_client` L:18, which U5 moved to `tests/qa/qap_ms004_support.py`; every function and metric matches, none added, none worse; no file-size finding on either module.
- Sizes: tests/qa/test_qap_ms004.py: 235, tests/qa/test_qap_ms004_adversarial.py: 130
- Findings: none
- Noticed: `diff` of HEAD's `tests/qa/test_qap_ms004.py:239-347` against `tests/qa/test_qap_ms004_adversarial.py:22-130` is empty, the `# Adversarial tier` banner included; the new file adds only its one-line docstring, `from __future__`, and the import block; `tests/qa/test_qap_ms004.py`'s diff is one dropped import plus that block, with every test body byte-identical.
  Ask: no — the move is byte-for-byte and the 6 tests collect and pass unchanged.
- Noticed: the map row's import list names `urlparse`, which none of the three moved tests uses, and omits `CERT_MARKER`, which the R5 test asserts on; the new module imports exactly what its bodies reference, and `tests/qa/test_qap_ms004.py` keeps every name it still uses.
  Ask: no — an unused import would trip the `F` gate and an omitted one would fail at collection; the bodies are unchanged either way.
- Noticed: `tests/qa/test_qap_ms004.py`'s module docstring still describes the adversarial tier and its run line names only the old path; no map row authorizes a docstring edit in this unit, and the authoritative `qa-plan.md:87,93` now names both modules.
  Ask: no — stale prose changes no test collection and misleads nobody who follows the updated qa-plan.
- Noticed: `codify knowledge check --profile strict` reports `missing-version` in `invariants/buster_ripper/dump/invariants.md:1`, `invariants/buster_ripper/server/invariants.md:1` and `invariants/index.md:1`, plus the two `curated-index-generated-section` errors in `invariants/index.md:24` and `:43`; `git diff HEAD -- invariants/` is empty, so every diagnostic exists at HEAD in files this unit never touched.
  Ask: no — `invariants/` is unchanged by this unit and the re-pointed witness is a session `qa-plan.md`, which the registry does not track.

## U7 Forward
- Moved: nothing
- Reshaped: `test_tc114_one_posture_governs_every_outbound_call`: extracted `_run_arm(tmp_path, tag, extra_args)` (peer, proxy, `_drive_all_call_sites`, the arm's own evidence, stop, read the report, `assert_single_client`, `assert_all_calls_through`, `assert_clean_shutdown`; returns `(report, connector)`), `_assert_optout_everywhere(result, peer)` (run A's in-try claims) and `_assert_refused_everywhere(result, peer, proxy)` (run B's in-try claims); the test keeps the two `_run_arm` calls, the `ssl_kwarg` / `OUTBOUND_PATHS <= call_paths` report assertions and the "both postures from one connector" block
- Reshaped: `test_r1_upstream_url_shape_does_not_change_posture`: extracted `_run_url_shape_arm(tmp_path, tag, extra_args)` (peer, proxy on `upstream=peer.url() + "/qa-base/../"`, the arm's own witness, stop, read the report, `assert_single_client`, `assert_clean_shutdown`; returns `(report, connector)`); the test keeps the two calls and the `ssl_kwarg` assertions
- Reshaped: `test_tc003_untrusted_peer_refused_by_default`: extracted `_assert_refused_before_the_peer(proxy, peer)` (the post, status/error class, abort witness, empty peer log; returns `(status, body)`); the test keeps the lifecycle pair and every post-try report assertion
- Reshaped: `test_tc113_same_peer_accepted_with_optout`: extracted `_assert_optout_reaches_every_site(peer)` (the `{/v1/models, /metrics}` poll, the proxied hit, no aborts); the test keeps the post and its status/marker assertions
- Reshaped: `assert_single_client`: extracted `_assert_one_connector(report)` and `_assert_one_session(report, connector)`; the public signature, the assertion set and the returned connector are unchanged
- Files: tests/qa/test_qap_ms004.py, tests/qa/test_qap_ms004_adversarial.py, tests/qa/qap_ms004_support.py
- Tests edited: none
- Ran: `.venv/bin/python -m pytest tests/qa/test_qap_ms004.py tests/qa/test_qap_ms004_adversarial.py -o addopts=` -> 6 passed (after row 1); 6 passed (after row 2); 6 passed (after row 3); 6 passed (after row 4); 6 passed (after row 5); `.venv/bin/python -m pytest tests/qa/test_qap_ms002.py tests/qa/test_qap_ms004.py -o addopts=` (Verify) -> 7 passed; `.venv/bin/python -m pytest tests/qa/test_qap_ms002.py tests/qa/test_qap_ms002_hostile.py tests/qa/test_qap_ms002_flush.py tests/qa/test_qap_ms004.py tests/qa/test_qap_ms004_adversarial.py -o addopts=` -> 21 passed (the family, equal to the Safety net's 21 passed, 0 skipped)
- Scan: `codify scan path tests/qa/test_qap_ms004.py tests/qa/test_qap_ms004_adversarial.py tests/qa/qap_ms004_support.py` -> `tests/qa/test_qap_ms004.py` — no file-size finding; 4 issues in 4 functions: Medium Length 4; `test_tc113_same_peer_accepted_with_optout` L:27 N:0 C:1 P:1, `test_tc003_untrusted_peer_refused_by_default` L:25 N:0 C:2 P:1, `_run_arm` L:21 N:1 C:2 P:3, `test_tc114_one_posture_governs_every_outbound_call` L:20 N:1 C:2 P:1. `tests/qa/test_qap_ms004_adversarial.py` — no file-size finding; 3 issues in 3 functions: Long Function 1, Medium Length 2; `_run_url_shape_arm` L:33 N:1 C:3 P:3, `test_r1_duplicate_flag_occurrence_honored_once` L:21 N:0 C:2 P:1, `test_r5_trust_identity_pair_is_not_merged` L:16 N:0 C:1 P:1. `tests/qa/qap_ms004_support.py` — 125 lines, `scan clean` (No refactoring issues found). Every Long Function finding the Safety net printed in the four reshaped tests is gone; `test_tc114` and `test_tc003`/`test_tc113` dropped from L:71/L:36/L:34 to L:20/L:25/L:27, and the URL-shape test's L:45 Long Function moved into the map's own `_run_url_shape_arm` (L:33)
- Sizes: tests/qa/test_qap_ms004.py: 249, tests/qa/test_qap_ms004_adversarial.py: 126, tests/qa/qap_ms004_support.py: 125
- Noticed: row 4's `_assert_optout_reaches_every_site(peer)` has no proxy parameter, so it cannot post the proxied request itself; the post and its `status == 200` / `MARKER in body` assertions stay in the test and now run before the helper's poll, which keeps the same predicate and the same `PATH_DEADLINE`. Every assertion, its message and its inputs are unchanged; only the order of two independent checks moved.
  Ask: no — the passing set is identical and a failing run still reports one of the same two assertions.
- Noticed: `_run_arm` dispatches on `extra_args` to pick the arm's evidence, because the map returns only `(report, connector)` while `_assert_refused_everywhere` needs the still-running proxy's `stderr()`; `extra_args` is exactly the posture each arm encodes.
  Ask: no — each arm runs the same evidence calls on the same inputs as before the reshape.
- Noticed: `_run_arm` and `_run_url_shape_arm` name their temp dirs `dump-{tag}` / `logs-{tag}` / `probe-{tag}.json` with `tag` = `optout`/`default` and `verify`/`optout` instead of HEAD's `-a`/`-b`; the request system strings stay byte-identical (`qap-ms004-tc114-{tag}`, `qap-ms004-r1-url-verify`) and nothing asserts on the temp path names.
  Ask: no — the files live under pytest's `tmp_path` and no assertion names them.
- Noticed: `_assert_refused_before_the_peer` returns `(status, body)` so TC-003's lifecycle check keeps `status2 == status`; the map row names no return value, and the helper's parenthetical (status, error class, abort witness, empty peer log) is what it now contains.
  Ask: no — the return is the post's own result, not an added assertion.
- Noticed: `assert_single_client` now indexes `report["connectors"][0]` before `_assert_one_session` checks `total_sessions == 1`, where HEAD checked `total_sessions` first; a report with zero connectors and a non-one session count would raise IndexError instead of the session-count message. Every report a passing run produces has one connector and one session, and the assertion set is unchanged.
  Ask: no — only a broken-report diagnostic changed, never a pass/fail outcome.
- Noticed: `test_r1_upstream_url_shape_does_not_change_posture` discards the two returned reports with `_`, since only the connectors are asserted; this keeps the map's `(report, connector)` return without adding assertions.
  Ask: no — no assertion was removed or added.
- Noticed: `_run_url_shape_arm` is still a Long Function (L:33) because it is the map's own helper, the union of both arms' bodies; no map row names a further split, so it stays.
  Ask: no — shrinking it would mean deviating from the helper the map names.
- Noticed: the map's row 1 lists `_assert_refused_everywhere(result, peer, proxy)` and row 2's driver as `upstream=peer.url() + hostile_suffix`; both are implemented as written, including the `hostile_suffix` literal moving into the helper.
  Ask: no — the reshapes follow the rows exactly.
- Noticed: ruff/pyflakes is not installed in this worktree, so the `F` gate was not run; every import in the three files was audited by use, and `json`, `urlparse`, `OUTBOUND_PATHS`, `call_paths` and the moved assertions are all still referenced.
  Ask: no — no import was added or left unused.
- Noticed: the unit re-pointed no witness line, so `codify knowledge sync-registry --apply` was not run; `invariants/` and the roadmap name none of these three files' helpers, and no other module imports names from `tests.qa.test_qap_ms004` or `tests.qa.qap_ms004_support`.
  Ask: no — the re-point list is empty.

## U7 Yield
- Verdict: ACCEPT
- Ran: `.venv/bin/python -m pytest tests/qa/test_qap_ms002.py tests/qa/test_qap_ms004.py -o addopts=` -> 7 passed in 48.19s (the 3 adversarial tests are no longer collected by this command, as the map row says); `.venv/bin/python -m pytest tests/qa/test_qap_ms002.py tests/qa/test_qap_ms002_hostile.py tests/qa/test_qap_ms002_flush.py tests/qa/test_qap_ms004.py tests/qa/test_qap_ms004_adversarial.py -o addopts=` -> 21 passed in 71.03s, equal to the Safety net's 21 passed, 0 skipped; `uv run pytest` (Final verify) -> 575 passed, 22 warnings in 11.14s
- Scan: `codify scan path tests/qa/test_qap_ms004.py tests/qa/test_qap_ms004_adversarial.py tests/qa/qap_ms004_support.py` -> `tests/qa/test_qap_ms004.py` 4 issues in 4 functions: Medium Length 4 — `test_tc113_same_peer_accepted_with_optout` L:27 N:0 C:1 P:1, `test_tc003_untrusted_peer_refused_by_default` L:25 N:0 C:2 P:1, `_run_arm` L:21 N:1 C:2 P:3, `test_tc114_one_posture_governs_every_outbound_call` L:20 N:1 C:2 P:1; `tests/qa/test_qap_ms004_adversarial.py` 3 issues in 3 functions: Long Function 1, Medium Length 2 — `_run_url_shape_arm` L:33 N:1 C:3 P:3, `test_r1_duplicate_flag_occurrence_honored_once` L:21 N:0 C:2 P:1, `test_r5_trust_identity_pair_is_not_merged` L:16 N:0 C:1 P:1; `tests/qa/qap_ms004_support.py` 125 lines, `scan clean` (No refactoring issues found). Every Long Function finding the Safety net printed in the reshaped tests is gone (TC-114 L:71→L:20, TC-003 L:36→L:25, TC-113 L:34→L:27, URL-shape L:45→`_run_url_shape_arm` L:33) and no function it printed got worse; the only names the Safety net did not print are the map's own new helpers `_run_arm` and `_run_url_shape_arm`, and `assert_single_client`'s L:18 Medium Length is gone with the support module clean. No file-size finding on any of the three files.
- Sizes: tests/qa/test_qap_ms004.py: 249, tests/qa/test_qap_ms004_adversarial.py: 126, tests/qa/qap_ms004_support.py: 125
- Findings: none
- Noticed: `test_tc113_same_peer_accepted_with_optout` now posts the proxied turn and asserts `status == 200` / `MARKER in body` before `_assert_optout_reaches_every_site` runs its `{/v1/models, /metrics}` poll, where HEAD polled first; the map's helper signature carries no proxy, so the post cannot move into it.
  Ask: no — every assertion, message and input is unchanged and the poll still gates the same predicate, so a passing run is identical and a failing run only reports the same two assertions in the other order.
- Noticed: `assert_single_client` now indexes `report["connectors"][0]` before `_assert_one_session` checks `total_sessions == 1`, because the map's row makes the connector the helper's argument; a zero-connector report with a wrong session count would raise IndexError instead of the session-count message.
  Ask: no — every report a passing run produces has one connector and one session, so only a broken-report diagnostic differs.
- Noticed: `_run_arm` also absorbed the two report assertions each arm ran after `assert_single_client` (`assert_all_calls_through`, `assert_clean_shutdown`) and dispatches its in-try evidence on `extra_args`, because the map's row lists only `assert_single_client` and returns only `(report, connector)` while `_assert_refused_everywhere` needs the running proxy's stderr.
  Ask: no — `extra_args` is exactly the posture each HEAD arm encoded, and the same assertions run on the same inputs in the same order.
- Noticed: `assert_clean_shutdown` now runs before the test's `ssl_kwarg` / `OUTBOUND_PATHS` assertions in both paired-run tests, where HEAD ran it last; it only reads `rc` and `proxy.stderr()`.
  Ask: no — the assertion set, messages and inputs are unchanged, so only which failure surfaces first in an already-failing run differs.
- Noticed: the extracted blocks are the HEAD bodies with `_a`/`_b` suffixes dropped and the temp paths renamed `-a`/`-b` to `optout`/`default` and `verify`/`optout`; the request system strings (`qap-ms004-tc114-{tag}`, `qap-ms004-r1-url-verify`) and `hostile_suffix` are byte-identical, and an AST comparison of the whole family shows 58 asserts at HEAD and 58 now, the only textual difference being `assert_single_client(report)["ssl_kwarg"] == ...` rewritten as `connector_*["ssl_kwarg"] == ...` on the connector the same call returns.
  Ask: no — nothing asserts on the temp path names and the assertion multiset is identical.
- Noticed: `codify knowledge sync-registry` reports `Mode: preview — current` (no pending sync); `codify knowledge check --profile strict` reports 5 diagnostics, all pre-existing at HEAD in files this unit never touched: `missing-version` in `buster_ripper/dump/invariants.md:1`, `buster_ripper/server/invariants.md:1` and `index.md:1`, plus the two `curated-index-generated-section` errors in `invariants/index.md:24` and `:43`.
  Ask: no — the unit re-pointed no witness line, `git diff HEAD -- invariants/` is empty, and every diagnostic exists at HEAD.

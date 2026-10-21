# Split the two free QA gate modules

- Goal: split `tests/qa/test_qap_ms002.py` and `tests/qa/test_qap_ms004.py` along their scenario/helper boundaries, with behavior unchanged.
- Target: open (global roadmap); chosen candidates #4/#5; no roadmap row.
- In scope: `tests/qa/test_qap_ms002.py`, `tests/qa/test_qap_ms004.py`
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

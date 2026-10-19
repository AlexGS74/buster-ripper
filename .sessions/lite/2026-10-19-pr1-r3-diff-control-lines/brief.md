# Terminate the unified diff's control lines in both dump writers

- Session: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
- Findings: PR #1 review comment 4316745969 — `buster_ripper/dump.py:265-273` (P2); family: `_write_diff_turn` (`:183-192`, the documented `turn_NNN.diff`) and `_change_focused_record` (`:212-219`, the `change-focused` record)
- Goal: Every diff the dump writes is a valid unified diff — the `---`, `+++` and `@@` control lines stand on their own terminated lines — and the QA gate rejects the glued shape instead of mirroring it.
- Done when: (a) a documented two-turn session's `turn_001.diff` carries `--- turn_000\n`, `+++ turn_001\n` and a terminated `@@ … @@\n` each on its own line (red before the fix: `--- turn_000+++ turn_001@@ …`); (b) a `change-focused` turn-1 record carries the same terminated control lines; (c) the `+A/-R` counts and both summary lines are byte-unchanged; (d) the `QAP-MS001` expected side is recomputed with terminated control lines and its coarse check fails on a glued artifact.
- Wrong witness: only the `change-focused` record is fixed and `turn_NNN.diff` stays glued (one helper feeds both); or `"\n"` is inserted at the join while `lineterm=""` stays (every content line is double-terminated); or the gate is left mirroring the glued shape (`S21` N5).
- In scope: `buster_ripper/dump.py` (`_unified_diff`); `tests/test_dump.py` (`_expected_diff`), `tests/test_dump_records.py` (`_expected_diff`), `tests/qa/test_qap_ms001.py` (`production_diff_text`, the coarse check, and the three docstrings that describe the glued join).
- Out of scope: `_diff_counts` (it reads the yielded list, not the text); `_flush_state`'s record separators; `turn_000.json`; the `self-contained` rendering; `README.md` (already says "unified diff"); the `S21` composition verdict (capped).
- Agrees with: `README.md:165-166` ("unified diff: turn 000 → turn 001"); `README.md:198` (the `change-focused` record's "unified diff"); `G-006` ("well-formed", "without additional tooling"); `G-008`(a)'s reading — "today's content and line format" bound to the three files and their record shapes, not to base's malformed control-line join (see [125](../../codify/02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/iterations/125-assumption-decide.md) §The `G-008`(a) binding); `S21` N5.
- Verify: `uv run pytest tests/test_dump.py tests/test_dump_records.py tests/qa/test_qap_ms001.py -q`; `uv run pytest -q`.
- Capacity: `dump.py` is 319 raw lines; one operand keeps it net 0 in the 301–500 band.

## Forward
- Changed:
  - `buster_ripper/dump.py` — `_unified_diff` passes `lineterm="\n"`, so the `""` join both writers use emits terminated `---`/`+++`/`@@` lines; docstring says lineterm touches only control lines; net 0 raw lines.
  - `tests/test_dump.py` — `_expected_diff` re-pointed to `lineterm="\n"`; added `test_diff_control_lines_are_terminated` for `turn_001.diff`.
  - `tests/test_dump_records.py` — `_expected_diff` re-pointed; added `test_later_turn_record_control_lines_are_terminated` for the `change-focused` turn-1 record.
  - `tests/qa/test_qap_ms001.py` — `production_diff_text` re-pointed to `lineterm="\n"`; `diff_changed_lines`/`_content_lines` read the well-formed shape; the coarse check is extracted as `assert_wellformed_diff` and rejects a glued artifact; CE-1 sensitivity test rewritten to witness that rejection.
- Ran: `uv run pytest tests/test_dump.py tests/test_dump_records.py tests/qa/test_qap_ms001.py -q` -> 73 passed (subset coverage addopts reports 40.8%, not a test failure); `uv run pytest -q` -> 540 passed, coverage 96.78%; red check with `lineterm=""` restored -> both new witnesses failed.
- Family sweep (every artifact, audited):
  - documented `turn_NNN.diff` (`_write_diff_turn`, `:186`) -> **fixed** by the same operand.
  - `change-focused` record (`_change_focused_record`, `:219`) -> **fixed** by the same operand.
  - `self-contained` record (`_self_contained_record`) -> safe: writes `current_text`, always terminated at `_render_body:91`.
  - `_flush_state` record separators -> safe: every record ends with `"\n"`, so each starts on its own line.
  - `index.txt` (`_full_summary`/`_diff_summary`) -> safe: both README clauses matched; `+A/-R` counts byte-unchanged (verified old vs new `lineterm`).
  - `turn_000.json` -> safe: `json.dumps(indent=2)`.
  - `QAP-MS001` expected side (`production_diff_text`, coarse check) -> **re-pointed and strengthened**.
- Noticed:
  - `buster_ripper_legacy.py:600` still joins with `lineterm=""`; the legacy reference copy is out of scope and not imported by the package. Ask: no — it is a frozen reference copy with no runtime effect.
  - `invariants/buster_ripper/assumption-claims.md:80` still quotes `lineterm=""` as the writer's actual; the knowledge record is outside this brief. Ask: no — stale evidence text, not runtime behavior.
  - `tests/qa/test_qap_ms001.py`'s module docstring 099 entry still names the structural reader that recovered glued lines; the three current function docstrings now describe the well-formed shape. Ask: no — historical re-entry record, current behavior documented where it is read.
  - `S21` composition stays capped and its recomposition stays deferred to the named trigger. Ask: no — no task, Fork or wave opens from it.

## Yield
- Verdict: ACCEPT
- Ran: `uv run pytest tests/test_dump.py tests/test_dump_records.py tests/qa/test_qap_ms001.py -q` -> 73 passed (subset coverage 40.8% under the addopts 80% gate, not a test failure); `uv run pytest -q` -> 540 passed, coverage 96.78%; `uv run pytest tests/test_dump.py tests/test_dump_records.py -q --no-cov` -> 56 passed; `uv run pytest tests/qa/test_qap_ms001.py -q --no-cov` -> 17 passed; in-process 3-turn run -> documented `turn_001.diff` is `'--- turn_000\n+++ turn_001\n@@ -6,4 +6,9 @@\n…'`, change-focused turn-1 record has the same terminated control lines, counts old/new `(6,1)/(6,1)` and `(7,2)/(7,2)`; gate branches checked: too-short diff, glued `---`+`+++` first line, glued `@@` hunk header, missing hunk, well-formed diff.
- Wrong witness: Both writers were read back (`turn_001.diff` and the change-focused pending record) and carry terminated control lines; content lines carry exactly one `\n` (repr), so the `lineterm=""` + `"\n"`-join shape is absent; `assert_wellformed_diff` rejected the fully glued artifact ("does not open with a terminated file header") and the hunk-header-glued artifact ("hunk header … not terminated").
- Findings: none
- Noticed: `tests/qa/test_qap_ms001.py:34-40` — the 099 module-docstring entry still describes the structural reader as current. Ask: no — historical re-entry record; the current behavior is documented in the three function docstrings.
- Noticed: `tests/test_dump.py:5,131` — "byte-for-byte unchanged" / "stays byte-for-byte unchanged (G-008(a))" are now false for the diff text. Ask: no — test docstrings only; the brief's G-008(a) binding says the protected reading is not byte-identity of the malformed join.
- Noticed: Unlisted session records (`.sessions/codify/.../handoff-context.md`, `roadmap.md`, `roadmap/2026-10-18-message-dump-bulk-flush.md`, `iterations/124-assumption-decide.md`, `iterations/125-assumption-decide.md`) are modified/untracked and were left unstaged. Ask: no — assumption-decide phase artifacts, not this change.

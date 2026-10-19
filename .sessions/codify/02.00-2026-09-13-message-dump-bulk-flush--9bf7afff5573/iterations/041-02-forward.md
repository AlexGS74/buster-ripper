## Orchestrator Prompt (verbatim)

> You are a subagent spawned by another session.
> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
> Task ID: T009

## Increment Brief

- T009 retry: make failed rendered-text UTF-8 sizing atomic before collapsed append; jointly retain T010's already-landed shutdown repair.
- Anchors touched: dump entry point → `dump.py:dump_turn` extend (via its existing `_append_record` helper); per-session dump state → `dump.py:SessionState` extend/reuse; shutdown flush seam → `server.py:on_cleanup` extend (already landed in 041-01).
- Will NOT create a new buffer, serializer, timer, signal handler, or artifact writer.
- Reuse evidence: no touched NEW anchor; inventions: none.
- VP-T002-volume-flush and VP-T003-shutdown-flush at MS-002; persistent QA gates remain QA-owned.

## Failure-Path Evidence

| Operation / condition | Expected outcome and authority | Handling / observation evidence | Disposition |
|---|---|---|---|
| UTF-8-unencodable rendered turn, then valid same-session turn | Encoding failure leaves pending/bytes/turn/prev_text unchanged; next turn begins turn 0 and flushes normally (Y-041-01-1, G-005, sequential chain) | `test_unencodable_rendering_does_not_poison_next_turn` for both collapsed formats; fail-first 2/2 red, then green | safe |
| Multibyte at/below inclusive threshold | Accumulate encoded rendered bytes; flush at `>=`, defer strictly below (T009/claim-91/102) | Existing per-format threshold tests in `test_dump_triggers.py`, green | safe |
| Separate sessions, successful flush | Isolate byte counts and reset only after successful write | Existing `test_utf8_pending_bytes_are_session_local` and flush tests; no change in flush | safe |
| Partial artifact write failure | Retain batch for retry; no new write policy | Existing `_flush_session` isolation and `test_flush_never_raises_on_write_failure`; unchanged | safe |
| Already-failed KV or timer task at cleanup | Continue peer cancellation, dump flush, session close (T010/seam-S2) | 041-01 `server.py` repair and focused done-task regressions; unchanged this retry | safe |
| Outer cleanup cancellation | Propagates, not mistaken for a completed task's ordinary failure | 041-01 `test_outer_cleanup_cancellation_propagates`; unchanged | safe |
| Documented format, dump disabled | No collapsed buffer/threshold; documented writer unchanged | Existing documented-path/guard regressions; unchanged | safe |

Excluded: hard kill, mid-process format mutation (once-per-process format contract), and unrelated route parse-boundary remediation T003.001. R1 parse handling, R2 writer targets, R3 capture→act, R4 payload fields and R5 keys are not changed by this ordering repair; the repair-class scan covered both collapsed dispatch branches and their shared append helper.

## Implementation and Evidence

- `dump.py:_append_record`: compute `len(current_text.encode("utf-8"))` before appending a record, then update `pending_bytes` and perform the existing inclusive threshold check. This preserves the failure-isolation caller and never creates an uncounted poisoned pending record.
- `tests/test_dump_triggers.py`: parametric JSON `"\\ud800"` failure followed by a valid turn for each collapsed format; assert unchanged pending/count/turn/prev_text, then turn-0 provenance, encoded count, exact successful artifact and reset. The two cases failed at `state.pending == []` before the production fix, then 17/17 focused trigger tests passed.
- Joint T010 from 041-01 remains in scope for Yield: `server.py:_cancel_background_task` logs/isolates ordinary completed-task failures, cancels the peer before synchronous final flush and session close, and propagates outer cancellation. Existing tests in `test_server_hooks.py` cover both slots and outer cancellation; no T010 file changed on retry.
- No new user choice, policy reversal, invariant edit, override exercise, or above-MVP deferral. Assumption impact implemented for Yield, not marked resolved: claim-91/102, S1/S5, Y-041-01-1 (T009); S2 (T010).
- Capacity gate: no additive production surface (`dump.py` local reorder plus one binding, 304→305 lines in review band; `_append_record` below ideal L/N/C/P). No extraction required. Backlog pickup: QB:dump/file-size-yellow remains below 500 and is not worsened into must-extract; QB:render-body/gray-compound and QB:dump-flush/residual-fmt-mixing do not match the touched function and remain deferred; test file 257→290 remains Green.

## Validation

- Focused `uv run pytest tests/test_dump_triggers.py -q -o addopts=''`: 17 passed after fail-first 2 failed.
- Full hermetic suite `uv run pytest`: **225 passed**, 25 warnings (preexisting async-mock coroutine warnings), 92.88% total coverage; `dump.py` 100% statement coverage. No project AGENTS.md or separate lint/type-check gate; `git diff --check` clean.
- LNCP: `codify scan path` per modified file with `--git --detailed`: `dump.py` 0 touched-function issues, `_append_record` L4/N0/C1/P3; `test_dump_triggers.py` one permitted gray metric (new test L26/N1/C1/P3). Preexisting `_render_body` L26/N2/C9 untouched. Raw file sizes 305 and 290 respectively; no 501+ ratchet violation.
- No visual UI/CLI output change in this repair; 041-01 evidence already shows threshold and shutdown behavior. Next: Yield T009/T010, then T003.001 and MS-002 re-verdict.

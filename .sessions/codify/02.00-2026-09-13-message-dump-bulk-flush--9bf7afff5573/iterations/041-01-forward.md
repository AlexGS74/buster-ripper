## Orchestrator Prompt (verbatim)

> You are a subagent spawned by another session.
> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
> Task ID: T009

## Increment Brief

- Tasks T009/T010: fix rendered-text UTF-8 byte threshold and preserve final dump flush when a completed background task failed.
- Anchors touched: dump entry point → `dump.py:dump_turn` extend; per-session dump state → `dump.py:SessionState` extend; shutdown flush seam → `server.py:on_cleanup` extend.
- Will NOT create: new buffer, timer, signal handler, format selector or artifact writer; all existing ownership reused.
- Reuse evidence: no touched NEW anchors. Inventions: none (test-local fixtures/assertion helper only).
- VP-T002-volume-flush / MS-002; VP-T003-shutdown-flush / MS-002; QAP-MS002 retains independent gate authorship.

## Failure-Path Evidence

| Operation / condition | Expected outcome and authority | Handling / observation evidence | Disposition |
|---|---|---|---|
| Collapsed append below UTF-8 threshold | Buffer remains pending, per G-005 and T009 | Per-format `test_multibyte_below_threshold_remains_pending` | safe |
| Append exactly at byte threshold, including non-ASCII | Flush immediately once, record retained and byte count resets, T009/claim-102 | Fail-first threshold test, both formats; exactly one append-open | safe |
| Per-session accumulation, multibyte and ASCII; successful flush | Counts isolated; pending and bytes clear only on success, T009/claim-91 | Parametrized `test_utf8_pending_bytes_are_session_local`, existing failure-retains-batch test | safe |
| Already-failed background task at cleanup (KV or dump timer) | Log ordinary error, cancel other task, flush before close, T010/seam-S2 | Two real done-task regressions with on-disk payload and close-time assertion | safe |
| Outer cleanup cancellation | Not swallowed; distinguish task cancellation from caller cancellation via `current_task().cancelling()`; T010 | `test_outer_cleanup_cancellation_propagates` cancels in-flight cleanup before flush and close | safe |
| Dump write failure after partial attempt | Existing per-session flush isolates exception and retains batch (no new policy), T003 | Existing `test_flush_never_raises_on_write_failure`; no change in `_flush_state` | safe |
| Documented format / DUMP_DIR unset | No buffered flush or threshold accounting; original documented writes unchanged | Existing documented-path/guard tests and new tests scoped to collapsed values | safe |

Out of scope: external-input parsing, concurrent format mutation and hard kill; T009/T010 do not alter those contracts.

## Implementation

- `dump.py:_append_record` now adds `len(current_text.encode("utf-8"))`; inclusive comparison remains after append and on the collapsed path only. Fixed the stale state comment. The renderer includes a Unicode box-drawing separator even for ASCII payloads, so the previous character-count unit undercounted all nonempty system/message renderings, not only the explicit `é` example.
- `server.py:_cancel_background_task` now logs ordinary failed-task exceptions and continues cleanup. A cancelled background task is still swallowed, but cancellation of the cleanup coroutine itself is re-raised using `current_task().cancelling()`; no `BaseException` catch was introduced. Both task slots are cancelled/awaited in order before final flush and shared-session close.
- Local tests: corrected prior count assertions, added multibyte/ASCII per-session and threshold property examples (both collapsed formats), real completed-task failure scenarios for each task slot, and an in-flight outer cancellation regression. Test hygiene: patched `dump.poll_flush` in the spawn test that mocks `create_task`, consuming QB:test-server-hooks/unawaited-coroutine.
- Files: `buster_ripper/dump.py`, `buster_ripper/server.py`, `tests/test_dump.py`, `tests/test_dump_triggers.py`, `tests/test_server_hooks.py`, session status, backlog, evidence index/artifact, this report.

## Validation

- Fail-first: ten focused cases failed on the old character count and failed-task exception behavior.
- Focused: `uv run pytest tests/test_dump.py tests/test_dump_triggers.py tests/test_server_hooks.py -q -o addopts=''` → 78 passed before the final fixture refactor; final targeted two-file run → 31 passed. The fixture refactor only moved setup, no production change.
- Full hermetic suite: `uv run pytest` → **223 passed**, 22 warnings, total 92.87% coverage; `dump.py` 100% (server omitted by project coverage config). Existing unrelated coroutine warnings remain.
- LNCP: `codify scan path` with `--git --detailed`, one touched file at a time: no production violations. Baseline/final `dump.py` 304 raw lines (yellow); `_render_body` L26/N2/C9 pre-existing, untouched. `server.py` 123→126 raw lines, no violations; `_cancel_background_task` L8/N1/C3/P2. `tests/test_dump.py` 413 lines, two test functions single gray P=4; `tests/test_dump_triggers.py` 257 lines, one test single gray P=4; `tests/test_server_hooks.py` 300 lines, outer-cancellation test single gray L20. No function has two gray metrics; `git diff --check` clean.
- Capacity gate: not triggered. `dump.py` is replacing, not adding production pressure (304→304); `server.py` additive 123→126 lines, no touched hard/two-gray function or 501+ file, final <500. No extraction required.
- Backlog pickup: QB:test-server-hooks/unawaited-coroutine fixed; QB:dump/file-size-yellow assessed, remains yellow with no net growth; QB:render-body/gray-compound and QB:dump-flush/residual-fmt-mixing out of touched function and unrelated to scope; QB:test-dump/file-size remains yellow, below 500, split deferred to T004.
- Evidence: [terminal capture](../evidence/t009-t010-terminal.txt), [index](../evidence/INDEX.md). The failed-task traceback is the intentional logged exception, not an escaped cleanup error.

## Assumptions and Durable Signal

- Assumption impact implemented for Yield: claim-91, claim-102, seam-S1, seam-S5 (T009); seam-S2 (T010). Not marked resolved here. Existing batch and format invariants preserved; no registry edit.
- No decision reversal, no approved override exercised beyond preserving OI output layout and sequential chain, no deferred above-floor feature scope.
- No new user question or plan disagreement. Next: Yield T009/T010, then T003.001 and MS-002 re-verdict.

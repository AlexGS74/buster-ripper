## Orchestrator Prompt (verbatim)

> You are a subagent spawned by another session.
> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
> Task ID: T010.001

## Increment Brief

- T010.001: retain the final collapsed dump when aiohttp enters cleanup with the runner already cancelling.
- Anchors touched: shutdown flush seam → `buster_ripper/server.py:on_cleanup` extend; signal handling → aiohttp runner reuse.
- Will NOT create: another signal handler, flush mechanism, writer, buffer or session store.
- Reuse evidence: no touched NEW anchors. Inventions: none (test-local runner coroutines only).
- VP-T003-shutdown-flush / MS-002; QAP-MS002 owns real-signal gate authorship.

## Failure-Path Evidence

| Operation / condition | Expected outcome and authority | Handling / observation evidence | Disposition |
|---|---|---|---|
| Cancelled child during runner's first signal cleanup | Both children awaited; flush pending artifact before close, US-001/G-002/T003 and seam-S2 | Baseline `cancelling()` count compared around each child await; fail-first `test_runner_cancel_count_does_not_skip_pending_dump` inspects artifact, both slots and close | safe |
| Additional cancellation while awaiting child | Propagate new outer cancellation rather than swallowing, T010 preserved behavior | `test_new_cancel_during_runner_cleanup_propagates` and pre-existing `test_outer_cleanup_cancellation_propagates` | safe |
| Child already failed ordinarily | Log failure, continue both slots, flush and close, T010 | Existing parametrized `test_done_failed_task_does_not_skip_final_flush` for both slots | safe |
| Missing task slot or dump write failure | Preserve existing cleanup ordering and isolation | Existing `test_handles_missing_dump_flush_task`, `test_shutdown_flush_failure_never_raises` | safe |

Out of scope: force-kill (no graceful hook), format switching, external input parsing, duplicate signal policy; this increment changes only cleanup's distinction between child and new parent cancellation.

## Implementation and validation

- `server.py:_cancel_background_task` snapshots the cleanup task's cancellation count before cancelling/awaiting each child. It swallows only a child cancellation with no new request to cancel cleanup, and propagates a fresh outer cancellation. Retains T010 ordinary-error handling and cancel-both → flush → close order.
- Two local regressions simulate aiohttp's retained first cancellation in the same runner task: one checks both cancelled child slots and durable artifact at close, the other injects a second cancellation during a child await and checks propagation/no flush or close. The first failed before the production edit (1 failed, 18 passed); after edit focused tests 19 passed.
- Capacity gate: additive production surface `server.py` 126 lines baseline, `_cancel_background_task` L8/N1/C3/P2; 128 lines final, no hard/two-gray metric, <300. Not triggered. Test file 300→356 lines (yellow, no 500-line breach); new tests each only single-gray L (17 and 27); no extraction needed.
- Backlog pickup: none match touch-set (the existing test hygiene entry is resolved; remaining items target dump, main or plan). No quality item silently dropped.
- Assumption impact: scoped:S2, seam-S2, closure DC-seam-S2-r3 implemented for Yield, **not marked resolved**. Claim 119 describes old code and needs post-change recheck; shared claim 35's aiohttp signal behavior is not altered. No invariant registry edit, decision reversal, or above-floor deferral.
- Concerns: process-level SIGINT/SIGTERM replay remains MS-002 QA/Yield Phase ownership; this Forward did not author or inspect its gate tests.
- Full hermetic suite: `uv run pytest` → **227 passed**, 25 pre-existing warnings, total **92.88%** coverage (server.py omitted by project configuration). `git diff --check` clean. Final `codify scan path ... --git --detailed`: production no violations; both new tests only one gray metric (L17, L27), test file yellow at 356 lines. `_cancel_background_task` final L10/N1/C3/P2; no new production function.
- Questions asked: none. Next: Yield T010.001, then MS-002 signal QA and fresh S2 recomposition.

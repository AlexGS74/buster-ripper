## Orchestrator Prompt (verbatim)

You are a subagent spawned by another session.
Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
Mode: scoped adhoc misread gate
Tasks: T009, T010

## Scope and result

Scoped clean-room pass over T009/T010 and affected G-005/G-002 goal shapes. Read only plan-summary.md, handoff-context.md, qa-plan.md and relevant code. No interview/decision/iteration report was read. Two plausible task misreadings were constructible; fewer than three because the handoff specifies the UTF-8 unit, inclusive comparison, task-exception boundary and cancellation/flush/close order concretely. Neither survives the explicit exclusions. Goal-shape rows below do not count toward the task ceiling.

| # | Surface (`G-*` or task) | Plausible misreading | Excluded by | Verdict |
|---|---|---|---|---|
| 1 | G-005 | A turn containing multibyte text is below threshold when counted as Python characters, so the artifact may wait for timer/shutdown despite meeting the configured byte volume. | plan-summary.md:19, :66 and :164 explicitly bind rendered-text UTF-8 bytes and threshold attainment; handoff-context.md:239-255 gives the non-ASCII separating case and inclusive `>=` comparison. The witness and met state cannot coincide under this measure; the documented-path vacuity is explicit. | Excluded |
| 2 | G-002 | A signal shutdown that flushes the artifact but exits with an ordinary background-task traceback could be classified as meeting “process exits,” because the goal observation surface only names directory contents. | qa-plan.md:53, :160-167 requires clean exit/exit code/no traceback, and handoff-context.md:299-302 explicitly forbids an ordinary task exception escaping cleanup. G-002's failure witness is illustrative rather than exhaustive; the QA surface supplies the missing exit observation for this scoped path. | Excluded |
| 3 | T009 | Count the encoded collapsed record (including JSON/diff and header) rather than the rendered body, or count Unicode code points because the existing comment says `len(_render_body(body))`. | handoff-context.md:239-255 specifies `len(_render_body(body).encode("utf-8"))` per turn, not record/JSON/code points, and immediate inclusive comparison; plan-summary.md:19/:164 states the same. The stale code comment at dump.py:37-38 is an implementation defect to correct, not plan authority. | Excluded |
| 4 | T010 | Catch only `CancelledError` when awaiting tasks, treating a task's already-stored `RuntimeError` as a shutdown failure that skips flush/close. | handoff-context.md:297-302 names exactly that counterexample and requires ordinary stored failures in either task logged/isolated, both cancellation attempts, final flush before session close and no ordinary exception escape. | Excluded |

## Goal-shape check

G-005's threshold pass and failure witness are separated by the encoded rendered-text count; no alternative count basis or undocumented-path observable is licensed. G-002's narrow failure witness does not enumerate the already-failed-task outcome, but its exit/traceback classification is bound by the scoped T010 contract and QA TC-110. Neither row leaves a materially different finished state unclassified by the forward-visible artifacts.

## Footnotes (Forward discretion, yield-reviewable)

None. RED: 2 task misreadings | 0 AMBIGUOUS | 0 discretion footnotes.

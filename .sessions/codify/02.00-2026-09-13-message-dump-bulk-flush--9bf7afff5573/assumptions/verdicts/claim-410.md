VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`asyncio.sleep(0)` yields control to the event loop, letting another ready task run

## What it actually does
The asyncio docs state that `sleep()` always suspends the current task, allowing other
tasks to run, and call out `delay=0` specifically as an "optimized path to allow other
tasks to run" — the documented use being to stop a long-running coroutine from occupying
the event loop for the whole duration of its call. Since asyncio uses cooperative
scheduling and runs one task at a time, suspending the awaiting task returns control to
the loop, which then runs other ready tasks (or callbacks/IO) before resuming it.
Note the guarantee is "allows other tasks to run", not "a particular task will run":
whether anything else executes depends on the loop having other ready work at that
iteration. Nothing in the claim requires more than that.

## Source
https://docs.python.org/3/library/asyncio-task.html#sleeping ("Sleeping" section,
`asyncio.sleep(delay, result=None)` entry; corroborated by the "Task Object" section,
"Event loops use cooperative scheduling: an event loop runs one Task at a time. While a
Task awaits for the completion of a Future, the event loop runs other Tasks, callbacks,
or performs IO operations.").

Exact fetched bytes: `/Users/alexhubitski/.local/share/opencode/tool-output/tool_113546a0a001D5622SMH2Jzups`
(the full `asyncio-task.html` page as fetched, markdown format).

Cross-checked against the same section in the repo's minimum supported release,
https://docs.python.org/3.11/library/asyncio-task.html#sleeping — identical wording
("`sleep()` always suspends the current task, allowing other tasks to run." /
"Setting the delay to 0 provides an optimized path to allow other tasks to run.").
Documentation-only claim: no local source root was supplied and none was needed;
the installed interpreter was not read or executed (P1).

## Source Version
Python 3.14.8 documentation (stable docs URL); corroborating page: Python 3.11.17
documentation. Repo declares `requires-python = ">=3.11"` in `pyproject.toml`, so the
claim holds across the supported version range as documented.

## Source Fingerprint
sha256:0e10a7cd1a7a3a44b1ac67d649a40b8ba34121acd69002a49012e515a85b8cee (full fetched page)
sha256:2f85eef501369e2d88a5bdc7a9f05659d1d57bb0580ce364a10a1a348c3a36dd (the `## Sleeping`
section, `awk '/^## \[Sleeping\]/,/^## \[Running tasks concurrently\]/'` on the fetched file)

## Invalidates On
- The "Sleeping" section of `Doc/library/asyncio-task.rst` changes the `delay=0` wording
  or removes the "optimized path to allow other tasks to run" statement.
- CPython changes `asyncio.sleep` so a zero delay no longer suspends the calling task
  (e.g. short-circuits without scheduling a loop callback) — a change in
  `Lib/asyncio/tasks.py` / the Task protocol would be the source to re-check.
- The fetched page content hash differs from the one recorded above.
- Support for Python < 3.11 is dropped or added such that the consulted doc releases no
  longer cover the deployed interpreter version.

## Checked At
2026-10-06

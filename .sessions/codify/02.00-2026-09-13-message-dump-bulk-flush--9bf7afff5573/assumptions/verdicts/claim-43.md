VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
asyncio provides no built-in periodic-timer/scheduler facility — a periodic loop is hand-built around `asyncio.sleep`

## What it actually does
Both halves checked independently against the installed library. (1) asyncio's entire callback-scheduling surface is single-fire: `AbstractEventLoop` exposes only `call_soon` (defined as `self.call_later(0, ...)`), `call_soon_threadsafe`, `call_later`, and `call_at`; `call_later` resolves to one `TimerHandle` pushed on a heap, and its docstring states "Each callback will be called exactly once." A package-wide search of all 35 `.py` files under the installed `asyncio/` for periodic/repeat/recurring/interval/every/cron/call_every/call_periodic finds no API — only prose in comments and docstrings. The official version-pinned docs agree: the High-level API Index (Tasks/Queues/Subprocesses/Streams/Synchronization/Exceptions) and the Low-level API Index's "Scheduling Callbacks" section (`loop.call_soon()`, `loop.call_soon_threadsafe()`, `loop.call_later()`, `loop.call_at()`) list no periodic timer or scheduler facility. (2) With no built-in facility, coroutine-based periodic execution must be hand-built, and asyncio's only awaitable timing primitive for coroutines is `asyncio.sleep` ("Coroutine that completes after a given time", itself a one-shot await over `loop.call_later`), so the standard hand-built periodic pattern is the sleep loop (`while True: await asyncio.sleep(interval)`). Corroboration note, not a reading choice: a periodic callback can also be hand-built by chaining `loop.call_later`, which does not contradict the claim — the claim asserts no built-in facility and the sleep loop as the characteristic hand-built pattern, not exclusivity of the sleep loop.

## Source
Installed asyncio package source, Python 3.14.3 (homebrew): `.../lib/python3.14/asyncio/events.py` (AbstractEventLoop scheduling methods at lines 305–334; `TimerHandle` at line 141), `asyncio/base_events.py` (`call_later` at lines 775–797, "Each callback will be called exactly once"; `call_at` at 799–815), `asyncio/tasks.py` (`sleep` at lines 687–704), `asyncio/__init__.py` (`__all__` aggregate at lines 26–42), `asyncio/timeouts.py` (`__all__` = Timeout/timeout/timeout_at — deadline facility, not periodic). Corroborating docs: https://docs.python.org/3.14/library/asyncio-api-index.html and https://docs.python.org/3.14/library/asyncio-llapi-index.html ("3.14.7 Documentation", "Scheduling Callbacks" lists only one-shot methods).

## Source Version
CPython 3.14.3 installed source (homebrew Cellar `python@3.14/3.14.3_1`); corroborating docs.python.org pages served as "3.14.7 Documentation".

## Source Fingerprint
sha256:b1a138015c6e1878d7f8c6be51896da451d7f25c24f6ef18d76ceff249050d94  asyncio/events.py
sha256:a82285b1ccb379d4bc18f3c1b05a2876040daed1523e7fb6bc05c0a86977894d  asyncio/base_events.py
sha256:8554946e93b93d70288a2ab0ad7f255cf3119c4861f89e3c9194956d0f3f591a  asyncio/tasks.py
sha256:8148552daea881e3b33639d6e9bdab0da2021367ab694ee0604431c33c16983f  asyncio/__init__.py
sha256:3154c617c94bf0124b0b5eaf1a6aef89cc0776e95d00b5e4e5302e32b29cb06b  asyncio/timeouts.py

## Invalidates On
- A Python release whose asyncio adds a periodic/recurring timer or scheduler API (e.g., `loop.call_every`, `asyncio.repeat_every`, or a built-in periodic scheduler class) — the package-wide search would then match an API.
- A different installed CPython version or patch level (re-read and re-hash the `asyncio/` package source; this verdict is scoped to 3.14.3).
- A restructure of the docs.python.org asyncio API index that adds a periodic-timer section under "Event Loop Methods".

## Checked At
2026-09-19

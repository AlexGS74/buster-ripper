VERDICT: SAFE
PROMOTION: INELIGIBLE — supplied session verdicts do not provide permanent handles for all member claims; several server.py member fingerprints predate the currently consulted source bytes.

## Seam
S2 — SIGINT/SIGTERM through aiohttp runner cancellation and cleanup into background-task settlement and the pending dump batch artifact. Members in ledger order: 14, 35, 123, 124, 125, 16, 96, 113, 114, 126, 118, 127, 128, 97, 95, 104, 120, 121, 122, 36.

## Composition
`main.main` delegates to `web.run_app` without installing package signal handlers. Where supported, aiohttp installs handlers that raise `GracefulExit`; `run_app` cancels and awaits `_run_app`, whose `finally` awaits runner cleanup, including the registered `on_cleanup`. Startup placed the monitor and periodic flush tasks in distinct app keys. Cleanup sequentially cancels/awaits the monitor, then the periodic flush task, calls `dump.flush_all()` under `_final_dump_flush`, and only then closes the shared session. The helper snapshots the cleanup task's cancellation count before each child cancellation/await: a child `CancelledError` with unchanged count is suppressed; an increased count is re-raised. A completed child with an ordinary stored exception cannot be turned into a cancelled task by `cancel()` and its exception is instead logged/suppressed by the helper. With a fixed non-`documented` format and a nonempty pending batch, `_flush_state` appends the joined records to the format's `.log` in one write if the target is writable and I/O succeeds. The `documented` path does not buffer turns, and `_flush_state` returns before an artifact append. Ordinary exceptions from `flush_all()` are logged/suppressed by `_final_dump_flush`; other cleanup-hook ordinary exceptions can escape `run_app`.

## Dimension Coverage
- Single signal / retained runner count: the cancellation that enters `_run_app`'s `finally` leaves a cancellation-request count on that same task; delivery alone does not reset it. Each helper snapshots that count, so a child cancellation without a *new* request does not satisfy the increased-count predicate. Both task positions can settle and the later flush/close sequence remains reachable.
- Fresh cancellation during either child await: cancellation of the cleanup task increments its count relative to that child's snapshot, so a caught `CancelledError` is re-raised. The sequential cleanup may then stop before the next child, shutdown flush, or session close; the sources do not promise completion under a second interruption.
- Completed ordinary failure versus active child cancellation: cancelling a completed failed task returns false, its await raises the stored ordinary exception, and the helper logs/suppresses it. Cancelling an active child can make its await raise `CancelledError`; without a newly increased cleanup count the helper suppresses it. Both positions take the same helper path.
- Pending batch / format / target: for fixed `change-focused` or `self-contained` format, pending records are joined and written to that format's artifact on successful writable-target/I/O path; failed target/I/O retains pending and the per-session flush logs the failure. For fixed `documented`, turns bypass pending buffering and the flush guard creates no collapsed artifact. The documented guard would retain preexisting pending records, but production evidence for fixed documented turns does not create those records; changing format is outside the fixed-format witness.
- Exception boundary: ordinary background failure does not skip subsequent cleanup; ordinary shutdown-flush failure is logged, and session close is still attempted. A fresh outer cancellation or an ordinary exception from another cleanup hook can interrupt/escape aiohttp's cleanup. No incompatible composition found within the listed dimensions.

## Counterexample
None found within the fixed-format, single-signal successful-cleanup path. The fresh-cancellation and failed-I/O paths above are conditional limits, not claims of unconditional artifact delivery.

## Sources
- `buster_ripper/main.py:157-160`
- `buster_ripper/server.py:25-45,55-111`
- `buster_ripper/dump.py:83-125,150-156,189-195,269-305`
- `.venv/lib/python3.13/site-packages/aiohttp/web.py:325-331,421-425,452-521`
- `.venv/lib/python3.13/site-packages/aiohttp/web_runner.py:278-320,386-399`
- `/opt/homebrew/Cellar/python@3.14/3.14.7/Frameworks/Python.framework/Versions/3.14/lib/python3.14/asyncio/tasks.py:183-274`
- `/opt/homebrew/Cellar/python@3.14/3.14.7/Frameworks/Python.framework/Versions/3.14/lib/python3.14/asyncio/futures.py:195-209,292-300`

## Source Versions
Repository production sources: unavailable. Installed aiohttp: 3.13.3. Installed CPython standard library: 3.14.7.

## Source Fingerprint
unavailable (no ordered permanent member handles supplied). Consulted full-file SHA-256: main.py `cbe2802d474abe21ada5bb2e94c09cce15237ea53d526c6d31fed25207c950d1`; server.py `6a3e45ef69413838dc53de2939ff1bb06b268746231bf63f3589be407f3d3a5a`; dump.py `07b0b076552f72e62c625caade1b01821b3237375900233812d14c0f275b3c2b`; aiohttp/web.py `27348d99e8e0e46e987850279088197f2b6a6d4f3ab0dbbce38c982a6a14a6ab`; aiohttp/web_runner.py `bf51b59ca88e3908059d3491e0831ce88f5178414331a1ed30bbcefeba0333e0`; asyncio/tasks.py `a97ffdb60a42104a86f2ece781878b3371bd88e9eea8f1ac1fa8f397cc11cca5`; asyncio/futures.py `2f482524292a7c5d1b818aeb92434d183bd089e014cf50b785432284d2c8bc12`.

## Invalidates On
Any change to a listed member verdict or its source fingerprint; changes to signal registration, aiohttp cancellation/cleanup, Python task cancellation/await semantics, either task's startup or cleanup settlement, cleanup ordering/exception handling, dump format dispatch, pending buffering, target guard, or batch write; a format switch that makes `documented` state contain pending collapsed records; changed runtime/library versions. Recompose from all dimensions when member evidence is stale rather than carrying an old dimension forward.

## Checked At
2026-09-27

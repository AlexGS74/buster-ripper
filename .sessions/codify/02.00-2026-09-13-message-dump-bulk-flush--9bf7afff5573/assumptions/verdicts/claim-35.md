VERDICT: HOLDS
PROMOTION: INELIGIBLE — establishing both signal installation and cleanup requires two installed source files, not one exact source locator.

## Claim
`web.run_app` handles SIGINT/SIGTERM with a graceful shutdown that runs the app's cleanup hooks before the process exits

## What it actually does
With default `handle_signals=True` on platforms supporting asyncio signal handlers, the runner installs SIGINT/SIGTERM handlers; `run_app` catches their `GracefulExit`, cancels its main task, and `_run_app` awaits runner cleanup, which invokes application cleanup hooks before `run_app` returns. This is an attempted graceful shutdown, not a guarantee that failing or interrupted hooks finish successfully; platforms without `add_signal_handler` support are excepted.

## Source
Installed aiohttp `.venv/lib/python3.13/site-packages/aiohttp/web.py` lines 421–425, 452–521; `.venv/lib/python3.13/site-packages/aiohttp/web_runner.py` lines 35–40, 278–289, 295–320, 386–399.

## Source Version
aiohttp 3.13.3

## Source Fingerprint
sha256:27348d99e8e0e46e987850279088197f2b6a6d4f3ab0dbbce38c982a6a14a6ab (web.py); sha256:bf51b59ca88e3908059d3491e0831ce88f5178414331a1ed30bbcefeba0333e0 (web_runner.py)

## Invalidates On
Change to aiohttp signal-handler registration or cleanup implementation in web_runner.py.
Change to aiohttp run_app cancellation/finally handling in web.py.

## Checked At
2026-09-27

VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
The `buster_ripper` package installs no SIGINT/SIGTERM handler of its own

## What it actually does
Its CLI delegates signal handling to `aiohttp.web.run_app`; the package does not register its own SIGINT/SIGTERM handlers. Its application registers an aiohttp cleanup callback, which is not a signal handler.

## Source
`buster_ripper/main.py:157-160` (CLI hands control to `web.run_app`; inspection of production modules finds no signal registration).

## Source Version
unavailable

## Source Fingerprint
sha256:cbe2802d474abe21ada5bb2e94c09cce15237ea53d526c6d31fed25207c950d1

## Invalidates On
Change to `buster_ripper/main.py` or another production module that adds SIGINT/SIGTERM registration.

## Checked At
2026-09-27

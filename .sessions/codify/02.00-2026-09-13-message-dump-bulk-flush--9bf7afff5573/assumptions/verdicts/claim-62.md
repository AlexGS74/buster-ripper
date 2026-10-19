VERDICT: UNCLEAR
PROMOTION: INELIGIBLE — the claim admits two different readings of “awaits `backend.detect_backend` (the `/v1/models` probe)”; neither unambiguously establishes a single verdict.

## Claim
`on_startup` awaits `backend.detect_backend` (the `/v1/models` probe) during startup, before the app serves requests

## What it actually does
`on_startup` unconditionally awaits `detect_backend` as an aiohttp startup callback, before serving requests. If “the probe” means actually issuing the GET, that occurs only without a CLI override or cached result; if it means awaiting the detection function that can probe, it does occur on every startup. The claim needs to distinguish these readings.

## Source
`buster_ripper/server.py:55-65,102-106`; `buster_ripper/backend.py:41-54` (override/cache short circuits); aiohttp Application startup lifecycle: https://docs.aiohttp.org/en/stable/web_reference.html#aiohttp.web.Application.on_startup

## Source Version
Repository source revision unavailable; aiohttp documentation: stable (installed version unavailable).

## Source Fingerprint
sha256:fbd579a7995d0f4a2ae8a6d2994bb9e5fda344c1a900ec581caee38f599647f8 (`buster_ripper/server.py`); sha256:145f150cbcceda7ca8f7715d8ff14d9c779aace39406affce86982b2dcbf1034 (`buster_ripper/backend.py`); fetched documentation not fingerprinted.

## Invalidates On
Changes to `buster_ripper/server.py` startup callback registration or detection await.
Changes to `buster_ripper/backend.py` override/cache branches.

## Checked At
2026-09-27

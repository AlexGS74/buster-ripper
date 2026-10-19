VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
The `/v1/models` GET in `detect_backend` passes `aiohttp.ClientTimeout(total=5)` as its per-request timeout

## What it actually does
The probe calls `session.get(f"{backend_url}/v1/models", timeout=aiohttp.ClientTimeout(total=5))`.

## Source
`buster_ripper/backend.py:53-56`

## Source Version
unavailable

## Source Fingerprint
sha256:c2b71468af6b53efde753b0648d47eb6e06c8db4c738e08f7c01f4e4682d76ef

## Invalidates On
Modification of `buster_ripper/backend.py`, particularly the probe GET timeout.

## Checked At
2026-09-28T00:15:27Z

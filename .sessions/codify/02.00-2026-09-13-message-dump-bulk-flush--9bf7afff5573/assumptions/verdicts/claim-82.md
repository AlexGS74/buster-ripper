VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`detect_backend` issues a `/v1/models` GET when neither a CLI override nor a cached backend is available

## What it actually does
When the truthy CLI override and cached-state early returns do not apply, the function calls `session.get(f"{backend_url}/v1/models", ...)` in its probe `try` block.

## Source
`buster_ripper/backend.py:41-56`

## Source Version
unavailable

## Source Fingerprint
sha256:c2b71468af6b53efde753b0648d47eb6e06c8db4c738e08f7c01f4e4682d76ef

## Invalidates On
Changes to the early returns or probe call in `buster_ripper/backend.py`.

## Checked At
2026-09-28T00:13:19Z

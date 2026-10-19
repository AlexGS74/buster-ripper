VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`detect_backend` returns a CLI backend override without issuing a `/v1/models` GET

## What it actually does
When `cli_override` is truthy, it stores and returns that value before reaching the `session.get` probe.

## Source
`buster_ripper/backend.py:41-55`

## Source Version
unavailable

## Source Fingerprint
sha256:c2b71468af6b53efde753b0648d47eb6e06c8db4c738e08f7c01f4e4682d76ef

## Invalidates On
Changes to the CLI override branch or probe ordering in `buster_ripper/backend.py`.

## Checked At
2026-09-28T00:13:19Z

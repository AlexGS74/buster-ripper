VERDICT: MALFORMED
PROMOTION: INELIGIBLE — the claim requires checking the meaning of T008 in the prohibited session directory.

## Claim
`handle_messages` dispatches on `body.get("stream", False)` (`messages.py:65-68`) between a streaming and a non-streaming handler — the mirror for T008's branch

## What it actually does
The production source shows the dispatch, but establishing that this is the mirror for T008's branch requires reading session material. MALFORMED per P2 and P3; a partial verdict on dispatch alone would not settle the whole claim.

## Source
`buster_ripper/routes/messages.py:65-70`; T008's meaning is not available from the permitted source.

## Source Version
unavailable

## Source Fingerprint
unavailable

## Invalidates On
Re-extraction of the claim without the T008-dependent clause.

## Checked At
2026-09-27

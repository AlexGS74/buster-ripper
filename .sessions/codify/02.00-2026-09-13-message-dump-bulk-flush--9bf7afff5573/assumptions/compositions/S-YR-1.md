VERDICT: MALFORMED
PROMOTION: INELIGIBLE — member-state guard failed: `YR-P011` is `DOES-NOT-HOLD`, so composition ran before atomic resolution; no seam trace was performed and no promotion fingerprint exists.

## Seam
`S-YR-1` — `YR-P011` → `YR-P012` → `YR-P013` (normalization step → digest key → estimate source).

## Composition
Not performed. Composition requires every supplied member verdict to start with `VERDICT: HOLDS`; a `DOES-NOT-HOLD` member is routed before composition and cannot be smuggled in as an alternate premise. Member `YR-P011` starts with `VERDICT: DOES-NOT-HOLD` (line 1 of `assumptions/verdicts/yr-p011.md`), so the guard stopped the pass before the boundary value was traced from producer through any transformation to the consumer.

Blocking member: `YR-P011` — its claim states that `_normalize_system`'s only change to a body's `system` text is removing the billing nonce; its own verdict records that `_strip_billing_from_blocks` unconditionally trims every text block and drops text blocks that become empty after trimming, so `system` can change without any nonce being removed. That is the first link of this seam (normalization step), and it does not hold.

## Dimension Coverage
No dimensions enumerated or traced. No dimension was carried forward from a prior composition.

## Counterexample
Not reached — composition did not run.

## Notes
Supplied member states, verbatim first lines:
- `assumptions/verdicts/yr-p011.md` — `VERDICT: DOES-NOT-HOLD` (blocking member).
- `assumptions/verdicts/yr-p012.md` — `VERDICT: HOLDS`.
- `assumptions/verdicts/yr-p013.md` — `VERDICT: HOLDS`.

The two holding members do not permit a partial or reduced-scope chain: the protocol forbids omitting a member claim or substituting an alternate premise, and the seam's first link is the member that failed. Adjudication of `YR-P011` must precede any composition of this seam. This file records only the malformed state; it makes no finding about the design.

## Sources
None consulted. No production source was read for this pass, because the member-state guard failed before any trace.

## Source Versions
Not applicable — no production sources consulted.

## Source Fingerprint
unavailable

## Invalidates On
- A fresh verdict for `YR-P011` that starts with `VERDICT: HOLDS` (and is current relative to `normalize.py`), together with current `HOLDS` verdicts for `YR-P012` and `YR-P013`, permitting a full composition of this seam.
- A re-declared seam `S-YR-1` whose member set does not include a `DOES-NOT-HOLD` claim.

## Checked At
2026-10-17T22:39:00Z

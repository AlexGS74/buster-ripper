VERDICT: MALFORMED
PROMOTION: INELIGIBLE — member `YR-P009` resolved `DOES-NOT-HOLD`; the seam cannot be composed from the supplied atomic materials and must be routed before recomposition.

## Seam
`S-YR-3` — boundary value: the base app's effective HTTP request-body limit (`YR-P009`) feeding the package app's limit (`YR-P010`).

## Composition
Not traced. Precondition failure: the member-state guard requires every supplied session verdict to begin with `VERDICT: HOLDS`. `YR-P009` begins with `VERDICT: DOES-NOT-HOLD`, so composition stopped before any producer → transformation → consumer trace was attempted. Per the protocol, a `DOES-NOT-HOLD` claim is routed before composition and cannot be smuggled in as an alternate premise; no re-check, re-wording, or repair of the member claim was performed.

Failing member and the measured fact that falsified it:

- Member: `YR-P009` — "In `buster_ripper_legacy.py`, the aiohttp application is created/run without setting `client_max_size` … so base's effective body limit is aiohttp's default (1 MiB) or an explicit finite value."
- Measured fact: `buster_ripper_legacy.py` contains no aiohttp application at all. It constructs a **FastAPI** app (`app = FastAPI(lifespan=_lifespan)`, line 237) and serves it with `uvicorn.run(app, host=host, port=port, log_level="warning")` (line 1009); it imports `fastapi`, `uvicorn`, and `httpx` and never `aiohttp`, and the string `client_max_size` appears nowhere in the file. Base is therefore FastAPI/uvicorn, not aiohttp, so no aiohttp default (1 MiB or otherwise) is in play — the premise that the base limit is an aiohttp `client_max_size` value is false, and the claimed producer of the seam's boundary value does not exist.
- Cited source: `/Users/alexhubitski/Projects/buster-ripper/buster_ripper_legacy.py` — imports lines 84–87, app creation line 237, server run line 1009; fingerprint `sha256:7967563dc3ab4f66f0dc322c7945c5647b96c9f9a77fbafe3cd1d4780e8764bd`.

`YR-P010` was not composed either: it is `HOLDS` (`create_app()` passes `client_max_size=0` to `web.Application` in `buster_ripper/server.py`, line 125), but a single falsified member is sufficient to malform the seam. Its verdict is recorded here only as supplied state, not as a composed premise.

## Dimension Coverage
None. No dimension was enumerated or traced; the member-state guard fails before the seam trace begins. The dimensions implied by the seam row (limit value produced at base, any transformation or normalization en route, limit consumed by the package app) are all untraced.

## Counterexample
None found — not assessed. A counterexample would presuppose a composed boundary value, and the producer premise is falsified rather than merely surprising.

## Notes
The seam as written pairs a base-side limit that does not exist in the cited base source with a package-side limit that does exist. Whatever the intended comparison was, it is not recoverable from these atomic materials without re-resolving `YR-P009` against the actual base framework (FastAPI/uvicorn body handling) — which is outside this pass and was deliberately not performed.

## Sources
- `/Users/alexhubitski/Projects/buster-ripper/buster_ripper_legacy.py` (via `assumptions/verdicts/yr-p009.md`)
- `/Users/alexhubitski/Projects/buster-ripper/buster_ripper/server.py` (via `assumptions/verdicts/yr-p010.md`)

## Source Versions
- `buster_ripper_legacy.py`: unavailable — no version string in the file (1013 lines); local worktree copy
- `buster_ripper/server.py`: no version string in the source; worktree file as supplied at check time

## Source Fingerprint
unavailable — composition did not complete; no ordered member-handle digest was computed because promotion is ineligible. Member fingerprints carried as supplied: `YR-P009` = `sha256:7967563dc3ab4f66f0dc322c7945c5647b96c9f9a77fbafe3cd1d4780e8764bd`; `YR-P010` = `sha256:b3b486615ca797febfa86795e0e18a04e4355aece31c43305a8d40bcad0e28d5`.

## Invalidates On
- `YR-P009` is re-resolved to `HOLDS` against the base framework actually in use (FastAPI/uvicorn), or the seam row is rewritten so its base member describes a body limit that exists in `buster_ripper_legacy.py`.
- Either member verdict file is replaced or its `Checked At` advances.
- Either member source changes so its sha256 no longer matches: `buster_ripper_legacy.py` (gains an aiohttp application, a `client_max_size` assignment, or replaces the `uvicorn.run(app, ...)` call at line 1009) or `buster_ripper/server.py` (`create_app()` stops passing `client_max_size=0`).

## Checked At
2026-10-17T22:35:00Z

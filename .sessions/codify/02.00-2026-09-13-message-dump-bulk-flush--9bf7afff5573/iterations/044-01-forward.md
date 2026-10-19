## Orchestrator Prompt (verbatim)

> You are a subagent spawned by another session.
> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
> Task ID: T007

## Increment Brief

- T007: bound uncached `/v1/models` detection to five seconds without changing priority or cancellation.
- Anchors touched: shared client construction → `server.py:on_startup` reuse unchanged (600s default); T007 handoff binding → `backend.py:detect_backend` extend.
- Will NOT create: another client, connector, detection cache, probe handler, or startup lifecycle.
- Reuse evidence: no touched NEW anchor. Inventions: none; `_classify_models` is a private extraction of the existing inline classification concern.
- VP handles: no T007-specific VP in the original matrix; bounded startup is a local T007 regression, with MS-002 boundary re-verdict pending.

## Failure-Path Evidence

| Operation / condition | Expected outcome and authority | Handling / observation evidence | Disposition |
|---|---|---|---|
| Uncached probe, stalled upstream | Five-second per-request deadline, then cached `unknown` (T007, S8) | Real stalled aiohttp peer with shared 600s default; elapsed 4.5–8s and unknown state in `test_stalled_probe_uses_five_second_request_deadline` | safe |
| Uncached probe, HTTP/connect failure | Ordinary `Exception` falls back to `unknown` (claim-60 corrected by claim-77) | Existing failing-server and connection tests; `except Exception` unchanged | safe |
| In-flight cancellation | Propagate `CancelledError`; no false `unknown` cache (claim-78, S8) | Real in-flight request cancelled in `test_cancellation_does_not_cache_unknown`; state remains uncached | safe |
| Override or previously cached backend | No GET, priority unchanged (claim-62 corrected by claims 80–82) | Parametrized `test_short_circuits_do_not_probe` with request spy; existing override/cache tests | safe |
| Unrecognized or empty model response | Existing `unknown` warning and caching retained | Existing `test_auto_detect_unknown_backend`, `test_empty_models_returns_unknown`; characterization suite passed before and after extraction | safe |

Out of scope: malformed model payload and independent startup session cleanup; neither is changed by the per-request deadline. No retry or new fallback policy.

## Implementation

- `backend.py:detect_backend` passes `aiohttp.ClientTimeout(total=5)` directly to the conditional GET; the shared 600s session, override/cache short circuits, normal probe fallback, and cancellation propagation remain intact.
- Preventive capacity extraction: moved inline model classification to private `_classify_models` before adding the timeout. This is the same existing classification concept, not a new backend adapter; warnings and precedence are unchanged.
- `tests/test_backend.py`: real stalled-peer timeout and cancellation tests, plus parametrized no-probe priority witnesses. No persistent QA gate authored or inspected.
- Assumption impact: claim-60, claim-62, seam-S8 implemented for Yield, not marked resolved. The dependency premise (aiohttp per-request timeout replaces the session default) is single-task claim-57, confirmed in installed `aiohttp/client.py:584-590`; no correction to a shared claim.

## Validation

- Baseline capacity: `backend.py` 96 raw lines; `detect_backend` L36/N3/C7 (hard L/N, gray C), so additive pressure triggered extraction. Before extraction the 14 existing backend cases passed (focused run's project-wide coverage gate failed at 12% solely because the narrow file was selected). After extraction `uv run pytest tests/test_backend.py -q -o addopts=''` → 14 passed; after feature and test additions → 18 passed, 20 pre-existing bare-handler warnings.
- The first new tests failed on a test URL that erroneously truncated a port digit; corrected it before the timeout edit. A true pre-feature stalled test would have waited for the 600s session deadline, so no false fail-first claim is made. The actual stalled request now takes approximately 5 seconds and returns `unknown`.
- Full hermetic suite: `uv run pytest` → 242 passed, 23 warnings (pre-existing bare-handler/coroutine warnings), total coverage 92.90%; `backend.py` 39/39 statements covered (100%). Final focused run: 18 passed.
- LNCP: `codify scan path buster_ripper/backend.py --git --detailed` → `detect_backend` L23/N1/C4/P3 (single gray L), `_classify_models` within ideal thresholds (L≈11/N0/C3/P1). `tests/test_backend.py` scan → two single-gray tests (N2 cancellation, P4 priority); no compound violation. Test file 258→329 lines, yellow review band, below 500. `backend.py` 96→95 raw lines; `git diff --check` clean.
- Capacity gate: triggered at hard L/N; classification extracted and characterized before feature, final `detect_backend` L23/N1/C4, file shrank one raw line. Backlog pickup: none match touch-set.
- Durable signal: no override exercised, no decision reversal, no invariant registry edits, no above-MVP deferral. T007 remediation resolves startup latency locally; MS-002 process QA/re-verdict remains separate.
- Questions asked: none. Next: local Yield for T007.

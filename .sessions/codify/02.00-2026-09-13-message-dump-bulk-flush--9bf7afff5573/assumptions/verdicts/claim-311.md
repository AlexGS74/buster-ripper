VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`time.time` reflects the system wall clock and can return a smaller value after a clock adjustment

## What it actually does
Both halves are stated by CPython's own documentation. `time.time()` reads the system-wide real-time (wall) clock — its "Clock:" section says it calls `clock_gettime(CLOCK_REALTIME)` where available, otherwise `gettimeofday()` on Unix and `GetSystemTimePreciseAsFileTime()` on Windows — and the docs say explicitly: "While this function normally returns non-decreasing values, it can return a lower value than a previous call if the system clock has been set back between the two calls." The same page contrasts this with `time.monotonic()` ("a clock that cannot go backwards. The clock is not affected by system clock updates"), and notes `CLOCK_REALTIME` "may have discontinuities if the time is changed using `settimeofday()` or similar."

Nuances the claim does not disturb: leap seconds are platform dependent and are normally not counted (Unix time), and precision may be no better than 1 second on some systems. "Normally non-decreasing" means a backward step is possible, not that it is frequent — the adjustment must actually move the clock backwards between calls (a step backward, or an NTP step; ordinary forward adjustment keeps values increasing).

## Source
https://docs.python.org/3.14/library/time.html#time.time — section `time.time()`, paragraphs "While this function normally returns non-decreasing values…" and the "Clock:" list. Corroborated on the same page: `time.monotonic()` (cannot go backwards, unaffected by system clock updates), `time.CLOCK_REALTIME` ("Real-time clock. … The clock is the same for all processes"), `time.clock_settime()` (sets `CLOCK_REALTIME`), `time.CLOCK_BOOTTIME` (`CLOCK_REALTIME` "may have discontinuities if the time is changed using settimeofday() or similar"). Consulted as the fetched HTML of that page (`time-314.html`), byte-identical to `https://docs.python.org/3/library/time.html`.

## Source Version
Python 3.14.8 documentation (`Doc/library/time.rst`, branch 3.14); page "Last updated on Oct 05, 2026 (10:27 UTC)". Fetched 2026-10-05. Not checked against an installed interpreter's own docs — this is a documentation-only claim about standard-library semantics.

## Source Fingerprint
sha256:34c3ef44481e8b7818876e3b4e9cb415a5d085ce685c4e6d8c78f570fcdd3adf — the exact fetched bytes of the consulted page

## Invalidates On
- The fetched page's `time.time()` section changes sha256 (i.e. `Doc/library/time.rst` wording about the backward system clock or the "Clock:" call list changes) on the 3.14 branch.
- CPython re-routes `time.time()` away from `CLOCK_REALTIME`/`gettimeofday()`/`GetSystemTimePreciseAsFileTime()` to a monotonic source, or the docs drop the set-back caveat.
- Applying this verdict to a different documented version than 3.14.8 without re-reading that version's page.
- Reusing it for a non-Wall-clock sibling (`time.monotonic()`, `time.perf_counter()`, `time.process_time()`) — those are documented as unaffected by system clock updates.

## Checked At
2026-10-05T18:59:23Z

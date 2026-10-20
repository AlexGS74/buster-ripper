# Plan — 02.00-2026-09-13-message-dump-bulk-flush

## Session Metadata

- Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
- Session UID: 9bf7afff-5573-402a-b102-827423a1dded
- Topic: Buffer captured messages in memory and flush the dump to disk in bulk; flush on Ctrl-C/SIGTERM before exit
- Branch: refactor/package-structure
- Question Mode: deferred
- Created: 2026-09-13
- Last Updated: 2026-10-19 (PR #1 merged; session closed and archived)
- **Status**: COMPLETED → SQUASHED

## Problem

`buster-ripper` intercepted every `/v1/messages` request and wrote the dump to disk
immediately: one turn file (`turn_000.json` / `turn_{N:03d}.diff`) plus an `index.txt` append,
i.e. two write-bearing operations per turn. The user asked to collect captured messages in
memory and write them in bulk, reducing the number of writes, and to flush the buffered dump on
Ctrl-C and SIGTERM before the process terminates. Durability of in-flight messages was explicitly
waived (a hard shutdown may lose them). The user later added a second requirement: the documented
per-turn layout must not be deleted until the new behavior is implemented, demonstrated better, and
judged worth switching to — and the existing layout must stay the default meanwhile.

## Goals

| Goal | Outcome | Met when | Failure witness |
|---|---|---|---|
| G-001 | Captured messages are held in memory and written in bulk instead of once per request | For N turns flushed F times **with a collapsed format selected**, write-bearing operations `= F` versus `2N`; reduction `1 − F/(2N)` in `[50%, 100%)`; the artifact is append-only, so bytes grow with `F`, not `N²` | **A**: a flush writes one file per pending turn (`N + F`). **B**: a flush rewrites the whole artifact (`O(N²)` bytes). **C**: both writers run for the same turn (`2N + F`) |
| G-002 | Buffered dump reaches disk on SIGINT/SIGTERM | After SIGINT or SIGTERM, buffered turns are on disk in order and the process exits cleanly | Process exits cleanly but the buffered turns are missing |
| G-003 | `/v1/messages` behavior is unchanged | Existing suite green; a raising dump/flush still yields the upstream response | A request fails because of dump/buffer/flush behavior. **Not a latency bound** |
| G-004 | Periodic flush bounds staleness while the process runs | A turn appended at `t` reaches the artifact by `t + I + ε` with the process still running | The timer runs but the artifact never gains the turn |
| G-005 | Volume-triggered flush | Reaching the configured pending-byte threshold puts the turns on disk with no shutdown and no timer | Turns accumulate past the threshold with nothing on disk |
| G-006 | The dump stays readable (function, not layout) | A reader inspecting only `DUMP_DIR/<sid>/` can locate a changed turn and identify what changed, per shipped format | The artifact is well-formed but a reader cannot find the changed turn |
| G-007 | The upstream's certificate is verified unless the opt-out is explicitly configured; the opt-out is process-wide and defaults off | With no opt-out an untrusted-certificate HTTPS upstream fails; with it the same peer completes | **W1**: a self-signed upstream succeeds with no opt-out. **W2**: the flag sets its global but the connector is built without it |
| G-008 | The documented layout is unaffected and its displacement is gated on a demonstration the user judges worth switching to | With no format configured, N turns produce exactly `turn_000.json`, `turn_{N:03d}.diff`, `index.txt` with today's content at `2N` writes; a comparison record exists at `MS-002` on M1 (write-bearing operations) and M2 (files created) | **W1**: a default run writes the collapsed artifact. **W2**: the documented files or `index.txt` line format drift. **W3**: no comparison record exists anywhere |

**Derived constraint — at most one write path runs per turn.** For any configuration a session of N
turns performs **either** `2N` **or** `F` write-bearing operations, never `2N + F`. This is arithmetic
(`2N + F > 2N` for all `F ≥ 1`) plus two normative sources (`FK-002`'s "one bulk flush is ONE write"
and `US-012`), not a preference. It forces the format selector to choose which writer runs.

## Settled options

| Option | Flush trigger | Flush payload | Write reduction | Status |
|---|---|---|---|---|
| 1 — both triggers + documented layout | time **and** volume **and** shutdown | documented per-turn files | `2N → N + F` | **payload rejected** (`FK-002`) |
| 2 — shutdown only + documented layout | shutdown only | documented per-turn files | `2N → N + 1` | **rejected** (`FK-001`) |
| 3 — both triggers + collapsed single file | time **and** volume **and** shutdown | one collapsed per-session artifact, appended once per flush | `2N → F` **when selected** | **SELECTED** (`FK-002`); addition (`FK-007` option 1). The collapsed formats are **opt-in**; the default is the documented layout |

The closed format set is `documented`, `change-focused`, `self-contained`, and the default is
`documented` (`US-012`). The reduction is therefore **opt-in**: a default run buffers nothing, starts
no timer and performs no shutdown flush. This is the direct consequence of `US-012`'s gating clause, not a
silent decision. `RM-013` carries the future default switch, gated on the user's judgment of the `MS-002`
comparison.

## Constraints

- **Write reduction**: one collapsed flush costs one write regardless of how many turns it carries; `F` is a
  per-`(sid, format)` count. With no format configured the count is `2N`, not `F`.
- **Append-only**: a rewrite-per-flush artifact is excluded by `G-001` witness B, not only by the count.
- **Coverage**: `pyproject.toml` gates the suite at 80% and `dump.py` is not omitted. The timer loop body
  lives in `dump.py` (tested), not `server.py` (coverage-omitted).
- **Capacity**: `dump_turn` was a hard LNCP length violation, so `T002` extracted the record builders, the
  documented writer, format dispatch and the artifact path before adding behavior. `main.py:main` was a hard
  parameter/length violation, so `T006` extracted `_apply_cli_config` and wired options through callbacks.
- **No new runtime dependency**: `aiohttp` and `click` only.
- **`G-003`**: no new endpoints, no normalization changes, no latency bound. The append-time flush runs
  synchronously inside `dump_turn` by design.

## Delivered milestones

| Milestone | Constituent tasks | Complete when | Validation points |
|---|---|---|---|
| MS-001 | T001, T002, T005 | A documented-format session produces today's three files with today's content at `2N` writes and its README workflow is unchanged; **and** a run with no format configured produces the documented layout; **and** with a collapsed format selected N turns produce no artifact before a flush; a raising dump/flush cannot fail a request | `VP-T001-writes-deferred`, `VP-T001-dump-failure-isolated`, `VP-T002-documented-format-preserved`, `VP-T002-default-encoding` |
| MS-002 | T003 | A collapsed-format session's turns reach disk only after a volume-triggered, timer or shutdown flush, in arrival order, turn 0's body present, write count `F`, file count `1`; after a real SIGINT/SIGTERM the turns are on disk and the process exits cleanly; the timer flushes without shutdown; the paired M1/M2 comparison record exists | `VP-T001-session-isolation`, `VP-T002-collapsed-artifact`, `VP-T002-change-readable`, `VP-T002-turn0-available`, `VP-T002-write-count-reduced`, `VP-T002-volume-flush`, `VP-T002-file-count-reduced`, `VP-T003-shutdown-flush`, `VP-T003-interval-flush`, `VP-T003-timer-resilience` |
| MS-003 | T004 | Existing suite green with the new `dump.py` lines covered and total coverage ≥ 80% | `VP-T003-passthrough-regression` |
| MS-004 | T006 | With no opt-out an untrusted-certificate HTTPS upstream fails; with it the same peer succeeds; both postures come from one connector; the existing suite is green | `VP-T006-verify-on-by-default`, `VP-T006-optout-honored`, `VP-T006-posture-is-process-wide` |

## Validation points

| Validation point | Observable behavior | Validate at |
|---|---|---|
| `VP-T001-writes-deferred` | With a collapsed format selected, after N requests and before any flush no artifact exists on disk; the documented path still writes per turn | MS-001 |
| `VP-T001-session-isolation` | Two distinct system prompts produce two independently flushed artifacts | MS-002 |
| `VP-T001-dump-failure-isolated` | A raising dump/flush still yields the upstream response body | MS-001 |
| `VP-T002-collapsed-artifact` | In a fresh dir with a collapsed format selected, a flush of turns 0..2 leaves exactly one artifact for that format, no documented files, in arrival order | MS-002 |
| `VP-T002-change-readable` | A reader can locate a changed turn and see/compare the differing content, per shipped format, with no tool | MS-002 |
| `VP-T002-turn0-available` | Turn 0's full body and rendering are present in every collapsed artifact | MS-002 |
| `VP-T002-write-count-reduced` | Instrumented write-bearing operations `= F` versus `2N`, per collapsed format, with the paired documented run; payload bytes grow with `F`, not `N²` | MS-002 |
| `VP-T002-volume-flush` | Reaching the volume threshold puts the pending turns on disk with no shutdown and the timer unable to fire | MS-002 |
| `VP-T002-file-count-reduced` | A collapsed dir holds `1` artifact; a documented dir holds `N + 1` files | MS-002 |
| `VP-T002-default-encoding` | With no format configured the output is the documented layout at `2N` writes and no `.log` artifact | MS-001 |
| `VP-T002-documented-format-preserved` | With no format configured N turns produce exactly the three documented files with today's content and line format at `2N` writes | MS-001 |
| `VP-T003-interval-flush` | With the interval small and the volume threshold unreachable, a turn appears in the artifact while the process runs | MS-002 |
| `VP-T003-timer-resilience` | A flush that raises is logged and later ticks keep flushing; a cancelled timer stops without a traceback | MS-002 |
| `VP-T003-shutdown-flush` | After SIGINT/SIGTERM the buffered turns are on disk and the process exits without a traceback | MS-002 |
| `VP-T003-passthrough-regression` | Existing `tests/test_routes_integration.py` suite green | MS-003 |
| `VP-T006-verify-on-by-default` | With no opt-out and an untrusted-certificate HTTPS upstream, the request does not complete and the failure is a certificate-verification error | MS-004 |
| `VP-T006-optout-honored` | With the opt-out and the same peer, the request completes | MS-004 |
| `VP-T006-posture-is-process-wide` | One opt-out governs every outbound call — `/metrics` and `/v1/models` use the same posture as the proxied request | MS-004 |

**Not created:** a human-facing viewer VP (`FK-007` option 1 ships no tool); a runtime posture-record VP (`FK-008` option 1 ships no record); `VP-T006-optout-is-explicit`; `VP-T006-dump-cluster-unaffected`; `VP-T002-human-facing`; `VP-T002-single-write-per-flush` (folded into `VP-T002-write-count-reduced`).

## Decisions

The canonical decision ledger is [decisions.md](decisions.md): 11 `FK-` records and 17 `US-`
records, 0 reserved. The load-bearing ones:

- `FK-001` — flush on both a time interval and a data-volume threshold, plus shutdown.
- `FK-002` — **superseded by `US-012`**. The surviving clause is "one bulk flush is one write";
  the "documented layout is not preserved" half is revised.
- `FK-007` — option 1: the self-contained full-text encoding is the addition, **no tool**.
- `FK-008` — option 1: no runtime record when the TLS opt-out is used.
- `FK-009` — option 1: the TLS VP evidence tier is the process-level `QAP-MS004` gate.
- `FK-010` — option 1: the serve script honors its documented `DUMP_DIR=""` disable form.
- `FK-011` — option A: the serve path keeps `PRESERVED_THINKING` on by default and documents it.
- `US-012` — the documented layout must not be deleted before the new behavior is demonstrated
  better and judged worth switching to; it stays the default.
- `US-014` — the user ratified the three payload overrides and the six `PI:T002/dump_format/*` as
  written.
- `US-017` — "Keep only gate evidence": 10 evidence artifacts keep, 29 remove.

## Invariants

Session-local invariants are in [invariants.md](invariants.md); the global registry is
`invariants/`. The session promoted 8 planned invariants (6 `PI:T002/dump_format/*`, 2
`PI:T006/upstream_tls/*`), 4 override provenance blocks (`OI:`), and preserved 6 existing
invariants (`EI:`). No superseded block remains.

## Verification

Session-close code bytes green at `793a5f0`: **551 passed** (513 hermetic @ 96.71%,
`QAP-MS001` 17, `QAP-MS002` 15, `QAP-MS004` 6). The merged PR head `3128cc3`
re-runs the hermetic suite at **575 passed @ 96.83%** after the PR review repairs.
Evidence is indexed in [evidence/INDEX.md](evidence/INDEX.md):
39 rows, 10 keep / 29 remove; kept artifacts are the gate transcripts, the `MS-002` comparison
record, the CR2/CR3/CR4 probe matrix, and the two claim-425 / claim-425 reach records. Review
history: three external `codex-review` passes and the closing check, all recorded in the evidence
index and the chronicle; Yield Red `YR-MS001-r4` CLEAN. The session's outcome is in
[chronicle.md](chronicle.md).

## Deliberately left out

- No viewer/tool (`FK-007` option 1), no machine-oriented dump format (`RM-014`), no default-format
  switch (`RM-013`), no header-lane content-coding change (`RM-008`), no request-direction content-coding
  posture (`RM-011`).
- No custom CA bundle/truststore, no mTLS, no client certificates, no inbound TLS.
- No custom signal handler; no fsync/journal/WAL/crash recovery; SIGKILL loss accepted.
- No runtime record of the TLS skip (`FK-008` option 1).
- No production dump reader; `get_cached_image` remains unwired.
- No migration or removal of old dump files.

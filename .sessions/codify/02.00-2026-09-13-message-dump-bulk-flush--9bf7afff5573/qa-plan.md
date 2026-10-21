# QA Plan: buffered message dump, bulk flush, and the upstream-TLS opt-out

## Overview

- **Created**: 2026-09-16 (interview round 18 — final-approval gate, step 3)
- **Session**: `02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573`
- **Scope**: supporting local proof (the hermetic `tests/` suite) and stable `MS-*` integration gates run
  through the project harness. Four milestones: `MS-001` (the format selector and the documented path's
  preservation — no flush is observable here, `022`), `MS-002` (the collapsed path and its flush triggers),
  `MS-003` (the coverage/regression gate, no QA author pass), `MS-004` (the upstream-TLS opt-out).
- **Project QA approach**: [`.agents/skills/qa-project/SKILL.md`](../../../.agents/skills/qa-project/SKILL.md)
  — process-level end-to-end gates under `tests/qa/`, real subprocess, real signals, live remote upstream
  `http://hyperipper.local:30000` (**with the `MS-004`-scoped amendment** in that skill's "Upstream" row:
  the TLS gate alone starts its own loopback self-signed HTTPS peer, per `FK-009` option 1). Endpoint corrected
  **2026-10-17** from `:30001`, which is a peer Buster Ripper proxy on the same host rather than the model server
  (`decisions.md` §US-009; the app's own `--port` default and `README.md`'s `:30001` occurrences are unchanged).
- **Execution strategy**: each scenario gate runs at its milestone, after all constituent tasks are accepted
  by local Yield and before the milestone's Yield Phase.
- **QA capability gate**: **PASSES.** Every validation point below requires process-level
  trigger→observation, deterministic trigger injection, and (for the signal scenarios) real signal delivery —
  all three are **declared** in `qa-project` §Capability Declarations. The `MS-004` local peer is declared as a
  **scoped amendment** on the "Upstream" row. **Signal-contract check: `N/A`** — the project declares no
  telemetry-verification capability (deps are `aiohttp` + `click`, `pyproject.toml:5-8`; no OTel), so the
  check passes without one. Every behavior gated here is provable by artifact contents, a directory listing,
  a process exit code, or an instrumented write count.

## Milestone Gates

Copied from `plan-summary.md` §Milestone Matrix.

| Milestone | Constituent tasks | Complete when | QA author pass |
|---|---|---|---|
| MS-001 | T001, T002, T005 | A **documented-format** session of N turns produces today's three files with today's content and line format at `2N` write-bearing operations, its README workflow unchanged; **and** a run with **no format configured** produces the documented layout (the default clause); **and** with a **collapsed format** selected, N intercepted turns produce **no artifact on disk** — every write is deferred to a flush — and a raising dump/flush still cannot fail a request. ⚠ **022**: no flush is observable here — `flush_all` exists after `T002` but has no caller until `T003` | QAP-MS001 |
| MS-002 | T003 | A **collapsed-format** session's turns reach disk only after a flush — a **volume-triggered** flush, a **timer** tick, or a **shutdown** flush — in arrival order, with turn 0's full body present, the write count `F` with `F` controlled, the artifact's name stating its configured value, and its file count `1` against the documented path's `N + 1`; **and** after a real SIGINT/SIGTERM buffered turns are in the artifact and the process exits cleanly; **and** with the timer running, a pending turn reaches the artifact without shutdown; **and** the two counts (collapsed `F` vs documented `2N` at the same N) are recorded side by side **in the `MS-002` Yield Phase's comparison record — a durable, human-readable artifact — as the comparison `G-008`(b) requires**; **and the bytes written across the session track the pending batches, not the accumulated artifact** | QAP-MS002 |
| MS-003 | T004 | Existing suite green with the new `dump.py` lines covered and total coverage ≥ 80% | — (existing suite is the gate) |
| MS-004 | T006 | With the opt-out absent an untrusted-certificate HTTPS upstream fails; with it present the same upstream succeeds; both postures come from one connector; the existing suite is green | QAP-MS004 |

## QA Gate Matrix

| ID | Validation point | Scenario | Earliest task | Required at | QA author pass | Test tier | Evidence |
|---|---|---|---|---|---|---|---|
| TC-001 | VP-T001-dump-failure-isolated | A raising dump/flush still yields the upstream response body | T001 | MS-001 | QAP-MS001 | E2E (real proxy, live upstream) | HTTP status/body + the logged exception in stderr — **ready** [070](iterations/070-forward-qa.md): `tests/qa/test_qap_ms001.py::test_tc001_documented_write_failure_still_returns_the_upstream_response`, `::test_tc001_collapsed_flush_failure_still_returns_the_upstream_response` |
| TC-002 | VP-T003-passthrough-regression | The existing passthrough suite is green | T003 | MS-003 | — (existing suite) | unit/integration (`tests/`) | green `tests/test_routes_integration.py` |
| TC-003 | VP-T006-verify-on-by-default | With no opt-out, the gate's untrusted-cert HTTPS peer is refused with a certificate-verification error | T006 | MS-004 | QAP-MS004 | E2E (loopback self-signed peer, `MS-004` amendment) | the failed request's error class + the connector construction — **ready** [055](iterations/055-forward-qa.md): `tests/qa/test_qap_ms004.py::test_tc003_untrusted_peer_refused_by_default` |
| TC-100 | VP-T001-writes-deferred | With a collapsed format selected and no flush performed, no artifact exists on disk | T002 | MS-001 | QAP-MS001 | E2E | `DUMP_DIR/<sid>/` listing before the flush — **ready** [070](iterations/070-forward-qa.md): `tests/qa/test_qap_ms001.py::test_tc100_collapsed_format_defers_every_write_until_a_flush` |
| TC-101 | VP-T001-session-isolation | Two system prompts → two session dirs, each flushed independently | T003 | MS-002 | QAP-MS002 | E2E | two dirs + their contents |
| TC-102 | VP-T002-collapsed-artifact | Fresh dir + collapsed format + a flush of turns 0..2 → exactly one `<format>.log`, no `turn_000.json`/`turn_NNN.diff`/`index.txt`, three records in arrival order, the filename stating the configured value | T003 | MS-002 | QAP-MS002 | E2E | `DUMP_DIR/<sid>/` listing + artifact contents |
| TC-103 | VP-T002-change-readable | Turn 2's change is locatable and identifiable from the directory alone — for **each** collapsed format | T003 | MS-002 | QAP-MS002 | E2E | artifact contents, per format |
| TC-104 | VP-T002-turn0-available | Turn 0's full normalized JSON body **and** its rendering are recoverable from the artifact — in both collapsed formats | T003 | MS-002 | QAP-MS002 | E2E | artifact contents, per format |
| TC-105 | VP-T002-write-count-reduced | Instrumented write-bearing operations `= F` for the collapsed format vs `2N` for the documented one, at the same N; one write per K ≥ 2 flush **with its payload bytes tracking the pending batch**; **run per collapsed format** | T003 | MS-002 | QAP-MS002 | E2E + instrumented write counter | the instrumented write counter (call count and payload bytes) + two directory listings, recorded as a pair + the `MS-002` Yield Phase's comparison record (a durable, human-readable artifact) |
| TC-106 | VP-T002-volume-flush | Threshold reached with the timer disabled → pending turns on disk, no shutdown | T003 | MS-002 | QAP-MS002 | E2E | artifact contents before/after the threshold turn |
| TC-107 | VP-T002-default-encoding | With no `--dump-format` passed → the **documented** layout at `2N` writes, no `.log` | T002 | MS-001 | QAP-MS001 | E2E | `DUMP_DIR/<sid>/` listing + contents — **ready** [070](iterations/070-forward-qa.md): `tests/qa/test_qap_ms001.py::test_tc107_no_format_configured_produces_the_documented_layout`; content arm hardened at [096](iterations/096-forward-qa.md) (nonce-bearing body, complete normalized-value comparison) |
| TC-108 | VP-T002-documented-format-preserved | Documented format (or none configured) → today's three files, today's content and line format, `2N` writes | T002 | MS-001 | QAP-MS001 | E2E + instrumented write counter | listing + contents + write count — **ready** [070](iterations/070-forward-qa.md): `tests/qa/test_qap_ms001.py::test_tc108_documented_format_preserves_contents_at_2n_writes` (+ `::test_tc108_sensitivity_documented_line_pattern_distinguishes_drift`); content arms hardened at [096](iterations/096-forward-qa.md): complete normalized body + diff source derived from the `self-contained` artifact's renderings |
| TC-109 | VP-T002-file-count-reduced | `1` artifact collapsed vs `N + 1` files documented, same N | T003 | MS-002 | QAP-MS002 | E2E | two directory listings |
| TC-110 | VP-T003-shutdown-flush | After a real SIGINT **and** SIGTERM: buffered turns on disk, clean exit | T003 | MS-002 | QAP-MS002 | E2E (real signal) | subprocess + post-exit listing + exit code |
| TC-111 | VP-T003-interval-flush | Interval injected small, volume unreachable → one appended turn reaches the artifact while the process runs | T003 | MS-002 | QAP-MS002 | E2E | live artifact contents/mtime, no shutdown |
| TC-112 | VP-T003-timer-resilience | A raising flush is logged and later ticks still flush; a cancelled timer stops without a traceback | T003 | MS-002 | QAP-MS002 | E2E | log record + a later successful tick + exit code |
| TC-113 | VP-T006-optout-honored | With the opt-out and the **same** untrusted-cert peer, the request completes | T006 | MS-004 | QAP-MS004 | E2E (loopback peer) | completed request + connector construction — **ready** [055](iterations/055-forward-qa.md): `tests/qa/test_qap_ms004.py::test_tc113_same_peer_accepted_with_optout` |
| TC-114 | VP-T006-posture-is-process-wide | One opt-out governs the `/metrics` poll and the `/v1/models` probe as well as the proxied request | T006 | MS-004 | QAP-MS004 | E2E + instrumented call sites | one-connector assertion + the instrumented call sites — **ready** [055](iterations/055-forward-qa.md): `tests/qa/test_qap_ms004.py::test_tc114_one_posture_governs_every_outbound_call` (paired runs, one peer shape, `tests/qa/tls_probe_entry.py`) |

## QA Author Passes

| QA pass | Runs after | Runs before | Validation points | Scenarios |
|---|---|---|---|---|
| QAP-MS001 | All MS-001 tasks (T001, T002, T005) accepted by local Yield | MS-001 Yield Phase | VP-T001-writes-deferred, VP-T001-dump-failure-isolated, VP-T002-documented-format-preserved, VP-T002-default-encoding | TC-100, TC-001, TC-107, TC-108 |
| QAP-MS002 | T003 accepted by local Yield | MS-002 Yield Phase | VP-T001-session-isolation, VP-T002-collapsed-artifact, VP-T002-change-readable, VP-T002-turn0-available, VP-T002-write-count-reduced, VP-T002-volume-flush, VP-T002-file-count-reduced, VP-T003-shutdown-flush, VP-T003-interval-flush, VP-T003-timer-resilience | TC-101, TC-102, TC-103, TC-104, TC-105, TC-106, TC-109, TC-110, TC-111, TC-112 |
| QAP-MS004 | T006 accepted by local Yield | MS-004 Yield Phase | VP-T006-verify-on-by-default, VP-T006-optout-honored, VP-T006-posture-is-process-wide | TC-003, TC-113, TC-114 |

`MS-003` carries **no QA author pass**: its gate is the existing hermetic suite (`TC-002`), which is the
gate itself rather than an authored scenario. This is the plan's own disposition (§Milestone Matrix) and is
not an exemption — no user-visible behavior is exempted anywhere in this plan; every dump- and TLS-cluster
behavior is gated above.

## Authored Gate Tests

- **QAP-MS002** — authored ([034-forward-qa](iterations/034-forward-qa.md)), **ready** for the MS-002
  Yield Phase. Tests: `tests/qa/test_qap_ms002.py` + `tests/qa/test_qap_ms002_hostile.py` (TC-101…TC-112 + R1/R2 hostile tier + R5 identity
  near-misses), shared harness `tests/qa/harness.py`, instrumented write-count entry
  `tests/qa/counted_proxy_entry.py`. Selected explicitly via
  `.venv/bin/python -m pytest tests/qa/test_qap_ms002.py tests/qa/test_qap_ms002_hostile.py -o addopts=` (excluded from the hermetic
  default run by `norecursedirs`). The paired M1/M2 comparison numbers for the Yield Phase's
  `G-008`(b) comparison record are in the iteration report §Harness Execution. Yield Phase re-runs and
  writes the authoritative verdicts. [048-forward-qa](iterations/048-forward-qa.md) repairs the
  upstream-response assertions and adds a mandatory live-upstream preflight; the tests remain authored,
  but their current execution is **blocked by upstream reachability**, not passing on a 502 baseline.
- **QAP-MS004** — authored ([055-forward-qa](iterations/055-forward-qa.md)), **ready** for the MS-004
  Yield Phase. Tests: `tests/qa/test_qap_ms004.py` (TC-003, TC-113, TC-114 + the R1/R2/R4/R5 tier
  dispositions), the gate's own loopback self-signed HTTPS peer `tests/qa/tls_peer.py` (committed
  certificate material, `FK-009` option 1), and the instrumented client-posture entry
  `tests/qa/tls_probe_entry.py`. Shares `tests/qa/harness.py`, extended with `upstream=` /
  `require_live_upstream=` / `probe_report=` so this one gate can run on its own peer — the §Upstream
  amendment's harness hook; the live-upstream preflight still fails every other gate closed. Selected
  explicitly via `.venv/bin/python -m pytest tests/qa/test_qap_ms004.py -o addopts=`. This gate does
  **not** require `hyperipper.local:30000` (the gate upstream since the 2026-10-17 correction of `:30001`;
  verified: 6/6 with the remote upstream unroutable), so it is
  not blocked by the reachability condition holding the QAP-MS002 re-run. Yield Phase re-runs and writes
  the authoritative verdicts.
- **QAP-MS001** — authored ([070-forward-qa](iterations/070-forward-qa.md)), **ready** for the MS-001
  Yield Phase. Tests: `tests/qa/test_qap_ms001.py` — `TC-100`, `TC-001` (both the documented write
  path and the collapsed flush path), `TC-108` (+ its pure sensitivity check), `TC-107`, with the
  R1 hostile tier (unknown selector values refused without a write; the body parse boundary on the
  documented writer), R2 (the documented writer's symlink containment — its sibling audit of the
  writer `MS-002` covered), and the R5 near-miss `sid` pair against a pre-existing collapsed
  artifact. No gate in this pass passes `--dump-flush-interval` / `--dump-flush-bytes` (the `022`
  directive stands); `TC-100` bounds its pre-flush window by observation instead. Shares
  `tests/qa/harness.py`, extended by this pass: `read_text` / `log_artifacts` /
  `assert_upstream_response` promoted out of `test_qap_ms002.py`, and `startup_refused()` added.
  Selected explicitly via `.venv/bin/python -m pytest tests/qa/test_qap_ms001.py tests/qa/test_qap_ms001_witnesses.py tests/qa/test_qap_ms001_tiers.py -o addopts=`. The
  documented half of the `G-008`(b) pair is measurable here too (`write_ops = 2N`, `files = N + 1`).
  `ready` is not a verdict: Yield Phase re-runs and writes the authoritative `pass`/`fail` on the
  four `VP-*` points.
  - **Re-entered at [077-forward-qa](iterations/077-forward-qa.md) — still `ready`, no scenario
    re-authored.** This pass drains the two `forward-qa`-owned backlog rows whose named trigger is a
    touch of `tests/qa/harness.py` (`QB:qa-harness/unexercised-path-keyword`, naming this very pass,
    and `QB:qa-harness/static-body-cannot-connect-limb`, routed to drain together with it):
    `Proxy.url()` / `Proxy.post(body, timeout=…)` restored to their pre-055 signatures, and
    `assert_upstream_response`'s `b"Cannot connect" not in body` limb dropped — the status limb is
    the discriminator and a proxy-local transport failure answers `502`, so no scenario's approved
    assertion is weakened. One process-free witness added for the drain, in the module this pass
    owns: `tests/qa/test_qap_ms001.py::test_sensitivity_upstream_oracle_distinguishes_transport_failure_from_a_relayed_body`
    (red if the dropped limb is restored). **The `022` directive is still honored: no `QAP-MS001`
    gate passes a flush parameter** (`grep -n "dump-flush" tests/qa/test_qap_ms001.py` → the
    docstring stating the rule, nothing in any `extra_args`). No `TC-*` / `VP-*` semantics changed;
    the four scenario gates and their hostile/identity tiers are byte-identical to 070's, and their
    `G-008`(b) documented-half figures re-measured the same on current bytes.
  - **Re-entered again at [087-forward-qa](iterations/087-forward-qa.md) — still `ready`, still no
    scenario re-authored and still zero test bytes moved** (`git log -1 -- tests/qa/` → Checkpoint
    183, 077's own commit). This pass exists because the **CR3 wave** (`T018`+`T019`+`T020`) and
    `T019.001` landed production bytes on the interception path every `QAP-MS001` gate triggers
    through (`server.py`, `routes/messages.py`) while the approved gate stayed put. Required
    Check 8 re-proved on `8dfa5af`: `tests/qa/test_qap_ms001.py` **11/11**, shared-harness
    consumers `tests/qa/test_qap_ms002.py` 15/15 and `tests/qa/test_qap_ms004.py` 6/6, hermetic
    default **476 @ 95.91%**. The documented half of the `G-008`(b) pair re-measures unchanged
    (`write_ops = 2N = 6`, `files = N + 1 = 4`) and `TC-100` still witnesses `write_ops = 1` for
    three buffered turns. **`022` honored on these bytes too** (`grep -n "dump-flush"
    tests/qa/test_qap_ms001.py` → the docstring stating the rule, nothing in any `extra_args`), and
    no `forward-qa`-owned backlog row was open to this pass — 077's two rows are resolved there.
    One disposition closed without a test edit: `T018`'s `client_max_size=0` widened the input
    domain behind this writer, no `VP-*` here speaks to body size, and the property is owned at the
    cheaper tier by `tests/test_transport_parity.py::TestNoRequestBodyCap` — named in 087 so the
    boundary shows it was considered and routed, not overlooked. Test paths unchanged:
    `tests/qa/test_qap_ms001.py`.
  - **Re-entered at [092-forward-qa](iterations/092-forward-qa.md) — endpoint correction, `ready`
    retained, still zero scenario bytes moved.** The user corrected the gate upstream on 2026-10-17:
    `http://hyperipper.local:30000` is the model server; `:30001` is a peer Buster Ripper proxy
    (`decisions.md` §US-009, Checkpoint 211). `tests/qa/harness.py:5`/`:39` and this file's `:13`,
    `:94`, `:147` were re-pointed with dated notes; `README.md`'s `:30001` and the app's `--port`
    default were left alone as the proxy's own listen port. Re-proved on `HEAD 39585db`: `:30000`
    preflight HTTP 200, `tests/qa/test_qap_ms001.py` **11/11**, `tests/qa/test_qap_ms002.py`
    15/15, `tests/qa/test_qap_ms004.py` 6/6 (own loopback peer, per the `MS-004` amendment),
    **32/32 gates green**, hermetic **511 @ 96.71%**. `022` re-checked by grep; no `TC-*`/`VP-*`
    semantics changed.
  - **Re-entered at [096-forward-qa](iterations/096-forward-qa.md) — `T021` gate-strengthening,
    `ready` retained, no scenario semantics widened.** `YR-QA-1`'s loophole: `assert_documented_contents`
    was presence-only. It now compares `turn_000.json` against the **complete** normalized body (the sent
    body with the nonce-bearing `system` replaced by the nonce-free text) and each `turn_NNN.diff`'s changed
    lines against a `difflib` diff over the **`self-contained` artifact's public per-turn renderings** —
    never a call into `_render_body`. `TC-107`/`TC-108` now drive nonce-bearing bodies, so the route's
    normalize-then-dump ordering is observable; the nonce-free scenarios (`TC-100`, `TC-001`, hostile tier,
    R5) keep their body bytes and `sid` derivation unchanged. `022` re-checked by grep. Shared-harness
    change: `turn_body` gained the optional `nonce`, `billing_system` was added, and `record_section` was
    promoted out of `test_qap_ms002.py` (one public-artifact reader for both gates). Re-proved on
    `HEAD c9b4751`: **12/12** `QAP-MS001`, 15/15 `QAP-MS002`, 6/6 `QAP-MS004`, hermetic
    **511 @ 96.71%**. The two mutation replays remain local Yield's.
  - **Re-entered at [099-forward-qa](iterations/099-forward-qa.md) for the run-49 gate-repair wave
    (`T021.001`/`T021.002`) — `ready` retained, no scenario semantics widened.** `assert_documented_contents`
    now derives the diff over the writer's own line identity (`splitlines(keepends=True)`, `""` join) and reads
    both sides with one structural reader that recovers a hunk's first changed line; `rendering_section` removes
    only the artifact's record-separator newlines instead of `.strip()`ing the renderings. `Proxy.post` builds
    `ProxyHandler({})` so the oracle judges the local proxy under ambient proxy settings. Four focused regressions
    (`S21` CE-1/CE-2/CE-3, `S22`) are green on the repaired gate and red on the pre-repair helpers/harness.
    Re-proved on `HEAD f3468c7`: **16/16** `QAP-MS001`, 15/15 `QAP-MS002`, 6/6 `QAP-MS004`, hermetic
    **511 @ 96.71%**. `022` re-checked by grep. No `TC-*`/`VP-*` semantics changed.
  - **Re-entered at [105-forward-qa](iterations/105-forward-qa.md) — `ready` retained, no scenario bytes authored by this pass.**
    The five concurrent QA/harness edits (gate model alias `qa-gate-model` → `sloppus`, declared in `qa-project`
    §Upstream) were validated and committed with the gate: direct upstream probes measured `sloppus` **200** /
    `qa-gate-model` **404**, and no approved assertion moved. Re-proved on `HEAD 8d27f6e`: **16/16** `QAP-MS001`,
    15/15 `QAP-MS002`, 6/6 `QAP-MS004`, hermetic **513 @ 96.71%**; the `G-008`(b) documented half re-measured
    `write_ops = 2N = 6`, `files = N + 1 = 4`, and `TC-100` still witnesses `write_ops = 1` for three buffered turns.
    `022` re-checked by grep. Test paths unchanged: `tests/qa/test_qap_ms001.py`.
  - **Re-entered at [108-forward-qa](iterations/108-forward-qa.md) — `T021.003` (`S21` CE-2 residual), `ready`
    retained, no scenario semantics widened.** `read_text` gained an optional `newline` parameter (default
    unchanged), and `assert_documented_contents`'s observed `turn_NNN.diff` plus `TC-108`'s `self-contained`
    artifact are now read with `newline=""`, so the expected diff is computed over the same line endings
    production diffed before the write. One focused artifact round-trip regression
    (`test_sensitivity_s21_ce2_artifact_read_preserves_line_endings`) is red on the pre-repair universal read
    with the exact `expected=[]` / `observed=['-line1\n', '+line1\n']` shape and green after; a writer that drops
    the line-ending change is rejected. Re-proved on `HEAD 65c298c`: **17/17** `QAP-MS001`, 15/15 `QAP-MS002`,
    6/6 `QAP-MS004`, hermetic **513 @ 96.71%**. `022` re-checked by grep. Test path unchanged:
    `tests/qa/test_qap_ms001.py`.

## Scenario Contracts

**TC-100 — buffering defers the write (MS-001).**
*Trigger*: start `.venv/bin/buster-ripper --host 127.0.0.1 --port <port> --upstream http://hyperipper.local:30000 --dump-dir <fresh tmp> --dump-format change-focused`; send N turns through `POST /v1/messages` (endpoint corrected 2026-10-17 from `:30001`, a peer proxy — `decisions.md` §US-009). ⚠ **022 — no flush parameter is passed**: `--dump-flush-interval`/`--dump-flush-bytes` arrive with `T003`, and at `MS-001` no trigger exists at all.
*Expected owner*: `dump_turn` (`buster_ripper/dump.py`) buffers; `flush_all` is the only writer on this path.
*Visible result*: `DUMP_DIR/<sid>/` holds no artifact containing those turns.
*Forbidden result*: a per-turn file or `index.txt` appearing on the collapsed path — the deferral broken; a flush that writes one artifact per pending turn.
*Lifecycle*: process stays healthy across all N turns; the deferral ends at the first flush.
*Telemetry*: none required (no telemetry capability declared).
*Evidence source*: `DUMP_DIR` listing + directory contents.

**TC-101 — sessions stay isolated (MS-002).**
*Trigger*: as TC-100, with `--dump-flush-interval <large> --dump-flush-bytes <unreachable>` added; send two distinct system prompts; drive one flush.
*Expected owner*: `dump_turn` buffers per `sid`; `flush_all` flushes each session independently.
*Visible result*: two session directories, each holding its own flushed artifact.
*Forbidden result*: both prompts landing in one session directory; one session's flush writing the other's records.
*Lifecycle*: process stays healthy across all turns.
*Telemetry*: none.
*Evidence source*: two dirs + their contents.

**TC-001 — a raising dump does not fail the request (MS-001).**
*Trigger*: drive a request whose dump path raises (injected failure in the write/append path).
*Expected owner*: the swallow-all contract at the caller (`routes/messages.py:handle_messages`).
*Visible result*: the upstream response body is returned unchanged; the exception is logged.
*Forbidden result*: a 5xx caused by dump behavior, or a silent absence of the log record.
*Lifecycle*: the process continues serving.
*Telemetry*: none.
*Evidence source*: HTTP status/body + the logged exception on stderr.

**TC-102 / TC-103 / TC-104 — the collapsed artifact's shape, readability, and turn 0 (MS-002).**
*Trigger*: a fresh session dir, `--dump-format <collapsed value>`, turns 0..2 sent, a flush triggered by the volume threshold.
*Expected owner*: the format dispatch in `dump.py` selects the collapsed writer; `flush_all` appends once.
*Visible result*: exactly one `<format>.log` for the configured value; no `turn_000.json`, `turn_NNN.diff`, `index.txt`; three records in arrival order; the filename states the configured value; turn 0's raw normalized JSON body **and** its rendering are recoverable; a reader can locate turn 2's change from the directory alone (TC-103: for `self-contained`, by comparing turn 2's record against turn 1's).
*Forbidden result*: two artifacts for one configured format; a documented file appearing on the collapsed path; records out of arrival order; turn 0 stored only as a diff; a summary line that lost the turn number or the change counts.
*Lifecycle*: append-only — the artifact is never rewritten; a second flush extends it.
*Telemetry*: none.
*Evidence source*: directory listing + artifact contents, per format.

**TC-105 / TC-109 — the write-count and file-count measures (MS-002).**
*Trigger*: two runs at the same N — one per collapsed format, one documented — with both flush parameters injected so `F` is chosen by the test. The documented half is `TC-108`'s run at `MS-001`; **the pair is completed here**.
*Expected owner*: one write path per turn (`PI:T002/dump_format/one_write_path_per_turn`); `flush_all` writes once per flush.
*Visible result*: instrumented write-bearing file operations `= F` collapsed vs `2N` documented; one write per flush of K ≥ 2 turns; **and the bytes written across the session track the pending batches, not the accumulated artifact**; files `1` collapsed vs `N + 1` documented. **The paired runs are the measurement `G-008`(b) requires; the `MS-002` Yield Phase records them side by side in its comparison record.**
*Forbidden result*: `2N + F` in any configuration (witness C); a flush that writes once per pending turn (witness A); a rewrite-per-flush whose bytes grow `O(N²)` (witness B); a collapsed run that also writes the documented files.
*Lifecycle*: counts are per session.
*Telemetry*: none.
*Evidence source*: the instrumented write counter (**call count and payload bytes**) + two directory listings, recorded as a pair + the `MS-002` Yield Phase's comparison record (a durable, human-readable artifact).

**TC-106 — the volume trigger (MS-002).**
*Trigger*: interval injected large so the timer cannot fire; turns appended until `pending_bytes` reaches `--dump-flush-bytes`.
*Expected owner*: the append-time threshold check inside `dump_turn`, reading the same field the append wrote.
*Visible result*: the pending turns are in the artifact with no shutdown and no timer tick.
*Forbidden result*: turns accumulating past the threshold with nothing on disk until shutdown; a check evaluated before the append or off by one.
*Lifecycle*: the check is collapsed-path only — on the documented path nothing accumulates and the check never fires.
*Telemetry*: none.
*Evidence source*: artifact contents before/after the threshold turn.

**TC-108 — the documented format is preserved (MS-001).**
*Trigger*: `--dump-format documented` (or none configured) at the same N as `TC-105`; no flush parameter is passed — the documented path writes per turn, so `MS-001` can run it.
*Expected owner*: the format dispatch's documented branch — today's three write sites, retained.
*Visible result*: `turn_000.json`, `turn_{N:03d}.diff`, `index.txt` with today's content and line format, at `2N` write-bearing operations. This is the documented half of the `G-008`(b) pair; `TC-105` supplies the collapsed half at `MS-002`.
*Forbidden result*: a drifted `index.txt` line format; a missing or renamed documented file; a `.log` artifact on the documented path.
*Lifecycle*: unchanged from today.
*Telemetry*: none.
*Evidence source*: listing + contents + the instrumented write count.

**TC-107 — the default is the documented layout (MS-001).**
*Trigger*: no `--dump-format` passed, no config override.
*Expected owner*: `config.DUMP_FORMAT`'s `"documented"` default.
*Visible result*: `turn_000.json`, `turn_{N:03d}.diff`, `index.txt` at `2N` writes; no `.log` artifact.
*Forbidden result*: a collapsed artifact appearing with nothing configured — witness W1 of `G-008`, the exact shape `US-012` forbids.
*Lifecycle*: unchanged from today.
*Telemetry*: none.
*Evidence source*: `DUMP_DIR/<sid>/` listing + contents.

**TC-110 / TC-111 / TC-112 — the flush triggers and timer resilience (MS-002).**
*Trigger*: TC-110 — real SIGINT and (separately) SIGTERM to the proxy subprocess with turns pending; TC-111 — a small interval and an unreachable volume threshold; TC-112 — an injected raise inside one flush iteration.
*Expected owner*: `on_cleanup` cancels the flush task, then flushes, then closes the shared session — in that order; the timer loop catches per-iteration exceptions and continues.
*Visible result*: TC-110 — buffered turns on disk and a clean exit (exit 0, no traceback); TC-111 — the turn reaches the artifact while the process runs; TC-112 — the raise is logged and a later tick flushes successfully.
*Forbidden result*: a traceback replacing `GracefulExit`; buffered turns lost at shutdown; a timer task that dies silently on its first exception; a flush that raises out of `on_cleanup`.
*Lifecycle*: quiescence is established by waiting for process exit before inspecting the directory; TC-111 polls the artifact rather than sleeping a fixed multiple of the interval.
*Telemetry*: none.
*Evidence source*: subprocess + real signal + post-exit listing + exit code + stderr log record.

**TC-003 / TC-113 / TC-114 — the upstream-TLS postures (MS-004).**
*Trigger*: the gate's own loopback self-signed HTTPS peer (`https://127.0.0.1:<peer-port>`, committed fixture); TC-003 with no opt-out, TC-113 with `--skip-tls-verify`, TC-114 with the opt-out and the `/metrics` poll and `/v1/models` probe exercised.
*Expected owner*: the explicit `aiohttp.TCPConnector(ssl=not config.SKIP_TLS_VERIFY)` built inside `on_startup` and passed as `connector=` to the package's only `ClientSession` (`server.py:47`).
*Visible result*: TC-003 — the request does not complete and the failure is a certificate-verification error; TC-113 — the same peer, same request, completes; TC-114 — every outbound call carries the one posture.
*Forbidden result*: **W1** — a self-signed peer accepted with no opt-out (the connector built `ssl=False` unconditionally, or dropped); **W2** — the flag accepted and its global set but the connector built without it, so both postures behave identically; a per-call-site posture.
*Lifecycle*: the peer is the gate's own; the remote upstream is not used by this gate (the scoped `FK-009` amendment); the proxy exits cleanly (exit code 0, no traceback) in all three scenarios.
*Telemetry*: none — `FK-008` option 1 ships no runtime record, so no scenario asserts a posture log line.
*Evidence source*: the failed/completed request, its error class, and the connector construction.

## Phase Requirements

- **MS-001 Yield Phase** executes `QAP-MS001` — TC-100, TC-001, TC-107, TC-108 — after every constituent task passes local Yield and before the milestone verdict. ⚠ **022**: no scenario here may pass `--dump-flush-interval` or `--dump-flush-bytes` — those options do not exist until `T003`.
  ✅ **Ruled at [071-yield-phase](iterations/071-yield-phase.md) §Check 6 (2026-10-06): the directive is honored literally and no plan change is owed.** Its stated premise ("those options do not exist until `T003`") expired once `T003` landed and they became passable, but the ban was kept as written: no `QAP-MS001` gate passes either flag (`grep -n "dump-flush" tests/qa/test_qap_ms001.py` → the docstring stating the rule, and nothing in any `extra_args`). `TC-100` establishes "before any flush" by observation rather than by disabling a timer — the artifact is append-only and never removed, so a snapshot holding no artifact **is** a pre-flush reading — which satisfies `VP-T001-writes-deferred`'s own wording ("the assertion is an **absence**, so it needs no trigger") and avoids the sleep-and-hope race the directive existed to prevent. Relaxing the directive to inject a disabled timer would widen this milestone's contract, so it stays as written.
- **MS-002 Yield Phase** executes `QAP-MS002` — TC-101, TC-102, TC-103, TC-104, TC-105, TC-106, TC-109, TC-110, TC-111, TC-112.
- **MS-003** executes the existing hermetic suite (`TC-002`) plus the coverage floor (`U ≤ 3.2 + 0.2·K`, `pyproject.toml:19`); no authored scenario.
- **MS-004 Yield Phase** executes `QAP-MS004` — TC-003, TC-113, TC-114 — against its own loopback peer.
- The hermetic default suite (`uv run pytest`) stays green and network-free; the `tests/qa/` gates are selected explicitly and are not part of it.

## Traceability Audit

| Validation point | Observable behavior | QA scenarios | Gate matches plan | Status |
|---|---|---|---|---|
| VP-T001-writes-deferred | no artifact before a flush, collapsed format selected | TC-100 | yes | covered |
| VP-T001-session-isolation | two prompts → two independently flushed dirs | TC-101 | yes | covered |
| VP-T001-dump-failure-isolated | raising dump still returns the response body | TC-001 | yes | covered |
| VP-T002-collapsed-artifact | one `<format>.log`, no documented files, arrival order, name states the value | TC-102 | yes | covered |
| VP-T002-change-readable | turn 2's change locatable and identifiable, per format | TC-103 | yes | covered |
| VP-T002-turn0-available | turn 0's raw body **and** rendering, per format | TC-104 | yes | covered |
| VP-T002-write-count-reduced | `F` vs `2N`, one write per flush, per collapsed format, paired | TC-105 | yes | covered |
| VP-T002-volume-flush | threshold reached, timer disabled, turns on disk | TC-106 | yes | covered |
| VP-T002-default-encoding | no `--dump-format` → documented layout at `2N` | TC-107 | yes | covered |
| VP-T002-documented-format-preserved | documented format byte-unchanged at `2N` | TC-108 | yes | covered |
| VP-T002-file-count-reduced | `1` vs `N + 1` files | TC-109 | yes | covered |
| VP-T003-shutdown-flush | real SIGINT/SIGTERM → turns on disk, clean exit | TC-110 | yes | covered |
| VP-T003-interval-flush | turn reaches the artifact with the process running | TC-111 | yes | covered |
| VP-T003-timer-resilience | raising flush logged, loop continues, cancel is clean | TC-112 | yes | covered |
| VP-T003-passthrough-regression | existing suite green | TC-002 | yes | covered |
| VP-T006-verify-on-by-default | untrusted cert refused with no opt-out | TC-003 | yes | covered |
| VP-T006-optout-honored | same peer accepted with the opt-out | TC-113 | yes | covered |
| VP-T006-posture-is-process-wide | one connector, every outbound call | TC-114 | yes | covered |

**Orphans: none.** Every `TC-*` names a `VP-*`; every `VP-*` in `plan-summary.md` §Validation Point Matrix is
named by at least one `TC-*`. The three retired/not-created handles (`VP-T002-single-write-per-flush`,
`VP-T002-human-facing`, `VP-T006-optout-is-explicit`, `VP-T006-dump-cluster-unaffected`) carry **no** scenario
and must not be authored — their dispositions are closed in the plan (`F-3`, `FK-007` option 1, `FK-008`
option 1, 017).

**Behavior disposition coverage.** Every `B-*` row in `plan-summary.md` §Behavior Disposition Matrix maps to a
`VP-*` above or carries an explicit approved disposition: the "(new) skipping is visible at runtime" row is
**out of scope** by `FK-008` option 1 (no VP, correctly — asserting the absence of a log line would prove the
log, not the posture), and the "(new) the flag appears in `--help`, the README and the serve script" row is
**structural** (`T006`'s done-when, no VP). No behavior is deferred beyond the earliest boundary at which it
can be tested end to end.

# Iteration 056 — Yield Phase (MS-004 / QAP-MS004)

## Orchestrator Prompt (verbatim)

```text
You are a subagent spawned by another session.
Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
Milestone: MS-004
QA Pass: QAP-MS004
Target base: main
Target object: 72e3a1c6861bf0d8de6317adf52289e5f6cfa4e1
```

## Invocation Contract Verification

- Session handle `02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573` verified against
  `session-identity.json` (`session_id: 02.00-2026-09-13-message-dump-bulk-flush`,
  `session_uid: 9bf7afff-5573-402a-b102-827423a1dded`) — MATCH. Files read directly; nothing globbed.
- Full milestone gate: `Milestone: MS-004` + `QA Pass: QAP-MS004` + `Target base: main` + full
  `Target object` all present — invocation contract satisfied.
- **BASE-SYNC gate**: latest report `BASE-SYNC: unchanged`
  ([054-01-forward](054-01-forward.md), Checkpoint 126) at target `main @ 72e3a1c6861bf0d8de6317adf52289e5f6cfa4e1`.
  Re-verified this pass: `git rev-parse main` = the target object exactly;
  `git merge-base --is-ancestor 72e3a1c6… HEAD` → **true** at HEAD `25c2b77` (Checkpoint 127); 154+ ahead /
  0 behind; worktree clean. No correctness-required integration repair was outstanding, and MS-004 is the
  final phase so the no-deferral clause was evidenced in 054 rather than assumed. Gate PASSES.
- Constituent tasks: `MS-004` = **T006 only** (Milestone Matrix), accepted by local Yield
  ([051-01-yield](051-01-yield.md)). `QAP-MS004` authored ([055-forward-qa](055-forward-qa.md),
  Checkpoint 127) — `READY-PASSING`.
- Snapshot identity: the production tree at HEAD is byte-identical to the tree local Yield validated —
  `git diff --exit-code 9ae82b0 HEAD -- buster_ripper serve_buster_ripper.sh pyproject.toml README.md` → **0**.
  The only files moved since Checkpoint 119 are the four `tests/qa/` artifacts of this pass
  (`harness.py`, `test_qap_ms004.py`, `tls_peer.py`, `tls_probe_entry.py`).
- Review scope: cumulative diff `72e3a1c6…HEAD` (418 files). This boundary's increment is `98391ac`
  (T006 production) + `25c2b77` (QAP-MS004 QA artifacts). Checks 1–5 examine the TLS cluster and the shared
  harness amendment; previously certified surfaces are re-read only where this increment touches them.
- `decisions.md` sweep: `Verification: yield-phase:MS-004 … PENDING` → **two rows**, decisions.md:963 and
  :967 (`FK-008` factual claims). Owned by this arbiter — adjudicated in §Assumption Routing below.
  (`yield-phase:MS-001` at :792 is held by MS-001's own arbiter, untouched.)
- Codex findings: none supplied at dispatch; the orchestrator's parallel review is merged upstream.

## Check 1: Caller Verification — PASS

Every new production entry point in the increment, and every new QA-side symbol:

| New surface | Site | Caller / consumer | Verdict |
|---|---|---|---|
| `config.SKIP_TLS_VERIFY` | `buster_ripper/config.py:17` (`= False`) | read at `buster_ripper/server.py:65`; written only at `buster_ripper/main.py:32` | WIRED |
| `main._set_skip_tls_verify` | `main.py:30-33` | registered as `callback=_set_skip_tls_verify` on the `--skip-tls-verify` option, `main.py:64-67`; click invokes it at parse time | WIRED |
| `main._apply_cli_config` | `main.py:36-58` | called at `main.py:139-149` inside `main()` | WIRED |
| explicit `TCPConnector(ssl=…)` | `server.py:65` | inside `on_startup` (`server.py:55`), passed as `connector=` to the only `ClientSession` (`server.py:66-68`); `on_startup` registered at `server.py:120` from `create_app()`, called at `main.py:188` | WIRED |
| `SKIP_TLS_VERIFY` env var | `serve_buster_ripper.sh:27` | guard at `:39` appends `--skip-tls-verify` to the `EXTRA_FLAGS` array expanded at `:50-54` (`exec buster-ripper …`) | WIRED |
| `tests/qa/tls_peer.py` | new | imported by the gate (`test_qap_ms004.py:51`), started in all 6 tests | WIRED |
| `tests/qa/tls_probe_entry.py` | new | launched by `harness.py:38` `TLS_PROBE_ENTRY` when `probe_report=` is set (`harness.py:156-161`); 7 gate call sites | WIRED |
| `harness.Proxy(upstream=, require_live_upstream=, probe_report=)` | `harness.py:117-119` | 7 call sites in `test_qap_ms004.py` | WIRED |
| `harness.Proxy.post(path=)` / `Proxy.url(path=)` | `harness.py:175-178` | **zero callers** across `tests/` — see P3-1 | SEVERED (QA hygiene only) |

Removed callbacks/hooks in this increment: `main()`'s inline config-assignment block was replaced by
`_apply_cli_config` — the replacement is called and its 17 assignments are item-for-item equivalent (§Check 2).
The old implicit default-connector path (`ClientSession(timeout=…)` alone) is replaced by the explicit
connector, which is constructed and consumed. No production entry point has zero callers; nothing is
test-only.

## Check 2: Migration Parity — PASS

Two replaced paths, both compared mechanically, not by reading the forward report.

**(a) `main.py`: 17 config assignments inline → `_apply_cli_config`.**
AST comparison of the leading `config.*` assignment run in `98391ac^:buster_ripper/main.py::main` against
`buster_ripper/main.py::_apply_cli_config`:
- count 17 → 17; **targets identical and in identical order**;
- right-hand sides identical modulo the `cli["<name>"]` indirection (`EVAL_PROFILE`'s
  `config.PROFILES[eval_profile]` → `config.PROFILES[cli["eval_profile"]]` is the same expression);
- the call site passes exactly 17 keyword arguments and the key set equals the helper's input set
  (`missing: []`, `extra: []`) — a dropped key would raise, not pass silently;
- `main()`'s parameter list is **unchanged** — the capacity-gate closure (`main.py:70-86` extraction so
  `main()` gains no parameter) is honored, as the plan required.
The two side-effectful blocks (`dump_dir` mkdir, `stats_db` init) were deliberately left in `main()` and are
untouched. No omission, intentional or accidental.

**(b) `server.py`: `ClientSession(timeout=…)` → `TCPConnector(ssl=not config.SKIP_TLS_VERIFY)` +
`ClientSession(timeout=…, connector=…)`.** Parity of every old side effect, checked against the installed
library (`aiohttp 3.13.3`), because "behaviour-preserving when the flag is unset" is the plan's central claim:
- old: session with no connector constructs `TCPConnector(loop=loop, ssl_shutdown_timeout=…)` internally
  (`ClientSession.__init__`), whose `ssl` parameter defaults to `True`;
- new: `TCPConnector(ssl=True)` on the unset path; `_merge_ssl_params(True, None, None, None)` returns
  `True` unchanged (no deprecation branch fires) → same `_SSL_CONTEXT_VERIFIED` resolution. **Unset path is
  behaviour-identical**, and TC-003 observes exactly that (a verifying refusal) rather than inferring it.
- loop binding: `ClientSession.__init__` adopts `loop = connector._loop` when a connector is passed, so the
  `connector._loop is not loop` `RuntimeError` guard cannot fire on this path (the claim-38 mechanism
  correction holds); the connector is built inside `on_startup`, i.e. on a running loop.
- ownership/teardown: `connector_owner` still defaults `True`, so `await app["session"].close()`
  (`server.py:89`) closes the connector exactly as it closed the internally-owned one. Nothing lost.
- `timeout=aiohttp.ClientTimeout(total=600)` unchanged.

**(c) Serve script.** The `--skip-tls-verify` append uses the boolean `== "1"` peer shape the plan mandated
(not `-n`), so the default posture cannot invert; `EXTRA_FLAGS` array assembly and the nounset-safe guarded
expansion are T011/T012's accepted form and the flag rides it as one whole word.

## Check 3: Data Flow Completeness — PASS

The posture's full chain, traced by reading code:

`SKIP_TLS_VERIFY=1` (env, `serve_buster_ripper.sh:27`) → guard `:39` → `--skip-tls-verify` in the exec'd argv
`:50-54` → click parses and fires `_set_skip_tls_verify` **before** the command body (`main.py:64-67,30-33`) →
`config.SKIP_TLS_VERIFY` (`config.py:17`) → read at `on_startup` (`server.py:65`) →
`TCPConnector(ssl=not …)` → `ClientSession(connector=…)` (`server.py:66-68`) → `app["session"]` →
**every** outbound consumer:

| Consumer | Outbound call | Path |
|---|---|---|
| `server.py:71-74` → `backend.detect_backend` | `backend.py:54` `session.get(f"{backend_url}/v1/models")` | `/v1/models` probe |
| `server.py:21` → `spawn_kv_cache_monitor` → `compaction.poll_kv_cache` | `compaction.py:36` `session.get(f"{config.UPSTREAM}/metrics")` | `/metrics` poll |
| `routes/chat.py:37,125` | `session.post(.../v1/chat/completions)` | proxied chat |
| `routes/messages.py:127,130` and `:228,231` | `session.post(.../v1/messages)` | proxied messages |
| `routes/passthrough.py:25,29` | `session.request(...)` | catch-all |

- No link drops the data: `grep -rn "ClientSession(\|TCPConnector(" buster_ripper/` → the **only** occurrence
  pair is `server.py:65-66`. Every consumer takes `app["session"]`; none constructs its own client.
- No queue/channel with an unread end; no task spawned without a caller (`on_startup` spawns the kv monitor and
  the dump flush, both cancelled and awaited in `on_cleanup`, `server.py:85-87`).
- Timing gap check: the only way the flag could be too late is if `on_startup` read the global before click
  wrote it. It cannot — the callback runs during parameter processing, strictly before the command body and
  therefore before `web.run_app` (`main.py:189`) drives the startup hooks. Witnessed at process level by
  TC-113/TC-114 run A, where the connector is built `ssl=False` in a real subprocess.
- Nothing bypasses the connector: `grep -rn "urlopen\|urllib.request\|import requests\|httpx" buster_ripper/`
  → zero; the package's only outbound HTTP client is aiohttp on the shared session.
- The QA-side data path is complete too: instrumented entry writes the JSON report in a `finally` before exit →
  `read_report()` after process exit (quiescence), and the peer's own request log is an independent channel.
  Every produced report key is consumed except `call_hook`, whose absence cannot produce a false pass (§Check 4).

## Check 4: Consumer Contract Tracing — PASS

- **`config.SKIP_TLS_VERIFY: bool`** — producer values: `False` (decorator default, `main.py:13`-style import
  binding) and click's `is_flag` `True`/`False`. Consumer: `server.py:65` `not <bool>` → `ssl`. aiohttp's
  parameter type is `Union[SSLContext, bool, Fingerprint]` and both booleans are handled. No third reachable
  value; nothing dispatches on it elsewhere.
- **Posture never inferred** — invariant clause "never from the URL scheme, the host, or the environment":
  `grep -rn "https" buster_ripper/` → **zero** occurrences; `grep -rnE "environ|getenv|SSL_CERT|_create_unverified|set_default_verify"`
  → **zero** occurrences in the package; `trust_env` is not passed and defaults `False` in aiohttp 3.13.3, so
  no environment-sourced CA file can move the posture. Sole input is the global.
- **The instrumented report (`tls_probe_entry.py` → gate)** — dispatch fields and their consumers:
  `total_connectors`/`total_sessions` (asserted `== 1` before any `[0]` index, so a missing element fails with
  an assertion, not an `IndexError`), `connectors[0].ssl_kwarg` (producer emits `repr(...)`; the gate compares
  the strings `"True"`/`"False"` — an `SSLContext` posture would produce a third value and fail loudly rather
  than merge), `sessions[0].ssl_kwarg` must be `"<absent>"` (a per-session posture is the W2 shape and is
  rejected), `calls[].connector_id`/`session_id` must be the single ids, `distinct_connector_ids_serving_calls`
  must be `[the one connector]`. `report["calls"]` empty ⇒ `assert_all_calls_through` fails ("the proxy made no
  outbound call at all"), so the diagnostic key `call_hook` being unconsumed cannot hide a dead hook.
- **Merged/coalesced data**: none in this increment — no batching, aggregation or coalescing is introduced
  (the dump cluster's coalescing was verified at MS-002 and is untouched here: `git diff` shows no change to
  `dump.py` in this increment).
- Per-call-site posture: `grep -rn "ssl" buster_ripper/` returns exactly one line — `server.py:65`. The
  structural half of `PI:T006/upstream_tls/optout_is_process_wide` is therefore true by construction, and TC-114
  proves it black-box rather than mirroring the implementation (the plan's stated pairing requirement).

## Check 5: Plan Fulfillment — PASS

T006's done-when, item by item, against the diff (the plan declares **ADDED ×9, REMOVED: nothing**):

| # | Plan clause | Evidence in the tree | Verdict |
|---|---|---|---|
| 1 | `config.SKIP_TLS_VERIFY: bool = False`, `KV_POLL_INTERVAL` global shape, `False`-default peer shape | `config.py:14-17` (comment + declaration between `UPSTREAM` and `STRIP_DATE`) | present |
| 2 | `--skip-tls-verify` `is_flag=True`, help naming certificate verification + self-signed, assigned **through its own callback** so `main()` gains no parameter | `main.py:64-67` + `:30-33`; `main()` signature unchanged (§Check 2a) | present |
| 3 | `server.py` builds explicit `TCPConnector(ssl=not config.SKIP_TLS_VERIFY)` passed as `connector=`, **inside `on_startup`**; unset path behaviour-preserving; connector is the only seam | `server.py:55,65-68`; parity proven against aiohttp 3.13.3 (§Check 2b) | present |
| 4 | smallest coherent L extraction of the config-application block into a module-private helper | `main.py:36-58` + call at `:139-149` | present |
| 5 | **committed** self-signed `127.0.0.1` certificate/key placed with the `tests/qa/` TLS gate; untrusted-issuer failure mode, not hostname mismatch | `tests/qa/tls_peer_cert.pem`, `tls_peer_key.pem`; `openssl x509`: `Subject == Issuer == CN=127.0.0.1`, `X509v3 SAN = IP:127.0.0.1`, no DNS name, valid 2026→2126 | present |
| 6 | unit coverage for the default and the flag→global link, plus the process-level `QAP-MS004` gate on the gate's own loopback peer (`FK-009` option 1) | `tests/test_tls_posture.py` (6, in the 301-test hermetic run) + `tests/qa/test_qap_ms004.py` | present |
| 7 | README serve-script section documents `SKIP_TLS_VERIFY` and the flag, stating it disables upstream certificate verification | `README.md:88-92` | present |
| 8 | serve-script env var defaulting off, **boolean** guard shape, not `-n` (whose truthiness would invert the default posture) | `serve_buster_ripper.sh:27` (`${SKIP_TLS_VERIFY:-0}`) + `:39` (`== "1"`) | present |
| 9 | REMOVED: nothing (default posture unchanged; `--upstream` default unchanged; no dump-cluster behavior change) | confirmed by construction: the old implicit-connector statement is replaced in-place; `config.py:13` `UPSTREAM` untouched; no `dump.py` change in this increment | **no removal owed, none outstanding** |

**Relay-vs-transfer test** (this check's stated purpose): the architectural goal is "the outbound TLS posture is
a *named, checkable site* rather than an inherited library default". The increment is additive by plan, so the
question is whether an old owner still holds what the plan moved. It does not: the library default is gone as a
*source* of the posture (`server.py:65` is now the only `ssl` in the package — grep-verified), no session-level or
per-call posture remains to fall back on, and the previously-invisible seam is asserted from outside by a real
process (`/v1/models`, `/metrics`, `/v1/messages` all through the one connector in both postures). This is the
transfer, not a relay: no old path survives alongside the new one.

`G-007`'s chain (`config.SKIP_TLS_VERIFY` global → CLI flag assigns it through its own callback → `server.py`
builds the shared `TCPConnector` carrying the posture → one connector governs every outbound call) is complete
end-to-end — that chain is precisely §Check 3's traced path, and its Met-when "the opt-out defaults OFF" is
witnessed at `serve_buster_ripper.sh:27/39` where a presence guard would have broken it.

Invariant cross-reference (`invariant-cross-reference`, narrowed to this cycle's packages):
`invariants/buster_ripper/server/invariants.md` — `Override: OI:buster_ripper.server/on_startup/shared_client_verifies_by_default`,
`Decision: preserve`, is **fully applied**: its `Reason` predicts exactly the diff ("the old wording clause
('built with no `ssl`/`connector=` argument') becomes literally false"), and the connector now carries the
posture. Both `PI:T006/upstream_tls/*` statements hold on the evidence in Checks 3-4 and 6; their `Verification:`
rows name TC-003/TC-113/TC-114, all of which pass. No planned `Override:` entry is left unapplied.

## Check 6: QA Milestone Closure — PASS

**Handle resolution.** Invocation `MS-004` matches the Milestone Matrix row; its `QA author pass` column is
`QAP-MS004` and the invocation's `QAP-MS004` matches qa-plan.md §QA Author Passes row `QAP-MS004`
(after-T006-local-Yield → MS-004 Yield Phase). Constituent task T006 passed local Yield
([051-01-yield](051-01-yield.md)). Scenarios whose `Required at` is exactly `MS-004`: **TC-003, TC-113, TC-114**
(plus the author pass's R1/R5 hostile tier). Each names a `VP-*` whose `Observable after` = `T006` and
`Validate at` = `MS-004` — matching the approved Validation Point Matrix. All three `VP-T006-*` points due at
this milestone have a selected scenario; no orphan. `VP-T006-optout-is-explicit` and
`VP-T006-dump-cluster-unaffected` are retired/not-created handles — correctly **not** authored.

**The `QAP-MS004` pass supplied the artifacts Check 6 must not author itself**: persistent gate
`tests/qa/test_qap_ms004.py`, contract draft and the boundary-inspection log — both in
[055-forward-qa](055-forward-qa.md) (§Contract Draft, §Revisions, §Boundary Inspection Exceptions: none raised;
`tests/test_tls_posture.py` and every implementation report were never opened by the author). Adequate: the
draft is written from the approved `VP-*`/`TC-*` records and the public CLI surface, and its Stage-2 revisions
moved witnesses to shapes that carry the same claim without weakening one.

**Fidelity inspection** — each test read against its approved contract (trigger / owner / visible result /
forbidden result / lifecycle / telemetry):

| Contract element | How the committed test honors it |
|---|---|
| TC-003 real trigger: real proxy subprocess, no opt-out, real `POST /v1/messages` | `Proxy(...)` spawns the real binary path through the instrumented entry; `extra_args` carries no opt-out; request is a real HTTP POST |
| TC-003 visible result + **W1** forbidden | `status != 200` **and** `MARKER not in body` **and** `peer.paths() == set()` **and** the failure *class* must be certificate verification (`assert_cert_verification_failure` requires `CERTIFICATE_VERIFY_FAILED` + `certificate verify failed`, rejecting a generic "Cannot connect"); `assert_single_client` rejects 0-or-≥2 connectors |
| TC-003 "refusal is a trust decision, not a broken peer" | `start_peer()` asserts `trusted_probe()` answered **before** `reset()`, so the same material demonstrably serves a client that trusts it, and the gate's own control traffic is excluded from the subject log |
| TC-113 same peer, same shape, opt-out → completes | `status == 200` **and** peer `MARKER` in the caller's body **and** the peer recorded the `POST /v1/messages` **and** zero handshake aborts; connector `ssl=False` |
| TC-114 one opt-out governs every outbound call, paired so it cannot be vacuous | run A (opt-out): all three paths served, `OUTBOUND_PATHS ⊆ call_paths`, one connector `ssl=False`; run B (default): `peer.paths() == set()`, cert-verification class, ≥1 abort, `OUTBOUND_PATHS ⊆ call_paths` (the call sites *were* exercised), one connector `ssl=True`; then `connector_a.ssl != connector_b.ssl` and `distinct_connector_ids_serving_calls == [the one id]` per run. Same `PATH_DEADLINE` both halves — the positive half proves the bound, so the absence claim is a refusal, not a short wait |
| forbidden: per-call-site posture | `sessions[0].ssl_kwarg == "<absent>"` in every test, plus `assert_all_calls_through` pinning every call to the single connector **and** the single session, plus `grep`-level structural confirmation (§Check 4) |
| lifecycle | bounded waits on the peer's own request log / `poll_until` — no fixed sleep; process-alive check after the refusal (`proc.poll() is None`) and a second request failing identically; real `SIGTERM`, `rc == 0`, no traceback |
| telemetry | none asserted — correct: `FK-008` option 1 ships no runtime posture record, and the project declares no telemetry-verification capability, so `trace-testing` is properly unengaged |
| harness amendment | `require_live_upstream=False` + `upstream=peer.url()` only in this gate (7 call sites); the gate additionally proves `calls leaving the gate's peer: none`, i.e. the amendment's "the remote upstream is not used by this gate" clause is checked, not assumed |

**Independent execution at phase scope (Yield Phase's own runs, not 055's):**

```text
$ .venv/bin/python -m pytest tests/qa/test_qap_ms004.py -o addopts= -v
6 passed in 44.26s        (tc003, tc113, tc114, r5, r1-dupflag, r1-url-shape — each PASSED individually)
$ .venv/bin/python -m pytest tests/qa -o addopts= -q -rs
21 passed in 58.71s
$ .venv/bin/python -m pytest tests/qa/test_qap_ms004.py tests/qa/test_qap_ms002.py -o addopts= -q
21 passed in 58.88s
```

First-party paired observation (over the committed harness/peer/entry, unchanged):
`evidence/ms004-tls-posture-gate.txt` — default posture: 502
`SSLCertVerificationError / [SSL: CERTIFICATE_VERIFY_FAILED] … self-signed certificate`, peer completed paths
`[]`, 7 aborted handshakes, 1 connector `ssl=True`, session `ssl=<absent>`, rc 0, no traceback; opt-out
posture: 200 carrying the peer's own marker, `/metrics` + `/v1/messages` + `/v1/models` each served once, 0
aborts, 1 connector `ssl=False`, rc 0. The flag is the only difference between the runs.

**Verdicts (this gate's authoritative evidence, per `G-007`):**

| Validation point | Scenario | Test | Verdict |
|---|---|---|---|
| `VP-T006-verify-on-by-default` | TC-003 | `test_tc003_untrusted_peer_refused_by_default` | **pass** |
| `VP-T006-optout-honored` | TC-113 | `test_tc113_same_peer_accepted_with_optout` | **pass** |
| `VP-T006-posture-is-process-wide` | TC-114 | `test_tc114_one_posture_governs_every_outbound_call` | **pass** |
| adversarial R5 (trust-identity pair) | — | `test_r5_trust_identity_pair_is_not_merged` | **pass** |
| adversarial R1 (duplicate flag occurrence) | — | `test_r1_duplicate_flag_occurrence_honored_once` | **pass** |
| adversarial R1 (upstream-URL shape) | — | `test_r1_upstream_url_shape_does_not_change_posture` | **pass** |
| `R2` durable-writer tier | — | not applicable to MS-004 (no file is written by this increment; the artifact writers are MS-001/MS-002 gates' surface) | **not_applicable** — recorded, not silently skipped |
| `R4` payload-decoder census | — | not applicable (no payload field added or changed; one decoder language: `aiohttp` + `click`) | **not_applicable** — recorded |

**Intermediate QA sweep.** Files naming T006 surface under `tests/`: `tests/test_tls_posture.py` (hermetic unit
suite — part of the 301, not a QA intermediate), `tests/test_serve_script_argv.py` (T011/T012 witnesses, own
tasks), and the three gate artifacts `tests/qa/{test_qap_ms004,tls_peer,tls_probe_entry}.py`. `tls_peer.py` and
`tls_probe_entry.py` are the gate's own retained support modules (imported/exec'd by it — 055 §Files Changed
records the roles), `harness.py` is the shared retainer that 055 **reused rather than duplicated**, and
`counted_proxy_entry.py` is correctly left unsupsumed (it counts file opens, a different witness). No unpromoted
intermediate exists, so nothing is deleted by this sweep. Worktree clean apart from this iteration file.

**MS-002 note (not this gate's verdict).** The live remote upstream answered `HTTP 200` during this pass, and
QAP-MS002's 15 gates therefore ran for real twice inside the combined runs above (21/21). That satisfies the
reachability precondition the handoff records as still owed for the QAP-MS002 re-run → MS-002 re-verdict
([US-016](decisions.md#us-016)). MS-002's re-verdict is a separate gate and is not issued here; it is now
unblocked and should be dispatched while reachability holds.

## Check 7: Integration Suite Execution — PASS

**Scope derived from the diff before running anything.** The changed production package is `buster_ripper`
(this increment: `config.py`, `main.py`, `server.py`, `serve_buster_ripper.sh`, `README.md`); it is the only
production package in the repository, so it has no *other* package downstream. The downstream consumers that do
exist are in-diff and were included on that basis:
- every route module and `backend.py` / `compaction.py` (they consume the one session whose construction this
  increment replaced) — covered by the hermetic suite's integration files (`tests/test_routes_integration.py`,
  `tests/test_server_hooks.py`, `tests/test_main_cli.py`, `tests/test_tls_posture.py`);
- `tests/qa/harness.py`, which this increment modified, and its **other** consumer
  `tests/qa/test_qap_ms002.py` (15 gates) — a shared-fixture change is upstream of every gate that imports it, so
  it ran in scope rather than being skipped as "not this milestone's";
- the dump cluster (`tests/test_dump*.py`, `tests/test_serve_script_argv.py`) — the T006 commit also tightened
  `--dump-flush-interval`/`--dump-flush-bytes` to `click.IntRange(min=1)`, which is upstream of MS-002's gates:
  every value those gates inject is ≥ 1 (`1`, `2`, `3`, `100_000`, `3600`, `10**12`), verified before running.

**Executed on this snapshot:**

```text
$ .venv/bin/python -m pytest -q                                   # full hermetic suite (Check 7 step 0)
301 passed, 23 warnings in 9.59s
Required test coverage of 80% reached. Total coverage: 93.88%
$ .venv/bin/python -m pytest tests/qa -o addopts= -q -rs          # both process-level gate files
21 passed in 58.71s
$ .venv/bin/python -m pytest tests/qa/test_qap_ms004.py tests/qa/test_qap_ms002.py -o addopts= -q
21 passed in 58.88s
```

The hermetic run is the one that could catch a regression in an unrelated module (per-task Yield runs only
affected tests); it reproduces the standing baseline exactly (301 @ 93.88%), as expected for a byte-identical
production tree. Nothing failed, so nothing was bisected. `tests/qa/` is excluded from the default run by
`norecursedirs` (`pyproject.toml:23`) — a rule pinned by `tests/test_pytest_config.py` — so the hermetic run
stayed network-free while the gate run was explicit.

**Exclusions, named:** `buster_ripper_legacy.py` — the tracked pre-refactor monolith, referenced by nothing under
`buster_ripper/` (grep: zero references anywhere) and explicitly excluded from scope by the approved plan
("Not tasks" section); `docs/eval.md` — documentation, no executable surface; no test selector, package, or spec
was dropped at any point, and no scope was reduced after a failure because no failure occurred.

**Shared-run rule applied:** this single independent Yield Phase invocation covers Check 6's required scope
(the three `QAP-MS004` scenarios + hostile tier, mapped by handle above) and Check 7's required scope (full
hermetic suite + both process-level gate files) on the same snapshot and the same harness configuration. The
runs above are mapped to each gate's tests; nothing was executed twice merely to produce a second report.
Forward QA's runs (055 §Harness Execution) are **not** counted as this gate's independent execution — the three
commands above are.

## Findings

No P1. No P2. Three P3 — all in this pass's QA/registry artifacts, none affecting a production path, a gate's
witness, or the milestone's evidence.

### [P3] Shared QA harness carries an exercised-nowhere `path=` keyword justified by a use the gate does not make
- **Priority**: P3
- **Check**: 1 (Caller Verification) — QA-side surface; Check 6 fidelity note
- **Location**: `tests/qa/harness.py:175-178` (`Proxy.url(path=…)`, `Proxy.post(body, timeout, path=…)`);
  claim in [055-forward-qa](055-forward-qa.md) §Files Changed
- **What is wrong**: `forward-qa` extended the shared harness with a `path` keyword "so a gate can drive the
  compaction-policy endpoint". No test in the repository passes it (`grep -rn "post(.*path=\|url(.*path=" tests/`
  → no call sites; the only use is `harness.py:182`'s internal default). The authored gate drives `/v1/models` and
  `/metrics` by starting the real proxy and observing the peer's request log, which is the stronger trigger — so
  the extension is unused surface in the one file every future gate imports, and the report's stated reason
  describes a consumer that does not exist. Severity is P3 rather than P2 because the surface is test
  infrastructure only, no assertion depends on it, and it cannot produce a false pass.
- **Evidence**: §Check 1 table row (zero callers); `tests/qa/test_qap_ms004.py` call sites listed above.
- **Data path affected**: none (production unaffected).
- **Reach**: unreachable-in-production — QA harness parameter; the exercised path is the default value.

### [P3] The gate's module docstring still advertises the identity near-miss the pass explicitly retracted
- **Priority**: P3
- **Check**: 6 (QA Milestone Closure — contract/documentation fidelity)
- **Location**: `tests/qa/tls_peer.py:12-14` vs `tests/qa/tls_peer.py:148-152` and
  [055-forward-qa](055-forward-qa.md) §Revisions item 1
- **What is wrong**: the module docstring states that because the certificate has `IP:127.0.0.1` and no DNS name,
  "the hostname-identity near-miss (connect as `localhost`) **is available** to the adversarial tier without any
  second certificate". 055's Stage-2 run established the opposite (a `localhost` refusal is not attributable to a
  trust decision on this host — nothing listens on `::1`), replaced that shape with the trust-anchor pair, and the
  class docstring two pages later says exactly that ("the identity controls this gate uses … not a second
  hostname"). The stale sentence is an open invitation to re-add a non-witnessing test. The committed tests do not
  contain the `localhost` variant, so no gate is wrong today.
- **Evidence**: `openssl x509 -ext subjectAltName` → `IP Address:127.0.0.1` only;
  `grep -n "localhost" tests/qa/test_qap_ms004.py` → no match; 055 §Revisions item 1.
- **Data path affected**: none.
- **Reach**: latent — documentation defect in a QA artifact.

### [P3] T006's deferred invariant referent was never elevated, though its precondition is long discharged
- **Priority**: P3
- **Check**: 5 (Plan Fulfillment — registry parity with the cluster's own precedent)
- **Location**: `invariants/buster_ripper/server/invariants.md:79-85` (§Deferred at elevation, row
  "upstream TLS verification opt-out | `buster_ripper/config.py:SKIP_TLS_VERIFY` + `buster_ripper/main.py:--skip-tls-verify` | T006 Yield");
  `invariants/index.md:14-17` ("the TLS referent remains deferred to `T006` Yield")
- **What is wrong**: the registry deferred one TLS referent because its authorized symbols did not exist at
  elevation time, with the named register-at point "T006 Yield". T006's local Yield ([051-01-yield](051-01-yield.md))
  did not perform the elevation (no registration or deferral-discharge language there), and the milestone gate now
  passes with both symbols in the tree (`config.py:17`, `main.py:64-67`). The dump cluster set the precedent for
  exactly this step — its six deferred referents were registered at T002 Yield (Checkpoint 64) and T003 Yield
  (Checkpoint 66), and its deferral table was updated accordingly. This is a durable-record gap, not a code or
  witness gap: both `PI:T006/upstream_tls/*` are registered as planned with witness rows (matching the dump
  cluster's still-`planned` statuses — status elevation is not this gate's action), and `E:T006/server/shared-client-construction`
  exists.
- **Evidence**: `grep -n "config.py:SKIP_TLS_VERIFY\|main.py:--skip-tls-verify" invariants/**/*.md` → appears only
  in the deferred tables; no `Entity:` block binds either symbol; `invariants/index.md` Handle Index has no
  `T006` config/main row.
- **Data path affected**: none.
- **Reach**: latent — registry hygiene.

## Assumption Routing (owned `Factual claims`)

`decisions.md` rows with `Factual claims` → `Verification: yield-phase:MS-004 PENDING` — two, both under
**[FK-008](decisions.md#fk-008)** (decisions.md:961-969), the user's own stated basis for choosing option 1
("One way you cannot establish connection, other way it works"). Both exercised through this gate's milestone
arbiters (`VP-T006-verify-on-by-default`, `VP-T006-optout-honored`) and adjudicated **HOLDS**; the ledger rows are
updated by this pass to `HOLDS` / `Consequence: none` with the locator below. No non-holding load-bearing claim,
so no User Authority Boundary finding and no escalation.

| Claim | Adjudication | Evidence locator |
|---|---|---|
| "With no opt-out and an untrusted-certificate HTTPS upstream, a request through the proxy does not complete, and the failure is a certificate-verification error" | **HOLDS** | [056-yield-phase](056-yield-phase.md) §Check 6 (TC-003 **pass**) + `evidence/ms004-tls-posture-gate.txt` §2 default-posture block — 502 with `SSLCertVerificationError / [SSL: CERTIFICATE_VERIFY_FAILED] … self-signed certificate`, peer completed-request set empty, 7 aborted handshakes |
| "With the opt-out passed against that same peer, the request completes" | **HOLDS** | [056-yield-phase](056-yield-phase.md) §Check 6 (TC-113 **pass**) + `evidence/ms004-tls-posture-gate.txt` §2 opt-out block — 200 carrying the peer's own `qa-ms004-peer-answered` marker, peer recorded the `POST /v1/messages` |

Legacy-record inspection: `FK-008` carries the canonical `Factual claims` field, so no detailed handoff/interview
record is its only home → no `MIGRATION-REQUIRED`. No owned claim is left `PENDING` or `UNCLEAR`.

## Deterministic Closure (pre-question gate applied)

```text
CLOSURE: absorbed
Finding: 056-P3-1 — `tests/qa/harness.py` `Proxy.post(path=)` / `Proxy.url(path=)` have zero consumers, while
  055 §Files Changed justifies them by a use the authored gate does not make.
Required outcome: the shared QA harness exposes only surface some gate exercises, and the author pass's record
  does not claim a consumer that does not exist.
Authority: (1) approved plan — QAP-MS004's scenario set is exactly TC-003/TC-113/TC-114, closed;
  (2) `qa-project` §Test file location / harness contract and §Conventions "Hermetic default suite";
  (3) ownership — `forward-qa` owns QA harness surface, not implementation Forward (055 declares no production
  source changed and this pass confirms it); (4) evidence — grep shows no `path=` call site anywhere in tests/.
Admissible candidates: none surviving (see rejected list).
Rejected candidates:
  - author a gate that uses `path=` -> broadens scope beyond the approved, already-covered scenario set; a new
    scenario needs plan authority, and TC-114's real trigger (the proxy's own probe + poll) is strictly stronger
    than posting to `/metrics` by hand.
  - keep the keywords and re-word 055's justification to "forward-looking convenience" -> preserves the verified
    defect (unexercised shared surface) and knowingly leaves an inaccurate durable record.
  - also remove `upstream=` / `require_live_upstream=` / `probe_report=` -> none of those is unused; all three
    have 7 live call sites, and removing them would break the FK-009 amendment this gate depends on.
Selected closure: drop the two unused `path` keywords (restore `Proxy.post(body, timeout=…)` / `Proxy.url()`),
  or exercise them in a gate if a later milestone needs them; correct 055's §Files Changed sentence in the same
  touch. Smallest complete remediation; diff-reducing.
Risk shape: decreased (less shared surface a future gate can rely on untested).
Deferred additive work: none — the rest of the amendment is exercised.
Owning route: `forward-qa` (next touch of `tests/qa/harness.py`); non-blocking for MS-004.
Question required: no
```

```text
CLOSURE: absorbed
Finding: 056-P3-2 — `tests/qa/tls_peer.py` module docstring still claims the `localhost` hostname near-miss is
  available to the adversarial tier, contradicting 055 §Revisions item 1 and the class docstring at :148-152.
Required outcome: the gate's documentation names the identity controls it actually uses, so a later reader does
  not re-add a variant whose outcome is not attributable to a trust decision.
Authority: (2) `FK-009` option 1 (loopback peer, TLS gate only) — the peer's identity controls are part of the
  approved proof route; (4) verified counterexample — the committed certificate carries `IP:127.0.0.1` and no DNS
  name, and 055 records that a `localhost` connection on this host fails on transport (`::1`), not on trust.
Admissible candidates: none surviving.
Rejected candidates:
  - re-add a `localhost` variant -> knowingly installs a gate that can fail for a reason no other test fails for
    and whose pass would not witness a posture; that is the defect 055 already corrected.
  - delete the sentence with no replacement -> admissible and observably equivalent, but loses the reason.
Selected closure: rewrite the module-docstring clause to state that the near-miss this gate uses is the
  trust-anchor pair (`trusted_probe` / `untrusted_probe`) over identical bytes, and that a DNS-name variant is
  not available from this material — mirroring the class docstring. (Deleting the sentence is the same outcome
  with a weaker record, so the version that preserves the reason wins.)
Risk shape: decreased — documentation matches the executable contract.
Deferred additive work: none.
Owning route: `forward-qa` (next touch of `tests/qa/tls_peer.py`); non-blocking for MS-004.
Question required: no
```

```text
CLOSURE: absorbed
Finding: 056-P3-3 — the TLS referent deferred "to `T006` Yield" was never elevated into the invariant registry.
Required outcome: every referent whose symbols exist in the tree is registered, per the registry's own
  no-dangling-binding rule and the precedent set at T002/T003 Yield.
Authority: (2) the registry is secondary authority after the code, and code wins — `config.py:17` and
  `main.py:64-67` now exist and are green; (3) ownership — registry projection is Authority Registrar's, not
  Yield Phase's; (4) evidence — `invariants/buster_ripper/server/invariants.md:79-85` and `invariants/index.md:14-17`
  still list the deferral; the dump cluster's equivalents are registered.
Admissible candidates: none surviving.
Rejected candidates:
  - leave it deferred until some later phase -> waits on its own acceptance and leaves a stale deferral row that
    contradicts the tree.
  - elevate `PI:T006/*` statuses to active at the same time -> beyond this finding; status transitions follow
    phase acceptance (Yield Red), and the dump cluster's `PI:*` set the precedent for staying `planned`.
Selected closure: register one `Entity:` block for the opt-out's injectability seam (binding
  `buster_ripper/config.py:SKIP_TLS_VERIFY` + `main.py:--skip-tls-verify`, gloss mirroring the
  `E:T003/dump/flush-trigger-parameters` shape: assigned once at parse time through the option's own callback so
  `main()` gains no parameter), clear the §Deferred row, and re-run `speeckle/scripts/rebuild_index.py`.
Risk shape: unchanged — record only, no code or test movement.
Deferred additive work: none.
Owning route: Authority Registrar (may be folded into this cycle's post-Yield-Red finalization pass);
  non-blocking for MS-004.
Question required: no
```

No finding has two materially different admissible outcomes that settled authority fails to rank, so **no user
question is owed** and no `PLAN-GAP` / `BLOCKED-APPROVAL` is emitted.

## Codex Findings Supplied at Dispatch

None. The orchestrator launched no merged Codex output with this invocation; if the parallel review returns
findings on this snapshot, they must be re-merged here — any source re-threading invalidates this verdict.

## Rejection Retry Cap

No rejection issued, so no counter is incremented. (`MS-002`/root T003 remains at 1 from
[035-yield-phase](035-yield-phase.md); untouched.)

## Blind Spots / Notes for the Next Agent

- 055 asked this gate to confirm two things rather than assume them. Both confirmed:
  (1) the `ClientSession._request` hook is private aiohttp surface, and this run observed `call_hook: installed`
  with `total_calls` 7 / 3 — the *primary* posture witnesses (peer request log, connector count and posture,
  session carrying no `ssl`) do **not** depend on it; what the call-site log uniquely witnesses is "no call left
  the gate's peer". If aiohttp ever removes `_request`, that assertion goes red rather than passing vacuously,
  which is the safe direction, so no change is required now.
  (2) the `rc == 0` + no-traceback lifecycle witness matches the project's declared teardown convention
  (`qa-project` §Teardown / §Synchronization) and the same contract TC-110 pins; it hardens rather than
  substitutes. It is, however, currently named only in 055's contract draft and not in `qa-plan.md`'s
  TC-003/113/114 `Lifecycle` clause (which speaks only of the peer) — one clause there would close the mapping.
  Deterministic, record-only, owner: the `qa-plan.md` scenario contract's author (Authority Registrar route). Not
  a separate finding: no test diverges from the approved behavior; only the durable mapping is thinner than the
  test.
- This milestone declares no write-bearing artifact and no telemetry, so `trace-testing` and the R2 tier are
  correctly unengaged; both are recorded as `not_applicable` above rather than omitted.

## Verdict

**APPROVED — MS-004 milestone gate PASSES.** Zero P1, zero P2; three P3 findings, each closed by the
deterministic-closure gate as `absorbed` with a named owner and no user question.

- `VP-T006-verify-on-by-default`, `VP-T006-optout-honored` and `VP-T006-posture-is-process-wide`: **pass**, on
  independently executed real-trigger evidence (`evidence/ms004-tls-posture-gate.txt`).
- `FK-008`'s two owned factual claims: **HOLDS**, ledger updated by this pass.
- Integration scope (hermetic 301 @ 93.88% + 21 process-level gates, twice) fully green; no scope reduction.
- Routing for the P3s: `forward-qa` ×2 (harness `path=` surface; `tls_peer.py` docstring) and Authority Registrar
  ×1 (T006 deferred referent elevation). One further **record-mapping note** (not a finding — no test diverges
  from the approved behavior) routes to Authority Registrar: name the `rc == 0` / no-traceback lifecycle witness
  in `qa-plan.md`'s TC-003/113/114 `Lifecycle` clause, where today only 055's contract draft carries it.
- Per this agent's contract, this verdict **does not** accept the phase: no Delivered chronicle entry, no child
  completion publication, no `HEAD` advancement. **Next: assumptions recheck, then Yield Red**, which owns final
  acceptance and publication. MS-004 is the session's final milestone, so after Yield Red's CLEAN the
  orchestrator dispatches Forward's `phase-boundary-finalize`, and the session-close full-suite gate still
  remains owed (including `QAP-MS001`, never authored, and the now-unblocked QAP-MS002 re-run → MS-002
  re-verdict: the live upstream answered HTTP 200 throughout this pass).

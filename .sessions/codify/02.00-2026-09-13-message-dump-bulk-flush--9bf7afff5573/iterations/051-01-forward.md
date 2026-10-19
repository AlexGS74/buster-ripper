# Iteration 051-01 — Forward — T006: Upstream TLS verification opt-out

## Orchestrator Prompt (verbatim)

> You are a subagent spawned by another session.
> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
> Task ID: T006

(Task text carried by the session's plan; T006 is dispatched per `US-016`'s user-approved re-order: "reorder and finish coding first" — T006 authorized to run before the MS-002 re-verdict.)

## Increment Brief

- **Task and goal (my words):** make the proxy verify the upstream's TLS certificate by default and add an explicit, process-wide `--skip-tls-verify` opt-out for self-signed upstreams — one shared `TCPConnector` carries the posture for every outbound call; wire the flag through CLI, serve script, and README; commit the self-signed `127.0.0.1` peer fixture the MS-004 QA gate needs.
- **Anchors touched:** upstream TLS verification opt-out → `buster_ripper/config.py:SKIP_TLS_VERIFY` + `buster_ripper/main.py:--skip-tls-verify` → **NEW** (RB-005) · shared client construction → `buster_ripper/server.py:on_startup` (was `:47`, now `:58-69`) → **extend**.
- **Will NOT create:** no QA gate test (QAP-MS004 is `forward-qa`'s at MS-004, per qa-plan + the QA Plan Boundary), no read-side tool, no new module/buffer symbol, no inbound TLS, no CA-bundle option.
- **Reuse evidence:** RB-005 replayed this iteration: `grep -rniE '\b(ssl|tls|insecure|noverify|cafile|truststore)\b'` over `buster_ripper/` + `tests/` → still zero code matches (only English "Verify" in docstrings); `ClientSession.__init__` signature re-inspected in installed aiohttp 3.13.3 → takes `connector=`, **no `ssl` parameter** (connector is the only seam); `TCPConnector.__init__` takes `ssl=True` default at `connector.py` (verified by reading `__init__` + `_merge_ssl_params` source: a `True`/`False` bool passes through unchanged, so `connector._ssl` is exactly the posture we pass). No viable owner exists; the NEW row stands.
- **Inventions:** none.

## Pre-flight gates

- **Plan approval:** Interview COMPLETE (historical 2026-09-17, `US-015` "approve, begin"); planning closed; `US-016` (2026-09-28) explicitly authorizes forward T006 before the MS-002 re-verdict. PASS.
- **QA contract:** `qa-plan.md` exists; T006's VPs (`VP-T006-verify-on-by-default`/TC-003, `VP-T006-optout-honored`/TC-113, `VP-T006-posture-is-process-wide`/TC-114) all validate at **MS-004 / QAP-MS004** — authored later by forward-qa, not here. `VP-T006-optout-is-explicit` NOT CREATED (FK-008 option 1: no runtime record; the startup summary is unchanged — verified: `main.py`'s log lines gained nothing).
- **Protected invariants:** the two `PI:T006/upstream_tls/*` are `Status: planned` **from this session** → implement toward them (done). No protected EI/OI contradicted; `OI:buster_ripper.server/on_startup/shared_client_verifies_by_default` preserve-provenance honored (unset path behaviour-preserving, see below).
- **Planned Task Context consumed:** claim-38 and claim-34 blocks (below), per their mandatory criteria.

## What was implemented

1. **`buster_ripper/config.py`** — `SKIP_TLS_VERIFY: bool = False` in Core proxy settings (after `UPSTREAM`), comment states verification-on-by-default and never-inferred-from-scheme. Follows `KV_POLL_INTERVAL` module-global shape; `False`-default shape of `STRIP_DATE`/`VERBOSE`.
2. **`buster_ripper/main.py`** —
   - `_set_skip_tls_verify` callback + `--skip-tls-verify` option (`is_flag=True`, `expose_value=False`, help states it disables upstream certificate verification and is for self-signed certificates), placed after `--upstream`. Assigns its global at parse time; `main()`'s parameter list unchanged (P stays 21 — ratchet held).
   - **Capacity-gate extraction (017 packet 2, plan-mandated):** the pure-assignment config block (the 17 `config.X = x` statements) moved into module-private `_apply_cli_config(**cli)`; `main()` calls it with the CLI locals. Side-effectful `dump_dir`/`stats_db` blocks stay in `main()`.
3. **`buster_ripper/server.py:on_startup`** — builds `aiohttp.TCPConnector(ssl=not config.SKIP_TLS_VERIFY)` and passes it as `connector=` to the package's only `ClientSession`, **inside `on_startup`** (running loop) per the claim-38-corrected attribution: the running-loop requirement is `TCPConnector.__init__`'s (`loop = loop or asyncio.get_running_loop()`); the session adopts the connector's loop. Unset path behaviour-preserving: `ssl=True` → `_merge_ssl_params` passes the bool through → same verified posture aiohttp's default connector yields. Docstring names it the process-wide posture site.
4. **`serve_buster_ripper.sh`** — `SKIP_TLS_VERIFY=${SKIP_TLS_VERIFY:-0}` + the **boolean** peer guard `[[ "${SKIP_TLS_VERIFY}" == "1" ]] && EXTRA_FLAGS="${EXTRA_FLAGS} --skip-tls-verify"` (STRIP_DATE/VERBOSE shape, NOT the `-n` shape — the settled default `"0"` is non-empty, so `-n` would invert the default posture). Usage-header example added. Probe matrix (evidence): unset/`0`/`x` → no flag; `1` → flag.
5. **`README.md`** — serve-script options section documents `SKIP_TLS_VERIFY=1` / `--skip-tls-verify`: skips upstream TLS certificate verification for self-signed certificates, ON by default, process-wide. Addition-only.
6. **`tests/qa/tls_peer_cert.pem` + `tls_peer_key.pem`** — committed self-signed fixture for `127.0.0.1` (CN + SAN `IP:127.0.0.1`, RSA-2048/SHA-256, valid to 2126 so it never expires mid-project), placed as the MS-004 gate's peer material. Failure mode is the **untrusted issuer** (issuer==subject, verified), not hostname mismatch. Key modulus == cert modulus (verified). Generated with openssl 3.6.3 (no test-time CLI dependency). No test consumes it yet — QAP-MS004 will.
7. **`tests/test_tls_posture.py`** (new, 6 tests) —
   - CLI: flag assigns `config.SKIP_TLS_VERIFY=True` at parse time (`make_context`, `skip_tls_verify` not in `ctx.params`); no-flag restores `False` over a prior mutation (import-bound decorator default, per claim-34 — no post-import config mutation assumed); `--help` lists the flag.
   - Posture: `on_startup` (backend.detect_backend + both spawn helpers mocked via `mocker`) builds `ClientSession` whose connector is the explicit `TCPConnector` with `_ssl is True` (default) / `is False` (opt-out); the session binds the connector's loop with **no RuntimeError** — claim-38's regression companion (no test asserts a `RuntimeError` on the default adoption path; one asserts its absence).
8. **`tests/test_main_cli.py`** — backlog-absorption evidence (below): zero/negative `--dump-flush-interval`/`--dump-flush-bytes` rejected (`IntRange(min=1)`), 4 parametrized cases; help-rendering assertion updated to the range form.

## Backlog pickup

- **Absorbed: `QB:dump-flush/trigger-value-validation`** (`buster_ripper/main.py:51-60` — "a positivity guard … is the remedy when next touching main.py options"; T006 touches exactly that surface). Remedy: `type=click.IntRange(min=1)` on both trigger options. Safety check before encoding: every value the existing `tests/qa/test_qap_ms002.py` passes is ≥1 (literals 1/2/3; constants 100_000, 10¹², 3600) — no gate impact. `--help` now renders `[default: 10; x>=1]` / `[default: 4194304; x>=1]` (defaults, flags, globals unchanged) — the accepted T003 help test's two rendering substrings updated accordingly (intent unchanged: options listed with defaults). Zero/negative values now die as Click `UsageError` before reaching the global, so the timer can never become a `asyncio.sleep(0)` hot loop.
- Not absorbed (no touch-set match): `QB:capacity-baseline/function-metrics` (plan-side artifact; its remedy is already reflected in the current capacity paragraph — planning-side to flip), `QB:render-body/gray-compound`, `QB:dump/file-size-yellow`, `QB:dump-flush/residual-fmt-mixing` (dump.py untouched), route/test-size items (files untouched).

## Capacity gate

- Additive surfaces: `config.py` (+5 lines, 120→124), `server.py` (+10, 128→138), `main.py` (+29 net, 164→193 — the new callback+option+helper minus the extracted block), `serve_buster_ripper.sh` (+3), README (+6). All files Green (<300). No deleting/replacing production surface.
- `main.py:main`: **L:52→46, N:1→1, C:4→4, P:21→21** (scanner-measured at HEAD vs working tree) — the plan-mandated L extraction ran; the pre-existing hard violations (Long Function, Too Many Parameters) are **preserved, not worsened** (ratchet satisfied). Note: the plan's "L≈31" was an organize-time hand count; the scanner's own baseline is L:52, and the same scanner shows −6 after the extraction.
- New `_apply_cli_config`: **L:17 N:0 C:1 P:1** — single gray metric (L), acceptable per the one-gray-metric rule; assignments-only straight-line body, `**cli` keyed by the CLI parameter names (greppable one-to-one with the old block).
- `server.py:on_startup`: scan-clean. All other touched files: scan-clean (`codify scan path <f> --git --detailed`).

## L→N→C→P compliance (new/changed functions)

| Function | L | N | C | P | Verdict |
|---|---|---|---|---|---|
| `main._set_skip_tls_verify` | 2 | 0 | 1 | 3 | green (mirrors the three accepted `_set_dump_*` callbacks) |
| `main._apply_cli_config` | 17 | 0 | 1 | 1 | gray on L only — one metric, acceptable |
| `main.main` (touched) | 46 | 1 | 4 | 21 | pre-existing hard violations, **improved** (52→46), not worsened |
| `server.on_startup` (touched) | 10 | 0 | 1 | 1 | green |
| test functions | ≤14 | ≤2 | ≤2 | ≤4 | green |

## Tests and coverage

- New: `tests/test_tls_posture.py` (6) + 4 parametrized IntRange cases in `tests/test_main_cli.py` (+ help-assertion update).
- **Full suite: 278/278 passed @ 93.88%** (baseline 268 @ 93.87%); `main.py`/`server.py` are coverage-omitted by config (TLS cluster is coverage-neutral for the gate, as the plan records); `config.py` 95%→95% (the new statement is import-covered).
- `tests/qa/` untouched except the two fixture files — `norecursedirs` keeps them out of collection; the QAP-MS002 gate file is untouched.

## Dependency-claim checks (recorded per contract)

- **`ClientSession.__init__` has no `ssl` kwarg; `connector=` is the only seam** — re-verified by `inspect.signature` against installed aiohttp 3.13.3 (plan/claim-38 asserted; confirmed).
- **`TCPConnector` requires a running loop at construction; session adopts the connector's loop; the `connector._loop is loop` guard is vacuous on this path** — verified by reading `connector.py` (`loop = loop or asyncio.get_running_loop()`) and the adoption path; unit test `test_session_binds_connectors_loop` pins the binding without asserting any `RuntimeError`.
- **`ssl=True` bool passes `_merge_ssl_params` unchanged → same verified context as the default connector** — verified by reading `_merge_ssl_params` source; the unset path is behaviour-preserving as the plan requires.
- **Click option callbacks fire for defaults** (load-bearing for "no flag restores `False` over prior mutation") — confirmed empirically by the passing `test_default_keeps_verification_on` and the T003-era `--dump-format` precedent.
- No corrections to any recorded decision (no STOP condition); no shared-claim conflicts — `assumptions/claims.md` reach for claim-34/claim-38 is T006-only (S3/S4/S5 scoping already absorbed by the routing pass).

## Planned Task Context — finding dispositions (not marked resolved; Yield moves statuses)

- **claim-38 / DC-claim-38-r1 (absorbed):** implemented exactly as corrected — connector built inside `on_startup`, passed as `connector=`; no test asserts the vacuous `RuntimeError`; the session-binds-loop witness exists. The plan text and Anchor Map row already carry the corrected attribution (the routing pass amended them); nothing for Forward to edit there.
- **claim-34 / DC-claim-34-r1 (absorbed):** the opt-out is bound through its option callback before `on_startup` can run (parse-time); default leaves verification on; both CLI invocations and both postures exercised without post-import config mutation (the no-flag test *sets* the global pre-invocation to prove the decorator default wins); no unrelated CLI option retrofitted; suite green.

## Failure-path evidence

| Unhappy path | Approved outcome | Evidence |
|---|---|---|
| Zero/negative flush-trigger value at CLI parse | UsageError before the global is touched (absorbed backlog guard) | 4 parametrized cases, `test_main_cli.py` |
| Opt-out requested via serve script with junk value (`x`, empty, `0`) | Flag NOT appended; verification stays ON (posture can't invert by typo) | guard matrix in `evidence/t006-tls-optout.txt` §2 |
| Opt-out set but config read late/never | N/A by construction — connector reads `config.SKIP_TLS_VERIFY` once, inside `on_startup`, after parse-time assignment | `test_optout_posture_disables_verification` |
| Connector constructed without a running loop | Cannot happen on the app path (`on_startup` is a coroutine); aiohttp raises at `TCPConnector` construction, not at session adoption | `test_session_binds_connectors_loop` |
| Untrusted-cert upstream, no opt-out (TC-003's unit half) | Connector carries `ssl=True` → verified context → handshake refuses untrusted issuer | `test_default_posture_verifies` + fixture issuer==subject; full E2E is QAP-MS004's |

## Task clauses vs. delivered (T006 ADDED list, all 8)

1. `config.SKIP_TLS_VERIFY: bool = False` ✅ (clause as written)
2. `--skip-tls-verify` flag, callback-assigned, help per clause ✅
3. Explicit `TCPConnector(ssl=not config.SKIP_TLS_VERIFY)` + `connector=` inside `on_startup`, behaviour-preserving unset path ✅
4. Smallest coherent extraction of the config-application block into a module-private helper ✅ (`_apply_cli_config`)
5. Committed self-signed `127.0.0.1` cert/key fixture with the `tests/qa/` TLS gate ✅ (`tls_peer_cert.pem`/`tls_peer_key.pem`)
6. Unit coverage for the default + flag→global link ✅ (the process-level `tests/qa/` gate is QAP-MS004's — not authored here, per the QA Plan Boundary) ✅
7. README serve-script section documents `SKIP_TLS_VERIFY` + flag ✅
8. Serve-script env var, `:-0` default, boolean `== "1"` guard (not `-n`) ✅
REMOVED: nothing — no dump-cluster behavior, no `--upstream` default, no connector-default change ✅

## Session-file updates (this iteration)

- `plan-summary.md`: T006 task row → implemented; Progress Overview/Current Phase → T006 implemented, next Yield.
- `handoff-context.md`: §Current Status replaced.
- `quality-backlog.md`: `QB:dump-flush/trigger-value-validation` → resolved with evidence.
- `evidence/INDEX.md`: T006 row added (Disposition left empty for the user).

## Durable signal

- Overrides exercised: none (no override register entry was needed — the TLS PIs were implemented toward, not against).
- Decision changes: none. The plan's "L≈31" main() figure was an organize-time hand count; the scanner baseline is L:52→46 — recorded here, no plan edit made (triad rule).
- Bug-fix rationale that matters later: the help-rendering change (`[default: 10; x>=1]`) is the *intended* visible effect of the absorbed positivity guard, not drift — any future agent diffing `--help` snapshots should read it against `QB:dump-flush/trigger-value-validation`.
- S6 recomposition trigger fired by this landing (`SKIP_TLS_VERIFY` wiring in the script — the seam's third boundary var) and S4 refresh is now unblocked (the TLS option/connector path is implemented): both are composition-boundary work for Forward Collect, not this task.

## Invariant Contradictions Raised

None. Both `PI:T006/upstream_tls/*` planned invariants are now implemented toward: verification on unless explicitly disabled (never inferred from scheme/host/env — only the flag/env-var-explicitly-`1` path), and one connector/one posture for every outbound call (the only `ClientSession`, consumed by `server.py` + all routes).

## Concerns (for Yield, non-blocking)

- `handoff-context.md` stands at 208 lines against the 200 cap — **pre-existing** (208 on arrival, before my edit); my §Current Status replacement is character-shorter than the text it replaced and no planning-owned record was deleted to force the cap. If compaction is wanted, the §Composition Routing history blocks are the natural candidates — they are routing-owned, so I left them.

- The scanner counts `main()` at L:46 (hard violation persists). The plan's own wording ("T006 additionally performs the smallest coherent L extraction") is satisfied — the extraction ran and L fell — but main() remains over-length with P=21, pre-existing and ratchet-preserved. Further reduction (e.g. extracting the startup-logging block) was **not** in the plan's mandated scope ("smallest coherent"), so I stopped at the mandated boundary.
- `tests/qa/test_qap_ms002.py` help-adjacent behavior: none asserted the `[default: 10]` form, so the gate is unaffected by the IntRange rendering; verified by grep before encoding (all gate values ≥1).
- `test_server_hooks.py`'s pre-existing unawaited-coroutine warnings (backlog-noted) are unrelated to this diff.

## Demo evidence

`evidence/t006-tls-optout.txt` — `--help` excerpt, serve-guard matrix (4 postures), fixture identity, 6/6 unit postures.
DEMO-EVIDENCE: .sessions/codify/02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/evidence/INDEX.md — no pause

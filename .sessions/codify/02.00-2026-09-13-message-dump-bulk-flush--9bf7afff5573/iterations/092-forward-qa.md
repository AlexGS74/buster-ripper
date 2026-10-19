# 092 — FORWARD QA (endpoint correction + gate re-entry on current bytes): QAP-MS001

## Orchestrator Prompt (verbatim)

```text
You are a subagent spawned by another session.
Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
Milestone: MS-001
Mode: endpoint correction + gate re-entry on current bytes

The gate upstream has been corrected by the user: it is `http://hyperipper.local:30000`, not `:30001`. The user's words: "So the upstream should be 30,000, not 30,001. 30,001 is another Buster Reaper instance. But 30,000 is up right now." The authority record was amended at Checkpoint 211 — `decisions.md` §US-009 carries a dated correction naming `:30001` as a peer proxy rather than the model server, and the in-repo contract `.agents/skills/qa-project/SKILL.md` already reads `:30000` in its Start command and Upstream rows.

Two lanes are yours and still name the old port: `tests/qa/harness.py:5` (the docstring) and `:39` (`UPSTREAM`), plus the `qa-plan.md` citations the registrar identified at `:13`, `:92` and `:144`. Correct those to the settled endpoint, with a dated note where the file's own conventions carry provenance. Do not touch `README.md`'s `:30001` occurrences or the app's `--port` default — those are the proxy's own listen port, not the gate upstream, and the registrar left them deliberately.

Then run the gates that the 091 boundary could not: the 32 process-level gates on this one HEAD — `test_qap_ms001.py` (11) + `test_qap_ms002.py` (15) + `test_qap_ms004.py` (6) — against the corrected endpoint, plus the hermetic default suite. `091-yield-phase` established Checks 1–5 from source and a 511 @ 96.71% hermetic baseline on these bytes; no production byte has moved since, so this is a transport re-run, not a re-derivation. `MS-004`'s upstream-TLS gate keeps its own loopback peer by the amendment recorded in the contract — that one does not use the remote upstream, and its 6 gates should be read with that in mind.

Preflight the endpoint before the run and report the probe result. If `:30000` is unreachable, fail closed exactly as 091 did — no stub, no substitution, no narrowing after a failure. `READY-PASSING` is not a verdict; the MS-001 Yield Phase owns `pass`/`fail` and will re-execute or accept your run on the record.
```

## Result

**`READY-PASSING`** — the endpoint lane is corrected and the 32 process-level gates plus the hermetic default suite are green on the corrected transport, with no scenario re-authored and no production byte moved.

## QA Pass

- **QA pass**: `QAP-MS001` — `MS-001` has exactly one author pass in `qa-plan.md` §QA Author Passes. The 32-gate set is the dispatch's own scope; `QAP-MS002` (15) and `QAP-MS004` (6) were re-run because `tests/qa/harness.py`, which all three share, changed in this pass — the same reason 091 named the 32-gate set.
- **Session**: `02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573` verified against `session-identity.json` (`session_uid: 9bf7afff-5573-402a-b102-827423a1dded` → suffix `9bf7afff5573`). MATCH; not `BLOCKED-SESSION-IDENTITY`.
- **Constituent tasks**: `MS-001 = T001, T002, T005`, all accepted by local Yield (`plan-completed.md:7,8,10`). Not `BLOCKED-PLAN — milestone incomplete`.
- **Scheduling**: `QAP-MS001` runs after every constituent task passes local Yield and before the MS-001 Yield Phase — the position 091 occupied. Not `BLOCKED-PLAN`.
- **Validation points / scenarios in scope**: `VP-T001-writes-deferred`, `VP-T001-dump-failure-isolated`, `VP-T002-documented-format-preserved`, `VP-T002-default-encoding`; `TC-100`, `TC-001`, `TC-107`, `TC-108` — unchanged from 070/077/087.
- **Repository state**: `HEAD = 39585db` (Checkpoint 211), branch `refactor/package-structure`. `git diff --stat 83de78c..HEAD -- buster_ripper tests pyproject.toml README.md serve_buster_ripper.sh` → **empty**: 091's byte basis is intact. The two commits since 091 (`add0585`, `39585db`) touch session records only, and the only uncommitted changes at run time are this pass's QA files.

## Endpoint Correction (the bytes this pass moves)

Preflight at `2026-10-17T21:42:27Z`, no ambient proxy, before any edit or gate:

```
:30000 GET /v1/models -> HTTP 200 in 0.007860 s
  {"object":"list","data":[{"id":"DeepSeek-V4.1-Flash",...,"owned_by":"vllm",...}]}
:30001 GET /v1/models -> curl exit 7, "Couldn't connect to server" (no listener)
```

`:30000` answers as the vLLM-compatible model server the proxy expects; `:30001` has no listener. This matches the handoff's bounded observation and `decisions.md` §US-009's dated correction.

| File | Line | Change |
|---|---|---|
| `tests/qa/harness.py` | `:5` | module docstring names `http://hyperipper.local:30000`; a dated paragraph records the 2026-10-17 correction, that `:30001` is a peer Buster Ripper proxy, and that the app's `--port` default and `README.md`'s `:30001` occurrences are the proxy's listen port and stay unchanged |
| `tests/qa/harness.py` | `:45-47` | `UPSTREAM = "http://hyperipper.local:30000"` with a two-line dated provenance comment citing `decisions.md` §US-009 |
| `qa-plan.md` | `:13-16` | §Overview upstream corrected, dated 2026-10-17, `:30001` named as the peer proxy |
| `qa-plan.md` | `:94` | QAP-MS004 note: "does not require `hyperipper.local:30000` (the gate upstream since the 2026-10-17 correction of `:30001`)" |
| `qa-plan.md` | `:147` | `TC-100` trigger corrected to `--upstream http://hyperipper.local:30000`, with the dated note |

Deliberately untouched, per the dispatch and `decisions.md` §US-009: `README.md`'s `:30001` occurrences (the documented `ANTHROPIC_BASE_URL` / app listen port) and `main.py`'s `--port` default. `tests/qa/test_qap_ms004.py` and `tls_peer.py` were already independent of the constant and are byte-identical.

## Contract Draft

Stage 1 re-derived from the approved records (`qa-plan.md` §Scenario Contracts, `plan-summary.md` §Validation Point Matrix, `qa-project` harness declaration, `README.md` §Dump mode / §Reading the index) and byte-identical to 070's draft — this pass changes the transport endpoint only, never an assertion. No implementation byte was read to derive it.

| Scenario | Real trigger | Observable assertions | Forbidden assertions | Lifecycle proof | Telemetry proof | Harness entry |
|---|---|---|---|---|---|---|
| **TC-100** `VP-T001-writes-deferred` | real proxy, fresh `--dump-dir`, `--dump-format change-focused`, **no flush parameter**; `N=3` turns | every turn acknowledged with no artifact under the dump root; post-run one `change-focused.log` holding all three turns; every instrumented write targets that one file | never the session directory's absence; never model output; never a flush parameter | `rc == 0`; the deferral ends at a flush, proven by the same directory being non-empty post-run | none declared | `Proxy(..., extra_args=["--dump-format", COLLAPSED], counted_report=…)` |
| **TC-001** `VP-T001-dump-failure-isolated` | (a) documented path with an unwritable session dir, faulted turn + healthy control; (b) collapsed path, raise surfacing in the shutdown flush | faulted request's status class equals the healthy control's and `< 500`; `PermissionError` on stderr; healthy sibling still captures; `rc == 0` | never the fault wording; never a 5xx caused by dump behavior; never a body substring as the oracle | process keeps serving, exits `rc == 0` including a raise inside `on_cleanup` | none | `Proxy(...)` with pre-created unwritable `DUMP_DIR/<sid>/` |
| **TC-107** `VP-T002-default-encoding` | no `--dump-format`, no `DUMP*` env override; `N=3` turns | `turn_000.json`, `turn_001.diff`, `turn_002.diff`, `index.txt`; no `*.log` | never the collapsed layout | `rc == 0`; no session dir before any request | none | `Proxy(dump, logs)` — defaults only |
| **TC-108** `VP-T002-documented-format-preserved` | `--dump-format documented` through the instrumented entry; `N=3`, no flush parameter | `N + 1` files; `turn_000.json` parses and carries the turn; every `index.txt` line matches `^turn [0-9]{3}  [0-9]{2}:[0-9]{2}` with turn number, msgs/tools counts, `FULL`/`KB`, `+A/-R lines`; each diff is a unified diff; **write ops `= 2N`** | never a drifted line format, missing/renamed file, `.log`, or a count other than `2N` | `rc == 0`; counts per session | none | `Proxy(..., counted_report=…)` via `harness.COUNTED_ENTRY` |

Tiers unchanged in bytes (R1 hostile ×2, R2 symlink containment, R5 near-miss `sid` pair), and `022` still honored: `grep -n "dump-flush" tests/qa/test_qap_ms001.py` → the docstring rule at `:27` only; no gate passes either flush flag.

## Boundary Inspection Exceptions

None. The endpoint is declared by `qa-project` and `decisions.md` §US-009, so no implementation source was read to connect the harness; the constant is the harness's own declaration. No assertion was derived from implementation behavior.

## Files Changed

| File | Change |
|---|---|
| `tests/qa/harness.py` | endpoint constant + docstring corrected to `:30000`, dated provenance note; no other byte |
| `.sessions/…/qa-plan.md` | three `:30001` upstream citations corrected, dated notes added |
| `.sessions/…/evidence/ms001-reboundary-092-gates-30000.txt` | new — preflight probe, full 32-gate verbose run |
| `.sessions/…/evidence/ms001-reboundary-092-hermetic.txt` | new — full hermetic default run with coverage |
| `.sessions/…/iterations/092-forward-qa.md` | this report |
| `.sessions/…/plan-summary.md` | §Forward QA status row: this re-entry recorded |
| `.sessions/…/handoff-context.md` | §Current Status: this pass's result; 091's restated detail moved verbatim to `handoff-archive.md` |

Zero production bytes, zero scenario assertions, zero `VP-*`/`TC-*` semantics changed.

## Harness Execution

```
$ .venv/bin/python -m pytest tests/qa/test_qap_ms001.py tests/qa/test_qap_ms002.py \
      tests/qa/test_qap_ms004.py -o addopts= -v
32 passed in 61.80s        # HEAD 39585db; full log in evidence/ms001-reboundary-092-gates-30000.txt
```

| Run | Result |
|---|---|
| Preflight `require_upstream()` target, direct probe | `:30000` → HTTP 200 in 7.86 ms, vLLM-shaped model list; `:30001` → curl exit 7, no listener. Fail-closed path not triggered, no stub and no substitution taken |
| `QAP-MS001` — `test_qap_ms001.py` (TC-100, TC-001 ×2, TC-107, TC-108 + 2 sensitivity, R1 ×2, R2, R5) | **11 passed** |
| `QAP-MS002` — `test_qap_ms002.py` (TC-101…TC-112 + R1/R2/R5 tiers), shared-harness consumer on the live upstream | **15 passed** |
| `QAP-MS004` — `test_qap_ms004.py` (TC-003, TC-113, TC-114 + R1 ×2, R5), own loopback self-signed peer | **6 passed** — by the contract's `MS-004` amendment this gate never uses the remote upstream, so its pass is a regression witness for the shared harness bytes, not an observation of `:30000` |
| Combined process-level tier | **32/32 gates green** |
| Hermetic default `uv run pytest` / `.venv/bin/python -m pytest -q` | **511 passed @ 96.71%**, coverage floor 80% reached — identical to 091's baseline; log in `evidence/ms001-reboundary-092-hermetic.txt` |

`MS-004`'s independence is re-proved by construction, not assumed: every one of its six gates passes `upstream=peer.url(), require_live_upstream=False` (`test_qap_ms004.py:174,223,290,322,406,434/462`), so the endpoint correction cannot have reached them. Their green result therefore witnesses only that the shared `Proxy`/`read_text`/`log_artifacts` bytes still behave; the remote-upstream evidence comes from the 11 + 15 gates above.

## Result

**`READY-PASSING`** — every linked scenario is authored, executable, and observes its approved contract on the corrected endpoint at `HEAD 39585db`; `pass`/`fail` on the four `VP-*` remains the MS-001 Yield Phase's to write.

## Next

MS-001 Yield Phase: re-execute `QAP-MS001` or accept this run on the record (same `HEAD`, same tree) and issue the four verdicts; Checks 1–5 and the 511-test hermetic baseline from [091-yield-phase](091-yield-phase.md) still stand, so no source re-derivation is owed. Yield Red still owns `resolved`.

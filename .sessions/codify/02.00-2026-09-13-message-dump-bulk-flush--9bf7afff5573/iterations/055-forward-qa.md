# Iteration 055 — Forward QA (QAP-MS004 author pass)

## Orchestrator Prompt (verbatim)

```text
You are a subagent spawned by another session.
Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
QA Pass: QAP-MS004
```

## QA Pass

- `QAP-MS004` — runs after `T006` accepted by local Yield ([051-01-yield](051-01-yield.md) §Yield
  Verdict: **ACCEPT**), before `MS-004` Yield Phase. `MS-004`'s only constituent task is `T006`, so the
  milestone is complete for this pass (qa-plan.md §QA Author Passes row `QAP-MS004`).
- Session handle verified against `session-identity.json`: `02.00-2026-09-13-message-dump-bulk-flush`
  + UID `9bf7afff-5573-402a-b102-827423a1dded` → the `--9bf7afff5573` handle. No mismatch.
- Validation points: `VP-T006-verify-on-by-default`, `VP-T006-optout-honored`,
  `VP-T006-posture-is-process-wide`.
- Scenarios: `TC-003`, `TC-113`, `TC-114` (qa-plan.md §QA Gate Matrix + §Scenario Contracts, the
  `TC-003 / TC-113 / TC-114` group).
- Gate handles: the milestone verdict is `MS-004` Yield Phase's, not this pass's.
- Harness contract: `.agents/skills/qa-project/SKILL.md` §Upstream — **the `MS-004`-scoped amendment**:
  this one gate does not use the live remote upstream; it starts its own loopback self-signed HTTPS peer
  (`https://127.0.0.1:<peer-port>`) with committed certificate material
  (`tests/qa/tls_peer_cert.pem`, `tests/qa/tls_peer_key.pem`, `CN=127.0.0.1`,
  `X509v3 SubjectAlternativeName = IP:127.0.0.1`, valid to 2126). Every other gate keeps the remote
  upstream; this gate must not require it.
- Intermediates considered for promotion: `tests/qa/harness.py` (shared process-level fixture location
  created by QAP-MS002) is **reused, not duplicated** — extended with two keyword arguments so a gate can
  name its own upstream and skip the live-upstream preflight. `tests/qa/counted_proxy_entry.py` is the
  pattern for a QA-side instrumented subprocess entry; it counts file opens, not client constructions, so
  it is **not** subsumed and **not** reused here — `TC-114`'s "instrumented call sites / one-connector
  assertion" needs a distinct entry (`tests/qa/tls_probe_entry.py`). No QAP-MS002 scenario touches `T006`'s
  surface, so nothing from `test_qap_ms002.py` is promoted or retired by this pass.
- Telemetry: none. The project declares no telemetry-verification capability and `FK-008` option 1 ships no
  runtime posture record, so no scenario asserts a posture log line; `trace-testing` is therefore not
  engaged. Every assertion is on a completed/aborted request, the peer's own request log, an exit code, or
  an instrumented construction count.

## Contract Draft

Stage 1 draft — written from the approved `VP-*`/`TC-*` records, `qa-project`, and the public CLI/runtime
surface (`buster-ripper --help`, `README.md` §Serve script options) **before** any implementation source,
implementation unit test, or implementation iteration report was read. Later work may connect the harness
but must not weaken or replace these assertions.

| Scenario | Real trigger | Observable assertions | Forbidden assertions | Lifecycle proof | Telemetry proof | Harness entry |
|---|---|---|---|---|---|---|
| TC-003 (`VP-T006-verify-on-by-default`) | Real proxy subprocess, `--upstream https://127.0.0.1:<peer>` with **no** `--skip-tls-verify`; real `POST /v1/messages` through it | The proxied request does not complete against the peer (no `2xx` carrying the peer's body); the failure's error class is certificate verification, observable in the evidence bundle; the peer records **zero** completed requests and **≥ 1** aborted TLS handshake; instrumented: exactly one `TCPConnector` and one `ClientSession`, the connector built in the verifying posture; control probe with the peer cert as the trust anchor completes the same handshake, so the refusal is the trust decision and not a broken peer | **W1** — the self-signed peer accepted with no opt-out (peer sees a completed request / caller gets the peer body); zero or more than one connector; a failure with no certificate-verification error class anywhere | Process still serving after the failure: a second request fails identically, port still accepting, clean `SIGTERM` exit with no traceback | none (no posture record ships) | `tests/qa/test_qap_ms004.py::test_tc003_untrusted_peer_refused_by_default` |
| TC-113 (`VP-T006-optout-honored`) | **Same** peer, **same** request shape, proxy started **with** `--skip-tls-verify` | The request completes: `200` with the peer's marker body, and the peer records the completed `POST` on `/v1/messages`; instrumented: still exactly one `TCPConnector`/`ClientSession`, now built in the non-verifying posture — the two postures come from one connector | **W2** — the flag accepted but the connector built without it, i.e. both postures behave identically (peer refuses here, or accepts in TC-003); a per-call-site posture (a second connector/session) | Process healthy across the request; clean `SIGTERM` exit; exit code and stderr show no traceback | none | `tests/qa/test_qap_ms004.py::test_tc113_same_peer_accepted_with_optout` |
| TC-114 (`VP-T006-posture-is-process-wide`) | With the opt-out, drive real `POST /v1/messages` and the compaction-policy request; wait on the peer's own observed-path set for `/v1/messages`, `/v1/models` and `/metrics` | All three distinct outbound paths reach the peer through the **one** instrumented session/connector; paired negative half on the same peer within the **same** bound: with no opt-out the peer records zero completed requests on any path — non-vacuous because the opt-out run is the positive control that proves all three paths are observable inside that bound | Any path present in one posture and absent in the other in a way that shows a per-call-site posture; more than one connector/session; the proxied request succeeding at the peer while `/v1/models` or `/metrics` never does (and vice versa) | Bounded wait on an application-side observable (the peer's request log), never a fixed sleep; process exits cleanly in both halves | none — the peer's request log and construction counts are the evidence | `tests/qa/test_qap_ms004.py::test_tc114_one_posture_governs_every_outbound_call` |
| R1/R5 hostile variants | Same peer, hostile identity and hostile CLI shapes: `https://localhost:<port>` against the `IP:127.0.0.1`-only SAN; a duplicate `--skip-tls-verify` occurrence | Hostname-identity near-miss: verification refuses the mismatched identity and the opt-out accepts it (two identities over identical cert material must not merge); duplicate flag occurrence still yields one connector in the non-verifying posture and one completed request | A trust decision that merges the two peer identities (either both accepted, or the mismatch accepted with no opt-out); posture inverting on a repeated flag | Process stays up in both shapes; clean exit | none | `tests/qa/test_qap_ms004.py::test_tc003_*` / `test_tc113_*` variant tests |
| R2 / R4 tiers | — | **Not applicable to `MS-004`, recorded rather than skipped silently.** `R2` (durable writer into a pre-existing tree): this milestone writes no file; its only new surface is an outbound client posture, and the artifact writers are `MS-001`/`MS-002` gates' surface (`TC-102`…`TC-109` already carry that tier). `R4` (payload decoder census): no payload field is added or changed, and the project has exactly one decoder language (`aiohttp` + `click`, `pyproject.toml:5-8`) | — | — | — | — |

### Revisions the Stage-2 run forced on the *harness*, not on the contract

The draft above is the contract. Two of its rows named a shape that turned out not to exist as drawn; the
assertions were moved to a shape that carries the same witness, and nothing was weakened:

1. **`https://localhost:<port>` hostname near-miss → replaced by the trust-anchor near-miss.** The draft
   assumed a second *connection identity* was available from the one committed certificate. It is not: the
   certificate carries `IP:127.0.0.1` and no DNS name, so connecting as `localhost` depends on how the host
   resolves it (`::1` first on this machine, where nothing listens) — a refusal there would not be
   attributable to a trust decision, and an acceptance test could fail on a transport error. The pair this
   milestone actually puts on a knife edge is *identical bytes, two client trust postures*, which the peer
   can produce deterministically: `trusted_probe()` (the committed cert as the trust anchor) versus
   `untrusted_probe()` (the default store). That is what `test_r5_trust_identity_pair_is_not_merged`
   asserts, and it is the sensitivity check TC-003 needs. No assertion in any `TC-*` row was removed.
2. **Peer-side abort record is asserted only for the proxy's client.** Under TLS 1.3 the stdlib client in
   `untrusted_probe()` sends `close_notify` as it tears down, so the peer records *no* accept error at all,
   while `aiohttp`'s client leaves `SSLEOFError [SSL: UNEXPECTED_EOF_WHILE_READING]` at `accept()`. The
   peer-side witness is therefore required where it is real (TC-003, TC-114 run B, the R1 URL-shape run)
   and explicitly not claimed in the R5 pair, whose proof is the two client outcomes plus the completed-
   request count of exactly one. The reason is in the test, so a later reader does not "fix" it back.

## Boundary Inspection Exceptions

**None raised.** No implementation source body, implementation unit test (`tests/test_tls_posture.py` was
not opened), or implementation Forward iteration report was read.

The one Stage-2 connection question that could have justified an exception — *what makes the `/metrics`
poll and the `/v1/models` probe occur, so the gate can drive them as `TC-114` requires?* — was answered by
**running the harness** instead: with the gate's own peer as the upstream, both paths arrive at the peer
automatically within ~0.13 s of process start (the `/v1/models` backend probe and the `/metrics` poll), and
`/v1/messages` arrives on the proxied request. Because the answer came from the peer's observation log, the
trigger each scenario drives is "start the proxy, then send one turn", with the wait taken on the peer's
observed-path set (`poll_until`, the declared timer-gate mechanism) rather than on a fixed sleep — and the
instrumented call-site log independently confirms all three paths were dispatched in both postures.

## Files Changed

New QA artifacts (all under `tests/qa/`, the location `qa-project` §Test file location authorizes):

- **`tests/qa/test_qap_ms004.py`** — the gate. 6 tests: `TC-003`, `TC-113`, `TC-114` (two paired runs),
  the R5 trust-identity pair, and two R1 hostile variants (duplicate flag occurrence; upstream-URL shape).
- **`tests/qa/tls_peer.py`** — the `MS-004` amendment's loopback self-signed HTTPS peer: per-connection
  accept accounting (completed request vs. client-aborted handshake vs. post-handshake reset), a
  reset-on-`start_peer()` subject log, and the two trust controls (`trusted_probe` / `untrusted_probe`).
  Serves `/v1/models`, `/metrics`, `/v1/messages`, `/v1/chat/completions`, `/v1/messages/count_tokens` and
  a generic `200`, all carrying `MARKER = b"qa-ms004-peer-answered"` so "this answer came from the peer" is
  checkable and nothing else on the machine can produce it.
- **`tests/qa/tls_probe_entry.py`** — QA-side instrumented subprocess entry (the `counted_proxy_entry.py`
  pattern, applied to the client layer instead of the file layer): wraps `aiohttp.TCPConnector.__init__`,
  `aiohttp.ClientSession.__init__` and `ClientSession._request` to write a JSON report of every connector
  and session constructed, the posture each was spelled with, and which connector served every outbound
  URL. Nothing in `buster_ripper` is patched.

Modified QA harness (shared fixture location — reused, not duplicated, per §QA Pass):

- **`tests/qa/harness.py`**
  - `Proxy(..., upstream=UPSTREAM, require_live_upstream=True, probe_report=None)` — the amendment's hook.
    Defaults preserve every existing gate exactly; `require_upstream()`'s fail-closed preflight now runs
    only for gates that require the live remote upstream, and `probe_report` selects the client-posture
    entry (mutually exclusive with `counted_report`).
  - `Proxy.post(body, timeout, path="/v1/messages")` / `Proxy.url(path=...)` — so a gate can drive the
    compaction-policy endpoint; existing callers pass positionally and are unaffected.
  - Module docstring records the amendment so the exemption is visible where the harness is learned.

**Production source, implementation unit tests, `VP-*`/`TC-*` semantics, invariant records and
implementation Forward reports: unchanged.** The committed certificate material is reused as-is.

## Harness Execution

All commands from the repository root. The `tests/qa/` gates are selected explicitly; the hermetic default
run does not collect them (`norecursedirs` includes `qa`, `pyproject.toml:23`).

### 1. The authored gate

```text
$ .venv/bin/python -m pytest tests/qa/test_qap_ms004.py -o addopts= -q
......                                                                   [100%]
6 passed in 44.26s
```

Three consecutive runs: `44.24s`, `44.26s`, `44.26s` — 6 passed each, no flakes.

### 2. Observed evidence, one run per posture (the paired proof)

```text
===== posture optout: --skip-tls-verify | upstream=https://127.0.0.1:55686 =====
proxied POST /v1/messages -> HTTP 200
  body[:200] = b'{"id": "qa-peer-msg", ... "text": "qa-ms004-peer-answered"} ...'
peer: completed paths   = ['/metrics', '/v1/messages', '/v1/models']
peer: counts            = {"/metrics": 1, "/v1/messages": 1, "/v1/models": 1}
peer: handshake aborts  = 0 []
peer: control probes    = [{"label": "trusted", "answered": true}]
proxy: exit rc          = 0  (traceback in stderr: False)
client: connectors      = 1 -> ssl_kwarg=False kwargs=['ssl']
client: sessions        = 1 -> connector_kwarg=True ssl_kwarg=<absent>
client: outbound calls  = ['/metrics', '/v1/messages', '/v1/models'] distinct connector ids=[4421282384]
remote upstream untouched: True

===== posture default: NO opt-out | upstream=https://127.0.0.1:55693 =====
proxied POST /v1/messages -> HTTP 502
  body[:200] = b'{"error": "Cannot connect to host 127.0.0.1:55693 ssl:True [SSLCertVerificationError: '
               b"(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self-signed certificate"
               b' (_ssl.c:1032)\')]"}'
peer: completed paths   = []
peer: counts            = {"/metrics": 0, "/v1/messages": 0, "/v1/models": 0}
peer: handshake aborts  = 5 ['SSLEOFError [SSL: UNEXPECTED_EOF_WHILE_READING] ...', ...]
proxy: exit rc          = 0  (traceback in stderr: False)
client: connectors      = 1 -> ssl_kwarg=True kwargs=['ssl']
client: sessions        = 1 -> connector_kwarg=True ssl_kwarg=<absent>
client: outbound calls  = ['/metrics', '/v1/messages', '/v1/models'] distinct connector ids=[4442139216]
certificate oracle: ACCEPTED the real evidence shape
remote upstream untouched: True
```

Reading: the only difference between the two runs is the flag; the connector count is `1` in both, its
`ssl` posture is `False`/`True`, the peer serves all three call sites in one and none in the other, and the
error class the caller sees is `SSLCertVerificationError / CERTIFICATE_VERIFY_FAILED`. W1 and W2 are each
excluded by a different half of the pair, so neither witness can survive silently.

### 3. Oracle falsification — every assertion rejects its bad-evidence shape

```text
REJECTED as required :: assert_cert_verification_failure(502, 'Cannot connect to host', '')
REJECTED as required :: assert_cert_verification_failure(502, 'SSLError: handshake failure', '')
ACCEPTED real evidence :: certificate oracle is not inverted
REJECTED as required :: assert_single_client(2 connectors)
REJECTED as required :: assert_single_client(2 sessions)
REJECTED as required :: assert_single_client(session carries ssl=False)
REJECTED as required :: assert_single_client(session on an unrecorded connector)
ACCEPTED real shape  :: one-connector assertion is not inverted
REJECTED as required :: assert_all_calls_through(a call that left the peer)
REJECTED as required :: assert_all_calls_through(a call on a different connector)
REJECTED as required :: assert_all_calls_through(no calls at all)
REJECTED as required :: assert_clean_shutdown(rc=1)
REJECTED as required :: assert_clean_shutdown(traceback in stderr)
```

A generic "Cannot connect" is *not* enough to pass TC-003 — the oracle requires the verification failure
specifically. That run is scratch, not persisted; the oracles it exercised are the ones the gate imports.

### 4. The amendment is real in both directions

With `harness.UPSTREAM` pointed at an unroutable name (`hyperipper-blackhole.invalid`) — i.e. the live
remote upstream declared unreachable, the condition that currently holds the QAP-MS002 re-run:

```text
QAP-MS004: 6 passed in 44.26s      exit=0   (gate owns its peer; no live-upstream dependency)
QAP-MS002: 15 failed in 0.10s      exit=1   (live-upstream gates still fail closed, fast)
```

So the exemption did not leak: `TC-101`…`TC-112` still refuse to run against a dead upstream, while this
gate is executable on this machine right now.

### 5. No regression to the shared harness or the hermetic suite

```text
$ .venv/bin/python -m pytest tests/qa/test_qap_ms002.py -o addopts= -q
15 passed in 14.83s

$ .venv/bin/python -m pytest tests/qa -o addopts= -q          # both gates, one invocation
21 passed in 58.53s

$ uv run pytest                                               # hermetic default, network-free
301 passed, 23 warnings in 10.09s
Required test coverage of 80% reached. Total coverage: 93.88%
```

`301 @ 93.88%` matches the standing baseline recorded at [053-01-yield](053-01-yield.md) and
[054-01-forward](054-01-forward.md): this pass changed no production line, so nothing about the
implementation verdict moved.

### Scenario coverage as authored

| Scenario | Test | Postures exercised | Peer witness | Instrumented witness |
|---|---|---|---|---|
| TC-003 | `test_tc003_untrusted_peer_refused_by_default` | default only | 0 completed, ≥1 abort, 502 + cert class | 1 connector `ssl=True`, 1 session, all calls to the peer |
| TC-113 | `test_tc113_same_peer_accepted_with_optout` | opt-out only | 3 paths served, 0 aborts | 1 connector `ssl=False` |
| TC-114 | `test_tc114_one_posture_governs_every_outbound_call` | both, paired | opt-out: `/v1/models`+`/metrics`+`/v1/messages` all served; default: none served, ≥1 abort | both runs: 1 connector / 1 session, all three paths dispatched through the one connector, postures differ |
| R5 | `test_r5_trust_identity_pair_is_not_merged` | two client trust postures, no proxy | exactly 1 completed `/v1/models`, trusted/`untrusted` outcomes split | — |
| R1 flag | `test_r1_duplicate_flag_occurrence_honored_once` | opt-out ×2 occurrences | 200 + marker | 1 connector `ssl=False`, clean exit |
| R1 URL | `test_r1_upstream_url_shape_does_not_change_posture` | both, with `/qa-base/../` | default: refused; opt-out: served, 0 aborts | 1 connector per run, `True`/`False` |

## Result

**`READY-PASSING`**

- The persistent gate `tests/qa/test_qap_ms004.py` is authored against the approved `TC-003`/`TC-113`/
  `TC-114` contracts and currently observes the contract: 6/6 pass, stable across three runs and inside the
  combined `tests/qa` run.
- `READY-PASSING` is **not** approval. Yield Phase (`MS-004`) independently inspects contract fidelity and
  re-runs the gate; the authoritative `pass`/`fail` verdicts for `VP-T006-verify-on-by-default`,
  `VP-T006-optout-honored` and `VP-T006-posture-is-process-wide` are its `G-007` evidence, not this
  report's.
- Nothing is BLOCKED: the harness executed every linked scenario (see §Harness Execution 4 — the gate is
  runnable with the remote upstream unreachable), and no `QAP-MS004` scenario needed a capability
  `qa-project` does not declare, so no Capability Gate miss is reportable for this pass.
- For Yield Phase's attention, two facts it should confirm rather than assume:
  1. `tests/qa/tls_probe_entry.py` reaches `ClientSession._request` (private aiohttp surface) to attribute
     each outbound URL to a connector. The peer's request log already proves process-wide posture on its
     own; if the private hook ever disappears, the *primary* assertions must survive without the call-site
     log — the one-connector and peer-observation assertions do not depend on it.
  2. The gate asserts `rc == 0` and no traceback on `SIGTERM` shutdown as its lifecycle witness. That is
     the same contract `TC-110` pins for the dump cluster; if `MS-004`'s verdict wants a different
     lifecycle claim, it belongs to `qa-plan.md`, not to a test edit here.

---

### Correction to §Files Changed, issued by 077-forward-qa (2026-10-06)

The §Files Changed bullet reading

> `Proxy.post(body, timeout, path="/v1/messages")` / `Proxy.url(path=...)` — so a gate can drive the
> compaction-policy endpoint; existing callers pass positionally and are unaffected.

is **inaccurate as written**, and the sentence above it in the same section is the reason it can be
corrected rather than retracted: this gate drives `/v1/models` and `/metrics` by starting the real
proxy and reading `tests/qa/tls_probe_entry.py`'s request log, **not** by POSTing to those paths. No
gate in `tests/qa/` ever passed `path=` to either method (`grep -rnE "\.post\(.*path=|\.url\(.*path="
tests/` → zero call sites; the `proxy.post("/v1/messages", json=…)` hits in `tests/test_upstream_answer_relay.py`
and `tests/test_routes_integration.py` are the aiohttp test client, a different object, and `ms004`'s
`peer.url()` is `TlsPeer.url()`, also different). So the two keywords were unused shared surface on the
one file every later gate imports.

Recorded as `QB:qa-harness/unexercised-path-keyword` (056-yield-phase §Findings P3-1), re-routed to this
lane, and **drained by [077-forward-qa](077-forward-qa.md)**: `Proxy.url()` / `Proxy.post(body, timeout=…)`
are restored to their pre-055 signatures. The original §Files Changed text is left above verbatim rather
than rewritten, per this session's report-accuracy convention (the same disposition as
`QB:report-accuracy/t012-counts`). Nothing this gate asserted changed; `tests/qa/test_qap_ms004.py`
re-ran **6/6** on the restored signatures.

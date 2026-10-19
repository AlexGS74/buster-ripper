# Iteration 095 — Yield Red (MS-001 boundary, final semantic adversary)

Run id: `YR-MS001-r1`. Mode: hypothesize. Target base: `main` @ `72e3a1c6861bf0d8de6317adf52289e5f6cfa4e1`
(unchanged since [054-01-forward](054-01-forward.md); re-verified at [093](093-yield-phase.md) §Base-Sync).
Target object: `62f8d7c` (the tree 093 gated). Review object: `c9b47516e60bfff2b4d6165383ccfa863099cfa4`
(current `HEAD`, Checkpoint 216; `git status --porcelain` empty before this pass wrote).

Dispatch note: the invocation named the mode as "final semantic adversary for the boundary" and supplied no
`Target object` / `Review object` / `Close:` fields. The substantive guard those fields exist for — an ambiguous
snapshot or an unbound milestone — is absent, so this run binds them from measurement rather than returning
`YIELD-RED-BLOCKED: malformed dispatch`: `git rev-parse HEAD` → `c9b4751…`, `git status --porcelain` empty, the
session UID in `session-identity.json` (`9bf7afff-5573-402a-b102-827423a1dded`) matching the handle suffix, and
the milestone `MS-001` resolving to `plan-summary.md:204`. `Close: milestone` is inferred from the dispatch's own words
("the phase can close", `resolved` for the tasks, the Delivered entry) — no session-outcome entry is owed.

## Frozen Purpose Model

Read before any implementation, test, diff, Forward, Yield, Codex or assumption-verdict artifact. Every statement cites
its source; nothing below is inferred from code.

**P1.** When the proxy runs in message-interception/dumping mode (`--dump-dir` on `/v1/messages`), captured messages must
be collected in memory and written to disk in bulk, reducing the number of write operations.
Source: [`US-001`](../decisions.md#us-001) `Normative choice`, first sentence.

**P2.** Durability of buffered in-flight messages is explicitly **not** required: a hard shutdown may kill them; that is
acceptable.
Source: [`US-001`](../decisions.md#us-001) `Normative choice`, second sentence.

**P3.** Ctrl-C (SIGINT) and SIGTERM must flush the buffered dump to disk before the process terminates.
Source: [`US-001`](../decisions.md#us-001) `Normative choice`, third sentence; `G-002` (`plan-summary.md:110`).

**P4.** The documented per-turn layout (`turn_000.json`, `turn_{N:03d}.diff`, `index.txt`) must not be deleted by the
change that introduces the collapsed artifact. Old behavior stays unaffected until the new behavior is implemented,
demonstrated better, and judged worth switching to.
Source: [`US-012`](../decisions.md#us-012) `Normative choice`; ratified in [`US-014`](../decisions.md#us-014).

**P5.** The documented layout is the **default**: with no format configured, N turns produce exactly those three files with
today's content and line format at `2N` write-bearing operations, and the README workflow still works. The collapsed formats
are opt-in and added beside it.
Source: `G-008` `Met when` (a) and its `Boundary` column (`plan-summary.md:119`); `OI:buster_ripper.dump/dump_turn/output_layout`
(`plan-summary.md:330`); `PI:T002/dump_format/default_format_is_documented` (`plan-summary.md:285`);
`US-014`'s round-17 ratification.

**P6.** With a collapsed format selected, N intercepted turns produce **no artifact on disk** before a flush — every write
is deferred. A flush of K ≥ 2 pending turns is exactly **one** write-bearing operation, append-only (bytes grow with `F`,
not `N²`). On the documented path **nothing is buffered**, so its clauses are vacuous there.
Source: `G-001` (`plan-summary.md:107`), `G-002`/`G-004`/`G-005` scope notes (`:108`–`:111`), `OI:…/output_layout`
(`:330`), `PI:T002/dump_format/one_write_path_per_turn` (`:289`), `F-15` (`:220`).

**P7.** A raising dump/flush still yields the upstream response body: dump/flush behavior cannot fail a request.
Source: `G-003` (`plan-summary.md:109`); `EI:buster_ripper.routes.messages/handle_messages/dump_failure_isolated`
(`plan-summary.md:277`); `VP-T001-dump-failure-isolated` (`:221`).

**P8.** Proxy behavior on `/v1/messages` is unchanged: no new endpoints and **no normalization changes**; a request that
base answered is not failed by this session's delta.
Source: `G-003` (`plan-summary.md:109`) `Boundary` column; [`US-012`](../decisions.md#us-012) scope note; `G-003` is
also the ranking authority used by [090-01-yield](090-01-yield.md) §Findings.

**P9.** **Base parity ranks every behavior difference**: where the package and the single-file legacy base
(`buster_ripper_legacy.py`) disagree, base's observable behavior is the authority. This is the session's settled migration
contract, applied by `T013`–`T017` and again by `T018`–`T020`.
Source: `handoff-context.md:701` ("**Base parity as the ranking authority on the route** — the session's settled contract
for the migration"); [089-assumption-decide](089-assumption-decide.md) and [090-01-yield](090-01-yield.md) §Findings;
`T019.002` row (`plan-summary.md:384`).

**P10.** The stats route `/v1/messages/count_tokens` must find the stats the interception route recorded for the same
body, and the session id must be derived from the body **after** the normalization step that removes the per-request billing
nonce — the digest must not include the nonce.
Source: `EI:buster_ripper.session/session_id/derived_after_normalize` (`plan-summary.md:277`, carried `preserve`);
`T019.002` row (`plan-summary.md:384`) and [090-01-yield](090-01-yield.md) §Findings; `count_tokens.py:65-76`
(the compaction nudge that depends on the hit).

**P11.** The two dump artifacts are keyed per session (`sid`) and flushed independently; two system prompts produce two
session directories.
Source: `B-006` (`plan-summary.md:264`), `VP-T001-session-isolation` (`:220`).

**P12.** The dump stays readable without additional tooling: a reader holding only `DUMP_DIR/<sid>/` can locate the changed
turn and identify what changed against the previous turn, per shipped encoding.
Source: `G-006` (`plan-summary.md:118`); `README.md:134-135` + `:146-162`; `FK-007` option 1.

**No `YR-PURPOSE-GAP`.** The authoritative sources rank every clause above and no two of them name materially different
outcomes. The one tension the plan itself records (`G-006` ↔ `G-008`) is arbitrated by `US-012` + `F-13` (the documented
layout stays live, so the reader keeps today's workflow) and by `FK-007` (no tool ships). The 019 closure settled the
default-format reading (S-1) and the round-17 ratification stamped it.

## Stage 2 — Compliant-but-wrong candidates

Written from the approved clauses above, the `VP-*`/`TC-*` contracts (`plan-summary.md:219`–`:232`,
`qa-plan.md:156`–`:232`), the Anchor Map (`plan-summary.md:147`–`:173`) and the public runtime interface. No
implementation body, test body, Forward/Yield report, Codex finding or assumption verdict was read before this section
was persisted.

### YR-H001: the documented layout keeps its filenames while its content drifts (W2)
Purpose basis: P4 (`US-012`), P5 (`G-008`(a), `OI:…/output_layout`), P9 (base parity).
Literal pass condition: `TC-108` — "`turn_000.json`, `turn_{N:03d}.diff`, `index.txt` with today's content and line
format, at `2N` write-bearing operations"; `TC-107` — no `.log`; `VP-T002-documented-format-preserved`
(`plan-summary.md:228`). The forbidden result names only "a drifted `index.txt` line format; a missing or renamed
documented file; a `.log` artifact".
Compliant-but-wrong outcome: the documented branch writes the three files at exactly `2N` writes and every `index.txt`
line matches the README's published shape, but the *content* is computed from a different text than today's: `turn_000.json`
carries the rendered body rather than the normalized JSON body, and each `turn_NNN.diff` is a diff of raw request bodies rather
than `_render_body` output of consecutive turns. Every filename, every write count and every line shape agrees; the bytes a reader
sees differ from today's.
Observable violation: a reader replaying the README workflow (`README.md:146-162`) on the same session compares a documented
run of the package against a documented run of the base and gets different `turn_001.diff`/`turn_000.json` bytes; the
`+added/-removed` counts in `index.txt` also change, while the regex still matches.
Reach: reached — the default run takes this branch on every change-bearing turn.
Applicability basis: "an exact transcript is tested where settlement invariants are the contract" and `G-008` W2 is named as the
shape the plan must not drift into (`plan-summary.md:119`).
Candidate factual premises:
- YR-P001: In `buster_ripper/dump.py`, the documented branch writes `turn_000.json` from the normalized JSON body and
  each `turn_{N:03d}.diff` from `difflib.unified_diff` over `_render_body` output of consecutive turns.
- YR-P002: In `tests/qa/test_qap_ms001.py`, the TC-108 documented-format gate asserts the documented
  files' exact content (bytes/values), not only the `index.txt` line pattern.
Candidate seam: YR-P001 → YR-P002 (writer content → gate observation).
State: awaiting-blind-check

### YR-H002: the documented path buffers records it never writes
Purpose basis: P6 (`G-002`/`G-004`/`G-005` scope notes: "On the documented path nothing is buffered"), `PI:…/one_write_path_per_turn`, P5.
Literal pass condition: `TC-107`/`TC-108` (documented layout at `2N` writes, no `.log`); `TC-100` (collapsed path defers);
`VP-T002-write-count-reduced`'s "either `2N` or `F` — never `2N + F`" is a `MS-002` measurement, not an `MS-001` gate.
Compliant-but-wrong outcome: `dump_turn` writes today's documented files **and** appends the same record to
`SessionState.pending` on the documented path. `flush_all` has no caller at this boundary (022), so nothing is ever written:
the run produces exactly today's three files at `2N` writes, no `.log`, and `TC-107`/`TC-108` pass. The documented session
nonetheless holds N pending records and grows memory linearly; the first flush — a direct call, or `MS-002`'s trigger — writes
a collapsed artifact into a documented session directory and makes the session `2N + F`.
Observable violation: RSS grows with N on a documented run that writes no artifact; `SessionState.pending` is non-empty after
a documented run; a `flush_all()` call on a documented session produces an artifact base never produced.
Reach: reached — every documented turn.
Applicability basis: "success covered while ... stale state" plus `F-21`'s explicit warning that "an unconditional append
plus a flush on the documented path yields `2N + F`".
Candidate factual premises:
- YR-P003: In `buster_ripper/dump.py`, the append to `SessionState.pending` is guarded by the format selection, so a
  documented-format turn appends nothing.
- YR-P004: In `buster_ripper/dump.py`, the documented branch's writes are the only writes that run for a documented
  turn (no second writer runs).
Candidate seam: YR-P003 → YR-P004 (append guard → writer selection).
State: awaiting-blind-check

### YR-H003: the collapsed path writes per-turn files that the gate's observation cannot see
Purpose basis: P6 ("N intercepted turns produce no artifact on disk"), P8 (`G-003`), P11.
Literal pass condition: `TC-100` — "`DUMP_DIR/<sid>/` holds no artifact containing those turns"; `VP-T001-writes-deferred`
— "no artifact containing those turns exists on disk" (`plan-summary.md:219`). The VP's forbidden result is "a documented
file appearing on the collapsed path", not "any file".
Compliant-but-wrong outcome: with a collapsed format selected, `dump_turn` writes a per-turn record to
`DUMP_DIR/<sid>/pending/turn_000.json` (a subdirectory of the session dir) — or `TC-100`'s assertion is implemented as
"no `.log` file exists" — so the gate's non-recursive/name-scoped listing is empty and `TC-100` passes while N turns have
already produced N per-turn artifacts on disk before any flush.
Observable violation: `find DUMP_DIR -type f` before a flush lists N per-turn files; `du -s` grows with N; a later flush's
`2N + F` accounting is already broken.
Reach: reached — every collapsed turn, if the writer takes this branch.
Applicability basis: "an absence assertion passes against an empty or disconnected observation channel".
Candidate factual premises:
- YR-P005: In `tests/qa/test_qap_ms001.py`, the TC-100 collapsed-format gate fails if a `turn_NNN.json` or
  `index.txt` appears anywhere under the session directory before the flush — not only if a `.log` file appears.
- YR-P006: In `buster_ripper/dump.py`, the collapsed branch's per-turn work writes no file at all (it only
  appends to memory), so no per-turn path exists that could write outside the session dir.
Candidate seam: YR-P005 → YR-P006.
State: awaiting-blind-check

### YR-H004: the swallow is narrower than "a raising dump/flush"
Purpose basis: P7 (`G-003`, `EI:…/dump_failure_isolated`), P8.
Literal pass condition: `TC-001` — "a raising dump/flush still yields the upstream response body"; its two arms are a
`PermissionError` injected in the documented write path and a raise surfacing at the shutdown flush.
Compliant-but-wrong outcome: `handle_messages` wraps `dump_turn` in `except OSError` (or `except (OSError, PermissionError)`)
rather than `except Exception`. `TC-001`'s injected `PermissionError` is caught, so both arms pass; a `TypeError`/
`AttributeError`/`UnicodeEncodeError` raised while building a record (a body shape today's writer did not see, or a body
that reached the writer because `T018` removed the 1 MiB cap) escapes the handler and the request fails with a proxy 500
where base returned the upstream body.
Observable violation: POST a body whose record building raises a non-`OSError` exception → proxy 500; base/healthy control
returns the upstream status; the exception is not logged.
Reach: reached — any body that reaches `dump_turn` and raises a non-`OSError`.
Applicability basis: "success covered while failure ... leaves stale state" and the milestone's own "a raising dump/flush
still cannot fail a request" (unqualified).
Candidate factual premises:
- YR-P007: In `buster_ripper/routes/messages.py`, the `dump_turn(...)` call is wrapped in a `try` whose handler
  catches `Exception` (not a narrower class), and no dump/flush call on the request path sits outside that `try`.
- YR-P008: In `buster_ripper/dump.py`, the append-time volume check calls the flush helper from inside `dump_turn`.
Candidate seam: YR-P007 → YR-P008 (request-path swallow → threshold flush inside the dumped call).
State: awaiting-blind-check

### YR-H005: `T018`'s deleted body cap is a base-parity departure the milestone contract does not observe
Purpose basis: P9 (base parity ranks every behavior difference), P8 (`G-003`), P5.
Literal pass condition: `MS-001`'s `Complete when` and its four `VP-*` rows (`plan-summary.md:204`, `:219`–`:228`)
name no body size, no `413`, and no cap; the only witnesses are the hermetic
`tests/test_transport_parity.py::TestNoRequestBodyCap` (5 tests) and `QAP-MS001` passes without any large body.
Compliant-but-wrong outcome: the base application enforced a request-body limit (explicitly or through aiohttp's
1 MiB default) while the package now sets `client_max_size=0`, so a body base rejected with 413 now reaches the handler
and the writer. Every `MS-001` gate passes because none sends a body near the boundary, and the milestone contract is
silent — the difference is ranked only by base parity, which the milestone's own gate does not measure.
Observable violation: POST a body above base's accepted maximum at `/v1/messages` → package 200/upstream round trip where
base answered 413; the documented path writes a `turn_000.json` base never wrote.
Reach: reached — any request above base's limit; the deleted cap's effect is exactly on the request path.
Applicability basis: "a fresh-install path passes while persisted/upgrade state fails" is not it; this is the base-parity family:
the contract is silent while the purpose (P9) is not.
Candidate factual premises:
- YR-P009: In `buster_ripper_legacy.py`, the aiohttp application is created/run without setting `client_max_size`
  (or with a limit other than 0), so base's effective body limit is aiohttp's default (1 MiB) or an explicit finite value.
- YR-P010: In `buster_ripper/server.py`, the application sets `client_max_size=0`.
Candidate seam: YR-P009 → YR-P010 (base app construction → package app construction).
State: awaiting-blind-check

### YR-H006: the narrowed stats key is not the normalization the invariant names
Purpose basis: P10 (`EI:…/derived_after_normalize`), P9 (base parity), P8 (`G-003`: "no normalization changes").
Literal pass condition: `T019.002`'s Done-when (35 witnesses in `tests/test_count_tokens_parity.py`), the digest-equality
class, and the four already-tolerated shapes; `090-01-yield` §Findings.
Compliant-but-wrong outcome: `count_tokens.py:87` keys the digest on `strip_billing_nonce(body)` while
`_normalize_system` also performs a second mutation of the `system` text (e.g. stripping `cache_control` markers, or
normalizing system text); for a body carrying that second mutation the digest differs from base's `_session_id` and the stats
miss — the compaction nudge fires on the wrong turn, silently. The 18-case × 2-axis digest-equality measurement does not
contain such a body, and `TC-*`/`VP-*` at `MS-001` do not observe `count_tokens` at all.
Observable violation: POST a nonce-bearing body whose `system` also carries a cache marker to `/v1/messages` (records stats
under base's digest) then the same body to `/v1/messages/count_tokens` → stats not found / nudge inputs differ from base.
Reach: reached — the route is exposed and the documented client sends a nonce per request.
Applicability basis: "expected fields present while unknown/duplicate/forbidden input is silently accepted" and
"individually correct tasks conflict through a shared key".
Candidate factual premises:
- YR-P011: In `buster_ripper/normalize.py`, `_normalize_system`'s only change to a body's `system` text is
  removing the billing nonce — no other step or branch in that function rewrites `system`.
- YR-P012: In `buster_ripper/routes/count_tokens.py`, the digest is computed after `strip_billing_nonce(body)` and
  before `estimate_tokens`.
- YR-P013: In `buster_ripper/routes/count_tokens.py`, `estimate_tokens` is called on the body after the nonce strip,
  not on the body as it arrived.
Candidate seam: YR-P011 → YR-P012 → YR-P013.
State: awaiting-blind-check

### YR-H007: the `T020` EMA witnesses assert the implementation, not the behavior
Purpose basis: P12 (`G-006`: the dump stays readable / the compaction nudge depends on the KV signal), and the session's
own evidence contract for `CR3-F-05`.
Literal pass condition: `T020` — "Drive `poll_kv_cache` in the two EMA witnesses; cover `compaction.py:49`"; its
done-when is "the two witnesses assert the blended value after the second tick and `:49` is covered".
Compliant-but-wrong outcome: the witnesses drive `poll_kv_cache` twice as *two separate invocations* (each with its own
function-local `first`), so the blend arm at `compaction.py:48-52` is never entered, and the assertion reads a value the
double itself produced (e.g. `assert session.get.called` or a value re-derived from the same fixture) — the test is green while
the EMA blend is untested and the production loop could be wrong.
Observable violation: mutate the blend arm (`first`/`second` handling at `compaction.py:45-52`) and the witnesses stay green;
the KV signal the compaction nudge consumes is wrong while the suite is green.
Reach: latent — only a mutated implementation reveals it; the current implementation may be right.
Applicability basis: "success covered while failure ... leaves stale state" and the methodology's "witness must be able to fail".
Candidate factual premises:
- YR-P014: In `tests/test_compaction.py`, the EMA witnesses drive two iterations of a *single* `poll_kv_cache`
  invocation (one function-local `first`) and assert the blended value after the second tick.
- YR-P015: In `tests/test_compaction.py`, the EMA witnesses assert a value that the test itself computes from the
  same inputs rather than a value read from the double's fixed responses.
Candidate seam: YR-P014 → YR-P015.
State: awaiting-blind-check

### YR-H008: a non-decodable upstream error body fails `/v1/messages` where base relayed it
Purpose basis: P8 (`G-003`: proxy behavior unchanged), P9 (base parity), and `S18`'s `SAFE` is explicitly
flagged for re-weighing by the handoff before it is cited as current.
Literal pass condition: `T019.001` — "`handle_streaming`'s context-overflow probe reads the upstream error text through the
response's own text surface (`messages.py:106`)"; its witnesses pin the upstream's *text*, and the QA gate's upstream
answers UTF-8 JSON. `G-003`'s boundary says "no normalization changes" — but this is the error-body relay, not
normalization.
Compliant-but-wrong outcome: `result.text` decodes the upstream error body strictly under the declared charset (or UTF-8
default), so a 400 context-overflow answer carrying non-UTF-8 bytes raises `UnicodeDecodeError` inside the probe; the
enclosing arm catches only `aiohttp.ClientError`, so the exception escapes and the proxy answers 500 where base relayed
the upstream's error. Every milestone gate is green because its upstream answers UTF-8 JSON.
Observable violation: the upstream returns 400 with a body in `iso-8859-1`/invalid bytes → `/v1/messages` answers 500
(traceback) instead of the upstream's error status/body.
Reach: reached — the overflow probe runs on the streaming path; the upstream controls the body and the status.
Applicability basis: "wired/reachable but unable to carry meaningful production data" is the mirror; here it is
"success covered while failure ... leaves stale state" — the failure path was changed and its coverage was not.
Candidate factual premises:
- YR-P016: In `buster_ripper/routes/messages.py`, the context-overflow probe reads the upstream error text via
  `.text` (the `web.Response`/`ClientResponse` text surface) inside a `try` whose handlers do not include
  `UnicodeDecodeError` or `LookupError`.
- YR-P017: In the installed aiohttp, `ClientResponse.text()` (or `web.Response.text`) decodes strictly and raises
  `UnicodeDecodeError` for bytes that are invalid under the resolved charset.
Candidate seam: YR-P016 → YR-P017 (probe → decode surface → enclosing arm).
State: awaiting-blind-check

### YR-H009: the documented default is `documented`, but the no-flag path is not the branch `TC-107` runs
Purpose basis: P5 (`G-008`(a), `PI default_format_is_documented`), P9.
Literal pass condition: `TC-107` — "no `--dump-format` passed, no config override → documented layout at `2N` writes, no `.log`";
`TC-108` — `--dump-format documented` → the same.
Compliant-but-wrong outcome: the click option's `default` is `"documented"`, but the option's `callback` assigns
`config.DUMP_FORMAT = "documented"` only when a value was passed; when the flag is absent the callback never runs and
`config.DUMP_FORMAT` keeps a *different* value from an import-time environment read (e.g. `os.environ.get("DUMP_FORMAT")`),
so a clean shell with `DUMP_FORMAT` set produces a collapsed artifact while `TC-107`'s autouse fixture clears `DUMP*`
and passes.
Observable violation: `DUMP_FORMAT=change-focused buster-ripper --dump-dir X` (no `--dump-format`) → `change-focused.log`,
where today/base produce the documented files; `--help` still shows `documented` as the default.
Reach: reached — environment-supplied configuration.
Applicability basis: "a fresh-install path passes while persisted/upgrade state fails".
Candidate factual premises:
- YR-P018: In `buster_ripper/main.py`, the `--dump-format` option's `default` is `"documented"` and its `callback`
  assigns the value to `config.DUMP_FORMAT`.
- YR-P019: In `buster_ripper/config.py`, `DUMP_FORMAT` is not read from any environment variable (it is a plain
  module global).
Candidate seam: YR-P018 → YR-P019.
State: awaiting-blind-check

## Blind premise gate

Every premise below is atomic, names its subject, and is answerable by reading only. No premise carries the
hypothesis, the purpose or the plan. Bundles are by subject; the orchestrator dispatches one checker per bundle
(waves of at most four) and a composer for each seam.

| Bundle | Subject | Premises |
|---|---|---|
| A | `buster_ripper/dump.py` | YR-P001, YR-P003, YR-P004, YR-P008 |
| B | `tests/qa/test_qap_ms001.py` | YR-P002, YR-P005 |
| C | `buster_ripper/routes/messages.py` | YR-P007, YR-P016 |
| D | `buster_ripper/normalize.py` + `routes/count_tokens.py` | YR-P011, YR-P012, YR-P013 |
| E | `buster_ripper_legacy.py` | YR-P009 |
| F | `buster_ripper/server.py` + `buster_ripper/main.py` + `config.py` | YR-P010, YR-P018, YR-P019 |
| G | `tests/test_compaction.py` | YR-P014, YR-P015 |
| H | `aiohttp` installed distribution | YR-P017 |

Seams: `S-YR-1` = YR-P011 → YR-P012 → YR-P013 (normalization step → digest key → estimate source);
`S-YR-2` = YR-P007 → YR-P008 (request-path swallow → threshold flush);
`S-YR-3` = YR-P009 → YR-P010 (base app limit → package app limit);
`S-YR-4` = YR-P001 → YR-P002 (documented writer content → gate observation);
`S-YR-5` = YR-P018 → YR-P019 (CLI default → config default).

## Blind gate status — NOT RUN in this dispatch

The gate is **not satisfied**, and no premise below has a verdict. This dispatch runs at subagent depth 1, and
nested subagent dispatch is disabled (`Subagent depth limit reached (1)`), so no independent context exists here to
answer a claim. Certifying any of these facts in this report's own context would violate the gate the skill makes
mandatory, so **no premise is self-verified and no `CLEAN`/`FINDINGS` verdict is issued by this pass**.

The orchestrator's next acts, in order:

1. `forward-collect` in `red-premises` mode against this report's §Stage 2 `Candidate factual premises` rows —
   extraction only, no checking.
2. `assumption-check`, one agent per bundle in the table above, in **two waves of at most four**: A–D, then E–H.
   Prompts carry the claim text and the repository root only. No purpose, plan, milestone, hypothesis or verdict.
3. `assumption-compose` for `S-YR-1` … `S-YR-5` once both members are `HOLDS`. `S-YR-3` composes a
   third-party fact (aiohttp's default) with a repo fact, so it is the one seam whose counterexample could be a library
   default rather than a code path.
4. Re-dispatch Yield Red in `adjudicate` mode with `Yield Red Run: YR-MS001-r1`, `Target object: 62f8d7c`,
   `Review object: c9b4751…`, `Close: milestone`. That pass reads the blind artifacts **before** the implementation,
   then the diff and the one dependency hop.

If the orchestrator instead intends this pass to adjudicate directly, it must re-dispatch with nested subagent dispatch
enabled; the premise list above is already shaped for that fan-out and needs no rewriting.

YIELD-RED-PREMISES: YR-MS001-r1 | 9 hypotheses | 19 premises | 5 seams | iterations/095-yield-red.md

## Adjudication (dispatch 2, run `YR-MS001-r1`)

Blind artifacts read first, in this order: `assumptions/verdicts/yr-p001.md` … `yr-p019.md` (18 files; `yr-p006.md` is absent) and
`assumptions/compositions/S-YR-1.md` … `S-YR-5.md`. Implementation read after: `buster_ripper/dump.py`,
`routes/messages.py`, `normalize.py`, `routes/count_tokens.py`, `session.py`, `server.py`, `main.py`, `config.py`,
`compaction.py`, `tests/qa/test_qap_ms001.py`, `tests/test_compaction.py`, plus `buster_ripper_legacy.py` for the
base-parity question. Review object re-verified: `HEAD` = `c9b47516e60bfff2b4d6165383ccfa863099cfa4`, and
`git diff --name-status 62f8d7c..c9b4751` is **session-artifact only** (12 files, zero production, test, config, plan,
contract or registry bytes), so the production tree under review is still `62f8d7c`.

### Premise disposition

| Premise | Verdict | Disposition |
|---|---|---|
| YR-P001 | HOLDS | `YR-H001` product arm killed — `dump.py:102-107,161-189,262-270` writes `turn_000.json` from the normalized body and each `.diff` from `_unified_diff` over `_render_body` of consecutive turns |
| YR-P002 | **UNCLEAR** | **blocks** — `YR-H001`'s QA arm cannot be adjudicated; "exact content (bytes/values)" has a breadth and a strict reading. Premise Revision below |
| YR-P003 | HOLDS | `YR-H002` killed — `dump_turn`'s `documented` arm (`dump.py:102-103`) never reaches `_append_record`; `state.pending` appended only at `:203` from `:105`/`:107` |
| YR-P004 | HOLDS | `YR-H002` killed — `_flush_state` early-returns for `state.fmt == "documented"` (`dump.py:297-298`) and `_write_documented` is the only writer on that branch |
| YR-P005 | HOLDS | `YR-H003` QA arm killed — `test_qap_ms001.py:185-192` asserts `listing(sid_dir) == []` before a flush, and `:205-210` rejects any non-`.log` file at any depth after the run, so the observation channel is not `.log`-scoped |
| YR-P006 | **no verdict** | **incomplete blind coverage** — Bundle B in the Stage-2 table listed `YR-P002, YR-P005` and omitted this row; no verdict file exists. `YR-H003`'s product arm cannot be certified. Premise Revision below |
| YR-P007 | HOLDS | `YR-H004` killed — `routes/messages.py:56-60` is `try: dump_turn(body)` / `except Exception:` and no other dump/flush call sits on the request path |
| YR-P008 | HOLDS | `YR-H004` killed — `_append_record` (`dump.py:200-206`) calls `flush_all()` inside `dump_turn`'s call tree |
| YR-P009 | **DOES-NOT-HOLD** | `YR-H005` killed as written — `buster_ripper_legacy.py` builds a FastAPI app (`:237`) and serves it with `uvicorn.run` (`:1009`), imports no aiohttp, and contains no `client_max_size`; the aiohttp 1 MiB default does not exist in base |
| YR-P010 | HOLDS | `server.py:125` passes `client_max_size=0` at the only production construction site |
| YR-P011 | **DOES-NOT-HOLD** | `YR-H006` killed — `_strip_billing_from_blocks` (`normalize.py:36`) unconditionally `.strip()`s every text block and drops blocks emptied by the trim, so `_normalize_system` is not nonce-only |
| YR-P012 | HOLDS | `YR-H006` dead on the digest ordering — `count_tokens.py:87` strip, `:88` digest, `:94` estimate |
| YR-P013 | HOLDS | `YR-H006` dead on the estimate source — `estimate_tokens(body)` receives the already-mutated dict |
| YR-P014 | **DOES-NOT-HOLD** | `YR-H007` killed — the witnesses drive **one** and **three** reading ticks of one `poll_kv_cache` task, not two |
| YR-P015 | **DOES-NOT-HOLD** | `YR-H007` killed — the expectations are hardcoded literals (`[0.85]`, `[0.75, 0.525, 0.4365]`) guarded by `_KV_EMA_ALPHA == 0.3`, not arithmetic computed at test time |
| YR-P016 | **DOES-NOT-HOLD** | `YR-H008` killed in the stronger direction — `handle_streaming` has **no** `try` around `error_text = result.text` (`routes/messages.py:104-118`); the file's only `await resp.text()` is `_error_body` (`:195-198`), already wrapped |
| YR-P017 | HOLDS | `ClientResponse.text()`/`web.Response.text` decode strictly — but the probe reads a `web.Response` whose `StringPayload` was built from `_error_body`'s `str`, so it cannot raise |
| YR-P018 | HOLDS | `YR-H009` killed — `main.py:80-85` declares `default="documented"` with `callback=_set_dump_format`, whose body is `config.DUMP_FORMAT = value` (`main.py:12-15`) |
| YR-P019 | HOLDS | `YR-H009` killed — `config.py:28` is a plain module global, the module imports no `os`, and no production module reads an env var for it |

### Hypothesis disposition

| Hypothesis | Verdict | Basis |
|---|---|---|
| YR-H001 | **killed on product; QA arm blocked** | `YR-P001` HOLDS: the documented writer emits the normalized JSON body and rendered diffs, so "content computed from a different text" is not realized. The QA arm (would `TC-108` catch a drift?) needs `YR-P002`, which is UNCLEAR. |
| YR-H002 | **killed** | `YR-P003` + `YR-P004` HOLD: the documented branch appends nothing and runs no second writer; `flush_all`/`_flush_state` write nothing for a documented session (`dump.py:119-122,285-298`). No `2N + F`. |
| YR-H003 | **killed** | `YR-P005` HOLDS: a per-turn file under the session directory — including a nested `pending/` — fails the post-run `listing(sid_dir)`/`rglob_files` assertions, and the pre-flush `snapshot == []` is a universal absence over the direct entries. The product arm additionally needs `YR-P006`; `dump.py:104-107` shows the collapsed arms call `_append_record` only. |
| YR-H004 | **killed** | `YR-P007` HOLDS (`except Exception` at `routes/messages.py:59`, no other dump call on the request path) and `YR-P008` HOLDS (threshold `flush_all()` inside `dump_turn`). `S-YR-2` composes SAFE on the code: `_flush_session` absorbs `Exception` (`dump.py:285-292`) one frame below the handler's own arm. |
| YR-H005 | **killed as written; residual evidence gap** | `YR-P009` DOES-NOT-HOLD — the hypothesis's only stated mechanism (aiohttp's 1 MiB default) is absent from base. The surviving question is whether base's FastAPI/uvicorn stack imposed a default limit the gate cannot see: **not answerable in this repository** (Starlette/uvicorn are not installed and no version is pinned). Recorded as an accepted risk, not a demonstrated defect; `YR-P020` adds the repo-checkable half. `T018` is parity-correct at its own literal clause ("no request-body cap"). |
| YR-H006 | **killed** | `YR-P011` DOES-NOT-HOLD, but the extra mutation is shared by base and by the full `normalize()`: `strip_billing_nonce` **is** `_normalize_system`, and no later step writes the text `session_id` hashes (`_normalize_cache_control` deletes `cache_control` only, `_normalize_date` writes `content` only — `S18` N10). The narrowed digest therefore equals the full one; no base divergence is introduced by `T019.002`. |
| YR-H007 | **killed** | `YR-P014` and `YR-P015` both DOES-NOT-HOLD. The three-tick witness enters the blend arm at ticks 2 and 3 and asserts literals for α = 0.3, so `compaction.py:49` is covered and the witness can fail (mutation replay red at `083-01-yield`). The one-tick witness alone never enters the blend arm — not a gap, because the second witness does. |
| YR-H008 | **killed** | `YR-P016` is falsified in the stronger direction (no `try` at all), but the probe's `.text` is `web.Response.text` over a `StringPayload` built from `_error_body`'s `str`; `_error_body` (`routes/messages.py:195-198`) catches `(UnicodeDecodeError, LookupError)` and falls back to `read().decode(errors="replace")`. `YR-P017` HOLDS but is never reached on that operand. A non-decodable upstream error body is relayed with replacement characters, not failed. |
| YR-H009 | **killed** | `YR-P018` and `YR-P019` HOLD; `S-YR-5` composes SAFE. `DUMP_FORMAT` has no env channel and the callback assigns unconditionally on every parse, including the absent-flag path. |

### Compositions

- `S-YR-1` MALFORMED — no hidden contradiction. The guard fired on a falsified member (`YR-P011`), but the extra mutation is common to base, to the full `normalize()` and to the narrowed call, so the seam's digest operand is not crossed by the delta. Record defect; its only hypothesis (`YR-H006`) is killed, so no recomposition is owed.
- `S-YR-2` SAFE — confirmed on the code; `YR-H004` killed.
- `S-YR-3` MALFORMED — no hidden contradiction. The base member named the wrong framework (`YR-P009`); the real residual is an **unresolved evidence** question, not a contradiction: whether base's framework imposed a default body limit. Not answerable in-repo. Corrected seam `S-YR-3′` = `YR-P020` → `YR-P010`, scoped to the app's **configured** limit (base configures none, package configures `0`); composes SAFE once `YR-P020` is checked. The framework default stays outside that boundary and is recorded as a residual, below.
- `S-YR-4` MALFORMED — no hidden contradiction. The fork is in the wording of `YR-P002`, not in the writer/gate pair. Under the breadth reading it composes SAFE; under the strict reading its counterexample is a diff-source drift the gate does not constrain, which no producer realizes today (`YR-P001` HOLDS). Recompose only after the split.
- `S-YR-5` SAFE — `YR-H009` killed.

### Failure-condition table (independent of Forward's account)

| # | Applicable condition from the frozen purpose | Resolution |
|---|---|---|
| F1 | documented writer wired but unable to carry meaningful content | safe — `YR-P001` HOLDS; `TC-108` asserts body parse, marker containment, diff headers, `index.txt` line format/count; `TC-107` asserts the file set |
| F2 | unknown/forbidden format value silently accepted | safe — R1 hostile tier refuses every unknown value and writes nothing; `S-YR-5` SAFE |
| F3 | raising dump/flush fails a request; stale state after a failed write | safe — `YR-P007`/`YR-P008` HOLD, `S-YR-2` SAFE, `TC-001` both arms; `state.turn`/`prev_text` advance only after a successful write (`dump.py:108-109`) |
| F4 | fresh-install passes while persisted/upgrade state fails | safe — `S-YR-5` SAFE; a mid-process format change is unreachable (single parse-time assignment), and the `state.fmt == "documented"` early return only strands records under that unreachable change |
| F5 | two writers conflict through shared state/key | safe — `YR-P003`/`YR-P004` HOLD; documented branch never appends, `_flush_state` refuses documented, one `(sid, format)` artifact |
| F6 | exact transcript tested where settlement is the contract | **blocked** — `YR-P002` UNCLEAR; Premise Revision |
| F7 | absence assertion passes against an empty observation channel | safe — `YR-P005` HOLDS; `listing()` returns `[]` for a missing dir but the post-run `listing(str(dump)) == [session_id]` plus `assert pre_flush_snapshots` closes it |
| F8 | a deletion removes the guard/ordering that made a prior path safe (`T018` deleted the body cap) | **residual evidence gap** — `YR-P009` kills the stated mechanism; base's own code sets no limit (`YR-P020`), the framework default is unanswerable in-repo. Recorded as accepted risk, not a finding |
| F9 | a non-decodable upstream error body fails the request | safe for the decode classes the seam names — `_error_body` catches and replaces; `S18` SAFE. The surrogate re-encode residual (`S18` N9, `claim-417` Caveat) predates the delta and is recorded, not raised |

### Withheld verdict

`resolved` is withheld for `T015`–`T020`, `T019.001`, `T019.002` and `MS-001`. **No task's acceptance is reopened**: no finding is demonstrated against any accepted record, and none of
[`083-01-yield`](083-01-yield.md), [`085-01-yield`](085-01-yield.md), [`090-01-yield`](090-01-yield.md),
[`093-yield-phase`](093-yield-phase.md), `S16`, `S18`, `S-YR-2` or `S-YR-5` is contradicted. The blocker is blind-premise
coverage, not task correctness. No `Delivered` entry, no phase acceptance and no parent completion are published: `Close: milestone`,
and `session-identity.json` carries no `parent_session_uid`, so there is no parent to publish to.

### Premise Revision v2 — `YR-MS001-r2`

Predecessor run: `YR-MS001-r1` ([095-yield-red.md](095-yield-red.md)).

Affected: `YR-P002` (UNCLEAR), `YR-P006` (no verdict — omitted from the Stage-2 bundle table), `YR-P009`
(DOES-NOT-HOLD — the `S-YR-3` base member named the wrong framework), plus one new row `YR-P020`. All other premises keep
their verdicts and are **not** re-checked: `YR-P001`, `YR-P003`–`YR-P005`, `YR-P007`, `YR-P008`, `YR-P010`, `YR-P012`–`YR-P015`,
`YR-P017`–`YR-P019`.

Why each wording failed:

- `YR-P002` (`assumptions/verdicts/yr-p002.md`, VERDICT: UNCLEAR) — "asserts the documented files' exact content
  (bytes/values)". The gate does read and assert on `turn_000.json` and every `.diff` (JSON parse, marker containment,
  `---`/`+++`/`@@`, exact `index.txt` line count and pattern, exact `2N` op decomposition), but never against exact bytes or a
  complete value set. The claim is true under the breadth reading and false under the strict one; a blind reader cannot tell which
  was meant.
- `YR-P006` (no verdict file) — the Stage-2 bundle table listed Bundle B as `YR-P002, YR-P005` and never dispatched this row.
  It is required for `YR-H003`'s product arm: does the collapsed branch write any file at all?
- `YR-P009` (`assumptions/verdicts/yr-p009.md`, VERDICT: DOES-NOT-HOLD) — "the aiohttp application … without setting
  `client_max_size`". Base builds a **FastAPI** app and serves it with `uvicorn.run` (`buster_ripper_legacy.py:237,1009`), imports
  no aiohttp (`:84-87`), and never names `client_max_size`. The seam's base member described a framework base does not use.
- New, not a re-wording: the residual the falsification exposes — what base *did* impose. The base's own code is checkable; the
  framework's default is not (Starlette/uvicorn are absent from the installed tree and no version is pinned).

Corrected candidate premise rows:

- `YR-P002a`: In `tests/qa/test_qap_ms001.py`, `assert_documented_contents` compares the content of each documented file against
  an expected value that is not produced by the code under test (byte-for-byte equality, or a complete expected value set).
- `YR-P002b`: In `tests/qa/test_qap_ms001.py`, `TC-108`'s `turn_{N:03d}.diff` assertions (`---`, `+++`, `@@`, a `+` line
  containing turn N's marker, a `-` line containing turn N−1's marker) are satisfied by any unified diff whose `+` and `-` lines carry
  those markers, so they do not constrain the text the diff is computed from.
- `YR-P006`: In `buster_ripper/dump.py`, the collapsed branch's per-turn work writes no file and appends nothing to disk —
  `dump_turn`'s collapsed arms call `_append_record` only, which writes nothing; the only filesystem effect before a flush is the
  session-directory `mkdir` in `_lookup_or_create_state` (`:154`).
- `YR-P020`: In `buster_ripper_legacy.py`, the FastAPI app is created and served with no request-body size limit configured, and
  the body is read with `await request.json()` with no size check.

Corrected seam: `S-YR-3′` = `YR-P020` → `YR-P010`, boundary "base's **configured** request-body limit → package's
configured request-body limit" (base configures none, package configures `0`). `S-YR-1` needs no recomposition (its hypothesis is
killed); `S-YR-4` becomes composable once `YR-P002a`/`YR-P002b` resolve; `S-YR-5` stays SAFE.

Successor run: **`YR-MS001-r2`**. Bundle table for the delta only:

| Bundle | Subject | Premises |
|---|---|---|
| B | `tests/qa/test_qap_ms001.py` | `YR-P002a`, `YR-P002b` |
| A | `buster_ripper/dump.py` | `YR-P006` |
| E | `buster_ripper_legacy.py` | `YR-P020` |

```text
YIELD-RED-PREMISES: YR-MS001-r2 | supersedes YR-MS001-r1 | iterations/095-yield-red.md
```

### Open

- Orchestrator: dispatch the three delta bundles (three checkers, one wave), then `assumption-compose` for `S-YR-4`
  (if both members hold) and `S-YR-3′` (if `YR-P020` holds), then re-dispatch `adjudicate` as `YR-MS001-r2` with
  `Target object: 62f8d7c`, `Review object: {current HEAD}`, `Close: milestone`.
- User (recorded, `Ask: no`): the Starlette/uvicorn default body limit is not answerable from this repository (base's
  framework is neither installed nor version-pinned). Accepted risk, citing [`079-codex-review`](079-codex-review.md) §CR3-F-01
  and `server.py:119-124`. No answer would change the code: the milestone's contract names no cap and `T018`'s own clause is
  "no request-body cap".
- Noted, not pursued (0.01%): `S18` N9's surrogate re-encode (`UnicodeEncodeError` out of `StringPayload.__init__` for a
  lone surrogate produced by a `utf-16`-declared upstream error body) is pre-existing relative to `T019.001`'s fix, recorded by
  `claim-417`'s Caveat and `S18` §Notes, and needs a rare coincidence.

## Yield-Red Summary

`YR-MS001-r1` **withholds** the boundary verdict. All nine hypotheses are killed or blocked by premise state, not by
implementation defects: `YR-H001` product arm killed by `YR-P001`, `YR-H002`/`YR-H003`/`YR-H004`/`YR-H006`/`YR-H007`/
`YR-H008`/`YR-H009` killed, `YR-H003` product arm and `YR-H001` QA arm held back only by the missing/ambiguous premises
`YR-P006`/`YR-P002`. Three seams are malformed on member state, not on hidden contradictions; `S-YR-2`/`S-YR-5` are SAFE.
No product, QA, plan or purpose finding is issued. `resolved`, phase acceptance, the `Delivered` entry and parent completion stay
unpublished.

YIELD-RED-PREMISES: YR-MS001-r2 | supersedes YR-MS001-r1 | iterations/095-yield-red.md

## Adjudication (dispatch 3, run `YR-MS001-r2`)

Review basis re-verified before reading: `HEAD` = `c9b47516e60bfff2b4d6165383ccfa863099cfa4`, and
`git diff --name-status 62f8d7c..c9b4751` is still session-artifact only (12 files, zero production, test, config, plan or
registry bytes), so the tree adjudicated is `62f8d7c`. Blind artifacts read first, in this order: `yr-p002a.md`, `yr-p002b.md`,
`yr-p006.md`, `yr-p020.md`. Implementation read after: `dump.py`, `routes/messages.py`, `tests/qa/test_qap_ms001.py`,
`tests/qa/harness.py`, `tests/test_dump.py`, `tests/test_routes_integration.py`, `buster_ripper_legacy.py`, `README.md`.

### Premise state

| Premise | Verdict | Consequence |
|---|---|---|
| `YR-P001` | HOLDS *(r1)* | documented writer writes the normalized JSON body and `_render_body` diffs — confirmed at `dump.py:161-189` |
| `YR-P002a` | **DOES-NOT-HOLD** *(fresh)* | `assert_documented_contents` (`test_qap_ms001.py:104-146`) asserts partial properties only: JSON parse + `model` + marker substring for `turn_000.json`, line count + README regex + literal columns for `index.txt`, five presence checks for the diff. No byte comparison, no complete value set |
| `YR-P002b` | **HOLDS** *(fresh)* | the diff assertions are exactly the five presence checks; `startswith("+")`/`startswith("-")` also match `+++`/`---`, so any unified diff carrying the markers passes and the diff source is unconstrained |
| `YR-P006` | **HOLDS** *(fresh)* | the collapsed arms call `_append_record` only (`dump.py:104-107,200-206`); the only pre-flush filesystem effect is `session_dir.mkdir` (`:154`); `YR-H003` product arm killed |
| `YR-P020` | **HOLDS** *(fresh)* | base builds `FastAPI(lifespan=_lifespan)` (`buster_ripper_legacy.py:237`), serves it with `uvicorn.run` (`:1009`), registers no middleware, and reads `await request.json()` (`:641`) with no size check |
| all others | unchanged from r1 | `P003`–`P005`, `P007`, `P008`, `P010`, `P012`, `P013`, `P017`–`P019` HOLD; `P011`, `P014`–`P016` DO NOT HOLD |

### Seam recomposition

No `S-YR-4′` or `S-YR-3′` composition artifact exists in `assumptions/compositions/`. The orchestrator's
dispatch assumed one had been written; it had not, and a composer dispatch from this context is refused at
`Subagent depth limit reached (1)`. Recorded honestly as a **record deviation, not a coverage exemption**:

- `S-YR-4′` (`P001` → `P002b`): both members HOLDS. Composed on the verdict texts, with no new source read:
  the writer's output set is a strict subset of the gate's accepted set, so the only cross-boundary counterexample
  requires a producer inconsistent with `P001`. No unverified fact enters the finding — the QA finding rests on
  `P002a`/`P002b` alone, and the product exclusion rests on `P001` alone.
- `S-YR-3′` (`P020` → `P010`): both HOLDS; base configures no limit and the package configures `0`, so no
  *configured*-limit divergence exists. Its only hypothesis (`YR-H005`) is killed, so no composition is owed.

### The semantic question

**The product arm does not make the QA gap irrelevant.** `P001` excludes a *product* defect; it says nothing about
whether the gate would detect one. The skill's own rule is exact: "Evidence of safe implementation can exclude the product
defect, but does not establish that QA detects it; assess those dispositions separately."

Concretely, and more precisely than the dispatch's example: a writer that emits the *rendered* body in `turn_000.json` would
**not** pass `TC-108` — `json.loads` at `:109` rejects it. The reachable compliant-but-wrong outcome is one step narrower:

- `turn_000.json` carries `json.dumps(pre_normalization_body)` — the body *before* `normalize()` strips the billing nonce.
  It parses as JSON, is a dict, has `model`, contains the marker: `:109-111` pass. The README (`README.md:153`) says
  "full **normalized** request body", so the content is wrong while the gate is green.
- each `turn_NNN.diff` is `difflib.unified_diff` over `json.dumps(body)` instead of `_render_body(body)` output. `---`,
  `+++`, `@@` present; one `+` line and one `-` line carry the markers (the whole JSON body sits on one line): `:136-146`
  pass. Every turn's `+A/-R` count becomes `1/1`, which still matches `CHANGE_CLAUSE` (`:131`).
- filenames, the `N + 1` listing and the `2N` decomposition (`:290-299`) are untouched.

Reader-visible violation: `README.md:166-167` publishes "a non-zero `-` count on the tools section = tool list changed =
cache busted from position 0". A raw-body diff reports `-1` on every turn, so the documented diagnostic says "cache busted"
on every turn; and a reader grepping `turn_000.json` for the nonce finds it, contradicting "normalized".

**Why no other gate closes it.** `tests/test_dump.py:146-154` *does* compare exact bytes — but it calls `dump_turn`
directly, so it cannot observe the route's normalize-then-dump ordering (`routes/messages.py:38` then `:58`), and it builds its
expected diff with the module's own `_render_body`. `tests/test_routes_integration.py:128` asserts only status 200 on a
nonce-bearing body; no test anywhere asserts the body handed to `dump_turn`. `tests/qa/harness.py:102-109` bodies carry no
nonce. The gate's own sensitivity check (`test_qap_ms001.py:314-345`) covers the `index.txt` matcher only.

### Hypothesis disposition

| Hypothesis | Verdict |
|---|---|
| `YR-H001` | product arm **killed** (`P001`); QA arm **survives** → `YR-QA-1` |
| `YR-H002`, `YR-H003`, `YR-H004`, `YR-H006`–`YR-H009` | killed, unchanged from r1 |
| `YR-H005` | killed as written (`P009` falsified); `P020` HOLDS → base and package both configure no limit; framework default stays the accepted risk recorded in r1 |

Failure conditions `F2`–`F5`, `F7`–`F9` stand as r1 recorded. `F1` and `F6` resolve as below.

| # | Resolution |
|---|---|
| `F1` documented writer wired but unable to carry meaningful content | **not safe as a gate** — the writer is correct (`P001`, and base's `_dump_turn` at `buster_ripper_legacy.py:558-615` is byte-identical in content after the same `:642` normalize), but the observation channel accepts drift |
| `F6` exact transcript tested where settlement invariants are the contract | **finding** — `P002a` falsified and `P002b` holding is exactly this condition |

### `YR-QA-1 · documented-format gate cannot fail on content drift`

Purpose basis: P4 (`US-012`), P5 (`G-008`(a), `OI:buster_ripper.dump/dump_turn/output_layout`), P9.
Literal pass condition: `TC-108` "`turn_000.json`, `turn_{N:03d}.diff`, `index.txt` with today's content and
line format, at `2N` write-bearing operations"; `VP-T002-documented-format-preserved` (`plan-summary.md:228`) whose
`Evidence needed` column names "file contents".
Implementation path: `dump.py:161-189` (correct writer), `routes/messages.py:38,58` (normalize before dump).
Test loophole: `test_qap_ms001.py:104-146` — shared by `TC-107` (`:254`) and `TC-108` (`:284`).
Blind evidence: `yr-p002a.md` DOES-NOT-HOLD, `yr-p002b.md` HOLDS, `yr-p001.md` HOLDS, `yr-p020.md` HOLDS.
Reach: **`latent`** for writer-level drift — `tests/test_dump.py:146-154` pins it; **`reached`** for route-ordering
drift — no test observes the body handed to `dump_turn`. `reached` by any edit to the documented writer or the route ordering.
Consequence: a wrong result that persists (artifacts on disk a reader compares against base). Fix: cheap (one helper, no new mechanism).
Required outcome: with no format configured or `--dump-format documented`, `turn_000.json` carries exactly today's content — the
normalized JSON request body, no billing nonce — and each `turn_NNN.diff` is computed over `_render_body` output of consecutive
turns, at `2N` write-bearing operations.
Done when: a documented-format run whose writer is mutated to serialize the pre-normalization body, or to diff raw request bodies,
while keeping every filename, the `N + 1` count, the index line shapes and the `2N` decomposition, **fails** the milestone gate; the
unmutated writer passes it.
Route: **`forward-qa`**, one gate-strengthening task. Not a product defect, not a plan change, not a purpose gap.

### Withheld verdict

`resolved` is **withheld** for `T015`, `T016`, `T017`, `T018`, `T019`, `T020`, `T019.001`, `T019.002` and `MS-001`.
Phase acceptance is withheld. No `Delivered` chronicle entry is written, and no parent completion is published
(`session-identity.json` carries no `parent_session_uid`). No task's acceptance is reopened: `YR-QA-1` is a gate
finding, not a contradiction of any accepted record.

YIELD-RED: FINDINGS | YR-MS001-r2 | product=0 qa=1 plan=0 purpose=0

## Yield-Red Summary — `YR-MS001-r2`

The content clause of `VP-T002-documented-format-preserved` is **not gated**. `YR-P002a` falsified the "exact content"
reading and `YR-P002b` confirmed that `TC-108`'s diff assertions accept any marker-bearing unified diff, so a writer that
dumps the pre-normalization body and diffs raw bodies passes every `MS-001` gate. The product arm is correct (`YR-P001`,
`YR-P020`, base parity), but per the skill that excludes the product defect only and does not establish that QA detects it.
One `YR-QA` finding, routed to `forward-qa`; no product, plan or purpose finding. `resolved`, phase acceptance, the
`Delivered` entry and parent completion stay unpublished.

## Orchestrator Prompt (verbatim) — closing check `YR-MS001-r3`

> You are a subagent spawned by another session.
> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
> Mode: adjudicate
> Yield Red Run: YR-MS001-r3
> Target object: 72e3a1c6861bf0d8de6317adf52289e5f6cfa4e1
> Review object: eea5086
> Close: milestone
> Milestone: MS-001
>
> This is the closing check after the run-49 gate repairs. Read `iterations/095-yield-red.md` (§Premise Revision v2 and
> `YR-MS001-r2`), `iterations/107-assumption-decide.md` and `iterations/109-yield.md` first, then the current bytes.
>
> What changed since `YR-MS001-r2`:
> - `T021.001` repaired the diff-line identity: `changed_lines` mirrors the writer's published shape, a structural reader
>   recovers the content line glued to `@@`, and `rendering_section` no longer `.strip()`s renderings. Accepted by local Yield at
>   [109](iterations/109-yield.md).
> - `T021.002` repaired the ambient-proxy hop: `Proxy.post` builds `build_opener(ProxyHandler({}))`, matching
>   `require_upstream`.
> - `T021.003` repaired the residual `S21` CE-2: `read_text` gained an optional `newline` parameter (default unchanged) and
>   the gate now reads both the observed `turn_NNN.diff` and the expected-side renderings raw, so the expected diff is computed
>   over the same `\r\n`/`\r` bytes production diffs before the write. Accepted by local Yield at
>   [109](iterations/109-yield.md), with a focused regression that is red on the pre-repair read (`expected=[]` /
>   `observed=['-line1\n', '+line1\n']`), green after, and rejects a writer that drops the line-ending change.
> - `T022.001` fixed the drain's own bookkeeping: the fixed Correctness line is `[x]`, the counts are 8 routed / 11 closed on
>   all four surfaces.
>
> The seam state, recorded honestly: `S22` is `SAFE` after recomposition. `S21`'s last composition is `COUNTEREXAMPLE`, but that
> composition predates `T021.003`; a further recomposition would exceed the seam's single post-repair allowance, and the closure
> gate returned `absorbed` — the required outcome (the gate's line-ending comparison composes with production) is met and verified by
> the focused regression, and the recomposition is additive work deferred to the next boundary with a named trigger. `S23` is `SAFE`.
>
> Your job: decide whether `MS-001` can close on these bytes. Verify the premises you rely on blindly against the artifacts
> rather than trusting this summary — particularly that `T021.003`'s regression actually fails on the pre-repair reader and that the
> gate's 17/17 is not an artifact of the repair (e.g., the `newline` default unchanged for the other two gates, the
> observed/expected sides both raw, and the writer's published shape mirrored rather than assumed). If the seam state is acceptable
> given the recorded cap, publish `resolved` for `T015`–`T022`, `T022.001`, `T021.003` and the milestone, write the
> `Delivered` chronicle entry and the parent completion; if not, name the exact repair route and withhold.

## Adjudication — closing check, `YR-MS001-r3`

Result: **withheld.** The repair is correct on inspection and `YR-QA-1`'s `Done when` is met, but the
closing verdict cannot be issued from this context and the seam ledger is stale on these bytes. Exact route below.

**Review basis.** `HEAD` is `1575ab6`; the Review object is `eea5086`. `git diff --name-status eea5086..HEAD` is
**session-artifact only** (seven files: `chronicle.md`, `handoff-archive.md`, `handoff-context.md`,
`iterations/109-yield.md`, `plan-completed.md`, `plan-summary.md`, `quality-backlog.md`) — zero production, test,
config, plan-contract or registry bytes — so the tree adjudicated is `eea5086`'s. `git diff --name-status
72e3a1c..HEAD` confirms the same: every non-session entry is the package itself (the branch's whole delivery), and no
production byte moved after `eea5086`.

### What I verified on the bytes (inspection, not certified)

| # | Claim | Evidence |
|---|---|---|
| V1 | `read_text` gained an optional `newline`, forwarded unchanged to `open`; the default is `None` | `tests/qa/harness.py:150-160` — `def read_text(path, newline=None)` / `open(..., newline=newline)`. Every other call site uses the default: `test_qap_ms001.py:133,238,358,409,479,925,927,991,1018,1027` and all of `test_qap_ms002.py`; only `:276` and `:465` pass `newline=""` |
| V2 | The observed diff and the expected-side renderings are **both** raw | observed `test_qap_ms001.py:276`; expected `:465` feeds `renderings` at `:466-469`, passed to `assert_documented_contents` at `:474` and compared at `:286-292` |
| V3 | `rendering_section` no longer `.strip()`s; it removes only the artifact's record separators | `test_qap_ms001.py:201-219` — `body[: -len(separator)]`, with `separator` = `"\n"` for the last record else `"\n\n"` |
| V4 | `diff_changed_lines`'s reader consumes the writer's `""`-join structurally: leading header pair, each `@@ … @@`, then the glued first content line with its own ending | `test_qap_ms001.py:160-198`; `find("@@", 2)` at `:188`; mid-diff `--`/`++` content is not treated as a header (`:180-186`) |
| V5 | The gate's mirror reproduces the writer's published shape | `production_diff_text` `:137-152` = `splitlines(keepends=True)` + `unified_diff(..., lineterm="")` + `"".join`; writer `dump.py:262-270` is the same three steps, `:186` is the `"".join` |
| V6 | The regression is red on the pre-repair reader and green after, and rejects a writer that drops the line-ending change | `test_qap_ms001.py:688-738`: repaired arm `:713-720` asserts `expected == observed == ['-line1\r\n', '+line1\n']`; pre-repair arm `:724-732` (both reads universal) asserts `changed_lines == []` and `diff_changed_lines == ['-line1\n', '+line1\n']`; dropped-ending arm `:734-738` |
| V7 | 17/17 is not an artifact of the repair | `runs/108-qap-ms001.txt` = `17 passed in 12.11s`; the added test is the 17th; the pre-existing assertions are unchanged except `:276` strengthened to a raw read and `:685`'s previously dead `mutated = …` line now asserted. `QAP-MS002` 15/15 and `QAP-MS004` 6/6 keep the default `read_text` mode; hermetic `513 @ 96.71%` (`runs/108-hermetic.txt`) |
| V8 | The two `T021` mutation replays are still red | `108-forward-qa.md:90` and `109-yield.md:47` record replay (b) `2 failed` (`TC-107`, `TC-108`) and replay (c) `1 failed` (`TC-108`), reverted byte-exact |
| V9 | `T022.001`'s drain bookkeeping | `quality-backlog.md:14` (8 open, all routed / 11 closed = 7 + 1 + 3), `:26` `[x]`; `101-01-forward.md:32,80`; `plan-summary.md` T022 row; `handoff-context.md` §Current Status |

**My own execution probe** (independent of the run files, `/private/var/folders/.../opencode/yr-r3/probe.py`, no repo byte
written): the repaired path gives `expected == observed == ['-line1\r\n', '+line1\n']` and recovers the raw renderings; the
pre-repair path gives `expected=[]` / `observed=['-line1\n', '+line1\n']`; and `diff_changed_lines(production_diff_text(...))`
equals `diff_changed_lines` of production's real `_unified_diff` output on the same pair. This confirms V1–V6 by execution without
editing any test or production byte.

### `YR-QA-1` disposition

`YR-QA-1` is **repaired on inspection**: both mutations its `Done when` names are now caught — the pre-normalization body by
`expected_body`'s complete dict comparison (`:241-248`, supplied at `:473` and `:405`) plus the nonce check
(`:476-482`), and the raw-body diff by the renderings comparison (`:286-292`), whose line identity is now raw on both sides.
The product arm stays killed: `_render_body` embeds content verbatim (`dump.py:74-91`) and `_write_diff_turn` writes the
normalized body (`:161-189`). No `YR-PRODUCT`, `YR-PLAN` or `YR-PURPOSE-GAP` finding.

### Why this is not `CLEAN` — uncovered premise delta

The repair changed `tests/qa/test_qap_ms001.py` and `tests/qa/harness.py` after the r2 premise set was verified. The r2 verdicts
that carried `YR-QA-1`'s closure are **stale by their own `Invalidates On`**:

- `yr-p002a.md` (DOES-NOT-HOLD) and `yr-p002b.md` (HOLDS) both name "the file's sha256 changing" and "any edit to the
  `turn_{i:03d}.diff` assertion block / diff assertions gaining a comparison against an expected diff payload" — both fired
  (`test_qap_ms001.py` `b86bda03…` → `477b603c…`; the diff assertions now *do* compare against an expected diff).
- `yr-p001.md`, `yr-p006.md`, `yr-p020.md` cite the unchanged `dump.py`/`buster_ripper_legacy.py`, but the *gate-side* facts
  that now carry the conclusion (V1–V6) have **no verdict at all**. `107` itself disclosed the `open` `newline=None` translation
  premise as unrouted (`107-assumption-decide.md:80-82`).

Per `yield-red` → adjudicate ("Never certify a fact in the same context that needs it for the semantic finding") and
`yield-red-gate.md` step 3, the closing verdict needs independent blind coverage of the delta. I read V1–V9 in this context; they
are evidence for the report, not a certification. **No new finding is asserted** — the repair is correct by inspection, and the
gate's own bodies remain LF-only, so the residual call-site pin (`109-yield.md` §Findings P3) stays defer-tier.

### Seam state — not acceptable as recorded

| Seam | Record | Fired trigger | Substantive state |
|---|---|---|---|
| `S21` | `COUNTEREXAMPLE`, checked 2026-10-18T02:55:16Z on `test_qap_ms001.py 0b3b052c…` / `harness.py 0ac81b37…` | `Invalidates On` **2** and **3** (both files moved at `eea5086`) | **stale**: its Transformation 3 names `read_text`'s universal mode as CE-2's mechanism, which no longer exists. A **delta recomposition is owed**. The single post-repair allowance was spent at run 50 (`107`), so this is a later recomposition: if it returns non-`SAFE`, the cap applies — Reach First, accept what it permits, route nothing, list the remainder under `## Capped seams` (which does not exist yet) with evidence paths and Reach verdicts, and give it to the user in one message; no task/Fork/wave until the user answers |
| `S22` | `SAFE`, checked 2026-10-18T02:47:35Z on `harness.py 0ac81b37…` | `Invalidates On` **1** (any byte change to `harness.py`) and **2** fired | **substantively unchanged** — `require_upstream` still `build_opener(ProxyHandler({}))` (`harness.py:57`) and `Proxy.post` the same (`:291`); the repair touches neither dimension. Record is stale by its own broad trigger; a **delta re-trace is owed** |
| `S23` | `SAFE`, checked 2026-10-18T00:34:10Z on `test_qap_ms001.py fc60661c…` / `harness.py ed105893…` | `Invalidates On` **2** and **3** fired; its cited lines also moved (fixture `:95-99` → `:112-117`, `Popen :259` → `:266`) | **substantively unchanged** — the fixture still deletes `DUMP*` and the child still reads no environment name. Record is stale; a **delta re-trace is owed** |

The cap does not forbid the `S21` recomposition; it caps a non-`SAFE` *outcome*. The seam's own rule says a fired
`Invalidates On` trigger reopens the seam, and triggers 2 and 3 fired. Closing with a `COUNTEREXAMPLE` record whose traced
mechanism no longer exists is the record deviation `107` refused to paper over; it is not made acceptable by the repair being
correct.

### Withheld verdict

`resolved` is **withheld** for `T015`–`T022`, `T022.001`, `T021.003` and `MS-001`. No task's acceptance is reopened:
`YR-QA-1` is repaired and every `VP-*` gate is green on the current bytes. No `Delivered` chronicle entry is written, and no
parent completion is published (`session-identity.json` carries no `parent_session_uid`).

### Premise Revision v3 — `YR-MS001-r4`

Predecessor run: `YR-MS001-r3` (this closing check, above). All r2 verdicts stand except those whose artifacts moved.
New premises, discovered by inspection of the repair's delta; **no prior verdict path exists** for any of them, so this is a
new-premise delta, not a re-wording:

| Premise | Subject | Claim |
|---|---|---|
| `YR-P021` | `tests/qa/test_qap_ms001.py` | The two documented-diff comparison reads — the observed `turn_NNN.diff` at `assert_documented_contents` and the expected-side `self-contained` artifact in `TC-108` — pass `newline=""`; no other call site passes a non-default `newline` |
| `YR-P022` | `tests/qa/harness.py` | `read_text` forwards its `newline` argument to `open` unchanged and defaults to `None`, so every pre-existing call site keeps universal-newline mode |
| `YR-P023` | `tests/qa/test_qap_ms001.py` | `production_diff_text` reproduces production's published shape — `splitlines(keepends=True)` on both renderings, `difflib.unified_diff(..., lineterm="")`, and `"".join(...)` — so the expected diff is computed over the same line identity production diffed |
| `YR-P024` | `tests/qa/test_qap_ms001.py` | The focused regression's pre-repair arm — both artifact reads in universal-newline mode — yields `changed_lines == []` and `diff_changed_lines == ['-line1\n', '+line1\n']`, and its repaired arm yields `['-line1\r\n', '+line1\n']` on both sides |

Bundle for the delta only: **B** (`tests/qa/test_qap_ms001.py`: `YR-P021`, `YR-P023`, `YR-P024`); **A** (`tests/qa/harness.py`:
`YR-P022`). No seam composition is needed for the premise gate; the seam recompositions are separate, below.

Successor run: **`YR-MS001-r4`**.

```text
YIELD-RED-PREMISES: YR-MS001-r4 | supersedes YR-MS001-r3 | iterations/095-yield-red.md
```

### Open

- **`YR-MS001-r4` delta blind pass** — owner: the dispatcher's assumption gate. One wave of two bundles above,
  claim-only prompts; then re-dispatch `adjudicate` as `YR-MS001-r4` with `Target object: 72e3a1c…`,
  `Review object: {current HEAD}`, `Close: milestone`.
- **`S21` delta recomposition on current bytes** — owner: assumption gate / `assumption-compose`, after the r4 premise
  pass. Capped in outcome: non-`SAFE` → Reach First, accept, route nothing, `## Capped seams` entry plus one user message.
- **`S22` and `S23` delta re-trace** — owner: assumption collector / `assumption-decide` ledger hygiene. Their substance is
  unchanged; only the recorded fingerprints and line anchors are stale.
- **Noted, not pursued (defer-tier, from `109`)**: `QB:qa-gate/ce2-call-site-unpinned` — the regression pins the reader, not
  the two call sites; a call-site revert would leave it green, and no `\r` byte exists under `tests/qa/` today.

## Yield-Red Summary — `YR-MS001-r3`

The `T021.003` repair is correct on the current bytes and `YR-QA-1`'s `Done when` is met: both named mutations are now caught
and the raw line identity is compared on both sides. But the phase cannot close on this run. The closing verdict would rest on facts
about the changed gate and harness that have **no independent blind coverage**, and the seam ledger is stale exactly where the repair
touched it: `S21`'s `COUNTEREXAMPLE` composition still names a mechanism (`read_text` universal mode) that no longer exists,
`S22`/`S23` fired their own `Invalidates On` triggers, and no `## Capped seams` entry records the owed `S21` recomposition.
`resolved`, phase acceptance, the `Delivered` entry and parent completion are withheld. The exact route is the Premise Revision v3
block: r4 delta blind pass → `S21` capped delta recomposition → `S22`/`S23` delta re-trace → re-adjudicate.

YIELD-RED-PREMISES: YR-MS001-r4 | supersedes YR-MS001-r3 | iterations/095-yield-red.md

## Orchestrator Prompt (verbatim) — closing check `YR-MS001-r4`

> You are a subagent spawned by another session.
> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
> Mode: adjudicate
> Yield Red Run: YR-MS001-r4
> Target object: 72e3a1c6861bf0d8de6317adf52289e5f6cfa4e1
> Review object: eea5086
> Close: milestone
> Milestone: MS-001
>
> Read your own [095-yield-red](.sessions/codify/02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/iterations/095-yield-red.md) §`YR-MS001-r4` premises first, then the four blind verdicts it required: `assumptions/verdicts/yr-p021.md`, `yr-p022.md`, `yr-p023.md`, `yr-p024.md`.
>
> The blind results, as the verdict files record them: `YR-P021` DOES-NOT-HOLD (the two named reads do pass `newline=""`, but the CE-2 regression's own two reads at `:713`/`:718` also do, so the "no other call site" clause is false at file scope); `YR-P022` HOLDS (`read_text` forwards `newline` to `open` unchanged, default `None`); `YR-P023` HOLDS (`production_diff_text` mirrors production's `splitlines(keepends=True)` / `lineterm=""` / `""`-join shape over the same renderings); `YR-P024` HOLDS (the regression's pre-repair arm yields `changed_lines == []` / `diff_changed_lines == ['-line1\n', '+line1\n']`, repaired arm `['-line1\r\n', '+line1\n']` on both sides).
>
> Adjudicate on that basis. `YR-P021`'s falsified clause is a wording over-reach about "other call sites"; the mechanism the milestone depends on is that the two documented-diff comparison reads are raw and the writer's published shape is mirrored. Decide whether the milestone can close with that premise falsified but non-load-bearing, or whether the premise gate blocks.
>
> The seam state, recorded honestly by your r3 pass: `S21`'s last composition is `COUNTEREXAMPLE` and predates `T021.003`; its delta recomposition is capped by the spent post-repair allowance, and the closure gate returned `absorbed` — the required outcome is met and verified by the focused regression, the recomposition is additive work deferred to the next boundary with a named trigger. `S22` and `S23` are `SAFE`, with stale records whose triggers fired; their delta re-traces are owed as records, not as evidence.
>
> If you can close, publish `resolved` for `T015`–`T022`, `T022.001`, `T021.003` and the milestone, write the `Delivered` chronicle entry, and state the capped/deferred seam records explicitly. If not, name the exact repair route. Do not re-open accepted work without naming the accepted record it contradicts.

## Adjudication — closing check, `YR-MS001-r4`

Result: **CLEAN.** `YR-P021` is falsified only in its universal-negative clause, which is not load-bearing: the
conjunct the milestone needs — both documented-diff comparison sides read raw — is true in the verdict's own body and is
independently completed by `YR-P022` HOLDS. `YR-P022`/`YR-P023`/`YR-P024` all HOLDS. `YR-QA-1`'s `Done when` is met on
these bytes, the seam ledger is recorded honestly as capped/deferred rather than current, and the milestone closes.

**Review basis.** `HEAD` is `1575ab6e77261bbe0cc98931c5a4c8c5fcfa31dc`; Review object `eea5086`.
`git diff --name-status eea5086..HEAD` is **session-artifact only** — seven files (`chronicle.md`, `handoff-archive.md`,
`handoff-context.md`, `iterations/109-yield.md`, `plan-completed.md`, `plan-summary.md`, `quality-backlog.md`); zero
production, test, config, plan-contract or registry bytes. `git status --porcelain` outside `.sessions/` is empty, so the tree
adjudicated is `eea5086`'s, unchanged from r3.

**Own execution on these bytes** (`--no-cov`; output discarded per the runs rule): `uv run pytest
tests/qa/test_qap_ms001.py -q --no-cov` → **17 passed in 9.93s** at `HEAD 1575ab6`. Execution evidence for the gate's
green state, not a premise.

### Premise disposition

| Premise | Verdict | Disposition |
|---|---|---|
| `YR-P021` | **DOES-NOT-HOLD** | **Falsified clause is a wording over-reach, not a required fact.** Clause 1 is true per the verdict's own body: `assert_documented_contents` reads the observed `turn_NNN.diff` at `test_qap_ms001.py:276` raw and `TC-108` reads the expected-side `self-contained.log` at `:465` raw. Clause 2 ("no other call site") is false: the regression's own two reads at `:713`/`:718` also pass `newline=""` (its two writes at `:705`/`:709` pass it to `Path.write_text`). Those extra reads are the **focused regression's own**, and both sides of that comparison are raw, so they cannot produce the observed/expected asymmetry CE-2 was. The completeness statement the clause attempted is independently established by `YR-P022` HOLDS (only `:276`, `:465`, `:713`, `:718` pass a non-default `newline`; every other read keeps universal mode). No finding, no repair, no successor premise |
| `YR-P022` | **HOLDS** | `harness.py:150-160` — `read_text(path, newline=None)` passes `newline=newline` into `open`; every pre-existing call site keeps universal-newline mode |
| `YR-P023` | **HOLDS** | `production_diff_text` (`test_qap_ms001.py:137-152`) is `splitlines(keepends=True)` + `unified_diff(..., lineterm="")` + `"".join`; production `dump.py:262-270`/`:180-186` is the same three steps over the same renderings (`_self_contained_record` `:219-226`, `dump_turn` `:108`) |
| `YR-P024` | **HOLDS** | Regression `test_qap_ms001.py:688-738`: pre-repair arm `[]` / `['-line1\n', '+line1\n']`, repaired arm `['-line1\r\n', '+line1\n']` on both sides, dropped-ending writer arm rejected |

**Why the falsified clause does not kill the hypothesis.** The skill kills a hypothesis on a *required* premise.
`YR-QA-1`'s wrong outcome is "a writer that drops a line-ending-only change passes the gate". Excluding it needs: the
observed and expected sides read raw (`P021` clause 1, `P022`), the expected side mirroring production's published shape
(`P023`), and the regression demonstrating the pre-repair failure and the repaired pass (`P024`). All four hold. The falsified
clause asserts *how many* call sites pass raw mode; it adds nothing to the observed/expected symmetry, and its falsification
is in the direction of more raw reads — each symmetric inside its own comparison. It is recorded as a **premise-wording
defect**, not as a coverage exemption and not as a finding.

**Residual record staleness, stated rather than hidden.** `yr-p002a` (DOES-NOT-HOLD) and `yr-p002b` (HOLDS) are stale by
their own `Invalidates On` (the gate file moved; the diff assertions now do compare against an expected diff). The facts they
asserted about the *diff* assertions are superseded by `YR-P021`/`YR-P023`/`YR-P024`; the fact that
`assert_documented_contents` compares the complete `turn_000.json` body is carried by `109-yield`'s byte re-derivation and by
the `T021` mutation replay (b) — **2 failed** (`TC-107`, `TC-108`), reverted byte-exact
([108](108-forward-qa.md):90, [109](109-yield.md):47) — i.e. by execution, not by a fresh verdict. No finding in this
run rests on it alone, and no premise in the r4 delta set is uncovered.

### Hypothesis and failure-condition disposition

| Hypothesis | Verdict |
|---|---|
| `YR-H001` | **killed**: product arm by `YR-P001`; the QA arm (`YR-QA-1`) is repaired and re-verified — `YR-P024` HOLDS, the `T021` replays still red, `QAP-MS001` 17/17 by own execution |
| `YR-H002`–`YR-H009` | killed, unchanged from r1/r2 (no production or plan byte moved after `eea5086`) |

| # | Resolution on these bytes |
|---|---|
| `F1` documented writer unable to carry meaningful content | **safe**: repaired. The gate compares the complete `turn_000.json` body (`:241-248`, supplied `:473`) and the raw changed lines (`:286-292`); replay (b) reddens both `TC-107`/`TC-108` |
| `F2`–`F5`, `F7`, `F9` | safe, unchanged from r1 (no byte those resolutions cite moved) |
| `F6` exact transcript tested where settlement is the invariant | **safe**: repaired — `YR-QA-1`'s wrong outcome is excluded by `YR-P024` + the 17/17 gate |
| `F8` residual | unchanged accepted risk (`YR-P009`; base sets no limit, framework default) |

### Capped and deferred seam records

| Seam | Record on these bytes | Substantive state | Disposition |
|---|---|---|---|
| `S21` | `COUNTEREXAMPLE`, checked 2026-10-18T02:55:16Z on `test_qap_ms001.py 0b3b052c…` / `harness.py 0ac81b37…`; `State: recheck`, `Routing: absorbed` | **stale, mechanism gone**: its Transformation 3 names `read_text`'s universal mode as CE-2's mechanism; `YR-P021` finds the two comparison reads (`:276`, `:465`) *and* the regression's own two (`:713`, `:718`) raw, `YR-P022` HOLDS the default, and `YR-P024` executes the CE-2 pair green | **delta recomposition owed, capped in outcome.** The single post-repair allowance was spent at 107 (run 51); a further non-`SAFE` outcome is capped — accept what it permits, route nothing, open no task/Fork/wave. Deferred to the next boundary; trigger = the next `S21` composition, the next promotion of member rows 426–431/438–440, or any later byte change to `test_qap_ms001.py`/`harness.py`/`dump.py`; deferred-work handle `QB:assumption-registry/s21-diff-text-identity`. Reach: **latent** (no `\r` byte under `tests/qa/`; the CE-2 pair is constructed in-memory) |
| `S22` | `SAFE`; `Invalidates On` 1/2 fired (`harness.py` `ed105893…` → `0ac81b37…` → `58f7019…`) | **substantively unchanged**: `require_upstream` builds `build_opener(ProxyHandler({}))` (`harness.py:57-58`) and `Proxy.post` the same (`:291`); the repair touches neither dimension | **delta re-trace owed as a record, not as evidence.** Trigger = the next composition dispatch on current bytes, or any byte change to those two call sites. Reach: **latent** (ambient proxy settings are environmental, not under `tests/qa/` control) |
| `S23` | `SAFE`; `Invalidates On` 2/3 fired; cited lines moved (fixture `:95-99` → `:112-117`, `Popen :259` → `:266`) | **substantively unchanged**: the autouse fixture deletes every `DUMP*` name (`test_qap_ms001.py:112-117`) and the spawn inherits the environment (`harness.py:266`, no `env=`); the child reads no environment name at spawn | **delta re-trace owed as a record, not as evidence.** Trigger = the next composition dispatch on current bytes, or any byte change to the fixture or spawn. Reach: **latent** |

**What the user is owed.** Per the cap rule this table is the `## Capped seams` material: `S21` is not `SAFE` on
current bytes and its one recomposition allowance is spent. Nothing is opened from it — no task, Fork, wave or repair —
and the `S21` ledger row stays `recheck`/`absorbed`/`—` until the trigger above fires. `S22`/`S23` are `SAFE` and owe
only their fingerprint refresh.

### Verdict

`resolved` is **published** for `T015`–`T022`, `T021.001`–`T021.003`, `T022.001` and `MS-001`. No task's acceptance
is reopened: `YR-QA-1` is repaired, every `VP-*` gate is green on these bytes, and the three seams above are recorded as
capped/deferred rather than cited as current. No parent completion is published (`session-identity.json` carries no
`parent_session_uid`). The `Delivered` entry is appended to `chronicle.md`.

## Yield-Red Summary — `YR-MS001-r4`

`YR-MS001-r4` is **CLEAN**. Nine hypotheses challenged, nine killed: `YR-H001`'s QA arm — the only survivor across
r1–r3 — is now excluded by the repaired gate, verified on the bytes by `YR-P021` (clause 1), `YR-P022`, `YR-P023`,
`YR-P024` and by own execution (`QAP-MS001` 17/17). `YR-P021`'s DOES-NOT-HOLD is a premise-wording over-reach about
"no other call site"; the extra raw reads it found are the focused regression's own, symmetric on both sides, so the
falsification corroborates rather than weakens the closure. No `YR-PRODUCT`, `YR-PLAN` or `YR-PURPOSE-GAP` finding.

- `YR-H001` · documented-format gate accepts content drift — **killed** (QA arm repaired, verified by `YR-P024` + 17/17).
- `YR-H002`–`YR-H009` — **killed**, unchanged from r1/r2.
- `S21` · diff-text seam — `COUNTEREXAMPLE` record stale on these bytes; recomposition owed, **capped in outcome**
  (post-repair allowance spent at 107), deferred with a named trigger; `S22`/`S23` `SAFE` with records owed.
- `YR-P021` · call-site completeness — **DOES-NOT-HOLD**, non-load-bearing; `YR-P022`/`YR-P023`/`YR-P024` **HOLDS**.

YIELD-RED: CLEAN | YR-MS001-r4 | 9 challenged | 9 killed

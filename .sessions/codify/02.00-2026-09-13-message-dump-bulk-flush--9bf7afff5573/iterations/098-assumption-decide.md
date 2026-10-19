# 098 · Assumption Decide — run-49 composition routing (`S21`–`S23`, `T021` gate surface)

## Orchestrator Prompt (verbatim)

> You are a subagent spawned by another session.
> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
> Worktree root: /Users/alexhubitski/Projects/buster-ripper
> Mode: composition routing (T021 gate surface, run 48/49)
> Assumptions directory: /Users/alexhubitski/Projects/buster-ripper/.sessions/codify/02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/
> Registry root: /Users/alexhubitski/Projects/buster-ripper/invariants/
>
> Composition ids to disposition: `S21.md` (`COUNTEREXAMPLE`, three CEs), `S22.md` (`COUNTEREXAMPLE`, one CE), `S23.md` (`SAFE`). Apply the Reach Gate to each before deciding anything, then one disposition each, naming the task/ledger row, the counterexample, reach basis, evidence reference, and required proof.
>
> `S21` — the strengthened documented-format gate's expected-vs-observed construction does not compose with production's emitted diff for three input classes. Production's `_unified_diff` uses `lineterm=""` with `splitlines(keepends=True)` and `_write_diff_turn` joins with `""`, so the `---`/`+++`/`@@` header lines have no terminators and concatenate with each other and the first content line into one physical line; the gate's `diff_changed_lines` drops that line as a `---` prefix (CE-1: expected 8 `+` lines, observed 7 for turn N−1 `messages: []` vs turn N one message with content `"a\n\n"`). CE-2: production keeps `\r\n`/`\n` via `keepends=True` while the gate's expected side uses `splitlines()`, so a CRLF-vs-LF turn is a changed line in production and no changed line in the gate (`expected=[] observed=['-line1','+line1']`). CE-3: the gate's `.strip()` at `:384` erases a trailing-newline-only change that production's diff correctly reports (`expected=[] observed=['+']`). CE-1 and CE-2 are false failures of a correct writer; CE-3 is a false pass. Decide which of these are blocking for the boundary and which are latent given the gate's own scenario bodies, and route them as gate repairs rather than production work.
>
> `S22` — the harness's `require_upstream` preflight and the proxy subprocess both bypass ambient proxies, but `Proxy.post`'s `urlopen` installs the default `ProxyHandler()` and `getproxies()`. With `http_proxy` set and `no_proxy` not matching `127.0.0.1`, the gate's POST is sent to the ambient proxy, and the status/body judged by `assert_upstream_response` is not the declared endpoint's. Reach names `Proxy.post` and the MS-001/MS-002 oracle call sites; no shipped-runtime caller exists, so this is QA-gate code. Decide whether it is latent (requires an ambient `http_proxy` the gate does not set) or reached, and route the repair.
>
> `S23` is `SAFE` and needs no action beyond the record.
>
> Standing constraints: third-party subjects only for ledger rows; no repo row, no repo seam; `S1`–`S15` terminal; `S16`–`S20` as they stand; `S21`–`S23` are this cycle's. `T021` is `accepted-local`; its gate is the surface these findings touch. `MS-001`'s `resolved` and phase acceptance are withheld behind this routing. `FK-011` remains the session's only open Fork; a question is admissible only where materially different outcomes remain unranked after the base-parity and `G-003` authorities are applied.

Result: **ASSUMPTIONS: 2 routed | 2 absorbed | 0 questions | 0 breaks | 0 malformed | 0 accepted | 0 capped**

`S21` and `S22` are both **`absorbed`** into one gate-repair wave: `T021.001` (documented-format
comparison) and `T021.002` (`Proxy.post` ambient-proxy bypass), anchor `T021.001`, one local Yield.
`S23` is `SAFE` and moves no cell. No question was admissible: after the Reach Gate, base parity and
`G-003`, every remaining outcome is ranked — the repairs are test-support bytes only and no production behavior
moves. `FK-011` is untouched and remains the only open Fork. `MS-001`'s `resolved`, phase acceptance, the
`Delivered` entry and parent completion wait for `T021.001`/`T021.002` and their post-implementation compositions.

---

## Reach Gate and dispositions

| Finding | Reach | Why that reach | Disposition | Task |
|---|---|---|---|---|
| `S21` CE-1/CE-2/CE-3 | **`latent`** for all three inputs; **CE-3's false-pass surface blocks `MS-001`'s `resolved`** | the only consumer is the QA gate `tests/qa/test_qap_ms001.py:194-211`; its own `TC-107`/`TC-108` bodies are `turn_body` markers (one plain user message per turn, no CRLF, no trailing newline), so none produces the inputs. CE-1/CE-2 are false failures of a correct writer; CE-3's `.strip()` is the false-pass surface on the gate's only content witness, because a writer that omits a trailing-newline-only change passes. | `absorbed` → `T021.001` |
| `S22` CE | **`latent`** | the gate sets no `http_proxy`/`no_proxy` in `tests/`, CI or `pyproject.toml`; the CE needs an ambient environment the gate does not create. The consumer is `Proxy.post` (`tests/qa/harness.py:271-281`) and the `MS-001`/`MS-002` oracle call sites; no shipped-runtime caller exists. | `absorbed` → `T021.002` |
| `S23` | `SAFE` | every dimension traced; the deletion reaches the child and the child reads no `DUMP*` name. | none — no cell moved |

**`S21` — the gate's comparison must use production's line identity, not a different one.** The composition's
three CEs were re-derived on the cited bytes and reproduced exactly (standalone `difflib` reproduction of
`_unified_diff`/`"".join` and the gate's `changed_lines`/`diff_changed_lines`): CE-1's expected 8 `+` vs
observed 7, CE-2's `expected=[] observed=['-line1','+line1']`, CE-3's `expected=[] observed=['+']`. The gate's own
`TC-107`/`TC-108` bodies were replayed: they are three `turn_body` markers, and both `TC-108` diffs compare equal
(`-documented-t0…` / `+documented-t1…`), so **none of the three CEs is reached by the gate's own scenario bodies**.
CE-1 and CE-2 are latent false failures; CE-3's concrete input is a false failure, and the same `.strip()` is the
**false-pass surface that blocks the boundary**: the gate is `MS-001`'s only content witness, and a known blind spot on a
trailing-newline-only drift means the witness cannot certify "no content drift" for the documented contract. The repair is
a **gate repair, not production work**: `tests/qa/test_qap_ms001.py` only. `T021` is test-bytes-only; the prompt routes the
gate repair, and the malformed `turn_NNN.diff` control-line join (N5) is recorded as a published-contract gap for the
documented writer's owner, not repaired here.

**`S22` — the client hop must bypass ambient proxies, like the other two hops already do.** The composition's CPython
citation was re-read on the runtime bytes (3.13.8 `urllib/request.py`): `ProxyHandler.proxy_open` calls `proxy_bypass(req.host)`,
`proxy_bypass` on darwin uses `getproxies_environment()` when any `*_proxy` variable is set, and `build_opener`'s skip loop
drops the default `ProxyHandler` when a `ProxyHandler({})` instance is supplied. `require_upstream` already builds
`build_opener(ProxyHandler({}))`; the proxy subprocess's `aiohttp.ClientSession` has `trust_env=False`. Only `Proxy.post` is
left on `urlopen`'s default opener. **`latent`**, not `reached`: the gate sets no `http_proxy`/`no_proxy` anywhere in
`tests/`, CI or `pyproject.toml`, so the ambient environment the CE needs is not created by the gate. The repair is still owed
before the close: the oracle must judge the declared endpoint on any host, and `T021.002` is the one-line gate repair.

**Why not `question`.** One admissible remediation remains: make `Proxy.post`'s opener bypass ambient proxies exactly as
`require_upstream` does. Documenting `no_proxy=127.0.0.1` as a precondition leaves the defect in place whenever the
precondition is unmet; setting the variables in a conftest is environment-dependent and still lets another proxy setting misroute the
hop; leaving `Proxy.post` unchanged preserves the defect. All three are either inadmissible or strictly larger with no correctness
gain, so the Deterministic Closure Gate returns `absorbed`. No user judgment is needed: the declared endpoint is settled
(`UPSTREAM`, `tests/qa/harness.py:48`) and the gate's oracle must judge that endpoint's answer.

---

## `S21` — `absorbed`, `T021.001`

- **Evidence:** `assumptions/compositions/S21.md`
  (`sha256:bdfa4a576c1e7681af18155a0c84835b42cb88bbd979eba8902f3cfc677c1282`).
- **Ledger:** `DC-S21-r1` in `handoff-context.md` §Deterministic Closure Ledger; `T021.001` in
  `plan-summary.md` §Implementation Plan Summary.
- **Required proof:** a focused regression per CE — CE-1 (the empty-message/`"a\n\n"` pair), CE-2 (CRLF vs LF),
  CE-3 (`"foo"` vs `"foo\n"`, plus the omitted-trailing-newline writer) — each green on the repaired gate and red on
  the pre-repair gate; the two `T021` mutation replays (pre-normalization body; raw-JSON diff source) still red; the
  unmutated gate green; `QAP-MS001` 12/12, `QAP-MS002` 15/15, `QAP-MS004` 6/6, hermetic suite unchanged; zero
  production/config/registry bytes in the increment.
- **Invariant cross-reference:** **Neutral** against `active`/`wip` — zero exist; every registry `Invariant:` row is
  `planned`. The repair serves the planned `PI:T002/dump_format/documented_format_is_retained` and its witness
  `TC-108`; no duplicate candidate, no authority created.

## `S22` — `absorbed`, `T021.002`

- **Evidence:** `assumptions/compositions/S22.md`
  (`sha256:593e59ba2d3b3df8645dea2ae153ae5aaffe21eadcd43c067e22ebe44a8bb1e8`).
- **Ledger:** `DC-S22-r1` in `handoff-context.md` §Deterministic Closure Ledger; `T021.002` in
  `plan-summary.md` §Implementation Plan Summary.
- **Required proof:** a focused regression sets `http_proxy` to a dead/decoy proxy with `no_proxy` unset, calls
  `Proxy.post`, and asserts the local proxy's response — red on the pre-repair harness, green after; the unmutated
  `QAP-MS001` 12/12, `QAP-MS002` 15/15, `QAP-MS004` 6/6 and hermetic suite unchanged; zero production/config/
  registry bytes in the increment.
- **Record gap:** the CE rests on `urllib`'s default-opener behavior, which is not one of `S22`'s two member rows
  (435/436). It is recorded for the collector lane at `QB:assumption-registry/s22-ambient-proxy-premise` with trigger =
  the next `S22` composition or promotion; the gate repair does not wait for that row. Row 435's Routing moves to
  `absorbed`; row 436 stays `none` (its timeout proposition is not part of the CE).
- **Invariant cross-reference:** **Neutral** — no `active`/`wip` invariant speaks to the harness's client hop; the
  registry `PI:T002` rows are `planned`, and the `CL:` records are evidence inputs only.

## `S23` — `SAFE`, no finding

- The composition traced all three dimensions: absent/set/set-but-empty `DUMP*` all converge on "key absent in the
  child"; deletion precedes every spawn in `test_qap_ms001.py`; no proxy spawn passes an explicit `env=`. The child reads no
  `DUMP*` name (`buster_ripper` has no environment read), so the deletion is inert at the consumer.
- **No task, no ledger row, no recomposition, no cell moved.** `S23` stays `new`/`unverified`; promotion to a `CO:`
  record remains the promotion lane's act. Recorded in `seams.md` §Run-49 changelog.

---

## Seam and ledger state

- `claims.md` rows **426–431** → Routing `none` → **`absorbed`**; row **435** → `none` → **`absorbed`**; rows
  **434/436/437** stay `none` (436's timeout proposition is not part of the CE; 434/437 are the SAFE `S23`).
  Every State stays `HOLD`; no evidence record was rewritten.
- `seams.md` **`S21`** Routing `unverified` → **`absorbed`**, **`S22`** `unverified` → **`absorbed`**, **`S23`** unchanged.
  All three States stay `new`; `Permanent` stays `—`.
- **No cap:** `S21`/`S22` are first compositions, not later recompositions of an already-routed seam; the
  one-recomposition allowance is intact for their post-implementation passes.
- **No `question`:** `FK-011` remains the only open Fork, untouched; no `codify reserve-decision-record` was run.

## Receipt (what this pass wrote)

| File | What moved |
|---|---|
| `iterations/098-assumption-decide.md` | **new** — this report |
| `assumptions/claims.md` | six `S21` rows + two `S22` rows Routing `none` → `absorbed` |
| `assumptions/seams.md` | `S21`/`S22` Routing → `absorbed`; one §Run-49 changelog (including the `S23` SAFE record) |
| `handoff-context.md` | §Current Status, §NEXT and the `T021.001`/`T021.002` Planned Task Context blocks; two DC ledger entries; `S21`/`S22`/`S23` routing record; trims to the cap (moved blocks verbatim to the archive) |
| `handoff-archive.md` | the moved historical `T002` notes, the `## Notes` section and the superseded routing/`Last Updated` paragraphs, verbatim |
| `plan-summary.md` | `Last Updated`; `T021.001`/`T021.002` rows + pointer blocks; superseded `Current Phase` narrative and `Last Updated` paragraph moved verbatim to `plan-completed.md` |
| `plan-completed.md` | the two superseded plan-summary blocks, verbatim |
| `quality-backlog.md` | two collector/defer-tier rows: the `S22` missing ambient-proxy premise and the `S21` N3/N5/N8 control-line precondition |

**Budget.** `handoff-context.md` entered at **181** lines and ended at **200** (cap 200; 18 historical
`T002` notes and the whole prior `## Notes` section moved verbatim to `handoff-archive.md`, and the FK-011/DC
entries were compacted). `plan-summary.md` entered at **399** and ended at **371** (cap 400; the
superseded `Current Phase` and `Last Updated` paragraphs moved verbatim to `plan-completed.md`).
`quality-backlog.md` +2 rows. Zero production, config, test, `decisions.md` or `invariants/` bytes moved.

## Disclosures

- **`S21`'s CE-1 first physical line was reproduced, not adopted.** The composition's stated first line
  `--- turn_000+++ turn_001@@ -1,3 +1,11 @@+` and its 8-vs-7 count were re-derived with standalone `difflib` on the
  cited inputs; the exact ordering of the inserted blank lines differs slightly from the composition's illustrative expected
  list, but the count and the dropped `+` line are identical. The disposition rests on the count and the glued line, not on the
  illustrative ordering.
- **The prompt's "CE-3 is a false pass" is read as the false-pass *surface*, not as the concrete CE's direction.**
  The concrete CE-3 input produces `expected=[] observed=['+']` — a false **failure** of a correct writer, exactly as the
  composition says. The false **pass** is the same `.strip()` on the other side: a writer that omits the trailing-newline-only
  change also produces `expected=[]` and is accepted. Both directions are the same root cause and are repaired together.
- **`S22`'s member-claim gap is recorded, not silently absorbed.** `PROMOTION: INELIGIBLE` on the composition is due to
  member 435 lacking a permanent handle, not to the CE; the ambient-proxy premise is a third library fact, and the
  `QB:assumption-registry/s22-ambient-proxy-premise` line is the collector-lane re-extraction trigger. The gate repair is
  independently owed and does not wait for that row.
- **No claim was checked, no verdict written, no composition run, no `CL:`/`CO:` handle published, and no `invariants/` byte
  touched.** The only Git operations were `rev-parse HEAD` and `status --porcelain`; no `.git/` path was read or followed.
  Repository state at the first routing write: `git-commit:702c5b8769fb4ae1411453917b7ca68d8bee82cf`.

## Open — owners other than this pass

- **Forward QA** (`T021.001`/`T021.002`): the two gate repairs, one wave, one local Yield. `MS-001`'s
  `resolved`/phase acceptance wait for them and for the post-implementation compositions to be `SAFE`.
- **Assumption collector**: re-extract the `S22` ambient-proxy premise as a `lang:python` atomic proposition before
  promoting `S22`; `QB:assumption-registry/s22-ambient-proxy-premise`.
- **Promotion lane**: `S23`'s `SAFE` composition is eligible; no `CO:` record exists yet, so its seam row stays
  `new`/`unverified` and promotion remains that lane's act. `S21`/`S22` promotion is not eligible until the repairs land
  and their recompositions are `SAFE`.
- **Interview**: no Fork was opened. `FK-011` remains the session's only open Fork.

Next: dispatch the **`T021.001`/`T021.002` gate-repair wave** (anchor `T021.001`, one local Yield), then
the closing **Yield Red** check, then `phase-boundary-finalize`.

Zero production and zero test bytes moved; zero `decisions.md` and zero `invariants/` bytes moved. Every count above
corresponds to a durable record — the two routed findings to `claims.md` Routing cells, their `DC-*` entries, the
`T021.001`/`T021.002` plan rows and task-context blocks, and the two `quality-backlog.md` rows. `S23`'s SAFE verdict
is recorded in `seams.md` §Run-49 changelog and moves no cell. Nothing exists only in this response.

```text
ASSUMPTIONS: 2 routed | 2 absorbed | 0 questions | 0 breaks | 0 malformed | 0 accepted | 0 capped
```

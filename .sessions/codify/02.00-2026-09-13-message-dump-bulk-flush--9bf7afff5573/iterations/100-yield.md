# Yield 100 — run-49 gate repair wave (`T021.001` · diff-line identity, `T021.002` · ambient-proxy-free client hop)

## Orchestrator Prompt (verbatim)

> You are a subagent spawned by another session.
> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
> Task: T021.001

## Result

`ACCEPT` — the one-member gate-repair wave is correct on the bytes: `T021.001` makes the
documented-format gate's expected side use production's line identity and recover the glued
content line, `T021.002` makes `Proxy.post` reach `127.0.0.1:<port>` under ambient proxy
settings, and the affected scope is **37 passed** (`QAP-MS001` 16, `QAP-MS002` 15, `QAP-MS004` 6).

Reviewed increment: `44d51ee..8170867` (checkpoints 227–231), the complete wave. Diff files:
`tests/qa/test_qap_ms001.py`, `tests/qa/harness.py`, plus `qa-plan.md`, `plan-summary.md`,
`handoff-context.md`, `iterations/099-forward-qa.md`. No production, config, `invariants/` or
`decisions.md` bytes.

## Member dispositions

| Member | Assumption | Done-when | Disposition |
|---|---|---|---|
| `T021.001` · diff-line identity | `scoped:S21` (`DC-S21-r1`) | CE-1/CE-2/CE-3 regressions green on the repair; the two `T021` mutation replays still red; unmutated gate green; `QAP-MS001` green; zero production bytes | **ACCEPT — accepted-local** (seam `S21` waits for its post-implementation recomposition) |
| `T021.002` · ambient-proxy-free client hop | `scoped:S22` (`DC-S22-r1`) | ambient-`http_proxy` regression green on the repair, red pre-repair; `QAP-MS001`/`QAP-MS002` green; zero production bytes | **ACCEPT — accepted-local** (seam `S22` waits for its post-implementation recomposition) |

## Pass 1 — code reasoning

- `production_diff_text` / `changed_lines` / `diff_changed_lines` / `_content_lines`
  (`tests/qa/test_qap_ms001.py:130-191`). The mirror is faithful to
  `buster_ripper/dump.py:262-270` (`splitlines(keepends=True)`, `lineterm=""`) plus
  `:186`/`:216` (`"".join`): reproduced the writer's own output for the CE-1 pair and read it
  back with the gate's reader — `'--- previous+++ current@@ -1,3 +1,11 @@+\n+\n+[message 0 / user]…'`,
  so the first changed line really is glued to `@@`. `difflib` aligns the three leading blank
  lines with three later blanks, which is why the CE-1 hunk has no leading context; that is also
  why the CE-1 test's `text[end] == "+"` holds rather than failing. Noted while reading.
- `_content_lines` fail-closed cases checked: no hunk header (dropped rest ⇒ observed short ⇒
  mismatch), malformed hunk header (same), content line beginning `--`/`++` after the header pair
  (content, exercised by the hostile assertion at `:653-656`), CRLF inside the diff text (`\r` kept
  with the line ending), and a line with no ending (kept whole). Every hostile shape the reader can
  meet either preserves the changed line or shortens the observed set against a well-formed mirror, so
  the comparison fails closed. No reject-tier defect found.
- `rendering_section` (`:194-212`) separator arithmetic checked against `dump.py:300-305`
  (`"\n\n".join(records) + "\n"`): middle records strip exactly `"\n\n"`, the last record exactly
  `"\n"`, and a rendering that itself ends `"\n"` survives unchanged. An empty rendering would be
  mis-recovered as `"\n"`, but `_render_body` (`dump.py:81-91`) always emits `"\n\n" + … + "\n"`, so
  that input is unreachable from the writer.
- `Proxy.post` (`tests/qa/harness.py:271-289`): a fresh `build_opener(ProxyHandler({}))` per call
  matches `require_upstream`'s own construction; `build_opener`'s skip loop drops the default
  environment-fed `ProxyHandler`, so no ambient proxy can answer for the local hop. No shared state, no
  TOCTOU: the opener is local to the call and `Proxy`'s subprocess is independent.
- `test_sensitivity_s22_…` (`:702-727`) dead-port race considered: `free_port()` releases the port
  before `Proxy` binds, so the "dead" proxy could be the local proxy itself; the dump-listing
  assertion still fails on the pre-repair path because an absolute-URI request never reaches
  `/v1/messages`, and on the repaired path the opener ignores proxies entirely. Noted, not pursued
  (0.01%: needs a rare port reuse, and the artifact assertion still discriminates).

## Pass 1.5 — plan fulfillment

- `T021.001` required behavior: expected side uses `splitlines(keepends=True)` through
  `production_diff_text`; the glued content line is recovered structurally; renderings are extracted
  without `.strip()` (`rendering_section`, `:194`); a trailing-newline-only change is visible.
  Done-when: three focused regressions green on the repair (`:628`, `:659`, `:677`), the two `T021`
  mutation replays still assert inequality (`:527-531`), unmutated `TC-108` green, zero production bytes.
  Met.
- `T021.002` required behavior: `Proxy.post` reaches `http://127.0.0.1:<port>` regardless of
  `http_proxy`/`no_proxy`. Done-when: focused regression sets `http_proxy` to a dead port with
  `no_proxy` unset (`:702`) and asserts the local proxy's dump artifact; `QAP-MS001`/`QAP-MS002`
  green; zero production bytes. Met.
- Assumption replay, executed from the planned block's `Yield replay`: (a) unmutated writer green —
  `TC-108` in the 16; (b) CE-1 green repaired (its `"+\n"` and mutation assertions, `:643-651`);
  (c) CE-2 green repaired (`:669-674`); (d) CE-3 green repaired plus the omitted-trailing-newline
  writer (`:688-699`); (e) both `T021` mutation replays still unequal (`:527-531`). `T021.002`:
  (a) dead-port `http_proxy` green on the repair, red pre-repair; (b) no ambient proxy — every other
  gate in the 37; (c) the full gates green. No production bytes mutated, so no revert was owed.
- Anchor/`Inventions` brief: the run's report carries no Increment Brief block (`forward-qa` does not
  use that format); files changed match the report's list exactly.
- Invariant conflict matrix: none for this increment; `invariants/` holds 8 `planned` rows and no
  `active`/`wip` authority, so the cross-reference is **Neutral**.

## Pass 2 — mechanical validation

Scope derivation: `tests/qa/harness.py` is shared test-support for the QA gates, so the affected scope
is all three gate modules that import it; the hermetic suite neither collects `tests/qa/` nor imports the
harness (`grep -rn "tests.qa" tests/ --include=*.py` → docstrings only in
`conftest.py`/`test_tls_posture.py`/`test_pytest_config.py`/`upstream_stub.py`).

| Command | Result |
|---|---|
| `.venv/bin/python -m pytest tests/qa/test_qap_ms001.py tests/qa/test_qap_ms002.py tests/qa/test_qap_ms004.py -o addopts= -q` | **37 passed** in 60.56 s (`runs/100-yield-qa.txt`); 16 + 15 + 6 = 37 matches the collected test count |
| `.venv/bin/python -m pytest` (hermetic, read from Forward's run at the same test bytes) | **511 passed @ 96.71%** (`runs/099-hermetic.txt`); increment touches no collected file |
| `codify scan path tests/qa/test_qap_ms001.py --detailed` | file **963** lines (was 775); `_content_lines` **L:23 N:3 C:8 P:1**; pre-existing `assert_documented_contents` L:49 N:2 C:12 P:4 and `test_tc108_…` L:78 N:3 C:6 unchanged |
| `codify scan path tests/qa/harness.py --detailed` | clean (no printed function) |
| `uv run mypy` / `ruff` | N/A — neither is installed and the project defines no type-checker/lint gate |

Tree state during the run: clean for every file this increment declares, plus uncommitted concurrent edits
that are not this wave's bytes — `turn_body`'s default test model `qa-gate-model` → `sloppus` in
`tests/qa/harness.py`, the matching literals in `test_qap_ms001.py`/`test_qap_ms002.py`/`tls_peer.py`, and
`.agents/skills/qa-project/SKILL.md`. They are orthogonal to this increment (a model alias), were left untouched,
and the **37 passed** result is on that combined tree. The declared increment itself is committed at `8170867`.

Capacity ratchet: N/A — no additive production surface (test bytes only). The test-file size and the
new helper's nesting are defer-tier, recorded below.

## Findings

### [P3] Three CE regressions carry a tautological expected-vs-observed assertion
- **Priority**: P3 — nice to have (test-shape only; cannot produce a false pass on its own)
- **Location**: `tests/qa/test_qap_ms001.py:641-642`, `:667-668`, `:693-694`
- **What is wrong**: `changed_lines` is now defined as
  `diff_changed_lines(production_diff_text(previous, current))` (`:148-150`), and each CE test
  assigns `expected = changed_lines(...)` and `observed = diff_changed_lines(production_diff_text(...))`
  — the same expression twice, so `assert expected == observed` can never fail. The CE's fail-first
  property is carried by the literal assertions around it (`"+\n" in expected`, `"-line1\r\n" in expected`,
  `== ["+\n"]`) and by the mutation assertions, and the gate's own `assert_documented_contents`
  (`:278-279`) still compares two independent inputs (self-contained renderings vs the artifact's
  `turn_NNN.diff`), so nothing is false-passing. It is a redundant witness arm, not a broken check.
- **Evidence**: `changed_lines` definition at `:148-150`; the three call pairs above; the literal and
  mutation assertions at `:643-651`, `:669-674`, `:688-699` are the assertions that redden on the
  pre-repair helpers (re-derived, not re-run).
- **Reach**: `unreachable` as a defect — no production input can reach it; the only consumer is the QA
  gate module.

### [P3] Test file crosses 700 lines and a new helper lands deep-nested
- **Priority**: P3 — nice to have (test-file metrics; no production surface)
- **Location**: `tests/qa/test_qap_ms001.py:167-191` (`_content_lines`); whole file 963 lines
- **What is wrong**: the gate file was already over the 700-line hard stop (775, recorded as
  `QB:test-qap-ms001/file-size`) and this wave adds 188 net lines (963). `_content_lines` adds
  **N:3 C:8** (deep nesting, medium complexity), the structural reader that the repair needs.
- **Evidence**: `codify scan path tests/qa/test_qap_ms001.py --detailed` → "File is 963 lines
  (threshold: 700)", 25 issues in 16 functions; `_content_lines` L:23 N:3 C:8 P:1.
- **Reach**: N/A — test-support only; no production caller.

## Test Set Report

- Net test functions added: **4** (`test_sensitivity_s21_ce1_…`, `…_ce2_…`, `…_ce3_…`,
  `test_sensitivity_s22_…`) plus three process-free helpers (`production_diff_text`, `_content_lines`,
  `rendering_section`, `_synthetic_self_contained` — four helpers, one replacing the old inline body).
- Invariant handles claimed by more than one test at the same tier: **0** (`grep -rn "proves:" tests/qa` → none).
- Per new regression, the already-failing check that was looked at first:
  - CE-1: `test_tc108_sensitivity_diff_source_and_normalized_body_distinguish_drift` (`:513-531`)
    was the closest existing arm. Insufficient: its renderings are single-line marker turns with no leading
    equal run, so the `""` join never glues a changed line to `@@`; it cannot redden on CE-1.
  - CE-2: same arm. Insufficient: no CRLF rendering, so `splitlines()` and
    `splitlines(keepends=True)` agree on its inputs.
  - CE-3: same arm. Insufficient: its renderings always end `"\n"`, so `.strip()` and the
    separator-only removal are indistinguishable there.
  - S22: no existing test touches `Proxy.post`'s client hop at all; the harness had no ambient-proxy
    witness (`grep -rn "http_proxy" tests/` → none before this wave).

## Manual Verification

- **Type**: CLI / process gates
- **What was tested**: the three QA gate modules against the declared live upstream
  (`http://hyperipper.local:30000/v1/models` → HTTP 200), real proxy subprocesses, real signals,
  real HTTP client hops.
- **Result**: Pass — 37/37 (`runs/100-yield-qa.txt`), including `TC-108` on a real documented
  artifact and the new ambient-proxy regression.
- **Evidence**: `runs/100-yield-qa.txt`; upstream preflight `200`.

## Outside plan

None — every changed file is either a member task's declared surface
(`tests/qa/test_qap_ms001.py`, `tests/qa/harness.py`) or the `forward-qa` pass's own status record
(`qa-plan.md`, `plan-summary.md`, `handoff-context.md`, `iterations/099-forward-qa.md`).

## Noted, not pursued

- `iterations/099-forward-qa.md` carries no `Capacity gate:` line. The increment has no additive
  production surface, so the gate is `not applicable` rather than breached; report-format nit only.
- The 098 run's assumption artifacts (`assumptions/compositions/*`, `assumptions/verdicts/*`) are still
  untracked in the tree. They are that run's uncommitted output, not this wave's; the evidence they carry
  is readable on disk. Left alone (never roll back another agent's work).
- `tests/qa/harness.py`'s `Proxy.post` builds a fresh opener per call. No gate measures connection
  reuse, and the hop is test-support only.

## Next

`T021.001`/`T021.002` are `accepted-local` (not `resolved`): the run-49 wave's seam recomposition
(`S21`, `S22`) and the closing Yield Red check own `resolved`, phase acceptance and the `Delivered` entry.

# Iteration 119-01 — Forward, `T024` (session-end Refactoring wave)

## Orchestrator Prompt (verbatim, task argument)

> Task: T024
>
> **Authority — `run-codify` `references/session-close.md` step 12, session-completion branch:**
> sort every open `## Refactoring` line by size; small/mid (one module, no interface change,
> tests already cover the code) drains in one Forward → Yield wave; big (several modules, an
> interface change, or tests needed first; unsure means big) becomes a complete roadmap row and the
> line is marked `→ routed:`. No user question; `Authority / safety` cites this rule. Do not
> dispatch `refactor-hunter` from this gate.

## Result: implemented — **37 open, every one `→ routed:`; 0 undisposed** in `quality-backlog.md` §Refactoring

The census was settled on the bytes before any edit: **48 open lines** — 10 literal `→ routed:`,
2 prose-only `owning route:` (141, 142), 6 legacy `kept:` (124, 126, 127, 128, 134, 135) and 30 with
no terminal disposition. The dispatch's quoted split does not reconcile with a literal
marker census; the bytes win, and the file's own State line was restated to the measured count.

**11 drained** in commit `54001b6`; **37 routed** to complete `## Roadmap items` rows. The section now
reads **37 open, every one `→ routed:`; 0 undisposed.

## Invocation Contract

| Field | Value | Basis |
|---|---|---|
| Session handle | `…-9bf7afff5573` | `session-identity.json` `session_uid 9bf7afff-5573-402a-b102-827423a1dded` |
| Task | `T024` | `plan-summary.md` §Implementation Plan Summary |
| Authority | `run-codify` `references/session-close.md` step 12, session-completion branch | quoted in the task's Planned Task Context |
| Write surface | `quality-backlog.md`, `roadmap.md`, `handoff-context.md`, the promoted chunk, plus only files a drained line's own remedy names | task dispatch |
| Verification profile | `parallel` | task dispatch; no user decision open |
| Assumption impact | `required:new-surface` | task dispatch |
| Base / pre-wave head | `main @ 72e3a1c` / `191e676` | handoff §Current Status |

## Census (measured on `5433970`, before any edit)

| Bucket | Lines | Count |
|---|---|---|
| literal `→ routed:` | 120, 121, 122, 123, 125, 129, 130, 131, 132, 133 | 10 |
| prose `owning route:` with no literal marker | 141, 142 | 2 |
| legacy `kept:` | 124, 126, 127, 128, 134, 135 | 6 |
| no terminal disposition | 136, 137, 143, 148–158 (11), 159–167 (9), 171, 201, 209, 221, 227, 237, 244 | 30 |
| **total** | | **48** |

The State line at `quality-backlog.md:98` read **46 open**; the section held **48** open
`- [ ]` lines. Both figures are now stated in the same line, and the close reads 37 open — all routed — with 0 undisposed.

## Drained — 11 lines, commit `54001b6`

| Line | Item | What changed |
|---|---|---|
| 132 | `qa-plan.md` TC-003/TC-113/TC-114 `Lifecycle` clause | added the exit-code-0 / no-traceback witness, matching TC-110's convention |
| 143 | `evidence/cr2-wave-surface-and-scan.txt` §5 | retitled "Full hermetic suite (checkpoint run)"; §6 keeps the coverage figure |
| 148 | `utils.py:29` input annotation | widened to `Mapping[str, str]`; unused `CIMultiDictProxy` import removed |
| 150 | `tests/` lint hygiene | five `F` findings deleted (two F401 imports, one F401 name, two F841 locals) and a minimal `[tool.ruff.lint] select = ["F"]` pinned; `uvx ruff check` → All checks passed |
| 151 | `buster_ripper_legacy.py:1` | four-line provenance header below the shebang, naming the tracked reference copy and its scope exclusion |
| 163 | `tests/test_transport_parity.py:160` `_relay` | `P:6` → `P:4` by a `_CASES` table and a `(self, mocker, raw_proxy, case)` signature; scan reports no issue |
| 165 | `tests/test_upstream_answer_relay.py` + `tests/test_dump.py` | both normalized to one trailing newline |
| 166 | `quality-backlog.md:105` | restated as "counts the body as it arrived less the billing nonce, the one residue the invariant requires", naming the no-stats arm |
| 167 | `tests/test_count_tokens_parity.py:95-103` | docstring now names the digest class, the `/v1/messages` posture class, and this class's own over-reach direction (a stricter `_read_body`) |
| 209 | `iterations/104-01-forward.md:24,45` | "three `P:5`" → "two `P:5` and one `P:4`" |
| 237 | `tests/test_serve_script_argv.py:260` + `tests/qa/test_qap_ms001.py` | README citations re-pointed after `T023` (`:153`→`:164`, `:154`→`:165`, `:188`→`:199`, `:192`→`:203`; `:87` unchanged) |

## Routed — 37 lines

| Destination | Lines | Why big |
|---|---|---|
| RM-001 (extended) | 121, 122, 123, 142, 159, 160 | route-handler restructure / unenumerated hard violations; design decision under `062 R-A` and `DC-S10-C4-r1`, `chat.py` at 300/300 |
| RM-002 | 120 | `dump.py` module split; crosses module boundaries |
| RM-004 (extended) | 130, 141 | dependency bound plus the undeclared direct `multidict` import |
| RM-006 | 125 | harness/method belongs on its own branch (`codify-scope` rule 2) |
| RM-007 (extended) | 129, 161 | private-attribute pinning; the cure is the real-upstream tier, not a re-pin |
| RM-008 (extended) | 149 | mutable `CIMultiDict` return at a relay boundary — an interface decision |
| RM-017 (new) | 124, 126, 127, 128, 134, 135, 137, 162, 164 | test-module shape/duplication; every remedy is a suite reorganization |
| RM-018 (new) | 171, 221, 227 | QA-gate size (hard stop 700), harness size, and a call-site coverage gap |
| RM-019 (new) | 131, 133, 152, 153, 154, 155, 156, 157, 158, 201, 244 | registry/invariant record hygiene — the registry lane owns it |
| RM-020 (new) | 136 | additive behavior (verbatim bytes when no transform applies) — a behavior change, not a drain |

Each routed line carries `→ routed:` naming
`.sessions/codify/roadmap/2026-10-18-message-dump-bulk-flush.md` and its `RM-*` handle. RM-001,
RM-004, RM-007 and RM-008 were extended rather than duplicated; RM-017…RM-020 were minted in the
session roadmap, the promotion register and the promoted chunk.

## Verification

- `git diff --stat 191e676..54001b6 -- buster_ripper/ tests/ pyproject.toml buster_ripper_legacy.py`
  → 14 files, +39/−25; all test bytes, one annotation, one comment header, one lint pin.
- Hermetic suite on the resulting head `793a5f0` (production/test bytes unchanged since `54001b6`):
  **513 passed**, coverage **96.71%** (`Required test coverage of 80% reached`).
- `uv run pytest tests/qa -q --no-cov` → **38 passed** in 87.97 s against the live
  `http://hyperipper.local:30000` (`QAP-MS001` 17, `QAP-MS002` 15, `QAP-MS004` 6).
- `uvx ruff@0.16.10 check buster_ripper/ tests/ buster_ripper_legacy.py` → **All checks passed!**
- Full transcript: `evidence/t024-wave-suite.txt`.

## Concerns

- **`[ruff]` selection is F only.** The row's own remedy names `select = ["E","F"]` with a
  house line-length. E is not selected because the tracked legacy reference copy opts out of E501 and
  no line-length policy exists; selecting E would turn a hygiene drain into a formatting decision.
  The F gate is the half that was actually red.
- **`response_headers` annotation is wider than the call sites.** `Mapping[str, str]` now admits the
  `dict`, `CIMultiDict` and `CIMultiDictProxy` spellings the tests already pass; no type checker runs,
  so this is a declaration-precision change only.
- **The mutable-return row (149) was not drained.** Wrapping the return in `CIMultiDictProxy` changes the
  declared interface and belongs to RM-008's owner, as the row itself says.
- **`evidence/INDEX.md` dispositions were not touched.** The 20 empty cells are the finalize gate's own
  step 16; this wave only added one evidence row with an empty Disposition, as the index requires.

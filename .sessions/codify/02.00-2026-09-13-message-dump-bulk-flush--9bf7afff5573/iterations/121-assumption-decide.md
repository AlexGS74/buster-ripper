# Iteration 121 — Closure-packet routing (`assumption-decide`)

Session `02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573`, worktree
`/Users/alexhubitski/Projects/buster-ripper`, base `main @ 72e3a1c`, head
`96264eb2d91ea77e5f90a60f09f0caee675361fc`.

Source report: `runs/121-closing-check-codex.txt` — the step-13 stability round's
codex run over the full session diff (`191e676..HEAD`). The run completed; only the
finalize dispatch after it was interrupted. It relays **one** finding, reproduced
verbatim below; no second finding exists in the transcript.

`Review round:` **N/A** — this is the local step-13 stability round, not a hosted
pull-request round. The pull request is not open, so there are no hosted rounds to
count and the late-round rule does not apply.

## The finding, re-measured on the bytes

> [P2] Make the new Ruff gate executable in the dev environment —
> `pyproject.toml:25-28`. `ruff` is not included in the `dev` dependency group
> (and `.venv/bin/ruff` is absent), while this config is described as a gate.
> Consequently, a fresh `uv sync --group dev` cannot run the intended check, and no
> repository automation invokes it, so the new F-rule policy is unenforced. Add Ruff
> to the managed dev toolchain and invoke it from the project's verification
> entrypoint.

Every factual clause holds; the *characterisation* of the block as a gate is the one
claim that does not:

| Clause | Measurement at `96264eb` |
|---|---|
| `ruff` is not in `[dependency-groups] dev` | **confirmed** — `pyproject.toml:41-48` holds the five pytest-family packages only |
| `.venv/bin/ruff` is absent | **confirmed** — `ls .venv/bin/ruff` → `No such file or directory`; `grep -c 'name = "ruff"' uv.lock` → `0` |
| a fresh `uv sync --group dev` cannot run the check | **confirmed** — `uv run --frozen ruff --version` → `error: Failed to spawn: 'ruff'` |
| no repository automation invokes it | **confirmed** — no `.github/workflows`, no Makefile/justfile/`scripts/`, and `[tool.pytest.ini_options] addopts` carries only `--cov=buster_ripper --cov-report=term-missing --cov-fail-under=80` |
| the config "is described as a gate" | **confirmed** — `pyproject.toml:26` reads "Minimal gate: the F family only." |
| the policy itself is executable | **confirmed by the recorded invocation** — `uvx --offline ruff@0.16.10 check buster_ripper/ tests/ buster_ripper_legacy.py` → `All checks passed!` (`evidence/t024-wave-suite.txt:34-35`; the same command is `119-01-yield.md:18`) |

So the refutation half is real: the comment's word "gate" names an enforcement
that does not exist. The proposal half — "add Ruff to the managed dev toolchain and
invoke it from the project's verification entrypoint" — is a separate claim, and the
settled record already assigns it elsewhere.

## Required outcome (re-derived, not adopted)

The packet's required outcome is *"the F-rule policy is executable from the declared
dev environment and enforced by the project's verification entrypoint."* Settled
authority does not support it as this session's obligation:

- `119-01-yield.md:49`, the accepted record of `T024`'s own drain: *"**Gap resolved:** accepted as the one reasonable reading — the gate that was red is the F family, and **a lint-gate decision is not part of this drain. A future lint-policy pass owns adding E.**"*
- `079-codex-review.md:225` and `080-assumption-decide.md:133`, the settled classification of this row: *"no lint gate runs on this tree (deps are aiohttp+click only), so it is **hygiene, not behavior**"* / *"hygiene, no gate runs it"*.
- `QB:lint/ruff-f-select-findings`'s own remedy: *"delete the six unused imports + the two dead locals, and either pin a minimal `[tool.ruff]` … or drop the ad-hoc noqa markers — one decision, one line in `pyproject.toml`."* The row is titled **"[lint hygiene, no gate]"**. The pin is landed; the five findings are deleted.

Re-derived required outcome: **the record this session publishes about the pinned
F-family policy is true of its bytes — the policy is a recorded ad-hoc-run policy, and
no text in the repository claims an automated gate that does not run.** That is met by
one inert-wording repair. The enforcement itself is a developer-tooling decision the
settled record assigns to a future lint-policy pass, not to this drain.

## Reach First

**Reach: inert wording.** The only false byte is `pyproject.toml:26`'s comment word
"gate". Nothing reads it: no program consumes it, and the repository's only
lint-bearing automation is the recorded ad-hoc invocation
`uvx --offline ruff@0.16.10 check buster_ripper/ tests/ buster_ripper_legacy.py`
(`119-01-yield.md:18`, `evidence/t024-wave-suite.txt:34`). The policy is real and
runnable; only its label overstates.

The enforcement half is not a defect in anything this session delivers. It is a
developer-tooling feature, and `codify-phase-close` rule 2 sends it to the roadmap
("This is the default for shared tooling: test runners, build and CI scripts,
developer tools"). It is also exactly the "one more small thing" the close-out rule
forbids adding at session close, and it would change what `uv run pytest` means for
every later pass.

## Deterministic Closure Gate

`CLOSURE: accepted` — not one of the gate's four outcomes, because Reach First
settled the finding before closure ran. Recorded in the gate's packet shape for the
dispatcher:

```text
CLOSURE: accepted
Finding: 121-P2 (runs/121-closing-check-codex.txt)
Required outcome: the record published about pyproject.toml:25-28's F-family policy is true of its bytes — an ad-hoc-runnable pinned policy, not an automated gate
Authority: 119-01-yield.md:49 (the drain's accepted reading: "a lint-gate decision is not part of this drain. A future lint-policy pass owns adding E"); 079-codex-review.md:225 and 080-assumption-decide.md:133 ("hygiene, no gate runs it"); QB:lint/ruff-f-select-findings's own remedy ("pin a minimal [tool.ruff] … one decision, one line in pyproject.toml", row titled "[lint hygiene, no gate]"); external-review.md ("Never revert work traceable to an approved decision because a review said so")
Admissible candidates: (a) reword pyproject.toml:26's comment to name a policy rather than a gate; (b) add ruff to [dependency-groups] dev, refresh uv.lock and invoke uv run ruff check from the verification entrypoint; (c) add a pytest lint plugin (e.g. pytest-ruff) and an addopts entry so uv run pytest enforces select = ["F"]; (d) revert [tool.ruff.lint]
Rejected candidates: (b) -> shared tooling (test runner / dev toolchain) that codify-phase-close rule 2 sends to the roadmap, and it changes what "the suite is green" means for every later pass; the drain's accepted record already assigns the lint-gate decision to a future lint-policy pass, so the change broadens scope beyond every task's outcome; (c) -> the same shared-tooling rejection, and it changes the meaning of uv run pytest for every future pass while adding no correctness to anything this session delivers; (d) -> external-review.md: "Never revert work traceable to an approved decision because a review said so" — T024's pin is the row's own named remedy and is accepted-local; (e) leave it -> preserves the verified overclaim
Selected closure: (a) the comment reword, recorded as the doc line QB:lint/ruff-config-gate-wording with the enforcement proposal routed as RM-021
Risk shape: unchanged — no production, test, gate, config or lockfile byte moves now; the pinned policy and the five F deletions stand exactly as accepted
Deferred additive work: RM-021 (lint posture: whether ruff joins the managed dev toolchain and the verification entrypoint enforces it), owner the lint-policy pass, trigger the next lint-policy pass or any change to [tool.ruff.lint]/addopts/the dev group
Owning route: none in this session — doc line (quality-backlog.md) plus roadmap row RM-021; no Forward dispatch, no assumption seam
Question required: no
```

## Invariant cross-reference

- Invariants scanned: 0 session (`invariants.md` absent) + 8 global registry entries (`invariants/`; all `Status: planned`, none `active` or `wip`).
- Findings checked: 1.
- Neutral: 1 — no registry record's `Statement:` addresses lint tooling or the dev dependency group. Two permanent `Claim:` records are evidence inputs only: `CL:pyproject.toml/runtime-dependency-set` (its `Statement` is about `[project] dependencies`; `[dependency-groups] dev` appears only in its `Source` line) and `CL:pyproject.toml/coverage-gate-config` (the coverage gate, untouched by this finding). Neither is classified as an aligned invariant and neither authorizes or forbids the proposal.
- Aligned: 0. Contradictions: 0. Ambiguous: 0. Overrides matched: 0.

Noted, not routed: `CL:pyproject.toml/pytest-collection-scoping`'s `Statement` ("no `testpaths` and no `norecursedirs`") is stale against `pyproject.toml:20-23`'s `norecursedirs`, and its own `Invalidates-On` fired. This finding names neither that handle nor that byte, so per the routing boundary it is recorded here as an observation for the registry lane, not opened as a task or a second row.

## Disposition

**`accepted`** — Reach First settled it; nothing below applies. The finding gets:

1. one doc line in `quality-backlog.md` §Refactoring (`QB:lint/ruff-config-gate-wording`, closed `[x]`, no task opens);
2. one line under `handoff-context.md` §Accepted risks (finding id, evidence path, Reach verdict and citation);
3. one roadmap row `RM-021` for the enforcement proposal, because `codify-phase-close` rule 2 sends out-of-scope shared tooling to the roadmap, and `119-01-yield.md:49` already names a future lint-policy pass as its owner.

No task is allocated or amended, and no Deterministic Closure Ledger entry is written: the ledger is reserved for `absorbed` findings, and `121-P2` is `accepted`. `T024`'s accepted-local record repairs are not reopened; its drain note and the evidence transcript are accurate as written. No user question: the settled record ranks the enforcement decision to its own future owner, and no answer from the user would change what this close needs.

## Durable writes

| File | Write |
|---|---|
| `quality-backlog.md` | §Refactoring State line closed count 30 → 31; new `[x]` inert-wording line `QB:lint/ruff-config-gate-wording` under a closing-check subsection |
| `handoff-context.md` | §Accepted risks line for `121-P2` |
| `roadmap.md` (session) | §Promotion Register row and `### RM-021` section, `promoted-onward` |
| `.sessions/codify/roadmap/2026-10-18-message-dump-bulk-flush.md` | `RM-021` row in `## Roadmap items` |
| `chronicle.md` | one routing entry |

## Orchestrator Prompt (verbatim)

> T023

## Result: IMPLEMENTED — `README.md`'s serve-script section now documents `--preserved-thinking` and `PRESERVED_THINKING` (default `1`), the ON-by-default posture (`enable_thinking=true`, `clear_thinking=false`), the `PRESERVED_THINKING=0` off-switch, and names `FK-011 · preserved-thinking serve default` as the decision; `git diff --stat` is `README.md` only.

## Increment Brief

- Task `T023 · document the preserved-thinking serve default`: add the flag and its env var to the README's serve-script section at the standard the TLS and dump-format entries already meet.
- Anchors touched: none of the Anchor Map's code rows. The doc surface is the existing `README.md` §Serve script options; `serve_buster_ripper.sh:30`/`:42`, `main.py:107-110` and `routes/messages.py:46-47`/`:73-80` are read-only witnesses, not edits.
- Will NOT create: a decision record, a README subsection, a script/header entry, a test, a config global, a roadmap row or an invariant.
- Reuse evidence: the two documented peers this section already carries — `SKIP_TLS_VERIFY`/`--skip-tls-verify` (`README.md:89-92`) and `--dump-format` (`README.md:169-194`) — supply the comment-block-plus-prose shape; no new artifact.
- Inventions: none.

## Done-when re-derived

| Clause | Result | Evidence |
|---|---|---|
| 1. Names the flag and variable, states ON by default (`enable_thinking=true`, `clear_thinking=false`), shows the off-switch | holds | `README.md:94-97` (comment + `PRESERVED_THINKING=0 bash serve_buster_ripper.sh`); `README.md:100-104` (the `1` default, the binary's own off default). Behavior read-back: `serve_buster_ripper.sh:30` `PRESERVED_THINKING=${PRESERVED_THINKING:-1}`, `:42` `== "1"` appends `--preserved-thinking`; `routes/messages.py:46-47` injects, `:79-80` `setdefault("enable_thinking", True)` / `setdefault("clear_thinking", False)`; `config.py:50` `PRESERVED_THINKING: bool = False` |
| 2. Names `FK-011 · preserved-thinking serve default` and creates no second record | holds | `README.md:103`; `git diff --stat` = `README.md` only; `decisions.md` untouched; the task is the projection, not a new record |
| 3. `git diff --stat` shows `README.md` only; script sha256 `c58256fb…`; `tests/test_serve_script_argv.py` untouched | holds | `git diff --stat` → `README.md \| 11 +++++++++++`; `shasum -a 256 serve_buster_ripper.sh` → `c58256fbec793c520720c9e86fee8740653939275b508f50b212ab60bd0908af`; the test file is absent from the diff |
| Wrong witness: an opt-in presentation or a missing off-switch fails clause 1 | holds | the section opens "Preserved Thinking is the serve script's default, not an opt-in" and carries `PRESERVED_THINKING=0 bash serve_buster_ripper.sh` |

Validation read-back ran against `serve_buster_ripper.sh:30`/`:42`, `buster_ripper/main.py:107-110` and the injection site — `evidence/t023-preserved-thinking-readme.txt`. Doc-standard comparison: the new entry mirrors `--skip-tls-verify`'s four-line comment (default, flag, off-switch) and `--dump-format`'s prose paragraph (signature, default, example).

## What changed

| Surface | State | Evidence |
|---|---|---|
| `README.md:94-97` | comment block: ON by default, `enable_thinking=true` / `clear_thinking=false` in `chat_template_kwargs` unless the client set them, off-switch command | `sed -n '69,112p' README.md` |
| `README.md:100-104` | prose: serve-script default vs binary default, `1` default, `FK-011 · preserved-thinking serve default` (Option A) | same read-back |

No other file moved: no script, test, config, route, plan-contract or registry byte. The 11 added lines are the whole diff.

## Failure-path evidence

No behavior path changed — documentation only. The failure path this section documents is the script's own off-arm: `PRESERVED_THINKING=0` is not `"1"`, so line 42 suppresses the flag and the binary keeps `config.PRESERVED_THINKING = False` (`config.py:50`); the section states that outcome rather than implying the flag is mandatory. A client that already set `chat_template_kwargs` keeps its own values (`messages.py:79-80` uses `setdefault`), which the comment states.

## Capacity gate

`Capacity gate: not applicable - no additive production surface.` The touch set is `README.md` (289 → 300 lines), a documentation surface, not Python. No capacity scan is owed and no function metric moves. `serve_buster_ripper.sh` and `tests/test_serve_script_argv.py` are byte-identical (sha256 above).

## Backlog pickup

Matching row not absorbed: `QB:cli-surface/undocumented-flags-and-stats-db` (`quality-backlog.md:24`) names this touch-set's README CLI surface, but it is already routed to `roadmap.md` §RM-005 as a surface task in its own right and spans `main.py`, `config.py`, `image.py` semantics and the architecture tree. T023 documents one of its five flags; the row stays open with a dated PARTIAL note recording that `--preserved-thinking` is now covered and the remaining set reads four.

New row written: `QB:docs/readme-line-citations-after-t023` under `## Refactoring` — the 11-line insertion at `README.md:94-104` shifts the numeric README citations in two test modules' docstrings; clause 3 of the done-when forbids touching those files here. No other open line names the touch-set.

## Durable signal

- Implemented `FK-011 · preserved-thinking serve default` **Option A**: no code, script or test byte moved; the default-on posture is now a documented, recorded choice.
- No decision changed or reversed. `roadmap.md` §RM-005 is untouched per the registrar's instruction (its owner re-reads the remaining set); the quality-backlog row records the partial delivery.
- Task-relevant invariant entries: none changed; `invariants/` is untouched.

## Concerns

1. The README states the binary's own default is off (`config.PRESERVED_THINKING: bool = False`), while the serve script's default is on. That asymmetry is the settled posture, but a reader who invokes `buster-ripper` directly sees the opposite default — stated explicitly so it is not a surprise.
2. `README.md` is now 300 lines; no test reads it, so no test coverage follows the change.
3. The 11-line insertion shifts every numeric README citation at or after old line 94. `tests/test_serve_script_argv.py:260` (`README:192`) and `tests/qa/test_qap_ms001.py`'s docstring citations (`:140`, `:156`, `:232`–`:234`, `:402`, `:435`) now point 11 lines early; `:87` still points correctly because it precedes the insertion. Nothing asserts on a line number — the gate's reader contract is the `README_LINE_PATTERN` regex — so this is record drift, not behavior. Done-when clause 3 forbids touching those files here, so the drift is recorded as `QB:docs/readme-line-citations-after-t023` and left to the next touch of those docstrings.

## Open

- Acceptance and `resolved` are Yield's, not this pass's; the task stays `implemented`.
- The `parallel` verification profile's compact route owes no assumption checker for this task; Forward Red and the direct Yield read-back are the orchestrator's dispatches.

## Next

Yield reads the diff (`README.md` only) against the three done-when clauses; no production or test verification is owed.

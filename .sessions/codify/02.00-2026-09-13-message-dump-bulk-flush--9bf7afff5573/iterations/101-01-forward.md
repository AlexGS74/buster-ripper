## Orchestrator Prompt (verbatim)

> T022

## Result: DRAINED — `quality-backlog.md` now carries `## Correctness` / `## Refactoring`; the one open in-scope Correctness line that was still true is fixed, every other open line is routed to a named owner or closed with its reason.

## Increment Brief

- Task `T022 · step-6a boundary drain`: sort the backlog, triage each open line against current bytes, drain the in-scope Correctness set.
- Anchors touched: none of the plan's Anchor Map rows. The surviving fix is test bytes in `tests/test_stream_relay_commit.py`; `aiohttp.web.StreamResponse.write` is a dependency method, not a plan referent.
- Will NOT create: a `write_faults=` knob on `tests/upstream_stub.py` (the row's prescribed mechanism — it cannot reach `response.write`), a production helper, a new QA gate, or a registry/invariant record.
- Reuse evidence: `tests/upstream_stub.py::StubStream` (the existing upstream double), `tests/conftest.py::raw_proxy` (the existing real-socket fixture), `tests/test_stream_relay_commit.py`'s own `POST_COMMIT_FAULTS` table.
- Inventions: none.

## What changed

| Line | Subject | Section | Disposition | Evidence |
|---|---|---|---|---|
| 18 | `dump.py:86-89` residual fmt-mixing | Correctness | `[x]` accepted risk, reach `unreachable` | `dump_turn` still re-captures `state.fmt`; `_set_dump_format` still parse-time-only; §RM-003 keeps the binding |
| 40 | `config.py:65` + README CLI surface | Correctness | `[ ]` `→ routed:` §RM-005 | `main.py:74-75` help still claims a cache `image.py:172-173` says has no reader; `STATS_DB` still has no writer |
| 51 | `chat.py:174-188` post-commit second head | Correctness | `[ ]` `→ routed:` roadmap §Chat route | `_relay_chat_stream` still prepares before the relay; 067-01-yield ruling stands |
| 52 | `test_stream_relay_commit.py:48-54` write-position hole | Correctness | `[x]` **fixed in this wave** | new `write-position-os-error` case; fail-first mutation reds exactly it (2 failed, 12 passed); restored 14 passed; scan clean |
| 72 | `utils.py:27` Content-Encoding relay fidelity | Correctness | `[ ]` `→ routed:` §RM-008 | duplicate-name half landed at T017; content-coding half is a parity deviation |
| 78 | `chat.py:56` + `messages.py:242` overflow text | Correctness | `[ ]` `→ routed:` collector lane | both sites byte-unchanged; the intent question is unranked by settled authority |
| 105 | `messages.py:30`/`chat.py:31`/`count_tokens.py:23` LookupError | Correctness | `[x]` accepted risk, reach `unreachable` | `web_request.py:686-694` raises only the three named classes; claims row 416 `HOLD` |
| 109 | `messages.py:162-166` charset doc divergence | Correctness | `[x]` accepted risk, reach `unreachable` | zero `charset=` arguments under `buster_ripper/`; `verdicts/claim-425.md` |
| 121 | `test_qap_ms001.py:460-469` inert witness arm | Correctness | `[ ]` `→ routed:` `forward-qa` | no change to `assert_documented_contents` reddens it |
| 149 | `test_qap_ms001.py:641-694` tautological arm | Correctness | `[ ]` `→ routed:` `forward-qa` | `changed_lines` is still `diff_changed_lines(production_diff_text(...))` |
| 161 | `S22.md` ambient-proxy premise | Correctness | `[ ]` `→ routed:` collector lane | CE grounded in CPython `urllib`, not a member claim |
| 168 | `test_qap_ms001.py:196` substring precondition | Correctness | `[ ]` `→ routed:` `forward-qa` | glued control-line artifact still passes it |

The 42 remaining open lines are `## Refactoring` (LNCP/file-shape, test scaffolding, lint, record accuracy); each keeps its recorded owner and trigger, and every one was re-asked the Correctness question — none can make the program behave wrongly. Per-line reasons for all 50 open lines (8 Correctness, 42 Refactoring) are in [quality-backlog.md](../quality-backlog.md); every routed line carries the literal `→ routed:` marker beside its named destination, so the file's state is checkable without reading prose.

## Dependency claims checked

| Claim | What the plan/row implied | What the dependency does | Where read | Action |
|---|---|---|---|---|
| a `write_faults=` knob on `StubStream` reaches `response.write` | the row's remedy | `StubStream` is the *upstream* answer; `response.write` is the downstream `StreamResponse` — no path connects them | `buster_ripper/routes/messages.py:222`, `tests/upstream_stub.py:64-75` | **corrected**: the witness arms `aiohttp.web.StreamResponse.write` itself; recorded in the backlog line and here |
| `StreamResponse.write` awaits `self._payload_writer.write(data)` | implied by patching it as the write position | true | installed aiohttp 3.13.3 `web_response.py` (`async def write`, `await self._payload_writer.write(data)`) | none |
| the relay writes through `StreamResponse.write` | claims rows 67/149 already assert it | true | `messages.py:222`; claims rows 67/149 | none |
| patching the class intercepts the handler's instance call | not stated | true; proven by execution — the new case fails on the mutated build and passes on the real one | `tests/test_stream_relay_commit.py` run (below) | none |

`assumptions/claims.md` reach lookup: rows 67 and 149 are `shared` but assert the *existing* relay behavior; this increment adds a witness for it and corrects no claim, so no shared-claim escalation applies.

## Failure-path evidence

| Path | Outcome implemented | Evidence |
|---|---|---|
| `OSError` from `response.write` after the head is committed | log one `Post-commit relay error (OSError)` line, abandon, one head, chunked terminator, no 502 | `test_one_head_one_terminal_and_no_error_turn[write-position-os-error]`, `test_the_log_names_position_not_cause[write-position-os-error]` |
| `ClientPayloadError` mid-iteration | same terminal, logged | `[client-payload-error]` |
| upstream payload `AttributeError` in `_note_message_start` | same terminal, logged | `[upstream-payload-attribute-error]` |
| bare `asyncio.TimeoutError` mid-iteration | same terminal, logged | `[bare-asyncio-timeout-error]` |
| `ClientConnectionResetError` mid-iteration | same terminal, logged | `[client-connection-reset]` |
| `asyncio.CancelledError` | propagates untouched, `write_eof` not awaited | `test_cancellation_still_propagates` |
| pre-commit `ClientError` | fresh 502 (unchanged) | `test_a_pre_commit_client_error_still_answers_502` |
| clean relay, no fault | chunks verbatim, usage recorded, terminator | `test_chunks_are_relayed_verbatim_and_usage_recorded` |

## Capacity gate

`Capacity gate: not applicable - no additive production surface.` Touch set is `tests/test_stream_relay_commit.py` only (175 → 193 lines, Green band). `codify scan path tests/test_stream_relay_commit.py --git --detailed` → `No refactoring issues found!`

## Backlog pickup

Absorbed: `QB:test-stream-relay/write-position-injection` (line 52) — its own trigger is this module. No other open line names this file. The row's cross-reference to `QB:test-routes/duplicate-upstream-stub` no longer holds, because the chosen mechanism adds no knob to the shared stub; recorded in the line.

## Test results

- Narrow, after the edit: `uv run pytest tests/test_stream_relay_commit.py -q --no-cov` → **14 passed**.
- Fail-first: mutation wrapping the iteration but not `write()` (`relayed = []` outside the `try`, writes after it) → **2 failed, 12 passed** — exactly the two `write-position-os-error` ids. Reverted byte-exact (`cmp` clean).
- LNCP: `codify scan path tests/test_stream_relay_commit.py --git --detailed` → `No refactoring issues found!`

## Concerns

1. The backlog row's remedy was wrong and is corrected in place: a `write_faults=` knob on `StubStream` cannot reach `response.write`. The witness therefore arms `web.StreamResponse.write` class-wide for that one case.
2. `tests/qa/test_qap_ms001.py` is at **963** raw lines (red FL-LNCP, 700 hard stop) and carries three routed `forward-qa` rows. Out of implementation Forward's lane; routed, not touched.
3. Assumption profile is `required:new-surface`. The surviving fix's surface is test-only and its claims are about aiohttp's `StreamResponse.write` (read from installed source and proven by execution). No blind dispatch was performed by this agent; the orchestrator owns any assumption-gate re-entry.

## Next

Yield reviews the drain: the file's own state (both headings, 8 routed / 11 closed in Correctness, 42 open in Refactoring) plus the fixed witness.

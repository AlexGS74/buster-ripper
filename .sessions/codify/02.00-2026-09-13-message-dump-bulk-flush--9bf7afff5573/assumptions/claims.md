# Assumptions — claims ledger (extraction run 1, 2026-09-19)

Session: `02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573`. Plan: `../plan-summary.md` (approved, implementable, round 18; scoped replan 022 applied). Tasks `T001`–`T006`, milestones `MS-001`–`MS-004`.

## Run basis

- **First extraction.** No prior `assumptions/claims.md` existed and there is nothing to migrate (the ledger postdates `Check`/`Routing`/`Permanent` by construction). Legacy `handoff-context.md` carries **no** `## User Choices Resolved` records — `decisions.md` is the complete canonical source — so no `MIGRATION-REQUIRED` return.
- **Permanent-evidence lookups performed, both empty:** project registry `invariants/index.md` `## Claim Index` has zero rows and `## Composition Index` has zero rows. So `i = 0` here is a fact of the empty registry, not a skipped lookup; every row below is `new`, and a promotable checker verdict may create the first `CL:` handles.
- **Package/dependency binding:** every task lands in the single package `buster_ripper` (`pyproject.toml`); runtime deps `aiohttp` + `click`; dev group pytest 9.0.2 / pytest-asyncio / pytest-aiohttp / pytest-mock. Installed versions confirmed in `.venv`: aiohttp **3.13.3**, click **8.3.1**, pytest **9.0.2** (Python 3.13.8). `pkg:` subjects carry these versions.
- **Universe-of-discourse rule for absence rows:** `buster_ripper_legacy.py` (repo root, tracked pre-refactor monolith, referenced by nothing under `buster_ripper/`) is **excluded** — every package-wide absence says "under `buster_ripper/`" or "under `buster_ripper/` or `tests/`", so a checker scanning the package tree cannot be confused by the legacy copy.
- Planned session invariants (`PI:T002/dump_format/*`, `PI:T006/upstream_tls/*`) are **not bound to working code** (their symbols do not exist yet), so their stay-true assumptions are extracted below rather than excluded; the four `OI:` provenance blocks and the three preserve-only `EI:` drafts are already bound to working code and are excluded as maintenance.

## Claims

| # | Claim | Subject | Tasks | Scope | Seams | Check | Bundle | State | Routing | Permanent |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `dump.py` imports `json` and `difflib` | repo:buster_ripper.dump | T002, T004 | shared | — | user-claim | B1 | HOLD | none | CL:buster_ripper.dump/import-topology |
| 2 | `pyproject.toml` declares `aiohttp` and `click` as the only runtime dependencies | repo:pyproject.toml | T002, T014 | shared | S14 | user-claim, shared, seam:S14 | B226 | HOLD | none | CL:pyproject.toml/runtime-dependency-set |
| 3 | The only write-bearing file operations in the `buster_ripper` package are `dump.py`'s three sites (`dump.py:82`, `:95`, `:105`) | repo:buster_ripper | T001, T002, T003, T004 | shared | S1 | shared, absence, seam:S1 | B3 | UNCLEAR | absorbed | — |
| 4 | `dump_turn` performs two write-bearing file operations per intercepted turn: the turn-file write and the `index.txt` append | repo:buster_ripper.dump#dump_turn | T001, T002, T003, T004 | shared | S1 | user-claim, shared, seam:S1 | B4 | HOLD | none | CL:buster_ripper.dump#dump_turn/write-count-per-turn |
| 5 | `dump.py` writes turn 0's record as `json.dumps(body, indent=2, ensure_ascii=False)` to `turn_000.json` | repo:buster_ripper.dump#dump_turn | T002, T004 | shared | S1 | shared, seam:S1 | B4 | HOLD | none | CL:buster_ripper.dump#dump_turn/turn-zero-record-form |
| 6 | `dump.py` writes each turn ≥ 1's record as the `difflib.unified_diff` of `_render_body(body)` against the previous turn's rendering to `turn_{turn:03d}.diff` | repo:buster_ripper.dump#dump_turn | T002, T004 | shared | S1 | shared, seam:S1 | B4 | HOLD | none | CL:buster_ripper.dump#dump_turn/turn-diff-record-form |
| 7 | `dump.py` appends one summary line per turn to `index.txt` — the `FULL` form on turn 0, the `+added/-removed` form on later turns | repo:buster_ripper.dump#dump_turn | T002, T004, T005 | shared | S1 | shared, seam:S1 | B4 | HOLD | none | CL:buster_ripper.dump#dump_turn/index-summary-form |
| 8 | `dump.py` advances `state.prev_text` and `state.turn` after processing each turn | repo:buster_ripper.dump#dump_turn | T001, T002 | shared | S1 | shared, seam:S1 | B56 | HOLD | none | CL:buster_ripper.dump#dump_turn/state-advance-timing |
| 9 | `_render_body` emits only `[tools]`, `[system]` and `[message i / role]` sections — every other top-level request-body key is absent from its output | repo:buster_ripper.dump#_render_body | T002, T004, T005 | shared | S1 | user-claim, shared, seam:S1 | B57 | HOLD | none | CL:buster_ripper.dump#_render_body/section-surface |
| 10 | `_render_body`'s output carries the request's conversation as one `[message i / role]` section per message | repo:buster_ripper.dump#_render_body | T002, T005 | shared | S1 | user-claim, shared, seam:S1 | B57 | HOLD | none | CL:buster_ripper.dump#_render_body/message-section-mapping |
| 11 | `session_id(body)` derives the session key from the body's system-prompt content alone | repo:buster_ripper.session#session_id | T001, T002, T003, T014 | shared | S1, S14 | user-claim, shared, seam:S1/S14 | B7 | HOLD | none | CL:buster_ripper.session#session_id/key-derivation-input |
| 12 | `dump.py`'s `SessionState` is the per-session dump state, keyed by session id in the module-level `_sessions` dict, holding `turn`/`prev_text`/`session_dir` | repo:buster_ripper.dump#SessionState | T001, T002, T003, T009, T004 | shared | S1 | shared, seam:S1 | B65 | HOLD | none | CL:buster_ripper.dump#SessionState/per-session-keying |
| 13 | `routes/messages.py`'s `handle_messages` wraps its `dump_turn` call in a swallow-all `try/except Exception` that logs and never fails the request | repo:buster_ripper.routes.messages#handle_messages | T001, T002, T003, T004 | shared | S1 | shared, seam:S1 | B8 | HOLD | none | CL:buster_ripper.routes.messages#handle_messages/dump-failure-isolation |
| 14 | The `buster_ripper` package installs no SIGINT/SIGTERM handler of its own | repo:buster_ripper | T003 | single | S2 | user-claim, absence, seam:S2 | B58 | HOLD | none | CL:buster_ripper/signal-handler-ownership |
| 15 | `buster_ripper`'s `on_cleanup` performs no dump flush during shutdown | repo:buster_ripper.server#on_cleanup | T003 | single | S2 | user-claim, absence, seam:S2 | B10 | HOLD | none | CL:buster_ripper.server#on_cleanup/shutdown-dump-flush |
| 16 | `server.py`'s `spawn_kv_cache_monitor`/`on_cleanup` pair establishes the background-task lifecycle: `create_task` at startup, an `app`-state key, cancel-and-await at cleanup | repo:buster_ripper.server#spawn_kv_cache_monitor | T003 | single | S2 | seam:S2 | B75 | HOLD | none | CL:buster_ripper.server#spawn_kv_cache_monitor/background-task-lifecycle |
| 17 | No symbol under `buster_ripper/` or `tests/` owns a message buffering, batching or flush role for the dump | repo:buster_ripper | T001, T002, T003 | shared | S1 | absence, seam:S1 | B3 | HOLD | none | CL:buster_ripper/dump-buffering-ownership |
| 18 | No periodic-timer, scheduler or tick symbol exists under `buster_ripper/` | repo:buster_ripper | T003 | single | S5 | absence, seam:S5 | B11 | UNCLEAR | absorbed | — |
| 19 | No symbol under `buster_ripper/` owns the collapsed-artifact role — no artifact filename constant, no per-session append target | repo:buster_ripper | T002 | single | S1 | absence, seam:S1 | B3 | DOES-NOT-HOLD | absorbed | CL:buster_ripper/collapsed-artifact-ownership |
| 20 | Nothing under `buster_ripper/` or `tests/` reads `index.txt`, `turn_000.json` or a `turn_*.diff` | repo:buster_ripper | T002, T005 | shared | S1 | absence, seam:S1 | B59 | DOES-NOT-HOLD | absorbed | CL:buster_ripper/dump-file-readers |
| 21 | No symbol in the `buster_ripper` package selects a dump record format — no config global and no CLI flag carries that role | repo:buster_ripper | T002 | single | S3 | absence, seam:S3 | B13 | HOLD | none | CL:buster_ripper/dump-format-selection |
| 22 | No numeric `config` global has a CLI flag today (`KV_POLL_INTERVAL` included) | repo:buster_ripper.main | T002, T003 | shared | S3, S5 | absence, seam:S3/S5 | B14 | DOES-NOT-HOLD | absorbed | CL:buster_ripper.main/numeric-config-cli-wiring |
| 23 | `--dump-dir` is the only dump-related CLI option in `main.py` today | repo:buster_ripper.main | T002, T005 | shared | S3 | absence, seam:S3 | B14 | HOLD | none | CL:buster_ripper.main/dump-cli-surface |
| 24 | No ssl, TLS or certificate-verification symbol, option or context exists anywhere in `buster_ripper/` | repo:buster_ripper | T006 | single | S4 | absence, seam:S4 | B15 | HOLD | none | CL:buster_ripper/tls-surface |
| 25 | The `buster_ripper` package constructs exactly one `aiohttp.ClientSession` (`server.py`'s `on_startup`) | repo:buster_ripper | T006 | single | S4 | absence, seam:S4 | B143 | HOLD | none | CL:buster_ripper/client-session-construction |
| 26 | Every outbound HTTP call in the `buster_ripper` package goes through `app["session"]` | repo:buster_ripper | T006 | single | S4 | absence, seam:S4 | B143 | HOLD | none | CL:buster_ripper/outbound-call-routing |
| 27 | `pyproject.toml` omits `buster_ripper/routes/*`, `main.py` and `server.py` from coverage and gates the suite at 80% (`--cov-fail-under=80` in `addopts`) | repo:pyproject.toml | T002, T003, T004 | shared | — | shared | B16 | HOLD | none | CL:pyproject.toml/coverage-gate-config |
| 28 | `pyproject.toml`'s pytest ini configures no collection scoping — no `testpaths` and no `norecursedirs` | repo:pyproject.toml | T002 | single | — | absence | B16 | HOLD | none | CL:pyproject.toml/pytest-collection-scoping |
| 29 | `pyproject.toml` defines a `buster-ripper` console-script entry point binding to `buster_ripper.main:main` | repo:pyproject.toml | T003, T006 | shared | — | shared | B16 | HOLD | none | CL:pyproject.toml/console-script-entry-point |
| 30 | `tests/` contains no dump test module today | repo:tests | T004 | single | — | absence | B17 | HOLD | none | — |
| 31 | The existing integration suite substitutes a mock for the upstream session rather than using a real network client | repo:tests | T002, T004 | shared | — | shared | B17 | HOLD | none | CL:tests/upstream-mocking |
| 32 | `serve_buster_ripper.sh` passes CLI options to the binary only through its `EXTRA_FLAGS` variable and forwards no positional arguments | repo:serve_buster_ripper.sh | T005, T006 | shared | S6 | shared, seam:S6 | B18 | DOES-NOT-HOLD | absorbed | CL:serve_buster_ripper.sh/option-forwarding |
| 33 | `README.md`'s dump section documents the serve script (`DUMP_DIR=… bash serve_buster_ripper.sh`) as its run path | repo:README.md | T005, T006, T011 | shared | S6 | shared, seam:S6 | B152 | HOLD | none | CL:README.md/documented-run-path |
| 34 | `buster_ripper`'s modules read `config` attributes at call time rather than binding them at import, so values assigned before serving are visible everywhere | repo:buster_ripper.config | T002, T003, T006 | shared | S3, S4, S5 | shared, seam:S3/S4/S5 | B41 | DOES-NOT-HOLD | absorbed | CL:buster_ripper.config/config-read-timing |
| 35 | `web.run_app` handles SIGINT/SIGTERM with a graceful shutdown that runs the app's cleanup hooks before the process exits | pkg:aiohttp@3.13.3 | T003, T010.001 | shared | S2 | shared, seam:S2 | B42 | HOLD | none | CL:aiohttp/signal-shutdown |
| 36 | An exception raised inside an app cleanup hook other than aiohttp's own `GracefulExit` propagates out of `run_app` | pkg:aiohttp@3.13.3 | T003, T010 | shared | S2 | shared, seam:S2 | B42 | HOLD | none | CL:aiohttp/cleanup-hook-error-propagation |
| 37 | aiohttp's `ClientSession.__init__` accepts `connector=` and has no `ssl` parameter — `ssl` is a `TCPConnector.__init__` parameter and a per-request parameter | pkg:aiohttp@3.13.3 | T006 | single | S4 | seam:S4 | B144 | HOLD | none | CL:aiohttp/ssl-parameter-topology |
| 38 | aiohttp's `ClientSession` requires its connector to be bound to the running event loop and raises otherwise | pkg:aiohttp@3.13.3 | T006 | single | S4 | seam:S4 | B22 | DOES-NOT-HOLD | absorbed | CL:aiohttp/connector-loop-binding |
| 39 | aiohttp's `TCPConnector` verifies the upstream certificate with its default `ssl=True` and disables verification with `ssl=False` | pkg:aiohttp@3.13.3 | T006 | single | S4 | seam:S4 | B144 | HOLD | none | CL:aiohttp/ssl-verification-polarity |
| 40 | `click` invokes an option's `callback` during CLI parsing, before the command function body runs | pkg:click@8.3.1 | T002, T003, T006 | shared | S3, S4, S5 | shared, seam:S3/S4/S5 | B145 | HOLD | none | CL:click/option-callback-timing |
| 41 | `click.Choice` rejects values outside its choices list | pkg:click@8.3.1 | T002, T003 | shared | S3, S5 | shared, seam:S3/S5 | B23 | HOLD | none | CL:click/choice-validation |
| 42 | pytest's `norecursedirs` ini option replaces the default directory-exclusion list rather than extending it | pkg:pytest@9.0.2 | T002 | single | — | native | B44 | HOLD | none | CL:pytest/norecursedirs-semantics |
| 43 | asyncio provides no built-in periodic-timer/scheduler facility — a periodic loop is hand-built around `asyncio.sleep` | lang:python | T003 | single | S5 | native, seam:S5 | B25 | HOLD | none | CL:python/asyncio-periodic-timer |
| 44 | `Path.open("a")` appends to the file, creating it if it does not exist | lang:python | T002, T004 | shared | S1 | shared, native, seam:S1 | B45 | HOLD | none | CL:python/path-open-append-semantics |

## Bundles (checker fan-out — one agent per bundle)

| Bundle | Rows | Subject read |
|---|---|---|
| B1 | 1 | `dump.py` header/imports |
| B2 | 2 | `pyproject.toml` `[project]` |
| B3 | 3, 17, 19 | one write-side scan of `buster_ripper/` + `tests/` |
| B4 | 4, 5, 6, 7 | `dump.py:61-109` (the documented writer) |
| B5 | 8, 12 | `dump.py` state handling (`SessionState`, `:108-109`) |
| B6 | 9, 10 | `dump.py:27-58` (`_render_body`) |
| B7 | 11 | `session.py:93-98` |
| B8 | 13 | `routes/messages.py:55-59` |
| B9 | 14 | signal-handling absence scan of the package |
| B10 | 15, 16 | `server.py:25-34`, `:60-72` |
| B11 | 18 | timer/scheduler absence scan of the package |
| B12 | 20 | read-side scan for dump-file readers |
| B13 | 21 | format/encoding-selector absence scan |
| B14 | 22, 23 | `main.py` option list + `config.py` numeric globals |
| B15 | 24, 25, 26 | package-wide aiohttp-client scan |
| B16 | 27, 28, 29 | `pyproject.toml` (coverage, pytest ini, scripts) |
| B17 | 30, 31 | `tests/` listing + `test_routes_integration.py` upstream mock |
| B18 | 32 | `serve_buster_ripper.sh` |
| B19 | 33 | `README.md` dump section |
| B20 | 34 | `config.py` docstring + call-site read pattern |
| B21 | 35, 36 | installed aiohttp `web.py`/`web_runner.py` |
| B22 | 37, 38, 39 | installed aiohttp `client.py`/`connector.py` |
| B23 | 40, 41 | installed click parameter/Choice handling |
| B24 | 42 | installed pytest `_pytest/main.py` `norecursedirs` registration |
| B25 | 43 | asyncio module surface |
| B26 | 44 | pathlib/io docs |

## User-claim provenance mapping (canonical records ↔ rows)

| Canonical record / claim | Rows |
|---|---|
| `US-001` claim 1 (no own SIGINT/SIGTERM handler; no dump flush during shutdown) | 14, 15 |
| `US-001` claim 2 (dump writes once per intercepted request) | 4 |
| `FK-002` claim 1 (a preserved layout cannot deliver one write per bulk) | 4 |
| `US-004` claim 2 (`_render_body` is a lossy projection) | 9 |
| `US-006` claim 1 (the conversation is reconstructible from the artifact — records reading) | 10 |
| `US-006` claim 3 (tool definitions/system prompt constant within an artifact — session-keying reading) | 11 |
| `US-007` claim 2 (encodings implementable on the installed dependency set) | 1, 2 |
| `US-008` claim 1 (a human-friendly element addable on the installed dependency set) | 1, 2 |

## Claims collected but NOT rowed — routes recorded, nothing silent

- **`US-001` claim 3** ("seek term" = SIGTERM) and **`FK-002` claim 3** ("flash format" = "flush format") — linguistic interpretations of the user's own words, not library/repo propositions. Both are discharged by the user's round-18 approval of the plan whose Topic, `G-002` and `VP-T003-shutdown-flush` name SIGTERM verbatim (and by the settled session vocabulary for flush/format). No blind-checkable proposition remains; no arbiter can witness a typo's meaning. Their canonical `Verification` fields are left untouched (still `assumptions PENDING`) for the orchestrator to close as user-ratified.
- **`FK-002` claim 2, `US-002` claims 1–2** (SSD-wear rationale: fewer writes reduce wear; the payload is materially harmful; a database avoids per-file writes while remaining write-heavy) — external systems claims, not source-checkable here. The session already holds grounded research (`iterations/007-collect.md`: wear is bytes × WAF, write-call count is not the wear quantity; SQLite is not a wear win; payload wear negligible versus TBW ratings) and the plan's own arithmetic does **not** rest on them (`G-008`(b): M1/M2 decisive, M3 the real axis and unmeasurable in-repo, M4 explicitly not the proof). Route: the recorded 007 research disposition plus the `MS-002` comparison record; canonical rows stay `assumptions PENDING` for the orchestrator.
- **`US-003` claim 1** (current format preserved while the bulk-flush change lands) — realized by settled shape S-1; arbiter: `VP-T002-documented-format-preserved` (`MS-001`). **`US-003` claim 2** ("small tui" meaning) — moot: `FK-007` ships no tool.
- **`US-004` claim 1** (the trade-off depends on the definition of "full") — absorbed by `F-6`'s explicit definition in the plan.
- **`US-006` claim 2** ("not full of it" meaning) — absorbed by the settled two-collapsed-format set (`US-007`/`FK-007`).
- **`US-006` claim 3's empirical half** (0 of 137 real-dump diffs touch `[tools]`/`[system]`) — measured provenance on the unmounted real dump; the repo-checkable half (session keying) is row 11.
- **`US-007` claim 1** (encoding switchable without altering the one-artifact/one-write contract) — realized by T002's dispatch; arbiter: `VP-T002-write-count-reduced` + `VP-T002-collapsed-artifact` (`MS-002`). **`US-007` claim 3** (byte multipliers ≈2.3× / 1.047×) — carried provenance, not re-measured (real dump unmounted); M4 is explicitly not the proof.
- **`US-008` claim 2** (the `FK-006` presentation already contained a human-facing addition) — provenance of a superseded presentation; `FK-007` repaired the set and the pick is made.
- **`US-009` claim 1** (nothing running locally on 30000/30001) — host runtime state, stale by nature; the operative decision (remote upstream) is the user's own `US-009` normative choice. **`US-009` claim 2** — already routed `yield-phase:MS-001 PENDING`. **`FK-008` claims 1–2** — already routed `yield-phase:MS-004 PENDING`. No new rows; the recorded arbiters stand.
- **`T006` route D authoring facts** (`cryptography`/`trustme` absent; `openssl` present; fixture committed rather than generated at test time) — host/authoring-time state recorded as verified in the task text; Forward meets it at write time. No checker row.

## Freshness basis

No permanent `Claim:` record was reused (the registry `## Claim Index` is empty), so no freshness comparison was owed. `pkg:` subjects pin the versions actually installed in `.venv` and confirmed this run: aiohttp 3.13.3, click 8.3.1, pytest 9.0.2 — these are the identities the Speeckle freshness rules key on, and a dependency upgrade fires their recheck independent of repo edits.

---

# Run 2 — scoped re-extract (2026-09-19, assumption-decide dispatch)

## Run 2 basis

- **Dispatch:** `handoff-context.md` §Composition Routing (assumption-decide, 2026-09-19). Five composition verdicts returned `MALFORMED` (`compositions/seam-S{1,3,4,5,6}.md`; `seam-S2.md` is `SAFE` and is not a finding). Root cause uniform: compositions ran while member claims carried `UNCLEAR`/`DOES-NOT-HOLD` verdicts. The atomic routing pass (`DC-claim-{3,18,19,22,32,38}-r1`) resolved the findings but left evidentiary States untouched, so the six members are **re-extracted as corrected claims whose wording binds the recorded resolutions**, to be blind-verified fresh, and the seams recomposed. Six rows, numbered **45–50** in the dispatch's own order (claim-3→45, claim-18→46, claim-19→47, claim-22→48, claim-32→49, claim-38→50). Run-1 rows **stay evidence history** — including their `absorbed` routings and the four published `DOES-NOT-HOLD` handles — and are not edited by this run.
- **Everything else inherited:** the other 38 rows are unchanged (no plan text change on their tasks, no task started or created, no phase exercised their subjects — the interim passes were read-only checks). Per §3d they inherit their on-file verdicts: session-ledger match, `verdicts/claim-{n}.md` on disk, State/Routing/Permanent preserved byte-for-byte. `i = 38`.
- **No canonical-record edits owed:** none of the six re-extracted claims is user-claim-linked (verified against the run-1 provenance map — rows 1, 2, 4, 9, 10, 11, 14, 15 only), so `decisions.md` is untouched this run.
- **Registry lookup for the six (§3e):** the registry `## Claim Index` now holds 40 rows published from this session's run-1 verdicts. Four carry `DOES-NOT-HOLD` on the **falsified run-1 wordings** — `CL:buster_ripper/collapsed-artifact-ownership` (old claim-19), `CL:buster_ripper.main/numeric-config-cli-wiring` (old claim-22), `CL:serve_buster_ripper.sh/option-forwarding` (old claim-32), `CL:aiohttp/connector-loop-binding` (old claim-38); claims 3 and 18 (`UNCLEAR`) are unpromotable and hold no handles. **No registry statement asserts any of the six corrected propositions** — the corrected wordings are materially different claims — so no record answers a run-2 row; each `Permanent` stays `—`, and a promotable fresh verdict may update those four handles in place (the promotion pass owns that). Adjacent-record check: `CL:buster_ripper/dump-buffering-ownership` (claim-17, HOLDS) does **not** answer row 46 — different universe (`buster_ripper/` + `tests/` vs `buster_ripper/`) and different proposition (buffering/batching/flush roles generally vs the periodic flush role specifically).
- **State note:** run-1 rows for these six existed with materially changed claim/evidence/extraction fingerprints → §3d precedence gives State `recheck`, Routing `unverified` for rows 45–50.
- **Subject note (row 48):** run-1 claim-22 was typed `repo:buster_ripper.main`; the corrected inventory is a property of the wiring relation between `main.py`'s option list and `config.py`'s numeric globals, so the run-2 row is typed `rel:buster_ripper.main→buster_ripper.config` (fires on either end or the mechanism changing — the correct freshness semantics for a flag↔global inventory).

## Run 2 claims

| # | Claim | Subject | Tasks | Scope | Seams | Check | Bundle | State | Routing | Permanent |
|---|---|---|---|---|---|---|---|---|---|---|
| 45 | The only write-bearing content-write file operations the `buster_ripper` package performs on dump artifacts under `DUMP_DIR` are `dump.py`'s three sites — `dump.py:82` (`turn_000.json`), `:95` (`turn_NNN.diff`) and `:105-106` (the `index.txt` append). Content-write means `write_text`/`write_bytes`/`open` for write-or-append plus `f.write`; directory creation (`mkdir`, `dump.py:72` / `main.py:90`) is excluded by definition; SQLite persistence (`session.py:35-46`, `:59-71`) is not a dump-artifact content write | repo:buster_ripper | T001, T002, T003, T004 | shared | S1 | shared, absence, seam:S1 | B27 | HOLD | none | CL:buster_ripper/dump-artifact-write-sites |
| 46 | No symbol under `buster_ripper/` owns a periodic flush role for the dump — no symbol buffers dump records for later writing or flushes a pending dump buffer on a timer | repo:buster_ripper | T003 | single | S5 | absence, seam:S5 | B28 | HOLD | none | CL:buster_ripper/periodic-flush-ownership |
| 47 | No symbol under `buster_ripper/` owns the collapsed-artifact role — no filename constant or per-session append target for a collapsed-format dump artifact (`<format>.log`, written once per flush) exists. The documented layout's `index.txt` append (`dump.py:104-106`) is the existing per-session append target and is not this role | repo:buster_ripper | T002 | single | S1 | absence, seam:S1 | B27 | HOLD | none | CL:buster_ripper/collapsed-artifact-ownership |
| 48 | The numeric `config` globals wired to CLI flags in `buster_ripper/main.py` are `MAX_MODEL_LEN`, `COMPACT_TOKEN_RATIO`, `COMPACT_LATENCY_MS`, `COMPACT_KV_RATIO`, `EVAL_MAX_TOKENS`, `EVAL_THINKING_BUDGET`, `IMAGE_CACHE_MAX_SIZE`, `IMAGE_CACHE_TTL`; the numeric globals without a CLI flag are `KV_POLL_INTERVAL`, `COMPACT_NUDGE_RATIO` and `MIN_COMPLETION_TOKENS` | rel:buster_ripper.main→buster_ripper.config | T002, T003, T014 | shared | S3, S5, S14 | shared, absence, seam:S3/S5/S14 | B227 | DOES-NOT-HOLD | unverified | CL:rel:buster_ripper.main→buster_ripper.config/numeric-config-cli-wiring |
| 49 | `serve_buster_ripper.sh` passes CLI options to the binary through two channels — the `exec` line's always-on `--upstream`/`--host`/`--port` (serve_buster_ripper.sh:36-40) and the conditional `EXTRA_FLAGS` flags (:28-32) — and forwards no positional arguments | repo:serve_buster_ripper.sh | T005, T006 | shared | S6 | shared, seam:S6 | B30 | HOLD | none | CL:serve_buster_ripper.sh/option-forwarding |
| 50 | aiohttp's `TCPConnector.__init__` resolves `loop = loop or asyncio.get_running_loop()` (connector.py:313), so constructing a connector with no running event loop raises `RuntimeError`; a `ClientSession` constructed with `connector=c` and no explicit `loop=` adopts `c._loop` as its own loop (client.py:314-316), so its `connector._loop is not loop` guard (client.py:378-379) is vacuous on that default path | pkg:aiohttp@3.13.3 | T006 | single | S4 | seam:S4 | B144 | HOLD | none | CL:aiohttp/connector-loop-binding |

## Run 2 bundles (fresh keys — run-1 keys never reused)

| Bundle | Rows | Subject read |
|---|---|---|
| B27 | 45, 47 | one dump-write-side sweep of `buster_ripper/` (the package write-site enumeration and the collapsed-role absence are settled by the same read of `dump.py` + the package sweep) |
| B28 | 46 | periodic-flush-role absence sweep of `buster_ripper/` (`dump.py`, `compaction.py`, `server.py` — `compaction.poll_kv_cache` stays in the record as the periodic metrics poll, not a flush owner) |
| B29 | 48 | `main.py` option list + `config.py` numeric globals (positive inventory, reading-independent) |
| B30 | 49 | `serve_buster_ripper.sh` (two-channel inventory + no-positional clause) |
| B31 | 50 | installed aiohttp `client.py:314-316, :373-381` + `connector.py:292-313, :342` (loop resolution, adoption, guard) |

## Run 2 freshness basis

No run-2 row was answered from the registry or the session ledger — all six are fresh checks by dispatch. The four `DOES-NOT-HOLD` handles are evidence history of the falsified wordings, retained for lineage; their recorded fingerprints (e.g. `sha256:243b2143…` dump.py, `sha256:b4794796…` main.py, `sha256:7ce41fc1…` aiohttp/client.py, `sha256:bd3db604…` aiohttp/connector.py) were recorded by the run-1 checkers at 2026-09-19T14:59:49Z and later, and no subject change has occurred since (working tree unmodified between the run-1 checks and this run). `pkg:aiohttp@3.13.3` identity unchanged (verified in `.venv` this run). The corrected claims' own freshness starts at their fresh verdicts.

---

# Run 3 — adhoc extraction, scoped to T003.001 / T007 / T008 (2026-09-22, MS-002 boundary)

## Run 3 basis

- **Dispatch:** three remediation tasks created at the MS-002 boundary after the yield-phase rejection and codex review (`plan-summary.md` §Implementation Plan Summary; `handoff-context.md` §Planned Task Context). Insertions — no diff against the ledger can see them — so all three get fresh extraction. `T001`–`T003` landed and were yield-accepted (git Checkpoints 60–67); the assumption gate closed at 50 claims checked / 6 seams composed SAFE (Checkpoint 57); `T004`–`T006` and `T005`'s script/README clauses are still pending (`serve_buster_ripper.sh` carries no `DUMP_FORMAT` mapping — verified by grep this run).
- **No user-claim linkage:** all three tasks stem from yield/codex findings, not `US-*` records; no `decisions.md` edits owed.
- **Registry lookups (§3e), both performed:** the `## Claim Index` now holds 44 `CL:` rows (run-2 verdicts promoted; the four corrected handles updated in place — e.g. `CL:aiohttp/connector-loop-binding` carries the corrected mechanism, and `CL:rel:buster_ripper.main→buster_ripper.config/numeric-config-cli-wiring` is the corrected inventory's new handle). **No record matches any of the 17 new propositions** (parse-boundary decode, exception-hierarchy siblings, per-request timeout override, ContentTypeError family, stream dispatch/relay) — all dispatch fresh, Permanent `—`.
- **Phase-exercised recheck set (6 rows, edited in place per §3d "retain the row and re-dispatch", fresh bundle keys):** MS-001/MS-002 ran the code against the libraries these claims describe — **34** (config call-time reads: every trigger read injected CLI values), **35/36** (aiohttp signal shutdown + cleanup-hook propagation: the QAP-MS002 gates delivered real SIGINT/SIGTERM and the wrapped shutdown flush ran, `tests/qa/harness.py` + `test_qap_ms002.py`), **40** (click callback timing: `--dump-format`/`--dump-flush-*` assigned globals via callbacks and the format took effect per-format), **42** (norecursedirs replaces-defaults: bare hermetic runs stayed green after `tests/qa/` existed), **44** (`Path.open("a")`: flush appends ran under the write-count VPs). Their registry handles are retained in Permanent — a promotable fresh verdict updates them in place. Source freshness does not waive the exercised check.
- **Not exercised → inherited:** **41** (Choice rejection — no invalid value was ever passed), **43** (asyncio absence — not behaviorally testable), **37/39/50** (T006 pending), and every repo row whose named aspect the phases preserved (`4`–`13`, `16`, `24`–`27`, `29`, `31`–`33`, `49`) — MS-001/MS-002's VPs behaviorally confirmed the documented-writer forms, the state-advance timing and the swallow-all.
- **Consumed-by-design authorizations — annotated history, deliberately NOT re-dispatched (nothing silent):** `15` (on_cleanup no dump flush), `17` (no buffering/flush role), `21` (no format selector), `23` (`--dump-dir` only dump option), `28` (no norecursedirs/testpaths), `30` (no dump test module), `45`–`48` (write-site census, periodic-flush-role absence, collapsed-artifact-role absence, numeric-CLI inventory). Each was a new-surface authorization whose subject the landed task deliberately changed — `flush_all`, `poll_flush`, `DUMP_FORMAT`/`--dump-format`, `--dump-flush-*`, `norecursedirs`, `tests/test_dump*.py` now exist by design, and each change was adjudicated at the implementing task's yield (VP-verified). A blind recheck would return DOES-NOT-HOLD findings against accepted deliverables. Their verdicts and absorbed routings stand as evidence history.
- **Peer listing (structural, by grep):** client-body parse sites = exactly 3 (`request.json()`); session call sites = 6, of which 3 carry per-request timeouts; stream dispatch = exactly 1 site; `resp.json()` = 2 sites. Universes name `buster_ripper/` (the legacy monolith stays excluded).

## Run 3 claims

| # | Claim | Subject | Tasks | Scope | Seams | Check | Bundle | State | Routing | Permanent |
|---|---|---|---|---|---|---|---|---|---|---|
| 51 | aiohttp's `Request.json()` decodes the request body using the charset from the Content-Type header (default utf-8) before parsing, so bytes undecodable under that charset raise `UnicodeDecodeError`, not `json.JSONDecodeError` | pkg:aiohttp@3.13.3 | T003.001, T019 | single | S7, S18 | seam:S7/S18 | B97 | HOLD | none | CL:aiohttp/request-json-decoding |
| 52 | `UnicodeDecodeError` and `json.JSONDecodeError` are sibling subclasses of `ValueError` in the stdlib exception hierarchy — neither is a subclass of the other, so `except json.JSONDecodeError` cannot catch a `UnicodeDecodeError` | lang:python | T003.001 | single | S7 | seam:S7 | B33 | UNCLEAR | malformed | — |
| 53 | The client-body parse boundary under `buster_ripper/` lives at exactly three `await request.json()` sites — `messages.py:27`, `chat.py:29`, `count_tokens.py:31` | repo:buster_ripper.routes | T003.001 | single | S7 | absence, seam:S7 | — | superseded | none | — |
| 54 | `messages.py:28-29` and `chat.py:30-31` return a typed 400 (`{"error": "Invalid JSON"}`) when the client body fails JSON parsing — the precedent the fix mirrors | repo:buster_ripper.routes | T003.001 | single | S7 | seam:S7 | — | superseded | none | CL:buster_ripper.routes/client-json-parse-error-response |
| 55 | `count_tokens.py:32-33` swallows the client-body parse failure and proceeds with an empty body instead of returning a typed 4xx — the family's divergent sibling | repo:buster_ripper.routes.count_tokens#handle_count_tokens | T003.001 | single | S7 | seam:S7 | B34 | HOLD | none | CL:buster_ripper.routes.count_tokens#handle_count_tokens/client-json-parse-error-response |
| 56 | `chat.py:146` returns a typed 502 ("Invalid JSON from upstream") when the upstream response fails JSON parsing — the upstream-attributed sibling of the parse boundary | repo:buster_ripper.routes.chat#_make_chat_request | T003.001 | single | S7 | seam:S7 | B34 | HOLD | none | CL:buster_ripper.routes.chat#_make_chat_request/upstream-json-parse-error-response |
| 57 | aiohttp: a per-request `timeout=` argument to a `ClientSession` request method (`get`/`post`/`request`) replaces the session's default `ClientTimeout` for that single request | pkg:aiohttp@3.13.3 | T007, T018 | single | S8, S17 | seam:S8/S17 | B315 | HOLD | none | CL:aiohttp/per-request-timeout-override
| 58 | The shared `ClientSession` is constructed with `aiohttp.ClientTimeout(total=600)` inside `on_startup` (`server.py:58`) — the default every call without a per-request timeout inherits | repo:buster_ripper.server#on_startup | T007 | single | S8 | seam:S8 | B108 | HOLD | none | — |
| 59 | `detect_backend`'s `/v1/models` probe (`backend.py:54`) issues `session.get` with no per-request timeout, so it inherits the session default | repo:buster_ripper.backend#detect_backend | T007 | single | S8 | seam:S8 | B36 | HOLD | none | — |
| 60 | `detect_backend` falls back to `"unknown"` on any probe failure — the `except Exception` at `backend.py:78-83` logs and returns `"unknown"` | repo:buster_ripper.backend#detect_backend | T007 | single | S8 | seam:S8 | B36 | DOES-NOT-HOLD | absorbed | — |
| 61 | Exactly three outbound call sites under `buster_ripper/` pass an explicit per-request `ClientTimeout` — `compaction.py:32` (5 s, `/metrics` poll), `chat.py:122` (300 s, chat completions), `passthrough.py:26` (60 s) | repo:buster_ripper | T007 | single | S8 | absence, seam:S8 | B37 | HOLD | none | — |
| 62 | `on_startup` awaits `backend.detect_backend` (the `/v1/models` probe) during startup, before the app serves requests | repo:buster_ripper.server#on_startup | T007 | single | S8 | seam:S8 | B36 | UNCLEAR | absorbed | — |
| 63 | aiohttp's `resp.json()` raises `ContentTypeError` when the response's Content-Type is not `application/json` — e.g. `text/event-stream` (SSE) | pkg:aiohttp@3.13.3 | T008 | single | S9 | seam:S9 | B38 | DOES-NOT-HOLD | absorbed | CL:aiohttp/response-json-content-type-validation |
| 64 | `aiohttp.ContentTypeError` is a subclass of `aiohttp.ClientError` | pkg:aiohttp@3.13.3 | T008, T008.001 | shared | S9 | shared, seam:S9 | B113 | HOLD | none | CL:aiohttp/content-type-error-inheritance |
| 65 | `chat.py`'s `_make_chat_request` parses every upstream response with `await resp.json()` (`chat.py:131`) regardless of the request's `stream` flag | repo:buster_ripper.routes.chat#_make_chat_request | T008 | single | S9 | seam:S9 | — | superseded | none | CL:buster_ripper.routes.chat#_make_chat_request/upstream-response-parsing |
| 66 | `handle_messages` dispatches on `body.get("stream", False)` (`messages.py:65-68`) between a streaming and a non-streaming handler — the mirror for T008's branch | repo:buster_ripper.routes.messages#handle_messages | T008 | single | S9 | seam:S9 | B40 | MALFORMED | malformed | — |
| 67 | `messages.py`'s streaming handler relays the upstream body raw — a prepared `StreamResponse` fed by `resp.content.iter_any()` chunks written through unchanged | repo:buster_ripper.routes.messages#handle_streaming | T008, T008.001, T013 | shared | S9, S11 | shared, seam:S9/S11 | B196 | HOLD | none | CL:buster_ripper.routes.messages#handle_streaming/upstream-body-relay |

## Run 3 bundles (fresh keys — run-1/run-2 keys never reused)

| Bundle | Rows | Subject read |
|---|---|---|
| B32 | 51 | installed aiohttp `web_request.py` body decode / `json()` |
| B33 | 52 | stdlib exception hierarchy (`json.JSONDecodeError`, `UnicodeDecodeError` under `ValueError`) |
| B34 | 53, 54, 55, 56 | one read of the `routes/` parse-boundary family (the census grep settles all four) |
| B35 | 57 | installed aiohttp `client.py` per-request timeout merge |
| B36 | 58, 59, 60, 62 | the probe-inheritance chain read (`server.py:44-80` + `backend.py:19-87`) |
| B37 | 61 | package-wide `ClientTimeout` call-site census |
| B38 | 63, 64 | installed aiohttp `client_exceptions.py` / `client.py` `resp.json()` content-type enforcement |
| B39 | 65 | `routes/chat.py:114-149` (the bug site) |
| B40 | 66, 67 | `routes/messages.py:24-70` + `handle_streaming` (the mirror pattern, one read) |
| B41 | 34 | re-dispatch: `config.py` + call-site read pattern (exercised by T001–T003) |
| B42 | 35, 36 | re-dispatch: aiohttp `web.py`/`web_runner.py` signal/cleanup (exercised by real-signal QAP gates) |
| B43 | 40 | re-dispatch: click option-callback timing (exercised by the three new flags) |
| B44 | 42 | re-dispatch: pytest `norecursedirs` semantics (exercised by post-`tests/qa/` hermetic runs) |
| B45 | 44 | re-dispatch: `Path.open("a")` append semantics (exercised by flush appends under the write-count VPs) |

---

# Run 4 — scoped malformed re-extraction (2026-09-27)

## Run 4 basis

- Dispatch: handoff §Malformed Assumption Routing, claim-52 / S7 and claim-66 / S9 only. `verdicts/claim-52.md` returned `UNCLEAR` because “sibling” could mean direct children or separate descendant branches; `verdicts/claim-66.md` returned `MALFORMED` because its T008 mirror clause required session context. Preserve both old rows and verdicts as history with terminal `malformed` routing. The corrected facts below omit the ambiguous and plan-dependent language and are new blind-checkable inputs; no verdict is inherited from a partial reading.
- S7's exception-hierarchy ingredient is split into five atomic propositions. S9's dispatch ingredient is reduced to the production-only route assertion. No other task or seam is re-extracted.

## Run 4 claims

| # | Claim | Subject | Tasks | Scope | Seams | Check | Bundle | State | Routing | Permanent |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 68 | `UnicodeDecodeError` derives from `UnicodeError` | lang:python | T003.001, T019 | single | S7, S18 | seam:S7/S18 | B98 | HOLD | none | — |
| 69 | `UnicodeError` derives from `ValueError` | lang:python | T003.001, T019 | single | S7, S18 | seam:S7/S18 | B98 | HOLD | none | — |
| 70 | `json.JSONDecodeError` derives directly from `ValueError` | lang:python | T003.001, T019 | single | S7, S18 | seam:S7/S18 | B98 | HOLD | none | CL:python/json-decode-error-inheritance |
| 71 | `UnicodeDecodeError` is not a subclass of `json.JSONDecodeError` | lang:python | T003.001, T019 | single | S7, S18 | seam:S7/S18 | B98 | HOLD | none | — |
| 72 | `json.JSONDecodeError` is not a subclass of `UnicodeDecodeError` | lang:python | T003.001, T019 | single | S7, S18 | seam:S7/S18 | B99 | HOLD | none | — |
| 73 | `handle_messages` dispatches to `handle_streaming` or `handle_non_streaming` according to `body.get("stream", False)` | repo:buster_ripper.routes.messages#handle_messages | T008, T008.001, T013 | shared | S9 | shared, seam:S9 | B196 | HOLD | none | CL:buster_ripper.routes.messages#handle_messages/stream-dispatch |

## Run 4 bundles

| Bundle | Rows | Subject read |
|---|---|---|
| B46 | 68, 69, 70, 71 | Python exception inheritance (`UnicodeError`, `UnicodeDecodeError`, `json.JSONDecodeError`) |
| B47 | 72 | Python exception inheritance (reverse subclass relation) |
| B48 | 73 | `buster_ripper/routes/messages.py` dispatch |

---

# Run 5 — scoped atomic routing re-extraction (2026-09-27)

## Run 5 basis

- Dispatch: corrected atomic premises for terminal absorbed claims 34, 60, 62 and 63. The old rows, their verdicts, routings and permanent handles remain historical evidence; none is reopened or overwritten. Their replacement propositions are new rows requiring blind checks. No implementation or composition judgment is made here.
- Claim 34's blanket assertion is divided into the import-bound CLI default and the two existing runtime-consumer families; the planned T006 TLS reader is not asserted as existing code. Claim 60's universal failure quantifier is divided into the probe's ordinary-exception fallback and cancellation's exception hierarchy. Claim 62's startup await is separated from the conditional branches inside detection. Claim 63's categorical media-type rejection is divided into the SSE and structured-suffix cases.
- No user-claim-linked row is corrected. The pre-existing 34, 60, 62 and 63 remain terminal `absorbed`; corrected rows have no inherited verdict or permanent evidence handle. Other claims and seams retain their recorded histories.

## Run 5 claims

| # | Claim | Subject | Tasks | Scope | Seams | Check | Bundle | State | Routing | Permanent |
|---|---|---|---|---|---|---|---|---|---|---|
| 74 | The `click.option` defaults in `buster_ripper/main.py` that refer to `config` attributes are bound when `main.py` is imported | repo:buster_ripper.main | T002, T003, T006 | shared | S3, S4, S5 | shared, seam:S3/S4/S5 | B146 | HOLD | none | CL:buster_ripper.main/cli-default-binding-time |
| 75 | The dump-format dispatch under `buster_ripper/` reads `config.DUMP_FORMAT` when processing a turn | repo:buster_ripper.dump | T002 | single | S3 | seam:S3 | B50 | HOLD | none | CL:buster_ripper.dump/format-dispatch-read-timing |
| 76 | The periodic dump-flush trigger under `buster_ripper/` reads its configured interval at runtime | repo:buster_ripper | T003 | single | S5 | seam:S5 | B51 | HOLD | none | CL:buster_ripper/periodic-flush-interval-read-timing |
| 77 | `detect_backend` returns `"unknown"` when an `Exception` is raised inside its probe `try` block | repo:buster_ripper.backend#detect_backend | T007 | single | S8 | seam:S8 | B109 | HOLD | none | CL:buster_ripper.backend#detect_backend/probe-exception-fallback
| 78 | `asyncio.CancelledError` is not a subclass of `Exception` | lang:python | T007, T013 | single | S8, S11 | seam:S8/S11 | B110 | HOLD | none | CL:python/cancelled-error-inheritance |
| 79 | `on_startup` awaits `detect_backend` before the app serves requests | repo:buster_ripper.server#on_startup | T007 | single | S8 | seam:S8 | B108 | HOLD | none | CL:buster_ripper.server#on_startup/backend-detection-startup-order |
| 80 | `detect_backend` returns a CLI backend override without issuing a `/v1/models` GET | repo:buster_ripper.backend#detect_backend | T007 | single | S8 | seam:S8 | B109 | HOLD | none | CL:buster_ripper.backend#detect_backend/cli-override-probe-bypass
| 81 | `detect_backend` returns its cached backend without issuing a `/v1/models` GET | repo:buster_ripper.backend#detect_backend | T007 | single | S8 | seam:S8 | B109 | HOLD | none | CL:buster_ripper.backend#detect_backend/cached-backend-probe-bypass
| 82 | `detect_backend` issues a `/v1/models` GET when neither a CLI override nor a cached backend is available | repo:buster_ripper.backend#detect_backend | T007 | single | S8 | seam:S8 | B109 | HOLD | none | CL:buster_ripper.backend#detect_backend/uncached-backend-probe
| 83 | With its default expected JSON content type, aiohttp `ClientResponse.json()` raises `ContentTypeError` for a `text/event-stream` response | pkg:aiohttp@3.13.3 | T008, T008.001 | shared | S9 | shared, seam:S9 | B113 | HOLD | none | CL:aiohttp/sse-json-content-type-validation |
| 84 | With its default expected JSON content type, aiohttp `ClientResponse.json()` accepts `application/problem+json` | pkg:aiohttp@3.13.3 | T008, T008.001 | shared | S9 | shared, seam:S9 | B113 | HOLD | none | CL:aiohttp/structured-suffix-json-content-type-validation |
| 85 | The size-based dump-flush trigger under `buster_ripper/` reads its configured byte threshold at runtime | repo:buster_ripper | T003, T009 | shared | S5 | shared, seam:S5 | B93 | HOLD | none | CL:buster_ripper/size-flush-threshold-read-timing |

## Run 5 bundles

| Bundle | Rows | Subject read |
|---|---|---|
| B49 | 74 | `main.py` option decorators and import-time binding |
| B50 | 75 | dump-format dispatch read |
| B51 | 76, 85 | dump-flush interval and threshold read sites |
| B52 | 77, 80, 81, 82 | `backend.py` exception and priority branches |
| B53 | 78 | Python cancellation exception hierarchy |
| B54 | 79 | `server.py` startup ordering |
| B55 | 83, 84 | installed aiohttp response JSON content-type validation |

---

# Run 6 — scoped realized-chain re-extraction (2026-09-27)

Dispatch: handoff §Composition Routing 2026-09-27, malformed S1/S2/S3/S5. Historical authorizations 4–7, 15, 17, 21, 23, 45–48 retain their verdicts, routings and permanent handles; their pre-buffer or pre-CLI inventories do not establish current behavior. Rows 8–10, 12, 14, 16 and 20 retain their propositions but require new checks against changed source scope. No current TLS path is asserted: S4 remains deferred to the T006 boundary, including historical 25. No composition or claim verdict is made here. `US-001` claim 1's old linkage to row 15 is historical; its factual shutdown-flush premise is superseded by new row 95, not a revision to its Normative Choice. The orchestrator must reconcile its canonical verification linkage after the blind verdict.

| # | Claim | Subject | Tasks | Scope | Seams | Check | Bundle | State | Routing | Permanent |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 86 | Dump-artifact content writes under `buster_ripper/` occur only in the documented turn writer, index append, and collapsed batch append | repo:buster_ripper | T001, T002, T003, T004 | shared | S1 | shared, absence, seam:S1 | B60 | HOLD | none | — |
| 87 | `_write_documented` writes a turn file and appends one index line per turn | repo:buster_ripper.dump#_write_documented | T002, T004 | shared | S1 | shared, seam:S1 | B61 | HOLD | none | CL:buster_ripper.dump#_write_documented/documented-turn-write |
| 88 | `_write_full_turn` writes turn 0's body as indented, non-ASCII-preserving JSON to `turn_000.json` | repo:buster_ripper.dump#_write_full_turn | T002, T004 | shared | S1 | shared, seam:S1 | B62 | HOLD | none | CL:buster_ripper.dump#_write_full_turn/turn-zero-record-form |
| 89 | `_write_diff_turn` writes later turns' unified rendered-body diff to `turn_{turn:03d}.diff` | repo:buster_ripper.dump#_write_diff_turn | T002, T004 | shared | S1 | shared, seam:S1 | B63 | HOLD | none | CL:buster_ripper.dump#_write_diff_turn/turn-diff-record-form |
| 90 | `_append_index_line` appends a summary line to `index.txt` | repo:buster_ripper.dump#_append_index_line | T002, T004, T005 | shared | S1 | shared, seam:S1 | B64 | HOLD | none | CL:buster_ripper.dump#_append_index_line/index-line-append |
| 91 | `SessionState` tracks pending records and their rendered-text byte count per session | repo:buster_ripper.dump#SessionState | T001, T002, T003 | shared | S1, S5 | shared, seam:S1/S5 | B65 | DOES-NOT-HOLD | absorbed | CL:buster_ripper.dump#SessionState/pending-byte-accounting |
| 92 | `_append_record` adds a collapsed record to the session's pending list | repo:buster_ripper.dump#_append_record | T001, T002, T003, T009, T004 | shared | S1 | shared, seam:S1 | B85 | HOLD | none | CL:buster_ripper.dump#_append_record/collapsed-record-buffering |
| 93 | `_flush_state` appends a session's pending batch to its format-named artifact in one file write | repo:buster_ripper.dump#_flush_state | T001, T002, T003, T009, T010 | shared | S1, S2 | shared, seam:S1/S2 | B86 | DOES-NOT-HOLD | absorbed | CL:buster_ripper.dump#_flush_state/batch-append |
| 94 | `_flush_state` clears pending records and byte count after the append succeeds | repo:buster_ripper.dump#_flush_state | T001, T002, T003, T009, T004 | shared | S1 | shared, seam:S1 | B86 | HOLD | none | CL:buster_ripper.dump#_flush_state/pending-clear-after-append |
| 95 | `on_cleanup` invokes `_final_dump_flush` after cancelling background tasks and before closing the shared session | repo:buster_ripper.server#on_cleanup | T003, T010 | shared | S2 | shared, seam:S2 | B87 | HOLD | none | CL:buster_ripper.server#on_cleanup/shutdown-flush-order |
| 96 | `spawn_dump_flush` stores a `dump.poll_flush()` task in app state for cleanup | repo:buster_ripper.server#spawn_dump_flush | T003, T010 | shared | S2, S5 | shared, seam:S2/S5 | B69 | HOLD | none | CL:buster_ripper.server#spawn_dump_flush/dump-flush-task-lifecycle |
| 97 | `on_cleanup` cancels and awaits each background task through `_cancel_background_task` | repo:buster_ripper.server#on_cleanup | T003, T010 | shared | S2 | shared, seam:S2 | B87 | HOLD | none | CL:buster_ripper.server#on_cleanup/background-task-cancellation |
| 98 | `_resolve_format` reads `config.DUMP_FORMAT` and rejects values outside `config.DUMP_FORMATS` | repo:buster_ripper.dump#_resolve_format | T002 | single | S3 | seam:S3 | B70 | HOLD | none | CL:buster_ripper.dump#_resolve_format/format-validation |
| 99 | `main.py` offers `--dump-format` alongside `--dump-dir` | repo:buster_ripper.main | T002, T005, T006 | shared | S3 | shared, seam:S3 | B153 | HOLD | none | CL:buster_ripper.main/dump-format-option |
| 100 | `main.py` exposes `--dump-flush-interval` and `--dump-flush-bytes` as callback-backed numeric options | repo:buster_ripper.main | T002, T003, T006 | shared | S5 | shared, seam:S5 | B153 | HOLD | none | CL:buster_ripper.main/dump-flush-options |
| 101 | `poll_flush` repeatedly calls `flush_all` then awaits `asyncio.sleep(config.DUMP_FLUSH_INTERVAL)` | repo:buster_ripper.dump#poll_flush | T003 | single | S5 | seam:S5 | B72 | HOLD | none | CL:buster_ripper.dump#poll_flush/periodic-flush-loop |
| 102 | `_append_record` compares pending rendered-text bytes with `config.DUMP_FLUSH_BYTES` before triggering a flush | repo:buster_ripper.dump#_append_record | T003 | single | S1, S5 | seam:S1/S5 | B66 | DOES-NOT-HOLD | absorbed | CL:buster_ripper.dump#_append_record/byte-threshold-comparison |
| 103 | `dump_turn` selects the documented writer or one of two collapsed record builders using the resolved format | repo:buster_ripper.dump#dump_turn | T002 | single | S1, S3 | seam:S1/S3 | B56 | HOLD | none | CL:buster_ripper.dump#dump_turn/format-dispatch |
| 104 | `_final_dump_flush` catches ordinary exceptions from `dump.flush_all` | repo:buster_ripper.server#_final_dump_flush | T003, T010 | shared | S2 | shared, seam:S2 | B88 | HOLD | none | CL:buster_ripper.server#_final_dump_flush/flush-error-isolation |
| 105 | `_artifact_path` constructs a per-session `<format>.log` path | repo:buster_ripper.dump#_artifact_path | T002, T004 | single | S1 | seam:S1 | B74 | HOLD | none | CL:buster_ripper.dump#_artifact_path/collapsed-artifact-path |

Fresh checker bundles: B56 (8, 103), B57 (9, 10), B58 (14), B59 (20), B60 (86), B61 (87), B62 (88), B63 (89), B64 (90), B65 (12, 91), B66 (92, 102), B67 (93, 94), B68 (95, 97), B69 (96), B70 (98), B71 (99, 100), B72 (101), B73 (104), B74 (105), B75 (16). The old bundle keys remain evidence history, not dispatch keys for this run. Historical row 48's exhaustive numeric inventory is superseded for the affected trigger chain by 100; no claim that the old complete inventory still applies is inherited.

---

# Run 7 — scoped atomic routing re-extraction (2026-09-27)

Dispatch: absorbed claim-20, claim-91, claim-102 and affected S1/S5. Their DOES-NOT-HOLD verdicts, terminal `absorbed` routings and permanent handles remain unchanged as evidence history. The closure for claim-20 distinguishes production read ownership from existing documented-format test reads; the closures for 91/102 retain the current character-count mechanism as a source fact while T009's UTF-8 byte repair remains pending. No claim below asserts that repair has landed. The old bundle references are historical, not re-dispatches.

| # | Claim | Subject | Tasks | Scope | Seams | Check | Bundle | State | Routing | Permanent |
|---|---|---|---|---|---|---|---|---|---|---|
| 106 | No production code under `buster_ripper/` reads `index.txt`, `turn_000.json` or `turn_*.diff` | repo:buster_ripper | T002, T004, T005 | shared | S1 | shared, absence, seam:S1 | B76 | HOLD | none | CL:buster_ripper/dump-file-readers |
| 107 | `tests/test_dump.py` reads `turn_000.json`, `turn_001.diff` and `turn_002.diff` with `read_text()` | repo:tests.test_dump | T004 | single | S1 | seam:S1 | B77 | HOLD | none | CL:tests.test_dump/documented-turn-file-reads |
| 108 | `tests/test_dump.py` reads `index.txt` with `read_text()` | repo:tests.test_dump | T004 | single | S1 | seam:S1 | B77 | HOLD | none | CL:tests.test_dump/documented-index-read |
| 109 | `_append_record` increments the per-session `pending_bytes` accumulator by `len(current_text)` | repo:buster_ripper.dump#_append_record | T003, T009 | shared | S1, S5 | shared, seam:S1/S5 | B78 | HOLD | none | CL:buster_ripper.dump#_append_record/pending-unit-accumulation |
| 110 | `_append_record` calls `flush_all()` when accumulated `pending_bytes >= config.DUMP_FLUSH_BYTES` | repo:buster_ripper.dump#_append_record | T003, T009 | shared | S1, S5 | shared, seam:S1/S5 | B85 | HOLD | none | CL:buster_ripper.dump#_append_record/threshold-flush-trigger |
| 111 | For non-ASCII Python strings, `len(text)` can differ from `len(text.encode("utf-8"))` | lang:python | T009 | single | S1, S5 | seam:S1/S5 | B79 | HOLD | none | CL:python/string-utf8-length-relation |

Fresh checker bundles: B76 (106, production reader absence), B77 (107–108, documented-artifact test reads), B78 (109–110, `_append_record` accounting and comparison), B79 (111, Python character/UTF-8 byte distinction). Recorded verdicts and closures supply extraction provenance only; they do not pre-answer the fresh rows.

---

# Run 8 — scoped new-task extraction (2026-09-27)

Dispatch: inserted T009/T010 and affected S1/S2/S5. Earlier row verdicts are inherited as source propositions, not implementation acceptance for these tasks. The pending UTF-8 accounting repair is not asserted as present code; its existing current-state facts are rows 109–111. T010's failed-task cleanup is likewise pending, while row 115 names the present exception boundary. No task test/run, atomic verdict, or seam composition is performed here. Both tasks land in `buster_ripper` (`pyproject.toml`: `aiohttp`, `click`); T009's `dump.py` peers are `_render_body`, `_append_record`, `_flush_state`, `dump_turn`; T010's `server.py` peers are both spawners, `_cancel_background_task`, `_final_dump_flush`, `on_cleanup`.

| # | Claim | Subject | Tasks | Scope | Seams | Check | Bundle | State | Routing | Permanent |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 112 | Encoding a Python string with UTF-8 produces a `bytes` value whose length counts its encoded bytes | lang:python | T009 | single | S1, S5 | seam:S1/S5 | B89 | HOLD | none | CL:python/utf8-encoded-byte-length |
| 113 | Cancelling an already-completed `asyncio.Task` does not replace its stored exception with cancellation | lang:python | T010 | single | S2 | seam:S2 | B90 | HOLD | none | CL:python/completed-task-cancellation |
| 114 | Awaiting a completed `asyncio.Task` that failed with an ordinary exception raises its stored exception | lang:python | T010 | single | S2 | seam:S2 | B90 | HOLD | none | CL:python/failed-task-await-exception |
| 115 | `_cancel_background_task` catches `asyncio.CancelledError` but not ordinary exceptions raised by awaiting the task | repo:buster_ripper.server#_cancel_background_task | T010 | single | S2 | seam:S2 | B82 | HOLD | none | CL:buster_ripper.server#_cancel_background_task/exception-handling |

Fresh bundles: B80 (112, UTF-8 encoding), B81 (113–114, completed asyncio task cancellation and await), B82 (115, server cancellation helper). Existing T009-linked rows 109–111 and T010-linked cleanup rows 36/95–97/104 are inherited; no new `Mode: NEW` role is authorized by either remediation task. No relevant unlinked `assumptions PENDING` factual claim in canonical `decisions.md`; no choice linkage edited. Registry matching for fresh propositions requires independent freshness evidence before reuse; no permanent handle assigned by extraction.

---

# Run 9 — post-acceptance scoped recheck (2026-09-27)

Dispatch: accepted-local T009/T010; only S1, S2 and S5. Earlier rows 109 and 115 preserve their prior source verdicts as historical pre-repair facts; neither asserts current behavior. This extraction records the implemented mechanisms as new atomic inputs, without assigning a verdict. T009 changes the `_append_record` byte unit and mutation ordering; T010 changes the helper's ordinary-failure boundary. No new role or dependency is added. Previously checked language premises 111–114 remain on file; source properties of Python have not changed. Current implementation acceptance is not inferred from local Yield, and no seam is composed here.

| # | Claim | Subject | Tasks | Scope | Seams | Check | Bundle | State | Routing | Permanent |
|---|---|---|---|---|---|---|---|---|---|---|
| 116 | `_append_record` computes `len(current_text.encode("utf-8"))` before mutating its pending list | repo:buster_ripper.dump#_append_record | T009, T004 | single | S1, S5 | seam:S1/S5 | B83 | HOLD | none | CL:buster_ripper.dump#_append_record/encoding-before-buffer-mutation |
| 117 | `_append_record` increments `pending_bytes` by the rendered text's UTF-8 encoded length | repo:buster_ripper.dump#_append_record | T003, T009, T004 | shared | S1, S5 | shared, seam:S1/S5 | B83 | HOLD | none | CL:buster_ripper.dump#_append_record/pending-byte-accumulation |
| 118 | `_cancel_background_task` catches ordinary exceptions raised by awaiting the stored task | repo:buster_ripper.server#_cancel_background_task | T010 | single | S2 | seam:S2 | B84 | HOLD | none | CL:buster_ripper.server#_cancel_background_task/ordinary-error-isolation |
| 119 | `_cancel_background_task` re-raises `CancelledError` when its current cleanup task is cancelling | repo:buster_ripper.server#_cancel_background_task | T010, T010.001 | shared | S2 | shared, seam:S2 | B84 | HOLD | none | CL:buster_ripper.server#_cancel_background_task/cleanup-cancellation-propagation |

Fresh bundles B83 (116–117, same `_append_record` mechanism) and B84 (118–119, same cancellation helper). Exercised prior HOLD rows 92–95, 97, 104, 110, 112–114 re-dispatch as B85 (92, 110), B86 (93–94), B87 (95, 97), B88 (104), B89 (112), B90 (113–114); the old bundle keys remain historical. No canonical user claim newly linked and no unrelated seam re-extracted. Dropped from the **current seam membership**, not the ledger: 109 (character-count premise) and 115 (uncaught ordinary task failure premise), because each describes the pre-repair implementation.

---

# Run 10 — absorbed claim-93 scoped re-extraction (2026-09-27)

Dispatch: claim-93's unconditional proposition is terminal `DOES-NOT-HOLD`/`absorbed` evidence history and is not re-dispatched. Its verdict describes the `documented` guard; the T004 closure requests conditional collapsed-format append and documented no-buffer source facts for S1/S2. These rows describe the existing source under a fixed selected format, not a runtime-switching or migration guarantee. Neither a claim nor a seam is checked or composed here.

| # | Claim | Subject | Tasks | Scope | Seams | Check | Bundle | State | Routing | Permanent |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 120 | For a non-`documented` session state with pending records, `_flush_state` appends their joined batch to `<format>.log` with one `f.write` | repo:buster_ripper.dump#_flush_state | T001, T002, T003, T004, T010 | shared | S1, S2 | shared, seam:S1/S2 | B91 | HOLD | none | CL:buster_ripper.dump#_flush_state/conditional-batch-append |
| 121 | `_flush_state` returns without an artifact append when the session state's format is `documented` | repo:buster_ripper.dump#_flush_state | T002, T003, T004, T010 | shared | S1, S2 | shared, seam:S1/S2 | B91 | HOLD | none | CL:buster_ripper.dump#_flush_state/documented-artifact-guard |
| 122 | `dump_turn` does not add pending records for turns processed in `documented` format | repo:buster_ripper.dump#dump_turn | T001, T002, T004 | shared | S1, S2 | shared, seam:S1/S2 | B92 | HOLD | none | CL:buster_ripper.dump#dump_turn/documented-pending-buffer |

Rows 120–121 share the `_flush_state` guard/append source read; row 122 is a separate `dump_turn` subject. The fixed-format usage witness and no runtime format mutation remain T004's plan/QA arbiter, not a source claim. The previous claim-93 row, its verdict, routing and permanent handle remain unchanged.

---

# Run 11 — S5 threshold-read refresh (2026-09-27)

Dispatch: malformed S5 after accepted-local T009. Retain claim 85's atomic runtime-read proposition, now also linked to T009, and re-dispatch it as `recheck`/`unverified` in fresh solo bundle B93. Its prior verdict's incidental character-accumulation explanation and `_append_record` fingerprint predate T009; that explanation is not claim 85's proposition and cannot establish the current threshold-read source. Current UTF-8 accumulation is represented by rows 116–117, and inclusive threshold comparison by row 110. Historical character-count row 109 remains outside the current seam. No claim is checked and no seam is composed here.

---

# Run 12 — scoped T010.001 extraction (2026-09-27)

Dispatch: inserted T010.001 and affected S2. The pending repair distinguishes retained cancellation of aiohttp's runner from a fresh cancellation during the cleanup await. New rows below expose runner-to-cleanup and child-await premises without asserting that the repair is present. Existing S2 source claims retain their verdicts; row 119 also names T010.001 because its current-helper predicate is the boundary the task changes. No source proposition, seam, or runtime outcome is checked here. The task lands in `buster_ripper` (`pyproject.toml`: `aiohttp`, `click`); peers are both background-task spawners, `_cancel_background_task`, `_final_dump_flush`, and `on_cleanup`. No new role or dependency is planned.

| # | Claim | Subject | Tasks | Scope | Seams | Check | Bundle | State | Routing | Permanent |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 123 | On `GracefulExit`, aiohttp `run_app` cancels and awaits its `_run_app` task | pkg:aiohttp@3.13.3 | T010.001 | single | S2 | seam:S2 | B94 | HOLD | none | CL:aiohttp/runner-task-cancellation |
| 124 | aiohttp `_run_app` awaits runner cleanup in its own `finally` block | pkg:aiohttp@3.13.3 | T010.001 | single | S2 | seam:S2 | B94 | HOLD | none | CL:aiohttp/runner-cleanup-await |
| 125 | `asyncio.Task.cancelling()` retains the count of cancellation requests after `CancelledError` delivery unless explicitly reduced | lang:python | T010.001 | single | S2 | seam:S2 | B95 | HOLD | none | CL:python/task-cancellation-count-retention |
| 126 | Awaiting a child task that terminates cancelled raises `CancelledError` in the awaiting task | lang:python | T010.001, T020 | single | S2, S19 | seam:S2/S19 | B95 | HOLD | none | CL:python/cancelled-child-await-propagation |

The aiohttp pair shares a runner lifecycle source; the Python pair shares asyncio task cancellation semantics. The helper's `cancelling()` branch remains row 119, task slots and final flush order remain rows 96–97/95, and pending collapsed batch append remains row 120. Fresh outer-cancellation distinction is planned behavior and a Forward/QA arbiter, not a claim about existing code. No rows dropped; no canonical `assumptions PENDING` factual choice newly linked.

---

# Run 13 — accepted-local T010.001 scoped recheck (2026-09-27)

Dispatch: accepted-local T010.001 and S2 only. The helper's prior absolute cancellation predicate in row 119 remains historical checked evidence, not a current S2 member. The current helper introduces a pre-await cancellation-count snapshot and a count-increase condition on re-raising child cancellation. These are separate source premises for blind checks, not a claim that signal shutdown succeeds. The aiohttp and Python premises 123–126, both spawned slots, ordinary-failure isolation and final flush order remain inherited; no new role or dependency is added. No source claim is checked and no seam is composed here.

| # | Claim | Subject | Tasks | Scope | Seams | Check | Bundle | State | Routing | Permanent |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 127 | `_cancel_background_task` snapshots the cleanup task's cancellation count before cancelling and awaiting the stored background task | repo:buster_ripper.server#_cancel_background_task | T010.001 | single | S2 | seam:S2 | B96 | HOLD | none | CL:buster_ripper.server#_cancel_background_task/cancellation-count-snapshot |
| 128 | `_cancel_background_task` re-raises a child's `CancelledError` when the cleanup task's cancellation count increased during the await | repo:buster_ripper.server#_cancel_background_task | T010.001 | single | S2 | seam:S2 | B96 | HOLD | none | CL:buster_ripper.server#_cancel_background_task/cancellation-count-increase-propagation |

The two rows share one helper-source read. Row 119's prior verdict is retained as provenance but no longer supplies the current cancellation branch; no rows dropped and no newly linked canonical `assumptions PENDING` factual choice.

---

# Run 14 — accepted-local T003.001 / S7 scoped recheck (2026-09-27)

The exercised aiohttp decode premise (51) and Python hierarchy premises (68–72) return to blind checks with fresh bundles. Prior route rows 53–56 remain historical: their line pins and JSON-only exception mappings predate the accepted-local change. The current client parse sites, their separate exception mappings, count_tokens delegation, and upstream attribution are new source premises below. No suite outcome or response contract is asserted as a claim; no claim is checked or seam composed here.

| # | Claim | Subject | Tasks | Scope | Seams | Check | Bundle | State | Routing | Permanent |
|---|---|---|---|---|---|---|---|---|---|---|
| 129 | Under production `buster_ripper/routes/`, all client `await request.json()` parse sites belong to messages, chat, or count_tokens | repo:buster_ripper.routes | T003.001 | single | S7 | absence, seam:S7 | B100 | HOLD | none | — |
| 130 | `handle_messages` maps client `UnicodeDecodeError` from `request.json()` to a JSON 400 response | repo:buster_ripper.routes.messages#handle_messages | T003.001 | single | S7 | seam:S7 | B101 | HOLD | none | CL:buster_ripper.routes.messages#handle_messages/client-unicode-decode-error-response |
| 131 | `handle_messages` maps client `json.JSONDecodeError` from `request.json()` to a JSON 400 response | repo:buster_ripper.routes.messages#handle_messages | T003.001 | single | S7 | seam:S7 | B101 | HOLD | none | CL:buster_ripper.routes.messages#handle_messages/client-json-decode-error-response |
| 132 | `handle_chat` maps client `UnicodeDecodeError` from `request.json()` to a JSON 400 response | repo:buster_ripper.routes.chat#handle_chat | T003.001 | single | S7 | seam:S7 | B102 | HOLD | none | CL:buster_ripper.routes.chat#handle_chat/client-unicode-decode-error-response |
| 133 | `handle_chat` maps client `json.JSONDecodeError` from `request.json()` to a JSON 400 response | repo:buster_ripper.routes.chat#handle_chat | T003.001 | single | S7 | seam:S7 | B102 | HOLD | none | CL:buster_ripper.routes.chat#handle_chat/client-json-decode-error-response |
| 134 | `handle_count_tokens` delegates client body parsing to `_read_body` before using the returned body | repo:buster_ripper.routes.count_tokens#handle_count_tokens | T003.001 | single | S7 | seam:S7 | B103 | HOLD | none | CL:buster_ripper.routes.count_tokens#handle_count_tokens/client-body-parse-delegation |
| 135 | `_read_body` maps client `json.JSONDecodeError` from `request.json()` to an empty dictionary | repo:buster_ripper.routes.count_tokens#_read_body | T003.001 | single | S7 | seam:S7 | B104 | HOLD | none | CL:buster_ripper.routes.count_tokens#_read_body/client-json-decode-error-fallback |
| 136 | `_read_body` maps client `UnicodeDecodeError` from `request.json()` to `web.HTTPBadRequest` | repo:buster_ripper.routes.count_tokens#_read_body | T003.001 | single | S7 | seam:S7 | B104 | HOLD | none | CL:buster_ripper.routes.count_tokens#_read_body/client-unicode-decode-error-raise |
| 137 | An uncaught aiohttp `web.HTTPBadRequest` from a request handler is sent as an HTTP 400 response | pkg:aiohttp@3.13.3 | T003.001, T019 | single | S7, S18 | seam:S7/S18 | B106 | HOLD | none | — |
| 138 | `_read_body` sets the raised `HTTPBadRequest` content type to `application/json` | repo:buster_ripper.routes.count_tokens#_read_body | T003.001 | single | S7 | seam:S7 | B104 | HOLD | none | CL:buster_ripper.routes.count_tokens#_read_body/unicode-decode-error-content-type |
| 139 | `_make_chat_request` maps upstream response `UnicodeDecodeError` from `resp.json()` to a JSON 502 response | repo:buster_ripper.routes.chat#_make_chat_request | T003.001 | single | S7 | seam:S7 | B105 | HOLD | none | CL:buster_ripper.routes.chat#_make_chat_request/upstream-unicode-decode-error-response |
| 140 | `_make_chat_request` maps upstream response `json.JSONDecodeError` from `resp.json()` to a JSON 502 response | repo:buster_ripper.routes.chat#_make_chat_request | T003.001 | single | S7 | seam:S7 | B105 | HOLD | none | CL:buster_ripper.routes.chat#_make_chat_request/upstream-json-decode-error-response |

Fresh bundles B97–B106 are subject-local; B98 holds four tied hierarchy rows and B99 holds the fifth. Existing source verdicts are not inherited for the phase-exercised premises. The route rows are atomic source descriptions, not proof of the typed HTTP behavior or a composition verdict.

---

# Run 15 — accepted-local T007 / S8 scoped recheck (2026-09-27)

The post-acceptance recheck concerns the conditional probe's timeout, fallback and bypass. Existing aiohttp timeout semantics (57), shared startup timeout/order (58, 79), ordinary-exception and cancellation boundaries (77–78), and priority branches (80–82) are re-dispatched with fresh subject-local bundles, not treated as proved by local Yield. Historical row 59 describes the pre-repair GET without a request timeout, and row 61's three-site census predates the newly bounded probe; both remain as evidence history, outside the current S8 membership. Rows 60 and 62 likewise retain their terminal absorbed history. No runtime outcome, milestone acceptance, or seam verdict is inferred here. The task adds no dependency or new role; its package remains `buster_ripper` (`pyproject.toml`: `aiohttp`, `click`).

| # | Claim | Subject | Tasks | Scope | Seams | Check | Bundle | State | Routing | Permanent |
|---|---|---|---|---|---|---|---|---|---|---|
| 141 | The `/v1/models` GET in `detect_backend` passes `aiohttp.ClientTimeout(total=5)` as its per-request timeout | repo:buster_ripper.backend#detect_backend | T007 | single | S8 | seam:S8 | B111 | HOLD | none | CL:buster_ripper.backend#detect_backend/probe-request-timeout
| 142 | `detect_backend` caches `"unknown"` after an ordinary exception in the probe's `try` block | repo:buster_ripper.backend#detect_backend | T007 | single | S8 | seam:S8 | B111 | HOLD | none | CL:buster_ripper.backend#detect_backend/probe-exception-cache
| 143 | A `CancelledError` escaping the probe's `try` block in `detect_backend` bypasses its cache assignment | repo:buster_ripper.backend#detect_backend | T007 | single | S8 | seam:S8 | B111 | HOLD | none | — |
| 144 | All explicit per-request `ClientTimeout` call sites under production `buster_ripper/` are in `backend.py`, `compaction.py`, `routes/chat.py`, and `routes/passthrough.py` | repo:buster_ripper | T007 | single | S8 | absence, seam:S8 | B112 | HOLD | none | — |

Fresh bundles: B107 (57, aiohttp request override), B108 (58, 79, server startup), B109 (77, 80–82, backend priority/fallback), B110 (78, Python exception ancestry), B111 (141–143, backend request and post-request settlement), B112 (144, package-wide explicit-timeout inventory). The current S8 chain substitutes 141 for 59 and 144 for 61, retaining the established shared-session and priority premises. No canonical user-claim link is newly affected; no `decisions.md` edit is owed. Source claims here remain unchecked; the orchestrator dispatches the bundles and then the seam composer.

---

# Run 16 — accepted-local T008 / S9 scoped recheck (2026-09-27)

Dispatch: accepted-local T008, SSE dispatch, byte relay and non-stream JSON. This is a source-premise delta, not an acceptance verdict or composition. The pre-repair unconditional parse premise 65 remains historical and is removed from current S9 membership. The unchanged messages peer dispatch/relay premises 73 and 67 remain inherited. Aiohttp media-type and exception premises 83, 84 and 64 were exercised by the local increment and return to blind checking in a fresh subject-local bundle; their prior verdicts are not inferred from local Yield. The chat route's new branch, stream relay, error transport, nonstream parse and response transformation are separate source facts below. No new dependency, public route or canonical user-claim linkage is added; the package remains `buster_ripper` (`pyproject.toml`: `aiohttp`, `click`). Runtime byte-exactness and nonstream regression remain Forward/Yield/QA arbiters rather than source-checker verdicts.

| # | Claim | Subject | Tasks | Scope | Seams | Check | Bundle | State | Routing | Permanent |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 145 | `_make_chat_request` selects `_relay_chat_stream` when the outgoing request body has a truthy `stream` value | repo:buster_ripper.routes.chat#_make_chat_request | T008, T008.001 | shared | S9 | shared, seam:S9 | B114 | HOLD | none | CL:buster_ripper.routes.chat#_make_chat_request/stream-dispatch |
| 146 | `_make_chat_request` calls `resp.json()` after the stream branch returns on the nonstream path | repo:buster_ripper.routes.chat#_make_chat_request | T008, T008.001 | shared | S9 | shared, seam:S9 | B114 | HOLD | none | CL:buster_ripper.routes.chat#_make_chat_request/nonstream-json-parsing |
| 147 | `_make_chat_request` passes parsed nonstream response data to `_apply_response_transforms` | repo:buster_ripper.routes.chat#_make_chat_request | T008, T008.001 | shared | S9 | shared, seam:S9 | B114 | HOLD | none | CL:buster_ripper.routes.chat#_make_chat_request/nonstream-transform-dispatch |
| 148 | `_make_chat_request` converts `aiohttp.ClientError` from the upstream path into a JSON 502 response | repo:buster_ripper.routes.chat#_make_chat_request | T008, T008.001 | shared | S9 | shared, seam:S9 | B114 | HOLD | none | CL:buster_ripper.routes.chat#_make_chat_request/upstream-client-error-response |
| 149 | `_relay_chat_stream` forwards successful upstream chunks from `resp.content.iter_any()` to `StreamResponse.write` unchanged | repo:buster_ripper.routes.chat#_relay_chat_stream | T008, T008.001 | shared | S9 | shared, seam:S9 | B115 | HOLD | none | CL:buster_ripper.routes.chat#_relay_chat_stream/upstream-chunk-relay |
| 150 | `_relay_chat_stream` prepares a `text/event-stream` response before forwarding successful upstream chunks | repo:buster_ripper.routes.chat#_relay_chat_stream | T008, T008.001 | shared | S9 | shared, seam:S9 | B115 | HOLD | none | CL:buster_ripper.routes.chat#_relay_chat_stream/sse-response-preparation |
| 151 | `_relay_chat_stream` returns upstream HTTP error status and raw body without calling `resp.json()` | repo:buster_ripper.routes.chat#_relay_chat_stream | T008, T008.001 | shared | S9 | shared, seam:S9 | B120 | HOLD | none | CL:buster_ripper.routes.chat#_relay_chat_stream/upstream-error-relay |
| 152 | `_apply_response_transforms` serializes the nonstream response as JSON after applying applicable eval-mode or non-eval transformations | repo:buster_ripper.routes.chat#_apply_response_transforms | T008, T008.001 | shared | S9 | shared, seam:S9 | B116 | HOLD | none | CL:buster_ripper.routes.chat#_apply_response_transforms/nonstream-json-serialization |
| 153 | `_relay_chat_stream` labels upstream HTTP error responses `application/json` regardless of the upstream Content-Type | repo:buster_ripper.routes.chat#_relay_chat_stream | T008.001 | single | S9 | seam:S9 | B117 | HOLD | none | CL:buster_ripper.routes.chat#_relay_chat_stream/upstream-error-content-type |
| 154 | `ClientResponse.headers` exposes the upstream response's Content-Type header value | pkg:aiohttp@3.13.3 | T008.001 | single | S9 | seam:S9, native | B121 | HOLD | none | CL:aiohttp/response-header-exposure |
| 155 | `aiohttp.web.Response` accepts an explicit Content-Type response header alongside a byte body | pkg:aiohttp@3.13.3 | T008.001 | single | S9 | seam:S9, native | B122 | HOLD | none | CL:aiohttp/web-response-explicit-content-type |
| 156 | `ClientResponse.headers.get("Content-Type")` returns `None` when the upstream response has no Content-Type header | pkg:aiohttp@3.13.3 | T008.001 | single | S9 | seam:S9 | B121 | HOLD | none | — |

Fresh bundles B113 (64, 83–84, installed aiohttp content-type/exception surface), B114 (145–148, chat request dispatch/parse/error boundary), B115 (149–151, chat stream relay), B116 (152, nonstream response transforms). The source facts are pending blind verdicts; the peer claims 73/67 are inherited. Dropped from current seam membership, not the ledger: row 65 (pre-repair unconditional JSON parse). No claims were silently discarded.

---

# Run 17 — inserted T008.001 / S9 scoped extraction (2026-09-27)

T008.001 adds an upstream-error representation boundary without changing the planned request-flag branch, successful SSE relay, nonstream JSON transforms or messages peer. Rows 153–156 are new source premises, not verdicts. Existing S9 rows 73, 67, 145–152, 64 and 83–84 remain in the chain as inherited evidence for T008; this task also consumes them structurally, without changing their propositions or dispatch state. The existing error status/body assertion 151 does not assert a media type. No `Mode: NEW` or canonical user factual claim applies. Runtime byte equality, 429 replay and suite passage belong to Forward/Yield/QA rather than source checks. No claims were dropped.

Fresh bundles: B117 (153, chat relay's present error label), B118 (154 and 156, aiohttp upstream response header access), B119 (155, aiohttp downstream response header capability). All await blind checks; the S9 chain awaits composition after their verdicts. The planned replacement of the label is not asserted as present implementation.

---

# Run 18 — accepted-local T008.001 / S9 scoped recheck (2026-09-27)

Dispatch: accepted-local T008.001 and S9 only. Row 153 retains its checked pre-repair JSON-label proposition as historical evidence, outside current S9 membership. The current raw-error path's present-header and absent-header branches are separate source premises below. Row 151's raw status/body transport and exercised aiohttp header/response premises 154–156 return to blind checks; their prior verdicts are not implementation acceptance. Unchanged stream dispatch, successful SSE relay, nonstream JSON path and messages peer retain their recorded verdicts. No new role or dependency, canonical factual choice, runtime outcome or seam verdict is asserted.

| # | Claim | Subject | Tasks | Scope | Seams | Check | Bundle | State | Routing | Permanent |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 157 | `_relay_chat_stream` copies an upstream HTTP error response's Content-Type header into the downstream response when present | repo:buster_ripper.routes.chat#_relay_chat_stream | T008.001 | single | S9 | seam:S9 | B120 | HOLD | none | CL:buster_ripper.routes.chat#_relay_chat_stream/upstream-error-content-type-copy |
| 158 | `_relay_chat_stream` passes no explicit response headers when an upstream HTTP error lacks Content-Type | repo:buster_ripper.routes.chat#_relay_chat_stream | T008.001 | single | S9 | seam:S9 | B120 | HOLD | none | CL:buster_ripper.routes.chat#_relay_chat_stream/upstream-error-absent-header-handling |

Fresh bundles B120 (151, 157–158: raw-error transport and both header branches), B121 (154, 156: aiohttp upstream header access), B122 (155: aiohttp downstream response header). The old B115/B118/B119 references are historical. No source claim is checked and no seam is composed here.

---

# Run 19 — accepted-local T005 / S6 scoped recheck (2026-09-27)

Dispatch: accepted-local T005 and S6 only. T005 landed the 022 wiring clause: the serve script's `DUMP_FORMAT` env-var mapping and the README's collapsed run-path example. The completed phase exercised S6's whole surface, so row 33 (README run path) returns to a blind check against the edited section, and row 49's pre-T005 in-row citations are replaced in current membership by rows 159–163 — its two-channel and no-positional conjuncts are re-stated atomically against the current script. Terminal history rows 32 (`absorbed`, DC-claim-32-r1) and 49 (`HOLD`) stay in the ledger as evidence; run-3's "no `DUMP_FORMAT` mapping" observation is likewise historical. T006's `SKIP_TLS_VERIFY` half of the seam is still pending — no TLS wiring is asserted as present. T005 adds no dependency, package or role (it touches no Python module; deps remain `aiohttp` + `click`), and no canonical user factual claim applies. Rows 159–164 are source premises, not verdicts; argv behavior, suite passage and milestone acceptance belong to Forward/Yield/QA downstream.

| # | Claim | Subject | Tasks | Scope | Seams | Check | Bundle | State | Routing | Permanent |
|---|---|---|---|---|---|---|---|---|---|---|
| 159 | `serve_buster_ripper.sh` initializes `DUMP_FORMAT` from the environment with an empty default | repo:serve_buster_ripper.sh | T005, T006, T011, T012 | shared | S6 | shared, seam:S6 | B168 | HOLD | none | CL:serve_buster_ripper.sh/dump-format-env-default |
| 160 | `serve_buster_ripper.sh` appends a `--dump-format` flag to `EXTRA_FLAGS` only when `DUMP_FORMAT` is non-empty | repo:serve_buster_ripper.sh | T005, T006, T011, T012 | shared | S6 | shared, seam:S6 | B168 | HOLD | none | CL:serve_buster_ripper.sh/dump-format-flag-guard |
| 161 | `serve_buster_ripper.sh` appends a `--dump-dir` flag to `EXTRA_FLAGS` only when `DUMP_DIR` is non-empty | repo:serve_buster_ripper.sh | T005, T006, T011, T012 | shared | S6 | shared, seam:S6 | B168 | HOLD | none | CL:serve_buster_ripper.sh/dump-dir-flag-guard |
| 162 | `serve_buster_ripper.sh`'s `exec` line passes the always-on `--upstream`/`--host`/`--port` options and the assembled `EXTRA_FLAGS` to the `buster-ripper` binary | repo:serve_buster_ripper.sh | T005, T006, T011, T012 | shared | S6 | shared, seam:S6 | B169 | HOLD | none | CL:serve_buster_ripper.sh/exec-line-option-assembly |
| 163 | `serve_buster_ripper.sh` forwards no positional arguments to the exec'd binary | repo:serve_buster_ripper.sh | T005, T006, T011, T012 | shared | S6 | shared, seam:S6, absence | B169 | HOLD | none | CL:serve_buster_ripper.sh/positional-argument-forwarding |
| 164 | `README.md`'s dump section shows selecting a collapsed format on the serve-script run path it documents | repo:README.md | T005, T006, T011 | shared | S6 | shared, seam:S6 | B152 | HOLD | none | CL:README.md/collapsed-format-run-path |

Fresh bundles: B123 (159–161, the serve script's guard block — env-var defaults and conditional appends), B124 (162–163, exec-line argv assembly and no-positional forwarding), B125 (33, 164, README dump-section run path). The old B19/B30 references are historical. Row 49 is dropped from current S6 membership, not from the ledger. Both evidence lookups were performed and neither answered a row without a checker: the session-ledger matches (32, 49, 33) are terminal-history or recheck-triggered (the phase edited both named sources), and the registry `CL:` handles on rows 33/49 cannot establish freshness against T005's edits, so they are re-dispatched with handles retained rather than reused. No source claim is checked and no seam is composed here.

---

# Run 20 — accepted-local T004 / S1-S2 scoped recheck (2026-09-28)

Dispatch: accepted-local T004 and its recorded seam scopes S1/S2 only. T004 was a tests-and-coverage increment — five test files changed, zero production files (diff verified in [050-01-yield]) — so no production-source row fires on byte freshness: working-tree `dump.py` `07b0b076…`, `session.py` `67a66b8a…` and `config.py` `7e1eff53…` are byte-identical to the recorded composition basis. What the completed phase did exercise is the claim-3/20/93 replay surface: the new witness classes (write-count per shipped collapsed format, documented 2N run, paired F=1-vs-2N=8 measurement, file count, artifact-level readability and turn-0, flush-session isolation, route-level dump-failure isolation) plus the independently executed 24/24 `io.open`-level replay behaviorally exercised the corrected premise rows, so those rows return to blind checks rather than inheriting their HOLDS from local Yield acceptance. The settled write-count instrument (DC-claim-3-r1 — one write-bearing operation is one `Path.open` in a write/append mode plus its `write`, on a path under `DUMP_DIR`; directory creation and the SQLite stats path sit outside the measured surface) is now implemented in `tests/test_dump_write_count.py` and enters the ledger as the new rows below. Terminal history rows 3 (`UNCLEAR`/absorbed), 20 and 93 (`DOES-NOT-HOLD`/absorbed) and their routings stay untouched; local Yield acceptance is not a verdict for any row. No claim is checked and no seam is composed here.

| # | Claim | Subject | Tasks | Scope | Seams | Check | Bundle | State | Routing | Permanent |
|---|---|---|---|---|---|---|---|---|---|---|
| 165 | `tests/test_dump_write_count.py`'s `_WriteHook` records a write-bearing operation only for `Path.open` calls whose mode string contains no `"r"` and whose target path lies under the instrumented dump root | repo:tests.test_dump_write_count#_WriteHook | T004 | single | S1 | seam:S1 | B129 | HOLD | none | CL:tests.test_dump_write_count#_WriteHook/write-recording-gate |
| 166 | `_WriteHook` measures each recorded write's payload length as its UTF-8 encoded byte count | repo:tests.test_dump_write_count#_WriteHook | T004 | single | S1 | seam:S1 | B129 | HOLD | none | CL:tests.test_dump_write_count#_WriteHook/payload-length-measurement |
| 167 | `tests/test_dump_write_count.py` runs its write-count, paired-measurement and file-count tests once per shipped collapsed format | repo:tests.test_dump_write_count | T004 | single | S1 | seam:S1 | B129 | UNCLEAR | malformed | — |
| 168 | `pathlib.Path.write_text` performs its write by opening the target through `self.open(mode="w", …)` | lang:python | T004 | single | S1 | native, seam:S1 | B130 | HOLD | none | CL:python/path-write-text-open-funnel |
| 169 | `tests/test_dump_records.py` reads the collapsed `<format>.log` artifact with `read_text()` in its readability and turn-0 tests | repo:tests.test_dump_records | T004 | single | S1 | seam:S1 | B131 | HOLD | none | CL:tests.test_dump_records/collapsed-artifact-reads |

T004-exercised members return to blind checks in fresh bundles: B126 (94, 120, 121 — one `_flush_state` source read), B127 (122 — `dump_turn` documented pending-buffer), B128 (107–108 — the documented-artifact reads in the phase-edited `tests/test_dump.py`), B129 (165–167 — the new instrument module), B130 (44, 168 — `Path.open` append and `Path.write_text` stdlib funnel semantics), B131 (169 — collapsed-artifact test reads), B132 (86, 106 — the package-wide write-census and production-reader-absence sweep), B133 (87), B134 (88), B135 (89), B136 (90) — the four documented writer units, B137 (92, 116, 117 — `_append_record` buffering and sizing), B138 (105), B139 (12), B140 (13). Rows 12, 13, 92, 94, 105, 116 and 117 name T004 in `Tasks` from this run. The re-dispatched rows' prior HOLDS and `CL:` handles are provenance, not fresh verdicts.

Not re-dispatched, with reasons (nothing dropped silently): S1/S2 members 8, 9, 10, 11, 103, 110, 111, 112, 98, 95, 97, 101, 104 were exercised by T004 only through pre-existing tests re-running green in the suite — re-running an existing suite is not phase exercise — and their subjects' production bytes are unchanged, so neither a freshness nor an exercise trigger fires. Terminal and superseded rows (3, 4–7, 15, 17, 20, 21, 23, 45–48, 91, 93, 102, 109, 115, 119 and the other recorded history) stay exactly as recorded.

Both evidence lookups were performed and neither answered a row without a checker: the session-ledger match found no existing subject for the new rows (both test modules are files T004 created; no `lang:python` row asserts the `write_text`→`Path.open` funnel), and the registry Claim Index holds no record for `repo:tests.test_dump_write_count`, `repo:tests.test_dump_records` or the funnel premise (the `CL:tests.test_dump/*` records cover only the documented reads, on rows re-dispatched here by exercise). Exercised rows cannot inherit by rule, so `i` is 0 with both lookups recorded. Source hashes were compared for freshness basis only; no proposition about changed code is decided here. A tests-only increment adds no dependency, package or role (deps remain `aiohttp` + `click`), and no canonical `assumptions PENDING` factual choice applies, so no `decisions.md` linkage is owed.

---

# Run 21 — malformed claim-167 re-extraction (2026-09-28)

Dispatch: handoff §Malformed Assumption Routing, claim-167 / S1 only. `verdicts/claim-167.md` returned `UNCLEAR` (checker class P9): "write-count tests" admitted two scopes — the four collapsed-format classes (reading (a), HOLDS) versus every write-counting test in the module including the two unparametrized classes (reading (b), DOES-NOT-HOLD). Row 167, its `UNCLEAR` State, bundle B129 and terminal `malformed` routing stay in the ledger as history; no verdict is inherited from the partial reading. The replacement rows below name the exact classes and pin the test constant to the production closed set, removing the ambiguous term entirely — per the verdict's own Invalidates-On, naming the exact classes settles the reading. No claim is checked and no seam is composed here.

| # | Claim | Subject | Tasks | Scope | Seams | Check | Bundle | State | Routing | Permanent |
|---|---|---|---|---|---|---|---|---|---|---|
| 170 | `tests/test_dump_write_count.py` parametrizes exactly `TestOneWritePerFlush`, `TestFlushCountIsF`, `TestPairedMeasurement` and `TestFileCount` with `@pytest.mark.parametrize("fmt", COLLAPSED_FORMATS)` | repo:tests.test_dump_write_count | T004 | single | S1 | seam:S1, absence | B141 | HOLD | none | CL:tests.test_dump_write_count/fmt-parametrize-scope |
| 171 | `tests/test_dump_write_count.py` binds `COLLAPSED_FORMATS` to `["change-focused", "self-contained"]`, exactly the two non-`documented` members of `config.DUMP_FORMATS` | rel:tests.test_dump_write_count#COLLAPSED_FORMATS→buster_ripper.config#DUMP_FORMATS | T004 | single | S1 | seam:S1, absence | B142 | HOLD | none | CL:rel:tests.test_dump_write_count#COLLAPSED_FORMATS→buster_ripper.config#DUMP_FORMATS/collapsed-format-set-equation |
| 172 | `tests/test_dump_write_count.py`'s `TestWriteUnit` executes one unparametrized run that hardcodes `config.DUMP_FORMAT = "change-focused"` | repo:tests.test_dump_write_count#TestWriteUnit | T004 | single | S1 | seam:S1 | B141 | HOLD | none | CL:tests.test_dump_write_count#TestWriteUnit/unparametrized-run-format-override |
| 173 | `tests/test_dump_write_count.py`'s `TestDocumentedCount` executes one unparametrized run on the `"documented"` dump format | repo:tests.test_dump_write_count#TestDocumentedCount | T004 | single | S1 | seam:S1 | B141 | HOLD | none | CL:tests.test_dump_write_count#TestDocumentedCount/unparametrized-run-format-default |

Fresh bundles (never-reuse rule; B129 stays on the terminal row): B141 (170, 172, 173 — one read of the module's class/decorator layout settles all three), B142 (171 — the equation spans the test module and `config.py`, so it cannot share the module-only bundle). Nothing from row 167's content is dropped: its scope decomposes without residue into the four-class parametrization premise (170), the test-constant↔production-set equation (171), and the two single-format run shapes (172, 173). Both evidence lookups were performed and neither answered a row without a checker: the session ledger holds no other row asserting these propositions (rows 165–169 are the module's only rows, and 167 is the terminal row being replaced), and the registry Claim Index holds no `Claim:` record for the parametrization scope, the `COLLAPSED_FORMATS` equation, or the two run shapes (its `CL:tests.test_dump_write_count#*` records cover only the `_WriteHook` instrument). S1's membership replaces 167 with 170–173 at this run; its `recheck`/`unverified` cells are unchanged and its recomposition rides the composition boundary awaiting fresh member verdicts.

---

# Run 22 — accepted-local T006 / S4-S6 scoped recheck (2026-09-28)

Dispatch: accepted-local T006 and its recorded seam scopes only — S4 (per handoff §Composition Routing 2026-09-27: "After T006 lands, refresh affected atomic members against the actual callback→connector→outbound chain and recompose") and S6 (per seam-S6.md's own Invalidates-On: T006 landing `SKIP_TLS_VERIFY` wiring triggers recomposition with its member set), plus the corrected premises of claims 38 and 34 (rows 50 and 74). T006 landed the TLS opt-out: `config.SKIP_TLS_VERIFY: bool = False`, the callback-wired `--skip-tls-verify` flag, the explicit `TCPConnector(ssl=not config.SKIP_TLS_VERIFY)` inside `on_startup`, the serve-script boolean guard, the README wiring, the committed self-signed fixture and 6 posture unit tests ([051-01-forward]); the yield replays of claim-38/claim-34 returned PASS/PASS+CONFIRMED with statuses accepted-local ([051-01-yield]) — replays are not atomic verdicts, so both corrected premises return to blind checks. This is a source-premise delta only: no claim is checked, no seam is composed, and accepted-local Yield is not a verdict for any row. S1–S3, S5, S7–S9 are outside the dispatch scope; their rows are untouched and no unrelated composition was inspected for staleness.

Rows 174–182 record the landed chain as new atomic premises read from the working tree (`config.py`, `main.py`, `server.py`, `serve_buster_ripper.sh`, `README.md`). T006 adds no dependency or package (deps remain `aiohttp` + `click`); the new option/callback joins `main.py`'s existing config-backed option family and the serve-script guard joins the existing boolean-peer guard block — no `Mode: NEW` role is authorized beyond the plan's recorded one. No canonical user factual claim in `decisions.md` is newly linked; no `decisions.md` edit is owed.

**Re-dispatched members (14, edited in place per §3d "retain the row and re-dispatch", fresh bundle keys, handles retained):**
- **S4 exercised/stale set:** 25, 26 (the connector+session pair and the six outbound sites are T006's core; `CL:buster_ripper/client-session-construction` pins `server.py` `e008d3c9…`, now `a9e49823…` — freshness cannot be established), 37/39 (aiohttp ssl topology/polarity — the posture tests and the connector construction exercised them), 40 (click callback timing — exercised by the new callback-wired flag), 50 (claim-38's corrected premise — the connector is constructed inside `on_startup`'s running loop per exactly this mechanism; dispatch-named), 74 (claim-34's corrected premise — `main.py` changed, `cbe2802d…` → `fd36dc12…`; dispatch-named).
- **S6 stale set:** 159–163, 33, 164 — the SAFE S6 composition's own Invalidates-On fired on both source fingerprints (`serve_buster_ripper.sh` `adbe88be…` → `f4076d1c…`; `README.md` `cfad3093…` → `fe80cb9a…`) and requires member re-verification before recomposition; run-19's freshness reasoning applies (registry `CL:` handles cannot establish freshness against the phase's edits, so rows are re-dispatched with handles retained rather than reused).

**Dropped from current S4 membership, not the ledger (nothing silent):** row 24 (`CL:buster_ripper/tls-surface`, HOLD) — the pre-T006 absence authorization whose subject T006 deliberately populated; its registry record's own Invalidates-On (any ssl/TLS symbol appearing in package source) has now fired **by design**. A blind recheck would return DOES-NOT-HOLD against an accepted deliverable; per the run-3 consumed-authorization precedent its verdict and routing stand as evidence history. Rows 32, 38, 49 and the other recorded history stay exactly as recorded.

| # | Claim | Subject | Tasks | Scope | Seams | Check | Bundle | State | Routing | Permanent |
|---|---|---|---|---|---|---|---|---|---|---|
| 174 | `buster_ripper/config.py` declares the module global `SKIP_TLS_VERIFY` with default `False` | repo:buster_ripper.config | T006 | single | S4 | seam:S4 | B147 | HOLD | none | CL:buster_ripper.config/skip-tls-verify-global |
| 175 | `main.py` declares `--skip-tls-verify` as an `is_flag=True` option with `expose_value=False` | repo:buster_ripper.main | T006 | single | S4 | seam:S4 | B146 | HOLD | none | CL:buster_ripper.main/skip-tls-verify-option |
| 176 | `_set_skip_tls_verify` assigns `config.SKIP_TLS_VERIFY` from the option value | repo:buster_ripper.main#_set_skip_tls_verify | T006 | single | S4 | seam:S4 | B146 | HOLD | none | CL:buster_ripper.main#_set_skip_tls_verify/config-assignment |
| 177 | `on_startup` constructs its connector as `aiohttp.TCPConnector(ssl=not config.SKIP_TLS_VERIFY)` | repo:buster_ripper.server#on_startup | T006 | single | S4 | seam:S4 | B148 | HOLD | none | CL:buster_ripper.server#on_startup/tls-connector-construction |
| 178 | `on_startup` passes the explicit connector to its `aiohttp.ClientSession` construction | repo:buster_ripper.server#on_startup | T006 | single | S4 | seam:S4 | B148 | HOLD | none | CL:buster_ripper.server#on_startup/explicit-connector-session |
| 179 | `serve_buster_ripper.sh` initializes `SKIP_TLS_VERIFY` from the environment with default `0` | repo:serve_buster_ripper.sh | T006, T011, T012 | shared | S6 | shared, seam:S6 | B170 | HOLD | none | CL:serve_buster_ripper.sh/skip-tls-verify-env-default |
| 180 | `serve_buster_ripper.sh` appends a `--skip-tls-verify` flag to `EXTRA_FLAGS` only when `SKIP_TLS_VERIFY` equals `1` | repo:serve_buster_ripper.sh | T006, T011, T012 | shared | S6 | shared, seam:S6 | B170 | HOLD | none | CL:serve_buster_ripper.sh/skip-tls-verify-flag-guard |
| 181 | `README.md` documents `SKIP_TLS_VERIFY=1 bash serve_buster_ripper.sh` as the TLS opt-out's run path | repo:README.md | T006, T011 | shared | S6 | shared, seam:S6 | B152 | HOLD | none | CL:README.md/skip-tls-verify-run-path |
| 182 | No call site under `buster_ripper/` passes a per-request `ssl` argument to a session request method | repo:buster_ripper | T006 | single | S4 | absence, seam:S4 | B143 | HOLD | none | CL:buster_ripper/per-request-ssl-absence |

Fresh bundles (never-reuse rule): B143 (25, 26, 182 — one package-wide client-surface sweep: session-construction census, outbound provenance, per-request `ssl` absence), B144 (37, 39, 50 — installed aiohttp client/connector ssl topology, polarity, loop binding), B145 (40 — installed click callback timing), B146 (74, 175, 176 — `main.py` option decorators, the TLS callback, and import-time binding in one read), B147 (174 — `config.py` TLS global), B148 (177, 178 — `server.py` `on_startup` connector/session pair), B149 (159, 160, 161 — the serve script's guard block: env defaults and conditional appends), B150 (162, 163 — exec-line argv assembly and no-positional forwarding), B151 (179, 180 — the serve script's `SKIP_TLS_VERIFY` wiring), B152 (33, 164, 181 — README run-path documentation). The old B15/B22/B31/B43/B49/B123–B125 references are historical.

Both evidence lookups were performed and neither answered a row without a checker: the session ledger holds no prior row asserting the landed-TLS propositions (rows 24–26/37–40/50/74 assert the pre-T006 surface or library facts; 159–164 assert the pre-T006 script/README), and the registry Claim Index holds no `Claim:` record for `repo:buster_ripper.config`'s TLS global, the `--skip-tls-verify` option/callback, the explicit connector construction, the serve-script `SKIP_TLS_VERIFY` wiring, or the README TLS run path (its `CL:serve_buster_ripper.sh/*` and `CL:README.md/*` records cover only the dump-side wiring, on rows re-dispatched here by fingerprint staleness). aiohttp 3.13.3 / click 8.3.1 identities re-verified in `.venv` this run, so the `pkg:` rows re-dispatch by phase exercise, not identity change. Exercised and stale rows cannot inherit by rule, so `i = 0` with both lookups recorded. Runtime posture behavior (VP-T006-*, QAP-MS004), suite passage and milestone acceptance belong to Forward/Yield/QA downstream, not to any row here.

---

# Run 23 — scoped stale-composition re-extraction, S3/S5 (2026-09-28)

Dispatch: scoped re-extraction for stale compositions. S3's and S5's on-disk SAFE compositions (both 2026-09-27, both `PROMOTION: INELIGIBLE` — no permanent member handles) consumed the pre-T006 `main.py` fingerprint `cbe2802d…`; T006's landing moved the working tree to `fd36dc12…` (re-verified by hash this run), firing both compositions' own Invalidates-On — S3's "any member verdict/fingerprint change" (its member fingerprint list pins `cbe2802d…` on rows 99 and 74) and S5's "`main.py` flush-option declarations". Per the claim-34 hold note (handoff §Post-acceptance Finding Resolution — 2026-09-28 (T006), resolution trigger): rows 74 and 40 need no new extraction — their run-22 re-dispatch verdicts postdate the change (`CL:buster_ripper.main/cli-default-binding-time` HOLDS @ `fd36dc12…`, 2026-09-28; `CL:click/option-callback-timing` HOLDS @ click core `53a05fc6…`, 2026-09-28, pkg identity unchanged) — while rows 99 and 100, whose verdicts pin `cbe2802d…` with their own Invalidates-On on `main.py` option declarations, return to blind checks. No new proposition is written: both rows' claims are unchanged; only their freshness state moves. No claim is checked and no seam is composed here.

**Re-dispatched members (2, edited in place per §3d "retain the row and re-dispatch", fresh bundle key, handles retained):**

- **99** (`CL:buster_ripper.main/dump-format-option`) and **100** (`CL:buster_ripper.main/dump-flush-options`) — `repo:buster_ripper.main` property rows whose named unit changed at T006. `T006` appended to both rows' `Tasks` from this run (its edit is the recheck trigger). Fresh bundle **B153** (99, 100 — same subject, one read of `main.py`'s option declarations settles both; the never-reuse rule applies to run-6's B71, which stays history at line 312's note). The `CL:` handles stay in `Permanent` retained per the run-19 precedent: registry records cannot establish freshness against the phase's edits, so the rows re-dispatch rather than reuse.

**Inherited without a checker (16 of the 18 unique S3/S5 member premises):** 74 and 40 (fresh HOLDS above), and 41, 43, 75, 76, 85, 96, 98, 101, 103, 110–112, 116–117 — their subjects' named units or aspects are untouched by T006: `dump.py` `07b0b076…` is byte-identical; `server.py`'s T006 edit (`a9e49823…`) touches `on_startup` only, so the `repo:buster_ripper.server#spawn_dump_flush` property row does not fire (a byte move elsewhere in the same file is not a change to the named unit); `pkg:click@8.3.1`, `pkg:aiohttp@3.13.3` and `lang:python` identities are unchanged. Both evidence lookups were performed: the session ledger (plus the two fresh permanent records) answered the 16 inherited rows, and the registry Claim Index holds no fresh record answering 99 or 100 — their published records pin `cbe2802d…` (Checked-At 2026-09-27), so freshness cannot be established and both re-dispatch with handles retained.

**Seam membership (`seams.md`):** S3 (99, 74, 40, 41, 98, 75, 103) and S5 (100, 74, 40, 43, 76, 85, 116, 117, 111, 112, 96, 101, 110) — unchanged from the hold note's sets; no member is added or dropped. Both seams stay `recheck`/`unverified` with Permanent `—`: no registry `CO:` record covers either member set (both compositions were INELIGIBLE for promotion), and the on-disk SAFE compositions are in-session history whose Invalidates-On fired — neither inherited nor discarded. The recomposition consumes B153's fresh verdicts plus the fresh-HOLDS members. Per the hold note, SAFE recompositions of both seams resolve claim-34 with no further routing — a routing fact recorded there and repeated for provenance only; no judgment is made here.

Nothing dropped: no other row, bundle, verdict, routing or permanent handle was altered, and S1/S2/S4/S6–S9 are outside the dispatch scope with no unrelated composition inspected for staleness.

---

# Run 24 — scoped adhoc extraction, T011 / S6 (2026-10-04)

Dispatch: scoped adhoc extraction for the newly registered **T011 · Honor the serve script's documented `DUMP_DIR=""` disable form** and its recorded seam scope **S6** only. Nothing is checked and nothing is composed here.

**Basis.** `decisions.md` §FK-010 is the task's only authority (`resolved 2026-10-03, option 1`); its `Factual claims:` cell reads **"none routed"** — the answer's clauses are a need judgment plus a presupposition of the already-verified defect — so **no `Check: user-claim` row is owed by this run** and no `Verification` cell needed relinking. No legacy `## User Choices Resolved` record exists, so no `MIGRATION-REQUIRED` return. Task text was read from `plan-summary.md` §Implementation Plan Summary row `T011` and `handoff-context.md` §T011 Planned Task Context (compact verification profile).

**Package/dependency binding.** T011 edits `serve_buster_ripper.sh` (repo root, plain script — no `pyproject.toml` of its own, no new dependency) and adds one module under `tests/`, which belongs to the single `buster_ripper` project package: runtime deps remain `aiohttp` + `click`, dev group pytest 9.0.2 / pytest-asyncio / pytest-aiohttp / pytest-cov / pytest-mock. No `pkg:` row is extracted: the task touches no library call. The subject types this run needs are **`lang:bash`** (the script's interpreter semantics) and **`repo:`** units.

**Why 23 runs produced no `lang:bash` row and this one produces ten.** The serve-script chain was previously extracted only as `repo:` guard facts (rows 159–163, 179–180). The shell *semantics* those guards rest on were carried as prose inside `compositions/seam-S6.md` — and that prose was wrong once already: the 2026-09-28 composition had to correct its own predecessor's assertion that `DUMP_DIR=""` "collapses through to empty → guard false → flag suppressed", because it mis-stated bash `:-` semantics. T011's whole fix is one parameter-expansion operator, so the semantics are now load-bearing premises rather than commentary, and they have never been answered by anyone blind to the plan.

**New rows (17, ids 183–199).** Fifteen carry a check reason and a Bundle; rows 195 and 198 are affirmative single-task claims outside every seam and stay write-time checks for Forward per §3.

| # | Claim | Subject | Tasks | Scope | Seams | Check | Bundle | State | Routing | Permanent |
|---|---|---|---|---|---|---|---|---|---|---|
| 183 | bash's `${VAR:-word}` substitutes `word` when `VAR` is unset or empty | lang:bash | T011 | single | S6 | seam:S6 | B154 | HOLD | none | CL:bash/colon-default-value-form-semantics |
| 184 | bash's `${VAR-word}` substitutes `word` only when `VAR` is unset | lang:bash | T011 | single | S6 | seam:S6 | B154 | HOLD | none | CL:bash/no-colon-default-value-form-semantics |
| 185 | A prefixed assignment `VAR="" cmd` leaves `VAR` set-and-empty in `cmd`'s environment | lang:bash | T011 | single | S6 | seam:S6 | B154 | HOLD | none | CL:bash/prefix-assignment-environment |
| 186 | bash's `${VAR-word}` on an unset `VAR` raises no nounset error under `set -u` | lang:bash | T011 | single | S6 | seam:S6 | B154 | HOLD | none | CL:bash/nounset-default-value-exemption |
| 187 | bash's `[[ -n s ]]` conditional expression is true only for a non-empty string | lang:bash | T011, T012 | shared | S6 | shared, seam:S6 | B155 | HOLD | none | CL:bash/dash-n-primary-semantics |
| 188 | In bash, a command left of `&&` that fails in an `&&` list does not trigger errexit under `set -e` | lang:bash | T011 | single | S6 | seam:S6 | B155 | HOLD | none | CL:bash/errexit-and-list-exemption |
| 189 | An unquoted bash parameter expansion word-splits, and an empty value contributes zero words | lang:bash | T011, T012 | shared | S6 | shared, seam:S6 | B172 | HOLD | none | CL:bash/unquoted-expansion-word-splitting |
| 190 | bash executes the first matching executable found in `$PATH`, so an earlier entry shadows an installed command | lang:bash | T011 | single | S6 | seam:S6, native | B156 | HOLD | none | CL:bash/path-command-search-order |
| 191 | `serve_buster_ripper.sh`'s usage header documents `DUMP_DIR=""` as disabling prompt diffing | repo:serve_buster_ripper.sh | T011, T012 | shared | S6 | shared, seam:S6 | B171 | HOLD | none | CL:serve_buster_ripper.sh/usage-header-dump-dir-documentation |
| 192 | `serve_buster_ripper.sh`'s shebang names `/usr/bin/env bash` as its interpreter | repo:serve_buster_ripper.sh | T011, T012 | shared | S6 | shared, seam:S6 | B171 | HOLD | none | CL:serve_buster_ripper.sh/shebang-interpreter |
| 193 | `serve_buster_ripper.sh`'s exec line names the binary as the bare command `buster-ripper` | repo:serve_buster_ripper.sh | T011, T012 | shared | S6 | shared, seam:S6 | B169 | HOLD | none | CL:serve_buster_ripper.sh/exec-line-command-name |
| 194 | No `README.md` example uses the `DUMP_DIR=""` disable form | repo:README.md | T011 | single | S6 | seam:S6, absence | B158 | HOLD | none | CL:README.md/dump-dir-example-set |
| 195 | `main.py`'s `--dump-dir` option declares `default=None` | repo:buster_ripper.main | T011 | single | — | — | — | HOLD | none | — |
| 196 | `main.py` assigns `config.DUMP_DIR` only inside the branch that tests a truthy `--dump-dir` value | repo:buster_ripper.main | T011 | single | — | absence | B159 | HOLD | none | CL:buster_ripper.main/dump-dir-assignment-guard |
| 197 | `handle_messages` calls `dump_turn` only when `config.DUMP_DIR` is not None | repo:buster_ripper.routes.messages#handle_messages | T011 | single | — | absence | B160 | HOLD | none | CL:buster_ripper.routes.messages#handle_messages/dump-turn-invocation-guard |
| 198 | `flush_all` returns without writing when `config.DUMP_DIR` is None | repo:buster_ripper.dump#flush_all | T011 | single | — | — | — | HOLD | none | — |
| 199 | No module under `tests/` invokes `serve_buster_ripper.sh` | repo:tests | T011 | single | — | absence | B161 | DOES-NOT-HOLD | absorbed | CL:tests/serve-script-invocation |

**Fresh bundles (never-reuse rule; B1–B153 are history).** B154 (183–186 — one read of bash's Parameter Expansion and simple-command assignment sections settles the `:-`/`-` contrast, the environment-prefix setness question, and the nounset interaction); B155 (187–188 — both are answered from the Conditional Expressions and `set -e` conditional-execution exemption in one read, and 187's answer is what makes 188's failing command reachable); B156 (189–190 — Word Expansion and Command Execution, the two shell rules by which the exec line becomes the child's argv); B157 (191–193 — one read of the 46-line script: header comment, shebang, exec-line command word); B158 (194 — README example inventory); B159 (196 — `main.py`'s `--dump-dir` assignment site); B160 (197 — `routes/messages.py`'s dump gate); B161 (199 — a `tests/` sweep for script invocations). No bundle crosses subjects (rule 1); none exceeds 4 rows (rule 3); solos are solos (rule 4).

**Session-ledger match — 10 inherited rows, no new row written (state preserved, `T011` appended to `Tasks`).** Rows **159, 160, 161, 162, 163, 33, 164, 179, 180, 181** are S6's existing members. Their bytes have **not** moved: `serve_buster_ripper.sh` is re-verified this run at `sha256:f4076d1ceae359f201f4e68d62c41ea7af540a0620a6b65943d405351756fa08` (46 lines) and `README.md` at `sha256:fe80cb9acabfe699fac070a69b43ac425436606d627b243825d4a028d75a1c15`, both exact matches for the fingerprints their 2026-09-28 verdicts pin, and no dependency or domain changed. §3d therefore preserves each `HOLD`/`none` cell rather than re-dispatching a blind check against unchanged bytes. The re-verification T011's `Done when` names for the seven script-pinned members (**159–163, 179, 180**) is an *event consequence*, not a present trigger: it fires when the line-28 edit lands and moves that fingerprint, and that pass is the one that must mark them `recheck` with fresh bundle keys. Nothing in this run re-opens a settled verdict.

**Mechanical consistency corrections in the same 10 rows (§3: `SHARED` = named by two or more tasks).** Appending `T011` takes every one of the ten to two or more tasks, so each is `shared`: rows 159–163 already carried that value, and rows 33, 164, 179, 180 and 181 move to it now with `shared` added to their `Check` value to match the 159–163 shape (33 and 164 were already two-task rows marked `single` — a cell left behind when `T006` was appended in run 22; 179–181 were correctly `single` at one task and are two-task rows now). All five already hold a Bundle (`B151`/`B152`), so no dispatch key changed and no verdict moved.

**Permanent-evidence lookups (both performed, `i = 0` for new propositions).** `invariants/index.md` §Claim Index was read this run for each extracted proposition: it holds **no `lang:bash` record of any kind**, and no `Claim:` row for the script's usage header, its shebang, its exec-line command word, a README disable-form absence, `main.py`'s `--dump-dir` default or assignment site, `handle_messages`' `DUMP_DIR` gate, `flush_all`'s None guard, or a `tests/` script-invocation census. The nearest records, `CL:serve_buster_ripper.sh/*` and `CL:README.md/*`, cover the *other* env vars' defaults, guards and argv channel — a different proposition each, not a differently-worded one — and `CL:buster_ripper.main/dump-cli-surface` pins `main.py` at `b4794796…`, which is neither the option-default nor the assignment-site proposition. So no new row is answered without a checker. `§Composition Index` was consulted for S6's exact ordered member set, boundary and dimensions only (no unrelated composition was inspected): `CO:serve-script-env-to-argv/env-var-flag-crossing` matches the **pre-run** member set, and this run adds twelve members, one subject (`lang:bash`) and the disable-form discriminator, so the index record no longer describes the seam's set; freshness cannot be established for a set it does not contain. The handle stays in S6's `Permanent` and the state stays `recheck` per §3c.

**Two directions, per §1.** T011's new member may need what the peers have — rows 159/160 and 179/180 record the `DUMP_FORMAT` and `SKIP_TLS_VERIFY` shapes the `DUMP_DIR` peer is required to keep distinct from, and row 183/184's contrast is what makes line 28's form differ from line 29's. The peers may now need what T011 introduces: once line 36's guard becomes reachable-false, rows 160 and 180 sit in the same guarded-list shape under `set -e`, and the header's disable promise (row 191) is documentation the *other* documented run paths (rows 33, 164, 181) do not carry — row 194 is the record that no README example does either. Both directions are rows, not judgments.

**Nothing dropped silently.** No candidate was discarded. Two rows (195, 198) are written without a check reason because they are affirmative single-task claims outside every seam — Forward meets them at write time — and row 198 is the shutdown-flush branch of the same off-mode chain as 196 and 197. No claim about the design, about repo *state* (suite results, coverage, the working-tree diff), or about this session's governance was written: the "not lie" acceptance and the three-state witness are the task's own done-when and its arbiter, recorded in `handoff-context.md`, and the task's `Verification profile` route (**compact**) is governance, not a premise. `PI:T002/dump_format/*`, `PI:T006/upstream_tls/*`, `OI:` and `EI:` blocks and the `E:T002/dump/dump-dir-config` entity are excluded per §2 — the first two are planned and bound to landed code, the entity is already bound to working code — so no invariant is extracted, cited as authority, or displaced by this run.

---

# Run 25 — accepted-local T011 / S6 scoped recheck (2026-10-05)

Dispatch: accepted-local **T011** and seam **S6** only — re-dispatch the ten script-pinned members fresh against the post-fix bytes, re-extract row 199, and settle rows 195/198 from the Forward write-time verification. Nothing is checked and nothing is composed here; the `seam-S6` recomposition this landing fired belongs to the next pass.

**What moved (freshness basis — hashes establish scope change only, never whether a proposition still holds).** T011's only production edit is one character on `serve_buster_ripper.sh` line 28: `${DUMP_DIR:-${HOME}/mllm/prompt-diffs}` → `${DUMP_DIR-${HOME}/mllm/prompt-diffs}` ([052-01-forward](../iterations/052-01-forward.md) §What was implemented, accepted at [052-01-yield](../iterations/052-01-yield.md) §Yield Verdict). The file is now `sha256:d5db2144a2452840e7a5284c524c3fbfeef60dc117a55d61fab57a5aeafc5c90` (46 lines) where all ten member verdicts pin `sha256:f4076d1ceae359f201f4e68d62c41ea7af540a0620a6b65943d405351756fa08` in their own `## Source Fingerprint`, so every one of them fires on its own Invalidates-On fingerprint clause — rows 179/180 state it plainly ("The file's sha256 changes (re-check the line on any edit)") and row **161** self-flagged this exact event for re-extraction rather than silent reuse ("Line 28's initialization changes such that `DUMP_DIR` could be empty at line 36 without changing the guard"). Line numbering did not move (46 lines before and after), so the named units are the same lines; the byte-keyed instance basis is what expired. `README.md` re-verified this run at `sha256:fe80cb9acabfe699fac070a69b43ac425436606d627b243825d4a028d75a1c15` — byte-identical to the fingerprint rows 33/164/181/194 pin — so those four are not fired. T011's landing is the event run 24 recorded as a *future* trigger ("it fires when the line-28 edit lands and moves that fingerprint, and that pass is the one that must mark them `recheck` with fresh bundle keys"); this is that pass.

**Settled from the Forward write-time verification (2 rows, no checker).** Rows 195 and 198 were written at run 24 as affirmative single-task claims outside every seam — §3's write-time route — and T011's Forward performed exactly that check while writing, with Yield re-reading both subjects independently: row 195 HOLDS at `buster_ripper/main.py:78` (`--dump-dir` declares `default=None`; Forward quoted `:79`, Yield's re-read corrected the pointer — the proposition, not the locator, is what settled, and the row pins no line), and row 198 HOLDS at `buster_ripper/dump.py:108-109` (`flush_all`'s `DUMP_DIR is None` early return), which with 196/197 makes "no `--dump-dir` word" mean "nothing is ever written" ([052-01-forward](../iterations/052-01-forward.md) §Dependency-claim checks; [052-01-yield](../iterations/052-01-yield.md) §Claims exercised). Both rows now read `HOLD`/`none` per the ledger's verdict-vocabulary mapping. The route is the recorded arbiter, not a blind verdict: these are `repo:` facts the plan needed *at write time*, and no `CL:` handle is invented for them here — promotion is a separate pass's act (§3e). Neither gains a Bundle: no checker is owed.

**Re-extraction of row 199 (falsified by design; row 200 is what gets checked).** Row 199's `Invalidates On` names two events that both fired with this landing — a new Python module added under `tests/`, and the introduction of a subprocess reference to `serve_buster_ripper.sh` — and the module is a deliverable the task's own `Done when` mandates ("a hermetic witness in the default suite observes the three states on the script's assembled argv"). Both the writing agent and the verifying agent recorded it as false by design, so the row is not sent back out as it stands: a blind checker would be asked to discover a change the plan required. It becomes terminal evidence — `DOES-NOT-HOLD`/`absorbed`, the shape rows 102 and 32 carry — with its durable disposition being T011's own acceptance ([052-01-yield](../iterations/052-01-yield.md) §Yield Verdict, closure ledger `DC-seam-S6-disable-form-r1`) and its `CL:tests/serve-script-invocation` handle retained as the record of the pre-T011 census. The registry copy still pins the `sha256:c3e83beb…` module-set aggregate, which the new file moved, so the permanent record is stale-by-this-edit and is not reused as fresh evidence (§3e). What the plan now takes for granted in that territory is the *boundary*, not the census (§2: observed counts are the checker's `Actual`, not the proposition): rows 200 and 201.

| # | Claim | Subject | Tasks | Scope | Seams | Check | Bundle | State | Routing | Permanent |
|---|---|---|---|---|---|---|---|---|---|---|
| 200 | No module under `tests/` other than `tests/test_serve_script_argv.py` invokes `serve_buster_ripper.sh` | repo:tests | T011, T012 | shared | — | shared, absence | B166 | HOLD | none | — |
| 201 | `tests/test_serve_script_argv.py` asserts on the script's complete assembled argv word list rather than on individual flag presence | repo:tests.test_serve_script_argv | T011, T012 | shared | — | shared, absence | B167 | HOLD | none | CL:tests.test_serve_script_argv/argv-assertion-completeness |

Row 200 is the surviving half of row 199's authorization — the script has exactly one consumer in the test surface, which is what bounds how far a wrapper edit reaches and is the inventory T012's assembly rewrite must not silently grow. Row 201 is the property that makes "exactly one `--dump-dir`", "no `--dump-dir` at all" and "no positional forwarding" observable at all; T012's `Done when` reuses the same module for its split-word and swallowed-neighbour-flag cases, so that wave's validation is built on the shape this one left. Both are `absence`-flavoured (§3: exhaustive claims are checked regardless of task count) and both are `shared` on the two-task rule; T012's planned context names "the existing argv-witness module" and the whole-list word sets, so the attribution is the plan's text, not an inference about intent. Neither is an S6 member: the witness observes the seam's boundary, it does not sit on the producer→transformation→consumer chain, and §3c forbids inventing a semantic link that no hypothesis supports (E8). Row 201's subject is a different unit from row 200's (`repo:tests.test_serve_script_argv` versus the `repo:tests` sweep), so rule 1 keeps them in separate bundles.

**Re-dispatched members (10, edited in place per §3d "retain the row and re-dispatch", fresh bundle keys, `CL:` handles retained).** **159, 160, 161** → **B162** (one read of the guard block: the `DUMP_DIR`/`DUMP_FORMAT` inits at lines 28–29 and the two `-n` appends at lines 36–37 — row 161's re-extraction question is what line 28's new form means for line 36, which the same read settles); **162, 163, 193** → **B163** (the `exec` statement at lines 42–46 plus the whole-file positional-parameter sweep — 193 was pulled out of its run-24 bundle because the line it names *is* 162's and 163's line); **179, 180** → **B164** (the TLS env init at line 27 and its `== "1"` append at line 35); **191, 192** → **B165** (the usage header at lines 11–16 and the shebang at line 1). All ten move `HOLD`/`none` → `recheck`/`unverified` with Scope, Seams and Tasks untouched; a checker that receives any of them gets the claim and nothing else, exactly as before. No bundle crosses subjects (rule 1), none exceeds four rows (rule 3), every key is new — `B149`/`B150`/`B151`/`B157` are history and no checker receives a claim it already answered.

**Not re-dispatched, with reasons (nothing dropped silently).** **183–190** (`lang:bash`, B154–B156): bash's parameter-expansion, conditional-expression, errexit-exemption, word-splitting and command-search rules did not move, and the fix is precisely the `:-`-versus-`-` contrast rows 183/184 state — no dependency or domain change fires them; the recomposition re-consumes their documentation digests on the new bytes, and T011's `Done when` says the same ("the eight `lang:bash` members 183–190 keep their documentation digests"). **33, 164, 181, 194** (`repo:README.md`, B152/B158): source bytes re-verified identical this run. **196, 197** (`repo:buster_ripper.main`, `repo:buster_ripper.routes.messages#handle_messages`, B159/B160): subjects outside this diff, verdicts fresh on unchanged bytes. **Terminal history:** rows 32 (`DOES-NOT-HOLD`/absorbed) and 49 (`HOLD`, pre-T005 `67d194bc…`) stay exactly as recorded — row 49 left S6 membership at run 19 with its conjuncts restated atomically by 162/163, which are being re-checked here, so its own staleness is not a trigger.

**Both evidence lookups performed; `i = 0` for this run's two new propositions.** Session ledger: no existing row asserts row 200's or row 201's proposition — row 199 asserted the opposite of 200 and is now terminal, and the ledger holds no row on `repo:tests.test_serve_script_argv`, the module T011 created. Registry `## Claim Index` (consulted for these two propositions only): no `Claim:` record for either — its `repo:tests` rows are `serve-script-invocation` (the superseded census, stale aggregate fingerprint) and `upstream-mocking` (a different subject and proposition) — so `new`/`unverified` is correct for both and no row is answered without a checker. The ten re-dispatched rows cannot inherit from permanent evidence either: all nine `CL:serve_buster_ripper.sh/*` records still pin `f4076d1c…`, so freshness cannot be established for any of them (§3e, third row of the table) — handles retained, checkers dispatched. `## Composition Index` was consulted for S6's exact ordered member set only: `CO:serve-script-env-to-argv/env-var-flag-crossing` describes the 22-member set with ten now-`recheck` members on expired bytes, so its verdict is not current, the handle stays in `Permanent`, and S6's State stays `recheck`/`unverified`. No unrelated composition was inspected for staleness, and none is recomposed here.

**Two directions, per §1.** T011's edit consumes what the peers have: the `DUMP_FORMAT` presence-guard shape (159/160) and the boolean `== "1"` value-guard shape (179/180) are what the changed line had to stay distinct from, which is why both peers return on new bytes. The peers consume what T011 introduced: line 36's false arm is now reachable, so row 188's errexit reasoning and row 189's zero-word arm stop being vacuous about a dead branch, row 191's header promise is now documentation a witness holds the script to, and row 201's whole-list shape is what makes the "no `--dump-dir` at all" state distinguishable from a truncated one. Both directions are rows, not judgments — this pass states none of them holds.

**Collected and deliberately not rowed (E4/E2, with routes).** *The witness's interception mechanism* — that a `buster-ripper` shim placed earlier on the child `PATH` receives the exec, that no server starts, that the three argv states come out as the `Done when` lists them — is settled by running the module, which the task's own arbiter did ([052-01-forward](../iterations/052-01-forward.md) §Verification against the task's `Done when`, [052-01-yield](../iterations/052-01-yield.md) replay); §2's exclusion table puts it with the done-when clauses, and row 190 already carries the shell half as a blind `lang:bash` claim. *Suite passage, test counts, coverage, the acceptance status, `DC-seam-S6-disable-form-r1`'s lifecycle and the compact verification profile* are session governance and arbiter state, not premises. *`E:T002/dump/bulk-flush`, `E:T002/dump/dump-dir-config`, `PI:T002/dump_format/*`, `PI:T006/upstream_tls/*`, `OI:`/`EI:`* are bound to working code and excluded as maintenance (§2); row 198 restates `flush_all`'s no-op as the assumption the fix leans on without citing any invariant as authority, and none is created, weakened or displaced by this run. No design claim was written, no compound row was kept (E5), nothing was filtered for risk (E6), and no seam was judged safe (E8).

---

# Run 27 — accepted-local T012 / S6 scoped recheck (2026-10-05)

Dispatch: accepted-local **T012** and seam **S6** only — re-dispatch the ten script-pinned members **159, 160, 161, 162, 163, 179, 180, 191, 192, 193** plus **189** and **162** fresh on the `c58256fb…` bytes, and write the one owed atomic value-grammar premise row (rider 2 of the 2026-10-05 routing). Nothing is checked and nothing is composed here; the `seam-S6` recomposition this landing fires belongs to the next pass, and it is to be dispatched **with the 23 `CL:` handles in its arguments** (rider 1 — the composition role's, not this pass's).

**Numbering.** The claims ledger's own sequence ends at run 25; `seams.md` numbers the 2026-10-05 T011 recomposition as its own run 26. This pass takes **27** so one handle names one pass across both ledgers.

**What moved (freshness basis — hashes establish scope change only, never whether a proposition still holds).** T012 rewrote the assembly inside the lines the task's own Scope names: `EXTRA_FLAGS` went from a string to an array, the two presence-guard appends now interpolate their value **quoted** (`serve_buster_ripper.sh:40-41`), and the exec's final word became the guarded array expansion `${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}` (`:54`), with the six-guard block renumbered to `:37-42` and the file at 46 → **54 lines**. `serve_buster_ripper.sh` is now `sha256:c58256fbec793c520720c9e86fee8740653939275b508f50b212ab60bd0908af`, where all ten script-pinned verdicts read this run pin `sha256:d5db2144a2452840e7a5284c524c3fbfeef60dc117a55d61fab57a5aeafc5c90` in their own `## Source Fingerprint`, so every one of them fires on its byte-keyed instance basis — and row **162**'s `Invalidates On` names this exact edit in words ("`EXTRA_FLAGS` is removed from the `exec` line, or replaced by per-flag conditionals / **an array expansion**"; "The file's bytes change at all … re-check rather than reuse"). Line numbers moved for the guard block and the exec, so the `repo:` instance rows' named units had to be re-located as well as re-hashed: the `DUMP_DIR`/`DUMP_FORMAT` appends are now 40/41 (were 36/37), the `SKIP_TLS_VERIFY` append is 39 (was 35), and the `exec` statement is 50–54 (was 42–46). Lines 27–29 and the header block 11–16 did **not** move (`README.md` re-verified this run at `sha256:fe80cb9acabfe699fac070a69b43ac425436606d627b243825d4a028d75a1c15`, byte-identical to the fingerprint rows 33/164/181/194 pin), so those four README-pinned members and the unchanged-init rows' *propositions* stand — it is their evidence that expired, which is a re-dispatch, not a reversal. Row **189** is `lang:bash` and no repo edit fires a `lang:` claim by §2's subject typing; it returns because the dispatcher names it and because the chain it was cited for moved — its null-argument-removal half is what the composition traced at the *unquoted* expansion that no longer exists, and the recomposition re-matches its page digest (`f92d0209…`) before reuse per the on-file record's own `Invalidates On`. **162** was already in the ten; it is named twice in the dispatch because it is both byte-fired and self-naming.

**The owed atomic value-grammar premise row (rider 2).** The 2026-10-05 routing records that Forward Collect owes "one atomic premise row for the value-grammar contract (every forwarded value reaches argv as exactly one word; `$IFS` whitespace, glob metacharacters and a leading dash carry no wrapper-level restriction)", because the `SAFE` composition otherwise classifies C2's observable as a *named non-finding* for want of a premise, and a non-finding is not a closure. That sentence as written is three propositions (one-word transport, glob behavior, dash behavior), so it cannot go in one row (E5); the three are one **domain** over a single predicate — how many argument words a transported value becomes — and the atom the seam was missing is the argv-boundary rule:

| # | Claim | Subject | Tasks | Scope | Seams | Check | Bundle | State | Routing | Permanent |
|---|---|---|---|---|---|---|---|---|---|---|
| 202 | bash's `"${array[@]}"` expansion in double quotes contributes exactly one argument word per array element, whatever characters an element contains | lang:bash | T012 | single | S6 | seam:S6 | B172 | HOLD | none | CL:bash/quoted-array-expansion-word-identity |

One subject, one verdict, one proposition: the word-identity of an array expansion at the point argv is built. Its domain ("whatever characters an element contains") is where the rider's three named classes live — a checker answers it from bash's Arrays and Word Expansion sections and need not know why anyone is asking. It is `single`-scope (T012 alone names it), an S6 member, and therefore carries a Bundle under §3c. The append-side half of the same contract — a quoted scalar expansion becoming one *element* rather than several — is not restated here as a second row: it is the contrapositive of row 189's own first half, which the shell manual settles on the same page, and 189 is being re-checked on this wave; that is why 189 and 202 share B172 rather than each getting a solo key.

**Re-dispatched members (11, edited in place per §3d, fresh bundle keys, `CL:` handles retained).** **159, 160, 161** → **B168** (the three defaulting inits at 27–29 and the two `-n` appends at 40–41 — one read of the renumbered guard block); **162, 163, 193** → **B169** (the `exec` statement at 50–54 plus the whole-file positional-parameter sweep); **179, 180** → **B170** (the TLS env init at 27 and its `== "1"` append at 39); **191, 192** → **B171** (the usage header at 11–16 and the shebang at line 1); **189, 202** → **B172** (the two halves of bash's word-identity rule, one read of §3.5). All eleven existing rows move `HOLD`/`none` → `recheck`/`unverified`; Seams and Permanent are untouched, and a checker that receives any of them gets the claim and nothing else. No bundle crosses subjects (rule 1), none exceeds four rows (rule 3), and every key is new — `B162`–`B167` are history, so no checker receives a claim it already answered.

**Task attribution and the shared cells (§3, mechanical).** Appending `T012` — the plan's own re-verification set names 159–163, 179, 180, 191–193 against this task — takes rows **163, 179, 180, 191, 192, 193** to two or more tasks, so each moves `single` → `shared` with `shared` added to its `Check` value; rows 159–162 already carried `shared`. **Row 187** receives the same mechanical correction with **no** re-dispatch: it has read `T011, T012` since run 24 while still marked `single`, its `[[ -n s ]]` proposition is unaffected by the edit (the six guard conditions are byte-identical, only their line numbers moved), no `lang:` freshness trigger fired, and its `HOLD`/`none` cells and `B155` key are retained — a Check-cell that matches Scope is bookkeeping, and §3d's "no recheck trigger applies" leaves the verdict alone.

**Not re-dispatched, with reasons (nothing dropped silently).** **183–186, 188, 190** (`lang:bash`): bash's parameter-expansion, nounset, errexit-exemption and command-search rules did not move, and the fix changes neither the `-`-versus-`:-` contrast nor the `[[ -n ]]` guard conditions; the recomposition re-consumes their documentation digests, and T012's `Done when` names only 189 and 162 as the re-check set. **33, 164, 181, 194** (`repo:README.md`): source bytes re-verified identical this run. **195–198** (`repo:buster_ripper.main`, `#handle_messages`, `#flush_all`): subjects outside this diff, verdicts fresh on unchanged bytes; no `buster_ripper/` file moved. **199** stays `DOES-NOT-HOLD`/`absorbed` terminal history, and **200, 201** (`repo:tests`, `repo:tests.test_serve_script_argv`) stay `HOLD` — T012 extended the witness module in place and its whole-list-equality shape is what claim-201 pins, so its `Invalidates-On` ("a new environment-state test that asserts only part of the argv") did not fire: every new case asserts the complete list ([053-01-forward](../iterations/053-01-forward.md) §What was implemented). Row 200's census did not grow either — the script still has exactly one consumer in the test surface.

**Both evidence lookups performed; `i = 0` for the new proposition.** Session ledger: no existing row asserts row 202's proposition — the ledger's `lang:bash` rows cover default-value forms (183/184), setness (185), nounset (186), the `-n` primary (187), errexit (188), *unquoted* splitting (189) and command search (190); none states the quoted array expansion's word identity, so `new`/`unverified` is correct and no checker is skipped for it. Registry `## Claim Index` (consulted for this proposition only): **no `lang:bash` `Claim:` record for it** — the nearest, `CL:bash/unquoted-expansion-word-splitting`, is the opposite polarity of the same mechanism, a different proposition rather than a differently-worded one — so no row is answered from permanent evidence. The eleven re-dispatched rows cannot inherit either: the nine `CL:serve_buster_ripper.sh/*` records still pin `d5db2144…` and `CL:bash/unquoted-expansion-word-splitting` pins the page digest the recomposition must re-match, so freshness cannot be established for any of them (§3e, third row of the table) — handles retained, checkers dispatched. `## Composition Index` was consulted for S6's exact ordered member set only (no unrelated composition inspected for staleness): `CO:serve-script-env-to-argv/env-var-flag-crossing` still carries the **`fb247496…` COUNTEREXAMPLE** basis over the 22-member pre-T011 set, which both T011's and T012's landings fired and which this run widens to 23 members and one new dimension, so its verdict is not current, the handle stays in `Permanent`, and S6's State stays `recheck`/`absorbed` with its Routing unchanged — the handle-bearing republish rider carries it.

**Two directions, per §1.** T012 consumes what the peers have: the guard *conditions* row 187 settles and the `:-`-versus-`-` forms rows 183/184 settle are byte-identical under the rewrite, which is why the task's Scope pins line 28, header line 14 and `README.md` as out of bounds — and 161/160's presence-guards are the appends whose interpolation changed. The peers consume what T012 introduces: row 189's zero-word arm now belongs to the `${arr[@]+…}` guard rather than to a null unquoted expansion, so its re-check must be read against a chain that no longer has an unquoted accumulator at the exec; row 163's "forwards no positional arguments" stops being merely true of the exec's own words and becomes load-bearing against split values, since under the old assembly a whitespace-bearing `DUMP_DIR` manufactured a positional the script never wrote; and rows 191/194's documented forms are now transported by a channel none of the documentation describes. Both directions are rows, not judgments — this pass states none of them holds.

**Collected and deliberately not rowed (E4/E2/E5, with routes).** *The bash-3.2 empty-array `nounset` behavior* — that bare `"${EXTRA_FLAGS[@]}"` aborts on an empty array under `set -u` on the interpreter `env bash` resolves to here (3.2.57), and that the `${arr[@]+"${arr[@]}"}` guard expands to zero words — is real version-gated semantics, but it is not the row this dispatch names, and the interpreter identity Forward established is a probe result: §2's route for it is the recorded Forward/Yield arbiter ([053-01-forward](../iterations/053-01-forward.md) §Dependency-claim checks) rather than a blind doc read. If the recomposition traces the zero-word arm as a dimension in its own right, that atom is the next collect pass's to write, and this note is the record that it was seen and not filtered (E6). *`tests/test_serve_script_argv.py`'s new cases, their pass/fail counts, the mutation runs, suite passage at 301 @ 93.88%, the coverage figure, `QB:serve-script-argv/zero-word-arm`'s absorption and the compact verification profile* are arbiter state and session governance (E4). *Whether the wrapper validates a value's characters* — no `--dump-dir=` protocol, no rejection of a leading dash — is the wrapper's design posture, recorded in the task's `Required behavior` and Forward's §Concerns 3; §3's "no claim about the design" (E3) routes it away from this ledger, and row 202 carries the shell half the composition needs. *`E:T002/dump/bulk-flush`, `E:T002/dump/dump-dir-config`, `PI:T002/dump_format/*`, `PI:T006/upstream_tls/*`, `OI:`/`EI:`* are bound to working code and excluded as maintenance. No invariant was created, weakened or displaced; nothing was judged, checked or composed here, and no seam was called safe (E8).

# Run 28 — scoped extraction for composition blocker, S6 dim 9 (2026-10-05)

**Dispatch: seam S6 only — the memberless arm.** `seams.md` S6's ninth dimension is "zero-word arm when every append is suppressed", and it is the one dimension of this seam that carries **no member row**: the arm's mechanism post-T012 is the `${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}` guard at the `exec` (`serve_buster_ripper.sh:54`), and run 27 recorded the guard's semantics as seen, deliberately not rowed, and belonging to the next collect pass (E6 — that note is why this row exists rather than being a rediscovery). This pass rows the missing atomic premise and adds it to S6 membership. Nothing is checked and nothing is composed here; the recomposition this landing fires belongs to the next pass and is to be dispatched with S6's `CO:` handle in its arguments, as run 27's rider 1 already routes.

**Numbering.** Claims ledger sequence ran to 27, seams ledger to 26; this pass takes **28** in both, keeping one handle on one pass.

**Two rows, not one (E5).** The dispatch names the behavior as "zero words vs nounset abort" — that phrase is the *dimension's* alternative, not one proposition, and the two halves are settled by two different sentences of the manual: which arm the alternate-value test takes (§3.5 Shell Parameters), and whether the nounset check fires inside a setness test (the `set -u` entry, which exempts parameters "being tested for its being set or unset"). This ledger already splits exactly that pair: rows 183/184 state what the default-value forms substitute and row 186 states that the same form raises no nounset error — one proposition per row, two rows for one form. A single compound row would earn one `HOLDS` covering a part nobody checked, which is the failure mode E5 exists to prevent. Both rows are about the **guarded** form, because that is the form now at the boundary.

| # | Claim | Subject | Tasks | Scope | Seams | Check | Bundle | State | Routing | Permanent |
|---|---|---|---|---|---|---|---|---|---|---|
| 203 | bash's `${array[@]+word}` alternate-value expansion contributes zero words when `array` is unset or has no elements | lang:bash | T012 | single | S6 | seam:S6 | B173 | HOLD | none | CL:bash/array-alternate-value-word-contribution |
| 204 | Under `set -u`, bash's `${array[@]+word}` expansion on an unset or empty `array` raises no nounset error | lang:bash | T012 | single | S6 | seam:S6 | B173 | HOLD | none | CL:bash/array-alternate-value-nounset-exemption |

**Universe of discourse, and what is deliberately excluded from the wording.** Both rows name a *form*, not this repository's file: no line number, no `EXTRA_FLAGS` identifier, no interpreter version. A checker holding either row answers it from bash's own documentation and cannot tell why anyone is asking, which is the requirement (§2). The version question run 27 flagged stays out on purpose: that the *unguarded* bare `"${array[@]}"` aborts on an empty array under `set -u` on the interpreter `/usr/bin/env bash` resolves to here (3.2.57, versus the 4.4+ relaxation) is real semantics, but it is (i) about a form no longer present at the `exec`, and (ii) settled by identifying the installed interpreter, which is a probe result — §2 routes it to the recorded Forward/Yield arbiter, not to a blind doc read (E2/E4). Rowing it would smuggle repo state into a `lang:` claim; the rows above are the arm the composition needs and are version-independent as stated. If a checker finds the guard's behavior is itself version-qualified, that belongs in its evidence `Actual`, and the recomposition reads it there.

**Bundle (§3b).** Same `Subject` (`lang:bash`), and both halves are settled by one read of the manual's Shell Parameters section together with the `set -u` entry — rule 2, "both are settled by the same source read once". Two rows in **B173**, a fresh key: `B172` (rows 189/202) is closed, both members of it now sit `HOLD`/`none`, and no key may straddle a settled wave and a new one. No bundle crosses subjects (rule 1); four-row ceiling not approached (rule 3).

**Both evidence lookups performed; `i = 0`.** *Session ledger*: the `lang:bash` set covers the two default-value forms (183/184), prefix-assignment setness (185), the scalar nounset exemption (186), the `-n` primary (187), errexit (188), *unquoted* splitting and its empty-value zero-word arm (189), command search (190) and *quoted* array word-identity (202) — no existing row asserts anything about the `+` alternate-value form applied to `[@]`, so `new`/`unverified` is correct for both rows and no checker is skipped. *Registry `## Claim Index`*: consulted for these two propositions only; it holds nine `lang:bash` records, and the two nearest are each a different proposition — `CL:bash/nounset-default-value-exemption` is the **scalar** `${VAR-word}` form (different expansion, different subscript), and `CL:bash/quoted-array-expansion-word-identity` is what the *inner* expansion contributes **once the guard has chosen that arm**, not whether the guard takes it. Neither answers a `+`-form claim, so no row is answered from permanent evidence and no `Permanent` handle is invented here (extraction never creates one).

**S6 consequence (membership only).** The seam's ordered member set widens **23 → 25** (rows 203, 204 appended) and dim 9 gains its rows; `Dimensions` is re-worded at that one dimension to name them, and the seam's State is already `recheck`, which a member-set change keeps rather than alters. `Routing` (`absorbed`) and `Permanent` (`CO:serve-script-env-to-argv/env-var-flag-crossing`) are left exactly as run 27 and run 26 set them: the recorded basis covers the 22- and 23-member sets, so it cannot be current for a 25-member set, and whether the widened seam now composes is the composition pass's to determine (E8) — this pass states only that dim 9 has premises to be composed, not that they compose.

**Nothing else moved.** No existing row's State, Routing, Scope, Seams, Bundle or Tasks cell was edited on this pass; rows 159–202 are untouched, including the eleven run-27 re-dispatched members now sitting `HOLD`. **Two directions (§1), recorded as rows not judgments:** the new member needs what the peers have — the guard is reached only when every append is suppressed, which is rows 159–161/179–180's `[[ -n ]]` presence conditions and row 187's primary; and the peers now need what the new member supplies — row 162's claim about the exec's final word, row 189's zero-word arm, and row 202's per-element word identity all describe the channel *inside or beside* the guard, so the recomposition's transport story for the suppressed case now runs through 203/204 rather than terminating at a memberless dimension. Nothing dropped silently (E7): the only item collected and not rowed is the bare-`[@]` nounset abort named above, with its arbiter route. No seam was called safe; no claim was checked.

---

# Run 30 — scoped adhoc extraction, T013 · header-parity wave (2026-10-05)

Dispatch: **scoped adhoc extraction**, mode `adhoc`. Required ledger delta named by the dispatcher: **T013** — F-4 (five-site filter substitution via the shared `utils` helpers), G-1 (post-commit error handling at the `messages.py:154` boundary), G-2 (`REFACTORING_PLAN.md` phantom `convert/` rows) — **plus the aiohttp response-commit/second-head premises, plus the `seam-S9` rider** (member row 67 and `seam-S9.md`'s §Sources range `routes/messages.py:121-176`, evaluated on current bytes). Extraction only: nothing here is checked, composed, or judged, and no seam is called safe (E1/E8).

**Numbering.** The claims ledger's own sequence ended at run 28; `seams.md` numbers the 2026-10-05 T012 resolution pass as its run 29. This pass takes **30** in both ledgers, keeping run 27's rule that one handle names one pass across both files.

**Run basis.**
- **Blueprint.** `plan-summary.md` §Implementation Plan Summary row `T013` is a pointer; the single blueprint is `handoff-context.md` §Planned Task Context → `T013 Planned Task Context` (`:333-510`), carrying `062-decide`'s inline repairs `R-A`…`R-E`. Its two invariant *candidates* (shared-filter policy; no fresh response after commit) are non-authoritative and Interview's to record, so they are **not** treated as invariants here (§2). No active/wip invariant governs this surface: `invariants/index.md` §Handle Index lists only `planned` `PI:` rows (dump format, upstream TLS) and `OI:` overrides, none about headers or response lifecycle, so the packet's "zero active/wip invariants" is this pass's own reading of the index, not an extracted claim.
- **Package/dependency binding.** Every member lands in the single package `buster_ripper` (`pyproject.toml`). Runtime deps unchanged: `aiohttp`, `click`; installed identities read this run from `.venv/lib/python3.13/site-packages`: **aiohttp 3.13.3**, **click 8.3.1** (Python 3.13). Every `pkg:` row below therefore carries `pkg:aiohttp@3.13.3`.
- **Peer set found by listing, not judgment** (§1): the package's route family is `buster_ripper/routes/{chat,count_tokens,messages,passthrough}.py` (registered at `server.py:127-132`); the header-helper family is `buster_ripper/utils.py:12-33`; the base contract is the tracked pre-refactor monolith `buster_ripper_legacy.py`; the response-commit family is `aiohttp`'s `web_response`/`web_protocol`.
- **Universe of discourse for every absence/exhaustive row:** `buster_ripper/` means the package tree (the root-level `buster_ripper_legacy.py` is named explicitly by its own subject where it is the referent); test sweeps say **Python test sources under `tests/`**, which is needed because `tests/__pycache__/test_chat_routes.cpython-314-pytest-9.0.2.pyc` is a stale artifact of a file deleted in `1737663` and a checker sweeping `tests/` for `bearer` would otherwise hit it (verified: the only such hit).
- **`user-claim` route:** `decisions.md` holds no `Verification: assumptions PENDING` factual claim relevant to this wave (its only `PENDING` is `yield-phase:MS-001`, an arbiter route), and `062-decide` records `0 user forks`, so **no `user-claim` row and no `MIGRATION-REQUIRED` return**: the T013 block is a closure-packet artifact, not a detailed legacy `## User Choices Resolved` record, and every source-checkable proposition inside it is rowed below as an ordinary atomic row.
- **HEAD at extraction:** `bcf596ec2066fe8432abf4987bf575d695524d01`.

## Run 30 claims

| # | Claim | Subject | Tasks | Scope | Seams | Check | Bundle | State | Routing | Permanent |
|---|---|---|---|---|---|---|---|---|---|---|
| 205 | `utils.forward_headers` drops `host`, `content-length` and every name in `utils._HOP_BY_HOP` | repo:buster_ripper.utils | T013 | single | S10 | seam:S10 | B174 | HOLD | none | — |
| 206 | `utils.response_headers` drops exactly the names in `utils._HOP_BY_HOP` and no others | repo:buster_ripper.utils | T013 | single | S10 | seam:S10 | — | superseded | none | — |
| 207 | `utils._HOP_BY_HOP` is a frozenset naming `connection`, `keep-alive`, `proxy-authenticate`, `proxy-authorization`, `te`, `trailers`, `transfer-encoding` and `upgrade` | repo:buster_ripper.utils | T013 | single | S10 | seam:S10 | B174 | HOLD | none | — |
| 208 | The eight names in `utils._HOP_BY_HOP` are the hop-by-hop header fields RFC 9110 §7.6.1 defines | rel:buster_ripper/utils#_HOP_BY_HOP→rfc9110§7.6.1 | T013 | single | S10 | seam:S10 | B175 | DOES-NOT-HOLD | absorbed | — |
| 209 | `utils.forward_headers` is never called by any module under `buster_ripper/` | repo:buster_ripper | T013 | single | S10 | absence, seam:S10 | — | superseded | none | — |
| 210 | `utils.response_headers` is never called by any module under `buster_ripper/` | repo:buster_ripper | T013 | single | S10 | absence, seam:S10 | — | superseded | none | — |
| 211 | No outbound HTTP request issued by a module outside `buster_ripper/routes/` passes a `headers` argument | repo:buster_ripper | T013 | single | S10 | absence, seam:S10 | B176 | HOLD | none | — |
| 212 | `buster_ripper/routes/chat.py` imports `forward_headers` without calling it | repo:buster_ripper.routes.chat | T013 | single | S10 | seam:S10 | — | superseded | none | — |
| 213 | `_prepare_headers` deletes an `Authorization` header whose value strips to `Bearer` or `Bearer ` | repo:buster_ripper.routes.chat#_prepare_headers | T013 | single | S10 | seam:S10 | B177 | HOLD | none | — |
| 214 | No header filter under `buster_ripper/routes/` drops any name in `utils._HOP_BY_HOP` other than `connection` and `transfer-encoding` | repo:buster_ripper.routes | T013 | single | S10 | absence, seam:S10 | — | superseded | none | — |
| 215 | Every response-side header filter under `buster_ripper/routes/` drops `content-length` | repo:buster_ripper.routes | T013 | single | S10 | absence, seam:S10 | B178 | HOLD | none | — |
| 216 | Every header-filtering comprehension under `buster_ripper/routes/` iterates `.items()` on a headers mapping | repo:buster_ripper.routes | T013 | single | S10 | absence, seam:S10 | B178 | UNCLEAR | malformed | — |
| 217 | Every header filter under `buster_ripper/routes/` tests membership against the lowercased header name | repo:buster_ripper.routes | T013 | single | S10 | absence, seam:S10 | — | superseded | none | — |
| 218 | `routes/count_tokens.py`'s handler issues no upstream HTTP request | repo:buster_ripper.routes.count_tokens#handle_count_tokens | T013 | single | S10 | absence, seam:S10 | B179 | HOLD | none | — |
| 219 | `buster_ripper_legacy.py`'s `_HOP_BY_HOP` names the same eight header values as `buster_ripper/utils.py`'s `_HOP_BY_HOP` | rel:buster_ripper_legacy#_HOP_BY_HOP→buster_ripper/utils#_HOP_BY_HOP | T013 | single | S10 | seam:S10 | B180 | HOLD | none | — |
| 220 | `buster_ripper_legacy.py`'s `_forward_headers` drops `host`, `content-length` and every member of its `_HOP_BY_HOP` | repo:buster_ripper_legacy | T013 | single | S10 | seam:S10 | B181 | HOLD | none | — |
| 221 | `buster_ripper_legacy.py`'s `_forward_headers` applies no rule to an `Authorization` header | repo:buster_ripper_legacy | T013 | single | S10 | absence, seam:S10 | B181 | HOLD | none | — |
| 222 | `buster_ripper_legacy.py`'s `_response_headers` forwards a `content-length` header it receives | repo:buster_ripper_legacy | T013 | single | S10 | absence, seam:S10 | B181 | HOLD | none | — |
| 223 | Every upstream request in `buster_ripper_legacy.py` builds its header set through `_forward_headers` | repo:buster_ripper_legacy | T013 | single | S10 | absence, seam:S10 | B182 | DOES-NOT-HOLD | absorbed | — |
| 224 | `buster_ripper_legacy.py`'s empty-`Bearer` deletion appears only inside its chat-completions handler | repo:buster_ripper_legacy | T013 | single | S10 | absence, seam:S10 | B182 | HOLD | none | — |
| 225 | No module under `buster_ripper/` or `tests/` imports `buster_ripper_legacy` | repo:buster_ripper_legacy | T013 | single | — | absence | B182 | HOLD | none | — |
| 226 | aiohttp's `web.StreamResponse.prepare()` sends the response head for that object on its first call | pkg:aiohttp@3.13.3 | T013 | single | S11 | seam:S11 | B183 | HOLD | none | — |
| 227 | aiohttp's `web.StreamResponse.prepare()` returns without writing a head when that same response object is already prepared | pkg:aiohttp@3.13.3 | T013 | single | S11 | seam:S11 | B183 | HOLD | none | — |
| 228 | A freshly constructed `aiohttp.web.json_response` is unprepared and holds no payload writer | pkg:aiohttp@3.13.3 | T013 | single | S11 | seam:S11 | B183 | HOLD | none | — |
| 229 | Preparing a second, distinct response object for a request whose first response is already committed writes a second status line and header block onto that connection | pkg:aiohttp@3.13.3 | T013 | single | S11 | seam:S11 | B184 | HOLD | none | — |
| 230 | When an aiohttp request handler raises, the aiohttp server sends its own generated error response for that request | pkg:aiohttp@3.13.3 | T013, T014, T018 | shared | S11, S14, S16 | shared, seam:S11/S14/S16 | B184 | HOLD | none | — |
| 231 | `aiohttp.ClientPayloadError` is a subclass of `aiohttp.ClientError` | pkg:aiohttp@3.13.3 | T013 | single | S11 | seam:S11 | B185 | HOLD | none | — |
| 232 | aiohttp's client-side timeout errors are subclasses of `aiohttp.ClientError` | pkg:aiohttp@3.13.3 | T013 | single | S11 | seam:S11 | B185 | UNCLEAR | malformed | — |
| 233 | aiohttp's `web.StreamResponse.write()` signals a closed client connection by raising an exception derived from `Exception` | pkg:aiohttp@3.13.3 | T013 | single | S11 | seam:S11 | B186 | HOLD | none | — |
| 234 | aiohttp's client session supplies a `Host` request header when the caller passes none | pkg:aiohttp@3.13.3 | T013 | single | S10 | native, seam:S10 | B187 | HOLD | none | — |
| 235 | aiohttp's client session computes `Content-Length` for a request body it sends | pkg:aiohttp@3.13.3 | T013 | single | S10 | native, seam:S10 | B187 | HOLD | none | — |
| 236 | aiohttp's `web.Response` sets `Content-Length` from its body when it sends the response | pkg:aiohttp@3.13.3 | T013, T015 | single | S10 | native, seam:S10 | B188 | HOLD | none | — |
| 237 | Iterating `aiohttp.web.Request.headers.items()` yields each header name spelled as the client sent it | pkg:aiohttp@3.13.3 | T013 | single | S10 | seam:S10 | B189 | DOES-NOT-HOLD | absorbed | — |
| 238 | Iterating `aiohttp.ClientResponse.headers.items()` yields each header name spelled as the upstream sent it | pkg:aiohttp@3.13.3 | T013 | single | S10 | seam:S10 | B189 | DOES-NOT-HOLD | absorbed | — |
| 239 | Python's built-in `str` type has no `get` attribute | lang:python | T013 | single | S11 | seam:S11 | B190 | HOLD | none | — |
| 240 | `json.loads` returns a Python `str` for a JSON string value nested in an object | lang:python | T013 | single | S11 | seam:S11 | B190 | HOLD | none | — |
| 241 | `_stream_single_request`'s `try` block covers both its `response.prepare()` call and its chunk relay loop | repo:buster_ripper.routes.messages#_stream_single_request | T013 | single | S11 | seam:S11 | — | superseded | none | — |
| 242 | The only `except` clause in `_stream_single_request` catches `aiohttp.ClientError` | repo:buster_ripper.routes.messages#_stream_single_request | T013 | single | S11 | absence, seam:S11 | B191 | HOLD | none | — |
| 243 | `_stream_single_request`'s `ClientError` arm returns a newly constructed `web.json_response` | repo:buster_ripper.routes.messages#_stream_single_request | T013 | single | S11 | seam:S11 | B191 | HOLD | none | — |
| 244 | `find_message_start` returns a parsed event on a `type` match without checking any other field's shape | repo:buster_ripper.utils#find_message_start | T013 | single | S11 | absence, seam:S11 | B192 | HOLD | none | — |
| 245 | `_stream_single_request` calls `.get()` on the `message` value of a `message_start` event without a type check | repo:buster_ripper.routes.messages#_stream_single_request | T013 | single | S11 | seam:S11 | — | superseded | none | — |
| 246 | No Python test source under `tests/` asserts that a hop-by-hop header is dropped at a proxy boundary | repo:tests | T013 | single | S10 | absence, seam:S10 | — | superseded | none | — |
| 247 | No Python test source under `tests/` asserts anything about an `Authorization` header value | repo:tests | T013 | single | S10 | absence, seam:S10 | — | superseded | none | — |
| 248 | No Python test source under `tests/` exercises a failure raised after `prepare()` inside `_stream_single_request` | repo:tests | T013 | single | S11 | absence, seam:S11 | — | superseded | none | — |
| 249 | No directory named `convert` exists under `buster_ripper/` | repo:buster_ripper | T013 | single | — | absence | B194 | HOLD | none | — |
| 250 | Every `buster_ripper/convert/` path named in `REFACTORING_PLAN.md` is absent from the working tree | repo:REFACTORING_PLAN.md | T013 | single | — | absence | B195 | HOLD | none | — |
| 251 | No file tracked by git other than `REFACTORING_PLAN.md` names a `buster_ripper/convert/` path | repo:REFACTORING_PLAN.md | T013 | single | — | absence | B195 | DOES-NOT-HOLD | absorbed | — |
| 252 | No module under `buster_ripper/` or `tests/` reads `REFACTORING_PLAN.md` | repo:REFACTORING_PLAN.md | T013 | single | — | absence | B195 | HOLD | none | — |

**Row-shape notes (E5 — why some of these are pairs rather than one row).** The packet states several of these as single sentences that carry two facts, and they are split here: "`aiohttp recomputes it`" is `Content-Length` on the client request (235) and on the server response (236) — two library mechanisms on two different classes, and the fix's parenthetical needs only the second; "`ClientError` (incl. `ClientPayloadError` …; aiohttp timeouts resolve to `ClientError`)" is three propositions (231, 232, and the already-settled 64) over one hierarchy page; "a failure after `prepare()` is logged and the stream abandoned" is not a fact at all (E3 — it is the design posture, recorded in the task's `Required behavior`) and is absent from the table; and the second-head sentence — "`prepare()` does not short-circuit, so a second header block is written" — is the conjunction of a short-circuit rule (227), a precondition about the new object (228), and the transport consequence (229), which are settled by different pages and can disagree. Row 208 is included although it looks unsurprising (E6): the invariant candidate's phrase "the full RFC 9110 §7.6.1 set" is an assumption about a standard, and it is the only row that can tell the plan its eight-name target is or is not the standard's whole list.

## Run 30 bundles (fresh keys — B1–B195 are history; one agent per bundle)

| Bundle | Rows | Subject(s) | Why tied |
|---|---|---|---|
| B174 | 205, 206, 207 | repo:buster_ripper.utils | one read of `utils.py:12-33` |
| B175 | 208 | rel:…→rfc9110§7.6.1 | solo — an external standard, no repo read settles it |
| B176 | 209, 210, 211 | repo:buster_ripper | one package-wide sweep for calls and `headers=` arguments |
| B177 | 212, 213 | repo:buster_ripper.routes.chat, #_prepare_headers | one read of `chat.py` |
| B178 | 214, 215, 216, 217 | repo:buster_ripper.routes | one read of the five route-side comprehensions |
| B179 | 218 | repo:buster_ripper.routes.count_tokens#handle_count_tokens | solo — a different unit's absence |
| B180 | 219 | rel:buster_ripper_legacy#_HOP_BY_HOP→buster_ripper/utils#_HOP_BY_HOP | solo — a set-identity relation across two files |
| B181 | 220, 221, 222 | repo:buster_ripper_legacy | one read of the legacy filter helpers |
| B182 | 223, 224, 225 | repo:buster_ripper_legacy | one sweep of the legacy file's call sites plus an import sweep |
| B183 | 226, 227, 228 | pkg:aiohttp@3.13.3 | one read of `web_response.py`'s prepare/`_start` |
| B184 | 229, 230 | pkg:aiohttp@3.13.3 | one read of the server protocol's error path |
| B185 | 231, 232 | pkg:aiohttp@3.13.3 | one read of `client_exceptions.py`'s hierarchy |
| B186 | 233 | pkg:aiohttp@3.13.3 | solo — a different class's write path |
| B187 | 234, 235 | pkg:aiohttp@3.13.3 | one read of the client request-header defaults |
| B188 | 236 | pkg:aiohttp@3.13.3 | solo — the server response's own header defaults |
| B189 | 237, 238 | pkg:aiohttp@3.13.3 | one read of the case-insensitive multidict's iteration |
| B190 | 239, 240 | lang:python | one read of the stdlib `json` docs and `str`'s API |
| B191 | 241, 242, 243, 245 | repo:buster_ripper.routes.messages#_stream_single_request | one read of the handler body |
| B192 | 244 | repo:buster_ripper.utils#find_message_start | solo — a different unit, deliberately not in B174 (that bundle's freshness trigger is `utils.py`'s header defs, which this wave edits; this one is not) |
| B193 | 246, 247, 248 | repo:tests | one sweep of Python test sources under `tests/` |
| B194 | 249 | repo:buster_ripper | solo — a directory listing |
| B195 | 250, 251, 252 | repo:REFACTORING_PLAN.md | one read of the document plus a tracked-name sweep |
| B196 | 67, 73 | repo:buster_ripper.routes.messages#handle_streaming, #handle_messages | the two re-dispatched S9 members — one read of `messages.py` |

No bundle crosses a library and a repo pattern or two libraries (rule 1); none exceeds four rows (rule 3); every `B*` key above is new, and re-dispatched rows are re-bundled **among themselves** — B40 and B48 are history, so no checker receives a claim it already answered. Solo is the default, not padding (rule 4). **23 bundles ⇒ 23 checker agents**, of which one (B196) carries only rows already on file.

## Run 30 freshness basis and evidence lookups

**Hashes establish scope change only — never whether a proposition still holds** (§3d/§3e). Read this run at HEAD `bcf596ec…`: `buster_ripper/routes/messages.py` = `sha256:721a2afe4d1ca35da9461c95f5c6dfe48dcf04ec4c4c0a606763cc29b0ca96da`; `buster_ripper/utils.py` = `sha256:dd60987f7ef0741e55712e0469e854500a80318dedc70c130645ccb51bbc0d00`; `buster_ripper/routes/chat.py` = `sha256:16c7e99938b55e508b36c16092e7a89883baeff293385c3dfa3e9c161b413cae`; `buster_ripper/routes/passthrough.py` = `sha256:072a58096586fcb3cf94c3cde37357bcee22abdffd8d89ae5952cfc4437494b2`; `REFACTORING_PLAN.md` = `sha256:f7562fa81f1b1e8a1deb4ae9fe1bc135752ed1e7cc9fe06f3494a53b78699aa7`.

**The S9 rider, on current bytes.** `verdicts/claim-67.md` and `verdicts/claim-73.md` both pin `## Source Fingerprint sha256:9c234c4f183aa03a80ab82356a6fdfaad0220f3bef69edf3bab5789ef808277d`, and both name `## Invalidates On: Any change to buster_ripper/routes/messages.py`. The file is now `721a2afe…`; the single change since those verdicts was written by `67ced13` ("Type undecodable JSON client errors consistently"), which edited `handle_messages` at line 28 — inside row 73's named unit, and inside the file row 67's own invalidation condition names. Separately, this wave edits the file again on both sides: F-4 rewrites `messages.py:62-63` (inside `handle_messages`, i.e. row 73's unit and `seam-S9.md`'s §Sources range `messages.py:65-70,73-85`) and G-1 rewrites the relay section (inside row 67's cited range `:135-176` and the rider's named `:121-176`). So **freshness cannot be established for either row from the record on file** (§3e, third row of the table), and both move `HOLD`/`none` → `recheck`/`unverified` in B196 with their `CL:` handles retained — a re-dispatch, not a reversal. Row 67 additionally gains `S11` membership: it is the row that names the committed `StreamResponse` whose head the second-head premises are about. Row 73 is named by this rider although the dispatch named row 67, because it is the other S9 member pinned to the same expired fingerprint and the same wave edits both units; §3c fires a seam on a member change, so S9's State stays `recheck` either way and no additional seam is disturbed by including it. `chat.py` is byte-identical to the fingerprint rows 145–148 and 149–152/155/157–158 pin (`16c7e999…`), so the chat-side S9 members are **not** re-dispatched: F-4's `chat.py:101-102` substitution is planned, not landed, and none of those rows' units (`_make_chat_request`, `_relay_chat_stream`, `_apply_response_transforms`) is the `_prepare_headers` filter site. No other `messages.py`-pinned verdict was swept (§3e's no-sweep rule): `verdicts/claim-13/-15/-17/-46/-53/-54.md` carry the same `9c234c4f…` digest and belong to S1/S2, which this dispatch does not name — recorded here so the observation is not lost (E7) and left for the owning route's next S1/S2 pass.

**Session-ledger matching (§3d), performed for every row above.** No existing row asserts any proposition in 205–252: the ledger's aiohttp rows are JSON decoding (51), `ContentTypeError`/`ClientError` inheritance (64), response-header exposure and explicit Content-Type (154–156), connector/runner/signal lifecycle (95–104, 113–114, 123–128) and per-request timeouts (58, 141) — no row about response heads, payload writers, `ClientPayloadError`, header-name spelling or header filtering. Its `repo:` rows cover the dump chain, the CLI/config surface, the serve script, the TLS surface and the chat/messages stream dispatch — no row on `utils.py`'s header helpers, the route-level skip sets, `buster_ripper_legacy.py`, or `REFACTORING_PLAN.md`. So 48 rows are `new`/`unverified`, and **one row is inherited rather than re-extracted**: **78** (`asyncio.CancelledError` is not a subclass of `Exception`) already asserts exactly what the G-1 case table's fourth row takes for granted, no freshness trigger fires (`lang:python`, and the wave edits no interpreter), so its `HOLD`/`none` cells and `B110` key are preserved and `T013`/`S11` are appended — answered by the verdict already on file, and the task is now recorded against that row so *"do other tasks depend on this"* stays a lookup. Rows **64** and **149/150/151** were considered for S11 membership and left out: their propositions are the chat path's, and this wave's S11 chain is the messages path's.

**Registry `## Claim Index` lookups, performed for these propositions only.** 175 `CL:` records read; **no match for any of the 48 new rows** — the aiohttp set has no `prepare`/payload-writer/second-head/header-spelling/`ClientPayloadError`/timeout-class record (`CL:aiohttp/content-type-error-inheritance` is a different exception; `CL:aiohttp/per-request-timeout-override` is what a `timeout=` argument does, not what class its error is); there is no `CL:` record naming `utils.forward_headers`, `utils.response_headers`, `_HOP_BY_HOP`, any `routes/*.py` header filter, `buster_ripper_legacy.py`, `find_message_start`, `REFACTORING_PLAN.md`, a `convert/` path, or RFC 9110; and the `lang:python` set has no `str`-attribute or `json.loads`-return-type record (`CL:python/json-decode-error-inheritance` is the exception hierarchy). So `i = 0` from the registry and **no `Permanent` handle is invented** for a new row (extraction never creates one). The one inheritance is corroborated there: row 78's `CL:python/cancelled-error-inheritance` is the registry row, unchanged.

**`## Composition Index` consulted for S9's exact ordered member set only** (no unrelated composition inspected for staleness): the index carries `CO:aiohttp-runner-to-on_cleanup/lifecycle-teardown`, `CO:cli-flag-to-outbound-call/tls-posture-crossing`, `CO:cli-option-to-dump-dispatch/format-value-crossing` and `CO:serve-script-env-to-argv/env-var-flag-crossing` — **no record for S9's set**, and none for a header-parity or response-head boundary, so S10 and S11 are `new` with no handle to carry into `Permanent`, and S9 keeps `Permanent: —` as recorded. `assumptions/compositions/seam-S9.md`'s own `## Invalidates On` names "Any change to a listed member verdict/fingerprint or the production boundaries: messages stream dispatch/raw relay…", which both this wave's edits and the expired `9c234c4f…` pin fire — recorded, not decided: whether S9 still composes is the composition pass's call (E8), and S9 is **not citable as current** until B196 returns and it is recomposed, which is the rider the plan row names.

## Two directions (§1), as rows not judgments

**The wave consumes what the peers have.** F-4's five substitutions are constrained by families the plan does not restate: the helpers' own definitions (205/206) and the eight-name set (207, 219) are the target shape; `chat.py`'s `_prepare_headers` empty-`Bearer` deletion (213) is the route-local behavior the eval contract depends on and which the shared helper must not absorb — the base file carries the same asymmetry (221, 224), which is the precedent the closure cites; 234/235/236 are what makes dropping `host`/`content-length` safe rather than merely different; 237/238 are what makes lowercase-membership filtering correct against a case-insensitive transport; and 225 is what licenses reading `buster_ripper_legacy.py` as a reference copy at all. **The peers consume what the wave introduces:** `chat.py`'s stale import becomes a live caller (212 → contradicted by 209 the moment F-4 lands, which is the point of checking 209 now), the five local copies stop being the policy owners so 214/215/216/217 become statements about `utils.py` alone, and 216 is the row the "rename-proof" site check rests on — if any filter in that family does not iterate `.items()`, the zero-match grep is not exhaustive and a sixth boundary survives the fix. On the G-1 side, the post-commit `except Exception` the task must write is bounded by 239/240 (a `str` in an upstream payload is reachable *without* monkeypatching, so the class of post-commit failures is not `ClientError`-shaped), by 231/232 (which failures today's single arm already catches), and by 78 (which one it must never catch); rows 226–230 are the library half of why the *return* is the defect rather than the log line, and 248 is the row that says nothing on the bench currently holds either side of the commit boundary.

## Collected and deliberately NOT rowed — routes recorded, nothing silent (E7)

- **`pyproject.toml:30-34` omits `buster_ripper/routes/*` from coverage** — read this run (`omit` = routes, `main.py`, `server.py`). It explains *why the drift survived the gate*, which is the packet's diagnosis, and §2's exclusion table puts coverage posture with the QA gate, not with an assumption a blind checker answers. Route: `handoff-context.md` §Owning route and the QA plan.
- **`tests/qa/tls_peer.py`'s two `keep-alive` comment mentions** — saw them while establishing B193's universe; they are prose in a `norecursedirs`-excluded QA module, not assertions, and naming them in a row would put a test-run boundary inside an absence claim.
- **Whether the fix preserves pre-refactor semantics / whether the five substitutions close the regression / whether a post-commit `except Exception` is the right handler** — design and correctness, i.e. E3/E1/E8. The task's `Required behavior`, its per-member `Done when` and its `Yield replay` clauses already carry these, and the arbiter for "the committed headers appear once" is a replay, not a doc read.
- **Suite passage, test counts, coverage percentages, `QAP-MS004` re-run, the 057 finding dispositions, the `DC-F-4-r2`/`DC-G-1-r2`/`DC-G-2-r1` closure ledger rows, and the wave's execution order and gating status** — session governance and arbiter state (E4), checkable only by reading this session directory, which would spend the checker's blindness.
- **The two proposed invariant candidates in the T013 block** — non-authoritative by their own label ("Interview's to record or promote"). This pass creates no invariant and promotes nothing; row 208 is the factual hook the first candidate leans on, and it is an ordinary `rel:` claim, not an invariant.
- **`_prepare_headers`'s case-insensitive `Authorization` lookup** (`chat.py:105`) — the same mechanism row 237 settles for the transport; rowing it twice would put one proposition in two rows.
- **Whether `git` still tracks `buster_ripper_legacy.py`** — the file's tracked-ness is arbiter-side state; row 225 states the structural relation (nothing imports it) that any route needs.

No claim was checked, no composition performed, no design claim written, no row compound beyond one proposition, nothing filtered for risk, and no seam — new or inherited — was called safe.

---

# Run 31 — scoped adhoc extraction, T014 · image-slice wave (2026-10-05)

Dispatch: **scoped adhoc extraction**, mode `adhoc`. Required ledger delta named by the dispatcher: **T014** — member **F-1** (`has_images` detector scope across all user messages), member **F-2** (content-addressed placeholder keys on `block["source"]`, with `063 R-A`'s canonical-serialization binding), member **F-3** (write-path tail global sweep with `R-C`'s absolute-age TTL semantics) — **plus `R-B`'s W1–W4 placement witnesses and `065-forward-red`'s W5 recommendation** — over `buster_ripper/image.py`, the `routes/messages.py` call sites, `normalize.py`'s step ordering and the `tests/test_image.py` pins. Extraction only: this pass checked nothing, composed nothing, and called no seam safe (E1/E8).

**Numbering.** The claims ledger's sequence ended at run 30's row 252; this pass takes **31** in both ledgers and its rows run **253–319**, with bundle keys from **B197** (B1–B196 are history).

**Run basis.**
- **Blueprint.** `plan-summary.md:392` is a pointer; the single blueprint is `handoff-context.md` §Planned Task Context → `T014 Planned Task Context — Make --strip-images cover the whole request (057 F-1 + F-2 + F-3)`, carrying `063-decide`'s inline repairs `R-A`…`R-E`, as re-attacked by [065-forward-red](../iterations/065-forward-red.md) (`0 AMBIGUOUS`, nine footnotes). The wave's three *invariant candidates* are labelled non-authoritative by their own text and are Interview's to record, so §2 excludes them as invariants; the factual hooks they lean on are rowed here as ordinary rows (e.g. 253/256 carry the "every image block in every user message" candidate's two halves).
- **The `new-surface` claim the task row asserts is discharged by this pass, not assumed.** `buster_ripper/image.py` matched nothing in `claims.md` except row 48's enumeration of CLI-wired numeric globals, and nothing in `seams.md`; no `CO:` record in the registry's `## Composition Index` names an image boundary. So rows 253–319 are the wave's first coverage rather than an addition to an existing set — which is why no row here is marked `inherited` on image-surface grounds.
- **`R-B`'s rationale was handed to this route by name, and it is rowed, not verified.** 065 row 4 records the block's sentence "`cleanup_expired_images()` … is entirely caller-less — so this is a wiring fix, not a new mechanism" as *false as written* in its second half and directs "Forward Collect checks the claim against `image.py:176-179` at extraction time". This pass does not check (E1). It writes the two atoms whose conjunction the rationale asserts — **299** (what the primitive actually does) and **295** (what the inner function never does), with **300**/**301** for the caller-less half — so a blind checker settles them and the *addition* half of the drop is left as the task's `Required behavior`, where it belongs.
- **Package/dependency binding.** Every member lands in the single package `buster_ripper` (`pyproject.toml`). Runtime deps unchanged: `aiohttp`, `click`; installed identities read this run from `.venv/lib/python3.13/site-packages`: **aiohttp 3.13.3**, **click 8.3.1** (Python 3.13). Every `pkg:` row carries `pkg:aiohttp@3.13.3`; stdlib facts are `lang:python`, which is the correct subject for `json`, `hashlib`, `time` and dict/tuple semantics.
- **Peer set found by listing, not judgment** (§1): `buster_ripper/image.py`'s peers are the two other module-level per-session mappings in the package — `buster_ripper/dump.py`'s `_sessions` and `buster_ripper/session.py`'s `_stats` (rows 318/319, the direction the plan does not describe: the wave gives one of three a reclamation path); `repo:buster_ripper.session#session_id` is the digest-shape precedent for `R-A`'s formula (row 286); `tests/test_routes_integration.py`'s `TestClient` fixture is the peer harness a "route-level witness" would be written in (row 317); `buster_ripper/normalize.py` is the second owner of the same guarded call (rows 264–268).
- **Universe of discourse for every absence/exhaustive row:** `buster_ripper/` means the package tree; `buster_ripper_legacy.py` (repo root, referenced by nothing) is excluded except where its own subject names it; test sweeps say **Python test sources under `tests/`** (the `tests/qa/` gates are `norecursedirs`-excluded and `tests/__pycache__` holds a stale artifact of a deleted module).
- **`user-claim` route:** `decisions.md` carries no `Verification: assumptions PENDING` factual claim bearing on this wave (its only `PENDING` is `yield-phase:MS-001`, an arbiter route) and `063-decide` records **0 user forks**, so there is **no `user-claim` row and no `MIGRATION-REQUIRED` return**: the T014 block is a closure-packet artifact, and every source-checkable proposition inside it is an ordinary atomic row below.
- **HEAD at extraction:** `bcf596ec2066fe8432abf4987bf575d695524d01` — the head 065 read, so the block's code citations are current at the head Forward will be dispatched against. `git status` shows only session artifacts modified (run 30's ledger edits are uncommitted); no file in `buster_ripper/` or `tests/` differs from HEAD.

## Run 31 claims

| # | Claim | Subject | Tasks | Scope | Seams | Check | Bundle | State | Routing | Permanent |
|---|---|---|---|---|---|---|---|---|---|---|
| 253 | `has_images` inspects only the first user message it meets when scanning `messages` from the end | repo:buster_ripper.image#has_images | T014 | single | S12 | seam:S12 | — | superseded | none | — |
| 254 | `has_images` returns False when the body has no `messages` key or an empty list | repo:buster_ripper.image#has_images | T014 | single | S12 | seam:S12 | B197 | HOLD | none | — |
| 255 | `_has_image_in_blocks` reports an image only for a mapping block whose `type` is `image` | repo:buster_ripper.image#_has_image_in_blocks | T014 | single | S12 | seam:S12 | B197 | HOLD | none | — |
| 256 | `strip_and_cache_images` processes every message whose role is `user`, in list order | repo:buster_ripper.image#strip_and_cache_images | T014 | single | S12 | seam:S12 | B198 | UNCLEAR | malformed | — |
| 257 | `strip_and_cache_images` leaves messages whose role is not `user` untouched | repo:buster_ripper.image#strip_and_cache_images | T014 | single | S12 | seam:S12 | B198 | HOLD | none | — |
| 258 | `strip_and_cache_images` skips a user message whose `content` is not a list | repo:buster_ripper.image#strip_and_cache_images | T014 | single | S12 | seam:S12 | B198 | HOLD | none | — |
| 259 | `strip_and_cache_images` returns the body before its message loop when `messages` is absent or empty | repo:buster_ripper.image#strip_and_cache_images | T014 | single | S12, S14 | seam:S12/S14 | B198 | HOLD | none | — |
| 260 | No path in `buster_ripper/image.py` inspects a `tool_result` content block for nested images | repo:buster_ripper.image | T014 | single | S12 | absence, seam:S12 | B199 | HOLD | none | — |
| 261 | `handle_messages` calls `strip_and_cache_images` only when `config.STRIP_IMAGES` and `has_images(body)` are both true | repo:buster_ripper.routes.messages#handle_messages | T014 | single | S12 | seam:S12 | B200 | HOLD | none | — |
| 262 | `handle_messages` appends its image-strip change string inside the same guarded branch as the writer call | repo:buster_ripper.routes.messages#handle_messages | T014 | single | S12 | seam:S12 | B200 | HOLD | none | — |
| 263 | No `try` block in `handle_messages` covers its guarded image-strip call | repo:buster_ripper.routes.messages#handle_messages | T014 | single | S14 | absence, seam:S14 | B200 | HOLD | none | — |
| 264 | `normalize`'s image step runs only when the caller passes a truthy `session_id` argument | repo:buster_ripper.normalize#normalize | T014 | single | S12 | seam:S12 | B201 | HOLD | none | — |
| 265 | `normalize` appends its image-strip change string inside that same guarded step | repo:buster_ripper.normalize#normalize | T014 | single | S12 | seam:S12 | B201 | HOLD | none | — |
| 266 | `normalize` runs `_normalize_cache_control` after its image step | repo:buster_ripper.normalize#normalize | T014 | single | S13 | seam:S13 | B201 | HOLD | none | — |
| 267 | `_normalize_cache_control` removes `cache_control` keys from user message content blocks | repo:buster_ripper.normalize#_normalize_cache_control | T014 | single | S13 | seam:S13 | B201 | HOLD | none | — |
| 268 | The only `normalize` call under `buster_ripper/` passes an empty `session_id` | repo:buster_ripper | T014 | single | S12 | absence, seam:S12 | B202 | HOLD | none | — |
| 269 | The `--strip-images` option's help text describes stripping images from user messages | repo:buster_ripper.main#strip-images-option | T014 | single | S12 | seam:S12 | B203 | HOLD | none | — |
| 270 | `main.py` assigns `config.STRIP_IMAGES` from the parsed `--strip-images` flag | repo:buster_ripper.main#strip-images-option | T014 | single | S12 | seam:S12 | B203 | HOLD | none | — |
| 271 | `config.STRIP_IMAGES` defaults to False | repo:buster_ripper.config#STRIP_IMAGES | T014 | single | S12 | seam:S12 | B203 | HOLD | none | — |
| 272 | `_replace_image_with_placeholder` derives its cache key from the session mapping's current entry count | repo:buster_ripper.image#_replace_image_with_placeholder | T014 | single | S13 | seam:S13 | — | superseded | none | — |
| 273 | `_replace_image_with_placeholder` stores the block's `source` value as the cache entry's payload | repo:buster_ripper.image#_replace_image_with_placeholder | T014 | single | S13 | seam:S13 | B204 | UNCLEAR | malformed | — |
| 274 | `_replace_image_with_placeholder` returns a `type` text block whose `text` embeds the cache key | repo:buster_ripper.image#_replace_image_with_placeholder | T014 | single | S13 | seam:S13 | B204 | HOLD | none | — |
| 275 | `_replace_image_with_placeholder` calls `_evict_by_lru` after inserting the entry | repo:buster_ripper.image#_replace_image_with_placeholder | T014 | single | S14 | seam:S14 | B204 | HOLD | none | — |
| 276 | A second write under an existing image key replaces that key's stored entry | repo:buster_ripper.image#_replace_image_with_placeholder | T014 | single | S14 | seam:S14 | B205 | HOLD | none | — |
| 277 | The cache entry holds a reference to the request body's own `source` mapping, not a copy | repo:buster_ripper.image#_replace_image_with_placeholder | T014 | single | S13 | absence, seam:S13 | B205 | HOLD | none | — |
| 278 | `buster_ripper/image.py` imports no `buster_ripper` module other than `config` | repo:buster_ripper.image | T014 | single | — | — | — | HOLD | none | — |
| 279 | `json.dumps` with `sort_keys` enabled yields identical text for two equal mappings built in different insertion orders | lang:python | T014 | single | S13 | native, seam:S13 | B206 | HOLD | none | — |
| 280 | `json.dumps`'s `sort_keys` option sorts nested mappings at every depth | lang:python | T014 | single | S13 | native, seam:S13 | B206 | HOLD | none | — |
| 281 | `json.dumps` with a two-item `separators` tuple emits no whitespace between items or after the key separator | lang:python | T014 | single | S13 | native, seam:S13 | B206 | HOLD | none | — |
| 282 | `json.dumps` raises `TypeError` for a value it cannot serialize | lang:python | T014 | single | S13 | seam:S13 | B206 | HOLD | none | — |
| 283 | `hashlib.sha256` raises `TypeError` when passed a `str` rather than a bytes-like object | lang:python | T014 | single | S13 | seam:S13 | B207 | HOLD | none | — |
| 284 | A `hashlib.sha256` hexdigest is identical across processes for identical input bytes | lang:python | T014 | single | S13 | seam:S13 | B207 | HOLD | none | — |
| 285 | `hashlib.sha256`'s hexdigest is 64 lowercase hexadecimal characters | lang:python | T014 | single | S13 | seam:S13 | B207 | HOLD | none | — |
| 286 | `session_id` derives its value from the first twelve characters of a `hashlib.sha256` hexdigest | repo:buster_ripper.session#session_id | T014 | single | S13 | seam:S13 | B208 | HOLD | none | — |
| 287 | `get_cached_image` has no caller under `buster_ripper/` | repo:buster_ripper | T014 | single | S13 | absence, seam:S13 | B209 | HOLD | none | — |
| 288 | Every positive placeholder assertion in `tests/test_image.py` pins a literal `img_` key | repo:tests.test_image | T014 | single | S13 | absence, seam:S13 | — | superseded | none | — |
| 289 | Every `strip_and_cache_images` call in `tests/test_image.py` adds one image to a freshly reset cache | repo:tests.test_image | T014 | single | S13 | absence, seam:S13 | B210 | DOES-NOT-HOLD | absorbed | — |
| 290 | `tests/test_image.py` asserts `has_images` is False for the earliest-image, latest-text-only body | repo:tests.test_image | T014 | single | S12 | seam:S12 | — | superseded | none | — |
| 291 | `tests/test_image.py`'s eviction cases cache a distinct `source` mapping per image | repo:tests.test_image | T014 | single | S13 | seam:S13 | B210 | HOLD | none | — |
| 292 | One `tests/test_image.py` case asserts `get_cached_image` returns None for a key that was never cached | repo:tests.test_image | T014 | single | S13 | seam:S13 | B211 | HOLD | none | — |
| 293 | `tests/test_image.py` asserts TTL expiry only through a `get_cached_image` call | repo:tests.test_image | T014 | single | S14 | absence, seam:S14 | — | superseded | none | — |
| 294 | An autouse `tests/test_image.py` fixture clears `_image_cache` before and after each test | repo:tests.test_image | T014 | single | S14 | seam:S14 | B211 | HOLD | none | — |
| 295 | `_cleanup_expired_images` removes no entry from the top-level `_image_cache` mapping | repo:buster_ripper.image#_cleanup_expired_images | T014 | single | S14 | absence, seam:S14 | B212 | HOLD | none | — |
| 296 | `_cleanup_expired_images` compares a cache entry's second element against `config.IMAGE_CACHE_TTL` | repo:buster_ripper.image#_cleanup_expired_images | T014 | single | S14 | seam:S14 | B212 | HOLD | none | — |
| 297 | `_cleanup_expired_images` treats an entry as expired only when its age strictly exceeds the configured TTL | repo:buster_ripper.image#_cleanup_expired_images | T014 | single | S14 | seam:S14 | B212 | HOLD | none | — |
| 298 | `_cleanup_expired_images` reads `config.IMAGE_CACHE_TTL` at call time rather than binding it at import | repo:buster_ripper.image#_cleanup_expired_images | T014 | single | S14 | seam:S14 | B212 | HOLD | none | — |
| 299 | `cleanup_expired_images` calls `_cleanup_expired_images` once per session and performs no other operation | repo:buster_ripper.image#cleanup_expired_images | T014 | single | S14 | seam:S14 | — | superseded | none | — |
| 300 | `cleanup_expired_images` has no caller under `buster_ripper/` or `tests/` | repo:buster_ripper | T014 | single | S14 | absence, seam:S14 | — | superseded | none | — |
| 301 | `clear_session_cache` has no caller under `buster_ripper/` | repo:buster_ripper | T014 | single | S14 | absence, seam:S14 | B214 | HOLD | none | — |
| 302 | `_evict_by_lru` removes entries until the session mapping holds at most `config.IMAGE_CACHE_MAX_SIZE` entries | repo:buster_ripper.image#_evict_by_lru | T014 | single | S14 | seam:S14 | B215 | HOLD | none | — |
| 303 | `_evict_by_lru` orders eviction by the cache entry's second element | repo:buster_ripper.image#_evict_by_lru | T014 | single | S14 | seam:S14 | — | superseded | none | — |
| 304 | `_evict_by_lru` reads `config.IMAGE_CACHE_MAX_SIZE` at call time rather than binding it at import | repo:buster_ripper.image#_evict_by_lru | T014 | single | S14 | seam:S14 | B215 | HOLD | none | — |
| 305 | `_evict_by_lru` limits its work to one session's mapping | repo:buster_ripper.image#_evict_by_lru | T014 | single | S14 | absence, seam:S14 | B215 | HOLD | none | — |
| 306 | `_image_cache` is module-level state shared by every request handled in one process | repo:buster_ripper.image | T014 | single | S14 | seam:S14 | B216 | HOLD | none | — |
| 307 | `get_cached_image` indexes `_image_cache` by session after calling `_cleanup_expired_images` for that session | repo:buster_ripper.image#get_cached_image | T014 | single | S14 | seam:S14 | B216 | HOLD | none | — |
| 308 | `get_cached_image` unpacks its cache entry into exactly two targets | repo:buster_ripper.image#get_cached_image | T014 | single | S14 | seam:S14 | — | superseded | none | — |
| 309 | Assigning to a key a `dict` already holds does not increase the dict's size | lang:python | T014 | single | S13 | seam:S13 | B217 | HOLD | none | — |
| 310 | Unpacking a three-element tuple into two targets raises `ValueError` | lang:python | T014 | single | S14 | seam:S14 | B218 | HOLD | none | — |
| 311 | `time.time` reflects the system wall clock and can return a smaller value after a clock adjustment | lang:python | T014 | single | S14 | seam:S14 | B219 | HOLD | none | — |
| 312 | The Python standard library provides no TTL or LRU cache container | lang:python | T014 | single | — | native, absence | B220 | UNCLEAR | malformed | — |
| 313 | aiohttp runs each request handler as a coroutine on the application's event loop, so a synchronous section of a handler cannot interleave with another handler | pkg:aiohttp@3.13.3 | T014 | single | S14 | native, seam:S14 | B228 | HOLD | none | — |
| 314 | `logging_middleware` awaits the handler without catching exceptions | repo:buster_ripper.server#logging_middleware | T014 | single | S14 | seam:S14 | B222 | HOLD | none | — |
| 315 | `create_app` registers `logging_middleware` as its only middleware | repo:buster_ripper.server | T014 | single | S14 | absence, seam:S14 | B222 | HOLD | none | — |
| 316 | No Python test source under `tests/` sets `config.STRIP_IMAGES` true | repo:tests | T014 | single | S12 | absence, seam:S12 | — | superseded | none | — |
| 317 | `tests/test_routes_integration.py` posts to `/v1/messages` through an aiohttp `TestClient` with the upstream session mocked | repo:tests.test_routes_integration | T014 | single | S12 | native, seam:S12 | B223 | HOLD | none | — |
| 318 | `buster_ripper/dump.py`'s `_sessions` mapping is never pruned by expiry or by size | repo:buster_ripper.dump#SessionState | T014 | single | — | absence | B224 | HOLD | none | — |
| 319 | `buster_ripper/session.py`'s `_stats` mapping is never pruned by expiry or by size | repo:buster_ripper.session | T014 | single | — | absence | B225 | HOLD | none | — |

**Row-shape notes (E5 — where the blueprint's sentences carry two facts).** The block states several premises as one clause, and they are split: "`_cleanup_expired_images` is reached only from `get_cached_image`, and the two public janitors have zero non-test callers" is **307** and **300/301** (a source read and a grep sweep); "the third and fourth images both get `img_2` … and the key is re-derived per request" is **272** plus **276** (the derive-from-count rule and the overwrite-on-repeat rule are what make both symptoms possible, and a checker answers them from different lines); "`_image_cache` keys by `session_id`, which is `sha256(system)[:12]`, so each distinct system prompt adds a key that is never reclaimed" is **11** (inherited), **286** and **295**; "`first_seen`, which TTL compares, and `last_touched`, which `_evict_by_lru` sorts on" presupposes today's single stamp, which is **296** and **303** — the two sites a 3-tuple must be reconciled with, and the reason **308**/**310** are in the ledger (065 row 7: an unpacking site the shape change silently breaks). The clause "`has_images` returns False for an empty `messages` list … and both sites guard on it" is **254**, **261** and **264** — three units, three sources. The rationale that 065 row 4 handed to this route is deliberately **not** one row: "already exists" and "caller-less" and "performs no drop" are three propositions answered by reading `:176-179`, sweeping for callers, and reading `:47-59` respectively (**299**, **300**, **295**).

**Rows kept although unsurprising (E6).** **254**, **271**, **309** and **311** look inert and are the load-bearing kind: the first makes the empty-body branch unreachable (065 row 5), the second is what makes the whole wave non-user-visible, the third is the entire mechanism of F-2's "same `source` twice → one cache entry" case, and the fourth is what `R-C`'s "absolute age" binds the TTL to a wall-clock subtraction rather than a monotonic one. **283** is included for the same reason: `R-A`'s formula as written hands `hashlib.sha256` a `str`, so the row decides whether the blueprint's own text is executable or needs the encode step 065's footnote F1 half-supplies.

**The dispatch's named witnesses, mapped to rows (no verdict implied).** `R-B`'s four placement witnesses and `065`'s W5 recommendation are acceptance sets, not premises (E3/E4), so they are not rows; the atomic facts each one can only produce by composing are: **W1** same-session expiry → **276** (a same-key write replaces the entry), **296**/**297**/**298** (what expiry compares, how strictly, and when the limit is read), **302**/**304**/**305** (what the size trim does to the same mapping); **W2** abandoned-session reclaim → **299** (the janitor's actual body), **295** (the inner primitive drops nothing at the top level), **306** (one mapping for every session), **300** (today it is called by nobody); **W3** no-crash regression → **263** (no `try` covers the guarded call), **314**/**315** (nothing between the handler and aiohttp catches), **230** (inherited — a raise produces aiohttp's own error response), **313** (whether another handler can be inside the same mapping); **W4** read path untouched → **293**/**294** (the pins that break if the read path gains a delete), **307**/**308** (the index-after-cleanup shape and the unpack the tuple change reaches), **310** (the arity failure itself); **W5**, the residue 065 recommends adding → **275** (the sweep's precedent position relative to the write), **276** + **296** + **311** (the three facts that make "an expired entry refreshed instead of re-created" observable at all: the write happens under an existing key, the comparator reads the stamp the write may have just moved, and the stamp is wall-clock), plus **259**/**254** (the one body shape where a tail sweep never runs). Whether these compose to the required behavior is the composition pass's to determine (E8); this pass records the chain and stops.

## Run 31 bundles (fresh keys — B1–B196 are history; one agent per bundle)

| Bundle | Rows | Subject(s) | Why tied |
|---|---|---|---|
| B197 | 253, 254, 255 | repo:buster_ripper.image#has_images, #_has_image_in_blocks | one read of the detector section of `image.py` |
| B198 | 256, 257, 258, 259 | repo:buster_ripper.image#strip_and_cache_images | one read of the writer's message loop |
| B199 | 260 | repo:buster_ripper.image | solo — a module-wide shape sweep for `tool_result` |
| B200 | 261, 262, 263 | repo:buster_ripper.routes.messages#handle_messages | one read of the handler's guarded branch and its `try` structure |
| B201 | 264, 265, 266, 267 | repo:buster_ripper.normalize#normalize, #_normalize_cache_control | one read of `normalize.py`'s step sequence |
| B202 | 268 | repo:buster_ripper | solo — a package-wide call-site census, not a file read |
| B203 | 269, 270, 271 | repo:buster_ripper.main#strip-images-option, repo:buster_ripper.config#STRIP_IMAGES | the two ends and the help text of one control |
| B204 | 272, 273, 274, 275 | repo:buster_ripper.image#_replace_image_with_placeholder | one read of the placeholder unit |
| B205 | 276, 277 | repo:buster_ripper.image#_replace_image_with_placeholder | same unit; cap of four forces the split |
| B206 | 279, 280, 281, 282 | lang:python | one read of the stdlib `json` documentation |
| B207 | 283, 284, 285 | lang:python | one read of the stdlib `hashlib` documentation |
| B208 | 286 | repo:buster_ripper.session#session_id | solo — the in-repo digest-shape precedent |
| B209 | 287 | repo:buster_ripper | solo — a package sweep for read-path callers |
| B210 | 288, 289, 290, 291 | repo:tests.test_image | one read of `tests/test_image.py`; cap forces the split with B211 |
| B211 | 292, 293, 294 | repo:tests.test_image | same file, remaining pins |
| B212 | 295, 296, 297, 298 | repo:buster_ripper.image#_cleanup_expired_images | one read of the expiry primitive |
| B213 | 299 | repo:buster_ripper.image#cleanup_expired_images | solo — the global janitor's body |
| B214 | 300, 301 | repo:buster_ripper | one package-plus-tests sweep for the two janitors' callers |
| B215 | 302, 303, 304, 305 | repo:buster_ripper.image#_evict_by_lru | one read of the eviction primitive |
| B216 | 306, 307, 308 | repo:buster_ripper.image, #get_cached_image | one read of the module state and the read path |
| B217 | 309 | lang:python | solo — one `dict` assignment semantic |
| B218 | 310 | lang:python | solo — unpacking arity |
| B219 | 311 | lang:python | solo — the `time` module's clock source |
| B220 | 312 | lang:python | solo — a stdlib-wide absence |
| B221 | 313 | pkg:aiohttp@3.13.3 | solo — the library's own dispatch model |
| B222 | 314, 315 | repo:buster_ripper.server#logging_middleware, repo:buster_ripper.server | one read of the middleware definition and its registration |
| B223 | 316, 317 | repo:tests, repo:tests.test_routes_integration | one sweep of `tests/` for the flag and for the route harness |
| B224 | 318 | repo:buster_ripper.dump#SessionState | solo — a peer module's reclamation absence |
| B225 | 319 | repo:buster_ripper.session | solo — the other peer's reclamation absence |

No bundle crosses a library and a repo pattern or two libraries (rule 1); none exceeds four rows (rule 3); every key is new, and no row already carrying a verdict is placed in a bundle with a fresh one (rule 2's blindness cost). **29 bundles ⇒ 29 checker agents.** Row 278 is the only new row with `Check: —` and therefore no bundle; row 230 is not dispatched by this pass — see the inheritance note below.

## Run 31 freshness basis and evidence lookups

**Hashes establish scope change only — never whether a proposition still holds** (§3d/§3e). Read this run at HEAD `bcf596ec…`: `buster_ripper/image.py` = `sha256:611df5acd7bf061242084de992a7d84102e0645a5c4b023e0d350841aa32dca7`; `tests/test_image.py` = `sha256:b603bfd1624e30eab1c6d57390194a71cfd3716525abc33c742292bc5a3dfda0`; `buster_ripper/normalize.py` = `sha256:a7ddbf062ad57951debb654988853a49a4bae549cb900da544fa3cd5a632ffca`; `buster_ripper/routes/messages.py` = `sha256:721a2afe4d1ca35da9461c95f5c6dfe48dcf04ec4c4c0a606763cc29b0ca96da`; `buster_ripper/server.py` = `sha256:a9e49823ea456af9c0076781c4713c956745b21b026dfa768faf947e0b130592`; `buster_ripper/config.py` = `sha256:22348c68d21b91a30e75eeea802d4d390b2bcdabbf974b0ddc743c9354635522`; `buster_ripper/main.py` = `sha256:fd36dc12055f67693d9b852fb15a66aa4a3a7864103ac3de0ee67d9520048d47`; `buster_ripper/session.py` = `sha256:67a66b8aa733b31e982be4922350a7996ebdddb0fe5d391a6129a1cf342a9dff`; `pyproject.toml` = `sha256:c623bdc2824331933f9a02ede763170ef8fa07b5d09be8047dc9b9d85a22e19e`. No `verdicts/claim-*.md` pins `image.py`, `tests/test_image.py`, `normalize.py`, `server.py`'s middleware or the two peer mappings, so nothing in 253–319 could have been on file: all 67 rows are `new` / `unverified`.

**Session-ledger matching (§3d) — three inheritances and one membership, performed row by row.**
- **2** (`pyproject.toml` declares `aiohttp` and `click` as the only runtime dependencies) — `HOLD`, `CL:` handle, and `verdicts/claim-2.md`'s `Invalidates On` names only edits to the `[project] dependencies` key or a new dependency mechanism; this wave's declared surface is `image.py` + `tests/test_image.py`, so no trigger fires. Task appended, `S14` membership appended (the row is the dependency half of the reuse-before-build question for a TTL/LRU mechanism), Scope moves `single → shared`. **No checker spent.**
- **11** (`session_id(body)` derives the session key from the body's system-prompt content alone) — `HOLD`, `CL:` handle, `verdicts/claim-11.md` pins `buster_ripper/session.py` at `67a66b8a…`, which is the current byte state. This is the row F-3's "each distinct system prompt adds an `_image_cache` key" leans on. Task and `S14` membership appended. **No checker spent.**
- **48** (the numeric `config`/CLI partition, whose flagged half names `IMAGE_CACHE_TTL`) — `HOLD`, `CL:` handle, pinned to `config.py` + `main.py`; both unchanged by this wave's surface. It is the row the block cites as *evidence that the TTL control is advertised*, so F-3's premise inherits rather than re-extracts. Task and `S14` membership appended. **No checker spent.**
- **230** (an aiohttp handler that raises produces aiohttp's own error response) — asserted already, so **no new row** and no re-bundle: it is still `new`/`unverified` in run 30's **B184**, queued for the header-parity wave, and this pass only appends `T014` and `S14` so *"do other tasks depend on this"* stays a lookup. It is therefore **not** counted in `i` (no verdict is on file to answer from), and re-dispatching it would spend a second agent on a claim one bundle already carries.

**Registry `## Claim Index` lookups, performed for these propositions only** (175 records; no staleness swept for anything this run did not extract): **no match for any of rows 253–319** — the index holds no `CL:` record naming `image.py` or any of its units, `has_images`, `strip_and_cache_images`, `_image_cache`, `_evict_by_lru`, `_cleanup_expired_images`, `cleanup_expired_images`, `clear_session_cache`, `get_cached_image`, `STRIP_IMAGES`, `normalize`'s step order, `cache_control`, `hashlib`, a `json.dumps` canonicalization, `time.time`, dict-assignment or tuple-unpack semantics, `logging_middleware`, or any `tests/test_image` pin. The nearest adjacency is `CL:rel:buster_ripper.main→buster_ripper.config/numeric-config-cli-wiring`, which is row 48's own record and is reused through that inheritance rather than re-matched. So `i = 3` comes from the session ledger and **no `Permanent` handle is invented** for a new row (extraction never creates one).

**`## Composition Index` consulted for S12/S13/S14's member sets only:** the index holds four records (`aiohttp-runner-to-on_cleanup/lifecycle-teardown`, `cli-flag-to-outbound-call/tls-posture-crossing`, `cli-option-to-dump-dispatch/format-value-crossing`, `serve-script-env-to-argv/env-var-flag-crossing`) — none covers an image, cache-lifecycle or placeholder-identity boundary, so all three seams are `new` with `Permanent: —` and no composer can be skipped by registry reuse.

**No existing seam fires.** The wave's declared surface is `buster_ripper/image.py` and `tests/test_image.py`; `seams.md` shows no seam member unit in either (the sole prior mention of the file was row 48's global enumeration, which the freshness note above keeps current). S1–S11's rows, dimensions and cells are untouched, and the four membership edits above add rows to *new* seams only — no existing seam's member set changes, so none moves to `recheck` on this pass's account.

## Two directions (§1), as rows not judgments

**The wave needs what the surface already has:** the sweep's precedent position is the writer's own post-insert call (**275**); the config globals must be readable at call time or the W-rows cannot inject a TTL (**298**, **304**); the digest shape the block prescribes already exists in `session.py` (**286**), which is the row that decides whether the blueprint's `[:12]` lands on the digest or the operand; the route-level witness F-1 owes has a harness to be written in (**317**) and today has no example (**316**); and the empty-body reachability argument (065 row 5) runs through **254**, **259**, **261** and **264**. **The surface needs what the wave introduces:** a 3-tuple entry breaks every 2-target unpack, which is **308** read against **310**, and the expiry comparator and the eviction sort currently read the *same* element, which is **296** against **303** — R-C's dual stamp is only additive if both keep their own axis. A content-derived key makes **272** false while making **277** (a stored reference, not a copy) newly load-bearing, because the digest is taken at write time from an object the caller still owns. And the global sweep is safe to run against a live `_image_cache` only on the strength of **306** read with **313** — the same read that makes **312** and **2** the reason the mechanism is hand-rolled. Finally, the peers: `dump.py`'s and `session.py`'s per-session mappings (**318**, **319**) are the same shape the wave fixes only in `image.py`, and nothing in the plan mentions them because the plan is about the image surface.

## Collected and deliberately NOT rowed — routes recorded, nothing silent (E7)

- **`pyproject.toml:29-34` omits `buster_ripper/routes/*`, `main.py` and `server.py` from coverage** — read this run while establishing B223's universe. It explains why the guarded call sites in `messages.py` carry no witness today, but coverage posture is a QA gate (§2's exclusion table), and the structural halves of that observation are already rows: **316** and **263**.
- **`tests/test_image.py`'s eviction cases patch `config.IMAGE_CACHE_MAX_SIZE` with `mocker.patch.object`** — the *mechanism* the W-rows would reuse is settled by **298** and **304** (call-time reads); rowing the test's patch idiom as well would put one proposition in two rows.
- **The 065 footnotes F1, F3–F9 as prescriptions** (slice the digest not the operand; place the sweep after the loop and note the empty-body path in the commit message; carry `first_seen` forward unconditionally and add W5; update the unpacks; patch the `time.time` seam rather than sleeping; keep the `img_999` negative; cite findings by producer; commit shape `F-1`, then `F-2`+`F-3`) — every one is a reading of the design or a test-authoring instruction, i.e. E3. Their *factual* ingredients are rowed (**285** for the slice, **254**/**259** for the branch, **276**/**296**/**303** for the stamps, **308**/**310** for the unpacks, **292** for the negative, **316**/**317** for the witness), and W5 itself is the task's acceptance set, not a claim about a library.
- **Suite passage, test counts, coverage percentages, the `DC-F-1-r2`/`DC-F-2-r2`/`DC-F-3-r2` dispositions, the wave's ordering and gating status, and the "Forward is not authorized yet" / `0 AMBIGUOUS` gate state** — session governance and arbiter state (E4), checkable only by reading this session directory, which would spend the checker's blindness.
- **Whether the three members' closures are the right fix, whether content-addressing is better than a counter, whether a tail sweep beats a background task, whether the wave should be one commit** — E1/E3/E8. The block's own `Selected closure`, `Rejected candidates` and `Risk shape` fields carry these with their reasons, and they are dispositions, not premises.
- **The `STRIP_IMAGES` flag's *intent* ("every image block in every user message")** — the invariant candidate is Interview's and non-authoritative by its own label; the two halves a blind checker can answer are rowed as **256** (the writer's scope) and **269** (the help text's words), and the detector's current narrower scope as **253**.
- **`_has_image_in_blocks`'s behavior on a `type: image` block that has no `source` key** — recorded as collected-not-rowed rather than dropped: it is genuinely unwritten territory in the block (F-2's formula reads `block["source"]` and the placeholder path defaults it to `{}`), the answer is settled by reading `image.py:80` (**273**'s unit) so a row would duplicate that read, and *what should happen* is a plan gap for the composition pass and Forward to meet, not a fact this pass can name.
- **`buster_ripper_legacy.py`'s image handling** — excluded from every universe of discourse above by the run-basis rule; no row here is about the pre-refactor monolith, and the sibling wave's rows 219–225 own that file.

No claim was checked, no composition performed, no design claim written, no row compound beyond one proposition, nothing filtered for risk, and no seam — new or inherited — was called safe.


---

# Run 32 — routing dispositions on run-30/31's non-holding rows (2026-10-05, assumption-decide)

Dispatch: **failed-assumption and unsafe-composition routing** over `verdicts/` (runs 30/31) and the four
non-`SAFE` composition records (`compositions/seam-S11.md`, `-S12.md`, `-S13.md`, `-S14.md`). Routing only: this
pass wrote dispositions, cell routing and durable handoff records. It checked nothing, composed nothing, edited no
production file, created no invariant, and promoted nothing. Every disposition was derived through
`invariant-cross-reference` first and then the `codify-deterministic-closure` gate; the full packets are in
[handoff-context.md](../handoff-context.md) §Deterministic Closure Ledger — 2026-10-05 repair round 3 and
§Assumption Routing — 2026-10-05 (claims run 32).

**Numbering.** This is claims run **32** and seams run **32** in both ledgers, keeping run 27's rule that one handle
names one pass across both files.

## Cell writes

| Row | Verdict on file | Routing written | Why |
|---|---|---|---|
| 208 | DOES-NOT-HOLD | `absorbed` | DC-claim-208-r1 → T013/F-4 premise correction; the eight-name target is the repo's own set, not an RFC enumeration |
| 216 | UNCLEAR | `malformed` | definition fork on "header-filtering comprehension" → back to `forward-collect` (run-30 collector, B178's unit) |
| 223 | DOES-NOT-HOLD | `absorbed` | DC-claim-223-r1 → T013/F-4 base-precedent scope (four forwarded requests; the header-less metrics poll is not a proxy boundary) |
| 232 | UNCLEAR | `malformed` | two referents (named timeout classes vs whatever a timeout raises) → back to `forward-collect`; blocks `seam-S11` |
| 237 | DOES-NOT-HOLD | `absorbed` | DC-claim-237-r1 → T013/F-4 rationale correction; re-verify at 3.13.3 with a provisioned source root before promotion |
| 238 | DOES-NOT-HOLD | `absorbed` | DC-claim-238-r1 → same closure on the response half |
| 251 | DOES-NOT-HOLD | `absorbed` | DC-claim-251-r1 → T013/G-2 premise narrowed to code/config; tracked session artifacts are out of scope |
| 256 | UNCLEAR | `malformed` | fork on "processes" (visits vs rewrites) → back to `forward-collect`; blocks `seam-S12` |
| 273 | UNCLEAR | `malformed` | fork on "payload" (element 0 vs the whole tuple); the verdict names its own settling rewording → back to `forward-collect`; blocks `seam-S13` |
| 289 | DOES-NOT-HOLD | `absorbed` | DC-claim-289-r1 → T014/F-2 premise correction; blocks `seam-S13` until the replacement row lands |
| 312 | UNCLEAR | `malformed` | fork on "container" (strict mapping vs loose facility) → back to `forward-collect`; no seam member, so it blocks no composition |
| 278 | HOLD | `none` | settled by reading in **Yield's** lane at [068-01-yield](../iterations/068-01-yield.md) — the row carries `Check: —` so no bundle and no blind checker was spent on it (run-31 §bundles: "Row 278 is the only new row with `Check: —` and therefore no bundle"); Forward read it at write time (068-01 §ledger row 278) and left the cell for Yield, per the rows 195/198 precedent at run 24, which is a cell write plus the iteration record and no separate verdict file. Re-read independently at `d70c876`: the file's every import is `hashlib`, `json`, `time`, `typing.Optional`, `from . import config`, so the only in-package import is still `config`. `hashlib`/`json` are stdlib and are required by `063 R-A`'s own formula, so row 2's runtime-dependency set (`aiohttp` + `click`) is uncontradicted by this wave. Bytes at settle time: `image.py` `sha256:f4894b65af98b731b4098555d1978f4762bc5fd01a313741e3d2d518bef587ef` |

**Three re-dispatches, not reversals.** Rows **2** (`B2` → `B226`), **48** (`B29` → `B227`) and **313** (`B221` →
`B228`) move `HOLD`/`none` → `recheck`/`unverified` on a fired freshness trigger — row 48's own third `Invalidates
On` bullet (a numeric global added to `config.py`) fired through T005/T006, row 2's pinned `pyproject.toml` bytes
moved (`e1b70358…` → `c623bdc2…`), and row 313's consulted bytes are aiohttp `v3.14.1` while its subject and the
`seam-S14` row pin `pkg:aiohttp@3.13.3`. Their propositions are not refuted, their `CL:` handles are retained where
they have one, and the run-30 convention applies: a re-dispatch is not a reversal. **237/238 keep their
`DOES-NOT-HOLD` verdict and `absorbed` routing** — the version gap is a promotion condition, recorded as deferred
re-verification in their ledger entries, not a trigger this pass owns.

**Rows this pass deliberately did not touch.** 293, 294, 297, 298, 315 — `HOLD` verdicts whose records declare
"fingerprinted" without a `sha256:`; the `seam-S14` composer recomputed their digests and they match current bytes,
so the defect is the record, not the claim. Re-issue belongs to the verdict lane and is recorded at
`QB:assumption-registry/run-31-verdict-fingerprints` with its trigger; no checker is spent twice on unchanged bytes.
No `State` cell of a *refuted* row was altered, no evidence was deleted, and no row was reworded in place.

---

# Run 33 — scoped ledger amendment: post-routing re-extraction of run 30/31's malformed and absorbed rows (2026-10-05)

Dispatch: **scoped ledger amendment**, mode `post-routing re-extraction`. The required delta named three jobs: re-row each `Routing: malformed` row to one proposition per row using the rewording brief *its own verdict file* names; replace the `absorbed` `DOES-NOT-HOLD` rows `T013`/`T014` still lean on a corrected premise, per the absorbed entries recorded in [handoff-context.md](../handoff-context.md) §T013/T014 Planned Task Context, retiring any whose correction leaves no surviving premise; and move `seams.md` membership so every live seam carries only rows with current verdicts. Extraction only — this pass checked nothing, composed nothing, re-verified no seam and called nothing safe (E1/E8). It wrote no verdict file, created no invariant and promoted nothing.

**Numbering.** The ledger's sequence ended at run 31's row 319; this pass takes row **33** in both ledgers (run 27's rule: one handle names one pass across both files) and its rows run **320–339**, with bundle keys from **B229** (B1–B228 are history).

**Deliberately untouched, per the dispatch.** Rows **2** (`B226`), **48** (`B227`) and **313** (`B228`) keep the `State: recheck` / `Routing: unverified` cells run 32 wrote, with their bundles and `CL:` handles as-is — no re-row, no re-bundle, no new key. Rows **293/294/297/298/315**'s missing digests are an evidence-record cluster owned by the verdict lane (`QB:assumption-registry/run-31-verdict-fingerprints`); this pass writes ledgers, not verdict records.

**Run basis.**
- **Blueprint for each replacement is the retired row's own `## Invalidates On`.** Five malformed rows name their settling rewording in their verdict (`verdicts/claim-216.md`, `-232.md`, `-256.md`, `-273.md`, `-312.md`); six absorbed rows name theirs in the closure ledger (`DC-claim-208-r1`, `-223-r1`, `-237-r1`, `-238-r1`, `-251-r1`, `-289-r1`) and the matching handoff task block. Nothing here re-litigates a verdict: the old rows keep their terminal `State`, `Routing` and `Permanent` cells and drop out of live seam membership, exactly as rows 3, 20, 93 (and 52, 66, 46, 109, 115, 119, 153, 199) were retired — cells untouched, supersession recorded here and in `seams.md`.
- **No source moved, so no *other* row is invalidated by this pass.** Read at HEAD `bcf596ec2066fe8432abf4987bf575d695524d01` — the same head runs 30 and 31 recorded. `buster_ripper/routes/chat.py` `16c7e999…`, `routes/messages.py` `721a2afe…`, `routes/passthrough.py` `072a5809…`, `buster_ripper/image.py` `611df5ac…`, `tests/test_image.py` `b603bfd1…`, `buster_ripper_legacy.py` `7967563d…`, `REFACTORING_PLAN.md` `f7562fa8…`, `pyproject.toml` `c623bdc2…` — every one byte-identical to the digest that run's own freshness basis recorded. The amendment's trigger is routing, not a changed dependency, domain or observation; hashes establish scope change only (§3d/§3e), and no scope changed.
- **Universe of discourse carried forward.** `buster_ripper/` means the package tree; `buster_ripper_legacy.py` is named by its own subject where it is the referent; test sweeps mean **Python test sources under `tests/`** (`tests/qa/` is `norecursedirs`-excluded, `tests/__pycache__` holds a stale artifact of a deleted module). `pkg:` subjects carry the installed identity **aiohttp 3.13.3**; stdlib facts are `lang:python`.

## Retirement delta (old row → disposition → live rows that now carry it)

| Retired row | Cells kept | Why it retires rather than reopens | Successor rows |
|---|---|---|---|
| 208 | `DOES-NOT-HOLD` / `absorbed` | The correction removes a *citation*, not a premise: base parity owns the eight-name target and `207`/`219`/`220` already assert that equality, with `322` now asserting the scoped base precedent. Nothing survives for a checker to answer. | none — **retired, not replaced** |
| 216 | `UNCLEAR` / `malformed` | The fork was definitional ("header-filtering comprehension"). The two senses are now separate rows, each with its own verdict available. | 320, 321 (S10) |
| 223 | `DOES-NOT-HOLD` / `absorbed` | The corrected scope is a different proposition from the refuted one, so a new row is required rather than a reopen. | 322, 323 (S10) |
| 232 | `UNCLEAR` / `malformed` | Two referents, now two rows. | 330, 331 (S11) |
| 237 | `DOES-NOT-HOLD` / `absorbed` | Request-direction half of the spelling premise; the corrected fact is the opposite of what was rowed. | 324, 326, 327, 329 (S10) |
| 238 | `DOES-NOT-HOLD` / `absorbed` | Response-direction half of the same mechanism. | 325, 326, 328, 329 (S10) |
| 251 | `DOES-NOT-HOLD` / `absorbed` | The population was wrong, not the property; narrowed to code/config plus the artifact-side fact the narrowing rests on. | 336, 337 (no seam) |
| 256 | `UNCLEAR` / `malformed` | The fork was "visits" versus "rewrites"; the visit proposition is rowed, and the rewrite-mutation proposition is **already asserted by row 258** (`HOLD`/`none`), so §3d forbids a second row for it. | 332 (S12) — 258 carries the mutation half by inheritance, no checker spent |
| 273 | `UNCLEAR` / `malformed` | Its own verdict names the settling rewording verbatim; the fork was "payload" (element 0 versus the whole value). | 333 (S13) |
| 289 | `DOES-NOT-HOLD` / `absorbed` | Both conjuncts were false as written; the surviving atoms are the in-test reset boundary and the two-image single call. | 338, 339 (S13) |
| 312 | `UNCLEAR` / `malformed` | "Container" admitted a strict and a loose stdlib-vocabulary reading; one row per sense. | 334, 335 (no seam) |

## Run 33 claims

| # | Claim | Subject | Tasks | Scope | Seams | Check | Bundle | State | Routing | Permanent |
|---|---|---|---|---|---|---|---|---|---|---|
| 320 | Each comprehension under `buster_ripper/routes/` that builds a kept-or-dropped header collection iterates `.items()` on the header mapping it filters | repo:buster_ripper.routes | T013 | single | S10 | absence, seam:S10 | — | superseded | none | — |
| 321 | `buster_ripper/routes/chat.py`'s case-insensitive `authorization` name lookup iterates the bare keys of the already-filtered header dict rather than `.items()` on a headers mapping | repo:buster_ripper.routes | T013 | single | S10 | absence, seam:S10 | B229 | HOLD | none | — |
| 322 | Every upstream request in `buster_ripper_legacy.py` that carries a client-derived header set builds that header set through `_forward_headers` | repo:buster_ripper_legacy | T013 | single | S10 | absence, seam:S10 | B230 | HOLD | none | — |
| 323 | `buster_ripper_legacy.py`'s `_poll_kv_cache` metrics request is issued with no `headers` argument | repo:buster_ripper_legacy | T013 | single | S10 | absence, seam:S10 | B230 | HOLD | none | — |
| 324 | With aiohttp's default C parser, a request header name recognized case-insensitively is stored under aiohttp's own canonical spelling rather than the spelling the client sent | pkg:aiohttp@3.13.3 | T013 | single | S10 | seam:S10 | B231 | HOLD | none | — |
| 325 | With aiohttp's default C parser, a response header name recognized case-insensitively is stored under aiohttp's own canonical spelling rather than the spelling the upstream sent | pkg:aiohttp@3.13.3 | T013 | single | S10 | seam:S10 | B231 | HOLD | none | — |
| 326 | aiohttp's pure-Python HTTP parser stores received header names with the spelling they arrived in | pkg:aiohttp@3.13.3 | T013 | single | S10 | seam:S10 | B231 | HOLD | none | — |
| 327 | aiohttp's `web.Request.raw_headers` yields request header names exactly as the client sent them | pkg:aiohttp@3.13.3 | T013 | single | S10 | native, seam:S10 | B232 | HOLD | none | — |
| 328 | aiohttp's `ClientResponse.raw_headers` yields response header names exactly as the upstream sent them | pkg:aiohttp@3.13.3 | T013 | single | S10 | native, seam:S10 | B232 | HOLD | none | — |
| 329 | Key lookup in the header mappings aiohttp exposes as `web.Request.headers` and `ClientResponse.headers` is case-insensitive | pkg:aiohttp@3.13.3 | T013 | single | S10 | native, seam:S10 | B233 | HOLD | none | — |
| 330 | Every exception class aiohttp defines for a client-side timeout is a subclass of `aiohttp.ClientError` | pkg:aiohttp@3.13.3 | T013, T018 | single | S11, S17 | seam:S11/S17 | B234 | HOLD | none | — |
| 331 | aiohttp's cumulative total-request timeout raises bare `asyncio.TimeoutError`, which is not a subclass of `aiohttp.ClientError` | pkg:aiohttp@3.13.3 | T013, T018 | single | S11, S17 | seam:S11/S17 | B234 | HOLD | none | — |
| 332 | `strip_and_cache_images` considers every message in the `messages` list whose `role` is `user`, in forward list order | repo:buster_ripper.image#strip_and_cache_images | T014 | single | S12 | seam:S12 | B235 | HOLD | none | — |
| 333 | `_replace_image_with_placeholder` stores the cache entry's value as a `(source, timestamp)` tuple whose first element is the block's `source` value | repo:buster_ripper.image#_replace_image_with_placeholder | T014 | single | S13 | seam:S13 | — | superseded | none | — |
| 334 | The Python standard library provides no instantiable cache container type that expires entries by age or evicts them by least-recently-used order | lang:python | T014 | single | — | native, absence | B237 | HOLD | none | — |
| 335 | The Python standard library provides a ready-made LRU caching facility in `functools.lru_cache` | lang:python | T014 | single | — | native | B237 | HOLD | none | — |
| 336 | No tracked file under `buster_ripper/`, `tests/` or `docs/`, and no top-level configuration file, names a `buster_ripper/convert/` path | repo:buster_ripper | T013 | single | — | absence | B238 | HOLD | none | — |
| 337 | Outside `REFACTORING_PLAN.md`, every `buster_ripper/convert/` path mention in a tracked file occurs in a dated session-history artifact under `.sessions/codify/` | repo:.sessions/codify | T013 | single | — | absence | B239 | HOLD | none | — |
| 338 | Nothing in `tests/test_image.py` clears `_image_cache` between statements inside one test function | repo:tests.test_image | T014 | single | S13 | absence, seam:S13 | B240 | UNCLEAR | unverified | — |
| 339 | At least one case in `tests/test_image.py` passes a message containing two image blocks in a single `strip_and_cache_images` call | repo:tests.test_image | T014 | single | S13 | absence, seam:S13 | B240 | HOLD | none | — |

**Row-shape notes (E5 — what each retired row was actually carrying).** **216** forked on a noun, so the two senses are 320 (the keep/drop comprehensions, the shape the rename-proof census greps for) and 321 (the one header-name decision in that family the shape does not reach) — the census's exhaustiveness is a composition question over both, not a verdict on either (E8). **232** forked on a referent, so 330 names the classes and 331 names the exception the cumulative budget actually raises; they cannot be one row because the verdicts point opposite ways. **256** forked on a verb: 332 states the visit, and the rewrite half is 258, already `HOLD`, which is why this run's `i` is 1 and why no second row exists for it. **273**'s settling rewording is a single proposition, so it becomes one row (333) rather than a pair — the element-0 identity and the paired stamp are one fact about one stored value. **312**'s fork is vocabulary-scoped, so 334 is the strict (mapping-type) sense and 335 the loose (ready-made facility) sense; the TTL half the verdict called stable under both readings is carried inside 334's qualifier rather than rowed a third time. **237/238**'s correction is one mechanism seen from two directions and two accessors plus the lookup semantics that make `217`'s `k.lower()` shape sufficient — 324/325 (canonicalization per direction), 326 (the backend the preservation depends on), 327/328 (the verbatim accessor per direction), 329 (case-insensitive lookup); `237` and `238` together retire to those six live rows, and every one of them is about the library rather than about the fix (E3). **223**'s narrowing needs its discriminator on file, which is why 323 (the header-less poll) travels with 322 (the scoped rule). **251**'s narrowing is a claim about which population was scanned, so 336 states the code/config absence and 337 states where the counterexamples actually live — read as repository files, and naming no governance of this session (E4). **289**'s correction is about reset granularity and call arity, so 338 and 339 replace it; row 294 (`HOLD`) already states the per-test reset itself, so this pass rows only what 294 does not: that nothing clears *inside* a test, and that one call is not single-image.

## Run 33 bundles (fresh keys — B1–B228 are history; one agent per bundle)

| Bundle | Rows | Subject(s) | Why tied |
|---|---|---|---|
| B229 | 320, 321 | repo:buster_ripper.routes | one read of the route-side header comprehensions |
| B230 | 322, 323 | repo:buster_ripper_legacy | one read of the legacy file's upstream request sites |
| B231 | 324, 325, 326 | pkg:aiohttp@3.13.3 | one read of the parser's header-name handling (C and pure-Python) |
| B232 | 327, 328 | pkg:aiohttp@3.13.3 | one read of the two `raw_headers` accessors |
| B233 | 329 | pkg:aiohttp@3.13.3 | solo — the header mapping's lookup semantics are a different source than the parser's |
| B234 | 330, 331 | pkg:aiohttp@3.13.3 | one read of the client exception hierarchy plus the timer helper |
| B235 | 332 | repo:buster_ripper.image#strip_and_cache_images | solo — B198's other members (257/258/259) are settled and are not re-dispatched |
| B236 | 333 | repo:buster_ripper.image#_replace_image_with_placeholder | solo — B204's other members (272/274/275) are settled |
| B237 | 334, 335 | lang:python | one read of the `functools`/`collections` documentation and the stdlib module listing |
| B238 | 336 | repo:buster_ripper | solo — a tracked source/config census, not a file read |
| B239 | 337 | repo:.sessions/codify | solo — a different tracked population than B238's (rule 1) |
| B240 | 338, 339 | repo:tests.test_image | one read of `tests/test_image.py`; B210/B211's members there are settled |

**Bundle discipline.** Same subject only (rule 1); tied means tied (rule 2) — 324/325/326 are one read of the parser sources, 327/328 one read of the two accessors, 329 a different class's documented semantics, 330/331 one read of the client exception hierarchy plus the timer helper. Nothing exceeds four rows (rule 3); 332, 333, 336, 337 and 329 are solo because there is nothing to tie them to (rule 4). No key is reused from run 30/31, and no bundle mixes a fresh row with a settled one: 320/321 could have joined the settled 214/215/217 in B178 and 332 the settled 257/258/259 in B198, but those bundles' members are `HOLD`/`none`, so the re-dispatch convention (re-bundle among themselves, fresh key per tie) applies. **12 bundles ⇒ 12 checker agents.** The three run-32 re-dispatches (`B226`/`B227`/`B228`) stay in the orchestrator's queue untouched; this pass neither spends nor cancels them.

## Run 33 seam delta (recorded in [seams.md](seams.md) §Run-33; no seam re-verified)

| Seam | Members before → after | Movement |
|---|---|---|
| S10 | 27 → 32 | drops 208, 216, 223, 237, 238; adds 320–325, 326–329 in place; `rel:buster_ripper/utils#_HOP_BY_HOP→rfc9110§7.6.1` leaves the subject list with its only member |
| S11 | 18 → 19 | drops 232; adds 330, 331 in place |
| S12 | 19 → 19 | drops 256; adds 332 in place |
| S13 | 20 → 21 | drops 273, 289; adds 333, 338, 339 in place |
| S14 | 29 → 29 | **unchanged** — no S14 member is retired or replaced (312 and 251 name no seam; 2/48/313 keep run 32's cells), so its member list is not edited |

**S11 is included although the dispatch named S10/S12/S13 (+S14).** Row 232 is an S11 member (`malformed` is recorded there as the blocking member), and the delta's own rule — every live seam carries only rows with current verdicts — cannot hold while a retired row stays in the list. Dropping 232 and adding its two replacements is a membership write, not a re-verification; S11's `State: new` / `Routing: malformed` / `Permanent: —` cells are left exactly as run 32 wrote them. The same holds for S10/S12/S13: State, Routing and Permanent are untouched in all four rows, because a disposition and a composition verdict are neither of them this pass's to write (E1/E8).

## Run 33 freshness basis and evidence lookups

**Session-ledger matching (§3d), performed for every new proposition.** One proposition this pass extracted was already on file: 256's rewrite-mutation half is row **258** (`HOLD`/`none`, `repo:buster_ripper.image#strip_and_cache_images`, `B198`) — existing row, no recheck trigger (image.py unchanged), so no new row was written and no checker spent; `T014` already stands against it. Nothing else in 320–339 is asserted by an existing row: the ledger's aiohttp rows are JSON decoding (51, 64, 83/84, 149–152), connector/loop/session lifecycle (37, 38→50, 95–104, 113–128, 174–178), `Host`/`Content-Length` defaults (234–236), Content-Type exposure (154–156) and per-request timeouts (58, 141) — none names parser spelling, `raw_headers`, `CIMultiDict` lookup case or the timeout classes' identity. Its `repo:` rows on `routes/` header filters are 214/215/217 (drop sets and lowercasing, all `HOLD`), and on the legacy file 219–225 — no row states either half of 322/323. `lang:python` rows cover `asyncio` timers (43), `Path.open` (44), cancellation (78, 119, 125), `json`/`str`/`dict`/tuple semantics (239/240, 309, 310) and hashing (283–285) — no stdlib cache-container row. `repo:tests.test_image` rows are 288/290–294, none of which states an in-test reset boundary or a two-image call. `repo:REFACTORING_PLAN.md` rows are 250/252 — 336/337 name a different population. So all **20 rows are `new`/`unverified`**, and one further proposition this pass extracted was answered by the row already on file rather than by a new one — `i = 1`, no checker spent on it.

**Registry `## Claim Index` consulted for these propositions only** (176 rows read; no staleness swept for anything this run did not extract): **no match for any of 320–339.** The `pkg:aiohttp@3.13.3` set is `cleanup-hook-error-propagation`, `connector-loop-binding`, `content-type-error-inheritance` (a different exception), `per-request-timeout-override` (what a `timeout=` argument does, not what class its error is), `request-json-decoding`, `response-header-exposure` (a value, not a name spelling), `response-json-content-type-validation`, `runner-cleanup-await`, `runner-task-cancellation`, `signal-shutdown`, `sse-…`, `ssl-…`, `structured-suffix-…`, `web-response-explicit-content-type` — none names parser casing, `raw_headers`, or the timeout hierarchy. There is no `CL:` record for `functools`/`lru_cache`/a stdlib cache container, no record naming `buster_ripper_legacy.py`'s `_forward_headers` call sites or its metrics poll, no record on any `routes/` header comprehension or the `authorization` lookup, no record on `REFACTORING_PLAN.md`/`convert/`, and none on `tests/test_image.py`. So `i = 1` comes from the session ledger, and **no `Permanent` handle is invented** for a new row (extraction never creates one). Where the retired row had a handle (208 none, 216 none, 223 none, 232 none, 237/238 none, 251 none, 256 none, 273 none, 289 none, 312 none) nothing had to be released — all eleven carried `Permanent: —`.

**`## Composition Index` consulted for the four moved member sets only** (no unrelated composition inspected for staleness): the index holds four records — `aiohttp-runner-to-on_cleanup/lifecycle-teardown`, `cli-flag-to-outbound-call/tls-posture-crossing`, `cli-option-to-dump-dispatch/format-value-crossing`, `serve-script-env-to-argv/env-var-flag-crossing` — and none covers a header-parity, response-head, strip-scope or placeholder-identity boundary. So S10/S11/S12/S13 keep `Permanent: —` with no handle to retain, and no composer may be skipped by registry reuse. The four on-disk records `compositions/seam-S11.md`/`-S12.md`/`-S13.md` (`MALFORMED`) and `-S14.md` (`COUNTEREXAMPLE`) are left on disk, neither inherited nor discarded: whether the amended member sets now compose is the composition pass's call (E8), and this pass re-verified none of them.

## Two directions (§1), as rows not judgments

**The waves consume what the corrected rows now say.** F-4's rename-proof site census rests on a pair, not a single row: 320 gives the grep its shape and 321 names the one header-name decision that shape does not reach — so "zero matches" is exhaustive only over the population 320 describes, which is the composition's to weigh, not this pass's. The base-parity precedent is now scoped rather than universal (322 with its discriminator 323), which is what the shared helper must match at four boundaries and not at the fifth. And the spelling premise inverts: 324–329 are the rows that make the filters' `k.lower()` shape (217, `HOLD`) the thing carrying correctness rather than a claim about wire casing, and 326 is the row that says any spelling-based reasoning is backend-conditional. On G-1's side, 330/331 split the arm's reach — one named-class hierarchy inside `except ClientError`, one bare `asyncio.TimeoutError` outside it — and the handler's own shape rows (241/242, `HOLD`) are what they bound. **The peers consume what this amendment introduces:** S13's identity chain now runs through a 2-tuple (333) rather than a bare payload, which is the row 296/303 read as element 1 and 308 unpacks; S13's witnesses inherit per-test state (338) with one call carrying two images (339), which is the pre-state R-D's "every row is a witness, not an idea" is imposed on; and 334/335 are the pair that makes the hand-rolled TTL a reuse-before-build decision rather than an assumption that nothing exists.

## Collected and deliberately NOT rowed — routes recorded, nothing silent (E7)

- **F-4's witness-form obligation** ("site witnesses assert header absence through a case-insensitive lookup, never against an expected wire spelling"; `DC-claim-237-r1`/`DC-claim-238-r1` §Done when) — a test-authoring requirement on unwritten tests, i.e. E3/E4: no blind checker can answer what a test not yet written will assert. Its library-side ingredient is rowed (329), the repo-side shape is row 217 (`HOLD`), and the obligation itself lives in the task block, which is where [handoff-context.md](../handoff-context.md) §Member F-4's `Done when` already carries it.
- **The RFC 9110 §7.6.1 citation as a justification** — retired with 208 and replaced by no row: the closure removed a rationale from the plan's own invariant-candidate wording, and "the eight names are the repo's set" is already 207/219/220. Rowing "RFC 9110 §7.6.1 defines no hop-by-hop field set" would re-put a standard-exegesis premise into the seam the closure just decided is not load-bearing (E6 does not require rowing a fact no premise leans on).
- **`_has_image_in_blocks`'s behavior on a `type: image` block with no `source` key** — kept from run 31's not-rowed list for the same reason: settled by the read 333's own unit is about. It is recorded again because 333 now names element 0 explicitly; the `{}` default at `image.py:80` is visible in that row's own cited range, and *what should happen* for a missing key remains a plan gap for composition and Forward, not a fact this pass can name.
- **`asyncio.TimeoutError` is `TimeoutError` in the interpreter this project runs, and `TimeoutError` descends from `OSError`** — a real wrinkle in 331's neighborhood, but settling it takes the installed interpreter, which is E4 repo state; the version-appropriate reading is the checker's from `lang:python` documentation, and aiohttp's own client handler is what names the class. Recorded so it is not silently lost if 331 comes back `UNCLEAR`.
- **Which of 334/335 the design "actually leans on"** — the dispatch offered that narrowing as an alternative to one row per sense. This pass rowed both senses and left the choice downstream: picking one would be deciding a reuse-before-build question in the extraction context (E1), and 335 is the row that could otherwise be filtered out for looking inconvenient.
- **Whether the amended seams now compose, whether S10/S11/S12/S13 may leave `malformed`, and whether run 32's five `DC-claim-*` closures are now discharged** — E1/E8. This pass moved cells and membership only; the routing columns are run 32's writes and stay as they are until a composer and the decide route move them.
- **Suite passage, test counts, coverage posture, the `DC-*` dispositions, wave ordering and gate state** — session governance and arbiter state (E4), checkable only by reading this session directory.

No claim was checked, no composition performed, no design claim written, no row compound beyond one proposition, nothing filtered for risk, and no seam — amended, inherited or new — was called safe.


---

# Run 34 — scoped ledger amendment: re-rowing claims 48 and 338 after their checks (2026-10-05)

Dispatch: **scoped ledger amendment**, mode `two cells`, over exactly two rows — **48** (`DOES-NOT-HOLD` / `unverified`, its
run-30 recheck settled by [verdicts/claim-48.md](verdicts/claim-48.md)) and **338** (`UNCLEAR` / `unverified` / `B240` — run 33
wrote that Routing and left it there; the *check* is what came back with the P9 fork, and `malformed` is the routing S13's own
seam cell carries, settled by [verdicts/claim-338.md](verdicts/claim-338.md)). Required delta: re-row each to one proposition per row, fresh bundle keys,
`new`/`unverified`, retire 48's claim as worded, carry the complete ten-plus-three census, keep F-3's premise alive, split 338
on the fork its verdict names, and move seam membership so no live seam carries a retired row. Extraction only — this pass
checked nothing beyond reading what its two cited verdicts and the cited source bytes already record, composed nothing,
re-verified no seam, called nothing safe, wrote no verdict file, created no invariant and promoted nothing (E1/E8).

**Numbering.** The ledger's sequence ended at run 33's row 339; this pass takes row **34** in both ledgers (run 27's rule: one
handle names one pass across both files) and its rows run **340–347**, with bundle keys from **B241** (B1–B240 are history).

**Retirement mechanics, and why 48's and 338's cells are not edited.** Both rows keep their `State`, `Routing`, `Bundle` and
`Permanent` cells exactly as runs 32/33 wrote them, and their supersession is recorded here and in
[seams.md](seams.md) §Run-34 — the same form runs 31/33 used for rows 3, 20, 93, 52, 66, 46, 109, 115, 119, 153, 199, 208, 216,
223, 232, 237, 238, 251, 256, 273, 289, 312. Two reasons, both structural rather than stylistic: 48's `Permanent` handle
`CL:rel:buster_ripper.main→buster_ripper.config/numeric-config-cli-wiring` describes the **eight-name** wording, and re-pointing
a retired row's text at that handle while the row now says something else would attach the refuted census to the corrected
proposition (extraction never invents, and never re-aims, a permanent handle); and each row's `verdicts/claim-{n}.md` is the
blind record of the *old* wording's falsifier, which this amendment is built on. A row's `Seams` cell is likewise its
extraction-time declaration, not live membership — see the S3/S5 note.

## Retirement delta (old row → disposition → live rows that now carry it)

| Retired row | Cells kept | Why it retires rather than reopens | Successor rows |
|---|---|---|---|
| 48 | `DOES-NOT-HOLD` / `unverified` / `B227` / `CL:rel:buster_ripper.main→buster_ripper.config/numeric-config-cli-wiring` | Its verdict settles the falsifier exactly: `main.py` wires **ten** numeric config globals, not eight, so the closed partition was short by `DUMP_FLUSH_INTERVAL` and `DUMP_FLUSH_BYTES`. The wording is a compound row by construction — two enumerations in one cell, one verdict for both — and the correction is a different proposition from the refuted one, so new rows are required rather than a reopen. | 340, 341 (census halves), 342, 343 (F-3's per-item premise, S14), 344 (the enumeration route) |
| 338 | `UNCLEAR` / `unverified` / `B240` | Its verdict had to choose a meaning of "clears `_image_cache`" to answer at all (P9): the file's two `_image_cache.clear()` statements are fixture-only, while `image.clear_session_cache("session-1")` at line 260 mutates entries mid-function. The fork is between two senses of one verb, and the second sense's truth is not settled by the test file at all — `clear_session_cache` lives in `buster_ripper/image.py`, outside the bounded source the check was given. Its Routing cell is `unverified` as run 33 wrote it; the malformed finding is in its verdict and in S13's own seam routing, and this pass writes neither. | 345, 346 (S13, test bytes), 347 (S13, `image.py`) — plus **294** carrying the fixture-side fact by inheritance, no new row |

## Run 34 claims

| # | Claim | Subject | Tasks | Scope | Seams | Check | Bundle | State | Routing | Permanent |
|---|---|---|---|---|---|---|---|---|---|---|
| 340 | The numeric `buster_ripper/config.py` globals that `buster_ripper/main.py` wires to a CLI flag are exactly these ten: `MAX_MODEL_LEN`, `COMPACT_TOKEN_RATIO`, `COMPACT_LATENCY_MS`, `COMPACT_KV_RATIO`, `EVAL_MAX_TOKENS`, `EVAL_THINKING_BUDGET`, `IMAGE_CACHE_MAX_SIZE`, `IMAGE_CACHE_TTL`, `DUMP_FLUSH_INTERVAL`, `DUMP_FLUSH_BYTES` | rel:buster_ripper.main→buster_ripper.config | T002, T003, T014 | shared | — | shared, absence | B241 | HOLD | none | — |
| 341 | The only numeric `buster_ripper/config.py` globals that no CLI flag in `buster_ripper/main.py` supplies are `KV_POLL_INTERVAL`, `COMPACT_NUDGE_RATIO` and `MIN_COMPLETION_TOKENS` | rel:buster_ripper.main→buster_ripper.config | T002, T003, T014 | shared | — | shared, absence | B241 | HOLD | none | — |
| 342 | `buster_ripper/main.py` assigns the parsed value of `--image-cache-ttl` onto `buster_ripper/config.py`'s `IMAGE_CACHE_TTL` global | rel:buster_ripper.main→buster_ripper.config | T014 | single | S14 | seam:S14 | B241 | HOLD | none | — |
| 343 | `--image-cache-ttl`'s option declaration in `buster_ripper/main.py` carries `help=` text that presents the value as the image cache TTL in seconds | repo:buster_ripper.main | T014 | single | S14 | seam:S14 | B242 | HOLD | none | — |
| 344 | A CLI option in `buster_ripper/main.py` can assign a `buster_ripper/config.py` global through a parse-time `callback=` while its name appears in neither `main()`'s parameter list nor `_apply_cli_config`'s keys | repo:buster_ripper.main | T002, T003, T014 | shared | — | shared, absence | B242 | HOLD | none | — |
| 345 | Every statement in `tests/test_image.py` that empties the whole `image._image_cache` mapping occurs inside the autouse `reset_image_cache` fixture rather than inside a test function body | repo:tests.test_image | T014 | single | S13 | seam:S13, absence | — | superseded | none | — |
| 346 | One test function body in `tests/test_image.py` calls `image.clear_session_cache` between two of its own assertions | repo:tests.test_image | T014 | single | S13 | seam:S13 | B243 | HOLD | none | — |
| 347 | `clear_session_cache` deletes the named session's whole entry from the module-level `_image_cache` mapping in `buster_ripper/image.py` | repo:buster_ripper.image#clear_session_cache | T014 | single | S13 | seam:S13 | B244 | HOLD | none | — |

## Row-shape notes (E5 — what each retired row was actually carrying, and why its successors are one proposition each)

**48 was two censuses in one cell, and a census of a closed partition is worth exactly as much as each half's exhaustiveness.**
Its verdict reads it as a closed partition ("only carries information if each list is the whole half") and finds the wired half
short by two. So the partition is rowed as **two** rows — 340 (the wired half, ten names) and 341 (the unflagged half, three
names) — never one: a single "the partition is ten-and-three" row would get one verdict covering two enumerations, which is the
defect this amendment exists to remove. Both keep the retired row's `rel:` subject because the proposition *is* the relation
between two modules' surfaces. Each row's length is enumeration, not conjunction — the name list is the proposition, which is
the one case §2 allows a census in the Claim column ("a count, name, or locator may appear in a claim only when it IS the
proposition"). 340's `exactly these ten` and 341's `only` are the exhaustiveness markers the verdict needed in order to call
the partition closed; the "there are thirteen numeric globals in all" closure the verdict also verified is carried inside those
two words rather than rowed a third time (§3d forbids a second row for what an existing row already asserts, and a thirteenth
global would falsify 341 as written).

**342 and 343 are the F-3 survivor, split because "wired" and "advertised" are two properties.** [handoff-context.md](../handoff-context.md)
§Member F-3's cross-reference note reads 48 as "the row that records `IMAGE_CACHE_TTL ← --image-cache-ttl` as a wired CLI global
— evidence that the control is advertised", and `DC-F-3-r1`/`DC-CE-2-r1` both put that sentence in their Closure basis. That
premise is verified per item by 48's own verdict (its first bullet names `IMAGE_CACHE_TTL` at `main.py:57` option `:127`), and
it must not die with the census — so it is rowed per item instead of being inherited from a ten-name set, where a falsifier on
any *other* name would take it down with it. 342 is the wiring; 343 is the advertising, which the census never said at all
(48's wording is about flags existing, not about what the CLI tells the user). They are separate rows on separate subjects with
separate bundles because a checker answers one from `main.py`'s `_apply_cli_config`/signature and the other from one decorator's
keyword arguments, and they can return different verdicts.

**344 is the premise 48's failure actually demonstrates, rowed because the ledger's own two consumers can only re-fall into it.**
The verdict's mechanism sentence — the two missing globals "reach no `main()` parameter and no `_apply_cli_config` key, which is
the only route by which the claim's enumeration of flag-wired globals was taken — but the flag exists and the assignment happens"
— is a fact about the repository's two enumeration surfaces, not about the plan. The dispatch offered it conditionally; this pass
rows it because the falsifier is reproducible from it (any future census taken from the signature or the keys undercounts by
exactly the callback-wired options), rows 99 and 100 already establish that the callback shape is the package's own preferred
form for the dump controls, and 48's surviving `Invalidates On` list names *two* wiring routes ("a new `@click.option` **or** a
new `_apply_cli_config` assignment") — a premise with two routes and one enumerated surface is the kind of thing a later phase
leans on without noticing. It is stated existentially ("can assign … while") so the proposition does not smuggle in the current
name list, which 340 owns.

**338's split follows its verdict's own fork, and the second half needed a different subject.** 345 is the narrow reading
("clears" = empties the whole mapping), settled entirely by `tests/test_image.py` bytes; 346 is the syntactic half of the broad
reading that the test file *does* settle (a call sits at line 260 between the assertions at 258 and 261 — name and position, no
semantics claimed); 347 is the semantic half, which the verdict explicitly refused to answer because it lives outside the
bounded source it was given: "`buster_ripper/image.py`'s `clear_session_cache` implementation — this is outside the supplied
bounded source … and was not read." The dispatch anticipated exactly this ("if the second's truth depends on what
`clear_session_cache` does inside `buster_ripper/image.py`, row it with that subject instead"), so the mutation proposition is
rowed against `repo:buster_ripper.image#clear_session_cache` and not against the test file. 345/346 share `repo:tests.test_image`
and one read of one file; 347 cannot share their bundle (rule 1). **Reason for the split, recorded once:** the retired row
contained one verb with two meanings, one of which was answerable from the supplied source and the other not, so no checker
could have returned a verdict that meant anything; 345 answers the first meaning, 346+347 the second, and a verdict can now
disagree with one without covering the other.

**Why no row states "the autouse fixture clears the cache between tests".** That is row **294** (`HOLD` / `none`,
`repo:tests.test_image`, `B211`, S14) — an existing row asserting it, `tests/test_image.py` unchanged at `sha256:b603bfd1…`, no
recheck trigger. §3d forbids a second row and forbids spending a checker on it; `T014` already stands against it. 345 adds only
what 294 does not say: that the fixture holds *all* of the file's whole-mapping emptying statements.

## Run 34 bundles (fresh keys — B1–B240 are history; one agent per bundle)

| Bundle | Rows | Subject(s) | Why tied |
|---|---|---|---|
| B241 | 340, 341, 342 | rel:buster_ripper.main→buster_ripper.config | one read of `main.py`'s option declarations plus `_apply_cli_config` and `config.py`'s numeric globals; 342's answer is one element of 340's set, so a checker cannot sensibly be given one without the other |
| B242 | 343, 344 | repo:buster_ripper.main | one read of the same option decorators; 344 is the route by which 343's flag and the two dump-flush flags reach their globals |
| B243 | 345, 346 | repo:tests.test_image | one read of `tests/test_image.py` — 338's own bounded source, and the pair is what that source can settle on its own |
| B244 | 347 | repo:buster_ripper.image#clear_session_cache | solo — a different file; the verdict names `image.py` as the source the broad reading requires |

**Bundle discipline.** Same subject only (rule 1); 347 is deliberately not bundled with 345/346 even though one person could
read both files, because bundling is by subject, not by convenience. Tied means tied (rule 2) — 340/341 are the two halves of
one partition settled by one pair of reads, and 342 is an element of 340's set. Nothing exceeds four rows (rule 3). No key is
reused from runs 30–33, and no bundle mixes a fresh row with a settled one: 345/346 could have joined the settled
288/290/291/339 in **B210**/**B240**, but those bundles' members are `HOLD`/`none`, so the re-dispatch convention (re-bundle
among themselves, fresh key per tie) applies. **4 bundles ⇒ 4 checker agents.** Retired 338's `B240` key now names only settled
rows (339), so that bundle carries no dispatch; retired 48's `B227` is run 32's queue entry and this pass neither spends nor
cancels it — see the queue note below.

## Run 34 seam delta (recorded in [seams.md](seams.md) §Run-34; no seam re-verified)

| Seam | Members before → after | Movement |
|---|---|---|
| S3 | 7 → 7 | **unchanged** — S3's list has not carried 48 since run 6 replaced the obsolete inventory row with 99/100/101/103; nothing to drop, and 340/341/344 name no seam |
| S5 | 13 → 13 | **unchanged** — same: run 6 dropped 48 in favour of 100–102; no live S5 member is retired by this amendment |
| S13 | 21 → 23 | drops retired **338**; gains **345**, **346**, **347** in its place; **339** (run 33's replacement for retired 289) keeps its current cells and its position untouched, as the dispatch requires; `repo:buster_ripper.image#clear_session_cache` enters the subject list |
| S14 | 29 → 30 | drops retired **48**; gains **342**, **343** in its place — the two atoms 48's S14 membership was actually granted for; `repo:buster_ripper.main` enters the subject list |

`State`, `Routing` and `Permanent` are untouched in all four rows, including S13's `malformed` and S14's `absorbed`. Whether
S13's amended set may leave `malformed` — every one of its members is now either `HOLD`/`none` on file or freshly dispatched for
this amendment — is the composer's and the decide route's write, not this pass's (E1/E8); this pass moved cells and membership
only.

## Run 34 freshness basis and evidence lookups

**Read at HEAD `bcf596ec2066fe8432abf4987bf575d695524d01` — the same head as runs 30–33.** Bytes consulted for row *shape*
only, all matching the digests the two cited verdicts pin: `buster_ripper/main.py` `fd36dc12…`, `buster_ripper/config.py`
`22348c68…`, `tests/test_image.py` `b603bfd1…`, `buster_ripper/image.py` `611df5ac…`. Nothing here rests on a new observation:
both amendments are triggered by routing, not by a changed dependency, domain or observation, and no row outside 48/338's
successors is invalidated by this pass.

**Session-ledger matching (§3d), performed for all eight new propositions.** One is already on file: 338's fixture-side fact is
row **294** (`HOLD`/`none`, `repo:tests.test_image`, `B211`) — existing row, no trigger (test bytes unmoved), so no new row and
no checker; that is this run's `i = 1`. Nothing else: the ledger's numeric-CLI rows are 22 (`DOES-NOT-HOLD`/`absorbed`, "no
numeric global has a flag"), 23 (dump-related option surface), 99/100 (the two callback-backed dump controls exist) and 74 (the
decorator-default binding time) — none states either half of the corrected partition, the TTL item's assignment, its `help=`
text, or the callback route's invisibility to the signature. `repo:tests.test_image` rows are 288/290–294 and 338/339, none of
which states 345's exhaustiveness or 346's call site; no row in the ledger has `image#clear_session_cache` as its subject (301
is a package-wide caller absence, a different proposition). So **8 rows are `new`/`unverified`**.

**Registry `## Claim Index` consulted for these propositions only** (no staleness swept for anything this run did not extract):
**no match supplies a verdict**, and two near-matches are recorded rather than reused.
- `CL:rel:buster_ripper.main→buster_ripper.config/numeric-config-cli-wiring` — same subject as 340/341/342, and its `Statement`
  is retired 48's **compound** eight-and-three wording, recorded `HOLDS` on main.py `sha256:14ac4018…`. Current main.py is
  `fd36dc12…`, and the fresh blind verdict on the *same* wording at those bytes returns `DOES-NOT-HOLD`. So all three §3e
  conditions fail (a compound statement is not 340's or 341's atomic proposition; freshness is not established; a new
  observation contradicts the record). No handle enters `Permanent` for any successor row — attaching the refuted census's
  handle to the corrected rows would re-import the falsified statement as coverage for them. The record's own disposition is the
  promotion lane's write, and this pass promotes nothing.
- `CL:buster_ripper.main/dump-cli-surface` (row 23's basis, `HOLDS`, pinned to main.py `b4794796…`) is stale on the same axis
  and its statement — "`--dump-dir` is the only dump-related CLI option in `main.py` today" — is contradicted by bytes the 48
  verdict read directly (`--dump-format`, `--dump-flush-interval`, `--dump-flush-bytes` all carry `help=` text in `main.py`, and
  rows 99/100 are `HOLD`). **Row 23's cells are not this pass's to write** — the dispatch scopes two cells (48, 338), and
  retiring or re-checking a row on a contradictory observation is the routing lane's §3d call with its own disposition vocabulary.
  Recorded here so the contradiction is not silently lost and the orchestrator can admit `23` → `recheck` in its own lane.
- `CL:buster_ripper.main#_set_skip_tls_verify/config-assignment` — same *file* as 344 but a different proposition (one named
  callback's assignment, not the route's existence and its two enumeration surfaces). No reuse, no handle.
- No record on `tests/test_image.py`, `_image_cache`, `image.py` or any of its units (the index names none) — so 345/346/347 have
  nothing to inherit. `i` therefore comes from the session ledger alone, and **no `Permanent` handle is invented for any new row**.

**`## Composition Index` consulted for the two moved member sets only:** its four records are
`aiohttp-runner-to-on_cleanup/lifecycle-teardown`, `cli-flag-to-outbound-call/tls-posture-crossing`,
`cli-option-to-dump-dispatch/format-value-crossing` and `serve-script-env-to-argv/env-var-flag-crossing`. The second and third
are *adjacent* to 48's subject (a CLI flag crossing into config) but cover the TLS posture and the format value, not the numeric
census and not the TTL item, and none covers an in-test cache-state boundary. So no seam gains a `CO:` handle, no composer is
skipped by registry reuse, and S13/S14 keep `Permanent: —`. The on-disk `compositions/seam-S13.md` (`MALFORMED`) and
`seam-S14.md` (`COUNTEREXAMPLE`) are left exactly as their authors wrote them: whether the amended sets now compose is the
composition pass's call (E8), and `DC-CE-2-r1`'s citation of premise 48 in its evidence list is a routing-lane amendment this
pass records but does not perform.

## Two directions (§1), as rows not judgments

**The waves consume what the corrected rows now say.** F-3's closure basis and `DC-CE-2-r1`'s premise list both name 48; they now
have 342 and 343 — the control is wired *and* the CLI says what it claims to control, with the enforcement gap the findings
describe untouched by this amendment (that gap is rows 287/295–301's business, all `HOLD`). T002/T003's census consumers get the
partition that is actually closed (340/341), which is the difference between "these eight controls have flags" and "these three
numeric controls are the only ones a user cannot reach from the CLI" — the second is what a peer-consistency check needs, and it
is the half that survives 48's refutation unchanged. S5's "import-bound decorator defaults versus runtime trigger reads"
dimension, whose members 100/101 already record the callback-backed flush controls, now has 344 to say why a signature-shaped
audit of that seam's flags finds nothing: the two trigger globals are wired and *not* enumerable from the command's parameter
list.

**The peers consume what this amendment introduces.** S13's witness-state dimension is now carried by three rows rather than one
ambiguous one, which changes what a composition may conclude: 345 says a witness cannot empty the whole mapping inside its own
body, 346 says a witness issues a `clear_session_cache` call mid-function, and 347 says what such a call removes — that session's
whole per-session mapping, one `del` of the session key, not the entries under it. Read together with the still-live 333 (a `(source, timestamp)` pair), 339 (one call, two image blocks) and the
inherited 294 (reset per test, not per call), the per-test isolation a witness may assume is narrower than 338 appeared to say —
stated here as the shape of the ingredients, not as a verdict about whether they compose (E8).

## Collected and deliberately NOT rowed — routes recorded, nothing silent (E7)

- **"`click` prints an option's `help=` text in the command's `--help` output"** — the `pkg:click@8.3.1` native-capability premise
  one step behind 343's word "advertised". Not rowed: every closure that uses the word ("advertised controls that bind nothing",
  `DC-F-3-r1`'s "advertise the TTL as a live control") means *the CLI surface documents it*, which 343 settles from the
  declaration itself; no premise's truth turns on the renderer. It is recorded here so that if the wave ever wants a
  user-visible-surface claim (`--help` output as evidence), the row it needs is named rather than rediscovered.
- **"`config.py` declares exactly thirteen numeric module-level globals"** — the population-closure fact 48's verdict verified.
  Not rowed: it is not a free-standing premise, it is what makes 340/341 a partition, and it is inside both rows' exhaustiveness
  markers ("exactly these ten", "the only … are these three"). A fourteenth numeric global, or a deleted or retyped one, falsifies
  341 on its own terms — which is also the retired row's own third `Invalidates On` bullet, so the trigger that fired run 32 is
  preserved in the live rows rather than restated as a count (E4-adjacent: a bare count is state, the partition is the property).
- **`--dump-flush-interval` / `--dump-flush-bytes`'s `click.IntRange(min=1)` validation** — settled, single-task, outside every
  seam, and already the subject of rows 100/101/41's neighbourhood; Forward meets it at write time. Listed so its omission is
  visible.
- **Rows 48's and 338's own cells, and their two queue entries (`B227`, `B240`).** This pass retires wording, it does not re-route rows:
  `absorbed` / `question` / `breaks-it` / `user-resolved` are the decide route's dispositions (§claims-migration table), and
  writing `absorbed` into 48's Routing from the extraction context would make a routing decision this pass is not blind to. The
  ledger consequence is stated for the orchestrator instead: **B227 now points at retired wording whose falsifier is already on
  file**, so the pending re-dispatch is either cancelled as superseded by 340–344 or allowed to land as confirmation of
  `DOES-NOT-HOLD` — either way the design's live premise is the new rows, not 48's text. **338 is the same shape with a different
  terminal risk:** its Routing is `unverified` and its `B240` still names it, so it is dispatchable, but a checker given the
  retired wording returns the same P9 fork it returned before — the skill's own rule is that a `malformed` finding reopens only
  on a *materially corrected* claim, and the correction here is rows 345–347 rather than an edit to 338's text, which this pass
  does not own. With 338 retired, `B240` holds only settled 339 (`HOLD`/`none`), so that bundle carries nothing dispatchable
  either way. Both Routing cells are left exactly as runs 32/33 wrote them.
- **Row 23's `HOLD` against the same `main.py` bytes, and the two stale registry fingerprints** (`14ac4018…` for the refuted
  census record, `b4794796…` for `dump-cli-surface`). Both recorded above as §3d/§3e observations for the routing and promotion
  lanes; neither cell is this pass's to move, and no row outside the two named cells was edited.
- **Whether the amended seams now compose, whether S13 may leave `malformed`, and whether `DC-CE-2-r1`'s premise citation of 48
  needs re-aiming at 342/343** — E1/E8. This pass moved cells and membership only.
- **Suite passage, test counts, coverage posture, the `DC-*` dispositions, wave ordering and gate state** — session governance and
  arbiter state (E4), checkable only by reading this session directory.

No claim was checked, no composition performed, no design claim written, no row carries more than one proposition, nothing was
filtered for risk, and no seam — amended, inherited or new — was called safe.

---

# Run 36 — accepted-local T013 + T014 scoped recheck and new-surface extraction (2026-10-06)

> ⚠ **Read [§Run 36 scope correction](#run-36-scope-correction--ledger-scope-correction-2026-10-06) at the end of this file before acting on anything below.** This run's dispatch was a scope defect: the assumption gate covers third-party subjects only, and `repo:` rows already in a ledger are history. Every `recheck` marking in Duty A, and every row of Duty B except **379**, has been reverted, superseded or retired there. The prose below stays as the record of what was written and why it was wrong; it is not a dispatch instruction.

Dispatch: **post-implementation recheck** over both accepted waves — T013 through its yield checkpoint `a5ef525` (F-4 / G-1 / G-2, F-4 via retry 1) and T014 at `4c0f4eb` (Checkpoint 150) — with two duties in one run: **(A)** a stale-row census of every live row whose pinned bytes or stated subject names a moved surface, plus each row's own `Invalidates On`; **(B)** new-surface extraction of what the post-wave seams take for granted about what these two waves *created*, which no pre-wave row covers. Plus the re-issue of the record-defect rows the dispatch names (246, 247, 267, 287, 293, 294, 297, 298, 315, 316). Extraction only: this pass verified no seam, composed nothing, judged no seam safe, re-verified no verdict, wrote no verdict file, created no invariant, promoted nothing, and checked nothing itself (E1/E8). A previous attempt at this dispatch was interrupted before it wrote; this pass started from the untouched ledgers — `claims.md` `sha256:787f94fe18de97618247b86a3da25149ae7f2ba709895f2e80965a0ea09ce904`, `seams.md` `sha256:abc66a586d57d088a1456686648a88a60e47f98fecbeb03a07427929b6bd6e83` — recorded before the first cell write.

**Numbering.** The claims ledger's sequence ended at run 34's row 347; this pass takes row **36** in both ledgers (run 27's rule: one handle names one pass across both files) and its rows run **348–383**, with bundle keys from **B245** (B1–B244 are history; `seams.md` numbers the S10–S13 routing pass as its run 35). Every re-dispatched row is re-bundled **among itself** under a fresh key — no run-30/31/33/34 key is reused, so no checker receives a claim it already answered.

## What moved (freshness basis — hashes establish scope change only, never whether a proposition still holds)

Read at HEAD `4c0f4ebea19a0831312fdd51774827bec8cebfc4`, working tree clean, against the pre-wave head `41d3fc6`:

| Surface | pre-wave | now | units the wave edited |
|---|---|---|---|
| `buster_ripper/image.py` | `611df5ac…` | `f4894b65…` | `_has_image_in_blocks`' neighbours, `has_images`, `_cleanup_expired_images`, `_evict_by_lru`, `_replace_image_with_placeholder`, `strip_and_cache_images`, `get_cached_image`, `cleanup_expired_images`, module state; **new** `_now`, `_user_content`, `_scoped_contents`, `_image_key`, `_cache_source` |
| `buster_ripper/routes/chat.py` | `16c7e999…` | `8da1d171…` | `_prepare_headers` rewritten; **new** `_carries_token`; every line below ~110 renumbered |
| `buster_ripper/routes/messages.py` | `721a2afe…` | `560a81be…` | imports (+2 lines, so every cited locator ≥20 shifted), `handle_messages` guard, `_stream_single_request`, `_non_stream_single_request`; **new** `_relay_after_commit`, `_note_message_start` |
| `buster_ripper/routes/passthrough.py` | `072a5809…` | `c4eef101…` | imports, both header filters replaced by the shared helpers |
| `buster_ripper/utils.py` | `dd60987f…` | `76de98c1…` | `response_headers` only — `forward_headers` (18–24), `_HOP_BY_HOP` (12–15) and `find_message_start` are byte-identical |
| test tree | `tests/test_image.py` `b603bfd1…` → `5b1452c6…`; `tests/conftest.py` `c427e781…` → `e821283c…` | — | **added**: `tests/upstream_stub.py`, `tests/image_bodies.py`, `tests/test_header_parity.py`, `tests/test_image_stability.py`, `tests/test_stream_relay_commit.py` |
| `REFACTORING_PLAN.md` | `f7562fa8…` | moved (G-2) | the `buster_ripper/convert/` rows re-scoped from "done" to "not built" |

Not moved, re-verified this run and used to *exclude* rows: `normalize.py` `a7ddbf06…` (identical to run 31's read), `session.py` `67a66b8a…`, `config.py` `22348c68…`, `main.py` `fd36dc12…`, `server.py` `a9e49823…`, `dump.py` `07b0b076…`, `buster_ripper_legacy.py` `7967563d…`, `pyproject.toml` `c623bdc2…`, `serve_buster_ripper.sh`, `README.md`, `tests/test_routes_integration.py`, `tests/test_dump*.py`, `tests/test_serve_script_argv.py`. No `pkg:` or `lang:` identity moved: aiohttp 3.13.3, click 8.3.1, pytest 9.0.2, Python 3.13.

## Census rule, stated before the results

The skill's subject typing decides what fires, so the pass applied five mechanical tests and nothing else — every one of them is a change in the *record or the cited unit*, never a judgment about whether a proposition still holds (E1):

- **P** — the row's own `verdicts/claim-N.md` `## Source Fingerprint` pins one of the moved surfaces at its pre-wave bytes.
- **U** — the `Subject` names a unit inside a moved surface and the wave's diff has a hunk inside that unit.
- **L** — the Claim text cites a line locator inside a moved surface and the diff shifted that region (instance/locator claims are byte-keyed and fire normally, §2).
- **I** — the row's own `## Invalidates On` condition occurred, verified only by reading (a caller added, a file added to the scanned population, a fingerprint-mismatch clause the row itself wrote).
- **R** — freshness cannot be established from the record on file (the `## Source Version` says "fingerprinted below" while there is no `## Source Fingerprint`, or the record says `unavailable`), which is the §3e third-row trigger; the dispatch names ten such rows for re-issue.

A changed digest on a file the row does not name is **not** a trigger, and neither is a byte move elsewhere in a file whose named unit is untouched — §2: "pin re-freshening is bookkeeping that rides the next atomic pass". That is why `repo:` property rows on `forward_headers`, `_HOP_BY_HOP`, `find_message_start`, `handle_chat`, `_make_chat_request` (apart from its cited locators), `_relay_chat_stream`, `_apply_response_transforms`, `clear_session_cache`, `logging_middleware`, `normalize`, `dump.py`, `server.py`, `config.py` and `main.py` stay as they are.

## Duty A — stale-row census (81 live rows moved to `recheck` / `unverified`, fresh bundles, `CL:` handles retained)

**[VOID — see §Run 36 scope correction.]** Not one of these 81 firings stands: 54 are reverted as out-of-scope repo rechecks, 27 are superseded — 24 because the accepted waves refuted the sentence, 3 because a blind verdict already on file settled them against a `HOLD`. The table below is kept as the record of the census's *observations* (what moved, which unit, which of the run's own tests it claimed), not as a dispatch set.

| Row | Trigger | Per-row reason |
|---|---|---|
| 13 | P, U | pins `messages.py` at expired `9c234c4f…`; `handle_messages` was edited at its own cited region's file (guard now `:64`), and the block at `:55-59` is now at different lines |
| 17 | P, I | pins `messages.py`; the census population (`buster_ripper/` + `tests/` symbol set) gained `_image_key`, `_cache_source`, `_now`, `_scoped_contents`, `_user_content`, `_relay_after_commit`, `_note_message_start`, `_carries_token` and five test modules |
| 26 | P | pins `chat.py` `16c7e999…`, `messages.py` `721a2afe…`, `passthrough.py` `072a5809…` — all three moved |
| 45 | I | its own second bullet: "Any byte of any file under `buster_ripper/` changes (fingerprint mismatch — re-run the package sweep)" |
| 46 | P | pins `messages.py` at expired bytes |
| 53 | L | cites `messages.py:27`; the wave added two import lines above it |
| 54 | L | cites `messages.py:28-29`, same shift |
| 56 | L | cites `chat.py:146`; the `_prepare_headers` rewrite moved it |
| 61 | L | cites `chat.py:122` and `passthrough.py:26`; both moved (no `ClientTimeout` call site was added or removed — the locators did) |
| 65 | L | cites `chat.py:131`; moved |
| 67 | U, P | the `seam-S9` rider the dispatch names: the relay loop left `handle_streaming`'s sightline into `_relay_after_commit`, so the row's own sentence ("a prepared `StreamResponse` fed by `resp.content.iter_any()`") describes a moved shape |
| 73 | U, P | the second `seam-S9` rider member; `handle_messages` edited, and its B196 verdict is one commit behind |
| 129 | I | routes-family census over `request.json()` sites; all three routes files were edited |
| 130, 131 | U | `handle_messages` body edited |
| 197 | U, P | guard branch edited; pins `messages.py` `721a2afe…` |
| 200 | P | pins `tests/conftest.py` `c427e781…`; the fixture file was rewritten (both proxy fixtures now come from it) |
| 206 | U, **sentence** | `response_headers` now drops `("content-length", *_HOP_BY_HOP)` — "exactly the names in `_HOP_BY_HOP` and no others" must not be recomposed as current support |
| 209 | I | call-site census; `forward_headers` is now called at `chat.py:103`, `messages.py:64`, `passthrough.py:17` |
| 210 | I | same; `response_headers` now called at `messages.py:285`, `passthrough.py:41` |
| 212 | U, **sentence** | chat.py now *calls* what it imported; "imports `forward_headers` without calling it" describes the pre-wave file |
| 213 | U, **sentence** | `_prepare_headers` was rewritten: the single-value `strip() in ("Bearer", "Bearer ")` deletion is gone, replaced by `getall` + `_carries_token` + canonical re-insertion |
| 214, 215, 217 | I, **sentence** | all three are census claims over "every header filter under `buster_ripper/routes/`" — the five local filters no longer exist, so each row's population is now empty-or-delegated; a vacuous population is the checker's answer, not this pass's |
| 219 | I | its own first bullet: "Either file's bytes no longer match its fingerprint above" — `utils.py` moved |
| 225 | I | its own first bullet ("any new or edited file under `buster_ripper/` or `tests/` containing the string `buster_ripper_legacy`") occurred at `tests/test_header_parity.py:4,239`; whether a docstring citation is an import is the checker's call |
| 241, 242, 243, 245 | U | `_stream_single_request` was split at `prepare()`; the try/catch topology the four rows describe is a different structure now |
| 246, 247, 248 | I, R | each row's own bullet fired (`tests/test_header_parity.py` asserts forwarded/withheld headers and `Authorization`/`Bearer` values; `tests/test_stream_relay_commit.py` exercises post-`prepare()` failures); all three records also say `Source Fingerprint: unavailable — tree-wide negative`, so no freshness basis exists |
| 250 | P, I | `REFACTORING_PLAN.md` bytes moved, and the row's own second bullet says the fingerprint is what covers "which `buster_ripper/convert/` paths it names" — G-2 changed that list |
| 253 | U, **sentence** | `has_images` was rewritten to scan every user message; "inspects only the first user message it meets" is the pre-wave behaviour |
| 254 | U | same unit edited |
| 257, 258, 259, 332 | U | `strip_and_cache_images`' loop replaced by `_scoped_contents` enumeration |
| 260 | P | module-wide absence over `image.py`, which gained four units |
| 261, 262, 263 | U | the guarded branch in `handle_messages` was edited at `:64` |
| 267 | R | **re-issue per dispatch**: `Source Version` says "fingerprint below is the stability basis" while the record has no `## Source Fingerprint` section; `normalize.py` is byte-identical, so the defect is the record alone |
| 268 | L, P | cites `messages.py:34` (now `:36`); pins `messages.py` |
| 272 | U, **sentence** | `_replace_image_with_placeholder` no longer derives the key from the entry count — `_image_key` does, from content |
| 274, 275, 276, 277 | U | the placeholder unit's body was rewritten (`_cache_source` inserted between the key derivation and the eviction) |
| 287 | P, R | pins `image.py` `611df5ac…` plus a search-surface digest; also named for re-issue |
| 288, 290, 291, 292 | P | `tests/test_image.py` rewritten (`b603bfd1…` → `5b1452c6…`); 288's "pins a literal `img_` key" now sits in a file whose positive assertions pin derived keys (`test_image.py:214`) |
| 293 | P, R, **sentence** | **the dispatch's named example**: the row asserts TTL expiry is reached *only* through a `get_cached_image` call, and the rewritten file asserts expiry on `_image_cache` directly (`test_image.py:295-300`, `test_image_stability.py:274,324,337`) — it must not be recomposed against as current support |
| 294 | P, R | the autouse fixture survives in `test_image.py`, but `test_image_stability.py:45` defines a second one of the same name, so the row's file-scoped sentence is no longer the whole reset surface |
| 295, 296, 297, 298 | U (+R on 297, 298) | `_cleanup_expired_images` now reads `_now()` and destructures a triple; 297/298's records declare a fingerprint with none written |
| 299 | U, **sentence** | "calls `_cleanup_expired_images` once per session and performs no other operation" — the janitor now also drops the emptied session key |
| 300, 301 | I, P | `cleanup_expired_images` has a production caller (`image.py:198`) and test callers; both pins are `image.py` |
| 302, 303, 304, 305 | U (+**sentence** on 303) | `_evict_by_lru`'s sort key moved from element 1 to element 2, so "orders eviction by the cache entry's second element" is the pre-wave slot |
| 306, 307, 308 | U (+**sentence** on 308) | the entry type is now a 3-tuple and `get_cached_image` unpacks three targets — "exactly two" describes the old shape |
| 315 | R | **re-issue per dispatch**: `Source Version` names a fingerprint "below" that the record does not contain; `server.py` is unmoved |
| 316 | I, P, R | both of its own bullets occurred (files under `tests/` now set `STRIP_IMAGES` true; the Python-file set changed) and its aggregate `tests/` digest recomputes differently |
| 320, 321 | I, (+**sentence** on 320) | 320's population — comprehensions under `routes/` that build a kept-or-dropped header collection — is now empty; 321's bare-key lookup was replaced by `getall` plus a deletion loop |
| 333 | U, **sentence** | the stored value is no longer a `(source, timestamp)` pair |
| 339, 345, 346 | P | `tests/test_image.py` bytes moved; 345/346's whole-mapping and mid-function statements are in a rewritten file |
| 246, 247 *(also listed above)* | — | counted once; the re-issue and the census share one bundle |

**Not marked stale, with reasons (nothing dropped silently).** `pkg:` and `lang:` rows — none fired: no dependency, interpreter or documented source moved, and §2's subject typing keeps a repo edit from touching them (the `## Claim Index` digests for aiohttp/click/pytest/CPython are unchanged). **`utils.py` rows whose unit is byte-identical** — 205 (`forward_headers`, lines 18–24), 207 (`_HOP_BY_HOP`, 12–15), 244 (`find_message_start`): the wave's only `utils.py` hunk is inside `response_headers`, and each row's own `Invalidates On` names the unchanged tuple, the constant's membership, or a new guard on the `message_start` path, none of which occurred; their `dd60987f…` pins are bookkeeping for the next atomic pass. **`chat.py` rows on untouched units** — 132/133 (`handle_chat`), 139/140/145–148 (`_make_chat_request`, no locator cited), 149–153/157/158 (`_relay_chat_stream`, `_apply_response_transforms`): the diff's two hunks are inside `_prepare_headers` and the new `_carries_token`. **`56`/`65` fired only on their cited locators**, which is why their neighbours did not. **`normalize.py` rows 264–266** and `session.py` 11/286/319, `dump.py` rows, `config.py`/`main.py` rows (23, 269–271, 340–344), `server.py` 314, `pyproject.toml` 27–29, `serve_buster_ripper.sh`/`README.md`/`lang:bash` rows (S6's set), `count_tokens.py` rows (55, 134–138, 218), `backend.py` rows, `buster_ripper_legacy.py` rows 219–224 (the file is unmoved; 219 is a *relation* whose own first bullet fired on `utils.py`, and 220–224's units did not move), test rows on unchanged modules (30, 31, 107/108, 165–173, 199, 201, 317, and every `tests/test_dump*` pin), **278** (its `HOLD` was set at `d70c876` on the *current* `image.py` bytes `f4894b65…` — run 32's cell write is fresh against this wave, which is exactly why it was verified rather than assumed), **347** (`clear_session_cache`'s body is byte-identical; the wave's new `del` lives in `cleanup_expired_images`, a different unit), and the retired rows (`3, 18, 19, 20, 22, 32, 34, 38, 47*-adjacent, 48, 52, 60, 62, 63, 66, 91, 93, 102, 109, 115, 119, 153, 167, 199, 208, 216, 223, 232, 237, 238, 251, 256, 273, 289, 312, 338` — terminal cells are history, per §"repo rows already in a ledger" and runs 31/33/34's retirement convention). **One row the dispatch did not name shares the record-defect shape**: 255 (`_has_image_in_blocks`, `Source Version` says "fingerprinted below", no fingerprint section) — its unit is byte-identical and no census trigger fires, so it is *not* re-dispatched here; it belongs to `QB:assumption-registry/run-31-verdict-fingerprints`, and is recorded so the omission is visible rather than silent (E6).

**Re-issue disposition for the ten named record-defect rows.** 246, 247, 287, 293, 294, 297, 298 and 316 are **subsumed** — they are stale by bytes or by their own fired invalidation as well, and each returns in a fresh bundle on current bytes (no checker is spent twice on unchanged bytes). 267 and 315 return **on the record defect alone** (`R`): their files are unmoved, their propositions are not refuted, and what is missing is the evidence field that would let a later pass reuse them — a re-dispatch, not a reversal.

## Duty B — new-surface extraction: what the post-wave seams take for granted about what these waves created

**[VOID except row 379 — see §Run 36 scope correction §3.]** Every row of the table below is a `repo:` subject, and the gate has no authority to create one; 348–383 are retired in place, and only **379** (`pkg:aiohttp@3.13.3`) survives as an in-scope premise.

No pre-wave row covers any of 348–383: the ledger's `image.py` rows name `has_images`, `strip_and_cache_images`, `_replace_image_with_placeholder`, `_cleanup_expired_images`, `_evict_by_lru`, `get_cached_image`, `cleanup_expired_images` and `clear_session_cache` — none names `_now`, `_user_content`, `_scoped_contents`, `_image_key` or `_cache_source`; its `chat.py` rows name `_prepare_headers` and the request helpers, never `_carries_token`; its `messages.py` rows name `handle_messages`, `handle_streaming` and `_stream_single_request`, never `_relay_after_commit` or `_note_message_start`; and it has no row on `tests/upstream_stub.py`, `tests/conftest.py`, `tests/image_bodies.py` or `tests/test_image_stability.py`, which is what the new witnesses stand on.

| # | Claim | Subject | Tasks | Scope | Seams | Check | Bundle | State | Routing | Permanent |
|---|---|---|---|---|---|---|---|---|---|---|
| 348 | `has_images` decides from the pairs `_scoped_contents` returns | repo:buster_ripper.image#has_images | T014 | single | S12 | — | — | retired | none | — |
| 349 | `strip_and_cache_images` rewrites exactly the pairs `_scoped_contents` returns | repo:buster_ripper.image#strip_and_cache_images | T014 | single | S12 | — | — | retired | none | — |
| 350 | `_user_content` returns None for anything that is not a user message with list content | repo:buster_ripper.image#_user_content | T014 | single | S12 | — | — | retired | none | — |
| 351 | `_scoped_contents` yields no pairs for a `messages` value that is not a list | repo:buster_ripper.image#_scoped_contents | T014 | single | S12 | — | — | retired | none | — |
| 352 | `_image_key` takes the image block's `source` mapping as its operand, not the block | repo:buster_ripper.image#_image_key | T014 | single | S13 | — | — | retired | none | — |
| 353 | `_image_key` serializes its operand with `sort_keys=True` | repo:buster_ripper.image#_image_key | T014 | single | S13 | — | — | retired | none | — |
| 354 | `_image_key` serializes with the compact two-item `separators` form | repo:buster_ripper.image#_image_key | T014 | single | S13 | — | — | retired | none | — |
| 355 | `_image_key` encodes the serialization as UTF-8 before handing it to `hashlib.sha256` | repo:buster_ripper.image#_image_key | T014 | single | S13 | — | — | retired | none | — |
| 356 | `_image_key` returns `img_` plus the first twelve characters of the hex digest | repo:buster_ripper.image#_image_key | T014 | single | S13 | — | — | retired | none | — |
| 357 | `image._now` is a module-level name bound to `time.monotonic` | repo:buster_ripper.image#_now | T014 | single | S14 | — | — | retired | none | — |
| 358 | Every retention stamp `image.py` writes is taken by calling `_now` | repo:buster_ripper.image | T014 | single | S14 | — | — | retired | none | — |
| 359 | `_cache_source` stores each entry as a `(source, first_seen, last_touched)` triple | repo:buster_ripper.image#_cache_source | T014 | single | S14 | — | — | retired | none | — |
| 360 | `_cache_source` carries an existing entry's `first_seen` forward when the same key is written again | repo:buster_ripper.image#_cache_source | T014 | single | S14 | — | — | retired | none | — |
| 361 | `image.py` takes no timestamp from `time.time` | repo:buster_ripper.image | T014 | single | S14 | — | — | retired | none | — |
| 362 | `strip_and_cache_images` calls `cleanup_expired_images` after its message loop | repo:buster_ripper.image#strip_and_cache_images | T014 | single | S14 | — | — | retired | none | — |
| 363 | `cleanup_expired_images` deletes a session's key from `_image_cache` once that session's mapping is empty | repo:buster_ripper.image#cleanup_expired_images | T014 | single | S14 | — | — | retired | none | — |
| 364 | `strip_and_cache_images` returns before the sweep when `messages` is absent or not a list | repo:buster_ripper.image#strip_and_cache_images | T014 | single | S14 | — | — | retired | none | — |
| 365 | `_carries_token` is True for an `Authorization` value that is neither blank nor the bare `Bearer` form | repo:buster_ripper.routes.chat#_carries_token | T013 | single | S10 | — | — | retired | none | — |
| 366 | `_prepare_headers` reads the candidate `authorization` values with `getall` on the request mapping, not on the dict it returns | repo:buster_ripper.routes.chat#_prepare_headers | T013 | single | S10 | — | — | retired | none | — |
| 367 | `_prepare_headers` deletes every `authorization` name from the forwarded headers before re-inserting one kept value under the canonical spelling | repo:buster_ripper.routes.chat#_prepare_headers | T013 | single | S10 | — | — | retired | none | — |
| 368 | `_prepare_headers` keeps a value-less `Authorization` line when no token-bearing value arrived | repo:buster_ripper.routes.chat#_prepare_headers | T013 | single | S10 | — | — | retired | none | — |
| 369 | `_stream_single_request` delegates the post-`prepare()` relay to `_relay_after_commit` and returns the committed response object | repo:buster_ripper.routes.messages#_stream_single_request | T013 | single | S11 | — | — | retired | none | — |
| 370 | `_relay_after_commit`'s only `except` arm catches `Exception` | repo:buster_ripper.routes.messages#_relay_after_commit | T013 | single | S11 | — | — | retired | none | — |
| 371 | `_relay_after_commit`'s `Exception` arm logs and does not re-raise | repo:buster_ripper.routes.messages#_relay_after_commit | T013 | single | S11 | — | — | retired | none | — |
| 372 | `_note_message_start` takes its timing origin and session id from the state mapping `_stream_single_request` builds before its `try` | repo:buster_ripper.routes.messages#_note_message_start | T013 | single | S11 | — | — | retired | none | — |
| 373 | `utils.response_headers` drops `content-length` in addition to every name in `_HOP_BY_HOP` | repo:buster_ripper.utils | T013 | single | S10 | — | — | retired | none | — |
| 374 | `tests/upstream_stub.py`'s `StubStream` yields caller-supplied chunks from `content.iter_any()` and raises a caller-supplied exception after them | repo:tests.upstream_stub | T013 | single | S11 | — | — | retired | none | — |
| 375 | `tests/upstream_stub.py`'s `StubUpstream` presents the `status`, `headers`, `read`, `text` and `json` surface the route handlers read from an upstream answer | repo:tests.upstream_stub | T013, T014 | shared | — | — | — | retired | none | — |
| 376 | `tests/conftest.py`'s `raw_proxy` fixture serves the real app on a loopback port with backend detection and the image flag neutralized | repo:tests.conftest | T013 | single | S10 | — | — | retired | none | — |
| 377 | `RawProxy.send` ends its response read at the chunked terminator, at EOF, or after a fixed quiet window | repo:tests.conftest | T013 | single | S10 | — | — | retired | none | — |
| 378 | `tests/conftest.py`'s `proxy` fixture drives the same app through an aiohttp `TestClient` | repo:tests.conftest | T013 | single | S10 | — | — | retired | none | — |
| 379 | aiohttp's client merges two request header names that differ only in case into one entry before the wire | pkg:aiohttp@3.13.3 | T013 | single | S10 | native, seam:S10 | B287 | HOLD | none | — |
| 380 | `tests/image_bodies.py`'s body builders return a new structure on each call | repo:tests.image_bodies | T014 | single | S12 | — | — | retired | none | — |
| 381 | `tests/image_bodies.py`'s `reset_cache` empties `buster_ripper.image._image_cache` wholesale | repo:tests.image_bodies | T014 | single | S13, S14 | — | — | retired | none | — |
| 382 | `tests/image_bodies.py`'s `stamp_source` patches `buster_ripper.image._now` with a movable clock object | repo:tests.image_bodies | T014 | single | S14 | — | — | retired | none | — |
| 383 | `tests/test_image_stability.py` defines its own autouse fixture that resets `image._image_cache` around each test | repo:tests.test_image_stability | T014 | single | S14 | — | — | retired | none | — |

**Row-shape notes (E5 — where one blueprint sentence is two facts).** The seam needs the serializer as four atoms, not one: the operand (`source`, not the block — 352), the canonical form (353/354), and the coercion Yield ruled on (355) are settled by three different lines and could disagree with each other, while the slice-and-prefix (356) is the discriminator against slicing the serialization. The retention contract is two rows because "a triple" (359) and "age is not refreshed by a re-write" (360) are different lines in `_cache_source` and only the second is what makes `IMAGE_CACHE_TTL` bound absolute age. The stamp seam is two rows because a checker reads the binding (`_now = time.monotonic`, 357) and the call sites (358) at different places, and 361 is the negative half the wall-clock premises (296/297/311) now depend on. `_relay_after_commit` is split at the arm and at what the arm *does* (370/371), since `except Exception: pass` and `except Exception: raise` differ in the one dimension S11 traces. The `_prepare_headers` selection is three rows — where the values come from (366), what happens to the names already in the forwarded dict (367), and which value survives when none carries a token (368) — because the multi-value collapse is a different line from the predicate that scores each value (365). The witness mechanics are rows only where a seam's dimension cannot be traced without them: `StubStream`'s fault hook is the only way S11's post-commit cases exist at all, and `raw_proxy`'s quiet-window read (377) is the only way "the response has one terminal" is observable.

**Pairs deliberately kept apart from retired wording (no re-routing here).** 206/373, 253/348-351, 272/352-356, 299/363, 303/359-360, 308/359, 333/359 and 213/365-368 are not duplicates: in each pair the first row states the pre-wave fact and is dispatched to be *refuted or re-confirmed on current bytes*, and the second states the post-wave fact nothing on file asserts. Which of each pair the seam keeps is the composition and decide lanes' write (§3d: "a terminal negative verdict remains true without repeating its already-routed consequence"); this pass writes neither disposition (E1).

## Run 36 bundles (fresh keys — B1–B244 are history; one agent per bundle)

**[VOID — see §Run 36 scope correction §5.]** Bundle keys **B245–B286 and B288–B289 are void and must not be dispatched**: every row they carried is a `repo:` subject, retired or reverted, and none of them carries a `Bundle` cell any more. The only live key this run produced is **B287** (row **379**, `pkg:aiohttp@3.13.3`). The table stays as the record of what the erroneous pass allocated.

| Bundle | Rows | Subject(s) | Why tied |
|---|---|---|---|
| B245 | 206, 373 | repo:buster_ripper.utils | one read of `response_headers` — the pre-wave drop set and the post-wave one are settled by the same line |
| B246 | 219 | rel:buster_ripper_legacy#_HOP_BY_HOP→buster_ripper/utils#_HOP_BY_HOP | solo — a set-identity relation across two files |
| B247 | 56, 65, 212, 213 | repo:buster_ripper.routes.chat, #_make_chat_request, #_prepare_headers | one read of `chat.py` (the two locator rows and the two rewritten units) |
| B248 | 53, 54, 129 | repo:buster_ripper.routes | one sweep of the `request.json()` sites across the routes family |
| B249 | 214, 215, 217 | repo:buster_ripper.routes | one sweep of the route-side filter population |
| B250 | 320, 321 | repo:buster_ripper.routes | one read of the same comprehensions and the `authorization` lookup |
| B251 | 13, 73, 130, 131 | repo:buster_ripper.routes.messages#handle_messages | one read of the handler body |
| B252 | 197, 261, 262, 263 | repo:buster_ripper.routes.messages#handle_messages | same unit; the cap of four forces the split |
| B253 | 67 | repo:buster_ripper.routes.messages#handle_streaming | solo — the `seam-S9` rider's relay-shape row |
| B254 | 241, 242, 243, 245 | repo:buster_ripper.routes.messages#_stream_single_request | one read of the handler after the split |
| B255 | 17, 26, 45, 46 | repo:buster_ripper | one package-wide write/buffering/HTTP-call sweep |
| B256 | 61, 209, 210, 268 | repo:buster_ripper | one package-wide call-site census (timeouts, header helpers, `normalize` call) |
| B257 | 287, 300, 301 | repo:buster_ripper | one sweep for the read path's and the janitors' callers |
| B258 | 246, 247, 248 | repo:tests | one sweep of Python test sources for header, `Authorization` and post-`prepare()` assertions |
| B259 | 200, 316 | repo:tests | one sweep of the same population for the serve-script and `STRIP_IMAGES` surfaces |
| B260 | 288, 290, 291, 292 | repo:tests.test_image | one read of the rewritten `test_image.py` |
| B261 | 293, 294, 339, 345 | repo:tests.test_image | same file, remaining pins; cap forces the split with B262 |
| B262 | 346 | repo:tests.test_image | solo — the mid-function call site's own line |
| B263 | 253, 254 | repo:buster_ripper.image#has_images | one read of the detector |
| B264 | 257, 258, 259, 332 | repo:buster_ripper.image#strip_and_cache_images | one read of the writer's loop and its guards |
| B265 | 260 | repo:buster_ripper.image | solo — a module-wide shape sweep for `tool_result` |
| B266 | 272, 274, 275, 276 | repo:buster_ripper.image#_replace_image_with_placeholder | one read of the placeholder unit |
| B267 | 277, 333 | repo:buster_ripper.image#_replace_image_with_placeholder | same unit; cap of four forces the split |
| B268 | 295, 296, 297, 298 | repo:buster_ripper.image#_cleanup_expired_images | one read of the expiry primitive |
| B269 | 299 | repo:buster_ripper.image#cleanup_expired_images | solo — the janitor's own body |
| B270 | 302, 303, 304, 305 | repo:buster_ripper.image#_evict_by_lru | one read of the eviction primitive |
| B271 | 306, 307, 308 | repo:buster_ripper.image, #get_cached_image | one read of the module state and the read path |
| B272 | 250 | repo:REFACTORING_PLAN.md | solo — a document read plus a working-tree check |
| B273 | 225 | repo:buster_ripper_legacy | solo — an import/string sweep across both roots |
| B274 | 315 | repo:buster_ripper.server | solo — the middleware registration list |
| B275 | 267 | repo:buster_ripper.normalize#_normalize_cache_control | solo — the record re-issue on unmoved bytes |
| B276 | 348, 349, 350, 351 | repo:buster_ripper.image#has_images, #strip_and_cache_images, #_user_content, #_scoped_contents | one read of the shared scope section |
| B277 | 352, 353, 354, 355 | repo:buster_ripper.image#_image_key | one read of the serializer |
| B278 | 356 | repo:buster_ripper.image#_image_key | same unit; cap of four forces the split |
| B279 | 357, 358, 359, 360 | repo:buster_ripper.image#_now, #_cache_source, repo:buster_ripper.image | one read of the stamp binding and the retention writer |
| B280 | 361 | repo:buster_ripper.image | solo — a module-wide sweep for the wall-clock call |
| B281 | 362, 364 | repo:buster_ripper.image#strip_and_cache_images | one read of the writer's tail and its early return |
| B282 | 363 | repo:buster_ripper.image#cleanup_expired_images | solo — the janitor's new drop, a different unit from B281's |
| B283 | 365, 366, 367, 368 | repo:buster_ripper.routes.chat#_carries_token, #_prepare_headers | one read of the credential-selection block |
| B284 | 369, 370, 371, 372 | repo:buster_ripper.routes.messages#_stream_single_request, #_relay_after_commit, #_note_message_start | one read of the relay split |
| B285 | 374, 375 | repo:tests.upstream_stub | one read of the stand-in module |
| B286 | 376, 377, 378 | repo:tests.conftest | one read of the shared fixtures |
| B287 | 379 | pkg:aiohttp@3.13.3 | solo — a library behaviour, never bundled with a repo pattern (rule 1) |
| B288 | 380, 381, 382 | repo:tests.image_bodies | one read of the builders module |
| B289 | 383 | repo:tests.test_image_stability | solo — a different file's fixture |

**[Voided — the corrected figure is 1 bundle ⇒ 1 checker: B287 / row 379.]** ~~**45 bundles ⇒ 45 checker agents**~~ **(31 carrying the 81 re-dispatched rows, 14 carrying the 36 new rows; B245 carries one of each)** — see §Run 36 scope correction §5. No bundle crosses a library and a repo pattern or two libraries (rule 1); none exceeds four rows (rule 3); every key is new (no checker receives a claim it already answered); solo is the default, not padding (rule 4). Run 32's and 33's three queue entries **B226** (row 2), **B227** (retired 48) and **B228** (row 313) are untouched by this pass — rows 2 and 313 are not stale by any of P/U/L/I/R (their sources are unmoved and their invalidation conditions did not fire), and this pass neither spends nor cancels a queue entry that belongs to the routing lane.

## Run 36 freshness basis and evidence lookups

**Session-ledger matching (§3d), performed for all 36 new propositions.** No existing live row asserts any of 348–383: the ledger's `image.py` rows describe the pre-wave detector, writer, key derivation, tuple shape, janitor body and eviction order (all now re-dispatched, none a substitute for a row about `_scoped_contents`, `_image_key`, `_cache_source` or `_now`); its `chat.py` rows are the filter/relay/request units plus retired 213; its `messages.py` rows are `handle_messages`, `handle_streaming` and `_stream_single_request` (241–245, now `recheck`) and never a post-commit helper; and there is no row anywhere in runs 1–34 whose subject is `repo:tests.upstream_stub`, `repo:tests.conftest`, `repo:tests.image_bodies` or `repo:tests.test_image_stability` — the two `repo:tests` rows that exist (200, 246–248, 316) are negative censuses over the population, not the mechanics those censuses now point at. So **117 rows are `new` or `recheck` with `Routing: unverified`, and `i = 0`**: no proposition this run extracted was answerable from a verdict already on file, because the whole point of the pass is that the tree the on-file verdicts describe no longer exists.

**Registry `## Claim Index` consulted for these propositions only** (175 records read; no unrelated record swept for staleness, §3e): **no `CL:` record names any of 348–383** — the index contains no `CL:buster_ripper.image/*` record at all, none naming `utils.forward_headers`, `utils.response_headers`, `_HOP_BY_HOP`, any `routes/` header filter, `_carries_token`, `_relay_after_commit`, `_note_message_start`, `upstream_stub`, `conftest`, `image_bodies` or `test_image_stability`; its `pkg:aiohttp` records are JSON decoding, exception inheritance, header *value* exposure, connector/runner/signal lifecycle, per-request timeouts and Content-Type defaults — none is the client's duplicate-name merge (379). So no row gains a `Permanent` handle and **no handle is invented** (extraction never creates one). For the re-dispatched rows, thirteen carry `CL:` handles (13, 17, 26, 45, 46, 54, 56, 65, 67, 73, 130, 131, 197) and each registry copy was checked by comparing the source digest it pins with current bytes: they pin the same expired `9c234c4f…`/`243b2143…`-class fingerprints as their `verdicts/` records, so **freshness cannot be established for any of them** (§3e third row) — handles retained in `Permanent`, checkers dispatched, no reversal implied.

**`## Composition Index` consulted for the moved member sets only:** its four records are runner→`on_cleanup` teardown, CLI-flag→outbound-call TLS posture, CLI-option→dump-dispatch format, and serve-script→argv env. None covers a header-parity, response-head, strip-scope, placeholder-identity or cache-lifecycle boundary, so **no composer is skipped by registry reuse**, S9–S14 keep `Permanent: —`, and no `CO:` handle enters or leaves the ledger.

**No `user-claim` row and no `MIGRATION-REQUIRED` return.** `decisions.md` holds no `Verification: assumptions PENDING` factual claim bearing on this recheck (its only `PENDING` remains the `yield-phase:MS-001` arbiter route), `062`/`063` record 0 user forks, and the wave closures are `DC-*` dispositions — governance records, not user claims (§1). The Yield rulings the dispatch points at (the `.encode("utf-8")` coercion, the post-commit abandon) are rowed as ordinary source-checkable atoms (355, 369–371), not as user-claim rows.

## Two directions (§1), as rows not judgments

**The seams consume what the waves created.** S12's "one scope, gate and writer" premise is now carried by four rows (348–351) rather than by the detector and writer readings 253/332 gave it, and S12's empty-body dimension runs through 351 and 364, not through 254/259 alone. S13's identity chain now runs through 352–356 (operand, canonical form, coercion, slice) plus 333's re-check — the six `lang:python` rows 279–285 that run 35 recorded as phantom premises now have a production call site, which is the observation the routing lane asked to be made at the accepted-local recheck; this pass records that the atoms exist (353/354/355) and leaves whether the pair still composes to the composer (E8). S14's stamp dimensions run through 357–361 and its sweep dimensions through 362–364. S10's credential dimension runs through 365–368 and 373, with 379 being the library premise that forces the duplicate-spelling case onto `raw_proxy` (376–378). S11's post-commit dimension runs through 369–372 and 374. **The peers consume what the waves left behind:** `cleanup_expired_images` gained a caller, so 300/301's caller absence and 318/319's peer-reclamation contrast are now statements about different code than they were; `forward_headers` gained three callers, so 209/212's pre-wave readings and 214/215/217/320/321's "under `routes/`" populations are now statements about `utils.py` alone; and `has_images`/`strip_and_cache_images` now share a function, which is the property S12's message-position dimension needs and no pre-wave row states.

## Collected and deliberately NOT rowed — routes recorded, nothing silent (E7)

- **`_image_key`'s `TypeError` propagation** (an unserializable `source` reaches `handle_messages` through no guard) — the library half is row 282 (`HOLD`, `json.dumps` raises `TypeError`) and the repo half is row 263's re-check (no `try` covers the guarded call); a row saying "`_image_key` has no `try`" would put one proposition in two rows. What the *composition* of the two means for the status class is S12/S14's dimension, not a verdict (E8).
- **`_has_image_in_blocks`'s missing-`source` case** — carried forward from runs 31 and 33's lists unchanged: the block is settled by the read 352's own unit is about, and *what should happen* is a plan gap for composition and Forward.
- **`image.py`'s walrus in `_scoped_contents`' comprehension and the `[k for k in headers …]` deletion list** — implementation shape inside units rows 349 and 367 already name; restating them is a second row for one read.
- **`config.IMAGE_CACHE_TTL` / `IMAGE_CACHE_MAX_SIZE`'s new dual role** (one bound now read by the writer's sweep as well as the read path) — the two call-time reads are rows 298/304, both re-dispatched; the third site is inside 362's unit, and a row per site would be a census (E4-adjacent).
- **`tests/test_image_stability.py`'s W1–W7 case list, `tests/test_header_parity.py`'s M1–M8 matrices, the `SIX_NAMES`/`ALL_NAMES` word sets, and `test_stream_relay_commit.py`'s four fault params** — acceptance sets the tasks' own `Done when` carry (E3/E4); the atomic ingredients a blind checker can answer are 374–378 and 383.
- **Whether `_now` being a module attribute rather than an injected parameter makes the TTL witness hermetic, whether the tail sweep is the right place, whether content-addressing beats the counter, whether the post-commit `except Exception` is the right handler, whether 206/373's pair or 253/348's pair is what the seams keep** — E1/E3/E8. These are dispositions for the composition, decide and routing lanes; this pass moved cells, wrote rows and membership, and judged nothing.
- **Suite passage, test counts, coverage, mutation results, the checkpoint list, `DC-*`/`QB:*` dispositions, wave ordering and gate state** — session governance and arbiter state (E4), checkable only by reading this session directory, which would spend the checker's blindness.
- **One tension this pass could not resolve by itself, recorded for the orchestrator:** the skill's depth limits say claims are "third-party only" and that repo rows in a ledger are history, while its §2 subject typing, §3c seam rules, §3d ledger-matching table and "Re-running at a phase boundary" section — and this session's 34-run convention, which the dispatch invokes by name — all require exactly the repo census and repo new-surface rows the dispatch orders. The dispatch and the operative sections governed; the "third-party only" bullet did not. If the bullet is current policy, it contradicts the phase-boundary rules and needs an amendment, not a silently split convention.

No claim was checked, no composition performed, no design claim written, no row carries more than one proposition, nothing was filtered for risk, and no seam — amended, inherited or new — was called safe.

---

## Run 36 scope correction — ledger scope correction (2026-10-06)

**Authority, cited before any cell.** Three rules plus the user's confirmation of this reading for this session:

1. **`run-codify` assumption gate** (`references/assumption-gate.md`, as reloaded): *"**Third-party subjects only.** The gate covers libraries, runtimes and external services. Repo code is not a subject, and a plan change that touches no third-party subject gives this gate nothing to run."*
2. **`forward-collect`, Limits on depth, width and rechecking**: *"`repo:` rows and repo-only seams already in a ledger are history: do not recheck or recompose them"*, and *"Check once, recheck only for a listed reason … Nothing else marks a row `recheck`: not a move elsewhere in the file, a re-base, a new phase, a new task relying on it, or doubt. A recheck with no reason from this list is not dispatched."*
3. **`forward-collect` §2 subject typing**: *"`repo:` **property** claims go `recheck` only when the named unit or aspect changed — a byte/line move elsewhere in the same file does NOT fire them; pin re-freshening is bookkeeping that rides the next atomic pass."*

Run 36's dispatch — recheck the repo census, then extract repo new-surface premises — **was** the scope defect. This run's own §"Collected and deliberately NOT rowed" recorded the conflict rather than resolving it; the user has now resolved it **against the run**: the third-party bullet is current policy, and §2's subject typing, §3c's seam rules, §3d's ledger-matching table and "Re-running at a phase boundary" operate under it, not over it. Rows 348–383 and every repo-subject `recheck` in the census were created or fired outside the gate's subject authority, and the tension note closes here rather than staying an open amendment question.

**What this pass writes, and does not.** Cell dispositions and their reasons, in the two ledgers this phase owns — nothing else. No extraction, no new rows, no bundle created, nothing checked, no seam composed, no verdict written, re-issued or copied into a State cell, no invariant touched, no promotion (E1/E8). Where a row's sentence is contradicted — by the accepted tree, or by a blind verdict already on file — this pass records that, because the alternative, restoring a `HOLD` the evidence contradicts, would put false support into a ledger the composition and Yield lanes cite. That is a disposition, not a verdict: no row gains `HOLDS` or `DOES-NOT-HOLD` here, and no superseded row is a finding.

Hashes recorded before the first cell write: `claims.md` `sha256:98677d496db506b099b9a104d61becdec2642f9e785aee030eba9eb99922de4b`, `seams.md` `sha256:5d0c6922c111c109c316603d2749d2c059f2297f66ed57806de628372d4700de` (run 36's committed output at `b01cf5c`). The baseline the reverts go back to is `4c0f4eb` — `claims.md` `sha256:787f94fe18de97618247b86a3da25149ae7f2ba709895f2e80965a0ea09ce904`, `seams.md` `sha256:abc66a586d57d088a1456686648a88a60e47f98fecbeb03a07427929b6bd6e83` — read out of Git, byte for byte, not from memory.

### 1. Reverted — 54 repo rows whose only trigger was out of scope

Each returns to its exact `4c0f4eb` cells: pre-run-36 `Bundle`, `State: HOLD`, `Routing: none`, `Permanent` handle retained. Two tests have to pass for a revert: the gate may not recheck a `repo:` row at all, **and** nothing on the current tree or in a verdict already on file may contradict the sentence. Where one of them did, the row went to §2 instead.

| Row | Trigger run 36 recorded | Basis for the revert |
|---|---|---|
| 13 | `U, P` | `handle_messages` was edited, but the guarded dump block's swallow-all `try/except Exception` is unchanged at `:57-61` — the wave's own code change left the proposition standing |
| 17 | `P, I` | the census population gained `_image_key`, `_cache_source`, `_now`, `_scoped_contents`, `_user_content`, `_relay_after_commit`, `_note_message_start`, `_carries_token` and five test modules, none of which owns a dump buffering, batching or flush role. A population change is not a listed trigger |
| 26 | `P` | all three pins moved; every outbound call still goes through `app["session"]` (`server.py:66`, `:72`; `chat.py:37`, `messages.py:127`, `:256`, `passthrough.py:26`) — a byte move elsewhere in three files |
| 45 | `I` | its bullet is repo-wide by construction ("any byte of any file under `buster_ripper/`"); §2 says a byte move elsewhere does not fire a `repo:` property row, and `dump.py` is unmoved (`07b0b076…`) |
| 46 | `P` | same family — the periodic-flush absence stands on unmoved `dump.py`/`server.py` bytes |
| 56 | `L` | `chat.py:146` renumbered; the `except (json.JSONDecodeError, UnicodeDecodeError)` 502 arm is intact in `_make_chat_request`, a unit T013 did not edit |
| 61 | `L` | `chat.py:122`→`:141`, `passthrough.py:26`→`:27`; still exactly three per-request `ClientTimeout` call sites (`server.py:67` is the session's own default, not a call site) |
| 67 | `U, P` | the relay moved out of `_stream_single_request` into `_relay_after_commit`, but the chain still prepares a `StreamResponse` and writes `resp.content.iter_any()` chunks through unchanged (`messages.py:145-156`, `:180-184`) |
| 73 | `U, P` | the `stream` dispatch is unchanged at `:66-71` |
| 129 | `I` | the routes-family parse census still lands on messages, chat and count_tokens |
| 130 | `U` | the `UnicodeDecodeError` half of the arm at `:28-31` still maps to a JSON 400 |
| 131 | `U` | same arm, same JSON 400, for `json.JSONDecodeError` |
| 197 | `U, P` | the `if config.DUMP_DIR is not None:` guard is unchanged at `:57` |
| 200 | `P` | `conftest.py` was rewritten, but `serve_buster_ripper.sh` is still invoked only by `tests/test_serve_script_argv.py` |
| 213 | `U`, **sentence** | T013 did replace the mechanism this row names (the single-value `strip() in ("Bearer", "Bearer ")` deletion is gone), but the blind check that ran on the post-wave bytes returned **`HOLDS`** (`verdicts/claim-213.md`, 08:17), so the current tree does not contradict the sentence and a `HOLD` is not false support. See §7 |
| 215 | `I`, **sentence** | the response-side filter population under `routes/` dissolved, yet the blind check on current bytes returned **`HOLDS`** (`verdicts/claim-215.md`, 08:21): nothing in the tree refutes the row. Returns to `HOLD`. See §7 |
| 219 | `I` | the relation's own bullet fired because `utils.py`'s bytes moved *outside* `_HOP_BY_HOP` (`:12-15`, byte-identical) while the legacy file is unmoved (`7967563d…`). Neither end nor the mechanism moved, so §2's `rel:` rule did not fire either |
| 225 | `I` | `test_header_parity.py:4` and `:239` name the legacy module in docstrings; no import was added under `buster_ripper/` or `tests/` |
| 242 | `U` | `_stream_single_request` still has exactly one `except`, and it is `aiohttp.ClientError` (`:158`); the new `Exception` arm is in a different unit |
| 243 | `U` | that arm still returns a newly constructed `web.json_response` (`:160`) |
| 250 | `P, I` | the plan's bytes moved under G-2, but the paths it names are the same and every `buster_ripper/convert/` path is still absent — there is no `convert/` directory |
| 254 | `U` | `has_images({})` and a missing or empty `messages` still answer False, now through `_scoped_contents` |
| 257 | `U` | `_user_content` still returns None for a non-`user` role, so those messages stay untouched |
| 258 | `U` | a `user` message with non-list content still yields None and is skipped |
| 259 | `U` | the early return on absent / non-list / empty `messages` is unchanged (`:188-190`) |
| 260 | `P` | module-wide absence over a file that gained units but no `tool_result` path (`DC-F-1-r1` carries that as deferred additive work) |
| 261 | `U` | the guard at `:40` still requires both `config.STRIP_IMAGES` and `has_images(body)` |
| 262 | `U` | `changes.append("stripped and cached images")` is still inside that same branch (`:42`) |
| 263 | `U` | no `try` covers `:40-42`; the handler's two `try` blocks remain the parse at `:28` and the dump at `:58` |
| 267 | `R` | a record defect on unmoved bytes: `normalize.py` is identical to run 31's read (`a7ddbf06…`) and the proposition is not refuted. The missing `## Source Fingerprint` is `QB:assumption-registry/run-31-verdict-fingerprints` material, not a dispatch reason |
| 268 | `L, P` | the package's only `normalize` call still passes `session_id=""` (`messages.py:36`) |
| 274 | `U` | the returned block is still `{"type": "text", "text": f"[image:{image_key}]"}` (`:148-151`) |
| 275 | `U` | `_evict_by_lru` is still called after the insert (`:142-145`) |
| 276 | `U` | a write onto an existing key still replaces that key's entry, carrying `first_seen` forward (`:128-134`) |
| 277 | `U` | stored element 0 is still the request body's own `source` mapping, not a copy (`:139` → `:134`) |
| 287 | `P, R` | `get_cached_image` still has no caller under `buster_ripper/` |
| 291 | `P` | the eviction cases still cache one distinct `source` per image (`one_image({"id": i})`) |
| 292 | `P` | `test_get_cached_image_returns_none_for_missing` still asserts None for the never-cached `"img_999"` |
| 294 | `P` | the autouse `reset_image_cache` still resets before and after each test (`:36-41`) — a second such fixture elsewhere in the tree is not this row's population |
| 295 | `U` | `_cleanup_expired_images` still deletes only from the session mapping (`:87-88`); the outer drop is in `cleanup_expired_images` |
| 296 | `U` | the comparator still ages on the entry's second slot, `first_seen` (`:83-86`) — the tuple grew a third slot, the aged one did not move |
| 297 | `U` | still a strict `>` against `config.IMAGE_CACHE_TTL` |
| 298 | `U` | still a call-time attribute read, not an import binding |
| 301 | `I, P` | `clear_session_cache` still has no caller under `buster_ripper/`; the test callers at `test_image.py:340`, `:345` are outside the row's stated population |
| 302 | `U` | eviction still reduces the session mapping to `config.IMAGE_CACHE_MAX_SIZE` (`:96-105`) |
| 304 | `U` | still a call-time read of the bound |
| 305 | `U` | still scoped to one session's mapping |
| 306 | `U` | `_image_cache` is still module-level state shared by every request in a process (`:28`) |
| 307 | `U` | the read path still calls `_cleanup_expired_images(session_id)` and then indexes (`:217-222`) |
| 315 | `R` | a record defect on unmoved bytes: `server.py` is unchanged (`a9e49823…`) and `create_app` still registers `middlewares=[logging_middleware]` (`:119`) |
| 321 | `U` | the bare-key name iteration is still there (`chat.py:118`); T013 added a `getall` **value** selection beside it, which the row does not speak to |
| 332 | `U` | `_scoped_contents` still considers every `user` message in forward list order (`image.py:58`) |
| 339 | `P` | `test_strips_multiple_images` still passes two image blocks in one `strip_and_cache_images` call (`:216-224`) |
| 346 | `P` | `test_clear_removes_session` still calls `image.clear_session_cache` between two of its own assertions (`:338-341`) |

**`CL:` handles come back with the cells** on 13, 17, 26, 45, 46, 56, 67, 73, 130, 131 and 197 — a revert restores the cell run 36 overwrote; it neither re-opens nor re-freshens a registry record. Rows **54** and **65** keep their handles too, but under §2b's terminal cell, and the same sentence applies to them: a retained handle is not a claim of currentness. Run 36's separate observation that these records pin expired bytes stands as a **registry** fact for the promotion lane, which is not a dispatch reason for this gate.

### 2a. Superseded by the accepted waves — 24 repo rows the waves genuinely invalidated

State `superseded`, Routing `none`, Bundle `—`: undispatchable and unciteable, and explicitly **not** `DOES-NOT-HOLD` — no checker answered these rows on these bytes and this pass is not a checker. The named wave supersedes them; the waves' own closure records (`057` F-1…F-4/G-1/G-2, `063` R-A/R-C/R-D, the `DC-*` ledger entries) and the tests that landed with them are the current record of the behaviour.

| Row | Wave | What the current tree asserts instead |
|---|---|---|
| 206 | T013 | `response_headers` now drops `("content-length", *_HOP_BY_HOP)` (`utils.py:34-37`), so "exactly the names in `_HOP_BY_HOP` and no others" is refuted by the unit it names |
| 209 | T013 | three production callers now — `chat.py:103`, `messages.py:64`, `passthrough.py:17` — so "never called" is refuted |
| 210 | T013 | now called at `messages.py:285` and `passthrough.py:41` |
| 212 | T013 | `chat.py` calls what it imports: `_prepare_headers` → `forward_headers(request_headers)` |
| 214 | T013 | referent dissolved: the five route-local filters this row quantified over were replaced by the shared helpers. Superseded, **not refuted** — a vacuous population is not current support |
| 217 | T013 | the same dissolution of the `routes/` filter population this row quantified over |
| 241 | T013 | the `try` no longer spans a relay loop — the loop is `_relay_after_commit`'s own `try` with an `Exception` arm (`messages.py:129-160`, `:180-190`), which is the split `DC-G-1-r2` landed |
| 245 | T013 | the `.get()` on the `message_start` payload now happens in `_note_message_start` (`:200`); `_stream_single_request` no longer performs it |
| 246 | T013 | `tests/test_header_parity.py` asserts hop-by-hop names dropped at the proxy boundary (`:30`, `:75`, `:241-242`) |
| 247 | T013 | `test_header_parity.py:267-283` asserts `Authorization` values, forwarded and withheld |
| 248 | T013 | `tests/test_stream_relay_commit.py` drives faults raised after the first chunk through the committed relay (`:46-58`, `:85-100`) |
| 253 | T014 | `has_images` scans **every** user message through `_scoped_contents` (`image.py:61-69`) — the dispatch's own example of a HOLD the current tree contradicts |
| 272 | T014 | the key is content-derived from the block's `source` (`_image_key`, `:139-141`), not the session mapping's entry count |
| 288 | T014 | the rewritten file's positive placeholder assertions pin keys the helpers compute — `test_image.py:214` pins `img_04b45e9ae5ea`, a digest slice, and `:241` reads `placeholders(request)[0]`; the count-derived literal form the row described is gone |
| 290 | T014 | the position table asserts `True` for the earliest-image / latest-text-only body (`test_image.py:74-78`, `:103-107`), inverting the pre-wave pin (`DC-F-1-r2`) |
| 293 | T014 | expiry is asserted on `_image_cache` directly (`test_image.py:259`, `:295`, and `test_image_stability.py`), so "only through a `get_cached_image` call" is refuted |
| 299 | T014 | the janitor also drops the emptied session key (`image.py:246-249`) — "performs no other operation" is refuted |
| 300 | T014 | `cleanup_expired_images` now has a production caller at `image.py:198`, plus test callers |
| 303 | T014 | eviction orders on the entry's **third** slot, `last_touched` (`image.py:100-102`) |
| 308 | T014 | `get_cached_image` unpacks three targets (`:222`) — "exactly two" is refuted |
| 316 | T014 | test sources now set `STRIP_IMAGES` true — `test_image.py:140`, `:150`, `:161`, `:192` and `test_image_stability.py:93` |
| 320 | T013 | the one surviving `routes/` header comprehension iterates **bare keys** (`chat.py:118`), the shape this row excludes; the `.items()` population is gone |
| 333 | T014 | the stored value is a `(source, first_seen, last_touched)` triple (`:134`), not a `(source, timestamp)` pair |
| 345 | T014 | `reset_cache()` empties the whole mapping (`image_bodies.py:85-87`) and is called from inside test function bodies (`test_image.py:283`, `:302`, `:320`), so "inside the autouse fixture rather than inside a test function body" is refuted |

Read 253, 272, 293, 300, 303, 308 and 333 as the pattern: a row that stated *how the code worked before* now states nothing current — run 36's own "sentence" column already said so, and those sentences are why a plain scope revert was not enough for these rows. Five of the twenty-four are corroborated by a blind verdict that landed after the census (206, 212, 214, 320 `DOES-NOT-HOLD`, 217 `UNCLEAR` — §7), which is the strongest form of the same finding.

### 2b. Superseded by evidence already on file — 3 rows a blind check settled against a `HOLD`

The census's repo bundles were dispatched before this correction ran, and these three rows came back non-holding (§7). This pass cannot revert them to `HOLD` — that is exactly the false support the correction exists to prevent — and it will not copy a verdict token into a State cell, which is the promotion lane's write. Terminal and flagged is the middle it can honestly occupy.

| Row | Verdict on file | Token | Why a revert is not available |
|---|---|---|---|
| 53 | B248 · `verdicts/claim-53.md` (08:16) | `UNCLEAR` | The checker could not settle a package-wide count from the bounded source root it was given, so the row has an answer that is neither `HOLDS` nor refuted. Restoring `HOLD` would assert what no verdict established |
| 54 | B248 · `verdicts/claim-54.md` (08:16) | `UNCLEAR` | The verdict forks on "typed" and on whether the cited ranges are assertions or pointers. Choosing a reading is the checker's and the decide lane's job, not a scope correction's |
| 65 | B247 · `verdicts/claim-65.md` (08:16) | `DOES-NOT-HOLD` | `_make_chat_request` returns `_relay_chat_stream(request, resp)` for a `stream` body instead of parsing it, which the row's "regardless of the request's `stream` flag" denies. This one predates the wave — the census's `L` trigger was right that something was wrong with the row, and the checker found what |

### 3. Retired — 35 repo rows this gate had no authority to create

Rows 348–383 are `repo:` subjects. Under authority (1) the gate may not write them at all, so each is **created in error** — not extracted-and-then-refuted, and not "never extracted" either. The row stays in the ledger with its text and its seam reference so the trail shows what was written and what it was retired for. Cells written: `State` `new` → `retired`, `Routing` `unverified` → `none`, and `Check`/`Bundle` voided to `—` (a row the gate may not create has no check reason). The run-36 cells are reproduced per row below, so nothing is lost. No Bundle is allocated, no checker is dispatched, and none of these rows enters a composition.

Row **379** (`pkg:aiohttp@3.13.3` — the client merging two request header names that differ only in case) is the exception: a third-party subject, a legitimately in-scope new premise, and it **stays dispatched** on `B287` with `Check: native, seam:S10`, `State: new`, `Routing: unverified`. Its cells are untouched by this pass.

| Row | Subject | Seam membership it wrote | Run-36 cells (history) | Per-row basis for retirement |
|---|---|---|---|---|
| 348 | `repo:buster_ripper.image#has_images` | S12 | Check `seam:S12` → `—` · Bundle `B276` → `—` · State `new` → `retired` · Routing `unverified` → `none` | `repo:` subject — the strip-scope atoms — the shared `_scoped_contents`/`_user_content` section that now stands behind both the gate and the writer |
| 349 | `repo:buster_ripper.image#strip_and_cache_images` | S12 | Check `seam:S12` → `—` · Bundle `B276` → `—` · State `new` → `retired` · Routing `unverified` → `none` | `repo:` subject — the strip-scope atoms — the shared `_scoped_contents`/`_user_content` section that now stands behind both the gate and the writer |
| 350 | `repo:buster_ripper.image#_user_content` | S12 | Check `seam:S12` → `—` · Bundle `B276` → `—` · State `new` → `retired` · Routing `unverified` → `none` | `repo:` subject — the strip-scope atoms — the shared `_scoped_contents`/`_user_content` section that now stands behind both the gate and the writer |
| 351 | `repo:buster_ripper.image#_scoped_contents` | S12 | Check `absence, seam:S12` → `—` · Bundle `B276` → `—` · State `new` → `retired` · Routing `unverified` → `none` | `repo:` subject — the strip-scope atoms — the shared `_scoped_contents`/`_user_content` section that now stands behind both the gate and the writer |
| 352 | `repo:buster_ripper.image#_image_key` | S13 | Check `seam:S13` → `—` · Bundle `B277` → `—` · State `new` → `retired` · Routing `unverified` → `none` | `repo:` subject — the placeholder key-serializer atoms — operand, `sort_keys`, `separators`, `.encode("utf-8")`, digest slice |
| 353 | `repo:buster_ripper.image#_image_key` | S13 | Check `seam:S13` → `—` · Bundle `B277` → `—` · State `new` → `retired` · Routing `unverified` → `none` | `repo:` subject — the placeholder key-serializer atoms — operand, `sort_keys`, `separators`, `.encode("utf-8")`, digest slice |
| 354 | `repo:buster_ripper.image#_image_key` | S13 | Check `seam:S13` → `—` · Bundle `B277` → `—` · State `new` → `retired` · Routing `unverified` → `none` | `repo:` subject — the placeholder key-serializer atoms — operand, `sort_keys`, `separators`, `.encode("utf-8")`, digest slice |
| 355 | `repo:buster_ripper.image#_image_key` | S13 | Check `seam:S13` → `—` · Bundle `B277` → `—` · State `new` → `retired` · Routing `unverified` → `none` | `repo:` subject — the placeholder key-serializer atoms — operand, `sort_keys`, `separators`, `.encode("utf-8")`, digest slice |
| 356 | `repo:buster_ripper.image#_image_key` | S13 | Check `seam:S13` → `—` · Bundle `B278` → `—` · State `new` → `retired` · Routing `unverified` → `none` | `repo:` subject — the placeholder key-serializer atoms — operand, `sort_keys`, `separators`, `.encode("utf-8")`, digest slice |
| 357 | `repo:buster_ripper.image#_now` | S14 | Check `seam:S14` → `—` · Bundle `B279` → `—` · State `new` → `retired` · Routing `unverified` → `none` | `repo:` subject — the monotonic-stamp and retention atoms — the `_now` binding, its call sites, the triple, `first_seen` carry-forward, the wall-clock absence |
| 358 | `repo:buster_ripper.image` | S14 | Check `seam:S14` → `—` · Bundle `B279` → `—` · State `new` → `retired` · Routing `unverified` → `none` | `repo:` subject — the monotonic-stamp and retention atoms — the `_now` binding, its call sites, the triple, `first_seen` carry-forward, the wall-clock absence |
| 359 | `repo:buster_ripper.image#_cache_source` | S14 | Check `seam:S14` → `—` · Bundle `B279` → `—` · State `new` → `retired` · Routing `unverified` → `none` | `repo:` subject — the monotonic-stamp and retention atoms — the `_now` binding, its call sites, the triple, `first_seen` carry-forward, the wall-clock absence |
| 360 | `repo:buster_ripper.image#_cache_source` | S14 | Check `seam:S14` → `—` · Bundle `B279` → `—` · State `new` → `retired` · Routing `unverified` → `none` | `repo:` subject — the monotonic-stamp and retention atoms — the `_now` binding, its call sites, the triple, `first_seen` carry-forward, the wall-clock absence |
| 361 | `repo:buster_ripper.image` | S14 | Check `absence, seam:S14` → `—` · Bundle `B280` → `—` · State `new` → `retired` · Routing `unverified` → `none` | `repo:` subject — the monotonic-stamp and retention atoms — the `_now` binding, its call sites, the triple, `first_seen` carry-forward, the wall-clock absence |
| 362 | `repo:buster_ripper.image#strip_and_cache_images` | S14 | Check `seam:S14` → `—` · Bundle `B281` → `—` · State `new` → `retired` · Routing `unverified` → `none` | `repo:` subject — the sweep-position atoms — the tail sweep's call, the emptied-session drop's owner, the early return that skips it |
| 363 | `repo:buster_ripper.image#cleanup_expired_images` | S14 | Check `seam:S14` → `—` · Bundle `B282` → `—` · State `new` → `retired` · Routing `unverified` → `none` | `repo:` subject — the sweep-position atoms — the tail sweep's call, the emptied-session drop's owner, the early return that skips it |
| 364 | `repo:buster_ripper.image#strip_and_cache_images` | S14 | Check `absence, seam:S14` → `—` · Bundle `B281` → `—` · State `new` → `retired` · Routing `unverified` → `none` | `repo:` subject — the sweep-position atoms — the tail sweep's call, the emptied-session drop's owner, the early return that skips it |
| 365 | `repo:buster_ripper.routes.chat#_carries_token` | S10 | Check `seam:S10` → `—` · Bundle `B283` → `—` · State `new` → `retired` · Routing `unverified` → `none` | `repo:` subject — the credential-selection atoms — the `_carries_token` predicate and the `getall`/delete/reinsert sequence |
| 366 | `repo:buster_ripper.routes.chat#_prepare_headers` | S10 | Check `seam:S10` → `—` · Bundle `B283` → `—` · State `new` → `retired` · Routing `unverified` → `none` | `repo:` subject — the credential-selection atoms — the `_carries_token` predicate and the `getall`/delete/reinsert sequence |
| 367 | `repo:buster_ripper.routes.chat#_prepare_headers` | S10 | Check `seam:S10` → `—` · Bundle `B283` → `—` · State `new` → `retired` · Routing `unverified` → `none` | `repo:` subject — the credential-selection atoms — the `_carries_token` predicate and the `getall`/delete/reinsert sequence |
| 368 | `repo:buster_ripper.routes.chat#_prepare_headers` | S10 | Check `seam:S10` → `—` · Bundle `B283` → `—` · State `new` → `retired` · Routing `unverified` → `none` | `repo:` subject — the credential-selection atoms — the `_carries_token` predicate and the `getall`/delete/reinsert sequence |
| 369 | `repo:buster_ripper.routes.messages#_stream_single_request` | S11 | Check `seam:S11` → `—` · Bundle `B284` → `—` · State `new` → `retired` · Routing `unverified` → `none` | `repo:` subject — the post-commit relay-split atoms — the delegation, the arm's class, its non-reraise, the state mapping the timing origin travels in |
| 370 | `repo:buster_ripper.routes.messages#_relay_after_commit` | S11 | Check `seam:S11` → `—` · Bundle `B284` → `—` · State `new` → `retired` · Routing `unverified` → `none` | `repo:` subject — the post-commit relay-split atoms — the delegation, the arm's class, its non-reraise, the state mapping the timing origin travels in |
| 371 | `repo:buster_ripper.routes.messages#_relay_after_commit` | S11 | Check `seam:S11` → `—` · Bundle `B284` → `—` · State `new` → `retired` · Routing `unverified` → `none` | `repo:` subject — the post-commit relay-split atoms — the delegation, the arm's class, its non-reraise, the state mapping the timing origin travels in |
| 372 | `repo:buster_ripper.routes.messages#_note_message_start` | S11 | Check `seam:S11` → `—` · Bundle `B284` → `—` · State `new` → `retired` · Routing `unverified` → `none` | `repo:` subject — the post-commit relay-split atoms — the delegation, the arm's class, its non-reraise, the state mapping the timing origin travels in |
| 373 | `repo:buster_ripper.utils` | S10 | Check `seam:S10` → `—` · Bundle `B245` → `—` · State `new` → `retired` · Routing `unverified` → `none` | `repo:` subject — the response-side drop-set atom — `content-length` alongside `_HOP_BY_HOP` |
| 374 | `repo:tests.upstream_stub` | S11 | Check `seam:S11` → `—` · Bundle `B285` → `—` · State `new` → `retired` · Routing `unverified` → `none` | `repo:` subject — the upstream stand-in's fault hook, the witness mechanic S11's post-commit cases rest on |
| 375 | `repo:tests.upstream_stub` | — | Check `shared` → `—` · Bundle `B285` → `—` · State `new` → `retired` · Routing `unverified` → `none` | `repo:` subject — the upstream stand-in's surface (`status`/`headers`/`read`/`text`/`json`) — test-tree code |
| 376 | `repo:tests.conftest` | S10 | Check `seam:S10` → `—` · Bundle `B286` → `—` · State `new` → `retired` · Routing `unverified` → `none` | `repo:` subject — the raw-loopback fixture — test-tree code |
| 377 | `repo:tests.conftest` | S10 | Check `seam:S10` → `—` · Bundle `B286` → `—` · State `new` → `retired` · Routing `unverified` → `none` | `repo:` subject — the chunked-terminator read termination — test-tree code |
| 378 | `repo:tests.conftest` | S10 | Check `seam:S10` → `—` · Bundle `B286` → `—` · State `new` → `retired` · Routing `unverified` → `none` | `repo:` subject — the `TestClient` fixture — test-tree code |
| 380 | `repo:tests.image_bodies` | S12 | Check `seam:S12` → `—` · Bundle `B288` → `—` · State `new` → `retired` · Routing `unverified` → `none` | `repo:` subject — test body-builder / cache-reset / clock-patch witness mechanics — test-tree code |
| 381 | `repo:tests.image_bodies` | S13, S14 | Check `seam:S13/S14` → `—` · Bundle `B288` → `—` · State `new` → `retired` · Routing `unverified` → `none` | `repo:` subject — test body-builder / cache-reset / clock-patch witness mechanics — test-tree code |
| 382 | `repo:tests.image_bodies` | S14 | Check `seam:S14` → `—` · Bundle `B288` → `—` · State `new` → `retired` · Routing `unverified` → `none` | `repo:` subject — test body-builder / cache-reset / clock-patch witness mechanics — test-tree code |
| 383 | `repo:tests.test_image_stability` | S14 | Check `seam:S14` → `—` · Bundle `B289` → `—` · State `new` → `retired` · Routing `unverified` → `none` | `repo:` subject — a second test file's autouse reset fixture — test-tree code |

| **379** | `pkg:aiohttp@3.13.3` | S10 | Check `native, seam:S10` · Bundle `B287` · State `new` · Routing `unverified` | **kept** — `pkg:` subject, in scope, dispatched normally |

### 4. Third-party rows: the restore set is empty, and named as such

Authority (3) says a `pkg:`/`lang:` row never fires on a repo edit, so every such row run 36 fired had to be restored with its false trigger named. It fired **none**: the 81 census rows parse to `repo:` ×80 and `rel:` ×1 — row 219, the relation between `buster_ripper_legacy.py#_HOP_BY_HOP` and `buster_ripper/utils.py#_HOP_BY_HOP`, repo code at both ends and therefore outside the gate's subject authority on that count too (reverted in §1). Run 36 reached the same place from the other direction in its own "not marked stale" note: no package, interpreter or documented source moved — aiohttp 3.13.3, click 8.3.1, pytest 9.0.2, Python 3.13.8, unchanged since run 1's `.venv` read. **No `pkg:`/`lang:` restore was needed and none was made, and no `pkg:`/`lang:` row was retired or superseded.** The empty set is the verified result of parsing all 383 subjects, not an unexamined claim.

### 5. Corrected dispatch, and the run-36 statements this voids

The dispatch filter is `State ∈ {new, recheck}` ∧ `Routing: unverified` ∧ `Check != —`. After this pass the whole run-36 delta satisfies it once:

| | run 36 recorded | after this correction |
|---|---|---|
| rows marked `recheck` | 81 (all repo/rel) | **0** |
| rows marked `new` and dispatched | 36 (35 repo + 1 `pkg:`) | **1** — row 379 |
| checker agents | 45 (B245–B289) | **1** — B287 |
| seams awaiting recomposition | S10–S14 at `recheck` | **0** — see [seams.md](seams.md) §Run-36 scope correction |
| run-36 delta disposition | — | reverted **54** · superseded **27** (24 by waves, 3 by verdicts on file) · retired **35** · kept **1** |

Bundle keys **B245–B286 and B288–B289 are void** and must not be dispatched: every row they carried is a `repo:` subject, now reverted, superseded or retired, and none of them carries a `Bundle` cell any more. They stay listed in §Run 36's bundle table as the record of what the erroneous pass allocated — that table's heading, the "45 bundles ⇒ 45 checker agents" paragraph, the Duty A and Duty B headings all carry inline void markers, so no reader reaches a void bundle through a live row.

Operative statements in the run-36 sections this pass supersedes, listed so none of them reads as a live instruction while the prose stays on file for the audit: the Duty A heading's *"81 live rows moved to `recheck`"*; every trigger cell in the Duty A census table (P/U/L/I/R against repo rows is history-bookkeeping, not a firing condition); Duty B in its entirety ("new-surface extraction" over repo code); the "45 bundles ⇒ 45 checker agents" paragraph and its "31 carrying the 81 re-dispatched rows, 14 carrying the 36 new rows" split; "**117 rows are `new` or `recheck` with `Routing: unverified`**"; the re-issue dispositions for 246/247/267/287/293/294/297/298/315/316 (record defects are `QB:assumption-registry/run-31-verdict-fingerprints` material, and 246/247/248/293/316 are superseded here, so there is nothing to re-issue); the §"Two directions" paragraph's seam-consumes-what-the-waves-created reading, which is built on retired rows; and the "Pairs deliberately kept apart from retired wording" paragraph, which after this pass has no live second half. The freshness basis, the "what moved" table, the hashes and the census's *observations* remain valid as history — the waves did move those bytes, and this correction does not pretend otherwise.

### 6. What still owes a check, and to whom it belongs

Nothing in this gate is left outstanding on the repo side, and the repo side is not small: T013's header parity, `_prepare_headers`' credential selection and the post-commit relay abandon, and T014's strip scope, content-addressed key, monotonic stamps, triple shape and tail-sweep lifecycle are implemented and accepted. Whether that implementation **conforms** is a conformance question about repo code, and under authority (1) it belongs to **Yield Phase and the external review** — the phase-scoped QA gate tests, `tests/test_header_parity.py`, `tests/test_image_stability.py`, `tests/test_stream_relay_commit.py`, `tests/test_routes_integration.py`, the traceability and parity passes, and the review in front of the user — not to this gate. The gate's honest remaining scope after this correction is one library premise: row 379.

Recorded rather than fixed, because fixing it is not this phase's authority: **(a)** `superseded` and `retired` are now States in this ledger and are absent from the skill's State vocabulary — the routing and promotion lanes treat them as terminal because `Routing` is `none`, and if they become permanent, `forward-collect` and the gate's steps (ii)/(iii) filters should name them; **(b)** rows **48** and **338** keep their pre-existing `Routing: unverified` cells from runs 32/33/34 with `B227`/`B240` pointing at retired wording — not this pass's writes, untouched, and visible here instead of silently tidied; **(c)** S9's `State: recheck` is run 30's cell and this pass did not touch it — see [seams.md](seams.md) §Run-36 scope correction.

### 7. Handoff — blind verdicts that landed after run 36's census were dispatched

Bundles B245–B252 ran before this correction, so 23 verdict files were rewritten on post-wave bytes at 08:16–08:21 (`verdicts/claim-*.md`; 373 is new on file). This pass read **only** the `VERDICT:` token of each, to make sure no cell it wrote contradicts an answer already in hand. It did not read them as instructions, did not copy a token into a State cell, and did not let a verdict decide a disposition this file's authority rules already settle: the retired rows stay retired whatever their checkers said, because the defect is the row's existence, not its answer. What the orchestrator owns now is the routing of this evidence, and it is listed here so none of it rides silently behind a scope correction:

| Row | Verdict on file | Token | Void bundle it came from | This ledger's cell |
|---|---|---|---|---|
| 13 | `verdicts/claim-13.md` (08:18) | `HOLDS` | B251 | reverted |
| 53 | `verdicts/claim-53.md` (08:16) | `UNCLEAR` | B248 | superseded (§2b) |
| 54 | `verdicts/claim-54.md` (08:16) | `UNCLEAR` | B248 | superseded (§2b) |
| 56 | `verdicts/claim-56.md` (08:16) | `HOLDS` | B247 | reverted |
| 65 | `verdicts/claim-65.md` (08:16) | `DOES-NOT-HOLD` | B247 | superseded (§2b) |
| 73 | `verdicts/claim-73.md` (08:18) | `HOLDS` | B251 | reverted |
| 129 | `verdicts/claim-129.md` (08:17) | `HOLDS` | B248 | reverted |
| 130 | `verdicts/claim-130.md` (08:20) | `HOLDS` | B251 | reverted |
| 131 | `verdicts/claim-131.md` (08:20) | `HOLDS` | B251 | reverted |
| 197 | `verdicts/claim-197.md` (08:18) | `HOLDS` | B252 | reverted |
| 206 | `verdicts/claim-206.md` (08:16) | `DOES-NOT-HOLD` | B245 | superseded (§2a) |
| 212 | `verdicts/claim-212.md` (08:16) | `DOES-NOT-HOLD` | B247 | superseded (§2a) |
| 213 | `verdicts/claim-213.md` (08:17) | `HOLDS` | B247 | reverted |
| 214 | `verdicts/claim-214.md` (08:21) | `DOES-NOT-HOLD` | B249 | superseded (§2a) |
| 215 | `verdicts/claim-215.md` (08:21) | `HOLDS` | B249 | reverted |
| 217 | `verdicts/claim-217.md` (08:21) | `UNCLEAR` | B249 | superseded (§2a) |
| 219 | `verdicts/claim-219.md` (08:16) | `HOLDS` | B246 | reverted |
| 261 | `verdicts/claim-261.md` (08:20) | `HOLDS` | B252 | reverted |
| 262 | `verdicts/claim-262.md` (08:20) | `HOLDS` | B252 | reverted |
| 263 | `verdicts/claim-263.md` (08:20) | `HOLDS` | B252 | reverted |
| 320 | `verdicts/claim-320.md` (08:20) | `DOES-NOT-HOLD` | B250 | superseded (§2a) |
| 321 | `verdicts/claim-321.md` (08:20) | `HOLDS` | B250 | reverted |
| 373 | `verdicts/claim-373.md` (08:16) | `HOLDS` | B245 | retired (§3) |

Nothing in this table is a finding, and nothing in it authorizes a checker: nine of the twenty-three returned `HOLDS` on current bytes, which corroborates the §1 reverts rather than requiring them, and the three non-holding verdicts in §2b are the reason those rows were not reverted. **Whether any of this evidence is promoted, routed, or left as history is the promotion and decide lanes' call, and the void bundle keys above are the record of what was spent on rows this gate had no authority to dispatch.**

No claim was checked, no composition performed, no verdict written or copied, no invariant created, nothing promoted, and no row in §1, §2a, §2b or §3 was dispatched.

---

# Run 37 — third-party premise extraction for `QB:utils/response-header-relay-fidelity` (2026-10-06)

Dispatch: **phase-boundary assumption recheck, third-party subjects only**, over the CR2 wave accepted at Checkpoint 168 on `c8df363`. Ledger entered at `claims.md` `sha256:b78459c7d0472e704e8db6c70f3cad8ccc647cd4d218b4ae8062cd494e26a411`, `seams.md` `sha256:3e745ee47b7eaa15e02dc89f189502361ccbada14f0528ae2638cd8a0fc7f248` — recorded before the first write, re-read after it.

**Scope honored, as ruled by the user this session and reloaded with the gate.** No `repo:` row is created, rechecked or recomposed. `S1`–`S14` are terminal (`out-of-scope`; `S6` citable at `registry-safe`) and were not opened. The ledger's eligible-set baseline was confirmed mechanically before this pass wrote anything: all 376 claim rows parse to `HOLD` 283 · `DOES-NOT-HOLD` 19 · `UNCLEAR` 11 · `superseded` 27 · `retired` 35 · `MALFORMED` 1, and **zero rows are in `new` or `recheck`** — so the dispatch's premise is a fact of this file, not an assumption. Extraction only: nothing here was checked, no seam composed or judged safe, no verdict read as an instruction, no invariant created, nothing promoted (E1/E8). No project code was run (E2); every command was a read.

## The four narrow sets, and what each yielded

| Set | Result |
|---|---|
| 1 — tasks not yet started whose plan text changed since extraction | **empty.** `T015`/`T016` are accepted, not started-and-moved; no open task's plan text was reworded between Checkpoint 163 and this pass. |
| 2 — checked claims with changed dependencies/domain or contradictory observations | **empty**, and named as such below (§"Why the restore set is empty"). |
| 3 — tasks that did not exist at the last extraction (`T015`, `T016`) | Their **surfaces** are repo code — chat header parity, the raw-relay fall-through, the four static `ClientError` bodies, the `chat_template_kwargs` guard — and stay outside this gate. Their **third-party premise** does not: the accepted behavior leans on a library fact no row ever covered. Six rows, 384–389. |
| 4 — seams whose member, transformation, consumer, fallback, precedence or dimension moved | `S10`–`S14` terminal, unopened. One **fresh** seam, `S15`, over this run's rows only, at the library endpoint. |

Set 3's premise, sourced not judged: `073-01-yield` §Findings D-1 raised `QB:utils/response-header-relay-fidelity` — the shared `utils.response_headers` filter relays `Content-Encoding` downstream over bytes aiohttp's client already decompressed, and collapses duplicate header names — and the wave put that helper at all three response boundaries (`chat.py:172` new at `T015`; `messages.py:291`, `passthrough.py:41` since `T013`). The finding shipped with **no `pkg:` row behind it**: `grep` over this ledger finds no row naming `auto_decompress`, decompression, `Content-Encoding` or `Accept-Encoding` (the aiohttp rows are JSON decoding 51/63/64/83/84, connector and session lifecycle 37/38/50/95–128/174–178, `Host`/`Content-Length` defaults 234–236, Content-Type exposure 154–156, header spelling and parser backends 237/238/324–329, timeouts 57/141/330/331, and 379's request-direction case merge).

## Claims (6 new rows, all `pkg:aiohttp@3.13.3`)

| # | Claim | Subject | Tasks | Scope | Seams | Check | Bundle | State | Routing | Permanent |
|---|---|---|---|---|---|---|---|---|---|---|
| 384 | aiohttp's `ClientSession.__init__` `auto_decompress` parameter defaults to `True` | pkg:aiohttp@3.13.3 | T013, T015 | shared | S15 | shared, seam:S15 | B290 | HOLD | accepted-consequence | CL:aiohttp/auto-decompress-session-default |
| 385 | A request that passes no `auto_decompress` to a `ClientSession` request method uses the session's own value | pkg:aiohttp@3.13.3 | T013, T015 | shared | S15 | shared, seam:S15 | B290 | HOLD | accepted-consequence | CL:aiohttp/auto-decompress-request-inheritance |
| 386 | When `auto_decompress` is enabled, aiohttp's `ClientResponse.read()` returns the decompressed payload rather than the encoded wire bytes | pkg:aiohttp@3.13.3 | T013, T015 | shared | S15 | shared, seam:S15 | B291 | HOLD | accepted-consequence | CL:aiohttp/auto-decompress-read-decompressed |
| 387 | When `auto_decompress` is enabled, aiohttp's `ClientResponse.headers` still contains the upstream's `Content-Encoding` after the body is decompressed | pkg:aiohttp@3.13.3 | T013, T015 | shared | S15 | shared, seam:S15 | B291 | HOLD | accepted-consequence | CL:aiohttp/auto-decompress-content-encoding-retention |
| 388 | `aiohttp.ClientResponse.headers.items()` yields one pair per occurrence when the upstream repeats a header name | pkg:aiohttp@3.13.3 | T013, T015 | shared | S15 | shared, seam:S15, native | B292 | HOLD | absorbed | CL:aiohttp/repeated-header-items-iteration |
| 389 | When neither the session's nor the request's headers name `Accept-Encoding`, aiohttp's client sends a default one naming the encodings it can decode | pkg:aiohttp@3.13.3 | T013, T015 | shared | S15 | shared, seam:S15, native | B293 | HOLD | accepted-consequence | CL:aiohttp/default-accept-encoding-header |

**One inherited cell write, no new row (§3d).** Row **236** (`aiohttp's web.Response sets Content-Length from its body when it sends the response`, `HOLD`/`none`, `B188`) already asserts the proposition the shared filter's `content-length` drop rests on, and the chat boundary now leans on it too (`chat.py:172` answers with `web.Response(body=…, headers=response_headers(resp.headers))` since `T015`). Its `Tasks` cell goes `T013` → `T013, T015` — State and Routing untouched, `B188` not re-dispatched, no checker spent. That is this run's `i`. Recorded rather than tidied, because it is not this pass's write to make: the row's `Scope` still reads `single`, which is now inconsistent with its own `Tasks` cell. Run 24's precedent for an inherited row is the Tasks append and nothing more, §3d prescribes exactly that cell, and a `HOLD`/`none` row dispatches either way, so the drift is left visible for whichever pass owns row-shape re-derivation.

## Row-shape notes (E5 — one proposition per row)

- **384/385 are the two halves of "the session default governs these calls", and they are separate verdicts.** 384 is a signature fact; 385 is the propagation rule that makes a session-level default reach a request that never mentions it. A checker can find one true and the other false (a per-request parameter with its own independent default), and only the pair together says the proxy's client decompresses. Bundled because 385 decides what 384 governs, and both are one read of `aiohttp/client.py`.
- **386/387 are deliberately split at the seam's hinge: the body and the header set.** One row naming both would get one `HOLDS` covering a decompression answer and a header-mutation answer at once. 387 is the row the dispatch asks to be typed as **library, not repo**: whether `Content-Encoding` survives the pair is a statement about what aiohttp does to a `ClientResponse`, and its Subject says `pkg:` for exactly that reason. The repo's use of the surviving header is the *consumer* end, and it is not claimed here.
- **388 is the library side of the finding's second gap.** The collapse itself is repo code; what makes it a loss is a library that surfaces more than one occurrence. It is not 237/238 (spelling as sent, both `DOES-NOT-HOLD`/`absorbed`, corrected by 324–329) and not 379 (`HOLD`, request direction, names differing only in **case**, before the wire): 388 is response direction and repeated **identical** names.
- **389 is `native`** — the client's own capability to advertise compression — and is the reachability half of the premise: with it false, no upstream the proxy asks would answer encoded and the pairing in 386/387 cannot occur on this tree's default path. Its universe is named ("neither the session's nor the request's headers"), because a checker counting the repo's forwarded headers and a checker counting aiohttp's own defaults would otherwise both be right.
- No row states a line number, a hash or a current count: `client.py:299` and `:601-602` are the Yield's and the backlog's *evidence*, not the propositions. Observed values belong in the checker's `Actual`.

## Run 37 bundles (fresh keys — B1–B289 are history; one agent per bundle)

| Bundle | Rows | Subject | Why tied |
|---|---|---|---|
| B290 | 384, 385 | pkg:aiohttp@3.13.3 | one read of `ClientSession.__init__` and `_request`; 385 decides whether 384's default governs a request at all |
| B291 | 386, 387 | pkg:aiohttp@3.13.3 | one read of the decompression path answers both halves of the body-versus-header-set hinge |
| B292 | 388 | pkg:aiohttp@3.13.3 | solo — a different mechanism (the header container's iteration), with nothing tied to answer alongside it |
| B293 | 389 | pkg:aiohttp@3.13.3 | solo — request-side default generation; no claim's answer depends on it and it depends on none |

**Bundle discipline.** Same subject only (rule 1): all six rows are `pkg:aiohttp@3.13.3`, so no bundle crosses subjects and no checker receives the cross-boundary seam itself. Tied means tied (rule 2): 388 and 389 are *not* folded into B290/B291 for tidiness — different mechanism, different source read. Nothing exceeds four rows (rule 3). No key is reused: B287 still names the settled 379 (`HOLD`/`none`) and B245–B286/B288–B289 are void per §Run 36 scope correction §5, so this run allocates B290–B293 among itself. **4 bundles ⇒ 4 checker agents.**

## Freshness basis and evidence lookups

Read at HEAD `8438158a321b6fa701c67e48aec0c0f3e9cd3c06`, working tree clean, against Checkpoint 168's `c8df363`. Hashes establish only that a recorded subject's scope did **not** move — never whether a proposition still holds.

| Identity | Recorded | Now | Movement |
|---|---|---|---|
| `aiohttp` | 3.13.3 (run 1's `.venv` read; every `pkg:aiohttp` subject cell) | `aiohttp-3.13.3.dist-info` → `Version: 3.13.3` | none |
| `aiohttp/client.py` | `sha256:7ce41fc1c2142f5346015455e030e3ebec22a41cde0fc299a66ce13be2cb2a9e` (registry `CL:aiohttp/connector-loop-binding`, `CL:aiohttp/ssl-parameter-topology`) | identical | none |
| `aiohttp/client_reqrep.py` | `sha256:7844440eb67433c6454edd708c776e47e3fcfec9b8d0aeb980dcfe88c1986ac9` (registry `CL:aiohttp/response-header-exposure`, `CL:aiohttp/structured-suffix-json-content-type-validation`) | identical | none |
| `aiohttp/connector.py` | `sha256:bd3db604db820ed6da744d54abb1c2ef3a4e5821ccc48e27dcfb42cfbcd9664c` | identical | none |
| `aiohttp/web_response.py` | `sha256:3ca73388d5382e67ed5ea295be844cac06ce7950a5a5237e8b39c7b2259ece65` (registry `CL:aiohttp/web-response-explicit-content-type`) | identical | none |
| `click` / `pytest` / `Python` | 8.3.1 / 9.0.2 / 3.13.8 | `click-8.3.1.dist-info`, `pytest-9.0.2.dist-info`, `.venv/bin/python -V` | none |

**Why the restore set is empty (set 2, by trigger rather than by assertion).** The gate lists four recheck reasons; each was tested against every `pkg:` and `lang:` row and none fired:
1. *the unit or lines the verdict cites changed* — the cited units are installed-package files, and the four that carry `pkg:` verdict fingerprints are byte-identical above. Repo bytes are not any `pkg:`/`lang:` row's cited unit, and §2's subject typing says a `pkg:`/`lang:` row never fires on a repo edit. `lang:` rows cite stdlib modules (`asyncio`, `json`, `pathlib`, `hashlib`, `time`) on an unchanged Python 3.13.8.
2. *the package version or runtime changed* — nothing moved: same aiohttp, same click, same pytest, same interpreter, and `pyproject.toml` untouched since before run 1 (last edit `e0ce6c2`).
3. *for a relation, either end or the mechanism changed* — the ledger's only live `rel:` row is 219 (`buster_ripper_legacy#_HOP_BY_HOP → utils#_HOP_BY_HOP`), repo code at both ends and therefore outside this gate's subject authority, whose terminal disposition this pass does not touch.
4. *a new observation contradicts it* — the CR2 wave's new observations are `QB:utils/response-header-relay-fidelity` (a **missing** premise, which is what rows 384–389 supply) and five sibling backlog lines, none of which is an observation that contradicts a `HOLDS` on file. `073-01-yield` §Findings D-1's own evidence line reports the client-side decompression basis as holding, i.e. it is the observation that *created* the gap rather than one that falsified a row. Row **84** (`application/problem+json` accepted by the default `ClientResponse.json()` check) was re-verified first-party at Checkpoint 168 with its fingerprint intact and is untouched here: no cell, no re-dispatch.

**Session-ledger matching (§3d), performed for each of the six.** None is asserted by an existing row; the nearest neighbours are named and separated in the row-shape notes (237/238/324–329 for spelling, 379 for the request-direction case merge, 154/156 for which header value `ClientResponse.headers` exposes, 57 for a per-request override that *replaces* rather than falls back, 234/235/236 for client- and server-supplied lengths). All six are therefore `new`/`unverified` with fresh bundle keys.

**Registry `## Claim Index` consulted for these propositions only** (§3e; no unrelated record swept for staleness): **no `CL:` record names decompression, `auto_decompress`, `Content-Encoding`, `Accept-Encoding` or response duplicate-name occurrence counts.** The index's aiohttp records are `cleanup-hook-error-propagation`, `connector-loop-binding`, `content-type-error-inheritance`, `per-request-timeout-override`, `request-json-decoding`, `response-header-exposure`, `response-json-content-type-validation`, `runner-cleanup-await`, `runner-task-cancellation`, `signal-shutdown`, `sse-json-content-type-validation`, `ssl-parameter-topology`, `ssl-verification-polarity`, `structured-suffix-json-content-type-validation`, `web-response-explicit-content-type`. No match at any freshness level, so `Permanent` stays `—` on all six and every one goes to a checker; a promotable verdict may create the handles. `## Composition Index` was consulted for the exact ordered member set, boundary and dimensions of the seam this run opens and contains no matching record, so no composer is skipped and `S15` carries no `CO:` handle.

## Two directions (§1), as rows not judgments

`T015` moved the chat boundary onto the filter the other two boundaries already used, so the question is not only what the new placement takes for granted about aiohttp (rows 384–389) but what the **peer** boundaries now share with it. Both directions are the same six library facts, which is exactly why they are `shared` against `T013` and `T015` rather than single-task rows: a verdict that lands `DOES-NOT-HOLD` is not local to the chat boundary — `messages.py:291` and `passthrough.py:41` were built on it at `T013` and are accepted. No repo-side consequence is rowed here; the peer filter's conformance remains Yield Phase's and the external review's, per §Run 36 scope correction §6.

## Collected and deliberately NOT rowed — routes recorded, nothing silent (E7)

- **`lang:python` — a dict comprehension over a key that arrives twice keeps the last value.** The mechanism of the collapse half of D-1. Not rowed: it is uncontested runtime semantics that `tests/test_header_parity.py`'s kept/drop census shows the moment anyone feeds it a repeated name, so §"write only when being wrong would matter" fails it. Route: Forward read-time, and the header-lane design pass's own duplicate-name ruling (`roadmap.md` §RM-001's owner is the nearest).
- **aiohttp's transparent decompression also applies to the payload *stream* reader, so `resp.content.iter_any()` hands over decoded chunks.** True premise of the finding's shape, but no accepted boundary leans on it: `_relay_chat_stream` and `_stream_single_request` build their own header literals and never relay `Content-Encoding`, so the pairing exists only where `.read()` is used. Rowing it now would spend a checker on a fact nothing depends on. Route: re-extract if and when the header-lane pass routes the streaming boundaries' headers through the shared filter — that move makes it load-bearing.
- **Whether the deployed upstreams (SGLang/vLLM, or a gateway in front of them) actually answer with `Content-Encoding`.** An external-deployment observation, not a library or runtime fact, and not answerable blind from any source root; `073-01-yield` already fixes its reach as *latent* — no witness, no declared compressing upstream on this tree. Route: `QB:utils/response-header-relay-fidelity`'s own recorded trigger (the next decision about relayed names, or a field report from behind a compressing gateway). Row 389 is the library-side half: what the proxy asks for, which is decidable; what a given upstream then does with it, which is not.
- **The repo-side conformance questions the finding raises** (whether the filter *should* drop `content-encoding`, or whether egress should force `identity`, and which of the three boundaries either change touches). Those are design decisions about repo code, and the gate has no authority to open them; they belong to the header-lane design pass with rows 384–389 as its library evidence.

## Run 37 seam delta (recorded in [seams.md](seams.md) §Run-37)

| Seam | Members | State | Movement |
|---|---|---|---|
| S15 | 384, 385, 386, 387, 388, 389 | new | fresh seam at the library endpoint — the response's payload bytes and header set as one `ClientResponse` hands both to the caller |
| S1–S14 | — | out-of-scope | untouched: no member cell moved, no recomposition authorized or performed |

No claim was checked, no composition performed, no verdict written or copied, no invariant created, nothing promoted, and no existing row's State or Routing was moved. The single inherited write is row 236's `Tasks` cell.

# Run 40 — third-party premise extraction for the CR3 wave: `T018` + `T019` + `T020` (2026-10-06)

Dispatch: **scoped adhoc extraction, CR3 remediation wave, third-party subjects only**, wave anchor `T018`, members `T019`/`T020` covered in this one pass. Blueprints and Verification profiles read from `handoff-context.md` §Planned Task Context → `T018`/`T019`/`T020`; task rows read from `plan-summary.md` §Task Queue. Ledger entered at `claims.md` `sha256:c31387380dcc3a85c14f333432550b11cb802d3767919bf91beac8f47c2c823e`, `seams.md` `sha256:e014a6a4fc20797c1ea91fdfbcff2cbaa8f5c27de014c418645f132d4c1b9b2c` — recorded before the first write, re-read after it.

**Scope honored, as the dispatch rules it.** No `repo:` row is created, rechecked or recomposed. `S1`–`S14` are terminal (`out-of-scope`/`registry-safe`) and were not opened; `S15` is live and no cell of it moved. Every row below has subject `pkg:aiohttp@3.13.3`, `pkg:pytest-asyncio@1.3.0` or `lang:python`; repo units appear only in seam `Subjects`/boundary columns as the consumer end of a library boundary, per run 37's `S15` precedent. Extraction only: nothing here was checked or composed, no seam judged safe, no verdict read as an instruction, nothing promoted (E1/E8). No project code was run (E2) — every command was a read of `pyproject.toml`, installed aiohttp source, `invariants/index.md`, this ledger, and the three tasks' sites.

## The four narrow sets, and what each yielded

| Set | Result |
|---|---|
| 1 — tasks not yet started whose plan text changed since extraction | **empty.** `T015`/`T016`/`T017` are accepted; no open task's plan text was reworded between Checkpoint 168 and this pass. |
| 2 — checked claims with changed dependencies/domain or contradictory observations | **empty**, trigger by trigger, in §"Freshness basis". The CR3 findings are *missing premises* (a cap nobody rowed, a deadline span nobody rowed, an exception class nobody rowed), which is set 3's business, not a contradiction of a `HOLDS` on file. |
| 3 — tasks that did not exist at the last extraction (`T018`, `T019`, `T020`) | **22 new rows, 390–411**, plus **12 inherited rows** whose existing propositions these three tasks now also stand on. All three wave members are third-party-bearing: `T018` is a library-configuration task, `T019` is a decode-classification task across two boundaries, `T020` is a test-double/runner-semantics task. |
| 4 — seams whose member, transformation, consumer, fallback, precedence or dimension moved | `S1`–`S14` terminal, unopened; `S15` unmoved. Four **fresh** seams, `S16`–`S19`, each recorded at the library endpoint, because no terminal seam has any of these boundary values. |

## Claims (22 new rows — 12 `pkg:aiohttp@3.13.3`, 1 `pkg:pytest-asyncio@1.3.0`, 9 `lang:python`)

| # | Claim | Subject | Tasks | Scope | Seams | Check | Bundle | State | Routing | Permanent |
|---|---|---|---|---|---|---|---|---|---|---|
| 390 | `aiohttp.web.Application.__init__` accepts a `client_max_size` keyword argument | pkg:aiohttp@3.13.3 | T018 | single | S16 | seam:S16, native | B300 | HOLD | none | CL:aiohttp/app-client-max-size-keyword |
| 391 | aiohttp's `web.Request` enforces its body-size limit from the value configured on the application serving the request | pkg:aiohttp@3.13.3 | T018 | single | S16 | seam:S16 | B300 | UNCLEAR | accepted-consequence | — |
| 392 | An aiohttp `web.Application` constructed without `client_max_size` limits a request body to 1024**2 bytes | pkg:aiohttp@3.13.3 | T018 | single | S16 | seam:S16 | B300 | HOLD | none | CL:aiohttp/client-max-size-default |
| 393 | aiohttp skips the request-body size check whenever the configured `client_max_size` is falsy, so `0` imposes no limit on any body read | pkg:aiohttp@3.13.3 | T018 | single | S16 | seam:S16, native | B301 | HOLD | none | CL:aiohttp/client-max-size-falsy-skip |
| 394 | aiohttp raises `web.HTTPRequestEntityTooLarge` while reading a request body that exceeds the application's configured `client_max_size` | pkg:aiohttp@3.13.3 | T018 | single | S16 | seam:S16 | B301 | HOLD | none | CL:aiohttp/client-max-size-enforcement |
| 395 | `aiohttp.web.HTTPRequestEntityTooLarge` carries the HTTP status code 413 | pkg:aiohttp@3.13.3 | T018 | single | S16 | seam:S16 | B301 | HOLD | none | CL:aiohttp/entity-too-large-status-code |
| 396 | `aiohttp.ClientTimeout(total=None)` leaves the cumulative deadline unset, so no total-time timeout fires for a request built with it | pkg:aiohttp@3.13.3 | T018 | single | S17 | seam:S17, native | B302 | HOLD | none | CL:aiohttp/client-timeout-unset-total |
| 397 | aiohttp's cumulative `ClientTimeout.total` deadline covers consumption of the response payload, not only the wait for response headers | pkg:aiohttp@3.13.3 | T018 | single | S17 | seam:S17 | B302 | HOLD | none | CL:aiohttp/client-timeout-payload-span |
| 398 | aiohttp's `ClientResponse.text()` decodes the payload with the charset named in the response's Content-Type header | pkg:aiohttp@3.13.3 | T019 | single | S18 | seam:S18 | B303 | HOLD | none | CL:aiohttp/client-response-text-charset-source |
| 399 | aiohttp's `ClientResponse.text()` raises `LookupError` when the response's Content-Type names a charset whose codec Python does not have | pkg:aiohttp@3.13.3 | T019 | single | S18 | seam:S18 | B303 | DOES-NOT-HOLD | absorbed | CL:aiohttp/client-response-text-unknown-charset-error |
| 400 | aiohttp's `ClientResponse.read()` hands over payload bytes without applying the response's charset | pkg:aiohttp@3.13.3 | T019 | single | S18 | seam:S18 | B303 | HOLD | none | CL:aiohttp/client-response-read-charset-application |
| 401 | aiohttp's `web.Request.json()` raises `LookupError`, not `UnicodeDecodeError`, when the request's Content-Type names a charset whose codec Python does not have | pkg:aiohttp@3.13.3 | T019 | single | S18 | seam:S18 | B304 | HOLD | none | CL:aiohttp/request-json-unknown-charset-error |
| 402 | Python's `bytes.decode()` uses strict error handling when its `errors` argument is omitted, so invalid bytes raise `UnicodeDecodeError` | lang:python | T019 | single | S18 | seam:S18 | B305 | HOLD | none | — |
| 403 | `bytes.decode(errors="replace")` substitutes a replacement character for bytes invalid under the target codec instead of raising | lang:python | T019 | single | S18 | seam:S18 | B305 | HOLD | none | — |
| 404 | `bytes.decode(errors="replace")` leaves bytes valid under the target codec unchanged in the decoded result | lang:python | T019 | single | S18 | seam:S18 | B305 | UNCLEAR | accepted-consequence | — |
| 405 | Python raises `LookupError` when a codec name is looked up and no such codec exists | lang:python | T019 | single | S18 | seam:S18 | B306 | HOLD | none | CL:python/codecs-lookup-unknown-name |
| 406 | `LookupError` is not a subclass of `UnicodeDecodeError` | lang:python | T019 | single | S18 | seam:S18 | B306 | HOLD | none | CL:python/lookup-error-inheritance |
| 407 | `UnicodeDecodeError` is not a subclass of `LookupError` | lang:python | T019 | single | S18 | seam:S18 | B306 | HOLD | none | CL:python/unicode-decode-error-inheritance |
| 408 | `unittest.mock.AsyncMock` supplies awaitable `__aenter__` and `__aexit__`, so an `AsyncMock` can stand in for an object consumed with `async with` | lang:python | T020 | single | S19 | seam:S19, native | B307 | HOLD | none | CL:python/asyncmock-async-context-manager |
| 409 | A plain `unittest.mock.MagicMock` supplies no awaitable `__aenter__`, so it cannot stand in for an object consumed with `async with` without explicit configuration | lang:python | T020 | single | S19 | seam:S19, absence, native | B307 | DOES-NOT-HOLD | absorbed | CL:python/magicmock-async-context-manager |
| 410 | `asyncio.sleep(0)` yields control to the event loop, letting another ready task run | lang:python | T020 | single | S19 | seam:S19 | B308 | HOLD | none | CL:python/asyncio-sleep-yields-to-loop |
| 411 | Under pytest-asyncio's `asyncio_mode = "auto"`, an `async def` test function is collected and run on an event loop without an explicit `asyncio` marker | pkg:pytest-asyncio@1.3.0 | T020 | single | S19 | seam:S19, native | B309 | HOLD | none | CL:pytest-asyncio/auto-mode-async-test-collection |

No row states a line number, a hash or a current repo count: `server.py:119`, `web_request.py:675`, `web_app.py:128` and `messages.py:276` are the blueprint's and the review's *evidence*, not the propositions (§2). The one library constant a row does name (`392`'s 1024**2) is the proposition itself — the size of the cap the plan sets out to disable — and its Subject is `pkg:`, so its freshness keys on the aiohttp identity, not on a repo hash.

**Twelve inherited cell writes, no new rows (§3d).** Each is the Tasks append and the seam-membership cell, State/Routing/Bundle/Permanent untouched: **57** (`+ T018`, `+ S17`) — `T018`'s whole `timeout=` mechanism is the proposition run 3 checked; **330**/**331** (`+ T018`, `+ S17`) — what class a firing deadline raises, which is what the corrected `_relay_after_commit` docstring must not misstate about the hierarchy; **230** (`+ T018`, `+ S16`) — the mechanism by which a raise inside a handler becomes the client's error response; **51** (`+ T019`, `+ S18`) — the request-side charset decode that precedes the JSON parse; **68–72** (`+ T019`, `+ S18`) — the checked ancestry on the `UnicodeDecodeError`/`JSONDecodeError` side of the new catch lists; **137** (`+ T019`, `+ S18`) — the `HTTPBadRequest` → 400 transport `_read_body`'s existing shape depends on; **126** (`+ T020`, `+ S19`) — cancelling and awaiting the never-returning coroutine the rewritten witness drives. All twelve are `HOLD`/`none` with their verdict-recorded source fingerprints intact (§"Freshness basis"), so none is re-dispatched and none needs a bundle for its new membership (§3c). Their `Scope` cells still read `single` where the Tasks cell now names two or three tasks — the same drift run 37 left visible on row 236 rather than re-deriving row shape on a settled verdict, and it is not this pass's write to make.

## Row-shape notes (E5 — one proposition per row)

- **390/391/392 are the three separable halves of "the app carries the cap"**: that the knob exists on the constructor, that the value reaches each request, and what it is when untouched. A checker can find the kwarg accepted but ignored, or honored but defaulted elsewhere; only the pair 390+391 says the single `create_app()` site governs all four route classes, and 392 is what makes the defect live today rather than hypothetical.
- **393 is the fix's whole mechanism and is written as the property, not the line.** `web_request.py:675` is where the blueprint saw it; the proposition the plan needs is that *falsy means unlimited on every read path* — the chunked path and the length-advertised path are one claim's universe precisely because the four route classes take both (`/v1/messages` bodies arrive with and without `Content-Length` depending on the client).
- **396 and 397 are the two halves of the stream-deadline premise and they fail differently.** 396 says `total=None` imposes nothing; 397 says the budget that *does* exist reaches past the response head into the payload. With 397 false, the counterexample's silent truncation is not a cumulative-total effect at all and `T018`'s reasoning would need re-deriving; with 396 false, the chosen spelling does not express "no deadline". 57 (inherited) is the third leg — that a request-local value replaces rather than merges — and is why this seam can compose wrongly even with every leg true.
- **398/399/400 are split at `ClientResponse`'s two consumption paths**, because `T019` step 2(b) wraps one and falls back to the other: what charset `text()` applies, what class it raises on a name Python cannot resolve, and what `read()` does *not* do. 400 is not row 386 (`read()` returns the decompressed payload rather than the encoded wire bytes — the decompression question, `auto_decompress`-conditional); the decoding question and the decompression question are separate verdicts on the same method.
- **401 is the client-request direction and is not row 51.** 51 is bytes invalid under a *known* charset → `UnicodeDecodeError`; 401 is an *unknown* charset name → `LookupError`. Row 51's own verdict text leaves the unknown-codec case open, so this is a new proposition rather than a re-reading, which is why it gets a row and 51 only gains the task.
- **406/407 follow the ledger's own precedent for non-inheritance claims** (71/72 split the `UnicodeDecodeError`/`JSONDecodeError` pair directionally): "no subclass relationship" has two directions and a checker can settle one from a hierarchy diagram and the other only by reading `LookupError`'s own bases. 405 is the separate fact that a bad codec name raises at all.
- **408/409 are the two directions of the test double's contract**, and 409 is deliberately an `absence`+`native` row: it is the capability the rewrite must *not* assume it gets for free from the sibling idiom already in the file (`mocker.Mock(spec=aiohttp.ClientSession)`), and the whole defect class of `CR3-F-05` is a witness that passes while never reaching the production branch. 411 names its mode in the proposition (`asyncio_mode = "auto"`) rather than asserting this repo's setting, so the universe of discourse is the library's option semantics — the same shape as row 42's `norecursedirs` claim — and a checker never has to read the session to know which tree was meant.

## Run 40 bundles (fresh keys — B1–B293 are history; one agent per bundle)

| Bundle | Rows | Subject | Why tied |
|---|---|---|---|
| B300 | 390, 391, 392 | pkg:aiohttp@3.13.3 | one read of `web_app.py` and `web_request.py`'s config plumbing answers the knob, its propagation and its default |
| B301 | 393, 394, 395 | pkg:aiohttp@3.13.3 | one read of the body-read paths and `web_exceptions.py`: the skip condition, the raise, the class's status |
| B302 | 396, 397 | pkg:aiohttp@3.13.3 | one read of where `ClientTimeout.total` is armed and what the armed window spans |
| B303 | 398, 399, 400 | pkg:aiohttp@3.13.3 | one read of `ClientResponse`'s `text()`/`read()` pair answers both consumption paths |
| B304 | 401 | pkg:aiohttp@3.13.3 | solo — the request-side codec arm, a different file and a different boundary than B303 |
| B305 | 402, 403, 404 | lang:python | one read of the codec error-handling contract settles strict default, replace-not-raise, and replace's effect on valid bytes |
| B306 | 405, 406, 407 | lang:python | one read of `codecs.lookup` and the exception hierarchy |
| B307 | 408, 409 | lang:python | one read of `unittest.mock`'s async-magic-method configuration answers both directions of the capability |
| B308 | 410 | lang:python | solo — a scheduling fact with nothing tied to answer alongside it |
| B309 | 411 | pkg:pytest-asyncio@1.3.0 | solo — a different package entirely (rule 1 forbids folding it into B307) |

**Bundle discipline.** Same subject only (rule 1): no bundle crosses `pkg:` and `lang:` or two packages. Tied means tied (rule 2): 397 is not folded into B301 and 410 is not folded into B307 for tidiness. Nothing exceeds four rows (rule 3). No key reused (the never-reuse rule): B290–B293 still name run 37's settled rows and B1–B289 are history, so this run allocates B300–B309 among itself. **9 bundles ⇒ 9 checker agents**, all nine dispatchable as `new`/`unverified`.

## Freshness basis and evidence lookups

Both lookups performed before any row was written, as the dispatch requires.

| Identity | Recorded | Now | Movement |
|---|---|---|---|
| `aiohttp` | 3.13.3 (every `pkg:aiohttp` Subject cell) | `aiohttp/__init__.py` → `__version__ = "3.13.3"`, `aiohttp-3.13.3.dist-info` | none |
| `aiohttp/web_request.py` | `sha256:2aaacea7…` (row 51's verdict fingerprint, registry `CL:aiohttp/request-json-decoding`) | identical | none |
| `aiohttp/web_protocol.py` | `sha256:eacf5d33…` (row 230's verdict fingerprint) | identical | none |
| `aiohttp/client.py` | `sha256:7ce41fc1…` (registry `CL:aiohttp/connector-loop-binding`, `ssl-parameter-topology`, `auto-decompress-*`) | identical | none |
| `aiohttp/client_reqrep.py` | `sha256:7844440e…` (registry `CL:aiohttp/response-header-exposure`, `sse-…`) | identical | none |
| `aiohttp/helpers.py` | `sha256:435df4ec…` (rows 51, 331's verdict fingerprint) | identical | none |
| `pytest-asyncio` / `Python` | — / 3.13.8 | `pytest_asyncio-1.3.0.dist-info`; `.venv` interpreter 3.13.8 (run 1's recorded identity) | none — 411 is the first `pkg:pytest-asyncio` subject in this ledger |

1. *the unit or lines the verdict cites changed* — the cited units for all twelve inherited rows are installed-package files, byte-identical above. Repo bytes are not a `pkg:`/`lang:` row's cited unit and §2's subject typing says such a row never fires on a repo edit; `T018`/`T019`'s forthcoming edits to `server.py`/`messages.py`/`count_tokens.py`/`chat.py` therefore fire nothing here.
2. *the package version or runtime changed* — nothing moved. One disclosure, kept visible rather than corrected: row **57**'s verdict records its source as `aiohttp v3.12.15 upstream source (installed version unavailable)` while its Subject cell has always named `pkg:aiohttp@3.13.3`. That is a limitation noted in the original check, re-affirmed by the registry publish and by runs 36/37, and the named subject has not changed since; a listed trigger is a change to the named version, not a shortfall in the source the checker could reach. So 57 inherits, and the orchestrator has this note if it weighs otherwise.
3. *for a relation, either end or the mechanism changed* — the ledger's only live `rel:` row (219) is repo code at both ends, outside this gate's subject authority; untouched.
4. *a new observation contradicts it* — none. `CR3-F-01`/`-F-03`/`-F-04` are the review observing that a library behavior was never assumed *into* a row; the review's reproduced 413 confirms rather than contradicts.

**Session-ledger matching (§3d), performed for each of the 22.** No existing row asserts any of them: `grep` over this ledger finds no row naming `client_max_size`, `HTTPRequestEntityTooLarge`, a 413, a body-size cap, `LookupError`, a codec name, `errors=`, `text()`, `AsyncMock`/`MagicMock`, `__aenter__`, `sleep(0)`, or pytest-asyncio's collection mode. The nearest neighbours are named and separated above (51 for the known-charset request decode, 386 for `read()`'s decompression half, 63/83/84 for the response JSON *content-type* check, 71/72 for a different pair's non-inheritance, 57/330/331 for what a timeout does and raises rather than what its span covers, 42 for another library's option semantics). All 22 are therefore `new`/`unverified` with fresh bundle keys.

**Registry `## Claim Index` consulted for these propositions only** (§3e; 176+ rows read, no unrelated record swept for staleness): **no `CL:` record answers any of the 22.** The index's aiohttp records are the fifteen named in run 37's lookup plus the `auto-decompress-*`/`repeated-header-items-iteration`/`default-accept-encoding-header` set published from run 37 — none names `client_max_size`, a body cap, 413, `total=None`, the payload span of a deadline, an unknown codec, `LookupError`, `errors="replace"` or `bytes.decode` strictness. Its `CL:python/*` records are the cancellation/`json`/`pathlib`/`utf-8`/`string` set — none covers `LookupError`'s bases, `codecs.lookup`, `AsyncMock`, `MagicMock` or `asyncio.sleep`. Its pytest record is `CL:pytest/norecursedirs-semantics`; there is no pytest-asyncio record at all. The grep hits for "413"/"AsyncMock"/"aenter"/"strict" inside `invariants/buster_ripper/assumption-claims.md` are hash substrings and a verdict's `Actual` prose about repo test files, not `Statement` matches. So `Permanent` stays `—` on all 22 and every one goes to a checker; a promotable verdict may create the handles. The twelve inherited rows already carry their handles where published (57, 70, 126, 51) or keep `—` where the original verdict was unpromotable (68, 69, 71, 72, 137, 230, 330, 331) — nothing released, nothing invented.

**`## Composition Index` consulted for the four seams this run opens** (§3c): the index holds five records — `aiohttp-runner-to-on_cleanup/lifecycle-teardown`, `cli-flag-to-outbound-call/tls-posture-crossing`, `cli-option-to-dump-dispatch/format-value-crossing`, `client-response-to-relay-consumer/payload-and-header-pairing`, `serve-script-env-to-argv/env-var-flag-crossing`. None matches `S16`–`S19`'s member sets, boundary values or dimensions, so no composer is skipped and none of the four carries a `CO:` handle.

## Two directions (§1), as rows not judgments

`T018` edits the one `web.Application` construction (`create_app`) and the one streaming `session.post`. Both directions are live: the app-wide `client_max_size` change reaches **every** route class's body read, including `chat.py`/`count_tokens.py`/`passthrough.py`, which the plan mentions only as "must keep" — that reach is rows 391 and 393 as `S16`'s hinge, not a repo claim this gate may row. And `T019` reaches into `_stream_single_request`, the function `T018` edits two lines away: 399/400 (`S18`) and 396/397 (`S17`) meet at one call site, so `T019`'s decode fallback must be read against the deadline `T018` sets, and `S17`'s post-commit dimension is where the wave's two members can compose into an answer neither task intended. The peer direction for `T020` is `tests/test_compaction.py`'s other poll tests — the timeout/first-branch arms the blueprint puts under "must not touch" share 408/409's double contract with the two witnesses being rewritten, so a verdict on 409 lands on files the wave declares stable.

## Collected and deliberately NOT rowed — routes recorded, nothing silent (E7)

- **`pkg:aiohttp` — `client_max_size` also bounds a chunked body with no `Content-Length` header.** Subsumed into 393's universe ("any body read"); rowing the two paths separately would put one mechanism in two rows and split a bundle for nothing.
- **`lang:python` — a bare coroutine cannot be used as an async context manager (`TypeError`).** The consequence of 408/409, answerable only after them; the composition pass reads it across the pair.
- **`lang:python` — a function reads a module global at call time, so patching `compaction._kv_cache_usage` or `config.KV_POLL_INTERVAL` is visible to the running coroutine.** Uncontested runtime semantics whose failure makes the rewritten witness assert wrong immediately, not silently (§"write only when being wrong would matter"). Route: Forward read-time; the repo-side call-time reads already live as historical rows 34/298/304, which this gate does not reopen.
- **`pkg:pytest-cov` / the coverage gate — `--cov-report=term-missing` shows or hides line 49, and the suite's 455+/≥95.75% figures.** Records and tools settling `T018`/`T019`/`T020`'s Done-when: an arbiter of state, resolved by running (E4). Route: the wave's single Yield and the task blueprints' own gate lines.
- **The mutation replay (`compaction.py:49-52` operand swap → both old witnesses stay green).** Not a third-party proposition and not repo *state* either — it is an experiment on the tree, which is the Yield replay the blueprint already prescribes. Route: `T020`'s own recorded replay step.
- **vLLM's `/metrics` really publishing `vllm:kv_cache_usage_perc` as the line's trailing value.** An external-deployment observation, not answerable blind from a source root, and `T020` fabricates its own body so no accepted boundary newly leans on it — the same judgment run 37 recorded for the compressing-gateway question. Route: the observability lane, and `CR3-F-05`'s witness realism is the repo regex's business.
- **The streaming upstream outliving 600 s in production.** Deployment reach, not library semantics; the counterexample is a slow-stream witness Forward builds. Route: `T018`'s Done-when, and `S17`'s post-commit dimension is where it lands.
- **`buster_ripper`'s own repo-side premises** — `normalize(body, session_id="")` staying image-inert at `normalize.py:202-210`, the per-request billing nonce the documented client sends, `_KV_METRIC_RE`'s grammar, which arms already catch what, `resp.status` handling, the `_HOP_BY_HOP` and dump-path must-not-touch sets, and `handle_messages`'s `C:11` / `handle_count_tokens`'s `L:17 N:1 C:3` measures. All repo code or repo state; outside this gate's subject authority for this run (E4 and the dispatch's scope limit). Route: invariants `EI:buster_ripper.session/session_id/derived_after_normalize` (the blueprint's own binding), Forward/Yield, and the QA gates.

## Run 40 seam delta (recorded in [seams.md](seams.md) §Run-40)

| Seam | Members | State | Movement |
|---|---|---|---|
| S16 | 390–395, 230 | new | the application's configured body-size limit crossing one construction site into every route class's body read |
| S17 | 57, 396, 397, 330, 331 | new | the deadline for one upstream call crossing the session default into a request-local value and an already-committed stream |
| S18 | 51, 68–72, 137, 398–407 | new | a peer-supplied Content-Type charset crossing aiohttp's decode step into the exception class each parse arm catches |
| S19 | 126, 408, 409, 410, 411 | new | the `/metrics` response a test double hands across `async with` into the production poll loop |
| S1–S14 | — | out-of-scope | untouched: no member cell moved, no recomposition authorized or performed |
| S15 | — | registry-counterexample | untouched: no row added or removed, no dimension moved |

No claim was checked, no composition performed, no verdict written or copied, no invariant created, nothing promoted, and no existing row's State or Routing was moved. The twelve inherited writes are `Tasks` and seam-membership cells only.


---

# Run 40 atomic promotion pass (2026-10-06) — `claims.md` cells + registry `CL:` publication

Dispatch: mode `atomic`, `assumption-promote`. Inputs read: this ledger, `seams.md` (row-scope only), the 22 verdict artifacts `verdicts/claim-390.md`–`claim-411.md`, and the registry's `Claim:` records plus the generated `## Claim Index`. No plan, handoff, chronicle, QA plan, iteration report, or production source was opened; no `.git/` path was read; nothing was checked, nothing was composed, no invariant/entity/scenario/override record was created or touched, and no behavior was approved. Descriptive records say what sources established, not what is intended.

**In-scope set: 22 rows (390–411).** All were `new`/`unverified` with `Check != —`, matching run 40's own §"Claims (22 new rows…)" statement. Rows **48** and **338** are excluded by the mode's own filter — they carry `Routing: unverified` but their State is a verdict token (`DOES-NOT-HOLD`/`UNCLEAR`), not `new`/`recheck`, so they sit outside this invocation; this pass left both cells exactly as the run-36 scope correction recorded leaving them (`B227`/`B240` pointing at retired wording). No in-scope row carries `Check: —`, and none is a `user-claim` row, so `decisions.md` was not opened and no Verification/Evidence/Consequence link was written (nothing to migrate, nothing malformed to report on that axis).

**Verdicts read (checker bookkeeping, not promotion):** `HOLD` ×18 (390, 392–398, 400–403, 405–408, 410–411), `DOES-NOT-HOLD` ×2 (399, 409), `UNCLEAR` ×2 (391, 404), `MALFORMED` ×0. State cells now carry those results; Routing is `none` on every `HOLD` and `pending` on every non-holding verdict, so Assumption Decide routes each one exactly once. Two non-holding rows also carry a `CL:` handle (399, 409) — promotion publishes the evidence, Routing still sends the consequence to Decide; the two are independent writes.

**Promoted: 18 new `CL:` records, 0 updated in place.** All 18 landed in `invariants/buster_ripper/assumption-claims.md` (the registry's existing descriptive-evidence file — no record was moved), and `rebuild_index.py` was run followed by `--check`: `CHECK OK: 12 invariants/overrides, 12 entities, 199 claims, 5 compositions, index.md current` (181 → 199 claims; no duplicate identity reported). Handles: 390 `CL:aiohttp/app-client-max-size-keyword` · 392 `CL:aiohttp/client-max-size-default` · 393 `CL:aiohttp/client-max-size-falsy-skip` · 394 `CL:aiohttp/client-max-size-enforcement` · 395 `CL:aiohttp/entity-too-large-status-code` · 396 `CL:aiohttp/client-timeout-unset-total` · 397 `CL:aiohttp/client-timeout-payload-span` · 398 `CL:aiohttp/client-response-text-charset-source` · 399 `CL:aiohttp/client-response-text-unknown-charset-error` · 400 `CL:aiohttp/client-response-read-charset-application` · 401 `CL:aiohttp/request-json-unknown-charset-error` · 405 `CL:python/codecs-lookup-unknown-name` · 406 `CL:python/lookup-error-inheritance` · 407 `CL:python/unicode-decode-error-inheritance` · 408 `CL:python/asyncmock-async-context-manager` · 409 `CL:python/magicmock-async-context-manager` · 410 `CL:python/asyncio-sleep-yields-to-loop` · 411 `CL:pytest-asyncio/auto-mode-async-test-collection`. No existing `CL:` record matches any of the 18 by subject plus statement, so every handle is a first publication rather than an in-place refresh — the run-40 collector's registry lookup (§"Freshness basis and evidence lookups") reached the same result, and this pass re-checked each proposition against the `## Claim Index` before writing. The nearest neighbours are separate facts under their own handles and were not rewritten: `CL:aiohttp/request-json-decoding` (invalid bytes under a *known* charset) versus the new 401 (unknown charset *name*), `CL:aiohttp/auto-decompress-read-decompressed` (`read()`'s decompression question) versus the new 400 (its charset question), `CL:aiohttp/response-json-content-type-validation` and the `sse-`/`structured-suffix-` records (the content-type gate, not the codec lookup), and `CL:python/json-decode-error-inheritance` / `CL:python/cancelled-error-inheritance` (other classes' hierarchy positions) versus the new 406/407. `S16`–`S19` gained no `CO:` handle from this pass — composition promotion is a separate invocation, and no seam row's `State` or `Permanent` cell was touched.

**Not promoted: 4.** Rows **391** and **404** are `UNCLEAR` — never promotable; their verdicts stay valid for this session and route to Decide as `pending`. 391's artifact names its own repair (pin which application "the application serving the request" means — root/serving app reads `HOLDS`, `request.app` under `add_subapp` reads `DOES-NOT-HOLD`), which is a rewording for the collector lane, not a source gap. Rows **402** and **403** returned `VERDICT: HOLDS` with fingerprinted sources, but their line 2 is `PROMOTION: ELIGIBLE — {annotation}` rather than exactly `PROMOTION: ELIGIBLE`, which is the gate as written and the same reading this ledger already carries on rows 129, 226, 227 and 228. **This is a record-format defect, not a verdict defect**: neither proposition is in doubt and neither is re-dispatched. The one-line remedy sits with the verdict lane, not here: re-issue both headers bare, moving the annotation into the body, and the next atomic promotion publishes them (suggested aspects, subject to the value-flip litmus at write time: `CL:python/bytes-decode-strict-default`, `CL:python/bytes-decode-replace-marker`). Nothing was inferred in their absence — no handle, no registry row, no `Permanent` cell for either.

**What this pass did not do.** No claim was checked and no fingerprint recomputed — the checker captured both, and re-reading installed source or the docs here would put the plan's evidence and the promote role in one context. No stale record was swept or refreshed: promotion touched only run 40's artifacts. `## Composition Index` was read for existence only; no composition was promoted, and `S15`'s retained `CO:` record and its `registry-counterexample`/`accepted-consequence` cells are unchanged. The `QB:assumption-registry/*` class of registry-projection items stays with Authority Registrar — this pass publishes eligible verdicts and reports what it could not publish, and routes nothing itself.


---

# Run 40 atomic routing pass (2026-10-06) — the four `pending` rows dispositioned

Dispatch: mode `atomic non-holding routing` (CR3 wave run 40), `assumption-decide`. Inputs read: this ledger's four
`Routing: pending` rows and their four verdict artifacts, `seams.md` rows `S16`–`S19` (member sets and dimensions
only — no composition artifact was opened), the CR3 blueprints and `DC-CR3-F-01-r1`…`-r5` in `../handoff-context.md`,
the `T018`/`T019`/`T020` rows in `../plan-summary.md`, the registry `## Claim Index`/`## Handle Index`/§Source
Sessions, and the production sites the verdicts cite (`routes/messages.py`, `routes/count_tokens.py`, `compaction.py`,
`tests/test_compaction.py`, `server.py`). Repository state before the first routing write:
`git-commit:e204b18e4f82f618e03c808743361637e3b149d2` (resolved through Git from the supplied worktree root; no other
repository state inspected, no `.git/` path read). Ledger hashes before the first cell write: `claims.md`
`sha256:2011b68229e2f12c3e6c6365c87b25e769419b1b55c9823f70b50a585f0bea82`, `seams.md`
`sha256:999d807102eaf32565ae3fff7ad0da12cb249bbada416e6f668e55813b24e425`.

**Routing is now zero-pending: 4 of 4 rows written exactly once, and no State cell moved.**

| Row | State | Routing written | Reach | Cross-reference | Disposition rests on |
|---|---|---|---|---|---|
| 391 | `UNCLEAR` | `accepted-consequence` | **inert wording** | Neutral | `codify-agents` Reach Gate — the two readings decide nothing for this repo (one flat `web.Application`, `server.py:119`; `add_subapp` greps zero under `buster_ripper/` and `tests/`), and rows 390/392–395 carry the app→request propagation the plan actually stands on. Rewording is a collector-lane doc line: `QB:assumption-ledger/row-391-serving-app-wording`. Holds nothing open. |
| 399 | `DOES-NOT-HOLD` | `absorbed` | `reached` (the arm is on the exposed `/v1/messages` stream path; the class accounting is the defect's) | Neutral | `DC-claim-399-r1` — one admissible remediation: amend `T019`'s blueprint (the response side's live class is `UnicodeDecodeError` off the fallback decode; the `charset=bogus` counterexample alternative is falsified), keep the approved catch list. Asymmetry vs 401 (`CL:aiohttp/request-json-unknown-charset-error`, HOLDS) is now explicit in the task. |
| 404 | `UNCLEAR` | `accepted-consequence` | **inert wording** | Neutral | Reach Gate — the pinned reading is the one the sources settle and the one `T019` step 2(a) needs ("same characters as `errors='strict'` for complete sequences"); the rival reading ("same bytes") is false of every `decode()` call, so no consumer can stand on it. Doc line: `QB:assumption-ledger/row-404-replace-preservation-wording`. Holds nothing open. |
| 409 | `DOES-NOT-HOLD` | `absorbed` | n/a — false evidence (Reach Gate's own exclusion) | Neutral | `DC-claim-409-r1` — the witness rewrite was drafted against the falsified premise; `T020`'s done-when now states the double shapes that are forbidden and the value assertions that make the mutation replay red *for the production arithmetic*. |

**Row 399 read against row 401 — the consequence recorded here, not inferred.** `CL:aiohttp/client-response-text-unknown-charset-error`
(`DOES-NOT-HOLD`) and `CL:aiohttp/request-json-unknown-charset-error` (`HOLDS`) are separate handles for separate
subjects' paths: the client-request direction genuinely raises a bare `LookupError`, the upstream-response direction
suppresses it inside `get_encoding()` and decodes with the fallback. `CR3-F-04`'s blueprint assumed symmetry across the
two directions of `S18`; only the request half of that assumption survives. The registry records publish what the
sources established and neither authorizes a behavior nor changes a disposition.

**Record defect on rows 402/403 (and the session's six annotated headers) — routed `malformed` to the verdict lane.**
Not a ledger-row write: rows 402 and 403 keep `HOLD`/`none`, and the propositions are not in doubt. What is malformed
is the machine-readable field, not the claim: `verdicts/claim-402.md:2` and `claim-403.md:2` read
`PROMOTION: ELIGIBLE — {annotation}` where the promoter's eligibility gate reads line 2 as exactly
`PROMOTION: ELIGIBLE` (`assumption-promote` §Eligibility; "It never infers eligibility from prose"). Owning lane: the
**checker** (`assumption-check`, the verdict lane) — it is the only role that writes a verdict file's line 2, and its
own output template is the ambiguity that produced the annotation. Not the **collector** (`forward-collect` writes
`claims.md`/`seams.md` cells and "wrote no verdict file"), and not the **promoter** (its refusal to infer is the gate
working, and manufacturing eligibility from prose is exactly what it is forbidden to do). Remedy: re-issue all six
headers bare (`129`, `226`, `227`, `228`, `402`, `403`), moving each annotation into the body. **This is not a
re-check** — no source re-consultation, no fingerprint recomputation, no blind dispatch: the bytes the verdicts
fingerprint are unchanged and their evidence is sound. Durable row:
`QB:assumption-registry/annotated-promotion-headers`. Consequence until it lands: `S18`'s post-implementation
composition cannot be promoted — a `Composition:` record requires every member claim to resolve to a permanent `CL:`
handle, and rows 402/403 have none.

**What this pass did not do.** No claim checked, no fingerprint recomputed, no verdict file written or edited, no
composition performed or judged `SAFE`, nothing promoted, no `CL:`/`CO:` handle allocated, no `Invariant:`/`Override:`/
`Entity:` byte touched, no `invariants/` file opened for write. Zero `question` dispositions, so
`reserve-decision-record` was not invoked and `FK-011` stays the only open Fork. Zero `breaks-it`: `T018`/`T019`/`T020`
stand, two blueprints amended. No new task: both absorbed rows land on pending tasks the 080 pass created, so
`plan-adhoc-task` was not invoked. No `repo:` row created, rechecked or recomposed; `S1`–`S14` stay terminal and
`S15`'s cells are unchanged — the standing third-party-subjects-only scope rule held. No production or test byte
touched (`git status --porcelain -- buster_ripper tests` empty before and after).


---

# Run 41 — scoped ledger amendment: CR3 wave seam membership (re-rowing 391, 399, 404, 409) (2026-10-06)

Dispatch: **scoped ledger amendment**, mode `wave seam membership`, over exactly four rows — **391** (`UNCLEAR` /
`accepted-consequence` / `B300` / `S16`), **399** (`DOES-NOT-HOLD` / `absorbed` / `B303` / `S18`), **404** (`UNCLEAR` /
`accepted-consequence` / `B305` / `S18`) and **409** (`DOES-NOT-HOLD` / `absorbed` / `B307` / `S19`). Required delta:
re-row each to one settled proposition with fresh bundle keys, and move [seams.md](seams.md) membership for `S16`,
`S18` and `S19` so no live seam carries a row whose verdict cannot answer it. All four sit non-`HOLD` inside seams that
are still `new`, which is what blocks composition over them; the four `HOLD` neighbours in the same bundles are settled
and were not touched.

**Extraction only.** This pass checked nothing, composed nothing, re-verified no seam, called nothing safe, wrote no
verdict file, created no invariant and promoted nothing (E1/E8). It ran no project code and read no session file for
anything about a library (E2/E4).

**Every settling wording below is the checker's own.** Each of the four rows already carries verdict evidence naming
exactly what settles it — `verdicts/claim-391.md`, `claim-399.md`, `claim-404.md`, `claim-409.md` — and the dispatch
requires that wording rather than a re-derivation. Re-deriving here would put the plan and the checking role in one
context, which is what this gate exists to prevent. The four successor rows are therefore transpositions of the
`## What it actually does` and `## Invalidates On` text of their own verdicts, not new readings of aiohttp, CPython or
the codec docs.

**Numbering.** The ledger's sequence ended at run 40's row 411; this pass takes row **41** in both ledgers (run 27's
rule: one handle names one pass across both files) and its rows run **412–415**, with bundle keys from **B310**
(B1–B309 are history).

**Retirement mechanics, and why the four cells are not edited.** All four rows keep their `State`, `Routing`, `Bundle`
and `Permanent` cells exactly as run 40's promotion and routing passes wrote them — `UNCLEAR`/`accepted-consequence`
on 391 and 404, `DOES-NOT-HOLD`/`absorbed` on 399 and 409 — and their supersession is recorded here and in
[seams.md](seams.md) §Run-41. That is the convention this ledger already runs for rows 3, 20 and 93, and it is
structural rather than stylistic here for two reasons: 399's and 409's `Permanent` handles
(`CL:aiohttp/client-response-text-unknown-charset-error`, `CL:python/magicmock-async-context-manager`) publish *the
retired wording's* verdict, so re-aiming either row's text at a handle whose `Statement` still reads the falsified
direction would attach a refuted proposition to the corrected one; and each row's `verdicts/claim-{n}.md` is the blind
record the amendment is built on. A row's `Seams` cell is its extraction-time declaration, not live membership
(run 34's note), so 391/399/404/409 keep naming `S16`/`S18`/`S18`/`S19` there while this file's live lists no longer
carry them. **Both `accepted-consequence` routings and both `absorbed` routings are left exactly as found**, per the
dispatch: this pass re-routes nothing.

**These are authorized second versions, not a checker cycle.** §"Rewrite a row once" allows one material correction
and then hands the row to `assumption-decide`. Both `UNCLEAR` rows were decided, and the decision named the collector
lane as the owner of the remedy — `QB:assumption-ledger/row-391-serving-app-wording` ("Rewording is a collector-lane
doc line") and `QB:assumption-ledger/row-404-replace-preservation-wording` (same shape) — and 391's own verdict says
the repair "is an extraction fix, not a source gap". The input each successor sends to its checker differs materially
from the input that produced the retired verdict (a pinned referent, a pinned property), so `BLOCKED-EVIDENCE-DEFECT`
does not apply: the words are not the same.

## Retirement delta (old row → disposition → live rows that now carry it)

| Retired row | Cells kept | Why it retires rather than reopens | Successor rows |
|---|---|---|---|
| 391 | `UNCLEAR` / `accepted-consequence` / `B300` / `—` | The verdict settled the mechanism outright — `Request` carries `self._client_max_size`, injected by the application that constructs it (`Application._make_request` passes `client_max_size=self._client_max_size`, bound as the server's request factory) — and named the fork as *which* application "the application serving the request" picks: the app whose runner/handler serves the connection and built the request (**HOLDS**), versus `match_info.current_app`, the innermost matched app (**DOES-NOT-HOLD** under `add_subapp`, because routing happens after `_make_request` and never rewrites the attribute). Its `Invalidates On` line 1 is "the repair, and it is an extraction fix, not a source gap". The sub-app sense is established inert on this tree by the routing's own DC-level disposition (one flat `web.Application`, zero `add_subapp`), so the seam needs the constructing-app sense — a different proposition from the ambiguous one, which is why it is a new row rather than a reopen. | 412 (`S16`) |
| 399 | `DOES-NOT-HOLD` / `absorbed` / `B303` / `CL:aiohttp/client-response-text-unknown-charset-error` | What the verdict falsified is one direction: that an unknown response charset reaches the caller as `LookupError`. What it established in the same read is positive and load-bearing for `S18` — `get_encoding()` performs the codec validity check itself inside `contextlib.suppress(LookupError, ValueError)` and control falls through, so `text()` decodes with UTF-8 for `application/json`/`application/rdap` or with `self._resolve_charset(...)`, the session's `fallback_charset_resolver`, and "the observable failure mode on an unknown charset is therefore a silently wrong encoding". The seam's dimension is the exception class each parse arm catches, and it cannot read that off a falsified negative; the corrected proposition is rowed rather than inherited from 399's `Invalidates On`. | 413 (`S18`) |
| 404 | `UNCLEAR` / `accepted-consequence` / `B305` / `—` | Two terms each carried two readings: "unchanged" as same **characters as `errors='strict'`** versus same **bytes** (the verdict calls the second "false of every `decode()` call rather than a property of `'replace'`"), and "valid under the target codec" read **per byte** versus read as **complete, in-context well-formed sequences**. Its `## What Would Settle It` names both pins, and `Invalidates On` line 1 supplies the settling phrasing and states it "reads as HOLDS on the above sources". The character-and-in-context sense is the one `T019` step 2(a) stands on; the error-path half of the checker's suggested sentence is already live row **403** (`HOLD`/`none`), so only the preservation half is rowed here. | 414 (`S18`) |
| 409 | `DOES-NOT-HOLD` / `absorbed` / `B307` / `CL:python/magicmock-async-context-manager` | The falsified claim is the one `T020`'s witness was drafted on. What the verdict established instead is that a plain `MagicMock` folds `_async_method_magics` (`__aenter__`, `__aexit__`, `__anext__`) into `orig_magics`, installs each as a `MagicProxy` on the type, and routes those children through `_get_child_mock`'s `issubclass(_type, MagicMock) and _new_name in _async_method_magics` branch to `AsyncMock` ("Any asynchronous magic becomes an `AsyncMock`"), so `async with` succeeds unconfigured, with `__aexit__` preconfigured `False`. The real failures live on bare `Mock()`/`NonCallableMock()` and on a `spec=` lacking the dunder — named in the verdict, and not what the amended done-when leans on. | 415 (`S19`) |

## Run 41 claims

| # | Claim | Subject | Tasks | Scope | Seams | Check | Bundle | State | Routing | Permanent |
|---|---|---|---|---|---|---|---|---|---|---|
| 412 | aiohttp's `web.Request` enforces its body-size limit from the `client_max_size` configured on the application whose runner serves the connection and built the request | pkg:aiohttp@3.13.3 | T018 | single | S16 | seam:S16 | B310 | HOLD | none | CL:aiohttp/client-max-size-sourcing |
| 413 | aiohttp's `ClientResponse.text()` raises no `LookupError` for an unresolvable Content-Type charset, because `get_encoding()` suppresses it and decodes with UTF-8 or the session's `fallback_charset_resolver` | pkg:aiohttp@3.13.3 | T019 | single | S18 | seam:S18, absence | B311 | HOLD | none | CL:aiohttp/client-response-text-unknown-charset-error |
| 414 | With `errors="replace"`, every complete sequence the target codec decodes validly yields the same characters as `errors="strict"` would | lang:python | T019 | single | S18 | seam:S18 | B312 | HOLD | none | CL:python/errors-replace-valid-decode-parity |
| 415 | A plain, unspecced `unittest.mock.MagicMock` folds `__aenter__`/`__aexit__` into its magic set and routes those children to `AsyncMock`, so `async with` succeeds on it without explicit configuration | lang:python | T020 | single | S19 | seam:S19, native | B313 | HOLD | none | CL:python/magicmock-async-context-manager |

Each row's subject is its retired predecessor's subject, so no bundle crosses `pkg:` and `lang:` and every successor
stays inside this gate's third-party subject authority. No row states a line number, a hash, or a repo count:
`web_request.py:198`, `web_app.py:510`, `client_reqrep.py:689-740` and `mock.py:1033-1035` are the four verdicts'
*evidence*, not propositions (§2).

## Row-shape notes (E5 — what each retired row was actually carrying, and why its successor is one proposition)

**412 names the referent 391 could not.** The verdict's Reading A is reproduced with its own two identifying clauses —
"whose runner serves the connection" and "built the request" — because they are the same application by the
mechanism the verdict traced (`_make_handler` binds that app's `_make_request` as the request factory), not two
conjuncts a verdict could split. What 412 does *not* say is anything about sub-applications: the sub-app sense is the
fork, and a row that named both senses is the compound row 391 was. The proposition is also not 394's (`HOLD`,
published as `CL:aiohttp/client-max-size-enforcement`), which reads "the application's configured `client_max_size`"
— the very phrase whose referent 391 left open. 394 says a too-large body raises; 412 says whose configured value the
raised decision was taken from. A checker can settle one and leave the other unanswered, which is the test for two
rows.

**413 keeps the mechanism and its consequence in one row because one source read settles both.** The verdict's own
opening is a single sentence-pair about a single function: "`An unrecognised charset in `Content-Type` never reaches
the caller as `LookupError`. `get_encoding()` performs the codec validity check itself and swallows the failure`".
Splitting the "no raise" from the "falls through to UTF-8 or the resolver" would give a checker a negative with
nothing to answer it from — and the fallback chain is *what the caller's catch list has to be written against*, which
is `S18`'s boundary value. The `absence` marker records that the proposition's head is negative. What 413 deliberately
excludes is the verdict's second path — an explicit `encoding=` argument reaches `bytes.decode()` unchecked and *can*
raise `LookupError` — which the verdict itself files as a re-scope ("a different claim with a different answer"), not
as a live premise; it is in the not-rowed list with its route.

**414 is the character reading, both pins on the page, and only the half row 403 does not already hold.** The
checker's `Invalidates On` settling sentence has two clauses — "only byte positions that raise a decoding error are
substituted" and "every complete sequence the codec decodes validly yields the same characters as `errors='strict'`"
— and the first is already live row **403** (`HOLD`/`none`/`B305`). §3d forbids a second row for what an existing row
asserts, so 414 is the preservation clause alone, carrying the two pins its `## What Would Settle It` demands:
*characters*, not bytes, and *complete sequences*, which is what makes "valid under the target codec" a property of a
sequence in context rather than of a byte in isolation (`b'\xc3'` alone is a well-formed lead byte and a decoding
error). Row 402 (`HOLD`) keeps the strict-default half. A verdict on 414 can now disagree with the preservation
proposition without silently covering 403's substitution proposition, which is precisely what 404 could not do.

**415 states the positive and pins its universe so a HOLDS cannot be over-read.** The qualifier "plain, unspecced" is
the verdict's own distinguishing word ("The distinguishing word in the claim is 'plain', and for a plain MagicMock the
library does the opposite of what the claim asserts"), and it is load-bearing because the verdict's `Invalidates On`
lines 3 and 4 warn that the corrected reading must *not* be reused for `MagicMock(spec=...)`, `spec_set=True`,
`patch(..., autospec=True)` or bare `Mock()`/`NonCallableMock()`. The two clauses inside the row — folds the dunders
into the set, routes those children to `AsyncMock` — are one mechanism named by the verdict in one passage of
`_get_child_mock`/`_mock_set_magics`, and the consequence ("so `async with` succeeds … without explicit
configuration") is what `T020`'s amended done-when actually leans on; three verdicts from one read of one file is not
possible, so this is one row. It is not 408 (`CL:python/asyncmock-async-context-manager`, `HOLDS`), which is about
`AsyncMock`'s own surface: a checker could confirm 408 while the wiring from a `MagicMock` to an awaitable
`__aenter__` did not exist.

## Run 41 bundles (fresh keys — B1–B309 are history; one agent per bundle)

| Bundle | Rows | Subject | Why tied |
|---|---|---|---|
| B310 | 412 | pkg:aiohttp@3.13.3 | solo — the server-side request-construction path (`web_request.py`/`web_app.py`); no other fresh row shares that read |
| B311 | 413 | pkg:aiohttp@3.13.3 | solo — the client-side response decode (`client_reqrep.py`'s `get_encoding`/`text`), a different file and mechanism from B310 |
| B312 | 414 | lang:python | solo — the codec error-handler contract (`stdtypes`/`codecs`/the unicode howto) |
| B313 | 415 | lang:python | solo — `unittest/mock.py`'s async-magic wiring, a different module and source from B312 |

**Bundle discipline.** Fresh keys per the re-dispatch convention: run 40's B300/B303/B305/B307 each still name two or
three settled rows, so a key that spanned old and new would hand a checker claims it has already answered (rule:
re-bundle among themselves). Same subject only (rule 1) — no bundle crosses `pkg:`/`lang:` and none crosses two
libraries. Tied means tied (rule 2) — 412 and 413 share a *package* but not a source read (the server's request
factory versus the client's response decode), and 414 and 415 share a *runtime* but not a module; nothing here is
padded for tidiness, so all four are solo (rule 4). Nothing exceeds four rows (rule 3). **4 bundles ⇒ 4 checker
agents.**

**Queue consequence, as ledger state rather than a decision.** With 391, 399, 404 and 409 out of live membership, the
four bundles they leave name only settled rows — `B300` → 390/392, `B303` → 398/400, `B305` → 402/403, `B307` → 408,
all `HOLD`/`none` — so none of the four carries anything dispatchable, and the CR3 wave's live dispatch is exactly
B310–B313. Rows 390/392–398/400–403/405–408/410–411 keep every cell they had; nothing already answering was
re-bundled. Whether the two `QB:assumption-ledger/row-*` doc lines close on this landing is the decide route's write,
not this pass's.

## Freshness basis and evidence lookups

Both ledgers hashed before the first write and re-read after it, as recorded in the receipt below. Repository state:
this pass read no project source and ran nothing; the only files opened were this ledger, `seams.md`, the four
`verdicts/claim-{391,399,404,409}.md` artifacts the dispatch names, and the registry index.

**Session-ledger matching (§3d), performed for each of the four.** No existing live row asserts any of them, and the
nearest neighbours are named and separated in the row-shape notes rather than silently re-used: 390/392/394–395 for
the knob, its default and its raise (412 is the referent, none of those is); 398/400 for `text()`'s charset source and
`read()`'s non-application, 401 for the request direction that genuinely raises a bare `LookupError` (413 is the
response direction's silence, the asymmetry run 40's routing recorded); 402/403 for the strict default and the
substitution (414 is what survives of the text); 408 for `AsyncMock`'s own surface (415 is the `MagicMock` wiring).
All four are therefore `new`/`unverified` with fresh bundle keys. No row was marked `recheck`: none of the four
listed triggers applies to a proposition this pass has never sent to a checker, and the only observations that
contradict anything are the four retired verdicts themselves, whose consequence is the re-rowing rather than a second
dispatch of the same words.

**Registry `## Claim Index` consulted for these four propositions only** (§3e; no unrelated record swept for
staleness). Four records are near-neighbours and none matches, so `Permanent` stays `—` on all four rows and every
one goes to a checker — extraction never invents a handle and never re-aims one:

| Candidate record | Why it is not a match for the successor |
|---|---|
| `CL:aiohttp/client-response-text-unknown-charset-error` (`DOES-NOT-HOLD`, `pkg:aiohttp@3.13.3`, `client_reqrep.py` `7844440e…`) | Its `Statement` is retired 399's wording — "raises `LookupError` …" — and its verdict is the *refutation* of that statement. §3e rule 1 requires a record whose Statement asserts the same atomic proposition; a negative of the proposition is not it. The handle stays on retired 399's row, released by nobody. |
| `CL:python/magicmock-async-context-manager` (`DOES-NOT-HOLD`, `lang:python`, `mock.py` `2576119e…`) | Same shape: the Statement is the falsified direction, so it cannot answer 415 without the checker reading a refutation as a proof. |
| `CL:python/asyncmock-async-context-manager` (`HOLDS`) | Different subject unit — `AsyncMock`'s own surface, not a plain `MagicMock`'s wiring to it. 415's verdict can go the other way from this one and does for the spec'd and bare-`Mock()` constructions per 409's own `Invalidates On`. |
| `CL:aiohttp/client-max-size-enforcement`, `CL:aiohttp/app-client-max-size-keyword`, `CL:aiohttp/client-max-size-default` (all `HOLDS`) | The kwarg, the default, and the raise — 412's proposition is *whose configured value the request was built with*, which no `Statement` in the index names. There is no record about `client_max_size` propagation at all. |

No `CL:` record exists for `errors="replace"` preservation under either spelling: rows 402 and 403 returned `HOLDS`
but were **not promoted** because of the annotated-header record defect routed to the verdict lane as
`QB:assumption-registry/annotated-promotion-headers`, so the suggested `CL:python/bytes-decode-*` handles do not yet
exist to match against. That is this pass's disclosure, not a reason to skip a checker.

**`## Composition Index` consulted for the three moved member sets** (§3c, S16/S18/S19's new ordered membership,
boundary values and dimensions): the index holds five records — `aiohttp-runner-to-on_cleanup/lifecycle-teardown`,
`cli-flag-to-outbound-call/tls-posture-crossing`, `cli-option-to-dump-dispatch/format-value-crossing`,
`client-response-to-relay-consumer/payload-and-header-pairing`, `serve-script-env-to-argv/env-var-flag-crossing` — and
none covers a body-size cap propagation, a charset-decode exception class, or a test double's context-manager surface.
So no composer is skipped and no `CO:` handle enters or leaves any seam's `Permanent`.

**Fingerprint continuity, recorded for the orchestrator's dispatch economy only.** The bytes each retired verdict
fingerprinted are unchanged: `web_request.py` `2aaacea7…` / `web_app.py` `94653f68…` (391), `client_reqrep.py`
`7844440e…` (399), the three fetched CPython doc pages (404), `Lib/unittest/mock.py` `2576119e…` with the `v3.14.0`
corroboration `667d9ed8…` (409), and the aiohttp identity is still 3.13.3 on an unchanged Python 3.13.8. This is not a
reuse claim and no verdict was copied into a successor's cells — the checkers re-establish their own sources and
record their own digests.

## Collected and deliberately NOT rowed — routes recorded, nothing silent (E7)

- **The sub-app sense of 391's phrase** (`match_info.current_app`, where the verdict reads `DOES-NOT-HOLD` because
  `_handle` routes *after* `_make_request` and never rewrites `_client_max_size`). Not rowed: the routing's DC-level
  disposition establishes it inert on this tree (a flat single `web.Application`, zero `add_subapp` anywhere under
  `buster_ripper/` or `tests/`), and asserting that tree-structure fact in a row is a `repo:` claim this gate has no
  authority to make (E4, and the dispatch's standing scope rule). Route: `QB:assumption-ledger/row-391-serving-app-wording`,
  whose collector-lane remedy this amendment lands; the forward-looking library case ("a future aiohttp that applies
  the matched sub-application's value") is 391's own `Invalidates On` bullet 2 and lives with the recheck triggers, not
  with a row.
- **A spec'd or bare double's failure under `async with`** — `MagicMock(spec=…)`/`spec_set=True`/
  `patch(..., autospec=True)` pruning the proxies, and `Mock()`/`NonCallableMock()` having no magic support at all
  (`TypeError: 'Mock' object does not support the asynchronous context manager protocol`). Real library facts, named by
  409's verdict, and deliberately not rowed: the plan's consumer of them is `T020`'s amended done-when, which now
  states the forbidden double shapes in the task text (a statement about repo test code, outside this gate's subject
  authority), and 415's "plain, unspecced" qualifier exists precisely so a `HOLDS` there cannot be reused for those
  constructions. Route: `DC-claim-409-r1` and `T020`'s done-when; re-extract if and when a second seam member comes to
  stand on a spec'd double's context-manager surface.
- **`text(encoding="…")` — an explicit encoding argument goes straight to `bytes.decode()` with no validity check, so
  `LookupError` can still escape that way.** 399's verdict's own second path, filed by it as a re-scope rather than a
  premise. No task in this wave passes an explicit `encoding=` to a `ClientResponse` method. Route: Forward read-time,
  and re-extract the moment `T019`'s implementation reaches for that argument.
- **`UnicodeDecodeError` as the response direction's observable failure off the fallback decode.** Carried inside
  413's consequence clause and stated in `T019`'s amended blueprint; rowing it apart would put one mechanism in two
  rows and split a bundle for nothing — the same treatment run 40 gave the chunked-body case under 393.
- **Rows 391/399/404/409's own cells, their routings, and the four handles/queue keys they carry.** This pass retires
  wording and membership; it re-routes nothing. Both `accepted-consequence` routings, both `absorbed` routings, the
  two `CL:` handles on 399/409, and the `B300`/`B303`/`B305`/`B307` keys stay exactly as run 40 wrote them.
- **The repo-side premises of all three tasks** — which arm catches what today in `messages.py`/`count_tokens.py`,
  `_read_body`'s existing shape, `compaction.py:55-56`'s swallow, the sibling poll tests' double choice, the mutation
  replay, and the coverage/`term-missing` figures. All repo code, repo state or an arbiter of state; outside this
  gate's subject authority for this run (E4), with the routes run 40 already recorded.

## Run 41 seam delta (recorded in [seams.md](seams.md) §Run-41; no seam re-verified)

| Seam | Members before → after | Movement |
|---|---|---|
| S16 | 7 → 7 | drops retired **391**; gains **412** in its place; the unset-default-versus-explicit-value dimension clause re-worded to cite the live row |
| S17 | 5 → 5 | **unchanged** — no member retired, no dimension moved |
| S18 | 17 → 17 | drops retired **399** and **404**; gains **413** and **414** in their places; three dimension clauses re-worded to name live rows (the direction asymmetry, the nonexistent-codec arm, the `errors="replace"` preservation axis) |
| S19 | 5 → 5 | drops retired **409**; gains **415** in its place; the awaitable-result-versus-async-context-manager dimension clause re-worded to cite the live row |
| S1–S14 | — | untouched, terminal, third-party-subjects-only scope held; no member list, State, Routing, Permanent or dimension cell of the fourteen moved |
| S15 | — | untouched at `registry-counterexample`/`accepted-consequence`; no row joined or left it |

`State`, `Routing` and `Permanent` are untouched in all four rows of that table, `S16`/`S18`/`S19`'s `new`/`unverified`
cells included (and `S15`'s and the terminal seams' left alone): whether an amended member set composes, and whether
either seam may leave its routing, is the composer's and the decide route's write (E1/E8). Rows 390/392–398/400–403/
405–408/410–411 keep their current cells, and no row that already answers was re-bundled.

No claim was checked, no composition performed, no verdict written or copied, no invariant created, nothing promoted,
and no existing row's State or Routing was moved. The ledger's four new rows are `new`/`unverified` with solo fresh
bundles; the four retired rows are terminal history.


---

# Run 43 — scoped amendment (2026-10-06): the one `malformed` atom `S18` §MC-2 names (row **416**)

Dispatch: **scoped amendment**, mode `one malformed atom`, over exactly one proposition — the class Python's
`bytes.decode()` produces for a codec name that is **registered** but **not a text encoding**. Required delta: one new
row under a new bundle key, and its membership in `S18` only. Retire nothing, re-bundle no existing row, leave `S16`,
`S17`, `S19` and every terminal seam alone, `S1`–`S14` stay terminal, third-party subjects only.

**Why this row exists at all.** `compositions/S18-decode-failure-surface.md` §MC-2 records that the composition cannot
certify `{UnicodeDecodeError, LookupError, json.JSONDecodeError}` as the complete class set at this boundary, only as
the complete **named** set: 402/403 speak only of strict-versus-replace handling of decoding errors *under a usable*
codec, 405 only of names absent from the registry, and nothing speaks of a name that is in the registry and produces no
characters. That record's §Invalidates On 4 names this exact atom as the thing that reopens the composition, and the
run-42 routing pass put it in the collector's lane at
`QB:assumption-ledger/s18-s19-unrowed-class-atoms` item **(A)** — "the only one that can change required behavior",
with the standing prohibition that **`T019` must not pre-empt it by adding a fourth name to a catch list on
speculation**. Row 416 is that atom.

**Extraction only.** This pass checked nothing, composed nothing, re-verified no seam, called nothing safe, wrote no
verdict file, created no invariant and promoted nothing (E1/E8). It ran no project code (E2) and read no session file
for anything about a library (E4); the only files opened were this ledger, `seams.md`, `S18`'s composition artifact and
`quality-backlog.md`'s owning queue entry, to get the atom's own wording right.

**Numbering.** The ledger's sequence ended at run 41's row 415; run 42 was a `seams.md`-only routing pass that wrote no
claim row. This pass takes **run 43** in both ledgers (run 27's rule: one handle names one pass across both files), row
**416**, bundle key **B314** (B1–B313 are history).

## What the row asks, and what it deliberately does not answer

The question the seam needs settled is: **for a codec name that is in Python's codec registry but is not a text
encoding, what does `bytes.decode(name)` do — does it raise, and if so which named exception class?** The answer space
`S18` needs covered is the one §MC-2 leaves open: either the call raises a class the arms already bind, or it raises
something none of them binds (including a successful return of a non-`str` value, which binds nothing at all).

The row therefore asserts **membership in the union the seam already carries**, and nothing more:

- It does **not** name which of the three classes the call produces. The collector's saying so would be E1 — resolving
  a claim in the same context that wrote it — and the composition's whole difficulty is that a *named* set is not a
  *certified* one.
- It **is** falsified by the third branch. If the call instead returns a non-`str`, or raises a class outside the three,
  the verdict is `DOES-NOT-HOLD`, and `assumption-check` §P6 obliges the artifact to say *what it does instead* — which
  is precisely the fourth name `T019` step 2(c) must not guess. A `HOLDS` likewise records the class in its
  `## What it actually does`, so both outcomes reach the recomposition with a name in hand.
- A negative form ("raises no class outside the union") was rejected: it is **vacuously true of a successful return**, so
  the one outcome the seam most needs to catch would have come back `HOLDS`. The positive "raises one of …" form fails
  in exactly that case.

**Why the union is written into the proposition rather than referenced.** `T019` step 2(c) and its class list are plan
text a blind checker cannot open (E3), so the three names are spelled out in the row's own words — the same three
`S18`'s dimension 3 already carries and rows 68–72/406/407 already establish as separate targets. The row stands alone;
no checker needs the plan to grade it.

**Errors-argument scope.** The row is the one-argument call, `bytes.decode(name)`. The two-argument spelling
(`errors="replace"` under a non-text codec name) is a different proposition and is recorded below as collected-and-not
rowed rather than being folded in (E5).

## Run 43 claims

| # | Claim | Subject | Tasks | Scope | Seams | Check | Bundle | State | Routing | Permanent |
|---|---|---|---|---|---|---|---|---|---|---|
| 416 | `bytes.decode(name)` raises one of `LookupError`, `UnicodeDecodeError` or `json.JSONDecodeError` when `name` is a codec registered in Python's codec registry that is not a text encoding (a bytes-to-bytes transform such as `base64`, `hex` or `rot_13`) | lang:python | T019 | single | S18 | seam:S18 | B314 | HOLD | none | CL:python/bytes-decode-non-text-codec-error |

The row states no line number, no hash, no repo count and no observed value (§2): the registry's contents, the
text-encoding flag and the raise site are the checker's `## What it actually does` evidence, not the proposition. Its
`lang:python` subject keeps it inside this gate's third-party-only authority and selects the freshness rule — a
runtime/interpreter identity change, never an unrelated repo edit.

## Run 43 bundles (fresh keys — B1–B313 are history; one agent per bundle)

| Bundle | Rows | Subject | Why tied |
|---|---|---|---|
| B314 | 416 | lang:python | solo — the CPython codec-resolution path for a registered non-text codec (`codecs.lookup`'s text-encoding flag and `bytes.decode`'s use of it). Rule 1 forbids pairing it with the settled `pkg:aiohttp` members of `S18`, and the dispatch forbids re-bundling any existing row, so it takes its own key (rule 4) |

**Bundle discipline.** Fresh key, per the re-dispatch rule that a key spanning old and new work hands a checker claims
it has already answered. Nothing exceeds four rows (rule 3); nothing is padded to look tidy. **1 bundle ⇒ 1 checker
agent.** `Check` is `seam:S18` alone, matching the ledger's own sibling class-atoms 402/403/405/406/407 rather than the
`native` spelling the queue entry suggests for A–D: `native` marks a claim that a mechanism *supplies a capability*
(408/411/415, 396), which is not this atom's shape, and the row is dispatched either way because it is a seam member
(§3b rule ordering is about applicability, not about widening coverage).

**Queue consequence, as ledger state rather than a decision.** This pass discharges item **(A)** of
`QB:assumption-ledger/s18-s19-unrowed-class-atoms` and nothing else: items **(B)** (`aiohttp.ClientError`'s base and its
disjointness from the built-in decode errors — `S18` §MC-1), **(C)** (`unittest.mock`'s default `__eq__` for an
auto-created child — `S19` §Notes N2) and **(D)** (`asyncio.CancelledError` is a `BaseException`, not an `Exception` —
`S19` §Notes N1) are **not rowed here**, because the dispatch scoped this pass to the one `malformed` atom and forbade
touching `S19` at all. The queue key therefore stays open after this landing, with (A) answered by row 416's dispatch.
Whether the key's remaining items move to another lane is the decide route's write, not this pass's.

## Freshness basis and evidence lookups

`claims.md` hashed before the first write and re-read after it
(`988bc3f9a8f096338c9cb3e181093db5838f3280c3b0fb7bd7b7c1b4f2b833ca` → the receipt value below); `seams.md` likewise
(`706cc1d3bdb31fd97aac16a812b67a44dcf997afe43bd6e106e90c46c58334eb` → its receipt value). Repository state unchanged from
what run 42 recorded: `git-commit:5f7f9b9d43a50f571c497c0c52bf18406e77a567`. The interpreter identity the `lang:python`
rows answer against is the one recorded at the head of this ledger (CPython 3.13.8, with the checker lane's own doc
sources at 3.14.x); nothing in this pass re-established it.

**Session-ledger matching (§3d), performed for this proposition.** No live row asserts it, and the four nearest rows are
named and separated rather than silently re-used: **405** is a codec name **absent** from the registry, which is the
opposite arm of dimension 2 and was the composition's own contrast for §MC-2; **402** and **403** are what strict and
`errors="replace"` do to **decoding errors under a usable codec**, a phrase §MC-2 quotes as exactly what they do not
cover; **414** is the `replace`-parity premise whose scope note is the ledger's only *mention* of `base64`/`hex`/`rot_13`
as codecs that "do not produce characters at all" — a mention, not an assertion about the raise, which is why the
composition still could not certify the set. So 416 is `new`/`unverified` with a fresh key. **No existing row was marked
`recheck`**: none of §"Check once" triggers fires on a proposition this pass has never sent to a checker — §MC-2 is an
*absence* of coverage, not a contradiction of an existing verdict, and the four recheck reasons are explicitly not
"the ledger is missing something". Rows 402/403/405/406/407/414 keep every cell run 40–42 left them; nothing already
answering was re-bundled.

**Registry `## Claim Index` consulted for this proposition only** (§3e; no unrelated record swept for staleness). The
index carries eighteen `lang:python` records and none is a match, so `Permanent` stays `—` and the row goes to a
checker — extraction never invents a handle:

| Candidate record | Why it is not a match for 416 |
|---|---|
| `CL:python/codecs-lookup-unknown-name` (`HOLDS`) | Its Statement is a name **not in** the registry; 416's domain is a name that **is** registered and is not a text encoding. §MC-2 reads the two arms as distinct, which is why 405 could not settle the boundary. |
| `CL:python/lookup-error-inheritance`, `CL:python/unicode-decode-error-inheritance`, `CL:python/json-decode-error-inheritance` (all `HOLDS`) | Hierarchy placement of the three names — the relation among them, not what `bytes.decode()` produces for a non-text codec. |
| `CL:python/errors-replace-valid-decode-parity` (`HOLDS`, row 414's handle) | The `replace` parity property under a codec that decodes at all; it does not speak to the non-text-codec arm. |

No record in the index concerns `bytes.decode()` and the codec registry's text-encoding flag under any spelling.

**`## Composition Index` consulted for the moved member set** (§3c — `S18`'s ordered membership, boundary value and
dimensions after 416 joins): the index holds the same five `CO:` records run 41 enumerated
(`aiohttp-runner-to-on_cleanup/lifecycle-teardown`, `cli-flag-to-outbound-call/tls-posture-crossing`,
`cli-option-to-dump-dispatch/format-value-crossing`, `client-response-to-relay-consumer/payload-and-header-pairing`,
`serve-script-env-to-argv/env-var-flag-crossing`) and none covers this boundary, so no composer is skipped, no
`registry-safe`/`registry-counterexample` shortcut is taken, and no `CO:` handle enters or leaves `S18`'s `Permanent`.
`S18`'s `State` and `Routing` cells are untouched by this pass, per the dispatch and per run 42's note that membership
moves by the collector's rule while the composition verdict does not.

## Collected and deliberately NOT rowed — routes recorded, nothing silent (E7)

- **The `errors=` spelling of the same arm** — `bytes.decode(name, errors="replace")` (or any non-default `errors`) for a
  registered non-text codec name. Not rowed: one proposition per row (E5), and the seam's completeness question is
  answered by 416 as written. 414's scope note already records that `errors=` is not error-sensitive for these codecs,
  which is *mention*, not coverage of the raise. Route: Forward read-time at `T019` step 2(c), and re-extract the moment
  the implementation reaches for the argument at a site whose `charset` can name a non-text codec.
- **`bytes.decode()` under a *bytes-to-bytes* codec name returning successfully** — the branch 416 is falsified by. Not
  rowed apart: it is the same proposition's negation, and a second row would split one source read into two verdicts
  that could disagree (P4's hazard from the other side).
- **`aiohttp`'s own handling between `charset` and `bytes.decode`** — whether `web.Request.text()`/`json()` wrap or
  suppress whatever the codec step raises for such a name. Not rowed: that is a `pkg:aiohttp@3.13.3` subject and a
  different source read, and rows 401/413 already carry the two directions of the *unknown-name* case; the non-text-name
  case reaches `bytes.decode` only through the same unvalidated `charset` hand-off those verdicts established. Route:
  the `S18` recomposition reads 416's class against 401/413's transport claims; re-extract if the verdict places the
  outcome outside what 401 describes.
- **`S19`'s unrowed atoms (C) and (D), and `S18` §MC-1's (B)** — the other three items sharing
  `QB:assumption-ledger/s18-s19-unrowed-class-atoms`. Not rowed: the dispatch scopes this pass to the one `malformed`
  atom and instructs that `S19` stay untouched. Recorded loudly, not silently: the queue key remains open.
- **Rows 402/403/405/406/407/414's own cells, their routings and handles**, and every `S1`–`S17`, `S19` seam row. This
  pass retires no wording, re-routes nothing, and moves one member list plus one dimension clause in `S18`.
- **The repo-side premises** — which arm catches what today at `messages.py:28-31`/`chat.py:29-32`/
  `count_tokens.py:18-25`, `T019` step 2(c)'s wording, and any coverage or test-run figure. All repo code, repo state or
  an arbiter of state; outside this gate's subject authority (E4), with the routes run 42 already recorded.

## Run 43 seam delta (recorded in [seams.md](seams.md) §Run-43; no seam re-verified)

| Seam | Members before → after | Movement |
|---|---|---|
| S18 | 17 → 18 | gains **416**; dimension 2's known-codec arm is extended with the registered-but-not-a-text-encoding case, named to the live row. `State`/`Routing`/`Permanent` cells and every existing member unchanged |
| S16, S17, S19 | — | untouched, per the dispatch: no member, dimension, State, Routing or Permanent cell moved |
| S1–S14 | — | untouched, terminal, third-party-subjects-only scope held |
| S15 | — | untouched at `registry-counterexample`/`accepted-consequence`; no row joined or left it |

No claim was checked, no composition performed, no verdict written or copied, no invariant created, nothing promoted, no
existing row's cells or bundle were moved, retired or re-bundled, and no seam's `State` or `Routing` was touched. The
ledger's one new row is `new`/`unverified` under a solo fresh key, and its membership is `S18` only.

---

# Run 44 — scoped amendment (2026-10-06): post-repair invalidation — row **57** fired; seams `S16`–`S19` opened for their recompositions

Dispatch: **scoped amendment**, mode `post-repair invalidation firing`. Required ledger delta: (i) row **57** moves to
`recheck`/`unverified` under a **fresh** bundle key; (ii) seam `State` for **`S16`, `S17`, `S18`, `S19`** is set so the
four unspent post-repair recompositions can run. Check nothing, compose nothing. Create no row for anything current
evidence already answers, retire nothing, re-bundle no row that still answers, leave every terminal row and terminal seam
alone (`S1`–`S15` as they stand; rows `3`/`20`/`93`/`391`/`399`/`404`/`409` stay retired). Third-party subjects only, per
this session's standing scope ruling.

**Why now.** The CR3 remediation wave (`T018` + `T019` + `T020`) landed at Checkpoint 191 and changed production bytes
that all four seams were composed against: `server.py`'s `create_app()` now passes `client_max_size=0`; the streaming
`session.post` in `routes/messages.py` now carries an explicit all-`None` `ClientTimeout`; `routes/count_tokens.py`
normalizes before keying and has a third decode arm; `tests/test_compaction.py`'s two EMA witnesses now drive production
`poll_kv_cache`. `S16`'s composer recorded `T018`'s own landing among its seven `Invalidates On` triggers — line 2 names
`server.py:119` starting to pass `client_max_size`, and line 3 states "this is `T018` step 1's own surface, so the
post-implementation recheck of this seam is mandatory"; `S17`'s §Invalidates On names the `timeout=` spelling at
`messages.py:137-141` and the session default's armed component set; `S18` and `S19` each carry a member row retired by
run 41's amendments. Run 42 recorded that every one of the four still has **its one post-repair recomposition unspent**,
so this landing dispatches four compositions rather than capping anything.

**Extraction only.** This pass checked no claim, composed no seam, called nothing safe, wrote no verdict file, created no
invariant and promoted nothing (E1/E8). It ran no project code (E2); the only files opened were these two ledgers, the
four `S16`–`S19` composition artifacts for their `Invalidates On`/Notes text, the installed distribution's own identity
files, and the registry record named below — reading, never executing (E4).

## Row 57 — the one atomic firing

**Cell writes — one row, one column pair changed, the rest held.**

| Row | Bundle | State | Routing | Permanent | Claim text | Subject | Tasks | Scope | Seams | Check |
|---|---|---|---|---|---|---|---|---|---|---|
| 57 | `B107` → **`B315`** | `HOLD` → **`recheck`** | `none` → **`unverified`** | `CL:aiohttp/per-request-timeout-override` — **retained** | unchanged | `pkg:aiohttp@3.13.3` — unchanged | `T007, T018` — unchanged | `single` — unchanged | `S8, S17` — unchanged | `seam:S8/S17` — unchanged |

**Which listed trigger fires, named honestly.** Of the four recheck reasons, this is **reason 4 — a new observation
contradicts it** — where the contradicted thing is the *coverage basis*, not the proposition. `S17`'s composition record
`compositions/S17-stream-deadline.md` §Notes **N1** states that member 57 "is grounded in a different aiohttp version
than the seam subject declares": its verdict cites upstream `client.py` at tag `v3.12.15` (`sha256:526c30a0…`,
`Source Version: aiohttp v3.12.15 upstream source (installed version unavailable)`), while 396/397/330/331 and the
installed bytes are 3.13.3, and the member's own `Invalidates On` line — "Use of an aiohttp version whose request timeout
behavior differs from v3.12.15" — "is not discharged by evidence at 3.13.3". That same artifact's §Invalidates On line 1
names this exact repair: "**Re-verification of member 57 at the pinned 3.13.3 bytes (N1)** — replaces `sha256:526c30a0…`
with `sha256:7ce41fc1…` and would make the version skew a resolved rather than an open condition." The seam leans on 57
for the timeout-construction semantics its dimension 1 and CE-1 are built from, and the wave just landed the all-`None`
spelling whose interpretation depends on them; recomposing `S17` while its premise-57 remains undischargeable would spend
the recomposition on an open question.

Reason **2** is *adjacent but not the trigger*, and the ledger's earlier reading of it stands: run 40's §freshness-basis
disclosure (`claims.md:2047`) declined to fire 57 on the ground that "a listed trigger is a change to the named version,
not a shortfall in the source the checker could reach" — the Subject cell has named `pkg:aiohttp@3.13.3` since run 3 and
the installed distribution has never moved. This pass does not restate reason 2 as having fired. What changed since run 40
is that the drift became a **recorded load-bearing gap on a seam** (N1) rather than an incidental shortfall, and that the
seam carrying the gap is now due its post-repair composition.

**The proposition is unchanged, and this pass did not test whether it should be.** Per E1/E2 the collector read no
timeout-selection logic and reached no view on whether 3.13.3's semantics moved; `S17`'s N1 records only that "nothing in
the consulted 3.13.3 bytes contradicts the replacement rule", which is the composer's observation about bytes it was
reading for a different purpose, not a verdict. The sentence as written is already falsified outright if the installed
source behaves differently — a `DOES-NOT-HOLD` naming what 3.13.3 does instead is the checker's answer, not a
rewording's — so the row re-enters blind checking with its text verbatim. No second row: the proposition is already
rowed, and dispatch forbids duplicating what current evidence answers.

## The version identity the check can actually pin

The fired verdict reached for an upstream git tag because that is the identity it could resolve; the installed tree is
reachable from the checker's own read, and every other aiohttp verdict on file is written in that identity (e.g.
`verdicts/claim-63.md`: `Source: Installed .venv/lib/python3.13/site-packages/aiohttp/client_reqrep.py`,
`Version: aiohttp 3.13.3`). The recheck therefore goes out with the installed distribution's identity named as the
universe of discourse:

| Identity | What the fired verdict carries | What this dispatch pins | Basis, by reading |
|---|---|---|---|
| `aiohttp` distribution | `v3.12.15` upstream tag, "installed version unavailable" | **3.13.3** | `.venv/lib/python3.13/site-packages/aiohttp-3.13.3.dist-info` (`Version: 3.13.3`) and `aiohttp/__init__.py:1` → `__version__ = "3.13.3"` |
| `aiohttp/client.py` — the unit the proposition names | `sha256:526c30a03bab9830…` | **`sha256:7ce41fc1c2142f5346015455e030e3ebec22a41cde0fc299a66ce13be2cb2a9e`** | digest of the installed file, which is the same digest rows 50/58/232/234/328/331/379/384/385/389/396/397/413 already record for that path |
| Subject grammar | `pkg:aiohttp@3.13.3` | `pkg:aiohttp@3.13.3` — unchanged, and now answerable at the identity it declares | §2 typing: a `pkg:` row fires on package/runtime identity and dependency-change rules, never on a repo edit |

These digests establish only that the source identity the row names is reachable and differs from the one the fired
verdict cited — never whether the proposition still holds (§3e: a freshness comparison, not a verification).

**Closing evidence for the row.** `verdicts/claim-57.md` re-written against installed 3.13.3 `client.py` with its
`## Source Version` naming `aiohttp 3.13.3` and its `## Source Fingerprint` at `7ce41fc1…`, and its `## Invalidates On`
restated against the installed identity rather than against "an aiohttp version whose request timeout behavior differs
from v3.12.15". The registry's `CL:aiohttp/per-request-timeout-override` record carries the same skew
(`invariants/buster_ripper/assumption-claims.md`, `Source`/`Version`/`Fingerprint` at `v3.12.15` / `526c30a0…`);
republishing those fields is the promotion lane's write — this pass edits no registry record and invents no handle.

## Run 44 bundle (fresh key — B1–B314 are history; one agent per bundle)

| Bundle | Rows | Subject | Why tied |
|---|---|---|---|
| B315 | 57 | `pkg:aiohttp@3.13.3` | solo — installed `client.py`'s request-timeout selection (`ClientSession._request`'s `timeout` parameter, and `get`/`post`/`request` forwarding into it). Rule 2 would pair it with 396/397, which are settled from the same file and object, but the dispatch forbids re-bundling a row that still answers and both are `HOLD`/`none` on these exact bytes, so it takes its own key (rule 4). Rule 1 holds (no cross-subject padding); rule 3 not approached. **1 bundle ⇒ 1 checker agent.** |

## Not fired, with reasons — nothing silent (E6/E7)

- **`S16`'s seven other members** 390, 412, 392, 393, 394, 395, 230 — all `HOLD`/`none`, and their cited units are
  installed `web_app.py`/`web_request.py`/`web_protocol.py`/`web_exceptions.py`/`web.py`/`web_runner.py`/`web_server.py`,
  which the wave did not touch. The wave's change is at `server.py:125` — *repo* code, which is not any `pkg:` row's
  cited unit (§2 typing; run 37's §1 and run 40's §2 both hold this line). `S16` moves because its **chain** moved, which
  is the composer's trigger, not a member recheck.
- **`S17`'s 396, 397, 330, 331** — same reason, plus their verdicts already carry the `7ce41fc1…` installed digest.
- **`S18`'s 51, 68–72, 137, 398, 413, 400, 401, 402, 403, 414, 405, 406, 407, 416** and **`S19`'s 126, 408, 415, 410,
  411** — every one `HOLD`/`none` on installed-package or stdlib sources under an unchanged CPython 3.13.8. The
  `count_tokens.py` third decode arm and the rewritten `test_compaction.py` witnesses are repo code at those seams'
  repo ends; the recomposition reads them as chain movement. Row 416's check and promotion landed after run 43 wrote it
  (`HOLD`, `CL:python/bytes-decode-non-text-codec-error`), so `S18` §MC-2's atom is now evidence rather than a gap.
- **`S8`** — row 57 is also a member of `S8`, whose `State` is `out-of-scope`. Its cell is **untouched**, per the
  dispatch's "`S1`–`S15` stay as they are". Recorded so 57's firing is not read as an opening of `S8`: the verdict returns
  to the row; the recomposition set this landing dispatches is the four CR3 seams.
- **Rows `3`/`20`/`93`/`391`/`399`/`404`/`409`** and seams `S1`–`S15` in all seven columns: terminal cells are history.
- **No new row** for the wave's repo-side facts — `create_app`'s new kwarg, the all-`None` spelling, the normalization
  order, the third decode arm, the EMA witnesses driving `poll_kv_cache`. All are repo code and repo **state** (E4): the
  composers read them through the seam records and Forward/Yield/QA arbitrate them; this gate's subject authority does
  not extend to them, and run 36's scope correction is the standing precedent. Recorded rather than dropped silently.
- **No re-bundling of any `HOLD` row**, and no bundle key spanning old and new work: `B315` carries exactly one row.

## Run 44 seam delta (written in [seams.md](seams.md) §Run-44; no seam re-verified)

Four `State` cells, `new` → `recheck`, and nothing else — no member, dimension, boundary value, subject list, `Routing`
or `Permanent` cell moved. `S16`–`S19` keep `Routing: absorbed` (run 42's routing of their first-composition findings
stands: those findings were absorbed into `T018`/`T019`/`T020` and are now the thing being re-composed), which is the
ledger's own pairing — `S6` carried `recheck` + `absorbed` deliberately through run 26 for the same reason. `Permanent`
stays `—` on all four: the `## Composition Index` still holds the same five `CO:` records (`lifecycle-teardown`,
`tls-posture-crossing`, `format-value-crossing`, `payload-and-header-pairing`, `env-var-flag-crossing`) and none covers
these boundaries, so §3c's "retain its handle" clause has nothing to retain and no composer is skipped.

| Seam | State | Members | Boundary value | Dimensions | Routing | Permanent |
|---|---|---|---|---|---|---|
| S16 | `new` → **`recheck`** | unchanged (7) | unchanged | unchanged | `absorbed` | `—` |
| S17 | `new` → **`recheck`** | unchanged (5) | unchanged | unchanged | `absorbed` | `—` |
| S18 | `new` → **`recheck`** | unchanged (18) | unchanged | unchanged | `absorbed` | `—` |
| S19 | `new` → **`recheck`** | unchanged (5) | unchanged | unchanged | `absorbed` | `—` |

**Dispatch shape for the orchestrator:** 1 claim bundle (`B315`, row 57, installed aiohttp 3.13.3 `client.py`) → then the
four seam recompositions, each of which is the unspent post-repair allowance run 42 recorded. No row that still answers
is in scope for either step.

## Hashes and repository state

`claims.md` before the first write: `sha256:3afed35fad28ca62eeebe26aa31e2a5ab09fe5ad0bbe163270448646ffd59eda`; after the
row-57 cell write and this section: the value in this pass's receipt. `seams.md` before:
`sha256:b5cd927702fc168d5b41647f1e22ba785bec949d621d236d8bec97881a25bd12`; after the four `State` cells and its §Run-44
changelog: its receipt value. Repository state at this landing: `git-commit:9c90675f69998e0ea7847c6b456c2e3f46f8a81c`,
which supersedes the `5f7f9b9d…` baseline runs 41–43 recorded; the installed aiohttp distribution is
`3.13.3`, unchanged across that movement, and `client.py` is byte-identical to what run 37 recorded.

---

# Run 44 atomic promotion pass (2026-10-06) — one `claims.md` cell + one registry `CL:` refreshed in place

Dispatch: mode `atomic`, `assumption-promote`. Inputs read: this ledger, the single in-scope verdict artifact `verdicts/claim-57-r2.md`, and the registry's `Claim:` records plus the generated `## Claim Index`. `seams.md` was opened for row-scope only and no seam cell moved. No plan, handoff, chronicle, QA plan, iteration report, production source, or installed source was opened; no `.git/` path was read or resolved; nothing was checked, nothing was composed, no `Invariant:`/`Override:`/`Entity:`/`Scenario:` record was created or touched, and no behavior was approved. The descriptive record says what the installed source established, not what is intended. Ledger hash before the first write: `sha256:ca56decab8b187b91c26e6e14bbf877093ebbb4e26a3fe646c76c0fe2644e213`.

**In-scope set: 1 row (57).** The mode's filter was applied mechanically to every numbered claim row before any write; exactly one row had State `new`/`recheck` with Routing `unverified` and `Check != —` — row **57** (`pkg:aiohttp@3.13.3`, `Check: seam:S8/S17`, `B315`), the row run 44 dispatched. No in-scope row carries `Check: —`, so nothing was excluded on that clause. Rows **48** and **338** again sit outside the invocation for the reason run 40 recorded (their State is a verdict token, not `new`/`recheck`); both cells untouched.

**Verdict read (checker bookkeeping, not promotion):** row 57 → `HOLD`. Its State cell now carries that result and its Routing is `none`, so Assumption Decide has nothing to route for it. `Check` does not include `user-claim`, so `decisions.md` was not opened and no Verification/Evidence/Consequence link was written — nothing to migrate and nothing malformed to report on that axis.

**Promoted: 0 new records · 1 updated in place · 0 ineligible.** Row 57 already carries `CL:aiohttp/per-request-timeout-override` from the 2026-09-27 pass, and the fresh verdict confirms the same proposition about the same subject, so the upsert law governs: the existing record was rewritten in place in `invariants/buster_ripper/assumption-claims.md`, at its current location, keeping its handle. No second current record was minted, no new aspect-slug was coined, and the row's `Permanent` cell needed no change. `Statement` is the claim row verbatim (identical to the prior publication, so identity is unchanged) and `Subject` is the row subject.

What the refresh replaces is the record's **evidence basis**, which is exactly why run 44 fired the row: the prior publication was pinned to `aiohttp v3.12.15` upstream GitHub source with installed version unavailable and a single digest, while the fresh verdict is pinned to the installed distribution — **aiohttp 3.13.3** — with the decisive locator at `client.py:584-596` and three content-addressed digests (`client.py`, `helpers.py`, `connector.py`). `Actual:` now states what 3.13.3 does, including the consequences the v3.12.15 reading could not establish from installed bytes: an all-unset request-time `ClientTimeout()` arms no timer at all; `timeout=None` and a bare number take the same replacing branch and so also discard the session's `total`; nothing merges with `DEFAULT_TIMEOUT` at request time; and `ws_connect`'s `ClientWSTimeout` is ruled out as a counterexample rather than left unexamined. `Invalidates-On:` was re-derived for the `pkg:` subject type — version plus the keyed file digests, plus the four specific mechanism changes the verdict names — so the record cannot fire on an unrelated repo edit or a line shift. `Provenance` points at `verdicts/claim-57-r2.md`; the r1 artifact stays on disk as history and git retains the superseded field values.

**Eligibility, as the gate reads it — not inferred.** Line 1 is exactly `VERDICT: HOLDS`, line 2 is exactly `PROMOTION: ELIGIBLE` with no annotation (contrast the `PROMOTION: ELIGIBLE — {annotation}` record-format defect run 40 reported on rows 402/403), the row is atomic and names one subject, and Source, Source Fingerprint, Invalidates On and Checked At are all present and non-placeholder. Source Version is available, so the `unavailable` clause is not in play; the digests are the stability requirement and they are content-addressed.

**What this pass did not do.** No claim was checked and no fingerprint recomputed — the checker captured both, and re-opening installed source here would put the plan's evidence and the promote role in one context. No stale record was swept or refreshed: only row 57's artifact was touched. `S17`'s `State`, `Routing` and `Permanent` cells were **not** moved, and no `CO:` record was written — composition promotion is a separate invocation. What this pass does supply to it is a row 57 that reads `HOLD`/`none` on the 3.13.3 bytes, which is the cell run 44's recomposition had to flag as **N7** because the ledger still showed it unresolved; `S8`, whose member list also contains 57, keeps its `out-of-scope` cells per run 44's "`S1`–`S15` stay as they are". No `QB:assumption-registry/*` item was opened: the refreshed record sits under a handle the registry already carried, so nothing new is projected.

**Index.** `rebuild_index.py` was run for `invariants/` and then in `--check` mode. Rebuild: `REBUILT: invariants/index.md (12 invariants/overrides, 12 entities, 202 claims, 5 compositions)`. Check: `CHECK OK: 12 invariants/overrides, 12 entities, 202 claims, 5 compositions, index.md current`, exit 0 — no duplicate identity, and the `## Claim Index` row for `CL:aiohttp/per-request-timeout-override` now carries the 3.13.3 version and the three digests. The 202 is the file's own count of `Claim:` records (the 412–416 records published by other lanes since run 40's 199 are in it); this pass added no record, removed none, and moved no other record's fields.

**After this fold, the atomic queue is empty.** No numbered claim row in this ledger remains in State `new`/`recheck`, so there is no further atomic promotion work for run 44's dispatch; the four `recheck` seams are composition-lane work.

---

# Run 45 — scoped amendment (2026-10-06): the one unrowed atom `S18` Note N6 names — the `web.Response` body round trip

Dispatch: **scoped amendment**, mode `one unrowed atom with production reach`. Required ledger delta: rows settling
aiohttp's `web.Response` body round trip, each a single third-party proposition under fresh bundle key(s), plus a live
seam only if the atoms genuinely cross a boundary. Check nothing, compose nothing, decide nothing, fix nothing.

**Why now.** `S18`'s post-repair recomposition recorded Note N6 and explicitly refused to make it a finding: its words
were that the CR3 delta's output flows into `messages.py:101-107` and "there the text never arrives", because
`web.Response(body=<str>)` was reported to store a payload object rather than bytes, so `isinstance(result.body, bytes)`
is false and `str(result.body)` yields an object repr. It declined the finding because "the deciding sources
(`aiohttp/web_response.py`, `aiohttp/payload.py`'s payload classes) are cited by no S18 member claim" — i.e. the
argument's every link is a library proposition this ledger has never rowed. That is the class of thing this pass exists
for: the assumption is still visible as an assumption, and the note left it in prose owned by no lane. Per the dispatch's
own freeze, **nothing here asserts whether `buster_ripper`'s retry fires** — repo code is out of this gate's scope
(run 36's standing correction), so every row below is about the library or the language alone.

**Extraction only** (E1/E2/E3/E4/E8). No project code was run and no session file was opened by a checker; the only
files read were these two ledgers, `compositions/S18-decode-failure-surface.md` §Notes N6/§Sources for the reasoning
being rowed, `plan-summary.md`'s `T019` row for task attribution, and the installed distribution's own identity.
Registry consulted (§3e): `invariants/index.md` `## Claim Index` carries **no** record for the `web.Response` body
round trip — the only `web_response.py`-pinned record is `CL:aiohttp/web-response-explicit-content-type` (row 155,
explicit `Content-Type` alongside a byte body), which asserts nothing about what `body=` stores or what the getters
return. Session-ledger match: rows 155, 226-228 and 236 cite `web_response.py` for content-type, prepare/head and
`Content-Length`; none settles any proposition below. So **`i = 0`, and it is the fact of no coverage, not a skipped
lookup** — all eight rows are `new` and every one goes to a checker.

## Identity the eight rows name

| Identity | Value | Basis, by reading |
|---|---|---|
| `aiohttp` distribution | **3.13.3** | `.venv/lib/python3.13/site-packages/aiohttp-3.13.3.dist-info`, `aiohttp/__init__.py:1` → `__version__ = "3.13.3"` — unchanged since run 44 pinned it |
| `aiohttp/web_response.py` — the unit rows 417-422 name | `sha256:3ca73388d5382e67ed5ea295be844cac06ce7950a5a5237e8b39c7b2259ece65` | digest of the installed file; **identical** to the pin run 37's table recorded for that path and to the digest the registry carries under `CL:aiohttp/web-response-explicit-content-type`, so these rows enter checking against bytes no verdict has ever read for these propositions |
| `aiohttp/payload.py` — the unit row 423 names | `sha256:3ba9ec60d50b2fb01e33673227a4d85fbde77159cbe71270b7901e831c0c2aac` | digest of the installed file; **first pin in this ledger** — N6 read it for its own note and cited no member claim, per its `## Sources` entry "`:151`, `:350`, `:430-457`, `:1109-1120` (registry) — N6" |
| Subject grammar | `pkg:aiohttp@3.13.3` (417-423), `lang:python` (424) | §2 typing — neither fires on an unrelated repo edit |

Hashes establish reachability of the named source only, never whether any proposition holds (§3e).

## Rows

| # | Claim | Subject | Tasks | Scope | Seams | Check | Bundle | State | Routing | Permanent |
|---|---|---|---|---|---|---|---|---|---|---|
| 417 | Constructing aiohttp's `web.Response` with a `str` `body=` argument raises no exception in `__init__` or in the `body` setter | pkg:aiohttp@3.13.3 | T019 | single | S20 | seam:S20 | B316 | HOLD | none | CL:aiohttp/web-response-str-body-acceptance |
| 418 | For an aiohttp `web.Response` constructed with a `str` `body=` argument, the `body` getter returns a `bytes` value rather than a payload object | pkg:aiohttp@3.13.3 | T019 | single | S20 | seam:S20 | B316 | DOES-NOT-HOLD | absorbed | CL:aiohttp/web-response-body-getter-type |
| 419 | For an aiohttp `web.Response` built with a `str` `body=`, the bytes its `body` getter returns decode back to that same string | pkg:aiohttp@3.13.3 | T019 | single | S20 | seam:S20 | B316 | UNCLEAR | malformed | — |
| 420 | An aiohttp `web.Response` built with a `str` `body=` argument sends the encoded text of that string as its body over the wire | pkg:aiohttp@3.13.3 | T019 | single | S20 | seam:S20 | B317 | HOLD | none | CL:aiohttp/web-response-str-body-wire-bytes |
| 421 | The `text` property of an aiohttp `web.Response` built with a `str` `body=` argument returns that string | pkg:aiohttp@3.13.3 | T019 | single | S20 | seam:S20 | B317 | HOLD | none | CL:aiohttp/web-response-str-body-text-property |
| 422 | aiohttp's `web.Response` class exposes no accessor that returns the `str` passed to its `body=` initializer | pkg:aiohttp@3.13.3 | T019 | single | S20 | seam:S20, absence, native | B317 | DOES-NOT-HOLD | absorbed | CL:aiohttp/web-response-str-body-accessor-surface |
| 423 | `str()` of an `aiohttp.payload.StringPayload` instance returns the text it was created with, not the default object repr | pkg:aiohttp@3.13.3 | T019 | single | S20 | seam:S20 | B318 | DOES-NOT-HOLD | absorbed | CL:aiohttp/string-payload-str-representation |
| 424 | Python's `str()` on an object whose type defines no `__str__` falls back to that object's `__repr__` | lang:python | T019 | single | S20 | seam:S20 | B319 | HOLD | none | CL:python/str-fallback-mechanism |

**Task attribution.** Every row is `T019`, the CR3 member whose stream error arm landed the module-private
`_error_body` producing the `str` that N6 says "the delta put … onto the same dead path" (`plan-summary.md:384`).
N6 records that the condition predates that delta, so these are not `T019`-introduced facts — `T019` is simply the
task this ledger has that this surface under, which is what the `Tasks` column is for ("do other tasks depend on
this" stays a lookup, §3). No row is `shared`: no second task names this crossing.

**Each row is one proposition** (E5), each is answerable by reading installed source (E2) without the plan (E3), and
none is about repo code or repo state (E4). N6's mechanism details — which class the setter instantiates, the
line ranges `payload.py:430-457`, the "no `__str__`/`__repr__` in `payload.py`" grep, the `tuple`/`reify` shape — are
**evidence for the checker's `Actual:`, not propositions to row**. Row 424 is the general language rule behind 423 and
is independently falsifiable; 418 and 419 are deliberately type-versus-content rather than one compound claim, because
`isinstance` and "the text survives" are separately falsifiable (a getter could hand back `b""`).

**Polarity, stated per row, because N6's outcome is the thing a verdict must discriminate against.** Six rows are
worded so that the outcome N6 assumed **falsifies the row**; two are worded **as N6 states them**, and the reason
differs for each:

| Row | Wording versus N6's assumed outcome | Why |
|---|---|---|
| 417 | neither — a premise N6 shares with any fix | N6 never asks whether `body=<str>` is accepted at all; its scenario requires the object to exist. Rowed at the neutral form so the checker names the outcome: if 3.13.3 **rejects** a `str` body, that answer is a `DOES-NOT-HOLD` here and N6's chain is wrong at its first link, which is exactly the "row the question that way" case the dispatch names. |
| 418, 419, 421, 423 | **opposed** — N6's outcome falsifies each | These are the four load-bearing links: `HOLDS` retires N6 at that link, `DOES-NOT-HOLD` confirms it and puts the `Actual:` in the verdict. Mutually exclusive with the note, so no verdict can agree with both. |
| 420 | **as N6 states it** | N6 offers "the peer-visible bytes are unaffected … so nothing at this boundary mis-settles a request" — a reassurance carried by no row. It is rowed at the note's own polarity because the conservative direction is the one worth spending a checker on: `HOLDS` retires the reassurance as evidenced, `DOES-NOT-HOLD` escalates the note's own severity classification. |
| 422 | **as N6 states it**, and bracketed by 421 | "there the text never arrives" is an exhaustive-surface claim about the class, which no single-accessor row can settle. 421 and 422 cannot both `HOLD` by construction, so the pair returns a settled answer whichever way the bytes go; only 422 carries the `absence, native` reason, which is what §3 requires be checked regardless of task count. |

**Nothing here is a verdict on N6.** This pass read no library source beyond the two digests above and the version
identity, held no view on which way any row lands (E1), and reached no conclusion about whether the note's chain, the
retry path, or the note's refusal to raise a finding is correct (E8). `S18`'s `State`, `Routing` and `Permanent` cells
are **untouched** — its post-repair recomposition is spent and its Notes stand as written.

## Run 45 bundles (fresh keys — B1–B315 are history; one agent per bundle)

| Bundle | Rows | Subject | Why tied |
|---|---|---|---|
| B316 | 417, 418, 419 | `pkg:aiohttp@3.13.3` | the `body=` initializer → `body` getter round trip, settled by one read of `web_response.py`'s `__init__` and the `body` property; 417's answer decides whether 418/419's state exists at all (rule 2). 3 rows, rule 3 not approached. |
| B317 | 420, 421, 422 | `pkg:aiohttp@3.13.3` | the response's own text and serialization surface — the wire write, the `text` property, and the class's accessor set — one read of the same file; 421 and 422 are the two readings of one question and a checker must see them together to avoid contradicting itself (rule 2). |
| B318 | 423 | `pkg:aiohttp@3.13.3` | solo — `payload.py`'s `StringPayload`, a different unit than B316/B317's; rule 2 would pair it with 418, which is settled from `web_response.py`, so it takes its own key (rule 4). |
| B319 | 424 | `lang:python` | solo by rule 1 — a `lang:` subject never shares a bundle with a `pkg:` one, however adjacent the mechanism. |

**Dispatch shape:** 4 claim bundles ⇒ 4 checker agents, 8 rows. B316/B317/B318 are one-subject-per-agent; B319 is the
language rule. All three `pkg:` bundles are `aiohttp 3.13.3` and may run concurrently (≤4 in flight).

## Seam delta (written in [seams.md](seams.md) §Run-45; no seam re-verified)

One new seam, **`S20`**, state `new`, routing `unverified`, `Permanent: —`, members 417–424. Recorded because these
atoms genuinely cross boundaries: one `str` value enters a library object through the constructor and leaves through
three different doors — an attribute read back by a caller, a property, and the response's own payload write to the
wire — and no single row can establish that the value survives the crossing (only their composition can, which is the
composer's job, not this pass's, E8). Producer is `pkg:aiohttp@3.13.3`, so §3c's record-a-seam condition is met; the
consumer end is described as the response object's own read-back surface, not as any repo call site.

Why a fresh seam rather than membership of an existing one: N6's own suggested home was "`S9`/`S11`'s error-transport
family", and both are terminal — `S1`–`S15` are frozen by this dispatch and by run 44's standing "`S1`–`S15` stay as
they are". `S18` cannot carry it either: N6 states the atoms are "outside `S18`'s dimensions" and that its deciding
sources are cited by no `S18` member, so adding members there would move a `recheck` seam's member list against the
note's own scope statement while leaving its dimensions mis-described. `S20` keeps the freeze and gives the
composition lane a place for the chain. No other seam's cells moved.

## Not rowed, with reasons — nothing silent (E6/E7)

- **`isinstance(result.body, bytes)`'s outcome, and `str(result.body)`'s** — repo call-site shape. The rows give the
  library's side (418's type, 419's content, 423's coercion); which branch a particular consumer takes is composition
  over repo code and out of this gate's scope per dispatch and run 36.
- **Whether the SGLang streaming retry fires, whether `parse_context_overflow_error` matches, and whether
  `config.py:122-123`'s patterns are ASCII** — repo code, repo facts. Explicitly fenced by the dispatch.
- **`StringPayload`'s construction-time encoding choice, `payload.py:1109-1120`'s registry entry, and the
  `set_payload`/`_payload` precedence** — mechanism, not property; the checker records them under `Actual:` for 418,
  420 and 423 rather than each becoming a row. Rowing them would put N6's grep in the ledger as five propositions
  with no independent reach.
- **The default response charset (utf-8) for a text payload** — already reached by row 420's phrasing ("the encoded
  text of that string"): the charset *name* is evidence, and no link of N6's depends on which charset, only on
  whether the text survives. If the checker finds the wire path encodes under a charset that is not the response's
  own advertised one, that belongs in 420's `Actual:` and the composer's dimension list.
- **N6's `Predates the delta` claim, and its severity classification** — governance/history about the note itself,
  not a library proposition (E4); the composition lane already has it in `compositions/S18-decode-failure-surface.md`.
- **N8** (the `ClientResponse._released` read-after-failed-decode state machine, `client_reqrep.py:701-702`) — a second
  unrowed atom the same artifact recorded, and the composer there explicitly said its conclusion is "re-checked rather
  than treated as verified". Not this dispatch's atom; left visible for whichever lane is given it rather than dropped
  silently.

## Hashes and repository state

`claims.md` before the first write: `sha256:911db55e45c59ba0a42073b8be37a596836e4bd97fe2f0cc67ea32d01e64472b`; after
the eight rows and this section: the value in this pass's receipt. `seams.md` before:
`sha256:7a9f79f1ec0463d9a74bdbd355ba39b8658cebb26423e1abf8fed1266c6554bd`; after the `S20` row and its §Run-45
changelog: its receipt value. Installed aiohttp is `3.13.3` with `web_response.py` at `3ca73388…` (unchanged from
run 37's table) and `payload.py` at `3ba9ec60…`; no repo file was modified by this pass, so no repository-state row
is recorded — this run writes ledger only.

---

# Run 45 atomic promotion pass (2026-10-06) — eight `claims.md` cells + seven registry `CL:` publications

Dispatch: mode `atomic`, `assumption-promote`. Inputs read: this ledger (run 45's §Rows, §Identity, §Run 45 bundles
and §Hashes sections), `seams.md` for row-scope only, the eight verdict artifacts
`verdicts/claim-417.md`–`claim-424.md`, and the registry's `Claim:` records plus the generated `## Claim Index`. No
plan, handoff, chronicle, QA plan, iteration report, production source, or installed library source was opened; no
`.git/` path was read or resolved; nothing was checked, nothing was composed, no `Invariant:`/`Override:`/`Entity:`/
`Scenario:` record was created or touched, and no behavior was approved. Each descriptive record says what the
fingerprinted source established, never what is intended. Ledger hash before the first cell write:
`sha256:fef64e6e2598cf38cf919a219c122b37af38d269ee2d7337364d66960bb17402`; after the eight cells:
`sha256:713d6298bad42d446d4f1b0616cae5abbaa7d986992322efcc9b1d653eadd2b2`; after this section: the value in this
pass's receipt.

**In-scope set: 8 rows (417–424).** The mode's filter was applied mechanically to every numbered claim row before any
write: exactly these eight had State `new` with Routing `unverified` and `Check != —`, which is run 45's own dispatched
set. No in-scope row carries `Check: —`, so nothing was excluded on that clause. Rows **48** and **338** again sit
outside the invocation for the reason run 40 recorded (their State is a verdict token, not `new`/`recheck`); both cells
untouched. No in-scope row's `Check` includes `user-claim`, so `decisions.md` was not opened and no
Verification/Evidence/Consequence link was written — nothing to migrate and nothing malformed to report on that axis.

**B316's empty return message — handled on artifact content, not on the report.** The dispatcher reports that the
checker for bundle B316 (rows 417, 418, 419) returned an empty message. The three artifacts exist on disk and were read
directly; eligibility was taken from their own bytes, which is the gate this skill applies to any artifact. Two
independent content facts tie them to this dispatch rather than to a stale file elsewhere in the directory: each names
its row's proposition as the claim under check, and each fingerprints `web_response.py` at `3ca73388…` and
`payload.py` at `3ba9ec60…` under aiohttp `3.13.3` — the digests and version run 45's §Identity table pinned as the
bytes these rows entered checking against — with `Checked-At: 2026-10-06`, the dispatch date. Nothing was inferred from
the missing report: no result was imputed, no row was treated as settled because a sibling was, and the three verdicts
are mutually independent in the ledger.

**Verdicts read (checker bookkeeping, not promotion):** `HOLD` ×4 (417, 420, 421, 424), `DOES-NOT-HOLD` ×3 (418, 422,
423), `UNCLEAR` ×1 (419), `MALFORMED` ×0. State cells now carry those results; Routing is `none` on every `HOLD` and
`pending` on every non-holding verdict, so Assumption Decide routes each one exactly once. Rows 417/420/421/424 are
therefore terminal in this ledger; rows 418/422/423 carry a published `CL:` handle **and** stay `pending`, which are
independent writes — promotion publishes the evidence, routing sends the consequence to Decide (the run-40 precedent
on rows 399 and 409).

**Promoted: 7 new `CL:` records · 0 updated in place · 1 ineligible.** All seven landed in
`invariants/buster_ripper/assumption-claims.md` (the registry's existing descriptive-evidence file — no record was
moved), and `rebuild_index.py` was run followed by `--check`: `REBUILT: invariants/index.md (12 invariants/overrides,
12 entities, 209 claims, 6 compositions)` then `CHECK OK: 12 invariants/overrides, 12 entities, 209 claims, 6
compositions, index.md current`, exit 0 (202 → 209 claims; no duplicate identity reported). Handles: 417
`CL:aiohttp/web-response-str-body-acceptance` · 418 `CL:aiohttp/web-response-body-getter-type` · 420
`CL:aiohttp/web-response-str-body-wire-bytes` · 421 `CL:aiohttp/web-response-str-body-text-property` · 422
`CL:aiohttp/web-response-str-body-accessor-surface` · 423 `CL:aiohttp/string-payload-str-representation` · 424
`CL:python/str-fallback-mechanism`. Every `Statement:` is the claim row verbatim, every `Subject:` is the row subject,
and each record's Source/Version/Fingerprint/Invalidates-On/Checked-At are the checker's own structured fields —
absolute site-packages paths in `Source:` were made repo-relative (`.venv/lib/python3.13/site-packages/…`), the form
this file already uses; no digest, locator or caveat was re-derived.

**Why each handle is a first publication, not a duplicate.** Run 45's collector recorded registry coverage `i = 0` for
this surface; this pass re-checked each proposition against the `## Claim Index` before writing and found no match by
normalized subject plus semantic statement. The nearest neighbours are separate facts under their own handles and were
not rewritten: `CL:aiohttp/web-response-explicit-content-type` (row 155 — an explicit `Content-Type` header alongside a
**byte** body, which says nothing about what a `str` body is accepted as, what the getters return, or what reaches the
wire), `CL:aiohttp/response-header-exposure` and the `client-response-text-*`/`client-response-read-charset-application`
records (the **client** response side, `ClientResponse`, not `web.Response`), `CL:aiohttp/repeated-header-items-iteration`
(header collection, not body), and on the language side `CL:python/codecs-lookup-unknown-name` and the
`*-error-inheritance` records (other mechanisms, other fallbacks). The new 418/421/422 triple is three facets of one
class — getter return type, the `text` property's value, and the accessor surface as a whole — under three slugs
precisely because each flips independently: none of the three statements can be rewritten into another's verdict.
`CL:python/str-fallback-mechanism` is named for the mechanism rather than its outcome: under the value-flip litmus,
were the fallback anything other than `__repr__` the handle still names the claim, so no observed value leaked into
identity. `Invalidates-On:` clauses were taken from the verdicts and are keyed to the `pkg:`/`lang:` subject types —
version plus the named file digests, or the named docs pages — so none can fire on an unrelated repo edit or a line
shift.

**Eligibility, as the gate reads it — not inferred.** For all seven: line 1 is exactly `VERDICT: HOLDS` or `VERDICT:
DOES-NOT-HOLD`, line 2 is exactly `PROMOTION: ELIGIBLE` with no annotation (contrast the `PROMOTION: ELIGIBLE —
{annotation}` record-format defect run 40 reported on rows 402/403, which did not recur on any row dispatched after
that routing pass), the row is atomic and names one subject, and Source, Source Fingerprint, Invalidates On and Checked
At are present and non-placeholder. Source Version is available on all seven, so the `unavailable` clause is not in
play; every fingerprint is a content-addressed `sha256`, so a history rewrite cannot orphan the evidence. Row 422's
`DOES-NOT-HOLD` is a permanent verdict on an `absence`-flagged row, published as evidence of what the enumeration
found; whether its consequence matters is Decide's route, not this pass's.

**Not promoted: 1.** Row **419** returned `VERDICT: UNCLEAR` with line 2 `PROMOTION: INELIGIBLE — …`: never promotable,
so no handle, no registry row and no `Permanent` cell — and its `pending` routing stands on its own. The artifact names
its blocker as the noun, not the behaviour ("those `.body` bytes" has no bytes-valued referent, since 418 settles that
the getter hands back a `StringPayload`), and names its repair as re-extraction — either the type claim already
answered under 418's handle, or a round-trip claim about the payload's bytes. That wording belongs to the collector
lane; nothing here re-words the row, re-checks the source, or adopts either reading. The verdict stays valid for this
session.

**What this pass did not do.** No claim was checked and no fingerprint recomputed — the checker captured both, and
re-opening installed source here would put the plan's evidence and the promote role in one context. No verdict file was
written or edited, including B316's three: the empty return message is a dispatch-surface anomaly, and the remedy for
it is not an edit to evidence this pass has no authority over. Nothing was composed and no `CO:` record was written —
`S20`'s State (`new`), Routing (`unverified`) and `Permanent` (`—`) cells were **not** moved; composition promotion is a
separate invocation, and what this pass supplies to it is six member rows resolved to permanent `CL:` handles, one
`UNCLEAR` row without a handle, and 417's answer on the record — the link run 45's §Polarity table says the other
seven depend on for their state to exist at all. `S18`'s and `S19`'s records and Note N6 stand as written: this pass
neither confirms nor retires the note, it publishes what the rows the note left unrowed now say. No stale record was
swept or refreshed, no `QB:assumption-registry/*` item was opened (projection of these seven handles is Authority
Registrar's lane), and no `repo:` row was created, rechecked or recomposed. **After this fold the atomic queue is
empty**: no numbered claim row in this ledger remains in State `new`/`recheck` with Routing `unverified`.

---

# Run 45 routing (2026-10-07) — the four `pending` cells, by `assumption-decide` (iteration 084)

Dispatch: mode `atomic non-holding routing`. Inputs read: run 45's §Rows/§Identity/§Polarity/§Bundles sections,
`seams.md`'s `S20` row, the four verdict artifacts (`claim-418`, `-419`, `-422`, `-423`),
`compositions/S18-decode-failure-surface.md` §Notes N6/§Sources, and — for the Reach Gate, which asks for a
determination about repo code — the repository bytes at `messages.py`, `utils.py`, `config.py`, `chat.py`,
`backend.py`, `main.py`, `server.py` and `buster_ripper_legacy.py`, plus one executed check against the installed
library. Recorded in `evidence/n6-reach-and-base-comparison.txt`. Nothing was recomposed; `S20`'s `State` (`new`),
`Routing` (`unverified`) and `Permanent` (`—`) cells were not moved. No `repo:` claim row and no repo seam was
created: the run-45 fence holds, and the finding it produced is routed as a **task**, not as a ledger row.

**Cells written — Routing only, never State.**

| Row | State (unchanged) | Routing | Routed to |
|---|---|---|---|
| 418 | DOES-NOT-HOLD | `absorbed` | `T019.001`, ledger `DC-claim-418-r1` |
| 419 | UNCLEAR | `malformed` | returned to `forward-collect` — successor **row 425** named below |
| 422 | DOES-NOT-HOLD | `absorbed` | `T019.001`, ledger `DC-claim-422-r1` |
| 423 | DOES-NOT-HOLD | `absorbed` | `T019.001`, ledger `DC-claim-423-r1` |

Rows 417/420/421/424 stay `HOLD`/`none` — evidence history, not new work.

**What the four verdicts jointly decided outside this ledger.** The chain N6 described is real and is reached:
`messages.py:151-155` passes `body=` (not `text=`), so `:106` always takes its `else` branch and the matcher receives
an object repr, so the SGLang **streaming** context-overflow retry can never `continue` — on either arm of the loop.
The client-visible response is unaffected (420 and 421 hold, and the executed check printed the wire bytes and
`application/json`), which is why this is routed as an inert-feature defect and not a corruption defect. Base parity
was applied as the ranking authority and is **silent, not protective**: the tracked pre-refactor monolith contains no
`overflow`/`sglang`/`retry`/`backend`/`max_completion` at all, so nothing is owed to base and nothing is excused by
it. The condition is migration-introduced and predates this session (`iterations/001-collect.md` §B-005), and the
CR3 delta did **not** change which branch `:106` takes — it changed only which bodies arrive there, by making the arm
total. The repairs, the required proof and the rejected candidates live in `handoff-context.md`
§Deterministic Closure Ledger — 2026-10-07 and §Planned Task Context → `T019.001`.

**For the collector — the successor to row 419.** One new row, **425**, `pkg:aiohttp@3.13.3`, seam `S20`, fresh
bundle key, one proposition, checker-answerable from the digests this run already pins. It must give the noun a
decode-bearing referent: e.g. *"For an aiohttp `web.Response` built with a `str` `body=`, the payload its `body`
getter returns decodes back to that same string under the response's own charset."* Forbidden in the wording: the
phrase "those `.body` bytes" (419's own blocker), a re-row of the getter's **type** (owned by 418 under
`CL:aiohttp/web-response-body-getter-type`), a re-row of the `text` **property** (owned by 421 under
`CL:aiohttp/web-response-str-body-text-property`), and any `repo:` subject. Row 419 is not re-worded, re-checked or
re-dispatched under its own id; its `UNCLEAR` verdict stands as evidence. **`S20` cannot recompose while any member
is non-`HOLD`**, so 425 gates that recomposition; `S20`'s Routing stays `unverified` until then.

---

# Run 46 — scoped amendment (2026-10-07): the successor to malformed row 419 — row **425**, seam **S20** membership

Dispatch: **scoped amendment**, mode `one malformed row successor`. Required ledger delta: row **419** retired to
terminal history and its successor row **425** written under a fresh bundle key, with `S20`'s membership updated so no
seam carries a row whose verdict cannot answer it. 419 is retired the way rows 3, 20, 93, 391, 399, 404 and 409 are —
**its `State`, `Routing`, `Bundle` and `Permanent` cells and its extraction-time `Seams` declaration stand exactly as
run 45's passes wrote them, and it leaves live membership only.** Rows **417, 418, 420, 421, 422, 423, 424** keep their
current cells: all seven are `HOLD`/`none` or `DOES-NOT-HOLD`/`absorbed`, and all seven already carry a published
`CL:` handle from the run-45 promotion fold. `S1`–`S15` stay terminal and untouched; third-party subjects only; and
`S20` stays live and answerable, because 425 gates its first composition. This pass checked nothing, composed nothing,
decided nothing, fixed nothing (E1/E8), ran no project code (E2), and wrote no verdict artifact.

**Why this row exists at all.** 419 came back `UNCLEAR`, and its own artifact names the reason: the blocker is the
noun, not the behaviour. `verdicts/claim-419.md` records that "those `.body` bytes" **has no bytes-valued referent**
once 418 settles that the getter hands back the `StringPayload` object rather than bytes — "the claim is false or true
depending only on which reading of 'those `.body` bytes' is taken, so picking would be the whole basis of the verdict"
— and it files the repair as re-extraction, into either the type claim already answered under 418 or a payload-decode
claim about the round trip. Assumption Decide (`iterations/084-assumption-decide.md` §The malformed return — what row
425 must ask (for the collector), restated under the same date in `handoff-context.md` §Assumption routing — 2026-10-07)
routed the row `malformed` back to this lane **and specified the successor's wording**. That wording is held here;
which way it lands is the checker's business (E1).

**Numbering.** The ledger's sequence ended at run 45's row 424; run 45's promotion and routing passes wrote no claim
row. This pass takes **run 46** in both ledgers (run 27's rule: one handle names one pass across both files), row
**425**, bundle key **B320** (B1–B319 are history).

## The wording, and the four things it may not say

> For an aiohttp `web.Response` built with a `str` `body=`, the payload its `body` getter returns decodes back to that
> same string under the response's own charset

Taken verbatim from the routing's §For the collector, with one change to the **marks** and none to the **words**: the
sentence-final period is dropped, because no cell in this column carries one and a promotion pass copies the row into
a `Statement:` unchanged. Nothing was added, removed or re-ordered, and no qualifier was tightened or loosened.

| Constraint the routing imposed | How the row respects it |
|---|---|
| No phrase presupposing `.body` is bytes — **419's own blocker** | The operand is named as `the payload its body getter returns`, and the predicate is `decodes back to`. The row never calls the getter's return a bytes value, and never asks what bytes are there |
| The getter's **type** is owned by 418 (`CL:aiohttp/web-response-body-getter-type`) | "Payload" is the referent 418's verdict supplies, not a re-assertion of it. The row states no class, no `isinstance` outcome and no return-shape alternative, so a checker that read the return type again would be answering 418, not 425 |
| The `text` **property** is owned by 421 (`CL:aiohttp/web-response-str-body-text-property`) | Neither the word `text` nor any property appears in the row; `string` is the value going in and coming back out, not an accessor. The operation named is the payload's own decode, and the property that wraps such a call is a different door — 425 and 421 can disagree without contradicting each other, which is why they are two rows |
| No `repo:` subject | `pkg:aiohttp@3.13.3`, the same subject 419 carried. No call site, no `messages.py`, no read-back by a consumer, no repo state (E4) |

**One proposition (E5), and where the length comes from.** The row runs twenty-six words, and both qualifiers are
content the routing required rather than second claims: `the payload its body getter returns` is the operand 419 lacked, and
`under the response's own charset` names the condition under which the single predicate is asserted — it is what stops
the row from silently becoming "decodes under *some* codec the payload happens to carry", which would be a different
proposition with a different `Invalidates On`. There is one subject, one operand, one predicate, and one read settles
it, so a verdict can disagree with all of 425 or none of it. What the row deliberately does **not** answer: the class
the getter returns (418), what `str()` of the payload yields (423), what reaches the wire (420), what the `text`
property returns (421), whether the response's advertised charset and the payload's stored encoding can differ (the
charset *name* is the checker's `## What it actually does` evidence, recorded below as collected-and-not-rowed), and
anything at all about repo code.

## Run 46 claims

| # | Claim | Subject | Tasks | Scope | Seams | Check | Bundle | State | Routing | Permanent |
|---|---|---|---|---|---|---|---|---|---|---|
| 425 | For an aiohttp `web.Response` built with a `str` `body=`, the payload its `body` getter returns decodes back to that same string under the response's own charset | pkg:aiohttp@3.13.3 | T019 | single | S20 | seam:S20 | B320 | new | accepted-consequence | — |

**Cell inheritance from the retired row.** `Subject`, `Tasks`, `Scope`, `Seams` and the `Check` spelling are 419's own
cells, carried across unchanged — run 41's rule for a successor ("Each row's subject is its retired predecessor's
subject"), and the reason no new `Tasks` id appears: `T019.001`, the task routed from 418/422/423, is a repo call-site
remediation whose consequence already lives outside this ledger as `DC-claim-418-r1`/`-422-r1`/`-423-r1`, and naming it
here would put a `repo:` read-back inside a third-party row's provenance (E4). `Scope` stays `single` — no second task
names this crossing, exactly as run 45 found for its eight rows. `Check` is `seam:S20` alone, matching 417/418/420/421/
423/424 rather than taking `native`: the row asserts what the library does, not that a mechanism supplies a capability
(run 43's note on the same distinction). `State`/`Routing` are `new`/`unverified` and `Permanent` is `—`: extraction
never invents a handle, and no record on file asserts this proposition (§3e below).

## Retirement delta (old row → disposition → live rows that now carry it)

| Retired row | Cells kept | Why it retires rather than reopens | Successor rows |
|---|---|---|---|
| 419 | `UNCLEAR` / `malformed` / `B316` / `—`, and its `Seams: S20` declaration | The verdict is sound and stands: it refused to pick a reading of a noun with no referent, which is what `UNCLEAR` is for. What cannot be answered is not stale evidence awaiting a refresh but a proposition that was never well-formed, so a `recheck` would resend the same input to the same fork. Its `malformed` routing is terminal and is **left exactly as run 45's routing pass wrote it** — this pass re-routes nothing — and the artifact stays on disk as the blind record the successor is built from. §"Rewrite a row once" is satisfied, not evaded: 419 was never rewritten, and this is the one material correction the rule allows, made under a fresh id because the routing forbade re-dispatch under its own. `BLOCKED-EVIDENCE-DEFECT` does not apply — the corrected row's input differs materially from the input that produced 419's verdict: a named operand instead of a noun with no bytes-valued referent. | 425 (`S20`) |

**419's `Seams` cell still says `S20`, and that is correct.** A row's `Seams` entry is its extraction-time
declaration, not live membership (run 34's note, applied the same way to 391/399/404/409 at run 41). This file's
bundle table for `B316` and `seams.md`'s `S20` membership are where live membership is recorded, and both move.

## Run 46 bundles (fresh key — B1–B319 are history; one agent per bundle)

| Bundle | Rows | Subject | Why tied |
|---|---|---|---|
| B320 | 425 | `pkg:aiohttp@3.13.3` | solo (rule 4) — one read of `web_response.py`'s `body` getter and `payload.py`'s decode settles it, and every other `S20` row that reads those two units is already settled. Rule 1 bars pairing it with 424's `lang:python` subject; re-bundling any of 417/418/420–423 into a key that also names 425 would hand a checker claims it has already answered, which is the re-dispatch rule run 41 applied to `B300`/`B303`/`B305`/`B307`. Rule 3 not approached |

**Dispatch shape:** 1 claim bundle ⇒ **1 checker agent**, at most four in flight — trivially satisfied. The bundle's
subject is the same library and the same two pinned files the run-45 `pkg:` bundles read, so nothing about the domain
or the version moved since B316–B318 ran.

**Queue consequence, as ledger state rather than a decision.** With 419 out of live membership, `B316` names only
`417` (`HOLD`/`none`) and `418` (`DOES-NOT-HOLD`/`absorbed`), both settled and both published, so it carries nothing
dispatchable; `B317`/`B318`/`B319` already carried nothing. The atomic queue, which run 45's promotion fold recorded
as empty, is **no longer empty**: exactly one row — 425 — is `new`/`unverified` with `Check != —`, and it is the only
dispatch this ledger owes. `S20`'s live member set now consists of seven rows whose verdicts answer their
propositions plus 425, dispatched to receive one; whether the seam composes on its return, and whether it may leave
`new`, is the composer's and the decide route's write, not this pass's (E8). No cell of rows 417/418/420–424 was
edited, retired or re-bundled.

## Freshness basis and evidence lookups

`claims.md` hashed before the first write at `sha256:36fee430c8e250dde3dd1412c2d08d590f3d44c06a82419c9b85249ff81dec98`
— the value the run-45 routing pass recorded as its own after-write hash, so no pass wrote between the two;
`seams.md` at `sha256:7fc22845f14b3a3564cae9878b226ec901d1a9da092b75887763d80410cf2654` (run 45's routing pass left
its after-value to its checkpoint receipt, so this is this pass's recorded baseline). Both re-read after the writes at
the receipt values below. Repository state at this landing: `git-commit:08c0a9f7206635e1058faa85dd73d842e64a3384` —
`T019.001` landed (`e645ff0`) and was accepted-local (`7a6639b`) after run 45's routing recorded
`9c90675f69998e0ea7847c6b456c2e3f46f8a81c`. **No repo file was modified by this pass**; it writes ledger only.

**Identity the successor names, re-read by hashing the named sources to establish scope change only** (§3e — a digest
never says whether a proposition holds):

| Identity | Value | Basis |
|---|---|---|
| `aiohttp` distribution | **3.13.3** | `.venv/lib/python3.13/site-packages/aiohttp-3.13.3.dist-info`; `aiohttp/__init__.py:1` → `__version__ = "3.13.3"` |
| `aiohttp/web_response.py` | `sha256:3ca73388d5382e67ed5ea295be844cac06ce7950a5a5237e8b39c7b2259ece65` | **byte-identical** to the pin run 45's §Identity table recorded for these rows — the same bytes 417/418/420/421/422 were checked against |
| `aiohttp/payload.py` | `sha256:3ba9ec60d50b2fb01e33673227a4d85fbde77159cbe71270b7901e831c0c2aac` | **byte-identical** to run 45's first pin of that file, and to the digest 419's own artifact fingerprints |

So 425 enters checking against the same bytes its predecessor read; the amendment is in the words, not in the sources.

**Session-ledger matching (§3d), performed for this proposition.** No live row asserts it. The neighbours are named and
separated rather than quietly reused:

| Row | Cells | Why it is not 425 |
|---|---|---|
| 418 | `DOES-NOT-HOLD` / `absorbed` / `CL:aiohttp/web-response-body-getter-type` | The getter's **return type** — "a `bytes` value rather than a payload object". It is the verdict that makes 419's noun impossible and supplies 425's operand; it asserts nothing about any decode |
| 421 | `HOLD` / `none` / `CL:aiohttp/web-response-str-body-text-property` | The response's `text` **property** returns the string — a different door of the same object. A verdict on 425 can go either way without touching 421, which is the test that they are two propositions |
| 420 | `HOLD` / `none` / `CL:aiohttp/web-response-str-body-wire-bytes` | What leaves over the wire, not what a read-back decodes to |
| 422 | `DOES-NOT-HOLD` / `absorbed` | The class's accessor surface as a whole; an enumeration, not a decode |
| 423 | `DOES-NOT-HOLD` / `absorbed` / `CL:aiohttp/string-payload-str-representation` | `str()` of a `StringPayload` — a **coercion**, not a decode. 423's falsification carries to no decode claim, and 425 does not contradict it |
| 413, 416, 414 | all `HOLD` | Client-side `ClientResponse` decode behaviour and `lang:python` codec/`errors=` semantics at `S18` — other subjects, other boundary |

**No existing row was marked `recheck`.** None of §"Check once"'s four reasons fires: neither named unit moved (both
digests above match run 45's pins), the package version is unchanged at 3.13.3, no relation end or mechanism moved,
and 419's `UNCLEAR` is the *absence* of an answer to a malformed proposition — not a new observation contradicting
another row's verdict. So rows 417/418/420–424 keep every cell run 45 left them.

**Registry `## Claim Index` consulted for this proposition only** (§3e; 209 `Claim:` records, the 39 `CL:aiohttp/*`
rows read for the match, nothing swept for staleness). No record states the payload-decode round trip, so `i = 0`,
`Permanent` stays `—`, and the row goes to a checker. The nearest neighbours, and why each is a separate fact:

| Candidate record | Why it is not a match for 425 |
|---|---|
| `CL:aiohttp/web-response-body-getter-type` (`DOES-NOT-HOLD`) | The getter's return type, verbatim row 418's wording — no decode predicate anywhere in it |
| `CL:aiohttp/web-response-str-body-text-property` (`HOLDS`) | The `text` property's value. Same input, same object, different surface; 421's `Statement` cannot be rewritten into 425's verdict |
| `CL:aiohttp/web-response-str-body-wire-bytes` (`HOLDS`) | The bytes written to the transport |
| `CL:aiohttp/string-payload-str-representation` (`DOES-NOT-HOLD`) | String coercion of the payload, not its decode |
| `CL:aiohttp/web-response-explicit-content-type` (`HOLDS`, row 155's handle) | An explicit `Content-Type` alongside a **byte** body — run 45 already ruled it out for these rows, and nothing about a `str` body's read-back decode follows from it |
| `CL:aiohttp/client-response-text-charset-source`, `CL:aiohttp/client-response-read-charset-application` (both `HOLDS`) | The **client** response (`ClientResponse`), a different class on the opposite side of the proxy |

**`## Composition Index` consulted for the moved member set** (§3c — `S20`'s membership after 419 leaves and 425
joins): the index holds six `CO:` records (`aiohttp-runner-to-on_cleanup/lifecycle-teardown`,
`cli-flag-to-outbound-call/tls-posture-crossing`, `cli-option-to-dump-dispatch/format-value-crossing`,
`client-response-to-relay-consumer/payload-and-header-pairing`, `serve-script-env-to-argv/env-var-flag-crossing`,
`test-double-to-poll-loop/async-witness-eligibility`) and none covers a `web.Response` constructor-in / read-back-out
boundary. So no composer is skipped and no `registry-safe`/`registry-counterexample` shortcut is available, and no `CO:`
handle enters or leaves `S20`'s `Permanent` — which stays `—`, along with its `new`/`unverified` cells.

## Collected and deliberately NOT rowed — routes recorded, nothing silent (E7)

- **The charset *name* a `str` body ends up advertised under, and whether the payload's stored encoding and the
  response's own `charset` can disagree** — the first is evidence, the second is the failure mode 425's verdict
  records if it is real. Not rowed: 425's own phrase ("under the response's own charset") already puts the condition in
  the proposition, and run 45 filed the same atom against 420 for the same reason. Route: `S20`'s composer reads
  whichever the verdict's `Actual:` names against 420's wire-direction clause.
- **`Payload.decode`'s signature, its default `errors=` policy, and `StringPayload`'s construction-time encoding
  choice** — mechanism behind a single predicate, not propositions with independent reach (run 45's rule for the same
  family of `payload.py` internals).
- **`len(resp.body)` raising and `resp.body.size`** — consequences of the return type 418 owns, recorded in 419's
  artifact as reading-A colour. No `S20` dimension asks for a byte count of the payload.
- **Any `repo:` read-back shape** — `messages.py:106` and the matcher behind it, `utils.py:123-140`,
  `config.py:122-123`, and `T019.001`'s done-when or its acceptance. Repo code and repo state, outside this gate's
  subject authority (E4) and fenced by the standing third-party-only ruling; their routes are `DC-claim-418-r1`,
  `-422-r1`, `-423-r1` and `handoff-context.md` §Deterministic Closure Ledger — 2026-10-07, all already written.
- **419 itself** — not re-worded, not re-checked, not re-dispatched under its own id, and not re-routed. The routing
  forbade it and the dispatch repeats it; its `UNCLEAR`/`malformed` cells are evidence, which is why they are quoted
  above rather than edited.
- **`S18` Note N6's standing and `N8`** — governance about another lane's composition artifact, and the second unrowed
  atom that artifact recorded. Neither is this dispatch's; both stay visible where they are, and this pass neither
  confirms nor retires either.

## Run 46 seam delta (written in [seams.md](seams.md) §Run-46; no seam re-verified)

| Seam | Members before → after | Movement |
|---|---|---|
| S20 | 8 → 8 | retired **419** leaves the list and **425** enters **in place**, keeping 419's position (run 41's convention); the `equality with the input across that round trip` dimension clause is re-worded to cite **425** in place of 419's three bytes-shaped alternatives. `Subjects`, `Boundary value`, every other dimension clause, and the `State`/`Routing`/`Permanent` cells (`new`/`unverified`/`—`) are untouched, as are the other seven members' positions |
| S16, S17, S18, S19 | — | untouched: no member, subject, boundary value, dimension, State, Routing or Permanent cell moved. Their post-implementation recompositions are their own lanes |
| S1–S15 | — | untouched and terminal, per the dispatch's freeze and run 44's standing "S1–S15 stay as they are"; `S15` stays `registry-counterexample`/`accepted-consequence` |

No claim was checked, no composition performed, no verdict written or copied, no invariant created, nothing promoted, no
settled row's cells or bundle moved or re-bundled, and no seam's `State` or `Routing` was touched. The ledger's one new
row is `new`/`unverified` under a solo fresh key, and its membership is `S20` only.

**Hashes.** `claims.md` after this section, and `seams.md` after the `S20` member list, its one re-worded dimension
clause and the §Run-46 changelog, are the two values in this pass's `COLLECT-PERSISTED` receipt — a file cannot record
its own post-write digest. Pre-write baselines are in §Freshness above, and they establish scope only: neither ledger's
bytes say anything about whether 425 holds.

---

# Run 46 routing (2026-10-07) — row **425**'s Routing cell, by `assumption-decide` (iteration 086)

Dispatch: mode `atomic non-holding routing plus one seam disposition decision`, over `claims.md` row **425** — the
successor run 46 rowed, the checker returned `VERDICT: DOES-NOT-HOLD` for, and the only non-`HOLD` member of the live
seam `S20`. Inputs read: run 46's §The wording/§Run 46 claims/§Freshness sections, `verdicts/claim-425.md`, `seams.md`'s
`S20` row and its §Run-45/§Run-46 changelogs, the run-45 rows' cells, the eight run-45 verdict artifacts for the seam's
other members, `compositions/S18-decode-failure-surface.md` §Notes N6 (the note `S20` exists to settle),
`handoff-context.md` §Deterministic Closure Ledger — 2026-10-07 and §Accepted risks, `plan-summary.md`'s `T019`/`T019.001`
rows, the registry `## Claim Index`/`buster_ripper/assumption-claims.md` for the four `CL:` handles this row neighbours,
`invariants/index.md` + both `packages`-equivalent `invariants.md` files for the cross-reference — and, because the Reach
Gate asks a question about repo code, the repository bytes at `routes/messages.py`, `routes/chat.py`,
`routes/count_tokens.py`, `routes/passthrough.py`, `utils.py`, `config.py` and `server.py`, plus one executed check
against the installed library. Recorded in `evidence/claim-425-charset-arm-reach.txt`.

**Cells written — Routing only, on one row.**

| Row | State | Routing | Routed to |
|---|---|---|---|
| 425 | `new` (**not written by this pass** — see below) | `unverified` → **`accepted-consequence`** | §Accepted risks `claim-425`, with `QB:assumption-ledger/str-body-charset-doc-divergence` |

Rows 417/418/420/421/422/423/424 are evidence history, not new work, and were not touched: 417/420/421/424 stay
`HOLD`/`none`, 418/422/423 stay `DOES-NOT-HOLD`/`absorbed` with their `DC-claim-*-r1` ledgers and their `T019.001`
destination standing as run 45's routing left them. Row 419 stays `UNCLEAR`/`malformed` in terminal history.

**`State` and `Permanent` are deliberately not written here.** `Never alter its evidentiary State` is the routing rule,
and the `CL:` publication is the promotion lane's act, not this one's. The row therefore still reads `new`/`—` while
`verdicts/claim-425.md` carries `VERDICT: DOES-NOT-HOLD` / `PROMOTION: ELIGIBLE` at the three digests §Freshness above
pins — byte-identical to the verdict's own fingerprints (`web_response.py` `3ca73388…`, `payload.py` `3ba9ec60…`,
`helpers.py` `435df4ec…`), re-hashed at this pass. What remains open on this row is bookkeeping in another lane, and
nothing else: publication records what the sources established and does not re-decide the disposition recorded here.

**The Reach adjudication, in one line per arm** (full census, spelling table and executed output in
`evidence/claim-425-charset-arm-reach.txt`):

| Arm of the verdict's counterexample | Reach | Established by |
|---|---|---|
| **A** — a declared non-utf-8 charset reaching the read, giving mojibake (`iso-8859-1`) or `UnicodeDecodeError` (`ascii`) | **`unreachable`** | a closed census: five `body=` sites (`messages.py:163` the only `str`; `chat.py:172`/`:188`, `messages.py:328`, `passthrough.py:44` all bytes from `resp.read()`), nine `text=` sites through `json_response`/`HTTPBadRequest`, and **zero** `charset=` arguments anywhere under `buster_ripper/` — the word appears only inside three comments. The falsifier needs a call-site *spelling*; the upstream controls `error_content`'s **value** and the status, never the presence of a charset argument, so no network, file or configuration input reaches it either. |
| **B** — a charset argument silently dropped (`charset=` with no `content_type=`) | **`unreachable`** | same census; and its own harm is the ignored argument, not a failed decode — the executed drop leaves `text/plain; charset=utf-8`, which is what the payload already is, so the round trip *holds* under this arm. |
| **C** — the spelling the package actually uses: `content_type="application/json"`, no charset declared, `.charset is None`, `.text` applies `or "utf-8"`, `StringPayload` encoded utf-8 | **`reached`, and HOLDS** | `messages.py:117` reads `result.text`, and `result` has exactly three producers (`_stream_single_request` :160-166 the str-body arm, :169-180 a prepared `StreamResponse` that `isinstance(…, web.Response)` shows can never pass the :105 guard, :182-184 the `json_response` 502 arm). Executed at the shipped spelling: `.charset` `None`, header `application/json`, and `.text == input` for an ASCII overflow body, a non-ASCII body, and a `U+FFFD` replacement body — `parse_context_overflow_error(.text)` returns a `ContextOverflowError`. Certified on file by `CL:aiohttp/web-response-str-body-text-property` (421, HOLDS) and `CL:aiohttp/web-response-str-body-accessor-surface` (422, its enumeration naming `.text` the only such accessor). |

Not `latent`: `latent` is a guard elsewhere preventing an input a path could otherwise deliver. Here the guard *is* the
call-site census — static, closed, re-checkable by one grep — so the Reach Gate's `unreachable` row applies, with its
freshness rule carrying the durable protection: a new `charset=` at that constructor, a new `str`-`body=` site, or a
return to reading `.body` makes the citation stale and the finding returns. This is not the 1%/0.01% category and no
bigger defect gates it; it simply has no producer.

**What the landed code is entitled to depend on, stated as the routing's answer to the dispatch.** `T019.001` — landed
at Checkpoint 197, accepted-local at [085-01-yield](../iterations/085-01-yield.md) on these same bytes — is entitled to
depend on arm **C** and nothing broader: *a `str` handed to `web.Response(body=)` round-trips through `.text` exactly
because the response declares no charset and `.text` falls back to the one codec `StringPayload` always encodes with.*
It is **not** entitled to the general premise that a `str` body survives the `body=`/read-back round trip, and no later
task, comment or invariant should state that — 425 refutes the universal, and both neighbouring `HOLDS` records already
carry the mismatch inside their own `Actual:` caveats, so nothing is being newly invented by refusing it. Two riders
this pass checked rather than assumed: (i) the hold is **total at that site** — `_value` is exactly
`s.encode("utf-8")`, so `.text` cannot raise on any payload `_error_body` can return, which is what keeps the repaired
probe from becoming a 500 path; (ii) the hold rests on **two independent defaults agreeing** (payload.py's hardcoded
`real_encoding = "utf-8"` and `Response.text`'s `self.charset or "utf-8"`) with nothing reconciling them, and aiohttp's
stable documentation promises the opposite rule — the encoding is *not* header-driven in the installed source. That
divergence is the latent trap the dispatch asked about, and it is recorded (§Accepted risks, and one doc line at
`QB:assumption-ledger/str-body-charset-doc-divergence`) rather than made into a row: the only reader of the
documentation is a future author at that constructor, which is the Reach Gate's inert-wording class, and the durable
guard against the trap is the **Invariant candidate** `T019.001`'s blueprint already leaves with Interview — *"reading a
body back off an `aiohttp.web.Response` the code constructed goes through `.text`"* — now evidenced by a live call site
and by this row's falsification of the alternative.

**Zero successor rows are owed by this row**, and none is named for the collector: the proposition a successor would
state — the arm-C round trip — is row 421's own `HOLDS` wording at the spelling 421's verdict enumerates, and
claims run 46 §"The wording, and the four things it may not say" already bars re-rowing the `text` **property** under
421's handle. `S20`'s terminal disposition is this pass's other write and is recorded in
[seams.md](seams.md) §Run-46 routing changelog.

**Invariant cross-reference (row 425): Neutral.** Session `invariants.md` does not exist for this session; the registry
holds `buster_ripper/dump/invariants.md` (six `PI:T002/dump_format/*`, all `Status: planned`) and
`buster_ripper/server/invariants.md` (two `PI:T006/upstream_tls/*`, all `Status: planned`); there is no
`buster_ripper/routes/*` file, and **zero `Status: active` or `Status: wip` entries** to be upheld, contradicted or
obviated. The `CL:` records cited are evidence inputs only — they authorize no behavior and were not classified as
aligned invariants. No invariant was written or promoted by this pass; the candidate above stays a candidate, and
recording it remains Interview's act.

**Zero Forks, zero `breaks-it`, zero `malformed`.** `FK-011` remains the session's only open Fork. Nothing here leaves
materially different outcomes unranked: the reached arm is certified, the unreachable arms have no producer, the repair
is landed and accepted, and the seam's question is closed — so the deterministic-closure gate had no candidate set to
rank and the Deterministic Closure Ledger gains no entry (it carries `absorbed` findings only, and this row is
`accepted`). `T019.001`'s done-when, the CR3 wave and every earlier seam stand untouched.

**Recomposition cap: none applies**, and none is recorded at `handoff-context.md` → §Capped seams. No composition
artifact for `S20` was supplied to this pass, none exists on disk, and no earlier routing pass ever routed a composition
of this seam — so there is no second pass to cap. `S1`–`S15` stay terminal and were not read for staleness; `S16`–`S19`
keep the `recheck`/`absorbed` pairing run 44 left them and their spent post-repair allowances; `S18`'s Note N6 stands as
written and is adjudicated outside that seam.

**Hashes.** `claims.md` before this pass's first write: not recoverable from the ledger — run 46's collect pass left its
own after-value to its `COLLECT-PERSISTED` receipt, exactly as run 45's promotion pass did, and this pass is the only
writer since (`verdicts/claim-425.md`, `Checked At` 2026-10-07T03:39:26Z, is the only artifact that landed after the row
was written). Measured after the one Routing cell and before this section:
`sha256:f7aab38e71edf4aa42833d9db82c840cbd44fb14cadce42fa9493b62726537e3`; the value after this section is in this
pass's checkpoint receipt. `seams.md` before this pass's first write, measured at this pass's first read:
`sha256:e09ad9b8f52a57c6830a223be6ba2d212f32100fb86ea62f818d8c7266ea57a4`; after the one `S20` Routing cell and the
§Run-46 routing changelog: this pass's receipt value. Repository state at this landing:
`git-commit:08c0a9f7206635e1058faa85dd73d842e64a3384` — the same head run 46 recorded, so `T019.001`'s landing
(`e645ff0`) and its acceptance (`7a6639b`) both precede this pass and no repo file moved under it. **No repo file was
modified by this pass**; it reads ledger and installed bytes and writes session artifacts.

---

# Run 48 — scoped adhoc extraction for `T021`'s gate surface (2026-10-17)

Dispatch: **scoped adhoc extraction, `T021` gate surface, third-party subjects only**. Blueprint read from
`../handoff-archive.md` §Trimmed 2026-10-17 (`097-yield`) → `T021 Planned Task Context`; task row from
`plan-summary.md:386`; the gate's own bytes from `tests/qa/test_qap_ms001.py` (`changed_lines` `:120-132`,
`diff_changed_lines` `:135-140`, `assert_documented_contents` `:143-211`, the process-free sensitivity witness
`:429-469`) and `tests/qa/harness.py` (`require_upstream` `:50-67`, `assert_upstream_response` `:174-194`,
`Proxy` `:197-313`); contract from `qa-plan.md` §TC-108 (`:222-232`) and traceability `:268`; pytest configuration
from `pyproject.toml` `[tool.pytest.ini_options]`. Ledger entered at `claims.md`
`sha256:936d912449fbc16ac13bebe20a23a28163c06db19492cff82f0e3d0df5bea457` and `seams.md`
`sha256:de78c8428aae8262f2ff30994563f6d67ee7c15ed5af1306f99cb7b611a28b1d` — recorded before the first
write, re-read after it.

**Scope honored, as the dispatch rules it.** No `repo:` row is created, rechecked or recomposed; no repo-only seam is
written. `S1`–`S20` are untouched and were not read for staleness. Every row below has subject `lang:python` or
`pkg:pytest@9.0.2`; repo units appear only in the seams' `Subjects`/boundary columns as the consumer end of a
library or external-service boundary, per run 40's `S15`/`S16` precedent. Extraction only: nothing here was checked or
composed, no seam judged safe, no verdict read as an instruction, nothing promoted (E1/E8). No project code was run
(E2) — every command was a read of `pyproject.toml`, the gate's test bytes, the installed `difflib`/`_pytest` sources,
`invariants/index.md`, and this ledger.

## The four narrow sets, and what each yielded

| Set | Result |
|---|---|
| 1 — tasks not yet started whose plan text changed since extraction | **empty.** `T021` is the only task added since the last extraction; no open task's plan text moved. |
| 2 — checked claims with a recheck reason from §"Check once" | **empty.** No prior row names `difflib`, `pytest.raises`, `monkeypatch`, `urlopen` or `subprocess.Popen`; the nearest neighbours are named and separated under §Freshness below. |
| 3 — tasks that did not exist at the last extraction | **`T021`, 12 new rows, 426–437.** The gate surface is `required:new-surface` per [096-forward-qa](../iterations/096-forward-qa.md) §Next: "the complete-body and rendering-source propositions have no existing seam". |
| 4 — seams whose member/transformation/consumer/dimension moved | `S1`–`S20` untouched; three **fresh** seams, `S21`–`S23`, each at a library or external-service endpoint. |

## Claims (12 new rows — 9 `lang:python`, 3 `pkg:pytest@9.0.2`)

| # | Claim | Subject | Tasks | Scope | Seams | Check | Bundle | State | Routing | Permanent |
|---|---|---|---|---|---|---|---|---|---|---|
| 426 | `difflib.unified_diff` prefixes each removed content line with `-` and each added content line with `+` | lang:python | T021 | single | S21 | seam:S21 | B321 | HOLD | absorbed | CL:python/unified-diff-content-line-prefixes |
| 427 | `difflib.unified_diff` prefixes each unchanged context line with a single space | lang:python | T021 | single | S21 | seam:S21 | B321 | HOLD | absorbed | CL:python/unified-diff-context-line-prefix |
| 428 | `difflib.unified_diff` prefixes its file-header lines with `---`/`+++` and its hunk-header lines with `@@` | lang:python | T021 | single | S21 | seam:S21 | B321 | HOLD | absorbed | CL:python/unified-diff-header-line-prefixes |
| 429 | `difflib.unified_diff` appends its `lineterm` value to each file-header and hunk-header line, not to content lines | lang:python | T021 | single | S21 | seam:S21 | B322 | HOLD | absorbed | CL:python/unified-diff-lineterm-placement |
| 430 | `difflib.unified_diff`'s `n` argument sets the number of unchanged context lines around each hunk and defaults to 3 | lang:python | T021 | single | S21 | seam:S21 | B322 | HOLD | absorbed | CL:python/unified-diff-context-line-count |
| 431 | `difflib.unified_diff` compares input lines for exact string equality, without whitespace or case normalization | lang:python | T021 | single | S21 | seam:S21 | B322 | HOLD | absorbed | CL:python/unified-diff-line-equality |
| 432 | `pytest.raises` fails the test when its block raises no exception of the expected type | pkg:pytest@9.0.2 | T021 | single | — | — | — | new | unverified | — |
| 433 | `pytest.raises(match=…)` searches its pattern as a regex within the formatted exception string, not a full match | pkg:pytest@9.0.2 | T021 | single | — | — | — | new | unverified | — |
| 434 | `monkeypatch.delenv(name, raising=False)` leaves an absent variable absent without raising `KeyError` | pkg:pytest@9.0.2 | T021 | single | S23 | seam:S23 | B323 | HOLD | none | CL:pytest/monkeypatch-delenv-missing-name |
| 435 | `urllib.request.urlopen` raises `urllib.error.HTTPError` carrying the response status on `.code` for an HTTP 4xx or 5xx answer | lang:python | T021 | single | S22 | seam:S22 | B324 | HOLD | absorbed | — |
| 436 | `urllib.request.urlopen`'s `timeout` argument bounds the wait for a peer that never answers | lang:python | T021 | single | S22 | seam:S22 | B324 | HOLD | none | CL:python/urllib-urlopen-timeout-scope |
| 437 | `subprocess.Popen` with no `env=` argument starts the child with the parent's environment at spawn time | lang:python | T021 | single | S23 | seam:S23 | B325 | HOLD | none | CL:python/subprocess-popen-environment-inheritance |
| 438 | `difflib.unified_diff` yields its `---` and `+++` file-header lines before any `@@` hunk-header line | lang:python | T021.001 | single | S21 | seam:S21 | B326 | HOLD | none | CL:python/unified-diff-header-yield-order |
| 439 | `difflib.unified_diff`'s hunk-header line contains a second `@@` after the opening one | lang:python | T021.001 | single | S21 | seam:S21 | B326 | HOLD | none | CL:python/unified-diff-hunk-header-closing-marker |
| 440 | `difflib.unified_diff` retains the input line's terminator on each content line it yields | lang:python | T021.001 | single | S21 | seam:S21 | B326 | HOLD | none | CL:python/unified-diff-content-line-terminator |
| 441 | `urllib.request.build_opener` skips its default `ProxyHandler` when the caller supplies a `ProxyHandler` instance | lang:python | T021.002 | single | S22 | seam:S22 | B327 | HOLD | none | CL:python/urllib-build-opener-default-handler-substitution |
| 442 | `urllib.request.ProxyHandler` with an empty proxy mapping installs no proxy for any host | lang:python | T021.002 | single | S22 | seam:S22 | B327 | HOLD | none | CL:python/urllib-proxy-handler-empty-mapping |

## Row-shape notes (E5 — one proposition per row)

- **426/427/428 are the three line kinds the gate's filter distinguishes, not one compound
  "diff format" row.** `diff_changed_lines` keeps a line iff `ln and ln[0] in "+-" and not
  ln.startswith(("+++", "---"))`; that single predicate needs the removed/added marking (426), the
  context-line prefix that must *not* be in `"+-"` (427), and the header lines that share the first
  character with content lines (428). A checker can find 426 true and 428 false — a header convention
  with no `---`/`+++` prefix — and the verdicts are independent.
- **429 is the terminator half of the same parse, and it is narrower than "unified_diff emits a
  newline".** The installed `difflib` appends `lineterm` only to the `---`/`+++`/`@@` lines
  (`difflib.py` `unified_diff`); content lines are `'-' + line` / `'+' + line` and carry whatever
  terminator the input line already had. Production passes `splitlines(keepends=True)` and joins with
  `""`, so its content lines stay separated while its headers abut the next line; the gate's
  `splitlines()` then recovers the content lines. The row states the library fact; the parse's
  dependence on it is the seam's dimension, not a second proposition.
- **430 is the default, not the invariance.** Production omits `n` and so emits three context lines
  (`_unified_diff`, `dump.py:262-270`); the gate passes `n=0`. That default is what puts at least one
  space-prefixed context line ahead of the first changed line in the gate's scenarios, which is why the
  hunk-header/content glue never swallows a changed line here. Whether `n` also leaves *which* lines are
  marked unchanged is 426/427's business and the seam's dimension; it is not asserted here.
- **431 is exactness, not "line order".** `SequenceMatcher` compares sequence elements with `==`; the
  gate's expected set and production's observed set are both produced by the same library from the same
  renderings, so a whitespace- or case-folding change in the library would hide a drift from *both* sides
  at once — the false-green shape this row exists to name. Order and hunk grouping are the same
  library's output on both sides and belong to `S21`'s dimensions, not to a row.
- **432/433 are the sensitivity witness's own two guards, written as pytest facts rather than as "the
  witness is sensitive".** `pytest.raises` fails on a block that raises nothing (432) — without it, a
  helper that stopped raising would leave the witness green and inert; `match=` is a regex search within
  the formatted exception string (433) — without it, the witness would pass on any `AssertionError`. Both
  are affirmative single-task claims outside a seam, so they are write-time checks for the gate author
  (`Check: —`), not dispatchable rows.
- **434/437 are the two halves of the environment boundary, split because they are different
  mechanisms.** `monkeypatch.delenv(..., raising=False)` is the deletion the autouse fixture performs
  (434, `pkg:pytest`); `subprocess.Popen` without `env=` is what carries that deletion into the proxy
  child (437, `lang:python`). Neither implies the other: a fixture that mutated a copy of the environment
  would leave the child dirty, and a `Popen` that passed an explicit environment would ignore the deletion.
- **435/436 are the preflight's two error arms, split because they are different classes.**
  `urlopen` raises `HTTPError` for an HTTP error answer and exposes `.code` (435); `timeout=` bounds
  a peer that never answers, which surfaces as `TimeoutError`/`URLError` (436). The preflight's
  `except urllib.error.HTTPError` arm and its `except (URLError, TimeoutError, OSError)` arm are separate
  code paths, and a checker can settle one without the other.

## Run 48 bundles (fresh keys — B1–B320 are history; one agent per bundle)

| Bundle | Rows | Subject | Why tied |
|---|---|---|---|
| B321 | 426, 427, 428 | `lang:python` | one read of `unified_diff`'s emitted line shapes answers all three prefix kinds — removed/added, unchanged, and header |
| B322 | 429, 430, 431 | `lang:python` | one read of the same function's `lineterm`, `n` and sequence-comparison semantics — the three arguments the gate's parse and the comparison's validity rest on. Rule 1 does not let B321 absorb them: a checker holding six rows has more than rule 3's four |
| B323 | 434 | `pkg:pytest@9.0.2` | solo (rule 4) — the fixture's own deletion contract, with nothing tied to answer alongside it |
| B324 | 435, 436 | `lang:python` | one read of `urllib.request`'s error and timeout contract settles both arms of the preflight's catch |
| B325 | 437 | `lang:python` | solo (rule 4) — a runtime inheritance fact with a different subject from B323; rule 1 forbids folding it into the `pkg:pytest` bundle |

**Dispatch shape:** 5 claim bundles ⇒ **5 checker agents**, at most four in flight — the orchestrator runs
them in two waves. Rows 432/433 carry `Check: —` and are not dispatched; they are the gate author's
write-time checks. Nothing exceeds four rows per bundle (rule 3). B1–B320 are history and no key is reused.

## Freshness basis and evidence lookups

Both lookups were performed before any row was written, as the dispatch requires.

**Session-ledger matching (§3d), performed for each of the 12.** No existing row asserts any of them.
`grep` over this ledger finds no row naming `unified_diff`, `difflib`, `splitlines`, `pytest.raises`,
`monkeypatch.delenv`, `tmp_path`, `urlopen`, `HTTPError`, `ProxyHandler` or `subprocess.Popen`. The nearest
neighbours are named and separated rather than quietly reused:

| Row | Cells | Why it is not 426–437 |
|---|---|---|
| 1 | `HOLD` / `none` / `CL:buster_ripper.dump/import-topology` | `dump.py` **imports** `difflib` — a repo import fact, not the library's output convention |
| 6 | `HOLD` / `none` / `CL:buster_ripper.dump#dump_turn/turn-diff-record-form` | `dump.py` **writes** a `difflib.unified_diff` — a repo writer fact; it says nothing about what the library emits |
| 42 | `HOLD` / `none` / `CL:pytest/norecursedirs-semantics` | pytest's `norecursedirs` **replacement** semantics — the gate's run path needs the option's *value* to contain `qa`, not the replacement rule; a different proposition, and no recheck reason fires (pytest 9.0.2 unchanged) |
| 44 | `HOLD` / `none` / `CL:python/path-open-append-semantics` | `Path.open("a")` append semantics — a file-write fact, not a diff or environment fact |
| 111/112 | `HOLD` / `none` | `len(text)` versus UTF-8 byte length — the volume measure, not line identity |
| 169 | `HOLD` / `none` | `tests/test_dump_records.py`'s `read_text()` calls — a repo test read, not `difflib` |
| 279–282 | `HOLD` / `none` | `json.dumps` sort/separators/`TypeError` — the complete-body premise's writer half, already answered; the gate's `json.loads` equality is a different proposition |
| 402–404 | `HOLD` / `UNCLEAR` | `bytes.decode` strict/`replace` — a decode fact, not a diff fact |
| 411 | `HOLD` / `none` / `CL:pytest-asyncio/auto-mode-async-test-collection` | pytest-asyncio's `auto` mode — this module's tests are sync, so the premise is not the gate's |

**Registry `## Claim Index` consulted for these propositions only** (§3e; 209 `Claim:` records and six
`CO:` records read, nothing swept for staleness): **no `CL:` record answers any of the 12.** The index's
`pytest` records are `CL:pytest/norecursedirs-semantics` and the `pytest-asyncio` record; there is no
`difflib`, `unified_diff`, `pytest.raises`, `monkeypatch`, `urlopen`, `HTTPError` or `subprocess` record.
The `CL:python/*` set covers `json`/`hashlib`/`pathlib`/`codecs`/`asyncio`/`unittest.mock` and the
exception hierarchy — none names a diff line kind, a terminator, a context count, a pytest assertion guard or an
environment inheritance. So `i = 0`, `Permanent` stays `—` on all 12, and every row goes to a checker; a
promotable verdict may create the handles. **`i = 0` is the result of both lookups, not a skipped lookup** — the
nearest neighbours are listed above and the registry was read record by record for these propositions.

**`## Composition Index` consulted for the three seams this run opens** (§3c): the index holds six records
(`aiohttp-runner-to-on_cleanup/lifecycle-teardown`, `cli-flag-to-outbound-call/tls-posture-crossing`,
`cli-option-to-dump-dispatch/format-value-crossing`, `client-response-to-relay-consumer/payload-and-header-pairing`,
`serve-script-env-to-argv/env-var-flag-crossing`, `test-double-to-poll-loop/async-witness-eligibility`). None
matches `S21`–`S23`'s member sets, boundary values or dimensions, so no composer is skipped and none carries a
`CO:` handle.

## Collected and deliberately NOT rowed — routes recorded, nothing silent (E7)

- **The external service's own availability at `http://hyperipper.local:30000`** — a live deployment observation,
  not answerable blind from a source root; the gate's `require_upstream` preflight is its arbiter, and the
  `qa-project` §Upstream contract records the endpoint and the fail-closed rule. Route: `T021`'s Yield run (which
  prints the preflight result) and [062](../iterations/062-yield-phase.md)'s `:30000` correction. `S22` carries the
  endpoint as its boundary; no claim row states the deployment's liveness.
- **`urllib.request.ProxyHandler({})` bypassing ambient proxy settings** — the docstring states the design intent
  ("without ambient HTTP proxy settings"), and a wrong reading reds the preflight at once. Route: the QA run; not rowed
  because its failure is loud and its reach is this host's environment.
- **`tmp_path` supplying a distinct directory per test invocation** — a pytest isolation fact; if it were shared, TC-107
  would read TC-108's files and red at once. Route: the QA run.
- **`-o addopts=` overriding the ini `addopts`** — the module docstring's run path depends on it; if it did not
  override, `--cov-fail-under=80` would apply to the QA run and red at once. Route: the QA run and `qa-plan.md`
  §Phase Requirements.
- **Naming a test file explicitly collecting it despite `norecursedirs`** — recorded as verified by `T002`'s note
  (`handoff-context.md` §Notes: "explicit gate selection still collects"); claim 42 is the nearest neighbour and a
  different proposition (replacement versus extension). Loud if wrong (zero tests collected). Route: `T002`'s own gate run.
- **`json.loads` returning a `dict` for the documented body, so `body == expected_body` is a `dict` comparison** — the
  complete-body proposition's other half; shown at once by the unmutated run, and the library's `sort_keys`/`separators`
  neighbours are rows 279–282. Route: Forward's write-time read.
- **`difflib`'s absence of a ready-made changed-line parser** — the gate hand-rolls `diff_changed_lines`; a stdlib
  helper would change the design only by saving five lines, and its failure is loud. Route: Forward's write-time choice.
- **`SequenceMatcher`'s `autojunk` heuristic** — it needs a second sequence of at least 200 elements; the gate's
  renderings are a handful of lines, so no reach. Route: none while `N_TURNS = 3`; a future large-N gate would re-extract.
- **`str.splitlines`' boundary set** — both the gate's expectation and production's diff split with the same method, so
  they decompose identically; a repo-content claim about the renderings, not a library premise.
- **Any `repo:` premise** — `assert_documented_contents`'s complete-body comparison, `changed_lines`' rendering source,
  `record_section`'s public section delimiter, the `self-contained` artifact's `--- turn NNN rendered ---` header, the
  `N + 1`/`2N` layout, and the nonce-bearing body. Repo code and repo state, outside this gate's subject authority
  (E4) and fenced by the dispatch's scope limit; their routes are `T021`'s own `Done when`, [096](../iterations/096-forward-qa.md)
  §Contract Draft, and the closing Yield Red check, all already written.
- **The `self-contained` artifact as a *reader* of production's own writer** — a repo transformation, not a library
  fact; `S21`'s consumer end is the gate, and its producer end is the library's emitted text, not the artifact.

## Run 48 seam delta (written in [seams.md](seams.md) §Run-48; no seam re-verified)

| Seam | Members | State | Movement |
|---|---|---|---|
| S21 | 426–431 | new | `difflib.unified_diff`'s emitted unified-diff text crossing the gate's `splitlines()` + first-character changed-line parser |
| S22 | 435, 436 | new | the external upstream's HTTP answer crossing the harness's `require_upstream` preflight and `assert_upstream_response` status-class oracle |
| S23 | 434, 437 | new | the test process's `DUMP*` environment crossing `monkeypatch.delenv` into the proxy subprocess at `subprocess.Popen` spawn |
| S1–S20 | — | untouched | no member, subject, boundary value, dimension, State, Routing or Permanent cell moved; `S1`–`S15` stay terminal, `S16`–`S19` keep run 44's `recheck`/`absorbed` pairing, `S20` stands as run 46's successor left it |

No claim was checked, no composition performed, no verdict written or copied, no invariant created, nothing promoted, and
no settled row's cells or bundle moved or re-bundled. The ledger's twelve new rows are `new`/`unverified`; five carry
fresh bundle keys and seven are single-task claims outside a seam (rows 432/433 with `Check: —`, the rest seam members).

**Hashes.** `claims.md` before this pass's first write:
`sha256:936d912449fbc16ac13bebe20a23a28163c06db19492cff82f0e3d0df5bea457`; `seams.md` before:
`sha256:de78c8428aae8262f2ff30994563f6d67ee7c15ed5af1306f99cb7b611a28b1d`. Both were re-read after
the writes; the after-values are this pass's `COLLECT-PERSISTED` receipt, because a file cannot record its own
post-write digest. Repository state at this landing: `git-commit:e80ebc3c8304c7ce2105f185dc240e9387a702da`
(`T021` landed at `92d3d4d` and was accepted-local at `e80ebc3`). **No repo file was modified by this pass**; it reads
ledger, installed and test bytes and writes the two session ledgers.

---

# Run 48 — assumption promotion (atomic mode, 2026-10-17)

Promotion pass over the twelve run-48 checker verdicts. Atomic mode only: no seam was read, composed or
retrospectively repaired; `seams.md` and the compositions are out of scope for this pass.

**Verdicts read (12).** `verdicts/claim-426.md`–`claim-436.md` (rows 426–437). Rows 432/433 carry `Check: —` and no
verdict file exists — they are the gate author's write-time checks and are intentionally settled at Forward write time, so no
checker artifact was promoted and no cell moved. `verdicts/claim-437.md` **does not exist**: row 437 stays `new`/`unverified`
and nothing is inferred from the absence (a missing artifact is not a result).

**Cells written (9 rows).** `HOLD`/`none` for every verdict that came back `HOLDS`; `Permanent` receives the `CL:` handle
for every verdict whose second line is exactly `PROMOTION: ELIGIBLE`.

| Row | Verdict on file | State | Routing | Permanent | Why |
|---|---|---|---|---|---|
| 426 | HOLDS / ELIGIBLE | `HOLD` | `none` | `CL:python/unified-diff-content-line-prefixes` | one `lang:python` subject, one proposition; promoted |
| 427 | HOLDS / ELIGIBLE | `HOLD` | `none` | `CL:python/unified-diff-context-line-prefix` | promoted |
| 428 | HOLDS / ELIGIBLE | `HOLD` | `none` | `CL:python/unified-diff-header-line-prefixes` | promoted |
| 429 | HOLDS / ELIGIBLE | `HOLD` | `none` | `CL:python/unified-diff-lineterm-placement` | promoted |
| 430 | HOLDS / ELIGIBLE | `HOLD` | `none` | `CL:python/unified-diff-context-line-count` | promoted |
| 431 | HOLDS / ELIGIBLE | `HOLD` | `none` | `CL:python/unified-diff-line-equality` | promoted |
| 434 | HOLDS / ELIGIBLE | `HOLD` | `none` | `CL:pytest/monkeypatch-delenv-missing-name` | promoted |
| 435 | HOLDS / **INELIGIBLE** | `HOLD` | `none` | — | the verdict rests on a two-file chain with no single exact locator; no permanent record minted |
| 436 | HOLDS / ELIGIBLE | `HOLD` | `none` | `CL:python/urllib-urlopen-timeout-scope` | promoted |

**Promotion is publication, not approval.** The eight `CL:` handles are graph-inert descriptive evidence: they record what
the sources established and grant no design authority. No `Invariant:`, `Override:`, `Entity:` or `Scenario:` record was written,
edited or inferred, and no invariant was promoted or deprecated.

**Registry receipts (read before this pass's first registry write).** `invariants/buster_ripper/assumption-claims.md`
`sha256:a47a62d2ca048a1198b0b49949ad67f5208624fcc8b9fe362d0242990e93f907`; `invariants/index.md`
`sha256:fac33e0960606d9bd92edc86812add4074b68108fd677de4e08e0403e708a57e`.

**Registry posture after the writes.** `codify knowledge sync-registry --apply` converged the tracked companion
(`invariants/generated-index.md`). `codify knowledge check --profile strict` reports **four pre-existing diagnostics, none produced
by this pass**: two `missing-version` warnings on `buster_ripper/dump/invariants.md` and `buster_ripper/server/invariants.md`, and
two `curated-index-generated-section` errors for the `## Handle Index` / `## Entity Index` headings in the human-owned
`invariants/index.md`. Those files belong to the invariant/entity lane and were deliberately left untouched; repairing them is a
registry-format migration (`codify knowledge migrate-registry`) that would promote invariant records, which this pass may not do.
`buster_ripper/assumption-claims.md` itself parses clean: the eight new fenced `Claim:` blocks are current v1 registry records
with zero diagnostics; the 209 pre-existing unfenced claim blocks and six unfenced composition blocks remain the engine's
out-of-grammar census (`OutOfGrammarCount`: 209 `Claim` at line 17, 6 `Composition` at line 706), exactly as the current grammar
disposes of unfenced evidence-identity lines.

## Run 48 — assumption promotion, pass 2 (atomic mode, row 437, 2026-10-18)

The first promotion pass ran before `verdicts/claim-437.md` existed; it recorded that absence as "no result" and inferred nothing.
The verdict has since landed (`VERDICT: HOLDS`, `PROMOTION: ELIGIBLE`, checked `2026-10-18T00:24:36Z`), so this pass promotes it.

**Cells written (1 row).** `HOLD` / `none`; `Permanent` receives `CL:python/subprocess-popen-environment-inheritance` — one
`lang:python` subject, one proposition, and complete structured metadata (`Source`, `Source Version`, the
`sha256:bcbee8309e8e…` `Source Fingerprint`, three `Invalidates-On` clauses, `Checked-At`). Row 425 (`accepted-consequence`) and
rows 432/433 (`Check: —`, settled at Forward write time) are outside atomic scope and untouched.

**Registry.** New record in `invariants/buster_ripper/assumption-claims.md` — no existing `CL:` record asserted this proposition (the
file's `subprocess`/`Popen` mentions are repo and test-tree records, not a library environment fact). `invariants/index.md`'s
generated Claim Index was rebuilt by `speeckle/scripts/rebuild_index.py`; `codify knowledge sync-registry --apply` reports the tracked
companion current, and `codify knowledge check --profile strict` reports the same five pre-existing diagnostics as the first pass
(two `missing-version` warnings and two `curated-index-generated-section` errors, plus the `index.md` file-level `missing-version`
warning) — none produced by this pass.

**Promotion is publication, not approval.** The record is graph-inert descriptive evidence. No `Invariant:`, `Override:`, `Entity:` or
`Scenario:` record was written, edited or inferred, and no invariant was promoted or deprecated.

---

# Run 50 — `MS-001` boundary recheck (2026-10-18)

Dispatch: **phase-boundary assumption recheck**, `MS-001` after `T021`, `T021.001`, `T021.002`, `T022` and
`T022.001`; third-party subjects only. Nothing was checked, composed, promoted or re-verified; this pass reads the
plan rows, the two changed seams, their compositions and the gate/harness bytes those seams cite, then writes this ledger.

**The four narrow sets, and what each yielded.**

1. **Tasks not yet extracted** — `T022` and `T022.001` have no row in this ledger (`grep` for `T022` returns none).
2. **Checked rows with a listed recheck reason** — `S21`/`S22`'s member rows keep their verdicts; the seams themselves
   are `recheck` because their transformation and member set changed (see [seams.md](seams.md) §Run-50).
3. **Tasks that did not exist at the last extraction** — `T022`/`T022.001`; both are new since run 48.
4. **Seams whose member, transformation, consumer, fallback, precedence or dimension changed** — `S21` and `S22` only;
   `S18` is unchanged (record condition below).

**What moved since run 48.** Production bytes are unchanged: `buster_ripper/dump.py` is still
`sha256:536ac105…`. The moved bytes are QA-gate and test-support files: `tests/qa/test_qap_ms001.py`
`fc60661c…` → `0b3b052c…` (`T021.001` repair, its CE-1 guard, and the gate-model alias), `tests/qa/harness.py`
`ed105893…` → `0ac81b37…` (`T021.002` repair and the alias), `tests/qa/test_qap_ms002.py` `e50c6732…` → `63167cd8…`
(alias), `.agents/skills/qa-project/SKILL.md` `f5cffb18…` → `10e9a18d…` (alias declaration), and
`tests/test_stream_relay_commit.py` `43d553a6…` (one comment word, `T022.001`). Session bookkeeping moved with them.

**Inherited, not re-extracted (evidence reuse).** These rows keep their verdicts: each row's consulted bytes still match
and no named `Invalidates On` clause fired. A repo edit does not fire a `lang:` or `pkg:` row (§2 subject typing), and no
interpreter, package or documented source moved. This run's `i = 15`.

| Rows | State retained | Why no recheck reason |
|---|---|---|
| 426–431 (`S21` members) | `HOLD` / `absorbed` / `CL:` handles | `difflib` bytes unchanged; the claims are `lang:python` propositions and the gate's parser edit does not alter what the library emits |
| 435, 436 (`S22` members) | `HOLD` / `absorbed`; 436's `CL:python/urllib-urlopen-timeout-scope` | `urllib`/`socket` bytes unchanged; the harness edit changes the caller's opener construction, not the error or timeout contract |
| 434, 437 (`S23` members) | `HOLD` / `none` / `CL:` handles | The autouse fixture and the `subprocess.Popen` call are byte-identical; `T022.001`'s comment word does not touch them |
| 279–282 (`json.dumps`/`loads`), 393 (`client_max_size`) | `HOLD` / `none` | The gates exercised these rows, but exercise is not a listed recheck reason, and their consulted `json`/`aiohttp` bytes did not move |

**Not rechecked, with reasons.** `432`/`433` are `new`/`unverified` write-time checks (`Check: —`), not `HOLD` rows;
the gate's `pytest.raises`/`match=` use is unchanged. The wider `HOLD` set a full gate run touches (streaming relay, response
round trip, header policy) is not re-read here: a green run is exercise, not an invalidation, and no row's `Invalidates On` names
the gate or harness files.

**`T022` / `T022.001` — zero eligible rows.** `T022` is the step-6a drain of `quality-backlog.md`; `T022.001` closes its own
fixed line and restates the counts. Every premise they take for granted is repo file state (`quality-backlog.md:14`/`:26`,
`101-01-forward.md:32`/`:80`, `plan-summary.md`'s `T022` row, `handoff-context.md` §Current Status,
`tests/test_stream_relay_commit.py:51`). Under this session's standing scope ruling those are not gate subjects, no `repo:`
row is written, and neither task has a library, runtime or external-service premise. The required task delta is recorded here and
produces no row.

## Run 50 bundles (fresh keys — B1–B325 are history; one agent per bundle)

| Bundle | Rows | Subject | Why tied |
|---|---|---|---|
| B326 | 438, 439, 440 | `lang:python` | one read of `difflib.unified_diff`'s yield shape settles the header order, the hunk-header's closing `@@`, and the content-line terminator the repaired `_content_lines` consumes |
| B327 | 441, 442 | `lang:python` | one read of `urllib.request`'s handler installation and `ProxyHandler` mapping settles both halves of the repaired `Proxy.post` opener |

**Dispatch shape:** 2 claim bundles ⇒ **2 checker agents**, at most four in flight — one wave. Nothing exceeds four rows per
bundle (rule 3). B1–B325 are history and no key is reused.

## Freshness basis and evidence lookups

**Session-ledger matching (§3d), performed for each of the five new rows.** No existing row asserts any of them. Rows 426–431
state `unified_diff`'s content/context/header prefixes, its `lineterm` placement, its `n` default and its equality basis, but
none states the yield order, the closing `@@`, or the content-line terminator. Rows 435/436 state `urlopen`'s error and timeout
contracts, but neither states `build_opener`'s handler substitution or an empty proxy map. The nearest neighbours are therefore
named and separated rather than quietly reused, and rows 426–431/435/436 keep their cells.

**Registry `## Claim Index` consulted for these propositions only** (§3e; the index was read for `unified_diff`, `difflib`,
`build_opener`, `ProxyHandler`, `urlopen`, `HTTPError` and `splitlines`, not swept for staleness): **no `CL:` record answers any
of the five.** The `CL:python/*` set covers the diff line kinds, `lineterm`, `n`, equality and `urlopen`'s timeout; none names
the yield order, the closing `@@`, the content-line terminator, `build_opener`'s handler substitution or an empty proxy map. So
`Permanent` stays `—` on all five and every row goes to a checker; a promotable verdict may create the handles.

**Composition Index.** No `CO:` record matches `S21`'s or `S22`'s new ordered member set, boundary value or dimensions, so no
composer is skipped and neither seam carries a handle.

## Record conditions (stated, not absorbed)

- **`S18` is unchanged and stands.** Its recomposition was made on the CR4 bytes; its `Invalidates On` **5** named the
  `count_tokens` reordering (`count_tokens.py:79-95` keying relative to normalization). That clause was re-read at run 47 and
  stands on today's bytes: `count_tokens.py` is `c6d6f924…`, the digest the run-47 record cites, and `messages.py` is
  `64be411b…`, also unchanged. This pass re-read neither the record nor its members for freshness; it states the prior reading.
- **A stale registry record is owned by the assumptions/registry lane, not this pass.** The registry carries
  `CL:buster_ripper.routes.chat#_make_chat_request/nonstream-json-parsing` (`invariants/index.md:169`) whose statement says
  `_make_chat_request` calls `resp.json()` after the stream branch returns on the nonstream path. That call no longer exists: the
  function now reads `await resp.read()` and delegates to `_chat_answer` (`buster_ripper/routes/chat.py:134-153`), and the record's
  fingerprint is `unavailable`. The final Codex review flagged it as a registry-refresh item. This pass neither reuses nor repairs
  it; refresh belongs to the assumptions/registry lane.

**Hashes.** `claims.md` before this pass's first write:
`sha256:e4c252aff2c4c59da815cd2725cdcdc356c9bff692640a696731ef776d8a3a3b`; `seams.md` before:
`sha256:132f83ed95df2ab6a477fdb6e43ac232f165e5e828457be06cf14150e31e410e`. Both were re-read after the writes; the
after-values are this pass's `COLLECT-PERSISTED` receipt, because a file cannot record its own post-write digest.
Repository state at this pass: `git-commit:4bdfafd` (`Checkpoint 245`). **No repo file was modified by this pass**; it reads
the plan, the two changed seams, their compositions and the gate/harness bytes, and writes the two session ledgers.



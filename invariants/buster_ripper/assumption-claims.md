Speeckle-Version: 1

# Claims — assumption evidence (`buster_ripper`)

Descriptive evidence promoted mechanically by assumption-promote from the blind checker verdicts of
session `02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573` (extraction run 1, checked 2026-09-19;
run 48's `T021` gate surface, checked 2026-10-17).
Created 2026-09-19; the registry held no `Claim:` records before this file. These records carry no
design authority — they record what sources established, never what is approved.

The `Claim:` blocks fenced as ```` ```text ```` are current Speeckle v1 registry records. The unfenced
`Claim:`/`Composition:` blocks above them predate the file-level version declaration: the current grammar
reads them as its out-of-grammar census, never as actionable records, and this pass did not migrate them.

---

Claim: CL:buster_ripper.dump/import-topology
Statement: `dump.py` imports `json` and `difflib`
Subject: repo:buster_ripper.dump
Verdict: HOLDS
Actual: `buster_ripper/dump.py` imports `difflib` (line 4) and `json` (line 5); both are live imports, used in the file body (`json.dumps` at :34/:50/:82, `difflib.unified_diff` at :89).
Source: buster_ripper/dump.py:3-5 (import block)
Version: unavailable
Fingerprint: sha256:243b21434c14e6ea8144329fe1d0ac4e0c499ed9a7e6e09bdcac5e2417a56b1d
Invalidates-On: buster_ripper/dump.py's import block changes (the `json` or `difflib` import is added, removed or replaced)
Checked-At: 2026-09-19
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-1.md

---

Claim: CL:pyproject.toml/runtime-dependency-set
Statement: `pyproject.toml` declares `aiohttp` and `click` as the only runtime dependencies
Subject: repo:pyproject.toml
Verdict: HOLDS
Actual: The `[project]` `dependencies` key lists exactly `aiohttp` and `click`, with no version constraints; the only other dependency declaration is the `[dependency-groups]` `dev` group, and no `[project.optional-dependencies]` or other runtime-dependency mechanism exists in the file.
Source: pyproject.toml lines 1-8 ([project] `dependencies`), lines 32-39 ([dependency-groups] dev)
Version: unavailable (project version 0.1.0)
Fingerprint: sha256:e1b7035805a8e946b71f947026e86e47e328f7e0b9ed9568b8cdb648e059079e
Invalidates-On: pyproject.toml's `[project]` `dependencies` key changes (an entry is added, removed or constrained)
Invalidates-On: pyproject.toml gains another runtime-dependency mechanism (e.g. `[project.optional-dependencies]`, a Poetry/PDM section, or a build-backend requirement consuming deps at runtime)
Checked-At: 2026-09-19
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-2.md

---

Claim: CL:buster_ripper.dump#dump_turn/write-count-per-turn
Statement: `dump_turn` performs two write-bearing file operations per intercepted turn: the turn-file write and the `index.txt` append
Subject: repo:buster_ripper.dump#dump_turn
Verdict: HOLDS
Actual: Each `dump_turn` call performs exactly two content-writing file operations — one `Path.write_text` for the turn file (`turn_000.json` on turn 0 at dump.py:81-82, `turn_{turn:03d}.diff` on later turns at :94-95) and one append-mode write of the summary line to `index.txt` (:104-106); the only other filesystem call is the first-turn `session_dir.mkdir(parents=True, exist_ok=True)` (:72), which creates a directory, not file content.
Source: buster_ripper/dump.py `dump_turn` — turn-file writes at :81-82 and :94-95, index append at :104-106, mkdir at :72
Version: unavailable
Fingerprint: sha256:243b21434c14e6ea8144329fe1d0ac4e0c499ed9a7e6e09bdcac5e2417a56b1d
Invalidates-On: dump_turn's per-turn write operations change in number or kind (a write is added, removed or merged, e.g. buffering turn files and flushing later, or writing the index inside the turn-file branch)
Invalidates-On: index.txt writing moves out of dump_turn into a caller or a separate flush routine
Checked-At: 2026-09-19
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-4.md

---

Claim: CL:buster_ripper.dump#dump_turn/turn-zero-record-form
Statement: `dump.py` writes turn 0's record as `json.dumps(body, indent=2, ensure_ascii=False)` to `turn_000.json`
Subject: repo:buster_ripper.dump#dump_turn
Verdict: HOLDS
Actual: On turn 0 (`state.turn == 0`), `dump_turn` writes `json.dumps(body, indent=2, ensure_ascii=False)` — verbatim call, same arguments — to `{session_dir}/turn_000.json` via `out.write_text(..., encoding="utf-8")` (dump.py:80-82); the raw request body is stored pretty-printed, with no rendering or diffing applied to the stored bytes.
Source: buster_ripper/dump.py `dump_turn` lines 80-82
Version: unavailable
Fingerprint: sha256:243b21434c14e6ea8144329fe1d0ac4e0c499ed9a7e6e09bdcac5e2417a56b1d
Invalidates-On: the turn-0 write branch changes (the `json.dumps` call arguments or the `turn_000.json` filename in buster_ripper/dump.py)
Invalidates-On: turn 0's record switches to the `_render_body`/unified-diff path used for later turns
Checked-At: 2026-09-19
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-5.md

---

Claim: CL:buster_ripper.dump#dump_turn/turn-diff-record-form
Statement: `dump.py` writes each turn ≥ 1's record as the `difflib.unified_diff` of `_render_body(body)` against the previous turn's rendering to `turn_{turn:03d}.diff`
Subject: repo:buster_ripper.dump#dump_turn
Verdict: HOLDS
Actual: For every turn ≥ 1, `dump_turn` renders the current body with `_render_body(body)` (:77), splits both it and `state.prev_text` (the previous turn's rendering, assigned at :108) into lines, computes `difflib.unified_diff(prev_lines, curr_lines, fromfile=f"turn_{turn - 1:03d}", tofile=f"turn_{turn:03d}", lineterm="")` (:89-93), and writes the joined diff lines to `{session_dir}/turn_{turn:03d}.diff` via `write_text` (:94-95).
Source: buster_ripper/dump.py `dump_turn` — rendering at :77, `difflib.unified_diff` at :89-93, output path and write at :94-95, `state.prev_text` update at :108
Version: unavailable
Fingerprint: sha256:243b21434c14e6ea8144329fe1d0ac4e0c499ed9a7e6e09bdcac5e2417a56b1d
Invalidates-On: the turn-≥1 diff branch changes (the `difflib.unified_diff` call, its input renderings, or the `turn_{turn:03d}.diff` filename in buster_ripper/dump.py)
Invalidates-On: `state.prev_text` stops being the previous turn's `_render_body` output (e.g. raw JSON or a compact form)
Checked-At: 2026-09-19
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-6.md

---

Claim: CL:buster_ripper.dump#dump_turn/index-summary-form
Statement: `dump.py` appends one summary line per turn to `index.txt` — the `FULL` form on turn 0, the `+added/-removed` form on later turns
Subject: repo:buster_ripper.dump#dump_turn
Verdict: HOLDS
Actual: Every `dump_turn` call builds exactly one summary string and appends it (`f.write(summary + "\n")`) to `{session_dir}/index.txt` opened in append mode (:104-106); turn 0's summary is the `FULL` form (`turn 000  {now}  FULL  {n} msgs  {n} tools  {kb} KB`, :84) and turns ≥ 1 use the `+{added}/-{removed} lines` form (:98-101), where `added`/`removed` count `+`/`-` diff lines excluding the `+++`/`---` headers.
Source: buster_ripper/dump.py `dump_turn` — FULL summary at :84, +/- summary at :98-101, append at :104-106
Version: unavailable
Fingerprint: sha256:243b21434c14e6ea8144329fe1d0ac4e0c499ed9a7e6e09bdcac5e2417a56b1d
Invalidates-On: either summary format string, the `FULL` marker, or the `+{added}/-{removed}` form changes in buster_ripper/dump.py
Invalidates-On: the index line is written more or less than once per `dump_turn` call, or the append target changes away from `index.txt`
Checked-At: 2026-09-19
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-7.md

---

Claim: CL:buster_ripper.dump#dump_turn/state-advance-timing
Statement: `dump.py` advances `state.prev_text` and `state.turn` after processing each turn
Subject: repo:buster_ripper.dump#dump_turn
Verdict: HOLDS
Actual: After the selected writer or record-appender returns, `dump_turn` assigns the rendered current body to `state.prev_text` and increments `state.turn`. If processing raises, those lines are not reached.
Source: buster_ripper/dump.py:83-98
Version: unavailable
Fingerprint: sha256:c8f093c89b61c355ac182feb070cd90b84924ca0d187332117d76b5c766ad9b9
Invalidates-On: Changes to `dump_turn`'s state update or processing order in `buster_ripper/dump.py`.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-8.md

---

Claim: CL:buster_ripper.dump#_render_body/section-surface
Statement: `_render_body` emits only `[tools]`, `[system]` and `[message i / role]` sections — every other top-level request-body key is absent from its output
Subject: repo:buster_ripper.dump#_render_body
Verdict: HOLDS
Actual: The renderer explicitly reads only `tools`, `system`, and `messages` and constructs sections only for these fields; it does not render other top-level keys as fields. Their names or values can nevertheless appear incidentally inside a tool definition or message text, so “absent” describes field rendering, not a guarantee that arbitrary strings never occur.
Source: `buster_ripper/dump.py:49-80` (`_render_body`)
Version: unavailable
Fingerprint: sha256:c8f093c89b61c355ac182feb070cd90b84924ca0d187332117d76b5c766ad9b9
Invalidates-On: Change to `buster_ripper/dump.py`'s `_render_body` implementation or its input-field handling.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-9.md

---

Claim: CL:buster_ripper.dump#_render_body/message-section-mapping
Statement: `_render_body`'s output carries the request's conversation as one `[message i / role]` section per message
Subject: repo:buster_ripper.dump#_render_body
Verdict: HOLDS
Actual: It enumerates `body.get("messages") or []` and appends exactly one `[message {i} / {role}]` section for each message, rendering its content as text. The separate `system` field is rendered in `[system]`, not as a message section; structured content may lose its original representation.
Source: `buster_ripper/dump.py:67-80` (`_render_body`)
Version: unavailable
Fingerprint: sha256:c8f093c89b61c355ac182feb070cd90b84924ca0d187332117d76b5c766ad9b9
Invalidates-On: Change to `buster_ripper/dump.py`'s `_render_body` message loop or its input-field handling.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-10.md

---

Claim: CL:buster_ripper.session#session_id/key-derivation-input
Statement: `session_id(body)` derives the session key from the body's system-prompt content alone
Subject: repo:buster_ripper.session#session_id
Verdict: HOLDS
Actual: `session_id(body)` reads exactly one input — `body.get("system", "")` — flattens a list-valued `system` by joining the `text` fields of its dict blocks, and returns the first 12 hex chars of `hashlib.sha256(system.encode())`; no other body field, no external state and no clock enter the hash. Refinements, not contradictions: non-dict blocks and non-`text` fields inside a list `system` are dropped from the digest input, and an absent `system` hashes the empty string to a constant key.
Source: buster_ripper/session.py:93-98 (`def session_id`); callers buster_ripper/dump.py:64, buster_ripper/routes/count_tokens.py:35, buster_ripper/routes/messages.py:35
Version: unavailable
Fingerprint: sha256:67a66b8aa733b31e982be4922350a7996ebdddb0fe5d391a6129a1cf342a9dff
Invalidates-On: session_id mixes any other body field, request metadata or external state into the digest input
Invalidates-On: session_id changes how a list-valued `system` is flattened (e.g. hashing raw blocks instead of joined `text` fields), or changes the digest/truncation (algorithm or `[:12]`)
Checked-At: 2026-09-19
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-11.md

---

Claim: CL:buster_ripper.dump#SessionState/per-session-keying
Statement: `dump.py`'s `SessionState` is the per-session dump state, keyed by session id in the module-level `_sessions` dict, holding `turn`/`prev_text`/`session_dir`
Subject: repo:buster_ripper.dump#SessionState
Verdict: HOLDS
Actual: `SessionState` declares those three fields; `_sessions` maps session-id strings to instances, and `_lookup_or_create_state` retrieves or creates one by `session_id(body)`.
Source: `buster_ripper/dump.py:32-46,136-147`
Version: unavailable
Fingerprint: sha256:c8f093c89b61c355ac182feb070cd90b84924ca0d187332117d76b5c766ad9b9
Invalidates-On: Changes to `buster_ripper/dump.py`'s `SessionState` declaration or `_sessions` lookup/creation.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-12.md

---

Claim: CL:buster_ripper.routes.messages#handle_messages/dump-failure-isolation
Statement: `routes/messages.py`'s `handle_messages` wraps its `dump_turn` call in a swallow-all `try/except Exception` that logs and never fails the request
Subject: repo:buster_ripper.routes.messages#handle_messages
Verdict: HOLDS
Actual: The sole `dump_turn(body)` call site sits inside `if config.DUMP_DIR is not None: try: dump_turn(body) / except Exception: request.app["log"].exception("dump failed")` (routes/messages.py:55-59); the clause logs the failure with traceback and falls through — no re-raise, no error response — so a dump failure never fails the request. Precision notes: the wrap is gated on `config.DUMP_DIR is not None`, but whenever the call executes it executes inside the `try`; and the clause is `except Exception`, not `except BaseException`, so `asyncio.CancelledError`/`KeyboardInterrupt` would still propagate — the claim names the exact construct `try/except Exception`, which matches verbatim.
Source: buster_ripper/routes/messages.py:55-59 (call site; `handle_messages` defined at :24; sole call site — one import at :12, one call at :57)
Version: unavailable
Fingerprint: sha256:9c234c4f183aa03a80ab82356a6fdfaad0220f3bef69edf3bab5789ef808277d
Invalidates-On: the `try/except Exception` block at routes/messages.py:55-59 is removed, narrowed to specific exception types, or changed to return an error response or re-raise
Invalidates-On: the `dump_turn` call moves out of `handle_messages`, an additional unwrapped call site is added to this file, or the `config.DUMP_DIR` gating changes so the call executes outside the `try`
Checked-At: 2026-09-19T15:02:45Z
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-13.md

---

Claim: CL:buster_ripper/signal-handler-ownership
Statement: The `buster_ripper` package installs no SIGINT/SIGTERM handler of its own
Subject: repo:buster_ripper
Verdict: HOLDS
Actual: Its CLI delegates signal handling to `aiohttp.web.run_app`; the package does not register its own SIGINT/SIGTERM handlers. Its application registers an aiohttp cleanup callback, which is not a signal handler.
Source: `buster_ripper/main.py:157-160` (CLI hands control to `web.run_app`; inspection of production modules finds no signal registration).
Version: unavailable
Fingerprint: sha256:cbe2802d474abe21ada5bb2e94c09cce15237ea53d526c6d31fed25207c950d1
Invalidates-On: Change to `buster_ripper/main.py` or another production module that adds SIGINT/SIGTERM registration.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-14.md

---

Claim: CL:buster_ripper.server#on_cleanup/shutdown-dump-flush
Statement: `buster_ripper`'s `on_cleanup` performs no dump flush during shutdown
Subject: repo:buster_ripper.server#on_cleanup
Verdict: HOLDS
Actual: `on_cleanup` (the only one in the package, buster_ripper/server.py:60-72) does exactly two things on shutdown — cancels and awaits the `kv_monitor_task` (swallowing `CancelledError`) and closes the shared `aiohttp.ClientSession` — and makes no call to anything dump-related; the negative holds doubly: there is no flush call anywhere in `on_cleanup`, and no flush function exists in the package to call (`dump_turn` writes each turn's file synchronously per request from routes/messages.py:57; no pending buffer, no `atexit`/signal hook, and no dump interaction inside the cancelled coroutine's path).
Source: buster_ripper/server.py:60-72 (`on_cleanup` body); corroborated by buster_ripper/dump.py (no flush function), buster_ripper/compaction.py:28-57, buster_ripper/routes/messages.py:57, buster_ripper/main.py:88-89
Version: unavailable
Fingerprint: sha256:e008d3c971db787ebd1eb890544c6d73f0479bb97d0060ee64d4f77cf150b824
Fingerprint: sha256:243b21434c14e6ea8144329fe1d0ac4e0c499ed9a7e6e09bdcac5e2417a56b1d
Fingerprint: sha256:ac9d42143c98694a0b6672dd753afcef537fc5e03ec747b4bb3c1208e23535e7
Fingerprint: sha256:9c234c4f183aa03a80ab82356a6fdfaad0220f3bef69edf3bab5789ef808277d
Fingerprint: sha256:b4794796d28ad3ab9ce5f2e0e03dddcfa882178299c7b969751697993cb76089
Invalidates-On: on_cleanup adds a call to dump/flush logic (server.py fingerprint change), or a second `on_cleanup`/alternate shutdown hook registering dump flushing appears
Invalidates-On: a flush/pending-buffer function is added to buster_ripper/dump.py that `on_cleanup` or its callees invoke (dump.py fingerprint change)
Invalidates-On: a `finally`/cancellation-path dump flush is added to `compaction.poll_kv_cache` (compaction.py fingerprint change)
Checked-At: 2026-09-19
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-15.md

---

Claim: CL:buster_ripper.server#spawn_kv_cache_monitor/background-task-lifecycle
Statement: `server.py`'s `spawn_kv_cache_monitor`/`on_cleanup` pair establishes the background-task lifecycle: `create_task` at startup, an `app`-state key, cancel-and-await at cleanup
Subject: repo:buster_ripper.server#spawn_kv_cache_monitor
Verdict: HOLDS
Actual: `on_startup` calls `spawn_kv_cache_monitor`, which creates a task and stores it under `app["kv_monitor_task"]`. `on_cleanup` passes that key to `_cancel_background_task`, which cancels and awaits the task, swallowing `CancelledError`.
Source: `buster_ripper/server.py:25-34,55-91,102-106` (production source)
Version: unavailable
Fingerprint: sha256:fbd579a7995d0f4a2ae8a6d2994bb9e5fda344c1a900ec581caee38f599647f8
Invalidates-On: Changes to `buster_ripper/server.py`'s startup registration, monitor spawning, task storage, cleanup registration, or cancellation helper.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-16.md

---

Claim: CL:buster_ripper/dump-buffering-ownership
Statement: No symbol under `buster_ripper/` or `tests/` owns a message buffering, batching or flush role for the dump
Subject: repo:buster_ripper
Verdict: HOLDS
Actual: The dump path is write-through with no intermediate symbol holding messages — routes/messages.py:55-57 calls `dump_turn(body)` synchronously once per request and `dump_turn` (dump.py:61-109) renders and writes immediately (:82/:95/:105-106), one call per request, nothing held for later; the only dump-path state is `SessionState`/`_sessions` (dump.py:16-24), where `prev_text` stores the previous turn's rendering as a diff baseline, not a buffer of messages awaiting a write. A vocabulary sweep (`buffer|batch|flush|deque|queue|pending|accumul|drain`, plus FileHandler/aiofiles/fsync/tempfile) over all 16 package and 10 test files found only incidental matches (a docstring word, an HTTP header literal, an in-memory stats-accumulation test name), no symbol with a dump buffering/batching/flush role, and no `.flush()` call in either scope; tests/ contains no reference to the dump module at all.
Source: buster_ripper/dump.py:61-109 (render-and-write-immediately) and buster_ripper/routes/messages.py:55-57 (sole synchronous call site); survey scope: all 16 .py files under buster_ripper/ and all 10 .py files under tests/
Version: unavailable
Fingerprint: sha256:243b21434c14e6ea8144329fe1d0ac4e0c499ed9a7e6e09bdcac5e2417a56b1d
Fingerprint: sha256:9c234c4f183aa03a80ab82356a6fdfaad0220f3bef69edf3bab5789ef808277d
Fingerprint: sha256:60aebdf2191e7785586f3d1aa8fe0663136eadbeb76be53e072fefb4d8bf6381
Fingerprint: sha256:291263720cb7ae79b9481b88abfb090556db712d9fb5ffd61301a7fa4180f620
Invalidates-On: any symbol added under buster_ripper/ or tests/ holding messages for deferred dump writes (queue/deque/list accumulation feeding the dump, a pending-write state, a batcher), or dump_turn changed to defer or batch writes
Invalidates-On: the routes/messages.py:55-57 call site moves behind any buffering/queueing layer, or any .flush()-driven dump path is introduced in either scope
Checked-At: 2026-09-19T14:59:49Z
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-17.md

---

Claim: CL:buster_ripper/collapsed-artifact-ownership
Statement: No symbol under `buster_ripper/` owns the collapsed-artifact role — no filename constant or per-session append target for a collapsed-format dump artifact (`<format>.log`, written once per flush) exists. The documented layout's `index.txt` append (`dump.py:104-106`) is the existing per-session append target and is not this role
Subject: repo:buster_ripper
Verdict: HOLDS
Actual: All three halves verify against the source: (1) no `<format>.log` filename constant — the string `.log` has zero occurrences under `buster_ripper/`, artifact filenames are inline literals (`"turn_000.json"` dump.py:81, `f"turn_{turn:03d}.diff"` :94, `"index.txt"` :104), the only dump-related config global is `DUMP_DIR` (config.py:16, a directory root), and the only "log" symbols are logging logger objects that name no artifact; (2) no once-per-flush append target — the word "flush" has zero occurrences under `buster_ripper/` (no function, constant or concept implements a flush), and the package's only append-mode open is dump.py:105-106, firing once per `dump_turn` call, itself called once per intercepted request (routes/messages.py:55-57); nothing else appends; (3) the positive half — dump.py:104 computes `index = state.session_dir / "index.txt"` with `session_dir = config.DUMP_DIR / sid` (:71), one directory per session, then :105-106 opens it in append mode and writes one summary line per turn — exactly the per-session artifact the documented layout describes (README.md:146-152, reading workflow :154-159).
Source: buster_ripper/dump.py:104-106 (per-session append), :71 (`session_dir = config.DUMP_DIR / sid`), :81/:94/:104 (inline filename literals); buster_ripper/config.py:16; negative sweeps (`\.log`, `flush`, collapse-named symbols) over buster_ripper/*.py; buster_ripper/routes/messages.py:55-57; README.md:146-159 (documented output layout)
Version: unavailable
Fingerprint: sha256:243b21434c14e6ea8144329fe1d0ac4e0c499ed9a7e6e09bdcac5e2417a56b1d
Fingerprint: sha256:a8d597c72b462d9df5009906db82e1096187889de21efb8c51745a1728b605c3
Invalidates-On: any `.log` artifact filename constant, or any flush-named symbol that is an append target, appears under `buster_ripper/`
Invalidates-On: the `index.txt` append at dump.py:104-106 moves or changes mode, the per-session dir computation at dump.py:71 changes, README.md's Output layout section (the `index.txt` entry) is removed or changed, or any byte of buster_ripper/dump.py or README.md changes (fingerprint mismatch — re-run the sweeps)
Checked-At: 2026-09-19T17:32:21Z
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-47.md

---

Claim: CL:buster_ripper/dump-file-readers
Statement: No production code under `buster_ripper/` reads `index.txt`, `turn_000.json` or `turn_*.diff`
Subject: repo:buster_ripper
Verdict: HOLDS
Actual: The production dump module writes these three artifact types but does not read their contents: it writes the turn JSON and diffs via `write_text` and appends index lines using `open("a")`. Its writable-target guard inspects filesystem metadata (`is_symlink`, `exists`, `is_file`), not file contents; a search of production Python source found no other references to these artifact names or readers of them.
Source: `buster_ripper/dump.py:150-186,296-304` (production source; package-wide source search for artifact names and file reads)
Version: unavailable
Fingerprint: sha256:c8f093c89b61c355ac182feb070cd90b84924ca0d187332117d76b5c766ad9b9
Invalidates-On: Changes to `buster_ripper/dump.py` artifact handling or writable-target guard.
Invalidates-On: Addition of a production artifact reader elsewhere under `buster_ripper/`.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-106.md

---

Claim: CL:buster_ripper/dump-format-selection
Statement: No symbol in the `buster_ripper` package selects a dump record format — no config global and no CLI flag carries that role
Subject: repo:buster_ripper
Verdict: HOLDS
Actual: The dump record format is hardcoded inline in `buster_ripper/dump.py:dump_turn` (turn 0 writes `turn_000.json` via `json.dumps(body, indent=2, ensure_ascii=False)`, every later turn writes `turn_{N:03d}.diff` via `difflib.unified_diff`, `index.txt` accumulates a summary line) and no symbol selects between these — the switch is the internal `turn == 0` condition on per-session state; the only dump-related config global is `config.DUMP_DIR` (a root path / mode switch) and the only dump-related CLI flag is `--dump-dir PATH` (the same role) — no `DUMP_FORMAT`-style global, no `--dump-format`-style option, no env-var read (`os.environ`/`getenv` absent from the package), and no format/encoding parameter or constant exists anywhere in the package. Planning/registry documents reference a planned `--dump-format` flag, but docs cannot establish behaviour; the production source is the ground.
Source: buster_ripper/dump.py:61-95 (dump_turn's hardcoded format branch); buster_ripper/config.py:16 (sole dump global), buster_ripper/main.py:25 (sole dump CLI option), buster_ripper/routes/messages.py:55 (guards on `config.DUMP_DIR is not None`)
Version: unavailable
Fingerprint: sha256:243b21434c14e6ea8144329fe1d0ac4e0c499ed9a7e6e09bdcac5e2417a56b1d
Fingerprint: sha256:14ac40180f081d8e7caeb280d9a9ab194cbd06c93e3958ebf9bef362ca7b7aab
Fingerprint: sha256:b4794796d28ad3ab9ce5f2e0e03dddcfa882178299c7b969751697993cb76089
Invalidates-On: any `DUMP_FORMAT`-style global or dump-format constant/enum appears in `buster_ripper/config.py` or any other package module, or any `--dump-format`-style click option appears in `buster_ripper/main.py`
Invalidates-On: `dump_turn` gains a format/encoding parameter, or the turn-0/turn-N branches are replaced by a symbol-selected writer
Checked-At: 2026-09-19
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-21.md

---

Claim: CL:buster_ripper.main/numeric-config-cli-wiring
Statement: No numeric `config` global has a CLI flag today (`KV_POLL_INTERVAL` included)
Subject: repo:buster_ripper.main
Verdict: DOES-NOT-HOLD
Actual: Eight numeric globals in `buster_ripper/config.py` have CLI flags wired in `buster_ripper/main.py`, each assigned straight onto `config.<NAME>` in `main()` — `MAX_MODEL_LEN` ← `--max-model-len` (config.py:39 → main.py:27,75), `COMPACT_TOKEN_RATIO` ← `--compact-token-ratio` (:40 → :29,76), `COMPACT_LATENCY_MS` ← `--compact-latency-ms` (:41 → :31,77), `COMPACT_KV_RATIO` ← `--compact-kv-ratio` (:43 → :33,78), `EVAL_MAX_TOKENS` ← `--eval-max-tokens` (:34 → :49,81), `EVAL_THINKING_BUDGET` ← `--eval-thinking-budget` (:36 → :54,83), `IMAGE_CACHE_MAX_SIZE` ← `--image-cache-max-size` (:21 → :56,84), `IMAGE_CACHE_TTL` ← `--image-cache-ttl` (:22 → :58,85) — so the universal negative fails. The claim's parenthetical is the true part: `KV_POLL_INTERVAL` (config.py:44, int) has no CLI flag — its only assignment in the package is its config.py default, and compaction.py:57 reads it directly; the numeric globals without a flag are `KV_POLL_INTERVAL`, `COMPACT_NUDGE_RATIO` (config.py:42) and `MIN_COMPLETION_TOKENS` (config.py:103). The verdict is reading-independent: it fails both if "numeric config global" means any numeric module-level global in config.py and if it means only the mutable values `main()` assigns.
Source: buster_ripper/main.py click option decorators (lines 25-61) and their `config.*` assignments in `main()` (lines 75-85); buster_ripper/config.py lines 21-44 (definitions); server.py lines 106-110 hold a second argparse parser defining only `--host`/`--port`, touching no config global
Version: unavailable
Fingerprint: sha256:b4794796d28ad3ab9ce5f2e0e03dddcfa882178299c7b969751697993cb76089
Fingerprint: sha256:14ac40180f081d8e7caeb280d9a9ab194cbd06c93e3958ebf9bef362ca7b7aab
Invalidates-On: any click option added to or removed from main.py that wires a numeric config global
Invalidates-On: any numeric global added, removed or retyped in config.py's setting sections, or CLI wiring moved out of main.py (e.g. into server.py or a new entrypoint module)
Checked-At: 2026-09-19T15:10:34Z
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-22.md

---

Claim: CL:buster_ripper.main/dump-cli-surface
Statement: `--dump-dir` is the only dump-related CLI option in `main.py` today
Subject: repo:buster_ripper.main
Verdict: HOLDS
Actual: `buster_ripper/main.py` defines 21 click options (lines 13-61) and exactly one of them concerns dumping — `--dump-dir` (line 25, "Enable session prompt diffing under PATH/<session_id>/."), wired to `config.DUMP_DIR` at lines 88-89; the string "dump" appears in main.py only at lines 25, 63 and 88-89, all that one option. The dump code path corroborates the exclusivity: dump.py consumes only `config.DUMP_DIR` (assert at dump.py:63, path use at :71) and the sole caller gates on `config.DUMP_DIR is not None` (routes/messages.py:55) — no other CLI-wired config value feeds dump behaviour; even under a strained reading where "dump-related" means any option touching dump execution, nothing else qualifies (`--verbose` raises the log level to DEBUG but dump logs at INFO, and dump output content depends only on the request body and DUMP_DIR). No definitional choice was needed.
Source: buster_ripper/main.py option decorators (lines 12-61) and `dump_dir` wiring (lines 88-89); corroborated by buster_ripper/dump.py:61-109 and buster_ripper/routes/messages.py:55
Version: unavailable
Fingerprint: sha256:b4794796d28ad3ab9ce5f2e0e03dddcfa882178299c7b969751697993cb76089
Fingerprint: sha256:243b21434c14e6ea8144329fe1d0ac4e0c499ed9a7e6e09bdcac5e2417a56b1d
Invalidates-On: any dump-related option added to (or `--dump-dir` removed from) main.py's option decorators
Invalidates-On: dump.py or its callers start consuming another CLI-wired config value, or the dump gate in routes/messages.py changes from `config.DUMP_DIR` to another flag-driven value
Checked-At: 2026-09-19T15:10:34Z
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-23.md

---

Claim: CL:buster_ripper/tls-surface
Statement: No ssl, TLS or certificate-verification symbol, option or context exists anywhere in `buster_ripper/`
Subject: repo:buster_ripper
Verdict: HOLDS
Actual: The package's 16 source files contain zero occurrences of any ssl/TLS/certificate-verification vocabulary — case-insensitive package-wide searches for `ssl`, `tls`, `certificate`, `cert_`, `verify`, `fingerprint`, `SSLContext`, `create_default_context`, `cafile`, `capath`, `load_cert_chain`, `assert_hostname`, `check_hostname`, `wrap_socket`, `trust_env`, `ciphers`, `insecure`, `trust_all`, `no_verify`, `skip_tls`, `self_signed`, `hostname` all returned no matches; the only URL scheme present anywhere is plain `http://` (`config.py:13` `UPSTREAM` default, `main.py:13` `--upstream` CLI default, backend.py:33 docstring) — no `https`, no TLS context, no connector SSL option, no verification flag.
Source: package source tree buster_ripper/ (16 .py files, __pycache__ excluded), searched with the pattern set above; representative locator for the only URL scheme: buster_ripper/config.py:13 (`UPSTREAM: str = "http://localhost:30000"`)
Version: unavailable
Fingerprint: sha256:60aebdf2191e7785586f3d1aa8fe0663136eadbeb76be53e072fefb4d8bf6381
Invalidates-On: any occurrence of an ssl/TLS/certificate-verification symbol, kwarg (e.g. `ssl=`, `ssl_context=`, `trust_env=`), connector option, or `https://` URL appearing in package source, or any change to the URL-scheme constants in config.py/main.py
Invalidates-On: any new file added under `buster_ripper/` (absence claims are file-set claims; a new file voids the file-set fingerprint)
Checked-At: 2026-09-19
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-24.md

---

Claim: CL:buster_ripper/client-session-construction
Statement: The `buster_ripper` package constructs exactly one `aiohttp.ClientSession` (`server.py`'s `on_startup`)
Subject: repo:buster_ripper
Verdict: HOLDS
Actual: The only `aiohttp.ClientSession(` construction under `buster_ripper/` is `app["session"] = aiohttp.ClientSession(...)` at `server.py:66`, inside `on_startup` (def at `server.py:55`), built on an explicit `aiohttp.TCPConnector` (line 65). Every other mention of `ClientSession` in the package (`backend.py:20`, `compaction.py:28`) is a type annotation, not a construction. Scope note: `buster_ripper_legacy.py` at the repo root is a separate top-level module outside the `buster_ripper/` package and constructs no `aiohttp.ClientSession` anyway (it uses `httpx.AsyncClient`), so the count stays one even under a whole-repo reading of "package".
Source: buster_ripper/server.py lines 55–68 (on_startup; construction at line 66), corroborated by a package-wide grep for `ClientSession` over `buster_ripper/`
Version: unavailable
Fingerprint: sha256:a9e49823ea456af9c0076781c4713c956745b21b026dfa768faf947e0b130592
Invalidates-On: A second `ClientSession(` construction (or `web.ClientSession`/session factory) added anywhere under `buster_ripper/`
Invalidates-On: The construction at server.py:66 moved out of `on_startup` or of server.py
Invalidates-On: A change that makes backend.py:20 or compaction.py:28 construct rather than annotate
Checked-At: 2026-09-28
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-25.md

---

Claim: CL:buster_ripper/outbound-call-routing
Statement: Every outbound HTTP call in the `buster_ripper` package goes through `app["session"]`
Subject: repo:buster_ripper
Verdict: HOLDS
Actual: The package contains exactly six outbound HTTP call sites, all invoked on the one shared session stored at `app["session"]`: routes/passthrough.py:29 `session.request(...)` (session read at :25), routes/messages.py:130 and :231 `session.post(...)` (read at :127/:228), routes/chat.py:125 `session.post(...)` (read at :37, passed into `_make_chat_request`), compaction.py:36 `session.get(...)` (parameter; sole caller server.py:21-22 passes `app["session"]`), and backend.py:54 `session.get(...)` (parameter; sole caller server.py:50-51 passes `session=app["session"]`); no alternative outbound mechanism exists anywhere in the package — no urllib/httpx/requests/http.client imports, no module-level `aiohttp.get/post/request`, no `asyncio.open_connection`/`create_connection`/`urlopen` (the only `connect` match is session.py:36 `sqlite3.connect`, a local database, not HTTP).
Source: call-site enumeration across buster_ripper/; session-provenance locators: buster_ripper/server.py lines 21-22 and 50-51, routes/passthrough.py:25, routes/messages.py:127/:228, routes/chat.py:37
Version: unavailable
Fingerprint: sha256:60aebdf2191e7785586f3d1aa8fe0663136eadbeb76be53e072fefb4d8bf6381
Invalidates-On: any new outbound HTTP call site added under `buster_ripper/` (any `session.<verb>(` on an object other than the shared session, or any new HTTP client import)
Invalidates-On: any change to server.py's passing of `app["session"]` into `backend.detect_backend` or `compaction.poll_kv_cache`, or any new file added under `buster_ripper/`
Checked-At: 2026-09-19
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-26.md

---

Claim: CL:pyproject.toml/coverage-gate-config
Statement: `pyproject.toml` omits `buster_ripper/routes/*`, `main.py` and `server.py` from coverage and gates the suite at 80% (`--cov-fail-under=80` in `addopts`)
Subject: repo:pyproject.toml
Verdict: HOLDS
Actual: Both halves are present in pyproject.toml: `[tool.coverage.run]` (with `source = ["buster_ripper"]`) has `omit = ["buster_ripper/routes/*", "buster_ripper/main.py", "buster_ripper/server.py"]`, so the route handlers, the package's main module and the package's server module are excluded from coverage; `[tool.pytest.ini_options]` `addopts = "--cov=buster_ripper --cov-report=term-missing --cov-fail-under=80"` sets the 80% gate exactly as claimed. The omit entries are package-qualified; with `source` limited to the `buster_ripper` package there is no other reading in which different files are meant.
Source: pyproject.toml line 19 (addopts) and lines 24-30 ([tool.coverage.run] source/omit)
Version: unavailable (project version 0.1.0)
Fingerprint: sha256:e1b7035805a8e946b71f947026e86e47e328f7e0b9ed9568b8cdb648e059079e
Invalidates-On: any edit to `[tool.coverage.run]`'s omit list (removal or change of the three omit entries)
Invalidates-On: any edit to `[tool.pytest.ini_options]` `addopts` removing or changing `--cov-fail-under=80`, or coverage/pytest-cov semantics changing so `omit` patterns no longer match `source`-rooted paths
Checked-At: 2026-09-19
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-27.md

---

Claim: CL:pyproject.toml/pytest-collection-scoping
Statement: `pyproject.toml`'s pytest ini configures no collection scoping — no `testpaths` and no `norecursedirs`
Subject: repo:pyproject.toml
Verdict: HOLDS
Actual: `[tool.pytest.ini_options]` in pyproject.toml contains exactly two keys — `asyncio_mode = "auto"` and `addopts = "--cov=buster_ripper --cov-report=term-missing --cov-fail-under=80"`; neither `testpaths` nor `norecursedirs` appears anywhere in the file, so the pytest ini configures no collection scoping; no competing pytest config file exists in the repo root (`pytest.ini`, `tox.ini`, `setup.cfg`, `.coveragerc` all absent), so pyproject.toml's ini is the whole config surface and the absence is real, not overridden elsewhere.
Source: pyproject.toml lines 17-19 ([tool.pytest.ini_options] section)
Version: unavailable (project version 0.1.0)
Fingerprint: sha256:e1b7035805a8e946b71f947026e86e47e328f7e0b9ed9568b8cdb648e059079e
Invalidates-On: addition of `testpaths` or `norecursedirs` (or any other collection-scoping key) to `[tool.pytest.ini_options]`
Invalidates-On: creation of a `pytest.ini`/`tox.ini`/`setup.cfg` that changes which ini source pytest reads, or deletion/restructuring of the `[tool.pytest.ini_options]` section
Checked-At: 2026-09-19
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-28.md

---

Claim: CL:pyproject.toml/console-script-entry-point
Statement: `pyproject.toml` defines a `buster-ripper` console-script entry point binding to `buster_ripper.main:main`
Subject: repo:pyproject.toml
Verdict: HOLDS
Actual: pyproject.toml declares `[project.scripts]` with exactly one entry — `buster-ripper = "buster_ripper.main:main"`, a console script named `buster-ripper` bound to the `main` callable in the `buster_ripper.main` module, as claimed.
Source: pyproject.toml lines 10-11 ([project.scripts] section)
Version: unavailable (project version 0.1.0)
Fingerprint: sha256:e1b7035805a8e946b71f947026e86e47e328f7e0b9ed9568b8cdb648e059079e
Invalidates-On: rename or removal of the `buster-ripper` key in `[project.scripts]`
Invalidates-On: change of the binding target string away from `buster_ripper.main:main`, or removal/renaming of `buster_ripper/main.py`'s `main` callable such that the declared target no longer resolves
Checked-At: 2026-09-19
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-29.md

---

Claim: CL:tests/upstream-mocking
Statement: The existing integration suite substitutes a mock for the upstream session rather than using a real network client
Subject: repo:tests
Verdict: HOLDS
Actual: The integration suite is now four endpoint modules — tests/test_messages_routes.py, tests/test_chat_routes.py, tests/test_count_tokens_routes.py and tests/test_passthrough_routes.py — all served by the `proxy` fixture in tests/conftest.py:45-50, which builds the real app with backend detection and the image flag neutralised (tests/conftest.py:36-42); no module defines a local client fixture. `app["session"]` is the shared upstream client session (buster_ripper/server.py:80), and every route test in the four modules that reaches upstream patches that shared session's methods with `mocker.patch.object(proxy.server.app["session"], "post", ...)` (tests/test_messages_routes.py:30-33, 61, 77-80, 113, 151, 161-164; tests/test_chat_routes.py:22, 45, 63, 84-87, 115, 132, 151-154; tests/test_count_tokens_routes.py:45-46) or `.request` for passthrough (tests/test_passthrough_routes.py:18-21), returning `StubUpstream`/`StubStream` instances (tests/upstream_stub.py:38-75) whose read/text/json/content.iter_any and __aenter__/__aexit__ answer without a network. No test in the suite sends a real network request toward the upstream. Precision note: the mocks are patched onto the real session object's methods rather than replacing the session object itself, and the incoming-request side does use aiohttp TestServer/TestClient (real local HTTP) — it is the upstream side that is fully mocked. tests/test_backend.py and test_compaction.py use real sessions/Mocks against in-process apps but do not contradict the claim as stated about the integration suite. Re-verified 2026-10-21 against the four endpoint modules and their shared fixtures.
Source: tests/test_messages_routes.py (lines 30-33, 61, 77-80, 113, 151, 161-164), tests/test_chat_routes.py (lines 22, 45, 63, 84-87, 115, 132, 151-154), tests/test_count_tokens_routes.py (lines 45-46), tests/test_passthrough_routes.py (lines 18-21), tests/conftest.py (lines 36-50), tests/upstream_stub.py (lines 38-75); corroborated by buster_ripper/server.py:80
Version: unavailable
Fingerprint: sha256:96e26634e582e8374fc868d3881d18c4fb3820d1d9ba5cd7b9c3408eeeefd6fa
Fingerprint: sha256:ab4b6f9519486a0c6c300b7295b5e729b794b11d7629c59c8efb961ded0d75e7
Fingerprint: sha256:35de7eacbfc237cf112fd56fe906fb4b8c5a1358fefc68156a44cb0d064f9fdb
Fingerprint: sha256:81c6314169493f3285c0d3cb20043ff42333f7b92ee154ab75bc2f4ab031bc23
Fingerprint: sha256:e821283c228bd63abdc9dbe215cd0e3867efdf77b65bf0e201d3beacd23c8de9
Fingerprint: sha256:33778bc9bd851b2070ca05944fc199457e51c25337184e7c9f59c30901339d82
Fingerprint: sha256:7caa78a05a7a434adeb0555282c96dc9d484ba5c868e608837a143da6fb9b4e7
Invalidates-On: any of tests/test_messages_routes.py, tests/test_chat_routes.py, tests/test_count_tokens_routes.py, tests/test_passthrough_routes.py stops patching `app["session"]` post/request before exercising routes (i.e. a test performs a real upstream request)
Invalidates-On: buster_ripper/server.py stops storing the upstream client session at `app["session"]`, or the integration suite moves to a different module again or a second integration-level suite appears using real network clients toward the upstream
Checked-At: 2026-10-21
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-31.md

---

Claim: CL:serve_buster_ripper.sh/option-forwarding
Statement: `serve_buster_ripper.sh` passes CLI options to the binary through two channels — the `exec` line's always-on `--upstream`/`--host`/`--port` (serve_buster_ripper.sh:36-40) and the conditional `EXTRA_FLAGS` flags (:28-32) — and forwards no positional arguments
Subject: repo:serve_buster_ripper.sh
Verdict: HOLDS
Actual: The 40-line script invokes the binary exactly once, on the `exec buster-ripper` statement spanning lines 36-40: it passes the three direct named options `--upstream "${UPSTREAM}"` (:37), `--host "${HOST}"` (:38) and `--port "${PORT}"` (:39), then the unquoted `${EXTRA_FLAGS}` expansion (:40), which word-splits into the flags conditionally appended at :29-32 (`--strip-date`, `--verbose`, `--dump-dir ${DUMP_DIR}`, `--preserved-thinking`, with `EXTRA_FLAGS` initialised empty at :28). No other invocation, source or argument-forwarding mechanism exists in the file — no `"$@"`, `"$*"` or `$N` positional reference anywhere, so no positional arguments reach the binary; the `echo` on line 34 writes to stdout only. Minor citation nuance: the direct arguments strictly sit on lines 37-39, but the cited range 36-40 is the full multi-line exec statement, so the locator is accurate as stated.
Source: serve_buster_ripper.sh lines 20-40 (variable defaults at 20-26, EXTRA_FLAGS construction at 28-32, exec statement at 36-40); whole file searched for positional-parameter references
Version: unavailable
Fingerprint: sha256:67d194bc95dcc42ed3d09ee0214e97fd754465a615ebd555c8d3280f170361a9
Invalidates-On: any edit to the exec statement (lines 36-40) adding, removing or reordering arguments passed to `buster-ripper`
Invalidates-On: any edit adding `"$@"`, `"$*"` or a positional-parameter expansion anywhere in the script, or adding a second invocation of `buster-ripper` or sourcing another file that invokes it
Checked-At: 2026-09-19
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-49.md

---

Claim: CL:README.md/documented-run-path
Statement: `README.md`'s dump section documents the serve script (`DUMP_DIR=… bash serve_buster_ripper.sh`) as its run path
Subject: repo:README.md
Verdict: HOLDS
Actual: The "## Dump mode (`--dump-dir`)" section's fenced bash block gives exactly `DUMP_DIR=~/mllm/prompt-diffs bash serve_buster_ripper.sh` as the way to run dump mode — the serve script invoked with the `DUMP_DIR` env var, not a direct `buster-ripper` command. (A direct-call variant appears only later, inside the collapsed-formats subsection, as an "or directly" alternative.)
Source: README.md — "## Dump mode (`--dump-dir`)" section, fenced bash block `DUMP_DIR=~/mllm/prompt-diffs bash serve_buster_ripper.sh` (line 145 of the file)
Version: unavailable (working-tree README.md at repo root; git metadata not consulted)
Fingerprint: sha256:fe80cb9acabfe699fac070a69b43ac425436606d627b243825d4a028d75a1c15
Invalidates-On: The dump-mode section's run command changes away from the `DUMP_DIR=… bash serve_buster_ripper.sh` serve-script invocation (e.g. replaced by a direct `buster-ripper --dump-dir …` call)
Invalidates-On: sha256 of README.md no longer equals fe80cb9acabfe699fac070a69b43ac425436606d627b243825d4a028d75a1c15
Checked-At: 2026-09-28
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-33.md

---

Claim: CL:buster_ripper.config/config-read-timing
Statement: `buster_ripper`'s modules read `config` attributes at call time rather than binding them at import, so values assigned before serving are visible everywhere
Subject: repo:buster_ripper.config
Verdict: DOES-NOT-HOLD
Actual: Most request-path reads use `config.<NAME>` at call time, but `buster_ripper.main` evaluates `config.UPSTREAM`, `config.DUMP_FLUSH_INTERVAL`, and `config.DUMP_FLUSH_BYTES` as Click option defaults while defining the command at import time (lines 30–57). Assigning these attributes after importing `main` but before serving does not update those already-bound CLI defaults; the blanket “everywhere” claim is false.
Source: `buster_ripper/main.py:30-57` (Click option decorators); `buster_ripper/main.py:104-105` (parsed upstream reassigned to config)
Version: unavailable
Fingerprint: sha256:cbe2802d474abe21ada5bb2e94c09cce15237ea53d526c6d31fed25207c950d1
Invalidates-On: Change to `buster_ripper/main.py` option defaults or how the CLI assigns configuration at startup.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-34.md

---

Claim: CL:aiohttp/signal-shutdown
Statement: `web.run_app` handles SIGINT/SIGTERM with a graceful shutdown that runs the app's cleanup hooks before the process exits
Subject: pkg:aiohttp@3.13.3
Verdict: HOLDS
Actual: With the default `handle_signals=True`, `run_app` → `AppRunner` → `BaseRunner.setup()` installs `loop.add_signal_handler(signal.SIGINT, _raise_graceful_exit)` and the same for SIGTERM (web_runner.py:281-287); `_raise_graceful_exit` raises `GracefulExit(SystemExit)` (web_runner.py:35-40), which escapes `loop.run_until_complete(main_task)` (web.py:510) and is caught by `except (GracefulExit, KeyboardInterrupt): pass` (web.py:511-512); the `finally` block then cancels the main task and runs it to completion with `suppress(asyncio.CancelledError)` (web.py:513-517), so `_run_app` unwinds through its `finally: await runner.cleanup()` (web.py:424-425) — that cleanup stops the sites, calls `AppRunner.shutdown()` → `app.shutdown()` (on_shutdown signal), shuts the server down within `shutdown_timeout`, then calls `AppRunner._cleanup_server()` → `app.cleanup()` (web_runner.py:386-310, 398-399), which sends `on_cleanup` including cleanup-context exit phases (web_app.py:174, 481-490); only after the task completes does `run_app` cancel remaining tasks, shut down asyncgens and close the loop (web.py:518-521), and the process exits after that. Corroborated by the v3.13.3 docs; caveat: behaviour depends on the default `handle_signals=True`, and on Windows `add_signal_handler` raises NotImplementedError so the handlers are silently not installed (web_runner.py:285-287).
Source: installed: .venv/lib/python3.13/site-packages/aiohttp/web.py (lines 452-521 `run_app`, 305-425 `_run_app`), aiohttp/web_runner.py (lines 35-40, 278-320, 366-399), aiohttp/web_app.py (lines 167-174, 474-490); corroborated by docs.aiohttp.org v3.13.3 web_reference.html and web_advanced.html (Graceful-shutdown section)
Version: aiohttp==3.13.3
Fingerprint: sha256:27348d99e8e0e46e987850279088197f2b6a6d4f3ab0dbbce38c982a6a14a6ab
Fingerprint: sha256:bf51b59ca88e3908059d3491e0831ce88f5178414331a1ed30bbcefeba0333e0
Fingerprint: sha256:94653f68030dfa1df0cbe2d34c78ba49e287f3276dd46e75057702b29803e590
Fingerprint: sha256:d4ecce5be78af57777803c52bdf20da5d5b964298ce53d8537e044dcc5f51fbd
Fingerprint: sha256:b0c18a7e6ed6f4bc6cc13fd7af94c75bd308abdd0f790a94a0633b053623cc9d
Invalidates-On: aiohttp upgrade that moves signal-handler installation out of `BaseRunner.setup()` or changes the `handle_signals=True` default in `run_app`
Invalidates-On: aiohttp change to `run_app`'s `finally` block that no longer runs the cancelled main task to completion (cleanup would no longer execute before exit)
Invalidates-On: aiohttp change to `_run_app`'s `finally: await runner.cleanup()` ordering, or to `BaseRunner.cleanup()`/`Application.cleanup()` dispatch of on_shutdown/on_cleanup
Checked-At: 2026-09-19T15:17:24Z
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-35.md

---

Claim: CL:aiohttp/cleanup-hook-error-propagation
Statement: An exception raised inside an app cleanup hook other than aiohttp's own `GracefulExit` propagates out of `run_app`
Subject: pkg:aiohttp@3.13.3
Verdict: HOLDS
Actual: `_run_app` awaits runner cleanup in `finally`, and `run_app` does not suppress ordinary exceptions from its main task: the exception escapes after loop teardown. Even when the first `run_until_complete` is interrupted by a signal, its second await of the cancelled main task suppresses only `CancelledError`, so a cleanup failure escapes; multiple cleanup-context failures may instead escape as an aggregate `CleanupError`.
Source: Installed aiohttp `.venv/lib/python3.13/site-packages/aiohttp/web.py` lines 421–425, 508–521.
Version: aiohttp 3.13.3
Fingerprint: sha256:27348d99e8e0e46e987850279088197f2b6a6d4f3ab0dbbce38c982a6a14a6ab
Invalidates-On: Change to aiohttp `_run_app` cleanup await or `run_app` exception/suppression handling in web.py.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-36.md

---

Claim: CL:aiohttp/ssl-parameter-topology
Statement: aiohttp's `ClientSession.__init__` accepts `connector=` and has no `ssl` parameter — `ssl` is a `TCPConnector.__init__` parameter and a per-request parameter
Subject: pkg:aiohttp@3.13.3
Verdict: HOLDS
Actual: Verified in the installed source: `ClientSession.__init__` (client.py:274-309) is a keyword-only signature accepting `connector: Optional[BaseConnector] = None` (client.py:278) with no `ssl` parameter and no `**kwargs` — the only ssl-named parameter is the distinct, deprecated `ssl_shutdown_timeout` (client.py:308), so `ClientSession(ssl=...)` is a `TypeError`. `ssl` lives on `TCPConnector.__init__` as `ssl: Union[bool, Fingerprint, SSLContext] = True` (connector.py:978) and on requests as `ssl: Union[SSLContext, bool, Fingerprint] = True` in `ClientSession._request` (client.py:513, also `ws_connect` at client.py:1007); a per-request value takes precedence over the connector's in `TCPConnector._get_ssl_context` (connector.py:1262-1268).
Source: aiohttp/client.py:274-309 (`ClientSession.__init__` signature); corroborated by aiohttp/connector.py:978 and aiohttp/client.py:513
Version: aiohttp 3.13.3 (installed at .venv/lib/python3.13/site-packages/aiohttp, Python 3.13)
Fingerprint: sha256:7ce41fc1c2142f5346015455e030e3ebec22a41cde0fc299a66ce13be2cb2a9e (client.py); sha256:bd3db604db820ed6da744d54abb1c2ef3a4e5821ccc48e27dcfb42cfbcd9664c (connector.py)
Invalidates-On: aiohttp adds an `ssl` (or `**kwargs`) parameter to `ClientSession.__init__`
Invalidates-On: aiohttp removes the `ssl` parameter from `TCPConnector.__init__` or from per-request methods
Invalidates-On: installed client.py/connector.py bytes no longer match the fingerprints above (version drift)
Checked-At: 2026-09-28
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-37.md

---

Claim: CL:aiohttp/connector-loop-binding
Statement: aiohttp's `TCPConnector.__init__` resolves `loop = loop or asyncio.get_running_loop()` (connector.py:313), so constructing a connector with no running event loop raises `RuntimeError`; a `ClientSession` constructed with `connector=c` and no explicit `loop=` adopts `c._loop` as its own loop (client.py:314-316), so its `connector._loop is not loop` guard (client.py:378-379) is vacuous on that default path
Subject: pkg:aiohttp@3.13.3
Verdict: HOLDS
Actual: All three parts verified against installed aiohttp 3.13.3: connector.py:313 is exactly `loop = loop or asyncio.get_running_loop()`; it executes on every `TCPConnector(...)` construction because `TCPConnector.__init__` passes `loop=loop` to `BaseConnector.__init__` via `super().__init__` (connector.py:993-1001) — the line technically lives in `BaseConnector.__init__`, which `TCPConnector.__init__` always invokes. With `loop` unset and no running loop, `asyncio.get_running_loop()` raises `RuntimeError` (Python stdlib contract), before `loop.create_future()` at connector.py:363-364 could be reached. In `ClientSession.__init__`, client.py:314-316 is exactly `if loop is None: if connector is not None: loop = connector._loop` (adopted as `self._loop` at client.py:381); then client.py:378-379 checks `if connector._loop is not loop: raise RuntimeError(...)`. On that default path `loop` **is** `connector._loop` by identity, so the check is always False and the guard can never fire — it only triggers for an explicitly passed `loop=` differing from the connector's. Caveat: passing an explicit non-None `loop=` to the connector short-circuits the `or` and avoids the `RuntimeError`; the claim's own citation (the `loop or` fallback) makes the default-path reading the intended one.
Source: aiohttp/connector.py:313 (in `BaseConnector.__init__`, invoked by `TCPConnector.__init__` at connector.py:993-1001) and aiohttp/client.py:314-316, 378-379 (`ClientSession.__init__`)
Version: aiohttp 3.13.3 (installed at .venv/lib/python3.13/site-packages/aiohttp, Python 3.13)
Fingerprint: sha256:bd3db604db820ed6da744d54abb1c2ef3a4e5821ccc48e27dcfb42cfbcd9664c (connector.py); sha256:7ce41fc1c2142f5346015455e030e3ebec22a41cde0fc299a66ce13be2cb2a9e (client.py)
Invalidates-On: aiohttp stops resolving the connector loop via `asyncio.get_running_loop()` (e.g. switches to a lazy/deprecated-tolerant getter) or moves the resolution out of the `TCPConnector` construction path
Invalidates-On: aiohttp changes `ClientSession.__init__` so a connector-supplied loop is no longer adopted verbatim (client.py:314-316) or the same-loop guard (client.py:378-379) is reworked
Invalidates-On: installed connector.py/client.py bytes no longer match the fingerprints above (version drift)
Checked-At: 2026-09-28
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-50.md

---

Claim: CL:aiohttp/ssl-verification-polarity
Statement: aiohttp's `TCPConnector` verifies the upstream certificate with its default `ssl=True` and disables verification with `ssl=False`
Subject: pkg:aiohttp@3.13.3
Verdict: HOLDS
Actual: `TCPConnector.__init__` defaults `ssl: Union[bool, Fingerprint, SSLContext] = True` (connector.py:978) and stores it via `_merge_ssl_params` (connector.py:1003). `TCPConnector._get_ssl_context` (connector.py:1243-1274) maps the connector-level `True` to the module-level `_SSL_CONTEXT_VERIFIED` (connector.py:1274) and a not-True value (`False`) to `_SSL_CONTEXT_UNVERIFIED` (connector.py:1271-1273). `_make_ssl_context(True)` builds `ssl.create_default_context()` — the stdlib verified context (check_hostname on, `CERT_REQUIRED`) — while `_make_ssl_context(False)` sets `check_hostname = False` and `verify_mode = ssl.CERT_NONE` (connector.py:912-913), i.e. verification disabled. Note: a per-request `ssl=` overrides the connector setting (connector.py:1262-1268); the claim is about connector-level behaviour, which is exactly what the source does.
Source: aiohttp/connector.py:1243-1274 (`TCPConnector._get_ssl_context`), with connector.py:902-925 (`_make_ssl_context`) and connector.py:978 (default)
Version: aiohttp 3.13.3 (installed at .venv/lib/python3.13/site-packages/aiohttp, Python 3.13)
Fingerprint: sha256:bd3db604db820ed6da744d54abb1c2ef3a4e5821ccc48e27dcfb42cfbcd9664c (connector.py)
Invalidates-On: aiohttp changes `_get_ssl_context` so connector-level `ssl=True` no longer selects a verified context or `ssl=False` no longer selects `CERT_NONE`
Invalidates-On: aiohttp changes the `TCPConnector.__init__` `ssl` default away from `True`
Invalidates-On: installed connector.py bytes no longer match the fingerprint above (version drift)
Checked-At: 2026-09-28
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-39.md

---

Claim: CL:click/option-callback-timing
Statement: `click` invokes an option's `callback` during CLI parsing, before the command function body runs
Subject: pkg:click@8.3.1
Verdict: HOLDS
Actual: Confirmed in the installed source. During `Command.parse_args` (core.py:1226-1227), each parameter's `handle_parse_result` runs, which calls `Parameter.process_value`; there, `if self.callback is not None: value = self.callback(ctx, self, value)` (core.py:2438-2463) — so an option's callback executes while parsing, inside `make_context`. The command function body runs strictly later: `BaseCommand.main` builds the context via `self.make_context(...)` (which itself calls `parse_args` and whose docstring states "It does not invoke the actual command callback though") and only then calls `self.invoke(ctx)` (core.py:1405-1406), where `Command.invoke` executes `ctx.invoke(self.callback, **ctx.params)` (core.py:1268-1269). This is also the documented basis for eager options (e.g. `--help`) exiting from a callback before the command body starts.
Source: `.venv/lib/python3.13/site-packages/click/core.py` — `Parameter.process_value` (lines 2438-2463) reached via `Command.parse_args` (lines 1226-1227) / `handle_parse_result` (line 2577); command body in `Command.invoke` (lines 1268-1269) after `make_context` in `BaseCommand.main` (lines 1405-1406)
Version: click 8.3.1 (installed in repo `.venv`, pinned by `uv.lock`)
Fingerprint: sha256:53a05fc6df06923343ab22741ea5ef96e247b72678b18e54480bccd4276aee64 (click/core.py)
Invalidates-On: click version changes in `.venv`/`uv.lock` away from 8.3.1 with a rework of parameter processing order
Invalidates-On: sha256 of `click/core.py` no longer matches the fingerprint above
Invalidates-On: click defers parameter callbacks past `Command.invoke` (e.g. a parsing-model redesign in a future major version)
Checked-At: 2026-09-28
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-40.md

---

Claim: CL:click/choice-validation
Statement: `click.Choice` rejects values outside its choices list
Subject: pkg:click@8.3.1
Verdict: HOLDS
Actual: In click 8.3.1, `Choice.convert` normalizes the incoming value and searches the normalized choices; when no choice matches it calls `self.fail(self.get_invalid_choice_message(...))` (types.py:344-358), and `ParamType.fail` raises `BadParameter` (types.py:136-143), which click surfaces as a usage error; the class docstring states "The resulting value will always be one of the originally passed choices" (types.py:240). Nuance: the match is done on normalized values — with `case_sensitive=False` or a `token_normalize_func` in the context, a value differing in case/whitespace from the choice literal can still be accepted (and the original choice is returned); values with no normalized match are rejected.
Source: installed: .venv/lib/python3.13/site-packages/click/types.py — `Choice.convert` lines 336-358, `Choice.get_invalid_choice_message` lines 360-372, `ParamType.fail` lines 136-143
Version: click==8.3.1
Fingerprint: sha256:7a4e7804d485c0f2acaad7d3ee3b2a71ce161ee8bc00815530a338a156485e17
Invalidates-On: click upgrade where `Choice.convert` no longer calls `self.fail`/raises `BadParameter` for non-matching values (e.g. returning the raw value instead)
Invalidates-On: click upgrade where `ParamType.fail` stops raising `BadParameter`
Checked-At: 2026-09-19
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-41.md

---

Claim: CL:pytest/norecursedirs-semantics
Statement: pytest's `norecursedirs` ini option replaces the default directory-exclusion list rather than extending it
Subject: pkg:pytest@9.0.2
Verdict: HOLDS
Actual: Setting `norecursedirs` yields exactly the patterns the user wrote and nothing else — the library's docs state it verbatim: "Setting a ``norecursedirs`` replaces the default." In the installed source, `Config._getini` returns the registered `default` list (`*.egg`, `.*`, `_darcs`, `build`, `CVS`, `dist`, `node_modules`, `venv`, `{arch}`) only when no value is set at all; when a value is set — in the single config file pytest reads or via `-o` — it is selected alone and split by `shlex.split` (type `args`) with no union with the default, and collection-time exclusion in `pytest_ignore_collect` consumes `config.getini("norecursedirs")` directly, so only the replaced list applies. Separate, genuinely additive mechanisms — `collect_ignore`/`collect_ignore_glob` conftest lists and `--ignore`/`--ignore-glob` — are distinct from `norecursedirs` and do not merge into it.
Source: pytest 9.0.x docs, `.. confval:: norecursedirs` section of doc/en/reference/reference.rst (raw.githubusercontent.com; the doc's `pytest.__version__` doctest reads '9.0.2'); installed: .venv/lib/python3.13/site-packages/_pytest/main.py lines 213-228 and 455-458, _pytest/config/__init__.py `_getini` lines 1673-1704 and `_getini_ini` lines 1745-1746
Version: pytest==9.0.2
Fingerprint: sha256:bb65f8dc28e3dc60631dcfa4cf4c13051db7da90eac75e96c7bf340b03322987
Fingerprint: sha256:7ca2b20cc41fbfc2761bcb231eecb26967d553651035c40000f44d70e174b693
Fingerprint: sha256:ff967dc8cbd4baeaa10b2e370f638ead045306b13926b06d06eb778c3f81e1ea
Invalidates-On: any pytest upgrade in the project venv that changes the `norecursedirs` default list in `addini` or the `_getini` resolution semantics (e.g. introducing default merging)
Invalidates-On: update of the `9.0.x` docs branch tip altering the `.. confval:: norecursedirs` section (branch is mutable; recheck against a pinned tag such as 9.0.2), or introduction of a documented mechanism that appends to the default list
Checked-At: 2026-09-19T15:16:11Z
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-42.md

---

Claim: CL:python/asyncio-periodic-timer
Statement: asyncio provides no built-in periodic-timer/scheduler facility — a periodic loop is hand-built around `asyncio.sleep`
Subject: lang:python
Verdict: HOLDS
Actual: Both halves checked independently against the installed library. (1) asyncio's entire callback-scheduling surface is single-fire: `AbstractEventLoop` exposes only `call_soon` (defined as `self.call_later(0, ...)`), `call_soon_threadsafe`, `call_later`, and `call_at`; `call_later` resolves to one `TimerHandle` pushed on a heap, and its docstring states "Each callback will be called exactly once"; a package-wide search of all 35 `.py` files under the installed `asyncio/` for periodic/repeat/recurring/interval/every/cron/call_every/call_periodic finds no API, and the official version-pinned docs agree (no periodic timer or scheduler facility in either API index). (2) With no built-in facility, coroutine-based periodic execution must be hand-built, and asyncio's only awaitable timing primitive for coroutines is `asyncio.sleep` (itself a one-shot await over `loop.call_later`), so the standard hand-built periodic pattern is the sleep loop (`while True: await asyncio.sleep(interval)`). Corroboration note: a periodic callback can also be hand-built by chaining `loop.call_later`, which does not contradict the claim — the claim asserts no built-in facility and the sleep loop as the characteristic hand-built pattern, not exclusivity of the sleep loop.
Source: installed: homebrew python@3.14/3.14.3_1 lib/python3.14/asyncio/ — events.py (AbstractEventLoop scheduling methods at lines 305-334; `TimerHandle` at line 141), base_events.py (`call_later` at 775-797, `call_at` at 799-815), tasks.py (`sleep` at 687-704), __init__.py (`__all__` at 26-42), timeouts.py; corroborating docs.python.org/3.14 asyncio-api-index.html and asyncio-llapi-index.html
Version: CPython 3.14.3
Fingerprint: sha256:b1a138015c6e1878d7f8c6be51896da451d7f25c24f6ef18d76ceff249050d94
Fingerprint: sha256:a82285b1ccb379d4bc18f3c1b05a2876040daed1523e7fb6bc05c0a86977894d
Fingerprint: sha256:8554946e93b93d70288a2ab0ad7f255cf3119c4861f89e3c9194956d0f3f591a
Fingerprint: sha256:8148552daea881e3b33639d6e9bdab0da2021367ab694ee0604431c33c16983f
Fingerprint: sha256:3154c617c94bf0124b0b5eaf1a6aef89cc0776e95d00b5e4e5302e32b29cb06b
Invalidates-On: a Python release whose asyncio adds a periodic/recurring timer or scheduler API (e.g. `loop.call_every`, `asyncio.repeat_every`, or a built-in periodic scheduler class)
Invalidates-On: a different installed CPython version or patch level (re-read and re-hash the `asyncio/` package source; this verdict is scoped to 3.14.3), or a restructure of the docs.python.org asyncio API index that adds a periodic-timer section under "Event Loop Methods"
Checked-At: 2026-09-19
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-43.md

---

Claim: CL:python/path-open-append-semantics
Statement: `Path.open("a")` appends to the file, creating it if it does not exist
Subject: lang:python
Verdict: HOLDS
Actual: `pathlib.Path.open(mode="a")` opens with built-in `open()` semantics for mode `'a'`: the stream is opened for writing and positioned at the end, and writes append ("on some Unix systems, all writes append to the end of the file regardless of the current seek position"); the file is created if it does not exist when opened for appending. `pathlib.Path.open` defers to `io.open` with the mode passed through unchanged (installed pathlib/__init__.py:763), so `'a'` semantics are exactly built-in `open()` semantics.
Source: docs.python.org/3.14/library/functions.html#open (mode table row `'a'`; prose on appending), docs.python.org/3.14/library/io.html#io.FileIO (creation on append), docs.python.org/3.14/library/pathlib.html#pathlib.Path.open (delegation); installed pathlib/__init__.py:763 (homebrew python@3.14)
Version: CPython 3.14.3
Fingerprint: sha256:3ff7c1378e9e6fd6fb9ad5d209ab2caf5c4a559e939d7cdb217080452083996b
Fingerprint: sha256:7b186d639315fec59ace6429e7dd649030cc8657c5546873d200ee10fde7d629
Fingerprint: sha256:90d01dd0a3f67ca236cfad9c5fe04fe3d57dc16e59448e8789c5e0d67001a9ab
Fingerprint: sha256:042a4d0a5e1ef66778daa86dc269af205bd42d06e793e53d8b99dff4c45928ca
Invalidates-On: Python documentation for mode `'a'` or `io.FileIO` no longer stating creation-on-append/append semantics (e.g. a future version changes the mode table row)
Invalidates-On: installed pathlib `Path.open` stops delegating to `io.open` with the mode passed through unchanged, or an interpreter major/minor upgrade past the consulted 3.14 line (re-check docs for the new version)
Checked-At: 2026-09-19
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-44.md

---

Claim: CL:buster_ripper/dump-artifact-write-sites
Statement: The only write-bearing content-write file operations the `buster_ripper` package performs on dump artifacts under `DUMP_DIR` are `dump.py`'s three sites — `dump.py:82` (`turn_000.json`), `:95` (`turn_NNN.diff`) and `:105-106` (the `index.txt` append). Content-write means `write_text`/`write_bytes`/`open` for write-or-append plus `f.write`; directory creation (`mkdir`, `dump.py:72` / `main.py:90`) is excluded by definition; SQLite persistence (`session.py:35-46`, `:59-71`) is not a dump-artifact content write
Subject: repo:buster_ripper
Verdict: HOLDS
Actual: The package's entire content-write surface is exactly the three cited sites, all writing under `DUMP_DIR` (`session_dir = config.DUMP_DIR / sid`, dump.py:71): `out.write_text(json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8")` for `turn_000.json` (:82), `out.write_text("".join(diff_lines), encoding="utf-8")` for `turn_{turn:03d}.diff` (:95), and the append-mode `index.open("a", ...)` + `f.write(summary + "\n")` for `index.txt` (:105-106). Every other candidate operation falls outside the claim's unit by its own definition or is not a file operation at all: `session_dir.mkdir(...)` (dump.py:72) and `config.DUMP_DIR.mkdir(...)` (main.py:90) are directory creation; `session.py`'s `db_init`/`db_upsert` (sqlite3, :35-71) is database persistence; `logging.basicConfig` (main.py:100-104) sets no `filename`, so logging writes to stderr; `await response.write(chunk)` (routes/messages.py:174) is an `aiohttp.web.StreamResponse` write to the HTTP client — a network stream, not a file, matching none of the unit's forms.
Source: buster_ripper/dump.py:82, :95, :105-106 (the three sites), with :71 (`session_dir = config.DUMP_DIR / sid`), :81/:94/:104 (artifact filenames are inline literals) and config.py:16 (`DUMP_DIR: Optional[Path] = None`); negative sweeps over all 16 package files (glob-enumerated): `write_text|write_bytes|.write(|open(`, plus write-vector sweep `tempfile|fdopen|json.dump|pickle|shutil|copyfile|rename|touch|mkdir|sqlite|aiofiles|NamedTemp|mkstemp|unlink|remove(`, `subprocess|Popen|os.system|dup2|os.write`, and `FileHandler|basicConfig|addHandler|StreamHandler|log_file|logfile|print(|file=|sys.std`
Version: unavailable
Fingerprint: sha256:243b21434c14e6ea8144329fe1d0ac4e0c499ed9a7e6e09bdcac5e2417a56b1d
Fingerprint: sha256:38baf4bba6508a5c17df853e1e05393850044456cb5a65a226f4d98e62a36795
Invalidates-On: any content-write (`write_text`/`write_bytes`, or write-or-append `open` + `f.write`) appears under `buster_ripper/` outside dump.py:82/:95/:105-106
Invalidates-On: any byte of any file under `buster_ripper/` changes (fingerprint mismatch — re-run the package sweep and re-check the three sites), or dump.py:71 stops deriving `session_dir` from `config.DUMP_DIR` (the sites would no longer be writes on DUMP_DIR dump artifacts)
Checked-At: 2026-09-19T17:32:21Z
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-45.md

---

Claim: CL:buster_ripper/periodic-flush-ownership
Statement: No symbol under `buster_ripper/` owns a periodic flush role for the dump — no symbol buffers dump records for later writing or flushes a pending dump buffer on a timer
Subject: repo:buster_ripper
Verdict: HOLDS
Actual: Confirmed absent, by enumeration of every wake-up mechanism in the package. The dump has no pending-records queue to drain: `dump_turn()` writes each record synchronously and immediately at call time (`write_text` for turn 0 full JSON or turn N diff, plus an append to `index.txt`), and the module's only state (`_sessions: dict[str, SessionState]`) is per-session bookkeeping (turn counter, previous rendered text, session dir) — not a buffer of unflushed records; its sole call site is request-driven (routes/messages.py:57, inline inside `handle_messages`, guarded by `config.DUMP_DIR is not None`, wrapped in try/except). The only recurring wake-up in the package is `compaction.poll_kv_cache()` (`while True:` + `await asyncio.sleep(config.KV_POLL_INTERVAL)`, spawned by `server.spawn_kv_cache_monitor()` in `on_startup`, cancelled in `on_cleanup`); its job is polling upstream `/metrics` for `vllm:kv_cache_usage_perc` — it imports only `config` and never touches the dump. Keyword sweeps (sleep/periodic/timer/schedule/cron/`while True`/`create_task`/`ensure_future`/`call_later`/`call_at`/threading/`Timer`/alarm/sched/TaskGroup) across all 16 modules surfaced no other periodic construct; `on_cleanup` performs no dump drain either — it only cancels the KV task and closes the aiohttp session.
Source: buster_ripper/dump.py (full file — synchronous write path, no queue/flush symbol), corroborated by buster_ripper/compaction.py:28-57 and buster_ripper/server.py:15-57 (sole periodic loop, non-dump job) and buster_ripper/routes/messages.py:55-59 (sole, synchronous call site)
Version: unavailable
Fingerprint: sha256:243b21434c14e6ea8144329fe1d0ac4e0c499ed9a7e6e09bdcac5e2417a56b1d
Fingerprint: sha256:ac9d42143c98694a0b6672dd753afcef537fc5e03ec747b4bb3c1208e23535e7
Fingerprint: sha256:e008d3c971db787ebd1eb890544c6d73f0479bb97d0060ee64d4f77cf150b824
Fingerprint: sha256:9c234c4f183aa03a80ab82356a6fdfaad0220f3bef69edf3bab5789ef808277d
Invalidates-On: any symbol added under `buster_ripper/` that buffers dump records (queue/batch/deferred list in dump.py or elsewhere) and flushes them from a periodic task
Invalidates-On: `compaction.poll_kv_cache()` or any new background loop gaining dump-drain behavior (e.g. importing dump or writing DUMP_DIR files), the routes/messages.py dump call site moving from inline synchronous execution to deferred/queued execution with a timer-based writer, or removal/restructuring of `dump_turn()`'s immediate `write_text` path in dump.py
Checked-At: 2026-09-19T17:29:26Z
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-46.md

---

Claim: CL:rel:buster_ripper.main→buster_ripper.config/numeric-config-cli-wiring
Statement: The numeric `config` globals wired to CLI flags in `buster_ripper/main.py` are `MAX_MODEL_LEN`, `COMPACT_TOKEN_RATIO`, `COMPACT_LATENCY_MS`, `COMPACT_KV_RATIO`, `EVAL_MAX_TOKENS`, `EVAL_THINKING_BUDGET`, `IMAGE_CACHE_MAX_SIZE`, `IMAGE_CACHE_TTL`; the numeric globals without a CLI flag are `KV_POLL_INTERVAL`, `COMPACT_NUDGE_RATIO` and `MIN_COMPLETION_TOKENS`
Subject: rel:buster_ripper.main→buster_ripper.config
Verdict: HOLDS
Actual: The partition is exactly right. `buster_ripper/config.py` contains precisely 11 numeric (int/float) module-level globals, exactly the 11 names the claim lists. Each of the eight flagged names has a `@click.option` in `buster_ripper/main.py` assigned straight onto `config.<NAME>` in `main()`: `MAX_MODEL_LEN` ← `--max-model-len` (main.py:27 → :75), `COMPACT_TOKEN_RATIO` ← `--compact-token-ratio` (:29 → :76), `COMPACT_LATENCY_MS` ← `--compact-latency-ms` (:31 → :77), `COMPACT_KV_RATIO` ← `--compact-kv-ratio` (:33 → :78), `EVAL_MAX_TOKENS` ← `--eval-max-tokens` (:49 → :81), `EVAL_THINKING_BUDGET` ← `--eval-thinking-budget` (:54 → :83), `IMAGE_CACHE_MAX_SIZE` ← `--image-cache-max-size` (:56 → :84), `IMAGE_CACHE_TTL` ← `--image-cache-ttl` (:58 → :85). Each of the three unflagged names has no CLI flag: `KV_POLL_INTERVAL` (config.py:44) and `MIN_COMPLETION_TOKENS` (config.py:104) do not appear in `main.py` at all, and `COMPACT_NUDGE_RATIO` (config.py:42) appears only as a read in the startup log line (main.py:120) — never a `@click.option`, never assigned from a CLI parameter. Scoping note: bools are int-typed in Python; the six bool globals are excluded from "numeric" as boolean switches, not numeric tunables — under the bool-inclusive pedantic reading the omitted six all also carry flags, so no reading makes any statement about the named names false.
Source: buster_ripper/config.py (numeric globals at lines 21-22, 34, 36, 39-44, 104) and buster_ripper/main.py (option decorators lines 13-61, `config.<NAME> = <param>` assignment block lines 70-86, the only `COMPACT_NUDGE_RATIO` occurrence at line 120); repo-wide symbol search for the unflagged names: compaction.py:57, routes/count_tokens.py:58, utils.py:126; other CLI surfaces: server.py:106-110 (argparse, `--host`/`--port` only)
Version: unavailable
Fingerprint: sha256:14ac40180f081d8e7caeb280d9a9ab194cbd06c93e3958ebf9bef362ca7b7aab
Fingerprint: sha256:b4794796d28ad3ab9ce5f2e0e03dddcfa882178299c7b969751697993cb76089
Invalidates-On: a `@click.option` wired to `config.KV_POLL_INTERVAL`, `config.COMPACT_NUDGE_RATIO` or `config.MIN_COMPLETION_TOKENS` appears in `main.py` (flips the unflagged half)
Invalidates-On: any of the eight `config.<NAME> = <param>` assignments at main.py:75-85 is removed, renamed or no longer fed by its option (flips the flagged half), or a numeric global is added to or removed from `buster_ripper/config.py` (changes the enumeration the partition is over)
Checked-At: 2026-09-19
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-48.md

---

Composition: CO:aiohttp-runner-to-on_cleanup/lifecycle-teardown
Members: CL:buster_ripper/signal-handler-ownership, CL:buster_ripper.server#on_cleanup/shutdown-dump-flush, CL:buster_ripper.server#spawn_kv_cache_monitor/background-task-lifecycle, CL:aiohttp/signal-shutdown, CL:aiohttp/cleanup-hook-error-propagation
Boundary: the SIGINT/SIGTERM lifecycle signal crossing aiohttp's runner into the app's cleanup hooks
Dimensions: lifecycle state (background-task cancellation, CancelledError swallowed); exception propagation (cleanup-hook raise replaces GracefulExit → traceback exit, so the flush is wrapped and must not raise); teardown order (cancel tasks → flush → close session); documented-path vacuity (nothing pending → flush no-ops)
Verdict: SAFE
Counterexample: none
Source: buster_ripper/server.py:25-34, :44-57, :60-72, :75-96, :99-102; buster_ripper/main.py:88-90, :123-126; installed aiohttp web.py:424-449, :452-521; web_runner.py:35-40, :278-289, :295-320, :386-399; web_app.py:168-174, :474-490, :594-620; aiosignal/__init__.py:44-52
Fingerprint: sha256:af5e5a3daeeff605fcd70c30e7ac3d4efa9d501c876ead2e64aaa90b9bebd050
Invalidates-On: buster_ripper/server.py digest changes (sha256:e008d3c971db787ebd1eb890544c6d73f0479bb97d0060ee64d4f77cf150b824) — above all a flush/compensation call added between the cancel-await step and the session close, a reordering of those steps, a second shutdown hook, or a change to the `except asyncio.CancelledError` guard; this seam must be recomposed when the flush lands (dimensions teardown-order and documented-path vacuity move from vacuous to substantive)
Invalidates-On: buster_ripper/dump.py (sha256:243b21434c14e6ea8144329fe1d0ac4e0c499ed9a7e6e09bdcac5e2417a56b1d), compaction.py (sha256:ac9d42143c98694a0b6672dd753afcef537fc5e03ec747b4bb3c1208e23535e7), routes/messages.py (sha256:9c234c4f183aa03a80ab82356a6fdfaad0220f3bef69edf3bab5789ef808277d) or main.py (sha256:b4794796d28ad3ab9ce5f2e0e03dddcfa882178299c7b969751697993cb76089) digest changes per the members' invalidation sets — especially a pending-buffer/flush function added to dump.py or any dump interaction added to `poll_kv_cache`'s cancellation path
Invalidates-On: the package-tree digest (sha256:60aebdf2191e7785586f3d1aa8fe0663136eadbeb76be53e072fefb4d8bf6381) changes — any `import signal`, `signal.signal(...)`, `loop.add_signal_handler(...)` or `KeyboardInterrupt` clause added under `buster_ripper/`, or the entry path leaving `web.run_app`
Invalidates-On: aiohttp digests (web.py sha256:27348d99e8e0e46e987850279088197f2b6a6d4f3ab0dbbce38c982a6a14a6ab, web_runner.py sha256:bf51b59ca88e3908059d3491e0831ce88f5178414331a1ed30bbcefeba0333e0, web_app.py sha256:94653f68030dfa1df0cbe2d34c78ba49e287f3276dd46e75057702b29803e590) or aiosignal (sha256:4c8926506f474c1b7875fab69c8498062662441db1c2f16d119c9d2cfd073cbe) change per the members' invalidation sets — especially handler install/removal timing in `BaseRunner.setup`/`cleanup`, the `run_app` finally's suppression scope, or `Signal.send` gaining exception handling
Invalidates-On: boundary changes — `handle_signals` no longer defaulted True on either entry call, `shutdown_timeout` semantics changed, Windows-targeted reuse, or a flush added anywhere on the shutdown path
Checked-At: 2026-09-19
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/compositions/seam-S2.md

---

Composition: CO:cli-option-to-dump-dispatch/format-value-crossing
Members: CL:buster_ripper/dump-format-selection, CL:rel:buster_ripper.main→buster_ripper.config/numeric-config-cli-wiring, CL:buster_ripper.main/dump-cli-surface, CL:buster_ripper.config/config-read-timing, CL:click/option-callback-timing, CL:click/choice-validation
Boundary: the configured format value crossing CLI option → `config.DUMP_FORMAT` global → `dump_turn`'s format dispatch
Dimensions: closed three-value set; default `documented` (no flag → documented layout, script path included); once-per-process assignment through the option's own callback; unknown configured value → `ValueError` (never silent fallback); `click.Choice` validation at the flag
Verdict: SAFE
Counterexample: none
Source: buster_ripper/dump.py:61-109 (hardcoded format branch :80-102, no format dispatch today); buster_ripper/main.py:12-61 (21 click options, no dump-format option; `:25` `--dump-dir`; `:46`/`:60` production `click.Choice` precedent), :70-86 (assignment block), :88-90, :123-126 (assignments precede create_app/run_app); buster_ripper/config.py:1-5, :16 (DUMP_DIR sole dump global, no DUMP_FORMAT); buster_ripper/routes/messages.py:55-59 (gate + swallow-all wrap); buster_ripper/server.py:52-53, :106-110 and routes/passthrough.py:19, :55 (call-time-read corroboration); installed click core.py (make_context/parse_args/consume_value/process_value/handle_parse_result/invoke/main regions) and types.py:136-143, :240, :336-358
Fingerprint: sha256:4ceec71bb33fa41405078e6dfbfa36d8ce2ca17b87ce1fcabb85df44568462ee
Invalidates-On: any member verdict superseded by a fresh record with different content (anchors, transformations, or cited sources) — including claim-21's newness fact (a `DUMP_FORMAT`-style symbol appearing in the package), claim-23's dump-option inventory, claim-34's call-time-read pattern, claim-40/41's click parse ordering/validation, claim-48's assignment-block/numeric-global enumeration
Invalidates-On: the planned chain lands in production — a `--dump-format` option in main.py, a `DUMP_FORMAT` global in config.py, or a format dispatch/`ValueError` branch in dump.py — post-implementation recomposition at Yield (this pre-implementation composition defers the chain-existence check)
Invalidates-On: the planned option is wired without a default while the callback assigns unconditionally — the no-flag path then feeds `None` to the callback (click UNSET→None, core.py:2441-2442) and `config.DUMP_FORMAT = None` trips the unknown-value branch at the first dump, failing the default-`documented` dimension
Invalidates-On: a decorator-argument import-time read of a `main()`-reassigned config value (e.g. `default=config.DUMP_FORMAT`)
Invalidates-On: `Choice` choices drifting from the dispatch mapping (a choice accepted at the flag with no dispatch arm, or a dispatch arm unreachable from the choice list), or the planned `documented` branch not reproducing the hardcoded turn-0/turn-N layout at dump.py:80-102, or the dump gate/catch-all at routes/messages.py:55-59 changing
Invalidates-On: click upgrade where `consume_value` no longer resolves the default before `process_value`, or where `Choice.convert` stops validating default/envvar/default-map sources
Checked-At: 2026-09-19T17:50:21Z
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/compositions/seam-S3.md

---

Composition: CO:serve-script-env-to-argv/env-var-flag-crossing
Members: CL:README.md/documented-run-path, CL:README.md/collapsed-format-run-path, CL:README.md/skip-tls-verify-run-path, CL:serve_buster_ripper.sh/usage-header-dump-dir-documentation, CL:bash/prefix-assignment-environment, CL:bash/colon-default-value-form-semantics, CL:bash/no-colon-default-value-form-semantics, CL:bash/nounset-default-value-exemption, CL:serve_buster_ripper.sh/dump-format-env-default, CL:serve_buster_ripper.sh/skip-tls-verify-env-default, CL:bash/dash-n-primary-semantics, CL:bash/errexit-and-list-exemption, CL:serve_buster_ripper.sh/dump-dir-flag-guard, CL:serve_buster_ripper.sh/dump-format-flag-guard, CL:serve_buster_ripper.sh/skip-tls-verify-flag-guard, CL:bash/unquoted-expansion-word-splitting, CL:serve_buster_ripper.sh/exec-line-option-assembly, CL:serve_buster_ripper.sh/positional-argument-forwarding, CL:serve_buster_ripper.sh/shebang-interpreter, CL:serve_buster_ripper.sh/exec-line-command-name, CL:bash/path-command-search-order, CL:README.md/dump-dir-example-set, CL:bash/quoted-array-expansion-word-identity, CL:bash/array-alternate-value-word-contribution, CL:bash/array-alternate-value-nounset-exemption
Boundary: env-var values (`DUMP_FORMAT` landed with T005; `SKIP_TLS_VERIFY` landed with T006; `DUMP_DIR`, whose unset-only `-` default form landed with T011) crossing shell parameter expansion (`serve_buster_ripper.sh:22-30`) → the `EXTRA_FLAGS` array accumulator (`:36-42`, per-value appends since T012, each boundary value interpolated quoted) → the `exec` line's guarded array expansion at `:54`, terminating in the binary's argv word set
Dimensions: empty-default guard (`DUMP_FORMAT`); boolean `== "1"` value-guard append (`SKIP_TLS_VERIFY`, the `STRIP_DATE`/`VERBOSE` peer shape) versus conditional presence-guard appends (`DUMP_FORMAT` in the `DUMP_DIR` peer's `-n` shape); two argv channels (exec-line always-on quoted trio + conditional `EXTRA_FLAGS` elements); no positional forwarding; README run path with no-format, collapsed and skip-TLS examples; unset versus set-but-empty discriminator (the environment-prefix assignment form the documented disable uses); `:-` versus `-` default form as the step that keeps or discards that discriminator before the guard consumes it; presence-guard false branch under `set -e` (the previously unreachable arm); zero-word arm when every append is suppressed (the `${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}` guard at line 54 — 203 for the word contribution, 204 for nounset safety, 189 and 188 corroborating); value grammar at the boundary (202: one argument word per array element, so `$IFS` whitespace, glob metacharacters and a leading dash are immaterial to the word set); header-documented disable form versus the README example inventory
Verdict: SAFE
Counterexample: none
Source: serve_buster_ripper.sh (repo root, working tree, 54 lines, 2566 bytes, sha256:c58256fbec793c520720c9e86fee8740653939275b508f50b212ab60bd0908af) — line 1 `#!/usr/bin/env bash`; line 2 `set -euo pipefail`; lines 11-16 usage header with the `DUMP_DIR=""` disable form at 14; lines 22-30 parameter-expansion defaults (22 `UPSTREAM`, 23 `HOST`, 24 `PORT`, 25 `STRIP_DATE`, 26 `VERBOSE`, 27 `SKIP_TLS_VERIFY=${SKIP_TLS_VERIFY:-0}`, 28 `DUMP_DIR=${DUMP_DIR-${HOME}/mllm/prompt-diffs}` — the post-T011 no-colon form, 29 `DUMP_FORMAT=${DUMP_FORMAT:-}`, 30 `PRESERVED_THINKING=${PRESERVED_THINKING:-1}`); line 36 `EXTRA_FLAGS=()`; lines 37-42 guarded appends (37 `--strip-date`, 38 `--verbose`, 39 `--skip-tls-verify`, 40 `-n "${DUMP_DIR}"` → `--dump-dir "${DUMP_DIR}"`, 41 `-n "${DUMP_FORMAT}"` → `--dump-format "${DUMP_FORMAT}"`, 42 `--preserved-thinking`); line 44 `echo`; lines 46-49 guard-rationale comment; lines 50-54 the single `exec` — bare `buster-ripper` command word at 50, quoted always-on trio at 51-53, `${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}` at 54; whole-file census: `exec` once, `EXTRA_FLAGS` at 36/37/38/39/40/41/42/54 only, the six appended option strings once each, no `unset` statement, no `IFS=`, `set -f`, `shopt`, `declare`/`typeset`/`local`, `shift`, `$@`, `$*` or `$N`. README.md (repo root, working tree, 289 lines, 10287 bytes, sha256:fe80cb9acabfe699fac070a69b43ac425436606d627b243825d4a028d75a1c15) — serve-script invocations at 49, 75, 78, 81, 84, 87, 92, 145, 192 (all `bash …`), direct CLI form at 194 (outside the seam), collapsed-format prose 169-184; disable-idiom probes (`DUMP_DIR=""` in either quote style, `--dump-dir ""`, `:`/`0`/`off`/`false`/`null`/`none`) all empty, no `PRESERVED_THINKING` mention. GNU Bash documentation and source, reached through the member verdicts' own citations and not re-fetched by the composition pass: `doc/bashref.texi` at tag `bash-5.3` `f3d37d57…` (183-186, 203, 204) with `bash-5.2` `32ced60b…` corroboration, the consulted Shell Parameter Expansion excerpt `13669991…`, Bash-Conditional-Expressions `401cf8ec…`, The-Set-Builtin `ca70b448…`, Command-Search-and-Execution `8182c7cb…`, Debian `bash(1)` `a827e4d7…`, the man7 `bash(1)` copy `e35fa296…`/`b4455651…` (189, 202), `subst.c` at `bash-5.3` `cf96a7f3…` with `master`/`bash-5.1` cross-checks (186), and `CHANGES` at `bash-4.4` `992ae56e…` (204's version boundary).
Fingerprint: sha256:cb7c0a5c82e5ac1b5f33738a61b582d925be4502a51a6ed1484e2ab415b60631 (handle-bearing: a 3759-byte 25-line header of `CL:<handle> sha256:<digests recorded in that verdict's ## Source Fingerprint>` in ledger order 33, 164, 181, 191, 185, 183, 184, 186, 159, 179, 187, 188, 161, 160, 180, 189, 162, 163, 192, 193, 190, 194, 202, 203, 204, then the serve_buster_ripper.sh bytes (2566) then the README.md bytes (10287); 16612 bytes total, no FINGERPRINT-MISSING row)
Invalidates-On: any change to a `## Source Fingerprint` recorded in the 25 member verdicts (re-match, not assumed: 183/184/185 against `f3d37d57…`/`13669991…`/`a827e4d7…` — `mirror/bash` master is mutable; 186 against `cf96a7f3…`, whose nounset exemption has no documentation behind it, so only the source gate clears it; 187 `401cf8ec…`, 188 `ca70b448…`, 190 `8182c7cb…`; 189 and 202 against `e35fa296…`/`b4455651…`, a periodically re-synced upstream mirror; 203 and 204 against `f3d37d57…` plus `32ced60b…` and, for 204, `992ae56e…`)
Invalidates-On: the interpreter-version scope limit that is the one clause able to reopen dimension 9 — this trace is established on bash-5.3/5.2 documentation and on 204's recorded bash-4.4-era onset of the nounset array exemption, and no member claim identifies which binary `/usr/bin/env bash` (line 1) resolves to on any platform (192 states line 1; 190 governs bash's own `$PATH` search, not `env`'s). Recompose dimension 9 and re-check 203's/204's version scope if a member claim or recorded deployment fact puts the resolved interpreter before bash-4.4, if 204's `-u` clause is absent from that interpreter's manual, or if a version-scoped premise for `${array[@]+word}` on bash 3.2 joins this seam. The script's own comment at lines 46-49 asserts bash 3.2 on macOS; that is production prose, not a verified contract, and this record does not read it as an answer
Invalidates-On: either production source off its digest — serve_buster_ripper.sh no longer sha256:c58256fbec793c520720c9e86fee8740653939275b508f50b212ab60bd0908af, or README.md no longer sha256:fe80cb9acabfe699fac070a69b43ac425436606d627b243825d4a028d75a1c15 — either move requires a full re-trace, not a scoped one
Invalidates-On: any change to line 54's expansion form — the bare `"${EXTRA_FLAGS[@]}"`, an `[*]` variant, quoting removed, or the guard dropped — fires 162 and re-traces dimensions 3 and 9 (the bare form reopens the pre-4.4 nounset question now outside this trace; `[*]` collapses the word count against 202)
Invalidates-On: line 36 back to a scalar (`EXTRA_FLAGS=""`), or lines 40/41 moving from a quoted `+=` pair to an unquoted or string append — fires 161/160/163 and re-traces dimensions 2, 3, 4, 9 and 10, reopening the value-word-splitting class as a traced behavior
Invalidates-On: line 28 back to `${DUMP_DIR:-…}`, any change that substitutes on null, or removal of the default — fires 161's third clause and reopens the unset-versus-set-but-empty counterexample (dimensions 1-3 and 6-11 re-trace)
Invalidates-On: line 40's or 41's guard shape (`-n` → `-z`, `== "1"`, `:+`-style, unconditional append), a second `--dump-dir`/`--dump-format` site, line 27's operator or default literal (fires 179), line 39's `== "1"` comparison (fires 180), or line 29's default (fires 159) — re-traces dimensions 1, 2, 6, 7 and 8
Invalidates-On: an `unset EXTRA_FLAGS` or `unset EXTRA_FLAGS[i]` statement entering the file, or any `+=`/indexed assignment able to put an empty string into the array — both change dimension 9 and 3: 203's own invalidation records that unsetting the last element does not unset the array (unreachable today only because no `unset` exists), and a set-but-null element contributes one empty word per 202, a materially different argv from the zero-word arm
Invalidates-On: line 30's `PRESERVED_THINKING` default moving off `1` or line 42's guard changing — makes the memberless dimension-9 arm the default arm rather than a two-override one
Invalidates-On: line 2's `set -euo pipefail` losing `pipefail`, `errexit` or `nounset`, or any `IFS=` assignment, `set -f`, or `shopt` change, or a quoting change to the trio at 51-53 — re-traces dimensions 3, 4, 6, 8, 9 and 10 (188's exemption and 186's and 204's nounset premises all cite line 2)
Invalidates-On: line 50's command word gaining a path, variable or launcher prefix (fires 193/162), line 1's shebang changing interpreter (fires 192), or any `$@`/`$*`/`$N`/`shift` reference entering the script (fires 163, and dimension 4's carried premise) — moves the terminal step and the applicability of every `lang:bash` premise
Invalidates-On: documentation movement — any edit to the header block 11-16 changing, removing, rewording or relocating the line 14 `DUMP_DIR=""` example or its "disable prompt diffing" annotation (fires 191, dimension 11); any README edit adding a `DUMP_DIR=""`/`--dump-dir ""` example or moving lines 84/145/192 or the skip-TLS block 89-92 (fires 33/164/181/194, dimensions 5 and 11); any documented invocation changing from `bash serve_buster_ripper.sh` to `sh …`/`./…`/`zsh …`, which invalidates the interpreter scoping underpinning dimensions 3-10 since 187/189/190/202/203/204 exclude non-bash shells by their own terms
Invalidates-On: member-set movement — adding or dropping a member, or re-extracting any of the 25 rows, changes this basis by construction (the header is the member list); carrying dimensions 1-2 and 4-8 and 10-11 forward requires those dimensions' members still pinned to the digests above, and bash semantics drift in the `:-`/`-` unset-or-null versus unset-only rule, the `PF_IGNUNBOUND` gate in `parameter_brace_expand`, null-valued prefix assignments entering the command environment, `-n` truth, the `-e` `&&`-list exemption, implicit-null-argument removal, `[@]` one-word-per-element expansion, the colon-less `+` array-setness test, or `$PATH` first-match resolution requires recomposition
Checked-At: 2026-10-05T05:19:41Z
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/compositions/seam-S6.md

---

Claim: CL:aiohttp/request-json-decoding
Statement: aiohttp's `Request.json()` decodes the request body using the charset from the Content-Type header (default utf-8) before parsing, so bytes undecodable under that charset raise `UnicodeDecodeError`, not `json.JSONDecodeError`
Subject: pkg:aiohttp@3.13.3
Verdict: HOLDS
Actual: `Request.json()` awaits `text()` before passing its string to `json.loads`; `text()` decodes bytes using the Content-Type charset or UTF-8 with strict errors, so undecodable input raises `UnicodeDecodeError` before parsing.
Source: `/Users/alexhubitski/Projects/buster-ripper/.venv/lib/python3.13/site-packages/aiohttp/web_request.py` lines 686–695; corroborated by `aiohttp/helpers.py` lines 769–776 and `aiohttp/typedefs.py` line 21
Version: aiohttp 3.13.3 (installed in project `.venv`, Python 3.13)
Fingerprint: sha256:2aaacea7a01e581e5eead2ab1b9e4ba3b65bc2ae3d0f176b2a78ae5b6b76bb4e
Invalidates-On: aiohttp version or installed `Request.json()`/`Request.text()` decode implementation changes
Invalidates-On: aiohttp charset resolution changes so Content-Type no longer drives decoding
Checked-At: 2026-09-22
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-51.md

---

Claim: CL:buster_ripper.routes/client-json-parse-error-response
Statement: `messages.py:28-29` and `chat.py:30-31` return a typed 400 (`{"error": "Invalid JSON"}`) when the client body fails JSON parsing — the precedent the fix mirrors
Subject: repo:buster_ripper.routes
Verdict: HOLDS
Actual: Both handlers catch `json.JSONDecodeError` from `await request.json()` and return `web.json_response({"error": "Invalid JSON"}, status=400)`.
Source: `buster_ripper/routes/messages.py:28-29` and `buster_ripper/routes/chat.py:30-31`
Version: repo working tree, 2026-09-22 (no VCS resolution performed; files as read)
Fingerprint: sha256:9c234c4f183aa03a80ab82356a6fdfaad0220f3bef69edf3bab5789ef808277d
Fingerprint: sha256:33cde288e27964a33edf77c448d1b34444efa3e9a474739fc4bd3cfa25e16698
Invalidates-On: Either handler's parse-error status, payload, or caught exception changes
Checked-At: 2026-09-22
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-54.md

---

Claim: CL:buster_ripper.routes.count_tokens#handle_count_tokens/client-json-parse-error-response
Statement: `count_tokens.py:32-33` swallows the client-body parse failure and proceeds with an empty body instead of returning a typed 4xx — the family's divergent sibling
Subject: repo:buster_ripper.routes.count_tokens#handle_count_tokens
Verdict: HOLDS
Actual: `except json.JSONDecodeError` assigns `body = {}` and continues through token estimation to a 200 JSON response rather than returning a 4xx.
Source: `buster_ripper/routes/count_tokens.py:32-33`, terminal 200 at line 76
Version: repo working tree, 2026-09-22 (no VCS resolution performed; file as read)
Fingerprint: sha256:1b43334494f0a5b3a53e95ab162e21b935ad248d5904d551726493d600db96f1
Invalidates-On: The handler's parse-failure branch or subsequent empty-body response changes
Checked-At: 2026-09-22
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-55.md

---

Claim: CL:buster_ripper.routes.chat#_make_chat_request/upstream-json-parse-error-response
Statement: `chat.py:146` returns a typed 502 ("Invalid JSON from upstream") when the upstream response fails JSON parsing — the upstream-attributed sibling of the parse boundary
Subject: repo:buster_ripper.routes.chat#_make_chat_request
Verdict: HOLDS
Actual: `await resp.json()` at line 131 can raise `json.JSONDecodeError`, caught at line 146; line 147 returns `{"error": "Invalid JSON from upstream"}` with status 502. Content-type errors instead take the `aiohttp.ClientError` branch.
Source: `buster_ripper/routes/chat.py:146-147` (exception raised from `await resp.json()` at line 131)
Version: repo working tree, 2026-09-22; aiohttp 3.13.3 installed at `.venv/lib/python3.13/site-packages/aiohttp/`
Fingerprint: sha256:33cde288e27964a33edf77c448d1b34444efa3e9a474739fc4bd3cfa25e16698
Invalidates-On: The parse-error response in `_make_chat_request` changes
Invalidates-On: aiohttp `ClientResponse.json()` stops propagating `json.JSONDecodeError` on malformed JSON
Checked-At: 2026-09-22
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-56.md

---

Claim: CL:aiohttp/per-request-timeout-override
Statement: aiohttp: a per-request `timeout=` argument to a `ClientSession` request method (`get`/`post`/`request`) replaces the session's default `ClientTimeout` for that single request
Subject: pkg:aiohttp@3.13.3
Verdict: HOLDS
Actual: `ClientSession._request` selects the timeout object wholesale — `self._timeout` is used only when the argument is left at the `sentinel` default, and any `ClientTimeout` passed at request time is assigned as-is (`real_timeout = timeout`) with no per-field fallback to the session value. Every downstream deadline is then read off that replacement object alone: `TimeoutHandle(loop, real_timeout.total)`, `read_timeout=real_timeout.sock_read`, and the `ClientTimeout` handed to `connector.connect(..., timeout=real_timeout)` for `timeout.connect`/`timeout.sock_connect`. An all-unset request-time `ClientTimeout()` therefore leaves no timer armed (`TimeoutHandle.start()` returns `None`, `.timer()` returns `TimerNoop`, `ceil_timeout(None)` is `asyncio.timeout(None)`, and `TCPConnector` carries no independent connect timeout to fall back on in 3.13.3) — the session's armed `total` is silently discarded, not partially honoured. A non-`ClientTimeout` value is converted to `ClientTimeout(total=timeout)` and an explicit `timeout=None` becomes `ClientTimeout(total=None)`; both take the same replacing branch, so either also disables the session's total deadline. The library documents replacement at `client.py:217-220`, telling callers to create a completely new object or use `attr.evolve`. Session construction stores the supplied object verbatim as `self._timeout`, so no merge with `DEFAULT_TIMEOUT` occurs at request time either. Scope checked: covers `ClientSession.request/get/post/...` and the redirect loop (one `TimeoutHandle` created before it); `ws_connect`'s `timeout` is a `ClientWSTimeout`, a different object and call, so it is not a counterexample.
Source: installed `.venv/lib/python3.13/site-packages/aiohttp/client.py` :584-596 (decisive — the `sentinel` branch and `real_timeout = timeout`), :202-224 (`ClientTimeout` field defaults and the in-source replacement note), :330-352 (session stores `self._timeout` verbatim); corroborated by `.venv/lib/python3.13/site-packages/aiohttp/helpers.py` :609-625 (`TimeoutHandle.start`/`timer`), :724-735 (`ceil_timeout`) and `.venv/lib/python3.13/site-packages/aiohttp/connector.py` :650, :1295-1297, :1332-1334. Corroboration only (newer than installed): `https://docs.aiohttp.org/en/stable/client_reference.html`, `ClientSession.request`'s `timeout` parameter — "override the session's timeout".
Version: aiohttp 3.13.3 (`__version__ = "3.13.3"` in `.venv/lib/python3.13/site-packages/aiohttp/__init__.py`)
Fingerprint: sha256:7ce41fc1c2142f5346015455e030e3ebec22a41cde0fc299a66ce13be2cb2a9e (client.py); sha256:435df4ecf0849d6cf844ff1cad4c3c764e7c734605d848b7272c2429f471cf91 (helpers.py); sha256:bd3db604db820ed6da744d54abb1c2ef3a4e5821ccc48e27dcfb42cfbcd9664c (connector.py)
Invalidates-On: an aiohttp version bump whose `client.py`, `helpers.py` or `connector.py` sha256 differs from the digests above
Invalidates-On: `ClientSession._request`'s timeout selection at `client.py:584-590` changing from whole-object assignment to a field-wise merge (e.g. `attr.evolve(self._timeout, **set_fields)`), or `sentinel` being redefined so a request-time `ClientTimeout` can mean "unset"
Invalidates-On: `ClientTimeout` defaults at `client.py:202-207` arming a field (a non-`None` `total` default on the class itself, as `DEFAULT_TIMEOUT` does at :224), so `ClientTimeout()` is no longer "all components unset"
Invalidates-On: an independent connect timeout reintroduced on `BaseConnector`/`TCPConnector`, or a connector-level fallback at `connector.py:650`/`_wrap_create_connection` when `timeout.connect`/`timeout.sock_connect` is `None`
Invalidates-On: `TimeoutHandle.start()`/`.timer()` (`helpers.py:609-625`) or `ceil_timeout()` (`helpers.py:724-735`) arming a deadline for a `None` value
Checked-At: 2026-10-06
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-57-r2.md

---

Claim: CL:aiohttp/response-json-content-type-validation
Statement: aiohttp's `resp.json()` raises `ContentTypeError` when the response's Content-Type is not `application/json` — e.g. `text/event-stream` (SSE)
Subject: pkg:aiohttp@3.13.3
Verdict: DOES-NOT-HOLD
Actual: Default `ClientResponse.json()` rejects `text/event-stream`, but accepts `application/problem+json`; callers may override or disable content-type checking via `content_type`, so the categorical assertion is false.
Source: Installed `.venv/lib/python3.13/site-packages/aiohttp/client_reqrep.py`, lines 102, 260–265, 742–773.
Version: aiohttp 3.13.3
Fingerprint: sha256:7844440eb67433c6454edd708c776e47e3fcfec9b8d0aeb980dcfe88c1986ac9
Invalidates-On: Change to installed aiohttp `client_reqrep.py` content-type matching or `ClientResponse.json()` defaults/validation.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-63.md

---

Claim: CL:aiohttp/content-type-error-inheritance
Statement: `aiohttp.ContentTypeError` is a subclass of `aiohttp.ClientError`
Subject: pkg:aiohttp@3.13.3
Verdict: HOLDS
Actual: `ContentTypeError` inherits `ClientResponseError`, which inherits `ClientError`.
Source: .venv/lib/python3.13/site-packages/aiohttp/client_exceptions.py:61-65,144
Version: aiohttp 3.13.3
Fingerprint: sha256:bb229bc48da979984a97b9442c1331dd47aeb0d91fa5e98f5a9185ab4afa25e3
Invalidates-On: Change to installed aiohttp version or `client_exceptions.py` inheritance declarations.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-64.md
---

Claim: CL:buster_ripper.routes.chat#_make_chat_request/upstream-response-parsing
Statement: `chat.py`'s `_make_chat_request` parses every upstream response with `await resp.json()` (`chat.py:131`) regardless of the request's `stream` flag
Subject: repo:buster_ripper.routes.chat#_make_chat_request
Verdict: HOLDS
Actual: Every upstream response entered through `session.post` is passed to `await resp.json()` before status handling or transformation; the function does not inspect the stream flag.
Source: `buster_ripper/routes/chat.py:114-149` (especially line 131)
Version: unavailable
Fingerprint: sha256:33cde288e27964a33edf77c448d1b34444efa3e9a474739fc4bd3cfa25e16698
Invalidates-On: Streaming-specific handling or response parsing changes in `_make_chat_request`
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-65.md

---

Claim: CL:buster_ripper.routes.messages#handle_streaming/upstream-body-relay
Statement: `messages.py`'s streaming handler relays the upstream body raw — a prepared `StreamResponse` fed by `resp.content.iter_any()` chunks written through unchanged
Subject: repo:buster_ripper.routes.messages#handle_streaming
Verdict: HOLDS
Actual: On successful upstream responses, `_stream_single_request` prepares a `StreamResponse` and writes each `resp.content.iter_any()` chunk unchanged; error responses take a separate path.
Source: `buster_ripper/routes/messages.py:135-176`
Version: unavailable
Fingerprint: sha256:9c234c4f183aa03a80ab82356a6fdfaad0220f3bef69edf3bab5789ef808277d
Invalidates-On: The streaming handler's upstream chunk relay or error-response path changes
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-67.md

---

Claim: CL:python/json-decode-error-inheritance
Statement: `json.JSONDecodeError` derives directly from `ValueError`
Subject: lang:python
Verdict: HOLDS
Actual: The installed standard-library decoder declares `class JSONDecodeError(ValueError):`, with `ValueError` as its sole direct base.
Source: /opt/homebrew/Cellar/python@3.14/3.14.7/Frameworks/Python.framework/Versions/3.14/lib/python3.14/json/decoder.py:20
Version: Python 3.14.7 (installed Homebrew source)
Fingerprint: sha256:dc98d3e677612ea7311303f06a144cec6b91ebadc7ec037d1777501647578a07
Invalidates-On: The installed json/decoder.py file changing bytes.
Invalidates-On: Using a different Python implementation or standard-library version.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-70.md

---

Claim: CL:buster_ripper.routes.messages#handle_messages/stream-dispatch
Statement: `handle_messages` dispatches to `handle_streaming` or `handle_non_streaming` according to `body.get("stream", False)`
Subject: repo:buster_ripper.routes.messages#handle_messages
Verdict: HOLDS
Actual: After normalization and request preparation, it assigns `stream = body.get("stream", False)` and awaits `handle_streaming` when truthy, otherwise `handle_non_streaming`.
Source: `buster_ripper/routes/messages.py:65-70`
Version: unavailable
Fingerprint: sha256:9c234c4f183aa03a80ab82356a6fdfaad0220f3bef69edf3bab5789ef808277d
Invalidates-On: Any change to `buster_ripper/routes/messages.py` affecting `handle_messages` dispatch.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-73.md

---

Claim: CL:buster_ripper.main/cli-default-binding-time
Statement: The `click.option` defaults in `buster_ripper/main.py` that refer to `config` attributes are bound when `main.py` is imported
Subject: repo:buster_ripper.main
Verdict: HOLDS
Actual: Exactly three `click.option` defaults reference `config` attributes — `default=config.UPSTREAM` (line 62), `default=config.DUMP_FLUSH_INTERVAL` (line 87), and `default=config.DUMP_FLUSH_BYTES` (line 92) — and all three sit inside module-level decorator expressions on `main`, so Python evaluates them once when the module body executes, i.e. at import time. The resulting values are stored on the click Option objects at decoration; later mutations of `config.UPSTREAM`/`config.DUMP_FLUSH_INTERVAL`/`config.DUMP_FLUSH_BYTES` after import would not change the CLI defaults. (Other config references in the file — `type=click.Choice(list(config.DUMP_FORMATS))`, `type=click.Choice(list(config.PROFILES.keys()))`, and the `--eval-profile` help f-string — are `type`/`help`, not `default`, and are likewise import-time evaluated.)
Source: buster_ripper/main.py lines 62, 86-95 (decorated `main` at lines 61-131); corroborated by buster_ripper/config.py lines 13, 36, 37 (module-level `UPSTREAM`, `DUMP_FLUSH_INTERVAL`, `DUMP_FLUSH_BYTES`)
Version: unavailable (working tree; fingerprint below)
Fingerprint: sha256:fd36dc12055f67693d9b852fb15a66aa4a3a7864103ac3de0ee67d9520048d47 (buster_ripper/main.py); corroborating buster_ripper/config.py sha256:22348c68d21b91a30e75eeea802d4d390b2bcdabbf974b0ddc743c9354635522
Invalidates-On: Any of the three `default=config.*` expressions is replaced with a literal, a callable, or a lookup deferred to parse time
Invalidates-On: `config.UPSTREAM` / `DUMP_FLUSH_INTERVAL` / `DUMP_FLUSH_BYTES` is renamed or moved off the `config` module's top level
Invalidates-On: The options are relocated from module-level decorators into a factory invoked at run time rather than import time
Checked-At: 2026-09-28
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-74.md

---

Claim: CL:buster_ripper.dump/format-dispatch-read-timing
Statement: The dump-format dispatch under `buster_ripper/` reads `config.DUMP_FORMAT` when processing a turn
Subject: repo:buster_ripper.dump
Verdict: HOLDS
Actual: `dump_turn` calls `_resolve_format()` for each turn, which reads `config.DUMP_FORMAT`; it then dispatches to the documented, change-focused, or self-contained writer based on that value.
Source: `buster_ripper/dump.py:83-98,128-133` (production source)
Version: unavailable
Fingerprint: sha256:c8f093c89b61c355ac182feb070cd90b84924ca0d187332117d76b5c766ad9b9
Invalidates-On: Change to `buster_ripper/dump.py` affecting `dump_turn` or `_resolve_format`.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-75.md

---

Claim: CL:buster_ripper/periodic-flush-interval-read-timing
Statement: The periodic dump-flush trigger under `buster_ripper/` reads its configured interval at runtime
Subject: repo:buster_ripper
Verdict: HOLDS
Actual: `poll_flush()` repeatedly flushes and then evaluates `config.DUMP_FLUSH_INTERVAL` as the argument to `asyncio.sleep`, rather than capturing its value when the module loads. The interval is read on every loop iteration.
Source: `buster_ripper/dump.py:114-125`
Version: unavailable
Fingerprint: sha256:c8f093c89b61c355ac182feb070cd90b84924ca0d187332117d76b5c766ad9b9
Invalidates-On: Changes to `buster_ripper/dump.py` affecting `poll_flush()` or its configuration access.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-76.md

---

Claim: CL:buster_ripper.backend#detect_backend/probe-exception-fallback
Statement: `detect_backend` returns `"unknown"` when an `Exception` is raised inside its probe `try` block
Subject: repo:buster_ripper.backend#detect_backend
Verdict: HOLDS
Actual: The `except Exception` handler sets `_backend_type` to `"unknown"`; after caching the result, the function returns `_backend_type`.
Source: `buster_ripper/backend.py:53-71`
Version: unavailable
Fingerprint: sha256:c2b71468af6b53efde753b0648d47eb6e06c8db4c738e08f7c01f4e4682d76ef
Invalidates-On: Changes to `detect_backend`'s exception handler or return path in `buster_ripper/backend.py`.
Checked-At: 2026-09-28T00:13:19Z
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-77.md

---

Claim: CL:python/cancelled-error-inheritance
Statement: `asyncio.CancelledError` is not a subclass of `Exception`
Subject: lang:python
Verdict: HOLDS
Actual: The installed Python 3.14.7 standard library defines `CancelledError` directly as a subclass of `BaseException`, not `Exception`.
Source: `/opt/homebrew/Cellar/python@3.14/3.14.7/Frameworks/Python.framework/Versions/3.14/lib/python3.14/asyncio/exceptions.py:10`
Version: Python 3.14.7
Fingerprint: sha256:a5971f88be14cd1417d59adf539ae48c5d818f95362a4e0eb00017e3690ab37b
Invalidates-On: Change to the installed `asyncio/exceptions.py` definition of `CancelledError` or a switch to a Python version with a different base class (notably pre-3.8).
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-78.md

---

Claim: CL:buster_ripper.server#on_startup/backend-detection-startup-order
Statement: `on_startup` awaits `detect_backend` before the app serves requests
Subject: repo:buster_ripper.server#on_startup
Verdict: HOLDS
Actual: `create_app` registers `on_startup` as an aiohttp startup callback; the callback directly awaits `backend.detect_backend` before it returns. aiohttp starts a serving site only after application startup completes, so request serving waits for this call to complete (or fails startup if it raises).
Source: `buster_ripper/server.py:55-71,102-105` (production implementation); aiohttp documentation, https://docs.aiohttp.org/en/stable/web_advanced.html#aiohttp-web-app-runners (runner setup precedes site start) and https://docs.aiohttp.org/en/stable/web_advanced.html#signals (startup handlers are asynchronous).
Version: Repository source revision unavailable; aiohttp dependency unpinned in `pyproject.toml`.
Fingerprint: sha256:fbd579a7995d0f4a2ae8a6d2994bb9e5fda344c1a900ec581caee38f599647f8
Invalidates-On: `buster_ripper/server.py` changes the startup callback registration or replaces the direct await with a spawned task.
Invalidates-On: aiohttp changes its startup-to-serving lifecycle ordering.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-79.md

---

Claim: CL:buster_ripper.backend#detect_backend/cli-override-probe-bypass
Statement: `detect_backend` returns a CLI backend override without issuing a `/v1/models` GET
Subject: repo:buster_ripper.backend#detect_backend
Verdict: HOLDS
Actual: When `cli_override` is truthy, it stores and returns that value before reaching the `session.get` probe.
Source: `buster_ripper/backend.py:41-55`
Version: unavailable
Fingerprint: sha256:c2b71468af6b53efde753b0648d47eb6e06c8db4c738e08f7c01f4e4682d76ef
Invalidates-On: Changes to the CLI override branch or probe ordering in `buster_ripper/backend.py`.
Checked-At: 2026-09-28T00:13:19Z
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-80.md

---

Claim: CL:buster_ripper.backend#detect_backend/cached-backend-probe-bypass
Statement: `detect_backend` returns its cached backend without issuing a `/v1/models` GET
Subject: repo:buster_ripper.backend#detect_backend
Verdict: HOLDS
Actual: When there is no truthy CLI override and `_backend_state_cached` is true, the function returns `_backend_type` before reaching `session.get`.
Source: `buster_ripper/backend.py:41-55`
Version: unavailable
Fingerprint: sha256:c2b71468af6b53efde753b0648d47eb6e06c8db4c738e08f7c01f4e4682d76ef
Invalidates-On: Changes to the cached-state branch or probe ordering in `buster_ripper/backend.py`.
Checked-At: 2026-09-28T00:13:19Z
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-81.md

---

Claim: CL:buster_ripper.backend#detect_backend/uncached-backend-probe
Statement: `detect_backend` issues a `/v1/models` GET when neither a CLI override nor a cached backend is available
Subject: repo:buster_ripper.backend#detect_backend
Verdict: HOLDS
Actual: When the truthy CLI override and cached-state early returns do not apply, the function calls `session.get(f"{backend_url}/v1/models", ...)` in its probe `try` block.
Source: `buster_ripper/backend.py:41-56`
Version: unavailable
Fingerprint: sha256:c2b71468af6b53efde753b0648d47eb6e06c8db4c738e08f7c01f4e4682d76ef
Invalidates-On: Changes to the early returns or probe call in `buster_ripper/backend.py`.
Checked-At: 2026-09-28T00:13:19Z
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-82.md

---

Claim: CL:aiohttp/sse-json-content-type-validation
Statement: With its default expected JSON content type, aiohttp `ClientResponse.json()` raises `ContentTypeError` for a `text/event-stream` response
Subject: pkg:aiohttp@3.13.3
Verdict: HOLDS
Actual: Its default expected type is `application/json`; `text/event-stream` does not match the JSON MIME regex, so after reading the response body it raises `ContentTypeError` at the content-type check (assuming the read succeeds).
Source: .venv/lib/python3.13/site-packages/aiohttp/client_reqrep.py:102,260-265,742-765
Version: aiohttp 3.13.3
Fingerprint: sha256:7844440eb67433c6454edd708c776e47e3fcfec9b8d0aeb980dcfe88c1986ac9
Invalidates-On: Change to installed aiohttp version or `client_reqrep.py` JSON MIME matching or response JSON handling.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-83.md
---

Claim: CL:aiohttp/structured-suffix-json-content-type-validation
Statement: With its default expected JSON content type, aiohttp `ClientResponse.json()` accepts `application/problem+json`
Subject: pkg:aiohttp@3.13.3
Verdict: HOLDS
Actual: The default expected type `application/json` uses a regex that matches `application/problem+json`, so this MIME type passes the content-type check; subsequent decoding still depends on a valid response body.
Source: .venv/lib/python3.13/site-packages/aiohttp/client_reqrep.py:102,260-265,742-772
Version: aiohttp 3.13.3
Fingerprint: sha256:7844440eb67433c6454edd708c776e47e3fcfec9b8d0aeb980dcfe88c1986ac9
Invalidates-On: Change to installed aiohttp version or `client_reqrep.py` JSON MIME matching or response JSON handling.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-84.md
---

Claim: CL:buster_ripper/size-flush-threshold-read-timing
Statement: The size-based dump-flush trigger under `buster_ripper/` reads its configured byte threshold at runtime
Subject: repo:buster_ripper
Verdict: HOLDS
Actual: For collapsed dump formats, `_append_record` adds the rendered text's UTF-8 byte length to the session's pending count and compares it directly to `config.DUMP_FLUSH_BYTES` on each append, calling `flush_all()` when reached. The documented format does not use this buffered trigger.
Source: `buster_ripper/dump.py:189-195` (production source)
Version: unavailable
Fingerprint: sha256:07b0b076552f72e62c625caade1b01821b3237375900233812d14c0f275b3c2b
Invalidates-On: Change to `buster_ripper/dump.py` that alters `_append_record`'s threshold lookup or comparison.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-85.md


---

Claim: CL:buster_ripper.dump#_write_documented/documented-turn-write
Statement: `_write_documented` writes a turn file and appends one index line per turn
Subject: repo:buster_ripper.dump#_write_documented
Verdict: HOLDS
Actual: On each successful invocation, it writes `turn_000.json` for turn zero or `turn_{N:03d}.diff` thereafter, then appends exactly one summary line to `index.txt`. An intervening write error can prevent the append.
Source: `buster_ripper/dump.py:150-186` (production source)
Version: unavailable
Fingerprint: sha256:c8f093c89b61c355ac182feb070cd90b84924ca0d187332117d76b5c766ad9b9
Invalidates-On: Changes to `buster_ripper/dump.py`, especially `_write_documented`, `_write_full_turn`, `_write_diff_turn`, or `_append_index_line`.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-87.md


---

Claim: CL:buster_ripper.dump#_write_full_turn/turn-zero-record-form
Statement: `_write_full_turn` writes turn 0's body as indented, non-ASCII-preserving JSON to `turn_000.json`
Subject: repo:buster_ripper.dump#_write_full_turn
Verdict: HOLDS
Actual: `_write_full_turn` targets `state.session_dir / "turn_000.json"` and writes `json.dumps(body, indent=2, ensure_ascii=False)` as UTF-8. Its caller `_write_documented` invokes it when `state.turn == 0`.
Source: `buster_ripper/dump.py:150-165`
Version: unavailable
Fingerprint: sha256:c8f093c89b61c355ac182feb070cd90b84924ca0d187332117d76b5c766ad9b9
Invalidates-On: Change to `buster_ripper/dump.py`'s `_write_documented` turn-0 dispatch or `_write_full_turn` serialization and target path.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-88.md


---

Claim: CL:buster_ripper.dump#_write_diff_turn/turn-diff-record-form
Statement: `_write_diff_turn` writes later turns' unified rendered-body diff to `turn_{turn:03d}.diff`
Subject: repo:buster_ripper.dump#_write_diff_turn
Verdict: HOLDS
Actual: For turns after zero, the documented writer calls `_write_diff_turn`, which obtains a unified diff of the prior and current rendered body text and writes its joined lines to the named `.diff` file.
Source: `buster_ripper/dump.py:150-175,250-258`
Version: unavailable
Fingerprint: sha256:c8f093c89b61c355ac182feb070cd90b84924ca0d187332117d76b5c766ad9b9
Invalidates-On: Changes to `buster_ripper/dump.py` affecting documented-format dispatch, `_write_diff_turn`, or `_unified_diff`.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-89.md


---

Claim: CL:buster_ripper.dump#_append_index_line/index-line-append
Statement: `_append_index_line` appends a summary line to `index.txt`
Subject: repo:buster_ripper.dump#_append_index_line
Verdict: HOLDS
Actual: It opens `session_dir / "index.txt"` in append mode and writes `summary + "\n"` after checking writability.
Source: `buster_ripper/dump.py:181-186`
Version: unavailable
Fingerprint: sha256:c8f093c89b61c355ac182feb070cd90b84924ca0d187332117d76b5c766ad9b9
Invalidates-On: Changes to `buster_ripper/dump.py`, particularly `_append_index_line` or `_require_writable`.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-90.md


---

Claim: CL:buster_ripper.dump#SessionState/pending-byte-accounting
Statement: `SessionState` tracks pending records and their rendered-text byte count per session
Subject: repo:buster_ripper.dump#SessionState
Verdict: DOES-NOT-HOLD
Actual: It tracks pending records per session, but `_append_record` increments `pending_bytes` by `len(current_text)`, which counts Python string characters rather than encoded bytes. Non-ASCII rendered text can therefore contribute fewer counted units than its UTF-8 byte length.
Source: `buster_ripper/dump.py:32-46,189-194`
Version: unavailable
Fingerprint: sha256:c8f093c89b61c355ac182feb070cd90b84924ca0d187332117d76b5c766ad9b9
Invalidates-On: Changes to `buster_ripper/dump.py`'s `SessionState` fields or `_append_record` accounting.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-91.md


---

Claim: CL:buster_ripper.dump#_append_record/collapsed-record-buffering
Statement: `_append_record` adds a collapsed record to the session's pending list
Subject: repo:buster_ripper.dump#_append_record
Verdict: HOLDS
Actual: `_append_record` appends `record` to `state.pending`, the per-session `SessionState` buffer. A threshold-triggered flush can subsequently clear that list.
Source: `buster_ripper/dump.py:32-46,189-195`
Version: unavailable
Fingerprint: sha256:07b0b076552f72e62c625caade1b01821b3237375900233812d14c0f275b3c2b
Invalidates-On: Changes to `buster_ripper/dump.py` `_append_record` or `SessionState.pending`.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-92.md

---

Claim: CL:buster_ripper.dump#_flush_state/batch-append
Statement: `_flush_state` appends a session's pending batch to its format-named artifact in one file write
Subject: repo:buster_ripper.dump#_flush_state
Verdict: DOES-NOT-HOLD
Actual: For non-`documented` states, it appends the joined pending records with one `f.write` to `<format>.log`. For `documented` states it returns without appending, even if pending records exist (for example, after a format change).
Source: `buster_ripper/dump.py:284-294`
Version: unavailable
Fingerprint: sha256:07b0b076552f72e62c625caade1b01821b3237375900233812d14c0f275b3c2b
Invalidates-On: Changes to the `documented` guard or append path in `buster_ripper/dump.py`.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-93.md

---

Claim: CL:buster_ripper.dump#_flush_state/conditional-batch-append
Statement: For a non-`documented` session state with pending records, `_flush_state` appends their joined batch to `<format>.log` with one `f.write`
Subject: repo:buster_ripper.dump#_flush_state
Verdict: HOLDS
Actual: It derives `<format>.log` from the session directory, opens it in append mode, and makes one `f.write("\n\n".join(state.pending) + "\n")` call. This is conditional on the writable-target guard and successful file I/O; a nonregular or symlink target is rejected instead.
Source: `buster_ripper/dump.py:269-271,284-294,303-305` (production source)
Version: unavailable
Fingerprint: sha256:07b0b076552f72e62c625caade1b01821b3237375900233812d14c0f275b3c2b
Invalidates-On: Changes to `buster_ripper/dump.py`'s `_flush_state`, `_artifact_path`, or writable-target guard.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-120.md

---

Claim: CL:buster_ripper.dump#_flush_state/documented-artifact-guard
Statement: `_flush_state` returns without an artifact append when the session state's format is `documented`
Subject: repo:buster_ripper.dump#_flush_state
Verdict: HOLDS
Actual: Its first branch returns immediately for `state.fmt == "documented"`, before resolving or opening an artifact. The pending buffer is not cleared in this branch.
Source: `buster_ripper/dump.py:284-294` (production source)
Version: unavailable
Fingerprint: sha256:07b0b076552f72e62c625caade1b01821b3237375900233812d14c0f275b3c2b
Invalidates-On: Changes to `buster_ripper/dump.py`'s `_flush_state` documented-format branch.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-121.md

---

Claim: CL:buster_ripper.dump#_flush_state/pending-clear-after-append
Statement: `_flush_state` clears pending records and byte count after the append succeeds
Subject: repo:buster_ripper.dump#_flush_state
Verdict: HOLDS
Actual: After the append context exits successfully, it clears `state.pending` and sets `state.pending_bytes` to zero. An exception before that point skips both resets.
Source: `buster_ripper/dump.py:284-294`
Version: unavailable
Fingerprint: sha256:07b0b076552f72e62c625caade1b01821b3237375900233812d14c0f275b3c2b
Invalidates-On: Changes to the append or post-append reset sequence in `buster_ripper/dump.py`.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-94.md

---

Claim: CL:buster_ripper.dump#dump_turn/documented-pending-buffer
Statement: `dump_turn` does not add pending records for turns processed in `documented` format
Subject: repo:buster_ripper.dump#dump_turn
Verdict: HOLDS
Actual: For `documented`, `dump_turn` calls `_write_documented` rather than `_append_record`; the documented writer writes the turn file and index line without appending to `state.pending`. Existing pending records from any earlier processing are not addressed by this claim.
Source: `buster_ripper/dump.py:83-98` (format dispatch); `buster_ripper/dump.py:150-156,189-195` (writer versus pending append).
Version: unavailable
Fingerprint: sha256:07b0b076552f72e62c625caade1b01821b3237375900233812d14c0f275b3c2b
Invalidates-On: Changes to `buster_ripper/dump.py`'s documented dispatch or `_write_documented` and its callees that add a pending-record append.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-122.md

---

Claim: CL:buster_ripper.server#on_cleanup/shutdown-flush-order
Statement: `on_cleanup` invokes `_final_dump_flush` after cancelling background tasks and before closing the shared session
Subject: repo:buster_ripper.server#on_cleanup
Verdict: HOLDS
Actual: On the normal cleanup path, `on_cleanup` awaits cancellation of both registered background tasks, calls `_final_dump_flush()`, then awaits `app["session"].close()`. An externally propagated cancellation can interrupt that sequence.
Source: `buster_ripper/server.py:74-102`
Version: unavailable
Fingerprint: sha256:cb30bc733586b0f3263c410c37499483a5c59ef8c6bec2aeee38f54f7edb5b7d
Invalidates-On: Change to `buster_ripper/server.py` cleanup ordering or cancellation behavior.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-95.md

---

Claim: CL:buster_ripper.server#spawn_dump_flush/dump-flush-task-lifecycle
Statement: `spawn_dump_flush` stores a `dump.poll_flush()` task in app state for cleanup
Subject: repo:buster_ripper.server#spawn_dump_flush
Verdict: HOLDS
Actual: `spawn_dump_flush` creates an asyncio task from `dump.poll_flush()` and stores it at `app["dump_flush_task"]`; `on_cleanup` cancels and awaits the task under that key.
Source: `buster_ripper/server.py:37-45,74-91` (production source)
Version: unavailable
Fingerprint: sha256:fbd579a7995d0f4a2ae8a6d2994bb9e5fda344c1a900ec581caee38f599647f8
Invalidates-On: Changes to `buster_ripper/server.py` task spawning or cleanup logic.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-96.md


---

Claim: CL:buster_ripper.server#on_cleanup/background-task-cancellation
Statement: `on_cleanup` cancels and awaits each background task through `_cancel_background_task`
Subject: repo:buster_ripper.server#on_cleanup
Verdict: HOLDS
Actual: `on_cleanup` calls `_cancel_background_task` for both background tasks spawned by this module (`kv_monitor_task` and `dump_flush_task`); the helper calls `task.cancel()` and awaits the task if its key is present. It does not enumerate arbitrary other tasks.
Source: `buster_ripper/server.py:25-45,67-89`
Version: unavailable
Fingerprint: sha256:cb30bc733586b0f3263c410c37499483a5c59ef8c6bec2aeee38f54f7edb5b7d
Invalidates-On: Change to `buster_ripper/server.py` background task spawning, cleanup calls, or `_cancel_background_task` implementation.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-97.md

---

Claim: CL:buster_ripper.dump#_resolve_format/format-validation
Statement: `_resolve_format` reads `config.DUMP_FORMAT` and rejects values outside `config.DUMP_FORMATS`
Subject: repo:buster_ripper.dump#_resolve_format
Verdict: HOLDS
Actual: It reads `config.DUMP_FORMAT` into `fmt`, raises `ValueError` when `fmt not in config.DUMP_FORMATS`, and otherwise returns `fmt`.
Source: `buster_ripper/dump.py:128-133`
Version: unavailable
Fingerprint: sha256:c8f093c89b61c355ac182feb070cd90b84924ca0d187332117d76b5c766ad9b9
Invalidates-On: Changes to `buster_ripper/dump.py`, especially `_resolve_format`.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-98.md


---

Claim: CL:buster_ripper.main/dump-format-option
Statement: `main.py` offers `--dump-format` alongside `--dump-dir`
Subject: repo:buster_ripper.main
Verdict: HOLDS
Actual: The `main` click command declares `--dump-dir` (lines 78–79, `metavar="PATH"`, default `None`) and, directly alongside it, `--dump-format` (lines 80–85) with `type=click.Choice(list(config.DUMP_FORMATS))`, default `"documented"`, `expose_value=False`, and `callback=_set_dump_format`, which assigns `config.DUMP_FORMAT` at parse time. The choice values exist in `buster_ripper/config.py` line 27: `("documented", "change-focused", "self-contained")`.
Source: buster_ripper/main.py, lines 78–85 (corroborated by buster_ripper/config.py line 27)
Version: unavailable (working tree; no revision string read)
Fingerprint: sha256:fd36dc12055f67693d9b852fb15a66aa4a3a7864103ac3de0ee67d9520048d47
Invalidates-On: `--dump-format` or `--dump-dir` is removed or renamed in buster_ripper/main.py
Invalidates-On: The `--dump-format` option stops being a sibling option on the `main` click command (moved to another entrypoint)
Checked-At: 2026-09-28
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-99.md


---

Claim: CL:buster_ripper.main/dump-flush-options
Statement: `main.py` exposes `--dump-flush-interval` and `--dump-flush-bytes` as callback-backed numeric options
Subject: repo:buster_ripper.main
Verdict: HOLDS
Actual: The `main` click command declares `--dump-flush-interval` (lines 86–90) and `--dump-flush-bytes` (lines 91–95), both `type=click.IntRange(min=1)` (numeric), both `expose_value=False` with parse-time callbacks: `callback=_set_dump_flush_interval` and `callback=_set_dump_flush_bytes`, defined at lines 18–27, which assign `config.DUMP_FLUSH_INTERVAL` / `config.DUMP_FLUSH_BYTES` and return the value. Defaults come from those config globals (`10` seconds and `4 * 1024 * 1024` bytes, config.py lines 36–37).
Source: buster_ripper/main.py, lines 18–27 and 86–95 (defaults corroborated by buster_ripper/config.py lines 36–37)
Version: unavailable (working tree; no revision string read)
Fingerprint: sha256:fd36dc12055f67693d9b852fb15a66aa4a3a7864103ac3de0ee67d9520048d47
Invalidates-On: Either option is removed, renamed, or loses its `callback=` in buster_ripper/main.py
Invalidates-On: Either option's type stops being numeric (e.g. changed off `click.IntRange`)
Checked-At: 2026-09-28
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-100.md


---

Claim: CL:buster_ripper.dump#poll_flush/periodic-flush-loop
Statement: `poll_flush` repeatedly calls `flush_all` then awaits `asyncio.sleep(config.DUMP_FLUSH_INTERVAL)`
Subject: repo:buster_ripper.dump#poll_flush
Verdict: HOLDS
Actual: The `while True` loop calls `flush_all()` inside a `try` block, logs any `Exception`, and then awaits `asyncio.sleep(config.DUMP_FLUSH_INTERVAL)` before repeating.
Source: `buster_ripper/dump.py:114-125`
Version: unavailable
Fingerprint: sha256:c8f093c89b61c355ac182feb070cd90b84924ca0d187332117d76b5c766ad9b9
Invalidates-On: Change to `buster_ripper/dump.py` affecting `poll_flush` or its loop structure.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-101.md


---

Claim: CL:buster_ripper.dump#_append_record/byte-threshold-comparison
Statement: `_append_record` compares pending rendered-text bytes with `config.DUMP_FLUSH_BYTES` before triggering a flush
Subject: repo:buster_ripper.dump#_append_record
Verdict: DOES-NOT-HOLD
Actual: It accumulates `len(current_text)` into `state.pending_bytes` and compares that count to `config.DUMP_FLUSH_BYTES` before calling `flush_all()`. Python string `len` counts Unicode code points, not encoded bytes, so non-ASCII rendered text makes the purported byte count inaccurate.
Source: `buster_ripper/dump.py:189-194`
Version: unavailable
Fingerprint: sha256:c8f093c89b61c355ac182feb070cd90b84924ca0d187332117d76b5c766ad9b9
Invalidates-On: Changes to `buster_ripper/dump.py` affecting `_append_record`'s accounting or flush comparison.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-102.md


---

Claim: CL:buster_ripper.dump#dump_turn/format-dispatch
Statement: `dump_turn` selects the documented writer or one of two collapsed record builders using the resolved format
Subject: repo:buster_ripper.dump#dump_turn
Verdict: HOLDS
Actual: It obtains `fmt` from `_resolve_format`; `documented` invokes `_write_documented`, `change-focused` invokes `_change_focused_record` through `_append_record`, and the remaining allowed format invokes `_self_contained_record` through `_append_record`. `_resolve_format` rejects unknown formats.
Source: buster_ripper/dump.py:83-98,128-133
Version: unavailable
Fingerprint: sha256:c8f093c89b61c355ac182feb070cd90b84924ca0d187332117d76b5c766ad9b9
Invalidates-On: Changes to `dump_turn`'s format dispatch or `_resolve_format` in `buster_ripper/dump.py`.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-103.md


---

Claim: CL:buster_ripper.server#_final_dump_flush/flush-error-isolation
Statement: `_final_dump_flush` catches ordinary exceptions from `dump.flush_all`
Subject: repo:buster_ripper.server#_final_dump_flush
Verdict: HOLDS
Actual: `_final_dump_flush` calls `dump.flush_all()` inside a `try` block and catches `Exception`, logging the failure rather than propagating ordinary exceptions.
Source: `buster_ripper/server.py:97-102`
Version: unavailable
Fingerprint: sha256:cb30bc733586b0f3263c410c37499483a5c59ef8c6bec2aeee38f54f7edb5b7d
Invalidates-On: Changes to `buster_ripper/server.py`, especially `_final_dump_flush`'s `try`/`except` around `dump.flush_all()`.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-104.md

---

Claim: CL:buster_ripper.dump#_artifact_path/collapsed-artifact-path
Statement: `_artifact_path` constructs a per-session `<format>.log` path
Subject: repo:buster_ripper.dump#_artifact_path
Verdict: HOLDS
Actual: It joins `state.session_dir` (created as `config.DUMP_DIR / sid`) with `f"{fmt}{_ARTIFACT_SUFFIX}"`; the suffix is `.log`.
Source: `buster_ripper/dump.py:29, 142-144, 268-270`
Version: unavailable
Fingerprint: sha256:c8f093c89b61c355ac182feb070cd90b84924ca0d187332117d76b5c766ad9b9
Invalidates-On: Change to `buster_ripper/dump.py` affecting `_artifact_path`, `_ARTIFACT_SUFFIX`, or session directory construction.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-105.md

---

Claim: CL:tests.test_dump/documented-turn-file-reads
Statement: `tests/test_dump.py` reads `turn_000.json`, `turn_001.diff` and `turn_002.diff` with `read_text()`
Subject: repo:tests.test_dump
Verdict: HOLDS
Actual: `TestDocumentedFormatPreserved.test_files_names_and_content_unchanged` calls `.read_text()` on each of these three files and compares their contents with expected text.
Source: `tests/test_dump.py:129-153` (the claim concerns the test suite itself).
Version: unavailable
Fingerprint: sha256:0f32dc28712742a9b89536d6f52a202cbbfab3543a5cd41f4a3f783cc96d7ab9
Invalidates-On: Any change to `tests/test_dump.py` that removes or replaces these reads.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-107.md

---

Claim: CL:tests.test_dump/documented-index-read
Statement: `tests/test_dump.py` reads `index.txt` with `read_text()`
Subject: repo:tests.test_dump
Verdict: HOLDS
Actual: `TestDocumentedFormatPreserved.test_index_line_format_unchanged` calls `.read_text()` on `index.txt`, then splits the result into lines for assertions.
Source: `tests/test_dump.py:156-173` (the claim concerns the test suite itself).
Version: unavailable
Fingerprint: sha256:0f32dc28712742a9b89536d6f52a202cbbfab3543a5cd41f4a3f783cc96d7ab9
Invalidates-On: Any change to `tests/test_dump.py` that removes or replaces the `index.txt` read.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-108.md

---

Claim: CL:buster_ripper.dump#_append_record/pending-unit-accumulation
Statement: `_append_record` increments the per-session `pending_bytes` accumulator by `len(current_text)`
Subject: repo:buster_ripper.dump#_append_record
Verdict: HOLDS
Actual: The function adds `len(current_text)` to `state.pending_bytes`; `state` is the per-session `SessionState` passed from `dump_turn`.
Source: `buster_ripper/dump.py:189-194` (per-session state lookup and call sites at lines 83-98, 136-147)
Version: unavailable
Fingerprint: sha256:c8f093c89b61c355ac182feb070cd90b84924ca0d187332117d76b5c766ad9b9
Invalidates-On: Change to `buster_ripper/dump.py` `_append_record` or its session-state lookup/call sites.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-109.md

---

Claim: CL:buster_ripper.dump#_append_record/threshold-flush-trigger
Statement: `_append_record` calls `flush_all()` when accumulated `pending_bytes >= config.DUMP_FLUSH_BYTES`
Subject: repo:buster_ripper.dump#_append_record
Verdict: HOLDS
Actual: `_append_record` adds the UTF-8 byte length of `current_text` to `state.pending_bytes` and calls `flush_all()` when that session's accumulated value reaches or exceeds `config.DUMP_FLUSH_BYTES`.
Source: `buster_ripper/dump.py:189-195`
Version: unavailable
Fingerprint: sha256:07b0b076552f72e62c625caade1b01821b3237375900233812d14c0f275b3c2b
Invalidates-On: Changes to `buster_ripper/dump.py` `_append_record` threshold or accumulation logic.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-110.md

---

Claim: CL:python/string-utf8-length-relation
Statement: For non-ASCII Python strings, `len(text)` can differ from `len(text.encode("utf-8"))`
Subject: lang:python
Verdict: HOLDS
Actual: `len(text)` counts string code points; UTF-8 encoding represents a non-ASCII code point with two to four bytes, so, for example, `len("é")` is 1 while `len("é".encode("utf-8"))` is 2. This is a possibility, not an assertion that every string containing non-ASCII must encode successfully (e.g. lone surrogates).
Source: Python 3.14 Unicode HOWTO, “Encodings,” “The String Type,” and “Converting to Bytes”: https://docs.python.org/3.14/howto/unicode.html#encodings (consulted HTML lines 349–353, 399–403, 474–476).
Version: Python 3.14 documentation (page fetched 2026-09-27).
Fingerprint: sha256:1b59c6964c21a7b3ae7ae3659388f88048623060575847d193d643f6009d5767
Invalidates-On: Change to the consulted Python Unicode HOWTO HTML bytes or to Python's `str`/UTF-8 encoding semantics.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-111.md

---

Claim: CL:python/utf8-encoded-byte-length
Statement: Encoding a Python string with UTF-8 produces a `bytes` value whose length counts its encoded bytes
Subject: lang:python
Verdict: HOLDS
Actual: `str.encode('utf-8')` returns a `bytes` object. Bytes objects are sequences of individual bytes, and `len` returns a sequence's length, so `len(s.encode('utf-8'))` counts encoded bytes rather than string characters.
Source: https://raw.githubusercontent.com/python/cpython/3.14/Doc/builtins/stdtypes.rst — `str.encode` (lines 1928–1933), common sequence operations `len(s)` (line 1039), and bytes objects (lines 3318–3321).
Version: CPython documentation branch 3.14; exact revision unavailable.
Fingerprint: sha256:5a6951b72ecc630a87f45a650fc11c186cfeb4251bedd1ea28f45bcb3116e8e1
Invalidates-On: Change to the referenced CPython `Doc/builtins/stdtypes.rst` bytes, particularly `str.encode`, bytes sequence semantics, or `len` sequence semantics.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-112.md

---

Claim: CL:python/completed-task-cancellation
Statement: Cancelling an already-completed `asyncio.Task` does not replace its stored exception with cancellation
Subject: lang:python
Verdict: HOLDS
Actual: `Task.cancel()` returns `False` when `self.done()` is true, before requesting cancellation or changing its completion state; its only preceding assignment clears the traceback-logging flag, not the stored exception.
Source: Installed CPython `asyncio/tasks.py`, `Task.cancel`, lines 183–223 (especially 205–207).
Version: CPython 3.14.7 (Homebrew installed standard library)
Fingerprint: sha256:a97ffdb60a42104a86f2ece781878b3371bd88e9eea8f1ac1fa8f397cc11cca5
Invalidates-On: Change to installed `asyncio/tasks.py` `Task.cancel` implementation or replacement of the selected Python runtime.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-113.md

---

Claim: CL:python/failed-task-await-exception
Statement: Awaiting a completed `asyncio.Task` that failed with an ordinary exception raises its stored exception
Subject: lang:python
Verdict: HOLDS
Actual: The Future await implementation immediately calls `result()` when already done, and `result()` raises the stored exception for a finished future with an exception; Tasks inherit this Future behavior.
Source: Installed CPython `asyncio/futures.py`, `Future.__await__` and `Future.result`, lines 195–209 and 292–298.
Version: CPython 3.14.7 (Homebrew installed standard library)
Fingerprint: sha256:2f482524292a7c5d1b818aeb92434d183bd089e014cf50b785432284d2c8bc12
Invalidates-On: Change to installed `asyncio/futures.py` Future await/result implementation or replacement of the selected Python runtime.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-114.md

---

Claim: CL:buster_ripper.server#_cancel_background_task/exception-handling
Statement: `_cancel_background_task` catches `asyncio.CancelledError` but not ordinary exceptions raised by awaiting the task
Subject: repo:buster_ripper.server#_cancel_background_task
Verdict: HOLDS
Actual: It cancels and awaits the task, swallowing only `asyncio.CancelledError`; any ordinary exception from `await task` propagates.
Source: `buster_ripper/server.py:82-91` (production implementation)
Version: unavailable
Fingerprint: sha256:fbd579a7995d0f4a2ae8a6d2994bb9e5fda344c1a900ec581caee38f599647f8
Invalidates-On: Changes to `buster_ripper/server.py` exception handling in `_cancel_background_task`.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-115.md

---

Claim: CL:buster_ripper.dump#_append_record/encoding-before-buffer-mutation
Statement: `_append_record` computes `len(current_text.encode("utf-8"))` before mutating its pending list
Subject: repo:buster_ripper.dump#_append_record
Verdict: HOLDS
Actual: It assigns `rendered_bytes = len(current_text.encode("utf-8"))` immediately before calling `state.pending.append(record)`.
Source: `buster_ripper/dump.py:189-193`
Version: unavailable
Fingerprint: sha256:07b0b076552f72e62c625caade1b01821b3237375900233812d14c0f275b3c2b
Invalidates-On: Changes to `_append_record` in `buster_ripper/dump.py`, particularly the order of encoding and pending-list mutation.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-116.md

---

Claim: CL:buster_ripper.dump#_append_record/pending-byte-accumulation
Statement: `_append_record` increments `pending_bytes` by the rendered text's UTF-8 encoded length
Subject: repo:buster_ripper.dump#_append_record
Verdict: HOLDS
Actual: It computes `rendered_bytes` from `len(current_text.encode("utf-8"))`, then adds that amount to `state.pending_bytes`; it does not use the byte length of `record`.
Source: `buster_ripper/dump.py:189-194`
Version: unavailable
Fingerprint: sha256:07b0b076552f72e62c625caade1b01821b3237375900233812d14c0f275b3c2b
Invalidates-On: Changes to `_append_record` in `buster_ripper/dump.py`, particularly the source or accumulation of `rendered_bytes`.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-117.md

---

Claim: CL:buster_ripper.server#_cancel_background_task/ordinary-error-isolation
Statement: `_cancel_background_task` catches ordinary exceptions raised by awaiting the stored task
Subject: repo:buster_ripper.server#_cancel_background_task
Verdict: HOLDS
Actual: After cancelling the stored task, it awaits it inside a try block; `except Exception` logs and suppresses ordinary exceptions from that await.
Source: `buster_ripper/server.py:82-94`
Version: unavailable
Fingerprint: sha256:cb30bc733586b0f3263c410c37499483a5c59ef8c6bec2aeee38f54f7edb5b7d
Invalidates-On: Changes to `buster_ripper/server.py`, especially `_cancel_background_task`'s try/except around `await task`.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-118.md

---

Claim: CL:buster_ripper.server#_cancel_background_task/cleanup-cancellation-propagation
Statement: `_cancel_background_task` re-raises `CancelledError` when its current cleanup task is cancelling
Subject: repo:buster_ripper.server#_cancel_background_task
Verdict: HOLDS
Actual: Its `except asyncio.CancelledError` branch checks `asyncio.current_task().cancelling()` and re-raises when that count is nonzero; otherwise it suppresses the awaited task's cancellation.
Source: `buster_ripper/server.py:82-94`
Version: unavailable
Fingerprint: sha256:cb30bc733586b0f3263c410c37499483a5c59ef8c6bec2aeee38f54f7edb5b7d
Invalidates-On: Changes to `buster_ripper/server.py`, especially `_cancel_background_task`'s `CancelledError` handler.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-119.md
---

Claim: CL:aiohttp/runner-task-cancellation
Statement: On `GracefulExit`, aiohttp `run_app` cancels and awaits its `_run_app` task
Subject: pkg:aiohttp@3.13.3
Verdict: HOLDS
Actual: `run_app` catches `GracefulExit`, then its `finally` cancels `main_task` (created from `_run_app`) and drives the loop until that task completes, suppressing `CancelledError`.
Source: Installed `.venv/lib/python3.13/site-packages/aiohttp/web.py`, lines 485–517.
Version: aiohttp 3.13.3
Fingerprint: sha256:27348d99e8e0e46e987850279088197f2b6a6d4f3ab0dbbce38c982a6a14a6ab
Invalidates-On: Change to installed aiohttp `web.py` implementation of `run_app` or aiohttp package version
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-123.md

---

Claim: CL:aiohttp/runner-cleanup-await
Statement: aiohttp `_run_app` awaits runner cleanup in its own `finally` block
Subject: pkg:aiohttp@3.13.3
Verdict: HOLDS
Actual: After `await runner.setup()`, `_run_app` enters a `try` whose `finally` performs `await runner.cleanup()`; failures before entering the `try` are not covered.
Source: Installed `.venv/lib/python3.13/site-packages/aiohttp/web.py`, lines 325–331 and 421–425.
Version: aiohttp 3.13.3
Fingerprint: sha256:27348d99e8e0e46e987850279088197f2b6a6d4f3ab0dbbce38c982a6a14a6ab
Invalidates-On: Change to installed aiohttp `web.py` implementation of `_run_app` or aiohttp package version
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-124.md

---

Claim: CL:python/task-cancellation-count-retention
Statement: `asyncio.Task.cancelling()` retains the count of cancellation requests after `CancelledError` delivery unless explicitly reduced
Subject: lang:python
Verdict: HOLDS
Actual: `cancel()` increments `_num_cancels_requested`, `cancelling()` returns it, and `uncancel()` decrements it; delivering `CancelledError` clears `_must_cancel` but does not decrement the count. The count is not itself a guarantee of another exception at the next await.
Source: Installed Python standard library `Lib/asyncio/tasks.py`, `Task.cancel`, `Task.cancelling`, `Task.uncancel`, and `Task.__step` (lines 183–274); corroborated by Python 3.14 `asyncio` documentation, Task object / cancelling().
Version: Python 3.14.7
Fingerprint: sha256:a97ffdb60a42104a86f2ece781878b3371bd88e9eea8f1ac1fa8f397cc11cca5
Invalidates-On: Change to installed Python `asyncio/tasks.py` cancellation-count or exception-delivery logic
Invalidates-On: Change to the active asyncio Task implementation or Python runtime version
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-125.md

---

Claim: CL:python/cancelled-child-await-propagation
Statement: Awaiting a child task that terminates cancelled raises `CancelledError` in the awaiting task
Subject: lang:python
Verdict: HOLDS
Actual: An awaited Task is Future-like: its `__await__` returns `result()`, and `result()` raises `CancelledError` when cancelled. The exception reaches the awaiting task unless its coroutine catches it; this does not by itself mean that the awaiting task was separately sent a cancellation request.
Source: Installed Python standard library `Lib/asyncio/futures.py`, `Future.result` and `Future.__await__` (lines 195–209, 292–300); Python 3.14 `asyncio` documentation, Task object, confirms Task inherits Future APIs.
Version: Python 3.14.7
Fingerprint: sha256:2f482524292a7c5d1b818aeb92434d183bd089e014cf50b785432284d2c8bc12
Invalidates-On: Change to installed Python `asyncio/futures.py` cancelled-result or await behavior
Invalidates-On: Change to the active asyncio Task implementation or Python runtime version
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-126.md

---

Claim: CL:buster_ripper.server#_cancel_background_task/cancellation-count-snapshot
Statement: `_cancel_background_task` snapshots the cleanup task's cancellation count before cancelling and awaiting the stored background task
Subject: repo:buster_ripper.server#_cancel_background_task
Verdict: HOLDS
Actual: After retrieving the stored task, it takes `asyncio.current_task()`, saves `cleanup.cancelling()`, calls `task.cancel()`, then awaits `task`.
Source: `buster_ripper/server.py:82-91`
Version: unavailable
Fingerprint: sha256:6a3e45ef69413838dc53de2939ff1bb06b268746231bf63f3589be407f3d3a5a
Invalidates-On: Change to `buster_ripper/server.py` affecting `_cancel_background_task` or its task storage.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-127.md

---

Claim: CL:buster_ripper.server#_cancel_background_task/cancellation-count-increase-propagation
Statement: `_cancel_background_task` re-raises a child's `CancelledError` when the cleanup task's cancellation count increased during the await
Subject: repo:buster_ripper.server#_cancel_background_task
Verdict: HOLDS
Actual: It catches `asyncio.CancelledError` from awaiting the stored task and re-raises it if `cleanup.cancelling()` exceeds the count saved before `task.cancel()` and the await. The branch does not distinguish the exception's origin; it applies to any `CancelledError` caught there.
Source: `buster_ripper/server.py:82-95`
Version: unavailable
Fingerprint: sha256:6a3e45ef69413838dc53de2939ff1bb06b268746231bf63f3589be407f3d3a5a
Invalidates-On: Change to `buster_ripper/server.py` affecting `_cancel_background_task` cancellation handling.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-128.md

---

Claim: CL:buster_ripper.routes.messages#handle_messages/client-unicode-decode-error-response
Statement: `handle_messages` maps client `UnicodeDecodeError` from `request.json()` to a JSON 400 response
Subject: repo:buster_ripper.routes.messages#handle_messages
Verdict: HOLDS
Actual: The `await request.json()` call is inside a try block catching `UnicodeDecodeError`; that branch returns `web.json_response({"error": "Invalid JSON"}, status=400)`.
Source: `buster_ripper/routes/messages.py:24-29`
Version: unavailable
Fingerprint: sha256:721a2afe4d1ca35da9461c95f5c6dfe48dcf04ec4c4c0a606763cc29b0ca96da
Invalidates-On: Change to `handle_messages` request parsing or its exception handler in `buster_ripper/routes/messages.py`.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-130.md

---

Claim: CL:buster_ripper.routes.messages#handle_messages/client-json-decode-error-response
Statement: `handle_messages` maps client `json.JSONDecodeError` from `request.json()` to a JSON 400 response
Subject: repo:buster_ripper.routes.messages#handle_messages
Verdict: HOLDS
Actual: The `await request.json()` call is inside a try block catching `json.JSONDecodeError`; that branch returns `web.json_response({"error": "Invalid JSON"}, status=400)`.
Source: `buster_ripper/routes/messages.py:24-29`
Version: unavailable
Fingerprint: sha256:721a2afe4d1ca35da9461c95f5c6dfe48dcf04ec4c4c0a606763cc29b0ca96da
Invalidates-On: Change to `handle_messages` request parsing or its exception handler in `buster_ripper/routes/messages.py`.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-131.md

---

Claim: CL:buster_ripper.routes.chat#handle_chat/client-unicode-decode-error-response
Statement: `handle_chat` maps client `UnicodeDecodeError` from `request.json()` to a JSON 400 response
Subject: repo:buster_ripper.routes.chat#handle_chat
Verdict: HOLDS
Actual: The `request.json()` call is inside a try block that catches `UnicodeDecodeError` and returns `web.json_response({"error": "Invalid JSON"}, status=400)`.
Source: `buster_ripper/routes/chat.py:23-31` (production source)
Version: unavailable
Fingerprint: sha256:e5323a69332745c7f766e617c63af98c9a906f4aa0e9dbbd1a0f229312d9f9f8
Invalidates-On: Any change to the client JSON parsing or exception handler in `buster_ripper/routes/chat.py`.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-132.md

---

Claim: CL:buster_ripper.routes.chat#handle_chat/client-json-decode-error-response
Statement: `handle_chat` maps client `json.JSONDecodeError` from `request.json()` to a JSON 400 response
Subject: repo:buster_ripper.routes.chat#handle_chat
Verdict: HOLDS
Actual: The `request.json()` call is inside a try block that catches `json.JSONDecodeError` and returns `web.json_response({"error": "Invalid JSON"}, status=400)`.
Source: `buster_ripper/routes/chat.py:23-31` (production source)
Version: unavailable
Fingerprint: sha256:e5323a69332745c7f766e617c63af98c9a906f4aa0e9dbbd1a0f229312d9f9f8
Invalidates-On: Any change to the client JSON parsing or exception handler in `buster_ripper/routes/chat.py`.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-133.md

---

Claim: CL:buster_ripper.routes.count_tokens#handle_count_tokens/client-body-parse-delegation
Statement: `handle_count_tokens` delegates client body parsing to `_read_body` before using the returned body
Subject: repo:buster_ripper.routes.count_tokens#handle_count_tokens
Verdict: HOLDS
Actual: The handler first assigns `body = await _read_body(request)` and then passes `body` to `session_id` and, when needed, `estimate_tokens`. `_read_body` performs `request.json()` with JSON syntax and Unicode decoding error handling.
Source: `buster_ripper/routes/count_tokens.py:16-25,28-50`
Version: unavailable
Fingerprint: sha256:a69c9884cad5997e442a4eaa5c5557f13a589b227777b71fbe1c4167c38fa203
Invalidates-On: Changes to `buster_ripper/routes/count_tokens.py`, especially the handler's initial body assignment or `_read_body`.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-134.md

---

Claim: CL:buster_ripper.routes.count_tokens#_read_body/client-json-decode-error-fallback
Statement: `_read_body` maps client `json.JSONDecodeError` from `request.json()` to an empty dictionary
Subject: repo:buster_ripper.routes.count_tokens#_read_body
Verdict: HOLDS
Actual: The `await request.json()` call is inside a try block whose `except json.JSONDecodeError` returns `{}`. The handler does not distinguish client-originated decode errors from other `JSONDecodeError`s raised by that call.
Source: `buster_ripper/routes/count_tokens.py:16-25`
Version: unavailable
Fingerprint: sha256:a69c9884cad5997e442a4eaa5c5557f13a589b227777b71fbe1c4167c38fa203
Invalidates-On: Changes to `_read_body` exception handling in `buster_ripper/routes/count_tokens.py`.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-135.md

---

Claim: CL:buster_ripper.routes.count_tokens#_read_body/client-unicode-decode-error-raise
Statement: `_read_body` maps client `UnicodeDecodeError` from `request.json()` to `web.HTTPBadRequest`
Subject: repo:buster_ripper.routes.count_tokens#_read_body
Verdict: HOLDS
Actual: The `await request.json()` call is inside a try block whose `except UnicodeDecodeError` raises `web.HTTPBadRequest` with a JSON error message.
Source: `buster_ripper/routes/count_tokens.py:16-25`
Version: unavailable
Fingerprint: sha256:a69c9884cad5997e442a4eaa5c5557f13a589b227777b71fbe1c4167c38fa203
Invalidates-On: Changes to `_read_body` exception handling in `buster_ripper/routes/count_tokens.py`.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-136.md

---

Claim: CL:buster_ripper.routes.count_tokens#_read_body/unicode-decode-error-content-type
Statement: `_read_body` sets the raised `HTTPBadRequest` content type to `application/json`
Subject: repo:buster_ripper.routes.count_tokens#_read_body
Verdict: HOLDS
Actual: Its `UnicodeDecodeError` handler raises `web.HTTPBadRequest` with `content_type="application/json"` explicitly supplied.
Source: `buster_ripper/routes/count_tokens.py:16-25`
Version: unavailable
Fingerprint: sha256:a69c9884cad5997e442a4eaa5c5557f13a589b227777b71fbe1c4167c38fa203
Invalidates-On: Changes to the `web.HTTPBadRequest` constructor arguments in `buster_ripper/routes/count_tokens.py`.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-138.md

---

Claim: CL:buster_ripper.routes.chat#_make_chat_request/upstream-unicode-decode-error-response
Statement: `_make_chat_request` maps upstream response `UnicodeDecodeError` from `resp.json()` to a JSON 502 response
Subject: repo:buster_ripper.routes.chat#_make_chat_request
Verdict: HOLDS
Actual: `resp.json()` executes inside a try block whose `except (json.JSONDecodeError, UnicodeDecodeError)` returns `web.json_response({"error": "Invalid JSON from upstream"}, status=502)`.
Source: `buster_ripper/routes/chat.py:114-149` (production source)
Version: unavailable
Fingerprint: sha256:e5323a69332745c7f766e617c63af98c9a906f4aa0e9dbbd1a0f229312d9f9f8
Invalidates-On: Change to `buster_ripper/routes/chat.py`, especially the `resp.json()` try/except or response construction.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-139.md

---

Claim: CL:buster_ripper.routes.chat#_make_chat_request/upstream-json-decode-error-response
Statement: `_make_chat_request` maps upstream response `json.JSONDecodeError` from `resp.json()` to a JSON 502 response
Subject: repo:buster_ripper.routes.chat#_make_chat_request
Verdict: HOLDS
Actual: `resp.json()` executes inside a try block whose `except (json.JSONDecodeError, UnicodeDecodeError)` returns `web.json_response({"error": "Invalid JSON from upstream"}, status=502)`.
Source: `buster_ripper/routes/chat.py:114-149` (production source)
Version: unavailable
Fingerprint: sha256:e5323a69332745c7f766e617c63af98c9a906f4aa0e9dbbd1a0f229312d9f9f8
Invalidates-On: Change to `buster_ripper/routes/chat.py`, especially the `resp.json()` try/except or response construction.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-140.md

---

Claim: CL:buster_ripper.backend#detect_backend/probe-request-timeout
Statement: The `/v1/models` GET in `detect_backend` passes `aiohttp.ClientTimeout(total=5)` as its per-request timeout
Subject: repo:buster_ripper.backend#detect_backend
Verdict: HOLDS
Actual: The probe calls `session.get(f"{backend_url}/v1/models", timeout=aiohttp.ClientTimeout(total=5))`.
Source: `buster_ripper/backend.py:53-56`
Version: unavailable
Fingerprint: sha256:c2b71468af6b53efde753b0648d47eb6e06c8db4c738e08f7c01f4e4682d76ef
Invalidates-On: Change to `detect_backend`'s probe GET timeout in `buster_ripper/backend.py`.
Checked-At: 2026-09-28T00:15:27Z
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-141.md

---

Claim: CL:buster_ripper.backend#detect_backend/probe-exception-cache
Statement: `detect_backend` caches `"unknown"` after an ordinary exception in the probe's `try` block
Subject: repo:buster_ripper.backend#detect_backend
Verdict: HOLDS
Actual: Its `except Exception` branch sets `_backend_type = "unknown"`; the following assignment sets `_backend_state_cached = True` before returning.
Source: `buster_ripper/backend.py:53-71`
Version: unavailable
Fingerprint: sha256:c2b71468af6b53efde753b0648d47eb6e06c8db4c738e08f7c01f4e4682d76ef
Invalidates-On: Change to `detect_backend`'s probe exception handler or cache assignment in `buster_ripper/backend.py`.
Checked-At: 2026-09-28T00:15:27Z
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-142.md

---

Claim: CL:buster_ripper.routes.chat#_make_chat_request/stream-dispatch
Statement: `_make_chat_request` selects `_relay_chat_stream` when the outgoing request body has a truthy `stream` value
Subject: repo:buster_ripper.routes.chat#_make_chat_request
Verdict: HOLDS
Actual: It sends `body` as the upstream JSON payload, then branches on `body.get("stream")` and returns `_relay_chat_stream(request, resp)` when truthy.
Source: buster_ripper/routes/chat.py:114-133
Version: unavailable
Fingerprint: sha256:3068981c63ff3737b223b5f9bb9018a23195034afbc67df77e3c8ec37ff0a9ce
Invalidates-On: Change to `_make_chat_request`'s outgoing body or stream branch in `buster_ripper/routes/chat.py`.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-145.md

---

Claim: CL:buster_ripper.routes.chat#_make_chat_request/nonstream-json-parsing
Statement: `_make_chat_request` calls `resp.json()` after the stream branch returns on the nonstream path
Subject: repo:buster_ripper.routes.chat#_make_chat_request
Verdict: HOLDS
Actual: The truthy-stream branch returns before `resp_data = await resp.json()`; the nonstream path falls through to that call.
Source: buster_ripper/routes/chat.py:130-135
Version: unavailable
Fingerprint: sha256:3068981c63ff3737b223b5f9bb9018a23195034afbc67df77e3c8ec37ff0a9ce
Invalidates-On: Change to stream control flow or response parsing in `buster_ripper/routes/chat.py`.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-146.md

---

Claim: CL:buster_ripper.routes.chat#_make_chat_request/nonstream-transform-dispatch
Statement: `_make_chat_request` passes parsed nonstream response data to `_apply_response_transforms`
Subject: repo:buster_ripper.routes.chat#_make_chat_request
Verdict: HOLDS
Actual: On the nonstream success path it assigns `resp_data = await resp.json()`, checks for overflow, then returns `_apply_response_transforms(resp_data, profile, resp.status)`.
Source: buster_ripper/routes/chat.py:130-142
Version: unavailable
Fingerprint: sha256:3068981c63ff3737b223b5f9bb9018a23195034afbc67df77e3c8ec37ff0a9ce
Invalidates-On: Change to response parsing, overflow handling, or transform invocation in `buster_ripper/routes/chat.py`.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-147.md

---

Claim: CL:buster_ripper.routes.chat#_make_chat_request/upstream-client-error-response
Statement: `_make_chat_request` converts `aiohttp.ClientError` from the upstream path into a JSON 502 response
Subject: repo:buster_ripper.routes.chat#_make_chat_request
Verdict: HOLDS
Actual: Its upstream request and response handling are inside a `try` whose `except aiohttp.ClientError` returns `web.json_response({"error": str(e)}, status=502)`.
Source: buster_ripper/routes/chat.py:124-140
Version: unavailable
Fingerprint: sha256:3068981c63ff3737b223b5f9bb9018a23195034afbc67df77e3c8ec37ff0a9ce
Invalidates-On: Change to upstream exception handling in `buster_ripper/routes/chat.py`.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-148.md

---

Claim: CL:buster_ripper.routes.chat#_relay_chat_stream/upstream-chunk-relay
Statement: `_relay_chat_stream` forwards successful upstream chunks from `resp.content.iter_any()` to `StreamResponse.write` unchanged
Subject: repo:buster_ripper.routes.chat#_relay_chat_stream
Verdict: HOLDS
Actual: For upstream status below 400, it iterates `resp.content.iter_any()` and calls `await response.write(chunk)` with each chunk directly; no decoding or transformation occurs in this function.
Source: buster_ripper/routes/chat.py:155-168
Version: unavailable
Fingerprint: sha256:3068981c63ff3737b223b5f9bb9018a23195034afbc67df77e3c8ec37ff0a9ce
Invalidates-On: Any change to `_relay_chat_stream` in `buster_ripper/routes/chat.py`, especially its iteration or write path.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-149.md

---

Claim: CL:buster_ripper.routes.chat#_relay_chat_stream/sse-response-preparation
Statement: `_relay_chat_stream` prepares a `text/event-stream` response before forwarding successful upstream chunks
Subject: repo:buster_ripper.routes.chat#_relay_chat_stream
Verdict: HOLDS
Actual: For upstream status below 400, it creates a `web.StreamResponse` with `Content-Type: text/event-stream`, awaits `response.prepare(request)`, and only then begins forwarding chunks.
Source: buster_ripper/routes/chat.py:155-168
Version: unavailable
Fingerprint: sha256:3068981c63ff3737b223b5f9bb9018a23195034afbc67df77e3c8ec37ff0a9ce
Invalidates-On: Any change to `_relay_chat_stream` in `buster_ripper/routes/chat.py`, especially response headers or the ordering of `prepare` and chunk forwarding.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-150.md

---

Claim: CL:buster_ripper.routes.chat#_relay_chat_stream/upstream-error-relay
Statement: `_relay_chat_stream` returns upstream HTTP error status and raw body without calling `resp.json()`
Subject: repo:buster_ripper.routes.chat#_relay_chat_stream
Verdict: HOLDS
Actual: For `resp.status >= 400`, it constructs `web.Response` with `status=resp.status` and `body=await resp.read()`; this branch never calls `resp.json()`.
Source: buster_ripper/routes/chat.py:155-160
Version: unavailable
Fingerprint: sha256:16c7e99938b55e508b36c16092e7a89883baeff293385c3dfa3e9c161b413cae
Invalidates-On: Change to the error branch of `_relay_chat_stream` in `buster_ripper/routes/chat.py`.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-151.md

---

Claim: CL:buster_ripper.routes.chat#_apply_response_transforms/nonstream-json-serialization
Statement: `_apply_response_transforms` serializes the nonstream response as JSON after applying applicable eval-mode or non-eval transformations
Subject: repo:buster_ripper.routes.chat#_apply_response_transforms
Verdict: HOLDS
Actual: The nonstream caller passes parsed upstream JSON to this function. It selects eval-mode or non-eval transformations for each choice, then serializes `resp_data` with `json.dumps(...).encode()` into an application/json `web.Response`, whether or not a transformation changed it. Transformation exceptions are suppressed, so serialization can follow only partially applied transformations.
Source: buster_ripper/routes/chat.py:131-142,171-200,245-279` (production source)
Version: unavailable
Fingerprint: sha256:3068981c63ff3737b223b5f9bb9018a23195034afbc67df77e3c8ec37ff0a9ce
Invalidates-On: Changes to `buster_ripper/routes/chat.py` response-transform or nonstream caller logic.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-152.md

---

Claim: CL:buster_ripper.routes.chat#_relay_chat_stream/upstream-error-content-type
Statement: `_relay_chat_stream` labels upstream HTTP error responses `application/json` regardless of the upstream Content-Type
Subject: repo:buster_ripper.routes.chat#_relay_chat_stream
Verdict: HOLDS
Actual: For upstream status >= 400, it reads the upstream body and returns `web.Response` with `content_type="application/json"`; it never inspects the upstream Content-Type in this branch.
Source: `buster_ripper/routes/chat.py:155-160`
Version: unavailable
Fingerprint: sha256:3068981c63ff3737b223b5f9bb9018a23195034afbc67df77e3c8ec37ff0a9ce
Invalidates-On: Changes to `buster_ripper/routes/chat.py` affecting `_relay_chat_stream`'s error-response branch.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-153.md

---

Claim: CL:aiohttp/response-header-exposure
Statement: `ClientResponse.headers` exposes the upstream response's Content-Type header value
Subject: pkg:aiohttp@3.13.3
Verdict: HOLDS
Actual: After response processing, aiohttp assigns the parsed response message's headers to `_headers`; `ClientResponse.headers` returns that case-insensitive header mapping, including its Content-Type value when present. This is the header value, not the separate `content_type` parsing/fallback property.
Source: `.venv/lib/python3.13/site-packages/aiohttp/client_reqrep.py:427-429,528-565`
Version: aiohttp 3.13.3
Fingerprint: sha256:7844440eb67433c6454edd708c776e47e3fcfec9b8d0aeb980dcfe88c1986ac9
Invalidates-On: Change to aiohttp's `ClientResponse.start` header assignment or `ClientResponse.headers` property.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-154.md

---

Claim: CL:aiohttp/web-response-explicit-content-type
Statement: `aiohttp.web.Response` accepts an explicit Content-Type response header alongside a byte body
Subject: pkg:aiohttp@3.13.3
Verdict: HOLDS
Actual: `Response(body=some_bytes, headers={"Content-Type": "application/example"})` retains the supplied header and stores the bytes as its body. Supplying `content_type` or `charset` constructor parameters in addition to that header is forbidden and raises `ValueError`.
Source: /Users/alexhubitski/Projects/buster-ripper/.venv/lib/python3.13/site-packages/aiohttp/web_response.py:628-688,697-703
Version: aiohttp 3.13.3 (installed)
Fingerprint: sha256:3ca73388d5382e67ed5ea295be844cac06ce7950a5a5237e8b39c7b2259ece65
Invalidates-On: Changes to installed aiohttp/web_response.py constructor header validation or bytes body setter.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-155.md

---

Claim: CL:buster_ripper.routes.chat#_relay_chat_stream/upstream-error-content-type-copy
Statement: `_relay_chat_stream` copies an upstream HTTP error response's Content-Type header into the downstream response when present
Subject: repo:buster_ripper.routes.chat#_relay_chat_stream
Verdict: HOLDS
Actual: In the HTTP error branch, when `"Content-Type" in resp.headers`, it passes `headers={"Content-Type": resp.headers["Content-Type"]}` to `web.Response`.
Source: buster_ripper/routes/chat.py:155-160
Version: unavailable
Fingerprint: sha256:16c7e99938b55e508b36c16092e7a89883baeff293385c3dfa3e9c161b413cae
Invalidates-On: Change to the error response header expression in `buster_ripper/routes/chat.py`.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-157.md

---

Claim: CL:buster_ripper.routes.chat#_relay_chat_stream/upstream-error-absent-header-handling
Statement: `_relay_chat_stream` passes no explicit response headers when an upstream HTTP error lacks Content-Type
Subject: repo:buster_ripper.routes.chat#_relay_chat_stream
Verdict: HOLDS
Actual: When the upstream error response has no `Content-Type` key, the conditional `headers` argument passed to `web.Response` is `None`; this does not assert that the framework emits no default headers.
Source: buster_ripper/routes/chat.py:155-160
Version: unavailable
Fingerprint: sha256:16c7e99938b55e508b36c16092e7a89883baeff293385c3dfa3e9c161b413cae
Invalidates-On: Change to the error response header expression in `buster_ripper/routes/chat.py`.
Checked-At: 2026-09-27
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-158.md

---

Claim: CL:serve_buster_ripper.sh/dump-format-env-default
Statement: `serve_buster_ripper.sh` initializes `DUMP_FORMAT` from the environment with an empty default
Subject: repo:serve_buster_ripper.sh
Verdict: HOLDS
Actual: Line 29 is `DUMP_FORMAT=${DUMP_FORMAT:-}`, which takes the inherited environment value when `DUMP_FORMAT` is set and non-empty, and otherwise assigns the empty string. It is the only assignment to `DUMP_FORMAT` in the script, so the variable is never given a non-empty fallback and never overwritten later.
Source: serve_buster_ripper.sh:29 (`DUMP_FORMAT=${DUMP_FORMAT:-}`); whole file read to confirm no other assignment
Version: unavailable (working-tree file, no version marker in the script itself — the fingerprint below is the basis)
Fingerprint: sha256:c58256fbec793c520720c9e86fee8740653939275b508f50b212ab60bd0908af
Invalidates-On: Line 29's default changes to a non-empty value (e.g. `${DUMP_FORMAT:-change-focused}`)
Invalidates-On: Any additional assignment to `DUMP_FORMAT` is introduced elsewhere in the script
Invalidates-On: The `DUMP_FORMAT` initialization line is removed or renamed
Checked-At: 2026-10-05
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-159.md

---

Claim: CL:serve_buster_ripper.sh/dump-format-flag-guard
Statement: `serve_buster_ripper.sh` appends a `--dump-format` flag to `EXTRA_FLAGS` only when `DUMP_FORMAT` is non-empty
Subject: repo:serve_buster_ripper.sh
Verdict: HOLDS
Actual: Line 41 is `[[ -n "${DUMP_FORMAT}" ]] && EXTRA_FLAGS+=(--dump-format "${DUMP_FORMAT}")`. The `-n` test is the sole gate on the append, so the pair `--dump-format <value>` is added to the `EXTRA_FLAGS` array only when `DUMP_FORMAT` expands to a non-empty string; when it is empty the guard short-circuits and nothing is appended. `--dump-format` appears nowhere else in the script, and `EXTRA_FLAGS` is only initialized (line 36) and appended to (lines 37-42) before being expanded into the `exec buster-ripper` command at line 54.
Source: serve_buster_ripper.sh:41 (`[[ -n "${DUMP_FORMAT}" ]] && EXTRA_FLAGS+=(--dump-format "${DUMP_FORMAT}")`); whole file read to confirm no other `--dump-format`/`EXTRA_FLAGS` writer
Version: unavailable (working-tree file, no version marker in the script itself — the fingerprint below is the basis)
Fingerprint: sha256:c58256fbec793c520720c9e86fee8740653939275b508f50b212ab60bd0908af
Invalidates-On: Line 41's guard changes from `-n` to unconditional, a different test, or a different variable
Invalidates-On: A second `--dump-format` append or an unconditional `EXTRA_FLAGS` write is added
Invalidates-On: `EXTRA_FLAGS` stops being an array (a string append would re-split the flag/value pair)
Checked-At: 2026-10-05
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-160.md

---

Claim: CL:serve_buster_ripper.sh/dump-dir-flag-guard
Statement: `serve_buster_ripper.sh` appends a `--dump-dir` flag to `EXTRA_FLAGS` only when `DUMP_DIR` is non-empty
Subject: repo:serve_buster_ripper.sh
Verdict: HOLDS
Actual: Line 40 is `[[ -n "${DUMP_DIR}" ]] && EXTRA_FLAGS+=(--dump-dir "${DUMP_DIR}")`. The `-n` (non-empty) test is the only gate on the append, so the pair `--dump-dir <value>` reaches the `EXTRA_FLAGS` array only when `DUMP_DIR` expands to a non-empty string. `--dump-dir` appears nowhere else in the script, and `EXTRA_FLAGS` is only initialized (line 36) and appended to (lines 37-42) before being expanded into the `exec buster-ripper` command at line 54.
Source: serve_buster_ripper.sh:40 (`[[ -n "${DUMP_DIR}" ]] && EXTRA_FLAGS+=(--dump-dir "${DUMP_DIR}")`); whole file read to confirm no other `--dump-dir`/`EXTRA_FLAGS` writer
Version: unavailable (working-tree file, no version marker in the script itself — the fingerprint below is the basis)
Fingerprint: sha256:c58256fbec793c520720c9e86fee8740653939275b508f50b212ab60bd0908af
Invalidates-On: Line 40's guard changes from `-n` to unconditional, a different test, or a different variable
Invalidates-On: A second `--dump-dir` append or an unconditional `EXTRA_FLAGS` write is added
Invalidates-On: Line 28's initialization changes (e.g. `${DUMP_DIR:-...}` instead of `${DUMP_DIR-...}`), which would alter when the `-n` guard can be false even though the guard itself is unchanged
Invalidates-On: `EXTRA_FLAGS` stops being an array (a string append would re-split the flag/value pair)
Checked-At: 2026-10-05
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-161.md

---

Claim: CL:serve_buster_ripper.sh/exec-line-option-assembly
Statement: `serve_buster_ripper.sh`'s `exec` line passes the always-on `--upstream`/`--host`/`--port` options and the assembled `EXTRA_FLAGS` to the `buster-ripper` binary
Subject: repo:serve_buster_ripper.sh
Verdict: HOLDS
Actual: The script's one and only `exec` statement (lines 50-54) invokes `buster-ripper` with `--upstream "${UPSTREAM}"`, `--host "${HOST}"` and `--port "${PORT}"` written literally on the exec line — unconditional, so "always-on" is accurate — and then appends the assembled array via `${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}`. Every conditional flag in the script is built into `EXTRA_FLAGS` (lines 36-42: `--strip-date`, `--verbose`, `--skip-tls-verify`, `--dump-dir <value>`, `--dump-format <value>`, `--preserved-thinking`) and reaches the binary only through that expansion, so the exec line carries both parts the claim names. Verified independently by reading the exec statement and the array-building block directly; the grep for `exec` in the file returns exactly one match (line 50), so there is no alternate exec path where this would differ.
Source: serve_buster_ripper.sh — lines 36-42 (EXTRA_FLAGS assembly) and lines 50-54 (the `exec` statement)
Version: unavailable as a revision string (mutable worktree file; Git metadata not read by the checking role — consulted bytes are identified by the fingerprint below)
Fingerprint: sha256:c58256fbec793c520720c9e86fee8740653939275b508f50b212ab60bd0908af
Invalidates-On: The `exec` statement drops, renames, or comments out any of `--upstream`/`--host`/`--port`.
Invalidates-On: Any of `--upstream`/`--host`/`--port` moves behind a conditional or into `EXTRA_FLAGS`, so it is no longer always-on.
Invalidates-On: The `${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}` expansion is removed from the exec line, or the array is replaced by a differently-named variable or a plain string.
Invalidates-On: A second `exec` (or non-`exec` invocation of the binary) is added that bypasses this line.
Invalidates-On: The sha256 fingerprint of `serve_buster_ripper.sh` above changes in any way.
Checked-At: 2026-10-05T04:38:29Z
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-162.md

---

Claim: CL:serve_buster_ripper.sh/positional-argument-forwarding
Statement: `serve_buster_ripper.sh` forwards no positional arguments to the exec'd binary
Subject: repo:serve_buster_ripper.sh
Verdict: HOLDS
Actual: The exec statement (lines 50-54) is made up entirely of long options and their option-arguments: `--upstream "${UPSTREAM}"`, `--host "${HOST}"`, `--port "${PORT}"`, plus whatever `EXTRA_FLAGS` contributes. Every `EXTRA_FLAGS` append (lines 37-42) is either a bare `--flag` or a `--flag "<quoted value>"` pair, so no array element can become a loose word — the block comment at lines 32-35 states that the array form exists precisely to stop word-splitting from handing the binary "an extra positional word". Separately, the script never touches its own positional parameters: a search of the file for `"$@"`, `$*`, `$1`-`$9`, `${N}` and `shift` returns no matches (the only hit for that pattern set is the `exec` keyword on line 50). So script-level arguments are neither accepted nor passed through, and nothing in the file re-executes the binary by another route. Term scope recorded by the check, not as doubt: "positional argument" is its standard CLI sense — an argument not bound to a named option; the values following `--upstream`/`--host`/`--port`/`--dump-dir`/`--dump-format` are option-arguments of options that declare them.
Source: serve_buster_ripper.sh — lines 32-42 (EXTRA_FLAGS assembly and its rationale) and lines 50-54 (the `exec` statement)
Version: unavailable as a revision string (mutable worktree file; Git metadata not read by the checking role — consulted bytes are identified by the fingerprint below)
Fingerprint: sha256:c58256fbec793c520720c9e86fee8740653939275b508f50b212ab60bd0908af
Invalidates-On: The exec line gains `"$@"`, `$*`, `$1`, or any script positional parameter.
Invalidates-On: An `EXTRA_FLAGS+=( ... )` append adds a value that is not a `--flag` and not directly preceded by the `--flag` it belongs to (e.g. `EXTRA_FLAGS+=("${DUMP_DIR}")`).
Invalidates-On: A flag's value is assembled unquoted or into a plain string variable that can word-split into an extra bare word.
Invalidates-On: Argument forwarding is introduced via `shift`, `getopts`, `set --`, or a wrapper array that reaches the exec line outside `EXTRA_FLAGS`.
Invalidates-On: The sha256 fingerprint of `serve_buster_ripper.sh` above changes in any way.
Checked-At: 2026-10-05T04:38:29Z
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-163.md

---

Claim: CL:README.md/collapsed-format-run-path
Statement: `README.md`'s dump section shows selecting a collapsed format on the serve-script run path it documents
Subject: repo:README.md
Verdict: HOLDS
Actual: The dump section's "### Collapsed dump formats (`--dump-format`)" subsection (which explicitly labels `change-focused` and `self-contained` as **collapsed** formats) shows `DUMP_FORMAT=change-focused DUMP_DIR=~/mllm/prompt-diffs bash serve_buster_ripper.sh` — i.e. a collapsed format selected via the `DUMP_FORMAT` env var on the same serve-script run path the dump section documents. The direct `buster-ripper --dump-format self-contained …` call appears only as the "or directly" alternative.
Source: README.md — "### Collapsed dump formats (`--dump-format`)" subsection of "## Dump mode", fenced bash block `DUMP_FORMAT=change-focused DUMP_DIR=~/mllm/prompt-diffs bash serve_buster_ripper.sh` (line 192 of the file)
Version: unavailable (working-tree README.md at repo root; git metadata not consulted)
Fingerprint: sha256:fe80cb9acabfe699fac070a69b43ac425436606d627b243825d4a028d75a1c15
Invalidates-On: The collapsed-formats subsection's serve-script example drops the `DUMP_FORMAT=…` env-var prefix on `bash serve_buster_ripper.sh` (e.g. only the direct `buster-ripper --dump-format …` call remains, or `DUMP_FORMAT` no longer maps to a collapsed format)
Invalidates-On: sha256 of README.md no longer equals fe80cb9acabfe699fac070a69b43ac425436606d627b243825d4a028d75a1c15
Checked-At: 2026-09-28
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-164.md

---

Claim: CL:tests.test_dump_write_count#_WriteHook/write-recording-gate
Statement: `tests/test_dump_write_count.py`'s `_WriteHook` records a write-bearing operation only for `Path.open` calls whose mode string contains no `"r"` and whose target path lies under the instrumented dump root
Subject: repo:tests.test_dump_write_count#_WriteHook
Verdict: HOLDS
Actual: `_WriteHook.covers` (the sole gate for recording) returns `"r" not in mode and path.is_relative_to(self.root)` — a `Path.open` call is wrapped in the recording `_RecordingFile` only when both conditions hold, and only that wrapper's `write` appends events. Modes containing `"r"` (including write-capable ones like `"r+"`) and opens outside the hook's `root` constructor argument return the bare handle and are never recorded.
Source: tests/test_dump_write_count.py — `_WriteHook.covers` (lines 106-108) and `_WriteHook.install`/`recording_open` (lines 92-104); recording occurs only via `_RecordingFile.write` (lines 74-76)
Version: unavailable
Fingerprint: sha256:63b2c4c8e049f9462d4b44277cb1a3c387680da2820154a3e583d601c814822e
Invalidates-On: any edit to `_WriteHook.covers` changes the mode filter (e.g. admitting `"r+"`) or drops the `is_relative_to(self.root)` check
Invalidates-On: any edit to `_WriteHook.install`/`recording_open` wraps handles without consulting `covers`, or records at open time rather than via `_RecordingFile.write`
Checked-At: 2026-09-28
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-165.md

---

Claim: CL:tests.test_dump_write_count#_WriteHook/payload-length-measurement
Statement: `_WriteHook` measures each recorded write's payload length as its UTF-8 encoded byte count
Subject: repo:tests.test_dump_write_count#_WriteHook
Verdict: HOLDS
Actual: `_WriteHook.record` computes `size = len(data.encode("utf-8")) if isinstance(data, str) else len(data)` and stores it as the event's `byte_len` — for the string payloads the measured writers issue (`dump.py` writes via `write_text` / `open("a")` + `write` of `str`), this is exactly the payload's UTF-8 encoded byte count; the defensive non-`str` branch falls back to `len(data)`, which for a bytes payload is likewise its byte count.
Source: tests/test_dump_write_count.py — `_WriteHook.record` (lines 110-113)
Version: unavailable
Fingerprint: sha256:63b2c4c8e049f9462d4b44277cb1a3c387680da2820154a3e583d601c814822e
Invalidates-On: any edit to `_WriteHook.record`'s size computation (e.g. `len(data)` without encoding, or a non-UTF-8 encoding)
Invalidates-On: any change to the measured writers that makes them issue non-`str` payloads where the semantic difference between `len(bytes)` and UTF-8 encoding would matter
Checked-At: 2026-09-28
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-166.md

---

Claim: CL:python/path-write-text-open-funnel
Statement: `pathlib.Path.write_text` performs its write by opening the target through `self.open(mode="w", …)`
Subject: lang:python
Verdict: HOLDS
Actual: `pathlib.Path` is `pathlib._local.Path`; its `write_text` delegates to `PathBase.write_text`, which performs the write exactly as `with self.open(mode='w', encoding=encoding, errors=errors, newline=newline) as f: return f.write(data)` (after a `TypeError` check that `data` is `str`). Since `self` is the concrete `Path`, `self.open` resolves to `_local.Path.open`, which opens the target via `io.open`.
Source: CPython 3.13.8 stdlib pathlib — `_abc.py` `PathBase.write_text` lines 644-652 (the `self.open(mode='w', ...)` call site), reached from `_local.py` lines 548-555; installed interpreter of the repo's `.venv` (uv-managed CPython 3.13.8)
Version: CPython 3.13.8 (stdlib `pathlib`, uv-managed interpreter used by this repo's `.venv`)
Fingerprint: sha256:75e6bc728013d11446126d535f92b8907fb6dae7404df3d0268ac7b6865a2090
Invalidates-On: the venv's interpreter changes (e.g. `.venv` rebuilt against a different Python) — re-locate the active `pathlib` source and re-check `write_text`
Invalidates-On: a Python upgrade where `write_text` stops routing through `self.open(mode='w', ...)` (e.g. a CPython change to `_abc.PathBase.write_text` or an override added in `_local.Path`)
Invalidates-On: CPython modifies these stdlib files in place (fingerprint change in the installed interpreter)
Checked-At: 2026-09-28T14:00:33Z
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-168.md

---

Claim: CL:tests.test_dump_records/collapsed-artifact-reads
Statement: `tests/test_dump_records.py` reads the collapsed `<format>.log` artifact with `read_text()` in its readability and turn-0 tests
Subject: repo:tests.test_dump_records
Verdict: HOLDS
Actual: The file contains exactly two `read_text()` calls, both reading `dump_env / sid / f"{fmt}.log"` where `fmt` comes from `COLLAPSED_FORMATS = ["change-focused", "self-contained"]`: line 212 in `TestArtifactReadability._flushed_artifact`, the shared helper invoked by both readability tests, and line 247 inside `TestArtifactTurnZero.test_body_and_rendering_recoverable_from_artifact`, parametrized over `COLLAPSED_FORMATS`. No other file-read mechanism (`open()`, `read_bytes`) appears in the file.
Source: tests/test_dump_records.py lines 212 and 247 (with `COLLAPSED_FORMATS` at line 21)
Version: unavailable
Fingerprint: sha256:d3084c680899d8760732435d6cbad7c9ec9d1f1a159cf5d3142298d5123f1444
Invalidates-On: `_flushed_artifact` (line 212) or the turn-0 test (line 247) switches from `.read_text()` to another read mechanism (e.g. `open().read()` or `read_bytes().decode()`)
Invalidates-On: the readability or turn-0 tests are removed/renamed so they no longer read `DUMP_DIR/<sid>/<format>.log`
Invalidates-On: `COLLAPSED_FORMATS` changes so the `{fmt}.log` artifact read is no longer a collapsed-format log
Checked-At: 2026-09-28T13:59:07Z
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-169.md

---

Claim: CL:tests.test_dump_write_count/fmt-parametrize-scope
Statement: `tests/test_dump_write_count.py` parametrizes exactly `TestOneWritePerFlush`, `TestFlushCountIsF`, `TestPairedMeasurement` and `TestFileCount` with `@pytest.mark.parametrize("fmt", COLLAPSED_FORMATS)`
Subject: repo:tests.test_dump_write_count
Verdict: HOLDS
Actual: The file contains exactly four `@pytest.mark.parametrize("fmt", COLLAPSED_FORMATS)` decorators (lines 164, 189, 239, 262), one on the single test method of each of `TestOneWritePerFlush`, `TestFlushCountIsF`, `TestPairedMeasurement`, and `TestFileCount` (the decorator sits on each class's one test method — the standard idiom — not on the `class` statement itself). The file's only other test classes, `TestWriteUnit` and `TestDocumentedCount`, carry no parametrize. `COLLAPSED_FORMATS` is defined at line 29 as `["change-focused", "self-contained"]`, so each of the four classes runs two `fmt` variants.
Source: tests/test_dump_write_count.py — lines 29, 141–275 (class/decorator layout confirmed by grep: decorators at 164, 189, 239, 262)
Version: unavailable
Fingerprint: sha256:63b2c4c8e049f9462d4b44277cb1a3c387680da2820154a3e583d601c814822e
Invalidates-On: any parametrize decorator added to or removed from any class in tests/test_dump_write_count.py
Invalidates-On: `COLLAPSED_FORMATS` renamed, removed, or its list membership changed (line 29)
Invalidates-On: a new test class added to the file carrying the same decorator
Checked-At: 2026-09-28
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-170.md

---

Claim: CL:rel:tests.test_dump_write_count#COLLAPSED_FORMATS→buster_ripper.config#DUMP_FORMATS/collapsed-format-set-equation
Statement: `tests/test_dump_write_count.py` binds `COLLAPSED_FORMATS` to `["change-focused", "self-contained"]`, exactly the two non-`documented` members of `config.DUMP_FORMATS`
Subject: rel:tests.test_dump_write_count#COLLAPSED_FORMATS→buster_ripper.config#DUMP_FORMATS
Verdict: HOLDS
Actual: `tests/test_dump_write_count.py` line 29 reads `COLLAPSED_FORMATS = ["change-focused", "self-contained"]` — the module's only definition of that name (all other occurrences at lines 164, 189, 239, 262 are `@pytest.mark.parametrize("fmt", COLLAPSED_FORMATS)` usages). `buster_ripper/config.py` line 23 defines `DUMP_FORMATS: tuple[str, ...] = ("documented", "change-focused", "self-contained")` — the closed set with no other definition or reassignment anywhere in the package (all other `DUMP_FORMATS` occurrences in `buster_ripper/dump.py` and `buster_ripper/main.py` are reads). The two non-`documented` members of that tuple are exactly `"change-focused"` and `"self-contained"`, matching the test constant element-for-element and even in order. The config comment (lines 20–22) independently corroborates the framing, calling them "the two collapsed values" against `"documented"`, today's per-turn layout and the default. The only type-level difference is list vs tuple; the claim asserts membership ("members"), which holds either way.
Source: tests/test_dump_write_count.py line 29 (the binding), jointly with buster_ripper/config.py line 23 (the format set the equation's second half cites)
Version: unavailable
Fingerprint: sha256:63b2c4c8e049f9462d4b44277cb1a3c387680da2820154a3e583d601c814822e
Fingerprint: sha256:7e1eff53eb56eb96aba4ff9badd0d9d16f3b02f062e2973676dcf8d670f71dde
Invalidates-On: `COLLAPSED_FORMATS` in tests/test_dump_write_count.py changes value, is renamed, or a second binding of that name appears in the module
Invalidates-On: `DUMP_FORMATS` in buster_ripper/config.py gains, drops, or renames a member (a third collapsed format or removal of one breaks the "exactly the two" equation)
Invalidates-On: `DUMP_FORMATS` becomes mutable at runtime or gains a second assignment site that changes its effective members
Checked-At: 2026-09-28
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-171.md

---

Claim: CL:tests.test_dump_write_count#TestWriteUnit/unparametrized-run-format-override
Statement: `tests/test_dump_write_count.py`'s `TestWriteUnit` executes one unparametrized run that hardcodes `config.DUMP_FORMAT = "change-focused"`
Subject: repo:tests.test_dump_write_count#TestWriteUnit
Verdict: HOLDS
Actual: `TestWriteUnit` (lines 141–158) contains a single test method, `test_mkdir_and_outside_paths_are_not_counted`, with no `parametrize` decorator, so it collects exactly one run. That method assigns `config.DUMP_FORMAT = "change-focused"` directly at line 149 (overriding the `"documented"` default the `dump_env` fixture set at line 129) before dumping its turns.
Source: tests/test_dump_write_count.py — lines 141–158 (assignment at line 149)
Version: unavailable
Fingerprint: sha256:63b2c4c8e049f9462d4b44277cb1a3c387680da2820154a3e583d601c814822e
Invalidates-On: the `config.DUMP_FORMAT = "change-focused"` assignment in TestWriteUnit changed or removed
Invalidates-On: a parametrize decorator added to TestWriteUnit (making the run parametrized rather than single)
Checked-At: 2026-09-28
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-172.md

---

Claim: CL:tests.test_dump_write_count#TestDocumentedCount/unparametrized-run-format-default
Statement: `tests/test_dump_write_count.py`'s `TestDocumentedCount` executes one unparametrized run on the `"documented"` dump format
Subject: repo:tests.test_dump_write_count#TestDocumentedCount
Verdict: HOLDS
Actual: `TestDocumentedCount` (lines 217–228) contains a single test method, `test_documented_run_counts_2n_writes`, with no `parametrize` decorator, so it collects exactly one run. The run executes under the `"documented"` format established by the shared `dump_env` fixture (line 129 sets `config.DUMP_FORMAT = "documented"`); the test body itself never overrides `DUMP_FORMAT`, so "documented" is the effective format throughout — noting for precision that the value comes from the fixture rather than an assignment inside the class body.
Source: tests/test_dump_write_count.py — lines 217–228 (fixture default at line 129)
Version: unavailable
Fingerprint: sha256:63b2c4c8e049f9462d4b44277cb1a3c387680da2820154a3e583d601c814822e
Invalidates-On: the `dump_env` fixture's `config.DUMP_FORMAT = "documented"` default (line 129) changed or removed
Invalidates-On: TestDocumentedCount gains a parametrize decorator or an explicit `config.DUMP_FORMAT` override
Checked-At: 2026-09-28
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-173.md

---

Claim: CL:buster_ripper.config/skip-tls-verify-global
Statement: `buster_ripper/config.py` declares the module global `SKIP_TLS_VERIFY` with default `False`
Subject: repo:buster_ripper.config
Verdict: HOLDS
Actual: Module-level statement `SKIP_TLS_VERIFY: bool = False` at `buster_ripper/config.py:17`, inside the "Core proxy settings" block. The module docstring states all other modules import it and read `config.<NAME>` so mutations by `main()` are visible everywhere, confirming its role as a mutable module global; the adjacent comment documents the `False` default as "certificates are verified unless the opt-out is explicitly requested" via `--skip-tls-verify` / `SKIP_TLS_VERIFY=1`.
Source: buster_ripper/config.py, line 17 (local production source in working tree)
Version: unavailable (local repo source, working tree; git metadata out of scope for this check)
Fingerprint: sha256:22348c68d21b91a30e75eeea802d4d390b2bcdabbf974b0ddc743c9354635522
Invalidates-On: The `SKIP_TLS_VERIFY` declaration at config.py:17 is removed or renamed
Invalidates-On: Its default initializer changes from `False` to any other value
Invalidates-On: The declaration moves out of `buster_ripper/config.py` (e.g., into a settings module or main)
Invalidates-On: The file bytes change such that the sha256 above no longer matches the consulted content
Checked-At: 2026-09-28
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-174.md

---

Claim: CL:buster_ripper.main/skip-tls-verify-option
Statement: `main.py` declares `--skip-tls-verify` as an `is_flag=True` option with `expose_value=False`
Subject: repo:buster_ripper.main
Verdict: HOLDS
Actual: Lines 64-67 declare `@click.option("--skip-tls-verify", is_flag=True, expose_value=False, callback=_set_skip_tls_verify, help=...)` on `main` — a boolean flag whose value is not passed to `main()` as a parameter; instead the callback `_set_skip_tls_verify` receives it at parse time. The claim's two assertions (`is_flag=True`, `expose_value=False`) both match verbatim.
Source: buster_ripper/main.py lines 64-67
Version: unavailable (working tree; fingerprint below)
Fingerprint: sha256:fd36dc12055f67693d9b852fb15a66aa4a3a7864103ac3de0ee67d9520048d47
Invalidates-On: The `--skip-tls-verify` option declaration changes — `is_flag` removed, `expose_value` flipped to true, or the option renamed/removed
Checked-At: 2026-09-28
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-175.md

---

Claim: CL:buster_ripper.main#_set_skip_tls_verify/config-assignment
Statement: `_set_skip_tls_verify` assigns `config.SKIP_TLS_VERIFY` from the option value
Subject: repo:buster_ripper.main#_set_skip_tls_verify
Verdict: HOLDS
Actual: Lines 30-33 define `_set_skip_tls_verify(ctx, param, value)` whose body is `config.SKIP_TLS_VERIFY = value` (then `return value`). Under click's callback protocol the `value` parameter is the parsed `--skip-tls-verify` option value (a bool, since the option is `is_flag=True`), so the config global is assigned directly from the option value at parse time. The target attribute `SKIP_TLS_VERIFY` exists at `config` module top level (initialized `False`), and no `skip_tls_verify` parameter is passed to `main()` because the option sets `expose_value=False`.
Source: buster_ripper/main.py lines 30-33 (callback) and 64-65 (wired via `callback=_set_skip_tls_verify` on the `--skip-tls-verify` option); corroborated by buster_ripper/config.py line 17 (`SKIP_TLS_VERIFY: bool = False`)
Version: unavailable (working tree; fingerprint below)
Fingerprint: sha256:fd36dc12055f67693d9b852fb15a66aa4a3a7864103ac3de0ee67d9520048d47 (buster_ripper/main.py); corroborating buster_ripper/config.py sha256:22348c68d21b91a30e75eeea802d4d390b2bcdabbf974b0ddc743c9354635522
Invalidates-On: The callback is deleted, renamed, or its body no longer assigns `config.SKIP_TLS_VERIFY` from its `value` parameter
Invalidates-On: `SKIP_TLS_VERIFY` is renamed or moved off the `config` module's top level
Invalidates-On: The `callback=_set_skip_tls_verify` wiring is removed from the `--skip-tls-verify` option
Checked-At: 2026-09-28
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-176.md

---

Claim: CL:buster_ripper.server#on_startup/tls-connector-construction
Statement: `on_startup` constructs its connector as `aiohttp.TCPConnector(ssl=not config.SKIP_TLS_VERIFY)`
Subject: repo:buster_ripper.server#on_startup
Verdict: HOLDS
Actual: `on_startup` (buster_ripper/server.py:65) constructs its connector with exactly that expression: `connector = aiohttp.TCPConnector(ssl=not config.SKIP_TLS_VERIFY)`. The referenced global exists as `SKIP_TLS_VERIFY: bool = False` in `buster_ripper/config.py:17`, imported into server.py via `from . import ... config` (server.py:9).
Source: buster_ripper/server.py line 65, inside `async def on_startup(app)` (lines 55–81); corroborated attribute definition at buster_ripper/config.py line 17
Version: unavailable (working-tree production source; identified by the sha256 fingerprint below)
Fingerprint: sha256:a9e49823ea456af9c0076781c4713c956745b21b026dfa768faf947e0b130592
Invalidates-On: any change to server.py's `on_startup` that alters the connector construction line (different `ssl` expression, different connector class, or removal of the explicit connector)
Invalidates-On: `config.SKIP_TLS_VERIFY` being renamed, moved, or `on_startup` starting to read the TLS posture from a different source than `config.SKIP_TLS_VERIFY`
Checked-At: 2026-09-28
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-177.md

---

Claim: CL:buster_ripper.server#on_startup/explicit-connector-session
Statement: `on_startup` passes the explicit connector to its `aiohttp.ClientSession` construction
Subject: repo:buster_ripper.server#on_startup
Verdict: HOLDS
Actual: `on_startup` (buster_ripper/server.py:66–68) constructs the session as `app["session"] = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600), connector=connector)`, where `connector` is the explicit `aiohttp.TCPConnector(ssl=not config.SKIP_TLS_VERIFY)` built on the preceding line (server.py:65) — so the session is constructed on that explicit connector, not on an aiohttp-created default connector.
Source: buster_ripper/server.py lines 65–68, inside `async def on_startup(app)` (lines 55–81)
Version: unavailable (working-tree production source; identified by the sha256 fingerprint below)
Fingerprint: sha256:a9e49823ea456af9c0076781c4713c956745b21b026dfa768faf947e0b130592
Invalidates-On: `on_startup`'s `ClientSession(...)` call dropping the `connector=connector` argument (or otherwise constructing the session without the explicit connector, e.g. via a nested/default connector)
Invalidates-On: the connector construction moving out of `on_startup` so the session no longer receives the startup-built connector
Checked-At: 2026-09-28
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-178.md

---

Claim: CL:serve_buster_ripper.sh/skip-tls-verify-env-default
Statement: `serve_buster_ripper.sh` initializes `SKIP_TLS_VERIFY` from the environment with default `0`
Subject: repo:serve_buster_ripper.sh
Verdict: HOLDS
Actual: Line 27 is `SKIP_TLS_VERIFY=${SKIP_TLS_VERIFY:-0}` — a shell parameter expansion that takes the value already exported in the environment and falls back to the literal `0` when the variable is unset or empty. The variable is not unconditionally reassigned, sourced from a config file, or given any other default elsewhere in the script, so an environment value other than `0` survives initialization.
Source: serve_buster_ripper.sh:27 (whole file read; 54 lines) — confirmed exhaustively by a search scoped to that one file: `SKIP_TLS_VERIFY` occurs only at line 16 (comment), line 27 (initialization) and line 39 (consumption)
Version: unavailable (mutable worktree file; no VCS revision consulted — stability is carried by the byte fingerprint below)
Fingerprint: sha256:c58256fbec793c520720c9e86fee8740653939275b508f50b212ab60bd0908af
Invalidates-On: Any edit to line 27 that replaces `${SKIP_TLS_VERIFY:-0}` with a hard assignment, a different default (e.g. `1`), or `${SKIP_TLS_VERIFY-0}`/`${SKIP_TLS_VERIFY:=0}` semantics that change the unset/empty distinction.
Invalidates-On: Introduction of another assignment to `SKIP_TLS_VERIFY` after line 27 (a later reinitialization would override the environment default).
Invalidates-On: Addition of a config-file or CLI-parse stage that sets `SKIP_TLS_VERIFY` before line 27 runs, or a change to the script's `set -u` status at line 2 that alters how unset environment variables are handled.
Invalidates-On: Any change to the bytes of `serve_buster_ripper.sh` (the fingerprint no longer matches, so the check must be re-run).
Checked-At: 2026-10-05
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-179.md

---

Claim: CL:serve_buster_ripper.sh/skip-tls-verify-flag-guard
Statement: `serve_buster_ripper.sh` appends a `--skip-tls-verify` flag to `EXTRA_FLAGS` only when `SKIP_TLS_VERIFY` equals `1`
Subject: repo:serve_buster_ripper.sh
Verdict: HOLDS
Actual: The flag is added by exactly one statement, line 39: `[[ "${SKIP_TLS_VERIFY}" == "1" ]] && EXTRA_FLAGS+=(--skip-tls-verify)`. The append is the right-hand side of an `&&` list guarded by a `[[ ]]` string-equality test, so `EXTRA_FLAGS` receives `--skip-tls-verify` if and only if the expansion of `"${SKIP_TLS_VERIFY}"` is the literal string `1`; when the guard does not match, the statement performs no append. Two consequences of the literal-string comparison are what "equals `1`" pins down: values such as `01`, `true`, `yes`, `2` or an empty value do not produce the flag, and the test is `==` against the quoted literal rather than an arithmetic `-eq`, so it is textual identity with `1` and not truthiness. Uniqueness of the append site was verified rather than inferred: `EXTRA_FLAGS` is created empty at line 36, appended to only at lines 37-42, and read once at line 54 (`${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}`), and a file-scoped search for the literal `--skip-tls-verify` matches line 39 only.
Source: serve_buster_ripper.sh:39 (whole file read; 54 lines) — plus the `EXTRA_FLAGS` lifecycle at lines 36, 37-42 and 54 and a file-scoped search for `--skip-tls-verify` to establish the "only when" half
Version: unavailable (mutable worktree file; no VCS revision consulted — stability is carried by the byte fingerprint below)
Fingerprint: sha256:c58256fbec793c520720c9e86fee8740653939275b508f50b212ab60bd0908af
Invalidates-On: Any change to the line-39 guard that broadens the condition (e.g. `!= "0"`, `-gt 0`, a `1|true|yes` pattern, or an `-n` non-empty test), which would append the flag for values other than `1`.
Invalidates-On: Adding any further statement that appends `--skip-tls-verify` to `EXTRA_FLAGS` unconditionally or under a different variable's guard.
Invalidates-On: Replacing the array `EXTRA_FLAGS` with a plain string, or routing the flag to the `exec buster-ripper` invocation at line 54 directly instead of through the array.
Invalidates-On: Any change to the bytes of `serve_buster_ripper.sh` (the fingerprint no longer matches, so the check must be re-run).
Checked-At: 2026-10-05
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-180.md

---

Claim: CL:README.md/skip-tls-verify-run-path
Statement: `README.md` documents `SKIP_TLS_VERIFY=1 bash serve_buster_ripper.sh` as the TLS opt-out's run path
Subject: repo:README.md
Verdict: HOLDS
Actual: In the "Serve script options" section, README.md's fenced bash block presents `SKIP_TLS_VERIFY=1 bash serve_buster_ripper.sh` (comment: "Skip upstream TLS certificate verification — for self-signed certificates. … SKIP_TLS_VERIFY=1 turns it off process-wide (--skip-tls-verify)") — i.e. the TLS-verification opt-out is documented via the serve-script run path with the env var, not via a direct `buster-ripper` invocation.
Source: README.md — "## Serve script options" section, the `SKIP_TLS_VERIFY=1 bash serve_buster_ripper.sh` line of the fenced bash block (lines 89–92 of the file)
Version: unavailable (working-tree README.md at repo root; git metadata not consulted)
Fingerprint: sha256:fe80cb9acabfe699fac070a69b43ac425436606d627b243825d4a028d75a1c15
Invalidates-On: README.md's serve-script-options block no longer shows `SKIP_TLS_VERIFY=1` as an env-var prefix on `bash serve_buster_ripper.sh` (e.g. opt-out moved to a direct `buster-ripper --skip-tls-verify` example or removed)
Invalidates-On: sha256 of README.md no longer equals fe80cb9acabfe699fac070a69b43ac425436606d627b243825d4a028d75a1c15
Checked-At: 2026-09-28
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-181.md

---

Claim: CL:buster_ripper/per-request-ssl-absence
Statement: No call site under `buster_ripper/` passes a per-request `ssl` argument to a session request method
Subject: repo:buster_ripper
Verdict: HOLDS
Actual: A grep for `ssl` across `buster_ripper/` yields exactly one occurrence: `server.py:65`, `aiohttp.TCPConnector(ssl=not config.SKIP_TLS_VERIFY)` — a connector-constructor argument applied session-wide at construction, not a per-request argument on a request method. Reading all six session request call sites confirms their keyword arguments contain no `ssl`: `backend.py:54` (timeout), `compaction.py:36` (timeout), `routes/chat.py:125` (json, headers, timeout), `routes/messages.py:130` (json, headers), `routes/messages.py:231` (json, headers), `routes/passthrough.py:29` (method, url, data, headers, params, timeout). TLS posture is set once on the connector in `on_startup`, per that function's docstring.
Source: buster_ripper/server.py line 65 — the sole `ssl` occurrence under `buster_ripper/`, a `TCPConnector` kwarg rather than a session-request kwarg
Version: unavailable (working-tree source of the buster-ripper repo; git metadata not consulted)
Fingerprint: sha256:a9e49823ea456af9c0076781c4713c956745b21b026dfa768faf947e0b130592
Invalidates-On: An `ssl=` keyword added to any `session.get/post/request/...` call site under `buster_ripper/`
Invalidates-On: A new session request call site introduced with a per-request `ssl` argument
Invalidates-On: Removal of the connector-level `ssl` at server.py:65 (would not make the claim false, but signals the TLS-posture mechanism moved — re-check where `ssl` lands)
Checked-At: 2026-09-28
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-182.md

---

Composition: CO:cli-flag-to-outbound-call/tls-posture-crossing
Members: CL:buster_ripper.main/skip-tls-verify-option, CL:buster_ripper.main#_set_skip_tls_verify/config-assignment, CL:buster_ripper.config/skip-tls-verify-global, CL:buster_ripper.main/cli-default-binding-time, CL:buster_ripper.server#on_startup/tls-connector-construction, CL:buster_ripper.server#on_startup/explicit-connector-session, CL:buster_ripper/client-session-construction, CL:buster_ripper/outbound-call-routing, CL:buster_ripper/per-request-ssl-absence, CL:aiohttp/ssl-parameter-topology, CL:aiohttp/ssl-verification-polarity, CL:aiohttp/connector-loop-binding, CL:click/option-callback-timing
Boundary: the TLS posture (`ssl` True/False) crossing the callback-wired `--skip-tls-verify` flag → `config.SKIP_TLS_VERIFY` → the explicit `TCPConnector(ssl=…)` construction into every outbound call
Dimensions: posture polarity (opt-out, default off, never inferred from scheme); import-bound decorator defaults versus explicit callback; process-wide one-connector scope with no per-request `ssl` override; loop binding at connector construction (session adopts `connector._loop`); unset-path behaviour preservation
Verdict: SAFE
Counterexample: none
Source: buster_ripper/main.py:30-33,36-58,61-67,86-95,131-189; buster_ripper/config.py:1-17; buster_ripper/server.py:1-90; buster_ripper/backend.py:46-57; buster_ripper/compaction.py:28-39; buster_ripper/routes/chat.py:34-39,122-129; buster_ripper/routes/messages.py:125-133,226-233; buster_ripper/routes/passthrough.py:25-32; installed aiohttp client.py + connector.py and click core.py (repo `.venv`: aiohttp 3.13.3, click 8.3.1; every click semantic re-verified against the repo venv)
Fingerprint: sha256:9e94ac6b1c63f9131979ba00c5c151307975467a07f6777ba15881300740d2e7
Invalidates-On: any member fingerprint recorded by the composition changing (member re-extraction / re-verification required before recomposition)
Invalidates-On: removal or rewiring of `--skip-tls-verify`, its `is_flag`/`expose_value=False` shape, or `callback=_set_skip_tls_verify`; any second writer of `config.SKIP_TLS_VERIFY` or rename of the global; a change to its `False` default
Invalidates-On: any change to `server.py:65-68` (different `ssl` expression, connector class, session built without the explicit connector, connector constructed outside the running loop)
Invalidates-On: any per-request `ssl=`/`verify_ssl=`/`ssl_context=`/`fingerprint=` added to a session call site, any new outbound HTTP mechanism or second `ClientSession` under `buster_ripper/`, or any call site leaving `app["session"]`
Invalidates-On: aiohttp changes to `_get_ssl_context` precedence or scheme gating, `ClientSession.__init__` (connector/loop adoption), or `TCPConnector.__init__` (`ssl` default/`_merge_ssl_params`); click changes to parse-time callback invocation or UNSET-flag resolution; `.venv`/`uv.lock` drift off aiohttp 3.13.3 / click 8.3.1
Invalidates-On: `main.py` body reordering that lets any `config.SKIP_TLS_VERIFY` read precede the parse-time assignment
Checked-At: 2026-09-28
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/compositions/seam-S4.md

---

Claim: CL:bash/colon-default-value-form-semantics
Statement: bash's `${VAR:-word}` substitutes `word` when `VAR` is unset or empty
Subject: lang:bash
Verdict: HOLDS
Actual: `${parameter:-word}` is bash's "Use Default Values" form and bash tests for a parameter that is *unset or null*: if either holds, the expansion of `word` is substituted, otherwise the value of `parameter` is. Bash defines the empty string as a valid value ("A parameter is set if it has been assigned a value. The null string is a valid value."), so "empty" is exactly the null case and not a third state. The manual's worked example demonstrates both triggers: `v=` with `${v:-unset-or-null}` yields the default, and `unset v` yields the default.
Source: GNU Bash Reference Manual, node "Shell Parameter Expansion" — doc/bashref.texi at https://raw.githubusercontent.com/mirror/bash/master/doc/bashref.texi ("Bash tests for a parameter that is unset or null" plus the `@item ${parameter:-word}` entry and its example block); corroborated by bash(1), section PARAMETER EXPANSION, at https://manpages.debian.org/testing/bash/bash.1.en.html
Version: GNU Bash 5.3 (doc/version.texi EDITION/VERSION 5.3, updated 18 May 2025; master-branch bytes as fetched); corroborating bash(1) from Debian package bash 5.3-3+b1
Fingerprint: sha256:f3d37d57a1061e24d266051de9bd47ffa43dc86584afea11576c535ad2be32d5 (full doc/bashref.texi, 415804 bytes); sha256:13669991f07ceabe7e30f8ee462af6d24176f02e8db1bd905a3f63dbd7b8554d (consulted excerpt, 802 bytes); sha256:a827e4d7e765538cfbb239118accd7b2d45f610cd474c496e119eb57861fc5e5 (bash(1) man page HTML, 532248 bytes)
Invalidates-On: the fetched doc/bashref.texi bytes change, or the "Use Default Values" entry is reworded away from "unset or null"
Invalidates-On: a bash release that changes `:-` to test only for unset (collapsing it onto `-`), or that redefines "null" so an empty assignment no longer counts
Invalidates-On: the mutable mirror branch diverges — re-verify against the digests above, or against www.gnu.org/software/bash/manual/html_node/Shell-Parameter-Expansion.html when reachable
Checked-At: 2026-10-04
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-183.md

---

Claim: CL:bash/no-colon-default-value-form-semantics
Statement: bash's `${VAR-word}` substitutes `word` only when `VAR` is unset
Subject: lang:bash
Verdict: HOLDS
Actual: The no-colon form tests only for existence: bash states that omitting the colon "results in a test only for a parameter that is unset", so `${VAR-word}` substitutes `word` when `VAR` is unset and the value of `VAR` otherwise — including when that value is the empty string. The manual's worked example separates the three states: `v=123` → `${v-unset}` prints `123`, `unset v` → prints `unset`, `v=` → prints nothing while `${v:-unset-or-null}` at the same value prints the default. That last pair is what makes "only" true rather than implied.
Source: GNU Bash Reference Manual, node "Shell Parameter Expansion" — doc/bashref.texi at https://raw.githubusercontent.com/mirror/bash/master/doc/bashref.texi ("Omitting the colon results in a test only for a parameter that is unset" plus the `@item ${parameter:-word}` example block exercising `${v-unset}` against `v=123`, unset `v`, and `v=`); corroborated by bash(1) PARAMETER EXPANSION at https://manpages.debian.org/testing/bash/bash.1.en.html
Version: GNU Bash 5.3 (doc/version.texi EDITION/VERSION 5.3, updated 18 May 2025); the prose rule is identical in bash(1) from Debian package bash 5.3-3+b1 — only the worked example block is new in the current manual
Fingerprint: sha256:f3d37d57a1061e24d266051de9bd47ffa43dc86584afea11576c535ad2be32d5 (full doc/bashref.texi, 415804 bytes); sha256:13669991f07ceabe7e30f8ee462af6d24176f02e8db1bd905a3f63dbd7b8554d (consulted excerpt, 802 bytes); sha256:a827e4d7e765538cfbb239118accd7b2d45f610cd474c496e119eb57861fc5e5 (bash(1) man page HTML, 532248 bytes)
Invalidates-On: the fetched doc/bashref.texi bytes change, or the "Omitting the colon results in a test only for a parameter that is unset" sentence is reworded
Invalidates-On: a bash release that makes the no-colon forms also test for null (collapsing `-` onto `:-`), or that changes which forms the "when not performing substring expansion" qualifier covers
Invalidates-On: the mutable mirror branch diverges — re-verify against the digests above, or against www.gnu.org/software/bash/manual/html_node/Shell-Parameter-Expansion.html when reachable
Checked-At: 2026-10-04
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-184.md

---

Claim: CL:bash/prefix-assignment-environment
Statement: A prefixed assignment `VAR="" cmd` leaves `VAR` set-and-empty in `cmd`'s environment
Subject: lang:bash
Verdict: HOLDS
Actual: Both halves are stated and nothing carves out empty values. `VAR=""` sets the variable — for `name=[value]`, "If value is not given, the variable is assigned the null string", and the null string is a valid value, so `VAR=""` and `VAR=` are the same thing. With a command name present, "the variables are added to the environment of the executed command and do not affect the current shell environment", and the environment is "a list of name-value pairs, of the form name=value"; the Environment node says the same for prefix assignments specifically ("part of that command's environment for as long as it executes"). The pair added therefore carries `VAR` present with an empty value. Scope note: when `cmd` names a shell function the assignments are local to the function and exported to its children — still set-and-empty for `cmd`.
Source: GNU Bash Reference Manual — doc/bashref.texi at https://raw.githubusercontent.com/mirror/bash/master/doc/bashref.texi, nodes "Environment", "Simple Commands" (Simple Command Expansion), and "Parameters"; corroborated by bash(1) sections PARAMETERS and SIMPLE COMMAND EXPANSION at https://manpages.debian.org/testing/bash/bash.1.en.html
Version: GNU Bash 5.3 (doc/version.texi EDITION/VERSION 5.3, updated 18 May 2025); corroborating bash(1) from Debian package bash 5.3-3+b1
Fingerprint: sha256:f3d37d57a1061e24d266051de9bd47ffa43dc86584afea11576c535ad2be32d5 (full doc/bashref.texi, 415804 bytes); sha256:a827e4d7e765538cfbb239118accd7b2d45f610cd474c496e119eb57861fc5e5 (bash(1) man page HTML, 532248 bytes)
Invalidates-On: the fetched doc/bashref.texi bytes change, or the Environment / Simple Command Expansion sentences are reworded
Invalidates-On: a bash release documenting null-valued prefix assignments as omitted from the command environment (i.e. `VAR="" cmd` dropping the pair instead of adding `VAR=`)
Invalidates-On: the claim being re-scoped to a special builtin, to `VAR=""` with no command name, or to `set -o posix`, where the manual routes the assignment to the current shell environment — a different claim this verdict does not cover
Checked-At: 2026-10-04
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-185.md

---

Claim: CL:bash/nounset-default-value-exemption
Statement: bash's `${VAR-word}` on an unset `VAR` raises no nounset error under `set -u`
Subject: lang:bash
Verdict: HOLDS
Actual: The unbound-variable diagnostic has exactly one site in brace expansion, and the default-value operators do not pass its gate. `parameter_brace_expand` looks the name up with unbound errors suppressed (`PF_IGNUNBOUND`), then guards the sole error site with `if (want_substring || want_patsub || want_casemod || c == '@' || c == '#' || c == '%' || c == RBRACE)`; for `${VAR-word}` (and for `${VAR:-word}`, where the parser consumes the colon and leaves `c == '-'`) none of those hold, so `err_unboundvar` is never reached and control falls into `case '-':` and substitutes the default. Plain `${VAR}` (`c == RBRACE`) does error — that is the distinction. Bash's documentation never states the exemption: it is established by the implementation, not the manual.
Source: GNU Bash source, `subst.c`, function `parameter_brace_expand` (the `PF_IGNUNBOUND` lookup, the "All the cases where an expansion can possibly generate an unbound variable error" gate, and the `case '-': case '=': case '?': case '+':` block) at https://raw.githubusercontent.com/mirror/bash/refs/tags/bash-5.3/subst.c; secondary (documentation, for what the manual does and does not say): bash(1) at https://manpages.debian.org/testing/bash/bash.1.en.html and doc/bashref.texi at https://raw.githubusercontent.com/mirror/bash/master/doc/bashref.texi
Version: GNU Bash 5.3 (refs/tags/bash-5.3), cross-checked against master (5.4-devel) and bash-5.1 for stability of the gate condition
Fingerprint: sha256:cf96a7f33e7f9281f18c7b02d8840ad2d817f14243dd38377f8090249a7edf85 (bash-5.3 subst.c, 382887 bytes); sha256:501f0941105ce39281f3c60c8e5853d8e77f1558ca2383e1757b8c2cc0a3ba56 (master subst.c, 383471 bytes); sha256:22012e6279690e372ad9112cca5918ac505ab42bb475022640d76b5b4aec4201 (bash-5.1 subst.c, 339642 bytes); sha256:f3d37d57a1061e24d266051de9bd47ffa43dc86584afea11576c535ad2be32d5 (consulted doc/bashref.texi, 415804 bytes)
Invalidates-On: the `unbound_vars_is_error` gate in `parameter_brace_expand` gaining a `'-'`/`':'` (or `check_nullness`-based) clause, or `err_unboundvar` being added to the `case '-'` default-value branch
Invalidates-On: a bash release changing the `PF_IGNUNBOUND` treatment of the brace-expansion name lookup
Invalidates-On: bash ever documenting the exemption differently — documentation then becomes the authority, and none exists as of the digests above
Checked-At: 2026-10-04
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-186.md

---

Claim: CL:bash/dash-n-primary-semantics
Statement: bash's `[[ -n s ]]` conditional expression is true only for a non-empty string
Subject: lang:bash
Verdict: HOLDS
Actual: `-n` is true exactly when the length of its operand is non-zero, and its companion `-z string` is true exactly when that length is zero, so `-n` is false only for the empty string. The bare form `string` is documented in the same entry as equivalent to `-n string`, and inside `[[ ... ]]` the words between the brackets do not undergo word splitting or filename expansion, so the literal `s` is a one-character, non-zero-length string and the expression as written evaluates true on that ground. No part of the primary's truth depends on the operand's content beyond its length.
Source: https://www.gnu.org/software/bash/manual/html_node/Bash-Conditional-Expressions.html — section 6.4 "Bash Conditional Expressions" (`-n string` / `string` — "True if the length of string is non-zero.", with `-z string` immediately above); consulted bytes are the HTML retrieved from that exact URL (13,125 bytes), read with markup stripped
Version: The GNU Bash Reference Manual, Edition 5.3, last updated 18 May 2025, for Bash version 5.3 (stated in the fetched document's header comment)
Fingerprint: sha256:401cf8ecf01b3aeb29423a6c5fb6f971083fbb733d4e4020261a6b022e9709fd
Invalidates-On: any regeneration or edit of that manual node changing the `-n`/`-z` wording (the fingerprint stops matching)
Invalidates-On: a bash release changing unary string-primary semantics, or letting `[[ ]]` collapse an empty expansion into a different truth value for `-n`
Invalidates-On: reuse against a shell other than bash (`sh`, `dash`, `ksh`, `zsh`), or against the `test`/`[` builtins, whose behaviour the manual says is determined by argument count and documents separately
Checked-At: 2026-10-05T01:47:58Z
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-187.md

---

Claim: CL:bash/errexit-and-list-exemption
Statement: In bash, a command left of `&&` that fails in an `&&` list does not trigger errexit under `set -e`
Subject: lang:bash
Verdict: HOLDS
Actual: The `-e` bullet carves the exemption out explicitly: "The shell does not exit if the command that fails is … part of any command executed in a && or || list except the command following the final && or ||". In `A && B` the only command following the final `&&` is `B`, so a failing `A` short-circuits the list without ending the shell — the list still returns `A`'s non-zero status, but errexit does not fire on it. The same holds for every position but the last: in `A && B && C` both `A` and `B` are exempt and only `C` can trigger the exit.
Source: https://www.gnu.org/software/bash/manual/html_node/The-Set-Builtin.html — section 4.3.1 "The Set Builtin", the `-e` / `errexit` bullet; consulted bytes are the HTML retrieved from that exact URL (18,020 bytes), read with `<code>`/`<samp>` markup stripped
Version: The GNU Bash Reference Manual, Edition 5.3, last updated 18 May 2025, for Bash version 5.3 (stated in the fetched document's header comment)
Fingerprint: sha256:ca70b4480745a3310417bc974ef3a4c74a1cb335156557fc114948ec006fa9aa
Invalidates-On: any regeneration or edit of that manual node changing the `-e` exemption clause (the fingerprint stops matching)
Invalidates-On: a bash release narrowing the exemption so a non-final command in an `&&`/`||` list can trigger errexit
Invalidates-On: reuse where the command left of `&&` is itself the final command of an enclosing list, or where the list runs in a context where `-e` is already ignored — the manual's separate compound-command/function paragraphs govern those
Invalidates-On: reuse against a shell other than bash, or against POSIX `sh` errexit wording
Checked-At: 2026-10-05T01:47:58Z
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-188.md

---

Claim: CL:bash/unquoted-expansion-word-splitting
Statement: An unquoted bash parameter expansion word-splits, and an empty value contributes zero words
Subject: lang:bash
Verdict: HOLDS
Actual: Both halves are the documented default rule. (a) The shell scans the results of parameter expansion that did not occur within double quotes for word splitting, treating each character of `$IFS` as a field terminator — so an unquoted `$var` holding `a b` becomes two words, and its contents are then also eligible for pathname expansion. (b) An empty unquoted expansion yields no field at all: "Unquoted implicit null arguments, resulting from the expansion of parameters that have no values, are removed", and since word splitting strips leading/trailing IFS whitespace and "null fields result only from quoting", an empty result leaves zero words; the manual states the removal explicitly even for the degenerate case ("If the value of IFS is null, no word splitting occurs, but implicit null arguments … are still removed."). Documented scope limits that bound where the rule applies rather than contradicting it: word splitting is not performed on an assignment's right-hand side, inside `[[ … ]]`, or in arithmetic evaluation, and `set -f` suppresses only the later pathname-expansion step. Explicitly quoted nulls (`""`, `''`) are the contrast case — retained and passed as empty strings.
Source: https://man7.org/linux/man-pages/man1/bash.1.html — man page sections EXPANSION → "Word Splitting" and EXPANSION → "Pathname Expansion", plus PARAMETERS ("A parameter is set if it has been assigned a value. The null string is a valid value."); quoted text verified present in the fetched document. Same text is GNU Bash Reference Manual §3.5.7 "Word Splitting" (https://www.gnu.org/software/bash/manual/html_node/Word-Splitting.html — fetch timed out in the check; the mirrored man page carries the identical wording)
Version: GNU Bash 5.3 man page (page header "GNU Bash 5.3 / 2025 April 7"); man7 copy obtained from upstream git://git.savannah.gnu.org/bash.git on 2026-08-03 (most recent upstream commit 2026-06-10) — not the locally installed shell, which was not executed for this check
Fingerprint: sha256:e35fa2966c44a4e9ee519760160f1aba14c670451a463177968ed2c4710bf846 (exact fetched bytes of https://man7.org/linux/man-pages/man1/bash.1.html); sha256:b44556511b41581c65d6dc2505ca306472c6b7f4a75c1c056c1f155ffe1010cb (tag-stripped, whitespace-normalized text of those same bytes, from which the quotations were read)
Invalidates-On: A bash release that changes Word Splitting (e.g. an explicit null-argument retention change or a new shell option that suppresses splitting of unquoted expansions) — recheck the "Word Splitting" section of the Bash 5.4+ manual.
Invalidates-On: A manual revision that rewords "Unquoted implicit null arguments … are removed" or moves the IFS-null sentence (fingerprint above changes).
Invalidates-On: Reuse against a shell that is not GNU bash (dash/sh/busybox ash) or a bash running in posix mode with `--posix`, where null-argument handling is specified by POSIX rather than by this section.
Invalidates-On: Reuse in a context the manual exempts (assignment RHS, `[[ ]]`, arithmetic), where the splitting step does not run at all.
Checked-At: 2026-10-05T04:46:10Z
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-189.md

---

Claim: CL:bash/path-command-search-order
Statement: bash executes the first matching executable found in `$PATH`, so an earlier entry shadows an installed command
Subject: lang:bash
Verdict: HOLDS
Actual: For a command name containing no slashes, and only after shell functions and builtins are ruled out, bash "searches each element of $PATH for a directory containing an executable file by that name" and executes what the search resolves. The name resolves to exactly one directory, so a matching executable in an earlier element prevents the search reaching a same-named executable in a later element — the earlier entry shadows it. Matching is executable-specific (a non-executable file of that name is not a match and the search proceeds). Two qualifiers, consistent with the claim: the function and builtin checks precede `$PATH`, and bash caches resolved pathnames in a hash table, performing a full `$PATH` search only when the command is not in the hash — so shadowing is order-governed but not guaranteed to take effect in an existing shell until the table is cleared.
Source: GNU Bash Reference Manual, section 3.7.2 "Command Search and Execution" — https://www.gnu.org/software/bash/manual/html_node/Command-Search-and-Execution.html ("If the name is neither a shell function nor a builtin, and contains no slashes, Bash searches each element of $PATH for a directory containing an executable file by that name.")
Version: Bash Reference Manual, Edition 5, documenting bash version 5.3 (version string present in the fetched page); section numbering 3.7.2 matches that edition
Fingerprint: sha256:8182c7cba5445d0a623b60860c8b8ebc65a09dce273a068cd0d1182f31e5c184
Invalidates-On: the Command-Search-and-Execution.html page (Edition 5 / bash 5.3 text above) changing how `$PATH` is searched — e.g. a last-match or all-match resolution, or removal of the function/builtin-before-`$PATH` ordering
Invalidates-On: a bash release whose manual resolves `$PATH` entries differently (a PATHORDER-style option, or hash behaviour rechecking later entries before honouring an earlier one)
Invalidates-On: the claim being re-scoped to a shell that resolves commands differently, or to a command name containing a slash (then `$PATH` is not consulted at all)
Invalidates-On: re-scoping to a wrapper that resolves the command itself (a launcher, `env`/`xargs` with an explicit path, or an absolute-path invocation), bypassing bash's `$PATH` search
Checked-At: 2026-10-05T01:51:01Z
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-190.md

---

Claim: CL:serve_buster_ripper.sh/usage-header-dump-dir-documentation
Statement: `serve_buster_ripper.sh`'s usage header documents `DUMP_DIR=""` as disabling prompt diffing
Subject: repo:serve_buster_ripper.sh
Verdict: HOLDS
Actual: The `# Usage:` comment block (lines 11–16) lists `DUMP_DIR="" bash serve_buster_ripper.sh` on line 14 with the trailing annotation `# disable prompt diffing`. The header therefore documents the empty-string setting exactly as the claim states. Line 40 (`[[ -n "${DUMP_DIR}" ]] && EXTRA_FLAGS+=(--dump-dir "${DUMP_DIR}")`) corroborates that the documented behaviour matches the implementation — an empty `DUMP_DIR` suppresses the `--dump-dir` flag — but the claim is about what the header says, and line 14 establishes that on its own.
Source: serve_buster_ripper.sh — line 14 (usage header)
Version: unavailable (mutable worktree file; no git metadata consulted — the fingerprint below is the basis)
Fingerprint: sha256:c58256fbec793c520720c9e86fee8740653939275b508f50b212ab60bd0908af
Invalidates-On: any edit to the usage header (lines 11–16) that removes the `DUMP_DIR=""` example or rewords its "disable prompt diffing" annotation
Invalidates-On: the file's bytes changing from sha256:c58256fbec793c520720c9e86fee8740653939275b508f50b212ab60bd0908af
Checked-At: 2026-10-05T04:38:23Z
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-191.md

---

Claim: CL:serve_buster_ripper.sh/shebang-interpreter
Statement: `serve_buster_ripper.sh`'s shebang names `/usr/bin/env bash` as its interpreter
Subject: repo:serve_buster_ripper.sh
Verdict: HOLDS
Actual: Line 1 is the shebang `#!/usr/bin/env bash` — the interpreter named is `/usr/bin/env bash`, i.e. env-resolved `bash`, not an absolute path to a bash binary. The comment at line 48 independently references "this script's `/usr/bin/env bash`" as its interpreter, which corroborates the reading of line 1 but is not what establishes the claim.
Source: serve_buster_ripper.sh — line 1
Version: unavailable (mutable worktree file; no git metadata consulted — the fingerprint below is the basis)
Fingerprint: sha256:c58256fbec793c520720c9e86fee8740653939275b508f50b212ab60bd0908af
Invalidates-On: any edit changing line 1 to a different interpreter (e.g. `#!/bin/bash`, `#!/usr/bin/env zsh`, or removing the shebang)
Invalidates-On: the file's bytes changing from sha256:c58256fbec793c520720c9e86fee8740653939275b508f50b212ab60bd0908af
Checked-At: 2026-10-05T04:38:23Z
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-192.md

---

Claim: CL:serve_buster_ripper.sh/exec-line-command-name
Statement: `serve_buster_ripper.sh`'s exec line names the binary as the bare command `buster-ripper`
Subject: repo:serve_buster_ripper.sh
Verdict: HOLDS
Actual: Line 50 reads `exec buster-ripper \` — the command word is the bare token `buster-ripper`. It carries no directory component (no `/`, no `./`, no absolute path), no variable or command substitution, and no launcher prefix such as `uv run` or `python -m`, so resolution is left to the shell's `PATH` lookup at exec time. A search of the file for `exec` returns exactly this one match, so there is no second invocation site naming the binary differently. The header comments at lines 7-9 describe `uv tool install .` / `uv run buster-ripper --help` as prerequisites — consistent with a PATH-resolvable console script named `buster-ripper`, but that is corroboration only; the verdict rests on line 50. Explicitly out of scope by role boundary: whether that bare name actually resolves on a given machine, which is state rather than source.
Source: serve_buster_ripper.sh — line 50 (the `exec` statement's command word), with lines 1-9 read for context
Version: unavailable as a revision string (mutable worktree file; Git metadata not read by the checking role — consulted bytes are identified by the fingerprint below)
Fingerprint: sha256:c58256fbec793c520720c9e86fee8740653939275b508f50b212ab60bd0908af
Invalidates-On: Line 50's command word becomes a path (`./buster-ripper`, `/usr/local/bin/buster-ripper`), a variable (`"${BIN}"`), or a substitution (`"$(command -v buster-ripper)"`).
Invalidates-On: A launcher is prefixed to the exec (`uv run buster-ripper`, `python -m buster_ripper`, `env ... buster-ripper`), so the bare command is no longer what is named.
Invalidates-On: The console-script name changes in the packaging metadata the install uses, so the bare token on the exec line is stale.
Invalidates-On: An additional `exec`/invocation site is added that names the binary by other means.
Invalidates-On: The sha256 fingerprint of `serve_buster_ripper.sh` above changes in any way.
Checked-At: 2026-10-05T04:38:29Z
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-193.md

---

Claim: CL:README.md/dump-dir-example-set
Statement: No `README.md` example uses the `DUMP_DIR=""` disable form
Subject: repo:README.md
Verdict: HOLDS
Actual: Every `DUMP_DIR` example in README.md assigns a real path (`~/mllm/prompt-diffs`), and the one CLI-form example passes `--dump-dir ~/mllm/prompt-diffs`. The file has four `DUMP_DIR` assignments total (lines 84, 145, 192, 194) and all four are enable-style; the remaining occurrences are a heading (line 139) and two prose path templates (lines 151, 180). Negative probes were all empty: no empty-string assignment in either quote style, no empty-string `--dump-dir` argument, and no alternative disable idiom (`:`, `0`, `off`, `false`, `null`, `none`).
Source: README.md — read in full (289 lines, 10287 bytes) and exhaustively enumerated; all 7 occurrences of `DUMP_DIR` / `--dump-dir` / `dump_dir` at lines 84, 139, 145, 151, 180, 192, 194
Version: unavailable (README.md working-tree bytes, fingerprinted rather than versioned by revision)
Fingerprint: sha256:fe80cb9acabfe699fac070a69b43ac425436606d627b243825d4a028d75a1c15
Invalidates-On: any edit to README.md adding a `DUMP_DIR=""` (or `--dump-dir ""`) example — the fingerprint above changes
Invalidates-On: the claim being widened beyond README.md — disable forms in docs/eval.md, serve_buster_ripper.sh comments, or the `buster-ripper --help` epilogue would not falsify this record but require their own check
Checked-At: 2026-10-04
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-194.md

---

Claim: CL:buster_ripper.main/dump-dir-assignment-guard
Statement: `main.py` assigns `config.DUMP_DIR` only inside the branch that tests a truthy `--dump-dir` value
Subject: repo:buster_ripper.main
Verdict: HOLDS
Actual: `buster_ripper/main.py` contains exactly one assignment to `config.DUMP_DIR` — line 152, `config.DUMP_DIR = Path(dump_dir).expanduser().resolve()` — inside the `if dump_dir:` block opened at line 151. `dump_dir` is the click-bound value of the `--dump-dir` option (declared lines 78-79 with `default=None`, received as the parameter at line 132), so the guard is precisely a truthiness test on that option's value. The only other `config.DUMP_DIR` occurrences are not assignments (line 153 mkdir on the already-assigned value, line 175 a read for the startup log). No click callback and nothing in `_apply_cli_config` (lines 36-58) touches `DUMP_DIR`, and there is no `setattr`/`globals()` write reaching it indirectly.
Source: buster_ripper/main.py — lines 151-153 (the `if dump_dir:` guard and its single assignment), with corroborating non-assignment references at lines 175 and 36-58 and the option/parameter binding at lines 78-79 and 132
Version: unavailable (no `__version__` or revision marker in `buster_ripper/__init__.py`; grounded in working-tree file bytes)
Fingerprint: sha256:fd36dc12055f67693d9b852fb15a66aa4a3a7864103ac3de0ee67d9520048d47
Invalidates-On: any edit to main.py adding an assignment to `config.DUMP_DIR` outside the `if dump_dir:` block (e.g. into `_apply_cli_config`, a new click callback, or module level) — "only" is what such an edit breaks
Invalidates-On: any change to how `--dump-dir` is bound so `if dump_dir:` is no longer a plain truthiness test on the option value (`expose_value=False` with a callback doing the assignment, a truthy non-`None` default, or a rewrite as `if dump_dir is not None`)
Invalidates-On: main.py's sha256 no longer equals fd36dc12055f67693d9b852fb15a66aa4a3a7864103ac3de0ee67d9520048d47
Checked-At: 2026-10-05T01:56:06Z
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-196.md

---

Claim: CL:buster_ripper.routes.messages#handle_messages/dump-turn-invocation-guard
Statement: `handle_messages` calls `dump_turn` only when `config.DUMP_DIR` is not None
Subject: repo:buster_ripper.routes.messages#handle_messages
Verdict: HOLDS
Actual: In `handle_messages` (routes/messages.py) the only call to `dump_turn` is at line 57, nested inside `if config.DUMP_DIR is not None:` at line 55 (itself inside a `try/except Exception` that logs "dump failed"). When `config.DUMP_DIR` is None the guard is false and `dump_turn` is never invoked from this handler. `config.DUMP_DIR` defaults to `None` (config.py:20), so the guard is the effective on/off switch.
Source: buster_ripper/routes/messages.py — `handle_messages` lines 24-70, guard at 55, call at 57, import at 12; corroborating default at buster_ripper/config.py:20 (`DUMP_DIR: Optional[Path] = None`)
Version: unavailable (working-tree source under buster_ripper/; no package version applies)
Fingerprint: sha256:721a2afe4d1ca35da9461c95f5c6dfe48dcf04ec4c4c0a606763cc29b0ca96da
Invalidates-On: routes/messages.py changing so `dump_turn` is called outside the `DUMP_DIR is not None` guard, or a second `dump_turn` call site being added to `handle_messages`
Invalidates-On: `DUMP_DIR` being replaced by a different dump-enable mechanism (a boolean flag, or a sentinel non-`None` default), changing what line 55 tests
Invalidates-On: `dump_turn` becoming reachable indirectly from `handle_messages` (via normalize, image handling, or the streaming/non-streaming helpers), which would make the guard insufficient as the sole gate
Checked-At: 2026-10-05T01:55:57Z
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-197.md

---

Claim: CL:tests/serve-script-invocation
Statement: No module under `tests/` invokes `serve_buster_ripper.sh`
Subject: repo:tests
Verdict: HOLDS
Actual: No file under `tests/` contains the string `serve_buster_ripper`, or any `.sh` reference at all. The only subprocess spawn in the whole test surface is `tests/qa/harness_process.py` (`Proxy.__init__`, `subprocess.Popen(cmd, ...)` at line 129), and `cmd` there is built from exactly two constants: `PROXY_BIN = <repo>/.venv/bin/buster-ripper` (line 30) or `[sys.executable, COUNTED_ENTRY]` with `COUNTED_ENTRY = <repo>/tests/qa/counted_proxy_entry.py` (line 31) — neither a shell script. `counted_proxy_entry.py` runs the proxy in-process by importing `buster_ripper.main` and calling `cli.main()` under a patched `sys.argv`; it spawns nothing. No `exec(`, `eval(`, `os.environ`/`getenv`, `shutil.which` or `runpy` appears anywhere under `tests/`, so no indirect route exists either. Holds under both readings of "module".
Source: tests/ — all 22 Python modules under it; decisive bytes at tests/qa/harness_process.py lines 29-31 and 111-129 and tests/qa/counted_proxy_entry.py; bounded census via recursive greps for `serve_buster_ripper`, `\.sh\b`, and the spawn/exec/env patterns
Version: unavailable (working-tree files under tests/; no package or version involved)
Fingerprint: sha256:c3e83bebc3d65b8fd94e0a2684e36c22844c1c435889b5cf2f87691cd58ac669 (aggregate over the consulted module set — per-file sha256 of every tests/**/*.py excluding `__pycache__`, sorted by path, re-hashed; regenerate from tests/); key files sha256:fd771cde67200a372ac0b60d35ba2c22f3f1089b19ae1a21068f17fd8369c2bc (qa/harness.py), sha256:624c8aaf02fb2e5f917300a6c4a4782b0e04216c29e13d2af7ec4c1d62c6c806 (qa/counted_proxy_entry.py)
Invalidates-On: any tests/**/*.py file whose content changes (the aggregate fingerprint above no longer recomputes), or a new Python module added under tests/
Invalidates-On: tests/qa/harness_process.py's `PROXY_BIN` / `COUNTED_ENTRY` / `cmd` construction altered to name a shell script or a launcher resolved from PATH or an environment variable
Invalidates-On: introduction of a subprocess/exec reference to serve_buster_ripper.sh anywhere under tests/
Checked-At: 2026-10-05T02:23:43Z
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-199.md

---

Claim: CL:tests.test_serve_script_argv/argv-assertion-completeness
Statement: `tests/test_serve_script_argv.py` asserts on the script's complete assembled argv word list rather than on individual flag presence
Subject: repo:tests.test_serve_script_argv
Verdict: HOLDS
Actual: Each of the four environment states the file runs ends in a whole-list `==` comparison against a fully enumerated Python list — no membership-only, prefix, slice, or subset assertion carries any of the verdicts: line 111 `serve_argv() == _TRIO + ["--dump-dir", default] + _THINKING` (with `DUMP_DIR` unset), line 117 `argv == _TRIO + _THINKING` (with `DUMP_DIR=""`), line 122 `argv == _TRIO + ["--dump-dir", "/tmp/served-diffs"] + _THINKING`, and line 127 `argv == _TRIO + ["--dump-format", "change-focused"] + _THINKING` (with `DUMP_DIR=""` and `DUMP_FORMAT="change-focused"`). The left side is genuinely the complete word list: the PATH shim records every positional word with `printf '%s\0' "$@"` and `_read_argv` decodes all NUL-separated fields (popping only the trailing empty field from the final terminator), so the compared value is the wrapper's entire assembled argv rather than a sampled prefix. `_TRIO` (line 45) enumerates the six always-present words and `_THINKING` (line 46) the trailing `--preserved-thinking`, so each right-hand side spells out every word the wrapper contributes, in order. Line 116's `"--dump-dir" not in argv` is an extra negative inside the `DUMP_DIR=""` case, redundant to the line-117 equality rather than a substitute for it; `TestDocumentationAgreement` (line 130) asserts against the script's text, not an environment state, so it is outside the claim's scope. Qualifier carried from the check: the runs override only `DUMP_DIR` and `DUMP_FORMAT`, so complete argv coverage of every wrapper setting does not follow from this file — the claim is about each state it exercises.
Source: tests/test_serve_script_argv.py — whole file read; assertion sites lines 45-46, 111, 117, 122, 127; argv-capture mechanism lines 37-40 and 80-102. Per the test-suite-is-the-surface rule, the subject of this claim is the test module itself, so its source bytes ground the verdict.
Version: unavailable (working-tree test module; no package version and no VCS metadata consulted — content identity is carried by the fingerprint below)
Fingerprint: sha256:461cc81d83c288b0b12488ebcbe3a12ebb68eaffabde283da2bc2c001860be74
Invalidates-On: Any `TestDumpDirDisableForm` method's whole-list `==` replaced by a partial assertion (membership, `not in`, index or slice, subset check), so that state no longer asserts the complete argv
Invalidates-On: `_read_argv` or the PATH shim changed to sample, truncate, reorder, or filter the NUL-separated words, making the compared value anything other than the full assembled argv
Invalidates-On: `_TRIO` or `_THINKING` becoming conditional, variadic, or dropped from an equality's right-hand side, so a listed equality no longer enumerates every word
Invalidates-On: A new environment-state test that asserts only part of the argv (the "each state" quantifier then fails), or that sets `PRESERVED_THINKING` to a flag-suppressing value without a complete-list equality
Invalidates-On: Any edit to tests/test_serve_script_argv.py changing the consulted bytes, i.e. anything that changes the fingerprint above
Checked-At: 2026-10-04
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-201.md

---

Claim: CL:bash/quoted-array-expansion-word-identity
Statement: bash's `"${array[@]}"` expansion in double quotes contributes exactly one argument word per array element, whatever characters an element contains
Subject: lang:bash
Verdict: HOLDS
Actual: This is the documented defining property of `[@]` under double quotes. The Arrays section states it directly: "If the word is double-quoted, `${name[*]}` expands to a single word with the value of each array member separated by the first character of the IFS special variable, and `${name[@]}` expands each element of name to a separate word." The EXPANSION section makes the cardinality explicit — word splitting and pathname expansion are the only steps that can increase word count, and "The only exceptions to this are the expansions of `"$@"` and `"${name[@]}"`" — so the one-word-per-element count comes from the expansion itself, not from splitting the elements afterwards. The "whatever characters" half follows from three further sentences: the elements are produced *inside* double quotes and only expansions not within double quotes are scanned for word splitting (spaces/tabs/newlines in an element are never delimiters); pathname expansion treats a word as a pattern only if `*`, `?`, or `[` appears "and is not quoted" (glob characters stay literal); and quote removal strips only unquoted occurrences of `\`, `'` and `"` that did not result from one of the above expansions (a quote or backslash *stored in* an element is not eaten). Each element becomes a separately quoted word, analogous to `"$@"` being "equivalent to `"$1"` `"$2"` …". Boundary cases the manual settles consistently with one word per element: an empty array contributes zero words ("When there are no array members, `${name[@]}` expands to nothing"), a set-but-null element still contributes one empty word (expansion within double quotes "produces a null field, which is retained and passed to a command as an empty string"), sparse indexed arrays expand assigned members only, and an expansion embedded in a larger word joins only the first and last elements to the surrounding text without changing the element-derived word count.
Source: https://man7.org/linux/man-pages/man1/bash.1.html — sections PARAMETERS → Arrays ("`${name[@]}` expands each element of name to a separate word. When there are no array members, `${name[@]}` expands to nothing"), PARAMETERS → Special Parameters (`@`: "`"$@"` is equivalent to `"$1"` `"$2"` …"), EXPANSION ("Quote removal is always performed last…"; "The only exceptions to this are the expansions of `"$@"` and `"${name[@]}"`"), EXPANSION → Word Splitting, and EXPANSION → Pathname Expansion; quoted text verified present in the fetched document. Same text is GNU Bash Reference Manual §3.4/§3.5 (https://www.gnu.org/software/bash/manual/html_node/Shell-Parameter-Expansion.html — fetch timed out in the check; the mirrored man page carries the identical wording)
Version: GNU Bash 5.3 man page (page header "GNU Bash 5.3 / 2025 April 7"); man7 copy obtained from upstream git://git.savannah.gnu.org/bash.git on 2026-08-03 (most recent upstream commit 2026-06-10) — not the locally installed shell, which was not executed for this check
Fingerprint: sha256:e35fa2966c44a4e9ee519760160f1aba14c670451a463177968ed2c4710bf846 (exact fetched bytes of https://man7.org/linux/man-pages/man1/bash.1.html); sha256:b44556511b41581c65d6dc2505ca306472c6b7f4a75c1c056c1f155ffe1010cb (tag-stripped, whitespace-normalized text of those same bytes, from which the quotations were read)
Invalidates-On: A bash release changing `[@]` semantics (e.g. a compat change to how empty members or no-members expansions are counted) — recheck PARAMETERS → Arrays in the 5.4+ manual.
Invalidates-On: A manual revision rewording "expands each element of name to a separate word" or the quote-removal sentence (fingerprint above changes).
Invalidates-On: Reuse where the construct is not `"${array[@]}"`: `${array[*]}` / `$*` (one joined word), unquoted `${array[@]}` (elements then undergo word splitting and pathname expansion), or `$array` / `${array[0]}` (single element only).
Invalidates-On: Reuse against a non-bash shell (dash/sh/busybox ash) or POSIX-mode bash where array expansions are not available or differ, or a context the manual exempts from expansion (e.g. inside `[[ ]]`, where the words do not undergo word splitting and pathname expansion and quoting behaves differently).
Checked-At: 2026-10-05T04:46:10Z
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-202.md

---

Claim: CL:bash/array-alternate-value-word-contribution
Statement: bash's `${array[@]+word}` alternate-value expansion contributes zero words when `array` is unset or has no elements
Subject: lang:bash
Verdict: HOLDS
Actual: For the colon-less alternate-value form the shell tests only whether the parameter exists: if it does not, nothing is substituted, so no word is emitted (quoted or unquoted). Bash defines an array variable as set only when at least one subscript has been assigned a value, so an array with no elements is not set for the purposes of that test, and `${name[@]}` with no members "expands to nothing". Both stated cases — an unset array and an array with no elements — therefore contribute zero words; `word` appears only when the array has at least one assigned element, and the value of the array itself is never used by this operator. The derivation is two sentences of the manual (the existence test from Shell Parameter Expansion plus the Arrays node's definition of array set-ness); no probe was run.
Source: https://raw.githubusercontent.com/mirror/bash/bash-5.3/doc/bashref.texi (GNU Bash Reference Manual source, tag `bash-5.3`; official upstream tree mirrored at bminor/mirror "bash") — three passages in that one file: @subsection Shell Parameter Expansion (byte offset 82361 ff.), "When not performing substring expansion, using the forms described below (e.g., `:-'), Bash tests for a parameter that is unset or null. Omitting the colon results in a test only for a parameter that is unset."; the same node's `@item $@{@var{parameter}:+@var{word}@}` entry, "If @var{parameter} is null or unset, nothing is substituted, otherwise the expansion of @var{word} is substituted. The value of @var{parameter} is not used."; and @section Arrays (offsets 313531 and 314676), "When there are no array members, `$@{name[@]@}` expands to nothing." / "An array variable is considered set if a subscript has been assigned a value. The null string is a valid value." Corroborating, not load-bearing: the same three sentences are present at tag `bash-5.2` (byte offsets 285377 / 286501 and the same `:+` item).
Version: bash-5.3 (`doc/bashref.texi`, 415,804 bytes, fetched 2026-10-05); cross-checked against the bash-5.2 copy of the same file (374,486 bytes)
Fingerprint: sha256:f3d37d57a1061e24d266051de9bd47ffa43dc86584afea11576c535ad2be32d5 (bash-5.3 doc/bashref.texi, the exact fetched bytes consulted); sha256:32ced60bb645d47be7bfa4e5767edb2ea1b0c840d8968c17f3eb9df960b44863 (bash-5.2 doc/bashref.texi, corroboration)
Invalidates-On: a bash release whose `doc/bashref.texi` no longer contains "An array variable is considered set if a subscript has been assigned a value" in the Arrays node, or drops "Omitting the colon results in a test only for a parameter that is unset" from Shell Parameter Expansion
Invalidates-On: a documented change making the colon-less `+` operator test the variable's existence rather than the array's assigned-subscript set-ness (i.e. `arr=()` starting to substitute `word`)
Invalidates-On: applying the claim to bash < 5.2-era semantics on a specific installed shell (e.g. macOS `/bin/bash` 3.2), where the nounset interaction differs — see `CL:bash/array-alternate-value-nounset-exemption`; re-check per version
Invalidates-On: reuse against a non-bash shell (zsh, ksh, dash) — this verdict is bash-specific
Invalidates-On: any change in what the claim means by "no elements": an array whose last element was `unset`-ed still exists as a variable ("Unsetting the last element of an array variable does not unset the array"), which the manual does not tie to the `+` test explicitly
Checked-At: 2026-10-05T05:11:24Z
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-203.md

---

Claim: CL:bash/array-alternate-value-nounset-exemption
Statement: Under `set -u`, bash's `${array[@]+word}` expansion on an unset or empty `array` raises no nounset error
Subject: lang:bash
Verdict: HOLDS
Actual: `set -u` explicitly exempts array variables subscripted with `@` or `*` from the unbound-variable check, so referencing `${array[@]}` in any expansion form — including the alternate-value `+` form — is not an error whether the array is unset or declared with no elements. Independently, the alternate-value operator never substitutes the parameter's value ("The value of `parameter` is not used"; when the parameter is unset or null "nothing is substituted"), so there is no value lookup for nounset to complain about: the shell performs an existence test and emits nothing. An empty array is "not considered set" by the same manual (set requires an assigned subscript), so it takes the identical no-error path as an unset array. Established from a different manual node and sentence than `CL:bash/array-alternate-value-word-contribution`.
Source: https://raw.githubusercontent.com/mirror/bash/bash-5.3/doc/bashref.texi — @subsection "The Set Builtin", `@item -u` (byte offset 221154): "Treat unset variables and parameters other than the special parameters @samp{@@} or @samp{*}, or array variables subscripted with @samp{@@} or @samp{*}, as an error when performing parameter expansion." Supporting from the same file (not load-bearing): the Shell Parameter Expansion `:+` item, "nothing is substituted … The value of @var{parameter} is not used" (offset 86495 ff.). Version boundary corroborated from https://raw.githubusercontent.com/mirror/bash/bash-4.4/CHANGES, "New Features in Bash": "Using ${a[@]} or ${a[*]} with an array without any assigned elements when the nounset option is enabled no longer throws an unbound variable error." The `-u` sentence's array clause is present in the bash-5.2 and bash-5.3 manuals and absent from the bash-4.3, 4.4, 5.0 and 5.1 copies of `doc/bashref.texi` (checked at each tag; the prose caught up with the 4.4 behaviour change).
Version: bash-5.3 (`doc/bashref.texi`, 415,804 bytes); clause also present at tag `bash-5.2` (374,486 bytes) and absent at tags bash-4.3 / bash-4.4 / bash-5.0 / bash-5.1; behavioural change recorded for bash-4.4 (`CHANGES`, 357,491 bytes)
Fingerprint: sha256:f3d37d57a1061e24d266051de9bd47ffa43dc86584afea11576c535ad2be32d5 (bash-5.3 doc/bashref.texi, exact fetched bytes consulted); sha256:32ced60bb645d47be7bfa4e5767edb2ea1b0c840d8968c17f3eb9df960b44863 (bash-5.2 doc/bashref.texi, same clause); sha256:992ae56e662f50e206ae8ed492cf9b1d8c54d7bf2f307e60a4153cd1f7191a7b (bash-4.4 CHANGES, version-boundary entry)
Invalidates-On: a bash release whose "The Set Builtin" `-u` item drops the clause "or array variables subscripted with @ or *"
Invalidates-On: reuse against a shell predating bash-4.4 (e.g. macOS `/bin/bash` 3.2), where nounset did report unbound-variable errors for `@`-subscripted arrays; the manuals for 4.3 and earlier do not document an array exemption, so the verdict must be re-checked per installed version
Invalidates-On: any documented nounset change that applies the check to the test forms (`+`, `-`, `?`, `=`) rather than only to value substitution
Invalidates-On: `set -u` semantics under `set -o posix`, or `nounset` interaction with associative arrays referenced without a subscript, if the target code shape differs from an `@`-subscripted array
Invalidates-On: the claim being extended to a subscripted scalar `${array[key]+word}` on an unset plain variable — that is a different nounset path and is not covered here
Checked-At: 2026-10-05T05:11:24Z
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-204.md

---

Claim: CL:aiohttp/auto-decompress-session-default
Statement: aiohttp's `ClientSession.__init__` `auto_decompress` parameter defaults to `True`
Subject: pkg:aiohttp@3.13.3
Verdict: HOLDS
Actual: `ClientSession.__init__` declares `auto_decompress: bool = True` and stores it verbatim as `self._auto_decompress`, so a session constructed without the argument has decompression enabled. The read-only property `ClientSession.auto_decompress` returns that same stored value. No normalization, sentinel, or environment-derived default intervenes between the signature and the attribute.
Source: `.venv/lib/python3.13/site-packages/aiohttp/client.py:299` (signature default), `:399` (`self._auto_decompress = auto_decompress`), `:1436-1438` (`auto_decompress` property). Installed package ships no docs and no `.pyi` stubs, so the installed source is the settling authority.
Version: aiohttp 3.13.3 (`__version__ = "3.13.3"` in `aiohttp/__init__.py`)
Fingerprint: sha256:7ce41fc1c2142f5346015455e030e3ebec22a41cde0fc299a66ce13be2cb2a9e (client.py); sha256:416b2c15a0fe0da1457303f7ea52d44334669526792b18af24153eca0e42d708 (aiohttp/__init__.py, version provenance)
Invalidates-On: any aiohttp release whose `client.py` bytes differ from the fingerprint above
Invalidates-On: `ClientSession.__init__` gaining a non-bool or sentinel default (e.g. `Union[bool, None] = None`), or resolving the default from an env var, config object, or class attribute instead of the literal `True`
Invalidates-On: an intervening coercion between the signature and `self._auto_decompress = auto_decompress`
Checked-At: 2026-10-06
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-384.md

---

Claim: CL:aiohttp/auto-decompress-request-inheritance
Statement: A request that passes no `auto_decompress` to a `ClientSession` request method uses the session's own value
Subject: pkg:aiohttp@3.13.3
Verdict: HOLDS
Actual: `ClientSession._request` declares `auto_decompress: Optional[bool] = None` and, before dispatch, substitutes the session's stored value whenever the argument was omitted (`if auto_decompress is None: auto_decompress = self._auto_decompress`); the resolved value is then passed to `conn.protocol.set_response_params(auto_decompress=...)`. Every public verb (`get`/`post`/`put`/`patch`/`delete`) reaches this code through `_request` forwarding only `**kwargs`, so none can inject its own default. `None` and "omitted" are the same case; an explicit `auto_decompress=False` is honored per-request and does not fall back. The protocol's own `set_response_params(auto_decompress: bool = True)` default never governs a session request because the client always passes an explicit resolved value.
Source: `.venv/lib/python3.13/site-packages/aiohttp/client.py:518` (per-request default), `:601-602` (session fallback), `:743-747` (passed to `set_response_params`), `:478,1287,1297,1317,1325,1333,1339` (verb funnel), `:195` (`_RequestOptions` declares `auto_decompress: Union[bool, None]`); `.venv/lib/python3.13/site-packages/aiohttp/client_proto.py:227`. Installed package ships no docs and no `.pyi` stubs.
Version: aiohttp 3.13.3 (`__version__ = "3.13.3"` in `aiohttp/__init__.py`)
Fingerprint: sha256:7ce41fc1c2142f5346015455e030e3ebec22a41cde0fc299a66ce13be2cb2a9e (client.py); sha256:416b2c15a0fe0da1457303f7ea52d44334669526792b18af24153eca0e42d708 (aiohttp/__init__.py, version provenance)
Invalidates-On: any aiohttp release whose `client.py` bytes differ from the fingerprint above
Invalidates-On: the per-request default changing from `None` to a concrete `True`/`False` or to a distinct sentinel type, removing or redirecting the session fallback
Invalidates-On: removal or relocation of the `if auto_decompress is None` block, or the resolved value no longer being passed to `set_response_params`
Invalidates-On: a public verb ceasing to forward `**kwargs` to `_request` and supplying its own `auto_decompress` argument
Checked-At: 2026-10-06
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-385.md

---

Claim: CL:aiohttp/auto-decompress-read-decompressed
Statement: When `auto_decompress` is enabled, aiohttp's `ClientResponse.read()` returns the decompressed payload rather than the encoded wire bytes
Subject: pkg:aiohttp@3.13.3
Verdict: HOLDS
Actual: With `auto_decompress` on and a recognized `Content-Encoding` (`gzip`, `deflate`, `br`, `zstd`), the payload parser wraps the `StreamReader` in a `DeflateBuffer`: raw wire chunks are decompressed and only the decompressed bytes are pushed into the inner `StreamReader` that `ClientResponse.content` exposes. `read()` awaits `self.content.read()` on that inner stream and caches the result in `self._body`, so it returns decompressed bytes. Wire-size accounting is kept separately as `total_compressed_bytes`, the source's own acknowledgement that the buffered content is not the compressed input. With `auto_decompress` disabled, or the encoding absent/unrecognized, `read()` returns the raw wire bytes instead. Both parser implementations (pure-Python and the Cython build actually used by this install) show the same wiring.
Source: `.venv/lib/python3.13/site-packages/aiohttp/http_parser.py:798-804,1007-1048,458`; `.venv/lib/python3.13/site-packages/aiohttp/client_reqrep.py:539,570,689-707`; `.venv/lib/python3.13/site-packages/aiohttp/_http_parser.pyx:460-473`. No installed docs ship with the package.
Version: aiohttp 3.13.3 (`aiohttp-3.13.3.dist-info/METADATA`)
Fingerprint: sha256:cfa763643394b3b85d3f3a654c4b00572cf4a1ffab400c13ef1cfc3a95e19ee6 (http_parser.py); sha256:f9823c608638b8a77fec1c2bd28dc5c043f08c775ec59e7e262dd74b4ebb7384 (_http_parser.pyx); sha256:7844440eb67433c6454edd708c776e47e3fcfec9b8d0aeb980dcfe88c1986ac9 (client_reqrep.py)
Invalidates-On: any aiohttp release whose `http_parser.py`, `_http_parser.pyx`, or `client_reqrep.py` bytes differ from the fingerprints above
Invalidates-On: a release moving decompression out of the payload parser (e.g. to a `content`-property hook or response middleware), or handing the `DeflateBuffer` wrapper rather than the inner `StreamReader` to `ClientResponse.content`
Invalidates-On: `ClientResponse.read()` reimplemented to bypass `self.content` or to return wire bytes
Invalidates-On: a change to the recognized-encoding set at `http_parser.py:557` / `_http_parser.pyx:438` that stops matching the encoding used on the wire
Invalidates-On: running with `AIOHTTP_NO_EXTENSIONS=1` in a build whose pure-Python parser diverges from the Cython one
Checked-At: 2026-10-06
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-386.md

---

Claim: CL:aiohttp/auto-decompress-content-encoding-retention
Statement: When `auto_decompress` is enabled, aiohttp's `ClientResponse.headers` still contains the upstream's `Content-Encoding` after the body is decompressed
Subject: pkg:aiohttp@3.13.3
Verdict: HOLDS
Actual: Decompression is decided *from* the header but never removes it. In both parser implementations every parsed header — `Content-Encoding` included — is appended to the headers collection that becomes the response message, and the `Content-Encoding` value is additionally copied out to a separate field used only to select the decompressor. `ClientResponse._headers` is assigned that message headers object verbatim and exposed read-only via `headers`, so `resp.headers["Content-Encoding"]` still yields the upstream value even though `resp.content`/`read()` serve decompressed bytes; `raw_headers` likewise keeps the original wire header bytes. Nothing on the client-response path deletes it — the only `del self.headers[...]` sites in `client_reqrep.py` are inside `class ClientRequest` for outgoing `COOKIE`, `CONTENT_LENGTH`, and `TRANSFER_ENCODING`; `class ClientResponse` performs no header removal.
Source: `.venv/lib/python3.13/site-packages/aiohttp/_http_parser.pyx:385-388,419,434-439`; `.venv/lib/python3.13/site-packages/aiohttp/http_parser.py:556-558,571`; `.venv/lib/python3.13/site-packages/aiohttp/client_reqrep.py:428-429,565,793,1109,1218,1222`. No installed docs ship with the package; read-only-proxy semantics corroborated from multidict 6.7.1.
Version: aiohttp 3.13.3 (`aiohttp-3.13.3.dist-info/METADATA`); multidict 6.7.1 (corroborating)
Fingerprint: sha256:f9823c608638b8a77fec1c2bd28dc5c043f08c775ec59e7e262dd74b4ebb7384 (_http_parser.pyx); sha256:cfa763643394b3b85d3f3a654c4b00572cf4a1ffab400c13ef1cfc3a95e19ee6 (http_parser.py); sha256:7844440eb67433c6454edd708c776e47e3fcfec9b8d0aeb980dcfe88c1986ac9 (client_reqrep.py); sha256:546439f0fed539dea545fdd65408a76fad9a0f5e833dd01646debcaa68893171 (multidict/_multidict_py.py)
Invalidates-On: any aiohttp release whose `_http_parser.pyx`, `http_parser.py`, or `client_reqrep.py` bytes differ from the fingerprints above
Invalidates-On: aiohttp adding header mutation on the response path (deleting or rewriting `Content-Encoding` in `_process_header`, `_on_headers_complete`, `HeadersParser.parse_headers`, or `ClientResponse.start`)
Invalidates-On: a release moving decompression into a `ClientResponse.content`/`headers` wrapper that reconciles the header with the decoded body
Invalidates-On: `ClientResponse._headers` becoming a mutable `CIMultiDict` that user code or middleware conventionally clears
Invalidates-On: a change to recognized-encoding detection at `_http_parser.pyx:438` / `http_parser.py:557` that leaves the header present but no longer matches it
Checked-At: 2026-10-06
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-387.md

---

Claim: CL:aiohttp/repeated-header-items-iteration
Statement: `aiohttp.ClientResponse.headers.items()` yields one pair per occurrence when the upstream repeats a header name
Subject: pkg:aiohttp@3.13.3
Verdict: HOLDS
Actual: `ClientResponse.headers` is a `CIMultiDictProxy[str]` built by the HTTP parser with one `headers.add(name, value)` call per response header line, so each occurrence of a repeated name is stored as a separate entry. `items()` returns a view whose iterator walks the raw entry list and yields `(key, value)` per stored entry — not per distinct key — including every occurrence of a duplicated header name, in insertion order. `len()` counts entries rather than distinct keys, consistent with per-occurrence iteration.
Source: `.venv/lib/python3.13/site-packages/multidict/_multidict_py.py:722-724,98-102,595-596,775-778,1196-1198,715-716`; chain in aiohttp at `.venv/lib/python3.13/site-packages/aiohttp/client_reqrep.py:428-429,565` and `.venv/lib/python3.13/site-packages/aiohttp/http_parser.py:149,235,238`; docs at `.venv/lib/python3.13/site-packages/multidict-6.7.1.dist-info/METADATA:76-78,108`. CPython loads the compiled `_multidict` extension (not readable as source); the verdict rests on the pure-Python reference implementation plus the package's own statement that the C extension exists only for speed.
Version: multidict 6.7.1, aiohttp 3.13.3 (dist-info directory names in site-packages)
Fingerprint: sha256:546439f0fed539dea545fdd65408a76fad9a0f5e833dd01646debcaa68893171 (multidict/_multidict_py.py); sha256:cfa763643394b3b85d3f3a654c4b00572cf4a1ffab400c13ef1cfc3a95e19ee6 (aiohttp/http_parser.py); sha256:7844440eb67433c6454edd708c776e47e3fcfec9b8d0aeb980dcfe88c1986ac9 (aiohttp/client_reqrep.py); sha256:4913fdaae936a309c29b89b98c2dbebd237203fb27ab9e68ae5e4ec17ba3024f (multidict-6.7.1.dist-info/METADATA)
Invalidates-On: any multidict release whose `_multidict_py.py` bytes differ from the fingerprint above (view/entry iteration: `_ItemsView._iter`, `iter_entries`, `add`)
Invalidates-On: any aiohttp release whose `http_parser.py` or `client_reqrep.py` bytes differ, e.g. switching from per-line `add()` to a deduplicating/last-wins merge, or changing the `headers` property type away from `CIMultiDictProxy`
Invalidates-On: an upgrade of multidict or aiohttp where the pure-Python reference implementation's duplicate-occurrence semantics differ
Checked-At: 2026-10-06T19:01:14Z
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-388.md

---

Claim: CL:aiohttp/default-accept-encoding-header
Statement: When neither the session's nor the request's headers name `Accept-Encoding`, aiohttp's client sends a default one naming the encodings it can decode
Subject: pkg:aiohttp@3.13.3
Verdict: HOLDS
Actual: `ClientRequest.DEFAULT_HEADERS` maps `hdrs.ACCEPT_ENCODING` to `_gen_default_accept_encoding()` — at minimum `"gzip, deflate"`, extended with `"br"`/`"zstd"` when brotli/zstd are available. `update_auto_headers()`, called unconditionally from `ClientRequest.__init__`, adds each default only if that header name is absent from `self.headers`. Because `ClientSession._request()` merges session `headers` defaults with request-level headers via `_prepare_headers()` before constructing the `ClientRequest`, an `Accept-Encoding` named at either level appears in `used_headers` and suppresses the default; when neither names it, the default is added to the outgoing request headers and sent. Header-name matching is case-insensitive (`istr`/`CIMultiDict`). A header can also be suppressed via `skip_auto_headers`, which the claim does not address and which does not contradict it.
Source: `.venv/lib/python3.13/site-packages/aiohttp/client_reqrep.py:105-114,803-806,895-896,1081-1095`; `.venv/lib/python3.13/site-packages/aiohttp/client.py:549,1207-1221`. Installed package ships no docs directory; verdict grounded in installed source.
Version: aiohttp 3.13.3 (`aiohttp/__init__.py` `__version__ = "3.13.3"`; `aiohttp-3.13.3.dist-info`)
Fingerprint: sha256:7844440eb67433c6454edd708c776e47e3fcfec9b8d0aeb980dcfe88c1986ac9 (client_reqrep.py); sha256:7ce41fc1c2142f5346015455e030e3ebec22a41cde0fc299a66ce13be2cb2a9e (client.py)
Invalidates-On: any aiohttp release whose `client_reqrep.py` or `client.py` bytes differ from the fingerprints above
Invalidates-On: `DEFAULT_HEADERS` dropping `hdrs.ACCEPT_ENCODING`, or `_gen_default_accept_encoding` returning no header/empty value
Invalidates-On: `update_auto_headers` or its `used_headers` membership check changing so defaults are added regardless of existing headers, or no longer being called from `ClientRequest.__init__`
Invalidates-On: `ClientSession._request`/`_prepare_headers` ceasing to merge session defaults into the header set passed to `ClientRequest`
Checked-At: 2026-10-06
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-389.md

---

Composition: CO:client-response-to-relay-consumer/payload-and-header-pairing
Members: CL:aiohttp/auto-decompress-session-default, CL:aiohttp/auto-decompress-request-inheritance, CL:aiohttp/auto-decompress-read-decompressed, CL:aiohttp/auto-decompress-content-encoding-retention, CL:aiohttp/repeated-header-items-iteration, CL:aiohttp/default-accept-encoding-header
Boundary: the upstream response's payload bytes and its header set as one `aiohttp.ClientResponse` hands both to the caller at each relay boundary — producer `pkg:aiohttp@3.13.3` (`ClientSession`/`ClientResponse` in `client.py`, `client_reqrep.py`, `http_parser.py`, `_http_parser.pyx`, with `multidict 6.7.1` (`_multidict_py.py`) behind the header collection); consumer `repo:buster_ripper.utils#response_headers` (`buster_ripper/utils.py:29-56`) and the three relay sites that call it (`routes/passthrough.py:41`, `routes/messages.py:291`, `routes/chat.py:172`). Order traced: request posture → header-set selection → payload decoding → single object → consumer split
Dimensions: session default versus per-request `auto_decompress` (384/385); encoded-on-the-wire versus decoded-as-handed-over body bytes (386); `Content-Encoding` relayed versus withheld alongside a body the client already decoded (387); single versus repeated occurrences of one header name, and one pair per occurrence at `.items()` (388); caller-supplied versus client-default `Accept-Encoding`, which decides whether an encoded answer is ever asked for (389)
Verdict: COUNTEREXAMPLE
Counterexample: CE-1 — a body the client decoded is relayed downstream under the upstream's encoding label. Concrete input: a client `GET`s any passthrough endpoint (for example `/v1/models`, `passthrough.py:11-20`) and the upstream answers `200`, `Content-Encoding: gzip`, `Content-Type: application/json`, gzip-compressed JSON body. That answer is asked for either because the client sent `Accept-Encoding: gzip, deflate` and `forward_headers` passed it through (`utils.py:20-26`; `passthrough.py:17`, `messages.py:61`, `chat.py:104`), or because the client named none and aiohttp added its own `gzip, deflate` default (389). Path: 384 → 385 (`server.py:65-68` names no `auto_decompress`, so `client.py:601-602` resolves it to `True`) → 386 (`http_parser.py:798-801` wraps the stream, so `resp.read()` at `passthrough.py:38` returns *decompressed* bytes) → 387 (`client_reqrep.py:565` leaves `resp.headers["Content-Encoding"] == "gzip"`) → consumer (`utils.py:52-56` keeps the name, since `_HOP_BY_HOP` at `:14-17` does not name `content-encoding`) → `web.Response(body=content, …)` at `passthrough.py:43-47`, whose `_start` writes `Content-Length` from the decoded length (`web_response.py:797-802`) and never revises `Content-Encoding` (`:472-535`). Observable result: the downstream client receives a head that says gzip with a body that is plain JSON, and a length that only matches the plain body; a conforming client attempts to inflate identity bytes — `curl --compressed` reports an unrecognized/corrupt content encoding and `requests`/`httpx`/the Anthropic and OpenAI SDKs raise a decompression error — so the proxied request fails although the upstream succeeded. Same ordering at the other two sites: `messages.py:270` + `:291-296` (which additionally *parses* those decoded bytes at `:272` and `:276`, so that site needs the decoded form for its own logic while relaying the encoded label), and `chat.py:149`/`:165` + `:170-172`. `chat.py:217`/`:221` is the strongest case: there the body is re-serialized JSON (`json.dumps(resp_data).encode()`), so even byte-identity with the upstream is gone while the `Content-Encoding` label is still forwarded. The two streaming relays are the coherent contrast, which shows the conflict is a property of this composition rather than of the library: `messages.py:151-161` + `_relay_after_commit` (`messages.py:170-197`) and `chat.py:191-198` prepare a repo-authored head that carries no `Content-Encoding` and then write the decoded chunks from `resp.content.iter_any()` — decompressed bytes, unlabelled. `Reach:` `buster_ripper/routes/passthrough.py:38-47` (`handle_passthrough`), `buster_ripper/routes/messages.py:270,291-296` (`_non_stream_single_request`) and `buster_ripper/routes/chat.py:170-172` (`_chat_response`, reached from `:149`, `:165`, `:217`, `:221`) — all read the value and put it on the wire; `buster_ripper/utils.py:52-56` is the shared consumer that reads `resp.headers` and decides the label survives. CE-2 (prior record) — no longer composes against these bytes: the earlier fold of `resp.headers.items()` per-occurrence pairs to the last occurrence is gone, because `response_headers` now returns a `CIMultiDict` built from a generator over `upstream_headers.items()` (`utils.py:52-56`), whose construction (`_multidict_py.py:635-641` → `:820-838` → `:845-848`), `StreamResponse.__init__`'s whole clone (`web_response.py:114`) and `write_headers` keep every entry, so `Set-Cookie: a=1` and `Set-Cookie: b=2` from the upstream both reach the client; the only member claim that previously conflicted at the consumer (388) now composes with it, and `content-encoding`'s presence in the relayed set is unaffected by the change, which is why CE-1 stands and CE-2 does not.
Source: library (producer side) — `.venv/lib/python3.13/site-packages/aiohttp/client.py:274,299,399,486,518,549,601-602,743-747,1207-1221,1287,1297,1317,1325,1333,1339`; `aiohttp/client_reqrep.py:105-114,428-429,536-544,565,689-707,803-806,895-896,1081-1095`; `aiohttp/http_parser.py:149,235,238,458,556-558,571,798-801,981-996`; `aiohttp/_http_parser.pyx:385-388,419,434-441,460-473`; `aiohttp/compression_utils.py:16-33`; `aiohttp/client_proto.py:197-204,227,313-329`; `aiohttp/streams.py`, `aiohttp/http_exceptions.py`, `aiohttp/client_exceptions.py` (N1's exception channel); `aiohttp/web_response.py:94-116,434-435,472-535,628-687,788-802,829`; `aiohttp/http_writer.py:210-219,363-377`; `aiohttp/_http_writer.pyx:125-151`; `multidict/_multidict_py.py:98-102,595-596,620-647,715-724,775-778,788-848,1196-1198`; `multidict-6.7.1.dist-info/METADATA:76-78,108`. Repo (consumer side) — `buster_ripper/utils.py:14-17` (`_HOP_BY_HOP`), `:20-26` (`forward_headers`), `:29-56` (`response_headers`, including the docstring posture statement at `:46-50`); `buster_ripper/server.py:64-68` (the only `ClientSession`); `buster_ripper/routes/passthrough.py:17,30-48`; `buster_ripper/routes/messages.py:61,151-161,170-197,265-297`; `buster_ripper/routes/chat.py:104,141-152,155-172,185-199,202-221`. Versions: aiohttp 3.13.3 (`aiohttp/__init__.py` `__version__ = "3.13.3"`; `aiohttp-3.13.3.dist-info`), multidict 6.7.1 (`multidict-6.7.1.dist-info`), CPython 3.13 environment under `.venv` with compiled extensions present (`aiohttp/_http_parser.cpython-313-darwin.so`, `aiohttp/_http_writer.cpython-313-darwin.so`, `multidict/_multidict.cpython-313-darwin.so`) — both parser implementations and both header serializers read, plus the pure-Python reference implementations the member verdicts rest on; `buster_ripper` working tree, digests in `Fingerprint:`
Fingerprint: sha256:d790d12b6513986bb6328589c229ae7c23656538588b26b9a8b2b0837987a673 (handle-bearing — basis over the ordered member handles plus their verdict-recorded primary digests, then the sha256 of every source file consulted by the composition pass. Member digests in ledger order: 384 `7ce41fc1…` (client.py), 385 `7ce41fc1…` (client.py), 386 `cfa76364…` (http_parser.py), 387 `f9823c60…` (_http_parser.pyx), 388 `546439f0…` (_multidict_py.py), 389 `7844440e…` (client_reqrep.py). Consulted files: aiohttp/client.py `7ce41fc1c2142f5346015455e030e3ebec22a41cde0fc299a66ce13be2cb2a9e`, aiohttp/client_reqrep.py `7844440eb67433c6454edd708c776e47e3fcfec9b8d0aeb980dcfe88c1986ac9`, aiohttp/http_parser.py `cfa763643394b3b85d3f3a654c4b00572cf4a1ffab400c13ef1cfc3a95e19ee6`, aiohttp/_http_parser.pyx `f9823c608638b8a77fec1c2bd28dc5c043f08c775ec59e7e262dd74b4ebb7384`, aiohttp/client_proto.py `e7afd6b4b4ad6460453d82b3804818eafdb8264855d72e891269aec5e253d92a`, aiohttp/streams.py `ae5c0bede93a0a430c62297f7bf128916bf6eae1e6b738b794aaa59cb36b5c27`, aiohttp/http_exceptions.py `0632310f82edad0832b6aa11e65388f7302db4d992ca046092c52c311cbbb2cb`, aiohttp/client_exceptions.py `bb229bc48da979984a97b9442c1331dd47aeb0d91fa5e98f5a9185ab4afa25e3`, aiohttp/compression_utils.py `849d8b5e1376396ba41479b96fbf1324518a70088bda4b618bd49fe4f4583be5`, aiohttp/web_response.py `3ca73388d5382e67ed5ea295be844cac06ce7950a5a5237e8b39c7b2259ece65`, aiohttp/http_writer.py `7db46d28f612a916ed01daff82aa63176fb8b08d4448bf1d3c317ec58fe600c6`, aiohttp/_http_writer.pyx `56514404ce87a15bfc6b400026d73a270165b2fdbe70313cfa007de29ddd7e14`, multidict/_multidict_py.py `546439f0fed539dea545fdd65408a76fad9a0f5e833dd01646debcaa68893171`, multidict-6.7.1.dist-info/METADATA `4913fdaae936a309c29b89b98c2dbebd237203fb27ab9e68ae5e4ec17ba3024f`, buster_ripper/utils.py `9abaa6b8ccbc2bdf8aadab44e64f2a07236ecad619cfc4b98fe78e897c8fdf24`, buster_ripper/server.py `a9e49823ea456af9c0076781c4713c956745b21b026dfa768faf947e0b130592`, buster_ripper/routes/passthrough.py `eebf21192eb1a73f52d8cd34a69cff32f12090515d5a6ea61de1e19c87366e4c`, buster_ripper/routes/chat.py `d33ef8b81bccb966529b6548b2da14c3ee938e142bbea4ed73fc90584261d6f3`, buster_ripper/routes/messages.py `c362d5e2e1e33761f3d12e16a9c7a4bbd8bae5b134f3cb82fb9f6e3f0ebbf238`. The hashed input is that 25-line components block verbatim (each line newline-terminated, 3196 bytes); every digest in it was re-verified against the working tree by the composition pass and all matched. This basis replaces `sha256:74ac46583a4976fab03d873c3a937eb1906f14476dd646d5cbd6422155d437ca` (checked 2026-10-06T19:05:05Z), whose `buster_ripper/utils.py` component pinned `76de98c1…`: the member-handle components are identical, the consulted set grew by five files (`http_writer.py`, `_http_writer.pyx`, `compression_utils.py`, `streams.py`, METADATA), and `utils.py` moved)
Invalidates-On: any of the six member `CL:` digests ceasing to match the bytes this composition hashed (`d790d12b…`) — re-read that member's verdict rather than reuse this record; a member's own `Invalidates-On` is not sufficient, because the findings are joint. In particular any aiohttp release changing `client.py:299/:399` or `:518/:601-602`, `http_parser.py:556-558/:798-801/:981-996`, `_http_parser.pyx:385-388/:434-441`, `client_reqrep.py:565/:1081-1095`, `web_response.py:114/:631-647/:788-802`, `http_writer.py:363-377` / `_http_writer.pyx:125-151`, or multidict's `_multidict_py.py` entry/view iteration and `MultiDict.__init__`/`_parse_args`/`_extend_items` copy paths
Invalidates-On: CE-1's chain, any one link of which invalidates the finding — (a) the package starting to pass `auto_decompress` (at `server.py:65-68` or per request), or aiohttp changing the session/per-request precedence of 384/385; (b) decompression moving out of the payload parser (386); (c) the response header set becoming reconciled with the decoded body, by aiohttp deleting `Content-Encoding` or by any middleware doing it (387); (d) `_HOP_BY_HOP` (`utils.py:14-17`) gaining `content-encoding`, or `response_headers`'s exclusion tuple (`utils.py:55`) gaining any content-coding or body-consistency filter, or the function's own docstring statement at `utils.py:46-50` being contradicted by code that does reconcile the pair; (e) any of the three sites (`passthrough.py:43-47`, `messages.py:292-296`, `chat.py:172`) ceasing to pair `resp.read()` with `response_headers(resp.headers)` on one `web.Response`; (f) the egress `Accept-Encoding` posture moving — `forward_headers` (`utils.py:25`) dropping `accept-encoding`, a session-level default header set appearing at `server.py:65-68`, or `skip_auto_headers` being used
Invalidates-On: dimension 4 and the closed CE-2 — any change to `response_headers`' return type (`utils.py:52`), to its generator-of-pairs construction shape (`utils.py:53-56`), to `MultiDict.__init__`/`_parse_args`'s per-pair entry creation, to `StreamResponse.__init__`'s `CIMultiDict(headers)` copy, or to either header serializer's per-entry iteration, reopens the fold; so does any change to 388's per-occurrence semantics. N5's open contract likewise moves with any site that starts passing `content_type=`/`charset=` alongside relayed headers (`web_response.py:653-662`)
Invalidates-On: N1's premise — installing `brotli`/`brotlicffi` or `backports.zstd`, or an aiohttp release changing `HAS_BROTLI`/`HAS_ZSTD` handling or the `HttpProcessingError` → `ClientResponseError` conversion at `client_reqrep.py:536-544`, moves dimension 2's `{br, zstd}` arm from note to a live member of the decoded class and must be re-traced
Invalidates-On: consumer-side drift — any edit to `buster_ripper/utils.py`, `server.py`, `routes/passthrough.py`, `routes/chat.py` or `routes/messages.py` at the locators cited above; the composition pass's scoping rule is member-anchor intersection with the change, and every finding chain here crosses `utils.py`
Invalidates-On: boundary wording — any seam-row edit to S15's boundary value, subject list or dimension list; this record traces the five named dimensions only and claims nothing beyond them
Checked-At: 2026-10-06T20:33:47Z
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/compositions/seam-S15.md

---

Claim: CL:aiohttp/app-client-max-size-keyword
Statement: `aiohttp.web.Application.__init__` accepts a `client_max_size` keyword argument
Subject: pkg:aiohttp@3.13.3
Verdict: HOLDS
Actual: `Application.__init__` declares `client_max_size: int = 1024**2` after a bare `*`, so the argument is accepted and is keyword-only; the value is stored verbatim as `self._client_max_size` and nothing in `__init__` rejects, renames, or deprecates it — unlike `router`, `loop`, and `debug`, each of which emits a `DeprecationWarning`. The attribute is also declared in `Application.ATTRS`.
Source: `.venv/lib/python3.13/site-packages/aiohttp/web_app.py` — `Application.__init__` signature at lines 121-131 (`*` at :123, `client_max_size: int = 1024**2` at :128), `self._client_max_size = client_max_size` at :175, `"_client_max_size"` in `Application.ATTRS` at :116. The installed package ships no docs tree, so the installed source is the establishing locator and upstream `web_advanced.html` was corroboration only.
Version: aiohttp 3.13.3 (`__version__ = "3.13.3"` in `aiohttp/__init__.py`)
Fingerprint: sha256:94653f68030dfa1df0cbe2d34c78ba49e287f3276dd46e75057702b29803e590 (web_app.py)
Invalidates-On: an aiohttp release where `client_max_size` is removed from or renamed in `Application.__init__`'s signature
Invalidates-On: a release that makes the argument positional-acceptable (the `*` sentinel removed) or moves it behind a deprecated alias, changing its keyword-only character
Invalidates-On: any `web_app.py` whose sha256 differs from the fingerprint above
Checked-At: 2026-10-06T22:21:04Z
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-390.md

---

Claim: CL:aiohttp/client-max-size-default
Statement: An aiohttp `web.Application` constructed without `client_max_size` limits a request body to 1024**2 bytes
Subject: pkg:aiohttp@3.13.3
Verdict: HOLDS
Actual: The parameter default `client_max_size: int = 1024**2` is stored as `self._client_max_size == 1048576`, and `Application._make_request` always forwards it (`client_max_size=self._client_max_size`), so every request that app serves inherits 1 MiB and `BaseRequest.read()` raises once the accumulated body reaches it. Two precisions that do not contradict the cap: the comparison is `body_size >= self._client_max_size`, so a body of exactly 1048576 bytes is already rejected (largest accepted body is 1048575 bytes, and the 413 still reports `max_size=1048576`); and enforcement lives in the buffered body APIs — `read()`/`text()`/`json()` via `read()`, `post()` via `max_size = self._client_max_size` — not in direct reads of `request.content`. `0` disables the check entirely and is not the default.
Source: `.venv/lib/python3.13/site-packages/aiohttp/web_app.py` — `client_max_size: int = 1024**2` at :128, stored at :175, forwarded at :510; `.venv/lib/python3.13/site-packages/aiohttp/web_request.py` — enforcement at :675-680 (`read()`) and :722/:746/:774 (`post()`), `Request`'s own default likewise `1024**2` at :161. Installed package ships no docs, so the verdict rests on installed source.
Version: aiohttp 3.13.3 (`__version__ = "3.13.3"` in `aiohttp/__init__.py`)
Fingerprint: sha256:94653f68030dfa1df0cbe2d34c78ba49e287f3276dd46e75057702b29803e590 (web_app.py); sha256:2aaacea7a01e581e5eead2ab1b9e4ba3b65bc2ae3d0f176b2a78ae5b6b76bb4e (web_request.py)
Invalidates-On: an aiohttp release that changes the `Application.__init__` default for `client_max_size` away from `1024**2`
Invalidates-On: a release that changes the `>=` boundary to `>`, or moves enforcement off `BaseRequest.read()`/`post()` onto the parser or transport layer
Invalidates-On: any file whose sha256 differs from the fingerprints above
Checked-At: 2026-10-06T22:21:04Z
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-392.md

---

Claim: CL:aiohttp/client-max-size-falsy-skip
Statement: aiohttp skips the request-body size check whenever the configured `client_max_size` is falsy, so `0` imposes no limit on any body read
Subject: pkg:aiohttp@3.13.3
Verdict: HOLDS
Actual: The size guard in `Request.read()` sits inside a truthiness test — `if self._client_max_size:` — so with `client_max_size=0` the accumulated body length is never compared and `HTTPRequestEntityTooLarge` is never raised; the body is read to completion without bound. Both body-accumulation paths in `Request.post()` are guarded the same way (`if 0 < max_size < size:`). `Application.__init__` stores the value with no validation and `_make_request` passes it straight to each `Request`, so `0` reaches the guard unchanged, and `Application.client_max_size` / `Request.client_max_size` report `0`. Nothing else caps total request-body size: the payload `StreamReader`'s `limit` becomes `_low_water`/`_high_water` backpressure marks only (`self._limit` exists solely on `DataQueue`). Scope the verdict records: `0` is the only falsy `int`, so this is exactly the `0` case — a negative value is truthy, so the guard runs and rejects every request that has a body rather than removing the limit.
Source: `.venv/lib/python3.13/site-packages/aiohttp/web_request.py` — `Request.read()` :670-684 (`if self._client_max_size:` / `if body_size >= self._client_max_size:`), `Request.post()` :722 and :746-750 and :774-777 (`if 0 < max_size < size:`); plumbing in `.venv/lib/python3.13/site-packages/aiohttp/web_app.py` :128/:175/:510 (no validation, direct propagation); `.venv/lib/python3.13/site-packages/aiohttp/streams.py` :150-155 (parser `limit` → water marks only). `HTTPRequestEntityTooLarge` is raised from only these three sites in the package.
Version: aiohttp 3.13.3 (installed source, `.venv/lib/python3.13/site-packages/aiohttp/__init__.py`)
Fingerprint: sha256:2aaacea7a01e581e5eead2ab1b9e4ba3b65bc2ae3d0f176b2a78ae5b6b76bb4e (web_request.py); sha256:94653f68030dfa1df0cbe2d34c78ba49e287f3276dd46e75057702b29803e590 (web_app.py)
Invalidates-On: the `if self._client_max_size:` guard in `Request.read()` becoming an unconditional or `is not None` comparison, or any new `client_max_size` enforcement site appearing outside `web_request.py`
Invalidates-On: `Application.__init__` or `Request.__init__` rejecting/normalizing `0` (e.g. `if client_max_size <= 0: raise ValueError`) or substituting a default when given `0`
Invalidates-On: the payload `StreamReader` enforcing its `limit` as a total-body cap rather than using it only for `_low_water`/`_high_water` backpressure
Invalidates-On: the installed aiohttp version's `web_request.py` no longer hashing to the fingerprint above
Checked-At: 2026-10-06T22:20:43Z
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-393.md

---

Claim: CL:aiohttp/client-max-size-enforcement
Statement: aiohttp raises `web.HTTPRequestEntityTooLarge` while reading a request body that exceeds the application's configured `client_max_size`
Subject: pkg:aiohttp@3.13.3
Verdict: HOLDS
Actual: While accumulating the request body, `Request.read()` compares the running length against the request's `_client_max_size` and raises `HTTPRequestEntityTooLarge(max_size=..., actual_size=...)` once the body reaches it; that value is the application's own, stored by `Application.__init__` and passed by `Application._make_request` into every `Request`. The same exception is raised from both multipart branches of `Request.post()` (`0 < max_size < size`), and `text()`/`json()` inherit the check because they go through `read()`. The exception then escapes the handler and `web_protocol.py` catches `HTTPException` and uses it directly as the response. Two precisions that do not overturn the claim: the comparison is `>=`, so a body whose size equals `client_max_size` also raises; and the check lives in the `Request` body API only, so reading `request.content`/`request.payload` directly bypasses it.
Source: `.venv/lib/python3.13/site-packages/aiohttp/web_request.py` — `Request.read()` :670-684 (`raise HTTPRequestEntityTooLarge(max_size=self._client_max_size, actual_size=body_size)` at :678-680), `Request.post()` :720-722/:746-750/:774-777; `.venv/lib/python3.13/site-packages/aiohttp/web_app.py` :128/:175 (stored, unvalidated) and :510 (`_make_request`); `.venv/lib/python3.13/site-packages/aiohttp/web_protocol.py` :513-515 (response conversion). A package-wide grep shows `web_request.py` as the only raiser.
Version: aiohttp 3.13.3 (installed source, `.venv/lib/python3.13/site-packages/aiohttp/__init__.py`)
Fingerprint: sha256:2aaacea7a01e581e5eead2ab1b9e4ba3b65bc2ae3d0f176b2a78ae5b6b76bb4e (web_request.py); sha256:94653f68030dfa1df0cbe2d34c78ba49e287f3276dd46e75057702b29803e590 (web_app.py)
Invalidates-On: the `>=` comparison or the `raise` in `Request.read()` being replaced (different exception type, a 413 returned rather than raised, or truncation instead of an error)
Invalidates-On: `Application._make_request` ceasing to forward `client_max_size`, or `Request.__init__` overriding it with a per-request or default value
Invalidates-On: enforcement moving to the parser or protocol layer (`http_parser.py` / `web_protocol.py`) so the error is raised before the handler reads the body
Invalidates-On: `Request.content`/`payload` starting to apply `client_max_size`, changing what "reading a request body" covers
Invalidates-On: the installed aiohttp version's `web_request.py` no longer hashing to the fingerprint above
Checked-At: 2026-10-06T22:20:43Z
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-394.md

---

Claim: CL:aiohttp/entity-too-large-status-code
Statement: `aiohttp.web.HTTPRequestEntityTooLarge` carries the HTTP status code 413
Subject: pkg:aiohttp@3.13.3
Verdict: HOLDS
Actual: `HTTPRequestEntityTooLarge` is defined as `class HTTPRequestEntityTooLarge(HTTPClientError)` with the class attribute `status_code = 413`, overriding the inherited `HTTPClientError.status_code = 400`. Because it subclasses `HTTPException`, itself a `Response` subclass that builds with `status=self.status_code`, an instance carries 413 both as `exc.status_code` and as the status of the response aiohttp sends. The name is re-exported from `aiohttp.web` (an explicit `as` re-export plus an entry in `web.__all__`), so `aiohttp.web.HTTPRequestEntityTooLarge` resolves to this class.
Source: `.venv/lib/python3.13/site-packages/aiohttp/web_exceptions.py` — :313-314 (`class HTTPRequestEntityTooLarge(HTTPClientError):` / `status_code = 413`); supporting :80/:85/:106 (`HTTPException(Response, Exception)`, `status=self.status_code`), :234/:239 (`HTTPClientError` defaults to 400). Export: `.venv/lib/python3.13/site-packages/aiohttp/web.py` :69 and :205. No `__init__.py` or other module overrides `status_code` for this class.
Version: aiohttp 3.13.3 (installed source, `.venv/lib/python3.13/site-packages/aiohttp/__init__.py`)
Fingerprint: sha256:ee722e8b0859dfdbc927d2ab5aa02b03941c59b51daa4cf608bc04a496a978df (web_exceptions.py); sha256:27348d99e8e0e46e987850279088197f2b6a6d4f3ab0dbbce38c982a6a14a6ab (web.py)
Invalidates-On: `status_code` on `HTTPRequestEntityTooLarge` changing value, or the attribute being removed so the inherited `HTTPClientError.status_code = 400` applies
Invalidates-On: the class being re-parented to a class with a different `status_code`, or `HTTPException.__init__` ceasing to derive the response status from `status_code`
Invalidates-On: the symbol being dropped from `aiohttp/web.py`'s re-exports/`__all__`, so `aiohttp.web.HTTPRequestEntityTooLarge` no longer resolves to this class
Invalidates-On: the installed aiohttp version's `web_exceptions.py` no longer hashing to the fingerprint above
Checked-At: 2026-10-06T22:20:43Z
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-395.md

---

Claim: CL:aiohttp/client-timeout-unset-total
Statement: `aiohttp.ClientTimeout(total=None)` leaves the cumulative deadline unset, so no total-time timeout fires for a request built with it
Subject: pkg:aiohttp@3.13.3
Verdict: HOLDS
Actual: `ClientTimeout.total` is passed verbatim into `TimeoutHandle`, and both of the handle's arms are conditional on a non-`None`, strictly positive timeout: `TimeoutHandle.start()` returns `None` — no `loop.call_at` deadline is scheduled at all — and `TimeoutHandle.timer()` returns a `TimerNoop` instead of a `TimerContext`. With `total=None` there is therefore no cumulative deadline object and no task-cancelling timer for the request; `connect`, `sock_connect` and `sock_read` are armed independently and are unaffected. Two source facts sharpen this: `total: Optional[float] = None` is the dataclass default, so a bare `ClientTimeout()` also disables the total deadline and only `ClientSession`'s own `DEFAULT_TIMEOUT = ClientTimeout(total=5 * 60, sock_connect=30)` supplies one; and `timeout=0` behaves the same as `None` (`timeout is not None and timeout > 0` is false), so `ClientTimeout(total=0)` does not produce an immediate timeout either. A side use exists at `connector.py:1458`/`:1467`, where `timeout.total` is forwarded as `ssl_handshake_timeout=` to `loop.start_tls()`; with `total=None` that is asyncio's own no-handshake-timeout, which creates no total-time deadline.
Source: `.venv/lib/python3.13/site-packages/aiohttp/helpers.py` :609-625 (`TimeoutHandle.start`/`timer`), :643-656 (`TimerNoop`), :659-673 (`TimerContext.assert_timeout`); constructed from `.venv/lib/python3.13/site-packages/aiohttp/client.py` :202-207 (`total: Optional[float] = None`), :224 (`DEFAULT_TIMEOUT`), :592-595 (`tm = TimeoutHandle(self._loop, real_timeout.total, ...)` / `handle = tm.start()`). Corroboration (not the basis): `https://docs.aiohttp.org/en/v3.13.3/_sources/client_reference.rst.txt`, `ClientTimeout` entry. No documentation is installed inside the package.
Version: aiohttp 3.13.3 (`__version__ = "3.13.3"` in `.venv/lib/python3.13/site-packages/aiohttp/__init__.py`)
Fingerprint: sha256:7ce41fc1c2142f5346015455e030e3ebec22a41cde0fc299a66ce13be2cb2a9e (client.py); sha256:435df4ecf0849d6cf844ff1cad4c3c764e7c734605d848b7272c2429f471cf91 (helpers.py); sha256:416b2c15a0fe0da1457303f7ea52d44334669526792b18af24153eca0e42d708 (aiohttp/__init__.py)
Invalidates-On: any change to `TimeoutHandle.start()` or `TimeoutHandle.timer()` in `helpers.py` that drops or inverts the `timeout is not None and timeout > 0` guards, or gives `None` a default deadline
Invalidates-On: any change in `client.py` substituting a non-`None` fallback for `real_timeout.total` before constructing `TimeoutHandle`, or changing `ClientTimeout.total`'s default
Invalidates-On: an aiohttp version bump whose `helpers.py` or `client.py` sha256 differs from the digests above (notably any 4.x rewrite of the client timeout plumbing)
Checked-At: 2026-10-06T22:21:05Z
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-396.md

---

Claim: CL:aiohttp/client-timeout-payload-span
Statement: aiohttp's cumulative `ClientTimeout.total` deadline covers consumption of the response payload, not only the wait for response headers
Subject: pkg:aiohttp@3.13.3
Verdict: HOLDS
Actual: The single `TimerContext` built from `total` is handed all the way down to the response `StreamReader`, and the payload reader re-enters that same context on every wait for data, so the total deadline is live across body consumption and not just header arrival. `client.py:591-595` constructs the handle from `real_timeout.total` immediately above the library's own comment that the timeout is cumulative for all request operations (request, redirects, responses, data consuming); `client.py:743-744` passes the *same* object as `conn.protocol.set_response_params(timer=timer, ...)` — `read_timeout=real_timeout.sock_read` is passed separately, so the payload timer and `sock_read` are distinct mechanisms; `client_proto.py:237-249` → `http_parser.py:272`/`:397-402`/`:419-424`/`:435-440` → `streams.py:166` keeps it as `self._timer`, which becomes `ClientResponse.content`. `StreamReader._wait()` (`streams.py:354-372`) does `with self._timer:` around `await waiter`, and `TimerContext.__enter__` (`helpers.py:675-694`) registers the current task — a task blocked awaiting body bytes is inside the cumulative deadline's task set, is cancelled by `TimerContext.timeout()` (`:716-721`), and `__exit__` (`:696-714`) converts that back into `asyncio.TimeoutError`; `_read_nowait()` (`streams.py:567-569`) calls `self._timer.assert_timeout()` on every buffered read. The deadline stays armed through body consumption because `client.py:908-913` does not cancel `handle` when `_request` returns — it registers `handle.cancel` as a connection-release callback (`connector.py:202-223`), fired only once the payload is consumed or the response released.
Source: `.venv/lib/python3.13/site-packages/aiohttp/streams.py` :166, :354-372 (`_wait`), :567-569 (`_read_nowait`); `.venv/lib/python3.13/site-packages/aiohttp/client.py` :591-595, :622-624, :743-748, :908-913; `.venv/lib/python3.13/site-packages/aiohttp/helpers.py` :675-721; `.venv/lib/python3.13/site-packages/aiohttp/client_proto.py` :237-249; `.venv/lib/python3.13/site-packages/aiohttp/http_parser.py` :272, :397-440. Corroboration (not the basis): `https://docs.aiohttp.org/en/v3.13.3/_sources/client_reference.rst.txt`, `ClientTimeout.total`. No documentation is installed inside the package.
Version: aiohttp 3.13.3 (`__version__ = "3.13.3"` in `.venv/lib/python3.13/site-packages/aiohttp/__init__.py`)
Fingerprint: sha256:ae5c0bede93a0a430c62297f7bf128916bf6eae1e6b738b794aaa59cb36b5c27 (streams.py); sha256:7ce41fc1c2142f5346015455e030e3ebec22a41cde0fc299a66ce13be2cb2a9e (client.py); sha256:435df4ecf0849d6cf844ff1cad4c3c764e7c734605d848b7272c2429f471cf91 (helpers.py); sha256:e7afd6b4b4ad6460453d82b3804818eafdb8264855d72e891269aec5e253d92a (client_proto.py); sha256:cfa763643394b3b85d3f3a654c4b00572cf4a1ffab400c13ef1cfc3a95e19ee6 (http_parser.py)
Invalidates-On: removing or narrowing `with self._timer:` in `streams.py::_wait`, or dropping the `self._timer.assert_timeout()` call in `_read_nowait`
Invalidates-On: `set_response_params(...)` no longer receiving the request-level `timer`, or `HttpResponseParser`/`StreamReader` no longer threading `timer` into the payload
Invalidates-On: `client.py` cancelling `handle` at `_request` return instead of deferring it to `resp.connection.add_callback(handle.cancel)`, which would stop the deadline at headers
Invalidates-On: a refactor introducing a separate response-body timer in place of `total` (the commented-out `response_body_timeout` field at `client.py:202-216`), or an aiohttp version bump whose listed files hash differently from the digests above
Checked-At: 2026-10-06T22:21:05Z
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-397.md

---

Claim: CL:aiohttp/client-response-text-charset-source
Statement: aiohttp's `ClientResponse.text()` decodes the payload with the charset named in the response's Content-Type header
Subject: pkg:aiohttp@3.13.3
Verdict: HOLDS
Actual: With the default `encoding=None`, `text()` calls `self.get_encoding()`, which parses the `Content-Type` header, takes its `charset` parameter, and returns `codecs.lookup(encoding).name`; `text()` then runs `self._body.decode(encoding, errors=errors)`. The Content-Type charset is therefore the encoding actually used — the first precedence, outranking the JSON/`rdap` UTF-8 default and the session's `fallback_charset_resolver`. Two qualifications that do not contradict the claim: an explicit `encoding=` argument to `text()` bypasses header detection entirely (`if encoding is None: encoding = self.get_encoding()`), and the header charset is used only when Python knows the codec — a `codecs.lookup()` failure is suppressed at `client_reqrep.py:715` and detection then falls through to the JSON/`rdap` UTF-8 default or the session's `fallback_charset_resolver`.
Source: `.venv/lib/python3.13/site-packages/aiohttp/client_reqrep.py` :709-740 (`ClientResponse.get_encoding`, `ClientResponse.text`); corroborated by upstream `docs/aiohttp/client_reference.rst` published as `ClientResponse.text` ("If *encoding* is `None` content encoding is determined from the Content-Type header, or using the `fallback_charset_resolver` function") and `ClientResponse.get_encoding`.
Version: aiohttp 3.13.3 (installed, `.venv/lib/python3.13/site-packages/aiohttp-3.13.3.dist-info/METADATA`); docs consulted from `docs.aiohttp.org/en/stable` as corroboration only
Fingerprint: sha256:7844440eb67433c6454edd708c776e47e3fcfec9b8d0aeb980dcfe88c1986ac9 (client_reqrep.py)
Invalidates-On: the installed `client_reqrep.py` digest changing and `ClientResponse.text`/`get_encoding` no longer resolving encoding from the `Content-Type` `charset` parameter
Invalidates-On: `get_encoding()` being changed to reorder precedence (body sniffing or a JSON default ahead of the header charset)
Invalidates-On: the response class in use being a `response_class=` override whose `text()` does not delegate to `ClientResponse.get_encoding`
Checked-At: 2026-10-06
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-398.md

---

Claim: CL:aiohttp/client-response-text-unknown-charset-error
Statement: aiohttp's `ClientResponse.text()` raises no `LookupError` for an unresolvable Content-Type charset, because `get_encoding()` suppresses it and decodes with UTF-8 or the session's `fallback_charset_resolver`
Subject: pkg:aiohttp@3.13.3
Verdict: HOLDS
Actual: `get_encoding()` wraps the only call that can resolve the declared charset — `codecs.lookup(encoding).name` — in `contextlib.suppress(LookupError, ValueError)` (:715), so an unresolvable Content-Type `charset=` never propagates: control falls through to the JSON/RDAP branch, which returns `"utf-8"` (:718-723), or to `self._resolve_charset(self, self._body)` (:730), which `ClientSession.__init__` binds to the `fallback_charset_resolver` argument whose default is `lambda r, b: "utf-8"`. `text(encoding=None)` therefore decodes `self._body` (:740) with a canonical codec name from a successful lookup, the literal `"utf-8"`, or whatever the session resolver returns — no `LookupError` on the Content-Type path. Boundaries the verdict records: passing `encoding=` explicitly bypasses `get_encoding()` and `bytes.decode()` then *does* raise `LookupError` for a bad name (the caller's argument, not the response's Content-Type); a caller-supplied resolver returning an unresolvable name puts the `LookupError` back at :740, since the suppression covers the Content-Type charset only; and `get_encoding()` on a not-yet-read body raises `RuntimeError`, unreachable via `text()` which awaits `read()` first.
Source: `.venv/lib/python3.13/site-packages/aiohttp/client_reqrep.py` :709-740 (`get_encoding`, `text`) — suppression at :715, utf-8 branch :718-723, resolver call :730, decode :740; corroborating `aiohttp/client.py` :230 (`_CharsetResolver` type alias), :306 (`fallback_charset_resolver: _CharsetResolver = lambda r, b: "utf-8"`), :425 (`self._resolve_charset = fallback_charset_resolver`); `aiohttp/helpers.py` :323-359 (`parse_mimetype` strips quotes/whitespace, so a bogus charset value does reach `codecs.lookup`). The wheel ships no docs, so installed production source is the establishing source.
Version: aiohttp 3.13.3 (`aiohttp/__init__.py` line 1 `__version__ = "3.13.3"`; installed as `aiohttp-3.13.3.dist-info`), CPython 3.13 on darwin
Fingerprint: sha256:7844440eb67433c6454edd708c776e47e3fcfec9b8d0aeb980dcfe88c1986ac9 (client_reqrep.py — establishes the verdict); sha256:7ce41fc1c2142f5346015455e030e3ebec22a41cde0fc299a66ce13be2cb2a9e (client.py — resolver binding and default); sha256:435df4ecf0849d6cf844ff1cad4c3c764e7c734605d848b7272c2429f471cf91 (helpers.py — charset parameter extraction)
Invalidates-On: `client_reqrep.py` no longer matching sha256:7844440eb67433c6454edd708c776e47e3fcfec9b8d0aeb980dcfe88c1986ac9, specifically if `contextlib.suppress(LookupError, ValueError)` at :715 is narrowed, removed, or moved to a `try/except` that re-raises
Invalidates-On: an aiohttp release where `get_encoding()` returns the raw declared charset instead of `codecs.lookup(...).name` on success, or where the fallthrough after a failed lookup stops routing to `_resolve_charset`
Invalidates-On: `client.py` changing the `fallback_charset_resolver` default away from `lambda r, b: "utf-8"` to a resolver that returns unvalidated or Content-Type-derived names
Invalidates-On: `text()` changed to call `codecs.lookup(encoding)` outside a suppress before decoding, or to resolve an explicit `encoding=` argument through `get_encoding()`
Checked-At: 2026-10-06
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-413.md
---

Claim: CL:aiohttp/client-response-read-charset-application
Statement: aiohttp's `ClientResponse.read()` hands over payload bytes without applying the response's charset
Subject: pkg:aiohttp@3.13.3
Verdict: HOLDS
Actual: `read()` is byte-level only. It fills the body cache from the stream reader — `self._body = await self.content.read()` — and returns `self._body`, annotated `-> bytes`. Its body never touches `self.headers`, never parses `Content-Type`, and never calls `get_encoding()`, `codecs.lookup`, or `.decode()`; `StreamReader.read()` (`streams.py:414`) likewise returns `bytes`. Charset handling lives entirely in the separate `text()`/`json()` layer, which calls `await self.read()` first and then decodes. The verdict records one distinction for a later re-check: `read()` does return *decompressed* bytes when `auto_decompress` is on, but that is `Content-Encoding` (gzip/deflate) applied at the transport layer during streaming, not the `Content-Type` charset — decompression is not charset application.
Source: `.venv/lib/python3.13/site-packages/aiohttp/client_reqrep.py` :689-707 (`ClientResponse.read`), contrasted with :732-740 (`ClientResponse.text`); `.venv/lib/python3.13/site-packages/aiohttp/streams.py` :414 (`StreamReader.read` returns `bytes`). Corroborated by upstream `client_reference.rst`: `read()` — "Read the whole response's body as `bytes`."
Version: aiohttp 3.13.3 (installed, `.venv/lib/python3.13/site-packages/aiohttp-3.13.3.dist-info/METADATA`)
Fingerprint: sha256:7844440eb67433c6454edd708c776e47e3fcfec9b8d0aeb980dcfe88c1986ac9 (client_reqrep.py)
Invalidates-On: the installed `client_reqrep.py` digest changing and `ClientResponse.read` beginning to reference `get_encoding()`, the `Content-Type` header, or any decode/codec call
Invalidates-On: `read()`'s return annotation or body changing from raw `self._body` bytes to a decoded or charset-normalised value
Invalidates-On: `response_class=`/`_body` being overridden so the cached value is not the raw stream bytes
Checked-At: 2026-10-06
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-400.md

---

Claim: CL:aiohttp/request-json-unknown-charset-error
Statement: aiohttp's `web.Request.json()` raises `LookupError`, not `UnicodeDecodeError`, when the request's Content-Type names a charset whose codec Python does not have
Subject: pkg:aiohttp@3.13.3
Verdict: HOLDS
Actual: `BaseRequest.json()` is a two-line delegation — `body = await self.text()` then `loads(body)` — and `BaseRequest.text()` ends in `bytes_body.decode(self.charset or "utf-8")`, where `self.charset` is the raw `charset` parameter lifted out of the Content-Type header with no check against Python's codec registry (`parse_content_type()` only splits the header, never validating it). So an unknown codec name reaches CPython's codec lookup and surfaces as `LookupError: unknown encoding: <charset>`, propagating out of `json()` untouched — neither method contains a `try`/`except`. `UnicodeDecodeError` belongs to the *other* branch only (a codec Python knows, bytes malformed for it), and by hierarchy it is `UnicodeDecodeError → UnicodeError → ValueError`, disjoint from `LookupError`, so the two are never alternatives for the same failure. Precision the verdict records: the `raise` executes one frame below, inside `text()`, but under either reading of "json() raises" the exception a caller of `json()` receives is `LookupError`, and the claim's "not `UnicodeDecodeError`" half is true either way. Negative check on the same path: every `LookupError` reference in the package is `payload.LookupError`, netrc/proxy helpers, multipart, or client-response code — none wraps `BaseRequest.text()`/`json()`, and no `UnicodeDecodeError` handler exists in `web_request.py` at all.
Source: `.venv/lib/python3.13/site-packages/aiohttp/web_request.py` :686-695 (`text()` at :689-690 — `encoding = self.charset or "utf-8"` / `return bytes_body.decode(encoding)`; `json()` at :692-695; `Request` subclasses `BaseRequest` at :842/:118 and overrides neither — a grep for `async def text`/`async def json` matches only :686 and :692); `.venv/lib/python3.13/site-packages/aiohttp/helpers.py` :769-776 (`HeadersMixin.charset`) and :388-399 (`parse_content_type()`). Corroborating CPython: `https://docs.python.org/3/library/codecs.html#codecs.lookup`, `https://docs.python.org/3/library/exceptions.html#LookupError`, and CPython `Python/codecs.c` @ tag `v3.13.0` `lookup_codec()` (`PyErr_Format(PyExc_LookupError, "unknown encoding: %s", encoding)`), reached from `bytes.decode` via `bytes_decode_impl → PyUnicode_FromEncodedObject → PyUnicode_Decode`.
Version: aiohttp 3.13.3 (`aiohttp/__init__.py:1`, inside the supplied source root); CPython docs consulted at the `3.14` build and upstream sources pinned at tag `v3.13.0`
Fingerprint: sha256:2aaacea7a01e581e5eead2ab1b9e4ba3b65bc2ae3d0f176b2a78ae5b6b76bb4e (web_request.py); sha256:435df4ecf0849d6cf844ff1cad4c3c764e7c734605d848b7272c2429f471cf91 (helpers.py)
Invalidates-On: an aiohttp release wrapping `BaseRequest.text()`'s `.decode()` in `try`/`except LookupError`, or validating/normalizing `charset` against the codec registry before decoding (re-check `web_request.py:686-695`, `helpers.py` `HeadersMixin.charset`)
Invalidates-On: an aiohttp release where `json()` stops delegating to `text()` (e.g. decoding through a bytes-capable `loads`), changing which frame raises and with what error
Invalidates-On: a CPython release reporting an unregistered encoding name as `ValueError`/`UnicodeDecodeError` instead of `LookupError` (re-check `lookup_codec()` in `Python/codecs.c` and `Doc/library/codecs.rst`)
Checked-At: 2026-10-06
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-401.md

---

Claim: CL:python/codecs-lookup-unknown-name
Statement: Python raises `LookupError` when a codec name is looked up and no such codec exists
Subject: lang:python
Verdict: HOLDS
Actual: `codecs.lookup(encoding)` is documented to scan the registry cache and then the registered search functions, and "If no `CodecInfo` object is found, a `LookupError` is raised." Every convenience function that resolves an encoding name through `lookup()` — `getencoder`, `getdecoder`, `getincrementalencoder`, `getincrementaldecoder`, `getreader`, `getwriter` — is documented to "raise a `LookupError` in case the encoding cannot be found", and the built-in-exceptions reference states that `LookupError` "can be raised directly by `codecs.lookup()`". `codecs.lookup_error(name)` raises `LookupError` the same way for an unknown error-handler name.
Source: `https://raw.githubusercontent.com/python/cpython/v3.14.8/Doc/library/codecs.rst` — `.. function:: lookup(encoding, /)` lines 66-74 (raise-on-not-found), lines 116-162 (`get*` helpers), line 475 (`lookup_error`). Cross-checked in `https://raw.githubusercontent.com/python/cpython/v3.12.0/Doc/library/codecs.rst` (lines 58-65, identical wording) and `https://docs.python.org/3.14/library/exceptions.html#exception-hierarchy` / `exception LookupError`.
Version: CPython 3.14.8 documentation source (tag `v3.14.8`); wording verified unchanged at tag `v3.12.0`
Fingerprint: sha256:a93ab5fd0291451dacdbc3f71a22fbbeba2807550e46b2e2f40734af6b5ae726 (codecs.rst @ v3.14.8); sha256:ed8acfbc828839f7e657406f40c486cc6ddd5b06d26bcd2d8afca89dd5760687 (codecs.rst @ v3.12.0, corroboration); sha256:afdce60fa94da72d1c097b472257c770c6bed41342509b34863b6da7833b745e (docs.python.org/3.14/library/exceptions.html, corroboration)
Invalidates-On: the `codecs.lookup` doc paragraph changing the not-found exception to something other than `LookupError`, or dropping the explicit raise sentence
Invalidates-On: CPython changing codec-registry semantics so an unknown name returns a fallback identity/ASCII codec or is auto-registered instead of raising
Invalidates-On: the project's interpreter being pinned to a Python release whose `Doc/library/codecs.rst` differs from the `v3.12.0`/`v3.14.8` bytes above (hash mismatch on that version's file)
Invalidates-On: a custom codec search function swallowing the miss and returning a `CodecInfo` (per-call behaviour; the documented registry contract is unchanged)
Checked-At: 2026-10-06T22:24:59Z
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-405.md

---

Claim: CL:python/lookup-error-inheritance
Statement: `LookupError` is not a subclass of `UnicodeDecodeError`
Subject: lang:python
Verdict: HOLDS
Actual: `LookupError` is a base class hanging directly off `Exception`, and its only documented subclasses are `IndexError` and `KeyError`. `UnicodeDecodeError` sits in a separate branch (`Exception → ValueError → UnicodeError → UnicodeDecodeError`), so `LookupError` is neither it nor derived from it — the two are unrelated by subclassing, and the same page states that "Two exception classes that are not related via subclassing are never equivalent." Consequence: an `except UnicodeDecodeError` clause does not catch a `LookupError` raised by a codec-name lookup.
Source: `https://docs.python.org/3.14/library/exceptions.html` — "Base classes" section, `exception LookupError` (plain-text lines 398-401 of the fetched HTML), and the "Exception hierarchy" tree (fetched HTML lines ~1403-1500): `├── LookupError` at `Exception`'s indent level with children `IndexError` and `KeyError`; separately `├── ValueError` → `└── UnicodeError` → `├── UnicodeDecodeError`. Checked from that page alone.
Version: Python 3.14.8 documentation (docs.python.org/3.14, page footer "Python 3.14.8 Documentation", last updated 2026-10-06 UTC)
Fingerprint: sha256:afdce60fa94da72d1c097b472257c770c6bed41342509b34863b6da7833b745e (exceptions.html, 144074 bytes)
Invalidates-On: the "Exception hierarchy" tree re-parenting `LookupError` (or `UnicodeError`) so the two branches merge
Invalidates-On: a documented change to the `LookupError` entry adding a Unicode/decode subclass
Invalidates-On: reuse against an interpreter version whose own doc page differs (3.9-3.13 pages not hashed here) — any hash change to the fetched page requires re-verification
Checked-At: 2026-10-06T22:24:59Z
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-406.md

---

Claim: CL:python/unicode-decode-error-inheritance
Statement: `UnicodeDecodeError` is not a subclass of `LookupError`
Subject: lang:python
Verdict: HOLDS
Actual: `UnicodeDecodeError` inherits `UnicodeError`, which inherits `ValueError` — "Raised when a Unicode-related error occurs during decoding. It is a subclass of `UnicodeError`." / "It is a subclass of `ValueError`." `LookupError` is a sibling branch off `Exception` whose only documented subclasses are `IndexError` and `KeyError`, so `UnicodeDecodeError` is not below it in the hierarchy. Consequence: `except LookupError` does not catch a decoding failure, and `except ValueError` does not catch an invalid codec-name lookup — the two need separate handlers (`UnicodeDecodeError` is reachable via `ValueError`/`UnicodeError`, `LookupError` only via `LookupError`).
Source: `https://docs.python.org/3.14/library/exceptions.html` — "Concrete exceptions": `exception UnicodeDecodeError` ("subclass of `UnicodeError`", plain-text lines 955-957 of the fetched HTML) and `exception UnicodeError` ("subclass of `ValueError`", lines 904-906); the "Exception hierarchy" tree, where `UnicodeDecodeError` is nested under `ValueError` → `UnicodeError` and `LookupError` sits at `Exception` level with only `IndexError`/`KeyError` beneath it. Checked from this page on its own; it does not rely on claim-406's verdict.
Version: Python 3.14.8 documentation (docs.python.org/3.14, page footer "Python 3.14.8 Documentation", last updated 2026-10-06 UTC)
Fingerprint: sha256:afdce60fa94da72d1c097b472257c770c6bed41342509b34863b6da7833b745e (exceptions.html, 144074 bytes)
Invalidates-On: the hierarchy tree re-parenting `UnicodeError`/`UnicodeDecodeError` under `LookupError` (or introducing multiple inheritance from it)
Invalidates-On: a documented change to the `UnicodeDecodeError` or `UnicodeError` entry naming a different base class
Invalidates-On: reuse against an interpreter version whose own doc page differs (3.9-3.13 pages not hashed here) — any hash change requires re-verification
Checked-At: 2026-10-06T22:24:59Z
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-407.md

---

Claim: CL:python/asyncmock-async-context-manager
Statement: `unittest.mock.AsyncMock` supplies awaitable `__aenter__` and `__aexit__`, so an `AsyncMock` can stand in for an object consumed with `async with`
Subject: lang:python
Verdict: HOLDS
Actual: `AsyncMock` inherits the magic-method installer (`AsyncMagicMixin` → `MagicMixin`), whose `_mock_set_magics()` puts a `MagicProxy` on the type for every entry of `_magics | _async_method_magics`, and `_async_method_magics` is exactly `{"__aenter__", "__aexit__", "__anext__"}`. Attribute lookup therefore routes through `MagicProxy.__get__` → `Mock._get_child_mock()`, which for an `AsyncMockMixin` parent names anything not in `_all_sync_magics` an `AsyncMock` — and calling an `AsyncMock` runs `CallableMixin.__call__` → `_mock_call` → `AsyncMockMixin._execute_mock_call`, an `async def`, so the result is a coroutine (awaitable). `async with AsyncMock()` therefore enters and exits through awaitables; `__aexit__` is preconfigured with `return_value = False` so the body is not suppressed, and `__aenter__` returns a new `AsyncMock` bound to the `as` target. Bound the verdict records: this is the un-spec'd construction — with `spec=`/`spec_set=` (or an autospec'd `patch`) of a type that has no `__aenter__`, `_mock_set_magics` computes `these_magics = orig_magics.intersection(self._mock_methods)` and deletes the extras, pruning the awaitable pair.
Source: `https://raw.githubusercontent.com/python/cpython/v3.11.10/Lib/unittest/mock.py` — lines 2003, 2008, 2038, 2114, 2133, 2149, 2182-2192, 1036-1042, 1119-1128, 2228, 2390. Corroboration: `https://docs.python.org/3.11/library/unittest.mock.html` (magic-methods support list, "Async Mock"). Nothing was executed; the local interpreter was not probed.
Version: CPython tag `v3.11.10`, `Lib/unittest/mock.py` (repo declares `requires-python = ">=3.11"`; mechanism re-checked unchanged at `v3.14.0` — `_async_method_magics` at 2070, `orig_magics` at 2181, "Any asynchronous magic becomes an AsyncMock" at 1080, `class AsyncMock(AsyncMockMixin, AsyncMagicMixin, Mock)` at 2476)
Fingerprint: sha256:2576119e5653ca21ab32dd5a1a5f5604deb2b1c7a240e315710ea927bc193749 (v3.11.10 Lib/unittest/mock.py); sha256:667d9ed88fd330a3d45419436b13ac41488fca77f39959dca6e93642cf80a678 (v3.14.0 Lib/unittest/mock.py, confirmation read only)
Invalidates-On: the consulted `Lib/unittest/mock.py` bytes changing (fingerprint mismatch) — specifically edits to `_async_method_magics`, `MagicMixin._mock_set_magics`, `MagicProxy`, `Mock._get_child_mock`, or `AsyncMockMixin._execute_mock_call`
Invalidates-On: upstream removing or renaming `_async_method_magics` / `AsyncMagicMixin`, or dropping `__aenter__`/`__aexit__` from the magics installed on `AsyncMock`
Invalidates-On: the `AsyncMock` in question being constructed with `spec=`, `spec_set=`, or an autospec `patch()` whose target lacks `__aenter__`/`__aexit__` — the proxies are then pruned and are not awaitable
Invalidates-On: the docs page for the resolved interpreter version no longer listing `__aenter__`/`__aexit__` as supported magic methods
Checked-At: 2026-10-06
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-408.md

---

Claim: CL:python/magicmock-async-context-manager
Statement: A plain, unspecced `unittest.mock.MagicMock` folds `__aenter__`/`__aexit__` into its magic set and routes those children to `AsyncMock`, so `async with` succeeds on it without explicit configuration
Subject: lang:python
Verdict: HOLDS
Actual: All three links of the chain verify in `Lib/unittest/mock.py`, identically on two pinned tags. (1) `_async_method_magics = {"__aenter__", "__aexit__", "__anext__"}` is unioned into `_all_magics` (`_all_magics = _all_sync_magics | _async_magics`) and `MagicMixin._mock_set_magics` installs a `MagicProxy` on the mock's type for every entry of `_magics | _async_method_magics`, so both async context-manager dunders are in the fold. (2) `MagicProxy.__get__` -> `create_mock()` builds each child via `parent._get_child_mock(name=entry, _new_name=entry, ...)`, which contains the explicit branch `if issubclass(_type, MagicMock) and _new_name in _async_method_magics: # Any asynchronous magic becomes an AsyncMock` -> `klass = AsyncMock`, so on a plain `MagicMock` the children `m.__aenter__` and `m.__aexit__` are `AsyncMock` instances. (3) An `AsyncMock` call is awaitable (`AsyncMockMixin._execute_mock_call` is `async def`) and `create_mock`'s `_set_return_value` applies the `_return_values` default `'__aexit__': False`, so `async with m:` enters (yielding `m.__aenter__.return_value`, itself an `AsyncMock`) and exits cleanly with no configuration. "Plain and unspecced" is load-bearing and consistent with the code: a `spec=`/`spec_set=`/autospec mock prunes `_mock_set_magics` to the spec's intersection, and a non-`MagicMock` parent (`Mock`, `NonCallableMock`) skips the `AsyncMock` routing branch and genuinely fails under `async with`.
Source: primary — CPython upstream `Lib/unittest/mock.py` at tag `v3.11.0` (https://raw.githubusercontent.com/python/cpython/v3.11.0/Lib/unittest/mock.py): ~1966-1972 (`_async_method_magics`/`_all_magics`), ~1991-2005 (`_return_values['__aexit__']`), ~1017 (`_get_child_mock` AsyncMock routing), ~2077 (`_mock_set_magics`), ~2140 (`MagicProxy.create_mock`). Corroborated structurally at tag `v3.13.1` (same constants and routing) and by https://docs.python.org/3/library/unittest.mock.html (supported magic methods — "Context manager: `__enter__`, `__exit__`, `__aenter__` and `__aexit__`"; MagicMock's default-magic-methods text; defaults table `__aexit__: False`; "Changed in version 3.8: Added support for `__aenter__`, `__aexit__`"). Repo `pyproject.toml` `requires-python = ">=3.11"` consulted only to choose which upstream tags to read. Nothing was executed.
Version: CPython 3.11.0 and 3.13.1 (upstream source at release tags); docs.python.org/3 fetched 2026-10-06 — the mechanism is present throughout the repo's supported `>=3.11` range (since 3.8)
Fingerprint: sha256:663efa5704c1146a2327248637fe1f427a4b517dfb28250f6894e844f0b5c1eb (Lib/unittest/mock.py @ v3.11.0, primary establishing source); sha256:7c6b6c92b11243ebe8e93e0230be077b83224ccf0030228c0aa45d1db5613811 (Lib/unittest/mock.py @ v3.13.1, corroboration); sha256:6513ae8d94214d97ca409b654b564a64390e4cd66dc7ff8cd6ebf705c3f01a9f (docs.python.org/3/library/unittest.mock.html as fetched, corroboration)
Invalidates-On: a CPython change removing `__aenter__`/`__aexit__` from `_async_method_magics`, from `_all_magics`/`_mock_set_magics`, or from the `_get_child_mock` AsyncMock-routing branch
Invalidates-On: a CPython change to `_return_values['__aexit__']` or to `MagicProxy.create_mock`'s `_set_return_value` wiring
Invalidates-On: running on Python < 3.8 (async magic support absent), or using the standalone `mock` backport at a version predating this routing
Invalidates-On: the mock in question ceasing to be plain and unspecced — `spec`/`spec_set`/`autospec` prune the magic set, and a non-`MagicMock` parent skips the AsyncMock branch
Checked-At: 2026-10-06
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-415.md
---

Claim: CL:python/asyncio-sleep-yields-to-loop
Statement: `asyncio.sleep(0)` yields control to the event loop, letting another ready task run
Subject: lang:python
Verdict: HOLDS
Actual: The asyncio docs state that `sleep()` always suspends the current task, allowing other tasks to run, and call out `delay=0` specifically as an "optimized path to allow other tasks to run" — the documented use being to stop a long-running coroutine occupying the event loop for the whole duration of its call. Because asyncio uses cooperative scheduling and runs one task at a time, suspending the awaiting task returns control to the loop, which then runs other ready tasks (or callbacks/IO) before resuming it. The guarantee the source gives is "allows other tasks to run", not "a particular task will run": whether anything else executes depends on the loop having other ready work at that iteration, which is all the claim requires.
Source: `https://docs.python.org/3/library/asyncio-task.html#sleeping` ("Sleeping", `asyncio.sleep(delay, result=None)` entry), corroborated by the same page's "Task Object" section ("Event loops use cooperative scheduling: an event loop runs one Task at a time. While a Task awaits for the completion of a Future, the event loop runs other Tasks, callbacks, or performs IO operations."). Cross-checked against `https://docs.python.org/3.11/library/asyncio-task.html#sleeping` — identical wording. Documentation-only claim; the installed interpreter was not read or executed.
Version: Python 3.14.8 documentation (stable docs URL); corroborating page Python 3.11.17 documentation; repo declares `requires-python = ">=3.11"`, so the claim holds across the supported range as documented
Fingerprint: sha256:0e10a7cd1a7a3a44b1ac67d649a40b8ba34121acd69002a49012e515a85b8cee (full fetched page); sha256:2f85eef501369e2d88a5bdc7a9f05659d1d57bb0580ce364a10a1a348c3a36dd (the `## Sleeping` section slice)
Invalidates-On: the "Sleeping" section of `Doc/library/asyncio-task.rst` changing the `delay=0` wording or removing the "optimized path to allow other tasks to run" statement
Invalidates-On: CPython changing `asyncio.sleep` so a zero delay no longer suspends the calling task (e.g. short-circuiting without scheduling a loop callback) — re-check `Lib/asyncio/tasks.py` / the Task protocol
Invalidates-On: the fetched page content hash differing from the digests above
Invalidates-On: support for Python < 3.11 being dropped or added such that the consulted doc releases no longer cover the deployed interpreter version
Checked-At: 2026-10-06
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-410.md

---

Claim: CL:pytest-asyncio/auto-mode-async-test-collection
Statement: Under pytest-asyncio's `asyncio_mode = "auto"`, an `async def` test function is collected and run on an event loop without an explicit `asyncio` marker
Subject: pkg:pytest-asyncio@1.3.0
Verdict: HOLDS
Actual: In auto mode the plugin's `pytest_pycollect_makeitem` hookwrapper adds the `asyncio` marker itself (`if _get_asyncio_mode(node.config) == Mode.AUTO and not node.get_closest_marker("asyncio"): node.add_marker("asyncio")`) and substitutes the plain `pytest.Function` item with the specialized `Coroutine` item class (`PytestAsyncioFunction.item_subclass_for`, whose `Coroutine._can_substitute` is `inspect.iscoroutinefunction`). That item's `runtest` resolves the scoped `asyncio.Runner` fixture (`_{loop_scope}_scoped_runner`, default loop scope `function` from the `asyncio_default_test_loop_scope` ini default) and wraps the coroutine with `_synchronize_coroutine`, which calls `runner.run(coro, context=context)` — so an author-unmarked `async def` test is both collected as a real test item and executed on an event loop. Under the reading "no `asyncio` marker is present on the item at all", the internal bookkeeping differs (the plugin injects one), but the asserted outcome — collected and run on an event loop — is unchanged under either reading. In strict mode the automatic marker/loop handling does not happen (an unmarked async test is left to pytest core), which is the counterpart behaviour this claim distinguishes.
Source: `.venv/lib/python3.13/site-packages/pytest_asyncio/plugin.py` — lines 118-123 (ini default `asyncio_default_test_loop_scope = "function"`), 198-207 (`_get_asyncio_mode`), 416-426 (`item_subclass_for`), 454-469 (`PytestAsyncioFunction.setup`/`runtest`), 498-504 (`Coroutine._can_substitute`), 575-611 (`pytest_pycollect_makeitem_convert_async_functions_to_subclass`), 703-718 (`_synchronize_coroutine`), 794-830 (`_create_scoped_runner_fixture`)
Version: pytest-asyncio version string not readable from the source root the verdict consulted (`__init__.py` derives `__version__` at runtime via `importlib.metadata`, and the `*.dist-info` directory sits outside that root); the ledger's subject cell names 1.3.0 and the run-40 freshness basis records `pytest_asyncio-1.3.0.dist-info`. Stability is carried by the fingerprint; the consulted API surface (Runner-based `_scoped_runner` fixtures, `asyncio_default_test_loop_scope`, `loop_scope` marker kwarg) corresponds to the pytest-asyncio 0.23+ Runner redesign.
Fingerprint: sha256:121cb98d74596a9e5db3269a4be7131baeeeab3754b5996a9b26c228910863aa (plugin.py, 896 lines, exact bytes consulted)
Invalidates-On: any change to `pytest_pycollect_makeitem_convert_async_functions_to_subclass` (plugin.py:599-611) removing or conditioning the auto-mode `node.add_marker("asyncio")` injection
Invalidates-On: the sha256 of `.venv/lib/python3.13/site-packages/pytest_asyncio/plugin.py` differing from `121cb98d74596a9e5db3269a4be7131baeeeab3754b5996a9b26c228910863aa` (a pytest-asyncio upgrade or downgrade)
Invalidates-On: rename or removal of the `asyncio_default_test_loop_scope` ini option or the `_{scope}_scoped_runner` fixture naming scheme (plugin.py:118-123, 794-830) on which the item's `runtest` depends
Invalidates-On: a pytest core change that stops reporting `async def` functions as collectible `Function` items before `pytest_pycollect_makeitem` runs, since the plugin substitutes an already-produced `Function`
Checked-At: 2026-10-06
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-411.md

---

Claim: CL:aiohttp/client-max-size-sourcing
Statement: aiohttp's `web.Request` enforces its body-size limit from the `client_max_size` configured on the application whose runner serves the connection and built the request
Subject: pkg:aiohttp@3.13.3
Verdict: HOLDS
Actual: The limit is a constructor-time snapshot copied from the `Application` that builds the request: `AppRunner._make_server()` calls `self._app._make_handler(...)`, which constructs `Server(handler=self._handle, request_factory=self._make_request)`; `Application._make_request` passes `client_max_size=self._client_max_size` (the app's own `__init__` kwarg, default `1024**2`) into `BaseRequest.__init__`, which stores it as `self._client_max_size` and raises `HTTPRequestEntityTooLarge` from `read()`/`text()`/`json()`/`post()` once the body reaches it. Each connection's `RequestHandler` holds that same bound factory (`self._request_factory = manager.request_factory`) and calls it per request, so the app that owns the runner is exactly the app that supplies the limit — no per-route or per-connection lookup happens. Scope details that qualify, not contradict: `0` disables the check (`if self._client_max_size:` at web_request.py:675); the gate lives in `Request`'s own body readers, so `request.content` (the raw payload `StreamReader`) is not bounded by it; multipart `post()` counts toward the same value (`max_size = self._client_max_size` at :722) and urlencoded `post()` delegates to `read()`; sub-apps added with `add_subapp` never supply their own `client_max_size`, so body reads stay bounded by the runner's (root) app's value; a `Server` built without a `request_factory` falls back to `Server._make_request`, which constructs `BaseRequest` with the default `1024**2` and no application involved at all; and `request.clone(...)`/`with_method(...)` preserve the app-derived value unless `client_max_size` is passed explicitly.
Source: `/Users/alexhubitski/Projects/buster-ripper/.venv/lib/python3.13/site-packages/aiohttp/web_app.py:492-511` (`Application._make_request` -> `client_max_size=self._client_max_size`), corroborated by `aiohttp/web_runner.py:389-396` (`AppRunner._make_server` -> `self._app._make_handler`), `aiohttp/web_app.py:442-447` (`Server(..., request_factory=self._make_request)`), `aiohttp/web_protocol.py:206, 587-593` (`RequestHandler` calls the manager's factory per request), `aiohttp/web_request.py:161, 198, 285-286, 675-680, 722` (storage plus enforcement), and `aiohttp/web_app.py:128, 175` (the `client_max_size` config point). The installed package ships no docs tree, so installed source is the primary source here.
Version: aiohttp 3.13.3 (`aiohttp-3.13.3.dist-info/METADATA`, `Version: 3.13.3`)
Fingerprint: sha256:94653f68030dfa1df0cbe2d34c78ba49e287f3276dd46e75057702b29803e590 (web_app.py); sha256:2aaacea7a01e581e5eead2ab1b9e4ba3b65bc2ae3d0f176b2a78ae5b6b76bb4e (web_request.py); sha256:bf51b59ca88e3908059d3491e0831ce88f5178414331a1ed30bbcefeba0333e0 (web_runner.py); sha256:eacf5d33399aa96efb6f33354f5d75b863522c5a3a80d99f0e0ed7cd89c0f319 (web_protocol.py); sha256:fbd58329402247d965fab49dc174aa1ba6236a85bbf5dd51e32d01192aa050c0 (web_server.py)
Invalidates-On: an aiohttp release that changes `Application._make_request` to look the limit up per request (e.g. from `request.app` or `match_info.current_app`) instead of `self._client_max_size`, or that adds a per-route/per-connection `client_max_size` override
Invalidates-On: a change to `AppRunner._make_server` or `Application._make_handler` that stops binding `request_factory=self._make_request` to the runner's own app (a router-selected or sub-app-selected factory)
Invalidates-On: a change in `BaseRequest.__init__`/`read()`/`post()` that drops or relocates the `_client_max_size` gate, or extends it to `request.content`
Invalidates-On: replacing the installed aiohttp version — any file above whose sha256 differs from its fingerprint
Checked-At: 2026-10-06T23:05:55Z
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-412.md

---

Claim: CL:python/errors-replace-valid-decode-parity
Statement: With `errors="replace"`, every complete sequence the target codec decodes validly yields the same characters as `errors="strict"` would
Subject: lang:python
Verdict: HOLDS
Actual: `errors` selects an *error handler*, and the documented contract consults a handler only "in case of an error": `'strict'` raises `UnicodeError`, `'replace'` substitutes U+FFFD for the malformed data. Nothing in either handler touches bytes the codec decodes without error, so for a sequence the codec decodes validly both settings run the codec's ordinary decode and produce identical characters — `'replace'` is additive on the failure path only and never rewrites or re-categorises a valid decode. Grounds read from the source: `Codec.decode()` defines the argument as "the error handling to apply", defaulting to `'strict'`; `register_error()` states the handler "will be called during encoding and decoding in case of an error", with "the replacement from the error handler … put into the output directly"; the Error Handlers table differs only at the error site (`'strict'` -> raise `UnicodeError`, `'replace'` -> "Replace with a replacement marker … On decoding, use U+FFFD"). The claim's "complete sequence" qualifier removes the one genuine divergence class: a truncated multi-byte tail is *not* validly decoded, so `'strict'` raises there while `'replace'` emits U+FFFD — the error path, outside the claim. Boundaries the verdict records: `'surrogatepass'` being codec-specific (utf-8/utf-16*/utf-32*) confirms the standard handlers are error-site substitutions layered on the codec's normal decode rather than a different decoder; the binary- and text-transform codecs (`base64`, `hex`, `rot_13`, …) accept the argument but are documented as not error-sensitive for it and produce no characters, so the consequent is trivially satisfied or inapplicable there; `encodings.mbcs` is the one place the docs record `errors` having been ignored (fixed `'replace'`/`'ignore'` before 3.2, "Changed in version 3.3: Support any error handler"), Windows-only and superseded; and the verdict rests on the documented contract for stdlib codecs — a codec registered through `codecs.register()` may interpret `errors` otherwise.
Source: https://docs.python.org/3/library/codecs.html#error-handlers — "Error Handlers" (handler table, `register_error()`, `replace_errors()`), "Stateless Encoding and Decoding" (`Codec.decode`), "Encodings and Unicode", and the `encodings.mbcs` change notes. Full-page markdown consulted as fetched; no local CPython source root was supplied, so this is answered from documentation only.
Version: Python 3.14.8 documentation (docs.python.org/3, live fetch)
Fingerprint: sha256:d1bd6e38bf008aa30dd2fc596e09ae03aefe6ed9b3395fde4b36ce7d33640c3a
Invalidates-On: a CPython `Doc/library/codecs.rst` edit that makes any standard handler (or `errors=` generally) affect decoding outside an error site — e.g. `'replace'` documented as changing normalisation, byte-order/BOM sniffing, or greedy multi-byte matching for valid input
Invalidates-On: a CPython change so a text codec's non-`strict` path dispatches to a different decoder implementation than its `strict` path for valid input (the fast-path vs. stateful-path split in `Objects/unicodeobject.c`/`Objects/stringlib/codecs.h`)
Invalidates-On: applying the claim on Python < 3.3 for `encodings.mbcs`, where `errors` was ignored and `'replace'`/`'ignore'` were fixed per direction
Invalidates-On: use of a codec registered through `codecs.register()` whose `decode(input, errors)` is not stdlib code — outside the scope this source establishes
Invalidates-On: a re-fetched page whose content hash differs from the fingerprint above (fetched documentation is reusable only byte-identically)
Checked-At: 2026-10-06T23:06:00Z
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-414.md

---

Claim: CL:python/bytes-decode-non-text-codec-error
Statement: `bytes.decode(name)` raises one of `LookupError`, `UnicodeDecodeError` or `json.JSONDecodeError` when `name` is a codec registered in Python's codec registry that is not a text encoding (a bytes-to-bytes transform such as `base64`, `hex` or `rot_13`)
Subject: lang:python
Verdict: HOLDS
Actual: For a registered non-text codec the raised class is `LookupError`, with the message `'<name>' is not a text encoding; use codecs.decode() to handle arbitrary codecs` — the first member of the claimed set, so the claim holds. The failure happens at codec *lookup*, before any decoding runs: `bytes.decode` routes through `PyUnicode_FromEncodedObject` (`Objects/bytesobject.c:2331`), whose fast paths cover only the utf-8/ascii/latin-1/utf-16/utf-32 aliases, falls through to "Decode via the codec registry" -> `_PyCodec_DecodeText` (`Objects/unicodeobject.c:3615`) -> `_PyCodec_TextDecoder` -> `codec_getitem_checked(..., "codecs.decode()", 1)` -> `_PyCodec_LookupTextEncoding` (`Python/codecs.c:522`), which reads the codec's `_is_text_encoding` attribute and calls `PyErr_Format(PyExc_LookupError, ...)` when it is falsy (`Python/codecs.c:545-552`). `UnicodeDecodeError` and `json.JSONDecodeError` are not reachable on this path — they would require a decoder to run, and the registry gate rejects the codec first. Two boundaries qualify without weakening the claim: the registry classifies a codec by the `_is_text_encoding` flag itself ("Backwards compatibility: assume any raw tuple describes a text encoding, and the same for anything lacking the private attribute", `Python/codecs.c:533-537`), so a custom bytes-to-bytes codec that omits the flag is treated *as* a text encoding, is not an instance of this claim's subject, and its failure surfaces later as `TypeError` ("'x' decoder returned 'bytes' instead of 'str'", `Objects/unicodeobject.c:3619-3623`) rather than at lookup; and every stdlib bytes-to-bytes codec checked declares the flag, so each takes the `LookupError` path (`base64_codec.py:54`, `hex_codec.py:54`, `rot_13.py:45`, `quopri_codec.py:55`; verified by fetch that `bz2_codec`, `uu_codec`, `zlib_codec` likewise — exactly the "Binary Transforms" set in the codecs docs), with `hex` resolving to the `hex_codec` registry entry.
Source: Primary: `Python/codecs.c` in `_PyCodec_LookupTextEncoding` at CPython tag `v3.11.0` — https://raw.githubusercontent.com/python/cpython/v3.11.0/Python/codecs.c (lines 522-559). Corroborating chain, same tag: `Objects/bytesobject.c:2327-2332`, `Objects/unicodeobject.c:3608-3626`, `Lib/encodings/base64_codec.py`, `Lib/encodings/hex_codec.py`, `Lib/encodings/rot_13.py`, `Lib/encodings/quopri_codec.py`. Documentation corroborates: `codecs` — "Python Specific Encodings -> Binary Transforms", https://docs.python.org/3.11/library/codecs.html (bytes-to-bytes codecs are not text encodings).
Version: CPython `v3.11.0` (project declares `requires-python = ">=3.11"` in `pyproject.toml`); the same lookup gate is present unchanged at tags `v3.13.0` and `v3.14.0`
Fingerprint: sha256:6cb5b75e0a2999d910e90d2ee92df8c47868dba11dc18673932689d07a3816f1 (Python/codecs.c @ v3.11.0); sha256:3cdf1a8a5cca8b7128ff19539073184e49af304208c6a44b7fe43b9ff7ba19e5 (Objects/bytesobject.c @ v3.11.0); sha256:7f3b3b3de45ad1bfb17bb6fffb3e39f65308ae5baf648d5aaa9db8f94a87cc2a (Objects/unicodeobject.c @ v3.11.0); sha256:cf9ac7a464f541492486241d1b4bf33e37b45c6499275cc4d69c5a8e564e5976 (Lib/encodings/base64_codec.py @ v3.11.0); sha256:fc5f0a31b59efe990b86efb98936769f33dd91d912ce55b49a5a4cfc516cd047 (Lib/encodings/hex_codec.py @ v3.11.0); sha256:14767f475acdc0bf48e6272280dd15b80efaecafb93c06be21136f83dd1ee7e4 (Lib/encodings/rot_13.py @ v3.11.0)
Invalidates-On: `_PyCodec_LookupTextEncoding` in `Python/codecs.c` changes the falsy-`_is_text_encoding` branch to raise something other than `LookupError`, or the branch is removed
Invalidates-On: any of the named stdlib codecs (`base64_codec`, `hex_codec`, `rot_13`) drops or flips its `_is_text_encoding=False` registry entry
Invalidates-On: `bytes.decode` stops routing through `PyUnicode_FromEncodedObject` -> `_PyCodec_DecodeText`, or a new fast path begins accepting non-text codec names before the registry lookup
Invalidates-On: the interpreter is not CPython (e.g. PyPy), or a CPython version whose `codecs.c` bytes differ from the fingerprint above
Checked-At: 2026-10-06
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-416.md

---

Composition: CO:test-double-to-poll-loop/async-witness-eligibility
Members: CL:python/cancelled-child-await-propagation, CL:python/asyncmock-async-context-manager, CL:python/magicmock-async-context-manager, CL:python/asyncio-sleep-yields-to-loop, CL:pytest-asyncio/auto-mode-async-test-collection
Boundary: the `/metrics` response a test double hands across `async with session.get(...)` into the production poll loop, and the witness's own right to run that loop more than once — producer `repo:tests.test_compaction` (`_TickSeam` `tests/test_compaction.py:349-368`, `_kv_answer` `:371-388`, `_settle` `:391-394`, `_drive_ticks` `:397-420`, the two witnesses at `:423-433` and `:436-451`); transformations `async with` binding → the `status` gate and the awaited `text()` arm → the two arms of the function-local `first` → the interval yield → in-task cancel-then-raise → per-tick sampling; consumer `repo:buster_ripper.compaction#poll_kv_cache` (`buster_ripper/compaction.py:28-57`), spawned in production at `buster_ripper/server.py:22` via `spawn_kv_cache_monitor` (`:25-34`, called from `on_startup` at `:78`); the value `compaction._kv_cache_usage` is read by production at `buster_ripper/routes/count_tokens.py:31`, `:52`, `:59`
Dimensions: awaitable call result versus an async-context-manager object at one call site (408 against 415); the response attributes the loop reads without awaiting (`status`) versus the one it awaits (`text()`); which double the file's sibling poll tests already use, and whether that shape reaches the production branch at all; where the driven loop yields so a second tick is reached (410) against the interval the loop sleeps on; the test function's own collection and event-loop eligibility (411); cancelling and awaiting a coroutine that never returns, so a swallowed cancellation cannot read as a pass (126)
Verdict: SAFE
Counterexample: none
Source: repo (consumer side) — `buster_ripper/compaction.py:25` (`_KV_EMA_ALPHA`), `:28-57` (`poll_kv_cache`), `:31` (`first`, function-local), `:36-39` (`async with session.get(...)`), `:40` (synchronous `status` read), `:41` (awaited `text()`), `:42-44` (regex, `float`), `:45-47` and `:48-52` (the two arms), `:55-56` (the swallow), `:57` (the interval sleep, outside the `try`); `buster_ripper/config.py:13` (`UPSTREAM`), `:63` (`COMPACT_KV_RATIO`), `:64` (`KV_POLL_INTERVAL: int = 10`); `buster_ripper/server.py:15-22` (`start_kv_cache_monitoring`), `:25-34` (`spawn_kv_cache_monitor`), `:78` (`on_startup` call site), `:84-106` (`on_cleanup`/`_cancel_background_task`, consulted for the settlement shape only); `buster_ripper/routes/count_tokens.py:29-40` (`_nudge_trigger_name`), `:43-60` (`_nudge_tokens`). Repo (witness side of this seam — read as the shapes and assertion vocabulary the members are jointly asked to support, never as behavioural evidence) — `tests/test_compaction.py:183-209`, `:212-227`, `:230-258`, `:261-283`, `:288-344` (the siblings' double shape and assertion vocabulary), `:349-368` (`_TickSeam`: sample, pop, cancel-then-raise), `:371-388` (`_kv_answer`: the permitted double and its stated forbidden list), `:391-394` (`_settle`), `:397-420` (`_drive_ticks`: one answer per tick, interval mutation and restore), `:423-433` and `:436-451` (the two witnesses under composition); `tests/test_server_hooks.py:89-102` (N7 only); `tests/conftest.py` and `tests/test_pytest_config.py` (consulted for any fixture or ini interaction bearing on dims 5-6; none found); `pyproject.toml:4` (`requires-python = ">=3.11"`), `:18` (`asyncio_mode = "auto"`), `:36-42` (the `pytest-asyncio>=1.3.0` / `pytest-mock>=3.15.1` dev pins). Library — `.venv/lib/python3.13/site-packages/pytest_asyncio/plugin.py` (411's cited bytes, re-hashed), plus the bytes named by the member verdicts themselves and not re-derived here: CPython `Lib/unittest/mock.py` at `v3.11.0` / `v3.11.10` / `v3.13.1` / `v3.14.0`, CPython `Lib/asyncio/futures.py`, the `asyncio-task` docs page, and the `unittest.mock` docs page. Versions: repo `buster_ripper/*` and `tests/*` are working-tree bytes, hashed in `Fingerprint:`; the interpreter running the seam's code is CPython 3.13.8 (`uv` managed, `.venv/pyvenv.cfg` `version_info = 3.13.8`); `pytest-asyncio` 1.3.0, `pytest-mock` (`>=3.15.1` declared), `pytest-aiohttp` 1.1.0, `aiohttp` 3.13.3; member consulted sources as recorded in each verdict — 126 "Python 3.14.7", 408 CPython `v3.11.10` (confirmed at `v3.14.0`), 415 CPython `v3.11.0` (corroborated at `v3.13.1`), 410 Python 3.14.8 docs (corroborated at 3.11.17), 411 the local plugin (896 lines); the repo Python floor `>=3.11` is covered by every mock-mechanism member's tag range. Nothing was executed; no test, fixture, snapshot or mock run was used as evidence for a behavioural claim, and every behavioural statement comes from a `HOLD` member verdict or from the production sources listed.
Fingerprint: sha256:b54ad7c8b85008a6182721d425b872e77ef26a28d0ee812ac62b1f4c85ca24cd (handle-bearing — the byte-exact recipe, so the digest is reproducible: material in this order, one item per line as `<handle> <fp> [<fp>…]` then `<sha256> <path>`, joined by single newlines with a trailing newline, prefixed by the two header lines `S19 members (ledger order) with verdict-recorded fingerprints:` and `consulted source bytes:`. Members: 1. `CL:python/cancelled-child-await-propagation` `2f482524292a7c5d…`; 2. `CL:python/asyncmock-async-context-manager` `2576119e5653ca21…`, `667d9ed88fd330a3…`; 3. `CL:python/magicmock-async-context-manager` `663efa5704c1146a…`, `7c6b6c92b11243eb…`, `6513ae8d94214d97…`; 4. `CL:python/asyncio-sleep-yields-to-loop` `0e10a7cd1a7a3a44…`, `2f85eef501369e2d…`; 5. `CL:pytest-asyncio/auto-mode-async-test-collection` `121cb98d74596a9e…`. Consulted source bytes: `ac9d42143c98694a…` `buster_ripper/compaction.py` · `22348c68d21b91a3…` `buster_ripper/config.py` · `b3b486615ca797fe…` `buster_ripper/server.py` · `195f3b24e327f57d…` `buster_ripper/routes/count_tokens.py` · `ae79036b7fe714c2…` `tests/test_compaction.py` · `2dbbe44dc3e0949d…` `tests/test_server_hooks.py` · `e821283c228bd63a…` `tests/conftest.py` · `c623bdc282433193…` `pyproject.toml` · `121cb98d74596a9e…` `.venv/lib/python3.13/site-packages/pytest_asyncio/plugin.py`. Delta against the prior record's basis `990d0278e86f444d…`: `compaction.py`, `config.py`, `conftest.py`, `pyproject.toml` and the `pytest-asyncio` plugin are byte-identical to the prior pins; `test_compaction.py`, `server.py` and `count_tokens.py` moved)
Invalidates-On: any member leaves `HOLD` or is re-extracted — including retired row 409 being re-opened, which may not re-enter this seam as an alternate premise (N8: `CL:python/magicmock-async-context-manager` is cited here for 415's fresh `HOLDS` verdict only; 409's terminal record is evidence history, not a premise)
Invalidates-On: `buster_ripper/compaction.py` moving off `ac9d42143c98694a…` — in particular `:36-39` (the `async with` form, or the call result it consumes), `:40` (a sync-versus-await change to the `status` read, or a truthiness rather than `== 200` test), `:41` (`text()` awaited versus `.text` read), `:31`/`:45-52` (`first` becoming module state, which would make a *second call* a second tick), `:55-56` (narrowing or removing the `except Exception`, which converts the silent-swallow routes in N1-N4 from loud to differently loud and makes `:366`'s terminating `ClientError` escape the loop), or `:57` (the sleep moving inside the `try`, or the interval becoming a parameter or a once-per-call local read before the loop, which kills the witness's mutation route at `:413`)
Invalidates-On: `buster_ripper/config.py`'s `KV_POLL_INTERVAL` moving off `22348c68d21b91a3…`, or its kind changing (a `float` or `0` default, or per-call injection). The distinction this record draws: the production default is still `int = 10`, and the `0` composed against is a test-side mutation with a `finally` restore (`tests/test_compaction.py:413`, `:419`), so a real default change re-opens CE-3 from the other side and must not be read as already settled here
Invalidates-On: `buster_ripper/routes/count_tokens.py` ceasing to read `compaction._kv_cache_usage` at `:31`/`:52`/`:59`, or `buster_ripper/server.py:22` / `:25-34` / `:78` ceasing to spawn `poll_kv_cache` — either removes the production reach that makes the three closures findings rather than inert test weaknesses; the lines are re-pinned to the post-T018/T019 bytes and any further shift re-locates them
Invalidates-On: `tests/test_compaction.py` moving off `ae79036b7fe714c2…` in the witness block `:349-451` — specifically `:386-387` (the self-binding that closes CE-1), `:383`/`:384-385` (the sync `status` / awaited `text` qualification that closes CE-2), `:364`/`:420` (the per-tick sample and the `[1:]` windowing that make the literals mean what they say), `:413`/`:419` (the interval mutation and its restore), `:366-367` (the in-task cancel-then-raise that terminates the run), `:393-394` (`wait_for`'s 2.0 s bound or the `suppress` target — widening the suppress to `Exception` or `BaseException` would let a stuck loop read as a pass), or `:433`/`:450` (the awaited-channel assertions, the CE-1/CE-2 tripwire). The four payload siblings at `:183-344` bear on dim 3 and N5 only
Invalidates-On: `pyproject.toml` off `c623bdc282433193…` at `:18` (`asyncio_mode`) or `:36-42` (the async/mock dev pins), or `.venv/lib/python3.13/site-packages/pytest_asyncio/plugin.py` off `121cb98d…` (dim 5)
Invalidates-On: any of the library fingerprints in 408 / 415 / 410 / 126 no longer resolving for the interpreter the suite runs on, or this venv moving off CPython 3.13.8 — per N9, 411 alone is pinned to this worktree's installed bytes; 408, 415, 410 and 126 are pinned to bytes outside it (CPython tags, docs pages, a recorded `Lib/asyncio/futures.py` digest), and this interpreter's `Lib/unittest/mock.py` (`b4164040…`) and `Lib/asyncio/futures.py` (`f55a9918…`) match neither
Invalidates-On: any future pass rowing one of the unrowed atoms N1-N4 — the verdict is currently SAFE because each of those atoms fails *loud*; a row that showed one failing quietly reopens the corresponding CE (N1 `CancelledError` is not an `Exception`, so `compaction.py:55-56` cannot swallow it; N2 a task yielding at a zero-delay sleep is resumed by the running loop so its own next iteration happens; N3 a cancellation requested while the task is running is delivered at its next suspension point, plus `asyncio.wait_for`'s cancel-and-await on timeout; N4 `Mock(side_effect=<callable>)` returns that callable's return value and a `spec=`'d session still allows `session.get` to be rebound)
Invalidates-On: boundary wording — any seam-row edit to S19's boundary value, subject list or dimension list; this record traces the six dimensions the seam row names and claims nothing beyond them
Checked-At: 2026-10-07T01:52:17Z
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/compositions/S19-async-witness.md

---

Claim: CL:aiohttp/web-response-str-body-acceptance
Statement: Constructing aiohttp's `web.Response` with a `str` `body=` argument raises no exception in `__init__` or in the `body` setter
Subject: pkg:aiohttp@3.13.3
Verdict: HOLDS
Actual: `Response.__init__` performs no type validation on `body` — its only guard is `body is not None and text is not None` — and delegates to the `body` setter, which routes a `str` (neither `None` nor `bytes`/`bytearray`) through `payload.PAYLOAD_REGISTRY.get(body)`; `str` is a registered type (`PAYLOAD_REGISTRY.register(StringPayload, str)`), so no `LookupError` is raised and the setter's `ValueError("Unsupported body type ...")` branch is never reached. Construction therefore succeeds. The exception path that remains available is value-specific, not type-specific: `StringPayload.__init__` calls `value.encode(real_encoding)` (default `utf-8`), so a `str` containing an unpaired surrogate (e.g. `"\ud800"`) raises `UnicodeEncodeError` out of `__init__`; a `str` above `TOO_LARGE_BYTES_BODY` triggers a `ResourceWarning`, not an exception.
Source: `.venv/lib/python3.13/site-packages/aiohttp/web_response.py` — `Response.__init__` lines 628-691 (`self.body = body` at line 688) and `Response.body` setter lines 697-721. Corroborated in the same package by `payload.py:1111` (`PAYLOAD_REGISTRY.register(StringPayload, str)`), `PayloadRegistry.get` lines 110-131 (fast `_normal_lookup` by exact type, `raise LookupError()` only when nothing matches), and `StringPayload.__init__` lines 430-458. Installed source is the ground: no documentation ships in the wheel.
Version: aiohttp 3.13.3 (`aiohttp/__init__.py:1` — `__version__ = "3.13.3"`, installed at `.venv/lib/python3.13/site-packages/aiohttp`, Python 3.13)
Fingerprint: sha256:3ca73388d5382e67ed5ea295be844cac06ce7950a5a5237e8b39c7b2259ece65 (aiohttp/web_response.py)
Fingerprint: sha256:3ba9ec60d50b2fb01e33673227a4d85fbde77159cbe71270b7901e831c0c2aac (aiohttp/payload.py)
Invalidates-On: the `Response.body` setter in `web_response.py` changes the `str` branch (adds an `isinstance(body, str)` rejection, or routes `str` off the payload registry)
Invalidates-On: `PAYLOAD_REGISTRY.register(StringPayload, str)` is removed from `payload.py`, or `str` is re-registered to a factory that raises
Invalidates-On: `PayloadRegistry.get` gains a path that raises `LookupError` for an exactly-registered type
Invalidates-On: an aiohttp upgrade whose `web_response.py`/`payload.py` bytes differ from the fingerprints above
Checked-At: 2026-10-06
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-417.md

---

Claim: CL:aiohttp/web-response-body-getter-type
Statement: For an aiohttp `web.Response` constructed with a `str` `body=` argument, the `body` getter returns a `bytes` value rather than a payload object
Subject: pkg:aiohttp@3.13.3
Verdict: DOES-NOT-HOLD
Actual: The getter returns a **payload object**, not bytes. `Response.body`'s getter is a bare `return self._body`, and the setter's `str` branch stores the registry result directly: `self._body = body = payload.PAYLOAD_REGISTRY.get(body)`. For a `str` that result is a `StringPayload` instance (`PAYLOAD_REGISTRY.register(StringPayload, str)`), which subclasses `BytesPayload` → `Payload`, so `isinstance(resp.body, bytes)` is `False`; the getter's own annotation says as much (`-> Optional[Union[bytes, Payload]]`), and downstream code in `Response` branches on `isinstance(self._body, Payload)` rather than treating the value as bytes. Only values that were already `bytes`/`bytearray` are stored raw (setter lines 701-702). The encoded bytes are reachable as `resp.body.decode()` (a `str`), `await resp.body.as_bytes()` (`bytes`), or the private `resp.body._value`; the byte count is `resp.body.size`.
Source: `.venv/lib/python3.13/site-packages/aiohttp/web_response.py` — `Response.body` getter lines 693-695 and setter lines 697-721 (`self._body = body = payload.PAYLOAD_REGISTRY.get(body)` at line 705). Corroborated in the same package by `payload.py:1111` (registry binding `str` → `StringPayload`), `class StringPayload(BytesPayload)` line 430, `class BytesPayload(Payload)` line 350 with `decode` lines 380-381 and `as_bytes` lines 383-390.
Version: aiohttp 3.13.3 (`aiohttp/__init__.py:1` — `__version__ = "3.13.3"`, installed at `.venv/lib/python3.13/site-packages/aiohttp`, Python 3.13)
Fingerprint: sha256:3ca73388d5382e67ed5ea295be844cac06ce7950a5a5237e8b39c7b2259ece65 (aiohttp/web_response.py)
Fingerprint: sha256:3ba9ec60d50b2fb01e33673227a4d85fbde77159cbe71270b7901e831c0c2aac (aiohttp/payload.py)
Invalidates-On: the `Response.body` setter starts normalising `str` to `text.encode(...)` bytes before storing (the `isinstance(body, (bytes, bytearray))` branch widened, or an explicit `str` branch added)
Invalidates-On: `PAYLOAD_REGISTRY` stops mapping `str` to a `Payload` subclass, or `PayloadRegistry.get` is changed to return encoded bytes
Invalidates-On: the `Response.body` getter is changed to unwrap or serialise `_body` instead of returning it verbatim
Invalidates-On: an aiohttp upgrade whose `web_response.py`/`payload.py` bytes differ from the fingerprints above
Checked-At: 2026-10-06
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-418.md

---

Claim: CL:aiohttp/web-response-str-body-wire-bytes
Statement: An aiohttp `web.Response` built with a `str` `body=` argument sends the encoded text of that string as its body over the wire
Subject: pkg:aiohttp@3.13.3
Verdict: HOLDS
Actual: `Response.__init__` with `text=None` assigns through the `body` setter, which for a `str` (not `bytes`/`bytearray`) calls `payload.PAYLOAD_REGISTRY.get(body)`; `str` is registered to `StringPayload`, so `self._body` becomes a `StringPayload` whose constructor immediately replaces the string with `value.encode(real_encoding)` (utf-8, since the setter passes no `encoding`/`content_type`) and stores it as `_value`. At send time `Response.write_eof` takes the `isinstance(self._body, Payload)` branch and calls `self._body.write(self._payload_writer)`, which is `BytesPayload.write` → `await writer.write(self._value)` — the encoded bytes of the original string, written verbatim. The payload object is never serialized; it supplies the bytes plus `Content-Type: text/plain; charset=utf-8` (copied in the setter when no Content-Type is set) and `Content-Length` from `payload.size` in `Response._start`. Caveats read out of the same source, none reversing the verdict: encoding for the plain `Response(body=s)` path is always utf-8 regardless of a caller-supplied `content_type=`/`Content-Type`, so a declared non-utf-8 charset mislabels utf-8 bytes but still does not serialize the payload object; and with compression enabled, `_do_start_compression` replaces the transmitted bytes with the deflate/gzip form of `self._body` — a transport encoding of the same text, not a payload serialization.
Source: `.venv/lib/python3.13/site-packages/aiohttp/web_response.py` — `Response.__init__` lines 628-691, `body` setter lines 697-721, `write_eof` lines 769-786, `_start` lines 788-804, `_do_start_compression` lines 806-830. `.venv/lib/python3.13/site-packages/aiohttp/payload.py` — `PAYLOAD_REGISTRY.register(StringPayload, str)` line 1111, `PayloadRegistry.get` lines 110-131, `StringPayload.__init__` lines 430-458, `BytesPayload.write` lines 392-407.
Version: aiohttp 3.13.3 (`__version__` in `.venv/lib/python3.13/site-packages/aiohttp/__init__.py`; installed source, no bundled documentation)
Fingerprint: sha256:3ca73388d5382e67ed5ea295be844cac06ce7950a5a5237e8b39c7b2259ece65 (aiohttp/web_response.py)
Fingerprint: sha256:3ba9ec60d50b2fb01e33673227a4d85fbde77159cbe71270b7901e831c0c2aac (aiohttp/payload.py)
Invalidates-On: `web_response.py` bytes change such that the `body` setter re-gains a `str` rejection (`TypeError: body argument must be bytes`) or stops routing through `PAYLOAD_REGISTRY`
Invalidates-On: `payload.py` bytes change such that `str` is no longer registered to `StringPayload`, or `StringPayload` stops encoding eagerly, or `BytesPayload.write` stops writing `self._value`
Invalidates-On: an aiohttp release where `Response.write_eof` sends payloads via `write_with_length` or another writer path that frames the body differently
Invalidates-On: rechecking against a different installed copy of aiohttp than the fingerprinted site-packages root above
Checked-At: 2026-10-06
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-420.md

---

Claim: CL:aiohttp/web-response-str-body-text-property
Statement: The `text` property of an aiohttp `web.Response` built with a `str` `body=` argument returns that string
Subject: pkg:aiohttp@3.13.3
Verdict: HOLDS
Actual: `Response.text` is `self._body.decode(self.charset or "utf-8")` when `_body` is not `None`. For a `str` given to `body=`, `_body` is a `StringPayload` (a `BytesPayload`), and `BytesPayload.decode` is `self._value.decode(encoding, errors)` where `_value` is the original string encoded to utf-8 by `StringPayload.__init__`. `charset` comes from `HeadersMixin.charset`, which parses the `Content-Type` header; the `body` setter copies `StringPayload.content_type` into that header as `text/plain; charset=utf-8` whenever the caller did not set one, so the decode uses utf-8 and yields a string equal to the one passed in. Read precisely: `.text` returns a newly decoded, value-equal `str`, not the same object the caller passed — the `str` itself is not retained anywhere (`_value` holds bytes). Second caveat, same formula: if the caller supplies a `Content-Type`/`content_type` naming a non-utf-8 charset, `body=` still encoded utf-8 while `.text` decodes with the declared charset, so `.text` can raise `UnicodeDecodeError` or return mojibake — a header-vs-payload mismatch the setter does not reconcile, not a different `.text` contract.
Source: `.venv/lib/python3.13/site-packages/aiohttp/web_response.py` — `text` getter lines 723-730, `body` setter lines 697-721, `charset` property lines 333-336. `.venv/lib/python3.13/site-packages/aiohttp/payload.py` — `StringPayload.__init__` lines 430-458, `BytesPayload.decode` lines 380-381. `.venv/lib/python3.13/site-packages/aiohttp/helpers.py` — `HeadersMixin.charset` lines 769-776.
Version: aiohttp 3.13.3 (`__version__` in `.venv/lib/python3.13/site-packages/aiohttp/__init__.py`; installed source, no bundled documentation)
Fingerprint: sha256:3ca73388d5382e67ed5ea295be844cac06ce7950a5a5237e8b39c7b2259ece65 (aiohttp/web_response.py)
Fingerprint: sha256:3ba9ec60d50b2fb01e33673227a4d85fbde77159cbe71270b7901e831c0c2aac (aiohttp/payload.py)
Fingerprint: sha256:435df4ecf0849d6cf844ff1cad4c3c764e7c734605d848b7272c2429f471cf91 (aiohttp/helpers.py)
Invalidates-On: `web_response.py` bytes change such that `text` reverts to returning a stored `_text` attribute (set only by the `text=` kwarg) instead of decoding `_body`
Invalidates-On: `payload.py` bytes change such that `StringPayload` stores the `str` rather than its encoded bytes, or `BytesPayload.decode` stops delegating to `self._value.decode`
Invalidates-On: `helpers.py`'s `HeadersMixin.charset` changes its default/None handling so `.text` decodes with a codec other than the one `StringPayload` encoded with
Invalidates-On: a release where the `body` setter stops copying `StringPayload.content_type` into `Content-Type` (charset inference for `.text` would change)
Checked-At: 2026-10-06
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-421.md

---

Claim: CL:aiohttp/web-response-str-body-accessor-surface
Statement: aiohttp's `web.Response` class exposes no accessor that returns the `str` passed to its `body=` initializer
Subject: pkg:aiohttp@3.13.3
Verdict: DOES-NOT-HOLD
Actual: `Response.text` is such an accessor, so the universal negative fails. It is a property defined on `Response` itself (lines 723-730) whose getter is `return self._body.decode(self.charset or "utf-8")` when `_body is not None`. A `str` passed to `body=` is stored by the setter as a `StringPayload` (`str` is registered to that factory at `payload.py:1111`) whose constructor encodes the string to utf-8 and whose `decode` (`BytesPayload.decode`) decodes those bytes with the charset `HeadersMixin.charset` parses out of the `Content-Type` header — the setter having set that header to `text/plain; charset=utf-8` — so `.text` hands back exactly the string that was passed. The full accessor surface was enumerated: own — `body`, `text`, `content_length`, `content_type`, `charset`, `last_modified`, `etag`; inherited from `StreamResponse` — `prepared`, `task`, `status`, `chunked`, `compression`, `reason`, `keep_alive`, `body_length`, `output_length`, `headers`, `cookies`; plus `MutableMapping.__getitem__`, which returns only values the handler itself stored in `_state`. `text` is the only one returning string content; `body` returns the `StringPayload` object, not the `str` (`resp.body.decode()` would give the string, but that is an accessor on `Payload`, not on `Response`). Stated so the verdict is auditable rather than quietly definitional: the string is not *retained* as an object — `_value` holds bytes — so `.text` returns a freshly decoded, value-equal `str`; only a strict object-identity reading of "returns the `str` value" would let the claim survive, and that is not the plain meaning of the sentence.
Source: `.venv/lib/python3.13/site-packages/aiohttp/web_response.py` — `text` getter lines 723-730, `body` setter lines 697-721, `body` getter lines 693-695, `charset` property lines 333-336. `.venv/lib/python3.13/site-packages/aiohttp/payload.py` — `PAYLOAD_REGISTRY.register(StringPayload, str)` line 1111, `StringPayload.__init__` lines 430-458, `BytesPayload.decode` lines 380-381. `.venv/lib/python3.13/site-packages/aiohttp/helpers.py` — `HeadersMixin.charset` lines 769-776.
Version: aiohttp 3.13.3 (`__version__` in `.venv/lib/python3.13/site-packages/aiohttp/__init__.py`; installed source, no bundled documentation)
Fingerprint: sha256:3ca73388d5382e67ed5ea295be844cac06ce7950a5a5237e8b39c7b2259ece65 (aiohttp/web_response.py)
Fingerprint: sha256:3ba9ec60d50b2fb01e33673227a4d85fbde77159cbe71270b7901e831c0c2aac (aiohttp/payload.py)
Fingerprint: sha256:435df4ecf0849d6cf844ff1cad4c3c764e7c734605d848b7272c2429f471cf91 (aiohttp/helpers.py)
Invalidates-On: `web_response.py` bytes change such that `Response.text` stops decoding `_body` (e.g. reverts to a `_text` attribute set only via the `text=` kwarg) — the counterexample would disappear and the claim could become true
Invalidates-On: `payload.py` bytes change such that `str` is unregistered from `StringPayload`, `StringPayload` stops encoding eagerly, or `BytesPayload.decode` is removed or changed
Invalidates-On: `helpers.py`'s `HeadersMixin.charset` changes so the decode codec no longer matches the encode codec for the default `text/plain; charset=utf-8`
Invalidates-On: a release adding or removing `Response` properties, which requires re-running the accessor enumeration
Checked-At: 2026-10-06
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-422.md

---

Claim: CL:aiohttp/string-payload-str-representation
Statement: `str()` of an `aiohttp.payload.StringPayload` instance returns the text it was created with, not the default object repr
Subject: pkg:aiohttp@3.13.3
Verdict: DOES-NOT-HOLD
Actual: `str()` on a `StringPayload` yields Python's default object representation — e.g. `<aiohttp.payload.StringPayload object at 0x7f...>` — not the construction text. Neither `StringPayload` nor any class in its MRO (`StringPayload` → `BytesPayload` → `Payload(ABC)` → `object`) defines `__str__` or `__repr__`, so `str()` falls through to `object.__str__`, which delegates to `object.__repr__`. The constructed text is not kept as a string at all: `StringPayload.__init__` encodes it (`value.encode(real_encoding)`) and passes the bytes to `BytesPayload`, which stores them in `self._value`. The text is recoverable only via the synchronous `BytesPayload.decode(encoding="utf-8", errors="strict")` (`return self._value.decode(encoding, errors)`), the async `await payload.as_bytes()` (bytes, not text), or direct `_value` access (bytes) — none of which `str()` touches.
Source: `.venv/lib/python3.13/site-packages/aiohttp/payload.py` — `class Payload(ABC)` line 151, `class BytesPayload(Payload)` line 350 with `decode` line 380 and `as_bytes` line 383, `class StringPayload(BytesPayload)` lines 430-458; the file contains no `def __str__` and no `def __repr__` (grep over `payload.py`, `formdata.py`, `multipart.py` returns none). Exported as `aiohttp.StringPayload` via `aiohttp/__init__.py:88` and `payload.py:47`.
Version: aiohttp 3.13.3 (`aiohttp/__init__.py:1` — `__version__ = "3.13.3"`)
Fingerprint: sha256:3ba9ec60d50b2fb01e33673227a4d85fbde77159cbe71270b7901e831c0c2aac (aiohttp/payload.py, full file bytes)
Invalidates-On: an aiohttp release that adds `__str__` or `__repr__` to `Payload`, `BytesPayload`, or `StringPayload` in `payload.py`
Invalidates-On: a change to `StringPayload.__init__` that stores the original `str` value instead of only the encoded bytes (this changes what `decode()`/`_value` return, not `str()`)
Invalidates-On: this installed package being replaced by a different aiohttp version whose `payload.py` bytes differ from the fingerprint above
Invalidates-On: a repo-local subclass of `StringPayload` overriding `__str__` — that would make the behaviour true for the subclass only, never for `aiohttp.StringPayload` itself
Checked-At: 2026-10-07T02:15:21Z
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-423.md

---

Claim: CL:python/str-fallback-mechanism
Statement: Python's `str()` on an object whose type defines no `__str__` falls back to that object's `__repr__`
Subject: lang:python
Verdict: HOLDS
Actual: `str(object)` calls `type(object).__str__(object)`; when the class defines no `__str__`, the inherited default implementation from the built-in `object` type calls `object.__repr__()`, which dispatches to the class's own `__repr__`. The documentation states the fallback directly: "If object does not have a `__str__()` method, then `str()` falls back to returning `repr(object)`." The mechanism is documented from both sides — the `str()` entry (stdtypes) and the data-model entry (`object.__str__` — "The default implementation defined by the built-in type object calls object.`__repr__()`"; `object.__repr__` — "If a class defines `__repr__()` but not `__str__()`, then `__repr__()` is also used when an 'informal' string representation of instances of that class is required"). Documentation-only claim about CPython's language/stdlib behaviour: no repo source and no execution were consulted, and no test artifact was used.
Source: https://docs.python.org/3/library/stdtypes.html#str (bytes saved locally as `stdtypes.html`; resolved title "Built-in Types — Python 3.14.8 documentation") — establishing source for the fallback wording. Corroborated by https://docs.python.org/3/reference/datamodel.html#object.__str__ and ...#object.__repr__ (saved as `datamodel.html`, same 3.14.8 build) for the `object.__str__` → `object.__repr__` delegation.
Version: Python 3.14.8 documentation (docs.python.org/3 stable release branch)
Fingerprint: sha256:bf149b1649601f39dfcbb435a1c1650e2607fe9d5f315fc93ea8432b28ebae6b (stdtypes.html, 3.14.8, fetched raw bytes)
Fingerprint: sha256:f057eb26e5f008e87d25fecace3f88ca29b46776007687627ddb26a2dd2e08ba (datamodel.html, 3.14.8, fetched raw bytes)
Invalidates-On: the `str()` entry in stdtypes.html changes the sentence "If object does not have a `__str__()` method, then `str()` falls back to returning `repr(object)`", or the `object.__str__` entry in datamodel.html stops saying the default implementation calls `object.__repr__()` (re-fingerprint either page)
Invalidates-On: a Python version whose data model gives `object` a `__str__` that does not delegate to `__repr__`, or that makes `str()` raise `TypeError` / return `object.__repr__`-independent output for classes without `__str__` (a different docs branch than the 3.14.8 pages fingerprinted here, or any 2.x-era semantics where old-style classes behaved otherwise)
Invalidates-On: a target runtime that is not CPython-3 semantics (an alternative interpreter), or a class that inherits `__str__` from a non-`object` base — that inherited method wins, so no `__repr__` fallback occurs
Checked-At: 2026-10-06
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-424.md

---

```text
Claim: CL:python/unified-diff-content-line-prefixes
Statement: `difflib.unified_diff` prefixes each removed content line with `-` and each added content line with `+`
Subject: lang:python
Verdict: HOLDS
Actual: `unified_diff` emits one-character prefixes on content lines: lines from `a` in a `replace`/`delete` opcode are yielded as `'-' + line`, and lines from `b` in a `replace`/`insert` opcode are yielded as `'+' + line`. The documentation's worked example shows exactly this (`-bacon`, `-eggs`, `-ham` then `+python`, `+eggy`, `+hamster`).
Source: https://docs.python.org/3/library/difflib.html (difflib.unified_diff section, worked example), corroborated by the implementation at https://raw.githubusercontent.com/python/cpython/3.13/Lib/difflib.py lines 1151-1161: `if tag in {'replace', 'delete'}: yield '-' + line`; `if tag in {'replace', 'insert'}: yield '+' + line`
Version: Python 3.15.0 documentation; CPython 3.13 branch source (`Lib/difflib.py`), fetched 2026-10-17
Fingerprint: sha256:48a2142e8b2fedece2c2c8b59376b5ee76599c6539dc94a308f9a90ffc5ab2c9 (docs page as fetched); sha256:1593ae08143632f4f81fea3beeeb279d803f8a9be922b12fe79664d78bba879d (Lib/difflib.py as fetched)
Invalidates-On: `unified_diff` stops yielding `'-' + line` / `'+' + line` for `replace`/`delete`/`insert` opcodes
Invalidates-On: the documented unified_diff example output changes its content-line prefixes
Invalidates-On: either consulted source file changes bytes (fingerprint mismatch)
Checked-At: 2026-10-17T23:41:03Z
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-426.md
```

---

```text
Claim: CL:python/unified-diff-context-line-prefix
Statement: `difflib.unified_diff` prefixes each unchanged context line with a single space
Subject: lang:python
Verdict: HOLDS
Actual: For an `equal` opcode, `unified_diff` yields each line as `' ' + line` — exactly one leading space, no other marker. The documentation's worked example shows the unchanged line as ` guido` (leading space), and the implementation yields `' ' + line` in the `tag == 'equal'` branch.
Source: https://docs.python.org/3/library/difflib.html (difflib.unified_diff section, worked example), corroborated by the implementation at https://raw.githubusercontent.com/python/cpython/3.13/Lib/difflib.py lines 1151-1155: `if tag == 'equal': yield ' ' + line`
Version: Python 3.15.0 documentation; CPython 3.13 branch source (`Lib/difflib.py`), fetched 2026-10-17
Fingerprint: sha256:48a2142e8b2fedece2c2c8b59376b5ee76599c6539dc94a308f9a90ffc5ab2c9 (docs page as fetched); sha256:1593ae08143632f4f81fea3beeeb279d803f8a9be922b12fe79664d78bba879d (Lib/difflib.py as fetched)
Invalidates-On: `unified_diff` stops yielding `' ' + line` for `equal` opcodes
Invalidates-On: the documented unified_diff example output changes the context-line prefix
Invalidates-On: either consulted source file changes bytes (fingerprint mismatch)
Checked-At: 2026-10-17T23:41:03Z
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-427.md
```

---

```text
Claim: CL:python/unified-diff-header-line-prefixes
Statement: `difflib.unified_diff` prefixes its file-header lines with `---`/`+++` and its hunk-header lines with `@@`
Subject: lang:python
Verdict: HOLDS
Actual: On the first group, `unified_diff` yields `'--- {fromfile}{fromdate}{lineterm}'` and `'+++ {tofile}{todate}{lineterm}'` for the two file headers, and for every group yields `'@@ -{range1} +{range2} @@{lineterm}'` for the hunk header. The documentation describes these as "the diff control lines (those with `---`, `+++`, or `@@`)" and its worked example shows `--- before.py`, `+++ after.py`, `@@ -1,4 +1,4 @@`.
Source: https://docs.python.org/3/library/difflib.html (difflib.unified_diff section, worked example), corroborated by the implementation at https://raw.githubusercontent.com/python/cpython/3.13/Lib/difflib.py lines 1138-1149: `yield '--- {}{}{}'.format(fromfile, fromdate, lineterm)`; `yield '+++ {}{}{}'.format(tofile, todate, lineterm)`; `yield '@@ -{} +{} @@{}'.format(file1_range, file2_range, lineterm)`
Version: Python 3.15.0 documentation; CPython 3.13 branch source (`Lib/difflib.py`), fetched 2026-10-17
Fingerprint: sha256:48a2142e8b2fedece2c2c8b59376b5ee76599c6539dc94a308f9a90ffc5ab2c9 (docs page as fetched); sha256:1593ae08143632f4f81fea3beeeb279d803f8a9be922b12fe79664d78bba879d (Lib/difflib.py as fetched)
Invalidates-On: `unified_diff` stops yielding `---`/`+++` file-header lines or `@@` hunk-header lines
Invalidates-On: the documented unified_diff example output changes its header prefixes
Invalidates-On: either consulted source file changes bytes (fingerprint mismatch)
Checked-At: 2026-10-17T23:41:03Z
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-428.md
```

---

```text
Claim: CL:python/unified-diff-lineterm-placement
Statement: `difflib.unified_diff` appends its `lineterm` value to each file-header and hunk-header line, not to content lines
Subject: lang:python
Verdict: HOLDS
Actual: `unified_diff` yields the two file headers as `'--- {}{}{}'.format(fromfile, fromdate, lineterm)` and `'+++ {}{}{}'.format(tofile, todate, lineterm)`, and the hunk header as `'@@ -{} +{} @@{}'.format(file1_range, file2_range, lineterm)`, so `lineterm` terminates every control line. Content lines are yielded as `' ' + line`, `'-' + line`, or `'+' + line` with no `lineterm` appended; their line endings come only from the input strings themselves. This matches the documentation: "the diff control lines (those with ---, +++, or @@) are created with a trailing newline", and setting `lineterm=''` produces uniformly newline-free output.
Source: https://raw.githubusercontent.com/python/cpython/3.11/Lib/difflib.py — `unified_diff`, lines 1095-1161 (header formatting at 1143-1149; content lines at 1152-1161). Corroborated by https://docs.python.org/3/library/difflib.html#difflib.unified_diff
Version: CPython 3.11 branch `Lib/difflib.py`; project declares `requires-python = ">=3.11"`. The same control-line formatting is present in the CPython 3.15 branch (lines 1101-1176), so the behavior is unchanged across the supported range checked.
Fingerprint: sha256:0c6afc23568d55b3e9ac914f9c5361e3033e778aa5b58d3cc82835fc5c638679 (CPython 3.11 `Lib/difflib.py`); cross-checked 3.15 `Lib/difflib.py` sha256:aaa3d1ccdb8d8f0d2e8c99fe1a320a983b770fad94779f1dd417cdd37c033159
Invalidates-On: `unified_diff` stops appending `lineterm` to the `---`/`+++`/`@@` lines in `Lib/difflib.py`
Invalidates-On: `unified_diff` starts appending `lineterm` to context/added/removed content lines
Invalidates-On: project minimum Python moves below 3.11 or to a version whose `difflib.unified_diff` differs on this point
Checked-At: 2026-10-17
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-429.md
```

---

```text
Claim: CL:python/unified-diff-context-line-count
Statement: `difflib.unified_diff`'s `n` argument sets the number of unchanged context lines around each hunk and defaults to 3
Subject: lang:python
Verdict: HOLDS
Actual: `unified_diff` has signature `unified_diff(a, b, fromfile='', tofile='', fromfiledate='', tofiledate='', n=3, lineterm='\n', *, color=False)` and calls `SequenceMatcher(None, a, b).get_grouped_opcodes(n)`. `SequenceMatcher.get_grouped_opcodes(n=3)` trims leading and trailing equal runs to at most `n` lines and splits long equal runs so each hunk carries at most `n` unchanged context lines on each side. The documentation states: "The number of context lines is set by *n* which defaults to three."
Source: https://raw.githubusercontent.com/python/cpython/3.11/Lib/difflib.py — `unified_diff` signature and call at lines 1095-1138; `get_grouped_opcodes(self, n=3)` at lines 547-592. Corroborated by https://docs.python.org/3/library/difflib.html#difflib.unified_diff
Version: CPython 3.11 branch `Lib/difflib.py`; project declares `requires-python = ">=3.11"`. The same signature and grouping call are present in the CPython 3.15 branch (lines 1101-1153), so the behavior is unchanged across the supported range checked.
Fingerprint: sha256:0c6afc23568d55b3e9ac914f9c5361e3033e778aa5b58d3cc82835fc5c638679 (CPython 3.11 `Lib/difflib.py`); cross-checked 3.15 `Lib/difflib.py` sha256:aaa3d1ccdb8d8f0d2e8c99fe1a320a983b770fad94779f1dd417cdd37c033159
Invalidates-On: the `n` default in `unified_diff` or `SequenceMatcher.get_grouped_opcodes` changes from 3
Invalidates-On: `unified_diff` stops passing `n` to `get_grouped_opcodes`, or `get_grouped_opcodes` stops using `n` as the context-line limit
Invalidates-On: project minimum Python moves below 3.11 or to a version whose `difflib.unified_diff` differs on this point
Checked-At: 2026-10-17
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-430.md
```

---

```text
Claim: CL:python/unified-diff-line-equality
Statement: `difflib.unified_diff` compares input lines for exact string equality, without whitespace or case normalization
Subject: lang:python
Verdict: HOLDS
Actual: `unified_diff` builds `SequenceMatcher(None, a, b)` — `isjunk=None`, so no lines are filtered as junk — and `SequenceMatcher` groups elements through a dict (`b2j.setdefault(elt, [])`) whose keys use the elements' own hash/equality; for `str` lines that is exact string equality. No whitespace stripping, case folding, or other normalization is applied anywhere in `unified_diff`; the input strings are emitted verbatim with only a one-character prefix. (Matching is longest-contiguous-block based and the default `autojunk` heuristic can leave some identical lines unaligned in sequences of 200+ items, but the equality predicate itself remains exact.)
Source: https://raw.githubusercontent.com/python/cpython/3.11/Lib/difflib.py — `unified_diff` builds `SequenceMatcher(None, a, b)` at line 1138; element grouping in `SequenceMatcher.__chain_b` at lines 266-302; verbatim content emission at lines 1152-1161. Corroborated by https://docs.python.org/3/library/difflib.html#difflib.SequenceMatcher (elements compared as hashable sequence elements; no normalization described).
Version: CPython 3.11 branch `Lib/difflib.py`; project declares `requires-python = ">=3.11"`. The same `SequenceMatcher(None, a, b)` construction and verbatim content emission are present in the CPython 3.15 branch (lines 1153 and 1166-1176), so the behavior is unchanged across the supported range checked.
Fingerprint: sha256:0c6afc23568d55b3e9ac914f9c5361e3033e778aa5b58d3cc82835fc5c638679 (CPython 3.11 `Lib/difflib.py`); cross-checked 3.15 `Lib/difflib.py` sha256:aaa3d1ccdb8d8f0d2e8c99fe1a320a983b770fad94779f1dd417cdd37c033159
Invalidates-On: `unified_diff` passes an `isjunk` callable to `SequenceMatcher` that ignores or normalizes lines
Invalidates-On: `SequenceMatcher` starts normalizing elements (whitespace stripping, case folding) before equality comparison
Invalidates-On: `unified_diff` transforms lines (e.g. `strip()`, `lower()`) before matching or emitting them
Invalidates-On: project minimum Python moves below 3.11 or to a version whose `difflib.unified_diff` differs on this point
Checked-At: 2026-10-17
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-431.md
```

---

```text
Claim: CL:pytest/monkeypatch-delenv-missing-name
Statement: `monkeypatch.delenv(name, raising=False)` leaves an absent variable absent without raising `KeyError`
Subject: pkg:pytest@9.0.2
Verdict: HOLDS
Actual: `MonkeyPatch.delenv` delegates to `self.delitem(os.environ, name, raising=raising)`. In `delitem`, when `name not in dic` it raises `KeyError(name)` only if `raising` is true; with `raising=False` it takes no action, so an absent variable remains absent and no `KeyError` is raised. (The `delenv` docstring states the same: "Raises ``KeyError`` if it does not exist, unless ``raising`` is set to False.")
Source: `/Users/alexhubitski/Projects/buster-ripper/.venv/lib/python3.13/site-packages/_pytest/monkeypatch.py` — `MonkeyPatch.delitem` (lines 299-311) and `MonkeyPatch.delenv` (lines 333-340)
Version: pytest 9.0.2 (per `/Users/alexhubitski/Projects/buster-ripper/.venv/lib/python3.13/site-packages/_pytest/_version.py` line 31)
Fingerprint: sha256:597dffe9ccd42e1bdcdf0485fb437c55eb1bbfdc0637777096261c0ef4fc11fb
Invalidates-On: any pytest release that changes `MonkeyPatch.delitem` so the `raising=False` branch raises or mutates the mapping
Invalidates-On: any pytest release that changes `MonkeyPatch.delenv` to stop delegating to `delitem` or to alter the `raising` semantics
Invalidates-On: the installed `_pytest/monkeypatch.py` bytes changing (fingerprint no longer matches)
Checked-At: 2026-10-17
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-434.md
```

---

```text
Claim: CL:python/urllib-urlopen-timeout-scope
Statement: `urllib.request.urlopen`'s `timeout` argument bounds the wait for a peer that never answers
Subject: lang:python
Verdict: HOLDS
Actual: `urlopen` passes `timeout` to `OpenerDirector.open`, which sets `req.timeout`; `AbstractHTTPHandler.do_open` constructs the connection as `http_class(host, timeout=req.timeout, ...)`, `HTTPConnection` forwards `self.timeout` to `socket.create_connection`, and `create_connection` calls `sock.settimeout(timeout)`. The socket then times out on every blocking operation, so reading the status line from a peer that sends nothing raises `TimeoutError` (`socket.timeout`) after at most that many seconds. Caveat: this is a per-socket-operation timeout, not a whole-request deadline — a peer that trickles bytes can extend the total wait indefinitely, and the DNS lookup inside `create_connection` is not covered by it.
Source: https://docs.python.org/3/library/urllib.request.html — `urlopen`: "The optional *timeout* parameter specifies a timeout in seconds for blocking operations like the connection attempt (if not specified, the global default timeout setting will be used). This actually only works for HTTP, HTTPS and FTP connections." Corroborated by the exact 3.11 source: `Lib/urllib/request.py` lines 139-152 and 500-509, 1330; `Lib/http/client.py` lines 890, 1010-1011, 275, 297; `Lib/socket.py` lines 843-845; and the `socket.settimeout` documentation ("subsequent socket operations will raise a timeout exception if the timeout period value has elapsed before the operation has completed").
Version: docs.python.org/3 — Python 3.15.0 documentation (page last updated 2026-10-17); project declares `requires-python = ">=3.11"` in pyproject.toml, and the CPython 3.11 source docstring carries identical wording
Fingerprint: sha256:256a5af44d3299f052f1833232633eea579a9cb3d4d4b314012e6fcadb24350d (fetched HTML of https://docs.python.org/3/library/urllib.request.html)
Invalidates-On: the documented meaning of `urlopen`'s `timeout` parameter changes from "blocking operations" to connection-attempt-only
Invalidates-On: `urlopen`/`OpenerDirector.open` stops propagating `timeout` to the `HTTPConnection` socket
Invalidates-On: `socket.create_connection` stops applying `settimeout` to the socket it returns
Invalidates-On: the project moves to a Python version whose urllib/socket timeout semantics differ from the documented ones
Checked-At: 2026-10-17T23:47:04Z
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-436.md
```

---

```text
Claim: CL:python/subprocess-popen-environment-inheritance
Statement: `subprocess.Popen` with no `env=` argument starts the child with the parent's environment at spawn time
Subject: lang:python
Verdict: HOLDS
Actual: `Popen`'s `env` parameter defaults to `None`. When `env` is `None`, the child is not given a replacement environment; the documented default is to inherit the current process' environment, i.e. the environment of the parent at the moment the process is created. Passing a mapping instead replaces that inherited environment entirely.
Source: https://docs.python.org/3/library/subprocess.html — "Popen Constructor" section, `env` parameter: "If *env* is not `None`, it must be a mapping that defines the environment variables for the new process; these are used instead of the default behavior of inheriting the current process' environment." The same wording appears in "Frequently Used Arguments" for `run()`.
Version: Python 3.15.0 documentation (docs.python.org/3, current development docs); the `env=None` default and inheritance behavior are long-standing and not marked with a version change on this page.
Fingerprint: sha256:bcbee8309e8e6c4aec03088b1ac0ab1e5c29390c8158ff35ccc00e1776876fc8
Invalidates-On: the `env` parameter's documented default stops being `None` in the Popen signature on docs.python.org/3/library/subprocess.html
Invalidates-On: the sentence "these are used instead of the default behavior of inheriting the current process' environment" is removed or altered
Invalidates-On: the fetched page content changes so its sha256 no longer matches the recorded fingerprint
Checked-At: 2026-10-18T00:24:36Z
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-437.md
```

---

```text
Claim: CL:python/unified-diff-header-yield-order
Statement: `difflib.unified_diff` yields its `---` and `+++` file-header lines before any `@@` hunk-header line
Subject: lang:python
Verdict: HOLDS
Actual: `unified_diff` tracks a `started` flag while iterating `SequenceMatcher(...).get_grouped_opcodes(n)`. On the first group only, before computing or yielding that group's hunk header, it yields `--- {fromfile}{fromdate}{lineterm}` and then `+++ {tofile}{todate}{lineterm}`. Only afterwards, in the same iteration, does it yield `@@ -{file1_range} +{file2_range} @@{lineterm}`. Thus the two file-header lines precede every `@@` hunk header; if there are no grouped opcodes at all, neither the header lines nor any `@@` line is yielded.
Source: CPython 3.15 `Lib/difflib.py`, function `unified_diff` (lines 1101-1170), https://raw.githubusercontent.com/python/cpython/3.15/Lib/difflib.py ; corroborated by the documented example at https://docs.python.org/3/library/difflib.html ("--- before.py", "+++ after.py", then "@@ -1,4 +1,4 @@").
Version: CPython 3.15 branch source; Python 3.15.0 documentation
Fingerprint: sha256:aaa3d1ccdb8d8f0d2e8c99fe1a320a983b770fad94779f1dd417cdd37c033159 (Lib/difflib.py); sha256:694e808b780a68f0481f36fde8ef38db17adf6b74373106fcbc49c22016506ff (docs page)
Invalidates-On: `unified_diff` stops emitting file-header lines before the first hunk header
Invalidates-On: `unified_diff` stops emitting `---`/`+++` lines entirely or reorders them after `@@`
Invalidates-On: the `started` flag / grouped-opcode loop in `Lib/difflib.py` is restructured
Invalidates-On: a different difflib implementation or major version replaces this behavior
Checked-At: 2026-10-17
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-438.md
```

---

```text
Claim: CL:python/unified-diff-hunk-header-closing-marker
Statement: `difflib.unified_diff`'s hunk-header line contains a second `@@` after the opening one
Subject: lang:python
Verdict: HOLDS
Actual: The hunk-header line is built as `f'{t.hunk}@@ -{file1_range} +{file2_range} @@{lineterm}{t.reset}'`. It opens with `@@`, then carries the two range descriptors, then a second literal `@@` before the line terminator (and, when color is enabled, the reset sequence). The documented example shows exactly `@@ -1,4 +1,4 @@`.
Source: CPython 3.15 `Lib/difflib.py`, function `unified_diff`, hunk-header yield line, https://raw.githubusercontent.com/python/cpython/3.15/Lib/difflib.py ; corroborated by the documented example at https://docs.python.org/3/library/difflib.html (`@@ -1,4 +1,4 @@`).
Version: CPython 3.15 branch source; Python 3.15.0 documentation
Fingerprint: sha256:aaa3d1ccdb8d8f0d2e8c99fe1a320a983b770fad94779f1dd417cdd37c033159 (Lib/difflib.py); sha256:694e808b780a68f0481f36fde8ef38db17adf6b74373106fcbc49c22016506ff (docs page)
Invalidates-On: the hunk-header format string in `unified_diff` drops or changes the trailing `@@`
Invalidates-On: the hunk header is reimplemented to use a single `@@` or a different range syntax
Invalidates-On: a different difflib implementation or major version replaces this behavior
Checked-At: 2026-10-17
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-439.md
```

---

```text
Claim: CL:python/unified-diff-content-line-terminator
Statement: `difflib.unified_diff` retains the input line's terminator on each content line it yields
Subject: lang:python
Verdict: HOLDS
Actual: Content lines are yielded as the original input string with only a one-character prefix prepended: `' ' + line` for equal, `'-' + line` for replace/delete, `'+' + line` for replace/insert. The input string `line` is embedded verbatim, so whatever line terminator it carries (or lacks) is retained; `lineterm` is applied only to the `---`, `+++`, and `@@` control lines. (With `color=True`, a reset escape sequence is appended after the line, but the line's own terminator is still retained.) The docs confirm this split: control lines are "created with a trailing newline" via `lineterm`, while for inputs without trailing newlines one sets `lineterm=''` "so that the output will be uniformly newline free."
Source: CPython 3.15 `Lib/difflib.py`, function `unified_diff`, content-line yields (`yield f'{t.context} {line}{t.reset}'`, `yield f'{t.removed}-{line}{t.reset}'`, `yield f'{t.added}+{line}{t.reset}'`), https://raw.githubusercontent.com/python/cpython/3.15/Lib/difflib.py ; corroborated by https://docs.python.org/3/library/difflib.html (diff control lines vs inputs without trailing newlines).
Version: CPython 3.15 branch source; Python 3.15.0 documentation
Fingerprint: sha256:aaa3d1ccdb8d8f0d2e8c99fe1a320a983b770fad94779f1dd417cdd37c033159 (Lib/difflib.py); sha256:694e808b780a68f0481f36fde8ef38db17adf6b74373106fcbc49c22016506ff (docs page)
Invalidates-On: content-line yields start appending or substituting `lineterm` instead of embedding the input line verbatim
Invalidates-On: content lines begin being stripped of trailing newlines before prefixing
Invalidates-On: a different difflib implementation or major version replaces this behavior
Checked-At: 2026-10-17
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-440.md
```

---

```text
Claim: CL:python/urllib-build-opener-default-handler-substitution
Statement: `urllib.request.build_opener` skips its default `ProxyHandler` when the caller supplies a `ProxyHandler` instance
Subject: lang:python
Verdict: HOLDS
Actual: `build_opener` builds a `default_classes` list whose first entry is `ProxyHandler`, then removes from that list any default class matched by a caller-supplied handler — either an instance (`isinstance`) or a subclass (`issubclass`) — and only then instantiates the remaining defaults and appends the caller's handlers. So passing a `ProxyHandler` instance (or subclass) suppresses the default `ProxyHandler`; the supplied handler is added instead. The documentation says the default classes are placed in front "unless the handlers contain them, instances of them or subclasses of them," and CPython 3.15's `Lib/urllib/request.py` implements exactly that skip loop.
Source: https://docs.python.org/3/library/urllib.request.html — `build_opener` entry (defaults list including `ProxyHandler (if proxy settings are detected)`, "unless the handlers contain them, instances of them or subclasses of them"); corroborated by CPython 3.15 `Lib/urllib/request.py`, `build_opener` lines 539-576
Version: Python 3.15.0 documentation (docs.python.org/3); installed interpreter version not determined (execution prohibited)
Fingerprint: sha256:a5bb216f0ea65aef0ba804f0183cdeabd275f3f003bc53898744442a4f1f0eb4
Invalidates-On: CPython changes the `skip` / `default_classes` logic in `Lib/urllib/request.py::build_opener`
Invalidates-On: the `build_opener` documentation paragraph stops listing `ProxyHandler` among the defaults suppressed by caller-supplied handlers
Invalidates-On: `build_opener` changes to skip defaults only for subclasses, not instances
Checked-At: 2026-10-17
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-441.md
```

---

```text
Claim: CL:python/urllib-proxy-handler-empty-mapping
Statement: `urllib.request.ProxyHandler` with an empty proxy mapping installs no proxy for any host
Subject: lang:python
Verdict: HOLDS
Actual: `ProxyHandler({})` keeps the empty mapping as-is — only `proxies=None` triggers the `getproxies()` environment lookup — and because the mapping has no entries it sets no `<protocol>_open` methods on the instance. `OpenerDirector.add_handler` explicitly skips the class's own `proxy_open` method, so nothing redirects any request through a proxy for any host. The documentation states "To disable autodetected proxy pass an empty dictionary" and that a `<protocol>_open()` method exists only "for every protocol which has a proxy in the `proxies` dictionary given in the constructor."
Source: https://docs.python.org/3/library/urllib.request.html — `ProxyHandler` class entry ("To disable autodetected proxy pass an empty dictionary") and "ProxyHandler Objects" section (`<protocol>_open` exists only for protocols present in the `proxies` dictionary); corroborated by CPython 3.15 `Lib/urllib/request.py`, `ProxyHandler.__init__` lines 762-777 and `OpenerDirector.add_handler` line 413
Version: Python 3.15.0 documentation (docs.python.org/3); installed interpreter version not determined (execution prohibited)
Fingerprint: sha256:a5bb216f0ea65aef0ba804f0183cdeabd275f3f003bc53898744442a4f1f0eb4
Invalidates-On: `ProxyHandler.__init__` changes to treat an empty mapping like `None` and call `getproxies()`
Invalidates-On: `ProxyHandler` gains a `default_open`/`unknown_open` or unconditional proxying path independent of the `proxies` mapping
Invalidates-On: `OpenerDirector.add_handler` stops excluding `proxy_open` from handler method discovery
Invalidates-On: the `ProxyHandler` documentation stops stating that an empty dictionary disables autodetected proxying
Checked-At: 2026-10-17
Provenance: session:02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/verdicts/claim-442.md
```

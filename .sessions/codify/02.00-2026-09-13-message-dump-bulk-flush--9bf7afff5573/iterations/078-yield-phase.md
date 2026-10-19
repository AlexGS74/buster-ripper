# Iteration 078 — Yield Phase (MS-001 / QAP-MS001, re-entry after the 077 harness drain)

## Orchestrator Prompt (verbatim)

```text
You are a subagent spawned by another session.
Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
Milestone: MS-001
QA Pass: QAP-MS001
Target base: main
Target object: 72e3a1c6861bf0d8de6317adf52289e5f6cfa4e1
```

Orchestrator instruction line:

```
Execute the yield-phase-interactive skill for this session. Arguments: $ARGUMENTS
```

## Invocation Contract Verification

- Session handle `02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573` verified against
  `session-identity.json` (`session_id: 02.00-2026-09-13-message-dump-bulk-flush`,
  `session_uid: 9bf7afff-5573-402a-b102-827423a1dded`) — **MATCH** (`--9bf7afff5573` = the uid's first
  8 + next 4 hex groups). Files read directly; nothing globbed or searched for. Not
  `BLOCKED-SESSION-IDENTITY`.
- Full milestone gate: `Milestone: MS-001` + `QA Pass: QAP-MS001` + `Target base: main` + full
  `Target object` — all present. Contract satisfied.
- **BASE-SYNC gate: verified here** — see §Base-Sync Re-Verification.
- Constituent tasks: `MS-001 = T001, T002, T005` (Milestone Matrix). All three accepted by local
  Yield: T001 [031-01-yield](031-01-yield.md), T002 [031-02-yield](031-02-yield.md), T005
  [049-01-yield](049-01-yield.md); archived in `plan-completed.md` rows 7, 8, 10.
- `QAP-MS001` matches `MS-001`'s row in `qa-plan.md` §QA Author Passes (line 63) — the invocation's
  `QAP-*` is that milestone's author pass, **re-entered** at
  [077-forward-qa](077-forward-qa.md) to drain the two `forward-qa`-owned `QB:qa-harness/*` rows whose
  named trigger is a touch of `tests/qa/harness.py`.
- **Trigger level: full cycle/phase boundary.** All 7 checks run. No parallel Codex findings were
  supplied with this dispatch, so §Additional review findings records none; the last external review
  of this boundary is the post-071 CR2 packet, whose four findings landed as `T015`/`T016` and are
  re-verified here as integration surface (§Check 2, §Check 3).

## Scope Of This Pass (what moved since 071, and therefore what is not inherited)

071 SEMANTIC-APPROVED this gate on HEAD `c7420ce`. The boundary then went **BLOCKED, not accepted**
(Checkpoint 163) on the post-071 `codex-review` CR2 batch, and the repair waves landed after it.
Everything this pass must therefore re-derive on current bytes:

```
$ git diff --stat c7420ce...HEAD -- buster_ripper/ tests/
 buster_ripper/routes/chat.py        |  51 +++---    (T015/T016: header parity, raw-relay fall-through, de-leak)
 buster_ripper/routes/messages.py    |  22 ++-      (T016: preserved-thinking guard, de-leak, dead state key)
 buster_ripper/routes/passthrough.py |   5 +-       (T016: de-leak the 502 body)
 buster_ripper/utils.py              |  24 ++-      (T017: one pair per occurrence, CIMultiDict)
 tests/qa/harness.py                 |  29 +++-     (077: the two QB drains)
 tests/qa/test_qap_ms001.py          |  48 ++++++   (077: the drain's sensitivity witness)
 tests/test_header_parity.py         |  99 +++++++-
 tests/test_routes_integration.py    |  30 ++--
 tests/test_upstream_answer_relay.py | 307 +++++++++ (CR2 wave's witnesses, incl. TestLeakFreeErrorBodies)
```

- **`MS-001`'s owned surface is byte-identical to 071's**: `git diff c7420ce...HEAD -- buster_ripper/dump.py
  buster_ripper/config.py buster_ripper/main.py README.md serve_buster_ripper.sh buster_ripper/server.py`
  → **empty**. Checks 1–5 re-derive the milestone surface from source anyway (below), and the
  post-071 production delta is verified where it touches this milestone's evidence — the response arm
  `TC-001` observes and the shared harness every scenario drives through.
- Cumulative review scope remains `git diff 72e3a1c6…HEAD` (619 files; `buster_ripper/` 14 modules,
  `tests/` + `tests/qa/`, `README.md`, `pyproject.toml`, `serve_buster_ripper.sh`, `uv.lock`,
  `buster_ripper_legacy.py`, the session/registry trees).

## Base-Sync Re-Verification

Latest base-sync report is `BASE-SYNC: unchanged` ([054-01-forward](054-01-forward.md)), target base
object `72e3a1c6861bf0d8de6317adf52289e5f6cfa4e1` — **the same object this invocation names**. No newer
base-sync report exists: `grep -rn "BASE-SYNC" --include=*.md .` over the session directory returns
carriers at 033, 035, 054, 056, 058, 069, 071, 072 + `chronicle.md`/`plan-completed.md`, and 054 is
still the newest *report* (071/072 re-verify it, they do not restate it). Re-verified at HEAD
`6508941`:

```
$ git rev-parse main origin/main
72e3a1c6861bf0d8de6317adf52289e5f6cfa4e1     # = the target object exactly — base has not moved
72e3a1c6861bf0d8de6317adf52289e5f6cfa4e1
$ git merge-base --is-ancestor 72e3a1c6… HEAD && echo true
true
$ git rev-list --count 72e3a1c6…HEAD ; git rev-list --count HEAD..72e3a1c6…
212
0
```

- Exact object is an ancestor of `HEAD`; **212 ahead / 0 behind**. The worktree carried no production
  change during this pass (`git status --porcelain` → one `??` entry: this pass's evidence transcript).
- Every correctness-required integration repair passed local Yield: `T001`/`T002`/`T005`
  ([031-01](031-01-yield.md)/[031-02](031-02-yield.md)/[049-01](049-01-yield.md)), and post-071
  `T015`+`T016` (CR2 wave) and `T017` accepted-local at [073-01-yield](073-01-yield.md) and
  [075-01-yield](075-01-yield.md) with zero reject-tier findings.
- No final-phase preventive merge work is deferred: `MS-004` is the final phase and closed at
  [056-yield-phase](056-yield-phase.md); [054-01-forward](054-01-forward.md) reviewed all then-open
  `quality-backlog.md` items for merge exposure and emitted no prework packet; nothing since has
  introduced a base delta (`main` unmoved).
- **`BASE-SYNC: unchanged` therefore holds at this boundary → the integration verdict is permitted to run.**

## Check 1: Caller Verification — PASS

Scope: the MS-001 surface from source, plus every production entry point the cumulative diff adds, plus
the wiring the post-071 waves and the 077 drain changed or removed.

**New public entry points → production callers (independent re-derivation)**

| Symbol | Added by | Production caller(s) | Verdict |
|---|---|---|---|
| `dump.dump_turn` | T001/T002 | `routes/messages.py:12` (import) + `:56` (call, inside the swallow-all at `:54-58`) — the only production call site, and the only route that dumps | ✔ wired |
| `dump.flush_all` | T002 | `server.py:112` (`_final_dump_flush`, reached from `on_cleanup` `:88`), `dump.py:206` (volume trigger), `dump.py:133` (`poll_flush`) | ✔ three callers (071-era: none but T003's; unchanged since) |
| `dump.poll_flush` | T003 | `server.py:45`, inside `spawn_dump_flush`, itself called at `server.py:81` from `on_startup`, registered at `:120` | ✔ task is actually started |
| `config.DUMP_FORMATS` / `config.DUMP_FORMAT` | T002 | `main.py:80` (`click.Choice(list(config.DUMP_FORMATS))`) writes via `main.py:14`; read at `dump.py:141-142` | ✔ both channels present, no orphan |
| `main --dump-format` | T002 | console script `pyproject.toml:11` `buster-ripper = "buster_ripper.main:main"`; `serve_buster_ripper.sh:29,:41` maps `DUMP_FORMAT` onto it | ✔ reachable from the documented run path |
| `routes.messages.handle_messages` | (pre-existing) | `server.py:127 app.router.add_post("/v1/messages", …)` | ✔ |

Every new public function in the cumulative diff (`normalize`, `session_id`, `response_headers`,
`forward_headers`, `handle_*`, `on_startup`/`on_cleanup`/`create_app`/`main`, the flush trio) has ≥ 1
non-definition reference inside `buster_ripper/`; zero test-only production symbols.

**Removed/replaced wiring in scope, and its replacement**

| Removed | By | Replacement + its caller | Verdict |
|---|---|---|---|
| `dump_turn`'s bare `assert config.DUMP_DIR is not None` | T002 | real guard raising `ValueError` (`dump.py:96-97`); still inside the caller's swallow-all (`messages.py:57`) | ✔ (no `assert ` survives in `dump.py`) |
| `Proxy.url(path=…)` / `Proxy.post(body, timeout, path=…)` keywords | 077 | `Proxy.url()` / `Proxy.post(body, timeout=90.0)`; `post`'s own call site now `self.url()` (`harness.py:229`) | ✔ `grep -rnE "\.post\(.*path=\|\.url\(.*path=" tests/` → **0 call sites**; `peer.url()` hits are `tests/qa/tls_peer.py:276`'s own `url(host=None)`, a different class |
| `assert_upstream_response`'s `b"Cannot connect" not in body` limb | 077 | the surviving status limb (`harness.py:148`) + the leak property re-pointed to its owner `tests/test_upstream_answer_relay.py::TestLeakFreeErrorBodies` | ✔ — see §Check 6 fidelity, sufficiency established not assumed |
| `{"error": str(e)}` client bodies on four arms | T016/T015 | fixed `{"error": "Upstream request failed"}` at `status=502`, **with the cause still logged** at all four (`chat.py:151`, `messages.py:166,:298`, `passthrough.py:51`) | ✔ no severed operator channel |
| `state["input_tokens"]` (write-only key) | T016 | none needed — `grep -rn 'state\["input_tokens"\]' buster_ripper/ tests/` → **0 consumers** on either tree | ✔ dead, not orphaned |

**Failure criteria**: no new public method with zero production callers; no removed callback without a
wired replacement; no production symbol reachable only from tests.

## Check 2: Migration Parity — PASS

**Parity pair 1 — the documented writer vs the pre-refactor monolith** (`buster_ripper_legacy.py:558-615`
`_dump_turn`, the code `T002`'s format branch replaced). Every old side effect against its counterpart:

| Old side effect (`buster_ripper_legacy.py`) | New counterpart (`buster_ripper/dump.py`) | Parity |
|---|---|---|
| `session_dir.mkdir(parents=True, exist_ok=True)` + `log.info("dump: new session %s → %s")` | `:153-157` `_lookup_or_create_state` | ✔ same call, same message |
| turn 0 → `out.write_text(json.dumps(body, indent=2, ensure_ascii=False))` | `:172-177` `_write_full_turn` | ✔ same path, same serializer, same args |
| turn 0 summary `turn 000  HH:MM:SS  FULL  n msgs  n tools  x.y KB` + `log.info("dump: %s")` | `:237-241` `_full_summary` + `:176` | ✔ byte-identical format string |
| turn ≥ 1 → `difflib.unified_diff(prev, curr, fromfile=turn_{N-1:03d}, tofile=turn_{N:03d}, lineterm="")` → `write_text("".join(...))` | `:180-189` + `:262-270` `_unified_diff` | ✔ same diff args, same file name |
| `+added/-removed` counts excluding `+++`/`---` → `turn NNN … +A/-R lines  n msgs  n tools` | `:273-277` `_diff_counts` + `:244-249` `_diff_summary` | ✔ same predicate, same line |
| `index.open("a")` + `write(summary + "\n")` — **once per turn, after the turn file** | `:192-197` `_append_index_line`, called after the branch (`:167`) | ✔ same order → `2N` write-bearing ops, witnessed at `TC-108` |
| `state.prev_text = current_text`; `state.turn += 1` at append | `:108-109` | ✔ advance at append, never at flush |
| caller: `try: _dump_turn(body) except Exception: log.exception("dump failed")` | `messages.py:54-58` | ✔ identical swallow, identical message |

Intentional, documented deltas (not omissions): legacy's `assert DUMP_DIR is not None` → `ValueError`
guard (`F-5`, T002's done-when); the added `_require_writable` symlink guard (`codify-adversarial-surface-discipline`
R2, recorded at 071 §Check 2); the `fmt` dispatch itself, which is the increment. Legacy computed `now`
once per turn, the new code inside each summary builder — same output for one turn, and `TC-108`'s
line-pattern witness does not depend on the difference.
**Zero accidental omissions found.**

**Parity pair 2 — `T002`'s format branch vs "always documented"**: the old behavior is *retained as one
branch*, not replaced (`US-012`, `PI:T002/dump_format/documented_format_is_retained`), and the branch is
exclusive so nothing is written twice — witnessed by `TC-107`/`TC-108` (`2N`) against `TC-100` (`F`).

**Parity pair 3 — the 077 harness drain vs the pre-drain helper**: the removed body limb protected
"distinguish a proxy-local transport failure from an answer that came from the peer". That outcome is
still carried: `grep -rn -B2 'status=502' buster_ripper/routes/*.py` shows **all four** arms answer
`502`, so `200 <= status < 500` refuses every proxy-local transport failure, and the *new* false-positive
shape the limb had acquired (a relayed peer body saying `Cannot connect`, made reachable by T015's
verbatim relay) is now explicitly accepted by the witness. Nothing the helper proved is lost; one thing
it mis-attributed is gone.

**Parity pair 4 — the CR2 body change vs the operator channel**: `str(e)` left the client body and stayed
in the log at all four arms (table in §Check 1); `test_upstream_answer_relay.py::TestLeakFreeErrorBodies`
witnesses both halves. This is `TC-001`'s "the exception is logged" clause, now pinned at the cheaper tier.

## Check 3: Data Flow Completeness — PASS

**Path A — the dump write path (this milestone's producer → artifact):**
`server.py:127` route → `messages.py:29` parse → `:36` `normalize` → `:37` `sid` → `:44-49`
preserved-thinking → **`:54-58` `dump_turn(body)` under the swallow-all** → `dump.py:98`
`_resolve_format()` → `:99` `_lookup_or_create_state` (dir created) → exclusive dispatch `:102-107` →
documented: `_write_documented` → `turn_000.json` / `turn_NNN.diff` / `index.txt`; collapsed:
`_append_record` → `state.pending` (+ volume check `:205-206`) → `flush_all` → `_flush_session` →
`_flush_state` → one append to `<format>.log` → **consumer: the human/`grep` reader documented at
`README.md:151-167,:197-211`**. Complete. Verified live: `TC-107`/`TC-108` layout+contents, `TC-100`
deferral then one artifact.

Two flow facts checked rather than assumed:
- **The sid is derived twice** — `messages.py:37` pre-injection and `dump.py:149` post-injection — and
  cannot diverge: `session.py:93-98` hashes **only** the system prompt, which neither `normalize`'s
  caller ordering nor `_inject_preserved_thinking` (a `chat_template_kwargs` mutation) touches. So
  `TC-001`'s pre-created session dir is the dir the writer opens. ✔
- **`counted_proxy_entry.py` really is on the write path**: it patches `io.open` **and**
  `builtins.open` (`:109-110`) before importing `buster_ripper.main`, and `Path.write_text`/`Path.open`
  resolve `io.open` at call time — the report's per-path decomposition (`index.txt` ×3, each turn file
  ×1) matches the writer's four write sites, so the count is the process's own behavior, not a mock. ✔

**Path B — the flush triggers (all three reach the same drain):**
volume `dump.py:205-206` → `flush_all`; timer `on_startup`(`:81`) → `spawn_dump_flush` → `create_task`
(`:45`) → `poll_flush` → `flush_all`; shutdown `on_cleanup`(`:84-88`) → cancel-and-await the task →
`_final_dump_flush` → `flush_all`. Order matches the protected invariant in `handoff-context.md`
§Notes ("cancel timer → final flush → close shared session"). ✔ Nobody writes and nobody reads a
channel that the other end ignores.

**Path C — `TC-001`'s request path across the post-071 delta (the check that matters at a re-entry):**
client → `handle_messages` → dump raises → swallowed and logged → `_non_stream_single_request` →
`session.post` → upstream 404 → `content = await resp.read()` → `response_headers(resp.headers)` (now a
`CIMultiDict`, `utils.py:52-55`) → `web.Response(body=content, status=resp.status, headers=…)` →
client. The fault changes nothing on this leg: `TC-001`'s control/broken status equality passed, and the
T017 container widening moved the helper **onto** `web.Response`'s own declared type
(`CIMultiDict[str]`), i.e. toward its consumer rather than away from it. ✔
The proxy-local branch of the same path (`ClientError` → `502`, `messages.py:297-299`) is what the
drained harness limb keys on — verified above.

**Fault-path integrity (no silent drop):** on the documented branch a raising write aborts before
`dump.py:108-109`, so the turn counter does not advance on a failed turn — identical to legacy, where
the raise also escaped before the same two assignments. On the collapsed branch the raise happens in
`_flush_state`, `_flush_session` logs it, and `pending` is cleared **only after** a successful write
(`:304`), so a failing flush retains the batch rather than losing it. ✔

**Timing:** `DUMP_FLUSH_INTERVAL` defaults to 10 s and `DUMP_FLUSH_BYTES` to 4 MiB (`config.py:36-37`),
so `TC-100`'s sub-second window cannot be pre-empted by a tick in practice, and the test's
`assert pre_flush_snapshots` names the environment as the reason if it ever is. 071 ruled the `022`
directive honored literally on exactly this construction; nothing since changed the defaults or the
test. ✔

## Check 4: Consumer Contract Tracing — PASS (one P3)

| Producer | Field | Values it can create | Consumers | Handled? |
|---|---|---|---|---|
| `main.py:14` / `config.DUMP_FORMAT` | the format selector | exactly `click.Choice(list(config.DUMP_FORMATS))` members: `documented`, `change-focused`, `self-contained`; anything else refused at parse time | `dump._resolve_format` (membership → `ValueError`), `dump_turn`'s three-arm dispatch, `_flush_state`'s `== "documented"` early return, `_artifact_path` | ✔ today — **see P3-1** |
| `dump.dump_turn` | `state.fmt` | the value `_resolve_format` just returned, assigned immediately before the arm that fills `pending` (`:100-107`) | `_flush_state` / `_artifact_path` | ✔ batch and filename always share one value |
| `_full_summary` / `_diff_summary` | summary lines in `index.txt` and in every collapsed record | `turn NNN  HH:MM:SS  FULL … KB` / `… +A/-R lines … msgs … tools` | the **human/shell reader**, whose published pattern is `README.md:205` `^turn [0-9]{3}  [0-9]{2}:[0-9]{2}` and whose columns are `FULL`/`+A/-R`, `n msgs`, `n tools`, `KB` | ✔ and *gated*, not assumed: `test_tc108_sensitivity_documented_line_pattern_distinguishes_drift` accepts the three README-published lines and rejects five drift forms |
| `counted_proxy_entry.main` | the JSON write report | `{dump_root, ops[{path,mode,bytes}], total_ops, total_bytes}` | `TC-100`/`TC-108` via `count_report`/`ops_by_name` | ✔ schema emitted = schema consumed |
| `routes/*.ClientError` arms | the proxy-local failure answer | `status=502`, body `{"error": "Upstream request failed"}` at all four arms | `harness.assert_upstream_response`'s `200 <= status < 500` | ✔ — this is the seam the 077 drain rests on, re-derived here |

**Merged/coalesced data (the case this check exists for):** `flush_all` coalesces a whole pending batch
into one write. The dispatch-bearing field inside the merge is each record's own `turn NNN` identity,
which is per record and survives `"\n\n".join` — witnessed by `TC-100` (three markers, one file, all
writes naming one path). The merge never spans sessions (`flush_all` iterates `_sessions.values()`, one
path per state) — witnessed by `R5`'s near-miss pair, which additionally proves a pre-existing
`change-focused.log` in a session dir is neither merged into nor removed. ✔

**Heterogeneous-state shape:** a `pending` batch built under format A with `state.fmt` reading B is
reachable only by mutating `config.DUMP_FORMAT` at runtime; `PI:T002/dump_format/format_is_per_artifact`
declares that state does not exist and the source agrees (one write to the global, at parse time, before
`create_app`). Recorded as such at claim-93's closure (`DC-claim-93-r1`); not a finding. ✔

## Check 5: Plan Fulfillment — PASS

Done-when text taken from `plan-summary.md` **as it stood at each task's acceptance** (`git show` of
Checkpoints 61, 102 and 109 — the current §Task List archives these tasks to `plan-completed.md`),
verified against source at HEAD, not against a report.

**`T001` — per-session pending-record buffer.** `REMOVED: nothing`; the ratchet the plan set is the
inverse of the usual one — nothing may have been deleted early.
- `SessionState.pending: list[str]` + `pending_bytes: int` → `dump.py:39-40` ✔
- `turn`/`prev_text` advance at append time → `:108-109`, after the exclusive branch ✔
- Four write sites still live (`:174`, `:186`, `:196`, `:302`) → no early removal ✔
- `pending_bytes` measures rendered-text UTF-8 bytes (`:202`), matching `FK-B` and `handoff-context.md`
  §Notes ✔

**`T002` — format branch, collapsed artifact, single-write flush.**

| Clause | Where | Verified |
|---|---|---|
| `DUMP_FORMATS` = exactly the three values, `DUMP_FORMAT = "documented"`; the R2 candidate two-set must NOT be authored | `config.py:27-28` | ✔ three values, default `documented`; the two-set was not authored |
| `--dump-format` = `click.Choice(list(config.DUMP_FORMATS))`, `default`, `show_default`, assigning **through the option's own callback** so `main()` gains no parameter | `main.py:80-85`, `:12-15` | ✔ `main()`'s signature carries no `dump_format` param |
| Capacity extraction spanning **both** writers; `dump_turn` carries no inline record construction | `_lookup_or_create_state`, `_resolve_format`, `_write_documented` (+`_write_full_turn`/`_write_diff_turn`/`_append_index_line`), `_change_focused_record`, `_self_contained_record`, `_turn_zero_block`, `_artifact_path` | ✔ `dump_turn` is 16 lines of guard + dispatch; construction is shared, not duplicated |
| Unknown format raises `ValueError`; the caller's swallow-all keeps the request alive | `dump.py:141-143` → `messages.py:57-58` | ✔ and gated from outside at `test_hostile_r1_unknown_dump_format_is_refused_and_writes_nothing` |
| Append to `pending` conditioned on the collapsed format (`F-21`; `2N + F` must be unreachable) | `dump.py:102-107` | ✔ `_append_record` is reachable only from the two collapsed arms |
| `flush_all()`: one write per session, clears after the write, no-ops on `DUMP_DIR is None`/nothing pending, must not raise, **no `await`**, never touches `turn`/`prev_text` | `dump.py:112-122`, `:285-305` | ✔ the module's only `await` is `poll_flush`'s sleep (`:136`, T003's) |
| Artifact = `DUMP_DIR/<sid>/<format>.log`, `<format>` exactly the configured collapsed value; `documented` builds no artifact path | `:29-30`, `:280-282`, `:297-298` | ✔ |
| Both collapsed records per spec; summary built in both, format-independent; documented summaries unchanged | `:209-249` | ✔ |
| bare `assert` → real `ValueError` guard (`F-5`) | `:96-97` | ✔ `grep -n "assert " buster_ripper/dump.py` → **zero hits** |
| (022) `pyproject.toml` `norecursedirs` retains all ten pytest defaults **plus** `qa` | `pyproject.toml:23`, gated by `tests/test_pytest_config.py` | ✔ re-proven at this boundary: the bare run collected 455 tests, no subprocess, no network, 9.17 s |
| **REMOVED (dropped by 018/019) — must NOT be present:** deletion of the three documented write sites | — | ✔ **not removed** — `G-008`(a)'s writer is retained as the documented branch |

**`T005` — extend the README's dump section.** `REMOVED: no tool section` (and the removal of the
documented block was itself refused by 018/019).
- collapsed artifact name/shape ✔ `README.md:178-184`; every shipped format and how it is selected ✔
  `:171-176,:186-189`; per-format `grep`/`less` reading workflow ✔ `:197-211`; the collapsed selection on
  the run path the section already documents ✔ `:192`; `F-8`'s arrival-order caveat ✔ `:213-216`;
  pre-existing-directory note ✔ `:218-220`
- documented format's block stays where it is, unchanged ✔ `:151-167` is the layout + index sample the
  `TC-108` matcher reads its contract from
- `serve_buster_ripper.sh` gains `DUMP_FORMAT` mirroring `DUMP_DIR`'s shape — empty default, presence
  guard, exactly one flag when set ✔ `:29`, `:41`. (The clause's literal string-append form was replaced
  by the array form at T012's word-integrity fix; the clause's *requirements* are what T012's witnesses
  pin, ruled at 071 and unchanged.)
- no tool section anywhere ✔

**Milestone `Complete when` (MS-001 row) — each conjunct against an artifact re-produced at this boundary:**

| Conjunct | Witness (078's own run) |
|---|---|
| documented-format session → today's three files, today's content/line format, at `2N` writes | `TC-108` **pass** — `write_ops=6 == 2N`, decomposition `index.txt:3 / turn_000.json:1 / turn_001.diff:1 / turn_002.diff:1`, contents asserted against the README's published lines |
| its README workflow unchanged | §Check 2 parity table + `README.md:151-167` |
| no format configured → documented layout | `TC-107` **pass** — exactly `N+1` documented files, zero `.log` anywhere under the root, every `*DUMP*` env var scrubbed by the autouse fixture |
| collapsed format selected → N turns produce **no artifact on disk**, every write deferred | `TC-100` **pass** — pre-flush snapshots `[(0.01, []), (0.02, []), (0.03, [])]`, then one artifact holding all three turns |
| a raising dump/flush still cannot fail a request | `TC-001` **pass** ×2 — status equal to a healthy control, `PermissionError` logged, `rc == 0` with the fault live across the shutdown flush |
| ⚠ 022: the milestone declares **no** flush | `grep -n "dump-flush" tests/qa/test_qap_ms001.py` → the docstring stating the rule; no `extra_args` carries either flag ✔ |

**Relay-vs-transfer test:** `MS-001`'s architectural goal is an *addition plus a retention*, not an
ownership transfer (`US-012` forbids displacing the documented writer before the `MS-002` comparison
persuades the user). So the removal test here is inverted: any `2N + F` double-write, any silent
deletion of the retained writer, any relay that merely forwards while old ownership persists? The
exclusive three-arm dispatch and the four-write-site census say no, and the switch the plan defers is
still deferred — `config.DUMP_FORMAT = "documented"` and `show_default` agree.
**Additions verified: 10/10 `T002` clauses + 6/6 `T005` clauses + `T001`'s three. Removals verified: 4
required removals taken (bare assert, inline construction, `path=` keywords, body limb), 2 forbidden
removals confirmed absent (documented write sites, documented README block).**

## Check 6: QA Milestone Closure — PASS

**Handles and coverage (steps 1-3).** `MS-001` → `T001, T002, T005`, author pass `QAP-MS001`
(the invocation matches `qa-plan.md:63`). All three constituents passed local Yield. Scenarios whose
`Required at` is exactly `MS-001`: **`TC-100`, `TC-001`, `TC-107`, `TC-108`** — and only those
(`TC-002` is `MS-003`'s, `TC-101…TC-112` are `MS-002`'s, `TC-003/113/114` are `MS-004`'s). Each names a
`VP-*` whose `Observable after`/`Validate at` in `plan-summary.md` §Validation Point Matrix agrees with
`qa-plan.md`; all four milestone VPs are covered — no orphan VP, no orphan scenario:

| Scenario | VP | Observable after | Validate at | Matches plan |
|---|---|---|---|---|
| `TC-100` | `VP-T001-writes-deferred` | T002 | MS-001 | ✔ |
| `TC-001` | `VP-T001-dump-failure-isolated` | T001 | MS-001 | ✔ |
| `TC-107` | `VP-T002-default-encoding` | T002 | MS-001 | ✔ |
| `TC-108` | `VP-T002-documented-format-preserved` | T002 | MS-001 | ✔ |

**Authorship (step 4).** The pass supplied a persistent gate module (`tests/qa/test_qap_ms001.py`,
committed), a §Contract Draft table authored before implementation reading, a §Boundary Inspection
Exceptions log with its answer recorded and scoped to four lines, the promotion record, and — this
re-entry — one added witness plus a `qa-plan.md` §Authored Gate Tests update. Adequate; no rejection to
`forward-qa` on authorship grounds. **This pass authored no gate test** (contract fidelity is checked,
not written).

**Contract fidelity (step 5), re-checked clause by clause with the drain in place.** The four scenario
gates and their hostile/identity tiers are byte-identical to 070's (verified: `git diff c7420ce...HEAD`
touches this module only by *adding* the one process-free witness and docstring lines — no existing
assertion moved), so 071's clause-by-clause fidelity table stands; what this pass re-derived is the
**oracle the drain changed**, and the two properties it is claimed to still carry:

| Approved element | Still implemented? | Evidence at 078 |
|---|---|---|
| `TC-001` "the upstream response body is returned unchanged" | ✔ (and more strongly than at 071) | the non-stream arm relays `content` verbatim (`messages.py:270,292-296`) since T015; `test_upstream_answer_relay.py::TestNonJsonRelay` witnesses bytes+status+headers travel; the gate's share is the fault-injection comparison + `< 500` |
| `TC-001` "the exception is logged" / forbidden "silent absence of the log record" | ✔ | `poll_until("PermissionError" in stderr)` on both paths, and the collapsed variant keeps the fault live across `stop()` so the raise happens inside `on_cleanup` — `rc == 0` is the sharpest form |
| `TC-001`/`TC-100`/`TC-107`/`TC-108` "the request actually reached the peer" | ✔ **not vacuous after the drain** | all four `ClientError` arms answer `502` (`grep -B2 'status=502'`), so the surviving status limb refuses a dead transport; a proxy whose own port is down raises `urllib.error.URLError` out of `post`, which errors the test rather than passing it; `require_upstream()` still runs fail-closed in every `Proxy.__init__` |
| the drain must not create a new blind spot | ✔ | the leak property named by the old docstring is witnessed at the tier that owns it: `TestLeakFreeErrorBodies::test_the_502_names_no_upstream_topology` (feeds a `ClientError` whose text *does* say `Cannot connect to host …` and asserts the client body does not) + `::test_the_exception_is_still_for_the_operator` |
| forbidden results for `TC-107` (witness W1 of `G-008`) | ✔ | `log_artifacts(dump) == []` + exactly the four documented files; the R2 candidate clause remains unauthored, as `US-012`/`G-008` require |
| forbidden results for `TC-108` | ✔ | drifted-`index.txt`, missing/renamed file and `.log`-on-documented-path all asserted absent, and `test_tc108_sensitivity_…` proves the matcher can fail |
| lifecycle + quiescence (`qa-project` §Synchronization) | ✔ | readiness by port poll, quiescence by `poll_until`/process exit, no fixed sleeps; counted ops asserted inside this session's dir |
| telemetry: none required | ✔ | project declares no telemetry capability (`aiohttp` + `click` only) |
| the `022` directive | ✔ honored literally | one `dump-flush` hit: the docstring; no `extra_args` carries either flag; `TC-100` bounds its pre-flush window by observation, per 071's standing ruling (re-checked, not inherited — `config.py:36-37` defaults unchanged) |

**Execution (step 6) — independently, at this boundary, not replayed from 070/077.** Transcript:
[`../evidence/ms001-reentry-078-gate-run.txt`](../evidence/ms001-reentry-078-gate-run.txt).
`.venv/bin/python -m pytest tests/qa/test_qap_ms001.py -o addopts= -v` → **11 passed in 2.65 s** on
HEAD `6508941`, upstream `hyperipper.local:30001` answered `GET /v1/models` → `200` at pass start and
the harness preflight passed inside every gate.

| Handle | Verdict |
|---|---|
| `VP-T001-writes-deferred` / `TC-100` | **pass** |
| `VP-T001-dump-failure-isolated` / `TC-001` (documented write path) | **pass** |
| `VP-T001-dump-failure-isolated` / `TC-001` (collapsed flush path) | **pass** |
| `VP-T002-default-encoding` / `TC-107` | **pass** |
| `VP-T002-documented-format-preserved` / `TC-108` | **pass** |
| `VP-T002-documented-format-preserved` / `TC-108` pattern sensitivity | **pass** (proves the above can fail) |
| drain witness `test_sensitivity_upstream_oracle_distinguishes_transport_failure_from_a_relayed_body` | **pass** |
| hostile tier R1 selector / R1 body parse, R2 containment, identity tier R5 | **pass** ×4 |
| `not_applicable`: `TC-101…TC-112`, `TC-003`, `TC-113`, `TC-114` | not `MS-001`'s; `TC-002` is `MS-003`'s |

**Does the drain reopen any of the four VPs? No.** All four were approved on artifact/process
assertions, none on the removed body substring; the drain changed only *how* the harness proves the
transport is alive, and that proof was re-derived here (§table above). No scenario's approved trigger,
owner, visible result, forbidden result, lifecycle or telemetry contract was weakened, reinterpreted or
widened. 077's open question 2 (a gate-tier witness that `require_upstream()` is itself load-bearing) is
**a new claim, not a gap**: `qa-plan.md` does not declare it, so it belongs to the plan and not to this
milestone's closure — recorded as a note, not a finding.

**Intermediate QA sweep (step 8).** `ls tests/qa/` → `__init__.py`, `harness.py`,
`counted_proxy_entry.py`, `tls_probe_entry.py`, `tls_peer.py` + its committed cert/key pair, and exactly
the three `test_qap_ms00{1,2,4}.py` gates. **No `MS-001` intermediate lingers; nothing was created by
this pass, so nothing is newly deletable.** The promotion record from 070 still stands —
`read_text`, `log_artifacts`, `assert_upstream_response`, `startup_refused` are each single-defined in
`harness.py` with zero surviving duplicates in `test_qap_ms002.py` (re-grepped). `MS-002`'s
module-local helpers stay: they are threshold-specific to the collapsed path this milestone cannot use.
The `MS-001` unit modules (`test_dump.py`, `test_dump_flush.py`, `test_dump_records.py`,
`test_dump_write_count.py`, `test_dump_triggers.py`, `test_main_cli.py`, `test_pytest_config.py`) are
permanent suite members carrying the `MS-003` floor, not intermediates.

## Check 7: Integration Suite Execution — PASS

**Step 0 — the full unit suite once on this boundary's snapshot** (per-task Yield runs only affected
tests, so this is the run that can catch a regression in an unrelated module): `uv run pytest` →
**455 passed @ 95.75%** in 9.17 s, zero failures, coverage floor 80 % reached, `dump.py` at **100 %**
(180 stmts / 0 miss). Nothing to bisect. Hermetic-ness proved rather than assumed: the default run
collects zero `tests/qa` items and `tests/test_pytest_config.py` gates that `norecursedirs` carries
`qa` alongside all ten pytest defaults.

**Scope derived FROM THE DIFF before running anything.** The synchronized cumulative diff changes code
in exactly one package, `buster_ripper/` (14 modules), plus its two test trees, the serve script,
`pyproject.toml` and `README.md`. The package *is* the application, so its downstream dependents are its
own consumers: the hermetic integration layer — notably `tests/test_routes_integration.py`, which *is*
`TC-002`/`VP-T003-passthrough-regression` — and every gate in `tests/qa/`, all of which read
`buster_ripper.config`'s globals and drive the real binary. The post-071 delta is inside the same
package, so no new dependent appeared.
**Excluded, with the reason:** `buster_ripper_legacy.py` — the tracked pre-refactor monolith, referenced
by nothing under `buster_ripper/` (its five appearances are docstring citations) and excluded from scope
by the plan itself (§Task List, "Not tasks"; `handoff-context.md` §Notes "deliberately not updated");
`docs/`, `.agents/`, `invariants/` and `.sessions/` — non-code. **Nothing downstream of a changed module
was dropped, and the scope was not narrowed after any failure — nothing failed.**

**Steps 1-2 — the integration/E2E layer over that scope:**

| Run | Command | Result |
|---|---|---|
| `MS-001` gate (also Check 6's execution) | `.venv/bin/python -m pytest tests/qa/test_qap_ms001.py -o addopts= -v` | **11 passed** in 2.65 s |
| `MS-002` + `MS-004` gates on these bytes (one command) | `.venv/bin/python -m pytest tests/qa/test_qap_ms002.py tests/qa/test_qap_ms004.py -o addopts= -q` | **21 passed** in 60.17 s |

Gate tree total at this boundary: **32 process-level gates green** (11 + 15 + 6) plus **455 hermetic
tests**, all on the one snapshot `6508941`. The two neighbor gates had to be re-run rather than
inherited: 077 changed `tests/qa/harness.py`, the file all three passes share (a shared-signature
change: `Proxy.url`/`Proxy.post`), and "changed snapshots or configurations require fresh applicable
execution". Both green, which independently confirms the drain broke neither neighbor — 077's counts
re-measured here (23 / 18 mentions of `assert_upstream_response` including the import, in `ms002` /
`ms001`) match its claim of 22 / 16 call sites.

**Shared-run rule applied.** This single independently executed invocation satisfies Checks 6 and 7: its
runs cover both complete required scopes on the same snapshot and the same harness configuration —
Check 6's required scope is `QAP-MS001`'s 11 gates with the four `VP-*` handles; Check 7's is the
hermetic suite plus the whole `tests/qa/` tree. No test was run twice to produce two reports, and
Forward QA's 077 run was not counted as this pass's execution. The session-close full suite remains
separately owed.

## Invariant Cross-Reference (scope: the cycle's affected packages)

Narrowed to `buster_ripper.dump` and `buster_ripper.server` (the milestone's packages); read only
`invariants/buster_ripper/dump/invariants.md`, `…/server/invariants.md` and the registry header. No
session-level `invariants.md` exists for this session; there are zero `active`/`wip` entries — all 8 are
`planned`. Classification of this milestone's findings: **Neutral** — none contradicts a registry
statement; two registry statements are *witnessed thinner* than their own text claims, and one registry
gloss is now stale (→ P3-1, P3-2).

Planned-invariant witness check for the four VPs due here:

| Handle | Witness due at MS-001 | Result |
|---|---|---|
| `PI:T002/dump_format/default_format_is_documented` | `TC-107` | **pass** |
| `PI:T002/dump_format/documented_format_is_retained` | `TC-108` | **pass** |
| `PI:T002/dump_format/format_is_per_artifact` | `TC-100` (collapsed side; `TC-102` completes it at `MS-002`) | **pass** |
| `PI:T002/dump_format/one_write_path_per_turn` | `TC-100` + `TC-108` (`F` vs `2N`, never `2N + F`) | **pass** |
| `PI:T002/dump_format/format_set_is_closed` | out-of-set rejection (R1 gate + `test_dump.py:110`, `test_main_cli.py:55`) | **pass** for the rejection clause; the "exactly three values" clause has **no witness** → P3-1 |
| `PI:T002/dump_format/format_does_not_change_write_count` | `TC-105` — `MS-002`'s, not due here | `not_applicable` |

**`Override:` entries — fully applied?** All three are `Decision: preserve`, i.e. the old property must
survive *scoped*, and all three were verified in code and evidence, not in prose:
`OI:…/output_layout` (documented layout intact at `2N` when selected or unconfigured — `TC-107`/`TC-108`),
`OI:…/sequential_diff_chain` (`turn`/`prev_text` advance at append, never at flush —
`dump.py:108-109`, and `flush_all`'s done-when forbids touching them),
`OI:…/turn_zero_is_full_body` (turn 0 carries the JSON body **and** the rendering in every format —
`_turn_zero_block`, `dump.py:229-234`). **No override was applied incompletely and none was exceeded** —
the removal half of the original `OI:…/output_layout` is still absent, as `US-012` requires.

## Deterministic Closure (pre-question gate applied to both P3s)

```text
CLOSURE: absorbed
Finding: 078-P3-1 — the session's own invariant `PI:T002/dump_format/format_set_is_closed`
  (invariants/buster_ripper/dump/invariants.md:199-210) states "The accepted format set is exactly three
  values", but no witness pins the set's contents: the two tests that name `DUMP_FORMATS`
  (`tests/test_dump.py:110`, `tests/test_main_cli.py:55`) exercise an out-of-set value only, and
  `test_help_lists_every_format_and_default` iterates whatever the tuple holds. Meanwhile `main.py:80`
  derives the CLI's accepted set from that same tuple and `dump.py:106-107`'s final dispatch arm is an
  `else`, so a fourth member would be accepted by the CLI and silently written by the **self-contained**
  record builder, under a correctly-named `<4th>.log`, with no test able to fail.
Required outcome: the gate surface notices an extension of the set whose size the invariant names as
  "the irreversible part of this decision" (its own `Why:`), so a fourth format cannot be silently
  absorbed by the wrong writer.
Authority: (1) approved plan / `US-014` — the invariant is one of the six confirmed `PI:T002/dump_format/*`
  entries, so the required property is settled, not proposed; (2) production contract —
  `config.DUMP_FORMATS` is the closed set and `dump_turn`'s dispatch is the consumer; (3) ownership — a
  witness in `tests/` over an already-accepted surface is unit-tier test work, not a milestone QA gate
  (the `tests/qa/` scenarios are exactly `TC-100/001/107/108`, closed); (4) `codify-test-cost` rule 3 —
  one pin in the module that already owns the CLI/selector behavior, no new module.
Admissible candidates: one — a test asserting `config.DUMP_FORMATS == ("documented", "change-focused",
  "self-contained")`, co-located with the existing selector tests in `tests/test_main_cli.py`.
Rejected candidates:
  - rewrite `dump_turn`'s `else` as `elif fmt == "self-contained"` + a final `raise` -> knowingly cannot
    fail: `_resolve_format` already raises for any member-outside-the-set value, so this adds an
    unreachable arm, touches production code and its LNCP profile for zero observable protection, and
    leaves the invariant's "exactly three" clause as un-witnessed as before. Strictly larger, no benefit.
  - close the gap at the expensive tier with a new `tests/qa/` gate -> broadens MS-001's approved scenario
    set and needs plan authority; a scenario gate adds nothing the unit pin does not.
  - amend the invariant's wording to drop "exactly three values" instead -> preserves the verified gap by
    deleting the requirement that exposed it, and rewrites a `US-014`-confirmed statement.
  - promote `PI:…/format_set_is_closed` from `planned` to `active` in the same touch -> status transitions
    follow phase acceptance and belong to the close, not to a witness-adding item.
Selected closure: the pin test above, recorded as a `quality-backlog.md` row.
Risk shape: unchanged for production (zero production bytes move); the witness surface grows by one
  assertion in a module that already owns the selector's contract.
Deferred additive work: none — the pin is the whole remediation; promoting the handle's status is the
  close's own, already-scheduled job.
Owning route: `Forward` (unit tier), trigger: next touch of `buster_ripper/dump.py`'s format dispatch or
  `buster_ripper/config.py:DUMP_FORMATS`. Non-blocking: no accepted behavior is wrong today.
Question required: no
```

```text
CLOSURE: absorbed
Finding: 078-P3-2 — `invariants/buster_ripper/dump/invariants.md:45` `E:T002/dump/bulk-flush`'s Gloss ends
  "No caller until T003's triggers land." `T003` landed at Checkpoint 66 and `flush_all` now has three
  production callers (`server.py:112`, `dump.py:133`, `dump.py:206`), verified at §Check 1 — so the
  permanent registry states a present-tense fact that is false, and it is the registry row for the very
  function whose absence `MS-001`'s `022` scoping turns on.
Required outcome: the durable registry does not assert a stale present-tense fact about the wiring of a
  function whose deferral is this milestone's central claim.
Authority: (1) the registry's own header (`invariants/index.md:3-5`): "If code and registry disagree,
  code wins and the registry must be refreshed"; (2) the precedent and owner already on file —
  `QB:assumption-registry/t006-tls-referent-elevation` (open, from 056-yield-phase §Findings P3-3) routes
  exactly this class of registry hygiene to the **Authority Registrar** at "this cycle's post-Yield-Red
  finalization pass", and its remedy already includes re-running `speeckle/scripts/rebuild_index.py`, which
  additionally clears the same file's stale "TLS referent remains deferred to T006 Yield" sentence;
  (3) `codify-scope` — projection of plan/registry authority is not Forward's; (4) the triad rule — this
  pass creates and edits no invariant record.
Admissible candidates: one — amend that one Gloss clause and fold it into the open row's existing
  trigger/window rather than minting a second lingering item.
Rejected candidates:
  - leave it because the sentence was true when written (2026-09-22) -> preserves the false fact; the
    registry is durable and outlives the increment that made the wording local-true.
  - delete the `E:T002/dump/bulk-flush` Entity -> removes a binding the working tree resolves; it is the
    wording that is wrong, not the entry.
  - implement it here by editing `invariants/` -> bypasses the authorized owner (Registrar) which has a
    viable route, and breaks the triad rule this boundary is holding.
  - open a separate `QB:` row with its own trigger -> duplicates an open row's owner and trigger and adds
    an item that can linger; the smallest complete remediation is an amendment to the row that already owns it.
Selected closure: amend the Gloss, and amend `QB:assumption-registry/t006-tls-referent-elevation` to name
  this second clause, so one Registrar dispatch carries both.
Risk shape: unchanged — record-only, zero code or test bytes.
Deferred additive work: none.
Owning route: `authority-registrar`, trigger already scheduled: this cycle's post-Yield-Red finalization
  pass (shared with the existing row). Non-blocking for MS-001.
Question required: no
```

No finding required a user question. `FK-011` (preserved-thinking serve default) is the session's only
open Fork, is untouched by this boundary, and is not reopened by anything here: `MS-001`'s gates run the
binary with `PRESERVED_THINKING` at its shipped `False` (`config.py:50`), and `session.py:93-98` shows
the sid derivation cannot be affected by the injection either way.

## Routing: owned factual claims

Enumerated every atomic entry under `Factual claims` in `decisions.md` carrying
`Verification: yield-phase:MS-001`: **exactly one** — `US-009` claim 2
(`decisions.md:791-792`), already adjudicated **HOLDS** at [071](071-yield-phase.md) §Routing/§Check 6.
There is **no** `yield-phase:MS-001 PENDING` row anywhere in the ledger
(`grep -n "PENDING" decisions.md` → two matches, neither a claim: `decisions.md:888` explicitly records a
proposition as deliberately **not** promoted to a PENDING claim, `decisions.md:1246` is a governance note
about the legacy report's own pending rows). Legacy-record inspection: `US-009` carries the canonical
`Factual claims` field and `handoff-archive.md`/`assumptions/claims.md` every reference to it names this
same arbiter row, so **no `MIGRATION-REQUIRED` return** is owed.

Because the ledger row is HEAD-scoped and the tree moved 22 commits since (`c7420ce` → `6508941`), the claim was
**re-established first-party on this snapshot rather than inherited**: `require_upstream()` — the same
fail-closed preflight every gate here runs through — passed inside all 11 gates; `GET
/v1/models` answered `200` at pass start; and `TC-001`'s control session received the upstream's own
typed `404` for the invented model (`assert_upstream_response` accepted a `< 500` peer answer while
`poll_until` found the proxy's own `PermissionError`), i.e. the pass is not credit against a
connection-error baseline. Transcript: [`../evidence/ms001-reentry-078-gate-run.txt`](../evidence/ms001-reentry-078-gate-run.txt).
**`HOLDS` stands; no ledger edit is owed** (the row's Evidence locator still names a durable artifact,
and this pass adds a second, not a correction).

## Findings

### [P3] Format-set closedness has no witness for its "exactly three values" clause
- **Priority**: P3
- **Check**: 4 Consumer Contract Tracing (+ invariant cross-reference)
- **Location**: `buster_ripper/dump.py:102-107`, `buster_ripper/config.py:27`, `tests/test_main_cli.py:55-72`
- **What is wrong**: the session's own planned invariant claims the accepted format set is *exactly* three
  values and names that size as the irreversible part of the decision, but only the rejection half is
  witnessed. `main.py`'s `click.Choice` is derived from the tuple and `dump_turn`'s final arm is an
  `else`, so a fourth member is accepted and routed to the self-contained writer silently.
- **Evidence**: `grep -rn "DUMP_FORMATS" tests/` → 3 hits, all iteration/rejection-shaped
  (`test_main_cli.py:69` `for fmt in config.DUMP_FORMATS`; `:56`, `test_dump.py:111` docstrings for the
  out-of-set case). No assertion anywhere compares the tuple to a literal.
- **Data path affected**: `main --dump-format` → `config.DUMP_FORMAT` → `dump_turn` dispatch →
  `_self_contained_record` → `DUMP_DIR/<sid>/<fmt>.log`
- **Reach**: **latent** — unreachable in any shipped configuration today (the set has three members and
  `_resolve_format` refuses everything else); fires on the first edit that adds a member without touching
  the dispatch. Input-dependent: it is a future-change hazard, not a current defect.
- **Disposition**: `CLOSURE: absorbed` — one pin test, owner `Forward`, non-blocking. Full packet above.

### [P3] Registry gloss asserts `flush_all` still has no caller
- **Priority**: P3
- **Check**: invariant cross-reference (registry hygiene) / Check 1 evidence trail
- **Location**: `invariants/buster_ripper/dump/invariants.md:45` (`E:T002/dump/bulk-flush` Gloss)
- **What is wrong**: the sentence "No caller until T003's triggers land" was local-true at T002's
  registration and is false now; the registry is the durable artifact, and this is the function whose
  uncalled state `MS-001`'s `022` scoping is built on. A reader auditing `MS-001` from the registry would
  re-open a closed question.
- **Evidence**: §Check 1 caller table (`server.py:112`, `dump.py:133`, `dump.py:206`); `invariants/index.md`
  carries the same class of staleness for the TLS referent, already rowed and open.
- **Data path affected**: none — inert wording; no code or witness consumes the Gloss.
- **Reach**: **inert wording** — the registry is read by agents and people, never by the runtime.
- **Disposition**: `CLOSURE: absorbed` — amend the Gloss and fold the clause into the already-open
  `QB:assumption-registry/t006-tls-referent-elevation`; owner `authority-registrar`, trigger already
  scheduled. Full packet above.

## Additional review findings (if supplied)

None supplied with this dispatch. The post-071 `codex-review` CR2 packet
([`../evidence/codex-review-post-071-packet.txt`](../evidence/codex-review-post-071-packet.txt)) is the
external review that blocked this boundary; its four findings were absorbed as `T015`/`T016`
(`DC-CR2-F-1-r1`…`DC-CR2-F-4-r1`) and this pass re-verified their integration effect on `MS-001`'s
surface rather than trusting the wave's own record: §Check 1's de-leak rows, §Check 2's parity pair 4,
§Check 3's Path C, §Check 4's `502` producer row. Of its two non-findings, `CR2-X-2` keeps its owner
(`README.md:240` stale FastAPI wording → `QB:cli-surface/…`, roadmap §RM-005) and `CR2-X-1`'s malformed
collector return stays with the collect lane; neither is a `MS-001` gate obligation and neither is
reopened here. (Re-checked `CR2-X-2` on current bytes rather than inherited: `README.md:241` still lists
`app.py  FastAPI app + lifespan` in the architecture tree, and no `buster_ripper/app.py` exists — inert
wording, owned by `QB:cli-surface/undocumented-flags-and-stats-db` → `roadmap.md` §RM-005, and no code or
witness consumes that block.) The four production files the post-071 waves *did* re-thread
(`messages.py`, `chat.py`, `passthrough.py`, `utils.py`) were re-verified first-hand above, so 071's
verdict needed no Codex re-run for this milestone's surface itself.

## Integration Verification Summary

**CODIFY Session ID**: `02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573`
**Session Directory**: `.sessions/codify/02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/`

- **Check 1 (Caller Verification): PASS** — every MS-001 entry point has a production caller
  (`dump_turn` ← `messages.py:56`; `flush_all` ← ×3; `poll_flush` ← `server.py:45` ← `on_startup`;
  `--dump-format` ← console script + serve script); all four removed/replaced wirings verified replaced or
  genuinely dead (bare assert, `path=` keywords with **0** call sites, body limb, `str(e)` bodies).
- **Check 2 (Migration Parity): PASS** — the documented writer is side-effect-for-side-effect complete
  against `buster_ripper_legacy.py:558-615` (8/8 effects, same order → `2N`); the two intentional deltas
  are documented; the drain lost no property the helper proved.
- **Check 3 (Data Flow Completeness): PASS** — producer→artifact, all three flush triggers, and
  `TC-001`'s relay leg re-traced across the post-071 delta; double sid derivation proven unable to
  diverge; the counted entry is proven to sit on the real write path; no silent-drop on the fault path.
- **Check 4 (Consumer Contract Tracing): PASS + 1 P3** — every dispatch-bearing field's value set is
  handled by its consumers; coalescing preserves per-record identity and never spans sessions; the
  closed-set *size* clause is un-witnessed (P3-1).
- **Check 5 (Plan Fulfillment): PASS** — `T001` 3/3, `T002` 10/10, `T005` 6/6 clauses from the
  acceptance-time text (`git show` Checkpoints 61/102/109); 4 required removals taken, 2 forbidden
  removals confirmed absent; all six `MS-001` `Complete when` conjuncts witnessed on artifacts produced at
  this boundary. Additions-and-retention goal, correctly not an ownership transfer; nothing relayed.
- **Check 6 (QA Milestone Closure): PASS** — `QAP-MS001`'s four VPs / four scenarios independently
  executed: **11 passed**; every contract element re-checked with the drained oracle in place, the `022`
  directive still honored literally, the drain ruled **not** reopening any approved VP; intermediate sweep
  clean (no `MS-001` intermediate; the shared helpers are single-defined).
- **Check 7 (Integration Suite): PASS** — scope derived from the diff before running: hermetic
  **455 passed @ 95.75 %** (`dump.py` 100 %) + **all 32 process-level gates green** (11 / 15 / 6) on the
  one snapshot `6508941`; exclusions named (`buster_ripper_legacy.py` by plan, non-code trees); no
  narrowing after failure — nothing failed; shared-run rule applied to Checks 6+7.

## Verdict

**SEMANTIC-APPROVE — `MS-001` / `QAP-MS001` milestone gate PASSES on current bytes.**
Zero P1, zero P2. Two P3 findings, both `CLOSURE: absorbed` with a named owner and no user question
(`Forward` ×1 unit-tier pin test; `authority-registrar` ×1 registry gloss, folded into an already-open
row). Routing: **proceed to the assumptions recheck and Yield Red** — this pass does **not** append a
Delivered chronicle entry, publish child completion, or mark the phase accepted; `resolved` for
`T015`/`T016`/`T017`, phase acceptance and the close's `Delivered` shape remain Yield Red's, per
`handoff-context.md` §"What the MS-001 boundary still owes". No `HEAD` advancement, no push, no squash:
the full boundary close (assumptions → Yield Red → `phase-boundary-finalize`) is still owed, and the
session-close full-suite gate remains separately owed on top of it.

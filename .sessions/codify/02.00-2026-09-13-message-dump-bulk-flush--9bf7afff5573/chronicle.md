# Session Chronicle — 02.00-2026-09-13-message-dump-bulk-flush
- **2026-09-27** — **T008.001 accepted locally**. A raw chat stream error now preserves upstream Content-Type with status and byte-exact body (including absent declaration), without changing successful SSE or nonstream +json paths; hermetic suite 250/250. Fresh S9 composition and MS-002 full process QA/re-verdict remain downstream. → [046-01-yield](iterations/046-01-yield.md)
- **2026-09-27** — **T008 delivered and accepted locally** (045-01-yield). Chat `stream:true` dispatches before upstream JSON parsing and relays SSE bytes unchanged; `stream:false` retains compatible `+json` transforms. Hermetic suite 248/248; claim-63/S9 post-implementation recheck and MS-002 process re-verdict remain downstream. → [045-01-yield](iterations/045-01-yield.md)
- **2026-09-27** — **T007 delivered and accepted locally** (044-01-yield). A five-second per-request `/v1/models` deadline bounds the uncached probe despite the shared 600-second client; timeout and ordinary errors cache unknown, cancellation propagates, and override/cache avoid GET. Hermetic suite 242/242, backend coverage 100%; MS-002 process re-verdict and assumption recheck remain downstream. → [044-01-yield](iterations/044-01-yield.md)

- **2026-09-27** — **T003.001 accepted-local** (043-01-yield). Client request JSON decode failures now return typed 400 on messages, chat and count_tokens; syntax fallback at count_tokens and upstream chat 502 attribution remain distinct. Hermetic 238/238 and focused process R1 pass; post-implementation S7 composition SAFE, so its finding is resolved. Full MS-002 process QA and milestone re-verdict remain downstream. → [043-01-yield](iterations/043-01-yield.md), [S7](assumptions/compositions/seam-S7.md)

- **2026-09-27** — **T010.001 accepted-local** (042-01-yield). Retained aiohttp runner cancellation no longer aborts cleanup when awaiting expected child cancellation: baseline task cancellation count distinguishes a fresh outer request. Both children cancel and pending collapsed artifact flushes before close in local runner regression; fresh cancellation propagates. Hermetic suite 227 green, 92.88%. Real SIGINT/SIGTERM QAP and S2 recomposition remain downstream. → [042-01-yield](iterations/042-01-yield.md)

- **2026-09-17** — Planning closed; the plan is approved as implementable ([`US-015`](decisions.md#us-015); [round-18 gate](iterations/020-interview.md#round-18--the-answer-to-the-final-approval-question-and-the-gate-it-runs)). **6 tasks** (`T001`–`T006`),
  4 milestones (`MS-001`–`MS-004`), 20 validation points, 15 QA scenarios. **Approved scope**: buffer captured
  messages in memory and flush the dump in bulk (volume threshold, interval timer, and a shutdown flush on
  SIGINT/SIGTERM) behind a closed three-value format selector whose default is today's documented per-turn
  layout — so the write reduction is **opt-in**; plus an independent upstream-TLS verification opt-out
  (`--skip-tls-verify`) for self-signed local deployments. → [plan-summary.md](plan-summary.md)

- **2026-09-22** — **T001 delivered and accepted** (031-01-yield). `SessionState` gains the per-session pending
  buffer (`pending: list[str]`, `pending_bytes: int` per FK-B's `len(_render_body(body))` measure);
  `dump_turn` appends each turn's rendered record and accumulates the byte measure unconditionally at append
  time, beside `turn`/`prev_text` advancement. Purely additive: the documented write sites and `index.txt`
  stay live; nothing drains until T003's triggers. 16 new unit tests (append, accumulation, isolation,
  advance timing, additivity); suite 143 green at 91.42%, `dump.py` 100%. Yield adjudications: T001's
  "encoded record" = the rendering (T002 owns the divergence); `dump_turn` L growth 42→44 plan-sanctioned
  with the extraction owned by T002. → [031-01-yield](iterations/031-01-yield.md)

- **2026-09-22** — **T002 delivered and accepted** (031-02-yield). The dump format became selectable:
  `config.DUMP_FORMATS = ("documented", "change-focused", "self-contained")` with `documented` as the
  default (`--dump-format`, callback-wired, `main()` gains no parameter); `dump_turn` branches on the
  configured format — the documented writer (three write sites, byte-preserved) or one of the two collapsed
  record builders (change-focused: turn-0 body+rendering then per-turn diffs; self-contained: every turn's
  full rendering) — with the F-21 collapsed-only buffer append and the F-5 `DUMP_DIR` ValueError guard.
  `flush_all` appends each session's pending batch to `DUMP_DIR/<sid>/<format>.log` in ONE write (no caller
  until T003's triggers); R2 symlink/non-regular refusal guards every write target; R5 `(sid, format)`
  artifact identity pinned. The T001-mandated capacity extraction landed before additive behavior
  (`dump_turn` L:44→ideal, `dump.py` 284 Green); `norecursedirs` (pytest defaults + `qa`) keeps the bare
  suite hermetic. Suite 189 green at 92.72%, `dump.py` 100%. Yield: claim-19 replay PASS (bounded probe) →
  accepted-local; 4 deferred registry entities registered (bulk-flush, format-selector, summary-line,
  collapsed-artifact); 3 defer-tier findings in the quality backlog.
  → [031-02-yield](iterations/031-02-yield.md)

- **2026-09-22** — **T003 delivered and accepted** (032-01-yield). `flush_all` gained its three callers: the
  append-time volume check (inside `_append_record`, after the append, reading the `pending_bytes` it just wrote —
  collapsed-path-only by construction), the periodic coroutine `dump.poll_flush` (flush then sleep, per-iteration
  exceptions logged and the loop continues, cancellation landing at the sleep await), and the shutdown seam —
  `on_cleanup` now cancels both background tasks, then runs a wrapped `flush_all` that cannot raise, then closes the
  shared session; `spawn_dump_flush`/`app["dump_flush_task"]` mirror the kv-monitor lifecycle. Trigger parameters
  `DUMP_FLUSH_INTERVAL` (10 s) / `DUMP_FLUSH_BYTES` (4 MiB rendered-text) exposed as callback-wired CLI flags.
  Backlog absorbed: duplicate `server.main` deleted; `SessionState.fmt` captures the buffer-time format so a mid-run
  format mutation can no longer misfile records. Suite 210 green at 92.87%, `dump.py` 100%. Yield: claim-18 replay
  PASS (local analogues) → accepted-local; two deferred registry entities registered (flush-timer,
  flush-trigger-parameters); 4 defer-tier findings in the quality backlog. MS-002's constituent task set complete.
  → [032-01-yield](iterations/032-01-yield.md)

- **2026-09-22** — **MS-002 boundary base-sync: unchanged** (033-01-forward). Target `main @ 72e3a1c` fetched from
  origin and verified already an ancestor of HEAD (Checkpoint 67) — 96 ahead / 0 behind, so no merge was performed,
  no re-threading owed and no prework packets emitted (no sync delta exposed work); suite evidence inherited from
  032-01-yield at the bit-identical production tree (no production file changed; not re-run per the no-re-run rule).
  Session-only checkpoint. MS-002 awaits its Yield-Phase/QA boundary; next forward T004.
  → [033-01-forward](iterations/033-01-forward.md)

- **2026-09-22** — **MS-002 Yield Phase REJECTED (rejection #1)** (035-yield-phase). BASE-SYNC re-verified at
  HEAD `f803b04`; Checks 1–5 PASS (callers, documented-path parity, data flow, dispatch contracts, plan
  fulfillment); QAP-MS002 independently re-executed — all 10 approved scenarios PASS (14/15 gates; hermetic
  210/210 @ 92.87%). One **P1**: R1 parse-boundary defect — non-UTF-8 `/v1/messages` body → unhandled 500
  `UnicodeDecodeError` (`routes/messages.py:26-29`; same class `chat.py:28-30`, `count_tokens.py:30-32`)
  where the empty-body precedent returns a typed 400; routed to implementation Forward as adhoc **T003.001**
  (failing R1 gate test is the regression test), plus codex-review absorptions **T007** (probe timeout) and
  **T008** (streaming relay regression). **G-008(b) comparison record written**
  (`evidence/ms002-comparison-record.txt`: F=2 vs 2N=12 ≈ 83.3% write-op reduction, opt-in S-1; files 1 vs 7;
  witnesses A/B/C excluded; M1-proxy caveat). Boundary evidence persisted review-preserving by 036-01-forward
  (phase-boundary-evidence mode; Checkpoint 70) before the repair route advances HEAD. Next: Forward remediates
  T003.001 → QAP re-run (15/15 expected) → MS-002 re-verdict → T004.
  → [035-yield-phase](iterations/035-yield-phase.md), [036-01-forward](iterations/036-01-forward.md)

- **2026-09-27** — **T009 delivered and accepted locally** (041-02-yield). Corrected collapsed per-session volume accounting to UTF-8 bytes of rendered text with inclusive immediate threshold in both formats; moved sizing before pending-state mutation so a JSON-valid lone surrogate cannot poison a later valid turn. Independent both-format recovery replay and hermetic suite (225 green, 92.88%) passed. Assumption claims/seams remain accepted-local pending post-implementation recheck; CLI/process proof stays milestone-owned. → [041-02-yield](iterations/041-02-yield.md)

- **2026-09-27** — **T010 delivered and accepted locally** (041-02-yield). Graceful cleanup isolates an ordinary completed KV or dump-timer task failure while cancelling its peer, flushing the pending dump before session close, and preserving outer cancellation. Both slots passed local tests; live SIGTERM/SIGINT proof remains MS-002 QA-owned. → [041-02-yield](iterations/041-02-yield.md)

- **2026-09-28** — **T005 delivered and accepted locally** (049-01-yield). README dump section extended addition-only (+53/−0; documented block byte-identical): collapsed `<format>.log` shape, `--dump-format` selector with default `documented`, per-format grep/less reading workflow matching dump.py's summary/marker formats byte-level, run-path example, F-8 arrival-order caveat, old-layout note. `serve_buster_ripper.sh` wires `DUMP_FORMAT` beside its `DUMP_DIR` peer (+3). claim-32 replay independently executed via argv shim: exactly one `--dump-format` when set, none when unset/empty, no positional forwarding, peer mapping untouched → accepted-local. Suite 250/250 @ 93.87%. No VP (structural); MS-001/QAP-MS001 scope. → [049-01-yield](iterations/049-01-yield.md), [049-01-forward](iterations/049-01-forward.md)
- **2026-09-28** — **claim-32 resolved** (assumption-decide, post-acceptance resolution; DC-claim-32-r1). The finding completes its lifecycle on recorded evidence: run-19 atomic rechecks (rows 159–164 + 33, all HOLDS) plus the post-implementation S6 composition SAFE over the T005-landed serve-script/README surface; both source fingerprints re-verified unchanged at resolution (basis `git-commit:168b078`). Disposition stays `absorbed`; no ledger correction. S6 stays open for T006's `SKIP_TLS_VERIFY` half (landing triggers recomposition); the MS-002 QA re-verdict remains next and unaffected — resolution grants no milestone acceptance. → [S6](assumptions/compositions/seam-S6.md), [049-01-yield](iterations/049-01-yield.md)
- **2026-09-28** — **US-016: user approves Forward STOP route 2 — re-order ahead of the blocked MS-002 re-verdict** (registrar batch). Verbatim "reorder and finish coding first". Effect: T004 and T006 authorized to run BEFORE the MS-002 Yield Phase re-verdict (the 035 T004 gate is waived in execution order only); the QAP-MS002 re-run + MS-002 re-verdict remain owed whenever `hyperipper.local:30001` is reachable (blocked: DNS `gaierror`, Checkpoints 98/99, 048-forward-qa — the 035 rejection stands until the re-verdict runs); no product scope, goal, or milestone criterion changes; the disclosed rework risk (a later MS-002 finding may require T004 test adjustment) is accepted. Projections: decisions.md §US-016, plan-summary §Current Phase/§Progress Overview/§Last Updated re-ordered, handoff §Current Status + registrar forward-readiness. Registry untouched (0 new/updated/superseded/upheld; execution order is not invariant content). Next: forward T004 → T006. → [US-016](decisions.md#us-016)

- **2026-09-28** — **T004 delivered and accepted locally** (050-01-yield). Dump cluster's hermetic witness set completed, test-only diff (+775/−105 across 5 test files, zero production changes): `test_dump_write_count.py` (settled-unit instrument — one `Path.open("a"/"w")`+`write` hook, mkdir/SQLite excluded; one write per flush of K≥2 with payload bytes per collapsed format; injected-F batch-only appends proving witness-B exclusion; documented 2N; paired G-008(b) measurement F=1 vs 2N=8 at N=4; file count 1 vs N+1), `test_dump_records.py` (record shapes moved from test_dump.py per QB:test-dump/file-size + artifact-level readability/turn-0 witnesses read from the flushed artifact), flush session-isolation, route-level dump-failure isolation (raising `dump_turn` → upstream 200). Yield independently replayed claim-3/claim-93 protocols at the `io.open` instrument layer (24/24; one initial probe FAIL traced to the probe's own double-run bug, not product) and reproduced the suite 268/268 @ 93.87%, `dump.py` 100%. Claims 3/20/93 accepted-local pending S1/S2 recomposition; MS-002 re-verdict still owed on reachability per US-016. → [050-01-yield](iterations/050-01-yield.md), [050-01-forward](iterations/050-01-forward.md)

- **2026-09-28** — **T006 delivered and accepted locally** (051-01-yield). Upstream TLS verification opt-out, +429/−34 across 15 files: `config.SKIP_TLS_VERIFY: bool = False` (verification on by default, never inferred from scheme); `--skip-tls-verify` assigned through its option callback at parse time so `main()`'s parameter list is unchanged; `on_startup` builds the package's only `ClientSession` on an explicit `TCPConnector(ssl=not config.SKIP_TLS_VERIFY)` inside the running loop — one connector, one posture, every outbound call (grep-audited: single session, zero per-request `ssl=` overrides); serve script wires `SKIP_TLS_VERIFY` on the boolean `== "1"` guard (unset/0/x never invert the default posture); README documents the opt-out; committed self-signed `127.0.0.1` QA fixture (CN=issuer=127.0.0.1, SAN IP, untrusted-issuer mode, valid to 2126, key↔cert modulus match) as QAP-MS004 peer material; 6 posture unit tests + `QB:dump-flush/trigger-value-validation` absorbed (`IntRange(min=1)` on both flush triggers — zero/negative now `UsageError` before the global, no hot flush loop). Plan-mandated `main()` extraction: `_apply_cli_config`, scanner-verified L 52→46 with P:21 held. Yield executed both claim replays (claim-38: both postures from one connector construction, session binds connector loop, no `RuntimeError` — PASS; claim-34: parse-time opt-out + import-bound-defaults counterexample CONFIRMED; statuses accepted-local) and reproduced suite 278/278 @ 93.88%. 2 P3 deferrals (`QB:test-tls-posture/private-attr-pinning`, `QB:deps/aiohttp-unpinned`). Next: MS-004 QA author pass (forward-qa) + S6 recomposition / S4 refresh (Forward Collect); MS-002 re-verdict still owed on reachability per US-016. → [051-01-yield](iterations/051-01-yield.md), [051-01-forward](iterations/051-01-forward.md)

- **2026-10-04** — **FK-010 · serve-script disable form resolved (option 1) and registered — T011 opens** (registrar batch;
  wave FK-010, order 1 of 1 — the settled-impacts flush; all alignment waves complete, no other fork open). The user's answer
  to the serve script's never-working `DUMP_DIR=""` disable form, verbatim: *"fix the bug, make it to not lie. I do not need the
  dumps always."* → the script must **honor** the form its header documents (option 1, picked unmodified): `serve_buster_ripper.sh:28`'s
  `${DUMP_DIR:-…}` becomes the unset-only `${DUMP_DIR-…}`, so an explicitly empty `DUMP_DIR` stays empty, `:36`'s `-n` guard is
  false, `--dump-dir` is not passed, and the binary runs at its `config.DUMP_DIR = None` off-mode; unset keeps today's default,
  header line 14 and every README example stand. Projections: decisions.md §FK-010 Impact (Plan → landed; Registry/Roadmap none),
  plan-summary §Implementation Plan Summary (**T011 · Honor the serve script's documented `DUMP_DIR=""` disable form**, `pending`,
  `scoped:S6`), §Current Phase, §Session Metadata, §Progress Overview, §Open Questions item 10; handoff §Planned Task Context
  (durable task context + compact verification profile), §Current Status, §Open Forks (pending-projection entry cleared).
  Registry untouched (0 new/updated/superseded/upheld, no elevation, no index rebuild; the `CO:serve-script-env-to-argv/…` update
  the fix implies is an `assumptions/` recomposition inside the new task). Pre-existing registry health diagnostics (7, at HEAD, `invariants/`
  clean) recorded as observations for their owner, not cured here. Script basis re-verified at registration: still
  `sha256:f4076d1c…`, 46 lines — the fix is **not** yet implemented. The task is dispatchable; it gates no milestone and no further user
  decision is owed. Next milestone work unchanged: **MS-004 QA author pass (forward-qa)**; the reachability-gated QAP-MS002 re-run →
  MS-002 re-verdict stays owed. → [FK-010](decisions.md#fk-010), [020](iterations/020-interview.md) §Round 21

- **2026-10-05** — **T011 delivered and accepted locally** (052-01-yield). The serve script now honors the `DUMP_DIR=""` disable form
  its own usage header promises, settling `FK-010` option 1 ("fix the bug, make it to not lie. I do not need the dumps always."):
  `serve_buster_ripper.sh:28`'s default becomes unset-only (`${DUMP_DIR-${HOME}/mllm/prompt-diffs}`), so an explicitly empty value
  survives to line 36's `-n` guard, `--dump-dir` is never appended, and the proxy runs at its `config.DUMP_DIR = None` off-mode
  (`main.py:78` `default=None` → `:151` assigns only inside `if dump_dir:` → `config.py:20` → `dump.py:108` early return); unset keeps
  today's default, header line 14 and every README example stand (0 header lines in the diff), and lines 32–38/46 are untouched — that
  assembly is T012's wave. Diff: 1 production line + `tests/test_serve_script_argv.py` (new, 137 lines, 5 hermetic tests — PATH shim
  records the assembled argv, no server, pinned never-created `HOME`, every wrapper-read var scrubbed; whole-list equality pins the
  three states, the `--dump-format` peer, the always-on trio, and no-positional-forwarding; a second class asserts the header promise
  verbatim so "fixing" it by deleting the doc fails the suite). Script `f4076d1c…` → `d5db2144…` (46 lines both sides); capacity gate
  N/A earned by measurement (no additive production surface), LNCP scan clean, zero gray-zone metrics, mypy/ruff N/A (not installed,
  no project config). Yield independently replayed all eight argv cases outside the test module — including the two arms nobody had
  ever reached (line 36 false; the all-suppressed zero-word arm of line 46 under `set -euo pipefail`, rc=0) and both sides of the C4
  `${HOME}` premise (empty+unset-HOME now runs; unset+unset-HOME still aborts, unchanged) — on bash 3.2.57, the only interpreter here
  and the one `env bash` resolves to. Suite reproduced **283 passed @ 93.88%** (278 baseline + 5 new, no skips). Zero reject-tier
  findings; 1 P3 deferred (`QB:serve-script-argv/zero-word-arm`, picked up by T012's own touch-set); two report-accuracy observations
  (backlog count 9 vs 12 unchecked; claim-195 pointer `:79` → `:78`). `seam-S6-disable-form` → **accepted-local**: members 159–163,
  179, 180, 191–193 need fresh verdicts on `d5db2144…`, claim-199 needs re-extraction, and a fresh `seam-S6` recomposition must land
  before S6 is citable or T012 is dispatchable. Gates no milestone: **MS-004's QAP-MS004 author pass (forward-qa) remains the next
  milestone work**, and the reachability-gated QAP-MS002 re-run → MS-002 re-verdict stays owed per US-016. → [052-01-yield](iterations/052-01-yield.md), [052-01-forward](iterations/052-01-forward.md)
- **2026-10-05** — **`seam-S6-disable-form` resolved; `CO:` basis disposed by route** (assumption-decide, post-acceptance resolution; DC-seam-S6-disable-form-r1).
  The recheck T011's landing fired is delivered: ten script-pinned members (159–163, 179, 180, 191–193) re-verified fresh on `serve_buster_ripper.sh`
  `d5db2144…`, claim-199 re-extracted to rows 200/201, and `assumptions/compositions/seam-S6.md` recomposed `VERDICT: SAFE` (`2026-10-05T03:47:59Z`,
  full ten-dimension re-trace, 22/22 members `HOLDS`, basis `sha256:dc54c8c5…`) — C1's disable-form chain closes at the step that consumes the
  unset-versus-set-empty discriminator, with C3's false arm now live and errexit-exempt. The finding completes its lifecycle on that post-implementation
  `SAFE`; disposition stays `absorbed`, no ledger correction and no `-r2` (a resolution is a lifecycle completion — DC-claim-38/34 precedent); both
  source digests re-hashed against the worktree at resolution. Basis `git-commit:0c206694d0740f317d5aa332209be0cd44bfadee`. Invariant cross-reference
  **Neutral** — no session `invariants.md`, zero `Status: active`/`wip` entries registry-wide (eight planned PIs + preserve-provenance `OI:` blocks +
  descriptive rows only); nothing created, activated, displaced or contradicted; **no user decision owed**. Permanent-record follow-up **routed, not
  self-written**: `CO:serve-script-env-to-argv/env-var-flag-crossing` keeps the fired pre-fix `COUNTEREXAMPLE` basis (`sha256:fb247496…` @ `f4076d1c…`)
  as retained history because the recomposition artifact is `PROMOTION: INELIGIBLE` on a digest-only basis — the composition role cannot read the
  registry, and this pass verified 22/22 `CL:` handles present and already ordered producer → transformation → consumer; the handle-bearing republish
  rides T012's wave (`QB:assumption-registry/co-serve-script-basis-republish`), since T012's assembly edit fires the same Invalidates-On. **Until that
  upsert lands the registry row is not citable as current**; S6's live verdict is the on-disk `SAFE`. `seam-S6-value-word-split` (C2) stays **open on
  T012** — the new record names value grammar a *named non-finding* for want of one atomic premise row, which is not C2's closure, so Forward Collect
  owes that row at T012's wave. S6's `Routing` cell moves to `absorbed`, `State` deliberately holds at `recheck` while the registry basis is stale;
  **T012 is dispatchable now** (its ordering gate is met). No milestone gate moved: MS-004's QAP-MS004 author pass remains the next milestone work and
  the reachability-gated QAP-MS002 re-run → MS-002 re-verdict stands per US-016. → [S6](assumptions/compositions/seam-S6.md), [handoff §Post-acceptance Finding Resolution — 2026-10-05](handoff-context.md), [052-01-yield](iterations/052-01-yield.md)
- **2026-10-05** — **T012 accepted-local** (`seam-S6-value-word-split`, ledger DC-seam-S6-value-word-split-r1; [053-01-yield](iterations/053-01-yield.md) ACCEPT).
  **Problem** (recomposition C2): `serve_buster_ripper.sh` interpolated every guarded value into one `EXTRA_FLAGS` string and expanded it unquoted at the exec, so the wrapper silently re-cut
  the caller's own value — `DUMP_DIR="/tmp/my docs"` reached `buster-ripper` as three words (path truncated at its space, `docs` an orphan positional), a whitespace-only value contributed
  **no** field and swallowed the next guard's flag (`--preserved-thinking` eaten, argc 8), and a `*.md` value was pathname-expanded against the cwd. **Resolution**: the six appends build an
  array with the flag and the value as separate, individually-quoted elements, and the exec expands it as `${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}` — the nounset guard is load-bearing, not
  stylistic, because on bash 3.2.57 (`bash` == `env bash` == `/bin/bash` here) a bare empty-array expansion under `set -u` aborts with `unbound variable`, which is precisely the
  all-suppressed run T011 made reachable. Six guard **conditions**, line 28's unset-only default, header line 14, `README.md` (`fe80cb9a…`) and every `buster_ripper/` file are unchanged;
  the consumer contract (`main.py:78` `metavar="PATH"`, `:151-152` `Path(...).expanduser().resolve()`) is what made verbatim one-word forwarding the right outcome — the wrapper adds no
  value protocol of its own. **Verified**: the witness module went 5 → 23 tests (240 lines, Green), each a whole-list equality so "no orphan word / no swallowed neighbour / no positional
  leak" is observable in one diff; the mandated Yield replay was executed independently of Forward's artifacts over 17 cases × **both byte states**, confirming one `--dump-dir`/`--dump-format`
  value word equal to the supplied string for the space path, the space and tab values, `DUMP_FORMAT="a b"`, both-values and the glob, argc 6 rc 0 for the all-suppressed arm, and **byte-identical
  word sets for all nine pinned forms** (T011's three states + six README options) — which also reproduces the claimed 8 fail-first cases and the guard's mutation (bare `"${A[@]}"` → rc 1) without
  touching the working tree. `--dump-format`'s space-bearing value is still rejected *downstream* by `click.Choice`, now with a clear error instead of a stray positional — correct layering, no
  wrapper-side validation added. Suite reproduced **301 passed @ 93.88%** (283 baseline + 18 new, no skips); LNCP/FL-LNCP clean, mypy/ruff N/A (not installed), `bash -n` clean; fingerprints
  `d5db2144…` → `c58256fb…` recompute exactly. `QB:serve-script-argv/zero-word-arm` **absorbed** (its own trigger named this touch-set) and spot-checked genuinely resolved. Zero reject-tier
  findings; **3 P3 deferred** (stale present-tense line pointers in the witness module — the increment's +8 lines invalidated them; the zero-word witness's bash-3.2 pinning is host-dependent;
  two report count figures — `TestDocumentedForms` is 6 cases not 7, backlog remainder 12 not 10, with the substantive pickup claim verified sound). **Gap resolved**: `Done when`'s
  member-recheck and recomposition clauses are this wave's post-acceptance assumption-route riders, not increment scope (T011's wave ran the same order), so the finding is **`accepted-local`,
  not `resolved`** — and **`seam-S6` is not citable as current** until members 159–163, 179, 180, 191–193 and claims 189/162 are re-verified on `c58256fb…` and the recomposition lands
  **with the 22 `CL:` handles in its arguments** (`QB:assumption-registry/co-serve-script-basis-republish`); Forward Collect still owes the one atomic value-grammar premise row. Invariant
  cross-reference **Neutral** (0 `Status: active`/`wip` entries registry-wide); nothing created, activated or displaced, **no user decision owed**. Note for the next checkpoint: `invariants/assumption-claims.md`
  and `invariants/index.md` are dirty in the working tree from the 2026-10-05 promotion pass (mtime 23:42, absent from Checkpoints 119/120) and must not be folded in as this increment's.
  Gates no milestone: **MS-004's QAP-MS004 author pass (forward-qa) is the next milestone work**, and the reachability-gated QAP-MS002 re-run → MS-002 re-verdict stays owed per US-016. → [053-01-yield](iterations/053-01-yield.md), [053-01-forward](iterations/053-01-forward.md)
- **2026-10-05** — **run-29 post-acceptance resolution: `seam-S6-value-word-split` → `resolved`, the `CO:` basis republished, S6 citable at `registry-safe`.**
  The terminal condition both routes named (`Done when`'s recomposition clause; [053-01-yield](iterations/053-01-yield.md) "`resolved` is the recheck's route alone")
  is met: `assumptions/compositions/seam-S6.md` reads `VERDICT: SAFE` / `PROMOTION: ELIGIBLE`, checked `2026-10-05T05:19:41Z`, **25/25** members `HOLDS` each with a
  `CL:` handle, on `serve_buster_ripper.sh` `c58256fb…` (54 lines) + `README.md` `fe80cb9a…` — digests re-hashed by this routing pass against the worktree, all ten
  script-pinned verdict records repinned to `c58256fb…` and the four README-pinned to `fe80cb9a…`. **C2 closed as a traced dimension, not as a declined non-finding**:
  dim 10 (value grammar) rests on member row **202** — the premise run-26 said was missing — one argument word per array element whatever characters an element contains,
  so a space-bearing path, a whitespace-only value, a glob and a leading dash all reach argv intact; dim 3 is closed **at both arities**; and the dim-9 `UNCLEAR` of the
  `d20c1f6b…` record is separated by rows **203**/**204** (the `${array[@]+word}` guard contributes **zero** words on a memberless array and is nounset-exempt for
  `@`-subscripted arrays), which is the arm T011 made reachable and T012 made safe by the guarded expansion. Publication rode the same wave as designed: the composition was
  dispatched **handle-bearing**, so the promoter could upsert `CO:serve-script-env-to-argv/env-var-flag-crossing` **in place** — now `Verdict: SAFE`, `Counterexample: none`,
  `Fingerprint: sha256:cb7c0a5c…`, §Composition Index rebuilt — retiring run-26's "not citable as current" interim rule and discharging
  `QB:assumption-registry/co-serve-script-basis-republish` (**RESOLVED**); `fb247496…`/`dc54c8c5…`/`d20c1f6b…` stay retained superseded history, which is where C2's original
  counterexample text survives. **Ledger:** `DC-seam-S6-value-word-split-r1` **consumed** — resolution is a lifecycle completion, so no `-r2` (per the DC-claim-38-r1 /
  DC-claim-34-r1 / DC-seam-S6-disable-form-r1 precedent); no claim row was routed (all 25 sit `HOLD`/`none`, no `pending` anywhere) and `claims.md` gets no run 29.
  **Gate re-check:** `CLOSURE: absorbed`, Question required **no** — no Fork opened, none owed. **Invariant cross-reference re-run: Neutral** (zero `Status: active`/`wip`
  registry-wide; only the eight planned PIs and preserve-provenance `OI:` blocks; descriptive `CL:`/`CO:` rows are evidence, never authority) — the candidate *"every
  environment value `serve_buster_ripper.sh` forwards reaches the `buster-ripper` argv as exactly one word; the wrapper adds no value protocol of its own"* stays
  **non-authoritative and Interview's to record or promote**. Both S6 findings are now terminal, so **nothing in the serve-script assumption family is dispatchable**; the
  seam's State cell moved `recheck` → `registry-safe`, and its `Routing` stays `absorbed` (that column records disposition, never lifecycle). Handoff budget kept losslessly:
  the two consumed DC entries and the superseded T011-`resolved` section moved **verbatim** to [handoff-archive.md](handoff-archive.md) §Trimmed 2026-10-05 (run-29).
  **Gates no milestone**: `MS-004`'s QAP-MS004 author pass (`forward-qa`) is the next milestone work, the reachability-gated QAP-MS002 re-run → MS-002 re-verdict stands per
  US-016, and the three deferred P3s keep their owner (next touch of `tests/test_serve_script_argv.py`). → [handoff §Post-acceptance Finding Resolution — 2026-10-05 (T012)](handoff-context.md), [seams run-29](assumptions/seams.md), [S6](assumptions/compositions/seam-S6.md), [053-01-yield](iterations/053-01-yield.md)

- **2026-10-05** — **MS-004 boundary base-sync: unchanged** (054-01-forward). Target resolved from the repository's own authority — `git remote show origin` →
  `HEAD branch: main`, no PR base — and refreshed with `git fetch origin main`: `main` = `origin/main` = `72e3a1c6861bf0d8de6317adf52289e5f6cfa4e1`, no movement.
  `merge-base main HEAD` **is** that tip and `--is-ancestor` holds (**154 ahead / 0 behind**) at clean Checkpoint 125 (`ea0b40b…`), so the base was already integrated:
  no merge, no conflict, no both-sides auto-merge to inspect, no source re-threading, no integration commit — and no earlier verdict staled, because HEAD's production
  tree is byte-identical to Checkpoint 119, the tree [053-01-yield](iterations/053-01-yield.md) validated (301 passed @ 93.88%), so the suite was not re-run.
  **No prework packets**: a sync that merges nothing exposes nothing; MS-004 is the final milestone, so its no-deferral clause was checked against all 15 open
  `quality-backlog.md` items and none is merge-exposed — each keeps its recorded owner (`QAP-MS004`, next touch of `tests/test_serve_script_argv.py`, next dependency
  change, a follow-up docs checkpoint). Registry untouched (8 `planned`, 0 `active`/`wip`); no milestone gate, fork, roadmap row or invariant moved; no question asked.
  Session-only checkpoint; nothing pushed, squashed or rewritten. Next: **`QAP-MS004` author pass (`forward-qa`)** — `TC-003`/`TC-113`/`TC-114` on the loopback
  self-signed peer — with the reachability-gated QAP-MS002 re-run → MS-002 re-verdict still owed per US-016. → [054-01-forward](iterations/054-01-forward.md)

- **2026-10-05** — **MS-004 boundary: milestone gate APPROVED, boundary BLOCKED — review evidence persisted review-preserving
  (Checkpoint 128).** [056-yield-phase](iterations/056-yield-phase.md) passes all 7 integration checks on HEAD `25c2b77`
  (base-sync re-verified `unchanged` at `72e3a1c6…`, ancestor true; production tree byte-identical to Checkpoint 119):
  `VP-T006-verify-on-by-default` / `-optout-honored` / `-posture-is-process-wide` **pass** on its own execution (gate 6/6,
  `tests/qa` 21/21 twice, hermetic **301 @ 93.88%**, `evidence/ms004-tls-posture-gate.txt`), `FK-008` claims 1–2 → **HOLDS**,
  zero P1/P2 and 3 P3. Then [057-codex-review](iterations/057-codex-review.md) (external `gpt-5.6-terra`, base `main`,
  418 files) returned **NOT clean**: 4 × P2 real defects + 4 further items, all in the branch's *earlier* aiohttp-migration
  surface — outside `02.00`'s `T001`–`T012` plan and outside `pyproject.toml:30-34`'s coverage, which is why no red pass or
  invariant watched them; **nothing reopens accepted scope**. Closure-packet routing (same day) absorbed **10** packets —
  `DC-F-1-r1`…`DC-F-4-r1`, `DC-G-1-r1`…`DC-G-3-r1`, `DC-056-P3-1-r1`…`DC-056-P3-3-r1` — into **`T013`** (header-parity: F-4,
  G-1, G-2 ride-along; carries the verified `seam-S9` §Sources rider) and **`T014`** (image-slice: F-1, F-2, F-3), both
  `pending` and **gating no milestone**, plus five `quality-backlog.md` rows with named triggers (`QB:cli-surface/undocumented-flags-and-stats-db`,
  `QB:qa-harness/unexercised-path-keyword`, `QB:qa-tls-peer/stale-localhost-claim`, `QB:assumption-registry/t006-tls-referent-elevation`,
  `QB:qa-plan/tc-tls-lifecycle-clause`; unchecked 15 → 20). One packet is `question`: **`FK-011`** preserved-thinking serve
  default (`serve_buster_ripper.sh:30`, `State: open`, Interview's to present — no task may be created for F-5 until answered).
  Two packet corrections rode into the task blocks: F-4 has **five** filter sites, not six; F-2's `img_N` literals also sit at
  `tests/test_image.py:196-198`. **`058-01-forward` (phase-boundary-evidence mode) then persisted this boundary as Checkpoint 128**
  on top of `25c2b77` — census: all 8 dirty paths under the session directory (056, 057, the gate evidence + its INDEX row,
  and the `decisions`/`plan-summary`/`handoff-context`/`quality-backlog` routing edits), zero production, test or `invariants/`
  file dirty. Mode honored: Yield Red CLEAN neither required nor claimed, findings byte-identical, no repair, **phase not
  accepted**, no push/squash/rewrite. Two observations recorded and deliberately not acted on: the new INDEX row fills the
  user's `Disposition` column (recurring since 036-01), and the routing edits put `handoff-context.md` at 753 lines (cap 200)
  and `plan-summary.md` at 443 (cap 400) — the lossless trim is its own operation (Checkpoint 122 precedent).
  **Next**: the owning repair route advances `HEAD` (`T013`/`T014` Forward, `FK-011` presented), then assumptions recheck →
  **Yield Red**, which owns final acceptance and publication, → `phase-boundary-finalize`; `QAP-MS001` is still never authored
  and the QAP-MS002 re-run → MS-002 re-verdict stays owed per `US-016` — reachability held (HTTP 200) throughout 056's run.
  → [056-yield-phase](iterations/056-yield-phase.md), [057-codex-review](iterations/057-codex-review.md), [058-01-forward](iterations/058-01-forward.md)

- **2026-10-05** — **Decide `red-findings` repairs registered for both MS-004-boundary waves — `T013` and `T014`
  blocks repaired, five `-r2` ledger rows appended, both plan restatements reduced to pointers** (registrar batch;
  sources [062-decide](iterations/062-decide.md) and [063-decide](iterations/063-decide.md), both `red-findings` mode
  over [060-forward-red](iterations/060-forward-red.md) and
  [061-forward-red](iterations/061-forward-red.md)). **11 rows disposed, 0 user forks, 0 plan breaks**: 062 = 1 invalid
  + 5 deterministic (`T013` header-parity), 063 = 5 deterministic (`T014` image-slice). What the batch settled, in the
  two blocks that are now their waves' single blueprints: `T013` — the mechanism is **fixed, not open** (in-place
  substitution at the five cited sites through the two existing helpers; no `@web.middleware`, no new referent,
  because `Mode: NEW` rows remain the only authorization and a middleware's edits are inert for a committed SSE
  response while reaching every other response app-wide); the empty-`Bearer` deletion is **route-local by base
  precedent** (`utils.forward_headers` gains no authorization semantics; `/v1/messages` and the catch-all keep
  forwarding an empty `Authorization: Bearer ` verbatim); the acceptance is rename-proof and per-site (five witnesses,
  receiver-level `headers.items()` grep) and its vacuous empty-`Bearer` clause became an authoring instruction with
  three cases including the base-parity half; and `G-1`'s exception surface is bounded by a case table at the
  `messages.py:154` commit boundary — pre-commit keeps `ClientError` → 502, post-commit gets its own `except Exception`
  that logs and abandons **without a replacement response**, `CancelledError` still propagating, both failure classes
  witnessed. `T014` — the placeholder key's referent is **`block["source"]`** under an order-independent canonical
  serialization with **no** `session_id` (a whole-block digest is per-request unstable, verified: `normalize.py:212`
  strips images *before* `_normalize_cache_control` at `:223`), gated by a six-case table; the write-path prune is the
  **global** `cleanup_expired_images()` sweep at the writer's tail with the session-dict delete living **only** there
  under a **binding positional prohibition** (a delete between the `:121-122` ensure and the `:84` index is HTTP 500 on
  the next request from a TTL-idle session — no handler-level `try` exists to absorb it), accepted by W1-W4; the TTL is
  **absolute age since first cached** with the entry carrying both `first_seen` and `last_touched` so `_evict_by_lru`'s
  order is untouched; `has_images`' acceptance is a five-row **position table** (earliest / later-turn / middle /
  none / non-user) so an earliest-only detector cannot pass the gate; and F-3's deferral gained its missing **leak
  trigger** (reclamation for a session with no further image traffic). **Projections:** `handoff-context.md`
  §Planned Task Context (both blocks, inline `*(062 R-…)*` / `*(063 R-…)*` — the two reports' labels collide, so every
  citation is report-qualified), §Deterministic Closure Ledger (fifteen live entries) + the five appended rows
  (`DC-F-4-r2`, `DC-G-1-r2`, `DC-F-1-r2`, `DC-F-2-r2`, `DC-F-3-r2`, superseded-by-nothing; `r1` dispositions refined,
  not reversed), §Current Status, §Session Metadata, §Question/note-disposal/wave schedule (dispatch precondition
  only — wave membership, order, owning routes unchanged); `plan-summary.md` §Session Metadata, §Current Phase,
  §Implementation Plan Summary rows `T013`/`T014`, and the **structural half of both reports' restroking notes in one
  pass** — the two `<details>` restatements are now pointers, which discharges 061's rows 14 + 19 (where `T014`'s
  "write path" bound to the **dump cluster** and "image source" disagreed with the blueprint) and 060's row 17 for
  `T013`, **editing no `T013` blueprint content and rewording neither rendering into agreement**. `decisions.md`:
  **no write** — zero `user-fork` rows, so no FK was created or resolved. **Registry: 0 new / 0 updated / 0 superseded
  / 0 upheld** — neither report owes a registry change; the four invariant candidates in the two blocks stay
  non-authoritative Interview material (063 `R-A` only supplies the referent `:427-428` lacked, so a later elevation
  has one binding), the planned `PI:T001-T006/*` and `E:T003/dump/*` entries are untouched, and no elevation or index
  rebuild ran — `QB:assumption-registry/t006-tls-referent-elevation` and `QB:qa-plan/tc-tls-lifecycle-clause` keep
  their recorded post-Yield-Red finalization trigger. `roadmap.md`: untouched, no source named roadmap impact.
  Freshness: every code and library citation in the landed repairs re-read at pre-write `HEAD 51530da…`;
  `buster_ripper/{utils,image,routes/messages,routes/chat,routes/passthrough}.py` and `tests/test_image.py` are
  byte-identical to the evidence head `25c2b77`, so no repair's premise is stale at the head Forward will run
  against. **Forward-readiness: `T013`/`T014` remain NOT authorized** — each wave's `forward-red` must be
  re-dispatched against its repaired block and return `0 AMBIGUOUS` first (the 029 chain); a clean re-run makes both
  dispatchable, `T013` still carrying its `seam-S9` §Sources rider. No milestone gate moves: `MS-004`'s APPROVED
  verdict, every accepted task's scope, `G-003` and `FK-011`'s ownership are untouched. ⚠ **Observations for their
  owners, deliberately not acted on by this batch:** (1) the batch's writes are uncommitted session files (as are
  059-063) — no registry change occurred, so the registrar's completion rule creates no checkpoint of its own and the
  next evidence-persisting pass owns the commit; (2) the handoff budget grew again (`handoff-context.md` 753 → 1080
  lines against the 200 cap, `plan-summary.md` 443 → 465 against 400) — the lossless trim is its own operation
  (Checkpoint 122 precedent), not a side effect of a repair batch; (3) §Session Metadata line 4 renders the session UID
  as `…b102-827423e1dded` where `session-identity.json` records `…b102-827423a1dded` — an index typo owned by the next
  trim/finalization pass, corrected by no source in this batch, and the handle suffix `9bf7afff5573` does resolve;
  (4) `plan-summary.md` §Current Phase's `➡` marker still points at the superseded "QAP-MS004 can be authored"
  base-sync paragraph although 055 authored and 056 approved that gate — a stale pointer owned by the next phase
  pass, which this batch does not re-own; the live gate is recorded in the ⚠ paragraph added below it.
  → [062-decide](iterations/062-decide.md), [063-decide](iterations/063-decide.md), [handoff §Planned Task Context](handoff-context.md), [handoff §Deterministic Closure Ledger](handoff-context.md)

- **2026-10-05** — **T013 delivered and accepted locally** (066-01-yield rejected member F-4; 067-01-yield accepted the
  retry). Header-parity wave, three members on base `41d3fc6`, one commit each so each restored behavior is
  separately attributable. **Problem** (057's merged reviews, outside the approved plan's surface, on the branch's
  earlier aiohttp migration): the refactor left the shared hop-by-hop helpers in `utils.py` with **zero production
  callers** while all five proxy boundaries carried their own narrowed 4-name/3-name filters, so six RFC-adjacent names
  (`TE`, `Upgrade`, `Keep-Alive`, `Proxy-Authorization`, `Proxy-Authenticate`, `Trailers`) were forwarded upstream and
  relayed downstream — a client's `Proxy-Authorization` credential reaching whatever `--upstream` names; the messages
  streaming relay returned a **fresh** 502 after `prepare()` had committed the head, landing a second status line inside
  the 200 body; and `REFACTORING_PLAN.md` still told a reader a `buster_ripper/convert/` package had shipped.
  **Resolution**: F-4 (`c31d1ac`) routes all five sites through the two existing helpers in place — no middleware, no
  new referent, no sixth call site (`062 R-A`) — extends `response_headers` to drop `content-length` so aiohttp derives
  the length of the body actually sent, and keeps the empty-`Bearer` rule **chat-route-local** with `forward_headers`
  gaining no authorization semantics, because base is asymmetric there and `/v1/messages` and the catch-all must keep
  forwarding an empty bearer (`062 R-B`). G-1 (`d0b1317`) splits the relay at `prepare()`: the pre-commit half keeps its
  502, the post-commit half becomes one `except Exception` that logs position-not-cause, never re-raises, and hands back
  **the committed response object** — a fresh object is CE-1's second head, `None` is a 500 head, and returning the
  committed instance lets `finish_response()`'s `write_eof()` end the stream cleanly; the third injection is a bare
  `asyncio.TimeoutError` so the arm cannot narrow itself to `ClientError`, and `CancelledError` still propagates. G-2
  (`96fe429`) makes the plan document describe the tree: `convert/` rows under an explicit not-implemented marker.
  **The retry** (F-4, `2be9783` + `ce55974`): the first pass made the bearer rule parser-backend-independent by reading
  every value under the name, but selected with a *negative* predicate — `_is_empty_bearer("")` is `False` because
  `bool(value)` short-circuits — so a **value-less** `Authorization` line read as a credential and, arriving first, won
  the selection and replaced the real token with an empty header: worse than the base the wave restores. The selection
  is now positive (`_carries_token` = non-blank and not the value-less `Bearer` form), with a second branch that keeps
  the blank line alone only when nothing carries a token (`None` = no value, `""` = the blank line), which is the
  base-parity case `062 R-D` pins. Verified: 17 shapes replayed against base with **zero credential regressions**, the
  two required witnesses proven red-before-green, backend independence confirmed **at the wire** on both parsers (two
  lines in, exactly one credential out), and all **eleven** mutation rows re-derived on a purged bytecode tree.
  **Also from this wave's review**: two chained matrix runs had reported M5 = 9 because CPython's `.pyc` tag is
  (whole-second mtime, size) and both mutations left `messages.py` at 11325 bytes — the corrected reading is 5, the
  mechanism reproduced in isolation; and two shared helpers made the identity function proved the boundary witnesses
  gate the filter, not aiohttp's own handling, so the 137 test consolidation left no witness passing for the wrong
  reason. Diff: 4 production files + `REFACTORING_PLAN.md`; 69 → 73 witnesses in two new modules on shared
  `proxy`/`raw_proxy` fixtures and one `tests/upstream_stub.py`; hermetic **374 @ 94.20%**, `QAP-MS004` 6/6 re-run at
  final bytes. Deferred: 3 P3 report-accuracy items + a `write()`-position witness hole. **Routed out of scope** (its
  own roadmap entry, no task here): the same second-head class is still live at `chat.py`'s `_relay_chat_stream`, which
  `057` §C never enumerated. Owed downstream: re-issue `DC-S10-C2-r1`'s verdict row and recompose `seam-S10`/`seam-S11`
  on the new bytes, re-verify the `seam-S9` rider's row 67, re-issue verdict rows 246/247 — `accepted-local`, not
  `resolved`. Gates no milestone: `MS-004`'s APPROVED verdict and every accepted scope stand untouched; `T014` remains
  undispatched. → [066-01-yield](iterations/066-01-yield.md), [067-01-forward](iterations/067-01-forward.md), [067-01-yield](iterations/067-01-yield.md)

- **2026-10-06** — **T014 delivered and accepted locally** ([068-01-forward](iterations/068-01-forward.md) →
  [068-01-yield](iterations/068-01-yield.md) ACCEPT, first attempt, zero rejections). The `--strip-images` flag now
  keeps the three promises its own help text makes, all inside `buster_ripper/image.py` (`a5ef525` → `c287a74`,
  witnesses `71fa289`). **F-1**: gate and writer take their scope from one function, so the detector reads every
  user message instead of abbreviating to the last (or first) one — a historical image no longer reaches upstream as
  base64 because the newest turn happened to be text; and a `messages` that is not a list is "nothing to strip",
  which retires the request-path 500 the flag used to raise (`reversed(dict)` with no `try` above it), so flag-on
  and flag-off now answer a malformed body with the same status class. The characterization pin that asserted the
  narrow detector was inverted into the position table rather than exempted (`DC-S12-CONTRACT-r1`). **F-2**: the
  placeholder is `img_<sha256(canonical source JSON)[:12]>` — content-addressed, order-independent, `session_id` not
  an operand, and the slice off the *digest* (realistic base64 shares its leading characters) — so one source keeps
  byte-identical marker text across turns, which is the marker's whole contract; the `cache_control` discriminator
  proves the key is over `block["source"]`, not the block, at unit level only because the `normalize` site is
  unreachable from production. The count-derived key's two failure shapes are now witnessed rather than assumed: two
  distinct sources at the cap no longer collapse onto one token, and eviction keeps running past the bound.
  **F-3**: `IMAGE_CACHE_TTL` and `IMAGE_CACHE_MAX_SIZE` both bind on the write path — the writer's tail sweeps every
  session, the emptied-session drop is **new code inside `cleanup_expired_images`** (not a call site: the primitive
  never owned it), `first_seen` ages retention absolutely while `last_touched` orders eviction, and both stamps come
  from `time.monotonic` through one injected name, because a backward wall-clock step made one entry immune to both
  advertised bounds at once. `get_cached_image` stays unwired (`FK-007` option 1) and the docstring that promised
  restorable placeholders now states the marker contract instead. **Verified by re-derivation, not by tally**: the
  6-red-at-145 state reproduced exactly, `img_04b45e9ae5ea` recomputed from the bound formula alone, W5/W6/W7 each
  traced against the specific build they exist to catch, and the janitor's new delete confirmed as growth in the
  sweep rather than wiring. Hermetic 173 affected / **406 suite**, `image.py` **98%**, 185→255 Green with scanner
  issues 4→3. **Two escalations ruled in Forward's favor**: the three-file test surface (772 lines past the >700
  hard stop; `063 R-D` forbids cutting a witness, and each W-row is the *only* thing that fails on its mutation) and
  the `.encode("utf-8")` correction (the printed formula raises `TypeError`; the pin reproduces from the bound
  operands, so the binding stands). **One gap resolved that Forward did not escalate**: `W7` as worded — inject the
  stamp source, step it backwards, require no negative age — is unsatisfiable by a *correct* build, so the
  wall-clock-pinned realization plus the `_now is time.monotonic` identity is the only reading that passes for the
  right reason. Deferrals held: no call-site consolidation, no read path, no `main.py` edit, no background janitor.
  Ledger row 278 settled in Yield's lane → `HOLD`. 3 defer-tier items (2 already in `quality-backlog.md`, 1
  single-gray nesting metric inherent to the mandated drop) + 3 accepted risks with named re-open triggers.
  **Owed downstream** (`accepted-local`, not `resolved`): `seam-S12`/`S13`/`S14` recomposition on these bytes, rows
  293/294/297/298 + 267/287/316 re-issued with fingerprints — 293's subject sentence is now stale, the file asserts
  expiry on `_image_cache` directly — and 279–285 re-verified against the serializer this wave created. Gates no
  milestone: `STRIP_IMAGES` still defaults `False`. → [068-01-yield](iterations/068-01-yield.md)

- **2026-10-06** — **MS-001 boundary: milestone gate SEMANTIC-APPROVED, boundary BLOCKED — review evidence
  persisted review-preserving (Checkpoint 163).** Three records had left no chronicle line at all (the tail stopped at
  T014's acceptance), so all three are named here before this boundary's own entry:
  [069-yield-phase](iterations/069-yield-phase.md) issued the owed **MS-002 re-verdict — SEMANTIC-APPROVE, zero
  findings** with the declared upstream live (mDNS → `192.168.50.132`, `/v1/models` → 200, `QAP-MS002` 15/15 twice);
  [070-forward-qa](iterations/070-forward-qa.md) authored the never-before-written **`QAP-MS001`** gate
  (`tests/qa/test_qap_ms001.py`, 10 gates + the R1/R2/R5 tiers, with `harness.py` gaining the promoted
  `read_text`/`log_artifacts`/`assert_upstream_response`/`startup_refused`); and
  [071-yield-phase](iterations/071-yield-phase.md) issued **MS-001's gate — SEMANTIC-APPROVE, zero findings**: all
  seven integration checks PASS on the cumulative diff `72e3a1c6…HEAD` at HEAD `c7420ce`, `QAP-MS001` re-run
  independently **10/10** with the four `VP-*` handles recorded `pass`, the `022` directive ratified as
  honored-literal-with-an-expired-premise (no plan change owed), `US-009` c2 arbitrated **HOLDS** first-party
  (`/v1/models` → 200, a proxied `POST /v1/messages` answering the upstream's own typed `404`) — leaving
  `decisions.md` with **no** `PENDING` verification row — on hermetic **407 @ 95.75%** and `tests/qa` **31/31**.
  Then a fresh external `codex-review` against `main` @ `72e3a1c6…` returned **NOT clean**: 3 × P1 + 1 × P2, all four
  classified real, zero false positives, each with a complete closure packet — the **CR2** batch (namespaced because
  the producer's `F-1`…`F-4` collide with `057` §E and the `T013`/`T014` member letters). Closure-packet routing
  (same day) absorbed **4** packets — `DC-CR2-F-1-r1`…`DC-CR2-F-4-r1` — into **`T015`** (chat.py: upstream
  response-header parity on both transform paths + raw-relay fall-through on parse failure + a leak-free 502, since
  `{"error": str(e)}` echoes the internal upstream URL) and **`T016`** (`messages.py`/`passthrough.py`: the
  preserved-thinking non-mapping guard — `"chat_template_kwargs": null` 500s under the shipped launcher default —
  plus static client-facing error bodies), **one wave, anchor `T015`, one local Yield, named prework of this
  boundary's close**; `CR2-F-4` absorbed **record-lane only** (pointers refreshed on `roadmap.md` §Chat route +
  `QB:routes-chat/post-commit-second-head`, the byte-level repro appended as confirmatory evidence, the
  `codify-scope` ownership ruling **explicitly rechecked against it and unchanged** — the session's records already
  stated the second-head shape). **1 packet returned `malformed`** to Forward Collect (`CR2-X-1`, retry-body egress:
  owes an observed request-log probe and the `max_tokens`/`max_completion_tokens` field-authority claim — not a task,
  not a question) and **1 `accepted`** as inert wording (`CR2-X-2`, `README.md:240`'s architecture tree → owner
  `QB:cli-surface/undocumented-flags-and-stats-db` / §RM-005). Two packet premises were corrected on the way in and
  ride the tasks: `CR2-F-2`'s "no witness can produce a non-parseable upstream answer" is false of
  `tests/test_routes_integration.py:239-243`, which does and **pins the defect shape** (replaced, not skipped, by
  T015), and `CR2-F-4`'s understatement claim matches no session record. `FK-011` stays the only open Fork; no seam
  recomposed, no `T013`/`T014` accepted scope reopened.
  **`072-01-forward` (phase-boundary-evidence mode) then persisted this boundary as Checkpoint 163** on top of the
  Review object `dbd3072` — census: all five dirty paths under the session directory (the packet copy + the
  `handoff-context`/`plan-summary`/`quality-backlog`/`roadmap` routing edits), **zero** production, test or
  `invariants/` file dirty, nothing run. Mode honored: Yield Red CLEAN neither required nor claimed, findings
  byte-identical, no repair, **phase not accepted**, no push/squash/rewrite. The same pass put the batch's stale
  **head-label** on the record as a label defect rather than letting it stand: the review's headline names
  `2be9783` (Checkpoint 141) while it **executed against the tree at `dbd3072`** (Checkpoint 162) — its own reported
  **407 passed** baseline is `dbd3072`'s figure (`2be9783` measured **374 @ 94.20%** at
  [067-01-yield](iterations/067-01-yield.md):109, and `git diff --stat 2be9783..dbd3072 -- buster_ripper tests` is
  12 files / +1568 −279), while `dbd3072` differs from the tree 071 measured only in session records. Because the
  routing pass re-derived **every** site at `dbd3072` and all six items hold there, the discrepancy moves the label,
  not the verdict — with two uses for later passes: pointer re-checks resolve against post-T014 bytes, and a repair
  branch that cannot reproduce 407 hermetic tests is not on this tree.
  Two observations recorded and deliberately not acted on: `evidence/INDEX.md`'s `Disposition` column is filled on
  earlier rows (third boundary to flag it — the user's ruling, not ours), and `handoff-context.md` (3017 lines, cap
  200) / `plan-summary.md` (459, cap 400) remain over budget — the lossless trim is its own operation (Checkpoint 122
  precedent).
  **Next**: the owning repair route advances `HEAD` — `T015` + `T016` as one CR2 wave with one local Yield — then the
  assumption recheck on those bytes → **Yield Red**, which owns final semantic acceptance, the `Delivered` entry and
  parent completion → `phase-boundary-finalize`; `CR2-X-1` waits with the collect lane and blocks nothing in the wave.
  → [069-yield-phase](iterations/069-yield-phase.md), [070-forward-qa](iterations/070-forward-qa.md),
  [071-yield-phase](iterations/071-yield-phase.md), [072-01-forward](iterations/072-01-forward.md),
  [`evidence/codex-review-post-071-packet.txt`](evidence/codex-review-post-071-packet.txt)

- **2026-10-06** — **CR2 wave delivered and accepted-local: `T015` + `T016` ACCEPT, zero reject-tier findings, one
  local Yield, both counters 0.** One increment reviewed on its complete range (`1c84ebd` → `a4d1d26` capacity
  extraction → `c8df363`), not per member. What landed, against the four ledgers: **`DC-CR2-F-1-r1`** both
  `_apply_response_transforms` return paths answer from `utils.response_headers`, so an upstream `X-Request-Id` /
  `Retry-After` / `X-RateLimit-Remaining` survives whichever path a body takes; **`DC-CR2-F-2-r1`** the chat non-stream
  arm reads its payload **once** (`_chat_answer`), parses only where the transform or the overflow check needs the
  parsed form, and falls through to the upstream's own status, bytes and filtered headers when the answer is
  well-formed but not JSON — the synthetic `502 {"error": "Invalid JSON from upstream"}` that erased all three is gone
  (`grep` confirms no occurrence left in package or tests), and all four `ClientError` arms on `chat`/`messages`/
  `passthrough` answer `{"error": "Upstream request failed"}` with the exception moved into `log.error` (which
  `passthrough.py` did not have, so the pair of obligations stayed intact instead of trading a leak for an undebuggable
  502); **`DC-CR2-F-3-r1`** `_inject_preserved_thinking` replaces a present-but-non-mapping `chat_template_kwargs`
  rather than 500ing on the shipped launcher's default — the flag's documented guarantee now holds for every forwarded
  body, a client's real mapping still merges (a deliberate `enable_thinking: false` still wins), and the flag-off arm
  is pinned byte-identical; **`DC-CR2-F-4-r1`** stayed record-lane and got no code. Verified by re-derivation rather
  than by tally: the **26-red** fail-first state reproduced in throwaway worktrees at **both** base commits with an
  identical id set (which also supplies the enumeration the wave's own evidence file short-listed), affected **244** and
  hermetic **439 @ 95.75%** re-measured, per-function LNCP re-scanned (`chat.py` 18→14 issues, `_make_chat_request`
  L:18→14, `handle_messages` L:30→28 with hard C:11 untouched, `chat.py` at exactly 300/300 with no baseline worsened),
  and both ownership boundaries proven by AST re-serialization — `_relay_chat_stream` and `T013`/`T014`'s five other
  named chat functions **byte-identical**, so the roadmap-owned second head stayed the roadmap's and nothing accepted
  was restyled. **Six items ruled** (all raised by Forward without a ruling): the three-file test surface in Forward's
  favor (forced arithmetic, `codify-test-cost` clean — the `test_header_parity.py` ingress site answers with a
  non-JSON body, so only the raw path is showable there); `tests/qa/harness.py:135` left untouched but ruled *worse
  than dead* — with bodies now relayed verbatim it can fire on an **upstream's own** "Cannot connect", owner
  `forward-qa`; the two remaining `{e}` overflow bodies out of the wave **with a lane correction** (`CR2-X-1` owes
  egress facts, so the ingress-body claim must ride its re-extraction explicitly or it orphans); the
  `_note_message_start` AST violation a **sanctioned** touch-set pickup of
  `QB:routes-messages/write-only-state-input-tokens` (dead key confirmed on both trees, remedy in the row's own words);
  `_non_stream_single_request`'s surviving hard violation defer-tier **but previously unenumerated anywhere**, like
  `handle_streaming`'s N:3; and the evidence enumeration gap closed by replay instead of read past. 5 new
  `quality-backlog.md` lines, the first of them the pass's only genuinely new fact: the shared
  `utils.response_headers` filter relays **`Content-Encoding` over bytes the proxy's own client already decompressed**
  (`auto_decompress=True`) — base parity, present at all three boundaries now, and named nowhere on this tree before,
  so it is a design-pass call rather than a wave deviation. `FK-011` untouched, zero `decisions.md` bytes moved, no
  invariant record created or edited (triad rule: code and tests are the only sides that moved); no
  `Verification: yield:` row was owed. Both members stay **`accepted-local`, not `resolved`** — the assumption recheck
  on `c8df363` and **Yield Red** own `resolved`, phase acceptance and the `Delivered` shape of this boundary's close;
  this pass accepted the wave, not the phase. → [073-01-yield](iterations/073-01-yield.md)

- **2026-10-06** — **`T017` delivered and accepted-local: ACCEPT, zero reject-tier findings, one local Yield,
  counter 0 (wave of one, first attempt).** Problem: `seam-S15` CE-2 — the operator-declared upstream answers
  `GET /v1/models` with `date` twice and `server` twice on **6/6** probed responses, 1/6 of them with two
  *different* `date` values, and `utils.response_headers` folded aiohttp's per-occurrence head into a plain
  `dict`, keeping the **last** pair and destroying the rest; the running proxy therefore answered the client with
  one line where the upstream sent two, and `Set-Cookie: a=1` beside `b=2` lost `a=1` outright. Reach `reached`,
  first-party, on the deployed path — and **not** base parity, which is what split it from its sibling CE-1: the
  base's httpx client *merged* occurrences into one comma-joined value, so the base never discarded a number,
  while the httpx→aiohttp swap plus the `dict` fold did. `T013` and the CR2 wave restored the base's *drop-set*
  at all three boundaries and left this half of the contract untested — no census witness had ever fed a repeated
  name. Resolution: the closure is one expression's container. `response_headers` now builds a `CIMultiDict` from
  the pairs themselves, so one occurrence is one entry and `web.Response` writes one field-line per entry
  (`http_writer.py:210` declares `CIMultiDict[str]`, so the widened signature moved the helper *onto* its
  consumer's type rather than off it); the drop predicate and its operands are byte-identical, the filter still
  canonicalizes for the **drop** and never for the relay, and all three sites hand the result straight to
  `web.Response` exactly as before — **one production file, zero route diff**, which is also the proof of the four
  prohibitions (`no content-encoding drop`, `no Accept-Encoding/auto_decompress change`, `no forward_headers
  change`, `no _HOP_BY_HOP change`). 16 witnesses in the module that already owns the census, **5 of them
  fail-first and red at `e68ae02`** — unit, `[chat]`, `[messages]`, `[passthrough]`, and a raw-socket row count
  that no comma-joining or de-duplicating build can satisfy; 11 more each named with the build that turns them
  red, including the two `istr`-spelling guards that keep the occurrence rule from silently merging `Connection`
  and `connection` or canonicalizing a kept name (`istr('date') != istr('Date')` is the fact that makes those
  witnesses rather than tautologies, and it is the opposite of the intuition). Verified by re-derivation, not
  tally: the blueprint's replay executed as **36** focused witnesses, hermetic **455 @ 95.75%** re-run on the
  commit with `utils.py` at 98% and `:85` (`split_thinking`, pre-existing) the only miss, `133 → 149` lines Green
  with the touched function unflagged on both trees and the whole file unchanged at 2 issues in 2 functions, and
  **`84 0`** on the test file — zero deleted lines, so `test_utils_holds_the_only_two_filters` and every accepted
  kept/drop and direction-pairing witness survived because none could have been edited. CE-1 is
  **documented, not decided**: the docstring states the posture the filter keeps, says it is base parity, and
  points at `roadmap.md` §RM-008, with **no behavioral witness** written for it so the design pass still has all
  three of its shapes open; §RM-008 has zero diff and `T017` was not folded into it. The eight dirty
  `invariants/`+`verdicts/` paths were re-checked rather than accepted on Forward's disclosure and are **not**
  this loop's: mtimes 15:00–15:08 against Checkpoint 173's 16:12, `invariants/` last committed at Checkpoint 125,
  every appended record naming a checker verdict file in its `Provenance:` line, and `074-assumption-decide`
  having recorded leaving them uncommitted before this wave existed — the triad rule holds absolutely. 2 new
  `quality-backlog.md` lines, neither a defect: the parameter union does not name the `CIMultiDict` its own new
  witnesses pass (`Mapping[str, str]` is exact; annotation-only, no type checker installed on this tree), and a
  mutable collection now crosses three boundaries where one future `headers[X] = …` would silently undo the very
  guarantee this wave bought (`popall`-then-`add`) — safe by a closed caller set, and wrapping is the pass that
  owns the relay rule's decision, not a repair wave's. Five more defer-tier items were already on rows: the
  444-line census module with its split remedy **ruled in Forward's favor** (`codify-test-cost` rule 3, and the
  new class reuses `_ingress_relay`, `_values`, `INGRESS_SITES`, `StubUpstream`, `proxy` and `raw_proxy` rather
  than rebuilding any of them), `QB:deps/multidict-direct-import` (owner §RM-004; the row's 0.01% both-parts test
  holds in both parts — an absent multidict fails at import, before the first request), the **fourth** response
  boundary that still bypasses the filter (`chat.py:174-188`, the streaming error arm, `roadmap.md`'s, barred here
  by `DC-CR2-F-4-r1`) — listed so the semantic pass does not read "every response boundary" as four — the
  httpx-side rationale, which this tree cannot source-read because `httpx` is not installed, so it was accepted
  from the ledger's executed comparison and flagged as the exact limit of what the review verified first-hand, and
  three unexercised input shapes safe by rule (`≥3` occurrences, a duplicated kept `Content-Type` that
  `web_response.py:525`'s `setdefault` can neither add a third of nor collapse). `FK-011` untouched, zero
  `decisions.md` bytes moved, no invariant created or edited, and **no `Verification: yield:` row owed or
  manufactured**. Both of the two things the close still wants were deliberately left undone here: the relay rule
  stays a *candidate* in `T017`'s blueprint for Interview, and `utils.response_headers` is still an **unanchored**
  shared helper that three consecutive waves have now edited through the Anchor Map — a gap the map's "private
  helpers are not rows" clause was never meant to cover, and the boundary's to raise, not this wave's to write.
  `T017` is **`accepted-local`, not `resolved`** — `seam-S15`'s one owed recomposition (its `Invalidates On` 3
  trigger already fired when the return type and comprehension shape moved, and it will still carry CE-1 as a
  routed `accepted` counterexample rather than a clean `SAFE`) and **Yield Red** own `resolved`, phase acceptance,
  and the `Delivered` shape of this boundary's close; this pass accepted the increment, not the phase. →
  [075-01-forward](iterations/075-01-forward.md), [075-01-yield](iterations/075-01-yield.md),
  [`evidence/t017-live-relay-transcript.txt`](evidence/t017-live-relay-transcript.txt),
  [`evidence/t017-occurrence-relay.txt`](evidence/t017-occurrence-relay.txt)

- **2026-10-06** — **MS-001's gate re-issued on current bytes, then blocked again by external review — CR3, sixteen
  findings, zero dispositions.** [077-forward-qa](iterations/077-forward-qa.md) returned the `QAP-MS001` lane to
  `READY-PASSING` by draining the two `tests/qa/harness.py` rows that named it as their trigger (`tests/qa/` bytes
  only, zero production); [078-yield-phase](iterations/078-yield-phase.md) then ran all seven integration checks on
  HEAD `6508941` (base `main @ 72e3a1c6` re-verified: ancestor, 212 ahead / 0 behind) and issued the gate —
  **SEMANTIC-APPROVE, zero P1, zero P2, two P3 both `CLOSURE: absorbed`** (`078-P3-1`: the clause "exactly three
  values" on `PI:T002/dump_format/format_set_is_closed` has no witness, so `dump_turn`'s `else` would silently route
  a fourth format to the self-contained builder → `Forward`, unit tier; `078-P3-2`:
  `E:T002/dump/bulk-flush`'s registry gloss still asserts "No caller until T003's triggers land" → `authority-registrar`,
  folded into an already-open row), `QAP-MS001` **11/11** independently executed, `US-009` c2 **HOLDS**, and MS-001's
  owned surface proven byte-identical to 071's. **The boundary did not pass.** The external `codex-review` run in
  parallel against **the same Review object** (`gpt-5.6-terra`, medium effort, `codex-cli 0.160.0`; base `72e3a1c6`,
  head `6508941` = Checkpoint 183, scope **212 commits / 619 files**, baseline **455 @ 95.75%**) returned **NOT
  clean**: **2 × `P1` + 4 × `P2` + 10 × `P3`** — the aiohttp `client_max_size` default governing every route
  (`server.py:119`, migration-introduced, 2 MB POST → 413 on all four routes); `count_tokens` deriving the session id
  from the **un-normalized** body against `EI:buster_ripper.session/session_id/derived_after_normalize`, with an
  end-to-end proof that `avg_latency` stays 0.0 and the token-fill/TTFT nudge becomes unreachable, plus a
  test-validity claim that the two integration witnesses can never exercise the session-trigger branch; the shared
  `ClientTimeout(total=600)` with no per-request override on the streaming path where base used `timeout=None`, so a
  long stream dies silently mid-body with no `[DONE]`; strict-UTF-8 `decode()` outside the only `except` arm turning a
  non-UTF-8 upstream error into a proxy 500 (and `text()`'s bare `LookupError` on an unknown `charset=` at all three
  parse sites); two `test_compaction` assertions on inline arithmetic whose target branch is the coverage report's one
  missing line; and `--strip-images`'s help promising retrieval of payloads that have no reader on the request path.
  Seven further boundaries were recorded **verified sound with no finding attached**. **This entry makes none of that
  adjudicated.** The report arrived as **dispatcher-held text only** — no session file mentioned `client_max_size` or
  CR3 at read time — so [079-codex-review](iterations/079-codex-review.md) persists it **as relayed, wording and
  severity labels unaltered**, assigns the citation keys `CR3-F-01`…`CR3-F-16` in relay order (the producer's own
  labels were not relayed), adds two clearly-marked locator notes and nothing else, and **leaves every disposition
  empty**: no `CLOSURE:` packet, no `DC-CR3-*` ledger row, no owning route, no repair task. Routing the sixteen is
  **Assumption Decide's** act. Mode honored (`phase-boundary-evidence`): Yield Red CLEAN neither required nor
  claimed, findings unaltered, no code touched, **phase not accepted**, no command executed beyond Git reads.
  Two things this persistence settles for whoever routes next: the **evidence chain begins exactly at the Review
  object** — Checkpoint 184's first parent *is* `6508941` and both it and **Checkpoint 185** carry session-artifact
  bytes only, so no repair advanced `HEAD` under these findings — and, unlike CR2, this review's **head label,
  baseline and reviewed tree all resolve to the same snapshot** (455 @ 95.75% is what 078 measured at `6508941`), so
  there is no label defect to settle and a repair branch that cannot reproduce **455** is not on these bytes.
  One tension flagged rather than resolved: the report clears "the post-commit failure arm and aiohttp 3.13's timeout
  class hierarchy" as verified sound while `CR3-F-03`'s harm runs through that same arm — recorded so the router
  reads the two together, adjudicated by neither. `FK-011` stays the only open Fork.
  **Next**: Assumption Decide over `CR3-F-01`…`CR3-F-16` → repair wave(s) + local Yield → assumptions recheck on
  those bytes → **Yield Red** (owns `resolved` for `T015`/`T016`/`T017`, phase acceptance, the `Delivered` entry and
  parent completion) → `phase-boundary-finalize`; the session-close full-suite gate remains separately owed, and
  `CR2-X-1` still waits with the collector lane blocking nothing. →
  [077-forward-qa](iterations/077-forward-qa.md), [078-yield-phase](iterations/078-yield-phase.md),
  [079-codex-review](iterations/079-codex-review.md),
  [`evidence/ms001-reentry-078-gate-run.txt`](evidence/ms001-reentry-078-gate-run.txt)
- **2026-10-06** — **CR3 routed: 5 absorbed into one wave (`T018`+`T019`+`T020`, anchor `T018`), 11 accepted, zero
  questions — `FK-011` stays the only open Fork.** [080-assumption-decide](iterations/080-assumption-decide.md)
  re-established all sixteen findings against current bytes at pre-write `HEAD` `4008e27b…` (Checkpoint 185;
  `6508941..HEAD` moves no file outside `.sessions/`, so the sites read **are** the reviewed bytes — every claim
  holds on the tree; the packet's two locator notes and its ruff premise verified, and the ruff **count**
  re-derived: 8, all in `tests/`). **Absorbed:** `F-01` (aiohttp's 1 MiB default caps every route — reach
  `reached` under the proxy's own contract: 200 K tokens at the package's 4-chars/token ⇒ ~0.8 MB before
  escaping/multibyte, images are few-tokens-per-megabyte, and nothing in the tree acknowledges the cap) and
  `F-03` (the shared 600 s governs the stream relay where base ran `timeout=None`; the flagged tension with
  verified-sound item 2 resolved as **mechanism cleared ≠ configuration certified** — the per-request
  `total=None` override must not touch the arm or its injected-fault witnesses) → `T018`; `F-02`
  (`count_tokens` keys stats on the un-normalized digest, so the billing nonce misses every request and the
  session-trigger nudge is permanently dead — cross-reference **Aligned** to the session-carried
  `EI:.../derived_after_normalize`, deliberately not a registry-active row per `invariants/index.md`; base
  origin verified on both sides and ranked below the carried rule) and `F-04` (strict decode arms where
  `chat.py:164`'s accepted raw-relay fall-through is the in-lane sibling) → `T019`; `F-05` (both `test_ema_*`
  witness their own arithmetic; `compaction.py:49`'s EMA branch uncovered on live code — Reach Gate n/a, false
  evidence) → `T020`. **Accepted (11):** with Reach verdicts and citations in §Accepted risks — seven land on
  **existing** owners without duplicate rows (`F-07` on 057 `F-1`'s deferral verbatim; `F-06`/`F-12`/`F-13` on
  §RM-005's row, amended to name the false retrieval clause, the re-confirmed dead global and the serve-script
  env axis; `F-08`–`F-10`, `F-14`, `F-16` on verified base parity / API-impossibility / documented-intentional
  grounds), three new records: `roadmap.md` §RM-009 (`F-11`, base-parity split-`message_start` scan, 1%
  category — also §Elevated to user, `Reported: —`), `QB:lint/ruff-f-select-findings`,
  `QB:legacy-monolith/in-file-provenance-header`. **Scope ruling stated plainly:** `F-01` is
  migration-introduced rather than dump/TLS-produced and was **absorbed, not routed out** — following the 057
  and CR2 precedent (branch-wide defects enter this session as waves; the branch merges as one unit; finish
  coding before boundary ceremony) — so the branch does not carry a reproduced 413 into review. The 078
  absorbed P3s (`078-P3-1`/`078-P3-2`) keep their owners. Dispositions and a §Routing Record filled into the
  packet itself ([079](iterations/079-codex-review.md)), which is the findings' only home; no relayed wording
  or severity label altered; zero `invariants/`, production or test bytes moved. **Next:** Forward `T018` →
  `T019` → `T020` → one local Yield for the wave → assumptions recheck on those bytes → **Yield Red** →
  `phase-boundary-finalize`; session-close full-suite gate still separately owed; `CR2-X-1` still with the
  collector lane, blocking nothing. →
  [079-codex-review](iterations/079-codex-review.md), [080-assumption-decide](iterations/080-assumption-decide.md)

- **2026-10-07** — **CR3 wave delivered and accepted-local: `T018` + `T019` + `T020` ACCEPT, zero reject-tier
  findings, one local Yield for the whole wave, all three counters still 0.** One increment reviewed on its
  complete range (`d5df8b06` → `9c90675`), not per member, exactly as `T015`/`T016` were. **Problem → resolution,
  member by member:** `T018` — aiohttp capped every request body at 1 MiB at the handler's *first* read (ahead of
  `normalize()` and `--strip-images`, so the feature that moves image bytes out of the forwarded body never saw a
  request over 1 MiB) and the shared session's `total=600` armed the streaming relay base had run undeadlined;
  resolved by `client_max_size=0` at the package's only `web.Application` — **deleting** the cap, which is why
  `S16-a`'s two readings (the `>=` boundary at `web_request.py:677`, and what the number counts) never had to be
  ruled — and `timeout=aiohttp.ClientTimeout(total=None)` on the streaming `session.post`, with a witness pinning
  **all four components `None`** on the operand actually passed, because a request-local `ClientTimeout` replaces
  the session's rather than merging and a spelling that re-inherited `total` (or armed `sock_read` "to be safe")
  would pass the truncation witness while leaving `S17-CE-1`/`CE-2` live. `T019` — `/v1/messages/count_tokens`
  keyed its stats lookup on the **un-normalized** digest, so the per-request billing nonce missed every documented
  call, `avg_latency` stayed `0.0` and the `per-session` nudge arm — the reason the endpoint exists — was
  permanently dead; resolved by deriving `sid` after `normalize()`, conforming the site to the session-carried
  `EI:.../derived_after_normalize` with `normalize.py`/`session.py` byte-unchanged, plus decode-totality in both
  directions at all five body sites. `T020` — the two `test_ema_*` computed the EMA formula in their own bodies and
  asserted on their own arithmetic, so `compaction.py:49` was an uncovered line on live code; resolved by driving
  production `poll_kv_cache` across ticks with per-tick **value** assertions, one response double per tick, an
  awaited-`text()` precondition per tick, the file's own sibling double as the only permitted construction, and
  `config.KV_POLL_INTERVAL=0` for the driven task's duration restored in the same `finally` (082's corrected item
  (6): `asyncio.sleep(0)` is a yield, not a clock). **Verified by re-derivation, not by tally:** full hermetic
  **471 passed @ 95.91%** re-run on the commit (baseline 455 @ 95.75%, +16, nothing removed — `def test_` counts
  checked per file), `codify scan` re-run on all eight files against a re-extracted `d5df8b06` baseline
  (`_stream_single_request` 30→**29** narrowed, `_non_stream_single_request` **L:33 N:4 C:8 identical at both ends**,
  `handle_messages`'s hard C:11 unmoved, `chat.py` 300/300, `handle_count_tokens` 17→18 the blueprint's own
  pre-authorized gray, no file crossing 500), `uvx ruff@0.16.10 --select F` → **5**, all in `tests/`, proving the
  wave's 3 claimed backlog pickups real, `:49` absent from the `-k ema --cov` missing list with `compaction.py` at
  **100%**, and the two replays the blueprint assigns to Yield run here on a throwaway mutation and reverted
  byte-exact — the EMA weight swap red on the **tick-2 value assertion** (`0.225 != 0.525`), the forced-non-200
  control red on values in **both** witnesses. Pass 1 read the library bytes rather than the doubles: `_error_body`'s
  `text()`-then-`read()` is safe on a real `ClientResponse` (`read()` caches `_body` and `_released` is set only by
  `release()`/`close()`, so the fallback cannot raise `ClientConnectionError` into the `ClientError` arm), every
  locator in the increment's comments is exact on 3.13.3, `web_request.py:722`'s **multipart** limit guards with
  `0 < max_size < size` so `client_max_size=0` disables it too, and row 57's replace-not-merge rule re-confirmed at
  the installed version — which is what `QB:assumption-ledger/row-57-version-pin` needs from its own lane.
  **Two things this pass closed that Forward could not:** `claims.md` **row 416 is `HOLD`**
  (`CL:python/bytes-decode-non-text-codec-error`), so `S18`'s `MC-2` contingency no longer lacks an atom and the
  three approved catch names cover the request direction's real classes — and the increment needed no fourth name,
  exactly as the blueprint demanded. **Three defer-tier findings, no rejection, no question:** `_relay`'s P:6 is a
  hard "Too Many Parameters" recorded in the wave's own table as "P:5 gray" (scan was run; every other figure in
  that table matched this pass's re-measurement — report accuracy, not false evidence); `except LookupError` at the
  three client parses also admits `IndexError`/`KeyError`, `unreachable` at the shipped bytes and deliberately not
  narrowed; and `T019`'s prescribed `normalize()` call necessarily also feeds the no-stats estimate and the
  malformed-shape path — the plan's own "mirroring `messages.py:36-37`" settled both, so they are consequences with
  citations, not a Deferred Ruling. §RM-010 keeps the three still-armed sibling arms (`chat.py:138`'s `total=300`
  included), §RM-001 keeps `handle_messages`, and `T019.001`'s `messages.py:106` is outside this range by design.
  Zero `decisions.md` bytes moved, zero `invariants/` bytes in range (triad rule held; the dirty `invariants/` +
  `verdicts/` paths in the working tree belong to the run-45 lanes), no `Verification: yield:` row owed — `grep`
  finds none in `decisions.md` at all. `FK-011` remains the session's only open Fork. **All three members stay
  `accepted-local`, not `resolved`** — the post-`T018`/`T019`/`T020` recompositions of `S16`–`S19` and **Yield Red**
  own `resolved`, the phase acceptance and the `Delivered` shape of this boundary's close; this pass accepted the
  wave, not the phase. **Next:** `T019.001` (unblocked; 084 places it before `S20`'s recomposition) → assumptions
  recheck on these bytes → **Yield Red** → `phase-boundary-finalize`; the session-close full-suite gate remains
  separately owed, and `CR2-X-1` still waits with the collector lane blocking nothing. →
  [083-01-forward](iterations/083-01-forward.md), [083-01-yield](iterations/083-01-yield.md)

- **2026-10-07** — **`T019.001` delivered and accepted-local: ACCEPT, zero reject-tier findings, one-member wave,
  the counter stays 0.** The run-45 remediation `084` routed from `claims.md` rows 418/422/423, reviewed on its
  complete range (`c96ca3e2` → `0c3a003`, both checkpoints as one increment: 195 landed the repair, 196 corrected
  that pass's own stale pointers). **Problem → resolution:** `handle_streaming`'s context-overflow probe read
  `result.body` back off a response `_stream_single_request` had built as `web.Response(body=<str>)`, and aiohttp
  stores a `str` given to `body=` as a `StringPayload` — so `isinstance(.body, bytes)` was always false and the
  `else` branch handed the matcher `<aiohttp.payload.StringPayload object at 0x…>`. No pattern in `config.py:122-123`
  matches a memory address, `parse_context_overflow_error` returned `None`, and the SGLang **streaming** retry was
  inert on **both** arms the loop can be handed (the error `Response`, and the static 502 whose body can never say
  "Context overflow"), which also left the loop's own terminal arm unreachable. Resolved by the smallest complete
  read — `error_text = result.text`, one expression, **no constructor touched** — because `.text` is the only
  `Response` accessor returning the string's content and it round-trips here: `StringPayload` encodes utf-8 while
  `application/json` names no charset, so `.text`'s fallback is that same codec. The constructor-side alternative
  (`text=`) was foreclosed before it shipped, not after: it appends `; charset=utf-8` to a relayed error, a
  client-visible change on the arm this session's accepted relay-fidelity work governs, and the non-overflow witness
  asserts `Content-Type == "application/json"` so that build goes red rather than green. **What the acceptance
  re-derived rather than inherited:** the blueprint's §Yield replay executed by the Yield pass itself — the
  restore-the-ternary mutation reddened exactly the two fail-first witnesses and was reverted byte-exact — plus
  hermetic **476 @ 95.91%** (471 + 5, none removed), `handle_streaming` re-scanned at **L:25 N:3 C:8 P:4 at both
  ends** from a re-extracted baseline, and the three `DC-claim-*-r1` atoms arbitrated **HOLDS** against the installed
  3.13.3 bytes and one executed probe. **One case neither the blueprint nor the increment named, added by the
  review:** the re-dispatch reaches the wire with a **mutated** body only now that the loop fires, so a stale
  `Content-Length` carried across attempts would have shipped as a fresh defect — closed on source
  (`utils.py:22-26` drops the header; `client_reqrep.py:1056-1058` builds a fresh `CIMultiDict` per request;
  `:1199-1201` re-derives the length from that call's payload). Zero `invariants/` and zero `decisions.md` bytes in
  range (the dirty `invariants/` + `verdicts/` paths in the working tree predate Forward's work by ~1 h and belong to
  the run-45 lanes; proven by mtime and by the range's file list), so the triad rule held; `claim-419` stays
  `malformed` with the collector and still gates `S20`'s recomposition; the blueprint's **Invariant candidate**
  ("reading a body back off an `aiohttp.web.Response` the code constructed goes through `.text`") is now evidenced by
  a live call site and remains a **candidate** — recording and promotion are Interview's act. Two defer-tier
  observations closed inside the pass (a miscounted figure in `handoff-context.md`'s own status line; the
  `Optional[str]` shape of `.text`, unreachable here and unenforceable in a project with no type checker).
  `QB:report-accuracy/001-collect-b005-retry-as-tested` is **closed** — its criterion was this acceptance.
  **`T019.001` is `accepted-local`, not `resolved`**: the boundary's assumptions recheck and **Yield Red** own
  `resolved`, the phase acceptance and the `Delivered` shape of the MS-001 close. **Next:** assumptions recheck on
  these bytes (`S16`–`S19`'s first recompositions, the row-57 re-pin, the row-425 wording `S20` waits on) →
  **Yield Red** → `phase-boundary-finalize`; the session-close full-suite gate remains separately owed. →
  [085-01-forward](iterations/085-01-forward.md), [085-01-yield](iterations/085-01-yield.md)

- **2026-10-07** — **`T019.002` delivered and accepted-local: ACCEPT, zero reject-tier findings, one-member wave,
  the counter stays 0.** The CR4 remediation `089` routed from `CR4-F-01` (P2, [088-codex-review](iterations/088-codex-review.md)),
  nested under `T019` on `T019`'s own surface, reviewed on its complete range (`c5e2be4` → `649c0ad`: Checkpoint 205
  lands the repair, 206 stamps pointers, zero code/test/assertion bytes). **Problem → resolution:** `T019` step 1 made
  this route normalize the body before keying its stats lookup — correctly, because the digest is derived after
  normalization — but by calling the **whole** of `normalize()`, whose `tools` and `messages` steps assume every
  element is a mapping. So `/v1/messages/count_tokens` acquired a 500 base never had: `{"tools":[null]}`,
  `{"messages":[null]}` and `{"messages":["x"]}` measured **200 at base and at `6508941`, 500 at HEAD**, the
  `AttributeError` escaping `handle_count_tokens` with aiohttp authoring the answer. Resolved by narrowing the call to
  the one step the invariant needs — new public delegating entry `normalize.strip_billing_nonce` over
  `_normalize_system`, the only one of the four that writes the text `session_id()` hashes (`session.py:93-98`,
  permanent record `CL:buster_ripper.session#session_id/key-derivation-input`, re-read rather than cited) — `+13/−0`
  there, `+8/−8` with zero line shift at the route, and **`/v1/messages` untouched byte-for-byte**: there the same three
  bodies raise the same error in base itself, so parity *protects* that 500, `G-003` holds the route still, and the
  wider candidate (hardening the loops) is measured **500 → 502** — ruled, not deferred, to `roadmap.md` §RM-012.
  **What the acceptance re-derived rather than inherited:** all seven Done-when clauses on the current bytes, and
  **both mutations the blueprint assigns to Yield executed by the pass itself** — (b) dropping the strip reddens **5**
  witnesses (the 3 route-oracle params, the byte-unchanged
  `test_a_billing_nonce_body_finds_the_stats_messages_recorded`, and `test_count_tokens_inflates_on_compaction_trigger[per-session]`),
  (c) hardening the element loops reddens the three `/v1/messages` posture guards **500 → 502** — each reverted
  byte-exact against Forward's §E sha256 pins; hermetic **511 @ 96.71%** re-run against a 476 @ 95.91% baseline (+35
  cases, none removed), `handle_count_tokens` **L:18 N:1 C:3 identical at both ends** from a re-extracted `c5e2be4`
  baseline, and the digest's insensitivity to the three dropped steps proved structurally (`body["system"]` is assigned
  at exactly two statements, both inside `_normalize_system`; `_normalize_cache_control` deletes only `cache_control`,
  a key `session_id()` never reads) before being pinned by 20 battery + 3 route witnesses. **Two record findings, one
  of them against this session's own prior claim:** §RM-012's "what the fix owes" census said three loops, four read
  sites — measurement says **five** (`_normalize_tools` reads `t.get("name","")` at `:88`, `:89` **and** `:90`; guarding
  `:88`+`:89` keeps `{"tools":[null]}` green at 500), corrected in-pass with both runs and the row's now-stale
  pre-`+13` coordinates; and the closed `QB:routes-count-tokens/normalize-side-effects-on-estimate-and-malformed-bodies`
  row's "counts the body as it arrived" overstates by the nonce the invariant forces into the estimate (~70 chars ≈ 17
  tokens, no-stats arm only) — left standing as a correct closure, with the erratum
  `QB:report-accuracy/090-01-estimate-minus-nonce-not-raw` carrying the wording. The `test_a_well_formed_body_still_reaches_the_relay`
  leak that would have made a posture guard pass on a dead route was checked rather than assumed: `messages.py:317`
  records stats only when `usage.input_tokens` is present, so that witness never writes the module-global `_stats`.
  Zero `invariants/` and zero `decisions.md` bytes in range (triad rule held, by Forward and by this pass), no
  `Verification: yield:T019.002` row existed to arbitrate, `FK-011` remains the session's only open Fork. **An earlier
  attempt at this same iteration id wrote its ACCEPT draft and stopped before any bookkeeping**; its verdict was not the
  session's record until this pass finished against the same bytes, and its draft is preserved verbatim at
  [evidence/090-01-yield-interrupted-draft.md](evidence/090-01-yield-interrupted-draft.md). **`T019.002` is
  `accepted-local`, not `resolved`**: the MS-001 boundary's assumptions recheck on these final bytes (`S16`–`S19`'s one
  recomposition; `claims.md` rows `55`/`134`/`218` re-pin, rows `264`–`266` bind `normalize#normalize` by symbol and do
  not re-base), then **Yield Phase** re-executing `QAP-MS001` (`087`'s `READY-PASSING` is byte-scoped to `8dfa5af`),
  then **Yield Red**, which owns `resolved` for `T015`–`T020`, `T019.001` and this. **Next:** assumptions recheck →
  Yield Phase → Yield Red → `phase-boundary-finalize`; the session-close full-suite gate remains separately owed. →
  [090-01-forward](iterations/090-01-forward.md), [090-01-yield](iterations/090-01-yield.md)

- **2026-10-17** — **T021 delivered and accepted locally** (097-yield). The `MS-001` documented-format gate now
  observes the content `VP-T002-documented-format-preserved` already claimed: `assert_documented_contents` compares
  `turn_000.json` against the complete normalized body (nonce-free `system`) and each `turn_NNN.diff`'s changed lines
  against a `difflib` diff over the `self-contained` artifact's public per-turn renderings — never `_render_body` — while
  `TC-107`/`TC-108` drive nonce-bearing bodies at the unchanged `N + 1`/`2N` layout. The gate was proven falsifiable by
  three production mutation replays on `HEAD 05b993a`, each reverted byte-exact: the route arm reds the layout check, a writer
  that re-attaches the billing header with `sid` stable reds `assert body == expected_body` (`differing=['system']`), and a
  raw-JSON diff source reds the changed-line comparison; the unmutated writer is **12/12** green. `QAP-MS001` 12/12,
  `QAP-MS002` 15/15, `QAP-MS004` 6/6, hermetic **511 @ 96.71%**; zero production, config or registry bytes. Two
  defer-tier rows recorded (`QB:qa-gate/body-drift-witness-inert` — the sensitivity witness's body arm is tautological;
  `QB:test-qap-ms001/file-size` — the gate file crossed 700 lines). **`T021` is `accepted-local`, not `resolved`**: the
  closing Yield Red check on the strengthened bytes owns `resolved` for `T015`–`T021`/`MS-001`, phase acceptance and the
  `Delivered` entry, and the `required:new-surface` assumption extraction is still owed to `forward-collect`. →
  [097-yield](iterations/097-yield.md)

- **2026-10-18** — **Run-49 gate repair `T021.001`/`T021.002` accepted locally** (100-yield). **Problem:** the
  `MS-001` documented-format gate compared its expected side against a different line identity than the writer emits —
  `splitlines()` without endings, `lineterm=""` with no join, `.strip()`ed renderings — so the glued first changed line,
  a CRLF-only change and a trailing-newline-only change each produced a false failure, and the last one was also a
  false-pass surface for a writer that omitted it (`S21` CE-1/CE-2/CE-3); and the gate's client hop used
  `urlopen`, which honors ambient `http_proxy`, so the oracle could judge a proxy that never reached the local endpoint
  (`S22` CE). **Resolution:** `changed_lines` now mirrors the writer's published shape (`splitlines(keepends=True)`,
  `lineterm=""`, `""` join) and both sides are read by one structural reader (`_content_lines`) that consumes the leading
  file-header pair and each glued `@@` control line; `rendering_section` removes only the artifact's record-separator
  newlines; `Proxy.post` builds a `ProxyHandler({})` opener. Three CE regressions and one ambient-proxy regression are
  green on the repair and red on the pre-repair helpers/harness; the two `T021` mutation replays still red; affected scope
  **37 passed** (`QAP-MS001` 16, `QAP-MS002` 15, `QAP-MS004` 6), hermetic **511 @ 96.71%**, zero production,
  config, `invariants/` or `decisions.md` bytes. Two defer-tier rows recorded (the CE tests' tautological
  `expected == observed` arm; the test file at 963 lines and `_content_lines` N:3 C:8). `T021` is `accepted-local`,
  not `resolved`: the `S21`/`S22` recomposition and the closing Yield Red check own `resolved`, phase acceptance and
  the `Delivered` entry. → [100-yield](iterations/100-yield.md)

- **2026-10-18** — **`MS-001` boundary base sync returned `unchanged`** (102-01-forward). Target
  `main @ 72e3a1c6861bf0d8de6317adf52289e5f6cfa4e1` refreshed by `git fetch origin` (no movement; `main` =
  `origin/main` = the prompt's object id) and already an ancestor of HEAD `85ca1ba` — **0 behind / 265 ahead**,
  `git log HEAD..origin/main` empty. No merge, no conflicts, no both-sides auto-merge, no re-threading, no prework
  packet, no integration commit; the checkpoint is session files only and nothing is pushed or rewritten. Production tree is
  byte-identical to Checkpoint 235 (`git diff --stat abaf6cd HEAD -- buster_ripper tests serve_buster_ripper.sh
  pyproject.toml README.md` → empty), so no earlier verdict is staled by this boundary. Evidence is inherited with its
  exact scope, not re-run: `runs/099-hermetic.txt` (**511 @ 96.71%**, 21:06) covers every hermetic test except
  `tests/test_stream_relay_commit.py`, which Checkpoint 235 changed and `101-01-forward` covers at **14 passed**;
  `runs/100-yield-qa.txt` (**37 passed**, 21:13:46) **predates** the concurrent model-alias edits at 21:14:27, so the
  three QA gate modules owe a re-proof on current bytes (owner Yield Phase / `forward-qa`). Worktree at dispatch: 5 modified
  tracked files + 50 untracked run-49 session artifacts, none of them this pass's bytes and none committed by it.
  → [102-01-forward](iterations/102-01-forward.md)

- **2026-10-18** — **`T022.001` closure accepted locally** (104-01-yield). **Problem:** `T022`'s drain marked
  `QB:test-stream-relay/write-position-injection` FIXED and Forward's row 52 marked it `[x]`, but the line was still
  `- [ ]` at `quality-backlog.md:26`, and the state line's "9 open, every one routed … 10 closed — 6 earlier"
  was false. **Resolution:** the line is `[x]` with its DRAIN and `DC-T022-Y-1-r1` evidence; `:14` and the three
  reporting surfaces read **8 routed / 11 closed** (7 earlier, 1 fixed in this wave, 3 accepted risks), re-derived by
  local Yield on `HEAD 73f52cd` with a block-aware count over the multi-line entries; `tests/test_stream_relay_commit.py:51`
  now says "the first write is swallowed", matching `side_effect=[None, fault]`. Focused module **14 passed**; zero production,
  config, `invariants/` or `decisions.md` bytes. Two defer-tier record-accuracy rows recorded: the closed line's DRAIN note
  still says "first write lands" (`QB:quality-backlog/drain-note-write-word`) and Forward's scan summary calls three
  helpers `P:5` where the scan prints two `P:5` and one `P:4` (`QB:quality-backlog/scan-helper-parameter-label`).
  `T022.001` is `accepted-local`, not `resolved`: the closing Yield Red check owns `resolved`, phase acceptance and the
  `Delivered` entry. → [104-01-yield](iterations/104-01-yield.md)

- **2026-10-18** — **`S21` CE-2 residual routed `absorbed` into `T021.003`** (107-assumption-decide). **Problem:** the documented-format gate's expected side reads the `self-contained` artifact's renderings through `read_text` (`tests/qa/harness.py:150-154`), whose universal-newline mode converts `\r\n`/`\r` to `\n` before `production_diff_text` computes the expected diff, while production's diff is computed before the write over raw renderings; a CRLF-vs-LF pair gives `expected=[]` / `observed=['-line1\n', '+line1\n']` and fails `assert observed == expected` at `tests/qa/test_qap_ms001.py:280`. **Reach: latent** — no `\r` byte exists anywhere under `tests/qa/`; the documented-format bodies are `turn_body` markers, and the only CRLF renderings are the in-memory CE-2 pair. The same read-mode asymmetry is a false-pass surface for a writer that normalizes line endings on a mixed change. **Routing:** absorbed, gate repair, test bytes only (`tests/qa/harness.py`, `tests/qa/test_qap_ms001.py`); required proof is one focused artifact round-trip regression (red on the pre-repair `read_text`, green after) plus the existing CE regressions, the two `T021` mutation replays, the three QA gates and the hermetic suite; `S21`'s single post-repair recomposition allowance is consumed by this pass, so a further non-`SAFE` `S21` composition is capped. `S21` stays `recheck`/`absorbed`/`—`; rows 426–431/438–440 stay `HOLD`/`absorbed`; `S22` and `S23` stay `SAFE`; `FK-011` remains the only open Fork. → [107-assumption-decide](iterations/107-assumption-decide.md)

- **2026-10-18** — **`T021.003` gate repair authored: the documented-format oracle reads its two comparison surfaces raw** (108-forward-qa). **Problem:** the gate's expected side read the `self-contained` artifact through `read_text`'s universal-newline mode, so a CRLF-vs-LF rendering pair arrived equal on the expected side while the written `turn_NNN.diff` reported `-line1\r\n`/`+line1\n` — `expected=[]` against `observed=['-line1\n', '+line1\n']` (`S21` CE-2 residual). **Repair:** `read_text` gained an optional `newline` parameter (default unchanged) and `assert_documented_contents`'s observed diff plus `TC-108`'s `self-contained` artifact are read with `newline=""`; one focused artifact round-trip regression is red on the pre-repair universal read with the exact CE-2 numbers and green after, and rejects a writer that drops the line-ending change. **Re-proof:** `QAP-MS001` **17/17**, `QAP-MS002` 15/15, `QAP-MS004` 6/6, hermetic **513 @ 96.71%**, zero production/config/plan/registry bytes; `022` re-checked by grep. → [108-forward-qa](iterations/108-forward-qa.md)


- **2026-10-18** — **`T021.003` accepted locally** (109-yield). **Problem:** the documented-format gate's expected
  side read the `self-contained` artifact through `read_text`'s universal-newline mode, so a CRLF-vs-LF rendering pair
  arrived equal on the expected side while the written `turn_NNN.diff` reported `-line1\r\n`/`+line1\n` — `expected=[]`
  against `observed=['-line1\n', '+line1\n']` (`S21` CE-2 residual). **Resolution:** both diff-comparison surfaces are
  read raw (`read_text(..., newline="")`, shared default unchanged), so the expected diff is computed over the same line
  endings production diffed before the write; one focused artifact round-trip regression is red on the pre-repair read with the
  exact CE-2 numbers and green after, and rejects a writer that drops the line-ending change. Re-proved on `HEAD eea5086`:
  `QAP-MS001` **17/17**, `QAP-MS002` 15/15, `QAP-MS004` 6/6, hermetic **513 @ 96.71%**, zero production/config/plan/
  registry bytes; the two `T021` mutation replays still red (2 failed / 1 failed), each reverted byte-exact. `T021.003` is
  **`accepted-local`**, not `resolved` — the capped post-fix `S21` recomposition and the closing Yield Red check own
  `resolved`, phase acceptance and the `Delivered` entry. → [109-yield](iterations/109-yield.md)

### 2026-10-18 — MS-001 · documented-format dump path preserved — DELIVERED
Delivered: the documented layout (`turn_000.json`, `turn_{N:03d}.diff`, `index.txt`) with today's content and line format at `2N` writes, the collapsed formats deferring every write to a flush, and a raising dump/flush unable to fail a request; `QAP-MS001` 17/17 on `eea5086`, zero production bytes in the close.
→ iterations/095-yield-red.md

- **2026-10-18** — **`FK-011` · preserved-thinking serve default resolved (Option A) and registered — `T023` opens** (registrar batch;
  wave 1, order 1 of 1 — the settled-impacts flush; no other FK open). The user's answer to the serve path's undocumented
  default-on preserved-thinking posture, verbatim: *"A shit why you even created this fork?"* → `A` selects **Option A**: the
  serve path keeps `PRESERVED_THINKING=${PRESERVED_THINKING:-1}` (`serve_buster_ripper.sh:30`) and line 42's `== "1"`
  append, so `buster_ripper/routes/messages.py:46-47` keeps injecting `enable_thinking=true, clear_thinking=false`; `README.md`'s
  serve-script section must document the flag and the variable with its `1` default, at the standard `--skip-tls-verify`
  (`README.md:89-92`) and `--dump-format` (`README.md:169-194`) entries. The remainder of the answer is an objection to the
  Fork's admission, preserved as a routing-quality note for `assumption-decide`, not as a change to the answer. Projections:
  decisions.md §FK-011 Impact (Plan → landed; Registry/Roadmap none); plan-summary §Session Metadata, §Progress Overview,
  §Current Phase, §Open Questions item 11, §Implementation Plan Summary (**T023 · document `--preserved-thinking` and
  `PRESERVED_THINKING` in `README.md`'s serve-script section**, `pending`, `lite:small`, verification profile `parallel`);
  handoff §Planned Task Context, §Current Status, §Open Forks (pending-projection entry replaced). Registry untouched
  (0 new / 0 updated / 0 superseded / 0 upheld, no elevation, no index rebuild; the promoted
  `CO:serve-script-env-to-argv/env-var-flag-crossing` record stays current because Option A moves no script byte).
  Pre-existing registry health diagnostics (5, at HEAD, `invariants/` clean) recorded as observations for their owner, not cured
  here. Roadmap: no row written; observation for its owner — §RM-005 lists `--preserved-thinking` among five undocumented
  flags, so its remaining set becomes four once `T023` lands, and RM-005 stays deferred with its own trigger. Basis re-verified at
  registration on `HEAD fd6fcd6` (working tree clean): script sha256 `c58256fb…` and `README.md` at **zero**
  `preserved.thinking`/`PRESERVED_THINKING` hits — the docs change is **not** yet implemented. `T023` is dispatchable; it
  gates no milestone, and no further user decision is owed. `MS-001` stays resolved. → [FK-011](decisions.md#fk-011),
  [115](iterations/115-interview-reconcile.md)

### 2026-10-18 — T023 · preserved-thinking serve default documented — DELIVERED
Delivered: `README.md` §Serve script options now documents `--preserved-thinking` and `PRESERVED_THINKING`
(default `1`) as the serve script's ON-by-default posture, with `PRESERVED_THINKING=0` as the off-switch and
`FK-011 · preserved-thinking serve default` named as the recorded decision; `README.md` only, script sha256
`c58256fb…` unchanged, 24 serve-script tests green. Accepted at [116-01-yield](iterations/116-01-yield.md).
→ iterations/116-01-yield.md

### 2026-10-18 — T024.001 · T024 record corrections routed — ROUTED (`DC-T024-Y-1-r1`, `DC-T024-Y-3-r1`)
Closure packet `T024-Y-1` rechecked on the bytes (`HEAD eb88254`): 37 open `## Refactoring` lines, **9 literal / 28 prose-only
/ 0 undisposed**, 37/37 naming a complete `## Roadmap items` row; the stray `<details>` is one unclosed element above the
`T024` archive pointer. Both findings `absorbed` into `T024.001` with no user question (marker replacement on the 28 lines;
wrapper deletion); the census-count finding stands corrected at Yield (11 / 37 / 30). No production or test byte moves.
→ iterations/120-assumption-decide.md

### 2026-10-18 — T024.001: T024 record corrections — RESOLVED
Problem: `T024`'s `## Refactoring` tails used the prose form `routed onward to` instead of the literal `→ routed:` marker, and `handoff-context.md` carried an unclosed `<details>`.
Resolution: all 37 `T024` tails now carry the literal marker (0 prose tails); handoff element census 1/1/1 with the `T024` pointer plain and all four following sections present; session text only, and the T024 wave suite at `793a5f0` stands green (551 passed).
→ iterations/119-02-yield.md

### 2026-10-18 — closing-check `121-P2` · Ruff policy labelled a gate it is not — ROUTED (`accepted`, doc line + `RM-021`)
Closure packet rechecked on the bytes (`HEAD 96264eb`): `ruff` is absent from `[dependency-groups] dev`
(`pyproject.toml:41-48`) and `uv.lock`, `.venv/bin/ruff` does not exist, `uv run --frozen ruff --version` →
`Failed to spawn: 'ruff'`, and no automation invokes it — so `pyproject.toml:26`'s comment word "gate" is the
only false byte, while the policy itself runs honestly by the recorded ad-hoc invocation
(`uvx --offline ruff@0.16.10 check …` → `All checks passed!`). Reach: **inert wording**; the settled record
already calls it "hygiene, no gate runs it" (`079-codex-review.md:225`, `080-assumption-decide.md:133`) and assigns
the lint-gate decision to a future lint-policy pass (`119-01-yield.md:49`). Disposition **`accepted`**, no user
question, no task and no closure-ledger entry: doc line `QB:lint/ruff-config-gate-wording` in `quality-backlog.md`
(§Refactoring closed 30 → 31) and roadmap row **`RM-021`** (lint posture: dev toolchain vs verification entrypoint);
`T024`'s accepted record repairs stand. → iterations/121-assumption-decide.md

### 2026-10-18 — SESSION CLOSED: session-close gate green, final semantic revalidation CLEAN (close-sequence step 15b, YR-MS001-r4)
Session outcome: **COMPLETE and CLOSED.** The close gate passes on `HEAD 4250ae6` — Source A drained and landed (Correctness **8 open / 8 routed**; Refactoring **37 open / 37 literal `→ routed:` / 0 prose tails**, 31 closed), Source B clear (`deferred.md` absent, no `DEFERRED` row), Source C clear (no `Required at: session-close`), roadmap **21/21 promoted-onward and landed** (RM-001…RM-021), decisions **11 FK + 17 US, 0 reserved**, and evidence **39/39 ruled** under [`US-017`](decisions.md#us-017) ("Keep only gate evidence" — 10 keep / 29 remove; no artifact deleted, `session-squash` owns the bytes). `T024`/`T024.001` published **`resolved`** by this outcome; the final code bytes are green at `793a5f0` (**551 passed**: 513 hermetic @ 96.71% + `QAP-MS001` 17 / `QAP-MS002` 15 / `QAP-MS004` 6), not re-run because no production or test byte has moved since. Codex over the closing delta `191e676..96264eb` (`runs/121-closing-check-codex`) returned one P2 (`121-P2`), `accepted` at [121-assumption-decide](iterations/121-assumption-decide.md) as inert wording (`QB:lint/ruff-config-gate-wording`, `RM-021`), zero code bytes; per-boundary reads at [057](iterations/057-codex-review.md), [079](iterations/079-codex-review.md) and [088](iterations/088-codex-review.md). Yield Red `YR-MS001-r4` CLEAN at [095-yield-red](iterations/095-yield-red.md) §Adjudication; no post-milestone change altered plan or contract meaning, so step 15b recorded the closing check at [122-closing-check](iterations/122-closing-check.md) instead of a fresh run. Review object `4250ae6` equals `HEAD`; base `main @ 72e3a1c` is an ancestor (0 behind / 309 ahead); tree clean; nothing pushed, squashed or rewritten; no parent completion owed.
→ iterations/123-01-forward.md

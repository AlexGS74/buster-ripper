# 016 — Organize (REPLAN-INCREMENT: upstream TLS cluster — `US-011`)

## Orchestrator Prompt (verbatim)

> Execute the organize-interactive skill for this session. Arguments: $ARGUMENTS
>
> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
> Mode: REPLAN-INCREMENT
> Cluster: upstream TLS cluster (US-011 — skip-verify flag for the upstream HTTP client)
> Reason (from Interview, verbatim): upstream TLS cluster — every agent-ready criterion fails. No done-when criteria, no bound `## Anchor Map` row, no validation point, no milestone mapping; the skip-verify flag is `Mode: NEW` and so owes a complete `RB-*` Reuse Evidence row.

(The remainder of the dispatch was the `organize-interactive`, `codify-agents`, `critical-thinker`,
`codify-user-prompt`, and `codify-deterministic-closure` skill bodies. `$ARGUMENTS` was empty.)

## Session Type / Mode

- **`REPLAN-INCREMENT`**, scoped to the **upstream TLS cluster ONLY**. Dump cluster untouched.
- **This is not a child session.** `session-identity.json` has **no `parent_session_uid`** → the parent
  coordination contract does not apply; there is no parent path to resolve and no `coordination status` to copy.
- **No new `US-*` reservation is owed.** The direct user context in scope is `US-011`, which already exists
  as the canonical record (`decisions.md#us-011`) and carries both utterances verbatim plus the placement
  answer. The orchestrator prompt contains no new user utterance, so there is no bootstrap/unsolicited statement
  to reserve.
- **Not a fresh collect input.** The primary data source is `iterations/009-interview.md` §Round 9
  §Replan Inputs (12 items), with §Round 8 §Source verification and §Round 9 §Source Re-verification as
  the fact base. Everything below is source-verified independently; nothing is carried on the interview's authority.

## Context Re-established

- Session identity re-verified: handle `…--9bf7afff5573`, UID `9bf7afff-5573-402a-b102-827423a1dded`
  (handle suffix = first 12 hex), display id `02.00-2026-09-13-message-dump-bulk-flush`.
- Read before acting: `decisions.md` (`US-011` in full), `plan-summary.md` (all sections), `handoff-context.md`,
  `iterations/009-interview.md` §Round 8–9, `quality-backlog.md`, and the source:
  `buster_ripper/{server,main,config,dump,backend,compaction}.py`, `routes/*`, `pyproject.toml`,
  `tests/test_server_hooks.py`, `tests/test_routes_integration.py`, `.agents/skills/qa-project/SKILL.md`,
  and installed `aiohttp` 3.13.3.
- **No context conflict.** Every surface agrees: PLAN-SELECTED landed; no FK open; the `qa-project` gap is
  closed; the outstanding items are the TLS replan (this pass) and the three `OI:` stamps.
- **Question Mode: deferred** — no `AskUserQuestion` is available on this dispatch path. Two rows are marked
  `OPEN` below and routed to decide/interview; neither is presented here.

## Interview Report Absorption

Which findings came from the interview (round 9) and which from this pass's own source reads:

| Finding | Origin | Treatment here |
|---|---|---|
| `US-011` placement is in-session → replan owed | **interview** (round 9, recording the user's answer) | Accepted as the pass's charter; not re-litigated |
| The 12 `§Replan Inputs` | **interview** (round 9, framed as inputs, not decisions) | Each dispositioned in §Replan Input Disposition below |
| `grep -rniE 'ssl\|tls\|verify\|cert'` → zero in `buster_ripper/` | interview read | **Re-verified, confirmed** (stricter pattern, adds `insecure\|nofail\|cafile\|truststore`) |
| `server.py:47` builds the shared client with aiohttp defaults | interview read | **Confirmed verbatim** |
| `connector.py:978` `ssl: … = True` | interview read | **Confirmed verbatim** |
| `main.py:13`/`config.py:13` `http://localhost:30000` | interview read | **Confirmed**; the *duplicate declaration* is a new finding (§Confusability) |
| six `is_flag=True` precedents | interview read | **Confirmed** (exactly 6: `:19,:21,:23,:38,:42,:51`) |
| touched files Green (104/130/112) | interview read | **Confirmed by line count, CORRECTED by function metric** — see §Capacity Gate |
| `server.py` is coverage-omitted | interview read | **Confirmed, and widened**: `main.py` and `routes/*` are omitted too |
| native capability is `aiohttp.TCPConnector(ssl=…)` | interview read | **Confirmed, and sharpened**: the connector is not an option, it is the *only* seam (§RB-005) |

## Replan Input Disposition (all 12, none dropped)

The sweep that matters for this pass: `US-011`'s placement answer deliberately left four things open
(spelling, default's expression, warning, `--upstream` default). Each is dispositioned here so nothing falls off.

| # | Input | Disposition | Where it now lives |
|---|---|---|---|
| 1 | Cluster scope: outbound TLS posture toward the upstream; deliverable is the **opt-out** | `carried` | §Anchor Map rows 1–2; §Goal Shape G-007 |
| 2a | The opt-out flag is `Mode: NEW`, owes a complete `RB-*` | `carried` — **RB-005 written**, searches re-run independently | plan §Reuse Evidence `RB-005` |
| 2b | `server.py:47` shared-client construction is `Mode: extend` | `carried` — verified as the **only** `ClientSession` in the package | §Anchor Map row 2; §Anchor Verification Evidence |
| 3 | Where the testable decision lives | **`resolved` from code, and widened** | §Where the Testable Decision Lives |
| 4 | ⚠ Harness conflict vs the settled plain-HTTP upstream | **`OPEN` — user authority** | §Goal Shape row `C-TLS-1`; §Notes for Decide |
| 5 | Milestone slot | `carried` — candidate `MS-004` proposed | §QA Planning Context |
| 6 | Goal rows | `carried` — candidate `G-007` drafted | §Goal Shape Analysis; plan §Goal Definition Matrix |
| 7 | Invariant candidate ("verification on unless explicitly disabled") | **`resolved`**: no registry exists and **no existing `EI:` is in the contradiction surface** | §Invariant Reconciliation Prep |
| 8 | Capacity: "One option, one global, one argument. No extraction is owed" | ⚠ **CORRECTED — the gate TRIGGERS on `main.py:main`** | §Capacity Gate (headline) |
| 9 | Security posture; whether skipping logs a warning | **`OPEN` — user authority** | §Goal Shape row 3 |
| 10 | The flag's spelling | **`OPEN` for decide** — candidates enumerated, no code-grounded read-off | §Goal Shape row 2; §Notes for Decide |
| 11 | Not settled, do not infer | `carried` — enumerated with owners, none inferred | §Open Items Carried Forward |
| 12 | Settled topics the replan must not touch | **Verified untouched** — no settled surface edited | §Validation Performed |

## Goal Shape Analysis

`G-007` is a **candidate** row (decide finalizes). Rows 2 and `C-TLS-1` are `OPEN`.

| Goal candidate | Source wording | Observation surface | Met when | Failure witness | Boundary / non-goals | Alternate readings | Authority |
|---|---|---|---|---|---|---|---|
| **G-007** — the upstream's certificate is verified when the upstream is HTTPS | *"if upstream will be with HTTPS, please do verification of certificate on it"* | the outbound TLS handshake the shared `aiohttp.ClientSession` performs against `config.UPSTREAM` | With no opt-out passed and `--upstream https://<host>` whose certificate does not chain to a trusted root (self-signed, or a hostname mismatch), a request through the proxy does not complete against that upstream and the failure is a certificate-verification error; with a trusted certificate the request completes | The proxy **starts, forwards, and returns 200** against a self-signed HTTPS upstream with **no opt-out flag set** — the shared client was built with `ssl=False` unconditionally (or the connector was dropped), so the posture is inverted. Every task completed, the flag exists and works, `--help` is accurate, and the security property the user asked for is absent | No mTLS / client-certificate support; no custom CA-bundle or truststore option (the user asked for the skip flag by name — a CA option is new scope the plan does not hold); no change to `--upstream`'s default; no inbound (client-facing) TLS | (a) verification-on is a *deliverable* the app must implement → **excluded by the user's own words** (*"if it's already implemented on the positive path"*) and by `connector.py:978`; (b) it requires a **positive affordance** the app controls (an explicit CA-bundle path) rather than the inherited library default → **not excluded by code**, but it is *additional* scope, recorded as a boundary | **`code-grounded`** for the half; **`user`** for the CA-bundle boundary |
| **G-007** — an explicit opt-out turns verification off for self-signed certificates | *"have explicit skip verify, like usual, skip TLS verify flag to go unverified on, let's say, self-signed certificates"* + *"We need to add it."* | the CLI surface + the connector the shared session is built with | With the opt-out passed and `--upstream https://<self-signed host>`, a request through the proxy completes against that upstream; without it, the identical request fails at the handshake | The flag is accepted, documented, sets its `config` global — and the connector is still built **without** it (the global is read at import time, or the session is constructed before `main()` assigns the global), so both postures behave identically. Observable: with and without the flag, the same self-signed HTTPS upstream yields the same outcome | Process-wide, not per-request; one posture for every outbound call; the spelling is a **surface** question, not a goal-shape one — three candidate tokens, no code-grounded read-off (§Notes for Decide) | (i) the opt-out is a `config` global only; (ii) it is a CLI flag. **`code-grounded` against (i)**: `on_startup` (`server.py:44-57`) cannot see a `click` parameter — it reads `config.<NAME>` only (`server.py:52`) — so a global is structurally forced. The CLI half is forced by the same QA-injectability rule that produced `F-1` (the gate must start the process with the flag) | **`code-grounded`** (global) / **`user`** (public CLI surface) |
| **G-007** — skipping is *explicit*; security stays strong by default | *"explicit skip verify"* + *"We should keep security strong."* | the default run's outbound posture + what the process emits when the opt-out is used | With no opt-out and an untrusted-certificate HTTPS upstream, the request fails; the opt-out defaults OFF (`config` global `False`, `click` flag unset); the opt-out is never inferred from the URL scheme | The operator passes the opt-out, requests succeed, and **nothing in stdout/stderr records that certificate verification is disabled** — every task complete, the flag correct, and an operator cannot tell from the process output that they are running insecure | Whether "explicit" requires a runtime warning is **not settled by the source words**; not a requirement that the flag be echoed on every request; not a requirement that the scheme be parsed | (i) "explicit" means only *"a flag exists"*; (ii) "explicit" means the skip is *visible at runtime* (a log line), so an operator cannot silently run insecure. **These classify differently** | **`OPEN` — user** |
| **C-TLS-1** — the QA harness contract (`US-010`) stands as approved | `US-010` approved `.agents/skills/qa-project/SKILL.md`; its Upstream row reads *"`http://hyperipper.local:30001` — the live remote vLLM-compatible endpoint. There is no local stub: a gate run fails if that host is unreachable."* | `.agents/skills/qa-project/SKILL.md` + the TLS gate's start command | The TLS verification point runs without the approved contract and the shipped harness disagreeing — either because the gate needs no peer, or because the user approved a stated amendment | A TLS gate ships under `tests/qa/` against a locally-started self-signed HTTPS peer while the skill still declares "There is no local stub" and "a gate run fails if that host is unreachable". Gate green, every task complete, and the user's approved contract silently amended by an agent | Not a constraint on the hermetic `tests/` suite (a different tier, explicitly out of the contract's scope); not a constraint on loopback use as such | (i) the clause scopes the **dump gates'** upstream, so a TLS gate's local peer is an *addition* to the contract → no fork owed; (ii) the clause is absolute for `tests/qa/` → any local peer there is a contract change → **fork owed**. **These classify differently** | **`OPEN` — user** |

**`OPEN` rows to carry into `handoff-context.md`**: row 3 (the warning) and `C-TLS-1` (the harness
contract). Both are user-visible; neither may be resolved by inference.

## Where the Testable Decision Lives (replan input 3 — resolved from code, and widened)

Input 3 asked whether the `ssl`/connector decision should be extracted into a covered module so the VP can be
deterministic without a live peer. Three source facts settle the shape of that question:

1. **Both ends of the plumbing are coverage-omitted.** `pyproject.toml:24-28` omits `buster_ripper/routes/*`,
   `buster_ripper/main.py`, **and** `buster_ripper/server.py`. The interview named only `server.py`. The flag
   lands in `main.py` and the connector in `server.py` — so **no** line of this cluster earns coverage credit
   wherever it is written, and the 80% gate (`U ≤ 3.2 + 0.2·K`) is untouched by it (`K = 0` for this cluster).
2. **The posture is not publicly observable from the session object.** `aiohttp.TCPConnector` stores it in
   `self._ssl` (`connector.py:1003`, assigned from `_merge_ssl_params`) and reads it through
   `_get_ssl_context` (`connector.py:1243`). Both are private; `grep -n "def .*ssl" connector.py` returns only
   `_close`/`_get_ssl_context`. There is no public accessor, so a test cannot read the posture off the connector.
3. **There is exactly one outbound client**, so one opt-out governs the whole surface:
   `server.py:47` is the package's only `ClientSession` (`grep -rn "ClientSession(" buster_ripper/` → one hit;
   `TCPConnector` → zero hits), and `app["session"]` is consumed by `server.py:21,51,72`,
   `routes/chat.py:37`, `routes/passthrough.py:25`, `routes/messages.py:127,228`. **Coupling this makes visible**:
   a single connector decides the posture for `/v1/messages`, `/v1/chat/completions`, the passthrough catch-all,
   the `/metrics` poll and backend detection at once — no call site can opt out individually.

Four routes exist for proving the posture; **the choice is decide's**, and only route D touches `US-010`:

| Route | What it proves | Hermetic? | Cert material | `US-010` impact |
|---|---|---|---|---|
| A — patch `aiohttp.TCPConnector` in `on_startup`, assert the `ssl=` kwarg for both flag states | the flag reaches the connector argument | yes | none | none (a mock-mirrors-implementation risk; must pair with C or D) |
| B — call `on_startup` for real, read `session.connector._ssl` | the real connector carries the intended posture | yes (constructing a connector opens nothing) | none | none — but **private attribute**, and `test_server_hooks.py`'s style is mock-based |
| C — construct an explicit `ssl.SSLContext` in the app instead of `ssl=False`, assert the **public** `ctx.verify_mode == ssl.CERT_NONE` / `ctx.check_hostname is False` | the context handed to aiohttp, on public API | yes | none | none — but it duplicates what `aiohttp`'s own `_make_ssl_context(False)` already does (`connector.py:897-917`) and is arguably a relay |
| D — loopback self-signed HTTPS peer; assert the handshake **fails** by default and **succeeds** with the flag | the actual effect, end to end | loopback only | **required** — `openssl` CLI at test time, or a committed fixture cert/key; `cryptography`/`trustme` are **not** installed (`ls .venv/…/site-packages` → absent) | **⚠ `tests/qa/` here is what puts `C-TLS-1` at risk**; a hermetic `tests/` placement is outside that contract's declared tier |
| E — point at the settled plain-HTTP upstream | nothing: no TLS is negotiated | — | — | — |

Route C is the only one that makes the posture assertable **without** a peer *and* without private attributes;
route D is the only one that proves the **effect** rather than the intent. That tension — public-API testability
versus not re-implementing aiohttp's own context — is the real design question, and it is decide's.

## Anchor Map Verification and Refinement

Two rows added; no existing row corrected (all pre-existing bindings resolve to the referent collect named —
re-verified in 002/013 and not re-opened by this pass, since the TLS cluster touches none of them).

### Anchor Verification Evidence

| Anchor / RB handle | Aspect | Proposition / required role | Supporting scope | Checked basis | Result |
|---|---|---|---|---|---|
| `shared client construction` (new row) | binding-exists | `server.py:47` is the package's **only** `ClientSession`, built with no `connector=` and no `ssl=` | `grep -rn "ClientSession(" buster_ripper/` → one hit; `grep -rn "TCPConnector" buster_ripper/ tests/` → **zero**; `app["session"]` consumers at `server.py:21,51,72`, `routes/{chat:37,passthrough:25,messages:127,228}` | `python -c` read of `server.py:44-57`; greps above | verified |
| `shared client construction` | scope-meaning | the one place the outbound posture is decided; `on_startup` is async so a `TCPConnector` may be built there | `server.py:44-57` (4 executable statements, N:0 C:1 P:1); `ClientSession` requires `connector._loop is loop` (`client.py:377`) | source read; `client.py:370-372` builds a default `TCPConnector(loop=loop, ssl_shutdown_timeout=…)` when none is passed | verified |
| `upstream TLS verification opt-out` (new row) | reuse-absence | no existing symbol owns the outbound-TLS opt-out role | Name search: `grep -rniE '\b(ssl\|tls\|insecure\|noverify\|cafile\|truststore)\b' --include='*.py' buster_ripper/ tests/` → **ZERO matches** (a broader pattern over `*.py/*.md/*.toml/*.sh` excluding `.venv/`/`.sessions/` returns only the English word "Verify" in `tests/` docstrings and the two `*.sessions/` prose trees). Role/peer search: the six `is_flag=True` options (`main.py:19,21,23,38,42,51`) are the *shape* peer — all six are **opt-in** (enable an extra behavior); **none has opt-out polarity**, and `grep -rn '"--no-'` → zero. Value selectors `--eval-profile` (`:45-48`) and `--backend-type` (`:60`) are a different shape. Native capability: `aiohttp` 3.13.3 `TCPConnector(ssl: Union[bool, Fingerprint, SSLContext] = True)` (`connector.py:978`); `ssl=False` → `_SSL_CONTEXT_UNVERIFIED` (`connector.py:925`, built at `:897-917` with `check_hostname=False`, `verify_mode=ssl.CERT_NONE`); **`ClientSession.__init__` accepts no `ssl` kwarg** (`client.py:295-315`), only `connector=` | searches + installed-source reads below | verified |
| `RB-005` | reuse-absence (as above) | as above | as above | as above | verified |
| `flush trigger parameters` / `dump encoding selector` / remaining 13 pre-existing rows | binding-exists | unchanged referents; this pass edits none of them | not re-read in full — carried by citation from `002-organize.md` §Anchor Map Verification and `013-organize.md` §Anchor Map Verification; no dump-cluster file is touched by the TLS cluster | verified (carried by citation) |

### New rows (added to plan §Anchor Map)

| Term | Binding (file:symbol) | Mode | Invariants | Notes |
|---|---|---|---|---|
| upstream TLS verification opt-out | `buster_ripper/config.py` (new global) + `buster_ripper/main.py` (new `is_flag=True` option) | NEW | `PI:T006/upstream_tls/verification_on_unless_explicitly_disabled`, `.../optout_is_process_wide` | **`RB-005`**; the injectability seam for the **outbound posture**. Follows the six `is_flag=True` precedents' shape (`main.py:19,21,23,38,42,51`) but is the repo's **first opt-out-polarity flag** — every existing flag *enables*, none *relaxes*. The global is **structurally forced**: `on_startup` (`server.py:44-57`) reads `config.<NAME>` only and can never see a `click` parameter (`server.py:52`). ⚠ The exact token is **not settled** (`US-011`; replan input 10) — three candidate spellings in §Notes for Decide |
| shared client construction | `buster_ripper/server.py:47` | extend | `PI:T006/upstream_tls/verification_on_unless_explicitly_disabled` | `US-011`'s own wording (*"if it's already implemented on the positive path"*). Today one statement, `aiohttp.ClientSession(timeout=…)`. Becomes a connector + session pair, because **this aiohttp version's `ClientSession` takes no `ssl` kwarg** (`client.py:295-315`; `ssl` is on `TCPConnector.__init__`, `connector.py:978`, and on per-request `session.request(..., ssl=…)`, `client.py:953`). Must be built **inside** `on_startup` (a running loop): `ClientSession` raises `RuntimeError` unless `connector._loop is loop` (`client.py:377`). `ssl=not config.<FLAG>` is behaviourally equivalent to today when the flag is unset: `_merge_ssl_params(True, …)` → `True`, and `ClientSession`'s own default connector passes `ssl=True` (`client.py:370-372`, `connector.py:978`) → both resolve to `_SSL_CONTEXT_VERIFIED` (`:924`) |

**Read path — deliberately NOT an anchor row.** `US-011` names no reader, no artifact and no file; the
cluster is entirely a client-construction concern. No read-side row is added.

## Reuse Evidence

One new row. `RB-001`…`RB-004` are unchanged (the dump cluster is settled).

| Evidence | NEW anchor | Name search | Role/peer search | Native capability | Conclusion |
|---|---|---|---|---|---|
| **RB-005** | upstream TLS verification opt-out (`config.py` new global + `main.py` new flag) | `grep -rniE '\b(ssl\|tls\|insecure\|noverify\|cafile\|truststore)\b' --include='*.py' buster_ripper/ tests/` → **ZERO matches**. Widened to `--include='*.py' --include='*.md' --include='*.toml' --include='*.sh'` over the repo excluding `.venv/` and `.sessions/`: every hit is either the English word "Verify" in `tests/` docstrings (`test_server_hooks.py:18,29,41,…`), prose in `README.md`/`docs/eval.md`, or this session's own planning prose under `.sessions/`. No TLS symbol, module, class, command, option, or context exists. `grep -rn '"--no-'` over `buster_ripper/` → **zero** (no negative-polarity flag precedent) | No registries or Protocol/ABC contracts exist in this repository. **Shape peer**: the six `is_flag=True` options — `--strip-date` (`main.py:19`), `--strip-images` (`:21`), `--verbose` (`:23`), `--preserved-thinking` (`:38`), `--eval-mode` (`:42`), `--eval-thinking` (`:51`). All six *enable* an extra behavior; **none relaxes a safety default**, so the polarity is new even though the `click` shape is established. **Value-selector peers** (`--eval-profile` `:45-48`; `--backend-type` `:60`) are a different shape and not reusable. **Wiring peers**: the only other outbound-client seam is `server.py:47` itself; `backend.detect_backend` (`backend.py:19-87`) and `compaction.poll_kv_cache` (`compaction.py:28-57`) both *consume* `app["session"]` and construct nothing — so there is no second construction site to extend. ⚠ `buster_ripper_legacy.py` contains **zero** `aiohttp` references (`grep -c aiohttp buster_ripper_legacy.py` → 0); a reader must not look there for the client | Verified in installed source, aiohttp **3.13.3**: `TCPConnector.__init__` takes `ssl: Union[bool, Fingerprint, SSLContext] = True` (`connector.py:978`) and merges it via `_merge_ssl_params` (`:1003`); `ssl=False` resolves to the prebuilt `_SSL_CONTEXT_UNVERIFIED` (`:925`, from `_make_ssl_context(False)` at `:897-917`, which sets `check_hostname=False` and `verify_mode=ssl.CERT_NONE`). **Load-bearing negative fact**: `ClientSession.__init__` (`client.py:295-315`) accepts `connector=` and **no `ssl` parameter** — `ssl` appears only on `_request` (`:513`), `request` (`:953`) and `ws_connect` (`:1007`) — so the connector is not one available mechanism among several, it is the **only** seam. Deps stay `aiohttp` + `click` (`pyproject.toml:5-8`); no new package is needed | No existing symbol owns the outbound-TLS opt-out role; a new `config` global + `main.py` flag are genuinely `Mode: NEW`. The mechanism is native and already installed, and what it requires — an explicit `TCPConnector` — is a change to the *existing* `server.py:47` row, not a new artifact |

## Confusability Findings

Two findings, both from **observed** confusion this pass (no speculative sweep).

1. **The plan's capacity baseline is stated by file line count only, so it reads "no capacity pressure"
   for a function that is already hard-violating.** This pass's own correction (replan input 8) is the trigger.
   `main.py:main` is at **P=21** (21 parameters: `upstream`…`backend_type`, `main.py:62-67`) and **L≈31**
   executable statements (`:70-126`) — a hard P violation and a hard L violation — while `plan-summary.md`
   §Goals and Constraints records capacity pressure only for `dump.py` and reports the touched files as
   "104 / 130 / 112 lines — no FL-LNCP pressure". A reader consulting the plan gets a green light for a
   function that is already red. **This is the trap the plan's own capacity gate exists to prevent**, and it will
   catch the next agent identically.

2. **The `--upstream` default is declared twice with nothing linking the two declarations.** `main.py:13`
   (`default="http://localhost:30000"`) and `config.py:13` (`UPSTREAM: str = "http://localhost:30000"`) hold the
   same literal; `main.py:70` overwrites the global unconditionally, so the `config.py` literal is **dead on the CLI
   path** and **live on the `server.py:110` argparse path** (the duplicate `main` already logged in the backlog).
   I had to resolve which is authoritative before I could answer "does the TLS conditional change the default?" — the
   answer is no, but a reader cannot get there without the same detour. Low severity, **not endangering this session**.

Both are upserted to `quality-backlog.md`.

**Not recorded as a finding**: my own first misread was `ClientSession(ssl=False)` — the `ssl` name appears on
`ClientSession._request` (`client.py:513`), `request` (`:953`) and `ws_connect` (`:1007`) but **not** on
`__init__`. That artifact is a *dependency*, not this repository's code; there is nothing here to fix, and the
remedy is the plan-side one already landed: `RB-005`'s native-capability cell and the `shared client construction`
row both state the negative fact, and the task's done-when must carry it.

## Invariant Reconciliation Prep

Scope narrowed to the TLS contradiction surface before reading anything: the outbound client construction, the
`config`/CLI seam, and the request path. **No invariant registry exists in this repository** (`.invariants/`
absent) — all `EI:` entries are session-local drafts.

### Existing invariants in contradiction scope

**None.** All six collected `EI:` entries are dump-cluster and none intersects the TLS surface:

- `EI:buster_ripper.dump/dump_turn/output_layout` — dump artifact layout. The TLS cluster writes no file.
- `EI:buster_ripper.dump/dump_turn/sequential_diff_chain` — `prev_text` advance timing. Untouched.
- `EI:buster_ripper.dump/dump_turn/turn_zero_is_full_body` — turn 0's record. Untouched.
- `EI:buster_ripper.routes.messages/handle_messages/dump_failure_isolated` — a *dump* failure must not fail the
  request. Unrelated: the TLS change can make the **upstream** call fail, which is the pre-existing request-error path,
  not the dump path.
- `EI:buster_ripper.session/session_id/derived_after_normalize` — untouched.
- `EI:buster_ripper.server/on_cleanup/graceful_shutdown_hook` — untouched (`on_startup` is the touched hook).

**The posture is not an existing invariant.** "The shared client verifies by default" is an *inherited `aiohttp`
default*, recorded nowhere in this session and nowhere in the repository. That is why it needs a name — see below.

### New draft (organize-drafted, `US-011`'s own cluster; decide finalizes)

- `EI:buster_ripper.server/on_startup/shared_client_verifies_by_default` — the shared `ClientSession`
  (`server.py:47`) is built with no `ssl`/`connector=` argument, so aiohttp's `ssl=True` default
  (`connector.py:978`) applies and the upstream's certificate is verified | Status: **intended** | disposition
  **refine** — the property survives the change, and the change is what makes it *checkable* for the first time.

### Likely planned invariant themes

- **verification is on unless explicitly disabled** — the shape `US-011` supports. Grounds `G-007` and the refine above.
- **the opt-out is process-wide** — one connector, one posture, for every outbound call
  (`server.py:47` → `server.py:21,51,72`, `routes/{chat,passthrough,messages}`). This is the coupling the
  `decomposition` rule asks to name: splitting the client construction into a connector + session pair does **not**
  create per-call-site postures, and the invariant is what keeps that true.
- **the opt-out changes no dump-cluster behavior** — `G-003` and `VP-T003-passthrough-regression` hold unchanged.

### Suspected contradiction pairs

- `EI:buster_ripper.server/on_startup/shared_client_verifies_by_default` ↔ *verification on unless explicitly
  disabled* — **refines, not contradicts**. The default survives; the change adds the named way off and a witness.
- `G-003` (proxy behaviour on `/v1/messages` unchanged) ↔ the new posture — **no conflict, and checked**: for an HTTP
  upstream nothing is negotiated; for a trusted HTTPS upstream verification already succeeds today. The existing regression
  gate `VP-T003-passthrough-regression` runs `tests/test_routes_integration.py`, whose `client` fixture calls
  `create_app()` and `TestServer(app)` (`:33-38`) — so **`on_startup` runs for real and the shared session is built
  for real** in the hermetic suite, with `backend.detect_backend` mocked. An explicit connector therefore must not break
  that fixture; it does not (a connector built inside the loop is exactly what the fixture already triggers).

### Out-of-scope invariants intentionally dropped

All six collected `EI:` entries — dump-cluster; the TLS cluster intersects none of them. They are not re-read here
beyond the intersection check above.

## Capacity Gate

**TRIGGERED — on `main.py:main`, not on the files the interview checked.** Headline correction to replan input 8.

Baseline at the pre-edit worktree, measured this pass:

| Surface | L (executable stmts) | N | C | P | Verdict |
|---|---|---|---|---|---|
| `config.py` (104 lines) | — (module-level globals only) | — | — | — | Green; +1 global is free |
| `server.py:on_startup` (`:44-57`) | 4 | 0 | 1 | 1 | Green; +2 statements stays Green |
| `server.py` (112 lines) | — | — | — | — | Green → ~114, Green |
| **`main.py:main` (`:62-126`)** | **≈31** | 1 | ≈4 | **21** | **Hard L violation AND hard P violation** |
| `main.py` (130 lines) | — | — | — | — | Green by line count — which is exactly why the plan missed it |

Trigger test (`codify-agents` → Preventive Change-Surface Capacity Gate): *"a touched function already has a
hard LNCP violation … and the task would add statements, branches, nesting, or parameters to it"*. `main.py:main` is
touched (the flag lands there) and the task adds **one parameter and one statement** (plus the `click.option`
decorator). **Triggered.**

The dump cluster already adds **three** options and three assignments to the same function (`T002`'s `--dump-format`;
`T003`'s `--dump-flush-interval`/`--dump-flush-bytes`), so the trigger fired for the settled cluster too and was not
recorded — that is a **plan gap**, reported, not cured (see §Gates Found Unmet).

**Extraction availability, reported honestly.** The smallest coherent extraction inside the touch set is the
config-application block (`main.py:70-86`, 17 statements) or the startup-logging block (`:100-121`). Either reduces **L**;
**neither reduces P**, because `click` mechanically requires one function parameter per declared option — the parameter list is
the option list. So the P=21 factor is arguably **essential complexity** requiring a yield-owned
`# ESSENTIAL COMPLEXITY` marker, and the L factor is reducible. This is a judgment call with a real cost either way
(a `click` option-group refactor changes the CLI contract; a marker is a yield decision), so it is **decide's to route** —
this pass records the measurement and the options, and does not pick.

**`server.py` / `config.py` capacity**: not triggered — both Green, +1 global and +2 statements, no violated function metric grows.

## QA Planning Context

### Regression Test Candidates (from Collect)

No `B-*` handle covers the TLS surface — the cluster is new scope from `US-011`, so there is nothing to regress
against except the cross-cluster gate below.

| Behavior ID | Description | Risk | Tier | Priority |
|---|---|---|---|---|
| `VP-T003-passthrough-regression` (existing) | existing `tests/test_routes_integration.py` suite green | High | hermetic unit/integration (`qa-project` "Hermetic default suite") | P0 — an explicit connector at `server.py:47` is the one change that can break it |

### New Feature Test Areas

| Feature | Expected Behaviors | Tier | Dependencies |
|---|---|---|---|
| upstream TLS verification, default posture | verification on with no opt-out: an untrusted-certificate HTTPS upstream fails; a trusted one succeeds | hermetic integration (in-process, `TestServer(app)`) **and** process-level e2e if route D is chosen | `server.py:47` connector |
| skip-verify opt-out | with the opt-out, the same untrusted-certificate upstream succeeds; process-wide for every outbound call | same as above | the flag global + the connector |
| opt-out default | with nothing passed, the global is `False` and the posture is verified | hermetic unit | `config.py` global |
| flag plumbing | the CLI flag reaches the `config` global | hermetic unit (both `main.py` and `config.py` are coverage-omitted, so this is behaviour-only, not coverage-bearing) | `main.py` option |
| documentation surface | `--help` shows the flag; README documents it | structural | decide's boundary call on README/`serve_buster_ripper.sh` |

### Validation Point Candidates

| Source behavior | Validation point | Observable behavior | Candidate observable after | Candidate validate at | QA author pass | Dependencies | Scope reason |
|---|---|---|---|---|---|---|---|
| `US-011` (positive path) | `VP-T006-verify-on-by-default` | With **no** opt-out passed and `--upstream https://<untrusted-cert host>`, a request through the proxy does **not** complete against that upstream, and the failure is a certificate-verification error | T006 | MS-004 | QAP-MS004 | `server.py:47` connector | The default is currently implicit (an `aiohttp` default) and the change is what can invert it, so it needs a witness for the first time. `VP-T002-default-encoding` is the precedent for making a default an observable contract |
| `US-011` (opt-out) | `VP-T006-optout-honored` | With the opt-out passed and the **same** untrusted-certificate HTTPS upstream, a request through the proxy completes against that upstream | T006 | MS-004 | QAP-MS004 | the flag global + the connector | The effect is only observable when both postures are exercised against one peer; a single-posture test cannot distinguish "flag works" from "flag ignored" |
| `US-011` ("explicit") | `VP-T006-optout-is-explicit` | **Conditional on the row-3 reading.** With the opt-out passed, the process emits a record stating that certificate verification is disabled; with it absent, no such record | T006 | MS-004 | QAP-MS004 | the row-3 `OPEN` resolution | **Do not author before the reading is settled** — under reading (i) there is no such behaviour and this VP must not exist |
| `US-011` (process-wide) | `VP-T006-posture-is-process-wide` | One opt-out governs every outbound call: the `/metrics` poll and the `/v1/models` probe use the same posture as the proxied request | T006 | MS-004 | QAP-MS004 | single `app["session"]` | Structural/instrumented — the single-connector fact is what it protects. Must be paired with a black-box VP so it cannot mirror the implementation |
| `US-011` (no dump regression) | `VP-T006-dump-cluster-unaffected` | The dump cluster's VPs behave identically with and without the opt-out | T006 | MS-004 | QAP-MS004 | — | The cross-cluster join: the flag must change no dump behavior. Can be satisfied by the existing dump VPs re-run under both flag states |

**Earliest-observable note.** No TLS VP can be observed before the connector is built, and the connector is the
whole of the increment — so T006 is the earliest boundary and no earlier one is sufficient. **Every TLS VP above is
`READY-BLOCKED` on the row-3 and `C-TLS-1` resolutions**, not on implementation.

### Behavior Dispositions

| Behavior | Disposition | Validation point or reason |
|---|---|---|
| (new) the upstream's certificate is verified with no opt-out | mapped | `VP-T006-verify-on-by-default` |
| (new) the opt-out turns verification off | mapped | `VP-T006-optout-honored` |
| (new) the opt-out is off by default | mapped | `VP-T006-verify-on-by-default` |
| (new) the opt-out is process-wide | mapped | `VP-T006-posture-is-process-wide` |
| (new) skipping is visible at runtime | **deferred** | row-3 `OPEN`; author only if reading (ii) is taken |
| (new) the flag appears in `--help` and the README | **deferred** | decide's documentation boundary |
| `B-001`…`B-007` (dump cluster) | unchanged | not restated; see `plan-summary.md` §Behavior Disposition Matrix |

### Candidate Milestones

| Milestone | Constituent tasks | Complete when | Validation points | QA author pass |
|---|---|---|---|---|
| `MS-004` (candidate — **new**, not a join) | T006 | With the opt-out absent, an untrusted-certificate HTTPS upstream fails; with it present, the same upstream succeeds; both postures come from one connector and the existing suite is green | VP-T006-verify-on-by-default, VP-T006-optout-honored, VP-T006-posture-is-process-wide | QAP-MS004 |

`MS-001`…`MS-003` are **not** joined: their completion criteria are dump-shaped (collapsed artifact, signals,
coverage) and the TLS cluster composes with none of them. `MS-003` is the coverage gate and the TLS cluster is
coverage-neutral (`K = 0` for covered modules), so it is not a constituent there either.

### Notes for Interview / Decide

- **The two `OPEN` rows must not be settled by an agent.** Row 3 (the warning) and `C-TLS-1` (the harness contract).
  Row 3's fork is a `preference`; `C-TLS-1`'s is a `preference` too, because no data settles whether the
  approved contract tolerates a TLS gate's peer.
- **The flag spelling is decide's call, not a fork unless decide judges it one.** The user delegated by shape
  ("like usual, skip TLS verify flag"). Candidates, none code-grounded: `--skip-tls-verify` (the literal read-off of
  the user's own words in repo kebab-case); `--insecure-skip-tls-verify` (the ecosystem spelling that best matches
  "like usual" — Kubernetes/OpenShift); `--upstream-skip-tls-verify` (repo domain-prefix convention, and the only
  one that disambiguates *outbound* from the inbound server surface). Replan input 10: route a fork instead of
  picking if the name is material as a documented public CLI surface.
- **`serve_buster_ripper.sh` is a boundary decide must rule on.** The user's rationale is *"very relevant for
  local deployments"* and the serve script is the local-deployment entry point (`serve_buster_ripper.sh:21-35` maps
  env vars to flags). It is not in any task's touch set today. Whether a `SKIP_TLS_VERIFY` env var joins it is
  decide's boundary call — the README's "Serve script options" section is the same question for documentation.
- **The `--upstream` default must not change.** A scheme change is a value change (`main.py:13`); no change is owed
  (replan input 11), and finding 2 in §Confusability shows the default is declared twice.

## Initial Task Identification

**Candidate task (decide finalizes the handle, the done-when and the split).**

**T006 — Upstream TLS verification opt-out**
- ADDED: `config.py`'s new opt-out global, defaulting **off** (`False`), following `KV_POLL_INTERVAL`'s
  module-global shape (`config.py:44`).
- ADDED: `main.py`'s new `is_flag=True` option assigning that global, following the six existing
  `is_flag=True` precedents' shape (`main.py:19,21,23,38,42,51`); the exact token per decide (§Notes).
- ADDED: `server.py:47` builds an explicit `aiohttp.TCPConnector` with the posture and passes it as
  `connector=` to the `ClientSession` — **inside `on_startup`**, because `ClientSession` requires
  `connector._loop is loop` (`client.py:377`). With the opt-out unset the posture is `ssl=True`, which
  `_merge_ssl_params` resolves to the same `_SSL_CONTEXT_VERIFIED` aiohttp's default already yields
  (`connector.py:924,978`), so the unset path is behaviour-preserving.
- ADDED: the proof route decide selects from §Where the Testable Decision Lives (A–E). If route D is selected,
  ADDED: the cert material it needs — the repo has none (`cryptography`/`trustme` absent from the venv;
  `openssl` 3.6.3 present at `/opt/homebrew/bin/openssl` on this machine).
- ADDED: unit coverage for both postures and the default.
- ADDED (decide's boundary call): `README.md` and/or `serve_buster_ripper.sh` documentation of the flag.
- REMOVED: **nothing.**
- Depends on: **—** (independent of `T001`–`T005`; `server.py`'s `on_startup` and `on_cleanup` are separate
  functions, and `main.py`'s option block is additive. Either order works; the only cost is adjacent-line merge
  friction in `main.py`).

**Not a task.** The capacity extraction at `main.py:main` rides whichever Forward increment first touches that
function (see §Capacity Gate) — it is not a separate micro-task, checkpoint, or Yield cycle.

## Dropped-Clause Sweep

**Not triggered.** This pass supersedes no prior definition: it is purely additive (new Anchor Map rows, a new
`RB-*`, a new candidate `G-*`/`MS-*`/`VP-*`/task). It replaces no task done-when, no contract clause list,
and no invariant statement. The clauses `US-011` deliberately left open are **enumerated with owners** in
§Replan Input Disposition (inputs 9–11) rather than omitted, so nothing falls off the edge.

## Compaction Decision

- Prior same-phase files reviewed: `iterations/002-organize.md`, `iterations/005-organize.md`,
  `iterations/006-organize.md`, `iterations/013-organize.md`.
- **Superseded**: none.
- **Not superseded**: all four.
- **Compaction performed: no.**
- **If no**: different slice, and the prior files carry live reasoning. `002` is the whole-session initial
  synthesis; `005` is the flush-policy cluster; `006` is the payload/layout cluster; `013` is the encoding
  sub-slice. This pass is the **TLS cluster**, which touches none of them. No prior organize file changes a
  decision for the same slice of work, so none is superseded and there is nothing to absorb.

## Deferred Questions

- **Does "explicit skip verify" require a runtime record that verification is disabled?**
  - Status: **DEFERRED → routed to decide** as the row-3 fork candidate (goal-shape `OPEN`, user authority).
  - Impact: decides whether `VP-T006-optout-is-explicit` exists at all. Under reading (i) it must not be authored.
  - Agent analysis: no ratification recommendation is possible — the source words settle neither reading. The
    adjacent evidence points both ways: the user asked for an *explicit* flag and said *"keep security strong"*
    (reading (ii) leaning), while the repo's existing opt-in flags are silent about their own activation
    (`--verbose` logs nothing about being verbose at startup beyond the one `log.info` at `main.py:107-115`,
    which does name the active modes). No `none`.
- **Does a TLS gate's local peer amend the `US-010` contract the user approved?**
  - Status: **DEFERRED → routed to decide** as the `C-TLS-1` fork candidate.
  - Impact: decides whether the TLS VP may live in `tests/qa/` (fork owed) or must live in the hermetic
    `tests/` tier (no fork). Route D's cert material requirement hangs off the same answer.
  - Agent analysis: no ratification recommendation. Reading (i) is supported by the clause's own wording — it scopes
    *the* upstream, and the dump gates are what the contract governs — while reading (ii) is supported by the clause
    being absolute as written. No `none`.

## Gates Found Unmet (reported, not cured)

**Count reported to the orchestrator: `gates found unmet 3`.**

1. **`main.py:main` is at P=21 / L≈31 — hard LNCP violations — and this cluster adds a parameter and a
   statement to it.** `plan-summary.md` §Goals and Constraints records capacity pressure only for `dump.py`.
   Not curable by this agent (the extraction choice is a judgment call with a real cost either way, and a
   `# ESSENTIAL COMPLEXITY` marker is yield-owned). **Reported** — see §Capacity Gate.
2. **Two goal-shape rows are `OPEN` on user authority** (row 3, the warning; `C-TLS-1`, the harness contract).
   Not curable by this agent. **Reported** — carried into `handoff-context.md`.
3. **`US-011`'s `Impact.Plan` still reads as a future obligation** ("those land with the scoped `organize` →
   `decide` replan"), while organize has now landed its half (Anchor Map rows, `RB-005`, `EI:` draft, goal-row
   candidate). Owner: `interview`. **Not cured** — the record is the interviewer's, and the destinations are only
   half-landed until decide writes the task.

## Questions Asked

**This pass: 0.** Two rows are `OPEN` and routed to decide/interview; neither is presented here. Why 0 is
correct: this dispatch is `REPLAN-INCREMENT` (organize → decide), `organize-interactive` routes goal-shape `OPEN`
rows to decide for a bounded Fork rather than settling them, and Question Mode is **deferred** — no `AskUserQuestion`
is available on this path. No `FK-*` or `US-*` ID was allocated, reserved, superseded, or overwritten by this agent.

## Validation Performed

| Claim | How verified | Result |
|---|---|---|
| Session identity / handle suffix | `session-identity.json` vs handle | matches (`9bf7afff5573` = first 12 hex of UID) |
| No parent coordination contract | `session-identity.json` has no `parent_session_uid` | confirmed — contract N/A |
| No TLS handling exists anywhere in the app | `grep -rniE '\b(ssl\|tls\|insecure\|noverify\|cafile\|truststore)\b' --include='*.py' buster_ripper/ tests/` | **ZERO** |
| One shared client, no second construction site | `grep -rn "ClientSession(" buster_ripper/` | one hit (`server.py:47`); `TCPConnector` → zero |
| Every outbound call uses that one client | `grep -rn 'app\["session"\]' buster_ripper/` | `server.py:21,47,51,72`; `routes/{chat:37,passthrough:25,messages:127,228}` |
| aiohttp's default is verification ON | `.venv/…/aiohttp/connector.py:978` | `ssl: Union[bool, Fingerprint, SSLContext] = True` |
| `ssl=False` actually disables verification | `.venv/…/aiohttp/connector.py:925,897-917` | `_SSL_CONTEXT_UNVERIFIED`; `check_hostname=False`, `verify_mode=ssl.CERT_NONE` |
| `ClientSession` takes **no** `ssl` kwarg | `.venv/…/aiohttp/client.py:295-315` vs `:513,:953,:1007` | confirmed — `ssl` is on `_request`/`request`/`ws_connect` only |
| The connector must be built inside the loop | `.venv/…/aiohttp/client.py:370-372,377` | `TCPConnector(loop=loop, …)`; `RuntimeError` unless `connector._loop is loop` |
| The connector's posture is not publicly readable | `grep -n "def .*ssl" connector.py` | only `_close`/`_get_ssl_context`; `self._ssl` is private (`:1003`) |
| `on_startup` is already exercised for real in the hermetic suite | `tests/test_routes_integration.py:33-38` | `create_app()` + `TestServer(app)` runs startup hooks; `backend.detect_backend` mocked |
| Six `is_flag=True` precedents, all opt-in | `grep -n "is_flag=True" main.py` | exactly 6; `grep -rn '"--no-'` → zero |
| No cert material in the repo | `ls .venv/…/site-packages` | `cryptography` and `trustme` **absent** |
| `main.py:main` is a hard LNCP violation | `main.py:62-67` (21 params), `:70-126` (≈31 stmts) | P=21 hard, L≈31 hard |
| `main.py`/`server.py`/`routes/*` are all coverage-omitted | `pyproject.toml:24-28` | confirmed |
| No settled surface was edited | diff of this pass's edits | only TLS rows added; `T001`…`T005`, the four `PI:`, three `OI:`, `MS-001`…`MS-003`, every `VP-*`, `G-001`…`G-006` unchanged |
| No `FK-`/`US-` ID allocated or reused | this pass wrote only iteration 016, plan TLS rows, `handoff-context.md`, `quality-backlog.md` | confirmed |
| The `OI:` entries are still unstamped | read of plan §Approved Override Register | all three read `pending — interview final approval` — **Topic 3 remains, untouched by this pass** |

## Next Step

1. **`decide`** takes this pass's TLS rows and settles: the task handle and done-when, the proof route
   (A–E), the milestone slot, the `G-007` row, the flag token, the documentation boundary, and whether either
   `OPEN` row is a fork to route to `interview`. It finalizes the candidate rows this pass added as drafts.
2. **`interview`** then re-runs the agent-ready check on the real TLS task(s), presents any user-visible fork
   decide routed, then **Topic 3** (the three `OI:` stamps), then the final-approval gate — now including the TLS
   milestone's `QAP-*`.

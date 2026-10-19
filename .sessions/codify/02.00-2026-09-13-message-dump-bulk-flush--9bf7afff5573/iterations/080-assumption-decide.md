# Iteration 080 — Assumption Decide (closure-packet routing — CR3 batch, MS-001 boundary)

Routes `CR3-F-01`…`CR3-F-16` from [079-codex-review](079-codex-review.md) (persisted `phase-boundary-evidence`,
Checkpoint 185). Closure-packet mode: the packet is the 079 report itself; no `invariants/` byte moved, no
production/test byte written, no `question` emitted (`FK-011` remains the only open Fork).

**Every finding was re-established against current bytes before disposition.** At read time `HEAD` =
`4008e27b5394d9bd994b0131bf47a94dff061d92` and `git diff --name-only 6508941..HEAD` contains **no** file outside
`.sessions/` — the production bytes at this pass are the reviewed bytes, so the sites below were read on the tree
the review ran against, not a descendant of it. The three notes the persisting pass supplied were checked and
hold: `CR3-F-07`'s functions are `buster_ripper/image.py:61`/`:166` ✓; `CR3-F-11`'s `find_message_start` is
`buster_ripper/utils.py:91` ✓; `CR3-F-15`'s premise (no `[tool.ruff]` in `pyproject.toml`) re-verified on the
current file ✓ — and the relayed **count** was re-derived by this pass: `uvx ruff check --select F
buster_ripper/ tests/ buster_ripper_legacy.py` → **Found 8 errors**, all eight in `tests/` (enumerated on the
new `QB:lint/ruff-f-select-findings` row).

## Deterministic Closure Gate — results

Five `absorbed`, eleven `accepted`, zero `question`/`breaks-it`/`malformed`. Ledger rows:
`DC-CR3-F-01-r1`…`DC-CR3-F-05-r1` in `handoff-context.md` §Deterministic Closure Ledger — 2026-10-06 (CR3 batch);
every row's `Repository state` is `git-commit:4008e27b5394d9bd994b0131bf47a94dff061d92`, captured by Git from
the supplied worktree root before the first routing write and never recomputed.

### The three findings the dispatch put pointedly

**`CR3-F-01` — the 1 MiB `client_max_size` default.** *Cap unset on the production path:* re-verified — the
package's **only** `web.Application` construction is `server.py:119` (`grep -rn "web.Application(" buster_ripper/`
→ 1 hit), passing no `client_max_size`; installed aiohttp 3.13.3 defaults it to `1024**2` (`web_app.py:128`) and
enforces it at body-read time with the check skipped only for a falsy value (`web_request.py:675` — which is also
what makes the repair's shape a one-word `client_max_size=0`). Every consumer reaches that construction:
`main.py:188` (the binary the serve script and the QA harness start), `tests/conftest.py` (`proxy`/`raw_proxy` —
a real `TestServer`, so the 413 class is reproducible and witnessable on the hermetic tier). *Acknowledgment:*
tree grep re-run — `client_max_size` zero hits outside `.venv`; the two `1048576` hits are `DUMP_FLUSH_BYTES`
CLI tests (different constant, unrelated); no 413 anywhere in `tests/` or `tests/qa/`. *Does a body that large
arrive under this proxy's own contract:* yes. The plan's `MAX_MODEL_LEN` default is 200 000 (`config.py:59`) and
the package's own estimator converts at **4 chars/token** (`session.py:89-90`) — a full-context Claude Code
session (the client README documents: `ANTHROPIC_BASE_URL=… claude`, `README.md:52-65`) serializes at ≈800 KB
*before* JSON escaping and multi-byte payloads, i.e. the default cap sits **inside the served range, not above
it**; and image traffic is first-class on this route (the whole `--strip-images`/`image.py` cluster exists for
base64 blocks, which cost ~1.6 K model-tokens per megabyte of payload — a session with two screenshots blows
past 1 MiB at trivial token counts). *Witnesses:* none has ever posted over the cap — every hermetic and QA body
is orders of magnitude smaller — which is a blind spot, not safety. Reach verdict: **reached** (exposed route
rejecting an input class its own documented configuration admits; no guard prevents the input).
*Scope decision, stated as the dispatch required:* **absorbed into this session**, following rather than
departing the precedent — 057's four branch-wide migration defects and the CR2 batch were both absorbed as
remediation waves (`T013`–`T017`) on the branch-merges-as-one-unit reasoning, this finding is the same class
(migration-introduced divergence from a base that imposed no cap), and the user's standing instruction is to
finish coding before boundary ceremony. Nothing is routed out, so no named destination and no re-open trigger is
owed; the branch does **not** carry a reproduced 413 into review.

**`CR3-F-02` — `count_tokens` deriving the sid before normalization.** *Claim vs bytes:* holds exactly.
`count_tokens.py:78-81` computes `sid = session_id(body)` on the un-normalized body while
`messages.py:33-37` deliberately normalizes first ("Normalize FIRST to strip billing nonce, THEN compute
session_id"); `session_id()` is a 12-hex sha256 digest **of the `system` text** (`session.py:93-98`) and
`normalize.py:11-15` records that the billing nonce "changes on every request and lives at position 0 of the
system prompt" — so under the documented client the two routes carry **different keys**, `_stats.get(sid)`
misses, `avg_latency` stays 0.0 (`count_tokens.py:89`), and `should_nudge_compact`'s `per_session` arm
(`compaction.py:85-86`, requires latency ≥ 15 s **and** fill ≥ 0.75×) is permanently false: only `kv_pressure`
can fire. The attached test-validity half also holds: `tests/test_routes_integration.py:254-333` seeds
`_stats` under the literal `"test_session"`, which can never equal a digest, and
`test_count_tokens_inflates_on_compaction_trigger` passes only because it forces `_kv_cache_usage = 0.85` — the
session arm is unwitnessed, as relayed. *Base check (this pass, not the relay's word):* the defect **is**
inherited — legacy `proxy_messages` normalizes before deriving (`buster_ripper_legacy.py:640-655`) while legacy
`count_tokens` does not (`raw_body = await request.body()` → `_session_id(body)` with no normalize) — base was
inconsistent the same way. *Invariant cross-reference, stated explicitly as required:* the handle
`EI:buster_ripper.session/session_id/derived_after_normalize` is **carried by this session's registry surface as
a preserve-only protected invariant** — `plan-summary.md` §Invariant Reconciliation lists it under "Existing
invariants in scope: preserve", `handoff-context.md` §Notes carries it as protected for yields, and
`invariants/index.md` §Source Sessions records that the three preserve-only `EI:` drafts were **deliberately not
registered** as active global rows ("the registry never held those handles"). The classification for a site that
violates the rule the session carries, with the finding demanding enforcement of that rule, is **Aligned** — not
`Contradiction`: that class is for findings whose *proposed change* would violate an active invariant (a
false-positive risk to route to the user); here the authority (the carried `EI:`) and the finding's required
outcome point the same way and the code is the outlier, so the task binds to the existing handle and no
duplicate invariant candidate is created. Whether the handle deserves promotion to a registry-active entry is
**Interview's** at the close, not this pass's (triad rule: zero `invariants/` bytes moved here).

**`CR3-F-03` — the shared 600 s total vs the long-lived stream.** *Bytes:* `server.py:66-68` builds the shared
session with `ClientTimeout(total=600)`; `_stream_single_request` (`messages.py:136-141`) passes no per-request
override, so the streaming relay carries a cumulative 600 s deadline base did not — base's stream path ran
`httpx.AsyncClient(timeout=None)` (`buster_ripper_legacy.py:664`, verified at the line the relay names; base's
non-stream arms ran 300 s at `:700`/`:843`). *The flagged tension, resolved rather than routed around:*
verified-sound item 2 certifies that **the post-commit failure arm handles failures correctly when they occur**
(one head, position-not-cause logging, `CancelledError` left propagating) **and that the timeout class hierarchy
is accounted for** (a bare `asyncio.TimeoutError` is not a `ClientError`, and the arm catches `Exception`
precisely because of it — the docstring at `messages.py:180-186` says so). `CR3-F-03` attacks a different
proposition: the **deadline policy upstream of the arm** manufactures a failure class base never delivered, and
the cleared arm then correctly — silently — truncates. Both statements are true of the same code at the same
time; item 2 clears the *mechanism*, the finding indicts the *configuration*. The closure reflects the split:
the repair is a per-request `ClientTimeout(total=None)` on the streaming `session.post` (registry
`CL:aiohttp/per-request-timeout-override` HOLDS — a per-request `timeout=` replaces the session default for that
call), the arm's logic and its injected-fault witnesses (`tests/test_stream_relay_commit.py`'s
`POST_COMMIT_FAULTS`, bare-`TimeoutError` case included) are must-not-touch — a stream can still die for real
reasons, and the arm must keep handling exactly what item 2 cleared it for. Reach: **latent** (needs a
stream > 600 s; plausible under the shipped launcher's default `PRESERVED_THINKING=1` on local backends; the
gate's latent rule demands fix-or-pin, and here the "guard" is the defect itself).

### The remaining absorbed pair

**`CR3-F-04`** — `messages.py:276`'s strict `content.decode()` sits in a `try` whose only arm is
`except aiohttp.ClientError` (`:297`); neither `UnicodeDecodeError` (→`ValueError`) nor `LookupError`
(→`ValueError`) is a `ClientError` — disjoint subtrees, so a non-UTF-8 upstream error body escapes and becomes a
proxy 500 with a traceback, where the sibling at `chat.py:155-167` already catches
`(json.JSONDecodeError, UnicodeDecodeError)` and raw-relays — that sibling is `T015`'s **accepted** shape, which
makes `messages.py` the outlier *within this batch's own settled lane* and the closure deterministic. The
`request.text()` half re-derives from installed source: `web_request.py:686-692` — `text()` decodes with the
Content-Type `charset`, `json()` goes through `text()`, an unknown codec raises `LookupError`; the three client
parse sites (`messages.py:29/30`, `chat.py:30/31`, `count_tokens.py:19/22-25`) catch everything **except**
`LookupError`. Reach: **latent** both halves (network-supplied bodies on exposed routes; the deployed upstream
emits UTF-8; no guard). Absorbed.

**`CR3-F-05`** — `tests/test_compaction.py:348`/`:369` read as relayed: both compute the EMA arithmetic **in the
test body** (`result1 = (1 - alpha) * prev + alpha * 0.75`; `result = raw`) and never call
`compaction.poll_kv_cache`; `compaction.py:49-52` is the EMA-update branch the tests name but do not exercise.
The Reach Gate **does not apply** (its own exclusion list: false evidence holds for every input) — and the
production branch does run (the monitor polls `/metrics` every 10 s in the shipped process), so this is not a
dead-path argument either; it is a suite-attestation defect on the coverage figure (95.75%) the boundary
certifies. Absorbed; tests-only.

## The ten P3s

| id | disposition | Reach verdict | authority / owner |
|---|---|---|---|
| `CR3-F-06` | accepted — inert wording | wording, no runtime reader (`grep` re-run: no witness pins the "retrieval" clause) | **already owned**: `QB:cli-surface/undocumented-flags-and-stats-db` / §RM-005, whose S10–S13 amendment already owns `main.py:74-75`'s help-text scope; CR3 re-verified `image.py:172-173` ("the cached source has no reader on the request path") vs `main.py:75`; marker added to the row |
| `CR3-F-07` | accepted — planned work decides | real input, additive scope | **already owned**: 057 `F-1` deferral verbatim ("owner = image-slice Forward task; trigger = when tool_result-nested images are in scope", [057](057-codex-review.md):61) + §RM-005's nested-half clause; detector/writer re-verified top-level-only (`image.py:31-36,:61-69,:166-200`) — the two **agree**, so it is missing scope, not divergence; no duplicate row |
| `CR3-F-08` | accepted | impossible-in-well-formed-inputs + zero digest effect | `session_id()` reads **`system` only** (`session.py:95-98`) — tool order cannot move the session key at all; duplicate tool names are API-invalid; base sorted the same stable way |
| `CR3-F-09` | accepted — base parity | no documented consumer of repeated params | base did the identical fold: `buster_ripper_legacy.py:908` `params=dict(request.query_params)` |
| `CR3-F-10` | accepted | no defect | module-bottom import is deliberate (`# noqa: E402`, `passthrough.py:56`) and works (call-time `config.UPSTREAM` read); verb widening is a strict relay-superset of base's five verbs (`legacy:898`) — the difference is which upstream answers, not a lost value |
| `CR3-F-11` | accepted — **1% category** | latent, self-healing | base parity verified (per-chunk `_find_message_start(chunk)`, `legacy:674-676`); a split costs one turn's stats, the next turn re-records; new §RM-009 + Accepted-risk + Elevated-to-user lines |
| `CR3-F-12` | accepted — already tracked | dead value | §RM-005 (`config.py:65` re-verified: sole occurrence, `main.py:155-161` uses the click param); re-confirm marker on the row |
| `CR3-F-13` | accepted — already tracked | planned work only | script re-read: no `STRIP_IMAGES`/`BACKEND_TYPE` passthrough, none promised; folded into the §RM-005 row's scope (serve-script axis named), no duplicate row |
| `CR3-F-14` | accepted | test material; a fixture is not a caller | intentional by `FK-009` option 1, documented at `tests/qa/tls_peer.py:11-14`; `openssl x509 -enddate` → **2126** (~100 y, loopback-only, guards nothing worth stealing) |
| `CR3-F-15` | accepted | hygiene, no gate runs it | no `[tool.ruff]` (premise re-verified current bytes); the 8 findings **re-derived** by this pass with `uvx ruff`; new row `QB:lint/ruff-f-select-findings`; two of the eight sit inside T019/T020 touch-sets and will likely be consumed there |
| `CR3-F-16` | accepted — condition met, gap recorded | never executed | the file self-declares its deps (PEP 723 header `legacy:1-9` — fastapi/uvicorn/httpx), is imported by **nothing** (`grep` → docstring citations only); the report's "if declared reference-only" condition is met in session artifacts but not in-file → new row `QB:legacy-monolith/in-file-provenance-header` |

## Wave

**One wave, anchor `T018`, members `T018`/`T019`/`T020`, one local Yield** — fewest safe waves per
`run-codify/references/remediation-waves.md`: all three share the owning route (implementation Forward), each is
a few-lines-complete remediation with witnesses, `T018`/`T019` share `messages.py` at **disjoint functions**
(editing the same file alone is not interference), `T020` touches a file no member else touches. Intra-wave
execution note: `T018` lands before `T019` (both edit `_stream_single_request` at different lines — a rebase
sequence, recorded as ordering, not separation). Retry owners: each task's own counter, starting 0.
Prework of the **MS-001 boundary close** — lands before the assumptions recheck on those bytes and before
Yield Red, same as the CR2 batch. `Assumption: required:new-surface` on all three members (no live seam covers
these sites — `S9` is table-marked `out-of-scope`/`unverified`; its rows' fingerprint staleness after the edits
is the boundary's owed recheck's business, not a seam these tasks hold open).

Durable transmission: full blueprints in `handoff-context.md` §Planned Task Context → `T018`/`T019`/`T020`;
plan rows in `plan-summary.md` §Implementation Plan Summary; dispositions filled into the 079 packet itself
(it is the findings' only home — 079 §Step 3).

## Pre-Flight Checks

- **Session identity: verified.** `session-identity.json` → `session_uid: 9bf7afff-…` matches the dispatch
  handle's `--9bf7afff5573`; files read directly. Not `BLOCKED-SESSION-IDENTITY`.
- **Mode contract:** closure-packet mode — packet + named source read (`079-codex-review.md`, plus 057/074/076
  precedent records and the §Notes-protected invariant chain); `plan-summary.md`, `decisions.md`,
  `handoff-context.md`, `quality-backlog.md`, `roadmap.md`, `invariants/index.md` read for scope.
- **Claims/seams:** `grep "Routing.*pending" assumptions/claims.md` → **0** — no atomic row waits on this
  pass; the CR3 packet names no seam.
- **Git metadata boundary:** honored — only `git rev-parse`/`git diff`/`git log`/`git show` on repo state plus the
  ledger's `HEAD` capture; nothing under `.git/` read as data.
- **Commands run:** read-only Git; `openssl x509 -enddate` on the fixture; `uvx ruff check --select F` (a
  linting tool against the tree — no project code executed, no app/server run, no suite executed). **No
  production/test byte written by this pass.**
- **Plan approval gate:** planning CLOSED round 18 (`US-015`) — routing is covered by the same ruling 058-01,
  072-01 and 079 applied to their passes.
- **Deterministic Closure Gate:** loaded and applied per finding (above); `question` admitted nowhere because no
  materially different admissible outcome survives the base-parity + `G-003`/plan authorities unranked — every
  absorbed closure had its competing candidates rejected on settled authority, and every accepted row is a
  Reach-First or 1%-category disposition with a named citation.
- **Forks:** no `reserve-decision-record` invoked; `FK-011` remains the only open Fork, untouched — nothing here
  outranks it.
- **Triad:** `git status --porcelain -- invariants` → empty, before and after; no invariant created, edited,
  superseded or deleted; the F-02 classification binds the task to the existing handle only.

## Durable Signal

- **CR3 is routed**: 5 `absorbed` → **`T018`/`T019`/`T020`** (one wave, anchor `T018`, one local Yield, prework
  of the MS-001 close) — 11 `accepted` with citations (7 land on **existing** owners: 057's F-1 deferral, §RM-005
  ×4, and four with fresh pointers; new rows: `§RM-009`, `QB:lint/ruff-f-select-findings`,
  `QB:legacy-monolith/in-file-provenance-header`). Zero questions, zero breaks, zero malformed.
- The 078 absorbed P3s (`078-P3-1`, `078-P3-2`) keep their existing owners (Forward unit tier /
  authority-registrar) — re-routing settled 078 work in this pass would be churn, and CR3 neither blocks nor
  amends either row.
- **Boundary owes next, in order:** Forward `T018` → `T019` → `T020` → one local Yield for the wave →
  assumptions recheck on those bytes (S9's stale fingerprints are that pass's, not these tasks') →
  **Yield Red** (owns `resolved` for `T015`/`T016`/`T017`, the new wave's acceptance trail, phase acceptance, the
  `Delivered` entry, parent completion) → `phase-boundary-finalize`; the session-close full-suite gate remains
  separately owed; `CR2-X-1` still waits with the collector lane, blocking nothing.
- Reference figures unchanged: reviewed tree `6508941`, base `main` = `72e3a1c6`, **455 hermetic @ 95.75%** is
  what a repair branch must reproduce; this pass executed no suite and certified nothing about the wave's
  future bytes.

```text
CLOSURE-ROUTED: CR3-F-01 | absorbed | T018 step 1 (wave anchor; DC-CR3-F-01-r1)
CLOSURE-ROUTED: CR3-F-02 | absorbed | T019 step 1 (DC-CR3-F-02-r1)
CLOSURE-ROUTED: CR3-F-03 | absorbed | T018 step 2 (DC-CR3-F-03-r1)
CLOSURE-ROUTED: CR3-F-04 | absorbed | T019 step 2 (DC-CR3-F-04-r1)
CLOSURE-ROUTED: CR3-F-05 | absorbed | T020 step 1 (DC-CR3-F-05-r1)
CLOSURE-ROUTED: CR3-F-06 | accepted | QB:cli-surface/undocumented-flags-and-stats-db (§RM-005)
CLOSURE-ROUTED: CR3-F-07 | accepted | 057 F-1 deferral (owner + trigger stand)
CLOSURE-ROUTED: CR3-F-08 | accepted | §Accepted risks (digest system-only; base parity)
CLOSURE-ROUTED: CR3-F-09 | accepted | §Accepted risks (base parity, legacy:908)
CLOSURE-ROUTED: CR3-F-10 | accepted | §Accepted risks (works + relay superset)
CLOSURE-ROUTED: CR3-F-11 | accepted | roadmap.md §RM-009 + §Accepted risks + §Elevated to user
CLOSURE-ROUTED: CR3-F-12 | accepted | §RM-005 (tracked; re-confirmed)
CLOSURE-ROUTED: CR3-F-13 | accepted | §RM-005 (axis named; no duplicate row)
CLOSURE-ROUTED: CR3-F-14 | accepted | §Accepted risks (FK-009 fixture, documented)
CLOSURE-ROUTED: CR3-F-15 | accepted | QB:lint/ruff-f-select-findings
CLOSURE-ROUTED: CR3-F-16 | accepted | QB:legacy-monolith/in-file-provenance-header
```

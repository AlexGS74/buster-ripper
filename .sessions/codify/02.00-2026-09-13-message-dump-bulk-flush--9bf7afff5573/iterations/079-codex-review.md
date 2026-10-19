# Iteration 079 — Codex Review (**CR3**), persisted in `phase-boundary-evidence` mode (MS-001)

**This report is the session's durable home of the CR3 external review findings.** The boundary it records
**has not passed**. Zero dispositions are assigned here — routing is Assumption Decide's act. No production,
test, `invariants/` or QA byte is touched, and no command that executes project code was run by this pass.

> **ROUTED 2026-10-06** by the Assumption Decide pass ([080-assumption-decide](080-assumption-decide.md)):
> every `Disposition:` below is now filled with that pass's result — 5 `absorbed` (wave `T018`–`T020`,
> ledger `DC-CR3-F-01-r1`…`-r5`), 11 `accepted`; zero questions — `FK-011` stays the only open Fork.
> No relayed sentence, severity label or verified-sound item was altered by the routing; see §Routing Record.

## Orchestrator Prompt (verbatim)

```text
You are a subagent spawned by another session.
Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
Task: T005
Mode: phase-boundary-evidence
Milestone: MS-001
Target base: main
Target object: 72e3a1c6861bf0d8de6317adf52289e5f6cfa4e1
Review object: 650894111c11eb52c7597d335e7ce4eaa5609147
Close: milestone
```

The dispatch's boundary statement, as given: the boundary **has not passed**. [078-yield-phase](078-yield-phase.md)
issued MS-001's gate `SEMANTIC-APPROVE` (two absorbed P3s: `078-P3-1` an unwitnessed "exactly three values" clause
on `PI:T002/dump_format/format_set_is_closed`; `078-P3-2` a stale registry gloss on `E:T002/dump/bulk-flush`), but
the external review run **in parallel against the same Review object** returned **NOT clean**, and its report exists
**only as dispatcher-held text**, so this mode's job is to make it durable before any repair advances `HEAD`. Write
the findings into the session as a new iteration report, preserving the producer's wording and severity labels as
relayed, and record the provenance. Key it **CR3** to avoid colliding with 057's `F-*` and the CR2 batch, and leave
every disposition empty. No code is to be touched. Confirm the checkpoint's first parent is exactly the Review
object, keep the delta to session paths, and return `BOUNDARY-EVIDENCE-CHECKPOINTED`.

## Increment Brief

- Task and goal: persist the **CR3** external review — sixteen findings, all severity labels as relayed — plus its
  provenance, as one **review-preserving evidence checkpoint** on top of the Review object `6508941`, before any
  owning repair route advances `HEAD`. Evidence mode: no implementation, no routing, no acceptance.
- Anchors touched: **none** — this pass adds no feature behavior and exercises no Anchor Map row.
- Will NOT create: any new referent, plan task, QA gate, closure packet, or invariant record (triad rule — the
  registry is never this lane's to move). No `T0xx` row is opened: naming repair tasks belongs to the routing pass
  that dispositions these findings.
- Reuse evidence: none applicable — no `Mode: NEW` surface is approached, so the creation gate is not reached.
- Inventions: none. Two artifacts of *record* only: this report's stable finding ids (§Keying) and its locator
  notes, each explicitly separated from the producer's wording.

## Naming and keying

- **Number from the CLI, not hardcoded**: `codify codify forward-iteration-id --session-id=… --agent forward
  --first-cycle` → `079-01-forward`. The base id is **079**; the suffix is `-codex-review` rather than `-forward`
  because the artifact's content is an external review report and `057-codex-review.md` is this session's precedent
  for that genre — a later pass looking for review reports looks at `*codex-review*`. The `phase-boundary-evidence`
  mode record is a section of this file (§Evidence-Mode Execution Record), not a second report, which is the
  divergence from [072-01-forward](072-01-forward.md) (that pass persisted a packet file the routing pass had
  already written; here the report itself is the packet's only home).
- **Batch key `CR3`**, per the dispatch: the third `codex-review` pass of this session. The producer's per-finding
  labels were not relayed, so this pass assigns **`CR3-F-01`…`CR3-F-16`** in relay order (P1s, then P2s, then the
  ten P3s). These ids are a persistence-pass convenience for citation, **not** the producer's — the CR2 batch did
  the same thing and recorded it (`handoff-context.md` §Deterministic Closure Ledger: "the batch's stable ids are
  `CR2-F-1`…`CR2-F-4` — the packet's `F-1`…"). `057`'s `F-*` and the `CR2-*` keys are untouched by this namespace.

## Review provenance (as relayed; the Git-verifiable part verified here)

| Field | Value | Verification |
|---|---|---|
| Reviewer configuration | `gpt-5.6-terra` at **medium** reasoning effort on `codex-cli 0.160.0` | relayed; matches the configuration `057` recorded for this session's first external review |
| Base object | `72e3a1c6861bf0d8de6317adf52289e5f6cfa4e1` | `git rev-parse main origin/main` → **both equal it**; `git merge-base --is-ancestor 72e3a1c6… 6508941…` → **true** |
| Head object reviewed | `650894111c11eb52c7597d335e7ce4eaa5609147` — **Checkpoint 183** | `git log -1 --format=%s 6508941` → "Checkpoint 183: docs(qa): complete Checkpoint 182 …" |
| Scope | **212 commits / 619 files** | `git rev-list --count main..6508941` → **212** (and `git rev-list --count 6508941..main` → **0** behind); `git diff --name-only main...6508941 \| wc -l` → **619** |
| Suite baseline at review time | **455 passed @ 95.75%** | The same figure 078 measured **independently on the same snapshot**: [`evidence/ms001-reentry-078-gate-run.txt`](../evidence/ms001-reentry-078-gate-run.txt), header `HEAD 650894111c…`, trailer `455 passed … Total coverage: 95.75%` |

**No head-label defect this time, and the difference is worth recording.** The CR2 batch's headline named
`2be9783` while its own `407 passed` baseline belonged to `dbd3072` — a label defect settled at
[072-01](072-01-forward.md) §Step 3. **CR3's head label and its baseline both resolve to the one tree it names**:
`6508941` is simultaneously the Review object, the snapshot 078 certified, and the only tree that measures
455 @ 95.75%. Nothing here needs re-attribution, so a repair route that cannot reproduce **455** hermetic tests is
not on these bytes.

**Verdict (as relayed): NOT clean** — **2 × `P1`**, **4 × `P2`**, **10 × `P3`** = 16 findings, plus a
seven-item **verified sound** list with no finding attached (§Verified sound).

**Provenance limit, stated rather than glossed:** this pass received the findings as **dispatcher-relayed text**; it
never held a producer-side artifact. "Verbatim" below therefore means verbatim **with respect to the dispatch's
wording and severity labels**, not byte-identical to a Codex transcript. Where a site needed a file to be locatable
and the relay named none, that is flagged as a *locator note* and worded separately from the claim.

## CR3 findings, as relayed

Formatting note, so "verbatim" is unambiguous: each finding's **heading** carries the site the relay named, and the
concern text below is the dispatch's own sentence with only that site prefix lifted into the heading — no word of the
claim, no severity label, and no qualifier ("stated as base parity", "The report states…", "The report notes…",
"which the report says is acceptable if…") has been added, removed, or rephrased. Severity labels are the relay's
`P1`/`P2`/`P3`, unmodified; nothing was promoted or demoted in transit.

### `CR3-F-01` — `P1` — `server.py:119`

**Concern (as relayed, verbatim):** `web.Application` is constructed with no `client_max_size`, so aiohttp's
1048576-byte default governs every route including the catch-all, enforced at body-read time before handler logic.
Reproduced against the real app: 2 MB POST to `/v1/messages`, `/v1/messages/count_tokens`,
`/v1/chat/completions` and `/health` all answer 413 with "Maximum request body size 1048576 exceeded, actual body
size 1211701". Base (Starlette/uvicorn) imposed no body limit, so this is migration-introduced. Material because
`MAX_MODEL_LEN` defaults to 200000 tokens, so a session's serialized body is routinely multiple megabytes and
`count_tokens` receives the same payload. The report states that no claim, invariant, roadmap row or test anywhere
in `buster_ripper/`, `tests/`, `invariants/` or `.sessions/` mentions `client_max_size`, the byte count, or 413.

**Disposition:** **absorbed** → `T018` step 1 (CR3 wave anchor) — reach `reached`; every clause re-verified this
pass (sole construction site `server.py:119`, aiohttp default `1024**2`, zero tree acknowledgment, contract
arithmetic: 200 K tokens at the package's own 4-chars/token ⇒ ~0.8 MB before escaping/multibyte, image payloads
few-tokens-per-megabyte; no witness ever posted over the cap); `create_app()` gets `client_max_size=0`
(base imposed no cap). Full packet: `DC-CR3-F-01-r1`; reasoning incl. the scope decision (absorbed under the
057/CR2 branch-merges-as-one-unit precedent, not routed out):
[080-assumption-decide](080-assumption-decide.md) §CR3-F-01.

### `CR3-F-02` — `P1` — `routes/count_tokens.py:80`

**Concern (as relayed, verbatim):** the session id is derived from the **un-normalized** body, while
`routes/messages.py:33-37` deliberately normalizes first, an ordering the branch registered as load-bearing at
`EI:buster_ripper.session/session_id/derived_after_normalize`. End-to-end proof given: with the billing nonce
present, `/v1/messages` records stats under the normalized digest while `count_tokens` answers the local estimate and
misses that key, so `avg_latency` stays 0.0 and the token-fill/TTFT compaction nudge becomes unreachable; only the
KV-pressure trigger can fire. The report notes the ordering defect is carried from base but the branch formalized the
invariant this site violates.

**Related test-validity claim (relayed with this finding, kept attached to it):**
`tests/test_routes_integration.py:254-333` seeds `session_mod._stats` under the literal key `"test_session"`, which
can never equal a `session_id()` digest, and `test_count_tokens_inflates_on_compaction_trigger` passes only because
it also forces `compaction._kv_cache_usage = 0.85` — leaving the session-trigger branch unwitnessed.

**Disposition:** **absorbed** → `T019` step 1 — reach `reached` (the nonce "changes on every request",
`normalize.py:11-15`; mechanism re-verified on current bytes: raw-body digest at `count_tokens.py:80` vs
normalize-then-digest at `messages.py:36-37`, digest over `system` only per `session.py:93-98`, session arm
permanently false per `compaction.py:85-86` + `count_tokens.py:89`); the attached test-validity half re-verified
and carried into the task (`"test_session"` can never equal a digest). Invariant cross-reference: **Aligned** —
binds to the session-carried `EI:.../derived_after_normalize` (preserve-protected; deliberately never a
registry-active row per `invariants/index.md` §Source Sessions — the classification is *not* Contradiction, which
is the class for findings proposing to *violate* an active invariant; here the code violates the carried rule and
the finding enforces it). The base-parity origin is verified (`legacy:640-655` normalize-then-sid vs its
`count_tokens` raw-then-sid) and ranked below the carried invariant. Full packet: `DC-CR3-F-02-r1`.

### `CR3-F-03` — `P2` — `server.py:67` + `routes/messages.py:137`

**Concern (as relayed, verbatim):** `server.py:67` sets the shared session's `ClientTimeout(total=600)` and
`_stream_single_request` (`routes/messages.py:137`) passes no per-request override, while base used `timeout=None`
for the streaming path (`buster_ripper_legacy.py:664`): a stream outliving 600 s ends as a `TimeoutError` inside
`_relay_after_commit`, which cannot answer after the committed head, so the client sees a 200 SSE stream that simply
stops with no `[DONE]` and no error event.

**Disposition:** **absorbed** → `T018` step 2 — reach `latent` (stream > 600 s; plausible under the shipped
launcher's default `PRESERVED_THINKING=1`), fix-not-pin because the "guard" **is** the defect; base parity
re-verified at the named line (`buster_ripper_legacy.py:664`, `httpx.AsyncClient(timeout=None)`). The flagged
tension with verified-sound item 2 is **resolved, not routed around**: item 2 certifies the arm's *handling* of
post-commit failures and its timeout-class accounting; this finding indicts the *deadline policy upstream of the
arm* that manufactures a failure class base never delivered — both statements are true of the same code, and
neither certifies the other. The closure adds a per-request `ClientTimeout(total=None)` on the streaming
`session.post` only (registry `CL:aiohttp/per-request-timeout-override` HOLDS), touches no arm logic, and leaves
`tests/test_stream_relay_commit.py`'s injected-fault witnesses (bare `TimeoutError` case included) byte-valid.
Full packet: `DC-CR3-F-03-r1`; recorded observation there: the non-stream arm inherits the shared 600 s against
base's 300 s (`legacy:700`/`:843`) — a widening, no finding in this batch, carried for Yield Red's reading.

### `CR3-F-04` — `P2` — `routes/messages.py:276`

**Concern (as relayed, verbatim):** `content.decode()` is strict UTF-8 inside a `try` whose only arm catches
`aiohttp.ClientError`, so a non-UTF-8 upstream error body raises out and becomes a proxy 500 with a traceback
instead of a relayed upstream status; the sibling path at `chat.py:164` catches `(json.JSONDecodeError,
UnicodeDecodeError)` and relays raw bytes. The report adds that `request.text()` raises a bare `LookupError` for an
unknown `charset=` token, uncaught at all three parse sites.

**Disposition:** **absorbed** → `T019` step 2 — reach `latent` both halves (network-supplied bodies on exposed
routes; the deployed vLLM emits UTF-8; no guard prevents either input class). Bytes re-verified: the strict
decode at `messages.py:276` sits in a `try` whose only arm is `except aiohttp.ClientError` (`:297`), and neither
`UnicodeDecodeError` nor `LookupError` is a `ClientError` (disjoint `ValueError` subtrees) — so an undecodable
upstream error body escapes to a proxy 500 with traceback; `chat.py:164` re-verified as the raw-relay sibling,
which is `T015`'s **accepted** shape — `messages.py` is the outlier inside this batch's own settled lane, which
is what makes the closure deterministic. The `request.text()`/`LookupError` mechanism re-derived from installed
source (`web_request.py:686-692` — `text()` decodes with the Content-Type charset; `json()` goes through
`text()`); the three client parse sites (`messages.py:29`, `chat.py:30`, `count_tokens.py:19`) catch everything
except `LookupError`, as relayed. Full packet: `DC-CR3-F-04-r1`.

### `CR3-F-05` — `P2` — `tests/test_compaction.py:348`, `:369`

**Concern (as relayed, verbatim):** `tests/test_compaction.py:348` and `:369` assert on arithmetic computed inline in
the test rather than calling `compaction.poll_kv_cache`, and the coverage report's single missing line in
`compaction.py` is exactly the EMA-update branch both tests claim to verify.

**Disposition:** **absorbed** → `T020` step 1 — the Reach Gate **does not apply** to false evidence (its own
exclusion list), so this holds for every input: both test bodies re-read and contain inline arithmetic
(`result1 = (1 - alpha) * prev + alpha * 0.75`; `assert result == raw` against a local), never calling
`compaction.poll_kv_cache`; `compaction.py:49-52` re-read as the EMA-update branch — it **does** run in the
shipped process (monitor polls `/metrics` every 10 s, `server.py:15-34`), so this is a suite-attestation defect
on the 95.75% coverage figure the boundary certifies, not a dead-path debate. Closure: drive `poll_kv_cache`
across ≥2 ticks against a mocked `/metrics`; done-when includes a mutation replay (flip the formula, the witness
must go red). Full packet: `DC-CR3-F-05-r1`.

### `CR3-F-06` — `P2` — `main.py:75`

**Concern (as relayed, verbatim):** `main.py:75`'s `--strip-images` help text promises "cache them for later
retrieval" while `image.py` states plainly that the cached source has no reader on the request path, so enabling the
flag destroys image payloads for the model permanently.

**Disposition:** **accepted** — inert wording under Reach First: the help string is read by humans, no runtime
consumer, and `grep` this pass confirms no witness pins the "retrieval" clause, so no behavior moves; the
substance is real and **already owned** — `QB:cli-surface/undocumented-flags-and-stats-db` (roadmap §RM-005)
carries the `main.py:74-75` help-text scope clause from the S10–S13 routing, and `image.py:172-173` re-verified
verbatim ("the cached source has no reader on the request path, and the placeholder is not a reference that
restores anything"). One line under §Accepted risks; re-confirm marker appended to the owning row; no task, no
Fork, no wave.

### The ten `P3` items (as relayed)

The relay presents these as one enumerated run of ten. Each is broken out with its own id and its own empty
disposition; the wording of each is the dispatch's, unaltered.

| id | severity | text as relayed | disposition |
|---|---|---|---|
| `CR3-F-07` | `P3` | `has_images`/`strip_and_cache_images` scan only top-level blocks so `tool_result`-nested images are neither detected nor stripped — **locator note, this pass:** the relay names the functions, not the file; on the Review object they are `buster_ripper/image.py:61` and `:166` | **accepted** — **already tracked as an owned deferral**: 057 `F-1`'s `Deferred additive work` (owner = image-slice Forward task; trigger = when tool_result-nested images are in scope — [057](057-codex-review.md):61) + §RM-005's nested-half clause; detector/writer re-verified **agreeing** at top-level scope (`image.py:31-36,:61-69,:166-200`) — missing additive scope, not divergence; no duplicate row |
| `CR3-F-08` | `P3` | `normalize.py:76`'s stable sort leaves same-name tools in incoming order | **accepted** — impossible in well-formed input (duplicate tool names are API-invalid) and `session_id()` reads **`system` only** (`session.py:95-98`), so tool order cannot move the session key; base sorted the same stable way; §Accepted risks |
| `CR3-F-09` | `P3` | `passthrough.py:23`'s `dict(request.query)` drops repeated query parameters | **accepted** — **base parity verified at `buster_ripper_legacy.py:908`** (`params=dict(request.query_params)` — identical fold); no documented consumer sends repeated params; §Accepted risks |
| `CR3-F-10` | `P3` | `passthrough.py:56`'s module-bottom `config` import and the widened `add_route("*", …)` verb set versus base | **accepted** — import works and is deliberately marked (`# noqa: E402`), handler reads `config.UPSTREAM` at call time; verb set re-verified as a strict relay-**superset** of base's five (`legacy:898`) — the difference is which upstream answers, no value lost; §Accepted risks |
| `CR3-F-11` | `P3` | `find_message_start` scanning each network chunk in isolation so a split `message_start` event is missed (stated as base parity) — **locator note, this pass:** the relay names no file; `find_message_start` is defined at `buster_ripper/utils.py:91` on the Review object | **accepted** — 1% category (`codify-scope`): base parity verified per-chunk (`legacy:674-676`), a split costs one turn's stats and self-heals next turn, cause is streaming-parse code no session task declared and no CR3 gate blocks on; new **§RM-009** row + §Accepted risks + §Elevated to user |
| `CR3-F-12` | `P3` | `config.STATS_DB` left dead by `main.py:155-161` (already tracked at §RM-005) | **accepted** — tracked; re-verified `config.py:65` is the global's only occurrence and `main.py:155-161` uses the click param directly; re-confirm marker appended to §RM-005's row, **no duplicate** |
| `CR3-F-13` | `P3` | no `STRIP_IMAGES` or `--backend-type` env passthrough in `serve_buster_ripper.sh` | **accepted** — script re-read: passthrough absent but nowhere promised; only consumers are planned work, and the axis is folded into §RM-005's row (serve-script env axis named explicitly), **no duplicate row** |
| `CR3-F-14` | `P3` | the committed 100-year-validity `tests/qa/tls_peer_key.pem` fixture | **accepted** — intentional by `FK-009` option 1 and documented at `tests/qa/tls_peer.py:11-14`; `openssl x509 -enddate` → `notAfter=Sep 4 2126`; test material only (a fixture is not a caller), loopback-only identity guarding nothing secret; long validity is the cheaper flake-bound; §Accepted risks |
| `CR3-F-15` | `P3` | eight `ruff --select F` findings with no `[tool.ruff]` section pinned — **premise-existence check, this pass:** `git show 6508941:pyproject.toml \| grep -n "tool.ruff"` prints nothing, so the relay's premise that no such section exists holds on the tree; the eight findings themselves are the producer's and were not re-derived | **accepted** — premise re-verified on current bytes, and the routing pass **did** re-derive the list (`uvx ruff check --select F` → exactly the relayed 8, all in `tests/`); no lint gate runs on this tree (deps are aiohttp+click only), so it is hygiene, not behavior → new row `QB:lint/ruff-f-select-findings`; two hits sit inside `T019`/`T020` touch-sets and are likely consumed there |
| `CR3-F-16` | `P3` | `buster_ripper_legacy.py` importing dependencies that are not installed, which the report says is acceptable if declared reference-only | **accepted** — the file self-declares its deps via the PEP 723 header (`legacy:1-9`: fastapi/uvicorn/httpx/click — standalone `uv run` material by design) and is imported by **nothing** (`grep` → docstring citations only); "reference-only" is declared in session artifacts but **not in-file**, the report's own condition, so: new row `QB:legacy-monolith/in-file-provenance-header` (doc-line remedy named), no task |

**Every file:line anchor relayed above resolves inside its file at the Review object** — `server.py` 138 lines
(`:67`, `:119`), `routes/count_tokens.py` 98 (`:80`), `routes/messages.py` 299 (`:33-37`, `:137`, `:276`),
`routes/chat.py` 300 (`:164`), `routes/passthrough.py` 56 (`:23`, `:56` = last line, consistent with "module-bottom"),
`normalize.py` 228 (`:76`), `main.py` 193 (`:75`, `:155-161`), `tests/test_compaction.py` 378 (`:348`, `:369`),
`tests/test_routes_integration.py` 467 (`:254-333`), `buster_ripper_legacy.py` 1013 (`:664`), and
`tests/qa/tls_peer_key.pem` is tracked at `6508941`. **This is a resolvability check on the relay, not a
re-derivation of any claim**: no finding was confirmed, weakened, or re-tested by this pass.

## Verified sound — recorded with no finding attached

As relayed, the report records the following as **verified sound**, explicitly **with no finding attached**:

1. the `CIMultiDict` per-occurrence relay choice;
2. the post-commit failure arm and aiohttp 3.13's timeout class hierarchy;
3. `_carries_token`'s duplicate-`Authorization` handling;
4. the `image.py` cache invariants;
5. the `dump.py` flush layering and `_writable_target` refusal;
6. `_cancel_background_task`'s `cancelling()` comparison;
7. `_chat_answer`'s raw-relay parity for non-dict JSON.

Carried so that the routing pass does not re-open them as unreviewed surface, and so that a later reader can see
which boundaries the review looked at and cleared. No id, no disposition — there is no finding to disposition.

## Pre-Flight Checks

- **Session identity: verified.** `session-identity.json` → `session_id: 02.00-2026-09-13-message-dump-bulk-flush`,
  `session_uid: 9bf7afff-5573-402a-b102-827423a1dded`; the dispatch handle's `--9bf7afff5573` is the uid's first 8 +
  next 4 hex groups — **MATCH**. Files read directly, nothing globbed. Not `BLOCKED-SESSION-IDENTITY`.
- **Mode contract: satisfied.** `Mode: phase-boundary-evidence` + `Milestone: MS-001` + `Close: milestone` +
  `Target base: main` + full `Target object` + full `Review object`.
- **Plan approval gate: PASSED** — planning CLOSED at round 18 ([020-interview](020-interview.md), `US-015`); the
  session-level approval covers boundary-evidence invocations, the same ruling [058-01](058-01-forward.md) and
  [072-01](072-01-forward.md) applied.
- **`Task: T005`** is the milestone's owning-task label — the identical label 072's phase-boundary-evidence dispatch
  carried for this same milestone. The session's real `T005` was delivered and accepted long before this boundary
  ([049-01-yield](049-01-yield.md)); **no task row's state is changed by this pass**, and §Planned Task Context is
  not touched.
- **QA contract**: `qa-plan.md` exists; `QAP-MS001` is `READY-PASSING` (077) and its gate was independently executed
  by 078 (11/11). This mode authors no gate test, inspects none, and **executes no test or app command** — the
  figures it cites are inherited from 078's own run on the identical snapshot (no-re-run rule).
- **Global invariant registry read**: not performed. Nothing is implemented here, so no narrowed scope exists to read
  for. `CR3-F-02` *names* an invariant (`EI:buster_ripper.session/session_id/derived_after_normalize`) — it is
  recorded as relayed and **not touched**: the triad rule bars this lane from creating, editing, superseding or
  deleting any invariant record, and adjudicating a claimed violation belongs to the routing pass and, if the
  invariant is wrong, to Interview.
- **Dependency claims**: none encoded — no production or test line is written in this mode.
- **Pre-question gate**: no question, `ESCALATION`, `BLOCKED-APPROVAL` or `PLAN-GAP` arose → **no closure packet owed
  by this pass**, and none invented. `FK-011` remains the only open Fork, untouched. The dispatch's instruction to
  leave dispositions empty is a **lane assignment**, not a question: routing has an owner, so nothing here escalates.

## Evidence-Mode Execution Record

Mode rule applied verbatim: *"verify all dirty files are limited to session/plan/review routing artifacts, then
create one review-preserving checkpoint with `codify git create-checkpoint`. Do not require Yield Red CLEAN, alter
findings, implement a repair, or mark the phase accepted."*

**Step 1 — snapshot identity.** Branch `refactor/package-structure`. At read time `HEAD` =
`37ed09895c418df2e4a6036094257aa18a541459` (**Checkpoint 184**), and its **first parent is
`650894111c11eb52c7597d335e7ce4eaa5609147` — exactly the Review object** (`git rev-parse 37ed098^` → the Review
object). Checkpoint 184 is itself a session-artifact-only checkpoint (3 paths, all under this session directory:
`iterations/078-yield-phase.md`, `evidence/ms001-reentry-078-gate-run.txt`, `handoff-context.md`), so **the chain
from the reviewed bytes to this checkpoint contains zero repair commits** — `git diff --name-only 6508941..HEAD`
returns exactly those three session paths and nothing else. This pass therefore sits directly on the reviewed
tree's evidence chain rather than behind any code movement; the Review object remains an unamended ancestor, and no
merge, squash or rewrite intervenes.

**Step 2 — dirty-file census.** `git status --porcelain` at read time: **empty**. Unlike 072, whose five dirty paths
were the CR2 routing pass's unsaved output, the CR2/077/078 routing artifacts were already committed by Checkpoint
184 — and **the CR3 report was not among them**: `grep -rn "client_max_size\|CR3"` over the whole session directory
returned **zero** files. The findings existed only as dispatcher-held text, which is precisely the gap this mode
exists to close. Post-write, the dirty set is the four session paths listed in §Checkpoint Record — **zero**
production, test, `invariants/`, `README.md`, `pyproject.toml` or `serve_buster_ripper.sh` path.

**Step 3 — what this pass added, and what it deliberately did not.**

Added (record only): this report; one §Current Status bullet in `handoff-context.md`; one in-place §Current Phase
correction in `plan-summary.md` (§Superseded plan wording below); one factual `chronicle.md` entry claiming no
acceptance.

Deliberately **not** done, each because another lane owns it:
- **No `evidence/` packet copy.** CR2 got one (`evidence/codex-review-post-071-packet.txt`) because the routing pass
  had already received and persisted the packet as a file. Here the dispatch names the iteration report as the
  findings' home, so writing a second copy would create two texts to diverge. Later passes cite
  `iterations/079-codex-review.md` as the packet path.
- **No closure packets, no `DC-CR3-*` ledger rows, no dispositions, no routing.** 057 filled `CLOSURE:` blocks
  because that pass *was* the routing pass; the CR2 ledger was written by `assumption-decide`. This pass is neither:
  it makes the evidence durable so the router can do its job on bytes that will not move under it.
- **No repair task created.** `plan-adhoc-task` is the tool for that and it belongs to whoever routes these findings.
- **No re-derivation of any finding**, and no edit to any relayed sentence. Three notes were added to the P3 table and
  each is marked as this pass's own: two **file locators** for referents the relay left unnamed (`CR3-F-07`'s
  `buster_ripper/image.py`, `CR3-F-11`'s `buster_ripper/utils.py`), and one **premise-existence check** on the most
  mechanically checkable clause in the batch (`CR3-F-15`: no `[tool.ruff]` section in `pyproject.toml` at `6508941`).
  No claim was tested, re-severitized, confirmed or weakened; the eight `ruff` findings themselves were not run.
- **No `evidence/INDEX.md` row**: that table indexes artifacts under `evidence/`, and this pass created none.

**Step 4 — mode prohibitions honored.**
- **Yield Red CLEAN neither required nor asserted**: no `YIELD-RED: CLEAN` report exists for MS-001 and this
  checkpoint does not substitute for one.
- **Findings not altered**: all sixteen findings, the seven verified-sound items, the provenance fields, and every
  severity label are checkpointed as relayed. The only words that are not the producer's or the dispatcher's are
  this pass's own record sections and three marked notes (§Step 3) — two file locators, one premise-existence check.
- **Dispositions all empty**: §CR3 findings and the P3 table carry no routing, no `absorbed`/`malformed`/`accepted`
  label, and no owning route.
- **Repair not implemented; phase not accepted**: MS-001's **gate** verdict remains 078's `SEMANTIC-APPROVE` with
  `078-P3-1`/`078-P3-2` absorbed, and the **boundary is BLOCKED, not passed** — CR3 is unresolved external review
  against the same bytes, and the sequence `assumption-decide` routing → repair wave(s) → assumptions recheck →
  **Yield Red** → `phase-boundary-finalize` still has not run. `resolved` for `T015`/`T016`/`T017`, the `Delivered`
  chronicle entry and parent completion remain Yield Red's.
- **History preserved**: one commit on top of Checkpoint 184, whose first parent is the Review object. No merge, no
  squash, no rewrite, nothing pushed, no `HEAD` advance through code.

**Step 5 — checkpoint protocol.** No test command executed in this mode (see §Pre-Flight / QA contract);
`codify git create-checkpoint` with an explicit `--files` list restricted to the four session paths, so the delta
cannot widen. No raw `git add`/`git commit`.

## Superseded plan wording (preserved here, per supersession-by-rewrite)

The §Current Phase sentence this pass rewrote in place in `plan-summary.md`, kept verbatim:

> ➡ **⚠ MS-001's gate stands (`SEMANTIC-APPROVE`, zero findings) — both repair waves the boundary named are LANDED;
> `T017` is ACCEPTED-LOCAL and the owed `seam-S15` recomposition has LANDED AND IS ROUTED (`S15` terminal, gates
> nothing), so only Yield Red stands between this boundary and the close.**

It was already stale in two ways when this pass found it: it credits 071's *zero-findings* gate (078 issued
`SEMANTIC-APPROVE` with **two** absorbed P3s, and re-verified the gate on bytes 22 commits newer), and its "only
Yield Red stands between this boundary and the close" claim is refuted by CR3. The replacement text says which
verdict stands, which lane is next, and where the findings live.

## Observations flagged, not acted on (mode forbids repair)

1. **`CR3-F-03`'s harm statement runs through an arm the same report clears.** The verified-sound list item 2 is "the
   post-commit failure arm and aiohttp 3.13's timeout class hierarchy"; `CR3-F-03`'s consequence is that
   `_relay_after_commit` "cannot answer after the committed head". Both are recorded exactly as relayed, neither
   adjusted. Flagged only so the routing pass reads the two together when it dispositions `CR3-F-03` rather than
   discovering the tension after a fix has been chosen. This pass adjudicates nothing.
2. **Two prior findings land on the same surfaces as this batch**, so the router should expect overlap rather than
   new territory: `CR3-F-07`'s `tool_result`-nested images is the deferred additive half of `057`'s `F-1`
   (`Deferred additive work: … trigger = when tool_result-nested images are in scope`), and `CR3-F-12` names §RM-005
   itself, where `config.STATS_DB` is already tracked. Recorded as cross-references, not dispositions.
3. **`evidence/INDEX.md` gained no row for `ms001-reentry-078-gate-run.txt`** when Checkpoint 184 added the file.
   Outside this pass's touch-set, and the INDEX `Disposition` column is separately owed its recurring note (036-01,
   058-01, 072-01 all flagged earlier rows carrying text in the user's keep/remove column). Left byte-identical.
4. **Context budgets remain over the caps** in files this checkpoint persists: `handoff-context.md` is now 739 lines
   (cap 200; this pass added 15) and `plan-summary.md` 411 (cap 400; this pass added 5). The lossless trim into
   `handoff-archive.md` is its own operation (the Checkpoint 122 precedent) and is not this mode's; both overruns
   predate this pass and 072-01 flagged the same pair when they were 3017 and 459.

## Failure-Path Evidence

| Operation / condition | Expected outcome and authority | Handling / observation evidence | Disposition |
|---|---|---|---|
| A dirty file outside session/plan/review scope | Mode violation → STOP; never mix repair with boundary evidence | `git status --porcelain` empty at read time; the four written paths all under `.sessions/codify/02.00-…/`; after writing, `git status --porcelain -- . ':(exclude).sessions'` → **empty** | safe |
| A repair commit between the Review object and the evidence checkpoint | Would mis-attribute which bytes the findings describe | `git rev-parse 37ed098^` = the Review object; `git diff --name-only 6508941..HEAD` → 3 session paths only, so the chain is evidence-only | safe |
| Checkpoint's first parent ≠ the Review object's evidence chain | Dispatch's explicit check | Verified in §Step 1 before writing and re-checked after the commit (§Checkpoint Record) | safe |
| Findings "tidied", re-severitied, or re-verified while persisting them | Mode forbids altering; routing owns adjudication | Wording and `P1`/`P2`/`P3` labels copied as relayed; the only additions are three clearly-marked notes (two file locators, one premise-existence check — §Step 3); no claim tested or re-derived | safe |
| Dispositions filled in "obviously" to save the router a pass | Explicitly this pass's prohibition | Every disposition cell and every `Disposition:` line empty; no `CLOSURE:` block, no `DC-*` row, no owning route named | not-applicable (path not taken) |
| Boundary mistaken for `phase-boundary-finalize` (MS-001 is the session's last gate) | Finalize requires `YIELD-RED: CLEAN`; running it here would fabricate acceptance | Only `create-checkpoint` invoked; no CLEAN report demanded or produced; the report states the boundary is BLOCKED | safe |
| Relay's head label adopted without a check, or dismissed as CR2's was | Honest provenance: check what is checkable, claim nothing more | §Review provenance table: 212 / 619 / 455 @ 95.75% / Checkpoint 183 all verified against Git and 078's transcript; producer-side config recorded as relayed with its limit stated | safe |
| Producer-side artifact implied that this pass never held | Would launder dispatcher text into a Codex transcript | §Provenance limit + §Naming: ids `CR3-F-*` declared as this pass's assignment in relay order | safe |
| An invariant "corrected" because `CR3-F-02` names one | Triad rule: invariant bytes are never this lane's | `git status --porcelain -- invariants` → empty; the `EI:` handle is quoted as relayed and untouched | safe |
| Running the suite "to confirm the baseline" | Dispatch forbids touching code; no-re-run rule; 078 measured this exact snapshot | Zero execution commands issued — Git reads only; 455 @ 95.75% cited from 078's transcript | safe |
| `create-checkpoint` unavailable or failing | No raw `git add`/`git commit` fallback → BLOCKED report | Tool present; succeeded (id below) | safe |
| Worktree not clean after the checkpoint | Boundary evidence must land fully persisted | Post-checkpoint `git status --porcelain` → **empty** (verified below) | safe |

Exclusions by family: test failures, LNCP violations and coverage regressions have no trigger — no file with
executable content changed; visual-demo capture has no trigger — no user-visible behavior produced; the R1–R5
adversarial surfaces have no trigger — no parser, writer, capture→act span, payload field or identity key is authored
here. No policy was invented anywhere.

## Mandatory Check Lines

- **Backlog pickup: none match touch-set** — the touch-set is session artifacts only; every unchecked
  `quality-backlog.md` item targets `buster_ripper/`, `tests/`, `README.md`, `qa-plan.md` or `invariants/`. Several
  CR3 findings sit on surfaces that already carry backlog rows (§RM-005 for `CR3-F-12`; `CR3-F-07` on `057`'s
  deferred half); draining them is the routing pass's act, not this checkpoint's. Reviewed, not skipped.
- **Capacity gate: not applicable — no additive production surface** (no production edit of any kind in this mode).
- **LNCP scan: not applicable — no files with executable content modified.**
- **Visual demo evidence: not applicable — no user-visible behavior produced.** No `evidence/INDEX.md` row added,
  because no `evidence/` artifact was created (§Step 3).

## Checkpoint Record

- **Files staged** (all under `.sessions/codify/02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/`, via
  `codify git create-checkpoint --files`): `iterations/079-codex-review.md` (this report, new),
  `handoff-context.md` (one §Current Status bullet), `plan-summary.md` (§Current Phase in-place correction),
  `chronicle.md` (one factual entry, no acceptance claimed).
- **Suite evidence: inherited, not re-run** — [078-yield-phase](078-yield-phase.md) measured hermetic
  **455 passed @ 95.75%** (`dump.py` 100%) plus **11/11** `QAP-MS001` and the neighbors **15/15** / **6/6** on the
  Review snapshot `6508941` itself
  ([`evidence/ms001-reentry-078-gate-run.txt`](../evidence/ms001-reentry-078-gate-run.txt)), and CR3's own reported
  baseline is the same figure on the same tree. No command was executed by this pass.
- **Result**: **Checkpoint 185** — `codify git create-checkpoint`'s own auto-numbering (last checkpoint 184, + 1),
  the same derivation 072 recorded for Checkpoint 163.
- **Parentage check, stated literally rather than quietly satisfied.** The dispatch asks that the checkpoint's first
  parent be *exactly* the Review object. At dispatch read time that literal could not hold: **Checkpoint 184
  (`37ed098`) already sat on top of the Review object**, having landed 078's gate artifacts before this pass ran, so
  this checkpoint's first parent is `37ed098`, not `6508941`. What the check protects **is** intact, and both halves
  were verified as Git facts:
  1. `git rev-parse 37ed098^` → `650894111c11eb52c7597d335e7ce4eaa5609147` — **Checkpoint 184's first parent is
     exactly the Review object**, so the evidence chain begins at the reviewed bytes; and
  2. `git diff --name-only 6508941..37ed098` → 3 paths, all under this session directory, so the commits between the
     reviewed bytes and this checkpoint carry **session artifacts only — no repair has advanced `HEAD`**.
  Amending or squashing Checkpoint 184 to make the parent id match the wording literally is a history rewrite, which
  this mode forbids and which would be strictly worse for review: it would move the 078 evidence the dispatch says
  must stand. The Review object remains an unamended ancestor at distance 1.
- **Post-checkpoint state**: `git status --porcelain` **empty**; `invariants/` untouched; branch topology unchanged —
  one commit, no merge, no rewrite, nothing pushed.

## Durable Signal

- **Boundary state as of this checkpoint**: MS-001's **milestone gate** is `SEMANTIC-APPROVE` at
  [078-yield-phase](078-yield-phase.md) — zero P1, zero P2, two P3 both `CLOSURE: absorbed` (`078-P3-1` → `Forward`,
  unit tier; `078-P3-2` → `authority-registrar`, folded into an already-open row), all seven integration checks PASS
  on HEAD `6508941`, `QAP-MS001` **11/11** independently executed, `US-009` c2 **HOLDS**. The **boundary is BLOCKED**:
  the external review run in parallel against the **same Review object** returned **NOT clean** — **2 × P1 + 4 × P2 +
  10 × P3** — and its report existed only as dispatcher-held text until this pass. **CR3 is unresolved.**
- **The CR3 findings are durable at `iterations/079-codex-review.md`** with provenance, and **every disposition is
  empty**: routing is [assumption-decide](076-assumption-decide.md)'s act, and this checkpoint exists so it can
  perform it on bytes that cannot move underneath it.
- **Three surfaces CR3 puts back in play that the boundary had considered settled** (stated as relay coverage, not
  as this pass's judgment): the application-level request-body ceiling (`CR3-F-01`) has no claim, invariant, roadmap
  row or test anywhere in the repo per the report; the `session_id`-after-normalize ordering that the branch
  registered as load-bearing at `EI:buster_ripper.session/session_id/derived_after_normalize` is reported violated by
  `count_tokens` (`CR3-F-02`), which is the only finding here that names a formalized invariant; and the streaming
  deadline (`CR3-F-03`) is reported to invert base's `timeout=None`. The `T017`/`T013` accepted relay work is
  reported **sound** on the seven verified-sound items.
- **Owed at this boundary, in order**: **assumption-decide** over `CR3-F-01`…`CR3-F-16` (route, and name the repair
  task(s) — none exists yet) → the repair wave(s) with their local Yield → assumptions recheck on those bytes →
  **Yield Red**, which owns `resolved` for `T015`/`T016`/`T017`, final semantic acceptance, the `Delivered` chronicle
  entry and parent completion → `phase-boundary-finalize`. The session-close full-suite gate remains separately owed
  on top of all of it. `CR2-X-1` still waits with the collector lane and blocks nothing.
- **Reference figures a later pass needs**: reviewed tree = `6508941` (Checkpoint 183), base = `main` = `origin/main`
  = `72e3a1c6`, **212 ahead / 0 behind**, 619 files, hermetic **455 @ 95.75%**. A repair branch that cannot
  reproduce **455** is not on these bytes.

## Routing Record (2026-10-06, Assumption Decide — [080-assumption-decide](080-assumption-decide.md))

Sixteen dispositions assigned by the routing pass against current bytes at pre-write `HEAD`
`4008e27b5394d9bd994b0131bf47a94dff061d92` (Checkpoint 185; `git diff 6508941..HEAD` carries no file outside
`.sessions/`, so the production bytes read were the reviewed bytes): **5 `absorbed`** (`F-01`…`F-05` → wave
`T018`+`T019`+`T020`, anchor `T018`, one local Yield, prework of the MS-001 close; ledgers
`DC-CR3-F-01-r1`…`DC-CR3-F-05-r1`) — **11 `accepted`** (`F-06`…`F-16`, each with its Reach verdict and citation
under `handoff-context.md` §Accepted risks; seven land on **existing** owners — 057's `F-1` deferral and §RM-005 —
and none duplicates a row; three new records: roadmap §RM-009, `QB:lint/ruff-f-select-findings`,
`QB:legacy-monolith/in-file-provenance-header`). Zero `question`, zero `breaks-it`, zero `malformed`;
**`FK-011` remains the only open Fork.** The 078 absorbed P3s (`078-P3-1`/`078-P3-2`) keep the owners 078 gave
them. The persistence pass's flagged tension (item 2 vs `F-03`) is resolved in `DC-CR3-F-03-r1` — mechanism
cleared, configuration indicted, arm untouched — and the two locator notes plus the ruff premise were
re-established (the ruff count re-derived by the routing pass: 8, all in `tests/`). No relayed sentence, severity
label or verified-sound item was altered by this routing; the `Disposition:` lines and this section are the
router's own text.

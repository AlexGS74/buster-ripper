# Session Roadmap — 02.00-2026-09-13-message-dump-bulk-flush

Seeded by `decide` (019) at plan finalization. This register holds honest,
above-the-floor **deferred scope** for this session — a map, not debt. It does not
gate session completion, and `quality-backlog.md` (debt on built work, must drain)
is a separate file.

## Promotion Register — step 14 (2026-10-18)

Every row below was re-evaluated at final head `191e676` and promoted onward to
`.sessions/codify/roadmap/2026-10-18-message-dump-bulk-flush.md`; none is obsolete and
none is wholly absorbed (RM-005's `--preserved-thinking` half is the only partial
absorption). The per-section status lines carry the same receipt.

| Handle | Section | Status | Outcome |
|---|---|---|---|
| RM-013 | §Switch the default dump format | `promoted-onward` | `.sessions/codify/roadmap/2026-10-18-message-dump-bulk-flush.md` `RM-013` |
| RM-014 | §Dump machine record | `promoted-onward` | `.sessions/codify/roadmap/2026-10-18-message-dump-bulk-flush.md` `RM-014` |
| RM-015 | §Human-facing dump tool | `promoted-onward` | `.sessions/codify/roadmap/2026-10-18-message-dump-bulk-flush.md` `RM-015` |
| RM-016 | §Chat route's post-commit second head | `promoted-onward` | `.sessions/codify/roadmap/2026-10-18-message-dump-bulk-flush.md` `RM-016` |
| RM-001 | §RM-001 | `promoted-onward` | `.sessions/codify/roadmap/2026-10-18-message-dump-bulk-flush.md` `RM-001` |
| RM-002 | §RM-002 | `promoted-onward` | `.sessions/codify/roadmap/2026-10-18-message-dump-bulk-flush.md` `RM-002` |
| RM-003 | §RM-003 | `promoted-onward` | `.sessions/codify/roadmap/2026-10-18-message-dump-bulk-flush.md` `RM-003` |
| RM-004 | §RM-004 | `promoted-onward` | `.sessions/codify/roadmap/2026-10-18-message-dump-bulk-flush.md` `RM-004` |
| RM-005 | §RM-005 | `promoted-onward` | `.sessions/codify/roadmap/2026-10-18-message-dump-bulk-flush.md` `RM-005` |
| RM-006 | §RM-006 | `promoted-onward` | `.sessions/codify/roadmap/2026-10-18-message-dump-bulk-flush.md` `RM-006` |
| RM-007 | §RM-007 | `promoted-onward` | `.sessions/codify/roadmap/2026-10-18-message-dump-bulk-flush.md` `RM-007` |
| RM-008 | §RM-008 | `promoted-onward` | `.sessions/codify/roadmap/2026-10-18-message-dump-bulk-flush.md` `RM-008` |
| RM-009 | §RM-009 | `promoted-onward` | `.sessions/codify/roadmap/2026-10-18-message-dump-bulk-flush.md` `RM-009` |
| RM-010 | §RM-010 | `promoted-onward` | `.sessions/codify/roadmap/2026-10-18-message-dump-bulk-flush.md` `RM-010` |
| RM-011 | §RM-011 | `promoted-onward` | `.sessions/codify/roadmap/2026-10-18-message-dump-bulk-flush.md` `RM-011` |
| RM-012 | §RM-012 | `promoted-onward` | `.sessions/codify/roadmap/2026-10-18-message-dump-bulk-flush.md` `RM-012` |
| RM-017 | §RM-017 | `promoted-onward` | `.sessions/codify/roadmap/2026-10-18-message-dump-bulk-flush.md` `RM-017` |
| RM-018 | §RM-018 | `promoted-onward` | `.sessions/codify/roadmap/2026-10-18-message-dump-bulk-flush.md` `RM-018` |
| RM-019 | §RM-019 | `promoted-onward` | `.sessions/codify/roadmap/2026-10-18-message-dump-bulk-flush.md` `RM-019` |
| RM-020 | §RM-020 | `promoted-onward` | `.sessions/codify/roadmap/2026-10-18-message-dump-bulk-flush.md` `RM-020` |
| RM-021 | §RM-021 | `promoted-onward` | `.sessions/codify/roadmap/2026-10-18-message-dump-bulk-flush.md` `RM-021` |

Handles `RM-013`–`RM-016` are minted at this first promotion from the four
sections that never carried one; `RM-001`–`RM-012` are retained verbatim.
`RM-017`–`RM-020` are minted by `T024`'s session-end Refactoring wave (2026-10-18)
and promoted onward in the same motion, assessed on the production bytes at `54001b6`.
`RM-021` is minted by the step-13 closing check (`121-closing-check-codex`, 2026-10-18)
after the step-14 promotion ran, and landed at its destination in the same pass;
assessed on the production bytes at `96264eb`.

**PR #1 review round 1 (2026-10-19, `0d465ec`) — three rows taken up and one minted.**
The user's round-1 instruction (*fix every finding that is not in the 1%/0.01% category*) overrides
the recorded deferrals on `RM-010` (timeout class), `RM-016` (chat post-commit) and `RM-020` (unfixed
re-serialization). Those three are routed to lite briefs in this PR, with the wave order recorded in
`handoff-context.md` §Planned Task Context; the rows stay `open` until the fixing commits exist.
`RM-022` is minted by the same pass for the one finding that remains a 1% record — the collapsed artifact's
numbering across a same-format restart — assessed on the production bytes at `0d465ec`.

## MVP floor

The floor is what this session must deliver to be worth shipping at all:

1. The dump payload becomes a single configured **format** with a closed set of
   three values (`documented`, `change-focused`, `self-contained`), default
   `documented` (`US-012`).
2. The collapsed formats deliver **one bulk flush = one write** (`FK-002`'s
   surviving clause), so a session of N turns flushed F times costs `F`
   write-bearing operations instead of `2N` — reachable, witnessed, and
   documented.
3. The documented per-turn layout is **unaffected**: with no format configured,
   `turn_000.json`, `turn_{N:03d}.diff` and `index.txt` are produced exactly as
   today, at today's `2N` writes (`G-008`(a)).
4. The flush-trigger policy (`FK-001`) is implemented for the collapsed path:
   volume threshold, interval, and shutdown flush.
5. The upstream-TLS opt-out cluster (`FK-006`/`FK-007`/`FK-008`, `T006`) ships as
   decided in 017.
6. The comparison evidence `G-008`(b) requires exists at `MS-002`, on named
   measures **M1** (write-bearing operations) and **M2** (files created).
   *(Corrected by `022`: the collapsed half of the pair needs a flush, and the
   flush triggers land with `T003` — `MS-001` cannot produce it.)*

Everything below is charted **above** that floor. Nothing below is required for the
session to be complete or correct.

---

### Switch the default dump format — DEFERRED (above MVP floor)

**Status:** `promoted-onward` → `.sessions/codify/roadmap/2026-10-18-message-dump-bulk-flush.md` `RM-013`.

Core delivered: the collapsed formats are implemented, selectable
(`--dump-format change-focused | self-contained`), documented, and witnessed by
`VP-T002-write-count-reduced` and `VP-T002-file-count-reduced`; the documented
format remains the default and is byte-identical to today, witnessed by
`VP-T002-documented-format-preserved`.

Deferred: performing the switch itself — flipping the default to a collapsed format
and/or deleting the documented writer's three write sites (`dump.py:82`, `:95`,
`:105`) once the collapsed format has been demonstrated better and the user has
judged it worth switching to.

Safe because: the switch is a one-value change in `config.py` plus a branch
deletion — no data migration, no contract break while both formats are shipped and
documented, and no reader depends on the new default until it lands. The decision to
make it is gated on evidence that this session produces rather than assumes: the
paired runs at `MS-002` on measures M1 and M2.

- **Owner**: a later session.
- **Trigger**: the user's judgment of the `MS-002` comparison evidence — a
  demonstration the new format is better, and a decision it is worth switching to
  (`US-012`, `decisions.md#us-012`). Not a date, not a metric threshold alone. The
  evidence is the `MS-002` Yield Phase's comparison record (a durable,
  human-readable artifact), not an in-test assertion.
- **Source**: `US-012` (2026-09-16) — *"Old behavior should stay unaffected until
  we prove a new one is better and it's worth switching to."*
- **What the switch would have to decide, not assume**: whether the demonstration
  rests on the write-count proxy (M1) or on physical flash wear (M3), which is
  **not measurable in this repository** — the byte figures carried through this plan
  come from `007` and were not re-derived (`/Volumes/data` is unmounted).

### Dump machine record — DEFERRED (above MVP floor)

**Status:** `promoted-onward` → `.sessions/codify/roadmap/2026-10-18-message-dump-bulk-flush.md` `RM-014`.

Core delivered: nothing of it. The dump payload carries human-readable records only;
the closed format set (`documented`, `change-focused`, `self-contained`) has no
machine-oriented value.

Deferred: a machine-oriented dump format — a compact record aimed at tooling rather
than at a person reading the file — as a fourth value of the format set.

Safe because: the format set is closed and validated (`--dump-format` rejects an
unknown value), so adding a fourth value is an additive change to one tuple, one
record builder, and the README's format table; nothing existing is displaced, and no
shipped value changes meaning. It was measured at ≈1.047× the logical bytes of the
change-focused record and was not taken because the reduction it might have supported
is on the physical axis, which this repository cannot measure.

- **Owner**: a later session.
- **Trigger**: a consumer that reads the dump programmatically, or a measurement of the
  physical axis.
- **Source**: `FK-007`'s `Impact.Roadmap` (element (a) — the pick did not take it).

### Human-facing dump tool — DEFERRED (above MVP floor)

**Status:** `promoted-onward` → `.sessions/codify/roadmap/2026-10-18-message-dump-bulk-flush.md` `RM-015`.

Core delivered: nothing of it. The dump is read by opening the files and reading the
README's *"Reading the index"* section; no tool is shipped.

Deferred: a human-facing viewer/tool that renders a session's dump — reconstructing
the per-turn view from a collapsed artifact, or browsing a session's turn history
without reading raw files.

Safe because: the collapsed formats keep every turn's content in the artifact
(`change-focused` carries the unified diff per turn; `self-contained` carries every
turn's full rendering), so the data a viewer needs is present and losslessly ordered
in the file. A viewer is an addition over a complete artifact, not a repair of an
incomplete one.

- **Owner**: a later session.
- **Trigger**: a user asking to read a collapsed dump without opening it by hand.
- **Source**: `FK-007`'s `Impact.Roadmap` (element (b) — the pick shipped the
  self-contained encoding instead), and `US-008` — *"add something more
  human-visible or human-friendly."*

### Chat route's post-commit second head — OUT OF SCOPE (066-01-yield P2, routed by `codify-scope` rules 1–2)

**Status:** `promoted-onward` → `.sessions/codify/roadmap/2026-10-18-message-dump-bulk-flush.md` `RM-016`.

Found: while verifying `T013`'s header-parity wave, [066-01-yield](iterations/066-01-yield.md) P2 traced the same
defect class that member G-1 closed in `/v1/messages` still live at its sibling boundary.
`buster_ripper/routes/chat.py:174-188` (`_relay_chat_stream`; was `:170-184` when this section was written — see
the 2026-10-06 amendment) calls `await response.prepare(request)` at `:184` (was `:180`) and
then relays inside `async with`, so a mid-relay `ClientPayloadError` escapes to `_make_chat_request`'s
`except aiohttp.ClientError` at `:155-157` (was `:151-153`), which returns a **fresh** `web.json_response({"error": …}, status=502)`.
`prepare()`'s re-entry guard is per-response-object while the writer is per-request, so that object is prepared onto
the already-committed stream: a second status line inside the first response's body, and no terminal the client can
trust. That is `DC-S11-CE1-r1` verbatim, at a route `057` §C never enumerated.

Why it is not a `T013` item: no `T013` clause names it — F-4's scope is the five filter sites, G-1's is the messages
relay per `057` §C and `DC-G-1-*`. It is byte-unchanged across the whole wave (`verified: git diff 41d3fc6..HEAD --
buster_ripper/routes/chat.py` touches only `_prepare_headers` and its helper), it gates no accepted criterion, and
folding it into the wave would have shipped an unattributed behavior change inside a wave whose members are
deliberately one commit each and separately attributable. It is not a `codify-scope` **rule 3** case either: this
session's retry needed no trustworthy result from it — the P1 was entirely inside `_prepare_headers`'s selection.

Reach and cost of leaving it: `latent` — a streaming `/v1/chat/completions` whose upstream body dies after the head
(an eval harness against a worker that goes away mid-generation). Consequence is a client parsing a status out of a
200 body, the exact harm `T013` was raised to remove elsewhere in the same package.

What the fix owes, so it is not re-derived later: `R-E`'s shape a second time (one post-commit arm that logs
position-not-cause and returns the committed object, with `CancelledError` left propagating), its own witness set at
the raw-bytes tier for the chat route, and a `seam-S11` reading — the seam currently records the messages relay's
semantics only. Do it on its own branch from main.

Handle: `roadmap.md` §Chat route's post-commit second head. Pointer kept (not resolved) at
`QB:routes-chat/post-commit-second-head` in `quality-backlog.md`.

**Amended 2026-10-06 by the CR2 closure-packet routing (ledger `DC-CR2-F-4-r1`):** the three line pointers above
were refreshed to the bytes at `dbd3072` (drift caused by `T013` retry-1's `_prepare_headers` rewrite; the relay
itself remains byte-unchanged and the defect still holds — re-verified). The post-071 `codex-review` batch brought
a **byte-level repro** — two `HTTP/1.1` status lines in one response, the synthetic 502 head inside the committed
200's body — which *confirms* the harm this section already states ("a second status line inside the first
response's body, and no terminal the client can trust"); the packet's premise that the record here understated it
as a "partial body without a terminator" matched nothing in the session and was corrected on the way in. The
reviewer's elevation note is recorded for the owner of this handle: **stronger confirmed severity is the reason
this slice should not sit indefinitely behind other roadmap rows.** The `codify-scope` rules 1–2 ownership ruling
above was explicitly rechecked against the repro and stands **unchanged** — no session gate needs this fix, and
the debts listed under "What the fix owes" are unchanged.

---

## Routings from the 2026-10-06 backlog drain (`/refactor-hunter`, run-codify step 6a)

Size triage: each row below is a backlog line judged **big** — spans modules, changes an
interface or behavior, or is a design decision in its own right — and is routed here rather
than started. `quality-backlog.md` carries the matching `→ routed:` marker on every line.

### RM-001: Route-handler LNCP restructure — DEFERRED (design decision under live bindings)

**Status:** `promoted-onward` → `.sessions/codify/roadmap/2026-10-18-message-dump-bulk-flush.md` `RM-001`.

Deferred: the restructures that `handle_messages` (`messages.py:26`, hard **C:11**, L:30 medium
after T013), `handle_chat` (`chat.py:23`, compound gray L:29/N:2/C:6) and `_make_chat_request`
(`chat.py:133`, three grays L:18/N:2/C:4 — its earlier "hard N4" premise withdrawn at 067-01 and
re-measured again at this drain) require. `handle_count_tokens` was drained separately at this
pass (checkpoint 154) and is **not** part of this chunk.

Not drained here because: `062 R-A` binds the hop-by-hop mechanism to in-place edits at the five
enumerated sites and `DC-S10-C4-r1` forbids touching the SSE header literal, so any handler
restructure is the item's own design decision (043-01-yield's framing, re-confirmed at 066-01);
and `chat.py` sits at 299 raw lines against the 300-line Green ceiling — extraction into that
file is arithmetically blocked until the split question (RM-002's sibling decision for `chat.py`)
is answered. This drain deliberately left the branch's route handlers untouched for the MS-004
boundary review.

- **Owner**: a quality pass after the MS-004 boundary review.
- **Trigger**: next scheduled touch of `buster_ripper/routes/` outside a gating wave.
- **Source**: `QB:routes-messages/handler-lncp`, `QB:routes-chat/handler-lncp`,
  `QB:routes-chat/request-lncp` in `quality-backlog.md`.
- **Extended by `T024` (2026-10-18)**: the enumeration now also carries `_non_stream_single_request`
  (`messages.py:258`, hard **L:33 N:4 C:8**), `handle_streaming` (`messages.py:85`, hard **N:3**,
  L:25 C:8 P:4) and `_stream_single_request` (`messages.py:131`, L:29 N:2 C:3), plus the
  file-size row for `messages.py` (334 raw lines) — `QB:routes-messages/deep-violations-unenumerated`,
  `QB:routes-messages/stream-single-request-gray-pair`, `QB:routes-messages/file-size-yellow`.

### RM-002: dump.py module split / misplacement check — DEFERRED (spans modules)

**Status:** `promoted-onward` → `.sessions/codify/roadmap/2026-10-18-message-dump-bulk-flush.md` `RM-002`.

Deferred: `dump.py` is at **316** lines — inside the 301–500 review band and *grown by this
drain* (+11: the `_render_body` extraction, checkpoint 153, cleared the 3-gray compound at the
cost of file length; the trade is recorded on the backlog line). The candidate split —
record builders/writers vs state+triggers — is a module-boundary decision, not a local fix.

- **Owner**: a quality pass; the split may also be mooted if the timer loop moves.
- **Trigger**: next touch of `dump.py` (the item's own wording: "Misplacement check due at next touch").
- **Source**: `QB:dump/file-size-yellow` (evidence recorded at 304 by 032-01-yield; 316 at this drain).

### RM-003: Embedder-safe flush format binding — DEFERRED (contingent design)

**Status:** `promoted-onward` → `.sessions/codify/roadmap/2026-10-18-message-dump-bulk-flush.md` `RM-003`.

Deferred: the residual fmt-mixing past the T003 fix — `state.fmt` is re-captured per turn, so a
mid-run format **change** followed by further turns files old-format buffered records under the
new fmt's artifact (last-wins). Unreachable via the CLI (`_set_dump_format` is the only
production writer, parse-time), and `PI:T002/dump_format/format_is_per_artifact` holds in
production. The remedy is a real design choice — per-record fmt capture, per-fmt buffers, or a
documented embedder contract — and should be taken when an embedder path is actually scoped,
not before.

- **Owner**: whoever scopes an embedder/import-time mutation path.
- **Trigger**: an embedder path appears, or a test harness mutates `DUMP_FORMAT` mid-run.
- **Source**: `QB:dump-flush/residual-fmt-mixing`.

### RM-004: Bound aiohttp and name the inspected internals — BIG (dependency bound)

**Status:** `promoted-onward` → `.sessions/codify/roadmap/2026-10-18-message-dump-bulk-flush.md` `RM-004`.

Deferred: `pyproject.toml:7` carries `aiohttp` unbounded while two witness families depend on
inspected 3.13.3 internals — the TLS-posture seam (`TCPConnector.__init__` ssl passthrough,
connector-loop adoption, no session-level `ssl` kwarg) and T013/F-4's parser-backend pins
(`HttpRequestParserC`/`Py`, `_http_parser.pyx` canonical-name substitution vs the pure-Python
parser's verbatim names, `CIMultiDict.getall`). A major upgrade can turn either family red
without any product change; the C-parser case skips **silently** when the extension is absent,
quietly narrowing the tier. The fix changes a dependency bound — out of drain scope by rule.
Suggested shape when taken: `aiohttp>=3.13,<4`, plus a note recording which internals the suite
pins (a `skip`-on-missing can be upgraded to `fail` on the pinned range).

- **Owner**: the next dependency touch (a `chore(deps)` increment on its own).
- **Trigger**: next touch of `pyproject.toml`'s dependency list, or any aiohttp upgrade.
- **Source**: `QB:deps/aiohttp-unpinned`.
- **Extended by `T024` (2026-10-18)**: the dependency list also carries a direct `multidict` import in
  production (`utils.py:7`) while `pyproject.toml` declares only `aiohttp` and `click`; the remedy is to
  declare it with a bound matching aiohttp's own, or record the transitive-import ruling —
  `QB:deps/multidict-direct-import`.

### RM-005: CLI-surface docs + `STATS_DB` dead surface — OUT OF SCOPE for draining (roadmap per `codify-scope`)

**Status:** `promoted-onward` → `.sessions/codify/roadmap/2026-10-18-message-dump-bulk-flush.md` `RM-005`.

Deferred: five CLI flags the binary accepts (`--strip-images`, `--preserved-thinking`,
`--backend-type`, `--eval-*`, `--host`) appear nowhere in `README.md` while `--skip-tls-verify`
and `--dump-format` are documented at length; and `config.STATS_DB` is assigned by nothing
(`main.py` sets `session._db_conn` directly). Amended by the S10–S13 routing pass to also own
`main.py:74-75`'s `--strip-images` help-text **scope** wording ("Strip images from user
messages…", unqualified about position and nesting — `seam-S12` composes two cases where the
flag does neither). Spans `main.py`, `config.py`, `image.py` semantics, and `README.md` — a
surface task, not a line edit. The nested-images half stays with `T014`'s follow-on
("when `tool_result`-nested images come into scope"); the wording half lands here.

- **Owner**: a dedicated CLI-surface documentation increment.
- **Trigger**: next touch of this surface (the item's own wording).
- **Source**: `QB:cli-surface/undocumented-flags-and-stats-db`.

### RM-006: Hand-rolled mutation matrices and `.pyc` staleness — harness/method (roadmap per `codify-scope` rule 2)

**Status:** `promoted-onward` → `.sessions/codify/roadmap/2026-10-18-message-dump-bulk-flush.md` `RM-006`.

Deferred: the measurement hazard at 067-01 — CPython's `.pyc` source tag is (whole-second mtime,
size), so two equal-size edits written in the same second validate against each other's cached
module, and a chained matrix reported M5=9 failed where the truth is 5. Reproduced deliberately
(`pyc_proof.py`, `evidence/t013-retry1-header-parity.txt`). This session's driver already does
all four remedies (purge `.pyc` per case, `PYTHONDONTWRITEBYTECODE=1`, two agreeing tallies,
restore in a `finally`); the owed work is in **shared tooling / method docs**, which belongs on
its own branch per `codify-scope` rule 2 — deliberately not fixed from this branch.

- **Owner**: the codify harness/method repo (dotfiles tooling), own branch off main.
- **Trigger**: next hand-rolled mutation or repair matrix outside this session's driver.
- **Source**: `QB:process/hand-rolled-matrix-pyc-staleness`.

### RM-007: TLS-posture unit pinning vs the QAP-MS004 authority — DEFERRED to the boundary verdict

**Status:** `promoted-onward` → `.sessions/codify/roadmap/2026-10-18-message-dump-bulk-flush.md` `RM-007`.

Deferred: `tests/test_tls_posture.py` pins private connector attrs (`connector._ssl` :85/:97,
`connector._loop` :105) because aiohttp 3.13.3 offers no public accessor; an aiohttp upgrade can
break the pinning with no behavior change. The item's own supersession condition has now
**fired**: `tests/qa/test_qap_ms004.py` exists and is green (6 passed at this drain, re-run
2026-10-06), i.e. QAP-MS004 landed with behavioral coverage of both postures against a real
loopback peer. Deciding what the unit pinning becomes (kept as a cheap tripwire, downgraded, or
removed) is a posture-authority call for the MS-004 boundary review that follows this drain
immediately — not a test-deletion made mid-session.

- **Owner**: the MS-004 boundary review, then a quality pass.
- **Trigger**: the boundary review's posture-authority verdict (RM-004's aiohttp bound interacts).
- **Source**: `QB:test-tls-posture/private-attr-pinning`.
- **Extended by `T024` (2026-10-18)**: the same private-attribute class also covers
  `tests/test_transport_parity.py:160` (`ClientSession._timeout` pinning) —
  `QB:test-transport-parity/private-attr-pinning`; the cure for both is the real-upstream tier,
  not a re-pin.

### RM-008: Response-relay content-coding posture — DEFERRED (1% category, `seam-S15` CE-1; routed 2026-10-06)

**Status:** `promoted-onward` → `.sessions/codify/roadmap/2026-10-18-message-dump-bulk-flush.md` `RM-008`.

Deferred: the shared filter `buster_ripper/utils.py:29-56` (`response_headers`; the locator moved from the
`:27-40` this row was written against, because `T017` added the posture paragraph at `:46-50` that points here)
relays the
upstream's `Content-Encoding` over body bytes the proxy's own client already decompressed (`ClientSession`
default `auto_decompress=True`; the package names the parameter nowhere —
`grep -rn "auto_decompress" buster_ripper/` → zero), so a compressing upstream
yields a labelled-encoding answer whose body is plain and whose `Content-Length` matches the plain bytes.
`codify-agents` → Reach Gate puts this at **latent**: the proxy solicits the encoding on both egress arms and
has no arm that declines decoding, and the harm is reproducible end-to-end through the real routes
(`curl: (61) Error while processing content unencoding: incorrect header check`), but the deployed upstream
(`http://hyperipper.local:30001`, vLLM/uvicorn — the endpoint the live `QAP-MS001`/`QAP-MS002` gates run against)
answers identity to `identity`, `gzip, deflate` and `gzip, deflate, br, zstd` on every path probed, including a
66 535-byte `text/plain` `/metrics`, and no test, gate, doc or evidence artifact on this tree has ever named a
`Content-Encoding` on this path. Full transcript: `evidence/s15-reach-and-base-comparison.txt` §1.

**Corrected 2026-10-17 (`decisions.md` §US-009):** the endpoint named in this row's basis is
`http://hyperipper.local:30000`, not `:30001` — `:30001` is a peer Buster Ripper instance, not the model
server. The encoding observation was made through that peer proxy, whose shared filter relays the upstream's
`Content-Encoding` unchanged (`buster_ripper/utils.py:29-56`), so it still describes the model server's behaviour
as relayed; whether the `latent` ruling's basis survives is this row owner's re-read, not re-decided here. The
`QAP-MS001`/`QAP-MS002` gates are owed a re-run against `:30000`.

Why this is a ruling and not a repair: the behavior **is** base parity. `buster_ripper_legacy.py:630-637` kept
the name (its drop-set was the eight hop-by-hop names only — it did not even drop `content-length`) and httpx
decoded `.content` the same way while advertising `Accept-Encoding` from its own defaults, so the monolith
carried the identical mismatch at the same three boundaries. This session's settled direction at those
boundaries is `T013`'s — *restore base header handling* — so any remedy is a deliberate **parity deviation**, and
the three shapes are not interchangeable: drop `content-encoding` at the filter (decoded bytes, no label) /
force `Accept-Encoding: identity` on egress (upstream bytes arrive verbatim, and `T015`'s accepted required
behavior 2 becomes actually true) / relay the wire bytes with `auto_decompress=False` (verbatim *and*
honest, but breaks the more common case — `messages.py:272/276` and `chat.py:163` parse the body for token
stats and the overflow check). `seam-S15`'s Note N3 records that nothing in the six member claims is normative
about the posture, and no session authority ranks them. Per `codify-scope` this is the **1% category**: recorded
once, no repair attempted here, and it **gates nothing** — no task, no Fork, no wave; `FK-011` remains the
only open Fork. The claim rows behind it (`claims.md` 384, 385, 386, 387, 389) are routed
`accepted-consequence` and their `HOLD` states stand as this row's library evidence.

**Do not confuse with what was fixed here:** the sibling finding on the same composition, **CE-2**
(repeated header names folded to the last occurrence), is **not** base parity — the base's client merged
occurrences, so it delivered every value — and it was repaired in-session as `T017`, which **landed and was
accepted-local on 2026-10-06**; the owed `seam-S15` recomposition (basis `sha256:d790d12b…`) shows the fold gone,
so CE-2's chain is closed and `S15` is terminal with `Routing: accepted-consequence`. A future reader of
`QB:utils/response-header-relay-fidelity` must not carry that line's original "both are base parity" framing
forward: it is correct for CE-1 and refuted for CE-2.

**And do not read this row as owning occurrence shape.** `T017`'s repair means a duplicated
*single-valued* name (`Date`, `Server`) now reaches the client on several field-lines where it used to reach it
on one — `seam-S15`'s Note **N5**, adjudicated by the recomposition pass and held **accepted** with reach
`reached`: no admissible remediation differs from the current relay, because collapsing or merging a duplicate
discards a value the base delivered in the case the deployment actually produces. The three postures below are
about **which names** travel and whether a label matches its bytes; they neither include nor exclude a
name-arity table, and adding one would be a new ruling, not a clause of this row. Full record:
`handoff-context.md` §Accepted risks (`S15-N5`) and
[076-assumption-decide](iterations/076-assumption-decide.md).

- **Owner**: the header-lane design pass (a product-posture ruling on relayed names, not a remediation wave).
- **Trigger**: the next decision about which response header *names* the proxy relays, **or** any upstream
  change that makes compression reachable — a gateway in front of `--upstream`, a backend whose server enables
  response compression, a client report of a decode failure through the proxy, or installing `brotli` /
  `brotli-cffi` / `backports.zstd`, which additionally fires `seam-S15`'s Note N1 and requires recomposition.
- **Source**: `assumptions/compositions/seam-S15.md` §Counterexample CE-1 + §Notes N1/N3;
  `evidence/s15-reach-and-base-comparison.txt` §1; routing record `iterations/074-assumption-decide.md`
  (§S15-CE-1) and `handoff-context.md` §Accepted risks / §Elevated to user; `QB:utils/response-header-relay-fidelity`.
- **Extended by `T024` (2026-10-18)**: the same helper's return container is a mutable `CIMultiDict`
  (`utils.py:52`), one mutating caller away from silently collapsing occurrences (`CIMultiDict.__setitem__`
  is `popall` + `add`); returning `CIMultiDictProxy(result)` is the decision this row's owner now also
  carries — `QB:utils/response-headers-mutable-return`.

### RM-009: `find_message_start` per-chunk scan misses a split `message_start` — DEFERRED (1% category, `codify-scope`; routed 2026-10-06 by the CR3 routing)

**Status:** `promoted-onward` → `.sessions/codify/roadmap/2026-10-18-message-dump-bulk-flush.md` `RM-009`.

Deferred: `buster_ripper/utils.py:91`'s `find_message_start` decodes and line-scans **each network chunk in
isolation**, so a `message_start` event whose bytes straddle a chunk boundary is never recognized — that turn's
TTFT + input tokens go unrecorded (`_note_message_start` at `routes/messages.py:200-215` re-scans every chunk
until it succeeds, so the next turn self-heals). Reported by `CR3-F-11` (P3) in the CR3 batch
([079](../iterations/079-codex-review.md) §CR3-F-11; disposition and reach in
[080](../iterations/080-assumption-decide.md): reach **latent**, 1%-category).

Why this is a deferral and not a wave member: the behavior is **base parity** — the pre-refactor monolith scans
the same way per chunk (`buster_ripper_legacy.py:674-676`, re-verified by the routing pass) — so the branch
carried no regression into review; the harm is one turn's telemetry on a rare timing, it blocks no session gate,
and the fix (carry the undelimited tail across chunks, or scan the accumulated stream head) lands on the
accepted relay surface, which the CR3 wave's must-not-touch sets deliberately keep cold.

What the fix owes, so it is not re-derived later: a chunk-carry (or accumulate-until-first-event) form of
`find_message_start`'s contract with a fail-first witness feeding `message_start` bytes split at every position
of one event (the split point the review needs is inside the `data:` line, not merely between events), while
keeping the current "found once, then stop scanning" behavior and the relay's byte-for-byte chunk passthrough
untouched.

- **Owner**: the streaming-parse quality pass (with `roadmap.md` §RM-001's handler restructures as a likely
  shared visit).
- **Trigger**: a field report of missing/zeroed per-session stats or a never-firing session-trigger nudge on a
  streaming client, **or** the next restructure touching `find_message_start`/`_note_message_start`.

### RM-010: the still-armed transport arms do not own the timeout class, and the generated-error envelope is unwritten — OUT OF SCOPE (routed 2026-10-06 by the run-42 composition pass)

**Status:** `promoted-onward` → `.sessions/codify/roadmap/2026-10-18-message-dump-bulk-flush.md` `RM-010`.

Deferred: three of the four upstream call sites still arm a `ClientTimeout.total` and catch only
`aiohttp.ClientError` — `_non_stream_single_request` (`routes/messages.py:265-297`, inheriting `server.py:67`'s
600 s), `_make_chat_request` (`routes/chat.py:139`/`:150`, 300 s) and `handle_passthrough`
(`routes/passthrough.py:27`/`:48`, 60 s). Per the session's own third-party evidence the class a firing **total**
raises is bare `asyncio.TimeoutError`, which is **not** a `ClientError` (`assumptions/claims.md` rows 330/331,
both `HOLD`; `verdicts/claim-331.md` for the pass-through at `client.py:793-798`), so at those three sites a
deadline expiry escapes the arm and is answered by aiohttp's protocol at `web_protocol.py:518` with a
**generated 504** + `text/plain` + a traceback log — never the site's declared 502 +
`{"error": "Upstream request failed"}` + `"Upstream error: %s"`. The paired missing contract, named by both
`S16`-(b) and `S17-N3`: *the status and body this proxy owes a client for a failure it does not author* —
aiohttp's generated `400`/`500`/`504` are English `text/plain` documents while every repo-authored client-facing
failure is JSON `{"error": …}` (`assumptions/compositions/S16-body-limit.md` §What Is Required (b);
`assumptions/compositions/S17-stream-deadline.md` §Notes N3, whose gap row **230** already fills).

Why this is a deferral and not a wave member: **base settled it worse.** `buster_ripper_legacy.py` has no
upstream-transport arm at all — no `httpx.HTTPError`, no `TransportError`, no `"Upstream error"` string anywhere in
the file — so an `httpx` timeout propagated out of `proxy_messages` (`:639`), the chat handler (`:811`) and the
passthrough (`:901`) and became a Starlette 500. The 502 is **branch-authored** (`T013`–`T017`, witnessed by
`tests/test_upstream_answer_relay.py`), so the parity authority that ranks the CR3 wave obliges nothing here; the
gap is between the branch's own arm and the class set aiohttp hands it. `DC-CR3-F-03-r1` said the same thing in
the negative when it rejected "drop/raise the shared session's 600 s" for silently moving the non-stream arm, and
carried the sibling observation "so Yield Red does not read silence as oversight" — this row is where that
observation finally gets an owner. The `/v1/messages` **streaming** path is *not* in this set: `T018` step 2 arms
no deadline there (base's posture, `buster_ripper_legacy.py:664`), which is what closes `S17-CE-1`/`CE-2`.

What the fix owes, so it is not re-derived later: a posture decision first, then the mechanism. (1) **Which status
is the proxy's answer to "the upstream exceeded its deadline"** — the site's existing 502 (class-list widening:
`except (aiohttp.ClientError, asyncio.TimeoutError)`, which is the same shape at all three arms) or the protocol's
504 (leave the arms, pin the observable, and give the generated body a name in the docs). Neither is base-parity
ranked; both are defensible, which is precisely why it needs a design pass rather than a wave. (2) Whichever way
it goes, the repair must state its scope as the **whole** arm family (four call sites including `compaction.py`'s
poll and `backend.py`'s probe, which already answer differently — `:55`'s `except Exception` covers the class) so
one label does not mean three things across the package, and each arm needs a fail-first witness whose pre-fix
failure names `asyncio.TimeoutError` as the class that escaped — a witness that only asserts "a 5xx came back"
proves nothing about which arm ran. (3) `S17-N4` rides the same decision: the armed budget is **per upstream
attempt**, so the SGLang overflow retry (`messages.py:99-114`) can spend ≈3× the number for one client-visible
request — say now whether that is intended, and if not, where the per-request ceiling lives. (4) Capacity:
`routes/chat.py` is at **300/300** raw lines, so any added import or arm breaches the FL-LNCP ceiling — §RM-001's
handler restructure is the natural shared visit, and this row must not be paid for by a new inline helper.

- **Owner**: the transport-error-envelope design pass, jointly with §RM-008 (response-direction content coding) and
  §RM-001 (the handler restructures) — the three meet at the same relay surface.
- **Trigger**: the first field report of a client unable to parse a proxy error body, **or** any new config channel
  for upstream timeouts (which turns `S17-N2`'s currently-inert `None`/`0`/omitted-`total` spellings into live
  inputs and re-composes `S17` per its `Invalidates On` 5), **or** the next restructure touching any of the four
  arms.
- **Source**: `assumptions/compositions/S17-stream-deadline.md` (CE-1's class facts, `N3`, `N4`) and
  `S16-body-limit.md` ((b)'s envelope contract), routed by
  [082-assumption-decide](../iterations/082-assumption-decide.md); `handoff-context.md` §Accepted risks
  (`S17-sibling-arms`) and §Elevated to user (`Reported: —`).

### RM-011: the server decompresses request bodies and nothing downstream says so — OUT OF SCOPE (1% category, `codify-scope`; routed 2026-10-06 by the run-42 composition pass)

**Status:** `promoted-onward` → `.sessions/codify/roadmap/2026-10-18-message-dump-bulk-flush.md` `RM-011`.

Deferred: aiohttp's request parser inflates a request body whose `Content-Encoding` names `gzip|deflate|br|zstd`
(`http_parser.py:556-558, 799-804`, with `auto_decompress` on for the server too —
`web_protocol.py:206, 224-232`), so every handler sees **decompressed** bytes. Three consequences, none of them a
finding on the CR3 surfaces and all of them recorded so they do not vanish: (i) any future `client_max_size` bounds
*decompressed* bytes rather than wire bytes — moot while `T018` step 1 sets `0`, live the moment a cap channel
appears (`S16` dimension 1 is recomposed on that landing either way); (ii) `DeflateBuffer`'s
`DEFAULT_MAX_DECOMPRESS_SIZE = 2**25` (`http_parser.py:972, 1040-1044`, `compression_utils.py:37`) raises
`DecompressSizeError` — `PayloadEncodingError ← BadHttpMessage ← HttpProcessingError`, **not** an
`HTTPException` — so it lands in `web_protocol.py:522`'s generic arm and the client is answered a generated **500**
where a size rejection wants a 413 (row 230 for the dispatch); and (iii) the relay never reconciles the label with
the bytes: `passthrough.py:14` reads inflated bytes and `:31` forwards them under the peer's own
`Content-Encoding` (`utils.py:20-26` keeps the name), and the `json=` paths at `messages.py:139`/`chat.py:141`
re-serialize a body the peer labelled encoded, so the upstream is handed plain bytes with a compressed promise.

Why this is a deferral and not a wave member: real but rare, with no obvious fix where it shows — it needs a
client that compresses a request body, which nothing in this package's documented surface asks for and the
deployed launcher never sends, and the 32 MiB half needs that body to exceed `2**25` *after* inflation. The cause
lives in the request direction's content-coding posture, which is a **design** posture, not a defect elimination;
and base could not produce this state at all, because base read bodies through Starlette and never decompressed a
request body (zero `gzip`/`decompress`/`Content-Encoding` references in `buster_ripper_legacy.py`) — so nothing the
CR3 wave is repairing regressed it. The agent did no repair and built no proof.

What the fix owes, so it is not re-derived later: the same three-way posture §RM-008 names for the response
direction, decided for requests instead — **drop `Content-Encoding` on egress when the body was inflated**, **ask
peers not to encode request bodies and reject `Content-Encoding` on ingress with a client-facing status the proxy
authors**, or **re-encode on egress** (the heaviest, and the only one that preserves the peer's framing for an
upstream that wants it). Whichever is chosen owes: a witness per route class that sends a gzip-encoded
`/v1/messages` body and asserts what the upstream actually receives (a stub that decodes and re-encodes is the
only way to observe this — the current `tests/qa/` stubs do not compress request bodies), one witness for the
`DecompressSizeError` shape asserting a status the proxy names rather than a generated 500, and an explicit note in
`utils.py`'s own docstring that the shared filter's kept-name list is now load-bearing in **both** directions.

- **Owner**: the content-coding posture pass — §RM-008's own owner, extended to the request direction; the two
  halves should be decided together so one filter does not carry two unreconciled postures.
- **Trigger**: the first client that compresses a request body against this proxy (a field report, or a documented
  client adding it), **or** any change that arms a request-body cap again, **or** §RM-008 being taken up.
- **Source**: `assumptions/compositions/S16-body-limit.md` §Notes (third bullet) and §What Is Required (a),
  re-read against `utils.py:20-26`, `routes/passthrough.py:14/31` and `buster_ripper_legacy.py` by
  [082-assumption-decide](../iterations/082-assumption-decide.md); `handoff-context.md` §Accepted risks
  (`S16-decompress`) and §Elevated to user (`Reported: —`).
- **Source**: `CR3-F-11` via `iterations/079-codex-review.md`; `handoff-context.md` §Accepted risks
  (`CR3-F-11`) and §Elevated to user (`Reported: —`).

### RM-012: `/v1/messages` answers a generated 500 to a schema-invalid request body — DEFERRED (base-parity posture, `codify-scope`; routed 2026-10-07 by the CR4 closure-packet routing)

**Status:** `promoted-onward` → `.sessions/codify/roadmap/2026-10-18-message-dump-bulk-flush.md` `RM-012`.

Deferred: any `tools` or `messages` array carrying a **non-mapping element** — `{"tools":[null]}`,
`{"messages":[null]}`, `{"messages":["x"]}` — raises `AttributeError` inside `normalize()`'s element loops
(`normalize.py:75` reads `t.get("name","")`, `:117` reads `msg.get("content")`, and `:192` reads `msg.get("role")`
under `STRIP_DATE` + sglang), the raise escapes `handle_messages`, and aiohttp authors the answer: a generated 500
with a server-side traceback, where a proxy that validates input would author a 400. Reach: `latent` — exposed route,
no guard prevents the input; malformed per the Anthropic schema, so the documented client never sends it.

Why this is a deferral and not a wave member: **base does exactly the same thing, at the same point in its own
`normalize()`.** Measured at all three points by the routing pass — `evidence/cr4-probe-matrix.txt` §3 runs base
(`72e3a1c6`) against a live upstream stub, so the 500s are attributable and the well-formed control's 200 proves the
harness can succeed, and the direct `normalize()` table in the same section shows the same raise with no transport in
the way; §1 shows the package matching base at the pre-range branch `6508941` **and** at `HEAD`. Nothing this session
introduced needs restoring on this route, which is the whole of the distinction from `T018`: what `T018` absorbed was
a rejection base never imposed, and what is declined here is a rejection base imposes today. Hardening the loops
therefore *changes* `/v1/messages` — measured **500 → 502** with the upstream unreachable (§7), i.e. the request now
survives normalization and reaches the relay — and `G-003` holds that route's behavior still, its constraint column
reading "No new endpoints, **no normalization changes**". The remaining question is a posture, not a defect: what
should this proxy say when the body is well-formed JSON and invalid schema? Three admissible answers, none ranked by
any settled authority available to this session, all of them out of its scope.

What the fix owes, so it is not re-derived later: **all three loops in one motion** — a two-loop repair is incomplete,
because `:192` is masked by `:117` today and inherits the crash the moment `:117` stops being it (§6 measures exactly
that promotion). **And the loops are not the granularity — there are five mapping-assuming read sites, not three**:
`_normalize_tools` reads `t.get("name", "")` **three** times (`:88` the `original_names` comprehension, `:89` the
`sorted()` key, `:90` the `sorted_names` comprehension), `_strip_cache_control_from_messages` once (`:130`), and
`_normalize_date` once (`:205`). Measured, twice independently. `090-01-forward`'s evidence §D records the mechanism
while executing its replay (c): guarding `:88`, `:89` and the `:130` loop still left `{"tools":[null]}` 500ing, "because
`_normalize_tools` reads `t.get('name','')` a second time in `sorted_names`" — but the same report's §Concerns 1
enumerated the tools reads as two and the total as four, which is the count this row carried. `090-01-yield` re-ran the
mutation per-site and **corrected it**: `{"tools":[null]}` stays green at **500** until `:90` is guarded too, and turns
500 → 502 only when all three tools reads are covered. So a repair stated as "three loops" is incomplete by two sites,
and one stated as "four read sites" is incomplete by one. A
`continue`-style guard must also cover `:131` (`"cache_control" in msg`, `TypeError` on `None`) and `:209`
(`msg["content"] = …`), both masked today by the read above them; a substitution-style guard does not reach them.
Coordinates here are post-`T019.002` bytes: the `+13` delegating entry shifted every line below it, so where the
paragraph above reads `normalize.py:75`/`:117`/`:192`, those sites are `:88`/`:130`/`:205` now. Then: the answer shape decided rather than drifted — `400` with the repo's own
`{"error": …}` envelope (`messages.py:33`, `chat.py:32`, `count_tokens.py:23-25` are the existing authored shapes),
**or** tolerate-and-forward and let the upstream reject it, **or** validate and drop the offending element and log the
change the way `normalize()` logs every other mutation. Whichever is chosen owes witnesses at `/v1/messages` **and**
`/v1/chat/completions` for each of the three element classes, one witness that the session-id digest is unaffected
(`session.py:93-98` reads `system` only), and a decision on whether `count_tokens` then returns to the full
`normalize()` call — because `T019.002` deliberately narrows that route to the system-strip step, and its Done-when
clause 5 pins `/v1/messages` at 500 with witnesses that go red the moment this posture changes without this row being
taken up.

- **Owner**: the route-input posture design pass — the same owner as §RM-001 (the route-handler restructure), since
  both decide what `/v1/messages`' handler does with a body it did not ask for; §RM-008/§RM-011's content-coding
  postures are the adjacent decisions on the same request/response boundary.
- **Trigger**: any pass that takes up §RM-001, **or** a field report of a 500 from this proxy on a malformed body,
  **or** the first feature that makes schema-invalid input reachable (a client other than Claude Code, or an
  input-validation surface on the route).
- **Source**: `CR4-F-01` (second half) via [088-codex-review](../iterations/088-codex-review.md), adjudicated at
  [089-assumption-decide](../iterations/089-assumption-decide.md); `handoff-context.md` §Accepted risks
  (`CR4-F-01` second half), §Elevated to user (`Reported: —`), and `DC-CR4-F-01-r1`'s rejected-candidate list.
  — **PR #1 review round 4 (2026-10-19) adds the root shape to this row's owed work, and leaves it here.** A
  syntactically valid non-object body (`null`, `[1]`, `5`) raises `AttributeError` at `normalize()`'s first
  `body.get("system")` (`normalize.py:52`) before any element loop — measured 500. It is the same posture question
  (400 with the repo's `{"error": …}` envelope vs tolerate-and-forward vs drop-and-log), it is not one of the three
  element classes this row already enumerates, and it is **not** repaired in the review round: `G-003` holds
  `/v1/messages` behavior still, and `handle_messages` carries a hard **C:11** violation (`L:28 N:2 C:11`), so an
  additive guard there triggers the capacity gate and requires §RM-001's extraction first. The chat and `count_tokens`
  root shapes are different: neither route is bound by `G-003`, neither has a recorded posture, and both are repaired in
  [126-assumption-decide](../iterations/126-assumption-decide.md) §Findings and dispositions.

### RM-017: Test-module shape and duplication — DEFERRED (suite reorganization)

**Status:** `promoted-onward` → `.sessions/codify/roadmap/2026-10-18-message-dump-bulk-flush.md` `RM-017`.

Deferred: the test modules whose backlog lines name a file-shape or duplication remedy —
`tests/test_header_parity.py` (444 raw lines), `tests/test_routes_integration.py` (495 raw lines, plus its
duplicate `_create_mock_response`/`client` pair), `tests/test_backend.py` (329), `tests/test_image.py` (364)
with `tests/test_image_stability.py` (337), `tests/test_upstream_answer_relay.py` (307),
`tests/test_compaction.py` (451) and `tests/test_context_overflow.py` (309). Every line's own remedy
names the split or the migration; none changes production behavior. The two image `reset_image_cache`
fixtures stay duplicated until a third image witness module appears, per `codify-test-cost` rule 2.

- **Owner**: the quality pass that owns each module's shape, or the next touch that is not a repair wave.
- **Trigger**: the next touch of a named module that is not a gating or repair wave.
- **Source**: `QB:test-header-parity/file-size`, `QB:test-routes/duplicate-upstream-stub`,
  `QB:test-backend/file-size`, `QB:test-image-stability/file-size`,
  `QB:test-image/sibling-module-duplication`, `QB:test-upstream-answer-relay/file-size`,
  `QB:test-compaction/file-size-yellow`, `QB:test-context-overflow/file-size-yellow`.

### RM-018: QA-gate surface — size, strength and coverage — DEFERRED (test-support shape)

**Status:** `promoted-onward` → `.sessions/codify/roadmap/2026-10-18-message-dump-bulk-flush.md` `RM-018`.

Deferred: `tests/qa/test_qap_ms001.py` crossed the 700-line hard stop at **1027** raw lines
(`assert_documented_contents` L:49 N:2 C:12 P:4); `tests/qa/harness.py` sits at **365** raw lines; and the
CE-2 artifact round-trip regression pins `read_text`'s raw mode but not the gate's two call sites
(`:276`, `:465`), so a call-site revert would stay green.

- **Owner**: `forward-qa`, on a QA author pass that is not a gate repair.
- **Trigger**: the next QA author pass touching these modules, or a CRLF-bearing `TC-108` body.
- **Source**: `QB:test-qap-ms001/file-size`, `QB:qa-harness/file-size-yellow`,
  `QB:qa-gate/ce2-call-site-unpinned`.

### RM-019: Assumption-registry and invariant record hygiene — DEFERRED (registry lane)

**Status:** `promoted-onward` → `.sessions/codify/roadmap/2026-10-18-message-dump-bulk-flush.md` `RM-019`.

Deferred: every record-shape and freshness line the Refactoring section routed to a registry lane —
the TLS opt-out `Entity:` elevation and the `qa-plan.md` lifecycle clause (Authority Registrar); the
digest-less and stale-pin verdict records and the annotated `PROMOTION:` headers (verdict lane); the ten
handleless `S16`/`S17`/`S18` members and the promoter input-rule arm; rows 391/404 wording, the four
third-party class atoms, row 57's version pin and the `S16` Dimensions wording (collector lane); and the five
README-digest records staled by `T023` (composition/promotion lane). Record-only; no code, test or gate waits.

- **Owner**: the registry lane (`assumption-promote`, with `assumption-check`, `forward-collect` and the
  Authority Registrar for their named arms).
- **Trigger**: the next composition dispatch or promotion touching any named record, or the post-`T023` re-trace.
- **Source**: `QB:assumption-registry/t006-tls-referent-elevation`, `QB:qa-plan/tc-tls-lifecycle-clause`,
  `QB:assumption-registry/annotated-promotion-headers`,
  `QB:assumption-registry/handleless-members-blocking-co-publication`,
  `QB:assumption-ledger/row-391-serving-app-wording`,
  `QB:assumption-ledger/row-404-replace-preservation-wording`,
  `QB:assumption-ledger/row-57-version-pin`, `QB:assumption-ledger/row-s16-dimension-route-count`,
  `QB:assumption-registry/readme-digest-after-t023`, `QB:assumption-registry/co-serve-script-basis-republish`.

### RM-020: Chat non-stream relay — verbatim bytes when no transform applies — DEFERRED (behavior addition)

**Status:** `promoted-onward` → `.sessions/codify/roadmap/2026-10-18-message-dump-bulk-flush.md` `RM-020`.

Deferred: when no transform applies (`fixed=False`) and the upstream body parsed as JSON, the chat relay
re-serializes (`json.dumps(resp_data)`) instead of sending the upstream bytes verbatim. Base re-serialized only
the arm it had actually rewritten (`buster_ripper_legacy.py:882-884`), so the unfixed re-serialization is
branch-era drift. The row's own condition — `chat.py` at 300/300 — blocks it without a prior extraction, and
`DC-CR2-F-1-r1` classes it "correct but additive, not competing".

- **Owner**: the chat non-stream relay design pass (with §RM-001's split decision).
- **Trigger**: the next touch of the chat non-stream relay, or a dedicated branch off main.
- **Source**: `QB:routes-chat/raw-bytes-passthrough-when-unfixed`, `DC-CR2-F-1-r1` deferred half.
  — **PR #1 review round 1 (2026-10-19) closed the response direction** (`DC-PR1-R1-3-r1`, lite brief
  `2026-10-18-pr1-chat-relay-fidelity`). **PR #1 review round 4 (2026-10-19) closed the request direction**
  (`DC-PR1-R4-6-r1`, `-E1-r1`): `chat.py:141-146` and `messages.py:155-158`/`:305-311` always passed
  `json=body`, so a UTF-16 request body reached the upstream as UTF-8 under a stale `charset=utf-16`; measured
  on `HEAD`. The request direction is **not** this row's content-coding posture (§RM-011) — it is representation
  fidelity when no transform changed the body, and base forwarded the client's bytes outside eval mode
  (`buster_ripper_legacy.py:812-847`). Fixed in lite briefs `2026-10-19-pr1-r4-chat-relay` and
  `2026-10-19-pr1-r4-messages-relay`.


### RM-021: Lint posture — the pinned F policy vs the verification entrypoint — DEFERRED (developer tooling)

**Status:** `promoted-onward` → `.sessions/codify/roadmap/2026-10-18-message-dump-bulk-flush.md` `RM-021`.

Deferred: `pyproject.toml:25-28` pins `[tool.ruff.lint] select = ["F"]` and its comment calls the
selection a "Minimal gate", but nothing in the declared dev environment runs it — `ruff` is absent from
`[dependency-groups] dev` (`:41-48`) and `uv.lock`, `.venv/bin/ruff` does not exist, and no repository
automation invokes it (no `.github/workflows`, no Makefile/justfile/`scripts/`; `addopts` at `:19` carries
only the coverage gate). The policy is real and executable by the repo's recorded ad-hoc invocation
(`uvx --offline ruff@0.16.10 check buster_ripper/ tests/ buster_ripper_legacy.py` → `All checks passed!`),
and the drain's own record says a lint-gate decision is not part of it (`119-01-yield.md:49`). The decision this
row owes: whether `ruff` joins the managed dev toolchain and the verification entrypoint enforces `select = ["F"]`
(a pytest lint plugin or a documented `uv run ruff check` step), or the pinned block stays an ad-hoc policy and its
comment stops calling itself a gate. No code moves in this session; the comment's wording is the doc line
`QB:lint/ruff-config-gate-wording`.

- **Owner**: the lint-policy pass (`run-codify` on its own branch off main).
- **Trigger**: the next lint-policy pass, or any change to `[tool.ruff.lint]`, `addopts`, or the dev group.
- **Source**: `runs/121-closing-check-codex.txt` (P2) + `QB:lint/ruff-config-gate-wording`;
  `119-01-yield.md:49`, `079-codex-review.md:225`, `080-assumption-decide.md:133`; HEAD `96264eb`.

### RM-022: Collapsed artifact numbering across a same-format restart — DEFERRED (1% category; PR #1 round 1)

**Status:** minted and routed onward in the same pass (2026-10-19, `0d465ec`) → `.sessions/codify/roadmap/2026-10-18-message-dump-bulk-flush.md` `RM-022`.

Deferred: a process restart with the same `DUMP_DIR`, session id and collapsed format leaves the
append-only `<format>.log` on disk while `_lookup_or_create_state` always builds
`SessionState(turn=0, prev_text="")` (`dump.py:147-158`). The next flush appends another `turn 000`
followed by diffs numbered from it, so one per-session log carries duplicate turn ids and a broken
cross-restart diff chain, while `VP-T002-collapsed-artifact` presents it as arrival-ordered session history.

Why this is a 1% record and not a repair: the trigger is an unusual-but-plausible restart, the result
persists in the artifact, and the fix is not cheap — the artifact is append-only and `(sid, format)`-keyed,
so continuing needs a production reader plus a change-focused diff-replay path that does not exist, while
rotating a distinctly identified run breaks `PI:T002/dump_format/format_is_per_artifact`, truncating breaks
append-only, and refusing the second run breaks `G-002`. Four materially different outcomes, none ranked by a
settled authority. Recorded once; no task, no Fork, no wave. PR #1 review comment
[4309720779](https://github.com/AlexGS74/buster-ripper/pull/1#discussion_r4309720779).

- **Owner**: the dump-artifact restart design pass (with `RM-002`'s split decision).
- **Trigger**: the next decision about what a collapsed artifact does across a process restart, or the first
  reader that consumes a collapsed artifact programmatically.
- **Source**: [124-assumption-decide](../02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/iterations/124-assumption-decide.md)
  §Why 6 is the 1% category; `VP-T002-collapsed-artifact`; `PI:T002/dump_format/format_is_per_artifact`;
  `OI:buster_ripper.dump/dump_turn/sequential_diff_chain`; HEAD `0d465ec`.

### RM-023: Collapsed/index append retried without rollback after a partial write — DEFERRED (1% category; PR #1 round 5)

**Status:** minted and routed onward in the same pass (2026-10-19, `713d3af`) → `.sessions/codify/roadmap/2026-10-18-message-dump-bulk-flush.md` `RM-023`.

Deferred: `dump._flush_state` (`:300-307`) writes the joined pending batch with one `f.write` and
clears the buffer only after it returns; `_flush_session` (`:288-294`) catches and retains the batch. A
mid-write `OSError` (ENOSPC/EIO/EDQUOT) leaves a truncated prefix, and the next periodic or shutdown
flush appends the whole batch after it — measured 7 bytes + `pending=1` → 483 bytes, head
`turn 00turn 000 …`. The documented `_append_index_line` (`:195-200`) has the same shape: a torn line,
then the same turn is retried (the exception skips `state.turn += 1`) and the new summary is joined to it —
measured `turn 000 turn 000 …` on one line.

Why this is a 1% record and not a repair: the trigger is an unusual-but-plausible mid-write failure, the
result persists in an append-only artifact (a torn prefix plus duplicated records and broken arrival order), and
the round is late — round 5 — so the late-round rule records an unusual-or-rare finding even when its fix is
cheap. No normal-use finding sends a repair into `dump.py` this round. The proper fix is one shared append
helper that records the pre-write size and truncates back to it on failure, used by both append sites (about eight
lines plus two failure-path tests). Recorded once; no task, no Fork, no wave. PR #1 review comment
[4318096152](https://github.com/AlexGS74/buster-ripper/pull/1#discussion_r4318096152).

- **Owner**: the dump-artifact write design pass (with `RM-002`'s split decision).
- **Trigger**: the next touch of `dump._flush_state` / `_append_index_line`, the next dump-artifact design
  decision, or the first field report of a torn or duplicated dump artifact.
- **Source**: [127-assumption-decide](../02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/iterations/127-assumption-decide.md)
  §Why #1 is the 1% category; `VP-T002-collapsed-artifact`; `PI:T002/dump_format/format_is_per_artifact`;
  `CL:buster_ripper.dump#_flush_state/conditional-batch-append`; HEAD `713d3af`.

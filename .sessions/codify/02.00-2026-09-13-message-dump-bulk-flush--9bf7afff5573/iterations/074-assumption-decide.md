# 074 · Assumption Decide — seam-S15 composition routing (MS-001 boundary)

## Orchestrator Prompt (verbatim)

> You are a subagent spawned by another session.
> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
> Worktree root: /Users/alexhubitski/Projects/buster-ripper
> Mode: composition-finding routing (seam S15, MS-001 boundary)
> Assumptions directory: /Users/alexhubitski/Projects/buster-ripper/.sessions/codify/02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/
> Registry root: /Users/alexhubitski/Projects/buster-ripper/invariants/
>
> Source: `assumptions/compositions/seam-S15.md` — `VERDICT: COUNTEREXAMPLE` over six promoted `pkg:aiohttp@3.13.3` members (rows 384–389, handles `CL:aiohttp/auto-decompress-session-default`, `…/auto-decompress-request-inheritance`, `…/auto-decompress-read-decompressed`, `…/auto-decompress-content-encoding-retention`, `…/repeated-header-items-iteration`, `…/default-accept-encoding-header`), now also published descriptively as `CO:client-response-to-relay-consumer/payload-and-header-pairing`.
>
> The composition traces two counterexamples at the relay boundaries: **CE-1** a body that aiohttp already decompressed is relayed downstream still carrying the upstream's `Content-Encoding` label, reachable at all three `utils.response_headers` call sites, with `chat.py:217/221` sharpest because that path re-serializes fresh JSON under the inherited label; **CE-2** repeated occurrences of one response header name are folded into a plain `dict`, so only the last occurrence reaches the client. The composer recorded two things it deliberately kept out of its verdict: the `br`/`zstd` arm, where no codec package is installed so the failure lands inside the existing `ClientError` handling (its Note N1, with an invalidation clause if a codec ever appears), and any preference about which posture a fix should take (Note N3).
>
> Two obligations before disposition. First, assign each a **Reach verdict** under the Reach Gate, and do it on evidence rather than assumption — CE-1 depends on whether an encoded answer actually reaches this proxy: what the deployed upstream returns, whether the client's own `Accept-Encoding` is forwarded, and whether any test or gate has ever observed a `Content-Encoding` on this path. Second, establish the base comparison honestly: the pre-refactor monolith used a different HTTP client whose body-handling default differs from aiohttp's, and `T013`/`T015` deliberately restored the base's *header* contract at these sites using `utils.response_headers`. So determine whether CE-1 and CE-2 are (a) branch-introduced divergences that the header-parity work cemented, (b) faithful restorations of base behavior, or (c) not comparable because base never relayed this shape — and let that finding, not the framing in this prompt, decide the disposition.
>
> Context to respect: the session's assumption gate is third-party-subjects-only, so `S1`–`S14` stay terminal and no repo row is re-extracted; `FK-011` remains the only open Fork; and `QB:utils/response-header-relay-fidelity` already carries the operator-facing half of this class from the CR2 wave's Yield. If a disposition creates implementation work, it is a wave with its own anchor task and local Yield before MS-001's Yield Red can run; if a finding is unreachable or an accepted risk, route it to the roadmap or backlog with its re-open trigger and say plainly that it gates nothing.

Result: **ASSUMPTIONS: 2 routed | 1 absorbed | 0 questions | 0 breaks | 0 malformed | 1 accepted | 0 capped**

The base comparison, established by running both clients on the same header head, **splits the two
counterexamples apart** — and it is the split, not the composition's framing, that decides each
disposition. `CE-1` is base parity and unreachable-by-deployment ⇒ accepted. `CE-2` is *not* base
parity and is being delivered live ⇒ absorbed into a repair wave.

---

## Finding rows

### S15-CE-1 — decoded body relayed under the upstream's `Content-Encoding` label — **accepted**

- **Reach: `latent`.** Evidence: `evidence/s15-reach-and-base-comparison.txt` §1. The proxy
  solicits an encoded answer on **both** egress arms — observed at the socket, not inferred: a
  client naming no `Accept-Encoding` gets aiohttp's own `gzip, deflate` forwarded (§1a arm 1), and
  a client naming one is forwarded unchanged because `accept-encoding` is absent from
  `utils.py:23`'s drop list (arm 2). The proxy never turns decoding off
  (`grep -rn "auto_decompress\|skip_auto_headers" buster_ripper/` → zero hits; the single session at
  `server.py:65-68` names only `timeout=`/`connector=`). What prevents the class is the
  **deployed upstream's posture**: `http://hyperipper.local:30001` — the endpoint
  `tests/qa/harness.py:39` declares and the `QAP-MS001`/`QAP-MS002` live gates ran against — answered
  with **no `Content-Encoding`** to `identity`, `gzip, deflate`, and `gzip, deflate, br, zstd` on
  `/v1/models`, on `/metrics` (a 66 535-byte `text/plain` body that is a prime compress candidate),
  and on a real `POST /v1/chat/completions`; `server: uvicorn` is the terminal value, so no
  compressing intermediary sits in front (§1c). No witness has ever seen one either:
  `grep -rni "content-encoding" buster_ripper/ tests/ docs/ README.md` → zero, and
  `tests/test_header_parity.py`'s census is parametrized over the eight hop-by-hop names plus
  `Content-Length` only (§1d).
- **Not `unreachable`:** the input is network- and configuration-supplied (`--upstream` is
  operator-set), and the Reach Gate says such input is never unreachable. **Not `reached`:** nothing
  on this deployment delivers it, and the guard is environmental — no test in this repo can pin it,
  so the `latent` row's alternative ("pin the guard with a test") is unavailable, not skipped.
- **Base comparison: (b) faithful restoration of base behavior.** The base's httpx decoded the body
  too (`Response.read()` → `b"".join(self.iter_bytes())` over "the decoded response content") and
  the base's `_response_headers` (`legacy.py:630-637`) dropped only the eight hop-by-hop names —
  keeping `Content-Encoding` — while httpx's own defaults advertised `Accept-Encoding` as well
  (`_client.py:120,310`). The header-parity work therefore *preserved* this mismatch, it did not
  introduce it. `T013`'s one deliberate deviation at these sites was the `content-length` drop
  (`utils.py:30-34`, row 236), which is documented and unrelated.
- **Cross-reference: Neutral.** Zero `active`/`wip` invariants exist in scope
  (`grep -rn "Status: active\|Status: wip" invariants/` → nothing; all eight registry `Invariant:`
  rows are `planned`); no session `invariants.md`. The published `CL:`/`CO:` records were used as
  evidence inputs only and classify nothing.
- **Why accepted rather than `question`.** The remedy is a **parity deviation**, and its shape is
  genuinely open: drop `content-encoding` at the shared filter, or force `identity` on egress, or
  relay the wire bytes with `auto_decompress=False`. They differ observably (which names a client
  sees; whether upstream bytes arrive intact; whether `messages.py:272/276` and `chat.py:163` still
  get parseable JSON), and the raw-bytes arm breaks the more common case — the proxy parses the body
  for its own logic — which `codify-scope` treats as settling the category: *"Two candidate fixes
  rejected for breaking a more common case settle this: stop trying."* Rare trigger, no obvious fix
  where it shows, cause split between a library default and the session's own filter ⇒ the **1%
  category**, recorded once, no repair built here (`codify-scope` §1% + its "does not block" row).
  Raising a Fork would be a *stroke* question at a milestone boundary that gates no task; the 1%
  lane reaches the user at the boundary summary with its three choices instead. **It gates nothing.**
- **Destinations.** `roadmap.md` §RM-008 (new — the relay-posture ruling, owner, re-open trigger);
  `handoff-context.md` §Accepted risks (one line, Reach verdict + citation) and §Elevated to user
  (one line, `Reported: —`); `QB:utils/response-header-relay-fidelity` amended with the corrected
  base comparison and the live-upstream observation. Composition rows 384, 385, 386, 387, 389 →
  Routing `accepted-consequence`; States stay `HOLD`.

### S15-CE-2 — repeated response-header occurrences folded, values destroyed — **absorbed**

- **Reach: `reached`.** Evidence: same file §2. The operator-declared upstream repeats header names
  on **every** response — a raw-socket read of `hyperipper.local:30001/v1/models` shows `date` twice
  and `server` twice in one head, and an earlier sample carried two *different* `date` values
  (`19:42:12 GMT` / `19:42:13 GMT` — 6/6 responses repeat both names, 1/6 carries two different `date`
  values); `utils.response_headers` run against that live response returns **3
  entries from 6 occurrences**, keeping the *last* `Date`; and the running production proxy answers
  the client with one `Date` and one `Server` line where the upstream sent two each. A test or
  fixture is not the basis for this verdict — the deployed endpoint is.
- **Base comparison: (a) branch-introduced divergence, which the header-parity work cemented — the
  existing "both are base parity" framing is wrong for this half.** The base's filter consumed
  `httpx.Headers.items()`, which *concatenates* occurrences of a name into one comma-joined value
  (`_models.py:216-229`, and `multi_items()` exists precisely as its non-merging alternative); this
  tree's filter consumes aiohttp's per-occurrence `.items()` (row 388) and folds last-wins. Run
  side by side on one head, verbatim from each source:
  | expression | relayed `date` | relayed `set-cookie` |
  |---|---|---|
  | base — `legacy.py:630-637` | `T1, T2` | `a=1, b=2` |
  | branch — `utils.py:36-40` | `T2` | `b=2` |
  The base never discarded an occurrence's value; the branch discards all but the last. The
  divergence arrived with the httpx→aiohttp swap and was **fixed in place** by `T013`/`T015`
  routing all three response boundaries through this one comprehension — the parity work chose the
  right helper and inherited its wrong fold. The asymmetry is worth recording: the *egress*
  direction is base parity (Starlette's `Request.headers.items()` yields per-occurrence pairs —
  `datastructures.py:534-535` — so base folded last-wins there too), which is exactly why N2 is a
  note and not a finding.
- **Cross-reference: Neutral** against the registry (same zero-active scan), **Aligned** with the
  session-settled direction `T013` ("restore base header handling at every proxy boundary") and with
  `tests/test_header_parity.py`'s own declared contract ("Pre-refactor (base) proxy semantics are
  the contract here") and `T015`'s accepted required behavior 2 ("filtered upstream headers"). The
  repair serves those; it does not need a new authority.
- **Closure: `absorbed`, one admissible remediation.** Required outcome, stated without a fix: the
  response relay discards no value the base relayed. Packet in §Deterministic Closure Ledger
  `DC-S15-CE-2-r1`. Rejected: comma-joining (base's *shape*, but it is not "verbatim", and for
  `Set-Cookie` — the composition's own concrete input — it is invalid under RFC 9110 §5.3, so it
  preserves part of the verified defect); documenting last-wins as intended (preserves the defect on
  a reached input); deferring to the design pass (that route's only stated reason was "both are base
  parity", which §2 refutes).
- **Task: `T017`**, wave anchor of a one-member repair wave with its own local Yield, **prework of
  the MS-001 close — it must land before `Yield Red`'s final semantic pass**, because Yield Red is
  the gate that certifies this boundary's header relay and would otherwise certify the false parity
  premise. Not a ride-along: no other repair is scheduled into `utils.py`, and `codify-scope` rule 5
  places this fault where it can be fixed — the shared filter, which is session code.
- Feasibility verified rather than assumed (§3): `web.Response(headers=CIMultiDict(pairs))` and
  `headers=list_of_pairs` both emit one line per occurrence (5/5), `CIMultiDict == dict` still holds
  so the census witnesses' dict comparisons survive, and case-insensitive `in` survives. The one
  witness the change can trip is `test_header_parity.py:261-263`'s `source.count("for k, v in") == 2`,
  named in `T017`'s done-when.

---

## Seam and ledger state

- `S15` Routing → `absorbed` (state `new` untouched); the seam is **not** closed: `T017` must land and
  `S15` be recomposed once — `resolved` then needs the post-implementation composition to return
  `SAFE`, or every remaining finding accepted. CE-1 is the accepted remainder; N1 (no `br`/`zstd`
  codec installed) and N2 (egress fold, base parity) stay notes and open nothing.
- Rows **388** → Routing `absorbed`; **384, 385, 386, 387, 389** → `accepted-consequence`. All six
  States remain `HOLD`; no evidence record was rewritten.
- Recomposition cap: **not** triggered — this is `S15`'s first composition
  (`seams.md` §Run-37 recorded no prior `compositions/seam-S15.md`), so `T017` earns the single
  post-repair recomposition rather than consuming a capped one.
- `S1`–`S14` untouched: no member list, State, Routing, Permanent or dimension cell of the fourteen
  terminal seams moved. No repo row re-extracted — the gate stays third-party-subjects-only, and
  `T017`'s fix is confined to recorded seams (`scoped:S15`).
- `FK-011` · preserved-thinking serve default remains the **only** open Fork. No `decisions.md` byte
  moved; no invariant record created, edited or promoted (triad rule — Interview owns recording).
- Repository state for the ledger: `git-commit:8438158a321b6fa701c67e48aec0c0f3e9cd3c06`, resolved
  through Git from the supplied worktree root before the first routing write.

Open:
- `T017` — owned by Forward; dispatch prompt is session id + task id, the block in
  §Planned Task Context is the transmission channel.
- `RM-008` response-relay content-coding posture — owned by the header-lane design pass; gates
  nothing, and `T017` must not silently satisfy it.
- The `## Elevated to user` line for CE-1 — owned by the orchestrator at the next boundary report
  and at session close (`Reported: —` until stamped).

Next: `T017` (Forward → one local Yield) → recompose `seam-S15` → MS-001 Yield Red.

## Writes, checkpoints, and disclosures

- **Durable records written** (all committed): `assumptions/claims.md` (six Routing cells; States untouched),
  `assumptions/seams.md` (S15 Routing + §Run-38 changelog), `handoff-context.md` (§T017 Planned Task Context,
  §DC-S15-CE-2-r1, §Composition routing index, §Accepted risks, §Elevated to user, §Current Status's owed list),
  `plan-summary.md` (T017 row + context pointer), `roadmap.md` §RM-008, `quality-backlog.md` (dated
  correction on `QB:utils/response-header-relay-fidelity`), `evidence/s15-reach-and-base-comparison.txt` +
  `evidence/INDEX.md`, and this report.
- **Checkpoints**: 170 (routing records) and 171 (refinements + the routing narrative moved verbatim to
  [handoff-archive.md](handoff-archive.md) §Trimmed 2026-10-06). `codify git create-checkpoint` only; no raw git.
- **Two disclosures.** (1) Checkpoint 170 also committed the composer's
  `assumptions/compositions/seam-S15.md`, which was untracked at dispatch; this pass wrote no byte of it, and
  committing it was the only way to have the routed evidence exist in the same history as its dispositions.
  The six `verdicts/claim-38{4-9}.md` files and the two modified `invariants/` records were **left
  uncommitted** — they belong to the checkers' and promoter's lanes, not this one.
  (2) `handoff-context.md` remains over its 200-line cap: 477 → **653**, and every added line is a
  protocol-required routing record (task block, ledger row, accepted-risk line, elevated line). The
  narrative was moved out rather than shortened, so nothing was lost; the deep trim stays
  `phase-boundary-finalize`'s, as disclosed at Checkpoint 168.
- **Suite state**: no production or test byte was touched — `git status --porcelain -- buster_ripper tests`
  is empty — so `T013`–`T016`'s accepted results stand as measured.

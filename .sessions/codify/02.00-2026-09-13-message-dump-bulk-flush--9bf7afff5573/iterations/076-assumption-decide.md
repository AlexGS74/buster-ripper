# 076 · Assumption Decide — seam-S15 recomposition routing (MS-001 boundary)

## Orchestrator Prompt (verbatim)

> You are a subagent spawned by another session.
> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
> Worktree root: /Users/alexhubitski/Projects/buster-ripper
> Mode: composition-verdict disposition — narrow confirmation, one seam
> Assumptions directory: /Users/alexhubitski/Projects/buster-ripper/.sessions/codify/02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/assumptions/
> Registry root: /Users/alexhubitski/Projects/buster-ripper/invariants/
>
> `seam-S15` was recomposed on post-T017 bytes after its `Invalidates On` 3 fired; the verdict is `COUNTEREXAMPLE` again and has been promoted in place as `CO:client-response-to-relay-consumer/payload-and-header-pairing` (new basis `sha256:d790d12b…`, prior CE-2 recorded closed). Your own routing at 074 dispositioned CE-1 as `accepted` with re-open triggers, and stated that the boundary may close with it unremediated and that it holds no seam.
>
> Answer one question with evidence: **does this recomposition introduce anything that 074's routing has not already dispositioned?** The composer carried CE-1 as still composes, closed CE-2, and carried N1/N2/N4 plus two new notes — **N3** (the consumer's docstring names the posture deliberate and defers to `roadmap.md §RM-008`, which the composer did not read) and **N5** (a newly-open contract that duplicated single-valued header names now reach the wire verbatim, stated without claiming which behavior is required).
>
> Rule N5 explicitly rather than letting it ride as a note: name its Reach on evidence and give it a disposition, or record why it needs none. If anything here is genuinely new to the routing, allocate it per the closure gate — but do not re-litigate CE-1, whose posture 074 already ranked, and do not consume a second recomposition on S15 that this pass already used. Then set S15's ledger Routing to whatever terminal state is honest for a seam whose one live counterexample is accepted-and-owned, so MS-001's Yield Red is not gated on it — and say plainly whether the seam is now citable, deferred, or closed, because Yield Red will ask.

Result: **ASSUMPTIONS: 2 routed | 0 absorbed | 0 questions | 0 breaks | 0 malformed | 2 accepted | 0 capped**

Answer to the question asked: **yes — three things, of which one is a real new disposition, one is
lifecycle only, and one the brief did not name.** CE-1 is untouched by this pass, the seam's allowance is now
spent, and `S15` is terminal: **citable as current, closed as a routing obligation, with its one live
counterexample deferred to §RM-008.** Nothing on it gates Yield Red.

---

## Finding rows

### S15-N5 — duplicated single-valued header names now reach the wire verbatim — **accepted** (new to routing)

- **Reach: `reached`, on the served path.** Not inferred from a test: `evidence/t017-live-relay-transcript.txt`
  §2 drives the real binary (`python -m buster_ripper.main --upstream http://hyperipper.local:30001`) and counts
  the lines a client actually receives. Base bytes: `date ×1, server ×1` downstream from an upstream head that
  carried each twice. **Fixed bytes: `date ×2, server ×2`** — and §2a of
  `evidence/s15-reach-and-base-comparison.txt` measured that the operator-declared upstream repeats both names on
  **every** response, with 1 in 6 carrying two *different* `Date` values. So N5's shape is on the wire today, at
  the highest possible frequency, and is not latent.
- **`reached` is where Reach First stops, not the disposition** — so the closure gate ran on it. Required outcome,
  stated independently of any fix and taken verbatim from the settled ledger: *the response relay discards no
  value the base relayed* (`DC-S15-CE-2-r1`, whose authority is `T013`'s "restore base header handling at every
  proxy boundary" plus `tests/test_header_parity.py`'s declared contract). Candidates:
  - **comma-join duplicates** — rejected *on the record already made*: `DC-S15-CE-2-r1` names it and rejects it
    ("transforms the representation instead of relaying it… invalid for names whose definition forbids joining…
    knowingly preserves part of the verified defect"). Re-inflating it as an open contract would re-ask a question
    this session already closed with a cited reason.
  - **a single-valued-name table (collapse or last-wins for `Date`, `Server`, `Content-Type`…)** — rejected: on
    the 1-in-6 differing-`Date` response it discards a value the upstream sent and the base delivered, i.e. it
    re-preserves the exact defect `T017` was created to remove, on a `reached` input.
  - **deduplicate only byte-identical pairs** — rejected: no value is lost, but nothing witnessed needs it (the
    composer traced no consumer that misroutes and this pass re-checked the three sites), it is new mechanism in
    the shared filter, and it re-fires `Invalidates On` 3 — which now has no recomposition left to service it.
  - **keep the current relay** — satisfies the required outcome; the duplicate is the *upstream's* malformation,
    relayed rather than invented, and `web.Response` neither adds nor collapses anything
    (`web_response.py:525-527` `setdefault`, re-read by the wave).
  **No admissible remediation differs from current behavior** ⇒ nothing to absorb, nothing to ask. `CLOSURE`
  answer: `absorbed` requires a remediation; here there is none, so the disposition is `accepted`. A `question`
  would be wrong for the same reason the candidates above are wrong: the choice is not unranked, it is closed by
  settled authority — "do not label it `question` merely because alternatives can be named".
- **Cross-reference: Neutral**, re-run not inherited: `grep -rn "Status: active\|Status: wip" invariants/` → zero
  (the registry's eight `Invariant:` rows are all `planned`, in `dump/` and `server/`, neither of which names a
  header relay); no session `invariants.md`. The `CL:`/`CO:` records were evidence inputs only. **No new normative
  RFC claim is made in this pass's records** — deliberate, on this ledger's own `claim-208` history, where an RFC
  9110 §7.6.1 citation turned out not to establish what it was cited for and the fetched bytes were not
  re-verifiable. The only spec point leaned on is the one already in `DC-S15-CE-2-r1` (joining is invalid for
  `Set-Cookie`).
- **Allocation: none.** One line under `handoff-context.md` → §Accepted risks with its re-open trigger
  (`Invalidates On` 3, or a field report of a client that cannot parse a repeated line). No task, no Fork, no
  wave, **no new roadmap row** — §RM-008 keeps its three content-coding shapes, and this pass wrote an explicit
  "do not read this row as owning occurrence shape" guard into it rather than letting N5 silently widen it.
- **Relationship to the wave's own note (recorded because the verdicts differ).** `075-01-yield` §D-7 reached
  `latent` → "defer-tier, no row" for the same class, scoped to a duplicated *kept* `Content-Type` and 3+
  occurrences. This pass reaches **`reached`** for the class as a whole, because the wave's own live transcript
  (§2, FIXED run) shows `Date`/`Server` duplicated on every response from the declared upstream — D-7's premise
  that "no declared upstream sends either today" holds for `Content-Type` and not for those two names. The
  disposition is unchanged (`no row`), and the reason is stronger: on a `reached` input whose only alternative
  remediation discards a value, accepting current behavior is a closure result, not a frequency judgment.

### S15-N6 — the widened input union's `dict` member is unreachable from this seam — **accepted** (new to the file, **not disclosed in the brief**)

The brief named two new notes (N3, N5). `compositions/seam-S15.md` §Notes carries **three**: N6 is also new. Ruled
rather than passed over:
- **Reach: `unreachable` from the seam's boundaries**, established here and not taken on faith —
  `grep -rn "response_headers" buster_ripper/` → the definition plus exactly three call sites
  (`passthrough.py:41`, `messages.py:291`, `chat.py:172`), each passing `resp.headers`, i.e. the
  `CIMultiDictProxy` member. A test that passes a plain `dict` is not a caller.
- Both arms behave identically (the predicate is `k.lower()` on name only), so no arm of the union takes a
  materially different path — N6 states no claim that fails. Per Reach First, `unreachable` ⇒ **accepted**, and
  the annotation's *actual* imprecision (it does not name the `CIMultiDict` the new witnesses pass) already has
  its row, `QB:utils/response-headers-input-annotation`, from the wave's Yield. **No second backlog row written;**
  the accepted consequence itself gets its line under `handoff-context.md` → §Accepted risks.

### S15-CE-1 — carried, unchanged — **no re-litigation**

Re-traced by the composer on the moved consumer bytes and still composes; its chain (384→385→386→387→consumer→
`web.Response`), its `latent` reach and its three remedy shapes are the ones 074 ranked. This pass read the
carried verdict, confirmed nothing in the delta moves it (the consumer change is occurrence-preserving, orthogonal
to the encoding label — the composer makes the same point and this pass checked `utils.py:52-56`: the exclusion
tuple names only `content-length` + `_HOP_BY_HOP`), and re-confirmed rows 384/385/386/387/389 stand at
`accepted-consequence`. 074's ruling is the disposition; §RM-008 stays its owner.

### S15-CE-2 — closed by the composition it was routed into — **lifecycle, not a new finding**

The recomposition shows the fold gone end-to-end (`CIMultiDict` construction → `StreamResponse.__init__` clone →
per-entry serializer) and the producer guarantee (388) and the consumption now agree. That is precisely the
terminal condition `DC-S15-CE-2-r1`'s `Status:` line named as owed ("S15 still owes the one recomposition"), so:
**no `DC-S15-CE-2-r2`** (the ledger is append-only and this is not a correction — run-29's
`DC-seam-S6-value-word-split-r1` precedent, "a lifecycle completion of its own tail conditions"), no rewrite of
that row, and **the `resolved` stamp is left to Yield Red**, exactly as that `Status:` line assigned it. Row 388's
Routing stays `absorbed`: the column records disposition, never lifecycle.

### N3 — the docstring's posture statement — **read, confirmed, owes nothing**

The composer could not read `roadmap.md`; this pass can and did. `buster_ripper/utils.py:46-50` says the filter
"does not reconcile" the label with the body it travels beside, names it base parity, and points at §RM-008 — the
routing 074 recorded, restated by the code that landed after it. Two checks were needed and both pass: the
deferral is to a row that exists and says what the docstring claims (§RM-008: 1% category, three shapes, owner
the header-lane design pass, gates nothing), and the pointer is *accurate* — so there is no gap in inert wording,
hence no doc line owed and no `quality-backlog.md` entry. Its only real cost was the stale locator 074's own row
carried (`utils.py:27-40` → `:29-56`), corrected here as bookkeeping on this lane's prior record.

### N1 / N2 / N4 — carried, unchanged in substance

- **N1** gained detail (the `ContentEncodingError` → `HttpProcessingError` → `ClientResponseError` channel, caught
  by all three `except aiohttp.ClientError` arms) and its own invalidation clause (now `Invalidates On` 4). Same
  note, same trigger (§RM-008's codec-installation arm names it), opens nothing.
- **N2** is S10's terminal boundary. The post-`T017` asymmetry — response direction preserves occurrences, egress
  direction still folds — is not new to *routing*: the wave's own Yield ruled it deliberate and recorded
  (`075-01-yield` §R6, `QB:utils/response-headers-mutable-return`), and 074 recorded the egress fold as evidenced
  base parity. Re-affirmed as a note; re-opens only through §RM-008.
- **N4** is the seam row's own boundary wording, whose fourth boundary (`chat.py:187-190`) already sits on a
  backlog row; recorded here in one clause so Yield Red does not read "every response boundary" as four.

---

## Seam and ledger state

- **`S15` State `new` → `registry-counterexample`, Routing `absorbed` → `accepted-consequence`.** Why those two
  cells and no others:
  - `State` had to move off `new`, or the gate's step (iv) reads `new`/`recheck` and dispatches a composer — i.e.
    the second recomposition this dispatch forbids. `registry-counterexample` is §3c's cell for a *fresh* matching
    `## Composition Index` record with a negative verdict, and freshness is established, not assumed: the index
    row carries `sha256:d790d12b…`, and this pass re-hashed `buster_ripper/utils.py` to
    `9abaa6b8ccbc2bdf8aadab44e64f2a07236ecad619cfc4b98fe78e897c8fdf24` — the digest the basis names — with
    `git status --porcelain -- buster_ripper tests` empty. `out-of-scope` would be a false statement (the gate
    *did* service this seam and its verdict is known); `registry-safe` would be a worse one (the verdict is not
    `SAFE`).
  - `Routing` moves because the seam's only live finding is now accepted-and-owned. Per protocol an `accepted`
    finding is written `accepted-consequence`; `absorbed` would claim open remediation work that no longer exists.
  - ⚠ **The `registry-counterexample → dispatch assumption-decide` branch is answered by the Routing cell.** That
    cell names the verdict on file, not work owed; re-dispatching this seam without a fired `Invalidates On`
    trigger would be routing an already-routed record.
- **Plainly, for Yield Red — citable, deferred, or closed? All three, and they are not in conflict:**
  - **citable as current**: yes, on `CO:client-response-to-relay-consumer/payload-and-header-pairing` @
    `sha256:d790d12b…`, fresh against the working tree.
  - **closed** as a routing obligation: yes — every finding left on the seam is accepted, which is the protocol's
    own terminal condition when a post-implementation composition is not `SAFE`.
  - **deferred**: its *content* — CE-1's posture, owner §RM-008, with N5 accepted alongside it and the codec arm
    behind `Invalidates On` 4.
  So: **cite it, do not re-run it, and do not gate on it.** It holds no seam and no task.
- **Recomposition allowance: spent, with a consequence recorded where it will be read.** 074 wrote that `T017`
  *earned* the single post-repair recomposition rather than consuming a capped one; this pass is that
  recomposition. Therefore any §RM-008 remedy (each of which fires `Invalidates On` 2(d)/2(f)/3 by design) lands
  on a seam with **no allowance left**: the owning route takes the user's ruling first and treats any further
  non-`SAFE` composition as capped-by-rule material for one user-facing message — no task, Fork, wave or
  re-extraction from it. Recorded in `assumptions/seams.md` §Run-39 and as a dated amendment on
  `QB:utils/response-header-relay-fidelity`, which is the row a future remedy-seeker actually reads.
- **No `## Capped seams` entry written.** That list exists to put a cap's withheld remainder in front of the user
  in one message. Nothing is withheld behind a cap today: CE-1 is already elevated with `Reported: —`, N5 is
  accepted with no open choice attached, and a second entry would duplicate an elevation the orchestrator still
  owes.
- **Writes to the ledgers: one cell pair, no new ledger row, no state writes.** `seams.md`: S15's `State` +
  `Routing` cells + §Run-39 changelog. `claims.md`: **zero cells touched** (all six rows keep their 074 Routing
  and their `HOLD` States — recomposition confirms, it does not re-verdict). `handoff-context.md`: new
  §Composition recomposition routing block, two §Accepted risks lines (`S15-N5`, `S15-N6`), three
  §Current Status corrections.
  `plan-summary.md`: the Current-Phase headline, the owed list, the relay-rule recommendation (N5's one clause),
  the `T017` row's assumption clause. `roadmap.md`: §RM-008 locator fix, the CE-2 tense, and the scope guard that
  keeps N5 out of it. `quality-backlog.md`: one dated amendment, no new row.
- **`FK-011` · preserved-thinking serve default remains the only open Fork.** No Fork reserved (nothing here
  reached `question`), no `decisions.md` byte moved, no invariant created/edited/promoted, no registry write —
  the published `CL:`/`CO:` records are evidence inputs only and classified nothing here (triad rule: Interview
  owns recording and promotion).
- **`S1`–`S14` untouched and terminal.** No member list, State, Routing, Permanent or dimension cell of the
  fourteen moved; no repo row re-extracted (the gate stays third-party-subjects-only); no other seam composed,
  re-verified or judged; `S15`'s verdict was neither re-derived nor widened.
- **Repository state for this pass**: `git-commit:ebf0381d8f6ca778b7f3443152de906d69434629`, resolved through Git
  from the supplied worktree root before the first routing write (no `.git/**` file was read). No `absorbed`
  finding, so no DC row consumed it; recorded because the seam's terminal claim needs a byte basis.

## Writes, checkpoints, and disclosures

- **Suite state:** no production or test byte touched — `git status --porcelain -- buster_ripper tests` empty
  before and after. Nothing was executed by this pass beyond digests and `grep`s; the reach verdicts cite the
  wave's executed transcripts.
- **Checkpoint:** `codify git create-checkpoint` for the routing records and this report. **Disclosure, same as
  074's:** it also commits `assumptions/compositions/seam-S15.md` (the recomposition being routed, untracked at
  dispatch) so the routed evidence and its dispositions share a history; this pass wrote no byte of it. The six
  `assumptions/verdicts/claim-38{4..9}.md` files and the two modified `invariants/` records stay **uncommitted**
  — the checkers' and `assumption-promote`'s lanes own them, per `075-01-yield` §Goalpost check.
- **Disclosure — a tooling slip, repaired in place:** the first pass of the `plan-summary.md` edits ran a `perl
  -pe` loop that auto-printed every line and duplicated the file (398 → 809 lines). Detected immediately by the
  line count. Repaired with `git checkout --` on that one session artifact (a restore, not a commit; no raw git
  write anywhere else), after which all four `plan-summary.md` edits were re-applied through the edit tool, with
  table-row integrity re-verified (row count unchanged, the `T017` row intact on one line, file at 406 lines).
- **Disclosure — the brief under-counted the new notes** (it said two; the record has three). N6 is adjudicated
  above rather than left to ride on the brief's enumeration.
- **Context budget:** `handoff-context.md` grew 652 → 701 with protocol-required routing records; 074's superseded
  §Composition routing index was moved **verbatim** to `handoff-archive.md`
  (§Trimmed 2026-10-06 (`076-assumption-decide`, seam-S15 recomposition routing)) to offset it, with a pointer left
  in place and `DC-S15-CE-2-r1`'s "named in §Composition routing" reference resolved by that pointer. The file
  remains over its cap; the deep trim stays `phase-boundary-finalize` / `session-squash`'s.

Next: **Yield Red** for MS-001. It may cite `S15` as current, is not gated by it, and owns `resolved` for
`T015`/`T016`/`T017` plus §RM-008's elevation stamp.

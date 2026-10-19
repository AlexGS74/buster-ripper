# 075-01 Forward — T017: relay every occurrence of a repeated upstream response header name (seam-S15 CE-2)

Session: `02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573` · Worktree: `/Users/alexhubitski/Projects/buster-ripper`
Task: **`T017`** — wave anchor and **sole member** of the seam-S15 CE-2 repair wave, one local Yield, own counter starting at 0 (no prior attempt of this wave reached Yield ⇒ `{NNN}-01`, not a retry)
Base: `e68ae02` (Checkpoint 172 — the tree `074-assumption-decide` left its routing on). Production + witnesses land in **Checkpoint 173** (`6a13bd7`); this record and the session-artifact updates in **Checkpoint 174** (`47ed5dd`), with two self-amendments to this file at **175** (`db5ef3c`) and **176** (`131600c`) — the roadmap note and the attribution below, no other byte and no code.
**Roadmap: no new row owed.** The only completeness this increment defers was already routed before it started — CE-1's content-coding posture is `roadmap.md` §RM-008's existing row, which `T017` deliberately did not touch, and the egress-direction fold (`forward_headers`) is recorded on that row's deferral line and on `seam-S15` N2. `roadmap.md` carries zero bytes of diff. What this pass did newly surface — production's first direct `multidict` import — is a dependency-manifest question for the next touch of `pyproject.toml`, so it went to `quality-backlog.md` (`QB:deps/multidict-direct-import`, keyed to §RM-004's owner) rather than to a roadmap row: an undeclared-but-guaranteed import is not deferred scope, and `Safe because:` would not have been truthful on a row that describes it as if it were. *(Amended after Checkpoint 174 by Checkpoint 175, which adds this paragraph and the two commit hashes above and moves no other byte.)*
Blueprint: `handoff-context.md` §Planned Task Context → `T017` · Ledger: `DC-S15-CE-2-r1` · Assumption: `scoped:S15`, finding `S15-CE-2`, disposition `absorbed`
Evidence: [t017-occurrence-relay.txt](../evidence/t017-occurrence-relay.txt) (fail-first transcript, run counts, dependency-claim table, clause-to-byte map) · [t017-live-relay-transcript.txt](../evidence/t017-live-relay-transcript.txt) (the running proxy's raw response bytes at the base commit and at Checkpoint 173, plus the live upstream replay)

## Orchestrator Prompt (verbatim)

> You are a subagent spawned by another session.
> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
> Task: T017

That is the whole dispatch. `T017`'s blueprint block states the rule directly — *"Forward receives session id + task id only; this block is the transmission channel"* — so the required behavior, done-when, Yield replay, out-of-scope list and verification profile were read from `handoff-context.md:309-376` and the ledger at `:505-529`, not from the prompt. No user-provided context arrived with the dispatch, so no `user-statement` record was reserved and none was owed.

## Increment Brief

T017 · make one upstream header name sent twice reach the client twice, on two field-lines, at
all three response boundaries — the fold is this branch's own, not the base's.

Anchors touched: **none of the Anchor Map's rows.** Its rows bind the dump and TLS clusters
(`plan-summary.md` §Anchor Map); `utils.response_headers` is named by no row, and this increment
**extends the existing referent in place** rather than binding a new one — the same reading
`073-01-forward` recorded for the CR2 wave, one task later and on the same helper. Reuse evidence:
no `Mode: NEW` anchor was consumed or created, so no `RB-*` replay was owed.
**Will NOT create:** a second header filter, a per-site accumulator at the three call sites (rejected
in `DC-S15-CE-2-r1`: breaks `062 R-A`'s one-shared-implementation binding), a relay/headers class or
registry, a new `config`/`main` surface, a new test module, or an invariant record.
`Inventions: none` in production — the diff is one return container, one widened annotation, one
import and 16 docstring lines in `utils.py`, with zero bytes moved in `routes/`. In the witness
module: two module-level constants (`_REPEAT`, `_REPEAT_PAIR`) and one builder (`_repeated_head()`),
test-local data shapes beside the module's existing `_wire_headers()` precedent — not load-bearing
referents, and `plan-summary.md` §Anchor Map's closing sentence keeps private helpers out of the map
by definition.

## What the defect is, and what was changed

`aiohttp` hands `ClientResponse.headers` as a per-occurrence mapping: one `date` **and** another
`date` are two pairs at `.items()` (claim 388, `CL:aiohttp/repeated-header-items-iteration`, `HOLDS`).
`response_headers` folded those pairs into a plain `dict`, which keeps the **last** value under one
`istr` key and destroys the rest, and `web.Response` then had one entry to write. On the deployed
upstream this is not hypothetical: `GET /v1/models` answers with `date` twice and `server` twice on
**6/6** probed responses, and 1 in 6 carries two *different* `date` values
(`evidence/s15-reach-and-base-comparison.txt` §2a) — `Reach: reached`. It is also **not** base parity:
base's httpx client *merged* occurrences into one comma-joined value, so every value travelled
(`§2c`). The loss arrived with the httpx→aiohttp swap; `T013` and `T015` restored the base's *drop-set*
at all three boundaries while leaving this half of the contract unfixed, and no census witness ever
fed a repeated name.

The closure is one expression's container:

```python
return CIMultiDict(
    (k, v)
    for k, v in upstream_headers.items()
    if k.lower() not in ("content-length", *_HOP_BY_HOP)
)
```

The drop decision, its operands and its spelling are byte-identical to the base bytes; the filter now
builds a per-occurrence collection instead of a folded one, and `web.Response(headers=…)` already wrote
one field-line per entry it was handed (`web_response.py:462-535`; the proxy adds no third `Date`
because `:526-527` `setdefault`s it). The three call sites hand its result straight to `web.Response`
exactly as before — `passthrough.py:41,46`, `messages.py:291,295`, `chat.py:172` carry **zero bytes of
diff**. `forward_headers` carries zero bytes of diff (`seam-S15` N2: the egress fold *is* base parity,
and S10's boundary is terminal). `_HOP_BY_HOP` carries zero bytes of diff.

## Failure-Path Evidence

Cases enumerated over the one operation the blueprint names — filter one upstream response head and
hand it to `web.Response` — before judging any result. Rows marked *excluded* name the authority that
excludes them; an omitted row is not an exclusion.

| Operation / condition | Expected outcome and authority | Handling / observation evidence | Disposition |
|---|---|---|---|
| `response_headers` / one name sent **twice with different values** | both occurrences reach the client, each on its own field-line (`DC-S15-CE-2-r1` required behavior; base's information contract per `evidence/s15…txt` §2c) | unit `test_response_headers_keeps_every_occurrence_of_one_name` + `[…][chat]/[messages]/[passthrough]` + `test_the_two_values_arrive_as_two_field_lines` (counts `X-Repeat:` in the raw socket bytes). **5 red at `e68ae02`** — the raw transcript shows `X-Repeat: second` alone, `first` nowhere in the head | safe |
| same / one name sent twice with **identical** values | both still travel: the rule is per-occurrence, not per-distinct-value (comma-joining and de-duplicating were both rejected candidates) | the same witnesses, whose fixture pair carries two distinct values; plus the raw-line count `== 2` which a de-duplicating build fails | safe |
| same / a **dropped** name sent twice (`Connection` ×2, `Content-Length` ×2, each of the eight) | every occurrence dropped, not one — preserving occurrences must not become preserving them before the drop runs (R5 kept-set, both directions) | `test_every_occurrence_of_a_dropped_name_is_dropped` × 9 params, red under a build that filters surviving *names* instead of *pairs* | safe |
| same / one dropped name under **two spellings** | the pair must still merge into "dropped". `istr` keys are not case-equal to each other (§Dependency claims), so this does not come for free | `test_two_spellings_of_one_dropped_name_are_both_dropped` | safe |
| same / one **kept** name under two spellings | both lines travel **with the spelling the peer used** — the filter lowercases for the drop, never for the relay (`T013`'s accepted rule, `test_header_parity.py`'s module docstring) | `test_two_spellings_of_one_kept_name_stay_two_lines_with_their_own_spelling`, which is exactly the guard against the tempting wrong fix of keying by `k.lower()` | safe |
| same / **empty** head, **empty value**, **non-ASCII value**, **many** occurrences of one name | nothing special happens: the filter is one comprehension | probed directly (`evidence/t017-occurrence-relay.txt` §4): `CIMultiProxy(())` → `[]`; `('X-Empty','')` survives; `('X-Note','café')` survives; five `Vary` occurrences → five pairs | safe |
| the three relay sites / `web.Response` handed a per-occurrence collection | one line per entry, and the **lying `content-length`** still dropped-and-derived (the `T013` rule this change must not cost) | `test_lying_upstream_content_length_never_reaches_the_client` × 3 site params, green unedited inside the 81; the raw witness's own head carries `Content-Length: 13` for a 13-byte body | safe |
| the three relay sites / a name arriving **after** the filter, e.g. `setdefault(DATE/SERVER)` | the proxy adds no duplicate of a name the upstream already sent | `web_response.py:526-527` read (`setdefault`, not `[…]=`); the raw witness asserts an exact line count of 2 for `X-Repeat` and no `Date` inflation is observable in its head | safe |
| **Hostile / adversarial input at the parse boundary (R1)** | upstream-supplied header names and values are external input; the filter's behavior must not depend on the *shape* of a value | the six-shape probe above plus the parametrized census over all nine dropped names. No new branch was added on the value side, so no value can take a different route — the comprehension's only predicate is `k.lower() not in (…)` | safe |
| **Durable writer into a pre-existing tree (R2)** | n/a | Excluded: this increment writes no file outside the session's own artifacts; no writer is in the diff | not-applicable (with evidence) |
| **Capture→act window (R3)** | n/a | Excluded: no read→write window is introduced. The filter remains synchronous, single-pass, and inside the existing `async with session…` scope; no new `await`, no environment read paired with an await result | not-applicable (with evidence) |
| **Payload decoder census (R4)** | n/a | Excluded on both counts: no payload **field** changes (the body bytes are relayed as `T015` left them), and no second consumer of `response_headers` exists to fall out of step — the three relay sites are its only callers (`grep -rn "response_headers" buster_ripper/` → the definition + three call sites) | not-applicable (with evidence) |
| **Identity-key injectivity (R5)** | in scope, and it is the defect | enumerated above: the "key" is the header *name*, and the two directions (kept-name occurrences must not merge; dropped-name spellings must merge) each got their own witness, including the near-miss pair the rule asks for | safe |
| **Boundary-symmetric constraint propagation (R6)** | the same fold exists on the **egress** side (`forward_headers` returns a plain dict, `utils.py:20-26`), so the constraint is now enforced on one side of the boundary only | **Explicit deferral, not silence.** `seam-S15` N2 records that the egress fold *is* base parity (Starlette yielded per-occurrence pairs into the base's dict comprehension too); `DC-S15-CE-2-r1`'s "Deferred additive work" holds it, its owner is `roadmap.md` §RM-008, and `T017`'s blueprint forbids touching `forward_headers` ("S10's boundary, terminal, and base parity there"). The asymmetry is therefore recorded where both a reader and the owning pass will find it | not-applicable (deferred, recorded) |
| **Cross-runtime comparator parity (R7)**, **regenerated-artifact delta (R8)** | n/a | Excluded: no order is defined in one runtime and consumed in another, and nothing generated was regenerated — `uv.lock`, `pyproject.toml` and every coverage artifact carry zero diff (`git diff --stat` → two files) | not-applicable (with evidence) |
| **Structural facts over text matching (R9)** | the increment adds no matcher over parsed input | The one text-matching witness in the diff is the module's *pre-existing* census, `test_utils_holds_the_only_two_filters`, which this pass satisfied rather than rewrote (`grep -c "for k, v in"` → 2). It is the plan's own pinned residual risk (`DC-S15-CE-2-r1` §Risk shape), not a new matcher | safe (no new site) |
| **`Content-Encoding` over already-decoded bytes (CE-1)** | **documented, not decided** — the blueprint is explicit that no ruling is taken here and no behavioral witness is written, so §RM-008 stays free to go either of its three ways | `utils.py:46-50` — the docstring names the posture the filter currently keeps, states it as base parity, and points at `roadmap.md` §RM-008. Zero witnesses in the class touch `Content-Encoding`; `roadmap.md` has zero diff | not-applicable (excluded by blueprint, documented per done-when) |
| **Retry / duplicate delivery / cancellation / recovery** on the touched path | none introduced | Excluded with evidence: the filter is a pure one-pass function of one head; no task, lock, handle, retry arm or state mapping is in the diff, and `handle_chat`'s retry arm and both streaming relays carry zero bytes | not-applicable (with evidence) |

## Capacity gate: not applicable — no additive production surface

Derived touch set and classification, per `codify-agents` → Preventive Change-Surface Capacity Gate:

| Surface | Classification | Baseline at `e68ae02` | Final |
|---|---|---|---|
| `buster_ripper/utils.py::response_headers` | **replacing** — one return expression's container + its annotation; the drop tuple and its operands are byte-identical | scanner: unflagged (whole file "2 total issues in 2 functions", `find_message_start` L:14 N:2 C:6 and `ContextOverflowError.__init__` P:3) | `codify scan path buster_ripper/utils.py --git --detailed` → **"No refactoring issues found!"**; whole file still **2 in 2**, same functions, same numbers |
| `buster_ripper/utils.py` (file) | additive **text** only: 1 import line + 16 docstring lines | **133** raw lines (Green ≤300) | **149** raw lines — Green, 151 lines of headroom |
| `buster_ripper/routes/{passthrough,messages,chat}.py` | **not touched** | — | zero bytes of diff |
| `tests/test_header_parity.py` | additive (test surface, not production) | **360** lines, review band | **444** lines, review band, 1 scanner report — `test_both_occurrences_reach_the_client(self, proxy, mocker, site)` P:4, a pytest fixture arity every other site-driven witness in the module already has |

No trigger condition applies: the only production function in the touch set is unflagged before and
after, its executable length is one `return` on both trees, and the file sits at half the Green ceiling.
No extraction is owed and none was performed. Test files are additive by design here — the blueprint's
own "at least one per boundary plus a `utils.response_headers` unit witness" — and the growth is on a
row that already tracks it (§Backlog pickup).

## Backlog pickup:

Read at the start of the task and cross-checked against the actual touch set
(`buster_ripper/utils.py`, `tests/test_header_parity.py`). **Two open lines name a touched file; one is
absorbed as far as this task's authority reaches, one is a recorded skip with its reason.**

- `QB:utils/response-header-relay-fidelity` (`buster_ripper/utils.py:27`) — **half absorbed, row stays
  open, annotated.** The row carries two findings and its own `074` correction already withdrew the
  duplicate-name half from its remedy list; that half is now **landed** by `T017`. The content-coding
  half is explicitly out of this task's scope (`roadmap.md` §RM-008 owns the posture ruling) and the
  row is *not* marked `[x]`: its remaining remedy is unresolved. What this pass did inside the row's
  reach is state the posture in the filter's docstring, which is the "documented, not decided" clause
  of the done-when, and add the 075-01 annotation recording which half moved and which re-opens
  `seam-S15` recomposition.
- `QB:test-header-parity/file-size` (`tests/test_header_parity.py:1`) — **skipped, with its reason, and
  re-measured 360 → 444.** Its remedy fires "when next touching this module" and this is that touch, so
  the skip is a decision on the record, not an omission. Two grounds, both already on the session's
  record rather than invented here: (1) the remedy moves `TestSharedHelpers` **and** `TestSiteCensus`
  out of this module, and `T017`'s done-when names `test_utils_holds_the_only_two_filters` — a
  `TestSiteCensus` witness — as one the boundary replay must find here, green and unweakened; (2)
  `codify-test-cost` rule 3 says a fix from review *extends* the test that already owns the behaviour
  rather than adding a module per round, which is the rule that put these 16 witnesses here in the
  first place. 444 is still inside the 301-500 review band, so no must-extract boundary moved. The row
  now carries the recommendation for the quality pass (which unit-tier witnesses to move, and that the
  right moment is a touch that is not a repair wave).

No other unchecked line names a touched file. The new `QB:deps/multidict-direct-import` line records
the one fact this increment introduces that no row held (§Concerns 2).

## Dependency claims — checked against installed source, none corrected

Per the plan-adherence rule these are facts, not judgment, and each was read at the exact claim rather
than surveyed. Full table in `evidence/t017-occurrence-relay.txt` §4.

| Claim encoded by the new line | Read at | Holds |
|---|---|---|
| `.items()` yields one pair per occurrence | claim 388 `CL:aiohttp/repeated-header-items-iteration` (`HOLDS`, `sha256:546439f0…`, `multidict/_multidict_py.py`) + live 6-pair probe | yes |
| `web.Response(headers=<CIMultiDict>)` writes one field-line per entry | `web_response.py:462-535`, `:788-804` + live response-header count + the raw-socket witness | yes |
| `web.Response` does not add a third `Date` beside a relayed pair | `web_response.py:526-527` — `headers.setdefault(DATE/SERVER, …)`, not assignment | yes |
| a `CIMultiDict` compares equal to a `dict` **when occurrence counts match** (so the accepted dict-shaped witnesses survive) | live: equal-count → `True`; a folded pair → `False`, correctly | yes, with the count condition |
| `istr('date') != istr('Date')` and their hashes differ, so a plain dict does **not** merge two spellings | live probe | yes — this is what makes the two spelling witnesses guards rather than tautologies, and it is recorded because the answer is the opposite of the intuition |
| `multidict` is present whenever `aiohttp` is | `aiohttp-3.13.3.dist-info/METADATA` → `Requires-Dist: multidict<7.0,>=4.5`; `uv.lock:435` pins 6.7.1 | yes, but **undeclared** as a direct dependency — §Concerns 2 |

**No plan claim was found false, so no STOP is owed and nothing was corrected silently.**
`DC-S15-CE-2-r1`'s feasibility section (routing evidence §3) predicted this closure's shape — including
the exact residual risk to the census witness — and every row above confirms it. `claims.md` was read:
rows **384-389** are this wave's, all `shared` against `T013`/`T015` and all `S15`-keyed. Row **388** is
the one this task serves and it is `HOLD`/`absorbed` — the claim holds, the *consumer* was wrong, which
is precisely what "absorbed ⇒ `T017`" routes. Rows 384/385/386/387/389 are CE-1's and are
`accepted-consequence` under §RM-008; this pass changed none of their premises (no `auto_decompress`
argument, no egress `Accept-Encoding` change, no name added to the drop set), so no shared claim needed
raising beyond the wave. **Assumption impact: `scoped:S15` — implemented for Yield; `S15` stays
`absorbed`, not closed, and owes its one recomposition after this lands.**

## Plan adherence — what the blueprint forbade, and whether it moved

| Prohibition | Status |
|---|---|
| no `content-encoding` drop | honored — `Content-Encoding` is still relayed; no name added anywhere in the diff (`_HOP_BY_HOP` at `utils.py:14-17` byte-identical) |
| no egress `Accept-Encoding` change, no `auto_decompress` argument | honored — `server.py`, `main.py`, `config.py` and all three route files carry zero bytes of diff |
| no change to `forward_headers` | honored — `utils.py:20-26` byte-identical; `test_no_header_comprehension_is_left_in_a_route` and the egress census green |
| no change to `_HOP_BY_HOP` | honored; `test_hop_by_hop_set_is_bases_eight_names` green |
| must not be folded into `RM-008` | honored — `roadmap.md` has zero diff; the docstring only *points* at §RM-008 |
| no proxy-authored header literal routed through the filter (`DC-S10-C4-r1`) | honored — the five header literals are untouched; the Yield replay's diff check is §Next-phase item 4 |
| invariant write ban (triad rule) | honored — no record under `invariants/` created, edited, superseded or deleted by this pass. **Disclosed, not done here:** `invariants/buster_ripper/assumption-claims.md`, `invariants/index.md` and six `assumptions/verdicts/claim-38*.md` were **already dirty** when this pass started, and dirty **deliberately** — `074-assumption-decide` says so at [074](iterations/074-assumption-decide.md):175-176: *"left uncommitted — they belong to the checkers' and promoter's lanes, not this one."* This pass excluded them from all three of its checkpoints via `--files` rather than fold another route's pending artifacts into a repair commit. **`assumption-promote` owns those eight paths**, so `git status` is expected to be non-empty when Yield opens; nothing in them is this wave's finding |
| `FK-011` untouched | honored — zero `decisions.md` bytes moved |

## Runs

| Tier | Command | Result |
|---|---|---|
| fail-first | production stashed to `e68ae02`, `TestRepeatedHeaderOccurrences` alone | **5 failed / 11 passed** — the unit witness, the three boundary params, the raw-line witness; the raw failure prints the relayed head with `X-Repeat: second` and no `first` |
| module | `uv run pytest tests/test_header_parity.py -q` | **81 passed** (65 accepted + 16 new, none edited) |
| affected scope (checkpoint run, once) | the 10 modules that touch `utils`/routes/server | **248 passed** |
| full hermetic — the plan's named gate | `uv run pytest` | **455 passed @ 95.75%**; `buster_ripper/utils.py` **98%** (the single miss, `:85`, is `split_thinking`'s early return — pre-existing, outside this task) |

Two disclosures about the runs, because both look like anomalies and neither is:

1. **The affected-scope run's coverage line fails at 68.24%.** `addopts` computes coverage over
   `source = ["buster_ripper"]` while only 10 of the suite's modules ran, so `--cov-fail-under=80`
   measures the *selection*, not the code. The full hermetic run on the same bytes passes at 95.75%.
   No test failed at any point.
2. **The blueprint's gate clause names the full hermetic suite, which the role skill normally reserves
   for Yield.** Plan adherence is stated as non-negotiable and the done-when says "full hermetic suite
   green", so it was run — **once**, at the checkpoint boundary, in addition to the affected scope
   rather than instead of it. Yield should still run it independently on this commit, as the role
   contract says it will.
3. `tests/qa/` was not run and not edited. It is excluded from the default collection by
   `norecursedirs`, its gates are the milestone lane's, and no cell of it changed. The one file in that
   tree whose behavior this change could touch is the `assert_upstream_response` limb already tracked
   as `QB:qa-harness/static-body-cannot-connect-limb`, owned by `forward-qa`.

## Concerns (for the single local Yield that follows; nothing ruled here)

1. **`utils.py` is now the third consecutive wave to extend the same helper, and no Anchor Map row
   binds it.** `T013` bound the five filter sites, `T015`/`T016` joined the chat boundary to it, and
   `T017` changed its return type. The map's closing sentence ("Private helpers are not rows") covers
   decompositions, not a shared cross-route contract that three waves have now edited through it. The
   session's own `073-01-yield` §Recommendation already asked for the relay rule to be **elevated at the
   close**, and `T017`'s blueprint supplies the candidate wording for Interview to consider —
   *"the response relay discards no upstream header occurrence beyond the eight hop-by-hop names and
   `content-length`"*, deliberately worded so it does **not** pre-decide CE-1's posture. Recorded here
   so the boundary owner sees both halves; writing an invariant is not this route's authority.
2. **A direct import of an undeclared package** — `from multidict import …` in production for the first
   time. Present in every install (aiohttp hard-requires it, `uv.lock` pins 6.7.1), already the repo's
   convention in `tests/`, and named by the plan's own closure text, so it broke nothing and adds no
   package. But a manifest decision is a manifest decision, and it is not a repair wave's to take.
   Raised as `QB:deps/multidict-direct-import`, keyed to `roadmap.md` §RM-004's owner with the 0.01%
   both-parts test written out on the row. The rejected alternative (return the pairs) would have had
   to edit `test_header_parity.py`'s `_values()`-based witnesses, which the done-when requires to stay
   green unweakened — that constraint, not preference, is what chose `CIMultiDict`.
3. **`test_header_parity.py` at 444.** In the review band, +84 by this wave, with the split's own
   remedy now colliding with this task's pinned witness location. Recorded and re-measured on the row;
   flagged rather than resolved.
4. **Yield's replay, from the blueprint, and how to run it.** (a) one proxied request per boundary
   against a peer repeating a name with differing values, asserting the count and both values on the
   wire — that is `[…][chat]/[messages]/[passthrough]` plus `test_the_two_values_arrive_as_two_field_lines`,
   and the honest note is that the client-tier witnesses can distinguish two lines from one comma-joined
   line only because the aiohttp client's `CIMultiDictProxy` is itself per-occurrence, which the raw
   witness removes all doubt about; (b) re-run the drop census — the 9 occurrence params plus
   `TestSharedHelpers`' existing table; (c) confirm from the diff that `forward_headers` is untouched
   and no proxy-authored header literal entered the filter — `git diff e68ae02 -- buster_ripper/` shows
   exactly one file, `utils.py`.
5. **`S15` is not closed by this increment.** Per `seams.md` Run-38, the seam's Routing is `absorbed`
   and its closure needs `T017` landed **and one recomposition** returning `SAFE` or every remaining
   finding accepted. CE-1 stays `accepted` under §RM-008, so the recomposition will still carry it as a
   known, routed counterexample rather than a clean `SAFE` — that is the boundary owner's call to make,
   and it should be made with the `Invalidates On` 3 trigger already fired (this pass changed
   `response_headers`' return type and its comprehension shape, which that clause names explicitly).
6. **No user question was asked, and none was gated.** Every fork in this increment was closed by
   sourced authority — the blueprint's container choice, the plan's pinned census witness, the
   `DC-S15-CE-2-r1` rejected-candidate list, and `RM-008`'s existing ownership of CE-1. No
   pre-question closure packet was owed, so none is attached: there was no `question` disposition, no
   `BLOCKED-APPROVAL`, no `ESCALATION` and no `PLAN-GAP`. Nothing here needed a snapshot baseline,
   a push, or any other escalation-floor action. **`CLOSURE: not-owed — zero unranked admissible
   outcomes remained.**

## Visual Demo Evidence

`T017` changes what a client receives on the wire, so the increment's evidence is the **running proxy**
rather than a rendered view: [`evidence/t017-live-relay-transcript.txt`](../evidence/t017-live-relay-transcript.txt)
starts `python -m buster_ripper.main` the way `serve_buster_ripper.sh` does and prints the exact response
bytes a raw-socket client gets, once with `utils.py` at `e68ae02` and once at Checkpoint 173 — the only
file that differs. Side by side: a loopback upstream that sends `X-Repeat: first` **and**
`X-Repeat: second` arrives as **`X-Repeat: second` alone** at the base bytes (`first`, `Set-Cookie: a=1`
and the earlier `Date` all destroyed) and as **both lines, in the order they were sent** at the fixed
bytes; the fixture's two `Connection` lines are gone on both trees (the lone `Connection: close` in each
head is aiohttp's own answer, not a relayed occurrence), so the drop census still runs over occurrences;
`Content-Length` is still re-derived from the body actually sent. The transcript's second section
replays `evidence/s15-reach-and-base-comparison.txt` §2b's own measurement against the operator-declared
live upstream and the counter moves from **1/1 to 2/2** on `date`/`server` — on a response whose two
`date` values differed by a second, i.e. the 1-in-6 case §2a measured, so the value this branch was
destroying on the deployed path was a real timestamp and not a cosmetic duplicate. Two INDEX rows added
(the wire transcript and the fail-first/claims record), `Disposition` left empty for the user's ruling.

DEMO-EVIDENCE: `.sessions/codify/02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/evidence/INDEX.md` — no pause

## Next phase

**Yield** — one local Yield for a wave of one (`DC-S15-CE-2-r1`, `handoff-context.md` §Composition
routing 2026-10-06). `T017`'s retry counter is at **0**; this is the first attempt of this wave.
This pass claims **no** phase acceptance: `T017` is prework of the MS-001 close and must land **before**
Yield Red's final semantic pass, which owns `resolved` for `T015`/`T016`/`T017`, the phase acceptance,
the `Delivered` chronicle entry and parent completion — then `phase-boundary-finalize`, then the
`seam-S15` recomposition. `plan-summary.md`'s `T017` row now reads `implemented (075-01)` for that
Yield to convert.

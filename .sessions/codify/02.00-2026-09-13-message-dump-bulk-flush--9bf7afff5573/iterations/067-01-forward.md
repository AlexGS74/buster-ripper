# 067-01 Forward — T013 retry 1 (F-4 reopened by 066-01-yield; G-1 and G-2 accepted)

Session: `02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573` · Worktree: `/Users/alexhubitski/Projects/buster-ripper`
Task: `T013` retry 1, F-4 only · Base at start: `74d935e` (Checkpoint 140, the rejected state)
Checkpoints this pass: **141** `2be9783` (production) · **142** `ce55974` (witnesses) · then this record
Prior report: [066-01-forward](066-01-forward.md) · Reopened by: [066-01-yield](066-01-yield.md) · Evidence: [t013-retry1-header-parity.txt](../evidence/t013-retry1-header-parity.txt)
Blueprint: `handoff-context.md` §Planned Task Context → `T013`, `062 R-A`…`R-E`, ledger rows `DC-S10-C1/C2/C3/C4/LAT/D6D8-r1`. G-1's `DC-S11-*` rows and G-2 are **accepted** and were not re-touched: `messages.py` and `REFACTORING_PLAN.md` are absent from this retry's diff.

## Orchestrator Prompt (verbatim)

> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
> Worktree root: /Users/alexhubitski/Projects/buster-ripper
> Task: T013 — Yield retry 1 (F-4 reopened; G-1 and G-2 are accepted and must not be retried or restyled)
>
> Read `iterations/066-01-yield.md` first — it carries the closure packet `066-01-yield P1` routed to `DC-S10-C2-r1`. The finding, in Yield's words: `chat.py:113-114` selects the surviving credential with `next(v for v in values if not _is_empty_bearer(v))`, and `_is_empty_bearer("")` is `False` because `bool(value)` short-circuits first. A value-less `Authorization` line therefore counts as a token, and when it arrives *before* a real credential the name is deleted and `Authorization: ""` is re-added.
>
> Reproduce the four orderings against base `41d3fc6` before changing anything, and fix so the result never regresses base:
> - `""` then `"Bearer sk-real"` (different spellings) — base forwards `["", "Bearer sk-real"]`; must not end up `[""]`
> - `""` then `"Bearer sk-real"` (same spelling) — base collapses to `["Bearer sk-real"]`; must not end up `[""]`
> - the already-fixed `"Bearer "`/real cases in both orders — keep passing
> - the lone blank line case pinned as correct base parity in the existing table — keep passing
>
> Honor `DC-S10-C2-r1`'s own words, which the current build violates: when a credential is present the forwarded request carries that credential. Decide from every value under the name, case-insensitively, so the outcome is identical under the C parser and the pure-Python backend. Keep `utils.forward_headers` free of authorization semantics (`062 R-B`) and stay inside the existing table at `tests/test_header_parity.py:320-329` for the regression witnesses — Yield requires the failing cases pinned in-tree before the fix. Then re-run the M1–M8 matrix plus the two new orderings, the suite, and `QAP-MS004`, and record the retry in a new iteration report. Commit production and tests separately from session bookkeeping.

## Increment Brief

One defect, in one function, with witnesses for both of its branches. The accepted members stay
put. The scope-relevant question for Yield is not "is F-4 right now" but "is F-4's rule now
decidable from the value rather than from the position it arrived in".

## What was implemented

### The hole, reproduced first (no change made before this)

`_is_empty_bearer(value)` is `bool(value) and value.strip() in ("Bearer", "Bearer ")`. On `""`
the `bool(value)` short-circuits, so it returns `False` — and `False` means "not the empty-bearer
form" means "is a credential". The selector `next(v for v in values if not _is_empty_bearer(v))`
therefore returns the **blank line** whenever it arrives first, after which the name is deleted
and `Authorization: ""` is re-added. Three-way table (base `41d3fc6` / rejected `74d935e` / this
build), 15 shapes, full transcript in [the evidence record](../evidence/t013-retry1-header-parity.txt):

| Arrival, under `authorization` | base `41d3fc6` | `74d935e` (P1) | `2be9783` |
|---|---|---|---|
| `""`, then real — different spellings | `['', 'Bearer sk-real']` | `['']` | `['Bearer sk-real']` |
| `""`, then real — same spelling | `['Bearer sk-real']` | `['']` | `['Bearer sk-real']` |
| real, then `""` — same spelling | `['']` | `['Bearer sk-real']` | `['Bearer sk-real']` |
| real, then `Bearer ` — same spelling | `[]` | `['Bearer sk-real']` | `['Bearer sk-real']` |
| real, then `Bearer ` — different spellings | `['Bearer sk-real', 'Bearer ']` | `['Bearer sk-real']` | `['Bearer sk-real']` |
| `""` alone (base parity) | `['']` | `['']` | `['']` |
| `Bearer ` alone, both spellings | `[]` | `[]` | `[]` |
| `""` then `Bearer ` | `['', 'Bearer ']` | `['']` | `['']` |

**Zero credential regressions against base** across all 15 shapes, and exactly two rows change
`74d935e → 2be9783` — the two the packet names. The three rows where base itself destroyed the
credential (`real` arriving before a blank or an empty bearer, same spelling) stay fixed: that is
F-4's own claim, re-checked rather than assumed.

### The fix — `2be9783`, `chat.py` only

The selection became **positive**: `_carries_token(value)` asks whether the line *names* a
credential (`bool(stripped) and stripped != "Bearer"`), so blank and the value-less bearer form
both fail it and neither can pose as a token. Only when no value carries a token does the
value-less line survive by itself — base forwarded `Authorization:` and `062 R-D`'s rule was never
about it. `None` means "no value at all"; `""` means "the blank line"; the caller keeps them apart.
`_is_empty_bearer` is gone (it had no other caller in the package or the tests — `grep -rn
"_is_empty_bearer"` clean). `utils.forward_headers` gained nothing: the only name resembling
authorization in it is `proxy-authorization`, a `_HOP_BY_HOP` entry (`062 R-B`).

`request_headers.getall("authorization", ())` still decides from the request's own mapping,
case-insensitively, so the outcome is identical under `HttpRequestParserC` and
`HttpRequestParserPy` — both hand the handler two items, and the pick no longer depends on which
one collapses in `forward_headers`' plain dict.

### The witnesses — `142`, `tests/test_header_parity.py` only

Four rows added **inside** the table the packet points at (`test_the_rule_reads_the_request_mapping_not_the_collapsed_copy`,
`Authorization` blank × real, both spellings × both orders), whose existing assertion *is* the
required behavior: the forwarded mapping holds exactly the credential. Plus `_BLANK` named at
module level, and `test_a_value_less_authorization_line_is_not_touched` extended with the
`""` + `Bearer ` pair — the second branch's own falsifier.

Authored red first, as the packet demands: `2 failed, 18 passed` on `74d935e`'s bytes, both with
`assert [''] == ['Bearer sk-real']`; `20 passed` after `2be9783`. The red transcript is in the
evidence record. No socket axis was added — see §Concerns #4 for why that would have been a
second copy of a claim the two-parser witness already pins.

## Failure-Path Evidence

Matrix re-run against the retry's tree — **one mutation per run, `.pyc` purged, two agreeing tallies
each, restore in a `finally`**. M1–M8 are the accepted members' falsifiers (re-run because `chat.py`
moved, not because G-1/G-2 were reopened); M9–M11 are this retry's.

| # | Wrong build | Fired | Proves |
|---|---|---|---|
| M1 | `response_headers` without the `content-length` extension | 3 / 61 | `DC-S10-C3-r1` framing (the two ingress sites + the helper kept-set case) |
| M2 | messages' egress wired to `response_headers` | 2 / 61 | `DC-S10-LAT-r1` direction pairing, by spy and by behavior |
| M3 | `chat.py` at `41d3fc6` | 13 / 61 | F-4's whole egress family + the collapsed-dict Bearer cases (was 9 / 57; the 4 blank rows now fail here too) |
| M4 | post-commit arm returns a fresh 502 | 8 / 12 | `DC-S11-CE1-r1` — accepted member, unchanged reading |
| M5 | arm narrowed to `except aiohttp.ClientError` | 5 / 12 | `DC-S11-CE3-r1` — **corrected reading**, see below |
| M6 | post-commit log asserts upstream causation | 8 / 12 | `DC-S11-D8-r1` — unchanged |
| M7 | abandonment returns `None` | 4 / 12 | `DC-S11-CE1-r1` — unchanged |
| M8 | `messages.py` at `c31d1ac` (pre-G-1) | 10 / 12 | regression baseline — unchanged |
| **M9** | **`chat.py` at `74d935e` — the defect P1 names** | **2 / 61** | the packet's regression requirement: *only* the two blank-first rows go red |
| **M10** | `_carries_token` lets blank pose as a credential | 2 / 61 | same two rows, at predicate granularity |
| **M11** | the blank-line fallback branch removed | 1 / 61 | the base-parity witness `test_a_value_less_authorization_line_is_not_touched` |

**M5's number moved, and it was a measurement defect, not a code one.** Two chained runs reported
M5 = 9 failures; the same bytes in isolation give 5 — and 5 is the semantically right answer (only
the non-`ClientError` half escapes a narrowed arm; `ClientPayloadError` and
`ClientConnectionResetError` are still caught). Cause: CPython's `.pyc` source tag is
(whole-second mtime, size), and M4's edit and M5's edit each leave `messages.py` at **exactly
11325 bytes**, so the module cached under M4 validates for M5's source — a fresh 502 return was
reported for a file that contained none. Reproduced deliberately and printed in the evidence record
(`pyc_proof.py`: file on disk carries `except aiohttp.ClientError`, loaded code object still
references `json_response`). The driver now purges bytecode, runs under
`PYTHONDONTWRITEBYTECODE=1`, requires two agreeing tallies, and restores in `finally` (an earlier
draft aborted mid-case and left `utils.py` mutated).

Scope, per `codify-scope`: the fix above is **rule 3** (this session could not get a trustworthy
matrix result without it) applied to this pass's own scaffolding in the temp dir — **no repo file
joined the diff for it**, and no harness was built. The general hazard is out of scope for T013 and
is tracked as `QB:process/hand-rolled-matrix-pyc-staleness`; it gets no wave here.

Also relevant to how much weight M9-M11 can bear: this is the same measurement-discipline family
`066-01-yield` P1 caught in the code. A chained matrix that had reported 5 for M5 and 9 for M4
would have been accepted on the strength of the same confidence the Bearer rule had.

## Repair delta — sensitive-data questions

`chat.py:_prepare_headers` is a credential path, so the `066-01-yield` "sensitive data" bar applies
to this retry as written:

| Question | Answer for `2be9783` |
|---|---|
| Can a secret be sent where it should not be? | No new path. The rule's *only* deletions are the value-less bearer form; a credential is forwarded to the configured upstream exactly as before. |
| Can a secret be dropped that should be forwarded? | This is what P1 did. Now: whenever any value under the name carries a token, that token is forwarded — pinned by 8 table rows (4 bearer-family, 4 blank-family) and falsified by M9/M10. |
| Can a secret be logged? | No. Nothing in the retry touches logging; the value never enters a log record. |
| Is a value trusted because of its shape rather than its origin? | Still shape-based, and that is the pre-existing lm-eval contract (`config.py:91`, `legacy.py:838` "Model-agnostic"). What changed is the question asked of the shape: not "is this the empty-bearer form" (which a blank line passes for a token) but "does this name a credential". A client-supplied `Authorization: Bearer`-shaped string is still just a string forwarded to the upstream the operator configured. |
| Same name, different spellings — does ordering change the outcome? | No. `getall` over the request mapping, first credential wins, all spellings of the name deleted before re-adding one. If **two** real credentials arrive, the first wins — mechanism, not contract (unchanged from `74d935e`; recorded in §Concerns #3). |

## L→N→C→P compliance (this tree)

| Function | Before (`74d935e`) | After (`2be9783`) |
|---|---|---|
| `chat._prepare_headers` | clean (`[+]`) | clean — 22 body lines, N:2, C:5, no report |
| `chat._is_empty_bearer` (removed) | clean | — |
| `chat._carries_token` (new) | — | clean, 2 statements |
| `chat.py` file scan | 18 issues in 8 functions | **18 issues in 8 functions** (unchanged) |
| `chat.py` raw lines | 295 | **299 — under the 300 Green ceiling** |
| `tests/test_header_parity.py` | 15 issues in 14 functions | 13 issues in 13 functions, 355 lines |

The FL-LNCP Green ceiling on raw file size (review at ≤300) was treated as a design input,
as was `chat.py`'s standing near it. It landed under, at 299, without dropping the reason the
branches exist: the second candidate shape was
to move the rule into `utils` (forbidden by `062 R-B`, and it would have smuggled chat's asymmetry
into a shared policy) and the third was to cut the blank-vs-`None` comment (the distinction is the
part a future reader is most likely to break). No FL-LNCP review item was needed. `Too Many
Parameters` across all four test files: 0.
(Erratum at drain — `QB:report-accuracy/067-01-invented-quotation`: this paragraph previously
quoted `066-01-yield` P2 as warning "chat.py at 296 lines and the ceiling is 300; the next chat
change has to land under it or file the review item with the change". That report contains no
`296` and no ceiling framing — `grep -n "296\|ceiling" iterations/066-01-yield.md` → no match;
its actual P2 is the post-commit second-head class still live in the chat relay, tracked at
`roadmap.md` §Chat route's post-commit second head. The underlying fact — 295 → 299, under the
Green ceiling — was true and handled correctly, so only the attribution is repaired. The echo
this item also named in a prior `handoff-context.md` §Current Status is no longer present —
`grep -n "land under it\|296 lines" handoff-context.md` → no match, re-proved at drain.)

## Tests and coverage

- `uv run pytest -q` → **374 passed**, 23 warnings, coverage **94.20 %** (370 at the rejected state; +4 rows)
- `uv run pytest tests/test_header_parity.py tests/test_stream_relay_commit.py -q --no-cov` → **73 passed in 0.10 s** (61 + 12; was 69)
- `uv run pytest tests/qa/test_qap_ms004.py --no-cov` → **6 passed in 44.31 s** (`QAP-MS004` done-when F-4 (4), re-run against `2be9783`'s production)
- Production-only state verified green before the witness commit: with `tests/` at `74d935e` and `chat.py` fixed, 370 passed — so `141` is not a red commit, and `142` is not load-bearing for correctness.

## Evidence

[evidence/t013-retry1-header-parity.txt](../evidence/t013-retry1-header-parity.txt) — generated by
script from live commands, not typed: the three-way reproduction table, the red-before-fix
transcript, M1–M11 with per-case failing ids, the `pyc_proof.py` transcript and what it cost, the
census and direction-pairing greps re-run against this tree, the `062 R-B` check on `utils.py`,
the `DC-S10-C4-r1` byte-unchanged proof re-run over `41d3fc6..HEAD`, the `DC-claim-208-r1`
zero-citation counts (diff and commit messages), the suite/gate timings, and the measured LNCP table.

## Concerns for Yield

1. **`_carries_token` keeps the rule's case-sensitivity exactly where it was.** `"bearer"` (lowercase
   scheme) is *not* the value-less bearer form, so it is forwarded as a credential — as base did and
   as `74d935e` did. Changing that would be a new behavior with no block asking for it; flagged
   because a reader may expect `strip()` to have made the scheme case-insensitive. It only normalizes
   surrounding OWS.
2. **`_prepare_headers` re-emits the surviving credential under canonical `Authorization`,** so a
   peer's `authorization:` spelling disappears on the wire. Inherited from `141`'s predecessor (this
   retry changed *which* value is chosen, not the name it is chosen under) and observable only under
   `AIOHTTP_NO_EXTENSIONS=1`. Unchanged claim, still disclosed.
3. **Two real credentials under one name: first wins.** Mechanism, not contract — base's outcome there
   was backend-dependent, and the block's words only require "when a credential is present the
   forwarded request carries that credential". No witness pins the *second* token's fate; if Yield
   wants that pinned, it is one row in the same table, not a new test.
4. **Why no raw-socket axis for the blank line.** The two-parser witness already proves both lines
   reach the handler (`len(arrived) == 2`) and that the *selection* is what differs per backend; the
   blank-family rows decide which of two values `getall` returned, which is a unit-level claim about
   `_prepare_headers`. Adding a shape axis to that witness would double 4 socket cases to 8 to re-prove
   an arrival mechanism already pinned (test-cost rules 3 and 5). If Yield wants the wire claim
   anyway, the cheapest honest version is one order added to `ids`, not a new test.
5. **One commit-message figure in this pass is wrong and was not amended:** `142`'s body says
   `335 -> 351` lines; the module is **355**. History here is append-only, so the correction lives
   here, in the evidence record, and in `QB:test-header-parity/file-size`.
6. **M3's reading grew (9/57 → 13/61) purely because the module grew.** Not new sensitivity; recorded
   so the delta isn't read as a behavioral change in `chat.py`'s base comparison.

## Invariant Contradictions Raised

None. `062 R-A`…`R-E` implemented as written; no binding reopened, no referent added, no middleware,
no sixth call site. `DC-S10-C2-r1`'s own words are now satisfied (the forwarded request carries the
credential when one is present). `DC-S10-C4-r1`'s prohibition re-proved by diff: the five
proxy-authored response header literals are untouched and `chat.py` still does not import
`response_headers`. `DC-claim-208-r1`: no RFC citation in the diff or in any of the twelve wave commit
messages (basis stated at drain per `QB:report-accuracy/commit-count-basis`: thirteen commits in
`41d3fc6..HEAD`, twelve excluding yield checkpoint 140 — `066-01-forward`'s "seven" counted the
tree at its own time, checkpoints 131–137, both bases are true of their range); `Proxy-Connection` not added, `trailers` not respelled. No new invariant, no invariant
edited, no milestone gate touched (`QAP-MS004` ran as a re-check, not as a gate claim).

## Assumption / seam obligations this pass owes on return

- `DC-S10-C2-r1`'s verdict row must be **re-issued**: its bytes moved (`74d935e → 2be9783`), and the
  claim it verified is precisely the one that was found false. Same for the `seam-S10` composition,
  which consumed the pre-fix selection semantics.
- `seam-S9`'s rider member-row and `seam-S10`/`seam-S11` recomposition are still owed (unchanged from
  066-01's list); `seam-S11` needs no recomposition for *behavior* this pass — `messages.py` is
  byte-unchanged — but its digest claim should be re-checked against the tree it was recorded on.
- New library fact this pass, for the claim registry rather than for code: **two mutations of equal
  source size written in the same second defeat `.pyc` validation**, which can silently falsify a
  hand-rolled mutation matrix. Belongs to `forward-collect`/assumption territory about *our* tooling,
  not about aiohttp.

## Next

One **local Yield** over `2be9783` and `ce55974`, scoped to F-4 (the two accepted members are not in
the diff and should be verified as untouched rather than re-reviewed). Then the assumption route for
the `DC-S10-C2-r1` re-issue and `seam-S10` recomposition.

## Session artifacts this pass wrote

`iterations/067-01-forward.md` (this file) · `evidence/t013-retry1-header-parity.txt` ·
`plan-summary.md` (T013 row: retry status, corrected counts) · `handoff-context.md` §Current Status ·
`evidence/INDEX.md` (retry row; Disposition left empty for the user) · `quality-backlog.md`
(the two stale pointers `066-01-yield` P3 named, corrected with measured numbers;
`QB:test-header-parity/file-size` 335 → 355; new `QB:process/hand-rolled-matrix-pyc-staleness`).

## Superseded plan wording (preserved verbatim per the rewrite rule)

From [066-01-forward](066-01-forward.md) §What was implemented, line 37 — the claim `066-01-yield` P1
proved false as implemented (the sentence's *intent* survives; the mechanism it describes did not):

> - `chat.py`'s empty-`Bearer` rule rewritten per `DC-S10-C2-r1`: it now reads **every** value under that name from the request's own mapping (`request_headers.getall("authorization", ())`), removes the name only when no non-empty bearer is among them, and re-adds the surviving credential when one is. It stays in `_prepare_headers`; `utils.forward_headers` gained no authorization semantics (`062 R-B`).

From the same report, §L→N→C→P line 115 (the helper this retry removed) and §Repair delta line 105
(the selection's criterion, restated):

> | `chat._is_empty_bearer` (new) | — | clean |

> | Is a value trusted because of its shape rather than its origin? | The rule decides on shape (`value.strip() == "Bearer"`), which is the pre-existing lm-eval contract (`config.py:91`, `legacy.py:838` "Model-agnostic"). What changed is *which* shape wins: a non-empty value now outranks an empty one instead of last-wins-by-backend. Origin is not a thing the proxy asserts. |

From the same report's §Evidence table — M5's superseded reading, replaced by the corrected
measurement in this report's matrix above:

> | M5 | arm narrowed to `except aiohttp.ClientError` | 5 / 12 | the non-`ClientError` half escapes again, and the structural one-catch-all witness fires (`DC-S11-CE3-r1`) |

*(That row's number was right; its provenance was not — it was re-measured this pass under bytecode
purging, and it is the later chained runs in 067 that were wrong. Preserved here so the lineage of
both readings is visible.)*

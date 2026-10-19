# 084 · Assumption Decide — run-45 atomic non-holding routing (`S20`, the `web.Response` body round trip)

## Orchestrator Prompt (abridged — the dispatch's own words)

> Mode: atomic non-holding routing (run 45 response-body set). Four rows need dispositions, all members of the
> fresh seam `S20` (`web.Response` body round trip, `State: new`): **418**, **422**, **423** `DOES-NOT-HOLD`, and
> **419** `UNCLEAR`/ineligible. Apply the Reach Gate before deciding any of them, and read the repository bytes to
> do it — the standing third-party-only ruling limits which propositions may become *ledger rows*, not what may be
> read to judge reach. Determine from the actual code whether `S18`'s Note N6 path is reachable and whether the
> SGLang streaming context-overflow retry is in fact inert: what constructs the response the matcher reads, whether
> it passes `body=` or `text=`, what the matcher receives, and what the pre-migration base did at the equivalent
> site — base parity is the authority that ranks whether this is owed a repair, deferred, or already accepted. The
> composer's hedge was that the delta's output flows into that site, "which means it may also have changed which of
> those is true; check that specifically rather than inheriting the note's framing." Constraint: third-party
> subjects only for ledger rows — no repo row, no repo seam, `S1`–`S15` stay terminal, `FK-011` remains the only
> open Fork.

## Reach adjudicated from the bytes, then by execution

Recorded in [`../evidence/n6-reach-and-base-comparison.txt`](../evidence/n6-reach-and-base-comparison.txt).
N6's chain is **real**, and it is **reached** — but the reach verdict corrects three things the note left open or
stated as a hedge.

1. **What constructs the object the matcher reads.** `_stream_single_request` at `messages.py:149-155`: on
   `resp.status >= 400` it calls `_error_body(resp)` (returns `str`) and constructs
   `web.Response(body=error_content, status=resp.status, content_type="application/json")` — **`body=`, not `text=`**.
2. **What the matcher receives.** `messages.py:106` reads `.body` back. `418` holds it up: a `str` at `body=` is
   stored as a `StringPayload`, so `isinstance(result.body, bytes)` is false, `str(result.body)` is
   `<aiohttp.payload.StringPayload object at 0x…>`, and `parse_context_overflow_error`
   (`utils.py:123-140`) matches that against `config.py:122-123`'s ASCII patterns — `CONTEXT_OVERFLOW_RE` cannot
   match an object repr. Executed against the installed aiohttp 3.13.3 and the production helper: the matcher
   returns `None`; on the same object `.text` returns the text and the matcher fires.
3. **The retry is inert on both arms, not one.** The loop's other producer, the `except aiohttp.ClientError` 502 at
   `:171-173`, returns `web.json_response(...)` — whose `text=` fast path stores **bytes**, so `.body` *is* bytes
   there and the `isinstance` branch works. It still cannot match: since `T016`'s de-leak the body is the static
   literal `{"error": "Upstream request failed"}`. So `handle_streaming`'s retry can never `continue` by any path,
   and its terminal arm at `:119-123` is unreachable too.
4. **Census bounds the surface.** `grep -rn "\.body\b|body=" buster_ripper/` → one `str` enters `body=`
   (`messages.py:152`) and one site reads `.body` back (`messages.py:106`). `chat.py:172`/`:188`, `messages.py:317`
   and `passthrough.py:44` all pass bytes from `resp.read()`; `chat.py:178` feeds the matcher
   `json.dumps(resp_data)`. This is a single producer/consumer pair, and the package's own convention at the peer
   sites is to hand the matcher text it built deliberately.
5. **Reach.** `reached`. Path: `POST /v1/messages {stream:true}` → `messages.py:67-68` → `handle_streaming:101-107`
   ← `_stream_single_request:149-155` → `None` at `:107` → relay at `:125`. The gate is
   `backend_type == "sglang"`, set by `--backend-type sglang` (`main.py:129`) or `/v1/models` auto-detect
   (`server.py:71`, `backend.py:81`) — configuration-supplied input, which the Reach Gate counts as reached
   ("exposed surfaces are reached by definition"; "priority is not reach"). Not the 1%/0.01% category: an oversized
   streaming turn against SGLang is the feature's designed trigger. Nothing pins it either
   (`tests/test_context_overflow.py`'s `TestBackendAwareRetry` asserts only `get_backend_state()` values — no test
   enters the loop), so fix-or-pin resolves to **fix**.

**The delta question, checked rather than inherited.** The note was right that the condition predates it: the prior
bytes already had `error_content = await resp.text()` — a `str` (`handoff-context.md:1278`, `S18-MC-1`), so `:106`
already took the same branch, and the delta changed only **how** the `str` is produced. But the note's hedge was
stronger than the truth in one direction and weaker in another, and both matter:

- **Stronger:** the delta did change *which inputs arrive at the dead branch*. Pre-`T019`, an error body undecodable
  under the response's charset raised `UnicodeDecodeError` past `:171` and never reached the probe at all; post-`T019`,
  `_error_body` returns replacement-decoded text that does. The population entering the dead path grew; the outcome
  inside it did not change.
- **Weaker:** the note offered the wire as the reassurance ("the peer-visible bytes are unaffected"). That survives
  and is now evidenced (`420`/`421` `HOLD`, plus the executed `Content-Type: application/json` and the payload bytes).
  What the note did not say is that the *other* arm of the loop is inert as well, so the feature is dead whole rather
  than dead in one branch.

**Base parity, applied as the ranking authority: silent, not protective.** The base this session cites for parity
elsewhere (`buster_ripper_legacy.py` — `:664` stream deadline, `:630-637` headers, `:674-676` `message_start`) contains
no `overflow`, `sglang`, `retry`, `backend` or `max_completion` in its 1013 lines, and its streaming branch has no
status-checked error arm at all — an upstream 400 arrived inside a 200 SSE body. There is no base behavior at this
site, so base parity neither owes the repair nor excuses it, and the route that base parity opened for §RM-008 and
§RM-009 (defer to the design pass) is not available here. The condition is **migration-introduced and inherited at
session entry** — `iterations/001-collect.md` §B-005 already listed "context-overflow retry … behave as tested" as
pre-existing behavior at `messages.py:73-268` — and **not** delta-introduced.

## Dispositions

| Row | State (untouched) | Routing | Basis |
|---|---|---|---|
| 418 | `DOES-NOT-HOLD` | **`absorbed`** | `DC-claim-418-r1` → `T019.001`; reach `reached`, fix-or-pin resolves to fix |
| 419 | `UNCLEAR` | **`malformed`** | the noun, not the behaviour; returned to `forward-collect` with successor wording |
| 422 | `DOES-NOT-HOLD` | **`absorbed`** | `DC-claim-422-r1` → `T019.001` step 1; certifies the repair's read surface |
| 423 | `DOES-NOT-HOLD` | **`absorbed`** | `DC-claim-423-r1` → `T019.001` step 1; forecloses every coercion-shaped candidate |

Rows 417/420/421/424 are `HOLD`/`none` and were not touched. No `DOES-NOT-HOLD` row here carries `Check: user-claim`,
so nothing in `decisions.md` is `CHALLENGED` and `USER-CLAIM FALSIFIED` is not emitted. All seven publishable rows are
already in the registry as `CL:` records — evidence inputs, cited and not treated as authority; no registry byte was
written or promoted by this pass.

**Deterministic Closure Gate: `CLOSURE: absorbed`, one remediation.** Required outcome: the probe receives the
upstream's error *text* so the retry behaves as `handle_non_streaming` already does at `:296-305`, with the relayed
response byte-identical to today. Selected closure: change `messages.py:106`'s read to the response's own text surface
(`.text`), leaving the constructor alone. Rejected and why:

- **`web.Response(text=error_content, …)`** — verified to store bytes (so the current consumer would start working)
  **and** verified to emit `application/json; charset=utf-8` where it emits `application/json` today: a
  client-visible header change on the relay arm this session's accepted fidelity work (`T015`/`T017`,
  `CL:aiohttp/web-response-explicit-content-type`) governs. Broader observable delta than the defect requires.
- **`_error_body` returns bytes** — displaces `T019`'s approved contract ("whose catch list is **exactly** as
  approved") and moves a landed surface for no gain over a consumer-side read.
- **Delete the retry loop as dead** — preserves the verified defect and contradicts the function's declared behavior.
- **Keep the coercion and probe for `decode`** — a new mechanism at one site where a named, certified accessor exists.
- **Accept as a recorded risk / route to the roadmap** — the first fails the gate's `reached` rule (fix or pin, and
  nothing pins it); the second fails `codify-scope` rule 5, since the cause is in `messages.py`, a file this
  session's tasks name and CR3 just changed.

**Invariant cross-reference: Neutral** for all four. No session `invariants.md`; `invariants/` has no
`buster_ripper/routes/*` file; all eight registry entries touching this session are `Status: planned`
(`PI:T002/dump_format/*`, `PI:T006/upstream_tls/*`) and none reaches this site; zero `active`/`wip` invariants. One
invariant candidate is recorded in `T019.001`'s block for Interview to rule on — nothing written or promoted here.

## Route, and the wave question answered plainly

`T019.001` (nested under the ledger's owning task for this surface, `T019`), its own one-member wave, **dispatched only
after the CR3 wave's local Yield accepts**.

The CR3 wave does **not** carry it, and the reason is not capacity:

- `T018`/`T019`/`T020` are **implemented**, not scheduled. Their evidence was produced against a snapshot that
  excludes this change — 14 fail-first witnesses replayed red at pre-fix bytes, byte-exact AST verification of the
  must-not-touch surfaces, hermetic 471 @ 95.91%. `remediation-waves.md` admits a ride-along into a wave "already
  scheduled"; riding into an already-executed one would force the wave's fail-first baseline to be re-derived at a
  second base and would put a 15th witness red against bytes it is about to have accepted.
- `handle_streaming` is in **no CR3 member's touch-set**, and `073-01-yield` verified it byte-identical across the
  previous wave — folding a change into the increment now would disturb a preservation a Yield already certified.
- Capacity was checked, not assumed, and it is not the binding reason: `chat.py` stays at 300/300 untouched, the fix
  is line-neutral, and `messages.py`'s review-band entry (`QB:routes-messages/file-size-yellow`, 323 lines) owes a
  misplacement check *at next touch* — which `T019.001`'s done-when (6) now carries rather than deferring.
- Honest counterweight: had the wave still been schedulable, this finding **would** have ridden along. It is one
  expression, in the file CR3 was already in, at reach `reached`.

Evidence is pinned rather than narrated: `evidence/n6-reach-and-base-comparison.txt` holds the cited bytes, the
executed check, the census, the base comparison, the witness-absence search and the reach verdict, each under its own
section, so the owning route and Yield Red can re-derive without this pass's context.

## The malformed return — what row 425 must ask (for the collector)

419 stays `UNCLEAR`/`malformed` as evidence; it is not re-worded, re-checked or re-dispatched under its own id. Its
successor is a **new row 425**, `pkg:aiohttp@3.13.3`, seam `S20`, fresh bundle key, one proposition, answerable from
the digests run 45 pins:

> "For an aiohttp `web.Response` built with a `str` `body=`, the payload its `body` getter returns decodes back to
> that same string under the response's own charset."

Wording constraints: the phrase "those `.body` bytes" is forbidden (419's own blocker — no bytes-valued referent);
the getter's **type** must not be re-rowed (owned by 418 / `CL:aiohttp/web-response-body-getter-type`); the `text`
**property** must not be re-rowed (owned by 421 / `CL:aiohttp/web-response-str-body-text-property`); and no `repo:`
subject. At the verdicted bytes it lands `HOLDS` (`payload.py:442` encodes utf-8; `:380-381` decodes), which is what
`S20`'s "equality with the input across that round trip" dimension needs from a member. **`S20` cannot recompose while
any member is non-`HOLD`**, so 425 gates the recomposition; `S20`'s `State`/`Routing`/`Permanent` cells stay as run 45
wrote them (`new`/`unverified`/`—`).

## What the falsified rows still establish, independent of the repair

`web.Response` accepts a `str` at `body=` without raising (417) and stores a `StringPayload` (418) — so any code that
reads `.body` back from a `str`-built response is reading a payload object, and the census shows this package has
exactly one such read. The value is not lost: it survives to the wire (420) and to `.text` (421), and `.text` is the
*only* accessor on the class that returns it (422's enumeration) — which retires "the text never arrives" as an
exhaustive-surface claim while leaving the call-site observation true, and converts N6 from a note into a
boundary whose three doors are now individually evidenced. `str()` of the payload is the default repr (423, via 424),
so no coercion of `.body` can yield text: the candidate space at this site is closed to one read.

## Durable writes

- `assumptions/claims.md` — four `Routing` cells only (418/422/423 → `absorbed`, 419 → `malformed`); `State` cells
  untouched; new §Run 45 routing changelog, including the row-425 wording for the collector.
- `assumptions/seams.md` — new §Run-45 atomic routing changelog. **No seam row's `State`/`Routing`/`Permanent`
  moved**; `S20` stays `new`/`unverified`/`—`; `S1`–`S15` untouched and terminal; `S11`'s §Invalidates On 5 is cited,
  not reopened — the direction established here is the opposite of the one it anticipated (no reachable arm's text can
  match), which strengthens `DC-S11-SGL-r1`'s foreclosure rather than waking it.
- `handoff-context.md` — §Deterministic Closure Ledger — 2026-10-07 (`DC-claim-418-r1` primary + `-422-r1` +
  `-423-r1`, all at repository state `git-commit:9c90675f69998e0ea7847c6b456c2e3f46f8a81c`);
  §Planned Task Context → `T019.001` with every transmitted field (counterexample, required behavior, done-when,
  Yield replay incl. the mutation, assumption impact, cross-reference, invariant candidate, closure ledger keys);
  §Assumption routing — 2026-10-07; one §Current Status bullet.
- `plan-summary.md` — `T019.001` row (nested immediately after `T019`) + its pointer block.
- `evidence/n6-reach-and-base-comparison.txt` — new.
- Zero `decisions.md` bytes, zero `invariants/` bytes, zero production or test bytes. **No Fork reserved: `FK-011`
  remains the session's only open Fork**, and no question was admissible — base parity is silent here rather than
  ranking two outcomes, and the file's own working peer supplies the required behavior.

## Context budget

`handoff-context.md` and `plan-summary.md` were already over cap before this pass (the standing disclosure at
[075-01-yield](iterations/075-01-yield.md) §Context budget and Checkpoints 122/168: the overage is the live material
of this boundary, and the deep trim belongs to `phase-boundary-finalize` / `session-squash`). Per the cap rule this
pass moved out at least what it added, verbatim, nothing rewritten: five per-pass routing narratives (CR3 run-42
composition routing, run-40 atomic routing, the CR2 closure-packet routing, both `seam-S15` composition routings)
→ `handoff-archive.md` §Trimmed 2026-10-07 (`084-assumption-decide`), with a §Per-pass routing narratives trimmed
index left in place; the `T015`/`T016`/`T017` pointer blocks → `plan-completed.md` §Trimmed 2026-10-07. Net:
`handoff-context.md` 1910 → 1878 lines, `plan-summary.md` 452 → 441. Every `DC-*` entry and every
`## Planned Task Context` blueprint those passes wrote stays in `handoff-context.md`.

One record row added so the discovery cannot be re-inherited: `quality-backlog.md`
`QB:report-accuracy/001-collect-b005-retry-as-tested` — `iterations/001-collect.md` §B-005's "context-overflow
retry … behave as tested" is now disproved for the streaming half on bytes, on execution and on the witness search.
Reports are not rewritten; the erratum lives on the backlog row, whose trigger is any later pass citing §B-005 as
characterization of the streaming path.

## Receipt (post-write hashes; repository state at routing time `git-commit:9c90675f69998e0ea7847c6b456c2e3f46f8a81c`)

| File | `sha256` after this pass | What moved |
|---|---|---|
| `assumptions/claims.md` | `36fee430c8e250dde3dd1412c2d08d590f3d44c06a82419c9b85249ff81dec98` | four `Routing` cells (418/422/423 `absorbed`, 419 `malformed`), no `State` touched; §Run 45 routing changelog with the row-425 wording |
| `assumptions/seams.md` | `7fc22845f14b3a3564cae9878b226ec901d1a9da092b75887763d80410cf2654` | §Run-45 atomic routing changelog appended only — no seam row's `State`/`Routing`/`Permanent` moved |
| `handoff-context.md` | `1ec39c03676689754b4260fa4f36eca863d82f7cae631e23b4ed3a9cba853e49` | one status bullet, three ledger rows, one `T019.001` blueprint, one routing section; five narratives moved out verbatim |
| `handoff-archive.md` | `5e3e8af869740719114ad0815d8ddca5555258e8b6237adbd7767a234511466c` | the five narratives, verbatim |
| `plan-summary.md` | `afb1add9bafbcc28befd6448f2c2faad00b4d6e9ce4967267278022c01077c02` | the `T019.001` row and its pointer block; three consumed pointer blocks moved out |
| `plan-completed.md` | `e2758f3cbc0d4fc04d29f7a25107f76f952ffad36d5171ecccc7245978658ba8` | the `T015`/`T016`/`T017` pointer blocks, verbatim |
| `quality-backlog.md` | `df4e7ba5835689bb4f750f14461a4f73b9d5c9be94778ed36f7355aeb40358bc` | one row: `QB:report-accuracy/001-collect-b005-retry-as-tested` |
| `evidence/n6-reach-and-base-comparison.txt` | `2d7d11cb934afd8b62d6b689726fefa7519ee3d78fd9534faf0ae98b8709b693` | new — the reach determination and base comparison, pinned |

Every absorbed finding has exactly one ledger row, one task-step landing and one `Planned Task Context` block; the
malformed row names its exact successor wording and the seam it gates; nothing was marked `resolved` by this pass.
Zero production and zero test bytes moved; zero `decisions.md` and zero `invariants/` bytes moved. Every count below
corresponds to a durable record — nothing exists only in this response.

## Assumption queue state

`S20` is the only seam this pass touched, and only by description: 418/422/423 routed `absorbed` and bound to
`T019.001`, 419 returned to the collector. No member of any seam remains at `Routing: pending`. `S20`'s recomposition
is gated on successor row 425; `S18` stays open on its `MC-2` atom as run 42 left it, and `S16`–`S19` stay open on
their post-implementation recompositions. Recomposition cap: none — `S20` has never been composed, so this is not a
second pass on any seam.

```text
ASSUMPTIONS: 4 routed | 3 absorbed | 0 questions | 0 breaks | 1 malformed | 0 accepted | 0 capped
```

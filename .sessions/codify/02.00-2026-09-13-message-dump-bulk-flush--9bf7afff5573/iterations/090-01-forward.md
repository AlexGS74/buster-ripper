# 090-01 Forward — `T019.002` (CR4 remediation, one-member wave, nested under `T019`)

Session: `02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573` · Worktree: `/Users/alexhubitski/Projects/buster-ripper`
Branch: `refactor/package-structure` · Wave base: `c5e2be4` (Checkpoint 204) · this increment: **f00216c (Checkpoint 205)**, worktree clean — no capacity extraction owed (see §Capacity gate)
Task: `T019.002` (adhoc remediation nested under `T019` on `T019`'s owning surface; **not** a `T019` retry — `T019`'s counter stays 0, its landed range `d5df8b06…9c90675` untouched)
Blueprint: `handoff-context.md` §Planned Task Context → `T019.002`
Ledger: `DC-CR4-F-01-r1` at `git-commit:c5e2be44ba1a7ab984f02104472f06f849e4f773`
Origin: `iterations/088-codex-review.md` §[P2], routed by `iterations/089-assumption-decide.md`
Evidence: [t019-002-fail-first-run.txt](../evidence/t019-002-fail-first-run.txt) · [t019-002-fail-first-and-replays.txt](../evidence/t019-002-fail-first-and-replays.txt) (§A fail-first, §B–§D the three mutation replays, §E–§H touch-set, sizes and LNCP) · premise matrix: [cr4-probe-matrix.txt](../evidence/cr4-probe-matrix.txt)

## Orchestrator Prompt (verbatim)

> You are a subagent spawned by another session.
> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
> Task: T019.002

Result: **DONE** — the narrowing lands as one call at `count_tokens.py:87` replaced by one call to a new public delegating entry in `normalize.py` (+13/−0 there, +8/−8 at the route, **zero net line shift** in `count_tokens.py`); `session.py`, `routes/messages.py`, `routes/chat.py`, `buster_ripper_legacy.py` and `invariants/` carry **zero bytes** in the diff, and `normalize()`'s four step bodies and `normalize()` itself are byte-identical. 35 new witnesses in one focused module: **3 red at pre-fix bytes** printing the escaping `AttributeError` frames, 12 green in that state by design, and all three of the blueprint's red-directions executed by this pass — replay (b) restores the pre-`T019` spelling and reddens **both** the new route oracle and the pre-existing nonce witness, replay (c) hardens `normalize()`'s loops and reddens the `/v1/messages` guard **500 → 502**, replay (d) adds a hypothetical `system`-rewriting step and reddens the pure digest battery on every affected shape. Hermetic **511 passed @ 96.71%** (baseline 476 @ 95.91%: +35, none removed, coverage up). `handle_count_tokens` re-measured **L:18 N:1 C:3 — identical at HEAD and at final bytes** (measured from the HEAD blob, because the backlog's recorded `L:17` predates `T019` step 1; see §Concerns 2).

## Increment Brief

- **Task and goal**: `T019.002` — make `/v1/messages/count_tokens` answer an estimate for every JSON object body the way base does, by keying its digest on the system-strip step alone instead of on the whole of `normalize()`.
- **Anchors touched**: none of §Anchor Map's rows — that map binds this session's dump/TLS cluster. The two sites are named by the route that created the task: `routes/count_tokens.py:handle_count_tokens` and `normalize._normalize_system`, both existing bound referents (`extend`/`reuse`). Precedent: `085-01-forward` §Increment Brief, same shape of remediation-routed task.
- **Will NOT create**: no second normalization pathway beside `normalize()` (the entry delegates, it does not re-implement); no copy of `_normalize_system`; no second session-digest surface beside `session.session_id`; no duplicate of `tests/upstream_stub.py` or of `conftest.py`'s `proxy`/`raw_proxy` fixtures; no new witness in `tests/test_routes_integration.py` (495 raw lines — its own band is the reason this module exists).
- **Creation gate**: one new symbol, `normalize.strip_billing_nonce`, authorized verbatim by the blueprint's Selected closure — "it stays byte-exact if the route imports the step directly, **or gains one delegating entry** if the implementation prefers a public name for a cross-module call … the naming is not a different outcome". Replayed before writing: name searches (`strip`, `nonce`, `billing`) and role/peer searches over `buster_ripper/` — `tests/test_normalize.py`'s 14 `normalize()` tests and `tests/header_parity.py`'s `_HOP_BY_HOP` precedent were the only near-matches, neither owns this step; no viable existing public owner (`normalize()` is the sole public name in the module and calling it is the defect). `Inventions: none` beyond that entry and module-private test scaffolding (`_copy`, `_battery`, the shape constants).
- **Protected invariants**: none contradicted. Cross-reference is **Aligned** — the narrowing implements the property `EI:buster_ripper.session/session_id/derived_after_normalize` carries, bound to the existing handle, no duplicate candidate; that invariant stays deliberately registry-inactive (§Notes). `G-003` is the settled goal the blueprint ranks the wider candidate out with, and clause 5's witnesses exist so this increment cannot drift into it. **Zero `invariants/` bytes moved** — the triad rule held.
- **QA contract**: `qa-plan.md` present. `T019.002` carries **no `VP-*`/`TC-*`/`MS-*` handle** of its own — its gate is this increment's local Yield; `QAP-MS001` re-executes at the boundary's Yield Phase on the final bytes. No `tests/qa/` file was inspected, authored or run here.
- **Assumption finding**: `DC-CR4-F-01-r1` (closure-packet mode, names no seam). **Assumption impact: none owed** per the blueprint. Implemented for Yield to rule on; not closed by this pass.

## Failure-Path Evidence

| Operation / condition | Expected outcome and authority | Handling / observation evidence | Disposition |
|---|---|---|---|
| `{"tools":[null]}` → `POST /v1/messages/count_tokens` | **200 + `{"input_tokens": <int>}`** — base `buster_ripper.py:672-696` has no `normalize` call on that path (matrix §2) | `[…][tools-null-element]` — red at pre-fix bytes as a 500 printing the `AttributeError` frame inside `_normalize_tools` (§A), green final | safe (repaired) |
| `{"messages":[null]}` → same route | 200, same authority (matrix §1/§2) | `[…][messages-null-element]` — red pre-fix at the `_strip_cache_control_from_messages` frame, green final | safe (repaired) |
| `{"messages":["x"]}` → same route | 200 (matrix §1/§2) | `[…][messages-string-element]` — red pre-fix (`'str' object has no attribute 'get'`), green final | safe (repaired) |
| `{"system":[null]}`, `{"messages":[{"content":[null]}]}`, `{}`, unparseable body | **200 before and after** — the over-reach ceiling; base and `6508941` both tolerate (matrix §1, §2) | `test_a_shape_base_already_tolerated_stays_tolerated` ×4 — green in the pre-fix run too, deliberately: these are the pins that show the repair did not widen the route's crash | safe (pin) |
| `[1,2]` (a non-mapping body) | **still 500** — base rejects it too, out of `_session_id` at `:449` (matrix §2); pre-range out of `session.py:95`, HEAD out of `_normalize_system`, same status throughout | `test_a_non_mapping_body_still_answers_500` — characterization witness, green in both states, reddens if the route ever answers 200 to a body base refused | safe (pin, labeled) |
| The trio → `POST /v1/messages` | **still base's generated 500, with no upstream call** — base raises the identical `AttributeError` in its own `normalize()` (`:374`/`:402`, matrix §3), and `G-003` holds that route "no normalization changes" | `TestMessagesMalformedMemberPosture` ×3 — green as shipped; **red 500 → 502** under replay (c), executed by this pass (§D) | safe (drift guard for the rejected candidate; §RM-012 owns the posture) |
| A well-formed body → `/v1/messages` under the same harness | reaches the relay and answers its 200 — otherwise the 500s above would be a dead route, not a normalization crash | `test_a_well_formed_body_still_reaches_the_relay` → `sent.called` True, 200 | safe (the pair that makes the guard falsifiable) |
| Route digest vs the digest `/v1/messages` records under, on a nonce body | must stay equal — `EI:…/session_id/derived_after_normalize` | route oracle ×3 spellings + the pre-existing `test_a_billing_nonce_body_finds_the_stats_messages_recorded` (byte-unchanged): **both red under replay (b)** (§B) | safe (repaired property, proven able to fail) |
| A hypothetical future `normalize()` step that rewrites `system`'s text | the narrowing must not silently diverge from the full-normalize digest | pure battery, 10 shapes × 2 axes: **6 string-system cases red under replay (d)** while the block-system cases correctly stayed (the simulated step was string-only) (§C) | safe (foreclosed by a witness, not by prose) |
| `strip_billing_nonce` handed a non-dict body (`[1,2]`) | raises `AttributeError` — inherited, not added: `normalize()` raised at the same statement for the same body pre-range and at HEAD | read of `_normalize_system` (`body.get("system")`) + row `[1,2]` measured at all three points | safe (base-parity, no branch — R10) |
| Stats lookup misses after the narrowing | falls back to `estimate_tokens(body)`, which is base's own behavior on that route (matrix §2's 200s are estimates) | every clause-1/2 witness exercises exactly this arm (no stats seeded); the seeded oracle above exercises the other arm | safe (both arms covered) |
| `.text` / payload / charset surfaces, image stripping, `STRIP_IMAGES` | untouched by this increment — the image step was already inert at `session_id=""` and remains so | `normalize()` and `image.py` carry no hunk; `STRIP_IMAGES` is patched True inside the pure battery precisely to prove the gate is not what makes the digests agree | not-applicable (outside this task's touch-set) |

## Changes

**Production (2 files, +21/−8):**
- `buster_ripper/normalize.py` **+13/−0** — one new public entry immediately after the step it delegates to:
  ```python
  def strip_billing_nonce(body: dict) -> list[str]:
      """Remove the per-request billing nonce from ``system``, and touch nothing else. …"""
      return _normalize_system(body)
  ```
  Its docstring records why a caller would want the step without the rest (`session_id()` hashes `system` alone; the `tools`/`messages` steps assume every element is a mapping, which is `CR4-F-01`). `_normalize_system`, `_normalize_tools`, `_normalize_cache_control`, `_normalize_date`, `normalize()` and every line above the insertion: **byte-identical** (§E).
- `buster_ripper/routes/count_tokens.py` **+8/−8, zero net line shift** — `from ..normalize import normalize` → `… import strip_billing_nonce`; `body, _ = normalize(body, session_id="")` → `strip_billing_nonce(body)`; the comment above it rewritten (it described the image-inertness of a call that no longer happens). Superseded wording preserved here rather than stacked in the plan: *"Normalize before keying, exactly as /v1/messages does (messages.py:36-37): … The empty session_id keeps the call image-inert (normalize.py:212 gates image stripping on a non-empty one); everything else it does is dict surgery."* The claim it made — that everything else is harmless dict surgery — is what `CR4-F-01` disproved.

**Unchanged, proven by `git diff --stat` being empty (§E):** `buster_ripper/session.py`, `buster_ripper/routes/messages.py` (334), `buster_ripper/routes/chat.py` (300/300), `buster_ripper_legacy.py`, `invariants/`.

**Tests (1 new file, 281 lines):** `tests/test_count_tokens_parity.py`, four classes / 35 cases — tolerance trio (fail-first), tolerated-shape pins, the non-dict characterization, the `/v1/messages` posture guard plus its falsifying pair, the seeded route-level digest oracle, and the pure 10-shape × 2-axis digest-equality battery. Runs on `conftest.py`'s existing `proxy` fixture; reuses `tests/upstream_stub.py`; no new helper anywhere outside this module.

## Done-when clauses → the byte that meets each

1. **red-first trio** — `[…][tools-null-element]`, `[messages-null-element]`, `[messages-string-element]`; the 500s and their frames are in §A, cut at the final test code. ✅
2. **green pins, base's tolerance** — the four-case parametrized shape test, docstring naming base and `6508941` as the authority. ✅
3. **green pin, parity is a ceiling too** — `test_a_non_mapping_body_still_answers_500`, labeled characterization, and it does move: at pre-fix bytes the raise is `_normalize_system`, at base it is `_session_id`, status identical at every point. ✅
4. **red-first, the invariant direction** — digest equality pinned twice: the pure battery (10 shapes × 2 axes, the §4 battery) and the route-level oracle; plus `test_a_billing_nonce_body_finds_the_stats_messages_recorded` **unchanged and green**. Both directions executed by this pass (§B, §C). ✅
5. **red-first, the rejected candidate** — the three `/v1/messages` guards, each docstring naming base (`buster_ripper.py:374`/`:402`) and pointing at `roadmap.md` §RM-012; executed red 500 → 502 under the hardening mutation (§D). ✅
6. **byte-exactness** — §E: `session.py`/`messages.py`/`chat.py` absent from the diff, `normalize.py` carrying only the delegating entry, legacy untouched, zero `invariants/` bytes. ✅
7. **suite** — hermetic **511 passed @ 96.71%**; clause 1's witnesses demonstrated red at pre-fix bytes; coverage up, not down. ✅

## Capacity gate

Pre-edit raw sizes: `normalize.py` 228 → 241, `count_tokens.py` 106 → 106, `tests/test_count_tokens_parity.py` 0 → 281. Classification: **additive** in all three.
- `normalize.py` 228 → 241: Green (<300) at both ends; the only other band-mate, `messages.py` (334, review band), is out of the touch-set by clause 6 — this fix needed neither route file, as the blueprint's §Capacity answer states.
- `count_tokens.py` 106: Green; no function grew (statement-for-statement replacement).
- New test module 281: Green, chosen over adding ~60 lines to `tests/test_routes_integration.py` (**495**, whose next line crosses into the extract-file band) and over `tests/test_transport_parity.py` (**220**, whose subject is the transport contract, not the normalization contract). The blueprint names the second option "a new focused module" as the implementation's choice; this is it.
- Trigger check: `handle_count_tokens` is **L:18 N:1 C:3** at HEAD measured from `git show HEAD:…` (§H), i.e. one gray metric (L in 16-30), N/C/P ideal — under every trigger condition, so no extraction is owed. The work is not "growing an oversized legacy file": the file is 106 lines and stays 106.

**Capacity gate: not triggered** — measured baseline above, file sizes after, and per-function scan at both ends.

## Backlog pickup

`quality-backlog.md`, matched against this increment's touch-set:
- **absorbed and resolved: `QB:routes-count-tokens/normalize-side-effects-on-estimate-and-malformed-bodies`** (row at `count_tokens.py:87`, unchecked, opened by `083-01-yield` F-3) — its two stated consequences are exactly what this task disposes: (i) `estimate_tokens` counting the *normalized* body, and (ii) a non-mapping `messages` element raising where the route previously answered an estimate. Marked `[x]` with this iteration id and the §A/§E evidence. The row's "no change requested" was the posture *before* `CR4-F-01` was routed; this increment is the change.
- **not picked up, trigger did not fire: `QB:tests/blank-line-at-eof`** (`CR4-F-02`, disposition `accepted`, owner "whichever Forward pass next edits `tests/test_upstream_answer_relay.py` or `tests/test_dump.py`") — neither file is in this touch-set (`git status --porcelain -- tests` shows only the new module). Left unchecked, with the reason here rather than silently.
- **not picked up: `QB:lint/ruff-f-select-findings`** — its five remaining sites are `qa/tls_probe_entry.py`, `test_normalize.py`, `test_server_hooks.py`, `test_session_stats.py`, `test_tls_posture.py`; none is a file this task touches, and the row's bound is explicitly not a sweep.

## Dependency claims checked

`assumptions/claims.md` exists (runs 1-46); it lists **no row against `T019.002`**, matching the blueprint's "Assumption impact: none owed", and no row named here is shared into other tasks (`55`, `134`, `218` are `single`; `135`-`138`, `264`-`266` likewise). Claims read at the exact assertion, none contradicting the plan:
- `session_id()` derives its digest from `system` alone (`session.py:93-98`) — read; this is what makes the narrowing digest-preserving.
- `normalize()` mutates the body in place and returns `(body, changes)`; step order is image → system → tools → cache_control → date (`normalize.py:202-228`) — read; the ordering is why the blueprint's §6 masking observation holds.
- `_normalize_system` writes only `body["system"]` — read; the other three steps never assign that key (`_normalize_cache_control` only `del`s `cache_control` from `system` blocks, a key `session_id()` never reads).
- an aiohttp handler with no error middleware answers a generated 500 on an escaping exception — **observed by execution**, §A's traceback, not assumed.
- `web.Application(...)`'s default body cap was already disposed by `T018`; nothing here re-litigates it.
**No correction to record** — every claim confirmed. What Yield's recheck lane inherits mechanically: `handle_count_tokens`'s unit bytes moved (rows `55`, `134`, `218` re-pin their fingerprints) and `normalize.py`'s coordinates shift +13 from line 64 (rows `264`-`266` cite `normalize#normalize` **by symbol**, whose body is byte-identical, so they do not re-base on content).

## Concerns

1. **`normalize()`'s element loops have four read sites, not three.** `evidence/cr4-probe-matrix.txt` §6 counts three loops (`_normalize_tools`, `_strip_cache_control_from_messages`, `_normalize_date`) and that count is right; while executing replay (c) this pass found the loops are not the granularity — `_normalize_tools` reads `t.get("name", "")` **twice** (the `original_names` comprehension and the `sorted()` key), and the first hardening attempt left `{"tools":[null]}` still 500ing because the second read was missed. Recorded for the owner of `roadmap.md` §RM-012, whose future repair is under-specified by "three loops". No byte of this session's records was edited to carry it — the roadmap row is not this route's to amend, and the finding is not a defect in the shipped code (`§RM-012` is a deferral, not a plan).
2. **A recorded measure in `quality-backlog.md` is stale by one line.** The resolved `QB:routes-count-tokens/handler-lncp` row states "handler now **L:17** N:1 C:3". At HEAD the same scan says **L:18** — `T019` step 1's `body, _ = normalize(…)` added the statement, and the row was written at Checkpoint 154 before that wave. This increment is metric-neutral (§H), so nothing needs a fix; the row was already `[x]`, so this pass did not rewrite its evidence — it records the discrepancy where a later reader will look for it.
3. **Plan adherence: nothing overridden.** The two choices the blueprint left open were taken as written — the public delegating name (its "either way … the naming is not a different outcome"), and the new focused witness module (its "That choice is the implementation's"). The rejected candidate (hardening `normalize()`) was not implemented, and clause 5 now makes that a checked condition rather than a promise.

## Questions asked

**None.** The pre-question gate had nothing to escalate: the closure, the shape of the fix, the witness set and its directions are prescribed by `DC-CR4-F-01-r1`; the only open Fork remains `FK-011`, which this increment neither reaches nor affects. No escalation-floor action was encountered (no snapshot baseline, no `--snapshot-update`, no push, no history rewrite).

## Test runs

| Run | Command | Result |
|---|---|---|
| Fail-first, pre-fix bytes | `uv run pytest tests/test_count_tokens_parity.py -q --tb=short` | **3 failed / 12 passed**, exit 1 — the trio with the escaping `AttributeError` frames (§A) |
| Replays (b), (c), (d) | same module, ± `test_routes_integration.py`, each mutation applied and reverted byte-exact | **5 / 3 / 7 failed** respectively — §B, §D, §C |
| Affected scope, once, at the checkpoint boundary | `uv run pytest -q` (hermetic; `tests/qa/` excluded by `norecursedirs`) | **511 passed @ 96.71%**, exit 0 — baseline 476 @ 95.91%, +35 tests, none removed, coverage up |
| LNCP scan gate | `codify scan path <file> --git --detailed`, one file per invocation | `normalize.py`: no issues; `count_tokens.py`: the pre-existing single gray (L:18, unchanged from HEAD); test module: pytest fixture/param signatures only, ≤1 gray each |

Per `codify-test-cost` no narrow run was repeated after the affected-scope pass, and no output was piped; the full-suite and `tests/qa/` gates belong to the boundary's Yield Phase.

## For Yield

- Wave: **`T019.002` alone**, one local Yield, retry counter 0 → this is its first pass. Range: `c5e2be4…f00216c` (Checkpoint 205), the complete increment; review it as one unit, the session-artifact paths inside it are records, not code.
- Replay each direction from `## Done-when clauses`; the transcripts in [t019-002-fail-first-and-replays.txt](../evidence/t019-002-fail-first-and-replays.txt) were produced by this pass and are re-runnable, but clause-by-clause independent re-derivation is your act, not mine.
- **Do not** mistake the two green-in-both-states classes for weak witnesses: clause 2 and clause 3 are pins by design, each naming the build that reddens it (over-reach and acceptance of a body base refused); the same holds for `test_a_well_formed_body_still_reaches_the_relay`, which exists solely so the posture guard cannot pass on a dead route.
- `T019.002` is `implemented`, **not** `resolved`: after your pass comes the MS-001 boundary's assumptions recheck on these bytes (`S16`-`S19`'s one recomposition, plus the row-57 re-pin), then Yield Phase re-executing `QAP-MS001` on final bytes, then Yield Red, which owns `resolved` for `T015`-`T020`, `T019.001` and this.

## Next

No further task on this route. After local acceptance: the assumptions recheck (the `S16`-`S19` recompositions must run on these final bytes — `count_tokens.py` coordinates move, which is the one re-base the blueprint permits) → Yield Phase → Yield Red → `phase-boundary-finalize`.

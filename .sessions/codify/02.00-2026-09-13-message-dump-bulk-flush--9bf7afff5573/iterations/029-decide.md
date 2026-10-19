# 029 — decide (RED-FINDINGS mode — 028's AMBIGUOUS row + FN-1)

## Orchestrator Prompt (verbatim)

> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
>
> Mode: red-findings
> Red report: /Users/alexhubitski/Projects/buster-ripper/.sessions/codify/02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/iterations/028-forward-red.md
>
> Disposition the report's rows: 1 AMBIGUOUS row (T006 serve-script clause × S6 planned chain — missing guard expression for SKIP_TLS_VERIFY) and 1 discretion footnote (FN-1, README citation drift on the T005 amended clause).

## Mode and scope

`Mode: red-findings`. Read per the clean-room rule: the named Red report (`028-forward-red.md`), the forward-visible artifacts it attacks (`plan-summary.md` Anchor Map + the amended rows; `handoff-context.md` §Current Status/§Notes/§Planned Task Context/§Deterministic Closure Ledger/§Composition Routing), and the source needed to verify its rows — `serve_buster_ripper.sh` (whole, 40 lines), `README.md:130-154`, `assumptions/compositions/seam-S6.md` (whole, the recomposed run-2 artifact the re-attack was dispatched for), G-007's rows (plan-summary :72, :88), and the registered invariant (`invariants/buster_ripper/server/invariants.md`, `PI:T006/upstream_tls/verification_on_unless_explicitly_disabled`). No prior iteration report reopened; no settled section re-decided; the `codify-deterministic-closure` gate applied to both rows before any disposition.

**Ownership note (skill constraint, recorded deliberately).** `red-findings` mode states: *"Do not edit the approved plan or invariant registry in this mode… Interview applies approved plan repairs and is the sole writer for every invariant-registry change."* Both dispositions below are therefore **authorized exactly, not applied**: they are handed to `interview` as one batch (§"Repairs authorized for `interview`"), and only this report and the compact handoff index are written by this pass. Nothing here re-decides a design: both repairs restate properties the approved artifacts already settle (the `"0"` default and its posture consequence; the README line that carries the run command).

## Verification of the row premises (this pass)

1. **The guard-shape gap is real.** T006's serve-script clause (plan-summary :211) fixes `SKIP_TLS_VERIFY=${SKIP_TLS_VERIFY:-0}` and the `--skip-tls-verify` mapping and names **no guard**. The script's convention splits by default form: the two `0`-defaulted posture peers guard on value equality (`STRIP_DATE` :23→:29, `VERBOSE` :24→:30), the `1`-defaulted `PRESERVED_THINKING` shares the value-equality shape (:26→:32), and the only presence-guarded var has a **non-empty** default (`DUMP_DIR` :25→:31, where `-n` distinguishes *set-empty-disable* from *unset*). Under `[[ -n "${SKIP_TLS_VERIFY}" ]]` with the settled `"0"` default the guard is true on every run: `--skip-tls-verify` appended unconditionally, verification OFF with no opt-out.
2. **The unsafe reading is plan-text-backed, as Red claims.** The amended T005 clause (:197) names `[[ -n "${DUMP_FORMAT}" ]]` inline and ties T006 to "this same script on exactly this reasoning"; the recomposed seam-S6 dimension 1 (:20) records "the planned `DUMP_FORMAT`/`SKIP_TLS_VERIFY` peers compose onto them without conflict" under the empty-default/`-n` heading, and its planned-peer paragraph (:14) asserts "unset/empty → guard false → no flag appended → shipped defaults preserved" for **both** peers. That is false for `SKIP_TLS_VERIFY`: its default is `"0"`, never unset-empty. The two forward-visible surfaces cannot both be right.
3. **The violated properties are registered.** `PI:T006/upstream_tls/verification_on_unless_explicitly_disabled` (protected, planned; "the upstream's certificate is still verified by default… the opt-out is never inferred from the URL scheme, the host or the environment", invariants/buster_ripper/server/invariants.md:47-50) and G-007's Met-when ("The opt-out defaults OFF and is never inferred from the URL scheme", plan-summary :72). US-011 verbatim: *"We should keep security strong."* No VP observes the serve script (structural by plan decision, T005 :197; `US-010`'s harness starts the binary directly), so no gate would catch the wrong shape.
4. **Second carriers of the same misreading, found while verifying the repair surfaces** (the R15 lesson applied at authorization time): the DC-claim-32-r1 ledger entry's Required-behavior sentence (handoff-context :341) describes **both** wiring clauses as "empty-default guarded" — wrong for `SKIP_TLS_VERIFY`'s `"0"` default — and seam-S6 carries two more instances beyond :20 (the :14 planned-peer paragraph and :16's "the same guard shape" systematic-check phrase). All are the same planned-chain misreading Red's row 1 dispositions; the batch below covers every carrier so no surface survives carrying it.
5. **FN-1 verified.** `README.md:140` is the run command (`DUMP_DIR=~/mllm/prompt-diffs bash serve_buster_ripper.sh`); :143 is the Session-ID sentence. T005's amended clause (:197) cites :143. Corroborated by seam-S6's own claim-33 read ("sole run command at line 140", :8/:24/:32). One-line drift, no second surface carries `README.md:143` (grepped across plan-summary, handoff, qa-plan, roadmap).

## Closure packets

### Packet 1 — Row 1 (T006 serve-script guard shape × S6 planned chain)

```text
CLOSURE: absorbed
Finding: Row 1 (iterations/028-forward-red.md — T006 :211 × seam-S6 dimension 1)
Required outcome: `--skip-tls-verify` reaches argv through the `EXTRA_FLAGS` channel iff the user
  explicitly sets `SKIP_TLS_VERIFY=1`; unset, set-empty or `"0"` → no flag → the upstream certificate
  stays verified. `PI:T006/upstream_tls/verification_on_unless_explicitly_disabled` and G-007's
  Met-when ("The opt-out defaults OFF") must hold on the script path.
Authority:
  - T006's settled default form `SKIP_TLS_VERIFY=${SKIP_TLS_VERIFY:-0}` (plan-summary.md:211; the
    017-settled token block, Anchor Map :122) — approved task scope
  - G-007 Met-when: "The opt-out defaults OFF and is never inferred from the URL scheme"
    (plan-summary.md:72) — approved goal
  - PI:T006/upstream_tls/verification_on_unless_explicitly_disabled (protected, planned;
    invariants/buster_ripper/server/invariants.md:47-50) — active planned invariant
  - Verified script convention: value-equality guards for value-defaulted vars
    (serve_buster_ripper.sh :23→:29, :24→:30, :26→:32); presence guard only for the non-empty-defaulted
    var (:25→:31)
  - US-011 verbatim: "We should keep security strong."
Admissible candidates:
  (a) name the boolean peer guard in T006's clause (with the explicit anti-pattern warning) and align
      the S6 planned-peer description + the DC-claim-32-r1 sentence to the per-peer split
Rejected candidates:
  - presence guard `[[ -n … ]]` with the settled `"0"` default -> true on every run, flag appended
    unconditionally: preserves the verified defect (closure step 3)
  - presence guard with a rewritten empty default -> rewrites the settled default form (T006 :211,
    017's settled token) and still mishandles the explicit-off case (`SKIP_TLS_VERIFY=0` is non-empty
    → guard true → verification off despite an explicit off); strictly worse on correctness than (a)
  - leave T006 guard-silent (rely on Forward's discretion) -> the freshest in-plan precedent (T005
    :197 names `[[ -n … ]]` inline and ties T006 to "exactly this reasoning") plus seam-S6's dimension-1
    record make the unsafe reading plan-text-backed; leaving a load-bearing expression implicit
    preserves the ambiguity that produces the defect
  - add a VP observing the serve script -> re-opens the plan's settled structural/no-VP decision
    (T005 :197; §Behavior Disposition Matrix) and broadens scope beyond the smaller complete
    remediation; the defect is prevented by the plan saying the one thing the implementer must not
    get wrong
  - user-fork (boolean vs presence to the user) -> not two materially different admissible outcomes:
    only (a) preserves the settled default and the invariant; the presence shape is a different
    (broken) outcome, not an alternative. Different guard syntax alone is never a different
    user-visible outcome (closure step 3, mechanism-collapse rule)
Selected closure: (a). T006's clause names the boolean peer guard and warns against the presence
  shape; seam-S6's planned-peer sentences (:14, :16, :20) and the DC-claim-32-r1 required-behavior
  sentence (handoff :341) are aligned to the per-peer split: `DUMP_FORMAT` presence/empty-default
  (T005), `SKIP_TLS_VERIFY` boolean/`"0"`-default (T006).
Risk shape: unchanged — no behavior, referent, Anchor Map row or gate changes; the repair makes the
  settled posture explicit where the plan was silent and its freshest precedents pointed at the
  wrong shape. seam-S6's SAFE verdict survives the alignment: both guard halves are verified present
  in the script (presence at :31, boolean at :29/:30/:32) and unset → guard-false holds for each
  peer under its own shape, so the dimension-1 trace still composes — only its description changes.
Deferred additive work: none
Owning route: interview (plan + session-artifact repair, one batch) → forward T006 (implements the
  boolean guard)
Question required: no
```

**Disposition: `deterministic-repair`.** Not `plan-break` — the selected approach (T006 as one task, the token, the default form, the two-channel wiring) is untouched; the repair adds the one missing expression to a clause that already carries the other two halves. Not `invalid` — the approved artifacts do not exclude the `-n` reading; they contain two surfaces that actively suggest it. Not `user-fork` — no admissible outcome pair remains unranked.

### Packet 2 — FN-1 (T005's README citation drift)

```text
CLOSURE: absorbed
Finding: FN-1 (iterations/028-forward-red.md footnote — T005 :197 cites README.md:143)
Required outcome: the clause's citation names the line that carries the documented run command.
Authority: verified file evidence — README.md:140 is the run command
  (`DUMP_DIR=~/mllm/prompt-diffs bash serve_buster_ripper.sh`); :143 is the Session-ID sentence;
  corroborated by seam-S6's claim-33 read (:8/:24/:32, "sole run command at line 140").
Admissible candidates: correct `:143` → `:140`.
Rejected candidates:
  - leave the drift -> preserves a verified factual error in an approved clause (closure step 3); a
    citation is a reference, not a binding, so no alternative reading exists to weigh
Selected closure: replace `README.md:143` with `README.md:140` in T005's 022 ADDED clause.
Risk shape: unchanged — one-line citation correction; no behavior, binding or gate touched.
Deferred additive work: none
Owning route: interview (plan repair)
Question required: no
```

**Disposition: `deterministic-repair`.** Red's recommended reading is adopted because it is the verified fact, not because Red preferred it — the closure gate selects it with no competitor.

## Disposition table

| # | Surface | Red verdict | Disposition | Closure | Plan-surface change |
|---|---|---|---|---|---|
| 1 | T006 :211 × seam-S6 dimension 1 | AMBIGUOUS | **deterministic-repair** | absorbed | no (no gate observes a different thing; no task moves; no dependency edge changes — task-clause precision, the T-2 class of 025) |
| 2 | T005 :197 (`README.md:143`) | Footnote FN-1 | **deterministic-repair** | absorbed | no (citation accuracy only) |

`DECIDE-RED` arithmetic: 2 rows | 0 invalid | 2 deterministic | 0 user forks | 0 plan breaks.

## Repairs authorized for `interview` (one batch — apply exactly these, nothing wider)

**R-A — `plan-summary.md` §Task List, `T006`, the serve-script ADDED clause (:211).** After "mapped to `--skip-tls-verify`." insert:

> The guard is the **boolean** peer shape — `[[ "${SKIP_TLS_VERIFY}" == "1" ]] && EXTRA_FLAGS="${EXTRA_FLAGS} --skip-tls-verify"` — the shape the script already uses for value-guarded flags (`STRIP_DATE` :23→:29 and `VERBOSE` :24→:30, the two `0`-defaulted posture peers; `PRESERVED_THINKING` :26→:32 shares the guard shape with a `1` default). **The `-n` presence shape must NOT be used here**, despite the `DUMP_FORMAT` clause (T005) and the recomposed S6 seam's planned-peer description spotlighting it: `DUMP_FORMAT`'s default is empty, so its presence guard is false by default, but this var's settled default is `"0"` — non-empty — so `[[ -n "${SKIP_TLS_VERIFY}" ]]` is true on every run, `--skip-tls-verify` is appended unconditionally, and the default posture inverts (verification off with no opt-out), violating `PI:T006/upstream_tls/verification_on_unless_explicitly_disabled` and `G-007`'s Met-when ("The opt-out defaults OFF") on a surface no VP observes (the serve-script rows are structural by plan decision).

**R-B — `assumptions/compositions/seam-S6.md`, the planned-peer sentences (three exact replacements; VERDICT/Invalidates-On/fingerprints untouched — the alignment is prose-only and the verdict stays SAFE under the corrected split, see Packet 1's risk shape).**
1. :14, last sentence: "As planned properties they reuse both verified halves of the peer shape — the `${VAR:-}` default-assignment form (lines 20-26) and the guarded conditional append into the `EXTRA_FLAGS` channel (lines 28-32) — so unset/empty → guard false → no flag appended → shipped defaults preserved; set → exactly one flag appended through channel 2." → "As planned properties they reuse both verified halves of the peer shape — the `${VAR:-}` default-assignment form (lines 20-26) and the guarded conditional append into the `EXTRA_FLAGS` channel (lines 28-32) — under **per-peer guard shapes**: `DUMP_FORMAT`'s default is empty, so its presence guard (`[[ -n "${DUMP_FORMAT}" ]]`, T005's clause) is false by default; `SKIP_TLS_VERIFY`'s default is `"0"`, so it is guarded on its value (`[[ "${SKIP_TLS_VERIFY}" == "1" ]]`, T006's clause — a `-n` guard would hold on every run and append the flag unconditionally, inverting the default posture). Unset/empty/`"0"` → guard false → no flag appended → shipped defaults preserved; set per each var's clause → exactly one flag appended through channel 2."
2. :16: "(all three boundary vars ride the same `EXTRA_FLAGS` channel and the same guard shape)" → "(all three boundary vars ride the same `EXTRA_FLAGS` channel; guard shapes differ per peer — presence for `DUMP_DIR` and `DUMP_FORMAT`, value-`"1"` for `SKIP_TLS_VERIFY`)"
3. :20: "the planned `DUMP_FORMAT`/`SKIP_TLS_VERIFY` peers compose onto them without conflict." → "the planned `DUMP_FORMAT` peer composes onto the empty-default/`-n` shape without conflict, and the planned `SKIP_TLS_VERIFY` peer composes onto the boolean `== "1"` shape (lines 29/30/32) — its `"0"` default makes the `-n` shape unsafe (T006's guard clause)."

**R-C — `handoff-context.md`, the DC-claim-32-r1 ledger entry's Required-behavior sentence (:341, second carrier of the same misreading).** "each rides `EXTRA_FLAGS` beside the `DUMP_DIR` peer (the established shape), empty-default guarded, one flag per env var." → "each rides `EXTRA_FLAGS` beside the `DUMP_DIR` peer (the established shape), guarded per peer — `DUMP_FORMAT` empty-default/`-n`, `SKIP_TLS_VERIFY` value-`"1"`/boolean (its default is `"0"`, not empty, so a `-n` guard would hold on every run) — one flag per env var."

**R-D — `plan-summary.md` §Task List, `T005`, the 022 ADDED clause (:197).** `README.md:143` → `README.md:140`. (No other surface carries the `:143` citation — verified by grep across plan-summary, handoff, qa-plan, roadmap.)

No invariant-registry change is owed or authorized: `PI:T006/upstream_tls/verification_on_unless_explicitly_disabled` is what the repair **enforces**; its statement is not edited, and no `OI:`/`PI:` entry is touched.

## Compaction Decision

- Prior same-phase files reviewed: `022-decide.md`, `025-decide.md`, `027-decide.md` (the decide files of the current planning cycle; earlier decide files belong to closed planning cycles).
- Superseded: **none.**
- Not superseded: **all three.** 022 closed the decide-completeness repair (different slice); 025 closed 024's eight goal/task rows and its repairs are landed (`R1`–`R15`, commits `5864051`/`6700914`/`75924f1`); 027 was the R16 cleanup pass. This run dispositions 028's one AMBIGUOUS row and one footnote — a slice none of them touches.
- Compaction performed: **no.**
- If no: `red-findings` mode explicitly forbids compacting prior Decide reports ("do not edit `plan-summary.md` or compact prior Decide reports"), and no prior file holds unresolved option or uncertainty state duplicated by this pass.

## Deterministic Closure Ledger entries

The session's §Deterministic Closure Ledger holds the assumption-routing `DC-*-r1` entries; this pass adds no new DC-claim entry (zero claims were re-verified or re-routed). The two closures above are the ledger content for this pass; `interview` lands them with the repairs in the same commit batch, per the 025 precedent (`R14`).

| # | Finding | Closure | Selected remediation | Plan-surface change | Question |
|---|---|---|---|---|---|
| 1 | 028 row 1 (T006 guard shape × S6) | absorbed | R-A + R-B + R-C (boolean guard named; per-peer split in seam-S6 and the claim-32 ledger sentence) | no | no |
| 2 | 028 FN-1 (README:143) | absorbed | R-D (`:143` → `:140`) | no | no |

## Assumptions made in this pass

1. **The boolean guard `== "1"` is the value-guard convention for `"0"`-defaulted vars in this script** — verified against the file, not assumed: `STRIP_DATE` and `VERBOSE` are the exact posture peers (default `0`, guard `== "1"`). `PRESERVED_THINKING` shares the guard shape but defaults `1` (enable-by-default), so it is cited as a shape peer, not a posture peer — Red's row-1 detail listed it among "the same `0` default" peers, which is imprecise; the authorized R-A text corrects that nuance.
2. **Editing seam-S6's prose does not invalidate its SAFE verdict or trigger re-composition** — grounded: the verdict's six dimensions trace verified mechanism facts plus planned properties; under the corrected per-peer split every dimension still composes (both guard shapes are verified present in the script), so only the description changes. The post-implementation recomposition at Yield is unchanged (`Invalidates-On` :44 already names it).
3. **The DC-claim-32-r1 ledger sentence (:341) is a repair surface, not settled authority overriding the plan** — the ledger records the assumption-routing closure; its Required-behavior sentence paraphrases the wiring clauses, and the paraphrase is wrong for `SKIP_TLS_VERIFY` (the settled default is `"0"`, never empty). The plan text is the authority; the ledger sentence is aligned to it.
4. **No VP is owed for the serve script** — the plan's settled structural/no-VP decision stands; the repair prevents the defect at the plan-text layer, which is where Red located the gap.

## No row requires the user

Both closures resolve by repairs that are **reversible in one commit and lose no data**, and neither touches a recorded user decision: the token, the default form (`:-0`), the posture and G-007 are all user-settled and stand unchanged — the repair **enforces** the user's explicit words (*"We should keep security strong"*, *"The opt-out defaults OFF"*) against the one reading that would defeat them. `US-013`'s delegation is in force; the escalation floor fits no clause (no public-surface change, no irreversible act, no data loss). **No `FK` was reserved, no question was asked, and none is owed.** After `interview` applies R-A…R-D, `forward-red` is re-dispatched scoped to the repaired artifacts; only `0 AMBIGUOUS` then leaves the chain at the assumption gate's continuation → first `forward`.

**DECIDE-RED: 2 rows | 0 invalid | 2 deterministic | 0 user forks | 0 plan breaks**

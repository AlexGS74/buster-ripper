# 027 — decide (RED-FINDINGS, follow-on to the scoped re-run)

## Origin

`026-forward-red.md` — the one permitted scoped re-run, dispatched against the eight rows 025
closed and round 19 applied — returned **`0 AMBIGUOUS`**. It verified all eight against the text on
disk after the repairs (including `R15`, which landed while its dispatch was in flight) and closed the
`qa-plan.md:48` cell it had been given to arbitrate.

It left **one footnote**, and `interview` (round 19d) flagged the same item while applying nothing to
it. This report disposes that footnote. No other row is open: 026's verdict table carries nine rows, all
excluded, and its own routing line is the only thing that still said "route G-3 to Decide" — superseded by
`R15` within the same report.

## `R15` — recorded here for the trace

`R15` was authorized in round 19c from a message, not from an iteration report, so its authorization is
recorded here rather than left only in a handoff bullet.

- **Surface**: `plan-summary.md:240`, `VP-T002-write-count-reduced`, `Observable behavior`.
- **Was**: *"**The pair of runs (documented vs collapsed at the same N) IS the comparison `G-008`(b)
  requires (018)**"* → **now**: *"… is the measurement `G-008`(b) requires (018), and the `MS-002` Yield
  Phase records it as the durable comparison record"*.
- **Why it was a repair and not a widening**: `R6`'s authorization (025) named that exact sentence and
  reworded its `T004` copy. The VP row holds the same sentence verbatim apart from the `(018)` tag, so
  repairing it is `R6` reaching its **second surface** — the same class as the QA Gate Matrix cell being
  `R3`'s second surface. The 025 note that called it "outside the authorized scope" was a misread of `R6`,
  not a scope boundary; it is corrected in place in `handoff-context.md` §Notes and in `020-interview.md`
  §Round 19.
- **Plan-surface**: none owed. `R6` already edited the same sentence in the same file, so this extends
  `G-3`'s recorded plan-surface entry. Applied `75924f1` (Checkpoint 54).

## F-1 → deterministic-repair (`R16`)

```
CLOSURE: absorbed
Finding: F-1 (026-forward-red.md §Footnotes) — `plan-summary.md:21` and `handoff-context.md:197`
  both restate "F-21 `T001`'s append is format-conditioned", while `T001`'s entry (`:146`) reads
  "in this increment the append is unconditional — no trigger exists yet, so nothing is drained; `T002`
  adds the condition that makes it collapsed-path-only".
Required outcome: no surface asserts the pre-`R9` reading of where the append condition lives.
Authority: US-013 (a delegation: one line, reversible in one commit, no data lost); `R9`'s authorization
  (025) named the two task entries, and these are the same statement on two further surfaces; 026's own
  arbitration invitation ("if Decide prefers symmetry with G-3's surviving surface, the fix is the same one-line
  class") — the choice is Decide's by that report's own routing.
Admissible candidates:
  (a) Leave both lines as Forward discretion, per 026's default recommendation.
  (b) Mark `F-21`'s disposition in place on both surfaces, matching the style of its neighbours in the
      same list (`F-1` carries "narrowed in 015"; `F-2` carries "corrected to DUMP_FLUSH_BYTES").
Rejected candidates:
  (a) rejected. 026's reconciliation premise — "both readings reconcile to the same final code" — is true of
      the endpoint and false of the increment, and the increment is the whole of T-1: implementing the condition
      in `T001` requires reading `config.DUMP_FORMAT`, which `T002` creates, which is precisely the dependency
      cycle T-1 was raised about. It is also the class this pass has now repaired twice (R3's cell, R6's VP row),
      so leaving it would preserve the one asymmetry the round-19 correction just removed.
  Deleting the line is not admissible: `F-21` is a real finding and the list is a findings index.
Selected closure: (b) — append `F-21`'s disposition to both lines so each reads as finding + correction, the
  form `F-1` and `F-2` already use in the same list. `plan-summary.md:21` is the approved plan and is applied
  by `interview` as `R16`; `handoff-context.md:197` is Decide's own handoff index and is edited in this pass.
Risk shape: editorial only. No gate, validation point, task boundary, invariant, `Anchor Map` binding or
  registry entry changes, so no plan-surface change is recorded and no `FK` is owed.
Deferred additive work: none.
Owning route: `interview` applies `R16` to `plan-summary.md:21`. No registry change owed.
Question required: no.
```

### `R16` — exact edit, for `interview`

`plan-summary.md:21`, in the "Twelve findings" list, replace:

> **F-21** `T001`'s append is format-conditioned.

with:

> **F-21** `T001`'s append is format-conditioned — **corrected** in 025: the append is unconditional in `T001`'s own increment, and the condition is `T002`'s.

### `R16` — the same edit, applied here

`handoff-context.md:197`, in the "Findings F-1…F-12 all landed" bullet, the same fragment is edited
to the same effect in this pass. Decide owns the handoff index; no application hop is owed for that surface.

## Plan-surface changes

None. `R15` extends `G-3`'s recorded plan-surface entry (already recorded at 025); `R16` is editorial
and changes what no gate can observe.

## Compaction decision

None. No prior Decide report is superseded by this pass.

## No row requires the user

Both items are one-line, reversible, and lose no data, so `US-013` delegates them to the agents. Neither
contradicts a recorded user decision nor touches a protected invariant: `US-012`'s reading and `US-014`'s stamps
are untouched, and no invariant registry entry is affected.

**DECIDE-RED: 1 row | 0 invalid | 1 deterministic | 0 user forks | 0 plan breaks**

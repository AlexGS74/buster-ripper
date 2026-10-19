# 040 — decide (scoped workflow-contract incompatibility)

## Orchestrator Prompt (verbatim)

> You are a subagent spawned by another session.
> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
> Mode: scoped workflow-contract incompatibility
> Source: interview-finalize FINALIZATION-BLOCKED and its CLOSURE: breaks-it packet concerning US-015 historical final approval, required resolved FK, and three milestone-arbiter PENDING factual claims (US-009 c2, FK-008 c1/c2). Determine authorized route without fabricating verdicts, changing user approval, or reopening unrelated product decisions.

## Bounded finding

This is a process-contract incompatibility, not an unsettled product choice. `US-015` preserves the actual round-18 final-approval presentation and the user's exact answer, `"approve, begin"` (`decisions.md:1194-1232`; `iterations/020-interview.md:425-441`). It is a recorded user statement, **not** a resolved FK. The current Finalize precondition requires a resolved final-approval FK and terminal supported verification, exact evidence, and resolved consequences for every factual claim (`interview-interactive/references/finalize.md:3-20`). Three claims remain assigned to later milestone arbiters: `US-009` claim 2 to `yield-phase:MS-001`; `FK-008` claims 1–2 to `yield-phase:MS-004` (`decisions.md:791-798,960-974`; `handoff-context.md:692-696`). They cannot truthfully receive terminal verdicts before those gates. MS-004 in particular depends on the planned T006 implementation. The old round-18 gate explicitly reported the PENDING conflict rather than curing it (`iterations/020-interview.md:506-519`). Later assumption rows have since been reconciled; these three are the bounded residue.

The finalization blocker cannot be discharged by relabeling `US-015` as an FK, silently treating PENDING as HOLDS, changing the verified owners to `assumptions`, substituting a stub for the settled remote QA upstream, or asking the user to attest to empirical milestone behavior. A new prospective final-approval FK, if demanded by the current protocol after registration, requires a real Sprint presentation and a fresh user answer; it cannot be marked resolved from the historical answer. That would be a new approval act, not a repair of the 2026-09-17 record.

## Deterministic closure packet

```text
CLOSURE: breaks-it
Finding: interview-finalize FINALIZATION-BLOCKED — legacy final approval and milestone-owned PENDING claims
Required outcome: preserve the historical user authorization exactly, keep factual claims owned by their named arbiters until evidence exists, and reach a truthful planning/finalization status without a dependency cycle.
Authority: decisions.md US-015 and FK-008/US-009; iterations/020-interview.md round 18 and §Gates Found Unmet; handoff-context.md §Canonical Verification Reconciliation; interview-interactive/references/finalize.md Preconditions; interview-interactive/SKILL.md Shared Authority Contract and final-approval wave rule.
Admissible candidates: none within the current Finalize preconditions and the already-executing approved plan. A versioned, explicitly authorized workflow-contract accommodation may preserve the legacy approval as historical authority and let milestone-owned claims remain PENDING until their arbiters; alternatively the process owner may require a prospective new final-approval wave after the necessary evidence exists. Neither accommodation is currently enacted or a license for Decide to mark FINALIZED.
Rejected candidates: migrate US-015 into a resolved FK without a real FK presentation/resolution -> fabricates provenance; write HOLDS/evidence/consequence for the three claims now -> fabricates verdicts; run the milestone Yield Phases before their implementation/gates -> circular dependency; change the product plan or QA upstream -> reopens settled authority without cause; ask the user to verify runtime facts -> wrong owner; mark finalization complete despite unmet preconditions -> false gate result.
Selected closure: none under the existing contract; escalate the smallest workflow-contract slice to the CODIFY process owner, leaving the product plan and all original decision records unchanged. Finalize remains blocked under its current rules.
Risk shape: unchanged product risk; a process exception without explicit authority would increase audit/provenance risk.
Deferred additive work: milestone arbiters retain US-009 c2 at MS-001 and FK-008 c1/c2 at MS-004; exact evidence and consequences are recorded then. The process owner must settle the legacy-approval/stage-order compatibility before a new FINALIZED assertion.
Owning route: CODIFY workflow-contract owner for bounded protocol repair or explicit exception; then interview-finalize reruns its preconditions. If a prospective new approval is required, Decide authors a new FK after Registrar has landed relevant impacts, and Sprint/Reconcile obtains the user's actual answer; never resolve it from US-015.
Question required: no product question; no user-facing option set at this stage.
```

## Route and boundaries

Do not undo the round-18 approval or claim it never happened. Do not mark the current Finalize pass successful. The workflow owner must reconcile its universal claim-terminal requirement with milestone-dependent facts, either by an explicit staged precondition that recognizes named, not-yet-due milestone claims, or by moving finalization to a point after those milestones without making implementation wait on itself. Separately it must specify how a legacy exact approval preserved as `US-015` is recognized, or direct a genuine successor approval wave; no record transformation by implication. These are process-contract alternatives, not user choices about dump format, TLS, QA endpoint, or the approved six-task plan. Any rule change must be recorded in its owning workflow source and then independently checked before Finalize resumes. This report does not edit that source, the product plan, decisions ledger, or registry.

## Compaction Decision

- Prior same-phase files reviewed: 003, 008, 010, 011, 012, 014, 015, 017, 019, 022, 025, 027, 029, 037 decide reports.
- Superseded: none.
- Not superseded: all; this is a distinct finalization-protocol slice.
- Compaction performed: no — the prior reports retain their separate causal and product-plan provenance.

## Assumptions and limitations

- The dispatcher-provided `FINALIZATION-BLOCKED`/`CLOSURE: breaks-it` packet is the trigger; its report path was not supplied and no such report was located in the session directory. The underlying conflict is independently visible in the cited canonical records and current Finalize rule.
- This report makes no factual assertion that the remote QA host is reachable or that either TLS posture has passed its gate.
- Implementation status and existing approved behavior are not re-adjudicated here. The legacy round-18 gate's historical outcome is preserved as a historical record, while a new pass under the current contract remains blocked.

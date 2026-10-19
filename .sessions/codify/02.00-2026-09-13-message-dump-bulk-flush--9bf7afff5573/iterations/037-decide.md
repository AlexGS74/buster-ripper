# 037 — decide (scoped registry-binding repair)

## Orchestrator Prompt (verbatim)

> You are a subagent spawned by another session.
> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
> Mode: scoped registry-binding repair
> Source: authority-registrar REGISTRATION-BLOCKED on composite bindings for E:T002/dump/dump-format-selector, E:T002/dump/dump-summary-line, E:T003/dump/flush-trigger-parameters. Classify deterministic binding repairs versus material decisions, preserving approved referents. Do not reopen unrelated plan choices.

## Bounded finding

`python3 /Users/alexhubitski/.dotfiles/.agents/skills/speeckle/scripts/rebuild_index.py invariants --check` reports three dangling bindings, and a stale index. The three are syntactically composite `Binding:` values in `invariants/buster_ripper/dump/invariants.md:51-87`. Speeckle Entity grammar permits exactly one `path:symbol` OR one bare path OR one registered invariant handle. It does not permit comma-separated symbols or a `+` joining files. The three composite names and their semantics are approved in `plan-summary.md:110-112`; this is a registry representation fault, not a reason to change the product contracts. Current symbols exist at `config.py:23-24,32-33`, `main.py:12-26,45-58`, and `dump.py:225-237`.

## Deterministic closure packets

### E:T002/dump/dump-format-selector

```text
CLOSURE: absorbed
Finding: registrar REGISTRATION-BLOCKED / E:T002/dump/dump-format-selector
Required outcome: one legal, existing binding for the same three-value output-shape selector; its value set and CLI injection remain documented.
Authority: plan-summary.md:111,143-145; invariants/buster_ripper/dump/invariants.md:51-55; Speeckle Entity one-binding grammar; buster_ripper/config.py:23-24 and main.py:12-16,45-49.
Admissible candidates: bind `buster_ripper/config.py:DUMP_FORMAT` as the selector's process-wide current value; retain `DUMP_FORMATS` and `--dump-format` in Gloss/Confusable (and Source if needed).
Rejected candidates: composite expression -> invalid grammar/dangling; new selector symbol -> unnecessary code change; splitting the approved selector into separate referents -> changes the Anchor Map's term identity; binding only `main.py:--dump-format` -> CLI option text is not a symbol and misses the current-value owner.
Selected closure: replace only `Binding:` with `buster_ripper/config.py:DUMP_FORMAT`; keep handle and full gloss.
Risk shape: unchanged behavior; narrower machine binding, with companion locations retained descriptively.
Deferred additive work: none.
Owning route: Authority Registrar, registry/index rebuild.
Question required: no
```

### E:T002/dump/dump-summary-line

```text
CLOSURE: absorbed
Finding: registrar REGISTRATION-BLOCKED / E:T002/dump/dump-summary-line
Required outcome: same encoding-independent two-form summary construction for turn zero and later turns, without inventing a single owner symbol.
Authority: plan-summary.md:112 (pre-existing inline role, not Mode: NEW); invariants/buster_ripper/dump/invariants.md:59-63; Speeckle permits bare path; buster_ripper/dump.py:225-237.
Admissible candidates: bare `buster_ripper/dump.py`, with `_full_summary` and `_diff_summary` named in Gloss as the two implementations of this single approved role.
Rejected candidates: composite symbols -> invalid grammar; either helper alone -> erases one of the two required forms; new `_summary_line` dispatch helper -> code/plan change for a registry syntax defect; separate entity handles -> unnecessary referent split.
Selected closure: replace only `Binding:` with `buster_ripper/dump.py`; retain the two helper names in Gloss and the existing Confusable text.
Risk shape: unchanged behavior; file-level binding is coarser than symbol-level but the one-line gloss identifies the pair unambiguously.
Deferred additive work: none.
Owning route: Authority Registrar, registry/index rebuild.
Question required: no
```

### E:T003/dump/flush-trigger-parameters

```text
CLOSURE: absorbed
Finding: registrar REGISTRATION-BLOCKED / E:T003/dump/flush-trigger-parameters
Required outcome: the same two distinct process-start-configured trigger parameters and flags, 10 s and 4 MiB, used only by the collapsed path.
Authority: plan-summary.md:110,163-166; invariants/buster_ripper/dump/invariants.md:83-87; Speeckle permits bare path; buster_ripper/config.py:32-33 and main.py:18-28,51-58.
Admissible candidates: bare `buster_ripper/config.py`, with both `DUMP_FLUSH_INTERVAL` and `DUMP_FLUSH_BYTES` plus both CLI flags named in Gloss.
Rejected candidates: composite expression -> invalid grammar; binding to either one global alone -> loses the other trigger; new aggregate config object or separate canonical entities -> unnecessary design/identity change.
Selected closure: replace only `Binding:` with `buster_ripper/config.py`; retain the two globals and two flags in Gloss and existing Confusable text.
Risk shape: unchanged behavior; path-level binding is coarser but points to the shared configuration owner and is unique against the selector's symbol-level binding.
Deferred additive work: none.
Owning route: Authority Registrar, registry/index rebuild.
Question required: no
```

## Authorized repair and boundaries

Registrar may apply the three exact `Binding:` replacements above, preserve each original handle, Source, Confusable and substantive Gloss (enrich the Gloss with companion symbol names where necessary), then regenerate `invariants/index.md` and run `rebuild_index.py invariants --check`. Do not alter `plan-summary.md`, decisions, production code or invariant statements. Neither a product-level fork nor a new task is warranted. The stale-index diagnostic may also reflect concurrent assumption promotions; rebuild only after those registry writes settle. These proposed bindings are pairwise distinct and distinct from other entity bindings in the dump registry.

## Compaction Decision

- Prior same-phase files reviewed: 003, 008, 010, 011, 012, 014, 015, 017, 019, 022, 025, 027, 029 decide reports (scope checked by filenames and 029's bounded reroute).
- Superseded: none.
- Not superseded: all above; this is a different, post-implementation registry-binding slice.
- Compaction performed: no.
- If no: no prior decide report addresses the present three invalid composite Entity bindings.

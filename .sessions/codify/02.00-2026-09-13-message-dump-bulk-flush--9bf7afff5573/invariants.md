# Invariants — 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573

> Squashed from a session completed 2026-10-18; PR #1 merged 2026-10-19.
> Load-bearing logic only. Incidental behavior is intentionally excluded.
> Reference policy: session/task/iteration anchors are the durable source for post-squash lookup; code paths name the implementation seam; pre-squash commit hashes are advisory only.

## Existing Invariants Preserved

### EI:buster_ripper.dump/dump_turn/output_layout
**Statement**: With the `documented` format selected — or nothing configured — a session directory holds `turn_000.json`, `turn_{N:03d}.diff` and `index.txt`, unchanged, at `2N` write-bearing operations.
**Scope**: `buster_ripper.dump/dump_turn`
**Why it persists**: It is the documented reading workflow and remains the default; the collapsed formats are added beside it.
**Source**: `US-012`; `plan-summary.md` §Approved Override Register entry 1; witness `VP-T002-documented-format-preserved` / `TC-108`.

### EI:buster_ripper.dump/dump_turn/sequential_diff_chain
**Statement**: Turn N's record derives from turn N−1's rendered text in arrival order; `prev_text` and `turn` advance at append time, never at flush time, so a flush never resets the chain.
**Scope**: `buster_ripper.dump/dump_turn`
**Why it persists**: It is format-independent; the documented writer's diff files are unchanged and each collapsed record keeps the same ordering.
**Source**: `plan-summary.md` §Approved Override Register entry 2; witness `VP-T002-change-readable`.

### EI:buster_ripper.dump/dump_turn/turn_zero_is_full_body
**Statement**: Turn 0's full normalized JSON body is present in every shipped format — `turn_000.json` on the documented path, the turn-0 block on either collapsed path — never only a diff.
**Scope**: `buster_ripper.dump/dump_turn`
**Why it persists**: It is the only record of the original request; a diff-only artifact cannot reconstruct it.
**Source**: `plan-summary.md` §Approved Override Register entry 3; witness `VP-T002-turn0-available`.

### EI:buster_ripper.routes.messages/handle_messages/dump_failure_isolated
**Statement**: A raising dump or flush cannot fail a `/v1/messages` request; the upstream response is still returned.
**Scope**: `buster_ripper.routes.messages/handle_messages`
**Why it persists**: The append path builds records and now covers more code, so the swallow-all contract matters more.
**Source**: `plan-summary.md` §Anchor Map; witness `VP-T001-dump-failure-isolated`.

### EI:buster_ripper.session/session_id/derived_after_normalize
**Statement**: The session id is derived from the body **after** normalization, so a per-request billing nonce cannot churn the session key.
**Scope**: `buster_ripper.session/session_id`
**Why it persists**: `count_tokens` and `/v1/messages` must key stats on the same stable id; deriving before normalization made the session-trigger nudge permanently dead.
**Source**: `T019`; `plan-summary.md` §Anchor Map; witness `test_a_billing_nonce_body_finds_the_stats_messages_recorded`.

### EI:buster_ripper.server/on_cleanup/graceful_shutdown_hook
**Statement**: On SIGINT/SIGTERM, aiohttp's `GracefulExit` runs `on_cleanup`, which cancels the background tasks, flushes the pending dump, and closes the shared client session before the process exits.
**Scope**: `buster_ripper.server/on_cleanup`
**Why it persists**: A raise in cleanup replaces `GracefulExit` and exits with a traceback; the flush is wrapped.
**Source**: `plan-summary.md` §Anchor Map; witness `VP-T003-shutdown-flush`.

## Planned Invariants Promoted To Existing

### EI:buster_ripper.dump/dump_format/format_is_per_artifact
**Origin**: `PI:T002/dump_format/format_is_per_artifact`
**Statement**: One artifact carries exactly one configured format. The format is chosen once per process (`config.DUMP_FORMAT`, set by `main()`, never mutated) and a collapsed artifact's filename states it.
**Scope**: `buster_ripper.dump/dump_turn`
**Why it matters**: A mixed artifact would have two record shapes under one name; `SessionState.fmt` captures the buffer-time format so a mid-run mutation cannot misfile records.
**Source**: `plan-summary.md` §Planned invariants; `US-014`; witness `VP-T002-collapsed-artifact` / `TC-102`.

### EI:buster_ripper.dump/dump_format/format_does_not_change_write_count
**Origin**: `PI:T002/dump_format/format_does_not_change_write_count`
**Statement**: The configured format changes what a record contains and which writer runs, never how many write-bearing operations a session performs for a given writer.
**Scope**: `buster_ripper.dump/dump_turn`
**Why it matters**: It protects `G-001`'s write-count reduction from a build that writes both paths.
**Source**: `plan-summary.md` §Planned invariants; `US-014`; witness `VP-T002-write-count-reduced` / `TC-105`.

### EI:buster_ripper.dump/dump_format/default_format_is_documented
**Origin**: `PI:T002/dump_format/default_format_is_documented`
**Statement**: With no format configured, the output is the documented per-turn layout at `2N` write-bearing operations; the collapsed formats are opt-in.
**Scope**: `buster_ripper.dump/dump_turn`
**Why it matters**: It is `US-012`'s gating clause; a default run that writes a collapsed artifact switches the old behavior before any proof.
**Source**: `US-012`; `US-014`; witness `VP-T002-default-encoding` / `TC-107`.

### EI:buster_ripper.dump/dump_format/format_set_is_closed
**Origin**: `PI:T002/dump_format/format_set_is_closed`
**Statement**: The accepted set is exactly the three values in `config.DUMP_FORMATS`; an unknown configured value is a hard failure, never a silent fallback.
**Scope**: `buster_ripper.dump/dump_turn`
**Why it matters**: A fourth value silently routed to the self-contained builder would ship an undocumented shape.
**Source**: `US-007`; `FK-007`; `US-014`; witness `tests/test_dump.py::test_unknown_format_raises_and_writes_nothing`.

### EI:buster_ripper.dump/dump_format/one_write_path_per_turn
**Origin**: `PI:T002/dump_format/one_write_path_per_turn`
**Statement**: For any configuration a session of N turns performs **either** `2N` **or** `F` write-bearing operations — never `2N + F`.
**Scope**: `buster_ripper.dump/dump_turn`
**Why it matters**: Producing both writers per turn is worse than today while every test passes (witness C).
**Source**: `US-012`; `FK-002`; `US-014`; witness the paired runs of `VP-T002-write-count-reduced` plus the two listings of `VP-T002-file-count-reduced`.

### EI:buster_ripper.dump/dump_format/documented_format_is_retained
**Origin**: `PI:T002/dump_format/documented_format_is_retained`
**Statement**: The documented writer remains in the code as a selectable value and produces today's three files with today's content and line format when selected; no increment removes it before a demonstration the user judged worth switching to.
**Scope**: `buster_ripper.dump/dump_turn`
**Why it matters**: It protects the exact act `US-012` called premature.
**Source**: `US-012`; `US-014`; witness `VP-T002-documented-format-preserved` / `TC-108`.

### EI:buster_ripper.server/on_startup/verification_on_unless_explicitly_disabled
**Origin**: `PI:T006/upstream_tls/verification_on_unless_explicitly_disabled`
**Statement**: The shared outbound client verifies the upstream's certificate unless the opt-out is explicitly configured. The opt-out is never inferred from the URL scheme, the host or the environment, and it defaults to off.
**Scope**: `buster_ripper.server/on_startup`
**Why it matters**: The default was implicit in an aiohttp library default; this change makes it a named, checkable posture.
**Source**: `US-011`; `US-013`; witness `VP-T006-verify-on-by-default` / `TC-003`.

### EI:buster_ripper.server/on_startup/optout_is_process_wide
**Origin**: `PI:T006/upstream_tls/optout_is_process_wide`
**Statement**: One connector carries one posture for every outbound call — the proxied request, the `/metrics` poll and the `/v1/models` probe alike. No call site has a posture of its own.
**Scope**: `buster_ripper.server/on_startup`
**Why it matters**: Exactly one `ClientSession` exists and every outbound call consumes it; splitting construction into a connector + session pair must not create per-call-site postures.
**Source**: `US-011`; `US-013`; witness `VP-T006-posture-is-process-wide` / `TC-114`.

## Approved Overrides

### OI:buster_ripper.dump/dump_turn/output_layout
**Old invariant**: `EI:buster_ripper.dump/dump_turn/output_layout`
**New invariant**: `EI:buster_ripper.dump/dump_format/documented_format_is_retained` + `EI:buster_ripper.dump/dump_format/default_format_is_documented`
**Decision**: preserve (session direction `refine`, corrected from `override` by organize 018)
**Reason**: The documented property survives whenever the documented format is selected; the collapsed artifact is added beside it. An override requires displacement, which `US-012` forbids. The default is `documented`.
**Approved**: yes — the user ratified round 17 (`US-014`); the removal half was refused in round 14 (`US-012`).
**Verification**:
  - `TC-108` / `VP-T002-documented-format-preserved` (documented format byte-unchanged at `2N` writes)
  - `TC-107` / `VP-T002-default-encoding` (no format configured produces the documented layout)
  - `TC-102` / `VP-T002-collapsed-artifact` (one `<format>.log`, no documented files, on the collapsed path)

### OI:buster_ripper.dump/dump_turn/sequential_diff_chain
**Old invariant**: `EI:buster_ripper.dump/dump_turn/sequential_diff_chain`
**New invariant**: `EI:buster_ripper.dump/dump_format/one_write_path_per_turn`
**Decision**: preserve (session direction `refine`)
**Reason**: The per-turn-diff property survives; the write-time clause changes. `turn`/`prev_text` advance at append time, never at flush time, so a flush never resets the chain. For a collapsed format each record derives from the previous turn's rendered text in arrival order; the append-time sequencing is format-independent.
**Approved**: yes — `US-014`.
**Verification**: `VP-T002-change-readable`.

### OI:buster_ripper.dump/dump_turn/turn_zero_is_full_body
**Old invariant**: `EI:buster_ripper.dump/dump_turn/turn_zero_is_full_body`
**New invariant**: `EI:buster_ripper.dump/dump_turn/turn_zero_is_full_body` (same handle, statement scoped per selected format)
**Decision**: preserve (session direction `refine`)
**Reason**: Turn 0's full normalized JSON body **and** its rendered rendering are present in the artifact once flushed, never only a diff — in every shipped collapsed format. The documented format's `turn_000.json` already satisfies this verbatim and is preserved.
**Approved**: yes — `US-014`.
**Verification**: `VP-T002-turn0-available`.

### OI:buster_ripper.server/on_startup/shared_client_verifies_by_default
**Old invariant**: `EI:buster_ripper.server/on_startup/shared_client_verifies_by_default`
**New invariant**: `EI:buster_ripper.server/on_startup/verification_on_unless_explicitly_disabled`
**Decision**: preserve (session direction `refine`)
**Reason**: The shared client is built with an explicit `aiohttp.TCPConnector` whose `ssl` is `not config.SKIP_TLS_VERIFY`, inside the startup hook. With the opt-out unset the posture resolves to the same verified context aiohttp's own default connector yields — the upstream's certificate is still verified by default, and the connector is now the named, checkable site. One connector governs every outbound call.
**Approved**: yes — by explicit delegation (`US-013`), round 15.
**Verification**:
  - `TC-003` / `VP-T006-verify-on-by-default`
  - `TC-113` / `VP-T006-optout-honored`
  - `TC-114` / `VP-T006-posture-is-process-wide`

## Rejected Current Behavior

### RB:buster_ripper.dump/dump_turn/buffer_only
**Observed behavior**: `dump_turn` always writes per-turn files and never buffers, so "bulk" changes nothing; or it buffers but a flush writes one artifact per pending turn.
**Why not preserved**: Witness A: the write count is still `N + F`, so the reduction is only *when* writes happen. The buffer must exist for the collapsed formats and a flush must carry the whole batch in one write.
**Replacement**: `EI:buster_ripper.dump/dump_format/format_does_not_change_write_count` + `flush_all`.

### RB:buster_ripper.dump/dump_turn/shutdown_only_flush
**Observed behavior**: Writing nothing until graceful shutdown; `index.txt` does not exist for the session's whole life.
**Why not preserved**: It makes the loss window the whole session and removes the documented live-reading workflow (`README.md:146-162`); the user's waiver covers in-flight messages, not the whole session.
**Replacement**: `FK-001` — both triggers plus shutdown.

### RB:buster_ripper.dump/dump_turn/rewrite_per_flush
**Observed behavior**: The flush rewrites the whole collapsed artifact each time; the write count falls while bytes grow `O(N²)`.
**Why not preserved**: The write *volume* rises while the count falls; `G-001` requires append-only, and `FK-002`'s presented option text says one bulk flush is one write.
**Replacement**: append-only `flush_all`.

### RB:buster_ripper.dump/dump_turn/both_writers_per_turn
**Observed behavior**: The documented writes and the collapsed artifact are both produced for the same turn: `2N + F`, worse than today, while every test passes.
**Why not preserved**: Arithmetic (`2N + F > 2N`) plus `FK-002`'s "one bulk flush is ONE write" and `US-012`.
**Replacement**: `EI:buster_ripper.dump/dump_format/one_write_path_per_turn`.

### RB:buster_ripper.config/DUMP_FORMAT/encoding_only_selector
**Observed behavior**: A two-value "encoding" selector that chooses only the record encoding, not the layout.
**Why not preserved**: A value names a complete output shape including the layout axis; an encoding-only set would let an implementer miss the documented writer.
**Replacement**: the closed three-value `DUMP_FORMATS` set.

### RB:buster_ripper.dump/dump_turn/artifact_rewrite
**Observed behavior**: A rewrite-per-flush artifact, or a per-turn file on the collapsed path.
**Why not preserved**: It makes bytes grow `O(N²)` and breaks the append-only contract; the collapsed artifact is one append-only file per `(sid, format)`.
**Replacement**: `flush_all`'s single append.

# Invariants — `buster_ripper.server`

Registry file for the server/outbound-client cluster. Created 2026-09-16 by session
`02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573` (interview round 18, final-approval gate).

## Entities

Entity: E:T003/server/shutdown-flush-seam
Binding: buster_ripper/server.py:on_cleanup
Gloss: The aiohttp application cleanup hook, which runs on a graceful SIGINT/SIGTERM exit; it owns the teardown order — cancel the background tasks, then flush, then close the shared client session.
Confusable: commonly misread as the `main` entry point — the duplicate dev launcher `server.py:99 def main` this block once cited was deleted at T003 (quality-backlog QB:entry-points/duplicate-main, absorbed into Checkpoint 66); the console entry point is `main.py:main` alone, and it is not the cleanup seam either
Source: session=02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573 task=T003

---

Entity: E:T006/server/shared-client-construction
Binding: buster_ripper/server.py:on_startup
Gloss: Where the package's only outbound `ClientSession` is built, inside the running loop; it becomes an explicit connector + session pair so the outbound TLS posture is a named, checkable site rather than an inherited library default.
Confusable: commonly misread as the `ClientSession(...)` statement at `server.py:47` alone — that statement cannot carry the posture, because this aiohttp version's `ClientSession.__init__` accepts no `ssl` kwarg and the connector is the only seam — session Anchor Map (decide 016/017)
Source: session=02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573 task=T006

---

## Overrides (existing invariants in the session's contradiction surface)

The session's direction token was `refine`; `Decision: preserve` is the Speeckle enum value for it — the property
survives and is only made checkable for the first time. No `EI:` statement block is invented below: the posture was
never a recorded invariant, only an inherited `aiohttp` default (`connector.py:978`), so there is no existing handle for
the registry to hold.

Override: OI:buster_ripper.server/on_startup/shared_client_verifies_by_default
Old: EI:buster_ripper.server/on_startup/shared_client_verifies_by_default
New: PI:T006/upstream_tls/verification_on_unless_explicitly_disabled
Decision: preserve
Reason: Session direction `refine`. The shared client is built with an explicit `aiohttp.TCPConnector` whose `ssl` is the negation of the configured opt-out, constructed inside the startup hook. With the opt-out unset the posture is `ssl=True`, which resolves to the same verified context the library's own default connector yields — so the upstream's certificate is still verified by default, and the connector is now the named, checkable site. One connector governs every outbound call, and the opt-out is never inferred from the URL scheme, the host or the environment. The old wording clause ("built with no `ssl`/`connector=` argument") becomes literally false, which is why the entry exists.
Approved: yes
Session: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
Source: session=02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573, plan-summary.md §Approved Override Register entry 4; approved by the user's explicit delegation (US-013), round 15
Verification:
  - TC-003 / VP-T006-verify-on-by-default (untrusted-certificate peer refused with no opt-out)
  - TC-113 / VP-T006-optout-honored (same peer completes with the opt-out)

---

## Planned Invariants

Invariant: PI:T006/upstream_tls/verification_on_unless_explicitly_disabled
Kind: behavioral
Strength: protected
Status: planned
Scope: buster_ripper.server/on_startup
Statement: The shared outbound client verifies the upstream's certificate unless the opt-out is explicitly configured. The opt-out is never inferred from the URL scheme, the host, or the environment, and it defaults to off.
Why: The user asked for strong security by default with an explicit way off for self-signed certificates in local deployments. The default is currently implicit in a library default, so this change is what can invert it — and therefore the first time the property needs a witness.
Source: session=02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573, plan-summary.md §Planned invariants; approved by the user's delegation round 15 (US-013); grounded in US-011
Witness: TC-003 and TC-113 (VP-T006-verify-on-by-default, VP-T006-optout-honored)
Evidence:
  - buster_ripper/server.py:on_startup
  - .sessions/codify/02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/qa-plan.md

---

Invariant: PI:T006/upstream_tls/optout_is_process_wide
Kind: structural
Strength: protected
Status: planned
Scope: buster_ripper.server/on_startup
Statement: One connector carries one posture for every outbound call — the proxied request, the metrics poll and the models probe alike. No call site has a posture of its own.
Why: Exactly one client session exists in the package and every outbound call is consumed from it, so splitting the construction into a connector + session pair must not create per-call-site postures. This is the coupling the split makes invisible, and this invariant is what keeps it visible.
Source: session=02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573, plan-summary.md §Planned invariants; approved by the user's delegation round 15 (US-013); grounded in US-011
Witness: TC-114 (VP-T006-posture-is-process-wide), paired with a black-box VP so it cannot mirror the implementation
Evidence:
  - buster_ripper/server.py:on_startup
  - .sessions/codify/02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/qa-plan.md

---

## Deferred at elevation (no registry entry yet)

| Term | Planned binding | Register at |
|---|---|---|
| upstream TLS verification opt-out | `buster_ripper/config.py:SKIP_TLS_VERIFY` + `buster_ripper/main.py:--skip-tls-verify` | T006 Yield |

The opt-out's flag and global are **authorized new symbols** that do not exist in the working tree yet; per the
no-dangling-binding precedent the entity is deferred to `T006`'s Yield rather than registered with a binding that cannot
resolve today.



## Trimmed 2026-10-07 (`083-01-yield`) — CR3 wave accepted-local (context budget)
Moved verbatim out of `handoff-context.md` by the CR3 wave's local Yield. Nothing rewritten; the pre-trim
file is preserved by acceptance Checkpoint 193 (`eefd718`). Contents: the two superseded `Last Updated`
entries, then the six accepted blueprints in their original order (`T015` `T016` `T017` `T018` `T019` `T020`).

### Superseded §Session Metadata `Last Updated` entries

- Last Updated: 2026-10-06 (**`QAP-MS001` re-entered, `READY-PASSING`, at [077-forward-qa](iterations/077-forward-qa.md)** — a QA-lane drain, not a re-authoring. The two `quality-backlog.md` rows whose owning route is `forward-qa` and whose *named trigger is this pass* are closed together: `QB:qa-harness/unexercised-path-keyword` (`Proxy.url()` / `Proxy.post(body, timeout=…)` restored to their pre-055 signatures; zero consumers ever passed `path`) and `QB:qa-harness/static-body-cannot-connect-limb` (`assert_upstream_response`'s `b"Cannot connect" not in body` limb dropped — it could no longer fire from a proxy-local failure and, after the raw-relay work, could fire for the wrong reason). The drain weakens no approved assertion: the status limb is the discriminator and a proxy-local `ClientError` answers `502`, confirmed by one bounded Boundary Inspection Exception rather than assumed. One process-free fail-first witness added (`…test_sensitivity_upstream_oracle_distinguishes_transport_failure_from_a_relayed_body`, red if the limb returns); 055's inaccurate §Files Changed sentence corrected as a dated append. No `TC-*`/`VP-*` semantics and no scenario assertion moved. Runs: `QAP-MS001` **11/11**, `QAP-MS002` **15/15**, `QAP-MS004` **6/6** (all three import the changed helper), hermetic **455 @ 95.75%**; the `G-008`(b) documented half re-measured identical (`write_ops=2N=6`, `files=N+1=4`); the `022` directive still honored by grep rather than memory. `READY-PASSING` is not a verdict — Yield Phase owns `pass`/`fail` and decides whether the drain reopens 071's approval.)
- Previous entry, kept verbatim: Last Updated: 2026-10-06 (**`T017` (seam-S15 CE-2 repair wave, wave of one) ACCEPTED-LOCAL at [075-01-yield](iterations/075-01-yield.md)** — zero reject-tier findings, counter 0; the blueprint's counterexample replayed as 36 focused witnesses and hermetic **455 @ 95.75%** re-run on the commit, one-file production diff proving the four must-not-touch prohibitions, `84 0` on the census module, the eight dirty `invariants/`+`verdicts/` paths independently proven not Forward's; 2 new backlog rows, 5 defer-tier items already on rows; no phase acceptance, no invariant, no `decisions.md` byte, no `Verification: yield:` row manufactured. The boundary owed **one `seam-S15` recomposition** — it landed and is routed at [076-assumption-decide](iterations/076-assumption-decide.md) (`S15` terminal: citable as current, CE-1 deferred to §RM-008, nothing on it gates the close) — and now owes **Yield Red**. Rulings, verification and the trim disposition are in §Current Status.)
- Prior entry, kept verbatim: Last Updated: 2026-10-06 (**MS-001 integration gate SEMANTIC-APPROVED, zero findings, at [071-yield-phase](iterations/071-yield-phase.md)** — the session's last milestone gate and its last owned `PENDING` claim: `QAP-MS001` **10/10** re-run independently against the live upstream at HEAD `c7420ce` (`TC-100`/`TC-001`×2/`TC-107`/`TC-108` + sensitivity + the R1/R2 hostile tier + the R5 identity tier), all seven checks PASS on the cumulative diff `72e3a1c6…HEAD`, the `022` directive ratified as honored-literal-with-an-expired-premise (no plan change owed), and `US-009` c2 arbitrated **HOLDS** first-party (`require_upstream()` 0.05 s, `/v1/models` → 200 vLLM-shaped, proxied `POST /v1/messages` answering the upstream's own typed `404` rather than a proxy-local `502 Cannot connect`) with `Consequence: none`. Check 7's scope: hermetic **407 @ 95.75%** (`dump.py` 100%) plus the whole `tests/qa/` tree on these bytes — `MS-002` 15/15 and `MS-004` 6/6 re-run because 070 changed the shared `harness.py`; **31/31 process-level gates green**. No phase acceptance, no `Delivered` chronicle entry, no parent completion — **Yield Red owns them**; the retry counter is untouched because nothing was rejected. Prior, verbatim, same day: **MS-002 integration gate SEMANTIC-APPROVED, zero findings, at [069-yield-phase](iterations/069-yield-phase.md)** — the owed re-verdict of the milestone rejected at 035, run at HEAD `bd16795` with the declared upstream reachable and confirmed first-party (mDNS → `192.168.50.132`, `GET /v1/models` → HTTP 200, the committed `require_upstream()` preflight passing in 0.02 s, proxied requests answering the upstream's own 404 rather than a proxy-local `502 Cannot connect`); `QAP-MS002` **15/15 twice**, hermetic **407 @ 95.75%** (`dump.py` 100%), combined `tests/qa` **21/21**; all seven checks PASS; Checks 1-5 re-derived against a tree in which **every gated file moved since 035** (`dump.py` ±50, `routes/messages.py` ±87, `main.py` ±67, `server.py` ±23, `config.py` ±4, the gate module ±71); T003.001's typed-400 removal criterion verified (`grep -rn "except json.JSONDecodeError" buster_ripper/` → the one deliberate syntax-fallback arm in `count_tokens.py:20`); the durable G-008(b) comparison record re-issued with the same numbers and its stale environment disclosure superseded; **retry counter stays at 1 — this pass issued no rejection**; **phase acceptance, the Delivered chronicle entry and parent completion are NOT published here — Yield Red owns them.** Prior, verbatim, same day: `T014` image-slice wave ACCEPTED at [068-01-yield](iterations/068-01-yield.md) — first attempt, zero rejections; all three members verified by independent re-derivation rather than by Forward's tallies: F-1 scan-all detector + `DC-CE-1-r1` shape safety (the flag-on/flag-off dict-shaped `messages` pair re-run through `/v1/messages`, same status class, no 500, no `try` added), F-2 `063 R-A` content-derived key with `img_04b45e9ae5ea` recomputed from the bound formula alone + `DC-S13-CE1-r1`'s at-cap converse + the 8-insert liveness loop, F-3 write-path tail sweep with the outer-key drop as **new code inside `cleanup_expired_images`** and both stamps off `time.monotonic` (W5/W6/W7 each hand-traced against the build they exist to catch). My runs: affected scope **173 passed**, full suite **406 passed**, and base `tests/test_image.py` against the new module = **6 red**, matching the claimed 145 state exactly; the two formerly-false witnesses confirmed load-bearing, not tautological. **Both escalated items ruled in Forward's favor** — three-file test surface ACCEPTED (the only arrangement that keeps every `063 R-D` witness under the >700 hard stop; production surface stayed at one file), `.encode("utf-8")` HONORS the `063 R-A` binding (the printed formula raises `TypeError`; the pin reproduces from the bound operands). **One gap resolved that Forward did not escalate:** `W7` as worded ("with the stamp source injected … no entry's `now - first_seen` is negative") is unsatisfiable by a correct build, so the wall-clock-pinned realization is the only admissible reading. Ledger row 278 settled in Yield's lane → `HOLD` (`image.py` imports `config` and stdlib only, bytes `f4894b65…`). Nothing T013 landed was restyled; the `DC-F-3-r2` / `FK-007` / `DC-S13-CE4-r1` deferrals verified honored; `image.py` 185→255 Green and the file's issue count 4→3. 3 defer-tier items, 3 accepted risks; 13 routed rows → `accepted-local`, `seam-S12`/`S13`/`S14` recomposition plus rows 293/294/297/298 + 267/287/316 + 279-285 still owed to the assumption route. `FK-011` still open. Previous: **`T014` implemented by [068-01-forward](iterations/068-01-forward.md)** — 52 witnesses, 15-mutation fail-first matrix, `image.py` 98%, affected scope 173 passed, the 772-line FL-LNCP split disclosed for Yield to rule. Previous: **`T013` header-parity wave ACCEPTED at [067-01-yield](iterations/067-01-yield.md)** — all three members verified: F-4 `c31d1ac` + retry `2be9783`/`ce55974`, G-1 `d0b1317`, G-2 `96fe429`; `066-01-yield P1` closed (positive `_carries_token`, zero credential regressions vs base, red-before-green and all eleven matrix rows re-derived, backend independence verified at the wire); 374 @ 94.20%; 16 routed rows → `accepted-local`; the wave's P2 sibling-boundary defect is out of scope and travels to roadmap.md. `062 R-A`…`R-E` stand unmodified, no new referent, no invariant touched, `MS-004`'s APPROVED verdict and every accepted scope unchanged.)

### The six accepted Planned Task Context blocks

### T015 Planned Task Context — Restore upstream response-header parity and raw-relay fall-through on the chat non-stream arm (CR2 wave, member 1)

Routed by the 2026-10-06 closure-packet routing pass (external review findings, post-071 `codex-review` — the CR2
batch). Every finding re-verified against current bytes at HEAD `dbd3072` before disposition; all hold — none was
already-closed. Forward receives only session id + task id; this block is the transmission channel.

- Finding ids: **CR2-F-1** (P1) + **CR2-F-2** (P1, chat arm). Evidence path:
  [`evidence/codex-review-post-071-packet.txt`](evidence/codex-review-post-071-packet.txt) (producer packet, reviewed
  head `2be9783`; fingerprint: none recorded by the producer — the packet states its reviewed head, and this pass
  re-derived every site at `dbd3072`).
- Disposition: **absorbed** — ledgers `DC-CR2-F-1-r1`, `DC-CR2-F-2-r1` (below, §CR2 ledger).
- Status: **accepted-local** — [073-01-yield](iterations/073-01-yield.md) ACCEPT, zero reject-tier findings;
  every done-when clause verified by re-derivation (incl. the fail-first replay at both base commits).
- Required behavior (base contract, per the boundary-parity direction `T013` settled):
  1. Every non-stream `/v1/chat/completions` answer relays the upstream's end-to-end response headers — both
     `_apply_response_transforms` return paths (`chat.py:207-211`, `:215-220` at HEAD) and the parse-failure
     fall-through — filtered by the shared `utils.response_headers` (hop-by-hop + `content-length` drop, the
     messages.py `:285-290` production shape).
  2. A well-formed non-JSON upstream answer (gateway HTML 502, plaintext 429) is relayed **as upstream sent it**:
     upstream status, upstream bytes verbatim, filtered upstream headers — never replaced by a synthetic 502.
     Read the body once; parse only where the transform or the overflow check needs the parsed form.
  3. A genuine transport failure (ClientError before anything is sent) still answers 502, but the client-facing
     body carries **no exception text** — `str(e)` currently leaks the internal upstream URL (the QA gates at
     069/071 recorded a proxy-local `502 Cannot connect to host <upstream>` body verbatim; the exception stays in
     the log line).
- Counterexamples (Yield-replayable): upstream `200` JSON carrying `X-Request-Id` → caller sees `Content-Type`
  only (CR2-F-1); upstream `text/html` 502 to a non-stream request → caller sees `502
  {"error": "Invalid JSON from upstream"}` with upstream status/body/headers lost (`chat.py:152`, `:158-159`);
  connection failure → upstream host:port echoed in the response body (`chat.py:155-157`).
- Touch-set: `buster_ripper/routes/chat.py` — `_make_chat_request` non-stream arm + `_apply_response_transforms`
  (thread upstream headers in), plus their witnesses. **Must not touch:** `_relay_chat_stream` (`:174-188`) — its
  post-commit second head is owned by `roadmap.md` §Chat route and must not be folded in (codify-scope, see
  `DC-CR2-F-4-r1`); `_prepare_headers`/`_carries_token` (T013 accepted surface — "Nothing in this batch reopens
  T013/T014's accepted scope"); `_apply_request_transforms` (chat's own try/except-swallowed injection is not in
  this batch's named scope); the `handle_chat` retry arm (CR2-X-1 is `malformed`, stays with the collect lane).
- Capacity: `chat.py` sits at **299/300 raw lines** (FL-LNCP Green ceiling; §RM-001 records extraction arithmetically
  blocked). The closure removes the two-arm `except` bodies it replaces and can land at or under the cap; if it
  cannot, file the capacity review item **with** the change — do not silently breach.
- Done when: fail-first witnesses green for all three required behaviors above — including the header relay on
  **both** transform return paths and a `Retry-After`/`X-Request-Id`-style end-to-end header observed on the wire;
  `tests/test_routes_integration.py:239-243` (which today **pins the defect shape**) is replaced by the raw-relay
  witness, and the `upstream.json.assert_awaited_once()` mechanism pin at `:179-192` is re-pinned to the
  read-once shape without weakening transform coverage; overflow-retry witnesses on parseable 4xx answers stay
  green (behavior unchanged); full hermetic suite green; `chat.py` ≤300 raw lines or the capacity item filed with
  the change.
- Yield replay: drive the mock upstream with (a) 200 JSON + end-to-end headers, (b) `text/html` 502, (c) an
  injected `ClientError`, at `/v1/chat/completions` non-stream, and assert relayed headers / verbatim body+status /
  leak-free 502 respectively; confirm `git diff` shows no bytes moved in `_relay_chat_stream`, `_prepare_headers`,
  `_apply_request_transforms`.
- Invariant cross-reference: **Neutral** against the registry (every registry entry in scope is `Status: planned`;
  no active EI/OI governs response relay or error-body shape) — **Aligned** with the session-settled direction
  `T013` (restore base header handling at every proxy boundary) and the `utils.response_headers` production
  contract as already used by messages.py.
- Invariant candidate: none authored here. If Yield wants the durable rule ("a proxied answer carries upstream's
  status, bytes and non-hop-by-hop headers verbatim"), it is Interview's to record — non-authoritative for later
  review until then.
- Closure ledger: `DC-CR2-F-1-r1`, `DC-CR2-F-2-r1`.
- Owning route: Forward → one local Yield for the whole CR2 wave. Retry owner: T015's own counter, starting 0.
- Execution order: wave anchor (T015), member 1; prerequisite of the **MS-001 boundary close** — land before
  Yield Red's final semantic pass over this batch. No `--before` referent exists (no pending planned task); the
  ordering is durable here and in the plan row.
- Verification profile (proportional): medium risk — response-shape change on an exposed route; easy recovery —
  one member commit; low uncertainty — base contract read directly from `buster_ripper_legacy.py:882-892`; gate —
  focused `tests/test_routes_integration.py::TestChatRoute` + full hermetic suite; budget — one Yield pass.

### T016 Planned Task Context — Guard the preserved-thinking injection; de-leak client-facing error bodies on messages + passthrough (CR2 wave, member 2)

- Finding ids: **CR2-F-3** (P1) + **CR2-F-2** (P1, the non-chat leak arms). Evidence path:
  [`evidence/codex-review-post-071-packet.txt`](evidence/codex-review-post-071-packet.txt).
- Disposition: **absorbed** — ledgers `DC-CR2-F-3-r1`, and `DC-CR2-F-2-r1` for the arms.
- Status: **accepted-local** — same wave, same Yield pass; the null-kwargs 500 counterexample reproduced at
  base bytes and closed at final bytes, with the flag-off arm pinned byte-identical.
- Counterexample (CR2-F-3, re-verified at `dbd3072`): with `PRESERVED_THINKING` on — the shipped launcher default,
  `serve_buster_ripper.sh:30` — a `POST /v1/messages` body carrying `"chat_template_kwargs": null` hits
  `messages.py:45-48`: `body.setdefault` returns the existing `None` (the key **is** present), the nested
  `kwargs.setdefault("enable_thinking", True)` raises `AttributeError`, nothing above `handle_messages` catches
  it, and the client receives HTTP 500 with **no upstream call**. `normalize`/`session_id`/image handling pass the
  key through untouched (verified); branch-added surface, not inherited.
- Required behavior:
  1. With the flag on, **every** forwarded `/v1/messages` request carries the injected `enable_thinking=true` /
     `clear_thinking=false` — including when the client sent a present-but-non-mapping `chat_template_kwargs`
     (`null`, a list, a scalar): isinstance-check and **replace with a fresh dict** before injecting. The flag's
     documented contract (`main.py:107-110`: "Inject enable_thinking=true + clear_thinking=false into /v1/messages
     requests") is what the guard must preserve; the route must not 500 from this block.
  2. No client-facing error body on any proxy route embeds exception text: replace the `{"error": str(e)}` 502
     bodies at `messages.py:158-160` and `:291-293` and `passthrough.py:48-49` with static JSON messages; the
     exception stays in the existing `log.error` lines. (`_relay_after_commit`'s post-commit contract already logs
     position-not-cause and returns no second response — T013 accepted; not in this touch-set.)
- Touch-set: `messages.py:45-52` (guard) + `messages.py:158-160`, `:291-293` + `passthrough.py:48-49` (static
  bodies); witnesses for all of it. **Must not touch:** `chat.py` (that is T015's file), and **nothing in
  `decisions.md`** — see the FK-011 line.
- Done when: fail-first witnesses green — a `/v1/messages` request with `"chat_template_kwargs": null` under
  `PRESERVED_THINKING` injects and forwards (never 500; witness at the route level with the flag actually on,
  per the shipped default), same for a list-valued and a scalar-valued field; flag-off path byte-identical (no
  cost on that arm); each of the three arms answers a static body that provably contains no upstream host:port
  (assert the configured `UPSTREAM` string is absent from the response text); full hermetic suite green.
- Yield replay: reproduce the null-kwargs 500 counterexample before/after; inject connection failures against
  all three routes and assert leak-free 502s; confirm the flag-off and non-error paths are byte-unchanged.
- Invariant cross-reference: **Neutral** against the registry; **Aligned** with the package convention
  (`normalize.py` and `image.py` guard every non-dict input) and the documented flag contract (`main.py:107-110`).
- FK-011: **stays open, untouched.** The guard is posture-independent — it holds under option 1 (default-on +
  document) and option 2 (default-off + opt-in) alike. Forward/Yield must not edit the `FK-011` record; its cited
  `messages.py:43-50` line pointers were already stale (block sits at `:45-52`) before this batch, and T016 moves
  them again — no re-pointing is this task's job.
- Invariant candidate: none.
- Closure ledger: `DC-CR2-F-3-r1` (+ `DC-CR2-F-2-r1`, arms half).
- Owning route: Forward → the same single local Yield for the CR2 wave. Retry owner: T016's own counter, starting 0.
- Execution order: wave member 2; files disjoint from T015 (no interference — different files entirely);
  prerequisite of the MS-001 boundary close, same as T015.
- Verification profile: low risk on the guard, medium on the body-shape change (three routes, exposed error
  surface); easy recovery; gate — focused route tests + full hermetic suite; budget — rides T015's Yield pass.

### T017 Planned Task Context — Relay every occurrence of a repeated upstream response header name (seam-S15 CE-2)

Routed by the 2026-10-06 `assumption-decide` composition-routing pass over
`assumptions/compositions/seam-S15.md` (`VERDICT: COUNTEREXAMPLE`). Forward receives session id + task id
only; this block is the transmission channel. Reach and the base comparison were established by execution
this pass, not inherited — `evidence/s15-reach-and-base-comparison.txt` §2 and §3 are the citations.

- Assumption finding: seam-S15 (counterexample **CE-2**)
- Evidence: `assumptions/compositions/seam-S15.md` §Counterexample CE-2 + §Reach; reach/base evidence
  `evidence/s15-reach-and-base-comparison.txt` §2 (CE-2 reach) and §3 (closure feasibility)
- Disposition: absorbed
- Status: **accepted-local (075-01-yield)** — landed at Checkpoint 173 (`6a13bd7`, base `e68ae02`) and accepted
  on independent re-derivation; **not `resolved`** — the `seam-S15` recomposition and Yield Red own that.
- Counterexample: the operator-declared upstream `http://hyperipper.local:30001` answers
  `GET /v1/models` with `date` **twice** and `server` **twice** in one head (raw-socket transcript, evidence
  §2a, measured over 6 consecutive requests: **6/6 responses repeat both names** and **1/6 carried two
   *different* `date` values** (`…19:42:12 GMT`, `…19:42:13 GMT`) — an occurrence is discarded on every
   response, a distinct value roughly one in six. aiohttp's
  `CIMultiDictProxy` yields 6 per-occurrence pairs; `utils.response_headers` returns **3 entries** and keeps
  the **last** value; the running production proxy answers the client with one `Date` and one `Server` line.
  With a peer set to `Set-Cookie: a=1` + `Set-Cookie: b=2`, the upstream sends 2 lines and the client
  receives 1 — `a=1` is destroyed.
- Required behavior: at all three response boundaries (`passthrough.py:41`, `messages.py:291`, `chat.py:172`),
  **every occurrence of a repeated upstream header name reaches the client on its own field-line**, dropping
  exactly the names `response_headers` drops today (the eight `_HOP_BY_HOP` names + `content-length`) and
  nothing else. This is the base's *information* contract, not its shape: base relayed
  `date: T1, T2` because `httpx.Headers.items()` merges occurrences
  (`httpx/_models.py:216-229`), so base never discarded a value; the aiohttp swap plus the plain-`dict`
  fold did. `web.Response(headers=…)` already emits one line per occurrence when handed a `CIMultiDict`
  or an iterable of pairs (verified, evidence §3) — the loss is in the filter, not in the server.
- Done when: `response_headers` returns a per-occurrence-preserving collection and all three sites still
  hand its result straight to `web.Response(headers=…)`; a fail-first witness fed two occurrences of one
  name with **different** values is red before the change and green after and asserts **both** values on the
  wire (one line each) — at least one per boundary plus a `utils.response_headers` unit witness; the
  `Content-Encoding` gap CE-1 leaves open is **documented, not decided**: the filter's docstring names the
  posture it currently keeps (the encoding label is relayed over already-decoded bytes, per base parity) and
  points at `roadmap.md` §RM-008, so the design pass inherits a stated posture instead of an undocumented
  one — and no behavioral witness is written for it, so that ruling stays free to go either way;
  `tests/test_header_parity.py`'s existing kept/drop, direction-pairing and
  six-name witnesses stay green **unweakened**, including
  `test_utils_holds_the_only_two_filters` (`source.count("for k, v in") == 2` in `utils.py`); no name enters
  `_HOP_BY_HOP`; full hermetic suite green; LNCP/line counts not worsened (`utils.py` has headroom).
- Yield replay: drive one proxied request per boundary against a peer that repeats a header name with
  differing values and assert the count and both values on the wire, then re-run the drop census and confirm
  from the diff that `forward_headers` is untouched (the egress fold is base parity — see N2) and that no
  proxy-authored header literal was routed through the filter (`DC-S10-C4-r1`'s prohibition stands).
- Assumption impact: scoped:S15
- Invariant cross-reference: Neutral against the registry — zero `active`/`wip` entries exist
  (`grep -rn "Status: active\|Status: wip" invariants/` → nothing; all eight `Invariant:` rows are
  `planned`); no session `invariants.md`. Aligned with the session-settled `T013` direction (restore base
  header handling at every proxy boundary) and with `T015`'s accepted required behavior 2. Descriptive
  handles cited as evidence only, authorizing nothing: `CL:aiohttp/repeated-header-items-iteration`,
  `CO:client-response-to-relay-consumer/payload-and-header-pairing`.
- Invariant candidate: for Interview to consider, not authored here — *"the response relay discards no
  upstream header occurrence beyond the eight hop-by-hop names and `content-length`"*. Non-authoritative
  until recorded; must be worded so it does **not** pre-decide the CE-1 content-coding posture.
- Closure ledger: `DC-S15-CE-2-r1`
- Owning route: Forward → one local Yield (wave of one). Retry owner: T017's own counter, starting 0.
- Execution order: **wave anchor** (sole member); prerequisite of the MS-001 boundary close — land before
  Yield Red's final semantic pass, which is the gate that certifies this boundary's header relay. No
  `--before` referent exists (no other pending planned task); the ordering is durable here and in the plan row.
  It does **not** wait on, and must not be folded into, `RM-008` (that row owns CE-1's posture ruling).
- Verification profile (proportional): medium risk — response-shape change on three exposed routes; easy
  recovery — one file, one commit; low uncertainty — both clients' `.items()` semantics read and the loss
  demonstrated end-to-end; gate — focused `tests/test_header_parity.py` + per-boundary witnesses + full
  hermetic suite; budget — one local Yield for the wave.
- Out of this task, explicitly: no `content-encoding` drop, no egress `Accept-Encoding` change, no
  `auto_decompress` argument (CE-1's posture belongs to `RM-008`); no change to `forward_headers`
  (S10's boundary, terminal, and base parity there); no change to `_HOP_BY_HOP`.

### T018 Planned Task Context — Restore the base transport contract on the aiohttp app: no body cap, no stream deadline (CR3 wave, member 1 — anchor)

Routed by the 2026-10-06 closure-packet routing pass ([080-assumption-decide](iterations/080-assumption-decide.md));
every finding re-verified against current bytes at `4008e27b…` (= the reviewed bytes, `6508941..HEAD` is
session-only). Forward receives session id + task id only; this block is the transmission channel.

- Finding ids: **CR3-F-01** (P1, `server.py:119`) + **CR3-F-03** (P2, `server.py:67` + `routes/messages.py:137`).
  Evidence path: `iterations/079-codex-review.md` §CR3-F-01/§CR3-F-03 (packet + this pass's re-verification notes
  in its filled `Disposition:` lines).
- Disposition: **absorbed** — ledgers `DC-CR3-F-01-r1`, `DC-CR3-F-03-r1`.
- Status: pending
- Required behavior:
  1. The app answers **no** request with a 413 for body size: `create_app()` — the package's only
     `web.Application` construction — passes `client_max_size=0` (aiohttp's own reader skips the check on a
     falsy value, `web_request.py:675`; default is `1024**2`, `web_app.py:128`). Base (Starlette/uvicorn)
     imposed no cap; every route class — `/v1/messages`, `/v1/messages/count_tokens`,
     `/v1/chat/completions`, the passthrough catch-all — must accept a body base would have accepted.
  2. The `/v1/messages` **streaming** relay lives as long as the upstream streams: `_stream_single_request`
     passes `timeout=aiohttp.ClientTimeout(total=None)` on its `session.post` (per-request `timeout=` replaces
     the session default for that call — registry `CL:aiohttp/per-request-timeout-override`, HOLDS). Base ran
     the stream path `timeout=None` (`buster_ripper_legacy.py:664`). **Every other consumer of the shared
     session keeps its current timeout** — the 600 s at `server.py:67` stays byte-identical, the non-stream
     arm is not this task, and chat/passthrough/`/metrics`/probe already set their own per-request values.
  3. The verified-sound arm survives untouched: `_relay_after_commit`'s `try/except` logic and
     `tests/test_stream_relay_commit.py`'s `POST_COMMIT_FAULTS` witnesses (bare-`TimeoutError` case included)
     must stay byte-valid — the review cleared the **mechanism**; this task changes only the **deadline
     policy upstream of it**. The one permitted edit inside that function's text is its docstring's clause
     citing the cumulative `ClientTimeout.total` budget as an in-path failure source, corrected to match the
     new configuration without asserting anything false about the class hierarchy.
- Counterexamples (Yield-replayable): (1) 2 MiB POST to each of the four route classes → 413 "Maximum request
  body size 1048576 exceeded" (reproduced by the review on the real app; witnessable on the existing
  `proxy`/`raw_proxy` real-server tier); (2) shared session with a small `total` + chunked upstream straddling
  the deadline → the 200 SSE stream truncates silently, no `[DONE]`, no error event (before) vs relays to
  completion (after).
- Touch-set: `buster_ripper/server.py` `create_app` (one kwarg + a comment naming the parity reason);
  `buster_ripper/routes/messages.py` `_stream_single_request` (kwarg) + `_relay_after_commit` **docstring
  clause only**; witnesses in `tests/`.
  **Must not touch:** `server.py:65-68` (T006's accepted connector+session construction — VP-T006's site),
  `_relay_after_commit`'s logic, `chat.py`/`passthrough.py` per-request timeouts, `_HOP_BY_HOP`,
  `tests/test_stream_relay_commit.py`.
- Done when: fail-first witnesses green — the four-route-class >1 MiB POST witnesses (each red at base bytes:
  413; green after: the route's normal body-read path runs), the short-total stream-completion witness (red
  before, green after), a companion witness that the **non-stream** arm's inherited deadline is unchanged by
  this task, **and one assertion that the streaming call's request-local `ClientTimeout` carries every component
  `None`** — `total`, `connect`, `sock_connect`, `sock_read` (added by
  [082-assumption-decide](iterations/082-assumption-decide.md): per `CL:aiohttp/per-request-timeout-override` a
  request-local `timeout=` *replaces* the session object rather than merging into it, so all-`None` is what makes
  "no deadline at all" true on this path, and a spelling that re-inherits the session's `total` or arms `sock_read`
  "to be safe" passes the truncation witness while leaving `S17-CE-1`/`S17-CE-2` live); full hermetic suite green
  (must reproduce **455+** at ≥95.75%, same coverage figure or better);
  LNCP/line counts not worsened (`server.py` is coverage-omitted; `messages.py` touched functions measured
  before/after — `handle_messages`'s hard **C:11** is §RM-001's own decision and must not move).
- Yield replay: re-drive the 2 MiB POST and the short-total slow stream on the landed bytes; confirm from the
  diff that `_relay_after_commit`'s `try/except` body and the injected-fault table are byte-unchanged; confirm
  `server.py:66-68` is byte-unchanged; and read the streaming call's `timeout=` operand against the new assertion
  — **all four components `None`**, not merely a `total=` that happens to look unlimited (`S17`'s `Invalidates On`
  makes any other spelling a re-composition trigger, and `S17-CE-1` survives exactly that miss).
- Assumption impact: required:new-surface **→ scoped:S16,S17 as of**
  [082-assumption-decide](iterations/082-assumption-decide.md) — 080 recorded `required:new-surface` before either
  seam existed; `S16` and `S17` are now composed, live, and **hold open** until their post-`T018` compositions come
  back `SAFE` (each record's `Invalidates On` names this task's own surface: the cap channel for `S16` step 1, the
  `timeout=` kwarg and the docstring clause for `S17` step 2). S9 stays table-marked `out-of-scope`; the boundary's
  owed assumptions recheck still re-bases any fingerprint these edits move.
- Invariant cross-reference: **Neutral** for both findings (zero `active`/`wip` registry entries govern request
  caps or client deadlines; eight `planned` PIs in unrelated packages). Invariant candidate (F-01) for
  Interview, not authored here: *"the application-level request-body cap is disabled at the construction site
  (`client_max_size=0`) — the proxy rejects no body the base stack accepted."* Non-authoritative.
- Closure ledger: `DC-CR3-F-01-r1`, `DC-CR3-F-03-r1`.
- Owning route: Forward → one local Yield for the whole CR3 wave. Retry owner: T018's own counter, starting 0.
- Execution order: **wave anchor**, member 1; lands before `T019` (both edit `_stream_single_request` at
  different lines — rebase sequence, not interference); prerequisite of the MS-001 boundary close, ahead of the
  assumptions recheck and Yield Red. No `--before` referent exists (no pending planned task).
- Verification profile (proportional): F-01 — medium risk (app-wide request surface), easy recovery (one
  kwarg), low uncertainty (mechanism re-verified on installed aiohttp + the review's reproduced 413), gate:
  focused route witnesses + full hermetic suite, budget: rides the wave's single Yield. F-03 — medium
  (stream-path timing behavior), low uncertainty (base line + registry claim both cited), gate adds the
  slow-stream witness + the unchanged-arm replay.

**Amended 2026-10-06 by the run-42 composition routing pass ([082-assumption-decide](iterations/082-assumption-decide.md)).**
Three composed findings land on `T018`; **all three are absorbed with zero work beyond the done-when clause above**,
because the repairs 080 routed already cover them — this sub-block is the transmission of *why*, so Forward does not
have to go find it. Nothing above stands less than it did.

```markdown
Assumption finding: seam-S16 (S16-a) — missing contract: what the configured `client_max_size` counts, and where the cap measures
Evidence: assumptions/compositions/S16-body-limit.md §What Is Required (a)
Disposition: absorbed
Status: pending
Counterexample: none — the seam's verdict is `UNCLEAR`, not `COUNTEREXAMPLE`. The two readings the record names:
  (i) `web_request.py:677` compares with `>=`, so the configured value is the **smallest rejected** size and the
  largest accepted body is `N−1`; (ii) the gate sits at the handler's **first** read, ahead of `normalize()`
  (`messages.py:36`) and `strip_and_cache_images()` (`:40-41`), so with `--strip-images` on, an image-bearing
  `/v1/messages` body over 1 MiB is answered 413 although the body the proxy would forward is far smaller — the
  feature that exists to move image bytes out of the forwarded body never sees the request.
Required behavior: no request is answered 413 for body size at any of the four route classes, on any framing, at
  any size — the base-parity posture `DC-CR3-F-01-r1` ranked and this task's step 1 implements.
Done when: step 1's four-route-class >1 MiB witnesses are red at pre-fix bytes (413) and green after; no new
  criterion is added by this finding, and none may be met by choosing a number (`N`, `N+1`, or "documented").
Yield replay: the 2 MiB POST ×4 on the landed bytes; additionally one image-bearing >1 MiB body with
  `--strip-images` enabled, which must be read and forwarded rather than rejected.
Assumption impact: scoped:S16
Invariant cross-reference: Neutral (zero active/wip registry entries govern a request cap; `DC-CR3-F-01-r1`'s
  non-authoritative candidate stands unchanged by this row)
Invariant candidate: none new
Closure ledger: DC-S16-a-r1
```

```markdown
Assumption finding: seam-S17 (S17-CE-1) — the one armed deadline component raises the class the call site's only arm cannot catch
Evidence: assumptions/compositions/S17-stream-deadline.md §Counterexample CE-1
Disposition: absorbed
Status: pending
Counterexample: `POST /v1/messages {"stream": true}` with the upstream answering headers later than 600 s. The
  `TimeoutHandle` armed from `server.py:67`'s `total=600` fires inside `with timer:` around the header wait
  (`client.py:622-624`), raises bare `asyncio.TimeoutError` (`helpers.py:696-714`, passed through unwrapped at
  `client.py:793-798`), and `messages.py:165`'s `except aiohttp.ClientError` does not match (330's exhaustive set;
  331's not-a-`ClientError` half). The site's declared answer — 502 + `{"error": "Upstream request failed"}` +
  `"Upstream error: %s"` — is never produced. **Severity as corrected by this pass:** what the client gets instead is
  aiohttp's **generated 504** with a `text/plain` body (`verdicts/claim-230.md`: `web_protocol.py:518`), plus a
  traceback log and `force_close()` — not an abandoned request. Same input class after the commit at `:161`: the same
  class is swallowed by `_relay_after_commit`'s `except Exception` and reaches the client as a 200 with a truncated
  SSE body — documented, cleared behavior (`N5`), cited here only for the asymmetry.
Required behavior: the streaming relay lives as long as the upstream streams, with **no** armed deadline on that
  call — step 2's `ClientTimeout(total=None)` under row 57's replace-not-merge, i.e. base's posture
  (`buster_ripper_legacy.py:664`), not an arm that catches the timeout class.
Done when: step 2 as written, **plus the new clause** — the witness pins all four components `None` on the operand
  actually passed to `session.post`. The existing short-total truncation witness and the non-stream-unchanged
  companion witness are unchanged by this finding.
Yield replay: read the landed `timeout=` operand for the four-`None` assertion, and confirm from the diff that
  `messages.py:165`'s arm and `:192`'s inner swallow are byte-unchanged — this finding is closed by removing the
  event, and any widening of either arm is a re-composition trigger (`S17` `Invalidates On` 5/6).
Assumption impact: scoped:S17
Invariant cross-reference: Neutral
Invariant candidate: "a catch list names classes, not a semantic kind — `except <library error>` does not own 'the
  upstream failed' until the deadline classes that library raises outside its subtree are accounted for" —
  non-authoritative, for Interview; not written here
Closure ledger: DC-S17-CE-1-r1
```

```markdown
Assumption finding: seam-S17 (S17-CE-2) — one configured number reaches one arm as two semantic kinds, decided by network phase
Evidence: assumptions/compositions/S17-stream-deadline.md §Counterexample CE-2
Disposition: absorbed
Status: pending
Counterexample: `--upstream https://…` (`main.py:62-63` → `config.UPSTREAM`, `server.py:65` carrying
  `ssl=not config.SKIP_TLS_VERIFY`). (a) The TLS handshake is the binding constraint: the handshake-scoped
  `asyncio.TimeoutError` is caught at `client.py:737-740` and re-raised as `ConnectionTimeoutError`, which **is** a
  `ClientError` → `messages.py:165` matches → the 502 and JSON body are produced. (b) Same upstream, handshake fast,
  header wait over the same `total`: bare `asyncio.TimeoutError` → no match → CE-1's escape. One number, one call
  site, one arm, two classifications of "the upstream timed out", only one of which the site's contract covers —
  and `messages.py:180-185`'s docstring states the split as a premise while `:165` ignores it.
Required behavior: unchanged from CE-1's — no armed deadline on the streaming call, which disarms **both** kinds
  (the handshake scope is fed by `timeout.total`, row 396), so the phase-determined class split cannot recur here.
Done when: the four-`None` assertion above, and `T018`'s already-permitted docstring correction carried out as
  written — the `ClientTimeout.total`-as-in-path-failure-source clause is corrected to the new configuration
  **without** asserting anything false about the class hierarchy (the hierarchy stays as `:180-185` describes it for
  the post-commit arm, which this task must not touch).
Yield replay: read the docstring diff — it must state the new configuration, not a corrected class relation; and
  confirm no `connect`/`sock_connect`/`sock_read` component appeared anywhere in the diff.
Assumption impact: scoped:S17
Invariant cross-reference: Neutral
Invariant candidate: shared with CE-1's (one statement covers both)
Closure ledger: DC-S17-CE-2-r1
```

### T019 Planned Task Context — Normalize before keying at count_tokens; make every body-parse arm total (CR3 wave, member 2)

- Finding ids: **CR3-F-02** (P1, `routes/count_tokens.py:80`, + its attached test-validity claim) +
  **CR3-F-04** (P2, `routes/messages.py:276` + the `LookupError` clause). Evidence path:
  `iterations/079-codex-review.md` §CR3-F-02/§CR3-F-04 (filled dispositions carry this pass's re-verification).
- Disposition: **absorbed** — ledgers `DC-CR3-F-02-r1`, `DC-CR3-F-04-r1`.
- Status: pending
- Required behavior:
  1. `/v1/messages/count_tokens` resolves the session **by the same digest `/v1/messages` records stats
     under**: derive `sid` after normalization, mirroring `messages.py:36-37` —
     `body, _ = normalize(body, session_id="")` before `session_id(body)` (the empty `session_id` keeps the
     call image-inert — `normalize.py:202-210` gates image stripping on a non-empty one; everything else the
     call does is the pure dict surgery the ordering invariant is about). Bound to
     `EI:buster_ripper.session/session_id/derived_after_normalize` — the session-carried preserve invariant
     (plan §Invariant Reconciliation; handoff §Notes; `invariants/index.md` §Source Sessions records the
     deliberate non-registration). Under the documented client the billing nonce "changes on every request"
     (`normalize.py:11-15`), so today the stats lookup misses every time, `avg_latency` stays 0.0
     (`count_tokens.py:89`) and `should_nudge_compact`'s `per_session` arm is permanently false
     (`compaction.py:85-86`) — the endpoint exists to fire that arm.
  2. **No body can 500 the proxy through a decode failure.** (a) `messages.py:276`'s overflow probe decodes
     leniently (`errors="replace"` — one word; the relay at `:292-296` already carries raw bytes; the ASCII
     overflow text survives replacement); (b) `messages.py:143-149`'s stream error arm wraps its `resp.text()`
     in `except (UnicodeDecodeError, LookupError)` falling back to `read()` + `errors="replace"` — observable
     shape unchanged for every body that decodes today, the proxy-500 path gone; (c) the three client parses
     (`messages.py:30`, `chat.py:31`, `count_tokens.py:22-25` `_read_body`) add `LookupError` to their existing
     decode catches and answer each site's existing 400 shape (unknown codec raises `LookupError`, not
     `UnicodeDecodeError` — disjoint `ValueError` subtrees; `web_request.py:686-692` mechanism verified).
  3. Test-validity half of F-02: the `"test_session"` literals seeding `session_mod._stats` at
     `tests/test_routes_integration.py:254-333` become `session_id()`-derived digests so the stats path is
     actually exercised; `test_count_tokens_inflates_on_compaction_trigger`'s forced `_kv_cache_usage=0.85`
     stays as the **kv-arm** case, and a new **session-arm** witness (correct-digest stats, latency ≥ 15 s,
     fill ≥ 0.75×200K, KV low) covers the branch the ordering defect made unreachable.
- Counterexamples (Yield-replayable): (1) one raw body carrying `x-anthropic-billing-header: …` to
  `/v1/messages` (records stats under the post-normalize digest) then to `/v1/messages/count_tokens` — today it
  answers the estimate with `avg_latency 0.0`; after, it answers the recorded path, and with seeded high-fill/
  high-latency stats the inflation fires **without** KV pressure. (2) upstream answers 500 with body
  `b"\xff\xfe\x00garbage"` (or `Content-Type: application/json; charset=bogus`) → today the proxy raises out
  (traceback, proxy-500); after, upstream status+bytes relay. (3) client POSTs `charset=bogus` to each of the
  three POST routes → 400 "Invalid JSON", never 500.
- Touch-set: `buster_ripper/routes/count_tokens.py` (`handle_count_tokens` + `_read_body`),
  `buster_ripper/routes/messages.py` (client arm `:30`, stream error arm `:143-149`, non-stream probe `:276`),
  `buster_ripper/routes/chat.py` (client arm `:31`), witnesses in `tests/test_routes_integration.py` + new
  fail-firsts. **Must not touch:** `_chat_answer`/`_chat_response` (T015's already-correct siblings),
  `_relay_after_commit`, `normalize.py`, `session.py`, `_HOP_BY_HOP`, the dump path, `forward_headers`.
  Two `F-15` lint hits (`test_routes_integration.py:308` F401, inside the seed-replacement range) may be
  consumed as touch-set pickups — sanctioned, not required.
- Done when: fail-first witnesses green for all three required behaviors (each red at base bytes — the digest
  equality witness, the session-arm witness, the two undecodable-upstream relays, the three bogus-charset 400s);
  the two seeded count_tokens tests now pin real digests with no weakened assertions; existing
  `test_count_tokens_rejects_undecodable_json` parametrization and the 33-test route module green; full hermetic
  suite green; LNCP not worsened — `handle_count_tokens` is **L:17 N:1 C:3** post-drain (checkpoint 154), the
  added normalize call keeps it there or the capacity note lands with the change; `handle_messages`'s hard
  **C:11** (§RM-001's own decision) must not move.
- Yield replay: replay counterexamples 1-3 independently; confirm from the diff that `normalize.py`/`session.py`
  are byte-unchanged (the fix conforms the site to the invariant; it never edits the rule); confirm the
  session-arm witness passes with `compaction._kv_cache_usage` at 0.0 (proves the per-session trigger is alive,
  not another kv-forced pass).
- Assumption impact: required:new-surface **→ scoped:S18 as of**
  [082-assumption-decide](iterations/082-assumption-decide.md) (080 recorded it before the seam existed; 081's
  `claim-399` amendment already said so inside its own block — `S18` now holds open until its post-`T019`
  composition is `SAFE`, and that composition additionally needs the `MC-2` atom the collector now owes).
- Invariant cross-reference: **Aligned** — `EI:buster_ripper.session/session_id/derived_after_normalize`
  (session-carried preserve; task bound to the handle; no duplicate candidate; registry untouched — triad).
  This is deliberately **not** Contradiction: the finding proposes enforcing the carried rule, the code is what
  violates it (base origin verified at `legacy:640-655` + legacy `count_tokens`, and ranked below the carried
  invariant). F-04: **Neutral**. Invariant candidate: none.
- Closure ledger: `DC-CR3-F-02-r1`, `DC-CR3-F-04-r1`.
- Owning route: Forward → the same single local Yield for the CR3 wave. Retry owner: T019's own counter, 0.
- Execution order: member 2; runs after `T018` (shared-file rebase note, disjoint functions otherwise);
  prerequisite of the MS-001 close.
- Verification profile: F-02 — medium risk (billing/session keying on the exposed route), easy recovery,
  **low** uncertainty now (the harm was re-derived from source this pass, not inherited from the relay), gate:
  route-level witnesses + full suite. F-04 — low-medium (arms strictly become total where they 500'd), gate:
  the three counterexample classes. Budget: rides the wave's Yield.

**Amended 2026-10-06 by the run-40 atomic routing pass ([081](iterations/081-assumption-decide.md)).** Everything
above stands except what this sub-block corrects: `T019` step 2(b) was drafted on a premise row 399 falsifies. Forward
receives only session id + task id, so this block — not a path — is the transmission.

```markdown
Assumption finding: claim-399
Evidence: assumptions/verdicts/claim-399.md (read against assumptions/verdicts/claim-401.md)
Disposition: absorbed
Status: pending
Counterexample: upstream answers the streaming request 500 with `Content-Type: application/json; charset=bogus` and a
  **valid-UTF-8** body (`b'{"error":"boom"}'`). The blueprint expects `LookupError` out of `messages.py:144` there;
  aiohttp suppresses its own inside `get_encoding()` (`contextlib.suppress(LookupError, ValueError)` around
  `codecs.lookup`), falls back to `utf-8` for `application/json`, and `text()` returns normally — so a fail-first
  witness written on that input is green **before** the fix and can never go red. The live class at that site is
  `UnicodeDecodeError`, from the same call when the body is undecodable as the fallback: `500` with
  `b"\\xff\\xfe\\x00garbage"` → `UnicodeDecodeError` (not an `aiohttp.ClientError`) → escapes `_stream_single_request`'s
  single arm → proxy 500 with a traceback.
Required behavior: unchanged in outcome, corrected in mechanism — no upstream answer can 500 the proxy through a
  decode failure. Step 2(b)'s catch list stays exactly as approved, `except (UnicodeDecodeError, LookupError)` with the
  `read()` + `errors="replace"` fallback: `LookupError` is retained there as defense (399 names the caller's explicit
  `encoding=` argument as the one live path to it — `messages.py:144` passes none today) and the totality the plan
  requires comes from the `UnicodeDecodeError` half. Steps 2(a) and 2(c) are **unaffected**: 2(c)'s three client-parse
  clauses are justified by 401 (`HOLDS` — `web.Request.json()` really does raise a bare `LookupError` on an unknown
  charset), and 2(a) rests on 402/403/404. The two directions of this boundary are asymmetric and the code must not be
  written as though they matched.
Done when: the response-side fail-first is driven by a body undecodable under the **fallback** charset (red at pre-fix
  bytes with `UnicodeDecodeError`, green after, upstream status + bytes relayed), the string `charset=bogus` no longer
  appears as a `LookupError` trigger anywhere in `T019`'s witnesses or rationale, and one added witness asserts the
  previously-falsified case as a **no-raise relay** (unknown response charset over a UTF-8-decodable error body → the
  upstream's status and bytes answer, never a proxy 500), so the asymmetry is pinned by a test rather than by prose;
  step 2(c)'s three bogus-charset client 400s and step 2(a)'s probe witness keep their existing criteria unchanged; full
  hermetic suite green and LNCP not worsened, as above.
Yield replay: replay both response-side classes on the landed bytes and read the exception class in the pre-fix
  failure — `UnicodeDecodeError`, not `LookupError`, is what proves the witness went red for the right reason — and
  replay the three client-side bogus-charset 400s, where a bare `LookupError` **is** the expected pre-fix class.
Assumption impact: scoped:S18 (080 recorded `required:new-surface` before the seam existed; `S18` now holds open until
  its post-implementation composition is `SAFE`)
Invariant cross-reference: Neutral — zero `active`/`wip` registry entries reach a charset arm; the session-carried
  `EI:buster_ripper.session/session_id/derived_after_normalize` binds step 1 and is untouched by this amendment, and
  080's **Aligned** classification for `CR3-F-02` neither extends here nor weakens
Invariant candidate: "a proxy's decode arms are written per boundary direction, because aiohttp's client and server
  halves of one charset failure raise different classes" — non-authoritative, for Interview; not written here
Closure ledger: DC-claim-399-r1
```

**Amended 2026-10-06 by the run-42 composition routing pass ([082-assumption-decide](iterations/082-assumption-decide.md)).**
Two composed findings land on `T019` and both are absorbed with **zero** change to its steps, touch-set or done-when:
step 2(c) is already `S18-CE-1`'s repair word for word, and step 2(b) — after 081's `claim-399` amendment — is
already `MC-1`'s. Recorded here so the seam's post-implementation composition can cite the coverage rather than
re-derive it.

```markdown
Assumption finding: seam-S18 (S18-CE-1) — a client naming a codec Python does not have escapes the 400 its body gets under either neighbouring spelling
Evidence: assumptions/compositions/S18-decode-failure-surface.md §Counterexample CE-1
Disposition: absorbed
Status: pending
Counterexample: `POST /v1/messages` with `Content-Type: application/json; charset=x-not-a-real-codec` and a
  well-formed JSON body (`{"model": "x", "messages": []}`). `web_request.py:689` keeps the non-empty name (the
  `or "utf-8"` arm does not rescue it) and `:690` decodes with it, `codecs.lookup` raises `LookupError` (405 + 401),
  `text()`/`json()` contain no `try`, and `messages.py:30`'s tuple names only `json.JSONDecodeError` and
  `UnicodeDecodeError` — 406/407 put `LookupError` on a branch off `Exception` unrelated to either. The handler
  never returns; `logging_middleware` catches nothing. Byte-identical shape at `chat.py:30-32`; sharpest at
  `count_tokens.py:19-25`, where *both* neighbouring arms settle the request (`{}` fallback for syntax,
  `HTTPBadRequest` for undecodable bytes, 137 settles it as a real 400) and the unknown-codec arm leaves
  `_read_body`, `handle_count_tokens` and the middleware with nothing to catch it.
Required behavior: step 2's headline — **no body can 500 the proxy through a decode failure** — with this site set
  answered by each site's existing 400 shape.
Done when: step 2(c) as already written, unchanged; its witnesses red at pre-fix bytes with the bare `LookupError`,
  green after answering 400 at all three POST routes.
Yield replay: step 2(c)'s three bogus-charset 400s, and read the pre-fix exception **class** in each failure —
  `LookupError`, not `UnicodeDecodeError`, is what proves the witness went red for the right reason (the mirror of
  the response-side rule 081 wrote for step 2(b)).
Assumption impact: scoped:S18
Invariant cross-reference: Neutral (zero active/wip entries reach a charset arm; `EI:.../derived_after_normalize`
  binds step 1 and is untouched)
Invariant candidate: none new — 081's direction-asymmetry candidate already covers this boundary
Closure ledger: DC-S18-CE-1-r1
```

```markdown
Assumption finding: seam-S18 (S18-MC-1) — missing contract: whether `messages.py:165`'s arm covers a decode failure raised by aiohttp's own charset step
Evidence: assumptions/compositions/S18-decode-failure-surface.md §MC-1 (answered from `DC-claim-399-r1`, this session's own settled record)
Disposition: absorbed
Status: pending
Counterexample: the upstream answers the streaming request ≥ 400 with `Content-Type: application/json;
  charset=utf-8` and a body carrying an invalid byte sequence; `messages.py:144`'s `await resp.text()` is strict
  (398 + 402), so `UnicodeDecodeError` is raised inside the `try` at `:136`. The composition cannot decide the
  *fate* from its member set (no `S18` member speaks to `aiohttp.ClientError`'s hierarchy — 137 covers
  `HTTPException`, 68–72/406/407 the built-in side). This session's record does decide it: `DC-claim-399-r1` names
  the class as **not** a `ClientError`, the escape as reaching `_stream_single_request`'s single arm, and the
  outcome as a proxy 500 with a traceback. `compaction.py:41`/`:55` is the contrast — same call shape, covered by
  its wide arm, which is why only the relay site is open.
Required behavior: step 2(b) as approved and re-aimed by 081 — an upstream answer that cannot be decoded is relayed,
  never answered from the proxy with a 5xx.
Done when: step 2(b) as written, unchanged, including 081's requirement that the fail-first be driven by a body
  undecodable under the **fallback** charset and that the previously-falsified `charset=bogus` case be pinned as a
  no-raise relay.
Yield replay: step 2(b)'s two response-side classes on the landed bytes, reading the pre-fix class as
  `UnicodeDecodeError`.
Assumption impact: scoped:S18
Invariant cross-reference: Neutral
Invariant candidate: none new
Closure ledger: DC-S18-MC-1-r1
```

**One contingency `S18` leaves open, and it is not Forward's to close.** `MC-2` (the union's fourth arm — a charset
name that *is* registered but is not a text encoding, `base64`/`hex`/`rot_13`) is routed `malformed` to the
collector, because no claim in the ledger states which class `bytes.decode(<that name>)` raises and this pass does
not check claims. **Do not add a fourth name to step 2(c)'s catches on speculation.** If the atom comes back as a
class outside `{JSONDecodeError, UnicodeDecodeError, LookupError}`, step 2(c) is total over the classes the
request direction *really* produces and the fix is one name on three sites in a follow-up; if it comes back as one
of the three, the composition certifies `SAFE` on this wave's work. `T019`'s wording must keep saying "no body can
500 the proxy through a **decode** failure" for the class set it has verified — the totality claim is the seam's to
earn post-implementation, not this task's to assert now.

### T020 Planned Task Context — Drive `poll_kv_cache` in the EMA witnesses; cover `compaction.py:49` (CR3 wave, member 3)

- Finding id: **CR3-F-05** (P2, `tests/test_compaction.py:348`, `:369`). Evidence path:
  `iterations/079-codex-review.md` §CR3-F-05 (both test bodies and `compaction.py:44-52` re-read by this pass).
- Disposition: **absorbed** — ledger `DC-CR3-F-05-r1`.
- Status: pending
- Required behavior: the two tests named `test_ema_*` verify the **production** EMA path, not their own
  arithmetic: drive `compaction.poll_kv_cache` against a mocked session whose `/metrics` answers arrive ≥2×
  (first tick → `_kv_cache_usage == raw1` — the `first` branch; second tick → the `(1-α)·prev + α·raw2`
  composition — covering `compaction.py:49-52`, the coverage report's single missing line), restoring
  `compaction._kv_cache_usage` (and any module globals touched) afterward. The inline-formula bodies
  (`result1 = (1 - alpha) * prev + alpha * 0.75`; `assert result == raw`) go away; the tests keep their names'
  promise: *reduces bounces* asserted across three distinct raw readings through the loop, *first reading is
  raw* asserted as the loop's first tick.
- Counterexample (why the current pair is false evidence): mutate `compaction.py:49-52` (e.g. swap to
  `(1 - _KV_EMA_ALPHA) * raw + _KV_EMA_ALPHA * _kv_cache_usage`) — today both tests stay green; that mutation
  is the Yield replay.
- Touch-set: `tests/test_compaction.py` only. **Must not touch:** `compaction.py` production bytes, the
  module's other poll tests (timeout/first-branch arms).
- Done when: rewritten witnesses green; `pytest tests/test_compaction.py -k ema --cov=buster_ripper.compaction
  --cov-report=term-missing` shows **no missing line at :49** (`compaction.py` 100% or its missing-list empty);
  mutation replay documented red in the report; full hermetic suite green (455+ @ ≥95.75%); the `F-15` hit
  `test_compaction.py:373 F841` is consumed by this rewrite (its variable disappears — sanctioned touch-set
  pickup); LNCP not worsened.
- Yield replay: run the mutation yourself on a throwaway copy — the rewritten witness must go red; re-run
  coverage and confirm :49 is executed.
- Assumption impact: required:new-surface (tests-only; no production surface added).
- Invariant cross-reference: Neutral. (Reach Gate not applicable — false evidence holds for every input; the
  production branch does run in the shipped process, `server.py:15-34` → poll every 10 s.)
- Closure ledger: `DC-CR3-F-05-r1`.
- Owning route: Forward → same wave Yield. Retry owner: T020's own counter, 0.
- Execution order: member 3; file-disjoint from T018/T019 — no ordering constraint within the wave.
- Verification profile: low risk (test file only), low uncertainty (defect read in-source), gate: focused
  `-k ema` + coverage delta + full suite; budget: rides the wave's Yield.

**Amended 2026-10-06 by the run-40 atomic routing pass ([081](iterations/081-assumption-decide.md)).** The rewrite
above was drafted against a premise row 409 falsifies, and the falsification does not remove the need for a double —
it removes the *evidence value* of a double that merely works. Forward receives only session id + task id, so this
block — not a path — is the transmission.

```markdown
Assumption finding: claim-409
Evidence: assumptions/verdicts/claim-409.md (+ assumptions/verdicts/claim-408.md, claim-410.md, claim-411.md,
  claim-126.md; sites `buster_ripper/compaction.py:34-57`, `tests/test_compaction.py:182-195`, `:229-245`, `:348`, `:369`)
Disposition: absorbed
Status: pending
Counterexample: with `session.get` returning a plain `MagicMock`, `async with session.get(...) as resp` **succeeds** —
  `_async_method_magics` is folded into `MagicMock`'s magic set and `Mock._get_child_mock` routes an async magic to an
  `AsyncMock` child, with `_return_values['__aexit__'] = False` so the body runs. What it binds to `resp` is a
  **newly auto-created child**, not the object the test configured, so `resp.status == 200` is False,
  `compaction.py:40-52` never runs, `_kv_cache_usage` never changes, and `except Exception: pass` at `:55-56` absorbs
  every other fixture failure too — including the two shapes that *do* raise (`async with Mock()` /
  `NonCallableMock()` → `TypeError`; a `MagicMock(spec=…)` or `autospec` response double whose spec prunes the dunder).
  A witness that asserts "two ticks completed without raising", or asserts only call counts, is green with `:49` never
  executed — and `DC-CR3-F-05-r1`'s own mutation replay would stay green with it.
Required behavior: unchanged — the two `test_ema_*` must verify production's EMA path, first tick raw and the
  `(1-α)·prev + α·raw` composition on the second and third, across ≥2 mocked `/metrics` ticks, with module globals
  restored; `compaction.py:49` must stop being a missing line. What 409 adds is the *capability condition*: the witness
  must be structurally unable to pass on an inert double, because "the double supports `async with`" is no evidence of
  anything.
Done when: every one of these holds on the rewritten pair —
  (1) **per-tick value assertions** on `compaction._kv_cache_usage`: tick 1 equals raw1 exactly; tick 2 equals
  `(1 - α)·raw1 + α·raw2` with the expectation written as an independent literal expression (never re-derived from or
  mirroring the source line's operand order — that is the self-proving shape `DC-CR3-F-05-r1` already rejected); tick 3
  carries the three-distinct-raws bounce claim the test's name promises;
  (2) **one awaited-channel precondition** — the awaited `text()` is asserted once per tick — so a wrong double reads as
  "the peer never delivered a 200 body", not as a production pass;
  (3) **forbidden double constructions**: bare `Mock()`/`NonCallableMock()` as the context manager, and any
  `spec=`/`patch(..., autospec=True)` double of the response type whose spec prunes `__aenter__`/`__aexit__`; an
  unconfigured `MagicMock` standing where the response belongs;
  (4) **permitted construction**: the file's own sibling idiom (`tests/test_compaction.py:182-195`, `:229-245`) —
  `mocker.AsyncMock()` response with `status`/`text` set and `__aenter__`/`__aexit__` configured,
  `mocker.Mock(spec=aiohttp.ClientSession)` session with `get` a plain `Mock` returning it — or a `MagicMock` whose
  `__aenter__.return_value` **is** the configured response;
  (5) the mutation replay (swap the weights at `compaction.py:49-52`) goes red **on a value assertion**, and the report
  names the failing assertion — a red on a call count, or an error escaping the swallow, does not satisfy this;
  (6) async witnesses rely on 411 (`asyncio_mode = "auto"` — no explicit marker needed), reach tick 2 without a real
  `KV_POLL_INTERVAL` sleep (410's yield, the cancel-and-await arm 126 licenses), and restore
  `compaction._kv_cache_usage`;
  (7) the existing criteria are kept: no missing line at `:49` under `-k ema --cov=buster_ripper.compaction`, the
  `F-15` hit at `:373` consumed by the rewrite, full hermetic suite green (455+ @ ≥95.75%), LNCP not worsened.
Yield replay: run the weight swap on a throwaway copy yourself — the witness must go red on the tick-2/tick-3 value
  assertion and Yield records which assertion fired; then re-run with the double's `status` forced to a non-200 value
  and confirm the witness goes red too (a witness that stays green there is not observing production at all); re-run
  coverage and confirm `:49` executes.
Assumption impact: scoped:S19 (080 recorded `required:new-surface` before the seam existed; `S19` now holds open until
  its post-implementation composition is `SAFE`)
Invariant cross-reference: Neutral — zero `active`/`wip` registry entries reach a test double; the eight `planned` PIs
  are dump-format and TLS-posture rules in unrelated packages
Invariant candidate: "a witness of an async loop is evidence only when it asserts the loop's computed state, never when
  it asserts that a double was usable" — non-authoritative, for Interview; not written here
Closure ledger: DC-claim-409-r1
```

**Amended 2026-10-06 by the run-42 composition routing pass ([082-assumption-decide](iterations/082-assumption-decide.md)).**
`S19` composed three counterexamples against the pair of witnesses 080 routed and 081 re-aimed. Two are covered
exactly as the blueprint now reads; **the third contradicts one clause of it**, and that clause is corrected below.
Everything in the two blocks above stands except the item marked as superseded.

**Superseded clause — `T020`'s done-when item (6), third part.** What 081 wrote: reach tick 2 "without a real
`KV_POLL_INTERVAL` sleep (410's yield, the cancel-and-await arm 126 licenses)". `S19-CE-3` establishes that this is
not satisfiable as stated: 410's verified guarantee is that `asyncio.sleep(0)` lets another **ready** task run, and
its own verdict limits itself the same way ("the guarantee is 'allows other tasks to run', not 'a particular task
will run'"), while the loop parks on `await asyncio.sleep(config.KV_POLL_INTERVAL)` at `compaction.py:57` with the
interval read from `config.py:64` as `10`. A task parked on a 10 s timer has no ready work inside a 50 ms window, so
`compaction.py:48-52` is never entered and the tick-2 assertion has no second tick to assert on.
**Corrected requirement (this is now the criterion):** the witness sets `config.KV_POLL_INTERVAL` to `0` **for the
duration of the driven task** — the attribute is read per iteration at `:57`, so the loop's own sleep becomes a
yield — and restores it with `compaction._kv_cache_usage`, in the same `try/finally` shape the current criteria
already require. It must then *observe* tick 2 rather than assume it: items (1) and (2) already enforce that (a
witness that never reaches the second iteration cannot produce a tick-2 value or a second awaited `text()`), and the
mutation replay in (5) stays red-on-a-value-assertion. 126 and 411 keep their roles for settlement and collection;
what 410 no longer carries is the *timer*.

```markdown
Assumption finding: seam-S19 (S19-CE-1) — `async with` succeeds and the configured payload is discarded
Evidence: assumptions/compositions/S19-async-witness.md §Counterexample CE-1
Disposition: absorbed
Status: pending
Counterexample: the sibling double shape with its two `__aenter__`/`__aexit__` self-binding lines removed
  (`session.get = mocker.Mock(return_value=resp)`, `resp.status = 200`, `resp.text = mocker.AsyncMock(return_value="vllm:kv_cache_usage_perc{engine=\"0\"} 0.75\n")`).
  Per 415/408 `async with resp:` **succeeds** and binds `resp.__aenter__.return_value` — an auto-child carrying none
  of those attributes — so `compaction.py:40`'s `== 200` does not match (or raises inside the body and `:55-56`
  consumes it), `:44-52` never runs, `:57` sleeps, and the observable is `session.get.called` **True** with
  `_kv_cache_usage` **unchanged** — exactly the assertion vocabulary `:208`, `:282` and `:311` already have, while
  the branch that feeds the compaction nudge never ran. Ordering variant: setting `session.get.return_value` on the
  spec'd session instead of replacing `get` yields a non-`MagicMock` child, `async with` raises `AttributeError`
  inside the `try`, and `:55-56` swallows it — identical observable.
Required behavior: 081's capability condition, unchanged — the witness must be structurally unable to pass on an
  inert double, and must assert the loop's computed state per tick.
Done when: items (1)–(7) as already written; this finding adds no criterion, and no witness may satisfy them with a
  binding removed (item (4) names the self-binding as the permitted construction for this reason).
Yield replay: the weight swap must go red on a value assertion, and the forced-non-200 negative control must go red
  too — a witness green under a discarded payload is CE-1 reproduced, not a pass.
Assumption impact: scoped:S19
Invariant cross-reference: Neutral
Invariant candidate: 081's statement covers it; none new
Closure ledger: DC-S19-CE-1-r1
```

```markdown
Assumption finding: seam-S19 (S19-CE-2) — the sync arm and the awaited arm of one response object need different mock kinds
Evidence: assumptions/compositions/S19-async-witness.md §Counterexample CE-2
Disposition: absorbed
Status: pending
Counterexample: CE-1's shape **with** the self-binding (so `resp` is the configured object and `resp.status = 200`
  passes the sync read at `:40`), but `resp.text = mocker.Mock(return_value=…)` — a **sync** Mock, the kind the same
  tests already use for `session.get` at `:195`. `:41`'s `await resp.text()` awaits a non-awaitable → `TypeError`
  inside the `if` body → `except Exception: pass` at `:55-56` → sleep. Observable identical to CE-1: call recorded,
  global unchanged, no signal of any kind.
Required behavior: unchanged — production's EMA path exercised across ≥2 ticks with per-tick value assertions.
Done when: items (1)+(2)+(4)+(5) as already written; (2)'s "the awaited `text()` is asserted once per tick" is the
  discriminator for this exact shape, so a wrong-kind double reads as "the peer never delivered a 200 body" rather
  than as a production pass.
Yield replay: as CE-1's; additionally, if a witness goes red with an un-awaited-coroutine or `TypeError` trace
  escaping the swallow, that is a fixture failure to fix in the double, not evidence to accept.
Assumption impact: scoped:S19
Invariant cross-reference: Neutral
Invariant candidate: none new
Closure ledger: DC-S19-CE-2-r1
```

```markdown
Assumption finding: seam-S19 (S19-CE-3) — no member grants a second tick of one `poll_kv_cache` call
Evidence: assumptions/compositions/S19-async-witness.md §Counterexample CE-3 + its "Named missing contract"
Disposition: absorbed (blueprint amended — the one finding of the fourteen this pass had to write into a task)
Status: pending
Counterexample: two distinct payloads queued (`side_effect=[resp_60, resp_20]`, both self-bound, status 200), task
  driven the way the file drives it, `await asyncio.sleep(0.05)` then `task.cancel()`. Tick 1 takes the `first` arm
  (`:45-47`) — `first` is function-local at `:31`, so the blend arm at `:48-52` needs a **second iteration of the
  same invocation** — then `:57` awaits `asyncio.sleep(config.KV_POLL_INTERVAL)` with `10` from `config.py:64`.
  410's guarantee is a yield, not a timer. Observable: `session.get.call_count == 1`, `_kv_cache_usage == 0.60`
  where the blend from a `0.0` seed is `0.7*0.0 + 0.3*0.6 = 0.18`; a witness asserting the blend **fails**, and a
  witness asserting "the value is no longer 0.0" **passes for the seeding arm**, which is not the behavior under
  test. The two tests named for this arithmetic (`:348`, `:369`) never call `poll_kv_cache` at all.
Required behavior: reach and observe tick 2 inside one driven call, by shortening the interval the loop itself
  reads — never by asserting less, and never by editing `compaction.py`.
Done when: the corrected item (6) above (interval `0` for the task's duration, restored in the same
  `try/finally`), plus the untouched items (1)–(5) and (7); `:49` still must not be a missing line under
  `-k ema --cov=buster_ripper.compaction`.
Yield replay: run the weight swap and confirm red on the **tick-2/tick-3 value assertion** — a witness that goes red
  on a call count or on an escaping error has not reached tick 2 either; then confirm the interval attribute is
  restored (a leaked `0` is a real-suite hazard: the monitor polls every 10 s in the shipped process).
Assumption impact: scoped:S19
Invariant cross-reference: Neutral
Invariant candidate: "a witness of a timer-driven loop is entitled to a second iteration only by a stated mechanism
  for the interval; a yield guarantee is not a clock" — non-authoritative, for Interview; not written here
Closure ledger: DC-S19-CE-3-r1
```



## Trimmed 2026-10-07 (`085-01-yield`) — `T019.001` accepted-local (context budget)

The block this pass consumed and the two narrative blocks it replaces, verbatim.

### Superseded §Session Metadata `Last Updated` entry

- Last Updated: 2026-10-07 (**CR3 wave `T018`+`T019`+`T020` ACCEPTED-LOCAL at [083-01-yield](iterations/083-01-yield.md)** — one local Yield for the whole wave on its complete range `d5df8b06…9c90675`: zero reject-tier findings, all three counters still 0, hermetic **471 @ 95.91%** re-run here, `codify scan` re-measured against a re-extracted `d5df8b06` baseline, the two `T020` replays run on a throwaway mutation and reverted byte-exact, 3 defer-tier rows added, and `claims.md` row 416 `HOLD` so `S18`'s `MC-2` atom is no longer missing. The two superseded `Last Updated` entries this line replaces are preserved verbatim in [handoff-archive.md](handoff-archive.md) §Trimmed 2026-10-07 (`083-01-yield`).)

### Superseded §Current Status — `T019.001` IMPLEMENTED, awaiting local Yield

- **T019.001 IMPLEMENTED — its own one-member wave, awaiting local Yield** (the run-45 remediation `084` routed
  from rows 418/422/423; its dispatch condition was the CR3 wave's acceptance at
  [083-01-yield](iterations/083-01-yield.md), which is met). Production is **one expression** at
  `messages.py:106` — `error_text = result.text` — plus the totality note done-when (5) prescribes:
  `git diff --numstat` is 1 file, **+12/−1** in one hunk inside `handle_streaming` — 11 of the additions are the note's prose, so **zero net executable lines**. No constructor moved:
  `_error_body`, `_stream_single_request`, `handle_non_streaming`, `_non_stream_single_request`,
  `_relay_after_commit` and the whole of `chat.py` (still **300/300**) are **AST-identical** to `c96ca3e2`.
- **Why it was dead, per the three rows**: `body=<str>` stores a `StringPayload` (418), `isinstance(.body, bytes)`
  is false and `str(.body)` is the object repr (423), so no `config.py:122-123` pattern could match and the loop
  returned the upstream's 400 after exactly one request; row 422 certifies `.text` as the only `Response`
  accessor that returns the string's content, and it round-trips here because `StringPayload` encodes utf-8 while
  `application/json` names no charset, so `.text`'s fallback is that same codec. Both objects the loop can be
  handed — the `:151-155` error payload and `:171-173`'s static 502 — read strictly, so nothing new raises.
- **Witnesses**: 5 route-level cases in `tests/test_context_overflow.py::TestStreamingOverflowRetry` — **2
  fail-first red at pre-fix bytes** (`assert 1 == 3`; the caller receiving the upstream's body where the loop's
  own `Context overflow after 2 retries` belongs), transcript in
  [evidence/t019-001-fail-first-run.txt](evidence/t019-001-fail-first-run.txt), re-captured against the **final**
  test bytes (twice: once after the recorder landed, once after the pointer correction below) — outcomes identical in all
  three runs, the *evidence* re-cut rather than the *result*. The 3 pins are green in both states by design: a non-overflow 4xx relays
  **once** with unchanged status/body and `Content-Type: application/json` with no charset (parametrized with the
  `\xff\xfe` hostile payload too), and the 502 crosses the probe without raising. Retry bodies are snapshotted
  **per call** because the loop rewrites one dict in place — the one thing added beyond the blueprint's wording,
  and one line to revert; see [085-01](iterations/085-01-forward.md) §Concerns 1.
- **Pointer discipline (Checkpoint 196, not 195).** This increment's own 12 inserted lines renumbered the file, so every
  blueprint coordinate at or below the old `:107` is **+11** here — the read is now `:117`, the terminal arm `:130-134`, the
  `:151-155` constructor `:162-166`, the `:171-173` 502 arm `:182-184`, the non-stream probe `:311`. Plan-side records keep
  quoting the plan's numbers; what must not is authored prose naming neighbours it just renumbered, which the first draft's
  in-source note and two witness docstrings did. They now name those arms **by role**. Same class 081 corrected at
  Checkpoint 189; [085-01](iterations/085-01-forward.md) §Pointer discipline holds the mapping.
- **Runs**: module **22 passed**; affected scope once → hermetic **476 @ 95.91%** (baseline 471 @ 95.91%; +5
  tests, none removed; `routes/*` stays coverage-omitted, so the repaired line is covered by the witnesses, not
  by the gate); `handle_streaming` re-measured **L:25 N:3 C:8 P:4 — unchanged**, done-when (6) met exactly; ruff
  `--select F` clean on both touched files; `messages.py` **323 → 334** raw lines, all +11 prose; `tests/test_context_overflow.py` **179 → 309**.
- **Backlog pickup** (full record in [085-01](iterations/085-01-forward.md)): the owed
  `QB:routes-messages/file-size-yellow` **misplacement check was run and declined with reasons** — the split moves
  `handle_streaming` itself and §RM-001 owns that design decision; `QB:test-context-overflow/file-size-yellow`
  added for this pass's own +129; `QB:report-accuracy/001-collect-b005-retry-as-tested` annotated but left
  `[ ]`, since its criterion is this task's **acceptance**, which is Yield's act. Two §RM-001-owned rows whose
  triggers fired were deliberately not absorbed.
- **Zero questions asked** — the pre-question gate had nothing to route. `FK-011` remains the session's only open
  Fork, untouched. No invariant written, superseded or promoted (triad rule); the blueprint's **Invariant
  candidate** stays a candidate in its block for Interview to rule on, now with a dependency table beside it.
- **Next**: local Yield for this one-member wave (the blueprint's §Yield replay items are executable on these
  bytes as written) → the MS-001 boundary's assumptions recheck on these bytes (owes the first recompositions of
  `S16`–`S19`, the row-57 re-pin, and the row-425 wording `S20` waits on; `084` places `T019.001` **before**
  `S20`'s recomposition, which this satisfies) → Yield Red (owns `resolved` for
  `T015`/`T016`/`T017`/`T018`/`T019`/`T020`, phase acceptance, the `Delivered` chronicle entry, parent
  completion) → `phase-boundary-finalize`.

### The accepted `T019.001 Planned Task Context` block

### T019.001 Planned Task Context — Make `handle_streaming`'s context-overflow probe read the upstream error text through the response's own text surface (`messages.py:106`)

- Owning route: own one-member wave, dispatched **only after the CR3 wave's local Yield (`T018`+`T019`+`T020`)
  accepts**. Not a member of that wave, and not a retry of `T019` — `T019`'s finding counter stays 0 and its landed
  increment diff range is untouched.
- Assumption finding: claim-418, claim-422, claim-423 (seam `S20`)
- Evidence: `assumptions/verdicts/claim-418.md`, `claim-422.md`, `claim-423.md`; reach determination
  `evidence/n6-reach-and-base-comparison.txt`; origin `assumptions/compositions/S18-decode-failure-surface.md`
  §Notes N6
- Disposition: absorbed
- Status: pending
- Counterexample: `POST /v1/messages` with `"stream": true` against a backend detected as `sglang` (`server.py:71`
  auto-detect, or `--backend-type sglang` — `main.py:129`); the upstream answers `400` with
  `{"error": "Context overflow: input 5000 > 4096 tokens, max_completion_tokens 4096"}`. `_stream_single_request`
  constructs `web.Response(body=<str>, status=400, content_type="application/json")` at `messages.py:151-155`, so
  `result.body` is a `StringPayload` and not bytes (row 418); `isinstance(result.body, bytes)` is false at `:106`;
  `str(result.body)` is `<aiohttp.payload.StringPayload object at 0x…>` (row 423); `parse_context_overflow_error`
  returns `None` at `:107`; the loop returns the upstream `400` at `:125` after exactly one upstream request.
  Verified by execution against the installed aiohttp 3.13.3 and the production helper (evidence file §4). The
  loop's other arm is inert too: the `ClientError` 502 at `:171-173` carries bytes but a static body that cannot
  contain the pattern — so the retry can never `continue` on any path.
- Required behavior: the probe receives the upstream's error **text**, so the retry reduces
  `max_completion_tokens` and re-issues exactly as `handle_non_streaming` already does at `messages.py:296-305` and
  as `chat.py:178` already does on the parsed form; the client-visible status, body bytes and `Content-Type` of a
  relayed upstream error stay byte-identical to today (`application/json`, no charset).
- Done when: (1) `messages.py:106`'s read is the response's own text surface and no constructor in the file changes;
  (2) fail-first route-level witness, red at pre-fix bytes, asserts `post.call_count == 3` and that the 2nd and 3rd
  request bodies carry the literal recomputed `max_completion_tokens` = `256` for the counterexample above (assert
  the value, never mirror the formula, and do not assert on the retry log line); (3) a witness pins that a
  non-overflow `>= 400` still relays **once** with unchanged status/body/`Content-Type`, so the fix cannot turn every
  4xx into a re-dispatch; (4) a witness reaches the terminal arm at `:119-123` (`overflow_err` truthy at
  `attempt == max_retries`), otherwise dead on both paths; (5) nothing new can raise out of `handle_streaming` from
  that read — the only two shapes `_stream_single_request` returns are the `StringPayload` response and the static
  `json_response` 502, and both `.text` reads are strict-decodable at those bytes; pin it with a witness or an
  in-source note naming both shapes; (6) `handle_streaming`'s metrics (`L:25 N:3 C:8 P:4`) do not worsen and the task
  performs the misplacement check `QB:routes-messages/file-size-yellow` owes at next touch of `messages.py` (323
  lines, inside the 301-500 review band); (7) `chat.py` (300/300), `_error_body`'s catch list, the `:151-155`
  constructor and the non-stream probe stay byte-exact.
- Yield replay: replay the overflow witness independently (`call_count == 3` + the literal `256`) and the
  non-overflow single-relay witness; then replay the mutation — restore
  `result.body.decode() if isinstance(result.body, bytes) else str(result.body)` and confirm the overflow witness
  goes red. A green suite with that mutation restored means the witness is fake.
- Assumption impact: required:new-surface — the repair sits at a repo call site no ledger row covers (run 45's own
  fence puts repo call-site shape outside the gate); `S20`'s members stay third-party and as verdicted.
- Invariant cross-reference: **Neutral** — `invariants/` has no `buster_ripper/routes/*` file, every registry entry
  touching this session is `Status: planned` (`PI:T002/dump_format/*`, `PI:T006/upstream_tls/*`), no active or wip
  invariant reaches this site, and there is no session `invariants.md`. Ranking comes from task-level settled
  material, not from an invariant. Base parity is **silent, not protective** here: `buster_ripper_legacy.py` contains
  no `overflow`/`sglang`/`retry`/`backend`/`max_completion` at all (evidence file §5).
- Invariant candidate: "Reading a body back off an `aiohttp.web.Response` the code constructed goes through `.text`;
  `body=` given a `str` stores a payload object, so `.body` is never that string's bytes." Candidate only — Interview
  owns recording/promotion; no agent on this route writes or promotes it.
- Closure ledger: `DC-claim-418-r1` (primary), `DC-claim-422-r1`, `DC-claim-423-r1`
- Rejected candidates (full reasons in the ledgers): construct with `text=error_content` (verified to store bytes,
  but it adds `; charset=utf-8` to the relayed `Content-Type` — a client-visible change on the arm this session's
  accepted relay-fidelity work governs); make `_error_body` return bytes (displaces `T019`'s approved catch-list
  contract for no gain over a consumer-side read); delete the retry loop as dead code (preserves the verified
  defect); accept as a risk (the Reach Gate's `reached` rule is fix-or-pin and nothing pins it); defer to the
  roadmap as out-of-scope (rule 5 places the fix where the cause is).
- Execution order: after the CR3 wave's local Yield accepts, and before `S20`'s recomposition in the MS-001
  assumptions recheck — which additionally needs row 419's reworded successor row (§Assumption routing — 2026-10-07).


## Trimmed 2026-10-07 (`090-01-forward`) — `T019.002` implemented (context budget)

The `## Current Status` section `090-01-forward` replaced, verbatim, and the `Last Updated` line above it.
Nothing here is owed by the remaining MS-001 boundary work; the dispositions it carries stay live in
§Deterministic Closure Ledger, §Open Forks / Retries and §Notes below.

### The replaced `Last Updated` entry

- Last Updated: 2026-10-07 (**`T019.001` ACCEPTED-LOCAL at [085-01-yield](iterations/085-01-yield.md)** — the run-45 remediation's own one-member wave, complete range `c96ca3e2…0c3a003` reviewed as one increment: zero reject-tier findings, all seven done-when clauses re-derived from bytes, the blueprint's §Yield replay executed by this pass (including the restore-the-ternary mutation → the two fail-first witnesses red, then reverted byte-exact), hermetic **476 @ 95.91%** re-run here against the 471 baseline, `handle_streaming` re-scanned at both ends (**L:25 N:3 C:8 P:4** — unchanged), `claim-418`/`-422`/`-423` arbitrated **HOLDS** at the installed 3.13.3 bytes and by execution, two defer-tier observations closed inside the pass, zero `invariants/`/`decisions.md` bytes in range, `FK-011` still the only open Fork.) The superseded `Last Updated` entries — this line replaces the CR3-wave acceptance and the `T019.001` IMPLEMENTED line — are preserved verbatim in [handoff-archive.md](handoff-archive.md) §Trimmed 2026-10-07 (`085-01-yield`).)

### The replaced `## Current Status` block

## Current Status

- **CR4 ROUTED (2026-10-07, [089](iterations/089-assumption-decide.md)) — one `absorbed`, one `accepted`, zero
  questions.** `CR4-F-01` (P2) → **`T019.002`**, a nested remediation under `T019` on `T019`'s own surface, one-member
  wave, dispatchable now: `count_tokens.py:87` narrows from the whole of `normalize()` to the system-strip step, which
  restores base's tolerance on that route and leaves `/v1/messages` byte-identical. `CR4-F-02` (P3, one byte at EOF) →
  `accepted`, `QB:tests/blank-line-at-eof`, owner = next Forward pass touching either file. **The reviewer's open half
  is ruled, not deferred:** `/v1/messages`' malformed-element 500 is base's own behavior at base's own `normalize()`
  (measured at `72e3a1c6`, `6508941` and `c5e2be4`), `G-003` holds that route still with "no normalization changes", and
  hardening the loops is 500 → 502 measured — so parity protects it, it is not absorbed, and no question is admissible.
  Owner named: `roadmap.md` §RM-012; §Accepted risks and §Elevated to user carry it; `T019.002`'s Done-when clause 5
  pins the current answers so the posture cannot be changed under cover of the repair. Every claim in this bullet was
  re-measured at three points of the comparison, not read off 088's table: `evidence/cr4-probe-matrix.txt`.
- **Ordering consequence for the orchestrator:** `T019.002` lands **before** the MS-001 assumptions recheck — that
  recheck is `S16`–`S19`'s single allowed post-repair recomposition and must run on final bytes. Then this wave's local
  Yield → the recheck → Yield Phase (re-executes `QAP-MS001` on the final bytes; 087's `READY-PASSING` is byte-scoped
  to `8dfa5af`) → Yield Red, which owns `resolved` for `T015`-`T020`, `T019.001` **and `T019.002`**.
- **`T019.001` ACCEPTED-LOCAL** — the SGLang **streaming** context-overflow retry fires. `messages.py:117`'s read is
  `error_text = result.text`; production is **+12/−1 in one hunk** inside `handle_streaming` (11 of them the totality
  note done-when (5) prescribes — **zero net executable lines**), no constructor moved, and `chat.py` (300/300),
  `_error_body`, `_stream_single_request`, `handle_non_streaming`, `_non_stream_single_request` and
  `_relay_after_commit` are AST-identical to `c96ca3e2`. `tests/test_context_overflow.py::TestStreamingOverflowRetry`
  adds 5 route-level witnesses (**+131/−1**, 179 → 309 lines): 2 red at pre-fix bytes — re-reddened by *this* pass's
  own mutation, not by inherited transcript — and 3 pins green in both states by design. Hermetic **476 @ 95.91%**.
  The mechanism, now certified at the shipped line: `body=<str>` stores a `StringPayload`, `str()` of it is the object
  repr the regexes cannot match, and `.text` is the one `Response` accessor returning the string's content — it
  round-trips because `StringPayload` encodes utf-8 while `application/json` names no charset.
- **What this Yield added beyond Forward's record** (detail in [085-01-yield](iterations/085-01-yield.md) §Pass 1 and
  §Failure-path evidence): a fourth failure case neither the blueprint nor the report named — the **re-dispatch now
  reaches the wire with a mutated body**, so a stale `Content-Length` would ship as a new defect — closed on source
  (`utils.py:22-26` drops `content-length`; `client_reqrep.py:1056-1058` builds a fresh `CIMultiDict` per request;
  `:1199-1201` derives the length from that call's payload). Also adjudicated: the `Optional[str]` shape of `.text`
  (F-2 — unreachable at this site, and the project configures no type checker) and one miscounted figure in this
  file's own status line (F-1 — corrected here).
- **Open for other lanes, unchanged by this acceptance**: the two §RM-001-owned backlog rows whose triggers have now
  fired twice (`QB:routes-messages/handler-lncp`, `QB:routes-messages/deep-violations-unenumerated`); and the
  blueprint's **Invariant candidate** ("reading a body back off an `aiohttp.web.Response` the code constructed
  goes through `.text`"), now evidenced by a live call site and still a candidate — recording and promotion are
  Interview's act. `FK-011` remains the session's only open Fork.
- **Closed since this acceptance, by the assumptions lane**: `claim-419`'s successor **row 425** landed
  (`DOES-NOT-HOLD`: the `str` round trip holds only where the response's charset resolves to utf-8) and was routed
  `accepted-consequence` at [086-assumption-decide](iterations/086-assumption-decide.md) — both falsifying arms are
  `unreachable` on a closed call-site census (`evidence/claim-425-charset-arm-reach.txt`; zero `charset=` arguments in
  the package), and the arm `messages.py:117` stands on is certified by `CL:aiohttp/web-response-str-body-text-property`.
  **`S20` is terminal** (`Routing: absorbed`; `assumptions/seams.md` §Run-46 routing changelog): no recomposition is owed
  and no successor row is owed to the collector, so nothing in `T019.001`'s acceptance waits on the assumptions lane.
- **`QAP-MS001` re-proved on these bytes by the QA lane** ([087-forward-qa](iterations/087-forward-qa.md),
  `READY-PASSING`): the approved gate is byte-unchanged since 077 (`tests/qa/` still at Checkpoint 183) and
  the CR3 wave + `T019.001` moved `server.py`/`routes/messages.py` underneath it, so the pass owes only
  Required Check 8 — re-execution. Observed at `8dfa5af`: `MS-001` 11/11, `MS-002` 15/15, `MS-004` 6/6,
  hermetic 476 @ 95.91%, `G-008`(b) documented half unchanged (`2N = 6` ops, `N + 1 = 4` files), `022` still
  held (no gate passes a flush parameter), and no `forward-qa`-owned backlog row open. One disposition
  closed without a test edit: `T018`'s `client_max_size=0` widened the body-size domain behind the documented
  writer, no `VP-*` at this milestone speaks to body size, and the property is owned at the hermetic tier by
  `tests/test_transport_parity.py::TestNoRequestBodyCap` — a QA-tier gate there would be a plan-level `VP-*`,
  not this lane's to author. `READY-PASSING` is not a verdict: Yield Phase owns the re-run and the
  `pass`/`fail` on the four `VP-*`.
- **Next**: `T019.002` (CR4 remediation, one-member wave, implementation Forward) → its local Yield → the MS-001
  boundary's assumptions recheck on those bytes (the first recompositions of `S16`–`S19` and the row-57 re-pin — `S20`
  and its row-425 wording are no longer on this list) → **Yield Phase** (`QAP-MS001` re-executed on the final bytes;
  087's `READY-PASSING` is byte-scoped to `8dfa5af`) → **Yield Red** (owns `resolved` for
  `T015`/`T016`/`T017`/`T018`/`T019`/`T020`, **`T019.001`** and now **`T019.002`**, the phase acceptance, the
  `Delivered` chronicle entry, parent completion) → `phase-boundary-finalize`.

## Trimmed 2026-10-07 (`089-assumption-decide`) — the superseded run-45 assumption-routing narrative (context budget)

Moved **verbatim** from `handoff-context.md` §Assumption routing — 2026-10-07 (run 45). Nothing rewritten; the pointer left in the baton file names where each durable consequence now lives. `T019.001` is accepted-local ([085-01-yield](iterations/085-01-yield.md)) and `S20` is terminal ([086-assumption-decide](iterations/086-assumption-decide.md)), so the narrative is history, not a live wave record.

## Assumption routing — 2026-10-07 (run 45, `S20` third-party atoms 418/419/422/423): dispositions, the reach determination, and one malformed return

- **Mode and boundary.** Atomic non-holding routing, four rows, all members of fresh seam `S20` (`State: new`).
  Rows 417/420/421/424 are `HOLD`/`none` and were not touched. No `DOES-NOT-HOLD` row here carries a
  `user-claim` check, so no `decisions.md` block is `CHALLENGED` and `USER-CLAIM FALSIFIED` is not emitted.
  All seven publishable rows are already in the registry (`CL:` handles at `invariants/index.md`); publication
  recorded what the sources established and did **not** pre-decide any disposition here, and no registry byte was
  moved by this pass.
- **Reach adjudicated from repo bytes, then by execution — the note's framing checked, not inherited.**
  `evidence/n6-reach-and-base-comparison.txt`. The standing third-party-only ruling limits which propositions may
  become ledger rows; it does not forbid reading repo code to decide reach, and this pass read it:
  the response the matcher reads is constructed at `messages.py:151-155` with **`body=`** and not with **`text=`**,
  from the `str` that `_error_body` returns; `messages.py:106` therefore always takes its `else` branch; the matcher at
  `utils.py:123-140` receives `<aiohttp.payload.StringPayload object at 0x…>` against `config.py:122-123`'s ASCII
  patterns and returns `None`; `parse_context_overflow_error` is never handed text on this path.
  **The retry is genuinely inert — and inert on both arms**, so `handle_streaming`'s loop can never `continue`
  (§6(d): the `ClientError` 502 arm carries bytes but a static body since `T016`'s de-leak). The composer's own
  reassurance survives and is now evidenced: what crosses the wire is the text and `.text` returns it (420/421 HOLD),
  so no client-visible byte, header or status is wrong today.
- **Base parity applied as the ranking authority — and it is silent, not protective.** `buster_ripper_legacy.py`
  contains no `overflow`, `sglang`, `retry`, `backend` or `max_completion` anywhere in its 1013 lines, and its
  streaming branch has no status-checked error arm at all. There is no base behavior at this site to restore or to
  excuse the current one with, so the deferral route that base parity opened for §RM-008 and §RM-009 is not
  available here. The inertness is **migration-introduced and inherited at session entry**
  (`iterations/001-collect.md` §B-005 already described the block as existing and "behav[ing] as tested"), **not
  introduced by the CR3 delta** — pre-delta `error_content` was already a `str` (`handoff-context.md` §S18-MC-1), so
  `:106` already took the same branch. One correction to N6's framing, checked specifically as dispatched: the delta
  did change *which inputs arrive* at the dead branch (pre-delta an undecodable error body raised `UnicodeDecodeError`
  past `:171` and never reached the probe; post-delta `_error_body` returns replacement text that does) — widened
  population, unchanged outcome.
- **Dispositions (one per row; `State` untouched on all four).**
  | Row | State | Routing | Basis |
  |---|---|---|---|
  | 418 | DOES-NOT-HOLD | `absorbed` | `DC-claim-418-r1` → `T019.001`; reach `reached`, fix-or-pin resolves to fix |
  | 419 | UNCLEAR | `malformed` | the noun, not the behaviour; returned to `forward-collect` — reworded successor below |
  | 422 | DOES-NOT-HOLD | `absorbed` | `DC-claim-422-r1` → `T019.001` step 1; it certifies the repair's read surface |
  | 423 | DOES-NOT-HOLD | `absorbed` | `DC-claim-423-r1` → `T019.001` step 1; it forecloses every coercion-shaped candidate |
- **What the falsified rows still establish for the record, independent of the repair.** (1) `web.Response` accepts a
  `str` at `body=` without raising (417) and stores a `StringPayload` (418), so any repo code that reads `.body` back
  from a `str`-built response is reading a payload object — the census shows exactly one such read in the package, and
  it is the one routed here. (2) The value is not lost: it survives to the wire (420), survives to `.text` (421), and
  `.text` is the *only* accessor on the class that returns it (422's enumeration), which is why no "the text never
  arrives" claim can be made about a `web.Response` — only about a chosen read surface. (3) `str()` of the payload is
  the default object repr (423, via 424), so no coercion of `.body` can be made to yield text: the candidate space at
  this site is closed to one read. (4) Rows 421 and 422 were checked independently and agree, which retires N6's
  "the text never arrives" as an exhaustive-surface claim while leaving its call-site observation true.
- **Return to `forward-collect` — the exact row and the wording its successor must ask (row 419, `S20`).** 419 stays
  `UNCLEAR`/`malformed` as evidence; it is not edited and not re-dispatched under its own id. The successor is a
  **new row 425**, subject `pkg:aiohttp@3.13.3`, seam `S20`, fresh bundle key, one proposition, answerable from
  `payload.py`/`web_response.py` at the pinned digests, and worded so its noun has a bytes-valued or decode-bearing
  referent — e.g. **"For an aiohttp `web.Response` built with a `str` `body=`, the payload its `body` getter returns
  decodes back to that same string under the response's own charset."** Constraints on the wording: the phrase
  "those `.body` bytes" is forbidden (no bytes-valued referent — 419's own blocker); the getter's *type* must not be
  re-rowed (owned by 418 / `CL:aiohttp/web-response-body-getter-type`); the `text` **property** must not be re-rowed
  (owned by 421 / `CL:aiohttp/web-response-str-body-text-property`); and the row must not name a repo call site
  (third-party subjects only). At the verdicted bytes it lands `HOLDS` (`StringPayload.__init__` encodes utf-8 at
  `payload.py:442`; `BytesPayload.decode` at `:380-381`), which is exactly what `S20`'s
  "equality with the input across that round trip" dimension needs from a member. **`S20` cannot recompose while any
  member is non-`HOLD`**, so 419's successor gates the recomposition, and `S20`'s `Routing` stays `unverified` until
  it lands and the seam composes.
- **Owning route, stated honestly against the wave the dispatch asked about.** `T019.001`, its own one-member wave,
  dispatched only after the CR3 wave's local Yield accepts. The CR3 wave does **not** carry it: `T018`/`T019`/`T020`
  are implemented, and their evidence was produced against a snapshot that excludes this change — 14 fail-first
  witnesses replayed red at pre-fix bytes, byte-exact AST verification of the must-not-touch surfaces, and the
  hermetic 471 @ 95.91% re-measure. `handle_streaming` is in no CR3 member's touch-set, and `073-01-yield` verified it
  byte-identical across the previous wave. `remediation-waves.md`'s ride-along rule admits a finding into a wave
  "already scheduled"; this one is already *executed*, so riding it in would force the wave's fail-first baseline to
  be re-derived at a second base and would put a 15th witness red against bytes the wave is about to have accepted.
  Had the wave still been scheduled, it would have ridden: the fix is one expression, and `messages.py` is the file
  CR3 was in. Capacity was checked, not assumed: `chat.py` stays at 300/300 untouched; the change is line-neutral in
  `messages.py` and the file's review-band misplacement check is written into `T019.001`'s done-when (6) rather than
  deferred.
- **Invariant cross-reference (all four rows): Neutral.** No session `invariants.md`; the registry has no
  `buster_ripper/routes/*` file; all eight entries touching this session are `Status: planned`
  (`PI:T002/dump_format/*`, `PI:T006/upstream_tls/*`) and none reaches this site; no active or wip invariant is
  touched, contradicted or upheld. The permanent `CL:` records named above are cited as evidence inputs only and
  authorize no behavior. One invariant candidate is recorded in `T019.001`'s block for Interview to rule on; this
  pass wrote and promoted nothing.
- **Zero questions.** FK-011 remains the session's **only** open Fork. Nothing here is materially-different-and-
  unranked: base parity is silent rather than ranking two outcomes, the file's own working peer supplies the required
  behavior, and `DC-CR3-F-04-r1`'s direction ranks keeping the text over keeping the repr. `S1`–`S15` stay terminal and
  untouched; `S20` is the only seam whose cells this pass describes, and only its gating is recorded — its `State`
  and `Routing` are unchanged.

## Trimmed 2026-10-07 (`090-01-yield`) — `T019.002` accepted-local (context budget)

`T019.002` is **`accepted-local`** at [090-01-yield](iterations/090-01-yield.md) — the wave's first pass, counter 0,
both mutations the blueprint assigns to Yield executed by that pass and reverted byte-exact. Three blocks move here
verbatim from `handoff-context.md`: the `Last Updated` line its acceptance supersedes, the `## Current Status` bullets
it replaces, and the consumed `T019.002` blueprint. Nothing rewritten, nothing dropped. The two records this pass wrote
outside `handoff-context.md` — `roadmap.md` §RM-012's corrected read-site census (four → five) and the
`quality-backlog.md` estimate-wording erratum, left as the interrupted draft wrote it after being re-verified against
source — carry their own locators in §Closure bookkeeping of that report.

### Superseded `Last Updated` line, verbatim

- Last Updated: **`T019.002` IMPLEMENTED at [090-01-forward](iterations/090-01-forward.md)** — the CR4 remediation's own one-member wave lands as a narrowed call: `count_tokens.py:87` keys on `normalize.strip_billing_nonce` instead of the whole of `normalize()`, `+13/−0` in `normalize.py` (delegating entry only) and `+8/−8` with zero line shift at the route, `session.py`/`routes/messages.py`/`routes/chat.py`/`buster_ripper_legacy.py`/`invariants/` carrying zero bytes. 35 witnesses in the new `tests/test_count_tokens_parity.py`, with all four of the blueprint's red directions executed by the same pass (fail-first trio 3-failed/12-passed at pre-fix bytes; the pre-`T019` spelling reddens both digest witnesses; hardening `normalize()` reddens the `/v1/messages` guard 500 → 502; a hypothetical `system`-rewriting step reddens the pure battery). Hermetic **511 @ 96.71%** against a 476 @ 95.91% baseline. Backlog: `QB:routes-count-tokens/normalize-side-effects-on-estimate-and-malformed-bodies` resolved; the blank-line-at-EOF row's trigger did not fire. The superseded `Last Updated` entries — this line replaces the `T019.001` ACCEPTED-LOCAL entry — are preserved verbatim in [handoff-archive.md](handoff-archive.md) §Trimmed 2026-10-07 (`090-01-forward`).

### Superseded `## Current Status` bullets, verbatim

- **`T019.002` IMPLEMENTED at [090-01-forward](iterations/090-01-forward.md)**, Checkpoint 205 `f00216c` (2026-10-07) —
  the CR4 remediation, its own one-member wave, first pass (its retry counter is still 0: no rejection has happened). `count_tokens.py:87` now
  calls `normalize.strip_billing_nonce` — the system-strip step alone — instead of the whole of `normalize()`.
  Production diff: `normalize.py` **+13/−0** (one delegating entry; its four step bodies and `normalize()` itself
  byte-identical) and `count_tokens.py` **+8/−8 with zero net line shift**. `session.py`, `routes/messages.py` (334),
  `routes/chat.py` (300/300), `buster_ripper_legacy.py` and `invariants/` carry **zero bytes**.
- **The witnesses**: `tests/test_count_tokens_parity.py`, 35 cases in one new focused module (281 lines, chosen over
  `tests/test_routes_integration.py` at 495 raw, where ~60 more would cross a file band). All seven Done-when clauses
  met, and every red direction **executed by the implementing pass** rather than inherited: the trio **3 failed / 12
  passed at pre-fix bytes** with the escaping `AttributeError` frames printed; replay (b) restoring the pre-`T019`
  spelling reddens **both** the new route oracle and the byte-unchanged
  `test_a_billing_nonce_body_finds_the_stats_messages_recorded`; replay (c) hardening `normalize()`'s loops reddens the
  `/v1/messages` guard **500 → 502**; replay (d) adding a hypothetical `system`-rewriting step reddens the pure digest
  battery on every affected shape. Transcripts: [evidence/t019-002-fail-first-and-replays.txt](evidence/t019-002-fail-first-and-replays.txt).
- **Suite**: hermetic **511 @ 96.71%** (baseline 476 @ 95.91% — +35 tests, none removed, coverage up, not down).
  `handle_count_tokens` re-measured **L:18 N:1 C:3 at HEAD and at final bytes** — taken from the HEAD blob, because the
  backlog row's recorded L:17 predates `T019` step 1 (090-01 §Concerns 2). Capacity gate: **not triggered**, with the
  measured baseline and both file sizes in the report.
- **Backlog pickup**: `QB:routes-count-tokens/normalize-side-effects-on-estimate-and-malformed-bodies` **resolved by
  this increment** — its two stated consequences (`estimate_tokens` on the normalized body; a non-mapping `messages`
  element raising where an estimate answered) are exactly what the narrowing disposes, measured rather than argued.
  `QB:tests/blank-line-at-eof` **not** picked up: neither named file is in this touch-set. `QB:lint/ruff-f-select-findings`
  stays open — none of its five sites is a touched file, and its own bound says pickup, not sweep.
- **One finding recorded for the §RM-012 owner, no session record amended**: `normalize()`'s element loops have **four**
  read sites, not the three `evidence/cr4-probe-matrix.txt` §6 counts — `_normalize_tools` reads `t.get("name", "")`
  twice, and the first hardening attempt left `{"tools":[null]}` still 500ing. A future §RM-012 repair specified as
  "three loops" is incomplete. Carried in 090-01 §Concerns 1.
- **Ordering unchanged**: `T019.002` lands **before** the MS-001 boundary's assumptions recheck, which is
  `S16`–`S19`'s single allowed post-repair recomposition and must run on these final bytes. Mechanically that means the
  one `count_tokens.py` coordinate re-base the blueprint permits (rows `55`/`134`/`218` re-pin their unit fingerprints;
  rows `264`–`266` cite `normalize#normalize` **by symbol**, whose body is byte-identical, so they do not re-base).
- **Next**: this wave's local **Yield** → the assumptions recheck → **Yield Phase** (re-executes `QAP-MS001` on the
  final bytes; 087's `READY-PASSING` is byte-scoped to `8dfa5af`) → **Yield Red**, which owns `resolved` for
  `T015`–`T020`, `T019.001` and **`T019.002`** → `phase-boundary-finalize`.
- **Carried, unchanged by this increment**: `FK-011` is the session's only open Fork; the two §RM-001-owned backlog rows
  whose triggers have fired twice (`QB:routes-messages/handler-lncp`, `QB:routes-messages/deep-violations-unenumerated`)
  stay open for other lanes; the `T019.001` **Invariant candidate** ("reading a body back off an `aiohttp.web.Response`
  the code constructed goes through `.text`") is still a candidate — recording and promotion are Interview's act.
- **Superseded by this entry**: the previous `Last Updated` line and the eight `## Current Status` bullets it replaces
  (the CR4 routing bullet with its ruled-not-deferred half, `T019.001`'s acceptance and the Yield additions to it, the
  open-for-other-lanes bullet, the `claim-425`/`S20` closure, the `QAP-MS001` re-entry, and the old Next line) are
  preserved verbatim in [handoff-archive.md](handoff-archive.md) §Trimmed 2026-10-07 (`090-01-forward`).

### Consumed blueprint `T019.002 Planned Task Context`, verbatim

### T019.002 Planned Task Context — Answer `count_tokens` the way base answers it: key on the system-strip step, not on the whole of `normalize()`

- Owning route: implementation Forward, **own one-member wave**, one local Yield. Nested remediation under `T019`
  (this is `T019`'s owning surface — the `count_tokens` route and its `normalize` call). `T019` stays
  `accepted-local`, its landed increment `d5df8b06…9c90675` is **not** reopened, and `T019`'s retry counter stays 0.
- Finding: `CR4-F-01` (P2) from [088-codex-review](iterations/088-codex-review.md) — closure-packet mode; the packet
  names no assumption seam, so no composition is required for resolution.
- Evidence: `iterations/088-codex-review.md` §[P2]; **re-established on current bytes by this pass, at all three
  points of the comparison** — `evidence/cr4-probe-matrix.txt` §1 (HEAD vs `6508941`, both axes), §2 (base
  `72e3a1c6`, live upstream stub), §4-§5 (the narrowing's digest equality and its route-level effect).
- Disposition: absorbed
- Status: **implemented** ([090-01-forward](iterations/090-01-forward.md), 2026-10-07) — all seven Done-when clauses met
  on the bytes recorded there; `resolved` remains Yield Red's, as the routing pass prescribed.
- Counterexample: `POST /v1/messages/count_tokens` with `{"tools":[null]}` (also `{"messages":[null]}`,
  `{"messages":["x"]}`) on HEAD answers **500** — `AttributeError: 'NoneType' object has no attribute 'get'` escapes
  `handle_count_tokens` from `normalize.py:75` (`_normalize_tools`) and `:117`
  (`_strip_cache_control_from_messages`), aiohttp logs the traceback, and the client gets a generated 500 instead of
  the route's own answer. The same three requests answer **200** at `6508941` and **200** at base. Measured, not
  quoted: `evidence/cr4-probe-matrix.txt` §1-§2.
- Required behavior: `/v1/messages/count_tokens` keeps answering an estimate for **every JSON object body**,
  whatever the shapes of its members, as base (`buster_ripper.py:672`) and the pre-range branch both do — while the
  stats lookup keeps keying on the post-strip `system` digest that `/v1/messages` records under, which is the
  property `DC-CR3-F-02-r1` bound to `EI:buster_ripper.session/session_id/derived_after_normalize`.
  `/v1/messages` is unchanged, byte-for-byte and behavior-for-behavior, including its base-parity 500 on the same
  three bodies.
- Selected closure: narrow the call at `count_tokens.py:87` from `normalize(body, session_id="")` to the
  normalization step the invariant actually needs — `_normalize_system` (`normalize.py:50-61`), the only one of the
  four steps that writes `system`'s text (`session_id()` reads `system` only, `session.py:93-98`). **No behavior in
  `normalize.py` changes**: it stays byte-exact if the route imports the step directly, or gains one delegating
  entry if the implementation prefers a public name for a cross-module call — either way its four step bodies and
  `session.py` are byte-exact, and the naming is not a different outcome.
- Done when — with the direction each witness fails first in:
  1. **Red-first, the direction of the defect.** Route witnesses over `{"tools":[null]}`, `{"messages":[null]}`,
     `{"messages":["x"]}` → 200 with `{"input_tokens": <int>}`. All three are **red at pre-fix bytes as 500s** with
     the `AttributeError`; keep that transcript as the fail-first evidence.
  2. **Green-pin, base's tolerance preserved.** `{"system":[null]}`, `{"messages":[{"content":[null]}]}`, `{}` and
     an unparseable body → 200 before and after. Green→green deliberately: these pin that the repair did not
     over-reach. Say so in the docstrings, naming base and `6508941` as the authority.
  3. **Green-pin, parity is a ceiling too.** The non-dict body `[1,2]` → **still 500**, which is what base
     (evidence §2) and `6508941` (§1) both do — at HEAD the raise moves from `session.py:95` to `normalize.py:52`,
     same status, same request. Characterization witness, so labeled; it reddens the moment anyone makes the route
     answer 200 to a body base refused.
  4. **Red-first, the direction of the invariant.** Digest equality survives the narrowing: a parametrized witness
     asserting `session_id(system-stripped b) == session_id(normalize(b)[0])` over the battery of evidence §4 (nonce
     as string and as block, the nonce-only block that is deleted outright, `cache_control` on `system` blocks,
     tools reordering, an image body), **and** `tests/test_routes_integration.py`'s
     `test_a_billing_nonce_body_finds_the_stats_messages_recorded` staying green **unchanged**. Reddened by dropping
     the system step — and by any future `normalize()` step that starts rewriting `system`'s text, which is the
     regression the narrowing otherwise makes possible.
  5. **Red-first, the direction of the rejected candidate.** Route witnesses at `/v1/messages` asserting the same
     three bodies **still answer 500 with no upstream call made**, each docstring naming base as the authority
     (`buster_ripper.py:374`/`:402` raise the identical `AttributeError`; evidence §3) and pointing at
     `roadmap.md` §RM-012 as the owner of any change to that posture. Green on the correct fix; **red, 500 → 502, if
     `normalize()`'s element loops are hardened instead** — this is the drift guard the finding was raised to close.
  6. **Byte-exactness.** `session.py`, `routes/messages.py` and `routes/chat.py` unchanged, `buster_ripper_legacy.py`
     untouched, zero `invariants/` bytes; `normalize.py` either byte-exact or carrying **only** a delegating entry for
     the system step (its four step bodies byte-exact, `normalize()` itself byte-exact — the route stops calling it, it
     is not edited). `messages.py` (334) and `chat.py` (300/300) stay out of the diff entirely — see §Capacity answer.
  7. **Suite.** Hermetic suite green on the new bytes, the clause-1 witnesses demonstrated red at pre-fix bytes,
     and coverage not reduced below the 95.91% the `T019.001` bytes certified.
- Yield replay: (a) replay the counterexample independently — the trio at `/v1/messages/count_tokens` answers 200,
  the trio at `/v1/messages` answers 500 with no upstream call; (b) execute the mutation for clause 4 — restore the
  pre-`T019` spelling (`sid = session_id(body)` on the un-normalized body) and confirm **both** the nonce witness
  and the digest-equality witness go red; (c) execute the mutation for clause 5 — harden one `normalize()` element
  loop and confirm the `/v1/messages` guard goes red 500 → 502, then revert byte-exact; (d) `git diff --name-only`
  over the increment contains none of `session.py`, `routes/messages.py` or `routes/chat.py`, and any
  `normalize.py` hunk in it is the delegating entry alone.
- Capacity answer, stated plainly: **the fix lands without touching `messages.py` or `chat.py`.** It is one call at
  `count_tokens.py:87` (106 lines) plus, if the implementation names it, a three-line public entry in
  `normalize.py` (228 lines, and byte-unchanged across the whole range so far). `messages.py` (334, review band, its
  split is §RM-001's design decision) and `chat.py` (frozen at 300/300) stay out of the touch-set, and clause 6
  requires that. Test-side capacity: `tests/test_routes_integration.py` is at **495** raw lines — review band
  (301-500), and a ~30-line addition crosses into extract-file (501-700) — so the witnesses belong in
  `tests/test_transport_parity.py` (220, whose subject is already "what base did and the package must keep doing")
  or in a new focused module. That choice is the implementation's; crossing a file band silently is not.
- Invariant cross-reference: **Aligned** — `EI:buster_ripper.session/session_id/derived_after_normalize` (carried
  `preserve`, `plan-summary.md` §Invariant Reconciliation + §Notes; deliberately never registry-active,
  `invariants/index.md` §Source Sessions). Bound to the existing handle; **no duplicate invariant candidate is
  created**. `G-003` (proxy behavior on `/v1/messages` unchanged; constraint column "No new endpoints, no
  normalization changes") is the settled goal that ranks the wider candidate out. Zero `invariants/` bytes moved.
- Closure ledger: `DC-CR4-F-01-r1`
- Assumption impact: **none owed.** Third-party-subjects-only ruling stands: this routing creates no `repo:` claim
  row and no `repo:` seam; `S1`–`S15` stay terminal and untouched, `S16`–`S19` have spent their single post-repair
  recomposition allowances and `S20` is terminal ([086](iterations/086-assumption-decide.md)). The fix's structural
  premise is not new evidence — it is the same `session.py:93-98` / `normalize.py:50-61` reading
  `DC-CR3-F-02-r1` already binds to the carried handle. Mechanical consequence for the recheck lane only:
  `count_tokens.py` coordinates move, so a fingerprint anchored in that file re-bases **once**, after this wave's
  Yield accepts.
- Execution order: **before the MS-001 boundary's assumptions recheck.** That recheck is `S16`–`S19`'s one allowed
  post-repair recomposition and must run on final bytes; dispatching it first spends the allowance on bytes this
  task is about to move. Then this task's local Yield → the recheck → **Yield Phase** (`QAP-MS001` re-executes on
  the final bytes; 087's `READY-PASSING` is byte-scoped to `8dfa5af`) → **Yield Red**, which owns `resolved` for
  `T015`-`T020` and `T019.001` and now also for this task.
- Deferred, with its owner named: the `/v1/messages` half of the reviewer's second question — the malformed-element
  500 that base, `6508941` and HEAD all share — is **not** this task's. It is `roadmap.md` §RM-012, it is recorded
  in §Accepted risks and §Elevated to user below, and clause 5 exists so this session cannot drift into it.

## Trimmed 2026-10-17 (`092-forward-qa`) — upstream-correction and 091-block detail (context budget)

Moved verbatim from `handoff-context.md` §Current Status when 092 corrected the gate endpoint and re-ran the
32 process-level gates. The current one-line states of these three blocks live in §Current Status; the durable
authority for the endpoint is `decisions.md` §US-009 and the full 091 reasoning is
[091-yield-phase](iterations/091-yield-phase.md).

- **⚑ UPSTREAM CORRECTION 2026-10-17 (user) — the QA gate upstream is `http://hyperipper.local:30000`, not `:30001`.**
  The user corrected `US-009`: `:30001` is **another Buster Ripper instance** (a peer proxy) on the same host,
  not the vLLM-compatible model server; `:30000` is the model server and is up. The `BLOCKED-UPSTREAM` verdict at
  [091-yield-phase](iterations/091-yield-phase.md) stands as that pass's verdict, but its **cause is re-read**: the
  gates were pointed at a peer proxy that was stopped, so the 8 `require_upstream()` errors were the peer proxy's
  absence, not the model server's. 071's `HOLDS` measured the same peer proxy — real, but non-discriminating by response
  shape — and 091's "transient host state" reading is superseded as a causal account (the host was up; the peer proxy was
  down). Bounded observation at `2026-10-17T21:31:45Z` from this host: `:30000` → `200`, `server: uvicorn`, vLLM-shaped
  model list; `:30001` → connection refused. The owed act is **not** to restore a `:30001` listener: re-run the 32
  process-level gates against `:30000`, after `tests/qa/harness.py:5` (docstring) and `:39` (`UPSTREAM`) are re-pointed —
  **`forward-qa`'s lane (QA author pass), test bytes**. Ledger: `decisions.md` §US-009's dated correction (claim 2 left as a historical `HOLDS`
  with the correction beneath it); §US-010, §US-016 and §FK-009 carry dated pointers. Still owed by their lanes:
  `qa-plan.md` §Overview/§Scenario Contracts (`forward-qa`'s lane) still cites `:30001`; `roadmap.md` §RM-008 now carries
  a dated correction but its row owner still owes the re-read of the `latent` ruling's basis.
  `README.md`'s `:30001` occurrences are the app's own listen port and the documented `ANTHROPIC_BASE_URL` — not the
  upstream, left alone.
- **⛔ MS-001 BOUNDARY: `BLOCKED-UPSTREAM` — no integration verdict issued at [091-yield-phase](iterations/091-yield-phase.md)**
  (2026-10-08, HEAD `83de78c`, clean tree). The declared live upstream `http://hyperipper.local:30001` is **refusing
  connections** from this host — `192.168.50.132` answers `icmp` at 0.25 ms and accepts ports 22 and 30000, so the
  machine is up and only the `30001` listener is gone; refused on 3 immediate probes, on all 43 periodic probes (20 s apart, 20:02:59-20:17:08) and on a final direct re-probe across the
  pass. All 8 endpoint-dependent `QAP-MS001` gates **failed** on `require_upstream()` (`ConnectionRefusedError
  [Errno 61]`, raised in the test body, not a fixture error)
  with **zero assertion failures** — the harness failing closed as `US-009`/`qa-project` designed it to. **The four
  `VP-*` (`VP-T001-writes-deferred`, `VP-T001-dump-failure-isolated`, `VP-T002-default-encoding`,
  `VP-T002-documented-format-preserved`) are recorded `unknown`; `pass`/`fail` was NOT inherited from 087**, whose
  `READY-PASSING` 11/11 is byte-scoped to `8dfa5af` and predates `T019.002`.
  **Re-read 2026-10-17:** the `:30001` endpoint above was a peer Buster Ripper proxy, not the model server
  (see the correction bullet above). The verdict and the four `unknown` handles stand; only the causal reading changes.
- **NEXT — owed before any acceptance**: restore/confirm the `192.168.50.132:30001` listener (an operational act on a
  remote host, **outside this session's scope** per `codify-scope`), then re-run Check 6 step 6 + Check 7 steps 1–2:
  **32 process-level gates on one HEAD** — `test_qap_ms001.py` (11) + `test_qap_ms002.py` (15) + `test_qap_ms004.py`
  (6), since the cycle moved bytes in the `tests/qa/harness.py` all three share. No stub fallback, no
  `:30000`/loopback-peer substitution — `US-009` settled the endpoint and `UPSTREAM` is a constant at
  `tests/qa/harness.py:39` with no override. **If no production byte moves meanwhile, Checks 1–5 and step 0 need no
  re-derivation; this pass is the baseline.** Only then do the assumption recheck and **Yield Red** run; Yield Red
  still owns `resolved` and the Delivered chronicle entry, and this blocked verdict publishes no child completion.
  **Corrected 2026-10-17:** no `:30001` listener is owed — `:30001` is a peer Buster Ripper proxy, not the
  model server. The re-run is owed on `http://hyperipper.local:30000`; the 32-gate set, the no-stub rule and the
  Check 1–5/step-0 baseline all stand. The "no `:30000`/loopback-peer substitution" clause above is superseded:
  `:30000` is now the settled endpoint; the loopback-peer substitution remains forbidden.

Superseded `Last Updated` line, moved verbatim by 092:

- Last Updated: **`T019.002` ACCEPTED-LOCAL at [090-01-yield](iterations/090-01-yield.md)** — the CR4 remediation's one-member wave is accepted on its first pass (counter 0, zero reject-tier findings): `count_tokens.py:87` keys on `normalize.strip_billing_nonce` instead of the whole of `normalize()`, `+13/−0` in `normalize.py` (delegating entry only) and `+8/−8` with zero line shift at the route, `session.py`/`routes/messages.py`/`routes/chat.py`/`buster_ripper_legacy.py`/`invariants/` carrying zero bytes; all seven Done-when clauses re-derived on the bytes and **both mutations the blueprint assigns to Yield executed by that pass** — (b) drop-the-strip → 5 witnesses red, (c) harden-the-loops → the three `/v1/messages` guards red **500 → 502** — each reverted byte-exact against Forward's §E pins; hermetic **511 @ 96.71%** re-measured against a 476 @ 95.91% baseline; `handle_count_tokens` L:18 N:1 C:3 identical at both ends. 35 witnesses in `tests/test_count_tokens_parity.py`. Two records written by the acceptance: `roadmap.md` §RM-012's read-site census corrected **four → five** (an inherited figure this pass measured wrong), and the estimate-wording erratum `QB:report-accuracy/090-01-estimate-minus-nonce-not-raw` left standing after being re-verified by reading. `accepted-local`, not `resolved`; next is the MS-001 assumptions recheck on these bytes. The superseded `Last Updated` entries — this line replaces the `T019.002` IMPLEMENTED entry — are preserved verbatim in [handoff-archive.md](handoff-archive.md) §Trimmed 2026-10-07 (`090-01-yield`).

## Trimmed 2026-10-17 (`093-yield-phase`) — MS-001 boundary verdict (context budget)

Superseded `Last Updated` line, moved verbatim by 093:

- Last Updated: **`READY-PASSING` at [092-forward-qa](iterations/092-forward-qa.md)** — the gate upstream was corrected by the user to `http://hyperipper.local:30000` (`:30001` is a peer Buster Ripper proxy, `decisions.md` §US-009); `tests/qa/harness.py:5`/`:39` and `qa-plan.md`'s three citations re-pointed with dated notes, `README.md`/`--port` left alone. 32/32 process-level gates green on `HEAD 39585db` (11 + 15 + 6, `:30000` preflight HTTP 200) and hermetic **511 @ 96.71%**; no scenario byte moved, and `pass`/`fail` stays the MS-001 Yield Phase's. The superseded `Last Updated` line is preserved verbatim in [handoff-archive.md](handoff-archive.md) §Trimmed 2026-10-17 (`092-forward-qa`).

The `## Current Status` block 093 replaced, moved verbatim:

## Current Status

- **⚑ GATE UPSTREAM CORRECTED 2026-10-17 (user) — `http://hyperipper.local:30000`; the transport block is CLEARED.**
  `:30001` is a peer Buster Ripper proxy on the same host, not the model server (`decisions.md` §US-009, Checkpoint 211).
  The `forward-qa` lane corrected `tests/qa/harness.py:5`/`:39` and the three `qa-plan.md` citations, then re-ran the
  32 process-level gates on `HEAD 39585db`: `:30000` preflight **HTTP 200** (vLLM-shaped list), `QAP-MS001` **11/11**,
  `QAP-MS002` **15/15**, `QAP-MS004` **6/6** (its own loopback peer, per the `MS-004` amendment), **32/32**, hermetic
  **511 @ 96.71%** — `READY-PASSING` at [092-forward-qa](iterations/092-forward-qa.md), no scenario byte moved.
  `README.md`'s `:30001` occurrences and the app's `--port` default are the proxy's own listen port and stay untouched.
  The pre-092 correction and block detail is preserved verbatim in [handoff-archive.md](handoff-archive.md)
  §Trimmed 2026-10-17 (`092-forward-qa`).
- **⛔ MS-001 BOUNDARY: 091 recorded `BLOCKED-UPSTREAM`, the four `VP-*` `unknown`, and issued no integration verdict**
  ([091-yield-phase](iterations/091-yield-phase.md), 2026-10-08, HEAD `83de78c`; the 8 endpoint-dependent gates failed on
  `require_upstream()` with zero assertion failures, fail-closed by design). The four handles — `VP-T001-writes-deferred`,
  `VP-T001-dump-failure-isolated`, `VP-T002-default-encoding`, `VP-T002-documented-format-preserved` — stay `unknown` until
  the Yield Phase writes them; 092's green run is not a verdict and does not inherit one.
- **What 091 did establish, on final bytes**: Checks **1–5 PASS** from source, with the milestone's owned surface
  (`dump.py`, `config.py`, `main.py`, `session.py`, `utils.py`, `README.md`, `serve_buster_ripper.sh`,
  `pyproject.toml`) and **all of `tests/qa/**`** shown byte-identical to 078's approved state (`git diff --stat
  6508941...HEAD` → empty; `git log -1 -- tests/qa/` → `6508941`), the dump call block content-anchored identical, and
  the cycle's exactly-two new definitions (`normalize.strip_billing_nonce` public → called at `count_tokens.py:87`;
  `messages._error_body` private → called at `:161`) both wired in production, none removed. Check 7 **step 0 PASS**:
  hermetic **511 @ 96.71 %**, zero failures, nothing to bisect. `BASE-SYNC: unchanged` re-verified at this HEAD
  (target object an ancestor, 238 ahead / 0 behind, base unmoved).
- **The two non-rerun items are disposed, with no gate invented for either** (091 §T018/T019.002 adjudication):
  `MS-001`'s contract is **silent by text** on both — no `VP-*`, scenario or `Complete when` clause names a body size
  or `count_tokens` keying. T018's widened domain went to `tests/test_transport_parity.py::TestNoRequestBodyCap`
  (4 witnesses, green here) — recorded honestly as *not* covering a >1 MiB body reaching the **writer**, a positive
  statement no `VP-*` asks for whose size-dependent failure mode has no producer in the source; T019.002's narrowed key
  went to 35 witnesses in `tests/test_count_tokens_parity.py`, `CL:buster_ripper.session#session_id/key-derivation-input`,
  and the now-`[x]` `QB:routes-count-tokens/normalize-side-effects-on-estimate-and-malformed-bodies`.
- **Ledger act of this pass** (record-only): `decisions.md` §US-009 claim 2 — the only `Verification:
  yield-phase:MS-001` factual claim — is **does not hold on this snapshot**, annotated as a dated re-check **under**
  the existing `HOLDS` cell rather than overwriting it (transient host state; 087 measured it live the day before).
  No `PENDING` row existed, so no `MIGRATION-REQUIRED` return is owed. Zero `invariants/` bytes moved by this pass.
  **Corrected 2026-10-17:** the "transient host state" reading is superseded — the endpoint was a peer proxy
  and the listener that went away was the peer proxy's, not a vLLM host's; see the correction bullet above. The
  annotation, the `HOLDS` cell and the zero-byte invariant claim stand.
- **Retry cap**: **not touched** — this is not a rejection of a root task. Zero findings attribute to the tree
  (zero assertion failures, zero hermetic failures), so `Forward->Yield Retries` stays as 090-01 left it: every
  counter in `T015`–`T020`, `T019.001`, `T019.002` at **0**.
- **NEXT — MS-001 Yield Phase on `HEAD 39585db`**: re-execute `QAP-MS001` or accept
  [092](iterations/092-forward-qa.md)'s run (same HEAD, same tree) and write the four `VP-*` verdicts. 091's Checks 1–5
  and its 511 @ 96.71% hermetic run stand as the baseline, so no source re-derivation is owed unless a production byte
  moves; then the assumptions recheck and **Yield Red** run, which still owns `resolved` and the Delivered chronicle entry.
  No `:30001` listener is owed — the gate upstream is `:30000` and it answered 200 before the run.
- **`T019.002` ACCEPTED-LOCAL at [090-01-yield](iterations/090-01-yield.md)**, Checkpoint 205 `f00216c` + 206 `649c0ad`
  (2026-10-07) — the CR4 remediation, its own one-member wave, accepted on its **first** pass: retry counter **0**
  (no rejection happened), `T019`'s counter untouched and its landed range `d5df8b06…9c90675` not reopened. Zero
  reject-tier findings; **`accepted-local`, not `resolved`** — the assumptions recheck and Yield Red own that.
- **What the increment is**: `count_tokens.py:87` keys on `normalize.strip_billing_nonce` — the system-strip step
  alone — instead of the whole of `normalize()`. `normalize.py` **+13/−0** (one delegating entry; the four step
  bodies and `normalize()` byte-identical), `count_tokens.py` **+8/−8, zero net line shift**; `session.py`,
  `routes/messages.py` (334), `routes/chat.py` (300/300), `buster_ripper_legacy.py` and `invariants/` carry **zero
  bytes**. 35 witnesses in `tests/test_count_tokens_parity.py` (281 lines, chosen so `test_routes_integration.py`
  at 495 never crosses a band).
- **What the Yield pass measured rather than inherited**: all seven Done-when clauses re-derived on the bytes; the
  blueprint's replays executed by that pass — **(b)** dropping the strip reddens **5** witnesses (the 3 route-oracle
  params, the byte-unchanged `test_a_billing_nonce_body_finds_the_stats_messages_recorded`, and
  `test_count_tokens_inflates_on_compaction_trigger[per-session]`), **(c)** hardening the element loops reddens the
  three `/v1/messages` posture guards **500 → 502** — each reverted byte-exact against Forward's §E sha256 pins.
  Hermetic **511 @ 96.71%** re-run (baseline 476 @ 95.91%, +35 cases, none removed); `handle_count_tokens`
  **L:18 N:1 C:3 identical at both ends** from a re-extracted `c5e2be4` baseline; capacity gate not triggered.
- **Two records this pass wrote, one of them against an inherited claim**: `roadmap.md` §RM-012's read-site census
  corrected **four → five** (`_normalize_tools` reads `t.get("name", "")` at `:88`, `:89` **and** `:90`; guarding
  `:88`+`:89` keeps `{"tools":[null]}` green at 500 — measured), with the row's now-stale `:75`/`:117`/`:192`
  coordinates noted; and the `quality-backlog.md` estimate-wording erratum
  `QB:report-accuracy/090-01-estimate-minus-nonce-not-raw`, **re-verified by reading and left as written** — the
  closed row's "as it arrived" overstates by the nonce the invariant forces into the estimate.
- **Backlog state**: `QB:routes-count-tokens/normalize-side-effects-on-estimate-and-malformed-bodies` **resolved**
  (spot-checked as required for a `[x]`: consequence (ii) proven red-able again by replay (c); (i) disposed modulo
  the nonce → the erratum above). `QB:tests/blank-line-at-eof` and `QB:lint/ruff-f-select-findings` left unchecked
  **with their triggers verified not to have fired** — no named file is in this touch-set.
- **Next**: the MS-001 boundary's **assumptions recheck on these final bytes is ROUTED, not pending** — `S16`–`S19`'s
  one post-repair recomposition landed `SAFE` on the Checkpoint 191 basis and its allowance is spent; run 47
  dispositioned `S16`/`S17`/`S18` terminal-for-boundary with their publication debt at
  `QB:assumption-registry/handleless-members-blocking-co-publication`, and named one evidence condition for **Yield
  Red** to weigh rather than re-compose: `S18`'s own §Invalidates On 5 (N1/N5(iii) and N6's re-read-before-citing
  clause) points at the `count_tokens.py:87` keying that `T019.002` narrowed and at the `messages.py` note
  `T019.001` inserted, both after that record was written — see
  [assumptions/seams.md](assumptions/seams.md) §Run-47. Remaining in the assumption lane: the `claims.md` row
  `55`/`134`/`218` re-pin (rows `264`–`266` bind `normalize#normalize` by symbol and do not
  re-base) → **Yield Phase** (re-executes `QAP-MS001`; `087`'s `READY-PASSING` is byte-scoped to `8dfa5af`)
  — **RAN AT [091](iterations/091-yield-phase.md) AND BLOCKED: `BLOCKED-UPSTREAM`, the four `VP-*` `unknown`, a
  32-gate re-run owed on a listening `hyperipper.local:30001` (corrected 2026-10-17: `:30000` — `:30001` is a peer
  proxy, not the model server); 091's Checks 1–5 and its 511-test hermetic run stand
  as this boundary's baseline, so no source re-derivation is owed if no production byte moves meanwhile** →
  **Yield Red**, which owns `resolved` for `T015`–`T020`, `T019.001` and **`T019.002`** → `phase-boundary-finalize`.
- **Carried, unchanged by this acceptance**: `FK-011` is the session's only open Fork; the two §RM-001-owned backlog
  rows whose triggers have fired twice (`QB:routes-messages/handler-lncp`,
  `QB:routes-messages/deep-violations-unenumerated`) stay open for other lanes; the `T019.001` **Invariant
  candidate** ("reading a body back off an `aiohttp.web.Response` the code constructed goes through `.text`") is
  still a candidate — recording and promotion are Interview's act.
- **An interrupted attempt at this same iteration id** wrote a full ACCEPT draft and stopped before any bookkeeping.
  Its verdict was not this session's acceptance record; its two durable writes were re-checked rather than inherited
  (one stands, one is corrected above), and its draft is preserved verbatim at
  [evidence/090-01-yield-interrupted-draft.md](evidence/090-01-yield-interrupted-draft.md). The superseded `Last
  Updated` line and the eight `## Current Status` bullets this entry replaces are preserved verbatim in
  [handoff-archive.md](handoff-archive.md) §Trimmed 2026-10-07 (`090-01-yield`).

## Trimmed 2026-10-17 (`094-assumption-decide`) — CR4 closure-packet routing (context budget)

## Closure-packet routing — 2026-10-07 (CR4 batch, persisted at [088](iterations/088-codex-review.md)): dispositions and the wave

Mode: closure-packet routing — the packet is 088 itself and it names no assumption seam, so resolution follows the
owning route's acceptance gate and no composition is owed. Read for scope: `plan-summary.md`, `decisions.md`,
`roadmap.md`, `quality-backlog.md`, `invariants/index.md`, both ledgers. `assumptions/claims.md` carries **zero live
`Routing: pending` rows** (its three matches are prose) and no seam awaits a disposition here; no seam row's cells
were touched by this pass.

| Finding | Disposition | Reach verdict | Owner / destination |
|---|---|---|---|
| `CR4-F-01` (P2) | **absorbed** | `latent` — exposed route, no guard prevents a non-mapping element, nothing pins it; the gate's latent rule resolves to **fix** | **`T019.002`** — nested under the owning task (`T019`'s surface: the `count_tokens` route and its `normalize` call), own one-member wave, one local Yield, prework of the MS-001 close. Ledger `DC-CR4-F-01-r1`, repository state `git-commit:c5e2be44ba1a7ab984f02104472f06f849e4f773`. Full record: [089](iterations/089-assumption-decide.md) |
| `CR4-F-02` (P3) | **accepted** | inert — nothing reads a file's final blank line: no `.github/workflows`, no `[tool.ruff]`, and the tree's only lint row (`QB:lint/ruff-f-select-findings`) runs `--select F`, which does not include `W391`; the sibling shape pre-exists at `tests/test_dump.py` and pre-dates the range | no task, no wave; one §Accepted risks line and one backlog row, `QB:tests/blank-line-at-eof`, whose owner is **the next Forward pass that edits either named file** (touch-set pickup — the treatment `DC-CR3-F-04-r1` gave in-range lint hits) |

**The `/v1/messages` half — adjudicated, not deferred quietly.** The reviewer split a deterministic repair from a
genuine ruling and asked for the ruling. The ruling: **that crash is protected by parity, it is not absorbed here,
and it is not a question.** Three measurements carry it. (i) base's own `normalize()` raises the identical
`AttributeError` on the identical three bodies at the identical route (`evidence/cr4-probe-matrix.txt` §3, against a
live upstream stub, so the 500s are attributable and the control body's 200 proves the harness can succeed);
(ii) the package matches base at `6508941` and at HEAD (§1's second block), so the range moved nothing there;
(iii) hardening the loops turns those 500s into relay traffic — **500 → 502 measured** (§7) — which is the proxy
answering differently than base does, on the one route `G-003` exists to hold still, whose constraint column reads
"No new endpoints, **no normalization changes**." The session's pattern of declining to route out a 500 its own
migration introduced (`T018` absorbed the `client_max_size` cap rather than defer it) is **followed, not departed**:
what `T018` absorbed was a rejection base never imposed; what is declined here is a rejection base imposes today.
One authority, opposite directions — which is exactly why the pre-question gate admits no question: after base
parity and `G-003` are applied, no materially different admissible outcome is left unranked. Owner of the posture,
named plainly: **`roadmap.md` §RM-012** — the route-input posture design pass, co-owned with §RM-001 (which already
owns `/v1/messages`' handler shape), with its trigger and three-way option set recorded there, together with the
evidence file and the fact that a two-loop fix would be incomplete (`normalize.py:192` is masked by `:117` today and
inherits the crash the moment `:117` is hardened — §6). The drift is closed from this side too: `T019.002`'s
Done-when clause 5 requires `/v1/messages` witnesses that go **red, 500 → 502**, if anyone hardens `normalize()`
under the cover of this repair.

**Wave.** One wave: anchor `T019.002`, single member, owning route implementation Forward, dispatchable now. No
existing wave can carry it — `T018`/`T019`/`T020`/`T019.001` are `accepted-local` with fail-first evidence produced
against bytes this change would invalidate, and `remediation-waves.md` admits a ride-along only into a wave that is
*scheduled*; none is. `CR4-F-02` opens no wave and joins none. **Ordering obligation:** this task lands **before**
the boundary's assumptions recheck, so `S16`–`S19`'s single allowed post-repair recomposition runs on final bytes;
then this wave's local Yield → the recheck → **Yield Phase** (re-executing `QAP-MS001` on the final bytes — 087's
`READY-PASSING` is byte-scoped to `8dfa5af`) → **Yield Red**, which owns `resolved` for `T015`-`T020`, `T019.001` and
now this task. **Zero questions: `FK-011` remains the session's only open Fork**, and no
`codify reserve-decision-record` was run.

The pointer left in `handoff-context.md` names this heading; every durable consequence the section carried
stays live in its own records (the `DC-CR4-F-01-r1` ledger row, the §Accepted risks line, and the
`roadmap.md` §RM-012 row).

## Trimmed 2026-10-17 (`097-yield`) — `T021` accepted-local; Forward→Yield retries

## Forward->Yield Retries
- **CR4 remediation `T019.002` (local, prework of the MS-001 close): 0 — ACCEPTED-LOCAL on the first pass** at
  [090-01-yield](iterations/090-01-yield.md) (2026-10-07). One-member wave, anchor `T019.002`, owning
  route implementation Forward, retry owner is this task's own counter, which never incremented: zero rejection, so
  nothing was capped and no repeat reached 2. Routed 2026-10-07 by
  [089-assumption-decide](iterations/089-assumption-decide.md) from [088-codex-review](iterations/088-codex-review.md);
  nested under `T019`, whose counter stays 0 and whose landed increment is not reopened. (An earlier *attempt* at this
  iteration id stopped after writing its ACCEPT draft and before any bookkeeping — a truncated pass, not a rejection;
  nothing incremented on its account.)
- **CR3 wave `T018` + `T019` + `T020` (local, prework of the MS-001 close): 0 — CLOSED, all three members
  accepted-local at [083-01-yield](iterations/083-01-yield.md) on the first pass**, routed 2026-10-06 by
  [080-assumption-decide](iterations/080-assumption-decide.md). One local Yield for the whole wave (range
  `d5df8b06…9c90675`); retry owner is each member's own counter, each still 0 — zero rejection, so nothing
  incremented and no repeat reached the cap.
- Older waves on this boundary (CR2 `T015`+`T016` at 0, `T013` header-parity at 1, `T017`, MS-002/T003 root at 1 —
  counters and their closures verbatim in [handoff-archive.md](handoff-archive.md) §Trimmed 2026-10-07
  (`083-01-yield`)); none is an open retry owner.


## Trimmed 2026-10-17 (`097-yield`) — planned-task blueprints, the live closure-ledger index and every closure/routing section

## Planned Task Context

Written by the assumption-decide routing pass (2026-09-19): the six `pending`-Routing claim rows
(`claim-3`/`claim-18`/`claim-19`/`claim-22`/`claim-32`/`claim-38`) were all routed `absorbed`; every finding amends
a pending task below. No finding is `resolved` — Forward/Yield move the statuses. (T002's claim-19 block was removed
at 032-01 after the task's acceptance — its status had reached accepted-local; full record in git Checkpoint 61 +
`assumptions/verdicts/claim-19.md`.)

<details>
<summary>T021 Planned Task Context — Harden the `MS-001` documented-format gate against content drift (`YR-QA-1`)</summary>

- Owning route: **`forward-qa`** — QA author pass, **test bytes only**. Hard constraint: **no production
  code, no config, no `invariants/`, no `decisions.md` bytes**. One gate-strengthening task, one local Yield.
  `T002` and every accepted record stay accepted — **no task acceptance is reopened**; this is a gate finding,
  not a contradiction of any accepted record.
- Finding: `YR-QA-1 · documented-format gate cannot fail on content drift`, adjudicated at
  [095-yield-red](iterations/095-yield-red.md) §`YR-QA-1` (run `YR-MS001-r2`). Product arm **not** defective:
  `YR-P001` HOLDS — `_write_full_turn` serializes the normalized body (`dump.py:161-189`) and
  `_write_diff_turn` diffs `_render_body` output, and base's `_dump_turn` is byte-identical in content after the
  same `normalize` (`buster_ripper_legacy.py:558-615`, `:642`). The finding is that the **gate would not detect**
  the drift, which is why `YR-P001` does not close it.
- Blind evidence: `assumptions/verdicts/yr-p002a.md` **DOES-NOT-HOLD**, `yr-p002b.md` **HOLDS**,
  `yr-p001.md` **HOLDS**, `yr-p020.md` **HOLDS**. `YR-P006` HOLDS (kills `YR-H003`'s product arm and is not
  this task's subject).
- Loophole, measured: `assert_documented_contents` (`tests/qa/test_qap_ms001.py:104-146`) is shared by `TC-107`
  (`:254`) and `TC-108` (`:284`). Its content observation is **presence-only** — `json.loads` + `model` key +
  marker substring for `turn_000.json` (`:109-111`); exact line count + `README_LINE_PATTERN` (`:72`) + literal
  columns for `index.txt` (`:114-133`); and five presence checks per diff (`:136-146`, where
  `startswith("+")`/`startswith("-")` also match `+++`/`---`, so **any** marker-bearing unified diff passes and the
  diff's source is unconstrained). No byte comparison, no complete value set. The gate's own sensitivity check
  (`:314-345`) covers the `index.txt` matcher only.
- Why no other gate closes it. `tests/test_dump.py:146-154` does compare exact bytes, but it calls `dump_turn`
  **directly**, so it cannot observe the route's normalize-then-dump ordering (`routes/messages.py:38` → `:58`), and it
  builds its expected diff with the module's own `_render_body`. `tests/test_routes_integration.py:128-153` sends a
  nonce-bearing body but asserts only `response.status == 200`; no test asserts the body handed to `dump_turn`. The QA
  harness bodies (`tests/qa/harness.py:102-109`) carry **no billing nonce**, so no `QAP-MS001` run can observe
  normalization at all today. Reach: **`latent`** for writer-level drift (`tests/test_dump.py:146-154` pins the current
  writer); **`reached`** for route-ordering drift (any edit to `normalize`/`dump_turn` ordering is unobserved).
- Required outcome: with no format configured or `--dump-format documented`, `turn_000.json` carries exactly today's
  content — the **normalized** JSON request body, no billing nonce (`README.md:153`: "full normalized request body") — and
  each `turn_NNN.diff` is computed over `_render_body` output of consecutive turns, at `2N` write-bearing operations.
  The contract is **not widened**: `VP-T002-documented-format-preserved` (`plan-summary.md:228`) already says "today's
  content" and its `Evidence needed` column already names "file contents"; `qa-plan.md:213`/`:217` say "content and line
  format" and "listing + contents + the instrumented write count"; `qa-plan.md:268` already claims
  "byte-unchanged". This task makes the gate observe the contract it already has.
- Done when: a documented-format run whose writer is mutated to serialize the **pre-normalization** body (or whose route
  hands `dump_turn` that body), **or** to diff **raw request bodies**, while keeping every filename, the `N + 1` count, the
  index line shapes and the `2N` decomposition, **fails** the milestone gate; the unmutated writer **passes** it. Both
  mutations are demonstrated red and reverted byte-exact. The route-ordering arm needs a nonce-bearing request, because a
  nonce-free body is identical before and after `normalize` and cannot distinguish the ordering.
- Harness hazard, recorded because it bites the route arm: `harness.session_id` (`tests/qa/harness.py:75-77`) hashes the
  **raw** system text, while the proxy keys `sid` on the **post-normalize** system (`routes/messages.py:38-39`,
  `session.py:93-98`). A nonce-bearing body therefore lands in the nonce-free `sid` directory — compute the expected dir
  from the nonce-free system, and keep every nonce-free scenario's body bytes and `sid` derivation unchanged.
- Yield replay: (a) unmutated writer → the strengthened gate green (`QAP-MS001` still 11/11, `TC-107`/`TC-108`
  both green); (b) mutate the writer to serialize the pre-normalization body (or move `dump_turn` before `normalize`) →
  the strengthened gate reds while filenames, `N + 1`, index shapes and `2N` stay; revert byte-exact; (c) mutate the diff
  source to `json.dumps(body)` → reds (the raw-body diff reports `-1` on every turn, which `CHANGE_CLAUSE`
  (`:75`) accepts, so the `-R`/rendered-line observation is the discriminator, not the regex); revert byte-exact;
  (d) `git diff --name-only` over the increment contains zero production, config, plan, contract or registry bytes.
- Capacity answer: `tests/qa/test_qap_ms001.py` is **617** raw lines (already in the extract-file band, 501–700), so
  additions keep it in the same band and cross none; `tests/qa/harness.py` is **312** (review band, 301–500) and a small
  optional nonce parameter keeps it there. A focused new QA module is also admissible if the gate file's size is judged
  worth splitting; that choice is the implementation's.
- Invariant cross-reference: **Neutral** — no invariant is touched or created; the gate strengthens an existing `VP-*`
  observation. Zero `invariants/` bytes. The session's protected-yield invariants (flush contains no `await`; cannot raise
  out of `on_cleanup`; idempotent per pending batch; `turn`/`prev_text` advance at append time) are untouched by a
  test-only task, and any temptation to edit `dump.py` to make the gate easier is out of lane — stop and route back.
- Closure ledger: **none owed** — `YR-QA-1` is a gate finding; the adjudication creates no `DC-*` row for it, and this
  task creates none. No assumption routing, no question, no Fork: `FK-011` stays the only open Fork and is untouched.
- Assumption impact: **`required:new-surface`** — the strengthened gate asserts new propositions (the `turn_000.json`
  value set and the diff's source) that no existing seam covers; `scoped:` is not used because the nearest seam
  (`S-YR-4`/`S-YR-4′`) was malformed on premise wording and no composition artifact exists. Extract and check the
  new surface before Forward; no full re-scan, no production re-derivation.
- Verification profile: Outcome — settled (`VP-T002-documented-format-preserved`, `TC-108`, `G-008`(a), `US-012`/`US-014`).
  Risk — **high**: the wrong outcome is the milestone closing with its only content witness unarmed; exposure is the
  default documented path on every run; containment is test bytes only (no production, data or external effect).
  Recovery — **easy**: revert one test-only commit; the mutation replays are scratch and reverted byte-exact.
  Uncertainty — whether both mutations are black-box distinguishable with the harness's observables, and the `sid` hazard above.
  Scope — `tests/qa/test_qap_ms001.py` (`assert_documented_contents` `:104-146`, `TC-107` `:254`, `TC-108` `:261-311`,
  sensitivity `:314-345`), `tests/qa/harness.py` (`turn_body` `:102-109`, `session_id` `:75-77`), `qa-plan.md` §TC-108
  (`:210-217`) + traceability (`:268`); referents `dump.py:161-189`, `routes/messages.py:38,58`,
  `README.md:153,166-167`, `test_dump.py:146-154`, `test_routes_integration.py:128-153`.
  Validation — the two mutation reds + unmutated green above.
  Route per gate — assumptions **parallel** (bounded, contained, easy recovery); Forward Red **compact** (behavior and
  done-when are settled; one bounded misread attempt, e.g. a gate that builds its expectation by calling `_render_body`, which
  would stay green under the mutation and mirror the implementation); Yield Red **full on this surface** (this task is itself a
  Yield-Red remediation; the closing check must replay both compliant-but-wrong outcomes against the strengthened gate).
  Evidence reused — the four `yr-p*` verdict files, `095` §Adjudication, `test_dump.py:146-154`,
  `test_routes_integration.py:128-153`; budget — one assumption bundle over the new gate surface + one composition for the
  route-ordering seam; no full re-scan.
- Execution order: **before the `MS-001` boundary's re-adjudication** (the closing Yield Red check). That check owns
  `resolved` for `T015`–`T020`, `T019.001`, `T019.002` and `MS-001`, phase acceptance, the `Delivered` entry and parent
  completion — all withheld until this gate passes. `093`'s `SEMANTIC-APPROVE` is byte-scoped to `62f8d7c`; the closing check
  runs on the strengthened bytes. Not a new `MS-001` constituent: the task gates an existing `VP-*`, it delivers no product
  clause, so the Milestone Matrix and the `VP-*` matrix are unchanged.
- Deferred, with its owner named: `CR5-F-01`'s docstring wording (`094-assumption-decide`) is that pass's lane, not this
  task's; `T018`'s framework-default body limit stays the accepted risk recorded in `095` §Open. Neither is reopened here.

</details>

<details>
<summary>T011 + T012 Planned Task Context — both findings terminal; blocks MOVED verbatim to handoff-archive.md (preserved by acceptance Checkpoint 121)</summary>

Both blocks live at [handoff-archive.md](handoff-archive.md) §Trimmed 2026-10-05 (053-01-yield follow-up), where they remain the full record of the
two findings' Counterexample, Required behavior, Done when, Yield replay, verification profile, invariant candidate and wave-separation rationale.
Both are now terminal: `seam-S6-disable-form` resolved 2026-10-05 (§Post-acceptance Finding Resolution — T011: `resolved`; `CO:` basis disposed) and
`seam-S6-value-word-split` resolved 2026-10-05 (§Post-acceptance Finding Resolution — T012: `resolved`; `CO:` basis republished), the latter's three
riders discharged as recorded in §Current Status. **Nothing is owed from either block.** `assumptions/verdicts/claim-*.md` and the retained
`CO:serve-script-env-to-argv/env-var-flag-crossing` history (`fb247496…` → `dc54c8c5…` → `d20c1f6b…` → `cb7c0a5c…`) remain the permanent authority chain.

</details>

<details>
<summary>T006 Planned Task Context — CONSUMED at acceptance (051-01-yield); full blocks preserved in the acceptance checkpoint</summary>

claim-38 (connector-loop mechanism attribution corrected) and claim-34 (import-bound CLI defaults) — both routed
`absorbed` (DC-claim-38-r1 / DC-claim-34-r1), implemented by [051-01-forward](iterations/051-01-forward.md), and
verified by [051-01-yield](iterations/051-01-yield.md): both `Yield replay` instructions executed — connector+session
inside the running loop serves both postures from one construction with no `RuntimeError` (PASS), the explicit opt-out
sets the global at parse time before `on_startup` (PASS), and the factual counterexample is CONFIRMED (a post-import
flush-global mutation is reset to the import-time decorator default — defaults are import-bound, not dynamically
rebound). Statuses: **accepted-local** (051-01-yield); `resolved` awaits the orchestrator's post-implementation
atomic/composition recheck. Canonical records: `assumptions/verdicts/claim-{34,38}.md`.

</details>
*(Consumed blocks archived with their acceptance checkpoints (61–109); canonical records in `assumptions/verdicts/claim-*.md` and the accepting yield reports: T003.001→043-01, T007→044-01, T008→045-01, T008.001→046-01, T009/T010→041-02, T010.001→042-01, T003 claim-18/claim-22→032-01, T005 claim-32→049-01, T004 claim-93/claim-3/claim-20→050-01-yield — replays 24/24, statuses resolved 2026-09-28 (S1/S2 SAFE, §Post-acceptance Finding Resolution), **T006 claim-38/claim-34→051-01-yield** — replays PASS/CONFIRMED; claim-38 **resolved** 2026-09-28 (fresh post-T006 S4 SAFE composition), claim-34 **resolved** 2026-09-28 (fresh S3/S5 SAFE recompositions) — §Post-acceptance Finding Resolution — 2026-09-28 (T006) and — 2026-09-28 (claim-34).)*

<details>
<summary>T013 / T014 Planned Task Context — both waves accepted and CLOSED (067-01-yield, 068-01-yield); blocks MOVED verbatim to handoff-archive.md (preserved by acceptance Checkpoint 168)</summary>

Full blocks in [handoff-archive.md](handoff-archive.md) §Trimmed 2026-10-06 (073-01-yield) → `T013/T014 Planned Task
Context blueprints`, including the `### Assumption findings routed into T013/T014` subsections (claims run 32 and the
S10–S13 composition passes) that rode inside them. Neither wave is reopened by the CR2 batch; `T013`/`T014` accepted
scope is re-verified as untouched at [073-01-yield](iterations/073-01-yield.md) §Ownership boundaries, and nothing in
this archive is owed by the remaining MS-001 boundary work.

</details>

### Trimmed 2026-10-07 (`083-01-yield`) — the accepted CR3 wave and the two older accepted waves

The `### T015`/`T016`/`T017`/`T018`/`T019`/`T020 Planned Task Context` blocks moved verbatim to
[handoff-archive.md](handoff-archive.md) §Trimmed 2026-10-07 (`083-01-yield`) — context budget: this file carries
the active route, not six closed blueprints. All six are `accepted-local`; their durable dispositions stay in
the `### DC-*` closure rows below and in
[080](iterations/080-assumption-decide.md) / [081](iterations/081-assumption-decide.md) / [082](iterations/082-assumption-decide.md) / [083-01-yield](iterations/083-01-yield.md)
(pre-trim bytes preserved by acceptance Checkpoint 193, `eefd718`). The block that was active below at that
acceptance — `T019.001` — moved with its own acceptance one pass later; see the next section.

### Trimmed 2026-10-07 (`085-01-yield`) — `T019.001` accepted-local

The `### T019.001 Planned Task Context` block moved verbatim to [handoff-archive.md](handoff-archive.md) §Trimmed 2026-10-07 (`085-01-yield`) — the task it blueprinted is `accepted-local` at [085-01-yield](iterations/085-01-yield.md), and no active task remains in this file: the boundary's next acts are the assumptions recheck and Yield Red, which address the `### DC-*` closure rows and the `assumptions/` ledgers below, not a blueprint. Its one live hand-off — the **Invariant candidate** for Interview — is restated in §Current Status so it cannot be lost to the move.

### Trimmed 2026-10-07 (`090-01-yield`) — `T019.002` accepted-local

The `### T019.002 Planned Task Context` block moved verbatim to [handoff-archive.md](handoff-archive.md) §Trimmed
2026-10-07 (`090-01-yield`) — the task it blueprinted is **`accepted-local`** at
[090-01-yield](iterations/090-01-yield.md) (first pass, counter 0; zero reject-tier findings; both Yield mutations
executed and reverted byte-exact; hermetic 511 @ 96.71% re-measured). Two things outlive the blueprint and are
restated here so they cannot be lost to the move: **(1)** the boundary's ordering constraint — `T019.002` landed
**before** the MS-001 assumptions recheck, which is `S16`–`S19`'s one allowed recomposition and must run on these
final bytes, re-pinning `claims.md` rows `55`/`134`/`218` and accounting for `normalize.py`'s `+13` shift (rows
`264`–`266` bind `normalize#normalize` by symbol, byte-identical, so they do not re-base on content); and **(2)**
`resolved` is **Yield Red**'s, not this acceptance's — `DC-CR4-F-01-r1` stays open for it, and `Yield Phase`
re-executes `QAP-MS001` on these bytes in between (`087`'s `READY-PASSING` is byte-scoped to `8dfa5af`).
`roadmap.md` §RM-012 still owns the `/v1/messages` malformed-element posture; its read-site census was corrected by
the Yield pass to **five sites** (`normalize.py:88`/`:89`/`:90`/`:130`/`:205`), which is what clause 5's witnesses
now pin against drift.

## Deterministic Closure Ledger

Live entries: **the CR4 row `DC-CR4-F-01-r1`** (2026-10-07,
[089](iterations/089-assumption-decide.md), below — one task, `T019.002`, one-member wave, dispatched, implemented at
[090-01-forward](iterations/090-01-forward.md) and **accepted-local at [090-01-yield](iterations/090-01-yield.md)**,
first pass, counter 0; `resolved` is Yield Red's, so the row stays open for that lane. It is no longer the only ledger
row on this boundary fronting undispatched work); the three **run-45** rows `DC-claim-418-r1`,
`DC-claim-422-r1`, `DC-claim-423-r1` (2026-10-07, [084](iterations/084-assumption-decide.md), below — `T019.001`,
accepted-local at [085-01-yield](iterations/085-01-yield.md), `resolved` is Yield Red's); the eight **run-42 composition**
rows `DC-S16-a-r1`, `DC-S17-CE-1-r1`, `DC-S17-CE-2-r1`,
`DC-S18-CE-1-r1`, `DC-S18-MC-1-r1`, `DC-S19-CE-1-r1`, `DC-S19-CE-2-r1`, `DC-S19-CE-3-r1` (2026-10-06,
[082](iterations/082-assumption-decide.md), below — all eight land on the same three pending tasks; two blueprint
clauses amended, no new task, wave **accepted-local at 083-01, recomposition owed by the boundary's recheck**);
the five **CR3** rows `DC-CR3-F-01-r1`…`DC-CR3-F-05-r1` (2026-10-06,
[080](iterations/080-assumption-decide.md), below — three tasks, one wave, **open**); the two **run-40 atomic** rows
`DC-claim-399-r1`, `DC-claim-409-r1` (below); the four **CR2** rows `DC-CR2-F-1-r1`…`DC-CR2-F-4-r1` (2026-10-06,
below — all four **implemented and accepted-local** at
[073-01-yield](iterations/073-01-yield.md); `F-4` by record amendment only, per its own selected closure); and
`DC-S15-CE-2-r1` (2026-10-06, below — accepted-local, `S15` terminal per the 076 recomposition routing; the
`resolved` stamp is Yield Red's). Every pre-CR2 row — the 2026-10-05 routing batch, both repair rounds, and the
nineteen S10–S13 composition rows — is archived verbatim with its routing sections in
[handoff-archive.md](handoff-archive.md) §Trimmed 2026-10-06 (073-01-yield), together with the superseded census
paragraph that used to open this section (preserved byte-identical in the same archive block and in acceptance
Checkpoint 168). None of the archived rows is owed by the remaining MS-001 boundary
work: `T013`/`T014` are accepted and closed, `T011`/`T012` resolved, the `056` P3s routed at the 2026-10-06 drain.
`F-5` has **no** ledger entry — the gate left it unranked, so it is [`FK-011`](decisions.md#fk-011), still the only
open Fork. Full records for the earlier batches remain in git Checkpoint 61 + `assumptions/verdicts/claim-*.md`.
## Deterministic Closure Ledger — 2026-10-05 batches (routing, repair rounds 2-3, composition routing S10–S13)

All forty-five pre-CR2 rows and their routing sections moved verbatim to
[handoff-archive.md](handoff-archive.md) §Trimmed 2026-10-06 (073-01-yield) → `Deterministic Closure Ledger — 2026-10-05
…`. Every one of them belongs to a consumed wave or an already-disposed route (`T013`/`T014` accepted and closed at
[067-01-yield](iterations/067-01-yield.md)/[068-01-yield](iterations/068-01-yield.md); `T011`/`T012` resolved; the
`056` P3s routed at the 2026-10-06 drain), and none is owed by the MS-001 close. **The live entries now are the four
`DC-CR2-F-1-r1`…`DC-CR2-F-4-r1` rows below**, whose routing preamble follows the ledger unchanged.


## Deterministic Closure Ledger — 2026-10-06 routing (MS-001 boundary, post-071 codex-review — CR2 batch)

Four `absorbed` findings routed by this closure-packet pass (the batch's stable ids are `CR2-F-1`…`CR2-F-4` — the
producer numbered them `F-1`…`F-4`, which collides with `057` §E's findings and the `T013`/`T014` member letters
already keyed in this ledger, so the batch is namespaced `CR2` (second `codex-review` pass of the session) and the
mapping is recorded in the packet copy). Two further packet items are **not** here: `CR2-X-1` is `malformed` (no
ledger row by rule) and `CR2-X-2` is `accepted` (no ledger row by rule); both are dispositioned in
§Closure-packet routing — 2026-10-06 below.

Every disposition was **re-derived against current bytes**, not adopted: all four findings hold at HEAD `dbd3072`
(none already-closed — no `-r2` of an earlier key applies; the earlier `DC-F-*`/`DC-G-*` keys belong to different
findings). Base contract read from the tracked pre-refactor copy (`buster_ripper_legacy.py:882-892`, `:716-718`,
`:911-913`) per the 2026-10-05 method, not git objects. Two packet details were corrected on the way in and are
carried to the tasks: `CR2-F-2`'s "no witness can produce a non-parseable upstream answer" is true of
`tests/upstream_stub.py` but not of `tests/test_routes_integration.py:239-243`, which *can* and *does* — and pins
the defect shape the closure discards; `CR2-F-4`'s "harm description understates" premise matches no session
record (both the roadmap entry and the backlog pointer already state the second-head shape — the quoted phrase
"partial body without a terminator" appears nowhere; grepped). Repository state for all four is the same pre-write
`HEAD`, captured by Git from the supplied worktree root before the first routing write and never recomputed after.

**The review's head-label is stale — a label defect, not a verdict defect (persisted by
[072-01-forward](iterations/072-01-forward.md) at Checkpoint 163, which added this paragraph and changed nothing
else in the batch).** The packet's headline names head `2be9783` (Checkpoint 141 — T013 retry 1's `fix(chat)`
commit); the review **ran against the tree at `dbd3072`** (Checkpoint 162), and the baseline it printed is the proof.
It reported **407 passed**, and 407 is `dbd3072`'s figure: [070-forward-qa](iterations/070-forward-qa.md):168 and
[071-yield-phase](iterations/071-yield-phase.md):436 each record **407 passed @ 95.75%** on this line (`c7420ce` and
`dbd3072` differ only in session records — `git diff --stat c7420ce..dbd3072` → 7 paths, every one under
`.sessions/codify/…`), while the recorded hermetic figure on `2be9783`'s own tree is **374 passed, 94.20%**
([067-01-yield](iterations/067-01-yield.md):109), with [068-01-yield](iterations/068-01-yield.md):28 holding the
intermediate 406. The two counts cannot describe one tree:
`git diff --stat 2be9783..dbd3072 -- buster_ripper tests` → **12 files, +1568/−279** (`dump.py`, `image.py`,
`routes/count_tokens.py`, `tests/image_bodies.py`, `tests/test_image.py`, `tests/test_image_stability.py`,
`tests/qa/test_qap_ms001.py` among them — T014's wave and 070's authoring both landed after Checkpoint 141). Why the
verdict is unmoved: this pass adopted no line number from the packet — every site was re-derived at `dbd3072` (the
packet's §ROUTING-PASS RECHECK NOTES site map is `dbd3072`'s, as are the four `Repository state` rows above), all
six items hold there, none was already-closed. A review that ran on the current tree and mislabeled which commit it
ran on has produced findings **about the current tree**; the batch needed no re-run. Two consequences later passes
must use: **(1)** pointer re-checks for this batch resolve against **post-T014 bytes** — checking a
`chat.py:152`-style citation at `2be9783` lands in the wrong function and makes current evidence look drifted;
**(2)** the baseline is the identity check — a repair branch that cannot reproduce **407** hermetic tests is not on
this tree. What recording this touches: nothing. No ledger row, disposition, or packet byte is altered (the copy is
checkpointed byte-identical, producer wording kept visible rather than corrected), `MS-001`'s `SEMANTIC-APPROVE`
gate stands, `CR2-X-1`'s `malformed` return and `CR2-X-2`'s accepted risk stand, `FK-011` stays the only open Fork.

### DC-CR2-F-1-r1
Finding: CR2-F-1 (packet `F-1`, P1 — non-stream chat responses rebuilt with only `Content-Type`)
Evidence: `evidence/codex-review-post-071-packet.txt` (sites re-verified at `chat.py:207-211`, `:215-220`)
Evidence fingerprint: none recorded by the producer; reviewed head `2be9783` (Checkpoint 141), re-verified at `dbd3072` (Checkpoint 162)
Repository state: git-commit:dbd3072403cc5014b19be229725dbc974337658a
Selected closure: build the headers of both `_apply_response_transforms` return paths from `utils.response_headers(resp.headers)` (thread the upstream headers into the function); relay upstream end-to-end headers minus hop-by-hop and `content-length`
Closure basis: base contract `buster_ripper_legacy.py:882-891` relays `_response_headers(resp.headers)` on the fixed and the fall-through path alike; the in-tree production shape for the same direction is messages.py `:285-290` (`utils.response_headers`, docstring records the `content-length` derivation rule); the `T013` wave's settled direction — restore base header handling at every proxy boundary — makes the chat response side the sibling of its accepted request side; no active EI/OI governs response relay (registry holds only `planned` entries), so no contradiction to route
Rejected candidates:
- fixed allowlist of relayed names: no settled authority names one, and it breaks base parity (inadmissible under the boundary-parity direction)
- leave `Content-Type`-only: preserves the verified defect
- raw-bytes passthrough when unfixed (avoid re-serialization): correct but additive to the header closure — deferred below, not competing
Risk shape: unchanged — restores base behavior on branch-added surface inside the session's own review lane; no new mechanism, no cumulative session risk beyond the header-parity lane already accepted at `T013`
Deferred additive work: `QB:routes-chat/raw-bytes-passthrough-when-unfixed` (new backlog line, this pass) — trigger: next touch of the chat non-stream relay (lands with T015 if it fits, else the next chat wave); owner: Forward touch-set pickup
Task: T015 step 1

### DC-CR2-F-2-r1
Finding: CR2-F-2 (packet `F-2`, P1 — unconditional `await resp.json()` erases well-formed non-JSON upstream answers; `{"error": str(e)}` leaks the internal upstream URL; the same arm shape at messages.py and passthrough.py)
Evidence: `evidence/codex-review-post-071-packet.txt` (sites re-verified at `chat.py:152`, `:155-159`; `messages.py:158-160`, `:291-293`; `passthrough.py:48-49`; leak body observed live by the 069/071 gates as a proxy-local `502 Cannot connect to host <upstream>`)
Evidence fingerprint: none recorded by the producer; reviewed head `2be9783`, re-verified at `dbd3072`
Repository state: git-commit:dbd3072403cc5014b19be229725dbc974337658a
Selected closure: chat non-stream arm reads the body once and parses only where a transform or the overflow check needs it; on parse failure relay raw content + upstream status + `response_headers(resp.headers)` (base fall-through parity); genuine transport failures keep a 502 whose client-facing body carries no exception text; apply the static-body rule to the messages.py and passthrough.py arms
Closure basis: base contract — legacy `:885-886` `except Exception: pass  # fall through to normal response on parse error` onto the raw relay at `:888-892`; the settled in-tree shape at messages.py `:264-290` (read once, relay raw, `response_headers`); the `502`/`{"error": str(e)}` arms are **branch-added** (base carried no such arm at all — connection failure propagated to the framework), so this is branch-surface repair inside the session's lane, not an inherited-code touch; the leak half is ranked by the gates' own recorded witness that upstream topology reaches the wire today
Rejected candidates:
- answer 400 to a non-JSON upstream answer: the upstream's answer is not the client's error; no authority authorizes it and it discards upstream status a proxy exists to relay
- swallow parse failure and re-serialize `{}`: fabricates a body, preserves the defect
- restore base's *no-arm* propagation for `ClientError`: larger than the required outcome (would churn currently-green 502 witnesses with no authority asking it) — the packet's 502-with-static-body shape is the smallest complete remediation
- keep `str(e)` for debuggability: the exception remains in `log.error`; the client does not need the internal URL
Risk shape: unchanged — chat bytes return to base parity for the shapes the branch changed; two existing witnesses pin the defect (`tests/test_routes_integration.py:239-243` asserts the synthetic 502 body; `:179-192` mechanism-pins `upstream.json`) and are replaced with fail-first relay/parity witnesses — that is fixing the cause, not weakening a valid test, because the pinned expectation contradicts the base contract the session restored elsewhere
Deferred additive work: none beyond `DC-CR2-F-1-r1`'s (same surface)
Task: T015 step 2 (chat arm) + T016 step 2 (messages/passthrough arms)

### DC-CR2-F-3-r1
Finding: CR2-F-3 (packet `F-3`, P1 — `"chat_template_kwargs": null` under shipped-default `PRESERVED_THINKING=1` → `AttributeError` → HTTP 500, no upstream call)
Evidence: `evidence/codex-review-post-071-packet.txt` (site re-verified at `messages.py:45-48`; `serve_buster_ripper.sh:30` default re-verified; `dict.setdefault` semantics confirmed by reading the block; nothing above `handle_messages` catches; no earlier stage touches the key — grepped `normalize.py`/`image.py`/`session.py`)
Evidence fingerprint: none recorded by the producer; reviewed head `2be9783`, re-verified at `dbd3072`
Repository state: git-commit:dbd3072403cc5014b19be229725dbc974337658a
Selected closure: isinstance-check `body.get("chat_template_kwargs")` and replace with a fresh dict when it is not a mapping, then inject — the flag's injection guarantee holds for every request body; branch-added surface, guard belongs at the injection site
Closure basis: the documented contract of the flag itself (`main.py:107-110` — "Inject enable_thinking=true + clear_thinking=false into /v1/messages requests"); package convention (`normalize.py`, `image.py` guard every non-dict input); exposed API — a caller-supplied key is reached by definition under the Reach Gate, and the shipped launcher runs the arm live
Rejected candidates:
- swallow-and-pass (chat.py's `try/except Exception: pass` shape): forwards `null` and injects nothing — violates the flag's documented guarantee, so it preserves a required-behavior defect even though the crash goes away (rejected by the producer reviewer and independently by this gate's defect-preservation criterion)
- answer 400: no settled authority forbids a null-valued optional field; base validated nothing here; it invents a rejection surface the API does not have
- resolve the posture alongside: `FK-011` is the only open Fork and owns on/off + documentation; the guard is posture-independent and this task pre-decides none of it
Risk shape: unchanged — one guarded block on the flag-on arm; the flag-off path is byte-identical
Deferred additive work: none
Task: T016 step 1

### DC-CR2-F-4-r1
Finding: CR2-F-4 (packet `F-4`, P2 — chat SSE relay writes a second HTTP head after a post-commit failure; byte-level repro: two `HTTP/1.1` status lines in one response, the 502 inside the committed 200's body)
Evidence: `evidence/codex-review-post-071-packet.txt` (mechanism re-verified at `dbd3072`: `_relay_chat_stream` called from inside `_make_chat_request`'s `try` at `:151`, commits at `:184`, a mid-relay `ClientError` returns the fresh 502 at `:155-157` onto the committed writer)
Evidence fingerprint: none recorded by the producer; reviewed head `2be9783`, re-verified at `dbd3072`
Repository state: git-commit:dbd3072403cc5014b19be229725dbc974337658a
Selected closure: **record amendment only, no code change in this session** — refresh the stale line pointers in `roadmap.md` §Chat route's post-commit second head and `QB:routes-chat/post-commit-second-head` (`:170-184`→`:174-188`, `prepare()` `:180`→`:184`, except arm `:151-153`→`:155-157`), append the CR2 byte-level repro as confirmatory evidence, and append the reviewer's prioritization flag. Executed by this pass (precedent: `DC-S12-CE2-r1` amended the `QB:cli-surface` marker in place)
Closure basis: the `codify-scope` rules 1–2 ruling at 066-01/067-01-yield stands and is **explicitly rechecked against the new evidence and not changed**: the repro confirms the harm the records *already* state ("a second status line inside the first response's body" / "prepares a second head inside the committed 200 body") — it adds confidence, not a new shape, and nothing about it blocks a session gate (MS-001/002/004 all passed over this unchanged surface); the producer packet itself directs no code change here; the fix's debts (`R-E`'s shape a second time, its own raw-bytes witness set, a `seam-S11` reading, own branch off main) are recorded on the handle
Rejected candidates:
- open a session task for the fix: rule 2 (no session clause names it; byte-unchanged since before `T013`; folding it in ships an unattributed behavior change — the exact reasoning of the standing ruling); rule 3 does not fire — no session result needs it
- rewrite the harm wording: nothing to correct — the packet's premise that the record said only "partial body without a terminator" matches no session record (grepped); appending the repro is the honest amendment
- elevate to the user: the ownership decision is settled authority; the prioritization flag ("should not sit indefinitely behind other roadmap rows") is recorded on the handle for the roadmap owner, not converted into a question — `FK-011` stays the only open Fork
Risk shape: unchanged — zero code moves
Deferred additive work: the fix itself, owned by `roadmap.md` §Chat route's post-commit second head (trigger: the roadmap owner's next scheduling pass, elevated by this batch's flag)
Task: none — record amendment executed by this pass (roadmap.md + quality-backlog.md pointers amended 2026-10-06)

## Deterministic Closure Ledger — 2026-10-06 routing (composition seam-S15, MS-001 boundary)

One `absorbed` finding routed by this composition pass; its sibling `S15-CE-1` is `accepted` and takes **no**
ledger row by rule (§Accepted risks, §Elevated to user, `roadmap.md` §RM-008). The composition's Notes N1 and
N2 are records, not findings, and open nothing. Both dispositions rest on evidence produced by execution —
`evidence/s15-reach-and-base-comparison.txt` — because the composition traced the producer/consumer mechanism
and covered neither **reach** nor the **base** side of the comparison, which is what split the two findings.
Full reasoning: [074-assumption-decide](iterations/074-assumption-decide.md).

### DC-S15-CE-2-r1
Finding: S15-CE-2 (repeated occurrences of one upstream response header name are folded to the last at `utils.response_headers`)
Evidence: `assumptions/compositions/seam-S15.md` §Counterexample CE-2 + §Reach; reach and base comparison established in `evidence/s15-reach-and-base-comparison.txt` §2, feasibility in §3
Evidence fingerprint: sha256:74ac46583a4976fab03d873c3a937eb1906f14476dd646d5cbd6422155d437ca (the composition's own basis; its six member digests were re-verified against the working tree by the composition pass and matched)
Repository state: git-commit:8438158a321b6fa701c67e48aec0c0f3e9cd3c06
Selected closure: make `utils.response_headers` return a per-occurrence-preserving header collection (a `CIMultiDict`, or the pairs themselves) instead of a plain `dict`, dropping exactly the names it drops today; all three relay sites keep handing its result straight to `web.Response(headers=…)`; add the missing repeated-name witness per boundary plus a `utils.response_headers` unit witness
Closure basis: the session-settled direction `T013` — *restore base header handling at every proxy boundary* — and `tests/test_header_parity.py`'s own declared contract, "Pre-refactor (base) proxy semantics are the contract here"; `T015`'s accepted required behavior 2 ("upstream status, upstream bytes verbatim, filtered upstream headers"); the base comparison, run on one header head through both clients — `buster_ripper_legacy.py:630-637` over `httpx.Headers.items()` (which concatenates occurrences: `{'date': 'T1, T2', 'set-cookie': 'a=1, b=2'}`, `httpx/_models.py:216-229`) preserved **every** occurrence's value, while `buster_ripper/utils.py:36-40` over aiohttp's per-occurrence `.items()` (row 388, `CL:aiohttp/repeated-header-items-iteration`) keeps the last and destroys the rest — so the loss is **branch-introduced, not base parity**, and repairing it serves settled authority instead of deviating from it; reach is `reached`, first-party on the deployed path (6/6 probed responses send `date` ×2 and `server` ×2, 1/6 of them with *differing* `date` values; the running proxy answers with one line); `codify-scope` rule 5 places the fix where the cause is — this session's own shared filter, not another component
Rejected candidates:
- comma-join duplicates (httpx's merge, byte-identical to base's shape): transforms the representation instead of relaying it, so it breaks `T015`'s "verbatim" clause, and it is invalid for names whose definition forbids joining — the composition's own concrete input is `Set-Cookie: a=1` / `b=2` (RFC 9110 §5.3), so this candidate knowingly preserves part of the verified defect
- keep the `dict` fold and document last-wins in the docstring: preserves the verified defect on a `reached` input and leaves the accepted `T013` parity claim false on the point its own census never tested (no witness feeds a repeated name — `grep -rn "Set-Cookie\|multi_items\|getall" tests/` → zero)
- route it to the header-lane design pass alongside CE-1: that route's only stated reason was "Both are **base parity**"; refuted for this half by the base comparison, so deferring would bank a reached information loss on a premise this pass disproved
- fix it at the three call sites (per-site accumulation): breaks `062 R-A`'s binding that the five sites delegate in place to one shared implementation, and multiplies the mechanism threefold for no benefit
- widen the drop set (drop `content-encoding`, or relay `identity` on egress): CE-1's posture question, out of this task's scope; the row stays confined to occurrence preservation
Risk shape: unchanged — one comprehension's return type in one shared helper the session already owns, no new mechanism, no name added to the drop set, no route's request-side behavior touched; `web.Response` already emits one line per occurrence (verified), and `CIMultiDict == dict` still holds so the existing kept/drop witnesses survive unweakened; residual risk is the census witness `test_utils_holds_the_only_two_filters` (`source.count("for k, v in") == 2`), which the comprehension-shaped form keeps satisfied and which the done-when pins
Deferred additive work: `roadmap.md` §RM-008 — the content-coding posture ruling for CE-1, owner the header-lane design pass, trigger the next decision about relayed names or the first upstream that answers encoded; and the egress-direction fold (`forward_headers`), which is base parity and S10's terminal boundary — re-open only through §RM-008, never as a ride-along here
Task: T017 step 1 (wave anchor, sole member; local Yield; prework of the MS-001 close — land before Yield Red)
Status: **absorbed → implemented → accepted-local** at [075-01-yield](iterations/075-01-yield.md) (2026-10-06, `6a13bd7`). Every done-when clause re-derived independently, the replay executed (36 focused witnesses, the raw-socket line count among them), hermetic 455 @ 95.75% re-run, and the four prohibitions proven by the one-file production diff. Closure is **local**: `S15` still owes the one recomposition named in §Composition routing, and `resolved` stays Yield Red's.

## Deterministic Closure Ledger — 2026-10-06 routing (MS-001 boundary, CR3 batch — codex review persisted at 079)

Five `absorbed` findings routed by this closure-packet pass (`CR3-F-01`…`CR3-F-05` — ids assigned by the
persistence pass in relay order, as `079` §Naming records; this router keeps them). The batch's other eleven
items are `accepted` — no ledger row by rule; each carries its Reach verdict and citation in §Accepted risks.
Every disposition was **re-derived against current bytes**, not adopted: `git diff --name-only 6508941..HEAD`
at read time contains no file outside `.sessions/`, so the sites read are the reviewed bytes; the persistence
pass's three marked notes (two locators, one ruff premise) were all verified, and the ruff **count** was
re-derived with `uvx ruff check --select F` → the relayed 8, all in `tests/`. Repository state for all five is
the same pre-write `HEAD` (`4008e27b…` — Checkpoint 185, the CR3 evidence checkpoint), captured by Git from the
supplied worktree root before the first routing write and never recomputed. The head label needs no settling:
unlike CR2, this review's head, baseline (455 @ 95.75%) and reviewed tree all resolve to `6508941` (079
§provenance). Full reasoning, the two scope rulings the dispatch asked to be made explicitly (F-01 absorbed
rather than routed out, under the 057/CR2 one-merge-unit precedent; the F-03 item-2 tension resolved as
mechanism-cleared vs configuration-indicted) and the per-finding Reach analyses:
[080-assumption-decide](080-assumption-decide.md).

### DC-CR3-F-01-r1
Finding: CR3-F-01 (P1 — `web.Application` carries no `client_max_size`, so aiohttp's 1 MiB default 413s large bodies on every route; migration-introduced, base imposed no cap)
Evidence: `iterations/079-codex-review.md` §CR3-F-01 (re-verified at `4008e27b…`: sole construction site `server.py:119`; `web_app.py:128` default `1024**2`; `web_request.py:675` skips the check on falsy value; tree grep: zero acknowledgment of cap/413/1 MiB outside `.venv` — the two `1048576` hits are `DUMP_FLUSH_BYTES` CLI tests)
Evidence fingerprint: unavailable (dispatcher-relayed text persisted at Checkpoint 185; each clause re-established on current bytes by this pass)
Repository state: git-commit:4008e27b5394d9bd994b0131bf47a94dff061d92
Selected closure: `create_app()` passes `client_max_size=0` (aiohttp's documented falsy=unlimited, re-verified in installed source), with a comment naming the base-parity reason; fail-first witnesses at the real-server tier (`proxy`/`raw_proxy` fixtures exist): 2 MiB POST to `/v1/messages`, `/v1/messages/count_tokens`, `/v1/chat/completions` and the catch-all → the 413 class gone at all four
Closure basis: base parity — the session's settled contract for the migration (`tests/test_header_parity.py:4` "Pre-refactor (base) proxy semantics are the contract here"; the direction `T013`–`T017` applied at every boundary) and base (Starlette/uvicorn) accepted any body; reach **reached** under the proxy's own contract: `MAX_MODEL_LEN 200_000` (`config.py:59`) at the package's own 4-chars/token estimate (`session.py:89-90`) ⇒ ~0.8 MB before escaping/multibyte, image payloads (the `--strip-images`/`image.py` cluster) cost ~1.6 K model-tokens per megabyte, documented client is Claude Code at the proxy (`README.md:52-65`) — the cap sits **inside** the served range; no witness ever posted over it (blind spot, not safety); absorbed per the 057/CR2 precedent (branch-wide migration defects enter this session as waves; the branch merges as one unit; standing instruction to finish coding before boundary ceremony)
Rejected candidates:
- explicit large cap (any fixed N): preserves an artificial rejection base never imposed, and no settled authority ranks any N — a parity deviation disguised as a fix
- new config/CLI cap: surface growth no requirement names (the plan's own dial-by-dial refusals are the pattern)
- keep the default, document the cap: preserves the verified defect on an exposed route
- route out under `codify-scope` to a roadmap row: the same class as `T013`–`T017` was absorbed here for; a reproduced P1 413 sitting on the branch's merge unit while its boundary is open departs precedent without an authority to depart **to**
Risk shape: unchanged — one kwarg on one construction site restoring base behavior; no new mechanism; witness tier already in the tree
Deferred additive work: none — the unlimited-body posture **is** base's; if the operator ever wants a resource cap it is a new product surface, not debt this finding names
Task: T018 step 1 (wave anchor)

### DC-CR3-F-02-r1
Finding: CR3-F-02 (P1 — `count_tokens` derives the session id from the un-normalized body; attached: `test_routes_integration.py:254-333` seeds a literal `"test_session"` no digest can match and the inflation witness rides forced KV pressure)
Evidence: `iterations/079-codex-review.md` §CR3-F-02 (re-verified: `count_tokens.py:78-81` raw-body digest vs `messages.py:33-37` normalize-first; digest over `system` only (`session.py:93-98`); nonce per-request (`normalize.py:11-15`); `avg_latency 0.0` ⇒ `per_session` false forever (`compaction.py:85-86`); both test claims read in place; base origin confirmed on both sides — legacy `proxy_messages` normalizes before sid (`legacy:640-655`), legacy `count_tokens` does not)
Evidence fingerprint: unavailable (dispatcher-relayed text persisted at Checkpoint 185; mechanism re-derived from source by this pass)
Repository state: git-commit:4008e27b5394d9bd994b0131bf47a94dff061d92
Selected closure: `handle_count_tokens` normalizes before deriving — `body, _ = normalize(body, session_id="")` then `sid = session_id(body)`, mirroring the accepted sibling exactly (the empty `session_id` keeps the call image-inert per `normalize.py:202-210`); witnesses: (a) nonce-carrying digest equality across the two routes, (b) the session-arm inflation fires with correct-digest stats and KV **low**, (c) the `"test_session"` seeds become real digests without weakening any surviving assertion
Closure basis: `EI:buster_ripper.session/session_id/derived_after_normalize` — carried by this session as a **preserve** protected invariant (`plan-summary.md` §Invariant Reconciliation, inside the user-approved plan; `handoff-context.md` §Notes protected-for-yield) and deliberately never a registry-active row (`invariants/index.md` §Source Sessions: "the registry never held those handles"); the code's own accepted comment at `messages.py:33-35` records the ordering as load-bearing; reach **reached** on the deployed client (the nonce is per-request by construction); harm: a product feature — the compaction nudge that exists because "compaction never fires" (`count_tokens.py:65-76`) — runs on one of its two triggers, silently
Rejected candidates:
- accept as base parity: the defect is inherited, but the branch **formalized the rule** and the user-approved plan carries `preserve`; parity describes the origin, not a claim on the repair, and authority order ranks active carried invariants above parity of a defect the branch itself flagged load-bearing
- re-key the messages side (record stats under the raw digest): edits the rule to fit the violating site — inverts the invariant's direction
- pass the sid across routes (app state/headers): invents a cross-request coupling no authority names
- roadmap deferral: a reached P1 bound to a protected invariant gates the same product path the dump cluster's stats witnesses rely on (`test_routes_integration.py` is `VP-T003-passthrough-regression`'s vehicle)
Risk shape: unchanged — three lines at one route handler conforming it to an existing rule; response semantics unchanged
Deferred additive work: whether `EI:.../derived_after_normalize` earns promotion to a registry-active row is **Interview's** at the close (this pass moves zero registry bytes; classification **Aligned**, binding the task to the existing handle — no duplicate candidate)
Task: T019 step 1

### DC-CR3-F-03-r1
Finding: CR3-F-03 (P2 — shared `ClientTimeout(total=600)` governs the streaming relay with no per-request override; base ran `timeout=None` there; a >600 s stream truncates silently inside the committed-200 window)
Evidence: `iterations/079-codex-review.md` §CR3-F-03 + this pass's tension note (re-verified: `server.py:66-68`; `messages.py:136-141` passes no `timeout=`; `legacy:664` `httpx.AsyncClient(timeout=None)`; the arm docstring `messages.py:180-186` already names the cumulative-total bare-`TimeoutError` as an arm-covered class — the handling, not the policy)
Evidence fingerprint: unavailable (dispatcher-relayed text persisted at Checkpoint 185; re-derived from source and installed-aiohttp docs by this pass)
Repository state: git-commit:4008e27b5394d9bd994b0131bf47a94dff061d92
Selected closure: per-request `timeout=aiohttp.ClientTimeout(total=None)` on `_stream_single_request`'s `session.post` — replaces the session default for that call only (`CL:aiohttp/per-request-timeout-override`, HOLDS, registry claim row); shared `server.py:67` byte-unchanged; `_relay_after_commit` docstring clause corrected only where the new configuration would make it false; arm logic and `POST_COMMIT_FAULTS` witnesses must-not-touch
Closure basis: base parity at the named line (`legacy:664`: the stream path had **no** deadline; the 600 s is migration-introduced); reach **latent** — long generations are within the documented client's normal operation (serve script defaults `PRESERVED_THINKING=1`), and the gate's latent rule (fix or pin the guard) resolves here to **fix** because the guard **is** the defect: the deadline manufactures the very failure whose silent truncation the cleared arm then handles correctly; the verified-sound tension: item 2 certifies *how the arm handles a post-commit failure and its class accounting*; this finding indicts *which failures the policy creates* — two different propositions, both true, no adjudication between them needed; repairing the policy leaves item 2's clearance exactly as certified (streams can still die for real reasons and the arm still covers them)
Rejected candidates:
- drop/raise the shared session's 600 s: reopens `T006`'s accepted connector+session construction (a VP-T006 site) and silently changes the **non-stream** arm, which no finding names — scope broadening where a one-kwarg site-scoped fix exists
- larger constant (600→6000): preserves the truncation class for any sufficiently long stream — defect-preserving
- new config knob for the stream deadline: no requirement names a dial; parity is the requirement
- accepted-risk deferral: silent truncation of proxied answers on the route this session's every dump gate drives, under the deployed launcher's own defaults
Risk shape: unchanged — one kwarg at the named site; must-not-touch set recorded in the task
Deferred additive work: none — observation for the close, no finding in this batch: the **non-stream** messages arm inherits the shared 600 s while base ran 300 (`legacy:700`/`:843`); a widening (it truncates nothing base completed), stated here so Yield Red does not read silence as oversight
Task: T018 step 2

### DC-CR3-F-04-r1
Finding: CR3-F-04 (P2 — strict `content.decode()` inside an arm catching only `aiohttp.ClientError` → proxy 500 for a non-UTF-8 upstream error body, while `chat.py:164` already raw-relays; plus `request.text()` raising a bare `LookupError` on unknown `charset=`, uncaught at all three client parse sites)
Evidence: `iterations/079-codex-review.md` §CR3-F-04 (re-verified: `messages.py:276` inside the `:264-297` try with the single `ClientError` arm; neither error type is a `ClientError` — disjoint `ValueError` subtrees; `chat.py:155-167` as relayed; `web_request.py:686-692` — `text()` decodes with the Content-Type charset and `json()` goes through `text()`, so unknown codecs raise `LookupError` before any JSON machinery sees the body; client parse sites `messages.py:29`, `chat.py:30`, `count_tokens.py:19` catch everything except `LookupError`)
Evidence fingerprint: unavailable (dispatcher-relayed text persisted at Checkpoint 185; mechanisms re-derived from installed aiohttp source by this pass)
Repository state: git-commit:4008e27b5394d9bd994b0131bf47a94dff061d92
Selected closure: make every body path total — (1) `messages.py:276` probe decodes `errors="replace"` (relay at `:292-296` already carries raw bytes; the ASCII overflow text survives replacement); (2) the stream error arm (`messages.py:143-149`) wraps `resp.text()` in `except (UnicodeDecodeError, LookupError)` with a `read()` + `errors="replace"` fallback — observable shape unchanged for every body that decodes today; (3) the three client parses add `LookupError` to their existing decode catches and answer their existing 400 shape; fail-first witnesses for all three classes (undecodable upstream error body, bogus-charset upstream answer, bogus-charset client POST ×3)
Closure basis: the branch's own **accepted** shape — `DC-CR2-F-2-r1`'s raw-relay fall-through, landed as `T015` and re-verified at `chat.py:155-167`, which makes `messages.py`'s arms the last outliers of a defect class this session already closed at the sibling boundary; correctness ranks `errors="replace"` on the probe above a skip-the-probe branch (a genuine overflow error whose body carries stray invalid bytes still parses — the alternative knowingly misses it); reach **latent** both halves (network-supplied bodies on exposed routes; deployed vLLM answers UTF-8; no guard prevents either class)
Rejected candidates:
- answer 400/415 for undecodable **upstream** answers: the upstream's answer is not the client's error — `DC-CR2-F-2-r1` rejected this exact class on this reasoning, and it discards upstream status a proxy exists to relay
- broaden the arms to `except Exception`: masks non-decode failures as relays — defect-adjacent widening
- switch the stream error arm wholesale to raw-bytes relay without the forced `application/json` label: correct on the merits but a behavior change beyond defect elimination, and that arm's relabeling is the §RM-005/relay-fidelity lane's — no ride-along
- keep strict decode + document the 500: preserves the verified defect
Risk shape: unchanged — arms strictly become total where they 500'd; no currently-working input changes observable behavior
Deferred additive work: none (the two in-range F401/F841 lint hits T019 may consume are touch-set pickups, not debt)
Task: T019 step 2

### DC-CR3-F-05-r1
Finding: CR3-F-05 (P2 — `tests/test_compaction.py:348`/`:369` assert inline-computed arithmetic and never call `poll_kv_cache`; the coverage report's single missing `compaction.py` line is the EMA-update branch both tests claim to verify)
Evidence: `iterations/079-codex-review.md` §CR3-F-05 (both bodies re-read: `result1 = (1 - alpha) * prev + alpha * 0.75` … `assert result == raw`; `compaction.py:44-52` re-read: `:49` begins the `else`-branch assignment; the branch **does** execute in the shipped process — monitor spawned `server.py:15-34`, 10 s polls — so the missing line is a witness gap on live code, not a dead path)
Evidence fingerprint: unavailable (dispatcher-relayed text persisted at Checkpoint 185; re-read in source by this pass)
Repository state: git-commit:4008e27b5394d9bd994b0131bf47a94dff061d92
Selected closure: rewrite both tests to drive `compaction.poll_kv_cache` across ≥2 mocked `/metrics` ticks, asserting first-raw then the EMA composition through production, globals restored; done-when pins the coverage delta (no missing line at `:49`) and a mutation replay (swap the weights in `:49-52` → the rewritten witness must go red)
Closure basis: the Reach Gate **does not apply** — its own exclusion list covers false evidence, which holds for every input; the hermetic suite's 95.75% is the figure this boundary certifies (077/078 ran it on these bytes), and `codify-test-cost` requires a test to prove something — these two prove their own arithmetic; MS-003's gate (T004) is this session's own coverage commitment, so the falsified witness sits on the session's certified surface
Rejected candidates:
- delete the two tests: removes the coverage **claim** while leaving the branch unwitnessed — weakening instead of fixing the cause
- keep + doc line: preserves false evidence on a certified figure
- assert `_KV_EMA_ALPHA`'s value or recompute the formula "against a second copy": same self-proving defect in new clothes
Risk shape: unchanged — tests file only; zero production bytes
Deferred additive work: none
Task: T020 step 1

## Deterministic Closure Ledger — 2026-10-06 routing (CR3 wave, run-40 atomic non-holding rows)

Two `absorbed` atomic findings routed by this pass (`claim-399`, `claim-409` — the ledger keys are the claim rows, as
`DC-claim-237-r1` did; both rows were also published as descriptive `CL:` evidence by the run-40 promotion pass, which
records what the sources established and does not pre-decide a disposition). The batch's other two rows (`claim-391`,
`claim-404`) are `accepted` under Reach First — no ledger row by rule; each carries its verdict and citation in
§Accepted risks. Nothing here re-derives a verdict, re-checks a source, or recomputes a fingerprint: the propositions
were checked blind at run 40 and this pass read those artifacts plus the production sites they bear on. Both
dispositions amend blueprints the 080 pass wrote on tasks that are already pending — no new task, no new wave, the
wave's anchor/order/retry owners are 080's unchanged. Repository state for both rows is the same pre-write `HEAD`,
captured by Git from the supplied worktree root before the first routing write and never recomputed.

### DC-claim-399-r1
Finding: claim-399 — the premise that an unknown charset in the **response's** `Content-Type` raises `LookupError` out of `ClientResponse.text()`, which `CR3-F-04`'s blueprint carried symmetrically with the request side (published as `CL:aiohttp/client-response-text-unknown-charset-error`, `DOES-NOT-HOLD`)
Evidence: `assumptions/verdicts/claim-399.md` (+ `assumptions/verdicts/claim-401.md`, read against it; blueprint text at §Planned Task Context → `T019`)
Evidence fingerprint: sha256:7844440eb67433c6454edd708c776e47e3fcfec9b8d0aeb980dcfe88c1986ac9 (`client_reqrep.py`, full file bytes) — 401's own: sha256:2aaacea7a01e581e5eead2ab1b9e4ba3b65bc2ae3d0f176b2a78ae5b6b76bb4e (`web_request.py`) + sha256:435df4ecf0849d6cf844ff1cad4c3c764e7c734605d848b7272c2429f471cf91 (`helpers.py`)
Repository state: git-commit:e204b18e4f82f618e03c808743361637e3b149d2
Selected closure: amend `T019` step 2(b) in place — keep the approved `except (UnicodeDecodeError, LookupError)` catch list and the `read()` + `errors="replace"` fallback, re-aim the response-side fail-first at the class that actually fires there (`UnicodeDecodeError` from the fallback decode on a body undecodable as UTF-8), strike the falsified `charset=bogus`-raises-`LookupError` counterexample alternative, and add one assertion that `charset=bogus` over a UTF-8-decodable upstream error body is a **no-raise relay**, so the direction asymmetry is pinned by a witness rather than by prose
Closure basis: `DC-CR3-F-04-r1`'s required outcome is untouched and still uniquely ranked — "no body can 500 the proxy through a decode failure" — and this row changes only which class carries it at which boundary; `CL:aiohttp/request-json-unknown-charset-error` (401, HOLDS) keeps step 2(c)'s `LookupError` clauses required on the request side, `CL:aiohttp/client-response-read-charset-application` (400, HOLDS) keeps `read()` the escape hatch step 2(b) falls back to; `G-003` (proxy behavior on `/v1/messages` unchanged) ranks a strictly-totalled arm above any new raising behavior; `T019`'s own done-when requires each witness be red at pre-fix bytes, which a never-raises input cannot satisfy
Rejected candidates:
 - drop `LookupError` from the response-side arm: edits a user-approved catch list for a clause that is harmless and defensive — 399 names the caller's explicit `encoding=` argument as a live `LookupError` source — and no authority asks the two directions to differ in *shape*, only in behavior
 - keep counterexample (2) as written: defect-preserving in the lane that matters — a witness green before the fix makes the plan's fail-first criterion unfalsifiable at that site
 - make the response side validate the header charset and raise: a behavior change in the opposite direction from F-04's tolerance, named by no finding, and it 5xx/4xx's bodies base relayed
 - new task or second wave for the amendment: the repair is blueprint text on pending `T019` inside the wave 080 already scheduled (`run-codify` → remediation waves — fewest safe waves; a member's own counter and the one local Yield stand)
Risk shape: unchanged — zero production bytes added by this routing; `T019`'s touch-set, must-not-touch set and the `messages.py:143-149` site are exactly as 080 recorded them; the arms still only become total where they 500'd
Deferred additive work: none in the code. Registry-side: `S18`'s `CO:` promotion is blocked by `QB:assumption-registry/annotated-promotion-headers` (owner: the checker/verdict lane; trigger: the next dispatch that legitimately touches those six verdict artifacts) — recorded so the composition pass does not re-derive the eligibility question
Task: T019 step 2 (CR3 wave, member 2)

### DC-claim-409-r1
Finding: claim-409 — the premise that a plain `unittest.mock.MagicMock` supplies no awaitable `__aenter__`, against which `T020`'s EMA-witness rewrite was drafted (published as `CL:python/magicmock-async-context-manager`, `DOES-NOT-HOLD`)
Evidence: `assumptions/verdicts/claim-409.md` (+ `assumptions/verdicts/claim-408.md`, `claim-410.md`, `claim-411.md`, `claim-126.md`; sites `buster_ripper/compaction.py:34-57`, `tests/test_compaction.py:182-195,229-245,348,369`)
Evidence fingerprint: sha256:2576119e5653ca21ab32dd5a1a5f5604deb2b1c7a240e315710ea927bc193749 (CPython `v3.11.10` `Lib/unittest/mock.py`) + sha256:667d9ed88fd330a3d45419436b13ac41488fca77f39959dca6e93642cf80a678 (`v3.14.0`, confirmation read only)
Repository state: git-commit:e204b18e4f82f618e03c808743361637e3b149d2
Selected closure: amend `T020`'s done-when so the witness's red-ness can only come from production's arithmetic — per-tick value assertions on `compaction._kv_cache_usage` (tick 1 = raw1, tick 2 = `(1-α)·raw1 + α·raw2` from an independently written literal expectation, tick 3 the three-distinct-raws bounce claim), one awaited-channel precondition (`text()` awaited once per tick) so a wrong double reads as a fixture failure rather than a production pass, an explicit forbid list of the double shapes that fail silently or loudly for the wrong reason (bare `Mock()`/`NonCallableMock()`; any `spec=`/`autospec` response double whose spec prunes the dunder; an unconfigured `MagicMock` whose auto-child `__aenter__` yields an object carrying no `status`), the mutation replay required to go red **on a value assertion with the failing assertion named**, and the sibling idiom at `tests/test_compaction.py:182-195`/`:229-245` named as the permitted construction
Closure basis: `DC-CR3-F-05-r1`'s required outcome stands — the two tests must verify the production EMA path and `compaction.py:49` must stop being a missing line — and 409's falsification is what makes that outcome reachable: with async magic auto-provided, `async with` success proves nothing, so only a value assertion distinguishes "the branch ran" from "the branch was skipped", and `compaction.py:55-56`'s `except Exception: pass` removes every other signal; `codify-test-cost` (a test must prove something) and `DC-CR3-F-05-r1`'s own rejected-candidate list (no self-proving recomputation, no deletion, no doc line) rank this closure; 411 (`CL:pytest-asyncio/auto-mode-async-test-collection`, HOLDS) licenses `async def` witnesses with no marker and 410 (`CL:python/asyncio-sleep-yields-to-loop`) plus 126 (`CL:python/cancelled-child-await-propagation`, HOLDS) license the two-tick drive without a real `KV_POLL_INTERVAL` sleep
Rejected candidates:
 - assert call counts / "two ticks completed without raising": defect-preserving — green under exactly the inert-double shape 409 makes possible
 - configure the double and stop there, on the original premise that `MagicMock` needs explicit async setup: keeps a falsified assumption as the design reason and hides the auto-child trap the premise was meant to avoid
 - drive a real `aiohttp` server for the metrics peer: scope growth on a tests-only member (`T020` is file-disjoint, zero production bytes) with no authority asking for it, and it would move the witness off the branch it exists to cover
 - delete the two tests, or keep them plus a doc line: both rejected verbatim in `DC-CR3-F-05-r1` — removes the coverage claim while leaving `:49` unwitnessed
Risk shape: unchanged — `tests/test_compaction.py` only, zero production bytes, the same file-disjoint member
Deferred additive work: none
Task: T020 step 1 (CR3 wave, member 3)

## Deterministic Closure Ledger — 2026-10-06 routing (CR3 wave, run-42 compositions `S16`–`S19`)

Eight `absorbed` findings routed by this composition pass (numbering and one-line reasoning:
[082-assumption-decide](iterations/082-assumption-decide.md)). The batch's other six items are five `accepted`
(§Accepted risks, each with its Reach verdict and citation) and one `malformed` **re-extraction** (`S18-MC-2` →
the collector lane at `QB:assumption-ledger/s18-s19-unrowed-class-atoms`; a record gap, not a task). Repository
state for all eight is the same pre-write `HEAD` (`5f7f9b9d…`), captured through Git from the supplied worktree
root before the first routing write and never recomputed. **No claim was re-checked and no fingerprint
recomputed**: the four composition records carry their own basis digests, and this pass read those artifacts, the
member verdicts they cite, and the production sites (`server.py`, `routes/messages.py`, `routes/chat.py`,
`routes/passthrough.py`, `utils.py`, `compaction.py`, `buster_ripper_legacy.py`). Six of the eight findings are
absorbed with **zero** blueprint change — the repair 080 already routed covers them — and two land as *criteria*
amendments on pending members: `T018`'s done-when gains the all-components-`None` pin, `T020`'s done-when (6) is
corrected where `S19-CE-3` shows its stated mechanism unsatisfiable. No task was created (`plan-adhoc-task` not
invoked), the wave keeps 080's shape exactly (anchor `T018`, members `T018`→`T019`→`T020`, one local Yield,
per-member retry counters at 0), and **zero `question`** was emitted — `FK-011` stays the only open Fork.

### DC-S16-a-r1
Finding: `S16`-(a) — missing contract: what the configured `client_max_size` number *counts* (`>=` at `web_request.py:677` makes it the smallest rejected size, so the largest accepted body is `N−1`) and *where* the cap measures (the gate sits at the handler's first read, ahead of `normalize()` at `messages.py:36` and `strip_and_cache_images()` at `:40-41`, so under `--strip-images` an image-bearing body over 1 MiB is 413'd although the forwarded body would be far smaller). No counterexample; the verdict is `UNCLEAR`.
Evidence: `assumptions/compositions/S16-body-limit.md` §What Is Required (a), §Dimension Coverage 1–3, §Reach citations
Evidence fingerprint: sha256:46fc6a70789eb775cb88bb79b91507479440b1f15b63b702a8bc48be25b7ccb9
Repository state: git-commit:5f7f9b9d43a50f571c497c0c52bf18406e77a567
Selected closure: **none of this pass** — the question is deleted, not answered, by the already-routed `T018` step 1: `create_app()` passes `client_max_size=0`, and `CL:aiohttp/client-max-size-falsy-skip` (row 393, HOLDS) skips the comparison entirely on a falsy value at all four read sites. There is then no configured number whose boundary semantics or measurement point a client can observe, and no `>=`-off-by-one to rule on. Absorbed with no amendment: `T018`'s existing done-when (2 MiB POST to each of the four route classes, red at pre-fix bytes with the 413, green after) *is* the proof, and it covers the `--strip-images` instance because a 2 MiB image-bearing body is one of the four.
Closure basis: `DC-CR3-F-01-r1` (base parity — base imposed no cap; the 1 MiB default is migration-introduced; reach **reached** under the proxy's own contract, absorbed rather than routed out under the 057/CR2 one-merge-unit precedent) and row 393 for the mechanism. The brief's "decide intended behavior against the base-parity authority" resolves to *no intended cap at all*, so both readings of (a) are inadmissible as work: they ask which side of a line the proxy draws when the settled answer is that it draws none.
Rejected candidates:
 - rule "`N` = largest accepted" by setting `client_max_size=N+1`: invents a number no authority names, on a gate parity says is disabled
 - move the cap after `normalize()`/`strip_and_cache_images()` (size the forwarded body): new mechanism on the request path, and it preserves a rejection base never imposed
 - keep the default and document `>=` semantics: preserves the reproduced P1 413 (rejected verbatim in `DC-CR3-F-01-r1`)
 - open a task for (a): the repair already exists in the wave as its step 1 — 081's precedent for absorbed-with-zero-change
Risk shape: unchanged — this routing moves zero bytes; `T018`'s touch-set is one kwarg + comment + witnesses
Deferred additive work: none in code. If the operator ever wants a resource cap it is a new product surface (`DC-CR3-F-01-r1` says so already); the *decompression* half of that decision is now §RM-011, not a re-open of this row
Task: T018 step 1 (wave anchor)

### DC-S17-CE-1-r1
Finding: `S17-CE-1` — the one deadline component the production session arms (`total`, `server.py:67`) raises exactly the class the streaming call site's only arm cannot catch (`messages.py:165` catches `aiohttp.ClientError`; 330's exhaustive client-timeout set is `ServerTimeoutError`/`ConnectionTimeoutError`/`SocketTimeoutError`, all reached from the socket components; 331: the cumulative `total` raises bare `asyncio.TimeoutError`, passed through unwrapped at `client.py:793-798`), and which arm sees the failure is decided by whether the fire lands before or after the commit at `messages.py:161`, which the request does not control
Evidence: `assumptions/compositions/S17-stream-deadline.md` §Counterexample CE-1, §Consumer, §Dimension Coverage 5–6; read against `verdicts/claim-57.md`, `claim-396.md`, `claim-330.md`, `claim-331.md`
Evidence fingerprint: sha256:93391795dc48184cf665303b6e52d7548bdb6a8dedf799188764b7bad3138712
Repository state: git-commit:5f7f9b9d43a50f571c497c0c52bf18406e77a567
Selected closure: **none of this pass beyond one criteria clause** — `T018` step 2's per-request `timeout=aiohttp.ClientTimeout(total=None)` removes the *event*, not the handler. Per 57 the request-local object replaces the session default wholesale (it does not merge), and per 396 a `None` component arms nothing (`helpers.py:609-625`), so all four components are `None`, no `TimeoutHandle` is constructed for that call, and `client.py:622-624`'s `with timer:` around the header wait is a no-op. The escape CE-1 describes then has no producer on this path, and neither does the post-commit truncation (which is documented, cleared behavior — `N5`). Added to `T018`'s done-when: a witness pinning that the request-local object carries **every** component `None`, because that pinning is what CE-1's deletion rests on and a spelling that re-inherits `total` (or arms `sock_read` "to be safe") would leave CE-1 live while still passing the existing truncation witness.
Closure basis: base parity at the named line (`buster_ripper_legacy.py:664`, `httpx.AsyncClient(timeout=None)` — the stream path had **no** deadline) + `DC-CR3-F-03-r1`, which already ranks this exact closure and rejects keeping any armed budget here (the guard *is* the defect); rows 57/396 for the mechanism. Severity corrected on the way in: what CE-1's escape actually reaches the client with is aiohttp's **generated 504** (`verdicts/claim-230.md` records `web_protocol.py:518 except asyncio.TimeoutError` → `handle_error(request, 504)` → sent with `force_close()`), not "no response" — a wrong-status/wrong-envelope error turn, which is why the arm-level restyling is not what parity owes.
Rejected candidates:
 - widen the pre-commit arm to `except (aiohttp.ClientError, asyncio.TimeoutError)`: dead code the moment step 2 lands (nothing on that path raises the class), it *preserves* the armed deadline parity says should not exist, and applied family-wide it breaches `chat.py` at 300/300 — restyling a symptom instead of removing the event
 - keep `total=600` and pin the escape with a test: the gate's latent rule resolves to **fix** here because the guard is the defect (`DC-CR3-F-03-r1`'s ruling, unchanged)
 - larger/other constant: preserves the truncation class for a long enough stream — defect-preserving
 - separate owning task for the arm family: the stream path needs none (this row closes it); the remaining armed sites are out of this session's scope and block no done-when → §RM-010
Risk shape: unchanged — one kwarg plus one witness assertion on a site `T018` already owns; the must-not-touch set (`_relay_after_commit`'s logic, `POST_COMMIT_FAULTS`, `server.py:65-68`) is untouched by this routing
Deferred additive work: §RM-010 (the three still-armed sibling arms + the generated-error envelope, owner + trigger named there); `QB:assumption-ledger/row-57-version-pin` (57 is pinned to `v3.12.15` bytes — `N1` — while the other four members and the installed tree are 3.13.3; the post-`T018` recomposition cannot certify the replace-not-merge rule on this path until that row is re-pinned)
Task: T018 step 2 (wave anchor)

### DC-S17-CE-2-r1
Finding: `S17-CE-2` — one configured number (`total=600`), one call site, one arm, **two** semantic kinds: with an `https://` upstream, a deadline that expires during the TLS handshake is re-raised at `client.py:737-740` as `ConnectionTimeoutError`, which **is** a `ClientError` and gets the site's 502, while the same number expiring during the header wait is bare `asyncio.TimeoutError` and is not; the arm tests the class and thereby discards the phase that produced it — while `messages.py:180-185`'s own docstring states that split as a premise of the neighbouring function
Evidence: `assumptions/compositions/S17-stream-deadline.md` §Counterexample CE-2, §Dimension Coverage 5; `verdicts/claim-396.md` (connector forwards `timeout.total` as `ssl_handshake_timeout`), `claim-331.md`, `claim-330.md`
Evidence fingerprint: sha256:93391795dc48184cf665303b6e52d7548bdb6a8dedf799188764b7bad3138712
Repository state: git-commit:5f7f9b9d43a50f571c497c0c52bf18406e77a567
Selected closure: same landing, same one clause. `total=None` disarms **both** kinds at once, because the handshake-scoped timeout is fed by the same component (row 396) and the request-local object is the whole policy (row 57) — so the two-cell split CE-2 names cannot recur on this call. `T018` already anticipates the docstring half: its required behavior 3 permits exactly one edit inside `_relay_after_commit` — the clause citing the cumulative `ClientTimeout.total` budget as an in-path failure source — "corrected to match the new configuration without asserting anything false about the class hierarchy", which is what CE-2's premise-clause needs. No further change.
Closure basis: `DC-CR3-F-03-r1` + base parity (`legacy:664`) + rows 396/57 for the mechanism; and `DC-S17-CE-1-r1` for the shared cause — CE-2 is the same defect observed across the component/phase axis, so one closure covers both without collapsing them into one ledger row.
Rejected candidates:
 - classify by phase at the arm (inspect the exception, or set `sock_connect`/`connect` so every timeout is a `ClientError`): arms *more* deadlines on a path base armed none of, to make a class test pass — inverts the parity closure
 - accept the split and pin it in prose only: `T019`'s own amendment precedent (claim-399) requires an asymmetry of this boundary to be pinned by a witness, not by wording
 - fold CE-2 into CE-1's row: the seam's dimension cells name them separately (component-vs-component and phase-vs-phase), and a later recomposition must be able to see which one moved
Risk shape: unchanged — same one kwarg, same one witness assertion, no new arm
Deferred additive work: none beyond `DC-S17-CE-1-r1`'s two pointers
Task: T018 step 2 (wave anchor)

### DC-S18-CE-1-r1
Finding: `S18-CE-1` — a client that names a codec Python does not have (`Content-Type: application/json; charset=x-not-a-real-codec`) escapes the 400 answer its body gets under either neighbouring spelling of "unparseable body": `web_request.py:686-690` decodes with the peer-named charset before anything parses, `codecs.lookup` raises `LookupError` (405 + 401, HOLDS), and all three production parse arms name only `json.JSONDecodeError` and `UnicodeDecodeError` — three spellings, two settlements (`messages.py:28-31`, `chat.py:29-32`, `count_tokens.py:18-25`)
Evidence: `assumptions/compositions/S18-decode-failure-surface.md` §Counterexample CE-1 (+ its three-row table), §Dimension Coverage 2/3/6; `verdicts/claim-401.md`, `claim-405.md`, `claim-406.md`, `claim-407.md`, `claim-51.md`
Evidence fingerprint: sha256:4551e202dac7425f7d5d36581b973d18f62308c19747288629150f34f48119a9
Repository state: git-commit:5f7f9b9d43a50f571c497c0c52bf18406e77a567
Selected closure: **already routed and complete — zero amendment.** `T019` step 2(c) is verbatim "the three client parses (`messages.py:30`, `chat.py:31`, `count_tokens.py:22-25`) add `LookupError` to their existing decode catches and answer each site's existing 400 shape", with counterexample (3) "client POSTs `charset=bogus` to each of the three POST routes → 400 'Invalid JSON', never 500" and the fail-first criterion "each red at base bytes". Site set, class, and settlement all match CE-1's three sites; and 081's amendment expressly leaves this step alone ("steps 2(a)/2(c) unchanged (401 HOLDS for the request side)"). Yield replays counterexample (3) on the landed bytes and reads the pre-fix class as a bare `LookupError` — that is the required proof, already in the blueprint.
Closure basis: `DC-CR3-F-04-r1` (the branch's own accepted raw-relay/tolerance shape makes these arms the outliers of a class this session closed at the sibling boundary) + rows 401/405/406/407 for the class facts + `G-003` ranking a strictly-totalled arm above any new raising behavior. Reach: **reached** — an exposed surface taking the client's own header verbatim, no upstream validation can withhold it.
Rejected candidates:
 - new task or wave member for CE-1: the repair is already a pending step of a pending task in the scheduled wave — 081's precedent (a finding landing on the routed step with zero criteria change is still `absorbed`, not `accepted`)
 - answer 415 Unsupported-Media-Type for an unresolvable charset: a new client-visible status no authority names, and it rejects bodies base served after decode
 - switch the three sites to `read()` + local parse (`passthrough.py:14`'s precedent, dimension 5): forecloses CE-1 but discards the peer's charset for *valid* spellings too — a behavior change beyond defect elimination
 - broaden the three arms to `except Exception`: rejected verbatim in `DC-CR3-F-04-r1` (masks non-decode failures as relays)
Risk shape: unchanged — this row adds no bytes to `T019`'s touch-set
Deferred additive work: `S18-MC-2`'s unrowed atom is the one thing that can make step 2(c)'s *totality* claim incomplete; routed as `malformed` to the collector (`QB:assumption-ledger/s18-s19-unrowed-class-atoms`), explicitly not implemented by Forward on an unverified class
Task: T019 step 2(c) (CR3 wave, member 2)

### DC-S18-MC-1-r1
Finding: `S18-MC-1` — missing contract on the response direction: `_stream_single_request` answers an upstream ≥ 400 with `error_content = await resp.text()` (`messages.py:143-144`), strict by 398 + 402, so a known-codec error body with invalid bytes raises `UnicodeDecodeError`; the only enclosing arm is `except aiohttp.ClientError` at `:165`, and no `S18` member says anything about `ClientError`'s hierarchy, so "settles inside the site or escapes it" was undecidable *from the seam's member set*
Evidence: `assumptions/compositions/S18-decode-failure-surface.md` §MC-1 (+ §Dimension Coverage 5); answered from this session's own settled evidence: `DC-claim-399-r1`'s counterexample text and `T019` step 2(b); contrast site `compaction.py:41`/`:55`
Evidence fingerprint: sha256:4551e202dac7425f7d5d36581b973d18f62308c19747288629150f34f48119a9
Repository state: git-commit:5f7f9b9d43a50f571c497c0c52bf18406e77a567
Selected closure: **already routed and complete — zero amendment.** The fate is not open at this boundary: `DC-claim-399-r1` states it as its counterexample — "`500` with `b"\xff\xfe\x00garbage"` → `UnicodeDecodeError` (not an `aiohttp.ClientError`) → escapes `_stream_single_request`'s single arm → proxy 500 with a traceback" — and `T019` step 2(b) repairs exactly that site with the approved `except (UnicodeDecodeError, LookupError)` + `read()`/`errors="replace"` fallback, with the response-side fail-first re-aimed (by 081) at the class that actually fires. So MC-1 is a gap in `S18`'s *member set*, not in the session's evidence. Absorbed into the step that already exists; the missing atom is handed to the collector so a later composition can certify the seam without inference (same QB row as MC-2).
Closure basis: `DC-CR3-F-04-r1`'s required outcome ("no body can 500 the proxy through a decode failure") + `DC-claim-399-r1` for the class and the fate + rows 400 (`CL:aiohttp/client-response-read-charset-application`, HOLDS) for the fallback the arm lands on. Reach: **reached** — any upstream answering ≥ 400 with a body carrying bytes invalid under the fallback charset, on the exposed `/v1/messages` stream path.
Rejected candidates:
 - treat MC-1 as `malformed`/open: the closure is settled by a routed ledger row of this same wave; declaring it undecided would re-litigate 081's disposition
 - row the missing atom here and treat it as evidence: writing a verdict is not this lane (E1), and the proposition is a third-party class fact — collector/checker territory
 - rely on `compaction.py:55`'s `except Exception` as the precedent shape for `:165`: rejected verbatim in `DC-CR3-F-04-r1` (`except Exception` masks non-decode failures)
Risk shape: unchanged — no bytes, no new criteria
Deferred additive work: the `ClientError`-base/disjointness atom at `QB:assumption-ledger/s18-s19-unrowed-class-atoms` (certification-only: `T019` step 2(b) does not wait on it)
Task: T019 step 2(b) (CR3 wave, member 2)

### DC-S19-CE-1-r1
Finding: `S19-CE-1` — `async with` succeeds and the payload is discarded: with the sibling shape minus its two `__aenter__`/`__aexit__` self-binding lines, the name bound by the `as` clause is `__aenter__.return_value` — an auto-child carrying none of the configured attributes (408 + 415, both HOLDS) — so `compaction.py:40`'s `resp.status == 200` does not match (or raises inside the body and is consumed at `:55-56`), `:44-52` never runs, `_kv_cache_usage` never moves, and the file's existing assertion vocabulary (`session.get.called`) stays green
Evidence: `assumptions/compositions/S19-async-witness.md` §Counterexample CE-1, §Dimension Coverage 1/3; `verdicts/claim-408.md`, `claim-415.md`
Evidence fingerprint: sha256:990d0278e86f444d24a9f26db05d422c5e958142ed64309b456434d30fae86a3
Repository state: git-commit:5f7f9b9d43a50f571c497c0c52bf18406e77a567
Selected closure: **already routed and complete — zero amendment.** `T020`'s done-when, as amended by 081 after `claim-409`, makes this shape structurally unable to pass: item (1) requires per-tick **value** assertions on `compaction._kv_cache_usage` (tick 1 = raw1 exactly, tick 2 = the blend from an independently written literal, tick 3 the three-distinct-raws claim), item (2) requires the awaited-channel precondition (`text()` asserted once per tick), item (3) forbids the double shapes that pass on an inert binding, and item (4) names the sibling idiom — self-binding included — as the permitted construction. Under CE-1's input every one of (1) and (2) fails. The Reach Gate does not apply (false evidence, the gate's own exclusion); the branch's production reach is noted for the record: `_kv_cache_usage` is read at `count_tokens.py:30`/`:51`, so an inert witness misstates a live compaction nudge.
Closure basis: `DC-CR3-F-05-r1`'s required outcome (the two `test_ema_*` must verify the production EMA path; `:49` must stop being a missing line) + `DC-claim-409-r1`'s capability condition ("the witness must be structurally unable to pass on an inert double") + `codify-test-cost` (a test must prove something).
Rejected candidates:
 - new task / wave member: the criteria already exist on a pending member of the scheduled wave
 - assert `session.get.call_count` or "the loop ran twice without raising": rejected verbatim in `DC-claim-409-r1` — green under exactly this shape
 - drive a real `aiohttp` server as the `/metrics` peer: scope growth on a tests-only member, and it moves the witness off the branch it exists to cover
Risk shape: unchanged — `tests/test_compaction.py` only, zero production bytes
Deferred additive work: none
Task: T020 step 1 (CR3 wave, member 3)

### DC-S19-CE-2-r1
Finding: `S19-CE-2` — the two arms of one response object need different mock *kinds*: `compaction.py:40` reads `.status` without awaiting while `:41` awaits `resp.text()`, so a **sync** `Mock` on `text` (the kind the same tests already use for `session.get`, `:195`) raises `TypeError` inside the `if` body and is silenced by `except Exception: pass` at `:55-56` — the same silent observable as CE-1
Evidence: `assumptions/compositions/S19-async-witness.md` §Counterexample CE-2, §Dimension Coverage 2
Evidence fingerprint: sha256:990d0278e86f444d24a9f26db05d422c5e958142ed64309b456434d30fae86a3
Repository state: git-commit:5f7f9b9d43a50f571c497c0c52bf18406e77a567
Selected closure: **already routed and complete — zero amendment.** Done-when item (2)'s awaited-channel precondition is the discriminator CE-2 needs (a wrong-kind `text` never completes an awaited `text()` per tick), item (4) names the permitted construction with `text` configured on an `AsyncMock` response, and items (1)+(5) make any residue red on a value assertion with the failing assertion named. A wrong double therefore reads as "the peer never delivered a 200 body", which is what (2)'s clause was written to guarantee.
Closure basis: same three authorities as `DC-S19-CE-1-r1`, plus 081's own words in the `T020` block: the witness "must be structurally unable to pass on an inert double, because 'the double supports `async with`' is no evidence of anything" — CE-2 is the awaited-arm instance of that same condition.
Rejected candidates:
 - add an explicit "text must be awaitable" clause: 081's amendment already covers it functionally through (2)+(4); a fifth wording on the same list is padding, and this row records the coverage rather than re-stating it
 - rely on `pytest-asyncio` surfacing the un-awaited coroutine as a warning: no authority in the ledger licenses a warning as an assertion (411 speaks to collection only)
Risk shape: unchanged
Deferred additive work: none
Task: T020 step 1 (CR3 wave, member 3)

### DC-S19-CE-3-r1
Finding: `S19-CE-3` — the members do not grant a second tick: `first` is function-local (`compaction.py:31`), so the blend arm at `:48-52` requires a **second iteration of the same invocation**, and `:57` awaits `asyncio.sleep(config.KV_POLL_INTERVAL)` with the interval `10` read from `config.py:64`; 410's verified guarantee is a **yield**, and its own verdict limits it ("whether anything else executes depends on the loop having other ready work at that iteration"), so a task parked on a 10 s timer never reaches `:48-52` under the file's driving idiom (`session.get` called once, `call_count == 1`, the value still raw)
Evidence: `assumptions/compositions/S19-async-witness.md` §Counterexample CE-3 + its "Named missing contract", §Dimension Coverage 4; `verdicts/claim-410.md`, `claim-126.md`, `claim-411.md`; sites `compaction.py:31,45-52,57`, `config.py:64`, `tests/test_compaction.py:348,369`
Evidence fingerprint: sha256:990d0278e86f444d24a9f26db05d422c5e958142ed64309b456434d30fae86a3
Repository state: git-commit:5f7f9b9d43a50f571c497c0c52bf18406e77a567
Selected closure: **amend `T020`'s done-when item (6) in place** — this is the one finding the composition shows that the amended blueprint does **not** cover; it in fact contradicts it. The clause to be corrected reads "reach tick 2 without a real `KV_POLL_INTERVAL` sleep (410's yield, the cancel-and-await arm 126 licenses)": 410 licenses a yield, not an elapsed timer, so as written the criterion is unsatisfiable. Corrected to name the mechanism that actually makes the interval elapse inside a test: the loop reads `config.KV_POLL_INTERVAL` per iteration at `:57`, so the witness sets that module attribute to `0` for the duration of the driven task and restores it (with `compaction._kv_cache_usage` restored as already required), and it must then *observe* tick 2 rather than assume it — which done-when (1)'s per-tick value assertions and (2)'s per-tick awaited-channel precondition already enforce, so a witness that never reaches the second iteration cannot be green.
Closure basis: `DC-CR3-F-05-r1`'s required outcome (`:49` must stop being a missing line; the tests must exercise production's composition) + `codify-test-cost` (test and suite run cost stay small — a real 10 s sleep on a hermetic suite this boundary certifies at 455 tests is not admissible) + `T020`'s own must-not-touch on `compaction.py` production bytes (which rules out making the interval a parameter) + 410's self-limited verdict. Reach Gate: not applicable (false evidence); the *witnessed branch* runs in the shipped process (`server.py:15-34`, 10 s polls), which is why the value assertions are the proof.
Rejected candidates:
 - leave (6) as written: defect-preserving in the strongest sense — a done-when that cannot be met as stated is met either by an unbounded wait or by a witness that quietly asserts less
 - patch `asyncio.sleep` wholesale: broader mechanism than one module-attribute read, and it hides the production sleep the test exists to cross
 - a frozen/async-clock harness (`freezegun`-style): a new test mechanism no authority asks for, on a tests-only member
 - sleep 10 s for real: fails `codify-test-cost` and the hermetic-suite budget 080's verification profile names
 - call `poll_kv_cache` twice instead: does not reach the blend arm at all — `first` is function-local (`:31`), so every call re-seeds the raw-assign arm
 - move `:57` inside the `try` or make the interval a parameter: edits `compaction.py` production bytes, which `T020`'s touch-set forbids
Risk shape: unchanged — one criteria clause on a tests-only member; zero production bytes; the wave's shape, order and counters stand
Deferred additive work: none for the code. Certification-side: `S19` recomposes over the landed `tests/test_compaction.py` bytes (`Invalidates On` 7), and the two certification-only atoms `N1`/`N2` (does `except Exception` reach `CancelledError`; what does `unittest.mock`'s default `__eq__` do to an auto-child `.status`) ride `QB:assumption-ledger/s18-s19-unrowed-class-atoms` — neither carries a finding and both are loud, not silent, if wrong.
Task: T020 step 1 (CR3 wave, member 3)

## Deterministic Closure Ledger — 2026-10-07 routing (run 45, `S20` `web.Response` body round trip — rows 418/419/422/423)

- Basis commit for all three entries: `git-commit:9c90675f69998e0ea7847c6b456c2e3f46f8a81c` (pre-write `HEAD`).
- One remediation serves all three; `T019.001` is the task each points at. Evidence for the reach determination that
  ranks all three is `evidence/n6-reach-and-base-comparison.txt` (read + executed at this pass; the note's framing was
  checked, not inherited — §6(c)-(d) correct and extend what N6 stated).

### DC-claim-418-r1
Finding: claim-418 (`assumptions/claims.md` run 45, `State: DOES-NOT-HOLD`, `Routing: pending`) — for an aiohttp `web.Response` constructed with a `str` `body=`, the `body` getter returns a `StringPayload`, not `bytes`, so the package's only `.body` read-back (`messages.py:106`) always takes its `else` branch
Evidence: `assumptions/verdicts/claim-418.md`; origin `assumptions/compositions/S18-decode-failure-surface.md` §Notes N6; task surface `buster_ripper/routes/messages.py:106` ← `:151-155`
Evidence fingerprint: `sha256:3ca73388d5382e67ed5ea295be844cac06ce7950a5a5237e8b39c7b2259ece65` (`web_response.py`); `sha256:3ba9ec60d50b2fb01e33673227a4d85fbde77159cbe71270b7901e831c0c2aac` (`payload.py`)
Repository state: git-commit:9c90675f69998e0ea7847c6b456c2e3f46f8a81c
Selected closure: change `messages.py:106`'s read to the response's own text surface (`.text`), leaving the `:151-155` constructor and the wire shape untouched — the smallest complete remediation, one expression at the only consumer of `.body` in the package (census: evidence file §2)
Closure basis: Reach **`reached`** — cited path `POST /v1/messages {stream:true}` → `messages.py:67-68` → `handle_streaming:101-107` ← `_stream_single_request:149-155` → `utils.parse_context_overflow_error` → `None` → `:125`; the gate is `backend_type == "sglang"`, a value an operator sets (`main.py:129` `--backend-type`, or `/v1/models` auto-detect at `server.py:71`), and the Reach Gate counts configuration-supplied input as reached ("exposed surfaces are reached by definition"; "priority is not reach"). Fix-or-pin resolves to **fix**: no witness enters the loop (`tests/test_context_overflow.py` asserts only `get_backend_state()` values — evidence file §7), while `iterations/001-collect.md` §B-005 has carried "context-overflow retry … behave as tested" since this session's first pass. Ranked by `DC-CR3-F-04-r1`'s landed direction (the stream error arm's text must survive to be usable — this is the text dying one line later), by the file's own working peer at `messages.py:296-305`, and by `chat.py:178`, so the required behavior is the repo's existing shape rather than a new design. **Base parity is silent, not protective**: `buster_ripper_legacy.py` has no `overflow`/`sglang`/`retry`/`backend`/`max_completion` anywhere (evidence file §5), so nothing is owed *to* base and nothing is excused *by* it; the inertness is migration-introduced and was already present at session entry (§6(a)), and the CR3 delta did not change which branch `:106` takes (§6(b)) — only the population arriving there (§6(c)).
Rejected candidates:
- construct with `text=error_content` at `:151-155`: verified to store bytes (so the current consumer would work) but verified to emit `application/json; charset=utf-8` where it emits `application/json` today — a client-visible header change on the relay arm this session's accepted fidelity work (`T015`/`T017`, `CL:aiohttp/web-response-explicit-content-type`) governs; broader observable delta than the defect requires
- make `_error_body` return bytes: displaces `T019`'s approved contract ("whose catch list is **exactly** as approved", `plan-summary.md` `T019`) and moves a landed surface, for no correctness gain over a consumer-side read
- delete the retry loop as dead code: preserves the verified defect and contradicts the function's declared behavior — the gate's defect-preserving rejection
- accept as a recorded risk: `reached` findings are fixed or pinned; nothing pins this one and the harm is the feature's designed trigger, so it is not the 1%/0.01% category
- route to `roadmap.md` as out-of-scope: `codify-scope` rule 5 places a repair where the cause lives, and the cause is in `messages.py`, a file this session's tasks name and CR3 just changed; rule 2 does not apply
Risk shape: unchanged — one expression at one consumer inside a file already in this session's remediation lane; no client-visible byte, header or status moves; adds witnesses and no new mechanism; `chat.py` (300/300) untouched, `handle_streaming`'s metrics unmoved
Deferred additive work: `roadmap.md` §RM-001 keeps `handle_streaming`'s hard `N:3` and the file's review-band debt (`QB:routes-messages/stream-single-request-gray-pair`, `QB:routes-messages/file-size-yellow`); the misplacement check is owed by `T019.001` at its own touch, the restructure is not
Task: T019.001 steps 1-2 (consumer-side read + the three witnesses)

### DC-claim-422-r1
Finding: claim-422 (`State: DOES-NOT-HOLD`, `Routing: pending`) — the exhaustive-surface claim is false: `Response.text` **is** an accessor returning the string passed to `body=` (freshly decoded, value-equal; the object is not retained), so a strict object-identity reading would have survived but the plain reading does not
Evidence: `assumptions/verdicts/claim-422.md` (independent of `claim-421.md`, and the two agree); `assumptions/claims.md` run-45 polarity table
Evidence fingerprint: `sha256:3ca73388d5382e67ed5ea295be844cac06ce7950a5a5237e8b39c7b2259ece65` (`web_response.py`); `sha256:3ba9ec60d50b2fb01e33673227a4d85fbde77159cbe71270b7901e831c0c2aac` (`payload.py`); `sha256:435df4ecf0849d6cf844ff1cad4c3c764e7c734605d848b7272c2429f471cf91` (`helpers.py`)
Repository state: git-commit:9c90675f69998e0ea7847c6b456c2e3f46f8a81c
Selected closure: none of its own — this row is the **verified repair channel** for `DC-claim-418-r1`: the accessor enumeration names `.text` as the only surface on `Response` returning the string's content, which is what makes the selected closure a read of a certified surface rather than an invention. Coverage is written into `T019.001`'s done-when so the next composition pass can cite it
Closure basis: same ranking material as `DC-claim-418-r1`; additionally `CL:aiohttp/web-response-str-body-text-property` (HOLDS, row 421) and this row together settle both readings of the `body=`/`text` pair, so no open point is left at the boundary for the repair to speculate about. Permanent `CL:` records are evidence inputs only and authorize no behavior here
Rejected candidates:
- treat the row as an accepted consequence of a library quirk: its falsification is load-bearing for the repair's mechanism; accepting it would leave `T019.001` without a certified read surface
- re-extract a successor row for the object-identity reading: no consumer compares identity — `:106` compares content against a regex — so the reading has no reach and would spend a checker on a proposition nothing asks
Risk shape: unchanged — no independent byte moves; it rides `DC-claim-418-r1`'s one expression
Deferred additive work: none
Task: T019.001 step 1 (the read it certifies)

### DC-claim-423-r1
Finding: claim-423 (`State: DOES-NOT-HOLD`, `Routing: pending`) — `str()` of a `StringPayload` yields Python's default object representation, not the text, because `payload.py` defines neither `__str__` nor `__repr__` and the text is not retained as a string on the object
Evidence: `assumptions/verdicts/claim-423.md`; corroborated by `assumptions/verdicts/claim-424.md` (HOLDS — the `__str__`→`__repr__` fallback); observed value at `evidence/n6-reach-and-base-comparison.txt` §4
Evidence fingerprint: `sha256:3ba9ec60d50b2fb01e33673227a4d85fbde77159cbe71270b7901e831c0c2aac` (`payload.py`)
Repository state: git-commit:9c90675f69998e0ea7847c6b456c2e3f46f8a81c
Selected closure: none of its own — establishes that `:106`'s `else` branch cannot be repaired by any coercion of `result.body` (there is no string to coerce: the text exists only as encoded bytes behind `decode()`), which closes the candidate space down to the single read `DC-claim-418-r1` selects
Closure basis: same ranking material as `DC-claim-418-r1`; plus row 424 for the mechanism, and the executed repr, so the claim "the matcher compares against an object representation" rests on bytes and an output rather than on N6's prose
Rejected candidates:
- keep the coercion and add a `getattr(result.body, "decode", …)`-style probe: a new mechanism on one call site where a named accessor exists and is certified (row 421); strictly larger and re-derives what the ledger already settled
Risk shape: unchanged — evidence row for the same one expression
Deferred additive work: none
Task: T019.001 step 1 (it forecloses the alternatives)

## Deterministic Closure Ledger — 2026-10-07 routing (CR4 batch — codex review persisted at [088](iterations/088-codex-review.md))

One `absorbed`, one `accepted`, zero `question`/`breaks-it`/`malformed`. Both findings were re-established on
current bytes **by execution at three points** (base `72e3a1c6`, pre-range `6508941`, HEAD `c5e2be4`) before any
field below was written: `evidence/cr4-probe-matrix.txt`.

### DC-CR4-F-01-r1
Finding: CR4-F-01 (P2 — `T019`'s accepted normalize-before-keying repair gives `/v1/messages/count_tokens` a 500 it did not have in the range and never had in base: a non-mapping `tools`/`messages` element raises `AttributeError` out of `normalize()` through `handle_count_tokens`)
Evidence: `iterations/088-codex-review.md` §[P2]; `evidence/cr4-probe-matrix.txt` §1 (the trio 200→500 across the range, both axes, escape frames `normalize.py:75`/`:117`), §2 (base answers 200 on all three — no `normalize` call in its `count_tokens`), §4 (zero digest differences, 18 cases × 2 axes), §5 (the narrowed call reproduces base's column exactly), §6 (`:192` is masked by `:117` today and becomes live only if `:117` is hardened), §7 (hardening the loops moves `/v1/messages` from 500 to 502)
Evidence fingerprint: unavailable (the reviewer's table is persisted at 088; every clause re-established on this pass's bytes and by execution rather than inherited — `git status --porcelain` was clean apart from 088 itself, so the measured tree is the reviewed tree)
Repository state: git-commit:c5e2be44ba1a7ab984f02104472f06f849e4f773
Selected closure: narrow the `count_tokens.py:87` call to the one normalization step the invariant depends on — the system-strip step (`normalize.py:50-61`) — leaving `normalize()`'s element loops byte-unchanged and `messages.py`'s full-normalize call untouched; `sid = session_id(body)` stays where `T019` put it, immediately after the step that removes the per-request billing nonce from the text the digest hashes. Witnesses and their fail-first directions: `T019.002`'s Done when, clauses 1-7.
Closure basis: **the carried invariant, satisfied rather than regressed** — `EI:buster_ripper.session/session_id/derived_after_normalize` (`preserve`; `plan-summary.md` §Invariant Reconciliation inside the user-approved plan; `handoff-context.md` §Notes; never registry-active per `invariants/index.md` §Source Sessions) protects the digest against the nonce, `session_id()` reads `system` only (`session.py:93-98`), and of the four steps only `_normalize_system` writes `system`'s text — measured, not asserted (evidence §4). **Base parity as the ranking authority on the route** — the session's settled contract for the migration (`tests/test_header_parity.py:4`, the direction `T013`-`T017` and `T018`-`T020` both applied), against a 500 this session's own delta introduced at a site where base and the pre-range branch both answered 200; the identical shape to `DC-CR3-F-01-r1`, whose rejected-candidate list already forecloses "route out under `codify-scope` a rejection the branch introduced." **Reach: `latent` in this session's established vocabulary — `CR3-F-04` set it for exactly this class (a network-supplied payload producing a proxy 500 on an exposed route), and the gate's literal reading is the same verdict wearing a different word: under "exposed surfaces are reached by definition" a POST of `{"tools":[null]}` is a cited path from an entry point to the site, so `reached` is defensible; either label forces the same rule, fix-or-pin, and no clause of this disposition turns on the word** — and **nothing pins it**: no witness anywhere sends a request body with a non-mapping element inside `tools` or `messages` at any route — `grep -rn null tests/ --include=*.py` reaches only `tests/test_upstream_answer_relay.py:163`'s `b"null"`/`b"[1, 2]"`/`b'\"a string\"'`, which are **upstream answers** relayed at `/v1/chat/completions`, not client request bodies, and not elements of an array, so it resolves to fix. **`G-003`** ("Proxy behavior on `/v1/messages` is unchanged"; constraint column: "No new endpoints, **no normalization changes**") is what ranks the wider candidate out.
Rejected candidates:
 - harden `normalize()`'s two element loops (the reviewer's alternative): changes `/v1/messages`, which `G-003` holds still — measured 500 → 502 on the same three bodies (evidence §7) where base 500s on those same bodies (§3) — so it converts a parity-protected answer into a new product posture the plan never authorized; and it is not even the complete form of itself, because `normalize.py:192` becomes the raising site the moment `:117` stops being it (§6), making the "complete" version a three-loop change with the same unauthorized blast radius
 - wrap the `normalize()` call in `try/except Exception` at the handler: defect-adjacent widening — answers for every class including future bugs and leaves a half-mutated body reaching `estimate_tokens` in an unspecified state; `DC-CR3-F-04-r1` rejected this exact family in this same file one pass ago
 - a tolerant/strict parameter on `normalize()`: new mechanism at a two-caller function for an observable outcome the narrowed call already produces with fewer bytes (gate step 2 collapses mechanisms that produce the same outcome), and it keeps three steps of dict surgery on a route whose only need is the digest
 - revert `T019` step 1 (key off the raw body): preserves the verified `CR3-F-02` defect, breaks the carried protected invariant, reddens `tests/test_routes_integration.py`'s accepted nonce witness, and reopens an increment `083-01-yield` accepted
 - accept the 500 as a recorded risk, or route it to the roadmap with the `/v1/messages` half: fails the `latent` fix-or-pin rule (nothing pins it) and is the candidate `DC-CR3-F-01-r1` rejected on the same reasoning — a rejection this branch introduced, sitting on the merge unit while its boundary is open
Risk shape: unchanged — one call site narrowed to the step the invariant needs; `session.py`, `routes/messages.py` (334, §RM-001's debt) and `routes/chat.py` (300/300) byte-exact and `normalize.py` byte-exact or carrying only a delegating entry for the system step (clause 6); `T019`'s accepted witnesses keep their green, so no accepted evidence is invalidated — checked, not assumed: no test anywhere asserts a `count_tokens` **estimate value** (`grep -rn count_tokens tests/` → the seeded-digest cases, `test_transport_parity.py:126`'s status-and-key check, and `test_upstream_answer_relay.py:331`'s route-id parametrization), so the estimate shifting back to base's byte-count is unasserted by construction; the only observable delta is a 500 becoming the 200 base and the pre-range branch both gave. No new mechanism, no new module, no new configuration.
Deferred additive work: the `/v1/messages` malformed-element 500 — shared by base, `6508941` and HEAD, protected by parity, and a **product posture** (what a proxy answers to schema-invalid input), not a defect this session introduced — is routed to `roadmap.md` **§RM-012** with this evidence pinned, owner and trigger named there, and carried as one §Accepted risks line plus one §Elevated to user line so the decline is visible rather than silent. Also deferred to Interview's promotion decision (zero registry bytes moved here): if the carried `EI:` handle is ever promoted, its statement should name the **step** the digest depends on, not the whole function, or a later pass will read "after normalize" as a demand for the four-step call and restore this 500.
Task: T019.002 step 1 (nested under the owning task; one-member wave, prework of the MS-001 close)

## Per-pass routing narratives trimmed 2026-10-07 (`084-assumption-decide`)

Five superseded per-pass narratives (the 2026-10-06 CR3 run-42 composition routing, the run-40 atomic routing,
the CR2 closure-packet routing, and both `seam-S15` composition routings) moved **verbatim** to
[handoff-archive.md](handoff-archive.md) §Trimmed 2026-10-07 (`084-assumption-decide`, per-pass routing
narratives), so this baton file does not grow past its already disclosed overage. Nothing was rewritten: every
`DC-*` entry those passes wrote is still in this file under its own §Deterministic Closure Ledger section, every
`## Planned Task Context` blueprint stays here, and `CR2-X-1`'s malformed return resolves through the archive
heading. Live wave records that did **not** move: §Closure-packet routing — 2026-10-06 (MS-001 boundary, CR3
batch) below, and this pass's §Assumption routing — 2026-10-07.

## Closure-packet routing — 2026-10-06 (MS-001 boundary, CR3 batch): dispositions and the wave

- **absorbed → one wave**: `CR3-F-01` + `CR3-F-03` → **T018** (server.py, messages.py stream arm — wave
  **anchor**); `CR3-F-02` + `CR3-F-04` → **T019** (count_tokens.py, messages.py arms, chat.py client arm);
  `CR3-F-05` → **T020** (tests/test_compaction.py). One local Yield for the whole wave. **Separation reasons:
  none needed** — same owning route (implementation Forward), each member few-lines-complete, `T018`/`T019`
  share `messages.py` at disjoint functions (`_stream_single_request` kwarg+arm vs the two parse arms; editing
  the same file alone is not interference), `T020` is file-disjoint. Intra-wave order `T018` → `T019` (a rebase
  sequence at the shared function, recorded so two edits do not collide), `T020` anywhere. Execution order:
  the batch is **prework of the MS-001 boundary close** — land before the assumptions recheck on those bytes
  and before Yield Red's final semantic pass; no plan `--before` referent exists (no pending planned task).
- **Retry owners**: each member task's own counter, starting 0 (the anchor is scheduling identity, not a shared
  cap). `Forward->Yield Retries` carries the wave row.
- **Assumption impact**: `required:new-surface` on all three members — the CR3 packet names no seam; `S9` is
  table-marked `out-of-scope`/`unverified`; the boundary's already-owed assumptions recheck (next act after the
  Yield) is the lane that re-bases any fingerprint these edits move. No seam is held open by this routing;
  `S15` stays terminal.
- **accepted → §Accepted risks**: the eleven P3s plus `CR3-F-06` (the P2 disposed by Reach First), each with
  its verdict and citation. Seven land on **existing** owners — `F-06`/`F-12`/`F-13` on §RM-005 /
  `QB:cli-surface/undocumented-flags-and-stats-db` (row amended to name the serve-script env axis and the CR3
  re-confirmation explicitly — no duplicate), `F-07` on 057 `F-1`'s deferral (owner + trigger stand verbatim) —
  and three new records exist: `roadmap.md` §RM-009 (`F-11`, 1%-category per `codify-scope`, base parity
  verified at `legacy:674-676`; also in §Elevated to user, `Reported: —`), `QB:lint/ruff-f-select-findings`
  (`F-15` — count re-derived: 8, all `tests/`), `QB:legacy-monolith/in-file-provenance-header` (`F-16` — the
  report's own "if declared reference-only" condition met in session artifacts, named as owed in-file).
- **078's absorbed P3s**: `078-P3-1` (unwitnessed format-set clause → `Forward`, unit tier) and `078-P3-2`
  (stale registry gloss → `authority-registrar`) **keep their owners** — 078 issued both minutes ago on this
  boundary's record; CR3 neither blocks nor amends either, and re-routing settled work in a new pass is churn,
  not schedule room.
- **F-01's scope ruling (stated plainly, as the dispatch required)**: absorbed **into this session**, following
  057's four branch-wide migration defects and the CR2 batch precedent (`T013`–`T017` absorbed on the
  branch-merges-as-one-unit reasoning) rather than departing it. The branch therefore does **not** carry a
  reproduced 413 into review, and no roadmap destination/re-open trigger is opened for it.
- **F-03 vs verified-sound item 2 (stated plainly)**: not a contradiction and not a re-open — the clearance
  covers the post-commit **mechanism** and the timeout-class accounting inside the arm; the finding indicts the
  **deadline policy** upstream of the arm. `T018` must-not-touch keeps the arm and its injected-fault witnesses
  byte-valid; the two relayed statements coexist in `DC-CR3-F-03-r1`'s closure basis.
- **FK-011**: untouched, still the only open Fork — no `question` was emitted, no `reserve-decision-record`
  invoked; nothing in this batch outranks it. The `T019` fix is posture-independent of it (like `T016`'s guard
  was of `CR2-F-3`'s).
- **Lane discipline honored**: closure-packet routing only — zero production/test/`invariants/` bytes moved;
  no seam recomposed, no claim re-extracted (`claims.md` pending-Routing rows: 0); the packet
  (`iterations/079-codex-review.md`) has its `Disposition:` lines and a §Routing Record filled **by its routing
  owner**, which is what 079 §Step 3 anticipated ("057 filled CLOSURE: blocks because that pass *was* the
  routing pass"); no relayed wording or severity label altered.
- **Context budget**: this pass grew `handoff-context.md` materially (one status bullet, three task blocks, one
  ledger section, one routing section, eleven accepted rows, one elevated row). The file's over-cap disclosure
  (739 → ~1050 lines against the 200 cap) stands as 079 §Observation 4 recorded it: the deep lossless trim into
  `handoff-archive.md` remains `phase-boundary-finalize`/`session-squash`'s act, not a routing pass's.

## Assumption routing — 2026-10-07 (run 45, `S20` third-party atoms 418/419/422/423) — SUPERSEDED, moved verbatim

The whole narrative moved to [handoff-archive.md](handoff-archive.md) §Trimmed 2026-10-07 (`089-assumption-decide`) as a context-budget move, nothing rewritten. Every durable consequence it carried is still live elsewhere: the three `absorbed` rows and the one `malformed` return in their `### DC-claim-4xx-r1` ledger rows above and in `T019.001`'s plan row, the reach determination and base comparison in `evidence/n6-reach-and-base-comparison.txt`, row 425's landing and `S20`'s terminal disposition at [086-assumption-decide](iterations/086-assumption-decide.md), and `T019.001`'s acceptance at [085-01-yield](iterations/085-01-yield.md). Full pass record: [084-assumption-decide](iterations/084-assumption-decide.md).


## Closure-packet routing — 2026-10-17 (CR5 batch, delta `c5e2be4…6a2069b`): dispositions

Mode: external-review routing on the fifth round's delta; the packet names no assumption seam, so resolution follows
the owning route's acceptance gate and no composition is owed. `assumptions/claims.md` carries **zero live
`Routing: pending` rows** and no seam awaits a disposition. `MS-001`'s `SEMANTIC-APPROVE` at Checkpoint 213 is
untouched: this pass opens no task, no wave and no recomposition.

| Finding | Disposition | Reach verdict | Owner / destination |
|---|---|---|---|
| `CR5-F-01` (P3) | **accepted** | `unreachable` in the production sense — a docstring is read by no caller or consumer; Reach First's inert-wording class | `QB:tests/count-tokens-parity/over-reach-docstring-witness` — the next pass that edits `tests/test_count_tokens_parity.py`; no task, no wave |

**No Forward touch before the boundary closes.** In a late round an unusual-or-rare finding's fix reads `not cheap`
and is fixed only as a ride-along in a repair already entering that code; no repair wave is scheduled and `MS-001` is
`SEMANTIC-APPROVE`, so the docstring waits for its module's next touch. The record-only label in
`evidence/t019-002-fail-first-and-replays.txt` §A is corrected in place, transcript body unchanged. Ledger:
[094-assumption-decide](iterations/094-assumption-decide.md).

*The CR4 batch's closure-packet routing section (2026-10-07) moved verbatim to
[handoff-archive.md](handoff-archive.md) §Trimmed 2026-10-17 (`094-assumption-decide`); its durable
consequences stay live above: `DC-CR4-F-01-r1`, §Accepted risks `CR4-F-01` (second half), and `roadmap.md` §RM-012.*


## Trimmed 2026-10-17 (`097-yield`) — accepted risks

## Accepted risks

- `CR4-F-01` (second half) — **the `/v1/messages` malformed-element 500, declined on parity, with its evidence
  pinned.** `{"tools":[null]}`, `{"messages":[null]}` and `{"messages":["x"]}` POSTed to `/v1/messages` answer 500
  with an `AttributeError` out of `normalize()`'s element loops (`normalize.py:75`/`:117`, and `:192` behind them
  once either is hardened). **Reach verdict: `latent`** — exposed route, no guard prevents the input. Disposition
  rests on authority, not preference: base answers the identical 500 at the identical site in its own `normalize()`
  (`evidence/cr4-probe-matrix.txt` §3), the package did so before the range and does so after it (§1), so nothing
  this session introduced needs restoring there, and `G-003` holds that route's behavior still with "no
  normalization changes" in its constraint column. Changing it is a product posture decision — what a proxy answers
  to schema-invalid input — with a three-way option set and a completeness obligation recorded at
  `roadmap.md` **§RM-012** (owner: the route-input posture design pass, co-owned with §RM-001). Citation's
  freshness: §RM-012 carries the probe harness, and `T019.002`'s Done-when clause 5 pins the current answers with
  witnesses that redden if the posture changes without the roadmap row being taken up.

- `CR4-F-02` — blank line at EOF in `tests/test_upstream_answer_relay.py:406`, introduced by the range
  (`9c90675`). **Reach verdict: inert** — no runtime reader, no consumer, and no gate on this tree reads the byte
  (no `.github/workflows`, no `[tool.ruff]`; `QB:lint/ruff-f-select-findings` selects `F` rules only, which exclude
  `W391`); the sibling shape pre-exists in `tests/test_dump.py` at `6508941` and still carries it
  (`evidence/cr4-probe-matrix.txt` §8). Owner: the next Forward pass that edits either file
  (`QB:tests/blank-line-at-eof`). One byte, no task, no wave.

- `CR5-F-01` — the over-reach guard's docstring at `tests/test_count_tokens_parity.py:95-103` names two triggers
  that do not redden it: a narrower key and hardened loops. Evidence: `iterations/094-assumption-decide.md` §`CR5-F-01`
  (re-measured, not adopted — dropping `strip_billing_nonce` reddens the digest class 3, hardening the loops reddens the
  `/v1/messages` posture class 3, and the class's own trigger is a stricter `_read_body` — 400 on the unparseable
  body, measured 1 of 4). **Reach verdict: `unreachable` / inert wording** — no production caller or consumer reads a
  docstring; the suite is sound and the guard exists, only its prose attribution is wrong. Doc line
  `QB:tests/count-tokens-parity/over-reach-docstring-witness`, whose owner is **the next pass that edits that module**
  (next-touch trigger); no task, no Fork, no wave; it does not hold any seam open. Cross-reference: **Neutral**.

- `S16-(b)` — missing contract: the error envelope this proxy owes a client for a **request-level rejection it does
  not author**. The size rejection arrives as `413` + `text/plain` + aiohttp's own English sentence
  (`HTTPRequestEntityTooLarge`'s inherited `text=`, `web_exceptions.py:316-321`; media type from
  `web_response.py:652-665`), against the repo's JSON `{"error": …}` at every failure site it does author
  (`messages.py:31`, `chat.py:32`, `count_tokens.py:23-25`). Evidence:
  `assumptions/compositions/S16-body-limit.md` §What Is Required (b). **Reach verdict: `unreachable` for the size
  path** — the only producer of that settlement is the gate `T018` step 1 disables
  (`client_max_size=0`; row 393 `CL:aiohttp/client-max-size-falsy-skip`, HOLDS, skips the comparison at all four
  reads), and no other route class reaches it. Disposition rests on Reach First's `unreachable` rule, not on
  preference; the generalized remainder (the protocol-level generated `400`/`500`/`504` bodies a raised-out handler
  produces — row 230, `web_protocol.py:513-524`) is out of this session's scope by `codify-scope` rule 2 and rides
  **`roadmap.md` §RM-010**. No task, no Fork, no wave, no re-extraction; `S16` is not held open by this line (it
  stays open on `T018`'s post-implementation recomposition). Cross-reference: **Neutral**.
- `S16-decompress` — the request direction's second, unrelated size gate, and what it implies. Evidence:
  `assumptions/compositions/S16-body-limit.md` §Notes, third bullet, re-read against `utils.py:20-26`
  (`forward_headers` keeps `Content-Encoding`) and `buster_ripper_legacy.py` (zero `gzip`/`decompress`/
  `Content-Encoding` references; base read bodies through Starlette's `request.json()`/`request.body()`). What is
  true on current bytes: the server decompresses a **request** body whose `Content-Encoding` names
  `gzip|deflate|br|zstd` (`http_parser.py:556-558, 799-804`), so (i) any future `client_max_size` bounds
  *decompressed* bytes, not wire bytes — moot while the value is `0`, which is why the disposition is accepted and
  not absorbed; (ii) `DeflateBuffer`'s `DEFAULT_MAX_DECOMPRESS_SIZE = 2**25` raises `DecompressSizeError`
  (`PayloadEncodingError ← BadHttpMessage ← HttpProcessingError`) — **not** an `HTTPException`, so it lands in the
  protocol's `except Exception` arm and becomes a generated 500 (row 230) rather than a 413; and (iii) the
  transparent decompression is invisible to the relay: `passthrough.py:14` + `:31` forward *decoded* bytes under the
  peer's own `Content-Encoding` label, and the two `json=` paths re-serialize a body the peer labelled encoded — a
  wire mismatch base could not produce because base never decompressed a request body. **Reach verdict: real but
  rare, and no obvious fix where it shows → the 1% category** (needs a client that compresses a request body; the
  deployed launcher documents none, and the harm splits between a 500 and an upstream decode error). The agent did
  no repair and built no proof: recorded once at **`roadmap.md` §RM-011**, one line in §Elevated to user
  (`Reported: —`), accepted meanwhile. No task, no Fork, no wave; `S16`'s dimensions are not widened to it (the
  composer's own note says it belongs to no listed dimension and no member claim). Cross-reference: **Neutral**.
- `S16-notes` (no-production-caller census) — the arms the composition traced and could not reach on this tree:
  `client_max_size=0` and a negative value (dimension 2 and row 393's note — a negative would 413 *every* request
  including body-less `GET`s, because `passthrough.py:14` reads unconditionally), `BaseRequest.post()`/`multipart()`
  enforcement (`web_request.py:722, 748, 775`), the `request.content`/`.payload` bypass, `clone()`/`with_method()`
  preservation, and a `Server` built without a `request_factory` falling back to the un-bound `1024**2`
  (`web_server.py:32, 55-63`). Evidence: `assumptions/compositions/S16-body-limit.md` §Notes, second bullet, with its
  greps. **Reach verdict: `unreachable`** — a cited closed set (`server.py:119` passes no value; no repo symbol
  assigns `_client_max_size`, which `web_app.py:116` exposes read-only; every construction path here runs through
  `run_app` → `AppRunner._make_server` → `Application._make_handler`, which always binds the factory). Accepted by
  rule; each arm becomes live only if a task adds a config channel or a streaming body consumer, and `T018` step 1
  adds the former as *unlimited*, which keeps all of them inert. No task, no wave. Cross-reference: **Neutral**.
- `S16-wording` — inert ledger wording. `seams.md`'s `S16` dimension cell says "the four POST route classes and the
  catch-all"; `server.py:127-136` registers **three** `add_post` routes plus one any-method catch-all (four
  POST-*receiving* endpoints, four body-reading handlers). Evidence:
  `assumptions/compositions/S16-body-limit.md` §Notes, first bullet, and §Dimension Coverage 4. **Reach verdict:
  inert wording** — nothing reads the count, and no composed outcome changes either way. Disposition: one collector
  doc line at `QB:assumption-ledger/row-s16-dimension-route-count`; no task, no Fork, no wave, and `S16` is not
  held open by it (081's rows 391/404 are the same class).
- `S17-sibling-arms` (with `S17-N3` and `S17-N4`) — raised by this pass while checking CE-1's reach, not by the
  composer. Every **other** arm of the shared session keeps an armed `total` and catches only `aiohttp.ClientError`:
  `_non_stream_single_request` (`messages.py:265-297`, inheriting `server.py:67`'s 600 s), `_make_chat_request`
  (`chat.py:139/150`, 300 s) and `handle_passthrough` (`passthrough.py:27/48`, 60 s). Per rows 330/331 the class a
  firing `total` raises is bare `asyncio.TimeoutError`, so at those three sites a deadline expiry escapes the arm
  and the protocol answers with its generated **504** (`web_protocol.py:518`; row 230) — not the site's declared
  502 + `{"error": …}` + `"Upstream error: %s"`. Evidence: `assumptions/compositions/S17-stream-deadline.md`
  §Counterexample CE-1 (same class, different site) plus the three sites re-read on current bytes. **Reach verdict:
  `reached`** (exposed surfaces, and 60 s/300 s are ordinary network time), **but out of this session's scope and
  blocking nothing**: `DC-CR3-F-03-r1` ranked its closure by parity at `legacy:664` and rejected touching the shared
  session or the non-stream arm explicitly, and base answered the *same* event with a 500 (it had no arm at all), so
  no parity obligation exists — the gap is between the branch's own arm and the class set aiohttp hands it. The
  widening is also not a few lines: `chat.py` is at **300/300** raw lines (measured; §RM-001 records the extraction
  arithmetic), so any import or arm added there breaches the FL-LNCP ceiling and drags a route restructure into a
  defect wave. Routed as one roadmap row, **§RM-010**, with the 504 mechanism, the base comparison, the capacity
  cost and the `N3` envelope question named as its owning material; `N4` (the deadline is per upstream attempt, so
  the SGLang retry loop can reach ≈3×N for one client request) rides the same row as its second open point —
  `DC-CR3-F-03-r1` already carried the sibling observation for Yield Red's reading, and this line is where it
  finally gets an owner. Also §Elevated to user (`Reported: —`). No task, no Fork, no wave; `S17` is not held open
  by it (the seam's subject path is closed by `DC-S17-CE-1-r1`/`-CE-2-r1`). Cross-reference: **Neutral**.
- `CR3-F-06` — `main.py:75`'s `--strip-images` help promises "cache them for later retrieval" while
  `image.py:172-173` (re-verified verbatim) states the cached source **has no reader on the request path**.
  Evidence: `iterations/079-codex-review.md` §CR3-F-06. Reach verdict: **inert wording** — no runtime reader;
  no witness pins the clause (grep re-run). Disposition rests on Reach First's inert-wording rule: the fix is a
  sentence, owned by **`QB:cli-surface/undocumented-flags-and-stats-db` / `roadmap.md` §RM-005**, whose S10–S13
  amendment already owns this help text's scope; CR3 re-confirmation marker appended to that row. No task, no
  Fork, no wave; the *behavior* (strip = payload gone for the model) is the flag's documented purpose.
- `CR3-F-07` — `has_images`/`strip_and_cache_images` (re-located: `image.py:61`/`:166`) see top-level user
  content blocks only, so `tool_result`-nested images are neither detected nor stripped. Evidence:
  `iterations/079-codex-review.md` §CR3-F-07. Reach verdict: real input class, but the remediation is **additive
  scope with an existing owner and trigger** — 057 `F-1`'s `Deferred additive work` line verbatim ("owner =
  image-slice Forward task; trigger = when tool_result-nested images are in scope",
  [057-codex-review](057-codex-review.md):61) — and the detector and writer **agree** at top-level scope
  (re-verified `image.py:31-36,:61-69,:166-200`), so this is missing scope, not divergence; §RM-005's note
  carries the same deferral. The only consumer is planned work → accepted; decided when that scope arrives, no
  duplicate row.
- `CR3-F-08` — `normalize.py:76`'s stable sort leaves same-name tools in incoming order. Evidence:
  `iterations/079-codex-review.md` §CR3-F-08. Reach verdict: **impossible in well-formed input, zero effect on
  the named consumer** — duplicate tool names are API-invalid, and `session_id()` reads **`system` only**
  (`session.py:95-98`), so tool order cannot move the session key at all; the residual (KV-prefix byte drift
  under a duplicate-name body) reproduces base exactly, which sorted the same stable way. Accepted; no row (a
  fix would be a parity deviation inventing tiebreaks for an API-invalid body).
- `CR3-F-09` — `passthrough.py:23`'s `dict(request.query)` drops repeated query parameters. Evidence:
  `iterations/079-codex-review.md` §CR3-F-09. Reach verdict: **latent, base parity** — the base folded
  identically (`buster_ripper_legacy.py:908` `params=dict(request.query_params)`, re-read this pass), and no
  documented consumer sends repeated params on these paths. Parity is this migration's settled contract (same
  terminal class as `forward_headers`' egress fold, S10); accepted, no row.
- `CR3-F-10` — `passthrough.py:56`'s module-bottom `config` import and the `add_route("*", …)` verb widening
  versus base's five enumerated verbs. Evidence: `iterations/079-codex-review.md` §CR3-F-10. Reach verdict:
  **no defect** — the import works and is deliberately marked (`# noqa: E402`; the handler reads
  `config.UPSTREAM` at call time), and the verb set is a strict relay-**superset** of base's
  (`legacy:898` GET/POST/PUT/DELETE/HEAD): the difference is which upstream answers, no value lost; no
  documented client sends the extra verbs for these endpoints. Accepted; no row.
- `CR3-F-11` — `find_message_start` (`utils.py:91`, locator re-verified) scans each network chunk in isolation,
  so a `message_start` split across chunks is missed and that turn's stats (TTFT + input_tokens) are lost;
  self-heals on the next turn's event. Evidence: `iterations/079-codex-review.md` §CR3-F-11. Reach verdict:
  **latent** — needs a chunk boundary inside the first SSE event; base parity verified per-chunk
  (`legacy:674-676`); 1%-category per `codify-scope` (real but rare; cause lives in the streaming parse, which
  no session task declared; no gate blocks). Routed to **`roadmap.md` §RM-009** (owner: the streaming-parse
  quality pass; re-open trigger named on the row) and listed in §Elevated to user (`Reported: —`). No task, no
  Fork, no wave; it gates nothing.
- `CR3-F-12` — `config.STATS_DB` (`config.py:65`) left dead: `main.py:155-161` sets `session._db_conn` from the
  click param and never touches the global (re-verified: the global's only occurrence in the tree). Evidence:
  `iterations/079-codex-review.md` §CR3-F-12. Reach verdict: dead value. **Already tracked** at
  `QB:cli-surface/undocumented-flags-and-stats-db` / `roadmap.md` §RM-005 — re-confirmation marker appended
  there, no duplicate row; accepted.
- `CR3-F-13` — no `STRIP_IMAGES` or `--backend-type` env passthrough in `serve_buster_ripper.sh` (script
  re-read: six env vars honored, none of these two; nothing promised them). Evidence:
  `iterations/079-codex-review.md` §CR3-F-13. Reach verdict: **the only consumers are planned work** — the CLI
  surface's README+serve-script documentation task, §RM-005, whose row this pass amends to name the serve-script
  env axis explicitly so the pointer cannot rot into an orphan. Accepted; no duplicate row, no task.
- `CR3-F-14` — the committed ~100-year-validity `tests/qa/tls_peer_key.pem` (measured: `openssl x509 -enddate`
  → `notAfter=Sep 4 14:43:57 2126 GMT`). Evidence: `iterations/079-codex-review.md` §CR3-F-14. Reach verdict:
  **test material — a fixture is not a caller**; the pair is intentional by `FK-009` option 1 and documented at
  `tests/qa/tls_peer.py:11-14` (loopback `CN=127.0.0.1`, `IP:`-only SAN, no production reader anywhere); the
  key guards no secret value and long validity is the flake-bound, rotation buys nothing the gate would use.
  Accepted; no row, no task.
- `CR3-F-15` — eight `ruff --select F` findings with no `[tool.ruff]` pinned. Evidence:
  `iterations/079-codex-review.md` §CR3-F-15. This pass re-verified the premise on current bytes
  (`grep tool.ruff pyproject.toml` → empty) **and re-derived the count** (`uvx ruff check --select F` →
  "Found 8 errors", all in `tests/`: `tls_probe_entry.py:29` F401 `os`; `test_compaction.py:6` F401 `pytest`,
  `:373` F841 `alpha`; `test_normalize.py:3` F401 `pytest`; `test_routes_integration.py:308` F401 config;
  `test_server_hooks.py:58` F841 `mock_create_task`; `test_session_stats.py:12` F401 `_stats`;
  `test_tls_posture.py:11` F401 `click`). Reach verdict: hygiene — no gate runs ruff (deps: aiohttp+click
  only); nothing behaviorally broken. Routed to the new `QB:lint/ruff-f-select-findings` row (owner: the
  quality pass / next lint touch; two hits sit inside `T019`/`T020` touch-sets and are sanctioned pickups).
  Accepted; opens no wave.
- `CR3-F-16` — `buster_ripper_legacy.py` imports dependencies that are not installed. Evidence:
  `iterations/079-codex-review.md` §CR3-F-16. Reach verdict: **never executed on any path** — the file is
  imported by nothing (`grep` → docstring/comment citations only), is coverage/package-excluded by location, and
  **self-declares its deps** via the PEP 723 header (`legacy:1-9`: fastapi/uvicorn[standard]/httpx/click — a
  standalone `uv run` script by construction). The report's condition — "acceptable if declared reference-only"
  — holds across session artifacts and the plan's scope exclusions, but **not in-file**: routed as one
  doc-line, the new `QB:legacy-monolith/in-file-provenance-header` row. Accepted meanwhile; no task.
- `S15-N5` — duplicated **single-valued** upstream response header names now reach the client as several
  field-lines (pre-`T017` the shared filter folded them to the last). Evidence:
  `assumptions/compositions/seam-S15.md` §Notes N5, with reach established on the served path by
  `evidence/t017-live-relay-transcript.txt` §2 (FIXED run counts `date: 2` / `server: 2` downstream of
  `hyperipper.local:30001`, which repeats both names on every response; the 1-in-6 sample carries two *different*
  `Date` values). Reach verdict: **reached** — this is on the wire today, not a hypothesis. Held **accepted**
  because the closure gate admits no other remediation: the required outcome is `DC-S15-CE-2-r1`'s own ("the
  response relay discards no value the base relayed"), comma-joining is rejected there on its face, collapsing
  duplicates of a name HTTP treats as single-valued discards a value in exactly the reached differing-`date`
  case, and deduplicating identical pairs adds a rule no witness or consumer needs while re-firing this seam's
  own `Invalidates On` 3. The composer found and this pass re-checked no consumer that misroutes on the
  duplicate lines, and the malformation is the upstream's, relayed rather than invented. Cross-reference:
  **Neutral**. No task, Fork, wave or roadmap row; re-opens only through `Invalidates On` 3 (a site that starts
  passing `content_type=`/`charset=` alongside relayed headers, or any change to the return type/comprehension
  shape) or a field report of a client that cannot parse a repeated line. Not §RM-008's: that row rules on
  content-coding posture and its three shapes only.
- `CR2-X-2` — `README.md:240` architecture tree still lists `app.py  FastAPI app + lifespan` (FastAPI-era wording
  after the aiohttp migration). Evidence: `evidence/codex-review-post-071-packet.txt` item (b); tree verified
  stale at `dbd3072`. Reach verdict: **inert wording** — no code or witness consumes the tree block
  (`tests/test_serve_script_argv.py` pins §Serve script options only); doc line owned by
  `QB:cli-surface/undocumented-flags-and-stats-db` (`roadmap.md` §RM-005), amended by this pass to name it.
  No task, no Fork, opens no wave.
- `S15-N6` — the widened `response_headers` input union (`dict | CIMultiDictProxy`, `utils.py:29`) carries a
  `dict` member that **no production caller reaches**. Evidence: `assumptions/compositions/seam-S15.md` §Notes N6,
  re-checked here: `grep -rn "response_headers" buster_ripper/` → the definition plus exactly three call sites
  (`passthrough.py:41`, `messages.py:291`, `chat.py:172`), all passing `resp.headers`, i.e. the
  `CIMultiDictProxy` member. Reach verdict: **`unreachable`** from this seam's boundaries (a witness that passes a
  plain `dict` is not a caller), and both arms take the identical path because the predicate reads
  `k.lower()` on the name alone — so N6 states no claim that fails. Held **accepted**; it opens no task, Fork or
  wave, and the annotation's real imprecision — it does not name the `CIMultiDict` the new witnesses pass — is
  already owned by `QB:utils/response-headers-input-annotation`, so no second row is written for it.
  Cross-reference: **Neutral**.
- `S15-CE-1` — the proxy relays the upstream's `Content-Encoding` label over body bytes its **own** client
  already decompressed, at all three `utils.response_headers` boundaries. Evidence:
  `assumptions/compositions/seam-S15.md` §Counterexample CE-1, with reach established by execution in
  `evidence/s15-reach-and-base-comparison.txt` §1. Reach verdict: **latent** — the proxy asks for an encoded
  answer on both egress arms (aiohttp's own `Accept-Encoding: gzip, deflate` observed on the wire when the
  client names none; a client-supplied one forwarded unchanged, since `utils.py:23` drops no
  `accept-encoding`) and has no arm that declines decoding (`grep -rn "auto_decompress" buster_ripper/` →
  zero; the only session, `server.py:65-68`, names `timeout=`/`connector=` only) — but the deployed
  upstream answers identity to `identity`, `gzip, deflate` and `gzip, deflate, br, zstd` on `/v1/models`,
  on a 66 535-byte `/metrics`, and on a live `POST /v1/chat/completions`, with `server: uvicorn` terminal
  (no compressing gateway), and no witness or gate has ever named `Content-Encoding` on this path
  (`grep -rni "content-encoding" buster_ripper/ tests/ docs/ README.md` → zero; the
  `test_header_parity.py` census covers only the eight hop-by-hop names + `Content-Length`). Held **accepted**
  rather than repaired because the current behavior **is** the settled contract: base parity
  (`buster_ripper_legacy.py:630-637` kept the name; httpx decoded `.content` the same way and advertised
  `Accept-Encoding` too), so the remedy is a *parity deviation* with three materially different shapes and
  no settled authority ranks them — the 1% category per `codify-scope`. Routed to
  `roadmap.md` §RM-008 (owner: the header-lane design pass; re-open trigger: the next decision about
  relayed names, or the first upstream that answers encoded) and listed in §Elevated to user.
  **It gates nothing:** no task, no Fork, no wave, no seam held open; `S15`'s open state is owed solely to
  CE-2's repair. The composition's Notes N1 (`br`/`zstd` not installed → the failure lands inside the
  existing `except aiohttp.ClientError` arms; re-trace if a codec is ever installed) and N2 (the egress
  direction folds the same way **and that is base parity**, Starlette yielding per-occurrence pairs into
  the base's dict comprehension) are records, not findings, and open nothing.
  Rows 384, 385, 386, 387, 389 → Routing `accepted-consequence`; States remain `HOLD`.
- `claim-391` — aiohttp's `web.Request` enforces its body-size limit "from the value configured on the application
  serving the request": the verdict settles the mechanism exactly and cannot settle which application that phrase
  names (root/serving app → HOLDS; `request.app` = `match_info.current_app` → DOES-NOT-HOLD under `add_subapp`, whose
  own `client_max_size` is inert). Evidence: `assumptions/verdicts/claim-391.md`. Reach verdict: **inert wording** —
  the fork cannot split an outcome on this tree: the package constructs one flat `web.Application`
  (`server.py:119`, re-read) and `add_subapp` greps **zero** under `buster_ripper/` and `tests/`, so under *either*
  reading the value that governs every request is the one `T018` sets to `0`; the app→request propagation the seam's
  boundary value needs is carried by rows 390/392–395 (`HOLD`, published). Disposition rests on Reach First's
  inert-wording rule: doc line at `QB:assumption-ledger/row-391-serving-app-wording` (collector lane, next
  `claims.md` touch); no task, no Fork, no wave, no re-extraction, and `S16` is **not** held open. Cross-reference:
  **Neutral**. Row 391 → Routing `accepted-consequence`; State remains `UNCLEAR`.
- `claim-404` — `bytes.decode(errors="replace")` leaves bytes valid under the target codec "unchanged": the artifact
  names both readings ("same characters as `errors='strict'`" vs "same bytes") and both readings of "valid"
  (per-byte vs complete sequence in context) and says the pinned pair settles it on three fingerprinted pages.
  Evidence: `assumptions/verdicts/claim-404.md`. Reach verdict: **inert wording** — the pinned reading is the one
  `T019` step 2(a) stands on (the ASCII overflow text survives replacement at `messages.py:276`), and the rival
  reading is false of *every* `decode()` call rather than a property of `'replace'`, so no consumer can be relying on
  it; rows 402/403 (`HOLD`) carry the strict default and the U+FFFD marker the arm actually depends on. Doc line at
  `QB:assumption-ledger/row-404-replace-preservation-wording` (collector lane); no task, no Fork, no wave, and `S18`
  is not held open by this row (it stays open on `claim-399`). Cross-reference: **Neutral**. Row 404 → Routing
  `accepted-consequence`; State remains `UNCLEAR`.
- `claim-425` — the general `str`-body round trip through `aiohttp.web.Response` fails: `Response.body`'s setter calls
  `PAYLOAD_REGISTRY.get(body)` with no encoding or content-type argument, so `StringPayload` always encodes UTF-8
  whatever the response advertises — a declared non-utf-8 charset yields mojibake (`iso-8859-1`) or raises
  `UnicodeDecodeError` (`ascii`) under `.text`, and `charset=` without `content_type=` is silently dropped.
  Evidence: `assumptions/verdicts/claim-425.md`, re-verified arm by arm by execution at
  [086-assumption-decide](iterations/086-assumption-decide.md) against its own three digests. **Reach verdict:
  `unreachable` for both falsifying arms**, by a closed call-site census — `evidence/claim-425-charset-arm-reach.txt`
  §2/§5: one `str` at `body=` in the package (`messages.py:163`, `content_type="application/json"` naming no charset),
  four bytes-body sites, nine `text=` sites, and **zero `charset=` arguments anywhere** (the word is in comments only);
  the falsifier needs a *spelling*, and the upstream controls the body's value and the status, never that. The arm
  `messages.py:117` stands on is `reached` **and holds**, certified by `CL:aiohttp/web-response-str-body-text-property`
  (421) with 422's accessor enumeration. Disposition rests on Reach First's `unreachable` rule; no task, no Fork, no
  wave, no re-extraction, and **it does not hold its seam open** — `S20` is terminal, `Routing: absorbed`
  (`assumptions/seams.md` §Run-46 routing). Cross-reference: **Neutral**. Row 425 → Routing `accepted-consequence`;
  `State`/`Permanent` stand for the promotion lane. Kept for the design pass: the hold rests on **two unrelated
  defaults agreeing** (`StringPayload`'s hardcoded `utf-8`; `.text`'s `or "utf-8"`), while aiohttp's stable docs promise
  header-driven encoding — so naming a charset at that constructor, or reading `.body` again, re-opens what
  `T019.001` closed, silently for non-ASCII. One inert-wording doc line at
  `QB:assumption-ledger/str-body-charset-doc-divergence`; the durable guard stays the non-authoritative Invariant
  candidate with Interview (*"…goes through `.text`"*).

- `S16-publication` — `seam-S16`'s post-`T018` recomposition is `SAFE` and unpublishable: member row **230** has no
  permanent `CL:` handle, so the record cannot satisfy *every member claim resolves to a permanent handle*. **Reach
  verdict: `reached`** for the boundary (`server.py:125`'s `client_max_size=0` → every route's body read —
  `assumptions/compositions/S16-body-limit.md` §Delta Scope, §Reach), and the *gap* is in a metadata value no code
  reads: its only consumer is a later session's collector deciding whether to skip a composer. Disposition rests on
  authority, not preference — `assumption-gate` §(iii): **"Promotion failure blocks reuse but does not rewrite a
  verdict"**, and §(iv)'s proceed rule asks `registry-safe` only of *inherited* seams — so `S16` is
  **terminal-for-boundary**, `State: recheck` + `Routing: absorbed` unchanged, `Permanent: —`, and **it does not hold
  its seam open**. Re-open path is the record's own `Invalidates On` **7** ("Row 230 gains a permanent `CL:` handle …
  promotion condition only, not a behavior change … requires no re-trace") for publication and **1–6** for behavior.
  Citation's freshness: the blocker artifact is `assumptions/verdicts/claim-230.md`, which reads a bare
  `PROMOTION: ELIGIBLE` over `web_protocol.py sha256:eacf5d33…` — the same bytes the composition re-hashed, so the
  substance is promotable the moment the promotion lane can reach it. Debt carried once, with `330`/`331` and
  `S18`'s five other handleless rows: `QB:assumption-registry/handleless-members-blocking-co-publication`.
  Cross-reference: **Neutral**. No task, no Fork,
  no wave, no recomposition. Full ruling: [assumptions/seams.md](assumptions/seams.md) §Run-47.

- `S17-publication` — same shape, member rows **330** and **331**: `SAFE` recomposition, no `CL:` handles, therefore
  no `CO:` record. **Reach verdict: `reached`** for the boundary (the streaming call's request-local
  `ClientTimeout(total=None)` at `messages.py` — `assumptions/compositions/S17-stream-deadline.md` §Composition,
  Transformation 1–2), `unreachable`-in-the-future for the gap: nothing in the package reads a `Permanent` cell, and
  only a later collector does. **Terminal-for-boundary**, cells unchanged, seam not held open. Re-open:
  `Invalidates On` **2** — elevation "changes the manifest's `NO-HANDLE` rows and therefore this fingerprint;
  recompute. The verdict itself is not otherwise affected" — with **1** the behavioral clause, which did **not** fire
  (all five members pin `pkg:aiohttp` bytes, still 3.13.3). The seam's stale *locators* are recorded, not treated as
  an invalidation: `T019.001` inserted a 12-line note at the old `:106`, so `:144-147` is the same text at
  `:155-158` today, and `assumption-promote` §Identity And Upsert is explicit that a line shift does not invalidate a
  claim. Also discharged here by measurement, for its owner to fold: `QB:assumption-ledger/row-57-version-pin` —
  `claims.md:185` now reads `B315 | HOLD | none | CL:aiohttp/per-request-timeout-override` and the registry record's
  fingerprint is the installed `client.py` `7ce41fc1…` at `Version: aiohttp 3.13.3`, closing the v3.12.15 skew §Notes
  N1 recorded. Debt carried at the same key. Cross-reference: **Neutral**.

- `S18-publication` — `SAFE`, and its publication is **barred rather than deferred**: of eighteen members, **402/403**
  are blocked by an annotated line-2 token (format — already keyed at
  `QB:assumption-registry/annotated-promotion-headers`, not duplicated here), **68/69/71** record
  `Source Fingerprint: unavailable` on `lang:python` propositions settled from documentation pages, which
  `assumption-promote` §Eligibility refuses to promote at all — no mechanical act publishes them, only new evidence at
  a hashable source would — and **72/137** are ineligible because "no single consulted locator establishes both", a
  row-shape defect whose cure is re-extraction, not promotion. **Reach verdict: `reached`** for the boundary
  (peer-named charset → `request.json()` → the three parse arms, `assumptions/compositions/S18-decode-failure-surface.md`
  §Composition) and the gap is metadata with one future reader. **Terminal-for-boundary** on the same authority, seam
  not held open — with **one condition this entry must not smooth over**, because it is not bookkeeping: `S18`'s own
  §Invalidates On **5** says "**N1** changes if … `count_tokens.py:79-95` reorders keying relative to normalization"
  and "a fix landing under another seam still requires this record to be re-read before it is cited as current"; both
  `T019.002` (which narrowed `count_tokens.py:87` from `normalize(body, session_id="")` to `strip_billing_nonce(body)`)
  and `T019.001` (which landed under `S20`) moved exactly what those clauses name, after the record was written. This
  pass recorded that and did **not** re-adjudicate it — whether N1's conclusion survives is a composition call, and no
  recomposition may be dispatched (allowance spent, and barred by this route). **Yield Red must weigh it before citing
  `S18`'s `SAFE` as current on the final bytes**; `S16`/`S17` carry no such clause. Re-open for publication: the new
  key's collector arm (72/137) and evidence arm (68/69/71); for behavior: `Invalidates On` 1–6.
  Cross-reference: **Neutral**. No task, no Fork, no wave, no registry write; nothing on `S18` was rewritten.



## Trimmed 2026-10-18 (`098-assumption-decide`) — historical `T002` notes moved verbatim

- **`FK-A`/`FK-B` NOT askable** (decide 008; Forward discretion): `DUMP_FLUSH_INTERVAL` = **10 s** (`KV_POLL_INTERVAL`
  peer), CLI-exposed; `DUMP_FLUSH_BYTES` = **4 MiB** measuring **rendered-text UTF-8 bytes** (`len(_render_body(body).encode("utf-8"))`; T009 repairs the old `len(str)` shorthand),
  not raw JSON bytes. Never cite the displaced `½·(1 − F/N)` figure.
- **Capacity gate TRIGGERED for `T002`** (plan §Goals and Constraints): `dump_turn` is a pre-existing hard-L violation
  (L:44 after T001); the extraction spans **both writers** and must precede T002's additive work; post-change
  `dump.py` stays Green (<300). `main.py:main`'s closure (017): new options assign their globals through the option's
  own `callback` — `main()` gains no parameter.
- **T002's `norecursedirs` facts (verified)**: (i) `norecursedirs` and `--ignore=` are the same collection mechanism
  (`pytest_ignore_collect`, `_pytest/main.py:425-458`) — explicit gate selection still collects; (ii) the ini value
  **replaces** the default list (`:213-222`) — retain the defaults alongside `qa` (`.venv` covered by the pyvenv check).
- **Artifact identity (015/019)**: `DUMP_DIR/<sid>/<format>.log`, collapsed values only; key `(sid, format)`; append-only;
  documented-layout files in a dir are neither migrated nor removed (still the default's output).
- **Research (`US-002`, 007)**: SSD wear is bytes × WAF, not write-call count; logical bytes ≈ unchanged (M4 ≈ +1.5%)
  — a count proof is a **proxy** proof; **SQLite is not a wear win**. Deferred: `DUMP_DIR` filesystem, compression.
  Roadmap ([roadmap.md](roadmap.md)): MVP floor + three deferred entries.
- **Findings F-1…F-24**: definitions/history in the planning iterations (013/015/018/019) and git; every corrected
  reading (F-5, F-6, F-13, F-15, F-21, F-24) is embedded in the plan's task entries, VP rows and `OI:` register.
  `buster_ripper_legacy.py` is excluded from scope, deliberately not updated.


## Trimmed 2026-10-18 (`098-assumption-decide`) — Notes section moved verbatim

## Notes (load-bearing for remaining tasks)
- **Protected invariants for yield**: the flush contains no `await`; cannot raise out of `on_cleanup`; is idempotent
  per pending batch; `turn`/`prev_text` advance at append time, never at flush time. `on_cleanup` order: cancel timer
  → final flush → close shared session.
- **Write-operation unit** (instrumented VPs): one write-bearing file op on a dump artifact (`write_text`, or
  `open(...,"a"/"w")`+`write`); mkdir not counted; SQLite outside the measured surface.
- **Settled, do not re-litigate or re-ask**: `FK-007`/Topic 1; `US-009`/`US-010` (QA harness → live remote
  `http://hyperipper.local:30001` — **endpoint corrected 2026-10-17 to `:30000`; `:30001` is a peer Buster
  Ripper proxy, not the model server**, never a stub); the four `OI:` statements; `MS-001` declares no flush / collapsed
  path observable at `MS-002` (022); every dump `VP-*`; `G-003`/`G-006`/`G-007`; the S-1 reading (⚠ the write
  reduction is **opt-in**); the format set + `documented` default; the measures (M1/M2 proof, M4 not, M3 unmeasurable);
  entries 1–3 and the six `PI:T002/dump_format/*` (`US-014`). TLS (`T006`): `FK-008` option 1 (no runtime record —
  `VP-T006-optout-is-explicit` NOT CREATED); `FK-009` option 1 (loopback peer, TLS gate only); `ClientSession` takes
  no `ssl` kwarg — `TCPConnector` is the only seam. Detail: plan-summary T006 + [017](iterations/017-decide.md).



## Trimmed 2026-10-18 (`100-yield`) — `T021` planned-task context moved verbatim

<details>
<summary>T021.001 Planned Task Context — make the documented-format gate's changed-line comparison compose with production's diff (S21)</summary>

- Owning route: **`forward-qa`** — **test bytes only**, `tests/qa/test_qap_ms001.py`; no production, config, `invariants/` or `decisions.md` byte. One member of the run-49 gate-repair wave (anchor `T021.001`), one local Yield.
- Assumption finding: **seam-S21**
- Evidence: `assumptions/compositions/S21.md` (`sha256:bdfa4a576c1e7681af18155a0c84835b42cb88bbd979eba8902f3cfc677c1282`)
- Disposition: **absorbed** — `DC-S21-r1`
- Status: **pending**
- Counterexample: CE-1 turn N−1 `messages: []` against turn N `[{"role":"user","content":"a\n\n"},{"role":"user","content":""}]` — production's `"".join` of `splitlines(keepends=True)` lines with `lineterm=""` glues the three control lines to the first inserted blank line; the gate drops that line as `---`-prefixed, so expected has 8 `+` lines and observed 7. CE-2 content `"line1\r\nline2"` against `"line1\nline2"` — production reports `-line1`/`+line1`, the gate's `splitlines()` expected side sees no change (`expected=[]`, `observed=['-line1','+line1']`). CE-3 content `"foo"` against `"foo\n"` — production reports the added final blank line, the gate's `.strip()` at `:384` erases it (`expected=[]`, `observed=['+']`); the same `.strip()` is a false-pass surface for a writer that omits the change. Reach: **latent** for all three — the gate's own `TC-107`/`TC-108` bodies are `turn_body` markers (one plain user message per turn, no CRLF, no trailing newline), so none produces the inputs; the only consumer is the QA gate `tests/qa/test_qap_ms001.py:194-211`, not a shipped-runtime caller. CE-3's false-pass surface is nevertheless **blocking for `MS-001`'s `resolved`** — it sits in the boundary's only content witness; CE-1/CE-2 ride along in the same repair.
- Required behavior: `assert_documented_contents` must compare the documented diff's changed lines over the public renderings with the same line identity production uses (`splitlines(keepends=True)` on both sides), recover the content line glued to the control lines rather than drop it, and extract the renderings without `.strip()` (removing only the artifact's record-separator newlines), so a trailing-newline-only change is visible. A correct writer must not be rejected on CE-1/CE-2, and a writer that omits any of the three changes must not pass.
- Done when: a focused regression per CE passes on the repaired gate and fails on the pre-repair gate — CE-1 (the empty-message/`"a\n\n"` pair), CE-2 (CRLF vs LF), CE-3 (`"foo"` vs `"foo\n"`, plus the omitted-trailing-newline writer). The two `T021` mutation replays (pre-normalization body; raw-JSON diff source) still red, the unmutated gate green, `QAP-MS001` 12/12, `QAP-MS002` 15/15, `QAP-MS004` 6/6, hermetic suite unchanged; `git diff --name-only` over the increment contains zero production, config, plan, contract or registry bytes.
- Yield replay: (a) unmutated writer → repaired gate green; (b) CE-1 pair → green repaired, red pre-repair; (c) CE-2 pair → green repaired, red pre-repair; (d) CE-3 pair and the omitted-trailing-newline writer → green repaired, red pre-repair; (e) the two `T021` mutation replays still red; revert every mutation byte-exact.
- Assumption impact: **scoped:S21**
- Invariant cross-reference: **Neutral** against `active`/`wip` (zero exist; every registry `Invariant:` row is `planned`). The repair serves the planned `PI:T002/dump_format/documented_format_is_retained` and its witness `TC-108`; no duplicate candidate and no invariant authority is created.
- Invariant candidate: none.
- Closure ledger: `DC-S21-r1`

</details>

<details>
<summary>T021.002 Planned Task Context — make `Proxy.post` reach the declared local endpoint under ambient proxy settings (S22)</summary>

- Owning route: **`forward-qa`** — **test-support bytes only**, `tests/qa/harness.py`; no production, config, `invariants/` or `decisions.md` byte. One member of the run-49 gate-repair wave (anchor `T021.001`), one local Yield.
- Assumption finding: **seam-S22**
- Evidence: `assumptions/compositions/S22.md` (`sha256:593e59ba2d3b3df8645dea2ae153ae5aaffe21eadcd43c067e22ebe44a8bb1e8`)
- Disposition: **absorbed** — `DC-S22-r1`
- Status: **pending**
- Counterexample: with `http_proxy=http://proxy.corp.example:3128` exported and `no_proxy` unset (or `localhost`-only), `Proxy.post`'s `urllib.request.urlopen(req, timeout=90)` installs the default `ProxyHandler()` fed by `getproxies()`; `proxy_bypass('127.0.0.1')` finds no match and the POST goes to the ambient proxy, so `assert_upstream_response` judges the ambient proxy's status/body, not the declared endpoint's. The preflight (`build_opener(ProxyHandler({}))`) and the proxy subprocess (`trust_env=False`) both bypass ambient proxies, so the gate's two hops disagree about which host answered. Reach: **latent** — the gate sets no `http_proxy`/`no_proxy` in `tests/`, CI or `pyproject.toml`; the consumer is QA-gate code (`tests/qa/harness.py:271-281`, called by the `MS-001`/`MS-002` oracle call sites), no shipped-runtime caller exists. The repair is still required before `MS-001`/`MS-002`'s `resolved`: the oracle must judge the declared endpoint on any host, not only on one with a clean proxy environment.
- Required behavior: `Proxy.post` must reach the local proxy at `http://127.0.0.1:<port>` regardless of ambient `http_proxy`/`no_proxy`, so the status and body judged by `assert_upstream_response` are the local proxy's relay of the declared upstream's answer.
- Done when: a focused regression sets `http_proxy` to a dead or decoy proxy (and leaves `no_proxy` unset), calls `Proxy.post`, and asserts the local proxy's response (e.g. 200 and the expected body) — red on the pre-repair harness, green after; the unmutated `QAP-MS001` 12/12, `QAP-MS002` 15/15, `QAP-MS004` 6/6 and hermetic suite unchanged; `git diff --name-only` over the increment contains zero production, config, plan, contract or registry bytes.
- Yield replay: (a) `http_proxy` set to a dead port → `Proxy.post` still returns the local proxy's status/body (red before the fix); (b) no ambient proxy → the same response; (c) the full `QAP-MS001`/`QAP-MS002` gates green.
- Assumption impact: **scoped:S22**
- Invariant cross-reference: **Neutral** — no `active`/`wip` invariant speaks to the harness's client hop; the registry `PI:T002` rows are `planned`, and the `CL:` records are evidence inputs only.
- Invariant candidate: none.
- Closure ledger: `DC-S22-r1`

</details>


## Trimmed 2026-10-18 (`101-01-forward`) — superseded `## Current Status`, verbatim

## Current Status

- **✅ T021 ACCEPTED-LOCAL (2026-10-17, [097-yield](iterations/097-yield.md))** — `assert_documented_contents`
  compares `turn_000.json` against the complete normalized body and each `turn_NNN.diff`'s changed lines
  against a `difflib` diff over the `self-contained` artifact's renderings; `TC-107`/`TC-108` drive
  nonce-bearing bodies at the unchanged `N + 1`/`2N` layout. Local Yield: three mutation replays reverted
  byte-exact, unmutated **12/12**, `QAP-MS001` 12/12, `QAP-MS002` 15/15, `QAP-MS004` 6/6, hermetic
  **511 @ 96.71%**; zero production/config/registry bytes. Defer-tier:
  `QB:qa-gate/body-drift-witness-inert`, `QB:test-qap-ms001/file-size`.
- **✅ RUN-49 GATE REPAIR ACCEPTED-LOCAL ([100-yield](iterations/100-yield.md), 2026-10-18).**
  `T021.001` uses production's line identity on both sides and recovers the glued content line; `T021.002`
  makes `Proxy.post` build its `ProxyHandler({})` opener. Affected scope **37 passed** (`QAP-MS001` 16,
  `QAP-MS002` 15, `QAP-MS004` 6) on the repaired bytes, hermetic 511 @ 96.71% unchanged, zero production bytes.
  Both members are `accepted-local`, not `resolved`. Detail: [100-yield](iterations/100-yield.md).
- **✅ MS-001 BOUNDARY: `SEMANTIC-APPROVE` at [093-yield-phase](iterations/093-yield-phase.md) (HEAD `62f8d7c`)**
  — four `VP-*` `pass`, first-party evidence (`:30000` preflight, 32/32 gates, hermetic 511 @ 96.71%); phase
  acceptance, `resolved` for `T001`/`T002`/`T005`/`T015`–`T021`, the `Delivered` entry and parent completion
  are **NOT published** — the closing Yield Red check owns them.
- **⚑ Gate upstream is `http://hyperipper.local:30000`** (user correction 2026-10-17; `:30001` is a peer
  Buster Ripper proxy). `README.md`'s `:30001` occurrences and the app's `--port` default stay untouched.
- **NEXT — the run-49 seam recomposition** (`S21`/`S22`), then the closing Yield Red check on the
  strengthened bytes, then `phase-boundary-finalize`. `FK-011` remains the only open Fork; every retry counter is 0.
- **CR5 (round 5) — no P1/P2; one P3 `CR5-F-01` accepted** (docstring wording, no task): [094-assumption-decide](iterations/094-assumption-decide.md).


## Trimmed 2026-10-18 (`103-assumption-decide`) — T022 drain blueprint (context budget)

<details>
<summary>T022 Planned Task Context — step-6a boundary drain of `quality-backlog.md` (sort → triage → drain in-scope Correctness)</summary>

 **Origin and why now.** `run-codify` SKILL.md step 6a ("Drain before review") applies `references/session-close.md` step 12 Source A at every cycle/phase boundary, before that boundary's review and before the base sync. This boundary has reached it: `MS-001`'s Yield Phase returned `SEMANTIC-APPROVE` at [093-yield-phase](iterations/093-yield-phase.md), and `T021`/`T021.001`/`T021.002` are `accepted-local` at [097-yield](iterations/097-yield.md) / [100-yield](iterations/100-yield.md). `resolved`, phase acceptance, the `Delivered` entry and parent completion are withheld pending **this drain + the boundary base sync + the closing Yield Red check**. This task is the drain; it is not the gate work.

 **Starting state (read it; do not trust stale judgements).** `quality-backlog.md` has no `## Correctness` and no `## Refactoring` heading. It is organized by the pass that recorded each line — `## Recorded by the CR2 wave's Yield (073-01-yield, 2026-10-06)`, `## Recorded by T017's Yield (075-01-yield, 2026-10-06)`, `## Recorded by T021's local Yield (097-yield, 2026-10-17)`, `## Recorded by run-49 assumption routing (098-assumption-decide, 2026-10-18)` — plus the undated head block. It holds **54 open `[ ]` lines** at this write. Many already carry `→ routed:`, `kept:` or `[x]` dispositions; those are dispositions, not sections.

 **The work, in order** (`session-close.md` step 12 Source A; `codify-plan-format` → Quality Backlog Record Contract):

 1. **Sort the file that has no section headings.** Create `## Correctness` and `## Refactoring` and move every line under one of them by the contract's one question: *can this line make the program behave wrongly, or hide that it does?* Yes → Correctness (a missing or weak test, an unhandled case, misleading output, a harness gap that can hide a bug). No → Refactoring (LNCP/FL-LNCP debt, restructuring, cosmetics). **Drop nothing**; keep each line's checkbox, locator prefix, `QB:` handle and evidence text verbatim. The pass headings may survive only as `###` group notes inside a section.
 2. **Re-ask the same question of every open line under `## Refactoring`** and move any line that can change behaviour into `## Correctness`.
 3. **Triage every open Correctness line as a fresh finding**, against the code as it is now — a line is an old judgement and the code has moved. Record **one line of reason per closure**, in the file: **still true** → close it with that evidence (`[x]` plus locator); **out of scope** → a `roadmap.md` row, and the line marked `→ routed:` to that row's `RM-*` handle; **reach** → unreachable / 0.01% / now-impossible → accepted risk, `[x]` with the Reach-Gate citation; **the 1% category** → recorded for the user, **not worked**; **now plain refactoring** → moves to `## Refactoring` and stays open. A line that survives as still-Correctness-and-in-scope is the wave's work.
 4. **Drain in-scope Correctness now, whatever its size** — one Forward → Yield wave. Scope is `codify-phase-close` rule 1, including its gray-area allowance. **Anything the wave finds is fixed in the wave, not recorded.**
    - Registry or invariant files (`invariants/…`, `assumptions/…`): closed by **authority-registrar**, which holds that write authority.
    - A line needing a user decision follows **Source B** (deterministic closure → resolved FK plus landed destination; an explicit deferral needs a complete `roadmap.md` row with the user's reason verbatim). Do not invent an answer.
 5. **Leave `## Refactoring` open.** At a cycle/phase boundary its lines are not drained; the session-end wave owns them. Yield must also read that section and **reject** if a line there can change behaviour.

 **Done when.** Both fixed headings exist; every open in-scope Correctness line is `[x]` with its evidence; every other line is `[x]` with its one-line reason or `→ routed:` to a named destination, and a routed line is not landed until that destination contains the complete `## Roadmap items` row (chunk plus `RM-*` handle, `Source` citing the `QB:*` handle). `## Refactoring` is left open for the session-end wave. The 54-open-line figure is the starting population, not a target: the task ends on the file's state, not on a number.

 **Boundary context — do not confuse this with the gate work.**
 - `T021`, `T021.001` and `T021.002` are `accepted-local`; **do not reopen them**, re-derive their clauses, or use the drain to add gate repairs. The `S21`/`S22` recomposition and the closing Yield Red check remain their own acts.
 - `MS-001`'s Yield Phase is `SEMANTIC-APPROVE`; this drain is the precondition the following base sync needs, not a second review.
 - `resolved` is withheld pending this drain + the boundary base sync + the closing Yield Red check. Do not publish phase acceptance, the `Delivered` entry or parent completion from this task.
 - Source B (deferred issues/features) is its own gate pass; this task handles Source A only.

 **Verification profile** (`proportional-verification`, recorded by plan-adhoc-task before dispatch).
 - Outcome and settled authority: every open in-scope Correctness line closed with evidence; every other line `[x]` with reason or `→ routed:`; `## Refactoring` open.
 - Risk: **uncertain** at plan time — the surviving in-scope set is not enumerated until triage runs. Concrete wrong outcome: a `still true` closed without evidence, a `reach` verdict on a case that is actually reached, or a silently dropped line — each hides a real defect behind a "drained" section. Containment: one line of reason per closure, and Yield reviews the closures with the fixes.
 - Recovery: **easy** — each triage closure is one line in `quality-backlog.md`; each surviving fix is a bounded diff, independently revertible, with no data or external effect.
 - Uncertainty: which lines are still true on current bytes; whether any Refactoring line can change behaviour; whether any Correctness line needs a user decision. Triage is the bounded assessment that resolves the first two; the third routes to Source B rather than being guessed.
 - Scope: `quality-backlog.md` in full (54 open lines at this write) plus only the code the surviving in-scope lines name.
 - Validation: observable success is the file's own state (both headings, every line `[x]` or `→ routed:`, Refactoring open) plus the full hermetic suite green on the wave's final bytes (baseline **511 @ 96.71%**; affected scope **37 passed** on the run-49 repaired bytes). Realistic wrong witness: every line marked `[x]` with "still true" but no evidence locator; a line silently dropped; a behaviour-changing line left under `## Refactoring`; or an accepted task reopened.
 - Route per gate: **parallel** — settled intent (the contract names the procedure), bounded uncertainty (the surviving set), contained consequences (per-line fixes), easy recovery. Any surviving fix whose surface is high-consequence (security, compatibility, ownership, migration) expands **only that affected gate** to full before acceptance. Triage itself is compact: it writes reasons and `[x]` markers and reads current bytes.
 - Evidence reused and budget: the per-line `QB:` evidence and the session's measured figures are the baseline, but each line is re-derived rather than trusted. Budget: one bounded triage read per open line; no blind re-dispatch and no re-extraction except where a surviving fix creates new surface.

 **Assumption impact: `required:new-surface`.** The surviving in-scope fixes' surfaces are unknown until triage names them, and no seam id exists for them yet — so `scoped:` is invalid here, and `skip:verification-only` / `skip:session-only` / `lite:small` do not describe a drain that may change production code. Once triage names a surviving fix, Forward Collect extracts that surface's claims and the normal assumption gate runs before the fix. Triage closures that write no code add no surface.

 **Execution order.** After `T021`/`T021.001`/`T021.002` are `accepted-local`; before `MS-001`'s boundary base sync and the closing Yield Red check. No `--before` target exists — those are milestone operations, not task ids — so the ordering lives here rather than in a `Blocks:` column. Do not reorder this row ahead of the `T021.*` rows: it blocks nothing retroactively.

 **Protected invariants.** This ad hoc task is **not** permission to override a protected or approved invariant. Surviving fixes must respect `PI:T002/dump_format/documented_format_is_retained`, `PI:T002/dump_format/default_format_is_documented`, `PI:T002/dump_format/format_is_per_artifact`, `PI:T006/upstream_tls/verification_on_unless_explicitly_disabled`, `PI:T006/upstream_tls/optout_is_process_wide`, the `OI:` entries in plan-summary §Approved Override Register, and `G-008`(a)'s default-run byte-identity. If a Correctness line's only closure would contradict one of them, record the contradiction here and stop for user review — do not implement through it.

</details>


## Trimmed 2026-10-18 (`105-forward-qa`) — superseded `Last Updated` paragraph moved verbatim

- Last Updated: **`T022.001` closure applied at [103](iterations/103-assumption-decide.md)** (2026-10-18) — `quality-backlog.md:26` is `[x]` with its DRAIN evidence, the four count statements read **8 routed / 11 closed** (7 earlier, 1 fixed, 3 accepted risks), and only one test-comment word moved; **`T022.001` accepted-local at [104-01-yield](iterations/104-01-yield.md)** (2026-10-18) — every done-when clause re-derived on `HEAD 73f52cd`, focused module 14 passed, two defer-tier record-accuracy rows, `resolved`/phase acceptance/`Delivered` still with the closing Yield Red check; **`MS-001` boundary base sync `BASE-SYNC: unchanged` at [102-01](iterations/102-01-forward.md)** — target `main @ 72e3a1c…` already an ancestor of HEAD `85ca1ba`, no delta; **run-49 gate repair accepted-local at [100-yield](iterations/100-yield.md)** (2026-10-18) — `T021.001`/`T021.002` meet their done-when clauses in test-side bytes only; the `S21`/`S22` recomposition and the closing Yield Red check own `resolved`, phase acceptance, the `Delivered` entry and parent completion. Context trimmed to this cap; every moved block is verbatim in [handoff-archive.md](handoff-archive.md) §Trimmed 2026-10-18 (`100-yield`, `103-assumption-decide`). **`T022.001` re-verified by Forward at [104-01](iterations/104-01-forward.md)** (2026-10-18): every done-when clause re-derived on `HEAD b84b224`, focused module 14 passed, scan clean, no new bytes.

## Trimmed 2026-10-18 (`106-yield-phase`) — superseded `QAP-MS001` re-proof paragraph moved verbatim

- **✅ `QAP-MS001` RE-PROVED (2026-10-17, [105-forward-qa](iterations/105-forward-qa.md))** — the five tracked QA/harness edits that made the tree dirty (`tests/qa/*`, `qa-project` §Upstream, alias `qa-gate-model` → `sloppus`) were validated against the approved contracts and committed with the gate, so the gate is reproducible from `HEAD`; the 50 untracked run-49 session artifacts remain earlier passes' bytes, left untouched.


## Trimmed 2026-10-18 (`107-assumption-decide`) — superseded `Last Updated` paragraph moved verbatim

- Last Updated: **`QAP-MS001` re-proved at [105](iterations/105-forward-qa.md)** (2026-10-17) — the five uncommitted QA/harness edits (gate model alias `qa-gate-model` → `sloppus`, declared in `qa-project` §Upstream) were validated and committed with the gate; direct probes measured `sloppus` **200** / `qa-gate-model` **404**, `QAP-MS001` **16/16**, `QAP-MS002` 15/15, `QAP-MS004` 6/6, hermetic **513 @ 96.71%**, zero production bytes. The superseded paragraph is preserved verbatim in [handoff-archive.md](handoff-archive.md) §Trimmed 2026-10-18 (`105-forward-qa`). **`T022.001` accepted-local at [104-01-yield](iterations/104-01-yield.md)** (2026-10-18) — every done-when clause re-derived on `HEAD 73f52cd`, focused module 14 passed; `resolved`, phase acceptance and the `Delivered` entry stay with the closing Yield Red check.

## Trimmed 2026-10-18 (`107-assumption-decide`) — superseded `## Current Status` bullets moved verbatim

- **✅ `MS-001` BOUNDARY BASE SYNC — `BASE-SYNC: unchanged` (forward, 2026-10-18, [102-01](iterations/102-01-forward.md)).**
  `main @ 72e3a1c…` (no movement; = `origin/main` = the prompt's id) is already an ancestor of HEAD `85ca1ba` — **0 behind / 265 ahead**; no merge, no conflicts, no re-threading, no prework packet; checkpoint = session files only.
- **T022 STEP-6A DRAIN — CLOSED (2026-10-18, [101-01](iterations/101-01-forward.md), [102-01](iterations/102-01-yield.md), [103](iterations/103-assumption-decide.md))** — `quality-backlog.md`
  carries `## Correctness` / `## Refactoring`; the fixed line is `[x]` at `quality-backlog.md:26` with its DRAIN evidence, and the file reads **8 open (every one routed) / 11 closed** (7 earlier, 1 fixed, 3 accepted risks) — corrected on all four surfaces from the local Yield's 9/10 rejection (`DC-T022-Y-1-r1`).

## Trimmed 2026-10-18 (`107-assumption-decide`) — consumed `T022.001` blueprint moved verbatim

<details>
<summary>T022.001 Planned Task Context — close the T022 drain's own fixed Correctness line and correct its counts</summary>

 **Finding.** [102-01-yield](iterations/102-01-yield.md) §Findings P1: `quality-backlog.md:26`
 (`QB:test-stream-relay/write-position-injection`) is still `- [ ]` although its DRAIN note says **FIXED in
 this wave** and [101-01-forward](iterations/101-01-forward.md) §What changed row 52 marks it `[x]`. The state
 line's "9 open, every one routed … 10 closed — 6 earlier, 1 fixed, 3 accepted risks" is false: it is 9 open
 (8 routed + that line) / 10 closed, 7 earlier.

 **Required outcome.** `:26` is `[x]` with its DRAIN evidence; `:14` states **8 open, every one routed … 11
 closed — 7 earlier, 1 fixed, 3 accepted risks**; the same figures appear in `101-01-forward.md:32`/`:80`,
 `plan-summary.md`'s `T022` row and `handoff-context.md` §Current Status; `tests/test_stream_relay_commit.py:51`
 changes "lands" → "is swallowed".

 **Authority and scope.** `T022`'s done-when ("every open in-scope Correctness line is `[x]` with evidence"); the
 DRAIN note; Forward's row 52; `session-close.md` step 12 Source A. Deterministic closure `DC-T022-Y-1-r1`,
 no user judgment. Scope: `quality-backlog.md`, `101-01-forward.md`, `plan-summary.md`, `handoff-context.md`,
 `102-01-yield.md` (record only), `tests/test_stream_relay_commit.py:51` (comment only). No production byte.

 **Done when.** `sed -n '26p' quality-backlog.md` starts `- [x]`; the four surfaces state 8 routed / 11
 closed (7/1/3); a paragraph-aware count gives 8 open / 11 closed with 8 routed; `:51` says "is swallowed";
 focused module 14 passed; `codify scan path tests/test_stream_relay_commit.py --git --detailed` clean.

 **Verification profile.** Parallel — settled intent, bounded uncertainty (which count surfaces are stale),
 contained consequences (session text only), easy recovery. Wrong witness: a stale count or an unbacked flip.

 **Execution order.** After `T022`'s drain ([101-01](iterations/101-01-forward.md)) and rejection
 ([102-01](iterations/102-01-yield.md)); before the run-49 `S21`/`S22` recomposition, closing Yield Red and
 `phase-boundary-finalize`. Does not reopen `T021.*`.

 Assumption impact: `lite:small` — one test-comment word plus session bookkeeping; no new surface, no seam.

</details>

## Trimmed 2026-10-18 (`107-assumption-decide`) — `DC-T022-Y-1-r1` moved verbatim (live row compacted to a pointer)

### DC-T022-Y-1-r1
Finding: T022-Y-1 (T022 local Yield, iterations/102-01-yield.md §Findings P1)
Evidence: iterations/102-01-yield.md
Evidence fingerprint: unavailable
Repository state: git-commit:19971fb6fd5a02a289c52986d7271ec0915e530a0
Selected closure: Flip `quality-backlog.md:26` to `[x]` with its DRAIN evidence; restate the Correctness state as 8 open (all routed) / 11 closed (7 earlier, 1 fixed in this wave, 3 accepted risks) at `quality-backlog.md:14`, `101-01-forward.md:32`/`:80`, `plan-summary.md`'s `T022` row and `handoff-context.md` §Current Status; change "lands" to "is swallowed" at `tests/test_stream_relay_commit.py:51`.
Closure basis: `T022`'s planned done-when ("every open in-scope Correctness line is `[x]` with its evidence"); the file's own DRAIN note says FIXED; `101-01-forward.md` §What changed row 52 marks it `[x]`; `session-close.md` step 12 Source A. The line is fixed, so only its checkbox and the arithmetic derived from it can be true.
Rejected candidates:
- Leave `quality-backlog.md:26` `[ ]` and record it as routed: inadmissible - the line is fixed, not routed; its destination would be a lie and `T022`'s done-when stays unmet.
- Revert the witness: inadmissible - destroys landed, fail-first-proven work.
Risk shape: unchanged - session-file state and one test comment word; no production, test-logic, config, registry or data effect.
Deferred additive work: none.
Task: T022.001 step 1

## Trimmed 2026-10-18 (`108-forward-qa`) — superseded `Last Updated` paragraph moved verbatim

**`S21` CE-2 residual routed `absorbed` at [107](iterations/107-assumption-decide.md)** (2026-10-18) — the documented-format gate's expected side reads the `self-contained` artifact through `read_text`'s universal-newline mode, so a CRLF-vs-LF rendering pair gives `expected=[]` / `observed=['-line1\n', '+line1\n']`; the residual is **latent** (no `\r` byte under `tests/qa/`), `T021.003` is the gate-only repair, and `S21`'s single post-repair recomposition allowance is consumed. The superseded paragraph is preserved verbatim in [handoff-archive.md](handoff-archive.md) §Trimmed 2026-10-18 (`107-assumption-decide`).


## Trimmed 2026-10-18 (`109-yield`) — superseded `Last Updated` paragraph moved verbatim

- Last Updated: **`T021.003` gate repair authored at [108](iterations/108-forward-qa.md)** (2026-10-18) — the documented-format gate reads the observed `turn_NNN.diff` and the `self-contained` artifact's renderings in raw text mode (`read_text(..., newline="")`; the shared default is unchanged), so the expected diff is computed over the same line endings production diffed before the write. One focused artifact round-trip regression is red on the pre-repair universal read (`expected=[]` / `observed=['-line1\n', '+line1\n']`) and green after. `QAP-MS001` re-proved on `HEAD 65c298c`: **17/17**, `QAP-MS002` 15/15, `QAP-MS004` 6/6, hermetic **513 @ 96.71%**, zero production bytes. The superseded paragraph is preserved verbatim in [handoff-archive.md](handoff-archive.md) §Trimmed 2026-10-18 (`108-forward-qa`).  **`QAP-MS001` re-proved at [105](iterations/105-forward-qa.md)** (2026-10-17) — the five uncommitted QA/harness edits (gate model alias `qa-gate-model` → `sloppus`, declared in `qa-project` §Upstream) were validated and committed with the gate; direct probes measured `sloppus` **200** / `qa-gate-model` **404**, `QAP-MS001` **16/16**, `QAP-MS002` 15/15, `QAP-MS004` 6/6, hermetic **513 @ 96.71%**, zero production bytes. **`T022.001` accepted-local at [104-01-yield](iterations/104-01-yield.md)** (2026-10-18) — every done-when clause re-derived on `HEAD 73f52cd`, focused module 14 passed; `resolved`, phase acceptance and the `Delivered` entry stay with the closing Yield Red check.

## Trimmed 2026-10-18 (`YR-MS001-r3` closing check) — superseded `Last Updated` and `Current Status` NEXT bullet moved verbatim

- Last Updated: **`T021.003` local Yield ACCEPT at [109](iterations/109-yield.md)** (2026-10-18) — the documented-format gate reads the observed `turn_NNN.diff` and the `self-contained` artifact's renderings in raw text mode, so the expected diff is computed over the same line endings production diffed before the write; the focused artifact round-trip regression is red on the pre-repair read (`expected=[]` / `observed=['-line1\n', '+line1\n']`) and green after. Re-proved on `HEAD eea5086`: `QAP-MS001` **17/17**, `QAP-MS002` 15/15, `QAP-MS004` 6/6, hermetic **513 @ 96.71%**, zero production bytes; both `T021` mutation replays still red. The superseded paragraph is preserved verbatim in [handoff-archive.md](handoff-archive.md) §Trimmed 2026-10-18 (`109-yield`).
- **NEXT — the capped post-fix `S21` recomposition, then the closing Yield Red check and `phase-boundary-finalize`.** `T021.003` is `accepted-local` at [109](iterations/109-yield.md) (`S21` CE-2 residual repaired: raw reads at both diff-comparison surfaces, `QAP-MS001` 17/17, `QAP-MS002` 15/15, `QAP-MS004` 6/6, hermetic 513 @ 96.71% on `HEAD eea5086`; both `T021` mutation replays red). `T022.001` is `accepted-local` at [104-01-yield](iterations/104-01-yield.md) (`DC-T022-Y-1-r1`: 8 routed / 11 closed re-derived on `HEAD 73f52cd`, focused module 14 passed); `resolved`, phase acceptance and the `Delivered` entry stay with the closing check.

## Trimmed 2026-10-18 (`YR-MS001-r4` closing check) — completed `T021.003` Planned Task Context moved verbatim

<details>
<summary>T021.003 Planned Task Context — preserve the renderings' line endings across the documented-format gate's artifact read (S21 CE-2 residual)</summary>

- Owning route: **`forward-qa`** — **test bytes only** (`tests/qa/harness.py`, `tests/qa/test_qap_ms001.py`); no production, config, `invariants/` or `decisions.md` byte. Nested under `T021`; the run-49 wave is `accepted-local` and is not reopened.
- Assumption finding: **seam-S21** CE-2 residual
- Evidence: `assumptions/compositions/S21.md` (`sha256:7f4c55b7ec6ef79a210f6697de4b361eb0eaaebf994a941c42775c774d4fb5fa`)
- Disposition: **absorbed** — `DC-S21-r2`
- Status: **accepted-local** ([109](iterations/109-yield.md))
- Forward QA: **authored** at [108-forward-qa](iterations/108-forward-qa.md) — `read_text(path, newline="")` at both diff-comparison surfaces plus one focused artifact round-trip regression; `QAP-MS001` 17/17, `QAP-MS002` 15/15, `QAP-MS004` 6/6, hermetic 513 @ 96.71%; local Yield pending.
- Counterexample: turn N−1 user content `"line1\r\nline2"`, turn N `"line1\nline2"` — production's `_unified_diff` + `"".join` writes `-line1\r\n`/`+line1\n`; the gate reads `turn_NNN.diff` and the `self-contained` artifact's renderings through `read_text` (`tests/qa/harness.py:150-154`, universal newlines), so both renderings arrive LF and `expected=[]` while `observed=['-line1\n', '+line1\n']`; `assert observed == expected` fails at `tests/qa/test_qap_ms001.py:280`. Reach: **latent** — no `\r` byte exists anywhere under `tests/qa/`; the documented-format bodies are `turn_body` markers, and the only CRLF renderings are the in-memory CE-2 sensitivity pair, which never reaches `read_text` or `assert_documented_contents`. The same read-mode asymmetry is a false-pass surface for a writer that normalizes line endings on a mixed change.
- Required behavior: `assert_documented_contents` must read the observed `turn_NNN.diff` and the `self-contained` artifact's renderings in raw text mode (`newline=""` or bytes), so the expected diff is computed over the same renderings production diffed before the write; a correct writer must not be rejected on CE-2, and a writer that drops a line-ending-only change on a changed line must not pass.
- Done when: a focused regression round-trips a CRLF rendering through the writer's `"\n\n".join(records) + "\n"` artifact shape, reads it with the gate's own reader, extracts with `rendering_section`, and asserts the recovered renderings are the raw ones and `changed_lines == diff_changed_lines` of the writer's own `"".join(unified_diff(...))` — red on the pre-repair `read_text` (`expected=[]`, `observed=['-line1\n', '+line1\n']`), green after; the dropped-line-ending writer arm rejected; the CE-1/CE-2/CE-3 in-memory regressions green; the two `T021` mutation replays still red; unmutated `QAP-MS001` 16/16, `QAP-MS002` 15/15, `QAP-MS004` 6/6, hermetic unchanged; `git diff --name-only` contains zero production, config, plan, contract or registry bytes.
- Yield replay: the exact CE-2 pair through a written `self-contained` artifact → red before the reader fix, green after; the writer that emits `-line1\n` where production emitted `-line1\r\n` → rejected; revert every mutation byte-exact.
- Assumption impact: **scoped:S21**
- Invariant cross-reference: **Neutral** against `active`/`wip` (zero exist; every registry `Invariant:` row is `planned`). The repair serves the planned `PI:T002/dump_format/documented_format_is_retained` and its witness `TC-108`; no duplicate candidate, no invariant authority.
- Invariant candidate: none.
- Closure ledger: `DC-S21-r2`
- Recomposition allowance: **consumed by this pass** — a further non-`SAFE` `S21` composition is capped by rule; its remainder goes to the user in one message.

</details>


## Trimmed 2026-10-18 (`116-01-forward`) — superseded `## Current Status` moved verbatim


- **Registered 2026-10-18 (registrar batch) — [`FK-011`](decisions.md#fk-011) · preserved-thinking serve default, RESOLVED
  2026-10-18, Option A.** The user's verbatim answer: *"A shit why you even created this fork?"* — `A` selects
  Option A; the remainder is preserved as an objection to the Fork's admission and as a routing-quality note for
  `assumption-decide`, not as a change to the answer. The serve path keeps
  `PRESERVED_THINKING=${PRESERVED_THINKING:-1}` (`serve_buster_ripper.sh:30`) and line 42's `== "1"` append, so
  `buster_ripper/routes/messages.py:46-47` keeps injecting `enable_thinking=true, clear_thinking=false`; `README.md`'s
  serve-script section must document `--preserved-thinking` and the variable with its `1` default, at the standard
  `--skip-tls-verify` (`README.md:89-92`) and `--dump-format` (`README.md:169-194`) already meet. Projection:
  **T023 · document `--preserved-thinking` and `PRESERVED_THINKING` in `README.md`'s serve-script section**
  (`pending`, `lite:small`, verification profile `parallel`) — plan §Implementation Plan Summary + §Planned Task Context
  below. Basis re-verified at registration on `HEAD fd6fcd6` (working tree clean): script sha256
  `c58256fbec793c520720c9e86fee8740653939275b508f50b212ab60bd0908af` and `README.md` still at **zero**
  `preserved.thinking`/`PRESERVED_THINKING` hits — **the docs change is not yet implemented**; no code, script or
  test byte moves.
- **DONE — `MS-001` boundary closed; `BOUNDARY-FINALIZED` at [110-01](iterations/110-01-forward.md).** One review-preserving checkpoint on `e46d58d`; worktree clean before and after; delta is this report + `plan-summary.md` + this file, all under `.sessions/`. Zero production, test, config, plan-contract or registry bytes moved after the reviewed object (`git diff --name-only eea5086..HEAD` → 0 non-session files; tree ids for `buster_ripper`/`tests`/`README.md`/`pyproject.toml`/`serve_buster_ripper.sh`/`.agents`/`invariants` identical).
- **Closing Yield Red `YR-MS001-r4` CLEAN** at [095](iterations/095-yield-red.md) §Adjudication (Review object `eea5086`, `HEAD 1575ab6`): `resolved` for `T015`–`T022`, `T021.001`–`T021.003`, `T022.001`, `MS-001`; `Delivered` written; `YR-P021` DOES-NOT-HOLD in its universal-negative clause only, `YR-P022`/`YR-P023`/`YR-P024` HOLD.
- **Deviation reported, not hidden:** the finalize mode requires the CLEAN report's Review object to equal the pre-checkpoint `HEAD`; it does not (`eea5086` reviewed / `1575ab6` report `HEAD` / `e46d58d` pre-checkpoint `HEAD`, the last being the dispatcher's own r4-artifact commit). No history rewrite was performed; substance is proven by tree identity, and the orchestrator may re-dispatch with `Review object: e46d58d` if it wants the literal shape.
- `MS-001` `SEMANTIC-APPROVE` at [106](iterations/106-yield-phase.md) and the `T021`/`T022` acceptances at [100](iterations/100-yield.md)/[104-01](iterations/104-01-yield.md) stand; nothing is reopened.
- **✅ `MS-001` BOUNDARY `SEMANTIC-APPROVE` at [106-yield-phase](iterations/106-yield-phase.md) (2026-10-18, `HEAD 0ac2a65`)** — `QAP-MS001` re-executed first-party: **16/16** in 11.82s, four `VP-*` `pass`, `TC-100` `write_ops=1`, `TC-108` `2N=6`/`N+1=4`, hermetic **513 @ 96.71%**; Checks 1–7 pass, zero findings. `resolved` and `Delivered` are published by the closing check ([095](iterations/095-yield-red.md) §Adjudication). The superseded `QAP-MS001` re-proof paragraph is preserved verbatim in [handoff-archive.md](handoff-archive.md) §Trimmed 2026-10-18 (`106-yield-phase`).
- **⚑ Gate upstream `http://hyperipper.local:30000`** (user correction 2026-10-17; `:30001` is a peer Buster Ripper proxy). `README.md`'s `:30001` occurrences and the app's `--port` default stay untouched.

- **NO OPEN FORK — [`FK-011`](decisions.md#fk-011) · preserved-thinking serve default RESOLVED Option A at [115](iterations/115-interview-reconcile.md), registered 2026-10-18** (2026-10-18): the serve path keeps its default-on posture and the posture becomes a documented, recorded choice. User words: "A shit why you even created this fork?" — the objection to the Fork's admission is preserved verbatim and recorded as a routing-quality note for `assumption-decide`, not as a change to the answer. **The adhoc docs task is registered as `T023 · document `--preserved-thinking` and `PRESERVED_THINKING` in `README.md`'s serve-script section` (`pending`, `lite:small`, verification profile `parallel`) and is dispatchable now** — no further user decision is owed. No milestone moves; `MS-001` stays resolved.

### Registrar batch — 2026-10-18 ([FK-011](decisions.md#fk-011) · preserved-thinking serve default)
- Sources: 1 resolved FK (`FK-011 · preserved-thinking serve default`, `Resolution` + `Impact` complete; wave 1,
  order 1 of 1; user answer recorded 2026-10-18T15:40:52Z at [115](iterations/115-interview-reconcile.md)).
  This is the settled-impacts flush — the session's only alignment wave is complete and no other FK is open.
- Destinations landed: `decisions.md` §`FK-011` §`Impact` (Plan → landed; Registry/Roadmap confirmed none);
  `plan-summary.md` §Session Metadata, §Progress Overview, §Current Phase, §Open Questions item 11,
  §Implementation Plan Summary (new row **T023 · document `--preserved-thinking` and `PRESERVED_THINKING` in `README.md`'s
  serve-script section** + its context pointer); this file §Session Metadata, §Current Status, §Open Forks
  (pending-projection entry replaced), §Planned Task Context (the new task's block), and `chronicle.md`.
- Registry root `invariants/`: **untouched** — 0 new / 0 updated / 0 superseded / 0 upheld; no elevation and no index
  rebuild (`FK-011`'s Registry impact states none, and the promoted `CO:serve-script-env-to-argv/env-var-flag-crossing`
  record stays current because Option A moves no script byte). No planned registry destination is pending-finalization here.
- Registry health (pre-existing, not created or cured by this batch — no registry content changed): `codify knowledge check`
  at HEAD reports **5 diagnostics** — `error: curated-index-generated-section` ×2 (`invariants/index.md:24`
  `## Handle Index first`, `:43` `## Entity Index first`) and `warning: missing-version` ×3
  (`buster_ripper/dump/invariants.md:1`, `buster_ripper/server/invariants.md:1`, `index.md:1`). `git status` shows
  `invariants/` clean, so the state predates this batch; no duplicate or dangling handle, so nothing here blocks
  registration. The cure belongs to the registry owner.
- Roadmap: no row written (the source's `Roadmap: none`). **Observation for the roadmap owner, not authority:**
  `roadmap.md` §RM-005 lists `--preserved-thinking` among five CLI flags absent from `README.md`; `T023` documents
  that one, so RM-005's remaining set becomes four once the task lands. RM-005 stays deferred with its own trigger and is
  not edited here.
- Forward-readiness: **`T023` is dispatchable now** — docs-only, single file, settled contract, no assumption gate owed.
  `MS-001` stays `resolved`; no milestone reopens and no other task waits on this batch.


## Trimmed 2026-10-18 (`116-01-yield`) — consumed `T023` Planned Task Context moved verbatim

<details>
<summary>T023 Planned Task Context — document `--preserved-thinking` and `PRESERVED_THINKING` in `README.md`'s serve-script section</summary>

**Authority:** [`FK-011`](decisions.md#fk-011) · preserved-thinking serve default — resolved 2026-10-18, **Option A**
(keep the default on; make it a documented, recorded choice). This block carries the projection only; the
canonical decision is `decisions.md`, and the task creates no second record.

**Outcome:** `README.md`'s serve-script section gains `--preserved-thinking` and the `PRESERVED_THINKING` variable
with its `1` default, at the standard `--skip-tls-verify` (`README.md:89-92`) and `--dump-format`
(`README.md:169-194`) already meet. No code, script or test byte moves: `serve_buster_ripper.sh:30`
`PRESERVED_THINKING=${PRESERVED_THINKING:-1}` and `:42`'s `== "1"` append stand, and
`buster_ripper/routes/messages.py:46-47` keeps injecting `enable_thinking=true, clear_thinking=false`.

**Done when:**
1. The serve-script section names the flag and the variable, states preserved thinking is **on by default**
   (`enable_thinking=true`, `clear_thinking=false`), and shows how to turn it off
   (`PRESERVED_THINKING=0 bash serve_buster_ripper.sh`).
2. It names `FK-011 · preserved-thinking serve default` as the decision and creates no second record.
3. `git diff --stat` shows `README.md` only — `serve_buster_ripper.sh` stays at sha256 `c58256fb…` and
   `tests/test_serve_script_argv.py` is untouched.

**Validation:** read-back against `serve_buster_ripper.sh:30`/`:42` and `buster_ripper/main.py:107`, plus the
doc-standard comparison with the two existing entries; no test run is owed for a docs-only change. Wrong witness: a section
that presents the flag as opt-in, or omits the off-switch, fails clause 1.

**Milestone:** none — `MS-001` is resolved and stays resolved; this task moves no milestone verdict (same shape as
`T011` · serve-script disable form and `T022.001` · T022 drain bookkeeping).

**Assumption impact:** `lite:small` — bounded, single-file docs edit; no new surface and no seam touched.

**Verification profile:** `parallel`. Outcome and authority settled (`FK-011` Option A); risk **low** (a docs sentence could
misstate the default, contained by the read-back); recovery **easy** (revert one README paragraph); uncertainty **none**
(line 30/42 and `main.py:107` re-read at registration); scope `README.md`'s serve-script section only; validation is the
three done-when clauses plus the wrong witness; route per gate **compact** (no assumption checker, one bounded Forward Red
attempt, direct Yield read-back); evidence reused from `FK-011`'s `Resolution` and this batch; no checker budget owed.

**Execution order:** normal plan order — after registration; no other task waits on it.

</details>


## Trimmed 2026-10-18 (`117-yield-phase`) — superseded `T023` Current Status bullet moved verbatim

- **ACCEPTED and `resolved` — [`T023`](iterations/116-01-yield.md) · document `--preserved-thinking` and `PRESERVED_THINKING`**
  (2026-10-18): [`FK-011`](decisions.md#fk-011) Option A landed as an 11-line `README.md` §Serve script
  options addition — the `PRESERVED_THINKING=0` off-switch, the ON-by-default posture
  (`enable_thinking=true`, `clear_thinking=false` unless the client set them), and
  `FK-011 · preserved-thinking serve default` named as the recorded decision. All three done-when
  clauses and the wrong witness re-derived on the bytes; `git diff --stat` = `README.md` only;
  `serve_buster_ripper.sh` at `sha256:c58256fb…` and `tests/test_serve_script_argv.py` untouched;
  `uv run pytest tests/test_serve_script_argv.py -o addopts= -q` → 24 passed. **No reject-tier
  finding; one defer-tier consequence:** the README digest change (`fe80cb9a…` → `f34692cb…`) stales
  five registry records on their own `Invalidates-On` — `CL:README.md/documented-run-path`,
  `CL:README.md/collapsed-format-run-path`, `CL:README.md/skip-tls-verify-run-path`,
  `CL:buster_ripper/collapsed-artifact-ownership` and
  `CO:serve-script-env-to-argv/env-var-flag-crossing` — remedy is their own named re-trace, owned by
  the assumption-registry lane, `QB:assumption-registry/readme-digest-after-t023`; script bytes
  unchanged, so the substance is intact.

## Trimmed 2026-10-18 (`T024`)

<summary>T024 Planned Task Context — session-end Refactoring wave (`quality-backlog.md` §Refactoring)</summary>

**Authority — `run-codify` `references/session-close.md` step 12, session-completion branch, quoted:**

> **Refactoring waits for the session end.** At a cycle or phase boundary, leave its lines open.
> At session completion, after every task and phase gate and before the stability round, sort each
> open line by size:
> - **Small and mid** (one module, no interface change, tests already cover the code): done in one
>   Forward → Yield wave, an adhoc task (`plan-adhoc-task`) whose done-when is that every such line is
>   `[x]` with its commit. The stability round verifies it with everything else.
> - **Big** (several modules, an interface change, or tests needed first; unsure means big): a row in the
>   session `roadmap.md` (`Kind` `refactor` or `quality`, `Route` `refactor-codify` or `run-codify`), and the
>   line is marked `→ routed:`. First look in the session and project roadmaps for a row naming the same file
>   and function, and cite that row instead of adding a second.
> No user question; `Authority / safety` cites this rule. Do not dispatch `refactor-hunter` from this gate.

**Why this task exists.** `phase-boundary-finalize` at [118-01](iterations/118-01-forward.md) returned
`BOUNDARY-BLOCKED`: gate **G7** (`Source A — Refactoring drained`) FAILED, so `BOUNDARY-FINALIZED` was not
returned and no closing session-outcome entry was written. The session may be declared complete only when Source A
is fully drained. `T022` (step 6a) deliberately left the section open for this wave; this is that wave.

**Referent to drain.** `quality-backlog.md` §Refactoring (lines 96–244 at dispatch). Its own State line at `:98`
reads **"46 open"**; the section actually holds **48** open `- [ ]` lines (`T021.003` and `T023` added rows after
that line was written). Restating the count is part of the done-when.

**Disposition census measured on `5433970` — re-derive it as the first act; the dispatch's split does not reconcile
with a literal marker census.**

| Bucket | Lines |
|---|---|
| literal `→ routed:` | **10** — 120, 121, 122, 123, 125, 129, 130, 131, 132, 133 |
| prose `owning route:` with **no** literal marker | **2** — 141 (`roadmap.md` §RM-004's owner), 142 (`roadmap.md` §RM-001's owner / next backlog drain) |
| legacy `kept:` | **6** — 124, 126, 127, 128, 134, 135 |
| no terminal disposition | **30** — 136, 137, 143, 148–158 (11), 159–167 (9), 171, 201, 209, 221, 227, 237, 244 |
| **total** | **48** |

The dispatch quotes 11 routed / 6 kept / 31 undisposed; `118-01` §G7 lists 141–142 inside the 31. The literal census
gives 10 + 2 / 6 / 30. Settle the census on the bytes before sorting — the done-when counts lines, not the quoted split.

**Roadmap-first rule — cite, do not duplicate.** The session roadmap `roadmap.md` §Routings from the 2026-10-06 backlog
drain holds RM-001…RM-012; the promoted project chunk is
`.sessions/codify/roadmap/2026-10-18-message-dump-bulk-flush.md` (RM-001…RM-016, all `open`, all `promoted-onward`).
Rows that already name the same file and function, and must be **cited and extended rather than duplicated**:

- **RM-001** — `handle_messages` (`routes/messages.py:26`), `handle_chat` (`routes/chat.py:23`),
  `_make_chat_request` (`routes/chat.py:134`). It does **not** name `_non_stream_single_request` /
  `handle_streaming` (backlog 142, whose own remedy is "add both to §RM-001's enumeration") or
  `_stream_single_request` (backlog 160, whose own remedy is "§RM-001's restructure"). Extend RM-001's
  `Anchor`/`Source` and cite it; do not open a second route-handler row.
- **RM-002** — `dump.py` file size / module split (backlog 120, already routed).
- **RM-003** — embedder-safe flush format binding (backlog 23, closed as accepted risk; do not reopen).
- **RM-004** — dependency bounds, `pyproject.toml`. Backlog 141 (`multidict` direct import) names the same file and the
  same concern, but RM-004's row names only `aiohttp`; cite RM-004 and extend its anchor, do not add a second dependency row.
- **RM-005** — CLI-surface docs + `STATS_DB` dead surface + `--strip-images` help scope (Correctness line 24, not this section).
- **RM-006** — hand-rolled mutation matrices / `.pyc` staleness (backlog 125).
- **RM-007** — `tests/test_tls_posture.py` private-attribute pinning (backlog 129). Backlog 161
  (`tests/test_transport_parity.py` `_timeout` pinning) is the same class in a different file — decide whether RM-007's
  anchor extends or a second row is owed.
- **RM-008** — `utils.response_headers` content-coding posture (backlog 33/37, Correctness).
- **RM-009**…**RM-012** — no Refactoring line names them; do not force-fit.

**Order of work.**

1. Settle the census on the bytes and restate `quality-backlog.md:98` to the measured count.
2. Sort every open line by the rule's own test — one module, no interface change, tests already cover the code ⇒ small/mid;
   several modules, an interface change, or tests needed first ⇒ big; **unsure means big**.
3. **Small/mid** → drain in this wave, one commit per line or per coherent group, each line `[x]` citing its commit.
   The line's own `Remedy` / `Proposed remedy` sentence is the specification; if a line has no specified remedy
   (e.g. "consider splitting", "re-read before scheduling"), it is **big** — `unsure means big`.
4. **Big** → a complete `## Roadmap items` table row in `roadmap.md` (or an extension of the cited row above), then
   `→ routed:` on the line naming the destination chunk and its `RM-*` handle. A prose mention or bulk inventory is
   evidence, never a destination.
5. `kept:` is **not** a terminal disposition under the current contract: each of the 6 becomes `[x]` (drained) or `→ routed:`.
6. Re-derive each line's stated measurement before acting on it — many are years-old line counts and scan figures, and a
   stale premise can turn a "small" line into a design decision.

**Out of scope / prohibitions.**

- **No behavior change.** Every drained line is behavior-preserving by construction. A line that cannot be drained without
  changing behavior is `big`, not a licence to change behavior.
- **Do not dispatch `refactor-hunter`** from this gate.
- **No user question.** The Authority is step 12 itself.
- **Do not touch `## Correctness`.** It is drained by `T022`/`T022.001`; this wave owns `## Refactoring` only.
- **Do not reopen accepted work.** `MS-001` stays `resolved`; `T015`–`T023` keep their recorded states.
- **Protected invariants.** This task is **not** permission to override them: the six `PI:T002/dump_format/*` entries, the two
  `PI:T006/upstream_tls/*`, the `OI:` entries in `plan-summary.md` §Approved Override Register, and `G-008`(a)'s default-run
  byte-identity (`plan-summary.md:246-305`). A line whose only closure would contradict one is recorded as a contradiction and
  routed back, never implemented through.
- **Write surface.** `quality-backlog.md`, `roadmap.md`, `handoff-context.md`, plus only the files a drained line's own remedy
  names. A test-shape split may create a module; it may not delete a witness.

**Plan-table trim note.** `plan-summary.md:348`'s archive note still reads "the `T022`/`T023` window this table must keep"; a trim pass must extend it to `T022`/`T024` so this row is not archived mid-wave.

**Stability round.** The step-13 round ran at [117](iterations/117-yield-phase.md) on `191e676` — full unscoped suite
**551 passed** (513 hermetic @ 96.71% + `QAP-MS001` 17 + `QAP-MS002` 15 + `QAP-MS004` 6) against the live `:30000`.
Session head before this task is `5433970`; base is `main @ 72e3a1c`. That round ran **before** this wave, so if the wave moves
any production or test byte, the unscoped suite must be **re-run on the resulting head** and recorded before
`phase-boundary-finalize` is re-dispatched with `Close: session`. A session-artifact-only wave records the empty
production/test diff instead.

**Verification profile — `parallel`.** Outcome and settled authority: step 12's session-completion branch; no user decision is
open. Risk: **low** — the concrete wrong outcome is a "behavior-preserving" split that deletes an assertion, a docstring-cited
witness, or a lint-flagged import the suite never exercises, while the suite still passes; exposure is every test module the
wave touches; containment is one commit per coherent group plus the step-13 re-run on the resulting head. Recovery: **easy** —
`git revert` of the wave's commits to `5433970`; no external effects; rework is re-opening the affected line. Uncertainty:
whether each line's recorded measurement still holds on current bytes, and whether an RM row already covers the line. Scope:
`quality-backlog.md` §Refactoring, `roadmap.md` §Routings + §Promotion Register, the promoted chunk, and every file a drained
line names. Validation: the done-when plus the re-run suite; the realistic wrong witness is a line marked `[x]` with a commit that
does not touch the file it names, or a `→ routed:` whose destination row omits the line's `QB:*` handle. Route per gate:
**parallel** — settled intent, bounded per-line uncertainty, contained consequences, easy recovery; assumptions are scoped
alongside Forward and reconciled at Yield. Evidence reused: [118-01](iterations/118-01-forward.md) §G7, the lines' own recorded
evidence, the existing RM rows. Budget: no new blind checkers; Yield re-derives each drained line's bytes and the census.

Assumption impact: `required:new-surface`.

Execution order: normal plan order — after `T023`, before the re-dispatched `phase-boundary-finalize`.

</details>

## Trimmed 2026-10-18 (`119-01-yield`)

- **STABLE — step-13 stability round GREEN on the final head `191e676`** (2026-10-18): the full,
  unscoped suite passes on one clean tree against the live upstream `:30000` — hermetic **513 passed @
  96.71%**, `QAP-MS001` **17/17**, `QAP-MS002` **15/15**, `QAP-MS004` **6/6** = **551 passed,
  0 failed**; base `main @ 72e3a1c` is an ancestor (0 behind / 291 ahead). Zero findings; **no scenario
  carries `Required at: session-close`**, so none was re-authored. Record: [117](iterations/117-yield-phase.md).
  `T023`'s acceptance and its one defer-tier registry consequence are unchanged (archived below).
- **`MS-001` BOUNDARY CLOSED and `BOUNDARY-FINALIZED` at [110-01](iterations/110-01-forward.md)** — closing
  Yield Red `YR-MS001-r4` CLEAN at [095](iterations/095-yield-red.md) §Adjudication; `T015`–`T022`,
  `T021.001`–`T021.003`, `T022.001` and `MS-001` are `resolved`; no parent completion is owed.
  `S21`'s delta recomposition is capped and deferred with its named trigger; `S22`/`S23` are `SAFE`
  with record-only re-traces owed.

## Trimmed 2026-10-18 (`120-assumption-decide`) — `DC-S21-r1` / `DC-S21-r2` moved verbatim (live rows compacted to pointers)

### DC-S21-r1
Finding: S21
Evidence: assumptions/compositions/S21.md
Evidence fingerprint: sha256:bdfa4a576c1e7681af18155a0c84835b42cb88bbd979eba8902f3cfc677c1282
Repository state: git-commit:702c5b8769fb4ae1411453917b7ca68d8bee82cf
Selected closure: Repair the gate's expected-vs-observed changed-line comparison in `tests/qa/test_qap_ms001.py` so it uses production's line identity (`splitlines(keepends=True)` on both sides), recovers the content line glued to the control lines, and extracts the renderings without `.strip()` (removing only the record-separator newlines); add one focused regression per CE.
Closure basis: `T021`'s approved outcome ("each `turn_NNN.diff` is computed over `_render_body` output of consecutive turns") and `PI:T002/dump_format/documented_format_is_retained`'s witness `TC-108`; the gate must observe the contract it already has, and a known false-pass surface in the boundary's only content witness cannot be accepted.
Rejected candidates:
- Fix production's `_unified_diff`/`_write_diff_turn` join and `lineterm` pairing: out of lane — `T021` is test-bytes-only and the prompt routes gate repairs, not production work; the malformed control-line join is recorded as a published-contract gap for the documented writer's owner.
- Mirror the join only (`splitlines(keepends=True)` + `lineterm=""` + `"".join`) without recovering the glued content line: fixes CE-1's false failure but leaves a writer that drops the first changed line passing, so it preserves part of the verified defect.
- Normalize line endings or strip trailing whitespace on both sides: erases the CE-2/CE-3 drifts the gate exists to catch, so it is a false pass and inadmissible.
Risk shape: unchanged - test-only bytes in one gate module; no production, config, registry or data effect.
Deferred additive work: QB:assumption-registry/s21-diff-text-identity (collector lane; trigger = the next `S21` composition or promotion) - the composition's N5 malformed-control-line gap and N8 no-hunk precondition are recorded, not repaired here.
Task: T021.001 step 1

### DC-S21-r2
Finding: S21 CE-2 residual (assumptions/compositions/S21.md, 2026-10-18T02:55:16Z)
Evidence: assumptions/compositions/S21.md
Evidence fingerprint: sha256:7f4c55b7ec6ef79a210f6697de4b361eb0eaaebf994a941c42775c774d4fb5fa
Repository state: git-commit:14dcec8972ce12e8ec8528f8c532801dcf71eb5e
Selected closure: Read the observed `turn_NNN.diff` and the `self-contained` artifact's renderings in raw text mode (`read_text(path, newline="")` or bytes) in `assert_documented_contents`, so the expected diff is computed over the same renderings production diffed before the write; add one focused artifact round-trip regression.
Closure basis: `T021.001`'s required behavior ("A correct writer must not be rejected on CE-1/CE-2"), `PI:T002/dump_format/documented_format_is_retained` and its witness `TC-108`, and `DC-S21-r1`'s selected closure ("uses production's line identity") which the artifact read still defeats. One admissible remediation remains: the read-mode change; normalizing line endings on both sides erases the drift and is inadmissible, and production is out of lane.
Rejected candidates:
- Change the shared `read_text` default to `newline=""`: widens the blast radius to MS-002/MS-004's JSON, index and nonce reads with no correctness gain; a parameter at the diff-comparison call sites joins the two surfaces that drifted without adding a third.
- Leave the expected side normalized and record a backlog row: preserves the verified oracle defect on a legitimate content shape and knowingly leaves `T021.001`'s required behavior unmet on the artifact path.
- Normalize line endings on both sides or strip whitespace: already rejected by `DC-S21-r1`; it erases the drift the gate exists to catch.
Risk shape: unchanged - test bytes in one gate module and one harness helper; no production, config, plan, contract, registry or data effect.
Deferred additive work: collector lane - re-extract `open`'s universal-newline default as a `lang:python` atomic proposition before `S21` is promoted (same disposition as `QB:assumption-registry/s22-ambient-proxy-premise`); trigger = the next `S21` composition or promotion. No task waits on it.
Task: T021.003 step 1

## Trimmed 2026-10-18 (`120-assumption-decide`) — superseded `Last Updated` prefix and Current Status bullets moved verbatim

**Superseded `Last Updated` prefix** (the rest of the line, from `Prior:` onward, is unchanged):

- Last Updated: **`T024` · session-end Refactoring wave — product bytes IMPLEMENTED, records REJECTED at Yield** (2026-10-18, [119-01-yield](iterations/119-01-yield.md)) — census settled on the bytes (48 open: 10 literal `→ routed:`, 2 prose-only `owning route:`, 6 legacy `kept:`, 30 undisposed); 11 drained in commit `54001b6`, 37 routed to complete `## Roadmap items` rows (RM-001, RM-002, RM-004, RM-006, RM-007, RM-008, and new RM-017…RM-020); hermetic **513 passed @ 96.71%** plus `QAP-MS001` 17/17, `QAP-MS002` 15/15, `QAP-MS004` 6/6 = **551 passed** on the resulting head. Prior:

**Superseded `## Current Status` bullets:**

- **`T024` · session-end Refactoring wave — product bytes SOUND, records REJECTED at Yield** (2026-10-18,
  [119-01-yield](iterations/119-01-yield.md)): the drained code is behaviour-preserving and the suite is green,
  but three durable records this wave wrote are factually wrong — the backlog State line's marker claim, the handoff's
  census counts, and `:72`'s unclosed `<details>` that swallows every section after it (evidence and exact figures
  in the report). Product scope unchanged: 11 small/mid lines drained in `54001b6` and 37 routed to complete
  `## Roadmap items` rows (RM-001, RM-002, RM-004, RM-006, RM-007, RM-008, and new RM-017…RM-020).
  Next: Forward applies the three record corrections from [119-01-yield](iterations/119-01-yield.md) and re-returns
  for Yield; only then re-dispatch `phase-boundary-finalize` with `Close: session`, whose per-row
  `evidence/INDEX.md` rulings remain its own step 16.
- **`phase-boundary-finalize` BLOCKED — session-close gate incomplete** (2026-10-18, [118-01](iterations/118-01-forward.md)):
  the finalize guards passed (clean tree, Review object `5433970` = `HEAD`, base `main @ 72e3a1c` an ancestor,
  reviewed bytes unchanged since `191e676`), but **`BOUNDARY-FINALIZED` was not returned** because `quality-backlog.md`
  §Refactoring was undrained (the wave above is the repair) and `evidence/INDEX.md` has **20 of 38 rows with an
  empty Disposition**. Closing-check record: [118-closing-check](iterations/118-closing-check.md).
- **No open fork and no open question.** `T024` is implemented (Forward) and **rejected at Yield** ([119-01-yield](iterations/119-01-yield.md)); `FK-011` was the last fork and its downstream task `T023` is accepted and `resolved` at [116-01-yield](iterations/116-01-yield.md). The workflow-contract blocker stays with its owner ([040](iterations/040-decide.md)). The record consequences now live in RM-019 (`QB:assumption-registry/readme-digest-after-t023` among them) and block nothing.


## Trimmed 2026-10-18 (`119-02-forward`) — superseded `T024.001` Current Status bullet moved verbatim

- **`T024.001` · `T024` record corrections ROUTED — no user question** (2026-10-18, [120](iterations/120-assumption-decide.md)):
  both durable-record findings from [119-01-yield](iterations/119-01-yield.md) are `absorbed` (`DC-T024-Y-1-r1`,
  `DC-T024-Y-3-r1`); the census counts are verified on the bytes (11 drained / 37 routed / 30 undisposed);
  the wave's product bytes are sound and green. Next: Forward applies the two record edits and re-returns for Yield,
  then `phase-boundary-finalize` with `Close: session` re-opens — its per-row `evidence/INDEX.md` rulings
  (20 of 38 undecided) remain its own step 16. `FK-011` stays resolved; no fork is open.

## Trimmed 2026-10-18 (`123-01-forward`) — superseded `Last Updated` line and `Current Status` block moved verbatim

- Last Updated: **`phase-boundary-finalize` BLOCKED at [122-01](iterations/122-01-forward.md) — no session-outcome entry, no completion claim** (2026-10-18) — the finalize's own guards pass: Review object `6736318` equals `HEAD`, base `main @ 72e3a1c` is an ancestor (0 behind / 307 ahead), the tree is clean, Source A is drained with every destination landed (**Correctness 8 open / 8 routed; Refactoring 37 open / 37 literal `→ routed:` / 0 prose tails**, 31 closed), Source B and C are clear, the roadmap is **21/21 promoted-onward and landed**, and the final code bytes are green at `793a5f0` (**551 passed**: 513 @ 96.71% + 17/15/6 gate). The session-close gate still fails step 16 / completion item 9: `evidence/INDEX.md` has **21 of 39 rows undecided** — eight regenerable rows can be `remove` with their commands retained, thirteen measurement transcripts need a per-row `keep`/`remove` ruling. One gate-scope deviation: the recorded local Codex read (`runs/121-closing-check-codex`) covers the closing delta `191e676..HEAD`, not the full session diff `72e3a1c..HEAD`; per-boundary reads exist at [057](iterations/057-codex-review.md), [079](iterations/079-codex-review.md), [088](iterations/088-codex-review.md). Record: [122-01](iterations/122-01-forward.md), [122-closing-check](iterations/122-closing-check.md). Prior: **`T024.001` · `T024` record corrections ACCEPTED at [119-02-yield](iterations/119-02-yield.md) — no user question** (2026-10-18; implemented at [119-02](iterations/119-02-forward.md)) — `## Refactoring` census on the post-edit bytes: **37 open / 37 literal `→ routed:` / 0 `routed onward to`**, every handle a complete roadmap row; `handoff-context.md` element census **1/1/1** and the `T024` pointer a plain line with all four following sections present; the stray `<details>` was already deleted at checkpoint `268`, so that half was verified rather than re-edited; session-record text only, zero production/test/config/`decisions.md`/`invariants/` bytes; the T024 wave's unscoped suite at `793a5f0` is green (**551 passed**: 513 @ 96.71% + 17/15/6 gate) and no code byte has changed since. Prior: **`T024.001` · `T024` record corrections ROUTED — `DC-T024-Y-1-r1` / `DC-T024-Y-3-r1`, no user question** (2026-10-18, [120](iterations/120-assumption-decide.md)) — both durable-record findings from [119-01-yield](iterations/119-01-yield.md) are `absorbed` into `T024.001`; re-measured on the bytes: **37 open, 9 literal `→ routed:` / 28 prose-only, 0 undisposed**, 37/37 naming a complete `## Roadmap items` row; **11** drained in `54001b6`, **30** closed overall; the handoff census counts (11 / 37 / 30) stand verified. Product bytes stay sound and green (**513 @ 96.71%** + `QAP-MS001` 17/17, `QAP-MS002` 15/15, `QAP-MS004` 6/6 = **551 passed**). Prior: `phase-boundary-finalize` BLOCKED at [118-01](iterations/118-01-forward.md) (2026-10-18) — the session-close gate is not clean, so **`BOUNDARY-FINALIZED` was not returned and no closing session-outcome entry was written**: `quality-backlog.md` §Refactoring holds **48 open lines**, 31 with no terminal disposition (the session-end Forward → Yield wave never ran), and `evidence/INDEX.md` has **20 of 38 rows undecided**, so step 16 cannot open. Review object = pre-checkpoint `HEAD` `5433970` (literal equality holds), base `main @ 72e3a1c` is an ancestor, tree clean, the step-13 suite stands green at `191e676` (**551 passed**) because no production or test byte moved since. Record: [118-01](iterations/118-01-forward.md), [118-closing-check](iterations/118-closing-check.md). Prior: **Step-13 stability round GREEN on the final head `191e676`** (2026-10-18) — full unscoped suite **551 passed** (`513` hermetic + `17`/`15`/`6` gate), base ancestor, tracked tree clean, zero findings; [117](iterations/117-yield-phase.md). Prior: **`T023` · preserved-thinking docs ACCEPTED and `resolved` at [116-01-yield](iterations/116-01-yield.md)** (2026-10-18) — `README.md`'s serve-script section documents `--preserved-thinking`, `PRESERVED_THINKING` (default `1`), the `PRESERVED_THINKING=0` off-switch and names `FK-011 · preserved-thinking serve default`; docs-only, `README.md` only, 24 serve-script tests green, zero reject-tier findings. One defer-tier consequence: the README digest change stales five registry records on their own `Invalidates-On` (`QB:assumption-registry/readme-digest-after-t023`). Prior: **`FK-011` · preserved-thinking serve default RESOLVED and registered at [115](iterations/115-interview-reconcile.md)** (2026-10-18) — wave 1, answered 2026-10-18T15:40:52Z: "A shit why you even created this fork?" → **Option A** (the serve path keeps preserved thinking on by default and the posture becomes a documented, recorded choice). Projection: **T023 · document `--preserved-thinking` and `PRESERVED_THINKING` in `README.md`'s serve-script section** (`pending`, `lite:small`, verification profile `parallel`); registry untouched, no roadmap entry, no milestone gate moved — `MS-001` stays resolved. See §Current Status §Registrar batch — 2026-10-18. Prior: **`BOUNDARY-FINALIZED` at [110-01](iterations/110-01-forward.md)** (2026-10-18) — closing Yield Red `YR-MS001-r4` CLEAN ([095](iterations/095-yield-red.md) §Adjudication, Review object `eea5086`, report `HEAD 1575ab6`); `T015`–`T022`, `T021.001`–`T021.003`, `T022.001` and `MS-001` are `resolved`, the `Delivered` entry is written, and no parent completion is owed. `S21`'s recomposition is capped/deferred with its named trigger; `S22`/`S23` are `SAFE` with record-only re-traces owed.

## Current Status

- **`phase-boundary-finalize` BLOCKED at [122-01](iterations/122-01-forward.md) — no session-outcome entry, no completion claim** (2026-10-18):
  the finalize's own guards pass (Review object `6736318` = `HEAD`, base `main @ 72e3a1c` an ancestor, tree clean, reviewed bytes
  unchanged), and the close gate is drained except step 16 / completion item 9 — `evidence/INDEX.md` has **21 of 39 rows
  undecided**. Ten are regenerable by a named command and can be `remove` with the command retained; eleven are measurement
  transcripts that need a per-row `keep`/`remove` ruling. One gate-scope deviation: the recorded Codex read covers the closing
  delta `191e676..HEAD`, not the full session diff `72e3a1c..HEAD`; per-boundary reads exist at [057](iterations/057-codex-review.md),
  [079](iterations/079-codex-review.md), [088](iterations/088-codex-review.md). Record:
  [122-01](iterations/122-01-forward.md), [122-closing-check](iterations/122-closing-check.md). Next: settle the evidence rows and
  the step-13 scope, then re-dispatch `phase-boundary-finalize` with `Close: session` and the same Review object.
- **`T024.001` · `T024` record corrections ACCEPTED at [119-02-yield](iterations/119-02-yield.md) — no user question** (2026-10-18; implemented at [119-02](iterations/119-02-forward.md)):
  `quality-backlog.md` §Refactoring now reads **37 open / 37 literal `→ routed:` / 0 `routed onward to`**, every
  handle naming a complete `## Roadmap items` row; `handoff-context.md`'s element census is **1/1/1**, the `T024`
  pointer is a plain line and all four following sections are present. Both findings `absorbed`
  (`DC-T024-Y-1-r1`, `DC-T024-Y-3-r1`); the stray `<details>` was already deleted at checkpoint `268`, so that
  half was verified rather than re-edited. Session-record text only — zero production/test/config/`decisions.md`/
  `invariants/` bytes; the T024 wave's unscoped suite at `793a5f0` is green (**551 passed**: 513 @ 96.71% + 17/15/6
  gate) and no code byte has changed since. Next: re-dispatch `phase-boundary-finalize` with `Close: session` — its per-row
  `evidence/INDEX.md` rulings (20 of 38 undecided) remain its own step 16. `FK-011` stays resolved; no fork is open.
- **⚑ Gate upstream `http://hyperipper.local:30000`** (user correction 2026-10-17; `:30001` is a peer
  Buster Ripper proxy). `README.md`'s `:30001` occurrences and the app's `--port` default stay untouched.
- **Context trimmed to this cap**: superseded blocks are verbatim in [handoff-archive.md](handoff-archive.md)
  §Trimmed 2026-10-18 — the pre-`116` `## Current Status` block (`116-01-forward`), the `T023` blueprint
  (`116-01-yield`), the `T023` bullet (`117-yield-phase`), the `STABLE`/`MS-001` bullets (`119-01-yield`)
  and the `T024.001` routed bullet (`119-02-forward`).

## Trimmed 2026-10-19 (`126-assumption-decide`) — PR #1 round-1/round-3 task context and closure rows moved verbatim

Moved from `handoff-context.md` at the round-4 routing pass so the baton stays under its cap while the live round-4 blocks stay. Nothing here is deleted; every block is byte-verbatim.


### round 3 Planned Task Context (three briefs, `f37d475`)

- **PR #1 review round 3 (2026-10-19, `f37d475`) — repair lane, ONE wave, three briefs.** The CODIFY session stays
  closed; the vehicles are lite briefs under `00bab65:.sessions/lite/`. Round line: **round 3, not late**. The three roots are
  independent (three modules, three causes): no prerequisite, no capacity ordering (`chat.py` untouched at 300/300), no `RM-001`.
  Order inside the wave is only for commit attribution: the two `server.py` briefs run sequentially so their edits never
  interleave in one lite run.

### Wave 1 — shared-session cookie jar (`2026-10-19-pr1-r3-session-cookie`)

- **Anchor**: lite brief `2026-10-19-pr1-r3-session-cookie`.
- **Members**: finding 1 (comment 4316745948, `server.py:66-68`, P1).
- **Evidence**: [125](iterations/125-assumption-decide.md) §Family sweep — the three types, every candidate audited; [comment](https://github.com/AlexGS74/buster-ripper/pull/1#discussion_r4316745948).
- **Prerequisite**: none. First because it is the P1 and the smallest change (one operand).
- **Owning route**: `lite-forward` → `lite-yield`.
- **Execution order**: 1 of 3 in the wave; before the backend brief (the two `server.py` edits land in separate commits).
- **Separation reason**: none from interference; different cause from the other two. Ordered before the backend brief so
  the two `server.py` edits land in separate commits.
- **Done when**: an upstream `Set-Cookie` on caller A's answer never reaches caller B's upstream request, caller A's own
  forwarded `Cookie` header travels byte-identical and is not overridden, `Set-Cookie` still reaches caller A downstream, and
  `app["session"].cookie_jar` is a `DummyCookieJar` shared by the probe and the KV poll; the witness is red before the operand.
- **Family sweep**: `server.py`'s `ClientSession` has no `headers`/`auth`/`proxy`/`trust_env`/`raise_for_status` operand and
  the connector carries no caller identity — the jar is the only cross-caller carrier, so the brief closes the whole family.
- **Invariant cross-reference**: Neutral — zero active/wip registry invariants; no invariant candidate. `PI:T006/upstream_tls/optout_is_process_wide`
  is respected (one session, one connector).
- **Closure ledger**: `DC-PR1-R3-1-r1`.

### Wave 1 — backend detection retry (`2026-10-19-pr1-r3-backend-retry`)

- **Anchor**: lite brief `2026-10-19-pr1-r3-backend-retry`.
- **Members**: finding 2 (comment 4316745957, `backend.py:62-69`, P2).
- **Evidence**: [125](iterations/125-assumption-decide.md) §Family sweep — the three types, every candidate audited; [comment](https://github.com/AlexGS74/buster-ripper/pull/1#discussion_r4316745957).
- **Prerequisite**: none. Runs after the cookie brief for `server.py` commit separation only.
- **Owning route**: `lite-forward` → `lite-yield`.
- **Execution order**: 2 of 3 in the wave.
- **Separation reason**: none from interference; different cause and a different module boundary (`backend.py` owns the cache,
  `server.py` only spawns/cancels it).
- **Done when**: an upstream that refuses `/v1/models` at `on_startup` and answers `owned_by: "SGLang"` later classifies the
  backend `sglang` without a restart; a CLI override still wins and issues no probe; a definitive answer caches once and stops the
  retries; an empty model list is retried, not cached; `on_cleanup` cancels and awaits the retry task before closing the session;
  an upstream that never answers never blocks a request; the witness is red before the retry.
- **Family sweep**: the empty-list path (`_classify_models:78-79`) is the same transient-as-permanent type and is covered by the
  same stop condition; `compaction._kv_cache_usage`, `image._image_cache`, `dump._sessions` and `session._stats` were audited and
  cache no failure.
- **Invariant cross-reference**: Neutral — zero active/wip registry invariants; no invariant candidate. T007's 5 s bound and the
  `unknown` fallback value stay byte-unchanged.
- **Closure ledger**: `DC-PR1-R3-2-r1`.

### Wave 1 — diff control-line terminators (`2026-10-19-pr1-r3-diff-control-lines`)

- **Anchor**: lite brief `2026-10-19-pr1-r3-diff-control-lines`.
- **Members**: finding 3 (comment 4316745969, `dump.py:265-273`, P2).
- **Evidence**: [125](iterations/125-assumption-decide.md) §Family sweep — the three types, every candidate audited; [comment](https://github.com/AlexGS74/buster-ripper/pull/1#discussion_r4316745969).
- **Prerequisite**: none; independent of the two `server.py` briefs.
- **Owning route**: `lite-forward` → `lite-yield`.
- **Execution order**: 3 of 3 in the wave.
- **Separation reason**: none from interference; different module and different cause.
- **Done when**: the documented `turn_001.diff` and the `change-focused` turn-1 record each carry `--- turn_000\n`,
  `+++ turn_001\n` and a terminated `@@ … @@\n` as their own lines; the `+A/-R` counts and both summary lines are
  byte-unchanged; the `QAP-MS001` expected side is re-pointed and its coarse check fails on a glued artifact; both witnesses are
  red before the operand.
- **Family sweep**: both production writers (`_write_diff_turn`, `_change_focused_record`) share `_unified_diff`, so one operand
  fixes both; `self-contained`, `_flush_state`'s separators, `index.txt` and `turn_000.json` were audited and are well-formed.
- **Invariant cross-reference**: Neutral — zero active/wip registry invariants; no invariant candidate. `G-008`(a)'s
  "today's content and line format" is bound to the file set and record shapes, not to base's malformed control-line join
  (see [125](iterations/125-assumption-decide.md) §The `G-008`(a) binding).
- **Closure ledger**: `DC-PR1-R3-3-r1`.

### round 1 Planned Task Context (three waves, `0d465ec`)

- **PR #1 review round 1 (2026-10-19, `0d465ec`) — repair lane, three waves.** The CODIFY session stays closed; the vehicles are lite briefs under `00bab65:.sessions/lite/`. Round line: **round 1, not late**. Anchor and membership are recorded below; every finding keeps its own ledger row, brief and lite-yield.

### Wave 1 — chat relay fidelity (`2026-10-18-pr1-chat-relay-fidelity`)

- **Anchor**: lite brief `2026-10-18-pr1-chat-relay-fidelity`.
- **Members**: finding 3 (comment 4309720772, `chat.py:221` unfixed re-serialization); finding 7 (4309720784, `chat.py:190` streaming error headers); finding 8 (4309720787, `chat.py:149` streaming overflow).
- **Prerequisite**: none. This is the **first** wave because it is the only net-negative `chat.py` change: the bool-return refactor in `_apply_response_transforms` (−2) and the `_chat_answer` reuse in `_relay_chat_stream` (−2) take `chat.py` from 300 to 296.
- **Owning route**: `lite-forward` → `lite-yield`, one brief.
- **Separation reason**: none from other briefs. It shares `_relay_chat_stream`/`_chat_answer` with the post-commit brief, but changes the error and body arms, not the post-commit success arm; editing the same file is not interference. It is ordered first for concrete capacity: the P1's +1/+2 arm cannot be the first edit at 300/300.
- **Done when**: (a) an unfixed valid JSON chat answer relays the upstream's exact bytes while a transformed answer still carries `json.dumps`; (b) a streaming chat 429 relays `Retry-After` and `X-Request-Id`; (c) an SGLang streaming chat 400 carrying the overflow message re-issues with the reduced `max_completion_tokens`; each test is red before the fix.
- **Invariant cross-reference**: Neutral — zero active/wip registry invariants; no invariant candidate.

### Wave 1 — dump containment (`2026-10-18-pr1-dump-containment`)

- **Anchor**: lite brief `2026-10-18-pr1-dump-containment`.
- **Members**: finding 5 (comment 4309720778, `dump.py:155` live symlinked session directory).
- **Prerequisite**: none; independent of every other brief (different component, no shared assumption).
- **Owning route**: `lite-forward` → `lite-yield`.
- **Separation reason**: different file and different cause; can run beside the chat relay brief.
- **Done when**: with `DUMP_DIR/<sid>` a live symlink to an outside directory, `dump_turn` writes nothing outside and the symlink survives; a healthy session still captures; the process still serves; the test is red before the guard.
- **Invariant cross-reference**: Neutral — zero active/wip registry invariants; the containment contract is the writer's own `_require_writable`/`_writable_target` and QAP-MS001 R2.

### Wave 2 — chat post-commit (`2026-10-18-pr1-chat-post-commit`)

- **Anchor**: lite brief `2026-10-18-pr1-chat-post-commit`.
- **Members**: finding 1 (comment 4309720759, `chat.py:197` P1 post-commit second head).
- **Prerequisite**: Wave 1's chat relay-fidelity brief — the arm adds two raw lines and only fits after 300 → 296. `RM-001` is the alternative prerequisite if this P1 must be the first `chat.py` edit; under the recorded order it is not needed.
- **Owning route**: `lite-forward` → `lite-yield`; **its own brief, commit and lite-yield** for independent attribution of the P1, inside this PR (the user's round-1 instruction overrides `RM-016`'s "own branch off main").
- **Done when**: a streaming chat whose upstream sends a 200 SSE head, one chunk, then raises `ClientPayloadError` produces exactly one HTTP status line, no `HTTP/1.1 502` bytes in the body, and a terminated stream; `CancelledError` still propagates; the test is red before the arm.
- **Invariant cross-reference**: Neutral — zero active/wip registry invariants; the closure basis is `RM-016` and `messages.py:_relay_after_commit` (T013).

### Wave 3 — transport parity (`2026-10-18-pr1-transport-parity`)

- **Anchor**: lite brief `2026-10-18-pr1-transport-parity`.
- **Members**: finding 2 (comment 4309720765, `passthrough.py:52` timeout class, plus the messages and chat arms); finding 9 (4309720790, `passthrough.py:34` redirects, all four relay calls).
- **Prerequisite**: Wave 1's chat relay-fidelity brief — the chat arm is +1 raw line and needs the headroom; Wave 2 is not a prerequisite.
- **Owning route**: `lite-forward` → `lite-yield`.
- **Separation reason**: none from Wave 1 (different cause: transport operands vs relay body/header fidelity); it is its own brief because the two findings share one cause — the aiohttp per-request operands drifted from base's httpx defaults.
- **Done when**: a fake upstream raising `asyncio.TimeoutError` yields the route's JSON 502 at all three timeout-bearing arms, and a 302 with `Location` reaches the client as 302 with `Location` on the passthrough, messages non-stream, messages stream and chat calls; both tests are red before the fix.
- **Invariant cross-reference**: Neutral — zero active/wip registry invariants; no invariant candidate.

### Consumed blueprints
- `T021.001`/`T021.002` blueprints are consumed; both are verbatim in [handoff-archive.md](handoff-archive.md) §Trimmed 2026-10-18 (`100-yield`).
- `T022.001` blueprint is consumed; verbatim in [handoff-archive.md](handoff-archive.md) §Trimmed 2026-10-18 (`107-assumption-decide`).

- `T021.003` blueprint is consumed (resolved at [095](iterations/095-yield-red.md) §Adjudication); verbatim in [handoff-archive.md](handoff-archive.md) §Trimmed 2026-10-18 (`YR-MS001-r4`).

- `T023` blueprint is consumed (accepted and `resolved` at [116-01-yield](iterations/116-01-yield.md)); verbatim in [handoff-archive.md](handoff-archive.md) §Trimmed 2026-10-18 (`116-01-yield`).

- `T024` blueprint is consumed (implemented in Forward 2026-10-18); verbatim in [handoff-archive.md](handoff-archive.md) §Trimmed 2026-10-18 (`T024`).

<details>
<summary>T024.001 Planned Task Context — apply the T024 record corrections</summary>

- Owning route: **Forward** (`T024` subtask), then re-Yield. **Session record text only** — `quality-backlog.md` and `handoff-context.md`; no production, test, config, `decisions.md` or `invariants/` byte.
- Findings: `T024-Y-1` and `T024-Y-3` ([119-01-yield](iterations/119-01-yield.md)); dispositions `absorbed`, ledger `DC-T024-Y-1-r1` / `DC-T024-Y-3-r1`; no user question.
- Required behavior: every open `## Refactoring` line carries the literal `→ routed:` marker naming exactly one complete `## Roadmap items` row, and the section's State line is true of its own bytes; `handoff-context.md` renders every live section with no stray wrapper.
- Selected closure: replace `routed onward to` with `**→ routed:**` in the 28 prose-only `T024 (2026-10-18)` tails; delete the stray `<details>` line. No user question.
- Invariant cross-reference: **Neutral** — zero `active`/`wip` registry invariants; no invariant candidate.
- Done when: a `python3` census over `## Refactoring` reports **37 open / 37 literal `→ routed:` / 0 `routed onward to`**, each naming one `RM-*` handle present as a complete row in `.sessions/codify/roadmap/2026-10-18-message-dump-bulk-flush.md`; every element line `<details>` in `handoff-context.md` has its `</details>` and a `<summary>` (measured 1/1/1), and the four sections after the `T024` pointer are still present; `git diff --name-only` names only the two record files.
- Yield replay: the same census and grep on the post-fix bytes; one remaining prose tail or one stray tag is a reject.
- Assumption impact: `skip:session-only`; Verification profile: **parallel**.
- Execution order: immediately, before re-dispatching `phase-boundary-finalize` with `Close: session`.

</details>

### round 1 Deterministic Closure Ledger rows (`DC-PR1-R1-*`)

### DC-PR1-R1-1-r1
Finding: PR1-R1-1 (PR #1 review comment 4309720759, P1 — `chat.py:197` post-commit second head)
Evidence: https://github.com/AlexGS74/buster-ripper/pull/1#discussion_r4309720759; iterations/124-assumption-decide.md §Findings and dispositions
Evidence fingerprint: unavailable
Repository state: git-commit:0d465ec873ab4b01daf936a37f99c15b51fdbf31
Selected closure: wrap the post-`prepare()` relay in `_relay_chat_stream` in one `except Exception` arm that logs position-not-cause through `log.error` and returns the already-prepared `response`; leave `asyncio.CancelledError` propagating and let `finish_response`'s `write_eof` terminate the stream.
Closure basis: `RM-016`'s "What the fix owes" (R-E's shape a second time — one post-commit arm, log position-not-cause, return the committed object, `CancelledError` propagating); `messages.py:_relay_after_commit` accepted precedent (`T013`); `G-003`; the user's round-1 instruction overriding `RM-016`'s branch deferral.
Rejected candidates:
- Return a fresh 502 from `_make_chat_request` on a post-commit `ClientError`: preserves the verified second head, the exact defect the finding names.
- Catch only `aiohttp.ClientError` in the relay: misses the `AttributeError`, `TimeoutError` and downstream `OSError` classes the committed relay can raise; `_relay_after_commit` catches `Exception` for that reason.
- Add a shared helper outside `chat.py`: the capacity gate forbids an arbitrary helper solely to move lines; `chat.py` at 300/300 is handled by Wave 1's net-negative brief, not by a new module.
Risk shape: unchanged - one private arm in an existing route function; the documented path, the error branch, the non-stream relay and the request transforms are byte-unchanged.
Deferred additive work: a `seam-S11` reading for the chat relay — `RM-016`'s recorded evidence obligation; trigger = the next `S11` composition dispatch on these bytes.
Task: lite brief `00bab65:.sessions/lite/2026-10-18-pr1-chat-post-commit/brief.md`, step 1

### DC-PR1-R1-2-r1
Finding: PR1-R1-2 (PR #1 review comment 4309720765, P2 — `passthrough.py:52` timeout class)
Evidence: https://github.com/AlexGS74/buster-ripper/pull/1#discussion_r4309720765; iterations/124-assumption-decide.md §Findings and dispositions
Evidence fingerprint: unavailable
Repository state: git-commit:0d465ec873ab4b01daf936a37f99c15b51fdbf31
Selected closure: widen the class list at the three timeout-bearing arms — `routes/passthrough.py:48`, `routes/messages.py:332` and `routes/chat.py:150` — to `except (aiohttp.ClientError, TimeoutError) as e:`, keeping each site's existing 502 `{"error": "Upstream request failed"}` envelope and its log line; `TimeoutError` is the built-in alias on the declared ≥3.11 range, so no import is added. One fail-first witness per arm names `asyncio.TimeoutError` as the escaped class.
Closure basis: every repo-authored client-facing failure uses the `{"error": …}` envelope (`messages.py:33`, `chat.py:32`, `passthrough.py:52`, `count_tokens.py:23-25`); `RM-010`'s whole-arm scope ("so one label does not mean three things across the package"); `G-003`; the user's round-1 instruction overriding `RM-010`'s posture deferral; base settled the class worse (an `httpx` timeout propagated to Starlette as a 500), so no parity constraint ranks aiohttp's generated 504.
Rejected candidates:
- Leave the arms and pin aiohttp's generated 504: preserves the generated `text/plain` body and traceback log the finding names; no settled authority adopts 504.
- Widen only `handle_passthrough`: leaves the same mismatch at the messages and chat arms, contrary to `RM-010`'s whole-arm scope.
- Catch `asyncio.TimeoutError` by importing `asyncio` at each site: the built-in `TimeoutError` is the same class on Python ≥3.11; the import adds lines and a module dependency for no behavioral gain.
Risk shape: unchanged - the only observable change is the timeout class reaching each site's existing authored envelope; no new mechanism and no new import.
Deferred additive work: `S17-N4`'s per-request ceiling remains with `RM-010`'s owner; trigger = a new upstream-timeout config channel.
Task: lite brief `00bab65:.sessions/lite/2026-10-18-pr1-transport-parity/brief.md`, step 1

### DC-PR1-R1-3-r1
Finding: PR1-R1-3 (PR #1 review comment 4309720772, P2 — `chat.py:221` unfixed re-serialization)
Evidence: https://github.com/AlexGS74/buster-ripper/pull/1#discussion_r4309720772; iterations/124-assumption-decide.md §Findings and dispositions
Evidence fingerprint: unavailable
Repository state: git-commit:0d465ec873ab4b01daf936a37f99c15b51fdbf31
Selected closure: change `_apply_response_transforms` to mutate `resp_data` in place and return `bool` (True when a transform changed the body), and have `_chat_answer` relay the original `content` bytes when it returns False, reserving `json.dumps` for the fixed path. Net −2 raw lines in `chat.py`.
Closure basis: base `buster_ripper_legacy.py:882-892` re-serialized only the arm it had actually rewritten; `DC-CR2-F-1-r1`'s deferred half; the finding's byte-integrity evidence (ETag/digest must describe the bytes the client receives); `G-003`; the user's round-1 instruction overriding the capacity deferral.
Rejected candidates:
- Keep the re-serialization and record it: normal-use byte drift, and the fix is net-negative; the deferral's own condition (`chat.py` 300/300) is removed by the same fix.
- Add `content` as a fourth parameter to `_apply_response_transforms`: takes the helper from two gray metrics to three and triggers the preventive capacity gate; the in-place bool return removes metrics instead.
- Relay the original bytes on both paths: the fixed path's body is not upstream's, so it must be re-serialized; only the unfixed path may relay `content`.
Risk shape: unchanged - one private helper's return contract and one call site; the fixed path still re-serializes; the non-JSON fall-through and the overflow probe are byte-unchanged.
Deferred additive work: none.
Task: lite brief `00bab65:.sessions/lite/2026-10-18-pr1-chat-relay-fidelity/brief.md`, step 1

### DC-PR1-R1-5-r1
Finding: PR1-R1-5 (PR #1 review comment 4309720778, P2 — `dump.py:155` live symlinked session directory)
Evidence: https://github.com/AlexGS74/buster-ripper/pull/1#discussion_r4309720778; iterations/124-assumption-decide.md §Findings and dispositions
Evidence fingerprint: unavailable
Repository state: git-commit:0d465ec873ab4b01daf936a37f99c15b51fdbf31
Selected closure: in `_lookup_or_create_state`, refuse when `config.DUMP_DIR / sid` is a symlink before `mkdir`, raising `ValueError` so the caller's swallow-all arm at `messages.py:55-59` keeps the request alive; the dangling case stays covered by `mkdir`'s own `FileExistsError`.
Closure basis: `_require_writable`/`_writable_target`'s own containment rule ("a write target must be absent or a regular, non-symlink file") extended to the session directory; QAP-MS001 R2's documented-writer containment contract; `G-006`/`G-008`; the caller's swallow-all contract.
Rejected candidates:
- Check only the artifact leaf: the verified defect is the ancestor symlink; `_require_writable` already checks the leaf.
- Replace or delete the symlink: R2 asserts the symlink survives, and rewriting it routes around the fault.
- Resolve the session directory and verify containment under `DUMP_DIR`: `Path.resolve()` follows the symlink and turns the escape into a legitimate-looking path; the writer must refuse, not normalize.
Risk shape: unchanged - one guard at state creation; both writers inherit it; no format, record or write-count change.
Deferred additive work: none.
Task: lite brief `00bab65:.sessions/lite/2026-10-18-pr1-dump-containment/brief.md`, step 1

### DC-PR1-R1-7-r1
Finding: PR1-R1-7 (PR #1 review comment 4309720784, P2 — `chat.py:190` streaming error headers)
Evidence: https://github.com/AlexGS74/buster-ripper/pull/1#discussion_r4309720784; iterations/124-assumption-decide.md §Findings and dispositions
Evidence fingerprint: unavailable
Repository state: git-commit:0d465ec873ab4b01daf936a37f99c15b51fdbf31
Selected closure: in `_relay_chat_stream`'s `resp.status >= 400` arm, relay the upstream's read bytes under the shared `utils.response_headers(resp.headers)` instead of a `Content-Type`-only header, so `Retry-After`, `X-RateLimit-*` and every other end-to-end name travel. Net −1 raw line.
Closure basis: base `buster_ripper_legacy.py` relayed `_response_headers(resp.headers)` on the same answer; `DC-CR2-F-2-r1`'s end-to-end-header rule; `T008.001`'s status/bytes/representation scope was a floor, not an exclusion; the user's round-1 instruction.
Rejected candidates:
- Keep the `Content-Type`-only arm and record the loss: normal-use header loss and base drift, with a net-negative fix; `T008.001` did not exclude other names.
- Hand-pick `Retry-After` and rate-limit names: a hand-kept list drifts from the upstream and duplicates the shared filter's job.
Risk shape: unchanged - status and bytes already travel; only which end-to-end names travel changes, and the shared hop-by-hop drop is untouched.
Deferred additive work: none.
Task: lite brief `00bab65:.sessions/lite/2026-10-18-pr1-chat-relay-fidelity/brief.md`, step 2

### DC-PR1-R1-8-r1
Finding: PR1-R1-8 (PR #1 review comment 4309720787, P2 — `chat.py:149` streaming chat overflow)
Evidence: https://github.com/AlexGS74/buster-ripper/pull/1#discussion_r4309720787; iterations/124-assumption-decide.md §Findings and dispositions
Evidence fingerprint: unavailable
Repository state: git-commit:0d465ec873ab4b01daf936a37c99c15b51fdbf31
Selected closure: in `_relay_chat_stream`, route the `resp.status >= 400` arm through the existing `_chat_answer(resp, await resp.read(), profile)`, so `_raise_chat_overflow` sees the error body and raises `ContextOverflowError` before the response is returned; `handle_chat`'s retry loop then catches it and re-issues with the reduced `max_completion_tokens`. The same arm carries findings 7 and 3.
Closure basis: `handle_chat`'s own docstring ("Includes context overflow retry for SGLang backends"); `messages.py:handle_streaming`'s accepted shape (`T008.001`); `TestStreamingOverflowRetry` as the messages precedent; `G-003`.
Rejected candidates:
- Leave streaming chat without retry and document it: contradicts the route's own advertised backend-aware retry and the messages-route precedent; the fix reuses existing helpers and adds no mechanism.
- Parse the error with `resp.json()`: a non-JSON error body would raise a new exception and turn a relayed error into a proxy 502; `_chat_answer`'s read-once JSON guard is the existing correct shape.
Risk shape: unchanged - the error arm now uses the same read-once parse and overflow probe as the non-stream arm; the success stream path and the status/bytes relay are byte-unchanged.
Deferred additive work: none.
Task: lite brief `00bab65:.sessions/lite/2026-10-18-pr1-chat-relay-fidelity/brief.md`, step 3

### DC-PR1-R1-9-r1
Finding: PR1-R1-9 (PR #1 review comment 4309720790, P2 — `passthrough.py:34` redirects followed)
Evidence: https://github.com/AlexGS74/buster-ripper/pull/1#discussion_r4309720790; iterations/124-assumption-decide.md §Findings and dispositions
Evidence fingerprint: unavailable
Repository state: git-commit:0d465ec873ab4b01daf936a37f99c15b51fdbf31
Selected closure: pass `allow_redirects=False` at all four relay calls — `routes/passthrough.py`'s `session.request`, `routes/messages.py`'s two `session.post` calls and `routes/chat.py`'s `session.post` — so a 3xx reaches the client as the upstream sent it (status, `Location`, body). One operand per call site.
Closure basis: base `httpx`'s default `follow_redirects=False` (base parity); `G-003` holds `/v1/messages` behavior; the relay contract at every route; the user's round-1 instruction.
Rejected candidates:
- Fix only `handle_passthrough`: leaves the same drift on the messages and chat POST calls the finding names.
- Set the operand on the shared `ClientSession`: `ClientSession` has no such constructor parameter; the per-request operand is the native aiohttp mechanism.
- Follow the redirect and relay the final response: changes the status and drops `Location`, the exact defect.
Risk shape: unchanged - one per-request operand per call site; the shared session and connector are untouched.
Deferred additive work: none.
Task: lite brief `00bab65:.sessions/lite/2026-10-18-pr1-transport-parity/brief.md`, step 2

### round 3 Deterministic Closure Ledger rows (`DC-PR1-R3-*`)

### DC-PR1-R3-1-r1
Finding: PR1-R3-1 (PR #1 review comment 4316745948, P1 — `server.py:66-68` shared-session cookie jar)
Evidence: https://github.com/AlexGS74/buster-ripper/pull/1#discussion_r4316745948; iterations/125-assumption-decide.md §Family sweep — the three types, every candidate audited
Evidence fingerprint: unavailable
Repository state: git-commit:f37d47529e44248f1c4a7e8cd460fe9273181528
Selected closure: construct the one shared `ClientSession` with `cookie_jar=aiohttp.DummyCookieJar()`, so no upstream `Set-Cookie` is stored or replayed and each request sends only the `Cookie` header its own caller forwarded; the response's relayed `Set-Cookie` is byte-unchanged.
Closure basis: base parity (base built a per-request `httpx.AsyncClient` at `buster_ripper_legacy.py:668/704/847/906`, so no cookie outlived a request); the relay contract (`utils.forward_headers` already forwards `Cookie`, `utils.response_headers` already relays `Set-Cookie` downstream); `G-003`; the user's round-3 instruction; installed aiohttp shows `_request` merging the jar into the caller's `Cookie` header and `ClientRequest.update_cookies` overriding a same-named caller value.
Rejected candidates:
- Build a fresh `ClientSession` per request: abandons the one-connector posture `PI:T006/upstream_tls/optout_is_process_wide` and T006's whole point; also re-opens the TLS handshake per request.
- Keep the jar and clear it between requests: a timer or middleware, a new mechanism that races concurrent requests and leaves the window open between them.
- Relay the upstream's `Set-Cookie` downstream but keep the jar: preserves the verified cross-caller leak.
Risk shape: unchanged - one constructor operand; the connector, the TLS posture, every route and every relay are byte-unchanged.
Deferred additive work: none.
Task: lite brief `00bab65:.sessions/lite/2026-10-19-pr1-r3-session-cookie/brief.md`, step 1

### DC-PR1-R3-2-r1
Finding: PR1-R3-2 (PR #1 review comment 4316745957, P2 — `backend.py:62-69` transient probe result cached permanently)
Evidence: https://github.com/AlexGS74/buster-ripper/pull/1#discussion_r4316745957; iterations/125-assumption-decide.md §Family sweep — the three types, every candidate audited
Evidence fingerprint: unavailable
Repository state: git-commit:f37d47529e44248f1c4a7e8cd460fe9273181528
Selected closure: cache only a definitive classification (`sglang`, `vllm`, or a successful non-empty model list); on an exception or an empty list leave the state uncached and re-probe from a background task spawned in `on_startup` and cancelled-and-awaited in `on_cleanup` through the existing `_cancel_background_task`, until a definitive answer arrives.
Closure basis: T007's done-when fixes the 5 s bound and the `unknown` fallback *value*, not the permanence of a transient result; the package's own background-poll pattern (`spawn_kv_cache_monitor`/`spawn_dump_flush` + `_cancel_background_task`, `server.py:25-45`, `:86-106`); `compaction.poll_kv_cache`'s "upstream may be starting up; ignore silently" is the same problem shape; `RM-005`'s no-new-CLI-surface posture; `G-002`'s clean shutdown; the user's round-3 instruction.
Rejected candidates:
- Leave `_backend_state_cached` False and re-probe lazily at the three consumers: adds an await to the request path and cannot fix the synchronous `normalize._normalize_date` read.
- Bounded retries inside `detect_backend`: still gives up permanently after the retry window, so it does not meet the required outcome for a longer outage.
- Treat `unknown` as `sglang`: wrong for vLLM — date stripping and the overflow retry would fire on a non-SGLang backend.
- Keep the permanent cache and record it: leaves the verified defect in place, contrary to the user's round-3 instruction.
Risk shape: unchanged - one private coroutine in `backend.py` plus one spawn and one cancel line in `server.py`; the probe, its 5 s bound, the `unknown` fallback value and all three consumers are byte-unchanged; no new CLI surface.
Deferred additive work: none.
Task: lite brief `00bab65:.sessions/lite/2026-10-19-pr1-r3-backend-retry/brief.md`, step 1

### DC-PR1-R3-3-r1
Finding: PR1-R3-3 (PR #1 review comment 4316745969, P2 — `dump.py:265-273` diff control lines unterminated; family: the documented `turn_NNN.diff` and the `change-focused` record)
Evidence: https://github.com/AlexGS74/buster-ripper/pull/1#discussion_r4316745969; iterations/125-assumption-decide.md §Family sweep — the three types, every candidate audited; assumptions/compositions/S21.md §N5
Evidence fingerprint: unavailable
Repository state: git-commit:f37d47529e44248f1c4a7e8cd460fe9273181528
Selected closure: pass `lineterm="\n"` in `_unified_diff`, so every yielded line — including `---`, `+++` and `@@` — is terminated and the two writers' `"".join` produces a valid unified diff; re-point the three test-side mirrors (`tests/test_dump.py::_expected_diff`, `tests/test_dump_records.py::_expected_diff`, `tests/qa/test_qap_ms001.py::production_diff_text`) and strengthen the gate's coarse check so a glued artifact fails.
Closure basis: `README.md:165-166` publishes `turn_NNN.diff` as a unified diff; `README.md:198` publishes the `change-focused` record as a unified diff; `G-006`'s failure column makes a not-well-formed artifact a goal failure and requires reading without tooling; `G-008`(a)'s "today's content and line format" is bound to the file set and record shapes, not to base's malformed control-line join (see [125](iterations/125-assumption-decide.md) §The `G-008`(a) binding); `S21` N5 records the gate's mirroring as the malformed-artifact/published-contract gap.
Rejected candidates:
- Join with `"\n"` while leaving `lineterm=""`: content lines already carry their own endings, so every content line is double-terminated and blank lines appear between them.
- Correct `README.md` to stop saying "unified diff": weakens the published contract and leaves `G-006`'s well-formedness failure in place.
- Leave the artifact and record it: leaves the advertised artifact not consumable by its reader, and the fix is one operand.
Risk shape: unchanged - one operand in the single helper both writers and all three mirrors share; the `+A/-R` counts, both summary lines, the record separators, the `self-contained` rendering and `turn_000.json` are byte-unchanged; `dump.py` stays net 0 lines in its 301–500 band.
Deferred additive work: `S21`'s recomposition is owed on the new bytes; the seam is already capped (`§Capped seams`), so no task, Fork or wave opens from it; trigger = the next composition dispatch on these bytes.
Task: lite brief `00bab65:.sessions/lite/2026-10-19-pr1-r3-diff-control-lines/brief.md`, step 1

### Older Planned Task Context, consumed blueprints and pre-round-4 ledger rows

### Consumed blueprints
- `T021.001`/`T021.002` blueprints are consumed; both are verbatim in [handoff-archive.md](handoff-archive.md) §Trimmed 2026-10-18 (`100-yield`).
- `T022.001` blueprint is consumed; verbatim in [handoff-archive.md](handoff-archive.md) §Trimmed 2026-10-18 (`107-assumption-decide`).

- `T021.003` blueprint is consumed (resolved at [095](iterations/095-yield-red.md) §Adjudication); verbatim in [handoff-archive.md](handoff-archive.md) §Trimmed 2026-10-18 (`YR-MS001-r4`).

- `T023` blueprint is consumed (accepted and `resolved` at [116-01-yield](iterations/116-01-yield.md)); verbatim in [handoff-archive.md](handoff-archive.md) §Trimmed 2026-10-18 (`116-01-yield`).

- `T024` blueprint is consumed (implemented in Forward 2026-10-18); verbatim in [handoff-archive.md](handoff-archive.md) §Trimmed 2026-10-18 (`T024`).

<details>
<summary>T024.001 Planned Task Context — apply the T024 record corrections</summary>

- Owning route: **Forward** (`T024` subtask), then re-Yield. **Session record text only** — `quality-backlog.md` and `handoff-context.md`; no production, test, config, `decisions.md` or `invariants/` byte.
- Findings: `T024-Y-1` and `T024-Y-3` ([119-01-yield](iterations/119-01-yield.md)); dispositions `absorbed`, ledger `DC-T024-Y-1-r1` / `DC-T024-Y-3-r1`; no user question.
- Required behavior: every open `## Refactoring` line carries the literal `→ routed:` marker naming exactly one complete `## Roadmap items` row, and the section's State line is true of its own bytes; `handoff-context.md` renders every live section with no stray wrapper.
- Selected closure: replace `routed onward to` with `**→ routed:**` in the 28 prose-only `T024 (2026-10-18)` tails; delete the stray `<details>` line. No user question.
- Invariant cross-reference: **Neutral** — zero `active`/`wip` registry invariants; no invariant candidate.
- Done when: a `python3` census over `## Refactoring` reports **37 open / 37 literal `→ routed:` / 0 `routed onward to`**, each naming one `RM-*` handle present as a complete row in `.sessions/codify/roadmap/2026-10-18-message-dump-bulk-flush.md`; every element line `<details>` in `handoff-context.md` has its `</details>` and a `<summary>` (measured 1/1/1), and the four sections after the `T024` pointer are still present; `git diff --name-only` names only the two record files.
- Yield replay: the same census and grep on the post-fix bytes; one remaining prose tail or one stray tag is a reject.
- Assumption impact: `skip:session-only`; Verification profile: **parallel**.
- Execution order: immediately, before re-dispatching `phase-boundary-finalize` with `Close: session`.

</details>

### DC-S22-r1
Finding: S22
Evidence: assumptions/compositions/S22.md
Evidence fingerprint: sha256:593e59ba2d3b3df8645dea2ae153ae5aaffe21eadcd43c067e22ebe44a8bb1e8
Repository state: git-commit:702c5b8769fb4ae1411453917b7ca68d8bee82cf
Selected closure: Make `Proxy.post` build its opener with `urllib.request.ProxyHandler({})`, matching `require_upstream`, so the gate's client hop reaches `127.0.0.1:<port>` regardless of ambient proxy settings.
Closure basis: the declared endpoint contract (`UPSTREAM`, `tests/qa/harness.py:48`) and `require_upstream`'s own direct-probe construction (`:57`); the oracle's job is to judge the declared endpoint's answer, and the proxy subprocess already bypasses ambient proxies (`buster_ripper/server.py:65-67`).
Rejected candidates:
- Require `no_proxy=127.0.0.1` as a documented gate precondition: leaves the defect in place whenever the environment is not prepared, so it does not repair the oracle.
- Set `http_proxy`/`no_proxy` in a conftest fixture: environment-dependent, and it would still let any other ambient proxy setting misroute the client hop.
- Leave `Proxy.post` unchanged and record the ambient-proxy requirement: preserves the verified defect on a reachable environment.
Risk shape: unchanged - test-support bytes only; no production caller and no shipped-runtime behavior.
Deferred additive work: QB:assumption-registry/s22-ambient-proxy-premise (collector lane; trigger = the next `S22` composition or promotion) - the CE rests on `urllib`'s default-opener behavior, which is not one of the two member claims; re-extract it as a `lang:python` atomic proposition before promoting the seam.
Task: T021.002 step 2

### DC-T024-Y-1-r1
Finding: T024-Y-1 (119-01-yield §Findings P2 — Refactoring marker form)
Evidence: iterations/119-01-yield.md
Evidence fingerprint: unavailable
Repository state: git-commit:eb88254354d8b044bf6c4eb319f5257c5245863a
Selected closure: Replace the prose verb phrase `routed onward to` with the literal marker `**→ routed:**` in the 28 open `## Refactoring` lines' `T024 (2026-10-18)` tails, keeping each destination path, `RM-*` handle and label byte-identical; leave the nine lines that already carry a literal `→ routed:` untouched. The section's State line then reads true of its own bytes.
Closure basis: `T024`'s done-when ("every open `## Refactoring` line is `[x]` with its commit or `→ routed:` to a complete `## Roadmap items` row"); `run-codify` `references/session-close.md` step 12, session-completion branch ("the line is marked `→ routed:`"; "The marker names the destination chunk and its `RM-*` handle"); the section's own census vocabulary, which separates literal markers from prose forms; the byte census 37 open / 9 literal / 28 prose-only / 0 undisposed with 37/37 destinations present as complete rows.
Rejected candidates:
- Reword the State line to describe both marker forms: rewrites the record to describe the deviation instead of meeting the named criterion; the approved task wording and the drain rule outrank record convenience.
- Insert the marker beside the prose (`→ routed: routed onward to …`): produces two disposition tokens for one act and a line that reads as both marker and prose; replacing the prose verb with the marker keeps the destination, handle and label while making the line read as the drain vocabulary does.
Risk shape: unchanged - session record text only; no code, test, gate, config, roadmap row or invariant byte moves.
Deferred additive work: none.
Task: T024.001 step 1

### DC-T024-Y-3-r1
Finding: T024-Y-3 (119-01-yield §Findings P2 — unclosed `<details>` in handoff-context.md)
Evidence: iterations/119-01-yield.md
Evidence fingerprint: unavailable
Repository state: git-commit:eb88254354d8b044bf6c4eb319f5257c5245863a
Selected closure: Delete the stray `<details>` line above the `T024` archive pointer in `handoff-context.md` §Planned Task Context, so the pointer stands as a plain line and every following section renders.
Closure basis: the handoff is the baton the next agent and the user read; the `T024` blueprint is already archived verbatim at `handoff-archive.md:2912` §Trimmed 2026-10-18 (`T024`); the element has no `<summary>` and no closing tag, so it swallows `## Deterministic Closure Ledger` through EOF in any rendered view while raw-text readers see nothing wrong.
Rejected candidates:
- Close it with `</details>` after the bullet: leaves an empty collapsible wrapper with no `<summary>`, still hides the pointer behind a disclosure triangle nobody opens, and makes the section boundary depend on a tag nobody maintains; the wrapper carries no content of its own, so deletion is complete.
Risk shape: unchanged - session record text only; no production, test, config, `decisions.md` or `invariants/` byte moves.
Deferred additive work: none.
Task: T024.001 step 2


- **PR #1 round-1 and round-3 closure rows** (`DC-PR1-R1-*`, `DC-PR1-R3-*`) are verbatim in [handoff-archive.md](handoff-archive.md) §Trimmed 2026-10-19 (`126-assumption-decide`); round 4's ten rows are live below.

## Trimmed 2026-10-19 (`127-assumption-decide`) — PR #1 round-4 Planned Task Context, ten closure rows and the headroom Forward section moved verbatim

## Planned Task Context
- **PR #1 review round 4 (2026-10-19, `6a089d8`) — repair lane, TWO waves (one capacity prerequisite, then three briefs in
  parallel).** The CODIFY session stays closed; the vehicles are lite briefs under `00bab65:.sessions/lite/`. Round line: **round 4, not
  late**. Wave 1 is not a finding: it creates the `chat.py` raw-line headroom Wave 2's four chat fixes need. Wave 2's three
  briefs are independent (different files/causes) and run in parallel after Wave 1.

### Wave 1 — `chat.py` capacity headroom (`2026-10-19-pr1-r4-chat-headroom`)

- **Anchor**: lite brief `2026-10-19-pr1-r4-chat-headroom`.
- **Members**: none. Capacity prerequisite for Wave 2 Brief A, not a finding repair; no closure ledger row.
- **Evidence**: [126](iterations/126-assumption-decide.md) §Wave structure; `wc -l buster_ripper/routes/chat.py` → **300**;
  `codify scan path` names `handle_chat` (L:29 N:2 C:6) and `_apply_request_transforms` (L:20 N:2 C:8) as three-gray-metric
  functions, so Wave 2's additions would trigger the preventive capacity gate at exactly 300 raw lines.
- **Prerequisite**: none. **First**, because the four chat fixes add statements and raw lines at exactly 300/300.
- **Owning route**: `lite-forward` → `lite-yield`, one brief.
- **Separation reason**: preparatory capacity, not a repair; the quality gate says "shrink first" and a separate commit keeps the
  shrink reviewable and behavior-free.
- **Done when**: `wc -l buster_ripper/routes/chat.py` ≤ **272**; `git diff -U0` names only comment and docstring lines; the existing
  chat suite is green; no import, statement, branch or signature moves. The named shrink set is the over-long comment/docstring block at
  `:100-115`, `:126-129`, `:156-161`, `:188-203` and the seven one-line docstrings — no code is deleted for room.
- **Family sweep**: `messages.py` (344) and `image.py` (255) do not need headroom this round: Brief B is net-negative and Brief C
  adds to a Green file.
- **Invariant cross-reference**: Neutral — zero active/wip registry invariants; no invariant candidate.
- **Closure ledger**: none (capacity prerequisite, not a finding).

#### Yield — `2026-10-19-pr1-r4-chat-headroom`

- Verdict: ACCEPT
- Ran: `uv run pytest tests/test_header_parity.py tests/test_upstream_answer_relay.py tests/test_routes_integration.py tests/test_stream_relay_commit.py tests/test_context_overflow.py tests/test_transport_parity.py -q` -> 206 passed (the coverage gate reports 52% because only those six files are collected; no test failed); `uv run pytest -q` -> 540 passed, coverage 96.78%; `wc -l buster_ripper/routes/chat.py` -> 272; `git diff --no-ext-diff -U0 --no-color -- buster_ripper/routes/chat.py` -> 6 added / 34 removed, every changed line a comment or docstring; `codify scan path buster_ripper/routes/chat.py` -> per-function L/N/C/P unchanged (`handle_chat` L:29 N:2 C:6, `_apply_request_transforms` L:20 N:2 C:8); `git diff --check` -> clean.
- Wrong witness: executable bytes deleted or moved to reach the bound — ruled out by `ast.dump` equality with docstrings stripped, plus a line-by-line check that every changed line is a comment or docstring; a docstring read elsewhere disappearing — no `__doc__` reader exists in `buster_ripper/` or `tests/` (only `buster_ripper_legacy.py` reads its own module docstring); a comment added past two lines — the four added comments are one line each.
- Findings: none
- Noticed: the named shrink set alone reaches 277, not 272 (`:100-115` −8, `:126-129` −3, `:156-161` −5, the seven one-line docstrings −7; `:188-203` holds only a one-line docstring and no long comment), so the bound also took `handle_chat`'s 4-line docstring (−3) and the two one-line comments at old `:42`/`:46` (−2) — comment/docstring only, both facts preserved. Ask: no — the bound is met and no executable byte moved.
- Noticed: no lite `brief.md` and no `## Forward` section existed at dispatch; the changed file was read from `git status`/`git diff`, and this Yield is recorded in the section the prompt named as the brief. Ask: no — the change is complete and reviewed.

### Wave 2 — three briefs in parallel (after Wave 1)

#### Brief A — chat relay totality (`2026-10-19-pr1-r4-chat-relay`)

- **Anchor**: lite brief `2026-10-19-pr1-r4-chat-relay`.
- **Members**: finding 1 (4317334042, `chat.py:35`, non-object body); finding 3 (4317334057, `chat.py:138`, armed stream
  deadline); finding 4 (4317334067, `chat.py:84`, non-mapping eval kwargs); finding 6 (4317334102, `chat.py:143`, raw request
  bytes); family sibling A1 (`chat.py:177-184`, non-object overflow answer).
- **Evidence**: [126](iterations/126-assumption-decide.md) §Family A / §Family C / §Family E; [comment 4317334042](https://github.com/AlexGS74/buster-ripper/pull/1#discussion_r4317334042).
- **Prerequisite**: Wave 1. Runs beside Briefs B and C.
- **Owning route**: `lite-forward` → `lite-yield`, one brief.
- **Separation reason**: different file and cause from Briefs B and C; one brief because all five edits share `chat.py` and the
  request/response relay concern.
- **Capacity shape**: `handle_chat` and `_apply_request_transforms` already carry three gray metrics, so the parse/root gate and the
  kwargs normalization live in two small private helpers and replace existing statements; `_make_chat_request` is near tolerance on Nesting
  and Parameters, so the raw-byte decision is a helper that reads `request.read()` and **no sixth parameter is added**; `_raise_chat_overflow`
  gains one guard line; the timeout is one operand.
- **Done when**: (a) body `null`, `[1]`, `5` → 400 `{"error": "Invalid JSON"}` in both backend states; (b) a >300 s streaming chat is
  not terminated and a non-stream request still arms 300 s; (c) `chat_template_kwargs: null`/`[]`/`"x"` under `EVAL_MODE` + `glm47`
  still receives `enable_thinking` and the budget; (d) a UTF-16 body forwards byte-identical with its `Content-Type`, a transformed body
  reserializes with `Content-Type: application/json`; (e) a non-object upstream overflow answer relays instead of 500; each test red
  before the fix; `wc -l buster_ripper/routes/chat.py` ≤ **300**.
- **Family sweep**: every chat client-body and nested mapping access audited in [126](iterations/126-assumption-decide.md) §Family A; every
  chat streaming call audited in §Family C; every chat request direction audited in §Family E.
- **Invariant cross-reference**: Neutral — zero active/wip registry invariants; no invariant candidate.
- **Closure ledger**: `DC-PR1-R4-1-r1`, `-3-r1`, `-4-r1`, `-6-r1`, `-A1-r1`.

##### Forward — `2026-10-19-pr1-r4-chat-relay`

- Changed: `buster_ripper/routes/chat.py` — `_read_chat_body` parses, gates a non-object root to the route's 400, and keeps the cached raw bytes; `_chat_template_kwargs` replaces a client-sent non-mapping; `_chat_payload` relays the client's bytes unless a transform changed the body, relabelling `Content-Type: application/json`; `_make_chat_request` sends `data=` with `total=None` on a stream and 300 s otherwise; `_raise_chat_overflow` returns on a non-object answer and reads `usage` only when it is a mapping; `handle_chat` loses three statements and its retry log is reflowed (299 raw lines).
- Changed: `tests/test_routes_integration.py` — non-object root (`null`, `[1]`, `5`) answers 400 `{"error": "Invalid JSON"}` with no upstream call in both backend states.
- Changed: `tests/test_transport_parity.py` — chat's streaming operand arms no total and the non-stream operand keeps 300 s.
- Changed: `tests/test_upstream_answer_relay.py` — `_sent_body` reads whichever request mechanism the route used; new eval `chat_template_kwargs` guard (non-mapping replaced, mapping merged) and non-object overflow answer relay.
- Changed: `tests/test_chat_request_bytes.py` — new module: UTF-16 body forwards byte-identical under its `Content-Type`; a transformed body reserializes under `Content-Type: application/json`.
- Changed: `tests/test_context_overflow.py` — the chat recorder reads the sent body from `data=` or `json=`, so the messages witnesses stay unchanged.
- Ran: red check (`git stash push -- buster_ripper/routes/chat.py`, then the new selection) -> 14 failed, 15 passed (the mapping-merge and non-stream-300 guards are green by design).
- Ran: `uv run pytest tests/test_context_overflow.py tests/test_header_parity.py tests/test_chat_request_bytes.py tests/test_upstream_answer_relay.py tests/test_transport_parity.py tests/test_routes_integration.py -q --no-cov` -> 206 passed.
- Ran: `uv run pytest -q` -> 556 passed, coverage 96.78%.
- Ran: `wc -l buster_ripper/routes/chat.py` -> 299.
- Ran: `codify scan path buster_ripper/routes/chat.py` -> `handle_chat` L:26 N:2 C:6, `_apply_request_transforms` L:20 N:2 C:8, `_make_chat_request` N:2 P:5 (all unchanged); no new helper above tolerance.
- Ran: `git diff --check` -> clean.
- Noticed: `_raise_chat_overflow`'s `usage` branch is unreachable through `parse_context_overflow_error` (it always sets `input_tokens` from the regex); the guard is defensive for a caller that builds the error differently. Ask: no — no current input reaches it.
- Noticed: on the vLLM arm a dict-shaped overflow error still escapes `handle_chat` as a 500, because only the SGLang loop catches `ContextOverflowError`; pre-existing and not a Brief A member. Ask: no — the caller is told with a 500, nothing is silent or written wrong.
- Noticed: `_apply_request_transforms` still wraps its body in `except Exception: pass`, the amplifier that hid the kwargs loss; no current input reaches it with a wrong forwarded body now. Ask: no — no lasting wrong result remains.
- Noticed: `messages.py` still re-serializes its request body and returns a non-object JSON root to `normalize()`; both are Brief B's E1/A3. Ask: no — assigned to Brief B.

##### Yield — `2026-10-19-pr1-r4-chat-relay`

- Verdict: ACCEPT
- Ran: `uv run pytest tests/test_context_overflow.py tests/test_header_parity.py tests/test_chat_request_bytes.py tests/test_upstream_answer_relay.py tests/test_transport_parity.py tests/test_routes_integration.py -q --no-cov` -> 206 passed; `uv run pytest -q` -> 556 passed, coverage 96.78%; `wc -l buster_ripper/routes/chat.py` -> 299; `codify scan path buster_ripper/routes/chat.py --detailed` -> `handle_chat` L:26 N:2 C:6, `_apply_request_transforms` L:20 N:2 C:8, `_make_chat_request` L:15 N:2 C:3 P:5, `_read_chat_body` and `_chat_payload` Green (no new helper above tolerance); `git diff --check` -> clean; helper probe -> unchanged UTF-16 body relayed with its `charset=utf-16`, changed body reserialized `application/json`, string-root overflow returns, dict-root overflow raises 5000/4096, non-mapping kwargs replaced; loopback wire probe -> a plain headers dict holding `content-type` and `Content-Type` reaches an aiohttp upstream as one value, the later one.
- Wrong witness: always-reserialize ruled out (UTF-16 raw relay probe returns the client's bytes and `test_a_utf16_body_forwards_byte_identical_with_its_content_type` asserts them); always-relay-raw ruled out (changed-body probe reserializes `{"a": 1, "b": 2}` under `application/json`); `total=300` on the stream ruled out (`test_a_streaming_chat_arms_no_total` reads the operand, `None`); non-mapping kwargs skipped ruled out (probe replaces `"x"` with `{}` and the guard test asserts the pair); non-object overflow answer 500 ruled out (probe returns on a string root; the test asserts the upstream's 400 body).
- Findings: none
- Noticed: `handoff-archive.md`, `quality-backlog.md`, `roadmap.md`, the codify roadmap and `iterations/124`–`126` are modified/untracked from the 126 routing pass, not Forward's list; left unstaged. Ask: no — session records, not code.
- Noticed: `_raise_chat_overflow` reads `usage` on every dict-shaped error answer although `parse_context_overflow_error` always fills `input_tokens`; an unused read, no wrong result. Ask: no — no input reaches a wrong answer.
- Noticed: a non-UTF-8, non-BOM request charset (e.g. `latin-1`) is reserialized rather than byte-relayed (measured `b'{"caf\\u00e9": 1}'` under `application/json`); meaning and label stay correct. Ask: no — nothing wrong reaches the upstream.
- Noticed: `_chat_payload` adds `Content-Type` beside a case-different client key in the plain headers dict, but a loopback aiohttp peer received exactly one value, the later one (measured). Ask: no — the wire carries a single label.

#### Brief B — messages/count_tokens relay totality (`2026-10-19-pr1-r4-messages-relay`)

- **Anchor**: lite brief `2026-10-19-pr1-r4-messages-relay`.
- **Members**: finding 2 (4317334048, `messages.py:174`, streaming error headers/body); family sibling E1 (`messages.py:156`,
  `:308`, raw request bytes); family sibling A2 (`messages.py:248`, wrong-shaped `message_start`); family sibling A3
  (`count_tokens.py:20`, non-object body).
- **Evidence**: [126](iterations/126-assumption-decide.md) §Family B / §Family E / §Family A; [comment 4317334048](https://github.com/AlexGS74/buster-ripper/pull/1#discussion_r4317334048).
- **Prerequisite**: none. Runs beside Briefs A and C.
- **Owning route**: `lite-forward` → `lite-yield`, one brief.
- **Separation reason**: different file and cause from Briefs A and C; one brief because the four edits share the messages/count_tokens
  request-and-relay concern.
- **Capacity shape**: `_stream_single_request` (L:35) and `_non_stream_single_request` (L:34) are hard Length violations, so every edit
  there is **net-negative or net-zero statements** (the error arm loses one, the probe swaps one expression, `_error_body` is deleted as
  dead code, its 11-line comment shrinks); `handle_streaming` is a hard Deep-Nesting violation, so the probe edit swaps one expression and
  adds nothing; `handle_messages` (C:11) is not touched — the raw-byte decision lives in one helper used by both request call sites;
  `_note_message_start` is within limits and gains two guarded reads.
- **Done when**: (a) a streaming 429 with `Retry-After`, `X-Request-Id`, `Content-Type: text/plain; charset=utf-8` and body `slow down`
  relays all four, and the overflow probe does not raise on an unknown charset; (b) a non-overflow error relays once and unchanged with
  the upstream's headers/bytes; (c) a UTF-16 `/v1/messages` body forwards byte-identical, a normalized body reserializes with
  `Content-Type: application/json`; (d) a `message_start` with `message: null` still relays the following chunk; (e) a non-object
  `count_tokens` body answers 400; each test red before the fix; `wc -l buster_ripper/routes/messages.py` ≤ **344** and below 500.
- **Re-pointed tests**: `tests/test_context_overflow.py::test_an_error_that_is_not_an_overflow_relays_once_and_unchanged` (upstream
  headers and bytes replace the `application/json`/re-encoded-body assertions) and
  `::test_a_stream_error_body_that_no_charset_can_decode_still_answers_itself` (`await response.read()` replaces `await response.text()`).
- **Family sweep**: every error path in the four routes audited in [126](iterations/126-assumption-decide.md) §Family B; the request
  direction in §Family E; the nested mapping accesses in §Family A.
- **Invariant cross-reference**: Neutral — zero active/wip registry invariants; no invariant candidate.
- **Closure ledger**: `DC-PR1-R4-2-r1`, `-E1-r1`, `-A2-r1`, `-A3-r1`.

##### Forward — `2026-10-19-pr1-r4-messages-relay`

- Changed: `buster_ripper/routes/messages.py` — the streaming error arm relays `resp.read()` under `response_headers`, `_error_body` is deleted, the overflow probe reads `result.body.decode(errors="replace")`, `_request_payload` relays the client's bytes at both call sites unless normalization changed the body, `_note_message_start` guards the `message`/`usage` reads, and the relay docstring's AttributeError example is generalized (339 raw lines).
- Changed: `buster_ripper/routes/count_tokens.py` — `_read_body` answers the route's own 400 for any non-object root while keeping the syntax-error fallback.
- Changed: `tests/test_messages_request_bytes.py` — new module: a UTF-16 body forwards byte-identical and a normalized body reserializes with `Content-Type: application/json`, at both call sites.
- Changed: `tests/test_context_overflow.py` — the non-overflow error cases assert the upstream's bytes and end-to-end headers.
- Changed: `tests/test_upstream_answer_relay.py` — new streaming 429 relay case; the undecodable stream error reads bytes; the unknown-response-charset witness now names the probe's own read.
- Changed: `tests/test_stream_relay_commit.py` — the wrong-shaped `message_start` is no longer a fault row; a new case witnesses the following chunk after a null `message` or non-mapping `usage`.
- Changed: `tests/test_count_tokens_parity.py` — a non-object body now expects the route's 400, over three root shapes.
- Changed: `tests/test_image.py` — two route witnesses read the sent body from `data=` or `json=`.
- Ran: red check (`git stash push -- buster_ripper/routes/messages.py buster_ripper/routes/count_tokens.py`, then the new and re-pointed selection) -> 13 failed, 12 passed (the plain-4xx byte half is green pre-fix because the old arm's re-encode is a no-op for decodable ASCII; its header half is red).
- Ran: `uv run pytest tests/test_messages_request_bytes.py tests/test_context_overflow.py tests/test_upstream_answer_relay.py tests/test_stream_relay_commit.py tests/test_count_tokens_parity.py tests/test_image.py tests/test_routes_integration.py tests/test_transport_parity.py tests/test_header_parity.py -q --no-cov` -> 295 passed.
- Ran: `uv run pytest -q` -> 563 passed, coverage 96.78%.
- Ran: `wc -l buster_ripper/routes/messages.py buster_ripper/routes/count_tokens.py` -> 339 / 109.
- Ran: `codify scan path buster_ripper/routes/messages.py --detailed` -> `_stream_single_request` L:35 N:2 C:4, `_non_stream_single_request` L:34 N:4 C:8, `handle_messages` L:28 N:2 C:11, `handle_streaming` L:25 N:3 C:8 (all unchanged); `_request_payload` Green.
- Ran: `codify scan path buster_ripper/routes/count_tokens.py --detailed` -> unchanged from HEAD (3 medium issues, `_read_body` Green).
- Ran: `git diff --check` -> clean.
- Noticed: `_request_payload` duplicates chat's `_chat_payload` (same eight lines); a shared `utils` helper would move Brief A's committed file, which is out of scope. Ask: no — both call sites use one shape and nothing behaves differently.
- Noticed: `tests/test_stream_relay_commit.py`'s `POST_COMMIT_FAULTS` lost its `upstream-payload-attribute-error` row because that payload no longer faults; the new case witnesses the relay instead. Ask: no — the state is closed, not deferred.
- Noticed: `tests/test_count_tokens_parity.py::test_a_non_mapping_body_still_answers_500` characterized base's 500; A3 supersedes it with the route's 400. Ask: no — the brief's done-when (e) rules it.
- Noticed: `handle_messages` still lets a non-object `/v1/messages` root and non-mapping array members reach `normalize()` and raise — `PR1-R4-MSG`, accepted and unchanged. Ask: no — recorded risk, `RM-012` owns the answer.

##### Yield — `2026-10-19-pr1-r4-messages-relay`

- Verdict: ACCEPT
- Ran: `uv run pytest tests/test_messages_request_bytes.py tests/test_context_overflow.py tests/test_upstream_answer_relay.py tests/test_stream_relay_commit.py tests/test_count_tokens_parity.py tests/test_image.py tests/test_routes_integration.py tests/test_transport_parity.py tests/test_header_parity.py -q --no-cov` -> 295 passed; `uv run pytest -q` -> 563 passed, coverage 96.78%; `wc -l buster_ripper/routes/messages.py buster_ripper/routes/count_tokens.py` -> 339 / 109; `codify scan path buster_ripper/routes/messages.py --detailed` -> `_stream_single_request` L:35 N:2 C:4, `_non_stream_single_request` L:34 N:4 C:8, `handle_messages` L:28 N:2 C:11, `handle_streaming` L:25 N:3 C:8 (all unchanged), `_request_payload` Green; `codify scan path buster_ripper/routes/count_tokens.py --detailed` -> `_read_body` Green; `git diff --check` -> clean; helper probe -> unchanged UTF-16 body relayed with its `charset=utf-16`, changed body reserialized `{"a": 1, "b": 3}` under `application/json`, `message: null` and `usage: []` both set `done` without raising; loopback wire probe -> a plain headers dict holding `content-type` and `Content-Type` reaches an aiohttp upstream as one value, the later one.
- Wrong witness: always-reserialize ruled out (the helper returns the client's UTF-16 bytes and the test asserts them); always-relay-raw ruled out (the normalized body reserializes under `application/json`); the error arm still relabelling ruled out (the 429 test asserts `Retry-After`, `X-Request-Id` and `text/plain; charset=utf-8`); `_note_message_start` still raising ruled out (the probe sets `done` on both wrong shapes and the test sees the following chunk); count_tokens still 500 ruled out (the three non-object roots answer the route's 400). Every new assertion was read against HEAD's `json=body`, `_error_body`/`content_type="application/json"`, unguarded `.get` chain and non-object return; all are red at those bytes except the unknown-charset witness, which pins behavior that was already green.
- Findings: none
- Noticed: `tests/test_messages_request_bytes.py`'s normalized-body assertion uses a capital `Content-Type` client key, while a real httpx client sends lowercase; the plain dict then holds two case-different names. A loopback aiohttp peer measured one value on the wire, the later `application/json`, so the result is correct. Ask: no — nothing wrong reaches the upstream.
- Noticed: `_request_payload` duplicates chat's `_chat_payload` (same eight lines); a shared `utils` helper would move Brief A's committed file, which is out of scope. Ask: no — both call sites use one shape and nothing behaves differently.

#### Brief C — image cache expiry (`2026-10-19-pr1-r4-image-expiry`)

- **Anchor**: lite brief `2026-10-19-pr1-r4-image-expiry`.
- **Members**: finding 5 (4317334078, `image.py:198`, cache expiry only on the write path).
- **Evidence**: [126](iterations/126-assumption-decide.md) §Family D; [comment 4317334078](https://github.com/AlexGS74/buster-ripper/pull/1#discussion_r4317334078).
- **Prerequisite**: none. Runs beside Briefs A and B.
- **Owning route**: `lite-forward` → `lite-yield`, one brief.
- **Separation reason**: different module and cause; no interference with the relay briefs.
- **Capacity shape**: `image.py` (255) and `server.py` are Green; the sweep mirrors `dump.poll_flush` / `spawn_dump_flush` /
  `_cancel_background_task` as a third instance of an existing pattern; no new CLI option.
- **Done when**: an image cached through `--strip-images` is gone after `IMAGE_CACHE_TTL` with **no further request**, driven by the
  periodic sweep; a source inside the TTL still resolves; `on_cleanup` cancels and awaits the sweep before closing the session; a
  non-positive TTL does not spin; the test is red before the sweep is spawned.
- **Family sweep**: `compaction._kv_cache_usage`, `session._stats`, `config.*` audited as safe in [126](iterations/126-assumption-decide.md)
  §Family D; `dump._sessions` is recorded as a Refactoring backlog row under the `RM-022` trigger, not repaired here.
- **Invariant cross-reference**: Neutral — zero active/wip registry invariants; `PI:T002/dump_format/documented_format_is_retained` is not
  touched (no writer, record or default moves).
- **Closure ledger**: `DC-PR1-R4-5-r1`.

##### Forward — `2026-10-19-pr1-r4-image-expiry`

- Changed: `buster_ripper/image.py` — `poll_expiry` sweeps `cleanup_expired_images()` every `max(IMAGE_CACHE_TTL, 1)` seconds, logging a failed sweep and continuing; `asyncio`/`logging` imported and the module docstring names the sweep (276 raw lines).
- Changed: `buster_ripper/server.py` — `spawn_image_expiry` stores the sweep task in app state as `image_expiry_task`, spawned in `on_startup` beside the other pollers and cancelled-and-awaited in `on_cleanup` before the session closes (170 raw lines).
- Changed: `tests/test_image.py` — new `TestPollExpiry`: one sweep tick drops a source past the TTL with no request, keeps one inside it, floors a non-positive TTL at 1 s, and logs a raising sweep then keeps looping.
- Changed: `tests/test_server_hooks.py` — new `TestSpawnImageExpiry` (task stored, `image.poll_expiry` spawned), `TestOnCleanupImageExpiry` (cancelled before session close) and a startup-wiring witness; the startup fixture stubs the new spawn.
- Ran: red check (`git stash push -- buster_ripper/image.py buster_ripper/server.py`, then `tests/test_image.py::TestPollExpiry`) -> 3 failed (`AttributeError: module 'buster_ripper.image' has no attribute 'asyncio'`); the server witnesses fail at import because the spawn symbol is the change.
- Ran: `uv run pytest tests/test_image.py tests/test_image_stability.py tests/test_server_hooks.py -q --no-cov` -> 81 passed.
- Ran: `uv run pytest -q` -> 571 passed, coverage 96.83%.
- Ran: `wc -l buster_ripper/image.py buster_ripper/server.py` -> 276 / 170.
- Ran: `codify scan path buster_ripper/image.py --detailed` -> 3 Medium Nesting (`make_data_url`, `_has_image_in_blocks`, `cleanup_expired_images`), the same set as HEAD; `poll_expiry` Green.
- Ran: `codify scan path buster_ripper/server.py --detailed` -> `on_startup` L:18 N:1 C:3 P:1 Medium Length (HEAD: L:17, same issue); `spawn_image_expiry` Green.
- Ran: `git diff --check` -> clean.
- Noticed: the sweep interval equals the TTL, so a source cached just after a tick can survive up to two TTLs before the next one. Ask: no — the bound is on absolute age, the entry still leaves with no request, and nothing is lost or written wrong.
- Noticed: `tests/test_server_hooks.py` emits pre-existing `RuntimeWarning`s from unawaited `AsyncMock` coroutines (the `dump.poll_flush` spawn test already does this at HEAD); no new warning kind. Ask: no — the tests pass and the warning is cosmetic.
- Noticed: `dump._sessions` still has no eviction; it stays the Family D backlog row under the `RM-022` trigger. Ask: no — recorded and out of scope.

##### Yield — `2026-10-19-pr1-r4-image-expiry`

- Verdict: ACCEPT
- Ran: `uv run pytest tests/test_image.py tests/test_image_stability.py tests/test_server_hooks.py -q --no-cov` -> 82 passed; `uv run pytest -q` -> 571 passed, coverage 96.83%; red check (`git stash push -- buster_ripper/image.py buster_ripper/server.py`, then `tests/test_image.py::TestPollExpiry`) -> **4 failed** (`AttributeError: module 'buster_ripper.image' has no attribute 'asyncio'`; the server witnesses fail at import because `spawn_image_expiry` is the change); `wc -l buster_ripper/image.py buster_ripper/server.py` -> 276 / 170; `codify scan path buster_ripper/image.py --detailed` -> the same three Medium Nesting (`make_data_url`, `_has_image_in_blocks`, `cleanup_expired_images`) as HEAD, `poll_expiry` Green; `codify scan path buster_ripper/server.py --detailed` -> `on_startup` L:18 N:1 C:3 P:1 Medium Length (HEAD L:17, same issue), `spawn_image_expiry` Green; `git diff --check` -> clean.
- Wrong witness: "expiry still only on the write path" ruled out by `server.py:105` spawning `spawn_image_expiry`, `server.py:56` creating `image.poll_expiry()`, and `image.py:271-276` sweeping `cleanup_expired_images()` before its sleep: the focused test caches through `strip_and_cache_images`, advances `image._now` past the TTL, runs one poll tick with no request at all, and the key is absent. "A source inside the TTL still resolves" ruled out by the 99/100 tick asserting `get_cached_image(...) == {"url": "data:1"}`. "on_cleanup closes the session before cancelling" ruled out by `server.py:113` preceding `:115` and the test recording `sweep.cancelled()` inside `session.close`'s side effect (`[True]`). "A non-positive TTL spins" ruled out by `image.py:276`'s `max(config.IMAGE_CACHE_TTL, 1)` and the recorded interval `[1]`. "A raising sweep ends the loop" ruled out by `image.py:272-275` logging and continuing, with one sweep, interval `100`, and the log record observed.
- Findings: none
- Noticed: the sweep interval equals the TTL, so a source cached just after a tick can survive up to roughly two TTLs before the next tick drops it; the entry still leaves with no request, the absolute-age comparator is unchanged, and nothing is lost or written wrong. Ask: no — memory retention only, and the brief's closure basis prescribes `max(IMAGE_CACHE_TTL, 1)`.
- Noticed: the brief's Capacity shape calls `server.py` Green, but `on_startup` was already Medium Length at HEAD (`L:17`, measured by scanning the stashed file); the change moves it to `L:18` with the same single issue and adds no new violation. Ask: no — no metric entered a new zone.
- Noticed: Forward's red check reports 3 failed; I measured **4 failed**, because all four `TestPollExpiry` cases touch the missing `image.asyncio` (the startup witness is separately red at import). Ask: no — every case is red for the intended reason; the count was understated.
- Noticed: `tests/test_server_hooks.py`'s startup fixture no longer resets `backend._backend_type`/`_backend_state_cached` before the yield (the teardown still resets both); the first two tests set the state via the CLI override and the third runs after that teardown, so the class stays isolated. Ask: no — no test in the class observes a stale pre-test state.
- Noticed: the full `test_server_hooks.py` file carries the same pre-existing unawaited-`AsyncMock` warning kind at HEAD and with the change (2 warnings with the change vs 3 at HEAD); the new tests introduce no new warning kind. Ask: no — cosmetic and fewer than HEAD.

### DC-PR1-R4-1-r1
Finding: PR1-R4-1 (PR #1 review comment 4317334042, P2 — `chat.py:29-35` non-object chat JSON → 500; family: the three JSON entry gates)
Evidence: https://github.com/AlexGS74/buster-ripper/pull/1#discussion_r4317334042; iterations/126-assumption-decide.md §Family sweep — every client-body parse site audited
Evidence fingerprint: unavailable
Repository state: git-commit:6a089d8d5a2bdd18eb8ece9de0653e1f89bf9aa3
Selected closure: validate the decoded chat body is a `dict` at the parse site and answer the route's own `400 {"error": "Invalid JSON"}` for any other JSON root, so `dict(body)` and `body.get("stream")` never see a non-mapping; carry the parse, the root gate and the raw-byte decision in one private helper so `handle_chat`'s statement count falls.
Closure basis: the route's authored 400 envelope (`chat.py:32`) is the uniquely ranked answer; the Reach Gate's "Gate invalid input at the entry" requires refusing every invalid kind of the body parameter at its entry; base parity (`buster_ripper_legacy.py:812-847`: `body = await request.body()`, `except Exception: pass`, raw forward) ranks forwarding above a 500 — the current crash is migration drift.
Rejected candidates:
- tolerate-and-forward the non-object body: leaves the proxy answering the upstream's rejection instead of authoring the client error the reviewer names, and `handle_chat`'s SGLang arm calls `dict(body)` before any relay.
- leave it and record an accepted risk: preserves a reachable 500 that base did not produce, and the fix is an entry gate.
- add the guard as two statements in `handle_chat`: triggers the preventive capacity gate on a function with three gray metrics (`L:29 N:2 C:6`); the helper is the gate's own extraction.
Risk shape: unchanged - one private parse helper replacing the existing four-line block; `handle_chat`'s statements fall by three, every transform and both request arms are byte-unchanged.
Deferred additive work: none.
Task: lite brief `00bab65:.sessions/lite/2026-10-19-pr1-r4-chat-relay/brief.md`, step 1

### DC-PR1-R4-2-r1
Finding: PR1-R4-2 (PR #1 review comment 4317334048, P2 — `messages.py:169-175` streaming error arm loses upstream headers and forces `application/json`)
Evidence: https://github.com/AlexGS74/buster-ripper/pull/1#discussion_r4317334048; iterations/126-assumption-decide.md §Family B — every error path audited; `DC-CR3-F-04-r1` rejected-candidate list
Evidence fingerprint: unavailable
Repository state: git-commit:6a089d8d5a2bdd18eb8ece9de0653e1f89bf9aa3
Selected closure: relay the upstream's raw bytes under `utils.response_headers(resp.headers)` on the streaming error arm, and read the overflow probe from `result.body.decode(errors="replace")` so an unknown charset cannot raise; delete `_error_body`, whose only caller was the arm, and shrink its 11-line comment above the probe.
Closure basis: `DC-PR1-R1-7-r1` settled the identical rule on the chat streaming sibling under the user's round-1 instruction; `DC-CR3-F-04-r1` explicitly deferred this arm's relabeling to "the §RM-005/relay-fidelity lane", which this round takes up; `T008.001` preserves the upstream's media type; the non-streaming (`:336-341`) and redirect (`:161-166`) arms already relay `response_headers`; measured `web.Response.text` raises `LookupError` on `charset=x-bogus`, so the probe must decode bytes.
Rejected candidates:
- keep `application/json` and add selected names: hand-keeps a list that drifts from the upstream and duplicates the shared filter's job.
- keep `_error_body` and relay raw bytes beside it: leaves a helper whose only purpose is a body no longer relayed; dead code is a change of its own.
Risk shape: unchanged - the error arm relays the upstream's status, bytes and end-to-end names instead of its decoded text; the success stream, the redirect arm and the non-stream arm are byte-unchanged.
Deferred additive work: none.
Task: lite brief `00bab65:.sessions/lite/2026-10-19-pr1-r4-messages-relay/brief.md`, step 1

### DC-PR1-R4-3-r1
Finding: PR1-R4-3 (PR #1 review comment 4317334057, P2 — `chat.py:138` 300 s total deadline stays armed across the SSE relay)
Evidence: https://github.com/AlexGS74/buster-ripper/pull/1#discussion_r4317334057; iterations/126-assumption-decide.md §Family C — every streaming call audited
Evidence fingerprint: unavailable
Repository state: git-commit:6a089d8d5a2bdd18eb8ece9de0653e1f89bf9aa3
Selected closure: make the per-request timeout `aiohttp.ClientTimeout(total=None if body.get("stream") else 300)`, so a streaming relay arms no deadline and a non-streaming request keeps base's 300 s bound.
Closure basis: `T018` step 2 settled the identical rule on `messages._stream_single_request` and records the reason ("a spelling that re-inherited the session's total would truncate a long stream in silence"); base's `httpx.AsyncClient(timeout=300)` applied to a non-streaming chat request, which still gets 300; round 1's post-commit catch (`chat.py:200-202`) turns the truncation into a normal EOF, which is the silent half the finding names.
Rejected candidates:
- two call sites (stream and non-stream): duplicates the body and a statement for no behavioral gain.
- no timeout at all for chat: loses the 300 s bound base had on the non-streaming request.
Risk shape: unchanged - one operand on the existing timeout construction; the relay, the retry loop and both response arms are byte-unchanged.
Deferred additive work: none.
Task: lite brief `00bab65:.sessions/lite/2026-10-19-pr1-r4-chat-relay/brief.md`, step 1

### DC-PR1-R4-4-r1
Finding: PR1-R4-4 (PR #1 review comment 4317334067, P2 — `chat.py:82-86` non-mapping eval `chat_template_kwargs` silently loses the injection after removing `max_gen_toks`)
Evidence: https://github.com/AlexGS74/buster-ripper/pull/1#discussion_r4317334067; iterations/126-assumption-decide.md §Family A
Evidence fingerprint: unavailable
Repository state: git-commit:6a089d8d5a2bdd18eb8ece9de0653e1f89bf9aa3
Selected closure: normalize a non-mapping `chat_template_kwargs` to a fresh dict before the nested `setdefault` calls, mirroring `messages._inject_preserved_thinking`'s guarded replacement, in a small helper that replaces the one `setdefault` statement.
Closure basis: `messages.py:73-80` already implements the exact rule ("a client-sent non-mapping is replaced, not skipped"); `T019.002`'s accepted shape for a non-mapping nested value; the route's own broad handler is what currently hides the loss.
Rejected candidates:
- leave the broad swallow: preserves the verified defect and forwards a partially transformed body.
- answer 400 for a non-mapping `chat_template_kwargs`: a new rejection no authority imposes, and the value is normalizable rather than invalid.
- add the guard as two statements in `_apply_request_transforms`: triggers the preventive capacity gate on a function with three gray metrics (`L:20 N:2 C:8`); the helper is the gate's own extraction.
Risk shape: unchanged - one guarded assignment replacing one `setdefault`; the eval profile, the budget and the response transforms are byte-unchanged.
Deferred additive work: none.
Task: lite brief `00bab65:.sessions/lite/2026-10-19-pr1-r4-chat-relay/brief.md`, step 1

### DC-PR1-R4-5-r1
Finding: PR1-R4-5 (PR #1 review comment 4317334078, P2 — `image.py:195-198` expiry sweep runs only on the image-write path)
Evidence: https://github.com/AlexGS74/buster-ripper/pull/1#discussion_r4317334078; iterations/126-assumption-decide.md §Family D — every cache audited
Evidence fingerprint: unavailable
Repository state: git-commit:6a089d8d5a2bdd18eb8ece9de0653e1f89bf9aa3
Selected closure: add a periodic expiry task mirroring `dump.poll_flush` — a loop that calls `cleanup_expired_images()` every `max(config.IMAGE_CACHE_TTL, 1)` seconds, spawned in `on_startup` and cancelled-and-awaited in `on_cleanup` through the existing `_cancel_background_task`, so expiry runs with no image-bearing request and no request at all.
Closure basis: `config.IMAGE_CACHE_TTL` advertises an absolute bound (`config.py:42`); `cleanup_expired_images`'s own docstring calls itself "the only place a session's own key leaves `_image_cache`"; the package already runs two periodic tasks through `spawn_kv_cache_monitor`/`spawn_dump_flush` and `_cancel_background_task` (`server.py:25-45`, `:86-106`); the reviewer names "a periodic task" as an admissible remedy; `G-002`'s clean-shutdown requirement ranks a cancelled-and-awaited task.
Rejected candidates:
- sweep on every `/v1/messages` request: does not cover image traffic stopping entirely and adds a statement to `handle_messages`, whose hard `C:11` triggers the capacity gate.
- sweep in the logging middleware: covers all routes but couples a logging concern to the image cache, and still does not run while the process is idle.
- leave it and record: leaves the advertised TTL broken for a cached base64 payload.
Risk shape: unchanged - one new coroutine and two lifecycle lines on the existing background-task pattern; the write path, the TTL comparator, the LRU bound and `get_cached_image` are byte-unchanged; no new CLI option.
Deferred additive work: none.
Task: lite brief `00bab65:.sessions/lite/2026-10-19-pr1-r4-image-expiry/brief.md`, step 1

### DC-PR1-R4-6-r1
Finding: PR1-R4-6 (PR #1 review comment 4317334102, P2 — `chat.py:141-146` always re-serializes the request body; family: the request direction of every route)
Evidence: https://github.com/AlexGS74/buster-ripper/pull/1#discussion_r4317334102; iterations/126-assumption-decide.md §Family E — every request direction audited
Evidence fingerprint: unavailable
Repository state: git-commit:6a089d8d5a2bdd18eb8ece9de0653e1f89bf9aa3
Selected closure: relay the client's cached raw bytes when the parsed body equals them, and otherwise reserialize and set `Content-Type: application/json`; carry the decision in one helper that reads `request.read()` (cached after `request.json()`) so `_make_chat_request` gains no sixth parameter.
Closure basis: base parity (`buster_ripper_legacy.py:812-847`: `body = await request.body()`, reserialized only inside the eval transform, forwarded raw otherwise); `DC-PR1-R1-3-r1` settled the response-direction rule ("relay the original bytes when no transform changed the body"); aiohttp copies a payload's own header only when the name is absent (`client_reqrep.py`, `update_body_from_data`), so `json=` cannot repair the client's stale `Content-Type`; measured UTF-16 → UTF-8 under `charset=utf-16` on `HEAD`.
Rejected candidates:
- preserve raw only when `EVAL_MODE` is off: leaves eval mode's no-op transform re-encoding, contrary to the finding's "even when eval mode applies no request transformation".
- add a sixth `raw` parameter to `_make_chat_request`: the function is near tolerance on Nesting and Parameters (`P:5`), and the capacity gate forbids growing it.
- leave it and record: base parity ranks the raw forward, and the fix is one helper plus one call-site expression.
Risk shape: unchanged - one helper and one call-site expression; the eval transforms, the SGLang retry loop and both response arms are byte-unchanged.
Deferred additive work: none.
Task: lite brief `00bab65:.sessions/lite/2026-10-19-pr1-r4-chat-relay/brief.md`, step 1

### DC-PR1-R4-A1-r1
Finding: PR1-R4-A1 (family sweep of 4317334042 — `chat.py:177-184` `_raise_chat_overflow` reads `.get` on a non-object upstream answer or a non-mapping `usage`)
Evidence: iterations/126-assumption-decide.md §Family sweep — every nested mapping access audited; measured 500 for `"Context overflow: 5000 > 4096"` and for `{"error": …, "usage": []}`
Evidence fingerprint: unavailable
Repository state: git-commit:6a089d8d5a2bdd18eb8ece9de0653e1f89bf9aa3
Selected closure: return early from `_raise_chat_overflow` when `resp_data` is not a mapping, and read `usage` only when it is a mapping, so a non-object upstream error answer relays instead of raising.
Closure basis: the route's relay contract (`chat.py:172-174` relays the upstream's status and body); `_chat_answer`'s documented non-JSON fall-through; the same class as finding 1 in the response direction.
Rejected candidates:
- leave it: an upstream error answer reaches the client as a proxy 500.
- catch the `AttributeError` in `_make_chat_request`'s arm: widens the transport arm to a body-shape failure and hides the class.
Risk shape: unchanged - one guard condition and one mapping check; the success relay, the transform path and the retry loop are byte-unchanged.
Deferred additive work: none.
Task: lite brief `00bab65:.sessions/lite/2026-10-19-pr1-r4-chat-relay/brief.md`, step 1

### DC-PR1-R4-A2-r1
Finding: PR1-R4-A2 (family sweep of 4317334042 — `messages.py:241-251` `_note_message_start` reads `.get` on a non-mapping `message`/`usage`, and `_relay_after_commit`'s catch ends the stream)
Evidence: iterations/126-assumption-decide.md §Family sweep — every nested mapping access audited; measured 200 with an empty body after a first chunk carrying the rest of the turn
Evidence fingerprint: unavailable
Repository state: git-commit:6a089d8d5a2bdd18eb8ece9de0653e1f89bf9aa3
Selected closure: guard the `message` and `usage` reads with `isinstance` and fall back to `0`, so a wrong-shaped `message_start` cannot raise out of `_note_message_start` and end the committed relay.
Closure basis: `_relay_after_commit`'s post-commit contract ("one head, one terminal") and `T013`; the same class as finding 1 in the silent-swallow form.
Rejected candidates:
- leave it: a valid-but-wrong-shaped upstream event truncates a live stream with no error to the client.
- catch it in `_relay_after_commit`: already happens, and that is the defect — the catch cannot distinguish a malformed event from a transport failure.
Risk shape: unchanged - two guarded reads in a small function; the relay loop, the stats write and the overflow arm are byte-unchanged.
Deferred additive work: none.
Task: lite brief `00bab65:.sessions/lite/2026-10-19-pr1-r4-messages-relay/brief.md`, step 1

### DC-PR1-R4-A3-r1
Finding: PR1-R4-A3 (family sweep of 4317334042 — `count_tokens.py:17-26` `_read_body` returns a non-object JSON root to `strip_billing_nonce`)
Evidence: iterations/126-assumption-decide.md §Family A — every client-body parse site audited; measured 500 for `null`
Evidence fingerprint: unavailable
Repository state: git-commit:6a089d8d5a2bdd18eb8ece9de0653e1f89bf9aa3
Selected closure: validate the decoded `count_tokens` body is a `dict` and answer the route's existing `400 {"error": "Invalid JSON"}` for any other JSON root.
Closure basis: the route's own authored 400 shape (`count_tokens.py:23-26`); the same entry gate as finding 1; base parity (base's `_session_id(body)` also 500s, so the route's authored shape is the ranked answer rather than base's crash).
Rejected candidates:
- return `{}` for a non-object root: silently answers an estimate for a body the route did not understand.
- leave it: preserves a reachable 500 with a traceback.
Risk shape: unchanged - one entry gate in a small function; the stats lookup, the nudge and the estimate are byte-unchanged.
Deferred additive work: none.
Task: lite brief `00bab65:.sessions/lite/2026-10-19-pr1-r4-messages-relay/brief.md`, step 1

### DC-PR1-R4-E1-r1
Finding: PR1-R4-E1 (family sweep of 4317334102 — `messages.py:155-158`, `:305-311` always re-serialize the request body; family: the request direction of every route)
Evidence: iterations/126-assumption-decide.md §Family E — every request direction audited; measured UTF-16 → UTF-8 under `charset=utf-16` on `HEAD`
Evidence fingerprint: unavailable
Repository state: git-commit:6a089d8d5a2bdd18eb8ece9de0653e1f89bf9aa3
Selected closure: use the same raw-bytes helper as the chat route at both `messages` request call sites, so an unchanged normalized body relays the client's bytes and a changed body reserializes with `Content-Type: application/json`.
Closure basis: base parity (`buster_ripper_legacy.py:645` re-serialized, so the raw direction is base's own rule for an unchanged body is not available — the rule is `DC-PR1-R1-3-r1`'s "original bytes when no transform changed the body" extended to the request direction); `normalize()` returns its own `changes` list; the finding names the request direction of every route.
Rejected candidates:
- preserve raw only when `changes` is empty but ignore `PRESERVED_THINKING`: incomplete, because that injection also mutates the body.
- add a `raw` parameter to `_stream_single_request`: the function is already a hard Length violation and a parameter is an addition.
Risk shape: unchanged - one helper used at two existing call sites; the streaming and non-streaming arms, the overflow retry and the error arms are byte-unchanged.
Deferred additive work: none.
Task: lite brief `00bab65:.sessions/lite/2026-10-19-pr1-r4-messages-relay/brief.md`, step 1

## Forward
- Changed: `buster_ripper/routes/chat.py` — shortened `handle_chat`, `_prepare_headers`, `_carries_token` and `_chat_answer` docstrings to one line each; shortened the two `_prepare_headers` comments; removed seven one-line docstrings and two inline comments; 300 → 272 raw lines, no import, statement, branch or signature moved.
- Ran: `.venv/bin/python -m pytest tests/test_context_overflow.py tests/test_header_parity.py tests/test_routes_integration.py tests/test_upstream_answer_relay.py -q -p no:cacheprovider --no-cov` -> 182 passed
- Ran: `.venv/bin/python -m pytest -q -p no:cacheprovider` -> 540 passed, coverage 96.78%
- Ran: `git diff -U0 -- buster_ripper/routes/chat.py` -> every changed line is a comment or docstring line; `wc -l` -> 272
- Noticed: the brief's `:188-203` range names no comment or docstring on these bytes (`_relay_chat_stream` has a single-line docstring and code follows); the target was reached from the other named blocks. Ask: no — no behavior or review outcome depends on the stale range.
- Noticed: the two removed inline comments (`# Non-SGLang backends: no retry`, `# SGLang: retry on context overflow`) restated the adjacent branches and retry loop; their point is still visible in the code. Ask: no — comment-only, no production result changes.

VERDICT: COUNTEREXAMPLE
PROMOTION: INELIGIBLE — the "permanent handles for all member claims" condition fails: every one of the 19 member rows carries `Permanent: —` in `claims.md` (session verdicts only, no registry `Claim:` record) and S12's own `Permanent` cell is `—`. Separately, member 316's recorded aggregate digest does not reproduce from the current bytes (see §Member-fingerprint audit), which is a per-member promotion condition, not a composition one.

## Seam

S12 — the detector boolean and the user-message content list crossing `--strip-images` →
`config.STRIP_IMAGES` → either guarded call site → detector → writer → the body forwarded upstream and
the changes log. Members 253, 254, 255, **332**, 257, 258, 259, 260, 261, 262, 264, 265, 268, 269, 270,
271, 290, 316, 317 (`seams.md` line 18; `State: new`, `Routing: malformed`, `Permanent: —` — cells are the
owning route's, untouched by this pass).

This is the re-run the prior record asked for (`compositions/seam-S12.md`, `MALFORMED`, §"What is required
before S12 can be composed"): retired 256 is replaced by 332, the member set is the same 19, and the full
trace was performed this time.

## Preconditions checked

All 19 verdict files were read from `verdicts/`; **every one begins `VERDICT: HOLDS`** (253, 254, 255, 332,
257, 258, 259, 260, 261, 262, 264, 265, 268, 269, 270, 271, 290, 316, 317), and the `claims.md` rows agree
(`State: HOLD`, `Routing: none`) at lines 868-886, 905, 931-932 and 1105. No member was omitted, and no
`DOES-NOT-HOLD` or `UNCLEAR` row was smuggled in as a premise. 332's proposition
("…considers every message whose `role` is `user`, in forward list order") is the visit half of retired 256's
fork; the rewrite half is carried by member 258 (`HOLD`), per `claims.md` line 1084 — so the fork that made
the prior pass malformed is closed at row level, not papered over here.

### Member-fingerprint audit

Recomputed against the current worktree bytes: `buster_ripper/image.py` `611df5ac…`,
`buster_ripper/routes/messages.py` `721a2afe…`, `buster_ripper/normalize.py` `a7ddbf06…`,
`buster_ripper/main.py` `fd36dc12…`, `buster_ripper/config.py` `22348c68…`, `tests/test_image.py` `b603bfd1…`,
`tests/test_routes_integration.py` `365aa9a7…` — **all match** the digests recorded in 253-262, 264-265, 268-271,
290, 317. No member is stale.

One discrepancy: row 316 records its 26-file `tests/` aggregate as `sha256:3c15681d3f6e38fcf9db20a3fd66b0dabcb7f06241457cd227cd3091e58d0372`.
Applying its own stated recovery recipe to the current bytes yields
`sha256:fc6b6189161b4a6429d13b795fba0b0c8839ecc372fa781dd1c9279975b71c5e` over the same 26-file set (and no
file under `tests/` has an mtime later than that verdict's `Checked At`). Variants (unsorted `find` order,
path forms, header forms) reproduce none of them, so 316's *recorded* digest is unreproducible — a bookkeeping
defect in that member's handle, not evidence that its proposition went false. Its substance was re-verified
directly for this trace: `grep -r "STRIP_IMAGES" tests/` returns no match, and the token `image` appears in no
test file outside `tests/test_image.py`. 316 is therefore used as a premise here and its digest defect travels
to the registry route, the way S11's missing-digest item did.

## Composition

Producer → transformations → consumer, as the 19 members joint it (all locators are production sources):

1. **Flag.** `main.py:74-75` declares `--strip-images` as a plain `is_flag=True` option whose `help` reads
   "Strip images from user messages and cache them for later retrieval." (269). No callback, value exposed →
   `main()` parameter (`main.py:132`) → `_apply_cli_config` → `config.STRIP_IMAGES = cli["strip_images"]`
   (`main.py:44`, 270). The process default is `False` (`config.py:40`, 271), so the whole downstream chain is
   dead unless the flag was given.
2. **Two guarded call sites, one live.** The package contains exactly two `has_images` call sites and two
   `strip_and_cache_images` call sites (verified by grep over `buster_ripper/`): `routes/messages.py:38-40` and
   `normalize.py:212-214`. The `normalize` site additionally requires a truthy `session_id` (264), and the
   package's only `normalize()` call passes `session_id=""` (`routes/messages.py:34`, 268) — so `normalize`'s
   step 0 never runs in production, never reaches `has_images` (Python short-circuits the `and` at
   `session_id`), and its `changes.append` (265) has no live producer. The route site is the only reachable
   gate: `if config.STRIP_IMAGES and image_handler.has_images(body):`.
3. **Detector.** `has_images(body)` (253, 254, 255): `body.get("messages", [])`, falsy → `False`; otherwise
   `for msg in reversed(messages)`, skip non-`"user"`, and on the **first user message met** evaluate
   `isinstance(content, list) and _has_image_in_blocks(content)` and then `break` **unconditionally**
   (`image.py:35-42`). Block predicate: `isinstance(block, dict) and block.get("type") == "image"`
   (`image.py:20`). So the boolean means exactly "the **last** user message has a top-level `type == "image"`
   dict block".
4. **Writer.** `strip_and_cache_images(body, sid)` (332, 257, 258, 259, 260): initializes
   `_image_cache[session_id]` (`image.py:121-122`), early-returns on absent/empty `messages`
   (`image.py:124-126`), then `for msg in messages:` **forward, over every user message** (332), skipping
   non-`"user"` (257) and non-`list` content (258), and reassigning `msg["content"]` for **every** list-content
   user message (258's note: even an image-free or empty list is rebuilt) using the same block predicate at
   `image.py:98`. Nothing anywhere in the module descends into a nested `content` (260).
5. **Consumers.** `routes/messages.py:39-40` mutate `body` in place and append one `"stripped and cached
   images"` string inside the same branch (262); that `body` is what `dump_turn(body)` sees at line 57 and what
   is forwarded upstream at lines 67-70; `changes` is rendered only as `"; ".join(changes)` in the VERBOSE log
   line at 52-53.

**The composed property:** the predicate that *permits* the mutation is computed over a strictly smaller
message set than the mutation itself. Gate scope = one message (the last user message). Effect scope = all
list-content user messages, with no positional qualification surviving the boundary.

## Dimension Coverage

The prior record traced **none** of these (its `MALFORMED` reason), so nothing was carried forward and all ten
were re-traced in full on this pass; the change that fired the re-run is the member swap 256 → 332
(`seams.md` line 326), which participates in dimensions 1, 3, 5 and 8 — and since the prior trace was empty,
"re-trace the dimensions the changed member participates in" collapses into a whole-seam trace.

1. **message position (earliest, middle, latest)** — re-traced. **Counterexample CE-1.**
2. **role (user versus assistant)** — re-traced. Consistent: both halves use `msg.get("role") != "user"`
   (`image.py:36`, `image.py:129`; 253, 257, 332), so an image in an assistant/system/role-less message is
   never gated on and never stripped, and the help text's own subject is "user messages" (269) — no conflict.
3. **content shape (list versus string) — 332 read against 257/258** — re-traced. The visit/rewrite fork that
   made 256 indeterminate is settled by two rows that do not contradict each other: 332 (visits all user
   messages, forward) and 258 (rewrites only list-shaped content). Both are compatible with the writer's
   mutation scope being a strict subset of its iteration scope, and 258's non-list skip is *strengthened*, not
   weakened, by the detector's own `isinstance(content, list)` test at `image.py:40`: a string-content latest
   user message makes the gate False and the writer unreachable. No counterexample on shape alone; shape feeds
   CE-1.
4. **absent or empty `messages`** — re-traced. No divergence: `has_images` returns False first (254), so the
   writer's own early return (259) and its pre-guard cache-entry initialization (`image.py:121-122`) are
   unreachable through either guard. Adjacent case (non-empty non-list `messages`) recorded as an observation,
   not a finding — see §Adjacent.
5. **image nesting (top-level block versus `tool_result`-nested)** — re-traced. **Counterexample CE-2**
   (documentation-scope only, no internal mismatch).
6. **guard ownership and reachability** — re-traced. No counterexample: only the route site is reachable
   (264 + 268), the two guards are the only writer call sites in the package, and a *duplicate* settlement is
   impossible even if the `normalize` site were made live, because after a strip the last user message's
   content holds only `{"type": "text", …}` placeholders, so the second `has_images` is False. Ordering hazard
   recorded under §Adjacent so the equivalence is not read as safety.
7. **the changes-string append as a side effect of the same branch** — re-traced. Traced as part of CE-1's
   observable effect: the append cannot fire without ≥1 real replacement (the gating message is itself in the
   writer's mutation scope), and it fires exactly once however many user messages were rewritten (262 +
   `messages.py:40,52-53`).
8. **the help text's stated scope versus the detector's scope** — re-traced. This is where CE-1 and CE-2 land.
9. **the characterization test that pins today's narrower reading** — re-traced. `test_has_images_checks_last_user_message`
   (`tests/test_image.py:49-58`, docstring "Should only check the last user message") asserts False for exactly
   CE-1's body A. It pins the **detector alone** — nothing in the file composes it with the writer or the
   guard — so the repo contains two intent-bearing production artifacts with different scopes: 269's
   position-unqualified help text and 290's last-message-only pin. Which one is authoritative is outside this
   role; named under §Counterexample.
10. **absence of any flag-enabled or route-level witness** — re-traced. 316 + 317 compose to: the route
    integration harness does drive `/v1/messages` end to end with the upstream mocked, but no test source ever
    sets `config.STRIP_IMAGES` true, and every writer call in `tests/test_image.py` invokes
    `strip_and_cache_images` **directly**, bypassing `has_images` and the guard (all twelve call sites —
    `tests/test_image.py:75,93,108,109,123,148,169,190,211,228,237,257`; the file contains no route call).
    The producer→gate→detector→writer→forwarded-body chain therefore has no witness at any level that joins its
    two halves; CE-1's divergence is invisible to the suite in the composed direction.

## Counterexample

### CE-1 — a user message's images are stripped or forwarded depending on a *later* message's content, while the flag's own help text is unqualified about position

Two requests to `POST /v1/messages` on a server started with `--strip-images` (so `config.STRIP_IMAGES` is
`True` per 270/271), with the same `system` value and the same `messages[0]`. `session_id(body)` derives the id
from the **system prompt alone** (`buster_ripper/session.py:93-98`, imported at `messages.py:14`, called at
`messages.py:35`), so both requests resolve to the same `sid` and therefore to the same
`_image_cache[sid]` bucket.

**Body A**
```json
{"messages": [
  {"role": "user",      "content": [{"type": "image", "source": {"data": "<X>"}}]},
  {"role": "assistant", "content": "ok"},
  {"role": "user",      "content": [{"type": "text", "text": "and now?"}]}
]}
```
**Body B** — identical apart from the third message's content list, which also carries
`{"type": "image", "source": {"data": "<Y>"}}` (so `messages[0]` and `system` are byte-for-byte the two requests).

Path through the verified claims: flag → `config.STRIP_IMAGES` (270, 271) →
`normalize(body, session_id="")` short-circuits its image step and the route's `normalize` call is the only one
(264, 265, 268) → `messages.py:35` computes `sid` → `messages.py:38` guard calls `has_images(body)`.

- **A:** `has_images` walks `reversed(messages)`, skips the assistant message, meets message 2 first, tests it,
  and `break`s without ever examining message 0 (253, 255). Result `False` → the conjunctive guard is false →
  `strip_and_cache_images` is never called (261) and `"stripped and cached images"` is never appended (262).
  **Observable:** image block `X` is forwarded upstream verbatim in the body at `messages.py:70`; `_image_cache[sid]`
  gains no entry for it (while B puts one there under the same `sid`); when a dump directory is configured the raw
  base64 block is also what `dump_turn(body)` at line 57 records; and the VERBOSE changes line at 52-53 says
  nothing about images.
- **B:** `has_images` is `True` on message 2 → writer called → the writer iterates **all** user messages
  forward (332), passes the role and list guards for messages 0 and 2 (257, 258), and rewrites both — `X` and
  `Y` are replaced by text placeholders and cached (two entries under `_image_cache[sid]`); the append fires
  once (262).

The conflict: message 0 is the same bytes in both requests, and in both the flag is on and message 0 is a user
message containing an image — yet its image is stripped-and-cached in B and forwarded raw in A. The gate's
qualification ("*the last* user message") is discarded before the mutation, and the only operator-visible
signal of the whole boundary is one undifferentiated string (dimension 7), which in A is absent. Under B, the
same single string stands for two rewritten messages, so the log cannot distinguish "the gating message was
stripped" from "every list-content user message was rewritten". 269's help text — the flag's user-facing
statement — says the flag strips images from user messages and caches them; for body A it does neither.

**What this pass does not decide:** whether the correct repair is widening the detector, narrowing the writer,
or re-wording the help text. That is the missing contract, and it is named in 290's opposite direction — the
suite asserts the narrow detector reading as intended ("Should only check the last user message"). The
counterexample is that the 19 verified members jointly admit a concrete input where the documented scope, the
pinned detector scope, and the writer's actual scope are three different message sets; choosing among them is
the owning route's, not a composer's.

### CE-2 — `tool_result`-nested images: the flag is a total no-op, contradicting the same help text with no internal mismatch

```json
{"messages": [
  {"role": "user", "content": [
    {"type": "tool_result", "content": [{"type": "image", "source": {"data": "<X>"}}]}]}
]}
```
`has_images` examines the last user message's top-level blocks only (253, 255): the single block's `type` is
`"tool_result"`, not `"image"` → False → nothing stripped, nothing cached, no changes string. Had the writer
been called it would have returned that block unchanged (260, `image.py:98-100`). Detector and writer **agree**
here, so there is no misrouting between the two halves; the only conflict is 269's unqualified "Strip images
from user messages" against behavior that reaches no image below the top level of a user message.

## Adjacent (traced, named, not adjudicated here)

- **Shared block predicate — the divergence is positional only.** `image.py:20` (detection) and `image.py:98`
  (stripping) carry the *same* literal predicate, so the two halves cannot disagree about which blocks are
  images (255). That equivalence is structural duplication, not a single helper: a change to one predicate that
  misses the other (e.g. accepting `"image_url"`, which today is not an image on either path) would create a
  detector/writer divergence the current member set does not cover.
- **No stale-selection window at the gate.** `has_images(body)` (`messages.py:38`) and
  `strip_and_cache_images(body, sid)` (`messages.py:39`) read the same object with no intervening statement,
  `sid` is computed before both (`messages.py:35`), and `normalize` cannot have stripped anything first
  (264/268) — so the message the detector tested is necessarily one the writer rewrites, and the cache key
  cannot shift between gate and settlement.
- **Ordering hazard if the `normalize` guard is ever made live.** `normalize.py:212-214` is step 0, ahead of
  `_normalize_cache_control` (step 3, `normalize.py:124-138`) and `_normalize_date` (step 4, 141-196). Stripping
  first replaces image blocks with freshly built text dicts, so a `cache_control` key that sat on an image block
  would be discarded un-counted and the `stripped cache_control from N message field(s)` string would change for
  the same input. Not current behavior (268); recorded so "the two guards differ only by `session_id`" is not
  read as "either site can own the call".
- **A non-list `messages` value turns the flag on into an uncaught exception.** `{"messages": {"k": 1}}` is
  valid JSON; `normalize` skips it (`normalize.py:133`, `normalize.py:187` are `isinstance(..., list)` checks),
  so with the flag **off** it is forwarded untouched, while with the flag **on** `has_images` iterates the
  mapping's keys and raises `AttributeError` at `image.py:36`, which propagates out of `handle_messages` (the
  `try` at `messages.py:26-29` covers only the JSON read). 254 records the `AttributeError` case as *outside*
  its claim, and the seam row's dimension is "absent or empty `messages`", so this sits just past the enumerated
  set. Whether body-shape validation belongs to the proxy is S7's parse-boundary question; named, not ruled.

## Sources

Production sources only; no plan, interview, handoff, iteration or QA artifact was opened, nothing was
executed, no test was run, and no `.git` path was read.

- `/Users/alexhubitski/Projects/buster-ripper/buster_ripper/image.py` — `_has_image_in_blocks` 17-22,
  `has_images` 25-44 (`reversed` 35, role skip 36-37, list test 40, `break` 42), `_replace_image_with_placeholder`
  78-93, `_process_content_block` 96-100, `_process_user_content` 103-105, `strip_and_cache_images` 108-138
  (cache init 121-122, empty guard 124-126, forward loop 128, role guard 129-130, list guard 132-134,
  reassignment 136).
- `/Users/alexhubitski/Projects/buster-ripper/buster_ripper/routes/messages.py` — imports 11-14,
  `handle_messages` 24-70 (json `try` 26-29, `normalize(..., session_id="")` 34, `sid` 35, guard 38, writer 39,
  append 40, VERBOSE log 52-53, `dump_turn` 55-57, dispatch/forward 65-70).
- `/Users/alexhubitski/Projects/buster-ripper/buster_ripper/normalize.py` — 100-138, 141-196,
  `normalize` 202-228 (docstring 205-206, step-0 guard 212, writer 213, append 214, steps 1-4 217-226).
- `/Users/alexhubitski/Projects/buster-ripper/buster_ripper/main.py` — `_apply_cli_config` 36-50
  (`config.STRIP_IMAGES` 44), `--strip-images` option 74-75, `def main(` 131 with the `strip_images`
  parameter on 132, `_apply_cli_config(` at 139 with `strip_images=strip_images` on 140.
- `/Users/alexhubitski/Projects/buster-ripper/buster_ripper/config.py` — 40.
- `/Users/alexhubitski/Projects/buster-ripper/buster_ripper/session.py` — 93-98 (`session_id` derives from the
  system prompt alone; read only to establish that two bodies differing in `messages` share an `sid`, because
  `messages.py:14,35` — inside 261's cited range — imports and calls it).
- `/Users/alexhubitski/Projects/buster-ripper/tests/test_image.py` — 49-58 (the 290 pin), and the
  direct-writer call sites listed in dimension 10.
- `/Users/alexhubitski/Projects/buster-ripper/tests/test_routes_integration.py` — consulted for 317's shape.
- `tests/**/*.py` (26 files) — swept for `STRIP_IMAGES` / image-flag mechanisms (316).
- Session materials: `assumptions/seams.md` (S12 row line 18; rationale 252; run-32 270-272; run-33 326-329),
  `assumptions/claims.md` (rows above), the 19 `assumptions/verdicts/claim-*.md` files, and the prior
  `assumptions/compositions/seam-S12.md`.

## Source Versions

No source carries a version or revision string, and repository revision metadata is out of scope for this role.
Stability rests on the byte digests below (the same basis every member verdict used). `pkg:click` and
`pkg:aiohttp` identities were not needed — no member of this seam is a `pkg:` subject.

## Source Fingerprint

`sha256:32e7cd17f51d74be212fa87e05cea1dd43e4af9c8d089bd1594131c213db51d5`

Recoverable as: sha256 over the concatenation, in this order, of —
`"SEAM S12 members\n"`; then for each row in the seam's member order
(253, 254, 255, 332, 257, 258, 259, 260, 261, 262, 264, 265, 268, 269, 270, 271, 290, 316, 317) the line
`CLAIM <row> verdicts/claim-<row>.md <sha256 of that verdict file's bytes>\n` followed by one
`FP <line>\n` for each line of that verdict file that begins with `sha256:` (verbatim, whitespace-stripped);
then `"SOURCES\n"`; then for each consulted source in the §Sources file order — the six `buster_ripper/`
modules (`image.py`, `routes/messages.py`, `normalize.py`, `main.py`, `config.py`, `session.py`) then the 26
`tests/` Python files sorted by path, from `find tests -type f -name '*.py' -not -path '*/__pycache__/*'` —
the line `SOURCE <relpath>\n` followed by that file's raw bytes.
The current `tests/` aggregate under row 316's own recipe is
`sha256:fc6b6189161b4a6429d13b795fba0b0c8839ecc372fa781dd1c9279975b71c5e`.

## Invalidates On

- Any change to `buster_ripper/image.py` away from `611df5ac…` fires 253, 254, 255, 332, 257, 258, 259, 260 —
  i.e. dimensions 1, 2, 3, 4, 5 — and reopens both counterexamples: notably moving `image.py:42`'s `break`
  inside the `if` (or reversing the loop) removes CE-1's gate/mutation mismatch, and any recursion or a
  `tool_result` branch in `image.py:17-22`/`96-100` removes CE-2.
- Any change to `buster_ripper/routes/messages.py` away from `721a2afe…` fires 261, 262, 268 — the guard
  ownership, the append's branch, and the route-level `normalize` argument — and re-traces dimensions 6 and 7.
- Any change to `buster_ripper/normalize.py` away from `a7ddbf06…` fires 264, 265, 268; a change that makes
  step 0's `session_id` conjunct satisfiable from production converts the §Adjacent ordering hazard into a live
  dimension-6 finding.
- Any change to `buster_ripper/main.py` (`fd36dc12…`) in the `--strip-images` `help=` string or the line-44
  assignment fires 269/270 and re-traces dimension 8 (the help-text half of CE-1 and all of CE-2); any change to
  `buster_ripper/config.py` (`22348c68…`) line 40 fires 271.
- `buster_ripper/session.py` (`67a66b8a…`) is a corroborating read, not a member subject: CE-1's cross-request
  comparison relies on `session_id(body)` deriving the id from `system` alone. If it ever folds `messages` into
  the digest, A and B land in different cache buckets — the gate/mutation mismatch stands, but the shared-`sid`
  sentence in §Counterexample would need re-tracing, and no member verdict here would fire.
- Any change to `tests/test_image.py` (`b603bfd1…`) fires 290 (dimension 9); any change to `tests/` membership
  or contents fires 316 (the aggregate must be re-digested — its recorded `3c15681d…` does not reproduce today
  and needs a checker's own re-issued handle) and any change to `tests/test_routes_integration.py`
  (`365aa9a7…`) fires 317 (dimension 10).
- A test that sets `config.STRIP_IMAGES` truthy, or that routes an image body through `/v1/messages`, falsifies
  316 or 317 and re-opens dimension 10 — the absence that makes CE-1 unobservable to the suite is load-bearing.
- Any edit that gives `image.py:20` and `image.py:98` different block predicates (deduplicating them into one
  helper, or widening one type check) requires a fresh atomic row: no member of this set states the *equality*
  of the two predicates, only each one's own shape (255).
- Any new `normalize(...)` or `has_images(...)` or `strip_and_cache_images(...)` call site under
  `buster_ripper/` invalidates 261/264/268's exclusivity and re-traces dimension 6 from scratch.

## Checked At

2026-10-05T21:43:41Z

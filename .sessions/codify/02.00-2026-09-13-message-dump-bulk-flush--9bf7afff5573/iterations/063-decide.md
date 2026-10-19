# 063 — decide (RED-FINDINGS mode — 061's five AMBIGUOUS rows, `T014` image-slice wave)

## Orchestrator Prompt (verbatim)

> You are a subagent spawned by another session.
> Session ID: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
> Mode: red-findings
> Reports: .sessions/codify/02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/iterations/061-forward-red.md
>
> Execute the decide-interactive skill for this session. Arguments: $ARGUMENTS

## Mode and inputs read

`Mode: red-findings`. Read the named Red report (`061-forward-red.md`), the forward-visible artifacts it attacks
(`handoff-context.md` §Planned Task Context → the `T014` `<details>` block `:371-482`, §Deterministic Closure Ledger —
2026-10-05 routing (`DC-F-1-r1` `:518-530`, `DC-F-2-r1` `:532-545`, `DC-F-3-r1` `:547-560`), §Question, note-disposal,
and wave schedule `:664-696`, §Current Status `:24-52`; `plan-summary.md` §Implementation Plan Summary `T014` row
`:367` + its `<details>` restatement `:369-387`, §Anchor Map `:108-134`, §Goal Definition Matrix `G-003` `:73`,
§Reuse Evidence), and only the source needed to verify its rows: `buster_ripper/image.py`, `tests/test_image.py`,
`routes/messages.py:24-62`, `normalize.py:95-138, 198-228`, `session.py:88-98`, `config.py:36-42`,
`main.py:70-80, 122-132`, `pyproject.toml:26-36`, `README.md:1-8`. `codify-deterministic-closure` applied to every row
before its disposition. No unrelated plan choice reopened; no settled decision re-decided.

**Prior settled decisions checked, not reopened:** `grep -niE "strip.?images|placeholder|image.?cache|img_[0-9]"
decisions.md` → **zero matches** across `US-001`…`US-016` and `FK-001`…`FK-011`. No user decision ranks any of these
five rows, so every ranking below rests on approved plan text (closure authority rank 3) plus verified code evidence
(rank 4) — and on nothing else.

**Freshness:** the packets record repository state `git-commit:25c2b77…`; current `HEAD` is `51530da…` and `25c2b77`
is its ancestor. Both subject files are byte-identical between them — `git rev-parse 25c2b77:buster_ripper/image.py`
= `9dab3461…` = working tree, and `tests/test_image.py` `4c990f64…` = working tree — so every row's code premise is
current, not stale at the reviewed head.

**Ownership:** nothing below is *applied* by this pass. Repairs are authorized exactly and handed to **Authority
Registrar** (sole writer for plan repairs and every invariant-registry change). Only this report was written: no
`plan-summary.md` edit, no invariant edit, no Fork reserved (zero `user-fork` rows → `decisions.md` and its handoff
index untouched).

## Disposition table

| # | Red row / surface | Disposition | Closure | Citation or exact repair | Reason (one sentence) |
|---|---|---|---|---|---|
| 1 | **A-1** — F-2's hash referent: `block["source"]` vs the whole image block, ± `session_id`, byte form unstated (`:427-428`, `:435`, `:439` × `plan-summary.md:378`) | **deterministic-repair** | absorbed | **R-A** | Content-addressing `block["source"]` with an order-independent serialization is the only candidate that keeps the key a function of the bytes the cache actually stores and returns, and a whole-block digest is provably per-request unstable on one of the two guarded call sites (`normalize.py:212` runs the image step **before** `_normalize_cache_control` at `:223`, so the block still carries `cache_control` there, while `messages.py:34` → `:223` strips it first — verified `normalize.py:100-112`). |
| 2 | **A-2** — F-3: the prune/drop's position relative to the writer's placeholder writes (`:471-477`) | **deterministic-repair** | absorbed | **R-B** | Of the three placements Red enumerates, the one between the `image.py:121-122` ensure and the `:84` write raises `KeyError` out of `messages.py:39` (no `try` there — the only handler-level `try` is `:26-29` for JSON), which knives in a new request failure, and the two safe ones make the mandated "session dict dropped when it empties" clause dead or unobservable, so only a sweep positioned **after** the message loop discharges the required behavior without a new failure mode. |
| 3 | **A-3** — F-2×F-3: refresh-on-resend vs insert-on-miss ⇒ idle vs absolute TTL, and what `_evict_by_lru` sorts on (`image.py:72, 84`) | **deterministic-repair** | absorbed | **R-C** | The approved required-behavior text says retained base64 is "**bounded by** `config.IMAGE_CACHE_TTL`" (`:469-470`, `plan-summary.md:379`), and idle-since-last-send leaves that bound unbounded for exactly the multi-turn shape F-2 dedupes — while absolute age costs nothing on the prefix axis because post-R-A the key is content-derived, so re-caching after an expiry emits byte-identical placeholder text. |
| 4 | **A-4** — F-1: every mandated witness puts the image in the **earliest** user message, so an earliest-only detector passes them all (`:409-412`, `:413-414`, `DC-F-1-r1 :523`) | **deterministic-repair** | absorbed | **R-D** | The intent wording is not in dispute; the acceptance set is — verified that `tests/test_image.py` holds no shape with a text-only **first** user message and an image on a later turn, so nothing observable separates "all user messages" from "first" and the fix is to bind the witnesses as a position case table. |
| 5 | **A-5** — F-3: a global reclamation claim discharged by a session-scoped mechanism (`:469-470` vs `:475-476`; `plan-summary.md:379`) | **deterministic-repair** | absorbed | **R-B + R-E** | The finding's own evidence is the cross-session axis (`session_id = sha256(system)[:12]`, `session.py:93-98`, verified: "each distinct system prompt adds an `_image_cache` key that is never reclaimed", `:456-457`) and `--image-cache-max-size` says "per session" in its help while `--image-cache-ttl` says no such thing (`main.py:125-128`), so a session-scoped-only prune preserves part of the reported defect — and the remediation already exists as the entirely caller-less global janitor `cleanup_expired_images()` (`image.py:176-179`, zero callers in `buster_ripper/` **and** zero in `tests/`). |

`DECIDE-RED` arithmetic: **5 rows | 0 invalid | 5 deterministic | 0 user forks | 0 plan breaks.**

No row is a `plan-break`: every repair stays inside `image.py` + `tests/test_image.py`, moves no `MS-*`/`QAP-*` gate,
and none widens the wave's surface. No row is `user-fork`: each alternate reading is excluded by the approved
required-behavior text or by a verified counterexample, and the surviving distinctions (canonical serialization form,
helper placement, digest length) are mechanism, not outcome.

## Closure packets

### Packet 1 — Row 1 / A-1 (the placeholder key's referent)

```text
CLOSURE: absorbed
Finding: 061 row 1 (handoff-context.md:427-428, :435, :439 × plan-summary.md:378)
Required outcome: one image source ⇄ one placeholder key, identical on every request that carries it, at both
  guarded call sites — the product contract README.md:3-5 states and DC-F-2-r1 cites as its basis.
Authority:
  - §Member F-2 Required behavior (handoff-context.md:432-434): "one image source ⇄ one placeholder key, identical
    on every request that carries it … so normalization must be content-deterministic"
  - Verified the cached value and the returned value are the source dict: image.py:80 `source = block.get("source",
    {})`, stored at :84, returned by get_cached_image at :160-161 — so "one image source maps to one placeholder"
    is a statement about those bytes
  - Verified the instability counterexample: normalize.py:212 strips images BEFORE _normalize_cache_control at
    :223, so at that call site the block still carries cache_control (removed by _strip_cache_control_from_content,
    normalize.py:106-108); on the live path messages.py:34 runs normalize(session_id="") first, so by messages.py:38
    the block is {"type":"image","source":…}. A block-level digest therefore yields per-turn-varying keys on one
    call site and stable keys on the other — i.e. the two sites disagree about the same image, which is F-2's
    defect class relocated rather than fixed.
  - plan-summary.md:133-134 "The `Mode: NEW` rows above remain the only authorization": binds the derivation
    inside the existing writer, not a new key abstraction (Red row 10 already excludes that)
Admissible candidates:
  (a) key = img_<sha256(canonical(source))[:12]> — canonical = order-independent JSON serialization
      (json.dumps(source, sort_keys=True, separators=(",", ":"))), no session_id
Rejected candidates:
  - whole-block digest -> fails the required outcome at normalize.py:212 (volatile sibling fields), so it preserves
    the verified defect; its live-path equivalence is an accident of call-site ordering, not a property (closure
    step 3)
  - repr(source)/insertion-order-dependent form -> two semantically identical sources with different key order
    hash differently, which is the same failure under a different name
  - session_id inside the digest -> contradicts the unqualified "on every request that carries it" (:428) and buys
    nothing, the cache being already keyed by session_id at :84; a per-session digest would make one image render
    different text in two sessions with no claim requiring it
  - user-fork (which referent) -> the instability counterexample plus the stored-value fact ranks (a); mechanism
    residue (digest length, separators, ensure_ascii) changes opaque bytes only, not an outcome (closure step 8)
Selected closure: (a), bound as R-A with a discriminating witness table so the call-site ordering cannot leak into
  the key by any future reordering.
Risk shape: unchanged — same function, same insertion point; the only new obligation is two extra test shapes.
Deferred additive work: none
Owning route: Authority Registrar (R-A) -> Forward T014 (F-2)
Question required: no
```

### Packet 2 — Rows 2 + 5 / A-2 + A-5 (where the prune sits, and what it covers)

```text
CLOSURE: absorbed
Finding: 061 rows 2 and 5 (handoff-context.md:469-470, :471-477, :479-480; plan-summary.md:379)
Required outcome: retained base64 bounded by IMAGE_CACHE_TTL and emptied per-session dicts reclaimed, on the write
  path, with no request that succeeds today starting to fail.
Authority:
  - §Member F-3 Required behavior (:469-470) is global and unqualified: "retained base64 is bounded by
    config.IMAGE_CACHE_TTL and empty per-session dicts are reclaimed"; plan-summary.md:379 repeats "empty
    per-session dicts reclaimed on the write path"
  - §Member F-3 Evidence (:456-457) names the cross-session axis as part of the harm: session.py:93-98 → each
    distinct system prompt adds an _image_cache key that is never reclaimed
  - Verified the crash reachability: messages.py:38-40 has no try (the only handler try is :26-29 for JSON) and
    server.py:48-53 logging_middleware does not catch, so a KeyError from image.py:84 surfaces as HTTP 500
  - Verified the primitive already exists and is entirely caller-less: cleanup_expired_images() at image.py:176-179
    iterates every session (grep across buster_ripper/ and tests/ → zero call sites), and it does NOT touch the
    read path — get_cached_image calls the per-session _cleanup_expired_images at :155, not the global
  - Verified the CLI wording asymmetry: main.py:125-126 "Maximum number of images to cache per session" vs
    main.py:127-128 "Image cache TTL in seconds" — the size control is session-scoped by its own help, the TTL is
    not
  - DC-F-3-r1 rejected candidate (a) is a background task "alone" (:555); it does not forbid the existing janitor
    gaining a write-path caller, and §Selected closure (:471-472) already cites the module's own precedent of the
    writer maintaining the cache after the write (:87)
Admissible candidates:
  (a) the writer's tail calls the GLOBAL sweep (every session), which prunes expired entries and deletes a session
      dict that has emptied; the delete lives only in that sweep, never in _cleanup_expired_images
Rejected candidates:
  - prune+drop positioned between the ensure at :121-122 and the loop's first write at :84 -> produces a KeyError
    → HTTP 500 on the second request from any session idle past the TTL: knifes in a new request failure the wave
    is not owed (closure step 3)
  - prune before the ensure, session-scoped delete -> the ensure re-creates the dict, so the mandated "dropped when
    it empties" clause is unobservable for every image-carrying request: does not perform the required outcome
  - prune after the loop, session-scoped delete -> unreachable, because after F-1 the writer always writes ≥ 1
    entry when it runs; leaves the reported leak and makes its own clause dead code
  - keep it session-scoped and narrow the required-behavior sentence to active sessions, deferring the idle axis
    -> preserves the finding's own verified harm (:456-470) and re-writes approved plan text to fit a mechanism
  - additionally install a background task now -> the rejected "alone" candidate's own reason: a second lifecycle
    task for the same job, and the additive remainder is already deferred with a named owner and trigger
    (:479-480)
  - user-fork (global vs session-scoped) -> not two admissible outcomes: one preserves part of the verified defect
Selected closure: (a), as R-B, with the positional prohibition written as a prohibition because no handler-level
  try exists to absorb a misplacement.
Risk shape: unchanged for the shipped default (config.STRIP_IMAGES = False, config.py:40); per-strip cost moves from
  O(this session's entries) to O(total entries) — the same class DC-F-3-r1's Risk shape already accepted (:558) and
  strictly smaller than the background-task alternative it rejected.
Deferred additive work: the traffic-idle residual (no image traffic at all → nothing reclaims) gets its named
  trigger in R-E, on the existing deferral line; owner unchanged (T014).
Owning route: Authority Registrar (R-B, R-E) -> Forward T014 (F-3)
Question required: no
```

### Packet 3 — Row 3 / A-3 (what the TTL timestamp measures)

```text
CLOSURE: absorbed
Finding: 061 row 3 (handoff-context.md:432-434, :469-470 × image.py:72, :84)
Required outcome: IMAGE_CACHE_TTL bounds how long a retained base64 payload can sit in memory, while the max-size
  bound this finding does not own keeps behaving as it does today.
Authority:
  - §Member F-3 Required behavior: "retained base64 is bounded by config.IMAGE_CACHE_TTL" (:469-470) — a bound on
    age, not on idleness; no other artifact statement supports an idle reading (config.py:42 is "# seconds";
    main.py:127-128 is "Image cache TTL in seconds"; the module docstring at image.py:112 says only "TTL-based
    expiration and LRU eviction" — Red is right that none of these disambiguate, so the required-behavior sentence
    is the only text that speaks to it)
  - Verified today's mechanism: image.py:84 writes unconditionally, so a repeated source refreshes ts; with F-2's
    dedupe this becomes visible for the first time (an image resent every turn is pinned for hours while
    --image-cache-ttl 300 claims a bound)
  - Verified the no-cost argument for the strict reading: post-R-A the key is content-derived, so an entry removed
    at absolute expiry and re-cached on the next send yields the identical placeholder text — F-2's "byte-identical
    placeholder text" clause (:439-440) still holds, so the strict bound does not fight the prefix contract
  - Verified the unowned axis: _evict_by_lru (:62-75) sorts on the same ts; F-3's own counterexample calls the
    per-session bound adequate ("per-session count *is* bounded by _evict_by_lru", :467-468), and no test
    discriminates the two orders (TestImageCacheLRUEviction :178-198 uses three distinct sources and no repeat)
Admissible candidates:
  (a) separate the two axes: the entry carries first_seen (what TTL compares against) and last_touched (what
      _evict_by_lru sorts on); a repeat refreshes last_touched only, so the TTL bound is absolute and eviction
      order is unchanged from today's observable behavior
Rejected candidates:
  - keep one timestamp and freeze it at first insert (absolute TTL, FIFO eviction) -> closes the owned axis but
      silently changes which payloads survive max-size pressure, an order the finding never claims, and leaves a
      function named _evict_by_lru implementing first-in-first-out (broader than the smaller complete remediation)
  - keep one timestamp and refresh it (idle TTL) -> retention is unbounded for a busy session, so the approved
      required-behavior sentence is false on the most common shape and "advertised controls that bind nothing" —
      this finding's own defect statement (:459) — survives (closure step 3)
  - narrow the required-behavior sentence to "idle since last send" -> re-writing approved plan text to fit one
      mechanism, and the narrowing has no authority behind it
  - user-fork (idle vs absolute, LRU meaning) -> the required outcome is ranked by the plan's own sentence, and the
      eviction-order residue is not user-observable today at all: get_cached_image has zero production callers
      (:449-451, settled by RB-003/FK-007 option 1), so no user can see which payload a retrieval returns
Selected closure: (a), as R-C, sized by its own case table so the pair is designed together the way the wave block
  requires (:377-378).
Risk shape: unchanged in kind — one more field in the cache tuple (:14) and one comparator change at :56/:72; no
  observable move on the size bound, and the TTL axis moves from "binds nothing" to "binds", which is the finding.
Deferred additive work: none
Owning route: Authority Registrar (R-C) -> Forward T014 (F-2 + F-3, one increment)
Question required: no
```

### Packet 4 — Row 4 / A-4 (the detector's witnesses)

```text
CLOSURE: absorbed
Finding: 061 row 4 (handoff-context.md:409-414; DC-F-1-r1 :523)
Required outcome: `has_images` returns True exactly when the writer would replace ≥ 1 block — i.e. the detector
  cannot be an earliest-only or latest-only abbreviation and still pass the gate.
Authority:
  - §Member F-1 Required behavior (:405) "one detector scope, one writer scope, all user messages"; DC-F-1-r1
    Selected closure (:523) names the scan-all fix and the two witnesses
  - Verified the non-discrimination: the only multi-message shape in the suite is tests/test_image.py:49-58, whose
    image sits in message 0 → `for msg in messages: if user: check; break` returns True there; the route-level
    witness as worded (:411-412) and the packet counterexample (:400-404) have the same earliest-image shape; and
    no test anywhere holds a text-only first user message with an image on a later turn
  - Verified the writer's scope is the target: image.py:128-136 iterates every user message, so "agree with the
    writer" is checkable per shape
Admissible candidates:
  (a) bind the acceptance as a position case table (R-D) whose rows include image-only-in-a-LATER-user-message and
      image-only-in-a-MIDDLE-user-message, plus the negative shapes
Rejected candidates:
  - leave the witnesses as written -> every mandated witness is green with a detector that forwards base64 for the
      common shape (text first, image attached on a later turn): a required gate that cannot fail for the defect
      it gates (closure step 3)
  - add a coverage gate for routes/* -> re-opens the settled pyproject.toml:30-34 omit list; broader cumulative
      risk than the smaller complete remediation
  - consolidate the two guarded call sites into one entry point -> excluded by the wave surface (:376-377) and by
      the clause's own wording, as Red row 8 records
  - user-fork (how many witnesses) -> test count is not a user-visible outcome; one admissible remediation remains
Selected closure: (a).
Risk shape: unchanged for production code; the member's test authoring grows by two discriminating shapes, which is
  what makes its own done-when satisfiable.
Deferred additive work: none (the tool_result-nested axis is already deferred with owner and trigger at :415-416)
Owning route: Authority Registrar (R-D) -> Forward T014 (F-1)
Question required: no
```

## Repairs authorized for Authority Registrar (apply exactly these, nothing wider)

All five land in `handoff-context.md` §Planned Task Context → the `T014` `<details>` block (`:371-482`). Line numbers
are pre-repair positions.

**R-A — §Member F-2, replace the two disagreeing phrase-clauses and the example-only closure.**
Replace `:435` ("derive the key from the image source (e.g. `img_<sha256(source)[:12]>`)") and `:439`'s opening
clause ("the key is a pure function of the source block") with one binding, and align `:427-428`:

> **Selected closure (referent bound).** The key is derived from **`block["source"]`** — the dict read at
> `image.py:80`, stored at `:84`, and returned by `get_cached_image` (`:160-161`) — through an **order-independent**
> canonical serialization: `img_<sha256(json.dumps(source, sort_keys=True, separators=(",", ":")))[:12]>`.
> `session_id` does **not** participate; the cache is already keyed by it at `:84`.
> **The whole image `block` is not the input.** Verified counterexample: `normalize.py:212` runs the image step
> *before* `_normalize_cache_control` at `:223`, so a block-level digest there covers a `cache_control` field that
> `messages.py:34`→`:223` has already deleted — a block-level key would therefore differ per call site and per turn
> position for one unchanged image, which is F-2's defect class, not a fix of it. The wording "image source"
> (`plan-summary.md:378`) and "source block" (`:439`) name **this** referent; neither is a synonym for the block.
> Insertion point unchanged: inline in `_replace_image_with_placeholder` (no new module/class/registry — the
> `Mode: NEW` rows remain the only authorization, `plan-summary.md:133-134`).
>
> Done when (replaces `:439`'s first clause; the rest of `:439-444` stands): the key is a pure function of
> `block["source"]` as above, and the case table below passes:
>
> | Case | Expected |
> |---|---|
> | same `source`, two separate requests | byte-identical placeholder text |
> | same `source`, JSON key order differs between the two requests | byte-identical placeholder text (kills an insertion-order-dependent form such as `repr()`) |
> | same `source`, stripped at `messages.py:38` vs stripped at `normalize.py:213` (block still carries `cache_control` there) | byte-identical placeholder text — the discriminator for the referent |
> | two different `source` dicts | different keys |
> | same `source` twice inside one request | one cache entry |
> | same `source` in session A and in session B | identical placeholder text (kills `session_id` in the digest) |

**R-B — §Member F-3, replace §Selected closure (`:471-474`) and §Done when (`:475-477`).**

> **Selected closure (scope and position bound).** At the **tail of `strip_and_cache_images`, after the message
> loop**, call the **global** sweep: prune expired entries for **every** session in `_image_cache` and delete any
> session dict that has emptied. The primitive already exists and is entirely caller-less — `cleanup_expired_images()`
> (`image.py:176-179`) has zero call sites in `buster_ripper/` **and zero in `tests/`** — so this is a wiring fix,
> not a new mechanism, and it lands at the module's own precedent position (`_evict_by_lru` is called after the
> write, at `:87`).
>
> **Positional prohibition (binding, not advisory).** No code may delete `_image_cache[session_id]` between the
> ensure at `image.py:121-122` and the writer's first index at `:84`: `routes/messages.py:38-40` has no `try` (the
> only handler-level `try` is `:26-29`, for JSON) and `logging_middleware` (`server.py:48-53`) does not catch, so
> such a placement returns **HTTP 500 on the second request from any session idle past the TTL**. The delete lives
> **only** in the global sweep; `_cleanup_expired_images` (`:47-59`) stays expiry-only, because `get_cached_image`
> calls it at `:155` and indexes at `:157` — a delete there gives the read path a `KeyError` and turns
> `tests/test_image.py:137-156` red (FN-1, honored).
>
> Done when (replaces `:475-477`): the writer's tail sweep prunes expired entries across all sessions and drops an
> emptied session dict, and the write-path focused test advances the clock past `IMAGE_CACHE_TTL` with **no
> `get_cached_image` call** across these cases:
>
> | Case | Setup → action | Expected |
> |---|---|---|
> | W1 same-session expiry | sid-A caches an image → advance past TTL → strip a *different* image for sid-A | A's expired payload gone; the new one present; `"A" in _image_cache` |
> | W2 abandoned-session reclaim | sid-A and sid-B each cache an image → advance past TTL → strip one image for sid-B | `"A" not in _image_cache` (the dict itself dropped) — this is the global axis and the witness the session-scoped wording could not produce |
> | W3 no-crash regression | sid-A's entry expired | a second request from sid-A completes: the writer does not raise, and no `KeyError` reaches the handler |
> | W4 read path untouched | sid-A's entry expired → `get_cached_image("A", key)` | returns `None` and leaves `"A" in _image_cache` — the failure witness for the prohibition above |

**R-C — §Member F-3, append to §Selected closure; §Member F-2's dedupe clause is unchanged.**

> **What the timestamp measures (bound).** `IMAGE_CACHE_TTL` is **absolute age since first cached**, because the
> required behavior is that retained base64 is "bounded by" it (`:469-470`); a re-send must not extend retention.
> `_evict_by_lru`'s order is **not** this finding's axis and must stay as it behaves today (least-recently-touched),
> so the cache entry carries both stamps: `first_seen`, which `_cleanup_expired_images` compares, and
> `last_touched`, which `_evict_by_lru` sorts on (`:72`); a repeat of an already-cached source refreshes
> `last_touched` only. Tuple shape at `:14` becomes `(source, first_seen, last_touched)`. The strict bound is free
> on the prefix axis: once the key is content-derived (R-A), an entry removed at expiry and re-cached on the next
> send emits **identical** placeholder text, so F-2's byte-identical clause still holds.
> Rejected readings, so they are not re-litigated at Forward: refreshing the single stamp (idle TTL — leaves the
> bound unbounded for a busy session and re-enacts "advertised controls that bind nothing", `:459`), and freezing
> the single stamp (absolute TTL but FIFO eviction, changing an order no finding here owns and leaving
> `_evict_by_lru`'s name false).

**R-D — §Member F-1, replace §Done when (`:409-412`).**

> Done when: `has_images` scans all user messages, and the acceptance is this position table — every row is a
> witness, not an idea:
>
> | Shape (`messages`, in order) | `has_images` | Writer effect |
> |---|---|---|
> | image in the **earliest** user message, latest user message text-only | `True` | every image block replaced (this is `tests/test_image.py:49-58`, **inverted to expect `True` as part of the fix** — the characterization test moves with the behavior, it is not exempted) |
> | **text-only first** user message, image on a **later** user turn | `True` | the historical image is replaced — kills an earliest-only detector |
> | image only in a **middle** user message (first and latest both text-only) | `True` | kills both earliest-only and latest-only abbreviations |
> | no image in any user message | `False` | body untouched; `changes` does not gain "stripped and cached images" |
> | image only in a **non-user** (assistant) message | `False` | untouched — the writer's scope is user-only (`image.py:128-130`) |
>
> Plus: a route-level witness through `/v1/messages` shows a text-only latest turn still strips a historical image;
> and both guarded call sites (`messages.py:38`, `normalize.py:212`) are covered by the same detector — by
> construction (one `has_images`), which is what the clause means; it is not a directive to consolidate the two
> sites (out of the wave surface at `:376-377`) and does not require one witness per site.

**R-E — §Member F-3's §Deferred additive work (`:479-480`), and the restroking rule.**

> The deferral keeps its owner and its cost trigger, and gains the leak trigger this pass found missing: *if
> reclamation must survive a session that receives no further image traffic at all* (a write path only runs when
> some request carries an image), the background janitor remains available as additive work with the same owner.
> Recorded, not owed now.

**Structural repair, carried with R-A (Red rows 14 and 19).** Keep the `handoff-context.md` `T014` block as the
single blueprint and reduce `plan-summary.md:376-385` to the pointer it already opens with (`:372-374`); the
restatement is where `T014`'s "write path" binds to the **dump cluster** for a reader working from the task list, and
where "deterministic function of the image source" disagrees with the blueprint's "source block". **The fix is not to
reword both renderings into agreement** — that doubles the next drift. The same rule is already recorded for
`T013`'s restatement (`062-decide.md` §note on Red row 17); one structural pass should discharge both, editing no
`T013` content.

Red's rows 6-13 (excluded-with-citation) need no action. Its rows 15-18 and 20-22 are discretion footnotes and need
no round: FN-1 is honored by R-B's prohibition, FN-2/FN-3/FN-5/FN-6/FN-7 stand as written, and FN-4's converse
(nothing forbids `strip_and_cache_images` from raising out of the request path) is the reason R-B's positional
prohibition is binding rather than advisory — recorded here, and no task is created for a handler-level guard, which
is outside this wave's evidence.

## Deterministic Closure Ledger entries for this pass

| # | Finding | Closure | Selected remediation | Plan-surface change | Question |
|---|---|---|---|---|---|
| 1 | 061 row 1 (A-1, key referent) | absorbed | R-A | yes — `F-2`'s done-when gains a case table; no milestone, no `MS-*`/`QAP-*` gate moves | no |
| 2 | 061 row 2 (A-2, prune position) | absorbed | R-B | yes — `F-3`'s done-when becomes the W1-W4 table | no |
| 3 | 061 row 3 (A-3, TTL semantics) | absorbed | R-C | yes — `F-3`'s closure gains the dual-stamp binding; cache tuple shape named | no |
| 4 | 061 row 4 (A-4, detector witnesses) | absorbed | R-D | yes — `F-1`'s done-when becomes the position table | no |
| 5 | 061 row 5 (A-5, global vs session-scoped) | absorbed | R-B + R-E | yes — the sweep is global; the deferral line gains its leak trigger | no |

Suggested keys for Registrar: `DC-F-1-r2` (row 4), `DC-F-2-r2` (row 1), `DC-F-3-r2` (rows 2, 3, 5 — one member's
repaired closure, superseded-by-nothing); each appends to §Deterministic Closure Ledger — 2026-10-05 routing without
editing `DC-F-1-r1`/`DC-F-2-r1`/`DC-F-3-r1`, whose `absorbed` dispositions stand and are refined, not reversed.

**No invariant-registry change is owed.** The two invariant candidates (`:398-399`, `:427-428`) remain
non-authoritative Interview material exactly as the block records them, and these repairs make both harder to violate
rather than restating them. R-A does supply the missing referent for `:427-428` ("a deterministic function of the
image source"), so if Interview later elevates it, it now has one binding to elevate — Registrar applies no registry
write on that account.

## Compaction Decision

- Prior same-phase files reviewed: `022-decide.md`, `025-decide.md`, `027-decide.md`, `029-decide.md`,
  `040-decide.md`, `062-decide.md` (decide reports of the current planning cycle; earlier ones belong to closed
  cycles).
- Superseded: **none.**
- Not superseded: **all six.** Each dispositions a different slice — 022 the completeness repair, 025/027 the 024/026
  goal and R16 rows, 029 028's `T006` guard shape, 040 the scoped-plan-blocker pass, 062 `060`'s six `T013`
  header-parity rows. This pass covers `T014`'s image-slice wave, which no prior report touches.
- Compaction performed: **no.**
- If no: `red-findings` mode forbids compacting prior Decide reports, no prior report holds an unresolved option or
  uncertainty this pass now represents, and `062-decide.md`'s rows are a different slice whose repairs (`R-A`…`R-E`
  there) are still awaiting Registrar — merging the two reports would make one of them unfindable for its own
  routing. The two passes' repair labels collide by accident (`R-A`/`R-B` in each); this report's keys are the `063`
  ones and Registrar should qualify by report number when applying.

## After R-A…R-E land

`T014` Forward is still **not** authorized until `forward-red` is re-dispatched against the repaired block and
returns `0 AMBIGUOUS` (the 029 chain, and the same route `062-decide.md` §"After" records for `T013`). Rows 14 + 19
travel with A-1 as one binding, per Red's §Routing. Nothing here moves `MS-004`'s APPROVED verdict, any accepted
task's scope, `G-003`'s text, or `FK-011`'s ownership (no `T014` member carries a preserved-thinking clause).

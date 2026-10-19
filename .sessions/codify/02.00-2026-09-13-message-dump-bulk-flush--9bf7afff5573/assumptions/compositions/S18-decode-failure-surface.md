VERDICT: SAFE
PROMOTION: INELIGIBLE — seven of the eighteen members still carry no permanent `CL:` handle in `claims.md` (68, 69, 71, 72, 137, 402, 403) and four record `Source Fingerprint: unavailable` (68, 69, 70, 71), so neither "permanent handles for all member claims" nor the fingerprint half of eligibility is met; `seams.md` S18 additionally stays at `Routing: absorbed`, `Permanent: —`.

## Seam

**S18** — the charset a peer names in its `Content-Type` crossing aiohttp's decode step into the
exception class each parse arm must catch, and the raw-bytes path that has to stay available on the
other side of that raise.

The question, as the seam row's `Boundary value` and `Dimensions` cells pose it (*seams.md* line 24;
no separate `Question:` key exists): *which exception classes can one peer-named charset produce at
this boundary, in each direction, and is every one of them caught where the site already answers a
bad body?*

**Answer on CR4 bytes: yes — the prior record's verdict stands and nothing in the CR4 delta weakens
it.** All three production parse sites still catch all three classes the boundary produces
(`messages.py:30`, `chat.py:31`, `count_tokens.py:23`), the response-side escape is still taken
(`_error_body`, `messages.py:187-198`), and the delta the prior record's §Invalidates On 5 named —
`count_tokens.py:79-95` reordering keying relative to normalization — **did not occur in the
reordering sense**: keying still runs after normalization on both decode sides, and the step that was
narrowed is exactly the step that writes the text the key digest consumes (§Composition).

**Members** (seam-row order): 51, 68, 69, 70, 71, 72, 137, 398, 413, 400, 401, 402, 403, 414, 405,
406, 407, 416. All eighteen `State: HOLD` in `claims.md` and all eighteen verdict files
(`verdicts/claim-*.md`) begin `VERDICT: HOLDS`, read **before** any verdict content was opened. Every
`pkg:`/`lang:` digest those verdicts pin was re-verified byte-for-byte against the installed tree by
this pass; none of the eighteen members is re-derived, per the delta-scoping instruction.

**Prior record:** this file at `sha256:fac1d48c…` (2026-10-07T01:35:23Z, `VERDICT: SAFE`), itself the
CR3/`T019` delta recomposition of the 2026-10-06 `COUNTEREXAMPLE` record. This pass is the **CR4
delta recomposition**: two accepted remediations landed on this seam's consumer end
(`T019.001`, `T019.002`), and §Invalidates On 5 of the prior record named the `count_tokens`
reordering directly, with the clause that a fix landing under another seam still requires this record
to be re-read before citation. That is why the prior `SAFE` is re-derived here rather than cited.

## Delta scope

**Changed members by anchor intersection: none.** All eighteen members pin library, interpreter or
documentation bytes; all ten installed-file pins recomputed this match their verdicts exactly
(`web_request.py 2aaacea7…`, `helpers.py 435df4ec…`, `client_reqrep.py 7844440e…`,
`client.py 7ce41fc1…`, `web_protocol.py eacf5d33…`, `web_exceptions.py ee722e8b…`,
`payload.py 3ba9ec60…`, `web_response.py 3ca73388…`, METADATA `09044e64…`,
`json/decoder.py dc98d3e6…`; aiohttp still 3.13.3). The fetched-documentation/CPython values
(68–71, 402, 403, 405, 406, 407, 414, 416) are carried as verdict-recorded, not re-fetched — re-
fetching re-checks an atomic claim, which is not this role, and those verdicts' own `Invalidates On`
make byte-identity on re-fetch the condition. No member is stale, so the recomposition stays scoped.

**Changed consumers — the anchors the seam's trace passes through:**

| file | prior pin | now | what moved |
|---|---|---|---|
| `routes/messages.py` | `2d50a0d2…` | `64be411b…` | **T019.001**: `handle_streaming`'s overflow probe reads `result.text` (`:117`) instead of `str(result.body)` — the N6 dead path is fixed at its consumer; `_error_body` gained a docstring (now `:187-198`, logic byte-identical); request arm (`:28-33`), normalize-then-key (`:38-39`), retry construction (`:162-166`), `errors="replace"` probe decode (`:311`) all intact, renumbered only |
| `routes/count_tokens.py` | `195f3b24…` | `c6d6f924…` | **T019.002**: `:87`'s `normalize(body, session_id="")` replaced by `strip_billing_nonce(body)` (import at `:10`); `:88` `session_id(body)` unchanged in position; `_read_body` (`:17-26`) byte-identical in behavior — both arms, all three class names |
| `buster_ripper/normalize.py` | `a7ddbf06…` | `87ebcd05…` | public `strip_billing_nonce` (`:64-74`) delegating to `_normalize_system` (`:50-61`); file renumbered (`normalize` now `:215-241`) |
| `routes/chat.py`, `server.py`, `routes/passthrough.py`, `utils.py`, `compaction.py`, `config.py`, `session.py` | unchanged | unchanged | `38e577e2…`, `b3b48661…`, `eebf2119…`, `9abaa6b8…`, `ac9d4214…`, `22348c68…`, `67a66b8a…` re-verified identical this pass |

**Dimensions re-traced versus carried:** re-traced — 2, 3, 4, 5, 6 (their consumer anchors
`count_tokens.py`, `messages.py` moved; under §3c a small edit inside a core dimension re-traces the
whole dimension). Carried with re-verification — 1 (both directions' class-production and the
asymmetry rest on library bytes alone, which are identical; the arms that observe it were re-read).
What is carried forward from the prior record is the premise set and the library-side line reads,
re-confirmed at the digests above.

## Composition

### Request direction — the client's own `Content-Type`

Producer chain unchanged: `HeadersMixin.charset` (`helpers.py:770-776`, never validated, 401) →
`BaseRequest.json()` = `text()` then `loads()` (`web_request.py:692-695`), `text()` =
`bytes_body.decode(self.charset or "utf-8")` with no `errors` argument (`:686-690`, 402). The union
of raise directions stands exactly as the prior record enumerated — five arms onto three classes:
absent/falsy-empty charset → utf-8; registered text codec + invalid bytes → `UnicodeDecodeError`;
unregistered name → `LookupError` (401 + 405); registered **non-text** codec → `LookupError` at the
registry's `_is_text_encoding` gate (416). `json.JSONDecodeError` on bad syntax (51, 70).

**Catch set re-verified at the delta — widened nothing, narrowed nothing:**

- `messages.py:28-33` — `except (json.JSONDecodeError, UnicodeDecodeError, LookupError)` →
  `web.json_response({"error": "Invalid JSON"}, status=400)`. Unchanged.
- `chat.py:29-32` — identical tuple → identical 400. Unchanged file bytes.
- `count_tokens.py:19-26` — `json.JSONDecodeError` → `{}` fallback; `(UnicodeDecodeError,
  LookupError)` → `raise web.HTTPBadRequest(...)` (137: HTTP 400). **The T019.002 edit is entirely
  downstream of `_read_body`**: the classes the boundary produces are produced inside
  `await request.json()` at `:20`, before any normalization exists in the call chain, so narrowing
  the post-parse chain cannot add, remove or re-route a raise direction *at this boundary*. Arm
  disjointness (71/72, 406/407) still forecloses shadowing.

**N2 re-run at the moved files** (the check is load-bearing since CE-1's closure rides on it):
`messages.py:6-7` and `count_tokens.py:6` import only `aiohttp` / `aiohttp.web`, not
`aiohttp.payload` (`payload.py:62`'s distinct `LookupError`); every bare `LookupError` in the three
arms binds the builtin — the class 401/405/416 actually produce.

### The normalization-before-keying ordering — §Invalidates On 5's named condition

The prior record's N1 cited `count_tokens.py:87-88` as `normalize(body, session_id="")` →
`session_id(body)` and invalidated on that pair reordering. On CR4 bytes:

- **Messages side:** `messages.py:38` `body, changes = normalize(body, session_id="")` → `:39`
  `sid = session_id(body)`. Ordering intact.
- **Count-tokens side:** `count_tokens.py:87` `strip_billing_nonce(body)` → `:88`
  `sid = session_id(body)`. Ordering intact — normalization (narrowed) still **before** keying, and
  the digest operand is written by the step that runs first.
- **Digest after the writing step:** `session_id` (`session.py:93-98`, unchanged bytes) computes
  `hashlib.sha256(system.encode()).hexdigest()[:12]` over `body["system"]` **at call time** — the
  hash therefore consumes the post-strip text. `strip_billing_nonce` → `_normalize_system` writes
  `body["system"]` in place (`normalize.py:55`/`:58`), so the pair reproduces the messages side's
  key-formation order on the narrowed path.
- **Key-identity of the narrowing** (the premise T019.002 rests on — "`session_id()` reads `system`
  alone" — which claim 11, `HOLD`, `CL:buster_ripper.session#session_id/key-derivation-input`,
  establishes for the *operand*, but not for the *writer set*): of the steps `normalize()` actually
  runs at `session_id=""` (step 0, images, is skipped by the `:225` truthy-`session_id` guard),
  `_normalize_system` is the **only writer of the hashed text**. `_normalize_tools` writes
  `body["tools"]` (`:85`, `:89`); `_normalize_cache_control` deletes only `cache_control` keys from
  system blocks and messages (`:107-109`, `:120-122`, `:131-132`) and never touches the `"text"`
  member `session_id` joins (`session.py:96-97`); `_normalize_date` writes `msg["content"]` (`:209`).
  So the narrowed call yields the same digest the full chain did. **N10** records that this
  writer-set argument rests on this pass's read of `normalize.py`/`session.py`, not on a member
  claim.

### Post-parse fallback — N1 re-checked against the narrowed chain

`count_tokens.py:21-22` still maps a syntax failure to `{}` and proceeds.
`strip_billing_nonce({})` → `_normalize_system({})` → `body.get("system")` → `None` → neither `list`
nor `str` → returns `[]`, mutating nothing (`normalize.py:52-61`); `session_id({})` → sha256 of `""`;
`estimate_tokens({})` → `json.dumps` → `max(1, 0)` = 1 (`session.py:88-90`). The `{}` request still
settles 200 with an estimate — route-local policy, unchanged in outcome, still not a conflict.

**On T019.002's review measuring five read sites in `normalize()`'s element loops:** the loops
(`normalize.py:88-90` over `tools`, `:130-131` over `messages`, `:204-208` over `messages`) index
elements as mappings. Under this seam they are outside the boundary in two independent ways: they run
only *after* `_read_body` has already settled every class the charset crossing produces, and the
`AttributeError` a non-mapping element raises is a payload-*shape* failure, not a decode failure.
Their removal from the count_tokens path deletes one out-of-boundary raise site (CR4-F-01) and adds
no class, arm or settlement at this one. The corrected read-site count changes **nothing this seam is
entitled to assert**; it matters to S12/S13/S14-family premises, not to S18's.

### Upstream-response direction

Unchanged in kind, re-confirmed at identical library bytes: `get_encoding()` suppresses `LookupError`
inside the response path (`client_reqrep.py:709-730`, 413) and returns only names that cleared the
text-codec gate (416), so the live class out of `text()` here is `UnicodeDecodeError` (398 + 402) —
and `_error_body` catches `(UnicodeDecodeError, LookupError)` by name around `resp.text()` and falls
back to `(await resp.read()).decode(errors="replace")` (`messages.py:195-198`): `read()` is
charset-free (400) and replacement raises nothing (403). The `LookupError` clause stays unreachable
as configured (N4, carried). `compaction.py:41`/`:55` unchanged. The read-after-failed-decode state
machine (N8) was re-verified only by digest identity (`client_reqrep.py` unchanged); the prior
resolution stands and remains unrowed.

### The probe's new consumer — dimension 4's live end

`_error_body`'s str now reaches `parse_context_overflow_error` *strictly*, replacing the prior
record's N6 dead repr path. The round trip, traced at the digests carried members pin (these two
files are cited by no S18 member — N8-pattern read, recorded in N11 below):

- `messages.py:162-166` constructs `web.Response(body=error_content, status=…,
  content_type="application/json")`. `Response.__init__` (`web_response.py:649-687`): `"charset" not
  in "application/json"` passes the `:649` guard; the header is written **without** a charset
  parameter (`:675-681`, no `charset=` argument); `body` is not bytes, so the setter
  (`:698-721`) stores `PAYLOAD_REGISTRY.get(body)` → `StringPayload` (`payload.py:1110`,
  `:430-457`), which at construction encodes the str to **utf-8** (`:444-447` defaults,
  `:454-456`) and keeps those bytes as its `BytesPayload` value; the payload's own
  `text/plain; charset=utf-8` content type is not copied because `CONTENT_TYPE` is already set
  (`web_response.py:708-709`).
- `result.text` (`messages.py:117`) reads the property at `web_response.py:726-730`:
  `self._body.decode(self.charset or "utf-8")` — `Response.charset` (`:334-336` → `_content_dict`)
  is `None` for the charset-less `application/json` the site built, so the operand is utf-8, matching
  what `StringPayload` encoded with; `BytesPayload.decode` (`payload.py:380-381`) is a strict
  `self._value.decode(encoding, "strict")`. A Python `str` → utf-8 → strict-utf-8 round trip returns
  the **identical string**, including every U+FFFD the fallback inserted.
- The other `web.Response` this loop can see is the 502 `json_response` at `:184` (`:105`'s
  `isinstance(result, web.Response)` excludes the committed `StreamResponse`): `json_response`
  (`web_response.py:833-853`) goes through the `text=` fast path (`:653-669`), which appends
  `; charset=utf-8` and stores utf-8 bytes — `.text` reads back the same string through
  `self.charset = "utf-8"`. The comment block at `messages.py:106-116` states exactly this; the read
  confirms it for both arms, and neither arm's `.text` decode can raise: the bytes were produced in
  this process from the very string being recovered, under a codec the app itself fixed at utf-8 — no
  peer-named charset reaches this property (the header value is constructed, not relayed).

So dimension 4's "what replacement preserves of the text being matched" is now fully live: the
replace-decoded text is the matched text, and the prior parity argument carries — 414 (replace leaves
validly-decodable sequences untouched), 403 (replacement-and-continue for the rest),
`config.py:122-123`'s two patterns matching only ASCII letters/digits/whitespace,
`utils.py:123-140` unchanged: replacement cannot suppress an overflow match, and `.text` hands the
matcher the exact string `_error_body` produced. The non-streaming probe (`messages.py:311`,
`content.decode(errors="replace")` on `read()`-handed bytes, 400) is unchanged.

## Dimension Coverage

- **1. Client-request versus upstream-response direction (401 against 398/413) — CARRIED,
  re-verified.** The asymmetry (loud `LookupError` on the request side, suppressed on the response
  side) lives entirely in library bytes, all identical to the prior pins; the consumer arms that
  observe it were re-read at the moved files and name the same classes. 416's sharpening carries.
- **2. Four-arm/three-class union arms (51/402 vs 413/401/405 vs 416) — RE-TRACED, unchanged
  result.** Five arms, three classes, all three names in all three arms; the delta's premise that
  could have moved this (`count_tokens` skipping the normalization chain) is downstream of the
  crossing and touches no arm (§Composition, request direction). Mapped/unmapped members still settle
  identically per site.
- **3. `UnicodeDecodeError`/`json.JSONDecodeError`/`LookupError` as three catch targets; `ValueError`
  reach; completeness (68–72, 406/407, 416) — RE-TRACED at the arms, carried at the premises.** The
  hierarchy premises are untouched members; the point of three names (407: `LookupError` off
  `ValueError`) is still encoded in all three arms after the delta. Completeness for any
  registry-rejected name stands with 416; N7's custom-codec `TypeError` residue still has no
  production caller.
- **4. Strict default versus `errors="replace"`, text being matched (402 against 403/414) —
  RE-TRACED, strengthened.** The matched text is now provably the exact replace-decoded string
  (`.text` round trip above); what the prior record had to argue around a dead path, this record
  argues through a live one. The one residue the delta exposed is N9 below — an encode-side failure
  downstream of the crossing, not a class of it.
- **5. `text()` versus `read()` escape hatch (398/400) — RE-TRACED, unchanged result.** The escape is
  still taken on the far side of the raise (`messages.py:198`); every `read()` site carried
  (`passthrough.py:14`/`:38`, `chat.py:149`/`:188`, `messages.py:302`); the second-`read()` state
  machine (N8) rides identical library bytes.
- **6. Raising out versus the site's existing 400 shape (137) — RE-TRACED, unchanged result.** Both
  channels intact after the delta: `return web.json_response(..., status=400)` at `messages.py:33`
  and `chat.py:32`, `raise web.HTTPBadRequest(...)` at `count_tokens.py:24-26`, settled as 400 by 137
  (`web_protocol.py:500-515`, `web_exceptions.py:234-239`, identical digests). No class the boundary
  produces leaves a handler.
- **Systematic checks (re-run).** Discriminator discarded: no — four arms, three names, all carried
  to the settlement (count_tokens' syntax-fallback-as-proceeding is route-local policy, N1).
  Fallback/precedence selecting a different namespace or semantic kind: the two prior instances are
  unchanged (N1's `{}` proceeding, N3's utf-8 re-render at `:198`); the delta's new fallback —
  narrowing `normalize` to `strip_billing_nonce` — was checked for exactly this and selects **no**
  different namespace: the key is the same digest (§Composition, key-identity). Duplicate identities:
  re-checked at the moved imports, still none (`messages.py:6-7`, `count_tokens.py:6`, N2). Stale or
  reordered values accepted: no — `charset` re-derived per read; and the ordering premise the seam
  cares about (normalize-before-key, digest after the writing step) holds on both sides. Selection
  changed between dispatch and settlement: no — one `body` dict object mutated in place at `:87`
  before `:88` reads it. Disposal/recreation: N8 (unchanged bytes). Dependency rejection leaving a
  request unsettled: does not fire at any parse site; the response-direction construction path admits
  one out-of-boundary class, N9, recorded below with its reach. More than one terminal settlement:
  no — arm structure unchanged, and the retry loop's per-attempt single `return`/`continue` at
  `:120-136` is S9/S11's subject, re-checked here only for the boundary's classes (none produced
  there).

## Counterexample

**None found.** Dispositions against CR4 bytes:

| finding | status | what holds it |
|---|---|---|
| **CE-1** — unknown-charset escape from the parse arms | **STILL CLOSED** | run 42's landed arms survive run 44's review and the CR4 edits: `messages.py:30`, `chat.py:31`, `count_tokens.py:23` each name all three classes at the digests hashed here; §Invalidates On 2(a) did **not** fire — no arm lost a name. N2's builtin-binding re-checked at both moved files. |
| **MC-1** — response-side `text()` error covered | **STILL SETTLED** | `_error_body`'s catch-and-`read()`-with-`errors="replace"` logic is byte-identical (renumbered to `:195-198`); `compaction.py` unchanged; N4's unreachability of the `LookupError` clause stands. |
| **MC-2** — non-text-codec arm unnamed | **STILL DISCHARGED** | 416 current at its pinned `codecs.c` digest; the class set `{JSONDecodeError, UnicodeDecodeError, LookupError}` remains the complete set at this boundary for any registry-rejectable name. |
| **N6** — matched text never reached its matcher (prior record's note, "owned elsewhere") | **CLOSED BY T019.001** | the probe now reads `result.text` (`messages.py:117`), whose round trip through `StringPayload`/utf-8 recovers the exact `_error_body` string (§Composition). Per §Invalidates On 5's own clause this record has now been re-read and can be cited as current on this point. |

## Notes

Inert for the verdict; each checked against a cited production source.

- **N1 (updated) — `count_tokens` still answers `json.JSONDecodeError` with a *proceeding* request.**
  `{}` flows through `strip_billing_nonce` (`normalize.py:52-61`, no-op), `session_id` (sha256 of "")
  and `estimate_tokens` (`session.py:88-90`, returns ≥ 1) without raising; outcome unchanged from the
  prior record, premise citations moved off the superseded `:87-88`/`normalize.py:202-228` shape.
  Route-local policy, not a conflict.
- **N2 (unchanged, re-checked at moved files) — two live classes named `LookupError`**
  (`payload.py:62` vs builtin); no site imports the payload class, so all arms bind the builtin.
  Load-bearing for CE-1; §Invalidates On 2(b).
- **N3 (unchanged, renumbered) — the `:198` fallback changes the decode operand**, not just the
  error policy: non-ASCII upstream text arrives as U+FFFD re-rendered under utf-8 when the declared
  codec could not decode the bytes. Reachable only on already-undecodable payloads; nothing is
  established as owed beyond it. Inert. Now matters slightly more than before — this string is the
  one the fixed probe matches against, and dimension 4's ASCII-parity argument is what keeps that
  harmless.
- **N4 (unchanged) — `_error_body`'s `LookupError` clause is unreachable as configured** (413 + 416 +
  no `encoding=` argument at `:196`, no `fallback_charset_resolver` at `server.py:64-68`). The new
  docstring at `messages.py:190-194` records the same reasoning the prior composition derived —
  the note now agrees with the code's own comment; breadth in an arm remains not a finding.
- **N5 (updated) — neighbours, delta status.** (i) `HTTPRequestEntityTooLarge` foreclosed by
  `client_max_size=0` (`server.py:125`, unchanged bytes; S16/T018). (ii) `chat.py:162-165` catches
  `(json.JSONDecodeError, UnicodeDecodeError)` on raw `read()` bytes — correct, no codec name
  reaches it (400); unchanged file. (iii) Non-mapping *body* (`[1,2]`, `"x"`): still `AttributeError`
  from `normalize.py:52` — on the messages side via `normalize` at `messages.py:38`, on the
  count-tokens side now via `strip_billing_nonce` at `count_tokens.py:87` (the **same frame and
  line**, since step 1 always ran `_normalize_system` first) — and from `session.py:95` thereafter.
  Pre-existing, outside the charset crossing, not in the seam row's dimension list; flagged for the
  ledger only.
- **N7 (unchanged) — custom raw-tuple codec's `TypeError`** remains the completeness claim's edge;
  no `codecs.register` under `buster_ripper/` or in `aiohttp`/`click` at the installed bytes.
  `Reach: no production caller found`.
- **N8 (carried) — the read-after-failed-decode `_released` state machine** (`client_reqrep.py:701-
  702`, `:652-662`, `:670-674`) is still resolved by this composition's own read rather than a
  member claim; the governing library bytes are unchanged, so the prior conclusion carries, but the
  gap stays recorded so it is re-checked, not inherited, at the next library movement.
- **N9 (new) — the `:162-166` construction can raise a class no arm on that path binds, for text
  produced *by* this boundary but not *of* it.** `text()` under a surrogate-capable declared codec
  can legitimately return a str containing a lone surrogate (utf-16/32 codecs decode unpaired
  surrogate code units without error); `StringPayload.__init__`'s `value.encode("utf-8")`
  (`payload.py:454-456`) then raises **`UnicodeEncodeError`** (a `UnicodeError`, hence `ValueError`
  — but not `ClientError`), which `messages.py:182`'s arm does not catch, so it raises out of
  `_stream_single_request` → `handle_streaming` → `handle_messages` and the protocol answers 500 for
  a request whose upstream answered ≥ 400. Sequence: upstream error response with
  `Content-Type: application/json; charset=utf-16-le` and body bytes `3D 00 D8 00` → `_error_body`
  returns `"=\ud800"` via the success path at `:196` → construction raises at `payload.py:456`.
  **This is not a seam counterexample:** the class is produced by the *application's own re-encode*
  after the crossing, the boundary's contract (every decode-side class caught where the site answers
  a bad body) is satisfied, and the deciding sources are cited by no S18 member — the same
  disqualification the prior record applied to N6, which lived on the very same statement. It
  **predates CR4** identically (the prior bytes constructed the same `web.Response` from the same
  `_error_body` str); the delta did not introduce it, only re-numbered it.
  `Reach: buster_ripper/routes/messages.py:162` ← `:161 _error_body` ← `:102 handle_streaming`,
  entered only when `backend.get_backend_state() == "sglang"` (`:94`) and a streaming request's
  upstream answers ≥ 400 under a surrogate-capable charset. **Suggested routing:** the S9/S11
  error-transport family, alongside (and now partly superseding) the closed N6 — a new atom about
  str-body round-trip failure modes, not about charsets.
- **N10 (new) — the key-identity premise behind T019.002 is composition-read, not member-covered.**
  "`_normalize_system` is the only step of `normalize(body, session_id="")` that writes the text
  `session_id` hashes" was established by reading `normalize.py:50-61, 77-151, 181-212` and
  `session.py:93-98` at the hashed digests. Claim 11 establishes the operand ("system alone") but is
  an S1/S14 member, not an S18 one; the writer-set half has no row. Recorded so the count_tokens
  keying premise is re-checked rather than treated as verified if the next change touches either
  file. `tests/test_count_tokens_parity.py` is cited by the code comment as the repo's own pin — not
  evidence for this pass (tests are not evidence here).
- **N11 (new) — the probe's `.text` round trip likewise rests on unrowed library reads.**
  `web_response.py:649-730` and `payload.py:380-457, 1110` are cited by no S18 member claim; N9's and
  dimension 4's live-path conclusions are composition reads at the pinned digests, recorded here to
  keep the same discipline N8 established.

## Sources

Library (installed; digests as in §Source Fingerprint):
- `.venv/lib/python3.13/site-packages/aiohttp/web_request.py` — `:665-695` — 51, 401, dimensions 2–3
  (bytes verified; line reads carried)
- `.venv/lib/python3.13/site-packages/aiohttp/helpers.py` — `:387-399`, `:770-776` — 51, 401
  (verified; carried)
- `.venv/lib/python3.13/site-packages/aiohttp/client_reqrep.py` — `:644-740`, `:775-790` — 398, 400,
  413, dimensions 1/4/5, N8 (verified; carried)
- `.venv/lib/python3.13/site-packages/aiohttp/client.py` — `:230`, `:306`, `:425` — 413, N4
  (verified; carried)
- `.venv/lib/python3.13/site-packages/aiohttp/web_protocol.py` — `:500-515` — 137 (verified; carried)
- `.venv/lib/python3.13/site-packages/aiohttp/web_exceptions.py` — `:80-113`, `:234-239` — 137
  (verified; carried)
- `.venv/lib/python3.13/site-packages/aiohttp/payload.py` — `:62` (N2), `:380-381`
  (`BytesPayload.decode`), `:430-457` (`StringPayload`), `:1109-1120` (registry) — re-read this pass
- `.venv/lib/python3.13/site-packages/aiohttp/web_response.py` — `:334-336` (`charset`), `:649-687`
  (`__init__`), `:692-721` (`body` getter/setter), `:726-730` (`text` property), `:833-853`
  (`json_response`) — re-read this pass for the probe round trip and N9; **cited by no S18 member**
- `/opt/homebrew/…/python3.14/json/decoder.py` — `:20` — 70, 71, 72 (verified; carried)

Repo (working tree at this session's root):
- `buster_ripper/routes/messages.py` — `:6-7`, `:28-33`, `:38-39`, `:94-139` (esp. `:102`,
  `:105-118`, `:120-136`), `:142-184` (esp. `:155-166`, `:182-184`), `:187-198` (`_error_body`),
  `:302`, `:311` — read this pass
- `buster_ripper/routes/count_tokens.py` — `:6`, `:8-11`, `:17-26`, `:79-95` — read this pass
- `buster_ripper/normalize.py` — `:50-61` (`_normalize_system`), `:64-74` (`strip_billing_nonce`),
  `:77-151`, `:181-212`, `:215-241` (`normalize`) — read this pass
- `buster_ripper/session.py` — `:88-98` (`estimate_tokens`, `session_id`) — read this pass
- `buster_ripper/routes/chat.py` — `:6-7`, `:29-32`, `:149`, `:162-165`, `:188` — digest-verified,
  carried
- `buster_ripper/server.py` — `:48-52`, `:64-68`, `:125`, `:133-142` — digest-verified, carried
- `buster_ripper/routes/passthrough.py` — `:14`, `:38` — digest-verified, carried
- `buster_ripper/utils.py` — `:123-140` — digest-verified, carried
- `buster_ripper/config.py` — `:122-123` — digest-verified, carried
- `buster_ripper/compaction.py` — `:41`, `:55` — digest-verified, carried

Documentation/CPython consulted by member verdicts — carried as verdict-recorded values, not
re-fetched: Python 3.14 `library/exceptions.html`, `library/stdtypes.html`, `library/codecs.html`;
CPython v3.11.0 `Python/codecs.c` (416), v3.14.8 `Doc/library/codecs.rst` (405),
`Lib/json/decoder.py` (70–72).

## Source Versions

- aiohttp **3.13.3** — METADATA digest below; identical to the prior record's basis and to
  `seams.md`'s record across the CR4 movement.
- CPython **3.13** interpreter for the installed aiohttp package (`.venv/lib/python3.13/…`);
  installed-stdlib read for 70–72 is CPython **3.14.7**'s `lib/python3.14/json/decoder.py`, as
  pinned.
- Python **3.14** documentation for the `lang:python` members (68–72, 402, 403, 405–407, 414);
  **416** rests on CPython **v3.11.0** source (project floor `>=3.11`), verdict recording the gate
  unchanged at v3.13.0/v3.14.0.
- `buster_ripper` working tree: `routes/messages.py` `64be411b…`, `routes/count_tokens.py`
  `c6d6f924…`, `normalize.py` `87ebcd05…` — the three CR4 moves; `routes/chat.py` `38e577e2…`,
  `server.py` `b3b48661…`, `routes/passthrough.py` `eebf2119…`, `utils.py` `9abaa6b8…`,
  `compaction.py` `ac9d4214…`, `config.py` `22348c68…`, `session.py` `67a66b8a…` — identical to the
  prior record's pins.

## Source Fingerprint

Basis over the **eighteen** member rows in the seam row's declared order — permanent handle where
`claims.md` carries one, otherwise the verdict-recorded primary digest — plus the sha256 of every
file this pass consulted or verified. The hashed input is the `Components:` block verbatim: **38
newline-terminated lines, 5188 bytes**. (`verified` = recomputed this pass and matched the carried
pin; `consulted` = read this pass.)

```text
sha256:1be97834970a118fde85daef6935034c82bf85b29d091a6618fd3f05f042363c
```

Components:

```text
51   CL:aiohttp/request-json-decoding                       sha256:2aaacea7a01e581e5eead2ab1b9e4ba3b65bc2ae3d0f176b2a78ae5b6b76bb4e  (aiohttp/web_request.py)
68   (no handle)                                            fingerprint unavailable  (docs.python.org/3/library/exceptions.html#UnicodeDecodeError)
69   (no handle)                                            fingerprint unavailable  (docs.python.org/3/library/exceptions.html#UnicodeError)
70   CL:python/json-decode-error-inheritance                fingerprint unavailable  (CPython v3.14.0 Lib/json/decoder.py:20)
71   (no handle)                                            fingerprint unavailable  (docs exception hierarchy + CPython Lib/json/decoder.py:20)
72   (no handle)                                            sha256:dc98d3e677612ea7311303f06a144cec6b91ebadc7ec037d1777501647578a07  (installed Lib/json/decoder.py)
137  (no handle)                                            sha256:eacf5d33399aa96efb6f33354f5d75b863522c5a3a80d99f0e0ed7cd89c0f319  (aiohttp/web_protocol.py)
398  CL:aiohttp/client-response-text-charset-source         sha256:7844440eb67433c6454edd708c776e47e3fcfec9b8d0aeb980dcfe88c1986ac9  (aiohttp/client_reqrep.py)
413  CL:aiohttp/client-response-text-unknown-charset-error  sha256:7844440eb67433c6454edd708c776e47e3fcfec9b8d0aeb980dcfe88c1986ac9  (aiohttp/client_reqrep.py)
400  CL:aiohttp/client-response-read-charset-application    sha256:7844440eb67433c6454edd708c776e47e3fcfec9b8d0aeb980dcfe88c1986ac9  (aiohttp/client_reqrep.py)
401  CL:aiohttp/request-json-unknown-charset-error          sha256:2aaacea7a01e581e5eead2ab1b9e4ba3b65bc2ae3d0f176b2a78ae5b6b76bb4e  (aiohttp/web_request.py)
402  (no handle)                                            sha256:bf149b1649601f39dfcbb435a1c1650e2607fe9d5f315fc93ea8432b28ebae6b  (docs stdtypes.html)
403  (no handle)                                            sha256:5f29770c85fc38bec0a19414d698d89c34d1ceb6bfa2c0b5421dfcbaf281b98d  (docs codecs.html)
414  CL:python/errors-replace-valid-decode-parity           sha256:d1bd6e38bf008aa30dd2fc596e09ae03aefe6ed9b3395fde4b36ce7d33640c3a  (docs codecs.html, separate fetch)
405  CL:python/codecs-lookup-unknown-name                   sha256:a93ab5fd0291451dacdbc3f71a22fbbeba2807550e46b2e2f40734af6b5ae726  (CPython v3.14.8 Doc/library/codecs.rst)
406  CL:python/lookup-error-inheritance                     sha256:afdce60fa94da72d1c097b472257c770c6bed41342509b34863b6da7833b745e  (docs exceptions.html)
407  CL:python/unicode-decode-error-inheritance             sha256:afdce60fa94da72d1c097b472257c770c6bed41342509b34863b6da7833b745e  (docs exceptions.html)
416  CL:python/bytes-decode-non-text-codec-error            sha256:6cb5b75e0a2999d910e90d2ee92df8c47868dba11dc18673932689d07a3816f1  (CPython v3.11.0 Python/codecs.c)
verified  aiohttp/web_request.py               sha256:2aaacea7a01e581e5eead2ab1b9e4ba3b65bc2ae3d0f176b2a78ae5b6b76bb4e
verified  aiohttp/helpers.py                   sha256:435df4ecf0849d6cf844ff1cad4c3c764e7c734605d848b7272c2429f471cf91
verified  aiohttp/client_reqrep.py             sha256:7844440eb67433c6454edd708c776e47e3fcfec9b8d0aeb980dcfe88c1986ac9
verified  aiohttp/client.py                    sha256:7ce41fc1c2142f5346015455e030e3ebec22a41cde0fc299a66ce13be2cb2a9e
verified  aiohttp/web_protocol.py              sha256:eacf5d33399aa96efb6f33354f5d75b863522c5a3a80d99f0e0ed7cd89c0f319
verified  aiohttp/web_exceptions.py            sha256:ee722e8b0859dfdbc927d2ab5aa02b03941c59b51daa4cf608bc04a496a978df
consulted aiohttp/payload.py                   sha256:3ba9ec60d50b2fb01e33673227a4d85fbde77159cbe71270b7901e831c0c2aac
consulted aiohttp/web_response.py              sha256:3ca73388d5382e67ed5ea295be844cac06ce7950a5a5237e8b39c7b2259ece65
verified  aiohttp-3.13.3.dist-info/METADATA    sha256:09044e6424ad868f9e6fbc62148b80ce3df426eba9114fe31e96031621bf1e13
verified  python3.14/json/decoder.py           sha256:dc98d3e677612ea7311303f06a144cec6b91ebadc7ec037d1777501647578a07
verified  buster_ripper/server.py              sha256:b3b486615ca797febfa86795e0e18a04e4355aece31c43305a8d40bcad0e28d5
consulted buster_ripper/routes/messages.py     sha256:64be411b2c2172354528ff46f1b394a29d3dcef949dfa199bd06e3c4e08b6251
verified  buster_ripper/routes/chat.py         sha256:38e577e2ae7a6f9f547ba28a17ed91ec5d4867c62cc10ea340ba26058c31da7e
consulted buster_ripper/routes/count_tokens.py sha256:c6d6f92404deb5c65b73a00de51d1d9dd1f02946696ec8a1628ca09d23898c1f
verified  buster_ripper/routes/passthrough.py  sha256:eebf21192eb1a73f52d8cd34a69cff32f12090515d5a6ea61de1e19c87366e4c
verified  buster_ripper/utils.py               sha256:9abaa6b8ccbc2bdf8aadab44e64f2a07236ecad619cfc4b98fe78e897c8fdf24
verified  buster_ripper/compaction.py          sha256:ac9d42143c98694a0b6672dd753afcef537fc5e03ec747b4bb3c1208e23535e7
verified  buster_ripper/config.py              sha256:22348c68d21b91a30e75eeea802d4d390b2bcdabbf974b0ddc743c9354635522
consulted buster_ripper/normalize.py           sha256:87ebcd05c013917e121f406de54bbc7de1e01bb8454674b723e97c84fc7d2150
consulted buster_ripper/session.py             sha256:67a66b8aa733b31e982be4922350a7996ebdddb0fe5d391a6129a1cf342a9dff
```

Re-verification performed by this pass, and its limits: all ten installed library/interpreter digests
were recomputed and **every one matched the digest its member verdict pins** — no premise row is
stale, and the scoping of this recomposition rests on that. The fetched-documentation and
fetched-CPython values (68–71, 402, 403, 405, 406, 407, 414, 416) were not re-fetched (role
boundary, as in the prior record); 403's and 414's `codecs.html` renders still differ
(`5f29770c…` vs `d1bd6e38…`) and nothing in dimension 4 depends on which is canonical. The four
`fingerprint unavailable` rows are why promotion is ineligible regardless of verdict.

## Invalidates On

Composition is joint, so a member's own `Invalidates On` is not sufficient. Recompose this seam on
any of:

1. **Member fingerprints.** Any of the eighteen rows ceasing to match the bytes this record hashed
   (`1be97834…`). Library-side specifics carried from the prior record: (a) any aiohttp release
   wrapping `BaseRequest.text()`'s `.decode()` or validating `charset` (`web_request.py:686-690`,
   `helpers.py:387-399`, `:770-776` → 401); (b) narrowing/moving/re-raising the
   `contextlib.suppress(LookupError, ValueError)` at `client_reqrep.py:715` (→ 413, dimension 1);
   (c) reordering `get_encoding()`'s precedence or `text()`'s `errors="strict"` default (→ 398,
   dimension 4); (d) `read()` touching `Content-Type`, or the `_released`/`_wait_released`/
   `_notify_content` state machine changing so a read-after-failed-decode sees a released connection
   (→ 400, dimension 5, **N8**); (e) any CPython move of `LookupError`/`UnicodeDecodeError`
   placement (406/407), `codecs.lookup`'s not-found class (405), `bytes.decode`'s `errors` default
   (402), `JSONDecodeError`'s base (70/72), or 416's own list (`_PyCodec_LookupTextEncoding`'s
   falsy-flag branch, the binary-transform codecs' `_is_text_encoding=False`, a non-CPython
   interpreter).
2. **CE-1's closure.** (a) `messages.py:30`, `chat.py:31` or `count_tokens.py:23` losing the
   `LookupError` name or being replaced by a `ValueError`-rooted clause; (b) any site starting to
   import `aiohttp.payload`'s `LookupError` (N2 — would silently dead all three arms); (c)
   `request.json()` replaced by `read()` + local parse at any site; (d) `logging_middleware`
   (`server.py:48-52`) or the registration set (`:133-142`) changing which handlers see the
   boundary, or a package-level error middleware appearing; (e) `charset` becoming validated before
   `web_request.py:689`.
3. **The response-side settlement.** Any change to `_error_body` (`messages.py:187-198`) — dropping
   either name, dropping `errors="replace"`, or removing the `read()` fallback; any site passing
   `errors=` or `encoding=` to `text()` (N4); any change to `compaction.py:41`/`:55` or to session
   construction adding a `fallback_charset_resolver`; **new:** any change at `messages.py:117` back
   from `result.text` to a `body`-getter or `str(result)` read (N6's regression), or at
   `messages.py:162-166` to the `content_type` the error response is built with (dimension 4's
   round trip above is stated for a charset-less `application/json`).
4. **Dimension 4's matched-text claim.** Any change to `config.py:122-123` making the patterns match
   non-ASCII, to `utils.py:123-140`, to `messages.py:311`'s decode operand, or to
   `web_response.py:726-730` / `payload.py:380-457` (the `.text` round trip — **N11**); 403's
   unresolved U+FFFD-count point becoming decisive for any consumer.
5. **Notes' premises and the delta's ordering.** `strip_billing_nonce` ceasing to be the sole-system-
   writer path it delegates to (`normalize.py:64-74` → `_normalize_system`), any new writer of the
   `system` text appearing in the steps `normalize(body, session_id="")` runs, `count_tokens.py:87-88`
   reordering keying relative to the strip, `session_id`'s operand widening beyond `system`
   (claim 11's own invalidation — S1/S14 territory but this seam's N10 premise), `normalize.py`'s
   guards ceasing to tolerate `{}`, `server.py:125` dropping `client_max_size=0` (N5(i)), edits to
   `chat.py:162-165` (N5(ii)) or to the non-mapping-body shapes (N5(iii)), any codec registration
   under `buster_ripper/` or a runtime dependency (N7), any change to the `messages.py:162-166`
   construction or `_error_body` that moves the N9 encode-failure window, or an upstream-facing
   change making a surrogate-capable declared codec reachable on the error path (N9).
6. **Boundary wording.** Any edit to S18's seam row — boundary value, subject list, member list or
   dimensions. This record traces the six named dimensions and the systematic checks and claims
   nothing beyond them; the row's `State`/`Routing`/`Permanent` cells are not this pass's to write.

## Checked At

2026-10-07T23:58:17Z

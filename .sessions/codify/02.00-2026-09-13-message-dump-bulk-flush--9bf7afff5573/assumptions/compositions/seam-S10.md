VERDICT: COUNTEREXAMPLE
PROMOTION: INELIGIBLE — no member row carries a permanent handle (all 32 rows show `Permanent | —` in the `claims.md` ledger), so the "permanent handles for all member claims" condition fails outright; rows 246/247 additionally record `Source Fingerprint: unavailable` (tree-wide negatives over 26 test files), so the ordered-handle+fingerprint digest is not closed over every member. Rows 209/210/211 are also self-declared `PROMOTION: INELIGIBLE` (package-wide / directory-wide surfaces). The composition verdict itself stands on exact production-source locators and is fingerprinted below over everything that *is* recoverable.

## Seam

**S10** — a header **name set** crossing two boundaries:

- **egress (client→proxy→upstream):** `web.Request.headers` → a name-filtering step → `headers=` on `session.post` / `session.request`.
- **ingress (upstream→proxy→client):** `ClientResponse.headers` → a name-filtering step → `web.Response` / `web.StreamResponse` headers.

Declared policy owner: `buster_ripper/utils.py`'s two helpers (`forward_headers` L18-24, `response_headers` L27-33) over `_HOP_BY_HOP` (L12-15). Declared authority for the eight-name target: **base (`buster_ripper_legacy.py`) parity**, not a standard's enumeration (seam row; members 207/219/220; retired member 208 has no replacement row).

**Trace mode: full trace.** No prior composition for S10 is on file (`compositions/` has S1-S9, S11-S14; the interrupted attempt wrote nothing), and every member verdict is current (32/32 `VERDICT: HOLDS`, re-verified against the ledger's `HOLD` state, and every cited working-tree byte still matches its recorded fingerprint). Nothing is carried forward.

## Composition

Name-set producers, transformations and consumers, as the members jointly describe them:

```
EGRESS   client bytes
         │  aiohttp C parser (324): recognized name → hdrs.py canonical istr; table miss → as-sent str
         │  pure-Python parser (326): every name → as-sent str            (327: raw_headers = wire bytes, unused)
         ▼
         web.Request.headers  — CIMultiDictProxy, CI *lookup*, iteration yields STORED keys (329)
         │
         ├─ POLICY OWNER  utils.forward_headers (205)          drop = {host, content-length} ∪ _HOP_BY_HOP (207) = 10 names
         │                  ⚠ zero call sites in the package (209); only a dead import at chat.py:14 (212)
         ├─ BASE          _forward_headers (220)               drop = {host, content-length} ∪ _HOP_BY_HOP = 10 names (219)
         └─ LIVE ×3       chat.py:102 / messages.py:63 / passthrough.py:16   (214/217/320)
                          drop = {host, content-length, connection, transfer-encoding} = 4 names
         ▼
         plain dict {k: v}  — case-SENSITIVE, last-writer-wins per stored key, no longer a CIMultiDict
         │   chat only: caller-level empty-Bearer delete over bare keys of the filtered dict (213/321)
         ▼
         session.post/request(headers=…) (211) → aiohttp ClientRequest.update_headers
            sets Host from URL first, then merges EVERY caller pair with .add() — no hop-by-hop filtering (234)
            computes Content-Length only when the caller supplied none (235)
         ▼
         upstream

INGRESS  upstream bytes
         │  C parser (325) canonicalizes recognized names; Python parser (326) preserves (328: raw_headers unused)
         ▼
         ClientResponse.headers — CIMultiDictProxy (329)
         │
         ├─ POLICY OWNER  utils.response_headers (206)   drop = _HOP_BY_HOP = 8 names; keeps content-length
         │                  ⚠ zero call sites, not even imported (210)
         ├─ BASE          _response_headers (222)         drop = 8 names; FORWARDS content-length
         ├─ LIVE ×2       messages.py:257-260 / passthrough.py:40-42  (215/217/320)
         │                drop = {connection, transfer-encoding, content-length} = 3 names
         └─ LIVE (whitelists + literal dicts, not filters — outside 215's and 320's subject sets)
                        chat.py:159-164 / :191 / :200 ; messages.py:141 / :147-152  ← name fields directly,
                           two of them injecting `Connection: keep-alive` downstream
         ▼
         plain dict {k: v}
         ▼
         web.Response / web.StreamResponse(headers=…) → CIMultiDict(dict) via _extend_items → add() per pair,
            no compaction (multidict 6.7.1 `_multidict_py.py`:620-641, :845)
            Response._start derives Content-Length only when absent (236);
            StreamResponse._prepare_headers honours a caller Connection and only fills DATE/SERVER/CONTENT-TYPE
         ▼
         client
```

The set arithmetic the members pin (all differences computed from 205/206/207/214/215/219/220/222):

| | policy owner | base | live package | live − owner |
|---|---|---|---|---|
| egress drop | 10 names | 10 names (equal, 219) | 4 names | **keeps 6:** `keep-alive`, `proxy-authenticate`, `proxy-authorization`, `te`, `trailers`, `upgrade` |
| ingress drop | 8 names | 8 names (equal, 219/222) | 3 names | **keeps 6** of those, and **drops `content-length`** which owner and base both keep |

So the two live policies are each a *different* strict subset of the declared owner in each direction, and the ingress delta is opposite-signed for `content-length`. That is the substance of the counterexamples below.

## Counterexample

Four concrete compositions. C1-C3 conflict on the seam's own stated authority (the helpers as single policy owner, plus base parity); C4 is the seam's census shape failing to reach a name decision that changes what the client receives.

### C1 — the declared policy owner does not own the boundary, and the live egress set forwards hop-by-hop names the owner and the base both drop

Exact input, `POST /v1/messages`:

```
Keep-Alive: timeout=5, max=99
Proxy-Authorization: Basic YWxhZGRpbjpvcGVuc2VzYW1l
TE: trailers
Upgrade: h2c
```

Path through verified claims: `messages.py:62-63` is one of the five filters (320) and its `skip_keys` is `{host, content-length, connection, transfer-encoding}` (214/217) — none of the four names above is in it, so all four survive into the plain dict; that dict is the `headers=` of `session.post` at `messages.py:133` (211). aiohttp's `ClientRequest.update_headers` (`client_reqrep.py`:1056-1079) merges **every** caller pair with `.add()` and filters nothing; only `Host` is special-cased (234). Same arithmetic for `chat.py:102`→`:128` and `passthrough.py:16`→`:33`.

Observable: upstream vLLM receives `Keep-Alive`, `TE`, `Upgrade` and a **`Proxy-Authorization` credential addressed to the proxy hop**. Under the declared owner (`forward_headers`, 205+207) and under base (`_forward_headers`, 220+219) all four are dropped. Nothing in the suite witnesses it (246).

Note the premise this breaks first: 209/210/212 jointly establish that neither helper has a call site anywhere under `buster_ripper/` (`forward_headers` only as a dead import at `chat.py:14`; `response_headers` not even imported), while 214/215 establish that the five live sites re-implement narrower lists inline. The seam's "single policy owner" is therefore not a description of the running system — it is inert code beside it. Any composition that reasons from the helpers' sets to boundary behavior is reasoning about code that does not execute.

### C2 — request-direction name collapsing is backend-dependent, and the caller-level empty-`Bearer` rule consumes the collapsed dict

`web.Request.headers` is case-insensitive for **lookup** (329), but the comprehension at `chat.py:102` (320) yields a **plain dict**, whose keys are the parser's stored spellings. Whether two wire lines with different casing collapse is decided by the parser backend, not by the filter:

- **C parser (the backend this repo actually runs — `_http_parser.cpython-313-darwin.so` is present in `.venv`, and `AIOHTTP_NO_EXTENSIONS` is set nowhere in the repo):** a recognized name is replaced by aiohttp's own canonical `istr` object (324; `hdrs.py`:96-99 etc.), so `Authorization` and `authorization` become the *same string key* and the dict comprehension keeps only the **last** value.
- **pure-Python parser:** every name keeps its sent spelling (326), so both keys coexist as distinct dict entries and both are re-added upstream (`update_headers` `.add()`, 234).

Exact input to `POST /v1/chat/completions`, in this wire order:

```
Authorization: Bearer sk-real
authorization: Bearer 
```

C-parser path: `chat.py:102` → `{"Authorization": "Bearer "}` — the real credential was silently discarded by the collapse; `chat.py:105` (`next` over bare keys, 321) finds that key; `chat.py:108` strips to `Bearer`, so `chat.py:109` deletes it (213). Upstream receives **no `Authorization` at all**, for a client that supplied a valid bearer. The rule 213 describes as deleting an *empty*-Bearer header in fact deletes a valid credential — because 213's own qualification ("the dict comprehension collapses duplicate `Authorization` lines to the last value, so at most one entry exists") is a property of the C parser (324), not of the filter, and it is the collapse, not the rule, that destroys the token.

Same input, `AIOHTTP_NO_EXTENSIONS=1`: both keys survive; `next()` returns `Authorization` with value `Bearer sk-real`, which is not in `("Bearer", "Bearer ")`, so nothing is deleted and the empty-bearer line the rule exists to strip is forwarded upstream as a second `authorization` header. Two different observable upstream header sets from one input, with neither path doing what the rule's subject says. Base's analogue is a third mechanism (caller-level `headers.get("authorization", "")` / `pop`, 221/224) on a transport that is not aiohttp.

Nothing witnesses it: 247 — no test in `tests/` mentions `Authorization`, `Bearer`, or any API-key vocabulary, and no test inspects the header kwargs handed to the upstream session.

### C3 — the ingress direction disagrees with its owner and its base in *both* signs

Exact upstream response on a passthrough request:

```
HTTP/1.1 407 Proxy Authentication Required
Proxy-Authenticate: Basic realm="vllm"
Keep-Alive: timeout=15
Upgrade: TLS/1.2, HTTP/1.1
Content-Length: 1234
```

Path: `passthrough.py:40-42` drops only `connection`, `transfer-encoding`, `content-length` (215/217/320); the other three survive into `web.Response(headers=resp_headers)` at `:45-49`. multidict re-adds them without compaction; `StreamResponse._prepare_headers` (`web_response.py`:472-535) writes them as-is. `Response._start` (`:788-804`, 236) then derives `Content-Length` from `body=content` because the header is absent.

Observable, sign 1: the client receives `Proxy-Authenticate`, `Keep-Alive` and `Upgrade` — names the owner (`response_headers`, 206+207) and base (`_response_headers`, 222+219) both strip. Sign 2, opposite: the client receives a **recomputed** `Content-Length`, where owner and base both forward the upstream's own value.

Sign 2 is currently harmless and its harmlessness is not owned by either policy: both live filter sites relay `resp.read()` verbatim, so upstream length == body length. The one place in the package that *re-serializes* a body (`chat.py:187` / `:196`) sits on a whitelist path, not a filter path. So the equivalence between "adopt the owner" and "keep the live filter" is incidental to which body a site relays. If `response_headers` (206, i.e. base parity per 222) is adopted as the owner for the filter sites, an upstream that sends both `Transfer-Encoding: chunked` and `Content-Length` — `transfer-encoding` is in the owner's drop set, `content-length` is not — yields a client-visible `Content-Length` taken from the lying upstream value, which 236 confirms aiohttp will honour instead of deriving (`if hdrs.CONTENT_LENGTH in self._headers:` short-circuits the derivation). Base had the same exposure because base forwards `content-length` **and** re-derives nothing at the four `_response_headers` call sites (222, corroborated at `legacy:718/891/913`). Adopting the owner therefore *removes* a protection the live filters provide, in the name of parity.

### C4 — the seam's rename-proof census shape does not reach the names that actually reach the client

320's subject set is header-collection **comprehensions** iterating `.items()`, and it explicitly excludes the package's literal header dicts; 215 explicitly excludes the two `chat.py` whitelists from "filter". Composing that scope with the ingress direction:

- `messages.py:145-153` builds a streaming `web.StreamResponse` whose literal headers include `"Connection": "keep-alive"` (`:151`); `chat.py:161-164` does the same (`:163`).
- `web_response.py`:529-535 only *fills* `Connection` when absent, and `_write_headers` :536-546 writes `self._headers` verbatim — so a caller-supplied `Connection` is honoured and reaches the wire.
- Base's streaming branch sends `{"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}` with no `Connection` (`legacy`:691-696, the downstream literal 322 enumerates).

Observable: a client of `POST /v1/messages` with `stream: true` receives a `Connection: keep-alive` field that neither the base nor any filter produced, on the very name `response_headers` is defined to drop (206/207). The seam's "single policy owner + five sites + the one decision the shape does not reach (321)" accounts for exactly one unreached name decision; the census leaves **five** response-direction header-naming sites unreached (`chat.py`:159-160, `:161-164`, `:191`, `:200`; `messages.py`:147-152 — two of which carry the hop-by-hop name `Connection`), and a further two that name `Content-Type` by kwarg rather than by dict (`messages.py`:141, and the `web.json_response` sites). Membership of the response-direction name set is therefore not enumerable by the shape the seam says it is enumerable by.

### Latent (not realized today, recorded because the seam names the dimension)

`forward_headers` and `response_headers` have the identical `.items()` comprehension shape (320 reaches both) and differ only in the extra `("host", "content-length", …)` terms of the egress tuple, so the **direction discriminator lives only in the function name and the tuple literal**. If `response_headers` (206: keeps `host`, keeps `content-length`) were ever used for egress, 234 shows aiohttp replaces its URL-derived `Host` with any caller key in `hdrs.HOST_ALL` (`hdrs.py`:119-120, all case permutations), so the client's original `Host` would be sent to `config.UPSTREAM` instead of the configured host — a vhost-level misroute of the upstream request; and 235 shows a caller-supplied `Content-Length` is respected rather than recomputed, while `chat.py:34` has already rewritten the body, so the length would describe the pre-transform payload. Currently unreachable only because 209/210/212 leave both helpers uncalled.

## Dimension Coverage

| # | Dimension (named by the seam row) | Result |
|---|---|---|
| D1 | eight-name hop-by-hop set membership (205/206/207/219) | Traced. Sets are identical across package and base (219), 8 members, all lowercase, `frozenset`. **Effectiveness caveat:** `_HOP_BY_HOP`'s `trailers` (207) has no wire referent — aiohttp's table canonical is `Trailer` (`hdrs.py`:97, no `TRAILERS` constant), and under 324 the C parser rewrites `Trailer`→`Trailer` and only a *table-miss* plural keeps `Trailers` (326 likewise preserves either). So 7 of the 8 names can match, `trailers` matches only a nonstandard spelling. Parity holds (base is identical, 219) — reported as coverage, not a conflict. |
| D2 | request direction vs response direction (205 vs 206; 220 vs 222) | **COUNTEREXAMPLE** — C3 (opposite-signed `content-length`), plus the latent direction-swap hazard. |
| D3 | five live sites' local skip sets vs the shared helper's drop sets (214/215 vs 205/206, with 209/210/212) | **COUNTEREXAMPLE** — C1: the live egress drop is a 4-name subset of a 10-name owner set; six names are in the delta, five of them reachable on the wire (D1), four demonstrated by C1's input. The "single policy owner" premise is contradicted by 209/210/212 directly. |
| D4 | lowercase-key matching vs the spelling aiohttp exposes — canonicalized in `.headers`, peer's own spelling only in `raw_headers`, one backend only (217 with 324-329) | **COUNTEREXAMPLE** — C2. Matching itself is backend-independent and clean for both parsers (217: every filter tests `k.lower()`, every literal is lowercase; canonical spellings all lowercase-fold correctly, incl. `TE`→`te`). What is *not* clean is the set that survives matching: the filter output is a plain dict of stored keys (329's CI applies only inside multidict; multidict 6.7.1 `istr` at `_multidict_py.py`:36-40 does not override `__eq__`/`__hash__`), so duplicate-name collapsing is C-parser-only and case-sensitive otherwise. `raw_headers` (327/328) is consulted by no filter (320) → no bytes/str namespace mixing. |
| D5 | `host` + `content-length` dropped on egress and recomputed by aiohttp (205/214 + 234/235/236) | Traced clean for the package. All three egress filters drop `host`, so `update_headers`' URL-derived Host always wins (234, and `HOST_ALL` is unreachable because the drop is case-insensitive — 217). All three drop `content-length`, so 235's "caller value respected, not recomputed" branch cannot fire on a stale inbound length; `json=body` / `data=body` supplies the payload length, and a body-less non-GET gets `Content-Length: 0`. Base's explicit `headers["content-length"] = str(len(body))` (`legacy`:834, 322's impurity caveat) reaches the same wire effect through a different owner — parity by different mechanism. |
| D6 | empty-`Bearer` `Authorization` deletion route-local vs absorbed into the shared helper (213/321 vs 205/206; base 221/224) | Traced. Today route-local in both implementations, and neither helper implements any auth rule (205/206), so absorbing it would extend the deletion to `/v1/messages` and the passthrough — where base does **not** delete (224: only inside its chat-completions handler). Parity therefore *forbids* absorption into the declared owner; the seam's owner premise and its base-parity authority conflict on this dimension. Realized behavior of the route-local rule is nonetheless backend-dependent → C2. |
| D7 | base parity of both directions, incl. base forwarding `content-length` (219/220/221/222/322/323/224) | **COUNTEREXAMPLE** — C1 (egress 6-name delta), C3 (ingress 6-name keep + `content-length` drop), C4 (downstream `Connection` base never sent). Parity holds on D1, D5, D6 and on the sets themselves (219). |
| D8 | the eight-name target's authority = base parity, not a standard's enumeration (207/219/220; retired member 208 unreplaced) | Traced clean *as an authority statement*: no member claims RFC completeness, and every delta above is stated against base, not against RFC 9110. Consequence to record: because the only surviving authority is base, the `Trailer`/`trailers` ineffectiveness (D1) and the leaked hop-by-hop names are **not** detectable as set-membership errors — only as owner/live divergence. Composition cannot say whether a standard-correct target is wanted; no member row remains that could. |
| D9 | `.items()` comprehension shape as rename-proof census, and the one decision it does not reach (320/321) | **COUNTEREXAMPLE** — C4: the shape reaches 5 filter sites (+2 inert helpers) but the response-direction name set also depends on five header-naming sites outside it (`chat.py`:159-160/`:161-164`/`:191`/`:200`, `messages.py`:147-152), two of which carry the hop-by-hop name `Connection`. Direction-blindness of the shape also feeds the latent hazard in D2. 321's own site (`chat.py:105`, bare-key iteration of the filtered dict) is confirmed as outside the shape and confirmed to interact with D4 → C2. |
| D10 | which base-file requests the shared filter owns; which header-less request is not a proxy boundary (322/323) | Traced clean. Base: 4 header-carrying upstream requests all built through `_forward_headers` (656/812/901 covering 665/701/844/903), 1 header-less `/metrics` GET (`_poll_kv_cache`, 323) correctly outside name-set ownership; `count_tokens` issues none. |
| D11 | non-route modules issue no header-bearing request; `count_tokens` issues no upstream request (211/218) | Traced clean and consistent with D10: package's two out-of-routes requests (`backend.py`:54-56, `compaction.py`:36-39) pass no `headers=`, and the single shared `ClientSession` (`server.py`:65-68) sets no default header bag, so no request inherits names either. Package `count_tokens` answers locally (218). Delta vs the base file: the package's `/v1/models` probe (`backend.py`:54-56) has no counterpart among the five outbound sites 322 enumerates in the base file — also header-less, so also outside the name set, and not a proxy boundary. |
| D12 | absence of any test witness in either direction (246/247) | Traced clean as an absence, and load-bearing for C1-C4: no assertion in `tests/` over a hop-by-hop name at a proxy boundary, none over any `Authorization` value, and no test inspects the header kwargs passed to the upstream session at all. Every counterexample above is therefore unfalsified by the suite. These two rows are also the reason promotion cannot be completed (no fingerprints). |

## Sources

Production sources only (no plan, handoff, QA or session material consulted; no test used as evidence for behavior; nothing run).

Package under `repo:` subjects:
- `buster_ripper/utils.py`:12-15 (`_HOP_BY_HOP`), :18-24 (`forward_headers`), :27-33 (`response_headers`)
- `buster_ripper/routes/chat.py`:11-17 (dead `forward_headers` import at :14), :34-36, :98-111 (`_prepare_headers`; filter :101-102, auth lookup :105-109), :125-132, :155-169 (stream whitelists), :188-201 (re-serialized body + literal `Content-Type`)
- `buster_ripper/routes/messages.py`:61-70, :121-134, :138-142, :145-153 (literal `Connection` at :151), :223-235, :257-265
- `buster_ripper/routes/passthrough.py`:14-16, :29-49
- `buster_ripper/routes/count_tokens.py`:1-10, :28-85
- `buster_ripper/backend.py`:54-56 · `buster_ripper/compaction.py`:36-39 · `buster_ripper/server.py`:65-68

Base file under `repo:buster_ripper_legacy`:
- `buster_ripper_legacy.py`:178-189 (`_HOP_BY_HOP`), :207-227 (`_poll_kv_cache`, request :213-214), :621-627 (`_forward_headers`, tuple :626), :630-636 (`_response_headers`, test :635), :639-719 (`proxy_messages`; :656, :665-669, :691-696, :701-704, :715-719), :726-792 (`count_tokens`), :794-892 (`chat_completions`; :812, :834, :838-841, :844, :882-884, :891), :898-913 (`passthrough`; :901, :903, :913)

`pkg:aiohttp@3.13.3` (installed copy read from `.venv/lib/python3.13/site-packages/aiohttp/`; its `client_reqrep.py`, `hdrs.py` and `web_response.py` bytes hash-match the digests recorded by members 234/235/236/324/325/328, i.e. same bytes as the consulted upstream tag/sdist):
- `client_reqrep.py`:1056-1079 (`update_headers` — Host from URL first, every caller pair `.add`ed, no hop-by-hop filtering)
- `hdrs.py`:50, :54, :69, :75, :83, :84, :96, :97 (`TRAILER = istr("Trailer")`), :98, :99 (canonical spellings), :119-120 (`HOST_ALL`)
- `web_response.py`:472-535 (`StreamResponse._prepare_headers` — caller `Connection` honoured), :788-804 (`Response._start` — explicit `Content-Length` short-circuits derivation)
- `_http_parser.cpython-313-darwin.so` present (C parser is the available/default backend for this install)

`multidict@6.7.1` (consulted by 326; installed `_multidict_py.py` bytes hash-match 326's recorded digest):
- `_multidict_py.py`:36-40 (`class istr(str)` — no `__eq__`/`__hash__` override, so CI lives only in multidict's entry identity), :620-641 + :845 (`_extend_items` → `_add_with_hash`: building a `CIMultiDict` from a plain dict preserves case-differing keys as separate entries)

Negative-surface searches (bounded roots, as recorded by the members): `buster_ripper/` reference scan for `forward_headers`/`response_headers`/`_HOP_BY_HOP`/`headers`/request call sites (209/210/211); `buster_ripper/routes/*.py` full reads + filter scans (214/215/217/320/321); repo-wide search for `AIOHTTP_NO_EXTENSIONS` (none).

## Source Versions

- `buster_ripper/` and `buster_ripper_legacy.py`: local working tree under the supplied source root. No VCS revision consulted (git metadata is out of scope for this role); stability is carried by the byte digests below. Legacy file as read: 1013 lines / 42129 bytes (mtime 2026-03-03), standalone FastAPI+httpx script.
- `aiohttp` **3.13.3** (`.venv/…/aiohttp-3.13.3.dist-info` `Version: 3.13.3`; the pin 234/235/236/324/325/327/328/329 record from `uv.lock`). Installed `client_reqrep.py` / `hdrs.py` / `web_response.py` digests are identical to the digests those members recorded from the upstream tag/sdist.
- `multidict` **6.7.1** (`multidict-6.7.1.dist-info`), matching 325/326's pin.
- Members 329's establishing evidence is aiohttp 3.13.3 **documentation** (version-matched), corroborated at 3.14.4; multidict's "latest" page is corroboration only.

## Source Fingerprint

Ordered member handles + member-recorded source fingerprints + consulted production source bytes:

`sha256:482e1a07988d99471715dac4557cbd9672917c1c5a0651cebf4e5a545f83ebed`

Computed over the ordered member handles (`claim-<row>`) plus each member's recorded source fingerprints — `FINGERPRINT-UNAVAILABLE` appears only for 246/247, which record none — followed by `<path>|<sha256>` for each consulted production source.

CANONICAL PREIMAGE (exact bytes fed to the digest; one line per entry, `\n` line endings, no trailing spaces):

```
SEAM S10
claim-205|dd60987f7ef0741e55712e0469e854500a80318dedc70c130645ccb51bbc0d00
claim-206|dd60987f7ef0741e55712e0469e854500a80318dedc70c130645ccb51bbc0d00
claim-207|dd60987f7ef0741e55712e0469e854500a80318dedc70c130645ccb51bbc0d00
claim-209|dd60987f7ef0741e55712e0469e854500a80318dedc70c130645ccb51bbc0d00,16c7e99938b55e508b36c16092e7a89883baeff293385c3dfa3e9c161b413cae
claim-210|dd60987f7ef0741e55712e0469e854500a80318dedc70c130645ccb51bbc0d00
claim-211|c2b71468af6b53efde753b0648d47eb6e06c8db4c738e08f7c01f4e4682d76ef,ac9d42143c98694a0b6672dd753afcef537fc5e03ec747b4bb3c1208e23535e7
claim-212|16c7e99938b55e508b36c16092e7a89883baeff293385c3dfa3e9c161b413cae
claim-213|16c7e99938b55e508b36c16092e7a89883baeff293385c3dfa3e9c161b413cae
claim-214|16c7e99938b55e508b36c16092e7a89883baeff293385c3dfa3e9c161b413cae,721a2afe4d1ca35da9461c95f5c6dfe48dcf04ec4c4c0a606763cc29b0ca96da,072a58096586fcb3cf94c3cde37357bcee22abdffd8d89ae5952cfc4437494b2,dd60987f7ef0741e55712e0469e854500a80318dedc70c130645ccb51bbc0d00,15f9c46d3a2f459030e372843dc6857a2848046f541d877f3393d74cde2a5cc4
claim-215|721a2afe4d1ca35da9461c95f5c6dfe48dcf04ec4c4c0a606763cc29b0ca96da,072a58096586fcb3cf94c3cde37357bcee22abdffd8d89ae5952cfc4437494b2,16c7e99938b55e508b36c16092e7a89883baeff293385c3dfa3e9c161b413cae,a69c9884cad5997e442a4eaa5c5557f13a589b227777b71fbe1c4167c38fa203,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
claim-320|072a58096586fcb3cf94c3cde37357bcee22abdffd8d89ae5952cfc4437494b2,721a2afe4d1ca35da9461c95f5c6dfe48dcf04ec4c4c0a606763cc29b0ca96da,16c7e99938b55e508b36c16092e7a89883baeff293385c3dfa3e9c161b413cae,a69c9884cad5997e442a4eaa5c5557f13a589b227777b71fbe1c4167c38fa203,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
claim-321|16c7e99938b55e508b36c16092e7a89883baeff293385c3dfa3e9c161b413cae
claim-217|16c7e99938b55e508b36c16092e7a89883baeff293385c3dfa3e9c161b413cae,721a2afe4d1ca35da9461c95f5c6dfe48dcf04ec4c4c0a606763cc29b0ca96da,072a58096586fcb3cf94c3cde37357bcee22abdffd8d89ae5952cfc4437494b2,a69c9884cad5997e442a4eaa5c5557f13a589b227777b71fbe1c4167c38fa203,e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
claim-218|a69c9884cad5997e442a4eaa5c5557f13a589b227777b71fbe1c4167c38fa203
claim-219|dd60987f7ef0741e55712e0469e854500a80318dedc70c130645ccb51bbc0d00,7967563dc3ab4f66f0dc322c7945c5647b96c9f9a77fbafe3cd1d4780e8764bd
claim-220|7967563dc3ab4f66f0dc322c7945c5647b96c9f9a77fbafe3cd1d4780e8764bd
claim-221|7967563dc3ab4f66f0dc322c7945c5647b96c9f9a77fbafe3cd1d4780e8764bd
claim-222|7967563dc3ab4f66f0dc322c7945c5647b96c9f9a77fbafe3cd1d4780e8764bd
claim-322|7967563dc3ab4f66f0dc322c7945c5647b96c9f9a77fbafe3cd1d4780e8764bd
claim-323|7967563dc3ab4f66f0dc322c7945c5647b96c9f9a77fbafe3cd1d4780e8764bd
claim-224|7967563dc3ab4f66f0dc322c7945c5647b96c9f9a77fbafe3cd1d4780e8764bd
claim-234|7844440eb67433c6454edd708c776e47e3fcfec9b8d0aeb980dcfe88c1986ac9,7ce41fc1c2142f5346015455e030e3ebec22a41cde0fc299a66ce13be2cb2a9e,7db46d28f612a916ed01daff82aa63176fb8b08d4448bf1d3c317ec58fe600c6,dab8f933203eeb245d60f856e542a45b888d5a110094620e4811f90f816628d1
claim-235|7844440eb67433c6454edd708c776e47e3fcfec9b8d0aeb980dcfe88c1986ac9,3ba9ec60d50b2fb01e33673227a4d85fbde77159cbe71270b7901e831c0c2aac
claim-236|3ca73388d5382e67ed5ea295be844cac06ce7950a5a5237e8b39c7b2259ece65
claim-324|f9823c608638b8a77fec1c2bd28dc5c043f08c775ec59e7e262dd74b4ebb7384,b3c52b9f1bfc08377636e2821330ac2d6cb835b5455df2cc36583eb366c28370,dab8f933203eeb245d60f856e542a45b888d5a110094620e4811f90f816628d1,9fbd35936f1d54f8f0467c798fa2e92612d37e3eddaaedaf26dedfd0eeb43b28,b3a79dac016fe1a618fd2e1d5a708ca9c632df6f2427e79c46666488304311dd,cfa763643394b3b85d3f3a654c4b00572cf4a1ffab400c13ef1cfc3a95e19ee6
claim-327|f9823c608638b8a77fec1c2bd28dc5c043f08c775ec59e7e262dd74b4ebb7384,2aaacea7a01e581e5eead2ab1b9e4ba3b65bc2ae3d0f176b2a78ae5b6b76bb4e,eacf5d33399aa96efb6f33354f5d75b863522c5a3a80d99f0e0ed7cd89c0f319,cfa763643394b3b85d3f3a654c4b00572cf4a1ffab400c13ef1cfc3a95e19ee6
claim-325|f9823c608638b8a77fec1c2bd28dc5c043f08c775ec59e7e262dd74b4ebb7384,b3c52b9f1bfc08377636e2821330ac2d6cb835b5455df2cc36583eb366c28370,dab8f933203eeb245d60f856e542a45b888d5a110094620e4811f90f816628d1,9fbd35936f1d54f8f0467c798fa2e92612d37e3eddaaedaf26dedfd0eeb43b28,b3a79dac016fe1a618fd2e1d5a708ca9c632df6f2427e79c46666488304311dd,cfa763643394b3b85d3f3a654c4b00572cf4a1ffab400c13ef1cfc3a95e19ee6
claim-328|7844440eb67433c6454edd708c776e47e3fcfec9b8d0aeb980dcfe88c1986ac9,f9823c608638b8a77fec1c2bd28dc5c043f08c775ec59e7e262dd74b4ebb7384,cfa763643394b3b85d3f3a654c4b00572cf4a1ffab400c13ef1cfc3a95e19ee6,e7afd6b4b4ad6460453d82b3804818eafdb8264855d72e891269aec5e253d92a,7ce41fc1c2142f5346015455e030e3ebec22a41cde0fc299a66ce13be2cb2a9e
claim-326|cfa763643394b3b85d3f3a654c4b00572cf4a1ffab400c13ef1cfc3a95e19ee6,34a63c28a8a427d5a0c50aa15bd2f0ececc0d67181e74fe65666daaec48cacd6,546439f0fed539dea545fdd65408a76fad9a0f5e833dd01646debcaa68893171,0bd9654ca72c8a152954d5f454a44dd0fe6c16da516c31e44d25a790b5d35cbc
claim-329|638fae090c8b220a96fae8fef2e44bac2aebf7cf3ac15a952ad8381b1938e48f,a56614286d9c86b6ca347057247299a7740d1ab6b417178ca46d6467a1d77fc6,e237bb0eee54dfd30097b612cd43f151a87a0f2ccbe3a0a2047433df05d10e67,e6bfb75424c58926afd31baea5ddd948282df58180e19dd93e8d4a54e5d469b3,64ab8621ff9bfed2186efdc33fd2cb47bf82a4a8dbbcb8f0cc119efc79306214
claim-246|FINGERPRINT-UNAVAILABLE
claim-247|FINGERPRINT-UNAVAILABLE
buster_ripper/utils.py|dd60987f7ef0741e55712e0469e854500a80318dedc70c130645ccb51bbc0d00
buster_ripper/routes/chat.py|16c7e99938b55e508b36c16092e7a89883baeff293385c3dfa3e9c161b413cae
buster_ripper/routes/messages.py|721a2afe4d1ca35da9461c95f5c6dfe48dcf04ec4c4c0a606763cc29b0ca96da
buster_ripper/routes/passthrough.py|072a58096586fcb3cf94c3cde37357bcee22abdffd8d89ae5952cfc4437494b2
buster_ripper/routes/count_tokens.py|a69c9884cad5997e442a4eaa5c5557f13a589b227777b71fbe1c4167c38fa203
buster_ripper/routes/__init__.py|e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
buster_ripper/backend.py|c2b71468af6b53efde753b0648d47eb6e06c8db4c738e08f7c01f4e4682d76ef
buster_ripper/compaction.py|ac9d42143c98694a0b6672dd753afcef537fc5e03ec747b4bb3c1208e23535e7
buster_ripper/server.py|a9e49823ea456af9c0076781c4713c956745b21b026dfa768faf947e0b130592
buster_ripper_legacy.py|7967563dc3ab4f66f0dc322c7945c5647b96c9f9a77fbafe3cd1d4780e8764bd
.venv/lib/python3.13/site-packages/aiohttp/client_reqrep.py|7844440eb67433c6454edd708c776e47e3fcfec9b8d0aeb980dcfe88c1986ac9
.venv/lib/python3.13/site-packages/aiohttp/hdrs.py|dab8f933203eeb245d60f856e542a45b888d5a110094620e4811f90f816628d1
.venv/lib/python3.13/site-packages/aiohttp/web_response.py|3ca73388d5382e67ed5ea295be844cac06ce7950a5a5237e8b39c7b2259ece65
.venv/lib/python3.13/site-packages/multidict/_multidict_py.py|546439f0fed539dea545fdd65408a76fad9a0f5e833dd01646debcaa68893171
```


## Invalidates On

Member-level (any one forces a full re-trace of the dimensions its anchor participates in — none of these rows has a permanent handle, so the fingerprints above are the anchors):

- `utils.py` (`dd60987f…`) — 205/206/207/209/210/214/219: edits to `:12-15`, `:23`, `:32`, or a **new call site for either helper** (which would make D3's premise true and C1's first half dissolve, while making the D2/C3 `content-length` exposure live).
- `routes/chat.py` (`16c7e999…`) — 212/213/321/320/214/215/217: `_prepare_headers` delegating to `forward_headers`; `:101` skip-literal changes; `:105` becoming a `.items()` lookup, a lookup over `request_headers`, or moving above `:102`; `:159-164`/`:191`/`:200` literals changing (C4).
- `routes/messages.py` (`721a2afe…`), `routes/passthrough.py` (`072a5809…`) — 214/215/217/320: any `:62-63`/`:15-16` or `:257-260`/`:40-43` skip-list change; `messages.py:151` `Connection` removal changes C4.
- `routes/count_tokens.py` (`a69c9884…`), `routes/__init__.py` (`e3b0c442…`) — 218/214/215/320: a new file under `routes/` or any header handling in these two invalidates the exhaustive enumerations.
- `backend.py`/`compaction.py` (`c2b71468…`/`ac9d4214…`), `server.py` (`a9e49823…`) — 211: a `headers=` on either probe, or a default `headers=` on the `ClientSession` constructor.
- `buster_ripper_legacy.py` (`7967563d…`) — 219/220/221/222/224/322/323: any byte change (base parity is the authority for D1/D7/D8/D10, so a base edit re-signifies every delta above).

Boundary / cross-member conditions this composition specifically depends on:

1. **aiohttp off 3.13.3** (all digests above, plus members 234/235/236/324/325/326/327/328/329): a release where `update_headers` filters hop-by-hop names, or where `Response._start`/`StreamResponse._prepare_headers` overrides a caller `Content-Length`/`Connection`, removes C1's leak mechanism and C3's derivation short-circuit; a change to a `hdrs.py` canonical spelling (e.g. `TRAILER` → `Trailers`) re-activates D1's inert member.
2. **backend switch** — `AIOHTTP_NO_EXTENSIONS` set in the environment, or the C extension missing from the install, flips D4/C2 to 326's preserve-spelling behavior; multidict changing `istr` to override `__eq__`/`__hash__`, or `CIMultiDict(dict)` to compact on construction, also flips C2.
3. **the seam's own premise** — if a design ruling declares the five inline lists (or the whitelists/literals) the intended policy owner, or declares RFC 9110 §7.6.1 rather than base the target for the eight names, then D3/D7/D8/D9 re-trace against that contract and C1/C3/C4 become intention questions rather than counterexamples. Nothing in the supplied members can settle it; that is the missing contract, and it belongs to the owning decision route, not here.

## Checked At

2026-10-05T23:41:26Z

All 32 member verdicts carry a `Checked At` at or before 2026-10-05T21:12:41Z, and every cited working-tree byte was re-hashed at composition time and still matches the fingerprint its members recorded, so no member went stale between its own check and this composition.

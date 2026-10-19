VERDICT: COUNTEREXAMPLE
PROMOTION: ELIGIBLE

## Seam

**S15** — the boundary value is *one* `aiohttp.ClientResponse` handing the caller its payload
bytes and its header set together, at each relay boundary.

- Producer: `pkg:aiohttp@3.13.3` — `ClientSession` / `ClientResponse` (`client.py`,
  `client_reqrep.py`, `http_parser.py`, `_http_parser.pyx`), with `multidict 6.7.1`
  (`_multidict_py.py`) behind the header collection.
- Consumer: `repo:buster_ripper.utils#response_headers` (`buster_ripper/utils.py:29-56`) and the
  three relay sites that call it — `routes/passthrough.py:41`, `routes/messages.py:291`,
  `routes/chat.py:172`.
- Members (ledger order, all `VERDICT: HOLDS`, all re-verified current this pass): 384, 385, 386,
  387, 388, 389.
- Prior composition for this seam: `assumptions/compositions/seam-S15.md` (2026-10-06T19:05:05Z,
  `COUNTEREXAMPLE`, basis `sha256:74ac4658…`), member set unchanged. This record replaces it.

## Composition

Request posture → header-set selection → payload decoding → single object → consumer split.

1. **Posture** *(carried)*. The package's only session is built at `server.py:65-68` with `timeout=`
   and `connector=` and nothing else: no `auto_decompress=`, no session-level `headers=`.
   `auto_decompress` occurs package-wide only as prose inside `utils.py:47`'s docstring — never as a
   keyword at the constructor or at any request site (`skip_auto_headers`: zero occurrences). So 384
   fixes the session value at `True` and 385 makes every verb inherit it, because `None` is the
   "not passed" sentinel (`client.py:518`, `:601-602`, `:747`). **There is no branch in this repo
   where auto-decompression is off**, and no per-request variation of it; the two arms of dimension 1
   collapse to one at every site.
2. **What the client asks for** *(re-traced — the moved file owns this step)*. 389: with no
   session-level header set, only the request-level `headers=` can name `Accept-Encoding`. All three
   egress paths build that dict through `forward_headers` (`passthrough.py:17`, `messages.py:61`,
   `chat.py:104`), whose exclusion tuple at `utils.py:25` names `host`, `content-length`,
   `connection`, `transfer-encoding` and `*_HOP_BY_HOP` — `accept-encoding` is not among them. A
   caller-supplied value is therefore forwarded and suppresses the client default; a caller that
   names none gets aiohttp's own `gzip, deflate` default. Both arms ask for an encoded answer.
3. **Body** *(carried; library bytes identical)*. 386: when the upstream answers with a recognized
   `Content-Encoding`, the payload parser wraps the stream in a `DeflateBuffer` and hands the
   *inner, decoded* `StreamReader` to the message tuple (`http_parser.py:798-801`, `:458`;
   `_http_parser.pyx:460-473`), so `resp.read()`/`resp.content` are decoded bytes
   (`client_reqrep.py:689-707`).
4. **Headers** *(library carried, consumer link re-traced)*. 387: the same object's `headers` is the
   parser's verbatim `CIMultiDictProxy`; the encoding value was *copied* out to choose the
   decompressor and nothing removes it (`_http_parser.pyx:385-388`, `:419`, `:434-441`;
   `http_parser.py:556-558`, `:571`; `client_reqrep.py:565`, `:428-429`). The consumer keeps it:
   `response_headers`'s filter (`utils.py:52-56`) excludes only `content-length` plus the eight
   `_HOP_BY_HOP` names (`utils.py:14-17`), and `content-encoding` is end-to-end, not hop-by-hop. The
   function's own docstring (`utils.py:46-50`) states that it "does not reconcile" the label with the
   body it travels beside. So the label of the *wire* representation still travels with the *decoded*
   representation.
5. **Consumption** *(re-traced)*. Each site reads both halves of that one object in the same block —
   the body at `passthrough.py:38` / `messages.py:270` / `chat.py:149`, the headers at
   `passthrough.py:41` / `messages.py:291` / `chat.py:172 — and hands both to one
   `web.Response(body=…, headers=…)`. `response_headers` now returns a **`CIMultiDict`** built from a
   generator over `upstream_headers.items()` (`utils.py:52-56`), not a plain `dict`. That copy is
   occurrence-preserving: `MultiDict.__init__`'s non-mapping branch (`_multidict_py.py:635-641` →
   `_parse_args` `:820-838` → `_extend_items` `:845-848`) creates one `_Entry` per pair, and
   `StreamResponse.__init__` clones it whole (`web_response.py:114` `CIMultiDict(headers)`; the
   same-`_ci` fast path `:631-633`/`_from_md` `:643-647` preserves every entry). The server then
   writes one field-line per entry — `_py_serialize_headers` joins `headers.items()`
   (`http_writer.py:363-369`) and the C serializer actually loaded in this install
   (`_http_writer.cpython-313-darwin.so`) iterates the same view (`_http_writer.pyx:141-151`).
   `Response._start` re-derives `Content-Length` from the decoded body whenever the name is absent
   (`web_response.py:788-802`) and never revises `Content-Encoding` (`:472-535`; the only server
   writes of that name are the explicit `enable_compression` paths at `:434` and `:829`).

The seam's two halves therefore still disagree about which representation the bytes are in: the body
consumer needs the decoded form (step 3) and the header consumer asserts the encoded form
(steps 4-5), and no member claim supplies a reconciliation step between them. Step 5's change
repairs the *other* prior finding — dimension 4's occurrence fold — and does not touch this one.

## Dimension Coverage

Delta-scoped recomposition. **No member row changed:** all six verdicts pin library digests that
this pass re-verified byte-for-byte against the working tree (`client.py` `7ce41fc1…`,
`client_reqrep.py` `7844440e…`, `http_parser.py` `cfa76364…`, `_http_parser.pyx` `f9823c60…`,
`_multidict_py.py` `546439f0…`, METADATA `4913fdae…`), so the precondition for scoping holds and the
recomposition is not a full trace. **One consulted source moved:** `buster_ripper/utils.py`
`sha256:76de98c1…` → `sha256:9abaa6b8…` — the file that holds the seam row's named consumer,
`repo:buster_ripper.utils#response_headers`. It is the consumer-side anchor of dimensions 2 (which
arm of the union is broken at the consumer), 3 (relayed versus withheld), 4 (per-occurrence pairing)
and 5 (egress `Accept-Encoding` posture), so those four are re-traced in full. Scoping unit is
anchor intersection, not diff size: the edit rewrote the return type, the filter expression and the
docstring of the one function every finding chain passes through.

- **1. Session default versus per-request `auto_decompress` (384/385) — CARRIED** from
  `compositions/seam-S15.md` (2026-10-06T19:05:05Z, `COUNTEREXAMPLE`), which found no conflict.
  Both member files are byte-identical to that record's pins, and the repo premise the carry depends
  on was re-checked against the moved file's scope: the package-wide search for `auto_decompress`
  over the *new* `utils.py` bytes still returns only docstring prose (`utils.py:47`), and
  `server.py` — the only construction site — is byte-identical. Nothing in the delta reaches this
  dimension's outcome.
- **2. Encoded-on-the-wire versus decoded-as-handed-over (386) — RE-TRACED.** Union re-enumerated at
  `http_parser.py:556-558` / `_http_parser.pyx:434-441` (`enc.isascii() and enc.lower() in
  {"gzip","deflate","br","zstd"}`, exact match, no list parsing): `{gzip, deflate}` recognized and
  installed → decoded; `{br, zstd}` recognized but not installed in this venv (no `brotli`,
  `brotlicffi` or `backports.zstd` entry under `site-packages`; `compression_utils.py:16-33` → both
  `HAS_BROTLI`/`HAS_ZSTD` false) → `ContentEncodingError` at `DeflateBuffer.__init__`
  (`http_parser.py:981-993`, see N1); absent / `identity` / unrecognized / multi-valued or chained
  (`"gzip, br"`, or two separate `Content-Encoding` lines where `.get()` returns only the first) →
  wire bytes pass through untouched. Mapped and unmapped members take materially different paths, and
  exactly one class is broken at the consumer: only the *decoded* class is relabeled (step 4).
  **Covered by CE-1.**
- **3. `Content-Encoding` relayed versus withheld (387) — RE-TRACED.** The library retains it (387),
  and the re-traced consumer retains it again: `content-encoding` is absent from `_HOP_BY_HOP`
  (`utils.py:14-17`) and from the `response_headers` exclusion tuple (`utils.py:55`). It is relayed,
  not withheld. **Covered by CE-1.**
- **4. Single versus repeated occurrences, one pair per occurrence at `.items()` (388) —
  RE-TRACED; prior CE-2 CLOSED.** Producer guarantees one pair per occurrence (388, re-verified at
  `_multidict_py.py:98-102`, `:595-596`, `:722-724`, `:775-778`, `:1196-1198`); the consumer now
  preserves every occurrence through `CIMultiDict` construction, `web.Response`'s clone, and
  `write_headers` (step 5). The guarantee and the consumption now agree, so the prior counterexample
  does not compose against these bytes. The dimension's remaining open question is recorded as N5.
- **5. Caller-supplied versus client-default `Accept-Encoding` (389) — RE-TRACED.** Both arms still
  deliver an encoding request to the upstream: `accept-encoding` is not in `forward_headers`'
  exclusion tuple (`utils.py:25`), so a client value is forwarded (suppressing the default per 389's
  presence test), and a client that names none gets `"gzip, deflate"`. Neither arm forecloses CE-1.
  The request-side mirror is N2.
- **6. Systematic checks at this boundary — re-checked where the delta reaches them.** Stale,
  reordered or late messages: not applicable — the producer is one response object read inside a
  single `async with` scope. Selection changed or cleared between dispatch and settlement, and
  disposal or recreation between dispatch and settlement: the one shared object (`app["session"]`,
  `server.py:66`) has an immutable posture per dimension 1, and `response_headers` snapshots the
  header set into a new `CIMultiDict` before the context exits, so there is nothing to re-select,
  recreate or read-after-close. Duplicate identities across namespaces: in scope — it was the prior
  CE-2 and it is now closed (dimension 4); the mirror case, a name the upstream sent twice now
  reaching the wire verbatim, is N5. Dependency rejection or exception leaving a request unsettled:
  checked and settled — N1. More than one terminal settlement for one accepted request: not reachable
  from these six members; the relay sites' `except` arms and their settlement policy belong to S11's
  verified chain.

## Counterexample

**CE-1 — a body the client decoded is relayed downstream under the upstream's encoding label.**
*(Carried forward as still composable; re-traced on the moved consumer bytes.)*

Concrete input: a client `GET`s any passthrough endpoint (for example `/v1/models`,
`passthrough.py:11-20`) and the upstream answers `200`, `Content-Encoding: gzip`,
`Content-Type: application/json`, gzip-compressed JSON body. That answer is asked for either because
the client sent `Accept-Encoding: gzip, deflate` and `forward_headers` passed it through
(`utils.py:20-26`; `passthrough.py:17`, `messages.py:61`, `chat.py:104`), or because the client named
no `Accept-Encoding` at all and aiohttp added its own default (389).

Path: 384 → 385 (`server.py:65-68` names no `auto_decompress`, so `client.py:601-602` resolves it to
`True`) → 386 (`http_parser.py:798-801` wraps the stream, so `resp.read()` at `passthrough.py:38`
returns *decompressed* bytes) → 387 (`client_reqrep.py:565` leaves `resp.headers["Content-Encoding"]
== "gzip"`) → consumer (`utils.py:52-56` keeps the name, since `_HOP_BY_HOP` at `:14-17` does not
name `content-encoding`) → `web.Response(body=content, …)` at `passthrough.py:43-47`, whose
`_start` writes `Content-Length` from the decoded length (`web_response.py:797-802`) and never
revises `Content-Encoding` (`:472-535`).

Observable result: the downstream client receives a head that says gzip with a body that is plain
JSON, and a length that only matches the plain body. A conforming client attempts to inflate identity
bytes: `curl --compressed` reports an unrecognized/corrupt content encoding,
`requests`/`httpx`/the Anthropic and OpenAI SDKs raise a decompression error, and the proxied request
fails although the upstream succeeded. The same ordering at the other two sites:
`messages.py:270` + `:291-296` (which additionally *parses* those decoded bytes at `:272` and `:276`,
so that site needs the decoded form for its own logic while relaying the encoded label), and
`chat.py:149`/`:165` + `:170-172`. `chat.py:217`/`:221` is the strongest case: there the body is
re-serialized JSON (`json.dumps(resp_data).encode()`), so even byte-identity with the upstream is
gone while the `Content-Encoding` label is still forwarded.

The two streaming relays are the coherent contrast, which shows the conflict is a property of this
composition rather than of the library: `messages.py:151-161` + `_relay_after_commit`
(`messages.py:170-197`) and `chat.py:191-198` prepare a repo-authored head that carries no
`Content-Encoding` and then write the decoded chunks from `resp.content.iter_any()` — decompressed
bytes, unlabelled.

`Reach:` `buster_ripper/routes/passthrough.py:38-47` (`handle_passthrough`) and
`buster_ripper/routes/messages.py:270,291-296` (`_non_stream_single_request`) and
`buster_ripper/routes/chat.py:170-172` (`_chat_response`, reached from `:149`, `:165`, `:217`,
`:221`) — all read the value and put it on the wire; `buster_ripper/utils.py:52-56` is the shared
consumer that reads `resp.headers` and decides the label survives.

**CE-2 (prior record) — no longer composes.** The earlier finding was that per-occurrence pairs from
`resp.headers.items()` were folded to the last occurrence by a plain-`dict` comprehension. Against
these bytes the fold is gone: `response_headers` returns a `CIMultiDict` (`utils.py:52-56`), the
response constructor and serializer keep every entry (step 5), so `Set-Cookie: a=1` and `Set-Cookie:
b=2` from the upstream both reach the client. The only member claim that could previously be shown
conflicting at the consumer (388) now composes with it; `content-encoding`'s presence in the relayed
set is unaffected by the change, which is why CE-1 stands and CE-2 does not.

## Notes

Inert for the verdict above; each was checked against a cited production source.

- **N1 — `br`/`zstd` are recognized by the header but not decodable in this install, and the
  resulting rejection is settled.** `http_parser.py:557` / `_http_parser.pyx:438` list all four
  names; `DeflateBuffer.__init__` raises `ContentEncodingError` when `HAS_BROTLI`/`HAS_ZSTD` is false
  (`http_parser.py:981-993`), and neither codec package is present under
  `.venv/lib/python3.13/site-packages` (directory scan: zero `brotli`/`zstd` entries), so both flags
  are false (`compression_utils.py:16-33`). The raise happens inside `HttpParser.feed_data` from
  `Connection.data_received`'s guarded call (`client_proto.py:197-204`, `:313-329`), becomes an
  `HttpProcessingError`, and `ClientResponse.start` converts it to a `ClientResponseError`
  (`client_reqrep.py:536-544`) — a `ClientError` subclass. So the "dependency rejection leaves a
  request unsettled" check does **not** fire: all three relay sites' `except aiohttp.ClientError`
  arms (`passthrough.py:48`, `messages.py:297`, `chat.py:150`) catch it at the `async with` line.
  Recorded because it is a member of dimension 2's union whose outcome is *installation*-dependent and
  therefore not settled by any of the six member claims.
- **N2 — the request direction still has the fold the response direction lost.** `forward_headers`
  returns a plain `dict` (`utils.py:20-26`), so a client that sends the same request header on
  several lines reaches the upstream as one. For dimension 5 the consequence is immaterial (389 keys
  on *presence*, not on the merged value), and the egress direction is S10's boundary, not this one.
- **N3 — nothing in the six members is normative about the fix, and the consumer now says so.** The
  `response_headers` docstring (`utils.py:46-50`) names the non-reconciliation explicitly and records
  it as deliberate parity, deferred to a design ruling in `roadmap.md` §RM-008 — a design record this
  role does not read. That is provenance for how CE-1 should be *routed*, not a contract that makes
  the composed output self-consistent: the wire head and the wire body still disagree. Per the
  protocol, a finding is not wrong merely because it is surprising, and this one is not reported
  because either intended posture was chosen here.
- **N4 — the seam row's own wording** ("at each relay boundary") counts four relay boundaries: the
  three `response_headers` sites plus the two streaming relays. The streaming relays were traced under
  CE-1's contrast paragraph and carry no finding; `chat.py:187-190` (the stream-path error arm) copies
  a single `Content-Type` explicitly and never relays `Content-Encoding`, so it is coherent for the
  same reason the streaming heads are.
- **N5 — the newly preserved occurrences can now reach the wire as several lines of one name.** With
  the fold gone, an upstream that sends `Content-Type` twice (or `Link`, `Vary`, `Warning`) is
  relayed as two field-lines where the prior consumer emitted the last one only. 388 establishes that
  the producer keeps occurrences, and step 5 establishes that this consumer no longer drops them; no
  member claim, and no production source cited here, states whether verbatim multi-line relay or
  merging is required for a name HTTP treats as single-valued. `web.Response` itself is indifferent —
  it raises only when a *constructed* `content_type=`/`charset=` argument collides with a relayed
  `Content-Type` (`web_response.py:653-662`), and none of the three sites passes either
  (`passthrough.py:43-47`, `messages.py:292-296`, `chat.py:172`). Recorded as an open contract
  alongside a closed finding, not as a second counterexample: nothing in the traced sources shows a
  consumer that misroutes on it.
- **N6 — the type change from `dict` to `CIMultiDict` also changed the function's declared input
  union** (`dict | CIMultiDictProxy`, `utils.py:29`). Every traced call site passes the
  `CIMultiDictProxy` member of it; the `dict` member is unreachable from the relay boundaries in this
  seam. The `.lower()` key test (`utils.py:54-55`) behaves identically for `istr` keys and plain
  `str` keys, so no arm of that union takes a materially different path.

## Sources

Library (producer side):
- `.venv/lib/python3.13/site-packages/aiohttp/client.py` — `:274,299,399` (384);
  `:486,518,549,601-602,743-747,1207-1221,1287,1297,1317,1325,1333,1339` (385, 389)
- `.venv/lib/python3.13/site-packages/aiohttp/client_reqrep.py` — `:105-114,803-806,895-896,1081-1095`
  (389); `:428-429,536-544,565,689-707` (386, 387, N1)
- `.venv/lib/python3.13/site-packages/aiohttp/http_parser.py` — `:149,235,238,458,556-558,571,798-801,981-996`
  (386, 387, 388, dim 2, N1)
- `.venv/lib/python3.13/site-packages/aiohttp/_http_parser.pyx` — `:385-388,419,434-441,460-473`
  (386, 387, dim 2)
- `.venv/lib/python3.13/site-packages/aiohttp/compression_utils.py` — `:16-33` (dim 2, N1)
- `.venv/lib/python3.13/site-packages/aiohttp/client_proto.py` — `:197-204,227,313-329` (385, N1)
- `.venv/lib/python3.13/site-packages/aiohttp/streams.py`, `http_exceptions.py`,
  `client_exceptions.py` — the N1 exception channel
- `.venv/lib/python3.13/site-packages/aiohttp/web_response.py` — `:94-116` (`headers=` →
  `CIMultiDict`), `:434-435,472-535,829` (server-side head writing), `:628-687`
  (`Response.__init__` content-type/charset conflict, N5), `:788-802` (`Content-Length` derivation)
- `.venv/lib/python3.13/site-packages/aiohttp/http_writer.py` — `:210-219,363-377`
  (`write_headers`, `_py_serialize_headers`, C-extension selection)
- `.venv/lib/python3.13/site-packages/aiohttp/_http_writer.pyx` — `:125-151`
  (`for key, val in headers.items()` — one field-line per entry)
- `.venv/lib/python3.13/site-packages/multidict/_multidict_py.py` — `:98-102,595-596,620-647,715-724,775-778,788-848,1196-1198`
  (388, step 5)
- `.venv/lib/python3.13/site-packages/multidict-6.7.1.dist-info/METADATA` — `:76-78,108` (388)

Repo (consumer side):
- `buster_ripper/utils.py` — `:14-17` (`_HOP_BY_HOP`), `:20-26` (`forward_headers`),
  `:29-56` (`response_headers`, incl. the docstring posture statement at `:46-50`)
- `buster_ripper/server.py` — `:64-68` (the only `ClientSession`)
- `buster_ripper/routes/passthrough.py` — `:17,30-48`
- `buster_ripper/routes/messages.py` — `:61,151-161,170-197,265-297`
- `buster_ripper/routes/chat.py` — `:104,141-152,155-172,185-199,202-221`

## Source Versions

- aiohttp 3.13.3 (`aiohttp/__init__.py` `__version__ = "3.13.3"`; `aiohttp-3.13.3.dist-info`)
- multidict 6.7.1 (`multidict-6.7.1.dist-info`)
- CPython 3.13 environment under `.venv`; compiled extensions present
  (`aiohttp/_http_parser.cpython-313-darwin.so`, `aiohttp/_http_writer.cpython-313-darwin.so`,
  `multidict/_multidict.cpython-313-darwin.so`) — both parser implementations and both header
  serializers read, plus the pure-Python reference implementations the member verdicts rest on
- `buster_ripper` — working tree, digests below

## Source Fingerprint

Basis over the ordered member handles plus their verdict-recorded primary digests, plus the sha256 of
every source file consulted by this pass. The hashed input is the `Components:` block below verbatim
(25 lines, each newline-terminated, 3196 bytes); every digest in it was re-verified against the
working tree by this pass and all matched.

```text
sha256:d790d12b6513986bb6328589c229ae7c23656538588b26b9a8b2b0837987a673
```

Components:

```text
384 CL:aiohttp/auto-decompress-session-default            sha256:7ce41fc1c2142f5346015455e030e3ebec22a41cde0fc299a66ce13be2cb2a9e  (aiohttp/client.py)
385 CL:aiohttp/auto-decompress-request-inheritance        sha256:7ce41fc1c2142f5346015455e030e3ebec22a41cde0fc299a66ce13be2cb2a9e  (aiohttp/client.py)
386 CL:aiohttp/auto-decompress-read-decompressed          sha256:cfa763643394b3b85d3f3a654c4b00572cf4a1ffab400c13ef1cfc3a95e19ee6  (aiohttp/http_parser.py)
387 CL:aiohttp/auto-decompress-content-encoding-retention  sha256:f9823c608638b8a77fec1c2bd28dc5c043f08c775ec59e7e262dd74b4ebb7384  (aiohttp/_http_parser.pyx)
388 CL:aiohttp/repeated-header-items-iteration            sha256:546439f0fed539dea545fdd65408a76fad9a0f5e833dd01646debcaa68893171  (multidict/_multidict_py.py)
389 CL:aiohttp/default-accept-encoding-header             sha256:7844440eb67433c6454edd708c776e47e3fcfec9b8d0aeb980dcfe88c1986ac9  (aiohttp/client_reqrep.py)
consulted  aiohttp/client.py                   sha256:7ce41fc1c2142f5346015455e030e3ebec22a41cde0fc299a66ce13be2cb2a9e
consulted  aiohttp/client_reqrep.py            sha256:7844440eb67433c6454edd708c776e47e3fcfec9b8d0aeb980dcfe88c1986ac9
consulted  aiohttp/http_parser.py              sha256:cfa763643394b3b85d3f3a654c4b00572cf4a1ffab400c13ef1cfc3a95e19ee6
consulted  aiohttp/_http_parser.pyx            sha256:f9823c608638b8a77fec1c2bd28dc5c043f08c775ec59e7e262dd74b4ebb7384
consulted  aiohttp/client_proto.py             sha256:e7afd6b4b4ad6460453d82b3804818eafdb8264855d72e891269aec5e253d92a
consulted  aiohttp/streams.py                  sha256:ae5c0bede93a0a430c62297f7bf128916bf6eae1e6b738b794aaa59cb36b5c27
consulted  aiohttp/http_exceptions.py          sha256:0632310f82edad0832b6aa11e65388f7302db4d992ca046092c52c311cbbb2cb
consulted  aiohttp/client_exceptions.py        sha256:bb229bc48da979984a97b9442c1331dd47aeb0d91fa5e98f5a9185ab4afa25e3
consulted  aiohttp/compression_utils.py        sha256:849d8b5e1376396ba41479b96fbf1324518a70088bda4b618bd49fe4f4583be5
consulted  aiohttp/web_response.py             sha256:3ca73388d5382e67ed5ea295be844cac06ce7950a5a5237e8b39c7b2259ece65
consulted  multidict/_multidict_py.py          sha256:546439f0fed539dea545fdd65408a76fad9a0f5e833dd01646debcaa68893171
consulted  multidict-6.7.1.dist-info/METADATA  sha256:4913fdaae936a309c29b89b98c2dbebd237203fb27ab9e68ae5e4ec17ba3024f
consulted  buster_ripper/utils.py              sha256:9abaa6b8ccbc2bdf8aadab44e64f2a07236ecad619cfc4b98fe78e897c8fdf24
consulted  buster_ripper/server.py             sha256:a9e49823ea456af9c0076781c4713c956745b21b026dfa768faf947e0b130592
consulted  buster_ripper/routes/passthrough.py sha256:eebf21192eb1a73f52d8cd34a69cff32f12090515d5a6ea61de1e19c87366e4c
consulted  buster_ripper/routes/chat.py        sha256:d33ef8b81bccb966529b6548b2da14c3ee938e142bbea4ed73fc90584261d6f3
consulted  buster_ripper/routes/messages.py    sha256:c362d5e2e1e33761f3d12e16a9c7a4bbd8bae5b134f3cb82fb9f6e3f0ebbf238
consulted  aiohttp/http_writer.py              sha256:7db46d28f612a916ed01daff82aa63176fb8b08d4448bf1d3c317ec58fe600c6
consulted  aiohttp/_http_writer.pyx            sha256:56514404ce87a15bfc6b400026d73a270165b2fdbe70313cfa007de29ddd7e14
```

This record supersedes the prior basis `sha256:74ac46583a4976fab03d873c3a937eb1906f14476dd646d5cbd6422155d437ca`
(2026-10-06T19:05:05Z), whose `buster_ripper/utils.py` component pinned `76de98c1…`; the member-handle
components are identical, the consulted set grew by five files (`http_writer.py`,
`_http_writer.pyx`, `compression_utils.py`, `streams.py`, METADATA) and `utils.py` moved.

## Invalidates On

Any of the following requires recomposition of this seam — a member's own `Invalidates On` is not
sufficient, because the findings are joint:

1. **Member fingerprints.** Any of the six `CL:` digests above ceasing to match the bytes this
   composition hashed (`d790d12b…`): re-read that member's verdict rather than reuse this record.
   In particular any aiohttp release changing `client.py` `:299/:399` or `:518/:601-602`,
   `http_parser.py` `:556-558/:798-801/:981-996`, `_http_parser.pyx` `:385-388/:434-441`,
   `client_reqrep.py` `:565/:1081-1095`, `web_response.py` `:114/:631-647/:788-802`,
   `http_writer.py` `:363-377` / `_http_writer.pyx` `:125-151`, or multidict's `_multidict_py.py`
   entry/view iteration and `MultiDict.__init__`/`_parse_args`/`_extend_items` copy paths.
2. **CE-1's chain — any one link moving invalidates it.** (a) the package starting to pass
   `auto_decompress` (at `server.py:65-68` or per request), or aiohttp changing the session/per-request
   precedence of 384/385; (b) decompression moving out of the payload parser (386); (c) the response
   header set becoming reconciled with the decoded body — aiohttp deleting `Content-Encoding`, or any
   middleware doing it (387); (d) `_HOP_BY_HOP` (`utils.py:14-17`) gaining `content-encoding`, or
   `response_headers`'s exclusion tuple (`utils.py:55`) gaining any content-coding or body-consistency
   filter, or the function's own docstring statement at `utils.py:46-50` being contradicted by code
   that does reconcile the pair; (e) any of the three sites (`passthrough.py:43-47`,
   `messages.py:292-296`, `chat.py:172`) ceasing to pair `resp.read()` with
   `response_headers(resp.headers)` on one `web.Response`; (f) the egress `Accept-Encoding` posture
   moving — `forward_headers` (`utils.py:25`) dropping `accept-encoding`, a session-level default
   header set appearing at `server.py:65-68`, or `skip_auto_headers` being used (389's
   mentioned-but-unrowed arm; currently zero occurrences package-wide).
3. **Dimension 4 / the closed CE-2.** Any change to `response_headers`' return type (`utils.py:52`),
   to its generator-of-pairs construction shape (`utils.py:53-56`), to
   `MultiDict.__init__`/`_parse_args`'s per-pair entry creation, to `StreamResponse.__init__`'s
   `CIMultiDict(headers)` copy, or to either header serializer's per-entry iteration, reopens the
   fold; so does any change to 388's per-occurrence semantics. N5's open contract likewise moves with
   any site that starts passing `content_type=`/`charset=` alongside relayed headers
   (`web_response.py:653-662`).
4. **N1's premise.** Installing `brotli`/`brotlicffi` or `backports.zstd`, or an aiohttp release
   changing `HAS_BROTLI`/`HAS_ZSTD` handling or the `HttpProcessingError` → `ClientResponseError`
   conversion at `client_reqrep.py:536-544`, moves dimension 2's `{br, zstd}` arm from note to a live
   member of the decoded class and must be re-traced.
5. **Consumer-side drift.** Any edit to `buster_ripper/utils.py`, `server.py`,
   `routes/passthrough.py`, `routes/chat.py` or `routes/messages.py` at the locators cited above —
   this pass's scoping rule is member-anchor intersection with the change, and every finding chain
   here crosses `utils.py`.
6. **Boundary wording.** Any seam-row edit to S15's boundary value, subject list or dimension list;
   this record traces the five named dimensions only and claims nothing beyond them.

## Checked At

2026-10-06T20:33:47Z

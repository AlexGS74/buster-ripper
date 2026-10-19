VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
Key lookup in the header mappings aiohttp exposes as `web.Request.headers` and `ClientResponse.headers` is case-insensitive

## What it actually does
aiohttp documents both attributes as `CIMultiDictProxy`, and its docs say in so many
words that each is "a case-insensitive multidict proxy". So name lookup, containment,
and `.get()` on either mapping match the key regardless of the case used at the call
site (`headers["content-type"]` finds a header stored as `Content-Type`). multidict,
the library that supplies the type, states the semantics for both CI classes
(`CIMultiDict` and `CIMultiDictProxy`) with a worked example:
`dct = CIMultiDict(key='val'); 'Key' in dct` → `True`; `dct['Key']` → `'val'`.

Boundary worth noting: the docs distinguish this from `raw_headers`, described as
"HTTP headers of response as unconverted bytes, a sequence of `(key, value)` pairs"
— the case-preserving/unparsed view. Nothing consulted says how the *rendered* or
iterated key text is cased; only that lookup by name is case-insensitive.

## Source
Checked each named attribute against its own doc entry (not one leaning on the other):

- `web.Request.headers` — https://docs.aiohttp.org/en/v3.13.3/web_reference.html,
  entry `aiohttp.web.BaseRequest.headers`: "A case-insensitive multidict proxy with
  all headers. / Read-only `CIMultiDictProxy` property." The `aiohttp.web.Request`
  entry on the same page: "The class in derived from BaseRequest, shares all
  p[roperties]" — which is what makes that entry `Request.headers`.
- `ClientResponse.headers` — https://docs.aiohttp.org/en/v3.13.3/client_reference.html,
  entry `aiohttp.ClientResponse.headers`: "A case-insensitive multidict proxy with
  HTTP headers of response, `CIMultiDictProxy`."
- corroborating (type semantics, not establishing): https://multidict.aio-libs.org/en/latest/
  — "Case insensitive (`CIMultiDict` and `CIMultiDictProxy`) ones assume the keys are
  case insensitive".
- current stable agrees verbatim on both pages (docs.aiohttp.org/en/stable/, aiohttp 3.14.4).

Documentation-only claim: no local source root or dependency source root was supplied,
so installed package source was not read (and site-packages was not searched).

## Source Version
aiohttp **3.13.3** documentation — deliberately version-matched to this repo's lock
(`pyproject.toml` declares `aiohttp` unpinned; `uv.lock` resolves aiohttp 3.13.3,
multidict 6.7.1). Corroborated against aiohttp stable 3.14.4. The multidict page
consulted is "latest" (aiohttp stable cross-references it as multidict v7.0); pinned
here is multidict 6.7.1, whose versioned docs URL 404s, so multidict is corroboration
only and the verdict rests on the aiohttp pages above.

## Source Fingerprint
sha256 of the exact fetched bytes consulted:

- `https://docs.aiohttp.org/en/v3.13.3/web_reference.html` — `638fae090c8b220a96fae8fef2e44bac2aebf7cf3ac15a952ad8381b1938e48f`
- `https://docs.aiohttp.org/en/v3.13.3/client_reference.html` — `a56614286d9c86b6ca347057247299a7740d1ab6b417178ca46d6467a1d77fc6`
- `https://docs.aiohttp.org/en/stable/web_reference.html` (3.14.4) — `e237bb0eee54dfd30097b612cd43f151a87a0f2ccbe3a0a2047433df05d10e67`
- `https://docs.aiohttp.org/en/stable/client_reference.html` (3.14.4) — `e6bfb75424c58926afd31baea5ddd948282df58180e19dd93e8d4a54e5d469b3`
- `https://multidict.aio-libs.org/en/latest/` — `64ab8621ff9bfed2186efdc33fd2cb47bf82a4a8dbbcb8f0cc119efc79306214`

## Invalidates On
- aiohttp changes either doc entry's declared type away from `CIMultiDictProxy`
  (e.g. to `MultiDictProxy` or a plain `dict`) — the case-insensitivity claim dies
  with the type, not with the prose.
- `aiohttp.web.Request` ceasing to derive from `BaseRequest`, or gaining its own
  `headers` entry with different wording — then `web.Request.headers` needs its own check.
- multidict altering CI key matching for `CIMultiDictProxy`, or scoping its
  "assume the keys are case insensitive" statement to `CIMultiDict` only.
- this repo moving aiohttp off the 3.13.x/3.14.x documentation line whose bytes are
  fingerprinted above without re-reading both entries.

## Checked At
2026-10-05

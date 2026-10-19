VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
aiohttp's `auto_decompress` defaults to `True` on `ClientSession.__init__`

## What it actually does
`ClientSession.__init__` declares the keyword parameter `auto_decompress: bool = True`
and stores it verbatim as `self._auto_decompress`, so a session constructed without the
argument has `auto_decompress` true. The read-only property `ClientSession.auto_decompress`
returns that same stored value. No normalization, sentinel, or environment-derived default
intervenes between the signature and the attribute.

## Source
`/Users/alexhubitski/Projects/buster-ripper/.venv/lib/python3.13/site-packages/aiohttp/client.py`
— `class ClientSession`, `def __init__` (line 274):
- line 299: `auto_decompress: bool = True,` (signature default)
- line 399: `self._auto_decompress = auto_decompress` (stored unmodified)
- lines 1436-1438: `def auto_decompress(self) -> bool:` → `return self._auto_decompress`

Ranked-source note: the installed package ships no documentation (`no docs dir` under
`.../site-packages/aiohttp`) and no `.pyi` stubs, so per the source ranking the installed
source is the settling authority for this library claim.

## Source Version
aiohttp 3.13.3 (`__version__ = "3.13.3"` in `aiohttp/__init__.py`)

## Source Fingerprint
sha256:7ce41fc1c2142f5346015455e030e3ebec22a41cde0fc299a66ce13be2cb2a9e  (client.py)
sha256:416b2c15a0fe0da1457303f7ea52d44334669526792b18af24153eca0e42d708  (aiohttp/__init__.py, version provenance)

## Invalidates On
- Any aiohttp release whose `client.py` bytes differ from the fingerprint above — re-read line 299 rather than reusing this verdict.
- `ClientSession.__init__` gaining a non-bool/sentinel default (e.g. `auto_decompress: Union[bool, None] = None`) or resolving the default from an env var, config object, or class attribute instead of the literal `True`.
- An intervening assignment between signature and `self._auto_decompress = ...` (coercion, `not`, or `or False`) at line 399.

## Checked At
2026-10-06

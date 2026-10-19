VERDICT: UNCLEAR
PROMOTION: INELIGIBLE — verdict is not HOLDS/DOES-NOT-HOLD; the blocker is the term "those .body bytes", not the behaviour (P9).

## Claim
On an aiohttp `web.Response` constructed with a `str` passed to `body=`, those `.body` bytes decode back to the string that was passed in

## What it actually does
The source settles the round-trip but not the noun. `str` reaches `StringPayload.__init__` (`payload.py` 430-458), which calls `super().__init__(value.encode(real_encoding), encoding=real_encoding, content_type=content_type, ...)`. Via the `body` setter the registry instantiates `StringPayload(data)` with no `encoding`/`content_type` arguments, so `real_encoding` is `"utf-8"` (line 442) and the stored `_value` is exactly `s.encode("utf-8")`. `BytesPayload.decode(encoding="utf-8", errors="strict")` returns `self._value.decode(encoding, errors)` (line 380-381). So `resp.body.decode()` returns the original string, and `Response.text` (`return self._body.decode(self.charset or "utf-8")`, `web_response.py` 724-730) returns it too, since the setter copies the payload's `Content-Type: text/plain; charset=utf-8` into the response headers.

What is **not** true is that `.body` is bytes: it is the `StringPayload` object itself, so "those `.body` bytes" has no bytes-valued referent, and `resp.body == s.encode("utf-8")` is `False` (`isinstance` of `bytes` fails).

## Why this is UNCLEAR and not a verdict on behaviour
The claim is false or true depending only on which reading of "those `.body` bytes" is taken, so picking would be the whole basis of the verdict (P9):
- **Reading A — `.body` is itself a bytes object to be decoded:** fails at the noun, not the round-trip. `resp.body` is a `Payload`; nothing is decoded because there is no bytes value there. `len(resp.body)` also raises (`Payload` defines no `__len__`); byte count is `resp.body.size`.
- **Reading B — the bytes behind the body (its transmitted content) decode back to the input string:** holds, as shown above.

Which reading settles it: Reading A is answered by the type of `.body` (`web_response.py` 693-705 + `payload.py` 1111); Reading B is answered by `payload.py` 380-381 and 430-458. Both sources are consulted; the claim does not say which it is about. Repair is re-extraction — either "the `.body` attribute is bytes" (a type claim, already answered in claim-418.md) or "the body payload's bytes / `resp.body.decode()` round-trip the input string under utf-8" (a behaviour claim, `HOLDS`).

## Source
`/Users/alexhubitski/Projects/buster-ripper/.venv/lib/python3.13/site-packages/aiohttp/payload.py` — `BytesPayload.decode` lines 380-381 and `StringPayload.__init__` lines 430-458. Corroborated in the same package by `web_response.py` `Response.body` getter/setter lines 693-721 and `Response.text` getter lines 723-730. Verified from production source, not from tests.

## Source Version
aiohttp 3.13.3 (`aiohttp/__init__.py` line 1: `__version__ = "3.13.3"`), installed source. No documentation ships in the wheel.

## Source Fingerprint
sha256:3ba9ec60d50b2fb01e33673227a4d85fbde77159cbe71270b7901e831c0c2aac (payload.py)
sha256:3ca73388d5382e67ed5ea295be844cac06ce7950a5a5237e8b39c7b2259ece65 (web_response.py)

## Invalidates On
- `StringPayload.__init__` changes its default encoding (no longer `utf-8` when neither `encoding` nor `content_type` is supplied), or stops pre-encoding to bytes.
- `BytesPayload.decode` / `Payload.decode` semantics change (default `encoding`, `errors` policy, or a `str`-aware override).
- `Response.body` setter starts storing encoded bytes for `str` instead of a payload, which would make Reading A a behaviour question rather than a noun error.
- Any aiohttp upgrade whose `payload.py`/`web_response.py` bytes differ from the fingerprints above.

## Checked At
2026-10-06

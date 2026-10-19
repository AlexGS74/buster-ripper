VERDICT: HOLDS
PROMOTION: ELIGIBLE — one exact source locator with fingerprint; behaviour is codec-independent per the standard error-handler table

## Claim
`bytes.decode(errors="replace")` substitutes a replacement character for bytes invalid under the target codec instead of raising

## What it actually does
`'replace'` is one of the standard error handlers available to every Python standard encoding codec. On decoding, the handler replaces the malformed input with U+FFFD (REPLACEMENT CHARACTER) and decoding continues, rather than raising — the raising behaviour belongs to `'strict'`. (The same handler uses `?` on *encoding*; U+FFFD is the decoding-side marker.) The documented `bytes.decode` signature lists `'replace'` among accepted `errors` values, and the HOWTO demonstrates it on `bytes.decode`.

## Source
Primary: https://docs.python.org/3/library/codecs.html#error-handlers — error-handler table: "`'replace'` — Replace with a replacement marker. On encoding, use `?` (ASCII character). On decoding, use `\ufffd` (U+FFFD, the official REPLACEMENT CHARACTER). Implemented in `replace_errors()`." Same page: `codecs.replace_errors(exception)` — "Implements the 'replace' error handling. Substitutes `?` (ASCII character) for encoding errors or `\ufffd` (U+FFFD, the official REPLACEMENT CHARACTER) for decoding errors."; and "Decoding and translating works similarly, except `UnicodeDecodeError` … will be passed to the handler and that the replacement from the error handler will be put into the output directly."

Corroborating: https://docs.python.org/3/howto/unicode.html#the-string-type — "`'replace'` (use U+FFFD, REPLACEMENT CHARACTER)" with `>>> b'\x80abc'.decode("utf-8", "replace")` → `'\ufffdabc'`; and https://docs.python.org/3/library/stdtypes.html#bytes.decode lists `'replace'` as a possible `errors` value.

Checked by reading fetched documentation only — the interpreter was not run (P1).

## Source Version
Python 3.14.8 documentation (docs.python.org/3), fetched 2026-10-06.

## Source Fingerprint
sha256:5f29770c85fc38bec0a19414d698d89c34d1ceb6bfa2c0b5421dfcbaf281b98d (codecs.html)
sha256:85626570292e087e7da3e0df0c941abdf85675c14f825df2ee1d5160d0fc8716 (howto/unicode.html)
sha256:bf149b1649601f39dfcbb435a1c1650e2607fe9d5f315fc93ea8432b28ebae6b (stdtypes.html)

## Invalidates On
- The `'replace'` row of `library/codecs.html#error-handlers` changes the decoding marker away from U+FFFD, or the handler is removed from the set available to standard encoding codecs.
- The consulted page bytes change on re-fetch (fingerprint mismatch).
- The claim is applied to a non-standard/registered codec or a custom error handler registered under the name `'replace'` via `codecs.register_error()` — that table governs the standard handlers only.
- Applying the claim to a *bytes*-level `?` expectation: U+FFFD is decoding-side; `?` is encoding-side.

## Not settled by this check
The docs do not state how many U+FFFD characters result from one malformed multi-byte sequence versus one stray byte (the HOWTO example shows a single byte). Byte-for-byte count behaviour is a codec-internal detail and was not established here; anything depending on "exactly one marker per invalid byte" needs its own claim.

## Checked At
2026-10-06

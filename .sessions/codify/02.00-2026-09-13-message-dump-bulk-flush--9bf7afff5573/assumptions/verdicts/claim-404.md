VERDICT: UNCLEAR
PROMOTION: INELIGIBLE — verdict is UNCLEAR; the claim's terms carry two readings that give opposite answers (P9)

## Claim
`bytes.decode(errors="replace")` leaves bytes valid under the target codec unchanged in the decoded result

## What it actually does
The sources settle what the `errors` argument governs, but not which property this sentence is asserting, because "unchanged" and "valid under the target codec" each admit two readings:

**Reading A — error-path isolation (the claim HOLDS).** Only positions that raise a decoding error are substituted; anything the codec decodes validly comes out as it normally would.
- `library/stdtypes.html#bytes.decode`: "`errors` controls how decoding errors are handled" — the argument governs errors, not valid input. Same entry: "For performance reasons, the value of `errors` is not checked for validity unless a decoding error actually occurs."
- `library/codecs.html#error-handlers`: the handler is invoked on the error object and "the replacement from the error handler will be put into the output directly", with processing "continu[ing] on original input at the specified position".
- `howto/unicode.html#the-string-type`: `b'\x80abc'.decode("utf-8", "replace")` → `'\ufffdabc'` — the valid `abc` survives.

**Reading B — literal byte identity (the claim does not hold, and the reason is decoding itself, not `errors`).** `decode()` returns `str`; a valid multi-byte sequence is mapped to a character (`b'\xc3\xa9'` → `'é'`, one char for two bytes). No byte ever appears "unchanged" in a decoded result under *any* `errors` value, so this reading asserts something false of every `decode()` call rather than a property of `'replace'`.

**Crossing axis — what counts as "valid under the target codec".** For a multibyte codec this is not a per-byte property: `b'\xc3'` is a well-formed UTF-8 *lead byte* in isolation yet an incomplete sequence, so it *is* a decoding error and *is* replaced. Read per byte, the claim fails even under Reading A; read as "complete sequences in context", it holds.

Because the verdict flips on which definition of "unchanged" / "valid" is picked, the word is the finding — not the behaviour (P9). This is not P6 softening: the sources do not plainly show `'replace'` doing something *else* to valid input; they show it does nothing to valid input, which is only expressible once the term is pinned.

## Source
https://docs.python.org/3/library/stdtypes.html#bytes.decode ; https://docs.python.org/3/library/codecs.html#error-handlers ; https://docs.python.org/3/howto/unicode.html#the-string-type

Checked by reading fetched documentation only — the interpreter was not run (P1).

## Source Version
Python 3.14.8 documentation (docs.python.org/3), fetched 2026-10-06.

## Source Fingerprint
sha256:bf149b1649601f39dfcbb435a1c1650e2607fe9d5f315fc93ea8432b28ebae6b (stdtypes.html)
sha256:5f29770c85fc38bec0a19414d698d89c34d1ceb6bfa2c0b5421dfcbaf281b98d (codecs.html)
sha256:85626570292e087e7da3e0df0c941abdf85675c14f825df2ee1d5160d0fc8716 (howto/unicode.html)

## Invalidates On
- Re-extraction wording the sources can settle, e.g.: "with `errors='replace'` only byte positions that raise a decoding error are substituted; every complete sequence the codec decodes validly yields the same characters as `errors='strict'`" — that phrasing reads as HOLDS on the above sources.
- A `'replace'` row in `library/codecs.html#error-handlers` that changes the decoding marker, or a documented statement that the handler may rewrite non-error input.
- A codec named by the consuming task that is not in the standard-encodings set (the table above governs standard codecs only).

## What Would Settle It
Naming the property and the unit: "unchanged" = same **characters as `errors='strict'`** (not same bytes), and "valid" = **complete, in-context well-formed sequences** for the target codec. With both pinned, the three pages above answer the claim directly.

## Checked At
2026-10-06

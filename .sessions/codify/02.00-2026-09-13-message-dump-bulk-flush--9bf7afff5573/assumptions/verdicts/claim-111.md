VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
For non-ASCII Python strings, `len(text)` can differ from `len(text.encode("utf-8"))`

## What it actually does
`len(text)` counts string code points; UTF-8 encoding represents a non-ASCII code point with two to four bytes, so, for example, `len("é")` is 1 while `len("é".encode("utf-8"))` is 2. This is a possibility, not an assertion that every string containing non-ASCII must encode successfully (e.g. lone surrogates).

## Source
Python 3.14 Unicode HOWTO, “Encodings,” “The String Type,” and “Converting to Bytes”: https://docs.python.org/3.14/howto/unicode.html#encodings (consulted HTML lines 349–353, 399–403, 474–476).

## Source Version
Python 3.14 documentation (page fetched 2026-09-27).

## Source Fingerprint
sha256:1b59c6964c21a7b3ae7ae3659388f88048623060575847d193d643f6009d5767

## Invalidates On
Change to the consulted Python Unicode HOWTO HTML bytes or to Python's `str`/UTF-8 encoding semantics.

## Checked At
2026-09-27

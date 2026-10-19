VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
With `errors="replace"`, every complete byte sequence that the target codec decodes validly yields the same characters as `errors="strict"` would

## What it actually does
`errors` selects an *error handler*, and in the documented contract a handler is consulted only "in case of an error": `'strict'` raises `UnicodeError`, `'replace'` substitutes U+FFFD for the malformed data. Nothing in either handler touches bytes the codec decodes without error, so for a byte sequence the codec decodes validly both settings run the codec's ordinary decode and produce identical characters. `'replace'` is therefore additive on the failure path only — it never rewrites or re-categorises a valid decode — which is exactly what makes it safe for bulk-flush decoding of already-valid payload bytes.

Reasoning from the source, in order:
- The `errors` argument is defined as *error handling*, never as a decode mode: `Codec.decode()` — "The *errors* argument defines the error handling to apply. It defaults to `'strict'` handling."
- The handler dispatch contract is explicit: `register_error()` — "The *error_handler* argument will be called during encoding and decoding **in case of an error**, when *name* is specified as the errors parameter." For decoding, a `UnicodeDecodeError` is passed to the handler "and the replacement from the error handler will be put into the output directly."
- The two named handlers differ only in what happens at an error site (the Error Handlers table): `'strict'` → "Raise UnicodeError (or a subclass), this is the default"; `'replace'` → "Replace with a replacement marker. … On decoding, use `\ufffd` (U+FFFD, the official REPLACEMENT CHARACTER)". `replace_errors()` restates it as substituting "for decoding errors".
- The claim's "complete byte sequence" qualifier is what removes the one genuine divergence class: a truncated multi-byte tail is *not* validly decoded, so `'strict'` raises there while `'replace'` emits U+FFFD. That is the error path, outside the claim.

Scope notes (checked, not assumed):
- `'surrogatepass'` is codec-specific (utf-8/utf-16*/utf-32* only), which confirms the standard handlers are error-site substitutions layered on the codec's normal decode rather than a different decoder.
- Binary-transform and text-transform codecs (`base64`, `hex`, `rot_13`, …) accept the argument and are documented as not error-sensitive for it; they do not produce characters at all, so the claim's consequent is either trivially satisfied (identical output either way) or not applicable. The claim is stated about text encodings, and holds there.
- `encodings.mbcs` is the one place the docs record `errors` having historically been ignored (`'replace'`/`'ignore'` hard-wired before 3.2) — "Changed in version 3.3: Support any error handler." Windows-only and superseded; it does not falsify the claim for any current codec.
- The verdict rests on the documented contract, not on a custom codec registered via `codecs.register()`. A third-party codec is free to interpret `errors` however it likes; only stdlib codecs are covered.

## Source
https://docs.python.org/3/library/codecs.html#error-handlers — sections "Error Handlers" (handler table, `register_error()`, `replace_errors()`), "Stateless Encoding and Decoding" (`Codec.decode`), "Encodings and Unicode", and `encodings.mbcs` change notes. Full-page markdown consulted as fetched; no local CPython source root was supplied, so this is answered from documentation only (documentation-only claim per protocol).

## Source Version
Python 3.14.8 documentation (docs.python.org/3, live fetch)

## Source Fingerprint
sha256:d1bd6e38bf008aa30dd2fc596e09ae03aefe6ed9b3395fde4b36ce7d33640c3a

## Invalidates On
- A CPython `Doc/library/codecs.rst` edit that makes any standard handler (or `errors=` generally) affect decoding outside an error site — e.g. `'replace'` documented as changing normalisation, byte-order/BOM sniffing, or greedy multi-byte matching for valid input.
- A CPython change so a text codec's non-`strict` path dispatches to a different decoder implementation than its `strict` path for valid input (the fast-path vs. stateful-path split in `Objects/unicodeobject.c`/`Objects/stringlib/codecs.h` is where such a divergence would first appear — verify against installed source if it matters).
- Python < 3.3 for `encodings.mbcs`, where `errors` was ignored and `'replace'`/`'ignore'` were fixed per direction — the documented contract does not hold there.
- Use of a codec registered through `codecs.register()` whose `decode(input, errors)` is not stdlib code — outside the scope this source establishes.
- Re-fetched page whose content hash differs from the fingerprint above (fetched documentation is only reusable byte-identically).

## Checked At
2026-10-06T23:06:00Z

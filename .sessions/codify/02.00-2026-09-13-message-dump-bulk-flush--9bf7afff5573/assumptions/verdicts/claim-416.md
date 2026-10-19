VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`bytes.decode(name)` raises one of `LookupError`, `UnicodeDecodeError` or `json.JSONDecodeError` when `name` is a codec registered in Python's codec registry that is not a text encoding (a bytes-to-bytes transform such as `base64`, `hex` or `rot_13`)

## What it actually does
For a registered non-text codec, `bytes.decode(name)` raises `LookupError` with the message
`'<name>' is not a text encoding; use codecs.decode() to handle arbitrary codecs` — the
first member of the claimed set, so the claim holds. The failure happens at codec *lookup*,
before any decoding runs: `bytes.decode` routes through `PyUnicode_FromEncodedObject`
(`Objects/bytesobject.c:2331`), whose fast paths cover only the utf-8/ascii/latin-1/utf-16/utf-32
aliases, falls through to "Decode via the codec registry" → `_PyCodec_DecodeText`
(`Objects/unicodeobject.c:3615`) → `_PyCodec_TextDecoder` → `codec_getitem_checked(..., "codecs.decode()", 1)`
→ `_PyCodec_LookupTextEncoding` (`Python/codecs.c:522`), which reads the codec's
`_is_text_encoding` attribute and calls `PyErr_Format(PyExc_LookupError, ...)` when it is falsy
(`Python/codecs.c:545-552`). `UnicodeDecodeError` and `json.JSONDecodeError` are not reachable on
this path — they would require a decoder to run, and the registry gate rejects the codec first.

Two boundaries worth recording, neither of which weakens the claim as written:
- The registry classifies a codec by the `_is_text_encoding` flag itself ("Backwards
  compatibility: assume any raw tuple describes a text encoding, and the same for anything lacking
  the private attribute", `Python/codecs.c:533-537`). A custom bytes-to-bytes codec that omits the
  flag is therefore treated *as* a text encoding by the registry; it is not an instance of the
  claim's subject, and its failure surfaces later as `TypeError` ("'x' decoder returned 'bytes'
  instead of 'str'", `Objects/unicodeobject.c:3619-3623`) rather than at lookup.
- Every stdlib bytes-to-bytes codec checked declares the flag, so each takes the `LookupError`
  path: `base64_codec.py:54`, `hex_codec.py:54`, `rot_13.py:45`, `quopri_codec.py:55` (verified by
  fetch: `bz2_codec`, `uu_codec`, `zlib_codec` likewise — exactly the "Binary Transforms" set in
  the codecs docs). `hex` resolves to the `hex_codec` registry entry.

## Source
Primary: `Python/codecs.c` in `_PyCodec_LookupTextEncoding` at CPython tag `v3.11.0` —
https://raw.githubusercontent.com/python/cpython/v3.11.0/Python/codecs.c (lines 522-559)
Corroborating chain, same tag: `Objects/bytesobject.c:2327-2332`, `Objects/unicodeobject.c:3608-3626`,
`Lib/encodings/base64_codec.py` / `hex_codec.py` / `rot_13.py` / `quopri_codec.py`.
Documentation corroborates: `codecs` — "Python Specific Encodings → Binary Transforms",
https://docs.python.org/3.11/library/codecs.html (bytes-to-bytes codecs are not text encodings).

## Source Version
CPython `v3.11.0` (project declares `requires-python = ">=3.11"` in `pyproject.toml`).
The same lookup gate is present unchanged at tags `v3.13.0` and `v3.14.0`.

## Source Fingerprint
sha256:6cb5b75e0a2999d910e90d2ee92df8c47868dba11dc18673932689d07a3816f1 (Python/codecs.c @ v3.11.0)
sha256:3cdf1a8a5cca8b7128ff19539073184e49af304208c6a44b7fe43b9ff7ba19e5 (Objects/bytesobject.c @ v3.11.0)
sha256:7f3b3b3de45ad1bfb17bb6fffb3e39f65308ae5baf648d5aaa9db8f94a87cc2a (Objects/unicodeobject.c @ v3.11.0)
sha256:cf9ac7a464f541492486241d1b4bf33e37b45c6499275cc4d69c5a8e564e5976 (Lib/encodings/base64_codec.py @ v3.11.0)
sha256:fc5f0a31b59efe990b86efb98936769f33dd91d912ce55b49a5a4cfc516cd047 (Lib/encodings/hex_codec.py @ v3.11.0)
sha256:14767f475acdc0bf48e6272280dd15b80efaecafb93c06be21136f83dd1ee7e4 (Lib/encodings/rot_13.py @ v3.11.0)

## Invalidates On
- `_PyCodec_LookupTextEncoding` in `Python/codecs.c` changes the falsy-`_is_text_encoding`
  branch to raise something other than `LookupError`, or the branch is removed
- Any of the named stdlib codecs (`base64_codec`, `hex_codec`, `rot_13`) drops or flips its
  `_is_text_encoding=False` registry entry
- `bytes.decode` stops routing through `PyUnicode_FromEncodedObject` → `_PyCodec_DecodeText`,
  or a new fast path begins accepting non-text codec names before the registry lookup
- The interpreter is not CPython (e.g. PyPy), or a CPython version whose `codecs.c` bytes differ
  from the fingerprint above

## Checked At
2026-10-06

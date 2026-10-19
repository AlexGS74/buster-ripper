VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`hashlib.sha256`'s hexdigest is 64 lowercase hexadecimal characters

## What it actually does
`hexdigest()` returns a `str` (not `bytes`) whose length is exactly `2 * digest_size` and
whose every character comes from the 16-symbol table `Py_hexdigits = "0123456789abcdef"`,
so the alphabet is `0-9` plus *lowercase* `a-f` only, with no separators or prefix. SHA-256
has a fixed 32-byte digest, hence 64 characters.

Mechanics (both `sha256` backends funnel into the same encoder):
- `_hashlib` / OpenSSL path: `Modules/_hashopenssl.c` `EVP_hexdigest_impl` (626-660) →
  `_Py_strhex((const char *)digest, (Py_ssize_t)digest_size)`.
- builtin `_sha2` path: `Modules/sha2module.c` `SHA256Type_hexdigest_impl` (369-377) →
  `_Py_strhex((const char *)digest, self->digestsize)`, with `#define SHA256_DIGESTSIZE 32`
  at line 42 and `new->digestsize = 32;` in the constructor at line 631.
- `Python/pystrhex.c` `_Py_strhex_impl`: `resultlen += arglen * 2` (line 55);
  `return_bytes == 0` for `_Py_strhex`, so `PyUnicode_New(resultlen, 127)` — an ASCII-only
  `str`; with `bytes_per_sep_group == 0` the loop at lines 84-90 writes exactly
  `Py_hexdigits[c >> 4]`, `Py_hexdigits[c & 0x0f]` per digest byte. No uppercase table is
  reachable from this path.
- Length 32 for the SHA-256 digest is fixed: `sha256()` takes no `digest_size` argument
  (only `data` and `usedforsecurity`), unlike `blake2*`/`shake_*`; the OpenSSL path's
  `EVP_MD_CTX_size()` returns the standard SHA-256 size, and the docs state hashlib's
  SHA256 is the algorithm "defined in the FIPS 180-4 standard" (256-bit digest).

Docs state the shape without naming case ("Like `digest()` except the digest is returned as
a string object of double length, containing only hexadecimal digits") and the 3.14.8 docs
example prints exactly 64 lowercase characters for `sha256`:
`'031edd7d41651593c5fe5c006fa5752b37fddff7bc4e843aa6af0c950f4b9406'`.

## Source
Primary: `Python/pystrhex.c` lines 55, 66-90, 145-148 (Python 3.14.8), plus
`Python/codecs.c:28` — `const char *Py_hexdigits = "0123456789abcdef";` (declared
`Include/codecs.h:170`). Length for the builtin path: `Modules/sha2module.c:42,631,369-377`.
Backend call site: `Modules/_hashopenssl.c:626-660`.
Docs corroboration: `https://docs.python.org/3/library/hashlib.html` (Python 3.14.8),
`hash.hexdigest()` and the Usage example.
Note: no `Source root:` / `Dependency source roots:` locator was supplied, so the installed
copy was not read; files were taken from the official `Python-3.14.8.tgz` (tarball sha256
`a65b20a728f169f4e66ae143f40b1bd3d33c38d770251663f627c9767b79b210`) and match the
`v3.14.8` raw files byte-for-byte. Repo declares `requires-python = ">=3.11"`
(`pyproject.toml:4`).

## Source Version
CPython 3.14.8 (tag `v3.14.8`). Installed interpreter version unverified (no source
locator supplied; running Python is out of scope for this role).

## Source Fingerprint
sha256:7842f6d7d9b58bcbf10cbb24d29f6d1f36668e20f935689e197bbe0abdccd4ed (Python/pystrhex.c)
sha256:bfa6673db880ba38762203dd913ab353e72538ae288bed6109df82669270484e (Python/codecs.c)
sha256:f59ab2485dabb24200909fbc62e4d8ed7e2c8044a52b2233fbaf1fb6c0d0b2c1 (Modules/sha2module.c)
sha256:70727f1311b9a2021ab4327e8d74fcbaf6d3d916b1b32fb1c76e2446c5893a77 (Modules/_hashopenssl.c)
sha256:016fd07f8fd5ce1ecd2150a8f43f44ee5a65f033dbc82bda2d6f722921f1e44e (docs.python.org/3/library/hashlib.html, 3.14.8)

## Invalidates On
- A CPython release changing `Py_hexdigits` to an uppercase table, or switching
  `hexdigest()` to `_Py_strhex_bytes`/`_Py_strhex_with_sep` (separator-bearing) variants —
  length would then exceed 64.
- A change in `Python/pystrhex.c`'s per-byte mapping or in the `resultlen` formula.
- `sha256()` gaining a variable `digest_size` argument, or an OpenSSL provider reporting a
  size other than 32 bytes for name `sha256`.
- A non-CPython implementation whose `hexdigest()` emits uppercase or a `0x`-prefixed form.

## Checked At
2026-10-05

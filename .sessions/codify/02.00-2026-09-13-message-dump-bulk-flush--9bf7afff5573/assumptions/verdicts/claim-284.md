VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
A `hashlib.sha256` hexdigest is identical across processes for identical input bytes

## What it actually does
The SHA-256 digest is a pure function of the bytes fed in, so two interpreter processes
that hash the same bytes get the same `hexdigest()` string. Nothing per-process enters the
computation: both `sha256` implementations derive the digest only from the hash object's
accumulated state, and the hex encoder is a pure table lookup. The one place CPython does
inject per-process randomness — `Py_HashSecret`, fed by `PYTHONHASHSEED` /
`bootstrap_hash` — is used by the built-in `hash()` path (`Python/pyhash.c`,
`Python/initconfig.c`, `Python/bootstrap_hash.c`) and does **not** appear anywhere in the
hashlib SHA-2 path.

Also checked, since either could have broken cross-process equality:
- Backend choice does not change the value. `usedforsecurity` selects
  `Py_ht_evp` vs `Py_ht_evp_nosecurity` (`_hashopenssl.c`, `_hashlib_HASH`), i.e. the
  OpenSSL provider table for availability in FIPS-restricted builds — not a different
  digest. The docs' 3.12 change note describes the non-OpenSSL fallback as "a verified
  implementation" of the same algorithms, and `Lib/hashlib.py` routes `sha256` to either
  `_hashlib` or builtin `_sha2`; both implement FIPS 180-4 SHA-256.
- `hashlib`'s "Randomized hashing" docs section is BLAKE2-specific (the `salt` parameter);
  `sha256()` has no randomization knob in its signature (`sha256([data, ]*,
  usedforsecurity=True)`).

## Source
- `Modules/_hashopenssl.c` lines 626-660 (`EVP_hexdigest_impl`: copies `self->ctx`,
  `EVP_DigestFinal`, returns `_Py_strhex((const char *)digest, digest_size)`) — no seed
  argument in any step.
- `Modules/sha2module.c` lines 363-377 (`SHA256Type_hexdigest_impl` →
  `_Py_strhex((const char *)digest, self->digestsize)`).
- `Python/pystrhex.c` lines 8-130, 145-148 (`_Py_strhex_impl` / `_Py_strhex`) — output is
  `Py_hexdigits[c >> 4]`, `Py_hexdigits[c & 0x0f]` per byte, nothing else.
- Negative check (method stated): case-insensitive grep for
  `hash_secret|Py_HashSecret|urandom|PYTHONHASHSEED|getpid|srandom|seed` over
  `Modules/sha2module.c`, `Modules/_hashopenssl.c`, `Python/pystrhex.c`,
  `Modules/hashlib.h`, `Lib/hashlib.py` returns no matches; the same symbol search over
  `Python/` finds `Py_HashSecret` only in `pyhash.c`, `initconfig.c`, `bootstrap_hash.c`.
- Docs: `https://docs.python.org/3/library/hashlib.html` (Python 3.14.8) — "the FIPS secure
  hash algorithms SHA224, SHA256, SHA384, SHA512 (defined in the FIPS 180-4 standard)" and
  the published worked value
  `hashlib.sha256(b"Nobody inspects the spammish repetition").hexdigest()` →
  `'031edd7d41651593c5fe5c006fa5752b37fddff7bc4e843aa6af0c950f4b9406'`.
Note: no `Source root:` / `Dependency source roots:` was supplied, so the installed copy
was not read; verification is against upstream CPython source at tag `v3.14.8`
(byte-identical to `Python-3.14.8.tgz`, tarball sha256
`a65b20a728f169f4e66ae143f40b1bd3d33c38d770251663f627c9767b79b210`). Repo declares
`requires-python = ">=3.11"` (`pyproject.toml:4`).

## Source Version
CPython 3.14.8 (tag `v3.14.8`). Installed interpreter version unverified (no source
locator supplied; running Python is out of scope for this role).

## Source Fingerprint
sha256:70727f1311b9a2021ab4327e8d74fcbaf6d3d916b1b32fb1c76e2446c5893a77 (Modules/_hashopenssl.c)
sha256:f59ab2485dabb24200909fbc62e4d8ed7e2c8044a52b2233fbaf1fb6c0d0b2c1 (Modules/sha2module.c)
sha256:7842f6d7d9b58bcbf10cbb24d29f6d1f36668e20f935689e197bbe0abdccd4ed (Python/pystrhex.c)
sha256:88fa624821fbdee7eb2ef87c2437b207afe328cf117c7463367e2db5eb1af492 (Lib/hashlib.py)
sha256:016fd07f8fd5ce1ecd2150a8f43f44ee5a65f033dbc82bda2d6f722921f1e44e (docs.python.org/3/library/hashlib.html, 3.14.8)

## Invalidates On
- A CPython release that mixes per-process state into hashlib's SHA-2 path
  (introduction of a keyed/salted `sha256`, or use of `Py_HashSecret` there).
- A hashlib `sha256` backed by a provider or fallback that is not FIPS 180-4 conformant
  (e.g. a substituted OpenSSL provider returning a different digest for name `sha256`).
- A change to `Python/pystrhex.c`'s mapping (`Py_hexdigits[c >> 4]`, `Py_hexdigits[c & 0xf]`)
  or to `EVP_hexdigest_impl` / `SHA256Type_hexdigest_impl` digest-to-string conversion.
- Cross-*build* comparison when a downstream distribution patches hashlib's SHA-2 backend
  with a non-conformant implementation.

## Checked At
2026-10-05

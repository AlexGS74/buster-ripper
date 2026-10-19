VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`hashlib.sha256` raises `TypeError` when passed a `str` rather than a bytes-like object

## What it actually does
Passing a `str` — to the `hashlib.sha256()` constructor or to `.update()` — raises
`TypeError` with the message `"Strings must be encoded before hashing"`. The check is
explicit, not incidental: CPython tests `PyUnicode_Check(obj)` before touching the
buffer API, so a `str` fails with `TypeError` even though `PyObject_GetBuffer` would
also reject it. Non-buffer, non-str objects (e.g. `int`) raise `TypeError` too, with
`"object supporting the buffer API required"`.

Both `sha256` implementations behave the same way, and `Lib/hashlib.py` binds the public
`sha256` name to exactly one of them depending on build/OpenSSL availability:

- `_hashlib` (OpenSSL / HACL* backed): `Modules/_hashopenssl.c:956` in
  `_hashlib_HASH()` (constructor `data` argument) and `:680` in `EVP_update_impl()`
  call `GET_BUFFER_VIEW_OR_ERROUT(obj, &view)`.
- `_sha2` builtin fallback: `Modules/sha2module.c:617` in `_sha2_sha256_impl()`
  (constructor) and `:413` / `:447` in the update path call the same macro.
- The macro is defined once in `Modules/hashlib.h:10-18` (`GET_BUFFER_VIEW_OR_ERROR`,
  aliased by `GET_BUFFER_VIEW_OR_ERROUT` at `:32`) and contains the `PyExc_TypeError`
  / "Strings must be encoded before hashing" branch.

The 3.14 docs support the same expectation without naming the exception type: "You can
now feed this object with bytes-like objects (normally bytes) using the update method"
and `hash.update(data)` — "Update the hash object with the bytes-like object". No
documented path accepts `str`.

## Source
Primary: `Modules/hashlib.h` (lines 10-18, 32-33), Python 3.14.8 —
`https://raw.githubusercontent.com/python/cpython/v3.14.8/Modules/hashlib.h`, byte-identical
to `Python-3.14.8.tgz` (tarball sha256 `a65b20a728f169f4e66ae143f40b1bd3d33c38d770251663f627c9767b79b210`).
Call sites: `Modules/_hashopenssl.c:680,956`; `Modules/sha2module.c:413,447,617`;
binder: `Lib/hashlib.py` (`__get_openssl_constructor` / `__get_builtin_constructor`).
Docs corroboration only: `https://docs.python.org/3/library/hashlib.html` (Python 3.14.8).
Note: no `Source root:` / `Dependency source roots:` locator was supplied, so the
*installed* copy was not read; verification is against upstream source at the tag that
matches the currently published stable docs. The repo's `pyproject.toml:4` declares
`requires-python = ">=3.11"`; the same message and `PyExc_TypeError` branch were read at
branches `3.11`, `3.12`, `3.13` and `3.14` of `Modules/hashlib.h`, so the verdict covers
the declared range.

## Source Version
CPython 3.14.8 (tag `v3.14.8`); same check present on branches 3.11, 3.12, 3.13.
Installed interpreter version unverified (no source locator supplied; executing Python is
out of scope for this role).

## Source Fingerprint
sha256:2a1c6c324e7e1959bc819fbf20f2727b4cb5d7f9f35057940fc03dbffc46db1a (Modules/hashlib.h)
sha256:70727f1311b9a2021ab4327e8d74fcbaf6d3d916b1b32fb1c76e2446c5893a77 (Modules/_hashopenssl.c)
sha256:f59ab2485dabb24200909fbc62e4d8ed7e2c8044a52b2233fbaf1fb6c0d0b2c1 (Modules/sha2module.c)
sha256:88fa624821fbdee7eb2ef87c2437b207afe328cf117c7463367e2db5eb1af492 (Lib/hashlib.py)
sha256:016fd07f8fd5ce1ecd2150a8f43f44ee5a65f033dbc82bda2d6f722921f1e44e (docs.python.org/3/library/hashlib.html, 3.14.8)

## Invalidates On
- A CPython release whose `Modules/hashlib.h` `GET_BUFFER_VIEW_OR_ERROR` drops or reorders
  the `PyUnicode_Check` / `PyExc_TypeError` branch (e.g. accepting `str` with implicit
  UTF-8 encoding).
- A change moving `sha256` to an implementation that does not use that macro
  (`_hashopenssl.c` / `sha2module.c` call sites).
- A non-CPython implementation (PyPy, etc.) or a vendored/patched interpreter where
  hashlib's data argument is converted differently.
- `Lib/hashlib.py` gaining a str-coercing wrapper around the constructors.

## Checked At
2026-10-05

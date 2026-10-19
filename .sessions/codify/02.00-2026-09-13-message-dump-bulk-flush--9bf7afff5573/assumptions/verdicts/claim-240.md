VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`json.loads` returns a Python `str` for a JSON string value nested in an object

## What it actually does
Yes — decoding is recursive and the string translation applies at every depth, not just at
the document root. `json.loads` with no hook/parse kwargs calls `_default_decoder.decode(s)`,
whose scanner resolves a `"` at any position to `parse_string` (`scanstring`), which returns
`''.join(chunks)` — a plain Python `str`. `JSONObject` takes that value from
`scan_once(s, end)` and stores it in `dict(pairs)` unchanged, so `{"a": "b"}` decodes to
`{"a": "b"}` with `type(...["a"]) is str`. The documented default translation table states
`object -> dict` and `string -> str` with no root/nested distinction. Caveat that does not
contradict the claim: `object_hook`/`object_pairs_hook` can replace the enclosing `dict`,
and neither can change a string *value*'s type (only `parse_int`/`parse_float`/
`parse_constant` substitute types, for numbers).

Evidence located in CPython 3.11.9 sources:
- `Doc/library/json.rst` line 321, inside `JSONDecoder` (`.. _json-to-py-table:`):
  `| string        | str               |` (and `| object | dict |` at line 317).
- `Lib/json/__init__.py` lines 343-346: default path is `return _default_decoder.decode(s)`.
- `Lib/json/decoder.py` line 126: `return ''.join(chunks), end` (py_scanstring's value), and
  line 130: `scanstring = c_scanstring or py_scanstring`.
- `Lib/json/decoder.py` line 186: `value, end = scan_once(s, end)` inside `JSONObject`'s
  key/value loop, appended as `pairs_append((key, value))` then `pairs = dict(pairs)` (212).
- `Lib/json/decoder.py` line 327: `self.parse_string = scanstring` — the string handler the
  scanner dispatches to at any nesting depth.

## Source
Primary locator: https://raw.githubusercontent.com/python/cpython/v3.11.9/Doc/library/json.rst
(rendered: https://docs.python.org/3.11/library/json.html#json.JSONDecoder)
Consulted together for the implementation path:
https://raw.githubusercontent.com/python/cpython/v3.11.9/Lib/json/decoder.py and
https://raw.githubusercontent.com/python/cpython/v3.11.9/Lib/json/__init__.py
(all fetched complete, not truncated).

## Source Version
CPython tag v3.11.9 documentation and stdlib source (`Lib/json`); project requires
`>=3.11` (`pyproject.toml:4`).

## Source Fingerprint
sha256:b154dd5e6e61514a209bb91d9bc51909574fc8f82e5f35173d78a560817a5610 (Doc/library/json.rst)
sha256:9f02654649816145bc76f8c210a5fe3ba1de142d4d97a1c93105732e747c285b (Lib/json/decoder.py)
sha256:d5d41e2c29049515d295d81a6d40b4890fbec8d8482cfb401630f8ef2f77e4d5 (Lib/json/__init__.py)

## Invalidates On
- Any change to the `json-to-py-table` in `Doc/library/json.rst` (e.g. strings decoding to a
  subclass of `str` or a custom string type) — re-hash the file at the deployed tag.
- Any change to `Lib/json/decoder.py` `py_scanstring` return, `scanstring = c_scanstring or
  py_scanstring`, or `JSONObject`'s `value, end = scan_once(...)` handling.
- Call sites that pass `object_hook`/`object_pairs_hook` or a `cls=` custom decoder: this
  verdict covers the default decode path only, so do not reuse it for hook-supplied returns.
- Deploying a Python line other than the one fingerprinted here (e.g. a future version with a
  rewritten pure-Python scanner).

## Checked At
2026-10-05T14:28:40-04:00

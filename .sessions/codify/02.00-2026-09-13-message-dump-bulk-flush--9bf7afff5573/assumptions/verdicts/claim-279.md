VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`json.dumps` with `sort_keys` enabled yields identical text for two equal mappings built in different insertion orders

## What it actually does
With `sort_keys=True` the docs state "the output of dictionaries will be sorted by key", and the encoder sorts the mapping's items (`sorted(dct.items())` in the pure-Python path) — ordering is a function of the key set alone, never of insertion order. Two equal mappings have the same key set, so the emitted sequence and text are identical. The docs name this exact purpose: the option "is useful for regression tests to ensure that JSON serializations can be compared on a day-to-day basis."

Caveat that does not change the verdict: the docs also note keys "are converted to strings" after sorting. For two equal mappings the same key objects map to the same stringified keys, so output stays identical; only a degenerate mapping whose distinct keys stringify identically (e.g. `1` and `1.0` collapsed by dict equality — such keys are not distinct in a dict) could differ, which equality makes unreachable.

## Reasoning
Grounded in the library's own documentation (API text for `json.dumps`/`JSONEncoder.sort_keys`), corroborated by the upstream pure-Python encoder at 3.11 (`encoder.py` line 353: `if _sort_keys: items = sorted(dct.items())`). The docs' unqualified "sorted by key" plus key-set-invariance of equal mappings settles insertion-order independence. No language-level "repr sorts containers" guarantee was found or needed (searched 3.11 and stable datamodel pages — the phrase does not exist there); this verdict does not rest on it.

## Source
https://docs.python.org/3.11/library/json.html (sections: `json.dumps` parameter `sort_keys`; `json.JSONEncoder` — consulted bytes also fingerprinted from https://raw.githubusercontent.com/python/cpython/3.11/Lib/json/encoder.py)

## Source Version
Python 3.11 documentation (project pins `requires-python = ">=3.11"`); upstream `Lib/json/encoder.py` at tag 3.11; stable docs (3.14.8) show identical wording.

## Source Fingerprint
sha256:74339a40369c00e2658c483bd28d7ce67b8228806fafeb7b81ab936819f3ee2a (json-3.11.html)
sha256:af7bd40a0d0d0a3e726a9b4b3a2a543019f6ab97a340d0162a9c29ca9da97869 (encoder-3.11.py)

## Invalidates On
- Upstream changes the `sort_keys` contract in `Doc/library/json.rst` (e.g. qualifies sorting as top-level-only or order-preserving for equal mappings)
- `Lib/json/encoder.py` (`_make_iterencode`) or `_json.c` stops sorting by key for every dict output at 3.11+
- The running interpreter is a non-CPython implementation whose `json` sorts differently than documented

## Checked At
2026-10-05

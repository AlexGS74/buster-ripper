VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`json.dumps`'s `sort_keys` option sorts nested mappings at every depth

## What it actually does
`sort_keys` is a single encoder-wide setting applied to every dictionary the encoder serializes, at any nesting depth. The docs state unconditionally that if `sort_keys` is true "then the output of dictionaries will be sorted by key" — no depth qualification — and the encoder implements it recursively: the per-nesting `_iterencode` closure captures `_sort_keys` once and each dict-level branch applies `if _sort_keys: items = sorted(dct.items())` before recursing into values.

## Reasoning
Checked independently of the other claims in this bundle. Primary grounding: upstream `Lib/json/encoder.py` at 3.11 — `_make_iterencode(..., _sort_keys, ...)` binds the flag in the closure; the dict branch does `items = sorted(dct.items())` under `if _sort_keys:`; nested containers re-enter the same `_iterencode(o, current_indent, markers)` with the captured flag, so every depth is sorted. The C fast path receives the same single flag (`c_make_encoder(..., self.sort_keys, ...)`), so it is not depth-limited either. The documentation's unqualified plural "dictionaries" corroborates at every depth. No source says or implies top-level-only sorting.

## Source
https://raw.githubusercontent.com/python/cpython/3.11/Lib/json/encoder.py (`_make_iterencode` dict branch; `iterencode` wiring to `c_make_encoder`) — corroborated by https://docs.python.org/3.11/library/json.html (`JSONEncoder.sort_keys`)

## Source Version
Python 3.11 (`Lib/json/encoder.py` at tag/branch `3.11`; project pins `requires-python = ">=3.11"`); stable docs (3.14.8) show identical wording.

## Source Fingerprint
sha256:af7bd40a0d0d0a3e726a9b4b3a2a543019f6ab97a340d0162a9c29ca9da97869 (encoder-3.11.py)
sha256:74339a40369c00e2658c483bd28d7ce67b8228806fafeb7b81ab936819f3ee2a (json-3.11.html)

## Invalidates On
- `Lib/json/encoder.py` changes so the sort is applied only at the top level (e.g. flag not propagated through the recursion) or `_json.c` diverges from the documented flag
- `Doc/library/json.rst` qualifies `sort_keys` to a specific depth
- Running interpreter ≥3.12 whose stdlib `json` wording/implementation changed relative to the 3.11 sources consulted

## Checked At
2026-10-05

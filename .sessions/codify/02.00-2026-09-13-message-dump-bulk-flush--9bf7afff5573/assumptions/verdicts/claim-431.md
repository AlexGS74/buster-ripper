VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`difflib.unified_diff` compares input lines for exact string equality, without whitespace or case normalization

## What it actually does
`unified_diff` builds `SequenceMatcher(None, a, b)` — `isjunk=None`, so no lines are filtered as junk — and `SequenceMatcher` groups elements through a dict (`b2j.setdefault(elt, [])`) whose keys use the elements' own hash/equality; for `str` lines that is exact string equality. No whitespace stripping, case folding, or other normalization is applied anywhere in `unified_diff`; the input strings are emitted verbatim with only a one-character prefix. (Matching is longest-contiguous-block based and the default `autojunk` heuristic can leave some identical lines unaligned in sequences of 200+ items, but the equality predicate itself remains exact.)

## Source
https://raw.githubusercontent.com/python/cpython/3.11/Lib/difflib.py — `unified_diff` builds `SequenceMatcher(None, a, b)` at line 1138; element grouping in `SequenceMatcher.__chain_b` at lines 266-302; verbatim content emission at lines 1152-1161. Corroborated by https://docs.python.org/3/library/difflib.html#difflib.SequenceMatcher (elements compared as hashable sequence elements; no normalization described).

## Source Version
CPython 3.11 branch `Lib/difflib.py`; project declares `requires-python = ">=3.11"`. The same `SequenceMatcher(None, a, b)` construction and verbatim content emission are present in the CPython 3.15 branch (lines 1153 and 1166-1176), so the behavior is unchanged across the supported range checked.

## Source Fingerprint
sha256:0c6afc23568d55b3e9ac914f9c5361e3033e778aa5b58d3cc82835fc5c638679 (CPython 3.11 `Lib/difflib.py`); cross-checked 3.15 `Lib/difflib.py` sha256:aaa3d1ccdb8d8f0d2e8c99fe1a320a983b770fad94779f1dd417cdd37c033159

## Invalidates On
- `unified_diff` passes an `isjunk` callable to `SequenceMatcher` that ignores or normalizes lines
- `SequenceMatcher` starts normalizing elements (whitespace stripping, case folding) before equality comparison
- `unified_diff` transforms lines (e.g. `strip()`, `lower()`) before matching or emitting them
- Project minimum Python moves below 3.11 or to a version whose `difflib.unified_diff` differs on this point

## Checked At
2026-10-17

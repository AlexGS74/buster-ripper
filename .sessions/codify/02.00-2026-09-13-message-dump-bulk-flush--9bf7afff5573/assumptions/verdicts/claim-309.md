VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
Assigning to a key a `dict` already holds does not increase the dict's size

## What it actually does
For a built-in `dict`, `d[key] = value` sets that key's entry: when the key is
already present the existing entry's value is **replaced** and no new item is
added, so `len(d)` — "the number of items in the dictionary" — is unchanged.
The documentation's own worked example shows a 4-key dict keeping exactly four
keys after `d["one"] = 42`, and states that "updating a key does not affect the
order" (i.e. the update is an in-place value replacement, not a re-insertion).
This also covers the equal-but-not-identical case: "Values that compare equal
(such as 1, 1.0, and True) can be used interchangeably to index the same
dictionary entry", so assigning through a fresh equal key still touches the one
existing entry rather than adding an item.

## Source
https://docs.python.org/3/library/stdtypes.html#mapping-types-dict — section
"Mapping Types — dict" (fetched plain-text rendering, from the `Mapping Types —
dict¶` heading through the `d[key] = value` operation entry). Statements
consulted:
- `len(d)` — "Return the number of items in the dictionary d."
- `d[key] = value` — "Set d[key] to value." (no key-creation clause; contrast
  the constructor text, which describes *adding* keys)
- "Dictionaries preserve insertion order. Note that updating a key does not
  affect the order." plus the example `{"one": 1, "two": 2, "three": 3,
  "four": 4}` → `d["one"] = 42` → `{'one': 42, 'two': 2, 'three': 3, 'four': 4}`
- "Values that compare equal (such as 1, 1.0, and True) can be used
  interchangeably to index the same dictionary entry."

Note on scope: this is a documentation-only claim about built-in `dict`
semantics; no local source root was supplied and none was needed. Reading of
the word "size" matters for reuse: the docs settle it for *item count*
(`len(d)`), which is the reading the claim states. The docs section consulted
does not speak to the memory footprint of the dict object
(`sys.getsizeof(d)`/internal table growth), so that alternative reading is
outside this evidence rather than contradicted by it.

## Source Version
Python 3.14.8 documentation (docs.python.org/3, current release docs), fetched
2026-10-05.

## Source Fingerprint
c36991965f1734469aa7b60fcafb05a1c8648e562efb1d62a83259f6c6e7fa78
(sha256 of the exact consulted bytes: the "Mapping Types — dict" section slice,
lines 5488–5620 of the fetched text rendering of
https://docs.python.org/3/library/stdtypes.html)
Full fetched page for reference: 09d91e119a73d325c964cd4ee53390362573dc42536657314cbb082df492ceb6

## Invalidates On
- The "Mapping Types — dict" section of stdtypes.html changes the wording of
  `d[key] = value` or `len(d)` (e.g. documents subscript assignment as adding an
  entry for an existing key) — recheck against the new section bytes.
- The Python version whose docs are being relied on changes such that dict item
  counting or equal-key entry sharing is documented differently (e.g. a release
  note altering "values that compare equal index the same entry").
- The claim is re-extracted to mean the dict object's memory footprint / hash
  table size rather than its item count — this evidence does not cover that
  reading and must be rechecked against CPython dict internals.
- The subject of the claim turns out to be a `dict` subclass or other mapping
  with a custom `__setitem__`/`__missing__` — those are user-defined and outside
  this documentation.

## Checked At
2026-10-05

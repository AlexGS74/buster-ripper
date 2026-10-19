VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`json.dumps` raises `TypeError` for a value it cannot serialize

## What it actually does
When a value has no JSON representation — not via the built-in conversion table, a `cls`/`JSONEncoder` subclass, or the `default` hook — the base `JSONEncoder.default()` raises `TypeError`. The docs state it directly: `default` "should return a JSON encodable version of the object or raise a `TypeError`. If not specified, `TypeError` is raised," and subclasses must "call the superclass implementation (to raise `TypeError`)."

## Reasoning
Checked independently of the other claims. Grounded in the `json.dumps` parameter list and the `JSONEncoder` class text of the module's own documentation; the docs use `TypeError` (not `ValueError`/`JSONEncodeError`) as the unserializable-value failure mode. Related distinctions verified so the claim isn't misread: `ValueError` is reserved for out-of-range floats when `allow_nan=False`; unencodable dict *keys* raise `TypeError` only when `skipkeys=False`; `RecursionError` covers circular structures when `check_circular=False`; and `JSONDecodeError` is decode-side only. "Cannot serialize" is read as the docs' own "objects that can't otherwise be serialized" — i.e. after any supplied `default`/subclass handler has also declined — under which reading the failure is unconditional.

## Source
https://docs.python.org/3.11/library/json.html (`json.dumps` `default` parameter; `json.JSONEncoder` `default()`; `skipkeys`/`allow_nan` caveats)

## Source Version
Python 3.11 documentation (project pins `requires-python = ">=3.11"`); stable docs (3.14.8) show identical wording.

## Source Fingerprint
sha256:74339a40369c00e2658c483bd28d7ce67b8228806fafeb7b81ab936819f3ee2a (json-3.11.html)

## Invalidates On
- `Doc/library/json.rst` changes the exception raised for unserializable objects (e.g. to a `json`-specific subclass)
- `Lib/json/encoder.py` or `_json.c` raises something other than `TypeError` from the base `default()` path
- Running interpreter ≥3.12 whose documented unserializable-value failure mode diverges from the 3.11 page consulted

## Checked At
2026-10-05

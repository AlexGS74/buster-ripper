VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`json.dumps` with a two-item `separators` tuple emits no whitespace between items or after the key separator

## What it actually does
The tuple's first element is inserted between items and the second between each key and its value — nothing is added around them. The docs state the default is `(', ', ': ')` (when `indent` is `None`) and that "To get the most compact JSON representation, you should specify `(',', ':')` to eliminate whitespace," with the compact-encoding example `json.dumps([1, 2, 3, {'4': 5, '6': 7}], separators=(',', ':'))` producing `'[1,2,3,{"4":5,"6":7}]'` — no space after any comma or colon.

## Reasoning
Checked independently of the other claims. As written, the claim names the two separator slots exactly as the docs define them — `(item_separator, key_separator)` — and asserts the whitespace-elimination property the docs state for the `(',', ':')` form: the separators are verbatim, unpadded strings. Caveat that does not soften the verdict: the claim constrains the tuple to be two-item, not to its contents, so a whitespace-containing pair such as `(', ', ': ')` would of course emit whitespace; the "no whitespace" property belongs to non-whitespace separator strings, which is the reading the docs' own `(',', ':')` instruction fixes.

## Source
https://docs.python.org/3.11/library/json.html (`json.dumps`/`json.dump` `separators` parameter; `JSONEncoder.separators`; "Compact encoding" example)

## Source Version
Python 3.11 documentation (project pins `requires-python = ">=3.11"`); stable docs (3.14.8) show identical wording; no "Changed in version" alters separator semantics after 3.4's default change.

## Source Fingerprint
sha256:74339a40369c00e2658c483bd28d7ce67b8228806fafeb7b81ab936819f3ee2a (json-3.11.html)

## Invalidates On
- `Doc/library/json.rst` changes the `separators` contract (slot order, padding, or the "eliminate whitespace" statement for `(',', ':')`)
- `Lib/json/encoder.py` or `_json.c` starts inserting additional whitespace around the given separators
- Running interpreter ≥3.12 whose documented separator behavior diverges from the 3.11 page consulted

## Checked At
2026-10-05

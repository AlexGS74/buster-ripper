VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
Python's `str()` falls back to calling `__repr__` when a type defines no `__str__`

## What it actually does
`str(object)` calls `type(object).__str__(object)`; when the class defines no
`__str__`, the inherited default implementation from the built-in `object` type
calls `object.__repr__()`, which dispatches to the class's own `__repr__`. The
documentation states this as a fallback directly: "If object does not have a
`__str__()` method, then `str()` falls back to returning `repr(object)`." The
mechanism is therefore documented both from the `str()` side (stdtypes) and from
the data-model side (`object.__str__` → "The default implementation defined by
the built-in type object calls object.`__repr__()`"; `object.__repr__` → "If a
class defines `__repr__()` but not `__str__()`, then `__repr__()` is also used
when an 'informal' string representation of instances of that class is
required").

## Source
https://docs.python.org/3/library/stdtypes.html#str (bytes saved locally as
`stdtypes.html`; resolved title "Built-in Types — Python 3.14.8 documentation")
— establishing source for the fallback wording.

Corroborated by https://docs.python.org/3/reference/datamodel.html#object.__str__
and ...#object.__repr__ (saved as `datamodel.html`, same 3.14.8 build) for the
`object.__str__` → `object.__repr__` delegation.

Documentation-only claim about CPython's language/stdlib behaviour; no repo
source and no execution were consulted (P1). No test artifact used (P10).

## Source Version
Python 3.14.8 documentation (docs.python.org/3 stable release branch)

## Source Fingerprint
sha256:bf149b1649601f39dfcbb435a1c1650e2607fe9d5f315fc93ea8432b28ebae6b (stdtypes.html, 3.14.8, fetched raw bytes)
sha256:f057eb26e5f008e87d25fecace3f88ca29b46776007687627ddb26a2dd2e08ba (datamodel.html, 3.14.8, fetched raw bytes)

## Invalidates On
- The `str()` entry in stdtypes.html changes the sentence "If object does not
  have a `__str__()` method, then `str()` falls back to returning `repr(object)`",
  or the `object.__str__` entry in datamodel.html stops saying the default
  implementation calls `object.__repr__()` (re-fingerprint either page).
- A Python version whose data model gives `object` a `__str__` that does not
  delegate to `__repr__`, or that makes `str()` raise `TypeError`/return
  `object.__repr__`-independent output for classes without `__str__` (e.g. a
  different docs branch than the 3.14.8 pages fingerprinted here, or any
  2.x-era semantics where old-style classes behaved otherwise).
- Target runtime is not CPython-3 semantics (an alternative interpreter, or a
  class that inherits `__str__` from a non-`object` base — that inherited
  method wins, so no `__repr__` fallback occurs).

## Checked At
2026-10-06

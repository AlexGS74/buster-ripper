VERDICT: UNCLEAR
PROMOTION: INELIGIBLE — UNCLEAR verdict; “sibling subclasses” has two plausible readings (P9).

## Claim
`UnicodeDecodeError` and `json.JSONDecodeError` are sibling subclasses of `ValueError` in the stdlib exception hierarchy — neither is a subclass of the other, so `except json.JSONDecodeError` cannot catch a `UnicodeDecodeError`

## What it actually does
`UnicodeDecodeError` inherits via `UnicodeError` from `ValueError`, while `JSONDecodeError` directly inherits `ValueError`; neither inherits from the other, so that `except` does not catch a `UnicodeDecodeError`. If “sibling” means immediate children of `ValueError`, the claim is false; if it means unrelated descendant branches under a common ancestor, it is true. The intended meaning of “sibling” would settle the whole-row verdict (P9).

## Source
Python 3.14 built-in exceptions documentation, “Exception hierarchy” and `UnicodeDecodeError`: https://docs.python.org/3.14/library/exceptions.html#exception-hierarchy ; installed `/opt/homebrew/Cellar/python@3.14/3.14.7/Frameworks/Python.framework/Versions/3.14/lib/python3.14/json/decoder.py:20-21` (`class JSONDecodeError(ValueError)`).

## Source Version
Python 3.14.7 documentation and installed stdlib 3.14.7.

## Source Fingerprint
unavailable — consulted documentation fetched without recoverable exact bytes for fingerprinting; installed decoder SHA-256: dc98d3e677612ea7311303f06a144cec6b91ebadc7ec037d1777501647578a07.

## Invalidates On
Python exception inheritance or `json.decoder.JSONDecodeError` base-class changes.
Clarification of whether “sibling” requires a direct shared parent.

## Checked At
2026-09-27

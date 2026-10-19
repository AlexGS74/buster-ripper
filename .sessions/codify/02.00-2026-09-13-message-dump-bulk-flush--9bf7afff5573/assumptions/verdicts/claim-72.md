VERDICT: HOLDS
PROMOTION: INELIGIBLE — establishing the full relationship requires both the JSON decoder source and the built-in exception hierarchy; no single consulted locator establishes both.

## Claim
`json.JSONDecodeError` is not a subclass of `UnicodeDecodeError`

## What it actually does
`JSONDecodeError` directly subclasses `ValueError`. `UnicodeDecodeError` subclasses `UnicodeError`, which subclasses `ValueError`; they are distinct branches beneath `ValueError`, so a `JSONDecodeError` is not a `UnicodeDecodeError`.

## Source
/opt/homebrew/Cellar/python@3.14/3.14.7/Frameworks/Python.framework/Versions/3.14/lib/python3.14/json/decoder.py:20; https://docs.python.org/3.14/library/exceptions.html#exception-hierarchy

## Source Version
Python 3.14.7 (installed JSON decoder and Python 3.14.7 documentation)

## Source Fingerprint
sha256:dc98d3e677612ea7311303f06a144cec6b91ebadc7ec037d1777501647578a07 (installed JSON decoder only; fetched exception documentation fingerprint unavailable)

## Invalidates On
Change to `JSONDecodeError` base classes in the installed JSON decoder.
Change to Python's `UnicodeDecodeError` exception hierarchy.

## Checked At
2026-09-27

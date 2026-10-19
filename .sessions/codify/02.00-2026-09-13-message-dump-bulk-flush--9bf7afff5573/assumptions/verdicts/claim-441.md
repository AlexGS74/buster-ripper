VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`urllib.request.build_opener` skips its default `ProxyHandler` when the caller supplies a `ProxyHandler` instance

## What it actually does
`build_opener` builds a `default_classes` list whose first entry is `ProxyHandler`, then removes from that list any default class matched by a caller-supplied handler — either an instance (`isinstance`) or a subclass (`issubclass`) — and only then instantiates the remaining defaults and appends the caller's handlers. So passing a `ProxyHandler` instance (or subclass) suppresses the default `ProxyHandler`; the supplied handler is added instead. The documentation says the default classes are placed in front "unless the handlers contain them, instances of them or subclasses of them," and CPython 3.15's `Lib/urllib/request.py` implements exactly that skip loop.

## Source
https://docs.python.org/3/library/urllib.request.html — `build_opener` entry (defaults list including `ProxyHandler (if proxy settings are detected)`, "unless the handlers contain them, instances of them or subclasses of them"); corroborated by CPython 3.15 `Lib/urllib/request.py`, `build_opener` lines 539-576

## Source Version
Python 3.15.0 documentation (docs.python.org/3); installed interpreter version not determined (execution prohibited)

## Source Fingerprint
sha256:a5bb216f0ea65aef0ba804f0183cdeabd275f3f003bc53898744442a4f1f0eb4

## Invalidates On
- CPython changes the `skip` / `default_classes` logic in `Lib/urllib/request.py::build_opener`
- The `build_opener` documentation paragraph stops listing `ProxyHandler` among the defaults suppressed by caller-supplied handlers
- `build_opener` changes to skip defaults only for subclasses, not instances

## Checked At
2026-10-17

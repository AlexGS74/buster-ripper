VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`urllib.request.ProxyHandler` with an empty proxy mapping installs no proxy for any host

## What it actually does
`ProxyHandler({})` keeps the empty mapping as-is — only `proxies=None` triggers the `getproxies()` environment lookup — and because the mapping has no entries it sets no `<protocol>_open` methods on the instance. `OpenerDirector.add_handler` explicitly skips the class's own `proxy_open` method, so nothing redirects any request through a proxy for any host. The documentation states "To disable autodetected proxy pass an empty dictionary" and that a `<protocol>_open()` method exists only "for every protocol which has a proxy in the `proxies` dictionary given in the constructor."

## Source
https://docs.python.org/3/library/urllib.request.html — `ProxyHandler` class entry ("To disable autodetected proxy pass an empty dictionary") and "ProxyHandler Objects" section (`<protocol>_open` exists only for protocols present in the `proxies` dictionary); corroborated by CPython 3.15 `Lib/urllib/request.py`, `ProxyHandler.__init__` lines 762-777 and `OpenerDirector.add_handler` line 413

## Source Version
Python 3.15.0 documentation (docs.python.org/3); installed interpreter version not determined (execution prohibited)

## Source Fingerprint
sha256:a5bb216f0ea65aef0ba804f0183cdeabd275f3f003bc53898744442a4f1f0eb4

## Invalidates On
- `ProxyHandler.__init__` changes to treat an empty mapping like `None` and call `getproxies()`
- `ProxyHandler` gains a `default_open`/`unknown_open` or unconditional proxying path independent of the `proxies` mapping
- `OpenerDirector.add_handler` stops excluding `proxy_open` from handler method discovery
- The `ProxyHandler` documentation stops stating that an empty dictionary disables autodetected proxying

## Checked At
2026-10-17

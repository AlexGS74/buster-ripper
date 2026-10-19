VERDICT: HOLDS
PROMOTION: INELIGIBLE — evidence is web-fetched rendered documentation; the exact consulted bytes cannot be recovered for a source-stable sha256 (only the URL is stable). See ## Source for the exact locator.

## Claim
The Python standard library provides a ready-made LRU caching facility in `functools.lru_cache`

## What it actually does
`functools.lru_cache` is a stdlib-supplied LRU cache that needs no implementation by the caller.
The docs define it as `@functools.lru_cache(maxsize=128, typed=False)` — "Decorator to wrap a
function with a memoizing callable that saves up to the maxsize most recent calls" — with
`maxsize=128` as the default, "the LRU feature" explicitly named as what turns off when `maxsize`
is `None`, and a "size limit [that] assures that the cache does not grow without bound". The
memoizing callable is instrumented with `cache_info()` (returning `CacheInfo(hits, misses, maxsize,
currsize)`), `cache_clear()`, `cache_parameters()`, and `__wrapped__`, so the LRU behaviour is
pre-wired and observable rather than hand-rolled. It has shipped since Python 3.2 (`typed` added in
3.3, direct-application form in 3.8, `cache_parameters()` in 3.9) and is unchanged in kind in the
current docs. The one thing it is *not* is a key/value container: it memoizes calls to one wrapped
function, keys are the call's positional/keyword arguments (so those must be hashable), and there is
no public get/set surface — which is the exact boundary that keeps this claim and claim 334
("no instantiable cache container type") simultaneously true. Checked independently (P4); the two
verdicts do not lean on each other and do not contradict.

## Source
https://docs.python.org/3.13/library/functools.html — section `@functools.lru_cache(user_function)` /
`@functools.lru_cache(maxsize=128, typed=False)` ("Decorator to wrap a function with a memoizing
callable that saves up to the maxsize most recent calls", "If maxsize is set to None, the LRU
feature is disabled", `cache_info()`/`cache_clear()`/`cache_parameters()`, "An LRU (least recently
used) cache works best when the most recent calls are the best predictors of upcoming calls"), plus
`@functools.cache` ("Returns the same as lru_cache(maxsize=None)"). Cross-checked at
https://docs.python.org/3/library/functools.html (current 3.14 stable docs), same wording.

## Source Version
CPython 3.13.16 documentation (page last updated 2026-10-01 UTC); wording identical on the current
3.14 stable page. Installed interpreter version was not probed (executing Python is outside this
role); the repo declares `requires-python = ">=3.11"` (pyproject.toml:4), and `lru_cache` has been
in the stdlib since 3.2, so it is present across that whole range.

## Source Fingerprint
unavailable — web-fetched rendered text, not recoverable byte-for-byte from the source

## Invalidates On
- docs.python.org/3/library/functools.html removing `lru_cache`, moving it out of the stdlib, or
  reclassifying it as a deprecated alias
- The documented eviction policy changing from least-recently-used to another replacement policy,
  or `maxsize` ceasing to bound stored entries
- The decorator becoming non-stdlib (e.g. requiring an install), or `functools` no longer being
  part of the Python standard library documentation tree
- Any Python version below the project's floor that lacks `lru_cache` (none in the >=3.11 range)

## Checked At
2026-10-05

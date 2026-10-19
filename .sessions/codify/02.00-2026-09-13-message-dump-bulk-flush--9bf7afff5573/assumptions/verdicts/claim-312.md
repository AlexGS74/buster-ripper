VERDICT: UNCLEAR
PROMOTION: INELIGIBLE — verdict is UNCLEAR (promotion requires HOLDS or DOES-NOT-HOLD); additionally the claim's term "container" has two stdlib-vocabulary readings that flip the verdict

## Claim

The Python standard library provides no TTL or LRU cache container

## What it actually does

The TTL half is stable under every reading: the standard library has no
time-bounded/expiring cache. `functools.lru_cache` takes only `maxsize` and
`typed` — no age/TTL parameter — and no top-level `Lib/` module is a cache
container. The stdlib's own documentation for "cache with expiry" is a
hand-written example class, `TimeBoundedLRU`, built out of
`collections.OrderedDict` + `time.monotonic`, presented as something *you*
implement ("An `OrderedDict` would also be useful for implementing variants of
`:deco:`functools.lru_cache`").

The LRU half depends on which meaning of "container" the claim intends, and the
two readings give opposite verdicts — so no single behavioural finding can be
reported (P9):

- **Loose reading — "cache container" = any ready-made caching facility.**
  The claim is false. `functools.lru_cache` is documented in the stdlib as "An
  LRU (least recently used) cache", with `maxsize` eviction, `cache_info()`,
  `cache_clear()` and `cache_parameters()`; `functools.cache` is the unbounded
  variant. So the stdlib *does* provide LRU caching.
- **Strict reading — "container" in Python's own vocabulary** (glossary: lists,
  tuples, sets, dicts; a "mapping" is "a container object that supports
  arbitrary key lookups and implements the methods specified in
  `collections.abc.Mapping` or `MutableMapping`"). The claim is true. The stdlib
  exposes no cache *type* you instantiate and use as a key/value store: no
  `Mapping`-based LRU or TTL class, no cache ABC in `collections.abc`.
  `lru_cache` is a decorator returning a memoizing *callable* whose dictionary
  is not exposed for lookup, iteration, or per-key eviction.

What settles it: whether the surrounding need is *memoizing a function* (LRU is
already provided — nothing to build) or *holding entries in a reusable
key/value object with its own eviction/expiry policy* (not provided — built
from `OrderedDict`/`dict` + timestamps, or taken from a third-party package
such as `cachetools`, which is not in the standard library).

## Source

- `Doc/library/functools.rst`, `lru_cache` entry — lines 151-190, 226-227
  (`https://raw.githubusercontent.com/python/cpython/3.14/Doc/library/functools.rst`)
- `Doc/library/collections.rst`, OrderedDict differences (line 1119, "suitable
  for implementing various kinds of LRU caches") and the
  `TimeBoundedLRU` / `MultiHitLRUCache` recipes (lines 1227-1256)
  (`https://raw.githubusercontent.com/python/cpython/3.14/Doc/library/collections.rst`)
- `Doc/glossary.rst`, `mapping` (line 1015) and `container` usage in the
  `hashable` entry (lines 764-765)
  (`https://raw.githubusercontent.com/python/cpython/3.14/Doc/glossary.rst`)
- Negative-existence corroboration for the TTL half: top-level `Lib/` listing on
  branch `3.14` (`https://api.github.com/repos/python/cpython/contents/Lib?ref=3.14`)
  — contains no cache/TTL module; `functools.py` and `collections/` are the
  modules stdlib caching lives in.

Documentation-only claim about a library; no local source root was needed and
none was read. Nothing was executed.

## Source Version

CPython `3.14` branch documentation (`docs.python.org/3` renders as 3.14.8);
doc sources fetched from `raw.githubusercontent.com/python/cpython/3.14`.
No interpreter, installed package, or repository source was consulted.

## Source Fingerprint

- functools.rst `sha256:e938fb679d763ffc02472865a1ffbd816d0f4665068d1419b2d3ac6d1a9fbc31`
- collections.rst `sha256:be2f90d14d3f488a9d5a6f23423e494a46b10a030d83d9308fd71af1f8eed7a7`
- glossary.rst `sha256:a5b7304a788ae34025a13f7e38fc18a6c2b5d3299deefd7f82cacc4cb47e0d91`

## Invalidates On

- Re-extraction of the claim naming which reading of "container" is intended —
  that alone converts this to HOLDS (strict) or DOES-NOT-HOLD (loose).
- A CPython release adding a TTL/expiry or mapping-type cache API to
  `functools`, `collections`, or `collections.abc` (would make the TTL half
  false even under the strict reading).
- Checking against a different stdlib version than 3.14 — re-fetch the
  `functools` and `collections` doc sources above; the LRU half is version-
  stable, the "what exists to build on" half is not.
- If the requirement turns out to be TTL caching specifically, this verdict is
  moot: no stdlib TTL cache exists under either reading, so a third-party
  package or a hand-rolled expiry policy is required regardless.

## Checked At

2026-10-05

VERDICT: HOLDS
PROMOTION: INELIGIBLE — evidence is web-fetched rendered documentation; the exact consulted bytes cannot be recovered for a source-stable sha256 (only the URL is stable). See ## Source for the exact locators.

## Claim
The Python standard library provides no instantiable cache container type that expires entries by age or evicts them by least-recently-used order

## What it actually does
Both disjuncts come out empty, so the negative holds. The standard library's caching
facilities are *function decorators*, not containers: `functools.lru_cache` is documented as
"Decorator to wrap a function with a memoizing callable", and `functools.cache` "never needs to
evict old values". Neither exposes a mapping interface — entries exist only as call-argument keys,
and the only surface is `cache_info()` / `cache_clear()` / `cache_parameters()`; neither has any
time-to-live or expiry knob. The module that *is* titled "Container datatypes" (`collections`:
`namedtuple`, `deque`, `ChainMap`, `Counter`, `OrderedDict`, `defaultdict`, `UserDict`, `UserList`,
`UserString`) provides no automatic eviction of any kind: `OrderedDict` has no size limit at all
(`move_to_end()` and `popitem(last=False)` are the raw material you drive yourself), and
`deque(maxlen=N)` discards "from the opposite end" on append — insertion order, i.e. FIFO, which
ignores read/access recency. `weakref`'s cache-oriented mappings (`WeakKeyDictionary`,
`WeakValueDictionary`, `WeakSet`) do drop entries, but because garbage collection reclaimed the
key/value — reachability, not age and not LRU order. The documentation itself shows the gap: the
`OrderedDict Examples and Recipes` section says OrderedDict "makes it suitable for implementing
various kinds of LRU caches" and then hands the reader hand-written classes to do it
(`TimeBoundedLRU(func, maxsize=128, maxage=30)` and `MultiHitLRUCache`), i.e. age-bounded LRU is a
recipe you write, not a stdlib type you instantiate.

Near-misses named explicitly so the boundary is auditable, not assumed: `functools.lru_cache`
(recency-evicting, but a decorator over a function, not a container you `put`/`get`),
`collections.deque(maxlen=)` (bounded, but FIFO not LRU), `weakref.Weak*` mappings and `WeakSet`
(auto-discarding, but GC-driven), `functools.cached_property` (caches into the instance `__dict__`
"for the life of the instance", no eviction), and `http.cookiejar`'s per-cookie `expires`
(a cookie store's own expiry metadata, not a general-purpose cache container). TTL/LRU *container*
classes with the API this claim negates exist only outside the stdlib (third-party `cachetools`:
`TTLCache`, `LRUCache`, `TLRUCache`, described in their own docs as "variants of the Python Standard
Library's `@lru_cache` function decorator").

Two caveats on wording, recorded rather than resolved by fiat:
1. The `functools` sentence "The cache keeps references to the arguments and return values until
   they age out of the cache" uses "age out" for LRU eviction; it is not time-based expiry and must
   not be read as supporting the first disjunct.
2. Claim 335 in this bundle is about the same library and also holds. They are consistent only
   because `lru_cache` is a caching *facility* and not a cache *container type*. Checked
   independently (P4): if a reader widened "instantiable cache container type" to cover the
   object `lru_cache` produces, this claim would flip to DOES-NOT-HOLD while 335 stayed HOLD. That
   reading is against the docs' own classification (decorator, vs. `collections` = "Container
   datatypes"), so it is noted as the seam and does not soften this verdict.

## Source
https://docs.python.org/3.13/library/collections.html — `collections — Container datatypes`,
sections "OrderedDict objects" and "OrderedDict Examples and Recipes" (`TimeBoundedLRU`,
`MultiHitLRUCache`), `deque objects` (`maxlen`), `ChainMap`/`Counter`/`defaultdict`.
Corroborating: https://docs.python.org/3.13/library/functools.html (`cache`, `lru_cache`,
`cached_property`) and https://docs.python.org/3.13/library/weakref.html (`WeakKeyDictionary`,
`WeakValueDictionary`, `WeakSet`). Version-dependence checked at
https://docs.python.org/3/whatsnew/3.14.html and https://docs.python.org/3/library/functools.html
(3.14 line) — 3.14's new modules are `compression` and `concurrent.interpreters`; no cache
container was added, and 3.14 still documents `lru_cache` as a decorator.

## Source Version
CPython 3.13.16 documentation (page last updated 2026-10-01 UTC), cross-checked against the
current 3.14 stable docs. Installed interpreter version was not probed (executing Python is
outside this role); the repo declares `requires-python = ">=3.11"` (pyproject.toml:4), and no
version in that range ships such a container.

## Source Fingerprint
unavailable — web-fetched rendered text, not recoverable byte-for-byte from the source

## Invalidates On
- Any CPython release adding a stdlib container type with built-in age/TTL expiry or automatic
  recency-based eviction (would falsify the claim even though `lru_cache`/`cache` alone stay decorators)
- docs.python.org/3.13/library/collections.html losing the `TimeBoundedLRU` / `MultiHitLRUCache`
  OrderedDict recipes, or `OrderedDict` gaining a size limit or eviction policy
- `collections.deque(maxlen=)` documentation changing from end-on-append discard to access-recency eviction
- `functools.lru_cache` / `functools.cache` documentation gaining a TTL/expiry parameter or a public
  mapping protocol (get/set/len/iter) on the wrapped callable
- The consulted doc branch moving from 3.13/3.14 to a version whose `functools` or `collections`
  page adds a cache container type

## Checked At
2026-10-05

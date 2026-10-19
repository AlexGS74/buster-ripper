VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
A plain, unspecced `unittest.mock.MagicMock` folds `__aenter__` and `__aexit__` into its magic set and routes those children to `AsyncMock`, so `async with` succeeds on it without further configuration

## What it actually does
Exactly as claimed, in all three links of the chain. In `Lib/unittest/mock.py` (verified identically on two pinned versions): (1) `_async_method_magics = {"__aenter__", "__aexit__", "__anext__"}` is unioned into `_all_magics` (`_all_magics = _all_sync_magics | _async_magics`), and `MagicMixin._mock_set_magics` installs a `MagicProxy` for every entry in `_magics | _async_method_magics` on the mock's type — so both async context-manager dunders are in the fold. (2) `MagicProxy.__get__` → `create_mock()` builds the child via `parent._get_child_mock(name=entry, _new_name=entry, ...)`, and `_get_child_mock` contains the explicit branch `if issubclass(_type, MagicMock) and _new_name in _async_method_magics: # Any asynchronous magic becomes an AsyncMock → klass = AsyncMock` — so on a plain `MagicMock` the children `m.__aenter__` and `m.__aexit__` are `AsyncMock` instances. (3) An `AsyncMock` call is awaitable (`AsyncMockMixin._execute_mock_call` is `async def`), and `create_mock`'s `_set_return_value` applies the `_return_values` default `'__aexit__': False` — so `async with m:` enters (yielding `m.__aenter__.return_value`, itself an `AsyncMock`) and exits cleanly with no configuration. "Plain and unspecced" is load-bearing and consistent with the code: a spec would prune `_mock_set_magics` to the spec's intersection, and non-`MagicMock` parents (e.g. `Mock`) do not take the `AsyncMock` routing branch.

## Source
Primary: CPython upstream `Lib/unittest/mock.py` at tag `v3.11.0` — https://raw.githubusercontent.com/python/cpython/v3.11.0/Lib/unittest/mock.py (lines ~1966–1972 `_async_method_magics`/`_all_magics`, ~1991–2005 `_return_values['__aexit__']`, ~1017 `_get_child_mock` AsyncMock routing, ~2077 `_mock_set_magics`, ~2140 `MagicProxy.create_mock`). Corroborated byte-for-byte structurally at tag `v3.13.1` (same constants and routing). Corroborated by official docs https://docs.python.org/3/library/unittest.mock.html: supported magic methods list ("Context manager: `__enter__`, `__exit__`, `__aenter__` and `__aexit__`"), MagicMock "default implementations of most of the magic methods… without having to configure the magic methods yourself", defaults table `__aexit__: False`, and "Changed in version 3.8: Added support for `__aenter__`, `__aexit__`…". Claim is about the library, not this repo; no session files read. (Repo `pyproject.toml` `requires-python = ">=3.11"` was consulted only to choose which upstream tags to check; the mechanism is present throughout that range, since 3.8.)

## Source Version
CPython 3.11.0 and 3.13.1 (upstream source at release tags); docs.python.org/3 fetched 2026-10-06. Repo declares >=3.11, fully covered.

## Source Fingerprint
- sha256:663efa5704c1146a2327248637fe1f427a4b517dfb28250f6894e844f0b5c1eb (Lib/unittest/mock.py @ v3.11.0, primary establishing source)
- sha256:7c6b6c92b11243ebe8e93e0230be077b83224ccf0030228c0aa45d1db5613811 (Lib/unittest/mock.py @ v3.13.1, corroboration)
- sha256:6513ae8d94214d97ca409b654b564a64390e4cd66dc7ff8cd6ebf705c3f01a9f (docs.python.org/3/library/unittest.mock.html as fetched, corroboration)

## Invalidates On
- A CPython change removing `__aenter__`/`__aexit__` from `_async_method_magics`, from `_all_magics`/`_mock_set_magics`, or from the `_get_child_mock` AsyncMock-routing branch
- A CPython change to `_return_values['__aexit__']` or `MagicProxy.create_mock`'s `_set_return_value` wiring
- Running on Python < 3.8 (async magic support absent), or using the standalone `mock` backport whose pinned version predates this routing
- The mock in question ceasing to be plain and unspecced (`spec`/`spec_set`/`autospec` prune the magic set; a non-`MagicMock` parent skips the AsyncMock branch)

## Checked At
2026-10-06

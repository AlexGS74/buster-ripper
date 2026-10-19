VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`unittest.mock.AsyncMock` supplies awaitable `__aenter__` and `__aexit__`, so an `AsyncMock` can stand in for an object consumed with `async with`

## What it actually does
`AsyncMock` inherits the magic-method installer (`AsyncMagicMixin` → `MagicMixin`), whose `_mock_set_magics()` puts a `MagicProxy` on the type for every entry of `_magics | _async_method_magics`, and `_async_method_magics` is exactly `{"__aenter__", "__aexit__", "__anext__"}`. Accessing `mock.__aenter__` / `mock.__exit__`-time lookup therefore routes through `MagicProxy.__get__` → `Mock._get_child_mock()`, which for an `AsyncMockMixin` parent names anything not in `_all_sync_magics` an `AsyncMock` — and calling an `AsyncMock` runs `CallableMixin.__call__` → `_mock_call` → `AsyncMockMixin._execute_mock_call`, an `async def`, so the result is a coroutine (awaitable). `async with AsyncMock()` therefore enters and exits through awaitables; `__aexit__` is preconfigured with `return_value = False` so the `async with` body is not suppressed, and `__aenter__` returns a new `AsyncMock` bound to the `as` target.

Chain, per line in the consulted source: `_async_method_magics` (2003) → `MagicMixin._mock_set_magics` `orig_magics` (2114) and `setattr(_type, entry, MagicProxy(entry, self))` (2133) → `MagicProxy.__get__`/`create_mock` (2182-2192) → `_get_child_mock` `elif issubclass(_type, AsyncMockMixin): ... klass = AsyncMock` (1036-1042, since `__aenter__`/`__aexit__` are absent from `_all_sync_magics = _magics | _non_defaults`, 2008) → `CallableMixin.__call__`/`_mock_call` (1119-1128) → `async def _execute_mock_call` (2228) → `_return_values['__aexit__'] = False` (2038). Class wiring: `class AsyncMagicMixin(MagicMixin)` (2149), `class AsyncMock(AsyncMockMixin, AsyncMagicMixin, Mock)` (2390).

Corroborating documentation (secondary, not the basis): `unittest.mock` docs list "Context manager: `__enter__`, `__exit__`, `__aenter__` and `__aexit__`" among supported magic methods and give `__aexit__: False` in the magic-method defaults table; the `AsyncMock` entry states "the result of a call is an awaitable".

Caveat that bounds the claim (does not soften it): this is the un-spec'd construction. With `spec=`/`spec_set=` (or an autospec'd `patch`) of a type that has no `__aenter__`, `_mock_set_magics` computes `these_magics = orig_magics.intersection(self._mock_methods)` and deletes the extras, so the awaitable pair is pruned.

## Source
https://raw.githubusercontent.com/python/cpython/v3.11.10/Lib/unittest/mock.py — lines 2003, 2008, 2038, 2114, 2133, 2149, 2182-2192, 1036-1042, 1119-1128, 2228, 2390. Corroboration: https://docs.python.org/3.11/library/unittest.mock.html ("Magic Mock" / magic-methods support list, "Async Mock").

## Source Version
CPython tag `v3.11.10`, `Lib/unittest/mock.py` (repo declares `requires-python = ">=3.11"`; mechanism re-checked unchanged at `v3.14.0` — `_async_method_magics` at 2070, `orig_magics` at 2181, "Any asynchronous magic becomes an AsyncMock" at 1080, `class AsyncMock(AsyncMockMixin, AsyncMagicMixin, Mock)` at 2476). Local interpreter was not probed and nothing was executed to establish this verdict (P1).

## Source Fingerprint
sha256:2576119e5653ca21ab32dd5a1a5f5604deb2b1c7a240e315710ea927bc193749 (v3.11.10 `Lib/unittest/mock.py`, exact bytes consulted)
sha256:667d9ed88fd330a3d45419436b13ac41488fca77f39959dca6e93642cf80a678 (v3.14.0 `Lib/unittest/mock.py`, confirmation read only)

## Invalidates On
- The consulted `Lib/unittest/mock.py` bytes change (fingerprint mismatch) — specifically edits to `_async_method_magics`, `MagicMixin._mock_set_magics`, `MagicProxy`, `Mock._get_child_mock`, or `AsyncMockMixin._execute_mock_call`.
- Upstream removes or renames `_async_method_magics` / `AsyncMagicMixin`, or drops `__aenter__`/`__aexit__` from the magics installed on `AsyncMock`.
- The `AsyncMock` in question is constructed with `spec=`, `spec_set=`, or an autospec `patch()` whose target lacks `__aenter__`/`__aexit__` — the proxies are then pruned and are not awaitable.
- Docs page for the resolved interpreter version no longer lists `__aenter__`/`__aexit__` as supported magic methods.

## Checked At
2026-10-06

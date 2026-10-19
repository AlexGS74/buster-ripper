VERDICT: DOES-NOT-HOLD
PROMOTION: ELIGIBLE

## Claim
A plain `unittest.mock.MagicMock` supplies no awaitable `__aenter__`, so it cannot stand in for an object consumed with `async with`

## What it actually does
A plain (un-spec'd) `MagicMock` **does** supply an awaitable `__aenter__`, and `async with MagicMock()` works. `class MagicMock(MagicMixin, Mock)` inherits `MagicMixin.__init__` → `_mock_set_magics()`, whose magic set is `orig_magics = _magics | _async_method_magics` with `_async_method_magics = {"__aenter__", "__aexit__", "__anext__"}`; each entry is installed on the type as a `MagicProxy` descriptor. Because `async with` resolves `__aenter__`/`__aexit__` on the type, the descriptor fires, and `Mock._get_child_mock` has an explicit branch for this: `if issubclass(_type, MagicMock) and _new_name in _async_method_magics:` → `klass = AsyncMock`, with the in-source comment "Any asynchronous magic becomes an AsyncMock". Calling that child returns a coroutine (`AsyncMockMixin._execute_mock_call` is `async def`, reached via `CallableMixin.__call__` → `_mock_call`), so both the enter and the exit are awaitable. `__aexit__` additionally carries the preconfigured default `_return_values['__aexit__'] = False`, so the `async with` body is not suppressed; `__aenter__` yields a new `AsyncMock` for the `as` target, and `mock.__aenter__.assert_awaited()` / `assert_called_once()` are available afterwards.

What the claim *would* describe, as real adjacent behaviour: a bare `Mock()`/`NonCallableMock()` has no magic-method support at all, so `async with Mock()` raises `TypeError: 'Mock' object does not support the asynchronous context manager protocol`. And a `MagicMock(spec=...)` / autospec'd mock whose spec lacks `__aenter__` gets the proxy pruned by `_mock_set_magics` (`these_magics = orig_magics.intersection(self._mock_methods)`, then `delattr(self, entry)`), which also fails under `async with`. The distinguishing word in the claim is "plain", and for a plain MagicMock the library does the opposite of what the claim asserts.

Per P4 this row was checked on its own evidence: the deciding location is the `issubclass(_type, MagicMock)` branch of `_get_child_mock`, not anything about `AsyncMock`'s own inheritance.

## Source
https://raw.githubusercontent.com/python/cpython/v3.11.10/Lib/unittest/mock.py — lines 2003 (`_async_method_magics`), 2114 (`orig_magics = _magics | _async_method_magics`), 2133 (`setattr(_type, entry, MagicProxy(entry, self))`), 2155 (`class MagicMock(MagicMixin, Mock)`), 2182-2192 (`MagicProxy.create_mock`/`__get__`), 1033-1035 (`issubclass(_type, MagicMock) and _new_name in _async_method_magics` → `klass = AsyncMock`), 2038 (`'__aexit__': False`), 2228 (`async def _execute_mock_call`), 1119-1128 (`__call__` → `_mock_call`), 2117-2126 (spec pruning of magics). Corroboration: https://docs.python.org/3.11/library/unittest.mock.html — "Context manager: `__enter__`, `__exit__`, `__aenter__` and `__aexit__`", "Changed in version 3.8: Added support for `__aenter__`, `__aexit__`, `__aiter__` and `__anext__`", and `__aexit__: False` in the "Magic Mock" defaults table.

## Source Version
CPython tag `v3.11.10`, `Lib/unittest/mock.py` (repo declares `requires-python = ">=3.11"`; mechanism re-checked unchanged at `v3.14.0`: `_async_method_magics` 2070, "Any asynchronous magic becomes an AsyncMock" 1080, `class MagicMock(MagicMixin, Mock)` 2218, `orig_magics` 2181). Nothing was executed to establish this verdict (P1) — behaviour read from the library's own source; the local interpreter was not probed.

## Source Fingerprint
sha256:2576119e5653ca21ab32dd5a1a5f5604deb2b1c7a240e315710ea927bc193749 (v3.11.10 `Lib/unittest/mock.py`, exact bytes consulted)
sha256:667d9ed88fd330a3d45419436b13ac41488fca77f39959dca6e93642cf80a678 (v3.14.0 `Lib/unittest/mock.py`, confirmation read only)

## Invalidates On
- The consulted `Lib/unittest/mock.py` bytes change (fingerprint mismatch) — especially the `_async_method_magics` membership, `MagicMixin._mock_set_magics`, or the `issubclass(_type, MagicMock) and _new_name in _async_method_magics` branch of `Mock._get_child_mock`.
- Upstream stops auto-converting async magic methods on `MagicMock` to `AsyncMock` (i.e. the claim becomes true for that version).
- Reuse against a spec'd mock: `MagicMock(spec=...)`, `spec_set=True`, or `patch(..., autospec=True)` over a type without `__aenter__`/`__aexit__` — there the proxies are pruned and `async with` fails, so the corrected reading must not be reused for those constructions.
- Applied to `Mock`/`NonCallableMock` instead of `MagicMock` — those have no magic support and genuinely fail under `async with`.

## Checked At
2026-10-06

VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
Under pytest-asyncio's `asyncio_mode = "auto"`, an `async def` test function is collected and run on an event loop without any per-test marker

## What it actually does
In auto mode the plugin's `pytest_pycollect_makeitem` hookwrapper adds the `asyncio` marker itself
(`if _get_asyncio_mode(node.config) == Mode.AUTO and not node.get_closest_marker("asyncio"): node.add_marker("asyncio")`)
and then substitutes the plain `pytest.Function` item with the specialized `Coroutine` item class
(`PytestAsyncioFunction.item_subclass_for`, whose `Coroutine._can_substitute` is `inspect.iscoroutinefunction`).
That item's `runtest` resolves the scoped `asyncio.Runner` fixture (`_{loop_scope}_scoped_runner`, default
loop scope `function` from the `asyncio_default_test_loop_scope` ini default) and wraps the coroutine with
`_synchronize_coroutine`, which calls `runner.run(coro, context=context)` — so an author-unmarked
`async def` test is both collected as a real test item and executed on an event loop.

Nuance recorded rather than resolved by picking a definition: under the reading "the test author writes no
`@pytest.mark.asyncio`", the claim is exactly what the source does; under the reading "no `asyncio` marker is
present on the item at all", the internal bookkeeping differs (the plugin injects one), but the asserted
outcome — collected and run on an event loop — is unchanged under both readings, so the verdict does not
depend on the choice. In strict mode the automatic marker/loop handling does not happen (an unmarked async
test is left to pytest core), which is the counterpart behaviour this claim implicitly distinguishes.

## Source
/Users/alexhubitski/Projects/buster-ripper/.venv/lib/python3.13/site-packages/pytest_asyncio/plugin.py
(lines 118-123 ini default `asyncio_default_test_loop_scope = "function"`; 198-207 `_get_asyncio_mode`;
416-426 `item_subclass_for`; 454-469 `PytestAsyncioFunction.setup`/`runtest`; 498-504 `Coroutine._can_substitute`;
575-611 `pytest_pycollect_makeitem_convert_async_functions_to_subclass`; 703-718 `_synchronize_coroutine`;
794-830 `_create_scoped_runner_fixture`)

## Source Version
pytest-asyncio version string not readable from the supplied bounded source root
(`__init__.py` derives `__version__` at runtime via `importlib.metadata`, and the `*.dist-info` directory
sits outside the supplied package root, so it was not read). Stability is carried by the fingerprint below.
Source files dated 2026-05-19; the API surface consulted (Runner-based `_scoped_runner` fixtures,
`asyncio_default_test_loop_scope`, `loop_scope` marker kwarg) corresponds to the pytest-asyncio 0.23+
Runner redesign.

## Source Fingerprint
sha256:121cb98d74596a9e5db3269a4be7131baeeeab3754b5996a9b26c228910863aa (plugin.py, 896 lines, exact bytes consulted)

## Invalidates On
- Any change to `pytest_pycollect_makeitem_convert_async_functions_to_subclass` (plugin.py:599-611) that removes
  or conditions the `node.add_marker("asyncio")` auto-mode injection.
- sha256 of `.venv/lib/python3.13/site-packages/pytest_asyncio/plugin.py` differing from
  `121cb98d74596a9e5db3269a4be7131baeeeab3754b5996a9b26c228910863aa` (upgrade/downgrade of pytest-asyncio).
- Rename or removal of the `asyncio_default_test_loop_scope` ini option or the `_{scope}_scoped_runner` fixture
  naming scheme (plugin.py:118-123, 794-830), which the item's `runtest` depends on.
- A pytest core change that stops reporting `async def` functions as collectible `Function` items before
  `pytest_pycollect_makeitem` runs (e.g. skipping unhandled coroutines at collection), since the plugin
  substitutes an already-produced `Function`.

## Checked At
2026-10-06

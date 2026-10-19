VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
aiohttp runs each request handler as a coroutine on the application's event loop, so a synchronous section of a handler cannot interleave with another handler

## What it actually does
On aiohttp 3.13.3 every request is dispatched into its own `asyncio.Task` created on the
loop the `Application` is bound to (`RequestHandler.start()` → `asyncio.Task(coro, loop=loop,
eager_start=True)` on Python ≥3.12, `loop.create_task(coro)` otherwise), and the user handler is
reached by plain `await` chains down that task (`_handle_request` → `await request_handler(request)`
→ `Application._handle` → `return await handler(request)`). Because a `Task` is a coroutine driven by a
single-threaded event loop that transfers control only at suspension points, code inside a handler that
contains no `await` runs to completion before any other handler task's code executes — no interleaving.
Scope of the verdict (so a later re-check is not measuring a different statement): the claim is about
*preemption within a synchronous section*, not about handlers being unable to be concurrent — two handlers
are routinely concurrent tasks, they simply only alternate at `await` points.
Boundaries, none of which contradict the claim:
- Bare (non-`async`) handler functions are not moved off the loop: `web_urldispatcher.py` wraps them in a
  coroutine `handler_wrapper` that calls them inline, so they get the same no-preemption guarantee.
- The only `run_in_executor` on a request path in this area is `StaticResource._handle` (path resolution for
  static files), which is aiohttp's own work, is awaited, and therefore is not a handler's sync section.
  Work *this repo* offloads with `loop.run_in_executor` / `asyncio.to_thread` / its own threads is outside the
  claim and *can* interleave with other handlers.
- Same-connection HTTP/1.1 keep-alive requests are additionally sequential (`resp, reset = await task` before
  the next message is consumed); overlap between handlers arises across connections / HTTP-2 streams. Either
  way the sync section is atomic.
- Multi-process serving (gunicorn / multiple `run_app` workers) means separate loops and separate `Application`
  instances, which is a different subject than "the application's event loop"; one `Application` cannot even be
  served on two loops (`_set_loop` raises `RuntimeError("web.Application instance initialized with different loop")`).

## Source
Primary (establishes the claim): `aiohttp/web_protocol.py` at tag `v3.13.3`, lines 552 (`loop = self._loop`),
596-602 (`coro = self._handle_request(...)`, `asyncio.Task(coro, loop=loop, eager_start=True)` /
`loop.create_task(coro)`), 510 (`resp = await request_handler(request)`) —
`https://raw.githubusercontent.com/aio-libs/aiohttp/v3.13.3/aiohttp/web_protocol.py`
Corroborating, same tag: `aiohttp/web_app.py` 433, 442-446, 527, 569, 271-279; `aiohttp/web_runner.py`
390-391 (`BaseSite._make_server`: `loop = asyncio.get_event_loop(); self._app._set_loop(loop)`);
`aiohttp/web_server.py` 75-84 (`Server.__call__` → `RequestHandler(self, loop=self._loop)`);
`aiohttp/web_urldispatcher.py` 207-219 (bare-function handler wrapped inline), 677-683 (static-file executor).
Version-pin note: behavior is identical on the previously-grounded `v3.14.1` for this claim — its
`web_protocol.py` 673-677 creates the per-request task with the same `eager_start` / `create_task` pair and
`web_app.py` 517, 559 still `await handler(request)`. This verdict is nonetheless grounded on 3.13.3 bytes.

## Source Version
aiohttp 3.13.3 (upstream git tag `v3.13.3`); version-compared against tag `v3.14.1`. Python `asyncio`
cooperative-scheduling semantics assumed as documented for the standard library.

## Source Fingerprint
sha256:eacf5d33399aa96efb6f33354f5d75b863522c5a3a80d99f0e0ed7cd89c0f319  (web_protocol.py @ v3.13.3)
sha256:94653f68030dfa1df0cbe2d34c78ba49e287f3276dd46e75057702b29803e590  (web_app.py @ v3.13.3)
sha256:bf51b59ca88e3908059d3491e0831ce88f5178414331a1ed30bbcefeba0333e0  (web_runner.py @ v3.13.3)
sha256:fbd58329402247d965fab49dc174aa1ba6236a85bbf5dd51e32d01192aa050c0  (web_server.py @ v3.13.3)
sha256:24cf9396b88a08d6d32cd2f8dd16bdb1d674cc287166620461033a6696f28c8e  (web_urldispatcher.py @ v3.13.3)

## Invalidates On
- `aiohttp/web_protocol.py` at the resolved tag changes `start()` so a request is no longer dispatched via
  `asyncio.Task(..., loop=loop, eager_start=True)` / `loop.create_task(coro)` on the server loop (e.g. an
  executor or thread-pool dispatch of handlers).
- `aiohttp/web_app.py` changes `_handle` so the matched handler is no longer reached by `await handler(request)`
  in the request task (e.g. scheduled onto another loop).
- aiohttp starts binding an `Application` to a loop other than the serving loop, or `_set_loop` stops raising on
  a second loop — that breaks "the application's event loop".
- The Python runtime stops guaranteeing single-threaded, non-preemptive coroutine execution for one loop
  (free-threaded build running the same loop concurrently, or an asyncio rewrite that preempts running frames).
- The handler's own code introduces off-loop work inside the "synchronous" region (`run_in_executor`,
  `asyncio.to_thread`, `threading`) or awaits into another handler — that is not this claim's subject and
  would need its own verdict.

## Checked At
2026-10-05T21:06:04Z

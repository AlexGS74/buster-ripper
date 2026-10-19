VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
aiohttp's `web.Request` enforces its body-size limit from the `client_max_size` configured on the application whose runner serves the connection and built the request

## What it actually does
The limit is a constructor-time snapshot copied from the `Application` that builds the request: `AppRunner._make_server()` calls `self._app._make_handler(...)`, which constructs `Server(handler=self._handle, request_factory=self._make_request)`; `Application._make_request` passes `client_max_size=self._client_max_size` (the app's `__init__` kwarg, default `1024**2`) into `BaseRequest.__init__`, which stores it as `self._client_max_size` and raises `HTTPRequestEntityTooLarge` from `read()` / `text()` / `json()` / `post()` when the body reaches it. Each connection's `RequestHandler` holds that same bound factory (`self._request_factory = manager.request_factory`) and calls it per request, so the app that owns the runner is exactly the app that supplies the limit — no per-route or per-connection lookup happens.

Scope details read from the same source (they qualify, not contradict):
- `0` disables the check (`if self._client_max_size:` at web_request.py:675).
- The gate lives in `Request`'s own body readers; `request.content` (the raw payload `StreamReader`) is not gated by `_client_max_size`.
- Multipart `post()` counts toward the same value (`max_size = self._client_max_size`, web_request.py:722); urlencoded `post()` delegates to `read()`.
- Sub-apps (`add_subapp`) never supply their own `client_max_size` — body reads stay bounded by the runner's (root) app's value, which matches the claim.
- `Server` built without a `request_factory` falls back to `Server._make_request`, which constructs `BaseRequest` with the default `1024**2` and no app involved (web_server.py:32, 55-63) — that path has no application at all.
- `request.clone(...)`/`with_method(...)` preserve the app-derived value unless `client_max_size` is passed explicitly (web_request.py:251-261).

## Source
`/Users/alexhubitski/Projects/buster-ripper/.venv/lib/python3.13/site-packages/aiohttp/web_app.py:492-511` (`Application._make_request` → `client_max_size=self._client_max_size`), corroborated by `aiohttp/web_runner.py:389-396` (`AppRunner._make_server` → `self._app._make_handler`), `aiohttp/web_app.py:442-447` (`Server(..., request_factory=self._make_request)`), `aiohttp/web_protocol.py:206, 587-593` (`RequestHandler` calls the manager's factory per request), `aiohttp/web_request.py:161, 198, 285-286, 675-680, 722` (storage + enforcement), and `aiohttp/web_app.py:128, 175` (the `client_max_size` config point). Package documentation is not shipped in site-packages, so the installed source is the primary source here.

## Source Version
aiohttp 3.13.3 (`aiohttp-3.13.3.dist-info/METADATA`, `Version: 3.13.3`); multidict was not consulted — the claim does not touch it.

## Source Fingerprint
sha256:94653f68030dfa1df0cbe2d34c78ba49e287f3276dd46e75057702b29803e590 aiohttp/web_app.py
sha256:2aaacea7a01e581e5eead2ab1b9e4ba3b65bc2ae3d0f176b2a78ae5b6b76bb4e aiohttp/web_request.py
sha256:bf51b59ca88e3908059d3491e0831ce88f5178414331a1ed30bbcefeba0333e0 aiohttp/web_runner.py
sha256:eacf5d33399aa96efb6f33354f5d75b863522c5a3a80d99f0e0ed7cd89c0f319 aiohttp/web_protocol.py
sha256:fbd58329402247d965fab49dc174aa1ba6236a85bbf5dd51e32d01192aa050c0 aiohttp/web_server.py

## Invalidates On
- Any aiohttp release that changes `Application._make_request` to look the limit up per request (e.g. from `request.app` / `match_info.current_app`) instead of `self._client_max_size`, or that adds a per-route/per-connection `client_max_size` override.
- A change to `AppRunner._make_server` / `Application._make_handler` that stops binding `request_factory=self._make_request` to the runner's own app (e.g. a router-selected or sub-app-selected factory).
- A change in `BaseRequest.__init__`/`read()`/`post()` that drops or relocates the `_client_max_size` gate, or extends it to `request.content`.
- Replacing the installed aiohttp version (fingerprint above) — re-check against the new bytes.

## Checked At
2026-10-06T23:05:55Z

"""PR #1 R3 — the shared session carries no cookie from one caller to the next.

``server.py`` builds the package's only ``ClientSession``. aiohttp's default
``CookieJar`` stores every upstream ``Set-Cookie`` on that session, so the cookie
reaches the *next* request — an unrelated caller's, or the ``/metrics`` poll's.
Base built a client per request (``buster_ripper_legacy.py:668/704/847/906``), so
no cookie outlived a request there.

``DummyCookieJar`` restores that without abandoning the one session, one
connector posture (``PI:T006/upstream_tls/optout_is_process_wide``). The
``Set-Cookie`` still travels downstream through ``response_headers``; only the
store is gone, so a caller's own ``Cookie`` is forwarded untouched rather than
replaced by a stored value of the same name.

Each test talks to a real upstream over a real socket, through the real app, with
``raw_proxy``: a test-client session has a jar of its own and would rewrite the
very header this module is about.
"""

import asyncio
from contextlib import suppress

import aiohttp
import pytest
from aiohttp import web

from buster_ripper import config
from buster_ripper.server import start_kv_cache_monitoring

# Path=/ so the cookie a /v1/models answer sets is one the /metrics poll would
# also match — otherwise the jar's replay is invisible to that poll.
_SET_COOKIES = ("affinity=1; Path=/", "caller=stored; Path=/")
_CLOSE = ("Connection", "close")
_WAIT = 2.0


class _Upstream:
    """A real upstream that records the raw ``Cookie`` header it receives."""

    def __init__(self):
        self.seen = {}
        self.metrics_hit = asyncio.Event()

    async def models(self, request: web.Request) -> web.Response:
        self.seen.setdefault("/v1/models", []).append(request.headers.get("Cookie"))
        return web.Response(
            text='{"data": []}',
            content_type="application/json",
            headers=[("Set-Cookie", value) for value in _SET_COOKIES],
        )

    async def metrics(self, request: web.Request) -> web.Response:
        self.seen.setdefault("/metrics", []).append(request.headers.get("Cookie"))
        self.metrics_hit.set()
        return web.Response(text="vllm:kv_cache_usage_perc 0.5\n")


@pytest.fixture
async def upstream(mocker):
    """A real upstream on loopback, and ``config.UPSTREAM`` pointed at it."""
    stub = _Upstream()
    app = web.Application()
    app.router.add_get("/v1/models", stub.models)
    app.router.add_get("/metrics", stub.metrics)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    _, port = runner.addresses[0][:2]
    # A hostname, not the bare IP: aiohttp's CookieJar refuses cookies for an IP
    # host unless it was built `unsafe=True`, and then the defect cannot show.
    mocker.patch.object(config, "UPSTREAM", f"http://localhost:{port}")
    mocker.patch.object(config, "KV_POLL_INTERVAL", 3600)
    try:
        yield stub
    finally:
        await runner.cleanup()


async def _probe(raw_proxy, cookie=None):
    """One caller's /v1/models request, with its own Cookie header if it has one."""
    headers = [_CLOSE] + ([("Cookie", cookie)] if cookie is not None else [])
    return await raw_proxy.send("/v1/models", headers=headers, method="GET")


class TestNoCookieOutlivesItsCaller:
    """The cookie belongs to the caller that earned it, and to no one else."""

    async def test_a_later_caller_sends_no_cookie(self, raw_proxy, upstream):
        """Caller A's answer sets the cookie; caller B's request must not carry it.

        The warm-up is what a real first caller produces: before the fix the jar
        replays it on the very next request, whatever caller makes it.
        """
        await _probe(raw_proxy)

        await _probe(raw_proxy)

        assert upstream.seen["/v1/models"] == [None, None]

    async def test_a_callers_own_cookie_is_forwarded_untouched(self, raw_proxy, upstream):
        """A stored cookie of the same name must not replace the caller's own value,
        and the upstream's Set-Cookie must still reach that caller downstream."""
        await _probe(raw_proxy)

        answer = await _probe(raw_proxy, cookie="caller=own")

        assert upstream.seen["/v1/models"][-1] == "caller=own", (
            "the jar's stored value replaced the caller's own Cookie header")
        for value in _SET_COOKIES:
            assert f"Set-Cookie: {value}".encode() in answer.raw, (
                f"the relay dropped Set-Cookie: {value}")

    async def test_the_metrics_poll_shares_the_cookie_free_session(self, raw_proxy, upstream):
        """One session for the probe and the poll, and its jar stores nothing: a
        Set-Cookie earned by a caller must not travel on the poll either."""
        app = raw_proxy.app
        await _probe(raw_proxy)

        task = asyncio.create_task(start_kv_cache_monitoring(app))
        try:
            await asyncio.wait_for(upstream.metrics_hit.wait(), timeout=_WAIT)
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

        assert upstream.seen["/metrics"] == [None]
        assert isinstance(app["session"].cookie_jar, aiohttp.DummyCookieJar)

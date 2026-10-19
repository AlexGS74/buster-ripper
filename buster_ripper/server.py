"""aiohttp application factory for buster-ripper."""

import asyncio
import logging

import aiohttp
from aiohttp import web

from . import backend, compaction, config, dump, image


log = logging.getLogger("buster-ripper")


async def start_kv_cache_monitoring(app: web.Application) -> None:
    """Start periodic KV cache monitoring. Runs for app lifetime.

    Delegates to compaction.poll_kv_cache() which polls /metrics
    every KV_POLL_INTERVAL seconds to track cache utilization.
    """
    session = app["session"]
    await compaction.poll_kv_cache(session)


def spawn_kv_cache_monitor(app: web.Application) -> None:
    """Spawn background task for periodic KV cache monitoring.

    Non-blocking: creates task and returns immediately.
    The monitor polls upstream /metrics every KV_POLL_INTERVAL seconds
    to track cache utilization for compaction heuristics.
    Task is stored in app state for cleanup on shutdown.
    """
    task = asyncio.create_task(start_kv_cache_monitoring(app))
    app["kv_monitor_task"] = task


def spawn_dump_flush(app: web.Application) -> None:
    """Spawn background task for periodic dump flushing.

    Non-blocking: creates task and returns immediately.
    The task flushes buffered dump records every DUMP_FLUSH_INTERVAL
    seconds and is stored in app state ("dump_flush_task") for
    cancel-and-await in on_cleanup, ahead of the shutdown flush.
    """
    app["dump_flush_task"] = asyncio.create_task(dump.poll_flush())


def spawn_image_expiry(app: web.Application) -> None:
    """Spawn background task for periodic image-cache expiry.

    Non-blocking: creates task and returns immediately.
    The task drops cached images older than IMAGE_CACHE_TTL seconds even
    while no image-bearing request arrives, and is stored in app state
    ("image_expiry_task") for cancel-and-await in on_cleanup.
    """
    app["image_expiry_task"] = asyncio.create_task(image.poll_expiry())


@web.middleware
async def logging_middleware(request: web.Request, handler) -> web.Response:
    """Log all requests."""
    log.info("%s %s", request.method, request.path)
    return await handler(request)


async def on_startup(app: web.Application) -> None:
    """Initialize app state on startup.

    The package's only ClientSession is built here on an explicit
    TCPConnector carrying the process-wide upstream TLS posture
    (verified unless --skip-tls-verify): one connector, one posture,
    for every outbound call. The connector must be constructed inside
    a running loop; the session adopts its loop.

    The jar is a DummyCookieJar: a cookie earned by one caller must not be
    replayed to the next, and this session is shared by every caller and poller.
    """
    app["log"] = logging.getLogger("buster-ripper")
    connector = aiohttp.TCPConnector(ssl=not config.SKIP_TLS_VERIFY)
    app["session"] = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=600), connector=connector,
        cookie_jar=aiohttp.DummyCookieJar(),
    )

    # Detect backend type (SGLang vs vLLM)
    detected = await backend.detect_backend(
        session=app["session"],
        backend_url=config.UPSTREAM,
        cli_override=config.BACKEND_TYPE_OVERRIDE,
    )

    # Re-probe in the background until the upstream answers definitively
    if detected == "unknown" and not backend.is_backend_state_cached():
        app["backend_retry_task"] = asyncio.create_task(
            backend._retry_detect_backend(app["session"], config.UPSTREAM)
        )

    # Spawn background KV cache monitor (non-blocking)
    spawn_kv_cache_monitor(app)

    # Spawn periodic dump flush (non-blocking)
    spawn_dump_flush(app)

    # Spawn periodic image-cache expiry (non-blocking)
    spawn_image_expiry(app)


async def on_cleanup(app: web.Application) -> None:
    """Clean up app resources on shutdown."""
    await _cancel_background_task(app, "backend_retry_task")
    await _cancel_background_task(app, "kv_monitor_task")
    await _cancel_background_task(app, "dump_flush_task")
    await _cancel_background_task(app, "image_expiry_task")
    _final_dump_flush()
    await app["session"].close()


async def _cancel_background_task(app: web.Application, key: str) -> None:
    """Cancel and await a task; isolate its ordinary failure during teardown."""
    if key not in app:
        return
    task = app[key]
    cleanup = asyncio.current_task()
    cancellation_count = cleanup.cancelling()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        if cleanup.cancelling() > cancellation_count:
            raise
    except Exception:
        log.exception("background task %s failed before cleanup", key)


def _final_dump_flush() -> None:
    """Shutdown flush of buffered dump records; cannot raise into cleanup."""
    try:
        dump.flush_all()
    except Exception:
        log.exception("dump: shutdown flush failed")


def create_app() -> web.Application:
    """Create aiohttp application."""
    # client_max_size=0 lifts the request-body cap: aiohttp's default is 1 MiB
    # (web_app.py:128) and its reader skips the check on a falsy value
    # (web_request.py:675), so no body is refused for being large. Base was
    # Starlette/uvicorn and imposed none, and the check sits at the handler's
    # first read - ahead of normalize() and --strip-images - so any number here
    # would reject bodies the proxy goes on to forward far smaller.
    app = web.Application(middlewares=[logging_middleware], client_max_size=0)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    # Import routes here to avoid circular imports
    from .routes import messages, chat, count_tokens, passthrough

    # Register routes
    app.router.add_post("/v1/messages", messages.handle_messages)
    app.router.add_post("/v1/chat/completions", chat.handle_chat)
    app.router.add_post("/v1/messages/count_tokens", count_tokens.handle_count_tokens)

    # Catch-all for passthrough (must be last)
    app.router.add_route(
        "*",
        "/{path:.*}",
        passthrough.handle_passthrough,
    )

    return app

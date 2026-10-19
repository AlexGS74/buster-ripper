"""Shared fixtures for buster-ripper tests.

`proxy` and `raw_proxy` both serve the real app with backend detection and the
image flag neutralised; they differ in how the test talks to it. `proxy` goes
through an aiohttp client, which is what a well-behaved caller does. `raw_proxy` is
for the two things a client cannot express: a request that carries the same header
name twice with different spellings (the aiohttp client merges those before the
wire), and a response that has to be read as bytes because the claim is about how
the response *ends*.

The upstream stand-in lives in `tests/upstream_stub.py` and is imported from there,
the way the QA gates import `tests.qa.harness` — a fixture for it would only add a
parameter to every test that uses one.
"""

import asyncio
from collections import namedtuple
from contextlib import suppress

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from buster_ripper import backend, config
from buster_ripper.server import create_app

RawResponse = namedtuple("RawResponse", "raw closed")

# A response that neither terminates nor closes is a defect, not a wait: this bound
# only has to be longer than a healthy round trip.
QUIET = 2.0

_TERMINATED = b"\r\n0\r\n\r\n"


def _app_with_mocked_backend(mocker):
    """The real app, with backend detection and the image flag neutralised."""
    app = create_app()
    mocker.patch.object(backend, "detect_backend", return_value=None)
    mocker.patch.object(backend, "get_backend_state", return_value="vllm")
    mocker.patch.object(config, "STRIP_IMAGES", False)
    return app


@pytest.fixture
async def proxy(mocker):
    """Test client for the real app; the upstream call is the test's to patch."""
    async with TestServer(_app_with_mocked_backend(mocker)) as server:
        async with TestClient(server) as client:
            yield client


class RawProxy:
    """The real app on a real socket, with the upstream call left to the test."""

    def __init__(self, app, port):
        self.app = app
        self._port = port

    async def send(self, path, *, headers=(), body=b"", method="POST"):
        """Write exact request bytes and return the exact response bytes.

        The read ends at the chunked terminator, at EOF, or once the peer goes
        quiet for QUIET seconds — so a stream that never terminates is reported as
        an unterminated response instead of hanging the suite. Ask for
        ``("Connection", "close")`` in `headers` when the claim is not about how the
        stream ends: a non-streaming answer on a kept-alive connection otherwise
        only finishes when the window expires.
        """
        head = f"{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{self._port}\r\n"
        for name, value in headers:
            head += f"{name}: {value}\r\n"
        if body:
            head += "Content-Type: application/json\r\n"
        head += f"Content-Length: {len(body)}\r\n\r\n"

        reader, writer = await asyncio.open_connection("127.0.0.1", self._port)
        pieces, closed = [], False
        try:
            writer.write(head.encode() + body)
            await writer.drain()
            while True:
                piece = await _read_or_quiet(reader)
                if piece is None:
                    break
                if not piece:
                    closed = True
                    break
                pieces.append(piece)
                if _TERMINATED in b"".join(pieces):
                    break
        finally:
            writer.close()
            with suppress(Exception):
                await writer.wait_closed()
        return RawResponse(b"".join(pieces), closed)


async def _read_or_quiet(reader):
    """Next chunk of the response: b"" for EOF, None when the peer went silent."""
    try:
        return await asyncio.wait_for(reader.read(65536), timeout=QUIET)
    except (asyncio.TimeoutError, TimeoutError):
        return None


@pytest.fixture
async def raw_proxy(mocker):
    """Serve the real app on a loopback port; the upstream call is the test's to patch."""
    app = _app_with_mocked_backend(mocker)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    try:
        yield RawProxy(app, runner.addresses[0][1])
    finally:
        await runner.cleanup()

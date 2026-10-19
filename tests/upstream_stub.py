"""Stand-in for the shared ``aiohttp.ClientSession``'s answer, so no test needs a
real upstream.

Mirrors the shape the route handlers actually read: ``status``, ``headers``, the
``read()/text()/json()`` coroutines, and — for the streaming relay —
``content.iter_any()``. Import it (``from tests.upstream_stub import ...``) the way
the QA gates import ``tests.qa.harness``; do not wrap it in a fixture, which only
costs every test an extra parameter.

``StubUpstream`` answers a body. ``StubStream`` answers chunks and can die partway
through them, which is how a post-commit relay failure is produced without
monkeypatching anything.
"""

from multidict import CIMultiDict

DEFAULT_BODY = b"upstream-body"
DEFAULT_JSON = {"id": "m", "content": [{"type": "text", "text": "hi"}]}


class StubContent:
    """ClientResponse.content: the relay loop pulls chunks from here."""

    def __init__(self, chunks, fault=None):
        self._chunks = list(chunks)
        self._fault = fault

    def iter_any(self):
        return self._iter()

    async def _iter(self):
        for chunk in self._chunks:
            yield chunk
        if self._fault is not None:
            raise self._fault


class StubUpstream:
    """A finished upstream answer: status, headers, and a whole body."""

    def __init__(self, *, status=200, headers=None, body=DEFAULT_BODY, json_body=None):
        self.status = status
        self.headers = CIMultiDict(headers or {})
        self._body = body
        self._json = DEFAULT_JSON if json_body is None else json_body
        self.content = StubContent([body])

    async def read(self):
        return self._body

    async def text(self):
        return self._body.decode()

    async def json(self):
        return self._json

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False


class StubStream(StubUpstream):
    """An upstream that streams `chunks` and then optionally raises `fault`.

    With ``fault=None`` and one chunk this is just a successful relay. With a fault
    it is a post-commit failure at the same position in the same `async for`, and
    with a payload that the handler itself trips over, ``fault`` is not needed at
    all — the chunk is the bug.
    """

    def __init__(self, *, headers=None, chunks=(), fault=None):
        super().__init__(headers=headers)
        self.content = StubContent(chunks, fault)

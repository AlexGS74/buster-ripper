"""FastAPI application factory."""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from . import compaction
from .routes import chat, count_tokens, messages, passthrough


@asynccontextmanager
async def _lifespan(app: FastAPI):
    task = asyncio.create_task(compaction.poll_kv_cache())
    yield
    task.cancel()


app = FastAPI(lifespan=_lifespan)

app.include_router(messages.router)
app.include_router(chat.router)
app.include_router(count_tokens.router)
app.include_router(passthrough.router)  # must be last — catches all remaining paths

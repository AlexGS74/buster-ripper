VERDICT: HOLDS
PROMOTION: INELIGIBLE — verdict is HOLDS from an immutable tag-pinned upstream locator, but no `sha256:` of the exact consulted bytes is available: the source was read via web fetch and the fetched bytes were not persisted as a hashable artifact (a re-transcription attempt was checked and found lossy, so it was discarded rather than hashed). Reuse needs a supplied `Dependency source roots:` package directory so the exact file bytes can be hashed locally.

## Claim
aiohttp's `web.StreamResponse.write()` signals a closed client connection by raising an exception derived from `Exception`

## What it actually does
`StreamResponse.write()` delegates to the response's `StreamWriter`, whose `_write()` / `_writelines()` raise `ClientConnectionResetError("Cannot write to closing transport")` when `self._protocol.transport is None or transport.is_closing()`. That class is declared `class ClientConnectionResetError(ClientConnectionError, ConnectionResetError)` with `ClientConnectionError(ClientError)` / `ClientError(Exception)` — so the object that propagates out of `await resp.write(...)` is an `OSError`/`ConnectionError`/`Exception` subclass, not a `BaseException`-only signal such as `asyncio.CancelledError`.

## Source
Upstream source at the pinned tag `v3.13.3` (single source basis: this tag's `aiohttp/` package tree), consulted files:
- `aiohttp/http_writer.py` — `StreamWriter._write()` and `StreamWriter._writelines()`: the `transport is None or transport.is_closing()` guard and the `raise ClientConnectionResetError("Cannot write to closing transport")`; `StreamWriter.write()` reaches them via `self._write(chunk)` / `self._write_chunked_payload(chunk)`.
- `aiohttp/web_response.py` — `StreamResponse.write()` body is `await self._payload_writer.write(data)` (it adds no disconnect handling of its own; its only own raises are `RuntimeError` for before-`prepare()`/after-`write_eof()`).
- `aiohttp/client_exceptions.py` — `ClientError(Exception)` → `ClientConnectionError(ClientError)` → `ClientConnectionResetError(ClientConnectionError, ConnectionResetError)`, establishing derivation from `Exception`.

Corroboration (same tag, not the basis of the verdict): `aiohttp/web_protocol.py` — `RequestHandler.start()` catches `ConnectionError` around the handler task ("Ignored premature client disconnection") and `finish_response()` catches `ConnectionError` and returns `reset=True`, i.e. aiohttp's own machinery expects the disconnect signal to arrive as a `ConnectionError`, which is `Exception`-derived.

Docs check: `https://docs.aiohttp.org/en/stable/web_reference.html#aiohttp.web.StreamResponse.write` documents only `TypeError` (non-byte-ish data) and `RuntimeError` (`prepare()` not called / `write_eof()` already called) — the official reference is **silent** on client-disconnect signalling, which is why the verdict is grounded in the tag-pinned source rather than the docs.

## Source Version
aiohttp 3.13.3 (the version this repo pins: `uv.lock` entry `name = "aiohttp" / version = "3.13.3"`; `pyproject.toml` requires bare `aiohttp`). Upstream files read at git tag `v3.13.3`. The HTML reference page read was `stable` (3.14.x-era) and is used only to show docs silence.

## Source Fingerprint
unavailable — see `PROMOTION:` line. (Package-artifact digests exist in `uv.lock` for aiohttp 3.13.3, but they hash the wheel/sdist, not the exact bytes consulted, so they are not recorded as this verdict's fingerprint.)

## Invalidates On
- Any aiohttp version change where `aiohttp/http_writer.py` no longer raises on `transport is None or transport.is_closing()` (e.g. a release that reverts to silently buffering/discarding writes to a closed transport) — re-check `StreamWriter._write`/`_writelines` in the new tag.
- A change to `ClientConnectionResetError`'s bases in `aiohttp/client_exceptions.py`, or a new disconnect-specific exception that derives only from `BaseException` (e.g. surfacing `asyncio.CancelledError` from the write path).
- A change to `StreamResponse.write()` in `aiohttp/web_response.py` that stops delegating to `_payload_writer.write()` (e.g. routing through a new writer abstraction with different error semantics).
- Reading the claim as "every write after a peer close raises": aiohttp only signals once the loop knows the transport is closing/closed — a client that has just sent FIN can still have a write accepted into the buffer, and `StreamWriter.write()` returns early without touching the transport when compression yields an empty chunk, when `length` has been exhausted, or when `chunk` is empty. This is a detection-timing caveat, not a different signalling type; it does not change the verdict.

## Checked At
2026-10-05T18:25:44Z

## Notes
- Single-claim bundle; nothing here required plan or session context, and no session file other than the resolved `assumptions/verdicts/` output directory was read. No bundling defect observable.
- Grounding is production/library source (P10 respected): no test, fixture, or QA artifact was used. No project code was executed (P1).

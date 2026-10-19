VERDICT: HOLDS
PROMOTION: ELIGIBLE — holds verdict, one exact tagged source locator, sha256 of consulted bytes, concrete invalidation conditions below

## Claim
aiohttp's `web.StreamResponse.prepare()` returns without writing a head when that same response object is already prepared

## What it actually does
`prepare()` opens with two guard clauses that return before any header work: `if self._eof_sent: return None`, then `if self._payload_writer is not None: return self._payload_writer`. Those two conditions are exactly what `prepared` is defined as (`return self._eof_sent or self._payload_writer is not None`), so for an already-prepared response object `prepare()` short-circuits — no `_start()`, no `_prepare_headers()`, no `on_response_prepare` hook, no `write_headers()`/`send_headers()`. It is an idempotent no-op that returns the existing writer (or `None` once `write_eof()` has run), and it does not raise.

## Evidence
`aiohttp/web_response.py` @ v3.13.3:

- L454-460 `prepare()`:
  ```python
  async def prepare(self, request: "BaseRequest") -> Optional[AbstractStreamWriter]:
      if self._eof_sent:
          return None
      if self._payload_writer is not None:
          return self._payload_writer
      self._must_be_empty_body = must_be_empty_body(request.method, self.status)
      return await self._start(request)
  ```
  Both head-writing paths (`_start` → `_write_headers`, L462-470 / L537-548) sit *below* the guards, so neither guard's result can write a head.
- L121-122 `prepared` — `return self._eof_sent or self._payload_writer is not None`: the claim's "already prepared" is precisely the union of the two guard conditions, so there is no prepared state that falls through to `_start()`.
- L87 class default `_payload_writer: Optional[AbstractStreamWriter] = None` — supports "first call" vs "same object" bookkeeping.
- L562-587 `write_eof()` — `self._eof_sent = True` … `self._payload_writer = None`, i.e. after EOF the first guard is the one that fires and the return value is `None`, still without writing a head.
- Request identity is not part of the guard: a *different* `request` passed to a second `prepare()` on the same response is likewise ignored (no `_req` rebind, no second head).
- L462-470 `_start()` sets `self._req = request` and `self._payload_writer = request._payload_writer` — the only place the writer is bound, and it is reached solely from the non-guarded path.

Corroborating documentation (same version): `web_reference.html` @ v3.13.3, `StreamResponse.prepared` — "Read-only bool property, True if prepare() has been called, False otherwise." The docs do not state the second-call behaviour; the verdict is grounded in the source above, not in the docs.

## Source
https://raw.githubusercontent.com/aio-libs/aiohttp/v3.13.3/aiohttp/web_response.py (L87, L121-122, L454-470, L537-548, L562-587)

## Source Version
aiohttp 3.13.3 (upstream tag `v3.13.3`; matches the version pinned in this repo's `uv.lock`)

## Source Fingerprint
sha256:3ca73388d5382e67ed5ea295be844cac06ce7950a5a5237e8b39c7b2259ece65 (web_response.py)

## Invalidates On
- Any change to the guard order/conditions in `prepare()` (L455-458) — e.g. replacing them with a `RuntimeError` on re-prepare, or moving `_must_be_empty_body`/`_start()` above the checks.
- A change to `prepared` (L121-122) that makes it true in a state where `_payload_writer` is still None and `_eof_sent` is still False (that would let a "prepared" object fall through to `_start()` and write a second head).
- Any code path that assigns `_payload_writer` without writing the head, which would make the second guard return "prepared" for an object whose head was never sent.
- aiohttp version bump whose `web_response.py` fingerprint differs from `3ca73388…` above.

## Checked At
2026-10-05T18:18:15Z

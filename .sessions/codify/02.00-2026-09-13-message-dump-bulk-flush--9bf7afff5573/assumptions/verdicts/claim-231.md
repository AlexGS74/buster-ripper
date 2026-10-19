VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`aiohttp.ClientPayloadError` is a subclass of `aiohttp.ClientError`

## What it actually does
In aiohttp 3.13.3 `ClientPayloadError` derives **directly** from `ClientError`
(`class ClientPayloadError(ClientError):`) — it is not routed through
`ClientConnectionError`/`ClientOSError`. So `except aiohttp.ClientError` catches a
payload error, but `except aiohttp.ClientConnectionError` does **not**, and
`ClientPayloadError` is not an `OSError`. The same direct-parent relationship holds
in the older 3.8.x line, so the claim is not version-fragile within aiohttp 3.x.

## Source
`https://raw.githubusercontent.com/aio-libs/aiohttp/v3.13.3/aiohttp/client_exceptions.py`
line 297: `class ClientPayloadError(ClientError):` (module exports it, and
`aiohttp/__init__.py` re-exports the `client_exceptions` names, so
`aiohttp.ClientPayloadError` / `aiohttp.ClientError` are the same objects).
Corroborated by the library's own docs for the installed version:
`https://docs.aiohttp.org/en/v3.13.3/client_reference.html` — under
`exception aiohttp.ClientPayloadError`: "Derived from ClientError", and the
"Hierarchy of exceptions" tree lists `ClientPayloadError` nested under `ClientError`.
Claim is about a library; no repo source and no session artifact was consulted.

## Source Version
aiohttp 3.13.3 (tag `v3.13.3` upstream; the version the repo lockfile resolves).

## Source Fingerprint
sha256:bb229bc48da979984a97b9442c1331dd47aeb0d91fa5e98f5a9185ab4afa25e3
(corroborating doc page: sha256:a56614286d9c86b6ca347057247299a7740d1ab6b417178ca46d6467a1d77fc6)

## Invalidates On
- An aiohttp release that changes the declaration at `aiohttp/client_exceptions.py:297`
  (re-parenting `ClientPayloadError`, e.g. under `ClientConnectionError`, or detaching
  it from `ClientError`).
- Checking a different installed aiohttp version than 3.13.3 — re-read
  `client_exceptions.py` for that version before reusing.
- An upstream 4.0 release that restructures the client exception module.

## Checked At
2026-10-05T18:25:22Z

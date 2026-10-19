VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
aiohttp's client session supplies a `Host` request header when the caller passes none

## What it actually does
`ClientRequest.update_headers()` — the request object `ClientSession._request()`
builds for every call — sets `self.headers[hdrs.HOST]` **unconditionally** to the
URL's `host_port_subcomponent` (host plus non-default port) *before* it merges any
caller-supplied headers, then replaces that value only if the caller did pass a
`Host`/`host` key (`hdrs.HOST_ALL`). So with no caller `Host`, the request carries
the URL-derived `Host`. The header is not filtered at send time:
`StreamWriter.write_headers()` serializes every entry of `self.headers`, and
`skip_auto_headers` has no effect on `Host` (it is consulted only for
`DEFAULT_HEADERS`, `User-Agent`, and payload-copied headers in
`update_auto_headers()` / `update_body_from_data()`). The only way `Host` is absent
is a URL with no host at all, which raises `InvalidURL` in `update_host()` before
the request is written.

## Source
`aiohttp/client_reqrep.py` @ tag `v3.13.3`,
`ClientRequest.update_headers` (lines 1056-1079); construction path
`ClientRequest.__init__` line 894-896; session path `aiohttp/client.py` @ `v3.13.3`
line 702 (`req = self._request_class(..., headers=headers,
skip_auto_headers=skip_headers, ...)`); send path `aiohttp/http_writer.py` @
`v3.13.3` line 217 + `_py_serialize_headers` (lines 363-369) and
`aiohttp/_http_writer.pyx` `_serialize_headers` (line 125) — both write every
header without omitting `Host`. `hdrs.HOST` / `hdrs.HOST_ALL` from
`aiohttp/hdrs.py` @ `v3.13.3` lines 69, 119-120.

Source basis: upstream source fetched at the exact version tag pinned in this
project's `uv.lock` (`aiohttp 3.13.3`). No local `Source root` /
`Dependency source roots` was supplied with the dispatch, so the installed copy
was not read; the fetched files are the consulted bytes.

## Source Version
aiohttp 3.13.3 (`v3.13.3` tag; matches the pin in `uv.lock`)

## Source Fingerprint
sha256:7844440eb67433c6454edd708c776e47e3fcfec9b8d0aeb980dcfe88c1986ac9 (client_reqrep.py)
sha256:7ce41fc1c2142f5346015455e030e3ebec22a41cde0fc299a66ce13be2cb2a9e (client.py)
sha256:7db46d28f612a916ed01daff82aa63176fb8b08d4448bf1d3c317ec58fe600c6 (http_writer.py)
sha256:dab8f933203eeb245d60f856e542a45b888d5a110094620e4811f90f816628d1 (hdrs.py)

## Invalidates On
- Any aiohttp version where `ClientRequest.update_headers` stops assigning
  `hdrs.HOST` before merging caller headers (e.g. it becomes conditional on
  `skip_auto_headers` or on the caller's header set).
- Any version where `_serialize_headers` (Python or Cython writer) omits or
  suppresses `Host` from the wire bytes.
- Any version that changes `url.host_port_subcomponent` semantics for the
  default-port / IPv6 / IDNA cases, since that value is the header's content.
- A pinned aiohttp version other than 3.13.3 without re-reading these files.

## Checked At
2026-10-05T14:25:00+10:00

VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
aiohttp's `web.Response` sets `Content-Length` from its body when it sends the response

## What it actually does
`Response._start()` (reached from `StreamResponse.prepare()` → `_start()` →
`_prepare_headers()` → `_write_headers()`, i.e. on the send path, just before headers are
written to the stream) computes and inserts `Content-Length` itself: `len(self._body)` for a
bytes/bytearray body, `"0"` when the body is empty/None, or `self._body.size` when the body is
a `Payload` that reports a size. Nothing sets `Content-Length` in `Response.__init__`, and the
`content_length` setter on `Response` raises `RuntimeError("Content length is set automatically")`
— the value is derived, not authored. Documented carve-outs, all inside the same `_start` /
compression / header paths: an explicit `Content-Length` already present in the response headers
wins (the `elif not self._chunked` branch is skipped); `enable_chunked_encoding()` uses
`Transfer-Encoding: chunked` instead and is mutually exclusive with a set content length;
compressed bodies get `Content-Length: len(_compressed_body)` (line 830), and bodies that must be
empty for the method/status (`HEAD`, `304`, etc. — `should_remove_content_length` /
`must_be_empty_body`) have the header removed again.

So the claim is true as stated, including its timing ("when it sends"): construction does not
set the header, sending does.

## Source
`aiohttp/web_response.py` @ tag `v3.13.3`, `class Response`:
`_start` lines 788–804, `content_length` property/setter lines 746–767,
`_do_start_compression` line 830; send-path confirmation in `class StreamResponse`:
`prepare` lines 454–460, `_start` lines 462–470, `_prepare_headers` lines 490–525,
`enable_chunked_encoding` lines 182–190.
Locator: https://raw.githubusercontent.com/aio-libs/aiohttp/v3.13.3/aiohttp/web_response.py
(no `Source root` / `Dependency source roots` locator was supplied with this dispatch, so the
pinned version's published source was fetched by exact URL rather than read from an installed
package directory; version read from the repo's committed `uv.lock`).

## Source Version
aiohttp 3.13.3 (pinned in `uv.lock`: `name = "aiohttp"`, `version = "3.13.3"`;
`pyproject.toml` declares bare `"aiohttp"`)

## Source Fingerprint
sha256:3ca73388d5382e67ed5ea295be844cac06ce7950a5a5237e8b39c7b2259ece65

## Invalidates On
- Any change to `Response._start` in `aiohttp/web_response.py` (the `elif not self._chunked`
  content-length block) — e.g. a release moving header derivation to construction time or
  removing the `CONTENT_LENGTH in self._headers` precedence check.
- A release changing `Response.content_length` setter semantics away from
  `RuntimeError("Content length is set automatically")`.
- A release altering `_do_start_compression`'s `CONTENT_LENGTH = len(_compressed_body)` line or
  `should_remove_content_length` / `must_be_empty_body` behaviour for HEAD/304.
- Consulted file bytes no longer matching `sha256:3ca73388…5ece65` (fetched source changed or a
  different aiohttp version installed than 3.13.3).

## Checked At
2026-10-05

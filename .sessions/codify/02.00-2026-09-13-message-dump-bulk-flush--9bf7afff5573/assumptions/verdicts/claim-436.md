VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`urllib.request.urlopen`'s `timeout` argument bounds the wait for a peer that never answers

## What it actually does
`urlopen` passes `timeout` to `OpenerDirector.open`, which sets `req.timeout`; `AbstractHTTPHandler.do_open` constructs the connection as `http_class(host, timeout=req.timeout, ...)`, `HTTPConnection` forwards `self.timeout` to `socket.create_connection`, and `create_connection` calls `sock.settimeout(timeout)`. The socket then times out on every blocking operation, so reading the status line from a peer that sends nothing raises `TimeoutError` (`socket.timeout`) after at most that many seconds. Caveat: this is a per-socket-operation timeout, not a whole-request deadline — a peer that trickles bytes can extend the total wait indefinitely, and the DNS lookup inside `create_connection` is not covered by it.

## Source
https://docs.python.org/3/library/urllib.request.html — `urlopen`: "The optional *timeout* parameter specifies a timeout in seconds for blocking operations like the connection attempt (if not specified, the global default timeout setting will be used). This actually only works for HTTP, HTTPS and FTP connections."

Corroborated by the exact 3.11 source: `Lib/urllib/request.py` lines 139–152 and 500–509, 1330; `Lib/http/client.py` lines 890, 1010–1011, 275, 297; `Lib/socket.py` lines 843–845; and the `socket.settimeout` documentation ("subsequent socket operations will raise a timeout exception if the timeout period value has elapsed before the operation has completed").

## Source Version
docs.python.org/3 — Python 3.15.0 documentation (page last updated 2026-10-17); project declares `requires-python = ">=3.11"` in pyproject.toml, and the CPython 3.11 source docstring carries identical wording

## Source Fingerprint
sha256:256a5af44d3299f052f1833232633eea579a9cb3d4d4b314012e6fcadb24350d (fetched HTML of https://docs.python.org/3/library/urllib.request.html)

## Invalidates On
- The documented meaning of `urlopen`'s `timeout` parameter changes from "blocking operations" to connection-attempt-only
- `urlopen`/`OpenerDirector.open` stops propagating `timeout` to the `HTTPConnection` socket
- `socket.create_connection` stops applying `settimeout` to the socket it returns
- The project moves to a Python version whose urllib/socket timeout semantics differ from the documented ones

## Checked At
2026-10-17T23:47:04Z

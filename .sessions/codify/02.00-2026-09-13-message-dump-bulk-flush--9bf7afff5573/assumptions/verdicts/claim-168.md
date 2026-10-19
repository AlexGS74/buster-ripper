VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`pathlib.Path.write_text` performs its write by opening the target through `self.open(mode="w", …)`

## What it actually does
`pathlib.Path` is `pathlib._local.Path` (`pathlib/__init__.py:9`, `from ._local import *`). Its `write_text` (`_local.py:548-555`) normalizes the encoding for warning stack level, then delegates to `PathBase.write_text` (`_abc.py:644-652`), which performs the write exactly as claimed: `with self.open(mode='w', encoding=encoding, errors=errors, newline=newline) as f: return f.write(data)` (after a `TypeError` check that `data` is `str`). Since `self` is the concrete `Path`, `self.open` resolves to `_local.Path.open` (`_local.py:529-537`), which calls `io.open(self, mode, buffering, encoding, errors, newline)`. So the write is done by opening the target through `self.open(mode="w", …)`.

## Source
`/Users/alexhubitski/.local/share/uv/python/cpython-3.13.8-macos-aarch64-none/lib/python3.13/pathlib/_abc.py`, `PathBase.write_text`, lines 644-652 (the `self.open(mode='w', ...)` call site), reached from `pathlib/_local.py` lines 548-555 — installed source of the project's venv interpreter (`.venv/bin/python` → CPython 3.13.8).

## Source Version
CPython 3.13.8 (stdlib `pathlib`, uv-managed interpreter used by this repo's `.venv`)

## Source Fingerprint
sha256:75e6bc728013d11446126d535f92b8907fb6dae7404df3d0268ac7b6865a2090 (`pathlib/_abc.py`); corroborating delegate `pathlib/_local.py` sha256:0530b4186ec5845e000bedf5817ff1ac970b9a39a6891ba6789a1776abdea42b

## Invalidates On
- The venv's interpreter changes (e.g. `.venv` rebuilt against a different Python) — re-locate the active `pathlib` source and re-check `write_text`
- A Python upgrade where `write_text` stops routing through `self.open(mode='w', ...)` — e.g. a CPython change to `_abc.PathBase.write_text` or an override added in `_local.Path` (watch `_abc.py`/`_local.py` in any newer stdlib; fingerprint will differ)
- CPython modifies these stdlib files in place (fingerprint change in the installed interpreter)

## Checked At
2026-09-28T14:00:33Z

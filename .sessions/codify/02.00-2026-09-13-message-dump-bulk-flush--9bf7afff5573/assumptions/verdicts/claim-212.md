VERDICT: DOES-NOT-HOLD
PROMOTION: ELIGIBLE

## Claim
`buster_ripper/routes/chat.py` imports `forward_headers` without calling it

## What it actually does
`forward_headers` is both imported and called. The import is in the `from ..utils import (...)` block
at chat.py:14, and the call is the first statement of `_prepare_headers`:

```python
def _prepare_headers(request_headers) -> dict:
    headers = forward_headers(request_headers)      # chat.py:103
```

Its return value is the mutable dict that the rest of `_prepare_headers` then rewrites the
`Authorization` entry in (chat.py:117-121), and that dict is what `_make_chat_request` sends upstream as
`headers=headers` (chat.py:147). So the symbol is load-bearing, not a dead import: `_prepare_headers` is
called at chat.py:36 in `handle_chat`, which makes the `forward_headers` call reachable on every
`/v1/chat/completions` request.

`grep -n "forward_headers" chat.py` returns three hits in the consulted bytes: line 14 (the import),
line 100 (the name inside `_prepare_headers`' docstring prose), and line 103 (the call). Only one of
those is a call, and there is exactly one call site — but the claim as written ("without calling it") is
false regardless of call-site count.

## Source
/Users/alexhubitski/Projects/buster-ripper/buster_ripper/routes/chat.py — chat.py:11-17 (import block), chat.py:98-121 (`_prepare_headers`, call at line 103), chat.py:36 (caller)

## Source Version
Repo working source, 299 lines; no VCS revision consulted (Git metadata is out of scope for this role)

## Source Fingerprint
sha256:8da1d171930433fcec7f3657e3b668026dc78fe3657a6525f3df3c78c46388c5

## Invalidates On
- Any edit to `_prepare_headers` that stops calling `forward_headers` (e.g. building the header dict inline) while leaving the name in the `from ..utils import (...)` block — that would make the claim true
- Removing `forward_headers` from `buster_ripper/utils` or renaming it, which changes this from a dead-import question to an import error
- Any change that deletes the `_prepare_headers` call at chat.py:36, which would leave the import present but the call unreachable

## Checked At
2026-10-06T12:16:35Z

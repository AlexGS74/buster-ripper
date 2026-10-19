VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`buster_ripper_legacy.py`'s empty-`Bearer` deletion appears only inside its chat-completions handler

## What it actually does
The only executable empty-`Bearer` deletion in the module is the three-line block at
`buster_ripper_legacy.py:838-841` —
`auth = headers.get("authorization", "")` / `if auth.strip() in ("Bearer", "Bearer "):` /
`headers.pop("authorization", None)` — and it sits inside the body of `chat_completions`, the handler
bound to `@app.post("/v1/chat/completions")` (`:794-795`, function body runs to `:892`). It runs
unconditionally there (outside the `if EVAL_MODE` block), i.e. even for non-eval chat-completions
requests. No other header-affecting code path in the file deletes an authorization value:
`_forward_headers` (`:621-627`) drops only `host`, `content-length`, and the `_HOP_BY_HOP` set
(`:178-189`), which contains `proxy-authorization` but **not** `authorization`, and it does so regardless
of scheme or emptiness — that is a different rule, not the empty-`Bearer` deletion. `_response_headers`
(`:630-636`) filters upstream *response* headers only. A full case-insensitive sweep of the file for
`bearer|authorization` returns five hits: `:120` and `:801` are prose (the eval-strategy comment block
and the `chat_completions` docstring) describing the same behaviour, `:183` is `proxy-authorization` in
`_HOP_BY_HOP`, and `:838-841` is the code — one site, inside the chat-completions handler.
`proxy_messages` (`:639`) and `passthrough` (`:898`) forward the client's `authorization` untouched.

## Source
`buster_ripper_legacy.py:838-841` (deletion site) within `:794-892`; negative evidence from `:621-627`,
`:178-189`, `:630-636`; whole file (1013 lines) read — production source, no test evidence used

## Source Version
Repo working tree file `buster_ripper_legacy.py`, 42129 bytes, mtime 2026-03-03 09:50:07; standalone
script header (`requires-python >=3.11`, deps fastapi/uvicorn/httpx/click), no package version string

## Source Fingerprint
sha256:7967563dc3ab4f66f0dc322c7945c5647b96c9f9a77fbafe3cd1d4780e8764bd

## Invalidates On
- Any edit to `buster_ripper_legacy.py` (fingerprint changes) — re-check before reuse
- The `:838-841` block moves into a shared helper (e.g. `_forward_headers`) or is called from
  `proxy_messages`/`passthrough`, making a second handler delete it
- A new empty-`Bearer`/auth-stripping condition is added anywhere else in the file
- `_HOP_BY_HOP` (`:178-189`) gains `authorization`, which would make header dropping at `:621` overlap
  the claim's subject

## Checked At
2026-10-05T18:18:03Z

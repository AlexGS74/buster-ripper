VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`_relay_chat_stream` forwards successful upstream chunks from `resp.content.iter_any()` to `StreamResponse.write` unchanged

## What it actually does
For upstream status below 400, it iterates `resp.content.iter_any()` and calls `await response.write(chunk)` with each chunk directly; no decoding or transformation occurs in this function.

## Source
`buster_ripper/routes/chat.py:155-168`

## Source Version
unavailable

## Source Fingerprint
sha256:3068981c63ff3737b223b5f9bb9018a23195034afbc67df77e3c8ec37ff0a9ce

## Invalidates On
Any change to `_relay_chat_stream` in `buster_ripper/routes/chat.py`, especially its iteration or write path.

## Checked At
2026-09-27

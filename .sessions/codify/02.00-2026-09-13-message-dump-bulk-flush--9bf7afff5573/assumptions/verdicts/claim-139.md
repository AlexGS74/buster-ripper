VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`_make_chat_request` maps upstream response `UnicodeDecodeError` from `resp.json()` to a JSON 502 response

## What it actually does
`resp.json()` executes inside a try block whose `except (json.JSONDecodeError, UnicodeDecodeError)` returns `web.json_response({"error": "Invalid JSON from upstream"}, status=502)`.

## Source
`buster_ripper/routes/chat.py:114-149` (production source)

## Source Version
unavailable

## Source Fingerprint
sha256:e5323a69332745c7f766e617c63af98c9a906f4aa0e9dbbd1a0f229312d9f9f8

## Invalidates On
Change to `buster_ripper/routes/chat.py`, especially the `resp.json()` try/except or response construction.

## Checked At
2026-09-27

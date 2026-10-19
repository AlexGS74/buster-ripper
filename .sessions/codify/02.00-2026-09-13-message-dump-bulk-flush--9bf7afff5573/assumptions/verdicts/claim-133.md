VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`handle_chat` maps client `json.JSONDecodeError` from `request.json()` to a JSON 400 response

## What it actually does
The `request.json()` call is inside a try block that catches `json.JSONDecodeError` and returns `web.json_response({"error": "Invalid JSON"}, status=400)`.

## Source
`buster_ripper/routes/chat.py:23-31` (production source)

## Source Version
unavailable

## Source Fingerprint
sha256:e5323a69332745c7f766e617c63af98c9a906f4aa0e9dbbd1a0f229312d9f9f8

## Invalidates On
Any change to the client JSON parsing or exception handler in `buster_ripper/routes/chat.py`.

## Checked At
2026-09-27

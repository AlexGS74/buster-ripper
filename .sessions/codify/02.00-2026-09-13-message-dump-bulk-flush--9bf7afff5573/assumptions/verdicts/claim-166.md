VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`_WriteHook` measures each recorded write's payload length as its UTF-8 encoded byte count

## What it actually does
`_WriteHook.record` computes `size = len(data.encode("utf-8")) if isinstance(data, str) else len(data)` and stores it as the event's `byte_len` — for the string payloads the measured writers issue (`dump.py` writes via `write_text` / `open("a")` + `write` of `str`), this is exactly the payload's UTF-8 encoded byte count. The defensive non-`str` branch falls back to `len(data)`, which for a bytes payload is likewise its byte count.

## Source
tests/test_dump_write_count.py — `_WriteHook.record` (lines 110-113)

## Source Version
unavailable (working-tree repo source; git metadata out of scope for this role)

## Source Fingerprint
sha256:63b2c4c8e049f9462d4b44277cb1a3c387680da2820154a3e583d601c814822e

## Invalidates On
- Any edit to `_WriteHook.record`'s size computation (e.g., `len(data)` without encoding, or a non-UTF-8 encoding)
- Any change to the measured writers that makes them issue non-`str` payloads where the semantic difference between `len(bytes)` and UTF-8 encoding would matter
- Note: this is a claim about the test's instrument, so the test file itself is the production surface (P10 exception)

## Checked At
2026-09-28

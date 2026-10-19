VERDICT: HOLDS
PROMOTION: INELIGIBLE — the verdict is an absence over the whole package; no single exact file establishes it, and the consulted surface is a directory scan of `buster_ripper/` (broad directory), not one recoverable file's bytes.

## Claim
`utils.response_headers` is never called by any module under `buster_ripper/`

## What it actually does
`response_headers(upstream_headers)` (utils.py:27-33) strips the 8 hop-by-hop headers from an upstream response dict and returns the rest. It is never called and — unlike its sibling `forward_headers` — it is not even imported: the name appears exactly once in the package, at its own `def`. Each route module that returns upstream response headers to the client re-implements the filtering inline instead: `routes/messages.py:257-259` and `routes/passthrough.py:40-42` each build `resp_headers = {k: v for k, v in resp.headers.items() if ...}` locally, and `routes/chat.py:159-161` hand-picks `Content-Type` for its `web.StreamResponse`.

## Source
Exhaustive reference search over the bounded source root `/Users/alexhubitski/Projects/buster-ripper/buster_ripper/` (all files, `__pycache__` excluded; the only non-`.py` file is `.DS_Store`): `grep -rn "response_headers" buster_ripper/` returns exactly one hit —
- `buster_ripper/utils.py:27` — the `def` itself (definition, not a call)

No invocation (`response_headers(`) and no import of the name exists anywhere under `buster_ripper/`; the qualified form (`utils.response_headers`) is covered by the same substring search. No `from ... import *`, no `__all__` re-export, and no `getattr(`/`globals()`/`importlib` dispatch anywhere under the source root (grep returned nothing), so no name-based indirection could reach it. The two inline header-strips in `routes/messages.py` and `routes/passthrough.py` are local dict comprehensions, not calls to this helper — confirmed by reading those lines, which contain no reference to `utils`.

Checked independently of the sibling claim about `forward_headers`; the two searches are separate and neither result was inferred from the other.

## Source Version
local working tree, `buster_ripper/` package (no version/revision consulted — out of scope for this role)

## Source Fingerprint
sha256:dd60987f7ef0741e55712e0469e854500a80318dedc70c130645ccb51bbc0d00  (buster_ripper/utils.py — the defining and only file containing the name)
Note: this fingerprints the defining file, not the package-wide search the absence verdict rests on; that is why promotion is ineligible.

## Invalidates On
- Any edit under `buster_ripper/` that adds a `response_headers(...)` / `utils.response_headers(...)` call site or an import of the name
- Replacement of the inline header-strip dict comprehensions in `buster_ripper/routes/messages.py:257` or `buster_ripper/routes/passthrough.py:40` with a call to this helper
- Introduction of a new module, star-import, re-export, or dynamic-attribute dispatch under `buster_ripper/` that could reach the helper without a literal name in source
- A change to what counts as `buster_ripper/` (e.g. test files moved inside the source root), which would widen the search surface

## Checked At
2026-10-05T18:08:51Z

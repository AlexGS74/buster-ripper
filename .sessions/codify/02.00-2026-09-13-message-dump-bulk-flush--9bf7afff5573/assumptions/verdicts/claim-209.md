VERDICT: HOLDS
PROMOTION: INELIGIBLE — the verdict is an absence over the whole package; no single exact file establishes it, and the consulted surface is a directory scan of `buster_ripper/` (broad directory), not one recoverable file's bytes.

## Claim
`utils.forward_headers` is never called by any module under `buster_ripper/`

## What it actually does
`forward_headers(headers)` (utils.py:18-24) builds an upstream header dict by dropping `host`, `content-length`, and the 8 hop-by-hop names in `_HOP_BY_HOP`. It is never called. The only other mention in the package is a **dead import** at `buster_ripper/routes/chat.py:14` (`from ..utils import (... forward_headers ...)`); an import binds a name, it does not invoke it. `routes/chat.py` instead constructs upstream headers with its own local `_prepare_headers()` (chat.py:98-111) and passes that result to `session.post(headers=headers, ...)` at chat.py:128 — so the helper's logic is duplicated inline in the one module that imports it, and the imported name is unused.

## Source
Exhaustive reference search over the bounded source root `/Users/alexhubitski/Projects/buster-ripper/buster_ripper/` (all files, `__pycache__` excluded; the only non-`.py` file is `.DS_Store`): `grep -rn "forward_headers" buster_ripper/` returns exactly two hits —
- `buster_ripper/utils.py:18` — the `def` itself (definition, not a call)
- `buster_ripper/routes/chat.py:14` — the import name (binding, not a call)

No `forward_headers(` invocation exists anywhere in the package. Qualified form (`utils.forward_headers`) is covered by the same substring search. There is no `from ... import *`, no `__all__` re-export, and no `getattr(`/`globals()`/`importlib` dynamic dispatch anywhere under `buster_ripper/` (grep returned nothing), so no name-based indirection could call it.

Reading of "called": invocation. The sole reference that exists is an import, which is not a call under any ordinary reading; it is reported above so the reader can apply a stricter "any reference" reading if that is what the claim was meant to test.

## Source Version
local working tree, `buster_ripper/` package (no version/revision consulted — out of scope for this role)

## Source Fingerprint
sha256:dd60987f7ef0741e55712e0469e854500a80318dedc70c130645ccb51bbc0d00  (buster_ripper/utils.py — the defining file)
sha256:16c7e99938b55e508b36c16092e7a89883baeff293385c3dfa3e9c161b413cae  (buster_ripper/routes/chat.py — the only referring file)
Note: these fingerprint the two files above, not the package-wide search the absence verdict rests on; that is why promotion is ineligible.

## Invalidates On
- Any edit under `buster_ripper/` that adds an actual `forward_headers(...)` / `utils.forward_headers(...)` call site
- Removal of the unused import at `buster_ripper/routes/chat.py:14` (changes the "one dead import" detail, not the verdict)
- Introduction of a new module, star-import, re-export, or dynamic-attribute dispatch under `buster_ripper/` that could reach the helper without a literal name in source
- A change to what counts as `buster_ripper/` (e.g. test files moved inside the source root), which would widen the search surface

## Checked At
2026-10-05T18:08:51Z

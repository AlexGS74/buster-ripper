VERDICT: HOLDS
PROMOTION: INELIGIBLE — exhaustive negative claim settled by a bounded search over two directory trees (42 `.py` files), not by one exact source file; no single recoverable byte range establishes it

## Claim
No module under `buster_ripper/` or `tests/` imports `buster_ripper_legacy`

## What it actually does
Nothing under either root references that module by any name, so there is no import of it — static or
dynamic. The token `buster_ripper_legacy` has **zero** occurrences across all 42 `.py` files in the two
roots, as does the looser token `legacy` in any case (`grep -rni "legacy" buster_ripper/ tests/` → exit 1,
no output), which also rules out aliased or attribute forms (`from buster_ripper_legacy import …`,
`import buster_ripper_legacy as X`, `from . import buster_ripper_legacy`). Dynamic forms are equally
absent: `buster_ripper/` and `tests/` contain no `importlib`, `import_module`, `__import__`, `runpy`,
`spec_from_file_location`, `exec(`/`eval(`, or `sys.modules` reference at all — the only hits for that
sweep are `create_app()` at `buster_ripper/server.py:117`, its import and call in
`buster_ripper/main.py:9,188`, and `from buster_ripper.server import create_app` at
`tests/test_routes_integration.py:9` (the package, not the legacy script). No `uvicorn "module:app"` /
`":app"` style path string appears either, so there is no indirect load-by-dotted-path.
Roots enumerated exhaustively: `buster_ripper/` = 16 `.py` files (incl. `routes/`), `tests/` = 26 `.py`
files (incl. `tests/qa/`); the only non-`.py` content is `.DS_Store`, two `.pem` fixtures, and
`__pycache__` `.pyc` output. A binary-inclusive case-insensitive sweep (`grep -rail`) also found no
hit, including in the compiled `.pyc` files.

## Source
Bounded content search over `buster_ripper/` and `tests/` at `/Users/alexhubitski/Projects/buster-ripper`
(whole-tree token sweep plus import-mechanism sweep). Claim 3's subject **is** partly the test suite — its
import graph — so test files are the production surface for this claim, per the P10 exception; the
package side was read from production source. Corroborating (not establishing): the legacy module's own
surface was checked by reading `buster_ripper_legacy.py` in full.

## Source Version
Repo working tree, 2026-10-05; `buster_ripper/` package has no version string consulted; the candidate
target `buster_ripper_legacy.py` is 42129 bytes, mtime 2026-03-03 09:50:07
(sha256:7967563dc3ab4f66f0dc322c7945c5647b96c9f9a77fbafe3cd1d4780e8764bd)

## Source Fingerprint
unavailable — the establishing evidence is the absence of a token across 42 files in two roots; there is
no single exact byte range to fingerprint (see PROMOTION reason)

## Invalidates On
- Any new or edited file under `buster_ripper/` or `tests/` containing the string `buster_ripper_legacy`
  (import, `importlib`/`runpy` path load, or a dotted-path string)
- A file under either root added outside the `.py` extension set that performs an import (e.g. a
  `conftest.py`-adjacent plugin, `.pth`, or `sitecustomize` placed inside these roots)
- The legacy script being renamed or copied into one of these roots under a package-relative name —
  the claim is about the module name `buster_ripper_legacy`, so a rename changes what must be re-searched

## Checked At
2026-10-05T18:18:03Z

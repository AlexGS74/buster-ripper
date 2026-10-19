VERDICT: HOLDS
PROMOTION: INELIGIBLE — the claim is exhaustive over the whole `tests/` tree; the negative half ("no other module") is established by a tree-scoped search plus a directory listing, not by one exact source locator whose bytes are fingerprinted.

## Claim
`tests/test_serve_script_argv.py` is the only module under `tests/` that invokes `serve_buster_ripper.sh`

## What it actually does
`tests/test_serve_script_argv.py` is the only file anywhere under `tests/` that names
`serve_buster_ripper.sh` at all, and it does invoke it: `SCRIPT =
Path(__file__).resolve().parent.parent / "serve_buster_ripper.sh"` (line 25) is run as
`subprocess.run(["bash", str(SCRIPT)], ...)` in the `serve_argv` fixture (lines 94-98) and
re-read as text in `test_header_still_documents_the_disable_form` (line 137). No other module
under `tests/` references or launches the wrapper: the only other subprocess-driving module,
`tests/qa/harness.py`, spawns `PROXY_BIN = <repo>/.venv/bin/buster-ripper` (line 30) or
`sys.executable` + `tests/qa/counted_proxy_entry.py` (line 31), never the `.sh` wrapper.

Search surface (bounded to the supplied source root `/Users/alexhubitski/Projects/buster-ripper/tests`,
build outputs `__pycache__`/`.pyc` excluded, no traversal outside the root):
- `serve_buster_ripper\.sh` → 4 matches, all in `test_serve_script_argv.py`.
- case-insensitive `serve_buster|serve\.sh|_ripper\.sh|serve_script` → same 4 matches.
- `\.sh\b` over all file types → same 4 matches (no other shell-script path in any test).
- `subprocess|Popen|os\.system|check_output|bash|\"sh\"|'sh'` → only `tests/qa/harness.py`
  (console script, see above) and `test_serve_script_argv.py`.
- `runpy|shlex|exec\(|py_compile|compile\(` → no matches, so no dynamically assembled path or
  indirect launch of the wrapper.
- `test_serve_script_argv|serve_argv` → no other module imports it or reuses its fixture.
- Full file listing under `tests/` (22 `test_*.py`/helper modules plus `conftest.py`,
  `qa/` package): no shell or non-Python test harness that could invoke the wrapper.

Note on evidence class (P10): the subject of this claim **is** the test suite — which test
modules invoke the wrapper — so the test files are the production surface for it, and reading
them establishes rather than merely corroborates the claim. Nothing was executed to settle this;
"invokes" here is a static property of the module source (it contains the invocation), and
`bash serve_buster_ripper.sh` was never run by this check.

## Source
`/Users/alexhubitski/Projects/buster-ripper/tests/test_serve_script_argv.py` (lines 25, 94-98,
137 — the invocation site), with the `tests/`-rooted searches and directory listing above
supplying the exhaustiveness half.

## Source Version
Working-tree source at the supplied source root; no package version applies and no revision
metadata was read (`.sessions/.../assumptions/verdicts/` was only resolved via the session
entrypoint). `tests/qa/harness.py` and `tests/conftest.py` consulted as corroborating locators.

## Source Fingerprint
`tests/test_serve_script_argv.py`: `sha256:461cc81d83c288b0b12488ebcbe3a12ebb68eaffabde283da2bc2c001860be74`
corroboration — `tests/qa/harness.py`: `sha256:fd771cde67200a372ac0b60d35ba2c22f3f1089b19ae1a21068f17fd8369c2bc`
corroboration — `tests/conftest.py`: `sha256:c427e7812a1061420ed332921fbcfb0c49b4a4a28bb75c8c38c1b55ba74a55d4`
(The tree-wide negative is not itself fingerprinted — the reason promotion is ineligible.)

## Invalidates On
- Any new module under `tests/` that runs, sources, or otherwise launches
  `serve_buster_ripper.sh` (e.g. a second wrapper-level test, a `tests/qa/` gate that shells
  out to the wrapper instead of `PROXY_BIN`).
- `tests/test_serve_script_argv.py` bytes differing from
  `sha256:461cc81d83c288b0b12488ebcbe3a12ebb68eaffabde283da2bc2c001860be74` (its invocation
  could move or be removed).
- `tests/qa/harness.py` bytes differing from
  `sha256:fd771cde67200a372ac0b60d35ba2c22f3f1089b19ae1a21068f17fd8369c2bc`, specifically its
  `PROXY_BIN` / `COUNTED_ENTRY` command selection.
- Renaming, moving, or splitting the wrapper so the invoked file is no longer
  `<repo>/serve_buster_ripper.sh` (the searched string, and therefore the uniqueness result,
  is name-bound), or moving the wrapper to a path that is not `parent.parent` of `tests/`.

## Checked At
2026-10-05T03:39:16Z

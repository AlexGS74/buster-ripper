VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
No module under `tests/` invokes `serve_buster_ripper.sh`

## What it actually does
No file under `tests/` contains the string `serve_buster_ripper`, or any `.sh`
reference at all, so no test module invokes that shell script. The only
subprocess spawn in the whole test surface is `tests/qa/harness.py`
(`Proxy.__init__`, `subprocess.Popen(cmd, ...)` at line 129), and `cmd` there is
built from exactly two constants: `PROXY_BIN = <repo>/.venv/bin/buster-ripper`
(line 30) or `[sys.executable, COUNTED_ENTRY]` where
`COUNTED_ENTRY = <repo>/tests/qa/counted_proxy_entry.py` (line 31). Neither
path is a shell script. `counted_proxy_entry.py` in turn runs the proxy
in-process by importing `buster_ripper.main` and calling `cli.main()` under a
patched `sys.argv` — it spawns nothing. There is no `exec(`, `eval(`,
`os.environ`/`getenv`, `shutil.which`, or `runpy` anywhere under `tests/`, so no
indirect route (env-var pointer, resolved-on-PATH launcher, computed script
name) exists either. The claim holds under both readings of "module" (Python
modules only, or every file under `tests/`).

## Source
Bounded source root: `/Users/alexhubitski/Projects/buster-ripper/tests` —
all Python modules under it (22 files), verified by
`rg -uu --hidden -g '!__pycache__' 'serve_buster_ripper' .` (0 matches),
`rg -uu --hidden -g '!__pycache__' '\.sh\b' .` (0 matches), and
`rg -uu --hidden -g '!__pycache__' -e 'subprocess' -e 'os\.system' -e 'Popen'
-e 'os\.exec' -e 'execv' -e 'runpy' -e 'shutil\.which' -e 'exec\(' -e 'eval\('
-e 'os\.environ' -e 'getenv' .` (matches only in `qa/harness.py`).
Decisive bytes: `/Users/alexhubitski/Projects/buster-ripper/tests/qa/harness.py`
lines 29–31 and 111–129; `/Users/alexhubitski/Projects/buster-ripper/tests/qa/counted_proxy_entry.py`.
Note (P10 exception applied): the claim's subject IS the test suite, so test
files are the production surface for this verdict.

## Source Version
Working-tree files at `/Users/alexhubitski/Projects/buster-ripper/tests`
(no package/version involved; content fingerprinted below)

## Source Fingerprint
Aggregate over the consulted module set (per-file `sha256  relpath`, sorted,
re-hashed):
`sha256:c3e83bebc3d65b8fd94e0a2684e36c22844c1c435889b5cf2f87691cd58ac669`
Regenerate from `tests/` with:
`LC_ALL=C find . -name '*.py' -not -path '*/__pycache__/*' -type f | LC_ALL=C sort | xargs shasum -a 256 | LC_ALL=C shasum -a 256`
Key files consulted: `sha256:fd771cde67200a372ac0b60d35ba2c22f3f1089b19ae1a21068f17fd8369c2bc` (qa/harness.py), `sha256:624c8aaf02fb2e5f917300a6c4a4782b0e04216c29e13d2af7ec4c1d62c6c806` (qa/counted_proxy_entry.py)

## Invalidates On
- Any `tests/**/*.py` file whose content changes (aggregate fingerprint above no longer recomputes)
- A new Python module added under `tests/` (changes the module set the manifest covers)
- `tests/qa/harness.py` `PROXY_BIN` / `COUNTED_ENTRY` / `cmd` construction altered to name a shell script or launcher resolved from PATH or an environment variable
- Introduction of `serve_buster_ripper.sh` plus a subprocess/exec reference to it anywhere under `tests/`

## Checked At
2026-10-05T02:23:43Z

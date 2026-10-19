VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
No Python test source under `tests/` sets `config.STRIP_IMAGES` true

## What it actually does
Nothing under `tests/` mentions the identifier `STRIP_IMAGES` in any casing, by any
of the mechanisms the repo's own tests use to flip a config flag. The exhaustive
search over the 26 Python files under `tests/` (excluding `__pycache__`) returns
zero hits for `STRIP_IMAGES`, zero hits for a case-insensitive `strip_images`
outside two test-function names (`test_strip_images_replaces_with_placeholder`,
`test_strips_multiple_images` in `test_image.py`), and zero hits for an env-var
name containing `IMAGE`. Where tests do turn a config flag on, they name the flag
literally — `monkeypatch.setattr(cfg, "STRIP_DATE", True)` in
`test_normalize.py:146,159,187` — and the only `config` attributes patched with a
truthy image-related name are cache settings, not a strip switch:
`mocker.patch.object(config, "IMAGE_CACHE_TTL", ...)` and
`mocker.patch.object(config, "IMAGE_CACHE_MAX_SIZE", ...)` in `test_image.py`.
Image stripping itself is exercised by calling `image.strip_and_cache_images(body, ...)`
directly (`test_image.py:75,93,108,...`), not by enabling a `config` flag. The two
indirect routes a flag could be set without its name appearing — a generic
`setattr(config, <var>, ...)`/`getattr` loop and an env-var-driven config reload —
are both absent: the only dynamic attribute writes found are the `STRIP_DATE`
monkeypatches above and `test_serve_script_argv.py`'s subprocess-environment
builder, whose complete consumed-name list
(`test_serve_script_argv.py:39-42`: `UPSTREAM, HOST, PORT, STRIP_DATE, VERBOSE,
SKIP_TLS_VERIFY, DUMP_DIR, DUMP_FORMAT, PRESERVED_THINKING`) contains no image
flag. `tests/conftest.py` holds no fixtures at all ("No shared fixtures needed"),
so there is no shared setup that could set a flag on every test's behalf.

Scope note: the claim is about test sources, so the test files are the production
surface for it (the P10 exception applies); the verdict does not depend on whether
`STRIP_IMAGES` exists as an attribute of `buster_ripper/config.py`, which lies
outside the supplied source root and was not read.

## Source
Source root `/Users/alexhubitski/Projects/buster-ripper/tests` — recursive
`grep -rn "STRIP_IMAGES"` (case-sensitive and `-i`) plus
`grep -rn "setattr|setenv|environ|reload(|__dict__|vars(|\.update("` over
`*.py` across all 26 Python files (23 top-level/`qa/` test and helper modules,
`__init__.py`, `conftest.py`).

## Source Version
Local working tree at `/Users/alexhubitski/Projects/buster-ripper/tests`; no
package version applies (subject is the repo's own test suite). Repository state
was not resolved — out of scope for this role.

## Source Fingerprint
sha256:3c15681d3f6e38fcf9db20a3fd66b0dabcb7f06241457cd227cd3091e58d0372
Recoverable as: for each file matching
`find tests -type f -name '*.py' -not -path '*/__pycache__/*'` sorted by path,
emit `FILE <relpath>\n` followed by the file's raw bytes, then sha256 the
concatenation. The 26-file set this digest covers is the exact surface the claim
quantifies over.

## Invalidates On
- Any file under `tests/` introduces `STRIP_IMAGES` (assignment, `setattr`,
  `monkeypatch.setattr`, `mocker.patch.object`, or an env/argv name the config
  loader reads) — the aggregate digest above changes and the claim must be re-run.
- A test gains a generic config mutation (a `setattr(config, name, value)` loop,
  a `config.__dict__.update(...)`, an `importlib.reload(config)` after
  `monkeypatch.setenv`, or a fixture in the currently empty `tests/conftest.py`)
  that could set a flag whose name never appears in test source.
- The set of Python files under `tests/` changes for any reason (file added,
  removed, or renamed), since the claim is exhaustive over that membership.

## Checked At
2026-10-05

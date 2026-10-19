VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`pyproject.toml` omits `buster_ripper/routes/*`, `main.py` and `server.py` from coverage and gates the suite at 80% (`--cov-fail-under=80` in `addopts`)

## What it actually does
Both halves of the row are present in `pyproject.toml`. `[tool.coverage.run]` (with `source = ["buster_ripper"]`) has `omit = ["buster_ripper/routes/*", "buster_ripper/main.py", "buster_ripper/server.py"]`, so the route handlers, the package's main module, and the package's server module are excluded from coverage. `[tool.pytest.ini_options]` `addopts = "--cov=buster_ripper --cov-report=term-missing --cov-fail-under=80"` sets the 80% gate exactly as claimed. Note the omit entries are package-qualified (`buster_ripper/main.py`, `buster_ripper/server.py`); with `source` limited to the `buster_ripper` package there is no other reading in which different files are meant.

## Source
`/Users/alexhubitski/Projects/buster-ripper/pyproject.toml`, lines 19 (addopts) and 24–30 ([tool.coverage.run] source/omit)

## Source Version
Repo file as present on disk, 2026-09-19 (project version string `0.1.0`)

## Source Fingerprint
sha256:e1b7035805a8e946b71f947026e86e47e328f7e0b9ed9568b8cdb648e059079e

## Invalidates On
- Any edit to `[tool.coverage.run]` omit list (removal or change of the three omit entries)
- Any edit to `[tool.pytest.ini_options]` `addopts` removing or changing `--cov-fail-under=80`
- Coverage or pytest-cov semantics change such that `omit` patterns are no longer matched against `source`-rooted paths

## Checked At
2026-09-19

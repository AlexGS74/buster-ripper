VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`pyproject.toml`'s pytest ini configures no collection scoping — no `testpaths` and no `norecursedirs`

## What it actually does
`[tool.pytest.ini_options]` in `pyproject.toml` contains exactly two keys: `asyncio_mode = "auto"` and `addopts = "--cov=buster_ripper --cov-report=term-missing --cov-fail-under=80"`. Neither `testpaths` nor `norecursedirs` appears anywhere in the file, so the pytest ini configures no collection scoping. No competing pytest config file exists in the repo root (`pytest.ini`, `tox.ini`, `setup.cfg`, `.coveragerc` all absent), so `pyproject.toml`'s ini is the whole config surface and the absence is real, not overridden elsewhere.

## Source
`/Users/alexhubitski/Projects/buster-ripper/pyproject.toml`, lines 17–19 (`[tool.pytest.ini_options]` section)

## Source Version
Repo file as present on disk, 2026-09-19 (project version string `0.1.0`)

## Source Fingerprint
sha256:e1b7035805a8e946b71f947026e86e47e328f7e0b9ed9568b8cdb648e059079e

## Invalidates On
- Addition of `testpaths` or `norecursedirs` (or any other collection-scoping key) to `[tool.pytest.ini_options]`
- Creation of a `pytest.ini`/`tox.ini`/`setup.cfg` that changes which ini source pytest reads
- Deletion or restructuring of the `[tool.pytest.ini_options]` section

## Checked At
2026-09-19

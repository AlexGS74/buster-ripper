VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`pyproject.toml` declares `aiohttp` and `click` as the only runtime dependencies

## What it actually does
The project's PEP 621 runtime dependency table is `dependencies = ["aiohttp", "click"]` —
exactly those two entries, with no version constraints. The file declares no
`[project.optional-dependencies]` table, no `dynamic = [...]` field, and no legacy
`[tool.setuptools]`/`extras_require` dependency source, so there is no other key in this
file that contributes a run-time requirement. The other dependency-bearing tables present
are explicitly not runtime: `[build-system] requires = ["hatchling"]` is the PEP 518 build
environment, and `[dependency-groups] dev = [pytest, pytest-aiohttp, pytest-asyncio,
pytest-cov, pytest-mock]` is a PEP 735 group, which by spec is not part of the project's
install requirements. `requires-python = ">=3.11"` constrains the interpreter, not a package.

## Source
/Users/alexhubitski/Projects/buster-ripper/pyproject.toml — lines 1–8 (`[project]
dependencies`), line 13–15 (`[build-system]`), lines 36–43 (`[dependency-groups] dev`).
The claim's subject is the declaration in this file, so the file is the production surface
for it; ranking it as configuration evidence (rank 3) is the strongest available source.
Read in full (43 lines, terminating `]\n` after `pytest-mock>=3.15.1`), so the exhaustive
half of the claim ("only") was checked against the whole file, not a section.
Terminology settled by the specifications rather than chosen: PEP 621 names
`project.dependencies` the run-time dependencies and `optional-dependencies` the extras;
PEP 735 states dependency groups are not recorded in the package's install metadata.
Corroborating (not establishing) behaviour: the only `[project.scripts]` entry is
`buster-ripper = "buster_ripper.main:main"`.

## Source Version
local repository file, no VCS revision consulted; contents pinned by fingerprint below
(project metadata: `buster-ripper` 0.1.0, `requires-python >= 3.11`)

## Source Fingerprint
sha256:c623bdc2824331933f9a02ede763170ef8fa07b5d09be8047dc9b9d85a22e19e

## Invalidates On
- Any entry added to `[project] dependencies` in pyproject.toml (fingerprint changes).
- A `[project.optional-dependencies]` table added, or `dynamic = ["dependencies"]` /
  `dynamic = ["optional-dependencies"]` set, moving the runtime set out of this file.
- A `[tool.hatch.metadata.hooks]` or other hatchling metadata hook introduced that
  injects dependencies at build time, making the static list non-exhaustive.
- The file replaced by a setup.py/setup.cfg dependency declaration, or a change to
  `[build-system]`/`[dependency-groups]` that promotes a build or dev package into the
  resolved runtime environment (e.g. a uv workspace/`[tool.uv]` block adding requirements).
- Reading standards differently: if a future consumer treats PEP 735 `dev` group members
  as runtime dependencies, the "only" half stops holding — that is a consumer change, not
  this file.

## Checked At
2026-10-05

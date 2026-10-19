VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`pyproject.toml` defines a `buster-ripper` console-script entry point binding to `buster_ripper.main:main`

## What it actually does
`pyproject.toml` declares `[project.scripts]` with exactly one entry: `buster-ripper = "buster_ripper.main:main"` — a console script named `buster-ripper` bound to the `main` callable in the `buster_ripper.main` module, as claimed.

## Source
`/Users/alexhubitski/Projects/buster-ripper/pyproject.toml`, lines 10–11 (`[project.scripts]` section)

## Source Version
Repo file as present on disk, 2026-09-19 (project version string `0.1.0`)

## Source Fingerprint
sha256:e1b7035805a8e946b71f947026e86e47e328f7e0b9ed9568b8cdb648e059079e

## Invalidates On
- Rename or removal of the `buster-ripper` key in `[project.scripts]`
- Change of the binding target string away from `buster_ripper.main:main`
- Removal/renaming of `buster_ripper/main.py`'s `main` callable such that the declared target no longer resolves (entry point definition itself is what this verdict covers)

## Checked At
2026-09-19

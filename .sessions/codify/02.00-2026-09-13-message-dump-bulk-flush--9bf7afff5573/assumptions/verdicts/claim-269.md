VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
The `--strip-images` option's help text describes stripping images from user messages

## What it actually does
The `--strip-images` click option is defined with
`help="Strip images from user messages and cache them for later retrieval."` —
the help text literally states that images are stripped from user messages (and
adds that they are cached for later retrieval). The option is a plain
`is_flag=True` boolean switch, so "strip" is opt-in.

## Source
/Users/alexhubitski/Projects/buster-ripper/buster_ripper/main.py lines 74-75
(`@click.option("--strip-images", is_flag=True, help=...)`)

## Source Version
local repo source under the supplied source root; no package/version string consulted

## Source Fingerprint
sha256:fd36dc12055f67693d9b852fb15a66aa4a3a7864103ac3de0ee67d9520048d47

## Invalidates On
- the `help=` string on the `--strip-images` decorator in `buster_ripper/main.py` is
  edited so it no longer mentions stripping images from user messages
- the `--strip-images` option is moved out of `main.py` (another module defines the CLI)

## Checked At
2026-10-05

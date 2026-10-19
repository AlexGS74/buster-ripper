VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`config.STRIP_IMAGES` defaults to False

## What it actually does
`config.py` declares `STRIP_IMAGES: bool = False` at module level under the
"Image handling settings" heading, so an import of `buster_ripper.config` without
`main()` having run reads `False`. Nothing overrides that initial value at import
time: the package has no `os.environ`/`getenv`/`setattr`/`globals()` mechanism
(grep across `buster_ripper/**/*.py` returns no matches), and the only assignment
to `config.STRIP_IMAGES` anywhere in the source root is `main.py` line 44. All
other references (`routes/messages.py:38`, `normalize.py:212`) only read it.

## Source
/Users/alexhubitski/Projects/buster-ripper/buster_ripper/config.py line 40;
cross-checked by a source-root grep for `STRIP_IMAGES` assignment sites

## Source Version
local repo source under the supplied source root; no package/version string consulted

## Source Fingerprint
sha256:22348c68d21b91a30e75eeea802d4d390b2bcdabbf974b0ddc743c9354635522

## Invalidates On
- the literal on `config.py` line 40 changes to `True` (or the annotation/initialiser is
  removed so the name has no default)
- an environment/config-file loading path is added to `config.py` (or an import-time
  `setattr`) that can set `STRIP_IMAGES` before `main()` runs

## Checked At
2026-10-05

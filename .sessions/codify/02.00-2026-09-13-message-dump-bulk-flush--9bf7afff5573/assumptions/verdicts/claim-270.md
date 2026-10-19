VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`main.py` assigns `config.STRIP_IMAGES` from the parsed `--strip-images` flag

## What it actually does
`main.py` line 44 performs `config.STRIP_IMAGES = cli["strip_images"]` inside
`_apply_cli_config()`, and `main()` calls that with `strip_images=strip_images`
(line 140), where `strip_images` is the `main()` parameter (line 132) that click
fills with the parsed `--strip-images` value. The option is declared as
`@click.option("--strip-images", is_flag=True, ...)` with no `expose_value=False`
and no callback, so click's parsed boolean is routed through the parameter into
the config global (unlike `--dump-format`/`--skip-tls-verify`, which use
`expose_value=False` plus a `_set_*` callback to write config directly).

## Source
/Users/alexhubitski/Projects/buster-ripper/buster_ripper/main.py — line 44
(assignment), line 132 (`main()` parameter), line 140 (argument passed),
lines 74-75 (option declaration)

## Source Version
local repo source under the supplied source root; no package/version string consulted

## Source Fingerprint
sha256:fd36dc12055f67693d9b852fb15a66aa4a3a7864103ac3de0ee67d9520048d47

## Invalidates On
- `config.STRIP_IMAGES = ...` is removed from `_apply_cli_config` in `main.py`, or the
  dict key it reads stops being `"strip_images"`
- `--strip-images` gains `expose_value=False`/a callback (assignment would then move to
  a `_set_strip_images`-style callback instead of `_apply_cli_config`)
- `strip_images` is dropped from `main()`'s signature or from the `_apply_cli_config`
  call at line 140

## Checked At
2026-10-05

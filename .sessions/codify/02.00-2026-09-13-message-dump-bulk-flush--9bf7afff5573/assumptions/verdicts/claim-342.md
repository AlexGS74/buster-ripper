VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`--image-cache-ttl`'s parsed value is assigned onto `config.IMAGE_CACHE_TTL`

## What it actually does
The option is a plain value option — `@click.option("--image-cache-ttl", default=300,
show_default=True, type=int, ...)` declares neither `expose_value=False` nor a callback, so click's
parsed value arrives as `main()`'s `image_cache_ttl` parameter, is forwarded as
`image_cache_ttl=image_cache_ttl` into `_apply_cli_config`, and is written by the single statement
`config.IMAGE_CACHE_TTL = cli["image_cache_ttl"]`. Because `main.py` binds the module object with
`from . import config`, that is an assignment onto the module global declared at `config.py:42`
(`IMAGE_CACHE_TTL: int = 300`) — the same object `image.py:56` reads at cache-expiry time, which is
the propagation the config docstring describes. The assignment is unconditional: it happens for the
default too, not only when the flag is supplied.

## Source
/Users/alexhubitski/Projects/buster-ripper/buster_ripper/main.py — L8 (`from . import config`),
L36–41 (`_apply_cli_config` contract), L57 (the assignment), L127–128 (the option), L131–136
(`main()` signature), L148 (keyword forwarded). Cross-checked for the target binding at
/Users/alexhubitski/Projects/buster-ripper/buster_ripper/config.py L42 and its reader at
/Users/alexhubitski/Projects/buster-ripper/buster_ripper/image.py L56. Production source only; no
test consulted, nothing executed (P1). Checked independently of claims 340/341.

## Source Version
Local source root `/Users/alexhubitski/Projects/buster-ripper/buster_ripper` as present at check
time; version/revision string unavailable (git metadata is out of scope for this role).

## Source Fingerprint
sha256:fd36dc12055f67693d9b852fb15a66aa4a3a7864103ac3de0ee67d9520048d47 (main.py)
sha256:22348c68d21b91a30e75eeea802d4d390b2bcdabbf974b0ddc743c9354635522 (config.py)
sha256: image.py not fingerprinted — read only as a corroborating consumer reference, not consulted
to establish the verdict

## Invalidates On
- main.py L57 removed or retargeted (e.g. assignment moved to a `expose_value=False` callback that
  writes a different name, or the `cli["image_cache_ttl"]` key renamed)
- the `--image-cache-ttl` option given `expose_value=False` or a callback that stops returning the
  value, so `image_cache_ttl` no longer reaches `main()`'s parameter at L136
- `_apply_cli_config`'s call at L148 dropping the `image_cache_ttl=` keyword
- config.py renaming `IMAGE_CACHE_TTL` (assignment would then create a separate attribute rather than
  set the documented global)

## Checked At
2026-10-05T21:36:27Z

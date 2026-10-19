VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
The `--image-cache-ttl` option declaration carries `help=` text presenting the TTL in seconds

## What it actually does
The `--image-cache-ttl` option is declared once, at lines 127–128, and its `help=`
string is literally `"Image cache TTL in seconds."` — the value is presented to the
user as a count of seconds (the declaration also gives `default=300, show_default=True,
type=int`, so `--help` renders the default as the bare integer `300` with the seconds
unit coming from the help text itself). No other unit is named anywhere in the option
declaration.

## Source
`/Users/alexhubitski/Projects/buster-ripper/buster_ripper/main.py` lines 127–128:

```python
@click.option("--image-cache-ttl", default=300, show_default=True, type=int,
              help="Image cache TTL in seconds.")
```

Established from production source (the option decorator in the CLI entrypoint), not
from a test, fixture, or snapshot. A case-insensitive search of the same file for
`image-cache-ttl` / `image_cache_ttl` / `IMAGE_CACHE_TTL` returns exactly this one
declaration plus the `_apply_cli_config` assignment (`main.py:57`), the `main()`
parameter (`main.py:136`), and the call site keyword (`main.py:148`) — no second or
competing declaration.

## Source Version
unavailable — no VCS metadata consulted (Git boundary); local working-tree file bytes
only.

## Source Fingerprint
sha256:fd36dc12055f67693d9b852fb15a66aa4a3a7864103ac3de0ee67d9520048d47

## Invalidates On
- Any edit to `buster_ripper/main.py` lines 127–128 that changes, removes, or rewords
  the `help=` string for `--image-cache-ttl` (e.g. to milliseconds, minutes, or a
  unitless wording).
- Renaming the option (`--image-cache-ttl`) or moving its declaration to another
  module or a shared options group.
- Replacing the literal `help=` with `help=` text that derives the unit from elsewhere
  (a constant, a docstring, or a formatter), which would make the unit no longer
  visible in the declaration itself.
- A different revision of this file whose sha256 is not
  `fd36dc12055f67693d9b852fb15a66aa4a3a7864103ac3de0ee67d9520048d47`.

## Checked At
2026-10-05

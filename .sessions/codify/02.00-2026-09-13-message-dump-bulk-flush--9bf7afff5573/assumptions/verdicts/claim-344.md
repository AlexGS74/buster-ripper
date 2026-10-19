VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
A `callback=`-wired click option reaches its `config` global while appearing in neither `main()`'s parameters nor `_apply_cli_config`'s keys

## What it actually does
The pattern exists and is the repo's established way to attach a CLI option to a config
global without threading it through the command function. Four options are wired with
`callback=`; each also carries `expose_value=False`, so click never passes its value to
`main()`, and each callback body performs the config assignment itself at parse time:

| option | line | callback | config global written |
|---|---|---|---|
| `--skip-tls-verify` | 64–65 | `_set_skip_tls_verify` (30–33) | `config.SKIP_TLS_VERIFY` |
| `--dump-format` | 80–82 | `_set_dump_format` (12–15) | `config.DUMP_FORMAT` |
| `--dump-flush-interval` | 86–88 | `_set_dump_flush_interval` (18–21) | `config.DUMP_FLUSH_INTERVAL` |
| `--dump-flush-bytes` | 91–93 | `_set_dump_flush_bytes` (24–27) | `config.DUMP_FLUSH_BYTES` |

None of the four corresponding parameter names (`skip_tls_verify`, `dump_format`,
`dump_flush_interval`, `dump_flush_bytes`) appears in `main()`'s signature
(lines 131–137) or as a key in `_apply_cli_config` (lines 42–58). The docstrings state
the intent explicitly — "Assign config.X at parse time; main() takes no new parameter."

The claim holds under either reading: existentially (four such options exist) and
universally (every `callback=`-wired option in this file does all three things). No
`callback=`-wired option here also appears in `main()`'s parameters or in
`_apply_cli_config`'s keys, and no callback fails to write a `config` global.

## Source
`/Users/alexhubitski/Projects/buster-ripper/buster_ripper/main.py` — production CLI
entrypoint source. Establishing locators: lines 12–33 (callback bodies, each
`config.<NAME> = value` then `return value`), lines 64–65, 80–82, 86–88, 91–93
(`expose_value=False` + `callback=`), lines 131–137 (`main()` signature), lines 42–58
(`_apply_cli_config` key set). Verified by a case-insensitive search of the file for
`callback=` and `expose_value`, which returns only these four options.

Grounded in source, not tests (P10). The wiring is read statically from the decorator
arguments and callback bodies; nothing here required executing the CLI. Note the
supplied source locator is `main.py` only, so `buster_ripper/config.py` was not opened —
the verdict is about the option reaching the `config` module attribute it names, which
`from . import config` (line 8) plus `config.<NAME> = value` establishes, not about
whether that name is pre-declared in `config.py`.

## Source Version
unavailable — no VCS metadata consulted (Git boundary); local working-tree file bytes
only.

## Source Fingerprint
sha256:fd36dc12055f67693d9b852fb15a66aa4a3a7864103ac3de0ee67d9520048d47

## Invalidates On
- Any of the four options gaining `expose_value=True`, or losing `expose_value=False`,
  which would put its value into `main()`'s parameters.
- Any of the four parameter names being added to `main()`'s signature (lines 131–137)
  or to `_apply_cli_config`'s key set (lines 42–58), collapsing the bypass.
- Any callback body (lines 12–33) having its `config.<NAME> = value` assignment removed,
  renamed, or redirected so the option no longer reaches a config global.
- A new `callback=`-wired option that does appear in `main()`'s parameters or in
  `_apply_cli_config`'s keys, which would falsify the claim under its universal reading.
- A different revision of this file whose sha256 is not
  `fd36dc12055f67693d9b852fb15a66aa4a3a7864103ac3de0ee67d9520048d47`.

## Checked At
2026-10-05

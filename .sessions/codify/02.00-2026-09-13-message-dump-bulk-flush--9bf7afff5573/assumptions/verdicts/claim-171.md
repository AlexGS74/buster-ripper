VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`tests/test_dump_write_count.py` binds `COLLAPSED_FORMATS` to `["change-focused", "self-contained"]`, exactly the two non-`documented` members of `config.DUMP_FORMATS`

## What it actually does
`tests/test_dump_write_count.py` line 29 reads `COLLAPSED_FORMATS = ["change-focused", "self-contained"]` — the module's only definition of that name (all other occurrences at lines 164, 189, 239, 262 are `@pytest.mark.parametrize("fmt", COLLAPSED_FORMATS)` usages). `buster_ripper/config.py` line 23 defines `DUMP_FORMATS: tuple[str, ...] = ("documented", "change-focused", "self-contained")` — the closed set with no other definition or reassignment anywhere in the package (all other `DUMP_FORMATS` occurrences in `buster_ripper/dump.py` and `buster_ripper/main.py` are reads). The two non-`documented` members of that tuple are exactly `"change-focused"` and `"self-contained"`, matching the test constant element-for-element and even in order. The config comment (lines 20–22) independently corroborates the framing, calling them "the two collapsed values" against `"documented"`, today's per-turn layout and the default. The only type-level difference is list vs tuple; the claim asserts membership ("members"), which holds either way.

P10 note: the claim's subject includes the test file itself, so `tests/test_dump_write_count.py` is the production surface for the binding half (P10 exception); the `config.DUMP_FORMATS` half is grounded in production source.

## Source
`tests/test_dump_write_count.py` line 29 (the binding), jointly with `buster_ripper/config.py` line 23 (the format set the equation's second half cites). The claim spans the two files; both locators jointly establish it.

## Source Version
Repo working tree at /Users/alexhubitski/Projects/buster-ripper, checked 2026-09-28 (no package version applicable; local source files).

## Source Fingerprint
sha256:63b2c4c8e049f9462d4b44277cb1a3c387680da2820154a3e583d601c814822e (tests/test_dump_write_count.py)
sha256:7e1eff53eb56eb96aba4ff9badd0d9d16f3b02f062e2973676dcf8d670f71dde (buster_ripper/config.py)

## Invalidates On
- `COLLAPSED_FORMATS` in `tests/test_dump_write_count.py` changes value, is renamed, or a second binding of that name appears in the module
- `DUMP_FORMATS` in `buster_ripper/config.py` gains, drops, or renames a member (a third collapsed format or removal of one breaks the "exactly the two" equation)
- `DUMP_FORMATS` becomes mutable at runtime or gains a second assignment site that changes its effective members

## Checked At
2026-09-28

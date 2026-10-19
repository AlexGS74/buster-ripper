VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`README.md`'s dump section shows selecting a collapsed format on the serve-script run path it documents

## What it actually does
The dump section's "### Collapsed dump formats (`--dump-format`)" subsection (which explicitly labels `change-focused` and `self-contained` as **collapsed** formats) shows `DUMP_FORMAT=change-focused DUMP_DIR=~/mllm/prompt-diffs bash serve_buster_ripper.sh` — i.e. a collapsed format selected via the `DUMP_FORMAT` env var on the same serve-script run path the dump section documents. The direct `buster-ripper --dump-format self-contained …` call appears only as the "or directly" alternative.

## Source
/Users/alexhubitski/Projects/buster-ripper/README.md — "### Collapsed dump formats (`--dump-format`)" subsection of "## Dump mode", fenced bash block `DUMP_FORMAT=change-focused DUMP_DIR=~/mllm/prompt-diffs bash serve_buster_ripper.sh` (line 192 of the file)

## Source Version
unavailable (working-tree README.md at repo root; git metadata not consulted)

## Source Fingerprint
sha256:fe80cb9acabfe699fac070a69b43ac425436606d627b243825d4a028d75a1c15

## Invalidates On
The collapsed-formats subsection's serve-script example drops the `DUMP_FORMAT=…` env-var prefix on `bash serve_buster_ripper.sh` (e.g. only the direct `buster-ripper --dump-format …` call remains, or `DUMP_FORMAT` no longer maps to a collapsed format)
sha256 of README.md no longer equals fe80cb9acabfe699fac070a69b43ac425436606d627b243825d4a028d75a1c15

## Checked At
2026-09-28

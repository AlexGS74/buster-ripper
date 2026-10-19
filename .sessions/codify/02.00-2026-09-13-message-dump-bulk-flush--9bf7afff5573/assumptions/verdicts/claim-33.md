VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`README.md`'s dump section documents the serve script (`DUMP_DIR=… bash serve_buster_ripper.sh`) as its run path

## What it actually does
The "## Dump mode (`--dump-dir`)" section's fenced bash block gives exactly `DUMP_DIR=~/mllm/prompt-diffs bash serve_buster_ripper.sh` as the way to run dump mode — the serve script invoked with the `DUMP_DIR` env var, not a direct `buster-ripper` command. (A direct-call variant appears only later, inside the collapsed-formats subsection, as an "or directly" alternative.)

## Source
/Users/alexhubitski/Projects/buster-ripper/README.md — "## Dump mode (`--dump-dir`)" section, fenced bash block `DUMP_DIR=~/mllm/prompt-diffs bash serve_buster_ripper.sh` (line 145 of the file)

## Source Version
unavailable (working-tree README.md at repo root; git metadata not consulted)

## Source Fingerprint
sha256:fe80cb9acabfe699fac070a69b43ac425436606d627b243825d4a028d75a1c15

## Invalidates On
The dump-mode section's run command changes away from the `DUMP_DIR=… bash serve_buster_ripper.sh` serve-script invocation (e.g. replaced by a direct `buster-ripper --dump-dir …` call)
sha256 of README.md no longer equals fe80cb9acabfe699fac070a69b43ac425436606d627b243825d4a028d75a1c15

## Checked At
2026-09-28

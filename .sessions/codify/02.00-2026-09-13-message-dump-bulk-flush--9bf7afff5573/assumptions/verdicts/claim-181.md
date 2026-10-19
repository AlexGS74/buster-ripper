VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`README.md` documents `SKIP_TLS_VERIFY=1 bash serve_buster_ripper.sh` as the TLS opt-out's run path

## What it actually does
In the "Serve script options" section, README.md's fenced bash block presents `SKIP_TLS_VERIFY=1 bash serve_buster_ripper.sh` (comment: "Skip upstream TLS certificate verification — for self-signed certificates. … SKIP_TLS_VERIFY=1 turns it off process-wide (--skip-tls-verify)") — i.e. the TLS-verification opt-out is documented via the serve-script run path with the env var, not via a direct `buster-ripper` invocation.

## Source
/Users/alexhubitski/Projects/buster-ripper/README.md — "## Serve script options" section, the `SKIP_TLS_VERIFY=1 bash serve_buster_ripper.sh` line of the fenced bash block (lines 89–92 of the file)

## Source Version
unavailable (working-tree README.md at repo root; git metadata not consulted)

## Source Fingerprint
sha256:fe80cb9acabfe699fac070a69b43ac425436606d627b243825d4a028d75a1c15

## Invalidates On
README.md's serve-script-options block no longer shows `SKIP_TLS_VERIFY=1` as an env-var prefix on `bash serve_buster_ripper.sh` (e.g. opt-out moved to a direct `buster-ripper --skip-tls-verify` example or removed)
sha256 of README.md no longer equals fe80cb9acabfe699fac070a69b43ac425436606d627b243825d4a028d75a1c15

## Checked At
2026-09-28

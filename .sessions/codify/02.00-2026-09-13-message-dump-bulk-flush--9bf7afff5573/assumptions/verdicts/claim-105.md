VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`_artifact_path` constructs a per-session `<format>.log` path

## What it actually does
It joins `state.session_dir` (created as `config.DUMP_DIR / sid`) with `f"{fmt}{_ARTIFACT_SUFFIX}"`; the suffix is `.log`.

## Source
`buster_ripper/dump.py:29, 142-144, 268-270`

## Source Version
unavailable

## Source Fingerprint
sha256:c8f093c89b61c355ac182feb070cd90b84924ca0d187332117d76b5c766ad9b9

## Invalidates On
Change to `buster_ripper/dump.py` affecting `_artifact_path`, `_ARTIFACT_SUFFIX`, or session directory construction.

## Checked At
2026-09-27

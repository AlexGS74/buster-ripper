VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
No directory named `convert` exists under `buster_ripper/`

## What it actually does
The supplied source root contains exactly two directories — the package root
itself and `buster_ripper/routes/` — and no entry at any depth is named `convert`
or even contains the substring `convert` case-insensitively. Complete directory
membership consulted: `buster_ripper/`, `buster_ripper/routes/`. All other
entries are files (`__init__.py`, `backend.py`, `compaction.py`, `config.py`,
`dump.py`, `image.py`, `main.py`, `normalize.py`, `server.py`, `session.py`,
`utils.py`, `routes/__init__.py`, `routes/chat.py`, `routes/count_tokens.py`,
`routes/messages.py`, `routes/passthrough.py`, `.DS_Store`).

The claim holds under both readings of "under" (direct children of the package
root, and recursive beneath it), so no term choice was needed to reach the
verdict.

## Source
Directory enumeration of the supplied source root
`/Users/alexhubitski/Projects/buster-ripper/buster_ripper`, exhaustive over
`find . -type d` (excluding `.git/` and `__pycache__`, neither of which is
present inside the source root). Read-only listing; nothing was executed and no
session file was opened.

## Source Version
Mutable worktree source at check time — revision unavailable (Git metadata is
outside this role's evidence boundary).

## Source Fingerprint
sha256:3947323a1cfb5e91e970409add094c3d86ba5c8e3928bc867bc1e0b02e87affd
(reproducible as: `cd buster_ripper && find . -type d -not -path '*/.git/*'
-not -path '*/__pycache__*' | sort | shasum -a 256`)
Full-membership (dirs + files) listing for corroboration:
sha256:bfa02b6bafe02376c26b493c2fca86589e91a92d0dbfd0f4e4b35eb20d248cd4

## Invalidates On
- Creation of any directory named `convert` at any depth beneath
  `buster_ripper/` (the dirs-only fingerprint changes).
- Repointing the source-root locator at a different tree, snapshot, or worktree
  whose membership differs from the one enumerated above.
- Restructuring that moves `routes/` or adds any subpackage (dirs-only
  fingerprint changes) — re-check rather than reuse.

## Checked At
2026-10-05T18:34:17Z

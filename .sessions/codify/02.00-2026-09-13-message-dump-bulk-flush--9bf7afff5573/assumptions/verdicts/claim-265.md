VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`normalize` appends its image-strip change string inside that same guarded step

## What it actually does
The only image-related change string, `"stripped and cached images"`, is appended by
`changes.append(...)` at line 214, which sits in the *body* of the `if` guard at line 212 —
confirmed independently by parsing the module's syntax tree: the `If` node spans lines 212-214
and its body contains exactly two statements, `image_handler.strip_and_cache_images(body,
session_id)` (213) and the `changes.append(...)` (214). The guard has no `else` branch, and no
other `changes` append mentioning images appears anywhere in `normalize`, so the string is
produced strictly inside the guarded step and cannot be reported when the guard is false.

## Source
/Users/alexhubitski/Projects/buster-ripper/buster_ripper/normalize.py — `normalize`, lines 212-214
(append at line 214, inside the guarded block). Structural confirmation via `ast.parse` of the
same file bytes — static syntax read only, no project code executed (P1).

## Source Version
unavailable — the bounded source (single module file) carries no version or revision string;
fingerprint below is the stability basis.

## Source Fingerprint
sha256:a7ddbf062ad57951debb654988853a49a4bae549cb900da544fa3cd5a632ffca

## Invalidates On
- Any edit that dedents the `changes.append("stripped and cached images")` line out of the
  line-212 `if` block, or adds an `else`/unconditional path that appends an image change string.
- Any edit that moves the append into `image_handler.strip_and_cache_images` while the caller
  still appends elsewhere (the string's production site would no longer be this guarded step).
- Renaming or rewording the change string (claim's referent "its image-strip change string"
  would need re-identification).

## Checked At
2026-10-05T18:37:49Z

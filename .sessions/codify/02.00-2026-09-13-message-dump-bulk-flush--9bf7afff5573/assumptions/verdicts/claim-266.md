VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`normalize` runs `_normalize_cache_control` after its image step

## What it actually does
`normalize` is straight-line: the image step occupies lines 210-214 (the guarded block) and the
`_normalize_cache_control` call is the statement at line 223, labelled step 3, which executes
after the image block. `_normalize_cache_control` has exactly one call site in the module
(line 223) and exactly one definition (line 124), and no loop, branch, or earlier statement in
`normalize` reaches it, so the ordering is unconditional whenever both steps are reached. One
qualification that does not affect the claim: when the image guard at line 212 is false the image
step does nothing at all, so the ordering is "cache-control stripping is positioned and executed
after image stripping", not "cache-control stripping only happens if images were stripped".

## Source
/Users/alexhubitski/Projects/buster-ripper/buster_ripper/normalize.py — `normalize`, image step
lines 210-214 and the `_normalize_cache_control(body)` call at line 223; call-site/definition
enumeration by grep over the same file (only lines 124, 212, 213, 223 match).

## Source Version
unavailable — the bounded source (single module file) carries no version or revision string;
fingerprint below is the stability basis.

## Source Fingerprint
sha256:a7ddbf062ad57951debb654988853a49a4bae549cb900da544fa3cd5a632ffca

## Invalidates On
- Any edit that moves the `changes.extend(_normalize_cache_control(body))` statement above the
  line-212 image block within `normalize`.
- Any edit that calls `_normalize_cache_control` from an earlier helper in the `normalize`
  sequence (e.g. inside `image_handler.strip_and_cache_images` or `_normalize_system`), which
  would make cache-control stripping precede the image step's mutation of `body`.
- Any edit that introduces control flow (loop/early return) reordering the step sequence.

## Checked At
2026-10-05T18:37:49Z

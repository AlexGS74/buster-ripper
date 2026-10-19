VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`normalize`'s image step runs only when the caller passes a truthy `session_id` argument

## What it actually does
In `normalize`, the image step is guarded by the conjunctive condition
`if config.STRIP_IMAGES and session_id and image_handler.has_images(body):`, so a falsy
`session_id` short-circuits the guard and the step never runs — truthiness of `session_id`
is a necessary condition, exactly as claimed. Note the claim is about necessity only: the
guard ALSO requires `config.STRIP_IMAGES` to be truthy and `image_handler.has_images(body)`
to be true, so a truthy `session_id` alone does not make the step run. The parameter is
declared `session_id: str = ""`, so a caller that omits it gets the falsy default and the
step is skipped; the docstring states this explicitly ("Otherwise, images are left
untouched for later handling").

## Source
/Users/alexhubitski/Projects/buster-ripper/buster_ripper/normalize.py — `normalize`, lines 202-214
(guard at line 212; default in the signature at line 202; docstring lines 205-206)

## Source Version
unavailable — the bounded source (single module file) carries no version or revision string;
fingerprint below is the stability basis.

## Source Fingerprint
sha256:a7ddbf062ad57951debb654988853a49a4bae549cb900da544fa3cd5a632ffca

## Invalidates On
- Any edit to `normalize.py` that removes `session_id` from the line-212 conjunct, moves the
  image handling out of that `if` block, or makes image stripping unconditional.
- A change to `buster_ripper/image.py` `has_images`/`strip_and_cache_images` combined with a
  rewritten guard that no longer tests `session_id`.
- A change to `buster_ripper/config.STRIP_IMAGES` semantics only matters jointly with the guard;
  it does not by itself invalidate this claim.

## Checked At
2026-10-05T18:37:49Z

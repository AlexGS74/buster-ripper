VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`click.Choice` rejects values outside its choices list

## What it actually does
In click 8.3.1, `Choice.convert` normalizes the incoming value and searches the normalized choices; when no choice matches it calls `self.fail(self.get_invalid_choice_message(...))` (types.py:344-358), and `ParamType.fail` raises `BadParameter` (types.py:136-143), which click surfaces as a usage error. The class docstring states "The resulting value will always be one of the originally passed choices" (types.py:240).

Nuance: the match is done on *normalized* values — with `case_sensitive=False` or a `token_normalize_func` in the context, a value that differs in case/whitespace from the choice literal can still be accepted (and the original choice is returned). Values with no normalized match are rejected; that is exactly the claim.

## Source
`/Users/alexhubitski/Projects/buster-ripper/.venv/lib/python3.13/site-packages/click/types.py` — `Choice.convert` lines 336-358 (fail on no normalized match), `Choice.get_invalid_choice_message` lines 360-372, `ParamType.fail` lines 136-143 (raises `BadParameter`)

## Source Version
click 8.3.1 (click-8.3.1.dist-info/METADATA)

## Source Fingerprint
sha256:7a4e7804d485c0f2acaad7d3ee3b2a71ce161ee8bc00815530a338a156485e17 (types.py)

## Invalidates On
- click upgrade where `Choice.convert` no longer calls `self.fail`/raises `BadParameter` for non-matching values (e.g. returning the raw value instead)
- click upgrade where `ParamType.fail` stops raising `BadParameter`

## Checked At
2026-09-19

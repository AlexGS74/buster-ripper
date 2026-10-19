VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`_prepare_headers` deletes an `Authorization` header whose value strips to `Bearer` or `Bearer `

## What it actually does
`_prepare_headers` classifies each incoming `Authorization` value with `_carries_token` (chat.py:124-130):
`stripped = value.strip()`, and the value carries a credential only if `bool(stripped) and stripped != "Bearer"`.
A value of `"Bearer"` or `"Bearer "` strips to `"Bearer"`, so `_carries_token` returns `False` for both —
the two spellings in the claim collapse to one condition, and both are true.

When no value carries a token, the fallback at chat.py:115 keeps only a value that strips to empty
(`next((v for v in values if not v.strip()), None)`) — the blank-line case `Authorization:`. `"Bearer"`
does not strip to empty, so it does not qualify and `keep` stays `None`. Then the delete loop runs
unconditionally over every authorization-named key in the forwarded dict (chat.py:117-118) and, because
`keep is None`, the re-add at chat.py:120 is skipped. The value-less bearer line is therefore deleted and
never re-added: `_prepare_headers` returns a dict with no `Authorization` key.

Ordering matters and the code handles it: in the mixed case (`Authorization: Bearer` arriving alongside a
real token), the token-bearing value wins `keep`, all authorization keys are still deleted, and only the
token-bearing value is written back — so the value-less bearer spelling never survives on its own either.
`handle_chat` feeds `_prepare_headers` `request.headers` (chat.py:36) and that result becomes the upstream
`headers=` argument (chat.py:147), so the deletion is the outbound behaviour.

Two limits on the rule, so the hold is not over-read:
- The comparison is case-sensitive string equality against `"Bearer"`. A value of `"bearer"` or
  `"BEARER "` strips to something != `"Bearer"`, so `_carries_token` returns `True` and it is *kept* as
  though it were a credential. The claim's spellings match the literal exactly, so this does not
  contradict it, but the rule is narrower than "empty bearer form".
- The `del` loop operates on the dict returned by `forward_headers(request_headers)` (chat.py:103), which
  lives in `..utils` — outside the bounded source surface supplied for this bundle, so I did not read it.
  The route's own docstring (chat.py:99-102) states that `forward_headers` supplies only hop-by-hop policy
  and that "the empty-`Bearer` rule below is chat-route-local, so `/v1/messages` and the catch-all keep
  forwarding an empty bearer as base did" — i.e. Authorization, empty bearer included, is present in the
  dict this function strips. Even under the opposite assumption, the claim's observable outcome (no
  value-less bearer leaves this function) still holds; only the literal `del` would be a no-op.

## Source
/Users/alexhubitski/Projects/buster-ripper/buster_ripper/routes/chat.py — chat.py:98-121 (`_prepare_headers`), chat.py:124-130 (`_carries_token`), call sites chat.py:36 and chat.py:147

## Source Version
Repo working source, 299 lines; no VCS revision consulted (Git metadata is out of scope for this role)

## Source Fingerprint
sha256:8da1d171930433fcec7f3657e3b668026dc78fe3657a6525f3df3c78c46388c5

## Invalidates On
- Any change to `_carries_token`'s predicate (chat.py:130) — e.g. a case-insensitive compare, prefix matching on `"Bearer "`, or dropping the `bool(stripped)` term — which changes which `Authorization` values are treated as value-less
- Any change to the `keep` selection at chat.py:109-115 that lets a value-less bearer line be chosen as `keep` (e.g. relaxing `not v.strip()`), which would re-add the header instead of deleting it
- Any change to `buster_ripper/utils.forward_headers` that stops forwarding `Authorization` altogether, which turns the literal `del` at chat.py:117-118 into a no-op (the outcome stays, the mechanism does not)
- Any change to `handle_chat` (chat.py:36) or `_make_chat_request` (chat.py:147) that bypasses `_prepare_headers` or overwrites headers after it

## Checked At
2026-10-06T12:16:35Z

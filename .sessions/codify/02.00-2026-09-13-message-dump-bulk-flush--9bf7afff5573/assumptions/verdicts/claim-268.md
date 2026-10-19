VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
The only `normalize` call under `buster_ripper/` passes an empty `session_id`

## What it actually does
`buster_ripper/routes/messages.py:34` is the sole call site of the public `normalize()`
function in the package source root, and it passes the empty string explicitly:
`body, changes = normalize(body, session_id="")`. The callee
(`buster_ripper/normalize.py:202`, `def normalize(body: dict, session_id: str = "")`)
treats a falsy `session_id` as "skip image handling" — line 212 gates image stripping
on `config.STRIP_IMAGES and session_id and image_handler.has_images(body)` — and the
caller then computes the real session id itself on line 35 (`sid = session_id(body)`),
with the comment at lines 31-33 stating normalize runs first to strip the billing nonce
so the id stays stable.

Counting basis (both conjuncts checked, whole row):
- **"the only ... call"**: a case-insensitive scan for `normali[sz]` over every file
  (including hidden ones) under `buster_ripper/` returned 21 matches. Of those, the only
  references to the `normalize` symbol are: the definition (`normalize.py:202`), one
  import (`messages.py:13`, not aliased), the docstring/module-text lines
  (`messages.py:1,25,31,53`; `main.py:77,138`; `utils.py:51`; `__init__.py:1`;
  `normalize.py:1,203`), and one invocation (`messages.py:34`). `main.py`, `server.py`,
  `session.py`, `dump.py`, `image.py`, `backend.py`, `compaction.py`, `config.py`, and
  the other route modules (`chat.py`, `count_tokens.py`, `passthrough.py`) contain no
  reference to it. A grep for `getattr`/`importlib`/`__import__`/`globals(`/`setattr`
  across the package found no matches, so no dynamic dispatch could add a call site.
- **Scope note on the identifier**: the private helpers `_normalize_system`,
  `_normalize_tools`, `_normalize_cache_control`, `_normalize_date` (called at
  `normalize.py:217,220,223,226`) are distinct names and take only `body` — they have no
  `session_id` parameter. They are not counted as `normalize` calls. Reading
  "the only `normalize` call" as "the only call of the public `normalize()` function" is
  the ordinary identifier reading here; the loose substring reading would falsify "only"
  on name-matching alone and is not what the backticked identifier denotes.

## Source
`/Users/alexhubitski/Projects/buster-ripper/buster_ripper/routes/messages.py:34`
(call site), with the signature and empty-string semantics at
`/Users/alexhubitski/Projects/buster-ripper/buster_ripper/normalize.py:202-214`.

## Source Version
local working tree, package source root `buster_ripper/` (no version/revision consulted)

## Source Fingerprint
sha256:721a2afe4d1ca35da9461c95f5c6dfe48dcf04ec4c4c0a606763cc29b0ca96da  (buster_ripper/routes/messages.py)
sha256:a7ddbf062ad57951debb654988853a49a4bae549cb900da544fa3cd5a632ffca  (buster_ripper/normalize.py)

## Invalidates On
- Any change to the argument at `buster_ripper/routes/messages.py:34`
  (e.g. `normalize(body, session_id=sid)` or dropping the keyword) — sha256 above changes.
- A new `normalize(...)` call site anywhere under `buster_ripper/`, or a new aliased
  import of it — re-run the exhaustive `normali[sz]` scan, the fingerprints of touched
  files change.
- A future rename/merge of the `_normalize_*` helpers onto the public name, or a change
  to `normalize()`'s signature making `session_id` required/semantically different
  (lines 202 and 212 in `buster_ripper/normalize.py`).

## Checked At
2026-10-05T18:41:37Z

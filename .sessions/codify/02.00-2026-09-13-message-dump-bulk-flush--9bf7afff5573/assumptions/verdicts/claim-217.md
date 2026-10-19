VERDICT: UNCLEAR
PROMOTION: INELIGIBLE — verdict is UNCLEAR, which is not promotable evidence

## Claim
Every header filter under `buster_ripper/routes/` tests membership against the lowercased header name

## What it actually does
The case-normalisation half of the claim is true everywhere; the "membership" half depends on which code counts as a filter and on whether `==` counts as a membership test.

- `utils.forward_headers` (`utils.py:23`) and `utils.response_headers` (`utils.py:39`) both do exactly what the claim describes: `k.lower() not in (…)` — a membership test against a lowercased name.
- `chat._prepare_headers` (`chat.py:117`), the one name test written inside `routes/`, deletes keys with `[k for k in headers if k.lower() == "authorization"]` — it does test against the **lowercased** name, but by **equality with a single literal**, not membership in a container. Its companion lookup, `request_headers.getall("authorization", ())` (`chat.py:109`), lowercases nothing: it relies on aiohttp's case-insensitive `CIMultiDict`.
- `chat._relay_chat_stream` (`chat.py:179`) contains `if "Content-Type" in resp.headers` — a genuine membership test against a **non-lowercased** (mixed-case) name, again leaning on `CIMultiDict` case-insensitivity rather than normalising.

So the claim is false under one enumeration ("membership" read literally, and/or the `chat.py:179` presence check counted as a header filter) and true under another ("tests membership against the lowercased name" read as "normalises the name to lowercase before comparing it to the drop list"). P9 applies: I had to choose a definition of "membership" / scope "header filter" to call this false, and choosing it loosely to call this true is the same act. The behaviour itself is not in dispute — no filter compares a raw-cased name against a drop list by accident; the disagreement is over the word, not the code.

## Source
- `buster_ripper/routes/chat.py:98-121` (`_prepare_headers`: `.lower() ==` at 117; `getall("authorization")` at 109), `buster_ripper/routes/chat.py:174-183` (`"Content-Type" in resp.headers` at 179)
- `buster_ripper/utils.py:18-24` and `:27-40` (the `k.lower() not in (…)` predicates these routes call)
- `buster_ripper/routes/messages.py:64`, `:285`; `buster_ripper/routes/passthrough.py:17`, `:41`; `buster_ripper/routes/count_tokens.py` (no header test) — exhaustive enumeration of `routes/`

What would settle the claim: the extractor naming which code it counts as a "header filter" (the `routes/`-local deletion loop and any single-header presence check, or only the shared drop-list functions) and whether an equality test against one lowercased literal satisfies "tests membership".

## Source Version
unavailable — git metadata is out of scope for this role; consulted bytes are fingerprinted instead.

## Source Fingerprint
- `buster_ripper/routes/chat.py` — `sha256:8da1d171930433fcec7f3657e3b668026dc78fe3657a6525f3df3c78c46388c5`
- `buster_ripper/utils.py` — `sha256:76de98c1bd07e1cf3b2852d8dd64dd861160f54fe40d2781fce1ad66ae913c72`
- `buster_ripper/routes/messages.py` — `sha256:560a81be69d42552204400a43e2fc5ddec9a983a0b67f165fbe8d7ffee713709`
- `buster_ripper/routes/passthrough.py` — `sha256:c4eef101710b057f0efcf7e99cbe86c48d84f88db9610ec60a17a5fab1b54bbb`
- `buster_ripper/routes/count_tokens.py` — `sha256:a69c9884cad5997e442a4eaa5c5557f13a589b227777b71fbe1c4167c38fa203`

## Invalidates On
- Not applicable to an unresolved claim; re-check if `chat.py:117`'s equality test is rewritten as an `in` test against a collection, or if `chat.py:179`'s `"Content-Type" in resp.headers` is normalised or removed
- Any change to the `k.lower()` predicates at `utils.py:23` or `utils.py:39` (e.g. comparing raw names against a lowercase set)
- A new name test introduced under `buster_ripper/routes/`, which would change the enumeration the claim is scored against

## Checked At
2026-10-06T12:20:51Z

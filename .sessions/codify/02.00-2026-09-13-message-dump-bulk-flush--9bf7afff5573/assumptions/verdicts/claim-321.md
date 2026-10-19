VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`buster_ripper/routes/chat.py`'s case-insensitive `authorization` name lookup iterates the bare keys of the already-filtered header dict rather than `.items()` on a headers mapping

## What it actually does
Both assertions are true of `chat.py:117`, inside `_prepare_headers`:

```python
for name in [k for k in headers if k.lower() == "authorization"]:
    del headers[name]
```

The comprehension's iterable is the dict object `headers` used directly — plain `for k in headers`, no `.items()` anywhere in the file (a `grep` for `items()` across `buster_ripper/routes/` returns no matches) — so Python yields the **bare keys** and applies the case-insensitive comparison `k.lower() == "authorization"` to each key. The matched names are then deleted from that same dict.

The iterated object is the **already-filtered** dict, not the incoming headers mapping: `headers` is bound eight lines earlier at `chat.py:103`, `headers = forward_headers(request_headers)`, where `forward_headers` (`buster_ripper/utils.py:18-24`) returns a dict comprehension that has already dropped `host`, `content-length`, `connection`, `transfer-encoding` and the `_HOP_BY_HOP` names. Its return type is `dict` — a plain dict, not aiohttp's case-insensitive `CIMultiDict` — which is why a case-insensitive comparison has to be written by hand here; the file's own docstring at `chat.py:106-108` states this reason ("``headers``, which is a plain dict").

Which lookup the claim names is not ambiguous: the clause "iterates the bare keys of the already-filtered header dict" picks out line 117, the only construct in the file that iterates the filtered `headers` dict. The other `authorization` access, `values = request_headers.getall("authorization", ())` at `chat.py:109`, performs no iteration in this source and reads the *unfiltered* request mapping, so it is not the lookup described.

## Source
`/Users/alexhubitski/Projects/buster-ripper/buster_ripper/routes/chat.py` — function `_prepare_headers`, lines 98-121 (establishing site line 117; filter site line 103). Supporting type/behaviour locator: `/Users/alexhubitski/Projects/buster-ripper/buster_ripper/utils.py:18-24` (`forward_headers` returns the filtered plain dict). Production source under the supplied source root; no test file consulted.

## Source Version
Working-tree bytes at the supplied source root; no version/revision string consulted — repo state is out of scope for this role.

## Invalidates On
- `chat.py:117` changed to iterate pairs or the unfiltered mapping, e.g. `[k for k, v in headers.items() if ...]` or `[k for k in request_headers if ...]` — the claim's "bare keys" or "already-filtered" predicate would then be false.
- `_prepare_headers` stops filtering before the lookup (line 103 removed, or the lookup moved above it), since "already-filtered header dict" would no longer describe the iterable.
- `forward_headers` changed to return aiohttp's case-insensitive mapping (e.g. a `CIMultiDict`) instead of a plain dict, which removes the need for the hand-written case-insensitive key scan and would likely replace it with `getall`/`pop`.
- The authorization handling extracted into `buster_ripper/utils` so this file only calls a helper — the claim is about this file's own lookup.

## Source Fingerprint
sha256:8da1d171930433fcec7f3657e3b668026dc78fe3657a6525f3df3c78c46388c5  (chat.py — establishing file)
sha256:76de98c1bd07e1cf3b2852d8dd64dd861160f54fe40d2781fce1ad66ae913c72  (utils.py — consulted for what "already-filtered" means)

## Checked At
2026-10-06T12:20:06Z

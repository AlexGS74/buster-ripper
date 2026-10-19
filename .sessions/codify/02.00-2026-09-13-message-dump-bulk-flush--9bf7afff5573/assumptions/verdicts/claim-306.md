VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`_image_cache` is module-level state shared by every request handled in one process

## What it actually does
`_image_cache` is bound once at module top level (`buster_ripper/image.py:14`,
`_image_cache: dict[str, dict[str, tuple[dict, float]]] = {}`) and the name is
never rebound anywhere else in the file — every other touch mutates that single
object in place (`_image_cache[session_id] = {}` at :122, `_image_cache[session_id][image_key] = ...`
at :84, `del cache[key]` at :59/:75, `del _image_cache[session_id]` at :185,
iteration over `_image_cache.keys()` at :178). There is no instance holder, no
thread-local, no `contextvars` copy, and no re-initialisation function, so the
dict is one process-wide object: all callers that import this module in the same
interpreter read and write the same cache, isolated only by the `session_id` key,
not by request or by task. A second OS process (e.g. another worker) would get
its own independent copy, which is exactly the boundary the claim draws with
"one process".

## Source
/Users/alexhubitski/Projects/buster-ripper/buster_ripper/image.py — module body,
line 14 (declaration) plus every mutation site: lines 52, 59, 67, 75, 81, 84,
121-122, 151, 157, 160, 178, 184-185

## Source Version
local working-tree file `buster_ripper/image.py`, 185 lines (no revision string
recorded; fingerprint below is the stability anchor)

## Source Fingerprint
sha256:611df5acd7bf061242084de992a7d84102e0645a5c4b023e0d350841aa32dca7

## Invalidates On
- Any change binding `_image_cache` to a new object per call/request (factory
  function, class instance, `threading.local`, `contextvars.ContextVar`, or a
  reassignment of the name) instead of one module-level dict literal
- Any change moving the cache behind an external store (Redis, disk, in-process
  cache library) so the module no longer owns the state
- Any deployment change making request handling cross-process for a single
  logical request (subprocess-per-request, import refresh per request), which
  breaks the "one process" scope of the shared-state wording
- Editing `buster_ripper/image.py` at all invalidates the fingerprint and
  requires re-reading before reuse

## Checked At
2026-10-05T18:55:46Z

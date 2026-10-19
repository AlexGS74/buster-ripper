VERDICT: DOES-NOT-HOLD
PROMOTION: ELIGIBLE

## Claim
No header filter under `buster_ripper/routes/` drops any name in `utils._HOP_BY_HOP` other than `connection` and `transfer-encoding`

## What it actually does
Header filtering reached from every route drops **all eight** members of `_HOP_BY_HOP`, not just `connection` and `transfer-encoding`. `utils.forward_headers` (used by `chat._prepare_headers`, `messages.py:64`, `passthrough.py:17`) drops `keep-alive`, `proxy-authenticate`, `proxy-authorization`, `te`, `trailers`, and `upgrade` as well — via `*_HOP_BY_HOP` unpacked into the drop tuple — along with `host` and `content-length`. `utils.response_headers` (used by `messages.py:285`, `passthrough.py:41`) drops the same six names plus `content-length`.

The counterexample holds under either enumeration of "a header filter under `routes/`":
- **Only filters defined in files under `routes/`** → the sole one is `chat._prepare_headers` (chat.py:98-121). It returns `forward_headers(request_headers)`, so the mapping it produces excludes `keep-alive`, `te`, `trailers`, `upgrade`, `proxy-authenticate`, `proxy-authorization`. Its output drops those names whether or not the predicate is written locally; `messages.py:63` describes the same call as "Build headers, removing hop-by-hop".
- **Filters applied by `routes/`, shared helpers included** → `forward_headers` and `response_headers` each drop all six names directly.

Nothing under `routes/` drops a hop-by-hop name in an isolated two-name list; the only `routes/`-local name test is `authorization`, which is not in `_HOP_BY_HOP` at all.

## Source
- `buster_ripper/utils.py:12-15` (`_HOP_BY_HOP` members), `:23` (`forward_headers` drop tuple `("host", "content-length", "connection", "transfer-encoding", *_HOP_BY_HOP)`), `:39` (`response_headers` drop tuple `("content-length", *_HOP_BY_HOP)`)
- `buster_ripper/routes/chat.py:103` and `:117-118`, `buster_ripper/routes/messages.py:64`, `:285`, `buster_ripper/routes/passthrough.py:17`, `:41`

Note on the source boundary: the claim's subject is the filters under `buster_ripper/routes/` (the supplied source root), but the term `utils._HOP_BY_HOP` is named verbatim in the claim, so `buster_ripper/utils.py` was read to resolve that term's members. No search outside the supplied root and that one claim-named module was performed.

## Source Version
unavailable — git metadata is out of scope for this role; consulted bytes are fingerprinted instead.

## Source Fingerprint
- `buster_ripper/utils.py` — `sha256:76de98c1bd07e1cf3b2852d8dd64dd861160f54fe40d2781fce1ad66ae913c72`
- `buster_ripper/routes/chat.py` — `sha256:8da1d171930433fcec7f3657e3b668026dc78fe3657a6525f3df3c78c46388c5`
- `buster_ripper/routes/messages.py` — `sha256:560a81be69d42552204400a43e2fc5ddec9a983a0b67f165fbe8d7ffee713709`
- `buster_ripper/routes/passthrough.py` — `sha256:c4eef101710b057f0efcf7e99cbe86c48d84f88db9610ec60a17a5fab1b54bbb`

## Invalidates On
- Any change to `_HOP_BY_HOP` members in `utils.py:12-15` (adding or removing a name changes which names the counterexample covers)
- Any change to the drop tuples at `utils.py:23` or `utils.py:39` that stops unpacking `*_HOP_BY_HOP` (e.g. replacing it with an explicit `connection`/`transfer-encoding` pair) would make the claim true
- `chat._prepare_headers` ceasing to delegate to `forward_headers`, or any route stop calling `forward_headers`/`response_headers`
- A new name-ignoring filter introduced under `routes/` that is not covered by the enumeration above

## Checked At
2026-10-06T12:20:51Z

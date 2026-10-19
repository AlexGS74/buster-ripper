VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
Unpacking a three-element tuple into two targets raises `ValueError`

## What it actually does
For a plain (non-starred) target list whose length differs from the iterable's
length, the interpreter raises `ValueError`. Unpacking `(1, 2, 3)` into two
targets consumes the two items, finds a third, and raises
`ValueError("too many values to unpack (expected 2, got 3)")` — the tuple/list/dict
fast path reports the actual length because `Py_SIZE(v)` is known. The specialised
`UNPACK_SEQUENCE_TUPLE` bytecode does not raise itself; on a size mismatch it
deoptimises into the generic `_UNPACK_SEQUENCE` → `_PyEval_UnpackIterableStackRef`
path, which is where the `ValueError` is raised. Nothing is bound when it is raised.

Scope of the reading (not hedging — the claim is true as stated, but the boundary
is worth naming): "two targets" here means two ordinary assignment targets.
`a, *b = (1, 2, 3)` is a different pattern and does not raise — with a starred
target (`argcntafter != -1`) the arity requirement is "at least", and `b` becomes
`[2, 3]`. `TypeError` is raised only when the right-hand object is not iterable at
all, which is not this case.

## Source
Upstream CPython `Python/ceval.c`, function `_PyEval_UnpackIterableStackRef`
(lines ~2390–2462 in the fetched bytes), branch `3.14`, retrieved from
`https://raw.githubusercontent.com/python/cpython/3.14/Python/ceval.c`.
Corroborated by `Python/bytecodes.c` `_UNPACK_SEQUENCE` / `_UNPACK_SEQUENCE_TUPLE`
(`DEOPT_IF(PyTuple_GET_SIZE(seq_o) != oparg)`) from
`https://raw.githubusercontent.com/python/cpython/3.14/Python/bytecodes.c`.
Corroborated by the official docs, which state the requirement but do **not** name
the exception: Python 3.14.8 Language Reference §7.2 Assignment statements —
"Else: The object must be an iterable with the same number of items as there are
targets in the target list" (https://docs.python.org/3/reference/simple_stmts.html),
plus §5.3 of the tutorial ("Sequence unpacking requires that there are as many
variables on the left side of the equals sign as there are elements in the
sequence"). Behaviour also confirmed in upstream tracker reports
(bugs.python.org #23949, #40202, #3071; python/cpython #128660, which shows
`a, b = (1, 2, 3)` → `ValueError: too many values to unpack` on 3.14).

Evidence basis: interpreter source. No `Source root`/`Source snapshot` was supplied
and none was needed to bound the search — this is a language-semantics claim settled
by exact upstream URLs. The project was not executed (P1); no local environment,
interpreter, or package cache was inspected.

## Source Version
CPython `3.14` branch raw source; docs page version 3.14.8. Exact interpreter patch
version of the repo's runtime was not established (no source root supplied, and
determining it would require an environment probe).

## Source Fingerprint
sha256:7b0206ef3aee2369cfa7c952fb31de2a3aa4b438ccdc0d7fae85b25642b28fe2 (ceval.c, 116772 bytes, 3728 lines)
sha256:3b4f8d958b7f24f741f8a3fd17ede2244986f1197032f0ec95f357208bc32e67 (bytecodes.c, 226021 bytes)

## Invalidates On
- The fetched `Python/ceval.c` bytes change at `sha256:7b0206ef…` (any edit to
  `_PyEval_UnpackIterableStackRef`, including a change of exception class or of the
  `too many values to unpack` arity check).
- The fetched `Python/bytecodes.c` bytes change at `sha256:3b4f8d95…` (tuple
  unpack specialisation raising or deoptimising differently).
- CPython changes sequence-unpacking arity mismatch from `ValueError` to another
  built-in (e.g. folding it under `LookupError`/`TypeError`) in a release after 3.14.
- The Language Reference §7.2 or Built-in Exceptions page is revised to name a
  different exception for target-list arity mismatch.
- Reuse against a non-CPython implementation (PyPy, GraalPy) or a Python 2 runtime
  where the unpacking code path is not this source.

## Checked At
2026-10-05T19:00:42Z

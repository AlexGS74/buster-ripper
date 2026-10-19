VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`routes/messages.py`'s `handle_messages` wraps its `dump_turn` call in a swallow-all `try/except Exception` that logs

## What it actually does
The `dump_turn(body)` call is the only statement in a `try` whose sole handler is
`except Exception: request.app["log"].exception("dump failed")` — it catches every
non-`BaseException` failure, writes it to the app logger (with traceback, at ERROR
level), and neither re-raises nor returns a value, so control continues to the
header/streaming code below. The wrapper is reached only when
`config.DUMP_DIR is not None`.

Source form (`buster_ripper/routes/messages.py` lines 57–61):

```python
if config.DUMP_DIR is not None:
    try:
        dump_turn(body)
    except Exception:
        request.app["log"].exception("dump failed")
```

`dump_turn` is imported at module level from `..dump` (line 12), so the call in the
`try` block is that function. "Swallow-all" matches in both senses checked here: the
handler catches the widest exception class Python handlers normally take, and it
discards the error after logging it. `BaseException` subclasses (`KeyboardInterrupt`,
`SystemExit`, `asyncio.CancelledError`) are not caught by this arm.

## Source
/Users/alexhubitski/Projects/buster-ripper/buster_ripper/routes/messages.py —
`handle_messages`, lines 57–61 (and the `from ..dump import dump_turn` at line 12)

## Source Version
working-tree file content, no revision string consulted (repository state is out of
scope for this role); fingerprint below identifies the exact bytes

## Source Fingerprint
sha256:560a81be69d42552204400a43e2fc5ddec9a983a0b67f165fbe8d7ffee713709

## Invalidates On
- the `except Exception` arm around `dump_turn` is narrowed to a specific exception class, or gains a `raise`/`return`, or is removed
- `dump_turn(body)` is moved out of the `try` block, or the call is relocated to another function or deferred to a later bulk flush outside this handler
- the handler body is replaced by something other than a logging call (e.g. a metric counter, a silent `pass`) or the `log.exception` is downgraded so errors are no longer logged
- the `if config.DUMP_DIR is not None:` guard is removed or changed so the wrapped call no longer sits at this site

## Checked At
2026-10-06

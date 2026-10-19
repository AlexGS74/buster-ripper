VERDICT: SAFE
PROMOTION: ELIGIBLE

## Seam
`S-YR-2` — the exception a flush can raise at the append-time volume trigger, produced inside `dump_turn` (`dump.py:205-206`), crossing the `handle_messages` swallow (`routes/messages.py:57-60`). Boundary value: a raise originating in `flush_all()` under `_append_record`'s threshold branch, observed at the request handler's `except Exception` arm — i.e. whether that raise reaches the request at all, or is absorbed before it can fail the request.

## Composition
**Producer.** `_append_record` (`dump.py:200-206`) buffers the record, adds the rendered-text UTF-8 byte count, and when `state.pending_bytes >= config.DUMP_FLUSH_BYTES` calls `flush_all()` at `dump.py:206`. `_append_record` is reached only from `dump_turn` (`dump.py:105` for `change-focused`, `dump.py:107` for `self-contained`), so the threshold flush executes inside `dump_turn`'s call tree.

**Transformations.**
1. `flush_all()` (`dump.py:112-122`) returns immediately when `config.DUMP_DIR is None`; otherwise it iterates `_sessions.values()` and calls `_flush_session(state)` for every session. It has no `try` of its own, so anything `_flush_session` raises would propagate from here.
2. `_flush_session()` (`dump.py:285-292`) returns when `state.pending` is falsy; otherwise it runs `try: _flush_state(state)` / `except Exception: log.exception(...)`. This is where every ordinary flush failure is absorbed: the `ValueError` raised by `_flush_state` at `dump.py:301` (non-regular artifact target) and any `OSError` from `path.open("a")` / `f.write(...)` at `dump.py:302-303` are logged and the batch is retained (`state.pending.clear()` and `state.pending_bytes = 0` at `dump.py:304-305` run only after a successful write).
3. If an `Exception` nonetheless escaped `flush_all()`, it would propagate synchronously through `_append_record` → `dump_turn` (a plain `def`, `dump.py:94`) → the call at `routes/messages.py:58`, which sits inside `try:` at `routes/messages.py:57` whose arm is `except Exception:` at `routes/messages.py:59`. The arm logs at `routes/messages.py:60` and does not re-raise.
4. **Consumer.** The handler's `except Exception` arm (`routes/messages.py:59-60`). Execution then falls through to `routes/messages.py:62-70` (header build, stream dispatch, `return await handle_streaming(...)` / `handle_non_streaming(...)`), so the request is settled exactly once on the normal path.

The flush's only escape class is `Exception`; the handler catches `Exception`; and `_flush_session` already absorbs that same class one frame below. The inner flush is therefore covered twice over, and no raise from it can fail the request.

## Dimension Coverage
1. **Stack continuity between the swallow and the inner flush.** `dump_turn` (`dump.py:94`), `flush_all` (`dump.py:112`), `_append_record` (`dump.py:200`), `_flush_session` (`dump.py:285`) and `_flush_state` (`dump.py:295`) are all plain `def`s. No `await`, `create_task`, callback, executor, or thread sits between `routes/messages.py:58` and the flush. A grep under `buster_ripper/` for `to_thread|run_in_executor|ThreadPool|threading|executor` finds no matches. A raise from the inner flush is delivered on the same stack as the handler's `try` and cannot bypass it.
2. **Exception class coverage at the inner flush.** `_flush_state` can originate `ValueError` (`dump.py:301`) and `OSError` from `Path.open("a")` / `f.write` (`dump.py:302-303`); both are `Exception` subclasses, caught by `_flush_session` at `dump.py:291`. The flush code originates no `BaseException` subclass.
3. **Exception class coverage at the handler.** `except Exception` at `routes/messages.py:59` covers every `Exception` `dump_turn` can raise, including a hypothetical one escaping `flush_all`. It does not cover `BaseException` (`KeyboardInterrupt`, `SystemExit`, `GeneratorExit`, `asyncio.CancelledError`); none of those is produced by the synchronous flush path (see Notes).
4. **Other flush entry points reachable from a request.** `flush_all` has exactly three call sites: `dump.py:133` (`poll_flush`, background task), `dump.py:206` (`_append_record`, under `dump_turn`), and `server.py:112` (`_final_dump_flush`, cleanup). Only `dump.py:206` is reachable from a request handler; the other two are app lifecycle. `dump_turn` has exactly one production call site: `routes/messages.py:58`.
5. **Other dump/flush calls on the request path.** A grep under `buster_ripper/` for `dump_turn|flush_all|_flush_state|_flush_session|poll_flush|_final_dump_flush` finds no dump/flush call in `routes/chat.py`, `routes/count_tokens.py`, `routes/passthrough.py`, `server.logging_middleware`, or anywhere else in the package's route modules outside `routes/messages.py:58`. The stats write on the same path (`update_session_stats` at `routes/messages.py:242` and `:319`) is not a dump/flush call and carries its own `except Exception` at `session.py:84-85`.
6. **Concurrent timer flush.** `poll_flush` (`dump.py:125-136`) runs as a separate task (`server.py:45`), wraps each `flush_all()` in its own `except Exception` (`dump.py:132-135`), and its failures stay in that task — they cannot propagate into a request task. `flush_all` and `dump_turn` are synchronous, so they cannot interleave with each other within one event-loop turn; a request never observes a half-executed timer flush.
7. **Re-entrancy.** `flush_all` → `_flush_session` → `_flush_state` never calls `_append_record` or `_lookup_or_create_state`; a flush cannot trigger a second flush or create a session.
8. **`_sessions` iteration stability.** No flush helper mutates `_sessions` — only `_lookup_or_create_state` does, at `dump.py:156` — and the synchronous loop cannot be suspended by another coroutine, so the threshold flush cannot raise `RuntimeError: dictionary changed size during iteration`.
9. **Threshold operand.** `config.DUMP_FLUSH_BYTES` is an `int` (`config.py:37`), so the comparison at `dump.py:205` cannot raise a `TypeError` on the default path; even a raise there would sit in `_append_record`, inside `dump_turn`, hence under the same handler `try`.
10. **Handler continuation and settlement.** The `except Exception` arm at `routes/messages.py:59-60` neither returns nor re-raises; lines 62-70 proceed to headers and stream/non-stream dispatch, and no dump/flush call follows. The request is settled exactly once on the normal path; a flush raise cannot leave it unsettled or doubly settled.

## Counterexample
None found. Every member of the seam's closed exception union — `ValueError` from the artifact-target guard, `OSError` from the append open/write, and any other `Exception` that could hypothetically escape `flush_all` — either is absorbed by `_flush_session`'s `except Exception` (`dump.py:291-292`) or, if it escaped that, lands on the same stack inside `routes/messages.py:57`'s `try` and is caught at `routes/messages.py:59`. The append-time volume flush is not a separate call site outside the swallow; it is one synchronous frame deeper than the swallowed `dump_turn(body)` call.

## Notes
- **`BaseException` boundary (inert).** Neither `except Exception` arm (`routes/messages.py:59`, `dump.py:291`) catches `KeyboardInterrupt`, `SystemExit`, `GeneratorExit`, or `asyncio.CancelledError`. The flush path originates none of these: `_flush_state` raises only `ValueError` (`dump.py:301`) or I/O errors, `dump_turn` has no `await` where cancellation could be delivered, and a signal-delivered `KeyboardInterrupt` is process-level, not the flush raising. `Reach: no production caller found` for a `BaseException` originating from the flush itself.
- **Partial-write retention (out of this seam's boundary value).** If `f.write` at `dump.py:303` raises after a partial append, `_flush_session` logs and the buffer is not cleared (`dump.py:304-305` skipped), so a later flush re-appends the whole batch and can duplicate the partially written prefix. This is an artifact-integrity consequence, not a request failure. `Reach:` `flush_all` from `_append_record` at `dump.py:206` and from `poll_flush` at `dump.py:133`.
- **Format-change stranding (out of this seam's boundary value).** If `state.fmt` moves from a collapsed value to `documented` between turns, `_flush_state` returns at `dump.py:297-298` without clearing the old pending batch, stranding those records. Again not a request failure. `Reach:` `_flush_state` called from `_flush_session` from `flush_all` at the threshold (`dump.py:206`), timer (`dump.py:133`) and shutdown (`server.py:112`).
- **Swallow's own arm (inert).** The catch body reads `request.app["log"]` (`routes/messages.py:60`); `on_startup` installs that key at `server.py:64` before any request is served, so no production caller reaches the arm with the key absent. Standard `logging` absorbs handler errors rather than raising them, so the logging call cannot turn the swallow into a request failure. `Reach: no production caller found`.
- **Ledger note (inert).** The `S-YR-2` row is not present in `assumptions/seams.md`; this composition uses the seam row supplied with the dispatch (members `YR-P007`, `YR-P008`; boundary: request-path swallow → threshold flush inside the dumped call). Both member verdicts are `HOLDS`, so the seam is composable.

## Sources
- `buster_ripper/routes/messages.py:56-70` — the guarded `dump_turn(body)` call and its `except Exception` arm; `:224-229` post-commit relay arm (not a dump call); `:242`, `:319` stats writes; `:332-334` upstream `ClientError` arm.
- `buster_ripper/dump.py:94-122` — `dump_turn`, `flush_all`, `poll_flush`; `:200-206` — `_append_record` and the threshold `flush_all()`; `:285-305` — `_flush_session` and `_flush_state`.
- `buster_ripper/server.py:37-45` — `spawn_dump_flush`; `:84-114` — `on_cleanup`, `_cancel_background_task`, `_final_dump_flush`.
- `buster_ripper/session.py:74-85` — `update_session_stats` and its own `except Exception`.
- `buster_ripper/config.py:20,27-28,36-37` — `DUMP_DIR`, `DUMP_FORMATS`, `DUMP_FORMAT`, `DUMP_FLUSH_INTERVAL`, `DUMP_FLUSH_BYTES`.

## Source Versions
- `buster_ripper/routes/messages.py`, `dump.py`, `server.py`, `session.py`, `config.py`: 0.1.0 (`pyproject.toml`); working-tree files, revision unavailable.

## Source Fingerprint
sha256:0eca48fb856110f9d12777f9fe3f6346c8060e2b0d999ddd1a9fe134a3a7e2d4

Computed over this exact ordered manifest:

```text
member assumptions/verdicts/yr-p007.md
member-sha256 6eaa5bd6ab638f97feea9c9f64a93b7808e8d5f7fb9c22ed72d2b82f1389b44c
member-verdict HOLDS
member-source-fingerprint sha256:64be411b2c2172354528ff46f1b394a29d3dcef949dfa199bd06e3c4e08b6251 buster_ripper/routes/messages.py
member assumptions/verdicts/yr-p008.md
member-sha256 b80fb44cd862dbc3d7577562ea60357df990d78db3c21601dd3091c42e906932
member-verdict HOLDS
member-source-fingerprint sha256:536ac10518ab44cce0ff09de984a89912a920bc67b27482cfb34f526123a95fc buster_ripper/dump.py
consulted buster_ripper/routes/messages.py sha256:64be411b2c2172354528ff46f1b394a29d3dcef949dfa199bd06e3c4e08b6251
consulted buster_ripper/dump.py sha256:536ac10518ab44cce0ff09de984a89912a920bc67b27482cfb34f526123a95fc
consulted buster_ripper/server.py sha256:b3b486615ca797febfa86795e0e18a04e4355aece31c43305a8d40bcad0e28d5
consulted buster_ripper/session.py sha256:67a66b8aa733b31e982be4922350a7996ebdddb0fe5d391a6129a1cf342a9dff
consulted buster_ripper/config.py sha256:22348c68d21b91a30e75eeea802d4d390b2bcdabbf974b0ddc743c9354635522
```

## Invalidates On
- Either member verdict leaves `HOLDS`, or either member's pinned source fingerprint moves: `routes/messages.py` `64be411b…` (the swallow) or `dump.py` `536ac105…` (the threshold flush).
- `dump_turn(body)` moves out of the `try` at `routes/messages.py:57-58`, or the handler is narrowed from `except Exception` at `:59`.
- `_append_record` stops calling `flush_all()` at `dump.py:206`, or the threshold branch stops executing inside `dump_turn`'s call tree (e.g. the flush is hoisted to a task, callback, executor, or thread).
- A dump/flush call is added to a request path outside the `dump_turn` `try` (`routes/messages.py`, `server.logging_middleware`, or another module reached by `/v1/messages`).
- `flush_all`/`_flush_session` stops absorbing `Exception` (removing the inner cover), or `_flush_state` gains a `BaseException`-raising path that escapes both `except Exception` arms.
- `_sessions` gains a mutation path inside `flush_all`'s iteration that can raise outside `_flush_session`'s `try`, or `config.DUMP_FLUSH_BYTES` stops being an `int` threshold operand.

## Checked At
2026-10-17T22:41:59Z

VERDICT: SAFE
PROMOTION: ELIGIBLE

## Seam

`S19` — members `126, 408, 415, 410, 411` (ledger order).

Boundary value: the `/metrics` response a test double hands across `async with session.get(...)` into the
production poll loop, **and the witness's own right to run that loop more than once**.

Question (the seam row carries no literal `Question:` label, so the question is its boundary value plus its
six dimension clauses): do the verified double-behaviour, yield, collection and cancellation facts jointly
permit a witness that (i) actually delivers a `/metrics` payload to the branch at
`buster_ripper/compaction.py:40-54`, (ii) reaches the EMA arm at `:48-52` at least twice in one call, and
(iii) settles honestly when the driven loop is stopped?

This is the **post-repair recomposition** required by this record's own `Invalidates On` 7 ("T020's witness
lands: recompose over the landed `tests/test_compaction.py` bytes"), which run 44's ledger amendment fired by
moving `S19`'s State to `recheck`. It is the second composition of this seam; the allowance run 42 recorded as
unspent is spent here.

Preconditions: every member read `State: HOLD` / `Routing: none` in `claims.md` (rows at lines 392, 1996,
1998, 1999, 2224) **before** any verdict file was opened; each supplied verdict file starts with
`VERDICT: HOLDS`. Row 409 (the falsified 408-against-409 premise, retired by run 41's amendment) was not
consumed. No plan, interview, handoff, iteration or QA artifact was opened. The project was not run.

Answer against each member's `Verdict:` —

| Member | Verdict / handle | What it licenses now |
|---|---|---|
| 126 `HOLD` `CL:python/cancelled-child-await-propagation` | `VERDICT: HOLDS` | The settle at `tests/test_compaction.py:393-394` is honest: the seam requests cancellation from inside the task, the task terminates cancelled, and `await`ing it raises `CancelledError` **in the awaiter**, which `contextlib.suppress` catches. A swallowed cancellation cannot read as a pass, because the pass signal is now the value list, not the absence of an exception. |
| 408 `HOLD` `CL:python/asyncmock-async-context-manager` | `VERDICT: HOLDS` | The `as` target is `__aenter__.return_value`; `__aexit__` is not a suppressor. `_kv_answer` (`:386-387`) sets both explicitly — `return_value=resp`, `return_value=False` — so the configured object *is* the loop's `resp`. Closes CE-1 at the construction level. |
| 415 `HOLD` `CL:python/magicmock-async-context-manager` | `VERDICT: HOLDS` | Confirms the converse hazard is real (an unconfigured mock still enters, silently binding a child), which is why the explicit self-binding is load-bearing and why the witness's `text.await_count` assertion (`:433`, `:450`) is the tripwire that makes the discarded-child shape go red. |
| 410 `HOLD` `CL:python/asyncio-sleep-yields-to-loop` | `VERDICT: HOLDS` | `config.KV_POLL_INTERVAL = 0` (`:413`) before the task exists makes `:57` a zero-delay suspension, so the loop is re-entered. 410 is now used as a *yield* premise, not as a clock — the CE-3 misread is gone. Its limit ("allows other tasks to run", not "a particular task will run") no longer bears: the second tick does not need another task, and the run's end is bounded by `wait_for`, not by a wall-clock window. |
| 411 `HOLD` `CL:pytest-asyncio/auto-mode-async-test-collection` | `VERDICT: HOLDS` | The two new unmarked module-level `async def` witnesses (`:423`, `:436`) are collected and run on a loop; `asyncio_mode = "auto"` is unchanged at `pyproject.toml:18` and the plugin bytes are unchanged. The delta's new helpers (`_TickSeam`, `_kv_answer`, `_settle`, `_drive_ticks`) are awaited or plain — none is collected — so no un-awaited coroutine is left behind. |

## Composition

Producer → transformations → consumer, for the value the seam names, on the landed bytes:

1. **Producer (test-side).** `session = mocker.Mock(spec=aiohttp.ClientSession)`;
   `session.get = mocker.Mock(side_effect=seam.get)` (`tests/test_compaction.py:410-411`). `seam.get` is a
   *plain bound method*, not a mock, so the **result of the call** is exactly one answer object per tick:
   `self.answers.pop(0)` (`:368`), where `answers[i]` is a distinct `_kv_answer(...)` (`:407`). No spec'd
   auto-child sits between the loop and the configured payload — CE-1's ordering variant is structurally
   excluded, not merely avoided.
2. **Transformation 1 — the response's own construction (`_kv_answer`, `:371-388`).** `resp =
   mocker.AsyncMock()`; `resp.status = 200` (an int attribute read synchronously);
   `resp.text = mocker.AsyncMock(return_value=...)` (an awaitable child);
   `resp.__aenter__ = mocker.AsyncMock(return_value=resp)`;
   `resp.__aexit__ = mocker.AsyncMock(return_value=False)`. Per 408/415 the `as` target is
   `__aenter__.return_value`, which is *the configured object itself*, so the two kinds of qualification the
   loop demands (`status` un-awaited at `:40`, `text()` awaited at `:41`) are both satisfied on the object
   the loop binds.
3. **Transformation 2 — the gate and the awaited arm.** `compaction.py:40` compares the configured `200`;
   `:41` awaits the configured `text`, whose call result is awaited once per answer. The witness observes
   that await *on the answer object itself* (`answers[i].text.await_count`), which is only 1 if the awaited
   `text` was the configured one — the CE-1/CE-2 tripwire.
4. **Transformation 3 — the two arms of one state variable.** `:45-47` is the `first` arm (assign raw, clear
   the flag); `:48-52` is the blend arm; `first` is function-local (`:31`), so the blend arm requires a
   second iteration **of the same invocation** — which is what one `_drive_ticks` call now supplies.
5. **Transformation 4 — the yield.** `:57` `await asyncio.sleep(config.KV_POLL_INTERVAL)`; the test set that
   module attribute to `0` at `tests/test_compaction.py:413`, *before* `asyncio.create_task(...)` at `:414`,
   and restores it in the same `finally` at `:419`. `config.py:64`'s default is still
   `KV_POLL_INTERVAL: int = 10`, unchanged bytes (`22348c68d21b91a3…`), so production ships the 10 s interval;
   only this witness's duration runs at 0.
6. **Termination, not wall-clock.** When the queued answers are exhausted, `seam.get` calls
   `self.task.cancel()` and raises `aiohttp.ClientError` (`:366-367`); production swallows it at `:55-56`,
   reaches `await asyncio.sleep(0)` at `:57`, and the pending cancellation is delivered there. `_settle`
   (`:391-394`) awaits that termination under `wait_for(..., timeout=2.0)`.
7. **Observation point.** `_TickSeam.get` samples `compaction._kv_cache_usage` *at the top of every call*
   (`:364`) and `_drive_ticks` returns `seam.observed[1:]` (`:420`) — i.e. the value production had written
   after each completed tick, with the final value sampled by the extra (terminating) call. For
   `[0.75, 0.0, 0.23]` that is `[0.75, 0.525, 0.4365]`, and those are the literals asserted at `:448`, in
   production's operand order `(1-α)·prev + α·raw` with α from `:25`.
8. **Consumer.** `compaction._kv_cache_usage` is read by production at
   `buster_ripper/routes/count_tokens.py:31` (`_nudge_trigger_name`, defined at `:29`), `:52` (the
   `should_nudge_compact` call inside `_nudge_tokens`, `:43`) and `:59` (the nudge log line), which decide
   whether the reported token count is inflated to the compaction target. The loop is spawned in production
   by `buster_ripper/server.py:22` (`await compaction.poll_kv_cache(session)`) via
   `spawn_kv_cache_monitor` (`:25-34`), called from `on_startup` at `:78`.

## Dimension Coverage

Delta-scoped re-trace: the unit of scoping is member-anchor intersection with the change, and the change is
the witness side of the boundary (`tests/test_compaction.py` `4323709589fa94c0…` → `ae79036b7fe714c2…`) plus
the consumer files above. All five member verdicts remain current — the `lang:`/`pkg:` sources they pin are
untouched by the CR3 wave, `.venv/…/pytest_asyncio/plugin.py` is still `121cb98d74596a9e…` and
`pyproject.toml` still `c623bdc282433193…` — so this stays a scoped recomposition rather than a full trace.

| # | Dimension (seam row) | Re-traced / carried | Result |
|---|---|---|---|
| 1 | awaitable call result versus an async-context-manager object at one call site (408 against 415) | **re-traced** (witness bytes moved) | Composes. The call result is a configured `_kv_answer` object; the `as` target is made identical to it by `:386`. CE-1 closed. |
| 2 | the response attributes the loop reads without awaiting (`status`) versus the one it awaits (`text()`) | **re-traced** | Composes, and the split is now *asserted* (`:433`, `:450`) rather than assumed. CE-2 closed. |
| 3 | which double the file's sibling poll tests already use, and whether that shape reaches the production branch at all | **re-traced** | `_kv_answer` is the siblings' own shape (`:193-196`, `:243-246`, `:268-271`, `:298-301`) lifted into one factory. "Reaches" is now falsified-or-confirmed by the value assertions, whereas the prior record had to report it as unfalsified. See Notes N5 for the four siblings that still do not observe the branch. |
| 4 | where the driven loop yields so a second tick is reached (410) against the interval the loop sleeps on | **re-traced** | Composes via interval mutation (`:413`/`:419`), with 410 used as a yield premise. CE-3 closed. Residual unrowed atoms N1/N2 below — both fail loud. |
| 5 | the test function's own collection and event-loop eligibility (411) | **carried**, with the delta's one new sub-case traced | Carried: `asyncio_mode = "auto"` (`pyproject.toml:18`, `c623bdc2…`) and the plugin bytes (`121cb98d…`) are unchanged, and 411's verdict is current. New sub-case traced: the delta added four non-test coroutines/callables — `_kv_answer` (sync factory), `_settle` (`:391`) and `_drive_ticks` (`:397`) (both awaited from inside a test, hence never collected: the names do not start with `test`), and `class _TickSeam` (`:349`, not collected: the name does not start with `Test`) — so 411's substitution reaches only `:423` and `:436`, and no coroutine is left un-awaited. |
| 6 | cancelling and awaiting a coroutine that never returns, so a swallowed cancellation cannot read as a pass (126) | **re-traced** (the settlement idiom changed: in-task cancel, `wait_for` bound, `contextlib.suppress`) | Composes; see the Settlement paragraph under **Counterexample**. The pass signal is the sampled value list, and the failure signal for a loop that will not stop is `TimeoutError`, which the suppress does not cover. |

## Counterexample

**None found.** Each of the three previously reported counterexamples is closed, and re-tracing the same
dimensions over the landed bytes surfaced no new one. Reported per finding, with what closes it and what
would now happen if it recurred:

**CE-1 — `async with` succeeds and the payload is discarded (dims 1, 3): CLOSED.** Two independent closures.
*Construction*: `resp.__aenter__ = mocker.AsyncMock(return_value=resp)` (`:386`) makes the `as` target the
configured object, which is exactly what 408/415 say the binding rule is; the answer reaches `:40-41` rather
than being discarded at the door. *Detection*: if the self-binding were ever removed, 415 says the call still
enters and binds `__aenter__.return_value` — an auto-child — so the child's `.status` would not equal `200`,
`:41-54` would not run, and the witness would fail **twice**: `observed[1:]` stays `[0.0]`/`[0.75, 0.75,
0.75]` against the asserted values (`:432`, `:448`), and `answers[i].text.await_count` would be `0` because
the awaited `text` was the child's, not the configured one (`:433`, `:450`). The prior record's fatal
observation — "`session.get.called` is True and `_kv_cache_usage` is **unchanged**", which passed under the
file's vocabulary — is now the failing case. My prior Note N2 (the unrowed `unittest.mock.__eq__` atom that
separated CE-1's two routes) no longer bears on the verdict: whichever route `:40` takes, `:44-52` does not
run, and not running is now red.

**CE-2 — the sync-read versus awaited-call split deleted by the swallow arm (dim 2): CLOSED.** The split
still exists in production (`:40` un-awaited, `:41` awaited) and the swallow still exists at `:55-56`; what
changed is that the witness now asserts on **both** sides of the split rather than on neither. `status` is
configured as a plain `200` (`:383`) for the synchronous read, `text` as an `AsyncMock` (`:384-385`) for the
awaited call, and `answers[i].text.await_count == 1` is asserted per tick (`:433`, `:450`). Build `text` as a
*synchronous* `Mock` instead and `await resp.text()` raises `TypeError` inside the `if` body, swallowed at
`:55-56` — the tick then contributes no write, so the value assertions fail, and `answers[i].text.await_count`
on a plain `Mock` is an auto-created child that does not compare equal to `1`, so the channel assertion fails
too. The mixed-kind build that the same file used for `session.get` (`:196`, `:246`, `:271`, `:301`) is now
the *forbidden* shape for the response, and `_kv_answer`'s docstring (`:371-381`) states that list of
forbidden shapes; the witness can detect every item on it.

**CE-3 — tick two unreachable because a task parked on a 10 s timer is never ready inside the yield (dim 4):
CLOSED.** `config.KV_POLL_INTERVAL = 0` is written at `:413` before `asyncio.create_task` at `:414` and
restored at `:419`, so `:57`'s sleep is a zero-delay suspension and the loop *is* re-entered; three readings
produce four `get` calls and two visits to the blend arm. The prior record's missing contract ("what makes
`:57`'s 10-second interval elapse inside a test, and whether mutating `config.KV_POLL_INTERVAL` before the
task is created is the sanctioned route") is answered on its first half by the source itself; the second half
— *sanctioned* — is a plan question and was not read or decided here. The arithmetic is now production's and
is falsifiable: the literals `[0.75, approx(0.525), approx(0.4365)]` come from
`(1-0.3)·prev + 0.3·raw` in `:50-51`'s operand order, so a weight swap gives `0.225` where `0.525` is
asserted, and deleting `:48-52` (the missing line these witnesses exist to catch) holds the global at `0.75`
for ticks 2 and 3 — both red, both for the arithmetic, neither for a missing tick. And unlike the four
older payload tests, neither new witness asserts "the value is no longer `0.0`", so the seeding arm can no
longer stand in for the blend arm.

**Settlement (dim 6): no counterexample, and the idiom changed materially — re-traced, not carried.** The
cancel is now requested from *inside* the task (`:366`), by the seam, rather than by the test after a fixed
window, and the awaiter is `wait_for(task, 2.0)` under `contextlib.suppress(CancelledError)` (`:393-394`).
The ordering composes: `seam.get` samples the global, requests the cancellation and raises
`aiohttp.ClientError` synchronously at `:36`'s call; there is no await between the raise and
`except Exception: pass` at `:55-56`, so the pending cancellation is delivered at the first suspension point
after it — `:57`, which sits **outside** the `try`, where nothing can swallow it. `126` then carries the last
step: a child that terminates cancelled raises in the awaiter, and the suppress makes that the *expected*
outcome. The honesty property the dimension asks for holds in both directions, because the run's end is no
longer the pass signal: if the cancellation were somehow swallowed and the loop kept ticking, the seam's
`get` would be re-entered with no answers left and the task would never return, so `wait_for`'s 2.0 s bound
would raise `asyncio.TimeoutError` — which `contextlib.suppress(asyncio.CancelledError)` does not catch — and
the test would go red after the `finally` restored both globals.

**Reach for the value at the boundary.** `buster_ripper/routes/count_tokens.py:31`, `:52`, `:59` read
`compaction._kv_cache_usage` to decide the inflated token count; `buster_ripper/server.py:22` (`:25-34`,
spawned at `:78`) is the loop's production owner. Both files' bytes moved since the prior record
(`server.py` `a9e49823ea456af9…` → `b3b486615ca797fe…`; `count_tokens.py` `3b1f9361e21dfa65…` →
`195f3b24e327f57d…`) and the reads were re-located rather than assumed: the nudge consumer's lines shifted by
one, so the prior record's `:30`/`:51`/`:58` are `:31`/`:52`/`:59` now. The reach is therefore intact and the
findings above are findings, not inert test weaknesses.

## Notes

- **N1 — carried, unrowed, fail-loud.** "`CancelledError` is not an `Exception`, so `compaction.py:55-56`
  cannot swallow it" is still an atom no S19 member carries (126 carries only the awaiter-side propagation).
  Its adverse resolution now produces `asyncio.TimeoutError` from `_settle`'s 2.0 s bound rather than a silent
  pass, so the verdict does not rest on it.
- **N2 — new, unrowed, fail-loud.** 410 licenses *suspension and yield*; nothing in the member set carries
  "a task that yields at a zero-delay sleep is resumed by the running loop, so its own next iteration
  happens". The witness's entitlement to ticks 2..N rests on that, not on other tasks being ready. Adverse
  resolution: no further `get`, `wait_for` times out, `TimeoutError` is not suppressed → red. Named so a
  checker that eventually rows the `asyncio.sleep` resumption atom knows which arm it is settling.
- **N3 — new, unrowed, fail-loud.** "A cancellation requested while the task is *running* is delivered at its
  next suspension point, not eagerly" carries the ordering in §Composition 6. Both resolutions terminate the
  task cancelled, so the observable is unchanged. `asyncio.wait_for`'s "cancels and awaits on timeout"
  behaviour is likewise unrowed and only matters for the loud path.
- **N4 — new, unrowed, fail-loud.** "`Mock(side_effect=<callable>)` returns that callable's return value, and
  a `spec=`'d session still allows `session.get` to be rebound" — the atoms that make §Composition 1's call
  result the configured answer. If the spec'd auto-child came back instead, 415's non-`MagicMock` qualifier
  would make `async with` raise `AttributeError` at `:36`, be swallowed at `:55-56`, and the values would
  stay at the seed → red. Also unrowed: `seam.task` is assigned at `:415` after `create_task` at `:414`, and
  the seam's `self.task.cancel()` at `:366` therefore always sees a task; the first `get` cannot precede
  `:417`'s first await.
- **N5 — carried, inert.** The four payload siblings still assert only `session.get.called` (`:209`, `:283`,
  `:312`) or nothing at all (`test_poll_kv_cache_updates_global_cache`, `:230-258`, still has no assertion),
  and `:231`/`:262` still name checks no assertion performs. They remain unable to falsify the discarded-child
  shape. No production caller reads them and the seam's own witnesses no longer depend on that vocabulary, so
  this is a weakness of those tests, not a counterexample here.
- **N6 — carried, inert.** The `timeout=timeout` kwarg at `compaction.py:38` is still inspected by nothing:
  `_TickSeam.get(self, url, **kwargs)` (`:363`) accepts and discards it, as do the siblings. A value nothing
  reads.
- **N7 — carried, outside this seam.** `tests/test_server_hooks.py:90, 102` substitute `poll_kv_cache`
  wholesale, so the spawn/delegate half of the chain (`server.py:15-34`, `:78`) crosses no real response
  object in any test. Another seam's territory; nothing is claimed about it here.
- **N8 — handle history.** `CL:python/magicmock-async-context-manager` is also the handle retired row 409
  left behind (`DOES-NOT-HOLD`, `absorbed`). This composition cites it for **415's** fresh `HOLDS` verdict
  only; 409's terminal record is evidence history and is not a premise here.
- **N9 — fingerprint locality, carried.** 411 is pinned to this worktree's installed bytes
  (`.venv/lib/python3.13/site-packages/pytest_asyncio/plugin.py` = `121cb98d…`, re-hashed and exact). 408,
  415, 410 and 126 are pinned to bytes outside this worktree (CPython tags, docs pages, a recorded
  `Lib/asyncio/futures.py` digest). This interpreter is still CPython 3.13.8, whose
  `Lib/unittest/mock.py` is `b4164040…` and `Lib/asyncio/futures.py` is `f55a9918…` — neither matches a
  member fingerprint, and all four members disclose that the local interpreter was not probed. Recorded as
  an invalidation condition, not re-checked here: re-verifying an atomic claim is not this pass's authority.

## Sources

Production sources consulted, all under `/Users/alexhubitski/Projects/buster-ripper/` (the claim source root):

- `buster_ripper/compaction.py` — `:25` (`_KV_EMA_ALPHA`), `:28-57` (`poll_kv_cache`), `:31` (`first`,
  function-local), `:36-39` (`async with session.get(...)`), `:40` (synchronous `status` read), `:41`
  (awaited `text()`), `:42-44` (regex, `float`), `:45-47` / `:48-52` (the two arms), `:55-56` (the swallow),
  `:57` (the interval sleep, outside the `try`).
- `buster_ripper/config.py` — `:64` (`KV_POLL_INTERVAL: int = 10`), `:63` (`COMPACT_KV_RATIO`), `:13`
  (`UPSTREAM`, the URL the call builds).
- `buster_ripper/server.py` — `:15-22` (`start_kv_cache_monitoring`), `:25-34` (`spawn_kv_cache_monitor`),
  `:78` (`on_startup` call site), `:84-106` (`on_cleanup` / `_cancel_background_task`, consulted for the
  settlement shape only; production's cancel-and-await is not the witness's and nothing above rests on it).
- `buster_ripper/routes/count_tokens.py` — `:29-40` (`_nudge_trigger_name`), `:43-60` (`_nudge_tokens`).
- `tests/test_compaction.py` — cited as the **witness side of this seam** (its subject is
  `repo:tests.test_compaction`), not as behavioural evidence: `:183-209`, `:212-227`, `:230-258`, `:261-283`,
  `:288-344` (siblings: their double shape and assertion vocabulary), `:349-368` (`_TickSeam`: sample, pop,
  cancel-then-raise), `:371-388` (`_kv_answer`: the permitted double and its stated forbidden list),
  `:391-394` (`_settle`), `:397-420` (`_drive_ticks`: one answer per tick, interval mutation and restore),
  `:423-433` and `:436-451` (the two witnesses under composition).
- `tests/test_server_hooks.py` — `:89-102` (N7 only).
- `tests/conftest.py` and `tests/test_pytest_config.py` — consulted for any fixture or ini interaction
  bearing on dims 5-6; none found.
- `pyproject.toml` — `:4` (`requires-python = ">=3.11"`), `:18` (`asyncio_mode = "auto"`), `:36-42` (the
  `pytest-asyncio>=1.3.0` / `pytest-mock>=3.15.1` dev pins).
- `.venv/lib/python3.13/site-packages/pytest_asyncio/plugin.py` — 411's cited bytes, re-hashed.
- Library bytes named by the member verdicts themselves (not re-derived here): CPython
  `Lib/unittest/mock.py` at `v3.11.0` / `v3.11.10` / `v3.13.1` / `v3.14.0`, CPython
  `Lib/asyncio/futures.py`, the `asyncio-task` docs page, and the `unittest.mock` docs page.

Nothing was executed. No test, fixture, snapshot or mock was used as evidence for a behavioural claim: the
test bytes are read only as the seam's own witness side — the shapes and the assertion vocabulary the
verified members are jointly asked to support — and every behavioural statement above comes from a `HOLD`
member verdict or from the production sources listed.

## Source Versions

- Repo `buster_ripper/*` and `tests/*`: working-tree bytes, hashed below.
- Interpreter running the seam's code: CPython 3.13.8 (`uv` managed, `.venv/pyvenv.cfg`
  `version_info = 3.13.8`, `.venv/bin/python` → `cpython-3.13.8-macos-aarch64-none`) — unchanged since the
  prior record.
- `pytest-asyncio` 1.3.0 (declared `>=1.3.0` at `pyproject.toml:40`;
  `.venv/…/pytest_asyncio-1.3.0.dist-info` present); `pytest-mock` (declared `>=3.15.1`, `:42`);
  `pytest-aiohttp` 1.1.0 (declared `>=1.1.0`, `:39`); `aiohttp` 3.13.3 (`aiohttp-3.13.3.dist-info`).
- Member consulted sources, as recorded in each verdict: 126 "Python 3.14.7"; 408 CPython `v3.11.10`
  (confirmed at `v3.14.0`); 415 CPython `v3.11.0` (corroborated at `v3.13.1`); 410 Python 3.14.8 docs
  (corroborated at 3.11.17); 411 the local plugin (896 lines).
- Repo Python floor `>=3.11`, which every mock-mechanism member's tag range covers.

## Source Fingerprint

sha256:b54ad7c8b85008a6182721d425b872e77ef26a28d0ee812ac62b1f4c85ca24cd

Material, in this order, one item per line (`<handle> <fp> [<fp>…]`, then `<sha256> <path>`), joined by
single newlines with a trailing newline, prefixed by the two header lines
`S19 members (ledger order) with verdict-recorded fingerprints:` and
`consulted source bytes:` — this is the byte-exact recipe, so the digest is reproducible:

1. `CL:python/cancelled-child-await-propagation` — `2f482524292a7c5d…`
2. `CL:python/asyncmock-async-context-manager` — `2576119e5653ca21…`, `667d9ed88fd330a3…`
3. `CL:python/magicmock-async-context-manager` — `663efa5704c1146a…`, `7c6b6c92b11243eb…`,
   `6513ae8d94214d97…`
4. `CL:python/asyncio-sleep-yields-to-loop` — `0e10a7cd1a7a3a44…`, `2f85eef501369e2d…`
5. `CL:pytest-asyncio/auto-mode-async-test-collection` — `121cb98d74596a9e…`
6. `ac9d42143c98694a…` `buster_ripper/compaction.py` · `22348c68d21b91a3…` `buster_ripper/config.py` ·
   `b3b486615ca797fe…` `buster_ripper/server.py` · `195f3b24e327f57d…`
   `buster_ripper/routes/count_tokens.py` · `ae79036b7fe714c2…` `tests/test_compaction.py` ·
   `2dbbe44dc3e0949d…` `tests/test_server_hooks.py` · `e821283c228bd63a…` `tests/conftest.py` ·
   `c623bdc282433193…` `pyproject.toml` · `121cb98d74596a9e…`
   `.venv/lib/python3.13/site-packages/pytest_asyncio/plugin.py`

Delta against the prior record (`990d0278e86f444d…`): `compaction.py`, `config.py`, `conftest.py`,
`pyproject.toml` and the `pytest-asyncio` plugin are byte-identical to the prior pins; `test_compaction.py`,
`server.py` and `count_tokens.py` moved.

## Invalidates On

Any one of these fires the seam back to a re-trace. CE-1/CE-2 ride `compaction.py` plus the witness's double
factory; CE-3 rides `config.py`, `compaction.py:57` and 410; dims 5-6 ride 411, 126 and `test_compaction.py`.

1. Any member leaves `HOLD` or is re-extracted — including retired row 409 being re-opened, which may not
   re-enter this seam as an alternate premise (N8).
2. `buster_ripper/compaction.py` moves off `ac9d42143c98694a…` — in particular `:36-39` (the `async with`
   form, or the call result it consumes), `:40` (a sync-versus-await change to the `status` read, or a
   truthiness rather than `== 200` test), `:41` (`text()` awaited versus `.text` read), `:31`/`:45-52`
   (`first` becoming module state, which would make a *second call* a second tick), `:55-56` (narrowing or
   removing the `except Exception`, which converts the silent-swallow routes in N1-N4 from loud to
   differently loud — and makes `:366`'s terminating `ClientError` escape the loop), or `:57` (the sleep
   moving inside the `try`, or the interval becoming a parameter or a once-per-call local read before the
   loop, which kills the witness's mutation route at `:413`).
3. `buster_ripper/config.py` `KV_POLL_INTERVAL` moves off `22348c68d21b91a3…`, or its **kind** changes (a
   `float` or `0` default, or per-call injection). Note the distinction this recomposition draws: the
   production *default* is still `int = 10`; the `0` this seam composes against is a test-side mutation with
   a `finally` restore (`test_compaction.py:413`, `:419`), so a real default change re-opens CE-3 from the
   other side and must not be read as already settled here.
4. `buster_ripper/routes/count_tokens.py` stops reading `compaction._kv_cache_usage` at `:31`/`:52`/`:59`, or
   `buster_ripper/server.py:22` / `:25-34` / `:78` stops spawning `poll_kv_cache` — either removes the
   production reach that makes the three closures findings rather than inert test weaknesses. Lines are
   re-pinned to the post-T018/T019 bytes; any further shift re-locates them.
5. `tests/test_compaction.py` moves off `ae79036b7fe714c2…` in the witness block `:349-451` — specifically
   `:386-387` (the self-binding that closes CE-1), `:383`/`:384-385` (the sync `status` / awaited `text`
   qualification that closes CE-2), `:364`/`:420` (the per-tick sample and the `[1:]` windowing that make the
   literals mean what they say), `:413`/`:419` (the interval mutation and its restore), `:366-367` (the
   in-task cancel-then-raise that terminates the run), `:393-394` (`wait_for`'s 2.0 s bound or the
   `suppress` target — widening the suppress to `Exception` or `BaseException` would let a stuck loop read
   as a pass), or `:433`/`:450` (the awaited-channel assertions, which are the CE-1/CE-2 tripwire). The four
   payload siblings at `:183-344` bear on dim 3 and N5 only.
6. `pyproject.toml` off `c623bdc282433193…` at `:18` (`asyncio_mode`) or `:36-42` (the async/mock dev pins),
   or `.venv/…/pytest_asyncio/plugin.py` off `121cb98d…` — dim 5.
7. Any of the library fingerprints in 408 / 415 / 410 / 126 no longer resolves for the interpreter the suite
   runs on, or this venv moves off CPython 3.13.8 — per N9 no member's mock/asyncio bytes are locally pinned
   today.
8. If a future pass rows any of the unrowed atoms N1-N4, recompose the affected dimension against it: the
   verdict is currently SAFE because each of those atoms fails *loud*; a row that showed one of them failing
   quietly would reopen the corresponding CE.

## Checked At

2026-10-07T01:52:17Z


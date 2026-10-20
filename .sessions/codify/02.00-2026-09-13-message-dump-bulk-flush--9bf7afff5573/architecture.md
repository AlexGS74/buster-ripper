# Architecture — 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573

> Squashed from a session completed 2026-10-18; PR #1 merged 2026-10-19.

## What was built

A format-selectable dump pipeline for buster-ripper's `/v1/messages` interception: the
documented per-turn layout (`turn_000.json`, `turn_{N:03d}.diff`, `index.txt`) stays the
default and is produced byte-identically at `2N` write-bearing operations, while two collapsed
formats buffer records in memory and append them to one `<format>.log` per session in a single
write per flush. Flushes fire on a volume threshold, an interval timer, and graceful shutdown.
The same branch also ships an upstream-TLS verification opt-out and the aiohttp-migration repairs
that the PR review surfaced: header parity, transport-error envelopes, streaming relay commit
discipline, request/response byte fidelity, image strip/cache correctness, and session-key derivation.

## Components

### `buster_ripper/dump.py` — dump pipeline
**Responsibility**: per-session state, format selection, the documented writer, both collapsed
record builders, the bulk flush and its timer.
**Location**: `buster_ripper/dump.py`
**Key design choices**:
- `SessionState` carries `turn`, `prev_text`, `pending: list[str]`, `pending_bytes: int` and
  `fmt`. `turn` and `prev_text` advance at **append** time, never at flush time, so a flush never
  resets the diff chain.
- `dump_turn` dispatches on `config.DUMP_FORMAT`: `documented` runs the three per-turn writes;
  `change-focused` / `self-contained` append one record to `state.pending` and return.
- `flush_all` no-ops when `DUMP_DIR is None` or nothing is pending, appends each session's whole
  pending batch to `DUMP_DIR/<sid>/<format>.log` in one write-bearing operation, then clears the
  buffer. It never raises and never touches `turn`/`prev_text`.
- `_flush_state` refuses a symlinked or non-regular artifact target; `_lookup_or_create_state`
  refuses a symlinked or non-directory session dir.
- `poll_flush` is flush-then-sleep; per-iteration exceptions are logged and the loop continues;
  cancellation lands at the sleep await.
- The volume trigger is checked in `_append_record` after the append, against the rendered-text UTF-8
  byte count. Sizing happens before the pending-state mutation, so a JSON-valid lone surrogate cannot
  poison a later valid turn.
- `_resolve_format` hard-fails on an unknown configured value; the documented value never builds a
  `.log` path. `_artifact_path` is the only artifact-name builder.

### `buster_ripper/config.py` — format and trigger seams
**Responsibility**: the closed format set and the trigger values.
**Location**: `buster_ripper/config.py`
**Key design choices**:
- `DUMP_FORMATS = ("documented", "change-focused", "self-contained")`;
  `DUMP_FORMAT = "documented"`.
- `DUMP_FLUSH_INTERVAL = 10` seconds; `DUMP_FLUSH_BYTES = 4 * 1024 * 1024`
  pending rendered-text bytes.
- `SKIP_TLS_VERIFY = False`; `PRESERVED_THINKING = False` (the serve script turns it on by
  default); `IMAGE_CACHE_TTL = 300`; `IMAGE_CACHE_MAX_SIZE = 100`.
- Every value is a module global assigned once at parse time through its option's own callback, so
  `main()` gains no parameter.

### `buster_ripper/main.py` — CLI surface
**Responsibility**: bind every config global and launch the app.
**Location**: `buster_ripper/main.py`
**Key design choices**:
- `--dump-format` is `click.Choice(list(config.DUMP_FORMATS))`, default `documented`.
- `--dump-flush-interval` and `--dump-flush-bytes` are `click.IntRange(min=1)` so zero and
  negative values are a `UsageError` before the global is touched.
- `--skip-tls-verify` is the repo's first opt-out-polarity flag; it assigns its global through
  its own callback.
- `_apply_cli_config` is the smallest coherent extraction from `main()`, keeping the hard
  parameter/length metrics from growing.

### `buster_ripper/server.py` — app lifecycle and outbound client
**Responsibility**: the app factory, startup wiring, background tasks and cleanup order.
**Location**: `buster_ripper/server.py`
**Key design choices**:
- `on_startup` builds the package's only `ClientSession` on an explicit
  `aiohttp.TCPConnector(ssl=not config.SKIP_TLS_VERIFY)` with a `DummyCookieJar`: one
  connector, one posture, every outbound call. The connector is constructed inside the running loop.
- `create_app` passes `client_max_size=0`, lifting aiohttp's 1 MiB request cap. The cap sits at the
  handler's first read, ahead of normalization and `--strip-images`, so any finite value would reject
  bodies the proxy forwards much smaller.
- `on_startup` spawns the KV monitor, dump flush, image expiry and (when detection is unresolved)
  backend retry tasks.
- `on_cleanup` cancels and awaits each task, runs a wrapped `flush_all`, then closes the shared
  session. A task's ordinary failure is isolated; outer cancellation still propagates.

### `buster_ripper/routes/messages.py` — `/v1/messages` proxy
**Responsibility**: normalize, key the session, strip images, inject preserved thinking, dump,
forward; streaming and non-streaming arms; SGLang overflow retry.
**Location**: `buster_ripper/routes/messages.py`
**Key design choices**:
- `normalize()` runs **before** `session_id()`, so the billing nonce does not churn the session key.
- `_request_payload` forwards the client's own bytes when `json.loads(raw) == body`; otherwise it
  re-encodes and relabels `Content-Type: application/json`. This preserves a UTF-16 request body that
  normalization did not change, instead of silently re-encoding it as UTF-8 under the client's stale label.
- `_inject_preserved_thinking` replaces a present-but-non-mapping `chat_template_kwargs` rather than
  skipping it, and is skipped entirely on a confirmed `sglang` backend, whose Anthropic schema drops
  the field.
- The SGLang context-overflow retry reduces `max_tokens` (the field the Anthropic schema declares),
  not `max_completion_tokens`.
- `_stream_single_request` arms no total deadline (`ClientTimeout(total=None)`), matching base's
  undeadlined streaming relay; a request-local timeout replaces the session's rather than merging.
- `_relay_after_commit` catches `Exception` and logs position-not-cause, then returns the committed
  response object so a post-commit failure cannot write a second status line. `CancelledError`
  propagates.
- Client-input decode failures (`JSONDecodeError`, `UnicodeDecodeError`, `LookupError`) answer a typed
  `400 {"error": "Invalid JSON"}` on all client routes.

### `buster_ripper/routes/chat.py`, `passthrough.py`, `count_tokens.py`
**Responsibility**: the chat parity, catch-all and token-count routes.
**Location**: `buster_ripper/routes/`
**Key design choices**:
- `_prepare_headers` applies the shared hop-by-hop policy plus a chat-local empty-`Bearer` rule:
  `_carries_token` selects a credential-bearing line positively, so a value-less `Authorization`
  line cannot delete the real token. `/v1/messages` and the catch-all keep forwarding an empty
  `Authorization: Bearer ` verbatim.
- `_chat_answer` reads the payload once; a well-formed but non-JSON upstream answer falls through to
  the upstream's own status, bytes and filtered headers instead of a synthetic 502.
- All transport-error arms answer `{"error": "Upstream request failed"}` and log the cause, so no internal
  upstream URL leaks to the client.
- `_relay_chat_stream` catches `Exception` post-commit, logs position-not-cause and returns the
  committed response, closing the second-head class at the chat boundary.
- `count_tokens` derives `sid` after `normalize.strip_billing_nonce` — the one normalization step that
  writes the text `session_id()` hashes — instead of calling the whole `normalize()`, so malformed
  element shapes keep base's behavior.
- `_chat_payload` / `_request_payload` forward the client's own request bytes unless a transform
  changed the body.

### `buster_ripper/utils.py` — shared proxy helpers
**Responsibility**: header filters, streaming parse helpers and overflow arithmetic.
**Location**: `buster_ripper/utils.py`
**Key design choices**:
- `forward_headers` and `response_headers` share one `_HOP_BY_HOP` set; `response_headers` also drops
  `content-length` so aiohttp derives the length of the body actually sent.
- `response_headers` returns a `CIMultiDict` built from the upstream's pairs, so one occurrence is one
  entry and repeated names survive. Folding into a plain `dict` destroyed all but the last value.
- `Content-Encoding` is deliberately relayed over bytes the proxy's client already decoded: that is base
  parity, and the alternative postures are deferred to the header-lane design pass.

### `buster_ripper/image.py` — strip and cache
**Responsibility**: `--strip-images` placeholder substitution and the image cache.
**Location**: `buster_ripper/image.py`
**Key design choices**:
- One detector and one writer share `_scoped_contents`, so every user message is inspected, not only
  the latest (or earliest) one.
- The placeholder is `img_<sha256(canonical source JSON)[:12]>`: content-addressed, order-independent,
  and `session_id` is not an operand, so one source keeps identical marker text across turns.
- `IMAGE_CACHE_TTL` ages retention absolutely from `first_seen`; `IMAGE_CACHE_MAX_SIZE` evicts by
  `last_touched`; both stamps come from one injected `time.monotonic` name.
- `cleanup_expired_images` is the global write-path sweep and also drops emptied session dicts;
  `poll_expiry` runs it for an idle proxy.
- `get_cached_image` stays unwired: no reader ships.

### `buster_ripper/normalize.py`, `session.py`, `backend.py`, `compaction.py`
**Responsibility**: request normalization, session identity/stats, backend detection and KV polling.
**Location**: `buster_ripper/`
**Key design choices**:
- `strip_billing_nonce` is the narrow public entry over `_normalize_system`; `session_id()` is
  derived after normalization so the billing nonce cannot churn it.
- `backend.detect_backend` caches `sglang` / `vllm` / `unknown` and re-probes in the background while
  unknown; the preserved-thinking gate reads that state.
- `compaction.poll_kv_cache` is the in-repo periodic-pattern peer the dump timer follows.

## Integration points

- `routes/messages.py:handle_messages` is the only `/v1/messages` interception site. It calls
  `dump_turn`; the swallow-all contract keeps a dump failure from failing a request.
- `server.py:on_startup` builds the only `ClientSession`; every outbound call (`routes/*`,
  `compaction.py`, `backend.py`) consumes `app["session"]`, so one connector governs every posture.
- `server.py:on_cleanup` owns the teardown order: cancel the tasks, flush the dump, close the session.
- `main.py` assigns every config global through its option callback; modules read `config.<NAME>` at call
  time, so a parse-time assignment is visible everywhere.
- `tests/qa/` drives the real CLI/server against a live upstream; the hermetic suite excludes `tests/qa`
  via `norecursedirs`.

## Data flows

1. Client `POST /v1/messages` → `handle_messages` parses JSON → `normalize()` strips the billing
   nonce, cache-control and date → `session_id()` → optional `strip_and_cache_images` → optional
   preserved-thinking injection → `dump_turn`.
2. `dump_turn` selects the writer by `config.DUMP_FORMAT`. `documented` writes the three per-turn
   files immediately. A collapsed format appends the record to `state.pending`; `flush_all` later appends
   the whole pending batch to `DUMP_DIR/<sid>/<format>.log` in one write.
3. The request is forwarded with `forward_headers` and `_request_payload`; the upstream answer is relayed
   with `response_headers`. Streaming relays raw chunks; the non-streaming arm parses only where a transform
   or overflow check needs the parsed form.
4. SIGINT/SIGTERM → aiohttp `GracefulExit` → `on_cleanup` → cancel tasks → `flush_all` →
   `session.close()` → process exits.

## Constraints that shaped the design

- `US-012`: the documented layout must not be deleted before the new behavior is demonstrated better
  and judged worth switching to. The default is therefore `documented`, and the collapsed formats are
  opt-in.
- Arithmetic from `FK-002`'s "one bulk flush is ONE write" plus `US-012`: for N turns a session performs
  **either** `2N` or `F` write-bearing operations, never `2N + F`. This forces the format selector to
  choose which writer runs per turn.
- `G-001`: reduction `1 − F/(2N)`, floor 50% at `F = N`; the artifact must be append-only, so bytes
  grow with `F`, not `N²`.
- `G-003`: `/v1/messages` behavior unchanged; a dump/flush failure cannot fail a request; no latency
  bound.
- `G-008`: with no format configured the documented layout is byte-identical at `2N`; the `MS-002`
  comparison record is the evidence a future switch consumes.
- `G-007`: upstream certificates are verified unless the opt-out is explicitly configured; the opt-out is
  process-wide and never inferred from the URL scheme, host or environment.
- Capacity: `dump_turn` was a hard LNCP length violation, so `T002` extracted the record builders,
  documented writer, dispatch and artifact path before adding behavior. `main.py:main` was a hard
  parameter/length violation, so `T006` extracted `_apply_cli_config` and wired options through callbacks.
- Coverage: `pyproject.toml` gates the suite at 80% and `dump.py` is not omitted, so the timer loop body
  lives in `dump.py` and is tested; `server.py` is coverage-omitted, which is why the loop is not there.
- No new runtime dependency: `aiohttp` and `click` only.

## What was deliberately left out

- No viewer or dump-reading tool (`FK-007` option 1): the self-contained encoding is the addition.
- No runtime record when the TLS opt-out is used (`FK-008` option 1): the flag is the only signal.
- No custom CA bundle/truststore, no mTLS, no client certificates, no inbound TLS.
- No custom signal handler: aiohttp's `GracefulExit` is reused.
- No durability machinery: no fsync, journal, WAL or crash recovery; SIGKILL loss is accepted.
- No machine-oriented dump format (`RM-014`), no default-format switch (`RM-013`), no header-lane
  content-coding change (`RM-008`), no request-direction content-coding posture (`RM-011`).
- No migration or removal of old dump files.
- No live-dump reader: `get_cached_image` and any production reader remain unwired.

## Dead ends and reversals

- **Shutdown-only flush** (`FK-001` option 2) was rejected: it makes the loss window the whole
  session and removes `index.txt` for the session's entire life, contradicting the documented live-reading
  workflow.
- **Preserving the documented layout only** (`FK-002` option 1) was rejected: it caps the reduction at
  about 50% and only collapses the index append.
- **Collapsed format as the default** (S-2) was rejected: it violates `US-012`'s gating clause under either
  reading of "unaffected". **No switch** (S-3) makes the reduction unreachable. **Both writers at once**
  (S-X) is `2N + F`, worse than today, and is excluded by arithmetic.
- **Encoding-only selector** was rejected: a value names a complete output shape, including the layout axis.
  The term and the `PI:` family were renamed `dump_format`, and the set is the closed three values.
- **A viewer/TUI** (`FK-007` options 2/3) was rejected: it is new product surface; the self-contained
  encoding answers the readability goal with no tool.
- **A runtime record that the skip happened** (`FK-008` option 2) was rejected: it asserts behavior the
  plan does not ship.
- **A negative empty-`Bearer` predicate** was tried in `T013` and rejected: `_is_empty_bearer("")` returned
  `False` because `bool(value)` short-circuits, so a value-less `Authorization` line won the selection and
  replaced a real token with an empty header. The positive `_carries_token` form is the fix.
- **Reading a `web.Response` body back as bytes** was rejected in `T019.001`: a `str` passed to `body=`
  becomes a `StringPayload`, so `isinstance(.body, bytes)` is always false. `result.text` is the one
  accessor that returns the string; the `text=` constructor was rejected because it appends
  `; charset=utf-8` to a relayed error, a client-visible change.
- **Calling the whole `normalize()` on `count_tokens`** was rejected in `T019.002`: it made malformed
  element shapes 500 where base answered 200. The narrow `strip_billing_nonce` is the fix; hardening the
  element loops measured `500 → 502` at `/v1/messages` and was routed to `RM-012`.
- **Universal-newline reads in the documented-format gate** were rejected in `T021.003`: `read_text`
  converted CRLF before the expected diff was computed, so a CRLF-vs-LF pair compared equal on the expected
  side while the written diff reported the change. Both diff surfaces are now read with `newline=""`.
- **Comparing expected and observed line identities differently** was rejected in `T021.001`: `splitlines()`
  without endings and `lineterm=""` without a join produced false failures on a glued first changed line, a
  CRLF-only change and a trailing-newline-only change. Both sides now use the writer's published shape.
- **Marking a backlog line fixed without editing it** was rejected in `T022.001`: the line stayed `[ ]` while
  the state count claimed it closed. The line is now `[x]` with its DRAIN evidence and the counts re-derived.
- **Prose `routed onward to` tails** were rejected in `T024.001`: the close gate reads the literal
  `→ routed:` marker, so 37 tails were rewritten and the census corrected.
- **Widening only the `ClientError` arms** was rejected in PR review: a bare `asyncio.TimeoutError` is not a
  `ClientError`, so a fired total produced aiohttp's generated 504 instead of the repo's 502 envelope. The
  arms now name both classes.
- **Re-serializing a parsed chat answer unconditionally** was rejected in PR review: it changed bytes the
  upstream sent. The unfixed arm now relays the upstream's raw bytes.
- **Injecting `chat_template_kwargs` on SGLang** was rejected in PR review: SGLang v0.5.9's Anthropic
  schema drops the field, so the injection advertised an effect that never happened. The guard skips it.
- **Using `max_completion_tokens` on an Anthropic body** was rejected in PR review: the Anthropic schema
  declares `max_tokens`, so the retry reduced a field the upstream ignored.

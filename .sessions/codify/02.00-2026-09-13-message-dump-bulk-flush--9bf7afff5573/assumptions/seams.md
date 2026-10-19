# Assumptions — seams ledger

Session: `02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573`. Member claim rows live in `claims.md` (run-1 rows 1–44 and run-2 rows 45–50 include evidence history and consumed authorizations; run-3 rows 51–67 are the adhoc set; run-4 rows 68–73 replace malformed premises; run-5 rows 74–85 replace absorbed premises; run-6 rows 86–105 replace stale realized-chain premises; run-20 rows 165–169 add the T004 write-count instrument, its stdlib funnel premise and collapsed-artifact test reads; run-21 rows 170–173 replace malformed member 167 — the write-count parametrization scope; run-22 rows 174–182 add the landed T006 TLS chain, its serve-script/README wiring and the per-request `ssl` absence; run-23 re-bases the S3/S5 recomposition on the post-T006 `main.py` bytes — rows 99/100 re-dispatched fresh in B153, rows 74/40 already fresh; run-24 adds rows 183–199 for the newly registered T011 and widens S6 to the shell-semantics premises its one-line fix rests on — S6's ten existing members inherit their verdicts and gain `T011` in `Tasks`; run-25 is T011's accepted-local recheck — the line-28 edit moved `serve_buster_ripper.sh` `f4076d1c…` → `d5db2144…`, firing S6's ten script-pinned members (159–163, 179, 180, 191–193) into fresh bundles B162–B165 while the four README-pinned members and the eight `lang:bash` members stay current on unchanged sources); run-26 is the post-acceptance resolution pass on S6 — the owed recomposition landed `SAFE` on `d5db2144…`, `seam-S6-disable-form` reached `resolved`, S6's Routing is `absorbed` (C2 open on T012) and State deliberately stays `recheck` because the registry `CO:` basis is retained-superseded pending its handle-bearing republish; run-27 is T012's accepted-local recheck — the assembly rewrite moved `serve_buster_ripper.sh` `d5db2144…` → `c58256fb…` (46 → 54 lines), firing the eleven re-dispatched members (159–163, 179, 180, 189, 191–193) into fresh bundles B168–B172, adding row **202** as the value-grammar premise the 2026-10-05 routing owed, and widening S6 to 23 members and two re-worded dimensions; the four README-pinned members and the remaining `lang:bash` members stay current on unchanged sources; run-28 is the scoped composition-blocker extraction on S6's memberless ninth dimension — rows **203** and **204** row the `${array[@]+word}` guard semantics that arm owed, widening S6 to 25 members in fresh bundle B173 and leaving every existing row's cells untouched); run-29 is the post-acceptance resolution pass on S6 — the 25-member handle-bearing recomposition landed `SAFE` on the post-T012 bytes and was published to `CO:serve-script-env-to-argv/env-var-flag-crossing` in place, so `seam-S6-value-word-split` reached `resolved`, the wave's three riders are discharged, `QB:assumption-registry/co-serve-script-basis-republish` is retired, and S6's State moves to `registry-safe` per §3c. Every member whose verdict is not already current carries a `Bundle` there.

| Seam | Claim rows | Subjects | Boundary value | Dimensions | State | Routing | Permanent |
|---|---|---|---|---|---|---|---|
| S1 | 86, 11, 12, 116, 117, 111, 112, 8, 9, 10, 13, 103, 87, 88, 89, 90, 92, 110, 105, 120, 121, 122, 94, 106, 107, 108, 44, 165, 166, 170, 171, 172, 173, 168, 169 | repo:buster_ripper, repo:buster_ripper.session#session_id, repo:buster_ripper.dump#SessionState, repo:buster_ripper.dump#dump_turn, repo:buster_ripper.dump#_render_body, repo:buster_ripper.routes.messages#handle_messages, repo:buster_ripper.dump#_write_documented, repo:buster_ripper.dump#_write_full_turn, repo:buster_ripper.dump#_write_diff_turn, repo:buster_ripper.dump#_append_index_line, repo:buster_ripper.dump#_append_record, repo:buster_ripper.dump#_artifact_path, repo:buster_ripper.dump#_flush_state, repo:tests.test_dump, repo:tests.test_dump_write_count, rel:tests.test_dump_write_count#COLLAPSED_FORMATS→buster_ripper.config#DUMP_FORMATS, repo:tests.test_dump_records, lang:python | the encoded record batch and sid key crossing request path → per-session pending buffer → flush → artifact file, with documented- and collapsed-artifact test reads and the instrumented write-count unit | three format values; fixed selected format (no runtime migration premise); unknown-format error under request swallow; sid keying; arrival order; documented per-turn writes without pending records and guard-return versus nonempty collapsed batch append; append-time versus flush-time sequencing; pre-mutation encoded-byte computation and rendered-text UTF-8 byte accumulation; no production reader, documented-file test reads; instrumented write-count boundary (write/append-mode `Path.open` under the dump root only — mkdir and outside-root uncounted; `write_text` funneled through `Path.open`; payload measured in UTF-8 bytes; count runs per shipped collapsed format) | out-of-scope | unverified | — |
| S2 | 14, 35, 123, 124, 125, 16, 96, 113, 114, 126, 118, 127, 128, 97, 95, 104, 120, 121, 122, 36 | repo:buster_ripper, pkg:aiohttp@3.13.3, lang:python, repo:buster_ripper.server#spawn_kv_cache_monitor, repo:buster_ripper.server#spawn_dump_flush, repo:buster_ripper.server#_cancel_background_task, repo:buster_ripper.server#on_cleanup, repo:buster_ripper.server#_final_dump_flush, repo:buster_ripper.dump#_flush_state, repo:buster_ripper.dump#dump_turn | the SIGINT/SIGTERM lifecycle signal crossing aiohttp's runner cancellation and cleanup hooks into the app's background-task awaits and pending batch artifact | single signal versus fresh cleanup cancellation; retained runner cancellation count versus cancelled child's exception; pre-await cancellation-count snapshot and count increase during child await; completed task with stored ordinary exception isolated at await versus active child cancellation; sequential cancellation of both task slots; shutdown batch flush before session close; cleanup-hook exception propagation; fixed selected format; nonempty collapsed batch append versus documented no-buffer/guard-return | out-of-scope | absorbed | — |
| S3 | 99, 74, 40, 41, 98, 75, 103 | repo:buster_ripper.main, pkg:click@8.3.1, repo:buster_ripper.dump#_resolve_format, repo:buster_ripper.dump, repo:buster_ripper.dump#dump_turn | the configured format value crossing CLI option → `config.DUMP_FORMAT` global → the format dispatch | three format values; default `documented`; import-bound decorator default versus runtime dispatch read; once-per-process callback assignment; unknown value → `ValueError`; `click.Choice` validation; no runtime format mutation assumed | out-of-scope | unverified | — |
| S4 | 175, 176, 174, 74, 177, 178, 25, 26, 182, 37, 39, 50, 40 | repo:buster_ripper.main, repo:buster_ripper.main#_set_skip_tls_verify, repo:buster_ripper.config, repo:buster_ripper.server#on_startup, repo:buster_ripper, pkg:aiohttp@3.13.3, pkg:click@8.3.1 | the TLS posture (`ssl` True/False) crossing the callback-wired `--skip-tls-verify` flag → `config.SKIP_TLS_VERIFY` → the explicit `TCPConnector(ssl=…)` construction into every outbound call | posture polarity (opt-out, default off, never inferred from scheme); import-bound decorator defaults versus explicit callback; process-wide one-connector scope with no per-request `ssl` override; loop binding at connector construction (session adopts `connector._loop`); unset-path behaviour preservation | out-of-scope | unverified | CO:cli-flag-to-outbound-call/tls-posture-crossing |
| S5 | 100, 74, 40, 43, 76, 85, 116, 117, 111, 112, 96, 101, 110 | repo:buster_ripper.main, pkg:click@8.3.1, lang:python, repo:buster_ripper, repo:buster_ripper.server#spawn_dump_flush, repo:buster_ripper.dump#poll_flush, repo:buster_ripper.dump#_append_record | trigger parameter values (interval, threshold) crossing CLI options → config globals → the two trigger sites | import-bound decorator defaults versus runtime trigger reads (85: threshold read only); collapsed-path-only scoping; inclusive threshold comparison (110) on UTF-8 rendered-text byte accumulator (116–117); pre-mutation encoded-byte computation; timer catch-and-continue; cancel at sleep await; `poll_kv_cache` is metrics poll, not flush owner | out-of-scope | unverified | — |
| S6 | 33, 164, 181, 191, 185, 183, 184, 186, 159, 179, 187, 188, 161, 160, 180, 189, 162, 163, 192, 193, 190, 194, 202, 203, 204 | repo:README.md, repo:serve_buster_ripper.sh, lang:bash | env-var values (`DUMP_FORMAT` landed with T005; `SKIP_TLS_VERIFY` landed with T006; `DUMP_DIR`, whose default form landed with T011 — the unset-only `-` form) crossing shell parameter expansion → the `EXTRA_FLAGS` array accumulator (per-value appends since T012, each value interpolated **quoted**) → the `exec` line's guarded array expansion, terminating in the binary's argv word set | empty-default guard (`DUMP_FORMAT`); boolean `== "1"` value-guard append (`SKIP_TLS_VERIFY`, the `STRIP_DATE`/`VERBOSE` peer shape) versus conditional presence-guard appends (`DUMP_FORMAT` in the `DUMP_DIR` peer's `-n` shape); two argv channels (exec-line always-on quoted trio + conditional `EXTRA_FLAGS` elements); no positional forwarding; README run path with no-format, collapsed and skip-TLS examples; **unset versus set-but-empty discriminator** (the environment-prefix assignment form the documented disable uses); **`:-` versus `-` default form** as the step that keeps or discards that discriminator before the guard consumes it; **presence-guard false branch under `set -e`** (the previously unreachable arm); **zero-word arm when every append is suppressed** (C2's mechanism was the pre-T012 unquoted string expansion at the exec; post-T012 the arm is the `${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}` guard, whose guard semantics are now member rows — **203** (the alternate-value expansion contributes zero words for an unset or empty array) and **204** (that same test raises no nounset error under `set -u`), added by `claims.md` run 28; run 27 had recorded this atom as collected-and-not-rowed); **value grammar at the boundary** (row 202: one argument word per array element, so `$IFS` whitespace, glob metacharacters and a leading dash are immaterial to the word set); **header-documented disable form versus the README example inventory** | registry-safe | absorbed | CO:serve-script-env-to-argv/env-var-flag-crossing |
| S7 | 51, 68, 69, 70, 71, 72, 129, 130, 131, 132, 133, 134, 135, 136, 138, 137, 139, 140 | pkg:aiohttp@3.13.3, lang:python, repo:buster_ripper.routes, repo:buster_ripper.routes.messages#handle_messages, repo:buster_ripper.routes.chat#handle_chat, repo:buster_ripper.routes.count_tokens#handle_count_tokens, repo:buster_ripper.routes.count_tokens#_read_body, repo:buster_ripper.routes.chat#_make_chat_request | client raw bytes → aiohttp charset decode → JSON parse → messages/chat handler catch or count_tokens helper catch → count_tokens HTTP exception transport → typed response; separately upstream response bytes → `resp.json()` → upstream-attributed catch | client decoding versus syntax failure; unrelated exception branches under `ValueError`; three production client sites; messages/chat direct JSON 400 vs count_tokens syntax empty-body fallback and undecodable-byte HTTPBadRequest; HTTPBadRequest transport status/content type; upstream parse failure maps to 502 rather than client 400 | out-of-scope | absorbed | — |
| S8 | 58, 79, 80, 81, 82, 141, 57, 77, 142, 78, 143, 144 | repo:buster_ripper.server#on_startup, repo:buster_ripper.backend#detect_backend, pkg:aiohttp@3.13.3, lang:python, repo:buster_ripper | the startup probe's request duration crossing the shared 600 s session default into the conditional `/v1/models` GET's 5 s request-local timeout and backend cache | override/cache bypass versus uncached GET; request-local timeout replacing session default; ordinary failure or timeout → cached `unknown`; cancellation outside `Exception` and cache assignment; startup await before serving; package-wide explicit-timeout peer set | out-of-scope | unverified | — |
| S9 | 73, 67, 145, 149, 150, 151, 154, 156, 157, 158, 155, 146, 83, 84, 64, 148, 147, 152 | repo:buster_ripper.routes.messages#handle_messages, repo:buster_ripper.routes.messages#handle_streaming, repo:buster_ripper.routes.chat#_make_chat_request, repo:buster_ripper.routes.chat#_relay_chat_stream, pkg:aiohttp@3.13.3, repo:buster_ripper.routes.chat#_apply_response_transforms | request `stream` value crossing the outgoing chat request → stream dispatch → successful SSE body relay or raw upstream error status/bytes/Content-Type → downstream error response; false/unset value → upstream JSON parse → response transformation | request-flag authority, not response content-type sniff; true versus false/unset; successful raw chunks and upstream error status/body; upstream error Content-Type present versus absent, copied header when present versus no explicit header when absent; SSE media type rejected by default `resp.json()` versus compatible `+json` accepted; `ContentTypeError` under `ClientError` and 502 mapping; nonstream eval/non-eval JSON transformation | out-of-scope | unverified | — |
| S10 | 205, 206, 207, 373, 209, 210, 211, 212, 213, 365, 366, 367, 368, 214, 215, 320, 321, 217, 218, 219, 220, 221, 222, 322, 323, 224, 234, 235, 236, 324, 327, 325, 328, 326, 329, 246, 247, 376, 377, 378, 379 | repo:buster_ripper.utils, repo:buster_ripper, repo:buster_ripper.routes.chat, repo:buster_ripper.routes.chat#_prepare_headers, repo:buster_ripper.routes.chat#_carries_token, repo:buster_ripper.routes, repo:buster_ripper.routes.count_tokens#handle_count_tokens, rel:buster_ripper_legacy#_HOP_BY_HOP→buster_ripper/utils#_HOP_BY_HOP, repo:buster_ripper_legacy, pkg:aiohttp@3.13.3, repo:tests, repo:tests.conftest | a header name set crossing the client→proxy boundary (`web.Request.headers`) into the upstream request (`headers=` on `session.post`/`session.request`) and the upstream→proxy boundary (`ClientResponse.headers`) into the downstream response (`web.Response`/`web.StreamResponse` headers), with `buster_ripper/utils.py`'s two helpers as the single policy owner | eight-name hop-by-hop set membership; request direction versus response direction; the five sites' local skip sets versus the shared helper's drop sets; lowercase-key matching against the name spelling aiohttp actually exposes — recognized names canonicalized in `.headers`, the peer's own spelling only in `raw_headers`, and only for one parser backend (217 with 324–329); `host` and `content-length` dropped on egress and recomputed by aiohttp; empty-`Bearer` `Authorization` deletion staying route-local versus being absorbed into the shared helper; base (pre-refactor) parity of both directions including the base response filter forwarding `content-length`; the eight-name target's authority being base parity rather than a standard's enumeration (207/219/220 — the RFC-completeness premise left no replacement row, its retired member being 208); the `.items()` comprehension shape a rename-proof site census greps for, and the one header-name decision in that family the shape does not reach (320/321); which base-file requests the shared filter owns, and which header-less request is not a proxy boundary (322/323); non-route modules issuing no header-bearing request and `count_tokens` issuing no upstream request; absence of any test witness in either direction (246/247, re-checked); which `Authorization` value survives when one request carries several (365–368); the response-side drop set including `content-length` (373); the exact-bytes channel a witness observes the boundary through (376–379) | out-of-scope | absorbed | — |
| S11 | 67, 78, 226, 227, 228, 229, 230, 231, 330, 331, 233, 239, 240, 241, 242, 243, 369, 370, 371, 372, 244, 245, 248, 374 | repo:buster_ripper.routes.messages#handle_streaming, lang:python, pkg:aiohttp@3.13.3, repo:buster_ripper.routes.messages#_stream_single_request, repo:buster_ripper.routes.messages#_relay_after_commit, repo:buster_ripper.routes.messages#_note_message_start, repo:buster_ripper.utils#find_message_start, repo:tests, repo:tests.upstream_stub | the response head crossing the handler→aiohttp-server boundary at the `await response.prepare(request)` commit, and the exception channel out of the relay loop through the handler's single `except` arm, or out of the handler altogether, into the server protocol's own generated error response on the same request | pre-commit versus post-commit failure; the catch scope of one `except aiohttp.ClientError` spanning both sections (payload errors and the named client timeout classes inside it, the bare `asyncio.TimeoutError` the cumulative total timeout raises and the non-`ClientError` payload-shape errors outside it — 330/331); returning a freshly constructed response object versus abandoning the stream; one header block versus two on an already-committed connection; handler raising out versus returning a value; `CancelledError` outside `Exception`, which must keep propagating; upstream-controlled payload shape reaching `.get()` on a `str`; `write()` against a closed client connection; absence of any witness on either side of the commit boundary (248, re-checked); which object the handler hands back once the head is flushed and where a post-commit exception is absorbed instead of raised out (369–372); the stand-in that produces a fault inside the relay loop without monkeypatching the handler (374) | out-of-scope | absorbed | — |
| S12 | 253, 254, 348, 349, 350, 351, 255, 332, 257, 258, 259, 260, 261, 262, 264, 265, 268, 269, 270, 271, 290, 380, 316, 317 | repo:buster_ripper.image#has_images, repo:buster_ripper.image#_has_image_in_blocks, repo:buster_ripper.image#_user_content, repo:buster_ripper.image#_scoped_contents, repo:buster_ripper.image#strip_and_cache_images, repo:buster_ripper.image, repo:buster_ripper.routes.messages#handle_messages, repo:buster_ripper.normalize#normalize, repo:buster_ripper, repo:buster_ripper.main#strip-images-option, repo:buster_ripper.config#STRIP_IMAGES, repo:tests.test_image, repo:tests, repo:tests.image_bodies, repo:tests.test_routes_integration | the detector boolean and the user-message content list crossing `--strip-images` → `config.STRIP_IMAGES` → either guarded call site → detector → writer → the body forwarded upstream and the changes log | message position (earliest, middle, latest); role (user versus assistant); content shape (list versus string — the writer reaches every user message but rewrites only list-shaped content, 332 read against 257/258); one scope function standing behind both the gate and the writer (348–351); fresh-versus-shared body structures under each position row (380); absent or empty `messages`; image nesting (top-level content block versus `tool_result`-nested); guard ownership and reachability (the route site versus the normalize site's truthy-`session_id` precondition, with the package's only production `normalize` call passing an empty one); the changes-string append as a side effect of the same branch; the help text's stated scope versus the detector's scope; the characterization test that pins today's narrower reading; absence of any flag-enabled or route-level witness | out-of-scope | absorbed | — |
| S13 | 266, 267, 272, 352, 353, 354, 355, 356, 333, 274, 277, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 345, 346, 347, 339, 381, 291, 292, 309 | repo:buster_ripper.normalize#normalize, repo:buster_ripper.normalize#_normalize_cache_control, repo:buster_ripper.image#_replace_image_with_placeholder, repo:buster_ripper.image#_image_key, repo:buster_ripper.image#get_cached_image, repo:buster_ripper.session#session_id, repo:tests.test_image, repo:buster_ripper.image#clear_session_cache, repo:tests.image_bodies, lang:python | the client-supplied image `source` mapping crossing canonical serialization → digest truncation → cache key → placeholder text → the forwarded body and the KV prefix | mapping insertion order; nesting depth of the key sort; whitespace form of the serialization; `cache_control` presence at the observation site (the discriminator between `messages.py`'s strip and `normalize.py`'s strip); whether `session_id` participates in the key; a repeated source inside one request (dict size after a same-key write); digest operand versus truncated serialization; `str` versus bytes at the hash call; digest length versus the twelve-character slice; cross-process stability; the shape of the stored entry value — pre-wave a `(source, timestamp)` pair whose element 0 is the block's `source` (333, re-checked), post-wave a triple written by `_cache_source` (359, 360) — and which object is the digest operand, in which canonical form, coerced how, and sliced where (352–356); aliasing between the stored payload and the caller's own mapping; unserializable values in `source`; test pins that assume positional keys, the one never-cached negative key that has no derived form, and the cache state a witness inherits from its own test function — reset per test, not per call (345 names where a whole-mapping emptying may occur, 346 names the one mid-test `clear_session_cache` call, 347 says what that call removes, and 339 the call's arity) | out-of-scope | absorbed | — |
| S14 | 2, 11, 342, 343, 230, 259, 263, 275, 276, 357, 358, 359, 360, 361, 362, 363, 364, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 310, 311, 313, 314, 315, 381, 382, 383 | repo:pyproject.toml, repo:buster_ripper.session#session_id, rel:buster_ripper.main→buster_ripper.config, repo:buster_ripper.main, repo:buster_ripper.image, repo:buster_ripper.image#_now, repo:buster_ripper.image#_cache_source, repo:buster_ripper.image#strip_and_cache_images, repo:buster_ripper.image#_replace_image_with_placeholder, repo:buster_ripper.image#_cleanup_expired_images, repo:buster_ripper.image#cleanup_expired_images, repo:buster_ripper.image#_evict_by_lru, repo:buster_ripper.image#get_cached_image, repo:buster_ripper, repo:buster_ripper.routes.messages#handle_messages, repo:buster_ripper.server#logging_middleware, repo:buster_ripper.server, repo:tests.test_image, repo:tests.image_bodies, repo:tests.test_image_stability, pkg:aiohttp@3.13.3, lang:python | per-entry retention stamps and session-mapping liveness crossing the writer's tail sweep into module-level state shared by concurrent handlers, with the exception channel out of that same call through the middleware to the HTTP status | sweep position (before the session ensure, inside the ensure-to-first-index window, after the message loop, or at the function end where the empty-`messages` return sits); sweep scope (one session versus every session); stamp semantics (absolute age versus idle refresh, `first_seen` versus `last_touched`, one stamp versus two — landed as two slots taken through `_now`: 357–361); comparator strictness at exactly the TTL, where no witness exists; which primitive owns the emptied-session-dict drop, and what the caller-less global janitor actually does today (362–364 name the sweep's position, the drop's owner and the early return that skips it, against 299/300/301 re-checked); read-path immutability (expiry-only cleanup because the reader indexes after it); tuple arity at every unpack site; config read timing versus import binding for both limits; wall-clock reversibility of the age computation; single-event-loop safety for iterating a mapping other handlers write; caller absence for both janitors and for the read path; uncaught-exception transport to aiohttp's generated error response; the autouse fixture that clears module state between tests | out-of-scope | absorbed | — |
| S15 | 384, 385, 386, 387, 388, 389 | pkg:aiohttp@3.13.3, repo:buster_ripper.utils#response_headers | the upstream response's payload bytes and its header set as one `aiohttp.ClientResponse` hands both to the caller at each relay boundary | session default versus per-request `auto_decompress` (384/385); encoded-on-the-wire versus decoded-as-handed-over body bytes (386); `Content-Encoding` relayed versus withheld alongside a body the client already decoded (387); single versus repeated occurrences of one header name, and one pair per occurrence at `.items()` (388); caller-supplied versus client-default `Accept-Encoding`, which decides whether an encoded answer is ever asked for (389) | registry-counterexample | accepted-consequence | CO:client-response-to-relay-consumer/payload-and-header-pairing |
| S16 | 390, 412, 392, 393, 394, 395, 230 | pkg:aiohttp@3.13.3, repo:buster_ripper.server#create_app, repo:buster_ripper.routes | the application's configured `client_max_size` value crossing the package's single `web.Application` construction site into the body-read path of every request each route class receives | unset default versus an explicit value (392 against 390/412 — 412 naming which application's configured value propagates: the one whose runner serves the connection and built the request); zero versus a positive byte count — the falsy-skip arm (393); length-advertised body versus chunked body inside one app-wide limit; the four POST route classes and the catch-all sharing one construction site; the raise landing inside a handler's own body-read `await` versus before the handler is reached (394 against 230); the exception class the code raises versus the status and body text the client sees (395) | recheck | absorbed | — |
| S17 | 57, 396, 397, 330, 331 | pkg:aiohttp@3.13.3, repo:buster_ripper.server#on_startup, repo:buster_ripper.routes.messages#_stream_single_request, repo:buster_ripper.routes.messages#_relay_after_commit | the deadline governing one upstream call crossing the shared session's default `ClientTimeout` into a request-local `timeout=` value and then into the lifetime of an already-committed SSE relay | session default versus request-local replacement rather than merge (57); the cumulative `total` component versus the connect/socket components of the same object (396); `None` versus a number as the spelling of "unset" (396); the wait for response headers versus the consumption of the payload that follows it (397); which class a firing deadline raises — a `ClientError` subclass the relay's arm catches versus bare `asyncio.TimeoutError` outside it (330 against 331); pre-commit versus post-commit visibility of that error on a connection that has already answered 200 | recheck | absorbed | — |
| S18 | 51, 68, 69, 70, 71, 72, 137, 398, 413, 400, 401, 402, 403, 414, 405, 406, 407, 416 | pkg:aiohttp@3.13.3, lang:python, repo:buster_ripper.routes.messages#handle_messages, repo:buster_ripper.routes.messages#_stream_single_request, repo:buster_ripper.routes.chat#handle_chat, repo:buster_ripper.routes.count_tokens#_read_body | the charset a peer names in its Content-Type crossing aiohttp's decode step into the exception class each parse arm must catch, and the raw-bytes path that has to stay available on the other side of that raise | client-request direction versus upstream-response direction (401 against 398/413); valid bytes under a known codec versus invalid bytes under a known codec (51/402) versus a codec name that does not exist at all (413/401/405) versus a codec name that **does** exist but is not a text encoding — a bytes-to-bytes transform (416); `UnicodeDecodeError`, `json.JSONDecodeError` and `LookupError` as three separate catch targets and which of them a `ValueError`-rooted clause reaches (68–72 against 406/407), and whether the three are the complete set at this boundary for that last arm (416); strict default versus `errors="replace"`, and what replacement preserves of the text being matched (402 against 403/414); `text()` versus `read()` as the consumer's escape hatch (398/400); raising out of the handler versus being caught and answered with the site's existing 400 shape (137) | recheck | absorbed | — |
| S19 | 126, 408, 415, 410, 411 | lang:python, pkg:pytest-asyncio@1.3.0, repo:tests.test_compaction, repo:buster_ripper.compaction#poll_kv_cache | the `/metrics` response a test double hands across `async with session.get(...)` into the production poll loop, and the witness's own right to run that loop more than once | awaitable call result versus an async-context-manager object at one call site (408 against 415); the response attributes the loop reads without awaiting (`status`) versus the one it awaits (`text()`); which double the file's sibling poll tests already use, and whether that shape reaches the production branch at all; where the driven loop yields so a second tick is reached (410) against the interval the loop sleeps on; the test function's own collection and event-loop eligibility (411); cancelling and awaiting a coroutine that never returns, so a swallowed cancellation cannot read as a pass (126) | recheck | absorbed | CO:test-double-to-poll-loop/async-witness-eligibility |
| S20 | 417, 418, 425, 420, 421, 422, 423, 424 | pkg:aiohttp@3.13.3, lang:python | one `str` body value crossing aiohttp `web.Response`'s constructor-in / read-back-out boundary: entered through the `body=` initializer and recoverable through the `body` attribute, the `text` property, or the response's own payload write to the wire | accepted versus rejected at construction (417 — a raise there breaks the round trip before it starts); representation the getter hands back (418 — `bytes` versus a payload object versus `None`); equality with the input across that round trip (425 — whether the payload read back off the `body` getter decodes to the string that went in, under the response's own charset, or to something else); which exit the value is readable at (420 wire write versus 421 `text` property versus 422 the class's accessor set as a whole); string-coercion result of whatever is stored (423's text versus default object repr, resting on 424's `__str__`→`__repr__` fallback); the charset applied to the bytes that reach the wire versus the response's own advertised one (420) | new | absorbed | — |
| S21 | 426, 427, 428, 429, 430, 431, 438, 439, 440 | lang:python | the unified-diff text `difflib.unified_diff` emits for two renderings crossing the gate's `"".join(...)` mirror, structural `_content_lines` recovery and first-character changed-line parser (`tests/qa/test_qap_ms001.py` `:120-240`), with production's `"".join(...)` of `splitlines(keepends=True)` lines on the other side (`buster_ripper/dump.py:262-270`) | six line kinds (`---`/`+++`/`@@`/`-`/`+`/space); header terminator presence (`lineterm=""` versus default); context count (`n=0` versus default 3); input split mode (`splitlines()` versus `splitlines(keepends=True)`); join mode (`""` versus `"\n"`); exact line identity; file-header pair order before any hunk header (438); hunk-header closing `@@` (439); content-line terminator retention (440) | recheck | absorbed | — |
| S22 | 435, 436, 441, 442 | lang:python | the external upstream's HTTP status and body crossing the harness's `require_upstream` preflight (`tests/qa/harness.py:50-67`) and `assert_upstream_response` status-class oracle (`:174-194`) into the gate's `assert_upstream_response(...)` calls, against the declared endpoint `http://hyperipper.local:30000`; the gate's own client hop now builds its opener with `urllib.request.ProxyHandler({})` (`Proxy.post`, `:271-290`) | status class (2xx/4xx/5xx); preflight `GET /v1/models` versus request `POST /v1/messages`; `HTTPError.code` versus `URLError`/`TimeoutError`; declared endpoint versus ambient proxy settings; default `ProxyHandler` installed versus skipped for a supplied instance (441); empty proxy map versus ambient map (442) | recheck | absorbed | — |
| S23 | 434, 437 | pkg:pytest@9.0.2, lang:python | the test process's environment map crossing `monkeypatch.delenv` (`tests/qa/test_qap_ms001.py:95-99`) into the proxy subprocess at `subprocess.Popen` spawn (`tests/qa/harness.py:259`) | absent versus set versus set-but-empty `DUMP*` value; deletion before versus after spawn; inherited versus explicit `env=` | new | unverified | — |

## Run-3 changelog (2026-09-22 — adhoc extraction at the MS-002 boundary)

- **S7, S8, S9 `new`** — the three remediation tasks' seams: T003.001's parse-boundary family (S7), T007's probe-timeout chain (S8), T008's stream-dispatch/relay chain (S9). Members are run-3 rows 51–67.
- **S1–S5 `recheck`** — two triggers fire together: (i) member re-dispatch (44 → S1; 35/36 → S2; 34/40 → S3/S4/S5 — the phase-exercised set, fresh verdicts pending), and (ii) the chains are now **realized**: T001–T003 implemented S1/S2/S3/S5's planned chains (flush buffer, shutdown flush, format dispatch, triggers), changing consumers (`on_cleanup` now cancels two tasks and flushes; `dump_turn` now dispatches). The pre-implementation compositions (Checkpoint 57, all SAFE, now published as `CO:` handles) were planned-chain readings; the recomposition consumes realized-chain evidence plus the MS-001/MS-002 yield records. Consumed-authorization members (15, 17, 21, 45–48) stay in their rows as evidence history — their falsification-by-design was adjudicated at the implementing tasks' yields.
- **S6 `inherited: S6`** — members 49 and 33 unchanged, `serve_buster_ripper.sh` and the README untouched (T005/T006 pending — verified by grep this run); no member, consumer or dimension moved.

## Dimension notes

- **S1** remains the plan's central chain; `PI:T002/dump_format/one_write_path_per_turn` is the composed property over rows 45, 4, 8, 12, 17, 47, 44.
- **S2** carries the realized teardown order (cancel kv → cancel flush → `_final_dump_flush` → close session — `server.py:69-78` as landed).
- **S3** and **S5** share mechanism rows (34, 40) with different consumers (format dispatch vs trigger sites).
- **S4** is T006's whole increment, still pending; its recomposition waits for the re-dispatched members' verdicts.
- **S6**'s `DUMP_FORMAT` half landed with accepted-local T005, its `SKIP_TLS_VERIFY` half with accepted-local T006, and its `DUMP_DIR` disable-form half with accepted-local T011 (registered 2026-10-04, accepted 2026-10-05 at [052-01-yield](../iterations/052-01-yield.md)) — the third boundary var's *disable* arm is the seam's discriminating case, so `lang:bash` joins its subjects and the header-documented `DUMP_DIR=""` form joins its dimensions. The landing is what fired the ten script-pinned members' byte-keyed fingerprints (run-25 changelog); the recomposition that closes C1/C3 landed `SAFE` on the post-fix bytes at run-26, and T012's assembly rewrite then fired the same set again (run-27/run-28), so the seam's third half — C2's value-word class — landed `SAFE` on the post-T012 bytes at run-29. **S6 is citable as current** on `registry-safe`: the registry's `CO:` row now carries the handle-bearing 25-member basis itself (run-29 changelog), with `fb247496…`/`dc54c8c5…`/`d20c1f6b…` retained as superseded history and C1/C2/C3 all closed.
- **S7/S8/S9** are pre-implementation seams for the pending remediation tasks: their compositions evaluate the planned fixes against verified mechanism facts (the aiohttp decode/timeout/ContentTypeError behavior) plus the in-repo family/precedent structure.

## Run-4 changelog (2026-09-27 — malformed re-extraction)

- **S7 `recheck`:** replace malformed historical member 52 with atomic rows 68–72. The hierarchy dimensions no longer use ambiguous “sibling” wording; the downstream `except` relationship is left to composition after fresh atomic checks.
- **S9 `recheck`:** replace malformed historical member 66 with production-only dispatch row 73. T008 relevance belongs to composition, not the atomic claim. S8 and all other seam membership unchanged.

## Run-5 changelog (2026-09-27 — absorbed atomic routing)

- **S3/S4/S5 `recheck`:** replace historical absorbed member 34 with import-bound default row 74. S3 adds runtime dump-format row 75; S5 adds separate interval and byte-threshold read rows 76 and 85. S4's new TLS consumer is planned rather than asserted as a present repo fact. The old blanket call-time claim remains terminal history.
- **S8 `recheck`:** replace historical absorbed members 60 and 62 with rows 77–82, separating ordinary exception fallback, cancellation ancestry, startup await, and override/cache/GET branches. The prior universal failure and unconditional probe assertions remain terminal history.
- **S9 `recheck`:** replace historical absorbed member 63 with rows 83–84, separating SSE rejection from compatible `+json` acceptance. S1, S2, S6 and S7 membership is unchanged.

## Run-6 changelog (2026-09-27 — realized-chain malformed routing)

- **S1 `recheck`:** replace historical pre-buffer rows 45, 4–7, 17, 47 with current writer, pending-buffer and batch-append rows 86–94, 102–103, 105. Recheck existing propositions 8–10, 12 and 20 against their current units; 11, 13 and 44 remain inherited. The superseded authorizations remain in `claims.md` as history.
- **S2 `recheck`:** replace historical no-shutdown-flush row 15 with current teardown rows 95, 97 and 104; add the actual flush task 96; recheck rows 14 and 16. The historical US-001 factual link to row 15 does not establish the current cleanup path. Preserve the normative choice; canonical verification reconciliation remains with the orchestrator after atomic verdicts.
- **S3 `recheck`:** replace pre-option absence rows 21 and 23 and obsolete inventory row 48 with current CLI/format rows 99 and 98, plus dispatch row 103. Rows 74, 40, 41 and 75 retain their existing verdicts. No cross-format runtime mutation is asserted.
- **S5 `recheck`:** replace no-periodic-flush row 46 and obsolete numeric inventory row 48 with current CLI and timer/threshold rows 100–102, joined to existing 74, 40, 43, 76, 85 and shared state/task rows 91, 96. The consumed absence rows stay historical.
- **S4 deferred:** its row and malformed composition history remain as recorded until T006 lands; no current TLS callback/connector implementation is claimed. S6–S9 unchanged.

## Run-7 changelog (2026-09-27 — absorbed atomic routing)

- **S1 `recheck`:** replace terminal historical member 20 with production reader absence 106 and documented-artifact test reads 107–108; replace terminal historical byte-accounting members 91/102 with present `_append_record` accumulator/comparison facts 109–110 and language distinction 111. Preserve old verdict/routing rows in `claims.md`; neither test reads nor an unimplemented UTF-8 repair are silently assumed away.
- **S5 `recheck`:** replace terminal historical members 91/102 with 109–111. The dimensions record the current character-count versus approved byte-volume boundary; T009's future behavior is an arbiter in the plan, not a present-code assertion or a verdict here. Other seam membership unchanged.

## Run-8 changelog (2026-09-27 — inserted T009/T010)

- **S1 `recheck`:** T009 consumes existing current-state accumulator and threshold facts 109–111 plus new UTF-8 encoded-byte length fact 112. The planned replacement of character counting is not asserted as existing implementation. T009 inherits state keying, pending append, single batch write and clear-after-success premises 12, 92–94.
- **S2 `recheck`:** T010 adds completed-task cancellation/await facts 113–114 and current helper exception-boundary fact 115 between spawned task 96 and sequential cleanup 97/95; batch write 93 supplies the artifact endpoint. Both task positions, ordinary failure versus cancellation, and cleanup propagation remain structural dimensions; no repair outcome is claimed.
- **S5 `recheck`:** T009 adds encoded-byte length fact 112 alongside present character-count accounting 109, inclusive threshold 110, and CLI/interval trigger chain. No interval-policy or format mutation premise is added. New seam members await blind checks, so prior absorbed composition is not inherited.

## Run-9 changelog (2026-09-27 — accepted-local T009/T010)

- **S1 `recheck`:** replace pre-repair character-count member 109 with current pre-mutation UTF-8 sizing and accumulation rows 116–117. Keep the encoded-length language premise 112, inclusive threshold 110, session keying, writer and flush members. The source facts are not composition verdicts.
- **S2 `recheck`:** replace pre-repair helper member 115 with ordinary failed-task isolation and outer-cancellation propagation rows 118–119. Both spawned task slots and the sequential flush/close path remain in the ordered chain.

## Run-14 changelog (2026-09-27 — accepted-local T003.001)

- **S7 `recheck`:** replace pre-remediation route pins 53–56 with current production parse-site census 129, per-route client exception mappings 130–136 and 138, aiohttp HTTPBadRequest transport 137, and upstream exception mappings 139–140. Re-dispatch the phase-exercised decode/hierarchy premises 51 and 68–72. The ordered chain and distinct syntax fallback/upstream attribution are recorded, not composed; previous S7 composition is not inherited.
- **S5 `recheck`:** replace pre-repair character-count member 109 with rows 116–117, retaining the CLI/config trigger path, inclusive comparison and timer dimensions. No interval or format-policy change is assumed.

## Run-10 changelog (2026-09-27 — absorbed claim-93 routing)

- **S1/S2 `recheck`:** replace unconditional historical member 93 with conditional collapsed batch append 120, documented guard-return 121, and documented-path no-buffer 122. Claim-93 remains terminal evidence history. The fixed-format witness is T004's arbiter; the seam records its scope without assuming format migration. No seam composition is performed here.

## Run-11 changelog (2026-09-27 — S5 malformed routing)

- **S5 `recheck`:** membership retains 85 as the runtime threshold-read premise, 116–117 as the post-T009 UTF-8 accumulation premises, and 110 as the inclusive comparison premise. Recheck 85 against the changed `_append_record` source in fresh bundle B93; its old character-count explanation is stale, not an atomic character-count claim. Historical row 109 remains excluded. No atomic verdict or composition is assigned here.

## Run-12 changelog (2026-09-27 — inserted T010.001)

- **S2 `recheck`:** add aiohttp runner cancellation/cleanup rows 123–124 and Python cancellation-count/child-await rows 125–126 around existing task-spawn, helper and flush members. Distinguish one signal's retained runner cancellation from a new cancellation during the child await, and active child cancellation from a completed ordinary failure. Row 119 is inherited current-source evidence, not the pending repair. No source check, composition verdict, or repair outcome is assigned here.

## Run-13 changelog (2026-09-27 — accepted-local T010.001)

- **S2 `recheck`:** replace historical absolute cleanup-cancellation predicate row 119 with current helper snapshot and count-increase rows 127–128. Preserve aiohttp runner rows 123–124, Python cancellation rows 125–126, both task slots, ordinary failure isolation and flush/artifact endpoint in chain order. The repaired source is only a premise; signal-process QA and post-check composition remain downstream. No source check or seam verdict is assigned here.

## Run-15 changelog (2026-09-27 — accepted-local T007)

- **S8 `recheck`:** replace the historical unbounded-request row 59 with request-local timeout row 141, and the stale three-site timeout census 61 with current package-wide row 144. Add caching and cancellation settlement premises 142–143 after the ordinary-exception boundary 77; re-dispatch exercised source premises 57–58 and 77–82 in fresh bundles. Preserve conditional override/cache priority, startup await, and the shared 600 s default. This is a source-premise delta only: atomic checks and post-check composition remain downstream, and accepted-local Yield does not clear the MS-002 milestone gate.

## Run-16 changelog (2026-09-27 — accepted-local T008)

- **S9 `recheck`:** replace historical unconditional chat JSON parse member 65 with current request-flag dispatch 145, successful stream byte relay and SSE response preparation 149–150, upstream error transport 151, and nonstream JSON parse/transformation 146–147/152. Add current client-error mapping 148; re-dispatch exercised aiohttp premises 64/83–84 in fresh B113. The unchanged messages dispatch/relay peer remains 73/67. This records the source boundary only; no atomic check, composition verdict, or milestone acceptance is assigned here.

## Run-17 changelog (2026-09-27 — inserted T008.001)

- **S9 `recheck`:** add upstream Content-Type access/absence 154/156, current chat error-label fact 153 and downstream `web.Response` header capability 155 between raw error transport 151 and the consumer. The response-media-type dimension now distinguishes an upstream media type from a forced JSON label and absent upstream Content-Type. The approved successful SSE and nonstream paths remain represented by existing members; no implementation result, atomic verdict or composition verdict is inferred.

## Run-19 changelog (2026-09-27 — accepted-local T005)

- **S6 `recheck`:** replace pre-T005 member 49 (stale in-row citations; its two-channel and no-positional conjuncts are re-stated atomically as 162–163) with the landed `DUMP_FORMAT` wiring rows 159–160, the `DUMP_DIR` peer-guard row 161, and current argv rows 162–163; re-dispatch the exercised README run-path row 33 and add collapsed-example row 164. Historical rows 32 and 49 remain in `claims.md` as terminal evidence. T006's `SKIP_TLS_VERIFY` clause is still pending, so the seam stays open beyond this delta. Source premises only; no atomic check, composition verdict or milestone acceptance is assigned here.

## Run-18 changelog (2026-09-27 — accepted-local T008.001)

- **S9 `recheck`:** replace historical forced-JSON-label member 153 with current conditional upstream Content-Type copy 157 and absent-header branch 158. Re-dispatch exercised raw status/body transport 151 and aiohttp header/response premises 154–156 in fresh subject-local bundles B120 (relay), B121 (upstream headers), B122 (downstream response). Preserve unchanged request-flag dispatch, successful SSE relay, nonstream JSON transformation and messages peer as existing members. This records the current source chain only; no source verdict or composition is assigned here.

## Run-20 changelog (2026-09-28 — accepted-local T004)

- **S1 `recheck`:** add the settled write-count instrument (DC-claim-3-r1) as now implemented in tests — rows 165–167 (`repo:tests.test_dump_write_count`: the `_WriteHook` counting predicate, the payload's UTF-8 byte measurement, the per-shipped-collapsed-format count runs), the `Path.write_text`→`Path.open(mode="w")` stdlib funnel premise 168, and collapsed-artifact test reads 169 — and re-dispatch the T004-exercised members 12, 13, 44, 86–90, 92, 94, 105–108, 116–117, 120–122 in fresh bundles B126–B140. T004 changed no production file: working-tree `dump.py`/`session.py`/`config.py` hashes are byte-identical to the recorded composition basis, but `routes/messages.py` drifted earlier at Checkpoint 84 (T003.001's client-error typing), which already fired the recorded composition's own Invalidates-On. The recorded SAFE composition stays on disk, unclobbered, and is not inherited: the member set changed here and no registry `CO:` record covers this member set. No composition verdict is assigned.
- **S2 `recheck`:** membership unchanged; members 120–122 (claim-93's corrected premises — the per-format flush coverage was T004's acceptance vehicle) return to blind checks. Production fingerprints re-verified as exact working-tree matches (`main.py` `cbe2802d…`, `server.py` `6a3e45ef…`, `dump.py` `07b0b076…`), so the in-session SAFE composition's source basis is intact and preserved; its own Invalidates-On ("any change to a listed member verdict") fires when the exercised members reopen, and no registry `CO:` record covers this member set, so State stays `recheck` and the recomposition consumes the fresh verdicts. State and Routing cells otherwise unchanged — composition routing is not this pass's to decide.
- **S3–S9:** outside the dispatch scope; rows untouched, and no unrelated composition was inspected for staleness.

## Run-21 changelog (2026-09-28 — malformed claim-167 re-extraction)

- **S1 `recheck`:** replace malformed member 167 (`UNCLEAR` — the "write-count tests" term admitted two scopes) with atomic rows 170–173: the exact four `COLLAPSED_FORMATS`-parametrized classes, the test-constant↔`config.DUMP_FORMATS` equation, and `TestWriteUnit`'s and `TestDocumentedCount`'s single-format runs. S1's `recheck`/`unverified` cells are unchanged; the recomposition rides the composition boundary awaiting the fresh member verdicts. All other seam membership unchanged.

## Run-22 changelog (2026-09-28 — accepted-local T006)

- **S4 `recheck`/`unverified`:** the deferred T006-boundary refresh. Membership drops consumed-by-design absence row 24 (its `CL:buster_ripper/tls-surface` Invalidates-On fired by design; the row stays in `claims.md` as terminal evidence) and adds the landed chain rows: the flag declaration 175, its callback assignment 176, the `config.SKIP_TLS_VERIFY` default 174, the `TCPConnector(ssl=not config.SKIP_TLS_VERIFY)` construction 177, the session's connector adoption 178, and the per-request `ssl` absence 182. Exercised/stale members 25, 26, 37, 39, 40, 50, 74 re-dispatch in fresh bundles B143–B148; the dimensions add the no-per-request-override conjunct of the process-wide scope. The corrected premises of claims 38 and 34 are rows 50 and 74 — both return to blind checks; the yield's PASS/PASS+CONFIRMED replays are provenance, not verdicts. No registry `CO:` record covers this member set (the on-file `seam-S4.md` composition is `UNCLEAR`/INELIGIBLE pre-implementation history); the recomposition waits for the fresh member verdicts.
- **S6 `recheck`/`unverified`:** T006 landed the seam's third boundary var. Membership adds the `SKIP_TLS_VERIFY` init 179, its boolean `== "1"` guard append 180, and the README skip-TLS run-path row 181; the dimensions now distinguish the boolean value-guard shape from the `DUMP_FORMAT`/`DUMP_DIR` presence-guard shape. Both source fingerprints moved (`serve_buster_ripper.sh` `adbe88be…`→`f4076d1c…`, `README.md` `cfad3093…`→`fe80cb9a…`), firing the recorded SAFE composition's own Invalidates-On ("member re-extraction and re-verification required before recomposition"), so all prior members 33, 159–164 re-dispatch in fresh bundles B149–B152 with handles retained. The `CO:serve-script-env-to-argv/env-var-flag-crossing` handle stays in `Permanent` with State `recheck`: its member set excludes the three new members and its basis fingerprints are stale, so it is neither reused nor discarded. Recomposition consumes the fresh verdicts.
- **S1–S3, S5, S7–S9:** outside the dispatch scope; rows untouched, and no unrelated composition was inspected for staleness.

## Run-23 changelog (2026-09-28 — stale-composition re-base after T006's main.py change)

- **S3/S5 `recheck`/`unverified` (cells unchanged — the recomposition basis moves):** both on-disk SAFE compositions (2026-09-27, both `PROMOTION: INELIGIBLE`) fired their own Invalidates-On at T006's `main.py` change (`cbe2802d…` → `fd36dc12…`, working-tree hash re-verified this run): S3's member fingerprint list pins `cbe2802d…` on rows 99/74, and S5's Invalidates-On names `main.py` flush-option declarations. Per the claim-34 hold note (handoff §Post-acceptance Finding Resolution — 2026-09-28 (T006), resolution trigger), the current member premises are now persisted in `claims.md`: **rows 74/40 are fresh-HOLDS** (run-22 re-dispatch verdicts postdate the change — `CL:buster_ripper.main/cli-default-binding-time` @ `fd36dc12…`, `CL:click/option-callback-timing` @ click core `53a05fc6…`, both 2026-09-28; no new extraction owed, no checker spent), and **rows 99/100 return to blind checks** in fresh bundle B153 (their verdicts and registry handles pin `cbe2802d…`; handles retained, `T006` appended to Tasks).
- **Membership unchanged:** S3 (99, 74, 40, 41, 98, 75, 103) and S5 (100, 74, 40, 43, 76, 85, 116, 117, 111, 112, 96, 101, 110) — exactly the hold note's sets; no member added or dropped. The other 14 members inherit (subjects' named units/aspects untouched by T006: `dump.py` `07b0b076…` byte-identical; `server.py`'s edit confined to `on_startup`, so `spawn_dump_flush` does not fire; `pkg:`/`lang:` identities unchanged). No registry `CO:` record covers either member set, so Permanent stays `—`;

## Run-24 changelog (2026-10-04 — scoped adhoc extraction, T011 / S6)

- **S6 `recheck`/`unverified` (cells unchanged; membership, boundary trace and dimensions all move).** Registered T011's premises land here: twelve new members — the parameter-expansion semantics of the step that discards or keeps the disable discriminator (183 `:-`, 184 `-`), the environment-prefix assignment form the documented form uses (185), nounset safety for the unset case (186), the presence-guard's discriminator (187) and the false arm's errexit exemption (188), the unquoted-`EXTRA_FLAGS` zero-word rule (189) and `$PATH` first-match resolution (190, the witness's interception premise), the script's own documented promise (191), interpreter line (192) and bare exec-line command word (193), and the README example inventory that bounds the untouched-documentation surface (194). `lang:bash` joins the subjects; the boundary is re-worded to name parameter expansion as a traced transformation; four dimensions are added (unset/set-but-empty discriminator; `:-` versus `-` form; presence-guard false branch under `set -e`; zero-word expansion; header-promise versus README-inventory). Members are re-ordered producer → transformation → consumer over the 22-row set; no existing member was dropped.
- **Existing members inherited without a checker (10):** 159–163, 179–181 and 33/164/181 — `serve_buster_ripper.sh` is re-verified unchanged at `f4076d1c…` (46 lines) and `README.md` at `fe80cb9a…`, exact matches for the fingerprints their 2026-09-28 verdicts pin, with no dependency or domain change; §3d preserves each `HOLD`/`none` cell and appends `T011` to `Tasks`. The seven script-pinned members' re-verification is triggered by the line-28 *edit* (the composition's own Invalidates-On: "line 28's default form changing in a way that alters guard reachability"), not by this dispatch, so it is not re-opened against unchanged bytes here.
- **Composition Index consulted for S6's set only:** `CO:serve-script-env-to-argv/env-var-flag-crossing` describes the pre-run ten-member set; the 22-member set, the added subject and the added dimensions are not in it, so freshness cannot be established and the state stays `recheck` with the handle retained in `Permanent`. The on-disk SAFE composition (2026-09-28) is history whose Invalidates-On names this edit. No other composition was inspected for staleness.
- **S1–S5, S7–S9:** outside the dispatch scope; rows untouched.
- **Nothing dropped:** no member was removed from S6, no verdict or routing was reset, and the two unchecked new rows (claims.md 195, 198 — affirmative single-task claims outside every seam) remain write-time checks for Forward rather than disappearing. the fired SAFE compositions stay on disk as history, neither inherited nor discarded. Recomposition consumes B153's fresh verdicts plus the fresh-HOLDS members; when both are SAFE, claim-34 resolves per the hold note — recorded for provenance, not judged here. No atomic check or composition verdict is assigned by this run.

## Run-25 changelog (2026-10-05 — accepted-local T011 / S6 scoped recheck)

- **S6 `recheck`/`unverified` (cells unchanged; the landing this seam was waiting for happened).** T011's one-character edit — line 28 `${DUMP_DIR:-…}` → `${DUMP_DIR-…}` — moved `serve_buster_ripper.sh` from `f4076d1ceae359f2…` to `sha256:d5db2144a2452840e7a5284c524c3fbfeef60dc117a55d61fab57a5aeafc5c90` (46 lines, line numbering unchanged; the file is committed, worktree clean). That is the composition's own Invalidates-On ("line 28's default form changing in a way that alters guard reachability") firing on the event run 24 recorded as future, and it is why the boundary cell now reads `DUMP_DIR`, whose default form landed with T011. **Membership, producer→transformation→consumer ordering and the ten dimensions are unchanged** — the chain itself moved, its shape did not, so no member is added or dropped and `s = 1` seam is still the same seam awaiting a recomposition it cannot be cited from yet.
- **Members re-dispatched fresh (10, edited in place in `claims.md`, new bundles B162–B165, `CL:` handles retained):** **159, 160, 161** (B162 — the two env inits at lines 28–29 and the two `-n` appends at lines 36–37, one guard-block read) and **162, 163, 193** (B163 — the `exec` statement at lines 42–46 and the positional sweep; 193 names that same line, so it left B157) and **179, 180** (B164 — line 27's init and line 35's boolean append) and **191, 192** (B165 — the usage header at lines 11–16 and the shebang at line 1). All ten move to `recheck`/`unverified` because each verdict's `## Source Fingerprint` pins the expired bytes; row 161 self-flagged this precise change for re-extraction rather than silent reuse, and rows 159/160 name line 29's default and line 37's guard as their own invalidation surface. This pass dispatches them; it does not answer them.
- **Members inherited without a checker (12):** 183–190 (`lang:bash`, B154–B156) — no bash documentation or dependency moved, and the edit is the `:-`-versus-`-` contrast rows 183/184 already state; 33, 164, 181, 194 (`repo:README.md`, B152/B158) — `README.md` re-verified byte-identical at `fe80cb9acabfe699…`. T011's `Done when` names both sets as carrying forward into the recomposition, which re-traces dimensions 2/3/6/7/8/9 on the new bytes.
- **Non-member ledger movement recorded here because the seam consumes it:** claims.md row **199** is superseded terminal evidence (`DOES-NOT-HOLD`/`absorbed`, T011's acceptance) — its Invalidates-On fired by design when the witness module landed — with successor rows **200** (the surviving test-surface boundary: the script's only consumer under `tests/`) and **201** (the witness's whole-word-set assertion shape, which the `--dump-dir`-absent state depends on) newly extracted and dispatched in B166/B167; and rows **195**/**198** are settled `HOLD`/`none` from the Forward write-time check that Yield re-read independently, per run 24's recorded route for them. None of the three is an S6 member, so S6's member list is untouched by this paragraph.
- **Composition Index consulted for S6's set only:** the index row `CO:serve-script-env-to-argv/env-var-flag-crossing` does carry this seam's current 22-member set, verdict `COUNTEREXAMPLE`, fingerprint `sha256:fb24749696624db45f54b5901c946c395afaaec6444d46d9306c3e46b5d43c3e` — the same record as `assumptions/compositions/seam-S6.md` (2026-10-05, `PROMOTION: ELIGIBLE`), established on `serve_buster_ripper.sh` `f4076d1c…` and `README.md` `fe80cb9a…`. Its counterexample **C1** (with **C3** as its reachability conjunct) is the finding T011 was created to close, so this is a matching record whose **freshness cannot be established**: ten of its members are now `recheck` on `d5db2144…`, and the boundary/source bytes its fingerprint hashed are the pre-fix ones. State therefore stays `recheck`/`unverified` with the `CO:` handle retained in `Permanent` — neither reused as current nor discarded — and its own Invalidates-On has fired ("line 28's default form changing in a way that alters guard reachability"). **S6 is not current until a fresh recomposition lands on the post-fix bytes**; T011's `Done when` says so and T012's `Execution order` gates on it. No other composition was inspected for staleness.
- **S1–S5, S7–S9:** outside the dispatch scope; rows, members and cells untouched. No composition was run, no member's proposition was evaluated, and no seam was called safe or unsafe by this pass.

## Run-26 changelog (2026-10-05 — post-acceptance resolution routing, seam **S6** only)

- **S6 `recheck`/`absorbed` (Routing moves; State deliberately holds).** The recomposition run-25 named as owed has landed:
  `assumptions/compositions/seam-S6.md` is now `VERDICT: SAFE`, checked `2026-10-05T03:47:59Z`, a full ten-dimension re-trace on the post-T011
  bytes (`serve_buster_ripper.sh` `d5db2144…` + `README.md` `fe80cb9a…`, both re-hashed against the worktree at this routing pass), 22/22
  members `HOLDS`, basis fingerprint `sha256:dc54c8c5ccf45b39400fc240ee62b006a997633b0817050b008fbffc55c84d71`. Routing therefore reads
  `absorbed`: the seam's routed findings are `seam-S6-disable-form` (absorbed → T011, now **resolved**) and `seam-S6-value-word-split`
  (absorbed → T012, still **open**). **State stays `recheck`** on purpose, and not because a verdict is missing: the registry's
  `CO:serve-script-env-to-argv/env-var-flag-crossing` is the only row a `registry-safe` cell may cite, and its basis is superseded — see the
  next bullet. Per §3c of the collecting contract a stale matching record keeps State at `recheck` with the handle retained in `Permanent`,
  which is exactly this case.
- **`CO:serve-script-env-to-argv/env-var-flag-crossing`: basis disposed as retained history, publication routed, handle retained.** The
  record still reads `COUNTEREXAMPLE` @ `sha256:fb247496…` (checked `2026-10-05T02:52:00Z`) on `f4076d1c…` script bytes; two of its own
  `Invalidates-On` clauses have fired ("either production source off its digest"; "T011's edit itself — line 28's expansion form"), and its
  `Provenance` path now carries a different verdict and fingerprint. The composition-promote decline was correct on its own gate, not an error
  to work around: the artifact's line 2 is `PROMOTION: INELIGIBLE` (its basis is digest-only — the composition role cannot read the registry to
  confirm member handles), and ineligible-but-valid session verdicts stay valid for the session. Disposition: **retain + route**, no
  `assumptions/` cell and no registry write by this pass. Next dispatch: re-run **S6's composition with the 22 `CL:` handles supplied in its
  arguments** (verified 22/22 present this pass and already ordered producer → transformation → consumer in the record's `Members` line), which
  is the only material the basis lacks; `composition-promote` then upserts the record in place under the same handle and rebuilds the index.
  **Trigger: fold into T012's mandated recomposition** rather than dispatching separately — T012's assembly rewrite fires this seam's own
  Invalidates-On, so one handle-bearing pass carries the post-T011 basis and supersedes it with the post-T012 basis, and a basis republished
  today would be stale within one wave. **Until that upsert lands the record is not citable as current** and this seam's live verdict is the
  on-disk `SAFE`. Registered as `QB:assumption-registry/co-serve-script-basis-republish`.
- **C1/C3 closed; C2 expressly not.** Resolution authority: the terminal condition is the affected seam's post-implementation composition being
  `SAFE` (§Durable Transmission). The new record closes C1's chain at the step that consumes the discriminator (dims 6/7/8/10; its §Counterexample
  reads "None found" and names C1's predecessor as closed on `d5db2144…`). C2's class is **not** closed by it — the value-grammar case is
  recorded there as a *named non-finding* (no member claim makes it normative) while T012's authority is the consumer contract
  (`buster_ripper/main.py:78` `metavar="PATH"` + `:152` `Path(dump_dir).expanduser().resolve()`); so T012's `Done when` clause about a
  recomposition closing C2's class remains unsatisfied, and the missing atomic premise (one argv word per forwarded value; admissibility of `$IFS`
  whitespace, glob metacharacters, a leading dash) returns to **Forward Collect at T012's wave** so the post-T012 composition can trace it as a
  dimension instead of declining it on premise grounds. `claims.md` member cells are untouched (all 22 already `HOLD`/`none`).
- **Consequence for the queue:** S6 is citable as current, so **T012 is dispatchable** (its `Execution order` gate was T011 accepted + this
  recomposition recorded). `MS-004`'s QAP-MS004 author pass remains the next **milestone** work; the reachability-gated QAP-MS002 re-run →
  MS-002 re-verdict is unchanged per [US-016](../decisions.md#us-016).
- **S1–S5, S7–S9:** outside the dispatch scope; rows, members and cells untouched. No composition was run or re-verified here, no member's
  proposition was evaluated, and no other seam or registry record was inspected for staleness. Full routing: [handoff-context.md](../handoff-context.md)
  §Post-acceptance Finding Resolution — 2026-10-05 (T011: `seam-S6-disable-form` → resolved; `CO:` basis disposed).

## Run-28 changelog (2026-10-05 — scoped composition-blocker extraction, seam **S6** only)

- **S6 members 23 → 25; dim 9 stops being memberless.** `claims.md` run 28 adds rows **203** (bash's `${array[@]+word}` alternate-value expansion contributes zero words when the array is unset or has no elements) and **204** (under `set -u`, that same setness test raises no nounset error) — the two atoms of the guard semantics the ninth dimension ("zero-word arm when every append is suppressed") names as its mechanism and previously had no row for. Both are `lang:bash`, `single` (T012), `new`/`unverified`, bundled together as **B173** — a fresh key, since `B172` (189/202) is closed with both members now `HOLD`. The dimension cell is re-worded to cite them; no other dimension, member, or subject changed.
- **Why two rows rather than the one the dispatch phrase suggests.** "Zero words vs nounset abort" is the dimension's alternative, not a proposition: the manual settles the arm the alternate-value test takes in §3.5 and the nounset exemption in the `set -u` entry, and this ledger already splits the same pair for the default-value forms (183/184 state substitution, 186 states the error exemption). One row would carry two propositions under a single verdict (E5). Both rows describe the **guarded** form, which is the form at the boundary post-T012; the *unguarded* bare `"${array[@]}"` abort, which depends on which interpreter `/usr/bin/env bash` resolves to, stays off this ledger as arbiter-settled repo state (E2/E4), as run 27 routed it — recorded here so the omission is visible, not filtered (E6/E7).
- **State, Routing, Permanent left as found.** State was already `recheck`, which a widened member set preserves. `Routing: absorbed` and `Permanent: CO:serve-script-env-to-argv/env-var-flag-crossing` are untouched: the recorded basis covers the 22- and 23-member sets, so it cannot be current for a 25-member set, and the handle stays retained per §3c's stale-record row. Whether the widened chain composes is the composition pass's call and is **not** made here (E8); the next recomposition should be dispatched with the `CO:` handle in its arguments, per run 27's rider 1. No other seam (`S1`–`S5`, `S7`–`S9`) was read for this pass beyond its own row, and no unrelated composition was inspected for staleness.

## Run-29 changelog (2026-10-05 — post-acceptance resolution routing, seam **S6** only)

- **S6 `registry-safe`/`absorbed` (State moves; Routing holds).** The recomposition run 27 and run 28 owed has landed and been published:
  `assumptions/compositions/seam-S6.md` is `VERDICT: SAFE`, `PROMOTION: ELIGIBLE`, checked `2026-10-05T05:19:41Z`, a 25-member
  handle-bearing re-trace on the post-T012 bytes (`serve_buster_ripper.sh` `c58256fb…` + `README.md` `fe80cb9a…`, both re-hashed by the
  composition pass and again by this routing pass against the worktree), basis `sha256:cb7c0a5c82e5ac1b5f33738a61b582d925be4502a51a6ed1484e2ab415b60631`
  — the header of which is the 25 `CL:` handles plus their verdict-recorded digests in ledger order, the component the `dc54c8c5…` basis lacked.
  Dims 3 and 9 are re-traced in full and dims 1–2 and 4–8 and 10–11 are carried on digests this pass re-confirmed identical, so **all eleven
  dimensions** rest on current bytes. `CO:serve-script-env-to-argv/env-var-flag-crossing` was upserted **in place** under the same handle
  (`Verdict: SAFE`, `Counterexample: none`, `Fingerprint: sha256:cb7c0a5c…`, `Checked-At: 2026-10-05T05:19:41Z`, 25 handles in ledger order), and
  `invariants/index.md` §Composition Index was rebuilt from it. §3c's condition for a `registry-safe` cell — matching member/source fingerprints,
  no fired `Invalidates-On` — is therefore now satisfied, which is exactly the condition run-26 recorded as *missing* when it held this row at
  `recheck` with the handle retained; `Routing: absorbed` is unchanged because that column records the disposition, never the lifecycle.
- **`seam-S6-value-word-split` → `resolved`; `seam-S6-disable-form` stays `resolved`.** The terminal condition both routes stated (§Durable
  Transmission; T012's `Done when` recomposition clause; [053-01-yield](../iterations/053-01-yield.md) §Routing Decision) is the affected seam's
  **post-implementation composition being `SAFE` on the post-task bytes** — and this record is that verdict: C2's class is closed as a *traced
  dimension* (dim 10, resting on member row 202), not as the named non-finding run-26 declined it on, and the dim-9 undecidability that made the
  `d20c1f6b…` record `UNCLEAR` is separated by rows 203/204. This is a lifecycle completion of `DC-seam-S6-value-word-split-r1`'s own tail
  conditions, not a correction, so **no `-r2`** is created — per the DC-claim-38-r1 / DC-claim-34-r1 / DC-seam-S6-disable-form-r1 precedent.
- **The wave's three riders are discharged, and the registry marker is retired.** (1) members **159–163, 179, 180, 191–193** re-verified fresh on
  `c58256fb…` and **189**/**162** re-checked on the current digests — all ten script-pinned verdict records now pin `c58256fb…` and the four
  README-pinned ones `fe80cb9a…`, re-read at this pass; (2) the recomposition dispatched **with the `CL:` handles in its arguments** and
  `composition-promote` upserted the record in place; (3) Forward Collect rowed the value-grammar premise (**202**) and run 28 rowed the dim-9 arm
  atoms (**203**/**204**) that the recomposition needed to decide rather than decline that arm. `QB:assumption-registry/co-serve-script-basis-republish`
  is therefore marked **RESOLVED** in [quality-backlog.md](../quality-backlog.md), and the run-26 interim rule ("this record is not citable as
  current until that upsert lands") is retired by the entry above it.
- **Invariant cross-reference re-run at this pass: Neutral.** The session declares no `invariants.md`; the registry holds **zero `Status: active`
  and zero `Status: wip`** entries — eight `Status: planned` PIs (`PI:T002/dump_format/*`, `PI:T006/upstream_tls/*`), the preserve-provenance `OI:`
  blocks, and descriptive `E:`/`CL:`/`CO:` rows, none addressing a shell or argv surface. Nothing contradicted, displaced, upheld, activated or
  created. The candidate recorded with the finding — *"Every environment value `serve_buster_ripper.sh` forwards reaches the `buster-ripper` argv as
  exactly one word; the wrapper adds no value protocol of its own"* ([handoff-archive.md](../handoff-archive.md) §Trimmed 2026-10-05, T012 block) —
  stays **non-authoritative and is Interview's to record or promote**; this pass promotes nothing by inference from a landed verdict.
- **`claims.md` gets no run 29.** This pass writes no claim row and touches no cell in that ledger: all 25 S6 members already read `HOLD`/`none`,
  no row carries `pending` Routing, and resolution consumes verdicts rather than producing them.
- **Consequence for the queue:** the assumption route for T012 is closed and nothing in this family is dispatchable. Unchanged and still owed:
  `MS-004`'s QAP-MS004 author pass (`forward-qa`) is the next **milestone** work; the reachability-gated QAP-MS002 re-run → MS-002 re-verdict stands
  per [US-016](../decisions.md#us-016); and the three `P3`s deferred at [053-01-yield](../iterations/053-01-yield.md) keep their owner (the next touch
  of `tests/test_serve_script_argv.py`). This resolution moves no milestone gate. Full routing: [handoff-context.md](../handoff-context.md)
  §Post-acceptance Finding Resolution — 2026-10-05 (T012: `seam-S6-value-word-split` → resolved; `CO:` basis republished).
- **S1–S5, S7–S9:** outside the dispatch scope; rows, members and cells untouched. No composition was run or re-verified here beyond S6's own
  record, no member's proposition was evaluated, and no registry write or invariant promotion was performed by this pass.

## Run-30 changelog (2026-10-05 — scoped adhoc extraction, **T013** · header-parity wave)

- **Dispatch:** claims run 30's scoped adhoc extraction for `T013` — member **F-4** (five-site header-filter substitution through `buster_ripper/utils.py`'s shared helpers), member **G-1** (post-`prepare()` error handling at the `routes/messages.py:154` commit boundary), member **G-2** (`REFACTORING_PLAN.md`'s phantom `convert/` rows) — plus the aiohttp response-commit/second-head premises, plus the recorded **`seam-S9` rider**. Extraction only: this pass checked nothing, composed nothing, and called no seam safe (E1/E8). Full basis, row-shape notes, bundle table, freshness basis and the not-rowed list: [claims.md](claims.md) §Run 30.
- **S10 `new`** — the header-parity chain F-4 moves: a header name set crossing the client→proxy boundary into the upstream request and the upstream→proxy boundary into the downstream response, with `utils.forward_headers`/`utils.response_headers` as the single policy owner. 27 members (claims 205–224, 234–238, 246–247), six `repo:` subjects, two `rel:` subjects (the RFC 9110 §7.6.1 target, and the base-vs-branch set equality), `pkg:aiohttp@3.13.3` for the transport's own header spelling and header recomputation, and `repo:tests` for the missing witness. The packet's own surface claim — "no seam named by the packet; `buster_ripper/utils.py` and the header-filter sites sit outside every recorded seam" — is what this row discharges: the sites were outside every *recorded* seam, so the wave gains one rather than inheriting one.
- **S11 `new`** — the response-head chain G-1 moves: the head crossing the handler→aiohttp-server boundary at the `prepare(request)` commit, and the exception channel leaving the relay loop through the handler's single `except` arm or out of the handler entirely into the server protocol's own generated error response. 18 members: the two library-side commit premises plus the two fresh hierarchy premises (claims 226–233), the two `lang:python` premises behind the upstream-controlled-payload case (239/240), **78** inherited (its `CancelledError` proposition is what the fourth case-table row takes for granted, `B110` retained, `HOLD`/`none` preserved, `T013`/`S11` appended), the five handler-shape premises (241–245, 248), and **67** shared with S9 — the row that names the committed `StreamResponse` whose head the second-head premises are about.
- **S9 `recheck` — the rider, on current bytes.** State was already `recheck` and stays `recheck`: a member re-dispatch preserves it, and this pass assigns no verdict. What the rider changes is evidence. `verdicts/claim-67.md` and `verdicts/claim-73.md` pin `sha256:9c234c4f183aa03a…` for `buster_ripper/routes/messages.py`, which is now `sha256:721a2afe4d1ca35d…`; the single post-verdict change was `67ced13`'s edit at line 28, inside row 73's own named unit (`handle_messages`) and inside the file both rows' `Invalidates On` names in words ("Any change to `buster_ripper/routes/messages.py`"). `seam-S9.md`'s §Sources range for this file — `messages.py:65-70,73-85,121-176` — is edited on both sides by the wave: F-4 at `:62-63` (inside `65-70`'s unit) and G-1 in the `:121-176` relay section, which is where the rider's named range sits. So both members move `HOLD`/`none` → `recheck`/`unverified` in fresh bundle **B196**, their `CL:` handles retained, and `S9` is **not citable as current** until B196 returns and the seam is recomposed — the same post-implementation route T008/T008.001 used (run-16/run-18). `Routing` (`unverified`) and `Permanent` (`—`) are left as found; the `## Composition Index` holds no record for S9's member set, so there is no `CO:` handle to retain or to fire.
- **Chat-side S9 members not fired.** `buster_ripper/routes/chat.py` reads `sha256:16c7e99938b55e50…` this run, byte-identical to the fingerprint rows 145–148, 149–152, 155, 157 and 158 pin, and none of those rows' units is the `_prepare_headers` filter site F-4 substitutes — so they keep their verdicts, and 154–156/64/83 stay current as recorded.
- **No sweep of unrelated seams.** §3e's no-sweep rule was honored: `verdicts/claim-13/-15/-17/-46/-53/-54.md` carry the same expired `9c234c4f…` `messages.py` fingerprint but belong to S1/S2, which this dispatch does not name. Recorded here so the observation survives, and left to the owning route's next S1/S2 pass rather than edited on this one.
- **S1–S8:** rows, members and cells untouched (S9's cells above are the rider's, and S11 gains shared membership with row 67 only). No composition was run, no member's proposition was evaluated, no `CO:` record was inspected for staleness beyond S9's own set, and no registry write or invariant promotion was performed by this pass.

## Run-31 changelog (2026-10-05 — scoped adhoc extraction, **T014** · image-slice wave)

- **Dispatch:** claims run 31's scoped adhoc extraction for `T014` — member **F-1** (detector scope across all user messages), member **F-2** (content-addressed placeholder keys on `block["source"]` under `063 R-A`'s canonical-serialization binding), member **F-3** (write-path tail global sweep under `R-C`'s absolute-age semantics) — **plus `R-B`'s W1–W4 placement witnesses and `065-forward-red`'s W5 recommendation**, over `buster_ripper/image.py`, the `routes/messages.py` call sites, `normalize.py`'s step ordering and the `tests/test_image.py` pins. Extraction only: nothing checked, nothing composed, no seam called safe (E1/E8). Full basis, row-shape notes, bundle table, freshness basis and the not-rowed list: [claims.md](claims.md) §Run 31.
- **The `required:new-surface` claim the task row carries is discharged here.** `buster_ripper/image.py` appeared in no seam's members and in no composition's §Sources before this pass (the sole prior mention was row 48's enumeration of CLI-wired numeric globals), so the wave gains three seams rather than inheriting one — the same form the header-parity wave used for its own new-surface claim (S10/S11 above).
- **S12 `new`** — the strip-scope chain F-1 moves: the detector boolean and the user-message content list crossing `--strip-images` → `config.STRIP_IMAGES` → either guarded call site → detector → writer → the forwarded body and the changes log. 19 members (claims 253–262, 264, 265, 268–271, 290, 316, 317), twelve subjects. The chain is a seam rather than a row-set because the wave's whole premise — one detector scope, one writer scope — is the *composition* of the detector's abbreviation (253), the writer's breadth (256), two guards with different reachability (261/264 with 268) and a help text (269): each is individually true while the flag's documented scope is not, which is exactly what an atomic verdict cannot settle (E8).
- **S13 `new`** — the placeholder-identity chain F-2 moves: a client-supplied `source` mapping crossing canonical serialization → digest truncation → cache key → placeholder text → the forwarded body and the KV prefix. 20 members (266, 267, 272–274, 277, 279–292, 309), seven subjects, spanning `lang:python` (`json`, `hashlib`, `dict` semantics) and `repo:` units on three different modules. `R-A`'s binding makes this the wave's sharpest seam: the referent is `block["source"]` **because** of an ordering fact about another module (266/267), and the case table's discriminating row is a *cross-site* equality (stripped at `messages.py` versus stripped at `normalize.py` → byte-identical), which no single subject can answer.
- **S14 `new`** — the cache-lifecycle and error-transport chain F-3 moves: per-entry retention stamps and session-mapping liveness crossing the writer's tail sweep into module-level state shared by concurrent handlers, with the exception channel out of that same call to the HTTP status. 29 members: 25 new rows plus the four inherited ones — **11** (the system-prompt keying behind "each distinct system prompt adds a key"), **48** (the `IMAGE_CACHE_TTL` CLI wiring that makes the control *advertised*), **2** (the runtime-dependency set behind hand-rolling a TTL), and **230** (handler raises → aiohttp's own error response, the transport half of `R-B`'s HTTP-500 sentence, already queued in run 30's **B184** and re-bundled by neither wave). W1–W4's placement axes and the W5 residue named by the dispatch are the seam's dimensions rather than rows: the position, the scope, the stamp semantics, the comparator strictness, the drop's owner, the read-path immunity and the arity of every unpack are all individually true facts that only compose into an answer at the boundary.
- **No existing seam moved.** The wave's declared surface is `image.py` and `tests/test_image.py`; no S1–S11 member unit lives in either, and the four membership edits add rows to *new* seams only — an existing seam's member set is unchanged, so none fires to `recheck` on this pass's account. S11's cells are untouched even though S14 shares member 230 with it (shared membership is not membership change), and S9's rider from run 30 stands exactly as recorded.
- **Registry consulted for these sets only.** The `## Composition Index`'s four records cover runner-teardown, TLS posture, dump format and serve-script env — none names an image, placeholder-identity or cache-lifecycle boundary — so S12/S13/S14 are `new` with `Permanent: —` and no composer is skipped. No `CO:` record outside S12–S14 was inspected for staleness, no composition was run, no member's proposition was evaluated, and no invariant was created or promoted.

## Run-32 changelog (2026-10-05 — routing dispositions on runs 30/31's seams)

- **Dispatch:** failed-assumption and unsafe-composition routing over `compositions/seam-S11.md` (`MALFORMED`),
  `seam-S12.md` (`MALFORMED`), `seam-S13.md` (`MALFORMED`) and `seam-S14.md` (`COUNTEREXAMPLE`, five live findings
  CE-1…CE-5), plus the atomic rows those seams name. Routing only: nothing was composed, re-checked, reworded in
  place, or written to production code; no invariant was created, elevated or promoted, and no `CO:` record outside
  these four was opened (§3e's no-sweep rule).
- **S11 `malformed`** — its blocking member is atomic row **232** (`UNCLEAR`, two referents), which this pass returns
  to `forward-collect`; the record's own §Invalidates On item 1 is the routing instruction, and the composer's item 4
  (members **233** and **248** carry no `sha256:` for their consulted bytes) is a promotion condition, not a
  composition one, so it travels to `QB:assumption-registry/run-31-verdict-fingerprints`. S11's `State` stays `new`:
  a disposition is a Routing write, never a State write.
- **S12 `malformed`** — blocking member atomic row **256** (`UNCLEAR`, fork on "processes": visits versus rewrites),
  returned to `forward-collect`. The composer's §"What is required" is adopted verbatim as the collector's brief:
  one proposition per row, then re-trace all ten dimensions over the full member set.
- **S13 `malformed`** — blocking members **273** (`UNCLEAR`, fork on "payload"; its own verdict names the rewording
  that settles it) and **289** (`DOES-NOT-HOLD`, routed `absorbed` into `T014`/F-2's premise correction). Both cells
  now read `malformed` / `absorbed` respectively in `claims.md`; the seam stays un-composable until the replacement
  for 273 lands **and** 289's replacement row is checked, because a falsified member may not be smuggled into a
  seam as an alternate premise.
- **S14 `absorbed`** — a `COUNTEREXAMPLE` is not a design ruling. All five findings route into `T014`'s already
  settled closures rather than around them: **CE-1** and **CE-4** add requirements the wave's own Required-behavior
  sentences already compel (no flag-conditional error class; a TTL that actually bounds retention), **CE-2**,
  **CE-3** and **CE-5** are the composition-level confirmation of `057` F-2/F-3's premises with one accuracy
  correction each carried to Forward. Ledger: `DC-CE-1-r1` … `DC-CE-5-r1`, plus `DC-claim-289-r1`. S14's `State`
  stays `new` and its record stays **not citable as current**: `Forward`'s edits to `image.py`/`tests/test_image.py`
  fire 17 of its 29 members anyway, so the recomposition is owed at the accepted-local recheck on the post-wave
  bytes, exactly as S3/S5/S6 were.
- **S10 stays `unverified` — recorded as blocked, not disposed.** It has no composition verdict to route; five of its
  27 members (208, 216, 223, 237, 238) were in a non-holding state at dispatch and are now routed (four `absorbed`,
  one `malformed`), so the seam becomes composable once row 216's replacement returns a verdict. Dispatching a
  composer before that would reproduce S13's failure mode: a member set that still contains an unresolved row.
- **Consequence for the queue:** both waves keep their membership, order, owning route and separation reasons
  unchanged, and **neither Forward is authorized** — `forward-collect` must return for rows 216, 232, 256, 273, 312
  (plus the three fired re-dispatches 2, 48, 313), then the checkers for the replacements, then composition of
  S10–S13. Full routing: [handoff-context.md](../handoff-context.md) §Deterministic Closure Ledger — 2026-10-05
  (repair round 3) and §Assumption Routing — 2026-10-05 (claims run 32).
- **S1–S9:** rows, members and cells untouched. No existing seam's member set changed on this pass's account — the
  routed rows move dispositions, not membership.

## Run-33 changelog (2026-10-05 — scoped ledger amendment: post-routing re-extraction of runs 30/31's malformed and absorbed rows)

- **Dispatch:** claims run 33's scoped amendment — re-row the five `Routing: malformed` rows (216, 232, 256, 273, 312)
  to one proposition per row using the rewording brief each retired row's own verdict names, replace the five
  `absorbed` `DOES-NOT-HOLD` rows whose corrected premise a wave still leans on (223, 237, 238, 251, 289) and retire
  the sixth (208) where the correction leaves no surviving premise, then move this file's member lists so no live
  seam carries a retired row. Membership and rows only: this pass re-verified no seam, composed nothing, ran no
  checker and called nothing safe (E1/E8). Full row table, bundle table, retirement mapping, evidence lookups and
  not-rowed list: [claims.md](claims.md) §Run 33.
- **S10 `new`/`unverified` (cells unchanged; members 27 → 32).** Drops its five retired members **208, 216, 223, 237,
  238** and gains **320–329** in place, ordered along the same producer → transformation → consumer chain: the route
  census pair (320, 321) at 216's position, the base-file pair (322, 323) at 223's, and the aiohttp spelling sextet
  (324, 327, 325, 328, 326, 329 — request canonicalization, request verbatim accessor, response canonicalization,
  response verbatim accessor, the fallback parser, the lookup case rule) across 237/238's positions. The
  `rel:buster_ripper/utils#_HOP_BY_HOP→rfc9110§7.6.1` subject leaves the subject list with its only member, 208.
  Three dimension phrases are re-worded to name the rows that now carry them — the peer-spelling premise becomes
  canonicalization-in-`.headers` / verbatim-only-in-`raw_headers` / backend-conditional (217 with 324–329), the
  RFC-completeness dimension becomes the base-parity authority for the eight-name target (207/219/220, with 208
  recorded as retired-and-unreplaced), and the `.items()` census dimension names the one header-name decision the
  shape does not reach (320/321). A fifth clause records which base-file requests the shared filter owns (322/323).
- **S11 `new`/`malformed` (cells unchanged; members 18 → 19).** Drops retired **232** and gains **330** and **331** in
  its place — the named client timeout classes inside `ClientError`, and the bare `asyncio.TimeoutError` the
  cumulative total timeout raises outside it. The catch-scope dimension is re-worded to name both atoms, since the
  single row it cited could not say which of the two it meant. **S11 is edited although the dispatch named S10/S12/
  S13 (+S14):** 232 is an S11 member and the delta's rule — a live seam carries only rows with current verdicts —
  cannot hold while a retired row stays listed. A membership write is not a re-verification, and `Routing: malformed`
  stands exactly as run 32 wrote it: reopening a seam's routing is the composition and decide routes' writes, not
  this pass's.
- **S12 `new`/`malformed` (cells unchanged; members 19 → 19).** Retired **256** is replaced in place by **332**, the
  visit-and-order proposition; the rewrite-mutation half of the fork is already asserted by member **258** (`HOLD`,
  no recheck trigger — `image.py` unchanged at `611df5ac…`), so no second row exists for it and none was added. The
  content-shape dimension is re-worded to cite the pair (332 read against 257/258) rather than the retired row.
- **S13 `new`/`malformed` (cells unchanged; members 20 → 21).** Retired **273** is replaced by **333** (the
  `(source, timestamp)` entry value with the block's `source` at element 0 — the settling rewording its own verdict
  names), and retired **289** by **338** and **339** (nothing clears `_image_cache` inside a test function; one case
  sends two image blocks in one call), all in place. Row **294** (`HOLD`) already states the per-test reset itself, so
  it stays the seam's only member for that fact and the two new rows carry only what it does not. Two dimension
  clauses are added: the stored value's shape (333) and the cache state a witness inherits from its own test
  (338/339).
- **S14 unchanged, and recorded as such.** None of its 29 members is retired or replaced by this amendment: **312**
  and **251** name no seam, so their successors (334/335, 336/337) name none either and join no member list; **2**,
  **48** and **313** keep the `recheck`/`unverified` cells and `B226`/`B227`/`B228` keys run 32 wrote, untouched by
  this pass. No S14 row, cell, dimension or subject was edited.
- **Cells held across all five seams.** `State`, `Routing` and `Permanent` are left as run 32 wrote them in every
  row of this table. This pass writes membership and the factual cells that describe the chain (Claim rows, Subjects,
  and the dimension wording that cites a row); it assigns no seam verdict, inherits none from the four on-disk
  `compositions/seam-S1{1,2,3,4}.md` records, and consulted the `## Composition Index` for the four moved member sets
  only — no record covers an image, header-parity, response-head or placeholder-identity boundary, so no handle
  enters or leaves `Permanent` and no composer is skipped.
- **Consequence for the queue, as ledger state rather than a decision:** twenty rows are `new`/`unverified` in twelve
  fresh bundles (B229–B240), and the four amended seams now list only rows whose verdicts are either current on file
  or freshly dispatched for this amendment. The composition pass is blocked until those bundles return; nothing here
  states whether the amended sets compose. Rows 2/48/313's three re-dispatches (B226–B228) and the
  `QB:assumption-registry/run-31-verdict-fingerprints` digest cluster (293/294/297/298/315) remain in their own lanes
  untouched.
- **S1–S9:** rows, members and cells untouched. No member of any earlier seam is retired, replaced or re-bundled by
  this amendment, and no unrelated composition was inspected for staleness.


## Run-34 changelog (2026-10-05 — scoped ledger amendment: re-rowing claims **48** and **338** after their checks)

- **Dispatch:** claims run 34's scoped amendment, mode `two cells`. Row **48** (`DOES-NOT-HOLD` / `unverified`, settled by
  [verdicts/claim-48.md](verdicts/claim-48.md)) and row **338** (`UNCLEAR` / `unverified` / `B240`, its check settled by
  [verdicts/claim-338.md](verdicts/claim-338.md) — the `malformed` finding is in its verdict and in this file's S13 seam routing,
  not in its own Routing cell, which run 33 wrote as `unverified`) are re-rowed to one proposition per row — eight live rows in fresh bundles
  **B241–B244** at `new`/`unverified` — and this file's membership moves so no live seam carries a retired row. Membership and
  rows only: this pass checked nothing, composed nothing, re-verified no seam, called nothing safe and wrote no verdict
  (E1/E8). Row table, bundle table, retirement mapping, evidence lookups and the not-rowed list: [claims.md](claims.md) §Run 34.
- **S13 `new`/`malformed` (cells unchanged; members 21 → 23).** Retired **338** leaves the list and is replaced in place by
  **345** (every statement that empties the whole `_image_cache` mapping sits in the autouse fixture, not in a test body),
  **346** (a test body does issue one entry-removing call mid-function) and **347** (what that call removes, and from which
  mapping). 338's fork was two senses of one verb — "clears" as *empties the mapping* versus as *removes entries* — and the
  senses now travel as separate rows so a verdict can disagree with one without covering the other. **339**, run 33's other
  successor to retired 289, keeps its `HOLD` / `none` / `B240` cells and its position in the list exactly as they are.
  `repo:buster_ripper.image#clear_session_cache` enters the subject list with 347; `repo:tests.test_image` stays. The
  witness-state dimension clause is re-worded to name the live rows instead of the retired one. **294** (`HOLD`, no recheck
  trigger — `tests/test_image.py` is still `sha256:b603bfd1…`) remains the seam's only member for the per-test reset itself.
- **S14 `new`/`absorbed` (cells unchanged; members 29 → 30).** Retired **48** is replaced in place by **342** (the parsed
  `--image-cache-ttl` value is assigned onto `config.IMAGE_CACHE_TTL`) and **343** (that option's declaration carries `help=`
  text presenting the value as the image-cache TTL in seconds). Those two atoms are what F-3's closure basis consumes — the
  handoff's "`IMAGE_CACHE_TTL ← --image-cache-ttl` … evidence that the control is advertised" — and they are rowed per item
  rather than inherited from the census because a verdict on a ten-name set is one verdict: a falsifier on any *other* name
  would otherwise strand the premise `DC-F-3-r1` and `DC-CE-2-r1` cite. `repo:buster_ripper.main` enters the subject list with
  343. The census rows **340**/**341** and the enumeration-route row **344** join no seam — 48's membership here was never the
  census, it was the TTL item, and run 31 recorded as much when it admitted 48 ("the `IMAGE_CACHE_TTL` CLI wiring that makes
  the control *advertised*").
- **S3 and S5: no write, and recorded as such.** `claims.md` still declares 48 against `S3, S5, S14`, but this file's S3
  members (`99, 74, 40, 41, 98, 75, 103`) and S5 members (`100, 74, 40, 43, 76, 85, 116, 117, 111, 112, 96, 101, 110`) have not
  carried 48 since run 6 retired the obsolete inventory row in favour of 99–103. A retired row cannot be dropped twice, and a
  row's own `Seams` cell is its extraction-time declaration, not live membership — run 33's precedent: retired 289 still names
  S13 in its cell while this file no longer lists it. Neither S3 nor S5 had a member, subject or dimension edited, so no live
  seam carries retired 48 after this write.
- **Cells held across both seams.** `State`, `Routing` and `Permanent` are untouched in every row of this table, S13's
  `malformed` and S14's `absorbed` included — whether an amended member set now composes, and whether either seam may leave its
  routing, is the composer's and the decide route's write (E1/E8). No `CO:` handle enters or leaves `Permanent`: the
  `## Composition Index` was consulted for the two moved member sets only, and none of its four records (runner→`on_cleanup`
  teardown, CLI-flag→outbound-call TLS, CLI-option→dump-dispatch format, serve-script→argv env) covers a numeric-CLI-wiring or
  an in-test-cache-state boundary. The four on-disk `compositions/seam-S1*.md` records are left as they stand, neither
  inherited nor discarded.
- **Consequence for the queue, as ledger state rather than a decision:** eight rows are `new`/`unverified` in four fresh
  bundles (B241–B244), one proposition this pass extracted was answered from the ledger with no checker spent (**294**,
  inherited), and both amended seams now list only rows whose verdicts are either current on file or freshly dispatched for this
  amendment. Run 32's row-48 re-dispatch (**B227**) points at wording this pass retires; the dispatch is run 32's write and this
  pass neither spends nor cancels it — see [claims.md](claims.md) §Run 34's queue note. Retired **338** is the mirror case from the
  other side: its Routing is `unverified` so it is dispatchable, but re-checking the retired wording returns the same P9 fork, and
  the correction lives in 345–347 rather than in an edit to its text; with it retired, **B240** names only settled 339
  (`HOLD`/`none`) and carries nothing dispatchable. Rows **2** and **313** (`B226`, `B228`)
  and the `QB:assumption-registry/run-31-verdict-fingerprints` digest cluster are untouched.
- **S1–S12, S15+:** rows, members and cells untouched. No member of any other seam is retired, replaced or re-bundled by this
  amendment, and no unrelated composition was inspected for staleness.

## Run-35 changelog (2026-10-05 — routing dispositions on the S10–S13 composition verdicts, T013/T014 wave second pass)

- **Dispatch:** composition-finding routing over the four records this file's cells had left open —
  `compositions/seam-S10.md`, `seam-S11.md`, `seam-S12.md`, `seam-S13.md`, each `VERDICT: COUNTEREXAMPLE` — 19 findings
  in total (12 concrete counterexamples plus the 7 missing contracts and recorded boundaries those reports name).
  Routing only: nothing was composed, re-checked, reworded in place, or written to production code; no invariant was
  created, elevated or promoted; no `CO:` record outside these four was opened (§3e's no-sweep rule).
- **Dedup before allocation was the pass's main work.** `seam-S14`'s CE-1…CE-5 and the `057` F-1…F-4/G-1/G-2 closures
  already occupy `T013`/`T014`'s steps, so **no finding here opens a task or a wave**. Where a finding is the same
  defect seen from a different seam it routes to the existing step with its own ledger key and `Selected closure:
  none added beyond …` (`DC-S10-C1-r1`→`T013` step 1, `DC-S10-C3-r1`→step 1, `DC-S11-CE1-r1`/`DC-S11-CE2-r1`/
  `DC-S11-CE3-r1`→step 2, `DC-S12-CE1-r1`→`T014` step 1, `DC-S13-CE1-r1`/`DC-S13-CE2-r1`→step 2,
  `DC-S13-CE3-r1`→step 3, `DC-S13-CE4-r1`→step 1). Where it is genuinely new it amends that step's binding
  (`DC-S10-C2-r1`'s parser-backend credential collapse, `DC-S10-C4-r1`'s census-scope correction,
  `DC-S10-LAT-r1`'s direction-pairing check, `DC-S11-D8-r1`'s disconnect attribution, `DC-S11-SGL-r1`'s third-head
  foreclosure, `DC-S12-CE2-r1`'s nested-image scope, `DC-S13-GAP-r1`'s placeholder contract).
- **S10 `absorbed`** — six findings (C1, C2, C3, C4, the latent direction swap, and the D6/D8 policy-owner-versus-
  base-parity contract). C1/C3 are F-4's own defect at this seam and route into `T013` step 1; C2, C4 and the latent
  hazard add requirements inside the same member. The named missing contract is **not** admissible as a question: the
  two axes it says conflict are each already decided by a landed binding — `062 R-A` (the helpers own the policy, the
  sites delegate) and `062 R-B` (the empty-`Bearer` rule stays route-local) — and `DC-claim-208-r1` already ranks
  base parity above a standard's enumeration for the eight names.
- **S11 `absorbed`** — CE-1/CE-2/CE-3 are `G-1`'s class at this seam, and CE-2 additionally corrects a *factual* cell
  of `DC-G-1-r2`'s case table: the post-commit non-`ClientError` raise-out does **not** put a second head on the wire
  (`web_protocol.py:748-752` raises `ConnectionError` whenever `request.writer.output_size > 0`, caught at `:601-603`
  → the turn is dropped and the connection closed). The Required column and `R-E`'s catch-all are unchanged; the
  Today column's mechanism is, and it is carried to Forward so no commit message repeats the false one. D8's
  disconnect ownership and the out-of-member SGLang third-head item route as records, not tasks.
- **S12 `absorbed`** — CE-1 is F-1's detector/writer scope mismatch (already bound by `DC-F-1-r1`/`r2` and by
  `seam-S14`'s CE-1), CE-2 is the `tool_result`-nested case F-1 already carries as **deferred additive work** with a
  named owner and trigger, and its help-text half joins the existing `QB:cli-surface` backlog marker rather than
  opening a task in `T014`'s surface. The named contract — the `--strip-images` help text's unqualified scope versus
  `tests/test_image.py:290`'s "Should only check the last user message" pin — is already ranked: `DC-F-1-r1` names
  the help text and the writer's iteration scope as the only statements of intended behavior, and `DC-F-1-r2`'s table
  inverts the pin as part of the fix. Not a question.
- **S13 `absorbed`** — CE-1 is F-2's count-derived key (`063 R-A` retires the mechanism; `seam-S14` CE-5 already
  imposed the eviction-liveness witnesses), CE-2 is F-2's instability half plus a **phantom-premise** record —
  `sort_keys`/`separators` have zero production call sites until this wave creates the first one, so the six
  `lang:python` rows 279–285 become load-bearing only *after* `T014` lands and must be re-verified at the
  accepted-local recheck rather than cited as current support —, CE-3's reclaim half is F-3's premise (already
  routed as `DC-CE-2-r1`/`DC-CE-3-r1`) while its namespace-collapse half is a new recorded boundary that no settled
  authority lets this wave move (`FK-007` option 1's deliberately absent reader; `session_id`'s system-only operand
  is accepted `T001`/`T002` behavior and `CL:buster_ripper.session#session_id/key-derivation-input`), and CE-4 is
  F-1 again, with the seam-membership gap it exposes recorded for the collector lane. The restorable-versus-marker
  contract resolves against `FK-007` option 1, so it is a docstring correction inside `T014`'s own file, not a fork.
- **Cells written by this pass:** `S10` Routing `unverified` → `absorbed`; `S11`, `S12`, `S13` Routing `malformed` →
  `absorbed`. `State` stays `new` in all four and `Permanent` stays `—`: a disposition is a Routing write, never a
  State write, and none of these records is promotable (S10 lacks fingerprints for 246/247; S11 lacks them for
  233/248; S12's 316 and S13's 267/287 carry record defects — routed to `QB:assumption-registry/run-31-verdict-fingerprints`,
  substance untouched because the verdicts still hold). **Not citable as current** in either direction: `T013`'s edits
  to `utils.py`/`routes/*.py` fire S10's and S11's anchors, and `T014`'s edits to `image.py`/`tests/test_image.py` fire
  S12's and S13's, so all four owe recomposition on the post-wave bytes at each wave's accepted-local recheck — the
  route S3/S5/S6 and `seam-S14`'s note already use.
- **Consequence for the queue:** both waves keep their membership, order, owning route, separation reasons and retry
  owners unchanged; this pass opens no wave and allocates no task id. The chain run 32 left open — `forward-collect`
  for 216/232/256/273/312, the checkers for the replacements, then composition of S10–S13 — is **closed** for these
  four seams: every one is composed and every finding is routed, `0 questions | 0 breaks | 0 malformed` among them,
  so no Fork was reserved and `decisions.md` is unchanged; `FK-011` remains the only open question in this routing.
  Residual ledger state this pass did not touch and records honestly: row **278** (`repo:buster_ripper.image`,
  `new`/`unverified`, no seam) is an unchecked premise on `T014`'s surface that blocked no composition, and retired
  rows **48**/**338** keep their `unverified` cells as dispatchable-but-superseded history. `forward-red` is *not*
  re-run: every correction landed here is narrower than a misreading and adds no referent, no artifact and no file
  outside the two blocks' declared surfaces, so 064/065's `0 AMBIGUOUS` stamps stand. **`T013`/`T014` Forward
  authorization is the orchestrator's call, not this pass's.** Gate step (v) — the `codify git create-checkpoint` over
  `assumptions/` — is still owed, as it was for 059–065.
- **S1–S9, S14–S15+:** rows, members and cells untouched. No member of any other seam moved a disposition on this
  pass's account, and `seam-S14`'s routing stands as run 32 wrote it.

## Run-36 changelog (2026-10-06 — accepted-local T013 + T014 recheck, seams **S9–S14** membership and State only)

> ⚠ **Superseded by [§Run-36 scope correction](#run-36-scope-correction-changelog-2026-10-06--ledger-scope-correction) below**, which follows this file's own numbering convention (a bounded correction pass gets its own changelog block and keeps the run handle it shares with `claims.md` — run 27's rule). S10–S14's `recheck` States and the claim-side census this block describes are void; the S10–S14 *membership* and *dimension* writes stand as history, with the rows they added now retired in `claims.md`.

- **Dispatch:** claims run 36's post-implementation recheck — **(A)** the stale-row census over every live row whose
  pinned bytes, stated subject or own `Invalidates On` names a surface the two waves moved (`image.py`,
  `routes/chat.py`, `routes/messages.py`, `routes/passthrough.py`, `utils.py`, the test tree, `REFACTORING_PLAN.md`),
  and **(B)** extraction of the new surfaces the waves created, which no pre-wave row covers. Membership, State and
  the factual cells that describe each chain only: this pass re-verified no seam, composed nothing, inherited no
  verdict from `compositions/seam-S9…S14.md`, judged nothing safe, wrote no verdict file, created no invariant and
  promoted nothing (E1/E8). Row table, per-row census reasons, bundle table, evidence lookups and the not-rowed list:
  [claims.md](claims.md) §Run 36.
- **S10–S14 `State: new` → `recheck`.** §3c fires a seam on a member change, and every one of the five has members
  whose pinned bytes moved: T013's edits to `utils.py` and the three route modules fire S10's and S11's anchors, and
  T014's rewrite of `image.py` plus the test tree fires S12's, S13's and S14's — the recomposition run 32 and run 35
  both recorded as owed "at the accepted-local recheck on the post-wave bytes". `Routing` is left exactly as run 35
  wrote it (`absorbed` in all five) and `Permanent` stays `—`: a disposition is a Routing write and this pass owns
  none; a `COUNTEREXAMPLE` or an `absorbed` routing is not inherited as a verdict either way.
- **S9 unchanged, and recorded as such.** Its State was already `recheck` from run 30's rider and both rider members
  (**67**, **73**) return in fresh bundles **B253**/**B251** against the post-wave bytes, so the cell writes land in
  [claims.md](claims.md) and no cell of this row moves. S9's member list is unchanged — no new row joins it (the
  relay-shape premises the split created are S11's chain, which already shares member 67).
- **S10 members 32 → 41** — gains **373** (`response_headers`' post-wave drop set) at the helper block's position,
  **365–368** (`_carries_token` and the multi-value selection) at the credential dimension's position, and
  **376–379** (`raw_proxy`/`proxy` and the aiohttp client's duplicate-name merge) at the witness position; the
  credential, response-side-drop and observation-channel clauses are added to its dimensions, and the witness clause
  now names 246/247 as re-checked rather than as current absence.
- **S11 members 19 → 24** — gains **369–372** (the split at `prepare()`, the committed object handed back, the
  post-commit arm's class and its non-reraising behaviour, the state mapping the timing origin now travels in) after
  the handler-shape rows, and **374** (the stand-in's fault hook) at the witness position. The chain description
  changes in kind, not in judgment: the head no longer crosses the boundary from a single function body, so a
  composer reading this seam traces producer → `_stream_single_request` → `_relay_after_commit` → writer.
- **S12 members 19 → 24** — gains **348–351** (the shared scope function both gate and writer enumerate, and
  `_user_content`'s two None conditions) between the detector and the writer, and **380** (fresh body structures) at
  the witness position; the content-shape dimension names the pair rather than a per-unit reading.
- **S13 members 23 → 29** — gains **352–356** (operand, `sort_keys`, `separators`, the `.encode("utf-8")` coercion,
  digest slice and prefix) at the key-derivation position and **381** at the witness-state position; the stored-value
  dimension now states both shapes and names 333 as re-checked, so no composer can read the pre-wave pair as current
  support. `repo:buster_ripper.image#_image_key` enters the subject list.
- **S14 members 30 → 41** — gains **357–361** (the `_now` binding, the every-stamp-through-it claim, the triple, the
  `first_seen` carry-forward, the wall-clock absence) between the writer and the expiry primitives, **362–364** (the
  sweep's position, the emptied-session drop's owner, the early return that skips it) at the sweep-position clause,
  and **381–383** at the witness-state position. `repo:buster_ripper.image#_now`, `#_cache_source`,
  `#strip_and_cache_images`, `repo:tests.image_bodies` and `repo:tests.test_image_stability` enter the subject list.
- **What the waves left behind, as ledger state.** Members 209/210/212/213/214/215/217/320/321 (S10), 241/242/243/245
  and 67 (S11), 253–262/268/316/332 (S12), 267/272/274/277/287/288/291/292/333/339/345/346/347-adjacent (S13) and
  293–308/299/300/301/315 (S14) move `HOLD`/`none` → `recheck`/`unverified` in [claims.md](claims.md) with their
  `CL:` handles retained; several of their **sentences** now describe the pre-wave tree (206, 212, 213, 214/215/217,
  253, 272, 293, 299, 303, 308, 320, 333) and are recorded as stale-by-wording so a composer cannot cite them as
  current support while their checkers are outstanding. 81 rows return in 31 fresh bundles; the 36 new rows ride 14
  more; **45 bundles ⇒ 45 checker agents**, and both ledgers now list, for every live seam, only rows whose verdicts
  are either freshly dispatched or fresh on unmoved bytes.
- **Rows deliberately left as they are.** `lang:`/`pkg:` members (78, 226–240, 279–286, 309–313, 324–331, 334/335,
  379's own library half) — no interpreter, package or documented source moved, and §2's subject typing keeps a repo
  edit from firing them; the run-35 phantom-premise note about 279–285 is recorded in [claims.md](claims.md) §Run 36
  as the observation that the wave created the first call site, which is routing-lane material, not a recheck trigger
  this pass may invent. `repo:` property rows whose named unit is byte-identical (`205`, `207`, `244`, chat's
  untouched relay/request/transform units, `347`, `278` — the last settled on the *current* `image.py` bytes at
  `d70c876`). Retired rows keep their terminal cells. Rows **2** and **313** (`B226`, `B228`) are not stale by any of
  the five tests this pass applied; run 32's queue entries are neither spent nor cancelled here, and **B227**/**B240**
  continue to point at retired wording as runs 33/34 recorded.
- **Consequence for the queue, as ledger state rather than a decision:** nothing in this file states whether S9–S14
  compose, whether any seam may leave `absorbed`, or whether a re-dispatched row's proposition survives; the
  composition pass runs after B245–B289 return, and the decide lane owns dispositions. Gate step (v) — the
  `codify git create-checkpoint` over `assumptions/` — remains owed for this pass as it was for 059–065 and runs
  30–35.

## Run-36 scope correction changelog (2026-10-06 — ledger scope correction, seams **S10–S14** State cells only)

- **Authority, cited before any cell.** The reloaded `run-codify` assumption gate says *"**Third-party
  subjects only.** The gate covers libraries, runtimes and external services. Repo code is not a subject, and
  a plan change that touches no third-party subject gives this gate nothing to run"*, and its §3c twin says to
  *"Record a seam only where a library, runtime or external service is its producer or consumer"*.
  `forward-collect`'s depth limits say the matching rule for what is already on file: *"`repo:` rows and
  repo-only seams already in a ledger are history: do not recheck or recompose them."* The user has confirmed
  this reading for this session. Run 36's dispatch — recheck the repo census, then recompose the seams whose
  repo members moved — was the defect, and the `recheck` it wrote on S10–S14 is void.
- **What this pass writes here.** Five `State` cells, one heading pointer, and this block. No seam was
  composed, re-verified, judged safe or unsafe, or given a member change; no `Routing` cell moved; no
  `Permanent` cell moved; no composition file was opened, written or invalidated; no invariant was created or
  elevated; and no row of `claims.md` was extracted, reworded or checked from this file (E1/E8). The cell-level
  claim writes and their per-row bases are in [claims.md](claims.md) §Run 36 scope correction; hashes before
  the first write of this pass: `claims.md` `sha256:98677d496db506b099b9a104d61becdec2642f9e785aee030eba9eb99922de4b`,
  `seams.md` `sha256:5d0c6922c111c109c316603d2749d2c059f2297f66ed57806de628372d4700de` (run 36's committed
  output at `b01cf5c`), against the pre-run-36 baseline `4c0f4eb` — `seams.md`
  `sha256:abc66a586d57d088a1456686648a88a60e47f98fecbeb03a07427929b6bd6e83` — read out of Git.
- **Why `out-of-scope` rather than a reverted `new`.** Run 35 left S10–S14 at `State: new` with `Routing:
  absorbed`: composed, all five `COUNTEREXAMPLE`, nineteen findings routed and every one of them implemented
  and accepted through T013 and T014. Reverting to `new` would put them back in step (iv)'s dispatch set
  (`new` or `recheck` ⇒ one composer each) and re-open verdicts that are already closed; leaving `recheck`
  claims a recomposition this gate has no subject authority to service. `out-of-scope` is the terminal cell
  that states which of the two it is: **the gate is not servicing these seams, and nothing may be inferred
  about whether they compose.**
- **The five cells, with the per-seam basis.** The third-party member sets are what the authority test turns
  on, so they are named rather than asserted; none of them moved, and none was fired by run 36 either.

  | Seam row | Members | Third-party members and their cells | What run 36's `recheck` was actually standing on | Corrected |
  |---|---|---|---|---|
  | `S10` row | 41 | `pkg:aiohttp@3.13.3` — 234, 235, 236, 324, 325, 326, 327, 328, 329 (`HOLD`), plus **379** (new, in scope) | 8 retired repo rows it added (365–368, 373, 376–378) and 9 repo members superseded by T013 (206, 209, 210, 212, 214, 217, 246, 247, 320) — every one a repo-code movement | `State: out-of-scope`, `Routing: absorbed` (run 35's), `Permanent: —` |
  | `S11` row | 24 | `lang:python` 78; `pkg:aiohttp` 226–231, 233, 239, 240, 330, 331 — all `HOLD` | 5 retired rows added (369–372, 374) and 3 superseded repo members (241, 245, 248) from T013's relay split | same |
  | `S12` row | 24 | **none — a repo-only seam on the gate's own definition** | 5 retired rows added (348–351, 380) and 3 superseded repo members (253, 290, 316) from T014's scope fix | same |
  | `S13` row | 29 | `lang:python` 279–285, 309 — all `HOLD` | 6 retired rows added (352–356, 381) and 4 superseded repo members (272, 288, 333, 345) from T014's content-addressed key | same |
  | `S14` row | 41 | `pkg:aiohttp` 230, 310, 311, 313 — all `HOLD` | 11 retired rows added (357–364, 381–383) and 5 superseded repo members (293, 299, 300, 303, 308) from T014's stamp and sweep work | same |

- **The `absorbed` findings are untouched, on purpose.** Nineteen findings came out of the S10–S14
  compositions and every one is routed — 12 concrete counterexamples plus the 7 missing contracts and recorded
  boundaries named in `compositions/seam-S10…S14.md` — into `T013` steps 1–2 and `T014` steps 1–3 as
  `DC-S10-C1…C4-r1`, `DC-S10-LAT-r1`, `DC-S11-CE1…CE3-r1`, `DC-S11-D8-r1`, `DC-S11-SGL-r1`, `DC-S12-CE1…CE2-r1`,
  `DC-S13-CE1…CE4-r1`, `DC-S13-GAP-r1`, with the run-35 closure ledger entries and both waves accepted through
  `a5ef525` and `4c0f4eb`. A scope correction is not a reason to re-open a routed finding, so `Routing:
  absorbed` stands in all five rows and this pass created, cancelled and re-allocated nothing. The five
  composition records stay on file as the historical verdicts that produced them; they are neither inherited
  nor disputed here.
- **Residue left on file, recorded so it is not mistaken for a live member set.** This pass wrote `State`
  cells only, so the S10–S14 member lists and dimension clauses still carry run 36's edits: the member lists
  name rows 348–383 (retired in [claims.md](claims.md) §3) and the dimensions name the superseded repo rows
  (206, 209, 210, 212, 214, 217, 241, 245–248, 253, 272, 288, 290, 293, 299, 300, 303, 308, 316, 320, 333, 345
  — all twenty-four wave-superseded rows are S10–S14 members; 213 and 215, which run 36 also flagged as
  stale-by-wording, are back at `HOLD` on the blind verdicts that landed, and 53, 54 and 65 sit in other
  seams) and quote
  post-wave atoms that no live row states (348–351, 352–356, 357–361, 362–364, 365–368, 369–372, 373–378,
  380–383). Deleting them would erase the audit trail this correction exists to leave, and editing the five
  rows down to their run-35 cells would be a membership write this dispatch does not authorize. The terminal
  `State` is what stands between that residue and a composer: no row here is `new` or `recheck`, so nothing is
  dispatched at it, and any later pass reading a dimension clause must read the `State` cell and the retired
  cell in `claims.md` before treating a member as current.
- **Consequence for step (iv), as ledger state rather than a decision.** No composer runs on S10–S14: their
  `State` is terminal and not `new`/`recheck`, so the coverage check ("every `new` or `recheck` seam has one
  composition file") is satisfied without a new file, and no `CO:` handle enters `Permanent`. The one live
  third-party premise this gate still owes is **379** on `B287`: its atomic check proceeds normally, and if it
  returns non-`HOLD` it routes through `assumption-decide` as an atomic finding — not as a recomposition of
  S10, which this gate is not servicing.
- **What belongs to whom now.** Whether T013's header parity, `_prepare_headers`' credential selection and the
  post-commit relay abandon conform, and whether T014's strip scope, content-addressed key, monotonic stamps,
  triple shape and tail-sweep lifecycle conform, are conformance questions about **repo code**. Under the
  authority above they belong to **Yield Phase and the external review** — the phase-scoped QA gate tests,
  `tests/test_header_parity.py`, `tests/test_image_stability.py`, `tests/test_stream_relay_commit.py`,
  `tests/test_routes_integration.py`, the traceability and parity passes, and the review in front of the user —
  and **not** to this gate. The gate's honest remaining scope after this correction is one library premise
  (379) and nothing else.
- **S9, recorded and not written.** Its `State: recheck` is run 30's cell, not run 36's; this pass neither
  fired nor reverted it, and its repo members 67 and 73 simply return to `HOLD`/`none` on the claims side. The
  same authority reading would make S9 terminal too — none of its `pkg:aiohttp` members moved either — but
  touching a run-30 write is outside this correction's dispatch, so it is flagged for the orchestrator rather
  than fixed. Same for `S1`–`S5`, `S7`, `S8`, whose `recheck` cells are older passes' and whose member rows
  this pass did not touch.
  *(Flag acted on: the orchestrator's follow-on dispatch wrote that one cell on 2026-10-06 — see
  [§S9 scope correction addendum](#s9-scope-correction-addendum-2026-10-06--ledger-scope-correction-seam-s9-state-cell-only)
  below. Nothing else in this bullet is changed: `S1`–`S5`, `S7`, `S8` stay older passes' cells, still flagged
  and still unfixed.)* *(Second flag acted on, same date: the follow-on dispatch this addendum deferred to has
  now terminalized all seven, with the member-row evidence each cell needed — see
  [§Run-36 scope correction addendum 2](#run-36-scope-correction-addendum-2-2026-10-06--ledger-scope-correction-seams-s1s5-s7-s8-state-cells-only).
  The quoted sentence above stands as what was true when run 36 wrote it.)*
- **Note on vocabulary, recorded rather than fixed:** `out-of-scope` is not in `forward-collect`'s State list
  for seams (which has `new`, `recheck`, `inherited: S<n>`, `registry-safe`, `registry-counterexample`). It is
  used here because the alternative tokens all assert something this pass may not assert — that a composition
  happened, or that the chain is unchanged — and it is named in [claims.md](claims.md) §Run 36 scope correction
  §6(a) as the thing to amend if the disposition becomes permanent.

## S9 scope correction addendum (2026-10-06 — ledger scope correction, seam **S9** State cell only)

- **Authority — identical to run 36's, cited before the cell.** The reloaded `run-codify` assumption gate is
  third-party-subjects-only: *"**Third-party subjects only.** The gate covers libraries, runtimes and external
  services. Repo code is not a subject, and a plan change that touches no third-party subject gives this gate
  nothing to run"*, with its §3c twin: *"Record a seam only where a library, runtime or external service is its
  producer or consumer"*. `forward-collect`'s depth limits hold the matching rule for what is already on file:
  *"`repo:` rows and repo-only seams already in a ledger are history: do not recheck or recompose them."* This
  addendum applies that same authority to the one row run 36 flagged and deliberately left alone, on the
  orchestrator's explicit dispatch. The authority basis is not new, extended, or re-argued here.
- **Why this cell was outside run 36 and is not outside this pass.** Run 36's dispatch was bounded to the rows
  its own recheck had fired; S9's `State: recheck` is run 30's cell, so writing it would have been a
  run-30 overwrite inside a run-36 scope. This pass exists to make exactly that one write, and nothing else.
- **What this pass writes here.** One `State` cell in the `S9` row, the run-36 bullet's pointer, and this block.
  `Routing` stays `unverified` (S9's is run 30's — unlike S10–S14 it was never `absorbed`, and this pass neither
  routed nor absorbed anything); `Permanent` stays `—`; the member list and every dimension clause are
  untouched; no seam was composed, re-verified, judged safe or unsafe, or given a member change; no composition
  file was opened, written or invalidated (`compositions/seam-S9.md` stands as the historical verdict it is);
  no invariant was created or elevated; and no row of `claims.md` was extracted, reworded or checked from this
  file (E1/E8).
- **The cell, with the per-seam basis.**

  | Seam row | Members | Third-party members and their cells | What run 30's `recheck` was standing on | Corrected |
  |---|---|---|---|---|
  | `S9` row | 18 | `pkg:aiohttp@3.13.3` — 64, 83, 84 (`B113`), 154, 156 (`B121`), 155 (`B122`) — every one `HOLD`, none fired by run 30, 35 or 36 and none moved since | its two repo members, 67 (`handle_streaming`'s relay split into `_relay_after_commit`) and 73 (`handle_messages` edited) — repo-code movement only, both rows back at `HOLD`/`none` | `State: out-of-scope`, `Routing: unverified` (run 30's, unchanged), `Permanent: —` |

- **Same terminal meaning.** `out-of-scope` here states only that **the gate is not servicing this seam, and
  nothing may be inferred about whether it composes** — not that a composition happened, not that the chain is
  unchanged. It carries the same vocabulary caveat run 36 records for S10–S14: the token is not in
  `forward-collect`'s seam State list and is named in [claims.md](claims.md) §Run 36 scope correction §6(a) as
  the thing to amend if the disposition becomes permanent.
- **Consequence for step (iv).** No composer runs on S9: its `State` is terminal and not `new`/`recheck`, so the
  coverage check is satisfied without a new file and no `CO:` handle enters `Permanent`. Whether T008/T008.001's
  stream dispatch and raw-error relay conform is a conformance question about **repo code** and belongs to
  Yield Phase and the external review, not to this gate — the same ownership run 36 records for S10–S14.
- **Census, honestly recorded.** This closes the rows run 36's authority covers: `S9`–`S14` are terminal, and
  no row in this ledger is at `State: new`. It does **not** empty step (iv)'s dispatch set. `S1`, `S2`, `S3`,
  `S4`, `S5`, `S7` and `S8` are still at `recheck` — older passes' cells, on rows this pass and run 36 both
  left untouched — so seven seams remain dispatch-eligible by their own state. `S6` is `registry-safe`. Any
  further terminalization of `S1`–`S5`, `S7`, `S8` needs its own dispatch and its own member-row evidence;
  nothing here authorizes it.
  *(Flag acted on: the orchestrator's follow-on dispatch terminalized all seven of those cells on 2026-10-06,
  with the member-row evidence this bullet asked for — see
  [§Run-36 scope correction addendum 2](#run-36-scope-correction-addendum-2-2026-10-06--ledger-scope-correction-seams-s1s5-s7-s8-state-cells-only)
  below. Nothing else in this bullet is changed, and its attribution "older passes' cells" is corrected there
  rather than here: those seven cells are runs 11/14/15/20/21/22/23's, not run 30's.)*
- **Hashes.** Before the first write of this pass: `claims.md`
  `sha256:b78459c7d0472e704e8db6c70f3cad8ccc647cd4d218b4ae8062cd494e26a411`, `seams.md`
  `sha256:28a1b9115d1575831c4b81b5034530730dd872fcb1aabe477a2031abef4fd70c`. `claims.md` is not modified by
  this pass.

## Run-36 scope correction addendum 2 (2026-10-06 — ledger scope correction, seams **S1–S5**, **S7**, **S8** State cells only)

- **Authority — identical to run 36's and the S9 addendum's, cited before the cells.** The reloaded `run-codify`
  assumption gate is third-party-subjects-only: *"**Third-party subjects only.** The gate covers libraries,
  runtimes and external services. Repo code is not a subject, and a plan change that touches no third-party
  subject gives this gate nothing to run"*, with its §3c twin: *"Record a seam only where a library, runtime or
  external service is its producer or consumer"*. `forward-collect`'s depth limits hold the matching rule for
  what is already on file: *"`repo:` rows and repo-only seams already in a ledger are history: do not recheck or
  recompose them"*, and *"Check once, recheck only for a listed reason … Nothing else marks a row `recheck`: not
  a move elsewhere in the file, a re-base, a new phase, a new task relying on it, or doubt."* This addendum
  applies that same authority to the seven rows the S9 addendum's census left at `recheck`, on the
  orchestrator's explicit dispatch to complete that work under the identical authority. The basis is not new,
  extended, or re-argued here.
- **The test applied to each row before its cell was written.** A `recheck` cell survives this pass only if it
  stands on a **third-party** ground — a `pkg:`/`lang:`/`rel:` member that is itself unfired-and-current *and*
  whose subject moved, or a real dependency or runtime change. It terminalizes if it stands only on repo bytes.
  Two reads decide it, both performed rather than assumed: the member-row census below (every third-party
  member of all seven rows, read out of `claims.md`'s live table) and the dependency read below.
- **Result: seven repo-grounded cells, zero third-party triggers.** In all seven rows every `pkg:`/`lang:`/`rel:`
  member is at `State: HOLD` / `Routing: none` on its recorded bundle, with no outstanding check, and the
  repo-kind members are likewise all `HOLD`/`none` — `claims.md` §Run 36 scope correction §5 records that after
  that pass **zero** rows in the ledger are at `recheck` and zero at `new` other than row **379**. What each of
  these seven cells was standing on is in the table's fourth column, and every entry in it is a repo event:
  accepted-local landings (T003–T010) and their member re-dispatches, plus the three rows (`13`, `26`,
  `129`/`130`/`131`) whose `messages.py`/`chat.py`/`passthrough.py` movement run 36's own correction reverted
  as out-of-scope repo triggers.
- **No real dependency change, verified against the tree.** Installed now: aiohttp **3.13.3**, click **8.3.1**,
  CPython **3.13.8**, and transitive `multidict` **6.7.1** (`uv.lock` and `.venv` agree) — byte-for-byte the set
  `claims.md` §Run basis records as read from `.venv` at run 1, and the set every `pkg:` subject string in this
  ledger names. A grep over both ledgers finds no upgrade, re-pin, or dependency-change statement in either
  direction, and §4 of the run-36 correction independently records the parsed result that run 36's 81-row census
  was `repo:` ×80 and `rel:` ×1 with **no** `pkg:`/`lang:` row fired. So this pass silenced no live third-party
  trigger: each of these seams' library and runtime premises remain verified and on file exactly as they were,
  and none of them is what the `recheck` cell was waiting on.
- **Correction to the premise this dispatch was framed on, recorded rather than absorbed.** The seven cells are
  not run 30's. Run 30 wrote no cell in any of these seven rows — its own §"S1–S8" bullet reads "rows, members
  and cells untouched" — and left only an *observation* (that `verdicts/claim-13/-15/-17/-46/-53/-54/-65.md`
  carry an expired `messages.py` fingerprint) "left to the owning route's next S1/S2 pass". The live cells are
  runs **11** (S5), **14** (S7), **15** (S8), **20** with **21** holding it (S1, S2), **22** (S4) and **3** with
  **23** re-basing it (S3, S5). The outcome is unaffected — every one of those passes was dispatched on repo
  bytes — but the attribution matters to the audit trail, and the S9 addendum's "older passes' cells" is the
  accurate phrase.
- **The seven cells, with the per-seam basis.** The third-party member sets are what the authority test turns
  on, so they are named rather than asserted; none of them moved, none was fired, and none is outstanding.

  | Seam row | Members | Third-party members and their cells | What the `recheck` was actually standing on | Corrected |
  |---|---|---|---|---|
  | `S1` row | 35 | `lang:python` 44, 111, 112, 168 and `rel:tests.test_dump_write_count#COLLAPSED_FORMATS→buster_ripper.config#DUMP_FORMATS` 171 — all `HOLD`/`none` (B45, B79, B89, B130, B142) | run 20's accepted-local **T004** recheck — test-tree rows 165–167/169 added and 21 exercised members re-dispatched in B126–B140 over `dump.py`/`session.py`/`config.py`/`routes/messages.py` repo bytes — held by run 21 after it replaced malformed 167 with 170–173; member **13** moved with T013 and was reverted in §1 | `State: out-of-scope`, `Routing: unverified` (unchanged), `Permanent: —` |
  | `S2` row | 20 | `pkg:aiohttp@3.13.3` 35, 36, 123, 124 and `lang:python` 113, 114, 125, 126 — all `HOLD`/`none` (B42, B94, B90, B95) | runs 8/12/13/20 — the **realized teardown** in `server.py` (`on_cleanup`'s cancel-then-flush order, `_final_dump_flush`) and members 120–122 reopened for T004's per-format flush; run 20 re-verified the repo files (`main.py cbe2802d…`, `server.py 6a3e45ef…`, `dump.py 07b0b076…`) and no library among them | `State: out-of-scope`, `Routing: absorbed` (run 35's disposition, unchanged), `Permanent: —` |
  | `S3` row | 7 | `pkg:click@8.3.1` 40, 41 — both `HOLD`/`none` (B145, B23) | run 3's realized-chain recheck once T001–T003 landed the format dispatch, re-based by run 23 onto T006's `main.py` bytes (`cbe2802d…`→`fd36dc12…`) — a pass that itself records "`pkg:`/`lang:` identities unchanged" | `State: out-of-scope`, `Routing: unverified` (unchanged), `Permanent: —` |
  | `S4` row | 13 | `pkg:aiohttp@3.13.3` 37, 39, 50 and `pkg:click@8.3.1` 40 — all `HOLD`/`none` (B144, B145) | run 22's accepted-local **T006** — the landed flag → callback → `config.SKIP_TLS_VERIFY` → `TCPConnector(ssl=…)` chain in `main.py`/`config.py`/`server.py`, absence row 24 dropped as consumed-by-design, seven exercised members re-dispatched in B143–B148; member **26** moved with T013's route edits and was reverted in §1 | `State: out-of-scope`, `Routing: unverified` (unchanged), `Permanent: CO:cli-flag-to-outbound-call/tls-posture-crossing` (left exactly as found — see the bullet below) |
  | `S5` row | 13 | `pkg:click@8.3.1` 40 and `lang:python` 43, 111, 112 — all `HOLD`/`none` (B145, B25, B79, B89) | runs 6/7/8/**11**/14 — **T009**'s UTF-8 byte accounting in `_append_record`, the trigger-read rechecks (85 against changed source, 109 → 116–117), and run 23's re-base on the post-T006 `main.py`; §Run-11 records the reason as "the changed `_append_record` source" | `State: out-of-scope`, `Routing: unverified` (unchanged), `Permanent: —` |
  | `S7` row | 18 | `pkg:aiohttp@3.13.3` 51, 137 and `lang:python` 68–72 — all `HOLD`/`none` (B97, B98, B99, B106) | run 14's accepted-local **T003.001** — pre-remediation route pins 53–56 replaced by the current parse-site census 129 and per-route mappings 130–136/138–140 over `routes/*.py`; members **129/130/131** moved with T013 and were reverted in §1 | `State: out-of-scope`, `Routing: absorbed` (run 30's disposition, unchanged), `Permanent: —` |
  | `S8` row | 12 | `pkg:aiohttp@3.13.3` 57 and `lang:python` 78 — both `HOLD`/`none` (B107, B110) | run 15's accepted-local **T007** — the request-local timeout landing in `backend.py`/`server.py`, row 59 → 141 and the three-site census 61 → 144, with 57–58 and 77–82 re-dispatched against repo source | `State: out-of-scope`, `Routing: unverified` (unchanged), `Permanent: —` |

- **Same terminal meaning.** `out-of-scope` states only that **the gate is not servicing these seams, and
  nothing may be inferred about whether they compose** — not that a composition happened, not that the chain is
  unchanged, and not that the library premises are wrong. It carries the same vocabulary caveat runs 36 and the
  S9 addendum record: the token is not in `forward-collect`'s seam State list and is named in
  [claims.md](claims.md) §Run 36 scope correction §6(a) as the thing to amend if the disposition becomes
  permanent.
- **Cells deliberately left alone.** `Routing` in all seven (`unverified` in five, `absorbed` in S2 and S7 — a
  disposition is a Routing write and this pass owns none, and an `absorbed` routing is neither inherited as a
  verdict nor reopened by a scope correction); `Permanent` in all seven, **including S4's
  `CO:cli-flag-to-outbound-call/tls-posture-crossing`**, which stays as found and is *not* made citable-as-current
  by this pass — a retained handle under a terminal State is retained history, the §3c condition run-26 recorded
  for S6 before run 29 republished that seam; and the member lists, subject lists, boundary cells and every
  dimension clause in all seven rows. No seam was composed, re-verified, judged safe or unsafe, or given a member
  change; no composition file was opened, written or invalidated (the on-disk `compositions/seam-S1…S8.md`
  records stand as the historical verdicts they are); no verdict token was copied into a State cell; no invariant
  was created or elevated; nothing was promoted; and no row of `claims.md` was extracted, reworded or checked
  from this file (E1/E8).
- **Residue, recorded so it is not mistaken for a live dispatch set.** This pass wrote `State` cells only, so
  these seven rows keep their member lists and dimension clauses, which name repo rows whose verdict records pin
  pre-wave bytes (run 30's observation about `verdicts/claim-13/-15/-17/-46/-53/-54/-65.md`, and run 36's §1
  statement that a retained `CL:` handle "is not a claim of currentness"). That is a **registry** fact for the
  promotion lane, not a dispatch reason for this gate, and no row here is `new` or `recheck`, so nothing is
  dispatched at any of them.
- **Consequence for step (iv).** No composer runs on any row of this ledger: `S1`–`S5`, `S7`–`S14` are terminal
  and `S6` is `registry-safe` and citable, so the coverage check — every `new` or `recheck` seam has one
  composition file — is satisfied with no new file, and no `CO:` handle enters or leaves `Permanent`. The one
  third-party premise the gate still had in flight, row **379** on **B287**, has since returned
  `VERDICT: HOLDS`; the ledger's cell is `HOLD`/`none`, and its `PROMOTION: INELIGIBLE` reason (the deciding
  semantics are `multidict`'s, which was not among the source roots the check was given) belongs to the
  promotion lane, not to a recomposition of S10. Whether the T013/T014 implementation and the earlier accepted
  chains **conform** is a conformance question about repo code and belongs to Yield Phase and the external
  review, under the same ownership run 36 records.
- **Census, honestly recorded.** Terminalized by this pass: **7** cells (`S1`, `S2`, `S3`, `S4`, `S5`, `S7`,
  `S8`). Seams remaining at `new` or `recheck`: **none**. Seams at `out-of-scope`: **13** (`S1`–`S5`, `S7`–`S14`).
  `S6` alone is `registry-safe`. Terminalized across the whole correction family: run 36 wrote five cells
  (`S10`–`S14`), the S9 addendum one, this pass seven — **13** rows, with `S6` never in scope because its
  recomposition landed and was published. `claims.md` needs no cell write from this pass and got none.
- **Hashes.** Before the first write of this pass: `claims.md`
  `sha256:b78459c7d0472e704e8db6c70f3cad8ccc647cd4d218b4ae8062cd494e26a411` (byte-identical to the S9
  addendum's pre-write hash, which is how this pass verified that `claims.md` has not moved since), `seams.md`
  `sha256:29364abbe7620b02da62424815e3909568577f360f6e4dd771de449301204af5`. After the seven `State` cells and
  before this block: `seams.md` `sha256:1ba6a56f4d9857c9454aee2b91ffda8b142872db339e2e237534d3cadc563f38` —
  reverting exactly those seven cells to `recheck` reproduces `29364abb…` byte for byte, which is this pass's
  proof that no other cell in those rows moved.

## Run-37 changelog (2026-10-06 — third-party premise extraction for `QB:utils/response-header-relay-fidelity`, seam **S15** only)

- **Dispatch:** claims run 37, the phase-boundary assumption recheck on the CR2 wave accepted at Checkpoint 168 on
  `c8df363`, restricted to third-party subjects. The ledger entered this pass with **zero** rows in `new` or
  `recheck` and `S1`–`S14` all terminal, so the eligible set was one fresh premise family, not a census. Pre-write
  hashes: `claims.md` `sha256:b78459c7d0472e704e8db6c70f3cad8ccc647cd4d218b4ae8062cd494e26a411`, `seams.md`
  `sha256:3e745ee47b7eaa15e02dc89f189502361ccbada14f0528ae2638cd8a0fc7f248` (both re-read after writing).
- **S15 `new`** — the fresh seam over claims run 37's rows 384–389. It is recorded at the **library endpoint**:
  producer `pkg:aiohttp@3.13.3`'s `ClientResponse`, which hands the caller a decoded payload and a header set in the
  same object; consumer `repo:buster_ripper.utils#response_headers`, named as a position only, with no claim row
  written about it. Boundary value and dimensions are the trace of what that one object exposes, not a scenario: the
  `auto_decompress` session default and its per-request fallback (384/385), decoded bytes versus encoded wire bytes
  (386), `Content-Encoding` relayed alongside a body the client already decoded (387), single versus repeated header
  names and what `.items()` yields per occurrence (388), and caller-supplied versus client-default
  `Accept-Encoding` (389).
- **S1–S14: untouched, and re-stated as terminal.** No member list, State, Routing, Permanent or dimension cell of any
  of the fourteen moved. `S10`'s header-relay chain is where the *repo* boundary already lives and this pass did not
  reopen it; `S15` exists because no terminal seam has the library endpoint as its boundary value — the fourteen are
  repo-anchored chains whose subjects happen to include `pkg:aiohttp@3.13.3`, whereas this one is producer-side at the
  library and is therefore the only seam in this file whose Subjects list has a `repo:` entry in a consumer position
  only. The composer that reads `S15` receives claims 384–389 alone.
- **No seam re-verified, composed or judged** (E8): no `compositions/seam-S15.md` was read or written, no verdict
  inherited, no `CO:` handle allocated, nothing promoted. `## Composition Index` was consulted for `S15`'s exact
  ordered member set, boundary and dimensions and holds no matching record, so `Permanent` stays `—` and no composer
  is skipped.
- **Recheck delta on this file's seams: none fired.** The four listed reasons were tested against every `pkg:` and
  `lang:` claim row and its cited source (see [claims.md](claims.md) §Run 37, "Why the restore set is empty"): the
  four installed files carrying `pkg:` verdict fingerprints are byte-identical to those fingerprints, aiohttp is
  still 3.13.3 on Python 3.13.8, the only live `rel:` row is repo at both ends, and the wave's new observation created
  an uncovered premise rather than contradicting a `HOLDS`. `S9`'s first-party re-verification of member 84 at
  Checkpoint 168 stands; this pass moved no cell of that row.

## Run-38 changelog (2026-10-06 — composition routing for seam **S15** only; `assumption-decide`)

- **Dispatch:** routing of `compositions/seam-S15.md` (`VERDICT: COUNTEREXAMPLE`, `PROMOTION: ELIGIBLE`,
  fingerprint `sha256:74ac4658…`), whose six members are run 37's rows 384–389. Mode:
  composition-finding routing at the MS-001 boundary. Report:
  [../iterations/074-assumption-decide.md](../iterations/074-assumption-decide.md).
- **S15 `Routing` `unverified` → `absorbed`.** `State` (`new`), `Members`, `Subjects`, the boundary value, every
  dimension cell and the `Permanent` handle `CO:client-response-to-relay-consumer/payload-and-header-pairing`
  are untouched. The seam is **not closed**: `T017` must land and `S15` be recomposed once, and
  `resolved` then needs that composition to return `SAFE` or every remaining finding accepted. This is `S15`'s
  **first** composition, so the one-recomposition allowance is earned by the repair, not consumed by it.
- **Two findings routed, in opposite directions, on a base comparison this pass ran rather than assumed**
  (`../evidence/s15-reach-and-base-comparison.txt`):
  - **CE-1** — `Reach: latent`, base parity **confirmed** (the base's httpx decoded `.content` too and
    `buster_ripper_legacy.py:630-637` kept the name) ⇒ disposition **`accepted`**; no task, no wave, no Fork,
    routed to `roadmap.md` §RM-008 with its re-open trigger and listed in `## Elevated to user`. Members
    **384, 385, 386, 387, 389** → Routing `accepted-consequence`.
  - **CE-2** — `Reach: reached` (the deployed upstream repeats `date`/`server` on every response and the
    running proxy relays one line where two arrived), and base parity **refuted**: `httpx.Headers.items()`
    concatenates occurrences, so the base delivered every value while the branch keeps the last ⇒
    **branch-introduced divergence cemented by the header-parity work**, disposition **`absorbed`**, task
    **`T017`** (wave anchor, local Yield, prework of the MS-001 close — lands before `Yield Red`). Member
    **388** → Routing `absorbed`. Ledger `DC-S15-CE-2-r1`.
  - All six member `State` cells remain `HOLD`; no verdict record was rewritten.
- **Notes carried, nothing opened:** N1 (`br`/`zstd` recognized but not installed → the rejection lands inside
  the existing `except aiohttp.ClientError` arms) stands as a note with its own installation trigger; N2
  (egress direction) is now *evidenced* base parity — Starlette yields per-occurrence pairs into the base's
  dict comprehension, so the egress fold is unchanged in kind, and it remains S10's boundary, not this one.
- **`S1`–`S14`: untouched, terminal, and re-stated as such.** No member list, State, Routing, Permanent or
  dimension cell of the fourteen moved; no repo row was re-extracted (the gate stays third-party-subjects-only),
  no seam was recomposed, no composition re-run, and `S15`'s composition verdict was neither re-derived nor
  widened. `T017`'s scope declaration keeps `forward_headers` and the proxy-authored header literals out of
  reach, so no S10/S11 cell can be read as reopened by it.
- **No registry write by this pass.** The published `CL:`/`CO:` records were consumed as evidence inputs only
  and classified nothing; `invariants/` bytes are unchanged by this routing (their prior state carries run 37's
  promotion), and no invariant was created, edited or promoted.

## Run-39 changelog (2026-10-06 — recomposition routing for seam **S15** only; `assumption-decide`, composition-verdict disposition)

- **Dispatch:** routing of the recomposed `compositions/seam-S15.md` (`VERDICT: COUNTEREXAMPLE`,
  `PROMOTION: ELIGIBLE`, basis `sha256:d790d12b…`, checked `2026-10-06T20:33:47Z`) — the single post-repair
  recomposition run 38 earned, run on the post-`T017` bytes after `Invalidates On` 3 fired. Mode: narrow
  confirmation on one seam — the question put to this pass was *what, if anything, is new since
  [074-assumption-decide](../iterations/074-assumption-decide.md)*, not a re-opening of what 074 ranked.
  Report: [../iterations/076-assumption-decide.md](../iterations/076-assumption-decide.md).
- **S15 `State` `new` → `registry-counterexample`, `Routing` `absorbed` → `accepted-consequence`.** `Members`,
  `Subjects`, the boundary value, every dimension cell and the `Permanent` handle
  `CO:client-response-to-relay-consumer/payload-and-header-pairing` are untouched, and no `claims.md` cell moved.
  `State` moves because §3c's condition is now met in the negative direction: the `## Composition Index` row for
  this exact ordered member set carries verdict `COUNTEREXAMPLE` at basis `d790d12b…`, and the recomposition
  re-verified every digest in that basis against the working tree (this pass re-read
  `buster_ripper/utils.py` = `9abaa6b8…` and found it matches, with `git status --porcelain -- buster_ripper tests`
  empty), so the record is **fresh** and no composer runs. That index row is carried in the working tree as an
  uncommitted `assumption-promote` write (two `invariants/` paths, the same state run 38 disclosed); this pass
  neither commits nor edits it, and the cell's claim is about the record's content and freshness, not about its
  commit state. `out-of-scope` would be wrong — that cell states the
  gate is not servicing the seam and nothing may be inferred about whether it composes, whereas this seam *was*
  serviced and its verdict is known. `registry-safe` would be a false claim: the verdict is not `SAFE`.
  ⚠ **The orchestrator's `registry-counterexample → dispatch assumption-decide` branch is answered by the
  Routing cell, not re-serviceable by it:** this cell names the verdict on file, and `accepted-consequence` says
  there is nothing left to route. A re-dispatch on S15 without a fired trigger is capped by rule (below).
- **What is new since 074, and what only looks new.**
  - **CE-1 — not new.** Carried as still composable, re-traced on the moved consumer bytes; its chain, its
    reach shape and its three candidate remedies are the ones 074 ranked `accepted` → `roadmap.md` §RM-008 with
    its re-open triggers. This pass re-litigates none of it. Rows 384/385/386/387/389 keep
    `accepted-consequence`.
  - **CE-2 — closed on the composer's own trace, and this is lifecycle, not a correction.** The recomposition
    shows the fold gone (`response_headers` → `CIMultiDict` → clone → one field-line per entry) and the producer
    guarantee and the consumption now agree. That is the terminal condition `DC-S15-CE-2-r1`'s `Status:` line
    named, so **no `-r2` row is created** (precedent: run-29 on `DC-seam-S6-value-word-split-r1`). The `resolved`
    stamp itself stays **Yield Red's**, exactly as that `Status:` line assigned it; row 388's Routing stays
    `absorbed` because the column records disposition, never lifecycle.
  - **N5 — genuinely new, and routed.** The repair's own consequence: duplicated *single-valued* header names now
    reach the wire verbatim as several field-lines. Reach: **`reached`**, established on the running production
    proxy against the operator-declared upstream — `evidence/t017-live-relay-transcript.txt` §2 FIXED run counts
    `date: 2` / `server: 2` downstream where the upstream sent two each, on **every** response, including the
    1-in-6 case where the two `date` values differ. Disposition: **`accepted`**, by the deterministic closure
    gate rather than by Reach First alone — the required outcome is 074's own ("discard no value the base
    relayed"), and every candidate that would change the current shape is inadmissible under
    `DC-S15-CE-2-r1`'s recorded rejections (comma-join) or discards a value on the reached differing-`date` case
    (a single-valued-name collapse table; deduplicating only identical pairs adds a rule with no witnessed need
    and re-fires this seam's own `Invalidates On` 3). No remediation differs from current behavior ⇒ nothing to
    absorb, nothing to ask. It gets one line under `handoff-context.md` → §Accepted risks with its re-open
    trigger, **no task, no Fork, no wave, no new roadmap row** — §RM-008 keeps its three content-coding shapes
    and does not become the owner of occurrence shape.
  - **N6 — new to this file, already owned, and disclosed as an omission in the dispatch brief.** The brief named
    two new notes (N3, N5); the record carries three. N6's claim was verified independently here: `grep -rn
    "response_headers" buster_ripper/` → the definition plus exactly three call sites, all passing
    `resp.headers`, so the `dict` member of the widened annotation is **`unreachable`** from this seam's
    boundaries and both arms behave identically ⇒ `accepted` by Reach First, opening nothing; it takes one line
    under `handoff-context.md` → §Accepted risks. The precision gap
    in that same annotation already has its row — `QB:utils/response-headers-input-annotation` — so no second row
    is written for it.
  - **N3 — new wording, no new obligation.** The docstring posture statement was read by *this* pass (the
    composer's role may not): `utils.py:46-50` says the filter "does not reconcile" the label with the body, names
    it base parity, and points at `roadmap.md` §RM-008. That is the routing 074 recorded, restated by the code
    that landed after it — provenance for `accepted`, not a contract that makes the composed head and body agree.
    Inert wording with no gap in it: no doc line owed.
  - **N1/N2/N4 — carried, unchanged in substance.** N1's exception-channel detail strengthens the same note and
    its installation trigger (`Invalidates On` 4 now names it explicitly); N2's egress fold is S10's terminal
    boundary and the post-`T017` asymmetry between the two directions is recorded-and-accepted by the wave's own
    Yield (`QB:utils/response-headers-mutable-return`, `075-01-yield` §R6); N4's fourth boundary is
    `quality-backlog.md`'s existing row, restated so Yield Red does not read "every response boundary" as four.
- **Recomposition allowance: consumed by this pass, and the cap applies from here.** 074 recorded that `T017`
  *earned* the single post-repair recomposition; this is that recomposition, so the allowance is now spent. Any
  further non-`SAFE` composition of S15 is a **later** one and is capped by rule: Reach First, accept what it
  permits, route nothing else, and give the remainder to the user in one message. Only a fired trigger —
  `Invalidates On` 1–6 — may reopen this seam, and each of §RM-008's three shapes fires 2(d)/2(f)/3 by design.
  No `## Capped seams` entry is written by this pass: nothing is being withheld behind a cap today, CE-1 is
  already elevated (`Reported: —`), and a second list entry would duplicate that elevation.
- **Invariant cross-reference for the two notes routed here: Neutral**, re-run rather than inherited — the
  registry holds zero `Status: active` and zero `Status: wip` entries (eight `planned` PIs under
  `dump/` and `server/`, neither of which names a header relay), and the session declares no `invariants.md`.
  The published `CL:`/`CO:` records were evidence inputs only and classify nothing. No invariant was created,
  edited or promoted by this pass, and no new normative claim about an RFC is made here — deliberately, on
  `claim-208`'s record that an RFC §7.6.1 citation in this lane did not establish what it was cited for.
- **`S1`–`S14`: untouched, terminal, and re-stated as such.** No member list, State, Routing, Permanent or
  dimension cell of the fourteen moved; no repo row was re-extracted (the gate stays third-party-subjects-only);
  no other seam was recomposed, and S15's verdict was neither re-derived nor widened.
- **Hashes.** Before the first cell write of this pass: `claims.md`
  `sha256:c31387380dcc3a85c14f333432550b11cb802d3767919bf91beac8f47c2c823e` (unmoved by this pass — this row is
  the only ledger write outside `handoff-context.md`), `seams.md`
  `sha256:674061369b0baa4dead68f75044baba79b6454d1ad91373e1b0dff9b0b524bfc`; repository state for the pass,
  resolved through Git from the worktree root before the first write:
  `git-commit:ebf0381d8f6ca778b7f3443152de906d69434629`.

## Run-40 changelog (2026-10-06 — scoped adhoc extraction, CR3 wave **`T018`+`T019`+`T020`**, third-party subjects only)

- **Dispatch:** `forward-collect` over the CR3 remediation wave routed by
  [080-assumption-decide](../iterations/080-assumption-decide.md), all three members in one pass so the shared
  premises are seen together. Blueprints and Verification profiles from `handoff-context.md` §Planned Task
  Context; task rows from `plan-summary.md` §Task Queue. Claims side in [claims.md](claims.md) §Run 40
  (rows 390–411, twelve inherited cells).
- **S16 `new`** — `T018` step 1's chain: `client_max_size` configured at the package's one `web.Application`
  construction reaching every route class's body read. Recorded at the **library endpoint**: the repo side is
  one kwarg, and what the plan takes for granted is that a falsy value there silences a check aiohttp performs
  on a path no repo code owns. Terminal `S1`–`S14` do not hold this boundary value; `S7`/`S11` are request-parse
  and commit-boundary seams respectively, not an app-configuration one.
- **S17 `new`** — `T018` step 2's chain, and the seam where the wave's two members meet: 57 (inherited, current)
  says a request-local `timeout=` replaces the session default; 396/397 say what that value means and how far the
  budget it governs reaches; 330/331 (inherited, current) name what a firing deadline raises and whether the
  relay's arm can see it. The post-commit dimension is deliberately *not* judged here — `T019` edits the same
  function's error arm two lines from `T018`'s kwarg, and that crossing is the composition pass's to evaluate.
- **S18 `new`** — `T019` step 2's chain across **both** directions of the boundary: the client's own Content-Type
  into `Request.json()`, and the upstream's into `ClientResponse.text()`, with `read()` as the fallback that must
  stay available. Members are the new decode rows (398–407) plus the current rows the new catch lists stand on
  (51, 68–72, 137). Not a reopening of `S7`: same family, different boundary value — `S7` is terminal
  `out-of-scope` and was not opened, and this seam exists because the unknown-codec arm and the raw-bytes escape
  were never premises there.
- **S19 `new`** — `T020`'s chain, recorded where the test double meets the production loop across `async with`.
  An external service (vLLM `/metrics`) stands behind the boundary but no claim is made about it; what is made
  for granted is the double's contract (408/409), the yield that lets a second tick happen (410), and the
  runner's right to execute the async witness at all (411). 126 (inherited, current) is the arm that stops a
  swallowed cancellation reading as a pass.
- **No seam re-verified, composed or judged** (E8): no `compositions/seam-S1x.md` was read or written, no
  verdict inherited into a seam State, nothing promoted, no `CO:` handle allocated. `## Composition Index` was
  consulted for the four exact ordered member sets, boundaries and dimensions and matched none of them, so no
  composer is skipped and all four carry `Permanent: —`.
- **`S1`–`S14`: untouched, terminal, not reopened** — no member list, State, Routing, Permanent or dimension cell
  of the fourteen moved, and no `repo:` row was extracted, rechecked or recomposed by this pass (the gate stays
  third-party-subjects-only by dispatch). **`S15`: untouched** — live at `registry-counterexample`/
  `accepted-consequence`; no row joined or left it and no dimension moved.
- **Hashes.** Before the first cell write of this pass: `claims.md`
  `sha256:c31387380dcc3a85c14f333432550b11cb802d3767919bf91beac8f47c2c823e`, `seams.md`
  `sha256:e014a6a4fc20797c1ea91fdfbcff2cbaa8f5c27de014c418645f132d4c1b9b2c`; re-read after the writes with the
  values recorded in this pass's receipt.


## Run-40 atomic routing changelog (2026-10-06 — the four non-`HOLD` member rows of `S16`–`S19` dispositioned; `assumption-decide`, atomic non-holding mode)

Dispatch routed `claims.md` rows **391**, **399**, **404**, **409** — the wave's four `Routing: pending` members —
exactly once each. **No seam row's `State`, `Routing` or `Permanent` cell moved**, and no composition artifact was
read or written (E8): all four seams stay `new`/`unverified` with `Permanent: —`, awaiting their first composition.
What changed is what the composer may now consume: every member of every seam is routed, so `S16`–`S19` are clear for
composition, and the wave's Forward act is unblocked.

- **`S16` (`T018`, body cap)** — member **391** routed `accepted-consequence` (Reach: **inert wording**; cross-ref
  **Neutral**). The two readings of "the application serving the request" cannot split an outcome here: the package
  has one flat `web.Application` (`server.py:119`) and `add_subapp` greps zero under `buster_ripper/` and `tests/`, so
  `client_max_size=0` governs every request under either reading, and the app→request propagation the seam's boundary
  value depends on is carried by 390/392–395 (`HOLD`, published). The seam composes over 391's verdict as written,
  with its `UNCLEAR` state retained as the record of *which* wording was ambiguous; the residual belongs to
  `QB:assumption-ledger/row-391-serving-app-wording`, not to this seam.
- **`S17` (`T018`, stream deadline)** — no member was non-`HOLD`; nothing routed here; unchanged.
- **`S18` (`T019`, decode arms both directions)** — member **399** routed `absorbed` (`DC-claim-399-r1`), member
  **404** routed `accepted-consequence` (inert wording). The seam's first dimension ("client-request direction versus
  upstream-response direction (401 against 398/399)") is now **evidenced asymmetric**: 401 `HOLDS` (the request side
  raises a bare `LookupError`), 399 `DOES-NOT-HOLD` (the response side suppresses it inside `get_encoding()` and
  decodes with the fallback). A composition that reads the two directions as one tolerance shape is the misread this
  routing exists to prevent, and the recomposition after `T019` lands is where the corrected reading is tested.
  **`S18` stays open on 399** until that post-implementation composition is `SAFE`. It also carries the two
  header-defective members 402/403: their verdicts stand and their rows are settled (`HOLD`/`none`), but the seam
  **cannot be promoted to a `CO:` record** until `QB:assumption-registry/annotated-promotion-headers` lands, because
  composition eligibility requires every member to resolve to a permanent `CL:` handle.
- **`S19` (`T020`, the EMA witness)** — member **409** routed `absorbed` (`DC-claim-409-r1`). The seam's first
  dimension ("awaitable call result versus an async-context-manager object at one call site (408 against 409)")
  resolves the other way from the premise `T020` was drafted on: a plain `MagicMock` does supply an awaitable
  `__aenter__`, and the enter yields an **auto-child**, not the object the test configured — which is precisely how a
  witness can be green with `compaction.py:49` never executed, because the loop swallows everything at `:55-56`.
  `T020`'s done-when now pins the double shapes and the value assertions that make that silent-green mode unreachable.
  **`S19` stays open on 409** until the post-implementation composition is `SAFE`.
- **Recomposition cap:** none of `S16`–`S19` has been composed at all, so no seam is on its second pass and none is
  capped. `S1`–`S14` untouched and terminal; `S15` untouched at `registry-counterexample`/`accepted-consequence`.

**Hashes.** Before the first write of this pass: `claims.md`
`sha256:2011b68229e2f12c3e6c6365c87b25e769419b1b55c9823f70b50a585f0bea82`, `seams.md`
`sha256:999d807102eaf32565ae3fff7ad0da12cb249bbada416e6f668e55813b24e425`; repository state resolved through Git from
the worktree root before the first write: `git-commit:e204b18e4f82f618e03c808743361637e3b149d2`. Re-read after the
writes in [../iterations/081-assumption-decide.md](../iterations/081-assumption-decide.md).


## Run-41 changelog (2026-10-06 — scoped ledger amendment: CR3 wave seam membership, seams **S16**, **S18**, **S19**)

- **Dispatch:** claims run 41's scoped amendment, mode `wave seam membership`, over the four non-`HOLD` members of the
  CR3 wave's fresh seams — **391** (`UNCLEAR`/`accepted-consequence`, `S16`), **399** (`DOES-NOT-HOLD`/`absorbed`,
  `S18`), **404** (`UNCLEAR`/`accepted-consequence`, `S18`) and **409** (`DOES-NOT-HOLD`/`absorbed`, `S19`). Each sits
  non-`HOLD` inside a seam that is still `new`, so composition cannot run over it as written; each already carries a
  checker's evidence naming exactly what settles it, and the four successor rows use that wording rather than a
  re-derivation. They are re-rowed to one settled proposition each in fresh solo bundles **B310–B313** at
  `new`/`unverified`, and this file's membership moves so no live seam carries a row whose verdict cannot answer it.
  Membership and rows only: this pass checked nothing, composed nothing, re-verified no seam, called nothing safe and
  wrote no verdict (E1/E8). Row table, retirement mapping, bundle table, evidence lookups and the not-rowed list:
  [claims.md](claims.md) §Run 41.
- **S16 `new`/`unverified` (cells unchanged; members 7 → 7).** Retired **391** leaves the list and is replaced in place
  by **412** — the body-size limit is enforced from the `client_max_size` of the application whose runner serves the
  connection and *built the request*, which is the sense 391's verdict read as holding and the sense `T018`'s cap
  premise needs. The fork 391 could not answer was the referent of "the application serving the request" (constructing
  app versus `match_info.current_app`), and the sub-app sense is established inert on this tree by the routing's own
  DC-level disposition — one flat `web.Application`, zero `add_subapp` — so it travels as 391's retired record rather
  than as a row. The `unset default versus an explicit value` dimension clause is re-worded to cite 390/**412** instead
  of 390/391; the boundary value, the subject list (391's `pkg:aiohttp@3.13.3` is 412's) and every other dimension
  clause are untouched. `230` and the `HOLD` rows 390/392–395 keep their positions and cells.
- **S18 `new`/`unverified` (cells unchanged; members 17 → 17).** Retired **399** is replaced in place by **413** and
  retired **404** by **414**. 413 rows the positive the seam's boundary needs: the response direction is never loud —
  `get_encoding()` suppresses the codec-lookup failure inside `contextlib.suppress(LookupError, ValueError)` and the
  payload decodes with UTF-8 or the session's `fallback_charset_resolver`, which is why the request direction (401,
  `HOLDS`, a bare `LookupError`) and the response direction are *not* one tolerance shape. 414 rows the character
  reading `T019` step 2(a) stands on — same **characters** as `errors="strict"`, for **complete in-context** sequences
  — with the error-path half of the checker's own settling sentence left to live row **403**, which already asserts it
  (`HOLD`/`none`), so §3d forbids a second row for it. Three dimension clauses are re-worded to name live rows: the
  direction clause (401 against 398/**413**), the nonexistent-codec arm (**413**/401/405) and the
  strict-versus-`replace` axis (402 against 403/**414**). No subject enters or leaves — both successors carry their
  retired row's subject — and 51, 68–72, 137, 398, 400–403, 405–407 keep their cells and positions exactly.
- **S19 `new`/`unverified` (cells unchanged; members 5 → 5).** Retired **409** is replaced in place by **415**: a
  plain, unspecced `MagicMock` folds `__aenter__`/`__aexit__` into its magic set and routes those children to
  `AsyncMock`, so `async with` succeeds without explicit configuration — the positive `T020`'s amended done-when leans
  on, against the falsified premise its witness was drafted on. The qualifier is part of the proposition: 409's own
  verdict names bare `Mock()`/`NonCallableMock()` and a `spec=` lacking the dunder as the constructions where
  `async with` genuinely fails, and the plan carries that as its done-when's forbidden-shape list rather than as a row
  (see [claims.md](claims.md) §Run 41's not-rowed list). The `awaitable call result versus an async-context-manager
  object` dimension clause is re-worded to cite 408/**415**. `126`, `408`, `410`, `411` untouched.
- **S17: no write, and recorded as such.** None of its five members (`57`, `396`, `397`, `330`, `331`) is retired or
  replaced by this amendment — all five are `HOLD`/`none` — so its member list, subjects, boundary value, dimensions
  and cells are exactly as run 40 wrote them.
- **Cells held across all four seams.** `State`, `Routing` and `Permanent` are untouched in every row of this table,
  the three amended seams' `new`/`unverified`/`—` included — whether an amended member set now composes, and whether any
  of the three may leave `new`, is the composer's and the decide route's write (E1/E8). No `CO:` handle enters or
  leaves `Permanent`: the `## Composition Index` was consulted for the three moved member sets only, and none of its
  five records covers a body-size cap propagation, a charset-decode exception class, or a test double's
  context-manager surface. No `compositions/*.md` artifact was read or written, and none inherited or discarded.
- **The retired rows' own cells.** 391, 399, 404 and 409 keep their `State`, `Routing`, `Bundle` and `Permanent` cells
  and their extraction-time `Seams` declarations, retired to terminal history exactly as rows 3, 20 and 93 are
  retired: verdicts and routings stand, they simply leave live membership. Both `accepted-consequence` routings and
  both `absorbed` routings were left as run 40's routing pass wrote them, per the dispatch. With them gone,
  `B300`/`B303`/`B305`/`B307` name only settled `HOLD`/`none` rows and carry nothing dispatchable; the wave's live
  dispatch is `B310`–`B313`, four agents. Whether the two `QB:assumption-ledger/row-*` doc lines close on this
  landing is the decide route's write.
- **Consequence for the queue, as ledger state rather than a decision:** four rows are `new`/`unverified` in four
  fresh solo bundles, and `S16`, `S18` and `S19` now list only rows whose verdicts are either current on file or
  freshly dispatched for this amendment — no live seam carries a row whose verdict cannot answer it. The composition
  pass remains blocked until B310–B313 return; nothing here states whether the amended sets compose, and `S18`'s
  promotion is still held by `QB:assumption-registry/annotated-promotion-headers` on members 402/403, unchanged by
  this pass.
- **S1–S15:** rows, members and cells untouched. No member of any earlier seam is retired, replaced or re-bundled by
  this amendment, no `repo:` row was created, rechecked or recomposed, and no unrelated composition was inspected for
  staleness.
- **Hashes.** Before the first cell write of this pass: `claims.md`
  `sha256:dc6fc0db73c3d498233c39ee185ea69b4f8f5cd1119186b9f89be8c0380b5d7f`, `seams.md`
  `sha256:16ceaa0526f224fe11d565700bec0aa4d349a29294b92cff23e7bb7134a32f7e`; re-read after the writes with the values
  recorded in this pass's receipt.

## Run-42 composition routing changelog (2026-10-06 — first compositions of **`S16`**, **`S17`**, **`S18`**, **`S19`** dispositioned; `assumption-decide`, composition-routing mode)

Dispatch: **composition routing**, CR3 wave run 42. Inputs read: the four composition artifacts
(`compositions/S16-body-limit.md`, `S17-stream-deadline.md`, `S18-decode-failure-surface.md`,
`S19-async-witness.md`), the member verdicts they cite, `claims.md`'s member rows **for cells only** (States,
Routings, Permanents, subjects), `../handoff-context.md` §Planned Task Context → `T018`/`T019`/`T020` (including
081's two amendment sub-blocks) and §Deterministic Closure Ledger, the `T018`–`T020` rows in `../plan-summary.md`,
the registry `## Handle Index`/`## Claim Index`/§Source Sessions, and the production sites the verdicts bear on
(`server.py`, `routes/messages.py`, `routes/chat.py`, `routes/count_tokens.py`, `routes/passthrough.py`,
`utils.py`, `compaction.py`, `config.py`, `buster_ripper_legacy.py`, plus `web_protocol.py`'s dispatch as recorded
by row 230's verdict). Repository state before the first routing write:
`git-commit:5f7f9b9d43a50f571c497c0c52bf18406e77a567` (resolved through Git from the supplied worktree root; no
other repository state inspected, no `.git/` path read). Full reasoning, the closure packets and the three
premise corrections: [082-assumption-decide](../iterations/082-assumption-decide.md).

**Nothing was checked, recomposed, promoted or dispatched.** No verdict file was written or edited, no State cell
moved, no composition performed or judged `SAFE`, no claim re-verified on unchanged bytes, no fingerprint
recomputed. This pass routes verdicts that already exist.

**Cells written — four `Routing` cells, `unverified` → `absorbed`, one per seam. `State` and `Permanent` cells are
untouched everywhere** (`S16`–`S19` stay `new`; `S1`–`S14` stay `out-of-scope`/`registry-safe`; `S15` stays
`registry-counterexample`/`accepted-consequence`). One token per seam row, following `S15`'s precedent, where a
seam with one `absorbed` and one `accepted` finding carries a single Routing value — the per-finding dispositions,
their ledgers and their Reach verdicts live in `../handoff-context.md` §Deterministic Closure Ledger (run-42) and
§Accepted risks, not in these cells.

| Seam | Verdict on file | Findings routed here | Routing written | Where each lands |
|---|---|---|---|---|
| `S16` | `UNCLEAR` | (a) cap meaning/measurement point — **absorbed**; (b) the 413 error envelope — **accepted** (`unreachable` post-`T018`); the `2**25` decompression note — **accepted** (1% → §RM-011); the no-production-caller arms census — **accepted** (`unreachable`); "four POST route classes" — **accepted** (inert wording → collector doc line) | `absorbed` | `DC-S16-a-r1` → `T018` step 1; §Accepted risks ×4 |
| `S17` | `COUNTEREXAMPLE` | CE-1 — **absorbed**; CE-2 — **absorbed**; the sibling-arm residual + `N3`/`N4` (raised by this pass while checking reach, not by the composer) — **accepted**, out of scope, → §RM-010 | `absorbed` | `DC-S17-CE-1-r1`, `DC-S17-CE-2-r1` → `T018` step 2, whose done-when gained the all-components-`None` pin |
| `S18` | `COUNTEREXAMPLE` | CE-1 — **absorbed**, zero blueprint change; MC-1 — **absorbed**, zero blueprint change (its fate is settled by `DC-claim-399-r1`); MC-2 — **`malformed`** → the collector | `absorbed` | `DC-S18-CE-1-r1` → `T019` step 2(c); `DC-S18-MC-1-r1` → `T019` step 2(b); MC-2 at `QB:assumption-ledger/s18-s19-unrowed-class-atoms` |
| `S19` | `COUNTEREXAMPLE` | CE-1 — **absorbed**, zero blueprint change; CE-2 — **absorbed**, zero blueprint change; CE-3 — **absorbed with the one amendment this pass wrote** (done-when item (6)'s 410-as-timer premise is unsatisfiable as written) | `absorbed` | `DC-S19-CE-1-r1`, `DC-S19-CE-2-r1`, `DC-S19-CE-3-r1` → `T020` step 1 |

**All four seams stay open; none is resolved, capped, or closed by this table.** Final `resolved` needs each
seam's post-implementation composition to come back `SAFE` (or every finding left on it `accepted`) — `S16`/`S17`
after `T018`, `S18` after `T019` **and** after the `MC-2` atom is rowed and checked, `S19` after `T020` over the
landed `tests/test_compaction.py` bytes. These were the **first** compositions of all four (each record says so:
"no prior `S1x` record exists"), so **no recomposition allowance has been spent** and none is capped: each still
has its one post-repair recomposition owed. Zero `question` was emitted and zero `breaks-it`
(`T018`/`T019`/`T020` and the wave's anchor/order/retry owners all stand), so `FK-011` remains the only open Fork.

**Promotion eligibility, restated against these cells (record conditions, not evidence).** `S16` ineligible for a
`Composition:` record twice over (`UNCLEAR` verdict; row 230 carries `Permanent: —`). `S17` ineligible (rows 330 and
331 carry no permanent handle; its `COUNTEREXAMPLE` would otherwise be promotable, as `S15`'s was). `S18`
ineligible on three counts (68, 69, 71, 72, 137, 402, 403 handleless; 68–71 record `Source Fingerprint:
unavailable`; 402/403's annotated line-2 headers already registered at `QB:assumption-registry/annotated-promotion-
headers`). **`S19` eligible** — `PROMOTION: ELIGIBLE` on its own header and all five members carry `CL:` handles.
Every one of these is identity bookkeeping: the named rows are all `State: HOLD` with verdict artifacts and
re-verified digests, and the precedent for the same condition is the ledger's own note at
`claims.md:577`: S3/S5 `PROMOTION: INELIGIBLE — no permanent member handles`, later published. One correction
carried for the record: rows 230/330/331/137 are `pkg:aiohttp@3.13.3` subjects and 68–72/402/403 are
`lang:python`, **not** repo-subject rows, so the standing third-party-subjects-only ruling is not the reason they
lack handles — promotion coverage is (run 40's promotion pass took rows 390–411 and nothing else).

**Membership and cells this pass did not touch, and why.** No member was added to or dropped from any seam: a
composition-routing pass routes verdicts, and `S18`'s missing atoms (its §Invalidates On 3/4) are the collector's to
row at the next re-base, at which point membership moves by that pass's rule. No `repo:` row or seam was created,
rechecked or recomposed; no `out-of-scope` seam was opened.

**Hashes.** `claims.md` was **not written** by this pass: `sha256:988bc3f9a8f096338c9cb3e181093db5838f3280c3b0fb7bd7b7c1b4f2b833ca`
before and after. `seams.md` after the four `Routing` cells and before this changelog section:
`sha256:2aed2ea82bf62346e1ef99c7fc3a1e3ed2ba905b158ddb7aeea95b47cc3eb03b`; the hash after this section is recorded
in this pass's receipt.


## Run-43 changelog (2026-10-06 — scoped amendment: the one `S18` §MC-2 atom is rowed; seam **S18** membership only)

Dispatch: **scoped amendment**, mode `one malformed atom`. Required delta: one new `claims.md` row under a new bundle
key settling what Python's `bytes.decode()` produces for a codec name that is **registered but not a text encoding**, and
its membership in `S18` **only** — retire nothing, re-bundle no existing row, leave `S16`, `S17`, `S19` and every
terminal seam alone. Inputs read: this file's `S18` row, `claims.md` rows 401/402/403/405/406/407/413/414 for cells and
scope, `compositions/S18-decode-failure-surface.md` §MC-2 and §Invalidates On 4, and the owning queue entry
`QB:assumption-ledger/s18-s19-unrowed-class-atoms` item **(A)** in `../quality-backlog.md`. No project source was read
and nothing was run (E2); no plan, interview, iteration or handoff file was opened for anything about a library (E4).

**What moved — one row's membership, and the clause that names it.** `S18`'s `Claim rows` cell gains **416**
(17 → 18), and dimension 2's known-codec arm is extended from two cases to three: invalid bytes under a known codec
(51/402) — a name that does not exist at all (413/401/405) — **a name that does exist but is not a text encoding
(416)**. Dimension 3's three-catch-target clause now also points at 416 for the completeness question §MC-2 left open,
so the recomposition can find the atom at the clause that was blocked rather than only at the codec arm. Nothing else in
the cell was re-worded, and the other five dimensions are verbatim.

**Cells this pass did not write.** `S18`'s `State` (`new`), `Routing` (`absorbed`) and `Permanent` (`—`) are exactly as
run 42 left them, and so are `S16`, `S17`, `S19` and every terminal seam `S1`–`S15` in all seven columns — no member,
no dimension, no boundary value, no subject list of theirs moved. Whether an 18-member set composes, and whether `S18`
may leave `absorbed`, is the composer's and the decide route's write (E1/E8); this pass rowed the atom §MC-2 said was
missing and stopped. No `registry-safe`/`registry-counterexample` state was taken from the `## Composition Index`: it
still holds the same five `CO:` records, none covering this boundary, so no composer is skipped and no handle enters or
leaves any seam's `Permanent`.

**Row 416 asks; it does not answer.** The proposition asserts that the call raises *one of the three classes the seam
already carries*, which leaves which-class open to the checker, and is falsified by both alternatives §MC-2 names — a
fourth class, or a successful return of a non-`str` (a negative form was rejected precisely because it would read `HOLDS`
on that second case). The collector entered no class name, no verdict and no `CL:` handle; the row is `new`/`unverified`
under solo fresh key `B314`, and `S18`'s one `lang:python` neighbour set was not re-bundled around it. Item **(A)** of
`QB:assumption-ledger/s18-s19-unrowed-class-atoms` is thereby discharged by dispatch; items **(B)**, **(C)** and **(D)**
are not rowed here — the dispatch scopes this pass to the single atom and puts `S19` off limits — so that key stays open.

**Hashes.** `claims.md` before the run-43 section, and after the re-read:
`sha256:988bc3f9a8f096338c9cb3e181093db5838f3280c3b0fb7bd7b7c1b4f2b833ca` → the value in this pass's receipt.
`seams.md` after the one `S18` row edit and before this changelog section:
`sha256:f48dec18cfd8b1f8af0387252085497049315ca2831d84ca3f03b45e8ab7f24d`; the hash after this section is recorded in
this pass's receipt. Repository state is unchanged from run 42's record: `git-commit:5f7f9b9d43a50f571c497c0c52bf18406e77a567`.

## Run-44 changelog (2026-10-06 — scoped amendment: post-repair invalidation, seams **`S16`**, **`S17`**, **`S18`**, **`S19`** `State` cells only)

Dispatch: **scoped amendment**, mode `post-repair invalidation firing`. Required delta: the four `State` cells set so the
four unspent post-repair recompositions can run, alongside `claims.md` row **57**'s firing to `recheck` under fresh key
**B315**. Check nothing, compose nothing. Retire nothing, re-bundle no row that still answers, leave every terminal row and
terminal seam alone; third-party subjects only.

**What moved — four `State` cells, `new` → `recheck`.** `Routing` (`absorbed`, run 42), `Permanent` (`—`), `Claim rows`,
`Subjects`, `Boundary value` and `Dimensions` are byte-identical to what run 43 left on all four rows. `S6`'s precedent
(run-26: "`State` deliberately stays `recheck` because the registry basis is retained-superseded") is the ledger's own
`recheck` + `absorbed` pairing, which is what these four now carry: their first-composition findings were routed into
`T018`/`T019`/`T020` and are exactly what the recomposition must now re-read.

**The trigger, per seam — each seam's own `Invalidates On`, not a general freshness sweep.**

| Seam | What the wave moved under it | The record that names it |
|---|---|---|
| `S16` | `server.py` `create_app()` now passes `client_max_size=0`, so dimension 1's "unset default versus an explicit value" arm is now the explicit one, and dimension 2's zero arm (393's falsy skip) became the live spelling | `compositions/S16-body-limit.md` §Invalidates On 2 ("`server.py:119` starts passing `client_max_size`") and 3 ("this is `T018` step 1's own surface, so the post-implementation recheck of this seam is mandatory") |
| `S17` | The streaming `session.post` in `routes/messages.py` now carries an explicit all-`None` `ClientTimeout`, and member **57** — the premise that says a request-local timeout *replaces* rather than merges — is being recomposed with an undischargeable version pin | `compositions/S17-stream-deadline.md` §Notes **N1** and §Invalidates On line 1 ("Re-verification of member 57 at the pinned 3.13.3 bytes"), plus the line naming a `timeout=` change at `messages.py:137-141` |
| `S18` | `routes/count_tokens.py` now normalizes before keying and has a third decode arm — the consumer end of the charset→exception-class crossing — and the seam carries a member row retired by run 41's amendment (`404` → `414`) | `S18`'s members and dimensions are unchanged; the chain's consumer moved, which is §3c's recheck condition. `416` (run 43's §MC-2 atom) is now `HOLD` with `CL:python/bytes-decode-non-text-codec-error`, so the recomposition reaches §MC-2's question with evidence in hand |
| `S19` | `tests/test_compaction.py`'s two EMA witnesses now drive production `poll_kv_cache`, so the witness's right to run the loop more than once is being exercised against the real loop, and the seam carries a member row retired by run 41's amendment (`409` → `415`) | `compositions/S19-async-witness.md`'s member/dimension set is unchanged; the boundary's consumer side moved |

**Membership: nothing joined, nothing left.** All four member lists are exactly run 43's (`S16` 7, `S17` 5, `S18` 18,
`S19` 5). Row 57 stays a member of `S17` while it is `recheck` — a fired premise stays on the seam it feeds; the
recomposition consumes its fresh verdict, and dispatching the composition before B315 returns is the orchestrator's
ordering call, recorded here so the dependency is visible rather than inferred.

**Cells this pass did not write.** `S1`–`S15` in all seven columns — the eleven `out-of-scope` rows, `S6` at
`registry-safe`, and `S15` at `registry-counterexample`/`accepted-consequence` all stand as run 36's scope correction and
runs 37–39 left them, including `S8`, which also carries row 57 as a member and whose `State` deliberately does **not**
move on this firing. No `registry-safe`/`registry-counterexample` state was taken from the `## Composition Index` for any
of the four: it still holds the same five `CO:` records, none covering these boundaries, so nothing is skipped and no
handle enters or leaves any `Permanent`. No seam was composed, re-verified, called safe, or given a verdict; no `Routing`
cell moved; no claim row was created, retired or re-bundled here; no `repo:` row or seam was opened.

**Hashes.** `seams.md` before the four `State` cells: `sha256:b5cd927702fc168d5b41647f1e22ba785bec949d621d236d8bec97881a25bd12`;
after the cells and this section: the value in this pass's receipt. `claims.md` before its row-57 cell write:
`sha256:3afed35fad28ca62eeebe26aa31e2a5ab09fe5ad0bbe163270448646ffd59eda`; after: its own receipt value. Repository state
at this landing: `git-commit:9c90675f69998e0ea7847c6b456c2e3f46f8a81c` — the CR3 wave's Checkpoint 191 landing, which
supersedes run 42's `5f7f9b9d…` baseline. The installed `aiohttp` distribution is `3.13.3` across that movement, with
`client.py` byte-identical to run 37's record (`7ce41fc1…`); no `pkg:` row other than **57** fired, and no `lang:` row
fired at all.

## Run-45 changelog (2026-10-06 — scoped amendment: `S18` Note N6's unrowed atom is rowed; seam **S20** added, no other cell moved)

- **`S20` `new`** — the seam `compositions/S18-decode-failure-surface.md` §Notes **N6** declined to compose. Its own
  words are the ledger's definition of an unrowed chain: "the deciding sources (`aiohttp/web_response.py`,
  `aiohttp/payload.py`'s payload classes) are cited by no S18 member claim". Members are `claims.md` run-45 rows
  **417–424**, all `new`/`unverified`, all bundling fresh (`B316`–`B319`) — no member inherits a verdict, because
  nothing on file has ever read either unit for any of these propositions. Boundary value: one `str` entering a
  response object through the constructor and leaving through three doors (the `body` attribute a caller reads back,
  the `text` property, the payload write to the wire). The dimensions are the discriminating questions N6 answers in
  prose with no row behind any of them — accepted-versus-rejected at construction, the representation the getter hands
  back, equality with the input across the round trip, which exit the value is readable at, what string-coercion of
  whatever is stored yields, and the charset applied to the bytes that reach the wire.
- **Why not `S9`/`S11` membership**, which is where N6 suggested routing: both are terminal; `S1`–`S15` are frozen by
  this dispatch and by run 44's standing "`S1`–`S15` stay as they are", and adding a member to an `out-of-scope` seam
  would spend a composition allowance that lane no longer holds.
- **Why not `S18` membership**: N6 states these atoms are "outside `S18`'s dimensions", and `S18`'s boundary value is
  the *charset a peer names* — no row 417–424 involves a peer-named charset, so joining it would move a `recheck`
  seam's member list while leaving its dimensions mis-described. `S18`'s `State`, `Routing` and `Permanent` cells are
  untouched, and `S16`, `S17`, `S19` likewise: the CR3 recompositions those three are waiting on are their own lane's
  work and a new seam does not spend them.
- **No seam re-verified, none recomposed here** (E8): whether these eight facts compose into N6's conclusion is the
  composition pass's decision, after the four atomic bundles return. `Permanent` stays `—` — the registry
  `## Composition Index` still holds the same five `CO:` records (`lifecycle-teardown`, `tls-posture-crossing`,
  `format-value-crossing`, `payload-and-header-pairing`, `env-var-flag-crossing`) and none covers a response-body
  round trip, so §3c's retain-your-handle clause has nothing to retain and no composer is skipped.
- **Dispatch shape for the orchestrator:** 4 claim bundles (`B316` rows 417-419, `B317` rows 420-422, `B318` row 423,
  `B319` row 424) → then `S20`'s first composition. Row 424's `lang:python` subject means it must not be folded into
  any `pkg:` bundle; the three `pkg:` bundles all name `aiohttp 3.13.3` at `web_response.py` `3ca73388…` and
  `payload.py` `3ba9ec60…`.
- **Hashes.** `seams.md` before this write: `sha256:7a9f79f1ec0463d9a74bdbd355ba39b8658cebb26423e1abf8fed1266c6554bd`;
  after the `S20` row and this changelog: the value in this pass's receipt. `claims.md`'s before/after pair is in that
  receipt and in `claims.md` §Run-45.

## Run-45 atomic routing changelog (2026-10-07 — `S20`'s four `Routing: pending` members dispositioned; `assumption-decide`, atomic non-holding mode)

Dispatch routed `claims.md` rows **418**, **419**, **422**, **423** — the eight-row run-45 set minus the four `HOLD`
members — exactly once each, and **no seam row's `State`, `Routing` or `Permanent` cell moved**: `S20` stays
`new`/`unverified`/`—`, because this pass composed nothing and promotion needs every member resolved. Rows
417/420/421/424 are evidence history, not new work, and were not touched.

- **`S20` (`T019`, the `web.Response` body round trip)** — 418 and 423 `absorbed`, 422 `absorbed`, 419 `malformed`
  (returned to `forward-collect`). What the four verdicts jointly settle is **not** that the seam's boundary is
  broken: it is that the boundary works on two of its three doors and not on the third. 420/421 `HOLD` (the text
  reaches the wire; `.text` returns the string), 418/423 `DOES-NOT-HOLD` (`.body` is the `StringPayload`, and `str()`
  of it is the object repr). The seam's dimension "equality with the input across that round trip (419)" therefore
  has **no member that can carry it** until successor row **425** lands, and `S20` cannot recompose while 419 is
  non-`HOLD`.
- **What left this ledger, and how.** The consequence of 418/422/423 is a repo call site — `messages.py:106` reads
  `.body` back off the object `messages.py:151-155` built with `body=<str>`, and `utils.py:123-140`/`config.py:122-123`
  then match an object repr. Under the standing third-party-only ruling that cannot be a row or a seam here, so it is
  routed as **task `T019.001`** with the reach determination, the base comparison and the executed check pinned in
  `evidence/n6-reach-and-base-comparison.txt`, and the closure in `handoff-context.md` §Deterministic Closure Ledger —
  2026-10-07 (`DC-claim-418-r1` primary, `-422-r1`, `-423-r1`). The SGLang **streaming** context-overflow retry is
  inert on both arms of its loop; the relayed response is not. No `repo:` row, no repo seam, no `CO:` record.
- **Recomposition cap:** none applies. `S20` has never been composed, so this is not a seam's second pass and nothing
  is capped. `S18`'s Note N6 stands as written and is now adjudicated outside that seam; `S18`'s cells were not
  touched. `S1`–`S15` untouched and terminal (`S15` stays `registry-counterexample`/`accepted-consequence`);
  `S16`–`S19` stay on their post-implementation recompositions as run 40/42 routed them. `S11`'s §Invalidates On 5 is
  **cited, not reopened** — the direction this pass established is the opposite of the one it anticipated (no
  reachable arm's text can match the pattern), which strengthens rather than loosens `DC-S11-SGL-r1`'s foreclosure.
- **Zero Forks.** `FK-011` remains the session's only open Fork: base parity is silent at this site rather than
  ranking two outcomes, and the file's own working peer supplies the required behavior, so the closure gate returned
  one admissible remediation.

**Hashes.** `claims.md` before this pass's first write: not recoverable from the ledger — the run-45 promotion pass
left its own after-value to its receipt (`713d6298…` is that pass's mid-section value); this pass is the only writer
since, on mtime and on the ledger's own "the atomic queue is empty" line. After this pass's four Routing cells and its
§Run 45 routing section: `sha256:36fee430c8e250dde3dd1412c2d08d590f3d44c06a82419c9b85249ff81dec98`. `seams.md` before
this section: `sha256:d56fc19c6b7b7342935254fbc5459a9276489e3d9b82b4ba8f1af967c6484c3f` (unchanged since run 45 wrote
it); after: the value in this pass's checkpoint receipt — a file cannot record its own post-write digest, and the
§Run-45 row and changelog were not rewritten, only this section is appended. Repository state at this landing:
`git-commit:9c90675f69998e0ea7847c6b456c2e3f46f8a81c`.

## Run-46 changelog (2026-10-07 — scoped amendment: malformed 419 retired to terminal history, successor **425** rowed; seam **S20** membership only)

- **Dispatch:** claims run 46's scoped amendment, mode `one malformed row successor`, over `claims.md` row **419**
  (`UNCLEAR`/`malformed`, `S20`). Its verdict says the blocker is the noun — "those `.body` bytes" has no bytes-valued
  referent once 418 settles the getter hands back the `StringPayload` — and run 45's routing pass returned the row to
  the collector lane with its successor's wording specified; row **425** is that wording, held verbatim. Membership and
  one row only: this pass checked nothing, composed nothing, re-verified no seam, called nothing safe and wrote no
  verdict (E1/E8). Row table, the wording's four prohibitions, retirement mapping, bundle table, evidence lookups and
  the not-rowed list: [claims.md](claims.md) §Run 46.
- **`S20` `new`/`unverified` (cells unchanged; members 8 → 8).** Retired **419** leaves the list and is replaced
  **in place** by **425**, keeping 419's position — run 41's convention for 391→412, 399→413, 404→414, 409→415. The
  dimension clause `equality with the input across that round trip` is re-worded, because its three named alternatives
  ("same object, encoded copy, or lost text") were 419's bytes-shaped answer space and the row that carried them no
  longer answers anything: it now reads **(425 — whether the payload read back off the `body` getter decodes to the
  string that went in, under the response's own charset, or to something else)**. Every other clause stands as run 45
  wrote it, 418's `bytes` versus a payload object versus `None` included — that is 418's own proposition, not a
  re-statement of 419's — as do the boundary value, the subject list (419's `pkg:aiohttp@3.13.3` is 425's) and the
  other seven members' positions.
- **Why the seam stays live and answerable rather than being closed.** The dispatch's requirement is that no seam
  carry a row whose verdict cannot answer it, and 419 was the only live member that failed it: its `UNCLEAR` refused
  the fork in its own noun, while 418/422/423's `DOES-NOT-HOLD` verdicts answer the propositions those rows actually
  state and are already `absorbed` with their consequences routed to `T019.001`. What run 45's routing pass wrote about
  the gate itself — "`S20` cannot recompose while any member is non-`HOLD`", and that the seam's
  `equality with the input across that round trip` dimension "has **no member that can carry it** until successor row
  **425** lands" — is quoted here, not interpreted: this pass neither reads that rule nor claims the amended set now
  satisfies it. Whether `S20` composes, and whether it may leave `new`, is the composer's and the decide route's write
  (E8). `State` and `Routing` stay `new`/`unverified` and `Permanent` stays `—`. The `## Composition Index` was
  consulted for this member set only; its six records cover no response-body round trip, so no composer is skipped and
  no handle enters or leaves `Permanent`. No `compositions/*.md` artifact was read or written, and none inherited or
  discarded.
- **The retired row's own cells.** 419 keeps `State`, `Routing`, `Bundle` and `Permanent` (`UNCLEAR`/`malformed`/`B316`/`—`)
  and its extraction-time `Seams: S20` declaration, retired to terminal history exactly as rows 3, 20, 93, 391, 399,
  404 and 409 are retired: verdict and routing stand, it simply leaves live membership. Both were left exactly as run
  45's passes wrote them — this pass re-routes nothing — and `verdicts/claim-419.md` stays on disk as the blind record
  the successor is built from. With 419 gone, `B316` names only 417 and 418, both settled, and carries nothing
  dispatchable; `B317`/`B318`/`B319` already carried nothing. **`B320` is the ledger's only dispatch — one agent**, and
  the atomic queue that run 45's promotion fold recorded as empty is open again for exactly that row.
- **Rows 417, 418, 420, 421, 422, 423, 424: no write, and recorded as such.** None was marked `recheck` and none was
  re-bundled: neither named unit moved (`web_response.py` `3ca73388…`, `payload.py` `3ba9ec60…`, both byte-identical to
  run 45's pins), aiohttp is unchanged at 3.13.3, no relation end or mechanism moved, and 419's `UNCLEAR` is the absence
  of an answer to a malformed proposition rather than a contradiction of another row's verdict.
- **`S1`–`S19`:** rows, members and cells untouched. `S1`–`S15` stay terminal and frozen; `S16`–`S19` stay on their
  post-implementation recompositions as runs 40/42/44 routed them. No member of any earlier seam is retired, replaced or
  re-bundled by this amendment, no `repo:` row or repo seam was created, rechecked or recomposed, and no unrelated
  composition was inspected for staleness. `S18`'s Note N6 stands as written: this pass neither confirms nor retires it.
- **Hashes.** Before the first cell write of this pass: `claims.md`
  `sha256:36fee430c8e250dde3dd1412c2d08d590f3d44c06a82419c9b85249ff81dec98` — the run-45 routing pass's own recorded
  after-value, so nothing wrote between the two — and `seams.md`
  `sha256:7fc22845f14b3a3564cae9878b226ec901d1a9da092b75887763d80410cf2654`; re-read after the writes with the values
  recorded in this pass's receipt. Repository state at this landing:
  `git-commit:08c0a9f7206635e1058faa85dd73d842e64a3384` (`T019.001` landed at `e645ff0` and was accepted-local at
  `7a6639b`, both after run 45's routing recorded `9c90675f…`). No repo file was modified by this pass; it writes
  ledger only.

## Run-46 routing changelog (2026-10-07 — row **425** dispositioned and seam **`S20`** given its terminal disposition; `assumption-decide`, atomic non-holding + seam-disposition mode)

Dispatch: route `claims.md` row **425** (`DOES-NOT-HOLD`, bundle `B320`, the only non-`HOLD` member of the live seam
`S20`) under the Reach Gate, reading the repository bytes where reach requires it, and decide `S20`'s terminal
disposition on the merits. Full reasoning, the closed caller census and the executed arm-by-arm check:
[086-assumption-decide](../iterations/086-assumption-decide.md) and
[`evidence/claim-425-charset-arm-reach.txt`](../evidence/claim-425-charset-arm-reach.txt); the row's own cells, the
reach table and the invariant cross-reference: [claims.md](claims.md) §Run 46 routing.

**Cells written — one per file, and nothing else anywhere.** `claims.md` row 425 `Routing`: `unverified` →
`accepted-consequence`, its `State` (`new`) and `Permanent` (`—`) left for the promotion lane. This file's `S20`
`Routing`: `unverified` → **`absorbed`**. `S20`'s `Claim rows`, `Subjects`, `Boundary value`, `Dimensions`, `State`
(`new`) and `Permanent` (`—`) are byte-identical to what run 46's amendment left; rows 417/418/420/421/422/423/424 and
retired 419 keep every cell run 45's routing and the run-45/46 passes wrote; `S1`–`S15` stay terminal and were not read
for staleness; `S16`–`S19` keep run 44's `recheck`/`absorbed` pairing and their spent post-repair allowances.

**`S20` closes `absorbed`. The evidence that discharges it is already published, and no recomposition is owed.**

1. **The seam's one question is closed, with the answer recorded rather than assumed.** Run 45 created `S20` solely
   because `S18`'s Note N6 "declined to compose" a chain whose every link was an unrowed library proposition — whether
   a `str` handed to `web.Response(body=)` survives the constructor-in / read-back-out crossing to the overflow
   matcher. It does not survive through `.body` (418: the getter hands back the `StringPayload`; 423/424: `str()` of it
   is the object repr, which `config.py:122-123`'s ASCII patterns cannot match); the text does survive to the wire
   (420) and to `.text` (421), and `.text` is the only accessor that returns it (422). The finding was real, `T019.001`
   repaired it (`messages.py:117` now reads `result.text`; landed Checkpoint 197 at `e645ff0`, accepted-local at
   [085-01-yield](../iterations/085-01-yield.md) with the two fail-first witnesses re-reddened by the reviewer's own
   mutation replay), and row 425 settles the last dimension the seam lists — equality across the round trip holds
   *only* where the response's charset resolves to utf-8.
2. **A recomposition could not run, and routing 425 does not change that.** `assumption-compose` §Preconditions:
   "Every supplied session verdict must start with `VERDICT: HOLDS` … If one does not, write `VERDICT: MALFORMED`:
   composition ran before atomic resolution. A `DOES-NOT-HOLD` claim is routed before composition and cannot be
   smuggled in as an alternate premise." Four of the eight live members (`418`, `422`, `423`, `425`) are
   `DOES-NOT-HOLD`, and a routing pass never alters a `State`, so accepting 425 leaves the guard exactly as it was: the
   composition returns `MALFORMED` on member state regardless of this pass's disposition.
3. **Satisfying that guard would be ceremony, and the ledger already says so in its own words.** It would take retiring
   `418`/`422`/`423`/`425` and rowing four positive successors — re-asking, under fresh bundles and fresh checkers,
   propositions whose verdicts are on file and whose falsification *is* the finding. Contrast run 41, which did
   re-row four non-`HOLD` members (`391→412`, `399→413`, `404→414`, `409→415`) and was right to: each successor rowed
   "the positive the seam's boundary needs" for a **live, unmet** premise of `T018`'s cap, `T019` step 2(a) or `T020`'s
   done-when, and the amendment unblocked compositions those seams still owed. `S20` has no unmet premise: the single
   thing the landed code stands on is 421's `HOLDS` wording at the spelling 421's verdict enumerates, published as
   `CL:aiohttp/web-response-str-body-text-property` with `CL:aiohttp/web-response-str-body-accessor-surface` naming
   `.text` the only such accessor. This pass therefore applies `DC-claim-422-r1`'s own rejected-candidate standard — a
   re-extraction "would spend a checker on a proposition nothing asks" — and names **no successor row for the
   collector**.
4. **What composition is for has no open edge here.** A composer hunts counterexamples that no single atom states:
   a discarded discriminator, a fallback selecting a different semantic kind, mapped and unmapped union members taking
   materially different paths, more than one terminal settlement. Traced against this seam's own dimension list, every
   one is answered by a member on file — accepted-versus-rejected at construction (417), the representation the getter
   returns (418), equality across the round trip (425), which exit the value is readable at (420 wire / 421 `text` /
   422 the whole accessor set), what string-coercion yields (423 resting on 424), and the charset applied to wire bytes
   versus the advertised one (420 read against 425). The one genuinely cross-atom question — *can the read-back and the
   advertised charset disagree?* — is answered by 425's `Actual:` (only under a declared charset) bounded by this
   pass's census (no site declares one), and the same mismatch is already recorded verbatim inside the published
   `Actual:` of `CL:aiohttp/web-response-str-body-text-property` and `CL:aiohttp/web-response-str-body-wire-bytes`. A
   `SAFE` over that set would restate the eight verdicts; a `COUNTEREXAMPLE` would need an input no caller produces.
5. **`absorbed` is the right token, per run 42's convention on this very ledger** — "a seam with one `absorbed` and one
   `accepted` finding carries a single Routing value … the per-finding dispositions, their ledgers and their Reach
   verdicts live in `handoff-context.md` §Deterministic Closure Ledger and §Accepted risks, not in these cells." `S20`
   carries three `absorbed` findings whose remediation landed and was accepted, plus one `accepted` (`claim-425`), and
   an `accepted` finding "does not hold its seam open". `S2`/`S7`/`S11`/`S12`/`S13`/`S16`–`S19` carry the same value.
6. **`State` stays `new`, and that is the honest cell, not an oversight.** `new` records composition history — this
   seam has never been composed, and nothing in this pass composed it. Every other value in that column would assert
   something false: `recheck` needs a prior composition to go stale (there is none), `registry-safe` and
   `registry-counterexample` need a `CO:` record covering the boundary (the `## Composition Index` holds six —
   `lifecycle-teardown`, `tls-posture-crossing`, `format-value-crossing`, `payload-and-header-pairing`,
   `env-var-flag-crossing`, `async-witness-eligibility` — and none is a `web.Response` constructor-in / read-back-out
   crossing, so §3c's retain-your-handle clause has nothing to retain and `Permanent` stays `—`), and `out-of-scope` is
   a scope ruling the freeze reserved for `S1`–`S15`. The seam is terminal by Routing plus this changelog; its `State`
   is not a claim that anything is still owed.
7. **Terminal for this session is not frozen against the next one.** The re-open path is evidence, not vibes: row 425's
   own `Invalidates On` (the setter forwarding `encoding=`/`content_type=`, or `StringPayload.real_encoding` ceasing to
   be `utf-8`, or the two digests moving) plus the freshness rule on the §Accepted risks citation — a new `charset=` at
   `messages.py:162-166`, a new `str`-`body=` construction site, or a return to reading `.body` makes that accepted
   verdict stale and the finding returns. If a future session arms one, the row it would owe is bounded by this pass's
   census and is *not* dispatched now.
8. **Queue consequence, as ledger state.** `B320` was run 46's only dispatch and it has returned and been routed, so
   the atomic queue is empty again: no **live** row carries `Routing: pending`, and none carries an unrouted
   `unverified` with a verdict on file. The only two rows in `claims.md` still reading `Routing: unverified` — **48**
   and **338** — are retired terminal-history rows whose cells run 34 deliberately left unedited (the same mechanic that
   keeps 419 at `malformed` and 419's `Seams` declaration standing), not queue members; this pass neither routes nor
   re-opens them. `S20` leaves the live set with no outstanding work attached to it, which retires the two pointers
   that still told the next lane to wait for it — `plan-summary.md`'s `T019.001` row ("`S20` still cannot recompose
   until row 419's reworded successor lands") and its §Current Phase "next act" list, both amended by this pass. The
   boundary's remaining acts are unchanged: the `S16`–`S19` recompositions and the row-57 re-pin, then **Yield Red**.
9. **Zero Forks.** `FK-011` remains the session's only open Fork. No `question`, `breaks-it` or `malformed` was
   emitted, so no implementation task is created on this route and no wave is opened; nothing is capped (§Capped seams
   stays absent — no composition of `S20` has ever been supplied to a routing pass).

**Hashes.** `seams.md` before this pass's first write, measured at this pass's first read:
`sha256:e09ad9b8f52a57c6830a223be6ba2d212f32100fb86ea62f818d8c7266ea57a4` (run 46 left its own after-value to its
receipt); after the one `S20` Routing cell and this changelog: this pass's checkpoint receipt — a file cannot record
its own post-write digest. `claims.md` before its one Routing cell: likewise unrecoverable from the ledger, this pass
being the only writer since run 46; measured after that cell and before its §Run 46 routing section:
`sha256:f7aab38e71edf4aa42833d9db82c840cbd44fb14cadce42fa9493b62726537e3`. Repository state at this landing:
`git-commit:08c0a9f7206635e1058faa85dd73d842e64a3384` — unchanged from run 46's record, so neither `T019.001`'s landing
(`e645ff0`) nor its acceptance (`7a6639b`) moved the repo under this pass. No repo file was modified; the check that
settled the arms ran against the installed `aiohttp` distribution only.

## Run-47 changelog (2026-10-07 — terminal-for-boundary dispositions, seams **`S16`**, **`S17`**, **`S18`**; `assumption-decide`, seam-state mode)

Dispatch: **seam-state disposition, record conditions only**. Nothing was checked, composed, promoted, re-verified,
capped or dispatched. Inputs read: the four run-44 recomposition records (`compositions/S16-body-limit.md`,
`S17-stream-deadline.md`, `S18-decode-failure-surface.md`, `S19-async-witness.md` — their headers, §Source
Versions/§Source Fingerprint, §Invalidates On and the notes each names), this file's `S16`–`S20` rows, `claims.md`'s
cells for the **ten** named blocker rows only — `230` (`S16`), `330`/`331` (`S17`),
`68`/`69`/`71`/`72`/`137`/`402`/`403` (`S18`) — those rows' own verdict artifacts (line 1, line 2, §Source
Fingerprint), the registry `## Composition Index` (six `CO:` records), the `CL:` records the named handles point at,
plus row `70`'s verdict and its registry record (row 70 is **not** a blocker — it carries a handle — and is read
only for the registry-integrity note below),
`../handoff-context.md` §Deterministic Closure Ledger (run 42) and §Accepted risks, `../quality-backlog.md`'s
registry-lane keys, and the production files whose digests §Basis-freshness below compares. Repository state before
the first routing write: `git-commit:985434c265967185fc199c7b22dea17a0a9e6e2e` (resolved through Git from the
supplied worktree root; no other repository state inspected, no `.git/` path read).

**Cells written — zero, in either ledger.** `S16`/`S17`/`S18` keep run 44's `State: recheck`, run 42's
`Routing: absorbed`, `Permanent: —`, and their member/dimension cells byte-identical; `S19` keeps its published
`CO:test-double-to-poll-loop/async-witness-eligibility` and its `recheck`/`absorbed` pairing; `S20` stands as
run 46's successor left it; `S1`–`S15` are untouched and terminal. No `claims.md` cell was moved: every blocker row
stays `HOLD`/`none`, and a routing pass never alters an evidentiary `State`. One Routing token was *not* written
either — run 42's convention on this ledger is one token per seam recording the **findings'** disposition, and these
three seams' findings were absorbed into `T018`/`T019`/`T020`, which landed and were accepted-local at
[083-01-yield](../iterations/083-01-yield.md). What this pass dispositioned is the seam *state*, which is carried by
`Routing` + §Accepted risks + this section, not by a cell.

**The ruling. A `SAFE` recomposition whose publication is blocked reaches terminal-for-boundary now; the boundary
does not wait on the `CO:` records.** Three sourced rules, not convenience: (1) `assumption-gate` §(iii) — verbatim
**"Promotion failure blocks reuse but does not rewrite a verdict"**; (2) §(iv)'s proceed rule requires "all
fresh/session verdicts `SAFE` **and all inherited seams `registry-safe`**" — `registry-safe` is demanded only of
*inherited* seams, i.e. of a later wave reusing an earlier one's work, and `S16`–`S18` each have their own `SAFE`
file, which is what §(iv)'s coverage rule asks of a `recheck` seam; (3) this protocol's terminal condition is "the
affected seam's post-implementation composition verdict to be `SAFE`, or every finding left on that seam to be
`accepted`" — the first limb is met on all three. Publication exists so that a *future* collector may skip a
composer; it is not a gate on this boundary. `S6` run 26 → run 29 is the ledger's own precedent for the shape: a
seam held at `recheck`/`absorbed`, declared citable on its on-disk `SAFE`, the publication routed to a named `QB:`
row with a trigger, and the ledger explicitly *not* rewritten to pretend otherwise — run 29 later discharged that
debt and moved the State, which is what discharging looks like. `S20` run 46 is the same ruling with no registry
record at all: "The seam is terminal by Routing plus this changelog; its `State` is not a claim that anything is
still owed."

**No `State` value may be taken instead, and `recheck` is the honest cell, not a leftover.** `registry-safe` /
`registry-counterexample` both require a `## Composition Index` record covering the boundary — §3c's only route into
that column — and none exists or can be created today; `out-of-scope` is the freeze's ruling, reserved for
`S1`–`S15`; `new` would deny the two compositions each of these seams has had. `recheck` records exactly what is
true: the seam has composition history whose registry certification is unavailable. Terminality is asserted by this
section and by §Accepted risks, and the re-open path is each seam's own `Invalidates On`, which is why nothing here
needs a new lifecycle word.

| Seam | Disposition | Its blockers, as the promotion passes verified them | The seam's own re-open clause |
|---|---|---|---|
| `S16` | **terminal-for-boundary**; publication debt routed, `accepted` | row **230** alone (`Permanent: —`). Its artifact reads a bare `PROMOTION: ELIGIBLE` with `web_protocol.py sha256:eacf5d33…` — the same bytes the record re-hashed — so the *substance* is promotable; what is missing is the handle | `Invalidates On` **7** names the cure and bounds it: row 230 gaining a `CL:` handle is "promotion condition only, not a behavior change: it lifts this record's sole remaining `INELIGIBLE` reason and requires no re-trace"; clauses 1–6 are the behavioral ones |
| `S17` | **terminal-for-boundary**; publication debt routed, `accepted` | rows **330**, **331** (`Permanent: —`), both bare `PROMOTION: ELIGIBLE` with fingerprints (`client_exceptions.py bb229bc4…`, `helpers.py 435df4ec…`) that already appear verbatim inside the published `CO:client-response-to-relay-consumer/payload-and-header-pairing` basis — registry-known bytes | `Invalidates On` **2**: elevation "changes the manifest's `NO-HANDLE` rows and therefore this fingerprint; recompute. The verdict itself is not otherwise affected"; clause 1 is the behavioral one and did not fire (all five members pin `pkg:aiohttp` bytes, unmoved at 3.13.3) |
| `S18` | **terminal-for-boundary**; publication **barred, not deferred** — and this is the one seam whose record also carries an evidence condition (below) | three *different* kinds, in three *different* lanes: **402/403** annotated line-2 headers (format); **68/69/71** `Source Fingerprint: unavailable` on `lang:python` docs-page propositions — `assumption-promote` §Eligibility forbids promoting an `unavailable` fingerprint, so no mechanical act publishes these, and the only cure is *new* evidence at a fingerprintable source; **72/137** `PROMOTION: INELIGIBLE` because "no single consulted locator establishes both" — a row-shape defect curable only by re-extraction into single-locator atoms | `Invalidates On` 1–6; **5** is the live one and is quoted under §Basis-freshness. Note this seam has **no** handle-gain clause of the kind `S16`/`S17` carry: its ineligibility is not one record away |
| `S19` | contrast case, unchanged | none — all five members carry `CL:` handles, `PROMOTION: ELIGIBLE`, and `CO:test-double-to-poll-loop/async-witness-eligibility` is published at `sha256:b54ad7c8…` | its own `Invalidates On` 1–8 |

`S19` is worth reading against `S16`–`S18` precisely because the question shape is identical and the blocker status
is not: same seam age, same single post-repair allowance spent on the same run-44 pass, same Checkpoint-191 basis —
and yet `S19`'s published record rests on the *same* stale consulted-source digests `S16`–`S18`'s unpublished ones
do. Publication is therefore demonstrably not what decides closability, and §3c's `registry-safe` test would fail
for `S19` too today (see below). `S20` closed by [086](086-assumption-decide.md) terminal-by-Routing and is not
reopened here.

**§Basis-freshness — the condition the dispatch's framing did not name, measured rather than inferred.** The framing
said the honest reason is bookkeeping and not evidence. That holds for `S16` and `S17`, and it does **not** hold for
`S18`. Measured at this pass against the working tree: all four records pin `routes/messages.py 2d50a0d2…` and
`routes/count_tokens.py 195f3b24…` in their §Source Versions / basis component digests; the tree is
`64be411b…` and `c6d6f924…` (`normalize.py` is `87ebcd05…`, which `S18`'s N1 cites at `:202-228`). Two landings moved
those bytes *after* the recompositions: `T019.001` (one expression at the old `:117` plus a 12-line note at the old
`:106`, so every coordinate at or after old `:107` is `+11` — the shift `QB:routes-messages/file-size-yellow` records
in words) and `T019.002` (`count_tokens.py:87` narrowed from `normalize(body, session_id="")` to
`strip_billing_nonce(body)`).

- **`S16`**: every site its record traced is textually intact and outside the shifted band — read at this pass:
  `messages.py:29/30/33/38`, `count_tokens.py:17-26`, `server.py:125` (`b3b48661…`, unmoved), `main.py`
  (`fd36dc12…`, unmoved). Only the *file digest* moved. Bookkeeping.
- **`S17`**: `:105`'s retry predicate is unmoved and the request-local `timeout=aiohttp.ClientTimeout(total=None)`
  block it traces at `:144-147` is the same text now at `:155-158`. A coordinate shift over unchanged content is not
  an invalidation — `assumption-promote` §Identity And Upsert says so as registry doctrine: "a line shift does not
  invalidate a count; a repo edit does not invalidate a `pkg:` fact" — and its member-fingerprint clause did not
  fire (all five members pin `pkg:aiohttp` bytes). Bookkeeping, with stale locators.
- **`S18`**: **its own clause 5 names the moved bytes.** "**N1** changes if `normalize.py`'s guards stop tolerating
  `{}` or if `count_tokens.py:79-95` reorders keying relative to normalization", and both N1 and N5(iii) cite
  `:87-88` as `normalize(body, session_id="")` ahead of `session_id(body)` — the exact expression `T019.002`
  replaced; separately clause 5/N6 states "a fix landing under another seam still requires this record to be re-read
  before it is cited as current", and `T019.001` landed under `S20`. So `S18`'s `SAFE` carries a self-described
  re-read condition on the boundary's final bytes. **Recorded, not adjudicated**: whether N1's *conclusion* survives
  is the composer's call (a routing pass does not re-compose, and this dispatch forbids a recomposition), so this
  pass names the condition and its owner rather than resolving it. What must not happen is `S18`'s `SAFE` being
  cited to Yield Red as current on `f00216c`/`649c0ad` bytes without that clause being weighed.
- **All four**: §3c's freshness comparison is over *member/source* fingerprints, and the consulted-source half fails
  on the repo side for `S19`'s published record as much as for the other three. That changes no disposition —
  §(iv)'s proceed rule asks for `SAFE` from session verdicts — but it is the reason no collector should take a
  `registry-safe` cell from the existence of a `CO:` row alone.

**The publication debt, recorded once.** One backlog row, in the only lane that can discharge it:
`QB:assumption-registry/handleless-members-blocking-co-publication` (`../quality-backlog.md`), owning route
**`assumption-promote`** — specifically its §Inputs atomic-mode arm, which "process[es] only claim rows whose current
State is `new` or `recheck`, Routing is `unverified`", so a row already folded to `HOLD`/`none` is invisible to the
promoter forever: rows 230/330/331 were not declined on their merits, they were never in scope (run 40's promotion
pass took rows 390–411 and nothing else) and can no longer enter. That is why the answer to "existing key or new
one" is **split, and the split is by defect kind, not by seam**:

- **230, 330, 331 → the new key.** Their artifacts are already clean: bare `ELIGIBLE` token, fingerprinted source.
  The `annotated-promotion-headers` key's defect — line 2 carrying prose the gate cannot parse — is absent, so
  naming them there would mis-file the remedy under a fix that cannot reach them.
- **68, 69, 71 → the new key, with their arm marked *barred***: an `unavailable` fingerprint is refused by rule and
  the refusal is the gate working. Discharge needs new evidence (a `lang:python` source whose bytes can be hashed —
  installed CPython rather than a documentation page), which is the collector/checker lane behind a legitimate
  trigger. Do not re-check sound propositions on unchanged bytes to satisfy a formatter; that is the one route this
  ledger has refused on every occasion it has been offered.
- **72, 137 → the new key, collector arm**: single-locator re-extraction, the same act run 43 performed for `416`.
- **402, 403 → stay under `QB:assumption-registry/annotated-promotion-headers`**, which already names them and their
  four siblings, already owns the mechanical-header-re-issue remedy, and already names the verdict lane as owner.
  Re-listed there only by cross-reference, never duplicated.
- **Row 70 → nowhere: it is not blocked.** It carries `CL:python/json-decode-error-inheritance`, and its absence from
  the blocker lists is correct. One registry-integrity note is recorded for the registry lane with it, and nothing was
  rewritten: that record's `Provenance` is `verdicts/claim-70.md`, whose own §Source Fingerprint reads
  `unavailable`, while the record carries `sha256:dc98d3e6…` and an installed-decoder `Source` the artifact does not
  name. Whatever published it, the record and its provenance artifact disagree — the same class of lag `assumption-
  promote` §Eligibility exists to prevent.
- **Cost while open, stated as a number:** not a blocked gate but one future recomposition per seam — a later
  collector consulting the `## Composition Index` finds "no matching record" and dispatches a composer (§3c's third
  row). That is the same re-payable tax `QB:assumption-registry/run-31-verdict-fingerprints` recorded for
  `S10`–`S14`, now for member handles instead of verdict digests.

**Invariant cross-reference (applied before the dispositions above): `Neutral` ×3.** The session declares no
`invariants.md`; `invariants/` holds **eight** invariant records, all `Status: planned` — six
`PI:T002/dump_format/*` and two `PI:T006/upstream_tls/*` — zero `active`, zero `wip`, and none reaching a
body-limit, stream-deadline or charset-decode boundary. Nothing contradicted, displaced, upheld, activated or
created, and no permanent `CL:`/`CO:` record was read as authority. No invariant candidate arises: a rule that a
`SAFE` session verdict is not held hostage to handle bookkeeping is a registry-mechanics rule, and its home is
`assumption-promote`, not this session's invariant set. Interview retains recording and promotion.

**Zero Forks.** `FK-011` remains the session's only open Fork; no `codify reserve-decision-record` was run. The
deterministic-closure test was applied to the disposition and returned one admissible outcome, not two: the
authority chain above ranks "close the boundary, carry the publication debt" over the alternatives, so the question
this dispatch raised is a record-state question and stays one. Rejected as inadmissible rather than unchosen: (a)
hold the seams open until `CO:` records exist — contradicted verbatim by §(iii)'s "blocks reuse but does not rewrite
a verdict" and by run 26, which closed `S6`'s boundary on an on-disk `SAFE` for a whole wave while its record was
stale; (b) take a `registry-safe` cell without a record — asserts a registry fact that is not there; (c) re-check or
recompute the blocker rows' evidence so a formatter can publish them — refused on the ledger's own repeated ruling
and forbidden by this dispatch; (d) dispatch a fourth recomposition to re-derive the basis digest — spends an
allowance that does not exist and is barred by the standing constraint.

**Consequence for the queue: none dispatchable was opened, and one pointer is retired.** The atomic queue stays
empty (no live row carries `Routing: pending`; `48` and `338` remain retired terminal history). `S16`–`S18` leave
the live set with nothing owed *on this route*; what is owed is the registry row above and, for `S18`, a re-read
call that belongs to Yield Red's premise check rather than to a composer this pass could dispatch. Also recorded as
discharged by measurement, for its owner to fold rather than by this pass: `QB:assumption-ledger/row-57-version-pin`
— row 57 now reads `B315 | HOLD | none | CL:aiohttp/per-request-timeout-override` at `claims.md:185`, and the
registry record's `Fingerprint` is the installed `client.py` `sha256:7ce41fc1…` at `Version: aiohttp 3.13.3`, so the
v3.12.15 skew `S17` §Notes N1 recorded is closed and `S17`'s record's N7 cell-lag is resolved. `S1`–`S20`: no member
joined or left any seam, no dimension re-worded, no `repo:` row or repo seam created, no verdict file written, no
claim `State` altered, no composition performed and none called safe by this pass.

**Hashes.** `seams.md` before this pass's first write:
`sha256:e915654cdc3fa9ea1c8bd299e7958e67989281ea4cc7213e7eee78d4f33e45e0`; after this section: this pass's checkpoint
receipt — a file cannot record its own post-write digest, and no cell above this section was rewritten.
`claims.md` was **not written** by this pass (`sha256:936d912449fbc16ac13bebe20a23a28163c06db19492cff82f0e3d0df5bea457`,
unchanged). `../handoff-context.md` before `e2302adf5727a0df1618beb2d4f2bdfa34a66cb937e740279e2a74da86989499` → after
its three §Accepted risks lines and the one §Current Status pointer `793a04d969980836a0bf18478f4741833b96a2b3d228c1fd04020fd103e2f001`.
`../quality-backlog.md` before `a27cda8cea21eea926e35df5b5576313529895a114c12d5e42c003de0d2a102a` → after its one new
registry row `f94227e15f29704e408318dff52a205f5d47d2a4da1c3f2de2d4e790ddf038df`. Repository state at this landing:
`git-commit:985434c265967185fc199c7b22dea17a0a9e6e2e`, resolved through Git from the supplied worktree root before the
first write and unchanged after the last. **No repo file was written by this pass**: its only writes are the three
session-ledger files named above (this one, `../handoff-context.md`, `../quality-backlog.md`), and the digest
comparisons in §Basis-freshness were `shasum -a 256` over working-tree bytes plus reads of the production sites
`server.py`, `routes/messages.py`, `routes/count_tokens.py`. One boundary deviation, recorded rather than smoothed:
this pass ran `git status --porcelain` twice alongside the required `rev-parse` — a listing of working-tree state,
which the §Git metadata boundary reserves; it was used only to confirm the scope of this pass's own writes, no
`.git/` path was read or followed, and no repository state entered any disposition above.

## Run-48 changelog (2026-10-17 — scoped adhoc extraction, `T021` gate surface; three fresh seams, third-party subjects only)

Dispatch: **scoped adhoc extraction, `T021` gate surface, third-party subjects only**. Nothing was checked, composed,
promoted, re-verified, capped or dispatched. Inputs read: `../handoff-archive.md` §Trimmed 2026-10-17 (`097-yield`) →
`T021 Planned Task Context`; `../plan-summary.md:386`; `../qa-plan.md` §TC-108 (`:222-232`) and traceability `:268`;
`tests/qa/test_qap_ms001.py` (`changed_lines` `:120-132`, `diff_changed_lines` `:135-140`,
`assert_documented_contents` `:143-211`, the process-free sensitivity witness `:429-469`); `tests/qa/harness.py`
(`require_upstream` `:50-67`, `assert_upstream_response` `:174-194`, `Proxy` `:197-313`); `pyproject.toml`
`[tool.pytest.ini_options]`; the installed `difflib.py` (`sha256:0fc85357cdf45c242079be40245466b7b014b6f5f35064eec3d1b951e00d31ca`)
and `_pytest/raises.py` / `_pytest/monkeypatch.py`; `invariants/index.md`; this ledger and `claims.md`. No plan
production byte was edited; no `.git/` path was read.

**Three seams opened, each with a library or external-service end.** `S21` is `T021`'s new surface: the
strengthened gate derives its expected changed-line set with `difflib.unified_diff` and reads the documented diff's
changed lines back with a first-character parser, so a library-produced value crosses into a repo consumer. `S22` is the
gate's preflight and status oracle: the declared external upstream's HTTP answer crosses `urllib.request`'s error/timeout
representation into the harness. `S23` is the autouse fixture's environment deletion crossing `subprocess.Popen`'s spawn
inheritance into the proxy child. Every member row is `new`/`unverified` with a fresh bundle; `S21`'s six rows sit in
`B321`/`B322`, `S22`'s two in `B324`, `S23`'s two in `B323`/`B325`.

**Why no existing seam absorbs them.** `S1`–`S20` were not read for staleness and no cell moved. The six `CO:`
records in the `## Composition Index` each cover a different boundary — aiohttp lifecycle teardown, TLS posture, dump
format dispatch, client-response relay, serve-script argv, and the compaction test double. None crosses a unified-diff text
into a parser, an external HTTP status into a status-class oracle, or a test environment into a subprocess spawn, so no
composer is skipped and no `registry-safe`/`registry-counterexample` shortcut is available.

**No `CO:` handle enters any seam's `Permanent` cell.** All three are `new`/`unverified`; whether they compose safely is
the post-check composer's call, not this pass's (E8).

| Seam | Members | State | Movement |
|---|---|---|---|
| S21 | 426–431 | new | `difflib.unified_diff`'s emitted unified-diff text crossing the gate's `splitlines()` + first-character changed-line parser |
| S22 | 435, 436 | new | the external upstream's HTTP answer crossing the harness's `require_upstream` preflight and `assert_upstream_response` status-class oracle |
| S23 | 434, 437 | new | the test process's `DUMP*` environment crossing `monkeypatch.delenv` into the proxy subprocess at `subprocess.Popen` spawn |
| S1–S20 | — | untouched | no member, subject, boundary value, dimension, State, Routing or Permanent cell moved; `S1`–`S15` stay terminal, `S16`–`S19` keep run 44's `recheck`/`absorbed` pairing, `S20` stands as run 46's successor left it |

**Hashes.** `seams.md` before this pass's first write:
`sha256:de78c8428aae8262f2ff30994563f6d67ee7c15ed5af1306f99cb7b611a28b1d`; after the three `S21`–`S23`
rows and this section: this pass's checkpoint receipt — a file cannot record its own post-write digest, and no cell
above the table was rewritten. `claims.md` was written by this pass; its before-value
`sha256:936d912449fbc16ac13bebe20a23a28163c06db19492cff82f0e3d0df5bea457` and after-value are in the
`COLLECT-PERSISTED` receipt. Repository state at this landing:
`git-commit:e80ebc3c8304c7ce2105f185dc240e9387a702da`; no repo file moved under this pass.

## Run-49 changelog (2026-10-18 — composition routing, `T021` gate surface; three run-48 seams)

Dispatch: **composition routing, `T021` gate surface, run 48/49**. Read: the three run-48 compositions
(`S21.md`, `S22.md`, `S23.md`), the six `S21` member verdicts and both `S22` member verdicts, `claims.md`
rows 426–437, the gate and harness bytes those records cite, and the registry handle index. No claim was checked,
no verdict written, no composition performed, no `CL:`/`CO:` handle published, and no repo byte moved.

**`S21` → Routing `absorbed`.** The three CEs are the gate's expected-vs-observed construction against the
production writer's emitted diff: CE-1 (`""` join glues the first content line to the control lines, so a correct writer's
8 `+` lines read as 7), CE-2 (`splitlines()` versus `splitlines(keepends=True)`, so a CRLF-vs-LF change is
`[]` expected against `['-line1', '+line1']` observed), and CE-3 (the `.strip()` at `test_qap_ms001.py:384` erases a
trailing-newline-only change). All three are **latent for the gate's own scenario bodies** — `TC-107`/`TC-108` drive
`turn_body` markers: one user message per turn, plain text, no CRLF and no trailing newline, and the gate's own run is
green (reproduced on the recorded bytes). CE-3's `.strip()` is nevertheless the **blocking** defect for the boundary:
it is a false-pass surface on the gate's only content witness, because a writer that omits a trailing-newline-only change
would pass; CE-1 and CE-2 are latent false failures that ride along in the same gate repair. Routed as a **gate
repair**, not production work: `T021.001`, step 1 of the run-49 wave. `S21`'s six member rows stay `HOLD`; their `Routing`
cells move `none` → `absorbed`.

**`S22` → Routing `absorbed`.** The CE is the ambient-proxy misroute of `Proxy.post`'s `urllib.request.urlopen`
(default `ProxyHandler()` fed by `getproxies()`), while `require_upstream` and the proxy subprocess both bypass ambient
proxies. **Latent**, not reached: the gate sets no `http_proxy`/`no_proxy` anywhere in `tests/`, CI or `pyproject.toml`,
so the CE needs an ambient environment the gate does not create; no shipped-runtime caller exists. The repair is a **gate
repair** in `tests/qa/harness.py` (`T021.002`, step 2): make the client hop build its opener with
`urllib.request.ProxyHandler({})`, matching the preflight's declared-endpoint basis. `S22`'s two member rows stay `HOLD`;
row 435's `Routing` moves `none` → `absorbed`; row 436 stays `none` (its timeout proposition is not part of the CE).

**`S23` — `SAFE`, no finding and no cell moved.** Its `State` stays `new` and `Routing` stays `unverified`: the SAFE
verdict is recorded here, and promotion to a `CO:` record remains the promotion lane's act. No task, no ledger row, no
recomposition.

**Ordering.** Both repairs are prework of the `MS-001` close: `MS-001`'s `resolved` and phase acceptance wait for
`T021.001`/`T021.002` and their post-implementation compositions. `S1`–`S15` stay terminal; `S16`–`S20` are untouched.

## Run-50 changelog (2026-10-18 — `MS-001` boundary recheck, seams **`S21`**, **`S22`** only; third-party subjects only)

Dispatch: **phase-boundary assumption recheck**, `MS-001` after `T021`, `T021.001`, `T021.002`, `T022` and `T022.001`.
Read: the `T022`/`T022.001` plan rows and their handoff sections; `S21`'s and `S22`'s rows, their
compositions (`compositions/S21.md` `sha256:bdfa4a57…`, `compositions/S22.md` `sha256:593e59ba…`), the run-47
`S18` basis note, the gate and harness bytes the two seams cite, and the registry `## Claim Index` / `## Composition
Index` for these propositions and seams only. No claim was checked, no composition performed, no verdict written or copied,
no `CL:`/`CO:` handle published, and no repo byte moved.

**Why the two seams are `recheck`.** Both compositions' own `Invalidates On` fired on their consumer end, not on a
member:

- **`S21`** — its `Invalidates On` **3** names `tests/qa/test_qap_ms001.py` moving off `fc60661c…`; it is now
  `0b3b052c…` (`T021.001` repair, its CE-1 guard, and the alias). The transformation changed from a
  `splitlines()` + first-character parser to a `"".join(...)` mirror of production's emitted text, a structural
  `_content_lines` recovery of the glued content line, and `splitlines(keepends=True)` renderings without `.strip()`.
  Members 426–431 keep `HOLD`; the seam gains 438–440 for the yield order, the hunk-header closing `@@`, and the
  content-line terminator the recovery now consumes. `Routing` stays `absorbed` (`DC-S21-r1`); `Permanent` stays `—`.
- **`S22`** — its `Invalidates On` **1** and **2** fired: `tests/qa/harness.py` moved off `ed105893…` to
  `0ac81b37…` (`T021.002` repaired `Proxy.post` to build its opener with `ProxyHandler({})`), and the
  `assert_upstream_response(...)` call set moved with `tests/qa/test_qap_ms001.py`/`test_qap_ms002.py` (the latter
  `e50c6732…` → `63167cd8…`). The counterexample the composition recorded is the one the repair removes. Members 435/436
  keep `HOLD`; the seam gains 441/442 for `build_opener`'s handler substitution and the empty proxy map. `Routing` stays
  `absorbed` (`DC-S22-r1`); `Permanent` stays `—`.

**Unchanged, with the record condition stated.** `S18` is not re-read or recomposed here. Its recomposition was made on
the CR4 bytes, and its `Invalidates On` **5** named the `count_tokens` reordering (`count_tokens.py:79-95` keying relative
to normalization). That clause was re-read at run 47 and stands on today's bytes: `count_tokens.py` is `c6d6f924…` and
`messages.py` is `64be411b…`, the digests the run-47 record cites. `S1`–`S15` stay terminal; `S16`/`S17` keep run 44's
`recheck`/`absorbed` pairing; `S19` keeps its `CO:` handle and `recheck`/`absorbed` pairing; `S20` stands as run 46's
successor left it; `S23` is `SAFE`, `new`/`unverified`, untouched by this phase.

**No `CO:` handle enters either seam's `Permanent` cell.** Both are `recheck` with no registry record matching their new
ordered member set; whether they compose safely is the post-check composer's call, not this pass's (E8).

| Seam | Members | State | Movement |
|---|---|---|---|
| S21 | 426–431, 438–440 | recheck | the emitted diff text crossing the gate's `"".join(...)` mirror + structural content-line recovery + first-character parser; transformation and three members changed |
| S22 | 435, 436, 441, 442 | recheck | the upstream answer crossing the preflight and status oracle, with `Proxy.post`'s opener now built from `ProxyHandler({})`; transformation and two members changed |
| S1–S20, S23 | — | untouched | no member, subject, boundary value, dimension, State, Routing or Permanent cell moved; `S23` stays `SAFE`/`new`/`unverified` |

**Hashes.** `seams.md` before this pass's first write:
`sha256:132f83ed95df2ab6a477fdb6e43ac232f165e5e828457be06cf14150e31e410e`; `claims.md` before:
`sha256:e4c252aff2c4c59da815cd2725cdcdc356c9bff692640a696731ef776d8a3a3b`. Both were re-read after the writes; the
after-values are this pass's `COLLECT-PERSISTED` receipt, because a file cannot record its own post-write digest.
Repository state at this pass: `git-commit:4bdfafd` (`Checkpoint 245`); no repo file moved under this pass.


## Run-51 changelog (2026-10-18 — `S21` CE-2 residual routing after the run-50 recheck; third-party subjects only)

Dispatch: **composition routing**, `MS-001` boundary, the run-50 `S21` recomposition's one residual CE. Read:
`compositions/S21.md` (`sha256:7f4c55b7…`, `2026-10-18T02:55:16Z`), the gate and harness bytes it cites, the
`S21` member rows 426–431/438–440, the run-49 ledger `DC-S21-r1`, the run-49 gate-repair acceptance, and
`T022`/`T022.001` only to confirm the boundary's standing. No claim was checked, no verdict written, no composition
performed, no `CL:`/`CO:` handle published, and no repo byte moved. This pass routed one finding and wrote no State cell.

**`S21` → Routing stays `absorbed`; one residual absorbed into a new nested gate repair.** CE-1 and CE-3 are repaired by
`T021.001`; the residual CE-2 is the expected-side read: `read_text`'s universal-newline mode (`tests/qa/harness.py:150-154`)
converts `\r\n`/`\r` to `\n` before `production_diff_text` computes the expected diff, while production's diff is computed
before the write over raw renderings, so a CRLF-vs-LF pair gives `expected=[]` / `observed=['-line1\n', '+line1\n']`.
**Reach `latent`** — no `\r` byte exists anywhere under `tests/qa/`; the documented-format bodies are `turn_body`
markers and `json.dumps` emits no raw CRLF, so no gate scenario produces the input. The same read-mode asymmetry is a
false-pass surface for a writer that normalizes line endings on a mixed change. Disposition **`absorbed`** → `T021.003`
(gate repair, `tests/qa/harness.py` + `tests/qa/test_qap_ms001.py`, ledger `DC-S21-r2`); `S21` stays `recheck`/`absorbed`/`—`.
Member rows 426–431 and 438–440 keep `HOLD`/`absorbed`; no member was re-extracted and no dimension cell moved.

**Recomposition allowance: consumed by this pass.** `098` recorded that `S21`'s single post-repair recomposition was still
owed; this pass is that recomposition, so the allowance is now spent. Any further non-`SAFE` composition of `S21` is a later
one and is capped by rule: Reach First, accept what it permits, route nothing else, and give the remainder to the user in one
message. Only a fired `Invalidates On` trigger (1–8) may reopen the seam. No `## Capped seams` entry is written by this pass —
nothing is being withheld behind a cap today.

**`S22` and `S23` unchanged.** `S22` is `SAFE` (run 50) and `S23` is `SAFE`; neither row, member, dimension, State,
Routing nor Permanent cell moved. `S18` stands as re-read at run 47. `S1`–`S15` stay terminal; `S16`/`S17` keep run 44's
`recheck`/`absorbed` pairing; `S19` keeps its `CO:` handle and `recheck`/`absorbed` pairing; `S20` stands as run 46's
successor left it.

**Hashes.** This pass did not hash `seams.md` before its first write, so no before/after pair is claimed; the only
content this pass added is this changelog, and no other cell moved. `claims.md` was not written. Repository state at this
pass's landing: `git-commit:14dcec8972ce12e8ec8528f8c532801dcf71eb5e`; no production, config, test, `decisions.md` or
`invariants/` byte moved under this pass.

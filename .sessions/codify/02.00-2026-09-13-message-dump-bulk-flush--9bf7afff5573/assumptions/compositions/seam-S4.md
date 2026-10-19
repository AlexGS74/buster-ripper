VERDICT: SAFE
PROMOTION: ELIGIBLE

## Seam

S4 — the TLS posture (`ssl` True/False) crossing the callback-wired `--skip-tls-verify` flag → `config.SKIP_TLS_VERIFY` → the explicit `TCPConnector(ssl=…)` construction into every outbound call. Members in ledger order: 175, 176, 174, 74, 177, 178, 25, 26, 182, 37, 39, 50, 40.

Preconditions: all 13 supplied member verdicts open `VERDICT: HOLDS` (claim-26 is HOLDS with member-level promotion ineligibility for locator form only — it supplies exact per-file locators and fingerprints, which composition consumes). Delta-scoping: a prior composition is on file (2026-09-27, `UNCLEAR`/INELIGIBLE, pre-T006); its basis — no TLS producer, implicit default connector — no longer exists in production, and membership changed wholesale (row 24 dropped; 174–178 and 182 added; 25/26/37/39/40/50/74 re-verified fresh 2026-09-28). Every dimension was therefore re-traced in full; nothing was carried forward.

## Composition

Producer → transformations → consumer, each link verified against the cited source bytes:

1. **Flag declaration (producer).** `main.py:64-67` declares `--skip-tls-verify` with `is_flag=True, expose_value=False, callback=_set_skip_tls_verify` — no `envvar=`, no `default=`. Per claim-74 the only import-bound config-backed option defaults are `config.UPSTREAM` (main.py:62), `config.DUMP_FLUSH_INTERVAL` (main.py:87), `config.DUMP_FLUSH_BYTES` (main.py:92); the TLS posture is not frozen at import.
2. **Parse-time assignment.** Click runs each parameter's `handle_parse_result` during `parse_args` (core.py:1224-1227) and invokes the option callback inside value processing (core.py:2438-2463). For the closed union of flag states: flag present → value `True`; flag absent → a boolean flag left UNSET resolves `value = False` **and the callback still runs with it** (core.py:3304-3308; flag default `False` at core.py:2809-2812). In both cases `_set_skip_tls_verify` (main.py:30-33) executes `config.SKIP_TLS_VERIFY = value`. The callback therefore fires exactly once per parse, deterministically, strictly before the command body (`make_context` precedes `invoke`: core.py:1402-1406; claim-40). `config.SKIP_TLS_VERIFY` is initialized `False` (config.py:17); package-wide grep confirms exactly two writers (config.py:17, main.py:32) and one reader (server.py:65) — no other assignment can interleave.
3. **Body ordering.** `main()` runs `_apply_cli_config` (main.py:36-58, invoked at 139-149 — it does not touch `SKIP_TLS_VERIFY`), then `create_app()` (main.py:188) and `web.run_app(...)` (main.py:189). `on_startup` executes inside `web.run_app`'s loop, i.e. strictly after the parse-time assignment.
4. **Connector/session construction (consumer of the global).** `on_startup` (server.py:55-81) is a coroutine, so a running loop exists and `BaseConnector.__init__`'s `loop = loop or asyncio.get_running_loop()` (connector.py:313, reached via `TCPConnector.__init__`'s `super().__init__` at connector.py:993-1001) binds the runner's loop without `RuntimeError`. server.py:65 constructs `aiohttp.TCPConnector(ssl=not config.SKIP_TLS_VERIFY)` — the sole transformation of the boundary value is this boolean inversion — and server.py:66-68 constructs the package's only `aiohttp.ClientSession(timeout=ClientTimeout(total=600), connector=connector)`. `ClientSession.__init__` has no `ssl` parameter (claim-37; client.py:274-309), adopts `connector._loop` when `loop` is unset (client.py:314-318), and its same-loop guard (client.py:378-379) is vacuous on that path (claim-50).
5. **Outbound consumption.** Exactly one `ClientSession` exists in the package (claim-25) and all six outbound HTTP call sites use it — backend.py:54, compaction.py:36, routes/chat.py:125, routes/messages.py:130 and :231, routes/passthrough.py:29 — with no alternative HTTP mechanism anywhere (claim-26). None of the six passes `ssl` (claim-182; package-wide grep: the only `ssl` occurrence is server.py:65). aiohttp then resolves each request: the request-level `ssl` default is literal `True` (client.py:513; passed through at client.py:721), `_get_ssl_context` returns `None` for non-TLS schemes (connector.py:1257-1258), and for https the request-level `True` defers to the connector's `self._ssl` (connector.py:1262-1274): `True` → `_SSL_CONTEXT_VERIFIED` (stdlib default context, check_hostname on, `CERT_REQUIRED`), `False` → `_SSL_CONTEXT_UNVERIFIED` (`check_hostname = False`, `CERT_NONE` — connector.py:903-925; claims 39/37).

Resulting posture table: flag absent → `SKIP_TLS_VERIFY=False` → `ssl=True` → HTTPS verified, HTTP plain; flag present → `SKIP_TLS_VERIFY=True` → `ssl=False` → HTTPS unverified (`CERT_NONE`), HTTP plain.

## Dimension Coverage

- **Posture polarity (opt-out, default off): full trace.** `ssl=not config.SKIP_TLS_VERIFY` with default `False` (config.py:17) → verification on unless the flag is explicitly given (main.py:64-67, server.py:65, connector.py:1268-1274).
- **Never inferred from scheme: full trace.** The posture is a constant connector-constructor parameter read once from the config global; the URL scheme only gates whether TLS is used at all (connector.py:1257-1258). No scheme- or host-derived posture exists anywhere in the package (sole `ssl` writer is server.py:65; config.py:14-16 documents the same).
- **Import-bound decorator defaults versus explicit callback: full trace.** Claim-74: exactly three config-backed defaults bind at import (main.py:62, 87, 92); the TLS option is not among them. Its value flows callback-at-parse → mutable module global → runtime read in `on_startup`. No staleness window: one assignment (parse) strictly precedes one read (startup), with no intervening writer; a post-construction mutation of the global (none exists) could not affect the already-built connector, freezing the posture at startup.
- **Process-wide one-connector scope with no per-request `ssl` override: full trace.** One `ClientSession` on one explicit connector (claims 25, 177, 178); all six call sites on `app["session"]` (claim-26); none passes `ssl` (claim-182). A session-level override is structurally impossible (`ClientSession` has no `ssl` parameter — claim-37); a per-request `ssl` would take precedence (connector.py:1262-1267) but none exists. `web.run_app` (main.py:189) runs one app lifecycle per process, and `on_startup` builds the connector once per lifecycle.
- **Loop binding at connector construction: full trace.** `on_startup` is async → connector binds the running loop (connector.py:313); the session adopts `connector._loop` (client.py:314-318); the same-loop guard cannot fire on that path (client.py:378-379; claim-50). All six request sites execute on that same loop; no cross-loop mismatch is reachable.
- **Unset-path behaviour preservation: full trace.** Flag unset → click resolves `False` and still calls the callback (core.py:3304-3308) → `config.SKIP_TLS_VERIFY=False` → `ssl=True` → verified context — observably identical to the pre-option state (aiohttp's own default connector `ssl=True`, claim-39) and to the untouched config default.
- **Closed-union coverage:** both flag states {set, unset} traced end-to-end; both scheme members {http, https} traced (http: no TLS regardless of posture).
- **Dispatch/settlement, duplicate identity, ordering, disposal, rejection, multi-settlement:** no request-selection or terminal-settlement transformation exists on this seam; the single assignment→read ordering is verified above; the connector is constructed once before serving and closed with its session in `on_cleanup` (server.py:89). No duplicate TLS identity, no fallback namespace, no rejection path leaves the boundary half-set.

## Counterexample

None found. Two observations recorded without verdicts, as both fall outside any contract the verified members state:

- The `SKIP_TLS_VERIFY=1` form named in config.py:14-16's comment has no consumer inside the package: the click option declares no `envvar=`, and package-wide grep finds no environment reader. The env-var channel lives outside this seam's package boundary (shell → argv), so within S4 the CLI flag is the only producer of the posture. The in-package chain is closed regardless.
- Repeating `--skip-tls-verify` on the command line keeps the boolean value `True` (click boolean flags are idempotent, not counting) — same posture, not a divergence.

## Sources

- `buster_ripper/main.py:30-33,36-58,61-67,86-95,131-189` (callback, `_apply_cli_config`, decorators, body, `create_app`/`web.run_app` at 188-189)
- `buster_ripper/config.py:1-17` (module contract, `SKIP_TLS_VERIFY: bool = False` at 17)
- `buster_ripper/server.py:1-90` (session provenance at 21, `on_startup` 55-81 with connector at 65 and session at 66-68, `on_cleanup` close at 89)
- `buster_ripper/backend.py:46-57` (`session.get` at 54); `buster_ripper/compaction.py:28-39` (`session.get` at 36); `buster_ripper/routes/chat.py:34-39,122-129` (session at 37, `session.post` at 125); `buster_ripper/routes/messages.py:125-133,226-233` (sessions at 127/228, `session.post` at 130/231); `buster_ripper/routes/passthrough.py:25-32` (session at 25, `session.request` at 29)
- `.venv/lib/python3.13/site-packages/aiohttp/client.py:274-318,370-387,494-525,702-721`; `.venv/lib/python3.13/site-packages/aiohttp/connector.py:308-317,899-926,969-1006,1243-1276,1432-1443`
- `.venv/lib/python3.13/site-packages/click/core.py:1224-1231,1265-1272,1402-1409,2438-2463,2800-2831,3300-3309`

## Source Versions

Working-tree sources, identified by fingerprint below (git metadata not consulted). Installed packages in the repo `.venv` (the runtime the app uses): aiohttp 3.13.3, click 8.3.1 (pinned by `uv.lock`). Provenance note: member verdicts 174/175/176 record click 8.5.0 at `/opt/homebrew/.../python3.14` as their checking environment's incidental resolution; their asserted facts are production-source facts pinned by the repo fingerprints, and every click semantic this composition relies on (parse-time callback invocation, UNSET-flag `False` resolution with callback, body-after-parse ordering) was verified directly against the repo `.venv`'s click 8.3.1 `core.py`.

## Source Fingerprint

Per-source drift check — all 11 consulted files match the member-recorded fingerprints byte-for-byte: main.py `fd36dc12…`, config.py `22348c68…`, server.py `a9e49823…`, backend.py `c2b71468…`, compaction.py `ac9d4214…`, routes/chat.py `16c7e999…`, routes/messages.py `721a2afe…`, routes/passthrough.py `072a5809…`, aiohttp client.py `7ce41fc1…`, aiohttp connector.py `bd3db604…`, click core.py `53a05fc6…`.

Composition digest: `sha256:9e94ac6b1c63f9131979ba00c5c151307975467a07f6777ba15881300740d2e7` — sha256 of the byte stream: line `seam-S4/v1`, then one line per member in ledger order (`claim-175` … `claim-40`, each with its verdict-recorded fingerprint(s) in recorded order, semicolon-joined), then the raw bytes of the 11 consulted source files concatenated in the Sources order above.

## Invalidates On

- Any member fingerprint above changing (member re-extraction / re-verification required before recomposition).
- Removal or rewiring of `--skip-tls-verify`, its `is_flag`/`expose_value=False` shape, or `callback=_set_skip_tls_verify`; any second writer of `config.SKIP_TLS_VERIFY` or rename of the global; a change to its `False` default.
- Any change to `server.py:65-68` (different `ssl` expression, connector class, session built without the explicit connector, connector constructed outside the running loop).
- Any per-request `ssl=`/`verify_ssl=`/`ssl_context=`/`fingerprint=` added to a session call site, any new outbound HTTP mechanism or second `ClientSession` under `buster_ripper/`, or any call site leaving `app["session"]`.
- aiohttp changes to `_get_ssl_context` precedence or scheme gating, `ClientSession.__init__` (connector/loop adoption), or `TCPConnector.__init__` (`ssl` default/`_merge_ssl_params`); click changes to parse-time callback invocation or UNSET-flag resolution; `.venv`/`uv.lock` drift off aiohttp 3.13.3 / click 8.3.1.
- `main.py` body reordering that lets any `config.SKIP_TLS_VERIFY` read precede the parse-time assignment.

## Checked At

2026-09-28

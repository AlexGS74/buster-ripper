VERDICT: SAFE
PROMOTION: INELIGIBLE — the seven supplied member verdicts carry no permanent claim handles (no `CL:`/registry handle appears in any verdict file), so the required handle-inclusive fingerprint cannot be constructed; additionally members 98/75/103 pin `dump.py` at `sha256:c8f093c8…` while the working tree hashes `07b0b076…`, so their whole-file basis is stale by their own file-level Invalidates-On and the ordered-member fingerprint basis is not stable until they re-pin.

## Seam
S3 — the configured dump-format value crossing CLI option (`--dump-format`) → `config.DUMP_FORMAT` global → the per-turn format dispatch in `dump_turn`.

## Composition
Producer → transformations → consumer, all re-verified against current working-tree bytes:

At import of `buster_ripper/main.py`, the module-body decorator for `--dump-format` (main.py:80-85) evaluates `type=click.Choice(list(config.DUMP_FORMATS))` — snapshotting the three-value tuple `("documented", "change-focused", "self-contained")` (config.py:27) — and binds the literal `default="documented"` (not a read of `config.DUMP_FORMAT`); both are import-time evaluations on the Option object (claim-74). At any `main()` parse, click 8.3.1 processes every parameter: `Command.parse_args` iterates `handle_parse_result` (core.py:1226-1227), `consume_value` applies precedence command-line → envvar (none declared) → default-map (none) → default (core.py:2294-2334), `Choice.convert` normalizes and returns an original member or fails with `BadParameter` before the callback (types.py:336-358; claim-41), and `process_value` then invokes the callback unconditionally, including when the value came from the default (core.py:2438-2475; claim-40). `expose_value=False` only suppresses the `ctx.params` entry (core.py:2588-2600); consumption already happened in `_set_dump_format` (main.py:12-15), which assigns `config.DUMP_FORMAT = value` exactly once per parse — with either a Choice-validated literal or `"documented"`. The command body runs strictly later (`Command.invoke`, core.py:1268-1269, after `make_context`).

Consumer: each `dump_turn` call reads the live global — `config.DUMP_DIR` guard, then `fmt = _resolve_format()` (dump.py:85-87), which reads `config.DUMP_FORMAT` and raises `ValueError` on any value outside `config.DUMP_FORMATS` (dump.py:128-133; claim-98), then stamps `state.fmt` and dispatches: `documented` → `_write_documented`, `change-focused` → `_append_record(_change_focused_record(...))`, and the remaining allowed member `self-contained` via the `else` branch → `_append_record(_self_contained_record(...))` (dump.py:88-98; claims 75, 103). Production census (repo-wide, *.py): the only writers of `DUMP_FORMAT` are config.py:28 (module init `"documented"`) and main.py:14 (the callback); the only readers are main.py:80 (import-time Choice list) and dump.py:130/131 (per-turn read + membership check). No runtime format or `DUMP_FORMATS` mutation exists in production code, matching the seam's no-mutation stipulation.

## Dimension Coverage
Full trace — nothing carried forward. The prior on-file composition (2026-09-27, SAFE/INELIGIBLE, superseded at the output path by this document) fired its own Invalidates-On when `main.py` moved `cbe2802d…`→`fd36dc12…` (T006); members 99/74 were re-verified fresh against `fd36dc12…` (2026-09-28). Delta-scoping was additionally barred because members 98/75/103 are stale by their own file-level invalidators (`dump.py` pin `c8f093c8…` ≠ working tree `07b0b076…`, divergence predating T006); per protocol rule 4 the recomposition is full-trace. Every dimension below was re-traced against current bytes:

- **Three format values (closed union, all members):** all three traced end-to-end. `documented` → per-turn documented writer; `change-focused` → collapsed change-focused record builder + buffered append; `self-contained` → collapsed self-contained record builder + buffered append. Total and disjoint; the `else` branch is reachable only by `self-contained` because `_resolve_format` has already restricted the value space to the three members. No unmapped or differently-pathed member.
- **Default `documented`:** two independent defaults agree — the module-init `DUMP_FORMAT: str = "documented"` (config.py:28) on the import-without-CLI path, and the import-bound literal option default plus the always-firing callback on the CLI path, so a pre-parse global value (e.g. left by an earlier in-process `main()` invocation) is always overwritten at parse time. Both entry paths converge on a tuple member.
- **Import-bound decorator default vs runtime dispatch read:** the Choice snapshot and default bind at import; the dispatch reads the live global per turn. Between them the only production writer is the parse-time callback, so the runtime read sees exactly the parse-assigned value (or module-init on the non-CLI path). The CLI never inherits a stale global because its default is a literal, not a config read.
- **Once-per-process callback assignment:** one production assignment site (main.py:14); the callback fires exactly once per `main()` parse — for a supplied value and for the default alike (core.py:2438-2475). No second writer exists in production.
- **Unknown value → `ValueError`:** `_resolve_format` raises before state lookup or writer dispatch (dump.py:131-132). In production the guard is unreachable via the CLI path (Choice rejects first, callback never assigns a non-member) and reachable only by non-CLI assignment to the global — which no production code performs. Note: `main.py` sets no `case_sensitive`/`token_normalize_func`, so Choice matching is exact and `Choice.convert` returns an original tuple member, never a raw unmatched string.
- **`click.Choice` validation:** fails pre-callback with `BadParameter` (types.py:336-358 → 136-143), so an invalid CLI value never reaches the global, the command body, or a writer. Default-map or envvar injection is structurally impossible here (none declared) and would still pass through `Choice.convert` in `process_value`.
- **No runtime format mutation assumed:** confirmed by census — no production writer besides module-init and the callback; no production writer of `DUMP_FORMATS` at all, so the import-time Choice snapshot and the live membership check use identical values. (Mid-process mutation would be picked up per-turn by design — `dump_turn` re-resolves and re-stamps `state.fmt` each turn — but nothing in production performs it; this stays inside the stipulated boundary.)
- **Protocol checklist, N/A items:** no discriminator is discarded before consumption (`expose_value=False` drops the value only after the callback consumed it); no fallback selects a different namespace; no duplicate identities (three distinct strings, one tuple); no message ordering, selection/disposal between dispatch and settlement, or multi-settlement semantics exist on a config-value boundary.

## Counterexample
None found within the traced dimensions and the no-runtime-mutation boundary. Two boundary-adjacent facts recorded, not counterexamples: (1) the third dispatch arm is an `else` keyed on "remaining", so a hypothetical fourth `DUMP_FORMATS` member added without touching `dump_turn` would silently route to the self-contained writer — impossible today because the tuple is closed at three with no production mutator; (2) members 98/75/103's substance (their cited locators and descriptions) matches current `dump.py` bytes exactly at dump.py:83-98,128-133 — only their whole-file pins are stale (see Source Fingerprint).

## Sources
- `buster_ripper/main.py:12-15,61-85,131-137` (production) — option/callback/`_apply_cli_config` exclusion of the format option
- `buster_ripper/config.py:23-28` (production) — `DUMP_FORMATS` tuple and `DUMP_FORMAT` init
- `buster_ripper/dump.py:83-98,128-133` (production) — `dump_turn` dispatch and `_resolve_format`
- `.venv/lib/python3.13/site-packages/click/core.py:1220-1253,1255-1269,2280-2334,2415-2602` (installed dependency)
- `.venv/lib/python3.13/site-packages/click/types.py:328-375` (installed dependency)
- `uv.lock` (click 8.3.1 pin, corroborating)

## Source Versions
- `buster_ripper/main.py` working tree, sha256 `fd36dc12055f67693d9b852fb15a66aa4a3a7864103ac3de0ee67d9520048d47` — matches claims 99/74 pins (fresh, post-T006)
- `buster_ripper/config.py` working tree, sha256 `22348c68d21b91a30e75eeea802d4d390b2bcdabbf974b0ddc743c9354635522` — matches claim-74 corroborating pin
- `buster_ripper/dump.py` working tree, sha256 `07b0b076552f72e62c625caade1b01821b3237375900233812d14c0f275b3c2b` — diverges from claims 98/75/103 pins (`c8f093c8…`); cited locators match their descriptions under current bytes
- click `core.py` sha256 `53a05fc6df06923343ab22741ea5ef96e247b72678b18e54480bccd4276aee64`, `types.py` sha256 `7a4e7804d485c0f2acaad7d3ee3b2a71ce161ee8bc00815530a338a156485e17` — installed click 8.3.1 (repo `.venv`, pinned by `uv.lock`), matching claims 40/41 pins

## Source Fingerprint
sha256:b9e2622d14e65647d6110de0c25a03406b5fc537b420739abd1ac25024ae47b2 — computed over, in order: the seven member lines (`claim-99`/`claim-74` @ `fd36dc12…`, `claim-40` @ `53a05fc6…`, `claim-41` @ `7a4e7804…`, `claim-98`/`claim-75`/`claim-103` @ `c8f093c8…`, one per line) followed by the concatenated current working-tree bytes of `main.py`, `config.py`, `dump.py`, `click/core.py`, `click/types.py`. Not handle-inclusive: no member verdict supplied a permanent handle (hence INELIGIBLE).

## Invalidates On
- Any change to a listed member verdict or fingerprint; re-pinning of claims 98/75/103 to current `dump.py` bytes (required for a stable recomposition basis).
- Any change to `main.py`'s `--dump-format` declaration (choices source, default literal, `expose_value`, callback) or to `_set_dump_format`; relocation of these options off module-level decorators.
- Any new production writer of `config.DUMP_FORMAT` or `config.DUMP_FORMATS`, or any runtime mutation path — breaks the once-per-parse assignment and snapshot/live-tuple equivalence dimensions.
- Any change to `dump.py`'s `_resolve_format` membership check or `dump_turn`'s three-way dispatch, including addition of a fourth `DUMP_FORMATS` member (the `else` arm silently claims new members).
- click moving off 8.3.1 or reworking `process_value`/`consume_value` precedence (default no longer flowing through the callback, or callbacks deferred past `Command.invoke`); sha256 drift of `click/core.py` or `click/types.py` from the pins above.

## Checked At
2026-09-28

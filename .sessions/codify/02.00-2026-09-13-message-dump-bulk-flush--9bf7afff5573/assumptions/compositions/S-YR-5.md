VERDICT: SAFE
PROMOTION: INELIGIBLE — member claims carry no permanent registry handles (`YR-P018`/`YR-P019` are session premise ids, absent from `claims.md` and from `invariants/index.md`); the remaining eligibility conditions (verdict, locators, fingerprint, invalidation conditions) are satisfied.

## Seam

`S-YR-5` — the dump-format value crossing the `--dump-format` option's declared default into `config.DUMP_FORMAT` (CLI default → config default).

Members supplied and verified: `YR-P018` = `VERDICT: HOLDS` (`assumptions/verdicts/yr-p018.md`), `YR-P019` = `VERDICT: HOLDS` (`assumptions/verdicts/yr-p019.md`). No member is `DOES-NOT-HOLD` or `UNCLEAR`, so the member-state guard passes and the boundary was traced in full.

## Composition

Producer — the option declaration: `buster_ripper/main.py:80-85` declares `--dump-format` with `type=click.Choice(list(config.DUMP_FORMATS))`, `default="documented"`, `show_default=True`, `expose_value=False`, and `callback=_set_dump_format`. The default is a string literal, not a read of `config.DUMP_FORMAT`, so the producer value is fixed at `"documented"` independent of the config global's current contents. The callback body is `config.DUMP_FORMAT = value` (`main.py:12-15`), an unconditional assignment returning `value`.

Transformation 1 — every declared parameter is processed on every parse: `click.core.Command.parse_args` (`click/core.py:1219-1243`) iterates `iter_params_for_processing(param_order, self.get_params(ctx))` and calls `param.handle_parse_result(ctx, opts, args)` for each. `iter_params_for_processing` (`click/core.py:116-140`) returns all declared parameters, not only the ones present on the command line, so absence of `--dump-format` does not skip the option.

Transformation 2 — value resolution order: `Parameter.consume_value` (`click/core.py:2294-2338`) takes `opts.get(self.name, UNSET)`; for an absent flag this is `UNSET`, so it consults the envvar (`value_from_envvar`), then `ctx.lookup_default(self.name)`, then `self.get_default(ctx)`. The option declares no `envvar=` and the command declares no `auto_envvar_prefix`, so `Option.resolve_envvar_value` (`click/core.py:3174-3190`) returns `None`. `Context.lookup_default` (`click/core.py:698-715`) returns `UNSET` because no production caller supplies a `default_map`. `Parameter.get_default` (`click/core.py:2258-2289`) therefore falls through to `self.default` (`click/core.py:2281-2285`), which `Parameter.__init__` stored from `default="documented"` (`click/core.py:2170`). Source is `ParameterSource.DEFAULT`.

Transformation 3 — type cast and callback invocation: `Parameter.process_value` (`click/core.py:2411-2466`) passes the resolved value through `type_cast_value`; `click.Choice.convert` (`click.types:336-355`) normalizes `"documented"` and returns the matching original choice `"documented"` from the import-time list `list(config.DUMP_FORMATS)` (`main.py:80`; `config.py:27`). Because `self.default` is a non-`UNSET` string, the legacy `UNSET` branch (`click/core.py:2438-2448`) is not taken and `process_value` calls `self.callback(ctx, self, "documented")` (`click/core.py:2448`). `expose_value=False` only suppresses storing the value into `ctx.params` (`click/core.py:2591-2601`); it does not suppress the callback. `_set_dump_format` therefore assigns `config.DUMP_FORMAT = "documented"` before `main()`'s body runs and before `create_app()` (`main.py:188`).

Consumer — read and branch: `_resolve_format` (`buster_ripper/dump.py:139-144`) reads `config.DUMP_FORMAT` per `dump_turn` call (not at import), checks membership in `config.DUMP_FORMATS` (`config.py:27`), and returns it. `dump_turn` (`dump.py:94-109`) then selects `if fmt == "documented": _write_documented(...)`, which writes `turn_000.json` / `turn_NNN.diff` / `index.txt` (`dump.py:161-197`) and never calls `_append_record`. The collapsed flush paths cannot synthesize an artifact for a documented session either: `flush_all` → `_flush_session` (`dump.py:112-122, 285-292`) returns early when `state.pending` is empty, and `_flush_state` (`dump.py:295-298`) returns early when `state.fmt == "documented"`. The periodic flush task (`server.py:37-45, 80-81`) and the shutdown flush (`server.py:88, 109-114`) therefore have nothing to write for a no-flag run.

Pre-existing-value branch of the question: because `_set_dump_format` assigns unconditionally (`main.py:14`), a `config.DUMP_FORMAT` value present before the callback runs (for example a collapsed value left by an earlier in-process parse or any other in-process writer) is overwritten with `"documented"` by the same parse. The only production writer of `config.DUMP_FORMAT` is that callback (`main.py:14`); `config.py:28` initializes the module global to `"documented"`; every production module imports the same module object with a relative import (`from . import config` / `from .. import config`), and no production source reads `DUMP_FORMAT` at import time. `YR-P019`'s env-var finding is consistent with this: `config.py` imports no `os` and no production module derives `DUMP_FORMAT` from the environment; the shell wrapper's `DUMP_FORMAT` environment variable reaches the binary only as a `--dump-format` argv word when non-empty (`serve_buster_ripper.sh:29, 41, 46-50`), which is the flag-present case, not this seam's absent-flag case.

Net result: with `--dump-format` absent, click resolves `"documented"` from the declared literal and invokes the callback with it, so `config.DUMP_FORMAT` is `"documented"` when the first `dump_turn` runs; the run takes the documented branch and no collapsed artifact is produced.

## Dimension Coverage

1. **Flag absent, no competing source** — traced: command line `UNSET` → envvar `None` (no `envvar=`, no `auto_envvar_prefix`) → `default_map` `UNSET` (none set in production) → declared default `"documented"` → callback assignment. Documented branch.
2. **Flag present with `documented`** — traced: command line value wins; `Choice` returns `"documented"`; callback assigns `"documented"`. Documented branch.
3. **Flag present with `change-focused`** — traced as the other closed-union member: `Choice` returns it; callback assigns it; `dump_turn` takes `_append_record`. Collapsed branch, only reachable with the flag present.
4. **Flag present with `self-contained`** — traced as the third closed-union member: same path, `_append_record` with the self-contained record builder. Collapsed branch, only reachable with the flag present.
5. **Flag present with a value outside the closed union** — traced: `click.Choice.convert` calls `self.fail` (`click/types.py:349-354`), parse exits with a usage error, `main()` never runs, no artifact of any kind.
6. **`config.DUMP_FORMAT` already holding a different value when the callback runs** — traced: the callback overwrites it unconditionally with the resolved value; a pre-existing collapsed value cannot survive the no-flag parse.
7. **`config.DUMP_FORMAT` module default versus CLI default** — traced: the producer default is the literal `"documented"`, the config module default is also `"documented"` (`config.py:28`), and the CLI literal is what is written; the module default is not consulted by the callback.
8. **`expose_value=False` versus callback invocation** — traced: the flag suppresses only `ctx.params` storage, not `process_value`'s callback call; the command callback signature omits `dump_format` (`main.py:131-137`).
9. **Callback timing relative to consumers** — traced: parse-time assignment precedes `main()`'s body, `create_app()` (`main.py:188`), `on_startup` and any request handling; the consumer reads the global per `dump_turn` call, not at import.
10. **Callback cardinality per process** — traced: exactly one assignment per `main()` parse; a second in-process parse re-assigns rather than accumulating.
11. **Documented-branch selection in the consumer** — traced: `_resolve_format` membership check → `fmt == "documented"` → `_write_documented`; the collapsed `else` arm receives only `"self-contained"` under current `config.DUMP_FORMATS`.
12. **Collapsed-artifact guards on the no-flag path** — traced: documented turns never call `_append_record`, so `state.pending` stays empty; both `_flush_session` and `_flush_state` additionally refuse to write for an empty batch / documented `state.fmt`.
13. **Environment channel** — traced: no `envvar=` on the option, no `auto_envvar_prefix` on the command, no env read in `config.py`; the wrapper's `DUMP_FORMAT` env is a flag-present argv channel only.
14. **Programmatic start without a CLI parse** — traced: direct `create_app()` leaves the `config.py:28` module default `"documented"` in place, so the documented branch is still taken; no production caller does this outside `main()`.

## Counterexample

None found.

## Notes

- The seam row for `S-YR-5` is not present in `assumptions/seams.md` (that ledger has no `YR` rows); the seam statement supplied in the dispatch — `S-YR-5` = `YR-P018` → `YR-P019` (CLI default → config default) — was used, and both supplied verdicts were read and are `HOLDS`. The composition itself needs no additional member.
- Inert duplication: `main.py:81`'s literal `"documented"` and `config.py:28`'s module default `"documented"` are two independent sources that agree today. Because the callback assigns unconditionally, the module default is not consulted on CLI runs, so changing `config.py:28` alone would not move the CLI behavior, and changing `main.py:81` alone would move it without moving the module default. This is a wording/design-intent gap only; the traced no-flag behavior is unambiguous and no production caller reads the module default on the CLI path.
- Inert latent shape: `dump_turn`'s `else` arm (`dump.py:106-107`) is a catch-all for any value other than `"documented"`/`"change-focused"`. `_resolve_format` rejects values outside `config.DUMP_FORMATS` first, so under current config the `else` arm only receives `"self-contained"`. If `config.DUMP_FORMATS` were reassigned at runtime while the `click.Choice` retained its import-time list (`main.py:80`), a value could pass `Choice` and then raise in `_resolve_format`; no production writer reassigns `config.DUMP_FORMATS`.
- Inert callback shape: `_set_dump_format` has no `None` guard. Safety on the absent-flag path rests on `self.default` being a non-`UNSET` string, so `click`'s legacy `UNSET`→`None` branch (`click/core.py:2438-2448`) is not taken. Removing `default=` would make the callback write `None` and make `_resolve_format` raise per turn (swallowed by `routes/messages.py:56-60`, which logs `"dump failed"`); no such change exists today.
- `--help` is eager and exits before `--dump-format` is processed, and an invalid earlier parameter aborts parsing before it; neither path starts the server, so neither can produce any artifact. No production caller supplies `ctx.default_map`, `auto_envvar_prefix`, `envvar=`, or `context_settings` to the command.
- `YR-P019`'s claim is about `config.py` only. The shell wrapper does export/forward a `DUMP_FORMAT` environment variable, but only as a `--dump-format` argv word when non-empty (`serve_buster_ripper.sh:29, 41, 46-50`); that is the flag-present case and does not contradict the config-module finding.

## Sources

- `buster_ripper/main.py:12-15` — `_set_dump_format`; unconditional `config.DUMP_FORMAT = value`.
- `buster_ripper/main.py:80-85` — `--dump-format` declaration: `default="documented"`, `expose_value=False`, `callback=_set_dump_format`, `type=click.Choice(list(config.DUMP_FORMATS))`.
- `buster_ripper/main.py:131-149, 188` — `main()` body, `_apply_cli_config`, `create_app()`.
- `buster_ripper/config.py:27-28` — `DUMP_FORMATS` closed tuple and `DUMP_FORMAT: str = "documented"`; module imports no `os`.
- `buster_ripper/dump.py:94-109` — `dump_turn`: `_resolve_format()`, `if fmt == "documented": _write_documented(...)`.
- `buster_ripper/dump.py:112-122, 139-144` — `flush_all`, `poll_flush`, `_resolve_format` (per-call read of `config.DUMP_FORMAT`).
- `buster_ripper/dump.py:161-197` — documented writer (`turn_000.json`, `turn_NNN.diff`, `index.txt`), no `_append_record`.
- `buster_ripper/dump.py:200-206, 280-298` — `_append_record`, `_artifact_path`, `_flush_session` (empty-batch return), `_flush_state` (`state.fmt == "documented"` return).
- `buster_ripper/server.py:37-45, 80-81, 88, 109-114` — periodic flush task spawn and shutdown flush.
- `buster_ripper/routes/messages.py:56-60` — the only production `dump_turn` call site, guarded by `config.DUMP_DIR is not None` and swallowing dump exceptions.
- `serve_buster_ripper.sh:29, 41, 46-50` — `DUMP_FORMAT` environment value forwarded only as a non-empty `--dump-format` argv word.
- `click/core.py:116-140` — `iter_params_for_processing` returns all declared parameters.
- `click/core.py:1219-1243` — `Command.parse_args` processes every declared parameter.
- `click/core.py:2170, 2258-2289` — `Parameter.__init__` stores `default`; `get_default` falls back to `self.default`.
- `click/core.py:2294-2338` — `Parameter.consume_value` resolution order: command line → envvar → `default_map` → default.
- `click/core.py:2411-2466` — `Parameter.process_value`: type cast, then unconditional callback invocation.
- `click/core.py:2538-2601` — `Parameter.handle_parse_result`: `expose_value` gates storage only.
- `click/core.py:3174-3210, 3234-3300` — `Option.resolve_envvar_value` / `value_from_envvar` / `consume_value`.
- `click/core.py:698-715` — `Context.lookup_default` returns `UNSET` when `default_map` is `None`.
- `click/types.py:233-355` — `Choice.__init__` / `normalize_choice` / `convert`; returns the matching original choice or fails.

## Source Versions

- `buster_ripper/main.py` — no version string in the source; worktree file as supplied, 193 lines.
- `buster_ripper/config.py` — no version string in the source; worktree file as supplied, 124 lines.
- `buster_ripper/dump.py` — no version string in the source; worktree file as supplied, 316 lines.
- `click` — 8.3.1, project venv `.venv/lib/python3.13/site-packages/click` (`importlib.metadata.version("click")`); CPython 3.13.8 venv interpreter.
- `serve_buster_ripper.sh` — no version string; worktree file as supplied.

## Source Fingerprint

`sha256:59d8903a3e601d786313fad523a38a3cec89454cfbaba900b3230409873da276`

Digest manifest (ordered member ids and fingerprints, then consulted source bytes):

```text
S-YR-5
member YR-P018 sha256:fd36dc12055f67693d9b852fb15a66aa4a3a7864103ac3de0ee67d9520048d47
member YR-P019 sha256:22348c68d21b91a30e75eeea802d4d390b2bcdabbf974b0ddc743c9354635522
source buster_ripper/main.py sha256:fd36dc12055f67693d9b852fb15a66aa4a3a7864103ac3de0ee67d9520048d47
source buster_ripper/config.py sha256:22348c68d21b91a30e75eeea802d4d390b2bcdabbf974b0ddc743c9354635522
source buster_ripper/dump.py sha256:536ac10518ab44cce0ff09de984a89912a920bc67b27482cfb34f526123a95fc
source click/core.py sha256:53a05fc6df06923343ab22741ea5ef96e247b72678b18e54480bccd4276aee64
source click/types.py sha256:7a4e7804d485c0f2acaad7d3ee3b2a71ce161ee8bc00815530a338a156485e17
```

Member-recorded fingerprints match the current bytes exactly (`main.py` `fd36dc12…`, `config.py` `22348c68…`).

## Invalidates On

- Any change to `buster_ripper/main.py` off `sha256:fd36dc12…` — in particular the `--dump-format` `default=` literal, `expose_value`, `callback`, `type=`, or `_set_dump_format`'s assignment.
- Any change to `buster_ripper/config.py` off `sha256:22348c68…` — in particular `DUMP_FORMAT`'s module default, `DUMP_FORMATS` membership, or the introduction of any environment/settings read for `DUMP_FORMAT`.
- Any change to `buster_ripper/dump.py` off `sha256:536ac105…` — in particular `_resolve_format`'s read timing or membership check, `dump_turn`'s branch order, or the `_flush_session` / `_flush_state` documented-format guards.
- Any change to `buster_ripper/server.py`'s flush spawn / shutdown flush, or to `buster_ripper/routes/messages.py`'s `dump_turn` call-site guard.
- A click upgrade or downgrade that changes `Parameter.consume_value` / `get_default` / `process_value` / `handle_parse_result` callback-on-default semantics, `Context.lookup_default`, or `Choice` normalization, off `sha256:53a05fc6…` / `sha256:7a4e7804…`.
- Any production code introducing `ctx.default_map`, `auto_envvar_prefix`, `envvar=` on `--dump-format`, a `context_settings` default map, or any other writer of `config.DUMP_FORMAT`.
- A re-issued or re-scoped member verdict for `YR-P018` or `YR-P019` (either leaving `HOLDS`), or assignment of permanent registry handles that changes the promotion basis.

## Checked At

2026-10-17T22:43:38Z

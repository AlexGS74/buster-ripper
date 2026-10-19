VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`click` invokes an option's `callback` during CLI parsing, before the command function body runs

## What it actually does
Confirmed in the installed source. During `Command.parse_args` (core.py:1226-1227), each parameter's `handle_parse_result` runs, which calls `Parameter.process_value`; there, `if self.callback is not None: value = self.callback(ctx, self, value)` (core.py:2438-2463) — so an option's callback executes while parsing, inside `make_context`. The command function body runs strictly later: `BaseCommand.main` builds the context via `self.make_context(...)` (which itself calls `parse_args` and whose docstring states "It does not invoke the actual command callback though") and only then calls `self.invoke(ctx)` (core.py:1405-1406), where `Command.invoke` executes `ctx.invoke(self.callback, **ctx.params)` (core.py:1268-1269). This is also the documented basis for eager options (e.g. `--help`) exiting from a callback before the command body starts.

## Source
`.venv/lib/python3.13/site-packages/click/core.py` — `Parameter.process_value` (lines 2438-2463) reached via `Command.parse_args` (lines 1226-1227) / `handle_parse_result` (line 2577); command body in `Command.invoke` (lines 1268-1269) after `make_context` in `BaseCommand.main` (lines 1405-1406)

## Source Version
click 8.3.1 (installed in repo `.venv`, pinned by `uv.lock`)

## Source Fingerprint
sha256:53a05fc6df06923343ab22741ea5ef96e247b72678b18e54480bccd4276aee64 (click/core.py)

## Invalidates On
- click version changes in `.venv`/`uv.lock` away from 8.3.1 with a rework of parameter processing order
- sha256 of `click/core.py` no longer matches the fingerprint above
- click defers parameter callbacks past `Command.invoke` (e.g. a parsing-model redesign in a future major version)

## Checked At
2026-09-28

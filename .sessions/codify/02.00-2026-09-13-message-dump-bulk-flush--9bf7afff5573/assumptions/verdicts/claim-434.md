VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`monkeypatch.delenv(name, raising=False)` leaves an absent variable absent without raising `KeyError`

## What it actually does
`MonkeyPatch.delenv` delegates to `self.delitem(os.environ, name, raising=raising)`. In `delitem`, when `name not in dic` it raises `KeyError(name)` only if `raising` is true; with `raising=False` it takes no action, so an absent variable remains absent and no `KeyError` is raised. (The `delenv` docstring states the same: "Raises ``KeyError`` if it does not exist, unless ``raising`` is set to False.")

## Source
/Users/alexhubitski/Projects/buster-ripper/.venv/lib/python3.13/site-packages/_pytest/monkeypatch.py — `MonkeyPatch.delitem` (lines 299-311) and `MonkeyPatch.delenv` (lines 333-340)

## Source Version
pytest 9.0.2 (per /Users/alexhubitski/Projects/buster-ripper/.venv/lib/python3.13/site-packages/_pytest/_version.py line 31)

## Source Fingerprint
sha256:597dffe9ccd42e1bdcdf0485fb437c55eb1bbfdc0637777096261c0ef4fc11fb

## Invalidates On
- Any pytest release that changes `MonkeyPatch.delitem` so the `raising=False` branch raises or mutates the mapping
- Any pytest release that changes `MonkeyPatch.delenv` to stop delegating to `delitem` or to alter the `raising` semantics
- The installed `_pytest/monkeypatch.py` bytes changing (fingerprint no longer matches)

## Checked At
2026-10-17

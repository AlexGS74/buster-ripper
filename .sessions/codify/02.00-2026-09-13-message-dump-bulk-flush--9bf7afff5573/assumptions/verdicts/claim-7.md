VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`dump.py` appends one summary line per turn to `index.txt` — the `FULL` form on turn 0, the `+added/-removed` form on later turns

## What it actually does
Every `dump_turn` call builds exactly one summary string and appends it (`f.write(summary + "\n")`) to `{session_dir}/index.txt` opened in append mode. Turn 0's summary is `f"turn 000  {now}  FULL  {n_msgs} msgs  {n_tools} tools  {size_kb:.1f} KB"`; turns ≥ 1 use `f"turn {turn:03d}  {now}  +{added}/-{removed} lines  {n_msgs} msgs  {n_tools} tools"`, where `added`/`removed` count `+`/`-` diff lines excluding the `+++`/`---` headers.

## Source
`buster_ripper/dump.py`, `dump_turn` — FULL summary at line 84, +/- summary at lines 98–101, append to `index.txt` at lines 104–106

## Source Version
unavailable (git metadata excluded per protocol boundary)

## Source Fingerprint
sha256:243b21434c14e6ea8144329fe1d0ac4e0c499ed9a7e6e09bdcac5e2417a56b1d

## Invalidates On
- any change to either summary format string, the `FULL` marker, or the `+{added}/-{removed}` form in `buster_ripper/dump.py`
- index line written more or less than once per `dump_turn` call, or the append target changed from `index.txt`

## Checked At
2026-09-19

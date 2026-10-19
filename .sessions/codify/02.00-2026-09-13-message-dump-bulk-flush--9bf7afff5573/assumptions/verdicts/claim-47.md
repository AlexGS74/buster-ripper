VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
No symbol under `buster_ripper/` owns the collapsed-artifact role — no `<format>.log` artifact filename constant, no once-per-flush append target — while `dump.py:104-106`'s per-session `index.txt` append is the existing per-session append target of the documented layout

## What it actually does
All three halves verify against the source. (1) No `<format>.log` filename constant: the string `.log` has zero occurrences under `buster_ripper/` (isolated grep); artifact filenames are inline literals — `"turn_000.json"` (dump.py:81), `f"turn_{turn:03d}.diff"` (:94), `"index.txt"` (:104) — and the only dump-related config global is `DUMP_DIR` (config.py:16), a directory root, not a filename constant. The only "log" symbols in the package are logging logger objects (`logging.getLogger("buster-ripper")` at dump.py:13 and server.py:46, consumed as `request.app["log"]` in routes/messages.py), which name no artifact. (2) No once-per-flush append target: the word "flush" has zero occurrences under `buster_ripper/` — no function, constant, or concept implements a flush — and the package's only append-mode open is dump.py:105 (`index.open("a", ...)` + `f.write(summary + "\n")` at :106), which fires once per `dump_turn` call, itself called once per intercepted request (routes/messages.py:55-57, guarded on `config.DUMP_DIR is not None`); nothing else in the package appends to anything. (3) The positive half: dump.py:104 computes `index = state.session_dir / "index.txt"` with `session_dir = config.DUMP_DIR / sid` (:71) — one directory per session — then :105-106 opens that file in append mode and writes one summary line per turn, exactly the per-session artifact the documented layout describes (README.md:146-152, "Output layout under `DUMP_DIR/<session_id>/`" listing `index.txt # one-line summary per turn`, with the documented reading workflow at :154-159).

## Source
buster_ripper/dump.py:104-106 (per-session append), with :71 (session_dir = config.DUMP_DIR / sid) and :81/:94/:104 (filenames as inline literals); buster_ripper/config.py:16 (`DUMP_DIR` is the only dump global); negative sweeps: grep `\.log`, `flush`, and `collaps|\.log"|\.log'|log"|LOG` under `buster_ripper/*.py` — no `.log` string, no flush occurrence, no collapse-named symbol, only logger-object matches; buster_ripper/routes/messages.py:55-57 (dump_turn call frequency); README.md:146-159 (the documented output layout and index reading workflow).

## Source Version
unavailable — local working tree; git metadata is out of scope for this role

## Source Fingerprint
sha256:243b21434c14e6ea8144329fe1d0ac4e0c499ed9a7e6e09bdcac5e2417a56b1d (buster_ripper/dump.py); sha256:a8d597c72b462d9df5009906db82e1096187889de21efb8c51745a1728b605c3 (README.md)

## Invalidates On
Any `.log` artifact filename constant, or any flush-named symbol that is an append target, appears under `buster_ripper/`
The `index.txt` append at dump.py:104-106 moves, changes mode, or the per-session dir computation at dump.py:71 changes
README.md's Output layout section (the `index.txt` entry) is removed or changed
Any byte of buster_ripper/dump.py or README.md changes (fingerprint mismatch — re-run the sweeps)

## Checked At
2026-09-19T17:32:21Z

## Interpretive note (per P9 audit)
"Collapsed-artifact role" and "flush" are the caller's terms; this verdict checked the claim's own operationalizations (`<format>.log` filename constant; once-per-flush append target) as literal facts. The only reading that would make the negative half false is one where each `dump_turn` call counts as "a flush", making the per-turn `index.txt` append "once-per-flush" — but the claim itself assigns that same append to the per-session role in the same sentence, so that reading is self-contradictory and is not a suppressed alternative meaning. No definition was picked to reach this verdict: both negative halves are literal-string verified (no `.log` anywhere; no flush concept; the sole append is the per-session index.txt one), and the positive half is located at the cited lines.

## Process disclosure — P2 breach by the checker
Per P2 (cited for audit): during this check, two grep invocations that were given a single-file path (`buster_ripper_legacy.py`, then `README.md`) were executed by the tool as repo-wide searches, and their output pulled matching lines from files inside the session directory — `.sessions/codify/02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/decisions.md` and `.sessions/codify/02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573/handoff-context.md` — into the checker's context (repo-level files `invariants/index.md` and `invariants/buster_ripper/*` also appeared; those are repo content, not session material, but they carry prior checker verdicts and were likewise not used as establishing sources). No session file was opened deliberately, and no fact in this verdict is established from any session file: the no-`.log` sweep, the no-flush sweep, the sole-append sweep, the dump.py reads, and config.py were all gathered from scoped production-source greps and reads before that grep ran; the only item first located by the contaminated output was README.md's layout section — a repo document, subsequently grounded by a direct clean read at :146-159 and fingerprinted above. This verdict is recorded on the merits rather than as `MALFORMED` because the defect is in the checker's execution, not in the claim's extraction — the claim is fully answerable by a blind reader from the package source and README, as this file demonstrates.

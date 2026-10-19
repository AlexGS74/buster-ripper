# Refuse a symlinked session directory before the dump writer uses it

- Session: 02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573
- Findings: PR #1 review comment 4309720778 — `buster_ripper/dump.py:155` (P2)
- Goal: A pre-existing live symlink at `DUMP_DIR/<sid>` cannot make the dump writer place any artifact outside `DUMP_DIR`.
- Done when: with `DUMP_DIR/<sid>` a symlink to an existing outside directory, `dump_turn` writes nothing outside and the symlink survives; a healthy session still captures; the process still serves; the witness is red before the guard.
- Wrong witness: the guard checks only the artifact leaf (`_require_writable` already does); or the writer resolves the session directory and writes through it, turning the escape into a legitimate-looking path; or it deletes/replaces the symlink and routes around the fault.
- In scope: `buster_ripper/dump.py` (`_lookup_or_create_state`); `tests/test_dump.py` and the R2 hostile tier in `tests/qa/test_qap_ms001.py`.
- Out of scope: `_require_writable`/`_writable_target`'s existing leaf checks; the documented writer's output bytes; the collapsed writer's artifact bytes.
- Agrees with: QAP-MS001 R2's documented-writer containment contract (`test_hostile_r2_documented_writer_does_not_escape_the_dump_root`, `:948`/`:951`); `_require_writable`'s "absent or a regular, non-symlink file" rule; `G-006`/`G-008`; the caller's swallow-all arm at `messages.py:55-59`.
- Verify: `uv run pytest tests/test_dump.py tests/qa/test_qap_ms001.py -q`; `uv run pytest -q`.
- Capacity: `dump.py` is 316 raw lines; one guard keeps it in the 301–500 review band.

## Forward
- Changed: `buster_ripper/dump.py` — `_lookup_or_create_state` refuses a session dir that is a symlink or an existing non-directory before `mkdir`, so no artifact can be routed outside `DUMP_DIR`.
- Changed: `tests/test_dump.py` — new `TestSessionDirGuard`: live symlink at `DUMP_DIR/<sid>` is refused, target stays empty, link survives, no state cached; a regular file there is refused and left intact.
- Changed: `tests/qa/test_qap_ms001.py` — R2 gains case (b): a live symlink at `DUMP_DIR/<sid>` to a real outside directory receives no file and survives; later cases relettered (c)/(d).
- Ran: `uv run pytest tests/test_dump.py -q` -> 44 passed (coverage gate red only because the file runs alone; new tests red before the guard: 2 failed).
- Ran: `uv run pytest tests/qa/test_qap_ms001.py -q --no-cov` -> 17 passed.
- Ran: `uv run pytest -q` -> 517 passed, coverage 96.72%.
- Table (`DUMP_DIR/<sid>` at state creation):
  - absent -> created under the root; writes stay inside.
  - real directory -> reused; writes stay inside.
  - live symlink → existing directory -> refused (`ValueError`), nothing written, link survives.
  - dangling symlink -> refused (was an incidental `FileExistsError`); nothing written, link survives.
  - regular file / other non-directory -> refused (was an incidental `FileExistsError`); path untouched.
- Noticed: a symlink swapped in at `DUMP_DIR/<sid>` *after* state creation still routes later writes outside; closing it needs dirfd-relative opens (mechanism change) and write access to `DUMP_DIR`. Ask: no — the brief scopes pre-existing links only, and the swap requires local write access to the dump root.
- Noticed: `DUMP_DIR` itself being a symlink is config-supplied, not request-derived; left alone. Ask: no — config is the operator's own choice.

## Yield
- Verdict: ACCEPT
- Ran: `uv run pytest tests/test_dump.py -q` -> 44 passed; coverage 40.87% red only because the file runs alone
- Ran: `uv run pytest tests/qa/test_qap_ms001.py -q --no-cov` -> 17 passed (the changed end-to-end tier, run once alone)
- Ran: `uv run pytest -q` -> 517 passed, coverage 96.72%
- Ran: `python3` probe of `Path.mkdir(parents=True, exist_ok=True)` -> on a symlink-to-directory it succeeds and the write lands at the outside target; on a regular file it raises `FileExistsError`, so both new unit tests are red without the guard
- Ran: `git diff --check -- buster_ripper/dump.py tests/test_dump.py tests/qa/test_qap_ms001.py` -> clean; `wc -l buster_ripper/dump.py` -> 319 (301–500 band)
- Wrong witness: the guard is on `session_dir` in `_lookup_or_create_state` (`dump.py:154-156`), not the artifact leaf; the diff adds no `resolve()`, `unlink`, `rmdir` or `shutil` call; the unit test and QA case (b) show the outside directory empty and the symlink still a symlink
- Findings: none
- Noticed: the guard re-expresses `_writable_target`'s symlink clause but cannot call it, because a session directory must be a directory and `_writable_target` requires a regular file. Ask: no — the two predicates cannot disagree on any input
- Noticed: the non-directory clause only changes the exception type for a regular file (`FileExistsError` -> `ValueError`); the only caller swallows both. Ask: no — no user-visible path reads the type
- Noticed: a symlink swapped in at `DUMP_DIR/<sid>` after state creation still routes later writes outside; closing it needs dirfd-relative opens (a mechanism change) and write access to `DUMP_DIR`. Ask: no — the brief scopes state-creation-time links and the residual needs local write access to the dump root
- Noticed: unlisted working-tree changes under `.sessions/codify/**` are the parent session's review-routing bookkeeping (iteration 124), not this change; they stay uncommitted. Ask: no — they belong to the coordinator

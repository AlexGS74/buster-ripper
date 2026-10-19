VERDICT: DOES-NOT-HOLD
PROMOTION: ELIGIBLE

## Claim
In `tests/qa/test_qap_ms001.py`, `assert_documented_contents` compares the content of each documented file against an expected value that is not produced by the code under test (byte-for-byte equality, or a complete expected value set).

## What it actually does
`assert_documented_contents` (lines 104-146) never compares a file's content to an expected byte string and never compares against a complete expected value set. It asserts partial properties only, per documented file:

- `turn_000.json` (109-111): the text parses as JSON, is a `dict`, has a truthy `"model"` key, and `markers[0]` occurs as a substring of `json.dumps(body)`.
- `index.txt` (114-133): the number of non-blank lines equals `len(markers)`; each line matches the regex `^turn [0-9]{3}  [0-9]{2}:[0-9]{2}`; `line.split()[0] == "turn"` and `line.split()[1] == f"{i:03d}"`; plus clause regexes (`\d+ msgs`, `\d+ tools`, and `FULL` + `\d+ KB` for turn 0 else `\+\d+/-\d+ lines`).
- `turn_NNN.diff` (136-146): `"---"`, `"+++"` and `"@@"` occur anywhere in the text; some line starting with `"+"` contains `markers[i]`; some line starting with `"-"` contains `markers[i-1]`.

So the oracle is structural/substring-based. The expected values it does use (the test-generated `markers`, the literal `"turn"` / `"---"` / `"+++"` / `"@@"` tokens, and the README-derived regexes at lines 72-77) are indeed test-authored rather than produced by the code under test, but none of them is a byte-for-byte expected file content or a complete expected value set: each check accepts a whole family of contents. Under either reading of the parenthetical — (a) the comparison must be exact/complete, or (b) the comparison form is one of the two named — the claim is false, because neither form occurs anywhere in the helper. A looser reading in which "expected value" means any hand-written assertion would make the claim vacuous (true of any test at all), and the parenthetical's specificity rules that reading out.

## Source
`/Users/alexhubitski/Projects/buster-ripper/tests/qa/test_qap_ms001.py`, lines 104-146 (`def assert_documented_contents`), plus its call sites at line 254 (TC-107) and line 284 (TC-108). Subject is the test suite itself, so the test file is the production surface per P10's exception.

## Source Version
Repository worktree file; no revision string supplied (Git metadata not consulted per the Git metadata boundary).

## Source Fingerprint
sha256:b86bda0340c37dcea5e44701f5a81344e9094f485e57a5b03205f6b37f1fe35c

## Invalidates On
- Any edit to `assert_documented_contents` (lines 104-146) or to the regex constants `README_LINE_PATTERN`, `FULL_CLAUSE`, `KB_CLAUSE`, `CHANGE_CLAUSE`, `MSGS_CLAUSE`, `TOOLS_CLAUSE` (lines 72-77).
- `assert_documented_contents` gaining a byte-for-byte content comparison or a complete expected-content value set for any documented file.
- The file's sha256 changing for any reason.

## Checked At
2026-10-17T22:55:53Z

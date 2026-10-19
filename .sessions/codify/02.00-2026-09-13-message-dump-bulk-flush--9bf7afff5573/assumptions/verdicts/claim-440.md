VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`difflib.unified_diff` retains the input line's terminator on each content line it yields

## What it actually does
Content lines are yielded as the original input string with only a one-character prefix prepended: `' ' + line` for equal, `'-' + line` for replace/delete, `'+' + line` for replace/insert. The input string `line` is embedded verbatim, so whatever line terminator it carries (or lacks) is retained; `lineterm` is applied only to the `---`, `+++`, and `@@` control lines. (With `color=True`, a reset escape sequence is appended after the line, but the line's own terminator is still retained.) The docs confirm this split: control lines are "created with a trailing newline" via `lineterm`, while for inputs without trailing newlines one sets `lineterm=''` "so that the output will be uniformly newline free."

## Source
CPython 3.15 `Lib/difflib.py`, function `unified_diff`, content-line yields (`yield f'{t.context} {line}{t.reset}'`, `yield f'{t.removed}-{line}{t.reset}'`, `yield f'{t.added}+{line}{t.reset}'`), https://raw.githubusercontent.com/python/cpython/3.15/Lib/difflib.py ; corroborated by https://docs.python.org/3/library/difflib.html (diff control lines vs inputs without trailing newlines).

## Source Version
CPython 3.15 branch source; Python 3.15.0 documentation

## Source Fingerprint
sha256:aaa3d1ccdb8d8f0d2e8c99fe1a320a983b770fad94779f1dd417cdd37c033159 (Lib/difflib.py); sha256:694e808b780a68f0481f36fde8ef38db17adf6b74373106fcbc49c22016506ff (docs page)

## Invalidates On
- Content-line yields start appending or substituting `lineterm` instead of embedding the input line verbatim
- Content lines begin being stripped of trailing newlines before prefixing
- A different difflib implementation or major version replaces this behavior

## Checked At
2026-10-17

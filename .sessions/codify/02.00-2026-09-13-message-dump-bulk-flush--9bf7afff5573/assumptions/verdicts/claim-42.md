VERDICT: HOLDS
PROMOTION: INELIGIBLE — establishing both the option's default and configured-value precedence requires two source files, not one exact source locator.

## Claim
pytest's `norecursedirs` ini option replaces the default directory-exclusion list rather than extending it

## What it actually does
pytest registers a default list for `norecursedirs`; when the option is configured, `getini` returns the configured list instead of combining it with that default. Other independent exclusion checks (such as virtualenv detection) still apply.

## Source
Installed pytest `_pytest/main.py:213-228,451-458` (registration and use); `_pytest/config/__init__.py:1627-1634,1673-1706,1743-1746,1805-1819` (configured-value precedence and parsing).

## Source Version
pytest 9.0.2 (installed package metadata).

## Source Fingerprint
`_pytest/main.py` sha256:7ca2b20cc41fbfc2761bcb231eecb26967d553651035c40000f44d70e174b693
`_pytest/config/__init__.py` sha256:ff967dc8cbd4baeaa10b2e370f638ead045306b13926b06d06eb778c3f81e1ea

## Invalidates On
Change to `norecursedirs` registration or exclusion handling in `_pytest/main.py`.
Change to default/configured-value precedence or args parsing in `_pytest/config/__init__.py`.

## Checked At
2026-09-27

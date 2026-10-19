VERDICT: HOLDS
PROMOTION: ELIGIBLE

## Claim
`subprocess.Popen` with no `env=` argument starts the child with the parent's environment at spawn time

## What it actually does
`Popen`'s `env` parameter defaults to `None`. When `env` is `None`, the child is not given a replacement environment; the documented default is to inherit the current process' environment, i.e. the environment of the parent at the moment the process is created. (Passing a mapping instead replaces that inherited environment entirely.)

## Source
https://docs.python.org/3/library/subprocess.html — "Popen Constructor" section, `env` parameter: "If *env* is not `None`, it must be a mapping that defines the environment variables for the new process; these are used instead of the default behavior of inheriting the current process' environment." The same wording appears in "Frequently Used Arguments" for `run()`.

## Source Version
Python 3.15.0 documentation (docs.python.org/3, current development docs); the `env=None` default and inheritance behavior are long-standing and not marked with a version change on this page.

## Source Fingerprint
sha256:bcbee8309e8e6c4aec03088b1ac0ab1e5c29390c8158ff35ccc00e1776876fc8

## Invalidates On
- The `env` parameter's documented default stops being `None` in the Popen signature on docs.python.org/3/library/subprocess.html
- The sentence "these are used instead of the default behavior of inheriting the current process' environment" is removed or altered
- The fetched page content changes so its sha256 no longer matches the recorded fingerprint

## Checked At
2026-10-18T00:24:36Z

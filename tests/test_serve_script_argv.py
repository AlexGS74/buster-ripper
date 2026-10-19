"""Hermetic argv witness for serve_buster_ripper.sh (T011, T012).

The wrapper's usage header documents ``DUMP_DIR="" bash serve_buster_ripper.sh`` as
"disable prompt diffing", but line 28's ``${DUMP_DIR:-…}`` default substituted the
non-empty path for an explicitly empty value. Line 40's ``-n`` guard (as renumbered
by T012; line 36 at the time of the fix) was therefore
always true from the environment and ``--dump-dir`` always reached the binary's
argv — the documented off-switch had never worked. FK-010 option 1 settles it:
honor the documentation by fixing the script, so line 28's default applies only
when ``DUMP_DIR`` is *unset* (``${DUMP_DIR-…}``).

T012 widens the witness from the *presence* of each flag to the *shape* of each
forwarded value. Lines 36-37 used to interpolate values into one ``EXTRA_FLAGS``
string that line 46 expanded unquoted, so ``DUMP_DIR="/tmp/my docs"`` reached the
binary as three words — the path truncated at its space and the remainder an
orphan positional — while a whitespace-only value contributed no field at all and
swallowed the flag that followed it. The assembly is an array now, expanded
word-by-word, so every value arrives as exactly one argument verbatim (no word
splitting, no pathname expansion), and the all-appends-suppressed run still execs
on bash 3.2, where a bare empty-array expansion aborts under ``set -u``.

Witness mechanism: a PATH shim stands in for the ``buster-ripper`` console script
and records the argv the wrapper would have exec'd. No server starts, no port is
bound, and no upstream is contacted. ``HOME`` is pinned to a path that is never
created, so the default case asserts without touching the real home directory, and
every run starts from a scrubbed copy of the environment so a developer's own
``DUMP_DIR``/``DUMP_FORMAT`` cannot leak into an expected argv. The witnesses run
the wrapper through PATH's ``bash``, so the interpreter that was proved is stated
in its own test id (see ``test_the_interpreter_is_stated_in_the_test_id``) —
the zero-word arm's protection is live on bash 3.2 and decorative on >= 4.4.
"""

import functools
import os
import re
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "serve_buster_ripper.sh"

# Every variable the wrapper reads; _serve_env() removes these from the inherited
# environment, so an omitted override really is "unset" for that run.
_CONSUMED = (
    "UPSTREAM", "HOST", "PORT", "STRIP_DATE", "VERBOSE", "SKIP_TLS_VERIFY",
    "DUMP_DIR", "DUMP_FORMAT", "PRESERVED_THINKING",
)

# Out-of-band channel to the shim: where it writes its argv, NUL-separated.
_ARGV_OUT = "BUSTER_RIPPER_ARGV_OUT"

_SHIM = """#!/bin/sh
printf '%s\\0' "$@" > "$BUSTER_RIPPER_ARGV_OUT"
exit 0
"""

# The exec block at lines 50-54 always contributes this trio; line 42 appends the
# thinking flag unless PRESERVED_THINKING says otherwise. The conditional appends
# at lines 37-42 land between them, in that line order.
_TRIO = ["--upstream", "http://localhost:30000", "--host", "0.0.0.0", "--port", "30001"]
_THINKING = ["--preserved-thinking"]

# Line 28's unset default, relative to the pinned HOME.
_DEFAULT_SUBPATH = "mllm/prompt-diffs"


def _pinned_home(tmp_path: Path) -> Path:
    """The fake HOME every run uses — never created, so the default stays inert."""
    return tmp_path / "home"


def _argv_file(tmp_path: Path) -> Path:
    """Where the shim records the argv the wrapper assembled."""
    return tmp_path / "argv"


def _install_shim(tmp_path: Path) -> None:
    """Put a buster-ripper stand-in ahead of the real console script on PATH."""
    shim = tmp_path / "bin" / "buster-ripper"
    shim.parent.mkdir(parents=True, exist_ok=True)
    shim.write_text(_SHIM)
    shim.chmod(0o755)


def _serve_env(tmp_path: Path, overrides: dict) -> dict:
    """Build one run's environment: shim first on PATH, fake HOME, scrubbed inputs."""
    env = {k: v for k, v in os.environ.items() if k not in _CONSUMED}
    env["PATH"] = f"{tmp_path / 'bin'}{os.pathsep}{env['PATH']}"
    env["HOME"] = str(_pinned_home(tmp_path))
    env[_ARGV_OUT] = str(_argv_file(tmp_path))
    env.update(overrides)
    return env


def _read_argv(path: Path) -> list:
    """Decode the shim's recording, keeping a legitimately empty word intact."""
    fields = path.read_bytes().split(b"\0")
    if fields and fields[-1] == b"":
        fields.pop()
    return [field.decode("utf-8") for field in fields]


@functools.lru_cache(maxsize=1)
def _bash_version() -> str:
    """First line of ``bash --version`` for the ``bash`` on PATH.

    The witnesses run the wrapper through PATH's ``bash``, so a green run only
    proves the bash 3.2 behaviors (the ``set -u`` empty-array abort that keeps
    the zero-word arm a live witness) when 3.2 is what ran. Each run records
    this as the ``bash_version`` test property, so the report says which
    interpreter the suite proved (QB:serve-script-argv/bash-pin).
    """
    out = subprocess.run(["bash", "--version"], capture_output=True, text=True, check=False)
    return out.stdout.splitlines()[0] if out.returncode == 0 and out.stdout else "unknown"


@pytest.fixture()
def serve_argv(tmp_path):
    """Run the wrapper with the given env vars and return the argv it assembled."""
    _install_shim(tmp_path)

    def run(**overrides: str) -> list:
        result = subprocess.run(
            ["bash", str(SCRIPT)],
            env=_serve_env(tmp_path, overrides),
            capture_output=True, text=True, check=False,
        )
        assert result.returncode == 0, f"wrapper aborted under bash {_bash_version()}: {result.stderr}"
        return _read_argv(_argv_file(tmp_path))

    return run


@pytest.fixture()
def matched_pattern(tmp_path, monkeypatch):
    """A glob pattern that really matches, from a cwd holding two matching files."""
    (tmp_path / "a.md").write_text("")
    (tmp_path / "b.md").write_text("")
    monkeypatch.chdir(tmp_path)
    return "*.md"


class TestDumpDirDisableForm:
    """The three DUMP_DIR states, observed on the script's assembled argv."""

    def test_unset_keeps_the_documented_default(self, tmp_path, serve_argv):
        """No DUMP_DIR → exactly one --dump-dir carrying the ${HOME} default."""
        default = f"{_pinned_home(tmp_path)}/{_DEFAULT_SUBPATH}"
        assert serve_argv() == _TRIO + ["--dump-dir", default] + _THINKING

    def test_empty_value_suppresses_the_flag(self, serve_argv):
        """The header's DUMP_DIR="" form emits no --dump-dir at all."""
        argv = serve_argv(DUMP_DIR="")
        assert "--dump-dir" not in argv
        assert argv == _TRIO + _THINKING

    def test_explicit_value_is_forwarded_unchanged(self, serve_argv):
        """A non-empty DUMP_DIR behaves exactly as it did before the fix."""
        argv = serve_argv(DUMP_DIR="/tmp/served-diffs")
        assert argv == _TRIO + ["--dump-dir", "/tmp/served-diffs"] + _THINKING

    def test_disable_leaves_the_format_peer_alone(self, serve_argv):
        """Suppressing --dump-dir does not suppress the --dump-format guard."""
        argv = serve_argv(DUMP_DIR="", DUMP_FORMAT="change-focused")
        assert argv == _TRIO + ["--dump-format", "change-focused"] + _THINKING


class TestValueWordIntegrity:
    """One argv word per forwarded value (T012): the wrapper adds no value protocol."""

    def test_path_containing_a_space_arrives_intact(self, serve_argv):
        """The counterexample: three words before, two now — and no orphan."""
        argv = serve_argv(DUMP_DIR="/tmp/my docs")
        assert argv == _TRIO + ["--dump-dir", "/tmp/my docs"] + _THINKING

    @pytest.mark.parametrize("value", [" ", "\t", " \t "])
    def test_whitespace_only_value_is_its_own_word(self, serve_argv, value):
        """`-n` is true for whitespace, so the value contributes a field instead of
        vanishing and letting the next guard's flag be read as --dump-dir's value."""
        argv = serve_argv(DUMP_DIR=value)
        assert argv == _TRIO + ["--dump-dir", value] + _THINKING

    def test_format_value_containing_a_space_is_one_word(self, serve_argv):
        """Line 41 misroutes the same way line 40 did (both renumbered by T012)."""
        argv = serve_argv(DUMP_DIR="", DUMP_FORMAT="a b")
        assert argv == _TRIO + ["--dump-format", "a b"] + _THINKING

    def test_both_values_keep_their_own_words(self, serve_argv):
        """Neither split borrows from the other: four words, in line order."""
        argv = serve_argv(DUMP_DIR="/tmp/my docs", DUMP_FORMAT="change focused")
        assert argv == _TRIO + ["--dump-dir", "/tmp/my docs",
                                "--dump-format", "change focused"] + _THINKING

    @pytest.mark.parametrize("value", ["--verbose", "--dump-format", 'q"uote', "x\ty\nz"])
    def test_value_is_forwarded_verbatim(self, serve_argv, value):
        """A value that looks like a flag or carries shell punctuation is still
        just an opaque path string — the wrapper re-cuts nothing."""
        argv = serve_argv(DUMP_DIR=value)
        assert argv == _TRIO + ["--dump-dir", value] + _THINKING

    def test_glob_metacharacters_are_not_expanded(self, serve_argv, matched_pattern):
        """The unquoted expansion also pathname-expanded, so a matching pattern in
        the cwd became one argv word per file. `matched_pattern` plants the files
        that make it match, so this witness is fail-first not vacuously true."""
        argv = serve_argv(DUMP_DIR=matched_pattern)
        assert argv == _TRIO + ["--dump-dir", matched_pattern] + _THINKING


class TestNothingAppended:
    """The zero-word arm: every append suppressed must still exec (QB:serve-script-argv/zero-word-arm)."""

    def test_all_appends_suppressed_still_reaches_the_binary(self, serve_argv):
        """The fixture fails with the interpreter's stderr if `set -u` aborts the
        wrapper before exec, which is what a bare empty-array expansion does on
        bash 3.2. The recorded argv is the always-on trio and nothing else."""
        assert serve_argv(DUMP_DIR="", DUMP_FORMAT="", PRESERVED_THINKING="0") == _TRIO


class TestInterpreterStated:
    """The zero-word arm above protects only on bash < 4.4; a green run states
    which bash it proved (QB:serve-script-argv/bash-pin)."""

    @pytest.mark.parametrize("proven_bash", [_bash_version()])
    def test_the_interpreter_is_stated_in_the_test_id(self, proven_bash):
        """The parametrize value is resolved at import — the first PATH ``bash``
        call, the same resolution the fixture's ``["bash", str(SCRIPT)]`` uses —
        so the test id itself is the statement. The assert fails only if the
        probe could not produce a bash version line to state."""
        assert re.match(r"^GNU bash, version \d+\.\d+", proven_bash)


class TestDocumentedForms:
    """README §Serve script options and the usage header — word sets unchanged."""

    @pytest.mark.parametrize(
        "case",
        [
            ({"STRIP_DATE": "1"}, ["--strip-date"]),
            ({"VERBOSE": "1"}, ["--verbose"]),
            ({"SKIP_TLS_VERIFY": "1"}, ["--skip-tls-verify"]),
            ({"STRIP_DATE": "1", "VERBOSE": "1", "SKIP_TLS_VERIFY": "1"},
             ["--strip-date", "--verbose", "--skip-tls-verify"]),
        ],
        ids=["strip-date", "verbose", "skip-tls-verify", "all-three-toggles"],
    )
    def test_boolean_forms_contribute_their_own_flag(self, serve_argv, case):
        """Each toggle is one whole word, still in line order ahead of the dump pair."""
        toggles, expected = case
        argv = serve_argv(DUMP_DIR="", **toggles)
        assert argv == _TRIO + expected + _THINKING

    def test_custom_upstream_and_port(self, serve_argv):
        """README:87 — the always-on trio carries the overrides, quoted as before."""
        argv = serve_argv(DUMP_DIR="", UPSTREAM="http://10.0.0.1:30000", PORT="30002")
        assert argv == ["--upstream", "http://10.0.0.1:30000", "--host", "0.0.0.0",
                        "--port", "30002", "--preserved-thinking"]

    def test_format_with_a_directory(self, serve_argv):
        """README:203 — the documented pair form, one flag and one value word each."""
        argv = serve_argv(DUMP_DIR="/tmp/prompt-diffs", DUMP_FORMAT="change-focused")
        assert argv == _TRIO + ["--dump-dir", "/tmp/prompt-diffs",
                                "--dump-format", "change-focused"] + _THINKING


class TestDocumentationAgreement:
    """The behavior witness above is only meaningful if the promise stays put."""

    def test_header_still_documents_the_disable_form(self):
        """Line 14 keeps advertising the empty value as the off-switch (FK-010:
        the documentation is not retracted — the script was the liar)."""
        documented = 'DUMP_DIR="" bash serve_buster_ripper.sh    # disable prompt diffing'
        assert documented in SCRIPT.read_text()

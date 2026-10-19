"""Tests for the pytest collection-exclusion config (T002).

A bare ``uv run pytest`` must stay hermetic: the ini ``norecursedirs`` value
replaces pytest's default list, so it must retain every default pattern
alongside ``qa`` — otherwise the network/signals gates under tests/qa/ would
be collected by the default run (MS-003's gate).
"""

import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"

# pytest 9's default norecursedirs (verified in _pytest/main.py addini),
# which the ini value replaces and therefore must restate.
PYTEST_DEFAULT_PATTERNS = [
    ".*", "build", "dist", "node_modules", "venv", "*.egg", "_darcs", "CVS", "{arch}",
]


def _norecursedirs() -> list[str]:
    """Read the configured norecursedirs patterns from pyproject.toml."""
    with PYPROJECT.open("rb") as f:
        data = tomllib.load(f)
    return data["tool"]["pytest"]["ini_options"]["norecursedirs"]


class TestNoRecurseDirs:
    """The exclusion keeps the default run hermetic."""

    def test_qa_is_excluded(self):
        """tests/qa/ stays out of a bare run."""
        assert "qa" in _norecursedirs()

    def test_pytest_defaults_are_retained(self):
        """Every default pattern is restated (the ini value replaces them)."""
        configured = _norecursedirs()
        for pattern in PYTEST_DEFAULT_PATTERNS:
            assert pattern in configured, f"missing default pattern: {pattern!r}"

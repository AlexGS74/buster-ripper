"""Tests for main.py's dump option wiring (T002/T003).

Pins the callback-time assignment (each option assigns its config global
during parsing, so main() gains no parameter), the closed choice set, the
documented default, and the flush-trigger defaults — including the
positivity guard on the trigger values (T006 backlog pickup: zero or
negative values are rejected, so the timer can never become a hot loop).
"""

import pytest
import click

from buster_ripper import config
from buster_ripper.main import main


@pytest.fixture()
def restore_format():
    """Restore the format global around each CLI-level test."""
    previous = config.DUMP_FORMAT
    yield
    config.DUMP_FORMAT = previous


@pytest.fixture()
def restore_flush_params():
    """Restore the flush-trigger globals around each CLI-level test."""
    previous = (config.DUMP_FLUSH_INTERVAL, config.DUMP_FLUSH_BYTES)
    yield
    config.DUMP_FLUSH_INTERVAL, config.DUMP_FLUSH_BYTES = previous


class TestDumpFormatOption:
    """--dump-format assigns config.DUMP_FORMAT at parse time."""

    def test_flag_assigns_global_at_parse_time(self, restore_format):
        """Passing the flag sets config.DUMP_FORMAT before main() runs."""
        ctx = main.make_context("buster-ripper", ["--dump-format", "change-focused"])
        try:
            assert config.DUMP_FORMAT == "change-focused"
            assert "dump_format" not in ctx.params
        finally:
            ctx.close()

    def test_default_is_documented_without_flag(self, restore_format):
        """No flag leaves the documented default in place."""
        config.DUMP_FORMAT = "self-contained"
        ctx = main.make_context("buster-ripper", [])
        try:
            assert config.DUMP_FORMAT == "documented"
            assert "dump_format" not in ctx.params
        finally:
            ctx.close()

    def test_invalid_choice_is_rejected(self, restore_format):
        """A value outside DUMP_FORMATS never reaches the global."""
        config.DUMP_FORMAT = "documented"
        with pytest.raises(click.UsageError):
            main.make_context("buster-ripper", ["--dump-format", "jsonl"])
        assert config.DUMP_FORMAT == "documented"

    def test_help_lists_every_format_and_default(self):
        """--help shows --dump-format, all three values and the default."""
        from click.testing import CliRunner

        result = CliRunner().invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "--dump-format" in result.output
        for fmt in config.DUMP_FORMATS:
            assert fmt in result.output
        assert "[default: documented]" in result.output


class TestDumpFlushOptions:
    """--dump-flush-interval/--dump-flush-bytes assign their globals at parse time."""

    def test_interval_flag_assigns_global_at_parse_time(self, restore_flush_params):
        """Passing the flag sets config.DUMP_FLUSH_INTERVAL before main() runs."""
        ctx = main.make_context("buster-ripper", ["--dump-flush-interval", "5"])
        try:
            assert config.DUMP_FLUSH_INTERVAL == 5
            assert "dump_flush_interval" not in ctx.params
        finally:
            ctx.close()

    def test_bytes_flag_assigns_global_at_parse_time(self, restore_flush_params):
        """Passing the flag sets config.DUMP_FLUSH_BYTES before main() runs."""
        ctx = main.make_context("buster-ripper", ["--dump-flush-bytes", "1048576"])
        try:
            assert config.DUMP_FLUSH_BYTES == 1048576
            assert "dump_flush_bytes" not in ctx.params
        finally:
            ctx.close()

    def test_defaults_hold_without_flags(self, restore_flush_params):
        """No flags leave 10 s / 4 MiB in place, restoring over any mutation."""
        config.DUMP_FLUSH_INTERVAL = 99
        config.DUMP_FLUSH_BYTES = 99
        ctx = main.make_context("buster-ripper", [])
        try:
            assert config.DUMP_FLUSH_INTERVAL == 10
            assert config.DUMP_FLUSH_BYTES == 4 * 1024 * 1024
        finally:
            ctx.close()

    def test_non_integer_value_is_rejected(self, restore_flush_params):
        """A non-integer trigger value never reaches the global."""
        config.DUMP_FLUSH_INTERVAL = 10
        with pytest.raises(click.UsageError):
            main.make_context("buster-ripper", ["--dump-flush-interval", "soon"])
        assert config.DUMP_FLUSH_INTERVAL == 10

    @pytest.mark.parametrize("bad", ["0", "-5"])
    def test_non_positive_interval_is_rejected(self, restore_flush_params, bad):
        """Zero/negative interval never reaches the global (no hot flush loop)."""
        config.DUMP_FLUSH_INTERVAL = 10
        with pytest.raises(click.UsageError):
            main.make_context("buster-ripper", ["--dump-flush-interval", bad])
        assert config.DUMP_FLUSH_INTERVAL == 10

    @pytest.mark.parametrize("bad", ["0", "-1"])
    def test_non_positive_bytes_is_rejected(self, restore_flush_params, bad):
        """Zero/negative byte threshold never reaches the global."""
        config.DUMP_FLUSH_BYTES = 99
        with pytest.raises(click.UsageError):
            main.make_context("buster-ripper", ["--dump-flush-bytes", bad])
        assert config.DUMP_FLUSH_BYTES == 99

    def test_help_lists_both_flush_options(self):
        """--help shows both trigger options with their defaults and range."""
        from click.testing import CliRunner

        result = CliRunner().invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "--dump-flush-interval" in result.output
        assert "--dump-flush-bytes" in result.output
        assert "[default: 10; x>=1]" in result.output
        assert f"[default: {4 * 1024 * 1024};" in result.output

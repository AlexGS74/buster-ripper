"""Unit-level context overflow behavior: error parsing, token math and the patterns.

The three route-level retry loops live in ``tests/test_context_overflow_retry.py``.
"""

from buster_ripper import config, utils


class TestContextOverflowParsing:
    """Test context overflow error parsing."""

    def test_parse_overflow_with_context_overflow_pattern(self):
        """Should parse Context overflow error message."""
        error_text = "Context overflow: 50000 > 4096"
        result = utils.parse_context_overflow_error(error_text)

        assert result is not None
        assert result.input_tokens == 50000
        assert result.ctx_limit == 4096

    def test_parse_overflow_with_different_numbers(self):
        """Should parse different number combinations."""
        error_text = "Context overflow detected: 100000 > 8192"
        result = utils.parse_context_overflow_error(error_text)

        assert result is not None
        assert result.input_tokens == 100000
        assert result.ctx_limit == 8192

    def test_parse_overflow_no_match(self):
        """Should return None when pattern doesn't match."""
        error_text = "Some other error message"
        result = utils.parse_context_overflow_error(error_text)

        assert result is None

    def test_parse_overflow_with_max_completion_tokens(self):
        """Should prefer CTX_LIMIT_RE for context limit if present."""
        error_text = "Context overflow: 50000 > 4096 max_completion_tokens 8192"
        result = utils.parse_context_overflow_error(error_text)

        assert result is not None
        assert result.input_tokens == 50000
        assert result.ctx_limit == 8192  # Uses CTX_LIMIT_RE value

    def test_context_overflow_error_attributes(self):
        """Should have correct attributes on error."""
        error = utils.ContextOverflowError("test error", 1000, 4096)

        assert str(error) == "test error"
        assert error.input_tokens == 1000
        assert error.ctx_limit == 4096


class TestTokenCalculation:
    """Test new max_completion_tokens calculation."""

    def test_calculate_normal_case(self):
        """Should calculate correct value with safety margin."""
        # ctx_limit=10000, input_tokens=5000, margin=100
        # new_max = max(256, 10000 - 5000 - 100) = 4900
        result = utils.calculate_new_max_tokens(10000, 5000)
        assert result == 4900

    def test_calculate_uses_min_tokens(self):
        """Should use MIN_COMPLETION_TOKENS when calculation falls below."""
        # ctx_limit=1000, input_tokens=900, margin=100
        # new_max = max(256, 1000 - 900 - 100) = max(256, 0) = 256
        result = utils.calculate_new_max_tokens(1000, 900)
        assert result == config.MIN_COMPLETION_TOKENS

    def test_calculate_exact_boundary(self):
        """Should handle boundary case correctly."""
        # ctx_limit=5000, input_tokens=4644, margin=100
        # new_max = max(256, 5000 - 4644 - 100) = max(256, 256) = 256
        result = utils.calculate_new_max_tokens(5000, 4644)
        assert result == 256

    def test_min_completion_tokens_constant(self):
        """Should have correct MIN_COMPLETION_TOKENS value."""
        assert config.MIN_COMPLETION_TOKENS == 256


class TestRegexPatterns:
    """Test regex patterns compile and match correctly."""

    def test_context_overflow_re_pattern(self):
        """CONTEXT_OVERFLOW_RE should match context overflow errors."""
        pattern = config.CONTEXT_OVERFLOW_RE

        # Should match
        match = pattern.search("Context overflow: 50000 > 4096")
        assert match is not None
        assert match.group(1) == "50000"
        assert match.group(2) == "4096"

    def test_context_overflow_re_various_formats(self):
        """CONTEXT_OVERFLOW_RE should handle various formats."""
        pattern = config.CONTEXT_OVERFLOW_RE

        # Various formats
        texts = [
            "Context overflow 1000>500",
            "Context overflow: 1000 > 500 tokens",
            "Error: Context overflow detected 99999 > 128000",
        ]

        for text in texts:
            match = pattern.search(text)
            assert match is not None, f"Should match: {text}"

    def test_ctx_limit_re_pattern(self):
        """CTX_LIMIT_RE should match max_completion_tokens values."""
        pattern = config.CTX_LIMIT_RE

        match = pattern.search("max_completion_tokens 4096")
        assert match is not None
        assert match.group(1) == "4096"

    def test_ctx_limit_re_various_formats(self):
        """CTX_LIMIT_RE should handle various formats."""
        pattern = config.CTX_LIMIT_RE

        texts = [
            "max_completion_tokens=8192",
            "max_completion_tokens: 4096",
            "set max_completion_tokens 16384",
        ]

        for text in texts:
            match = pattern.search(text)
            assert match is not None, f"Should match: {text}"

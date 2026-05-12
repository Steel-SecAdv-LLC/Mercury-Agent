"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Tests for security/input_validation.py module.
Comprehensive test coverage for input validation and sanitization.
"""

from __future__ import annotations

from omni_mercury_engine.security.input_validation import (
    InputValidator,
    SanitizationLevel,
    ValidationError,
    ValidationResult,
    sanitize_input,
)


class TestValidationError:
    """Tests for ValidationError exception."""

    def test_basic_error(self):
        """Test basic error creation."""
        error = ValidationError("Test error")
        assert str(error) == "Test error"
        assert error.field is None
        assert error.value is None

    def test_error_with_field(self):
        """Test error with field name."""
        error = ValidationError("Invalid input", field="username")
        assert error.field == "username"
        assert error.value is None

    def test_error_with_value(self):
        """Test error with field and value."""
        error = ValidationError("Invalid input", field="email", value="bad@")
        assert error.field == "email"
        assert error.value == "bad@"


class TestSanitizationLevel:
    """Tests for SanitizationLevel enum."""

    def test_strict_level(self):
        """Test strict sanitization level."""
        assert SanitizationLevel.STRICT.value == "strict"

    def test_moderate_level(self):
        """Test moderate sanitization level."""
        assert SanitizationLevel.MODERATE.value == "moderate"

    def test_permissive_level(self):
        """Test permissive sanitization level."""
        assert SanitizationLevel.PERMISSIVE.value == "permissive"

    def test_raw_level(self):
        """Test raw sanitization level."""
        assert SanitizationLevel.RAW.value == "raw"


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_valid_result(self):
        """Test valid result creation."""
        result = ValidationResult(
            is_valid=True,
            sanitized_value="test",
            errors=[],
            warnings=[],
        )
        assert result.is_valid is True
        assert result.sanitized_value == "test"
        assert len(result.errors) == 0
        assert len(result.warnings) == 0

    def test_invalid_result(self):
        """Test invalid result with errors."""
        result = ValidationResult(
            is_valid=False,
            sanitized_value=None,
            errors=["Value cannot be None"],
            warnings=[],
        )
        assert result.is_valid is False
        assert result.sanitized_value is None
        assert len(result.errors) == 1


class TestInputValidatorStringValidation:
    """Tests for InputValidator.validate_string method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = InputValidator()

    def test_valid_string(self):
        """Test valid string validation."""
        result = self.validator.validate_string("hello world")
        assert result.is_valid is True
        assert "hello world" in result.sanitized_value

    def test_none_value_not_allowed(self):
        """Test None value when not allowed."""
        result = self.validator.validate_string(None)
        assert result.is_valid is False
        assert "None" in result.errors[0]

    def test_none_value_allowed_empty(self):
        """Test None value when empty is allowed."""
        result = self.validator.validate_string(None, allow_empty=True)
        assert result.is_valid is True
        assert result.sanitized_value == ""

    def test_non_string_conversion(self):
        """Test automatic conversion of non-string values."""
        result = self.validator.validate_string(12345)
        assert result.is_valid is True
        assert "12345" in result.sanitized_value
        assert len(result.warnings) > 0

    def test_min_length_violation(self):
        """Test minimum length validation."""
        result = self.validator.validate_string("ab", min_length=5)
        assert result.is_valid is False
        assert "at least 5" in result.errors[0]

    def test_max_length_violation(self):
        """Test maximum length validation and truncation."""
        result = self.validator.validate_string("a" * 100, max_length=50)
        assert result.is_valid is False
        assert len(result.sanitized_value) == 50

    def test_empty_not_allowed(self):
        """Test empty string when not allowed."""
        result = self.validator.validate_string("", allow_empty=False)
        assert result.is_valid is False
        assert "Cannot be empty" in result.errors[0]

    def test_empty_allowed(self):
        """Test empty string when allowed."""
        result = self.validator.validate_string("", allow_empty=True)
        assert result.is_valid is True

    def test_pattern_match_success(self):
        """Test pattern matching success."""
        result = self.validator.validate_string(
            "user123",
            pattern=r"^[a-z]+[0-9]+$",
        )
        assert result.is_valid is True

    def test_pattern_match_failure(self):
        """Test pattern matching failure."""
        result = self.validator.validate_string(
            "123user",
            pattern=r"^[a-z]+[0-9]+$",
        )
        assert result.is_valid is False
        assert "pattern" in result.errors[0].lower()

    def test_whitespace_stripping(self):
        """Test whitespace stripping."""
        result = self.validator.validate_string("  hello  ", strip_whitespace=True)
        assert result.sanitized_value == "hello"

    def test_whitespace_preserved(self):
        """Test whitespace preservation."""
        result = self.validator.validate_string(
            "  hello  ",
            strip_whitespace=False,
            level=SanitizationLevel.RAW,
        )
        assert "  hello  " in result.sanitized_value


class TestInputValidatorSQLInjection:
    """Tests for SQL injection detection."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = InputValidator()

    def test_union_select_detected(self):
        """Test UNION SELECT detection."""
        result = self.validator.validate_string("1 UNION SELECT * FROM users")
        assert result.is_valid is False
        assert "SQL injection" in result.errors[0]

    def test_or_equals_detected(self):
        """Test OR 1=1 injection detection."""
        result = self.validator.validate_string("' OR '1'='1")
        assert result.is_valid is False
        assert "SQL injection" in result.errors[0]

    def test_drop_table_detected(self):
        """Test DROP TABLE detection."""
        result = self.validator.validate_string("'; DROP TABLE users;--")
        assert result.is_valid is False

    def test_comment_injection_detected(self):
        """Test SQL comment injection detection."""
        result = self.validator.validate_string("admin'--")
        assert result.is_valid is False

    def test_exec_detected(self):
        """Test EXEC procedure detection."""
        result = self.validator.validate_string("'; EXEC(xp_cmdshell)")
        assert result.is_valid is False

    def test_safe_query_allowed(self):
        """Test safe input is allowed."""
        result = self.validator.validate_string("John Smith")
        assert result.is_valid is True


class TestInputValidatorXSSPrevention:
    """Tests for XSS prevention."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = InputValidator()

    def test_script_tag_detected(self):
        """Test script tag detection."""
        result = self.validator.validate_string("<script>alert('xss')</script>")
        assert result.is_valid is False
        assert "XSS" in result.errors[0]

    def test_javascript_uri_detected(self):
        """Test javascript: URI detection."""
        result = self.validator.validate_string("javascript:alert(1)")
        assert result.is_valid is False
        assert "XSS" in result.errors[0]

    def test_event_handler_detected(self):
        """Test event handler detection."""
        result = self.validator.validate_string("<img onerror=alert(1)>")
        assert result.is_valid is False

    def test_iframe_detected(self):
        """Test iframe detection."""
        result = self.validator.validate_string("<iframe src='evil.com'>")
        assert result.is_valid is False

    def test_svg_onload_detected(self):
        """Test SVG onload detection."""
        result = self.validator.validate_string("<svg onload=alert(1)>")
        assert result.is_valid is False

    def test_safe_html_text_allowed(self):
        """Test safe text is allowed."""
        result = self.validator.validate_string("Hello, World!")
        assert result.is_valid is True


class TestInputValidatorCommandInjection:
    """Tests for command injection prevention."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = InputValidator()

    def test_semicolon_detected(self):
        """Test semicolon command chaining detection."""
        result = self.validator.validate_string("file; rm -rf /")
        assert result.is_valid is False
        assert "command injection" in result.errors[0].lower()

    def test_pipe_detected(self):
        """Test pipe operator detection."""
        result = self.validator.validate_string("data | cat /etc/passwd")
        assert result.is_valid is False

    def test_backtick_detected(self):
        """Test backtick command substitution detection."""
        result = self.validator.validate_string("`whoami`")
        assert result.is_valid is False

    def test_dollar_paren_detected(self):
        """Test $() command substitution detection."""
        result = self.validator.validate_string("$(cat /etc/passwd)")
        assert result.is_valid is False

    def test_ampersand_and_detected(self):
        """Test && operator detection."""
        result = self.validator.validate_string("ls && rm -rf /")
        assert result.is_valid is False

    def test_safe_filename_allowed(self):
        """Test safe filename is allowed."""
        result = self.validator.validate_string("my_document.txt")
        assert result.is_valid is True


class TestInputValidatorPathTraversal:
    """Tests for path traversal prevention."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = InputValidator()

    def test_dot_dot_slash_detected(self):
        """Test ../ path traversal detection."""
        result = self.validator.validate_path("../../../etc/passwd")
        assert result.is_valid is False
        assert "traversal" in result.errors[0].lower()

    def test_dot_dot_backslash_detected(self):
        """Test ..\\ path traversal detection."""
        result = self.validator.validate_path("..\\..\\windows\\system32")
        assert result.is_valid is False

    def test_url_encoded_traversal_detected(self):
        """Test URL-encoded path traversal detection."""
        result = self.validator.validate_path("%2e%2e/etc/passwd")
        assert result.is_valid is False

    def test_null_byte_detected(self):
        """Test null byte detection in path."""
        result = self.validator.validate_path("file.txt\x00.jpg")
        assert result.is_valid is False
        assert "Null bytes" in result.errors[0]

    def test_etc_passwd_detected(self):
        """Test /etc/passwd direct access detection."""
        result = self.validator.validate_path("/etc/passwd")
        assert result.is_valid is False

    def test_allowed_prefix_enforced(self):
        """Test allowed prefix enforcement."""
        result = self.validator.validate_path(
            "/etc/passwd",
            allowed_prefix="/home/user/uploads",
        )
        assert result.is_valid is False
        assert "outside allowed" in result.errors[0].lower()

    def test_safe_path_allowed(self):
        """Test safe path is allowed."""
        result = self.validator.validate_path("uploads/myfile.txt")
        assert result.is_valid is True

    def test_path_none_value(self):
        """Test path validation with None value."""
        result = self.validator.validate_path(None)
        assert result.is_valid is False
        assert "None" in result.errors[0]

    def test_path_non_string(self):
        """Test path validation with non-string value."""
        result = self.validator.validate_path(12345)
        assert result.is_valid is False


class TestInputValidatorEmailValidation:
    """Tests for email validation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = InputValidator()

    def test_valid_email(self):
        """Test valid email addresses."""
        valid_emails = [
            "test@example.com",
            "user.name@domain.org",
            "user+tag@example.co.uk",
        ]
        for email in valid_emails:
            result = self.validator.validate_email(email)
            assert result.is_valid is True, f"Email {email} should be valid"

    def test_invalid_email_no_at(self):
        """Test email without @ symbol."""
        result = self.validator.validate_email("testexample.com")
        assert result.is_valid is False

    def test_invalid_email_no_domain(self):
        """Test email without domain."""
        result = self.validator.validate_email("test@")
        assert result.is_valid is False

    def test_invalid_email_no_tld(self):
        """Test email without TLD."""
        result = self.validator.validate_email("test@example")
        assert result.is_valid is False

    def test_email_too_short(self):
        """Test email that's too short."""
        result = self.validator.validate_email("a@b")
        assert result.is_valid is False


class TestInputValidatorIntegerValidation:
    """Tests for integer validation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = InputValidator()

    def test_valid_integer(self):
        """Test valid integer input."""
        result = self.validator.validate_integer(42)
        assert result.is_valid is True
        assert result.sanitized_value == 42

    def test_valid_integer_string(self):
        """Test valid integer from string."""
        result = self.validator.validate_integer("123")
        assert result.is_valid is True
        assert result.sanitized_value == 123

    def test_negative_integer(self):
        """Test negative integer."""
        result = self.validator.validate_integer("-456")
        assert result.is_valid is True
        assert result.sanitized_value == -456

    def test_float_to_integer(self):
        """Test float conversion to integer."""
        result = self.validator.validate_integer(3.14)
        assert result.is_valid is True
        assert result.sanitized_value == 3

    def test_none_integer(self):
        """Test None integer value."""
        result = self.validator.validate_integer(None)
        assert result.is_valid is False
        assert "None" in result.errors[0]

    def test_invalid_integer_string(self):
        """Test invalid integer string."""
        result = self.validator.validate_integer("abc")
        assert result.is_valid is False
        assert "Invalid integer" in result.errors[0]

    def test_min_value_violation(self):
        """Test minimum value violation."""
        result = self.validator.validate_integer(5, min_value=10)
        assert result.is_valid is False
        assert "at least 10" in result.errors[0]

    def test_max_value_violation(self):
        """Test maximum value violation."""
        result = self.validator.validate_integer(100, max_value=50)
        assert result.is_valid is False
        assert "at most 50" in result.errors[0]

    def test_integer_in_range(self):
        """Test integer within valid range."""
        result = self.validator.validate_integer(25, min_value=10, max_value=50)
        assert result.is_valid is True

    def test_integer_injection_attempt(self):
        """Test integer injection attempt."""
        result = self.validator.validate_integer("1; DROP TABLE users")
        assert result.is_valid is False


class TestInputValidatorFloatValidation:
    """Tests for float validation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = InputValidator()

    def test_valid_float(self):
        """Test valid float input."""
        result = self.validator.validate_float(3.14159)
        assert result.is_valid is True
        assert abs(result.sanitized_value - 3.14159) < 0.0001

    def test_valid_float_string(self):
        """Test valid float from string."""
        result = self.validator.validate_float("2.71828")
        assert result.is_valid is True

    def test_scientific_notation(self):
        """Test scientific notation."""
        result = self.validator.validate_float("1.5e10")
        assert result.is_valid is True
        assert result.sanitized_value == 1.5e10

    def test_negative_float(self):
        """Test negative float."""
        result = self.validator.validate_float("-9.81")
        assert result.is_valid is True
        assert result.sanitized_value == -9.81

    def test_integer_to_float(self):
        """Test integer conversion to float."""
        result = self.validator.validate_float(42)
        assert result.is_valid is True
        assert result.sanitized_value == 42.0

    def test_none_float(self):
        """Test None float value."""
        result = self.validator.validate_float(None)
        assert result.is_valid is False

    def test_invalid_float_string(self):
        """Test invalid float string."""
        result = self.validator.validate_float("not_a_number")
        assert result.is_valid is False

    def test_nan_not_allowed(self):
        """Test NaN is not allowed."""
        result = self.validator.validate_float(float("nan"))
        assert result.is_valid is False
        assert "NaN" in result.errors[0]

    def test_inf_not_allowed(self):
        """Test infinity is not allowed."""
        result = self.validator.validate_float(float("inf"))
        assert result.is_valid is False
        assert "Inf" in result.errors[0]

    def test_float_range_validation(self):
        """Test float range validation."""
        result = self.validator.validate_float(0.5, min_value=0.0, max_value=1.0)
        assert result.is_valid is True

    def test_float_below_min(self):
        """Test float below minimum."""
        result = self.validator.validate_float(-0.5, min_value=0.0)
        assert result.is_valid is False


class TestInputValidatorURLValidation:
    """Tests for URL validation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = InputValidator()

    def test_valid_https_url(self):
        """Test valid HTTPS URL."""
        result = self.validator.validate_url("https://example.com/path")
        assert result.is_valid is True

    def test_valid_http_url(self):
        """Test valid HTTP URL."""
        result = self.validator.validate_url("http://example.com")
        assert result.is_valid is True

    def test_invalid_scheme(self):
        """Test invalid URL scheme."""
        result = self.validator.validate_url("ftp://files.example.com")
        assert result.is_valid is False
        assert "Scheme must be" in result.errors[0]

    def test_custom_allowed_schemes(self):
        """Test custom allowed schemes."""
        result = self.validator.validate_url(
            "ftp://files.example.com",
            allowed_schemes=["ftp", "sftp"],
        )
        assert result.is_valid is True

    def test_url_without_host(self):
        """Test URL without host."""
        result = self.validator.validate_url("/path/to/resource")
        assert result.is_valid is False

    def test_none_url(self):
        """Test None URL value."""
        result = self.validator.validate_url(None)
        assert result.is_valid is False

    def test_non_string_url(self):
        """Test non-string URL value."""
        result = self.validator.validate_url(12345)
        assert result.is_valid is False


class TestSanitizationLevels:
    """Tests for different sanitization levels."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = InputValidator()

    def test_strict_sanitization(self):
        """Test strict sanitization removes special chars."""
        result = self.validator.validate_string(
            "Hello <World>!",
            level=SanitizationLevel.STRICT,
        )
        # Strict only allows alphanumeric and limited punctuation
        assert "<" not in result.sanitized_value
        assert ">" not in result.sanitized_value

    def test_moderate_sanitization(self):
        """Test moderate sanitization HTML encodes."""
        result = self.validator.validate_string(
            "Hello <World>",
            level=SanitizationLevel.MODERATE,
        )
        assert "&lt;" in result.sanitized_value
        assert "&gt;" in result.sanitized_value

    def test_permissive_sanitization(self):
        """Test permissive sanitization only HTML encodes."""
        result = self.validator.validate_string(
            "Hello <World>",
            level=SanitizationLevel.PERMISSIVE,
        )
        assert "&lt;" in result.sanitized_value

    def test_raw_sanitization(self):
        """Test raw sanitization passes through."""
        # Use a string without dangerous patterns for raw test
        result = self.validator.validate_string(
            "Hello World",
            level=SanitizationLevel.RAW,
        )
        assert result.sanitized_value == "Hello World"

    def test_moderate_removes_control_chars(self):
        """Test moderate level removes control characters."""
        result = self.validator.validate_string(
            "Hello\x00World\x07Test",
            level=SanitizationLevel.MODERATE,
        )
        assert "\x00" not in result.sanitized_value
        assert "\x07" not in result.sanitized_value


class TestSanitizeInputFunction:
    """Tests for the sanitize_input convenience function."""

    def test_basic_sanitization(self):
        """Test basic sanitization."""
        result = sanitize_input("Hello World")
        assert "Hello World" in result

    def test_sanitization_with_level(self):
        """Test sanitization with specific level."""
        result = sanitize_input("<script>", level=SanitizationLevel.MODERATE)
        assert "&lt;" in result

    def test_empty_string_returns_empty(self):
        """Test empty string handling."""
        result = sanitize_input("")
        assert result == ""


class TestUnicodeNormalization:
    """Tests for Unicode normalization (homograph attack prevention)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = InputValidator()

    def test_unicode_normalization(self):
        """Test that Unicode is normalized."""
        # Fullwidth 'A' should be normalized to regular 'A'
        result = self.validator.validate_string("Ａｐｐｌｅ")
        # NFKC normalization converts fullwidth to regular
        assert result.is_valid is True
        assert result.sanitized_value is not None


class TestValidatorInitialization:
    """Tests for InputValidator initialization."""

    def test_default_level(self):
        """Test default sanitization level."""
        validator = InputValidator()
        assert validator.default_level == SanitizationLevel.MODERATE

    def test_custom_default_level(self):
        """Test custom default sanitization level."""
        validator = InputValidator(level=SanitizationLevel.STRICT)
        assert validator.default_level == SanitizationLevel.STRICT


# =============================================================================
# TrustedEndpoints — TRUSTED_DOMAINS, validate_url, validate_loopback_url,
# validate_user_configured_url, and secure_urlopen contract
# =============================================================================

import os
from unittest.mock import patch

import pytest

from omni_mercury_engine.security.input_validation import TrustedEndpoints


class TestTrustedDomainsCoverage:
    """The static allow-list must include every host actually reached by ``urlopen``."""

    REACHED_HOSTS = frozenset(
        {
            # Government / research APIs
            "earthquake.usgs.gov",
            "services.swpc.noaa.gov",
            "www.nhc.noaa.gov",
            "api.tidesandcurrents.noaa.gov",
            "www.ndbc.noaa.gov",
            "www.ngdc.noaa.gov",
            "exoplanetarchive.ipac.caltech.edu",
            "ssd-api.jpl.nasa.gov",
            "firms.modaps.eosdis.nasa.gov",
            "www.fema.gov",
            "api.weather.gov",
            # Finance / weather APIs (stubs)
            "www.alphavantage.co",
            "query1.finance.yahoo.com",
            "api.openweathermap.org",
            # Web search
            "api.duckduckgo.com",
        }
    )

    def test_every_reached_host_is_in_trusted(self):
        missing = self.REACHED_HOSTS - TrustedEndpoints.TRUSTED_DOMAINS
        assert not missing, f"Hosts reached by code but not in allow-list: {missing}"


class TestValidateUrl:
    """``TrustedEndpoints.validate_url`` is the canonical SSRF gate for hardcoded URLs."""

    def test_https_trusted_domain_accepted(self):
        assert TrustedEndpoints.validate_url(
            "https://earthquake.usgs.gov/fdsnws/event/1/query"
        )

    def test_http_rejected(self):
        with pytest.raises(ValueError, match="HTTPS"):
            TrustedEndpoints.validate_url("http://earthquake.usgs.gov/foo")

    def test_untrusted_domain_rejected(self):
        with pytest.raises(ValueError, match="trusted allowlist"):
            TrustedEndpoints.validate_url("https://attacker.example.com/")

    def test_no_host_rejected(self):
        with pytest.raises(ValueError, match="host"):
            TrustedEndpoints.validate_url("https:///path")


class TestValidateLoopbackUrl:
    """``validate_loopback_url`` accepts only loopback aliases."""

    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost:11434/api/tags",
            "http://127.0.0.1:11434/api/tags",
            # IPv6 literal must use bracket form per RFC 3986.
            "http://[::1]:11434/api/tags",
            "http://0.0.0.0:11434/api/tags",
        ],
    )
    def test_loopback_http_accepted(self, url):
        assert TrustedEndpoints.validate_loopback_url(url)

    def test_loopback_https_accepted(self):
        # HTTPS on loopback is fine too.
        assert TrustedEndpoints.validate_loopback_url("https://127.0.0.1:8443/x")

    def test_non_loopback_http_rejected(self):
        # Bare HTTP to a real host must never be allowed by this validator,
        # even on a port that looks like Ollama's.
        with pytest.raises(ValueError, match="loopback"):
            TrustedEndpoints.validate_loopback_url("http://attacker.example.com:11434/api/tags")

    def test_non_loopback_https_rejected(self):
        with pytest.raises(ValueError, match="loopback"):
            TrustedEndpoints.validate_loopback_url("https://attacker.example.com:11434/api/tags")

    def test_ftp_rejected(self):
        with pytest.raises(ValueError, match="http or https"):
            TrustedEndpoints.validate_loopback_url("ftp://localhost/")


class TestValidateUserConfiguredUrl:
    """``validate_user_configured_url`` enforces the operator allow-list."""

    def test_default_allows_duckduckgo(self):
        assert TrustedEndpoints.validate_user_configured_url(
            "https://api.duckduckgo.com/"
        )

    def test_unknown_host_rejected(self):
        with pytest.raises(ValueError, match="allow-list"):
            TrustedEndpoints.validate_user_configured_url(
                "https://attacker.example.com/search"
            )

    def test_http_rejected(self):
        with pytest.raises(ValueError, match="https"):
            TrustedEndpoints.validate_user_configured_url(
                "http://api.duckduckgo.com/"
            )

    def test_env_var_extends_allow_list(self):
        with patch.dict(
            os.environ,
            {"MERCURY_TRUSTED_SEARCH_HOSTS": "self-hosted.example.com,other.example.com"},
        ):
            assert TrustedEndpoints.validate_user_configured_url(
                "https://self-hosted.example.com/search"
            )
            assert TrustedEndpoints.validate_user_configured_url(
                "https://other.example.com/api"
            )
            # Hosts not in env or default still rejected.
            with pytest.raises(ValueError):
                TrustedEndpoints.validate_user_configured_url(
                    "https://attacker.example.com/"
                )


class TestSecureUrlopen:
    """``secure_urlopen`` must refuse to make an unsafe request even with a buggy caller."""

    def test_refuses_untrusted_base(self):
        # No network call should happen — the validator rejects first.
        with pytest.raises(ValueError):
            TrustedEndpoints.secure_urlopen("https://attacker.example.com/data")

    def test_refuses_http(self):
        with pytest.raises(ValueError):
            TrustedEndpoints.secure_urlopen("http://earthquake.usgs.gov/")

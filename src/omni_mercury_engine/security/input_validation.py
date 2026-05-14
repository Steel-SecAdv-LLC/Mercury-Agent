"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Production-grade input validation and sanitization.

Implements OWASP input validation guidelines:
- SQL injection prevention
- XSS prevention
- Command injection prevention
- Path traversal prevention
- Data type validation

Reference: OWASP Input Validation Cheat Sheet
https://cheatsheetseries.owasp.org/cheatsheets/Input_Validation_Cheat_Sheet.html
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar

T = TypeVar("T")


class ValidationError(Exception):
    """Input validation failed."""

    def __init__(self, message: str, field: str | None = None, value: Any = None) -> None:
        super().__init__(message)
        self.field = field
        self.value = value


class SanitizationLevel(Enum):
    """Levels of input sanitization."""

    STRICT = "strict"  # Alphanumeric only
    MODERATE = "moderate"  # Allow common punctuation
    PERMISSIVE = "permissive"  # HTML encode only
    RAW = "raw"  # No sanitization (use with caution)


@dataclass
class ValidationResult:
    """Result of input validation."""

    is_valid: bool
    sanitized_value: Any
    errors: list[str]
    warnings: list[str]


class InputValidator:
    """
    Production-grade input validator.

    Provides defense-in-depth input validation:
    1. Type checking
    2. Length limits
    3. Pattern matching (whitelist)
    4. Dangerous pattern detection (blacklist)
    5. Unicode normalization
    6. HTML encoding

    Example:
        validator = InputValidator()

        # Validate username
        result = validator.validate_string(
            user_input,
            min_length=3,
            max_length=50,
            pattern=r'^[a-zA-Z0-9_-]+$',
            field_name='username'
        )

        if not result.is_valid:
            raise ValidationError(result.errors[0])
    """

    # Dangerous patterns for various attack types
    SQL_INJECTION_PATTERNS = [
        r"(\bUNION\b\s+\bSELECT\b)",
        r"(\bOR\b\s+[\d'\"]+\s*=\s*[\d'\"]+)",
        r"(\bAND\b\s+[\d'\"]+\s*=\s*[\d'\"]+)",
        r"(;\s*\bDROP\b)",
        r"(;\s*\bDELETE\b)",
        r"(;\s*\bUPDATE\b)",
        r"(;\s*\bINSERT\b)",
        r"('--)",
        r"(\bEXEC\b\s*\()",
        r"(\bEXECUTE\b\s*\()",
        r"(\bxp_\w+)",  # SQL Server extended procedures
        r"(\bsp_\w+)",  # SQL Server stored procedures
    ]

    XSS_PATTERNS = [
        r"<script[^>]*>",
        r"</script>",
        r"javascript\s*:",
        r"vbscript\s*:",
        r"data\s*:\s*text/html",
        r"on\w+\s*=",  # Event handlers
        r"<iframe",
        r"<object",
        r"<embed",
        r"<svg[^>]*onload",
        r"expression\s*\(",  # CSS expressions
    ]

    COMMAND_INJECTION_PATTERNS = [
        r"[;&|`$]",  # Shell metacharacters
        r"\$\(",  # Command substitution
        r"`[^`]+`",  # Backtick command substitution
        r"\|\|",  # OR
        r"&&",  # AND
        r">\s*/dev/",  # Device redirect
        r"<\s*\(",  # Process substitution
    ]

    PATH_TRAVERSAL_PATTERNS = [
        r"\.\./",
        r"\.\.\\",
        r"%2e%2e[/%5c]",  # URL encoded
        r"%252e%252e",  # Double URL encoded
        r"\.\.%00",  # Null byte
        r"/etc/passwd",
        r"\\windows\\",
        r"c:\\",
    ]

    def __init__(self, level: SanitizationLevel = SanitizationLevel.MODERATE) -> None:
        """
        Initialize input validator.

        Args:
            level: Default sanitization level
        """
        self.default_level = level

    def validate_string(
        self,
        value: Any,
        min_length: int = 0,
        max_length: int = 10000,
        pattern: str | None = None,
        field_name: str = "input",
        level: SanitizationLevel | None = None,
        allow_empty: bool = False,
        strip_whitespace: bool = True,
    ) -> ValidationResult:
        """
        Validate and sanitize a string input.

        Args:
            value: Input value to validate
            min_length: Minimum string length
            max_length: Maximum string length
            pattern: Regex pattern to match (whitelist)
            field_name: Name of field for error messages
            level: Sanitization level
            allow_empty: Allow empty strings
            strip_whitespace: Strip leading/trailing whitespace

        Returns:
            ValidationResult with sanitized value
        """
        errors: list[str] = []
        warnings: list[str] = []
        level = level or self.default_level

        # Type check
        if value is None:
            if allow_empty:
                return ValidationResult(True, "", [], [])
            errors.append(f"{field_name}: Value cannot be None")
            return ValidationResult(False, None, errors, warnings)

        if not isinstance(value, str):
            try:
                value = str(value)
                warnings.append(f"{field_name}: Converted to string")
            except (TypeError, ValueError) as e:
                errors.append(f"{field_name}: Cannot convert to string: {type(e).__name__}")
                return ValidationResult(False, None, errors, warnings)

        # Unicode normalization (prevent homograph attacks)
        value = unicodedata.normalize("NFKC", value)

        # Strip whitespace
        if strip_whitespace:
            value = value.strip()

        # Length checks
        if len(value) < min_length:
            errors.append(f"{field_name}: Must be at least {min_length} characters")

        if len(value) > max_length:
            errors.append(f"{field_name}: Must be at most {max_length} characters")
            value = value[:max_length]  # Truncate

        # Empty check
        if not allow_empty and len(value) == 0:
            errors.append(f"{field_name}: Cannot be empty")

        # Pattern whitelist check
        if pattern and value and not re.match(pattern, value):
            errors.append(f"{field_name}: Does not match required pattern")

        # Dangerous pattern checks
        danger_checks = self._check_dangerous_patterns(value, field_name)
        errors.extend(danger_checks)

        # Sanitization based on level
        sanitized = self._sanitize(value, level)

        is_valid = len(errors) == 0
        return ValidationResult(is_valid, sanitized, errors, warnings)

    def validate_email(
        self,
        value: Any,
        field_name: str = "email",
    ) -> ValidationResult:
        """
        Validate email address.

        Args:
            value: Email to validate
            field_name: Field name for errors

        Returns:
            ValidationResult
        """
        # RFC 5322 compliant email pattern (simplified)
        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"

        result = self.validate_string(
            value,
            min_length=5,
            max_length=254,
            pattern=email_pattern,
            field_name=field_name,
            level=SanitizationLevel.STRICT,
        )

        return result

    def validate_integer(
        self,
        value: Any,
        min_value: int | None = None,
        max_value: int | None = None,
        field_name: str = "number",
    ) -> ValidationResult:
        """
        Validate integer input.

        Args:
            value: Value to validate
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            field_name: Field name for errors

        Returns:
            ValidationResult with integer value
        """
        errors: list[str] = []
        warnings: list[str] = []

        if value is None:
            errors.append(f"{field_name}: Value cannot be None")
            return ValidationResult(False, None, errors, warnings)

        # Convert to int
        try:
            if isinstance(value, str):
                # Prevent injection via numeric strings
                if not re.match(r"^-?\d+$", value.strip()):
                    errors.append(f"{field_name}: Invalid integer format")
                    return ValidationResult(False, None, errors, warnings)
                int_value = int(value.strip())
            elif isinstance(value, (int, float)):
                int_value = int(value)
            else:
                errors.append(f"{field_name}: Cannot convert to integer")
                return ValidationResult(False, None, errors, warnings)
        except (ValueError, OverflowError):
            errors.append(f"{field_name}: Invalid integer")
            return ValidationResult(False, None, errors, warnings)

        # Range checks
        if min_value is not None and int_value < min_value:
            errors.append(f"{field_name}: Must be at least {min_value}")

        if max_value is not None and int_value > max_value:
            errors.append(f"{field_name}: Must be at most {max_value}")

        is_valid = len(errors) == 0
        return ValidationResult(is_valid, int_value, errors, warnings)

    def validate_float(
        self,
        value: Any,
        min_value: float | None = None,
        max_value: float | None = None,
        field_name: str = "number",
    ) -> ValidationResult:
        """
        Validate float input.

        Args:
            value: Value to validate
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            field_name: Field name for errors

        Returns:
            ValidationResult with float value
        """
        errors: list[str] = []
        warnings: list[str] = []

        if value is None:
            errors.append(f"{field_name}: Value cannot be None")
            return ValidationResult(False, None, errors, warnings)

        # Convert to float
        try:
            if isinstance(value, str):
                # Prevent injection via numeric strings
                if not re.match(r"^-?\d+\.?\d*([eE][+-]?\d+)?$", value.strip()):
                    errors.append(f"{field_name}: Invalid number format")
                    return ValidationResult(False, None, errors, warnings)
                float_value = float(value.strip())
            elif isinstance(value, (int, float)):
                float_value = float(value)
            else:
                errors.append(f"{field_name}: Cannot convert to float")
                return ValidationResult(False, None, errors, warnings)
        except (ValueError, OverflowError):
            errors.append(f"{field_name}: Invalid number")
            return ValidationResult(False, None, errors, warnings)

        # Check for special values
        import math

        if math.isnan(float_value) or math.isinf(float_value):
            errors.append(f"{field_name}: Invalid numeric value (NaN/Inf not allowed)")
            return ValidationResult(False, None, errors, warnings)

        # Range checks
        if min_value is not None and float_value < min_value:
            errors.append(f"{field_name}: Must be at least {min_value}")

        if max_value is not None and float_value > max_value:
            errors.append(f"{field_name}: Must be at most {max_value}")

        is_valid = len(errors) == 0
        return ValidationResult(is_valid, float_value, errors, warnings)

    def validate_url(
        self,
        value: Any,
        allowed_schemes: list[str] | None = None,
        field_name: str = "url",
    ) -> ValidationResult:
        """
        Validate URL input.

        Args:
            value: URL to validate
            allowed_schemes: Allowed URL schemes (default: http, https)
            field_name: Field name for errors

        Returns:
            ValidationResult
        """
        from urllib.parse import urlparse

        errors: list[str] = []
        warnings: list[str] = []
        allowed_schemes = allowed_schemes or ["http", "https"]

        if value is None:
            errors.append(f"{field_name}: Value cannot be None")
            return ValidationResult(False, None, errors, warnings)

        if not isinstance(value, str):
            errors.append(f"{field_name}: Must be a string")
            return ValidationResult(False, None, errors, warnings)

        value = value.strip()

        # Parse URL
        try:
            parsed = urlparse(value)
        except ValueError:
            errors.append(f"{field_name}: Invalid URL format")
            return ValidationResult(False, None, errors, warnings)

        # Check scheme
        if parsed.scheme not in allowed_schemes:
            errors.append(f"{field_name}: Scheme must be one of: {', '.join(allowed_schemes)}")

        # Check for host
        if not parsed.netloc:
            errors.append(f"{field_name}: URL must have a host")

        # Check for dangerous patterns
        danger_checks = self._check_dangerous_patterns(value, field_name)
        errors.extend(danger_checks)

        is_valid = len(errors) == 0
        return ValidationResult(is_valid, value, errors, warnings)

    def validate_path(
        self,
        value: Any,
        allowed_prefix: str | None = None,
        field_name: str = "path",
    ) -> ValidationResult:
        """
        Validate file path input (prevent path traversal).

        Args:
            value: Path to validate
            allowed_prefix: Required path prefix (for chroot-like validation)
            field_name: Field name for errors

        Returns:
            ValidationResult
        """
        errors: list[str] = []
        warnings: list[str] = []

        if value is None:
            errors.append(f"{field_name}: Value cannot be None")
            return ValidationResult(False, None, errors, warnings)

        if not isinstance(value, str):
            errors.append(f"{field_name}: Must be a string")
            return ValidationResult(False, None, errors, warnings)

        value = value.strip()

        # Check null bytes first (security critical)
        if "\x00" in value:
            errors.append(f"{field_name}: Null bytes not allowed in path")

        # Check against allowed prefix FIRST if specified
        # This takes precedence over path traversal detection
        if allowed_prefix:
            try:
                from pathlib import Path

                normalized = str(Path(value).resolve())
                allowed_normalized = str(Path(allowed_prefix).resolve())

                if not normalized.startswith(allowed_normalized):
                    errors.append(f"{field_name}: Path outside allowed directory")
                    # Return early - path is outside allowed directory
                    return ValidationResult(False, value, errors, [])
            except (ValueError, OSError) as e:
                errors.append(f"{field_name}: Invalid path: {type(e).__name__}")

        # Check for path traversal patterns
        for pattern in self.PATH_TRAVERSAL_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                errors.append(f"{field_name}: Path traversal detected")
                break

        is_valid = len(errors) == 0
        return ValidationResult(is_valid, value, errors, warnings)

    def _check_dangerous_patterns(
        self,
        value: str,
        field_name: str,
    ) -> list[str]:
        """Check for dangerous patterns in input."""
        errors: list[str] = []

        # SQL injection
        for pattern in self.SQL_INJECTION_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                errors.append(f"{field_name}: Potential SQL injection detected")
                break

        # XSS
        for pattern in self.XSS_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                errors.append(f"{field_name}: Potential XSS detected")
                break

        # Command injection
        for pattern in self.COMMAND_INJECTION_PATTERNS:
            if re.search(pattern, value):
                errors.append(f"{field_name}: Potential command injection detected")
                break

        return errors

    def _sanitize(self, value: str, level: SanitizationLevel) -> str:
        """Sanitize value based on level."""
        if level == SanitizationLevel.RAW:
            return value

        if level == SanitizationLevel.PERMISSIVE:
            # HTML encode only
            return html.escape(value)

        if level == SanitizationLevel.MODERATE:
            # HTML encode and remove control characters
            sanitized = html.escape(value)
            # Remove control characters except newline and tab
            sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", sanitized)
            return sanitized

        if level == SanitizationLevel.STRICT:
            # Alphanumeric and limited punctuation only
            sanitized = re.sub(r"[^a-zA-Z0-9_\-@. ]", "", value)
            return sanitized

        return value


# Convenience function for quick validation
def sanitize_input(
    value: str,
    level: SanitizationLevel = SanitizationLevel.MODERATE,
) -> str:
    """
    Quick sanitization of string input.

    Args:
        value: String to sanitize
        level: Sanitization level

    Returns:
        Sanitized string
    """
    validator = InputValidator(level)
    result = validator.validate_string(value, level=level)
    return result.sanitized_value or ""


class TrustedEndpoints:
    """
    Hardcoded trusted API endpoints for external data sources with secure URL opening.

    These are constant URLs that are NOT derived from user input.
    CodeQL's taint analysis recognizes class constants as untainted sources,
    which properly resolves SSRF alerts without needing sanitization.

    Usage:
        from omni_mercury_engine.security.input_validation import TrustedEndpoints
        from omni_mercury_engine.security.safe_http import SafeHTTPClient

        body = SafeHTTPClient.get_bytes(
            TrustedEndpoints.USGS_EARTHQUAKE,
            params={"format": "geojson", "limit": "100"},
        )

    Security Note:
        - All URLs use HTTPS only
        - Query parameters should be constructed from typed function arguments
        - Never concatenate user input directly into these URLs
        - SafeHTTPClient validates scheme and domain before opening
    """

    # ==========================================================================
    # Trusted Domains Allowlist (for SSRF protection)
    # ==========================================================================
    TRUSTED_DOMAINS: frozenset[str] = frozenset(
        {
            # Government/Research APIs
            "earthquake.usgs.gov",
            "mrdata.usgs.gov",  # USGS Mineral Resources Data System
            "services.swpc.noaa.gov",
            "www.nhc.noaa.gov",
            "api.tidesandcurrents.noaa.gov",
            "www.ndbc.noaa.gov",
            "www.ngdc.noaa.gov",
            "www.ncei.noaa.gov",  # NCEI World Ocean Database
            "exoplanetarchive.ipac.caltech.edu",
            "firms.modaps.eosdis.nasa.gov",
            # FEMA Disaster Data
            "www.fema.gov",
            # Simons CMAP Ocean Data
            "simonscmap.com",
            # Copernicus Climate Data Store
            "cds.climate.copernicus.eu",
            # Academic/Research Datasets
            "archive.ics.uci.edu",
            "intrusion-detection.distrinet-research.be",  # CICIDS 2017 improved dataset
            # Weather APIs
            "archive-api.open-meteo.com",
            # NASA JPL Solar System Dynamics (CNEOS Near-Earth Objects)
            "ssd-api.jpl.nasa.gov",
            # Code/Data Repositories
            "raw.githubusercontent.com",
            "github.com",  # ADBench tabular anomaly-detection benchmarks
            # NOAA ERDDAP — Oceanographic / Climate Gridded Data
            "coastwatch.pfeg.noaa.gov",
            # USGS Water Quality Portal
            "www.waterqualitydata.us",
            # EPA Air Quality System
            "aqs.epa.gov",
            # PhysioNet — MIT-BIH Arrhythmia Database
            "physionet.org",
            # TimeEval HiDrive — SMAP/MSL/SMD mirrors
            "my.hidrive.com",
            # BATADAL — Water Network Attack Detection
            "www.batadal.net",
            # MAST — Kepler Light Curves
            "mast.stsci.edu",
            # Domain loader data sources
            "volcanoes.usgs.gov",  # USGS Volcano Hazards Program
            "volcano.si.edu",  # Smithsonian Global Volcanism Program
            "waterservices.usgs.gov",  # USGS Water Services (flood gauges)
            "water.weather.gov",  # NOAA AHPS (hydrologic prediction)
            "www.spc.noaa.gov",  # NOAA Storm Prediction Center (tornadoes)
            "api.stlouisfed.org",  # FRED (Federal Reserve Economic Data)
            "api.eia.gov",  # EIA (Energy Information Administration)
            "api.obis.org",  # OBIS (Ocean Biodiversity Information System)
            "ghoapi.azureedge.net",  # WHO Global Health Observatory
            "www.who.int",  # WHO Emergencies hub (pandemic loader)
            "maps.nccs.nasa.gov",  # NASA COOLR (landslide catalog)
            # UCR Time Series Archive (per-dataset mirror).  Hosted by
            # the time-series classification consortium that publishes
            # the UCR/UEA archive; used by ``datasets/ucr_archive.py``
            # for per-dataset downloads alongside the full-archive UCR
            # site.  Plain HTTP not required (HTTPS works).
            "www.timeseriesclassification.com",
            # CWRU Bearing Data Center (vibration / fault-diagnosis
            # benchmark).  Hosted on the Case Western Reserve University
            # engineering domain; used by ``datasets/ucr_archive.py``
            # MBA/CWRU loader.
            "engineering.case.edu",
        }
    )

    @classmethod
    def validate_url(cls, url: str) -> bool:
        """
        Validate that a URL is safe to open (HTTPS + trusted domain).

        Args:
            url: The URL to validate

        Returns:
            True if URL is safe, False otherwise

        Raises:
            ValueError: If URL is malformed or uses untrusted scheme/domain
        """
        import urllib.parse

        parsed = urllib.parse.urlparse(url)

        # Enforce HTTPS only
        if parsed.scheme != "https":
            raise ValueError(f"SSRF Protection: URL must use HTTPS scheme, got '{parsed.scheme}'")

        # Validate domain is in allowlist.  ``parsed.hostname`` strips
        # any explicit port and userinfo and unwraps IPv6 brackets, so
        # the lookup matches the bare hostnames in TRUSTED_DOMAINS even
        # when the caller passes ``https://host:443/`` or
        # ``https://[2001:db8::1]/``.
        host = parsed.hostname
        if not host:
            raise ValueError(f"SSRF Protection: URL '{url}' has no host component.")
        cls.validate_url_host(host)
        return True

    @classmethod
    def validate_url_host(cls, host: str) -> bool:
        """Validate a hostname against the TRUSTED_DOMAINS allowlist.

        Scheme-agnostic.  Used by SafeHTTPClient so the allowlist gate
        fires for both http:// (when the operator explicitly opts in
        with ``allow_http=True``) and https:// URLs.  Without this, a
        plain-HTTP dataset mirror could reach an arbitrary host with
        no allowlist check.

        Args:
            host: Hostname (no scheme, no path).  Lowercased for the
                lookup.

        Returns:
            True if the host is in TRUSTED_DOMAINS.

        Raises:
            ValueError: host is not in the allowlist.
        """
        normalised = host.lower()
        if normalised not in cls.TRUSTED_DOMAINS:
            raise ValueError(
                f"SSRF Protection: Host '{normalised}' not in trusted allowlist. "
                f"Trusted hosts: {sorted(cls.TRUSTED_DOMAINS)}"
            )
        return True

    # ==========================================================================
    # USGS - Earthquake Hazards Program
    # ==========================================================================
    USGS_EARTHQUAKE = "https://earthquake.usgs.gov/fdsnws/event/1/query"

    # ==========================================================================
    # NOAA - Space Weather Prediction Center
    # ==========================================================================
    NOAA_SWPC_BASE = "https://services.swpc.noaa.gov/json"
    NOAA_SWPC_KINDEX = "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json"
    NOAA_SWPC_XRAYS = "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json"
    NOAA_SWPC_PROTONS = (
        "https://services.swpc.noaa.gov/json/goes/primary/integral-protons-1-day.json"
    )
    NOAA_SWPC_KP_PRODUCTS = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"

    # ==========================================================================
    # NOAA - National Hurricane Center
    # ==========================================================================
    NOAA_NHC_ARCHIVE = "https://www.nhc.noaa.gov/gis/forecast/archive"
    NOAA_NHC_HURDAT2 = "https://www.nhc.noaa.gov/gis/forecast/archive/hurdat2-1851-2023-052424.txt"

    # ==========================================================================
    # NOAA - National Ocean Service / Tides
    # ==========================================================================
    NOAA_NOS_API = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"

    # ==========================================================================
    # NOAA - DART Buoy Network (Tsunami Detection)
    # ==========================================================================
    NOAA_DART_BUOY = "https://www.ndbc.noaa.gov/data/realtime2"

    # ==========================================================================
    # NOAA - Tsunami Events API
    # ==========================================================================
    NOAA_TSUNAMI_EVENTS = "https://www.ngdc.noaa.gov/hazel/hazard-service/api/v1/tsunamis/events"

    # ==========================================================================
    # NOAA - National Data Buoy Center (NDBC)
    # ==========================================================================
    NOAA_NDBC_REALTIME = "https://www.ndbc.noaa.gov/data/realtime2"

    # ==========================================================================
    # NASA - Exoplanet Archive
    # ==========================================================================
    NASA_EXOPLANET_TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

    # ==========================================================================
    # NASA - FIRMS (Fire Information for Resource Management System)
    # ==========================================================================
    NASA_FIRMS_MODIS_7D = (
        "https://firms.modaps.eosdis.nasa.gov/data/active_fire/"
        "modis-c6.1/csv/MODIS_C6_1_Global_7d.csv"
    )
    NASA_FIRMS_VIIRS_7D = (
        "https://firms.modaps.eosdis.nasa.gov/data/active_fire/"
        "viirs-i-npp/csv/VNP14IMGTDL_NRT_Global_7d.csv"
    )
    # SUOMI NPP VIIRS Collection 2 (alternative VIIRS source)
    NASA_FIRMS_VIIRS_SUOMI_7D = (
        "https://firms.modaps.eosdis.nasa.gov/data/active_fire/"
        "suomi-npp-viirs-c2/csv/SUOMI_VIIRS_C2_Global_7d.csv"
    )

    # ==========================================================================
    # Open-Meteo Weather Archive API
    # ==========================================================================
    OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"

    # ==========================================================================
    # UCI ML Repository - NSL-KDD Dataset
    # ==========================================================================
    UCI_NSL_KDD = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "kddcup99-mld/kddcup.data_10_percent.gz"
    )

    # ==========================================================================
    # GitHub - NSL-KDD Dataset Mirror (defcom17)
    # ==========================================================================
    GITHUB_NSL_KDD_TRAIN = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt"
    GITHUB_NSL_KDD_TEST = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt"

    # ==========================================================================
    # MITRE ATT&CK - Threat Intelligence
    # ==========================================================================
    MITRE_STIX = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
    # Alternative MITRE STIX data source (attack-stix-data repo)
    MITRE_STIX_DATA = (
        "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/"
        "master/enterprise-attack/enterprise-attack.json"
    )

    # ==========================================================================
    # CICIDS 2017 - Network Intrusion Detection Dataset
    # ==========================================================================
    CICIDS_DISTRINET = "https://intrusion-detection.distrinet-research.be/Dataset/dataset.zip"

    # ==========================================================================
    # OpenFEMA - Disaster Declarations API (v2)
    # ==========================================================================
    FEMA_DISASTER_DECLARATIONS = "https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries"
    FEMA_INDIVIDUAL_ASSISTANCE = "https://www.fema.gov/api/open/v2/HousingAssistanceOwners"
    FEMA_PUBLIC_ASSISTANCE = (
        "https://www.fema.gov/api/open/v2/PublicAssistanceFundedProjectsDetails"
    )

    # ==========================================================================
    # Simons CMAP - Ocean Biogeochemistry Data
    # ==========================================================================
    SIMONS_CMAP_API = "https://simonscmap.com/api"

    # ==========================================================================
    # NCEI World Ocean Database
    # ==========================================================================
    NCEI_WOD_SELECT = "https://www.ncei.noaa.gov/access/world-ocean-database-select"
    NCEI_WOD_DATA = "https://www.ncei.noaa.gov/data/oceans/wod/DATA"

    # ==========================================================================
    # Copernicus Climate Data Store - Sea Level
    # ==========================================================================
    COPERNICUS_CDS_API = "https://cds.climate.copernicus.eu/api/retrieve/v1"

    # ==========================================================================
    # USGS Mineral Resources Data System (MRData)
    # ==========================================================================
    USGS_MRDATA_GEOCHEM = "https://mrdata.usgs.gov/geochem"
    USGS_MRDATA_GEOPHYS = "https://mrdata.usgs.gov/geophysics"

    # ==========================================================================
    # NASA JPL - Center for Near Earth Object Studies (CNEOS)
    # ==========================================================================
    # Fireball API - Atmospheric impact data for bolides/fireballs
    # Data source: US Government sensors (CNEOS/JPL)
    NASA_CNEOS_FIREBALL = "https://ssd-api.jpl.nasa.gov/fireball.api"
    # Close Approach API - Near-Earth asteroid approaches
    NASA_CNEOS_CAD = "https://ssd-api.jpl.nasa.gov/cad.api"
    # Small-Body Database Query API
    NASA_SBDB_QUERY = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"
    # Sentry Impact Monitoring API - Potential future impact events
    NASA_SENTRY = "https://ssd-api.jpl.nasa.gov/sentry.api"

    # ==========================================================================
    # ADBench — Tabular Anomaly Detection Benchmarks (GitHub)
    # ==========================================================================
    # Pinned to the final ``raw.githubusercontent.com`` URL so the
    # SafeHTTPClient 3xx-rejection gate does not trip on the one-hop
    # ``github.com/.../raw/...`` -> ``raw.githubusercontent.com``
    # redirect.  ``raw.githubusercontent.com`` is in TRUSTED_DOMAINS
    # so the request goes through cleanly without a redirect.
    ADBENCH_BASE = (
        "https://raw.githubusercontent.com/Minqi824/ADBench/main/adbench/datasets/Classical/"
    )

    # ==========================================================================
    # NOAA ERDDAP — Oceanographic / Climate Gridded Data
    # ==========================================================================
    NOAA_ERDDAP_SSH = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/nesdisSSH1day.csv"
    NOAA_ERDDAP_CHL = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/nesdisVHNSQchlaDaily.csv"

    # ==========================================================================
    # NOAA NCEI — Storm Events Bulk CSV
    # ==========================================================================
    NOAA_STORM_EVENTS = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"

    # ==========================================================================
    # NOAA GSOD — Global Summary of the Day
    # ==========================================================================
    NOAA_GSOD_BASE = "https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/"

    # ==========================================================================
    # EPA Air Quality System — Daily PM2.5
    # ==========================================================================
    EPA_AQS_BASE = "https://aqs.epa.gov/aqsweb/airdata/"

    # ==========================================================================
    # USGS Water Quality Portal
    # ==========================================================================
    USGS_WATER_QUALITY = "https://www.waterqualitydata.us/data/Result/search"

    # ==========================================================================
    # PhysioNet — MIT-BIH Arrhythmia Database
    # ==========================================================================
    PHYSIONET_MITBIH = "https://physionet.org/content/mitdb/1.0.0/"

    # ==========================================================================
    # FEMA — Hazard Mitigation Grants (real API endpoint)
    # ==========================================================================
    FEMA_HAZARD_MITIGATION = "https://www.fema.gov/api/open/v2/HazardMitigationGrants"

    # ==========================================================================
    # TimeEval HiDrive — SMAP/MSL/SMD mirrors
    # ==========================================================================
    TIMEEVAL_HIDRIVE = "https://my.hidrive.com/share/ma4p8w4qqb"

    # ==========================================================================
    # BATADAL — Water Network Attack Detection
    # ==========================================================================
    BATADAL_TRAIN = "https://www.batadal.net/data/BATADAL_dataset03.csv"
    BATADAL_TEST = "https://www.batadal.net/data/BATADAL_dataset04.csv"

    # ==========================================================================
    # Domain Loader Endpoints (added for Mercury domain wiring)
    # ==========================================================================

    # USGS Volcano Hazards Program
    USGS_VOLCANO_ALERTS = "https://volcanoes.usgs.gov/vsc/api/volcanoApi/alerts"
    USGS_VOLCANO_LIST = "https://volcanoes.usgs.gov/vsc/api/volcanoApi/volcanoList"

    # USGS Water Services (flood gauge data)
    USGS_WATER_SERVICES = "https://waterservices.usgs.gov/nwis/iv/"

    # NOAA Storm Prediction Center (tornado data)
    NOAA_SPC_TORNADO_ARCHIVE = "https://www.spc.noaa.gov/wcm/data/1950-2023_actual_tornadoes.csv"
    NOAA_SPC_DAILY_REPORTS = "https://www.spc.noaa.gov/climo/reports/today.csv"

    # IBTrACS (hurricane/cyclone data)
    IBTRACS_CSV_BASE = (
        "https://www.ncei.noaa.gov/data/"
        "international-best-track-archive-for-climate-stewardship-ibtracs/"
        "v04r01/access/csv/"
    )

    # FRED (Federal Reserve Economic Data)
    FRED_OBSERVATIONS = "https://api.stlouisfed.org/fred/series/observations"

    # EIA (Energy Information Administration)
    EIA_API_BASE = "https://api.eia.gov/v2/"

    # OBIS (Ocean Biodiversity Information System)
    OBIS_OCCURRENCE = "https://api.obis.org/v3/occurrence"

    # WHO Global Health Observatory
    WHO_GHO_API = "https://ghoapi.azureedge.net/api/"

    # NASA Global Landslide Catalog (COOLR)
    NASA_COOLR = "https://maps.nccs.nasa.gov/arcgis/rest/services/" "global_landslide_catalog/"

    # PhysioNet Sepsis Challenge 2019
    PHYSIONET_SEPSIS_2019 = "https://physionet.org/content/challenge-2019/"


__all__ = [
    "InputValidator",
    "SanitizationLevel",
    "TrustedEndpoints",
    "ValidationError",
    "ValidationResult",
    "sanitize_input",
]

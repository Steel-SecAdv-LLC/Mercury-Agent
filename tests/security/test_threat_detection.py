"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisors LLC

Tests for security/threat_detection.py module.
Comprehensive test coverage for threat detection functionality.
"""

from __future__ import annotations

import time

from omni_mercury_engine.security.threat_detection import (
    BanishmentAction,
    ThreatDetector,
)


class TestBanishmentAction:
    """Tests for BanishmentAction enum."""

    def test_banish_action(self):
        """Test BANISH action value."""
        assert BanishmentAction.BANISH.value == "banish"

    def test_void_action(self):
        """Test VOID action value."""
        assert BanishmentAction.VOID.value == "void"

    def test_maintain_action(self):
        """Test MAINTAIN action value."""
        assert BanishmentAction.MAINTAIN.value == "maintain"

    def test_escalate_action(self):
        """Test ESCALATE action value."""
        assert BanishmentAction.ESCALATE.value == "escalate"


class TestThreatDetectorInitialization:
    """Tests for ThreatDetector initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        detector = ThreatDetector()
        assert detector.config == {}
        assert len(detector.sql_patterns) > 0
        assert len(detector.xss_patterns) > 0
        assert len(detector.path_traversal_patterns) > 0

    def test_initialization_with_config(self):
        """Test initialization with custom config."""
        config = {"strict_mode": True}
        detector = ThreatDetector(config=config)
        assert detector.config["strict_mode"] is True


class TestSQLInjectionDetection:
    """Tests for SQL injection detection."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = ThreatDetector()

    def test_union_select_detected(self):
        """Test UNION SELECT injection detection."""
        payload = "1 UNION SELECT * FROM users"
        result = self.detector.detect_sql_injection(payload)
        assert result["is_threat"] is True
        assert result["threat_type"] == "sql_injection"
        assert len(result["matched_patterns"]) > 0

    def test_or_equals_detected(self):
        """Test OR 1=1 injection detection."""
        payload = "' OR 1=1--"
        result = self.detector.detect_sql_injection(payload)
        assert result["is_threat"] is True

    def test_drop_table_detected(self):
        """Test DROP TABLE injection detection."""
        payload = "'; DROP TABLE users;--"
        result = self.detector.detect_sql_injection(payload)
        assert result["is_threat"] is True

    def test_comment_injection_detected(self):
        """Test SQL comment injection detection."""
        payload = "admin'--"
        result = self.detector.detect_sql_injection(payload)
        assert result["is_threat"] is True

    def test_exec_detected(self):
        """Test EXEC procedure detection."""
        payload = "'; EXEC(xp_cmdshell 'dir')"
        result = self.detector.detect_sql_injection(payload)
        assert result["is_threat"] is True

    def test_safe_input_allowed(self):
        """Test safe input is not flagged."""
        payload = "John Smith"
        result = self.detector.detect_sql_injection(payload)
        assert result["is_threat"] is False
        assert len(result["matched_patterns"]) == 0

    def test_confidence_calculation(self):
        """Test confidence is calculated correctly."""
        payload = "1 UNION SELECT * FROM users"
        result = self.detector.detect_sql_injection(payload)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_case_insensitive_detection(self):
        """Test case insensitive detection."""
        payload = "1 union SELECT * from users"
        result = self.detector.detect_sql_injection(payload)
        assert result["is_threat"] is True


class TestXSSDetection:
    """Tests for XSS attack detection."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = ThreatDetector()

    def test_script_tag_detected(self):
        """Test script tag detection."""
        payload = "<script>alert('xss')</script>"
        result = self.detector.detect_xss(payload)
        assert result["is_threat"] is True
        assert result["threat_type"] == "xss"

    def test_javascript_uri_detected(self):
        """Test javascript: URI detection."""
        payload = "javascript:alert(1)"
        result = self.detector.detect_xss(payload)
        assert result["is_threat"] is True

    def test_event_handler_detected(self):
        """Test event handler detection."""
        payload = "<img onerror=alert(1)>"
        result = self.detector.detect_xss(payload)
        assert result["is_threat"] is True

    def test_iframe_detected(self):
        """Test iframe detection."""
        payload = "<iframe src='evil.com'></iframe>"
        result = self.detector.detect_xss(payload)
        assert result["is_threat"] is True

    def test_safe_html_allowed(self):
        """Test safe text is not flagged."""
        payload = "Hello, World!"
        result = self.detector.detect_xss(payload)
        assert result["is_threat"] is False

    def test_safe_html_tags_allowed(self):
        """Test some HTML tags don't trigger XSS."""
        payload = "<p>Hello</p><br><b>World</b>"
        result = self.detector.detect_xss(payload)
        assert result["is_threat"] is False

    def test_multiple_xss_patterns(self):
        """Test multiple XSS patterns in one payload."""
        payload = "<script>alert(1)</script><iframe src='evil'>"
        result = self.detector.detect_xss(payload)
        assert result["is_threat"] is True
        assert len(result["matched_patterns"]) >= 2


class TestPathTraversalDetection:
    """Tests for path traversal detection."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = ThreatDetector()

    def test_dot_dot_slash_detected(self):
        """Test ../ path traversal detection."""
        payload = "../../../etc/passwd"
        result = self.detector.detect_path_traversal(payload)
        assert result["is_threat"] is True
        assert result["threat_type"] == "path_traversal"

    def test_dot_dot_backslash_detected(self):
        """Test ..\\ path traversal detection."""
        payload = "..\\..\\windows\\system32"
        result = self.detector.detect_path_traversal(payload)
        assert result["is_threat"] is True

    def test_url_encoded_traversal_detected(self):
        """Test URL-encoded path traversal detection."""
        payload = "%2e%2e/etc/passwd"
        result = self.detector.detect_path_traversal(payload)
        assert result["is_threat"] is True

    def test_url_encoded_backslash_detected(self):
        """Test URL-encoded backslash traversal detection."""
        payload = "%2e%2e\\windows"
        result = self.detector.detect_path_traversal(payload)
        assert result["is_threat"] is True

    def test_safe_path_allowed(self):
        """Test safe path is not flagged."""
        payload = "/home/user/documents/file.txt"
        result = self.detector.detect_path_traversal(payload)
        assert result["is_threat"] is False

    def test_relative_path_in_context(self):
        """Test relative path in normal context."""
        payload = "uploads/myfile.txt"
        result = self.detector.detect_path_traversal(payload)
        assert result["is_threat"] is False


class TestDetectAll:
    """Tests for detect_all comprehensive detection."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = ThreatDetector()

    def test_no_threats_detected(self):
        """Test clean payload returns no threats."""
        payload = "Hello, World!"
        result = self.detector.detect_all(payload)
        assert result["is_threat"] is False
        assert result["threat_count"] == 0
        assert len(result["threats"]) == 0

    def test_single_threat_detected(self):
        """Test single threat detection."""
        payload = "1 UNION SELECT * FROM users"
        result = self.detector.detect_all(payload)
        assert result["is_threat"] is True
        assert result["threat_count"] >= 1

    def test_multiple_threats_detected(self):
        """Test multiple threat detection."""
        payload = "' OR 1=1; <script>alert(1)</script>"
        result = self.detector.detect_all(payload)
        assert result["is_threat"] is True
        assert result["threat_count"] >= 2

    def test_banishment_action_returned(self):
        """Test banishment action is included in result."""
        payload = "Hello, World!"
        result = self.detector.detect_all(payload)
        assert "banishment_action" in result

    def test_banishment_with_context(self):
        """Test banishment action with context."""
        payload = "' OR 1=1--"
        context = {"timestamp": time.time(), "source_type": "user_input"}
        result = self.detector.detect_all(payload, context=context)
        assert "banishment_action" in result

    def test_all_threat_types_combined(self):
        """Test payload with all threat types."""
        payload = "1 UNION SELECT * FROM users; <script>alert(1)</script>; ../../../etc/passwd"
        result = self.detector.detect_all(payload)
        assert result["is_threat"] is True
        assert result["threat_count"] == 3
        threat_types = [t["threat_type"] for t in result["threats"]]
        assert "sql_injection" in threat_types
        assert "xss" in threat_types
        assert "path_traversal" in threat_types


class TestPasswordHashing:
    """Tests for password hashing functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = ThreatDetector()

    def test_hash_password(self):
        """Test password hashing."""
        password = "my_secure_password_123"
        hashed = ThreatDetector.hash_password(password)
        assert hashed is not None
        assert hashed != password
        assert len(hashed) > 0

    def test_verify_password_correct(self):
        """Test password verification with correct password."""
        password = "my_secure_password_123"
        hashed = ThreatDetector.hash_password(password)
        assert ThreatDetector.verify_password(password, hashed) is True

    def test_verify_password_incorrect(self):
        """Test password verification with incorrect password."""
        password = "my_secure_password_123"
        hashed = ThreatDetector.hash_password(password)
        assert ThreatDetector.verify_password("wrong_password", hashed) is False

    def test_hash_different_for_same_password(self):
        """Test that same password produces different hashes (salt)."""
        password = "my_secure_password_123"
        hash1 = ThreatDetector.hash_password(password)
        hash2 = ThreatDetector.hash_password(password)
        # Due to salting, hashes should be different
        assert hash1 != hash2

    def test_verify_pbkdf2_format(self):
        """Test verification of PBKDF2 format hash."""
        password = "test_password"
        # If bcrypt not available, hash will be in PBKDF2 format
        hashed = ThreatDetector.hash_password(password)
        assert ThreatDetector.verify_password(password, hashed) is True

    def test_verify_invalid_pbkdf2_format(self):
        """Test verification fails for invalid PBKDF2 format."""
        result = ThreatDetector.verify_password("test", "pbkdf2$invalid")
        assert result is False


class TestThreatValidity:
    """Tests for threat validity assessment."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = ThreatDetector()

    def test_no_threats_void_action(self):
        """Test empty threats list returns VOID action."""
        result = self.detector.assess_threat_validity([], {})
        assert result["is_valid"] is False
        assert result["recommended_action"] == BanishmentAction.VOID

    def test_high_confidence_threat(self):
        """Test high confidence threat assessment."""
        threats = [
            {"threat_type": "sql_injection", "confidence": 0.9},
            {"threat_type": "xss", "confidence": 0.85},
        ]
        context = {"timestamp": time.time(), "source_type": "user_input"}
        result = self.detector.assess_threat_validity(threats, context)
        assert result["is_valid"] is True
        assert result["recommended_action"] in [
            BanishmentAction.BANISH,
            BanishmentAction.ESCALATE,
        ]

    def test_low_confidence_threat(self):
        """Test low confidence threat assessment."""
        threats = [{"threat_type": "unknown", "confidence": 0.1}]
        context = {"timestamp": time.time() - 100000}  # Old timestamp
        result = self.detector.assess_threat_validity(threats, context)
        # Low confidence + old timestamp should result in lower validity
        assert isinstance(result["is_valid"], bool)

    def test_validity_includes_all_fields(self):
        """Test validity result includes all required fields."""
        threats = [{"threat_type": "sql_injection", "confidence": 0.7}]
        context = {"timestamp": time.time()}
        result = self.detector.assess_threat_validity(threats, context)
        assert "is_valid" in result
        assert "confidence" in result
        assert "temporal_relevance" in result
        assert "ethical_alignment" in result
        assert "recommended_action" in result


class TestTemporalRelevance:
    """Tests for temporal relevance evaluation."""

    def test_recent_threat_high_relevance(self):
        """Test recent threat has high relevance."""
        context = {"timestamp": time.time()}
        relevance = ThreatDetector.evaluate_temporal_relevance(context)
        assert relevance > 0.9

    def test_old_threat_low_relevance(self):
        """Test old threat has low relevance."""
        # 2 days old
        context = {"timestamp": time.time() - 172800}
        relevance = ThreatDetector.evaluate_temporal_relevance(context)
        assert relevance < 0.1

    def test_missing_timestamp_default_relevance(self):
        """Test missing timestamp returns default relevance."""
        context = {}
        relevance = ThreatDetector.evaluate_temporal_relevance(context)
        assert relevance == 0.5

    def test_relevance_bounded(self):
        """Test relevance is bounded between 0 and 1."""
        # Very old timestamp
        context = {"timestamp": time.time() - 1000000}
        relevance = ThreatDetector.evaluate_temporal_relevance(context)
        assert 0.0 <= relevance <= 1.0


class TestEthicalAlignment:
    """Tests for ethical alignment evaluation."""

    def test_user_input_higher_alignment(self):
        """Test user input source has higher alignment."""
        threats = [{"threat_type": "sql_injection"}]
        context = {"source_type": "user_input"}
        alignment = ThreatDetector._evaluate_ethical_alignment(threats, context)
        assert alignment >= 0.8

    def test_non_user_input_lower_alignment(self):
        """Test non-user input has lower base alignment."""
        threats = [{"threat_type": "unknown"}]
        context = {"source_type": "api"}
        alignment = ThreatDetector._evaluate_ethical_alignment(threats, context)
        assert alignment == 0.5

    def test_dangerous_threats_increase_alignment(self):
        """Test dangerous threat types increase alignment."""
        threats = [{"threat_type": "sql_injection"}, {"threat_type": "xss"}]
        context = {"source_type": "user_input"}
        alignment = ThreatDetector._evaluate_ethical_alignment(threats, context)
        assert alignment == 1.0

    def test_alignment_bounded(self):
        """Test alignment is bounded at 1.0."""
        threats = [
            {"threat_type": "sql_injection"},
            {"threat_type": "xss"},
            {"threat_type": "path_traversal"},
        ]
        context = {"source_type": "user_input"}
        alignment = ThreatDetector._evaluate_ethical_alignment(threats, context)
        assert alignment <= 1.0


class TestBanishmentActionDecisions:
    """Tests for banishment action decision logic."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = ThreatDetector()

    def test_escalate_for_very_high_validity(self):
        """Test ESCALATE action for very high validity threats."""
        threats = [
            {"threat_type": "sql_injection", "confidence": 1.0},
            {"threat_type": "xss", "confidence": 1.0},
        ]
        context = {"timestamp": time.time(), "source_type": "user_input"}
        result = self.detector.assess_threat_validity(threats, context)
        assert result["recommended_action"] == BanishmentAction.ESCALATE

    def test_void_for_low_validity(self):
        """Test VOID action for low validity threats."""
        threats = [{"threat_type": "unknown", "confidence": 0.1}]
        context = {"timestamp": time.time() - 200000, "source_type": "system"}
        result = self.detector.assess_threat_validity(threats, context)
        # Very low confidence + old + non-user source = low validity
        assert result["recommended_action"] in [
            BanishmentAction.VOID,
            BanishmentAction.MAINTAIN,
        ]


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = ThreatDetector()

    def test_empty_payload(self):
        """Test empty payload handling."""
        result = self.detector.detect_all("")
        assert result["is_threat"] is False

    def test_very_long_payload(self):
        """Test very long payload handling."""
        payload = "A" * 100000
        result = self.detector.detect_all(payload)
        assert result["is_threat"] is False

    def test_unicode_payload(self):
        """Test unicode payload handling."""
        payload = "你好世界 مرحبا بالعالم"
        result = self.detector.detect_all(payload)
        assert isinstance(result, dict)

    def test_special_characters(self):
        """Test special character handling."""
        payload = "!@#$%^&*()_+-=[]{}|;':\",./<>?"
        result = self.detector.detect_all(payload)
        assert isinstance(result, dict)

    def test_newlines_in_payload(self):
        """Test newlines in payload."""
        payload = "Line1\nLine2\r\nLine3"
        result = self.detector.detect_all(payload)
        assert isinstance(result, dict)

    def test_null_characters(self):
        """Test null character handling."""
        payload = "test\x00null\x00chars"
        result = self.detector.detect_all(payload)
        assert isinstance(result, dict)

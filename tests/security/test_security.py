"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.
"""

from __future__ import annotations

"""
Test security modules
"""

from omni_mercury_engine.security.encryption import SecureDataHandler
from omni_mercury_engine.security.rate_limiting import RateLimiter
from omni_mercury_engine.security.threat_detection import BanishmentAction, ThreatDetector


def test_threat_detector_initialization():
    """Test threat detector initialization"""
    detector = ThreatDetector()
    assert detector is not None


def test_threat_detector_sql_injection():
    """Test SQL injection detection"""
    detector = ThreatDetector()

    sql_payload = "SELECT * FROM users WHERE id='1' OR '1'='1'"
    result = detector.detect_sql_injection(sql_payload)

    assert "is_threat" in result
    assert isinstance(result["is_threat"], bool)
    assert result["is_threat"] is True


def test_threat_detector_xss():
    """Test XSS detection"""
    detector = ThreatDetector()

    xss_payload = "<script>alert('XSS')</script>"
    result = detector.detect_xss(xss_payload)

    assert "is_threat" in result
    assert isinstance(result["is_threat"], bool)
    assert result["is_threat"] is True


def test_threat_detector_path_traversal():
    """Test path traversal detection"""
    detector = ThreatDetector()

    path_payload = "../../etc/passwd"
    result = detector.detect_path_traversal(path_payload)

    assert "is_threat" in result
    assert isinstance(result["is_threat"], bool)
    assert result["is_threat"] is True


def test_threat_validity_assessment():
    """Test threat validity assessment"""
    import time

    detector = ThreatDetector()

    threats = [
        {
            "threat_type": "sql_injection",
            "confidence": 0.9,
        }
    ]

    context = {
        "timestamp": time.time(),
        "source_type": "user_input",
    }

    result = detector.assess_threat_validity(threats, context)
    assert "is_valid" in result
    assert isinstance(result["is_valid"], bool)
    assert "recommended_action" in result


def test_banishment_action_enum():
    """Test banishment action enum"""
    assert BanishmentAction.BANISH.value == "banish"
    assert BanishmentAction.VOID.value == "void"
    assert BanishmentAction.MAINTAIN.value == "maintain"
    assert BanishmentAction.ESCALATE.value == "escalate"


def test_secure_data_handler_sanitize():
    """Test input sanitization"""
    handler = SecureDataHandler()
    malicious_input = "<script>alert('XSS')</script>"
    sanitized = handler.sanitize_input(malicious_input)
    assert "<script>" not in sanitized
    assert "&lt;script&gt;" in sanitized


def test_secure_data_handler_encoding():
    """Test data encoding and decoding"""
    handler = SecureDataHandler()
    data = "sensitive information"

    encoded = handler.encode_data(data)
    assert encoded != data

    decoded = handler.decode_data(encoded)
    assert decoded == data.encode()


def test_rate_limiter():
    """Test rate limiting functionality"""
    limiter = RateLimiter(max_requests=5, window_seconds=60)

    client_id = "test_client"

    for i in range(5):
        assert limiter.is_allowed(client_id) is True

    assert limiter.is_allowed(client_id) is False


def test_rate_limiter_reset():
    """Test rate limiter reset"""
    limiter = RateLimiter(max_requests=2, window_seconds=1)

    client_id = "test_client"
    limiter.is_allowed(client_id)
    limiter.is_allowed(client_id)

    limiter.reset(client_id)
    assert limiter.is_allowed(client_id) is True

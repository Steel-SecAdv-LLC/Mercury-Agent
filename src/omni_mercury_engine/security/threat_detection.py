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
Cybersecurity threat detection module
Enhanced with Banish_Void_Undue threat validity assessment
"""

import hashlib
import hmac
import os
import re
import time
from enum import Enum
from typing import Any


# bcrypt is optional - provide fallback using hashlib
try:
    import bcrypt

    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False


class BanishmentAction(Enum):
    """Actions for threat handling"""

    BANISH = "banish"
    VOID = "void"
    MAINTAIN = "maintain"
    ESCALATE = "escalate"


class ThreatDetector:
    """
    Detect common security threats:
    - SQL injection
    - XSS attacks
    - Path traversal
    - Data exfiltration
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.sql_patterns = [
            r"(\bUNION\b.*\bSELECT\b)",
            r"(\bOR\b.*=.*)",
            r"(;.*\bDROP\b.*\bTABLE\b)",
            r"('.*--)",
            r"(\bEXEC\b.*\()",
        ]

        self.xss_patterns = [
            r"<script[^>]*>.*</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe",
        ]

        self.path_traversal_patterns = [
            r"\.\./",
            r"\.\.\\",
            r"%2e%2e/",
            r"%2e%2e\\",
        ]

    def detect_sql_injection(self, payload: str) -> dict[str, Any]:
        """Detect SQL injection attempts"""
        matches = []

        for pattern in self.sql_patterns:
            if re.search(pattern, payload, re.IGNORECASE):
                matches.append(pattern)

        is_threat = len(matches) > 0

        return {
            "is_threat": is_threat,
            "threat_type": "sql_injection",
            "confidence": min(len(matches) / len(self.sql_patterns), 1.0),
            "matched_patterns": matches,
        }

    def detect_xss(self, payload: str) -> dict[str, Any]:
        """Detect XSS attacks"""
        matches = []

        for pattern in self.xss_patterns:
            if re.search(pattern, payload, re.IGNORECASE):
                matches.append(pattern)

        is_threat = len(matches) > 0

        return {
            "is_threat": is_threat,
            "threat_type": "xss",
            "confidence": min(len(matches) / len(self.xss_patterns), 1.0),
            "matched_patterns": matches,
        }

    def detect_path_traversal(self, payload: str) -> dict[str, Any]:
        """Detect path traversal attempts"""
        matches = []

        for pattern in self.path_traversal_patterns:
            if re.search(pattern, payload, re.IGNORECASE):
                matches.append(pattern)

        is_threat = len(matches) > 0

        return {
            "is_threat": is_threat,
            "threat_type": "path_traversal",
            "confidence": min(len(matches) / len(self.path_traversal_patterns), 1.0),
            "matched_patterns": matches,
        }

    def detect_all(self, payload: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run all threat detection checks with banishment recommendation"""
        sql_result = self.detect_sql_injection(payload)
        xss_result = self.detect_xss(payload)
        path_result = self.detect_path_traversal(payload)

        threats = []
        if sql_result["is_threat"]:
            threats.append(sql_result)
        if xss_result["is_threat"]:
            threats.append(xss_result)
        if path_result["is_threat"]:
            threats.append(path_result)

        banishment_action = BanishmentAction.MAINTAIN
        if threats and context:
            validity_result = self.assess_threat_validity(threats, context)
            banishment_action = validity_result["recommended_action"]

        return {
            "is_threat": len(threats) > 0,
            "threats": threats,
            "threat_count": len(threats),
            "banishment_action": (
                banishment_action.value
                if isinstance(banishment_action, BanishmentAction)
                else banishment_action
            ),
        }

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using bcrypt (preferred) or PBKDF2 fallback.

        Returns:
            Hashed password string. Format depends on available library.
        """
        if BCRYPT_AVAILABLE:
            salt = bcrypt.gensalt()
            hashed = bcrypt.hashpw(password.encode(), salt)
            return str(hashed.decode())
        else:
            # Fallback to PBKDF2-SHA256 with random salt
            salt = os.urandom(16)
            key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100000)
            # Format: pbkdf2$salt_hex$key_hex
            return f"pbkdf2${salt.hex()}${key.hex()}"

    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """Verify password against hash.

        Supports both bcrypt and PBKDF2 formats.
        """
        if hashed.startswith("pbkdf2$"):
            # PBKDF2 format
            parts = hashed.split("$")
            if len(parts) != 3:
                return False
            salt = bytes.fromhex(parts[1])
            stored_key = parts[2]
            computed_key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100000)
            # Use constant-time comparison to prevent timing attacks
            return hmac.compare_digest(computed_key.hex(), stored_key)
        elif BCRYPT_AVAILABLE:
            return bool(bcrypt.checkpw(password.encode(), hashed.encode()))
        else:
            # bcrypt hash but bcrypt not available
            return False

    def assess_threat_validity(
        self,
        threats: list[dict[str, Any]],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Assess validity of detected threats to reduce false positives
        Extracted from Banish_Void_Undue Threat Engine

        Args:
            threats: List of detected threats
            context: Context information (timestamp, source, etc.)

        Returns:
            Validity assessment with recommended action
        """
        if not threats:
            return {
                "is_valid": False,
                "confidence": 0.0,
                "recommended_action": BanishmentAction.VOID,
            }

        total_confidence = sum(t.get("confidence", 0.5) for t in threats)
        avg_confidence = total_confidence / len(threats)

        temporal_relevance = self.evaluate_temporal_relevance(context)

        ethical_alignment = self._evaluate_ethical_alignment(threats, context)

        validity_score = avg_confidence * 0.5 + temporal_relevance * 0.3 + ethical_alignment * 0.2

        if validity_score > 0.8:
            action = BanishmentAction.ESCALATE
        elif validity_score > 0.5:
            action = BanishmentAction.BANISH
        elif validity_score > 0.3:
            action = BanishmentAction.MAINTAIN
        else:
            action = BanishmentAction.VOID

        return {
            "is_valid": validity_score > 0.5,
            "confidence": validity_score,
            "temporal_relevance": temporal_relevance,
            "ethical_alignment": ethical_alignment,
            "recommended_action": action,
        }

    @staticmethod
    def evaluate_temporal_relevance(context: dict[str, Any]) -> float:
        """
        Evaluate temporal relevance of threat
        Recent threats are more relevant

        Args:
            context: Context with timestamp information

        Returns:
            Temporal relevance score (0-1)
        """
        if "timestamp" not in context:
            return 0.5

        threat_time = context["timestamp"]
        current_time = time.time()

        time_diff = current_time - threat_time

        max_age = 86400

        relevance = max(0.0, 1.0 - (time_diff / max_age))

        return float(relevance)

    @staticmethod
    def _evaluate_ethical_alignment(
        threats: list[dict[str, Any]],
        context: dict[str, Any],
    ) -> float:
        """
        Evaluate ethical alignment of threat response
        Ensures survivor-first principles

        Args:
            threats: List of threats
            context: Context information

        Returns:
            Ethical alignment score (0-1)
        """
        alignment = 0.8 if context.get("source_type") == "user_input" else 0.5

        threat_types = [t.get("threat_type", "") for t in threats]
        if "sql_injection" in threat_types or "xss" in threat_types:
            alignment = min(alignment + 0.2, 1.0)

        return alignment

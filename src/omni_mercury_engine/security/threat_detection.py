# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cybersecurity threat detection module Enhanced with Banish_Void_Undue threat validity assessment."""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import time
from enum import Enum
from typing import TYPE_CHECKING, Any

# bcrypt is optional - provide fallback using hashlib
try:
    import bcrypt

    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False

if TYPE_CHECKING:
    from collections.abc import Callable


# Linear-time matching.  Every released pattern is ``a.*b`` (or a literal) and
# ``re.search`` retries it from each start, so an adversarial payload -- 20k
# quotes for ``'.*--`` -- cost quadratic time (0.23 s at 20k characters, 21.9 s
# at 200k, measured) on the engine's ``detect_security_threat`` surface.  The
# matchers below decide exactly what ``re.search(pattern, payload,
# re.IGNORECASE)`` decides (``tests/security/test_threat_detection_linear.py``
# fuzzes them against it) in one forward pass.  A pattern a caller adds to the
# lists is still searched with ``re``.

_I = re.IGNORECASE


def _in_order(*tokens: str) -> Callable[[str], bool]:
    """``t0.*t1.*...``: the tokens in order on one line (``.`` stops at a newline).

    Every token is fixed-width, so its leftmost match after a position also
    ends earliest; chaining leftmost matches is therefore exact.
    """
    compiled = [re.compile(t, _I) for t in tokens]

    def match(payload: str) -> bool:
        for line in payload.split("\n"):
            pos = 0
            for token in compiled:
                m = token.search(line, pos)
                if m is None:
                    break
                pos = m.end()
            else:
                return True
        return False

    return match


_SCRIPT_OPEN = re.compile("<script", _I)
_SCRIPT_CLOSE = re.compile("</script>", _I)


def _script_block(payload: str) -> bool:
    """``<script[^>]*>.*</script>``: a tag end, then ``</script>`` on that end's line.

    ``[^>]*`` stops at the first ">" after an opening, so the candidate tag
    ends are the first ">" after each "<script" -- increasing with the
    opening -- and every cursor only moves forward.
    """
    tag_end = -1
    close: re.Match[str] | None = None
    newline = -1
    for opening in _SCRIPT_OPEN.finditer(payload):
        if tag_end >= opening.end():
            continue  # same tag end as the previous opening
        tag_end = payload.find(">", opening.end())
        if tag_end == -1:
            return False
        end = tag_end + 1
        if close is None or close.start() < end:
            close = _SCRIPT_CLOSE.search(payload, end)
            if close is None:
                return False
        if newline != len(payload) and newline < end:
            newline = payload.find("\n", end)
            if newline == -1:
                newline = len(payload)
        if close.start() < newline:
            return True
    return False


_WORD_ASSIGN = re.compile(r"\b(\w+)\s*=")
_ON_WORD = re.compile(r"on\w", _I)


def _event_handler(payload: str) -> bool:
    """The event-handler pattern: a word run holding "on" plus a word char, then ``=``.

    ``on`` + word chars must reach the end of a word run (the next character
    is whitespace or ``=``), so the pattern holds exactly when some word run
    followed by optional whitespace and ``=`` contains "on" before its last
    character.
    """
    return any(_ON_WORD.search(m.group(1)) for m in _WORD_ASSIGN.finditer(payload))


def _literal(pattern: str) -> Callable[[str], bool]:
    compiled = re.compile(pattern, _I)
    return lambda payload: compiled.search(payload) is not None


_LINEAR_MATCHERS: dict[str, Callable[[str], bool]] = {
    r"(\bUNION\b.*\bSELECT\b)": _in_order(r"\bUNION\b", r"\bSELECT\b"),
    r"(\bOR\b.*=.*)": _in_order(r"\bOR\b", "="),
    r"(;.*\bDROP\b.*\bTABLE\b)": _in_order(";", r"\bDROP\b", r"\bTABLE\b"),
    r"('.*--)": _in_order("'", "--"),
    r"(\bEXEC\b.*\()": _in_order(r"\bEXEC\b", r"\("),
    r"<script[^>]*>.*</script>": _script_block,
    r"javascript:": _literal(r"javascript:"),
    r"on\w+\s*=": _event_handler,
    r"<iframe": _literal(r"<iframe"),
    r"\.\./": _literal(r"\.\./"),
    r"\.\.\\": _literal(r"\.\.\\"),
    r"%2e%2e/": _literal(r"%2e%2e/"),
    r"%2e%2e\\": _literal(r"%2e%2e\\"),
}


def _matches(pattern: str, payload: str) -> bool:
    """``re.search(pattern, payload, re.IGNORECASE)``, in linear time for the released patterns."""
    matcher = _LINEAR_MATCHERS.get(pattern)
    if matcher is not None:
        return matcher(payload)
    return re.search(pattern, payload, _I) is not None


class BanishmentAction(Enum):
    """Threat-*validity* verdicts from the Banish_Void_Undue assessment.

    Vocabulary note — deliberately distinct from the decision layer's
    :class:`omni_mercury_engine.decision.states.ResponseAction`:

    * ``BanishmentAction`` classifies how to *file* a detected threat report
      (validity triage of the report itself); it recommends and never
      executes anything.
    * ``ResponseAction`` (``monitor`` / ``alert`` / ``recommend_mitigation``
      / ``escalate_to_human`` / ``request_input`` / ``hold``) is the closed
      loop's bounded, non-destructive *response* vocabulary.

    The wire values do not collide (``escalate`` here vs
    ``escalate_to_human`` there), and neither enum's members may be folded
    into the other: validity triage and response planning are different
    axes.
    """

    #: High-mid validity (0.5 < score <= 0.8): treat the threat as real and
    #: remove/quarantine its subject from the trusted scope (recommendation).
    BANISH = "banish"

    #: Low validity (score <= 0.3, or no threats): discard the report as
    #: invalid/undue.
    VOID = "void"

    #: Uncertain validity (0.3 < score <= 0.5): keep the report under
    #: observation without acting on it.
    MAINTAIN = "maintain"

    #: High validity (score > 0.8): raise to a human/security operator.
    ESCALATE = "escalate"


class ThreatDetector:
    """Detect common security threats:.

    - SQL injection
    - XSS attacks
    - Path traversal
    - Data exfiltration
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the instance."""
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
        """Detect SQL injection attempts."""
        matches = []

        for pattern in self.sql_patterns:
            if _matches(pattern, payload):
                matches.append(pattern)

        is_threat = len(matches) > 0

        return {
            "is_threat": is_threat,
            "threat_type": "sql_injection",
            "confidence": min(len(matches) / len(self.sql_patterns), 1.0),
            "matched_patterns": matches,
        }

    def detect_xss(self, payload: str) -> dict[str, Any]:
        """Detect XSS attacks."""
        matches = []

        for pattern in self.xss_patterns:
            if _matches(pattern, payload):
                matches.append(pattern)

        is_threat = len(matches) > 0

        return {
            "is_threat": is_threat,
            "threat_type": "xss",
            "confidence": min(len(matches) / len(self.xss_patterns), 1.0),
            "matched_patterns": matches,
        }

    def detect_path_traversal(self, payload: str) -> dict[str, Any]:
        """Detect path traversal attempts."""
        matches = []

        for pattern in self.path_traversal_patterns:
            if _matches(pattern, payload):
                matches.append(pattern)

        is_threat = len(matches) > 0

        return {
            "is_threat": is_threat,
            "threat_type": "path_traversal",
            "confidence": min(len(matches) / len(self.path_traversal_patterns), 1.0),
            "matched_patterns": matches,
        }

    def detect_all(self, payload: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run all threat detection checks with banishment recommendation."""
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
        """Assess validity of detected threats to reduce false positives Extracted from.

        Banish_Void_Undue Threat Engine.

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
        """Evaluate temporal relevance of threat Recent threats are more relevant.

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
        """Evaluate ethical alignment of a threat response, weighting who is most exposed.

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

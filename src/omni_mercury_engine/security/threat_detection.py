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

from omni_mercury_engine.detectors.provenance import DecisionSource

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


#: Boolean-tautology injection: OR/AND followed by A = B where A equals B
#: (case-insensitively, quotes aside) or both sides are numeric -- the
#: ``' OR '1'='1`` / ``1 OR 1=1`` / blind ``OR 1=2`` family.  Prose like
#: "red or blue = fine" does not satisfy it.
_SQL_TAUT = re.compile(r"\b(?:or|and)\b\s*['\"()]*\s*([\w.]+)\s*['\"]*\s*=\s*['\"]*([\w.]+)", _I)
#: String-breakout injection: a quote, then OR/AND, with a real comparison
#: (an operand character before ``=``) somewhere after.
_SQL_QUOTE_OR = re.compile(r"['\"]\s*(?:or|and)\b", _I)
_SQL_CMP_AFTER = re.compile(r"[\w'\"]\s*=")


def _sql_or_tautology(payload: str) -> bool:
    """The tightened replacement for the retired ``(\\bOR\\b.*=.*)`` rule."""
    for m in _SQL_TAUT.finditer(payload):
        lhs, rhs = m.group(1), m.group(2)
        if lhs.lower() == rhs.lower() or (lhs.isdigit() and rhs.isdigit()):
            return True
    quote_or = _SQL_QUOTE_OR.search(payload)
    return quote_or is not None and _SQL_CMP_AFTER.search(payload, quote_or.end()) is not None


#: Event-handler injection: an ``on<letters> =`` attribute preceded by a
#: character that can split or terminate markup/attributes -- the same
#: preceding-character discipline as OWASP CRS rule 941160.  A bare query
#: parameter whose name merely starts with "on" ("onion=3") has no such
#: preceding character and no longer fires.
_XSS_ON_HANDLER = re.compile(r"""[\s"'`;/0-9=(,;]on[a-z]{2,24}\s*=""", _I)


def _xss_event_handler(payload: str) -> bool:
    """The tightened replacement for the retired ``on\\w+\\s*=`` rule."""
    return _XSS_ON_HANDLER.search(payload) is not None


#: Pattern-list keys for the two semantic matchers.  They are the rules'
#: documentation in ``matched_patterns`` output; ``_matches`` dispatches them
#: to the functions above instead of ``re.search``.
_SQL_OR_TAUTOLOGY_PATTERN = (
    "sql-or-tautology (OR/AND A=A | n=n | quote-breakout OR with comparison)"
)
_XSS_EVENT_HANDLER_PATTERN = "xss-event-handler (CRS-941160-style preceded on<letters>=)"


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
    _SQL_OR_TAUTOLOGY_PATTERN: _sql_or_tautology,
    _XSS_EVENT_HANDLER_PATTERN: _xss_event_handler,
}


def _matches(pattern: str, payload: str) -> bool:
    """``re.search(pattern, payload, re.IGNORECASE)``, in linear time for the released patterns."""
    matcher = _LINEAR_MATCHERS.get(pattern)
    if matcher is not None:
        return matcher(payload)
    return re.search(pattern, payload, _I) is not None


def _decoded_views(payload: str) -> list[str]:
    """The payload plus up to two URL-decode passes (deduplicated, in order).

    Attack payloads routinely arrive URL-encoded (``%27+OR+1%3D1``), and the
    OWASP CRS regression corpus carries them that way; WAF practice is to
    normalize before signature matching (ModSecurity ``urlDecodeUni``).  Two
    passes cover the common double-encoding evasion; a decode failure keeps
    the raw view.
    """
    views = [payload]
    current = payload
    for _ in range(2):
        try:
            from urllib.parse import unquote_plus

            decoded = unquote_plus(current)
        except (ValueError, TypeError):  # pragma: no cover - unquote is total
            break
        if decoded == current:
            break
        views.append(decoded)
        current = decoded
    return views


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

    decision_source = (
        DecisionSource(
            kind="calibrated",
            description=(
                "Signature screens measured against the OWASP Core Rule Set "
                "v4.7.0 regression corpus (tests/fixtures/crs/): recall "
                "0.180 (SQLi, 671 attack payloads), 0.182 (XSS, 137), 0.333 "
                "(traversal, 27); hit rates on the corpus's own "
                "false-positive-regression payloads 0.026 (SQLi, both "
                "residual hits genuine attack payloads scoped to other "
                "rules), 0.176 (XSS), 0.0 (traversal).  The retired rules "
                "flagged prose ('red or blue = fine') as SQLi and bare "
                "parameters ('onion=3') as XSS; the boolean-tautology and "
                "CRS-941160-style event-handler replacements do not.  This "
                "is a screen, not a WAF: obfuscated payloads the CRS "
                "engine normalises are beyond fixed signatures."
            ),
            false_alarm_rate=0.176,
            calibration_data=(
                "OWASP CRS v4.7.0 regression tests, families 942 (SQLi), "
                "941 (XSS), 930 (traversal); extraction recorded in "
                "tests/fixtures/crs/README.md, enforced by "
                "tests/security/test_threat_crs_corpus.py"
            ),
        ),
        DecisionSource(
            kind="experimental",
            description=(
                "The Banish_Void_Undue validity triage weights signature "
                "confidence with temporal relevance and an exposure "
                "heuristic (0.5/0.3/0.2) against the 0.3/0.5/0.8 action "
                "ladder -- in-code constants with no published source. "
                "Verdicts are validity triage of a report, never an "
                "executed response."
            ),
        ),
    )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the instance."""
        self.config = config or {}
        self.sql_patterns = [
            r"(\bUNION\b.*\bSELECT\b)",
            _SQL_OR_TAUTOLOGY_PATTERN,
            r"(;.*\bDROP\b.*\bTABLE\b)",
            r"('\s*--)",
            r"(\bEXEC\b.*\()",
            r"\bsleep\s*\(",
            r"\bbenchmark\s*\(",
            r"\bwaitfor\s+delay\b",
            r"\binformation_schema\b",
            r"\bxp_cmdshell\b",
            r"\bload_file\s*\(",
            r"\bextractvalue\s*\(",
            r"\bupdatexml\s*\(",
        ]

        self.xss_patterns = [
            r"<script[^>]*>.*</script>",
            r"javascript\s*:",
            _XSS_EVENT_HANDLER_PATTERN,
            r"<iframe",
            r"<svg\b",
            r"<embed\b",
            r"<object\b",
            r"srcdoc\s*=",
            r"vbscript\s*:",
            r"\bdocument\.(cookie|location|write)",
            r"\bexpression\s*\(",
        ]

        self.path_traversal_patterns = [
            r"\.\./",
            r"\.\.\\",
            r"%2e%2e/",
            r"%2e%2e\\",
            r"\.\.%2f",
            r"\.\.%5c",
            r"%252e%252e",
            r"%c0%ae",
            r"/etc/(passwd|shadow|group)\b",
            r"\\windows\\(system32|win\.ini)",
            r"\bboot\.ini\b",
            r"/proc/self/environ",
        ]

    def detect_sql_injection(self, payload: str) -> dict[str, Any]:
        """Detect SQL injection attempts."""
        matches = []

        views = _decoded_views(payload)
        for pattern in self.sql_patterns:
            if any(_matches(pattern, view) for view in views):
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

        views = _decoded_views(payload)
        for pattern in self.xss_patterns:
            if any(_matches(pattern, view) for view in views):
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

        views = _decoded_views(payload)
        for pattern in self.path_traversal_patterns:
            if any(_matches(pattern, view) for view in views):
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

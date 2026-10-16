# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Security Intelligence Fusion Module.

Comprehensive integration of multi-source intelligence disciplines for all-source
threat anomaly detection. Integrates OSINT, COMINT, HUMINT, GEOINT, IMINT, SIGINT,
ELINT, MASINT, CYBINT, FININT, and specialized intelligence modalities for unified
threat assessment and early warning.

Key Features:
- Multi-INT fusion with neurosymbolic correlation
- All-source intelligence synthesis
- Cryptanalysis-assisted pattern detection
- Operations security (OPSEC) anomaly detection
- Temporal threat progression modeling
- Golden ratio optimized fusion weights
- O(n) complexity for real-time threat assessment

Intelligence Disciplines:
- OSINT: Open-source intelligence (media, social, databases)
- COMINT: Communications intelligence (phone, email, radio)
- HUMINT: Human intelligence (interviews, espionage)
- GEOINT: Geospatial intelligence (location, terrain)
- IMINT: Imagery intelligence (satellite, drone)
- SIGINT: Signals intelligence (electronic intercepts)
- ELINT: Electronic intelligence (radar, sensors)
- MASINT: Measurement & signature intelligence (technical signatures)
- CYBINT: Cyber intelligence (network, malware)
- FININT: Financial intelligence (money laundering, terrorism financing)

⚠️ SIMULATION-BASED: Research/development tool for threat analysis patterns.
Operational deployment requires security clearance and legal authorization.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from omni_mercury_engine.core.centralized_constants import INTEL_MERIT
from omni_mercury_engine.core.exceptions import InputModalityError
from omni_mercury_engine.security.counterintelligence import (
    CounterintelligenceAssessment,
    CounterintelligenceAssessor,
)
from omni_mercury_engine.security.merit_attempt_log import MeritAttemptLog, log_from_env
from omni_mercury_engine.security.safe_torch import safe_torch_load

# Fusion weights replace evidence scoring only on measured merit: at least
# this many distinct labelled held-out cases, and a one-sided exact McNemar
# test on the discordant pairs below this level (on the first attempt against
# a held-out set; see ``merit_alpha_for_attempt``).
MIN_HELDOUT_CASES = INTEL_MERIT.MIN_HELDOUT_CASES
MERIT_ALPHA = INTEL_MERIT.MERIT_ALPHA


def merit_alpha_for_attempt(attempt: int) -> float:
    """Significance level for the ``attempt``-th merit test against one held-out set.

    Alpha spending over repeated attempts: ``MERIT_ALPHA * 2**-(attempt - 1)``.
    The first attempt keeps the published level; each retry against the same
    held-out set halves it. By the union bound the chance that *any* of an
    unlimited sequence of attempts admits weights no better than evidence
    scoring is at most ``2 * MERIT_ALPHA``. It is not held at ``MERIT_ALPHA``
    because the first attempt spends all of it.

    Args:
        attempt: 1-based attempt number against the same held-out fingerprint.

    Returns:
        The p-value threshold the attempt must beat.
    """
    if attempt < 1:
        raise ValueError(f"attempt must be at least 1, got {attempt}")
    return MERIT_ALPHA * 2.0 ** -(attempt - 1)


def _canonical(value: Any) -> Any:
    """Deterministic JSON-friendly form of a held-out case, for deduplication."""
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_canonical(item) for item in value]
    if isinstance(value, set | frozenset):
        return sorted(json.dumps(_canonical(item), sort_keys=True) for item in value)
    if isinstance(value, np.ndarray):
        return _canonical(value.tolist())
    if isinstance(value, np.generic):
        return _canonical(value.item())
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return repr(value)


def _canonical_case(reports: Any, level: int) -> str:
    """Canonical serialisation of one ``(intel_reports, level)`` held-out case."""
    return json.dumps(
        [_canonical(reports), level], sort_keys=True, separators=(",", ":"), allow_nan=True
    )


class IntelligenceDiscipline(Enum):
    """Intelligence collection disciplines."""

    OSINT = "open_source"
    COMINT = "communications"
    HUMINT = "human"
    GEOINT = "geospatial"
    IMINT = "imagery"
    SIGINT = "signals"
    ELINT = "electronic"
    MASINT = "measurement_signature"
    CYBINT = "cyber"
    FININT = "financial"
    CRYPTANALYSIS = "cryptanalysis"
    METEOROLOGICAL = "meteorological"
    TRAFFIC_ANALYSIS = "traffic"


class ThreatLevel(Enum):
    """Threat assessment levels."""

    LOW = 1
    MODERATE = 2
    SUBSTANTIAL = 3
    SEVERE = 4
    CRITICAL = 5


@dataclass
class IntelligenceFusionResult:
    """Result from multi-INT fusion analysis."""

    threat_detected: bool
    threat_level: str
    confidence: float
    risk_score: float

    primary_intel_sources: list[str] = field(default_factory=list)
    corroborating_sources: list[str] = field(default_factory=list)

    threat_indicators: list[str] = field(default_factory=list)
    temporal_patterns: dict[str, Any] = field(default_factory=dict)
    geospatial_context: dict[str, Any] = field(default_factory=dict)

    recommended_actions: list[str] = field(default_factory=list)
    collection_priorities: list[str] = field(default_factory=list)

    neurosymbolic_assessment: dict[str, Any] | None = None
    cryptographic_indicators: dict[str, Any] | None = None
    # Source-reliability, independence, coordination, fabrication and
    # impossible-travel screening of the reports (security/counterintelligence.py).
    counterintelligence: dict[str, Any] | None = None
    # Why report fields or whole reports did not count toward the decision
    # (missing, non-finite or out-of-range scores, zero screening weight),
    # and why the fusion network deferred to evidence scoring, if it did.
    evidence_notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FusionMerit:
    """Held-out comparison of candidate fusion weights against evidence scoring.

    ``network_only`` and ``evidence_only`` are the discordant pairs (cases
    exactly one method classified correctly); ``p_value`` is the one-sided
    exact McNemar probability of at least ``network_only`` network wins among
    them if both methods were equally accurate. ``n_cases`` counts distinct
    cases (exact duplicates by canonical serialisation count once);
    ``n_supplied`` is how many were passed in. ``attempt`` is the 1-based
    number of merit tests this engine has run against the same held-out set
    and ``alpha`` the level that attempt had to beat
    (:func:`merit_alpha_for_attempt`).
    """

    n_cases: int
    network_accuracy: float
    evidence_accuracy: float
    network_only: int
    evidence_only: int
    p_value: float
    n_supplied: int | None = None
    attempt: int = 1
    alpha: float = MERIT_ALPHA

    @property
    def admitted(self) -> bool:
        """Whether the network is significantly more accurate than the evidence path."""
        return self.p_value < self.alpha


@dataclass(frozen=True)
class MeritAttempt:
    """One merit test run by :meth:`IntelligenceFusionEngine.load_trained_weights`.

    Attributes:
        fingerprint: SHA-256 of the sorted canonical serialisations of the
            distinct held-out cases; retries against the same set share it.
        checkpoint: The checkpoint path that was tested.
        merit: The measured merit, including the attempt number and level.
    """

    fingerprint: str
    checkpoint: str
    merit: FusionMerit


@dataclass(frozen=True)
class ReliabilityFit:
    """Per-discipline reliabilities learned from adjudicated training cases.

    Attributes:
        table: The reliability table after the fit (learned where a
            discipline had enough reports, the previous value elsewhere).
        previous_table: The table the fit started from.
        fitted: Disciplines whose reliability was learned.
        n_cases: Training cases with at least one scored report.
        shrinkage: Penalty weight chosen by cross-validation.
        cv_error_learned: Cross-validated mean absolute threat-level error
            of evidence scoring with the learned table.
        cv_error_previous: The same error with the previous table, on the
            same folds.
        installed: Whether the engine now uses ``table``.
        reason: Why it was or was not installed.
    """

    table: dict[str, float]
    previous_table: dict[str, float]
    fitted: tuple[str, ...]
    n_cases: int
    shrinkage: float
    cv_error_learned: float
    cv_error_previous: float
    installed: bool
    reason: str


def _validated_reliability_table(table: Mapping[str, Any]) -> dict[str, float]:
    """A reliability table keyed by discipline value, each in the open interval (0, 1).

    Raises:
        ValueError: An unknown discipline or a value outside (0, 1).
    """
    known = {d.value for d in IntelligenceDiscipline}
    out: dict[str, float] = {}
    for key, value in table.items():
        if key not in known:
            raise ValueError(f"reliability table names unknown discipline {key!r}")
        number = float(value)
        if not (math.isfinite(number) and 0.0 < number < 1.0):
            raise ValueError(f"reliability of {key!r} must lie in (0, 1), got {value!r}")
        out[key] = number
    return out


def _mcnemar_one_sided(wins: int, losses: int) -> float:
    """Exact one-sided McNemar p-value: P(X >= wins), X ~ Binomial(wins + losses, 1/2)."""
    n = wins + losses
    if n == 0:
        return 1.0
    # Integer arithmetic throughout: exact, and no float overflow at large n.
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / (1 << n)


class AllSourceFusionNetwork(nn.Module):
    """Neural network for all-source intelligence fusion.

    Implements per-discipline encoders feeding multi-head attention across INT
    disciplines, with an optional temporal LSTM branch. The network starts
    untrained; ``IntelligenceFusionEngine`` only routes decisions through it
    after ``load_trained_weights`` has measured a checkpoint beating evidence
    scoring on held-out cases.
    """

    def __init__(
        self, input_dim: int = 128, num_int_types: int = len(IntelligenceDiscipline)
    ) -> None:
        """Initialize the instance.

        Raises:
            ValueError: If ``num_int_types`` disagrees with
                ``IntelligenceDiscipline`` -- the per-discipline encoders are
                built from that enum, so a divergent count would silently
                mis-size every downstream layer.
        """
        super().__init__()
        if num_int_types != len(IntelligenceDiscipline):
            raise ValueError(
                f"num_int_types={num_int_types} does not match the "
                f"{len(IntelligenceDiscipline)} IntelligenceDiscipline members "
                "the per-discipline encoders are built from"
            )

        # Capacity schedule: expand-then-contract around a fixed 1.618 factor.
        # The factor is an arbitrary width choice (it carries no optimality
        # claim); it is kept verbatim so layer shapes — and therefore any
        # trained state_dict — stay stable across releases.
        expansion = 1.618
        hidden_1 = int(input_dim * expansion)
        hidden_2 = int(hidden_1 * expansion)
        # Round to a (nonzero) multiple of the discipline count so the
        # cross-INT attention's embed_dim stays divisible by its head count.
        hidden_3 = max(1, round(int(hidden_2 / expansion) / num_int_types)) * num_int_types

        encoder_in = input_dim // num_int_types
        encoder_out = hidden_1 // num_int_types
        fused_dim = encoder_out * num_int_types

        self.int_encoders = nn.ModuleDict(
            {
                discipline.value: nn.Sequential(
                    nn.Linear(encoder_in, encoder_out),
                    nn.LayerNorm(encoder_out),
                    nn.ReLU(),
                    nn.Dropout(0.15),
                )
                for discipline in IntelligenceDiscipline
            }
        )

        self.fusion_encoder = nn.Sequential(
            nn.Linear(fused_dim, hidden_2),
            nn.LayerNorm(hidden_2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_2, hidden_3),
            nn.LayerNorm(hidden_3),
            nn.ReLU(),
        )

        self.cross_int_attention = nn.MultiheadAttention(
            embed_dim=hidden_3, num_heads=num_int_types, dropout=0.1, batch_first=True
        )

        self.temporal_lstm = nn.LSTM(
            input_size=hidden_3,
            hidden_size=hidden_3 // 2,
            num_layers=2,
            batch_first=True,
            dropout=0.15,
        )

        self.threat_classifier = nn.Sequential(
            nn.Linear(hidden_3 + hidden_3 // 2, hidden_3),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_3, 5),
        )

        self.confidence_head = nn.Sequential(nn.Linear(hidden_3, 1), nn.Sigmoid())

    def forward(
        self,
        int_features: dict[str, torch.Tensor],
        temporal_sequence: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through all-source fusion network.

        Args:
            int_features: Dictionary of features per INT discipline
            temporal_sequence: Optional temporal threat progression

        Returns:
            Tuple of (threat_class, confidence, attention_weights)
        """
        encoded_ints = []

        for discipline, encoder in self.int_encoders.items():
            if discipline in int_features:
                encoded = encoder(int_features[discipline])
                encoded_ints.append(encoded)

        if not encoded_ints:
            batch_size = 1
            return (
                torch.zeros(batch_size, 5),
                torch.zeros(batch_size, 1),
                torch.zeros(batch_size, 1),
            )

        fused = torch.cat(encoded_ints, dim=-1)

        encoded = self.fusion_encoder(fused)

        encoded_seq = encoded.unsqueeze(1)
        attended, attention_weights = self.cross_int_attention(
            encoded_seq, encoded_seq, encoded_seq
        )
        attended = attended.squeeze(1)

        temporal_features = torch.zeros_like(attended[:, : attended.shape[1] // 2])
        if temporal_sequence is not None:
            lstm_out, _ = self.temporal_lstm(temporal_sequence)
            temporal_features = lstm_out[:, -1, :]

        combined = torch.cat([attended, temporal_features], dim=-1)

        threat_logits = self.threat_classifier(combined)
        confidence = self.confidence_head(attended)

        return threat_logits, confidence, attention_weights.mean(dim=1)


class IntelligenceFusionEngine:
    """All-Source Intelligence Fusion Engine.

    Synthesizes multi-INT inputs for comprehensive threat assessment with neurosymbolic reasoning
    and cryptanalysis integration.
    """

    def __init__(
        self,
        enable_neurosymbolic: bool = True,
        enable_cryptanalysis: bool = True,
        counterintelligence: bool = True,
        merit_log: MeritAttemptLog | str | Path | None = None,
    ):
        """Initialize intelligence fusion engine.

        Args:
            enable_neurosymbolic: Enable symbolic threat reasoning
            enable_cryptanalysis: Enable pattern-based cryptanalysis
            counterintelligence: Screen reports before evidence scoring:
                derivative and circular reports lose their weight,
                corroboration counts independent origins rather than
                disciplines, and flagged reports are down-weighted. Reports
                without source, provenance, time or location fields pass
                through unchanged.
            merit_log: Where merit-test attempts are counted across restarts
                (:class:`~omni_mercury_engine.security.merit_attempt_log.MeritAttemptLog`
                or its path). ``None`` uses ``$MERCURY_MERIT_ATTEMPT_LOG`` when
                set, else counts on this instance only.
        """
        self.logger = logging.getLogger(__name__)
        self.enable_neurosymbolic = enable_neurosymbolic
        self.enable_cryptanalysis = enable_cryptanalysis

        # The fusion network starts with random weights.  Decisions route
        # through it only once ``load_trained_weights`` has measured a
        # checkpoint beating evidence scoring on held-out cases; until then
        # ``fuse_intelligence`` scores from the reported evidence.
        self.fusion_network = AllSourceFusionNetwork(input_dim=128)
        self._trained = False
        self.merit: FusionMerit | None = None
        self._merit_attempts: list[MeritAttempt] = []
        if merit_log is None:
            self._merit_log: MeritAttemptLog | None = log_from_env()
        elif isinstance(merit_log, MeritAttemptLog):
            self._merit_log = merit_log
        else:
            self._merit_log = MeritAttemptLog(merit_log)

        self.threat_knowledge_base = self._initialize_threat_kb()

        self.int_reliability_scores = self._initialize_reliability_scores()

        self.counterintelligence: CounterintelligenceAssessor | None = (
            CounterintelligenceAssessor.from_engine(self) if counterintelligence else None
        )

        self.logger.info(
            f"Intelligence Fusion Engine initialized with {len(IntelligenceDiscipline)} disciplines"
        )

    @property
    def merit_attempts(self) -> tuple[MeritAttempt, ...]:
        """Every merit test this engine has run, admitted or refused, oldest first."""
        return tuple(self._merit_attempts)

    def load_trained_weights(
        self, path: str | Path, heldout: Sequence[tuple[dict[str, Any], int]]
    ) -> FusionMerit:
        """Install fusion weights only if they measurably beat evidence scoring.

        Matching tensor shapes prove nothing: a freshly initialised network
        has them. The candidate is scored against the evidence path on
        labelled cases it was not trained on, and replaces it only when it is
        significantly more accurate (one-sided exact McNemar on the
        discordant pairs). On refusal the engine keeps scoring from evidence.

        Both methods are measured exactly as they decide in
        :meth:`fuse_intelligence`: on the same counterintelligence-screened
        inputs, with the network through :meth:`_trained_assessment`, and
        without temporal context. Held-out cases carry no temporal context,
        so admitted weights never decide through the temporal LSTM branch
        (``fuse_intelligence`` does not feed it); the branch is unmeasured.

        What the gate enforces, and what it does not:

        * Exact duplicate cases (same canonical serialisation of reports and
          label) count once, both toward ``MIN_HELDOUT_CASES`` and in the
          McNemar test, whose pairs must be independent. Near duplicates
          (one field changed) are not detected.
        * Every attempt is recorded on :attr:`merit_attempts`. The k-th
          attempt against the same held-out fingerprint must beat
          :func:`merit_alpha_for_attempt` (``MERIT_ALPHA`` halved per retry),
          bounding the false-admission probability over unlimited retries by
          ``2 * MERIT_ALPHA``. With a merit log (``merit_log`` or
          ``$MERCURY_MERIT_ATTEMPT_LOG``) the count survives restarts and is
          shared by every engine using that log; the attempt is reserved in
          it before any case is scored, so an interrupted test still counts,
          and a log that does not verify refuses the weights. Without one the
          count lives on this engine instance only. A held-out set changed by
          one case starts from attempt 1 either way. A checkpoint that fails
          to load is not a merit test and is not counted.

        Args:
            path: Checkpoint file holding the network ``state_dict`` (either
                bare, or under a ``"state_dict"`` key).
            heldout: Pairs of ``(intel_reports, true ThreatLevel value)``, at
                least ``MIN_HELDOUT_CASES`` of them distinct.

        Returns:
            The measured merit that admitted the weights (also kept on
            ``self.merit``).

        Raises:
            ValueError: Fewer than ``MIN_HELDOUT_CASES`` distinct cases, or a
                label that is not a ``ThreatLevel`` value.
            RuntimeError: The checkpoint's tensor shapes do not match the
                network, or the candidate is not significantly better.
        """
        supplied = list(heldout)
        distinct: dict[str, tuple[dict[str, Any], int]] = {}
        for reports, label in supplied:
            level = ThreatLevel(label).value
            distinct.setdefault(_canonical_case(reports, level), (reports, level))
        cases = list(distinct.values())
        if len(cases) < MIN_HELDOUT_CASES:
            raise ValueError(
                f"{len(cases)} distinct held-out cases ({len(supplied)} supplied; exact "
                f"duplicates count once); at least {MIN_HELDOUT_CASES} are needed to "
                "measure fusion weights against evidence scoring"
            )
        labels = np.array([level for _, level in cases])
        fingerprint = hashlib.sha256("\n".join(sorted(distinct)).encode()).hexdigest()

        checkpoint = safe_torch_load(Path(path), map_location="cpu")
        state = (
            checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        )
        # A checkpoint trained with a learned reliability table carries it:
        # the table scales the network's features, so the weights are only
        # meaningful (and only measured, and only served) together with it.
        carried = checkpoint.get("int_reliability_scores") if isinstance(checkpoint, dict) else None
        table = _validated_reliability_table(carried) if carried is not None else None
        candidate = AllSourceFusionNetwork(input_dim=128)
        candidate.load_state_dict(state)
        candidate.eval()
        previous_table = dict(self.int_reliability_scores)
        if table is not None:
            self._set_reliability_table({**previous_table, **table})
        try:
            merit = self._measure_candidate(candidate, path, fingerprint, cases, labels, supplied)
        except BaseException:
            if table is not None:
                self._set_reliability_table(previous_table)
            raise
        return merit

    def _measure_candidate(
        self,
        candidate: AllSourceFusionNetwork,
        path: str | Path,
        fingerprint: str,
        cases: list[tuple[dict[str, Any], int]],
        labels: np.ndarray[Any, np.dtype[Any]],
        supplied: list[tuple[dict[str, Any], int]],
    ) -> FusionMerit:
        """Run the merit test of a loaded candidate; install it only if admitted."""
        # The checkpoint is loadable, so this is a merit test: reserve its
        # attempt number before scoring anything (see MeritAttemptLog).
        if self._merit_log is not None:
            attempt = self._merit_log.reserve(fingerprint, path)
        else:
            attempt = 1 + sum(a.fingerprint == fingerprint for a in self._merit_attempts)

        # Measured as fuse_intelligence decides: the same screen feeds both
        # the network (screened inputs) and the evidence path.
        screens = [self._screen(reports) for reports, _ in cases]
        network_hits = (
            np.array(
                [
                    self._trained_assessment(candidate, reports, ci)[0]
                    for (reports, _), ci in zip(cases, screens, strict=True)
                ]
            )
            == labels
        )
        evidence_hits = (
            np.array(
                [
                    self._evidence_assessment(reports, ci)[0]
                    for (reports, _), ci in zip(cases, screens, strict=True)
                ]
            )
            == labels
        )
        network_only = int(np.sum(network_hits & ~evidence_hits))
        evidence_only = int(np.sum(evidence_hits & ~network_hits))
        merit = FusionMerit(
            n_cases=len(cases),
            network_accuracy=float(network_hits.mean()),
            evidence_accuracy=float(evidence_hits.mean()),
            network_only=network_only,
            evidence_only=evidence_only,
            p_value=_mcnemar_one_sided(network_only, evidence_only),
            n_supplied=len(supplied),
            attempt=attempt,
            alpha=merit_alpha_for_attempt(attempt),
        )
        self._merit_attempts.append(MeritAttempt(fingerprint, str(path), merit))
        if self._merit_log is not None:
            self._merit_log.record_result(
                fingerprint,
                attempt,
                {
                    "n_cases": merit.n_cases,
                    "network_accuracy": merit.network_accuracy,
                    "evidence_accuracy": merit.evidence_accuracy,
                    "p_value": merit.p_value,
                    "alpha": merit.alpha,
                    "admitted": merit.admitted,
                },
            )
        if not merit.admitted:
            raise RuntimeError(
                f"fusion weights from {path} refused: held-out accuracy "
                f"{merit.network_accuracy:.3f} vs evidence {merit.evidence_accuracy:.3f} "
                f"(n={merit.n_cases}, discordant {network_only}:{evidence_only}, "
                f"one-sided McNemar p={merit.p_value:.3g} >= {merit.alpha:g}, "
                f"attempt {attempt} on this held-out set)"
            )

        self.fusion_network = candidate
        self._trained = True
        self.merit = merit
        self.logger.info(
            "intelligence fusion: installed weights from %s (held-out accuracy %.3f vs "
            "evidence %.3f, n=%d, p=%.3g, attempt %d)",
            path,
            merit.network_accuracy,
            merit.evidence_accuracy,
            merit.n_cases,
            merit.p_value,
            attempt,
        )
        return merit

    def _set_reliability_table(self, table: Mapping[str, float]) -> None:
        """Use ``table`` for features and evidence weights; re-seed the screen's priors."""
        self.int_reliability_scores = dict(table)
        if self.counterintelligence is not None:
            self.counterintelligence = CounterintelligenceAssessor.from_engine(
                self, config=self.counterintelligence.config
            )

    def fit_reliability_scores(
        self,
        training: Sequence[tuple[dict[str, Any], int]],
        *,
        folds: int = 5,
        shrinkages: Sequence[float] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0),
        min_reports: int = 10,
        seed: int = 0,
        install: bool = True,
    ) -> ReliabilityFit:
        """Learn the per-discipline reliabilities from adjudicated training cases.

        The fixed table (:meth:`_initialize_reliability_scores`) is a declared
        choice. Here each discipline's reliability ``r_d`` is fitted so that
        evidence scoring's reliability-weighted mean of the reported
        ``threat_score`` values matches the adjudicated level (the midpoint
        of its fifth of ``[0, 1]``), by least squares with a penalty pulling
        ``logit(r_d)`` toward the current table. The penalty weight is chosen
        by K-fold cross-validation. A discipline with fewer than
        ``min_reports`` scored reports keeps its current value.

        The table is installed only if, on the same folds, the learned table's
        mean absolute threat-level error is lower than the current table's.
        Only training cases are used: the held-out split stays unspent for the
        network's merit gate (:meth:`load_trained_weights`). Installing a new
        table un-admits a trained network, whose merit was measured on the
        old feature scaling.

        Args:
            training: ``(intel_reports, ThreatLevel value)`` pairs, e.g.
                ``AdjudicationLedger.training_cases()``.
            folds: Cross-validation folds.
            shrinkages: Candidate penalty weights.
            min_reports: Reports a discipline needs to be fitted.
            seed: Fold assignment seed.
            install: Install the table when it wins (``False`` only measures).

        Returns:
            The :class:`ReliabilityFit`.

        Raises:
            ValueError: Fewer than ``MIN_HELDOUT_CASES`` usable cases, or a
                label that is not a ``ThreatLevel`` value.
        """
        from scipy.optimize import minimize

        keys = [d.value for d in IntelligenceDiscipline]
        rows: list[npt.NDArray[np.float64]] = []
        masks: list[npt.NDArray[np.float64]] = []
        levels: list[int] = []
        for reports, label in training:
            level = ThreatLevel(label).value
            scored = {key: score for key, _, score, _ in self._scored_reports(reports, None, None)}
            if not scored:
                continue
            rows.append(np.array([scored.get(k, 0.0) for k in keys]))
            masks.append(np.array([1.0 if k in scored else 0.0 for k in keys]))
            levels.append(level)
        if len(levels) < MIN_HELDOUT_CASES:
            raise ValueError(
                f"{len(levels)} training cases with a scored report; at least "
                f"{MIN_HELDOUT_CASES} are needed to fit reliabilities"
            )
        scores, mask, y_level = np.stack(rows), np.stack(masks), np.array(levels)
        target = (y_level - 0.5) / len(ThreatLevel)
        previous = dict(self.int_reliability_scores)
        prior = np.array([previous.get(k, 0.75) for k in keys])
        theta0 = np.log(prior) - np.log1p(-prior)
        free = mask.sum(axis=0) >= min_reports

        def weights_of(theta_free: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            theta = theta0.copy()
            theta[free] = theta_free
            return 1.0 / (1.0 + np.exp(-theta))

        def risk(w: npt.NDArray[np.float64], idx: npt.NDArray[np.intp]) -> npt.NDArray[np.float64]:
            m = mask[idx] * w
            return np.asarray((m * scores[idx]).sum(axis=1) / m.sum(axis=1))

        def level_error(w: npt.NDArray[np.float64], idx: npt.NDArray[np.intp]) -> float:
            predicted = np.clip(np.ceil(risk(w, idx) * len(ThreatLevel)), 1, len(ThreatLevel))
            return float(np.mean(np.abs(predicted - y_level[idx])))

        def fit(idx: npt.NDArray[np.intp], lam: float) -> npt.NDArray[np.float64]:
            if not free.any():
                return weights_of(theta0[free])

            def objective(
                theta_free: npt.NDArray[np.float64],
            ) -> tuple[float, npt.NDArray[np.float64]]:
                w = weights_of(theta_free)
                m = mask[idx]
                denom = (m * w).sum(axis=1)
                r = (m * w * scores[idx]).sum(axis=1) / denom
                residual = r - target[idx]
                # d r_i / d w_d = m_id (s_id - r_i) / denom_i; d w / d theta = w (1 - w).
                grad_w = (2.0 / idx.size) * (
                    (residual / denom)[:, None] * m * (scores[idx] - r[:, None])
                ).sum(axis=0)
                grad = grad_w * w * (1.0 - w)
                penalty = theta_free - theta0[free]
                value = float(np.mean(residual**2) + lam * np.sum(penalty**2))
                return value, grad[free] + 2.0 * lam * penalty

            result = minimize(objective, theta0[free], jac=True, method="L-BFGS-B")
            return weights_of(np.asarray(result.x))

        order = np.random.default_rng(seed).permutation(len(levels))
        fold_of = np.empty(len(levels), dtype=np.intp)
        fold_of[order] = np.arange(len(levels)) % max(2, folds)
        k = int(fold_of.max()) + 1
        cv: dict[float, float] = {}
        for lam in shrinkages:
            errors = []
            for f in range(k):
                train_idx, test_idx = np.flatnonzero(fold_of != f), np.flatnonzero(fold_of == f)
                errors.append(level_error(fit(train_idx, float(lam)), test_idx) * test_idx.size)
            cv[float(lam)] = float(sum(errors) / len(levels))
        best = min(cv, key=lambda lam: (cv[lam], -lam))  # ties: the stronger pull to the prior
        all_idx = np.arange(len(levels), dtype=np.intp)
        learned = fit(all_idx, best)
        previous_error = level_error(prior, all_idx)
        table = {**previous, **{k_: float(learned[i]) for i, k_ in enumerate(keys) if free[i]}}
        wins = cv[best] < previous_error
        reason = (
            f"cross-validated level error {cv[best]:.4f} vs {previous_error:.4f} with the "
            f"previous table (shrinkage {best:g}, {len(levels)} cases)"
        )
        installed = bool(wins and install)
        if installed:
            if self._trained:
                self._trained = False
                self.merit = None
                reason += "; the admitted network was measured on the old table and is un-admitted"
            self._set_reliability_table(table)
        return ReliabilityFit(
            table=table,
            previous_table=previous,
            fitted=tuple(k_ for i, k_ in enumerate(keys) if free[i]),
            n_cases=len(levels),
            shrinkage=best,
            cv_error_learned=cv[best],
            cv_error_previous=previous_error,
            installed=installed,
            reason=("installed: " if installed else "not installed: ") + reason,
        )

    def _initialize_threat_kb(self) -> dict[str, dict[str, Any]]:
        """Initialize threat pattern knowledge base."""
        return {
            "terrorism_indicators": {
                "patterns": ["recruitment", "training", "logistics", "finance", "communications"],
                "int_sources": [
                    IntelligenceDiscipline.HUMINT,
                    IntelligenceDiscipline.COMINT,
                    IntelligenceDiscipline.FININT,
                    IntelligenceDiscipline.CYBINT,
                ],
            },
            "cyber_attack_indicators": {
                "patterns": ["reconnaissance", "weaponization", "delivery", "exploitation", "c2"],
                "int_sources": [
                    IntelligenceDiscipline.CYBINT,
                    IntelligenceDiscipline.SIGINT,
                    IntelligenceDiscipline.OSINT,
                ],
            },
            "military_buildup": {
                "patterns": ["troop_movement", "equipment_positioning", "logistics_surge"],
                "int_sources": [
                    IntelligenceDiscipline.IMINT,
                    IntelligenceDiscipline.GEOINT,
                    IntelligenceDiscipline.SIGINT,
                    IntelligenceDiscipline.MASINT,
                ],
            },
            "espionage_indicators": {
                "patterns": ["insider_threat", "exfiltration", "covert_comms", "dead_drops"],
                "int_sources": [
                    IntelligenceDiscipline.HUMINT,
                    IntelligenceDiscipline.COMINT,
                    IntelligenceDiscipline.CYBINT,
                ],
            },
            "wmd_proliferation": {
                "patterns": ["procurement", "facility_construction", "testing", "delivery_systems"],
                "int_sources": [
                    IntelligenceDiscipline.IMINT,
                    IntelligenceDiscipline.MASINT,
                    IntelligenceDiscipline.SIGINT,
                    IntelligenceDiscipline.FININT,
                ],
            },
        }

    def _initialize_reliability_scores(self) -> dict[str, float]:
        """Initialize INT source reliability weights."""
        return {
            IntelligenceDiscipline.HUMINT.value: 0.85,
            IntelligenceDiscipline.IMINT.value: 0.92,
            IntelligenceDiscipline.SIGINT.value: 0.90,
            IntelligenceDiscipline.COMINT.value: 0.88,
            IntelligenceDiscipline.GEOINT.value: 0.93,
            IntelligenceDiscipline.MASINT.value: 0.87,
            IntelligenceDiscipline.ELINT.value: 0.89,
            IntelligenceDiscipline.CYBINT.value: 0.86,
            IntelligenceDiscipline.FININT.value: 0.84,
            IntelligenceDiscipline.OSINT.value: 0.75,
            IntelligenceDiscipline.CRYPTANALYSIS.value: 0.91,
            IntelligenceDiscipline.METEOROLOGICAL.value: 0.82,
            IntelligenceDiscipline.TRAFFIC_ANALYSIS.value: 0.83,
        }

    def fuse_intelligence(
        self,
        intel_reports: dict[str, Any],
        temporal_context: list[dict[str, Any]] | None = None,
        counterintelligence_history: Any = None,
    ) -> IntelligenceFusionResult:
        """Fuse multi-source intelligence for threat assessment.

        Args:
            intel_reports: Dictionary of intelligence reports by discipline:
                - osint: Open-source findings
                - comint: Communications intercepts
                - humint: Human source reports
                - geoint: Geospatial data
                - imint: Imagery analysis
                - sigint: Signals intercepts
                - elint: Electronic signatures
                - masint: Technical measurements
                - cybint: Cyber indicators
                - finint: Financial transactions
                - cryptanalysis: Encrypted pattern analysis
            temporal_context: Optional historical threat timeline. It is
                summarised in ``temporal_patterns``; it never reaches the
                fusion network, whose weights are admitted on held-out cases
                without temporal context (see :meth:`load_trained_weights`).
            counterintelligence_history: Source track records and baseline
                reporting rates for the counterintelligence screen (see
                ``AssessmentHistory.coerce``).

        Returns:
            Fused intelligence assessment with threat level

        Raises:
            InputModalityError: ``intel_reports`` is not a mapping of
                discipline to one report mapping (a list of reports under a
                discipline included).
        """
        notes: list[str] = []
        ci = self._screen(intel_reports, counterintelligence_history)
        if self._trained:
            if temporal_context:
                notes.append(
                    "temporal_context not fed to the fusion network: its weights were "
                    "admitted on held-out cases without temporal context"
                )
            threat_class, confidence_score, risk_score = self._trained_assessment(
                self.fusion_network, intel_reports, ci, notes=notes
            )
        else:
            # Untrained network: its logits are initialisation noise, so the
            # assessment comes from the reported evidence instead (reliability
            # -weighted threat scores, corroboration-scaled confidence).
            threat_class, confidence_score, risk_score = self._evidence_assessment(
                intel_reports, ci, notes=notes
            )

        threat_level_enum = ThreatLevel(threat_class)
        threat_detected = threat_level_enum.value >= ThreatLevel.SUBSTANTIAL.value

        primary_sources, corroborating = self._identify_sources(intel_reports, ci)

        indicators = self._extract_threat_indicators(intel_reports, threat_level_enum)

        temporal_patterns = (
            self._analyze_temporal_patterns(temporal_context) if temporal_context else {}
        )

        geospatial_context = self._extract_geospatial_context(intel_reports)

        actions = self._recommend_actions(threat_level_enum, indicators)

        collection_priorities = self._prioritize_collection(
            threat_level_enum, primary_sources, indicators
        )

        neurosymbolic_assessment = None
        if self.enable_neurosymbolic:
            neurosymbolic_assessment = self._apply_symbolic_reasoning(
                intel_reports, threat_level_enum, indicators
            )

        crypto_indicators = None
        if self.enable_cryptanalysis:
            crypto_indicators = self._analyze_cryptographic_patterns(intel_reports)

        result = IntelligenceFusionResult(
            threat_detected=threat_detected,
            threat_level=threat_level_enum.name,
            confidence=confidence_score,
            risk_score=risk_score,
            primary_intel_sources=primary_sources,
            corroborating_sources=corroborating,
            threat_indicators=indicators,
            temporal_patterns=temporal_patterns,
            geospatial_context=geospatial_context,
            recommended_actions=actions,
            collection_priorities=collection_priorities,
            neurosymbolic_assessment=neurosymbolic_assessment,
            cryptographic_indicators=crypto_indicators,
            counterintelligence=ci.as_dict() if ci is not None else None,
            evidence_notes=notes,
        )

        self.logger.info(
            f"Intelligence fusion: {threat_level_enum.name} threat "
            f"(confidence={confidence_score:.3f}, risk={risk_score:.3f})"
        )

        return result

    @staticmethod
    def _finite_or_none(value: Any) -> float | None:
        """``float(value)`` when it yields a finite number, else ``None``.

        An unusable report field — a non-numeric string, ``None``, NaN or
        an infinity — is not evidence: it is dropped (the caller logs it)
        rather than propagated into the decision or allowed to crash it.
        """
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    @classmethod
    def _unit_or_reason(cls, value: Any) -> tuple[float | None, str]:
        """``(number, "")`` for a finite number in ``[0, 1]``, else ``(None, why)``.

        Threat scores and confidences are probabilities. A value outside
        ``[0, 1]`` is not evidence on that scale: clipping only the weighted
        mean would let one report at ``-1e6`` or ``50`` decide the fused
        level whatever its reliability weight.
        """
        number = cls._finite_or_none(value)
        if number is None:
            return None, f"{value!r} is not a finite number"
        if not 0.0 <= number <= 1.0:
            return None, f"{number:g} is outside [0, 1]"
        return number, ""

    def _scored_reports(
        self,
        intel_reports: Mapping[str, Any],
        ci: CounterintelligenceAssessment | None,
        notes: list[str] | None,
    ) -> list[tuple[str, float, float, float | None]]:
        """The reports that count as evidence, with their weights.

        A report counts only when it states a valid ``threat_score`` (a finite
        number in ``[0, 1]``) and keeps a positive weight after screening.
        Its ``confidence`` counts when it is stated and valid; none is
        assumed when it is missing. Every exclusion is logged and, when
        ``notes`` is given, recorded there.

        Returns:
            ``(discipline, reliability weight, threat_score, confidence)`` per
            counting report, in ``IntelligenceDiscipline`` order.
        """

        def exclude(message: str) -> None:
            # Only a deciding call records (and logs); source ranking re-reads
            # the same reports with ``notes=None`` and stays quiet.
            if notes is not None:
                self.logger.warning("evidence assessment: %s", message)
                notes.append(message)

        scored: list[tuple[str, float, float, float | None]] = []
        for discipline in IntelligenceDiscipline:
            key = discipline.value
            report = intel_reports.get(key)
            if not isinstance(report, Mapping):
                continue
            reliability = self.int_reliability_scores.get(key, 0.75)
            if ci is not None:
                adjusted = ci.adjusted_reliability.get(key)
                if adjusted is not None:
                    reliability = adjusted
                reliability *= ci.fusion_weight_multipliers.get(key, 1.0)
                if reliability <= 0.0:
                    exclude(f"{key}: weight 0 after counterintelligence screening; excluded")
                    continue
            if "threat_score" not in report:
                exclude(f"{key}: no threat_score; the report does not count")
                continue
            score, why = self._unit_or_reason(report["threat_score"])
            if score is None:
                exclude(f"{key}: threat_score {why}; the report does not count")
                continue
            confidence: float | None = None
            if "confidence" in report:
                confidence, why = self._unit_or_reason(report["confidence"])
                if confidence is None:
                    exclude(f"{key}: confidence {why}; dropped")
            scored.append((key, reliability, score, confidence))
        return scored

    def _screen(
        self, intel_reports: dict[str, Any], history: Any = None
    ) -> CounterintelligenceAssessment | None:
        """Run the counterintelligence screen, after the modality check."""
        self._require_report_mapping(intel_reports)
        if self.counterintelligence is None:
            return None
        return self.counterintelligence.assess(intel_reports, history)

    @staticmethod
    def _require_report_mapping(intel_reports: dict[str, Any]) -> None:
        """Reject off-modality inputs with a clean, actionable error.

        The engine scores one report mapping per discipline. A list of
        reports under a discipline (which the counterintelligence normaliser
        accepts) is refused here, before screening, rather than screened as
        several reports, silently skipped by evidence scoring and crashing
        source ranking.

        Raises:
            InputModalityError: ``intel_reports`` is not a mapping, or one of
                its values is not a single report mapping.
        """
        if not isinstance(intel_reports, Mapping):
            shape_hint = (
                f" of shape {intel_reports.shape}" if hasattr(intel_reports, "shape") else ""
            )
            raise InputModalityError(
                "IntelligenceFusionEngine expects a multi-INT report mapping "
                "(dict keyed by discipline, e.g. "
                "{'open_source': {...}, 'signals': {...}}), got "
                f"{type(intel_reports).__name__}{shape_hint}."
            )
        for key, value in intel_reports.items():
            if isinstance(value, Mapping):
                continue
            hint = (
                "; fuse one report per discipline (screen a batch of reports with "
                "CounterintelligenceAssessor directly)"
                if isinstance(value, list | tuple)
                else ""
            )
            raise InputModalityError(
                "IntelligenceFusionEngine expects one report mapping per discipline; "
                f"{key!r} holds {type(value).__name__}{hint}."
            )

    def _extract_int_features(self, intel_reports: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Extract features from intelligence reports (O(n) complexity).

        Fails closed: a stated ``confidence``, ``timeliness``, ``relevance``,
        ``completeness`` or ``threat_score`` that is not a finite number in
        ``[0, 1]``, or ``indicators`` without a length, raises rather than
        reaching the network as NaN, an infinity or an out-of-scale value
        (a NaN feature made every logit NaN, and ``argmax`` then read LOW).

        Each discipline's features are scaled by the fixed
        ``int_reliability_scores`` entry, never by the counterintelligence
        screen's source-adjusted reliability: the scaling is the network's
        input contract. :meth:`_trained_assessment` defers to evidence
        scoring when the adjusted value differs.

        Raises:
            InputModalityError: ``intel_reports`` is not a report mapping.
            ValueError: A stated feature field is unusable (see above).
        """
        self._require_report_mapping(intel_reports)

        int_features = {}
        feature_dim = 128 // len(IntelligenceDiscipline)

        def unit_feature(disc_key: str, report: Mapping[str, Any], name: str) -> float | None:
            if name not in report:
                return None
            number, why = self._unit_or_reason(report[name])
            if number is None:
                raise ValueError(f"{disc_key}.{name}: {why}")
            return number

        for discipline in IntelligenceDiscipline:
            disc_key = discipline.value
            features = np.zeros(feature_dim, dtype=np.float32)

            if disc_key in intel_reports:
                report = intel_reports[disc_key]

                for index, name in enumerate(
                    ("confidence", "timeliness", "relevance", "completeness")
                ):
                    value = unit_feature(disc_key, report, name)
                    features[index] = 0.5 if value is None else value

                if "indicators" in report:
                    try:
                        count = len(report["indicators"])
                    except TypeError as exc:
                        raise ValueError(f"{disc_key}.indicators: has no length") from exc
                    features[4] = float(count) / 10.0

                score = unit_feature(disc_key, report, "threat_score")
                if score is not None:
                    features[5] = score

                reliability = self.int_reliability_scores.get(disc_key, 0.75)
                features = np.asarray(features * reliability)

            int_features[disc_key] = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        return int_features

    def _process_temporal_context(self, temporal_context: list[dict[str, Any]]) -> torch.Tensor:
        """Process temporal threat progression."""
        sequence_length = min(len(temporal_context), 10)
        feature_dim = int(self.fusion_network.temporal_lstm.input_size)

        temporal_features = np.zeros((1, sequence_length, feature_dim), dtype=np.float32)

        for i, event in enumerate(temporal_context[-sequence_length:]):
            temporal_features[0, i, 0] = float(event.get("threat_level", 0)) / 5.0
            temporal_features[0, i, 1] = float(event.get("confidence", 0.5))
            temporal_features[0, i, 2] = float(event.get("num_sources", 1)) / 10.0

        return torch.tensor(temporal_features, dtype=torch.float32)

    def _network_assessment(
        self,
        network: AllSourceFusionNetwork,
        intel_reports: dict[str, Any],
        temporal_context: list[dict[str, Any]] | None = None,
    ) -> tuple[int, float, float]:
        """Score the threat with a fusion network on the reports as given.

        This is the raw scorer: no screening, no fallback. Decisions go
        through :meth:`_trained_assessment`.

        Returns:
            Tuple of ``(threat_class, confidence_score, risk_score)``, as
            :meth:`_evidence_assessment` does.

        Raises:
            ValueError: A feature field is unusable (:meth:`_extract_int_features`).
            RuntimeError: The network produced a non-finite output.
        """
        int_features = self._extract_int_features(intel_reports)
        temporal_tensor = (
            self._process_temporal_context(temporal_context) if temporal_context else None
        )
        network.eval()
        with torch.no_grad():
            threat_logits, confidence, _attention = network(int_features, temporal_tensor)
        if not (
            bool(torch.isfinite(threat_logits).all()) and bool(torch.isfinite(confidence).all())
        ):
            raise RuntimeError(
                "fusion network produced non-finite outputs from finite features; "
                "refusing to decide from them"
            )
        threat_class = int(torch.argmax(threat_logits[0]).item()) + 1
        confidence_score = float(confidence[0].item())
        return threat_class, confidence_score, confidence_score * (threat_class / 5.0)

    def _trained_assessment(
        self,
        network: AllSourceFusionNetwork,
        intel_reports: dict[str, Any],
        ci: CounterintelligenceAssessment | None,
        *,
        notes: list[str] | None = None,
    ) -> tuple[int, float, float]:
        """Decide with admitted fusion weights, on the screened reports.

        The network takes per-discipline features and has no input for a
        weight, so screening reaches it the only sound way: a discipline the
        screen gave weight 0 (a derivative copy, an excluded report) is
        removed and the network sees it as not reported. Scaling the features
        by the multiplier instead would change their meaning (a halved
        ``threat_score`` reads as a lower threat, not as weaker evidence).
        For the same reason the network cannot honour a fractional weight
        (a coordinated, fabricated or impossible-travel flag): when any
        reporting discipline carries one, or a feature is unusable, the
        network does not score this input and the decision falls back to
        :meth:`_evidence_assessment`, which can apply it. Each case is
        recorded in ``notes``. :meth:`load_trained_weights` measures merit
        through this same method, so the gate tests what decides.

        Source-adjusted reliability is handled the same way. The network's
        features are scaled by the fixed ``int_reliability_scores`` table
        (:meth:`_extract_int_features`); that scaling is part of its input
        contract, so substituting the screen's source-adjusted reliability
        would shift the input distribution every checkpoint was trained and
        measured on. Instead, when a screened discipline's adjusted
        reliability differs from the fixed table (stated grades or an
        adjudicated track record moved it), the network does not score and
        the evidence path, which weights by the adjusted value, decides.

        Returns:
            Tuple of ``(threat_class, confidence_score, risk_score)``.

        Raises:
            RuntimeError: The network produced a non-finite output.
        """
        sink: list[str] = notes if notes is not None else []
        screened: dict[str, Any] = {}
        fractional: list[str] = []
        moved: list[str] = []
        for discipline in IntelligenceDiscipline:
            key = discipline.value
            if key not in intel_reports:
                continue
            multiplier = 1.0 if ci is None else ci.fusion_weight_multipliers.get(key, 1.0)
            if multiplier <= 0.0:
                sink.append(
                    f"{key}: weight 0 after counterintelligence screening; not shown to "
                    "the fusion network"
                )
                continue
            if multiplier < 1.0:
                fractional.append(f"{key} x{multiplier:g}")
            fixed = self.int_reliability_scores.get(key, 0.75)
            adjusted = None if ci is None else ci.adjusted_reliability.get(key)
            if adjusted is not None and not math.isclose(adjusted, fixed, abs_tol=1e-12):
                moved.append(f"{key} {fixed:g}->{adjusted:.3g}")
            screened[key] = intel_reports[key]
        if fractional:
            sink.append(
                "fusion network did not score: it cannot represent fractional screening "
                f"weights ({', '.join(fractional)}); decided by evidence scoring"
            )
            return self._evidence_assessment(intel_reports, ci, notes=notes)
        if moved:
            sink.append(
                "fusion network did not score: its features use the fixed discipline "
                f"reliabilities, but source information adjusted them ({', '.join(moved)}); "
                "decided by evidence scoring"
            )
            return self._evidence_assessment(intel_reports, ci, notes=notes)
        try:
            return self._network_assessment(network, screened)
        except ValueError as exc:
            sink.append(f"fusion network did not score: {exc}; decided by evidence scoring")
            return self._evidence_assessment(intel_reports, ci, notes=notes)

    def _evidence_assessment(
        self,
        intel_reports: dict[str, Any],
        ci: CounterintelligenceAssessment | None = None,
        *,
        notes: list[str] | None = None,
    ) -> tuple[int, float, float]:
        """Score the threat from the reported evidence alone.

        This is the decision path while the fusion network is untrained: a
        reliability-weighted mean of the per-discipline ``threat_score``
        values, with confidence taken from the reports and scaled by
        corroboration (a single-source assessment is damped; three or more
        scoring disciplines carry full weight, matching the three primary
        sources ``_identify_sources`` surfaces).

        Only a report stating a valid ``threat_score`` -- a finite number in
        ``[0, 1]`` -- counts, toward the score, the corroboration and the
        confidence. A report without one (``{}`` included) is not evidence and
        no default is fabricated for it; a score that is non-numeric, ``None``,
        NaN, infinite or outside ``[0, 1]`` excludes its report, because one
        report at ``-1e6`` or ``50`` would otherwise decide the weighted mean
        whatever its weight. A stated ``confidence`` outside ``[0, 1]`` (or
        unusable) is dropped; a missing one is not assumed. Every exclusion
        is logged and recorded in ``notes`` (surfaced as
        ``IntelligenceFusionResult.evidence_notes``).

        With a counterintelligence assessment ``ci``, each discipline's
        reliability is the screen's source-adjusted posterior times its weight
        multiplier; a discipline whose reports are all derivative copies or
        excluded (multiplier 0) contributes nothing, and corroboration counts
        independent origins rather than reporting disciplines.

        Args:
            intel_reports: Reports keyed by discipline.
            ci: The counterintelligence screen of the same reports, if run.
            notes: Collects the reason for every exclusion, when given.

        Returns:
            Tuple of ``(threat_class, confidence_score, risk_score)`` where
            ``threat_class`` indexes :class:`ThreatLevel` and both scores lie
            in ``[0.0, 1.0]``.
        """
        self._require_report_mapping(intel_reports)

        scored = self._scored_reports(intel_reports, ci, notes)
        total_weight = sum(weight for _, weight, _, _ in scored)
        if not scored or total_weight <= 0.0:
            return ThreatLevel.LOW.value, 0.0, 0.0

        # Every score lies in [0, 1], so the weighted mean does too.
        risk_score = float(sum(weight * score for _, weight, score, _ in scored) / total_weight)

        confidences = [conf for _, _, _, conf in scored if conf is not None]
        if confidences:
            contributing = [key for key, _, _, _ in scored]
            independent = (
                ci.independent_corroboration(contributing) if ci is not None else len(scored)
            )
            corroboration = min(1.0, independent / 3.0)
            confidence_score = float(
                min(max(float(np.mean(confidences)) * corroboration, 0.0), 1.0)
            )
        else:
            # Threat scores with no valid confidence anywhere: the risk is
            # reportable but nothing supports asserting confidence in it.
            confidence_score = 0.0

        threat_class = min(len(ThreatLevel), max(1, math.ceil(risk_score * len(ThreatLevel))))
        return threat_class, confidence_score, risk_score

    def _identify_sources(
        self,
        intel_reports: dict[str, Any],
        ci: CounterintelligenceAssessment | None = None,
    ) -> tuple[list[str], list[str]]:
        """Identify primary and corroborating intelligence sources.

        Only disciplines whose report counts as evidence (a valid
        ``threat_score`` and a positive screening weight, as in
        :meth:`_evidence_assessment`) are listed, ranked by confidence; an
        empty report or a derivative copy is not a source.
        """
        source_scores = [
            (key, confidence if confidence is not None else 0.0)
            for key, _, _, confidence in self._scored_reports(intel_reports, ci, None)
        ]
        # Stable sort: equal confidences keep IntelligenceDiscipline order.
        source_scores.sort(key=lambda x: x[1], reverse=True)

        primary: list[str] = [s[0] for s in source_scores[:3]]
        corroborating: list[str] = [s[0] for s in source_scores[3:6]]

        return primary, corroborating

    def _extract_threat_indicators(
        self, intel_reports: dict[str, Any], threat_level: ThreatLevel
    ) -> list[str]:
        """Extract key threat indicators from reports."""
        indicators = set()

        for report in intel_reports.values():
            if "indicators" in report:
                indicators.update(report["indicators"][:5])

        if threat_level.value >= ThreatLevel.SUBSTANTIAL.value:
            indicators.add("elevated_threat_posture")

        return list(indicators)[:15]

    def _analyze_temporal_patterns(self, temporal_context: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze temporal threat progression patterns."""
        if not temporal_context:
            return {}

        threat_levels = [e.get("threat_level", 2) for e in temporal_context]

        return {
            "trend": "escalating" if threat_levels[-1] > threat_levels[0] else "stable",
            "mean_threat": np.mean(threat_levels),
            "volatility": np.std(threat_levels),
            "time_span_days": len(temporal_context),
        }

    def _extract_geospatial_context(self, intel_reports: dict[str, Any]) -> dict[str, Any]:
        """Extract geospatial threat context."""
        geospatial = {}

        if IntelligenceDiscipline.GEOINT.value in intel_reports:
            geo_report = intel_reports[IntelligenceDiscipline.GEOINT.value]
            geospatial["locations"] = geo_report.get("locations", [])
            geospatial["proximity_to_assets"] = geo_report.get("proximity_km", 0)

        if IntelligenceDiscipline.IMINT.value in intel_reports:
            img_report = intel_reports[IntelligenceDiscipline.IMINT.value]
            geospatial["imagery_confidence"] = img_report.get("confidence", 0.0)

        return geospatial

    def _recommend_actions(self, threat_level: ThreatLevel, indicators: list[str]) -> list[str]:
        """Recommend actions based on threat assessment."""
        actions = []

        if threat_level == ThreatLevel.CRITICAL:
            actions.extend(
                [
                    "Immediate senior leadership notification",
                    "Activate crisis response protocols",
                    "Deploy additional collection assets",
                    "Coordinate with operational elements",
                ]
            )
        elif threat_level == ThreatLevel.SEVERE:
            actions.extend(
                [
                    "Elevate monitoring posture",
                    "Increase collection frequency",
                    "Prepare contingency responses",
                    "Brief stakeholders",
                ]
            )
        elif threat_level == ThreatLevel.SUBSTANTIAL:
            actions.extend(
                [
                    "Enhanced monitoring recommended",
                    "Cross-reference with allied intelligence",
                    "Update threat assessments",
                ]
            )
        else:
            actions.append("Continue routine monitoring")

        return actions[:6]

    def _prioritize_collection(
        self, threat_level: ThreatLevel, primary_sources: list[str], indicators: list[str]
    ) -> list[str]:
        """Prioritize intelligence collection efforts."""
        priorities = []

        if threat_level.value >= ThreatLevel.SUBSTANTIAL.value:
            priorities.append("Priority 1: Real-time monitoring")

        for source in primary_sources:
            priorities.append(f"Sustain {source} collection")

        valid_disciplines = [d.value for d in IntelligenceDiscipline]
        source_disciplines: set[IntelligenceDiscipline] = {
            IntelligenceDiscipline(s) for s in primary_sources if s in valid_disciplines
        }
        gaps = set(IntelligenceDiscipline) - source_disciplines

        for gap in list(gaps)[:2]:
            priorities.append(f"Fill gap: {gap.value}")

        return priorities[:6]

    def _apply_symbolic_reasoning(
        self, intel_reports: dict[str, Any], threat_level: ThreatLevel, indicators: list[str]
    ) -> dict[str, Any]:
        """Apply neurosymbolic threat reasoning."""
        all_matched_patterns: list[str] = []
        deductions: list[str] = []
        confidence_factors: list[str] = []

        for threat_type, pattern_info in self.threat_knowledge_base.items():
            matched_patterns = [
                p for p in pattern_info["patterns"] if p in " ".join(indicators).lower()
            ]

            if matched_patterns:
                all_matched_patterns.append(f"{threat_type}: {', '.join(matched_patterns)}")
                deductions.append(f"Potential {threat_type.replace('_', ' ')} activity")

        if len(all_matched_patterns) > 1:
            confidence_factors.append("High confidence: Multiple threat pattern matches")

        return {
            "matched_patterns": all_matched_patterns,
            "deductions": deductions,
            "confidence_factors": confidence_factors,
        }

    def _analyze_cryptographic_patterns(self, intel_reports: dict[str, Any]) -> dict[str, Any]:
        """Analyze cryptographic and pattern indicators."""
        encrypted_comms_detected: bool = False
        pattern_strength: float = 0.0
        recommendations: list[str] = []

        if IntelligenceDiscipline.COMINT.value in intel_reports:
            comint = intel_reports[IntelligenceDiscipline.COMINT.value]
            encrypted_comms_detected = bool(comint.get("encryption_detected", False))

            if encrypted_comms_detected:
                recommendations.append("Prioritize cryptanalysis resources")
                pattern_strength = 0.7

        if IntelligenceDiscipline.CYBINT.value in intel_reports:
            cybint = intel_reports[IntelligenceDiscipline.CYBINT.value]
            if "encryption_algorithm" in cybint:
                recommendations.append(f"Target: {cybint['encryption_algorithm']}")

        return {
            "encrypted_comms_detected": encrypted_comms_detected,
            "pattern_strength": pattern_strength,
            "recommendations": recommendations,
        }

    def extract_features(self, data: dict[str, Any]) -> torch.Tensor:
        """Extract features for ML fusion integration.

        Always emits the declared ``(1, 128)`` width: the 13 disciplines
        contribute ``128 // 13 = 9`` dims each (117), and the floor-division
        remainder is zero-padded so the populated and empty paths agree —
        historically the populated path emitted 117 dims while the empty
        path emitted 128, giving the same method two different widths.
        """
        int_features = self._extract_int_features(data)

        all_features = []
        for discipline in IntelligenceDiscipline:
            if discipline.value in int_features:
                all_features.append(int_features[discipline.value])

        if not all_features:
            return torch.zeros(1, 128, dtype=torch.float32)

        features = torch.cat(all_features, dim=-1)
        if features.shape[-1] < 128:
            # Match the source tensor's device and dtype so the concat stays
            # valid if the fusion engine is ever moved off CPU.
            pad = torch.zeros(
                *features.shape[:-1],
                128 - features.shape[-1],
                dtype=features.dtype,
                device=features.device,
            )
            features = torch.cat([features, pad], dim=-1)
        return features[..., :128]

    def predict(self, data: dict[str, Any]) -> dict[str, Any]:
        """Predict for engine integration."""
        result = self.fuse_intelligence(data)

        return {
            "anomaly_scores": np.array([result.risk_score], dtype=np.float32),
            "threat_level": result.threat_level,
            "confidence": result.confidence,
            "intel_sources": result.primary_intel_sources,
        }

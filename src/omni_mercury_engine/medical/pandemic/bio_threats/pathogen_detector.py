# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pathogen Detection using a Baseline-Calibrated Energy Model.

Scores a snapshot of biological sensor readings (one value per channel)
against reference measurements and flags snapshots whose energy exceeds a
false-alarm-calibrated threshold. Integrates MASINT (Measurement and Signature
Intelligence) reports when supplied.

Mathematical Foundation:
- Energy: E(x) = ½ (x-μ)ᵀ Λ (x-μ), the energy of a Gaussian Markov random
  field, the continuous counterpart of the Ising model; the couplings are
  J_ij = -Λ_ij. It is classical: no quantum term is involved.
- Fit: μ and Λ = Σ⁻¹ from the first 70% of the baseline, with Σ shrunk toward
  its diagonal (as in :mod:`omni_mercury_engine.detectors.energy_based`).
- Threshold: E > E_critical, where E_critical is the split-conformal quantile
  of the remaining 30% of baseline energies, so a normal snapshot is flagged
  with probability at most ``false_alarm_rate`` (Vovk et al. 2005).

An alarm is presumptive. Sensor readings cannot identify an agent: the type is
taken from a MASINT report when one names a known agent, and laboratory
confirmation (as in the BioWatch PCR process) comes before any public action.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from omni_mercury_engine.detectors.detection_tier import conformal_threshold

_FIT_FRACTION = 0.7
_SIGNATURE_SD = 3.0


@dataclass
class BioThreatResult:
    """Result from bio-threat detection."""

    threat_detected: bool
    pathogen_type: str
    energy_score: float
    confidence: float

    bio_signatures: list[str]
    masint_indicators: list[str]
    recommended_interdiction: list[str]


class PathogenDetector:
    """Baseline-calibrated pathogen detector (Medical Interdiction).

    Call :meth:`fit` with reference measurements (rows are snapshots, columns
    are sensor channels), then :meth:`detect_pathogen` on new snapshots.

    Features:
    - Gaussian energy model fitted to the baseline (couplings in ``J_matrix``)
    - Split-conformal threshold bounding the false-alarm rate
    - Per-channel signatures relative to the baseline
    - MASINT agent reports for classification and escalation
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize pathogen detector.

        Args:
            config: Configuration dict: ``false_alarm_rate`` (default 0.01),
                ``shrinkage`` toward the diagonal covariance (default 0.1),
                ``ridge`` (default 1e-6, relative to the mean variance), and an
                optional fixed ``energy_threshold`` that replaces calibration.

        Raises:
            ValueError: If a setting is out of range.
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}

        self.state_dim = self.config.get("state_dim", 50)
        # Kept for configuration compatibility; energies are reported at T = 1.
        self.temperature = self.config.get("temperature", 1.0)
        self.false_alarm_rate = float(self.config.get("false_alarm_rate", 0.01))
        self.shrinkage = float(self.config.get("shrinkage", 0.1))
        self.ridge = float(self.config.get("ridge", 1e-6))
        self.energy_threshold: float | None = self.config.get("energy_threshold")

        if not 0.0 < self.false_alarm_rate < 1.0:
            raise ValueError(f"false_alarm_rate must be in (0, 1), got {self.false_alarm_rate}")
        if not 0.0 <= self.shrinkage < 1.0:
            raise ValueError(f"shrinkage must be in [0, 1), got {self.shrinkage}")
        if not self.ridge > 0.0:
            raise ValueError(f"ridge must be > 0, got {self.ridge}")

        # A ``seed`` key is accepted for compatibility; fitting is deterministic.
        self.mean_: np.ndarray[Any, Any] | None = None
        self.scale_: np.ndarray[Any, Any] | None = None
        self.precision_: np.ndarray[Any, Any] | None = None
        self.J_matrix: np.ndarray[Any, Any] = np.zeros((self.state_dim, self.state_dim))
        self._calibration_energies: np.ndarray[Any, Any] | None = None

        self.known_pathogens = {
            "viral": ["influenza", "coronavirus", "ebola", "marburg"],
            "bacterial": ["anthrax", "plague", "tularemia", "botulinum"],
            "toxin": ["ricin", "saxitoxin", "vx_nerve_agent"],
        }

        self.logger.info(f"PathogenDetector initialized (dim={self.state_dim})")

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has learned a baseline."""
        return self.precision_ is not None

    def fit(self, baseline: np.ndarray[Any, Any]) -> PathogenDetector:
        """Learn the baseline energy model and calibrate the alarm threshold.

        Args:
            baseline: Reference snapshots of normal conditions, shape
                ``(n_snapshots, n_channels)``, oldest first.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If the baseline is not a finite 2-D array of at least two
                snapshots, has no variance, or is too short to calibrate
                ``false_alarm_rate`` (the held-out 30% needs at least
                ``1/false_alarm_rate - 1`` snapshots).
        """
        data = np.asarray(baseline, dtype=np.float64)
        if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] < 1:
            raise ValueError("baseline must be a 2-D array of >= 2 snapshots x channels")
        if not np.all(np.isfinite(data)):
            raise ValueError("baseline must be finite")
        n, d = data.shape
        calibrate = self.energy_threshold is None
        split = int(_FIT_FRACTION * n) if calibrate else n
        needed = int(np.ceil(1.0 / self.false_alarm_rate - 1.0))
        if calibrate and (split < 2 or n - split < needed):
            raise ValueError(
                f"baseline has {n} snapshots; calibrating false_alarm_rate="
                f"{self.false_alarm_rate} needs {needed} held-out snapshots "
                f"(about {int(np.ceil(needed / (1.0 - _FIT_FRACTION)))} in total)"
            )
        fit_rows = data[:split]

        cov = np.atleast_2d(np.cov(fit_rows, rowvar=False))
        diag = np.diag(np.diag(cov))
        cov = (1.0 - self.shrinkage) * cov + self.shrinkage * diag
        mean_variance = float(np.mean(np.diag(cov)))
        if not mean_variance > 0.0:
            raise ValueError("baseline has no variance in any channel")
        cov = cov + self.ridge * mean_variance * np.eye(d)

        self.state_dim = d
        self.mean_ = fit_rows.mean(axis=0)
        self.scale_ = np.sqrt(np.diag(cov))
        self.precision_ = np.linalg.inv(cov)
        self.J_matrix = -(self.precision_ - np.diag(np.diag(self.precision_)))

        if calibrate:
            energies = np.array([self._energy(row) for row in data[split:]])
            self._calibration_energies = np.sort(energies)
            self.energy_threshold = conformal_threshold(energies, alpha=self.false_alarm_rate)
        self.logger.info(
            f"PathogenDetector fitted on {n} snapshots x {d} channels "
            f"(E_critical={self.energy_threshold:.3f})"
        )
        return self

    def detect_pathogen(
        self, bio_data: np.ndarray[Any, Any], masint_data: dict[str, Any] | None = None
    ) -> BioThreatResult:
        """Score one snapshot of sensor readings against the fitted baseline.

        Args:
            bio_data: One reading per channel (e.g., sensor readings), the same
                channels and units as the baseline.
            masint_data: Optional MASINT report; ``agent`` names an identified
                agent, ``bio_signature_detected``, ``threat_score`` and
                ``indicators`` are relayed.

        Returns:
            BioThreatResult with threat assessment

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
            ValueError: If the snapshot has the wrong length or non-finite values.
        """
        if np.asarray(bio_data).size == 0:
            return self._no_threat_result()

        deviation = self._deviation(bio_data)
        energy = self._quadratic_energy(deviation)
        assert self.energy_threshold is not None  # set by fit()

        threat_detected = energy > self.energy_threshold

        masint_indicators = []
        if masint_data:
            masint_indicators = self._process_masint(masint_data)

        pathogen_type = self._classify_pathogen(threat_detected, masint_data)

        bio_signatures = self._identify_bio_signatures(deviation, threat_detected)

        recommended_interdiction = []
        if threat_detected:
            recommended_interdiction = self._recommend_interdiction(
                pathogen_type, energy, masint_indicators
            )

        confidence = self._significance(energy)

        result = BioThreatResult(
            threat_detected=threat_detected,
            pathogen_type=pathogen_type,
            energy_score=energy,
            confidence=confidence,
            bio_signatures=bio_signatures,
            masint_indicators=masint_indicators,
            recommended_interdiction=recommended_interdiction,
        )

        if threat_detected:
            self.logger.warning(
                f"Presumptive bio-threat: {pathogen_type} "
                f"(E={energy:.3f} > {self.energy_threshold:.3f}, conf={confidence:.3f})"
            )

        return result

    def _deviation(self, bio_data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Validate a snapshot and return its deviation from the baseline mean."""
        if self.mean_ is None:
            raise RuntimeError("PathogenDetector not fitted. Call fit(baseline) first.")
        x = np.asarray(bio_data, dtype=np.float64).ravel()
        if x.size != self.mean_.size:
            raise ValueError(f"expected {self.mean_.size} channel readings, got {x.size}")
        if not np.all(np.isfinite(x)):
            raise ValueError("bio_data must be finite")
        return x - self.mean_

    def _energy(self, snapshot: np.ndarray[Any, Any]) -> float:
        """Energy of a raw snapshot."""
        return self._quadratic_energy(self._deviation(snapshot))

    def _quadratic_energy(self, deviation: np.ndarray[Any, Any]) -> float:
        """E = ½ (x-μ)ᵀ Λ (x-μ); higher means less like the baseline."""
        assert self.precision_ is not None
        return float(0.5 * deviation @ self.precision_ @ deviation)

    def _significance(self, energy: float) -> float:
        """1 - conformal p-value of ``energy`` against the held-out baseline."""
        cal = self._calibration_energies
        if cal is None:  # a fixed ``energy_threshold`` was configured
            return 1.0 if energy > float(self.energy_threshold or 0.0) else 0.0
        at_least = cal.size - int(np.searchsorted(cal, energy, side="left"))
        return float(1.0 - (1 + at_least) / (cal.size + 1))

    def _classify_pathogen(self, threat_detected: bool, masint_data: dict[str, Any] | None) -> str:
        """Agent class from a MASINT report naming a known agent, else unclassified.

        Sensor readings alone cannot identify an agent, so without a report the
        type of an alarm is ``unclassified_anomaly``.
        """
        if not threat_detected:
            return "none_detected"
        agent = str((masint_data or {}).get("agent", "")).strip().lower()
        for agent_class, agents in self.known_pathogens.items():
            if agent in agents:
                return f"{agent_class}_reported"
        return "unclassified_anomaly"

    def _identify_bio_signatures(
        self, deviation: np.ndarray[Any, Any], threat_detected: bool
    ) -> list[str]:
        """Channels beyond ±3 baseline standard deviations, largest first."""
        assert self.scale_ is not None
        z = deviation / self.scale_
        order = np.argsort(-np.abs(z))
        signatures = [
            f"channel {i}: {z[i]:+.1f} sd {'above' if z[i] > 0 else 'below'} baseline"
            for i in order
            if abs(z[i]) >= _SIGNATURE_SD
        ][:10]
        if threat_detected and not signatures:
            signatures.append(
                "Joint pattern atypical: no single channel beyond 3 sd (correlation change)"
            )
        return signatures

    def _process_masint(self, masint_data: dict[str, Any]) -> list[str]:
        """Process MASINT intelligence for bio-threat correlation.

        MASINT: Measurement and Signature Intelligence (technical signatures).
        """
        indicators = []

        agent = str(masint_data.get("agent", "")).strip().lower()
        if agent:
            indicators.append(f"MASINT reports agent: {agent}")

        if masint_data.get("bio_signature_detected", False):
            indicators.append("MASINT bio-signature confirmed")

        if masint_data.get("threat_score", 0) > 0.7:
            indicators.append("MASINT high-confidence threat")

        if "genetic" in str(masint_data.get("indicators", [])).lower():
            indicators.append("Genetic modification detected (MASINT)")

        return indicators

    def _recommend_interdiction(
        self, pathogen_type: str, energy: float, masint_indicators: list[str]
    ) -> list[str]:
        """Recommend medical interdiction actions.

        Actions: quarantine, decontamination, vaccine deployment, antidote distribution.
        """
        actions = [
            "Presumptive alarm: collect samples for laboratory confirmation (e.g., PCR) "
            "before public action",
            "Notify the public-health laboratory and check the sensor for faults",
        ]

        if "viral" in pathogen_type:
            actions.append("Implement quarantine protocols")
            actions.append("Deploy rapid testing infrastructure")
            actions.append("Activate vaccine production pipeline")

        if "bacterial" in pathogen_type:
            actions.append("Distribute antibiotics to at-risk populations")
            actions.append("Activate bio-defense units")
            actions.append("Secure ventilation systems")

        if "toxin" in pathogen_type:
            actions.append("Distribute antidotes (if available)")
            actions.append("Activate decontamination teams")
            actions.append("Secure water supplies")

        if masint_indicators and "high-confidence" in str(masint_indicators):
            actions.append("Escalate to federal bio-threat response teams")

        return actions

    def _no_threat_result(self) -> BioThreatResult:
        """Return no-threat result for empty/invalid data."""
        return BioThreatResult(
            threat_detected=False,
            pathogen_type="none",
            energy_score=0.0,
            confidence=0.0,
            bio_signatures=[],
            masint_indicators=[],
            recommended_interdiction=[],
        )

    def extract_features(self, bio_data: np.ndarray[Any, Any]) -> torch.Tensor:
        """Extract features for ML fusion integration.

        Per-channel deviations in baseline standard deviations (length
        ``state_dim``); zeros for an empty input.
        """
        if np.asarray(bio_data).size == 0:
            return torch.zeros(self.state_dim, dtype=torch.float32)

        deviation = self._deviation(bio_data)
        assert self.scale_ is not None
        return torch.tensor(deviation / self.scale_, dtype=torch.float32)

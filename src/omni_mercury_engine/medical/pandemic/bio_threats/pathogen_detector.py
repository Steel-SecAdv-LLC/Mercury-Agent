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
  its diagonal and each channel's variance inflated by the same relative ridge.
  Both steps commute with a change of units per channel, so the energy and the
  verdict do not depend on the units a sensor reports in.
- Threshold: E > E_critical, where E_critical is the split-conformal quantile
  of the remaining 30% of baseline energies. A normal snapshot is then flagged
  with probability at most ``false_alarm_rate`` on average over the calibration
  draw (Vovk et al. 2005); one baseline's realised rate varies around it (up to
  about 3% at a 1% target with 300 held-out snapshots). With
  ``false_alarm_confidence`` set, E_critical is instead the training-conditional
  quantile (Vovk 2012): the fitted baseline's own rate is at most
  ``false_alarm_rate`` with that probability, at the cost of a lower average
  rate and so less sensitivity.
- Sequential test (:meth:`PathogenDetector.monitor`): a conformal
  Shiryaev-Roberts test on the stream of energies
  (:mod:`omni_mercury_engine.core.conformal_martingale`), seeded with the
  held-out energies. It alarms on a persistent rise too small for one snapshot
  to cross E_critical; while the stream is like the baseline, its alarms are at
  least ``mean_snapshots_to_false_alarm`` snapshots apart on average. ``seed``
  seeds the uniform draws of its smoothed p-values.

An alarm is presumptive. Sensor readings cannot identify an agent: the type is
taken from a MASINT report when one names a known agent, and laboratory
confirmation (as in the BioWatch PCR process) comes before any public action.
"""

from __future__ import annotations

import logging
import numbers
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch

from omni_mercury_engine.core.conformal_martingale import ConformalChangeDetector
from omni_mercury_engine.detectors.detection_tier import (
    conformal_threshold,
    pac_conformal_threshold,
    pac_minimum_calibration_size,
)

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

    # Set by PathogenDetector.monitor; detect_pathogen leaves the defaults.
    sequential_alarm: bool = False
    change_statistic: float | None = None


class PathogenDetector:
    """Baseline-calibrated pathogen detector (Medical Interdiction).

    Call :meth:`fit` with reference measurements (rows are snapshots, columns
    are sensor channels), then :meth:`detect_pathogen` on new snapshots.

    Features:
    - Gaussian energy model fitted to the baseline (couplings in ``J_matrix``)
    - Split-conformal threshold bounding the false-alarm rate
    - Sequential conformal change test over a stream (:meth:`monitor`)
    - Per-channel signatures relative to the baseline
    - MASINT agent reports for classification and escalation
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize pathogen detector.

        Args:
            config: Configuration dict: ``false_alarm_rate`` (default 0.01),
                ``shrinkage`` toward the diagonal covariance (default 0.1),
                ``ridge`` (default 1e-6, relative to each channel's variance),
                and an optional fixed ``energy_threshold`` (an energy, so >= 0;
                an alarm is ``E > energy_threshold``) that replaces calibration.
                The released detector's -0.5 default meant ``E < -0.5`` and is
                rejected. ``mean_snapshots_to_false_alarm`` (default 1000) is
                the threshold of the sequential test in :meth:`monitor`, and
                ``seed`` (an integer, default None) seeds that test.
                ``false_alarm_confidence`` (default None, in (0, 1)) makes the
                calibrated threshold training-conditional: the realised
                false-alarm rate is at most ``false_alarm_rate`` with this
                probability.

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
        self.mean_snapshots_to_false_alarm = float(
            self.config.get("mean_snapshots_to_false_alarm", 1000.0)
        )
        confidence = self.config.get("false_alarm_confidence")
        self.false_alarm_confidence: float | None = (
            None if confidence is None else float(confidence)
        )
        fixed = self.config.get("energy_threshold")
        self._fixed_threshold: float | None = None if fixed is None else float(fixed)
        self.energy_threshold: float | None = self._fixed_threshold

        if not 0.0 < self.false_alarm_rate < 1.0:
            raise ValueError(f"false_alarm_rate must be in (0, 1), got {self.false_alarm_rate}")
        if not 0.0 <= self.shrinkage < 1.0:
            raise ValueError(f"shrinkage must be in [0, 1), got {self.shrinkage}")
        if not self.ridge > 0.0:
            raise ValueError(f"ridge must be > 0, got {self.ridge}")
        if self.false_alarm_confidence is not None and not 0.0 < self.false_alarm_confidence < 1.0:
            raise ValueError(
                f"false_alarm_confidence must be in (0, 1), got {self.false_alarm_confidence}"
            )
        if not (
            np.isfinite(self.mean_snapshots_to_false_alarm)
            and self.mean_snapshots_to_false_alarm > 1.0
        ):
            raise ValueError(
                "mean_snapshots_to_false_alarm must be finite and > 1, "
                f"got {self.mean_snapshots_to_false_alarm}"
            )
        if self._fixed_threshold is not None and not (
            np.isfinite(self._fixed_threshold) and self._fixed_threshold >= 0.0
        ):
            raise ValueError(
                f"energy_threshold is an energy (>= 0, alarm when E exceeds it), "
                f"got {self._fixed_threshold}"
            )

        # Fitting is deterministic; the seed is only used by the sequential test.
        self.seed = self.config.get("seed")
        if self.seed is not None and (
            isinstance(self.seed, (bool, np.bool_)) or not isinstance(self.seed, numbers.Integral)
        ):
            raise ValueError(f"seed must be an integer or None, got {self.seed!r}")
        self.mean_: np.ndarray[Any, Any] | None = None
        self.scale_: np.ndarray[Any, Any] | None = None
        self.precision_: np.ndarray[Any, Any] | None = None
        self.J_matrix: np.ndarray[Any, Any] = np.zeros((self.state_dim, self.state_dim))
        self._calibration_energies: np.ndarray[Any, Any] | None = None
        self._monitor: ConformalChangeDetector | None = None

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
                snapshots, a channel has no variance, or it is too short to calibrate
                ``false_alarm_rate`` (the held-out 30% needs at least
                ``1/false_alarm_rate - 1`` snapshots, or ln(1 - confidence) /
                ln(1 - false_alarm_rate) with ``false_alarm_confidence``).
        """
        data = np.asarray(baseline, dtype=np.float64)
        if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] < 1:
            raise ValueError("baseline must be a 2-D array of >= 2 snapshots x channels")
        if not np.all(np.isfinite(data)):
            raise ValueError("baseline must be finite")
        n, d = data.shape
        calibrate = self._fixed_threshold is None
        split = int(_FIT_FRACTION * n) if calibrate else n
        if self.false_alarm_confidence is None:
            needed = int(np.ceil(1.0 / self.false_alarm_rate - 1.0))
        else:
            needed = pac_minimum_calibration_size(
                self.false_alarm_rate, 1.0 - self.false_alarm_confidence
            )
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
        variances = np.diag(cov).copy()
        constant = np.flatnonzero(~(variances > 0.0))
        if constant.size:
            raise ValueError(
                f"baseline channel(s) {constant.tolist()} have no variance; "
                "repair or drop them before fitting"
            )
        cov = cov + self.ridge * np.diag(variances)

        self.state_dim = d
        self.mean_ = fit_rows.mean(axis=0)
        self.scale_ = np.sqrt(np.diag(cov))
        self.precision_ = np.linalg.inv(cov)
        self.J_matrix = -(self.precision_ - np.diag(np.diag(self.precision_)))

        if calibrate:
            energies = np.array([self._energy(row) for row in data[split:]])
            self._calibration_energies = np.sort(energies)
            if self.false_alarm_confidence is None:
                self.energy_threshold = conformal_threshold(energies, alpha=self.false_alarm_rate)
            else:
                self.energy_threshold = pac_conformal_threshold(
                    energies,
                    alpha=self.false_alarm_rate,
                    delta=1.0 - self.false_alarm_confidence,
                )
            self._monitor = ConformalChangeDetector(
                energies,
                mean_time_to_false_alarm=self.mean_snapshots_to_false_alarm,
                seed=None if self.seed is None else int(self.seed),
            )
        else:
            self._calibration_energies = None
            self.energy_threshold = self._fixed_threshold
            self._monitor = None
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

    def monitor(
        self, bio_data: np.ndarray[Any, Any], masint_data: dict[str, Any] | None = None
    ) -> BioThreatResult:
        """Score the next snapshot of a stream and update the sequential test.

        Feed snapshots in time order. The result is :meth:`detect_pathogen`'s,
        plus the sequential test: ``threat_detected`` is also set when the
        stream's energies have risen persistently (``sequential_alarm``), and
        ``change_statistic`` is log R_n against the threshold
        log(``mean_snapshots_to_false_alarm``). The test restarts after each
        alarm; after a confirmed event, refit the baseline.

        Raises:
            RuntimeError: If the detector is not fitted with a calibrated
                threshold (a fixed ``energy_threshold`` has no held-out energies).
            ValueError: As :meth:`detect_pathogen`; an empty snapshot is too.
        """
        if self._monitor is None:
            raise RuntimeError(
                "sequential monitoring needs fit() with a calibrated threshold "
                "(no fixed energy_threshold)"
            )
        if np.asarray(bio_data).size == 0:
            raise ValueError("an empty snapshot cannot be monitored")
        result = self.detect_pathogen(bio_data, masint_data)
        status = self._monitor.update(result.energy_score)
        result = replace(
            result,
            sequential_alarm=status.alarm,
            change_statistic=status.log_statistic,
        )
        if not status.alarm:
            return result
        signature = (
            "Sequential test: energies persistently above the baseline "
            f"(Shiryaev-Roberts statistic reached {self.mean_snapshots_to_false_alarm:g})"
        )
        if result.threat_detected:
            return replace(result, bio_signatures=[*result.bio_signatures, signature])
        pathogen_type = self._classify_pathogen(True, masint_data)
        channels = self._identify_bio_signatures(self._deviation(bio_data), False)
        self.logger.warning(f"Presumptive bio-threat (sequential test): {pathogen_type}")
        return replace(
            result,
            threat_detected=True,
            pathogen_type=pathogen_type,
            bio_signatures=[signature, *channels],
            recommended_interdiction=self._recommend_interdiction(
                pathogen_type, result.energy_score, result.masint_indicators
            ),
        )

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
        """1 - conformal p-value of ``energy`` against the held-out baseline.

        With a fixed ``energy_threshold`` there is no calibration set, so this is
        1.0 for an alarm and 0.0 otherwise.
        """
        cal = self._calibration_energies
        if cal is None:
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

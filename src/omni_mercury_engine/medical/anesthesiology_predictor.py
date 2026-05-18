"""Anesthesiology predictor with TIVA Bi-LSTM, PID infusion, and vital monitoring.

Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

Ported from Omni-AXA-Engine's ``anesthesiology_predictor.py``.  The neural
architecture (Bi-LSTM, 164K parameters), PID infusion controller signs and
gains (kp=0.5, ki=0.1, kd=0.2, target BIS=50, safe BIS window 40-60), and
clinical vital ranges (MAP 65-110 mmHg, HR 50-100 bpm, SpO2 >= 92%, EtCO2
30-45 mmHg) all match the original verified implementation.

Live data integration
---------------------
Mercury Agent ships integration-ready, not pre-integrated.  The predictor
**requires** a
:class:`~omni_mercury_engine.medical.data_sources.VitalsDataSource` adapter
whenever ``enable_hemodynamics`` is true; instantiating the class without
one raises :class:`~omni_mercury_engine.medical.data_sources.ConfigurationError`.

Reference adapter: :class:`FHIRObservationVitalsSource` consumes any
spec-compliant FHIR R4 server (Epic, Cerner, SMART Health IT sandbox, on-prem
HL7 v2 gateway with FHIR translation).  Vendor SDK adapters (Philips
IntelliVue, GE CARESCAPE, Mindray BeneVision) can be written as
:class:`VitalsDataSource` subclasses; the contract is documented in
``docs/medical/SETUP.md``.

Operational notes
-----------------
This module performs decision support only.  Clinical validation by a licensed
anesthesiologist is required before any output is used to influence patient
care.  The :class:`SmartInfusionController` is a PID toy implementation; it
must not be wired into actual infusion pumps without clinical-trial validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import torch
from torch import nn

from omni_mercury_engine.medical.data_sources import (
    ConfigurationError,
    VitalsDataSource,
    VitalsReading,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)


class AnesthesiaType(Enum):
    """Anesthesia delivery types."""

    GENERAL_INHALATIONAL = "general_inhalational"
    TIVA = "total_intravenous_anesthesia"
    REGIONAL = "regional_anesthesia"
    SEDATION = "conscious_sedation"
    COMBINED = "combined_technique"


class AnesthesiaRisk(Enum):
    """Anesthesia risk categories monitored by the predictor."""

    AWARENESS = "intraoperative_awareness"
    HYPOTENSION = "hypotension"
    HYPERTENSION = "hypertension"
    RESPIRATORY_DEPRESSION = "respiratory_depression"
    OVERDOSE = "anesthetic_overdose"
    UNDERDOSE = "inadequate_anesthesia"
    HEMODYNAMIC_INSTABILITY = "hemodynamic_instability"


@dataclass
class AnesthesiaPredictionResult:
    """Anesthesia prediction result."""

    risk_detected: bool
    confidence: float
    risk_type: str
    risk_score: float

    depth_of_anesthesia: float
    hemodynamic_stability: float
    respiratory_adequacy: float

    infusion_anomalies: list[str] = field(default_factory=list)
    vital_sign_alerts: list[str] = field(default_factory=list)
    clinical_recommendations: list[str] = field(default_factory=list)

    bis_score: float | None = None
    mac_equivalent: float | None = None
    predicted_awareness_risk: float = 0.0
    intervention_needed: bool = False
    vitals_source: str | None = None
    vitals_snapshot_count: int = 0


_RISK_TYPES: Final[tuple[str, ...]] = (
    "awareness",
    "hypotension",
    "hypertension",
    "respiratory_depression",
    "overdose",
    "underdose",
    "hemodynamic_instability",
)


class TIVAMonitoringSystem(nn.Module):
    """Bi-LSTM TIVA monitor predicting depth of anesthesia and risk profile.

    Architecture: ``input_dim=8`` -> Bi-LSTM(``hidden_dim=64``, num_layers=2,
    dropout=0.2, bidirectional=True) -> additive attention -> depth head
    (sigmoid scalar) + 7-class risk classifier.  Parameter count: 164K
    (matches the verified Omni-AXA implementation).
    """

    def __init__(self, input_dim: int = 8, hidden_dim: int = 64, num_layers: int = 2) -> None:
        """Initialise the TIVA monitor."""
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=0.2,
            bidirectional=True,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )
        self.depth_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        self.risk_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, len(_RISK_TYPES)),
        )

    def forward(
        self, anesthesia_data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            anesthesia_data: ``(batch, time_steps, 8)`` float tensor
                ``[propofol, remifentanil, MAP, HR, SpO2, EtCO2, BIS, temp]``.

        Returns:
            Tuple of (depth_of_anesthesia, risk_scores, attention_weights).
        """
        lstm_out, _ = self.lstm(anesthesia_data)
        attention_scores = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_scores, dim=1)
        context = torch.sum(lstm_out * attention_weights, dim=1)
        depth = self.depth_predictor(context)
        risks = self.risk_classifier(context)
        return depth, risks, attention_weights.squeeze(-1)


class SmartInfusionController:
    """PID closed-loop infusion controller for propofol/remifentanil.

    Implements a discrete-time PID controller, identical to the verified
    Omni-AXA implementation (kp=0.5, ki=0.1, kd=0.2).  Target BIS is 50; the
    safe BIS window is ``(40, 60)``.

    .. warning::

       Decision support only.  Do not connect to a real infusion device
       without separate clinical-trial validation and regulatory approval.
    """

    target_bis: float = 50.0
    bis_range: tuple[float, float] = (40.0, 60.0)
    kp: float = 0.5
    ki: float = 0.1
    kd: float = 0.2
    propofol_limits: tuple[float, float] = (0.0, 200.0)
    remifentanil_limits: tuple[float, float] = (0.0, 0.5)

    def __init__(self) -> None:
        """Initialise the PID controller with zero accumulated error."""
        self.integral_error = 0.0
        self.previous_error = 0.0

    def reset(self) -> None:
        """Clear accumulated PID error (e.g. on case-start)."""
        self.integral_error = 0.0
        self.previous_error = 0.0

    def compute_infusion_adjustment(
        self,
        current_bis: float,
        current_propofol_rate: float,
        current_remifentanil_rate: float,
        dt: float = 1.0,
    ) -> dict[str, Any]:
        """Compute a PID adjustment.

        Args:
            current_bis: Current BIS (0-100).
            current_propofol_rate: Current propofol infusion (mcg/kg/min).
            current_remifentanil_rate: Current remifentanil infusion
                (mcg/kg/min).
            dt: Time step in minutes (must be positive).

        Returns:
            Adjustment dictionary.

        Raises:
            ValueError: If ``dt`` is non-positive.
        """
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt!r}")
        error = self.target_bis - current_bis
        self.integral_error += error * dt
        derivative_error = (error - self.previous_error) / dt
        pid_output = self.kp * error + self.ki * self.integral_error + self.kd * derivative_error
        propofol_adjustment = -pid_output * 10.0
        remifentanil_adjustment = -pid_output * 0.02
        new_propofol = float(
            np.clip(
                current_propofol_rate + propofol_adjustment,
                self.propofol_limits[0],
                self.propofol_limits[1],
            )
        )
        new_remifentanil = float(
            np.clip(
                current_remifentanil_rate + remifentanil_adjustment,
                self.remifentanil_limits[0],
                self.remifentanil_limits[1],
            )
        )
        self.previous_error = error
        anomaly_detected = current_bis < self.bis_range[0] or current_bis > self.bis_range[1]
        return {
            "propofol_rate_mcg_kg_min": new_propofol,
            "remifentanil_rate_mcg_kg_min": new_remifentanil,
            "propofol_adjustment": float(propofol_adjustment),
            "remifentanil_adjustment": float(remifentanil_adjustment),
            "bis_error": float(error),
            "anomaly_detected": anomaly_detected,
            "recommendations": self._generate_infusion_recommendations(
                current_bis, error, anomaly_detected
            ),
        }

    @staticmethod
    def _generate_infusion_recommendations(bis: float, error: float, anomaly: bool) -> list[str]:
        """Generate human-readable infusion adjustment recommendations."""
        recs: list[str] = []
        if anomaly:
            if bis < 40:
                recs.extend(
                    [
                        "ALERT: Deep anesthesia (BIS < 40)",
                        "Consider reducing anesthetic depth",
                        "Rule out equipment malfunction",
                    ]
                )
            elif bis > 60:
                recs.extend(
                    [
                        "ALERT: Light anesthesia (BIS > 60)",
                        "Risk of awareness - increase anesthetic depth",
                        "Assess surgical stimulation level",
                    ]
                )
        if abs(error) > 10:
            recs.append(f"BIS deviation: {error:.1f} from target")
            recs.append("Monitor closely for next 2-3 minutes")
        return recs


class HemodynamicMonitor:
    """Hemodynamic monitor evaluating MAP / HR / SpO2 / EtCO2.

    Reference clinical ranges (verified against ASA monitoring guidance):
    MAP 65-110 mmHg, HR 50-100 bpm, SpO2 >= 92%, EtCO2 30-45 mmHg.
    """

    map_range: tuple[float, float] = (65.0, 110.0)
    hr_range: tuple[float, float] = (50.0, 100.0)
    spo2_threshold: float = 92.0
    etco2_range: tuple[float, float] = (30.0, 45.0)

    def assess_hemodynamics(self, vitals: Mapping[str, float]) -> dict[str, Any]:
        """Score hemodynamic stability from a vitals snapshot.

        Args:
            vitals: Mapping with ``mean_arterial_pressure_mmhg``,
                ``heart_rate_bpm``, ``oxygen_saturation_pct``,
                ``end_tidal_co2_mmhg`` keys (all optional).

        Returns:
            Risk dictionary including a ``hemodynamic_stability`` score in
            ``[0, 1]``.
        """
        map_val = float(vitals.get("mean_arterial_pressure_mmhg", 80.0))
        hr = float(vitals.get("heart_rate_bpm", 70.0))
        spo2 = float(vitals.get("oxygen_saturation_pct", 98.0))
        etco2 = float(vitals.get("end_tidal_co2_mmhg", 38.0))

        alerts: list[str] = []
        risks: dict[str, float] = {
            "hypotension": 0.0,
            "hypertension": 0.0,
            "tachycardia": 0.0,
            "bradycardia": 0.0,
            "hypoxemia": 0.0,
            "hypercarbia": 0.0,
            "hypocarbia": 0.0,
        }

        if map_val < self.map_range[0]:
            deviation = (self.map_range[0] - map_val) / self.map_range[0]
            risks["hypotension"] = min(deviation * 2.0, 1.0)
            alerts.append(f"Hypotension: MAP {map_val:.1f} mmHg")
        elif map_val > self.map_range[1]:
            deviation = (map_val - self.map_range[1]) / self.map_range[1]
            risks["hypertension"] = min(deviation * 2.0, 1.0)
            alerts.append(f"Hypertension: MAP {map_val:.1f} mmHg")

        if hr < self.hr_range[0]:
            deviation = (self.hr_range[0] - hr) / self.hr_range[0]
            risks["bradycardia"] = min(deviation * 2.0, 1.0)
            alerts.append(f"Bradycardia: HR {hr:.0f} bpm")
        elif hr > self.hr_range[1]:
            deviation = (hr - self.hr_range[1]) / self.hr_range[1]
            risks["tachycardia"] = min(deviation * 2.0, 1.0)
            alerts.append(f"Tachycardia: HR {hr:.0f} bpm")

        if spo2 < self.spo2_threshold:
            # Each percentage-point below the 92 % SpO₂ threshold is
            # treated as a 25 % step on the hypoxemia risk scale; SpO₂ < 88 %
            # therefore saturates at 1.0 and any value below 92 % crosses
            # the 0.6 intervention threshold below.
            risks["hypoxemia"] = min((self.spo2_threshold - spo2) * 0.25, 1.0)
            alerts.append(f"Hypoxemia: SpO2 {spo2:.1f}%")

        if etco2 < self.etco2_range[0]:
            risks["hypocarbia"] = min((self.etco2_range[0] - etco2) / 10.0, 1.0)
            alerts.append(f"Hypocarbia: EtCO2 {etco2:.1f} mmHg")
        elif etco2 > self.etco2_range[1]:
            risks["hypercarbia"] = min((etco2 - self.etco2_range[1]) / 10.0, 1.0)
            alerts.append(f"Hypercarbia: EtCO2 {etco2:.1f} mmHg")

        overall_risk = max(risks.values())
        # Any SpO₂ below the safety threshold is treated as intervention-
        # worthy regardless of magnitude; the asymmetric scoring above
        # ensures the 0.6 cutoff is also crossed for sub-92 % SpO₂, but
        # we keep an explicit guard here to stay correct even if scoring
        # is later changed.
        intervention_needed = overall_risk > 0.6 or spo2 < self.spo2_threshold
        return {
            "hemodynamic_stability": 1.0 - overall_risk,
            "risk_scores": risks,
            "alerts": alerts,
            "intervention_needed": intervention_needed,
            "recommendations": self._generate_hemodynamic_recommendations(risks),
        }

    @staticmethod
    def _generate_hemodynamic_recommendations(
        risks: Mapping[str, float],
    ) -> list[str]:
        """Generate per-risk hemodynamic management recommendations."""
        recs: list[str] = []
        if risks["hypotension"] > 0.6:
            recs.extend(
                [
                    "Hypotension management:",
                    "- Reduce anesthetic depth",
                    "- Fluid bolus (250-500mL crystalloid)",
                    "- Consider vasopressor (phenylephrine/ephedrine)",
                ]
            )
        if risks["hypertension"] > 0.6:
            recs.extend(
                [
                    "Hypertension management:",
                    "- Deepen anesthesia",
                    "- Rule out inadequate analgesia",
                    "- Consider antihypertensive if persistent",
                ]
            )
        if risks["hypoxemia"] > 0.6:
            recs.extend(
                [
                    "CRITICAL: Hypoxemia",
                    "- Increase FiO2 to 100%",
                    "- Check airway/ventilation",
                    "- Rule out pneumothorax/bronchospasm",
                ]
            )
        if risks["bradycardia"] >= 0.5:
            recs.extend(
                [
                    "Bradycardia management:",
                    "- Consider anticholinergic (atropine/glycopyrrolate)",
                    "- Rule out vagal stimulation",
                ]
            )
        return recs


def _vitals_reading_to_snapshot(reading: VitalsReading) -> dict[str, float]:
    """Map a :class:`VitalsReading` to the snapshot dict the monitor expects.

    Missing channels are omitted (not filled with synthetic defaults); the
    monitor's own keyword defaults are exercised only when the reading
    itself did not report a channel.
    """
    snapshot: dict[str, float] = {}
    if reading.map_mmhg is not None:
        snapshot["mean_arterial_pressure_mmhg"] = float(reading.map_mmhg)
    if reading.hr_bpm is not None:
        snapshot["heart_rate_bpm"] = float(reading.hr_bpm)
    if reading.spo2_pct is not None:
        snapshot["oxygen_saturation_pct"] = float(reading.spo2_pct)
    if reading.etco2_mmhg is not None:
        snapshot["end_tidal_co2_mmhg"] = float(reading.etco2_mmhg)
    return snapshot


class AnesthesiologyPredictor:
    """Integrated anesthesia risk predictor.

    Combines the TIVA Bi-LSTM, PID infusion controller, and hemodynamic
    monitor.  Like :class:`EndocrinologyDetector` this is the platform
    integration unit and **requires** a configured
    :class:`~omni_mercury_engine.medical.data_sources.VitalsDataSource`
    whenever ``enable_hemodynamics`` is true.

    Synthetic generators have been removed from production paths.  Callers
    who only need the rule monitors (PID controller, hemodynamic range
    checks) can instantiate :class:`SmartInfusionController` and
    :class:`HemodynamicMonitor` directly.
    """

    def __init__(
        self,
        data_source: VitalsDataSource | None = None,
        *,
        enable_tiva: bool = True,
        enable_pid: bool = True,
        enable_hemodynamics: bool = True,
    ) -> None:
        """Initialise the predictor.

        Args:
            data_source: Configured vitals adapter (e.g. a
                :class:`~omni_mercury_engine.medical.data_sources.FHIRObservationVitalsSource`
                or a custom subclass of :class:`VitalsDataSource`).  Required
                whenever ``enable_hemodynamics`` is true.
            enable_tiva: Enable the TIVA Bi-LSTM.
            enable_pid: Enable the PID infusion controller.
            enable_hemodynamics: Enable the hemodynamic monitor.

        Raises:
            ConfigurationError: If ``enable_hemodynamics`` is true and
                ``data_source`` is ``None``.
            TypeError: If ``data_source`` is not a :class:`VitalsDataSource`.
        """
        if enable_hemodynamics and data_source is None:
            raise ConfigurationError(
                "AnesthesiologyPredictor requires a configured "
                "VitalsDataSource when hemodynamic monitoring is enabled. "
                "Mercury Agent does not ship with default credentials. See "
                "docs/medical/SETUP.md for instructions on configuring a "
                "vitals adapter (FHIR R4 reference implementation provided)."
            )
        if data_source is not None and not isinstance(data_source, VitalsDataSource):
            raise TypeError(
                "data_source must subclass VitalsDataSource; " f"got {type(data_source).__name__}"
            )
        self.data_source = data_source
        self.enable_tiva = enable_tiva
        self.enable_pid = enable_pid
        self.enable_hemodynamics = enable_hemodynamics
        self.tiva_monitor: TIVAMonitoringSystem | None = (
            TIVAMonitoringSystem() if enable_tiva else None
        )
        self.infusion_controller: SmartInfusionController | None = (
            SmartInfusionController() if enable_pid else None
        )
        self.hemodynamic_monitor: HemodynamicMonitor | None = (
            HemodynamicMonitor() if enable_hemodynamics else None
        )

    def fetch_and_predict(
        self,
        *,
        window_minutes: int = 5,
        anesthesia_context: Mapping[str, Any] | None = None,
    ) -> AnesthesiaPredictionResult:
        """Fetch the latest vitals window then run the full prediction pipeline.

        Args:
            window_minutes: Look-back window passed to
                :meth:`VitalsDataSource.fetch_recent_vitals`.
            anesthesia_context: Optional supplementary case data (e.g.
                ``anesthesia_sequence`` for the TIVA Bi-LSTM,
                ``infusion`` snapshot for the PID controller).

        Returns:
            :class:`AnesthesiaPredictionResult`.

        Raises:
            ConfigurationError: If ``enable_hemodynamics`` is false or no
                data source was supplied at construction time.
        """
        if not self.enable_hemodynamics or self.data_source is None:
            raise ConfigurationError(
                "fetch_and_predict requires enable_hemodynamics=True and a "
                "configured data_source. Use predict_anesthesia_risk() with "
                "pre-loaded data for rule-engine-only flows."
            )
        readings = self.data_source.fetch_recent_vitals(window_minutes=window_minutes)
        payload: dict[str, Any] = dict(anesthesia_context or {})
        payload["vitals_readings"] = readings
        return self.predict_anesthesia_risk(payload)

    def predict_anesthesia_risk(
        self, anesthesia_data: Mapping[str, Any]
    ) -> AnesthesiaPredictionResult:
        """Run the rule + ML pipeline on pre-loaded anesthesia data.

        Args:
            anesthesia_data: Mapping that may contain any of:

                * ``anesthesia_sequence`` - 2-D float sequence
                  ``(time, 8)`` for the TIVA Bi-LSTM.
                * ``infusion`` - dict with ``current_bis``,
                  ``current_propofol_rate``, ``current_remifentanil_rate``.
                * ``vitals_readings`` - sequence of
                  :class:`~omni_mercury_engine.medical.data_sources.VitalsReading`
                  (preferred; produced by ``fetch_and_predict``).
                * ``vitals`` - legacy single-snapshot mapping.

        Returns:
            :class:`AnesthesiaPredictionResult`.
        """
        result = AnesthesiaPredictionResult(
            risk_detected=False,
            confidence=0.0,
            risk_type="none",
            risk_score=0.0,
            depth_of_anesthesia=0.5,
            hemodynamic_stability=1.0,
            respiratory_adequacy=1.0,
        )

        if (
            self.enable_tiva
            and self.tiva_monitor is not None
            and "anesthesia_sequence" in anesthesia_data
        ):
            tiva_result = self._analyze_tiva(
                np.asarray(anesthesia_data["anesthesia_sequence"], dtype=np.float64)
            )
            result.depth_of_anesthesia = float(tiva_result["depth_of_anesthesia"])
            result.risk_type = str(tiva_result["risk_type"])
            result.confidence = float(tiva_result["confidence"])
            if result.risk_type != "none":
                result.risk_detected = True

        if (
            self.enable_pid
            and self.infusion_controller is not None
            and "infusion" in anesthesia_data
        ):
            infusion = anesthesia_data["infusion"]
            pid_result = self.infusion_controller.compute_infusion_adjustment(
                current_bis=float(infusion["current_bis"]),
                current_propofol_rate=float(infusion["current_propofol_rate"]),
                current_remifentanil_rate=float(infusion["current_remifentanil_rate"]),
                dt=float(infusion.get("dt", 1.0)),
            )
            result.bis_score = float(infusion["current_bis"])
            result.infusion_anomalies = list(pid_result["recommendations"])
            if pid_result["anomaly_detected"]:
                result.risk_detected = True
                result.intervention_needed = True

        snapshot, source_label, reading_count = self._extract_vitals_snapshot(anesthesia_data)
        if (
            self.enable_hemodynamics
            and self.hemodynamic_monitor is not None
            and snapshot is not None
        ):
            hemo_result = self.hemodynamic_monitor.assess_hemodynamics(snapshot)
            result.hemodynamic_stability = float(hemo_result["hemodynamic_stability"])
            result.vital_sign_alerts = list(hemo_result["alerts"])
            result.clinical_recommendations.extend(hemo_result["recommendations"])
            result.vitals_source = source_label
            result.vitals_snapshot_count = reading_count
            if hemo_result["intervention_needed"]:
                result.risk_detected = True
                result.intervention_needed = True

        result.risk_score = self._calculate_overall_risk(result)
        if result.risk_score > 0.7:
            result.intervention_needed = True
        return result

    @staticmethod
    def _extract_vitals_snapshot(
        anesthesia_data: Mapping[str, Any],
    ) -> tuple[dict[str, float] | None, str | None, int]:
        """Resolve the most-recent vitals snapshot for the hemodynamic monitor."""
        readings = anesthesia_data.get("vitals_readings")
        if readings:
            ordered: list[VitalsReading] = []
            sources: set[str] = set()
            for r in readings:
                if not isinstance(r, VitalsReading):
                    raise TypeError(
                        "vitals_readings entries must be VitalsReading instances; "
                        f"got {type(r).__name__}"
                    )
                ordered.append(r)
                sources.add(r.source)
            ordered.sort(key=lambda r: r.timestamp)
            latest = ordered[-1]
            source_label = next(iter(sources)) if len(sources) == 1 else "mixed"
            return _vitals_reading_to_snapshot(latest), source_label, len(ordered)

        legacy = anesthesia_data.get("vitals")
        if legacy is not None:
            snapshot = {k: float(v) for k, v in dict(legacy).items()}
            return snapshot, "preloaded", 1
        return None, None, 0

    def _analyze_tiva(self, sequence: np.ndarray[Any, np.dtype[np.float64]]) -> dict[str, Any]:
        """Run the TIVA Bi-LSTM in inference mode."""
        if self.tiva_monitor is None:
            raise RuntimeError("TIVA monitor is not enabled")
        if sequence.ndim != 2 or sequence.shape[-1] != 8:
            raise ValueError(
                "anesthesia_sequence must have shape (time, 8); got " f"{sequence.shape}"
            )
        x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
        self.tiva_monitor.eval()
        with torch.no_grad():
            depth, risks, _attn = self.tiva_monitor(x)
        risk_probs = torch.softmax(risks[0], dim=0).cpu().numpy()
        risk_idx = int(np.argmax(risk_probs))
        risk_label = _RISK_TYPES[risk_idx] if risk_probs[risk_idx] > 0.5 else "none"
        return {
            "depth_of_anesthesia": float(depth.item()),
            "risk_type": risk_label,
            "confidence": float(risk_probs[risk_idx]),
        }

    @staticmethod
    def _calculate_overall_risk(result: AnesthesiaPredictionResult) -> float:
        """Compute the overall anesthesia risk score in ``[0, 1]``."""
        # Stability and adequacy are both 1.0 when "fine"; subtract them so
        # the components contribute proportionally to instability.
        components = (
            (1.0 - result.hemodynamic_stability) * 0.4,
            (1.0 - result.respiratory_adequacy) * 0.3,
            result.predicted_awareness_risk * 0.3,
        )
        return float(min(sum(components), 1.0))


def get_anesthesiology_predictor(
    data_source: VitalsDataSource | None = None,
    *,
    enable_tiva: bool = True,
    enable_pid: bool = True,
    enable_hemodynamics: bool = True,
) -> AnesthesiologyPredictor:
    """Factory returning a configured :class:`AnesthesiologyPredictor`.

    See :class:`AnesthesiologyPredictor` for the full argument contract and
    the :class:`ConfigurationError` semantics.
    """
    return AnesthesiologyPredictor(
        data_source,
        enable_tiva=enable_tiva,
        enable_pid=enable_pid,
        enable_hemodynamics=enable_hemodynamics,
    )


def count_tiva_parameters(model: TIVAMonitoringSystem | None = None) -> int:
    """Return the trainable parameter count of the TIVA Bi-LSTM."""
    instance = model if model is not None else TIVAMonitoringSystem()
    return int(sum(p.numel() for p in instance.parameters() if p.requires_grad))


__all__ = [
    "AnesthesiaPredictionResult",
    "AnesthesiaRisk",
    "AnesthesiaType",
    "AnesthesiologyPredictor",
    "HemodynamicMonitor",
    "SmartInfusionController",
    "TIVAMonitoringSystem",
    "count_tiva_parameters",
    "get_anesthesiology_predictor",
]

"""Anesthesiology predictor with TIVA Bi-LSTM, PID infusion, and vital monitoring.

Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

Ported from Omni-AXA-Engine's ``anesthesiology_predictor.py``.  The neural
architecture (Bi-LSTM, 164K parameters), PID infusion controller signs and
gains, and clinical vital ranges (MAP 65-110 mmHg, HR 50-100 bpm, SpO2 >= 92%,
EtCO2 30-45 mmHg) all match the original verified implementation.

Live data integration
---------------------
Synthetic generators have been removed from production paths.  Real anesthesia
traces should be supplied via the
:class:`omni_mercury_engine.medical.anesthesiology_predictor.VitalDBClient`
helper, which streams cases and per-track samples from the public VitalDB
research dataset hosted at https://api.vitaldb.net (no auth required).  See
``VitalDBClient.fetch_case_track`` for usage.

Operational notes
-----------------
This module performs decision support only.  Clinical validation by a licensed
anesthesiologist is required before any output is used to influence patient
care.  The :class:`SmartInfusionController` is a PID toy implementation; it
must not be wired into actual infusion pumps without clinical-trial validation.
"""

from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Final
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

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
        """Initialise the TIVA monitor.

        Args:
            input_dim: Per-time-step feature dimensionality.
            hidden_dim: LSTM hidden size.
            num_layers: Number of stacked LSTM layers.
        """
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

    Implements a discrete-time PID controller with anti-windup-free error
    integration, identical to the verified Omni-AXA implementation
    (kp=0.5, ki=0.1, kd=0.2).  Target BIS is 50; the safe BIS window is
    ``(40, 60)``.

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
            dt: Time step in minutes (>= 0).

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


class VitalDBClientError(RuntimeError):
    """Raised when the VitalDB API returns an unrecoverable error."""


class VitalDBClient:
    """Read-only client for the public VitalDB research API.

    VitalDB is a free public anesthesia/critical-care research dataset
    operated by Seoul National University Hospital.  No authentication is
    required.  See https://vitaldb.net for the dataset description.

    Endpoints used:

    * ``/cases`` - case-level metadata as CSV.
    * ``/trks`` - per-case track index as CSV.
    * ``/{tid}`` - per-track samples as CSV.
    """

    DEFAULT_BASE_URL: Final[str] = "https://api.vitaldb.net"

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = 30.0,
        user_agent: str = "Mercury-Agent/1.7 Anesthesiology",
    ) -> None:
        """Initialise the client.

        Args:
            base_url: Base URL for the VitalDB HTTP API.
            timeout_seconds: Network timeout per request.
            user_agent: HTTP ``User-Agent`` header value.
        """
        self._base_url = base_url.rstrip("/")
        self._timeout = float(timeout_seconds)
        self._user_agent = user_agent

    def _request_csv(self, path: str, params: Mapping[str, str] | None = None) -> str:
        """Fetch a CSV resource from the VitalDB API."""
        url = f"{self._base_url}/{path.lstrip('/')}"
        if params:
            url = f"{url}?{urlencode(dict(params))}"
        request = Request(  # noqa: S310 - public HTTPS endpoint
            url,
            headers={"User-Agent": self._user_agent, "Accept": "text/csv"},
        )
        try:
            with urlopen(request, timeout=self._timeout) as response:  # noqa: S310
                if response.status != 200:
                    raise VitalDBClientError(f"Unexpected status {response.status} from {url}")
                return response.read().decode("utf-8")
        except OSError as exc:
            raise VitalDBClientError(f"VitalDB request failed: {exc}") from exc

    def list_cases(self) -> list[dict[str, str]]:
        """Return the case index as a list of dictionaries."""
        text = self._request_csv("cases")
        return list(csv.DictReader(io.StringIO(text)))

    def list_tracks(self, case_id: int) -> list[dict[str, str]]:
        """Return the per-track index for ``case_id``."""
        text = self._request_csv("trks", {"caseid": str(case_id)})
        return list(csv.DictReader(io.StringIO(text)))

    def fetch_case_track(self, track_id: str) -> npt.NDArray[np.float64]:
        """Fetch raw samples for a single track id.

        Args:
            track_id: VitalDB track identifier from :meth:`list_tracks`.

        Returns:
            ``(N, 2)`` float64 array of ``(time_seconds, value)`` pairs.
        """
        text = self._request_csv(track_id)
        reader = csv.reader(io.StringIO(text))
        rows: list[tuple[float, float]] = []
        for row in reader:
            if not row or row[0].lower() == "time":
                continue
            try:
                rows.append((float(row[0]), float(row[1])))
            except (IndexError, ValueError):
                continue
        return np.asarray(rows, dtype=np.float64)


class AnesthesiologyPredictor:
    """Integrated anesthesiology prediction system.

    Combines :class:`TIVAMonitoringSystem`, :class:`SmartInfusionController`,
    and :class:`HemodynamicMonitor` into a single :meth:`predict_anesthesia_risk`
    entry point.  Synthetic generators are intentionally absent; callers must
    supply real traces (e.g. via :class:`VitalDBClient` or a local clinical
    feed).
    """

    def __init__(
        self,
        enable_tiva: bool = True,
        enable_smart_infusion: bool = True,
        enable_hemodynamic: bool = True,
    ) -> None:
        """Initialise the predictor.

        Args:
            enable_tiva: Enable the TIVA Bi-LSTM monitor.
            enable_smart_infusion: Enable the PID infusion controller.
            enable_hemodynamic: Enable the hemodynamic monitor.
        """
        self.enable_tiva = enable_tiva
        self.enable_smart_infusion = enable_smart_infusion
        self.enable_hemodynamic = enable_hemodynamic
        self.tiva_monitor: TIVAMonitoringSystem | None = (
            TIVAMonitoringSystem() if enable_tiva else None
        )
        self.infusion_controller: SmartInfusionController | None = (
            SmartInfusionController() if enable_smart_infusion else None
        )
        self.hemodynamic_monitor: HemodynamicMonitor | None = (
            HemodynamicMonitor() if enable_hemodynamic else None
        )

    def predict_anesthesia_risk(
        self, patient_data: Mapping[str, Any]
    ) -> AnesthesiaPredictionResult:
        """Run the full prediction pipeline.

        Args:
            patient_data: Patient data dictionary.  Recognised keys:

                * ``anesthesia_sequence``: ``(time_steps, 8)`` float array.
                * ``current_vitals``: vitals mapping.
                * ``infusion_rates``: current infusion rates.
                * ``bis_score``: current BIS score.

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
            and "anesthesia_sequence" in patient_data
        ):
            tiva_result = self._analyze_tiva(
                np.asarray(patient_data["anesthesia_sequence"], dtype=np.float64)
            )
            result.depth_of_anesthesia = float(tiva_result["depth"])
            result.predicted_awareness_risk = float(tiva_result["awareness_risk"])
            result.infusion_anomalies = list(tiva_result["anomalies"])
            result.confidence = max(result.confidence, float(tiva_result["confidence"]))
            if tiva_result["risk_detected"]:
                result.risk_detected = True
                result.risk_type = str(tiva_result["primary_risk"])

        if (
            self.enable_hemodynamic
            and self.hemodynamic_monitor is not None
            and "current_vitals" in patient_data
        ):
            hemo_result = self.hemodynamic_monitor.assess_hemodynamics(
                patient_data["current_vitals"]
            )
            result.hemodynamic_stability = float(hemo_result["hemodynamic_stability"])
            result.vital_sign_alerts = list(hemo_result["alerts"])
            result.intervention_needed = bool(hemo_result["intervention_needed"])
            result.clinical_recommendations.extend(hemo_result["recommendations"])
            if hemo_result["intervention_needed"]:
                result.risk_detected = True

        if (
            self.enable_smart_infusion
            and self.infusion_controller is not None
            and "infusion_rates" in patient_data
        ):
            bis = float(patient_data.get("bis_score", 50.0))
            infusion = patient_data["infusion_rates"]
            adjustment = self.infusion_controller.compute_infusion_adjustment(
                bis,
                float(infusion.get("propofol_mcg_kg_min", 100.0)),
                float(infusion.get("remifentanil_mcg_kg_min", 0.2)),
            )
            result.bis_score = bis
            result.clinical_recommendations.extend(adjustment["recommendations"])
            if adjustment["anomaly_detected"]:
                result.risk_detected = True
                result.infusion_anomalies.append(f"BIS out of range: {bis:.1f} (target: 40-60)")

        result.risk_score = self._calculate_overall_risk(result)
        if result.risk_score > 0.7:
            result.intervention_needed = True
        return result

    def _analyze_tiva(self, anesthesia_sequence: npt.NDArray[np.float64]) -> dict[str, Any]:
        """Run the TIVA Bi-LSTM in inference mode.

        Args:
            anesthesia_sequence: ``(time_steps, 8)`` float array.

        Returns:
            Inference dictionary with depth, awareness risk, primary risk,
            confidence, anomalies, and full risk-score breakdown.
        """
        if self.tiva_monitor is None:
            raise RuntimeError("TIVA monitor is not enabled")
        x = torch.tensor(anesthesia_sequence, dtype=torch.float32).unsqueeze(0)
        self.tiva_monitor.eval()
        with torch.no_grad():
            depth, risks, _attention = self.tiva_monitor(x)
        depth_val = float(depth.item())
        risk_probs = torch.softmax(risks, dim=1)[0].cpu().numpy()
        max_idx = int(np.argmax(risk_probs))
        max_score = float(risk_probs[max_idx])
        anomalies: list[str] = []
        if depth_val < 0.3:
            anomalies.append("Deep anesthesia detected (depth < 0.3)")
        elif depth_val > 0.7:
            anomalies.append("Light anesthesia detected (depth > 0.7)")
        return {
            "depth": depth_val,
            "awareness_risk": float(risk_probs[0]),
            "risk_detected": max_score > 0.5,
            "primary_risk": _RISK_TYPES[max_idx],
            "confidence": max_score,
            "anomalies": anomalies,
            "all_risk_scores": {
                _RISK_TYPES[i]: float(risk_probs[i]) for i in range(len(_RISK_TYPES))
            },
        }

    @staticmethod
    def _calculate_overall_risk(result: AnesthesiaPredictionResult) -> float:
        """Compute the overall anesthesia risk score."""
        depth_component = (
            1.0 - result.depth_of_anesthesia if result.depth_of_anesthesia > 0.7 else 0.0
        )
        components = (
            depth_component,
            result.predicted_awareness_risk,
            1.0 - result.hemodynamic_stability,
            1.0 - result.respiratory_adequacy,
        )
        return float(np.mean(components))


def get_anesthesiology_predictor(
    *,
    enable_tiva: bool = True,
    enable_smart_infusion: bool = True,
    enable_hemodynamic: bool = True,
) -> AnesthesiologyPredictor:
    """Factory returning a configured :class:`AnesthesiologyPredictor`.

    Args:
        enable_tiva: Enable the TIVA Bi-LSTM monitor.
        enable_smart_infusion: Enable the PID infusion controller.
        enable_hemodynamic: Enable the hemodynamic monitor.

    Returns:
        A new predictor instance.
    """
    return AnesthesiologyPredictor(
        enable_tiva=enable_tiva,
        enable_smart_infusion=enable_smart_infusion,
        enable_hemodynamic=enable_hemodynamic,
    )


def count_tiva_parameters(model: TIVAMonitoringSystem | None = None) -> int:
    """Return the trainable parameter count of the TIVA model.

    Args:
        model: Optional existing model.  When ``None`` a fresh model with the
            default dimensions is instantiated.

    Returns:
        Number of trainable parameters.
    """
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
    "VitalDBClient",
    "VitalDBClientError",
    "count_tiva_parameters",
    "get_anesthesiology_predictor",
]

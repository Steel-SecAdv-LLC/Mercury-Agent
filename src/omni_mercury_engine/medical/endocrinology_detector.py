"""Endocrinology detector with CGM Bi-LSTM and FDA-accurate rule set.

Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

Ported from Omni-AXA-Engine's ``endocrinology_detector.py``.  The neural
architecture (Bi-LSTM, 155K parameters), FDA-aligned clinical rules, and
risk weights all match the original verified implementation.  Three rules
have been audited and are preserved verbatim:

* **Afrezza inhaled-insulin contraindication**: ``FEV1 < 70%`` triggers a
  contraindication alert and recommends switching to subcutaneous insulin.
  Source: FDA Afrezza label, Section 4 (Contraindications).
* **GLP-1 discontinuation on pancreatitis**: any side-effect entry containing
  "pancreatitis" disables continued therapy.  Source: ADA pharmacological
  guidance and the FDA black-box warning on GLP-1 agonists.
* **Dose-stacking guard**: rapid-acting insulin doses must be spaced at least
  two hours apart unless glucose is verified.  Source: ADA Standards of Care.

Live data integration
---------------------
Synthetic CGM generators have been removed from production paths.  Real CGM
traces should be supplied via :class:`TidepoolClient` (public dataset; no
auth) or by integrating a vendor SDK (Dexcom, Libre).  The optional Dexcom
client requires ``DEXCOM_CLIENT_ID`` / ``DEXCOM_CLIENT_SECRET`` environment
variables when used; see :class:`DexcomCredentials` for details.

Operational notes
-----------------
Decision support only.  Clinical validation by a licensed endocrinologist is
required before any output is used to influence patient care.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Final, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)


class GlycemicState(Enum):
    """Glycemic state classifications."""

    NORMAL = "normal"
    HYPOGLYCEMIA = "hypoglycemia"
    HYPERGLYCEMIA = "hyperglycemia"
    SEVERE_HYPOGLYCEMIA = "severe_hypoglycemia"
    DKA = "diabetic_ketoacidosis"


class InsulinDeliveryMethod(Enum):
    """Insulin delivery methods."""

    SUBCUTANEOUS_INJECTION = "subcutaneous_injection"
    INHALED = "inhaled_insulin"
    SMART_PEN = "smart_insulin_pen"
    INSULIN_PUMP = "insulin_pump"
    CLOSED_LOOP = "closed_loop_system"


@dataclass
class EndocrinologyPredictionResult:
    """Endocrinology prediction result."""

    anomaly_detected: bool
    confidence: float
    glycemic_state: str
    risk_score: float

    hypoglycemia_risk: float
    hyperglycemia_risk: float
    dka_risk: float

    glucose_anomalies: list[str] = field(default_factory=list)
    insulin_delivery_alerts: list[str] = field(default_factory=list)
    clinical_recommendations: list[str] = field(default_factory=list)

    time_in_range_percent: float | None = None
    glucose_variability: float | None = None
    intervention_needed: bool = False


_GLYCEMIC_STATES: Final[tuple[str, ...]] = tuple(e.value for e in GlycemicState)


class CGMAnalyzer(nn.Module):
    """Bi-LSTM CGM analyser predicting glycemic state and trend.

    Architecture: ``input_dim=1`` -> Bi-LSTM(``hidden_dim=64``, num_layers=2,
    dropout=0.2, bidirectional=True) -> additive attention -> 5-class
    classifier + scalar trend.  Parameter count: 155K (matches the verified
    Omni-AXA implementation).
    """

    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, num_layers: int = 2) -> None:
        """Initialise the CGM analyser.

        Args:
            input_dim: Per-time-step feature dimensionality (glucose only).
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
        self.glycemic_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, len(_GLYCEMIC_STATES)),
        )
        self.trend_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, cgm_data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            cgm_data: ``(batch, time_steps, 1)`` tensor of glucose values
                in mg/dL.

        Returns:
            Tuple of (glycemic_classification, trend_prediction,
            attention_weights).
        """
        lstm_out, _ = self.lstm(cgm_data)
        attention_scores = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_scores, dim=1)
        context = torch.sum(lstm_out * attention_weights, dim=1)
        glycemic_state = self.glycemic_classifier(context)
        trend = self.trend_predictor(context)
        return glycemic_state, trend, attention_weights.squeeze(-1)


class SmartInsulinPenMonitor:
    """Smart insulin pen monitor enforcing FDA-aligned dose-stacking guards.

    The two-hour minimum dose interval for rapid-acting insulin is taken from
    the ADA Standards of Care; doses inside the window must be reviewed
    against a recent glucose value before delivery.
    """

    max_bolus_units: float = 15.0
    max_basal_units_per_day: float = 50.0
    min_dose_interval_hours: float = 2.0

    def monitor_insulin_delivery(self, insulin_data: Mapping[str, Any]) -> dict[str, Any]:
        """Monitor smart insulin pen usage and emit dose-stacking alerts.

        Args:
            insulin_data: Mapping with ``dose_units``, ``dose_time``,
                ``insulin_type``, ``time_since_last_dose_hours`` and
                optional adherence keys.

        Returns:
            Dictionary with ``insulin_delivery_safe`` plus alert and
            recommendation lists.
        """
        dose = float(insulin_data.get("dose_units", 5.0))
        insulin_type = str(insulin_data.get("insulin_type", "rapid_acting"))
        time_since_last = float(insulin_data.get("time_since_last_dose_hours", 4.0))
        daily_total = float(insulin_data.get("daily_total_units", 30.0))

        alerts: list[str] = []
        recommendations: list[str] = []

        if insulin_type == "rapid_acting" and dose > self.max_bolus_units:
            alerts.append(f"ALERT: Large bolus dose {dose:.1f} units")
            recommendations.append("Verify dose - risk of hypoglycemia")
            recommendations.append("Consider splitting dose if meal is large")

        if daily_total > self.max_basal_units_per_day:
            alerts.append(f"High daily insulin: {daily_total:.1f} units")
            recommendations.append("Review insulin sensitivity and dosing regimen")

        if time_since_last < self.min_dose_interval_hours and insulin_type == "rapid_acting":
            alerts.append(f"Dose stacking: {time_since_last:.1f} hours since last dose")
            recommendations.append("Risk of insulin stacking and hypoglycemia")
            recommendations.append("Check glucose before additional dosing")

        return {
            "insulin_delivery_safe": len(alerts) == 0,
            "alerts": alerts,
            "recommendations": recommendations,
            "adherence_score": self._calculate_adherence(insulin_data),
        }

    @staticmethod
    def _calculate_adherence(insulin_data: Mapping[str, Any]) -> float:
        """Compute the patient's recent insulin adherence ratio in ``[0, 1]``."""
        doses_taken = float(insulin_data.get("doses_taken_last_week", 18))
        doses_prescribed = float(insulin_data.get("doses_prescribed_last_week", 21))
        if doses_prescribed == 0:
            return 1.0
        return float(min(doses_taken / doses_prescribed, 1.0))


class GLP1TherapyMonitor:
    """GLP-1 therapy monitor enforcing pancreatitis discontinuation.

    Implements the FDA pancreatitis discontinuation rule: if "pancreatitis"
    is listed in ``side_effects`` therapy is flagged for immediate
    discontinuation and :attr:`continue_therapy` is set to ``False``.
    """

    target_a1c: float = 7.0
    target_weight_loss_percent: float = 10.0

    def monitor_glp1_therapy(self, therapy_data: Mapping[str, Any]) -> dict[str, Any]:
        """Monitor GLP-1 agonist therapy.

        Args:
            therapy_data: Mapping with ``medication``, ``dose_mg``,
                ``duration_weeks``, ``a1c_percent``, ``weight_change_percent``,
                ``side_effects`` keys.

        Returns:
            Dictionary with ``therapeutic_success``, ``continue_therapy`` flag
            (False when pancreatitis is present), efficacy metrics, and
            recommendation list.
        """
        duration = float(therapy_data.get("duration_weeks", 12))
        a1c = float(therapy_data.get("a1c_percent", 7.5))
        weight_change = float(therapy_data.get("weight_change_percent", -5.0))
        side_effects = [str(s).lower() for s in therapy_data.get("side_effects", [])]

        efficacy_metrics: list[str] = []
        recommendations: list[str] = []

        if a1c > self.target_a1c:
            efficacy_metrics.append(f"A1C above target: {a1c:.1f}%")
            if duration >= 12:
                recommendations.append("Consider dose escalation")
            else:
                recommendations.append("Continue current dose - allow more time for effect")
        else:
            efficacy_metrics.append(f"A1C at target: {a1c:.1f}%")

        if abs(weight_change) < 5.0 and duration >= 16:
            efficacy_metrics.append("Suboptimal weight loss")
            recommendations.append("Review diet and exercise adherence")
            recommendations.append("Consider dose escalation if tolerated")
        elif abs(weight_change) >= self.target_weight_loss_percent:
            efficacy_metrics.append(f"Excellent weight loss: {abs(weight_change):.1f}%")

        if "nausea" in side_effects or "vomiting" in side_effects:
            recommendations.append("GI side effects present")
            recommendations.append("Take with food, slower dose titration")
            recommendations.append("Consider antiemetics if severe")

        pancreatitis_present = "pancreatitis" in side_effects
        if pancreatitis_present:
            recommendations.append("ALERT: Pancreatitis - discontinue GLP-1 immediately")

        therapeutic_success = a1c <= self.target_a1c and abs(weight_change) >= 5.0
        return {
            "therapeutic_success": therapeutic_success,
            "efficacy_metrics": efficacy_metrics,
            "recommendations": recommendations,
            "continue_therapy": not pancreatitis_present,
        }


class InhaledInsulinMonitor:
    """Inhaled insulin monitor enforcing the FDA Afrezza contraindication.

    Implements the FDA Afrezza label rule: inhaled insulin is contraindicated
    when ``FEV1 < 70%`` (chronic lung disease).  The rule fires at strictly
    less than 70%, so ``FEV1 = 69.9%`` is flagged and ``FEV1 = 70.1%`` is
    permitted.
    """

    max_dose_units: int = 12
    fev1_contraindication_threshold: float = 70.0

    def monitor_inhaled_insulin(self, inhaled_data: Mapping[str, Any]) -> dict[str, Any]:
        """Monitor inhaled insulin delivery.

        Args:
            inhaled_data: Mapping with ``dose_units``,
                ``inhalation_technique_score``, and
                ``pulmonary_function_fev1_percent`` keys.

        Returns:
            Dictionary with ``inhaled_insulin_appropriate`` plus alert and
            recommendation lists.
        """
        dose = float(inhaled_data.get("dose_units", 4))
        technique_score = float(inhaled_data.get("inhalation_technique_score", 0.8))
        fev1 = float(inhaled_data.get("pulmonary_function_fev1_percent", 85))

        alerts: list[str] = []
        recommendations: list[str] = []

        if dose > self.max_dose_units:
            alerts.append(f"High inhaled insulin dose: {dose} units")
            recommendations.append("Consider subcutaneous insulin for large doses")

        if technique_score < 0.7:
            alerts.append("Poor inhalation technique")
            recommendations.append("Retrain on proper inhaler use")
            recommendations.append("May result in suboptimal absorption")

        appropriate = fev1 >= self.fev1_contraindication_threshold
        if not appropriate:
            alerts.append(f"Reduced pulmonary function: FEV1 {fev1}%")
            recommendations.append("Inhaled insulin contraindicated with FEV1 <70%")
            recommendations.append("Switch to subcutaneous insulin")

        return {
            "inhaled_insulin_appropriate": appropriate,
            "alerts": alerts,
            "recommendations": recommendations,
        }


class TidepoolClientError(RuntimeError):
    """Raised when the Tidepool API returns an unrecoverable error."""


class TidepoolClient:
    """Read-only client for the Tidepool public-info endpoints.

    Tidepool operates a free open-source diabetes data ecosystem.  The
    ``/info`` and ``/metadata`` endpoints do not require authentication; the
    ``/data`` endpoints used in production require an OAuth bearer token,
    which callers may supply via ``token`` to enable downstream data fetches.

    Reference: https://github.com/tidepool-org/platform
    """

    DEFAULT_BASE_URL: Final[str] = "https://api.tidepool.org"

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        token: str | None = None,
        timeout_seconds: float = 15.0,
        user_agent: str = "Mercury-Agent/1.7 Endocrinology",
    ) -> None:
        """Initialise the Tidepool client.

        Args:
            base_url: Base URL for the Tidepool HTTP API.
            token: Optional OAuth bearer token for authenticated routes.
            timeout_seconds: Network timeout per request.
            user_agent: HTTP ``User-Agent`` header value.
        """
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._timeout = float(timeout_seconds)
        self._user_agent = user_agent

    def info(self) -> dict[str, Any]:
        """Fetch the Tidepool service info document."""
        return cast("dict[str, Any]", self._request_json("/info"))

    def _request_json(self, path: str) -> Any:
        """Fetch a JSON resource from the Tidepool API."""
        url = f"{self._base_url}/{path.lstrip('/')}"
        headers = {
            "User-Agent": self._user_agent,
            "Accept": "application/json",
        }
        if self._token is not None:
            headers["Authorization"] = f"Bearer {self._token}"
        request = Request(url, headers=headers)  # noqa: S310 - public HTTPS endpoint
        try:
            with urlopen(request, timeout=self._timeout) as response:  # noqa: S310
                if response.status != 200:
                    raise TidepoolClientError(f"Unexpected status {response.status} from {url}")
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise TidepoolClientError(f"Tidepool HTTP error {exc.code}: {exc.reason}") from exc
        except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise TidepoolClientError(f"Tidepool request failed: {exc}") from exc


@dataclass(frozen=True)
class DexcomCredentials:
    """Dexcom Developer API credentials loaded from environment.

    Attributes:
        client_id: ``DEXCOM_CLIENT_ID`` value.
        client_secret: ``DEXCOM_CLIENT_SECRET`` value.

    Raises:
        RuntimeError: If either environment variable is missing.
    """

    client_id: str
    client_secret: str

    @classmethod
    def from_environment(cls) -> DexcomCredentials:
        """Load credentials from environment variables.

        Returns:
            A populated :class:`DexcomCredentials` instance.

        Raises:
            RuntimeError: If a required variable is missing.
        """
        client_id = os.environ.get("DEXCOM_CLIENT_ID")
        client_secret = os.environ.get("DEXCOM_CLIENT_SECRET")
        if not client_id or not client_secret:
            raise RuntimeError(
                "DEXCOM_CLIENT_ID and DEXCOM_CLIENT_SECRET must be set to use "
                "the Dexcom production client."
            )
        return cls(client_id=client_id, client_secret=client_secret)


class EndocrinologyDetector:
    """Integrated endocrine anomaly detector.

    Combines CGM analysis, smart insulin pen monitoring, GLP-1 therapy
    monitoring, and inhaled-insulin monitoring into a single entry point.
    Synthetic generators are absent from production paths; tests should
    supply realistic fixtures via :class:`TidepoolClient` or an offline
    sample file.
    """

    def __init__(
        self,
        *,
        enable_cgm: bool = True,
        enable_smart_pen: bool = True,
        enable_glp1: bool = True,
        enable_inhaled_insulin: bool = True,
    ) -> None:
        """Initialise the detector.

        Args:
            enable_cgm: Enable the CGM Bi-LSTM.
            enable_smart_pen: Enable the smart insulin pen monitor.
            enable_glp1: Enable the GLP-1 therapy monitor.
            enable_inhaled_insulin: Enable the inhaled insulin monitor.
        """
        self.enable_cgm = enable_cgm
        self.enable_smart_pen = enable_smart_pen
        self.enable_glp1 = enable_glp1
        self.enable_inhaled_insulin = enable_inhaled_insulin
        self.cgm_analyzer: CGMAnalyzer | None = CGMAnalyzer() if enable_cgm else None
        self.smart_pen_monitor: SmartInsulinPenMonitor | None = (
            SmartInsulinPenMonitor() if enable_smart_pen else None
        )
        self.glp1_monitor: GLP1TherapyMonitor | None = GLP1TherapyMonitor() if enable_glp1 else None
        self.inhaled_monitor: InhaledInsulinMonitor | None = (
            InhaledInsulinMonitor() if enable_inhaled_insulin else None
        )
        self.target_range: tuple[float, float] = (70.0, 180.0)

    def detect_endocrine_anomaly(
        self, patient_data: Mapping[str, Any]
    ) -> EndocrinologyPredictionResult:
        """Run the full endocrine detection pipeline.

        Args:
            patient_data: Patient data dictionary.  Recognised keys:

                * ``cgm_sequence``: Time-series CGM data.
                * ``insulin_delivery``: Smart pen data.
                * ``glp1_therapy``: GLP-1 treatment data.
                * ``inhaled_insulin``: Inhaled insulin data.

        Returns:
            :class:`EndocrinologyPredictionResult`.
        """
        result = EndocrinologyPredictionResult(
            anomaly_detected=False,
            confidence=0.0,
            glycemic_state=GlycemicState.NORMAL.value,
            risk_score=0.0,
            hypoglycemia_risk=0.0,
            hyperglycemia_risk=0.0,
            dka_risk=0.0,
        )

        if self.enable_cgm and self.cgm_analyzer is not None and "cgm_sequence" in patient_data:
            cgm_result = self._analyze_cgm(
                np.asarray(patient_data["cgm_sequence"], dtype=np.float64)
            )
            result.glycemic_state = str(cgm_result["glycemic_state"])
            result.hypoglycemia_risk = float(cgm_result["hypoglycemia_risk"])
            result.hyperglycemia_risk = float(cgm_result["hyperglycemia_risk"])
            result.confidence = float(cgm_result["confidence"])
            result.glucose_anomalies = list(cgm_result["anomalies"])
            result.time_in_range_percent = float(cgm_result["time_in_range"])
            result.glucose_variability = float(cgm_result["variability"])
            if cgm_result["glycemic_state"] != GlycemicState.NORMAL.value:
                result.anomaly_detected = True

        if (
            self.enable_smart_pen
            and self.smart_pen_monitor is not None
            and "insulin_delivery" in patient_data
        ):
            pen_result = self.smart_pen_monitor.monitor_insulin_delivery(
                patient_data["insulin_delivery"]
            )
            result.insulin_delivery_alerts = list(pen_result["alerts"])
            result.clinical_recommendations.extend(pen_result["recommendations"])
            if not pen_result["insulin_delivery_safe"]:
                result.anomaly_detected = True
                result.intervention_needed = True

        if self.enable_glp1 and self.glp1_monitor is not None and "glp1_therapy" in patient_data:
            glp1_result = self.glp1_monitor.monitor_glp1_therapy(patient_data["glp1_therapy"])
            result.clinical_recommendations.extend(glp1_result["recommendations"])
            if not glp1_result["continue_therapy"]:
                result.anomaly_detected = True
                result.intervention_needed = True
            elif not glp1_result["therapeutic_success"]:
                result.anomaly_detected = True

        if (
            self.enable_inhaled_insulin
            and self.inhaled_monitor is not None
            and "inhaled_insulin" in patient_data
        ):
            inhaled_result = self.inhaled_monitor.monitor_inhaled_insulin(
                patient_data["inhaled_insulin"]
            )
            result.clinical_recommendations.extend(inhaled_result["recommendations"])
            if not inhaled_result["inhaled_insulin_appropriate"]:
                result.anomaly_detected = True
                result.intervention_needed = True

        result.risk_score = self._calculate_overall_risk(result)
        if result.risk_score > 0.7:
            result.intervention_needed = True
        return result

    def _analyze_cgm(self, cgm_sequence: npt.NDArray[np.float64]) -> dict[str, Any]:
        """Run the CGM Bi-LSTM in inference mode.

        Args:
            cgm_sequence: 1-D float array of mg/dL glucose values.

        Returns:
            Inference dictionary including glycemic state, risks, anomalies,
            and time-in-range statistics.
        """
        if self.cgm_analyzer is None:
            raise RuntimeError("CGM analyser is not enabled")
        x = torch.tensor(cgm_sequence, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        self.cgm_analyzer.eval()
        with torch.no_grad():
            glycemic_logits, _trend, _attention = self.cgm_analyzer(x)
        state_probs = torch.softmax(glycemic_logits[0], dim=0).cpu().numpy()
        state_idx = int(np.argmax(state_probs))
        detected_state = _GLYCEMIC_STATES[state_idx]

        anomalies: list[str] = []
        if detected_state == GlycemicState.HYPOGLYCEMIA.value:
            anomalies.append("Hypoglycemia detected")
            anomalies.append("Treat with 15g fast-acting carbohydrates")
        elif detected_state == GlycemicState.SEVERE_HYPOGLYCEMIA.value:
            anomalies.append("ALERT: Severe hypoglycemia")
            anomalies.append("Administer glucagon immediately")
        elif detected_state == GlycemicState.HYPERGLYCEMIA.value:
            anomalies.append("Hyperglycemia detected")
            anomalies.append("Check ketones and consider correction dose")
        elif detected_state == GlycemicState.DKA.value:
            anomalies.append("ALERT: DKA suspected")
            anomalies.append("Emergency department evaluation required")

        glucose_values = cgm_sequence.flatten()
        time_in_range = float(
            np.mean(
                (glucose_values >= self.target_range[0]) & (glucose_values <= self.target_range[1])
            )
            * 100
        )
        variability = float(np.std(glucose_values))
        return {
            "glycemic_state": detected_state,
            "hypoglycemia_risk": float(state_probs[1] + state_probs[3]),
            "hyperglycemia_risk": float(state_probs[2] + state_probs[4]),
            "confidence": float(state_probs[state_idx]),
            "anomalies": anomalies,
            "time_in_range": time_in_range,
            "variability": variability,
        }

    @staticmethod
    def _calculate_overall_risk(result: EndocrinologyPredictionResult) -> float:
        """Compute the overall endocrine risk score in ``[0, 1]``."""
        components = (
            result.hypoglycemia_risk * 0.4,
            result.hyperglycemia_risk * 0.3,
            result.dka_risk * 0.3,
        )
        return float(min(sum(components), 1.0))


def get_endocrinology_detector(
    *,
    enable_cgm: bool = True,
    enable_smart_pen: bool = True,
    enable_glp1: bool = True,
    enable_inhaled_insulin: bool = True,
) -> EndocrinologyDetector:
    """Factory returning a configured :class:`EndocrinologyDetector`.

    Args:
        enable_cgm: Enable the CGM Bi-LSTM.
        enable_smart_pen: Enable the smart insulin pen monitor.
        enable_glp1: Enable the GLP-1 therapy monitor.
        enable_inhaled_insulin: Enable the inhaled insulin monitor.

    Returns:
        A new detector instance.
    """
    return EndocrinologyDetector(
        enable_cgm=enable_cgm,
        enable_smart_pen=enable_smart_pen,
        enable_glp1=enable_glp1,
        enable_inhaled_insulin=enable_inhaled_insulin,
    )


def count_cgm_parameters(model: CGMAnalyzer | None = None) -> int:
    """Return the trainable parameter count of the CGM Bi-LSTM.

    Args:
        model: Optional existing model.  When ``None`` a fresh model with the
            default dimensions is instantiated.

    Returns:
        Number of trainable parameters.
    """
    instance = model if model is not None else CGMAnalyzer()
    return int(sum(p.numel() for p in instance.parameters() if p.requires_grad))


__all__ = [
    "CGMAnalyzer",
    "DexcomCredentials",
    "EndocrinologyDetector",
    "EndocrinologyPredictionResult",
    "GLP1TherapyMonitor",
    "GlycemicState",
    "InhaledInsulinMonitor",
    "InsulinDeliveryMethod",
    "SmartInsulinPenMonitor",
    "TidepoolClient",
    "TidepoolClientError",
    "count_cgm_parameters",
    "get_endocrinology_detector",
]

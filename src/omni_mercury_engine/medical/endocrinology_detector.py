# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Endocrinology detector with CGM Bi-LSTM and FDA-accurate rule set.

version.

Ported from Omni-AXA-Engine's ``endocrinology_detector.py``.  The neural
architecture is a Bi-LSTM with additive attention of approximately the
same parameter count as the upstream (~155K), and the FDA-aligned
clinical rules are preserved verbatim with their citations intact.
Two deviations from the upstream are recorded explicitly in
``CHANGELOG.md`` under
*"omni_mercury_engine.medical.endocrinology_detector - Deviations
from the original"* and must not be silently re-collapsed:

* :class:`CGMAnalyzer` widens the trend-prediction head from
  ``hidden_dim * 2 -> 32 -> 1`` to ``hidden_dim * 2 -> 64 -> 1`` to
  match the glycemic classifier's hidden width.  Parameter count
  stays approximately ~155K but the resulting weights are **not
  interchangeable** with upstream checkpoints.
* :class:`GLP1TherapyMonitor` and :class:`InhaledInsulinMonitor`
  retain Mercury-specific reinforcing rules (e.g. duration-based
  efficacy review, GI-side-effect tracking, MRD spirometry cadence)
  that the upstream did not ship.  These are additive only - no
  upstream rule has been weakened or removed.

The three FDA-aligned clinical rules below are preserved exactly as
upstream:

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
Mercury Agent ships integration-ready, not pre-integrated.  The detector
**requires** a :class:`~omni_mercury_engine.medical.data_sources.CGMDataSource`
adapter; instantiating the class with ``enable_cgm=True`` but no data source
raises :class:`~omni_mercury_engine.medical.data_sources.ConfigurationError`.

Two ways to provide a data source:

* Pass a configured adapter explicitly, e.g.
  ``EndocrinologyDetector(data_source=DexcomV3DataSource())``.  The reference
  adapter reads ``DEXCOM_CLIENT_ID`` / ``DEXCOM_CLIENT_SECRET`` /
  ``DEXCOM_REFRESH_TOKEN`` / ``DEXCOM_REDIRECT_URI`` from the environment.
* Implement :class:`CGMDataSource` for any other vendor (Abbott LibreView,
  Medtronic CareLink, etc.) and pass that instance.  The contract is the
  same; ``docs/medical/SETUP.md`` documents the extension point.

Operational notes
-----------------
required before any output is used to influence patient care.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from omni_mercury_engine.medical.data_sources import (
    CGMDataSource,
    CGMReading,
    ConfigurationError,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

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
    cgm_source: str | None = None
    cgm_reading_count: int = 0


_GLYCEMIC_STATES: Final[tuple[str, ...]] = tuple(e.value for e in GlycemicState)


class CGMAnalyzer(nn.Module):
    """Bi-LSTM CGM analyser predicting glycemic state and trend.

    Architecture: ``input_dim=1`` -> Bi-LSTM(``hidden_dim=64``, num_layers=2,
    dropout=0.2, bidirectional=True) -> additive attention -> 5-class
    classifier + scalar trend.  Parameter count: ~155K.

    Architecture derived from the upstream Omni-AXA ``CGMAnalyzer``, with
    the trend head widened from ``32`` to ``hidden_dim`` (``64``) units to
    match the glycemic classifier's hidden width.  Parameter count is
    approximately equal (~155K) but the resulting weights are **not
    interchangeable** with upstream checkpoints; any prior pretrained
    weights would need to be re-trained for this layout.  The guard test
    :func:`tests.test_endocrinology_detector.test_cgm_analyzer_parameter_count_is_approximately_155k`
    pins the count window.
    """

    def __init__(
        self,
        *,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        """Initialise the analyser network."""
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )
        self.attention_W = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attention_v = nn.Linear(hidden_dim, 1)
        self.glycemic_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(_GLYCEMIC_STATES)),
        )
        self.trend_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the analyser on a ``(batch, time, input_dim)`` tensor."""
        lstm_out, _ = self.lstm(x)
        attn_scores = torch.tanh(self.attention_W(lstm_out))
        attn_scores = self.attention_v(attn_scores).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        glycemic_logits = self.glycemic_classifier(context)
        trend = self.trend_head(context).squeeze(-1)
        return glycemic_logits, trend, attn_weights


class SmartInsulinPenMonitor:
    """Smart-pen monitor enforcing dose-stacking, bolus-ceiling and daily-total guards.

    Citations:

    * ADA Standards of Care - rapid-acting boluses spaced under two
      hours without verified glucose risk *insulin stacking*
      hypoglycaemia.
    * FDA insulin labeling and ADA Standards of Care - large boluses
      without sensitivity verification are a documented hypoglycemia
      risk; total daily insulin above ~50 U warrants a sensitivity /
      regimen review.
    """

    DOSE_STACK_WINDOW_HOURS: Final[float] = 2.0
    MAX_BOLUS_UNITS: Final[float] = 15.0
    MAX_DAILY_INSULIN_UNITS: Final[float] = 50.0

    def monitor_insulin_delivery(self, delivery_data: Mapping[str, Any]) -> dict[str, Any]:
        """Score smart-pen telemetry for delivery safety.

        Args:
            delivery_data: Mapping with ``recent_doses`` (sequence of dicts
                with ``time_hours`` and ``units`` keys), optional
                ``adherence_rate`` (0-1), optional ``patient_glucose``
                (mg/dL), optional ``dose_units`` (single bolus in units),
                optional ``insulin_type`` (``"rapid_acting"`` by default;
                bolus ceiling only fires for rapid-acting), and optional
                ``daily_total_units`` (cumulative insulin for the rolling
                24 h window).  Missing fields silently skip the relevant
                check so legacy callers keep working.

        Returns:
            Dictionary with ``insulin_delivery_safe`` flag, ``alerts`` and
            ``recommendations`` lists.
        """
        recent_doses: Sequence[Mapping[str, Any]] = delivery_data.get("recent_doses", [])
        glucose = delivery_data.get("patient_glucose")
        alerts: list[str] = []
        recommendations: list[str] = []

        for idx in range(1, len(recent_doses)):
            previous = recent_doses[idx - 1]
            current = recent_doses[idx]
            gap_hours = float(current["time_hours"]) - float(previous["time_hours"])
            if gap_hours < self.DOSE_STACK_WINDOW_HOURS:
                if glucose is None or float(glucose) > 250.0:
                    alerts.append(f"Possible insulin stacking: doses {gap_hours:.1f}h apart")
                    recommendations.append(
                        "Hold next rapid-acting dose; verify glucose before stacking"
                    )

        dose_units_raw = delivery_data.get("dose_units")
        insulin_type = str(delivery_data.get("insulin_type", "rapid_acting")).lower()
        if dose_units_raw is not None and insulin_type == "rapid_acting":
            try:
                dose_units = float(dose_units_raw)
            except (TypeError, ValueError):
                dose_units = 0.0
            if dose_units > self.MAX_BOLUS_UNITS:
                alerts.append(
                    f"Large rapid-acting bolus: {dose_units:.1f} U > "
                    f"{self.MAX_BOLUS_UNITS:.0f} U ceiling"
                )
                recommendations.append("Verify dose - risk of hypoglycemia")
                recommendations.append("Consider splitting dose if meal is large")

        daily_total_raw = delivery_data.get("daily_total_units")
        if daily_total_raw is not None:
            try:
                daily_total = float(daily_total_raw)
            except (TypeError, ValueError):
                daily_total = 0.0
            if daily_total > self.MAX_DAILY_INSULIN_UNITS:
                alerts.append(
                    f"High daily insulin total: {daily_total:.1f} U > "
                    f"{self.MAX_DAILY_INSULIN_UNITS:.0f} U"
                )
                recommendations.append("Review insulin sensitivity and dosing regimen")

        adherence = self._calculate_adherence(delivery_data)
        if adherence < 0.8:
            alerts.append(f"Low adherence: {adherence * 100:.0f}%")
            recommendations.append("Counsel patient on dosing schedule")

        return {
            "insulin_delivery_safe": not alerts,
            "alerts": alerts,
            "recommendations": recommendations,
            "adherence_rate": adherence,
        }

    @staticmethod
    def _calculate_adherence(delivery_data: Mapping[str, Any]) -> float:
        """Return adherence as a 0-1 fraction; missing data is treated as full."""
        raw = delivery_data.get("adherence_rate", 1.0)
        try:
            return max(0.0, min(1.0, float(raw)))
        except (TypeError, ValueError):
            return 1.0


class GLP1TherapyMonitor:
    """GLP-1 therapy monitor with FDA pancreatitis discontinuation rule.

    Citations:

    * FDA black-box warning on GLP-1 receptor agonists (e.g. semaglutide,
      liraglutide) - history or signal of pancreatitis is a
      discontinuation indication.
    * ADA pharmacological guidance and FDA semaglutide / liraglutide
      labeling - dose-titration windows (review A1C and weight response
      at week 12 / 16) and GI side-effect management (slower titration,
      take with food, antiemetics when severe).
    """

    A1C_ESCALATION_WEEK: Final[int] = 12
    A1C_INADEQUATE_DROP_PERCENT: Final[float] = -0.5
    WEIGHT_LOSS_REVIEW_WEEK: Final[int] = 16
    WEIGHT_LOSS_TARGET_KG: Final[float] = 2.5

    def monitor_glp1_therapy(self, therapy_data: Mapping[str, Any]) -> dict[str, Any]:
        """Score GLP-1 therapy telemetry.

        Args:
            therapy_data: Mapping with ``side_effects`` (iterable of strings),
                ``a1c_change_percent``, ``weight_loss_kg``, ``medication``,
                and optional ``duration_weeks`` (treatment duration; the
                A1C / weight-loss escalation rules only fire once the
                relevant titration window has elapsed).  ``side_effects``
                entries containing ``"nausea"`` or ``"vomiting"`` trigger
                titration / antiemetic guidance.

        Returns:
            Dictionary with ``continue_therapy``, ``therapeutic_success``,
            and ``recommendations`` keys.
        """
        side_effects = [str(s).lower() for s in therapy_data.get("side_effects", [])]
        a1c_change = float(therapy_data.get("a1c_change_percent", 0.0))
        weight_loss = float(therapy_data.get("weight_loss_kg", 0.0))
        medication = str(therapy_data.get("medication", "GLP-1 agonist"))
        duration_weeks = int(therapy_data.get("duration_weeks", 0))

        recommendations: list[str] = []
        continue_therapy = True

        if any("pancreatitis" in s for s in side_effects):
            continue_therapy = False
            recommendations.append(f"DISCONTINUE {medication}: pancreatitis signal (FDA black-box)")
            recommendations.append("Switch to alternative agent (e.g. SGLT2 inhibitor)")

        therapeutic_success = a1c_change <= -0.5 and weight_loss >= 2.5
        if continue_therapy:
            if (
                a1c_change > self.A1C_INADEQUATE_DROP_PERCENT
                and duration_weeks >= self.A1C_ESCALATION_WEEK
            ):
                recommendations.append(
                    "Inadequate A1C response after "
                    f"{duration_weeks} weeks; consider dose escalation per FDA label"
                )
            if (
                abs(weight_loss) < self.WEIGHT_LOSS_TARGET_KG
                and duration_weeks >= self.WEIGHT_LOSS_REVIEW_WEEK
            ):
                recommendations.append(
                    "Weight-loss response below ADA benchmark after "
                    f"{duration_weeks} weeks; review diet / exercise and "
                    "escalate dose if tolerated"
                )
            if not therapeutic_success and not recommendations:
                recommendations.append(
                    "Therapeutic response is below ADA benchmarks; "
                    "reassess dose escalation or adherence"
                )
            if any("nausea" in s or "vomiting" in s for s in side_effects):
                recommendations.append(
                    "GI side effects: take with food, slower dose titration; "
                    "consider antiemetics if severe"
                )

        return {
            "continue_therapy": continue_therapy,
            "therapeutic_success": therapeutic_success,
            "recommendations": recommendations,
            "a1c_change_percent": a1c_change,
            "weight_loss_kg": weight_loss,
            "duration_weeks": duration_weeks,
        }


class InhaledInsulinMonitor:
    """Inhaled-insulin (Afrezza) monitor enforcing FEV1, dose-ceiling and technique guards.

    Citations:

    * FDA Afrezza label, Section 4 (Contraindications) - patients with
      FEV1 < 70 % predicted must not receive inhaled insulin due to the
      risk of acute bronchospasm.
    * FDA Afrezza label, Section 5 (Warnings and Precautions) - large
      doses warrant consideration of subcutaneous insulin.
    * AARC inhaler-technique guidance - sub-threshold inhalation
      technique scores predict suboptimal absorption.
    """

    FEV1_THRESHOLD: Final[float] = 70.0
    MAX_DOSE_UNITS: Final[int] = 12
    MIN_TECHNIQUE_SCORE: Final[float] = 0.7

    def monitor_inhaled_insulin(self, inhaled_data: Mapping[str, Any]) -> dict[str, Any]:
        """Score Afrezza telemetry for appropriateness.

        Args:
            inhaled_data: Mapping with ``fev1_percent`` (predicted), optional
                ``post_meal_glucose`` (mg/dL), optional ``dose_units`` (single
                inhaled dose in units; ceiling fires above
                :attr:`MAX_DOSE_UNITS`), and optional
                ``inhalation_technique_score`` (0-1; technique alert fires
                below :attr:`MIN_TECHNIQUE_SCORE`).  Missing fields silently
                skip the relevant check so legacy callers keep working.

        Returns:
            Dictionary with ``inhaled_insulin_appropriate``, ``alerts``, and
            ``recommendations`` keys.
        """
        fev1 = float(inhaled_data.get("fev1_percent", 100.0))
        post_meal = inhaled_data.get("post_meal_glucose")
        dose_units_raw = inhaled_data.get("dose_units")
        technique_raw = inhaled_data.get("inhalation_technique_score")

        alerts: list[str] = []
        recommendations: list[str] = []
        appropriate = True

        if fev1 < self.FEV1_THRESHOLD:
            appropriate = False
            alerts.append(f"CONTRAINDICATION: FEV1 {fev1:.0f}% < 70%")
            recommendations.append("Discontinue inhaled insulin; switch to subcutaneous insulin")

        if dose_units_raw is not None:
            try:
                dose_units = float(dose_units_raw)
            except (TypeError, ValueError):
                dose_units = 0.0
            if dose_units > self.MAX_DOSE_UNITS:
                alerts.append(f"Inhaled dose {dose_units:.0f} U > {self.MAX_DOSE_UNITS} U ceiling")
                recommendations.append("Consider subcutaneous insulin for large doses")

        if technique_raw is not None:
            try:
                technique = float(technique_raw)
            except (TypeError, ValueError):
                technique = 1.0
            if technique < self.MIN_TECHNIQUE_SCORE:
                alerts.append(
                    f"Inhalation technique {technique:.2f} < " f"{self.MIN_TECHNIQUE_SCORE:.2f}"
                )
                recommendations.append("Retrain on proper inhaler use")
                recommendations.append("May result in suboptimal absorption")

        if post_meal is not None and float(post_meal) > 180.0:
            recommendations.append("Post-meal glucose above target; consider dose adjustment")

        return {
            "inhaled_insulin_appropriate": appropriate,
            "alerts": alerts,
            "recommendations": recommendations,
        }


class EndocrinologyDetector:
    """Integrated endocrine anomaly detector.

    Combines CGM analysis, smart insulin pen monitoring, GLP-1 therapy
    monitoring, and inhaled-insulin monitoring.  The detector is the
    platform integration unit: it **requires** a configured
    :class:`~omni_mercury_engine.medical.data_sources.CGMDataSource` whenever
    ``enable_cgm`` is true.

    Synthetic generators have been removed from production paths.  Operators
    who need to apply individual rule monitors to pre-loaded data without a
    live feed can instantiate :class:`SmartInsulinPenMonitor`,
    :class:`GLP1TherapyMonitor`, and :class:`InhaledInsulinMonitor` directly.
    """

    def __init__(
        self,
        data_source: CGMDataSource | None = None,
        *,
        enable_cgm: bool = True,
        enable_smart_pen: bool = True,
        enable_glp1: bool = True,
        enable_inhaled_insulin: bool = True,
    ) -> None:
        """Initialise the detector.

        Args:
            data_source: Configured CGM adapter (e.g. a
                :class:`~omni_mercury_engine.medical.data_sources.DexcomV3DataSource`
                or a custom subclass of :class:`CGMDataSource`).  Required
                whenever ``enable_cgm`` is true.
            enable_cgm: Enable the CGM Bi-LSTM.
            enable_smart_pen: Enable the smart insulin pen monitor.
            enable_glp1: Enable the GLP-1 therapy monitor.
            enable_inhaled_insulin: Enable the inhaled insulin monitor.

        Raises:
            ConfigurationError: If ``enable_cgm`` is true and ``data_source``
                is ``None``.  Mercury Agent never invents glucose readings;
                operators must wire a real adapter before instantiating.
            TypeError: If ``data_source`` is not a :class:`CGMDataSource`.
        """
        if enable_cgm and data_source is None:
            raise ConfigurationError(
                "EndocrinologyDetector requires a configured CGMDataSource "
                "when CGM analysis is enabled. Mercury Agent does not ship "
                "with default credentials. See docs/medical/SETUP.md for "
                "instructions on configuring a CGM adapter (Dexcom v3 "
                "reference implementation provided)."
            )
        if data_source is not None and not isinstance(data_source, CGMDataSource):
            raise TypeError(
                "data_source must subclass CGMDataSource; " f"got {type(data_source).__name__}"
            )
        self.data_source = data_source
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

    def fetch_and_detect(
        self,
        *,
        window_minutes: int = 180,
        patient_context: Mapping[str, Any] | None = None,
    ) -> EndocrinologyPredictionResult:
        """Fetch the latest CGM window then run the full detection pipeline.

        Args:
            window_minutes: Look-back window passed to
                :meth:`CGMDataSource.fetch_recent_readings`.
            patient_context: Optional supplementary patient data (e.g.
                ``insulin_delivery``, ``glp1_therapy``, ``inhaled_insulin``)
                merged into the detection input.

        Returns:
            :class:`EndocrinologyPredictionResult`.

        Raises:
            ConfigurationError: If ``enable_cgm`` is false or no data
                source was supplied at construction time.
        """
        if not self.enable_cgm or self.data_source is None:
            raise ConfigurationError(
                "fetch_and_detect requires enable_cgm=True and a configured "
                "data_source. Use detect_endocrine_anomaly() with pre-loaded "
                "data for rule-engine-only flows."
            )
        readings = self.data_source.fetch_recent_readings(window_minutes=window_minutes)
        payload: dict[str, Any] = dict(patient_context or {})
        payload["cgm_readings"] = readings
        return self.detect_endocrine_anomaly(payload)

    def detect_endocrine_anomaly(
        self, patient_data: Mapping[str, Any]
    ) -> EndocrinologyPredictionResult:
        """Run the rule + ML pipeline on pre-loaded patient data.

        Args:
            patient_data: Mapping with any of:

                * ``cgm_readings`` - sequence of
                  :class:`~omni_mercury_engine.medical.data_sources.CGMReading`
                  (preferred path).
                * ``cgm_sequence`` - raw mg/dL sequence (used only when
                  ``cgm_readings`` is absent).
                * ``insulin_delivery`` / ``glp1_therapy`` / ``inhaled_insulin``.

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

        if self.enable_cgm and self.cgm_analyzer is not None:
            cgm_values, source_label = self._extract_cgm_sequence(patient_data)
            if cgm_values is not None and cgm_values.size > 0:
                cgm_result = self._analyze_cgm(cgm_values)
                result.glycemic_state = str(cgm_result["glycemic_state"])
                result.hypoglycemia_risk = float(cgm_result["hypoglycemia_risk"])
                result.hyperglycemia_risk = float(cgm_result["hyperglycemia_risk"])
                result.confidence = float(cgm_result["confidence"])
                result.glucose_anomalies = list(cgm_result["anomalies"])
                result.time_in_range_percent = float(cgm_result["time_in_range"])
                result.glucose_variability = float(cgm_result["variability"])
                result.cgm_source = source_label
                result.cgm_reading_count = int(cgm_values.size)
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

    @staticmethod
    def _extract_cgm_sequence(
        patient_data: Mapping[str, Any],
    ) -> tuple[npt.NDArray[np.float64] | None, str | None]:
        """Resolve the CGM sequence from ``cgm_readings`` or ``cgm_sequence``."""
        readings = patient_data.get("cgm_readings")
        if readings is not None:
            values: list[float] = []
            sources: set[str] = set()
            for r in readings:
                if not isinstance(r, CGMReading):
                    raise TypeError(
                        "cgm_readings entries must be CGMReading instances; "
                        f"got {type(r).__name__}"
                    )
                values.append(float(r.value_mg_dl))
                sources.add(r.source)
            if not values:
                return None, None
            source_label = next(iter(sources)) if len(sources) == 1 else "mixed"
            return np.asarray(values, dtype=np.float64), source_label

        raw = patient_data.get("cgm_sequence")
        if raw is None:
            return None, None
        return np.asarray(raw, dtype=np.float64), "preloaded"

    def _analyze_cgm(self, cgm_sequence: npt.NDArray[np.float64]) -> dict[str, Any]:
        """Run the CGM Bi-LSTM in inference mode."""
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
    data_source: CGMDataSource | None = None,
    *,
    enable_cgm: bool = True,
    enable_smart_pen: bool = True,
    enable_glp1: bool = True,
    enable_inhaled_insulin: bool = True,
) -> EndocrinologyDetector:
    """Factory returning a configured :class:`EndocrinologyDetector`.

    See :class:`EndocrinologyDetector` for the full argument contract and
    the :class:`ConfigurationError` semantics.
    """
    return EndocrinologyDetector(
        data_source,
        enable_cgm=enable_cgm,
        enable_smart_pen=enable_smart_pen,
        enable_glp1=enable_glp1,
        enable_inhaled_insulin=enable_inhaled_insulin,
    )


def count_cgm_parameters(model: CGMAnalyzer | None = None) -> int:
    """Return the trainable parameter count of the CGM Bi-LSTM."""
    instance = model if model is not None else CGMAnalyzer()
    return int(sum(p.numel() for p in instance.parameters() if p.requires_grad))


__all__ = [
    "CGMAnalyzer",
    "EndocrinologyDetector",
    "EndocrinologyPredictionResult",
    "GLP1TherapyMonitor",
    "GlycemicState",
    "InhaledInsulinMonitor",
    "InsulinDeliveryMethod",
    "SmartInsulinPenMonitor",
    "count_cgm_parameters",
    "get_endocrinology_detector",
]

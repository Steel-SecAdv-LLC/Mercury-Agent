# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Sepsis Detector - Early Sepsis Detection & Septic Shock Prevention.

Advanced sepsis detection for humanitarian healthcare:
- SOFA score calculation (Sequential Organ Failure Assessment)
- qSOFA (quick SOFA) for rapid screening
- Sepsis-3 criteria implementation
- Septic shock prediction
- Multi-organ dysfunction monitoring
- Temporal progression tracking

Safety semantics (fail-closed): the SOFA and qSOFA instruments are deterministic
and literature-validated. A missing input is reported as **unassessed** and never
scored as a healthy organ/criterion — so absent data can never lower the score and
mask organ failure; a partial total is an explicit lower bound. The neural
progression model is gated behind ``is_fitted`` and refuses to emit a number until
trained weights are loaded (it never surfaces an untrained network's output as
clinical confidence). Every result carries a decision-support disclaimer,
provenance, and red-flag emergency routing.

Research sources:
- Sepsis-3 definitions (JAMA 2016)
- Surviving Sepsis Campaign guidelines
- SOFA (Vincent et al., Intensive Care Med 1996) / qSOFA validation studies
- MIMIC-III sepsis cohort research
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from omni_mercury_engine.medical.safety import (
    ClinicalSafetyEnvelope,
    build_provenance,
)

if TYPE_CHECKING:
    import numpy as np


class SepsisStage(Enum):
    """Sepsis progression stages."""

    NO_SEPSIS = "no_sepsis"
    SIRS = "systemic_inflammatory_response"
    SEPSIS = "sepsis"
    SEVERE_SEPSIS = "severe_sepsis"
    SEPTIC_SHOCK = "septic_shock"


@dataclass
class SepsisPredictionResult:
    """Sepsis detection results."""

    sepsis_detected: bool
    confidence: float
    sepsis_stage: str
    risk_score: float

    sofa_score: int | None = None
    qsofa_score: int | None = None

    septic_shock_risk: float = 0.0
    mortality_risk: float = 0.0

    organ_dysfunctions: list[str] = field(default_factory=list)
    time_to_intervention_hours: float | None = None

    clinical_recommendations: list[str] = field(default_factory=list)
    bundle_compliance: list[str] = field(default_factory=list)

    # Safety / honesty layer.
    ml_available: bool = False
    sofa_is_lower_bound: bool = False
    unassessed_organs: list[str] = field(default_factory=list)
    safety: ClinicalSafetyEnvelope = field(default_factory=ClinicalSafetyEnvelope)


class SOFACalculator:
    """Sequential Organ Failure Assessment (SOFA) score calculator.

    Quantifies organ dysfunction across 6 systems (Vincent et al., Intensive
    Care Med 1996). Fail-closed: an organ whose required input is absent is
    reported **unassessed** (``None``), never scored as a healthy 0 — so a
    missing lab can never lower the score and hide organ failure. When any organ
    is unassessed the reported total is an explicit **lower bound**.
    """

    #: Instrument version, for provenance. Bump when scoring logic changes.
    VERSION = "sepsis-3"

    def __init__(self) -> None:
        """Initialize the instance."""
        self.logger = logging.getLogger(__name__)

    def calculate_sofa(self, patient_data: dict[str, Any]) -> dict[str, Any]:
        """Calculate SOFA score from patient data.

        Args:
            patient_data: Clinical parameters for SOFA calculation

        Returns:
            SOFA score and organ-specific scores. Organs with absent inputs are
            reported as ``None`` and listed in ``unassessed_organs``; the total
            (``sofa_score``) is a lower bound whenever any organ is unassessed.
        """
        organ_scores: dict[str, int | None] = {
            "respiration": self._calculate_respiration(patient_data),
            "coagulation": self._calculate_coagulation(patient_data),
            "liver": self._calculate_liver(patient_data),
            "cardiovascular": self._calculate_cardiovascular(patient_data),
            "cns": self._calculate_cns(patient_data),
            "renal": self._calculate_renal(patient_data),
        }
        label = {
            "respiration": "respiratory",
            "coagulation": "coagulation",
            "liver": "hepatic",
            "cardiovascular": "cardiovascular",
            "cns": "neurological",
            "renal": "renal",
        }

        assessed = {k: v for k, v in organ_scores.items() if v is not None}
        unassessed = [k for k, v in organ_scores.items() if v is None]
        total_sofa = sum(assessed.values())
        organ_dysfunctions = [label[k] for k, v in assessed.items() if v >= 2]

        result: dict[str, Any] = {
            "sofa_score": total_sofa,
            "sofa_is_lower_bound": bool(unassessed),
            "assessed_organs": sorted(assessed),
            "unassessed_organs": unassessed,
            "organ_dysfunctions": organ_dysfunctions,
            "sepsis_indicated": total_sofa >= 2,
        }
        result.update(organ_scores)
        return result

    def _calculate_respiration(self, data: dict[str, Any]) -> int | None:
        """Respiration score based on PaO2/FiO2 ratio (``None`` if absent).

        SOFA scores 3 and 4 require respiratory support (Vincent 1996); without
        it, PaO2/FiO2 < 200 caps at 2.
        """
        pao2_fio2 = data.get("pao2_fio2_ratio")
        if pao2_fio2 is None:
            return None
        vent = bool(data.get("mechanical_ventilation", False))

        if pao2_fio2 >= 400:
            return 0
        elif pao2_fio2 >= 300:
            return 1
        elif pao2_fio2 >= 200:
            return 2
        elif pao2_fio2 >= 100:
            return 3 if vent else 2
        else:
            return 4 if vent else 2

    def _calculate_coagulation(self, data: dict[str, Any]) -> int | None:
        """Coagulation score based on platelets (``None`` if absent)."""
        platelets = data.get("platelets_k_ul")
        if platelets is None:
            return None

        if platelets >= 150:
            return 0
        elif platelets >= 100:
            return 1
        elif platelets >= 50:
            return 2
        elif platelets >= 20:
            return 3
        else:
            return 4

    def _calculate_liver(self, data: dict[str, Any]) -> int | None:
        """Liver score based on bilirubin (``None`` if absent)."""
        bilirubin = data.get("bilirubin_mg_dl")
        if bilirubin is None:
            return None

        if bilirubin < 1.2:
            return 0
        elif bilirubin < 2.0:
            return 1
        elif bilirubin < 6.0:
            return 2
        elif bilirubin < 12.0:
            return 3
        else:
            return 4

    def _calculate_cardiovascular(self, data: dict[str, Any]) -> int | None:
        """Cardiovascular score based on MAP and vasopressors (``None`` if absent).

        SOFA cardiovascular scoring:
        0: MAP >= 70, no vasopressors
        1: MAP < 70
        2: Dopamine <= 5 OR dobutamine (any dose) OR norepinephrine <= 0.1
        3: Dopamine > 5 AND <= 15 OR norepinephrine > 0.1 AND <= 0.5
        4: Dopamine > 15 OR norepinephrine > 0.5

        Assessable when MAP or any vasopressor value is present; absent
        vasopressor fields mean "not administered" (0), but an entirely empty
        cardiovascular input abstains rather than assuming a healthy MAP.
        """
        map_val = data.get("mean_arterial_pressure")
        dopamine = data.get("dopamine_mcg_kg_min")
        norepinephrine = data.get("norepinephrine_mcg_kg_min")

        if map_val is None and dopamine is None and norepinephrine is None:
            return None

        dopamine = dopamine or 0.0
        norepinephrine = norepinephrine or 0.0

        # Score 4: High-dose vasopressors
        if dopamine > 15 or norepinephrine > 0.5:
            return 4
        # Score 3: Moderate-dose vasopressors
        if dopamine > 5 or (norepinephrine > 0.1):
            return 3
        # Score 2: Low-dose vasopressors
        if dopamine > 0 or norepinephrine > 0:
            return 2
        # Score 1: Hypotension without vasopressors
        if map_val is not None and map_val < 70:
            return 1
        # Score 0: Normal
        return 0

    def _calculate_cns(self, data: dict[str, Any]) -> int | None:
        """CNS score based on Glasgow Coma Scale (``None`` if absent)."""
        gcs = data.get("gcs_score")
        if gcs is None:
            return None

        if gcs == 15:
            return 0
        elif gcs >= 13:
            return 1
        elif gcs >= 10:
            return 2
        elif gcs >= 6:
            return 3
        else:
            return 4

    def _calculate_renal(self, data: dict[str, Any]) -> int | None:
        """Renal score based on creatinine and urine output (``None`` if absent).

        Creatinine is the primary input. With only urine output present, only
        the severe-oliguria bands (scores 3/4) are determinable; a normal urine
        volume alone cannot establish a creatinine-based score, so it abstains.
        """
        creatinine = data.get("creatinine_mg_dl")
        urine_output = data.get("urine_output_ml_day")

        if creatinine is None and urine_output is None:
            return None

        if creatinine is None:
            # urine_output is not None here (the both-absent case returned above).
            if urine_output is not None and urine_output < 200:
                return 4
            if urine_output is not None and urine_output < 500:
                return 3
            return None

        if creatinine >= 5.0 or (urine_output is not None and urine_output < 200):
            return 4
        if creatinine >= 3.5 or (urine_output is not None and urine_output < 500):
            return 3
        if creatinine >= 2.0:
            return 2
        if creatinine >= 1.2:
            return 1
        return 0


class QuickSOFACalculator:
    """Quick SOFA (qSOFA) calculator for rapid sepsis screening.

    3-point bedside tool for early identification.
    """

    def __init__(self) -> None:
        """Initialize the instance."""
        self.logger = logging.getLogger(__name__)

    #: Instrument version, for provenance.
    VERSION = "sepsis-3"

    def calculate_qsofa(self, vital_signs: dict[str, Any]) -> dict[str, Any]:
        """Calculate qSOFA score.

        Args:
            vital_signs: Respiratory rate, mental status, blood pressure

        Returns:
            qSOFA score and interpretation. A criterion whose input is absent is
            reported in ``unassessed_components`` and does not silently count as
            normal; the score is a lower bound when any component is unassessed.
        """
        score = 0
        criteria_met: list[str] = []
        unassessed: list[str] = []

        resp_rate = vital_signs.get("respiratory_rate_bpm")
        if resp_rate is None:
            unassessed.append("respiratory_rate_bpm")
        elif resp_rate >= 22:
            score += 1
            criteria_met.append("tachypnea (RR ≥22)")

        gcs = vital_signs.get("gcs_score")
        if gcs is None:
            unassessed.append("gcs_score")
        elif gcs < 15:
            score += 1
            criteria_met.append("altered_mentation (GCS <15)")

        systolic_bp = vital_signs.get("systolic_bp_mmhg")
        if systolic_bp is None:
            unassessed.append("systolic_bp_mmhg")
        elif systolic_bp <= 100:
            score += 1
            criteria_met.append("hypotension (SBP ≤100)")

        positive = score >= 2

        return {
            "qsofa_score": score,
            "qsofa_is_lower_bound": bool(unassessed),
            "qsofa_positive": positive,
            "criteria_met": criteria_met,
            "unassessed_components": unassessed,
            "assessed_component_count": 3 - len(unassessed),
            "sepsis_screening_positive": positive,
            "recommendation": (
                "High risk for sepsis - urgent evaluation needed"
                if positive
                else "Continue monitoring"
            ),
        }


class SepsisProgressionPredictor(nn.Module):
    """Neural network for sepsis progression prediction.

    Predicts evolution from SIRS to septic shock using temporal patterns.
    """

    def __init__(self, input_dim: int = 32, hidden_dim: int = 64) -> None:
        """Initialize the instance."""
        super().__init__()

        # Fail-closed: a freshly-constructed network has RANDOM weights, so its
        # output is noise, not a prediction. ``is_fitted`` stays False until real
        # trained weights are loaded via ``load_trained_weights``; callers must
        # refuse to surface this model's output while it is False.
        self.is_fitted: bool = False

        self.temporal_encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=3,
            batch_first=True,
            dropout=0.3,
            bidirectional=True,
        )

        self.stage_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, len(SepsisStage)),
        )

        self.shock_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.mortality_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, temporal_sequence: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for sepsis progression prediction.

        Args:
            temporal_sequence: Temporal clinical features

        Returns:
            Tuple of (stage_classification, shock_risk, mortality_risk)
        """
        lstm_out, _ = self.temporal_encoder(temporal_sequence)

        final_state = lstm_out[:, -1, :]

        stage_logits = self.stage_classifier(final_state)
        shock_risk = self.shock_predictor(final_state)
        mortality_risk = self.mortality_predictor(final_state)

        return stage_logits, shock_risk, mortality_risk

    def load_trained_weights(self, state_dict: dict[str, Any]) -> None:
        """Load trained weights and mark the model fitted (safe to surface)."""
        self.load_state_dict(state_dict)
        self.is_fitted = True


class SepsisDetector:
    """Comprehensive sepsis detection system integrating SOFA, qSOFA, and temporal progression.

    prediction.
    """

    def __init__(self, enable_ml_prediction: bool = True) -> None:
        """Initialize the instance."""
        self.enable_ml_prediction = enable_ml_prediction

        self.sofa_calculator = SOFACalculator()
        self.qsofa_calculator = QuickSOFACalculator()
        self.progression_predictor = SepsisProgressionPredictor() if enable_ml_prediction else None

        self.logger = logging.getLogger(__name__)

    def detect_sepsis(self, patient_data: dict[str, Any]) -> SepsisPredictionResult:
        """Comprehensive sepsis detection and risk assessment.

        Args:
            patient_data: Patient data including:
                - vital_signs: Current vitals for qSOFA
                - laboratory_values: Labs for SOFA
                - temporal_sequence: Time-series clinical data

        Returns:
            Sepsis prediction with stage and recommendations
        """
        result = SepsisPredictionResult(
            sepsis_detected=False,
            confidence=0.0,
            sepsis_stage="no_sepsis",
            risk_score=0.0,
        )
        safety = result.safety

        if "vital_signs" in patient_data:
            qsofa = self.qsofa_calculator.calculate_qsofa(patient_data["vital_signs"])
            result.qsofa_score = qsofa["qsofa_score"]
            safety.note_unassessed([f"vital_signs.{c}" for c in qsofa["unassessed_components"]])
            safety.provenance["qsofa"] = build_provenance(
                instrument="qSOFA",
                version=self.qsofa_calculator.VERSION,
                inputs=patient_data["vital_signs"],
            )

            if qsofa["qsofa_positive"]:
                result.sepsis_detected = True
                result.confidence = 0.7

        if "laboratory_values" in patient_data:
            sofa = self.sofa_calculator.calculate_sofa(patient_data["laboratory_values"])
            result.sofa_score = sofa["sofa_score"]
            result.sofa_is_lower_bound = sofa["sofa_is_lower_bound"]
            result.unassessed_organs = sofa["unassessed_organs"]
            result.organ_dysfunctions = sofa["organ_dysfunctions"]
            safety.note_unassessed([f"sofa.{o}" for o in sofa["unassessed_organs"]])
            safety.provenance["sofa"] = build_provenance(
                instrument="SOFA",
                version=self.sofa_calculator.VERSION,
                inputs=patient_data["laboratory_values"],
            )

            if sofa["sepsis_indicated"]:
                result.sepsis_detected = True
                result.confidence = max(result.confidence, 0.8)

                if result.sofa_score >= 2:
                    result.sepsis_stage = "sepsis"
                if len(result.organ_dysfunctions) >= 2:
                    result.sepsis_stage = "severe_sepsis"

        # ML progression: only surfaced when the model carries trained weights.
        # An untrained network is refused, not dressed up as confidence.
        if self.enable_ml_prediction and "temporal_sequence" in patient_data:
            ml_result = self._predict_progression(patient_data["temporal_sequence"])
            result.ml_available = ml_result["available"]
            safety.provenance["progression_model"] = build_provenance(
                instrument="SepsisProgressionPredictor",
                version="untrained" if not ml_result["available"] else "v1",
                inputs={"temporal_sequence": "<array>"},
                model="SepsisProgressionPredictor",
                model_fitted=ml_result["available"],
            )
            if ml_result["available"]:
                result.sepsis_stage = ml_result["predicted_stage"]
                result.septic_shock_risk = ml_result["shock_risk"]
                result.mortality_risk = ml_result["mortality_risk"]
                result.confidence = max(result.confidence, ml_result["confidence"])

                if ml_result["shock_risk"] > 0.7:
                    result.sepsis_stage = "septic_shock"
                    result.sepsis_detected = True

        result.risk_score = max(
            result.septic_shock_risk,
            result.mortality_risk,
            float(result.sofa_score or 0) / 24.0,
        )

        # Red-flag emergency routing: sepsis and septic shock are time-critical
        # emergencies. Tell a lay reader to seek care now, not only clinicians.
        if result.sepsis_detected and result.sepsis_stage != "no_sepsis":
            safety.flag_emergency(f"possible sepsis ({result.sepsis_stage})")
        elif result.qsofa_score and result.qsofa_score >= 2:
            safety.flag_emergency("qSOFA positive (≥2) — high sepsis risk")

        result.clinical_recommendations = self._generate_recommendations(result)
        result.bundle_compliance = self._generate_bundle_checklist(result)
        result.time_to_intervention_hours = self._estimate_intervention_window(result)

        self.logger.info(
            "Sepsis detection: %s, SOFA=%s%s, ml_available=%s, unassessed=%d",
            result.sepsis_stage,
            result.sofa_score,
            " (lower bound)" if result.sofa_is_lower_bound else "",
            result.ml_available,
            len(safety.unassessed_inputs),
        )

        return result

    def _predict_progression(self, temporal_sequence: np.ndarray[Any, Any]) -> dict[str, Any]:
        """Predict sepsis progression using the ML model, or abstain.

        Fail-closed: if the model is absent or carries no trained weights
        (``is_fitted`` is False), this returns ``available=False`` and no
        numbers — an untrained network's output is noise and must never be
        surfaced as clinical risk/confidence.
        """
        predictor = self.progression_predictor
        if predictor is None or not getattr(predictor, "is_fitted", False):
            return {
                "available": False,
                "predicted_stage": None,
                "shock_risk": None,
                "mortality_risk": None,
                "confidence": None,
            }

        seq_tensor = torch.tensor(temporal_sequence, dtype=torch.float32).unsqueeze(0)

        predictor.eval()
        with torch.no_grad():
            stage_logits, shock_risk, mortality_risk = predictor(seq_tensor)

        stage_probs = torch.softmax(stage_logits[0], dim=0)
        stage_idx = torch.argmax(stage_probs).item()
        confidence = float(stage_probs[stage_idx].item())  # type: ignore[index, unused-ignore]

        stages = [e.value for e in SepsisStage]
        predicted_stage = stages[stage_idx]  # type: ignore[index, unused-ignore]

        return {
            "available": True,
            "predicted_stage": predicted_stage,
            "shock_risk": float(shock_risk[0].item()),
            "mortality_risk": float(mortality_risk[0].item()),
            "confidence": confidence,
        }

    def _generate_recommendations(self, result: SepsisPredictionResult) -> list[str]:
        """Generate clinical recommendations based on sepsis assessment."""
        recs = []

        if result.sepsis_stage == "septic_shock":
            recs.append("SEPTIC SHOCK - IMMEDIATE RESUSCITATION REQUIRED")
            recs.append("Administer 30 mL/kg crystalloid within 3 hours")
            recs.append("Start broad-spectrum antibiotics within 1 hour")
            recs.append("Vasopressors for MAP ≥65 mmHg")
            recs.append("Measure lactate - target <2 mmol/L")
        elif result.sepsis_stage in ["sepsis", "severe_sepsis"]:
            recs.append("Sepsis detected - initiate Sepsis-3 bundle")
            recs.append("Blood cultures before antibiotics")
            recs.append("Empiric antibiotics within 1 hour")
            recs.append("Initial fluid resuscitation 30 mL/kg")
            recs.append("Serial lactate measurements")
        elif result.qsofa_score and result.qsofa_score >= 2:
            recs.append("qSOFA positive - urgent sepsis evaluation")
            recs.append("Obtain blood cultures and lactate")
            recs.append("Consider empiric antibiotics if source identified")

        if "respiratory" in result.organ_dysfunctions:
            recs.append("Respiratory failure - consider intubation and lung-protective ventilation")
        if "renal" in result.organ_dysfunctions:
            recs.append("Acute kidney injury - nephrology consult, consider RRT")
        if "cardiovascular" in result.organ_dysfunctions:
            recs.append("Hemodynamic instability - ICU transfer, arterial line")

        return recs

    def _generate_bundle_checklist(self, result: SepsisPredictionResult) -> list[str]:
        """Generate Surviving Sepsis Campaign bundle checklist."""
        bundle = []

        if result.sepsis_detected:
            bundle.append("[ ] Measure lactate level")
            bundle.append("[ ] Obtain blood cultures before antibiotics")
            bundle.append("[ ] Administer broad-spectrum antibiotics")
            bundle.append("[ ] Begin rapid fluid resuscitation (30 mL/kg)")
            bundle.append("[ ] Apply vasopressors if hypotensive during/after fluid resuscitation")

        if result.sepsis_stage in ["severe_sepsis", "septic_shock"]:
            bundle.append("[ ] Re-measure lactate if initial >2 mmol/L")
            bundle.append("[ ] Maintain MAP ≥65 mmHg")
            bundle.append("[ ] Document normal volume status and tissue perfusion")

        return bundle

    def _estimate_intervention_window(self, result: SepsisPredictionResult) -> float | None:
        """Estimate time window for critical interventions."""
        if result.sepsis_stage == "septic_shock":
            return 1.0
        elif result.sepsis_stage in ["sepsis", "severe_sepsis"]:
            return 3.0
        elif result.qsofa_score and result.qsofa_score >= 2:
            return 6.0
        return None

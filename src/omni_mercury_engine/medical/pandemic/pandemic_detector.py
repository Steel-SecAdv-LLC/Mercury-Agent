# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pandemic & Epidemiology Detector - Outbreak Monitoring & Mutation Tracking.

Comprehensive pandemic detection for public health early warning:
- Outbreak detection (case surge identification)
- Pathogen mutation tracking (antigenic drift/shift)
- Effective reproduction number (R_e) estimated from observed growth
  (Wallinga-Lipsitch, delta serial interval -- an upper bound, and not R0)
- Epidemic curve modeling
- Variant classification (WHO nomenclature)
- Transmission pattern analysis
- Genomic surveillance integration
- Multi-pathogen monitoring (influenza, coronavirus, etc.)

Integrations:
- Genomic sequencing data
- Case surveillance systems
- Novel class discovery (novel_class_discovery.py)
- Medical cure predictor integration
- Geospatial spread modeling
- Contact tracing network analysis

Research sources:
- WHO Global Influenza Surveillance
- GISAID (viral genome database)
- CDC Epidemic Intelligence Service
- ECDC Surveillance Systems
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch
from torch import nn

from omni_mercury_engine.core.exceptions import InvalidObservationError
from omni_mercury_engine.detectors._hazard_input import InputGrader
from omni_mercury_engine.detectors.provenance import DecisionSource

#: Days per comparison window; the surge rule compares two adjacent windows.
SURGE_WINDOW_DAYS = 7
#: Week-on-week ratio of mean daily cases above which a surge is declared.
SURGE_RATIO = 1.5
#: Keys ``MutationTracker.track_mutations`` reads.
GENOMIC_INPUTS = frozenset(
    {
        "mutation_count",
        "spike_mutations",
        "antigenic_distance",
        "immune_escape_mutations",
        "resistance_mutations",
    }
)
#: Top-level sections ``PandemicDetector.predict_pandemic`` grades.
PANDEMIC_SECTIONS = frozenset(
    {"case_data", "genomic_data", "network_data", "geographic_spread", "effective_r_data"}
)


class OutbreakSeverity(Enum):
    """Outbreak severity levels."""

    SPORADIC = "sporadic"
    CLUSTER = "cluster"
    OUTBREAK = "outbreak"
    EPIDEMIC = "epidemic"
    PANDEMIC = "pandemic"


class VariantConcern(Enum):
    """WHO variant classifications."""

    MONITORING = "variant_under_monitoring"
    INTEREST = "variant_of_interest"
    CONCERN = "variant_of_concern"
    HIGH_CONSEQUENCE = "variant_of_high_consequence"


@dataclass
class PandemicPredictionResult:
    """Pandemic prediction results."""

    outbreak_detected: bool
    confidence: float
    severity_level: str

    case_surge_detected: bool = False
    doubling_time_days: float | None = None
    r0_estimate: float | None = None
    re_estimate: float | None = None

    mutation_detected: bool = False
    variant_type: str = "wild_type"
    concern_level: str = "monitoring"
    antigenic_distance: float | None = None

    genomic_surveillance_alerts: list[str] = field(default_factory=list)
    transmission_hotspots: list[str] = field(default_factory=list)

    vaccine_escape_probability: float = 0.0
    treatment_resistance_probability: float = 0.0

    public_health_actions: list[str] = field(default_factory=list)
    containment_measures: list[str] = field(default_factory=list)

    # Partial-input contract: what could not be graded, and why.
    partial: bool = False
    set_aside: list[dict[str, Any]] = field(default_factory=list)


class CaseSurgeDetector:
    """Epidemiological case surge detection.

    Compares the mean of the latest seven daily counts with the mean of the
    seven before them; the growth rate ``r = ln(recent / baseline) / 7``
    gives the doubling time ``ln 2 / r`` (growing series only) and the
    growth-implied effective reproduction number (Wallinga-Lipsitch).
    """

    decision_source = DecisionSource(
        kind="experimental",
        description=(
            "Surge when the mean of the latest 7 daily counts exceeds 1.5x the "
            "mean of the 7 before; the 1.5 ratio has no cited source and no "
            "selected operating point. R_e = exp(r * T_s) for growth rate r "
            "(Wallinga and Lipsitch 2007, Proc R Soc B 274:599-604) with a "
            "delta-distributed serial interval T_s (default 5 days, set by "
            "the caller): an upper bound on R_e, and not R0."
        ),
    )

    def __init__(self) -> None:
        """Initialize the instance."""
        self.logger = logging.getLogger(__name__)

    def detect_case_surge(self, case_data: dict[str, Any]) -> dict[str, Any]:
        """Detect case surge from time series.

        Args:
            case_data: ``daily_cases`` (counts per day, oldest first) and
                optionally ``serial_interval_days``.  A NaN day is a gap and
                a negative count is a reporting correction, not a count:
                both are set aside and the longest gap-free run is graded.

        Returns:
            Case surge detection results.  ``re_estimate`` and
            ``doubling_time_days`` are ``None`` when the counts cannot
            establish them; ``partial`` / ``set_aside`` are present when
            anything was set aside (including too short a history).

        Raises:
            InvalidObservationError: When ``daily_cases`` is not numeric, not
                one-dimensional, or holds no usable count at all.
        """
        grader = InputGrader()
        counts = self._graded_counts(case_data.get("daily_cases", []), grader)
        serial_interval = grader.usable_scalar(
            case_data.get("serial_interval_days", 5.0), "serial_interval_days", unit="days", lo=0.0
        )
        if serial_interval == 0.0:
            grader.discard("serial_interval_days", "must be > 0 days", serial_interval)
            serial_interval = None

        needed = 2 * SURGE_WINDOW_DAYS
        if counts.size < needed:
            # One window compared with itself can never surge: report the
            # history as insufficient, not as a negative verdict.
            grader.discard(
                "daily_cases",
                f"{counts.size} usable daily count(s); the surge rule compares two "
                f"adjacent {SURGE_WINDOW_DAYS}-day windows and needs {needed}: not assessed",
            )
            return {
                "surge_detected": False,
                "doubling_time_days": None,
                "re_estimate": None,
                "re_method": "insufficient_history",
                "r0_estimate": None,
                "week_on_week_ratio": None,
                **grader.result_fields(),
            }

        recent_mean = float(np.mean(counts[-SURGE_WINDOW_DAYS:]))
        baseline_mean = float(np.mean(counts[-needed:-SURGE_WINDOW_DAYS]))
        surge_detected = recent_mean > baseline_mean * SURGE_RATIO

        doubling_time_days: float | None = None
        re_estimate: float | None = None
        if baseline_mean > 0 and recent_mean > 0:
            ratio: float | None = recent_mean / baseline_mean
            growth_rate = float(np.log(recent_mean / baseline_mean)) / SURGE_WINDOW_DAYS
            if growth_rate > 0:
                doubling_time_days = float(np.log(2) / growth_rate)
            # Wallinga-Lipsitch under a *delta-distributed* serial interval:
            #   R = exp(r * T_s),  r the exponential growth rate (any sign).
            # This estimates the EFFECTIVE reproduction number R_e from observed
            # growth. It is NOT R0: the basic reproduction number is defined in a
            # fully susceptible population, and nothing here observes susceptibility,
            # immunity or intervention state. Reporting an R_e as R0 overstates it
            # whenever the population is partly immune -- which, during a surge, it
            # always is.
            #
            # The delta-serial-interval assumption also makes this an upper bound:
            # a dispersed serial interval yields a smaller R for the same growth
            # rate. Treat the value as "growth-implied R_e, delta-SI upper bound",
            # not a calibrated epidemiological estimate.
            if serial_interval is not None:
                re_estimate = float(np.exp(growth_rate * serial_interval))
            re_method = (
                "wallinga_lipsitch_delta_serial_interval"
                if serial_interval is not None
                else "no_serial_interval"
            )
        else:
            # A zero window has no growth rate: emergence from zero is a surge,
            # but no R_e or doubling time follows from it.
            ratio = float("inf") if recent_mean > 0 else None
            re_method = "zero_counts"

        return {
            "surge_detected": bool(surge_detected),
            "doubling_time_days": doubling_time_days,
            "re_estimate": re_estimate,
            "re_method": re_method,
            # Back-compatible alias for existing consumers. Same number, honest
            # name above; the alias is retained so a caller is not broken, not
            # because the quantity is an R0.
            "r0_estimate": re_estimate,
            "week_on_week_ratio": ratio,
            **grader.result_fields(),
        }

    @staticmethod
    def _graded_counts(raw: Any, grader: InputGrader) -> np.ndarray[Any, Any]:
        """The longest gap-free run of non-negative daily counts."""
        try:
            series = np.array(raw, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise InvalidObservationError(
                f"daily_cases must be a numeric series of counts per day, got "
                f"{type(raw).__name__}"
            ) from exc
        if series.size == 0:
            return series.reshape(0)
        negative = series < 0
        if negative.any():
            grader.discard(
                "daily_cases",
                f"{int(negative.sum())} negative count(s) are reporting corrections, "
                "not case counts; treated as gaps",
            )
            series = np.where(negative, np.nan, series)
        return grader.usable_run(series, "daily_cases", unit="cases/day")


class MutationTracker:
    """Viral mutation tracking via genomic surveillance.

    Identifies antigenic drift, shift, and emergence of variants.
    """

    decision_source = DecisionSource(
        kind="experimental",
        description=(
            "Mutation flag (> 10 mutations or > 3 spike mutations), the "
            "antigenic-distance tiers (> 1, > 2, > 4) mapped onto WHO variant "
            "labels, and the escape / resistance probabilities (list length "
            "over 10 and over 5) have no cited source; WHO does not designate "
            "variants by an antigenic-distance threshold."
        ),
    )

    def __init__(self) -> None:
        """Initialize the instance."""
        self.logger = logging.getLogger(__name__)

    def track_mutations(self, genomic_data: dict[str, Any]) -> dict[str, Any]:
        """Track viral mutations from sequences.

        Args:
            genomic_data: Viral genome sequences, mutations

        Returns:
            Mutation tracking results.  ``antigenic_distance`` is ``None``
            when it was not measured (it is never assumed to be 0);
            ``partial`` / ``set_aside`` are present when a value was unusable.
        """
        if not isinstance(genomic_data, dict):
            raise InvalidObservationError(
                f"genomic_data must be a mapping, got {type(genomic_data).__name__}"
            )
        grader = InputGrader()
        for key in genomic_data:
            if key not in GENOMIC_INPUTS:
                grader.discard(key, f"unknown key; accepted: {sorted(GENOMIC_INPUTS)}")
        mutation_count = grader.usable_scalar(
            genomic_data.get("mutation_count"), "mutation_count", unit="mutations", lo=0.0
        )
        distance = grader.usable_scalar(
            genomic_data.get("antigenic_distance"),
            "antigenic_distance",
            unit="antigenic units",
            lo=0.0,
        )
        lists: dict[str, list[Any]] = {}
        for key in ("spike_mutations", "immune_escape_mutations", "resistance_mutations"):
            value = genomic_data.get(key, [])
            if isinstance(value, (list, tuple)):
                lists[key] = list(value)
            else:
                grader.discard(key, f"must be a list of mutation names, got {type(value).__name__}")
                lists[key] = []
        spike_mutations = lists["spike_mutations"]
        unusable = {entry["where"] for entry in grader.set_aside}
        grader.require_usable(
            [k for k in genomic_data if k in GENOMIC_INPUTS and k not in unusable],
            f"genomic_data needs at least one of {sorted(GENOMIC_INPUTS)}",
        )

        mutation_detected = (mutation_count or 0) > 10 or len(spike_mutations) > 3

        if distance is None:
            # Unmeasured: the released default labels, reported with
            # antigenic_distance None so they visibly rest on no distance.
            concern_level = VariantConcern.MONITORING.value
            variant_type = "wild_type"
        elif distance > 4.0:
            concern_level = VariantConcern.HIGH_CONSEQUENCE.value
            variant_type = "highly_divergent"
        elif distance > 2.0:
            concern_level = VariantConcern.CONCERN.value
            variant_type = "concerning"
        elif distance > 1.0:
            concern_level = VariantConcern.INTEREST.value
            variant_type = "interest"
        else:
            concern_level = VariantConcern.MONITORING.value
            variant_type = "wild_type"

        vaccine_escape_prob = min(len(lists["immune_escape_mutations"]) / 10.0, 1.0)
        treatment_resistance_prob = min(len(lists["resistance_mutations"]) / 5.0, 1.0)

        return {
            "mutation_detected": bool(mutation_detected),
            "variant_type": variant_type,
            "concern_level": concern_level,
            "antigenic_distance": distance,
            "vaccine_escape_prob": float(vaccine_escape_prob),
            "treatment_resistance_prob": float(treatment_resistance_prob),
            **grader.result_fields(),
        }


class TransmissionNetworkAnalyzer(nn.Module):
    """Neural network for transmission network analysis.

    Identifies super-spreader events and transmission hotspots.
    """

    def __init__(self, input_dim: int = 64) -> None:
        """Initialize the instance."""
        super().__init__()

        # Fail-closed: a freshly-constructed network has RANDOM weights, so its
        # output is noise, not a prediction. ``is_fitted`` stays False until real
        # trained weights are loaded via ``load_trained_weights``; callers must
        # refuse to surface this model's output while it is False.
        self.is_fitted: bool = False

        phi = 1.618

        self.network_encoder = nn.Sequential(
            nn.Linear(input_dim, int(128 * phi)),
            nn.BatchNorm1d(int(128 * phi)),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(int(128 * phi), int(64 * phi)),
            nn.BatchNorm1d(int(64 * phi)),
            nn.ReLU(),
            nn.Linear(int(64 * phi), 64),
        )

        self.hotspot_detector = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, network_features: torch.Tensor) -> torch.Tensor:
        """Detect transmission hotspots."""
        features = self.network_encoder(network_features)
        hotspot_prob = self.hotspot_detector(features)

        return hotspot_prob

    def load_trained_weights(self, state_dict: dict[str, Any]) -> None:
        """Load trained weights and mark the model fitted (safe to surface)."""
        self.load_state_dict(state_dict)
        self.is_fitted = True


class PandemicDetector:
    """Comprehensive pandemic and outbreak detection system.

    Integrates case surveillance, genomic data, and transmission networks.
    """

    decision_source = (
        CaseSurgeDetector.decision_source,
        MutationTracker.decision_source,
        DecisionSource(
            kind="experimental",
            description=(
                "Severity ladder (pandemic: >= 2 continents and >= 10 countries; "
                "epidemic: >= 3 countries; outbreak: surge with growth-implied "
                "R > 1.5; cluster: surge) and the fixed confidences 0.7 (surge) "
                "and 0.8 (mutation) have no cited source; they are not WHO "
                "phase definitions or calibrated probabilities."
            ),
        ),
        DecisionSource(
            kind="experimental",
            description=(
                "The TransmissionNetworkAnalyzer (hotspots) ships untrained and "
                "abstains until trained weights are loaded."
            ),
        ),
    )

    def __init__(
        self,
        enable_case_surveillance: bool = True,
        enable_mutation_tracking: bool = True,
        enable_network_analysis: bool = True,
    ):
        """Initialize the instance."""
        self.enable_surveillance = enable_case_surveillance
        self.enable_mutations = enable_mutation_tracking
        self.enable_network = enable_network_analysis

        self.surge_detector = CaseSurgeDetector() if enable_case_surveillance else None
        self.mutation_tracker = MutationTracker() if enable_mutation_tracking else None
        self.network_analyzer = TransmissionNetworkAnalyzer() if enable_network_analysis else None

        self.logger = logging.getLogger(__name__)

    def predict_pandemic(self, pandemic_data: dict[str, Any]) -> PandemicPredictionResult:
        """Comprehensive pandemic prediction.

        Args:
            pandemic_data: Multi-source epidemiological data including:
                - case_data: Time series of case counts
                - genomic_data: Viral sequences and mutations
                - network_data: Contact tracing network
                - geographic_spread: ``countries_affected``, ``continents_affected``
                - effective_r_data: an externally estimated ``re``

        Returns:
            Pandemic prediction with public health recommendations.  A
            section that could not be graded is listed in ``set_aside`` and
            the result is marked ``partial``.

        Raises:
            InvalidObservationError: When no section is usable (an
                unassessed population is not a quiet one).
        """
        if not isinstance(pandemic_data, dict):
            raise InvalidObservationError(
                f"predict_pandemic expects a mapping, got {type(pandemic_data).__name__}"
            )
        grader = InputGrader()
        assessed: list[str] = []
        result = PandemicPredictionResult(
            outbreak_detected=False, confidence=0.0, severity_level="sporadic"
        )
        for key in pandemic_data:
            if key not in PANDEMIC_SECTIONS:
                grader.discard(
                    key, f"not read by any rule; graded sections: {sorted(PANDEMIC_SECTIONS)}"
                )
        disabled = {
            "case_data": self.surge_detector is None,
            "genomic_data": self.mutation_tracker is None,
            "network_data": self.network_analyzer is None,
        }
        for key, off in disabled.items():
            if off and key in pandemic_data:
                grader.discard(key, "this stream's analysis is disabled on this detector")

        if self.enable_surveillance and "case_data" in pandemic_data:
            if self.surge_detector is not None:
                surge = self.surge_detector
                surge_result = grader.attempt(
                    "case_data", lambda data: self._surge(surge, data), pandemic_data["case_data"]
                )
                if surge_result is not None:
                    self._absorb(grader, "case_data", surge_result)
                    result.case_surge_detected = surge_result["surge_detected"]
                    result.doubling_time_days = surge_result["doubling_time_days"]
                    result.r0_estimate = surge_result["r0_estimate"]
                    if surge_result["re_method"] != "insufficient_history":
                        assessed.append("case_data")

                    if surge_result["surge_detected"]:
                        result.confidence = 0.7
                        result.outbreak_detected = True

        if self.enable_mutations and "genomic_data" in pandemic_data:
            if self.mutation_tracker is not None:
                mutation_result = grader.attempt(
                    "genomic_data",
                    self.mutation_tracker.track_mutations,
                    pandemic_data["genomic_data"],
                )
                if mutation_result is not None:
                    self._absorb(grader, "genomic_data", mutation_result)
                    assessed.append("genomic_data")
                    result.mutation_detected = mutation_result["mutation_detected"]
                    result.variant_type = mutation_result["variant_type"]
                    result.concern_level = mutation_result["concern_level"]
                    result.antigenic_distance = mutation_result["antigenic_distance"]
                    result.vaccine_escape_probability = mutation_result["vaccine_escape_prob"]
                    result.treatment_resistance_probability = mutation_result[
                        "treatment_resistance_prob"
                    ]

                    if mutation_result["mutation_detected"]:
                        result.confidence = max(result.confidence, 0.8)
                        result.genomic_surveillance_alerts.append(
                            f"Variant of {mutation_result['concern_level']} detected"
                        )

        if self.network_analyzer is not None and "network_data" in pandemic_data:
            if self.network_analyzer.is_fitted:
                hotspots = self._analyze_transmission_network(pandemic_data["network_data"])
                result.transmission_hotspots = hotspots
                assessed.append("network_data")
            else:
                # An untrained network's output is noise; an empty hotspot list
                # from it would read as "no hotspot found".
                grader.discard(
                    "network_data",
                    "TransmissionNetworkAnalyzer has no trained weights; hotspots not assessed",
                )

        spread = self._graded_spread(pandemic_data.get("geographic_spread"), grader)
        if spread:
            assessed.append("geographic_spread")
        result.severity_level = self._determine_severity(result, {"geographic_spread": spread})

        if "effective_r_data" in pandemic_data:
            effective = pandemic_data["effective_r_data"]
            re_value = (
                grader.usable_scalar(
                    effective.get("re"), "effective_r_data.re", unit="dimensionless", lo=0.0
                )
                if isinstance(effective, dict)
                else None
            )
            if not isinstance(effective, dict):
                grader.discard("effective_r_data", "must be a mapping with 're'")
            if re_value is not None:
                result.re_estimate = re_value
                assessed.append("effective_r_data")

        grader.require_usable(
            assessed,
            "supply case_data (>= 14 daily counts), genomic_data, geographic_spread or "
            "effective_r_data",
        )
        result.partial = grader.partial
        result.set_aside = list(grader.set_aside)

        result.public_health_actions = self._generate_public_health_actions(result)
        result.containment_measures = self._generate_containment_measures(result)

        return result

    @staticmethod
    def _surge(detector: CaseSurgeDetector, case_data: Any) -> dict[str, Any]:
        """``detector`` on one ``case_data`` mapping."""
        if not isinstance(case_data, dict):
            raise InvalidObservationError(
                f"case_data must be a mapping with 'daily_cases', got {type(case_data).__name__}"
            )
        return detector.detect_case_surge(case_data)

    @staticmethod
    def _absorb(grader: InputGrader, section: str, stream: dict[str, Any]) -> None:
        """Carry a stream's set-aside entries, prefixed with its section."""
        for entry in stream.get("set_aside", []):
            grader.set_aside.append({**entry, "where": f"{section}.{entry['where']}"})

    @staticmethod
    def _graded_spread(spread: Any, grader: InputGrader) -> dict[str, float]:
        """The usable ``geographic_spread`` counts."""
        if spread is None:
            return {}
        if not isinstance(spread, dict):
            grader.discard("geographic_spread", f"must be a mapping, got {type(spread).__name__}")
            return {}
        graded: dict[str, float] = {}
        for key, value in spread.items():
            where = f"geographic_spread.{key}"
            if key not in ("countries_affected", "continents_affected"):
                grader.discard(
                    where, "unknown key; accepted: continents_affected, countries_affected"
                )
                continue
            count = grader.usable_scalar(value, where, unit="count", lo=0.0)
            if count is not None:
                graded[key] = count
        return graded

    def _analyze_transmission_network(self, network_data: dict[str, Any]) -> list[str]:
        """Analyze transmission network for hotspots."""
        if "network_features" in network_data:
            features = network_data["network_features"]
        else:
            contact_density = network_data.get("contact_density", 0.1)
            features = np.array([contact_density])
            features = np.pad(features, (0, 63), mode="constant")

        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        if self.network_analyzer is None:
            return []

        self.network_analyzer.eval()
        with torch.no_grad():
            hotspot_prob = self.network_analyzer(features_tensor)

        if float(hotspot_prob[0].item()) > 0.7:
            hotspots: list[str] = network_data.get("location_names", ["unspecified_location"])
        else:
            hotspots = []

        return hotspots

    def _determine_severity(self, result: PandemicPredictionResult, data: dict[str, Any]) -> str:
        """Determine outbreak severity level."""
        geographic_spread = data.get("geographic_spread", {})
        countries_affected = geographic_spread.get("countries_affected", 0)
        continents_affected = geographic_spread.get("continents_affected", 0)

        if continents_affected >= 2 and countries_affected >= 10:
            return OutbreakSeverity.PANDEMIC.value
        elif countries_affected >= 3:
            return OutbreakSeverity.EPIDEMIC.value
        elif result.case_surge_detected and result.r0_estimate and result.r0_estimate > 1.5:
            return OutbreakSeverity.OUTBREAK.value
        elif result.case_surge_detected:
            return OutbreakSeverity.CLUSTER.value
        else:
            return OutbreakSeverity.SPORADIC.value

    def _generate_public_health_actions(self, result: PandemicPredictionResult) -> list[str]:
        """Generate public health actions."""
        actions = []

        if result.severity_level == "pandemic":
            actions.append("Activate pandemic response protocols")
            actions.append("International coordination via WHO")
        elif result.severity_level == "epidemic":
            actions.append("Enhance surveillance systems")
            actions.append("Mobilize emergency response teams")

        if result.mutation_detected:
            if result.concern_level in ["variant_of_concern", "variant_of_high_consequence"]:
                actions.append("Urgent genomic sequencing of all cases")
                actions.append("Assess vaccine effectiveness against variant")

        if result.vaccine_escape_probability > 0.7:
            actions.append("Consider booster vaccination campaigns")

        return actions

    def _generate_containment_measures(self, result: PandemicPredictionResult) -> list[str]:
        """Generate containment measures."""
        measures = []

        if result.severity_level in ["outbreak", "epidemic", "pandemic"]:
            measures.append("Implement contact tracing")
            measures.append("Quarantine confirmed cases")

            if result.r0_estimate and result.r0_estimate > 2.0:
                measures.append("Consider social distancing measures")

        if result.transmission_hotspots:
            measures.append(f"Targeted interventions in: {', '.join(result.transmission_hotspots)}")

        return measures

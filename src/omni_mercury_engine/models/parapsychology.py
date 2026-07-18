# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""REG Statistical-Deviation Detection (parapsychology-research instrumentation).

What this module MEASURES: deviation-from-chance in random event generator
(REG) streams -- hardware-RNG trial sequences whose null distribution is
known exactly (e.g. Global Consciousness Project egg trials: sums of 200
XOR-whitened bits, Binomial(200, 0.5) under the null). The tools here are
plain statistics (z-scores, chi-square variance tails, effect sizes) plus a
trainable sequence model, and they answer exactly one question: *does this
stream deviate from its chance expectation?* Full stop.

Hypothesis under study (interpretation layer, one level up): the conjecture
that mind or collective attention correlates with REG deviations is a
genuinely studied research question -- PEAR laboratory (Princeton,
1979-2007), the U.S. government's Stargate program, the Global
Consciousness Project (1998-present), and the Koestler Parapsychology Unit
(University of Edinburgh) all investigated it -- and it remains CONTESTED
in mainstream science. This module asserts neither that psi is real nor
that it is disproven; a detected deviation is a statistical event whose
cause (hardware fault, environmental interference, analysis error, or the
hypothesis under study) must be established separately. The project's
pre-registered-null precedent is ``docs/PARAPSYCH_PREREGISTRATION.md``:
analysis choices are fixed before data, and a faithful null is a valid,
expected outcome.

Fault-detection application (labels true by construction): the same
deviation machinery doubles as a hardware-RNG health monitor. The shipped
``reg_deviation_gcp`` checkpoint is trained on REAL measured GCP streams
(null class) versus the SAME streams passed through known, recorded
fault-injection channels (anomaly class) -- see
:mod:`omni_mercury_engine.ml.hazard_training.consciousness_field` and
:mod:`omni_mercury_engine.security.rng_health`.

Key Features:
- Statistical deviation analysis against exact chance baselines
- REG mean-bias (Stouffer Z) and variance (chi-square) monitoring
- ESP-trial and presentiment analysis kept for legacy research datasets
- Effect sizes, confidence intervals, replication-focused recommendations
- Anti-theater quarantine: the neural analyser abstains until trained
  weights with provenance are loaded
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from scipy import stats
from torch import nn


class PsiPhenomenon(str):
    """Types of psi phenomena."""

    TELEPATHY = "telepathy"
    CLAIRVOYANCE = "clairvoyance"
    PRECOGNITION = "precognition"
    PSYCHOKINESIS = "psychokinesis"
    REMOTE_VIEWING = "remote_viewing"
    PRESENTIMENT = "presentiment"
    FIELD_CONSCIOUSNESS = "field_consciousness"


@dataclass
class ParapsychologyResult:
    """Result from parapsychology anomaly detection."""

    anomaly_detected: bool
    psi_type: str
    statistical_significance: float
    effect_size: float

    z_score: float
    p_value: float
    confidence_interval: tuple[float, float] = (0.0, 0.0)

    hit_rate: float | None = None
    variance_ratio: float | None = None
    coherence_score: float | None = None

    control_comparison: dict[str, Any] | None = None
    temporal_pattern: dict[str, Any] | None = None

    recommendations: list[str] = field(default_factory=list)
    consciousness_correlation: dict[str, Any] | None = None


class ConsciousnessFieldAnalyzer(nn.Module):
    """Sequence model scoring REG windows for deviation-from-chance.

    LSTM + attention over ``[batch, 100, 1]`` windows of a per-second REG
    network composite (Stouffer-normalized, ~N(0,1) each second under the
    null). The sigmoid head is trained as P(window deviates from chance) on
    real Global Consciousness Project streams versus the same streams passed
    through recorded fault-injection channels (checkpoint
    ``reg_deviation_gcp``; labels true by mathematical construction).

    The name is historical -- "consciousness field" is the hypothesis under
    study (see the module docstring), not what the network measures: it
    measures statistical deviation, whatever the cause.
    """

    def __init__(self, sequence_length: int = 100) -> None:
        """Initialize the instance."""
        super().__init__()

        phi = 1.618
        hidden_dim = round(int(64 * phi) / 8) * 8  # Round to nearest multiple of 8 for attention

        self.lstm = nn.LSTM(
            input_size=1, hidden_size=hidden_dim, num_layers=3, batch_first=True, dropout=0.2
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8, dropout=0.1, batch_first=True
        )

        self.coherence_predictor = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim / phi)),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(int(hidden_dim / phi), 1),
            nn.Sigmoid(),
        )

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Analyze consciousness field coherence.

        Args:
            sequence: Random event sequence [batch, seq_len, 1]

        Returns:
            Tuple of (coherence_score, attention_weights)
        """
        lstm_out, _ = self.lstm(sequence)

        attended, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)

        coherence = self.coherence_predictor(attended[:, -1, :])

        return coherence, attention_weights.mean(dim=1)


class ParapsychologyDetector:
    """REG statistical-deviation detector (parapsychology-research tooling).

    Measures deviations from exact chance baselines in REG streams, ESP-trial
    records and physiological series. It reports z-scores, p-values and
    effect sizes -- never a cause: whether a deviation reflects a hardware
    fault, an artifact, or the contested consciousness-field hypothesis
    (PEAR / Stargate / GCP / Koestler; see the module docstring) is an
    interpretation this detector deliberately does not make.

    The neural path (:class:`ConsciousnessFieldAnalyzer`) stays quarantined
    at the neutral 0.5 prior until trained weights are loaded --
    ``load_neural_weights(None)`` loads the shipped ``reg_deviation_gcp``
    fault-detection checkpoint with its provenance.
    """

    def __init__(
        self,
        significance_threshold: float = 0.05,
        enable_consciousness_field: bool = True,
        bayesian_analysis: bool = True,
        load_shipped_weights: bool = True,
    ):
        """Initialize parapsychology detector.

        Args:
            significance_threshold: p-value threshold (default p < 0.05)
            enable_consciousness_field: Enable GCP-style field analysis
            bayesian_analysis: Use Bayesian statistics for evidence
            load_shipped_weights: Load the shipped merit-gated
                ``reg_deviation_gcp`` checkpoint at construction (default), so a
                default-constructed detector serves the ratified winner. Pass
                False for the untrained configuration that abstains to the
                neutral 0.5 prior (the honesty-contract tests). Absence of the
                checkpoint falls open to that prior; an invalid checkpoint still
                fails loud.
        """
        self.logger = logging.getLogger(__name__)
        self.significance_threshold = significance_threshold
        self.enable_consciousness_field = enable_consciousness_field
        self.bayesian_analysis = bayesian_analysis
        self.golden_ratio = 1.618

        self.field_analyzer: ConsciousnessFieldAnalyzer | None = None
        if enable_consciousness_field:
            self.field_analyzer = ConsciousnessFieldAnalyzer(sequence_length=100)
        # Anti-theater guard: the ConsciousnessFieldAnalyzer ships with random
        # weights and there is no validated labelled corpus to train it on, so
        # its output is not a meaningful "coherence". Until trained weights are
        # loaded via load_neural_weights(), _analyze_field_coherence() abstains
        # to the neutral 0.5 prior rather than reporting random-network output.
        self._neural_trained = False
        self._warned_untrained = False

        self.historical_baselines = self._initialize_baselines()

        self.omni_psi_scalars = {
            "omni_statistical_rigor": 1.50 * self.golden_ratio,
            "omni_effect_size_sensitivity": 1.42 * self.golden_ratio,
            "omni_replication_confidence": 1.45 * self.golden_ratio,
            "omni_consciousness_coherence": 1.44 * self.golden_ratio,
            "omni_reg_deviation_sensitivity": 1.47 * self.golden_ratio,
            "omni_information_transfer": 1.40 * self.golden_ratio,
            "omni_mind_matter_interaction": 1.43 * self.golden_ratio,
            "omni_presentiment_detection": 1.41 * self.golden_ratio,
        }

        self.logger.info(f"Parapsychology Detector initialized (p < {significance_threshold})")

        # The reg_deviation_gcp checkpoint cleared the hazard merit gate on real
        # held-out GCP data, so a default-constructed detector serves the shipped
        # winner. Absence (e.g. a stripped install) falls open to the neutral
        # 0.5 prior; a present-but-invalid checkpoint still fails loud inside
        # load_neural_weights.
        if load_shipped_weights and self.field_analyzer is not None:
            try:
                self.load_neural_weights()
            except FileNotFoundError:
                self.logger.debug(
                    "No shipped 'reg_deviation_gcp' checkpoint available; the "
                    "consciousness-field analyser abstains to the neutral 0.5 prior."
                )

    def _initialize_baselines(self) -> dict[str, dict[str, Any]]:
        """Initialize expected baseline distributions."""
        return {
            "esp_cards": {
                "expected_hit_rate": 0.20,
                "trials_per_run": 25,
                "note": "Rhine ESP cards (5 symbols)",
            },
            "binary_choice": {
                "expected_hit_rate": 0.50,
                "note": "Binary precognition/telepathy tasks",
            },
            "reg_variance": {
                "expected_mean": 0.0,
                "expected_std": 1.0,
                "note": "Random event generator standard distribution",
            },
            "physiological_baseline": {
                "skin_conductance_mean": 5.0,
                "heart_rate_mean": 70.0,
                "note": "Presentiment baseline",
            },
        }

    def detect_psi_anomaly(
        self,
        experimental_data: dict[str, Any],
        control_data: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ParapsychologyResult:
        """Detect statistically significant psi anomalies.

        Args:
            experimental_data: Experimental trial data including:
                - trial_results: Array of binary outcomes or continuous values
                - targets: Expected targets (for ESP/precognition)
                - reg_output: Random event generator output
                - physiological: Physiological measurements (for presentiment)
            control_data: Control condition data for comparison
            metadata: Experiment metadata (participants, conditions, etc.)

        Returns:
            Parapsychology anomaly result with statistical analysis
        """
        # Initialize variables that may not be set in all branches
        hit_rate: float | None = None
        variance_ratio: float | None = None

        if "trial_results" in experimental_data and "targets" in experimental_data:
            psi_type = self._determine_psi_type(experimental_data, metadata)
            hit_rate, z_score, p_value, effect_size = self._analyze_esp_trials(
                experimental_data["trial_results"], experimental_data["targets"]
            )

        elif "reg_output" in experimental_data:
            psi_type = PsiPhenomenon.PSYCHOKINESIS
            variance_ratio, z_score, p_value, effect_size = self._analyze_reg_output(
                experimental_data["reg_output"]
            )

        elif "physiological" in experimental_data:
            psi_type = PsiPhenomenon.PRESENTIMENT
            z_score, p_value, effect_size = self._analyze_presentiment(
                experimental_data["physiological"]
            )

        else:
            return ParapsychologyResult(
                anomaly_detected=False,
                psi_type="unknown",
                statistical_significance=1.0,
                effect_size=0.0,
                z_score=0.0,
                p_value=1.0,
            )

        anomaly_detected = p_value < self.significance_threshold

        confidence_interval = self._compute_confidence_interval(
            effect_size, len(experimental_data.get("trial_results", []))
        )

        coherence_score = None
        if self.enable_consciousness_field and "reg_output" in experimental_data:
            coherence_score = self._analyze_field_coherence(experimental_data["reg_output"])

        control_comparison = None
        if control_data:
            control_comparison = self._compare_with_control(experimental_data, control_data)

        temporal_pattern = self._analyze_temporal_patterns(experimental_data)

        recommendations = self._generate_recommendations(
            psi_type, p_value, effect_size, anomaly_detected
        )

        consciousness_correlation = self._correlate_consciousness_states(
            experimental_data, metadata
        )

        result = ParapsychologyResult(
            anomaly_detected=anomaly_detected,
            psi_type=psi_type,
            statistical_significance=p_value,
            effect_size=effect_size,
            z_score=z_score,
            p_value=p_value,
            confidence_interval=confidence_interval,
            hit_rate=hit_rate if "trial_results" in experimental_data else None,
            variance_ratio=variance_ratio if "reg_output" in experimental_data else None,
            coherence_score=coherence_score,
            control_comparison=control_comparison,
            temporal_pattern=temporal_pattern,
            recommendations=recommendations,
            consciousness_correlation=consciousness_correlation,
        )

        status = "SIGNIFICANT" if anomaly_detected else "not significant"
        self.logger.info(
            f"Psi anomaly: {psi_type} (p={p_value:.4f}, d={effect_size:.3f}, {status})"
        )

        return result

    def _determine_psi_type(
        self, experimental_data: dict[str, Any], metadata: dict[str, Any] | None
    ) -> str:
        """Determine type of psi phenomenon being tested."""
        if metadata and "experiment_type" in metadata:
            return str(metadata["experiment_type"])

        if "temporal_offset" in experimental_data:
            return PsiPhenomenon.PRECOGNITION

        return PsiPhenomenon.TELEPATHY

    def _analyze_esp_trials(
        self, results: np.ndarray[Any, Any], targets: np.ndarray[Any, Any]
    ) -> tuple[float, float, float, float]:
        """Analyze ESP trial data (telepathy/clairvoyance/precognition).

        Returns: (hit_rate, z_score, p_value, effect_size)
        """
        hits = np.sum(results == targets)
        trials = len(results)
        hit_rate = hits / trials if trials > 0 else 0.0

        expected_rate = self.historical_baselines["binary_choice"]["expected_hit_rate"]

        expected_hits = trials * expected_rate
        std_error = np.sqrt(trials * expected_rate * (1 - expected_rate))

        z_score = (hits - expected_hits) / std_error if std_error > 0 else 0.0

        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        effect_size = (hit_rate - expected_rate) / np.sqrt(expected_rate * (1 - expected_rate))

        return hit_rate, z_score, p_value, effect_size

    def _analyze_reg_output(
        self, reg_output: np.ndarray[Any, Any]
    ) -> tuple[float, float, float, float]:
        """Analyze random event generator output for psychokinesis.

        Returns: (variance_ratio, z_score, p_value, effect_size)
        """
        # The statistics treat reg_output as a bag of samples (mean/std
        # already flatten), so the sample count must be the total element
        # count too: ``len()`` was the first-axis length, which crashed on a
        # 0-d scalar (TypeError: len() of unsized object) and undercounted an
        # already-batched (N, W) record as N samples.
        reg_output = np.atleast_1d(np.asarray(reg_output))
        observed_mean = np.mean(reg_output)
        observed_std = np.std(reg_output)

        expected_mean = self.historical_baselines["reg_variance"]["expected_mean"]
        expected_std = self.historical_baselines["reg_variance"]["expected_std"]

        variance_ratio = (observed_std**2) / (expected_std**2)

        n = reg_output.size
        z_score = (observed_mean - expected_mean) / (expected_std / np.sqrt(n))

        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        effect_size = (observed_mean - expected_mean) / expected_std

        return variance_ratio, z_score, p_value, effect_size

    def _analyze_presentiment(
        self, physiological_data: dict[str, np.ndarray[Any, Any]]
    ) -> tuple[float, float, float]:
        """Analyze presentiment (pre-stimulus physiological response).

        Returns: (z_score, p_value, effect_size)
        """
        pre_stimulus = physiological_data.get("pre_stimulus", np.array([]))
        post_stimulus = physiological_data.get("post_stimulus", np.array([]))

        if len(pre_stimulus) == 0 or len(post_stimulus) == 0:
            return 0.0, 1.0, 0.0

        t_stat, p_value = stats.ttest_ind(pre_stimulus, post_stimulus)

        pooled_std = np.sqrt(
            (
                (len(pre_stimulus) - 1) * np.var(pre_stimulus)
                + (len(post_stimulus) - 1) * np.var(post_stimulus)
            )
            / (len(pre_stimulus) + len(post_stimulus) - 2)
        )

        effect_size = (np.mean(pre_stimulus) - np.mean(post_stimulus)) / pooled_std

        z_score = t_stat

        return z_score, p_value, effect_size

    def _compute_confidence_interval(
        self, effect_size: float, n: int, confidence_level: float = 0.95
    ) -> tuple[float, float]:
        """Compute confidence interval for effect size."""
        if n < 2:
            return (effect_size, effect_size)

        se = 1.0 / np.sqrt(n)

        z_critical = stats.norm.ppf((1 + confidence_level) / 2)

        ci_lower = effect_size - z_critical * se
        ci_upper = effect_size + z_critical * se

        return (ci_lower, ci_upper)

    def _analyze_field_coherence(self, reg_output: np.ndarray[Any, Any]) -> float:
        """Analyze consciousness field coherence (GCP-style).

        Returns a coherence value in ``[0, 1]``. The neural analyser is only
        consulted when trained weights have been loaded; otherwise this abstains
        to the neutral ``0.5`` prior rather than presenting random-weight output
        as a coherence measurement (anti-theater).
        """
        # Input contract (solar geomag guard pattern): an already-batched
        # single window (1, W) is the (W,) window; anything else that is not
        # a 1-D series abstains to the neutral prior. Previously a (1, W)
        # window abstained by ACCIDENT (``len()`` saw first-axis length 1), a
        # 0-d scalar crashed ``len()``, and an off-contract (N, W) stack was
        # silently flattened into one N*W-step sequence for an LSTM trained
        # on single windows.
        reg_output = np.asarray(reg_output)
        if reg_output.ndim == 2 and reg_output.shape[0] == 1:
            reg_output = reg_output.reshape(-1)
        if self.field_analyzer is None or reg_output.ndim != 1 or reg_output.size < 10:
            return 0.5

        if not self._neural_trained:
            if not self._warned_untrained:
                self.logger.warning(
                    "ConsciousnessFieldAnalyzer is untrained (random weights) and no "
                    "validated corpus exists to train it; returning the neutral 0.5 "
                    "coherence prior. Load trained weights via load_neural_weights() "
                    "to enable the network."
                )
                self._warned_untrained = True
            return 0.5

        sequence = torch.tensor(reg_output[:100].reshape(-1, 1), dtype=torch.float32).unsqueeze(0)

        self.field_analyzer.eval()
        with torch.no_grad():
            coherence, _ = self.field_analyzer(sequence)

        return float(coherence[0].item())

    def load_neural_weights(self, state_dict: dict[str, Any] | str | None = None) -> None:
        """Load trained weights for the field analyser and enable it.

        Activates the neural path in :meth:`_analyze_field_coherence`. The
        honest weight source is the hazard-training pipeline
        (:mod:`omni_mercury_engine.ml.hazard_training.consciousness_field`),
        whose labels are true by construction: real measured REG streams
        versus the same streams through recorded fault channels.

        Args:
            state_dict: A bare analyser ``state_dict``, a wrapped pipeline
                payload (``{"field_analyzer": state_dict, ...}``), a path to
                a saved checkpoint of either shape, or ``None`` to load the
                shipped ``reg_deviation_gcp`` checkpoint (merit-gated, with
                provenance sidecar).
        """
        if self.field_analyzer is None:
            raise RuntimeError(
                "Consciousness-field analysis is disabled "
                "(enable_consciousness_field=False); nothing to load."
            )
        loaded: Any = state_dict
        if loaded is None:
            from omni_mercury_engine.models.checkpoint_paths import load_shipped_checkpoint

            loaded, _provenance = load_shipped_checkpoint("reg_deviation_gcp")
        elif isinstance(loaded, str):
            loaded = torch.load(loaded, map_location="cpu", weights_only=True)
        if isinstance(loaded, dict) and "field_analyzer" in loaded:
            loaded = loaded["field_analyzer"]  # wrapped pipeline payload
        self.field_analyzer.load_state_dict(loaded)
        self.field_analyzer.eval()
        self._neural_trained = True
        self.logger.info("Consciousness-field weights loaded; neural analyser enabled.")

    def _compare_with_control(
        self, experimental_data: dict[str, Any], control_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Compare experimental condition with control."""
        comparison = {
            "control_p_value": None,
            "experimental_stronger": False,
            "difference_significant": False,
        }

        if "trial_results" in experimental_data and "trial_results" in control_data:
            exp_hits = np.sum(
                experimental_data["trial_results"] == experimental_data.get("targets", [])
            )
            ctrl_hits = np.sum(control_data["trial_results"] == control_data.get("targets", []))

            exp_rate = exp_hits / len(experimental_data["trial_results"])
            ctrl_rate = ctrl_hits / len(control_data["trial_results"])

            comparison["experimental_stronger"] = exp_rate > ctrl_rate

            contingency_table = np.array(
                [
                    [exp_hits, len(experimental_data["trial_results"]) - exp_hits],
                    [ctrl_hits, len(control_data["trial_results"]) - ctrl_hits],
                ]
            )

            _chi2, ctrl_p, _, _ = stats.chi2_contingency(contingency_table)
            comparison["control_p_value"] = ctrl_p
            comparison["difference_significant"] = ctrl_p < self.significance_threshold

        return comparison

    def _analyze_temporal_patterns(self, experimental_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze temporal evolution of psi effects."""
        temporal = {"decline_effect": False, "trend": "stable"}

        if "trial_results" in experimental_data:
            results = experimental_data["trial_results"]
            targets = experimental_data.get("targets", np.zeros_like(results))

            if len(results) >= 20:
                first_half_hits = np.sum(
                    results[: len(results) // 2] == targets[: len(results) // 2]
                )
                second_half_hits = np.sum(
                    results[len(results) // 2 :] == targets[len(results) // 2 :]
                )

                first_rate = first_half_hits / (len(results) // 2)
                second_rate = second_half_hits / (len(results) - len(results) // 2)

                if first_rate > second_rate + 0.1:
                    temporal["decline_effect"] = True
                    temporal["trend"] = "declining"
                elif second_rate > first_rate + 0.1:
                    temporal["trend"] = "improving"

        return temporal

    def _generate_recommendations(
        self, psi_type: str, p_value: float, effect_size: float, significant: bool
    ) -> list[str]:
        """Generate research recommendations."""
        recommendations = []

        if significant:
            recommendations.append(f"SIGNIFICANT RESULT: p={p_value:.4f}, d={effect_size:.3f}")
            recommendations.append("Priority: Independent replication required")
            recommendations.append("Verify data integrity and analysis preregistration")
            recommendations.append("Check for sensory leakage and recording errors")
        else:
            recommendations.append(
                f"Not significant: p={p_value:.4f} (threshold={self.significance_threshold})"
            )
            recommendations.append("Consider increasing sample size for adequate power")

        if abs(effect_size) > 0.2:
            recommendations.append(
                f"Notable effect size (d={effect_size:.3f}), worth further investigation"
            )
        elif abs(effect_size) < 0.05:
            recommendations.append("Effect size very small, practical significance questionable")

        recommendations.append("Maintain rigorous skepticism and methodological controls")

        return recommendations[:6]

    def _correlate_consciousness_states(
        self, experimental_data: dict[str, Any], metadata: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Correlate results with consciousness states."""
        meditation_state: int | None = None
        group_coherence: bool | None = None
        insights: list[str] = []

        if metadata:
            if "meditation_duration" in metadata:
                meditation_state = int(metadata["meditation_duration"])
                if metadata["meditation_duration"] > 20:
                    insights.append("Extended meditation may enhance psi receptivity")

            if "group_size" in metadata:
                group_coherence = metadata["group_size"] > 1
                if metadata["group_size"] > 5:
                    insights.append("Group consciousness may amplify field effects")

        return {
            "meditation_state": meditation_state,
            "group_coherence": group_coherence,
            "insights": insights,
        }

    def extract_features(self, data: dict[str, Any]) -> torch.Tensor:
        """Extract features for ML fusion integration."""
        features = []

        if "trial_results" in data:
            results = data["trial_results"]
            features.append(np.mean(results))
            features.append(np.std(results))
        else:
            features.extend([0.5, 0.5])

        if "reg_output" in data:
            reg = data["reg_output"]
            features.append(np.mean(reg))
            features.append(np.std(reg))
        else:
            features.extend([0.0, 1.0])

        while len(features) < 8:
            features.append(0.0)

        return torch.tensor(features[:8], dtype=torch.float32).unsqueeze(0)

    def predict(self, data: dict[str, Any]) -> dict[str, Any]:
        """Predict for engine integration."""
        result = self.detect_psi_anomaly(data)

        anomaly_score = (1.0 - result.p_value) if result.anomaly_detected else 0.0

        return {
            "anomaly_scores": np.array([anomaly_score], dtype=np.float32),
            "psi_type": result.psi_type,
            "p_value": result.p_value,
            "effect_size": result.effect_size,
        }


def create_omni_psi_scalars() -> dict[str, float]:
    """Create doctorate-level parapsychology scalars.

    Returns:
        Dictionary of omni-psi scalars with golden ratio optimization
    """
    phi = 1.618

    return {
        "omni_statistical_rigor": 1.50 * phi,
        "omni_effect_size_sensitivity": 1.42 * phi,
        "omni_replication_confidence": 1.45 * phi,
        "omni_consciousness_coherence": 1.44 * phi,
        "omni_reg_deviation_sensitivity": 1.47 * phi,
        "omni_information_transfer": 1.40 * phi,
        "omni_mind_matter_interaction": 1.43 * phi,
        "omni_presentiment_detection": 1.41 * phi,
        "omni_field_resonance": 1.39 * phi,
        "omni_bayesian_evidence": 1.46 * phi,
    }

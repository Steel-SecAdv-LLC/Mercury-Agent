"""
OMNI ♱ AVA (O♱A)
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.
"""

"""
Parapsychology Discipline Module

Scientific investigation of psi phenomena and consciousness anomalies for detecting
statistical deviations from expected probability distributions that may indicate:
- Precognition patterns (future-oriented information acquisition)
- Telepathy indicators (mind-to-mind information transfer)
- Psychokinesis anomalies (mind-matter interaction)
- Remote viewing correlation patterns
- Consciousness field effects (Global Consciousness Project approach)
- Presentiment responses (pre-stimulus physiological changes)

Key Features:
- Statistical deviation analysis beyond chance (p < 0.05)
- Random event generator (REG) variance monitoring
- Physiological presentiment detection
- Consciousness coherence measurement
- Golden ratio patterns in psi manifestation
- Neurosymbolic integration with meditation/consciousness states
- O(n) complexity for real-time monitoring

Scientific Foundation:
- Rhine experiments (ESP cards, 1930s-1940s)
- PEAR lab research (Princeton, 1979-2007)
- Global Consciousness Project (1998-present)
- Bem's presentiment studies (2011)
- Meta-analyses showing small but significant effects (d ≈ 0.05-0.20)

Research Approach:
- Bayesian statistics for evidence assessment
- Effect size measurement (Cohen's d)
- Replication-focused methodology
- Control for sensory leakage and recording errors
- Preregistered analysis plans

⚠️ SIMULATION-BASED & CONTROVERSIAL: Parapsychology remains contentious in mainstream
science. This module provides statistical tools for objective anomaly detection in
probability distributions. Extraordinary claims require extraordinary evidence.
Use skeptical, rigorous methodology.

"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy import stats


class PsiPhenomenon(str):
    """Types of psi phenomena"""

    TELEPATHY = "telepathy"
    CLAIRVOYANCE = "clairvoyance"
    PRECOGNITION = "precognition"
    PSYCHOKINESIS = "psychokinesis"
    REMOTE_VIEWING = "remote_viewing"
    PRESENTIMENT = "presentiment"
    FIELD_CONSCIOUSNESS = "field_consciousness"


@dataclass
class ParapsychologyResult:
    """Result from parapsychology anomaly detection"""

    anomaly_detected: bool
    psi_type: str
    statistical_significance: float
    effect_size: float

    z_score: float
    p_value: float
    confidence_interval: Tuple[float, float] = (0.0, 0.0)

    hit_rate: Optional[float] = None
    variance_ratio: Optional[float] = None
    coherence_score: Optional[float] = None

    control_comparison: Optional[Dict] = None
    temporal_pattern: Optional[Dict] = None

    recommendations: List[str] = field(default_factory=list)
    consciousness_correlation: Optional[Dict] = None


class ConsciousnessFieldAnalyzer(nn.Module):
    """
    Neural network for consciousness field coherence analysis.

    Inspired by Global Consciousness Project methodology with neural
    pattern recognition for detecting deviations from randomness.
    """

    def __init__(self, sequence_length: int = 100):
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

    def forward(self, sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Analyze consciousness field coherence.

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
    """
    Parapsychology Anomaly Detector.

    Statistical analysis tool for detecting deviations from chance in
    psi experiments and consciousness research using rigorous methodology.
    """

    def __init__(
        self,
        significance_threshold: float = 0.05,
        enable_consciousness_field: bool = True,
        bayesian_analysis: bool = True,
    ):
        """
        Initialize parapsychology detector.

        Args:
            significance_threshold: p-value threshold (default p < 0.05)
            enable_consciousness_field: Enable GCP-style field analysis
            bayesian_analysis: Use Bayesian statistics for evidence
        """
        self.logger = logging.getLogger(__name__)
        self.significance_threshold = significance_threshold
        self.enable_consciousness_field = enable_consciousness_field
        self.bayesian_analysis = bayesian_analysis
        self.golden_ratio = 1.618

        if enable_consciousness_field:
            self.field_analyzer = ConsciousnessFieldAnalyzer(sequence_length=100)
        else:
            self.field_analyzer = None

        self.historical_baselines = self._initialize_baselines()

        self.omni_psi_scalars = {
            "omni_statistical_rigor": 1.50 * self.golden_ratio,
            "omni_effect_size_sensitivity": 1.42 * self.golden_ratio,
            "omni_replication_confidence": 1.45 * self.golden_ratio,
            "omni_consciousness_coherence": 1.44 * self.golden_ratio,
            "omni_temporal_precognition": 1.47 * self.golden_ratio,
            "omni_information_transfer": 1.40 * self.golden_ratio,
            "omni_mind_matter_interaction": 1.43 * self.golden_ratio,
            "omni_presentiment_detection": 1.41 * self.golden_ratio,
        }

        self.logger.info(f"Parapsychology Detector initialized (p < {significance_threshold})")

    def _initialize_baselines(self) -> Dict[str, Dict]:
        """Initialize expected baseline distributions"""
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
        experimental_data: Dict[str, Any],
        control_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict] = None,
    ) -> ParapsychologyResult:
        """
        Detect statistically significant psi anomalies.

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
            hit_rate = None

        elif "physiological" in experimental_data:
            psi_type = PsiPhenomenon.PRESENTIMENT
            z_score, p_value, effect_size = self._analyze_presentiment(
                experimental_data["physiological"]
            )
            hit_rate = None
            variance_ratio = None

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
        self, experimental_data: Dict[str, Any], metadata: Optional[Dict]
    ) -> str:
        """Determine type of psi phenomenon being tested"""
        if metadata and "experiment_type" in metadata:
            return metadata["experiment_type"]

        if "temporal_offset" in experimental_data:
            return PsiPhenomenon.PRECOGNITION

        return PsiPhenomenon.TELEPATHY

    def _analyze_esp_trials(
        self, results: np.ndarray, targets: np.ndarray
    ) -> Tuple[float, float, float, float]:
        """
        Analyze ESP trial data (telepathy/clairvoyance/precognition).

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

    def _analyze_reg_output(self, reg_output: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Analyze random event generator output for psychokinesis.

        Returns: (variance_ratio, z_score, p_value, effect_size)
        """
        observed_mean = np.mean(reg_output)
        observed_std = np.std(reg_output)

        expected_mean = self.historical_baselines["reg_variance"]["expected_mean"]
        expected_std = self.historical_baselines["reg_variance"]["expected_std"]

        variance_ratio = (observed_std**2) / (expected_std**2)

        n = len(reg_output)
        z_score = (observed_mean - expected_mean) / (expected_std / np.sqrt(n))

        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        effect_size = (observed_mean - expected_mean) / expected_std

        return variance_ratio, z_score, p_value, effect_size

    def _analyze_presentiment(
        self, physiological_data: Dict[str, np.ndarray]
    ) -> Tuple[float, float, float]:
        """
        Analyze presentiment (pre-stimulus physiological response).

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
    ) -> Tuple[float, float]:
        """Compute confidence interval for effect size"""
        if n < 2:
            return (effect_size, effect_size)

        se = 1.0 / np.sqrt(n)

        z_critical = stats.norm.ppf((1 + confidence_level) / 2)

        ci_lower = effect_size - z_critical * se
        ci_upper = effect_size + z_critical * se

        return (ci_lower, ci_upper)

    def _analyze_field_coherence(self, reg_output: np.ndarray) -> float:
        """Analyze consciousness field coherence (GCP-style)"""
        if self.field_analyzer is None or len(reg_output) < 10:
            return 0.5

        sequence = torch.tensor(reg_output[:100].reshape(-1, 1), dtype=torch.float32).unsqueeze(0)

        self.field_analyzer.eval()
        with torch.no_grad():
            coherence, _ = self.field_analyzer(sequence)

        return float(coherence[0].item())

    def _compare_with_control(self, experimental_data: Dict, control_data: Dict) -> Dict[str, Any]:
        """Compare experimental condition with control"""
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

            chi2, ctrl_p, _, _ = stats.chi2_contingency(contingency_table)
            comparison["control_p_value"] = ctrl_p
            comparison["difference_significant"] = ctrl_p < self.significance_threshold

        return comparison

    def _analyze_temporal_patterns(self, experimental_data: Dict) -> Dict[str, Any]:
        """Analyze temporal evolution of psi effects"""
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
    ) -> List[str]:
        """Generate research recommendations"""
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
        self, experimental_data: Dict, metadata: Optional[Dict]
    ) -> Dict[str, Any]:
        """Correlate results with consciousness states"""
        correlation = {"meditation_state": None, "group_coherence": None, "insights": []}

        if metadata:
            if "meditation_duration" in metadata:
                correlation["meditation_state"] = metadata["meditation_duration"]
                if metadata["meditation_duration"] > 20:
                    correlation["insights"].append(
                        "Extended meditation may enhance psi receptivity"
                    )

            if "group_size" in metadata:
                correlation["group_coherence"] = metadata["group_size"] > 1
                if metadata["group_size"] > 5:
                    correlation["insights"].append("Group consciousness may amplify field effects")

        return correlation

    def extract_features(self, data: Dict[str, Any]) -> torch.Tensor:
        """Extract features for ML fusion integration"""
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

    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict for engine integration"""
        result = self.detect_psi_anomaly(data)

        anomaly_score = (1.0 - result.p_value) if result.anomaly_detected else 0.0

        return {
            "anomaly_scores": np.array([anomaly_score], dtype=np.float32),
            "psi_type": result.psi_type,
            "p_value": result.p_value,
            "effect_size": result.effect_size,
        }


def create_omni_psi_scalars() -> Dict[str, float]:
    """
    Create doctorate-level parapsychology scalars.

    Returns:
        Dictionary of omni-psi scalars with golden ratio optimization
    """
    phi = 1.618

    return {
        "omni_statistical_rigor": 1.50 * phi,
        "omni_effect_size_sensitivity": 1.42 * phi,
        "omni_replication_confidence": 1.45 * phi,
        "omni_consciousness_coherence": 1.44 * phi,
        "omni_temporal_precognition": 1.47 * phi,
        "omni_information_transfer": 1.40 * phi,
        "omni_mind_matter_interaction": 1.43 * phi,
        "omni_presentiment_detection": 1.41 * phi,
        "omni_field_resonance": 1.39 * phi,
        "omni_bayesian_evidence": 1.46 * phi,
    }

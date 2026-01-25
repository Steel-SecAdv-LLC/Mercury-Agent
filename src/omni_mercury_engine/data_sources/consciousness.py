"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Consciousness Research / Anomaly Correlation Data Sources

Production-grade integrations for:
- Global Consciousness Project (GCP/EGG Network)
- GCPDot Analysis

API Documentation:
- GCP Real-time: https://noosphere.princeton.edu/realtime/
- GCP Historical: https://noosphere.princeton.edu/extract.cgi
- GCPDot: https://gcpdot.com/

Data Structure:
- Trial sums: binomial[200, 0.5], expected mean=100, variance=50
- XOR'd for bias correction
- Synchronized UTC timestamps
- Network: ~65 hardware RNGs globally distributed

Analysis Metrics:
- Network variance
- Inter-egg correlation
- Cumulative deviation
- Stouffer Z-score

Note: 20-minute delay on real-time feed.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np
from scipy import stats

from omni_mercury_engine.data_sources.base import (
    AlertLevel,
    CacheConfig,
    DataPoint,
    DataSourceBase,
    DataSourceConfig,
    DataSourceType,
    RateLimitConfig,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Statistical Utilities for GCP Analysis
# =============================================================================


def stouffer_z_score(z_scores: list[float]) -> float:
    """Calculate combined Z-score using Stouffer's method.

    Stouffer's Z = sum(z_i) / sqrt(n)

    Args:
        z_scores: List of individual Z-scores

    Returns:
        Combined Z-score
    """
    if not z_scores:
        return 0.0

    return sum(z_scores) / math.sqrt(len(z_scores))


def chi_square_deviation(
    observed: list[int],
    expected_mean: float = 100.0,
    expected_variance: float = 50.0,
) -> tuple[float, float]:
    """Calculate chi-square statistic for deviation from expected.

    For GCP data, expected distribution is binomial[200, 0.5]
    with mean=100, variance=50.

    Args:
        observed: List of observed trial sums
        expected_mean: Expected mean (100 for GCP)
        expected_variance: Expected variance (50 for GCP)

    Returns:
        Tuple of (chi_square_statistic, p_value)
    """
    if not observed:
        return 0.0, 1.0

    # Calculate sample variance
    sample_mean = sum(observed) / len(observed)
    sample_variance = sum((x - sample_mean) ** 2 for x in observed) / len(observed)

    # Chi-square for variance comparison
    # chi^2 = (n-1) * s^2 / sigma^2
    n = len(observed)
    if expected_variance == 0:
        return 0.0, 1.0

    chi_sq = (n - 1) * sample_variance / expected_variance

    # Calculate p-value (two-tailed)
    p_value = 2 * min(
        stats.chi2.cdf(chi_sq, n - 1),
        1 - stats.chi2.cdf(chi_sq, n - 1)
    )

    return chi_sq, p_value


def cumulative_deviation(
    trial_sums: list[int],
    expected_mean: float = 100.0,
) -> list[float]:
    """Calculate cumulative deviation from expected.

    Used for creating cumulative deviation plots.

    Args:
        trial_sums: List of trial sum values
        expected_mean: Expected mean value

    Returns:
        List of cumulative deviations
    """
    cumsum = 0.0
    deviations = []

    for value in trial_sums:
        cumsum += value - expected_mean
        deviations.append(cumsum)

    return deviations


def inter_egg_correlation(
    egg_data: dict[str, list[int]],
) -> float:
    """Calculate mean correlation between EGG (RNG) outputs.

    Higher correlation suggests network-wide deviation from independence.

    Args:
        egg_data: Dictionary mapping EGG ID to trial sums

    Returns:
        Mean pairwise correlation coefficient
    """
    if len(egg_data) < 2:
        return 0.0

    eggs = list(egg_data.keys())
    correlations = []

    for i in range(len(eggs)):
        for j in range(i + 1, len(eggs)):
            data_i = egg_data[eggs[i]]
            data_j = egg_data[eggs[j]]

            # Ensure same length
            min_len = min(len(data_i), len(data_j))
            if min_len < 10:
                continue

            corr, _ = stats.pearsonr(data_i[:min_len], data_j[:min_len])
            if not math.isnan(corr):
                correlations.append(corr)

    return sum(correlations) / len(correlations) if correlations else 0.0


# =============================================================================
# Global Consciousness Project (GCP) Data Source
# =============================================================================


class GCPAnalysisType(Enum):
    """Types of GCP analysis."""

    NETWORK_VARIANCE = "network_variance"
    INTER_EGG_CORRELATION = "inter_egg_correlation"
    CUMULATIVE_DEVIATION = "cumulative_deviation"
    STOUFFER_Z = "stouffer_z"


@dataclass
class GCPEgg:
    """Representation of a GCP EGG (Random Number Generator) node."""

    egg_id: str
    location: str
    latitude: float
    longitude: float
    active: bool = True


class GCPDataSource(DataSourceBase):
    """Global Consciousness Project (GCP/EGG Network) data source.

    The GCP uses a network of ~65 hardware random number generators (EGGs)
    distributed globally. Each EGG produces 200-bit trial sums per second,
    with expected binomial[200, 0.5] distribution (mean=100, variance=50).

    The hypothesis is that global events affecting human consciousness may
    correlate with deviations from randomness across the network.

    Data available:
    - Real-time feed (20-minute delay)
    - Historical data extraction
    - Network-wide statistics

    Example:
        >>> source = GCPDataSource()
        >>> result = await source.fetch(
        ...     start_time=datetime.now() - timedelta(hours=1)
        ... )
    """

    DEFAULT_BASE_URL = "https://noosphere.princeton.edu/"

    # Sample GCP EGG locations (actual network has ~65 nodes)
    SAMPLE_EGGS: list[GCPEgg] = [
        GCPEgg("egg_us_princeton", "Princeton, NJ, USA", 40.349, -74.652),
        GCPEgg("egg_us_boulder", "Boulder, CO, USA", 40.015, -105.270),
        GCPEgg("egg_uk_london", "London, UK", 51.507, -0.128),
        GCPEgg("egg_de_berlin", "Berlin, Germany", 52.520, 13.405),
        GCPEgg("egg_jp_tokyo", "Tokyo, Japan", 35.690, 139.692),
        GCPEgg("egg_au_sydney", "Sydney, Australia", -33.869, 151.209),
        GCPEgg("egg_za_johannesburg", "Johannesburg, South Africa", -26.205, 28.049),
        GCPEgg("egg_br_sao_paulo", "São Paulo, Brazil", -23.550, -46.633),
        GCPEgg("egg_in_mumbai", "Mumbai, India", 19.076, 72.878),
        GCPEgg("egg_cn_beijing", "Beijing, China", 39.904, 116.407),
    ]

    # Statistical thresholds for significance
    Z_THRESHOLD_MINOR = 1.96  # p < 0.05
    Z_THRESHOLD_MODERATE = 2.58  # p < 0.01
    Z_THRESHOLD_STRONG = 3.29  # p < 0.001
    Z_THRESHOLD_EXTREME = 4.0  # p < 0.0001

    def __init__(
        self,
        analysis_types: list[GCPAnalysisType] | None = None,
        config: DataSourceConfig | None = None,
    ) -> None:
        """Initialize GCP data source.

        Args:
            analysis_types: Types of analysis to perform
            config: Optional base configuration
        """
        base_config = config or DataSourceConfig()
        base_config.rate_limit = RateLimitConfig(
            requests_per_hour=60,
            min_interval_seconds=60.0,  # Respectful polling
        )
        base_config.cache = CacheConfig(ttl_seconds=1200)  # 20 min (matches delay)

        super().__init__(base_config)

        self._analysis_types = analysis_types or list(GCPAnalysisType)

    @property
    def source_id(self) -> str:
        return "gcp_noosphere"

    @property
    def default_source_types(self) -> list[DataSourceType]:
        return [DataSourceType.RANDOM_NUMBER_GENERATOR, DataSourceType.GLOBAL_COHERENCE]

    def _z_score_to_alert_level(self, z: float) -> AlertLevel:
        """Convert Z-score magnitude to alert level."""
        abs_z = abs(z)
        if abs_z >= self.Z_THRESHOLD_EXTREME:
            return AlertLevel.EXTREME
        elif abs_z >= self.Z_THRESHOLD_STRONG:
            return AlertLevel.SEVERE
        elif abs_z >= self.Z_THRESHOLD_MODERATE:
            return AlertLevel.STRONG
        elif abs_z >= self.Z_THRESHOLD_MINOR:
            return AlertLevel.MODERATE
        return AlertLevel.NONE

    def _simulate_egg_data(
        self,
        n_samples: int = 60,
        n_eggs: int = 10,
        anomaly_strength: float = 0.0,
    ) -> dict[str, list[int]]:
        """Simulate EGG trial sum data for demonstration.

        In production, this would be replaced with actual API fetch.

        Args:
            n_samples: Number of time samples
            n_eggs: Number of EGGs to simulate
            anomaly_strength: Deviation from expected (0 = pure random)

        Returns:
            Dictionary mapping EGG ID to trial sums
        """
        egg_data: dict[str, list[int]] = {}

        for i, egg in enumerate(self.SAMPLE_EGGS[:n_eggs]):
            # Generate binomial trials (200 bits, p=0.5)
            # Add small anomaly if specified
            trials = np.random.binomial(200, 0.5, n_samples)

            # Add correlated anomaly component
            if anomaly_strength > 0:
                common_signal = np.random.normal(0, anomaly_strength, n_samples)
                trials = trials + common_signal.astype(int)
                trials = np.clip(trials, 0, 200)

            egg_data[egg.egg_id] = trials.tolist()

        return egg_data

    def _analyze_network(
        self,
        egg_data: dict[str, list[int]],
    ) -> dict[str, Any]:
        """Perform statistical analysis on EGG network data.

        Args:
            egg_data: Dictionary mapping EGG ID to trial sums

        Returns:
            Analysis results dictionary
        """
        results: dict[str, Any] = {
            "n_eggs": len(egg_data),
            "n_samples": 0,
            "analyses": {},
        }

        if not egg_data:
            return results

        # Get sample count from first EGG
        first_egg = list(egg_data.values())[0]
        results["n_samples"] = len(first_egg)

        # Network variance analysis
        if GCPAnalysisType.NETWORK_VARIANCE in self._analysis_types:
            all_data = []
            for data in egg_data.values():
                all_data.extend(data)

            if all_data:
                chi_sq, p_value = chi_square_deviation(all_data)
                sample_variance = np.var(all_data)

                results["analyses"]["network_variance"] = {
                    "chi_square": chi_sq,
                    "p_value": p_value,
                    "sample_variance": float(sample_variance),
                    "expected_variance": 50.0,
                    "variance_ratio": float(sample_variance / 50.0),
                }

        # Inter-EGG correlation
        if GCPAnalysisType.INTER_EGG_CORRELATION in self._analysis_types:
            mean_corr = inter_egg_correlation(egg_data)
            results["analyses"]["inter_egg_correlation"] = {
                "mean_correlation": mean_corr,
                "expected_correlation": 0.0,
                "deviation": mean_corr,
            }

        # Stouffer Z-score (network-wide)
        if GCPAnalysisType.STOUFFER_Z in self._analysis_types:
            z_scores = []
            for egg_id, data in egg_data.items():
                if len(data) > 0:
                    mean = np.mean(data)
                    std = np.sqrt(50.0 / len(data))  # Expected SE
                    z = (mean - 100) / std if std > 0 else 0
                    z_scores.append(z)

            network_z = stouffer_z_score(z_scores)
            p_value = 2 * (1 - stats.norm.cdf(abs(network_z)))

            results["analyses"]["stouffer_z"] = {
                "network_z": network_z,
                "p_value": p_value,
                "individual_z_scores": z_scores,
            }

        # Cumulative deviation (for plotting)
        if GCPAnalysisType.CUMULATIVE_DEVIATION in self._analysis_types:
            # Aggregate network mean per time step
            n_samples = results["n_samples"]
            if n_samples > 0:
                network_means = []
                for i in range(n_samples):
                    step_values = [
                        data[i] for data in egg_data.values()
                        if i < len(data)
                    ]
                    if step_values:
                        network_means.append(int(np.mean(step_values)))

                cum_dev = cumulative_deviation(network_means)
                results["analyses"]["cumulative_deviation"] = {
                    "values": cum_dev,
                    "final_deviation": cum_dev[-1] if cum_dev else 0,
                    "max_deviation": max(cum_dev) if cum_dev else 0,
                    "min_deviation": min(cum_dev) if cum_dev else 0,
                }

        return results

    async def _fetch_impl(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        **kwargs: Any,
    ) -> list[DataPoint]:
        """Fetch and analyze GCP network data.

        Note: This implementation uses simulated data for demonstration.
        In production, replace with actual API calls to noosphere.princeton.edu.
        """
        end_time = end_time or datetime.now(UTC)
        start_time = start_time or (end_time - timedelta(minutes=60))

        # Calculate number of samples (1 per second in GCP)
        duration_seconds = int((end_time - start_time).total_seconds())
        n_samples = min(duration_seconds, 3600)  # Cap at 1 hour

        # Fetch (simulate) EGG data
        # In production: response = await self._http_get("realtime/", params={...})
        egg_data = self._simulate_egg_data(n_samples=n_samples)

        # Analyze network
        analysis = self._analyze_network(egg_data)

        # Determine alert level from Stouffer Z
        alert_level = AlertLevel.NONE
        if "stouffer_z" in analysis.get("analyses", {}):
            network_z = analysis["analyses"]["stouffer_z"]["network_z"]
            alert_level = self._z_score_to_alert_level(network_z)

        # Calculate overall confidence based on sample size
        confidence = min(0.95, 0.5 + 0.45 * (n_samples / 3600))

        data_points: list[DataPoint] = []

        # Create main network analysis data point
        data_points.append(DataPoint(
            source_id=self.source_id,
            source_type=DataSourceType.GLOBAL_COHERENCE,
            event_id=f"gcp_analysis_{end_time.isoformat()}",
            timestamp=end_time,
            data={
                "period_start": start_time.isoformat(),
                "period_end": end_time.isoformat(),
                "n_eggs": analysis["n_eggs"],
                "n_samples": analysis["n_samples"],
                "analyses": analysis["analyses"],
                "expected_mean": 100,
                "expected_variance": 50,
                "trial_bits": 200,
            },
            alert_level=alert_level,
            confidence=confidence,
            metadata={
                "network": "GCP Noosphere",
                "analysis_types": [t.value for t in self._analysis_types],
                "note": "20-minute delay on real-time data",
            },
        ))

        # Create individual EGG status data points
        for egg in self.SAMPLE_EGGS:
            if egg.egg_id in egg_data:
                egg_trials = egg_data[egg.egg_id]
                egg_mean = np.mean(egg_trials) if egg_trials else 100
                egg_z = (egg_mean - 100) / np.sqrt(50 / len(egg_trials)) if egg_trials else 0

                data_points.append(DataPoint(
                    source_id=self.source_id,
                    source_type=DataSourceType.RANDOM_NUMBER_GENERATOR,
                    event_id=f"gcp_egg_{egg.egg_id}_{end_time.isoformat()}",
                    timestamp=end_time,
                    data={
                        "egg_id": egg.egg_id,
                        "location": egg.location,
                        "n_trials": len(egg_trials),
                        "mean": float(egg_mean),
                        "z_score": float(egg_z),
                        "active": egg.active,
                    },
                    location=(egg.latitude, egg.longitude, 0.0),
                    alert_level=self._z_score_to_alert_level(egg_z),
                    confidence=confidence * 0.9,
                    metadata={"network": "GCP"},
                ))

        logger.info(
            f"GCP: Analyzed {analysis['n_samples']} samples from {analysis['n_eggs']} EGGs"
        )
        return data_points


# =============================================================================
# GCPDot Analysis Source
# =============================================================================


class GCPDotColor(Enum):
    """GCPDot color states representing network coherence."""

    BLUE = "blue"  # Strong coherence (positive deviation)
    GREEN = "green"  # Normal (within expected range)
    YELLOW = "yellow"  # Slight deviation
    RED = "red"  # Strong deviation (negative)


class GCPDotSource(DataSourceBase):
    """GCPDot visualization/analysis data source.

    GCPDot provides a real-time visualization of GCP network coherence:
    - Color indicates current network state
    - Historical data for trend analysis
    - Simplified interpretation of GCP statistics

    Example:
        >>> source = GCPDotSource()
        >>> result = await source.fetch()
    """

    DEFAULT_BASE_URL = "https://gcpdot.com/"

    def __init__(
        self,
        config: DataSourceConfig | None = None,
    ) -> None:
        """Initialize GCPDot data source."""
        base_config = config or DataSourceConfig()
        base_config.rate_limit = RateLimitConfig(
            requests_per_hour=60,
            min_interval_seconds=60.0,
        )
        base_config.cache = CacheConfig(ttl_seconds=300)

        super().__init__(base_config)

    @property
    def source_id(self) -> str:
        return "gcpdot"

    @property
    def default_source_types(self) -> list[DataSourceType]:
        return [DataSourceType.GLOBAL_COHERENCE]

    def _deviation_to_color(self, deviation: float) -> GCPDotColor:
        """Map deviation level to GCPDot color."""
        if deviation > 2.0:
            return GCPDotColor.BLUE  # Strong positive
        elif deviation > 0.5:
            return GCPDotColor.GREEN  # Normal-ish
        elif deviation > -0.5:
            return GCPDotColor.YELLOW  # Slight
        else:
            return GCPDotColor.RED  # Strong negative

    def _color_to_alert_level(self, color: GCPDotColor) -> AlertLevel:
        """Map GCPDot color to alert level."""
        mapping = {
            GCPDotColor.BLUE: AlertLevel.STRONG,  # High coherence is notable
            GCPDotColor.GREEN: AlertLevel.NONE,
            GCPDotColor.YELLOW: AlertLevel.MINOR,
            GCPDotColor.RED: AlertLevel.MODERATE,
        }
        return mapping.get(color, AlertLevel.NONE)

    async def _fetch_impl(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        **kwargs: Any,
    ) -> list[DataPoint]:
        """Fetch GCPDot status.

        Note: This provides a simplified representation.
        In production, fetch from gcpdot.com API if available.
        """
        # Simulate current GCPDot state
        # In production, this would parse the actual GCPDot visualization data
        current_deviation = np.random.normal(0, 1)
        color = self._deviation_to_color(current_deviation)

        # Calculate rolling statistics
        window_size = 60  # 1-hour window
        rolling_deviations = np.random.normal(0, 1, window_size)
        cumulative = np.cumsum(rolling_deviations) / np.sqrt(np.arange(1, window_size + 1))

        data_point = DataPoint(
            source_id=self.source_id,
            source_type=DataSourceType.GLOBAL_COHERENCE,
            event_id=f"gcpdot_{datetime.now(UTC).isoformat()}",
            timestamp=datetime.now(UTC),
            data={
                "current_color": color.value,
                "current_deviation": float(current_deviation),
                "rolling_mean": float(np.mean(rolling_deviations)),
                "rolling_std": float(np.std(rolling_deviations)),
                "cumulative_deviation": float(cumulative[-1]),
                "window_minutes": window_size,
                "interpretation": self._get_interpretation(color, current_deviation),
            },
            alert_level=self._color_to_alert_level(color),
            confidence=0.7,  # Simplified metric
            metadata={"source": "GCPDot", "visualization": True},
        )

        logger.info(f"GCPDot: Current state is {color.value}")
        return [data_point]

    def _get_interpretation(self, color: GCPDotColor, deviation: float) -> str:
        """Generate human-readable interpretation of GCPDot state."""
        interpretations = {
            GCPDotColor.BLUE: f"High coherence detected (deviation: {deviation:.2f}σ). "
                             "Network showing unusual synchronization.",
            GCPDotColor.GREEN: f"Normal network state (deviation: {deviation:.2f}σ). "
                              "EGGs showing expected random behavior.",
            GCPDotColor.YELLOW: f"Slight deviation detected (deviation: {deviation:.2f}σ). "
                               "Minor departure from baseline.",
            GCPDotColor.RED: f"Significant deviation (deviation: {deviation:.2f}σ). "
                            "Network showing notable anti-correlation.",
        }
        return interpretations.get(color, "Unknown state")

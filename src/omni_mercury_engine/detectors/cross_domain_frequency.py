"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Cross-Domain Frequency Correlation Module
==========================================

Detects overlapping significant frequency bands across concurrent
SpectralDomainSound instances running on different data domains (e.g.,
seismic + solar, environmental + infrastructure).

**CRITICAL**: This module provides CORRELATION, not causation, not
prediction.  Every output description includes "requires human assessment."
Never "earthquake predicted."

Usage:
    >>> from omni_mercury_engine.detectors.cross_domain_frequency import (
    ...     CrossDomainFrequencyCorrelator,
    ... )
    >>> correlator = CrossDomainFrequencyCorrelator()
    >>> result = correlator.correlate({
    ...     "environmental": env_influence_vector,
    ...     "space": space_influence_vector,
    ... })
    >>> print(result.alert_level)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BandOverlap:
    """A single overlapping frequency band between two domains.

    Attributes:
        domain_a: First domain name.
        domain_b: Second domain name.
        band_a: Band label in domain A.
        band_b: Band label in domain B.
        overlap_low_hz: Lower bound of overlap (Hz).
        overlap_high_hz: Upper bound of overlap (Hz).
        score_a: Anomaly score in domain A band.
        score_b: Anomaly score in domain B band.
        correlation_strength: Geometric mean of the two scores.
    """

    domain_a: str
    domain_b: str
    band_a: str
    band_b: str
    overlap_low_hz: float
    overlap_high_hz: float
    score_a: float
    score_b: float
    correlation_strength: float


@dataclass
class CrossDomainCorrelation:
    """Result of cross-domain frequency correlation analysis.

    Attributes:
        correlation_score: Overall cross-domain correlation [0, 1].
        alert_level: "none", "low", "medium", or "high".
        overlapping_bands: List of detected band overlaps.
        domain_pairs_checked: Number of domain pairs analysed.
        significant_overlaps: Count of overlaps with correlation_strength > 0.3.
        description: Human-readable summary (always includes
            "requires human assessment").
    """

    correlation_score: float
    alert_level: str
    overlapping_bands: list[BandOverlap]
    domain_pairs_checked: int
    significant_overlaps: int
    description: str


@dataclass
class DomainBandInfo:
    """Extracted band information from an Oracle influence vector."""

    domain: str
    band_label: str
    low_hz: float
    high_hz: float
    anomaly_score: float
    is_significant: bool


def _extract_bands_from_influence_vector(
    domain: str, influence_vector: Any
) -> list[DomainBandInfo]:
    """Extract band-level info from a FrequencyInfluenceVector or dict."""
    bands: list[DomainBandInfo] = []

    # Handle both FrequencyInfluenceVector dataclass and dict
    band_scores: dict[str, float] = {}
    if hasattr(influence_vector, "band_scores"):
        band_scores = influence_vector.band_scores
    elif isinstance(influence_vector, dict):
        band_scores = influence_vector.get("band_scores", {})

    # We need the band definitions to get Hz ranges
    try:
        from omni_mercury_engine.detectors.spectral_domain_sound import (
            get_domain_frequency_bands,
        )

        band_defs = get_domain_frequency_bands(domain)
    except ImportError:
        band_defs = []

    # Build a lookup: band_label -> (low_hz, high_hz)
    band_hz: dict[str, tuple[float, float]] = {}
    for low, high, label, _weight in band_defs:
        band_hz[label] = (low, high)

    for label, score in band_scores.items():
        if label in band_hz:
            low_hz, high_hz = band_hz[label]
        else:
            # Fallback: can't determine Hz range
            continue
        bands.append(
            DomainBandInfo(
                domain=domain,
                band_label=label,
                low_hz=low_hz,
                high_hz=high_hz,
                anomaly_score=float(score),
                is_significant=float(score) > 0.3,
            )
        )
    return bands


def _compute_overlap(
    a_low: float, a_high: float, b_low: float, b_high: float
) -> tuple[float, float] | None:
    """Return the overlapping Hz interval, or None if no overlap."""
    overlap_low = max(a_low, b_low)
    overlap_high = min(a_high, b_high)
    if overlap_low < overlap_high:
        return (overlap_low, overlap_high)
    return None


class CrossDomainFrequencyCorrelator:
    """Detect overlapping significant frequency bands across domains.

    This module takes a dict of ``{domain: FrequencyInfluenceVector}``
    from concurrent SpectralDomainSound instances and identifies
    frequency bands that are simultaneously anomalous in multiple
    domains.

    IMPORTANT: Results indicate *correlation only*.  Every output
    description states "requires human assessment."  This module
    never claims causation or prediction.
    """

    def __init__(
        self,
        significance_threshold: float = 0.3,
        alert_thresholds: dict[str, float] | None = None,
    ) -> None:
        """Initialise the correlator.

        Args:
            significance_threshold: Minimum anomaly score in both bands
                for an overlap to be considered significant.
            alert_thresholds: Mapping of alert level to minimum
                correlation_score.  Defaults to
                ``{"low": 0.2, "medium": 0.4, "high": 0.7}``.
        """
        self.significance_threshold = significance_threshold
        self.alert_thresholds = alert_thresholds or {
            "low": 0.2,
            "medium": 0.4,
            "high": 0.7,
        }

    def correlate(
        self,
        domain_vectors: dict[str, Any],
    ) -> CrossDomainCorrelation:
        """Analyse cross-domain frequency correlations.

        Args:
            domain_vectors: Mapping of domain name to
                FrequencyInfluenceVector (or equivalent dict).

        Returns:
            CrossDomainCorrelation with alert level and overlapping bands.
        """
        domains = sorted(domain_vectors.keys())

        # Extract band info for each domain
        domain_bands: dict[str, list[DomainBandInfo]] = {}
        for domain in domains:
            iv = domain_vectors[domain]
            domain_bands[domain] = _extract_bands_from_influence_vector(domain, iv)

        overlapping_bands: list[BandOverlap] = []
        pairs_checked = 0

        # Check all domain pairs
        for i, dom_a in enumerate(domains):
            for dom_b in domains[i + 1 :]:
                pairs_checked += 1
                for band_a in domain_bands[dom_a]:
                    for band_b in domain_bands[dom_b]:
                        overlap = _compute_overlap(
                            band_a.low_hz,
                            band_a.high_hz,
                            band_b.low_hz,
                            band_b.high_hz,
                        )
                        if overlap is None:
                            continue
                        # Geometric mean of scores
                        strength = float(
                            np.sqrt(max(band_a.anomaly_score, 0) * max(band_b.anomaly_score, 0))
                        )
                        overlapping_bands.append(
                            BandOverlap(
                                domain_a=dom_a,
                                domain_b=dom_b,
                                band_a=band_a.band_label,
                                band_b=band_b.band_label,
                                overlap_low_hz=overlap[0],
                                overlap_high_hz=overlap[1],
                                score_a=band_a.anomaly_score,
                                score_b=band_b.anomaly_score,
                                correlation_strength=strength,
                            )
                        )

        significant = [
            b for b in overlapping_bands if b.correlation_strength >= self.significance_threshold
        ]

        # Overall correlation score: mean of significant overlap strengths
        if significant:
            correlation_score = float(np.mean([b.correlation_strength for b in significant]))
        else:
            correlation_score = 0.0

        # Determine alert level
        alert_level = "none"
        for level in ["high", "medium", "low"]:
            if correlation_score >= self.alert_thresholds.get(level, 1.0):
                alert_level = level
                break

        # Build description (ALWAYS includes "requires human assessment")
        if not significant:
            description = (
                f"No significant cross-domain frequency correlations detected "
                f"across {pairs_checked} domain pair(s). "
                f"Requires human assessment."
            )
        else:
            band_summary = ", ".join(
                f"{b.domain_a}/{b.band_a} <-> {b.domain_b}/{b.band_b} "
                f"({b.overlap_low_hz:.2f}-{b.overlap_high_hz:.2f} Hz, "
                f"strength={b.correlation_strength:.3f})"
                for b in significant[:5]
            )
            description = (
                f"Detected {len(significant)} significant cross-domain "
                f"frequency correlation(s) (alert={alert_level}). "
                f"Top overlaps: {band_summary}. "
                f"Correlation only — requires human assessment."
            )

        return CrossDomainCorrelation(
            correlation_score=correlation_score,
            alert_level=alert_level,
            overlapping_bands=overlapping_bands,
            domain_pairs_checked=pairs_checked,
            significant_overlaps=len(significant),
            description=description,
        )

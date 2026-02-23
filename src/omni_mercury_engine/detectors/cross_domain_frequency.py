"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

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

from __future__ import annotations

"""
Cross-Domain Frequency Correlator
==================================

Detects spectral-band overlap between concurrent Oracle instances running
on different domain streams (e.g., USGS seismic + NOAA space weather).

**CRITICAL CONSTRAINT**: This module provides *correlation*, not causation
and not prediction.  Every output includes the disclaimer:

    "Correlated spectral anomaly detected -- requires human assessment."

The Schumann Resonance / seismic precursor validation path is an
*empirical observation channel*.  Any findings (including null results)
are scientifically valuable and must be documented honestly.

References:
    - Balser, M. & Wagner, C.A. (1960). Observations of Earth-ionosphere
      cavity resonances. Nature, 188(4751), 638-641.
    - Hayakawa, M. & Molchanov, O.A. (2002). Seismo-electromagnetics:
      Lithosphere-Atmosphere-Ionosphere Coupling. TERRAPUB.
    - Schumann, W.O. (1952). On the free oscillations of a conducting
      sphere which is surrounded by an air layer and an ionosphere shell.
      Z. Naturforsch. A, 7(2), 149-154.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from omni_mercury_engine.detectors.spectral_domain_oracle import (
    DOMAIN_FREQUENCY_BANDS,
    FrequencyInfluenceVector,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Alert thresholds
# ---------------------------------------------------------------------------
CORRELATION_ALERT_THRESHOLDS: dict[str, float] = {
    "LOW": 0.3,
    "MEDIUM": 0.6,
    "HIGH": 0.8,
}

# Human-assessment disclaimer required on all outputs
_HUMAN_ASSESSMENT_DISCLAIMER = (
    "Correlated spectral anomaly detected -- requires human assessment."
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OverlappingBand:
    """A single band overlap between two domains.

    Attributes:
        band_a_label: Band label from domain A.
        band_b_label: Band label from domain B.
        overlap_low_hz: Lower bound of overlapping frequency range (Hz).
        overlap_high_hz: Upper bound of overlapping frequency range (Hz).
        score_a: Anomaly score from domain A for this band.
        score_b: Anomaly score from domain B for this band.
    """

    band_a_label: str
    band_b_label: str
    overlap_low_hz: float
    overlap_high_hz: float
    score_a: float
    score_b: float


@dataclass(frozen=True)
class CrossDomainCorrelation:
    """Result of cross-domain frequency correlation analysis.

    Attributes:
        domain_a: Name of the first domain.
        domain_b: Name of the second domain.
        overlapping_bands: List of overlapping band pairs.
        correlation_score: Overall correlation strength [0, 1].
        alert_level: "LOW", "MEDIUM", "HIGH", or "NONE".
        description: Human-readable summary with mandatory disclaimer.
        timestamp: ISO-8601 timestamp of the analysis.
        methodology: Description of the correlation method used.
    """

    domain_a: str
    domain_b: str
    overlapping_bands: list[OverlappingBand]
    correlation_score: float
    alert_level: str
    description: str
    timestamp: str = ""
    methodology: str = (
        "Spectral band overlap detection with weighted anomaly score "
        "correlation. Provides CORRELATION only -- not causation or prediction."
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "domain_a": self.domain_a,
            "domain_b": self.domain_b,
            "overlapping_bands": [
                {
                    "band_a_label": b.band_a_label,
                    "band_b_label": b.band_b_label,
                    "overlap_low_hz": b.overlap_low_hz,
                    "overlap_high_hz": b.overlap_high_hz,
                    "score_a": b.score_a,
                    "score_b": b.score_b,
                }
                for b in self.overlapping_bands
            ],
            "correlation_score": self.correlation_score,
            "alert_level": self.alert_level,
            "description": self.description,
            "timestamp": self.timestamp,
            "methodology": self.methodology,
            "disclaimer": _HUMAN_ASSESSMENT_DISCLAIMER,
        }


# ---------------------------------------------------------------------------
# Core correlator
# ---------------------------------------------------------------------------


class CrossDomainFrequencyCorrelator:
    """Detect spectral-band overlap between concurrent Oracle results.

    Given a dict of ``{domain_name: FrequencyInfluenceVector}`` from
    concurrent Oracle instances, this correlator identifies overlapping
    frequency bands where *both* domains show elevated anomaly scores.

    This enables the Schumann Resonance / seismic precursor validation
    path: the Oracle's environmental bands include
    ``schumann_fundamental`` (7.5--8.0 Hz), ``schumann_harmonic_1``
    (13.5--14.5 Hz), and ``schumann_harmonics_upper`` (20.0--26.5 Hz).
    When USGS earthquake data and NOAA space weather data simultaneously
    show anomalies in overlapping bands, the correlator flags the
    coincidence for human review.

    **This module never claims causation or prediction.**

    Usage::

        correlator = CrossDomainFrequencyCorrelator()
        results = correlator.correlate({
            "environmental": env_influence_vector,
            "space": space_influence_vector,
        })
        for result in results:
            print(result.to_dict())
    """

    def __init__(
        self,
        min_overlap_hz: float = 0.5,
        min_band_score: float = 0.3,
    ) -> None:
        """Initialise the correlator.

        Args:
            min_overlap_hz: Minimum overlap width (Hz) to consider a
                band pair as overlapping.  Prevents spurious
                micro-overlaps from dominating.
            min_band_score: Minimum anomaly score in *both* domains for
                a band pair to be flagged.
        """
        self.min_overlap_hz = min_overlap_hz
        self.min_band_score = min_band_score
        self._analysis_count = 0

    def correlate(
        self,
        oracle_results: dict[str, FrequencyInfluenceVector],
        timestamp: str = "",
    ) -> list[CrossDomainCorrelation]:
        """Run cross-domain frequency correlation.

        Args:
            oracle_results: Mapping of domain name to the
                FrequencyInfluenceVector produced by that domain's
                Oracle instance.
            timestamp: Optional ISO-8601 timestamp for the analysis.

        Returns:
            List of CrossDomainCorrelation results, one per domain pair
            with at least one overlapping anomalous band.
        """
        self._analysis_count += 1
        domains = sorted(oracle_results.keys())
        results: list[CrossDomainCorrelation] = []

        for i, domain_a in enumerate(domains):
            for domain_b in domains[i + 1 :]:
                iv_a = oracle_results[domain_a]
                iv_b = oracle_results[domain_b]

                correlation = self._correlate_pair(
                    domain_a, iv_a, domain_b, iv_b, timestamp,
                )
                if correlation is not None:
                    results.append(correlation)

        if results:
            logger.info(
                "Cross-domain analysis found %d correlated domain pair(s).",
                len(results),
            )
        else:
            logger.debug(
                "Cross-domain analysis: no significant correlations "
                "across %d domain(s).",
                len(domains),
            )

        return results

    def _correlate_pair(
        self,
        domain_a: str,
        iv_a: FrequencyInfluenceVector,
        domain_b: str,
        iv_b: FrequencyInfluenceVector,
        timestamp: str = "",
    ) -> CrossDomainCorrelation | None:
        """Correlate a single pair of domain Oracle results.

        Finds overlapping frequency bands where both domains show
        elevated anomaly scores, computes an aggregate correlation
        score, and returns a structured result.
        """
        bands_a = DOMAIN_FREQUENCY_BANDS.get(domain_a, [])
        bands_b = DOMAIN_FREQUENCY_BANDS.get(domain_b, [])

        if not bands_a or not bands_b:
            return None

        overlapping: list[OverlappingBand] = []

        for lo_a, hi_a, label_a, _w_a in bands_a:
            score_a = iv_a.band_scores.get(label_a, 0.0)
            if score_a < self.min_band_score:
                continue

            for lo_b, hi_b, label_b, _w_b in bands_b:
                score_b = iv_b.band_scores.get(label_b, 0.0)
                if score_b < self.min_band_score:
                    continue

                # Compute frequency overlap
                overlap_lo = max(lo_a, lo_b)
                overlap_hi = min(hi_a, hi_b)
                overlap_width = overlap_hi - overlap_lo

                if overlap_width >= self.min_overlap_hz:
                    overlapping.append(
                        OverlappingBand(
                            band_a_label=label_a,
                            band_b_label=label_b,
                            overlap_low_hz=overlap_lo,
                            overlap_high_hz=overlap_hi,
                            score_a=score_a,
                            score_b=score_b,
                        ),
                    )

        if not overlapping:
            return None

        # Compute aggregate correlation score
        # Weighted geometric mean of per-band pair scores
        total_weight = 0.0
        weighted_score = 0.0
        for band in overlapping:
            pair_score = np.sqrt(band.score_a * band.score_b)
            bandwidth = band.overlap_high_hz - band.overlap_low_hz
            weight = bandwidth  # wider overlaps get more weight
            weighted_score += pair_score * weight
            total_weight += weight

        correlation_score = (
            weighted_score / total_weight if total_weight > 0 else 0.0
        )
        correlation_score = float(np.clip(correlation_score, 0.0, 1.0))

        # Determine alert level
        alert_level = "NONE"
        for level in ("HIGH", "MEDIUM", "LOW"):
            if correlation_score >= CORRELATION_ALERT_THRESHOLDS[level]:
                alert_level = level
                break

        # Build description
        band_summary = ", ".join(
            f"{b.band_a_label}/{b.band_b_label} "
            f"({b.overlap_low_hz:.1f}-{b.overlap_high_hz:.1f} Hz)"
            for b in overlapping[:5]  # limit to 5 for readability
        )
        extra = f" (+{len(overlapping) - 5} more)" if len(overlapping) > 5 else ""

        description = (
            f"Cross-domain spectral correlation between {domain_a} and "
            f"{domain_b}: {len(overlapping)} overlapping band pair(s) "
            f"[{band_summary}{extra}]. "
            f"Aggregate correlation score: {correlation_score:.3f} "
            f"({alert_level}). "
            f"{_HUMAN_ASSESSMENT_DISCLAIMER}"
        )

        return CrossDomainCorrelation(
            domain_a=domain_a,
            domain_b=domain_b,
            overlapping_bands=overlapping,
            correlation_score=correlation_score,
            alert_level=alert_level,
            description=description,
            timestamp=timestamp,
        )

    def correlate_schumann_seismic(
        self,
        environmental_iv: FrequencyInfluenceVector,
        space_iv: FrequencyInfluenceVector,
        timestamp: str = "",
    ) -> CrossDomainCorrelation | None:
        """Specialised Schumann Resonance / seismic precursor check.

        Focuses specifically on the Schumann fundamental (7.83 Hz) and
        harmonics, cross-referencing environmental and space domain
        Oracle results.

        This is the primary validation path described in Part 7 of the
        Mercury strategic improvements directive.

        Args:
            environmental_iv: Oracle result from environmental domain.
            space_iv: Oracle result from space domain.
            timestamp: ISO-8601 timestamp.

        Returns:
            CrossDomainCorrelation if Schumann-band overlap detected,
            else None.
        """
        # Focus on Schumann-relevant bands
        schumann_labels = {
            "schumann_fundamental",
            "schumann_harmonic_1",
            "schumann_harmonics_upper",
            "schumann_coupling",  # space domain equivalent
        }

        # Check if any Schumann bands are anomalous in either domain
        env_schumann_active = any(
            environmental_iv.band_scores.get(label, 0.0) >= self.min_band_score
            for label in schumann_labels
        )
        space_schumann_active = any(
            space_iv.band_scores.get(label, 0.0) >= self.min_band_score
            for label in schumann_labels
        )

        if not (env_schumann_active or space_schumann_active):
            logger.debug(
                "Schumann/seismic check: no Schumann-band anomalies "
                "detected in either domain.",
            )
            return None

        return self._correlate_pair(
            "environmental",
            environmental_iv,
            "space",
            space_iv,
            timestamp,
        )

    def get_statistics(self) -> dict[str, Any]:
        """Return correlator statistics."""
        return {
            "analysis_count": self._analysis_count,
            "min_overlap_hz": self.min_overlap_hz,
            "min_band_score": self.min_band_score,
            "supported_domains": list(DOMAIN_FREQUENCY_BANDS.keys()),
        }

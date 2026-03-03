# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) Steel Security Advisors LLC
"""Anomaly geometry classifier for targeted probe cluster activation.

Classifies training data into one of four anomaly geometry regimes:
    - ``point``: Individual outliers, distribution tail anomalies
    - ``distributional``: Shifts in mean, variance, or covariance structure
    - ``collective``: Groups of correlated anomalies, pattern deviations
    - ``temporal``: Sequential ordering violations, derivative anomalies

Classification uses unsupervised heuristics on training data only.
No labels are required.

Geometry -> Probe Preset mapping:
    point           -> ``robust`` (EthicalIQR, AnnealedZScore, VarianceAdapted,
                       AdditiveHarmonic)
    distributional  -> ``distributional`` (SVD, Boltzmann, QuantumSuperposition,
                       CatalanDecay, Helix)
    collective      -> ``collective`` (Topology, FractalSimilarity, R3Recursion,
                       EnergyMinimization, AnnealedZScore)
    temporal        -> ``temporal`` (WavePropagation, AdditiveHarmonic, Zeta,
                       Lyapunov, Momentum)
    unknown         -> ``all`` (fallback: all 17 probes)

When two geometries are detected simultaneously, both clusters are merged.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

# Geometry type aliases
GeometryType = Literal["point", "distributional", "collective", "temporal", "unknown"]


# Geometry -> Probe Preset mapping
GEOMETRY_PROBE_PRESETS: dict[str, list[str]] = {
    "point": [
        "EthicalIQRProbe",
        "AnnealedZScoreProbe",
        "VarianceAdaptedProbe",
        "AdditiveHarmonicProbe",
    ],
    "distributional": [
        "SVDProjectionProbe",
        "BoltzmannCouplingProbe",
        "QuantumSuperpositionProbe",
        "CatalanDecayProbe",
        "HelixMultiplicativeProbe",
    ],
    "collective": [
        "TopologyHomologyProbe",
        "FractalSelfSimilarityProbe",
        "R3RecursionResonanceProbe",
        "EnergyMinimizationProbe",
        "AnnealedZScoreProbe",
    ],
    "temporal": [
        "WavePropagationProbe",
        "AdditiveHarmonicProbe",
        "ZetaHarmonicProbe",
        "LyapunovChaosProbe",
        "MomentumProbe",
    ],
}

# Minimum probe set: always include these regardless of geometry
MINIMUM_PROBE_SET: list[str] = [
    "EthicalIQRProbe",
    "SVDProjectionProbe",
    "VarianceAdaptedProbe",
]


def classify_geometry(
    data: npt.NDArray[np.float64],
    *,
    autocorr_threshold: float = 0.30,
    kurtosis_threshold: float = 4.0,
    cluster_density_threshold: float = 0.15,
    variance_ratio_threshold: float = 3.0,
) -> list[GeometryType]:
    """Classify the dominant anomaly geometry in training data.

    Tests applied (in order):

    1. **Temporal**: Mean absolute row-lag-1 autocorrelation > 0.30.
    2. **Point**: Max kurtosis across features > 4.0 (heavy tails).
    3. **Distributional**: Variance ratio (max / min feature variance) > 3.0.
    4. **Collective**: Fraction of samples within r=1.5*sigma of their 5-NN
       centroid > 0.85 (dense clusters present).

    Multiple geometries can be active simultaneously. Returns all detected
    types. Returns ``["unknown"]`` if no heuristic fires.

    Args:
        data: Training data of shape (n_samples, n_features).
        autocorr_threshold: Lag-1 autocorrelation threshold for TEMPORAL.
        kurtosis_threshold: Excess kurtosis threshold for POINT.
        cluster_density_threshold: Dense cluster fraction threshold
            for COLLECTIVE (fraction of samples NOT in dense cluster).
        variance_ratio_threshold: Feature variance ratio threshold
            for DISTRIBUTIONAL.

    Returns:
        List of one or more ``GeometryType`` strings. Never empty.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    n_samples, n_features = data.shape
    detected: list[GeometryType] = []

    if n_samples < 4:
        return ["unknown"]

    # -- Test 1: Temporal (sequential row structure) --
    if n_samples >= 10:
        try:
            autocorrs = []
            for j in range(min(n_features, 10)):  # cap at 10 features for speed
                col = data[:, j]
                col_std = float(np.std(col))
                if col_std < 1e-10:
                    continue
                col_norm = (col - col.mean()) / col_std
                # Lag-1 autocorrelation via dot product
                ac = float(np.dot(col_norm[:-1], col_norm[1:]) / (n_samples - 1))
                autocorrs.append(abs(ac))
            if autocorrs and float(np.mean(autocorrs)) > autocorr_threshold:
                detected.append("temporal")
                logger.debug(
                    "Geometry: TEMPORAL detected (mean |autocorr|=%.3f > %.2f)",
                    float(np.mean(autocorrs)),
                    autocorr_threshold,
                )
        except Exception as exc:
            logger.debug("Temporal geometry test failed: %s", exc)

    # -- Test 2: Point outliers (heavy tail / kurtosis) --
    try:
        from scipy import stats as scipy_stats

        kurtoses = []
        for j in range(n_features):
            col = data[:, j]
            if float(np.std(col)) < 1e-10:
                continue
            kurtoses.append(float(scipy_stats.kurtosis(col, fisher=True)))
        if kurtoses and float(np.max(kurtoses)) > kurtosis_threshold:
            detected.append("point")
            logger.debug(
                "Geometry: POINT detected (max kurtosis=%.2f > %.1f)",
                float(np.max(kurtoses)),
                kurtosis_threshold,
            )
    except Exception as exc:
        logger.debug("Point geometry test failed: %s", exc)

    # -- Test 3: Distributional shift (variance heterogeneity) --
    try:
        variances = np.var(data, axis=0)
        variances = variances[variances > 1e-10]  # exclude zero-variance features
        if len(variances) >= 2:
            ratio = float(variances.max() / variances.min())
            if ratio > variance_ratio_threshold:
                detected.append("distributional")
                logger.debug(
                    "Geometry: DISTRIBUTIONAL detected (variance ratio=%.2f > %.1f)",
                    ratio,
                    variance_ratio_threshold,
                )
    except Exception as exc:
        logger.debug("Distributional geometry test failed: %s", exc)

    # -- Test 4: Collective anomalies (dense cluster structure) --
    if n_samples >= 50:
        try:
            from scipy.spatial import cKDTree

            tree = cKDTree(data)
            k = min(5, n_samples - 1)
            dists, _ = tree.query(data, k=k + 1)
            # Average distance to 5-NN (exclude self, column 0)
            avg_nn_dists = np.mean(dists[:, 1:], axis=1)
            global_std = float(np.std(data))
            if global_std < 1e-10:
                global_std = 1.0
            # Fraction of samples NOT in a dense cluster (far from neighbors)
            isolation_fraction = float(np.mean(avg_nn_dists > 1.5 * global_std))
            if isolation_fraction < cluster_density_threshold:
                detected.append("collective")
                logger.debug(
                    "Geometry: COLLECTIVE detected (isolation_fraction=%.3f < %.2f)",
                    isolation_fraction,
                    cluster_density_threshold,
                )
        except Exception as exc:
            logger.debug("Collective geometry test failed: %s", exc)

    if not detected:
        return ["unknown"]

    return detected


def probes_for_geometries(geometries: list[GeometryType]) -> list[str]:
    """Return the union of probe lists for the detected geometry types.

    Always includes ``MINIMUM_PROBE_SET`` regardless of geometry.
    Deduplicates while preserving insertion order (Python 3.7+).

    Args:
        geometries: List of detected geometry types.

    Returns:
        Ordered list of probe class names without duplicates.
    """
    if "unknown" in geometries:
        return list(dict.fromkeys(["all"]))  # sentinel for AnomalyMathArrest

    probe_list: list[str] = list(MINIMUM_PROBE_SET)
    for geom in geometries:
        for probe_name in GEOMETRY_PROBE_PRESETS.get(geom, []):
            if probe_name not in probe_list:
                probe_list.append(probe_name)

    return probe_list

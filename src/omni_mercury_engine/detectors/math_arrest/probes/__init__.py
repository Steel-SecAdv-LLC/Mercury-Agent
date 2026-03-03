# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) Steel Security Advisors LLC
"""Equation probes for the Anomaly Math Arrest ensemble."""

from omni_mercury_engine.detectors.math_arrest.probes.additive_harmonic import (
    AdditiveHarmonicProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.annealed_zscore import (
    AnnealedZScoreProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.boltzmann_coupling import (
    BoltzmannCouplingProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.catalan_decay import (
    CatalanDecayProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.energy_minimization import (
    EnergyMinimizationProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.ethical_iqr import (
    EthicalIQRProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.fractal_similarity import (
    FractalSelfSimilarityProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.helix import (
    HelixMultiplicativeProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.lyapunov_chaos import (
    LyapunovChaosProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.momentum import (
    MomentumProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.quantum_superposition import (
    QuantumSuperpositionProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.r3_recursion import (
    R3RecursionResonanceProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.svd_projection import (
    SVDProjectionProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.topology_homology import (
    TopologyHomologyProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.variance_adapted import (
    VarianceAdaptedProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.wave_propagation import (
    WavePropagationProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.zeta_harmonic import (
    ZetaHarmonicProbe,
)

__all__ = [
    "AdditiveHarmonicProbe",
    "AnnealedZScoreProbe",
    "BoltzmannCouplingProbe",
    "CatalanDecayProbe",
    "EnergyMinimizationProbe",
    "EthicalIQRProbe",
    "FractalSelfSimilarityProbe",
    "HelixMultiplicativeProbe",
    "LyapunovChaosProbe",
    "MomentumProbe",
    "QuantumSuperpositionProbe",
    "R3RecursionResonanceProbe",
    "SVDProjectionProbe",
    "TopologyHomologyProbe",
    "VarianceAdaptedProbe",
    "WavePropagationProbe",
    "ZetaHarmonicProbe",
]

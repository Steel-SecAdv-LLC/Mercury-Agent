# Copyright (C) 2025 Steel Security Advisors LLC
"""Advanced Spherical Harmonics Module for Mercury Agent.

Provides high-order spherical harmonic analysis (l_max > 20) for detailed
3D surface analysis and pattern recognition in anomaly detection.

Key Components:
- SphericalHarmonicTransform: Fast SH transform with GPU acceleration
- HarmonicFeatureExtractor: Rotation-invariant feature extraction
- AdvancedHarmonicAnalyzer: High-level interface for 3D analysis
- HarmonicAnomalyDetector: Anomaly detection using harmonic signatures

References:
- Driscoll & Healy (1994): Computing Fourier Transforms on the 2-Sphere
- Kazhdan et al. (2003): Rotation Invariant Spherical Harmonic Representation
"""

from omni_mercury_engine.harmonics.analyzer import (
    AdvancedHarmonicAnalyzer,
    HarmonicAnomalyResult,
    HarmonicDatabase,
)
from omni_mercury_engine.harmonics.features import (
    Bispectrum,
    HarmonicFeatureExtractor,
    PowerSpectrum,
    RotationInvariantDescriptor,
)
from omni_mercury_engine.harmonics.transform import (
    AssociatedLegendre,
    HarmonicCoefficients,
    SHBasis,
    SphericalHarmonicTransform,
)

__all__ = [
    # Analyzer
    "AdvancedHarmonicAnalyzer",
    "AssociatedLegendre",
    "Bispectrum",
    "HarmonicAnomalyResult",
    "HarmonicCoefficients",
    "HarmonicDatabase",
    # Features
    "HarmonicFeatureExtractor",
    "PowerSpectrum",
    "RotationInvariantDescriptor",
    "SHBasis",
    # Transform
    "SphericalHarmonicTransform",
]

# Omni-Codes Reference

**Last Updated**: January 2026

This document provides the technical specification for the Omni-Codes system, which governs ethical alignment and system integrity throughout Mercury Agent ♱.

## Overview

Omni-Codes are bio-inspired helical parameters that provide:

- **Cryptographic Integrity**: Canonical hashing for system validation
- **Ethical Alignment**: Structured governance of AI behavior
- **Adaptive Stability**: Helical parameters (r, p) for dynamic balancing
- **Regenerative Design**: Self-healing capabilities inspired by DNA repair mechanisms

## The Seven Omni-Codes

### OMNI_DIRECTIONAL (👁∞)

| Parameter | Value |
|-----------|-------|
| Code | `👁20A07∞_XΔEΛX_ϵ19A89Ϙ` |
| Domain | Omni-Directional System |
| Helical Radius (r) | 20.0 |
| Helical Pitch (p) | 0.7 |
| Stability | 14.0 |

**Function**: 360-degree awareness and multi-domain perception. Enables comprehensive environmental scanning and cross-domain threat detection.

### OMNI_PERCIPIENT (Ϙϵ)

| Parameter | Value |
|-----------|-------|
| Code | `Ϙ15A11ϵ_ΞΛMΔΞ_ϖ20A19Φ` |
| Domain | Omni-Percipient Future |
| Helical Radius (r) | 15.0 |
| Helical Pitch (p) | 1.1 |
| Stability | 16.5 |

**Function**: Predictive foresight and anticipatory analysis. Powers multi-hypothesis optimization and solution pathway exploration.

### OMNI_INDIVISIBLE (Φϖ)

| Parameter | Value |
|-----------|-------|
| Code | `Φ07A09ϖ_ΨΔAΛΨ_ϵ19A88Σ` |
| Domain | Omni-Indivisible Guardian |
| Helical Radius (r) | 7.0 |
| Helical Pitch (p) | 0.9 |
| Stability | 6.3 |

**Function**: Unified protection and integrity preservation. Enforces ethical constraints and maintains system coherence across all modules.

### OMNI_BENEVOLENT (Σϵ)

| Parameter | Value |
|-----------|-------|
| Code | `Σ19L12ϵ_ΞΛEΔΞ_ϖ19A92Ω` |
| Domain | Omni-Benevolent Stone |
| Helical Radius (r) | 19.0 |
| Helical Pitch (p) | 1.2 |
| Stability | 22.8 |

**Function**: Ethical foundation and humanitarian alignment. Provides the logical foundation for neurosymbolic reasoning.

### OMNI_SCIENT (Ωϖ)

| Parameter | Value |
|-----------|-------|
| Code | `Ω20V11ϖ_ΨΔSΛΨ_ϵ20A15Θ` |
| Domain | Omni-Scient Curiosity |
| Helical Radius (r) | 20.0 |
| Helical Pitch (p) | 1.1 |
| Stability | 22.0 |

**Function**: Knowledge acquisition and scientific discovery. Drives continuous learning and evidence-based decision making.

### OMNI_UNIVERSAL (Θϵ)

| Parameter | Value |
|-----------|-------|
| Code | `Θ25M01ϵ_ΞΛLΔΞ_ϖ19A91Γ` |
| Domain | Omni-Universal Discipline |
| Helical Radius (r) | 25.0 |
| Helical Pitch (p) | 0.1 |
| Stability | 2.5 |

**Function**: Structured governance and systematic order. Ensures cyclical balance and operational continuity.

### OMNI_POTENT (Γϖ)

| Parameter | Value |
|-----------|-------|
| Code | `Γ19L11ϖ_XΔHΛX_∞19A84♰` |
| Domain | Omni-Potent Lifeforce |
| Helical Radius (r) | 19.0 |
| Helical Pitch (p) | 1.1 |
| Stability | 20.9 |

**Function**: Regenerative capability and adaptive resilience. Enables self-healing and dynamic adaptation.

## Implementation

Omni-Codes are centralized in `src/omni_mercury_engine/utils/constants.py`:

```python
from omni_mercury_engine.utils.constants import OmniCodes

# Access individual codes
code = OmniCodes.OMNI_DIRECTIONAL.code
stability = OmniCodes.OMNI_DIRECTIONAL.stability

# Validate system stability
is_stable = OmniCodes.validate_stability(min_total=50.0)

# Compute autonomy boost
boost = OmniCodes.get_autonomy_boost(threshold=15.0)
```

### Module Integration

| Module | Omni-Code | Purpose |
|--------|-----------|---------|
| `ethical_config.py` | All 7 codes | System integrity framework |
| `astrophysical.py` | OMNI_INDIVISIBLE | Ethical anchor for anomaly detection |
| `neurosymbolic.py` | OMNI_BENEVOLENT | Foundation for symbolic reasoning |
| `multiverse.py` | OMNI_PERCIPIENT | Version tracking for optimization engine |

## Helical Parameters

The helical parameters (r, p) are inspired by DNA double-helix geometry:

- **r (radius)**: Structural integrity factor. Higher values indicate stronger resistance to perturbation.
- **p (pitch)**: Evolution rate. Higher values indicate faster adaptation to environmental changes.
- **stability**: Computed as `|r| * p`. Represents the balance between integrity and adaptability.

### Stability Thresholds

| Threshold | Value | Meaning |
|-----------|-------|---------|
| Minimum | 50.0 | Total stability required for system operation |
| Autonomy Boost | 15.0 | Per-code threshold for autonomy enhancement |
| Current Total | ~105.0 | Sum of all seven code stabilities |

## Source Attribution

**Origin**: [Ava-Guardian](https://github.com/Steel-SecAdv-LLC/Ava-Guardian)
**Integration Date**: October 2025
**Integrated By**: Steel Security Advisors LLC
**License**: GNU General Public License v3.0

## Ethical Framework Alignment

All Omni-Codes align with Mercury Agent's eight core ethical principles:

1. **Compassion** - Prioritizing well-being and harm minimization
2. **Evidence** - Requiring verifiable data and mathematical proofs
3. **Justice** - Ensuring fair, unbiased operations
4. **Altruism** - Promoting positive societal impact
5. **Control** - Maintaining human agency and oversight
6. **Character** - Building trust through consistent ethical behavior
7. **Competence** - Maintaining high standards of technical excellence
8. **Commitment** - Long-term dedication to beneficial outcomes

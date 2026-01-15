# Mercury Agent ♱ - Deprecation Guide

This document tracks deprecated modules, classes, methods, and parameters in Mercury Agent ♱. All deprecated items are scheduled for removal in version **2.0** unless otherwise noted.

---

## Quick Reference

| Deprecated | Replacement | Removal |
|------------|-------------|---------|
| `core.self_healing` | `resilience.self_healing` | v2.0 |
| `core.neurosymbolic_engine` | `core.code_analysis` / `models.neurosymbolic` | v2.0 |
| Legacy scalar names (38) | Omni-prefixed names | v2.0 |
| `sigma_immutable` param | `ethical_compliance_threshold` | v2.0 |
| `lambda_lyapunov` param | `convergence_rate` | v2.0 |
| `enable_quantum_terms` | `enable_optimization_terms` | v2.0 |

---

## Suppressing Deprecation Warnings

Set the environment variable to suppress warnings during migration:

```bash
export MERCURY_AGENT_SUPPRESS_DEPRECATION_WARNINGS=1
```

---

## 1. Deprecated Modules

### 1.1 `core/self_healing.py`

**Status:** Fully deprecated compatibility shim

**Before:**
```python
from omni_mercury_engine.core.self_healing import (
    SelfHealingEngine,
    AdaptiveDefenseSystem,
    CRISPRInspiredSelfHealing,
)
```

**After:**
```python
from omni_mercury_engine.resilience.self_healing import (
    SelfHealingEngine,
    AdaptiveDefenseSystem,
)
# Note: CRISPRInspiredSelfHealing is an alias for AdaptiveDefenseSystem
```

---

### 1.2 `core/neurosymbolic_engine.py`

**Status:** Fully deprecated compatibility shim

**Before:**
```python
from omni_mercury_engine.core.neurosymbolic_engine import (
    CodeAnalysisEngine,
    NeurosymbolicEngine,
)
```

**After:**
```python
# For AST-based code analysis:
from omni_mercury_engine.core.code_analysis import CodeAnalysisEngine

# For LTN-based anomaly detection:
from omni_mercury_engine.models.neurosymbolic import NeurosymbolicEngine
```

---

## 2. Deprecated Scalar Names

The Global Omni-Scalar Network (GOSNN) is transitioning to unified naming with omni-prefixes.

### Ethical Scalars

| Legacy Name | New Name |
|-------------|----------|
| `morality_scalar` | `omnimorality` |
| `empathy_scalar` | `omniempathy` |
| `compassion_scalar` | `omnicompassion` |
| `forgiveness` | `omniforgiveness` |
| `love_scalar` | `omnilove` |
| `determination_scalar` | `omnidetermination` |
| `loyalty_scalar` | `omniloyalty` |
| `integrity_scalar` | `omniintegrity` |
| `wisdom_scalar` | `omniwisdom` |
| `justice_scalar` | `omnijustice` |
| `altruism_scalar` | `omnialtruism` |
| `hope_scalar` | `omnihope` |
| `courage_scalar` | `omnicourage` |
| `accountability_scalar` | `omniaccountability` |
| `transparency_weight` | `omnitransparency` |
| `explainability_factor` | `omniexplainability` |
| `benevolence` | `omnibenevolence` |
| `equity` | `omniequity` |

### Cosmic Scalars

| Legacy Name | New Name |
|-------------|----------|
| `universe_adapt` | `omniuniverse_adapt` |
| `telos_scalar` | `omnitelos` |
| `black_hole_entropy_eth` | `omni_black_hole_entropy` |
| `harmonic_singularity_bridge` | `omni_harmonic_singularity` |
| `golden_ratio_phi` | `omni_golden_ratio_phi` |

### Quantum Scalars

| Legacy Name | New Name |
|-------------|----------|
| `quantum_weight` | `omniquantum_weight` |
| `entanglement_risk` | `omnientanglement_risk` |
| `quantum_entanglement_weight` | `omniquantum_entanglement` |
| `neuro_quantum` | `omnineuroquantum` |
| `consciousness_coherence` | `omniconsciousness_coherence` |

### Humanitarian Scalars

| Legacy Name | New Name |
|-------------|----------|
| `crisis_response_boost` | `omnicrisis_response` |
| `disaster_response_boost` | `omnidisaster_response` |
| `pandemic_monitoring` | `omnipandemic_monitoring` |
| `missing_persons_priority` | `omnimissing_persons_priority` |
| `medical_discovery_boost` | `omnimedical_discovery` |

### Security Scalars

| Legacy Name | New Name |
|-------------|----------|
| `threat_detection_sensitivity` | `omnithreat_detection` |
| `quantum_resistance` | `omniquantum_resistance` |
| `encryption_strength` | `omniencryption_strength` |
| `audit_compliance` | `omniaudit_compliance` |

**Migration:**
```python
# Before:
value = gosnn.get_scalar("morality_scalar")

# After:
value = gosnn.get_scalar("omnimorality")
```

---

## 3. Deprecated Parameters

### 3.1 AnomalyFusionEquation (3R Mechanism)

| Deprecated Parameter | Replacement |
|---------------------|-------------|
| `sigma_immutable` | `ethical_compliance_threshold` |
| `lambda_lyapunov` | `convergence_rate` |
| `sigma_immutable_override` | `ethical_threshold_override` |

**Before:**
```python
aafe = AnomalyFusionEquation(
    sigma_immutable=0.93,
    lambda_lyapunov=0.25,
)
result = aafe.compute(
    recursion_score=0.8,
    resonance_score=0.85,
    optimization_score=0.9,
    sigma_immutable_override=0.95,
)
```

**After:**
```python
aafe = AnomalyFusionEquation(
    ethical_compliance_threshold=0.93,
    convergence_rate=0.25,
)
result = aafe.compute(
    recursion_score=0.8,
    resonance_score=0.85,
    optimization_score=0.9,
    ethical_threshold_override=0.95,
)
```

---

### 3.2 EvolutionConfig (Double Helix Engine)

| Deprecated Property | Replacement |
|--------------------|-------------|
| `enable_quantum_terms` | `enable_optimization_terms` |

**Before:**
```python
config = EvolutionConfig(enable_quantum_terms=True)
```

**After:**
```python
config = EvolutionConfig(enable_optimization_terms=True)
```

---

## 4. Deprecated Methods

### 4.1 SelfHealingEngine

| Deprecated Method | Replacement |
|------------------|-------------|
| `save_signature_library(filepath)` | `save_library(filepath)` |
| `load_signature_library(filepath)` | `load_library(filepath)` |

---

## 5. Auto-Deprecated Features

### 5.1 Low-Precision Indicators

The `IndicatorSystem` automatically deprecates indicators with:
- Precision < 0.3 (> 70% false positive rate)
- Trigger count >= 10

These indicators receive `IndicatorStatus.DEPRECATED` status.

To disable auto-deprecation:
```python
indicator_system = IndicatorSystem(enable_auto_deprecation=False)
```

---

## Migration Timeline

| Version | Action |
|---------|--------|
| 1.x | Deprecated items emit warnings but remain functional |
| 2.0 | Deprecated modules and aliases will be removed |
| 2.0+ | Only new API patterns supported |

---

## Need Help?

If you encounter migration issues:

1. Check this guide for the replacement pattern
2. Search the codebase for updated usage examples
3. Open an issue at https://github.com/Steel-SecAdv-LLC/Mercury-Agent/issues

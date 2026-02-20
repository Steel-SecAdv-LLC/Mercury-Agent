# Mercury Agent - Deprecation and Migration Guide

This document tracks deprecated modules, classes, methods, and parameters in Mercury Agent.

## Policy Statement

**PRESERVATION PRINCIPLE:** Deprecated items remain functional indefinitely. No module, class, method, or parameter will be removed without documented justification meeting the following criteria:

### Removal Justification Requirements

For any item to be considered for removal, ALL of the following must be documented:

1. **Security Vulnerability**: The deprecated code introduces a security risk that cannot be mitigated
2. **Fundamental Incompatibility**: Continued support creates architectural conflicts with core functionality
3. **Zero Active Usage**: Telemetry confirms no production usage for 24+ months
4. **Migration Complete**: All known integrations have transitioned to replacement APIs
5. **Community Consensus**: RFC process with minimum 90-day comment period completed

Until these criteria are met, deprecated items operate via compatibility shims that:
- Emit warnings (suppressible via environment variable)
- Route to current implementations
- Maintain full backward compatibility

---

## Quick Reference

| Deprecated | Replacement | Status |
|------------|-------------|--------|
| `core.self_healing` | `resilience.self_healing` | **Preserved** - Compatibility shim active |
| `core.neurosymbolic_engine` | `core.code_analysis` / `models.neurosymbolic` | **Preserved** - Compatibility shim active |
| Legacy scalar names (38) | Omni-prefixed names | **Preserved** - Both naming conventions supported |
| `sigma_immutable` param | `ethical_compliance_threshold` | **Preserved** - Alias mapping active |
| `lambda_lyapunov` param | `convergence_rate` | **Preserved** - Alias mapping active |
| `enable_quantum_terms` | `enable_optimization_terms` | **Preserved** - Alias mapping active |
| `AnomalyFusionEquation` / `AAFE*` | `OmniAvaEquation` / `OAE*` | **Preserved** - Alias mapping active |

---

## Suppressing Deprecation Warnings

Set the environment variable to suppress warnings during migration:

```bash
export MERCURY_AGENT_SUPPRESS_DEPRECATION_WARNINGS=1
```

---

## 1. Module Compatibility Shims

### 1.1 `core/self_healing.py`

**Status:** Active compatibility shim - routes to `resilience.self_healing`

**Original import (continues to work):**
```python
from omni_mercury_engine.core.self_healing import (
    SelfHealingEngine,
    AdaptiveDefenseSystem,
    CRISPRInspiredSelfHealing,
)
```

**Recommended import (preferred for new code):**
```python
from omni_mercury_engine.resilience.self_healing import (
    SelfHealingEngine,
    AdaptiveDefenseSystem,
)
# Note: CRISPRInspiredSelfHealing is an alias for AdaptiveDefenseSystem
```

**Justification for preservation:**
- `core.self_healing` is used in existing integrations and documentation
- The compatibility shim adds negligible overhead (single import redirect)
- No security concerns with maintaining the alias

---

### 1.2 `core/neurosymbolic_engine.py`

**Status:** Active compatibility shim - routes to specialized modules

**Original import (continues to work):**
```python
from omni_mercury_engine.core.neurosymbolic_engine import (
    CodeAnalysisEngine,
    NeurosymbolicEngine,
)
```

**Recommended import (preferred for new code):**
```python
# For AST-based code analysis:
from omni_mercury_engine.core.code_analysis import CodeAnalysisEngine

# For LTN-based anomaly detection:
from omni_mercury_engine.models.neurosymbolic import NeurosymbolicEngine
```

**Justification for preservation:**
- Module refactoring separated concerns but original API should remain accessible
- Research references and papers cite the original import paths
- Compatibility shim maintains semantic equivalence

---

## 2. Scalar Name Aliases (Dual Support)

The Global Omni-Scalar Network (GOSNN) supports **both** legacy and omni-prefixed naming conventions. No migration is required, but new code should prefer the omni-prefixed names.

### Implementation

The GOSNN maintains a bidirectional alias registry:

```python
# Both of these return the same value:
value = gosnn.get_scalar("morality_scalar")      # Legacy name
value = gosnn.get_scalar("omnimorality")         # New name

# Both of these set the same scalar:
gosnn.set_scalar("morality_scalar", 0.95)        # Legacy name
gosnn.set_scalar("omnimorality", 0.95)           # New name
```

### Ethical Scalars

| Legacy Name | Omni Name | Status |
|-------------|-----------|--------|
| `morality_scalar` | `omnimorality` | Both supported |
| `empathy_scalar` | `omniempathy` | Both supported |
| `compassion_scalar` | `omnicompassion` | Both supported |
| `forgiveness` | `omniforgiveness` | Both supported |
| `love_scalar` | `omnilove` | Both supported |
| `determination_scalar` | `omnidetermination` | Both supported |
| `loyalty_scalar` | `omniloyalty` | Both supported |
| `integrity_scalar` | `omniintegrity` | Both supported |
| `wisdom_scalar` | `omniwisdom` | Both supported |
| `justice_scalar` | `omnijustice` | Both supported |
| `altruism_scalar` | `omnialtruism` | Both supported |
| `hope_scalar` | `omnihope` | Both supported |
| `courage_scalar` | `omnicourage` | Both supported |
| `accountability_scalar` | `omniaccountability` | Both supported |
| `transparency_weight` | `omnitransparency` | Both supported |
| `explainability_factor` | `omniexplainability` | Both supported |
| `benevolence` | `omnibenevolence` | Both supported |
| `equity` | `omniequity` | Both supported |

### Cosmic Scalars

| Legacy Name | Omni Name | Status |
|-------------|-----------|--------|
| `universe_adapt` | `omniuniverse_adapt` | Both supported |
| `telos_scalar` | `omnitelos` | Both supported |
| `black_hole_entropy_eth` | `omni_black_hole_entropy` | Both supported |
| `harmonic_singularity_bridge` | `omni_harmonic_singularity` | Both supported |
| `golden_ratio_phi` | `omni_golden_ratio_phi` | Both supported |

### Quantum Scalars

| Legacy Name | Omni Name | Status |
|-------------|-----------|--------|
| `quantum_weight` | `omniquantum_weight` | Both supported |
| `entanglement_risk` | `omnientanglement_risk` | Both supported |
| `quantum_entanglement_weight` | `omniquantum_entanglement` | Both supported |
| `neuro_quantum` | `omnineuroquantum` | Both supported |
| `consciousness_coherence` | `omniconsciousness_coherence` | Both supported |

### Humanitarian Scalars

| Legacy Name | Omni Name | Status |
|-------------|-----------|--------|
| `crisis_response_boost` | `omnicrisis_response` | Both supported |
| `disaster_response_boost` | `omnidisaster_response` | Both supported |
| `pandemic_monitoring` | `omnipandemic_monitoring` | Both supported |
| `missing_persons_priority` | `omnimissing_persons_priority` | Both supported |
| `medical_discovery_boost` | `omnimedical_discovery` | Both supported |

### Security Scalars

| Legacy Name | Omni Name | Status |
|-------------|-----------|--------|
| `threat_detection_sensitivity` | `omnithreat_detection` | Both supported |
| `quantum_resistance` | `omniquantum_resistance` | Both supported |
| `encryption_strength` | `omniencryption_strength` | Both supported |
| `audit_compliance` | `omniaudit_compliance` | Both supported |

---

## 3. Parameter Aliases

### 3.1 OmniAvaEquation (3R Mechanism)

**Status:** Alias mapping active - both parameter names accepted

| Legacy Parameter | Preferred Parameter | Status |
|-----------------|---------------------|--------|
| `sigma_immutable` | `ethical_compliance_threshold` | Both supported |
| `lambda_lyapunov` | `convergence_rate` | Both supported |
| `sigma_immutable_override` | `ethical_threshold_override` | Both supported |

**Legacy class name (continues to work):**
```python
from omni_mercury_engine.core.three_r.fusion import AnomalyFusionEquation
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

**Preferred usage (recommended for new code):**
```python
from omni_mercury_engine.core.three_r.fusion import OmniAvaEquation
oae = OmniAvaEquation(
    ethical_compliance_threshold=0.93,
    convergence_rate=0.25,
)
result = oae.compute(
    recursion_score=0.8,
    resonance_score=0.85,
    optimization_score=0.9,
    ethical_threshold_override=0.95,
)
```

---

### 3.2 EvolutionConfig (Double Helix Engine)

**Status:** Alias mapping active - both property names accepted

| Legacy Property | Preferred Property | Status |
|----------------|-------------------|--------|
| `enable_quantum_terms` | `enable_optimization_terms` | Both supported |

**Original usage (continues to work):**
```python
config = EvolutionConfig(enable_quantum_terms=True)
```

**Preferred usage (recommended for new code):**
```python
config = EvolutionConfig(enable_optimization_terms=True)
```

---

## 4. Method Aliases

### 4.1 SelfHealingEngine

**Status:** Method aliases active - both method names work identically

| Legacy Method | Preferred Method | Status |
|--------------|------------------|--------|
| `save_signature_library(filepath)` | `save_library(filepath)` | Both supported |
| `load_signature_library(filepath)` | `load_library(filepath)` | Both supported |

---

## 5. Auto-Deprecated Features

### 5.1 Low-Precision Indicators

The `IndicatorSystem` automatically deprecates indicators with:
- Precision < 0.3 (> 70% false positive rate)
- Trigger count >= 10

These indicators receive `IndicatorStatus.DEPRECATED` status.

**Note:** This is a runtime performance optimization, not a code removal. Deprecated indicators remain in the system but are weighted lower in fusion calculations.

To disable auto-deprecation:
```python
indicator_system = IndicatorSystem(enable_auto_deprecation=False)
```

---

## Support Policy

| Version | Commitment |
|---------|------------|
| 1.x | All items fully functional with optional warnings |
| 2.x | All items fully functional with optional warnings |
| Future | Removal only with documented justification per policy |

---

## Contributing

If you believe an item should be considered for removal, submit an RFC with:

1. Security analysis
2. Usage impact assessment
3. Migration path documentation
4. Proposed timeline (minimum 12 months)

Submit RFCs at: https://github.com/Steel-SecAdv-LLC/Mercury-Agent/issues

---

## Need Help?

If you encounter issues:

1. Check this guide for the correct usage pattern
2. Search the codebase for usage examples
3. Open an issue at https://github.com/Steel-SecAdv-LLC/Mercury-Agent/issues

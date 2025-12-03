# OMNI AVA Architecture

## Overview

OMNI AVA (O♱A) is a comprehensive multi-domain anomaly detection and intelligence fusion system. This document describes the unified architecture after the consolidation effort.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           OMNI AVA Architecture                               │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐      │
│  │ Statistical │   │  Temporal   │   │   Spatial   │   │ Dimensional │      │
│  │  Detector   │   │  Detector   │   │  Detector   │   │  Analyzer   │      │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘      │
│         │                 │                 │                 │              │
│         └────────────┬────┴────────┬────────┴────────┬────────┘              │
│                      ▼             ▼                 ▼                       │
│              ┌───────────────────────────────────────────────┐               │
│              │           DetectorRegistry Bridge              │               │
│              │     (50+ detectors, parallel execution)        │               │
│              └────────────────────┬──────────────────────────┘               │
│                                   │                                          │
│                                   ▼                                          │
│              ┌───────────────────────────────────────────────┐               │
│              │         Feature Extraction Layer               │               │
│              │   (128D per detector → aggregated tensors)     │               │
│              └────────────────────┬──────────────────────────┘               │
│                                   │                                          │
│                                   ▼                                          │
│              ┌───────────────────────────────────────────────┐               │
│              │          HybridFusionLayer (Attention)         │               │
│              │  Early Fusion + Late Fusion + Multi-Head Attn  │               │
│              └────────────────────┬──────────────────────────┘               │
│                                   │                                          │
│                                   ▼                                          │
│              ┌───────────────────────────────────────────────┐               │
│              │            OmniAvaEngine (Main)                │               │
│              │   Unified detection, fusion, and inference     │               │
│              └───────────────────────────────────────────────┘               │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. OmniAvaEngine (`engine.py`)

The main orchestration engine that integrates all detection capabilities:

```python
from omni_anomaly_engine.engine import OmniAvaEngine

engine = OmniAvaEngine(mode="fusion", device="cuda")
result = engine.detect_with_fusion(data)
```

**Features:**
- Manages 5 base detectors and 13+ specialized models
- Parallel feature extraction with thread pooling
- LRU feature caching for performance
- Memory monitoring and automatic garbage collection
- Batch processing for large datasets

### 2. DetectorRegistry (`core/detector_registry.py`)

Central bridge connecting 50+ detectors for unified access:

```python
from omni_anomaly_engine.core.detector_registry import DetectorRegistry

registry = DetectorRegistry(auto_discover=True)
features = registry.extract_all_features(data, parallel=True)
aggregated = registry.aggregate_features(features, target_dim=128)
```

**Detector Categories:**
- `BASE`: Statistical, Temporal, Spatial, Dimensional, Directive, Graph-based
- `MODEL`: Quantum, Astrophysical, Biometric, Affective, Neural, etc.
- `SECURITY`: Threat Detection, TEMPEST, PSYOP
- `INTELLIGENCE`: Intelligence Fusion, All-Source Intel
- `SPACE`: Solar Storm, Schumann Resonance, Disaster Precursor
- `MEDICAL`: Sepsis, Cardiology, Neurocritical Care
- `GEOLOGICAL`: Volcanic, Landslide, Wildfire
- `ECONOMIC`: Financial Crisis, Fraud Detection
- `ENERGY`: EMP Detection
- `MARINE`: Biodiversity Threat Detection

### 3. Neurosymbolic Engine (`models/neurosymbolic.py`)

Unified LTN-based neurosymbolic reasoning for explainable AI:

```python
from omni_anomaly_engine.models.neurosymbolic import (
    NeurosymbolicEngine,
    ReasoningMode,
)

engine = NeurosymbolicEngine(
    input_dim=64,
    reasoning_mode=ReasoningMode.HYBRID,
)

# Neural inference
neural_score = engine.neural_inference(features)

# Symbolic inference with explanation
result = engine.symbolic_inference("priority_high")
print(result["explanation"])

# Hybrid neuro-symbolic inference
hybrid_result = engine.hybrid_inference(data, context={"threat_score": 0.8})
explanation = engine.explain(hybrid_result)
```

**Architecture:**
- `LogicTensorNetwork`: PyTorch neural network with fuzzy logic
- `SymbolicReasoningLayer`: PyReason-inspired rule-based reasoning
- `NeurosymbolicEngine`: Unified interface combining both

### 4. Self-Healing Engine (`resilience/self_healing.py`)

CRISPR-inspired adaptive defense with component health monitoring:

```python
from omni_anomaly_engine.resilience.self_healing import SelfHealingEngine

healer = SelfHealingEngine(max_signatures=1000)

# Learn new anomaly pattern
signature = healer.learn_anomaly(anomaly_data, metadata={"source": "sensor_1"})

# Check for known patterns (Stage 3: Interference)
is_known, confidence, sig_id = healer.check_known_anomaly(new_data)

# Component health monitoring
healer.register_component("database", health_check_fn, recovery_action_fn)
health = healer.get_system_health()
```

**CRISPR-Inspired 3-Stage Defense:**
1. **Acquisition**: Capture novel anomaly signatures
2. **Expression**: Process signatures into detection patterns
3. **Interference**: Neutralize/block matching anomalies

### 5. PSYOP Analyzer (`security/psyop.py`)

Psychological operations analysis for All-Source Intelligence:

```python
from omni_anomaly_engine.security.psyop import PSYOPAnalyzer

analyzer = PSYOPAnalyzer()

# Analyze narrative/message
narrative = analyzer.analyze_narrative(narrative_data)

# Detect influence campaign
campaign = analyzer.detect_influence_campaign(campaign_data)

# Assess information environment
environment = analyzer.assess_information_environment(env_data)

# Integration with fusion pipeline
features = analyzer.extract_features(data)
prediction = analyzer.predict(data)
```

## Feature Pipeline

The feature pipeline transforms raw data into fused anomaly predictions:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Feature Pipeline Flow                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Raw Data                                                                  │
│      │                                                                      │
│      ▼                                                                      │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │                  DetectorRegistry.extract_all_features()          │     │
│   │                                                                   │     │
│   │  ┌──────────┐  ┌──────────┐  ┌──────────┐       ┌──────────┐    │     │
│   │  │Detector 1│  │Detector 2│  │Detector 3│  ...  │Detector N│    │     │
│   │  │ features │  │ features │  │ features │       │ features │    │     │
│   │  └────┬─────┘  └────┬─────┘  └────┬─────┘       └────┬─────┘    │     │
│   │       │             │             │                   │          │     │
│   │       └─────────────┴─────────────┴───────────────────┘          │     │
│   └───────────────────────────┬──────────────────────────────────────┘     │
│                               │                                             │
│                               ▼                                             │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │              DetectorRegistry.aggregate_features()                │     │
│   │                                                                   │     │
│   │    Per-detector features → Normalized to 128D → Tensor Dict      │     │
│   │                                                                   │     │
│   │    detector_features = {                                         │     │
│   │        "statistical": [batch, 128],                              │     │
│   │        "temporal": [batch, 128],                                 │     │
│   │        "quantum": [batch, 128],                                  │     │
│   │        ...                                                        │     │
│   │    }                                                              │     │
│   └───────────────────────────┬──────────────────────────────────────┘     │
│                               │                                             │
│                               ▼                                             │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │                 HybridFusionLayer.forward()                       │     │
│   │                                                                   │     │
│   │  1. Feature Projection: 128D → hidden_dim (128) per detector     │     │
│   │                                                                   │     │
│   │  2. Early Fusion: Concatenate all → MLP → 128D                   │     │
│   │                                                                   │     │
│   │  3. Late Fusion: Weighted average of detector scores             │     │
│   │                                                                   │     │
│   │  4. Attention Fusion: Multi-head attention over detector embeds  │     │
│   │                                                                   │     │
│   │  Output: fused_representation [batch, hidden_dim]                │     │
│   └───────────────────────────┬──────────────────────────────────────┘     │
│                               │                                             │
│                               ▼                                             │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │                   OmniFusionModel.forward()                       │     │
│   │                                                                   │     │
│   │  fused_features → classification_head → anomaly_probs            │     │
│   │                 → severity_head → severity_scores                 │     │
│   │                 → class_head → class_predictions                  │     │
│   └───────────────────────────┬──────────────────────────────────────┘     │
│                               │                                             │
│                               ▼                                             │
│   Final Output:                                                             │
│   {                                                                         │
│       "anomaly_prob": 0.87,                                                │
│       "is_anomaly": True,                                                  │
│       "severity": 0.72,                                                    │
│       "class_prediction": 3,                                               │
│       "detector_importance": {"temporal": 0.3, "quantum": 0.2, ...}       │
│   }                                                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Design Patterns

### 1. Protocol-Based Interface

All detectors follow a common interface:

```python
class DetectorProtocol(Protocol):
    def extract_features(self, data: Any) -> Any:
        """Extract features from input data."""
        ...

    def predict(self, data: Any) -> dict[str, Any]:
        """Make predictions on input data."""
        ...
```

### 2. Hybrid Fusion Strategy

The fusion layer combines multiple strategies:

1. **Early Fusion**: Concatenate features before processing
2. **Late Fusion**: Combine detector scores with learned weights
3. **Attention Fusion**: Learn which detectors are most relevant

### 3. CRISPR-Inspired Defense

The self-healing system uses biological metaphors:

- **Signature Library**: Memory of known threats
- **Acquisition**: Learn new threat patterns
- **Interference**: Block recognized threats

### 4. Neurosymbolic Reasoning

Combines neural networks with symbolic logic:

- **Neural**: LTN for pattern recognition
- **Symbolic**: Rule-based knowledge base
- **Hybrid**: Weighted combination with explainability

## Module Organization

```
omni_anomaly_engine/
├── engine.py                    # Main OmniAvaEngine orchestrator
├── core/
│   ├── base.py                  # BaseDetector, BaseModel abstracts
│   ├── config.py                # Engine configuration
│   ├── detector_registry.py     # Unified detector bridge
│   ├── fusion.py                # HybridFusionLayer, AttentionFusion
│   ├── code_analysis.py         # AST-based code analysis (renamed)
│   └── ...
├── models/
│   ├── neurosymbolic.py         # LTN-based neurosymbolic (primary)
│   ├── quantum.py               # Quantum anomaly detection
│   ├── biometric.py             # Biometric analysis
│   └── ...
├── detectors/
│   ├── statistical.py           # Statistical detector
│   ├── temporal.py              # Temporal detector
│   ├── spatial.py               # Spatial detector
│   └── ...
├── security/
│   ├── psyop.py                 # PSYOP analysis
│   ├── intelligence_fusion.py   # All-source intel fusion
│   ├── threat_detection.py      # Threat detection
│   └── ...
├── resilience/
│   ├── self_healing.py          # CRISPR-inspired defense
│   ├── circuit_breaker.py       # Fault tolerance
│   └── ...
└── ml/
    ├── fusion_network.py        # OmniFusionModel
    ├── training.py              # Training utilities
    └── inference.py             # Inference engine
```

## Configuration

### Engine Configuration

```python
from omni_anomaly_engine.core.config import EngineConfig

config = EngineConfig(
    threshold=0.5,
    enable_caching=True,
    cache_size=128,
    memory_threshold_mb=2048,
)

engine = OmniAvaEngine(config=config, mode="fusion", device="cuda")
```

### Detector Registry Configuration

```python
registry = DetectorRegistry(
    max_workers=8,           # Parallel workers
    timeout_seconds=30.0,    # Detector timeout
    auto_discover=True,      # Auto-register detectors
)
```

## Integration Points

### 1. With External Intelligence Sources

```python
from omni_anomaly_engine.security.int_sources import IntelligenceSourceRegistry

intel_registry = IntelligenceSourceRegistry()
sources = intel_registry.get_by_discipline("SIGINT")
```

### 2. With PSYOP Analysis

```python
from omni_anomaly_engine.security.psyop import PSYOPAnalyzer

psyop = PSYOPAnalyzer()
features = psyop.extract_features(data)  # For fusion pipeline
result = psyop.predict(data)             # For direct analysis
```

### 3. With Self-Healing System

```python
from omni_anomaly_engine.resilience.self_healing import SelfHealingEngine

healer = SelfHealingEngine()
engine.self_healing = healer

# Automatic anomaly learning
if result["is_anomaly"]:
    healer.learn_anomaly(data)
```

## Performance Considerations

1. **Parallel Execution**: DetectorRegistry uses ThreadPoolExecutor for parallel feature extraction
2. **Feature Caching**: LRU cache prevents redundant computations
3. **Memory Management**: Automatic GC when memory threshold exceeded
4. **Batch Processing**: Optimal batch sizes calculated based on memory

## Extensibility

### Adding New Detectors

1. Implement `extract_features()` and `predict()` methods
2. Register with DetectorRegistry:

```python
registry.register(
    name="my_detector",
    detector=MyDetector(),
    category=DetectorCategory.MODEL,
    description="My custom detector",
)
```

### Adding New Rules

```python
from omni_anomaly_engine.models.neurosymbolic import SymbolicRule

engine.add_rule(SymbolicRule(
    name="custom_rule",
    premise="condition_a AND condition_b",
    conclusion="custom_conclusion",
    confidence=0.9,
))
```

## Security Considerations

- All external inputs validated before processing
- PSYOP analysis includes bias detection
- Ethical constraints enforced via symbolic rules
- No sensitive data logged by default

## Version History

- **v1.0.0**: Initial consolidation with unified OmniAvaEngine
- Merged duplicate implementations (kept Location 2)
- Unified neurosymbolic architecture (LTN-based)
- Added DetectorRegistry bridge
- Added PSYOP analysis module
- Added CRISPR-inspired self-healing

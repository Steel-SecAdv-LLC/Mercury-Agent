# Step 6 — Coverage delta

Omni HEAD: `2a3c6dd9d7035e9fef39223ffb371af11cf0e0a3`  Mercury HEAD: `7af7837612008e86afe91d54a534e9a18b9e3804`

Mercury's coverage data could not be regenerated in the audit container (`pytest --cov` requires the full Mercury dependency tree: torch, sklearn, pyod, deepface, etc., which were not installed). Per Step 6 'best-effort' rules, only the Omni side of the delta is reported here; the Mercury column is marked **UNVERIFIED**.

## Omni files with high test coverage (>80%) where Mercury counterpart is missing or unknown

| omni file | omni coverage | omni symbols / matched in Mercury (by name) | flag |
|---|---|---|---|
| `src/omni_anomaly_engine/core/config.py` | 100.0% | 6/6 | investigate |
| `src/omni_anomaly_engine/core/exceptions.py` | 100.0% | 7/7 | investigate |
| `src/omni_anomaly_engine/ml/attention.py` | 100.0% | 4/4 | investigate |
| `src/omni_anomaly_engine/models/affective.py` | 100.0% | 1/1 | investigate |
| `src/omni_anomaly_engine/models/quantum.py` | 100.0% | 1/1 | investigate |
| `src/omni_anomaly_engine/resilience/retry.py` | 100.0% | 1/1 | investigate |
| `src/omni_anomaly_engine/core/novel_class_discovery.py` | 98.5% | 2/2 | investigate |
| `src/omni_anomaly_engine/core/quantum_kernels.py` | 98.4% | 1/1 | investigate |
| `src/omni_anomaly_engine/core/extended_anomaly_engine.py` | 98.1% | 5/6 | investigate |
| `src/omni_anomaly_engine/ml/optimizers.py` | 97.6% | 5/5 | investigate |
| `src/omni_anomaly_engine/core/self_healing.py` | 97.3% | 1/2 | investigate |
| `src/omni_anomaly_engine/ml/training.py` | 96.9% | 2/7 | investigate |
| `src/omni_anomaly_engine/ml/layers.py` | 96.7% | 0/3 | HIGH_VALUE_EXTRACTION_CANDIDATE |
| `src/omni_anomaly_engine/ml/regularizers.py` | 96.4% | 0/3 | HIGH_VALUE_EXTRACTION_CANDIDATE |
| `src/omni_anomaly_engine/models/neural.py` | 95.9% | 1/1 | investigate |
| `src/omni_anomaly_engine/core/ethical_governor.py` | 95.0% | 4/4 | investigate |
| `src/omni_anomaly_engine/models/consciousness.py` | 94.9% | 1/1 | investigate |
| `src/omni_anomaly_engine/truth_decipher.py` | 93.7% | 2/2 | investigate |
| `src/omni_anomaly_engine/core/fusion.py` | 93.6% | 1/4 | investigate |
| `src/omni_anomaly_engine/detectors/dimensional.py` | 93.6% | 1/2 | investigate |
| `src/omni_anomaly_engine/core/ethical_config.py` | 92.3% | 2/2 | investigate |
| `src/omni_anomaly_engine/detectors/directive.py` | 91.7% | 1/1 | investigate |
| `src/omni_anomaly_engine/core/chaos_evolutionary.py` | 91.5% | 2/2 | investigate |
| `src/omni_anomaly_engine/core/info_geometry.py` | 91.4% | 1/1 | investigate |
| `src/omni_anomaly_engine/detectors/spatial.py` | 91.4% | 1/1 | investigate |
| `src/omni_anomaly_engine/engine.py` | 90.7% | 0/1 | HIGH_VALUE_EXTRACTION_CANDIDATE |
| `src/omni_anomaly_engine/resilience/circuit_breaker.py` | 90.7% | 2/2 | investigate |
| `src/omni_anomaly_engine/ml/inference.py` | 90.3% | 1/1 | investigate |
| `src/omni_anomaly_engine/detectors/statistical.py` | 90.3% | 0/1 | HIGH_VALUE_EXTRACTION_CANDIDATE |
| `src/omni_anomaly_engine/core/ai_ethics.py` | 89.7% | 5/5 | investigate |
| `src/omni_anomaly_engine/federated/federated_detector.py` | 89.7% | 3/4 | investigate |
| `src/omni_anomaly_engine/core/ethical_risk_matrix.py` | 89.4% | 9/9 | investigate |
| `src/omni_anomaly_engine/detectors/temporal.py` | 88.2% | 1/1 | investigate |
| `src/omni_anomaly_engine/emergent/emergent_life_detector.py` | 86.5% | 5/5 | investigate |
| `src/omni_anomaly_engine/models/multiverse.py` | 85.2% | 3/3 | investigate |
| `src/omni_anomaly_engine/ml/encoders.py` | 84.2% | 6/6 | investigate |
| `src/omni_anomaly_engine/core/multivariate_timeseries.py` | 83.0% | 2/3 | investigate |
| `src/omni_anomaly_engine/core/neurosymbolic_engine.py` | 82.1% | 5/5 | investigate |
| `src/omni_anomaly_engine/resilience/health_monitoring.py` | 81.8% | 2/2 | investigate |
| `src/omni_anomaly_engine/ml/fusion_network.py` | 81.5% | 2/2 | investigate |
| `src/omni_anomaly_engine/models/simulation.py` | 81.5% | 1/1 | investigate |
| `src/omni_anomaly_engine/core/base.py` | 81.0% | 2/3 | investigate |

**HIGH_VALUE_EXTRACTION_CANDIDATE count:** 4
# Mercury Agent Training Report

**Generated:** 2025-12-08T00:37:58.087863+00:00

## Training Summary

- **Epochs Completed:** 30
- **Scenarios per Epoch:** 8
- **Total Executions:** 240

## Final Metrics

- **Average Success Rate:** 100.00%
- **Average Plan Confidence:** 0.99
- **Legacy Confidence (fixed heuristic):** 0.76
- **Confidence vs Legacy:** +0.237
- **Confidence Growth:** +0.232
- **Improvement from Start:** +0.00%

## Bayesian Confidence Calibrator

The fixed 0.76 confidence heuristic has been replaced with a learned Bayesian calibrator.

- **Total Contexts Learned:** 8
- **Total Observations:** 240
- **Average Posterior Mean:** 0.992

### Per-Context Statistics

| Context | Observations | Successes | Failures | Posterior Mean |
|---------|--------------|-----------|----------|----------------|
| medical:analysis | 30 | 30 | 0 | 0.992 |
| security:analysis | 30 | 30 | 0 | 0.992 |
| humanitarian:monitoring | 30 | 30 | 0 | 0.992 |
| infrastructure:generic | 30 | 30 | 0 | 0.992 |
| energy:analysis | 30 | 30 | 0 | 0.992 |
| scientific:analysis | 30 | 30 | 0 | 0.992 |
| financial:analysis | 30 | 30 | 0 | 0.992 |
| general:analysis | 30 | 30 | 0 | 0.992 |

## Memory Accumulation

- Short-term entries: 100
- Long-term entries: 0
- Episodic entries: 240
- Semantic entries: 240
- **Total:** 580

## Training Progress

| Epoch | Success Rate | Confidence | Legacy | Delta | Passed | Time (ms) |
|-------|--------------|------------|--------|-------|--------|----------|
| 1 | 100.00% | 0.76 | 0.76 | +0.005 | 8/8 | 2.1 |
| 2 | 100.00% | 0.78 | 0.76 | +0.029 | 8/8 | 1.5 |
| 3 | 100.00% | 0.82 | 0.76 | +0.069 | 8/8 | 1.5 |
| 4 | 100.00% | 0.87 | 0.76 | +0.113 | 8/8 | 1.6 |
| 5 | 100.00% | 0.91 | 0.76 | +0.159 | 8/8 | 1.5 |
| 6 | 100.00% | 0.96 | 0.76 | +0.205 | 8/8 | 1.6 |
| 7 | 100.00% | 0.97 | 0.76 | +0.211 | 8/8 | 1.5 |
| 8 | 100.00% | 0.97 | 0.76 | +0.215 | 8/8 | 1.5 |
| 9 | 100.00% | 0.97 | 0.76 | +0.218 | 8/8 | 1.5 |
| 10 | 100.00% | 0.98 | 0.76 | +0.221 | 8/8 | 1.5 |
| 11 | 100.00% | 0.98 | 0.76 | +0.223 | 8/8 | 1.6 |
| 12 | 100.00% | 0.98 | 0.76 | +0.225 | 8/8 | 1.5 |
| 13 | 100.00% | 0.98 | 0.76 | +0.227 | 8/8 | 1.5 |
| 14 | 100.00% | 0.98 | 0.76 | +0.228 | 8/8 | 1.5 |
| 15 | 100.00% | 0.98 | 0.76 | +0.229 | 8/8 | 1.5 |
| 16 | 100.00% | 0.98 | 0.76 | +0.230 | 8/8 | 1.8 |
| 17 | 100.00% | 0.99 | 0.76 | +0.231 | 8/8 | 1.5 |
| 18 | 100.00% | 0.99 | 0.76 | +0.232 | 8/8 | 1.6 |
| 19 | 100.00% | 0.99 | 0.76 | +0.232 | 8/8 | 1.6 |
| 20 | 100.00% | 0.99 | 0.76 | +0.233 | 8/8 | 1.5 |
| 21 | 100.00% | 0.99 | 0.76 | +0.234 | 8/8 | 1.7 |
| 22 | 100.00% | 0.99 | 0.76 | +0.234 | 8/8 | 1.5 |
| 23 | 100.00% | 0.99 | 0.76 | +0.235 | 8/8 | 1.5 |
| 24 | 100.00% | 0.99 | 0.76 | +0.235 | 8/8 | 1.6 |
| 25 | 100.00% | 0.99 | 0.76 | +0.235 | 8/8 | 1.5 |
| 26 | 100.00% | 0.99 | 0.76 | +0.236 | 8/8 | 1.6 |
| 27 | 100.00% | 0.99 | 0.76 | +0.236 | 8/8 | 1.8 |
| 28 | 100.00% | 0.99 | 0.76 | +0.236 | 8/8 | 1.5 |
| 29 | 100.00% | 0.99 | 0.76 | +0.237 | 8/8 | 1.5 |
| 30 | 100.00% | 0.99 | 0.76 | +0.237 | 8/8 | 1.5 |

## Final Epoch Details

### medical_vital_signs

- Domain: medical
- Goal: Analyze patient vital signs for anomalies
- Success Rate: 100.00%
- Tasks Completed: 5
- Reasoning Steps: 0
- Passed: Yes

### security_network_traffic

- Domain: security
- Goal: Detect potential intrusion patterns in network traffic
- Success Rate: 100.00%
- Tasks Completed: 5
- Reasoning Steps: 0
- Passed: Yes

### humanitarian_crisis_detection

- Domain: humanitarian
- Goal: Monitor humanitarian crisis indicators
- Success Rate: 100.00%
- Tasks Completed: 3
- Reasoning Steps: 0
- Passed: Yes

### infrastructure_monitoring

- Domain: infrastructure
- Goal: Assess critical infrastructure health
- Success Rate: 100.00%
- Tasks Completed: 3
- Reasoning Steps: 0
- Passed: Yes

### energy_grid_analysis

- Domain: energy
- Goal: Optimize energy distribution and detect anomalies
- Success Rate: 100.00%
- Tasks Completed: 5
- Reasoning Steps: 0
- Passed: Yes

### scientific_data_analysis

- Domain: scientific
- Goal: Analyze experimental data for significant findings
- Success Rate: 100.00%
- Tasks Completed: 5
- Reasoning Steps: 0
- Passed: Yes

### financial_fraud_detection

- Domain: financial
- Goal: Detect potential fraudulent transactions
- Success Rate: 100.00%
- Tasks Completed: 5
- Reasoning Steps: 0
- Passed: Yes

### general_anomaly_detection

- Domain: general
- Goal: Perform general anomaly detection on mixed data
- Success Rate: 100.00%
- Tasks Completed: 4
- Reasoning Steps: 0
- Passed: Yes

## Notes

Mercury Agent is a rule-based autonomous agent that learns through experience accumulation in its memory systems and Bayesian confidence calibration. Unlike neural networks, it does not have gradient-based training. Instead, it builds up episodic and semantic memories that influence future planning and reasoning.

The training process exercises the agent through diverse scenarios across all supported domains (medical, security, humanitarian, infrastructure, energy, scientific, financial, general) to build a comprehensive knowledge base.

**Bayesian Confidence Calibration:** The fixed 0.76 confidence heuristic has been replaced with a learned Bayesian calibrator that uses Beta-Bernoulli conjugate priors. Confidence starts at ~0.76 for novel contexts and climbs toward 0.95-0.99+ after repeated successes, providing a more accurate reflection of the agent's growing competence.

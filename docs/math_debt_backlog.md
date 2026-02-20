# Mathematical Debt Backlog — Mercury Agent

## Overview
This document tracks remaining mathematical issues discovered during the Phase 1-6 audit. Items are prioritized by severity and impact.

## Active Debt Items

| ID | Issue | Severity | Effort | Impact | Recommendation |
|----|-------|----------|--------|--------|----------------|
| MD-002 | Statistical detector fusion weights (0.4/0.3/0.3) are fixed | MEDIUM | LOW | Detector combination quality | **PARTIALLY RESOLVED**: `DomainAdaptiveAAFEWeights` class now learns per-domain weight profiles. Full validation pending real datasets. |
| MD-003 | Neural-symbolic fusion weights (0.6/0.4) lack cross-validation evidence | MEDIUM | MEDIUM | Hybrid scoring quality | Validate via cross-validation on labeled datasets. |
| MD-004 | GOSNN ethical score weights (0.4/0.4/0.2) are arbitrary | LOW | LOW | Ethical gate sensitivity | Learn from labeled ethical compliance scenarios. |
| MD-005 | Conformal prediction needs real-data validation | MEDIUM | HIGH | Uncertainty quantification reliability | Validate coverage guarantees on BATADAL, SMD, NSL-KDD. |
| MD-009 | No PCA redundancy analysis on 180+ omni-scalars | MEDIUM | LOW | Computational efficiency | Run PCA to determine how many principal components capture 95% variance. Remove redundant scalars (correlation > 0.95). |
| MD-011 | Calibration pipeline needs real dataset integration | HIGH | MEDIUM | Threshold accuracy | Run `calibration_pipeline.py` against BATADAL, SMD, NSL-KDD and store calibrated thresholds. |
| MD-012 | Domain-adaptive harmonics need MUSIC/ESPRIT for production | LOW | HIGH | Spectral analysis for unknown domains | Replace scipy `find_peaks` with MUSIC algorithm for sub-Nyquist resolution. |
| MD-013 | Ensemble anomaly score fallback weights (0.4/0.35/0.25) unjustified | MEDIUM | LOW | Conformal prediction fallback quality | Validate via cross-validation or learn from data. |
| MD-014 | Choquet integral not implemented for multi-criteria aggregation | LOW | HIGH | Richer aggregation capturing interactions | Evaluate feasibility; requires Mobius transform computation. |
| MD-015 | Sigma Directive component weights not validated | MEDIUM | LOW | Ethical governance accuracy | Validate via sensitivity analysis or stakeholder input. |

## Priority Matrix

| Priority | Count | Examples |
|----------|-------|---------|
| RESOLVED | 5 | MD-001, MD-006, MD-007, MD-008, MD-010 |
| PARTIALLY RESOLVED | 1 | MD-002 |
| CRITICAL (fix immediately) | 0 | — |
| HIGH (next sprint) | 1 | MD-011 |
| MEDIUM (next quarter) | 5 | MD-003, MD-005, MD-009, MD-013, MD-015 |
| LOW (when feasible) | 3 | MD-004, MD-012, MD-014 |

## Resolution Process
For each item:
1. Create GitHub issue referencing this ID
2. Implement fix with unit tests
3. Validate on at least one real dataset
4. Update `equations_inventory.md` and `MATH_SPEC.md`
5. Close issue with PR reference

---

## Resolved Items

| ID | Issue | Resolution |
|----|-------|------------|
| MD-001 | AAFE golden ratio exponent (Phi = 1.618) lacks empirical validation | **RESOLVED**: 1,000-trial Optuna sweep confirms Phi is statistically optimal (p < 0.001, t=8.05). Mean F1 near Phi=0.9045 vs 0.8944. Results in `benchmarks/parameter_sweep_results.json`. |
| MD-006 | Topological Data Analysis (TDA) not yet implemented | **RESOLVED**: `core/topological_analysis.py` implements Vietoris-Rips filtration, 0D/1D persistent homology, persistence diagrams, Betti numbers, Wasserstein/bottleneck distances, and `TopologicalAnomalyDetector`. |
| MD-007 | Information-geometric thresholds not yet derived | **RESOLVED**: `core/info_geometry.py` now includes `FisherInformationMatrix`, `NaturalGradient`, `FisherRaoAdaptiveThreshold` (tau = mu + k*sqrt(tr(F^{-1}))), and drift detection with auto-recalibration. |
| MD-008 | Riemannian optimization not applied to constrained parameters | **RESOLVED**: `core/riemannian_optimization.py` provides `SimplexManifold`, `SPDManifold`, `RiemannianGradientDescent`, `RiemannianAdam`, and `ConstrainedParameterOptimizer` for AAFE weights on simplex and covariance on SPD. |
| MD-010 | Lyapunov stability enforcement is monitoring-only | **RESOLVED**: `core/system_coherence.py` provides `LyapunovRuntimeEnforcer` with halt-on-violation mode, violation tracking, and full coherence audit. |

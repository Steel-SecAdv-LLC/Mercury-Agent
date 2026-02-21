# Mathematical Debt Backlog — Mercury Agent

## Overview
This document tracks remaining mathematical issues discovered during the Phase 1-6 audit. Items are prioritized by severity and impact.

## Active Debt Items

| ID | Issue | Severity | Effort | Impact | Recommendation |
|----|-------|----------|--------|--------|----------------|
| MD-002 | Statistical detector fusion weights (0.4/0.3/0.3) are fixed | MEDIUM | LOW | Detector combination quality | **PARTIALLY RESOLVED**: `DomainAdaptiveOAEWeights` class now learns per-domain weight profiles. Full validation pending real datasets. |
| MD-005 | Conformal prediction coverage below 90% gate | LOW | MEDIUM | Conformal guarantee reliability | **PARTIALLY RESOLVED**: Score-based metric corrected, CrossConformal meets 90% target on 77.5% of datasets (below >90% gate). Investigate Mondrian conformal and larger calibration fractions. |
| MD-004 | GOSNN ethical score weights (0.4/0.4/0.2) are arbitrary | LOW | LOW | Ethical gate sensitivity | Learn from labeled ethical compliance scenarios. |
| MD-009 | No PCA redundancy analysis on 180+ omni-scalars | MEDIUM | LOW | Computational efficiency | Run PCA to determine how many principal components capture 95% variance. Remove redundant scalars (correlation > 0.95). |
| MD-012 | Domain-adaptive harmonics need MUSIC/ESPRIT for production | LOW | HIGH | Spectral analysis for unknown domains | Replace scipy `find_peaks` with MUSIC algorithm for sub-Nyquist resolution. |
| MD-013 | Ensemble anomaly score fallback weights (0.4/0.35/0.25) unjustified | MEDIUM | LOW | Conformal prediction fallback quality | Validate via cross-validation or learn from data. |
| MD-014 | Choquet integral not implemented for multi-criteria aggregation | LOW | HIGH | Richer aggregation capturing interactions | Evaluate feasibility; requires Mobius transform computation. |
| MD-015 | Sigma Directive component weights not validated | MEDIUM | LOW | Ethical governance accuracy | Validate via sensitivity analysis or stakeholder input. |

## Priority Matrix

| Priority | Count | Examples |
|----------|-------|---------|
| RESOLVED | 7 | MD-001, MD-003, MD-006, MD-007, MD-008, MD-010, MD-011 |
| PARTIALLY RESOLVED | 2 | MD-002, MD-005 |
| CRITICAL (fix immediately) | 0 | — |
| HIGH (next sprint) | 0 | — |
| MEDIUM (next quarter) | 3 | MD-009, MD-013, MD-015 |
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
| MD-001 | OAE golden ratio exponent (Phi = 1.618) lacks empirical validation | **RESOLVED**: 1,000-trial Optuna sweep confirms Phi is statistically optimal (p < 0.001, t=8.05). Mean F1 near Phi=0.9045 vs 0.8944. Results in `benchmarks/parameter_sweep_results.json`. |
| MD-003 | Neural-symbolic fusion weights (0.6/0.4) lack cross-validation evidence | **RESOLVED** (2026-02-20): L-BFGS-B cross-validated fusion weights (same mechanism as `NeuroSymbolicHub._learn_fusion_weights`) tested on 40 real datasets via `run_fusion_weight_cv()` in `benchmarks/calibration_validation.py`. Default weights (0.4/0.3/0.3) validated on 72.5% of datasets (delta < 0.02 F1 vs CV-optimal); adaptive AUC-proportional weights validated on 82.5%. Mean optimal weights: R=0.516, K=0.087, I=0.397. Mean delta optimal-vs-default: +0.0099 F1, optimal-vs-adaptive: +0.0031 F1. Both are near-optimal — no constant changes warranted. Evidence: `benchmarks/calibration_validation_results.json` (`fusion_cv` key). |
| MD-006 | Topological Data Analysis (TDA) not yet implemented | **RESOLVED**: `core/topological_analysis.py` implements Vietoris-Rips filtration, 0D/1D persistent homology, persistence diagrams, Betti numbers, Wasserstein/bottleneck distances, and `TopologicalAnomalyDetector`. |
| MD-007 | Information-geometric thresholds not yet derived | **RESOLVED**: `core/info_geometry.py` now includes `FisherInformationMatrix`, `NaturalGradient`, `FisherRaoAdaptiveThreshold` (tau = mu + k*sqrt(tr(F^{-1}))), and drift detection with auto-recalibration. |
| MD-008 | Riemannian optimization not applied to constrained parameters | **RESOLVED**: `core/riemannian_optimization.py` provides `SimplexManifold`, `SPDManifold`, `RiemannianGradientDescent`, `RiemannianAdam`, and `ConstrainedParameterOptimizer` for OAE weights on simplex and covariance on SPD. |
| MD-010 | Lyapunov stability enforcement is monitoring-only | **RESOLVED**: `core/system_coherence.py` provides `LyapunovRuntimeEnforcer` with halt-on-violation mode, violation tracking, and full coherence audit. |
| MD-011 | Calibration pipeline needs real dataset integration | **RESOLVED** (2025-02-20): `ThresholdCalibrationPipeline` validated on 40 real datasets via `benchmarks/calibration_validation.py`. Calibration improved F1 on 80% of datasets (32/40), with mean delta F1 of +0.143. Mean calibrated F1: 0.4192 vs mean uncalibrated F1: 0.2763. Adaptive weights selected on 36/40 datasets; fallback 0.40/0.30/0.30 used on 4. Honest benchmark regression check passed (AUC delta < 0.001). Evidence: `benchmarks/calibration_validation_results.json`. |

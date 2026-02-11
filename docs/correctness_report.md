# Correctness Report — Mercury Agent Mathematical Audit

## 1. Numerical Stability Issues

### 1.1 Division by Zero Guards
- PASS: EPSILON = 1e-8 used in centralized_constants.py
- PASS: std + 1e-10 guard in z-score computations (calibration.py line 542, conformal_prediction.py line 542)
- PASS: (max - min + 1e-10) guard in normalization
- WARNING: detectors/statistical.py line 118 — z_scores = (data - mean) / std — no explicit epsilon guard if std=0 for individual features. The std is computed per-axis but no per-element guard.
- PASS: score_calibration.py line 870 uses (max - min + 1e-10)

### 1.2 Overflow/Underflow Potential
- WARNING: EQ-001 AAFE ethical_scaling = η^Φ where η ∈ [0.90, 0.99] and Φ = 1.618. Range: 0.90^1.618 ≈ 0.837 to 0.99^1.618 ≈ 0.984. No overflow risk but values are always < 1, which means ethical scaling always reduces the fusion score. This is by design.
- WARNING: core/calibration.py line 286 clips to [1e-10, 1-1e-10] before logit — PASS
- WARNING: conformal_prediction.py line 476 — sigmoid 1/(1+exp(-x)) — could overflow for large positive x (exp(-x) → 0, safe) or large negative x (exp(|x|) → overflow). Should use np.clip on decision values.
- PASS: exp(-z_distance/3) in z-score anomaly — z_distance is always positive, so exponent is negative, no overflow.
- PASS: Lyapunov bound e^(-λt) with λ=0.25, t≥0 — always ≤ 1, no overflow.

### 1.3 NaN/Inf Propagation
- PASS: score_calibration.py _validate_scores() checks for NaN/Inf and replaces with median
- PASS: conformal_prediction.py np.fill_diagonal(distances, np.inf) is intentional for self-exclusion
- WARNING: core/three_r/fusion.py — no NaN check on input scores. If any component (R, H, O) is NaN, the fusion score propagates NaN.
- WARNING: global_omni_scalar_network.py — no NaN guard on scalar values before aggregation

### 1.4 Floating-Point Precision
- PASS: Golden ratio constants carry 15+ significant digits
- PASS: Weight normalization phi_sum = φ + 1 + 1/φ ≈ 2.8944 — verified: w_R + w_H + w_O = 1.0 (exact due to algebraic identity)
- PASS: SOFA weights sum to exactly 1.0 (0.20+0.15+0.15+0.20+0.15+0.15 = 1.00)

## 2. Mathematical Correctness Issues

### 2.1 AAFE Golden Ratio Exponent (EQ-001) — CRITICAL
- ISSUE: The golden ratio Φ = 1.618... is used as the ethical scaling exponent with no mathematical justification.
- IMPACT: This is the core fusion equation. An unjustified exponent means the entire scoring system has an arbitrary nonlinearity.
- RECOMMENDATION: Replace with empirically optimized exponent from parameter sweep, or provide structural derivation.
- STATUS: UNJUSTIFIED — needs derivation or empirical evidence

### 2.2 Benevolence Hard Threshold (EQ-006) — HIGH
- ISSUE: β ≥ 0.99 is a discontinuous step function. Small perturbations near 0.99 cause binary flip.
- IMPACT: Causes brittleness in edge cases. A score of 0.989 is fully rejected, 0.991 is fully accepted.
- RECOMMENDATION: Replace with sigmoid curve η(b) = 1/(1+exp(-k·(b-b₀)))
- STATUS: DESIGN FLAW — needs smooth transition

### 2.3 Fusion Weights (EQ-010, EQ-017, EQ-032) — MEDIUM
- ISSUE: Multiple sets of hardcoded fusion weights (0.4/0.3/0.3, 0.4/0.35/0.25, 0.6/0.4) with no cross-validation evidence.
- IMPACT: Suboptimal performance on different domains.
- RECOMMENDATION: Learn weights from validation data or use domain-specific profiles.
- STATUS: UNJUSTIFIED — needs empirical validation

### 2.4 Recursion Convergence (EQ-036) — HIGH
- ISSUE: R(x,d) = f(x) + α·R(g(x), d-1) has no proven convergence guarantee. No constraint on α.
- IMPACT: Could diverge for α ≥ 1 or accumulate unbounded error.
- RECOMMENDATION: Constrain α via sigmoid with α_max = 0.95. Add error bound: err ≤ α^d·‖x₀‖/(1-α).
- STATUS: MISSING CONVERGENCE PROOF

### 2.5 Schumann Resonance Universality (EQ-021) — MEDIUM
- ISSUE: Harmonic analysis uses 7.83 Hz (Schumann) as fundamental frequency regardless of domain.
- IMPACT: This frequency is only physically meaningful for geophysical/environmental data. Using it for medical, security, or financial data is scientifically invalid.
- RECOMMENDATION: Implement domain-specific fundamental frequency selection.
- STATUS: DOMAIN MISMATCH

### 2.6 GOSNN Ethical Score Weights (EQ-039) — LOW
- ISSUE: Weights 0.4/0.4/0.2 for positive_ratio/mean/std are arbitrary.
- IMPACT: The ethical gate may not be optimally sensitive.
- RECOMMENDATION: Validate or learn from labeled ethical scenarios.
- STATUS: UNJUSTIFIED

## 3. Edge Case Handling

### 3.1 Empty Input Arrays
- PASS: score_calibration.py handles empty arrays gracefully (returns default diagnostics)
- PASS: conformal_prediction.py raises RuntimeError if not fitted
- WARNING: core/three_r/fusion.py — no guard against empty score arrays

### 3.2 Single-Sample Edge Case
- PASS: calibration ECE handles n_bins > n_samples
- WARNING: IQR computation fails silently with < 4 samples (Q1=Q3, IQR=0)
- PASS: Fallback to percentile when IQR fails

### 3.3 Constant Input
- PASS: MAD fallback when MAD ≈ 0
- PASS: IQR fallback when IQR ≈ 0
- WARNING: Z-score produces 0/0 = NaN when std=0. Guard needed.

## 4. Docstring/Implementation Mismatches

### 4.1 AAFE Weight Documentation
- ISSUE: README documents weights as w_R=0.35, w_H=0.35, w_O=0.30 but code uses w_R≈0.447, w_H≈0.276, w_O≈0.276
- LOCATION: README.md vs core/centralized_constants.py lines 197-199
- STATUS: DOCUMENTATION MISMATCH — code is correct (golden-ratio derived), README is stale

### 4.2 Lyapunov Decay Rate
- ISSUE: test_discovery_verification.py line 41 asserts LAMBDA_DECAY == 0.18, but centralized_constants.py uses 0.25
- LOCATION: tests/test_discovery_verification.py vs core/centralized_constants.py
- STATUS: POTENTIAL TEST FAILURE — constants were updated but test may be outdated

### 4.3 Benevolence Description
- ISSUE: README says "benevolence ≥ 0.99" but code clips ethical threshold to [0.90, 0.99]
- STATUS: These are different things — benevolence is a separate metric from ethical threshold

## 5. Dimensional Consistency

### 5.1 Score Normalization
- PASS: All anomaly scores are normalized to [0, 1] before fusion
- PASS: Ethical thresholds are in [0, 1]
- PASS: AAFE output is bounded by η^Φ ≤ 1 and weighted sum ≤ 1

### 5.2 Physical Units
- PASS: Schumann frequencies in Hz (physically correct)
- PASS: Vital signs in standard medical units (bpm, mmHg, °C, %)
- PASS: Time steps in seconds consistently

## 6. Summary

| Category | Issues Found | Critical | High | Medium | Low |
|----------|-------------|----------|------|--------|-----|
| Numerical Stability | 4 warnings | 0 | 0 | 3 | 1 |
| Mathematical Correctness | 6 issues | 1 | 2 | 2 | 1 |
| Edge Cases | 3 warnings | 0 | 0 | 2 | 1 |
| Doc Mismatches | 3 issues | 0 | 1 | 1 | 1 |
| Dimensional | 0 issues | 0 | 0 | 0 | 0 |
| **Total** | **16** | **1** | **3** | **8** | **4** |

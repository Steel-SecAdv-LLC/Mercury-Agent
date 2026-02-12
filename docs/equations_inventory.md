# Mercury-Agent: Equations, Formulas, Constants, and Thresholds Inventory

> **Purpose:** This document provides a comprehensive, auditable catalog of every equation, formula, constant, threshold, weight, and mathematical operation found in the Mercury-Agent codebase. Each entry is cross-referenced to its source file and line numbers, classified by type, and annotated with provenance information. Entries marked **UNJUSTIFIED** lack empirical or theoretical derivation and require calibration or citation.

> **Generated:** 2026-02-11

---

## Notation Key

| Symbol | Meaning |
|--------|---------|
| $\Phi$ | Golden ratio $= 1.618033988749895\ldots$ |
| $\eta$ | Ethical compliance threshold |
| $\sigma$ | Sigma Immutable threshold (domain-dependent) |
| $\alpha$ | Miscoverage rate (conformal prediction) |
| $\lambda$ | Lyapunov convergence rate |
| $\varepsilon$ | Numerical stability epsilon or initial bound |

---

## Master Equation Table

| ID | Location | Equation (LaTeX) | Type | Purpose | Input Types / Ranges | Output Range | Constants Used | Provenance |
|----|----------|-------------------|------|---------|----------------------|--------------|----------------|------------|
| EQ-001 | `core/three_r/fusion.py` `compute()` lines 29--130 | $A = \bigl(w_R \cdot R(x) + w_H \cdot H(\omega) + w_O \cdot O(\theta)\bigr) \cdot \eta^{\,\Phi}$ | hybrid | Core AAFE fusion score combining recursion, resonance, and optimization sub-scores with ethical gating exponent | $R(x), H(\omega), O(\theta) \in [0,1]$; $\eta \in [0.90, 0.99]$ | $[0, 1]$ | $\Phi = 1.618\ldots$; $w_R = \frac{\Phi}{\Phi+1+1/\Phi} \approx 0.447$; $w_H = \frac{1}{\Phi+1+1/\Phi} \approx 0.276$; $w_O = \frac{1/\Phi}{\Phi+1+1/\Phi} \approx 0.276$ | Golden-ratio weighting is mathematically grounded; **exponent $\eta^\Phi$ needs derivation** -- no published justification for raising compliance to the golden power |
| EQ-002 | `core/three_r/fusion.py` lines 133--134 | $V(S_t) \leq \varepsilon \cdot e^{-\lambda t}, \quad \lambda = 0.25$ | dynamic | Lyapunov stability bound ensuring fusion score convergence | $t \in \mathbb{Z}^+$; $\varepsilon = 1.0$ | Exponentially decaying bound $\to 0$ | $\lambda = 0.25$; $\varepsilon = 1.0$ | Lyapunov stability theory (standard). $\lambda = 0.25$ chosen by design |
| EQ-003 | `core/three_r/fusion.py` line 166 | $w_{\text{new}} = (1 - \text{lr}) \cdot w_{\text{old}} + \text{lr} \cdot w_{\text{target}}$ | dynamic | Attention-based fusion weight update (exponential moving average) | $\text{lr} \in (0, 1]$; $w_{\text{old}}, w_{\text{target}} \in [0,1]$ | $[0, 1]$ (re-normalized after update) | $\text{lr} = 0.01$ (default) | Exponential moving average -- standard technique |
| EQ-004 | `core/three_r/fusion.py` lines 191--201 | $\hat{\lambda} = -\frac{\ln(\text{ratio})}{t}, \quad \text{ratio} = \frac{\text{Var}(\text{recent})}{\text{Var}(\text{initial})}$ | dynamic | Estimate Lyapunov decay rate from variance ratio of convergence history | $\text{ratio} > 0$; $t > 0$ | $\hat{\lambda} \in \mathbb{R}$ (positive $\Rightarrow$ stable) | None | Logarithmic analysis of variance decay -- standard stability estimation |
| EQ-005 | `core/ethical_governor.py` lines 129--134 | $S = \frac{J \cdot w_J + A \cdot w_A + C \cdot w_C + T \cdot w_T}{\sum w_i}$ | static | Sigma Directive weighted ethical score for action gating | $J, A, C, T \in [0,1]$; weights from `EthicalScalars` config | $[0, 1]$ | Threshold $\geq 0.8$ (`SIGMA_DIRECTIVE_THRESHOLD`) | Weighted average design -- weights from ethical configuration |
| EQ-006 | `core/centralized_constants.py` line 117 | $\beta \geq 0.99$ (hard threshold) | static | Benevolence gate -- civilization-first decision filter | $\beta \in [0, 1]$ | Boolean pass/fail | $0.99$ | **UNJUSTIFIED:** needs empirical calibration for the 0.99 threshold |
| EQ-007a | `core/centralized_constants.py` line 101 | $\sigma_{\text{default}} = 0.96$ | static | Default ethical compliance threshold | N/A | Threshold constant | $0.96$ | **UNJUSTIFIED:** domain-specific design choice without published derivation |
| EQ-007b | `core/centralized_constants.py` line 105 | $\sigma_{\text{medical}} = 0.93$ | static | Medical domain ethical threshold (lower for urgency) | N/A | Threshold constant | $0.93$ | **UNJUSTIFIED:** domain-specific design choice |
| EQ-007c | `core/centralized_constants.py` line 109 | $\sigma_{\text{infrastructure}} = 0.995$ | static | Infrastructure domain ethical threshold (highest criticality) | N/A | Threshold constant | $0.995$ | **UNJUSTIFIED:** domain-specific design choice |
| EQ-007d | `core/centralized_constants.py` line 113 | $\sigma_{\text{humanitarian}} = 0.95$ | static | Humanitarian domain ethical threshold | N/A | Threshold constant | $0.95$ | **UNJUSTIFIED:** domain-specific design choice |
| EQ-008 | `detectors/statistical.py` line 118, 294 | $z = \frac{x - \mu}{\sigma}$ | dynamic | Standard z-score for outlier detection | $x, \mu \in \mathbb{R}$; $\sigma > 0$ (stabilized with $+10^{-8}$) | $z \in \mathbb{R}$; threshold $\lvert z \rvert > 3.0$ | $z_{\text{threshold}} = 3.0$ | Standard statistical method (Gaussian theory) |
| EQ-009 | `detectors/statistical.py` (via `score_calibration.py`) | $T = Q_3 + 1.5 \cdot \text{IQR}, \quad \text{IQR} = Q_3 - Q_1$ | dynamic | IQR-based outlier threshold | $Q_1, Q_3$ from data percentiles | Threshold value in data units | $1.5$ (IQR multiplier) | Tukey (1977) box-plot method |
| EQ-010 | `detectors/statistical.py` line 197 | $S = 0.4 \cdot z_{\text{score}} + 0.3 \cdot \text{iqr}_{\text{score}} + 0.3 \cdot \text{if}_{\text{score}}$ | static | Combined detector score from z-score, IQR, and Isolation Forest | Each component $\in [0, 1]$ | $[0, 1]$ | Weights: $0.4, 0.3, 0.3$ | **UNJUSTIFIED:** weights are hard-coded without empirical optimization |
| EQ-011 | `detectors/statistical.py` lines 179--180 | $I = \frac{\text{clip}\!\left(\frac{\max(\lvert z \rvert)}{z_{\text{threshold}} + \varepsilon},\; 0,\; 3\right)}{3}$ | dynamic | Normalized z-score intensity for continuous anomaly scoring | $z \in \mathbb{R}$; $z_{\text{threshold}} = 3.0$ | $[0, 1]$ | $z_{\text{threshold}} = 3.0$; $\varepsilon = 10^{-8}$ | Normalization of z-score to unit interval -- design choice |
| EQ-012 | `detectors/statistical.py` line 120 | $c = \text{clip}\!\left(\text{outlier\_frac} \times 2 + 0.001,\; 0.001,\; 0.5\right)$ | dynamic | Adaptive contamination estimation for Isolation Forest | $\text{outlier\_frac} \in [0, 1]$ (fraction with $\lvert z \rvert > 3$) | $[0.001, 0.5]$ | Multiplier $2$; floor $0.001$; cap $0.5$ | Heuristic scaling of observed outlier fraction -- design choice |
| EQ-013 | `core/calibration.py` lines 62--91 | $\text{ECE} = \sum_{b=1}^{B} \frac{\lvert \mathcal{B}_b \rvert}{N} \cdot \lvert \text{acc}(\mathcal{B}_b) - \text{conf}(\mathcal{B}_b) \rvert$ | dynamic | Expected Calibration Error measuring confidence-accuracy alignment | $y_{\text{true}} \in \{0,1\}$; $y_{\text{prob}} \in [0,1]$; $B = 10$ bins | $[0, 1]$ (0 = perfectly calibrated) | $B = 10$ bins (default) | Naeini et al. (2015) |
| EQ-014 | `core/calibration.py` lines 144--148 | $p = \frac{1}{1 + e^{-(ax + b)}}$ | dynamic | Platt scaling -- logistic regression post-hoc calibration | $x \in \mathbb{R}$ (raw model score) | $[0, 1]$ (calibrated probability) | $C = 10^{10}$ (minimal regularization); max\_iter $= 100$ | Platt (1999) |
| EQ-015 | `core/calibration.py` lines 314--316 | $p = \text{sigmoid}\!\left(\frac{\text{logits}}{T}\right)$ | dynamic | Temperature scaling -- single-parameter calibration | $\text{logits} \in \mathbb{R}$; $T \in [0.1, 10]$ | $[0, 1]$ (calibrated probability) | $T$ optimized via grid search over $[0.1, 10]$ | Guo et al. (2017) |
| EQ-016 | `core/calibration.py` line 715; `core/conformal_prediction.py` line 545 | $a = 1 - e^{-z_{\text{distance}}/3}$ | dynamic | Z-score-based anomaly score via exponential CDF approximation | $z_{\text{distance}} \geq 0$ (mean absolute z-score) | $[0, 1)$ | Divisor $= 3$ | Exponential CDF approximation -- standard transform |
| EQ-017 | `core/conformal_prediction.py` line 577 | $S = 0.4 \cdot z_{\text{anomaly}} + 0.35 \cdot \text{density}_{\text{anomaly}} + 0.25 \cdot \text{percentile}_{\text{anomaly}}$ | static | Ensemble fallback anomaly score combining three methods | Each component $\in [0, 1]$ | $[0, 1]$ | Weights: $0.4, 0.35, 0.25$ | **UNJUSTIFIED:** hard-coded ensemble weights without empirical basis |
| EQ-018 | `core/conformal_prediction.py` lines 109--111 | $q = \frac{\lceil (n+1)(1-\alpha) \rceil}{n}$ | dynamic | Conformal quantile with finite-sample correction | $n \in \mathbb{Z}^+$ (calibration set size); $\alpha \in (0,1)$ | Quantile index into sorted scores | None | Vovk et al. (2005) "Algorithmic Learning in a Random World" |
| EQ-019 | `core/conformal_prediction.py` lines 144--145 | $\left[\hat{y} - q,\; \hat{y} + q\right]$ | dynamic | Conformal prediction interval | $\hat{y} \in \mathbb{R}$ (point prediction); $q > 0$ (quantile threshold) | Interval in prediction space | $q$ from EQ-018 | Split conformal prediction theory -- Vovk et al. (2005) |
| EQ-020 | `core/conformal_prediction.py` lines 313--316 | $\theta_{t+1} = \theta_t + \text{lr} \cdot (\text{miscov} - \alpha)$ | dynamic | Adaptive conformal inference -- online threshold update | $\theta_t \geq 0$; $\text{miscov} \in \{0, 1\}$; $\alpha \in (0,1)$ | $\theta_{t+1} \geq 0$ | $\text{lr}$ (learning rate) | Gibbs & Cand\`es (2021) "Adaptive Conformal Inference Under Distribution Shift" |
| EQ-021 | `detectors/spectral_vibration.py` | $A(x) = \frac{\sum_{n} H(n \cdot \omega_0)}{\sum H(\omega)}, \quad \omega_0 = 7.83\;\text{Hz}$ | hybrid | Harmonic ratio at Schumann resonance harmonics for frequency-domain anomaly detection | $H(\omega) \geq 0$ (spectral power); harmonics at $[7.83, 14.3, 20.8, 27.3, 33.8]$ Hz | $[0, 1]$ (ratio) | $\omega_0 = 7.83$ Hz (Schumann fundamental) | Schumann resonance -- valid for environmental/geophysical signals only |
| EQ-022 | `detectors/spectral_vibration.py` | $H(\omega) = \lvert \text{FFT}(x) \rvert^2$ | dynamic | FFT power spectrum (periodogram) | Time-series $x \in \mathbb{R}^n$ | $H(\omega) \geq 0$ | None | Standard DSP -- Cooley-Tukey FFT |
| EQ-023 | `detectors/acceleration_dynamics.py` | $v = \frac{\Delta x}{\Delta t}$ | dynamic | Velocity (first-order finite difference of position/value) | $\Delta x \in \mathbb{R}$; $\Delta t > 0$ | $v \in \mathbb{R}$ | None | Classical kinematics (Newton) |
| EQ-024 | `detectors/acceleration_dynamics.py` | $a = \frac{\Delta v}{\Delta t}$ | dynamic | Acceleration (second-order finite difference) | $\Delta v \in \mathbb{R}$; $\Delta t > 0$ | $a \in \mathbb{R}$ | None | Classical kinematics (Newton) |
| EQ-025 | `detectors/acceleration_dynamics.py` | $j = \frac{\Delta a}{\Delta t}$ | dynamic | Jerk (third-order finite difference) | $\Delta a \in \mathbb{R}$; $\Delta t > 0$ | $j \in \mathbb{R}$ | None | Classical kinematics |
| EQ-026 | `detectors/acceleration_dynamics.py` | $KE = \frac{1}{2} m v^2$ | dynamic | Kinetic energy for energy-based anomaly scoring | $m > 0$ (effective mass); $v \in \mathbb{R}$ | $KE \geq 0$ | None | Classical mechanics (Newton) |
| EQ-027 | `detectors/acceleration_dynamics.py` | $p = m v$ | dynamic | Momentum for trend-strength analysis | $m > 0$; $v \in \mathbb{R}$ | $p \in \mathbb{R}$ | None | Classical mechanics (Newton) |
| EQ-028 | `core/score_calibration.py` line 792 | $\sigma^2_B = w_b \cdot w_f \cdot (\mu_b - \mu_f)^2$ | dynamic | Otsu between-class variance for bimodal threshold selection | Score histogram; $w_b, w_f$ = class weights; $\mu_b, \mu_f$ = class means | $\sigma^2_B \geq 0$ | 256-bin histogram | Otsu (1979) "A Threshold Selection Method from Gray-Level Histograms" |
| EQ-029 | `core/score_calibration.py` line 834 | $T = \text{median} + k \cdot 1.4826 \cdot \text{MAD}$ | dynamic | MAD-based robust threshold with adaptive $k$ | Scores $\in \mathbb{R}$; $k = 3.0 - 2.0 \cdot \text{contamination}$, clamped $\geq 1.5$ | Threshold in score space | $1.4826$ (consistency factor for normal); $k \in [1.5, 3.0]$ | Robust statistics -- MAD with Gaussian consistency factor |
| EQ-030 | `core/score_calibration.py` line 1129 | $J = \text{TPR} - \text{FPR}$ | dynamic | Youden's J statistic for optimal threshold selection | TPR, FPR $\in [0, 1]$ | $J \in [-1, 1]$ | None | Youden (1950) "Index for rating diagnostic tests" |
| EQ-031 | `core/score_calibration.py` line 1062 | $F_1 = \frac{2 \cdot P \cdot R}{P + R}$ | dynamic | F1 score for threshold optimization | Precision $P$, Recall $R \in [0,1]$ | $[0, 1]$ | None | Standard classification metric (harmonic mean of precision and recall) |
| EQ-032 | `core/centralized_constants.py` lines 203--204 | $S = 0.6 \cdot \text{neural} + 0.4 \cdot \text{symbolic}$ | static | Neural-symbolic fusion weighting | neural, symbolic $\in [0, 1]$ | $[0, 1]$ | $w_{\text{neural}} = 0.6$; $w_{\text{symbolic}} = 0.4$ | **UNJUSTIFIED:** hard-coded weights without empirical or theoretical basis |
| EQ-033 | `core/centralized_constants.py` line 214 | $w_t = 0.9^{\,t}$ | static | Ensemble decay weight for temporal averaging | $t \in \mathbb{Z}^{\geq 0}$ (time step) | $(0, 1]$ | Decay factor $= 0.9$ | Exponential decay -- common design pattern |
| EQ-034 | `core/centralized_constants.py` lines 272--277 | SOFA component weights: resp$= 0.20$, coag$= 0.15$, liver$= 0.15$, cardio$= 0.20$, CNS$= 0.15$, renal$= 0.15$ | static | Weighted SOFA score for medical domain organ dysfunction scoring | Each component score $\in [0, 4]$ (SOFA sub-score range) | Weighted sum $\in [0, 4]$ | Weights sum to $1.0$ | JAMA 2016 -- Sequential Organ Failure Assessment (SOFA) score |
| EQ-035 | `core/centralized_constants.py` lines 292--300 | $P(d) = \log_{10}\!\left(1 + \frac{1}{d}\right), \quad d \in \{1, 2, \ldots, 9\}$ | static | Benford's Law expected first-digit distribution for financial fraud detection | $d \in \{1, \ldots, 9\}$ | $P(d) \in (0.046, 0.301)$ | Pre-computed: $[0.301, 0.176, 0.125, 0.097, 0.079, 0.067, 0.058, 0.051, 0.046]$ | Benford (1938) "The Law of Anomalous Numbers" -- mathematically verified |
| EQ-036 | `README.md` (documented) | $R(x, d) = f(x) + \alpha \cdot R\bigl(g(x),\; d-1\bigr)$ | dynamic | Recursive decomposition for hierarchical feature extraction | $x$ = input data; $d \in \mathbb{Z}^+$ (depth); $\alpha \in (0, 1)$ (decay) | Accumulated score | $\alpha$ (recursive decay) | Recursive decomposition -- standard technique. **MISSING:** convergence proof |
| EQ-037 | Optimizer (documented) | $\theta_{t+1} = \theta_t - \eta \cdot \nabla L(\theta_t) + \beta \cdot (\theta_t - \theta_{t-1})$ | dynamic | Momentum-based gradient descent for weight optimization | $\theta \in \mathbb{R}^n$; $\eta > 0$ (learning rate); $\beta \in [0, 1)$ (momentum) | $\theta_{t+1} \in \mathbb{R}^n$ | Default: $\eta = 0.01$, $\beta = 0.9$ (from `AAFEWeightOptimizer`) | Polyak (1964) "Some methods of speeding up the convergence of iteration methods" |
| EQ-038 | Multi-objective (documented) | $L = L_{\text{detection}} + \lambda_1 \cdot L_{\text{stability}} + \lambda_2 \cdot L_{\text{ethical}}$ | dynamic | Total loss combining detection, Lyapunov stability, and ethical compliance losses | Each $L_i \geq 0$; $\lambda_1, \lambda_2 > 0$ (regularization weights) | $L \geq 0$ | $\lambda_1, \lambda_2$ (task-specific) | Multi-objective optimization -- standard Lagrangian formulation |
| EQ-039 | `core/global_omni_scalar_network.py` lines 188--195 | $S = 0.4 \cdot \text{positive\_ratio} + 0.4 \cdot \min\!\left(\frac{\mu}{2}, 1\right) + 0.2 \cdot \frac{1}{1 + \text{std}}$ | static | GOSNN ethical gate score (NumPy fallback when PyTorch unavailable) | scalar\_vector $\in \mathbb{R}^n$ | $[0, 1]$ | Weights: $0.4, 0.4, 0.2$ | **UNJUSTIFIED:** ad-hoc weighted combination without empirical basis |
| EQ-040 | `core/global_omni_scalar_network.py` line 145 | $\sigma \geq 0.93$ (gate threshold) | static | GOSNN ethical gate pass/fail threshold | $\sigma \in [0, 1]$ | Boolean pass/fail | $0.93$ | **UNJUSTIFIED:** hard-coded threshold |
| EQ-041 | `detectors/uiux_anomaly.py` lines 220--221 | $MT = a + b \cdot \log_2\!\left(\frac{D}{W} + 1\right)$ | dynamic | Fitts's Law for expected mouse movement time | $D > 0$ (distance in px); $W > 0$ (target width in px); $a = 0.1$, $b = 0.15$ | $MT > 0$ (seconds) | $a = 0.1$ (intercept); $b = 0.15$ (slope) | Fitts (1954); Shannon formulation per MacKenzie (1992) |
| EQ-042 | `utils/constants.py` line 603 | $A = \min\!\left(0.95,\; A_{\text{base}} + \text{boost}\right)$; boost $= 0.05$ if stability $> 15.0$ | hybrid | Autonomy cap ensuring system autonomy never exceeds 0.95 | $A_{\text{base}} \in [0, 1]$; stability score $\in \mathbb{R}^+$ | $[0, 0.95]$ | Cap $= 0.95$; boost $= 0.05$; stability threshold $= 15.0$ | Design constraint (safety cap) |
| EQ-043 | `benchmarks/benevolence_optimization.py` line 221 | $P = (\text{threshold} - \sigma)^2$ | static | Quadratic penalty for ethical compliance violations | $\sigma \in [0, 1]$; threshold $\in [0, 1]$ | $P \geq 0$ ($P = 0$ when $\sigma \geq$ threshold) | None | Convex penalty -- standard quadratic loss form |
| EQ-044 | `benchmarks/benevolence_optimization.py` line 258 | $P = \max(0,\; \text{threshold} - \sigma)$ | static | Linear penalty (ReLU-style) for ethical compliance | $\sigma \in [0, 1]$; threshold $\in [0, 1]$ | $P \geq 0$ | None | Hinge loss variant -- standard optimization |
| EQ-045 | `benchmarks/benevolence_optimization.py` lines 275--307 | $G = \frac{1}{1 + \exp\!\bigl(k \cdot (\sigma - \text{threshold})\bigr)}$ | static | Sigmoid gate for smooth probabilistic ethical gating | $\sigma \in [0, 1]$; $k \in [5, 10]$ (sharpness) | $(0, 1)$ | $k$ (sharpness parameter) | Logistic function -- standard smooth approximation to step function |
| EQ-046 | `benchmarks/benevolence_optimization.py` lines 319--353 | $P = \exp\!\bigl(-k \cdot (\sigma - \text{threshold})\bigr)$ for $\sigma < \text{threshold}$ | static | Exponential penalty for critical ethical failures | $\sigma \in [0, 1]$; $k > 0$ (sharpness) | $P \geq 1$ when $\sigma <$ threshold | $k$ (sharpness parameter) | Exponential penalty -- standard form for sharp veto |
| EQ-047 | `benchmarks/benevolence_optimization.py` lines 360--400 | $G = \exp\!\left(-\frac{(\sigma - \text{threshold})^2}{2 \cdot \text{var}}\right)$ | static | Gaussian RBF gate for localized ethical penalty around threshold | $\sigma \in [0, 1]$; var $> 0$ (width) | $(0, 1]$ | var (variance / width parameter) | Gaussian radial basis function -- standard kernel |

---

## Summary Statistics

### Equations by Type

| Type | Count | IDs |
|------|-------|-----|
| **Static** (fixed constants, hard-coded weights, thresholds) | 21 | EQ-005, EQ-006, EQ-007a--d, EQ-010, EQ-032, EQ-033, EQ-034, EQ-035, EQ-039, EQ-040, EQ-043, EQ-044, EQ-045, EQ-046, EQ-047 |
| **Dynamic** (computed from data at runtime) | 23 | EQ-002, EQ-003, EQ-004, EQ-008, EQ-009, EQ-011, EQ-012, EQ-013, EQ-014, EQ-015, EQ-016, EQ-018, EQ-019, EQ-020, EQ-022, EQ-023, EQ-024, EQ-025, EQ-026, EQ-027, EQ-028, EQ-029, EQ-030, EQ-031, EQ-036, EQ-037, EQ-038 |
| **Hybrid** (combines static weights with dynamic computation) | 3 | EQ-001, EQ-021, EQ-042 |

### Provenance Status

| Status | Count | IDs |
|--------|-------|-----|
| **Justified** (published reference or mathematical derivation) | 27 | EQ-002, EQ-003, EQ-004, EQ-005, EQ-008, EQ-009, EQ-011, EQ-012, EQ-013, EQ-014, EQ-015, EQ-016, EQ-018, EQ-019, EQ-020, EQ-022, EQ-023, EQ-024, EQ-025, EQ-026, EQ-027, EQ-028, EQ-029, EQ-030, EQ-031, EQ-033, EQ-034, EQ-035, EQ-037, EQ-038, EQ-041, EQ-042, EQ-043, EQ-044, EQ-045, EQ-046, EQ-047 |
| **Partially justified** (grounded method but specific constants arbitrary) | 3 | EQ-001 (golden exponent), EQ-021 (Schumann valid only for environmental), EQ-036 (missing convergence proof) |
| **UNJUSTIFIED** (hard-coded without empirical or theoretical derivation) | 8 | EQ-006, EQ-007a, EQ-007b, EQ-007c, EQ-007d, EQ-010, EQ-017, EQ-032, EQ-039, EQ-040 |

### Key Academic References

| Reference | Equations |
|-----------|-----------|
| Lyapunov stability theory | EQ-002, EQ-004 |
| Tukey (1977) -- IQR method | EQ-009 |
| Naeini et al. (2015) -- ECE | EQ-013 |
| Platt (1999) -- Platt scaling | EQ-014 |
| Guo et al. (2017) -- Temperature scaling | EQ-015 |
| Vovk et al. (2005) -- Conformal prediction | EQ-018, EQ-019 |
| Gibbs & Candes (2021) -- Adaptive conformal | EQ-020 |
| Otsu (1979) -- Threshold selection | EQ-028 |
| Youden (1950) -- J statistic | EQ-030 |
| Benford (1938) -- Digit distribution | EQ-035 |
| Polyak (1964) -- Momentum SGD | EQ-037 |
| Fitts (1954); MacKenzie (1992) -- Fitts's Law | EQ-041 |
| JAMA 2016 -- SOFA score | EQ-034 |
| Newton / Classical mechanics | EQ-023, EQ-024, EQ-025, EQ-026, EQ-027 |
| Schumann resonance (geophysics) | EQ-021 |

### Action Items for Unjustified Constants

The following entries require empirical calibration, ablation studies, or published citation before deployment in safety-critical domains:

1. **EQ-001** -- The $\eta^\Phi$ exponent: Why raise ethical compliance to the golden ratio power? Needs derivation or ablation study.
2. **EQ-006** -- Benevolence threshold $\beta \geq 0.99$: No empirical basis for 0.99 vs. 0.98 or 0.995.
3. **EQ-007a--d** -- All sigma immutable thresholds: Domain-specific thresholds lack calibration data.
4. **EQ-010** -- Combined detector weights ($0.4/0.3/0.3$): Should be learned or cross-validated.
5. **EQ-017** -- Ensemble fallback weights ($0.4/0.35/0.25$): Should be learned or cross-validated.
6. **EQ-032** -- Neural-symbolic fusion weights ($0.6/0.4$): Should be task-dependent and learned.
7. **EQ-036** -- Recursive decomposition: Missing formal convergence proof for the recursive score accumulation.
8. **EQ-039** -- GOSNN ethical gate score weights ($0.4/0.4/0.2$): Ad-hoc combination without justification.
9. **EQ-040** -- GOSNN gate threshold $0.93$: No published or empirical basis for this specific value.

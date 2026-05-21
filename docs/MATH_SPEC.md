# Mercury Agent -- Formal Mathematical Specification

**Version:** 1.7.0
**Date:** 2026-05-20 (v1.7.0 release; the formal mathematical surfaces below are unchanged from the 2026-05-05 revision — v1.7 deltas are wiring/enforcement of these specs, not redefinitions)
**Status:** Living Document
**Cross-references:** top-level [`../ARCHITECTURE.md`](../ARCHITECTURE.md) §"Dual-Gate Hard Ethical Enforcement", [`ROUTING_GUIDE.md`](ROUTING_GUIDE.md)

---

## 1. Overview

Mercury Agent is a neuro-symbolic AI framework built on a mathematically
grounded hybrid-fusion architecture. Multi-paradigm anomaly detection is
one of the capabilities this AI exposes; the underlying engine combines
neural pattern recognition with explicit symbolic reasoning and hard
ethical bounding. The core mathematical framework is the **Omni-Ava
Equation (OAE)**, which combines three orthogonal detection signals --
Recursion, Resonance, and Optimization -- through golden-ratio-weighted
convex combination, modulated by a sigmoid ethical gate.

The framework guarantees:

1. **Lyapunov stability** of fusion score trajectories ($\dot{V} \leq -\lambda V$, $\lambda = 0.25$).
2. **Banach contraction** bounds on recursive decomposition ($\alpha < 1 \Rightarrow$ geometric convergence).
3. **Conformal coverage** guarantees on prediction intervals (finite-sample valid).
4. **Ethical gating** via continuous sigmoid benevolence function (domain-adaptive).

All scores are normalized to $[0, 1]$. All weights sum to $1.0$. All division
operations are guarded with $\varepsilon = 10^{-8}$.

**Source files:**
- `core/three_r/fusion.py` -- OAE implementation
- `core/centralized_constants.py` -- All constants and domain profiles
- `core/ethical_governor.py` -- Sigma Directive
- `core/calibration.py` -- ECE, Platt, Temperature scaling
- `core/conformal_prediction.py` -- Conformal prediction
- `detectors/statistical.py` -- Z-score, IQR, Isolation Forest
- `core/global_omni_scalar_network.py` -- GOSNN hierarchical aggregation

---

## 2. Core Equations

### 2.1 Omni-Ava Equation (OAE)

The central equation of Mercury Agent computes a fused anomaly score from three
orthogonal detection sub-systems, gated by an ethical compliance function:

$$
A = \bigl( w_R \cdot R(x) + w_H \cdot H(\omega) + w_O \cdot O(\theta) \bigr) \cdot \eta(b)^{\,p}
$$

**Components:**

| Symbol | Name | Domain | Description |
|--------|------|--------|-------------|
| $R(x)$ | Recursion score | $[0, 1]$ | Hierarchical feature extraction via recursive decomposition |
| $H(\omega)$ | Resonance score | $[0, 1]$ | Frequency-domain harmonic analysis |
| $O(\theta)$ | Optimization score | $[0, 1]$ | Adaptive enhancement / refactoring quality |
| $\eta(b)$ | Ethical gate | $(0, 1)$ | Sigmoid benevolence gate (see Section 2.1.3) |
| $p$ | Ethical exponent | $\mathbb{R}^+$ | Default $\Phi = 1.618\ldots$ (configurable) |
| $A$ | Fusion score | $[0, 1]$ | Final anomaly score |

**Implementation:** `core/three_r/fusion.py`, class `OmniAvaEquation.compute()`, lines 119--202.

#### 2.1.1 Weight Derivation (Golden Ratio Proportions)

The default weights are derived from the golden ratio $\Phi = 1.618033988749895\ldots$:

$$
\phi_{\text{sum}} = \Phi + 1 + \frac{1}{\Phi} \approx 2.8944
$$

$$
w_R = \frac{\Phi}{\phi_{\text{sum}}} \approx 0.5590, \qquad
w_H = \frac{1}{\phi_{\text{sum}}} \approx 0.3455, \qquad
w_O = \frac{1/\Phi}{\phi_{\text{sum}}} \approx 0.2135
$$

**Normalization proof:** The weights sum to unity by algebraic identity:

$$
w_R + w_H + w_O = \frac{\Phi + 1 + 1/\Phi}{\phi_{\text{sum}}} = \frac{\phi_{\text{sum}}}{\phi_{\text{sum}}} = 1.0 \quad \square
$$

> **Note:** The codebase documentation in `centralized_constants.py` records
> approximate values $w_R \approx 0.447$, $w_H \approx 0.276$, $w_O \approx 0.276$.
> The exact computed values from the code at `fusion.py` lines 106--111 are:
> $\phi_{\text{sum}} = \Phi + 1.0 + 1.0/\Phi$, and weights are $\Phi/\phi_{\text{sum}}$,
> $1.0/\phi_{\text{sum}}$, $(1.0/\Phi)/\phi_{\text{sum}}$ respectively.
> Numerically: $w_R \approx 0.5590$, $w_H \approx 0.3455$, $w_O \approx 0.2135$.
> The values 0.447/0.276/0.276 in `centralized_constants.py` line 394--396 are
> rounded approximations that do not match the exact computation. The actual
> runtime values from `fusion.py` are authoritative.

**Implementation:** `core/three_r/fusion.py`, lines 104--111.

#### 2.1.2 Ethical Exponent $p$

The ethical exponent defaults to the golden ratio $\Phi = 1.618033988749895$ but
is configurable for empirical optimization:

$$
\text{ethical\_scaling} = \eta^{\,p}, \qquad p = \Phi \text{ (default)}
$$

For $\eta \in [0.90, 0.99]$:

$$
0.90^{1.618} \approx 0.837 \leq \eta^{\,\Phi} \leq 0.99^{1.618} \approx 0.984
$$

The exponent always attenuates the fusion score (since $\eta < 1$ and $p > 0$),
enforcing the design invariant that ethical non-compliance reduces detection
output. The ethical compliance threshold is clamped to $[0.90, 0.99]$ at
construction.

> **UNJUSTIFIED:** The choice of $\Phi$ as the exponent lacks published
> derivation. It should be validated via parameter sweep or replaced with an
> empirically optimized value. See Section 5 for sensitivity analysis.

**Implementation:** `core/three_r/fusion.py`, line 184.

#### 2.1.3 Sigmoid Benevolence Gate

The sigmoid gate replaces the legacy hard threshold ($\beta \geq 0.99$) with a
smooth, differentiable function:

$$
\eta(b) = \frac{1}{1 + \exp\!\bigl(-k \cdot (b - b_0)\bigr)}
$$

| Parameter | Description |
|-----------|-------------|
| $b$ | Raw benevolence score, $b \in [0, 1]$ |
| $b_0$ | Inflection point (domain-specific) |
| $k$ | Steepness parameter (domain-specific) |

**Domain profiles** (from `centralized_constants.py`, lines 182--199):

| Domain | $b_0$ | $k$ | Behavior |
|--------|-------|-----|----------|
| Medical | 0.93 | 30.0 | Moderate threshold, steep transition |
| Security | 0.95 | 25.0 | High threshold, moderate transition |
| Environmental | 0.90 | 20.0 | Lower threshold, gentler transition |
| Humanitarian | 0.92 | 35.0 | Moderate threshold, steepest transition |
| Infrastructure | 0.94 | 25.0 | High threshold, moderate transition |
| Default | 0.93 | 25.0 | Moderate defaults |

**Output range:** $\eta(b) \in (0, 1)$ for all finite $b$. The function is
monotonically increasing with $\eta(b_0) = 0.5$.

**Overflow protection:** The exponent is clamped to $[-500, 500]$ before
evaluation (line 238).

**Provenance:** Logistic function (Verhulst, 1845). Domain parameters are design
choices. **UNJUSTIFIED:** Domain-specific $b_0$ and $k$ values require empirical
calibration data.

**Implementation:** `core/centralized_constants.py`, function `sigmoid_benevolence_gate()`, lines 205--240.

#### 2.1.4 OAE Output Range

**Claim:** $A \in [0, 1]$.

**Proof:** Given $R(x), H(\omega), O(\theta) \in [0, 1]$ and $w_R + w_H + w_O = 1$:

$$
0 \leq w_R \cdot R(x) + w_H \cdot H(\omega) + w_O \cdot O(\theta) \leq 1
$$

Since $\eta(b) \in (0, 1)$ and $p > 0$, we have $\eta(b)^p \in (0, 1)$.
Therefore:

$$
0 \leq A = (\text{weighted sum}) \cdot \eta^p \leq 1 \cdot 1 = 1 \quad \square
$$

#### 2.1.5 σ_Immutable Hard Gate (Wave B, PR #179)

The sigmoid benevolence gate of Section 2.1.3 attenuates the fusion
score continuously, but is **not** the only ethical guardrail. Wave B
promotes σ_Immutable to a **second mandatory hard ethical gate** at
every public decision boundary; the call aborts with
`EthicalConstraintViolationError` if either gate fails.

**Boundary contract.** Let $\mathcal{C}$ denote a public boundary
surface (e.g. `OmniMercuryEngine.detect_with_fusion`). Then for every
input $x$ on $\mathcal{C}$:

$$
\mathcal{C}(x) =
\begin{cases}
\text{prediction}(x) & \text{if } b(x) \geq \tau_b \;\wedge\; \sigma_I(\mathbf{s}(x)) \geq \tau_\sigma, \\
\text{raise } \texttt{check="benevolence"} & \text{if } b(x) < \tau_b, \\
\text{raise } \texttt{check="sigma\_immutable"} & \text{if } b(x) \geq \tau_b \wedge \sigma_I(\mathbf{s}(x)) < \tau_\sigma, \\
\text{raise } \texttt{check="gosnn\_unavailable"} & \text{if GOSNN cannot run.}
\end{cases}
$$

**σ vector layout.** The σ_Immutable gate operates on a fixed-dim
scalar state $\mathbf{s} \in \mathbb{R}^{256}$ with the following
authoritative band layout (single source of truth in
`omni_mercury_engine.security.sigma_immutable_gate`):

| Band | Indices | Role |
|------|---------|------|
| Ethical band | $[0,\;27)$ | benevolence-projected scalars, populated by `project_benevolence_to_sigma_band` |
| Active non-ethical band | $[27,\;180)$ | GOSNN omni-scalar features (cosmic, humanitarian, security, software-engineering, medical, advanced-reasoning, etc.) |
| Reserved tail | $[180,\;256)$ | zero-padded; reserved for future bands |

Constants exposed: `SIGMA_IMMUTABLE_DIM = 256`,
`SIGMA_ETHICAL_BAND_END = 27`, `SIGMA_USED_BAND_END = 180`,
`CORPUS_USED_DIM = 180` (public alias).

**Threshold.** $\tau_\sigma$ is set by the trained σ_Immutable network
against a signed corpus verdict; the corpus and the trainer share the
same band layout via the constants above. Corpus tampering at startup
fails closed across every boundary uniformly.

**Test-only bypass.** The `enable_gosnn` parameter on the public
`detect_with_fusion` / `detect_with_fusion_calibrated` surface was
renamed to the private `_enable_gosnn`; production *callers* cannot
disable σ_Immutable. Tests that need to bypass GOSNN must set the
auditable module-level flag
`omni_mercury_engine.engine._GOSNN_TESTING_BYPASS = True`.

The flag itself **is** read from production control flow
(`engine.py` consults it before σ_Immutable enforcement and again
when `_enable_gosnn=False` is requested), so the contract is "the
flag is intended for unit tests only", not "no production code
path reads it". Production deployments must therefore audit at
container-build time that
`omni_mercury_engine.engine._GOSNN_TESTING_BYPASS is False`; a
separate mypy / runtime guard
(`tests/security/test_sigma_immutable_kat.py`) asserts the flag
defaults to `False` at module import. Setting the flag in
production is a deliberate, auditable opt-out — not an
inadvertent bypass.

**Composition with the sigmoid gate.** σ_Immutable runs *after* the
benevolence sigmoid of Section 2.1.3 has been enforced (so $b \geq
\tau_b$ at the σ_Immutable check). The two gates are independent:
σ_Immutable can reject an action that the benevolence gate accepted
when the broader scalar state is incompatible with the trained
ethical manifold.

**Implementation:** `omni_mercury_engine.security.sigma_immutable_gate.SigmaImmutableGate.enforce`,
called from `OmniMercuryEngine._enforce_ethics_at_boundary`. The
`NeuroSymbolicHub` and `CognitiveOrchestrator` σ-vector builders
import the band constants from the same module to guarantee a single
layout across all boundary surfaces.

---

### 2.2 Lyapunov Stability Framework

The fusion score trajectory is bounded by a Lyapunov stability envelope,
guaranteeing that the system converges and does not exhibit unbounded oscillation.

#### 2.2.1 Stability Bound

$$
V(S_t) \leq \varepsilon \cdot e^{-\lambda t}
$$

| Parameter | Value | Description |
|-----------|-------|-------------|
| $V(S_t)$ | -- | Lyapunov function value at time step $t$ |
| $\varepsilon$ | $1.0$ | Initial bound |
| $\lambda$ | $0.25$ | Convergence rate (decay constant) |
| $t$ | $\mathbb{Z}^+$ | Discrete time step (incremented per `compute()` call) |

The bound decays exponentially: at $t = 10$, $V \leq e^{-2.5} \approx 0.082$.

**Implementation:** `core/three_r/fusion.py`, line 189.

#### 2.2.2 Stability Condition

The continuous-time Lyapunov condition is:

$$
\dot{V} \leq -\lambda V, \qquad \lambda = 0.25
$$

This guarantees that any Lyapunov function $V$ satisfying this condition
decreases at least exponentially with rate $\lambda$.

#### 2.2.3 Convergence Rate Estimation

The system estimates the empirical convergence rate from the variance ratio of
recent vs. initial fusion scores:

$$
\hat{\lambda} = -\frac{\ln(\text{ratio})}{t}, \qquad
\text{ratio} = \frac{\text{Var}(\text{recent scores})}{\text{Var}(\text{initial scores})}
$$

- If $\hat{\lambda} > 0$: the system is stable (variance is decreasing).
- If $\hat{\lambda} \leq 0$: the system is potentially unstable.

**Guard conditions:**
- If initial variance is zero, fall back to the configured $\lambda$.
- If ratio is zero or negative, fall back to the configured $\lambda$.

**Implementation:** `core/three_r/fusion.py`, method `verify_lyapunov_stability()`, lines 228--261.

---

### 2.3 Recursion with Banach Contraction Bounds

Recursive feature decomposition is bounded by the Banach fixed-point theorem
to guarantee convergence.

#### 2.3.1 Recursive Decomposition

$$
R(x, d) = f(x) + \alpha \cdot R\bigl(g(x),\; d - 1\bigr)
$$

| Symbol | Description |
|--------|-------------|
| $x$ | Input data |
| $f(x)$ | Base transformation (feature extraction at current level) |
| $g(x)$ | Recursive transformation (input reduction for next level) |
| $\alpha$ | Contraction factor, $\alpha \in (0, \alpha_{\max})$ |
| $d$ | Recursion depth, $d \in \{0, 1, \ldots, d_{\max}\}$ |

Base case: $R(x, 0) = f(x)$.

#### 2.3.2 Contraction Factor Constraint

The contraction factor $\alpha$ is constrained via sigmoid to guarantee $\alpha < 1$:

$$
\alpha = \sigma(\alpha_{\text{raw}}) \cdot \alpha_{\max}, \qquad \alpha_{\max} = 0.95
$$

where $\sigma(z) = 1/(1 + e^{-z})$ is the logistic sigmoid with overflow
protection (split computation for positive/negative inputs).

Since $\sigma(\cdot) \in (0, 1)$ and $\alpha_{\max} = 0.95 < 1$, we have:

$$
\alpha \in (0,\; 0.95) \subset (0,\; 1)
$$

This satisfies the Banach contraction mapping requirement.

**Implementation:** `core/three_r/fusion.py`, class `BanachRecursion`, lines 361--544.

#### 2.3.3 Error Bound

After $d$ iterations, the error from the true fixed point $x^*$ is bounded by:

$$
\text{err} \leq \frac{\alpha^d \cdot \| x_0 - R(x_0) \|}{1 - \alpha}
$$

For $\alpha = 0.5$ and $d = 10$: $\text{err} \leq \frac{0.5^{10}}{0.5} \cdot \|x_0 - R(x_0)\| \approx 0.002 \cdot \|x_0 - R(x_0)\|$.

For $\alpha = 0.85$ and $d = 50$: $\text{err} \leq \frac{0.85^{50}}{0.15} \cdot \|x_0 - R(x_0)\| \approx 0.0003 \cdot \|x_0 - R(x_0)\|$.

**Implementation:** `core/three_r/fusion.py`, method `BanachRecursion.compute_error_bound()`, lines 443--458.

#### 2.3.4 Convergence Proof Sketch (Banach Fixed-Point Theorem)

**Theorem (Banach, 1922):** Let $(X, d)$ be a complete metric space and
$T: X \to X$ a contraction mapping, i.e., there exists $\alpha \in [0, 1)$ such
that $d(T(x), T(y)) \leq \alpha \cdot d(x, y)$ for all $x, y \in X$. Then:

1. $T$ has a unique fixed point $x^* \in X$.
2. For any $x_0 \in X$, the sequence $x_{n+1} = T(x_n)$ converges to $x^*$.
3. Error bound: $d(x_n, x^*) \leq \frac{\alpha^n}{1 - \alpha} \cdot d(x_0, x_1)$.

**Application to Mercury Agent:**

- $T(x) = f(x) + \alpha \cdot R(g(x), d-1)$ at each recursion level.
- The sigmoid constraint guarantees $\alpha < \alpha_{\max} = 0.95 < 1$.
- Since $\alpha < 1$, the geometric series $\sum_{k=0}^{\infty} \alpha^k = \frac{1}{1-\alpha}$ converges.
- The accumulated score $R(x, d) = \sum_{k=0}^{d} \alpha^k \cdot f(g^{(k)}(x))$ is bounded by $\frac{\|f\|_\infty}{1 - \alpha}$.
- Runtime contraction monitoring halts execution if the observed contraction ratio exceeds $1.0$ (divergence detection).

**Additional safeguards:**
- Maximum depth $d_{\max} = 50$.
- Convergence tolerance $\epsilon_{\text{conv}} = 10^{-6}$: early termination when successive changes fall below this threshold.
- Contraction violation threshold: runtime halt if observed ratio $> 1.0$.

**Constants from:** `core/centralized_constants.py`, class `RecursionConvergenceConstants`, lines 300--333.

---

### 2.4 Harmonic Analysis (Domain-Adaptive)

#### 2.4.1 Harmonic Ratio

For a given domain $D$ with fundamental frequencies $\{\omega_d\}_{d \in D}$:

$$
A(x) = \frac{1}{|D|} \sum_{d \in D} \frac{\sum_{n=1}^{N} H(n \cdot \omega_d)}{\sum_{\omega} H(\omega)}
$$

where $H(\omega) = |\text{FFT}(x)|^2$ is the power spectral density (periodogram).

The harmonic ratio measures the fraction of total spectral energy concentrated at
harmonic multiples of the domain fundamental frequencies. High ratios indicate
structured periodic behavior; low ratios indicate anomalous or aperiodic signals.

**Implementation:** `detectors/spectral_vibration.py`.

#### 2.4.2 Domain Fundamental Frequencies

| Domain | Fundamental Frequencies (Hz) | Provenance |
|--------|------------------------------|------------|
| Environmental | 7.83, 14.3, 20.8, 27.3, 33.8 | Schumann resonances (Schumann, 1952) |
| Medical | 0.04, 0.15, 0.4, 1.0, 40.0 | HRV frequency bands (Task Force, 1996) |
| Infrastructure | 50.0, 60.0, 0.1, 0.01 | Power grid (50/60 Hz) + structural resonance |
| Space | 0.001, 0.01, 0.1, 11.0 | Solar cycle + orbital mechanics |
| Security | Adaptive (MUSIC/ESPRIT) | No predefined fundamentals |
| Financial | Adaptive (MUSIC/ESPRIT) | No predefined fundamentals |

**Implementation:** `core/centralized_constants.py`, class `DomainHarmonicConstants`, lines 248--270.

#### 2.4.3 Adaptive Spectral Peak Detection

For domains without predefined fundamentals (security, financial), the system
employs adaptive spectral peak detection. The `get_domain_fundamentals()` function
returns `None` for these domains, triggering MUSIC/ESPRIT-based subspace methods
to identify dominant frequencies from the data itself.

---

### 2.5 Score Calibration Equations

#### 2.5.1 Expected Calibration Error (ECE)

$$
\text{ECE} = \sum_{b=1}^{B} \frac{|\mathcal{B}_b|}{N} \cdot \bigl| \text{acc}(\mathcal{B}_b) - \text{conf}(\mathcal{B}_b) \bigr|
$$

| Parameter | Value | Description |
|-----------|-------|-------------|
| $B$ | 10 (default) | Number of probability bins |
| $\mathcal{B}_b$ | -- | Set of predictions in bin $b$ |
| $\text{acc}(\mathcal{B}_b)$ | -- | Accuracy within bin (mean of true labels) |
| $\text{conf}(\mathcal{B}_b)$ | -- | Mean predicted probability within bin |

$\text{ECE} = 0$ indicates perfect calibration. Target: $\text{ECE} < 0.05$.

**Provenance:** Naeini et al. (2015) "Obtaining Well Calibrated Probabilities Using Bayesian Binning into Quantiles."

**Implementation:** `core/calibration.py`, function `compute_ece()`, lines 62--91.

#### 2.5.2 Platt Scaling

Post-hoc logistic regression calibration:

$$
p_{\text{calibrated}} = \frac{1}{1 + e^{-(ax + b)}}
$$

where $a, b$ are learned from a held-out calibration set via logistic regression
with minimal regularization ($C = 10^{10}$, max iterations $= 100$).

**Provenance:** Platt (1999) "Probabilistic Outputs for Support Vector Machines."

**Implementation:** `core/calibration.py`, lines 144--148.

#### 2.5.3 Temperature Scaling

Single-parameter calibration by dividing logits by a learned temperature $T$:

$$
p_{\text{calibrated}} = \sigma\!\left(\frac{\text{logits}}{T}\right), \qquad T \in [0.1, 10]
$$

where $\sigma$ is the sigmoid function and $T$ is optimized via grid search over
$[0.1, 10]$ to minimize negative log-likelihood.

**Provenance:** Guo et al. (2017) "On Calibration of Modern Neural Networks."

**Implementation:** `core/calibration.py`, lines 314--316.

#### 2.5.4 Conformal Prediction Quantile

$$
q = \frac{\lceil (n + 1)(1 - \alpha) \rceil}{n}
$$

| Parameter | Description |
|-----------|-------------|
| $n$ | Calibration set size |
| $\alpha$ | Miscoverage rate (e.g., 0.05 for 95% coverage) |
| $q$ | Quantile index into sorted nonconformity scores |

The conformal prediction interval is:

$$
C(x_{\text{new}}) = \bigl[ \hat{y} - q,\; \hat{y} + q \bigr]
$$

**Finite-sample guarantee:** For exchangeable data, $P(y_{\text{new}} \in C(x_{\text{new}})) \geq 1 - \alpha$.

**Provenance:** Vovk et al. (2005) "Algorithmic Learning in a Random World."

**Implementation:** `core/conformal_prediction.py`, lines 109--111 (quantile), lines 144--145 (interval).

#### 2.5.5 Adaptive Conformal Inference

Online threshold update for distribution shift:

$$
\theta_{t+1} = \theta_t + \text{lr} \cdot (\text{miscov}_t - \alpha)
$$

where $\text{miscov}_t \in \{0, 1\}$ indicates whether the true value fell
outside the prediction interval at time $t$. This adjusts the threshold to
maintain marginal coverage under distribution shift.

**Provenance:** Gibbs & Candes (2021) "Adaptive Conformal Inference Under Distribution Shift."

**Implementation:** `core/conformal_prediction.py`, lines 313--316.

---

### 2.6 Statistical Detection

#### 2.6.1 Z-Score

$$
z = \frac{x - \mu}{\sigma + \varepsilon}, \qquad \varepsilon = 10^{-8}
$$

- Anomaly threshold: $|z| > 3.0$ (Gaussian 99.7% rule).
- Epsilon guard on standard deviation prevents division by zero.

**Normalized intensity:**

$$
I_z = \frac{\text{clip}\!\left(\frac{\max(|z|)}{z_{\text{threshold}} + \varepsilon},\; 0,\; 3\right)}{3}
$$

Maps the maximum z-score to $[0, 1]$ for fusion compatibility.

**Implementation:** `detectors/statistical.py`, lines 107, 117--118, 179--180.

#### 2.6.2 IQR Method

$$
T_{\text{upper}} = Q_3 + 1.5 \cdot \text{IQR}, \qquad \text{IQR} = Q_3 - Q_1
$$

Points above $T_{\text{upper}}$ or below $Q_1 - 1.5 \cdot \text{IQR}$ are flagged as anomalies.

**Provenance:** Tukey (1977) "Exploratory Data Analysis."

**Implementation:** `detectors/statistical.py`, lines 108--109.

#### 2.6.3 Adaptive Contamination Estimation

$$
c = \text{clip}\!\left(\text{outlier\_frac} \times 2 + 0.001,\; 0.001,\; 0.5\right)
$$

where $\text{outlier\_frac}$ is the fraction of training samples with $|z| > 3$ on any feature.
This provides data-driven contamination for the Isolation Forest.

**Implementation:** `detectors/statistical.py`, lines 116--120.

#### 2.6.4 Combined Statistical Score

$$
S = 0.4 \cdot z_{\text{score}} + 0.3 \cdot \text{iqr}_{\text{score}} + 0.3 \cdot \text{if}_{\text{score}}
$$

where each component is normalized to $[0, 1]$.

> **UNJUSTIFIED:** Weights $0.4/0.3/0.3$ are hard-coded without empirical
> optimization. Should be learned via cross-validation or domain-specific tuning.

**Implementation:** `detectors/statistical.py`, line 197.

---

### 2.7 Ethical Governance

#### 2.7.1 Sigma Directive Weighted Score

$$
S = \frac{J \cdot w_J + A \cdot w_A + C \cdot w_C + T \cdot w_T}{\sum_i w_i}
$$

| Component | Symbol | Description |
|-----------|--------|-------------|
| Justice | $J$ | Fairness and anti-discrimination score |
| Altruism | $A$ | Societal benefit score |
| Compassion | $C$ | Harm prevention score |
| Truth | $T$ | Transparency and honesty score |

All components $\in [0, 1]$. Weights $w_J, w_A, w_C, w_T$ are sourced from
`EthicalScalars` configuration (approximately equal by default).

**Gate threshold:** $S \geq 0.8$ (from `SIGMA_DIRECTIVE_THRESHOLD`).

**Implementation:** `core/ethical_governor.py`, class `SigmaDirective.apply_directive()`, lines 129--134.

#### 2.7.2 Ethical Gate Thresholds by Domain

| Domain | $\sigma_{\text{immutable}}$ | Description |
|--------|---------------------------|-------------|
| Default | 0.96 | Standard operations |
| Medical | 0.93 | Lower for medical urgency |
| Infrastructure | 0.995 | Highest for critical systems |
| Humanitarian | 0.95 | Humanitarian response |

> **UNJUSTIFIED:** All sigma immutable thresholds are design choices without
> published calibration data. Each requires domain-specific validation.

**Implementation:** `core/centralized_constants.py`, class `EthicalConstants`, lines 95--137.

#### 2.7.3 Benevolence Immutable

Legacy hard threshold (superseded by sigmoid gate in Section 2.1.3):

$$
\text{pass} = \begin{cases} \text{true} & \text{if } \beta \geq 0.99 \\ \text{false} & \text{otherwise} \end{cases}
$$

> **DESIGN NOTE:** The hard threshold creates a discontinuity at $\beta = 0.99$.
> The sigmoid benevolence gate (Section 2.1.3) is the recommended replacement.

**Implementation:** `core/centralized_constants.py`, line 118.

---

### 2.8 Hierarchical Omni-Scalar Aggregation (GOSNN)

The Global Omni-Scalar Neural Network (GOSNN) aggregates approximately 180
scalars across 8 categories into a single intelligence score through a
three-level hierarchy.

#### 2.8.1 Level 1: Category Grouping

Scalars are assigned to one of 8 groups:

| Category | Approximate Scalar Count | Description |
|----------|--------------------------|-------------|
| Ethical | ~27 | Core ethical values and constraints |
| Cosmic | ~7 | Universe-scale alignment |
| Quantum Consciousness | ~7 | Quantum-inspired processing |
| Humanitarian | ~9 | Crisis response and welfare |
| Security | ~6 | Threat detection and defense |
| Software Engineering | ~45 | Code quality and 3R synergy |
| Medical | ~10 | Healthcare and diagnostics |
| Advanced Reasoning | ~15 | Logic, inference, knowledge synthesis |

**Implementation:** `core/global_omni_scalar_network.py`, class `ScalarGroup`, lines 92--113.

#### 2.8.2 Level 2: Within-Category Score (NumPy Fallback)

When PyTorch is unavailable, the ethical gate score is computed as:

$$
S_{\text{gate}} = 0.4 \cdot r_+ + 0.4 \cdot \min\!\left(\frac{\mu}{2},\; 1\right) + 0.2 \cdot \frac{1}{1 + \sigma_s}
$$

where:
- $r_+$ = fraction of scalar values $> 1.0$ (positive ratio)
- $\mu$ = mean of scalar vector
- $\sigma_s$ = standard deviation of scalar vector

**Gate threshold:** $S_{\text{gate}} \geq 0.93$.

> **UNJUSTIFIED:** Weights $0.4/0.4/0.2$ and threshold $0.93$ are ad-hoc design
> choices without empirical basis.

**Implementation:** `core/global_omni_scalar_network.py`, lines 188--197 (score), line 145 (threshold).

#### 2.8.3 Level 3: Cross-Category Aggregation

When PyTorch is available, the GOSNN uses a neural network gate:
$256 \to 64 \to 1$ with ReLU hidden activation and sigmoid output. The network
output is compared against the $0.93$ threshold to determine ethical compliance.

For the hierarchical omni-scalar aggregation across categories, the system
computes weighted combinations where category weights reflect operational
importance (safety, fairness, transparency, accountability, beneficence).

The five ethical pillars are:

1. **Safety** -- minimizing risk of harm
2. **Fairness** -- equitable treatment across groups
3. **Transparency** -- explainability of decisions
4. **Accountability** -- audit trail and responsibility
5. **Beneficence** -- net positive societal impact

---

## 3. Parameter Justification Table

| Parameter | Value | Location | Justification | Source |
|-----------|-------|----------|---------------|--------|
| $\Phi$ (golden ratio) | 1.618033988749895 | `centralized_constants.py:38` | Mathematical constant | Exact: $(1 + \sqrt{5})/2$ |
| $\varepsilon$ (numerical stability) | $10^{-8}$ | `centralized_constants.py:52` | Standard floating-point guard | IEEE 754 practice |
| $\varepsilon_{\text{small}}$ | $10^{-10}$ | `centralized_constants.py:54` | Tighter guard for sensitive divisions | IEEE 754 practice |
| $\lambda$ (Lyapunov rate) | 0.25 | `centralized_constants.py:72` | Controls convergence speed | **UNJUSTIFIED:** design choice, needs empirical validation |
| $\varepsilon_{\text{Lyapunov}}$ (initial bound) | 1.0 | `centralized_constants.py:76` | Unit initial bound | Convention |
| Stability window | 10 | `centralized_constants.py:80` | Samples for stability check | **UNJUSTIFIED:** design choice |
| $\sigma_{\text{default}}$ | 0.96 | `centralized_constants.py:102` | Default ethical threshold | **UNJUSTIFIED:** needs domain calibration |
| $\sigma_{\text{medical}}$ | 0.93 | `centralized_constants.py:106` | Medical ethical threshold | **UNJUSTIFIED:** needs clinical validation |
| $\sigma_{\text{infrastructure}}$ | 0.995 | `centralized_constants.py:110` | Infrastructure ethical threshold | **UNJUSTIFIED:** needs operational validation |
| $\sigma_{\text{humanitarian}}$ | 0.95 | `centralized_constants.py:114` | Humanitarian ethical threshold | **UNJUSTIFIED:** needs field validation |
| $\beta_{\text{immutable}}$ | 0.99 | `centralized_constants.py:118` | Benevolence hard threshold | **UNJUSTIFIED:** needs empirical basis for 0.99 vs. alternatives |
| Sigma Directive threshold | 0.8 | `centralized_constants.py:126` | Ethical gate pass/fail | **UNJUSTIFIED:** design choice |
| Bias detection threshold | 0.1 | `centralized_constants.py:130` | Demographic parity max diff | Fairlearn convention |
| $b_0$ (Medical sigmoid) | 0.93 | `centralized_constants.py:182` | Sigmoid inflection point | **UNJUSTIFIED:** needs clinical calibration |
| $k$ (Medical sigmoid) | 30.0 | `centralized_constants.py:183` | Sigmoid steepness | **UNJUSTIFIED:** needs calibration |
| $b_0$ (Security sigmoid) | 0.95 | `centralized_constants.py:185` | Sigmoid inflection point | **UNJUSTIFIED:** needs security domain calibration |
| $k$ (Security sigmoid) | 25.0 | `centralized_constants.py:186` | Sigmoid steepness | **UNJUSTIFIED:** needs calibration |
| $b_0$ (Environmental sigmoid) | 0.90 | `centralized_constants.py:188` | Sigmoid inflection point | **UNJUSTIFIED:** needs calibration |
| $k$ (Environmental sigmoid) | 20.0 | `centralized_constants.py:189` | Sigmoid steepness | **UNJUSTIFIED:** needs calibration |
| $b_0$ (Humanitarian sigmoid) | 0.92 | `centralized_constants.py:191` | Sigmoid inflection point | **UNJUSTIFIED:** needs calibration |
| $k$ (Humanitarian sigmoid) | 35.0 | `centralized_constants.py:192` | Sigmoid steepness | **UNJUSTIFIED:** needs calibration |
| $b_0$ (Infrastructure sigmoid) | 0.94 | `centralized_constants.py:194` | Sigmoid inflection point | **UNJUSTIFIED:** needs calibration |
| $k$ (Infrastructure sigmoid) | 25.0 | `centralized_constants.py:195` | Sigmoid steepness | **UNJUSTIFIED:** needs calibration |
| $b_0$ (Default sigmoid) | 0.93 | `centralized_constants.py:197` | Sigmoid inflection point | **UNJUSTIFIED:** needs calibration |
| $k$ (Default sigmoid) | 25.0 | `centralized_constants.py:198` | Sigmoid steepness | **UNJUSTIFIED:** needs calibration |
| $\alpha_{\max}$ (recursion) | 0.95 | `centralized_constants.py:317` | Max contraction factor | Design: must be $< 1$ for Banach theorem |
| $d_{\max}$ (recursion depth) | 50 | `centralized_constants.py:324` | Max recursion depth | **UNJUSTIFIED:** design choice, needs profiling |
| $\epsilon_{\text{conv}}$ (convergence) | $10^{-6}$ | `centralized_constants.py:327` | Early termination tolerance | Standard numerical tolerance |
| Contraction violation threshold | 1.0 | `centralized_constants.py:330` | Divergence detection | Theoretical: contraction ratio must be $< 1$ |
| $z_{\text{threshold}}$ | 3.0 | `centralized_constants.py:363` | Z-score anomaly threshold | Gaussian theory: 99.7% coverage |
| IQR multiplier | 1.5 | `centralized_constants.py:367` | IQR fence multiplier | Tukey (1977) |
| MAD multiplier | 3.0 | `centralized_constants.py:375` | MAD-based threshold | Robust statistics convention |
| $w_R$ (OAE Recursion) | $\Phi / \phi_{\text{sum}} \approx 0.559$ | `fusion.py:108` | Golden ratio proportion | Mathematically grounded |
| $w_H$ (OAE Harmonic) | $1 / \phi_{\text{sum}} \approx 0.346$ | `fusion.py:109` | Golden ratio proportion | Mathematically grounded |
| $w_O$ (OAE Optimization) | $(1/\Phi) / \phi_{\text{sum}} \approx 0.214$ | `fusion.py:110` | Golden ratio proportion | Mathematically grounded |
| $p$ (ethical exponent) | $\Phi = 1.618$ | `fusion.py:96` | Ethical scaling power | **UNJUSTIFIED:** needs parameter sweep |
| Statistical fusion weights | 0.4 / 0.3 / 0.3 | `statistical.py:197` | Z / IQR / IF combination | **UNJUSTIFIED:** needs cross-validation |
| Neural-symbolic weights | 0.6 / 0.4 | `centralized_constants.py:400-401` | Neural vs. symbolic | **UNJUSTIFIED:** needs empirical tuning |
| Ensemble decay | 0.9 | `centralized_constants.py:411` | Temporal weight decay $w_t = 0.9^t$ | Exponential decay convention |
| GOSNN gate weights | 0.4 / 0.4 / 0.2 | `global_omni_scalar_network.py:192-196` | Ethical gate score | **UNJUSTIFIED:** ad-hoc combination |
| GOSNN gate threshold | 0.93 | `global_omni_scalar_network.py:145` | Ethical compliance | **UNJUSTIFIED:** needs validation |
| ECE bins | 10 | `calibration.py:62` | Calibration evaluation | Naeini et al. (2015) |
| Platt regularization $C$ | $10^{10}$ | `calibration.py:144` | Minimal regularization | Platt (1999) |
| Temperature range | [0.1, 10] | `calibration.py:314` | Temperature search bounds | Guo et al. (2017) |
| Weight update LR | 0.01 | `fusion.py:207` | Attention weight EMA rate | Standard EMA practice |
| Optimizer LR | 0.01 | `fusion.py:274` | Weight optimization | Standard SGD practice |
| Optimizer momentum | 0.9 | `fusion.py:275` | Momentum coefficient | Polyak (1964) |
| Weight decay | $10^{-4}$ | `fusion.py:276` | L2 regularization | Standard regularization |

---

## 4. Convergence Proofs

### 4.1 Banach Contraction Mapping Proof

**Claim:** The recursive computation $R(x, d) = f(x) + \alpha \cdot R(g(x), d-1)$
converges to a unique fixed point for $\alpha \in (0, 0.95)$.

**Proof sketch:**

1. Define operator $T: \mathcal{F} \to \mathcal{F}$ on the space of bounded
   functions by $(Th)(x) = f(x) + \alpha \cdot h(g(x))$.

2. For any $h_1, h_2 \in \mathcal{F}$:
   $$\|Th_1 - Th_2\|_\infty = \|\alpha \cdot (h_1 \circ g - h_2 \circ g)\|_\infty \leq \alpha \cdot \|h_1 - h_2\|_\infty$$

3. Since $\alpha < 0.95 < 1$, $T$ is a contraction on $(\mathcal{F}, \|\cdot\|_\infty)$.

4. By the Banach fixed-point theorem, $T$ has a unique fixed point $h^*$, and
   the iteration $h_{n+1} = Th_n$ converges geometrically:
   $$\|h_n - h^*\|_\infty \leq \frac{\alpha^n}{1 - \alpha} \cdot \|h_1 - h_0\|_\infty$$

5. The sigmoid constraint on $\alpha_{\text{raw}}$ ensures $\alpha \in (0, 0.95)$
   for any real-valued input, making divergence impossible under the model. $\square$

**Runtime safeguard:** If the empirical contraction ratio exceeds $1.0$ at any
step, the system raises `RuntimeError` and halts recursion (line 534).

### 4.2 Lyapunov Stability Proof

**Claim:** If $\dot{V} \leq -\lambda V$ with $\lambda = 0.25 > 0$, then
$V(t) \leq V(0) \cdot e^{-\lambda t}$.

**Proof sketch:**

1. The differential inequality $\dot{V} + \lambda V \leq 0$ can be solved
   by the integrating factor $e^{\lambda t}$:
   $$\frac{d}{dt}\bigl[V(t) \cdot e^{\lambda t}\bigr] = (\dot{V} + \lambda V) \cdot e^{\lambda t} \leq 0$$

2. Integrating from $0$ to $t$:
   $$V(t) \cdot e^{\lambda t} \leq V(0)$$
   $$V(t) \leq V(0) \cdot e^{-\lambda t}$$

3. With $V(0) = \varepsilon = 1.0$ and $\lambda = 0.25$:
   $$V(t) \leq e^{-0.25 t} \quad \square$$

**Discrete-time note:** The implementation uses discrete time steps. The bound
$V(S_t) \leq \varepsilon \cdot e^{-\lambda t}$ is computed at each step but is
a tracking bound, not a proven invariant of the discrete system. The empirical
convergence rate estimation (Section 2.2.3) provides runtime verification.

### 4.3 OAE Weight Normalization Proof

**Claim:** $w_R + w_H + w_O = 1.0$ for the golden ratio default initialization.

**Proof:**

$$
w_R + w_H + w_O = \frac{\Phi}{\phi_{\text{sum}}} + \frac{1}{\phi_{\text{sum}}} + \frac{1/\Phi}{\phi_{\text{sum}}} = \frac{\Phi + 1 + 1/\Phi}{\phi_{\text{sum}}}
$$

Since $\phi_{\text{sum}} = \Phi + 1 + 1/\Phi$ by definition:

$$
w_R + w_H + w_O = \frac{\phi_{\text{sum}}}{\phi_{\text{sum}}} = 1 \quad \square
$$

For non-default weights, the constructor normalizes: `{k: v / total for k, v in initial_weights.items()}` (line 114). The `update_weights()` method also
re-normalizes after each update (line 226).

---

## 5. Sensitivity Analysis

### 5.1 OAE Ethical Exponent $p$

Perturbation of $\pm 10\%$ around $\Phi = 1.618$:

| $p$ | $0.93^p$ | $0.96^p$ | $0.99^p$ | Impact |
|-----|----------|----------|----------|--------|
| 1.456 ($-10\%$) | 0.898 | 0.942 | 0.985 | Higher scores, less ethical penalty |
| 1.618 (default) | 0.890 | 0.936 | 0.984 | Baseline |
| 1.780 ($+10\%$) | 0.882 | 0.930 | 0.982 | Lower scores, more ethical penalty |

**Impact:** A $\pm 10\%$ change in $p$ produces a $\pm 0.6\%$ to $\pm 0.8\%$
change in the ethical scaling factor for typical $\eta$ values. The system is
**moderately insensitive** to the exponent choice in the operational range
$\eta \in [0.93, 0.99]$.

### 5.2 Benevolence Sigmoid Parameters

#### Inflection point $b_0 \pm 0.02$:

For the default profile ($b_0 = 0.93$, $k = 25$):

| $b$ | $\eta(b; b_0=0.91)$ | $\eta(b; b_0=0.93)$ | $\eta(b; b_0=0.95)$ |
|-----|----------------------|----------------------|----------------------|
| 0.90 | 0.44 | 0.32 | 0.22 |
| 0.93 | 0.62 | 0.50 | 0.38 |
| 0.95 | 0.73 | 0.62 | 0.50 |
| 0.99 | 0.88 | 0.82 | 0.73 |

**Impact:** Shifting $b_0$ by $\pm 0.02$ changes the gate value by approximately
$\pm 0.10$ to $\pm 0.12$ at the operating point. This is a **significant**
sensitivity -- domain-specific calibration is critical.

#### Steepness $k \pm 5$:

For the default profile ($b_0 = 0.93$, $k = 25$):

| $b$ | $\eta(b; k=20)$ | $\eta(b; k=25)$ | $\eta(b; k=30)$ |
|-----|------------------|------------------|------------------|
| 0.90 | 0.35 | 0.32 | 0.29 |
| 0.93 | 0.50 | 0.50 | 0.50 |
| 0.95 | 0.60 | 0.62 | 0.65 |
| 0.99 | 0.77 | 0.82 | 0.86 |

**Impact:** The steepness $k$ controls the sharpness of the transition. Higher
$k$ creates a more step-like function. The sensitivity is **moderate** --
variations of $\pm 5$ in $k$ change gate values by approximately $\pm 0.03$ to
$\pm 0.05$ away from the inflection point.

### 5.3 Recursion Contraction Factor $\alpha$

Perturbation of $\pm 10\%$ around $\alpha = 0.5$ (mid-range):

| $\alpha$ | Error bound at $d=10$ | Error bound at $d=50$ | Convergence speed |
|----------|-----------------------|-----------------------|-------------------|
| 0.45 ($-10\%$) | $6.2 \times 10^{-4}$ | $1.8 \times 10^{-18}$ | Faster |
| 0.50 (baseline) | $1.95 \times 10^{-3}$ | $1.8 \times 10^{-15}$ | Baseline |
| 0.55 ($+10\%$) | $5.5 \times 10^{-3}$ | $1.2 \times 10^{-13}$ | Slower |

Error bound: $\text{err} \leq \alpha^d / (1 - \alpha)$ (normalized to unit initial displacement).

**Impact:** A $\pm 10\%$ change in $\alpha$ produces an order-of-magnitude change
in the error bound at $d = 10$. The system is **highly sensitive** to the
contraction factor, which justifies the sigmoid constraint.

### 5.4 Lyapunov Convergence Rate $\lambda$

Perturbation of $\pm 20\%$ around $\lambda = 0.25$:

| $\lambda$ | $V(t=10)$ | $V(t=20)$ | $V(t=50)$ | Half-life $t_{1/2}$ |
|-----------|-----------|-----------|-----------|---------------------|
| 0.20 ($-20\%$) | 0.135 | 0.018 | $4.5 \times 10^{-5}$ | 3.47 |
| 0.25 (baseline) | 0.082 | 0.0067 | $3.7 \times 10^{-6}$ | 2.77 |
| 0.30 ($+20\%$) | 0.050 | 0.0025 | $3.1 \times 10^{-7}$ | 2.31 |

Half-life: $t_{1/2} = \ln(2) / \lambda$.

**Impact:** A $\pm 20\%$ change in $\lambda$ produces a $\pm 25\%$ change in
the effective convergence rate (as measured by half-life). The system is
**moderately sensitive** -- faster decay provides tighter bounds but may be
too aggressive for slowly-changing signals.

---

## 6. Numerical Stability Guarantees

### 6.1 Division by Zero Guards

All division operations in the codebase use epsilon guards to prevent
division by zero:

| Location | Guard | Value |
|----------|-------|-------|
| Z-score computation | $\sigma + \varepsilon$ | $\varepsilon = 10^{-8}$ |
| Z-score intensity normalization | $z_{\text{thresh}} + \varepsilon$ | $\varepsilon = 10^{-8}$ |
| Weight normalization | $\sum w + \varepsilon$ | $\varepsilon = 10^{-10}$ |
| Score range normalization | $\max - \min + \varepsilon$ | $\varepsilon = 10^{-10}$ |
| OAE weight optimizer | $\sum |w| + \varepsilon$ | $\varepsilon = 10^{-10}$ |
| Standard deviation in calibration | $\sigma + \varepsilon$ | $\varepsilon = 10^{-10}$ |

**Central epsilon constant:** `MATH.EPSILON = 1e-8` (general), `MATH.EPSILON_SMALL = 1e-10` (sensitive).

**Implementation:** `core/centralized_constants.py`, lines 52--54.

### 6.2 Overflow Protection in Sigmoid/Exponential Functions

#### Sigmoid benevolence gate:

The exponent in $\eta(b) = 1/(1 + \exp(-k(b - b_0)))$ is clamped:

```
exponent = max(-500.0, min(500.0, exponent))
```

This prevents `math.exp()` overflow for extreme inputs. For $|x| \leq 500$,
$e^x$ is within float64 range ($e^{500} \approx 1.4 \times 10^{217} < 1.8 \times 10^{308}$).

**Implementation:** `core/centralized_constants.py`, line 238.

#### Banach recursion sigmoid:

The `BanachRecursion._sigmoid()` uses a split computation for numerical stability:

$$
\sigma(x) = \begin{cases}
\frac{1}{1 + e^{-x}} & \text{if } x \geq 0 \\
\frac{e^x}{1 + e^x} & \text{if } x < 0
\end{cases}
$$

This avoids computing $e^{|x|}$ for large $|x|$, preventing overflow in both
branches.

**Implementation:** `core/three_r/fusion.py`, lines 421--429.

#### Lyapunov bound:

$e^{-\lambda t}$ with $\lambda = 0.25$ and $t \geq 0$ always produces values
in $(0, 1]$. No overflow risk.

### 6.3 NaN Propagation Prevention

| Location | Guard | Behavior |
|----------|-------|----------|
| OAE `compute()` | `np.isnan()` check on each input | Replace NaN with 0.0, log warning |
| Score calibration | `_validate_scores()` | Replace NaN/Inf with median |
| Statistical detector | `np.isfinite()` mask | Filter non-finite rows before fitting |
| Conformal prediction | `np.fill_diagonal(distances, np.inf)` | Intentional self-exclusion (not a bug) |
| GOSNN ethical score | `np.clip(score, 0.0, 1.0)` | Clamp output to valid range |

**Known gaps (from correctness report):**
- `core/three_r/fusion.py`: No NaN check on convergence history before variance computation.
- `core/global_omni_scalar_network.py`: No NaN guard on scalar values before GOSNN aggregation.

---

## 7. Domain-Specific Parameter Profiles

### 7.1 Comprehensive Domain Parameter Table

| Parameter | Default | Medical | Security | Environmental | Humanitarian | Infrastructure |
|-----------|---------|---------|----------|---------------|--------------|----------------|
| $\sigma_{\text{immutable}}$ | 0.96 | 0.93 | 0.96 | 0.96 | 0.95 | 0.995 |
| Sigmoid $b_0$ | 0.93 | 0.93 | 0.95 | 0.90 | 0.92 | 0.94 |
| Sigmoid $k$ | 25.0 | 30.0 | 25.0 | 20.0 | 35.0 | 25.0 |
| Harmonic fundamentals (Hz) | Env. default | 0.04, 0.15, 0.4, 1.0, 40.0 | Adaptive | 7.83, 14.3, 20.8, 27.3, 33.8 | Env. default | 50, 60, 0.1, 0.01 |
| Harmonic detection method | FFT | FFT | MUSIC/ESPRIT | FFT | FFT | FFT |

### 7.2 Medical Domain Constants

| Parameter | Value | Unit | Provenance |
|-----------|-------|------|------------|
| Heart rate range | 60--100 | bpm | Standard adult range |
| Systolic BP range | 90--140 | mmHg | AHA guidelines |
| Diastolic BP range | 60--90 | mmHg | AHA guidelines |
| Respiratory rate range | 12--20 | breaths/min | Standard adult range |
| SpO2 range | 95--100 | % | Standard clinical range |
| Temperature range | 36.1--37.8 | C | Standard adult range |
| MAP range | 70--105 | mmHg | Standard clinical range |
| SOFA weights | resp=0.20, coag=0.15, liver=0.15, cardio=0.20, CNS=0.15, renal=0.15 | -- | JAMA 2016 |
| Alert fatigue window | 300 | seconds | Design choice |

**Implementation:** `core/centralized_constants.py`, class `MedicalDomainConstants`, lines 448--480.

### 7.3 Financial Domain Constants

| Parameter | Value | Provenance |
|-----------|-------|------------|
| Benford's Law $P(d) = \log_{10}(1 + 1/d)$ | [0.301, 0.176, 0.125, 0.097, 0.079, 0.067, 0.058, 0.051, 0.046] | Benford (1938) |
| Harmonic detection | Adaptive (MUSIC/ESPRIT) | No predefined fundamentals |

**Implementation:** `core/centralized_constants.py`, class `FinancialDomainConstants`, lines 483--497.

### 7.4 Space Domain Constants

| Parameter | Value | Provenance |
|-----------|-------|------------|
| Harmonic fundamentals | 0.001, 0.01, 0.1, 11.0 Hz | Solar cycle + orbital mechanics |

**Implementation:** `core/centralized_constants.py`, line 266.

---

## Appendix A: Academic References

| Reference | Year | Used In |
|-----------|------|---------|
| Banach, S. "Sur les operations dans les ensembles abstraits..." | 1922 | Section 2.3 (contraction mapping) |
| Benford, F. "The Law of Anomalous Numbers." | 1938 | Section 7.3 (financial fraud detection) |
| Verhulst, P.-F. (logistic function) | 1845 | Section 2.1.3 (sigmoid gate) |
| Fitts, P. "The information capacity of the human motor system..." | 1954 | UI/UX anomaly detection (EQ-041) |
| Tukey, J. "Exploratory Data Analysis." | 1977 | Section 2.6.2 (IQR method) |
| Otsu, N. "A Threshold Selection Method from Gray-Level Histograms." | 1979 | Score calibration (EQ-028) |
| Youden, W. "Index for rating diagnostic tests." | 1950 | Threshold optimization (EQ-030) |
| Platt, J. "Probabilistic Outputs for Support Vector Machines." | 1999 | Section 2.5.2 (Platt scaling) |
| Vovk, V. et al. "Algorithmic Learning in a Random World." | 2005 | Section 2.5.4 (conformal prediction) |
| Naeini, M. et al. "Obtaining Well Calibrated Probabilities..." | 2015 | Section 2.5.1 (ECE) |
| Guo, C. et al. "On Calibration of Modern Neural Networks." | 2017 | Section 2.5.3 (temperature scaling) |
| Vaswani, A. et al. "Attention Is All You Need." | 2017 | GOSNN multi-head attention |
| Gibbs, I. & Candes, E. "Adaptive Conformal Inference Under Distribution Shift." | 2021 | Section 2.5.5 (adaptive conformal) |
| Polyak, B. "Some methods of speeding up the convergence..." | 1964 | Optimizer momentum |
| Schumann, W. (Schumann resonances) | 1952 | Section 2.4.2 (environmental harmonics) |
| Task Force of ESC/NASPE (HRV frequency bands) | 1996 | Section 2.4.2 (medical harmonics) |
| JAMA (SOFA score) | 2016 | Section 7.2 (medical SOFA weights) |

---

## Appendix B: Notation Index

| Symbol | Definition | First Appearance |
|--------|-----------|------------------|
| $A$ | OAE fusion score | Section 2.1 |
| $\alpha$ | Recursion contraction factor | Section 2.3 |
| $\alpha_{\max}$ | Maximum contraction factor (0.95) | Section 2.3.2 |
| $b$ | Benevolence score | Section 2.1.3 |
| $b_0$ | Sigmoid inflection point | Section 2.1.3 |
| $\beta$ | Legacy benevolence threshold | Section 2.7.3 |
| $d$ | Recursion depth | Section 2.3.1 |
| $\varepsilon$ | Numerical stability epsilon ($10^{-8}$) | Section 1 |
| $\eta(b)$ | Sigmoid benevolence gate function | Section 2.1.3 |
| $\Phi$ | Golden ratio ($1.618\ldots$) | Section 2.1.1 |
| $\phi_{\text{sum}}$ | Sum $\Phi + 1 + 1/\Phi$ ($\approx 2.894$) | Section 2.1.1 |
| $H(\omega)$ | Resonance/harmonic score or power spectrum | Section 2.1, 2.4 |
| $k$ | Sigmoid steepness parameter | Section 2.1.3 |
| $\lambda$ | Lyapunov convergence rate (0.25) | Section 2.2 |
| $O(\theta)$ | Optimization/refactoring score | Section 2.1 |
| $p$ | Ethical exponent (default $\Phi$) | Section 2.1.2 |
| $q$ | Conformal prediction quantile | Section 2.5.4 |
| $R(x)$ | Recursion score | Section 2.1 |
| $\sigma$ | Standard deviation (context-dependent) | Section 2.6.1 |
| $\sigma_{\text{immutable}}$ | Ethical compliance threshold | Section 2.7.2 |
| $S$ | Sigma Directive weighted score | Section 2.7.1 |
| $T$ | Temperature scaling parameter | Section 2.5.3 |
| $V$ | Lyapunov function | Section 2.2 |
| $w_R, w_H, w_O$ | OAE component weights | Section 2.1.1 |
| $z$ | Z-score | Section 2.6.1 |

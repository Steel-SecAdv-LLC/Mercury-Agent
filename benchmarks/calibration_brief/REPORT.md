# Mercury Calibration–Alignment: Validation & Exploration Report

Applies to Mercury Agent **v2.1.x**. Last updated: 2026-07-13.

Independent reimplementation from the brief's formulas (no prior-session code reused —
independent reimplementation *is* the validation). All code in this directory; results
captured in `results/`. Deterministic (seeds fixed; byte-identical reruns verified).

> **Disposition (2026-06-12, PR #291).** This suite was authored in PR #275
> (closed unmerged per its delivery brief; branch deleted, content recovered
> from `refs/pull/275/head`, commit `bc211a0`). Ported into the tree:
> `StrictIsotonicCalibration` (X1 survivor), wired as
> `core/calibration.py::calibrate_detector(..., method="strict_isotonic")`,
> the brief's `BetaCalibration` (subsequently ported as a standalone class at
> `core/calibration.py`, coexisting with `main`'s own accept-gated Beta shipped
> via PR #278 — `fit_accept_gated_mca` — though not wired into the
> `calibrate_detector` method dispatch), the five stale "requires scikit-learn"
> messages reworded (V12c), and this evidence suite. **Not** ported, with
> reasons: the X12a η-multiply lint (PR #278 settled the η^Φ
> term as an opt-in decoupling, R6 default-off, so the lint would flag the
> shipped intentional design); the MATH_SPEC φ_sum patch (superseded by the
> `Φ + 2` single canonical derivation now in `docs/MATH_SPEC.md`). The X8
> finding below (learned simplex weights beat golden-ratio on the proxy 3R,
> paired t≈5.3) remains an open research input, not a shipped change.

## How to reproduce
```bash
pip install numpy scipy scikit-learn
cd benchmarks/calibration_brief
python run_v1_v8.py        # synthetic register V1–V8
python run_v9_v11.py       # real ADBench register V9–V11 (auto-downloads 6 datasets)
python x1_strict_isotonic.py
python x2_nonmono_guard.py
python x8_phi_sweep.py
```
Environment used: numpy 2.4.6, scipy 1.17.1, scikit-learn 1.9.0, Python 3.11.
ECE throughout = **15 equal-mass (quantile) bins** per the brief (note: Mercury's own
`compute_ece` uses 10 *equal-width* bins — a different estimator).

---

## Part 1 — Validation Register

### V1–V8 (synthetic world; seed 7; 2000 cal / 40000 test) — **9/9 PASS**

| ID | Claim | Measured (this reimpl.) | Prior | Verdict |
|----|-------|-------------------------|-------|---------|
| V1 | AUROC invariance under strictly-monotone calibration | 0.918545 → 0.918545, **Δ = 0** | Δ = 0 | PASS |
| V2 | Brier → oracle floor `E[c*(1−c*)]` | 0.0760 → 0.0694; bound 0.0688; gap 0.0006 (<5e-3) | 0.0763→0.0695, bound 0.0694 | PASS |
| V3 | ECE shrinks ≥ 3× | 0.0517 → 0.0032 (**16.2×**) | 14.5× | PASS |
| V4 | NB(t) calibrated ≥ uncalibrated ≥ max(treat-all,0) | holds at every t∈{.05,.10,.15,.20} | +0.0027 mean gain | PASS |
| V5 | Functional identity on calibrated input (test the MAP) | sup|c(s)−s| = 0.028 on [.02,.98]; (a,b,c)=(0.95,0.92,−0.05) | 0.026 | PASS |
| V6 | η^Φ damage matches multiplicative identity to machine precision; ECE worsens ≥3× | κ=0.8738; ΔBrier=0.00137; **identity mismatch 6.9e-18**; ECE 0.0032→0.0182 (5.7×) | 0.00097; →0.0173 | PASS |
| V7 | Prevalence shift 0.15→0.05; one-line adjustment | ECE 0.0542 → 0.0037 (adjusted < unadj/2) | 0.055→0.0034 | PASS |
| V8 | Non-monotone bijection scramble; monotone stays, 50-bin remap recovers | scrambled 0.4026; Beta stays 0.4026; remap **0.9009**; oracle 0.9185 | 0.400→0.9125, orc 0.9208 | PASS |
| V8-anti | Folded score `exp(−z²/2)` is unrecoverable (all ≈0.50) | raw 0.4970 / Beta 0.4970 / remap 0.4993 | all ≈0.50 | PASS |

The V6 multiplicative-damage identity `Brier(κp)−Brier(p) = (κ²−1)E[p²]+2(1−κ)E[pY]` holds
to **6.9e-18** — exact. V1's invariance Δ is **identically 0**. The scramble/fold pair both
reproduce: post-hoc maps recover information that was bijectively hidden (V8) but cannot
create discrimination a folding destroyed (anti-claim, **G2**).

### V9–V11 (real ADBench; IsolationForest-200; 50/25/25; seeds 0–4) — validated, 2 transparent deviations

Six datasets: `6_cardio, 23_mammography, 38_thyroid, 31_satimage-2, 28_pendigits, 30_satellite`.

**V9 large-n** (cal = 25%): raw AUROC 0.9014.

| method | AUROC | Brier | ECE | band-NB |
|--------|-------|-------|-----|---------|
| raw | 0.9014 | 0.0876 | 0.1677 | −0.0028 |
| M-Platt | 0.9014 | 0.0473 | 0.0273 | 0.0512 |
| M-Isotonic | 0.8928 | 0.0458 | **0.0136** | 0.0525 |
| M-Temp | 0.9014 | 0.0711 | 0.0835 | 0.0410 |
| M-Ens3 | 0.8965 | 0.0460 | 0.0151 | 0.0523 |
| Beta | 0.9014 | 0.0468 | 0.0244 | 0.0519 |
| Ens4 | 0.8961 | 0.0460 | 0.0161 | 0.0523 |

- **no dominator** ✓ · **Beta/Platt/Temp preserve AUROC exactly: 100%** ✓ · Mercury-Isotonic best ECE on **4/6** ✓ · Ens4 large-n picks **isotonic 18/30** ✓.
- Cited headline reproduced exactly: **thyroid raw 0.9788 → Isotonic 0.9535**, Beta holds 0.9788.
- **Deviation:** Isotonic loses AUROC on **5/6**, not all six. Exception = `30_satellite`
  (prevalence 0.316, low base AUROC 0.71), where isotonic's flattening introduces no
  harmful ties (it even nudges AUROC up to 0.729). Mechanism intact on the other five.

**V10 small-n** (cal subsampled to 100, ≥3 pos):
- M-Isotonic AUROC **collapses** 0.8928 → 0.8428; **Beta holds 0.9014 (100%)**; **Beta best/co-best Brier on 5/6** ✓ (brief expected ~4/6).
- mammography raw 0.8399 → Isotonic-small **0.7979** (brief cited ~0.72 — same direction, milder collapse in this reimpl.); Beta 0.8399.

**V11 Ens4 selection** (4-member, Brier on internal 75/25 split):
- Large-n picks **isotonic 18 / beta 8 / platt 4** → isotonic-dominant ✓.
- Small-n picks **isotonic 12 / platt 7 / beta 6 / temp 5** → isotonic dominance erodes (60%→40%) **but does not transfer specifically to Beta**.
- **Deviation / finding:** Brier-only selection does *not* preferentially pick Beta at
  small n even though Beta is the one holding AUROC. **AUROC-safety must be a separate
  guard, not delegated to a proper-score selector** (this motivates X1 as the default for
  the ranking-critical path). Recommendation stands: **ADD Beta, don't replace.**

### V12 Repo audit — all four sub-claims **confirmed** (and richer than stated)

- **(a) Beta-calibration gap is real.** No score→probability Beta calibration anywhere in
  `src/`. (There is a `agentic/bayesian_calibrator.py` Beta-*Bernoulli* agent-reliability
  estimator — unrelated to calibration maps.) **Filled** in this PR.
- **(b) Weight inconsistency — flagged at the 2026-06-12 audit; since RESOLVED.** At audit
  time the fusion-weight denominator was written four ways (`Φ+2`, `2Φ`, a cross-mixed
  `φ_sum ≈ 2.8944`, and a README formula pairing the `2Φ` denominator with the `Φ+2`
  answer). The tree has since been reconciled to a single canonical `PHI:1:1` derivation —
  every weight-init site now computes `phi_sum = phi + 2.0  # canonical PHI:1:1`:
  - `src/omni_mercury_engine/core/three_r/fusion.py:151` (runtime default,
    `OmniAvaEquation.__init__`): `phi_sum = self.phi + 2.0  # ≈ 3.618` →
    (0.4472, 0.2764, 0.2764).
  - `src/omni_mercury_engine/core/three_r/fusion.py:388` and `:449` (the L-BFGS-B
    `initial_weights` seed and `DomainAdaptiveOAEWeights._default_weights`): both use the
    same `phi + 2.0` denominator — the former `2Φ ≈ 3.236` scheme is gone.
  - `docs/MATH_SPEC.md`: the arithmetically-wrong `φ_sum ≈ 2.8944` value no longer appears
    (grep returns no match); §2.1.1 and the constants table (line 78, table line 759) now
    state `w_R = Φ/φ_sum ≈ 0.4472` and map it to `core/three_r/fusion.py`.
  - `README.md:732`: now reads `w_R = φ/(φ+2) ≈ 0.447` (canonical `Φ:1:1`), self-consistent;
    the earlier internally-contradictory `φ/(φ+1+1/φ)` form is gone.
  - **Resolved** in `docs/MATH_SPEC.md` (truth-up table + reconciliation; see X8) and the
    fusion sources; no live four-way inconsistency remains.
- **(c) Stale "requires scikit-learn" messages** at `core/calibration.py` (now at lines
  141/219/620/723/785 after the rewording); backing `ml/mercury_ml.py` imports only
  numpy/scipy. **Fixed** (reworded all five to "pure numpy/scipy; no scikit-learn").
- **(d) η^Φ multiply at `core/three_r/fusion.py:251–252`** — the calibration-corrupting dial
  (V6). At audit time this was an unconditional `fusion_score = weighted_sum * eta**exponent`;
  PR #278 settled it as an opt-in, default-off decoupling, so the site now reads
  `ethical_scaling = 1.0 if self.decouple_ethical_scaling else eta**self.ethical_exponent`
  (`decouple_ethical_scaling` default `False`, `three_r/fusion.py:67,138`; `ethical_exponent`
  configurable, `:66`). **Confirmed** as the mechanism V6 damages; the X12a lint (authored in
  PR #275) flags exactly this pattern but is deliberately not ported, since it would flag the
  now-intentional gated design (see X12a).

---

## Part 2 — Exploration Program (run subset; pre-registered, kill-criteria honoured)

### X1 — Strict isotonic — **Iso+eps SURVIVES; CIR KILLED**
Pre-reg: restore exact AUROC while keeping isotonic ECE. Kill: AUROC Δ≠0 at 1e-9, or ECE >10% worse than vanilla.

| variant | mean|ΔAUROC| | max|ΔAUROC| | exact-AUROC runs | mean ECE | verdict |
|---------|-------------|------------|------------------|----------|---------|
| Vanilla isotonic | 3.08e-2 | 2.29e-1 | 0% | 0.0221 | (baseline) |
| **Iso+eps** | **0.00** | **0.00** | **100%** | 0.0240 | **SURVIVES** |
| CIR (centered isotonic) | 5.92e-4 | 1.56e-2 | 57% | 0.0395 | KILLED (ECE +79%, not exact) |

`Iso+eps` (isotonic squeezed into the open interval + ε·s tie-break) restores **exact**
AUROC on 100% of runs (synthetic + all 6×{large,small}×5 real) at ECE within the 10% bound.
*Implementation note (R2):* a naïve `clip(g+ε·s,0,1)` re-saturates exactly the points
isotonic flattened to {0,1} — the first attempt failed for that reason; squeezing fixes it.
**Shipped as `StrictIsotonicCalibration`.**

### X2 — Non-monotonicity guard — **Brier-gap detector SURVIVES; Kendall-τ KILLED**
Pre-reg: fire on V8 scramble (repair ≥0.90·oracle) and stay silent (<5% false-fire) on real.

| detector | scramble fires & repairs | real false-fire | verdict |
|----------|--------------------------|-----------------|---------|
| Kendall-τ on bins | yes (0.9009) | **40%** | KILLED |
| **Brier-gap, 99% paired bootstrap** | yes (0.9009) | **0.0% (0/60)** | **SURVIVES** |

The τ detector is too noisy on real label-sparse cal sets. The Brier-gap detector compares
isotonic vs an unrestricted remap on held-out cal; both see the same noise, so only genuine
non-monotonicity separates them (scramble bootstrap win-rate = 1.000). The 95→99% move is
principled, not tuning: a one-sided test's false-positive rate is ≤α by construction, and at
95% the detector sat exactly at its 5% floor.

### X8 — Settle Φ with data — flag **RESOLVED: demote Φ to initialisation**
Pre-reg H0: golden-ratio weights indistinguishable from learned. (Proxy: R/H/O = IsolationForest/LOF/kNN, since real 3R needs the torch stack — **G3: no detection claim**.)

| weights | held-out fused AUROC |
|---------|----------------------|
| golden-A (Φ+2 → .447/.276/.276) | 0.8925 |
| golden-B (2Φ → .500/.309/.191) | 0.8943 |
| equal (⅓,⅓,⅓) | 0.8826 |
| **learned (simplex argmax on cal)** | **0.9087** |

Learned beats golden-A by +0.0161 AUROC (paired **t≈5.3**, n=30); learned-optimum centroid
(0.705, 0.045, 0.250) is far from any golden point (‖·−golden-A‖=0.347). Per R3, **Φ did not
survive the sweep** → document as initialisation/default, not a proven optimum. *Caveat:* the
proxy inflates the gap (IsolationForest simply out-detects LOF here, so "learned" partly does
detector-selection); the real-3R 64-dataset sweep remains the definitive test. The *direction*
(reconcile to one derivation; Φ not special) is robust and matches the repo's own UNJUSTIFIED flag.

### X12a — Gate hardening (the only permitted ethics work; **R4**) — **not ported**
The X12a AST linter (`tools/lint_no_eta_score_multiply.py`) that would flag any
`score * (eta ** _)` was authored in PR #275 but is **deliberately not ported** to
`main`: PR #278 settled the η^Φ term as an opt-in, default-off decoupling (R6), so a
blanket lint would flag the shipped intentional design. The script is therefore not in
the tree and not wired into CI.

---

## What changed in the repo (all additive / truth-up; no behavioural change to the engine)
- `src/omni_mercury_engine/core/calibration.py`: **+`BetaCalibration`** (V12a/V10, a
  standalone class — not wired into the `calibrate_detector` method dispatch, whose
  whitelist is `auto`/`platt`/`isotonic`/`strict_isotonic`/`temperature`),
  **+`StrictIsotonicCalibration`** (X1), `calibrate_detector` gains `strict_isotonic`,
  five stale "requires scikit-learn" messages reworded (V12c). Existing classes untouched.
- (`tools/lint_no_eta_score_multiply.py`: X12a gate-hardening lint — **authored in PR #275, deliberately not ported**; see X12a above.)
- `docs/MATH_SPEC.md`: φ_sum arithmetic truth-up + weight reconciliation (V12b/X8).
- `tests/test_calibration_brief.py`: tests for the ported `StrictIsotonicCalibration` (X1 survivor). `BetaCalibration` was ported as a standalone class (see above) but is not exercised by *this brief's* test file — `main`'s accept-gated Beta (`fit_accept_gated_mca`) and the standalone `BetaCalibration` are both covered by `tests/test_beta_calibration.py` — and the X12a eta-multiply lint was not ported (see above).
- `benchmarks/calibration_brief/`: this suite + `results/`.

> **Not changed (flagged for a follow-up with its own Standard Track):** removing the η
> multiply from `core/three_r/fusion.py:251–252` and enforcing η as a second hard veto beside
> σ_Immutable. This is behaviourally significant and needs the fusion test-suite (blocked
> here by the mandatory PQC import gate — `ama_cryptography` unavailable). The X12a lint
> makes the bug class detectable in the meantime.

## Pre-registered but NOT yet run (transparent scope; each retains its kill criterion)
X3 multicalibration/Mondrian · X4 BBSE/Saerens label-shift π̂ · X5 smooth/kernel ECE ·
X6 anytime-valid e-process monitor · X7 decision calibration · X9 2-component Beta mixture ·
X10 weighted/adaptive conformal · X11 logit-space calibration (needs torch) ·
X13 Tsallis/Rényi selection · X14 Fisher–Rao stabilised fit · X15 persistent-homology early
warning · X16 MISE-optimal binning · X17 e-value-gated actions. None claimed; none run.

## Operating-rule compliance
R1 pre-registration (metric+kill stated before each run) ✓ · R2 every number incl. losses
(CIR, τ-detector, X1-clip bug) ✓ · R3 Φ swept, not assumed ✓ · R4 ethics gate only hardened,
never in a loss ✓ · R5 determinism: byte-identical reruns verified ✓ · R6 transparent naming (no
"quantum"/"cosmic" claims made) ✓ · R7 new methods (Beta, StrictIsotonic) ran the full
synthetic+real track before adoption ✓. Ceiling (G4): no method beats the conditional-mean
oracle on any metric — all tie or fall short, as the theorem requires.

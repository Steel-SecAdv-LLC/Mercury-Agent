# Detection Mechanisms: Streaming, Statistical, and State-Space Detectors

This document is the design rationale for the detection-mechanisms expansion
(branch `steel/detection-mechanisms`). It records **what** each new detector is,
**why** the algorithm was chosen, **how** it plugs into Mercury Agent, and the
**calibration contract** that makes its scores usable by the fusion ensemble.

The standing rule is anti-theater: every detector here is a complete, enabled,
unit-tested `BaseDetector` — training, inference, feature extraction, and
manifest registration — not a scaffold or a flagged-off stub. All five are pure
NumPy/SciPy so they import without the optional PyTorch stack and run on the
lightweight core install.

## 1. Where they fit

Each detector subclasses
[`omni_mercury_engine.core.base.BaseDetector`](../src/omni_mercury_engine/core/base.py)
and implements the four-method contract consumed by the engine and fusion seam:

| Method | Contract |
| --- | --- |
| `fit(data)` | Learn parameters + the score-calibration scale; set `is_fitted()`. |
| `detect(data)` | Return `{"anomaly_score", "scores", "is_anomaly", "confidence"}` with per-sample `scores` in `[0, 1]`. |
| `extract_features(data)` | Return an `(n_samples, feature_dim)` float32 block for ML fusion. |
| `is_fitted()` | Report readiness. |

They are registered as **opt-in `BASE`** entries in
[`DETECTOR_MANIFEST`](../src/omni_mercury_engine/core/detector_registry.py) and
exported lazily from
[`detectors/__init__.py`](../src/omni_mercury_engine/detectors/__init__.py).
Being opt-in means `auto_discover_detectors()` can reach them and the engine can
`enable_detector(name)` them, but they are **not** in the default fusion set, so
the calibrated production ensemble is unchanged until a deployment enables them.
This mirrors how `geo_movement`, `graph_based`, and `kmeans_distance` are
handled, and is locked by
`tests/detectors/test_detector_manifest_integrity.py`.

## 2. The calibration contract

Ensemble stacking and decision thresholds require comparable, calibrated scores.
Two families are used:

- **Native-probability detectors** (BOCPD, SPOT) emit a quantity that is already
  a probability or an EVT tail statistic, so no squashing is applied.
- **Residual detectors** (SR, Hawkes, particle filter) emit a non-negative
  "surprise" `r` and squash it monotonically with `1 - exp(-r / scale)`. The
  `scale` is chosen so that a high **training quantile** `calibration_quantile`
  (default `0.98`) lands exactly on the `0.5` anomaly boundary
  (`scale = quantile(r, 0.98) / ln 2`). Consequently the normal-regime
  false-positive rate is approximately `1 - calibration_quantile`, i.e. a
  *chosen, controlled* budget rather than an ad-hoc threshold. This is verified
  empirically by the `test_low_false_positive_rate` cases.

All detectors coerce NumPy **or** `torch.Tensor` input by duck-typed `.detach()`
(no hard torch import), flatten to a 1-D series, and `nan_to_num` the input, so
they never raise on NaNs or tensor inputs.

## 3. The detectors

### 3.1 Spectral Residual (SR) — `spectral_residual.SpectralResidualDetector`

The SR transform (Ren et al., *Time-Series Anomaly Detection Service at
Microsoft*, KDD 2019) smooths the log-amplitude FFT spectrum, subtracts it from
itself, and inverts the residual spectrum to a time-domain **saliency map** whose
peaks localise points poorly explained by the signal's own periodic structure.
It is training-free; `fit` only records the saliency scale. The streaming point
is sharpened by extrapolating one point before the FFT (paper's boundary trick).
Signal: local saliency deviation above a moving-average baseline.

### 3.2 BOCPD — `bocpd.BOCPDDetector`

Bayesian Online Change-Point Detection (Adams & MacKay, 2007) maintains a
posterior over the **run length** (time since the last change point). A
Gaussian observation model with a Normal-Inverse-Gamma conjugate prior gives a
Student-t posterior predictive; a constant hazard sets the geometric
change-point prior. When statistics shift, mass collapses toward run length 0,
so `P(run length < change_grace)` is a directly-calibrated change-point score.
The run-length distribution is truncated at `max_run_length` with truncated mass
folded into the last bin (no silent loss), and the recursion is renormalised
defensively if it underflows.

### 3.3 SPOT / DSPOT — `spot_evt.SPOTDetector`

SPOT (Siffer et al., *Anomaly Detection in Streams with Extreme Value Theory*,
KDD 2017) sets thresholds from Extreme Value Theory. Excesses over a high
empirical quantile are modelled by a Generalized Pareto Distribution (fit here by
the closed-form, streaming-stable method of moments); a target **risk** `q`
(false-positive budget) maps to the data-driven threshold `z_q`. Normal peaks
update the tail online; anomalies do not feed back. DSPOT (`depth > 0`) first
removes a moving-average trend so the tail model applies to a locally-stationary
residual, handling drift a static threshold could not. This is Mercury's
principled dynamic-thresholding primitive: it gives the alerting path a bounded,
demonstrable false-positive rate.

### 3.4 Hawkes burst — `hawkes.HawkesBurstDetector`

A Hawkes process is self-exciting: each event transiently raises the intensity of
further events, capturing **clustering** that a fixed-rate Poisson baseline
misses. For a per-bin count stream the exponential-kernel intensity follows
`lambda_t = mu + g_t`, `g_t = exp(-beta)(g_{t-1} + alpha n_{t-1})`. `fit`
estimates `mu` from the mean count and the excitation gain `alpha` from the
lag-1 autocovariance, clamped to keep the branching ratio below 1 (sub-critical
/ stationary). Signal: the absolute Pearson residual
`|n_t - lambda_t| / sqrt(lambda_t)`, which flags both bursts and anomalous
silences.

### 3.5 Particle filter — `particle_filter.ParticleFilterDetector`

A bootstrap particle filter tracks a local-level state-space model
(`x_t = x_{t-1} + process_noise`, `y_t = x_t + obs_noise`) with a weighted
particle cloud, yielding a non-parametric one-step-ahead predictive. Signal: the
normalised innovation `|y_t - E[y_t | past]| / std[y_t | past]`, large when the
series departs from the tracked dynamics. `fit` estimates the process noise from
the robust scale of training first differences and the observation noise from the
series spread. Resampling is **systematic** (low-variance) under a **pinned
seed**, so output is deterministic and reproducible.

## 4. Testing

`tests/detectors/test_{spectral_residual,bocpd,spot_evt,hawkes,particle_filter}.py`
(38 tests) cover, per detector: the `BaseDetector` contract (shapes, ranges,
`is_fitted` transitions), input-validation errors, degenerate inputs (constant
series, negative counts), signal separation on an injected anomaly, and empirical
false-positive control on a held-out normal stream. The particle filter
additionally asserts bit-identical output for a fixed seed. The existing
parametrised manifest-integrity suite automatically covers the new manifest
entries (class resolution, `BaseDetector` subclassing, engine reachability).

## 5. Deliberately out of scope

The wider "detection mechanisms" wishlist also names deep generative and
neuromorphic detectors (diffusion/DDPM, energy-based models, Deep SVDD, spiking
networks, reservoir/echo-state), digital-twin residuals, and a dedicated RCA
module. Those require the PyTorch/simulation stack and substantial training and
evaluation infrastructure; bolting on flagged-off stubs would violate the
no-scaffolding rule. This PR delivers the classical streaming / statistical /
state-space tier **completely** — the detectors that are correct, cheap, and
fully testable in the lightweight core — as a solid, non-regressing foundation
the heavier tier can build on.

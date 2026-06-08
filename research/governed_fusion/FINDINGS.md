# Governed Fusion Substrate — Phase 2 correction pass (PR #278)

All numbers below were reproduced **in this branch from committed code** on the
real reachable suite. Nothing here is borrowed from FINDOYOU or any paper.

## Environment & suite

- AMA/PQC native backend required (hard import gate); env:
  `PYTHONPATH=.:src:/tmp/ama-cryptography`,
  `AMA_CRYPTO_LIB_PATH=…/build/lib/libama_cryptography.so`,
  `LD_LIBRARY_PATH=…/build/lib`, `MERCURY_ALLOW_SYNTHETIC=0`.
- Reachable suite: **29 real events / 9 domains** built by
  `research/governed_fusion/suite.py` from the live loaders — earthquake 5,
  pandemic 5, hurricane 4, tsunami 3, energy 3, fema 3, tornado 2, marine 2,
  network_security 2. `pandemic/ebola_2014` falls back to synthetic despite the
  flag and is **excluded** (stated, not averaged in). Unreachable domains
  (wildfire, flood, volcanic, landslide, financial, sepsis) are never
  synthesised.
- Metrics via `ml/mercury_ml` (no scikit-learn). Aggregation is the per-event
  **macro mean** (equal weight per event), matching `research/omni_equation`.
- Large events use a **seeded stratified row cap** (default 6000;
  `suite.stratified_subsample`, seed 42) for iteration. `MercuryAnomalyDetector`
  is deterministic post-`fit()`; each event's scores are cached.

## Baseline (default fixed ensemble) — `measure_baseline.py`

| events | AUROC | AUPRC | F1 | P | R |
|---:|---:|---:|---:|---:|---:|
| 29 | 0.840 | 0.426 | 0.288 | 0.317 | 0.519 |

Reproduces the PR's reported baseline (0.844 / 0.434 / 0.302 / 0.317 / 0.542)
within stratified-cap noise (precision matches exactly). Pooled row-weighted
AUROC is 0.585 — confirming the macro mean is the right, non-swamped statistic.

## Item 4 — conformal operating point: **LANDED** (opt-in, default-off)

`measure_conformal.py`. Per event: seeded stratified 50/50 calibration/eval
split; the detector is fit unsupervised, so calibration labels form a valid
split-conformal set. Threshold = class-1 LAC quantile (`1 - q_1`) from
`BinaryConformalClassifier`. Both thresholds act on identical eval scores, so
AUROC/AUPRC are rank-invariant — the lever is purely the operating point.

| metric | baseline (eval) | conformal (eval) | Δ |
|---|---:|---:|---:|
| AUROC | 0.820 | 0.820 | +0.000 (rank-invariant) |
| AUPRC | 0.448 | 0.448 | +0.000 (rank-invariant) |
| F1 | 0.311 | 0.416 | **+0.105** |
| precision | 0.344 | 0.318 | −0.026 |
| recall | 0.450 | 0.837 | +0.387 |

26 of 29 events used; **3 dropped** (earthquake `noto_2024`, `tohoku_2011`,
`nepal_2015` — no positive in a calibration split), reported never hidden. The
F1 win at controlled operating point is the measured win; the tradeoff is a
small precision dip for a large recall gain. Wired opt-in via
`conformal_operating_point` (default off → byte-exact Youden/F1 path); see
`tests/test_conformal_operating_point.py`.

## Item 2 — adversarial survivability on the REAL fused score — research only

`measure_survivability.py` + `research/adversarial/governed_attacks.py`. The
attack target is Mercury's **real fused ensemble anomaly score** (one
representative event per domain; `eps=0.6`, 160-row stratified cap), not a toy
`||x−loc||`. Fixed-budget battery (condmean / BPDA / NES / transfer) restricted
to a controlled subset of `m` channels (`most_informative_channels`).

### Floor curve — worst-case fused AUROC vs controlled-channel budget `m`

| domain | event | k | clean | floor curve (m: worst-case AUROC) |
|---|---|--:|--:|---|
| earthquake | turkey_syria_2023 | 8 | 0.987 | 2:0.484 4:0.327 6:0.201 |
| energy | quebec_1989 | 8 | 1.000 | 2:1.000 4:0.999 6:0.541 |
| fema | hurricane_2024 | 7 | 0.924 | 1:0.802 3:0.829 5:0.594 |
| hurricane | milton_2024 | 8 | 0.887 | 2:0.887 4:0.850 6:0.854 |
| marine | marine_heatwave | 3 | 0.997 | 1:0.969 2:0.928 |
| network_security | nsl_kdd | 9 | 0.913 | 2:0.877 4:0.817 6:0.759 |
| pandemic | mpox_2022 | 12 | 0.795 | 3:0.793 6:0.794 9:0.791 |
| tornado | super_outbreak | 12 | 0.964 | 3:0.858 6:0.357 9:0.420 |
| tsunami | tonga_2022 | 6 | 0.978 | 1:0.978 3:0.974 4:0.916 |

**Overall:** mean clean AUROC 0.938 → mean worst-case at `m=k/2` 0.768 (drop
0.170). Some domains collapse through chance under on-manifold half-channel
evasion (earthquake 0.987→0.327, tornado 0.964→0.357); others are robust
(hurricane, mpox). The masking flag fires iff NES (gradient-free) beats BPDA
(gradient-based), independent of the global worst; `win_counts` is a true
per-row tally.

### Cubic-moment escape — `D_φ` over `φ(z)=[z, z²−1, z³]` vs the Gaussian floor

`cubic_moment_score` fits a Gaussian manifold in the lifted moment space.

- **Gaussian control sanity:** floor 0.872, cubic 0.869, **escape −0.003** — the
  detector vanishes to the floor on Gaussian data (mean/cov-only structure).
- Real domains (escape = cubic AUC − floor AUC): hurricane **+0.119**, pandemic
  **+0.117**, network_security +0.047, tornado +0.029, tsunami +0.013, marine /
  earthquake / energy +0.000 (floor already 1.000), fema −0.009. Where the floor
  is not saturated, the cubic detector escapes it — the anomalies carry genuine
  3rd-moment structure mean+covariance cannot see.

Zero runtime change; see `tests/research/test_governed_adversarial_smoke.py`
(includes both sanity directions and masking-flag independence).

## Item 3 — reliability-weighted bounded-influence fusion: **KILL CONFIRMED**

`measure_reliability_fusion.py` + `core/robust_pooling.py`. This measures the
**specified** lever (not the naive variance weighting that was reverted):
reliability weights (per-component AUROC-above-chance, #38 self-down-weighting)
combined with the bounded-influence **clipped** / **trimmed** log-odds pool.
Per event: 50/50 cal/eval split, weights from calibration only, AUROC on eval.

| | baseline 0.40/0.30/0.30 | rel·linear | rel·clipped | rel·trimmed | best-single |
|---|---:|---:|---:|---:|---:|
| OVERALL (26 ev) | 0.827 | 0.865 | **0.869** | 0.838 | **0.894** |

The bounded-influence pool **improves over baseline** (0.827 → 0.869) but still
**does not reach best-single** (0.894); gap **−0.026**, with per-domain collapse
(hurricane −0.125, fema −0.047, network_security −0.019). On the reachable suite
the components are too redundant (the omni-equation harness measured mean
|corr| 0.66) for any pooling to beat picking the single best stream.

**Decision:** measured from committed code with the specified machinery, the
fusion-weighting lever is **conclusively killed** (I3: no clean full-suite gain
⇒ not wired into runtime). `core/robust_pooling.py` is retained as a tested
prototype + the reproducible script; `detect()` is unchanged. Best-single
selection / calibration (Item 4) remain the operational levers.

## Item 1 — info-geometry certificate: scoped honestly, boundary-correct

`core/governed_fusion.py` (`InfoGeometryCertificate`) +
`detectors/statistical.py`. `p_τ` is now `g⁻¹(component_threshold)` using the
info-geometry component's **own** adaptive operating point and its **real** score
map `g(p)=1−exp(−p²/2d)` — not the ensemble threshold under an assumed map.
Renamed `fusion_certificate → info_geometry_certificate`; the payload states it
certifies the component's price level-set, **not** the fused/gated verdict (a
verdict-level certificate through neural fusion + calibration + both gates is a
separate, larger task, out of scope). Optional, **default-off**, post-hoc and
read-only (exact reduction). Soundness test perturbs the input and verifies the
**real** `_compute_info_geometry_score` does not cross its threshold within ρ.

## Item E — certificate wiring threaded, no stale engine state

`OmniMercuryEngine._extract_detector_features` returns certificates in its tuple
and `detect_with_fusion` threads them straight into the result. The mutable
`self._last_detector_certificates` is gone, so two interleaved `detect` calls
cannot cross-contaminate certificates
(`tests/test_governed_certificate_threading.py`).

## Invariants

I1 both hard gates (`BenevolenceScorer`, `σ_Immutable`) byte-untouched, ethics
tests green. I2 every addition optional + default-off + exact-reducing; no new
heavy deps (`mercury_ml`, not sklearn). I3 the only AUROC-losing lever (Item 3)
is not wired into runtime. I4 measured on real reachable domains; no synthetic
claims.

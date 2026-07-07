# Detection Mechanisms Runbook

Operational runbook for the streaming / statistical / state-space detector tier
(package `omni_mercury_engine.detectors`, integration seam
`omni_mercury_engine.detectors.detection_tier`). Read
[`DETECTION_MECHANISMS.md`](./DETECTION_MECHANISMS.md) first for the design; this
runbook covers deploy, calibrate, monitor, and incident response. Companion
runbooks: [`RETRAIN_RUNBOOK.md`](./RETRAIN_RUNBOOK.md),
[`ESCALATION_AUDIT_RUNBOOK.md`](./ESCALATION_AUDIT_RUNBOOK.md).

## 1. What the tier is

Eighteen `BaseDetector` detectors across six paradigms, auto-discovered through
`core/detector_registry.py::DETECTOR_MANIFEST` and combined by
`detection_tier.StreamingScoreEnsemble` into a calibrated, false-positive-bounded
anomaly stream with graph-based root-cause attribution. Sixteen detectors are
pure NumPy/SciPy and always available; `srcnn` and `diffusion_ad` require the
`[ml]` (PyTorch) extra.

## 2. Environment & configuration

| Variable | Purpose | Default / fail mode |
|---|---|---|
| `AMA_CRYPTO_VERSION` | Declares the AMA-Cryptography release; must match the pinned `3.3.0`. | Unset is allowed; a mismatch is a loud, fail-closed startup error. |
| `MERCURY_REQUIRES_ML` | When `1`, aborts the session if PyTorch is missing (needed for `srcnn` / `diffusion_ad`). | Unset ⇒ torch detectors skip gracefully. |
| `contamination` (ensemble arg) | Expected anomaly fraction for threshold calibration. | `0.05` |
| `calibration_quantile` (detector arg) | Training-residual quantile placed at the 0.5 boundary; `1 − q` ≈ normal-regime FPR. | `0.98` |
| `alpha` (`conformal_threshold` arg) | Target distribution-free false-positive rate. | `0.05` |
| `OMNI_ENSEMBLE_CALIBRATION` | Per-detector ensemble score calibration: `rank`/`ecdf`/`isotonic`/`platt`/`none`. | `rank` (empirical-CDF, label-free) |
| `OMNI_ENSEMBLE_WARMUP` | Warm-up window the ensemble calibrators train on (int points / float fraction). | Unset ⇒ whole training series |
| `OMNI_DETECTOR_NAN_POLICY` | How guards treat NaN/Inf: `neutral`/`impute`/`flag`/`raise`. | `neutral` (replace with 0.0, clamp inf; never aborts) |
| `OMNI_DETECTOR_MAX_MAGNITUDE` | Single safe magnitude cap; `±inf` maps here and finite values are clamped into `[−cap, cap]`. | `1e100` (`_calibration.FINITE_CAP`) |
| `OMNI_DETECTOR_RIDGE_FACTOR` | Digital-twin scale-relative Tikhonov ridge factor (`λ = max(f·tr(G)/d, ridge)`). | `1e-6` |
| `OMNI_DETECTION_CONFIG` | Path to a YAML/JSON config file supplying the above knobs (env still wins). | Unset ⇒ defaults + env |

Precedence for the tier knobs: **defaults < config file (`OMNI_DETECTION_CONFIG`)
< environment variables < a detector's explicit `config` dict**. The same keys
(`nan_policy`, `max_magnitude`, `ridge_factor`, `ensemble_calibration`,
`ensemble_warmup`) are accepted in the config dict passed to a detector.

Install the full tier (including torch detectors):

```bash
pip install -e ".[ml]"
```

## 3. Deploy

### 3.1 Register + auto-discover

Detectors register themselves via the manifest; no code change is needed to bring
the tier online:

```python
from omni_mercury_engine.core.detector_registry import get_global_registry

registry = get_global_registry()  # auto-discovers all manifest detectors
```

### 3.2 Build a calibrated ensemble

```python
from omni_mercury_engine.detectors.detection_tier import (
    build_tier_detectors, StreamingScoreEnsemble,
)

detectors = build_tier_detectors(
    ["spectral_residual", "bocpd", "gaussian_process", "particle_filter"]
)
ensemble = StreamingScoreEnsemble(detectors, method="bma").fit(train_series, train_labels)
scores = ensemble.score(live_series)          # calibrated per-point probabilities
uncertainty = ensemble.ensemble_uncertainty(live_series)
```

On real streaming data without up-front labels, `average` is the recommended
default combiner — it leads on the real-data NAB benchmark (ROC-AUC 0.613, ahead
of `stacking` 0.571 / `bma` 0.563 on the labelled subset). Use `stacking` / `bma`
only when abundant point labels are available to fit the meta-learner; on
sparsely-labelled streams they do not beat `average`.

### 3.3 Bound the false-positive rate

```python
from omni_mercury_engine.detectors.detection_tier import conformal_flags

flags = conformal_flags(scores, calibration_scores=normal_scores, alpha=0.05)
```

## 4. Calibrate & retrain

- Re-fit detectors on a fresh normal window whenever the baseline drifts; the
  `calibration_quantile` anchors the 0.5 boundary in the normal tail, so the FPR
  tracks `1 − calibration_quantile` without manual threshold tuning.
- For labelled data, prefer stacking / BMA so the meta-learner reweights
  detectors to the current regime. Follow [`RETRAIN_RUNBOOK.md`](./RETRAIN_RUNBOOK.md)
  for the staged deploy / rollback path.
- Regenerate the benchmark after any detector change. A quick tier-only,
  real-data (NAB) summary — prints, does not commit:

```bash
AMA_CRYPTO_VERSION=3.3.0 python -m benchmarks.detection_tier_benchmark
```

  The committed numbers live in the `detection_tier` section of
  `benchmarks/mercury_benchmark_results.json`; refresh them with the canonical
  harness (also refreshes the main headline):

```bash
AMA_CRYPTO_VERSION=3.3.0 python benchmarks/mercury_benchmark.py
```

## 5. Monitor

Detectors emit the standard `omni_detection_*` Prometheus series through
`core/metrics.py` (`time_detection`, `time_feature_extraction`), so the existing
Grafana boards under `monitoring/` cover them. Watch:

- `omni_detection_duration_seconds` p95 per `detector_type` — latency regressions
  (BOCPD and IMM are the heaviest members; see the benchmark table).
- `omni_detection_success_total` ratio — a detector repeatedly failing trips its
  registry circuit breaker (`resilience/api_circuit_breakers.py`) and is skipped.
- `omni_detector_nonfinite_corrected{detector,policy,field}` rate — a rising rate
  means a detector is rescuing NaN/Inf in its input (`field="input"`) or metadata
  (`field="z_q"` / `"gamma"`). Under the default `neutral` policy this is silent
  and non-fatal, so the counter is the signal: investigate upstream data quality
  (a stuck sensor, a divide-by-zero feature) rather than assume the stream is
  healthy. Each increment is paired with a structured `logger.warning`.
- Ensemble uncertainty distribution — a sustained rise signals detector
  disagreement / regime change and should gate automated response.

## 6. Incident response & RCA

When many channels go anomalous together, localise the origin:

```python
from omni_mercury_engine.detectors.detection_tier import rca_localize

ranked = rca_localize(node_matrix, adjacency=service_graph, train=normal_matrix, top_k=5)
# -> [(node_index, attribution), ...] descending; attach to the CAP alert.
```

The ranked causes flow into the decision layer (`decision/bridge.py`) and appear
in the CAP alert metadata for on-call triage.

## 7. Failure modes & fixes

| Symptom | Cause | Fix |
|---|---|---|
| Startup `RuntimeError: AMA/PQC version mismatch` | `AMA_CRYPTO_VERSION` ≠ pinned `3.3.0`, or wrong AMA build. | Unset the var or set it to `3.3.0`; rebuild AMA via `scripts/build_ama_native.sh`. |
| `srcnn` / `diffusion_ad` not discovered | PyTorch not installed. | `pip install -e ".[ml]"`; they load lazily. |
| A detector's circuit breaker is OPEN | Repeated detector errors on malformed input. | Inspect `registry.health_check()`; the registry skips it and continues — no cascade. |
| Elevated false positives | Baseline drift moved the normal tail. | Re-fit on a current normal window; lower `alpha` / raise `calibration_quantile`. |
| Ensemble slower than budget | Heavy members (BOCPD, IMM, particle-filter). | Drop them from the `build_tier_detectors` subset; the tier is modular. |
| Ensemble no better than best single detector | Raw score averaging diluted by uninformative members. | Use the `consensus` combiner (default calibration `rank`); it takes a robust cross-detector high quantile instead of a mean. |
| Rising `omni_detector_nonfinite_corrected` | Upstream data emitting NaN/Inf (stuck sensor, div-by-zero feature). | Fix the source; to fail loudly instead of rescuing, set `OMNI_DETECTOR_NAN_POLICY=raise`. |
| Digital-twin scores unstable on large-magnitude signals | Ridge too small relative to the Gram scale. | The scale-relative ridge handles this by default; raise `OMNI_DETECTOR_RIDGE_FACTOR` if still ill-conditioned. |

## 8. Rollback

The tier is additive and manifest-driven. To disable it, remove the tier entries
from `DETECTOR_MANIFEST` (or pass an explicit non-tier detector set to the
ensemble); no schema migration is required and existing detectors are unaffected.

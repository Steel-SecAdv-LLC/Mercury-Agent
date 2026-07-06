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

## 8. Rollback

The tier is additive and manifest-driven. To disable it, remove the tier entries
from `DETECTOR_MANIFEST` (or pass an explicit non-tier detector set to the
ensemble); no schema migration is required and existing detectors are unaffected.

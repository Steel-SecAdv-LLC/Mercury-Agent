# Runbook: OmniMercuryDataDrift / OmniMercuryDriftBaselineMissing

**OmniMercuryDataDrift fires when** `omni_drift_detected_total` increased at
severity MEDIUM, HIGH or CRITICAL within 30 minutes, for 10 minutes.
**OmniMercuryDriftBaselineMissing fires when** drift checks ran with
`status="no_baseline"` for 15 minutes.

## What the check does

Every `detect_with_fusion` call tests its batch for distribution drift
(`EngineConfig.drift_detection`, on by default). Only the rows the model calls
normal are tested, so a batch is not reported as drifted for containing
anomalies. They are compared with reference rows the model also calls normal,
re-derived after every fit or load from, in order: an explicit
`enable_drift_detection(baseline)` (kept across fits and loads), the fitted
spatial detector's training rows, or the fusion training sample. An inference
batch is never the reference: a spatial detector auto-fitted on inference data
(`require_explicit_fit=False`) is skipped. The test is an ensemble:
Kolmogorov-Smirnov and PSI per feature, plus a covariance-structure test and a
principal-axis test (`ml/drift.py`). The principal-axis test compares each axis
of the reference's PCA basis, so it catches a shift along a low-variance
direction of correlated features that every single-feature test dilutes; its
message reads "mean shifted along a principal axis".

A batch with drift at MEDIUM or above has its decision demoted to DEFER
(`decision` record, reason "distribution drift … the calibration may no
longer hold"). The anomaly scores are still returned.

Each result carries `drift_detection.status`:

| Status | Meaning | Can defer? |
|---|---|---|
| `tested` | The normal-looking rows were tested | Yes, when `is_drift` |
| `insufficient_rows` | Fewer than 20 normal-looking rows (e.g. a single-row request, or a batch the model flags almost entirely) | No |
| `incomparable` | The input has a different number of features than the training data | No |
| `no_baseline` | No model has been fitted or loaded, or the only fitted detector was auto-fitted on inference data | No |
| `error` | The drift check raised; the detection itself still returns and `omni_drift_checks_total{status="error"}` counts it (details in the logs) | No |

Measured on the shipped checkpoint with fresh draws from its training
distribution (100 batches each): 50-row batches deferred for drift 1% of the
time, 200-row batches 0%; a +1.0 shift was detected in 100% of 50-row
batches, a +0.5 shift in 99% (16% with the principal-axis test turned off).
A drift check costs 25-35 ms; the first check after a fit or load also derives
the reference (about 2 s for the shipped checkpoint's 1168 training rows).

## First checks

1. Severity mix: `sum by (severity) (increase(omni_drift_detected_total[1h]))`.
   Occasional LOW detections are expected noise and never defer.
2. Is it the data or the model? The shipped `default_fusion.pt` was trained on
   a synthetic Gaussian mixture (`scripts/train_default_fusion.py`). Real
   data almost always drifts from it, and that drift is genuine: its
   calibration does not transfer.
3. A recent change upstream (new sensor, unit change, schema change) usually
   shows as CRITICAL across most features at once. Inspect the per-feature KS
   statistics in `drift_detection` on a sample request.

## Remedies

- Fit on representative data: `mercury-agent train -o /path/fusion.pt` (or
  `engine.fit_fusion(X, y)` and save), then point the served engine at it
  with `MERCURY_FUSION_CHECKPOINT=/path/fusion.pt` (API, MCP server and the
  `detect` / `explain` commands). The reference is derived from the new
  training rows automatically. A missing path fails the load; it never falls
  back to the bundled synthetic checkpoint.
- Supply a reference explicitly: `engine.enable_drift_detection(baseline)`
  with rows from the current, verified-normal distribution.
- `no_baseline`: the serving process never loaded a model. Check the startup
  logs for the checkpoint load (and `MERCURY_FUSION_CHECKPOINT`, if set).
- `incomparable` on every batch after a refit: the input's feature count
  differs from the training data's raw feature count (constant columns and
  PCA reduction inside the detector no longer matter).

Disabling the check (`MERCURY_DRIFT_DETECTION=0`) removes the deferrals, but
the model still runs on data it was not calibrated for. Prefer refitting.

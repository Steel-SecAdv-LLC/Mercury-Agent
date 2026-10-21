# Runbook: OmniMercuryDetectionSuccessRateDrop

**Fires when** `sum(rate(omni_detection_success_total[5m])) / sum(rate(omni_detection_requests_total[5m])) < 0.95`
for 10 minutes (`monitoring/prometheus/prometheus-rules.yaml`).

## What the series count

`core.metrics.time_detection` wraps every component `detect`/`predict` call in
`OmniMercuryEngine` (public `detect`, the detector and model feature paths of
`detect_with_fusion`) and every message scored by the streaming worker
(`detector_type="stream"`). A call that returns counts a request and a
success; a call that raises counts a request only.

A component that *declines* input of a modality it does not process
(`InputModalityError`) is not a request at all. It increments
`omni_component_declined_total` instead. So a drop here means components are
failing, not that the input is wrong for them.

## First checks

1. Which component: `sum by (detector_type) (rate(omni_detection_requests_total[5m]) - rate(omni_detection_success_total[5m]))`.
2. Matching errors: `sum by (model_name) (rate(omni_model_inference_errors_total[5m]))`.
   The same component should appear there.
3. Logs: the engine logs `Detector <name> feature extraction failed: <error>` /
   `Model <name> ...` at DEBUG. The worker logs `Detection error:` at ERROR.
   Filter by the request's `correlation_id`.

## Likely causes

- **A detector was not fitted.** If the engine runs with
  `require_explicit_fit`, an unfitted detector raises before it detects; check
  that `fit_fusion` / the checkpoint load ran on this replica.
- **Malformed payloads on the stream.** For `detector_type="stream"`, the
  default detector raises `TypeError`/`ValueError`/`KeyError` on non-numeric
  fields. Inspect a recent message from the input topic.
- **A new input modality without a decline.** A component that should refuse
  an input but raises a plain `ValueError` will look like a failure. Raise
  `InputModalityError` at its modality check instead.

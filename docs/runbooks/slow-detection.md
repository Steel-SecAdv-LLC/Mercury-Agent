# Runbook: OmniMercurySlowDetection

**Fires when** the recording rule `omni_mercury:detection_duration_seconds:p95`
(p95 of `omni_detection_duration_seconds` by `detector_type`) is above 10 s
for 10 minutes.

## What the series measures

The histogram records the wall-clock time of each component `detect`/`predict`
call (see [detection-success.md](detection-success.md)). Feature extraction is
timed separately in `omni_detector_extraction_duration_seconds`, and the fusion
head in `omni_fusion_inference_duration_seconds`. Feature-cache hits run no
detector and record nothing, so latency reflects real computation only.

## First checks

1. Which component: `histogram_quantile(0.95, sum by (le, detector_type) (rate(omni_detection_duration_seconds_bucket[5m])))`.
2. Batch size: per-call latency scales with rows per call. Compare
   `rate(http_request_duration_seconds_sum[5m])` on the detect routes against
   request volume.
3. Saturation: `omni_mercury:cpu_utilization:avg5m` and the HPA's current vs
   maximum replicas (`OmniMercuryHPAAtMax`).
4. Streaming: if `detector_type="stream"` is slow, check
   `omni_detection_queue_depth`. A growing backlog with flat throughput means
   the workers cannot keep up; scale out or reduce per-message work.

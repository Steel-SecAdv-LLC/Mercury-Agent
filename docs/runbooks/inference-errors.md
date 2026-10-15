# Runbook: OmniMercuryModelInferenceErrors

**Fires when** `sum(rate(omni_model_inference_errors_total[5m])) > 0.1` for 5 minutes.

## What the counter counts

One increment per failed component call, labelled `model_name`:

- a detector or domain model in the engine's feature/detection paths (name of
  the component);
- the fusion head (`model_name="fusion"`, recorded with the exception
  re-raised to the caller);
- the streaming worker's detector (`model_name="stream"`).

It does **not** count declines. A component refusing an off-modality input,
such as `medical_abms` on a plain array, raises `InputModalityError`, which
increments `omni_component_declined_total` only. A high decline rate is normal
when an engine with domain models serves generic tabular data.

## First checks

1. Which component: `sum by (model_name) (rate(omni_model_inference_errors_total[5m]))`.
2. `model_name="fusion"`: the fusion head failed on the feature groups it was
   given. Check that the loaded checkpoint's feature groups match the
   detectors this replica runs, and look for a re-raised exception in the API
   logs (`http_requests_total{status="500"}` on the detect routes rises with
   it).
3. A named detector or model: the engine skipped it and served the rest. Find
   the error in DEBUG logs (`<name> feature extraction failed`). If it is a
   modality mismatch that should be a decline, raise `InputModalityError` at
   that component's input check.

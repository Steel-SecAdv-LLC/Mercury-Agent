# Mercury Agent — Operator Tools

This document is the operator reference for the
``omni_mercury_engine.tools`` subpackage.  Every tool listed here is
runnable in three equivalent ways:

```bash
python -m omni_mercury_engine.tools.<name>            # canonical
mercury-agent tool <name>                              # CLI shortcut
mercury-agent tool list                                # discover tools
```

Every tool emits a JSON certificate to stdout (or to ``--output PATH``)
with this canonical envelope:

```json
{
  "tool": "<name>",
  "schema": "mercury.tools.<name>/v1",
  "status": "ok" | "warn" | "fail",
  "generated_at": "<RFC3339 UTC>",
  "mercury_version": "1.7.0",
  "body": { ... tool-specific ... },
  "warnings": [ "..." ]
}
```

Pass ``--sign-key-hex <64-hex-Ed25519-seed>`` to also write a
``<output>.sig.json`` side-car signed with Ed25519.  Pass ``--require``
to make the tool exit non-zero unless ``status == "ok"`` (the default
behaviour already maps ``fail`` → exit 1, ``warn`` → exit 0).

## Quick reference

### Ethical / mathematical certifiers

| Tool | What it proves |
| --- | --- |
| ``lyapunov_validator`` | The fusion linearisation $\dot V = -x^T Q x$ is negative-definite for the documented λ. |
| ``benevolence_certifier`` | Loads a checkpoint, runs the curated benevolence probe set, verifies score ≥ 0.99 on every known-good input. |
| ``oae_weight_certifier`` | Verifies the (w_R, w_H, w_O) ≈ (0.4472, 0.2764, 0.2764) fusion-weight tuple sums to 1.0 and matches the golden-ratio derivation. |
| ``convergence_proof_emitter`` | Derives the Lyapunov LaTeX proof from (A, P, λ) and splices it into ``docs/MATH_SPEC.md`` between sentinels so the proof and the certificate cannot diverge. |
| ``benevolence_calibration_report`` | 10-bin reliability diagram + Expected Calibration Error (ECE) over the curated probe set — the *how-well-calibrated* extension to the binary ≥0.99 floor. |
| ``oae_dimensionality_probe`` | Tensor-level dimensionality + L2-bound + identity invariants for R/H/O branches through OAE fusion (pairs with the scalar ``oae_weight_certifier``). |
| ``oae_eigen_monitor`` | Runtime sampling of the OAE fusion-matrix eigenvalues; alerts when the Lyapunov negative-definiteness margin shrinks below a floor. |
| ``ethical_gate_coverage_report`` | AST call-graph traversal: every public surface in ``omni_mercury_engine`` reaches Benevolence, σ_Immutable, *or* GOSNN. |
| ``sigma_immutable_drift_monitor`` | Rolling-window σ band re-evaluation; alerts on band shift beyond tolerance.  Suitable for a systemd ``--user`` timer. |
| ``fairness_subgroup_explorer`` | Intersectional subgroup discovery — cartesian product of sensitive features ranked by DPD/EOD (extends ``bias_audit_standalone``). |

### Cryptography

| Tool | What it proves |
| --- | --- |
| ``sigma_immutable_verifier`` | Verifies the σ_Immutable corpus signature bundle (Ed25519 + ML-DSA-65) — also wired in as ``mercury-agent verify-corpus``. |
| ``pqc_capability_probe`` | Walks the AMA C library at runtime and reports which of {Kyber-1024, ML-DSA-65, SLH-DSA, native HMAC} are real vs stub. |
| ``kat_runner_standalone`` | Re-runs RFC 8032 + FIPS 203/204/205 ACVP-Server vectors outside pytest and emits a signed JSON certificate suitable for an external auditor. |
| ``algorithm_name_drift_gate`` | Scans README/SECURITY/ARCHITECTURE for algorithm claims and cross-checks against ``pqc_backends.py`` exports.  Pre-commit-friendly. |
| ``corpus_resigner`` | When ``sigma_immutable_corpus.json`` is updated, re-signs with Ed25519 + ML-DSA-65 and updates the ``.sig.json`` atomically.  Accepts ``--hsm-uri`` for PKCS#11 HSM-resident keys. |
| ``pqc_handshake_simulator`` | Full ML-KEM-1024 encap/decap + ML-DSA-65 sign/verify with p50/p95/p99 latency.  Paired-claim check with ``pqc_capability_probe`` to refuse silent stub fallback. |
| ``hsm_attestation_probe`` | When ``MERCURY_HSM=pkcs11|tpm2|yubihsm``, fetches and verifies the device attestation chain.  Fails-closed in ``MERCURY_ENV=production``. |
| ``tls_posture_probe`` | For any operator-supplied URL, enumerates negotiated cipher suite, certificate chain, OCSP staple, HSTS, ALPN, and X25519MLKEM768 PQC hybrid availability through ``SafeHTTPClient``. |
| ``slsa_provenance_emitter`` | in-toto SLSA v1.0 provenance attestation for wheel + container image (complements the existing CycloneDX SBOM). |
| ``signed_release_bundle`` | Bundle release manifest + SBOM + SLSA + corpus signature + KAT cert into a tarball signed with Ed25519 + ML-DSA-65. |
| ``secret_scan_baseline`` | Handwritten entropy + regex scanner with ``.secrets.baseline`` allow-list.  Wired pre-commit. No gitleaks/detect-secrets dependency. |
| ``audit_log_signer`` | Append-only JSONL audit log with rolling HMAC chain — each entry HMACs the previous.  HSM URI accepted as key source. |
| ``audit_log_verifier`` | Sibling verifier: re-derives the HMAC chain and fails on any tamper or chain break. |
| ``hwrng_audit`` | Probes ``/dev/hwrng``, RDRAND, kernel hwrng; emits Shannon entropy estimates; fails-closed in production when hwrng absent. |

### Datasets

> **Live-first data policy.**  Mercury Agent does **not** prioritise synthetic
> data.  Live datasets are the source of truth for every detector and every
> certificate.  Synthetic data is permitted *only* as a transient
> reenactment of the most-recently-collected live corpus when the upstream
> live source is genuinely unreachable, and the
> ``live_dataset_protection_gate`` must pass before the run is released.
> Synthetic that drifts from live trips the gate and blocks the release.
> ``synthetic_fallback_auditor`` flags that a fallback occurred;
> ``synthetic_provenance_tag`` records *why* and *from which live snapshot*;
> ``live_dataset_protection_gate`` proves the reenactment is faithful to live.

| Tool | What it proves |
| --- | --- |
| ``loader_reachability_probe`` | Operator-runnable equivalent of the nightly ``dataset-reachability.yml`` workflow.  Reproduces probe outcomes locally for outage debugging. |
| ``dataset_checksum_manifest`` | Emits / verifies a SHA-256 manifest per downloaded dataset.  Closes the integrity gap that synthetic-fallback flagging only partially addresses. |
| ``bias_audit_standalone`` | Runs Fairlearn DPD / EOD / four-fifths metrics on a detector's predictions vs a sensitive attribute. |
| ``synthetic_fallback_auditor`` | Scans benchmark results for datasets using > 50 % synthetic data and flags. |
| ``live_dataset_protection_gate`` | **Defends the live dataset's primacy.**  Live is the reference distribution; any reenactment (emergency synthetic fallback) is judged against live via KS + symmetric-KL per column + AUROC delta.  Fail-closed when the reenactment drifts from live — synthetic is never accepted as a substitute, only as a transient reenactment of the most-recent live corpus. |
| ``dataset_license_auditor`` | Walks every loader for ``DATASET_LICENSE`` declarations and emits SPDX + upstream URL + redistribution terms; pre-commit-friendly. |
| ``pii_scrubber_probe`` | Handwritten regex/heuristic gate (emails, SSNs, US phone, ICD-10, lat/lon ≥5 dp) over loader outputs.  Fails on leak. |
| ``loader_schema_pinner`` | ``--emit`` writes a JSON schema (columns, dtypes, row count) per loader; ``--verify`` re-introspects and fails on schema drift. |
| ``synthetic_provenance_tag`` | When a loader synthesises data, embeds a sidecar provenance JSON with SHA-256 + method + upstream-fallback reason.  Makes ``synthetic_fallback_auditor`` structural. |
| ``network_egress_recorder`` | Parses ``SafeHTTPClient`` JSONL traces and emits per-run egress evidence; validates against the loader allow-list. |

### Benchmarks

| Tool | What it proves |
| --- | --- |
| ``run_hardware_benchmark`` | Records CPU/GPU/memory capability + deterministic matmul/FFT timings so benchmark numbers are reproducibility-anchored. |
| ``benchmark_diff`` | Per-dataset / per-detector diff of two benchmark JSON files with regression detection. |
| ``detector_profiler`` | Per-detector latency + RSS + cache-hit-rate profile.  Reproduces the README's ``<100ms`` and ``>50%`` claims off-CI. |
| ``gosnn_latency_sla_gate`` | Drives ``GOSNNDetector.detect`` for N iterations; asserts the README's <100 ms p50 / <250 ms p95 / ≥50 % cache-hit-rate SLA on every PR. |
| ``memory_leak_sentinel`` | ``tracemalloc`` sustained-load RSS-plateau check; computes bytes-per-iteration slope (second-half vs first-half) and fails on positive drift. |
| ``gpu_capability_probe`` | Sibling to ``pqc_capability_probe``: enumerates CUDA/ROCm/MPS, driver version, FP16/BF16/INT8 support; diffs against the release manifest. |
| ``thermal_throttle_probe`` | Samples ``psutil.sensors_temperatures`` (handwritten ``/sys/class/hwmon`` fallback) during a benchmark window; flags throttled intervals. |

### Configuration & deployment

| Tool | What it proves |
| --- | --- |
| ``config_validator`` | Schema-validates every ``configs/*.yaml`` against a documented contract. |
| ``helm_values_linter`` | Validates ``helm/mercury-agent/values.yaml`` for resource limits, security contexts, network policies, image pinning. |
| ``image_surface_auditor`` | Audits the runtime image / Dockerfile for non-root user, no dev tools, no apt cache, correct entrypoint, correct AMA ``LD_LIBRARY_PATH``. |
| ``workflow_version_drift_gate`` | Verifies ``pyproject.toml @ AMA_REF``, ``ci.yml AMA_REF``, ``pqc-production-check.yml AMA_REF`` all reference the same git tag. |
| ``network_policy_synthesiser`` | Renders a Kubernetes ``NetworkPolicy`` YAML from the egress allow-list and DNS resolution; complements ``helm_values_linter``. |
| ``pod_security_standard_gate`` | Verifies rendered manifests satisfy Pod Security Standards *restricted* profile (``runAsNonRoot``, ``readOnlyRootFilesystem``, ``allowPrivilegeEscalation=false``, ``drop: [ALL]``, ``seccompProfile``). |
| ``dockerfile_lockfile_gate`` | Asserts every apt/apk/pip install pins ``= version`` and every ``FROM`` pins ``@sha256:``; catches drift at build time before ``image_surface_auditor`` audits the built image. |
| ``reproducible_build_probe`` | Builds the wheel twice with ``SOURCE_DATE_EPOCH``, ``PYTHONHASHSEED=0``, ``TZ=UTC`` pinned and diffs SHA-256 digests. |
| ``config_secret_redactor`` | Pre-commit walker over ``configs/*.yaml`` refusing to commit secret-shaped values; reuses ``secret_scan_baseline`` heuristics. |

### Observability & runtime evidence

| Tool | What it proves |
| --- | --- |
| ``gate_trace_probe`` | Exercises every public detect/analyze/predict surface and emits a JSON of which gates (Benevolence, σ_Immutable, GOSNN) fired and in what order. |
| ``killswitch_tester`` | Trips the Phase-5 OODA kill-switch under load and measures trip latency against the documented SLA. |
| ``gosnn_scalar_dump`` | Emits the current omni-scalar values of a GOSNN instance as JSON. |
| ``federated_round_simulator`` | Drives the federated aggregator through a synthetic 3-node round and verifies aggregation + DP noise injection. |

### Release & supply chain

| Tool | What it proves |
| --- | --- |
| ``release_manifest_builder`` | Emits a JSON manifest of (version, python, numpy, AMA_REF, container digest, OAE weights, λ, σ thresholds, benevolence threshold) for the active tag. |
| ``sbom_emitter`` | Emits a CycloneDX 1.5 SBOM after ``pip install -e .[all]``. |
| ``api_contract_diff`` | Snapshots / diffs the public ``omni_mercury_engine.*`` re-export surface.  Catches accidental ABI-breaking removals. |

### ML governance

| Tool | What it proves |
| --- | --- |
| ``model_card_generator`` | Generates a Google-style model card (training data, metrics, fairness audit, limitations, intended use) for a fitted detector. |
| ``adversarial_probe`` | Small perturbations to detection inputs; measures empirical Lipschitz vs the Lyapunov bound. |

## Recommended pre-commit gates

The following tools are safe to run on every commit and have no
network or environment dependencies:

```yaml
- repo: local
  hooks:
    - id: algorithm-name-drift
      name: algorithm-name-drift-gate
      entry: python -m omni_mercury_engine.tools.algorithm_name_drift_gate
      language: system
      pass_filenames: false
    - id: workflow-version-drift
      name: workflow-version-drift-gate
      entry: python -m omni_mercury_engine.tools.workflow_version_drift_gate
      language: system
      pass_filenames: false
    - id: convergence-proof
      name: convergence-proof-emitter
      entry: python -m omni_mercury_engine.tools.convergence_proof_emitter --check
      language: system
      pass_filenames: false
```

## Recommended release-time gates

The following tools should run in the release workflow so each tagged
build ships with a signed evidence bundle:

```bash
mercury-agent verify-corpus
python -m omni_mercury_engine.tools.kat_runner_standalone \
    --output release/kat-cert.json --sign-key-hex "$KAT_SIGN_KEY"
python -m omni_mercury_engine.tools.release_manifest_builder \
    --container-digest "$IMAGE_DIGEST" --output release/manifest.json \
    --sign-key-hex "$RELEASE_SIGN_KEY"
python -m omni_mercury_engine.tools.sbom_emitter \
    --sbom-path release/sbom.json --output release/sbom-cert.json
```

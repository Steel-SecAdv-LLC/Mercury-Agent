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

### Cryptography

| Tool | What it proves |
| --- | --- |
| ``sigma_immutable_verifier`` | Verifies the σ_Immutable corpus signature bundle (Ed25519 + ML-DSA-65) — also wired in as ``mercury-agent verify-corpus``. |
| ``pqc_capability_probe`` | Walks the AMA C library at runtime and reports which of {Kyber-1024, ML-DSA-65, SLH-DSA, native HMAC} are real vs stub. |
| ``kat_runner_standalone`` | Re-runs RFC 8032 + FIPS 203/204/205 ACVP-Server vectors outside pytest and emits a signed JSON certificate suitable for an external auditor. |
| ``algorithm_name_drift_gate`` | Scans README/SECURITY/ARCHITECTURE for algorithm claims and cross-checks against ``pqc_backends.py`` exports.  Pre-commit-friendly. |
| ``corpus_resigner`` | When ``sigma_immutable_corpus.json`` is updated, re-signs with Ed25519 + ML-DSA-65 and updates the ``.sig.json`` atomically. |

### Datasets

| Tool | What it proves |
| --- | --- |
| ``loader_reachability_probe`` | Operator-runnable equivalent of the nightly ``dataset-reachability.yml`` workflow.  Reproduces probe outcomes locally for outage debugging. |
| ``dataset_checksum_manifest`` | Emits / verifies a SHA-256 manifest per downloaded dataset.  Closes the integrity gap that synthetic-fallback flagging only partially addresses. |
| ``bias_audit_standalone`` | Runs Fairlearn DPD / EOD / four-fifths metrics on a detector's predictions vs a sensitive attribute. |
| ``synthetic_fallback_auditor`` | Scans benchmark results for datasets using > 50 % synthetic data and flags. |

### Benchmarks

| Tool | What it proves |
| --- | --- |
| ``run_hardware_benchmark`` | Records CPU/GPU/memory capability + deterministic matmul/FFT timings so benchmark numbers are reproducibility-anchored. |
| ``benchmark_diff`` | Per-dataset / per-detector diff of two benchmark JSON files with regression detection. |
| ``detector_profiler`` | Per-detector latency + RSS + cache-hit-rate profile.  Reproduces the README's ``<100ms`` and ``>50%`` claims off-CI. |

### Configuration & deployment

| Tool | What it proves |
| --- | --- |
| ``config_validator`` | Schema-validates every ``configs/*.yaml`` against a documented contract. |
| ``helm_values_linter`` | Validates ``helm/mercury-agent/values.yaml`` for resource limits, security contexts, network policies, image pinning. |
| ``image_surface_auditor`` | Audits the runtime image / Dockerfile for non-root user, no dev tools, no apt cache, correct entrypoint, correct AMA ``LD_LIBRARY_PATH``. |
| ``workflow_version_drift_gate`` | Verifies ``pyproject.toml @ AMA_REF``, ``ci.yml AMA_REF``, ``pqc-production-check.yml AMA_REF`` all reference the same git tag. |

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
mercury-agent verify-corpus --require-mldsa
python -m omni_mercury_engine.tools.kat_runner_standalone \
    --output release/kat-cert.json --sign-key-hex "$KAT_SIGN_KEY"
python -m omni_mercury_engine.tools.release_manifest_builder \
    --container-digest "$IMAGE_DIGEST" --output release/manifest.json \
    --sign-key-hex "$RELEASE_SIGN_KEY"
python -m omni_mercury_engine.tools.sbom_emitter \
    --sbom-path release/sbom.json --output release/sbom-cert.json
```

# WS-E — Neural-submodule completeness sweep

Coverage for every neural submodule in **this PR's scope** — the neuro-symbolic
fusion stack and the embedded sub-nets the PR addresses. Each is either
fully-covered-and-active or explicitly **quarantined with a recorded reason**.
No half-wired module is left in this scope: every row has a dataset, seed,
metric, artifact, tests, and an off-path/quarantine contract.

| Module | Dataset (provenance) | Seed(s) | Metric | Artifact | Tests | Off-path / quarantine contract | Status |
|---|---|---|---|---|---|---|---|
| `OmniFusionModel` (fusion net) | ADBench (MIT) + domain loaders | 42 | ROC-AUC, oracle-F1 | `mercury_benchmark_results.json` (CI) | `test_fusion_*` (32) | baseline | **ACTIVE** (default) |
| `MercuryAnomalyDetector` (WS-A guard) | ADBench (MIT), 8 fixed sets | 42 | AUC/F1 floors | `anomaly_regression_baseline.json` | `test_anomaly_regression_guard` | deterministic; CI gate | **ACTIVE** |
| `SymbolicConstraintModule` (LTN, #262) | ADBench (MIT) | 0,1,2 | ΔAUC, ΔFP | `neurosymbolic_ablation.json` | `test_symbolic_constraint`, `test_fusion_symbolic_cotraining` | `symbolic_weight=0` byte-identical | **QUARANTINE** (sub-threshold) |
| `DomainEncoderStack` (WS-B) | ADBench (MIT) | 0,1,2 | ΔAUC | `domain_encoder_ablation.json` | `test_domain_encoders` (16), `test_fusion_domain_encoder` (5) | `domain_encoder=False` parity (≤1e-15) | **QUARANTINE** (sub-threshold) |
| `BinaryConformalClassifier` (#242/#262) | ADBench + synthetic | fixed | coverage @ {0.8,0.9,0.95} | in `test_*conformal` | `test_binary_conformal`, `test_fusion_conformal` | additive serve-path | **ACTIVE** (uncertainty) |
| `SchumannHarmonicAnalyzer` (WS-C) | NOAA Kp/GOES (public domain) labels; **synthetic** ELF | 0,1,2 | ROC-AUC | `schumann_eval.json` | `test_schumann_labeling` (5) | `load_neural_weights()` gate; FFT-physics fallback | **QUARANTINE** (synthetic signal + unstable) |
| `ConsciousnessFieldAnalyzer` (WS-D) | GCP (real **unreachable**); **synthetic** null | 0,1,2 | Stouffer Z, network var | `parapsych_eval.json` | `test_gcp_ingest` (5) | abstains untrained; never asserts psi | **QUARANTINE** (data unreachable; null) |

## Off-path / determinism invariants (asserted in tests)

* `symbolic_weight=0` and `domain_encoder=False` leave the fusion path
  **structurally identical** and numerically identical within the baseline's own
  ~1e-15 float non-determinism (the baseline is not bit-deterministic; this is
  pre-existing, not introduced here — see `docs/DOMAIN_ENCODERS.md`).
* `MercuryAnomalyDetector` is byte-identical across repeated runs (WS-A guard).
* Quarantined sub-nets are deterministic on fixed input (the #262 fix) and
  off-by-default; activation is an explicit, documented opt-in.

## Provenance contract (met by every row)

Dataset id + license + URL + content hash, RNG seed(s), metric definition,
artifact path, and commit are recorded — in `anomaly_regression_baseline.json`,
the ablation artifacts, and the labeling/ingestion provenance dicts.

## Out of scope (flagged, not silently skipped)

The wider tree has ~40 additional `nn.Module` classes (visual/VLM, SOTA
TranAD/MAAT, geological detectors, etc.). They are **not** part of this PR's
neuro-symbolic-fusion scope and were not re-audited here. A full-tree neural
audit is a separate, larger effort; this table is exhaustive for the modules
this PR touches or owns.

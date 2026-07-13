# WS-E — Neural-submodule completeness sweep

Applies to Mercury Agent **v2.1.x**. Last updated: 2026-07-11.

> **GENERATED — do not edit by hand.** Regenerate with
> `python scripts/neural_coverage.py --update`; CI runs
> `python scripts/neural_coverage.py --check`, which fails if this file is out of
> sync *or* if any referenced module, test file, or committed artifact is
> missing. The table therefore cannot silently rot.

Coverage for every neural submodule in the neuro-symbolic fusion scope plus the
verification gates that protect it. Each is either fully-covered-and-active or
explicitly **quarantined with a recorded reason**. No half-wired module is left
in scope: every row has a dataset, seed, metric, artifact, tests, and an
off-path/quarantine contract.

## Neural submodules

| Module | Dataset (provenance) | Seed(s) | Metric | Artifact | Tests | Off-path / quarantine contract | Status |
|---|---|---|---|---|---|---|---|
| `OmniFusionModel` | ADBench (MIT) + domain loaders | 42 | ROC-AUC, oracle-F1 | `mercury_benchmark_results.json` (CI) | `test_fusion_*` | baseline | **ACTIVE (default)** |
| `MercuryAnomalyDetector (WS-A guard)` | ADBench (MIT), 8 fixed sets | 42 | AUC/F1 floors | `anomaly_regression_baseline.json` | `test_anomaly_regression_guard` | deterministic; CI gate | **ACTIVE** |
| `SymbolicConstraintModule (LTN)` | ADBench (MIT) | 0,1,2 | ΔAUC, ΔFP (+confound guard) | `neurosymbolic_ablation.json` | `test_symbolic_constraint`, `test_fusion_symbolic_cotraining` | `symbolic_weight=0` byte-identical | **QUARANTINE (sub-threshold)** |
| `DomainEncoderStack (WS-B)` | ADBench (MIT) | 0,1,2 | ΔAUC (+confound guard) | `domain_encoder_ablation.json` | `test_domain_encoders` (16), `test_fusion_domain_encoder` (5) | `domain_encoder=False` parity (≤1e-15) | **QUARANTINE (sub-threshold)** |
| `BinaryConformalClassifier` | ADBench + synthetic | fixed | coverage @ {0.8,0.9,0.95} | in `test_*conformal` | `test_binary_conformal`, `test_fusion_conformal` | additive serve-path | **ACTIVE (uncertainty)** |
| `SchumannHarmonicAnalyzer (WS-C)` | NOAA Kp/GOES (public domain) labels; **synthetic** ELF | 0,1,2 | ROC-AUC | `schumann_eval.json`, `schumann_diagnostic.json` | `test_schumann_labeling` (5), `test_schumann_stability` (5) | stable recipe (minibatch); quarantine on data blocker | **QUARANTINE (synthetic signal; training stabilized)** |
| `ConsciousnessFieldAnalyzer (WS-D)` | GCP (real **unreachable**); **synthetic** null | 0,1,2 | Stouffer Z, network var | `parapsych_eval.json` | `test_gcp_ingest` (5) | abstains untrained; never asserts psi | **QUARANTINE (data unreachable; null)** |

## Verification gates

| Module | Dataset (provenance) | Seed(s) | Metric | Artifact | Tests | Off-path / quarantine contract | Status |
|---|---|---|---|---|---|---|---|
| `label_provenance (WS-A leak gate)` | all 40 dataset loaders | — | circular-label audit | registry in-module | `test_label_provenance_gate` (11) | repo-wide; CI `--check` | **ACTIVE (gate)** |
| `ablation_guard (WS-B confound guard)` | paired ablation AUCs | — | inverted-ranking detection | wired into both ablations | `test_ablation_guard` (10) | forces QUARANTINE on confound | **ACTIVE (gate)** |
| `event_coincidence (WS-D null-test)` | any score stream + event catalog | permutation null | pre-registered p, FDR/Bonferroni | `spaceweather_coincidence.json` | `test_event_coincidence` (offline) | pre-registered; transparent null | **ACTIVE (gate)** |

## Off-path / determinism invariants (asserted in tests)

* `symbolic_weight=0` and `domain_encoder=False` leave the fusion path
  **structurally identical** and numerically identical within the baseline's own
  ~1e-15 float non-determinism (the baseline is not bit-deterministic; this is
  pre-existing, not introduced here — see `docs/DOMAIN_ENCODERS.md`).
* `MercuryAnomalyDetector` is byte-identical across repeated runs (WS-A guard).
* Quarantined sub-nets are deterministic on fixed input (the #262 fix) and
  off-by-default; activation is an explicit, documented opt-in.
* `SchumannHarmonicAnalyzer.confidence_logits` is byte-identical to the sigmoid
  head at inference (WS-C); the seed-instability was a full-batch optimisation
  artifact, root-caused and fixed (mini-batch) — see
  `docs/SCHUMANN_PREREGISTRATION.md`.

## Provenance contract (met by every row)

Dataset id + license + URL + content hash, RNG seed(s), metric definition,
artifact path, and commit are recorded — in `anomaly_regression_baseline.json`,
the ablation artifacts, and the labeling/ingestion provenance dicts.

## Self-verification (the gate)

`scripts/neural_coverage.py --check` verifies, for every row above, that the
named source symbol, every referenced test file, and every committed artifact
actually exist, and that this document matches the registry byte-for-byte. It
runs in CI and as `tests/docs/test_neural_coverage_gate.py`.

## Out of scope (flagged, not silently skipped)

The wider tree contains 171 `nn.Module` subclasses in total (measured and
CI-gated in the README Codebase Scale block); the neuro-symbolic-fusion
submodules audited here are the critical subset. The remainder — visual/VLM,
SOTA TranAD/MAAT, the streaming/statistical detector tier, and the
geological/space detectors — are **not** part of that scope and were not
re-audited here. A full-tree neural audit is a separate, larger effort; this
table is exhaustive for the modules in scope.

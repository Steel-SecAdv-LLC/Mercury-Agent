# Mercury Agent - Deprecation and Migration Guide

| Property | Value |
|----------|-------|
| Document Version | 1.5 |
| Last Updated | 2026-10-21 |
| Applies to | Mercury Agent v3.0.x |

This document tracks deprecated modules, classes, methods, and parameters in Mercury Agent.

For breaking changes already landed in the v1.7 development cycle (the
small number of items where preservation was overridden by a security
or correctness criterion), see [`docs/MIGRATION-1.6-to-1.7.md`](docs/MIGRATION-1.6-to-1.7.md)
and the `## v1.7 Removals (security/correctness exceptions)` section
below.

## Policy Statement

**PRESERVATION PRINCIPLE:** Deprecated items remain functional indefinitely. No module, class, method, or parameter will be removed without documented justification meeting the following criteria:

### Removal Justification Requirements

For any item to be considered for removal, ALL of the following must be documented:

1. **Security Vulnerability**: The deprecated code introduces a security risk that cannot be mitigated
2. **Fundamental Incompatibility**: Continued support creates architectural conflicts with core functionality
3. **Zero Active Usage**: Telemetry confirms no production usage for 24+ months
4. **Migration Complete**: All known integrations have transitioned to replacement APIs
5. **Community Consensus**: RFC process with minimum 90-day comment period completed

Until these criteria are met, deprecated items operate via compatibility shims that:
- Emit warnings (suppressible via environment variable)
- Route to current implementations
- Maintain full backward compatibility

---

## Quick Reference

| Deprecated | Replacement | Status |
|------------|-------------|--------|
| `core.self_healing` | `resilience.self_healing` | **Preserved** - Compatibility shim active |
| `core.neurosymbolic_engine` | `core.code_analysis` / `models.neurosymbolic` | **Preserved** - Compatibility shim active |
| Legacy scalar names (39) | Omni-prefixed names | **Preserved** - Both naming conventions supported |
| `sigma_immutable` param | `ethical_compliance_threshold` | **Preserved** - Alias mapping active |
| `lambda_lyapunov` param | `convergence_rate` | **Preserved** - Alias mapping active |
| `enable_quantum_terms` | `enable_optimization_terms` | **Preserved** - Alias mapping active |
| `AnomalyFusionEquation` / `AAFE*` | `OmniAvaEquation` / `OAE*` | **Preserved** - Alias mapping active |
| `IntelligenceFusionEngine(golden_ratio_weights=...)` | _none needed_ (decisions come from the reported evidence or a measured checkpoint) | **Preserved** - accepted in its released position, no effect, `DeprecationWarning` |
| `compliance.tlp_handler` (was `security.tlp_handler`) | `omni_mercury_engine.compliance.tlp_handler` | **Relocated** - governance lives in `compliance/`; `security/` is reserved for implementation primitives |
| `SafeHTTPClient(..., allow_untrusted=True)` | `SafeHTTPClient(..., user_configured=True[, allow_private=True])` | **Removed in v1.7** - security exception (see §6) |
| `MercuryVoice(enable_llm=True)` (no `llm_provider`) | Explicit `llm_provider=` + `llm_model_name=` (+ `llm_revision=` for remote HF) | **Removed in v1.7** - silent `MockLLMAdapter` fallback removed (see §6) |
| `strict_ethics=False` engine flag | _no replacement_ | **Removed in v1.7** - dual hard gates are non-negotiable (see §6) |
| `result["gosnn_metadata"]["fallback_mode"] is True` path | `EthicalConstraintViolationError(check="gosnn_unavailable")` | **Removed in v1.7** - σ_Immutable second hard gate (see §6) |
| `ml.ppo_trainer` (`PPOTrainer` / `MultiEnvPPOTrainer` / `PPOConfig` / `TrainingStats` / `ConvergenceMonitor` / `CheckpointCallback`) | The same names, rebuilt natively in PyTorch (no stable-baselines3) | **Rebuilt in v3.0** - same entry points; changed behaviour listed in §6.6 |
| Geological `SolarFlareDetector` duplicate + `GeomagneticHMM` | Canonical `space.solar_storm_detector.SolarFlareDetector` (re-exported from `detectors.geological`) | **Removed in v3.0** - correctness exception (see §6.7) |
| `load_nasa_fireball_data` / `load_nasa_close_approach_data` / `load_nasa_sentry_data` | `data_sources.jpl_ssd.JPLFireballSource` / `NASANeoWsSource` / `data_sources.jpl_ssd.JPLSentrySource` | **Removed in v3.0** - correctness exception (see §6.8) |
| `USGSVolcanoSource.US_VOLCANOES` simulated volcano table | Real USGS HANS public API in the same class | **Removed in v3.0** - correctness exception (see §6.9) |
| `flood_detector.TopographicRunoffPredictor` + `enable_runoff` | `SoilSaturationModel` physics runoff coefficient (unchanged existing path) | **Removed in v3.0** - correctness exception (see §6.10) |
| `security.pqc_backends.slhdsa_sign_internal` | `slhdsa_sign_addrnd` (FIPS 205 §10.2, caller-supplied `addrnd`); production: `slhdsa_sign` | **Removed in v3.0** - security exception, aligned with AMA Cryptography v5 INVARIANT-50 (see §6.11) |
| `EMPPredictionResult.intentional_attack_probability` (`0.0` / untrained-network float) and `E3PulseDetector.detect_e3_pulse()["gic_amplitude_a"]` (`0.1 × \|dB/dt\|`) | The trained IEMI network's probability after `EMPDetector.load_neural_weights`; `detect_e3_pulse()["peak_dbdt_nt_per_s"]` / `["grid_impact"]`; `space.gic_detector.GICDetector` for the geoelectric field | **Changed in v3.0** - correctness exception: field and key kept in place, value `None` (no truthful number exists without a trained network or a network model; see §6.12) |
| `security.counterintelligence.OverwatchNexus` / `OverwatchNexusResult` | `security.counterintelligence.CounterintelligenceAssessor` | **Removed in v3.0** - correctness exception: scored reports it fabricated itself (see §6.13) |
| `truth_decipher.TruthDecipherFramework.decipher_truth` / `TruthDecipherResult` | `truth_decipher.StatementVerifier` (the `TruthDecipherFramework` name is kept as its alias) | **Removed in v3.0** - correctness exception: fixed ethics inputs, every case recorded as a success (see §6.14) |
| `security.quantum_risk_cyber.QuantumRiskCyber` / `PostQuantumMigrationPlanner` / `QuantumThreat` / `ThreatLevel` / `CryptoSystem` | `security.quantum_risk_cyber.classify_algorithm` / `assess_asset` / `prioritize` / `read_cbom` / `audit_mercury_posture` | **Removed in v3.0** - security exception: flagged ML-DSA as quantum-vulnerable, counted SIKE and Rainbow as post-quantum (see §6.15) |

---

## Suppressing Deprecation Warnings

Set the environment variable to suppress warnings during migration:

```bash
export MERCURY_AGENT_SUPPRESS_DEPRECATION_WARNINGS=1
```

---

## 1. Module Compatibility Shims

### 1.1 `core/self_healing.py`

**Status:** Active compatibility shim - routes to `resilience.self_healing`

**Original import (continues to work):**
```python
from omni_mercury_engine.core.self_healing import (
    SelfHealingEngine,
    AdaptiveDefenseSystem,
    CRISPRInspiredSelfHealing,
)
```

**Recommended import (preferred for new code):**
```python
from omni_mercury_engine.resilience.self_healing import (
    SelfHealingEngine,
    AdaptiveDefenseSystem,
)
# Note: CRISPRInspiredSelfHealing is an alias for AdaptiveDefenseSystem
```

**Justification for preservation:**
- `core.self_healing` is used in existing integrations and documentation
- The compatibility shim adds negligible overhead (single import redirect)
- No security concerns with maintaining the alias

---

### 1.2 `core/neurosymbolic_engine.py`

**Status:** Active compatibility shim - routes to specialized modules

**Original import (continues to work):**
```python
from omni_mercury_engine.core.neurosymbolic_engine import (
    CodeAnalysisEngine,
    NeurosymbolicEngine,
)
```

**Recommended import (preferred for new code):**
```python
# For AST-based code analysis:
from omni_mercury_engine.core.code_analysis import CodeAnalysisEngine

# For LTN-based anomaly detection:
from omni_mercury_engine.models.neurosymbolic import NeurosymbolicEngine
```

**Justification for preservation:**
- Module refactoring separated concerns but original API should remain accessible
- Research references and papers cite the original import paths
- Compatibility shim maintains semantic equivalence

---

### 1.3 `core/di.py` (deprecated module, 2026-07-21)

**Status:** Deprecated — functional, contract-pinned, no replacement shim
(there is nothing to route to; Mercury composes components by direct
construction).

**Deprecated import (continues to work, emits `DIDeprecationWarning`):**
```python
from omni_mercury_engine.core.di import ServiceContainer, ComponentFactory
```

**Recommended pattern (what the codebase actually does):**
```python
# Compose components by direct construction, e.g.:
from omni_mercury_engine.engine import OmniMercuryEngine
engine = OmniMercuryEngine()
```

**Evidence for deprecation (steel/maint1/2-coverage round):**
- Zero importers anywhere in src/tests/scripts/tools since inception —
  earlier claims of importers were regex false positives.
- Its built-in `detector_map`/`model_map` had bit-rotted to class names
  that never existed post-rename (fixed in the same round) — only
  possible in dead code.
- Constructor injection is inert for every class in this codebase:
  `from __future__ import annotations` (used across src/) makes
  `__init__` annotations strings, which the injector does not resolve.

**Justification for preservation:**
- Public import path under `core/`; external integrations may construct
  containers even though nothing in-repo does.
- Preservation is cheap: the module is self-contained and its contract
  is pinned by `tests/core/test_di.py` (56 tests, 94% coverage), so it
  cannot rot silently again.
- Removal would require the full policy criteria above (24-month
  zero-usage telemetry, RFC), which are not met.

---

## 2. Scalar Name Aliases (Dual Support)

The Global Omni-Scalar Network (GOSNN) supports **both** legacy and omni-prefixed naming conventions. No migration is required, but new code should prefer the omni-prefixed names. Every built-in scalar carries the `omni` prefix (pinned by `tests/core/test_gosnn_scalar_naming.py`).

### Implementation

GOSNN keeps a legacy-to-canonical alias table (`_initialize_legacy_aliases`). Reads resolve through `resolve_scalar_name`; writes through `register_scalars` resolve a legacy name to its canonical scalar when that scalar exists in the target group, so a legacy write cannot add a second entry and shift the σ_Immutable layout. There is no `set_scalar` method. Resolution logs at DEBUG level and emits no deprecation warning.

```python
# Both of these return the same value:
value = gosnn.get_scalar("morality_scalar")      # Legacy name
value = gosnn.get_scalar("omnimorality")         # New name

# Both of these write the same scalar:
gosnn.register_scalars("my_component", {"morality_scalar": 0.95})   # Legacy name
gosnn.register_scalars("my_component", {"omnimorality": 0.95})      # New name
```

Values written to the ETHICAL group must be finite and within the σ_Immutable network's trained support `[0, 2.0]` (`ETHICAL.SIGMA_TRAINED_SUPPORT_HIGH`); `register_scalars` raises `ValueError` and writes nothing otherwise. Names under a metric-only prefix such as `omni_diag_` are exempt because they never enter the σ vector.

Renaming a scalar is safe only in place: σ_Immutable depends on the order of the operational vector, not on names. After a rename, regenerate the baseline with `python scripts/harvest_sigma_baseline.py` (its diff should touch only the name lists) and re-run `python scripts/sigma_immutable_discrimination_check.py`. The signed training corpus holds no names and does not change.

### Ethical Scalars

| Legacy Name | Omni Name | Status |
|-------------|-----------|--------|
| `morality_scalar` | `omnimorality` | Both supported |
| `empathy_scalar` | `omniempathy` | Both supported |
| `compassion_scalar` | `omnicompassion` | Both supported |
| `forgiveness` | `omniforgiveness` | Both supported |
| `love_scalar` | `omnilove` | Both supported |
| `determination_scalar` | `omnidetermination` | Both supported |
| `loyalty_scalar` | `omniloyalty` | Both supported |
| `integrity_scalar` | `omniintegrity` | Both supported |
| `wisdom_scalar` | `omniwisdom` | Both supported |
| `justice_scalar` | `omnijustice` | Both supported |
| `altruism_scalar` | `omnialtruism` | Both supported |
| `hope_scalar` | `omnihope` | Both supported |
| `courage_scalar` | `omnicourage` | Both supported |
| `accountability_scalar` | `omniaccountability` | Both supported |
| `transparency_weight` | `omnitransparency` | Both supported |
| `explainability_factor` | `omniexplainability` | Both supported |
| `benevolence` | `omnibenevolence` | Both supported |
| `equity` | `omniequity` | Both supported |
| `civilization_first_principle` | `omni_civilization_first_principle` | Both supported |
| `bias_audit_compliance` | `omni_bias_audit_compliance` | Both supported |

### Cosmic Scalars

| Legacy Name | Omni Name | Status |
|-------------|-----------|--------|
| `universe_adapt` | `omniuniverse_adapt` | Both supported |
| `telos_scalar` | `omnitelos` | Both supported |
| `black_hole_entropy_eth` | `omni_black_hole_entropy` | Both supported |
| `harmonic_singularity_bridge` | `omni_harmonic_singularity` | Both supported |
| `golden_ratio_phi` | `omni_golden_ratio_phi` | Both supported |

### Quantum Scalars

| Legacy Name | Omni Name | Status |
|-------------|-----------|--------|
| `quantum_weight` | `omniquantum_weight` | Both supported |
| `entanglement_risk` | `omnientanglement_risk` | Both supported |
| `quantum_entanglement_weight` | `omniquantum_entanglement` | Both supported |
| `neuro_quantum` | `omnineuroquantum` | Both supported |
| `consciousness_coherence` | `omniconsciousness_coherence` | Both supported |

### Humanitarian Scalars

| Legacy Name | Omni Name | Status |
|-------------|-----------|--------|
| `crisis_response_boost` | `omnicrisis_response` | Both supported |
| `disaster_response_boost` | `omnidisaster_response` | Both supported |
| `pandemic_monitoring` | `omnipandemic_monitoring` | Both supported |
| `missing_persons_priority` | `omnimissing_persons_priority` | Both supported |
| `medical_discovery_boost` | `omnimedical_discovery` | Both supported |

### Security Scalars

| Legacy Name | Omni Name | Status |
|-------------|-----------|--------|
| `threat_detection_sensitivity` | `omnithreat_detection` | Both supported |
| `quantum_resistance` | `omniquantum_resistance` | Both supported |
| `encryption_strength` | `omniencryption_strength` | Both supported |
| `audit_compliance` | `omniaudit_compliance` | Both supported |

---

## 3. Parameter Aliases

### 3.1 OmniAvaEquation (3R Mechanism)

**Status:** Alias mapping active - both parameter names accepted

| Legacy Parameter | Preferred Parameter | Status |
|-----------------|---------------------|--------|
| `sigma_immutable` | `ethical_compliance_threshold` | Both supported |
| `lambda_lyapunov` | `convergence_rate` | Both supported |
| `sigma_immutable_override` | `ethical_threshold_override` | Both supported |

**Legacy class name (continues to work):**
```python
from omni_mercury_engine.core.three_r.fusion import AnomalyFusionEquation
aafe = AnomalyFusionEquation(
    sigma_immutable=0.93,
    lambda_lyapunov=0.25,
)
result = aafe.compute(
    recursion_score=0.8,
    resonance_score=0.85,
    optimization_score=0.9,
    sigma_immutable_override=0.95,
)
```

**Preferred usage (recommended for new code):**
```python
from omni_mercury_engine.core.three_r.fusion import OmniAvaEquation
oae = OmniAvaEquation(
    ethical_compliance_threshold=0.93,
    convergence_rate=0.25,
)
result = oae.compute(
    recursion_score=0.8,
    resonance_score=0.85,
    optimization_score=0.9,
    ethical_threshold_override=0.95,
)
```

---

### 3.2 EvolutionConfig (Double Helix Engine)

**Status:** Alias mapping active - both property names accepted

| Legacy Property | Preferred Property | Status |
|----------------|-------------------|--------|
| `enable_quantum_terms` | `enable_optimization_terms` | Both supported |

**Original usage (continues to work):**
```python
config = EvolutionConfig(enable_quantum_terms=True)
```

**Preferred usage (recommended for new code):**
```python
config = EvolutionConfig(enable_optimization_terms=True)
```

---

## 4. Method Aliases

### 4.1 SelfHealingEngine

**Status:** Method aliases active - both method names work identically

| Legacy Method | Preferred Method | Status |
|--------------|------------------|--------|
| `save_signature_library(filepath)` | `save_library(filepath)` | Both supported |
| `load_signature_library(filepath)` | `load_library(filepath)` | Both supported |

---

## 5. Auto-Deprecated Features

### 5.1 Low-Precision Indicators

The `IndicatorSystem` automatically deprecates indicators with:
- Precision < 0.3 (> 70% false positive rate)
- Trigger count >= 10

These indicators receive `IndicatorStatus.DEPRECATED` status.

**Note:** This is a runtime performance optimization, not a code removal. Deprecated indicators remain in the system but are weighted lower in fusion calculations.

To disable auto-deprecation:
```python
indicator_system = IndicatorSystem(enable_auto_deprecation=False)
```

---

## 6. Removals (security/correctness exceptions)

The five-criteria preservation policy at the top of this document is the
default. The surfaces below were **removed** because preservation was
incompatible with the project's hard ethical gate contract, with
documented SSRF / DNS-rebinding defence, or with the fail-loud
correctness doctrine (stub collaborators must hard-fail rather than
silently degrade).  Each entry records the criterion that overrode
preservation, the replacement surface, and the in-tree regression test
that pins the new behaviour.  §6.1–6.4 were removed in the v1.7
development cycle (§6.5 records a relocation, not a removal). §6.6–6.15
ship in v3.0, the first release after 2.0.0 (2.1.0 was prepared but never
released): §6.6–6.10 were removed across the improvement and live-wiring
passes (§6.6 since rebuilt under its released names) and §6.11 with the AMA
Cryptography v5 migration. §6.12 records a
value change, not a removal: the surfaces stay, their fabricated values do
not. §6.13-6.15 record three modules removed on `steel/version-cov3` and
rebuilt from scratch on `steel/version-cov4` under their released module
paths; the released classes are not restored. §7 lists every other 2.0.0
name that v3.0 renames or removes.

### 6.1 `SafeHTTPClient.allow_untrusted=True`

* **Removed by:** PR #210
* **Override criterion:** §1 (Security Vulnerability) — the kwarg was a
  per-call bypass of the `TrustedEndpoints.TRUSTED_DOMAINS` allowlist
  that had no production caller and could be misused to pivot through
  an off-allowlist host.
* **Replacement:** call `SafeHTTPClient` directly with
  `user_configured=True` so the private-network / IMDS gate fires
  explicitly. Add `allow_private=True` for intentional private-VPC
  use; IMDS / loopback / multicast / reserved / CGNAT ranges remain
  blocked even then.
* **Regression test:** `tests/security/test_safe_http.py::TestMigrationFromAllowUntrusted`
* **Migration guide:** [`docs/MIGRATION-1.6-to-1.7.md`](docs/MIGRATION-1.6-to-1.7.md) §1

### 6.2 `MercuryVoice(enable_llm=True)` silent `MockLLMAdapter` fallback

* **Removed by:** v1.7 audit Phase 2 (May 2026)
* **Override criterion:** §1 (Security Vulnerability) — a silent
  heuristic-only adapter masquerading as an LLM violates the
  production-mode primitive's "stub collaborators must hard-fail rather
  than silently degrade" contract. `MockLLMAdapter.__init__` now raises
  `NotImplementedError`; `MercuryVoice(enable_llm=True)` requires an
  explicit `llm_provider=` argument.
* **Replacement:** `MercuryVoice(enable_llm=True, llm_provider="huggingface",
  llm_model_name="...", llm_revision="<40-char SHA>")` (remote HF
  requires the revision pin enforced by `SafeHFLoader`; absolute local
  paths do not). Implemented providers: `huggingface`, `ollama`,
  `openai`, `anthropic`, `xai`, `gemini`, `cohere`, `deepseek`,
  `cursor`, `template`.
* **Regression test:** `tests/narrative/test_voice_llm.py`
* **Migration guide:** [`docs/MIGRATION-1.6-to-1.7.md`](docs/MIGRATION-1.6-to-1.7.md) §4

### 6.3 `strict_ethics=False` engine flag

* **Removed by:** v1.7 σ_Immutable promotion (preceded the v1.7 cut)
* **Override criterion:** §2 (Fundamental Incompatibility) — the flag
  let a public caller turn off the dual hard ethical gates, which is
  structurally incompatible with the "advisory mode does not exist"
  decision-boundary contract. The flag is now ignored on read; the
  gates always run.
* **Replacement:** there is no replacement at the public surface. Tests
  that genuinely need to bypass the gate must set the auditable
  module-level flag `omni_mercury_engine.engine._GOSNN_TESTING_BYPASS
  = True`; using it from production code is a contract violation.
* **Regression test:** `tests/ethical/test_hard_enforcement.py`
* **Migration guide:** [`docs/MIGRATION-1.6-to-1.7.md`](docs/MIGRATION-1.6-to-1.7.md) §2

### 6.4 `result["gosnn_metadata"]["fallback_mode"] is True` GOSNN-unavailable path

* **Removed by:** v1.7 σ_Immutable promotion
* **Override criterion:** §2 (Fundamental Incompatibility) — a
  "fallback to no-GOSNN" path is a silent disablement of the second
  hard ethical gate. v1.7 raises
  `EthicalConstraintViolationError(check="gosnn_unavailable")` instead.
* **Replacement:** catch the new exception and inspect `exc.check`;
  the `gosnn_metadata.fallback_mode` field is gone.
* **Regression test:** `tests/ethical/test_hard_enforcement.py`
* **Migration guide:** [`docs/MIGRATION-1.6-to-1.7.md`](docs/MIGRATION-1.6-to-1.7.md) §2

### 6.5 `compliance.tlp_handler` (location move, not a removal)

This is a relocation, not a removal. The implementation moved from
`omni_mercury_engine.security.tlp_handler` to
`omni_mercury_engine.compliance.tlp_handler` (PR #223) because
`security/` is reserved for implementation primitives (crypto, PQC,
threat detection, audit logging) and governance frameworks live in
`compliance/`. A repository-wide `git grep` confirmed no external call
sites at the time of relocation; no backwards-compatibility shim was
added. New code must use the `compliance.` import path.

### 6.6 `ml.ppo_trainer` (`PPOTrainer` / `MultiEnvPPOTrainer`): removed, then rebuilt

* **Removed by:** the v3.0 improvement pass (July 2026); **rebuilt** natively
  in v3.0 (October 2026) under the released module path and names.
* **Why the 2.0.0 module went:** §2 (Fundamental Incompatibility) + §3
  (Zero Active Usage). It was a stable-baselines3 wrapper that was dead in
  every supported install profile: `stable_baselines3` appears in no
  dependency extra (it was only a mypy override) and `gymnasium` nowhere at
  all, so `PPOTrainer.__init__` raised `NotImplementedError`
  unconditionally. With SB3 hand-installed it was *still* broken — its
  custom callback class lacks SB3's `init_callback` interface, so
  `model.learn()` raised `AttributeError`, which `pretrain()` swallowed and
  then logged "Pretraining complete" with zeroed stats: silent fake success,
  violating the §6.2 "stub collaborators must hard-fail" contract. A
  `_mock_pretrain` residue fabricated reward/convergence numbers, and
  `train_online`/`evaluate` silently substituted random actions when no
  model was loaded. Despite being import-guarded as if lazy, the module was
  eagerly imported on every `[ml]` install.
* **The rebuild:** PPO implemented in PyTorch with no reinforcement-learning
  dependency, plus `ml.rl_envs` (batched `CartPole-v1` / `Pendulum-v1`,
  bit-identical to Gymnasium's, and an adapter for any Gymnasium-API
  environment). It loads lazily. Every 2.0.0 entry point exists and works:
  `PPOConfig` (the 2.0.0 fields and defaults, plus new fields),
  `TrainingStats`, `BaseCallback`, `ConvergenceMonitor`,
  `CheckpointCallback`, `PPOTrainer.pretrain` / `train_online` /
  `save_model` / `load_model` / `evaluate` / `get_training_summary`,
  `MultiEnvPPOTrainer.pretrain_on_benchmarks`, `create_ppo_trainer`,
  `create_multi_env_trainer`. Measured against stable-baselines3 on identical
  settings: see `benchmarks/ppo_results.json`.
* **Changed from 2.0.0** (each 2.0.0 behaviour was fabricated or broken):
  * `HAS_STABLE_BASELINES` and `TrainingStats.ethics_score` are gone (the
    score was never computed; it was always 0.0).
  * `save_model` writes Mercury's own checkpoint (tensors and primitives,
    loaded through `security.safe_torch.safe_torch_load`); SB3 `.zip` files
    do not load. `load_model` and `pretrain` raise on failure instead of
    logging it.
  * `ConvergenceMonitor` stops on a plateau: `patience` episodes whose
    100-episode mean does not beat the best by `1 - convergence_threshold`
    of its magnitude. 2.0.0 declared convergence at the first full window
    (a new best divided by itself is 1), inverted the ratio for negative
    returns, and counted one finished episode per step.
  * `CheckpointCallback.save_freq` counts environment steps across all
    copies; checkpoint files end in `.pt`.
  * `train_online` updates on the episodes it runs (2.0.0 discarded them
    and collected `n_steps` more). `pretrain_on_benchmarks` trains and
    evaluates a separate policy per named benchmark (2.0.0 retrained the
    same environment once per name). `MultiEnvPPOTrainer` normalizes
    observations and rewards only by default, not over an explicit config.
* **Regression tests:** `tests/ml/test_ppo_trainer.py`,
  `tests/ml/test_rl_envs.py`

### 6.7 Geological `SolarFlareDetector` duplicate + `GeomagneticHMM`

* **Removed by:** the v3.0 live-wiring wave (July 2026)
* **Override criterion:** §2 (Fundamental Incompatibility) with the
  no-fabricated-data doctrine. Two unrelated classes named
  `SolarFlareDetector` existed (space vs geological). The geological
  copy filled `kp_index_predicted` / `dst_index_predicted` from
  hand-authored per-HMM-state lookup tables
  (`base_kp=[1,2,4,6,8]`, `base_dst=[0,-10,-30,-100,-300]`) — invented
  geomagnetic indices presented as predictions — and `GeomagneticHMM`
  carried a hand-authored transition matrix plus per-state
  storm-probability priors whose "state" was just the GOES flux class.
* **Replacement:** the canonical
  `omni_mercury_engine.space.solar_storm_detector.SolarFlareDetector`
  (re-exported from `detectors.geological.disaster_detectors` and
  `detectors.geological`, so existing imports keep working). Storm
  fields are populated only from a REAL observed planetary Kp (NOAA
  SWPC live feed or `observed_kp=`); Dst is estimated via the
  documented NOAA G-scale / Loewe & Prölss (1997) mapping. Field
  changes on `SolarFlarePredictionResult`: `hmm_state` /
  `transition_probability` are gone (`flux_class_index` replaces them);
  `flare_class` now carries NOAA letters (`"X"`), not `"x_class"`
  labels (the legacy `SolarFlareClass` enum with `"x_class"` values is
  preserved in `disaster_detectors` for import compatibility);
  `geomagnetic_storm_probability` / `kp_index_predicted` /
  `dst_index_predicted` are `float | None` and `None` offline.
* **Regression test:** `tests/test_live_wiring_space.py`,
  `tests/detectors/test_live_wiring_geological.py`

### 6.8 Private NASA CNEOS loaders in `disaster_detectors`

* **Removed by:** the v3.0 live-wiring wave (July 2026)
* **Override criterion:** §2 (Fundamental Incompatibility) — private
  module-level HTTP loaders (`load_nasa_fireball_data`,
  `load_nasa_close_approach_data`, `load_nasa_sentry_data`) duplicated
  the `data_sources` stack without its rate limiting/caching/circuit
  breaking, and two of them were quietly broken against the live APIs:
  the Sentry loader read nonexistent `ps`/`ts` keys (always emitting
  its `-10`/`0` fallbacks) and the fireball loader mislabelled the
  API's ×10¹⁰ J radiated-energy unit as joules.
* **Replacement:** `data_sources.jpl_ssd.JPLFireballSource` /
  `JPLSentrySource` (new real clients) and the existing
  `data_sources.space_weather.NASANeoWsSource` for close approaches.
  The `FireballEvent` / `CloseApproachEvent` / `SentryImpactRisk`
  dataclasses moved to `data_sources.jpl_ssd` and are re-exported from
  `disaster_detectors` for import compatibility. `MeteorDetector`
  consumes only these clients (6 h refresh preserved via their
  `CacheConfig`); `use_nasa_data=True` still means "construct default
  clients".
* **Regression test:** `tests/test_live_wiring_sources.py`,
  `tests/detectors/test_live_wiring_geological.py`

### 6.9 `USGSVolcanoSource.US_VOLCANOES` static table

* **Removed by:** the v3.0 live-wiring wave (July 2026)
* **Override criterion:** §2 (Fundamental Incompatibility) with the
  no-fabricated-data doctrine — the source looped a hardcoded 10-entry
  volcano dict and emitted `alert_level="normal"` /
  `aviation_color_code="green"` for every volcano on every fetch:
  fabricated all-clear alerts presentable as real monitoring.
* **Replacement:** the same class now queries the real USGS HANS public
  API (`getMonitoredVolcanoes` / `getElevatedVolcanoes`) and reports
  the official observatory alert levels and aviation color codes.
* **Regression test:** `tests/test_live_wiring_sources.py::TestUSGSVolcanoHANS`

### 6.10 `flood_detector.TopographicRunoffPredictor`

* **Removed by:** the v3.0 live-wiring wave (July 2026)
* **Override criterion:** §2 (Fundamental Incompatibility) with the
  no-fabricated-data doctrine. The class was an untrained neural network
  that `FloodDetector` instantiated (`enable_runoff=True`) but **never
  invoked** — dead surface whose only possible future use was emitting
  random-weight runoff/discharge numbers. No caller, no checkpoint, no
  labelled runoff corpus.
* **Replacement:** `FloodDetector` already derives its
  `runoff_coefficient` from the `SoilSaturationModel` physics
  (infiltration-based, deterministic); that path is unchanged. The
  `enable_runoff` constructor parameter was removed with the class (it
  gated only the dead attribute).
* **Regression test:**
  `tests/detectors/test_live_wiring_geological.py::TestFloodDetectorLiveWiring::test_dead_runoff_network_is_gone`

### 6.11 `security.pqc_backends.slhdsa_sign_internal`

* **Removed by:** the AMA Cryptography v4.0.0 -> v5.0.0 pin migration
  (October 2026)
* **Override criterion:** §1 (Security Vulnerability). The wrapper called
  the FIPS 205 §9.2 *internal* signing interface, which signs the raw
  message: under one key its signatures cross-verified with the §10.2 API,
  making it a signing oracle for pure signatures on attacker-chosen
  `(ctx, M)` pairs. FIPS 205 §9 says the internal functions must not be
  exposed to applications other than for testing. AMA v5 compiles them only
  under `AMA_TESTING_MODE`, and its INVARIANT-50 names "legacy" and
  "compatibility" entry points as no exemption, so Mercury keeps no shim
  under the old name either (AMA's own `pqc_backends` asserts the name is
  absent).
* **Replacement:** `slhdsa_sign_addrnd(message, secret_key, addrnd, ctx=b"",
  param_set=...)`, the §10.2 hedged signature with a caller-supplied
  `addrnd`; it applies the `M' = 0x00 || len(ctx) || ctx || M` wrapper
  itself, so callers pass the raw message. Production code uses
  `slhdsa_sign` (randomness from an approved RBG) or
  `slhdsa_sign_deterministic`.
* **Regression test:**
  `tests/security/test_pqc_guards_smoke.py::test_slhdsa_internal_interface_is_not_exposed`

### 6.12 EMP detector: `intentional_attack_probability` and `gic_amplitude_a` values

* **Changed by:** the v3.0 EMP detector rework (October 2026). The
  `EMPPredictionResult` field and the `detect_e3_pulse()` key keep their
  released names and positions; only the value is now `None`, so code that
  compares it with a number (`> 0.7`) or does arithmetic with it raises
  `TypeError` instead of acting on a fabricated number.
* **Override criterion:** §2 (Fundamental Incompatibility) with the
  no-fabricated-data doctrine.
  - `intentional_attack_probability` (released `float = 0.0`): whenever
    `signature_data` was supplied it was the output of
    `IntentionalEMIDetector` with random, never-trained weights (given only a
    1 MHz repetition rate, "critical" in 52 of torch seeds 0-99), and `0.0`
    otherwise, which states "no attack" although no classifier ran. No
    public labelled IEMI corpus exists, so no weights ship; without a trained
    network no probability is truthful, `0.0` included.
  - `detect_e3_pulse()["gic_amplitude_a"]` (released `float`):
    `0.1 × |dB/dt|`, labelled amperes. The coefficient has no source and the
    product no defined unit: a GIC in amperes follows from the geoelectric
    field through a model of the network and its grounding, not from dB/dt
    alone.
* **Replacement:** `intentional_attack_probability` is the trained
  network's probability once `EMPDetector.load_neural_weights` loads an
  `iemi-signature-v1` checkpoint (`train_iemi_network`); until then the
  physics IEMI screen reports through `hpem_environment`, `band_class`,
  `emp_type` and `threat_level`. For E3/GIC, `detect_e3_pulse()` reports
  the measured `peak_dbdt_nt_per_s` and the 1-minute-tier `grid_impact`;
  `space.gic_detector.GICDetector` gives the plane-wave geoelectric field
  (`geoelectric_peak_v_per_km`).
* **Regression tests:**
  `tests/detectors/test_emp_served_contract.py::test_nothing_measured_is_benign`,
  `::test_untrained_network_is_not_consulted` and
  `::test_released_e3_key_is_kept_without_a_fabricated_value`

### 6.13 `security.counterintelligence.OverwatchNexus` / `OverwatchNexusResult`

* **Removed by:** the v3.0 orphan-module vetting pass (October 2026,
  `steel/version-cov3`); the module path was rebuilt on `steel/version-cov4`.
* **Override criterion:** §2 (Fundamental Incompatibility) with the
  no-fabricated-data doctrine. With `enable_ci=False` (the default)
  `proactive_ci` returned a fixed "CI module disabled" result. With it on, the
  class scored intelligence reports it made up itself
  (`_generate_synthetic_intel`: constant OSINT/SIGINT/CYBINT entries with
  `threat_score` 0.5 and indicators `"synthetic_indicator_1/2"`), so the risk
  never depended on the input: zeros, unit noise and x1000 noise all gave
  `risk=0.2766` on the released code. Its "QBM" bio-threat flag meant
  `mean(data) > 2.0`, its ethical rollback could not trigger (the purity
  value is at least 1.055 for any input), and its humanitarian impact was a
  lookup table of casualty counts. Nothing in the released tree called it,
  and the maintainer decided it is not coming back.
* **Replacement:** `security.counterintelligence.CounterintelligenceAssessor`
  (Admiralty-code source posteriors, MinHash near-duplicate provenance, a scan
  statistic for coordinated reporting, digit-based fabrication screens). It
  assesses only the reports it is given and never synthesises one.
  `EngineConfig.enable_ci` stays as an inert field so existing configurations
  still load.
* **Regression tests:**
  `tests/security/test_counterintelligence.py::test_overwatch_nexus_is_not_restored`
  and `::TestAssessor::test_empty_input_yields_no_finding`

### 6.14 `truth_decipher.TruthDecipherFramework` API / `TruthDecipherResult`

* **Removed by:** the v3.0 orphan-module vetting pass (October 2026,
  `steel/version-cov3`); the module was rebuilt on `steel/version-cov4`.
* **Override criterion:** §2 (Fundamental Incompatibility) with the
  no-fabricated-data doctrine and the ethical gate contract. `determine_ethics`
  passed the ethics governor a context with `has_rollback`, `has_benchmarks`,
  `is_transparent`, `is_open_source`, `audit_enabled` and `is_extensible` set
  to `True` and `test_coverage` 0.85 for every action, so the gate judged fixed
  claims about the system instead of the action. `resolve_with_measures` set
  `applied=True` unconditionally, so every resolution was stored for
  case-based reasoning as `CaseOutcome.SUCCESS` with score 1.0 whatever
  happened, which trained later decisions on outcomes nobody observed.
* **Replacement:** `truth_decipher.StatementVerifier` (claim-level
  corroboration with source reliability and independence clusters, temporal
  and spatial contradiction checks, consent-gated Concealed Information Test).
  The name `TruthDecipherFramework` is kept as an alias of `StatementVerifier`;
  the released `decipher_truth` API and `TruthDecipherResult` are not restored.
* **Regression tests:**
  `tests/truth_decipher/test_statement_verification.py::test_alias_and_duplicate_ids`
  and `::test_released_decipher_api_is_not_restored`

### 6.15 `security.quantum_risk_cyber` released classes (`QuantumRiskCyber`, `PostQuantumMigrationPlanner`, `QuantumThreat`, `ThreatLevel`, `CryptoSystem`)

* **Removed by:** the v3.0 orphan-module vetting pass (October 2026,
  `steel/version-cov3`); the module path was rebuilt on `steel/version-cov4`.
* **Override criterion:** §1 (Security Vulnerability): the classes gave wrong
  cryptographic advice. Vulnerability was a substring match against a list
  containing `"DSA"`, so `PostQuantumMigrationPlanner.assess_algorithm_vulnerability`
  reported ML-DSA-65 and SLH-DSA-SHA2-128s, the FIPS 204/205 post-quantum
  signatures AMA Cryptography signs with, as `vulnerable_to_quantum=True`.
  `QuantumRiskCyber.post_quantum_readiness_check` counted SIKE and Rainbow,
  both broken classically in 2022, as post-quantum and returned readiness 1.0
  for them (all three results reproduced on the released code).
* **Replacement:** the rebuilt module's `classify_algorithm` (exact names and
  aliases, broken schemes as `BROKEN`), `assess_asset` / `prioritize` (Mosca
  inequality against a stated CRQC assumption), `read_cbom` (CycloneDX CBOM)
  and `audit_mercury_posture`.
* **Regression tests:**
  `tests/security/test_quantum_risk_cyber.py::TestClassification::test_ml_dsa_is_never_read_as_dsa`,
  `::TestClassification::test_exact_names_and_aliases` and
  `::test_released_classes_are_not_restored`

---

## 7. Renamed and removed in v3.0 (migration from 2.0.0)

v3.0 renames 2.0.0 names that described something the code was not (promotional
"Advanced/Enhanced" prefixes, mythological terms) and removes components whose
output was fabricated. A renamed item keeps its behaviour and values; only the
import changes. There are no aliases: the old names are the inaccurate ones.

### 7.1 Renamed modules (`omni_mercury_engine.` prefix omitted)

| 2.0.0 | v3.0 |
|-------|------|
| `cognitive.anomaly_detection_enhanced` | `cognitive.anomaly_detection` |
| `core.enhanced_base_domains` | `core.base_domains` |
| `core.enhanced_model_domains` | `core.model_domains` |
| `detectors.advanced_physics_integration` | `detectors.physics_integration` |
| `detectors.enhanced_statistical` | `detectors.statistical_extended` |
| `detectors.vlm.advanced_context_providers` | `detectors.vlm.context_providers_extended` |
| `ml.advanced_optimizers` | `ml.optimizers` |
| `models.biometric_advanced` | `models.biometric_multimodal` |
| `models.neurosymbolic_enhanced` | `models.neurosymbolic_temporal` |

### 7.2 Renamed classes, methods, fields and parameters

| 2.0.0 | v3.0 |
|-------|------|
| `EnhancedAnomalyDetector` | `IntegratedAnomalyDetector` |
| `EnhancedBaseDetector` | `BaseDomainDetector` |
| `EnhancedQuantumModel` / `EnhancedBiometricModel` / `EnhancedAffectiveModel` | `QuantumFeatureModel` / `MultimodalBiometricModel` / `AffectiveStateModel` |
| `AdvancedPhysicsConfig` / `AdvancedPhysicsIntegratedDetector` | `PhysicsIntegrationConfig` / `PhysicsIntegratedDetector` |
| `EnhancedStatisticalDetector` | `ExtendedStatisticalDetector` |
| `EnhancedCombinedContextProvider` | `CombinedContextProvider` |
| `harmonics.AdvancedHarmonicAnalyzer` | `harmonics.HarmonicAnalyzer` |
| `AdvancedBiometricEngine` | `MultimodalBiometricEngine` |
| `EnhancedLogicTensorNetwork` / `EnhancedNeurosymbolicEngine` | `TemporalLogicTensorNetwork` / `TemporalNeurosymbolicEngine` |
| `core.stacking_fusion.EthicallyConstrainedFusion` | `ReliabilityWeightedFusion`; `get_ethical_compliance()` -> `get_reliability_report()` (keys `average_reliability`, `reliability_floor`, `floor_met`, `detector_weights`, `detector_reliability_scores`); `benevolence_weight=` -> `reliability_penalty_weight=` |
| `add_detector(..., ethical_score=)` (`StackingFusion`, `BayesianModelAveraging`, `ReliabilityWeightedFusion`) | `reliability_score=` |
| `ml.fusion_network.OmniFusionModel.train_with_advanced_optimizers` | `train_with_optimizers` |
| `ethical.WisdomArchetype` / `ethical_constraint_engine.ArchetypalAnalysis` | `GeometricPattern` / `GeometricPatternAnalysis` (`dominant_archetype` -> `dominant_pattern`, `archetype_scores` -> `pattern_scores`) |
| `compute_divine_proportion` (`ImmutableGeometryProcessor`, `GeometricPatternProcessor`) | `compute_golden_ratio` |
| `EthicalScalars`: `omni_ineffable_transcendence`, `omni_seraphic_elevation`, `omni_scribe_precision`, `omni_hermetic_truth`, `omni_lunar_wisdom`, `omni_hieroglyphic_reasoning`, `omni_cosmic_balance`, `omni_isfet_prevention`, `omni_weighing_judgment`, `omni_pharaonic_accountability`, `omni_seasonal_regulation` | `omni_aspirational_excellence`, `omni_benevolent_uplift`, `omni_record_precision`, `omni_truthfulness_integrity`, `omni_reflective_deliberation`, `omni_symbolic_reasoning`, `omni_systemic_equilibrium`, `omni_disorder_prevention`, `omni_proportional_judgment`, `omni_executive_accountability`, `omni_cyclical_regulation` (values unchanged) |
| `EthicalScalars` protection scalar named for the retired mission framing (spelled only in `tests/test_retired_terminology.py`'s pattern) | `omni_civilization_first_protection` (value unchanged) |
| `SchumannResonanceDetector(enable_ancient_correlation=)` / `SchumannAnomalyResult.ancient_correlation` | `enable_cycle_correlation=` / `cycle_correlation` |
| `ChemistryAnomalyDetector(enable_alchemical_correlation=)` / `ChemicalAnomalyResult.ancient_alchemical_correlation` | `enable_classical_metals_correlation=` / `classical_metals_correlation` |
| `datasets.base.DatasetRegistry.register(name, loader_class=)` | `register(name, loader_factory=)` (positional calls unchanged) |
| `QuantumResistantEncryption.decrypt_hybrid(..., private_key=)` | `secret_key=` |
| `EarthquakePrecursorAnalyzer.forward(em_features=)` | `forward(features=)` |
| `core.neurosymbolic_hub.BENEVOLENCE_THRESHOLD` / `LYAPUNOV_LAMBDA` | `BENEVOLENCE_ADVISORY_LEVEL` (advisory only) / `core.centralized_constants.LYAPUNOV.LAMBDA_CONVERGENCE` |

### 7.3 Removed: the benevolence pass-bar

In 2.0.0 a sample passed the ethics checks when a benevolence score that
Mercury computed from its own detector output reached 0.99: a control any
score could satisfy, which screened no harm and refused benign data. v3.0
enforces the fail-closed harm-uplift gate at every decision surface instead;
benevolence is reported, never gating. Removed with the pass-bar:
`benevolence_threshold=` (`SymbolicLogicLayer`, `NeurosymbolicFusionEngine`,
`ValueExtractor`, `IntegratedAnomalyDetector`, `AdaptiveAnomalyDetector` and the
`AdversarialAE`/`COPOD`/`GWOEnsemble`/`Contrastive`/`MultiScaleTransformer`
configs), `sigma_immutable=` (`AdaptiveAnomalyDetector`,
`MultiScaleTransformerConfig`), `benevolence_weight=` (`BaseDomainDetector`),
`BenevolenceScorer.BENEVOLENCE_THRESHOLD`, `ThresholdConfig.benevolence_required`,
`ThresholdDefaults.BENEVOLENCE_IMMUTABLE`, `EthicalConstants.BENEVOLENCE_IMMUTABLE`,
`core.enhanced_model_domains.BENEVOLENCE_THRESHOLD`, `FusionResult.ethical_gate_passed`
/ `ethical_score`, `OmniAvaEquation(decouple_ethical_scaling=)`. The dual gate
`enforce_dual_ethical_gate(benevolence_scorer=, action=, context=, ...)` is now
`enforce_dual_ethical_gate(subject=, advisory_scorer=, ...)`: the subject is
screened by the harm-uplift gate and the scorer only advises.

### 7.4 Removed: components with fabricated or no output

| 2.0.0 | Why | Use instead |
|-------|-----|-------------|
| `cognitive.cognitive_evolution_engine`: `CognitiveEvolutionEngine`, `ChainOfThoughtReasoner`, `ThoughtChain`, `ReasoningStep`, `CounterfactualSimulator`, `Counterfactual`, `TheoryOfMind`, `IntentInference`, `SelfPlaySimulator`, `SimulationAgent`, `SimulationResult`, `AgentRole`, `RuleMutator`, `Rule`, `MutationResult`, `MutationType` | Canned text and hard-coded confidences; rule fitness was a random walk and mutation reversed strings; `evolve_population([])` never returned; the default safety check failed every cycle | `cognitive.chain_of_thought.ChainOfThoughtEngine`; `cognitive.causal_discovery.CausalDiscoveryEngine.counterfactual_query`; `ml.rule_evolution.EvolvedRuleSearch`; the module's `CuriosityEngine.explore` (its `ExplorationResult` no longer carries `discoveries`, `questions_generated`, `follow_up_actions`) |
| `HurricanePredictionResult.track_forecast` / `landfall_probability` / `time_to_landfall_hours` | Declared, never computed | none: no track model exists |
| `SolarFlarePredictionResult.hmm_state` / `transition_probability` | Came from `GeomagneticHMM` (§6.7) | `space.solar_storm_detector.SolarFlareDetector` |
| `truth_decipher.StatementVerifier(autonomy_level=, enable_cognitive=, enable_novel_discovery=, enable_self_healing=)` | The 2.0.0 framework's switches (§6.14) | `StatementVerifier` defaults |
| `datasets.adrepository.ODDSDatasetInfo`, `ADRepositoryLoader.ODDS_URLS` / `MIRROR_URLS` | The ODDS host is dead and no dataset routed through it | the DevNet tabular mirror and the dedicated loaders |
| `security.crypto_api.InvalidSignature` | PyCA left `src` with the move of Ed25519 to AMA | `Ed25519Provider.verify` returns `False` on a bad signature, as before |
| `security.encryption.SecureDataHandler.qr_encryption` / `public_key` / `private_key` | The 2.0.0 lattice keys were broken (zero-noise LWE); replaced by AMA ML-KEM-1024 + AES-256-GCM | `SecureDataHandler(enable_quantum_resistant=, at_rest_key=)` |
| `comparison.pyod_integration.PyODComparison.benchmark_against_pyod(omni_engine, test_data, ground_truth, pyod_algorithms)` | Redesigned around precomputed scores | `benchmark_against_pyod(mercury_scores, X_train, X_test, y_test, algorithms=, seed=)` |
| `cognitive.uncertainty.UncertaintyQuantifier.PHI` | A golden-ratio constant in the confidence formula with no basis; the prior is now parameter-free | none |

## Support Policy

| Version | Commitment |
|---------|------------|
| 1.x | All items fully functional with optional warnings, except the items enumerated in §6.1-§6.4 |
| 2.x | All items fully functional with optional warnings |
| 3.x | All items fully functional with optional warnings, except the items removed, changed or renamed in v3.0 (§6.6-§6.15, §7) |
| Future | Removal only with documented justification per policy |

---

## Contributing

If you believe an item should be considered for removal, submit an RFC with:

1. Security analysis
2. Usage impact assessment
3. Migration path documentation
4. Proposed timeline (minimum 12 months)

Submit RFCs at: https://github.com/Steel-SecAdv-LLC/Mercury-Agent/issues

---

## Need Help?

If you encounter issues:

1. Check this guide for the correct usage pattern
2. Search the codebase for usage examples
3. Open an issue at https://github.com/Steel-SecAdv-LLC/Mercury-Agent/issues

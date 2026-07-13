# Mercury Agent — Subagent Pantheon

Applies to Mercury Agent **v2.1.x**. Last updated: 2026-07-11.

The Mercury Agent hosts an **internal-only fleet of 33 named subagents** that the
root agent delegates work to — singly, across a batch, or to many replicas at
once ("in the masses"). This document is the canonical roster and the design
contract. The implementation lives in
[`src/omni_mercury_engine/agentic/subagents/`](../src/omni_mercury_engine/agentic/subagents/).

> **Scope:** this convention applies to **Mercury Agent only**. The sibling
> FINDΩYOU™ platform may adopt the same or a similar pattern later.
>
> **Terminology:** **Omni-Codes only.** No other code system is referenced
> anywhere in this subsystem; the term is enforced by
> `tests/test_subagent_roster.py::test_no_memorial_terminology_anywhere_in_subsystem`.

## Naming convention

Greek-pantheon identity + Roman numeral: `Themis_I` … `Rhea_XXXIII`. Every member
is a Greek deity or personification chosen so the myth matches the function —
e.g. **Hecate** (crossroads/thresholds) for the gateway, **Harmonia** (concord)
for normalization, and **Eleos** (mercy) for survivor support.

## Architecture

- **Root agent.** The [`MercuryAgent`](../src/omni_mercury_engine/agentic/mercury_a_agent.py)
  is the single top-level orchestrator that supervises the fleet. Per
  `utils/constants.py` ("Seven Omni-Codes governing Mercury Agent"), the root is
  governed by **all seven** Omni-Codes, not anchored to one. It delegates via
  `MercuryAgent.delegate()` / `delegate_masses()` (lazily enabling the fleet), or
  the engine enables it with `OmniMercuryEngine.enable_subagent_fleet()`.
- **Capability parity.** Each `SubAgent` *subclasses* `MercuryAgent`, so every
  member carries the full main-agent toolkit (hierarchical planning, ReAct
  reasoning, four-tier memory, tool execution, the benevolence gate). A subagent
  is never a thin wrapper.
- **Omni-Code anchor.** Each member is anchored to exactly one of the Seven
  Omni-Codes. The anchor's helical *stability* (`|r|·p`) genuinely sets the
  member's autonomy ceiling via `compute_ethical_autonomy` (capped at 0.95) — a
  real, monotonic binding, not a tag.
- **Depth tiers (both real, no theater).**
  - `deep` — a bespoke specialization with genuine domain logic.
  - `coordinator` — a `CoordinatorSubAgent` that is a genuine subsystem
    **operator**. For each member an adapter in
    [`operations.py`](../src/omni_mercury_engine/agentic/subagents/operations.py)
    invokes the member's **real** `omni_mercury_engine` entrypoint with inputs
    derived from `task.payload` (the "operation + inputs" column below) and
    returns the transparent result of that call (`output["mode"] == "operation"`). It
    **fails closed** (`SubAgentExecutionError`) on malformed inputs and never
    fabricates signal. When the entrypoint is input-gated and the payload lacks
    its inputs — or the caller asks for a readiness probe via
    `payload["mode"] == "introspect"` — it falls back to the transparent live
    **binding report** (`output["mode"] == "binding"`): it imports each declared
    subsystem, introspects its live public API, reports genuine
    availability/capability, and fails closed when *no* subsystem binds. The
    binding report is the transparent no-input floor, never the whole behavior.
- **Access boundary (internal-only).** Nothing here is exported from the public
  `omni_mercury_engine` surface. `SubAgent` / `SubAgentRegistry` / `SubAgentFleet`
  require the package-private access sentinel; direct construction raises
  `SubAgentAccessError`. Users never address a subagent — the root agent calls on
  them.
- **Autonomy governor.** `AutonomyGovernor` enforces, fail-closed: a capability
  ceiling (replicas / total-active / recursion depth), the Omni-Code autonomy
  cap, corrigibility (`pause`/`resume` + an irreversible `trip` kill-switch), and
  a failure-rate tripwire (ethical refusals count as correct, never as failures).
- **Dual-gate commit.** The fleet commits results through the same hard dual
  ethical gate — benevolence floor **and** σ-Immutable — used on the engine and
  orchestrator boundaries; fail-closed.
- **Detection bridge.** The detection-natured members (`Zeus_VIII`,
  `Dionysus_XIII`) run Mercury's *own* real multi-agent detection by bridging to
  `agentic/orchestration.py`'s `MultiAgentOrchestrator` (which coordinates real
  `DetectionAgent`s under `cognitive/multi_agent_coordination.py`). Members are
  **not** forced to subclass `DetectionAgent`: most of the 33 roles are not
  anomaly detectors, and imposing an anomaly-score contract on them would be
  theater. The orchestration linkage is used where it is genuine.

## The Seven Omni-Codes (anchors)

Verbatim from [`src/omni_mercury_engine/utils/constants.py`](../src/omni_mercury_engine/utils/constants.py).

| Omni-Code | Symbol | r | p | stability | Domain |
|---|---|---|---|---|---|
| `OMNI_DIRECTIONAL` | 👁∞ | 20.0 | 0.7 | 14.0 | 360° awareness & multi-domain perception |
| `OMNI_PERCIPIENT` | Ϙϵ | 16.0 | 1.1 | 17.6 | Predictive foresight & anticipatory analysis |
| `OMNI_INDIVISIBLE` | Φϖ | 7.0 | 0.9 | 6.3 | Unified protection & integrity preservation |
| `OMNI_BENEVOLENT` | Σϵ | 19.0 | 1.2 | 22.8 | Ethical foundation & humanitarian alignment |
| `OMNI_SCIENT` | Ωϖ | 20.0 | 1.1 | 22.0 | Knowledge acquisition & scientific discovery |
| `OMNI_UNIVERSAL` | Θϵ | 25.0 | 0.1 | 2.5 | Structured governance & systematic order |
| `OMNI_POTENT` | Γϖ | 19.0 | 1.1 | 20.9 | Regenerative capability & adaptive resilience |

**Code-bearers (★)** — the single lead member per Code (each bearer's anchor
equals the Code it bears): `Themis_I` (BENEVOLENT), `Zeus_VIII` (INDIVISIBLE),
`Helios_XVII` (DIRECTIONAL), `Tyche_XX` (PERCIPIENT), `Prometheus_XXVII`
(SCIENT), `Atlas_XXX` (UNIVERSAL), `Rhea_XXXIII` (POTENT).

## The 33-member roster

`Autonomy` is the Omni-Code-anchored ceiling (computed, capped 0.95). Generated
from the roster; the structural invariants are gated by
`tests/test_subagent_roster.py`.

| # | Member | Role | Subsystems | Anchor (Omni-Code) | Autonomy | Depth | Bearer |
|---|--------|------|------------|--------------------|----------|-------|--------|
| 1 | **Themis_I** | Ethics & benevolence gate | `ethical`, `safeguards` | OMNI_BENEVOLENT (Σϵ) | 0.95 | deep | ★ |
| 2 | **Hestia_II** | Core foundation / OAE stability | `core`, `utils` | OMNI_INDIVISIBLE (Φϖ) | 0.847 | coordinator |  |
| 3 | **Hermes_III** | Communication / user synchronization | `api`, `gui`, `narrative` | OMNI_DIRECTIONAL (👁∞) | 0.95 | coordinator |  |
| 4 | **Athena_IV** | Upload validation / intake vision | `validation`, `verifiers`, `biometric` | OMNI_SCIENT (Ωϖ) | 0.95 | coordinator |  |
| 5 | **Apollo_V** | Biometric matching | `biometric` | OMNI_SCIENT (Ωϖ) | 0.95 | coordinator |  |
| 6 | **Artemis_VI** | OSINT / registry cross-check | `data_sources`, `integrations` | OMNI_SCIENT (Ωϖ) | 0.95 | coordinator |  |
| 7 | **Hera_VII** | Regulatory compliance (BIPA/CCPA/CPRA) | `compliance` | OMNI_UNIVERSAL (Θϵ) | 0.788 | deep |  |
| 8 | **Zeus_VIII** | Sigma-immutable directive / anomaly authority | `security`, `engine`, `agentic` | OMNI_INDIVISIBLE (Φϖ) | 0.847 | deep | ★ |
| 9 | **Poseidon_IX** | Data flow / secure vault | `data`, `data_sources`, `crypto` | OMNI_DIRECTIONAL (👁∞) | 0.95 | coordinator |  |
| 10 | **Demeter_X** | Cognitive evolution / growth | `cognitive`, `emergent` | OMNI_POTENT (Γϖ) | 0.95 | coordinator |  |
| 11 | **Hephaestus_XI** | Infrastructure / auto-scaling | `scaling`, `infrastructure` | OMNI_POTENT (Γϖ) | 0.95 | coordinator |  |
| 12 | **Eleos_XII** | Empathy / survivor support | `narrative`, `medical` | OMNI_BENEVOLENT (Σϵ) | 0.95 | coordinator |  |
| 13 | **Dionysus_XIII** | Pattern / emergent recognition | `anomaly`, `detectors`, `harmonics`, `emergent` | OMNI_DIRECTIONAL (👁∞) | 0.95 | deep |  |
| 14 | **Ares_XIV** | Security / defense / instant bans | `security`, `safeguards` | OMNI_INDIVISIBLE (Φϖ) | 0.847 | deep |  |
| 15 | **Hades_XV** | Compression / deep (cold) storage | `data`, `crypto`, `utils` | OMNI_UNIVERSAL (Θϵ) | 0.788 | coordinator |  |
| 16 | **Selene_XVI** | Night/cron scheduling / temporal coordination | `streaming`, `scaling` | OMNI_UNIVERSAL (Θϵ) | 0.788 | coordinator |  |
| 17 | **Helios_XVII** | Telemetry / real-time monitoring | `metrics`, `alerting`, `streaming` | OMNI_DIRECTIONAL (👁∞) | 0.95 | coordinator | ★ |
| 18 | **Eos_XVIII** | Onboarding / session initiation | `api` | OMNI_POTENT (Γϖ) | 0.95 | coordinator |  |
| 19 | **Nemesis_XIX** | Fairness / bias audit / dispute resolution | `ethical`, `evaluation` | OMNI_BENEVOLENT (Σϵ) | 0.95 | coordinator |  |
| 20 | **Tyche_XX** | Risk / probabilistic decisioning | `decision`, `agentic`, `ml` | OMNI_PERCIPIENT (Ϙϵ) | 0.95 | coordinator | ★ |
| 21 | **Zelos_XXI** | Performance / throughput optimization | `scaling`, `metrics`, `ml` | OMNI_POTENT (Γϖ) | 0.95 | coordinator |  |
| 22 | **Kronos_XXII** | Time-series analysis / historical indexing | `detectors`, `streaming` | OMNI_PERCIPIENT (Ϙϵ) | 0.95 | coordinator |  |
| 23 | **Morpheus_XXIII** | Simulation / scenario generation | `cognitive`, `emergent` | OMNI_PERCIPIENT (Ϙϵ) | 0.95 | coordinator |  |
| 24 | **Iris_XXIV** | Notification / multi-channel routing | `alerting` | OMNI_DIRECTIONAL (👁∞) | 0.95 | coordinator |  |
| 25 | **Pan_XXV** | Sensor fusion / peripheral integration | `data_sources`, `core`, `integrations` | OMNI_DIRECTIONAL (👁∞) | 0.95 | coordinator |  |
| 26 | **Persephone_XXVI** | Lifecycle management / archival retrieval | `data`, `federation`, `resilience` | OMNI_POTENT (Γϖ) | 0.95 | coordinator |  |
| 27 | **Prometheus_XXVII** | Model training / resource provisioning | `ml`, `automl` | OMNI_SCIENT (Ωϖ) | 0.95 | coordinator | ★ |
| 28 | **Hecate_XXVIII** | Gateway / protocol translation | `api`, `integrations` | OMNI_UNIVERSAL (Θϵ) | 0.788 | coordinator |  |
| 29 | **Nyx_XXIX** | Secrets / secure enclave control | `crypto`, `security` | OMNI_INDIVISIBLE (Φϖ) | 0.847 | coordinator |  |
| 30 | **Atlas_XXX** | Distributed orchestration / cluster management | `distributed`, `agentic` | OMNI_UNIVERSAL (Θϵ) | 0.788 | coordinator | ★ |
| 31 | **Harmonia_XXXI** | Data normalization / canonicalization | `utils`, `data`, `models` | OMNI_UNIVERSAL (Θϵ) | 0.788 | coordinator |  |
| 32 | **Hyperion_XXXII** | High-performance compute / GPU scheduling | `scaling`, `distributed`, `ml` | OMNI_POTENT (Γϖ) | 0.95 | coordinator |  |
| 33 | **Rhea_XXXIII** | Dependency management / resilience control | `resilience` | OMNI_POTENT (Γϖ) | 0.95 | coordinator | ★ |

Plus one internal, non-public member — `_generalist` (the routing floor): full
main-agent capability, used only when capability routing finds no specialist, so
the fleet is never silent.

## Coordinator operations (real entrypoint + payload contract)

Every one of the 28 `coordinator` members is a genuine subsystem **operator**:
its adapter in
[`operations.py`](../src/omni_mercury_engine/agentic/subagents/operations.py)
invokes the real `omni_mercury_engine` entrypoint below with the listed
`task.payload` inputs and returns the transparent result (`mode="operation"`). An
*input-gated* member with no inputs — or any member asked for a readiness probe
(`payload["mode"]="introspect"`) — falls back to the live binding report
(`mode="binding"`). Each member has a real-op test (valid inputs, asserts the
real path) and a fallback test in
[`tests/test_subagent_operations.py`](../tests/test_subagent_operations.py).

| Member | Real entrypoint (operation) | Payload inputs |
|--------|-----------------------------|----------------|
| **Hestia_II** | `utils.normalize_data` / `utils.constants.OmniCodes` stability | `data?`, `method?` (none ⇒ stability) |
| **Hermes_III** | `narrative.create_mercury_interface().process_detection` | `detection_result` |
| **Athena_IV** | `validation.ValidationPipeline.validate(model, X, y)` | `model`, `X`, `y` |
| **Apollo_V** | `biometric.BiometricAnomalyDetector.detect_anomaly` | `iris_image` / `fingerprint_image` / `voice_sample` |
| **Artemis_VI** | `data_sources.fetch_all` (genuine reachability) | none; `sources?`, `timeout?` 🌐 |
| **Poseidon_IX** | `crypto.encrypt` + `crypto.hash_data` (round-trip verified) | `data`, `key?` |
| **Demeter_X** | `cognitive.CognitiveOrchestrator.analyze` | `detection_result`, `raw_data?` |
| **Hephaestus_XI** | `scaling.BainAIScaling.optimize_compute_allocation` / `infrastructure.instantiate_filtered_modules` | `workloads`,`resources` (none ⇒ provision) |
| **Eleos_XII** | `narrative.MercuryConversationInterface.process_detection` (empathetic) | `detection_result` |
| **Hades_XV** | `utils.compress_information` + `crypto.hash_data` | `data`, `compression_level?` |
| **Selene_XVI** | `streaming.StreamingDetector.ingest` (stateful) | `points` |
| **Helios_XVII** ★ | `metrics.AnomalyMetrics.compute_all(labels, scores)` | `labels`, `scores` |
| **Eos_XVIII** | `api.auth.JWTAuth.create_token` + native-JWT validate (in-proc) | `user_id`, `username`, `roles?`, `secret_key?` |
| **Nemesis_XIX** | `evaluation.evaluate_anomaly_detection` / `ethical.TwelveFoldVerificationSystem.verify` | `y_true`,`y_score` / `dimension_scores` |
| **Tyche_XX** ★ | `decision.DecisionAbstentionResponder.decide` | `detection_result`, `domain?` |
| **Zelos_XXI** | `ml.quick_anomaly_score` | `data`, `method?` |
| **Kronos_XXII** | `detectors.MercuryAnomalyDetector.fit().detect()` | `data`, `train?` |
| **Morpheus_XXIII** | `cognitive.CognitiveOrchestrator.analyze` (scenario context) | `detection_result`, `scenario?` |
| **Iris_XXIV** | `alerting.CAPAlertGenerator.generate_alert` | `headline`, `description`, `score?`, `area?`, `domain?` |
| **Pan_XXV** | `core.GlobalOmniScalarNetwork` register + `compute_global_intelligence_score` | `scalars` |
| **Persephone_XXVI** | `resilience.get_all_breaker_stats` | none |
| **Prometheus_XXVII** ★ | `ml.quick_anomaly_score` (light) / `automl.MercuryAutoML.fit` (heavy, budget-gated) | `X` / `X_train`,`y_train`,`automl` |
| **Hecate_XXVIII** | `integrations.routing.RequestRouter.match` (in-proc routing) | `request`, `routes?` |
| **Nyx_XXIX** | `crypto.encrypt`+`hash_data` (primary) / `security.MercuryCrypto.generate_signing_keypair` | `data` / `keygen` |
| **Atlas_XXX** ★ | `distributed.DistributedMercuryCluster.detect_anomalies` (asyncio, bounded) | `data`, `nodes?`, `timeout?` |
| **Harmonia_XXXI** | `utils.normalize_data` | `data`, `method?` |
| **Hyperion_XXXII** | `scaling.BainAIScaling.estimate_power_consumption`/`estimate_agentic_ai_impact` / `ml.quick_anomaly_score` | `model_size`,… / `data` |
| **Rhea_XXXIII** ★ | `resilience.get_all_breaker_stats` + `SelfHealingEngine.get_system_health` | none |

The five `deep` members invoke their bespoke specialization directly: `Themis_I`
(`system`), `Hera_VII` (`data_category`, `data_subject_id`, `context`),
`Zeus_VIII` / `Dionysus_XIII` (`data`, `train?`), and `Ares_XIV` (`action`,
`user_input`, `context?`).

## Provenance

The `deep` members consolidate the genuine capabilities transferred from
FINDΩYOU™'s former agent layer as that platform is made agent-free: `Themis_I`
(ethics enforcement), `Hera_VII` (compliance), and `Ares_XIV`
(guardrail/manipulation-resistance). `Zeus_VIII` and `Dionysus_XIII` wrap
Mercury's own detection. Mercury Agent is the AI centerpiece that hosts the
combined capabilities.

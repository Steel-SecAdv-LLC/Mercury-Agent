# Mercury Agent — Subagent Pantheon

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

Greek-pantheon identity + Roman numeral: `Themis_I` … `Rhea_XXXIII`. The sole
intentional exception is **`Selinus_XXXI`** (an ancient Sicilian city / the river
Selinos), kept for the river→flow→normalize fit and flagged in the roster's
`name_flag`.

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
  - `coordinator` — a `CoordinatorSubAgent` that *binds to and exercises* its
    real `omni_mercury_engine` subsystem(s): it imports them, introspects their
    live public API, reports genuine availability/capability, and **fails closed**
    when a subsystem is absent. The signal reflects the actual state of the repo.
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
| 12 | **Aphrodite_XII** | Empathy / survivor support | `narrative`, `medical` | OMNI_BENEVOLENT (Σϵ) | 0.95 | coordinator |  |
| 13 | **Dionysus_XIII** | Pattern / emergent recognition | `anomaly`, `detectors`, `harmonics`, `emergent` | OMNI_DIRECTIONAL (👁∞) | 0.95 | deep |  |
| 14 | **Ares_XIV** | Security / defense / instant bans | `security`, `safeguards` | OMNI_INDIVISIBLE (Φϖ) | 0.847 | deep |  |
| 15 | **Hades_XV** | Compression / deep (cold) storage | `data`, `crypto` | OMNI_UNIVERSAL (Θϵ) | 0.788 | coordinator |  |
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
| 28 | **Janus_XXVIII** | Gateway / protocol translation | `api`, `integrations` | OMNI_UNIVERSAL (Θϵ) | 0.788 | coordinator |  |
| 29 | **Nyx_XXIX** | Secrets / secure enclave control | `crypto`, `security` | OMNI_INDIVISIBLE (Φϖ) | 0.847 | coordinator |  |
| 30 | **Atlas_XXX** | Distributed orchestration / cluster management | `distributed`, `agentic` | OMNI_UNIVERSAL (Θϵ) | 0.788 | coordinator | ★ |
| 31 | **Selinus_XXXI** | Data normalization / canonicalization | `data`, `models` | OMNI_UNIVERSAL (Θϵ) | 0.788 | coordinator |  |
| 32 | **Hyperion_XXXII** | High-performance compute / GPU scheduling | `scaling`, `distributed`, `ml` | OMNI_POTENT (Γϖ) | 0.95 | coordinator |  |
| 33 | **Rhea_XXXIII** | Dependency management / resilience control | `resilience` | OMNI_POTENT (Γϖ) | 0.95 | coordinator | ★ |

Plus one internal, non-public member — `_generalist` (the routing floor): full
main-agent capability, used only when capability routing finds no specialist, so
the fleet is never silent.

## Provenance

The `deep` members consolidate the genuine capabilities transferred from
FINDΩYOU™'s former agent layer as that platform is made agent-free: `Themis_I`
(ethics enforcement), `Hera_VII` (compliance), and `Ares_XIV`
(guardrail/manipulation-resistance). `Zeus_VIII` and `Dionysus_XIII` wrap
Mercury's own detection. Mercury Agent is the AI centerpiece that hosts the
combined capabilities.

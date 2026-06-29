# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""The Mercury subagent pantheon — the canonical 33-member roster.

This module is pure, declarative data: the org chart of the internal subagent
fleet. Each :class:`RosterEntry` binds a named pantheon member to its
human-readable role, the **real** ``omni_mercury_engine`` subsystem(s) it
coordinates, and exactly one of the Seven **Omni-Codes**
(:class:`~omni_mercury_engine.utils.constants.OmniCodes`) as its autonomy anchor.

Naming convention (Mercury-only; FINDΩYOU™ may adopt a similar pattern later):
Greek-pantheon identity + Roman numeral (``Themis_I`` … ``Rhea_XXXIII``). Every
member is a Greek deity or personification chosen so the myth matches the
function — e.g. ``Hecate`` (crossroads/thresholds) for the gateway, ``Harmonia``
(concord) for normalization, and ``Eleos`` (mercy) for survivor support.

Two depth tiers, both real (no theater):

* ``deep`` — a bespoke specialization with genuine domain logic (``impl_path``
  points at the class).
* ``coordinator`` — a
  :class:`~omni_mercury_engine.agentic.subagents.coordinator.CoordinatorSubAgent`
  that binds to and exercises its real subsystem(s), reporting genuine
  availability/capability and failing closed when a subsystem is absent.

Seven of the members are **code-bearers** (``code_bearer=True``): the single lead
member per Omni-Code (each bearer's ``anchor`` equals the Code it bears). The
root agent — the :class:`~omni_mercury_engine.agentic.mercury_a_agent.MercuryAgent`
itself — is governed by *all seven* Codes (it supervises the fleet), so it is not
listed here.

The terminology here is **Omni-Codes only** — no other code system is referenced
anywhere in this subsystem.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from omni_mercury_engine.agentic.mercury_a_agent import DomainType

# Dotted-path prefix for the deep specialization classes.
_SPEC = "omni_mercury_engine.agentic.subagents.specializations"


@dataclass(frozen=True)
class RosterEntry:
    """One pantheon member: identity → real subsystems → one Omni-Code anchor.

    Attributes:
        id: Pantheon identity (e.g. ``"Themis_I"``); the routing/telemetry key.
        role: Human-readable role description.
        subsystems: Real ``omni_mercury_engine`` subpackage/module names this
            member coordinates (each must resolve — gated by the roster tests).
        anchor: Omni-Code attribute name (one of the Seven), e.g.
            ``"OMNI_BENEVOLENT"``; its stability sets the member's autonomy.
        keywords: Lowercase routing keywords.
        depth: ``"deep"`` (bespoke logic via ``impl_path``) or ``"coordinator"``.
        alias: Optional human-friendly alias.
        domain: Primary :class:`DomainType` for the member.
        code_bearer: Whether this member is the single lead for its anchor Code.
        impl_path: Dotted path to the deep specialization class, or ``None`` to
            use the generic :class:`CoordinatorSubAgent`.
        name_flag: Optional note for an intentional naming exception.
        internal: Internal infrastructure member (e.g. the routing floor) — not
            part of the public 33-member roster.
    """

    id: str
    role: str
    subsystems: tuple[str, ...]
    anchor: str
    keywords: tuple[str, ...] = field(default_factory=tuple)
    depth: str = "coordinator"
    alias: str | None = None
    domain: DomainType = DomainType.GENERAL
    code_bearer: bool = False
    impl_path: str | None = None
    name_flag: str | None = None
    internal: bool = False


def subsystem_module(subsystem: str) -> str:
    """Return the fully-qualified module path for a roster subsystem name."""
    return f"omni_mercury_engine.{subsystem}"


# =============================================================================
# The 33-member pantheon roster.
# =============================================================================

ROSTER: tuple[RosterEntry, ...] = (
    RosterEntry(
        id="Themis_I",
        alias="Governance Sentinel",
        role="Ethics & benevolence gate",
        subsystems=("ethical", "safeguards"),
        anchor="OMNI_BENEVOLENT",
        keywords=("ethics", "ethical", "benevolence", "bias", "fairness", "ieee", "governance",
                  "oversight", "principle", "eu ai act"),
        depth="deep",
        code_bearer=True,
        impl_path=f"{_SPEC}.ethics.EthicsEnforcementSubAgent",
    ),
    RosterEntry(
        id="Hestia_II",
        alias="Stability Anchor",
        role="Core foundation / OAE stability",
        subsystems=("core", "utils"),
        anchor="OMNI_INDIVISIBLE",
        keywords=("foundation", "stability", "core", "integrity", "constants", "baseline"),
        domain=DomainType.INFRASTRUCTURE,
    ),
    RosterEntry(
        id="Hermes_III",
        alias="Communications Relay",
        role="Communication / user synchronization",
        subsystems=("api", "gui", "narrative"),
        anchor="OMNI_DIRECTIONAL",
        keywords=("communication", "relay", "user", "interface", "narrative", "message",
                  "synchronization"),
    ),
    RosterEntry(
        id="Athena_IV",
        role="Upload validation / intake vision",
        subsystems=("validation", "verifiers", "biometric"),
        anchor="OMNI_SCIENT",
        keywords=("upload", "validation", "intake", "vision", "verify", "ingest", "screening"),
        domain=DomainType.SCIENTIFIC,
    ),
    RosterEntry(
        id="Apollo_V",
        role="Biometric matching",
        subsystems=("biometric",),
        anchor="OMNI_SCIENT",
        keywords=("biometric", "matching", "match", "face", "identity", "recognition"),
        domain=DomainType.SCIENTIFIC,
    ),
    RosterEntry(
        id="Artemis_VI",
        role="OSINT / registry cross-check",
        subsystems=("data_sources", "integrations"),
        anchor="OMNI_SCIENT",
        keywords=("osint", "registry", "cross-check", "lookup", "source", "external", "enrichment"),
        domain=DomainType.SECURITY,
    ),
    RosterEntry(
        id="Hera_VII",
        role="Regulatory compliance (BIPA/CCPA/CPRA)",
        subsystems=("compliance",),
        anchor="OMNI_UNIVERSAL",
        keywords=("compliance", "consent", "bipa", "ccpa", "cpra", "privacy", "retention",
                  "regulatory", "lawful", "gdpr"),
        depth="deep",
        impl_path=f"{_SPEC}.compliance.ComplianceSubAgent",
    ),
    RosterEntry(
        id="Zeus_VIII",
        role="Sigma-immutable directive / anomaly authority",
        subsystems=("security", "engine", "agentic"),
        anchor="OMNI_INDIVISIBLE",
        keywords=("anomaly", "detection", "detect", "sigma", "directive", "authority", "threat",
                  "outlier", "intrusion"),
        depth="deep",
        domain=DomainType.SECURITY,
        code_bearer=True,
        impl_path=f"{_SPEC}.detection.DetectionSubAgent",
    ),
    RosterEntry(
        id="Poseidon_IX",
        role="Data flow / secure vault",
        subsystems=("data", "data_sources", "crypto"),
        anchor="OMNI_DIRECTIONAL",
        keywords=("data", "flow", "vault", "pipeline", "ingest", "stream", "storage"),
        domain=DomainType.INFRASTRUCTURE,
    ),
    RosterEntry(
        id="Demeter_X",
        role="Cognitive evolution / growth",
        subsystems=("cognitive", "emergent"),
        anchor="OMNI_POTENT",
        keywords=("evolution", "growth", "learning", "cognitive", "emergent", "adaptation"),
        domain=DomainType.SCIENTIFIC,
    ),
    RosterEntry(
        id="Hephaestus_XI",
        role="Infrastructure / auto-scaling",
        subsystems=("scaling", "infrastructure"),
        anchor="OMNI_POTENT",
        keywords=("infrastructure", "scaling", "autoscale", "build", "provision", "capacity"),
        domain=DomainType.INFRASTRUCTURE,
    ),
    RosterEntry(
        id="Eleos_XII",
        role="Empathy / survivor support",
        subsystems=("narrative", "medical"),
        anchor="OMNI_BENEVOLENT",
        keywords=("empathy", "support", "survivor", "care", "wellbeing", "compassion", "human"),
        domain=DomainType.HUMANITARIAN,
    ),
    RosterEntry(
        id="Dionysus_XIII",
        role="Pattern / emergent recognition",
        subsystems=("anomaly", "detectors", "harmonics", "emergent"),
        anchor="OMNI_DIRECTIONAL",
        keywords=("pattern", "patterns", "emergent", "recognition", "anomaly", "detect",
                  "harmonic"),
        depth="deep",
        domain=DomainType.SCIENTIFIC,
        impl_path=f"{_SPEC}.detection.DetectionSubAgent",
    ),
    RosterEntry(
        id="Ares_XIV",
        role="Security / defense / instant bans",
        subsystems=("security", "safeguards"),
        anchor="OMNI_INDIVISIBLE",
        keywords=("security", "defense", "guardrail", "ban", "threat", "abuse", "manipulation",
                  "prohibited", "intrusion", "jailbreak"),
        depth="deep",
        domain=DomainType.SECURITY,
        impl_path=f"{_SPEC}.guardrail.GuardrailSubAgent",
    ),
    RosterEntry(
        id="Hades_XV",
        role="Compression / deep (cold) storage",
        subsystems=("data", "crypto"),
        anchor="OMNI_UNIVERSAL",
        keywords=("compression", "storage", "archive", "cold", "retention", "compress"),
        domain=DomainType.INFRASTRUCTURE,
    ),
    RosterEntry(
        id="Selene_XVI",
        role="Night/cron scheduling / temporal coordination",
        subsystems=("streaming", "scaling"),
        anchor="OMNI_UNIVERSAL",
        keywords=("schedule", "cron", "temporal", "nightly", "batch", "timing"),
        domain=DomainType.INFRASTRUCTURE,
    ),
    RosterEntry(
        id="Helios_XVII",
        role="Telemetry / real-time monitoring",
        subsystems=("metrics", "alerting", "streaming"),
        anchor="OMNI_DIRECTIONAL",
        keywords=("telemetry", "monitoring", "metrics", "observability", "realtime", "dashboard"),
        domain=DomainType.INFRASTRUCTURE,
        code_bearer=True,
    ),
    RosterEntry(
        id="Eos_XVIII",
        role="Onboarding / session initiation",
        subsystems=("api",),
        anchor="OMNI_POTENT",
        keywords=("onboarding", "session", "initiation", "bootstrap", "init", "welcome"),
    ),
    RosterEntry(
        id="Nemesis_XIX",
        role="Fairness / bias audit / dispute resolution",
        subsystems=("ethical", "evaluation"),
        anchor="OMNI_BENEVOLENT",
        keywords=("fairness", "bias", "audit", "dispute", "appeal", "equity", "evaluation"),
    ),
    RosterEntry(
        id="Tyche_XX",
        role="Risk / probabilistic decisioning",
        subsystems=("decision", "agentic", "ml"),
        anchor="OMNI_PERCIPIENT",
        keywords=("risk", "probabilistic", "decision", "uncertainty", "forecast", "bayesian"),
        domain=DomainType.FINANCIAL,
        code_bearer=True,
    ),
    RosterEntry(
        id="Zelos_XXI",
        role="Performance / throughput optimization",
        subsystems=("scaling", "metrics", "ml"),
        anchor="OMNI_POTENT",
        keywords=("performance", "throughput", "optimization", "latency", "tuning", "speed"),
        domain=DomainType.INFRASTRUCTURE,
    ),
    RosterEntry(
        id="Kronos_XXII",
        role="Time-series analysis / historical indexing",
        subsystems=("detectors", "streaming"),
        anchor="OMNI_PERCIPIENT",
        keywords=("time-series", "temporal", "historical", "index", "trend", "forecasting"),
        domain=DomainType.SCIENTIFIC,
    ),
    RosterEntry(
        id="Morpheus_XXIII",
        role="Simulation / scenario generation",
        subsystems=("cognitive", "emergent"),
        anchor="OMNI_PERCIPIENT",
        keywords=("simulation", "scenario", "what-if", "synthetic", "model", "generate"),
        domain=DomainType.SCIENTIFIC,
    ),
    RosterEntry(
        id="Iris_XXIV",
        role="Notification / multi-channel routing",
        subsystems=("alerting",),
        anchor="OMNI_DIRECTIONAL",
        keywords=("notification", "alert", "routing", "channel", "dispatch", "notify"),
    ),
    RosterEntry(
        id="Pan_XXV",
        role="Sensor fusion / peripheral integration",
        subsystems=("data_sources", "core", "integrations"),
        anchor="OMNI_DIRECTIONAL",
        keywords=("sensor", "fusion", "peripheral", "integration", "multimodal", "ingest"),
        domain=DomainType.SCIENTIFIC,
    ),
    RosterEntry(
        id="Persephone_XXVI",
        role="Lifecycle management / archival retrieval",
        subsystems=("data", "federation", "resilience"),
        anchor="OMNI_POTENT",
        keywords=("lifecycle", "archival", "retrieval", "retention", "restore", "federation"),
        domain=DomainType.INFRASTRUCTURE,
    ),
    RosterEntry(
        id="Prometheus_XXVII",
        role="Model training / resource provisioning",
        subsystems=("ml", "automl"),
        anchor="OMNI_SCIENT",
        keywords=("training", "train", "model", "automl", "provision", "hyperparameter"),
        domain=DomainType.SCIENTIFIC,
        code_bearer=True,
    ),
    RosterEntry(
        id="Hecate_XXVIII",
        role="Gateway / protocol translation",
        subsystems=("api", "integrations"),
        anchor="OMNI_UNIVERSAL",
        keywords=("gateway", "protocol", "translation", "adapter", "bridge", "interop"),
    ),
    RosterEntry(
        id="Nyx_XXIX",
        role="Secrets / secure enclave control",
        subsystems=("crypto", "security"),
        anchor="OMNI_INDIVISIBLE",
        keywords=("secrets", "enclave", "key", "secure", "vault", "encryption", "pqc"),
        domain=DomainType.SECURITY,
    ),
    RosterEntry(
        id="Atlas_XXX",
        role="Distributed orchestration / cluster management",
        subsystems=("distributed", "agentic"),
        anchor="OMNI_UNIVERSAL",
        keywords=("distributed", "orchestration", "cluster", "coordination", "shard", "node"),
        domain=DomainType.INFRASTRUCTURE,
        code_bearer=True,
    ),
    RosterEntry(
        id="Harmonia_XXXI",
        role="Data normalization / canonicalization",
        subsystems=("data", "models"),
        anchor="OMNI_UNIVERSAL",
        keywords=("normalization", "canonicalization", "schema", "standardize", "clean",
                  "transform"),
        domain=DomainType.SCIENTIFIC,
    ),
    RosterEntry(
        id="Hyperion_XXXII",
        role="High-performance compute / GPU scheduling",
        subsystems=("scaling", "distributed", "ml"),
        anchor="OMNI_POTENT",
        keywords=("compute", "gpu", "hpc", "scheduling", "accelerator", "parallel"),
        domain=DomainType.INFRASTRUCTURE,
    ),
    RosterEntry(
        id="Rhea_XXXIII",
        role="Dependency management / resilience control",
        subsystems=("resilience",),
        anchor="OMNI_POTENT",
        keywords=("dependency", "resilience", "failover", "circuit-breaker", "recovery",
                  "self-healing"),
        domain=DomainType.INFRASTRUCTURE,
        code_bearer=True,
    ),
)


# Internal routing floor — NOT a member of the public 33-member roster. Used only
# when capability routing finds no specialist match, so the fleet is never
# silent. Full main-agent capability.
GENERALIST_FLOOR = RosterEntry(
    id="_generalist",
    alias="Routing Floor",
    role="Generalist fallback — full main-agent pipeline",
    subsystems=("agentic",),
    anchor="OMNI_DIRECTIONAL",
    keywords=(),
    depth="deep",
    internal=True,
    impl_path=f"{_SPEC}.generalist.GeneralistSubAgent",
)

#: Every constructible entry (the 33 public members plus the internal floor).
ALL_ENTRIES: tuple[RosterEntry, ...] = (*ROSTER, GENERALIST_FLOOR)

_BY_ID: dict[str, RosterEntry] = {e.id: e for e in ALL_ENTRIES}


def entry_by_id(agent_id: str) -> RosterEntry:
    """Look up a roster entry by id.

    Raises:
        KeyError: If the id is not a known pantheon member or the floor.
    """
    if agent_id not in _BY_ID:
        raise KeyError(f"unknown subagent id {agent_id!r}; known: {sorted(_BY_ID)}")
    return _BY_ID[agent_id]


def code_bearers() -> dict[str, str]:
    """Return the Omni-Code -> bearer-id map (the single lead per Code)."""
    return {e.anchor: e.id for e in ROSTER if e.code_bearer}


def validate_roster() -> None:
    """Assert the roster's structural invariants (used by the roster tests).

    Validates: unique ids; valid depth labels; deep entries carry an
    ``impl_path`` and coordinators do not; every anchor is one of the Seven
    Omni-Codes; exactly seven code-bearers, one per Code, each bearer's anchor
    equal to the Code it bears; and every subsystem resolves to a real
    ``omni_mercury_engine`` module/package.

    Raises:
        AssertionError: On any violated invariant.
    """
    import importlib.util

    from omni_mercury_engine.utils.constants import OmniCodes

    valid_codes = set(OmniCodes.get_all())

    ids = [e.id for e in ALL_ENTRIES]
    assert len(ids) == len(set(ids)), "duplicate subagent ids in roster"
    assert len(ROSTER) == 33, f"expected 33 public members, found {len(ROSTER)}"

    for e in ALL_ENTRIES:
        assert e.depth in {"deep", "coordinator"}, f"{e.id}: bad depth {e.depth!r}"
        if e.depth == "deep":
            assert e.impl_path, f"{e.id}: deep member needs impl_path"
        else:
            assert e.impl_path is None, f"{e.id}: coordinator must not set impl_path"
        assert e.anchor in valid_codes, f"{e.id}: unknown anchor {e.anchor!r}"
        assert e.subsystems, f"{e.id}: must bind at least one subsystem"
        for sub in e.subsystems:
            spec = importlib.util.find_spec(subsystem_module(sub))
            assert spec is not None, f"{e.id}: subsystem {sub!r} does not resolve"

    bearers = code_bearers()
    assert len(bearers) == 7, f"expected 7 code-bearers, found {len(bearers)}"
    assert set(bearers) == valid_codes, "code-bearers do not cover all seven Omni-Codes"

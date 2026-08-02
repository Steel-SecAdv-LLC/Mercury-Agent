# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Multi-agent orchestration over the live detector ensemble (vision pillar B).

This module is the wiring that revives the dormant planning / coordination /
reflexion / chain-of-thought tier (DORMANCY_LEDGER rows 10-11) into a single
live, measurable execution path:

* ``hierarchical_planning.HierarchicalPlanner`` **plans and drives** the
  detection episode: the planner-executed stages of ``detect()`` are
  score -> consensus -> decide, registered as real Options whose
  initiation/termination predicates read the *actual* pipeline state
  (``fit()`` is the precondition stage that arms them, and reflection runs
  in ``run_episode()`` after detection when labels arrive);
  ``select_action`` chooses each next stage and ``update_on_feedback``
  receives real stage outcomes as TD rewards.
* ``multi_agent_coordination.AgentCoordinator`` + ``ConsensusProtocol``
  **coordinate** per-sample consensus across agents that wrap the engine's
  real detectors (``engine.detectors``) — not toy scorers.
* ``reflexion.AnomalyReflexion`` is the **critic**: every issued decision is
  recorded against real ground-truth feedback when it arrives, and the
  resulting threshold recommendation is routed through a
  :class:`~omni_mercury_engine.governance.self_improvement.ThresholdGovernance`
  policy before it can move the live operating point. The default policy is
  fail-closed (Phase 3 governed self-improvement): an autonomous threshold
  change is *withheld* pending promotion-gate evidence and human approval;
  measurement harnesses install an explicit ``MeasurementGovernance`` to apply
  it and measure its effect (verbal reinforcement with real signal, Shinn
  et al. 2023).
* ``chain_of_thought.AnomalyChainOfThought`` is the **depictor**: each
  decision can be explained by a reasoning trace whose stated determination
  is contractually locked to the issued decision (trace fidelity), because
  the trace classifies against the same operating threshold.

Transparency contract (anti-theater): no stage fabricates signal. Quorum failures
abstain explicitly instead of defaulting to "benign"; agent failures are
surfaced, never silently dropped below quorum; the ethical gates
(BenevolenceScorer + sigma_Immutable) run fail-closed at the decision
boundary exactly as on the engine and cognitive-orchestrator boundaries; and
the measurable claims for this layer are pinned by
``benchmarks/orchestration_validation.py`` on real ADBench labels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from omni_mercury_engine.cognitive.chain_of_thought import (
    AnomalyChainOfThought,
    ChainOfThoughtEngine,
)
from omni_mercury_engine.cognitive.ethical_bounding import (
    MINIMUM_BENEVOLENCE_FLOOR,
    BenevolenceScorer,
    sanitize_domain,
)
from omni_mercury_engine.cognitive.hierarchical_planning import (
    AbstractionLevel,
    GoalStatus,
    HierarchicalPlanner,
    Option,
    PlanExecutionState,
)
from omni_mercury_engine.cognitive.multi_agent_coordination import (
    AgentCoordinator,
    AgentRole,
    ConsensusMethod,
    ConsensusProtocol,
    CoordinationStrategy,
    DetectionAgent,
    DetectionResult,
)
from omni_mercury_engine.cognitive.reflexion import AnomalyReflexion
from omni_mercury_engine.governance.self_improvement import (
    FailClosedSelfImprovementGovernance,
    GovernanceOutcome,
    ProposedThresholdChange,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from omni_mercury_engine.agentic._incremental_serving import _ServingCache
    from omni_mercury_engine.governance.self_improvement import ThresholdGovernance

logger = logging.getLogger(__name__)

_DEFAULT_DOMAIN = "general"

# Pipeline stage names. These are the *actions* the hierarchical planner
# selects; each maps to one real orchestrator operation.
_STAGE_FIT = "fit_agents"
_STAGE_SCORE = "score_agents"
_STAGE_CONSENSUS = "form_consensus"
_STAGE_DECIDE = "issue_decisions"

_ROLE_BY_NAME: dict[str, AgentRole] = {
    "statistical": AgentRole.STATISTICAL,
    "temporal": AgentRole.TEMPORAL,
    "spatial": AgentRole.SPATIAL,
    "dimensional": AgentRole.DIMENSIONAL,
    "directive": AgentRole.BEHAVIORAL,
}

# Cap on the stored reference batch used to score single samples against a
# batch-relative detector (the detectors normalize scores within a batch, so
# a lone sample needs company to be scored meaningfully).
_MAX_REFERENCE_ROWS = 512


class OrchestrationError(RuntimeError):
    """Raised when the orchestration pipeline cannot proceed transparently.

    This is the fail-closed alternative to fabricating a verdict: below
    quorum, with an unexecutable plan, or with an unfaithful reasoning
    trace, the orchestrator refuses rather than guesses.
    """


def default_detector_suite() -> dict[str, Any]:
    """Build the engine's base detector set for standalone orchestration.

    Mirrors ``OmniMercuryEngine._init_detectors`` so the orchestrator
    coordinates the same real detectors whether constructed standalone or
    via :meth:`MultiAgentOrchestrator.from_engine`. Detectors whose imports
    fail (optional dependencies absent) are skipped with a warning — the
    quorum check downstream decides whether enough survive to proceed.
    """
    suite: dict[str, Any] = {}
    candidates: list[tuple[str, str, str]] = [
        ("statistical", "omni_mercury_engine.detectors.statistical", "MercuryAnomalyDetector"),
        ("temporal", "omni_mercury_engine.detectors.temporal", "TemporalAnomalyDetector"),
        ("spatial", "omni_mercury_engine.detectors.spatial", "SpatialAnomalyDetector"),
        ("dimensional", "omni_mercury_engine.detectors.dimensional", "DimensionalAnalyzer"),
        ("directive", "omni_mercury_engine.detectors.directive", "SigmaDirectiveDetector"),
    ]
    import importlib

    for name, module_path, class_name in candidates:
        try:
            module = importlib.import_module(module_path)
            suite[name] = getattr(module, class_name)()
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.warning("Detector %r unavailable for orchestration: %s", name, exc)
    return suite


class DetectorAgent(DetectionAgent):
    """A coordination agent backed by one of the engine's real detectors.

    Bridges the ``BaseDetector`` contract (``fit(X)`` then
    ``detect(X) -> {"scores": ...}``) to the multi-agent coordination
    protocol's :class:`DetectionAgent` interface. The agent's decision
    threshold and confidence scale are *calibrated from real score
    distributions* at fit time (contamination quantile / score spread), not
    asserted.
    """

    def __init__(
        self,
        agent_id: str,
        detector: Any,
        role: AgentRole | None = None,
        contamination: float = 0.1,
        seed: int | None = None,
    ) -> None:
        """Initialize the agent around a real detector.

        Args:
            agent_id: Unique agent identifier (conventionally the engine's
                detector name: "statistical", "temporal", ...).
            detector: Object satisfying the BaseDetector duck type
                (``fit(X)``; ``detect(X)`` returning a dict with a per-sample
                ``"scores"`` array).
            role: Coordination role; inferred from ``agent_id`` when omitted.
            contamination: Expected anomaly fraction used to place the
                agent's decision threshold at the ``1 - contamination``
                quantile of its own training scores.
            seed: Seed for the reference-batch subsample (determinism).
        """
        super().__init__(agent_id, role or _ROLE_BY_NAME.get(agent_id, AgentRole.SPECIALIST))
        self.detector = detector
        self.contamination = float(contamination)
        self.decision_threshold: float = 0.5
        self.confidence_scale: float = 0.25
        self._reference: np.ndarray[Any, Any] | None = None
        self._rng = np.random.default_rng(seed)
        self._fitted = False
        # Exact incremental single-sample serving (built lazily after fit;
        # None means "tried and unsupported" so the full path is used).
        self._serving_cache: _ServingCache | None = None
        self._serving_cache_built = False

    @property
    def is_fitted(self) -> bool:
        """Whether the underlying detector has been fitted and calibrated."""
        return self._fitted

    def fit(self, X: np.ndarray[Any, Any]) -> DetectorAgent:
        """Fit the wrapped detector and calibrate threshold/confidence.

        The decision threshold is the ``1 - contamination`` quantile of the
        detector's own scores on the fit data; the confidence scale is the
        standard deviation of those scores. Both are measured quantities.
        """
        X = np.asarray(X, dtype=np.float64)
        self.detector.fit(X)
        self._fitted = True

        scores = self.score_batch(X)
        self.decision_threshold = float(np.quantile(scores, 1.0 - self.contamination))
        self.confidence_scale = float(max(np.std(scores), 1e-6))

        if len(X) > _MAX_REFERENCE_ROWS:
            keep = self._rng.choice(len(X), size=_MAX_REFERENCE_ROWS, replace=False)
            self._reference = X[np.sort(keep)]
        else:
            self._reference = X
        self._serving_cache = None
        self._serving_cache_built = False
        return self

    def score_batch(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Score a batch with the real detector; per-sample scores in [0, 1].

        Purity contract: identical batches yield identical scores. Detectors
        that keep transient cross-call memory (the directive detector's
        recursive-memory buffer) are reset before scoring — the agent's
        contract is *batch-relative scoring against fitted state*, not a
        stream continuation, and without the reset the first
        ``memory_depth`` rows of every batch scored differently depending on
        what the previous call happened to leave behind (defect found
        2026-06-11; the detector keeps its documented streaming semantics
        for direct callers).

        Raises:
            OrchestrationError: If the agent is unfitted or the detector
                returns a malformed score vector (fail-closed — a wrong-length
                score array silently mis-assigns scores to samples).
        """
        if not self._fitted:
            raise OrchestrationError(f"agent {self.agent_id!r} used before fit()")
        X = np.asarray(X, dtype=np.float64)
        reset_state = getattr(self.detector, "reset_state", None)
        if callable(reset_state):
            reset_state()
        result = self.detector.detect(X)
        scores = np.asarray(result["scores"], dtype=np.float64).ravel()
        if scores.shape[0] != X.shape[0]:
            raise OrchestrationError(
                f"agent {self.agent_id!r} returned {scores.shape[0]} scores "
                f"for {X.shape[0]} samples"
            )
        if not np.all(np.isfinite(scores)):
            scores = np.nan_to_num(scores, nan=0.5, posinf=1.0, neginf=0.0)
        smin, smax = float(scores.min()), float(scores.max())
        if smin < 0.0 or smax > 1.0:
            # Preserve within-batch ordering while mapping onto the protocol's
            # [0, 1] score contract.
            spread = smax - smin
            scores = np.full_like(scores, 0.5) if spread < 1e-12 else (scores - smin) / spread
        return scores

    def confidence_for(self, score: float) -> float:
        """Calibration-derived confidence for a score.

        Distance from the agent's own threshold in units of its
        training-score spread, clipped to [0, 1].
        """
        return float(
            np.clip(abs(score - self.decision_threshold) / self.confidence_scale, 0.0, 1.0)
        )

    def detect(
        self,
        data: np.ndarray[Any, Any],
        context: dict[str, Any] | None = None,
    ) -> DetectionResult:
        """Protocol-compliant single-sample detection.

        The sample is scored against a reference batch (the calibration
        sample, or ``context["reference_batch"]``) because the underlying
        detectors normalize scores within a batch. For the profile-dominant
        detectors an exact incremental path serves the appended row without
        re-scoring the full reference
        (:mod:`omni_mercury_engine.agentic._incremental_serving`,
        bit-identical, fail-closed to the full path); a caller-supplied
        reference batch always uses the full path.

        Raises:
            ValueError: If given more than one sample — batch scoring must go
                through :meth:`score_batch`, where per-sample results stay
                attributable.
        """
        row = np.asarray(data, dtype=np.float64)
        if row.ndim == 1:
            row = row.reshape(1, -1)
        if row.shape[0] != 1:
            raise ValueError(
                "DetectorAgent.detect() scores a single sample; use score_batch() for batches"
            )
        reference = self._reference
        own_reference = True
        if context is not None and "reference_batch" in context:
            reference = np.asarray(context["reference_batch"], dtype=np.float64)
            own_reference = False
        if reference is not None and len(reference) > 0:
            score = self._serve_incremental(row) if own_reference else None
            if score is None:
                batch = np.vstack([reference, row])
                score = float(self.score_batch(batch)[-1])
        else:
            score = float(self.score_batch(row)[0])
        return DetectionResult(
            agent_id=self.agent_id,
            anomaly_score=score,
            is_anomaly=score > self.decision_threshold,
            confidence=self.confidence_for(score),
            features_used=[self.role.value],
            reasoning=(
                f"{self.agent_id} score {score:.4f} vs calibrated "
                f"threshold {self.decision_threshold:.4f}"
            ),
        )

    def _serve_incremental(self, row_2d: np.ndarray[Any, Any]) -> float | None:
        """Exact incremental score for one row against the fit-time reference.

        Returns ``None`` whenever the fast path cannot guarantee the
        bit-identical full-path score (unsupported detector, stale cache,
        non-finite input), in which case the caller runs the full path.
        """
        if not self._serving_cache_built:
            from omni_mercury_engine.agentic._incremental_serving import build_serving_cache

            self._serving_cache = build_serving_cache(self.detector, self._reference)
            self._serving_cache_built = True
        if self._serving_cache is None:
            return None
        return self._serving_cache.serve(row_2d[0])


@dataclass
class CoordinationBatch:
    """Per-sample consensus over the agent ensemble for one batch.

    Attributes:
        consensus_scores: Confidence-weighted consensus score per sample.
        decisions: Issued boolean decisions at the operating threshold.
        abstained: Per-sample abstention mask (no transparent quorum verdict).
        agreement: Per-sample consensus agreement ratio.
        per_agent_scores: Each participating agent's per-sample scores.
        dissent_counts: Number of dissenting agents per sample.
        participant_count: Agents that contributed scores to this batch.
    """

    consensus_scores: np.ndarray[Any, Any]
    decisions: np.ndarray[Any, Any]
    abstained: np.ndarray[Any, Any]
    agreement: np.ndarray[Any, Any]
    per_agent_scores: dict[str, np.ndarray[Any, Any]]
    dissent_counts: np.ndarray[Any, Any]
    participant_count: int


@dataclass
class ReflectionRecord:
    """Outcome of one reflexion pass over labeled feedback.

    The ``governance_*`` fields record how the Phase 3 governance seam
    disposed of an actionable threshold recommendation: ``applied`` is the
    *effective* outcome (the threshold only moved if governance authorised it),
    while ``governance_outcome``/``governance_reasons`` capture why, and
    ``governance_record`` preserves any routed promotion-gate decision for the
    append-only audit trail.
    """

    n_observations: int
    false_positives: int
    false_negatives: int
    recommendation: str
    threshold_before: float
    threshold_suggested: float
    threshold_after: float
    applied: bool
    reasoning: str = ""
    governed: bool = False
    governance_outcome: str = GovernanceOutcome.MAINTAIN.value
    governance_reasons: list[str] = field(default_factory=list)
    governance_record: dict[str, Any] | None = None


@dataclass
class PlanTrace:
    """Record of the planner-driven episode execution."""

    plan_id: str
    executed_actions: list[str] = field(default_factory=list)
    stage_rewards: list[float] = field(default_factory=list)
    goal_status: str = GoalStatus.PENDING.value
    goal_value: float = 0.0


@dataclass
class EpisodeResult:
    """Result of one orchestrated episode (detect, optionally reflect)."""

    coordination: CoordinationBatch
    plan: PlanTrace
    threshold: float
    reflection: ReflectionRecord | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    benevolence_score: float = 0.0


class MultiAgentOrchestrator:
    """Planner/critic/executor multi-agent orchestration for Mercury.

    Composes the four revived cognitive modules into the engine's own task:

    * **Planner** — :class:`HierarchicalPlanner` selects and sequences the
      real pipeline stages and learns stage values from real TD feedback.
    * **Executor** — :class:`AgentCoordinator` over :class:`DetectorAgent`
      wrappers of the live detectors, with :class:`ConsensusProtocol`
      consensus per sample.
    * **Critic** — :class:`AnomalyReflexion` turns labeled outcomes into an
      operating-threshold adaptation for the next episode.
    * **Depictor** — :class:`AnomalyChainOfThought` renders decision traces
      whose stated determination is locked to the issued decision.

    The decision boundary enforces the same dual hard ethical gates as the
    engine and cognitive orchestrator (benevolence floor + sigma_Immutable),
    fail-closed.
    """

    def __init__(
        self,
        detectors: Mapping[str, Any] | None = None,
        *,
        consensus_method: ConsensusMethod | str = ConsensusMethod.CONFIDENCE_WEIGHTED,
        min_participants: int = 3,
        contamination: float = 0.1,
        operating_threshold: float = 0.5,
        threshold_governance: ThresholdGovernance | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            detectors: Mapping of name -> BaseDetector-like. Defaults to the
                engine's base suite (:func:`default_detector_suite`).
            consensus_method: Consensus method for the protocol.
            min_participants: Quorum; below it every sample abstains.
            contamination: Expected anomaly fraction for agent calibration.
            operating_threshold: Initial decision boundary on the consensus
                score; reflexion *proposes* adapting it, but the change is
                applied only when ``threshold_governance`` authorises it.
            threshold_governance: Phase 3 governance policy consulted before any
                reflexion-proposed threshold move takes effect. Defaults to
                :class:`~omni_mercury_engine.governance.self_improvement.FailClosedSelfImprovementGovernance`
                — autonomous changes are withheld pending promotion-gate
                evidence and human approval. Measurement harnesses pass
                :class:`~omni_mercury_engine.governance.self_improvement.MeasurementGovernance`
                to apply adaptation and measure its effect.
            seed: Seed for deterministic agent calibration and reasoning.
        """
        detector_map = dict(detectors) if detectors is not None else default_detector_suite()
        if not detector_map:
            raise OrchestrationError("no detectors available to orchestrate")

        resolved_method = (
            consensus_method
            if isinstance(consensus_method, ConsensusMethod)
            else ConsensusMethod(consensus_method)
        )
        if resolved_method is not ConsensusMethod.CONFIDENCE_WEIGHTED:
            # Fail fast rather than expose an incoherent contract: the
            # orchestrator's continuous consensus score, reflexion-adapted
            # operating threshold, and trace fidelity are all defined in
            # CONFIDENCE_WEIGHTED semantics. Other protocol methods emit
            # discrete verdicts with no continuous score to threshold or
            # adapt; honoring them here would silently decouple the issued
            # decisions from the selected protocol. Per-method semantics
            # are a future measured extension, not a silent fallback.
            raise OrchestrationError(
                f"MultiAgentOrchestrator implements CONFIDENCE_WEIGHTED consensus "
                f"semantics only (got {resolved_method.value!r}); other methods "
                "remain available directly on ConsensusProtocol"
            )

        self.min_participants = int(min_participants)
        self.operating_threshold = float(operating_threshold)
        # Phase 3 governance seam: reflexion proposes, governance disposes.
        # Default fail-closed — no autonomous mutation of the live operating
        # point. Measurement contexts inject MeasurementGovernance explicitly.
        self._threshold_governance: ThresholdGovernance = (
            threshold_governance or FailClosedSelfImprovementGovernance()
        )
        self._seed = seed

        # --- Executor tier: coordinator over real-detector agents ---------
        self.coordinator = AgentCoordinator(
            strategy=CoordinationStrategy.CENTRALIZED,
            consensus_method=resolved_method,
        )
        self.agents: dict[str, DetectorAgent] = {}
        for index, (name, detector) in enumerate(detector_map.items()):
            agent = DetectorAgent(
                agent_id=name,
                detector=detector,
                contamination=contamination,
                seed=None if seed is None else seed + index,
            )
            self.agents[name] = agent
            self.coordinator.register_agent(agent)

        self.protocol = ConsensusProtocol(
            method=self.coordinator.consensus_method,
            min_participants=self.min_participants,
        )

        # --- Planner tier: pipeline stages as real options ----------------
        self.planner = HierarchicalPlanner()
        self._register_pipeline_options()
        # One stable goal identity across episodes: TD value estimates live
        # under the goal's ID, so a per-episode goal would silently reset the
        # learned values every call. Plan once; execute per episode.
        self._detection_goal = self.planner.create_goal(
            description="detect_anomaly batch via coordinated real-detector consensus",
            level=AbstractionLevel.STRATEGIC,
            priority=0.8,
            postconditions=["decisions_issued"],
        )
        self._pipeline_plan = self.planner.plan(
            self._detection_goal, self._initial_pipeline_state()
        )

        # --- Critic tier: reflexion at the live operating point -----------
        self.reflexion = AnomalyReflexion(anomaly_threshold=self.operating_threshold)

        # --- Depictor tier: trace generation locked to the boundary -------
        self._cot_engine = ChainOfThoughtEngine(seed=seed)
        self.chain_of_thought = AnomalyChainOfThought(
            cot_engine=self._cot_engine,
            anomaly_threshold=self.operating_threshold,
        )

        # --- Ethical gates (eager, fail-closed; mirrors the engine and the
        # cognitive orchestrator decision boundaries) -----------------------
        self._benevolence_scorer = BenevolenceScorer(
            benevolence_threshold=MINIMUM_BENEVOLENCE_FLOOR,
        )
        from omni_mercury_engine.security.sigma_immutable_gate import get_sigma_immutable_gate

        self._sigma_immutable_gate = get_sigma_immutable_gate()

        self._fitted = False
        logger.info(
            "MultiAgentOrchestrator initialized (agents=%s, quorum=%d, threshold=%.2f)",
            sorted(self.agents),
            self.min_participants,
            self.operating_threshold,
        )

    @classmethod
    def from_engine(cls, engine: Any, **kwargs: Any) -> MultiAgentOrchestrator:
        """Build an orchestrator over an engine's registered detectors."""
        return cls(dict(engine.detectors), **kwargs)

    # ------------------------------------------------------------------
    # Planner wiring
    # ------------------------------------------------------------------

    @staticmethod
    def _initial_pipeline_state() -> dict[str, Any]:
        """Canonical episode-start state (one state key for TD learning)."""
        return {
            "data_available": True,
            "agents_fitted": True,
            "scores_computed": False,
            "consensus_reached": False,
            "decisions_issued": False,
        }

    def _register_pipeline_options(self) -> None:
        """Register the real pipeline stages as planner options.

        Initiation and termination predicates read the actual pipeline
        state flags, so exactly one stage is applicable at each step and the
        planner's ``select_action`` genuinely sequences the episode.
        """
        stages = [
            Option(
                option_id="opt_pipeline_fit",
                name="Fit detector agents",
                initiation_set={"data_available": True, "agents_fitted": False},
                policy={"default": _STAGE_FIT},
                termination_condition={"agents_fitted": True},
                expected_duration=4.0,
                expected_reward=0.6,
                skill_level=0.9,
            ),
            Option(
                option_id="opt_pipeline_score",
                name="Score batch across agents",
                initiation_set={"agents_fitted": True, "scores_computed": False},
                policy={"default": _STAGE_SCORE},
                termination_condition={"scores_computed": True},
                expected_duration=2.0,
                expected_reward=0.7,
                skill_level=0.9,
            ),
            Option(
                option_id="opt_pipeline_consensus",
                name="Form per-sample consensus",
                initiation_set={"scores_computed": True, "consensus_reached": False},
                policy={"default": _STAGE_CONSENSUS},
                termination_condition={"consensus_reached": True},
                expected_duration=1.0,
                expected_reward=0.8,
                skill_level=0.9,
            ),
            Option(
                option_id="opt_pipeline_decide",
                name="Issue gated decisions",
                initiation_set={"consensus_reached": True, "decisions_issued": False},
                policy={"default": _STAGE_DECIDE},
                termination_condition={"decisions_issued": True},
                expected_duration=1.0,
                expected_reward=0.9,
                skill_level=0.9,
            ),
        ]
        for option in stages:
            self.planner.option_library.add_option(option)

    # ------------------------------------------------------------------
    # Executor tier
    # ------------------------------------------------------------------

    def fit(self, X_train: np.ndarray[Any, Any]) -> MultiAgentOrchestrator:
        """Fit and calibrate every agent on real training data.

        Agents whose detectors fail to fit are unregistered with a warning;
        if the survivors fall below quorum the orchestrator refuses to
        operate (fail-closed) rather than coordinate a hollow ensemble.
        """
        X_train = np.asarray(X_train, dtype=np.float64)
        failed: list[str] = []
        for name, agent in list(self.agents.items()):
            try:
                agent.fit(X_train)
            except Exception as exc:
                failed.append(name)
                logger.warning("Agent %r failed to fit and is excluded: %s", name, exc)
                self.coordinator.unregister_agent(name)
                del self.agents[name]
        if len(self.agents) < self.min_participants:
            raise OrchestrationError(
                f"only {len(self.agents)} agents fitted "
                f"(quorum {self.min_participants}); failed={failed}"
            )
        self._fitted = True
        return self

    def coordinate(self, X: np.ndarray[Any, Any]) -> CoordinationBatch:
        """Per-sample consensus across the fitted agents on a real batch.

        Each agent scores the batch once (batch-relative normalization is the
        detectors' native mode); the consensus protocol then aggregates the
        per-sample :class:`DetectionResult` votes. The continuous consensus
        score is the confidence-weighted mean of agent scores — the same
        statistic the CONFIDENCE_WEIGHTED protocol thresholds for its
        decision.

        Agents score concurrently on a thread pool: scoring is a pure
        function of (fitted state, batch) — the purity contract on
        :meth:`DetectorAgent.score_batch` — so results are bit-identical to
        serial scoring (pinned by ``tests/test_native_acceleration.py``);
        measured wall-clock 1.2x on the 4-core profiling box (72.8 -> 62.2 ms
        on cardio-scale, 285 -> 228 ms at 2.2k rows), bounded by the
        detectors' own internal parallelism. Per-agent failures are excluded
        with a warning exactly as before; the quorum check decides whether
        enough survive.
        """
        if not self._fitted:
            raise OrchestrationError("orchestrator used before fit()")
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]

        per_agent_scores: dict[str, np.ndarray[Any, Any]] = {}
        agent_items = list(self.agents.items())
        if len(agent_items) > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=len(agent_items)) as pool:
                futures = {name: pool.submit(agent.score_batch, X) for name, agent in agent_items}
            for name, future in futures.items():
                try:
                    per_agent_scores[name] = future.result()
                except Exception as exc:
                    logger.warning("Agent %r failed to score batch; excluded: %s", name, exc)
        else:
            for name, agent in agent_items:
                try:
                    per_agent_scores[name] = agent.score_batch(X)
                except Exception as exc:
                    logger.warning("Agent %r failed to score batch; excluded: %s", name, exc)

        participant_count = len(per_agent_scores)
        consensus_scores = np.full(n, 0.5, dtype=np.float64)
        decisions = np.zeros(n, dtype=bool)
        abstained = np.ones(n, dtype=bool)
        agreement = np.zeros(n, dtype=np.float64)
        dissent_counts = np.zeros(n, dtype=np.int64)

        if participant_count < self.min_participants:
            logger.warning(
                "Quorum failure: %d/%d agents scored; every sample abstains",
                participant_count,
                self.min_participants,
            )
            return CoordinationBatch(
                consensus_scores=consensus_scores,
                decisions=decisions,
                abstained=abstained,
                agreement=agreement,
                per_agent_scores=per_agent_scores,
                dissent_counts=dissent_counts,
                participant_count=participant_count,
            )

        agent_names = list(per_agent_scores)
        score_matrix = np.vstack([per_agent_scores[name] for name in agent_names])
        thresholds = np.array([[self.agents[name].decision_threshold] for name in agent_names])
        scales = np.array([[self.agents[name].confidence_scale] for name in agent_names])
        votes = score_matrix > thresholds
        confidences = np.clip(np.abs(score_matrix - thresholds) / scales, 0.0, 1.0)

        # Vectorized form of ConsensusProtocol._confidence_weighted — the
        # protocol remains the semantic authority: a deterministic subsample
        # of every batch is re-derived through the real protocol and any
        # divergence fails the batch closed. (The constructor guarantees the
        # CONFIDENCE_WEIGHTED method, so this is the only consensus path.)
        total_confidence = confidences.sum(axis=0)
        safe_total = np.where(total_confidence > 0, total_confidence, 1.0)
        weighted = (confidences * score_matrix).sum(axis=0)
        consensus_scores = np.where(total_confidence > 0, weighted / safe_total, 0.5)
        protocol_decisions = consensus_scores > 0.5
        agree_weight = (confidences * (votes == protocol_decisions[None, :])).sum(axis=0)
        agreement = np.where(total_confidence > 0, agree_weight / safe_total, 0.0)
        dissent_counts = (votes != protocol_decisions[None, :]).sum(axis=0)
        abstained[:] = False
        decisions = consensus_scores > self.operating_threshold
        self._spot_check_consensus(
            agent_names,
            score_matrix,
            votes,
            confidences,
            consensus_scores,
            agreement,
            dissent_counts,
        )

        return CoordinationBatch(
            consensus_scores=consensus_scores,
            decisions=decisions,
            abstained=abstained,
            agreement=agreement,
            per_agent_scores=per_agent_scores,
            dissent_counts=dissent_counts,
            participant_count=participant_count,
        )

    def _spot_check_consensus(
        self,
        agent_names: list[str],
        score_matrix: np.ndarray[Any, Any],
        votes: np.ndarray[Any, Any],
        confidences: np.ndarray[Any, Any],
        consensus_scores: np.ndarray[Any, Any],
        agreement: np.ndarray[Any, Any],
        dissent_counts: np.ndarray[Any, Any],
        n_checks: int = 8,
    ) -> None:
        """Verify the vectorized consensus against the real protocol.

        A deterministic, evenly-spaced subsample of the batch is re-derived
        sample-by-sample through ``ConsensusProtocol.reach_consensus``; any
        divergence beyond float tolerance raises (fail-closed) — the fast
        path is a compiled form of the protocol, never a replacement for it.
        """
        n = score_matrix.shape[1]
        if n == 0:
            return
        check_indices = np.unique(np.linspace(0, n - 1, num=min(n_checks, n)).astype(int))
        for i in check_indices:
            results = [
                DetectionResult(
                    agent_id=name,
                    anomaly_score=float(score_matrix[j, i]),
                    is_anomaly=bool(votes[j, i]),
                    confidence=float(confidences[j, i]),
                )
                for j, name in enumerate(agent_names)
            ]
            reference = self.protocol.reach_consensus(results)
            if isinstance(reference, dict):
                raise OrchestrationError("consensus protocol returned a non-result payload")
            # The protocol reports confidence = avg if decision else 1 - avg.
            reference_score = (
                reference.confidence if reference.final_decision else 1.0 - reference.confidence
            )
            if (
                abs(reference_score - float(consensus_scores[i])) > 1e-9
                or abs(reference.agreement_ratio - float(agreement[i])) > 1e-9
                or len(reference.dissenting_agents) != int(dissent_counts[i])
            ):
                raise OrchestrationError(
                    f"consensus fast path diverged from ConsensusProtocol at sample {i}: "
                    f"score {consensus_scores[i]:.12f} vs {reference_score:.12f}, "
                    f"agreement {agreement[i]:.12f} vs {reference.agreement_ratio:.12f}"
                )

    # ------------------------------------------------------------------
    # Decision boundary (ethical gates)
    # ------------------------------------------------------------------

    def _enforce_ethics(self, batch: CoordinationBatch, domain: str) -> float:
        """Run the dual hard ethical gates before decisions are issued.

        Two independent gates run in order, both fail-closed: the shared
        harm-uplift choke point
        (:func:`~omni_mercury_engine.cognitive.decision_gate.enforce_decision_boundary`)
        over the **real** orchestration decision, then the sigma_Immutable gate
        over the calibrated 256-d scalar vector. Both raise
        :class:`EthicalConstraintViolationError`; nothing downstream of a failed
        gate executes.

        The first gate used to be a benevolence pass-bar scored on a canned
        keyword string with caller text deliberately withheld. Under a
        pass-on-positive-vocabulary control that withholding was protective;
        under a block-on-harm control it only hides evidence, so the real
        decision is gated now.

        Returns:
            The advisory benevolence score (for the episode record).
        """
        from omni_mercury_engine.cognitive.decision_gate import (
            DecisionSubject,
            enforce_decision_boundary,
        )
        from omni_mercury_engine.security.sigma_immutable_gate import (
            PERMITTED_ETHICAL_POSTURE,
            build_sigma_immutable_vector,
        )

        safe_domain = sanitize_domain(domain or _DEFAULT_DOMAIN)
        severity = float(np.max(batch.consensus_scores)) if len(batch.consensus_scores) else 0.0
        anomaly_prob = (
            float(np.mean(batch.consensus_scores)) if len(batch.consensus_scores) else 0.0
        )

        verdict = enforce_decision_boundary(
            DecisionSubject(
                surface="MultiAgentOrchestrator.detect",
                operation=(
                    "coordinate planner, critic and executor agents over the caller's "
                    "input and issue a consensus anomaly verdict"
                ),
                domain=safe_domain,
                payload={"severity": round(severity, 4), "anomaly_prob": round(anomaly_prob, 4)},
            ),
            advisory_scorer=self._benevolence_scorer,
        )
        advisory_benevolence = float(
            verdict.benevolence if verdict.benevolence is not None else float("nan")
        )

        # The ethical band carries the configured posture, not a per-call
        # content score — see ``PERMITTED_ETHICAL_POSTURE``.
        sigma_vector = build_sigma_immutable_vector(
            benevolence_score=PERMITTED_ETHICAL_POSTURE,
            severity=severity,
            anomaly_prob=anomaly_prob,
        )
        self._sigma_immutable_gate.enforce(
            action=f"MultiAgentOrchestrator.detect:{safe_domain}",
            scalar_vector=sigma_vector,
            details={
                "boundary": "MultiAgentOrchestrator.detect",
                "domain": safe_domain,
                "severity": severity,
                "anomaly_prob": anomaly_prob,
            },
        )
        return advisory_benevolence

    # ------------------------------------------------------------------
    # Planner-driven episode
    # ------------------------------------------------------------------

    def detect(
        self,
        X: np.ndarray[Any, Any],
        domain: str = _DEFAULT_DOMAIN,
    ) -> EpisodeResult:
        """Run one planner-driven detection episode on a real batch.

        The hierarchical planner selects each pipeline stage from the live
        state via ``select_action``; each executed stage feeds a real reward
        back through ``update_on_feedback`` (TD learning on the actual
        pipeline, not a simulation). An unexecutable plan raises — the
        orchestrator never silently bypasses its planner.
        """
        if not self._fitted:
            raise OrchestrationError("orchestrator used before fit()")
        X = np.asarray(X, dtype=np.float64)

        state = self._initial_pipeline_state()
        initial_state = dict(state)
        root_goal = self._detection_goal
        root_goal.status = GoalStatus.ACTIVE
        execution = PlanExecutionState(plan=self._pipeline_plan, current_goal=root_goal)
        trace = PlanTrace(plan_id=self._pipeline_plan.plan_id)

        batch: CoordinationBatch | None = None
        benevolence_score = 0.0

        for _ in range(8):  # 3 stages + slack; the plan terminates well before
            if state["decisions_issued"]:
                break
            action, _option = self.planner.select_action(state, execution)
            previous_state = dict(state)

            if action == _STAGE_SCORE:
                batch = self.coordinate(X)
                state["scores_computed"] = True
                reward = 1.0 if batch.participant_count >= self.min_participants else 0.0
            elif action == _STAGE_CONSENSUS:
                if batch is None:
                    raise OrchestrationError("plan reached consensus stage without scores")
                state["consensus_reached"] = True
                # Reward consensus formation by how much of the batch obtained
                # a quorum verdict (abstentions are transparent but unrewarded).
                reward = float(1.0 - np.mean(batch.abstained))
            elif action == _STAGE_DECIDE:
                if batch is None:
                    raise OrchestrationError("plan reached decision stage without consensus")
                benevolence_score = self._enforce_ethics(batch, domain)
                state["decisions_issued"] = True
                # The TD reward reflects what was actually delivered: only
                # quorum-backed (non-abstained) decisions count. An
                # all-abstention batch completes the plan transparently but earns
                # no value — abstaining is correct, not rewarding.
                reward = float(1.0 - np.mean(batch.abstained))
            elif action == _STAGE_FIT:
                # Agents are fitted before detect(); the initiation predicate
                # (agents_fitted=False) makes this unreachable here, and
                # reaching it means the planner state is corrupt.
                raise OrchestrationError("plan selected fit stage on a fitted orchestrator")
            else:
                raise OrchestrationError(
                    f"plan produced no executable stage (action={action!r}); refusing to "
                    "bypass the planner"
                )

            trace.executed_actions.append(action)
            trace.stage_rewards.append(reward)
            self.planner.update_on_feedback(previous_state, action, reward, state, execution)

        if batch is None or not state["decisions_issued"]:
            raise OrchestrationError(
                f"planned episode did not complete (executed={trace.executed_actions})"
            )

        # Goal completion is judged on the goal's own postconditions against
        # the real final state — the planner's contract, applied transparently.
        if all(bool(state.get(cond)) for cond in root_goal.postconditions):
            root_goal.status = GoalStatus.COMPLETED
        trace.goal_status = root_goal.status.value
        trace.goal_value = self.planner.value_function.get_value(initial_state, root_goal)

        return EpisodeResult(
            coordination=batch,
            plan=trace,
            threshold=self.operating_threshold,
            benevolence_score=benevolence_score,
        )

    # ------------------------------------------------------------------
    # Critic tier (reflexion on real feedback)
    # ------------------------------------------------------------------

    def reflect(
        self,
        batch: CoordinationBatch,
        y_true: np.ndarray[Any, Any],
        *,
        apply: bool = True,
    ) -> ReflectionRecord:
        """Record real outcomes and route any threshold change through governance.

        Every non-abstained decision is recorded against its true label
        (TP/FP/FN/TN with real error magnitudes); the accumulated error balance
        produces a threshold recommendation. When ``apply`` is set and the
        recommendation is actionable, the proposed change is handed to the
        Phase 3
        :class:`~omni_mercury_engine.governance.self_improvement.ThresholdGovernance`
        policy, which decides — fail closed by default — whether it may move the
        live operating point. The returned :class:`ReflectionRecord` reports both
        the effective outcome (``applied``) and the governance disposition.
        """
        y = np.asarray(y_true).astype(bool).ravel()
        if y.shape[0] != batch.consensus_scores.shape[0]:
            raise ValueError(
                f"labels ({y.shape[0]}) and decisions "
                f"({batch.consensus_scores.shape[0]}) length mismatch"
            )

        recorded = 0
        for i in range(y.shape[0]):
            if batch.abstained[i]:
                continue  # an abstention is not a decision; nothing to critique
            self.reflexion.record_detection(
                prediction=float(batch.consensus_scores[i]),
                ground_truth=bool(y[i]),
                features={
                    "consensus_score": float(batch.consensus_scores[i]),
                    "agreement": float(batch.agreement[i]),
                    "dissent": int(batch.dissent_counts[i]),
                },
            )
            recorded += 1

        recommendation = self.reflexion.get_threshold_recommendation()
        suggested = float(recommendation["suggested_threshold"])
        before = self.operating_threshold
        rec = str(recommendation["recommendation"])
        actionable = rec != "maintain"

        # Phase 3 governance: an actionable threshold change is a *proposal*,
        # not a fait accompli. It moves the live operating point only if the
        # governance policy authorises it; the default policy (fail-closed)
        # withholds every autonomous move, so the boundary changes only via an
        # evidence-backed, human-approved promotion.
        applied = False
        governed = bool(apply and actionable)
        governance_outcome = GovernanceOutcome.MAINTAIN.value
        governance_reasons: list[str] = []
        governance_record: dict[str, Any] | None = None
        if actionable and not apply:
            governance_outcome = GovernanceOutcome.NOT_REQUESTED.value
        elif actionable and apply:
            change = ProposedThresholdChange(
                surface="reflexion_threshold",
                recommendation=rec,
                current_threshold=before,
                suggested_threshold=suggested,
                reasoning=str(recommendation.get("reasoning", "")),
                evidence={
                    "false_positives": int(recommendation["false_positives"]),
                    "false_negatives": int(recommendation["false_negatives"]),
                    "n_observations": recorded,
                },
            )
            review = self._threshold_governance.review_threshold_change(change)
            governance_outcome = review.outcome
            governance_reasons = list(review.reasons)
            governance_record = dict(review.record) if review.record is not None else None
            if review.applied:
                self.set_operating_threshold(suggested)
                applied = True

        return ReflectionRecord(
            n_observations=recorded,
            false_positives=int(recommendation["false_positives"]),
            false_negatives=int(recommendation["false_negatives"]),
            recommendation=rec,
            threshold_before=before,
            threshold_suggested=suggested,
            threshold_after=self.operating_threshold,
            applied=applied,
            reasoning=str(recommendation.get("reasoning", "")),
            governed=governed,
            governance_outcome=governance_outcome,
            governance_reasons=governance_reasons,
            governance_record=governance_record,
        )

    def set_operating_threshold(self, threshold: float) -> None:
        """Move the live operating point, keeping every tier in agreement.

        Operator-grade control: the threshold is clipped to [0.05, 0.95] so
        neither reflexion nor a caller can push the boundary into a region
        where every (or no) sample trivially alarms.
        """
        self.operating_threshold = float(np.clip(threshold, 0.05, 0.95))
        self.reflexion.anomaly_threshold = self.operating_threshold
        self.chain_of_thought.anomaly_threshold = self.operating_threshold

    def reset_reflexion(self) -> None:
        """Discard accumulated experience and start a fresh critic.

        Used when the feedback regime changes (new deployment, new label
        source) so stale error statistics cannot steer the threshold.
        """
        self.reflexion = AnomalyReflexion(anomaly_threshold=self.operating_threshold)

    def run_episode(
        self,
        X: np.ndarray[Any, Any],
        y_true: np.ndarray[Any, Any] | None = None,
        domain: str = _DEFAULT_DOMAIN,
        *,
        apply_reflection: bool = True,
    ) -> EpisodeResult:
        """Detect, and when ground truth is supplied, reflect and adapt.

        Metrics are computed only over non-abstained samples — an abstention
        is reported as such, never graded as a benign call.
        """
        episode = self.detect(X, domain=domain)
        if y_true is not None:
            y = np.asarray(y_true).astype(bool).ravel()
            episode.reflection = self.reflect(episode.coordination, y, apply=apply_reflection)
            episode.metrics = self.compute_metrics(episode.coordination, y)
        return episode

    @staticmethod
    def compute_metrics(batch: CoordinationBatch, y: np.ndarray[Any, Any]) -> dict[str, float]:
        """Transparent confusion metrics over the non-abstained decisions."""
        decided = ~batch.abstained
        if not np.any(decided):
            return {"n_decided": 0.0, "abstention_rate": 1.0}
        pred = batch.decisions[decided]
        truth = y[decided]
        tp = float(np.sum(pred & truth))
        tn = float(np.sum(~pred & ~truth))
        fp = float(np.sum(pred & ~truth))
        fn = float(np.sum(~pred & truth))
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return {
            "n_decided": float(np.sum(decided)),
            "abstention_rate": float(np.mean(batch.abstained)),
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "balanced_accuracy": (tpr + tnr) / 2.0,
        }

    # ------------------------------------------------------------------
    # Depictor tier (trace fidelity)
    # ------------------------------------------------------------------

    def explain(
        self,
        episode: EpisodeResult,
        index: int,
        domain: str = _DEFAULT_DOMAIN,
    ) -> dict[str, Any]:
        """Generate a chain-of-thought trace for one issued decision.

        Fidelity contract: the trace's stated determination must agree with
        the decision *as issued* — it classifies against the episode's own
        threshold (reflexion may have moved the live operating point since).
        A disagreement indicates internal corruption and raises instead of
        shipping an unfaithful depiction.
        """
        batch = episode.coordination
        if index < 0 or index >= batch.consensus_scores.shape[0]:
            raise IndexError(f"sample index {index} out of range")
        if batch.abstained[index]:
            return {
                "abstained": True,
                "conclusion": "ABSTAINED - no quorum verdict for this sample",
                "reasoning_chain": [],
                "anomaly_score": float(batch.consensus_scores[index]),
            }

        score = float(batch.consensus_scores[index])
        issued = bool(batch.decisions[index])
        depictor = AnomalyChainOfThought(
            cot_engine=self._cot_engine,
            anomaly_threshold=episode.threshold,
        )
        analysis = depictor.analyze_anomaly(
            {
                "score": score,
                "domain": domain,
                "detectors": sorted(batch.per_agent_scores),
                "evidence": [
                    f"{name}={float(scores[index]):.4f}"
                    for name, scores in sorted(batch.per_agent_scores.items())
                ],
                "agreement": float(batch.agreement[index]),
            },
            score,
            domain=domain,
        )
        stated_conclusion = str(analysis["conclusion"]).upper()
        stated_anomaly = "ANOMALY DETECTED" in stated_conclusion or stated_conclusion.startswith(
            "ANOMALY"
        )
        if stated_anomaly != issued:
            raise OrchestrationError(
                f"trace fidelity violation at sample {index}: trace says "
                f"anomaly={stated_anomaly}, issued decision={issued}"
            )
        analysis["abstained"] = False
        analysis["issued_decision"] = issued
        return dict(analysis)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_statistics(self) -> dict[str, Any]:
        """Aggregate statistics across all four orchestrated tiers."""
        return {
            "agents": {
                name: {
                    "fitted": agent.is_fitted,
                    "decision_threshold": agent.decision_threshold,
                    "role": agent.role.value,
                }
                for name, agent in self.agents.items()
            },
            "operating_threshold": self.operating_threshold,
            "min_participants": self.min_participants,
            "coordinator": self.coordinator.get_statistics(),
            "planner": self.planner.get_statistics(),
            "reflexion": self.reflexion.engine.get_statistics(),
            "chain_of_thought": self.chain_of_thought.cot_engine.get_statistics(),
        }

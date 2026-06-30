# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Detection subagent: Mercury's own multi-agent anomaly detection, delegated.

This specialization wraps Mercury's *real* detection capability — the
:class:`~omni_mercury_engine.agentic.orchestration.MultiAgentOrchestrator` over
the engine's live detector suite — so the main agent can fan detection work out
across the fleet (e.g. many replicas over shards of a stream) without
re-implementing detection. It performs genuine planner-driven, consensus-based,
ethically-gated detection; it never fabricates a verdict, and it abstains
honestly below quorum.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from omni_mercury_engine.agentic.subagents.base import (
    SubAgent,
    SubAgentExecutionError,
)

if TYPE_CHECKING:
    from omni_mercury_engine.agentic.subagents.base import SubAgentTask


class DetectionSubAgent(SubAgent):
    """Delegated multi-agent anomaly detection over a real detector ensemble."""

    def _perform(self, task: SubAgentTask) -> tuple[Any, float, str]:
        """Run real multi-agent detection on ``payload['data']``.

        ``payload['data']`` is the batch to score; optional ``payload['train']``
        supplies fit data (defaults to the batch itself — transductive). The
        episode runs the planner/consensus/critic tiers with the dual ethical
        gate at its own decision boundary; this subagent surfaces the honest
        outcome (including abstention) and never substitutes synthetic signal.
        """
        raw = task.payload.get("data")
        if raw is None:
            raise SubAgentExecutionError("detection requires payload['data'] (the batch to score)")
        X = np.asarray(raw, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[0] == 0:
            raise SubAgentExecutionError(
                f"detection expects a non-empty 2-D batch; got shape {X.shape}"
            )
        train = np.asarray(task.payload.get("train", X), dtype=np.float64)
        if train.ndim == 1:
            train = train.reshape(1, -1)

        # Imported here (not at module load) to avoid a construction-time
        # dependency on the heavy orchestration stack for non-detection fleets.
        from omni_mercury_engine.agentic.orchestration import (
            MultiAgentOrchestrator,
            OrchestrationError,
        )

        domain = getattr(task.domain, "value", str(task.domain))
        try:
            orchestrator = MultiAgentOrchestrator(seed=self._seed)
            orchestrator.fit(train)
            episode = orchestrator.detect(X, domain=domain)
        except OrchestrationError as exc:
            # A fail-closed orchestration refusal (e.g. below quorum) is an
            # honest failure of this task, surfaced — not a fabricated verdict.
            raise SubAgentExecutionError(f"detection orchestration refused: {exc}") from exc

        batch = episode.coordination
        decided_mask = ~batch.abstained
        n_decided = int(decided_mask.sum())
        n_anomalies = int(batch.decisions[decided_mask].sum()) if n_decided else 0
        confidence = float(np.mean(batch.agreement[decided_mask])) if n_decided else 0.0
        output = {
            "n_samples": int(X.shape[0]),
            "n_decided": n_decided,
            "n_abstained": int(batch.abstained.sum()),
            "n_anomalies": n_anomalies,
            "operating_threshold": episode.threshold,
            "benevolence_score": episode.benevolence_score,
            "consensus_scores": batch.consensus_scores.tolist(),
            "decisions": batch.decisions.tolist(),
        }
        reasoning = (
            f"multi-agent detection over {X.shape[0]} samples: {n_anomalies} anomalies "
            f"among {n_decided} decided ({int(batch.abstained.sum())} abstained)"
        )
        return output, confidence, reasoning

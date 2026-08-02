# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Global Omni-Scalar Network (GOSNN) - Intelligence Fusion Hub.

Implements a comprehensive scalar monitoring and fusion system.  Two
counts matter and they are kept distinct everywhere in this module:

  * **Registered scalars** (~209): every omni-scalar discoverable via
    ``scalar_groups``.  Includes diagnostic measurement families that
    describe code/system *under analysis* but do not drive the gate.
  * **Operational scalars** (127): the subset that participates in the
    σ_Immutable input vector and every fusion / aggregation that the
    boundary's decision depends on.  ``_is_metric_only_scalar`` is the
    single source of truth that separates the two — see the class-
    level ``_METRIC_ONLY_PREFIXES`` block for the contract.

Eight major categories:

- ETHICAL (~27 scalars): Core ethical values and operational constraints
- COSMIC (~7 scalars): Universe-scale harmony and telos alignment
- QUANTUM_CONSCIOUSNESS (~7 scalars): Quantum-inspired processing
- HUMANITARIAN (~9 scalars): Crisis response and human welfare
- SECURITY (~6 scalars): Threat detection and cyber defense
- SOFTWARE_ENGINEERING (~127 scalars: 45 operational + 82 diagnostic):
  Code quality, optimization, 3R synergy, plus ISO/IEC 25010 product
  quality, Halstead, McCabe/cognitive, MI variants, NIST SAMATE
  assurance, DORA delivery, SLSA supply-chain, supply-chain /
  repository-integrity checks (Mercury-native), ISO/IEC 5055 CISQ
  measures, and NIST SSDF (SP 800-218) practices
- MEDICAL (~10 scalars): Healthcare and diagnostic support
- ADVANCED_REASONING (~16 scalars): Logic, inference, and knowledge synthesis

Key Features:
- 37-dimensional quantum fusion with 32-head attention
- Ethical gating with sigma_Immutable threshold enforcement
- Component-based scalar registration and enhancement
- Global intelligence score computation
- Triadic harmony computation using golden ratio (phi = 1.618)
- Bidirectional synaptic integration with 3R mechanism

The GOSNN serves as a central hub for aggregating insights from multiple
specialized engines and maintaining system-wide ethical alignment.

References:
    - Multi-head attention: Vaswani et al. (2017) "Attention Is All You Need"
    - Golden ratio applications: Livio (2002) "The Golden Ratio"
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from omni_mercury_engine.security.safe_torch import safe_torch_load

if TYPE_CHECKING:
    from collections.abc import Iterator

# P2: Import from centralized constants
from omni_mercury_engine.core.centralized_constants import (
    ETHICAL,
    LYAPUNOV,
    MATH,
)

# Golden ratio constant for triadic harmony and phi-weighting
# P2: Now references centralized constant
PHI: float = MATH.GOLDEN_RATIO

# Lyapunov stability constant (elevated from 0.18; see benchmarks/ab_dominance.py
# for the measured comparison)
# P2: Now references centralized constant
LAMBDA_LYAPUNOV: float = LYAPUNOV.LAMBDA_CONVERGENCE

# Sigma Immutable thresholds for ethical gating (Civilization-First principle)
# P2: Now references centralized constants
SIGMA_IMMUTABLE_DEFAULT: float = ETHICAL.SIGMA_IMMUTABLE_DEFAULT
SIGMA_IMMUTABLE_MEDICAL: float = ETHICAL.SIGMA_IMMUTABLE_MEDICAL

try:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment, unused-ignore]
    nn = None  # type: ignore[assignment, unused-ignore]


class ScalarGroup(Enum):
    """Thematic groups for omni-scalars (~209 registered; 127 operational)."""

    # Core categories — counts reflect registered entries.  The operational
    # subset (what enters the σ_Immutable input vector) is the registered
    # count minus the diagnostic measurement scalars listed in
    # ``GlobalOmniScalarNetwork._METRIC_ONLY_PREFIXES`` / ``_METRIC_ONLY_KEYS``.
    ETHICAL = "ethical"  # ~27 scalars (all operational)
    COSMIC = "cosmic"  # ~7 scalars (all operational)
    QUANTUM_CONSCIOUSNESS = "quantum_consciousness"  # ~7 scalars (all operational)
    HUMANITARIAN = "humanitarian"  # ~9 scalars (all operational)
    SECURITY = "security"  # ~6 scalars (all operational)
    SOFTWARE_ENGINEERING = (
        "software_engineering"  # ~127 scalars (45 operational + 82 diagnostic measurement)
    )
    MEDICAL = "medical"  # ~10 scalars (all operational)
    ADVANCED_REASONING = "advanced_reasoning"  # ~16 scalars (all operational)

    # Legacy/specialized categories (for backward compatibility)
    MATHEMATICAL_MYSTERIES = "mathematical_mysteries"
    PARADOX_DEFENSE = "paradox_defense"
    PHYSICS_THEORIES = "physics_theories"
    SUSTAINABILITY = "sustainability"
    CRISIS_RESPONSE = "crisis_response"
    AI_GUARDIAN = "ai_guardian"
    PERFORMANCE = "performance"


@dataclass
class ScalarRegistration:
    """Registration record for component scalars."""

    component_name: str
    scalars: dict[str, float]
    group: ScalarGroup
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnhancementResult:
    """Result of scalar enhancement operation.

    ``fused_state`` is the raw 37-dimensional attention-fusion output the
    other fields are derived from.  It is carried so consumers with a
    merit-gated use for the full vector (the GOSNN detection head feeding
    the decision layer's disagreement overlay) do not have to re-run the
    fusion; ``None`` when the enhancement ran without a fusion pass.

    ``collected_scalars`` is the exact operational-scalar snapshot this
    enhancement evaluated its (advisory) ethical gate against.  It is
    carried so the caller's authoritative σ_Immutable gate reuses the
    *identical* snapshot instead of taking a second, independent collection
    -- both a per-call efficiency saving (one 127-scalar registry walk
    under lock instead of two) and a signal-integrity guarantee: the
    advisory and authoritative gates can never evaluate divergent vectors
    (a latent race when they collected separately).  ``None`` when the
    enhancement ran without collecting.
    """

    enhanced_scalars: dict[str, float]
    fusion_score: float
    ethical_gate_passed: bool
    intelligence_contribution: float
    warnings: list[str] = field(default_factory=list)
    fused_state: np.ndarray[Any, Any] | None = None
    collected_scalars: dict[str, float] | None = None


class EthicalGate:
    """Trained neural network gate for ethical compliance verification.

    Architecture: ``Linear(256, 64) → ReLU → Linear(64, 1) → Sigmoid``.

    The gate is trained by ``scripts/train_sigma_immutable.py`` on a
    labelled scalar-vector corpus (see that script's docstring for the
    labelling source).  Trained weights are persisted at
    ``security/sigma_immutable_weights.pt`` and loaded at construction
    time.  If the weights file is absent or PyTorch is unavailable, the
    gate falls back to a deterministic NumPy heuristic.

    The gate is a *second independent check* alongside
    :class:`~omni_mercury_engine.cognitive.ethical_bounding.BenevolenceScorer`.
    """

    _WEIGHTS_RELPATH = "security/sigma_immutable_weights.pt"

    def __init__(
        self,
        input_dim: int = 256,
        threshold: float = ETHICAL.SIGMA_IMMUTABLE_TRAINED_THRESHOLD,
    ) -> None:
        """Initialize the instance."""
        self.threshold = threshold
        self.input_dim = input_dim
        self.logger = logging.getLogger(__name__)
        self._trained = False

        if TORCH_AVAILABLE:
            self.gate_network = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )
            self._try_load_trained_weights()
        else:
            self.gate_network = None  # type: ignore[assignment, unused-ignore]

    def _try_load_trained_weights(self) -> None:
        """Load trained weights from the in-repo artifact if present."""
        # Resolve weights path relative to the package root
        pkg_root = Path(__file__).resolve().parent.parent
        weights_path = pkg_root / self._WEIGHTS_RELPATH
        if not weights_path.exists():
            self.logger.debug(
                "σ_Immutable weights not found at %s; using untrained gate",
                weights_path,
            )
            return
        try:
            state_dict = safe_torch_load(weights_path, map_location="cpu")
            self.gate_network.load_state_dict(state_dict)
            self.gate_network.eval()
            self._trained = True
            self.logger.info("σ_Immutable: loaded trained weights from %s", weights_path)
        except Exception as exc:
            self.logger.warning("σ_Immutable: failed to load weights: %s", exc)

    def evaluate(self, scalar_vector: np.ndarray[Any, Any]) -> tuple[bool, float]:
        """Evaluate ethical compliance of scalar vector.

        Args:
            scalar_vector: Input scalar values

        Returns:
            Tuple of (passes_gate, ethical_score)
        """
        if not np.all(np.isfinite(scalar_vector)):
            # Fail closed: a non-finite scalar (NaN / ±inf) means an
            # upstream computation broke.  The previous behaviour coerced
            # NaN->0 and scored anyway, which let a NaN-collapsed ethical
            # dim score above threshold as PASS — a fail-open hole.  An
            # unscoreable vector is an unsafe vector.
            self.logger.warning("non-finite value in scalar_vector; failing closed (score=0.0)")
            return False, 0.0

        if self.gate_network is not None and TORCH_AVAILABLE and self._trained:
            padded = np.zeros(self.input_dim)
            padded[: min(len(scalar_vector), self.input_dim)] = scalar_vector[: self.input_dim]

            with torch.no_grad():
                tensor_input = torch.tensor(padded, dtype=torch.float32).unsqueeze(0)
                score = self.gate_network(tensor_input).item()
        else:
            score = self._compute_ethical_score_numpy(scalar_vector)

        passes = score >= self.threshold
        return passes, score

    def _compute_ethical_score_numpy(self, scalar_vector: np.ndarray[Any, Any]) -> float:
        """Compute ethical score using NumPy fallback."""
        if len(scalar_vector) == 0:
            return 0.5

        positive_ratio = np.sum(scalar_vector > 1.0) / len(scalar_vector)
        mean_value = np.mean(scalar_vector)
        std_value = np.std(scalar_vector)

        score = (
            0.4 * positive_ratio
            + 0.4 * min(mean_value / 2.0, 1.0)
            + 0.2 * (1.0 / (1.0 + std_value))
        )
        return float(np.clip(score, 0.0, 1.0))


class TriadicPhiWeighting:
    """Triadic phi-weighting layer for harmonic synergy in attention fusion.

    Applies golden ratio (phi = 1.618) weighting to query-key-value attention
    scores for coherent frequency patterns in Resonance (H(omega) harmonics).

    The triadic structure groups attention heads into three bands:
    - Band 1 (Query-dominant): Weighted by phi
    - Band 2 (Key-dominant): Weighted by 1.0
    - Band 3 (Value-dominant): Weighted by 1/phi

    This creates harmonic synergy through mathematically grounded frequency
    coherence, not arbitrary scaling.
    """

    def __init__(self, num_heads: int = 32) -> None:
        """Initialize triadic phi-weighting.

        Args:
            num_heads: Number of attention heads (should be divisible by 3 for
                       optimal triadic grouping, but handles any count)
        """
        self.num_heads = num_heads
        self.phi = PHI
        self.phi_inverse = 1.0 / PHI

        # Compute triadic weights for each head
        self.head_weights = self._compute_triadic_weights()

    def _compute_triadic_weights(self) -> np.ndarray[Any, Any]:
        """Compute phi-based weights for each attention head."""
        weights = np.ones(self.num_heads)
        heads_per_band = self.num_heads // 3

        # Band 1: Query-dominant (phi weighting)
        weights[:heads_per_band] = self.phi

        # Band 2: Key-dominant (unity weighting)
        weights[heads_per_band : 2 * heads_per_band] = 1.0

        # Band 3: Value-dominant (1/phi weighting)
        weights[2 * heads_per_band :] = self.phi_inverse

        # Normalize to sum to num_heads for stable gradients
        normalized_weights: np.ndarray[Any, Any] = np.empty(weights.shape, dtype=np.float64)
        np.multiply(weights, self.num_heads / np.sum(weights), out=normalized_weights)
        weights = normalized_weights

        return weights

    def apply(self, attention_scores: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Apply triadic phi-weighting to attention scores.

        Args:
            attention_scores: Raw attention scores [num_heads, seq_len, seq_len]
                              or [batch, num_heads, seq_len, seq_len]

        Returns:
            Phi-weighted attention scores with harmonic synergy
        """
        if attention_scores.ndim == 3:
            # [num_heads, seq_len, seq_len]
            weighted = attention_scores * self.head_weights[:, np.newaxis, np.newaxis]
        elif attention_scores.ndim == 4:
            # [batch, num_heads, seq_len, seq_len]
            weighted = attention_scores * self.head_weights[np.newaxis, :, np.newaxis, np.newaxis]
        else:
            # Fallback: apply mean weight
            weighted = attention_scores * np.mean(self.head_weights)

        return np.asarray(weighted)  # type: ignore[no-any-return, unused-ignore]

    def compute_harmonic_synergy(self, attention_output: np.ndarray[Any, Any]) -> float:
        """Compute harmonic synergy score from attention output.

        The synergy score measures how well the triadic weighting produces
        coherent frequency patterns (H(omega) in the weighted fusion Equation).

        **Diagnostic only — known degeneracy.**  For real-valued input the
        FFT magnitude spectrum is conjugate-symmetric (``|X[k]| == |X[N-k]|``),
        so whenever the largest magnitude sits in a non-DC, non-Nyquist bin
        the top two sorted magnitudes are an equal mirror pair and the ratio
        is exactly 1.0, pinning the score to ``1/(1+|1-phi|) = 0.618034``.
        Measured on the production serve path (2026-07-17): the trained
        fusion's near-zero-mean attention output hit this degeneracy on 15/15
        real detect calls (bit-identical 0.618034005), and only DC-dominant
        inputs (a large mean level, e.g. raw stacked member states) move the
        value at all — a scale artefact, not harmonic structure.  The score
        is therefore not surfaced in ``detect_with_fusion`` results and must
        not be used to modulate computation (the historical serve-path
        modulation it fed was a constant ×1.0118 rescale and has been
        removed).

        Args:
            attention_output: Output from attention mechanism

        Returns:
            Harmonic synergy score (0-1)
        """
        if attention_output.size == 0:
            return 0.5

        # Compute FFT to analyze frequency coherence
        fft_result = np.fft.fft(attention_output.flatten())
        magnitudes = np.abs(fft_result)

        # Harmonic synergy is high when dominant frequencies align with phi ratios
        if len(magnitudes) > 1:
            sorted_mags = np.sort(magnitudes)[::-1]
            if sorted_mags[1] > 0:
                ratio = sorted_mags[0] / sorted_mags[1]
                # Score based on proximity to phi
                synergy = 1.0 / (1.0 + abs(ratio - self.phi))
            else:
                synergy = 0.5
        else:
            synergy = 0.5

        return float(np.clip(synergy, 0.0, 1.0))


class MultiHeadAttentionFusion:
    """Multi-head attention mechanism for 37D quantum fusion.

    Implements configurable attention (default 32-head at d_model=512, head_dim=16)
    with triadic phi-weighting for harmonic synergy in scalar dimension fusion.

    The triadic phi-weighting applies golden ratio (phi = 1.618) scaling to
    attention scores, creating coherent frequency patterns that enhance the
    H(omega) component of the weighted fusion Equation.

    Inference contract: the torch attention path runs **only when trained
    weights have been loaded** via :meth:`load_trained_weights` (the
    ``EthicalGate`` convention).  Until then, :meth:`fuse` uses the
    deterministic phi-weighted reference average for every environment —
    with or without torch.  Historically the torch path ran with its random
    initialisation (a fixed random projection followed by averaging
    masquerading as learned attention); that placeholder inference is no
    longer reachable.  (Mirrors the untrained-network disclosure pattern in
    docs/NEUROSYMBOLIC.md and the abstention pattern in
    ``models/parapsychology.py``.)
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 32,
        max_dimensions: int = 37,
        enable_triadic_phi: bool = True,
        load_shipped_weights: bool = True,
    ):
        """Initialize multi-head attention fusion.

        Args:
            d_model: Model dimension (default 512)
            num_heads: Number of attention heads (default 32 for head_dim=16)
            max_dimensions: Maximum dimensions for fusion (default 37)
            enable_triadic_phi: Enable triadic phi-weighting for harmonic synergy
            load_shipped_weights: Load the shipped merit-gated
                ``gosnn_attention_fusion`` checkpoint at construction when it
                exists (trained by ``scripts/train_gosnn_fusion.py`` on real
                harvested production state lists and gated against the
                phi-weighted reference). Pass False to pin the deterministic
                reference path. Absence falls open to the reference; an
                invalid checkpoint fails loud.
        """
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_dimensions = max_dimensions
        self.head_dim = d_model // num_heads
        self.enable_triadic_phi = enable_triadic_phi
        self.logger = logging.getLogger(__name__)
        # Learned path activates only after load_trained_weights() succeeds.
        self._trained = False
        # One-time disclosure that fusion is running the deterministic
        # phi-weighted reference (no trained attention weights loaded).
        self._disclosed_reference_path = False
        # Optional merit-gated detection head over the fused state (ships in
        # the checkpoint payload only when the detection-metric gate passed;
        # None => the fused state stays observability-only).
        self.detection_head: Any | None = None
        # Validation-selected disagreement-demotion thresholds shipped with
        # the head ({"demote_act_below": ..., "demote_clear_above": ...}).
        self.decision_thresholds: dict[str, float] | None = None

        # Triadic phi-weighting for harmonic synergy
        self.triadic_weighting = TriadicPhiWeighting(num_heads) if enable_triadic_phi else None

        if TORCH_AVAILABLE:
            # The architecture is defined once, in attention_fusion_stack —
            # the same builder the training program uses — so train and
            # serve cannot silently diverge.
            from omni_mercury_engine.core.attention_fusion_stack import build_fusion_modules

            self.projection, self.attention, self.output_projection = build_fusion_modules(
                d_model=d_model, num_heads=num_heads, max_dimensions=max_dimensions
            )
            if load_shipped_weights:
                self._try_load_shipped_weights()
        else:
            self.attention = None  # type: ignore[assignment, unused-ignore]
            self.projection = None  # type: ignore[assignment, unused-ignore]
            self.output_projection = None  # type: ignore[assignment, unused-ignore]

    def _try_load_shipped_weights(self) -> None:
        """Load the shipped merit-gated checkpoint if one exists.

        The ``gosnn_attention_fusion`` checkpoint ships only when
        ``scripts/train_gosnn_fusion.py``'s gate measured the learned fusion
        beating the phi-weighted reference on held-out production-harvested
        state lists.  Absence keeps the deterministic reference (fail-open);
        a present-but-invalid payload fails loud in
        :meth:`load_trained_weights` (integrity is a supply-chain control,
        not a fallback case).
        """
        try:
            from omni_mercury_engine.models.checkpoint_paths import load_shipped_checkpoint

            payload, _provenance = load_shipped_checkpoint("gosnn_attention_fusion")
        except FileNotFoundError:
            self.logger.debug(
                "No shipped gosnn_attention_fusion checkpoint; using the "
                "deterministic phi-weighted reference fusion."
            )
            return
        self.load_trained_weights(payload)

    def fuse(
        self, dimensional_states: list[np.ndarray[Any, Any]], return_synergy: bool = False
    ) -> np.ndarray[Any, Any] | tuple[np.ndarray[Any, Any], float]:
        """Fuse multiple dimensional states using multi-head attention with triadic phi-weighting.

        Args:
            dimensional_states: List of state vectors to fuse
            return_synergy: If True, also return harmonic synergy score

        Returns:
            Fused state vector, optionally with harmonic synergy score
        """
        if not dimensional_states:
            result = np.zeros(self.max_dimensions)
            return (result, 0.5) if return_synergy else result

        padded_states = []
        for state in dimensional_states:
            padded = np.zeros(self.max_dimensions)
            padded[: min(len(state), self.max_dimensions)] = state[: self.max_dimensions]
            padded_states.append(padded)

        stacked = np.stack(padded_states)
        harmonic_synergy = 0.5

        if self.attention is not None and TORCH_AVAILABLE and self._trained:
            from omni_mercury_engine.core.attention_fusion_stack import fuse_members

            with torch.no_grad():
                tensor_input = torch.tensor(stacked, dtype=torch.float32)
                # The shared canonical stack — identical bytes to what the
                # training program optimises and the merit gate measures.
                # The historical serve-only synergy modulation
                # (attn_output * (1 + 0.1*(synergy-0.5))) is gone: the gate
                # never measured it, and on production states the synergy
                # term was a constant (see compute_harmonic_synergy), so it
                # amounted to an undisclosed constant rescale of the
                # gate-verified computation.
                result = fuse_members(
                    self.projection, self.attention, self.output_projection, tensor_input
                ).numpy()
            if self.triadic_weighting is not None:
                # Diagnostic only, computed on the same member stack the
                # reference branch uses (never modulates the fused output).
                harmonic_synergy = self.triadic_weighting.compute_harmonic_synergy(stacked)
        else:
            # Deterministic phi-weighted reference average — the documented
            # inference behaviour whenever no trained attention weights are
            # loaded (and the only behaviour when torch is absent).  The
            # historical alternative — running the attention modules with
            # their random initialisation — presented a fixed random
            # projection as learned fusion and is deliberately unreachable.
            if not self._disclosed_reference_path and self.attention is not None:
                self.logger.info(
                    "MultiHeadAttentionFusion: no trained attention weights "
                    "loaded; using the deterministic phi-weighted reference "
                    "average (call load_trained_weights() to activate the "
                    "learned path)."
                )
                self._disclosed_reference_path = True
            weights = np.ones(len(padded_states)) / len(padded_states)
            if self.triadic_weighting is not None:
                # Apply phi-based weighting to state averaging
                phi_weights = np.array([PHI, 1.0, 1.0 / PHI])
                phi_weights = np.tile(phi_weights, len(padded_states) // 3 + 1)[
                    : len(padded_states)
                ]
                phi_weights = phi_weights / np.sum(phi_weights)
                weights = phi_weights
                harmonic_synergy = self.triadic_weighting.compute_harmonic_synergy(stacked)

            result = np.average(stacked, axis=0, weights=weights)

        return (result, harmonic_synergy) if return_synergy else result

    def load_trained_weights(self, source: str | Path | dict[str, Any]) -> None:
        """Load trained attention weights and activate the learned fusion path.

        Follows the ``EthicalGate`` trained-weights convention: the learned
        path is inert until a genuine checkpoint is supplied.  The payload
        must carry the three module state dicts produced by a training run
        of this exact architecture.

        Args:
            source: Path to a ``torch.save`` payload, or the already-loaded
                payload dict, with keys ``"attention"``, ``"projection"`` and
                ``"output_projection"``.  A payload may additionally carry a
                ``"detection_head"`` state dict plus ``"decision_thresholds"``
                (both shipped only when the detection-metric merit gate
                passed); loading them activates :meth:`detection_probability`.

        Raises:
            RuntimeError: If torch is unavailable (no learned path exists), or
                if a ``detection_head`` is present without valid
                ``decision_thresholds`` (a consequential head without its
                validated operating points must fail loud, not serve with
                made-up thresholds).
            KeyError: If the payload is missing a required module state dict.
        """
        if not TORCH_AVAILABLE or self.attention is None:
            raise RuntimeError(
                "MultiHeadAttentionFusion.load_trained_weights requires torch; "
                "the deterministic phi-weighted reference is the only path "
                "available without it"
            )
        payload: dict[str, Any]
        if isinstance(source, (str, Path)):
            payload = safe_torch_load(source, map_location="cpu")
        else:
            payload = source
        self.projection.load_state_dict(payload["projection"])
        self.attention.load_state_dict(payload["attention"])
        self.output_projection.load_state_dict(payload["output_projection"])
        self.projection.eval()
        self.attention.eval()
        self.output_projection.eval()
        self._trained = True
        if "detection_head" in payload:
            from omni_mercury_engine.core.attention_fusion_stack import FusionDetectionHead

            head = FusionDetectionHead(max_dimensions=self.max_dimensions)
            head.load_state_dict(payload["detection_head"])
            head.eval()
            thresholds = payload.get("decision_thresholds")
            if not isinstance(thresholds, dict) or not {
                "demote_act_below",
                "demote_clear_above",
            } <= set(thresholds):
                raise RuntimeError(
                    "gosnn_attention_fusion payload carries a detection_head "
                    "without valid decision_thresholds (demote_act_below / "
                    "demote_clear_above); refusing to serve a consequential "
                    "head without its validation-selected operating points"
                )
            act_below = float(thresholds["demote_act_below"])
            clear_above = float(thresholds["demote_clear_above"])
            # Range/order validation: out-of-range or inverted thresholds
            # turn the disagreement overlay degenerate (e.g. clear_above=0.0
            # demotes every grounded negative). A consequential head serves
            # only with sane operating points -- anything else fails loud.
            if not (0.0 <= act_below < clear_above <= 1.0):
                raise RuntimeError(
                    "gosnn_attention_fusion decision_thresholds are invalid: "
                    f"require 0 <= demote_act_below ({act_below}) < "
                    f"demote_clear_above ({clear_above}) <= 1; refusing to "
                    "serve a degenerate demotion rule"
                )
            self.detection_head = head
            self.decision_thresholds = {
                "demote_act_below": act_below,
                "demote_clear_above": clear_above,
            }
        self.logger.info("MultiHeadAttentionFusion: trained attention weights loaded")

    def detection_probability(self, fused_state: np.ndarray[Any, Any]) -> float | None:
        """Anomaly probability from the merit-gated detection head, if shipped.

        The head exists only when ``scripts/train_gosnn_fusion.py``'s
        detection-metric gate measured the learned fused-state detector
        beating both the phi-reference fusion and the mean-detector-score
        baseline on held-out labelled ADBench detections.  Without a shipped
        head this returns ``None`` and the fused state stays
        observability-only.

        Args:
            fused_state: The 37-dimensional fused state vector.

        Returns:
            ``P(anomaly)`` in ``[0, 1]``, or ``None`` when no head is loaded
            (or torch is unavailable).
        """
        if self.detection_head is None or not TORCH_AVAILABLE:
            return None
        with torch.no_grad():
            logit = self.detection_head(torch.tensor(fused_state, dtype=torch.float32))
            return float(torch.sigmoid(logit).item())


def get_sigma_immutable_threshold(domain: str | None = None) -> float:
    """Get the sigma_Immutable threshold for ethical gating (Civilization-First principle).

    The threshold can be configured via environment variable SIGMA_IMMUTABLE_THRESHOLD.
    Default is 0.96 for stricter ethical gating (~10-15% false positive reduction).
    Medical domains use 0.93 fallback to avoid false negatives in critical scenarios.

    The sigma_Immutable threshold represents an inviolable ethical constraint that
    cannot be overridden, ensuring Civilization-First principles are maintained.

    Args:
        domain: Optional domain identifier (e.g., "medical", "security", "humanitarian")

    Returns:
        sigma_Immutable threshold value (0.93-0.96)
    """
    # Medical domains use lower threshold to avoid false negatives
    MEDICAL_DOMAINS = {"medical", "healthcare", "clinical", "diagnostic", "patient"}

    if domain and domain.lower() in MEDICAL_DOMAINS:
        return SIGMA_IMMUTABLE_MEDICAL

    # Check environment variable for custom threshold
    env_threshold = os.environ.get("SIGMA_IMMUTABLE_THRESHOLD")

    if env_threshold:
        try:
            threshold = float(env_threshold)
            # Clamp to valid range; the hard minimum is the trained
            # network's calibrated decision threshold (authoritative
            # source: ETHICAL.SIGMA_IMMUTABLE_TRAINED_THRESHOLD = 0.93).
            return max(ETHICAL.SIGMA_IMMUTABLE_TRAINED_THRESHOLD, min(0.99, threshold))
        except ValueError:
            # Invalid threshold value in environment; use default
            pass

    # Default elevated threshold for precision dominance
    return SIGMA_IMMUTABLE_DEFAULT


_SW_ENG_METRICS_PATH = Path(__file__).resolve().parent / "sw_eng_metrics.json"


def _apply_measured_sw_eng_metrics(group: dict[str, float]) -> int:
    """Overlay real measured values onto existing diagnostic SW-eng scalars.

    Loads the artifact produced by ``scripts/collect_sw_eng_metrics.py`` (real
    Halstead / cyclomatic / Maintainability-Index measurements over
    ``src/omni_mercury_engine`` plus Mercury-native supply-chain /
    repository-integrity checks handwritten from repo config) and
    updates **only keys already present** in ``group`` — so the group's
    cardinality and the frozen σ_Immutable operational layout are unchanged, and
    every updated scalar stays metric-only (filtered from the gate) by its
    ``omni_halstead_`` / ``omni_mccabe_`` / ``omni_ossf_`` prefix or its
    ``_METRIC_ONLY_KEYS`` membership.  A missing or malformed artifact is a
    silent no-op — the static placeholders stand — so the engine never depends
    on the collector having run.

    Returns:
        The number of scalars updated with a measured value (0 if the artifact
        is absent).
    """
    import json  # local: json is only needed on this build-time-populated path

    try:
        payload = json.loads(_SW_ENG_METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    if not isinstance(payload, dict) or payload.get("schema") != "sw_eng_metrics/v1":
        return 0
    updated = 0
    for name, value in payload.get("scalars", {}).items():
        if name in group:
            try:
                group[name] = float(value)
                updated += 1
            except (TypeError, ValueError):
                continue
    return updated


class GlobalOmniScalarNetwork:
    """Global Omni-Scalar Network (GOSNN) - Central Intelligence Fusion Hub.

    Registers ~209 omni-scalars across 8 major categories; 127 of them
    are *operational* and feed the σ_Immutable gate / fusion path.  The
    remaining 82 are diagnostic measurement scalars (descriptions of
    code/system under analysis) that live in ``scalar_groups`` for
    discoverability and reporting but are filtered out of the gate's
    input vector by ``_is_metric_only_scalar``.

    Registered breakdown (operational + diagnostic):
    - ETHICAL (~27): Core ethical values and Civilization-First principles
    - COSMIC (~7): Universe-scale harmony and telos alignment
    - QUANTUM_CONSCIOUSNESS (~7): Quantum-inspired processing
    - HUMANITARIAN (~9): Crisis response and human welfare
    - SECURITY (~6): Threat detection and cyber defense
    - SOFTWARE_ENGINEERING (~127 = 45 op + 82 diag): Code quality, optimization,
      3R synergy; plus ISO/IEC 25010, Halstead, McCabe/cognitive, MI variants,
      NIST SAMATE, DORA, SLSA, supply-chain integrity, ISO/IEC 5055, NIST SSDF
    - MEDICAL (~10): Healthcare and diagnostic support
    - ADVANCED_REASONING (~16): Logic, inference, knowledge synthesis

    Key Features:
    - 37D quantum fusion with 32-head attention and triadic phi-weighting
    - Ethical gating with configurable σ_Immutable threshold (0.96 default, 0.93 for medical)
    - Component-based scalar registration
    - Global intelligence score computation
    - Triadic harmony using golden ratio (φ = 1.618)
    - Bidirectional synaptic integration with 3R mechanism

    The σ_Immutable threshold can be configured via SIGMA_IMMUTABLE_THRESHOLD
    environment variable. Default is 0.96 for ~10-15% false positive reduction
    via stricter ethical gating. Medical domains automatically use 0.93 fallback.

    This is implemented as a singleton to ensure consistent global state.
    """

    _instance: GlobalOmniScalarNetwork | None = None
    _lock = threading.Lock()

    # Class constants
    PHI = PHI  # Use module-level constant
    # Cite the single authoritative source (centralized_constants.ETHICAL)
    # instead of re-hardcoding the values here.
    SIGMA_IMMUTABLE_DEFAULT = ETHICAL.SIGMA_IMMUTABLE_DEFAULT
    SIGMA_IMMUTABLE_MEDICAL = ETHICAL.SIGMA_IMMUTABLE_MEDICAL
    MIN_EMPATHY = 1.22
    MIN_MORALITY = 1.20
    TARGET_BOOST_RATIO = 0.60

    # ------------------------------------------------------------------
    # Narrative/personality tuning scalars that live in the ETHICAL group
    # but are NOT ethical safety floors.
    #
    # These three carry the narrative engine's output-style knobs (see
    # ``narrative/personality.py`` and ``narrative/engine.py``): how much
    # reasoning to expose, how verbose to make explanations, how forgiving
    # the persona reads.  Their calibrated defaults are intentionally low
    # (forgiveness 0.10, transparency 0.18, explainability 0.90) precisely
    # because they tune *tone*, not ethics.  The σ_Immutable trainer's
    # labelling rule ("all 27 ETHICAL dims >= 0.93") wrongly treats them
    # as hard floors, which is why the production-default vector — which
    # the engine actually feeds the gate — sits below that rule yet must
    # operate.  The deterministic critical-ethical floor
    # (``SigmaImmutableGate.enforce_ethical_floor``) therefore excludes
    # them by name: the trained network's soft score still sees them, but
    # they cannot trip (or, being low, falsely trip) the hard floor.
    # ------------------------------------------------------------------
    _NARRATIVE_ETHICAL_SCALARS: frozenset[str] = frozenset(
        {
            "omniforgiveness",
            "omnitransparency",
            "omniexplainability",
        }
    )

    # ------------------------------------------------------------------
    # Diagnostic measurement scalars excluded from the σ_Immutable input
    # vector and from every operational aggregation (fusion, hierarchical
    # accountability score, dimensional-state preparation).
    #
    # The σ_Immutable trained gate (see
    # ``security/sigma_immutable_gate.py``) was trained on a fixed
    # 256-D layout where ``[SIGMA_USED_BAND_END, SIGMA_IMMUTABLE_DIM)``
    # is zero-padding by contract.  ``_collect_all_scalars`` produces
    # the vector that gets fed to the gate, so growing it past
    # ``SIGMA_USED_BAND_END=180`` would overflow into the zero-padded
    # tail and the trained network would interpret the leak as a
    # poisoned input.
    #
    # The ISO/IEC 25010, Halstead, McCabe + cognitive, Maintainability
    # Index variants, NIST SAMATE, DORA delivery, SLSA supply-chain,
    # supply-chain / repository-integrity, ISO/IEC 5055 (CISQ), and NIST
    # SSDF practice families are diagnostic measurement scalars
    # (descriptions of code
    # / system under analysis), not operational ethical signals that
    # drive the boundary's decision.  They remain in ``scalar_groups``
    # for discoverability, registration, and downstream reporting; they
    # are filtered out of every operational path by the single helper
    # ``_is_metric_only_scalar`` so the σ_Immutable layout contract,
    # the hierarchical accountability bucket's semantics, and the
    # fusion pipeline's dimensional layout stay consistent.
    #
    # HONESTY NOTE: of these ~82 diagnostic scalars, 36 now carry REAL
    # measurements collected by ``scripts/collect_sw_eng_metrics.py`` into
    # ``core/sw_eng_metrics.json`` and overlaid at init by
    # ``_apply_measured_sw_eng_metrics`` (updates existing keys only, so they
    # stay metric-only and out of the [0, 180) operational σ band):
    #   * source-tree code metrics (14): Halstead suite (7), cyclomatic (1),
    #     Maintainability Index (3) via stdlib ``ast``, plus the 3 MI variants;
    #   * supply-chain / repo-integrity checks (10) handwritten from repo config;
    #   * DORA delivery metrics (4) as honest VCS-history proxies from git log
    #     (commit cadence, inter-commit lead time, revert fraction, revert MTTR)
    #     — proxies, NOT production deploy/incident telemetry;
    #   * NIST SSDF practice groups (4) and SLSA build-track evidence (4) from
    #     repo state (policy/toolchain/pinning/provenance/SBOM);
    #   * the SAMATE subset computable offline (3): supply-chain assurance,
    #     assurance-evidence completeness, and residual risk (active accepted-CVE
    #     count from the ``.trivyignore`` ledger).
    # The remaining ~46 (the 31 ISO/IEC 25010 quality characteristics, the 7
    # SAMATE scalars needing the external SAMATE Reference Dataset / labelled
    # ground truth, the 4 ISO/IEC 5055 measures, and the essential/design/
    # cognitive/npath complexity variants) stay STATIC placeholder literals —
    # computing them would be fabrication or requires an external conformant
    # analyzer — and remain registered for naming/reporting only.
    #
    # Adding a new measurement family means updating these two
    # allowlists once; no other call site needs to change.
    # ------------------------------------------------------------------
    _METRIC_ONLY_PREFIXES: tuple[str, ...] = (
        # Generic diagnostic channel.  Raw measurements that are useful for
        # reporting but must NEVER enter the σ_Immutable operational vector
        # (unix timestamps, unbounded counters, identifiers) are registered
        # under this prefix.  The MercuryGuardianAdapter's posture-rotation /
        # algorithm-switch timestamps (~1.7e9 unix seconds) were previously
        # registered as plain SECURITY scalars and leaked into the gate input
        # of unrelated ``detect_with_fusion`` calls later in the same process,
        # collapsing the ethical score to 0.0 (defect F10).
        "omni_diag_",
        # Software-engineering diagnostic measurement families.
        "omni_iso25010_",
        "omni_halstead_",
        "omni_mccabe_",
        "omni_samate_",
        "omni_dora_",
        "omni_slsa_",
        "omni_ossf_",
        "omni_iso5055_",
        "omni_ssdf_",
        # Governance / medical / AI-assurance families (see omni_mercury_engine.governance).
        # Descriptive and three-state (GROUNDED/UNAVAILABLE/UNDECIDABLE): registered for
        # reporting, filtered out of the σ_Immutable operational vector exactly like the SE
        # families above.  Only families the per-family signal vet keeps appear here; dropped
        # UNDECIDABLE families (e.g. OWASP LLM Top 10) carry no prefix.
        "omni_sofa_",  # SOFA organ sub-scores + total
        "omni_ews_",  # NEWS2 + MEWS early-warning aggregates
        "omni_meld_",  # MELD-Na hepatic allocation score
        "omni_iso14971_",  # ISO 14971 medical-device risk index
        "omni_nist_airmf_",  # NIST AI RMF MEASURE conformance
        "omni_mitre_atlas_",  # MITRE ATLAS observed-tactic coverage
    )
    _METRIC_ONLY_KEYS: frozenset[str] = frozenset(
        {
            "omni_cognitive_complexity_sonar",
            "omni_npath_complexity",
            "omni_maintainability_index_sei",
            "omni_maintainability_index_vs",
            "omni_maintainability_index_delta",
        }
    )

    # ------------------------------------------------------------------
    # Gross-outlier tripwire for the operational scalar band.
    #
    # Expected range of an operational scalar (determined from the σ gate
    # input handling, not invented): the trained σ_Immutable network was
    # trained on ``U[0, 2]`` support (``security/sigma_immutable_gate.py``,
    # SIGMA_IMMUTABLE_PERMISSIBLE_HIGH == 2.0), the NumPy fallback and
    # ``compute_hierarchical_score`` both treat 2.0 as full scale
    # (``min(mean / 2.0, 1.0)`` / ``clip(arr / 2.0, 0, 1)``), and every
    # default scalar sits in ``[0.1, PHI]``.  Some legitimate component
    # registrations run hotter — posture action codes reach 4.0
    # (``integrations/mercury_amacrypto.ACTION_MAP``) and bounded anomaly
    # counters reach ~1e3 — so this limit is deliberately a *gross* bound,
    # not a strict range validator: three orders of magnitude above the
    # hottest legitimate registration and three below the smallest unit
    # error actually seen corrupting the gate (a raw unix timestamp,
    # ~1.7e9 seconds).  A finite operational scalar with ``|value|`` above
    # this limit cannot be a normalized signal; fed to the trained gate it
    # saturates the first linear layer and collapses the ethical score to
    # 0.0, which forced spurious fail-closed
    # ``EthicalConstraintViolationError``s on unrelated requests (F10).
    # Such scalars are EXCLUDED from every operational read and logged at
    # WARNING once per (group, scalar) — fail-visible, never silently
    # clamped (clamping would hide the misregistration upstream).
    # ------------------------------------------------------------------
    OPERATIONAL_SCALAR_ABS_LIMIT: float = 1.0e6

    def __new__(cls, *args: Any, **kwargs: Any) -> GlobalOmniScalarNetwork:
        """Singleton pattern implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        device: str | None = None,
        quantum_mode: bool | None = None,
        max_dimensions: int | None = None,
        domain: str | None = None,
        num_attention_heads: int | None = None,
        enable_triadic_phi: bool | None = None,
    ):
        """Initialize the Global Omni-Scalar Network singleton.

        First construction resolves and freezes the configuration.  Every
        parameter defaults to ``None`` meaning "not specified": a later
        no-argument construction is the established "give me the live
        singleton" idiom and stays a cheap no-op.  A later construction
        that EXPLICITLY requests a materially different configuration
        (different device, quantum mode, dimensions, attention heads,
        triadic weighting, or a domain resolving to a different
        σ_Immutable threshold) raises :class:`ValueError` instead of
        silently ignoring the request — call
        :func:`reset_global_network` first to reconfigure.  ``domain`` is
        material through its resolved threshold; because ``None`` doubles
        as "not specified", a caller cannot use ``domain=None`` to demand
        the non-domain default — reset the singleton instead.

        Args:
            device: Computation device ('cpu' or 'cuda'). Default 'cpu'.
            quantum_mode: Enable quantum-inspired operations. Default False.
            max_dimensions: Maximum dimensions for fusion. Default 37.
            domain: Domain identifier for threshold tuning (e.g., "medical").
            num_attention_heads: Number of attention heads. Default 32
                (head_dim=16).
            enable_triadic_phi: Enable triadic phi-weighting for harmonic
                synergy. Default True.

        Raises:
            ValueError: If the singleton is already initialized and an
                explicitly requested parameter differs materially from the
                live configuration.
        """
        # Serialize full initialization (and the re-init config check)
        # under the class lock: previously a second thread constructing
        # while the first was mid-``__init__`` saw ``_initialized=False``
        # and re-ran initialization concurrently.
        with self._lock:
            if getattr(self, "_initialized", False):
                self._raise_if_material_reinit(
                    device=device,
                    quantum_mode=quantum_mode,
                    max_dimensions=max_dimensions,
                    domain=domain,
                    num_attention_heads=num_attention_heads,
                    enable_triadic_phi=enable_triadic_phi,
                )
                return

            self.device = device if device is not None else "cpu"
            self.quantum_mode = quantum_mode if quantum_mode is not None else False
            self.max_dimensions = max_dimensions if max_dimensions is not None else 37
            self.domain = domain
            self.num_attention_heads = (
                num_attention_heads if num_attention_heads is not None else 32
            )
            self.enable_triadic_phi = enable_triadic_phi if enable_triadic_phi is not None else True
            self.logger = logging.getLogger(__name__)

            # Get domain-appropriate sigma_Immutable threshold
            self.sigma_immutable_threshold = get_sigma_immutable_threshold(domain)

            # Registry lock: guards every mutation of / snapshot read from
            # ``registered_scalars`` + ``scalar_groups`` and the ownership
            # index below.  Reentrant because operational readers
            # (``_collect_all_scalars`` → ``_operational_scalars_for``)
            # nest lock acquisitions for consistent multi-group snapshots.
            self._registry_lock = threading.RLock()

            self.registered_scalars: dict[str, ScalarRegistration] = {}
            self.scalar_groups: dict[ScalarGroup, dict[str, float]] = {
                group: {} for group in ScalarGroup
            }

            # Per-component ownership of group entries, so
            # ``unregister_scalars`` can remove exactly what a component
            # contributed. Keyed by (group, scalar name) -> component.
            self._scalar_owners: dict[tuple[ScalarGroup, str], str] = {}
            self._component_scalar_keys: dict[str, set[tuple[ScalarGroup, str]]] = {}

            # (group, scalar name) pairs already reported by the
            # gross-outlier tripwire — WARNING fires once per scalar, not
            # once per gate evaluation.
            self._quarantine_warned: set[tuple[ScalarGroup, str]] = set()

            # Initialize ethical gate with configurable threshold
            self.ethical_gate = EthicalGate(threshold=self.sigma_immutable_threshold)

            # Initialize 32-head attention with triadic phi-weighting
            self.attention_fusion = MultiHeadAttentionFusion(
                d_model=512,
                num_heads=self.num_attention_heads,
                max_dimensions=self.max_dimensions,
                enable_triadic_phi=self.enable_triadic_phi,
            )

            # Track harmonic synergy for weighted fusion Equation
            self.last_harmonic_synergy: float = 0.5

            self._initialize_default_scalars()

            # Baseline defaults snapshot: when a component that shadowed a
            # default scalar unregisters, the default value is restored so
            # the σ_Immutable operational layout never loses a column.
            self._baseline_scalars: dict[ScalarGroup, dict[str, float]] = {
                group: dict(scalars) for group, scalars in self.scalar_groups.items()
            }

            self._initialized = True

            self.logger.debug(
                "GOSNN initialized with %d dimensions and %d attention heads",
                self.max_dimensions,
                self.num_attention_heads,
            )

    def _material_config_divergence(
        self,
        device: str | None,
        quantum_mode: bool | None,
        max_dimensions: int | None,
        domain: str | None,
        num_attention_heads: int | None,
        enable_triadic_phi: bool | None,
    ) -> dict[str, tuple[Any, Any]]:
        """Compare explicitly requested config against the live instance.

        Only parameters the caller actually passed (non-``None``)
        participate; unspecified parameters mean "whatever the live
        instance has".  ``domain`` is compared through its resolved
        σ_Immutable threshold — two domain labels mapping to the same
        threshold are not a material difference.

        Returns:
            Mapping of field name -> ``(live_value, requested_value)`` for
            every material difference; empty when the request is
            compatible with the live configuration.
        """
        requested: dict[str, Any] = {}
        if device is not None:
            requested["device"] = device
        if quantum_mode is not None:
            requested["quantum_mode"] = quantum_mode
        if max_dimensions is not None:
            requested["max_dimensions"] = max_dimensions
        if num_attention_heads is not None:
            requested["num_attention_heads"] = num_attention_heads
        if enable_triadic_phi is not None:
            requested["enable_triadic_phi"] = enable_triadic_phi
        if domain is not None:
            requested["sigma_immutable_threshold"] = get_sigma_immutable_threshold(domain)

        diffs: dict[str, tuple[Any, Any]] = {}
        for field_name, want in requested.items():
            have = getattr(self, field_name)
            if have != want:
                diffs[field_name] = (have, want)
        return diffs

    def _raise_if_material_reinit(self, **requested: Any) -> None:
        """Fail loudly when re-construction requests a different config.

        Before this check, ``__init__`` early-returned and the second
        caller's device/domain/threshold/dimensions were SILENTLY ignored
        — e.g. the first caller's σ_Immutable threshold (0.96 default vs
        0.93 medical) stayed frozen for process life while a medical
        caller believed it had configured 0.93 (defect F10).

        Raises:
            ValueError: Listing each materially different field with the
                live and requested values, and directing the caller to
                ``reset_global_network()``.
        """
        diffs = self._material_config_divergence(**requested)
        if not diffs:
            return
        detail = ", ".join(
            f"{name}: live={have!r} requested={want!r}" for name, (have, want) in diffs.items()
        )
        raise ValueError(
            "GlobalOmniScalarNetwork is a process-wide singleton and is already "
            f"initialized with a materially different configuration ({detail}). "
            "Re-construction cannot reconfigure it; call "
            "omni_mercury_engine.core.global_omni_scalar_network."
            "reset_global_network() first if you really need a differently "
            "configured instance."
        )

    def _initialize_default_scalars(self) -> None:
        """Initialize default ethical and system scalars with omni- prefix.

        All scalars use the omni- prefix for unified naming convention. Legacy aliases (without
        omni- prefix) are maintained for backward compatibility and will be deprecated in v2.0.
        """
        # Core ethical scalars with omni- prefix
        # omnibenevolence uses ETHICAL.OMNIBENEVOLENCE_SCALAR (0.99)
        self.scalar_groups[ScalarGroup.ETHICAL] = {
            # Primary omni-scalars
            "omnimorality": self.MIN_MORALITY,
            "omniempathy": self.MIN_EMPATHY,
            "omnicompassion": 1.30,
            "omniforgiveness": 0.1,
            "omnilove": 1.30,
            "omnidetermination": 1.30,
            "omniloyalty": 1.30,
            "omniintegrity": 1.30,
            "omniwisdom": 1.30,
            "omnijustice": 1.30,
            "omnialtruism": 1.30,
            "omnihope": 1.30,
            "omnicourage": 1.30,
            "omniaccountability": 1.30,
            "omnitransparency": 0.18,
            "omniexplainability": 0.9,
            "omnibenevolence": ETHICAL.OMNIBENEVOLENCE_SCALAR,  # Core benevolence threshold
            "omniequity": 1.30,
            "omnigrace": 1.25,
            "omnipatience": 1.20,
            "omnihumility": 1.15,
            "omniresilience": 1.30,
            "omniperseverance": 1.25,
            "omnivigilance": 1.20,
            "omnistewardship": 1.25,
            # Operational scalars
            "survivor_first_principle": 1.35,
            "bias_audit_compliance": 1.25,
        }

        self.scalar_groups[ScalarGroup.COSMIC] = {
            "omniuniverse_adapt": 1.20,
            "omnitelos": 1.25,
            "omni_black_hole_entropy": 1.30,
            "omni_harmonic_singularity": 1.30,
            "omni_golden_ratio_phi": self.PHI,
            "omnicosmicharmony": 1.28,
            "omnistellarresonance": 1.22,
        }

        self.scalar_groups[ScalarGroup.QUANTUM_CONSCIOUSNESS] = {
            "omniquantum_weight": 0.12,
            "omnientanglement_risk": 0.1,
            "omniquantum_entanglement": 0.14,
            "omnineuroquantum": 1.30,
            "omniconsciousness_coherence": 1.25,
            "omniquantum_superposition": 1.18,
            "omniquantum_decoherence_shield": 1.20,
        }

        self.scalar_groups[ScalarGroup.HUMANITARIAN] = {
            "omnicrisis_response": 1.35,
            "omnidisaster_response": 1.30,
            "omnipandemic_monitoring": 1.25,
            "omnimissing_persons_priority": 1.40,
            "omnimedical_discovery": 1.30,
            "omnihumanitarian_aid": 1.35,
            "omnirefugee_protection": 1.30,
            "omnifood_security": 1.25,
            "omniclimate_resilience": 1.28,
        }

        self.scalar_groups[ScalarGroup.SECURITY] = {
            "omnithreat_detection": 1.25,
            "omniquantum_resistance": 1.30,
            "omniencryption_strength": 1.35,
            "omniaudit_compliance": 1.20,
            "omnicyber_fortress": 1.28,
            "omnizero_trust": 1.22,
        }

        # SOFTWARE_ENGINEERING scalars.
        # Spans ten families: one operational and nine diagnostic.
        # The diagnostic families are filtered out of the σ_Immutable
        # input vector and every operational aggregation; see the
        # class-level ``_METRIC_ONLY_PREFIXES`` block.
        #
        #   Operational (drive the gate, 45):
        #     1. Code-quality / optimization / 3R synergy (45)
        #
        #   Diagnostic measurement (registered but not operational, 82):
        #     2. ISO/IEC 25010:2011 product-quality model (31)
        #     3. Halstead complexity measures (7)
        #     4. McCabe + cognitive (SonarQube) complexity (5)
        #     5. Maintainability Index variants (3)
        #     6. NIST SAMATE software assurance (10)
        #     7. DORA / DevOps Research and Assessment delivery (4)
        #     8. SLSA Supply-chain Levels for Software Artifacts v1.0 (4)
        #     9. Supply-chain / repository-integrity checks (10, Mercury-native)
        #    10. ISO/IEC 5055 / CISQ automated source-code quality measures (4)
        #    11. NIST SP 800-218 SSDF practice groups (4)
        #
        # Weights >1.0 are positive-direction scalars (more is better);
        # weights <1.0 are penalty scalars (less is better, e.g. defect density).
        self.scalar_groups[ScalarGroup.SOFTWARE_ENGINEERING] = {
            # Code Quality Metrics (15 scalars)
            "omni_code_complexity": 1.20,  # Cyclomatic/cognitive complexity control
            "omni_code_coverage": 1.25,  # Test coverage percentage
            "omni_property_test_coverage": 1.22,  # Property-based testing depth
            "omni_type_safety_index": 1.28,  # Static type coverage
            "omni_lint_compliance": 1.15,  # Linting rule adherence
            "omni_documentation_quality": 1.18,  # Docstring/comment coverage
            "omni_api_consistency": 1.20,  # API design coherence
            "omni_dependency_health": 1.22,  # Dependency freshness/security
            "omni_code_duplication": 0.85,  # Lower is better (penalty scalar)
            "omni_technical_debt": 0.80,  # Lower is better (penalty scalar)
            "omni_maintainability_index": 1.25,  # Aggregate maintainability
            "omni_readability_score": 1.20,  # Code readability metrics
            "omni_modularity_factor": 1.22,  # Module coupling/cohesion
            "omni_interface_clarity": 1.18,  # Clean interface design
            "omni_abstraction_level": 1.20,  # Appropriate abstraction depth
            # Optimization Metrics (15 scalars)
            "omni_runtime_optimization": 1.30,  # Runtime performance efficiency
            "omni_memory_efficiency": 1.25,  # Memory usage optimization
            "omni_algorithmic_efficiency": 1.28,  # Big-O complexity control
            "omni_cache_hit_ratio": 1.22,  # Cache effectiveness
            "omni_latency_reduction": 1.25,  # Response time optimization
            "omni_throughput_factor": 1.24,  # Processing throughput
            "omni_resource_utilization": 1.20,  # CPU/GPU utilization balance
            "omni_parallel_efficiency": 1.26,  # Parallelization effectiveness
            "omni_io_optimization": 1.22,  # I/O operation efficiency
            "omni_network_efficiency": 1.20,  # Network call optimization
            "omni_garbage_collection_health": 1.18,  # GC pressure management
            "omni_startup_time": 1.15,  # Initialization speed
            "omni_shutdown_grace": 1.12,  # Clean shutdown efficiency
            "omni_hotpath_optimization": 1.28,  # Critical path performance
            "omni_vectorization_factor": 1.24,  # SIMD/vectorization usage
            # 3R Synergy & Correctness (15 scalars)
            "omni_3r_synergy_factor": 1.35,  # 3R mechanism integration strength
            "omni_recursion_depth_control": 1.22,  # Recursion safety bounds
            "omni_resonance_stability": 1.25,  # Frequency analysis coherence
            "omni_refactoring_confidence": 1.28,  # Safe refactoring score
            "omni_lyapunov_convergence_rate": 1.30,  # Convergence speed (λ=0.25)
            "omni_precision_recall_harmonic": 1.25,  # F1-like balance metric
            "omni_false_positive_reduction": 1.28,  # FP suppression strength
            "omni_false_negative_reduction": 1.22,  # FN recovery capability
            "omni_detection_confidence": 1.26,  # Anomaly detection certainty
            "omni_explanation_depth": 1.20,  # Explainability quality
            "omni_regression_prevention": 1.25,  # Regression test coverage
            "omni_invariant_preservation": 1.28,  # Invariant enforcement
            "omni_contract_compliance": 1.22,  # Design-by-contract adherence
            "omni_mutation_test_score": 1.24,  # Mutation testing effectiveness
            "omni_fuzzing_resilience": 1.26,  # Fuzz testing robustness
            # ISO/IEC 25010:2011 - Functional Suitability (3)
            "omni_iso25010_func_completeness": 1.22,  # Coverage of stated/implied needs
            "omni_iso25010_func_correctness": 1.28,  # Correct results with needed precision
            "omni_iso25010_func_appropriateness": 1.20,  # Fit-for-task suitability
            # ISO/IEC 25010 - Performance Efficiency (3)
            "omni_iso25010_perf_time_behavior": 1.25,  # Response/processing/throughput rates
            "omni_iso25010_perf_resource_util": 1.22,  # Amounts/types of resources used
            "omni_iso25010_perf_capacity": 1.20,  # Maximum limits meet requirements
            # ISO/IEC 25010 - Compatibility (2)
            "omni_iso25010_compat_coexistence": 1.18,  # Performs alongside other products
            "omni_iso25010_compat_interoperability": 1.22,  # Exchanges info across systems
            # ISO/IEC 25010 - Usability (6)
            "omni_iso25010_usab_appropriateness_recog": 1.18,  # Users recognize suitability
            "omni_iso25010_usab_learnability": 1.20,  # Effective learning with use
            "omni_iso25010_usab_operability": 1.20,  # Easy to operate and control
            "omni_iso25010_usab_user_error_protect": 1.25,  # Protects users from errors
            "omni_iso25010_usab_ui_aesthetics": 1.15,  # Pleasing interaction
            "omni_iso25010_usab_accessibility": 1.22,  # Usable by widest range of people
            # ISO/IEC 25010 - Reliability (4)
            "omni_iso25010_rel_maturity": 1.25,  # Meets reliability needs under normal use
            "omni_iso25010_rel_availability": 1.28,  # Operational and accessible when required
            "omni_iso25010_rel_fault_tolerance": 1.26,  # Operates despite faults
            "omni_iso25010_rel_recoverability": 1.24,  # Recovers data and re-establishes state
            # ISO/IEC 25010 - Security (5)
            "omni_iso25010_sec_confidentiality": 1.30,  # Accessible only to authorized
            "omni_iso25010_sec_integrity": 1.30,  # Prevents unauthorized modification
            "omni_iso25010_sec_non_repudiation": 1.25,  # Actions proven to have taken place
            "omni_iso25010_sec_accountability": 1.25,  # Actions traced to entity
            "omni_iso25010_sec_authenticity": 1.28,  # Identity can be proved
            # ISO/IEC 25010 - Maintainability (5)
            "omni_iso25010_maint_modularity": 1.25,  # Discrete components, low impact change
            "omni_iso25010_maint_reusability": 1.22,  # Asset reuse in more than one system
            "omni_iso25010_maint_analyzability": 1.24,  # Effective assessment of impact
            "omni_iso25010_maint_modifiability": 1.26,  # Effective modification w/o defects
            "omni_iso25010_maint_testability": 1.28,  # Test criteria established and met
            # ISO/IEC 25010 - Portability (3)
            "omni_iso25010_port_adaptability": 1.20,  # Adapts to different environments
            "omni_iso25010_port_installability": 1.18,  # Effective installation/uninstallation
            "omni_iso25010_port_replaceability": 1.18,  # Replaces another product for same use
            # Halstead complexity measures (Halstead 1977) - 7 derived measures.
            # n1=distinct operators, n2=distinct operands, N1=total operators, N2=total operands.
            # Penalty direction: lower volume/difficulty/effort/bugs is better.
            "omni_halstead_vocabulary": 0.92,  # n = n1 + n2 (penalty: smaller is simpler)
            "omni_halstead_length": 0.92,  # N = N1 + N2 (penalty)
            "omni_halstead_volume": 0.88,  # V = N * log2(n) (penalty)
            "omni_halstead_difficulty": 0.85,  # D = (n1/2) * (N2/n2) (penalty)
            "omni_halstead_effort": 0.85,  # E = D * V (penalty)
            "omni_halstead_time_to_program": 0.90,  # T = E / 18 seconds (penalty)
            "omni_halstead_delivered_bugs": 0.80,  # B = V / 3000 (penalty: fewer bugs = better)
            # McCabe cyclomatic + cognitive complexity (SonarQube definition) - 5 scalars
            "omni_mccabe_cyclomatic_complexity": 0.85,  # CC = E - N + 2P (penalty)
            "omni_mccabe_essential_complexity": 0.85,  # ev(G), unstructured residual (penalty)
            "omni_mccabe_design_complexity": 0.88,  # iv(G), integration complexity (penalty)
            "omni_cognitive_complexity_sonar": 0.82,  # SonarSource cognitive metric (penalty)
            "omni_npath_complexity": 0.88,  # NPATH, acyclic execution paths (penalty)
            # Maintainability Index variants (SEI / Microsoft VS) - 3 scalars.
            # MI_SEI    = 171 - 5.2*ln(HV) - 0.23*CC - 16.2*ln(LOC)
            # MI_VS     = MAX(0, (171 - 5.2*ln(HV) - 0.23*CC - 16.2*ln(LOC)) * 100/171)
            # Already present: omni_maintainability_index (aggregate).
            "omni_maintainability_index_sei": 1.25,  # SEI raw scale 0-171
            "omni_maintainability_index_vs": 1.25,  # Visual Studio scale 0-100
            "omni_maintainability_index_delta": 1.20,  # MI change over time (trend)
            # NIST SAMATE software assurance metrics - 10 scalars.
            # Drawn from SAMATE Reference Dataset (SARD), CWE coverage, SwAMM metrics.
            "omni_samate_cwe_coverage": 1.28,  # Fraction of relevant CWEs covered by tooling
            "omni_samate_sard_conformance": 1.24,  # Conformance to SARD test-suite expectations
            "omni_samate_weakness_density": 0.82,  # CWE findings per kLOC (penalty)
            "omni_samate_assurance_case_strength": 1.30,  # Assurance case argument strength
            "omni_samate_tool_effectiveness": 1.25,  # Recall x precision on SAMATE benchmarks
            "omni_samate_false_discovery_rate": 0.85,  # FDR on assurance tooling (penalty)
            "omni_samate_residual_risk": 0.80,  # Residual security risk score (penalty)
            "omni_samate_evidence_completeness": 1.25,  # Completeness of assurance evidence
            "omni_samate_independent_verification": 1.28,  # IV&V coverage strength
            "omni_samate_supply_chain_assurance": 1.30,  # SBOM/provenance assurance (NIST SSDF)
            # DORA / DevOps Research and Assessment delivery metrics - 4 scalars.
            # Four key metrics from Accelerate / State-of-DevOps reports
            # (Forsgren, Humble, Kim).  Throughput pair (deployment freq,
            # lead time) + stability pair (MTTR, change failure rate).
            "omni_dora_deployment_frequency": 1.20,  # Elite teams ship many times per day
            "omni_dora_lead_time_for_changes": 0.88,  # Time commit->prod (penalty: shorter better)
            "omni_dora_mean_time_to_restore": 0.85,  # MTTR after incident (penalty)
            "omni_dora_change_failure_rate": 0.82,  # Fraction of deploys causing incidents (penalty)
            # SLSA Supply-chain Levels for Software Artifacts v1.0 - 4 scalars.
            # Tracks the build-track maturity (L0..L3) plus the three
            # SLSA properties that anchor each level.
            "omni_slsa_level": 1.30,  # Achieved SLSA build-track level (0..3)
            "omni_slsa_source_integrity": 1.30,  # Source is version-controlled and verified
            "omni_slsa_build_provenance": 1.28,  # Build produces verifiable provenance attestation
            "omni_slsa_dependency_attestation": 1.26,  # Transitive deps carry signed provenance
            # Supply-chain & repository-integrity checks - 10 scalars.
            # Mercury-native signals, computed by handwritten checks over the
            # repo's OWN configuration (workflows, CODEOWNERS, dependabot,
            # SECURITY policy, SHA-pinning) in scripts/collect_sw_eng_metrics.py.
            # There is NO runtime dependency on any external scoring tool or
            # service — Mercury measures these itself. The ``omni_ossf_`` prefix
            # is a frozen grouping label for this open-source-supply-chain band;
            # the 10 signals track the most actionable, objectively-measurable
            # supply-chain hardening practices (branch protection, review,
            # pinning, SAST, least-privilege tokens, signed releases, ...).
            "omni_ossf_branch_protection": 1.22,  # Main branch protected from force-push
            "omni_ossf_code_review_required": 1.25,  # PRs require approving review
            "omni_ossf_ci_tests_required": 1.20,  # CI runs and gates on tests
            "omni_ossf_dependency_update_tool": 1.22,  # Dependabot / Renovate enabled
            "omni_ossf_dangerous_workflow": 0.85,  # Untrusted-input workflow patterns (penalty)
            "omni_ossf_pinned_dependencies": 1.24,  # Deps pinned by hash, not floating tag
            "omni_ossf_sast_enabled": 1.26,  # CodeQL / Semgrep / equivalent runs in CI
            "omni_ossf_token_permissions": 1.25,  # Workflow tokens scoped to least privilege
            "omni_ossf_signed_releases": 1.28,  # Release artifacts carry cryptographic signature
            "omni_ossf_vulnerabilities": 0.80,  # Open known-vuln count (penalty: fewer is better)
            # ISO/IEC 5055 automated source code quality measures (CISQ) - 4 scalars.
            # Code-base level health measures derived from rule catalogs
            # spanning CWE, CERT, OMG; standardised by the Consortium for
            # Information & Software Quality.
            "omni_iso5055_reliability": 1.25,  # ISO 5055 Reliability factor
            "omni_iso5055_performance_efficiency": 1.24,  # ISO 5055 Performance Efficiency
            "omni_iso5055_security": 1.30,  # ISO 5055 Security factor
            "omni_iso5055_maintainability": 1.22,  # ISO 5055 Maintainability factor
            # NIST SP 800-218 Secure Software Development Framework (SSDF) - 4 scalars.
            # The four practice groups from the SSDF v1.1.  Distinct from
            # the single ``omni_samate_supply_chain_assurance`` aggregate above
            # (which collapses SSDF into one assurance signal); these expand
            # the framework so coverage gaps surface per-group.
            "omni_ssdf_prepare_organization": 1.22,  # PO: define roles, policies, and toolchain
            "omni_ssdf_protect_software": 1.28,  # PS: protect components from tamper / exposure
            "omni_ssdf_produce_well_secured_software": 1.26,  # PW: design, build, verify securely
            "omni_ssdf_respond_to_vulnerabilities": 1.25,  # RV: identify, assess, remediate disclosures
        }

        # Overlay REAL measured values onto the diagnostic (metric-only) SW-eng
        # scalars a collector can compute honestly — Halstead / cyclomatic / MI
        # via ``ast``, Mercury-native supply-chain / repository-integrity checks
        # from repo config, DORA VCS-history proxies from git log, and the
        # SSDF / SLSA / computable-SAMATE families from repo state (36 of the 82
        # placeholders; the rest stay documented placeholders rather than
        # fabricated).  Updates existing keys only, so the count (127) and the
        # frozen σ_Immutable operational layout are untouched and the scalars
        # stay metric-only.  A missing artifact is a no-op.
        _apply_measured_sw_eng_metrics(self.scalar_groups[ScalarGroup.SOFTWARE_ENGINEERING])

        # MEDICAL scalars (~10 scalars for healthcare and diagnostics)
        self.scalar_groups[ScalarGroup.MEDICAL] = {
            "omni_diagnostic_accuracy": 1.30,  # Diagnostic precision
            "omni_patient_safety": 1.40,  # Patient harm prevention (highest)
            "omni_treatment_efficacy": 1.28,  # Treatment effectiveness
            "omni_false_alarm_minimization": 1.25,  # Reduce alert fatigue
            "omni_critical_alert_sensitivity": 1.35,  # Catch critical conditions
            "omni_hipaa_compliance": 1.30,  # Privacy compliance
            "omni_clinical_explainability": 1.28,  # Medical explanation quality
            "omni_drug_interaction_check": 1.32,  # Medication safety
            "omni_triage_accuracy": 1.30,  # Emergency prioritization
            "omni_outcome_prediction": 1.25,  # Prognosis reliability
        }

        # ADVANCED_REASONING scalars (~16 scalars for logic, inference, knowledge)
        self.scalar_groups[ScalarGroup.ADVANCED_REASONING] = {
            "omni_logical_consistency": 1.28,  # Logical coherence
            "omni_inference_depth": 1.25,  # Reasoning chain depth
            "omni_abductive_reasoning": 1.22,  # Hypothesis generation
            "omni_deductive_strength": 1.26,  # Logical deduction quality
            "omni_inductive_generalization": 1.24,  # Pattern generalization
            "omni_analogical_transfer": 1.22,  # Cross-domain reasoning
            "omni_causal_inference": 1.28,  # Causal relationship detection
            "omni_counterfactual_reasoning": 1.25,  # What-if analysis
            "omni_temporal_reasoning": 1.24,  # Time-based logic
            "omni_spatial_reasoning": 1.22,  # Spatial relationship understanding
            "omni_knowledge_synthesis": 1.26,  # Information integration
            "omni_uncertainty_quantification": 1.28,  # Uncertainty handling
            "omni_belief_revision": 1.24,  # Belief update consistency
            "omni_metacognitive_awareness": 1.22,  # Self-knowledge accuracy
            "omni_common_sense_reasoning": 1.25,  # Commonsense inference
            "omni_metasymbolic_grounding": 1.27,  # Grounding of metasymbolic reasoning in evidence
        }

        # Initialize legacy alias mapping for backward compatibility
        self._initialize_legacy_aliases()

    def _initialize_legacy_aliases(self) -> None:
        """Initialize backward-compatible legacy aliases (deprecated in v2.0).

        Maps old scalar names to new omni-prefixed names for seamless migration.
        """
        self._legacy_aliases: dict[str, str] = {
            # Ethical scalars
            "morality_scalar": "omnimorality",
            "empathy_scalar": "omniempathy",
            "compassion_scalar": "omnicompassion",
            "forgiveness": "omniforgiveness",
            "love_scalar": "omnilove",
            "determination_scalar": "omnidetermination",
            "loyalty_scalar": "omniloyalty",
            "integrity_scalar": "omniintegrity",
            "wisdom_scalar": "omniwisdom",
            "justice_scalar": "omnijustice",
            "altruism_scalar": "omnialtruism",
            "hope_scalar": "omnihope",
            "courage_scalar": "omnicourage",
            "accountability_scalar": "omniaccountability",
            "transparency_weight": "omnitransparency",
            "explainability_factor": "omniexplainability",
            "benevolence": "omnibenevolence",
            "equity": "omniequity",
            # Cosmic scalars
            "universe_adapt": "omniuniverse_adapt",
            "telos_scalar": "omnitelos",
            "black_hole_entropy_eth": "omni_black_hole_entropy",
            "harmonic_singularity_bridge": "omni_harmonic_singularity",
            "golden_ratio_phi": "omni_golden_ratio_phi",
            # Quantum consciousness scalars
            "quantum_weight": "omniquantum_weight",
            "entanglement_risk": "omnientanglement_risk",
            "quantum_entanglement_weight": "omniquantum_entanglement",
            "neuro_quantum": "omnineuroquantum",
            "consciousness_coherence": "omniconsciousness_coherence",
            # Humanitarian scalars
            "crisis_response_boost": "omnicrisis_response",
            "disaster_response_boost": "omnidisaster_response",
            "pandemic_monitoring": "omnipandemic_monitoring",
            "missing_persons_priority": "omnimissing_persons_priority",
            "medical_discovery_boost": "omnimedical_discovery",
            # Security scalars
            "threat_detection_sensitivity": "omnithreat_detection",
            "quantum_resistance": "omniquantum_resistance",
            "encryption_strength": "omniencryption_strength",
            "audit_compliance": "omniaudit_compliance",
        }

    def resolve_scalar_name(self, name: str) -> str:
        """Resolve a scalar name, supporting legacy aliases.

        Args:
            name: Scalar name (may be legacy or omni-prefixed)

        Returns:
            Resolved omni-prefixed scalar name
        """
        if hasattr(self, "_legacy_aliases") and name in self._legacy_aliases:
            self.logger.debug(
                f"Legacy scalar alias '{name}' resolved to '{self._legacy_aliases[name]}' "
                "(deprecated in v2.0)"
            )
            return self._legacy_aliases[name]
        return name

    def get_scalar(self, name: str, default: float = 0.0) -> float:
        """Get a scalar value by name, supporting legacy aliases.

        Args:
            name: Scalar name (may be legacy or omni-prefixed)
            default: Default value if scalar not found

        Returns:
            Scalar value
        """
        resolved_name = self.resolve_scalar_name(name)
        with self._registry_lock:
            for group_scalars in self.scalar_groups.values():
                if resolved_name in group_scalars:
                    return group_scalars[resolved_name]
        return default

    def get_group_scalars(self, group: ScalarGroup) -> dict[str, float]:
        """Return a consistent snapshot of one group's registered scalars.

        Includes diagnostic (metric-only) entries — this is the
        *registered* view, not the operational one.  External readers must
        use this instead of iterating ``scalar_groups[group]`` directly:
        the live dict is mutated by concurrent ``register_scalars`` /
        ``unregister_scalars`` calls and iterating it raises
        ``RuntimeError: dictionary changed size during iteration``.

        Args:
            group: Scalar group to snapshot.

        Returns:
            Copy of the group's name -> value mapping.
        """
        with self._registry_lock:
            return dict(self.scalar_groups[group])

    def register_scalars(
        self,
        component_name: str,
        scalars: dict[str, float],
        group: ScalarGroup = ScalarGroup.ETHICAL,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register scalars from a component.

        Thread-safe.  Ownership of each ``(group, name)`` entry is tracked
        per component so :meth:`unregister_scalars` can remove exactly the
        entries this component contributed.  Repeated registrations by the
        same component accumulate ownership (a component that registers
        scalar A and later scalar B owns both).  When a component
        overwrites an entry currently owned by a DIFFERENT component the
        write wins (last-write-wins, the pre-existing contract) but a
        WARNING is logged and ownership transfers to the new component;
        unregistering the previous owner will then no longer touch the
        entry, and unregistering the new owner deletes it without
        restoring the overwritten value (re-register instead of relying on
        restore).

        Args:
            component_name: Name of the registering component
            scalars: Dictionary of scalar name to value
            group: Scalar group classification
            metadata: Optional metadata about the registration
        """
        registration = ScalarRegistration(
            component_name=component_name,
            scalars=scalars,
            group=group,
            timestamp=time.time(),
            metadata=metadata or {},
        )

        with self._registry_lock:
            self.registered_scalars[component_name] = registration
            owned = self._component_scalar_keys.setdefault(component_name, set())

            for name, value in scalars.items():
                key = (group, name)
                previous_owner = self._scalar_owners.get(key)
                if previous_owner is not None and previous_owner != component_name:
                    self.logger.warning(
                        "GOSNN scalar %r in group %s overwritten by component %r "
                        "(previously owned by %r); last write wins and ownership "
                        "transfers — unregistering %r will delete the entry, not "
                        "restore %r's value",
                        name,
                        group.value,
                        component_name,
                        previous_owner,
                        component_name,
                        previous_owner,
                    )
                    previous_keys = self._component_scalar_keys.get(previous_owner)
                    if previous_keys is not None:
                        previous_keys.discard(key)
                self._scalar_owners[key] = component_name
                owned.add(key)
                self.scalar_groups[group][name] = value

        self.logger.debug(
            f"Registered {len(scalars)} scalars from {component_name} in group {group.value}"
        )

    def unregister_scalars(self, component_name: str) -> bool:
        """Remove a component's registration and its group contributions.

        Removes exactly the ``(group, name)`` entries currently owned by
        ``component_name`` (accumulated across all of its
        :meth:`register_scalars` calls).  Entries that shadowed a built-in
        default scalar are restored to the default value — deleting them
        would silently shrink the σ_Immutable operational layout.  Entries
        the component introduced are deleted.  Entries the component once
        owned but that another component has since overwritten are left
        untouched (ownership transferred at overwrite time).

        Args:
            component_name: Component whose registration to remove.

        Returns:
            True if the component had a registration or owned entries,
            False if there was nothing to remove.
        """
        with self._registry_lock:
            registration = self.registered_scalars.pop(component_name, None)
            owned = self._component_scalar_keys.pop(component_name, set())

            for key in owned:
                if self._scalar_owners.get(key) != component_name:
                    # Defensive: ownership index and per-component sets are
                    # updated together under the lock, so this cannot
                    # happen; skip rather than delete someone else's entry.
                    continue
                del self._scalar_owners[key]
                group, name = key
                baseline_value = self._baseline_scalars.get(group, {}).get(name)
                if baseline_value is not None:
                    self.scalar_groups[group][name] = baseline_value
                else:
                    self.scalar_groups[group].pop(name, None)
                # Allow the gross-outlier tripwire to fire again if the
                # scalar is later re-registered with another bad value.
                self._quarantine_warned.discard(key)

            removed = registration is not None or bool(owned)

        if removed:
            self.logger.debug(
                "Unregistered component %r (%d owned scalar entries removed/restored)",
                component_name,
                len(owned),
            )
        return removed

    @contextmanager
    def scalar_registration(
        self,
        component_name: str,
        scalars: dict[str, float],
        group: ScalarGroup = ScalarGroup.ETHICAL,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[GlobalOmniScalarNetwork]:
        """Register scalars for the duration of a ``with`` block.

        On exit ALL entries owned by ``component_name`` are unregistered
        via :meth:`unregister_scalars` — including entries from earlier
        registrations under the same component name — with shadowed
        default scalars restored to their default values.

        Args:
            component_name: Name of the registering component.
            scalars: Dictionary of scalar name to value.
            group: Scalar group classification.
            metadata: Optional metadata about the registration.

        Yields:
            This network instance, for chained calls inside the block.
        """
        self.register_scalars(component_name, scalars, group=group, metadata=metadata)
        try:
            yield self
        finally:
            self.unregister_scalars(component_name)

    def get_enhanced_scalars(
        self,
        requesting_component: str,
        base_scalars: dict[str, float],
        context: dict[str, Any] | None = None,
    ) -> EnhancementResult:
        """Get enhanced scalars with GOSNN fusion, ethical gating, and harmonic synergy.

        This method performs bidirectional synaptic integration:
        1. Collects all registered scalars from components
        2. Evaluates ethical compliance via sigma_Immutable threshold
        3. Fuses dimensional states using 32-head attention with triadic phi-weighting
        4. Computes harmonic synergy for the weighted fusion Equation H(omega) term
        5. Returns enhanced scalars with fusion metadata

        Args:
            requesting_component: Name of the requesting component
            base_scalars: Base scalar values to enhance
            context: Optional context for enhancement (e.g., domain for threshold tuning)

        Returns:
            EnhancementResult with enhanced scalars, fusion score, harmonic synergy,
            ethical gate status, and any warnings
        """
        context = context or {}
        warnings: list[str] = []

        all_scalars = self._collect_all_scalars()
        scalar_vector = np.array(list(all_scalars.values()))

        passes_gate, ethical_score = self.ethical_gate.evaluate(scalar_vector)

        if not passes_gate:
            warnings.append(
                f"Ethical gate warning: score {ethical_score:.3f} below threshold "
                f"{self.sigma_immutable_threshold:.2f}"
            )
            # The exact σ_Immutable threshold is a static governance constant
            # (already captured in the returned ``warnings`` above and in
            # configuration); the per-event log records only the actionable
            # facts — which component was gated and its score — rather than
            # re-emitting that constant on every trigger. Lazy %-args defer
            # formatting to the handler.
            self.logger.warning(
                "Ethical gate triggered for %s: score=%.3f (below the "
                "configured σ_Immutable floor)",
                requesting_component,
                ethical_score,
            )

        dimensional_states = self._prepare_dimensional_states(base_scalars, context)

        # Fuse with triadic phi-weighting and track harmonic synergy
        fuse_result = self.attention_fusion.fuse(dimensional_states, return_synergy=True)
        if isinstance(fuse_result, tuple):
            fused_state, harmonic_synergy = fuse_result
        else:
            fused_state = fuse_result
            harmonic_synergy = 0.5

        # Store harmonic synergy for weighted fusion Equation
        self.last_harmonic_synergy = harmonic_synergy

        enhanced_scalars = self._apply_enhancement(base_scalars, fused_state, ethical_score)

        intelligence_contribution = self._compute_intelligence_contribution(enhanced_scalars)

        return EnhancementResult(
            enhanced_scalars=enhanced_scalars,
            fusion_score=float(np.mean(fused_state)),
            ethical_gate_passed=passes_gate,
            intelligence_contribution=intelligence_contribution,
            warnings=warnings,
            fused_state=np.asarray(fused_state, dtype=np.float64),
            collected_scalars=all_scalars,
        )

    def fuse_37d_scalars(
        self, dimensional_states: list[np.ndarray[Any, Any]]
    ) -> np.ndarray[Any, Any]:
        """Perform 37-dimensional quantum fusion.

        Args:
            dimensional_states: List of dimensional state vectors

        Returns:
            Fused 37D state vector
        """
        return self.attention_fusion.fuse(dimensional_states)  # type: ignore[return-value, unused-ignore]

    def compute_global_intelligence_score(self) -> float:
        """Compute global intelligence score from all registered scalars.

        Returns:
            Global intelligence score (0-1)
        """
        all_scalars = self._collect_all_scalars()

        if not all_scalars:
            return 0.5

        scalar_values = np.array(list(all_scalars.values()))

        boost_count = np.sum(scalar_values > 1.0)
        penalty_count = np.sum(scalar_values < 1.0)
        total = boost_count + penalty_count

        boost_ratio = 0.5 if total == 0 else boost_count / total

        mean_value = np.mean(scalar_values)
        std_value = np.std(scalar_values)

        triadic_harmony = self._compute_triadic_harmony(scalar_values)

        intelligence_score = (
            0.3 * boost_ratio
            + 0.3 * min(mean_value / 2.0, 1.0)
            + 0.2 * (1.0 / (1.0 + std_value))
            + 0.2 * triadic_harmony
        )

        return float(np.clip(intelligence_score, 0.0, 1.0))

    def compute_triadic_harmony(self) -> float:
        """Compute triadic harmony using golden ratio (φ = 1.618).

        Triadic harmony represents the balance between boosts, penalties,
        and neutral scalars, weighted by the golden ratio.

        Returns:
            Triadic harmony score (0-1)
        """
        all_scalars = self._collect_all_scalars()
        scalar_values = np.array(list(all_scalars.values()))
        return self._compute_triadic_harmony(scalar_values)

    def _compute_triadic_harmony(self, scalar_values: np.ndarray[Any, Any]) -> float:
        """Internal triadic harmony computation."""
        if len(scalar_values) == 0:
            return 0.5

        boosts = scalar_values[scalar_values > 1.0]
        penalties = scalar_values[scalar_values < 1.0]
        neutrals = scalar_values[np.isclose(scalar_values, 1.0)]

        boost_mean = np.mean(boosts) if len(boosts) > 0 else 1.0
        penalty_mean = np.mean(penalties) if len(penalties) > 0 else 1.0
        neutral_mean = np.mean(neutrals) if len(neutrals) > 0 else 1.0

        triad_avg = (boost_mean + penalty_mean + neutral_mean) / 3.0

        harmony = self.PHI * triad_avg / (1.0 + self.PHI)

        return float(np.clip(harmony, 0.0, 1.0))

    def perform_bias_audit(self) -> dict[str, Any]:
        """Perform bias audit on current scalar distribution.

        Ensures 60/40 boost/penalty ratio for balanced operation.

        Returns:
            Audit results with recommendations
        """
        all_scalars = self._collect_all_scalars()
        scalar_values = np.array(list(all_scalars.values()))

        boost_count = np.sum(scalar_values > 1.0)
        penalty_count = np.sum(scalar_values < 1.0)
        neutral_count = np.sum(np.isclose(scalar_values, 1.0))
        total = len(scalar_values)

        if total == 0:
            return {
                "status": "no_scalars",
                "boost_ratio": 0.0,
                "penalty_ratio": 0.0,
                "recommendations": ["Register scalars before auditing"],
            }

        boost_ratio = boost_count / total
        penalty_ratio = penalty_count / total

        recommendations = []
        if boost_ratio > 0.65:
            recommendations.append("Consider reducing boost scalars for balance")
        elif boost_ratio < 0.55:
            recommendations.append("Consider increasing boost scalars for balance")

        ethical_snapshot = self.get_group_scalars(ScalarGroup.ETHICAL)
        omniempathy = ethical_snapshot.get("omniempathy", self.MIN_EMPATHY)
        omnimorality = ethical_snapshot.get("omnimorality", self.MIN_MORALITY)
        benevolence_threshold = ETHICAL.OMNIBENEVOLENCE_SCALAR
        omnibenevolence = ethical_snapshot.get("omnibenevolence", benevolence_threshold)

        if omniempathy < self.MIN_EMPATHY:
            recommendations.append(
                f"Omniempathy {omniempathy:.2f} below minimum {self.MIN_EMPATHY}"
            )
        if omnimorality < self.MIN_MORALITY:
            recommendations.append(
                f"Omnimorality {omnimorality:.2f} below minimum {self.MIN_MORALITY}"
            )
        if omnibenevolence < benevolence_threshold:
            recommendations.append(
                f"Omnibenevolence {omnibenevolence:.2f} below required "
                f"threshold {benevolence_threshold}"
            )

        return {
            "status": "passed" if not recommendations else "warnings",
            "total_scalars": total,
            "boost_count": int(boost_count),
            "penalty_count": int(penalty_count),
            "neutral_count": int(neutral_count),
            "boost_ratio": float(boost_ratio),
            "penalty_ratio": float(penalty_ratio),
            "target_boost_ratio": self.TARGET_BOOST_RATIO,
            "omniempathy": float(omniempathy),
            "omnimorality": float(omnimorality),
            "omnibenevolence": float(omnibenevolence),
            "recommendations": recommendations,
        }

    def get_scalar_statistics(self) -> dict[str, Any]:
        """Get comprehensive statistics about registered scalars.

        The per-group stats are split into operational and diagnostic
        measurement counts so the JSON output makes the σ_Immutable
        filter visible: ``total_scalars`` is operational, every group's
        ``count`` is operational, and ``count_metric_only`` exposes how
        many diagnostic measurement scalars are registered but excluded
        from the gate.  ``count_registered`` is the sum and matches what
        is discoverable via ``scalar_groups[group]``.
        """
        all_scalars = self._collect_all_scalars()

        if not all_scalars:
            return {
                "total_scalars": 0,
                "total_registered": 0,
                "total_metric_only": 0,
                "groups": {},
                "global_intelligence_score": 0.5,
            }

        group_stats = {}
        total_registered = 0
        total_metric_only = 0
        for group in ScalarGroup:
            group_scalars = self.get_group_scalars(group)
            if not group_scalars:
                continue
            operational = self._operational_scalars_for(group)
            metric_only = self._metric_only_scalars_for(group)
            total_registered += len(group_scalars)
            total_metric_only += len(metric_only)
            # Operational stats drive the gate; metric-only stats are
            # reported separately so operators can see what's registered
            # but not active without having to recompute the difference.
            operational_values = np.array(list(operational.values())) if operational else None
            group_stats[group.value] = {
                "count": len(operational),
                "count_registered": len(group_scalars),
                "count_metric_only": len(metric_only),
                "mean": (
                    float(np.mean(operational_values)) if operational_values is not None else 0.0
                ),
                "std": float(np.std(operational_values)) if operational_values is not None else 0.0,
                "min": float(np.min(operational_values)) if operational_values is not None else 0.0,
                "max": float(np.max(operational_values)) if operational_values is not None else 0.0,
            }

        return {
            "total_scalars": len(all_scalars),
            "total_registered": total_registered,
            "total_metric_only": total_metric_only,
            "registered_components": len(self.registered_scalars),
            "groups": group_stats,
            "global_intelligence_score": self.compute_global_intelligence_score(),
            "triadic_harmony": self.compute_triadic_harmony(),
            "bias_audit": self.perform_bias_audit(),
        }

    def ethical_observability(self) -> dict[str, float]:
        """Live ethical-gate snapshot for Prometheus export.

        Scores the current operational scalar vector through the ethical gate
        and reads the enforced benevolence scalar, so the deployment's
        ``mercury_agent_sigma_alignment`` and ``mercury_agent_benevolence_score``
        alerts (helm ``prometheusrule.yaml``) evaluate real series instead of
        sitting in ``NoData`` and silently failing to protect the gate.

        Read-only: it evaluates the gate purely to obtain the score and never
        mutates scalars nor emits the gate-trigger warning side effects that
        :meth:`get_enhanced_scalars` does.

        Returns:
            ``{"sigma_alignment": <current ethical-gate score>,
            "benevolence_score": <enforced omnibenevolence scalar>}``.
        """
        all_scalars = self._collect_all_scalars()
        scalar_vector = np.array(list(all_scalars.values()), dtype=np.float64)
        _passes, ethical_score = self.ethical_gate.evaluate(scalar_vector)
        benevolence = float(
            all_scalars.get(
                "omnibenevolence",
                self.get_scalar("omnibenevolence", ETHICAL.OMNIBENEVOLENCE_SCALAR),
            )
        )
        return {"sigma_alignment": float(ethical_score), "benevolence_score": benevolence}

    @classmethod
    def _is_metric_only_scalar(cls, key: str) -> bool:
        """Return True for diagnostic measurement scalars excluded from the σ vector.

        Single source of truth for the operational / diagnostic split.
        Consulted by every consumer that participates in the
        σ_Immutable layout contract (``_collect_all_scalars``,
        ``_prepare_dimensional_states``, ``compute_hierarchical_score``)
        as well as the reporting paths that surface the split to
        operators (``get_scalar_statistics``).
        """
        if key in cls._METRIC_ONLY_KEYS:
            return True
        return any(key.startswith(prefix) for prefix in cls._METRIC_ONLY_PREFIXES)

    def _quarantine_operational_outlier(self, group: ScalarGroup, name: str, value: float) -> None:
        """Log the gross-outlier tripwire WARNING once per (group, scalar).

        Includes the owning component (when the entry came through
        :meth:`register_scalars`) so the misregistration can be traced to
        its source instead of just to a scalar name.
        """
        key = (group, name)
        with self._registry_lock:
            if key in self._quarantine_warned:
                return
            self._quarantine_warned.add(key)
            owner = self._scalar_owners.get(key, "<built-in/unknown>")
        self.logger.warning(
            "GOSNN operational scalar %r in group %s (registered by %r) has value %r "
            "which is non-finite or grossly outside the expected normalized band "
            "(|value| > %.0e; σ_Immutable trained support is U[0, 2]); excluding it "
            "from the fusion/gate input. Fix the registering component — raw "
            "measurements (timestamps, counters) belong in the metric-only "
            "'omni_diag_' channel.",
            name,
            group.value,
            owner,
            value,
            self.OPERATIONAL_SCALAR_ABS_LIMIT,
        )

    def _operational_scalars_for(self, group: ScalarGroup) -> dict[str, float]:
        """Return the operational subset of one group's scalars.

        Operational scalars are those that participate in the
        σ_Immutable input vector, the dimensional-state fusion, and
        the hierarchical-score accountability bucket.  Diagnostic
        measurement families are filtered out via
        ``_is_metric_only_scalar``.

        Additionally applies the gross-outlier tripwire: a scalar that is
        non-finite or has ``|value| > OPERATIONAL_SCALAR_ABS_LIMIT``
        cannot be a normalized operational signal (it is a unit error —
        e.g. a raw unix timestamp) and is excluded with a once-per-scalar
        WARNING rather than silently clamped or silently fed to the gate.

        Centralising the filter here gives every operational call site
        a single helper to call — adding a new measurement family
        means updating ``_METRIC_ONLY_PREFIXES`` once and nothing else.
        Thread-safe: snapshots the group under the registry lock.
        """
        with self._registry_lock:
            items = list(self.scalar_groups[group].items())
        operational: dict[str, float] = {}
        for name, value in items:
            if self._is_metric_only_scalar(name):
                continue
            if not math.isfinite(value) or abs(value) > self.OPERATIONAL_SCALAR_ABS_LIMIT:
                self._quarantine_operational_outlier(group, name, value)
                continue
            operational[name] = value
        return operational

    def _metric_only_scalars_for(self, group: ScalarGroup) -> dict[str, float]:
        """Return the diagnostic measurement subset of one group's scalars.

        Counterpart to :meth:`_operational_scalars_for`.  Surfaces the
        scalars that ``_is_metric_only_scalar`` filters out so reporting
        paths (``get_scalar_statistics``) can show the operational /
        diagnostic split explicitly instead of letting a single
        ``count`` field hide the difference.
        Thread-safe: snapshots the group under the registry lock.
        """
        with self._registry_lock:
            items = list(self.scalar_groups[group].items())
        return {k: v for k, v in items if self._is_metric_only_scalar(k)}

    def critical_ethical_anchors(self) -> dict[str, float]:
        """Return the ETHICAL-group scalars subject to the hard ethical floor.

        These are the genuine civilization-first ethical anchors
        (benevolence, morality, empathy, justice, integrity,
        accountability, …) whose collapse is a categorical safety breach.
        Narrative/personality tuning scalars
        (:data:`_NARRATIVE_ETHICAL_SCALARS`) are excluded — they live in
        the ETHICAL group for historical reasons but tune output tone,
        not ethics, and their calibrated defaults are intentionally low.

        Consumed by :meth:`SigmaImmutableGate.enforce_ethical_floor` so
        the deterministic floor and the scalar definitions share a single
        source of truth (this method), never a duplicated name list.

        Returns:
            Mapping of anchor scalar name -> current value.
        """
        with self._registry_lock:
            ethical = dict(self.scalar_groups.get(ScalarGroup.ETHICAL, {}))
        return {
            name: float(value)
            for name, value in ethical.items()
            if name not in self._NARRATIVE_ETHICAL_SCALARS
        }

    def _collect_all_scalars(self) -> dict[str, float]:
        """Collect operational scalars (excludes diagnostic measurement scalars).

        Diagnostic ISO/IEC 25010, Halstead, McCabe + cognitive,
        Maintainability Index variants, NIST SAMATE, DORA, SLSA,
        supply-chain / repository-integrity, ISO/IEC 5055, and NIST SSDF
        entries live in
        ``scalar_groups`` but are filtered out here so the σ_Immutable
        trained gate continues to see the fixed operational layout it
        was trained on.  See the class-level ``_METRIC_ONLY_PREFIXES``
        comment for the contract.  Non-finite / gross-outlier entries are
        excluded by the same helper (see
        :meth:`_operational_scalars_for`).  The whole union is taken
        under the registry lock so concurrent registrations cannot
        produce a torn multi-group snapshot.
        """
        all_scalars: dict[str, float] = {}
        with self._registry_lock:
            for group in self.scalar_groups:
                all_scalars.update(self._operational_scalars_for(group))
        return all_scalars

    def _prepare_dimensional_states(
        self, base_scalars: dict[str, float], context: dict[str, Any]
    ) -> list[np.ndarray[Any, Any]]:
        """Prepare dimensional states for fusion (operational scalars only).

        Each per-group state vector contains only operational scalars;
        diagnostic measurement families are filtered out via
        :meth:`_operational_scalars_for` so the fusion pipeline's
        dimensional layout matches what the σ_Immutable gate sees.
        Previously this iterated the raw group dicts and relied on
        the implicit invariant that measurement scalars sit past the
        ``max_dimensions`` truncation horizon — that invariant is now
        explicit instead of accidental.  The per-group snapshots are
        taken under the registry lock so a concurrent registration
        cannot tear the multi-group layout mid-build.
        """
        states = []

        base_values = np.array(list(base_scalars.values()))
        states.append(base_values)

        with self._registry_lock:
            for group in ScalarGroup:
                group_scalars = self._operational_scalars_for(group)
                if group_scalars:
                    group_values = np.array(list(group_scalars.values()))
                    states.append(group_values)

        return states

    def _apply_enhancement(
        self,
        base_scalars: dict[str, float],
        fused_state: np.ndarray[Any, Any],
        ethical_score: float,
    ) -> dict[str, float]:
        """Apply enhancement to base scalars using fused state."""
        enhanced = {}

        fusion_factor = 1.0 + 0.1 * np.mean(fused_state)
        ethical_factor = ethical_score

        for i, (name, value) in enumerate(base_scalars.items()):
            enhancement = fused_state[i] * 0.1 if i < len(fused_state) else 0.0

            enhanced_value = value * fusion_factor * ethical_factor + enhancement
            enhanced[name] = float(enhanced_value)

        return enhanced

    def _compute_intelligence_contribution(self, enhanced_scalars: dict[str, float]) -> float:
        """Compute intelligence contribution from enhanced scalars."""
        if not enhanced_scalars:
            return 0.0

        values = np.array(list(enhanced_scalars.values()))
        contribution = np.mean(values) / (1.0 + np.std(values))
        return float(np.clip(contribution, 0.0, 1.0))

    def compute_hierarchical_score(
        self,
        domain_weights: dict[str, float] | None = None,
        aggregation_method: str = "geometric_mean",
    ) -> dict[str, Any]:
        """Compute hierarchical aggregation of omni-scalars (Phase 3).

        Implements 3-level hierarchical aggregation:

        Level 1 — Category groups: Group scalars by category
            (safety, fairness, transparency, accountability, beneficence)
        Level 2 — Category aggregation: Weighted mean within each group
            (weights set by domain priority)
        Level 3 — Cross-category: Final aggregation via geometric mean
            (penalizes any single low score)

        The mapping from ScalarGroups to the 5-category taxonomy:
            - safety: MEDICAL, SECURITY, HUMANITARIAN
            - fairness: ETHICAL (equity/justice subset)
            - transparency: ETHICAL (transparency/explainability subset)
            - accountability: SOFTWARE_ENGINEERING, ADVANCED_REASONING
            - beneficence: ETHICAL (benevolence/compassion subset), COSMIC

        Args:
            domain_weights: Optional per-category weights. If None, uses
                equal weighting. Keys: "safety", "fairness", "transparency",
                "accountability", "beneficence".
            aggregation_method: Cross-category aggregation method.
                "geometric_mean" (default, penalizes low scores),
                "arithmetic_mean", or "harmonic_mean".

        Returns:
            Dict with:
                - "overall_score": Final aggregated score in [0, 1]
                - "category_scores": Per-category scores
                - "category_sizes": Number of scalars per category
                - "method": Aggregation method used
        """
        # Default equal weights
        if domain_weights is None:
            domain_weights = {
                "safety": 1.0,
                "fairness": 1.0,
                "transparency": 1.0,
                "accountability": 1.0,
                "beneficence": 1.0,
            }

        # Level 1: Map scalars to categories
        categories: dict[str, list[float]] = {
            "safety": [],
            "fairness": [],
            "transparency": [],
            "accountability": [],
            "beneficence": [],
        }

        # Each bucket aggregates operational scalars only — diagnostic
        # measurement families are filtered out so the same architectural
        # split that protects the σ_Immutable input vector also protects
        # the meaning of each hierarchical category score.

        # Safety: Medical + Security + Humanitarian
        for group_key in [ScalarGroup.MEDICAL, ScalarGroup.SECURITY, ScalarGroup.HUMANITARIAN]:
            categories["safety"].extend(self._operational_scalars_for(group_key).values())

        # Fairness: Ethical equity/justice subset
        ethical = self._operational_scalars_for(ScalarGroup.ETHICAL)
        fairness_keys = [k for k in ethical if "equit" in k or "justic" in k or "bias" in k]
        categories["fairness"].extend(ethical[k] for k in fairness_keys)

        # Transparency: Ethical transparency/explainability subset
        transparency_keys = [
            k for k in ethical if "transparen" in k or "explain" in k or "account" in k
        ]
        categories["transparency"].extend(ethical[k] for k in transparency_keys)

        # Accountability: Software Engineering + Advanced Reasoning
        for group_key in [ScalarGroup.SOFTWARE_ENGINEERING, ScalarGroup.ADVANCED_REASONING]:
            categories["accountability"].extend(self._operational_scalars_for(group_key).values())

        # Beneficence: Ethical benevolence/compassion + Cosmic
        beneficence_keys = [
            k
            for k in ethical
            if "benevol" in k
            or "compass" in k
            or "love" in k
            or "empathy" in k
            or "altru" in k
            or "hope" in k
        ]
        categories["beneficence"].extend(ethical[k] for k in beneficence_keys)
        categories["beneficence"].extend(self._operational_scalars_for(ScalarGroup.COSMIC).values())

        # Level 2: Weighted mean within each category
        category_scores: dict[str, float] = {}
        category_sizes: dict[str, int] = {}

        for cat_name, values in categories.items():
            category_sizes[cat_name] = len(values)
            if values:
                arr = np.array(values)
                # Normalize: values > 1 are boosts, < 1 are penalties
                # Map to [0, 1] by dividing by max reasonable value (2.0)
                normalized = np.clip(arr / 2.0, 0.0, 1.0)
                category_scores[cat_name] = float(np.mean(normalized))
            else:
                category_scores[cat_name] = 0.5  # Neutral if empty

        # Level 3: Cross-category aggregation
        weighted_scores = []
        total_weight = 0.0
        for cat_name, score in category_scores.items():
            w = domain_weights.get(cat_name, 1.0)
            weighted_scores.append((score, w))
            total_weight += w

        if total_weight == 0 or not weighted_scores:
            overall = 0.5
        elif aggregation_method == "geometric_mean":
            # Weighted geometric mean: penalizes any single low score
            log_sum = sum(w * np.log(max(s, 1e-10)) for s, w in weighted_scores)
            overall = float(np.exp(log_sum / total_weight))
        elif aggregation_method == "harmonic_mean":
            # Weighted harmonic mean: even stronger penalty for low scores
            inv_sum = sum(w / max(s, 1e-10) for s, w in weighted_scores)
            overall = float(total_weight / inv_sum)
        else:
            # Arithmetic mean (default fallback)
            overall = float(sum(s * w for s, w in weighted_scores) / total_weight)

        overall = float(np.clip(overall, 0.0, 1.0))

        return {
            "overall_score": overall,
            "category_scores": category_scores,
            "category_sizes": category_sizes,
            "method": aggregation_method,
        }


# Global GOSNN singleton instance.
# Thread Safety: the accessor below serializes lazy creation under
# ``_global_network_lock``; the class itself serializes ``__init__`` under
# its own lock and guards the scalar registry with a per-instance RLock.
_global_network: GlobalOmniScalarNetwork | None = None
_global_network_lock = threading.Lock()

# Material-divergence WARNINGs already emitted by the accessor, keyed by the
# rendered diff string — each distinct ignored reconfiguration request is
# reported once per live instance, not once per call (the engine legitimately
# passes a per-request ``domain`` on every ``detect_with_fusion``).
_accessor_divergence_warned: set[str] = set()


def get_global_scalar_network(
    device: str | None = None,
    quantum_mode: bool | None = None,
    max_dimensions: int | None = None,
    domain: str | None = None,
    num_attention_heads: int | None = None,
    enable_triadic_phi: bool | None = None,
) -> GlobalOmniScalarNetwork:
    """Get the global GOSNN singleton instance.

    The GOSNN provides bidirectional synaptic integration with the 3R mechanism
    and other components. It uses 32-head attention with triadic phi-weighting
    for harmonic synergy and configurable sigma_Immutable threshold for ethical gating.

    Configuration arguments apply to the FIRST construction only.  Once a
    live instance exists this is an accessor: it returns the live instance
    and, when an explicitly requested parameter differs materially from
    the live configuration, logs a WARNING (once per distinct divergence)
    instead of silently ignoring the request.  Callers that genuinely need
    a differently configured network must call :func:`reset_global_network`
    first.  Direct construction of :class:`GlobalOmniScalarNetwork` with a
    materially different configuration raises ``ValueError`` instead —
    this accessor is deliberately non-raising because boundary code passes
    per-request hints (e.g. ``domain``) on every call.

    Args:
        device: Computation device ('cpu' or 'cuda'). Default 'cpu'.
        quantum_mode: Enable quantum-inspired operations. Default False.
        max_dimensions: Maximum dimensions for fusion. Default 37.
        domain: Domain identifier for threshold tuning (e.g., "medical" uses 0.93)
        num_attention_heads: Number of attention heads. Default 32 (head_dim=16).
        enable_triadic_phi: Enable triadic phi-weighting for harmonic synergy.
            Default True.

    Returns:
        GlobalOmniScalarNetwork singleton instance

    Note:
        The sigma_Immutable threshold can also be configured via the
        SIGMA_IMMUTABLE_THRESHOLD environment variable. Default is 0.96 for
        ~10-15% false positive reduction. Medical domains use 0.93 fallback.
    """
    global _global_network
    with _global_network_lock:
        live = _global_network or GlobalOmniScalarNetwork._instance
        if live is None or not getattr(live, "_initialized", False):
            try:
                _global_network = GlobalOmniScalarNetwork(
                    device=device,
                    quantum_mode=quantum_mode,
                    max_dimensions=max_dimensions,
                    domain=domain,
                    num_attention_heads=num_attention_heads,
                    enable_triadic_phi=enable_triadic_phi,
                )
                return _global_network
            except ValueError:
                # Raced a concurrent FIRST direct construction: the singleton
                # __new__ handed back the instance that call initialized, and
                # our materially different kwargs tripped the re-init check.
                # This accessor is documented non-raising, so recover to
                # accessor semantics (warn + return the live instance below).
                live = GlobalOmniScalarNetwork._instance
                if live is None or not getattr(live, "_initialized", False):
                    raise

        # Adopt an instance created via direct construction so both caches
        # agree, then surface (never silently swallow) any materially
        # different reconfiguration request.
        _global_network = live
        diffs = live._material_config_divergence(
            device=device,
            quantum_mode=quantum_mode,
            max_dimensions=max_dimensions,
            domain=domain,
            num_attention_heads=num_attention_heads,
            enable_triadic_phi=enable_triadic_phi,
        )
        if diffs:
            detail = ", ".join(
                f"{name}: live={have!r} requested={want!r}" for name, (have, want) in diffs.items()
            )
            if detail not in _accessor_divergence_warned:
                _accessor_divergence_warned.add(detail)
                logging.getLogger(__name__).warning(
                    "get_global_scalar_network() called with configuration that "
                    "differs materially from the live GOSNN singleton (%s); the "
                    "request is ignored and the live instance returned. Call "
                    "reset_global_network() first if you need a differently "
                    "configured network.",
                    detail,
                )
        return live


def reset_global_network() -> None:
    """Reset the global GOSNN instance (primarily for testing).

    Takes the module accessor lock and then the class construction lock (the
    same order the accessor implies, so no deadlock): clearing ``_instance``
    without the class lock could interleave with a concurrent ``__new__`` /
    ``__init__`` and leave one thread initializing an instance the registry
    no longer points at.
    """
    global _global_network
    with _global_network_lock:
        with GlobalOmniScalarNetwork._lock:
            GlobalOmniScalarNetwork._instance = None
        _global_network = None
        _accessor_divergence_warned.clear()

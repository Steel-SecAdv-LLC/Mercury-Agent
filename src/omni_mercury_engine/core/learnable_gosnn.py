"""
Mercury Agent - Learnable Global Omni-Scalar Network

Advanced GOSNN with learnable scalars, sparse attention, and
cross-domain correlation tracking for adaptive anomaly detection.

Features:
- Learnable scalar embeddings with gradient-based optimization
- Sparse attention for O(n log n) scalability
- Cross-domain scalar co-activation tracking
- Temporal scalar trajectory analysis
- Dynamic scalar adaptation based on domain context

Research References:
- Sparse Transformers: Child et al. (2019)
- Linformer: Wang et al. (2020) "Linformer: Self-Attention with Linear Complexity"
- Performer: Choromanski et al. (2021) "Rethinking Attention with Performers"
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, cast

import numpy as np
import numpy.typing as npt


logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.optim import Adam

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None
    Adam = None


from omni_mercury_engine.core.centralized_constants import (
    ETHICAL,
    LYAPUNOV,
    MATH,
)


PHI: float = MATH.GOLDEN_RATIO
LAMBDA_LYAPUNOV: float = LYAPUNOV.LAMBDA_CONVERGENCE
SIGMA_IMMUTABLE_DEFAULT: float = ETHICAL.SIGMA_IMMUTABLE_DEFAULT


class ScalarCategory(Enum):
    """Categories of learnable scalars."""

    ETHICAL = "ethical"
    COSMIC = "cosmic"
    QUANTUM_CONSCIOUSNESS = "quantum_consciousness"
    HUMANITARIAN = "humanitarian"
    SECURITY = "security"
    SOFTWARE_ENGINEERING = "software_engineering"
    MEDICAL = "medical"
    ADVANCED_REASONING = "advanced_reasoning"


@dataclass
class ScalarState:
    """State of a learnable scalar."""

    name: str
    category: ScalarCategory
    base_value: float
    learned_adjustment: float = 0.0
    gradient_history: list[float] = field(default_factory=list)
    activation_count: int = 0
    last_activation: float = 0.0
    co_activations: dict[str, int] = field(default_factory=dict)

    @property
    def effective_value(self) -> float:
        """Get effective scalar value."""
        return self.base_value + self.learned_adjustment


@dataclass
class TemporalTrajectory:
    """Temporal trajectory of scalar values."""

    scalar_name: str
    timestamps: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    trend: str = "stable"  # "increasing", "decreasing", "stable", "oscillating"
    volatility: float = 0.0


if TORCH_AVAILABLE:

    class LearnableScalarEmbedding(nn.Module):
        """Learnable embeddings for GOSNN scalars."""

        def __init__(
            self,
            n_scalars: int = 180,
            embedding_dim: int = 64,
            n_categories: int = 8,
        ):
            super().__init__()
            self.n_scalars = n_scalars
            self.embedding_dim = embedding_dim

            self.scalar_embeddings = nn.Parameter(torch.randn(n_scalars, embedding_dim) * 0.1)

            self.category_embeddings = nn.Embedding(n_categories, embedding_dim)

            self.base_values = nn.Parameter(torch.ones(n_scalars))

            self.learned_phi = nn.Parameter(torch.tensor(PHI))
            self.learned_lambda = nn.Parameter(torch.tensor(LAMBDA_LYAPUNOV))
            self.learned_sigma = nn.Parameter(torch.tensor(SIGMA_IMMUTABLE_DEFAULT))

            self.value_projector = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim // 2),
                nn.ReLU(),
                nn.Linear(embedding_dim // 2, 1),
            )

        def forward(
            self,
            scalar_indices: torch.Tensor | None = None,
            category_indices: torch.Tensor | None = None,
        ) -> dict[str, torch.Tensor]:
            """Forward pass computing scalar values from embeddings."""
            if scalar_indices is None:
                scalar_emb = self.scalar_embeddings
            else:
                scalar_emb = self.scalar_embeddings[scalar_indices]

            if category_indices is not None:
                cat_emb = self.category_embeddings(category_indices)
                scalar_emb = scalar_emb + cat_emb

            adjustments = self.value_projector(scalar_emb).squeeze(-1)

            effective_values = self.base_values + adjustments * 0.1

            return {
                "embeddings": scalar_emb,
                "base_values": self.base_values,
                "adjustments": adjustments,
                "effective_values": effective_values,
                "phi": self.learned_phi,
                "lambda_lyapunov": self.learned_lambda,
                "sigma_immutable": torch.sigmoid(self.learned_sigma) * 0.06 + 0.93,
            }

        def get_scalar_similarity(
            self,
            idx1: int,
            idx2: int,
        ) -> float:
            """Compute similarity between two scalars."""
            emb1 = self.scalar_embeddings[idx1]
            emb2 = self.scalar_embeddings[idx2]
            return float(F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)))

    class SparseAttention(nn.Module):
        """
        Sparse attention for efficient scalar fusion.

        Uses local + strided attention pattern for O(n sqrt(n)) complexity.
        """

        def __init__(
            self,
            d_model: int = 64,
            n_heads: int = 8,
            window_size: int = 16,
            stride: int = 16,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.d_model = d_model
            self.n_heads = n_heads
            self.head_dim = d_model // n_heads
            self.window_size = window_size
            self.stride = stride

            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
            self.o_proj = nn.Linear(d_model, d_model)

            self.dropout = nn.Dropout(dropout)
            self.scale = math.sqrt(self.head_dim)

        def forward(
            self,
            x: torch.Tensor,
            mask: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Apply sparse attention.

            Args:
                x: Input tensor [batch, seq_len, d_model]
                mask: Optional attention mask

            Returns:
                Tuple of (output, attention_weights)
            """
            batch_size, seq_len, _ = x.shape

            Q = self.q_proj(x)
            K = self.k_proj(x)
            V = self.v_proj(x)

            Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

            sparse_mask = self._create_sparse_mask(seq_len, x.device)

            attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

            attn_weights = attn_weights.masked_fill(~sparse_mask, float("-inf"))

            if mask is not None:
                attn_weights = attn_weights.masked_fill(~mask.unsqueeze(1), float("-inf"))

            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights)

            output = torch.matmul(attn_weights, V)
            output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
            output = self.o_proj(output)

            return output, attn_weights

        def _create_sparse_mask(
            self,
            seq_len: int,
            device: torch.device,
        ) -> torch.Tensor:
            """Create sparse attention mask (local + strided)."""
            mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

            for i in range(seq_len):
                start = max(0, i - self.window_size // 2)
                end = min(seq_len, i + self.window_size // 2 + 1)
                mask[i, start:end] = True

            for i in range(seq_len):
                for j in range(0, seq_len, self.stride):
                    mask[i, j] = True
                    mask[j, i] = True

            return mask.unsqueeze(0).unsqueeze(0)

    class LinformerAttention(nn.Module):
        """
        Linformer-style linear attention for O(n) complexity.

        Projects key and value to lower dimension before attention.
        """

        def __init__(
            self,
            d_model: int = 64,
            n_heads: int = 8,
            k_dim: int = 32,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.d_model = d_model
            self.n_heads = n_heads
            self.head_dim = d_model // n_heads
            self.k_dim = k_dim

            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
            self.o_proj = nn.Linear(d_model, d_model)

            self.E = nn.Parameter(torch.randn(k_dim, 256) / math.sqrt(k_dim))
            self.F = nn.Parameter(torch.randn(k_dim, 256) / math.sqrt(k_dim))

            self.dropout = nn.Dropout(dropout)
            self.scale = math.sqrt(self.head_dim)

        def forward(
            self,
            x: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Apply linear attention."""
            batch_size, seq_len, _ = x.shape

            Q = self.q_proj(x)
            K = self.k_proj(x)
            V = self.v_proj(x)

            Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

            E = self.E[:, :seq_len]
            F_proj = self.F[:, :seq_len]

            K_proj = torch.einsum("bhsd,ks->bhkd", K, E)
            V_proj = torch.einsum("bhsd,ks->bhkd", V, F_proj)

            attn_weights = torch.matmul(Q, K_proj.transpose(-2, -1)) / self.scale
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights)

            output = torch.matmul(attn_weights, V_proj)
            output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
            output = self.o_proj(output)

            return output, attn_weights

    class CrossDomainCorrelation(nn.Module):
        """Track and learn cross-domain scalar correlations."""

        def __init__(
            self,
            n_scalars: int = 180,
            embedding_dim: int = 64,
        ):
            super().__init__()
            self.n_scalars = n_scalars

            self.correlation_matrix = nn.Parameter(
                torch.eye(n_scalars) * 0.1,
                requires_grad=False,
            )

            self.correlation_learner = nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, 1),
                nn.Sigmoid(),
            )

        def update_correlation(
            self,
            idx1: int,
            idx2: int,
            co_activation: bool,
            learning_rate: float = 0.01,
        ) -> None:
            """Update correlation matrix based on co-activation."""
            with torch.no_grad():
                current = self.correlation_matrix[idx1, idx2]
                target = 1.0 if co_activation else 0.0
                new_value = current + learning_rate * (target - current)
                self.correlation_matrix[idx1, idx2] = new_value
                self.correlation_matrix[idx2, idx1] = new_value

        def get_correlated_scalars(
            self,
            idx: int,
            threshold: float = 0.5,
        ) -> list[tuple[int, float]]:
            """Get scalars correlated with the given scalar."""
            correlations = self.correlation_matrix[idx]
            mask = correlations > threshold
            indices = torch.where(mask)[0]
            return [(int(i), float(correlations[i])) for i in indices if i != idx]

        def forward(
            self,
            emb1: torch.Tensor,
            emb2: torch.Tensor,
        ) -> torch.Tensor:
            """Predict correlation between two scalar embeddings."""
            combined = torch.cat([emb1, emb2], dim=-1)
            return self.correlation_learner(combined)


class LearnableGOSNN:
    """
    Learnable Global Omni-Scalar Network.

    Extends base GOSNN with:
    - Learnable scalar embeddings
    - Sparse attention fusion
    - Cross-domain correlation tracking
    - Temporal trajectory analysis
    - Gradient-based optimization
    """

    def __init__(
        self,
        n_scalars: int = 180,
        embedding_dim: int = 64,
        n_heads: int = 8,
        attention_type: str = "sparse",
        device: str = "cpu",
    ):
        self.n_scalars = n_scalars
        self.embedding_dim = embedding_dim
        self.device = device
        self.attention_type = attention_type

        self.scalar_states: dict[str, ScalarState] = {}
        self._scalar_name_to_idx: dict[str, int] = {}
        self._idx_to_scalar_name: dict[int, str] = {}
        self._trajectories: dict[str, TemporalTrajectory] = {}

        self._lock = threading.RLock()
        self._optimization_step = 0

        if TORCH_AVAILABLE:
            self.scalar_embeddings = LearnableScalarEmbedding(
                n_scalars=n_scalars,
                embedding_dim=embedding_dim,
            ).to(device)

            if attention_type == "sparse":
                self.attention = SparseAttention(
                    d_model=embedding_dim,
                    n_heads=n_heads,
                ).to(device)
            elif attention_type == "linear":
                self.attention = LinformerAttention(
                    d_model=embedding_dim,
                    n_heads=n_heads,
                ).to(device)
            else:
                self.attention = nn.MultiheadAttention(
                    embed_dim=embedding_dim,
                    num_heads=n_heads,
                    batch_first=True,
                ).to(device)

            self.correlation_tracker = CrossDomainCorrelation(
                n_scalars=n_scalars,
                embedding_dim=embedding_dim,
            ).to(device)

            self.optimizer = Adam(
                list(self.scalar_embeddings.parameters()) + list(self.attention.parameters()),
                lr=0.001,
            )
        else:
            self.scalar_embeddings = None
            self.attention = None
            self.correlation_tracker = None
            self.optimizer = None

        self._initialize_default_scalars()

        logger.info(
            f"LearnableGOSNN initialized " f"(n_scalars={n_scalars}, attention={attention_type})"
        )

    def _initialize_default_scalars(self) -> None:
        """Initialize default scalar states."""
        default_scalars = {
            ScalarCategory.ETHICAL: [
                ("omnimorality", 1.20),
                ("omniempathy", 1.22),
                ("omnibenevolence", 0.99),
                ("omnijustice", 1.30),
                ("omniintegrity", 1.30),
            ],
            ScalarCategory.SOFTWARE_ENGINEERING: [
                ("omni_code_complexity", 1.20),
                ("omni_type_safety_index", 1.28),
                ("omni_3r_synergy_factor", 1.35),
                ("omni_precision_recall_harmonic", 1.25),
            ],
            ScalarCategory.MEDICAL: [
                ("omni_diagnostic_accuracy", 1.30),
                ("omni_patient_safety", 1.40),
                ("omni_clinical_explainability", 1.28),
            ],
            ScalarCategory.SECURITY: [
                ("omni_threat_detection", 1.25),
                ("omni_quantum_resistance", 1.30),
                ("omni_zero_trust", 1.22),
            ],
        }

        idx = 0
        for category, scalars in default_scalars.items():
            for name, base_value in scalars:
                self.register_scalar(name, category, base_value)
                idx += 1

    def register_scalar(
        self,
        name: str,
        category: ScalarCategory,
        base_value: float,
    ) -> None:
        """Register a scalar in the network."""
        with self._lock:
            idx = len(self._scalar_name_to_idx)
            self._scalar_name_to_idx[name] = idx
            self._idx_to_scalar_name[idx] = name

            self.scalar_states[name] = ScalarState(
                name=name,
                category=category,
                base_value=base_value,
            )

            self._trajectories[name] = TemporalTrajectory(scalar_name=name)

    def get_scalar_value(self, name: str) -> float:
        """Get effective scalar value."""
        if name not in self.scalar_states:
            return 1.0

        state = self.scalar_states[name]
        return state.effective_value

    def get_scalar_embedding(self, name: str) -> npt.NDArray[Any] | None:
        """Get scalar embedding vector."""
        if not TORCH_AVAILABLE or self.scalar_embeddings is None:
            return None

        if name not in self._scalar_name_to_idx:
            return None

        idx = self._scalar_name_to_idx[name]
        with torch.no_grad():
            emb = self.scalar_embeddings.scalar_embeddings[idx].cpu().numpy()
        return emb

    def activate_scalars(
        self,
        scalar_names: list[str],
        context: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """Activate a set of scalars and track co-activations."""
        with self._lock:
            timestamp = time.time()
            activated = {}

            for name in scalar_names:
                if name not in self.scalar_states:
                    continue

                state = self.scalar_states[name]
                state.activation_count += 1
                state.last_activation = timestamp

                trajectory = self._trajectories[name]
                trajectory.timestamps.append(timestamp)
                trajectory.values.append(state.effective_value)

                for other_name in scalar_names:
                    if other_name != name:
                        state.co_activations[other_name] = (
                            state.co_activations.get(other_name, 0) + 1
                        )

                        if self.correlation_tracker is not None:
                            idx1 = self._scalar_name_to_idx.get(name)
                            idx2 = self._scalar_name_to_idx.get(other_name)
                            if idx1 is not None and idx2 is not None:
                                self.correlation_tracker.update_correlation(idx1, idx2, True)

                activated[name] = state.effective_value

            return activated

    def fuse_scalars(
        self,
        scalar_values: dict[str, float],
        use_attention: bool = True,
    ) -> dict[str, Any]:
        """Fuse scalar values using attention mechanism."""
        if not TORCH_AVAILABLE or self.scalar_embeddings is None:
            return self._numpy_fusion(scalar_values)

        names = list(scalar_values.keys())
        indices = []
        values = []

        for name in names:
            if name in self._scalar_name_to_idx:
                indices.append(self._scalar_name_to_idx[name])
                values.append(scalar_values[name])

        if not indices:
            return {
                "fused_score": 0.5,
                "attention_weights": {},
                "harmonic_synergy": 0.5,
            }

        indices_tensor = torch.tensor(indices, device=self.device)
        values_tensor = torch.tensor(values, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            scalar_output = self.scalar_embeddings(scalar_indices=indices_tensor)
            embeddings = scalar_output["embeddings"]

            if use_attention and self.attention is not None:
                embeddings_seq = embeddings.unsqueeze(0)

                if isinstance(self.attention, (SparseAttention, LinformerAttention)):
                    attn_output, attn_weights = self.attention(embeddings_seq)
                else:
                    attn_output, attn_weights = self.attention(
                        embeddings_seq, embeddings_seq, embeddings_seq
                    )

                attn_output = attn_output.squeeze(0)
            else:
                attn_output = embeddings
                attn_weights = None

            output_norms = torch.norm(attn_output, dim=-1)
            weighted_values = values_tensor * output_norms / (output_norms.sum() + 1e-8)
            fused_score = weighted_values.sum().item()

            phi = scalar_output["phi"].item()
            sigma = scalar_output["sigma_immutable"].item()

            fused_score = fused_score * (sigma**phi)

            fft_result = torch.fft.fft(attn_output.flatten())
            magnitudes = torch.abs(fft_result)
            if len(magnitudes) > 1:
                sorted_mags = torch.sort(magnitudes, descending=True).values
                if sorted_mags[1] > 0:
                    ratio = sorted_mags[0] / sorted_mags[1]
                    harmonic_synergy = float(1.0 / (1.0 + abs(ratio - phi)))
                else:
                    harmonic_synergy = 0.5
            else:
                harmonic_synergy = 0.5

        attention_dict = {}
        if attn_weights is not None:
            attn_numpy = attn_weights.cpu().numpy()
            for i, name in enumerate(names[: len(indices)]):
                attention_dict[name] = float(attn_numpy[0, 0, i, :].mean())

        return {
            "fused_score": float(fused_score),
            "attention_weights": attention_dict,
            "harmonic_synergy": float(harmonic_synergy),
            "learned_phi": float(phi),
            "learned_sigma": float(sigma),
        }

    def _numpy_fusion(self, scalar_values: dict[str, float]) -> dict[str, Any]:
        """NumPy fallback for scalar fusion."""
        values = list(scalar_values.values())
        if not values:
            return {
                "fused_score": 0.5,
                "attention_weights": {},
                "harmonic_synergy": 0.5,
            }

        arr = np.array(values)
        fused_score = float(np.mean(arr) * (SIGMA_IMMUTABLE_DEFAULT**PHI))

        fft_result = np.fft.fft(arr)
        magnitudes = np.abs(fft_result)
        if len(magnitudes) > 1:
            sorted_mags = np.sort(magnitudes)[::-1]
            if sorted_mags[1] > 0:
                ratio = sorted_mags[0] / sorted_mags[1]
                harmonic_synergy = float(1.0 / (1.0 + abs(ratio - PHI)))
            else:
                harmonic_synergy = 0.5
        else:
            harmonic_synergy = 0.5

        return {
            "fused_score": fused_score,
            "attention_weights": {name: 1.0 / len(values) for name in scalar_values},
            "harmonic_synergy": harmonic_synergy,
        }

    def learn_from_feedback(
        self,
        scalar_names: list[str],
        target_score: float,
        actual_score: float,
    ) -> float:
        """Update scalar embeddings based on feedback."""
        if not TORCH_AVAILABLE or self.optimizer is None:
            return 0.0

        indices = []
        for name in scalar_names:
            if name in self._scalar_name_to_idx:
                indices.append(self._scalar_name_to_idx[name])

        if not indices:
            return 0.0

        indices_tensor = torch.tensor(indices, device=self.device)

        self.optimizer.zero_grad()

        scalar_output = self.scalar_embeddings(scalar_indices=indices_tensor)
        effective_values = scalar_output["effective_values"]

        predicted = effective_values.mean()
        target = torch.tensor(target_score, dtype=torch.float32, device=self.device)

        loss = F.mse_loss(predicted, target)
        loss.backward()

        self.optimizer.step()
        self._optimization_step += 1

        loss_value = float(loss.item())

        for name in scalar_names:
            if name in self.scalar_states:
                state = self.scalar_states[name]
                state.gradient_history.append(loss_value)
                if len(state.gradient_history) > 100:
                    state.gradient_history = state.gradient_history[-100:]

        return loss_value

    def analyze_trajectory(self, scalar_name: str) -> dict[str, Any]:
        """Analyze temporal trajectory of a scalar."""
        if scalar_name not in self._trajectories:
            return {"found": False}

        trajectory = self._trajectories[scalar_name]

        if len(trajectory.values) < 3:
            return {
                "found": True,
                "scalar": scalar_name,
                "n_samples": len(trajectory.values),
                "trend": "insufficient_data",
            }

        values = np.array(trajectory.values[-100:])
        _timestamps = np.array(trajectory.timestamps[-100:])

        if len(values) >= 2:
            coeffs = np.polyfit(range(len(values)), values, 1)
            slope = coeffs[0]

            if slope > 0.01:
                trend = "increasing"
            elif slope < -0.01:
                trend = "decreasing"
            else:
                diff = np.diff(values)
                sign_changes = np.sum(diff[:-1] * diff[1:] < 0)
                if sign_changes > len(values) // 4:
                    trend = "oscillating"
                else:
                    trend = "stable"
        else:
            trend = "stable"
            slope = 0.0

        volatility = float(np.std(values))

        trajectory.trend = trend
        trajectory.volatility = volatility

        return {
            "found": True,
            "scalar": scalar_name,
            "n_samples": len(values),
            "trend": trend,
            "slope": float(slope),
            "volatility": volatility,
            "mean": float(np.mean(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "current": float(values[-1]) if len(values) > 0 else None,
        }

    def get_correlated_scalars(
        self,
        scalar_name: str,
        min_correlation: float = 0.3,
    ) -> list[tuple[str, float]]:
        """Get scalars correlated with the given scalar."""
        if scalar_name not in self._scalar_name_to_idx:
            return []

        if self.correlation_tracker is None:
            state = self.scalar_states.get(scalar_name)
            if not state:
                return []

            results = []
            for other, count in state.co_activations.items():
                correlation = min(count / (state.activation_count + 1), 1.0)
                if correlation >= min_correlation:
                    results.append((other, correlation))

            results.sort(key=lambda x: x[1], reverse=True)
            return results

        idx = self._scalar_name_to_idx[scalar_name]
        correlated = self.correlation_tracker.get_correlated_scalars(idx, threshold=min_correlation)

        return [(self._idx_to_scalar_name.get(i, f"scalar_{i}"), corr) for i, corr in correlated]

    def get_statistics(self) -> dict[str, Any]:
        """Get network statistics."""
        return {
            "n_registered_scalars": len(self.scalar_states),
            "n_scalars_capacity": self.n_scalars,
            "embedding_dim": self.embedding_dim,
            "attention_type": self.attention_type,
            "optimization_steps": self._optimization_step,
            "torch_available": TORCH_AVAILABLE,
            "device": self.device,
            "most_activated": sorted(
                [(name, state.activation_count) for name, state in self.scalar_states.items()],
                key=lambda x: x[1],
                reverse=True,
            )[:10],
        }

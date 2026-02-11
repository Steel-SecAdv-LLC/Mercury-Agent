"""
Mercury Agent - Learnable 3R Fusion Module

State-of-the-art differentiable 3R (Recursion-Resonance-Refactoring) fusion
with learnable weights, dynamic gating, and spectral attention.

Features:
- Backprop-trainable fusion weights (w_R, w_H, w_O)
- Dynamic ethical gate function
- Multi-scale recursion with adaptive depth
- Wavelet + attention for multi-resolution spectral analysis
- End-to-end differentiable anomaly fusion

Mathematical Framework:
A = (w_R * R(x) + w_H * H(ω) + w_O * O(θ)) * η_gate(x)^Φ

Where:
- w_R, w_H, w_O are learnable weights (softmax normalized)
- η_gate is a learned ethical gating function
- Φ is the golden ratio (optionally learnable)

Research References:
- Attention Is All You Need (Vaswani et al., 2017)
- Wavelet Attention Networks (Li et al., 2021)
- Deep Sets (Zaheer et al., 2017)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.optim import AdamW

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


from omni_mercury_engine.core.three_r.types import (
    CONVERGENCE_RATE_PARAMETER,
    GOLDEN_RATIO_CONSTANT,
)

PHI = GOLDEN_RATIO_CONSTANT
LAMBDA = CONVERGENCE_RATE_PARAMETER


@dataclass
class Learnable3RConfig:
    """Configuration for Learnable 3R Fusion."""

    hidden_dim: int = 64
    n_attention_heads: int = 4
    max_recursion_depth: int = 10
    n_wavelet_scales: int = 8
    dropout: float = 0.1
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    phi_learnable: bool = True
    ethical_gate_hidden: int = 32
    use_spectral_attention: bool = True


@dataclass
class Learnable3RResult:
    """Result from learnable 3R fusion."""

    fusion_score: float
    recursion_score: float
    resonance_score: float
    optimization_score: float
    ethical_gate_output: float
    learned_weights: dict[str, float]
    learned_phi: float
    attention_weights: dict[str, Any] = field(default_factory=dict)
    lyapunov_bound: float = 0.0
    is_stable: bool = True
    loss: float = 0.0


if TORCH_AVAILABLE:

    class DynamicEthicalGate(nn.Module):
        """
        Learned ethical gating function.

        Replaces static η threshold with a learned gate that
        considers input context for adaptive ethical gating.
        """

        def __init__(
            self,
            input_dim: int = 3,
            hidden_dim: int = 32,
            min_gate: float = 0.93,
            max_gate: float = 0.99,
        ):
            super().__init__()
            self.min_gate = min_gate
            self.max_gate = max_gate

            self.gate_network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),
            )

            self.context_encoder = nn.LSTM(
                input_size=1,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
            )

        def forward(
            self,
            scores: torch.Tensor,
            context: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """
            Compute dynamic ethical gate.

            Args:
                scores: [batch, 3] tensor of (R, H, O) scores
                context: Optional context tensor

            Returns:
                Gate values in [min_gate, max_gate]
            """
            if context is not None:
                context = context.unsqueeze(-1)
                _, (hidden, _) = self.context_encoder(context)
                hidden = hidden.squeeze(0)
                scores = torch.cat([scores, hidden], dim=-1)

            raw_gate = self.gate_network(scores)
            scaled_gate = self.min_gate + raw_gate * (self.max_gate - self.min_gate)

            return scaled_gate  # type: ignore[no-any-return]

    class MultiScaleRecursion(nn.Module):
        """
        Multi-scale recursion with adaptive depth.

        Learns optimal recursion depth based on input complexity.
        """

        def __init__(
            self,
            input_dim: int = 64,
            hidden_dim: int = 64,
            max_depth: int = 10,
        ):
            super().__init__()
            self.max_depth = max_depth

            self.encoder = nn.Linear(input_dim, hidden_dim)

            self.recursion_cell = nn.GRUCell(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
            )

            self.depth_predictor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),
            )

            self.output_projection = nn.Linear(hidden_dim, 1)

        def forward(
            self,
            x: torch.Tensor,
            return_trajectory: bool = False,
        ) -> tuple[torch.Tensor, dict[str, Any]]:
            """
            Apply multi-scale recursion.

            Args:
                x: Input tensor [batch, seq_len]
                return_trajectory: Return intermediate states

            Returns:
                Recursion score and metadata
            """
            if x.dim() == 1:
                x = x.unsqueeze(0)

            if x.shape[-1] != self.encoder.in_features:
                x_padded = F.pad(x, (0, self.encoder.in_features - x.shape[-1]))
            else:
                x_padded = x

            h = self.encoder(x_padded)
            trajectory = [h.clone()]

            for depth in range(self.max_depth):
                h = self.recursion_cell(h, h)
                trajectory.append(h.clone())

                predicted_depth = self.depth_predictor(h)
                if (predicted_depth < 0.1).all():
                    break

            score = torch.sigmoid(self.output_projection(h)).squeeze(-1)

            meta = {
                "actual_depth": depth + 1,
                "trajectory": trajectory if return_trajectory else None,
            }

            return score, meta

    class SpectralAttention(nn.Module):
        """
        Spectral attention with wavelet decomposition.

        Combines FFT analysis with multi-scale wavelet attention
        for comprehensive frequency domain understanding.
        """

        def __init__(
            self,
            n_scales: int = 8,
            hidden_dim: int = 64,
            n_heads: int = 4,
        ):
            super().__init__()
            self.n_scales = n_scales

            self.scale_encoders = nn.ModuleList(
                [nn.Linear(64, hidden_dim) for _ in range(n_scales)]
            )

            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=n_heads,
                batch_first=True,
            )

            self.fft_encoder = nn.Sequential(
                nn.Linear(64, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

            self.output_projection = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )

        def forward(
            self,
            x: torch.Tensor,
        ) -> tuple[torch.Tensor, dict[str, Any]]:
            """
            Apply spectral attention.

            Args:
                x: Input tensor [batch, seq_len]

            Returns:
                Resonance score and attention weights
            """
            if x.dim() == 1:
                x = x.unsqueeze(0)

            batch_size, seq_len = x.shape

            fft_result = torch.fft.fft(x, dim=-1)
            fft_magnitude = torch.abs(fft_result)

            fft_padded = F.pad(
                fft_magnitude[:, :32],
                (0, 64 - min(32, seq_len)),
            )
            fft_encoded = self.fft_encoder(fft_padded)

            wavelet_features = []
            for i, encoder in enumerate(self.scale_encoders):
                scale = 2**i
                if scale < seq_len:
                    downsampled = F.avg_pool1d(
                        x.unsqueeze(1),
                        kernel_size=min(scale, seq_len),
                        stride=max(1, scale // 2),
                        padding=0,
                    ).squeeze(1)

                    padded = F.pad(downsampled, (0, 64 - downsampled.shape[-1]))
                    encoded = encoder(padded)
                    wavelet_features.append(encoded.unsqueeze(1))

            if wavelet_features:
                wavelet_stack = torch.cat(wavelet_features, dim=1)
                attn_output, attn_weights = self.attention(
                    wavelet_stack, wavelet_stack, wavelet_stack
                )
                wavelet_combined = attn_output.mean(dim=1)
            else:
                wavelet_combined = torch.zeros_like(fft_encoded)
                attn_weights = None

            combined = torch.cat([fft_encoded, wavelet_combined], dim=-1)
            score = self.output_projection(combined).squeeze(-1)

            meta = {
                "fft_magnitude": fft_magnitude,
                "attention_weights": attn_weights,
                "n_scales_used": len(wavelet_features),
            }

            return score, meta

    class OptimizationScorer(nn.Module):
        """
        Learned optimization/refactoring scorer.

        Evaluates optimization potential based on signal characteristics.
        """

        def __init__(
            self,
            input_dim: int = 64,
            hidden_dim: int = 64,
        ):
            super().__init__()

            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
            )

            self.snr_estimator = nn.Linear(hidden_dim // 2, 1)
            self.complexity_estimator = nn.Linear(hidden_dim // 2, 1)
            self.quality_estimator = nn.Linear(hidden_dim // 2, 1)

            self.combiner = nn.Sequential(
                nn.Linear(3, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid(),
            )

        def forward(
            self,
            x: torch.Tensor,
        ) -> tuple[torch.Tensor, dict[str, Any]]:
            """
            Compute optimization score.

            Args:
                x: Input tensor [batch, seq_len]

            Returns:
                Optimization score and component scores
            """
            if x.dim() == 1:
                x = x.unsqueeze(0)

            x_padded = F.pad(x, (0, max(0, 64 - x.shape[-1])))
            x_padded = x_padded[:, :64]

            encoded = self.encoder(x_padded)

            snr = torch.sigmoid(self.snr_estimator(encoded))
            complexity = torch.sigmoid(self.complexity_estimator(encoded))
            quality = torch.sigmoid(self.quality_estimator(encoded))

            combined = torch.cat([snr, complexity, quality], dim=-1)
            score = self.combiner(combined).squeeze(-1)

            meta = {
                "snr_estimate": snr.squeeze(-1),
                "complexity_estimate": complexity.squeeze(-1),
                "quality_estimate": quality.squeeze(-1),
            }

            return score, meta

    class Learnable3RFusion(nn.Module):
        """
        Complete learnable 3R fusion module.

        Combines multi-scale recursion, spectral attention, and
        optimization scoring with learnable weights and dynamic gating.
        """

        def __init__(
            self,
            config: Learnable3RConfig | None = None,
        ):
            super().__init__()

            config = config or Learnable3RConfig()
            self.config = config

            self.log_w_R = nn.Parameter(torch.tensor(math.log(1.0 / PHI)))
            self.log_w_H = nn.Parameter(torch.tensor(math.log(1.0 / (PHI**2))))
            self.log_w_O = nn.Parameter(torch.tensor(math.log(0.236)))

            if config.phi_learnable:
                self.phi = nn.Parameter(torch.tensor(PHI))
            else:
                self.register_buffer("phi", torch.tensor(PHI))

            self.recursion_module = MultiScaleRecursion(
                hidden_dim=config.hidden_dim,
                max_depth=config.max_recursion_depth,
            )

            self.resonance_module = (
                SpectralAttention(
                    n_scales=config.n_wavelet_scales,
                    hidden_dim=config.hidden_dim,
                    n_heads=config.n_attention_heads,
                )
                if config.use_spectral_attention
                else None
            )

            self.optimization_module = OptimizationScorer(
                hidden_dim=config.hidden_dim,
            )

            self.ethical_gate = DynamicEthicalGate(
                input_dim=3,
                hidden_dim=config.ethical_gate_hidden,
            )

            self.time_step = 0
            self.convergence_history: list[float] = []

        def get_normalized_weights(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Get softmax-normalized fusion weights."""
            log_weights = torch.stack([self.log_w_R, self.log_w_H, self.log_w_O])
            weights = F.softmax(log_weights, dim=0)
            return weights[0], weights[1], weights[2]

        def forward(
            self,
            x: torch.Tensor,
            context: torch.Tensor | None = None,
            return_components: bool = True,
        ) -> dict[str, Any]:
            """
            Compute learnable 3R fusion.

            Args:
                x: Input tensor [batch, seq_len]
                context: Optional context for ethical gating
                return_components: Return component scores

            Returns:
                Dictionary with fusion results
            """
            if x.dim() == 1:
                x = x.unsqueeze(0)

            R_score, R_meta = self.recursion_module(x)

            if self.resonance_module is not None:
                H_score, H_meta = self.resonance_module(x)
            else:
                H_score = self._simple_resonance(x)
                H_meta = {}

            O_score, O_meta = self.optimization_module(x)

            w_R, w_H, w_O = self.get_normalized_weights()

            weighted_sum = w_R * R_score + w_H * H_score + w_O * O_score

            scores_stack = torch.stack([R_score, H_score, O_score], dim=-1)
            eta = self.ethical_gate(scores_stack, context)

            phi = self.phi if isinstance(self.phi, torch.Tensor) else torch.tensor(self.phi)
            ethical_scaling = eta**phi

            fusion_score = weighted_sum * ethical_scaling.squeeze(-1)

            self.time_step += 1
            epsilon = 1.0
            lyapunov_bound = epsilon * math.exp(-LAMBDA * self.time_step)

            result = {
                "fusion_score": fusion_score,
                "ethical_gate": eta.squeeze(-1),
                "lyapunov_bound": lyapunov_bound,
                "weights": {
                    "w_R": w_R.item(),
                    "w_H": w_H.item(),
                    "w_O": w_O.item(),
                },
                "phi": phi.item() if isinstance(phi, torch.Tensor) else phi,
            }

            if return_components:
                result.update(
                    {
                        "recursion_score": R_score,
                        "resonance_score": H_score,
                        "optimization_score": O_score,
                        "recursion_meta": R_meta,
                        "resonance_meta": H_meta,
                        "optimization_meta": O_meta,
                    }
                )

            return result

        def _simple_resonance(self, x: torch.Tensor) -> torch.Tensor:
            """Simple FFT-based resonance fallback."""
            fft_result = torch.fft.fft(x, dim=-1)
            magnitudes = torch.abs(fft_result)
            normalized = magnitudes / (magnitudes.sum(dim=-1, keepdim=True) + 1e-8)
            entropy = -torch.sum(normalized * torch.log(normalized + 1e-8), dim=-1)
            max_entropy = math.log(x.shape[-1])
            return 1.0 - (entropy / max_entropy)


class Learnable3REngine:
    """
    High-level engine for learnable 3R fusion.

    Provides training, inference, and model management.
    """

    def __init__(
        self,
        config: Learnable3RConfig | None = None,
        device: str = "cpu",
    ):
        self.config = config or Learnable3RConfig()
        self.device = device

        if TORCH_AVAILABLE:
            self.model = Learnable3RFusion(config).to(device)
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            self.model = None  # type: ignore[assignment]
            self.optimizer = None  # type: ignore[assignment]

        self.training_history: list[float] = []

        logger.info(f"Learnable3REngine initialized (device={device})")

    def compute(
        self,
        data: np.ndarray | list[float],
        context: np.ndarray | None = None,
    ) -> Learnable3RResult:
        """
        Compute learnable 3R fusion score.

        Args:
            data: Input data array
            context: Optional context array

        Returns:
            Learnable3RResult with all scores
        """
        if not TORCH_AVAILABLE or self.model is None:
            return self._numpy_fallback(data)

        x = torch.tensor(data, dtype=torch.float32, device=self.device)
        ctx = None
        if context is not None:
            ctx = torch.tensor(context, dtype=torch.float32, device=self.device)

        self.model.eval()
        with torch.no_grad():
            result = self.model(x, ctx)

        return Learnable3RResult(
            fusion_score=float(result["fusion_score"].mean().item()),
            recursion_score=float(result["recursion_score"].mean().item()),
            resonance_score=float(result["resonance_score"].mean().item()),
            optimization_score=float(result["optimization_score"].mean().item()),
            ethical_gate_output=float(result["ethical_gate"].mean().item()),
            learned_weights=result["weights"],
            learned_phi=result["phi"],
            attention_weights=result.get("resonance_meta", {}).get("attention_weights"),
            lyapunov_bound=result["lyapunov_bound"],
            is_stable=True,
        )

    def train_step(
        self,
        data: np.ndarray,
        target: float,
    ) -> float:
        """
        Single training step.

        Args:
            data: Input data
            target: Target score

        Returns:
            Loss value
        """
        if not TORCH_AVAILABLE or self.model is None:
            return 0.0

        self.model.train()

        x = torch.tensor(data, dtype=torch.float32, device=self.device)
        target_tensor = torch.tensor(target, dtype=torch.float32, device=self.device)

        self.optimizer.zero_grad()

        result = self.model(x)
        prediction = result["fusion_score"].mean()

        loss = F.mse_loss(prediction, target_tensor)

        loss.backward()  # type: ignore[no-untyped-call]
        self.optimizer.step()

        loss_value = float(loss.item())
        self.training_history.append(loss_value)

        return loss_value

    def _numpy_fallback(self, data: np.ndarray | list[float]) -> Learnable3RResult:
        """NumPy fallback when PyTorch is unavailable."""
        arr = np.array(data)

        def recursion_score(x: np.ndarray, depth: int = 5) -> float:
            if depth == 0 or len(x) < 2:
                return float(np.std(x) / (np.mean(np.abs(x)) + 1e-8))
            mid = len(x) // 2
            left = recursion_score(x[:mid], depth - 1)
            right = recursion_score(x[mid:], depth - 1)
            return float(0.5 * (left + right) + 0.5 * np.std(x) / (np.mean(np.abs(x)) + 1e-8))

        fft = np.fft.fft(arr)
        magnitudes = np.abs(fft)
        entropy = -np.sum(magnitudes * np.log(magnitudes + 1e-8)) / (np.log(len(arr)) + 1e-8)
        resonance = float(np.clip(1.0 - entropy / 10.0, 0.0, 1.0))

        signal_var = np.var(arr)
        noise_var = np.var(np.diff(arr)) / 2
        snr = signal_var / (noise_var + 1e-8)
        optimization = float(np.clip(1.0 / (1.0 + np.exp(-np.log10(snr + 1))), 0.0, 1.0))

        r_score = float(np.clip(recursion_score(arr), 0.0, 1.0))

        phi_sum = PHI + 1.0 + 1.0 / PHI
        w_R = PHI / phi_sum
        w_H = 1.0 / phi_sum
        w_O = (1.0 / PHI) / phi_sum

        eta = 0.96
        weighted_sum = w_R * r_score + w_H * resonance + w_O * optimization
        fusion = weighted_sum * (eta**PHI)

        return Learnable3RResult(
            fusion_score=fusion,
            recursion_score=r_score,
            resonance_score=resonance,
            optimization_score=optimization,
            ethical_gate_output=eta,
            learned_weights={"w_R": w_R, "w_H": w_H, "w_O": w_O},
            learned_phi=PHI,
            lyapunov_bound=1.0,
            is_stable=True,
        )

    def get_weights(self) -> dict[str, float]:
        """Get current learned weights."""
        if not TORCH_AVAILABLE or self.model is None:
            phi_sum = PHI + 1.0 + 1.0 / PHI
            return {
                "w_R": PHI / phi_sum,
                "w_H": 1.0 / phi_sum,
                "w_O": (1.0 / PHI) / phi_sum,
            }

        with torch.no_grad():
            w_R, w_H, w_O = self.model.get_normalized_weights()
            return {
                "w_R": float(w_R.item()),
                "w_H": float(w_H.item()),
                "w_O": float(w_O.item()),
            }

    def get_phi(self) -> float:
        """Get current learned phi value."""
        if not TORCH_AVAILABLE or self.model is None:
            return PHI

        return float(self.model.phi.item())

    def save_model(self, path: str) -> None:
        """Save model checkpoint."""
        if not TORCH_AVAILABLE or self.model is None:
            logger.warning("Cannot save model: PyTorch not available")
            return

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "training_history": self.training_history,
                "config": self.config,
            },
            path,
        )

        logger.info(f"Model saved to {path}")

    def load_model(self, path: str, allow_unsafe: bool = False) -> None:
        """Load model checkpoint.

        Security Note: By default, uses safe loading (weights_only=True).
        Set allow_unsafe=True only for trusted checkpoints that require
        optimizer state restoration with custom objects.

        Args:
            path: Path to checkpoint file
            allow_unsafe: If True, allows loading checkpoints with pickle.
                         Only use for trusted checkpoint sources.
        """
        if not TORCH_AVAILABLE or self.model is None:
            logger.warning("Cannot load model: PyTorch not available")
            return

        try:
            # Default: safe loading with weights_only=True
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        except Exception as e:
            if allow_unsafe:
                logger.warning(
                    "Safe checkpoint loading failed. Falling back to unsafe mode "
                    "as explicitly requested. Only do this for trusted checkpoints. "
                    f"Original error: {e}"
                )
                checkpoint = torch.load(
                    path, map_location=self.device, weights_only=False
                )  # nosec B614 - intentional for trusted checkpoints with allow_unsafe=True
            else:
                raise RuntimeError(
                    f"Checkpoint at '{path}' cannot be loaded safely (weights_only=True). "
                    "This may indicate the checkpoint contains custom pickled objects. "
                    "If you trust this checkpoint source, re-run with allow_unsafe=True. "
                    f"Original error: {e}"
                ) from e

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_history = checkpoint.get("training_history", [])

        logger.info(f"Model loaded from {path}")

    def get_statistics(self) -> dict[str, Any]:
        """Get engine statistics."""
        return {
            "torch_available": TORCH_AVAILABLE,
            "device": self.device,
            "training_steps": len(self.training_history),
            "weights": self.get_weights(),
            "phi": self.get_phi(),
            "avg_recent_loss": (
                float(np.mean(self.training_history[-100:])) if self.training_history else 0.0
            ),
        }

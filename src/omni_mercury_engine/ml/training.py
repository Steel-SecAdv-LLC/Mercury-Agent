"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.
"""

from __future__ import annotations


"""
Training utilities for fusion model using PyTorch Lightning
Enhanced with Ava Equation state evolution optimizers
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset

from omni_mercury_engine.ml.fusion_network import OmniFusionModel


# pytorch_lightning is optional - gracefully degrade when not available
try:
    import pytorch_lightning as pl

    HAS_PYTORCH_LIGHTNING = True
except ImportError:
    HAS_PYTORCH_LIGHTNING = False
    pl = None


if TYPE_CHECKING:
    from collections.abc import Callable

    import pytorch_lightning as pl

__all__ = [
    "HAS_PYTORCH_LIGHTNING",
    "AnomalyDataset",
    "EarlyStopping",
    "FusionTrainer",
    "LearningRateScheduler",
    "LyapunovAnomalyLoss",
    "MercuryExponentialDecayOptimizer",
    "MercuryHarmonicOptimizer",
    "MercuryMomentumOptimizer",
    "MercuryOptimizer",
    "ThreeRAnomalyTrainer",
    "Trainer",
    "TrainingConfig",
    "create_mercury_optimizer",
]


@dataclass
class TrainingConfig:
    """Configuration for model training.

    Validates parameters and provides sensible defaults for training loops.
    """

    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 10
    weight_decay: float = 0.0
    device: str = "cpu"
    optimizer: str = "adam"
    scheduler: str | None = None
    gradient_clip: float | None = None
    early_stopping_patience: int | None = None

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")


class EarlyStopping:
    """Early stopping callback to prevent overfitting.

    Monitors a metric and stops training when no improvement is seen
    for a specified number of epochs (patience).
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = "min") -> None:
        """Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss (lower is better), 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: float | None = None
        self.early_stop = False

    def __call__(self, metric: float) -> bool:
        """Check if training should stop.

        Args:
            metric: Current metric value to evaluate

        Returns:
            True if training should stop, False otherwise
        """
        score = -metric if self.mode == "min" else metric

        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop


class LearningRateScheduler:
    """Wrapper for PyTorch learning rate schedulers.

    Provides a unified interface for different scheduler types
    including step decay and cosine annealing.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        mode: str = "step",
        step_size: int = 10,
        gamma: float = 0.1,
        T_max: int = 100,
        eta_min: float = 0.0,
        **kwargs: Any,
    ):
        """Initialize learning rate scheduler.

        Args:
            optimizer: PyTorch optimizer to schedule
            mode: Scheduler type ('step', 'cosine', 'plateau')
            step_size: Period of learning rate decay (for step mode)
            gamma: Multiplicative factor of learning rate decay
            T_max: Maximum number of iterations (for cosine mode)
            eta_min: Minimum learning rate (for cosine mode)
        """
        self.optimizer = optimizer
        self.mode = mode

        self._scheduler: (
            lr_scheduler.StepLR | lr_scheduler.CosineAnnealingLR | lr_scheduler.ReduceLROnPlateau
        )
        if mode == "step":
            self._scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif mode == "cosine":
            self._scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=T_max, eta_min=eta_min
            )
        elif mode == "plateau":
            self._scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=gamma, patience=step_size
            )
        else:
            raise ValueError(f"Unknown scheduler mode: {mode}")

    def step(self, metric: float | None = None) -> None:
        """Advance the scheduler by one step.

        Args:
            metric: Metric value for plateau scheduler
        """
        if self.mode == "plateau" and metric is not None:
            if isinstance(self._scheduler, lr_scheduler.ReduceLROnPlateau):
                self._scheduler.step(metric)
        elif not isinstance(self._scheduler, lr_scheduler.ReduceLROnPlateau):
            self._scheduler.step()

    def get_last_lr(self) -> list[float]:
        """Get the last computed learning rate."""
        return list(self._scheduler.get_last_lr())


class Trainer:
    """General-purpose trainer for PyTorch models.

    Provides training loop, validation, checkpointing, and
    integration with early stopping and learning rate scheduling.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        criterion: nn.Module | None = None,
    ):
        """Initialize trainer.

        Args:
            model: PyTorch model to train
            config: Training configuration
            criterion: Loss function (defaults to MSELoss)
        """
        self.model = model
        self.config = config
        self.criterion = criterion or nn.MSELoss()
        self.device = torch.device(config.device)

        self.model.to(self.device)

        self.optimizer: optim.Adam | optim.AdamW | optim.SGD
        if config.optimizer == "adam":
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        elif config.optimizer == "adamw":
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        elif config.optimizer == "sgd":
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        else:
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )

        self.scheduler = None
        self.early_stopping = None
        self.epoch = 0

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Execute a single training step.

        Args:
            x: Input tensor
            y: Target tensor

        Returns:
            Loss value as float
        """
        self.model.train()
        x = x.to(self.device)
        y = y.to(self.device)

        self.optimizer.zero_grad()
        output = self.model(x)
        loss = self.criterion(output, y)
        loss.backward()

        if self.config.gradient_clip is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)

        self.optimizer.step()

        return float(loss.item())

    def validate_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Execute a single validation step.

        Args:
            x: Input tensor
            y: Target tensor

        Returns:
            Loss value as float
        """
        self.model.eval()
        x = x.to(self.device)
        y = y.to(self.device)

        with torch.no_grad():
            output = self.model(x)
            loss = self.criterion(output, y)

        return float(loss.item())

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": {
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "epochs": self.config.epochs,
            },
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str, *, allow_unsafe: bool = False) -> None:
        """Load model checkpoint.

        Args:
            path: Path to checkpoint file
            allow_unsafe: If True, allows loading checkpoints that require full
                pickle deserialization. Only set this to True for legacy checkpoints
                from trusted sources. Default is False for maximum security.

        Security Note:
            By default, this uses weights_only=True to prevent arbitrary code
            execution from untrusted checkpoint files. Standard state_dict
            checkpoints (model weights, optimizer state, scalars) load safely.

            If you encounter errors with legacy checkpoints that stored custom
            objects, you can pass allow_unsafe=True after verifying the checkpoint
            source is trusted (i.e., generated by this application).

            See: https://pytorch.org/docs/stable/generated/torch.load.html

        Raises:
            RuntimeError: If checkpoint requires unsafe loading but allow_unsafe=False
        """
        import logging

        logger = logging.getLogger(__name__)

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
        self.epoch = checkpoint.get("epoch", 0)


class MercuryOptimizer(optim.Optimizer):
    """
    Base Mercury optimizer with state evolution dynamics
    """

    def __init__(
        self,
        params: Any,
        lr: float = 0.001,
        alpha: float = 0.1,
        beta: float = 0.9,
        quantum_noise: float = 0.0,
    ) -> None:
        defaults = {"lr": lr, "alpha": alpha, "beta": beta, "quantum_noise": quantum_noise}
        super().__init__(params, defaults)

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["state_vector"] = torch.zeros_like(p.data)

                state["step"] += 1
                alpha = group["alpha"]
                beta = group["beta"]
                lr = group["lr"]
                quantum_noise = group["quantum_noise"]

                state_evolution = alpha * grad + beta * state["state_vector"]

                if quantum_noise > 0:
                    noise = torch.randn_like(state_evolution) * quantum_noise
                    state_evolution = state_evolution + noise

                p.data.add_(-lr * state_evolution)

                state["state_vector"] = state_evolution

        return loss


class MercuryMomentumOptimizer(optim.Optimizer):
    """
    Mercury optimizer with momentum variant
    """

    def __init__(
        self, params: Any, lr: float = 0.001, alpha: float = 0.1, momentum: float = 0.9
    ) -> None:
        defaults = {"lr": lr, "alpha": alpha, "momentum": momentum}
        super().__init__(params, defaults)

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p.data)

                momentum = group["momentum"]
                alpha = group["alpha"]
                lr = group["lr"]

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad, alpha=1 - momentum)

                state_update = alpha * grad + (1 - alpha) * buf

                p.data.add_(-lr * state_update)

        return loss


class MercuryExponentialDecayOptimizer(optim.Optimizer):
    """
    Mercury optimizer with exponential decay
    """

    def __init__(
        self, params: Any, lr: float = 0.001, alpha: float = 0.1, decay_rate: float = 0.99
    ) -> None:
        defaults = {"lr": lr, "alpha": alpha, "decay_rate": decay_rate}
        super().__init__(params, defaults)

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)

                state["step"] += 1
                alpha = group["alpha"]
                decay_rate = group["decay_rate"]
                lr = group["lr"]

                exp_avg = state["exp_avg"]
                exp_avg.mul_(decay_rate).add_(grad, alpha=1 - decay_rate)

                effective_lr = lr * (decay_rate ** state["step"])

                p.data.add_(-effective_lr * (alpha * grad + (1 - alpha) * exp_avg))

        return loss


class MercuryHarmonicOptimizer(optim.Optimizer):
    """
    Mercury optimizer with harmonic oscillator variant
    """

    def __init__(
        self, params: Any, lr: float = 0.001, alpha: float = 0.1, omega: float = 0.1
    ) -> None:
        defaults = {"lr": lr, "alpha": alpha, "omega": omega}
        super().__init__(params, defaults)

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["oscillation"] = torch.zeros_like(p.data)

                state["step"] += 1
                alpha = group["alpha"]
                omega = group["omega"]
                lr = group["lr"]

                t = state["step"]
                harmonic_factor = np.cos(omega * t)

                state["oscillation"].mul_(0.9).add_(grad)

                modulated_update = (
                    alpha * grad + (1 - alpha) * harmonic_factor * state["oscillation"]
                )

                p.data.add_(-lr * modulated_update)

        return loss


class LyapunovAnomalyLoss(nn.Module):
    """
    Training loss with Lyapunov stability constraint for anomaly detection.

    Mathematical Foundation:
        Loss = L_recon + λ_kl·L_KL + λ_sup·L_supervised + μ·L_stability

        Where:
            L_recon = MSE(x, x_recon) - reconstruction loss
            L_KL = -0.5 * mean(1 + logvar - mu² - exp(logvar)) - VAE KL divergence
            L_supervised = BCE(anomaly_scores, labels) - direct anomaly supervision
            L_stability = max(0, dV/dt + alpha*V) - Lyapunov violation penalty

        Lyapunov Stability Condition:
            dV/dt <= -alpha*V ensures exponential convergence of anomaly scores

    This loss forces the model to learn dynamics where anomaly scores converge
    (stable) rather than diverge (unstable). For life-critical applications,
    unstable predictions are dangerous.

    The Lyapunov function V(s) = s² (quadratic) is used, where s is the anomaly
    score. The discrete-time derivative approximation V̇ ≈ V_t - V_{t-1} penalizes
    any increase beyond the allowed rate.

    Args:
        lambda_kl: Weight for KL divergence term. Set to 0.0 for non-VAE models.
        lambda_supervised: Weight for supervised anomaly loss (default: 1.0)
        mu_stability: Weight for Lyapunov stability term (default: 0.1)
        alpha: Lyapunov convergence rate parameter (default: 0.25, matches
               CONVERGENCE_RATE_PARAMETER from three_r_mechanism.py)
        reduction: Loss reduction method ('mean' or 'sum')

    Example:
        >>> loss_fn = LyapunovAnomalyLoss(lambda_kl=0.0, mu_stability=0.1)
        >>> output = model(x)
        >>> loss_dict = loss_fn(
        ...     x=x,
        ...     x_recon=output["reconstruction"],
        ...     anomaly_scores=output["anomaly_scores"],
        ...     labels=labels,  # Optional: add supervised signal
        ... )
        >>> loss_dict["total"].backward()

    Note:
        For non-VAE models (most transformer-based AD), set lambda_kl=0.0 and
        do not pass mu/logvar parameters. The loss will skip KL computation.

    Reference:
        - Lyapunov stability theory for neural networks
        - CONVERGENCE_RATE_PARAMETER = 0.25 from three_r_mechanism.py
    """

    def __init__(
        self,
        lambda_kl: float = 0.0,
        lambda_supervised: float = 1.0,
        mu_stability: float = 0.1,
        alpha: float = 0.25,
        reduction: str = "mean",
    ):
        super().__init__()
        self.lambda_kl = lambda_kl
        self.lambda_supervised = lambda_supervised
        self.mu_stability = mu_stability
        self.alpha = alpha
        self.reduction = reduction

        # State for tracking previous scores (for V̇ computation)
        self.prev_scores: torch.Tensor | None = None

        # Track stability metrics
        self.stability_violations = 0
        self.total_steps = 0

    def reset_state(self) -> None:
        """Reset previous scores state. Call at start of each epoch."""
        self.prev_scores = None

    def forward(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        anomaly_scores: torch.Tensor,
        labels: torch.Tensor | None = None,
        mu: torch.Tensor | None = None,
        logvar: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute Lyapunov-constrained anomaly detection loss.

        Args:
            x: Original input tensor [batch_size, seq_len, dim] or [batch_size, dim]
            x_recon: Reconstructed tensor (same shape as x)
            anomaly_scores: Anomaly scores [batch_size], expected in [0, 1]
            labels: Binary anomaly labels [batch_size] (optional, enables supervised loss)
            mu: VAE mean (optional, only for VAE models)
            logvar: VAE log variance (optional, only for VAE models)

        Returns:
            Dict containing:
                - total: Total weighted loss (for backward())
                - reconstruction: Reconstruction loss component
                - supervised: Supervised BCE loss (0 if labels not provided)
                - kl: KL divergence component (0 if lambda_kl=0)
                - stability: Lyapunov stability violation penalty
                - lyapunov_V: Current Lyapunov function value V(s)
                - stability_violated: Boolean indicating if dV/dt + alpha*V > 0
        """
        # Reconstruction loss (MSE)
        if self.reduction == "mean":
            L_recon = torch.nn.functional.mse_loss(x_recon, x, reduction="mean")
        else:
            L_recon = torch.nn.functional.mse_loss(x_recon, x, reduction="sum")

        # Supervised anomaly loss (BCE) - CRITICAL for accuracy
        if labels is not None and self.lambda_supervised > 0:
            # Convert labels to binary float in [0, 1]
            if labels.dim() > 1:
                labels = labels[:, 0]  # Take first column if multi-dim
            binary_labels = (labels > 0).float()
            L_supervised = torch.nn.functional.binary_cross_entropy(
                anomaly_scores, binary_labels, reduction="mean"
            )
        else:
            L_supervised = torch.tensor(0.0, device=x.device)

        # KL divergence (for VAE models only)
        if self.lambda_kl > 0 and mu is not None and logvar is not None:
            # KL(q(z|x) || p(z)) for Gaussian
            L_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        else:
            L_kl = torch.tensor(0.0, device=x.device)

        # Lyapunov stability constraint
        # V(s) = s² (quadratic Lyapunov function)
        V_t = anomaly_scores.pow(2).mean()

        if self.prev_scores is not None:
            V_prev = self.prev_scores.pow(2).mean()

            # Discrete approximation of dV/dt
            V_dot = V_t - V_prev

            # Stability violation: dV/dt + alpha*V > 0 means diverging
            # We penalize any positive value of (dV/dt + alpha*V)
            stability_violation = torch.nn.functional.relu(V_dot + self.alpha * V_t)
            L_stability = stability_violation

            # Track metrics
            self.total_steps += 1
            if stability_violation.item() > 0:
                self.stability_violations += 1
        else:
            L_stability = torch.tensor(0.0, device=x.device)

        # Store current scores for next iteration (detached to prevent graph issues)
        self.prev_scores = anomaly_scores.detach().clone()

        # Total loss: reconstruction + supervised + KL + stability
        total = (
            L_recon
            + self.lambda_supervised * L_supervised
            + self.lambda_kl * L_kl
            + self.mu_stability * L_stability
        )

        return {
            "total": total,
            "reconstruction": L_recon,
            "supervised": L_supervised,
            "kl": L_kl,
            "stability": L_stability,
            "lyapunov_V": V_t,
            "stability_violated": L_stability.item() > 0 if self.prev_scores is not None else False,
        }

    def get_stability_rate(self) -> float:
        """Get percentage of steps where stability was maintained."""
        if self.total_steps == 0:
            return 1.0
        return 1.0 - (self.stability_violations / self.total_steps)


def create_mercury_optimizer(
    params: Any, variant: str = "base", lr: float = 0.001, **kwargs: Any
) -> optim.Optimizer:
    """
    Factory function to create Mercury optimizer variants
    """
    if variant == "base":
        return MercuryOptimizer(params, lr=lr, **kwargs)
    elif variant == "momentum":
        return MercuryMomentumOptimizer(params, lr=lr, **kwargs)
    elif variant == "exp_decay":
        return MercuryExponentialDecayOptimizer(params, lr=lr, **kwargs)
    elif variant == "harmonic":
        return MercuryHarmonicOptimizer(params, lr=lr, **kwargs)
    else:
        raise ValueError(f"Unknown Mercury optimizer variant: {variant}")


class AnomalyDataset(
    Dataset[
        tuple[dict[str, torch.Tensor], torch.Tensor]
        | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]
    ]
):
    """Dataset for anomaly detection training"""

    def __init__(
        self,
        detector_features: dict[str, torch.Tensor],
        labels: torch.Tensor,
        scores: dict[str, torch.Tensor] | None = None,
    ):
        self.detector_features = detector_features
        self.labels = labels
        self.scores = scores
        self.num_samples = labels.shape[0]

    def __len__(self) -> int:
        return int(self.num_samples)

    def __getitem__(
        self, idx: int
    ) -> (
        tuple[dict[str, torch.Tensor], torch.Tensor]
        | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]
    ):
        features = {k: v[idx] for k, v in self.detector_features.items()}
        label = self.labels[idx]

        if self.scores is not None:
            scores = {k: v[idx] for k, v in self.scores.items()}
            return features, scores, label

        return features, label


def _get_lightning_base() -> type:
    """Get LightningModule base class or raise helpful error."""
    if not HAS_PYTORCH_LIGHTNING:
        raise ImportError(
            "pytorch_lightning is required for FusionTrainer and ThreeRAnomalyTrainer. "
            "Install with: pip install pytorch-lightning"
        )
    return cast(type, pl.LightningModule)


# Conditional base class - evaluated at class definition time
_LightningBase = _get_lightning_base() if HAS_PYTORCH_LIGHTNING else nn.Module


class FusionTrainer(_LightningBase):  # type: ignore[misc, valid-type]
    """
    PyTorch Lightning trainer for fusion model.

    Handles multi-task learning with:
    - Binary anomaly detection
    - Multi-class classification
    - Regression (anomaly severity)
    """

    def __init__(
        self,
        model: OmniFusionModel | None = None,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        anomaly_weight: float = 1.0,
        classification_weight: float = 0.5,
        regression_weight: float = 0.3,
    ):
        super().__init__()
        self.model = model or OmniFusionModel()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.anomaly_weight = anomaly_weight
        self.classification_weight = classification_weight
        self.regression_weight = regression_weight

        self.anomaly_criterion = nn.BCELoss()
        self.classification_criterion = nn.CrossEntropyLoss()
        self.regression_criterion = nn.MSELoss()

        self.save_hyperparameters(ignore=["model"])

    def forward(self, detector_features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = self.model(detector_features, return_attention=True)
        return result

    def training_step(
        self,
        batch: tuple[dict[str, torch.Tensor], torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        features, labels = batch

        outputs = self.forward(features)

        anomaly_labels = (labels > 0).float().unsqueeze(1)
        anomaly_loss = self.anomaly_criterion(outputs["anomaly_probs"], anomaly_labels)

        class_labels = labels.long()
        classification_loss = self.classification_criterion(outputs["class_logits"], class_labels)

        severity = labels.float().unsqueeze(1)
        regression_loss = self.regression_criterion(outputs["regression_output"], severity)

        total_loss = (
            self.anomaly_weight * anomaly_loss
            + self.classification_weight * classification_loss
            + self.regression_weight * regression_loss
        )

        self.log("train_loss", total_loss)
        self.log("train_anomaly_loss", anomaly_loss)
        self.log("train_classification_loss", classification_loss)
        self.log("train_regression_loss", regression_loss)

        result: torch.Tensor = total_loss
        return result

    def validation_step(
        self,
        batch: tuple[dict[str, torch.Tensor], torch.Tensor],
        batch_idx: int,
    ) -> None:
        features, labels = batch

        outputs = self.forward(features)

        anomaly_labels = (labels > 0).float().unsqueeze(1)
        anomaly_loss = self.anomaly_criterion(outputs["anomaly_probs"], anomaly_labels)

        class_labels = labels.long()
        classification_loss = self.classification_criterion(outputs["class_logits"], class_labels)

        severity = labels.float().unsqueeze(1)
        regression_loss = self.regression_criterion(outputs["regression_output"], severity)

        total_loss = (
            self.anomaly_weight * anomaly_loss
            + self.classification_weight * classification_loss
            + self.regression_weight * regression_loss
        )

        self.log("val_loss", total_loss)
        self.log("val_anomaly_loss", anomaly_loss)

        preds = (outputs["anomaly_probs"] > 0.5).float()
        accuracy = (preds == anomaly_labels).float().mean()
        self.log("val_accuracy", accuracy)

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer_type = getattr(self, "optimizer_type", "adamw")

        if optimizer_type.startswith("ava_"):
            variant = optimizer_type.replace("ava_", "")
            optimizer = create_mercury_optimizer(
                self.parameters(),
                variant=variant,
                lr=self.learning_rate,
            )
        else:
            optimizer = optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


class ThreeRAnomalyTrainer(_LightningBase):  # type: ignore[misc, valid-type]
    """
    PyTorch Lightning trainer for 3R Anomaly Transformer with Lyapunov stability.

    Integrates ThreeRAnomalyTransformer with LyapunovAnomalyLoss for
    stability-constrained anomaly detection training.

    Features:
        - Reconstruction-based anomaly detection
        - Lyapunov stability constraint (dV/dt <= -alpha*V)
        - Optional VAE-style KL divergence
        - Multi-scale 3R attention mechanism
        - Golden-ratio AAFE fusion weights

    Args:
        input_dim: Input feature dimension
        d_model: Model dimension (default: 256)
        n_heads: Attention heads (default: 8)
        num_layers: Number of 3R attention layers (default: 2)
        learning_rate: Learning rate (default: 0.001)
        lambda_kl: KL divergence weight, 0.0 for non-VAE (default: 0.0)
        mu_stability: Lyapunov stability weight (default: 0.1)
        alpha: Lyapunov convergence rate (default: 0.25)
        ethical_threshold: η_Ethical threshold (default: 0.96)

    Example:
        >>> trainer = ThreeRAnomalyTrainer(input_dim=25)
        >>> pl_trainer = pl.Trainer(max_epochs=100)
        >>> pl_trainer.fit(trainer, train_loader, val_loader)
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        num_layers: int = 2,
        num_scales: int = 3,
        max_freqs: int = 5,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        lambda_kl: float = 0.0,
        mu_stability: float = 0.1,
        alpha: float = 0.25,
        ethical_threshold: float = 0.96,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Import here to avoid circular imports
        from omni_mercury_engine.ml.three_r_attention import ThreeRAnomalyTransformer

        self.model = ThreeRAnomalyTransformer(
            input_dim=input_dim,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            num_scales=num_scales,
            max_freqs=max_freqs,
            ethical_threshold=ethical_threshold,
            dropout=dropout,
        )

        self.criterion = LyapunovAnomalyLoss(
            lambda_kl=lambda_kl,
            mu_stability=mu_stability,
            alpha=alpha,
        )

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Save hyperparameters for checkpointing
        self.save_hyperparameters(ignore=["model", "criterion"])

        # Metrics tracking
        self.training_stability_rate = 0.0
        self.validation_metrics: dict[str, float] = {}

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass through 3R Anomaly Transformer."""
        result: dict[str, torch.Tensor] = self.model(x)
        return result

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Training step with Lyapunov-constrained loss and supervised signal."""
        x, labels = batch

        # Forward pass
        output = self.forward(x)

        # Compute loss WITH labels for supervised signal (critical for accuracy)
        loss_dict = self.criterion(
            x=x,
            x_recon=output["reconstruction"],
            anomaly_scores=output["anomaly_scores"],
            labels=labels,  # Pass labels for supervised BCE loss
        )

        # Log metrics
        self.log("train_loss", loss_dict["total"], prog_bar=True)
        self.log("train_recon_loss", loss_dict["reconstruction"])
        self.log("train_supervised_loss", loss_dict["supervised"])
        self.log("train_stability_loss", loss_dict["stability"])
        self.log("train_lyapunov_V", loss_dict["lyapunov_V"])

        return loss_dict["total"]

    def on_train_epoch_start(self) -> None:
        """Reset loss state at start of each epoch."""
        self.criterion.reset_state()

    def on_train_epoch_end(self) -> None:
        """Log stability rate at end of epoch."""
        stability_rate = self.criterion.get_stability_rate()
        self.log("train_stability_rate", stability_rate)
        self.training_stability_rate = stability_rate

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Validation step with metric computation."""
        x, labels = batch

        # Forward pass
        output = self.forward(x)

        # Compute loss with labels for consistency
        loss_dict = self.criterion(
            x=x,
            x_recon=output["reconstruction"],
            anomaly_scores=output["anomaly_scores"],
            labels=labels,
        )

        # Compute F1 and recall
        anomaly_scores = output["anomaly_scores"]
        preds = (anomaly_scores > 0.5).float()

        # Handle multi-dimensional labels
        if labels.dim() > 1:
            labels = labels[:, 0]

        binary_labels = (labels > 0).float()

        # Metrics
        tp = ((preds == 1) & (binary_labels == 1)).float().sum()
        fp = ((preds == 1) & (binary_labels == 0)).float().sum()
        fn = ((preds == 0) & (binary_labels == 1)).float().sum()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        # Log metrics
        self.log("val_loss", loss_dict["total"], prog_bar=True)
        self.log("val_recon_loss", loss_dict["reconstruction"])
        self.log("val_f1", f1, prog_bar=True)
        self.log("val_recall", recall, prog_bar=True)
        self.log("val_precision", precision)

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure AdamW optimizer with cosine annealing."""
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100,
            eta_min=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def get_ablation_config(self) -> dict[str, Any]:
        """Get current configuration for ablation studies."""
        return {
            "input_dim": self.model.input_dim,
            "d_model": self.model.d_model,
            "lambda_kl": self.criterion.lambda_kl,
            "mu_stability": self.criterion.mu_stability,
            "alpha": self.criterion.alpha,
            "learning_rate": self.learning_rate,
        }

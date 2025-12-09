"""
OMNI ♱ AVA (O♱A)
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
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset

from omni_anomaly_engine.ml.fusion_network import OmniFusionModel

__all__ = [
    "AnomalyDataset",
    "AvaExponentialDecayOptimizer",
    "AvaHarmonicOptimizer",
    "AvaMomentumOptimizer",
    "AvaOptimizer",
    "EarlyStopping",
    "FusionTrainer",
    "LearningRateScheduler",
    "Trainer",
    "TrainingConfig",
    "create_ava_optimizer",
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
            self._scheduler.step(metric)
        else:
            self._scheduler.step()

    def get_last_lr(self) -> list[float]:
        """Get the last computed learning rate."""
        return self._scheduler.get_last_lr()


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

        return loss.item()

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

        return loss.item()

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
                checkpoint = torch.load(path, map_location=self.device, weights_only=False)
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


class AvaOptimizer(optim.Optimizer):
    """
    Base Ava optimizer with state evolution dynamics
    """

    def __init__(self, params, lr=0.001, alpha=0.1, beta=0.9, quantum_noise=0.0) -> None:
        defaults = {"lr": lr, "alpha": alpha, "beta": beta, "quantum_noise": quantum_noise}
        super().__init__(params, defaults)

    def step(self, closure=None):
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


class AvaMomentumOptimizer(optim.Optimizer):
    """
    Ava optimizer with momentum variant
    """

    def __init__(self, params, lr=0.001, alpha=0.1, momentum=0.9) -> None:
        defaults = {"lr": lr, "alpha": alpha, "momentum": momentum}
        super().__init__(params, defaults)

    def step(self, closure=None):
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


class AvaExponentialDecayOptimizer(optim.Optimizer):
    """
    Ava optimizer with exponential decay
    """

    def __init__(self, params, lr=0.001, alpha=0.1, decay_rate=0.99) -> None:
        defaults = {"lr": lr, "alpha": alpha, "decay_rate": decay_rate}
        super().__init__(params, defaults)

    def step(self, closure=None):
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


class AvaHarmonicOptimizer(optim.Optimizer):
    """
    Ava optimizer with harmonic oscillator variant
    """

    def __init__(self, params, lr=0.001, alpha=0.1, omega=0.1) -> None:
        defaults = {"lr": lr, "alpha": alpha, "omega": omega}
        super().__init__(params, defaults)

    def step(self, closure=None):
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


def create_ava_optimizer(
    params, variant: str = "base", lr: float = 0.001, **kwargs
) -> optim.Optimizer:
    """
    Factory function to create Ava optimizer variants
    """
    if variant == "base":
        return AvaOptimizer(params, lr=lr, **kwargs)
    elif variant == "momentum":
        return AvaMomentumOptimizer(params, lr=lr, **kwargs)
    elif variant == "exp_decay":
        return AvaExponentialDecayOptimizer(params, lr=lr, **kwargs)
    elif variant == "harmonic":
        return AvaHarmonicOptimizer(params, lr=lr, **kwargs)
    else:
        raise ValueError(f"Unknown Ava optimizer variant: {variant}")


class AnomalyDataset(Dataset):
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
        return self.num_samples

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


class FusionTrainer(pl.LightningModule):
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
        return self.model(detector_features, return_attention=True)

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

        return total_loss

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

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer_type = getattr(self, "optimizer_type", "adamw")

        if optimizer_type.startswith("ava_"):
            variant = optimizer_type.replace("ava_", "")
            optimizer = create_ava_optimizer(
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

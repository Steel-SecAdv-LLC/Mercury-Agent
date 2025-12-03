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

"""
Training utilities for fusion model using PyTorch Lightning
Enhanced with Ava Equation state evolution optimizers
"""


import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn, optim
from torch.utils.data import Dataset

from omni_anomaly_engine.ml.fusion_network import OmniFusionModel


class AvaOptimizer(optim.Optimizer):
    """
    Base Ava optimizer with state evolution dynamics
    """

    def __init__(self, params, lr=0.001, alpha=0.1, beta=0.9, quantum_noise=0.0):
        defaults = dict(lr=lr, alpha=alpha, beta=beta, quantum_noise=quantum_noise)
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

    def __init__(self, params, lr=0.001, alpha=0.1, momentum=0.9):
        defaults = dict(lr=lr, alpha=alpha, momentum=momentum)
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

    def __init__(self, params, lr=0.001, alpha=0.1, decay_rate=0.99):
        defaults = dict(lr=lr, alpha=alpha, decay_rate=decay_rate)
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

    def __init__(self, params, lr=0.001, alpha=0.1, omega=0.1):
        defaults = dict(lr=lr, alpha=alpha, omega=omega)
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

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], torch.Tensor] | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
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

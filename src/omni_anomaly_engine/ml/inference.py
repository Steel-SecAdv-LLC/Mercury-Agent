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
Production inference utilities for fusion model
"""

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn

from omni_anomaly_engine.ml.fusion_network import OmniFusionModel

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = [
    "BatchInference",
    "FusionInference",
    "InferenceEngine",
    "ModelEnsemble",
]


class InferenceEngine:
    """General-purpose inference engine for PyTorch models.

    Provides efficient inference with automatic device handling,
    numpy/tensor conversion, and no_grad context management.
    """

    def __init__(self, model: nn.Module, device: str = "cpu") -> None:
        """Initialize inference engine.

        Args:
            model: PyTorch model for inference
            device: Device to run inference on ('cpu', 'cuda', etc.)
        """
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()

    def predict(self, x: torch.Tensor | np.ndarray[Any, Any]) -> torch.Tensor:
        """Run inference on input data.

        Args:
            x: Input tensor or numpy array

        Returns:
            Model output tensor
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)

        x = x.to(self.device)

        with torch.no_grad():
            output: torch.Tensor = self.model(x)

        return output

    def predict_proba(self, x: torch.Tensor | np.ndarray[Any, Any]) -> torch.Tensor:
        """Run inference and apply softmax for probabilities.

        Args:
            x: Input tensor or numpy array

        Returns:
            Probability tensor
        """
        output = self.predict(x)
        return torch.softmax(output, dim=-1)


class BatchInference:
    """Batch inference processor for large datasets.

    Handles memory-efficient processing of large inputs by
    splitting into batches and optionally streaming results.
    """

    def __init__(self, model: nn.Module, batch_size: int = 32, device: str = "cpu") -> None:
        """Initialize batch inference.

        Args:
            model: PyTorch model for inference
            batch_size: Size of batches for processing
            device: Device to run inference on
        """
        self.engine = InferenceEngine(model, device)
        self.batch_size = batch_size
        self.device = torch.device(device)

    def predict(self, x: torch.Tensor | np.ndarray[Any, Any]) -> torch.Tensor:
        """Run batched inference on input data.

        Args:
            x: Input tensor or numpy array

        Returns:
            Concatenated output tensor
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)

        outputs = []
        for i in range(0, len(x), self.batch_size):
            batch = x[i : i + self.batch_size]
            output = self.engine.predict(batch)
            outputs.append(output)

        return torch.cat(outputs, dim=0)

    def predict_stream(self, data_stream: Iterator[torch.Tensor]) -> Iterator[torch.Tensor]:
        """Stream inference results for iterable input.

        Args:
            data_stream: Iterator of input tensors

        Yields:
            Output tensors for each input batch
        """
        for batch in data_stream:
            yield self.engine.predict(batch)


class ModelEnsemble:
    """Ensemble of models with aggregation strategies.

    Combines predictions from multiple models using mean,
    voting, or other aggregation methods. Supports uncertainty
    estimation from ensemble disagreement.
    """

    def __init__(
        self,
        models: list[nn.Module],
        aggregation: str = "mean",
        device: str = "cpu",
    ):
        """Initialize model ensemble.

        Args:
            models: List of PyTorch models
            aggregation: Aggregation method ('mean', 'voting', 'median')
            device: Device to run inference on
        """
        self.device = torch.device(device)
        self.models = [model.to(self.device) for model in models]
        self.aggregation = aggregation

        for model in self.models:
            model.eval()

    def predict(self, x: torch.Tensor | np.ndarray[Any, Any]) -> torch.Tensor:
        """Run ensemble inference with aggregation.

        Args:
            x: Input tensor or numpy array

        Returns:
            Aggregated output tensor
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)

        x = x.to(self.device)

        with torch.no_grad():
            outputs = [model(x) for model in self.models]

        stacked = torch.stack(outputs, dim=0)

        if self.aggregation == "mean":
            return stacked.mean(dim=0)
        elif self.aggregation == "voting":
            votes = torch.stack([torch.sign(out) for out in outputs], dim=0)
            return votes.mean(dim=0)
        elif self.aggregation == "median":
            return stacked.median(dim=0).values
        else:
            return stacked.mean(dim=0)

    def predict_with_uncertainty(
        self, x: torch.Tensor | np.ndarray[Any, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run ensemble inference with uncertainty estimation.

        Args:
            x: Input tensor or numpy array

        Returns:
            Tuple of (aggregated output, uncertainty estimate)
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)

        x = x.to(self.device)

        with torch.no_grad():
            outputs = [model(x) for model in self.models]

        stacked = torch.stack(outputs, dim=0)
        mean_output = stacked.mean(dim=0)
        uncertainty = stacked.std(dim=0)

        return mean_output, uncertainty


class FusionInference:
    """
    Production inference wrapper for fusion model.

    Provides:
    - Batch processing
    - Real-time inference optimization
    - Model loading from checkpoints
    - Fallback to individual detectors
    """

    def __init__(
        self,
        model: OmniFusionModel | None = None,
        checkpoint_path: str | None = None,
        device: str = "cpu",
    ):
        self.device = torch.device(device)

        if model is not None:
            self.model = model
        elif checkpoint_path is not None:
            self.model = self.load_model(checkpoint_path)
        else:
            self.model = OmniFusionModel()

        self.model.to(self.device)
        self.model.eval()

    def load_model(self, checkpoint_path: str) -> OmniFusionModel:
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)

        model = OmniFusionModel()
        model.load_state_dict(checkpoint["state_dict"])

        return model

    def predict(
        self,
        detector_features: dict[str, np.ndarray[Any, Any] | torch.Tensor],
        return_attention: bool = False,
        batch_size: int = 32,
    ) -> dict[str, Any]:
        """
        Run inference on detector features.

        Args:
            detector_features: Dict of features from each detector
            return_attention: Whether to return attention weights
            batch_size: Batch size for processing

        Returns:
            Dict containing predictions and optionally attention weights
        """
        features_tensor: dict[str, torch.Tensor] = {
            k: (torch.tensor(v, dtype=torch.float32) if isinstance(v, np.ndarray) else v)
            for k, v in detector_features.items()
        }

        for k in features_tensor:
            features_tensor[k] = features_tensor[k].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                features_tensor,
                return_attention=return_attention,
            )

        results = {
            "anomaly_probs": outputs["anomaly_probs"].cpu().numpy(),
            "class_predictions": outputs["class_logits"].argmax(dim=1).cpu().numpy(),
            "severity_scores": outputs["regression_output"].cpu().numpy(),
        }

        if return_attention and "attention_weights" in outputs:
            results["attention_weights"] = {
                k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                for k, v in outputs["attention_weights"].items()
            }

            importance = self.model.get_detector_importance(features_tensor)
            results["detector_importance"] = importance

        return results

    def predict_batch(
        self,
        detector_features_list: list[dict[str, np.ndarray[Any, Any] | torch.Tensor]],
        batch_size: int = 32,
    ) -> list[dict[str, Any]]:
        """
        Batch inference for multiple samples.

        Args:
            detector_features_list: List of feature dicts
            batch_size: Batch size for processing

        Returns:
            List of prediction dicts
        """
        results = []

        for i in range(0, len(detector_features_list), batch_size):
            batch = detector_features_list[i : i + batch_size]

            batch_features: dict[str, np.ndarray[Any, Any] | torch.Tensor] = {}
            for key in batch[0]:
                tensors: list[torch.Tensor] = []
                for sample in batch:
                    val = sample[key]
                    if isinstance(val, np.ndarray):
                        tensors.append(torch.tensor(val, dtype=torch.float32))
                    else:
                        tensors.append(val)
                batch_features[key] = torch.stack(tensors)

            batch_results = self.predict(batch_features, batch_size=batch_size)

            for j in range(len(batch)):
                anomaly_prob = batch_results["anomaly_probs"][j]
                if isinstance(anomaly_prob, np.ndarray):
                    anomaly_prob = anomaly_prob.item()

                severity_score = batch_results["severity_scores"][j]
                if isinstance(severity_score, np.ndarray):
                    severity_score = severity_score.item()

                class_pred = batch_results["class_predictions"][j]
                if isinstance(class_pred, np.ndarray):
                    class_pred = class_pred.item()

                results.append(
                    {
                        "anomaly_prob": float(anomaly_prob),
                        "class_prediction": int(class_pred),
                        "severity_score": float(severity_score),
                    }
                )

        return results

    def explain(
        self,
        detector_features: dict[str, np.ndarray[Any, Any] | torch.Tensor],
    ) -> dict[str, Any]:
        """
        Get explanation for a prediction via attention weights.

        Args:
            detector_features: Features from each detector

        Returns:
            Dict with prediction and explanation
        """
        result = self.predict(
            detector_features,
            return_attention=True,
        )

        anomaly_prob = result["anomaly_probs"][0]
        if isinstance(anomaly_prob, np.ndarray):
            anomaly_prob = anomaly_prob.item()

        severity = result["severity_scores"][0]
        if isinstance(severity, np.ndarray):
            severity = severity.item()

        explanation = {
            "prediction": {
                "is_anomaly": bool(anomaly_prob > 0.5),
                "confidence": float(anomaly_prob),
                "severity": float(severity),
            },
            "detector_contributions": result.get("detector_importance", {}),
        }

        return explanation

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
Production inference utilities for fusion model
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from omni_anomaly_engine.ml.fusion_network import OmniFusionModel


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
        model: Optional[OmniFusionModel] = None,
        checkpoint_path: Optional[str] = None,
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
        detector_features: Dict[str, Union[np.ndarray, torch.Tensor]],
        return_attention: bool = False,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """
        Run inference on detector features.

        Args:
            detector_features: Dict of features from each detector
            return_attention: Whether to return attention weights
            batch_size: Batch size for processing

        Returns:
            Dict containing predictions and optionally attention weights
        """
        features_tensor = {
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
        detector_features_list: List[Dict[str, Union[np.ndarray, torch.Tensor]]],
        batch_size: int = 32,
    ) -> List[Dict[str, Any]]:
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

            batch_features = {}
            for key in batch[0].keys():
                batch_features[key] = torch.stack(
                    [
                        (
                            torch.tensor(sample[key], dtype=torch.float32)
                            if isinstance(sample[key], np.ndarray)
                            else sample[key]
                        )
                        for sample in batch
                    ]
                )

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
        detector_features: Dict[str, Union[np.ndarray, torch.Tensor]],
    ) -> Dict[str, Any]:
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

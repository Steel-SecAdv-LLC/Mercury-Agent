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
Main OmniAnomalyEngine orchestrating all detectors and models
"""

import torch
from typing import Dict, Any, Optional, List, Union
import numpy as np
from omni_anomaly_engine.core.config import EngineConfig
from omni_anomaly_engine.ml.fusion_network import OmniFusionModel
from omni_anomaly_engine.ml.inference import FusionInference
from omni_anomaly_engine.detectors.statistical import (
    StatisticalAnomalyDetector,
)
from omni_anomaly_engine.detectors.temporal import TemporalAnomalyDetector
from omni_anomaly_engine.detectors.spatial import SpatialAnomalyDetector
from omni_anomaly_engine.detectors.dimensional import DimensionalAnalyzer
from omni_anomaly_engine.detectors.directive import SigmaDirectiveDetector
from omni_anomaly_engine.models.quantum import QuantumAnomalyModel
from omni_anomaly_engine.models.astrophysical import AstrophysicalAnomalyModel
from omni_anomaly_engine.models.biometric import BiometricAnomalyModel
from omni_anomaly_engine.models.affective import AffectiveAnomalyModel
from omni_anomaly_engine.models.neural import NeuralCognitiveModel
from omni_anomaly_engine.models.consciousness import (
    ConsciousnessPreservationModel,
)
from omni_anomaly_engine.security.threat_detection import ThreatDetector
from omni_anomaly_engine.resilience.self_healing import SelfHealingEngine
from omni_anomaly_engine.medical.abms_disciplines import ABMSDisciplineDetector
from omni_anomaly_engine.security.intelligence_fusion import IntelligenceFusionEngine
from omni_anomaly_engine.space.schumann_resonance import SchumannResonanceDetector
from omni_anomaly_engine.models.chemistry import ChemistryAnomalyDetector
from omni_anomaly_engine.models.parapsychology import ParapsychologyDetector


class OmniAnomalyEngine:
    """
    Unified anomaly detection engine with ML-Centric Hybrid Fusion.

    Integrates 13 specialized engines through neural network fusion.
    """

    def __init__(
        self,
        config: Optional[EngineConfig] = None,
        mode: str = "fusion",
        device: str = "cpu",
    ):
        self.config = config or EngineConfig()
        self.mode = mode
        self.device = torch.device(device)

        self._init_detectors()
        self._init_models()
        self._init_fusion()
        self._init_resilience()

    def _init_detectors(self) -> None:
        """Initialize all detectors"""
        self.detectors = {
            "statistical": StatisticalAnomalyDetector(),
            "temporal": TemporalAnomalyDetector(),
            "spatial": SpatialAnomalyDetector(),
            "dimensional": DimensionalAnalyzer(),
            "directive": SigmaDirectiveDetector(),
        }

    def _init_models(self) -> None:
        """Initialize all models"""
        from omni_anomaly_engine.models.multiverse import MultiverseOmniEngine
        from omni_anomaly_engine.models.neurosymbolic import NeurosymbolicEngine

        self.models = {
            "quantum": QuantumAnomalyModel(),
            "astrophysical": AstrophysicalAnomalyModel(),
            "biometric": BiometricAnomalyModel(),
            "affective": AffectiveAnomalyModel(),
            "neural": NeuralCognitiveModel(),
            "consciousness": ConsciousnessPreservationModel(),
            "multiverse": MultiverseOmniEngine(num_universes=10, state_dim=50),
            "neurosymbolic": NeurosymbolicEngine(input_dim=64),
            "medical_abms": ABMSDisciplineDetector(),
            "intelligence_fusion": IntelligenceFusionEngine(),
            "schumann_resonance": SchumannResonanceDetector(),
            "chemistry": ChemistryAnomalyDetector(),
            "parapsychology": ParapsychologyDetector(),
        }

        self.security = ThreatDetector()

    def _init_fusion(self) -> None:
        """Initialize ML fusion components"""
        if self.mode == "fusion":
            self.fusion_model = OmniFusionModel()
            self.fusion_model.to(self.device)
            self.fusion_inference = FusionInference(
                model=self.fusion_model,
                device=str(self.device),
            )

    def _init_resilience(self) -> None:
        """Initialize resilience components"""
        self.self_healing = SelfHealingEngine()

    def detect(
        self,
        data: Union[np.ndarray, torch.Tensor, Dict[str, Any]],
        detector_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Detect anomalies using specified detectors.

        Args:
            data: Input data (array, tensor, or dict)
            detector_types: List of detector names to use (None = all)

        Returns:
            Detection results with scores and flags
        """
        if detector_types is None:
            detector_types = list(self.detectors.keys())

        results = {}

        for detector_name in detector_types:
            if detector_name in self.detectors:
                detector = self.detectors[detector_name]

                if not detector.is_fitted():
                    if isinstance(data, dict):
                        continue
                    detector.fit(data)

                results[detector_name] = detector.detect(data)

        return {
            "detectors": results,
            "is_anomaly": any(
                (
                    r.get("is_anomaly", False)
                    if isinstance(r.get("is_anomaly"), bool)
                    else any(r.get("is_anomaly", []))
                )
                for r in results.values()
            ),
        }

    def _normalize_scores(self, scores: Any, batch_size: int) -> torch.Tensor:
        """Normalize scores to tensor format [batch_size, 1]"""
        if isinstance(scores, (list, np.ndarray)):
            scores = torch.tensor(scores, dtype=torch.float32)
            if scores.dim() == 1:
                scores = scores.unsqueeze(-1)
        elif isinstance(scores, bool):
            scores = torch.full((batch_size, 1), float(scores), dtype=torch.float32)
        else:
            scores = torch.full((batch_size, 1), float(scores), dtype=torch.float32)
        return scores

    def _extract_detector_features(
        self, data: Union[np.ndarray, torch.Tensor, Dict[str, Any]]
    ) -> tuple:
        """Extract features from all detectors"""
        detector_features = {}
        detector_scores = {}

        for name, detector in self.detectors.items():
            try:
                if not detector.is_fitted():
                    if isinstance(data, dict):
                        continue
                    detector.fit(data)

                features = detector.extract_features(data)
                detector_features[name] = features

                result = detector.detect(data)
                scores = result.get("scores", result.get("is_anomaly", 0))
                detector_scores[name] = self._normalize_scores(scores, features.shape[0])
            except Exception:
                continue

        return detector_features, detector_scores

    def _extract_model_features(
        self, data: Union[np.ndarray, torch.Tensor, Dict[str, Any]]
    ) -> tuple:
        """Extract features from all models"""
        model_features = {}
        model_scores = {}

        for name, model in self.models.items():
            try:
                features = model.extract_features(data)
                model_features[name] = features

                prediction = model.predict(data)
                scores = prediction.get("anomaly_scores", 0)
                model_scores[name] = self._normalize_scores(scores, features.shape[0])
            except Exception:
                continue

        return model_features, model_scores

    def detect_with_fusion(
        self, data: Union[np.ndarray, torch.Tensor, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Detect anomalies using ML fusion of all detectors.

        Args:
            data: Input data

        Returns:
            Fusion-based detection results with attention weights
        """
        if self.mode != "fusion":
            return self.detect(data)

        det_features, det_scores = self._extract_detector_features(data)
        mod_features, mod_scores = self._extract_model_features(data)

        all_features = {**det_features, **mod_features}

        fusion_result = self.fusion_inference.predict(
            all_features,
            return_attention=True,
        )

        anomaly_prob_val = fusion_result["anomaly_probs"][0]
        if isinstance(anomaly_prob_val, np.ndarray):
            anomaly_prob_val = anomaly_prob_val.item()
        elif hasattr(anomaly_prob_val, "item"):
            anomaly_prob_val = anomaly_prob_val.item()

        severity_val = fusion_result["severity_scores"][0]
        if isinstance(severity_val, np.ndarray):
            severity_val = severity_val.item()
        elif hasattr(severity_val, "item"):
            severity_val = severity_val.item()

        class_pred_val = fusion_result["class_predictions"][0]
        if isinstance(class_pred_val, np.ndarray):
            class_pred_val = class_pred_val.item()
        elif hasattr(class_pred_val, "item"):
            class_pred_val = class_pred_val.item()

        return {
            "anomaly_prob": float(anomaly_prob_val),
            "is_anomaly": bool(float(anomaly_prob_val) > 0.5),
            "class_prediction": int(class_pred_val),
            "severity": float(severity_val),
            "detector_importance": fusion_result.get("detector_importance", {}),
            "mode": "fusion",
        }

    def detect_biometric(
        self,
        reference_image: Union[str, np.ndarray],
        test_image: Optional[Union[str, np.ndarray]] = None,
        enable_age_progression: bool = False,
    ) -> Dict[str, Any]:
        """
        Biometric face matching and analysis.

        Args:
            reference_image: Reference face image
            test_image: Test image to match (if None, analyze reference only)
            enable_age_progression: Enable age progression estimation

        Returns:
            Biometric analysis results
        """
        biometric_model = self.models["biometric"]

        if test_image is not None:
            return biometric_model.predict(
                {
                    "reference": reference_image,
                    "test": test_image,
                }
            )
        else:
            return biometric_model.predict(reference_image)

    def detect_security_threat(
        self,
        payload: str,
        headers: Optional[Dict[str, str]] = None,
        source_ip: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Detect security threats in requests.

        Args:
            payload: Request payload
            headers: Request headers
            source_ip: Source IP address

        Returns:
            Threat detection results
        """
        threat_result = self.security.detect_all(payload)

        return {
            "is_anomaly": threat_result["is_threat"],
            "threats": threat_result["threats"],
            "source_ip": source_ip,
        }

    def train_fusion_model(
        self,
        training_data: str,
        validation_split: float = 0.2,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        optimizer_type: str = "adamw",
        early_stopping_patience: int = 10,
        checkpoint_dir: Optional[str] = None,
        use_mixed_precision: bool = False,
        gradient_accumulation_steps: int = 1,
    ) -> Dict[str, Any]:
        """
        Train the fusion model on custom data.

        Implements complete training pipeline with:
        - Data loading and preprocessing
        - Learning rate scheduling (ReduceLROnPlateau)
        - Early stopping with patience mechanism
        - Checkpoint saving with best model tracking
        - Optional mixed-precision training for performance
        - Gradient accumulation for effective larger batch sizes

        Args:
            training_data: Path to training data (numpy .npz or pickle file)
            validation_split: Validation split ratio (default 0.2)
            epochs: Number of training epochs (default 50)
            batch_size: Batch size (default 32)
            learning_rate: Learning rate (default 0.001)
            optimizer_type: Optimizer type ('adamw', 'ava_base', 'ava_momentum',
                           'ava_exp_decay', 'ava_harmonic')
            early_stopping_patience: Epochs to wait before early stopping (default 10)
            checkpoint_dir: Directory to save checkpoints (default None uses temp)
            use_mixed_precision: Enable mixed-precision training (default False)
            gradient_accumulation_steps: Steps for gradient accumulation (default 1)

        Returns:
            Dict containing training results with:
                - final_loss: Final validation loss
                - best_loss: Best validation loss achieved
                - epochs_trained: Number of epochs completed
                - checkpoint_path: Path to best model checkpoint
                - training_history: List of loss values per epoch

        Raises:
            ValueError: If training data path is invalid or data format unsupported
            RuntimeError: If training fails due to data or model issues
        """
        import os
        import pickle
        import tempfile
        from torch.utils.data import DataLoader, random_split
        from omni_anomaly_engine.ml.training import FusionTrainer, AnomalyDataset

        if self.mode != "fusion":
            raise ValueError("Training requires fusion mode. Initialize with mode='fusion'")

        # Validate parameters
        if gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")
        if not (0.0 < validation_split < 1.0):
            raise ValueError("validation_split must be between 0 and 1 (exclusive)")

        # Load training data
        if not os.path.exists(training_data):
            raise ValueError(f"Training data path does not exist: {training_data}")

        try:
            if training_data.endswith(".npz"):
                data = np.load(training_data, allow_pickle=True)
                features_dict = {
                    k: torch.tensor(v, dtype=torch.float32)
                    for k, v in data.items()
                    if k != "labels"
                }
                labels = torch.tensor(data["labels"], dtype=torch.long)
            elif training_data.endswith(".pkl") or training_data.endswith(".pickle"):
                with open(training_data, "rb") as f:
                    loaded = pickle.load(f)
                features_dict = {
                    k: torch.tensor(v, dtype=torch.float32) for k, v in loaded["features"].items()
                }
                labels = torch.tensor(loaded["labels"], dtype=torch.long)
            else:
                raise ValueError(f"Unsupported data format. Use .npz or .pkl: {training_data}")
        except Exception as e:
            raise RuntimeError(f"Failed to load training data: {e}")

        # Create dataset and split
        dataset = AnomalyDataset(features_dict, labels)
        total_size = len(dataset)
        val_size = int(total_size * validation_split)
        train_size = total_size - val_size

        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device.type == "cuda" else False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        # Setup checkpoint directory
        if checkpoint_dir is None:
            checkpoint_dir = tempfile.mkdtemp(prefix="omni_fusion_")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Create trainer module
        trainer_module = FusionTrainer(
            model=self.fusion_model,
            learning_rate=learning_rate,
        )
        trainer_module.optimizer_type = optimizer_type

        # Training state
        best_val_loss = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0
        training_history: List[Dict[str, float]] = []
        best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")

        # Setup mixed precision if requested
        scaler = (
            torch.cuda.amp.GradScaler()
            if use_mixed_precision and self.device.type == "cuda"
            else None
        )

        # Configure optimizer
        optimizer_config = trainer_module.configure_optimizers()
        optimizer = optimizer_config["optimizer"]
        scheduler = optimizer_config["lr_scheduler"]["scheduler"]

        self.fusion_model.train()

        for epoch in range(epochs):
            # Training phase
            train_losses = []
            self.fusion_model.train()

            for batch_idx, batch in enumerate(train_loader):
                if use_mixed_precision and scaler is not None:
                    with torch.cuda.amp.autocast():
                        loss = trainer_module.training_step(batch, batch_idx)
                    scaler.scale(loss / gradient_accumulation_steps).backward()

                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    loss = trainer_module.training_step(batch, batch_idx)
                    (loss / gradient_accumulation_steps).backward()

                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)

            # Validation phase
            val_losses = []
            self.fusion_model.eval()

            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    trainer_module.validation_step(batch, batch_idx)
                    # Calculate validation loss manually
                    features, labels_batch = batch
                    outputs = self.fusion_model(features, return_attention=True)
                    anomaly_labels = (labels_batch > 0).float().unsqueeze(1)
                    val_loss = torch.nn.functional.binary_cross_entropy(
                        outputs["anomaly_probs"], anomaly_labels
                    )
                    val_losses.append(val_loss.item())

            avg_val_loss = np.mean(val_losses) if val_losses else avg_train_loss

            # Update learning rate scheduler
            scheduler.step(avg_val_loss)

            # Track history
            training_history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

            # Check for improvement and save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.fusion_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": best_val_loss,
                    },
                    best_checkpoint_path,
                )
            else:
                epochs_without_improvement += 1

            # Early stopping
            if epochs_without_improvement >= early_stopping_patience:
                break

        # Load best model
        if os.path.exists(best_checkpoint_path):
            checkpoint = torch.load(
                best_checkpoint_path, map_location=self.device, weights_only=True
            )
            self.fusion_model.load_state_dict(checkpoint["model_state_dict"])

        self.fusion_model.eval()

        return {
            "final_loss": training_history[-1]["val_loss"] if training_history else 0.0,
            "best_loss": best_val_loss,
            "epochs_trained": len(training_history),
            "best_epoch": best_epoch,
            "checkpoint_path": best_checkpoint_path,
            "training_history": training_history,
            "early_stopped": epochs_without_improvement >= early_stopping_patience,
        }

    def save_model(self, path: str) -> None:
        """Save fusion model to file"""
        if self.mode == "fusion":
            torch.save(self.fusion_model.state_dict(), path)

    def load_model(self, path: str) -> None:
        """Load fusion model from file"""
        if self.mode == "fusion":
            self.fusion_model.load_state_dict(
                torch.load(path, map_location=self.device, weights_only=True)
            )

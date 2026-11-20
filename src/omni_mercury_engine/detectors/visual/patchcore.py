# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""PatchCore: Towards Total Recall in Industrial Anomaly Detection.

Implementation of PatchCore algorithm from CVPR 2022.
Achieves SOTA on MVTec AD with 99.1% image-level AUROC.

Key Innovations:
    1. Memory bank of locally-aware patch features
    2. Greedy coreset subsampling for efficiency (~90% reduction)
    3. Re-weighting for neighbourhood-aware scoring
    4. No training required - purely memory-based

Reference:
    Roth et al. "Towards Total Recall in Industrial Anomaly Detection"
    https://arxiv.org/abs/2106.08265
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch import nn

from omni_mercury_engine.detectors.provenance import DecisionSource
from omni_mercury_engine.detectors.visual.base_visual import (
    BaseVisualDetector,
    VisualDetectorConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class PatchCoreConfig(VisualDetectorConfig):
    """Configuration for PatchCore detector.

    Attributes:
        coreset_sampling_ratio: Fraction of patches to keep in memory bank
        coreset_ratio: Alias for coreset_sampling_ratio; resolved in
            ``__post_init__`` (setting either takes effect, the canonical
            field wins when both are customized and they disagree)
        num_neighbors: Number of memory items over which the image score is
            re-weighted (the paper's ``b``)
        k_nearest: Alias for num_neighbors; resolved like coreset_ratio
        faiss_on_gpu: Whether to use GPU for nearest neighbor search
        anomaly_score_num_nn: Number of neighbors for anomaly score computation
        seed: Seed of the coreset selection (random projection and starting
            point).  ``None`` draws it from OS entropy (a different memory
            bank on every fit).
    """

    coreset_sampling_ratio: float = 0.1  # Keep 10% of patches
    coreset_ratio: float = 0.1  # Alias, resolved in __post_init__
    num_neighbors: int = 9
    k_nearest: int = 9  # Alias, resolved in __post_init__
    faiss_on_gpu: bool = True
    anomaly_score_num_nn: int = 1
    layers: list[str] = field(default_factory=lambda: ["layer2", "layer3"])
    seed: int | None = None

    def __post_init__(self) -> None:
        """Resolve the alias fields so setting either name takes effect."""
        if self.coreset_ratio != 0.1 and self.coreset_sampling_ratio == 0.1:
            self.coreset_sampling_ratio = self.coreset_ratio
        self.coreset_ratio = self.coreset_sampling_ratio
        if self.k_nearest != 9 and self.num_neighbors == 9:
            self.num_neighbors = self.k_nearest
        self.k_nearest = self.num_neighbors


class PatchCoreDetector(BaseVisualDetector):
    """PatchCore anomaly detector.

    Memory-efficient patch-based anomaly detection using:
    - Pre-trained CNN features (WideResNet50 default)
    - Memory bank with greedy coreset subsampling
    - k-NN based anomaly scoring

    Achieves 99.1% image-level AUROC on MVTec AD benchmark.

    Example:
        >>> detector = PatchCoreDetector()
        >>> detector.fit(normal_images)  # [N, 3, 224, 224]
        >>> results = detector.detect(test_images)
        >>> print(results['scores'])  # Image-level scores
        >>> print(results['anomaly_maps'].shape)  # Pixel-level maps
    """

    decision_source = (
        DecisionSource(
            kind="published",
            description=(
                "The anomaly score is the re-weighted nearest-neighbour "
                "distance of pretrained patch features to a greedy-coreset "
                "memory bank, following the published PatchCore method."
            ),
            citation=(
                'Roth et al., "Towards Total Recall in Industrial Anomaly '
                'Detection", CVPR 2022, arXiv:2106.08265'
            ),
        ),
        DecisionSource(
            kind="calibrated",
            description=(
                "The binary decision threshold is calibrated in fit() by "
                "split conformal prediction on held-out normal images at "
                "visual_config.false_alarm_rate (default 0.05); the record "
                "is kept in self.calibration and preserved exactly by "
                "save()/load(). No decision is made before calibration."
            ),
            false_alarm_rate=0.05,
            calibration_data=(
                "held-out calibration_fraction (default 20%, seeded split) "
                "of the fit-time normal images, scored by "
                "core.conformal_prediction.split_conformal_threshold"
            ),
        ),
    )

    def __init__(self, config: PatchCoreConfig | dict[str, Any] | None = None) -> None:
        """Initialize PatchCore detector.

        Args:
            config: Detector configuration
        """
        if config is None:
            config = PatchCoreConfig()
        elif isinstance(config, dict):
            config = PatchCoreConfig(**config)

        super().__init__(config)
        self.patchcore_config: PatchCoreConfig = config
        # ``PatchCoreConfig.seed`` drives the greedy coreset-subsampling RNG
        # (the starting point); ``None`` is OS entropy.
        self._rng: np.random.Generator = np.random.default_rng(config.seed)

        # Initialize backbone
        self._init_backbone()

        # Memory bank (populated during fit)
        self.memory_bank: torch.Tensor | None = None
        self.memory_bank_spatial_info: list[tuple[int, int]] = []

        # Nearest neighbor index (FAISS or Mercury NearestNeighbors)
        self._nn_index: Any = None

        # Track feature extraction info
        self._patch_shapes: dict[str, tuple[int, int]] = {}

    def _aggregate_features(
        self, features: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        """Aggregate multi-scale features into patch embeddings.

        Uses average pooling to resize features to the same spatial
        resolution, then concatenates channel-wise.

        Args:
            features: Dict of layer features {layer_name: [B, C, H, W]}

        Returns:
            Tuple of:
                - Aggregated features [B, num_patches, total_channels]
                - Patch grid shape (h, w)
        """
        feature_list = list(features.values())

        # Find the reference spatial size (typically layer2)
        ref_size = feature_list[0].shape[-2:]

        # Resize all features to reference size
        resized_features = []
        for feat in feature_list:
            if feat.shape[-2:] != ref_size:
                feat = nn.functional.interpolate(
                    feat, size=ref_size, mode="bilinear", align_corners=False
                )
            resized_features.append(feat)

        # Concatenate along channel dimension: [B, sum(C_i), H, W]
        aggregated = torch.cat(resized_features, dim=1)

        # Reshape to patches: [B, H*W, C]
        batch_size, channels, h, w = aggregated.shape
        patches = aggregated.permute(0, 2, 3, 1).reshape(batch_size, h * w, channels)

        return patches, (h, w)

    def _apply_local_neighborhood_aggregation(
        self, patches: torch.Tensor, patch_shape: tuple[int, int], kernel_size: int = 3
    ) -> torch.Tensor:
        """Apply local neighborhood aggregation for locally-aware features.

        Averages features with their spatial neighbors for context.

        Args:
            patches: Patch features [B, num_patches, C]
            patch_shape: Spatial shape (h, w)
            kernel_size: Size of local neighborhood

        Returns:
            Aggregated patches [B, num_patches, C]
        """
        batch_size, num_patches, channels = patches.shape
        h, w = patch_shape

        # Reshape to spatial
        spatial = patches.view(batch_size, h, w, channels)
        spatial = spatial.permute(0, 3, 1, 2)  # [B, C, H, W]

        # Average pooling for neighborhood aggregation
        padding = kernel_size // 2
        aggregated = nn.functional.avg_pool2d(
            spatial, kernel_size=kernel_size, stride=1, padding=padding
        )

        # Reshape back to patches
        aggregated = aggregated.permute(0, 2, 3, 1).reshape(batch_size, num_patches, channels)

        return aggregated

    def _greedy_coreset_sampling(
        self, embeddings: torch.Tensor, sampling_ratio: float
    ) -> torch.Tensor:
        """Greedy k-centre coreset subsampling for memory efficiency.

        Runs on a seeded random projection to at most 24 dimensions with an
        incrementally updated min-distance vector (the pattern of
        ``vision.analyzer._greedy_coreset``), replacing the previous
        O(M*N*D) loop that also re-copied the selected set with ``vstack``
        on every step.  Selection is deterministic given ``config.seed``.

        Args:
            embeddings: All patch embeddings [N, D]
            sampling_ratio: Fraction of points to keep

        Returns:
            Selected coreset embeddings [M, D] where M = N * ratio
        """
        num_samples = embeddings.shape[0]
        target_size = max(1, int(num_samples * sampling_ratio))

        if target_size >= num_samples:
            return embeddings

        logger.info(
            f"Coreset sampling: {num_samples} -> {target_size} " f"({sampling_ratio * 100:.1f}%)"
        )

        embeddings_np = embeddings.detach().cpu().numpy()
        reduced_dim = min(embeddings_np.shape[1], 24)
        projection = self._rng.normal(size=(embeddings_np.shape[1], reduced_dim)) / np.sqrt(
            reduced_dim
        )
        projected = embeddings_np @ projection

        selected = np.empty(target_size, dtype=np.int64)
        selected[0] = int(self._rng.integers(num_samples))
        min_distances = np.sum((projected - projected[selected[0]]) ** 2, axis=1)
        for i in range(1, target_size):
            next_idx = int(np.argmax(min_distances))
            selected[i] = next_idx
            min_distances = np.minimum(
                min_distances, np.sum((projected - projected[next_idx]) ** 2, axis=1)
            )

        return embeddings[torch.from_numpy(selected).to(embeddings.device)]

    def _build_nn_index(self, embeddings: torch.Tensor) -> None:
        """Build nearest neighbor index for fast querying.

        Uses FAISS if available; otherwise the memory bank itself is kept as
        a float32 array and queried by the exact chunked brute-force path in
        :meth:`_chunked_knn` (the previous ``NearestNeighbors`` fallback
        built a dense float64 cdist and fully argsorted it per batch).

        Args:
            embeddings: Memory bank embeddings [N, D]
        """
        embeddings_np = embeddings.detach().cpu().numpy().astype(np.float32)

        try:
            import faiss

            dimension = embeddings_np.shape[1]

            if (
                self.patchcore_config.faiss_on_gpu
                and torch.cuda.is_available()
                and getattr(faiss, "GpuIndexFlatL2", None) is not None
            ):
                # GPU index
                res = faiss.StandardGpuResources()
                self._nn_index = faiss.GpuIndexFlatL2(res, dimension)
            else:
                # CPU index
                self._nn_index = faiss.IndexFlatL2(dimension)

            self._nn_index.add(embeddings_np)
            logger.info(f"Built FAISS index with {embeddings_np.shape[0]} vectors")

        except ImportError:
            logger.info("FAISS not available; using chunked exact k-NN on the memory bank")
            self._nn_index = embeddings_np

    @staticmethod
    def _chunked_knn(
        queries: np.ndarray[Any, Any], bank: np.ndarray[Any, Any], k: int, chunk: int = 1024
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Exact k-NN by chunked squared distances and ``argpartition``.

        Queries the bank in chunks (bounded memory) and partitions instead of
        fully sorting each row.  Distances are plain (non-squared) L2, the
        same metric the faiss branch is normalized to.

        Args:
            queries: Query embeddings [N, D]
            bank: Memory bank [M, D]
            k: Number of neighbors (clamped to the bank size)

        Returns:
            Tuple of (distances [N, k], indices [N, k]), nearest first.
        """
        k = min(k, bank.shape[0])
        bank64 = bank.astype(np.float64)
        bank_sq = np.sum(bank64 * bank64, axis=1)
        n_queries = queries.shape[0]
        distances = np.empty((n_queries, k), dtype=np.float64)
        indices = np.empty((n_queries, k), dtype=np.int64)
        for start in range(0, n_queries, chunk):
            block = queries[start : start + chunk].astype(np.float64)
            squared = np.maximum(
                np.sum(block * block, axis=1)[:, None] + bank_sq - 2.0 * (block @ bank64.T),
                0.0,
            )
            if k < squared.shape[1]:
                part = np.argpartition(squared, k - 1, axis=1)[:, :k]
            else:
                part = np.broadcast_to(np.arange(k), (squared.shape[0], k)).copy()
            part_sq = np.take_along_axis(squared, part, axis=1)
            order = np.argsort(part_sq, axis=1)
            indices[start : start + chunk] = np.take_along_axis(part, order, axis=1)
            distances[start : start + chunk] = np.sqrt(np.take_along_axis(part_sq, order, axis=1))
        return distances, indices

    def _query_nn(
        self, query: torch.Tensor, k: int
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Query k-nearest neighbors.

        Both branches return the same metric, plain (non-squared) L2:
        ``faiss.IndexFlatL2`` reports SQUARED L2, so its distances take a
        square root before they are returned.

        Args:
            query: Query embeddings [N, D]
            k: Number of neighbors

        Returns:
            Tuple of (distances, indices) arrays
        """
        query_np = query.detach().cpu().numpy().astype(np.float32)

        if not isinstance(self._nn_index, np.ndarray):
            try:
                import faiss

                # faiss-cpu ships no GpuIndexFlatL2: resolve the classes with
                # getattr so a missing attribute cannot raise mid-isinstance.
                flat_types = tuple(
                    cls
                    for cls in (
                        getattr(faiss, "IndexFlatL2", None),
                        getattr(faiss, "GpuIndexFlatL2", None),
                    )
                    if isinstance(cls, type)
                )
                if flat_types and isinstance(self._nn_index, flat_types):
                    searcher: Any = self._nn_index
                    squared, indices = searcher.search(query_np, k)
                    distances = np.sqrt(np.maximum(squared.astype(np.float64), 0.0))
                    return distances, indices
            except ImportError:
                pass
            # Unknown index type: fall back to the raw memory bank.
            if self.memory_bank is None:
                raise RuntimeError("No nearest-neighbour index or memory bank available")
            bank = self.memory_bank.detach().cpu().numpy().astype(np.float32)
            return self._chunked_knn(query_np, bank, k)

        return self._chunked_knn(query_np, self._nn_index, k)

    def fit(self, data: np.ndarray[Any, Any] | torch.Tensor) -> PatchCoreDetector:
        """Fit detector by building memory bank from normal images.

        Args:
            data: Normal (non-anomalous) images [N, C, H, W]

        Returns:
            Self for method chaining
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()

        data, calibration = self._split_for_calibration(data)
        data = self.preprocess(data)
        logger.info(f"Fitting PatchCore on {data.shape[0]} images")

        all_patches = []

        # Extract features in batches
        batch_size = self.patchcore_config.batch_size
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]

            with torch.no_grad():
                features = self.backbone(batch)

            # Aggregate multi-scale features
            patches, patch_shape = self._aggregate_features(features)

            # Apply local neighborhood aggregation
            patches = self._apply_local_neighborhood_aggregation(patches, patch_shape)

            # Flatten batch dimension
            patches = patches.reshape(-1, patches.shape[-1])
            all_patches.append(patches)

            self._patch_shapes = {layer: features[layer].shape[-2:] for layer in features}

        # Concatenate all patches
        all_patches_tensor = torch.cat(all_patches, dim=0)
        logger.info(f"Total patches before coreset: {all_patches_tensor.shape[0]}")

        # Apply coreset subsampling
        self.memory_bank = self._greedy_coreset_sampling(
            all_patches_tensor,
            self.patchcore_config.coreset_sampling_ratio,
        )
        logger.info(f"Memory bank size after coreset: {self.memory_bank.shape[0]}")

        # Build nearest neighbor index
        self._build_nn_index(self.memory_bank)

        self._is_fitted = True
        self._calibrate_threshold(calibration)
        return self

    def detect(self, data: np.ndarray[Any, Any] | torch.Tensor) -> dict[str, Any]:
        """Detect anomalies in images.

        Args:
            data: Test images [N, C, H, W]

        Returns:
            Dict containing:
                - scores: Image-level anomaly scores [N]
                - anomaly_maps: Pixel-level anomaly maps [N, H, W]
                - is_anomaly: Binary anomaly flags [N]
                - features: Extracted features for fusion [N, D]
        """
        if not self._is_fitted:
            raise RuntimeError("Detector must be fitted before detection")

        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()

        # Read the input's spatial size with the same channel-layout rule as
        # preprocess: shape[-2:] of a channel-last batch would be (W, C).
        original_size = self._input_spatial_size(data)
        data = self.preprocess(data)

        all_scores = []
        all_maps = []
        all_features = []

        batch_size = self.patchcore_config.batch_size
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]

            with torch.no_grad():
                features = self.backbone(batch)

            # Aggregate features
            patches, patch_shape = self._aggregate_features(features)
            patches = self._apply_local_neighborhood_aggregation(patches, patch_shape)

            batch_scores, batch_maps = self._compute_anomaly_scores(
                patches, patch_shape, original_size
            )

            all_scores.append(batch_scores)
            all_maps.append(batch_maps)

            # Extract features for fusion (global average of patches)
            all_features.append(patches.mean(dim=1))

        scores = torch.cat(all_scores, dim=0)
        anomaly_maps = torch.cat(all_maps, dim=0)
        features = torch.cat(all_features, dim=0)

        # Apply threshold
        is_anomaly = scores > self.threshold

        return {
            "scores": scores.cpu().numpy(),
            "anomaly_maps": anomaly_maps.cpu().numpy(),
            "is_anomaly": is_anomaly.cpu().numpy(),
            "features": features.cpu(),
        }

    def _compute_anomaly_scores(
        self,
        patches: torch.Tensor,
        patch_shape: tuple[int, int],
        original_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute anomaly scores using nearest neighbor distances.

        The image-level score follows the published re-weighting (Roth et
        al., eq. 7): the max-scoring patch's distance ``s*`` is scaled by
        ``w = 1 - softmax(d)[0]`` over its ``num_neighbors`` nearest memory
        distances ``d``, so a patch whose nearest memory item sits in a dense
        neighbourhood is down-weighted.

        Args:
            patches: Patch embeddings [B, num_patches, D]
            patch_shape: Spatial shape of patches (h, w)
            original_size: Original image size for upsampling

        Returns:
            Tuple of:
                - Image-level scores [B]
                - Pixel-level anomaly maps [B, H, W]
        """
        batch_size, _num_patches, dim = patches.shape
        h, w = patch_shape

        # Query nearest neighbors for each patch
        patches_flat = patches.reshape(-1, dim)
        distances, _ = self._query_nn(patches_flat, k=self.patchcore_config.anomaly_score_num_nn)

        # Use mean distance to k neighbors
        patch_scores = distances.mean(axis=1)
        patch_scores_t = torch.from_numpy(patch_scores).to(patches.device).float()

        # Reshape to spatial map
        score_maps = patch_scores_t.view(batch_size, h, w)

        # Image-level score: re-weighted distance of the max-scoring patch
        max_scores, max_indices = score_maps.view(batch_size, -1).max(dim=1)
        image_scores = max_scores
        num_neighbors = self.patchcore_config.num_neighbors
        bank_size = 0 if self.memory_bank is None else int(self.memory_bank.shape[0])
        if num_neighbors > 1 and bank_size > 1:
            max_patches = patches[torch.arange(batch_size, device=patches.device), max_indices]
            neighbor_distances, _ = self._query_nn(max_patches, k=min(num_neighbors, bank_size))
            neighbor_t = torch.from_numpy(neighbor_distances).to(patches.device).float()
            # w = 1 - exp(d0)/sum_j exp(dj), computed stably against d0 (the
            # smallest distance), so the exponentials cannot overflow.
            weights = 1.0 - 1.0 / torch.exp(neighbor_t - neighbor_t[:, :1]).sum(dim=1)
            image_scores = weights * max_scores

        # Upsample to original size
        score_maps_up = nn.functional.interpolate(
            score_maps.unsqueeze(1),
            size=original_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        # Apply Gaussian smoothing
        score_maps_np = score_maps_up.cpu().numpy()
        for i in range(batch_size):
            score_maps_np[i] = gaussian_filter(score_maps_np[i], sigma=4)

        score_maps_smooth = torch.from_numpy(score_maps_np).to(patches.device)

        return image_scores, score_maps_smooth

    def extract_features(self, data: np.ndarray[Any, Any] | torch.Tensor) -> torch.Tensor:
        """Extract features for ML fusion pipeline.

        Args:
            data: Input images [N, C, H, W]

        Returns:
            Feature tensor [N, 128] normalized for fusion
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()

        data = self.preprocess(data)

        all_features = []
        batch_size = self.patchcore_config.batch_size

        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]

            with torch.no_grad():
                features = self.backbone(batch)

            # Same locally-aware embedding as detect(): aggregation included.
            patches, patch_shape = self._aggregate_features(features)
            patches = self._apply_local_neighborhood_aggregation(patches, patch_shape)

            # Global pooling over patches
            global_feat = self._pool_for_fusion(patches, dim=1)  # [B, D]
            all_features.append(global_feat)

        features = torch.cat(all_features, dim=0)
        return self._project_features_for_fusion(features)

    def _fitted_state(self) -> dict[str, Any]:
        """The coreset memory bank -- PatchCore's whole fitted model."""
        return {"memory_bank": self.memory_bank}

    def _restore_fitted_state(self, state: dict[str, Any]) -> None:
        """Reinstate the memory bank and rebuild its nearest-neighbour index."""
        bank = state.get("memory_bank")
        if bank is not None:
            self.memory_bank = bank.to(self.device)
            self._build_nn_index(self.memory_bank)

    def get_memory_bank_size(self) -> int:
        """Get current memory bank size."""
        if self.memory_bank is None:
            return 0
        return int(self.memory_bank.shape[0])

    def get_memory_usage_mb(self) -> float:
        """Get memory bank memory usage in MB."""
        if self.memory_bank is None:
            return 0.0
        return float(self.memory_bank.numel() * 4 / (1024 * 1024))  # float32 = 4 bytes

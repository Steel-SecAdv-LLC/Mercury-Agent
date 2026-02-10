"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisors LLC

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
MVTec AD Dataset Loader.

MVTec Anomaly Detection Dataset for industrial defect detection.
Contains 15 categories of industrial objects and textures.

Reference:
    Bergmann et al., "MVTec AD - A Comprehensive Real-World Dataset
    for Unsupervised Anomaly Detection", CVPR 2019.
"""

from dataclasses import dataclass
from typing import Any

from .base_dataset import BaseDatasetConfig, BaseImageDataset

MVTEC_CATEGORIES: list[str] = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


@dataclass
class MVTecADConfig(BaseDatasetConfig):
    """Configuration for MVTec AD dataset.

    Attributes:
        root: Root directory for dataset
        category: Product category (e.g., 'bottle', 'cable', 'carpet')
        split: Dataset split ('train', 'test')
        transform: Optional transform function
        download: Whether to download if not present
        mask_transform: Optional transform for segmentation masks
        image_size: Target image size (height, width)
        normalize: Whether to apply ImageNet normalization
    """

    category: str = "bottle"
    mask_transform: Any = None
    image_size: tuple[int, int] = (224, 224)
    normalize: bool = True


class MVTecADDataset(BaseImageDataset):
    """MVTec Anomaly Detection Dataset.

    Industrial defect detection dataset with 15 categories.
    Training set contains only normal samples.
    Test set contains both normal and anomalous samples with pixel-level masks.

    Example:
        >>> from omni_mercury_engine.datasets.benchmarks import MVTecADDataset, MVTecADConfig
        >>> config = MVTecADConfig(root="./data/mvtec", category="bottle", split="test")
        >>> dataset = MVTecADDataset(config)
        >>> sample = dataset[0]
        >>> print(sample["label"])  # 0 for normal, 1 for anomaly
    """

    DATASET_URL = "https://www.mvtec.com/company/research/datasets/mvtec-ad"

    def __init__(self, config: MVTecADConfig | dict[str, Any] | None = None) -> None:
        """Initialize MVTec AD dataset.

        Args:
            config: Dataset configuration
        """
        if config is None:
            self.mvtec_config = MVTecADConfig()
        elif isinstance(config, dict):
            self.mvtec_config = MVTecADConfig(**config)
        else:
            self.mvtec_config = config

        self.category = self.mvtec_config.category
        self.mask_transform = self.mvtec_config.mask_transform

        super().__init__(self.mvtec_config)

    def _load_samples(self) -> None:
        """Load sample paths and labels from MVTec directory structure."""
        self._samples = []

        category_path = self.root / self.category / self.split

        if not category_path.exists():
            return

        if self.split == "train":
            good_path = category_path / "good"
            if good_path.exists():
                for img_path in sorted(good_path.glob("*.png")):
                    self._samples.append((img_path, 0, None))
        else:
            for defect_type in sorted(category_path.iterdir()):
                if not defect_type.is_dir():
                    continue

                is_normal = defect_type.name == "good"
                label = 0 if is_normal else 1

                mask_dir = self.root / self.category / "ground_truth" / defect_type.name

                for img_path in sorted(defect_type.glob("*.png")):
                    mask_path = None
                    if not is_normal and mask_dir.exists():
                        mask_name = img_path.stem + "_mask.png"
                        potential_mask = mask_dir / mask_name
                        if potential_mask.exists():
                            mask_path = potential_mask

                    self._samples.append((img_path, label, mask_path))

    def get_category_info(self) -> dict[str, Any]:
        """Get information about the current category.

        Returns:
            Dict with category statistics
        """
        normal_count = sum(1 for _, label, _ in self._samples if label == 0)
        anomaly_count = sum(1 for _, label, _ in self._samples if label == 1)

        return {
            "category": self.category,
            "split": self.split,
            "total_samples": len(self._samples),
            "normal_samples": normal_count,
            "anomaly_samples": anomaly_count,
        }

    @classmethod
    def get_categories(cls) -> list[str]:
        """Get list of all MVTec AD categories.

        Returns:
            List of category names
        """
        return MVTEC_CATEGORIES.copy()

    @classmethod
    def get_all_categories(cls) -> list[str]:
        """Get list of all MVTec AD categories (alias for get_categories).

        Returns:
            List of category names
        """
        return cls.get_categories()

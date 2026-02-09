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
Base classes for benchmark dataset loaders.

Provides abstract base classes for image and video anomaly detection datasets.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np  # noqa: TC002

try:
    import torch
    from torch.utils.data import Dataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    Dataset = object


def get_default_transforms(
    image_size: tuple[int, int] = (224, 224),
    normalize: bool = True,
) -> Any:
    """Get default image transforms for benchmark datasets.

    Args:
        image_size: Target image size (height, width)
        normalize: Whether to apply ImageNet normalization

    Returns:
        Composed transform function
    """
    try:
        from torchvision import transforms

        transform_list = [
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ]

        if normalize:
            transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            )

        return transforms.Compose(transform_list)
    except ImportError as e:
        raise ImportError("torchvision required for default transforms") from e


@dataclass
class BaseDatasetConfig:
    """Base configuration for benchmark datasets.

    Attributes:
        root: Root directory for dataset
        split: Dataset split ('train', 'test', 'all')
        transform: Optional transform function
        download: Whether to download if not present
    """

    root: str = "./data"
    split: str = "train"
    transform: Any = None
    download: bool = False


class BaseImageDataset(ABC):
    """Abstract base class for image anomaly detection datasets.

    Provides unified interface for datasets like MVTec AD.
    """

    def __init__(self, config: BaseDatasetConfig | dict[str, Any] | None = None) -> None:
        """Initialize dataset.

        Args:
            config: Dataset configuration
        """
        if config is None:
            self.config = BaseDatasetConfig()
        elif isinstance(config, dict):
            self.config = BaseDatasetConfig(**config)
        else:
            self.config = config

        self.root = Path(self.config.root)
        self.split = self.config.split
        self.transform = self.config.transform

        self._samples: list[tuple[Path, int, Path | None]] = []
        self._load_samples()

    @abstractmethod
    def _load_samples(self) -> None:
        """Load sample paths and labels."""
        pass

    def __len__(self) -> int:
        """Get number of samples."""
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a sample.

        Args:
            idx: Sample index

        Returns:
            Dict with 'image', 'label', and optionally 'mask'
        """
        image_path, label, mask_path = self._samples[idx]

        from PIL import Image

        # Use context manager to ensure file handle is released
        with Image.open(image_path) as img:
            image = img.convert("RGB")
            # Force load to memory before context exits
            image.load()

        if self.transform is not None:
            image = self.transform(image)

        result = {
            "image": image,
            "label": label,
            "image_path": str(image_path),
        }

        if mask_path is not None and mask_path.exists():
            with Image.open(mask_path) as msk:
                mask = msk.convert("L")
                mask.load()
            if self.transform is not None:
                from torchvision import transforms

                mask_transform = transforms.Compose(
                    [
                        transforms.Resize(
                            self.transform.transforms[0].size
                            if hasattr(self.transform, "transforms")
                            else (224, 224)
                        ),
                        transforms.ToTensor(),
                    ]
                )
                mask = mask_transform(mask)
            result["mask"] = mask

        return result


class BaseVideoDataset(ABC):
    """Abstract base class for video anomaly detection datasets.

    Provides unified interface for datasets like UCF-Crime, Shanghai Tech.
    """

    def __init__(self, config: BaseDatasetConfig | dict[str, Any] | None = None) -> None:
        """Initialize dataset.

        Args:
            config: Dataset configuration
        """
        if config is None:
            self.config = BaseDatasetConfig()
        elif isinstance(config, dict):
            self.config = BaseDatasetConfig(**config)
        else:
            self.config = config

        self.root = Path(self.config.root)
        self.split = self.config.split
        self.transform = self.config.transform

        self._videos: list[tuple[Path, int, np.ndarray[Any, Any] | None]] = []
        self._load_videos()

    @abstractmethod
    def _load_videos(self) -> None:
        """Load video paths and labels."""
        pass

    def __len__(self) -> int:
        """Get number of videos."""
        return len(self._videos)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a video sample.

        Args:
            idx: Video index

        Returns:
            Dict with 'frames', 'label', and optionally 'temporal_annotations'
        """
        video_path, label, temporal_annotations = self._videos[idx]

        frames = self._load_video_frames(video_path)

        if self.transform is not None:
            frames = [self.transform(f) for f in frames]
            if HAS_TORCH:
                frames = torch.stack(frames)

        result = {
            "frames": frames,
            "label": label,
            "video_path": str(video_path),
        }

        if temporal_annotations is not None:
            result["temporal_annotations"] = temporal_annotations

        return result

    def _load_video_frames(
        self,
        video_path: Path,
        max_frames: int | None = None,
    ) -> list[Any]:
        """Load frames from video file.

        Args:
            video_path: Path to video file
            max_frames: Maximum frames to load

        Returns:
            List of PIL Images
        """
        from PIL import Image

        frames = []

        if video_path.is_dir():
            frame_files = sorted(video_path.glob("*.jpg")) + sorted(video_path.glob("*.png"))
            for i, frame_path in enumerate(frame_files):
                if max_frames is not None and i >= max_frames:
                    break
                # Use context manager to prevent file descriptor exhaustion
                with Image.open(frame_path) as img:
                    frame = img.convert("RGB")
                    frame.load()
                frames.append(frame)
        else:
            try:
                import cv2

                cap = cv2.VideoCapture(str(video_path))
                try:
                    frame_count = 0
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        if max_frames is not None and frame_count >= max_frames:
                            break
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(Image.fromarray(frame_rgb))
                        frame_count += 1
                finally:
                    # Ensure release even if exception occurs
                    cap.release()
            except ImportError as e:
                raise ImportError("OpenCV required for video loading") from e

        return frames

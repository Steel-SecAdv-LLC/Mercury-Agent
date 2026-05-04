"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not,
see
https://www.gnu.org/licenses/.
"""

from __future__ import annotations

"""
UCF-Crime Dataset Loader.

UCF-Crime dataset for video anomaly detection in surveillance footage.
Contains 13 real-world anomaly categories.

Reference:
    Sultani et al., "Real-world Anomaly Detection in Surveillance Videos",
    CVPR 2018.
"""

from dataclasses import dataclass
from pathlib import Path  # noqa: TC003
from typing import Any

import numpy as np

from .base_dataset import BaseDatasetConfig, BaseVideoDataset

UCF_ANOMALY_TYPES: list[str] = [
    "Abuse",
    "Arrest",
    "Arson",
    "Assault",
    "Burglary",
    "Explosion",
    "Fighting",
    "RoadAccidents",
    "Robbery",
    "Shooting",
    "Shoplifting",
    "Stealing",
    "Vandalism",
]


@dataclass
class UCFCrimeConfig(BaseDatasetConfig):
    """Configuration for UCF-Crime dataset.

    Attributes:
        root: Root directory for dataset
        split: Dataset split ('train', 'test')
        transform: Optional transform function
        download: Whether to download if not present
        anomaly_types: List of anomaly types to include (None = all)
        max_frames: Maximum frames per video
        frame_stride: Stride for frame sampling
        frame_size: Target frame size (height, width)
        clip_length: Number of frames per clip
        fps: Frames per second for video processing
    """

    anomaly_types: list[str] | None = None
    max_frames: int | None = None
    frame_stride: int = 1
    frame_size: tuple[int, int] = (224, 224)
    clip_length: int = 16
    fps: float = 30.0


class UCFCrimeDataset(BaseVideoDataset):
    """UCF-Crime Video Anomaly Detection Dataset.

    Large-scale surveillance video dataset with 13 anomaly categories.
    Training uses weakly-supervised learning with video-level labels.
    Test set includes temporal annotations for anomaly segments.

    Example:
        >>> from omni_mercury_engine.datasets.benchmarks import UCFCrimeDataset, UCFCrimeConfig
        >>> config = UCFCrimeConfig(root="./data/ucf_crime", split="test")
        >>> dataset = UCFCrimeDataset(config)
        >>> sample = dataset[0]
        >>> print(sample["label"])  # 0 for normal, 1 for anomaly
    """

    DATASET_URL = "https://www.crcv.ucf.edu/projects/real-world/"

    def __init__(self, config: UCFCrimeConfig | dict[str, Any] | None = None) -> None:
        """
        Initialize UCF-Crime dataset.

        Args:
            config: Dataset configuration
        """
        if config is None:
            self.ucf_config = UCFCrimeConfig()
        elif isinstance(config, dict):
            self.ucf_config = UCFCrimeConfig(**config)
        else:
            self.ucf_config = config

        self.anomaly_types = self.ucf_config.anomaly_types
        self.max_frames = self.ucf_config.max_frames
        self.frame_stride = self.ucf_config.frame_stride

        super().__init__(self.ucf_config)

    def _load_videos(self) -> None:
        """Load video paths and labels from UCF-Crime directory structure."""
        self._videos = []

        split_path = self.root / self.split

        if not split_path.exists():
            split_path = self.root
            if not split_path.exists():
                return

        normal_path = split_path / "Normal"
        if normal_path.exists():
            for video_path in sorted(normal_path.glob("*.mp4")):
                self._videos.append((video_path, 0, None))
            for video_dir in sorted(normal_path.iterdir()):
                if video_dir.is_dir():
                    self._videos.append((video_dir, 0, None))

        anomaly_types = self.anomaly_types or UCF_ANOMALY_TYPES
        for anomaly_type in anomaly_types:
            anomaly_path = split_path / anomaly_type
            if not anomaly_path.exists():
                continue

            for video_path in sorted(anomaly_path.glob("*.mp4")):
                temporal_annotations = self._load_temporal_annotations(video_path)
                self._videos.append((video_path, 1, temporal_annotations))

            for video_dir in sorted(anomaly_path.iterdir()):
                if video_dir.is_dir():
                    temporal_annotations = self._load_temporal_annotations(video_dir)
                    self._videos.append((video_dir, 1, temporal_annotations))

    def _load_temporal_annotations(self, video_path: Path) -> np.ndarray[Any, Any] | None:
        """
        Load temporal annotations for a video.

        Args:
            video_path: Path to video file or directory

        Returns:
            Array of (start_frame, end_frame) tuples or None
        """
        annotation_file = video_path.with_suffix(".txt")
        if video_path.is_dir():
            annotation_file = video_path.parent / f"{video_path.name}.txt"

        if not annotation_file.exists():
            annotation_file = self.root / "annotations" / f"{video_path.stem}.txt"

        if annotation_file.exists():
            try:
                annotations = []
                with open(annotation_file) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            start, end = int(parts[0]), int(parts[1])
                            annotations.append((start, end))
                return np.array(annotations) if annotations else None
            except (OSError, ValueError):
                return None

        return None

    def get_anomaly_statistics(self) -> dict[str, Any]:
        """
        Get statistics about anomaly types in the dataset.

        Returns:
            Dict with anomaly type counts
        """
        by_type: dict[str, int] = {}
        stats: dict[str, Any] = {
            "total_videos": len(self._videos),
            "normal_videos": sum(1 for _, label, _ in self._videos if label == 0),
            "anomaly_videos": sum(1 for _, label, _ in self._videos if label == 1),
            "by_type": by_type,
        }

        for video_path, label, _ in self._videos:
            if label == 1:
                anomaly_type = video_path.parent.name
                by_type[anomaly_type] = by_type.get(anomaly_type, 0) + 1

        return stats

    @classmethod
    def get_anomaly_classes(cls) -> list[str]:
        """
        Get list of all UCF-Crime anomaly classes.

        Returns:
            List of anomaly class names
        """
        return UCF_ANOMALY_TYPES.copy()

    @classmethod
    def get_all_anomaly_types(cls) -> list[str]:
        """
        Get list of all UCF-Crime anomaly types (alias for get_anomaly_classes).

        Returns:
            List of anomaly type names
        """
        return cls.get_anomaly_classes()

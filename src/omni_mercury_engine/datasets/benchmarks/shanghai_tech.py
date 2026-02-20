"""
Mercury Agent
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
Shanghai Tech Campus Dataset Loader.

Shanghai Tech Campus dataset for video anomaly detection.
Contains surveillance footage from campus with various anomaly types.

Reference:
    Liu et al., "Future Frame Prediction for Anomaly Detection -
    A New Baseline", CVPR 2018.
"""

from dataclasses import dataclass
from pathlib import Path  # noqa: TC003
from typing import Any

import numpy as np

from .base_dataset import BaseDatasetConfig, BaseVideoDataset


@dataclass
class ShanghaiTechConfig(BaseDatasetConfig):
    """Configuration for Shanghai Tech Campus dataset.

    Attributes:
        root: Root directory for dataset
        split: Dataset split ('train', 'test')
        transform: Optional transform function
        download: Whether to download if not present
        scene: Specific scene to load (None = all scenes)
        max_frames: Maximum frames per video
        frame_stride: Stride for frame sampling
        frame_size: Target frame size (height, width)
        campus: Campus identifier ('campus1', 'campus2', etc.)
    """

    scene: str | None = None
    max_frames: int | None = None
    frame_stride: int = 1
    frame_size: tuple[int, int] = (224, 224)
    campus: str | None = None


class ShanghaiTechDataset(BaseVideoDataset):
    """Shanghai Tech Campus Video Anomaly Detection Dataset.

    Campus surveillance dataset with frame-level anomaly annotations.
    Contains 13 scenes with various anomaly types including:
    - Biking/skating in pedestrian areas
    - Vehicles in pedestrian areas
    - Fighting/chasing
    - Loitering

    Example:
        >>> from omni_mercury_engine.datasets.benchmarks import ShanghaiTechDataset, ShanghaiTechConfig
        >>> config = ShanghaiTechConfig(root="./data/shanghai_tech", split="test")
        >>> dataset = ShanghaiTechDataset(config)
        >>> sample = dataset[0]
        >>> print(sample["label"])  # 0 for normal, 1 for anomaly
    """

    DATASET_URL = "https://svip-lab.github.io/dataset/campus_dataset.html"

    NUM_SCENES = 13

    def __init__(self, config: ShanghaiTechConfig | dict[str, Any] | None = None) -> None:
        """Initialize Shanghai Tech dataset.

        Args:
            config: Dataset configuration
        """
        if config is None:
            self.shanghai_config = ShanghaiTechConfig()
        elif isinstance(config, dict):
            self.shanghai_config = ShanghaiTechConfig(**config)
        else:
            self.shanghai_config = config

        self.scene = self.shanghai_config.scene
        self.max_frames = self.shanghai_config.max_frames
        self.frame_stride = self.shanghai_config.frame_stride

        super().__init__(self.shanghai_config)

    def _load_videos(self) -> None:
        """Load video paths and labels from Shanghai Tech directory structure."""
        self._videos = []

        split_path = self.root / self.split

        if not split_path.exists():
            split_path = self.root
            if not split_path.exists():
                return

        if self.scene is not None:
            scenes = [self.scene]
        else:
            scenes = [f"{i:02d}" for i in range(1, self.NUM_SCENES + 1)]

        for scene_id in scenes:
            scene_path = split_path / scene_id
            if not scene_path.exists():
                scene_path = split_path / f"scene_{scene_id}"
            if not scene_path.exists():
                continue

            for video_dir in sorted(scene_path.iterdir()):
                if not video_dir.is_dir():
                    continue

                temporal_annotations = self._load_frame_annotations(video_dir)

                label = 1 if temporal_annotations is not None else 0

                self._videos.append((video_dir, label, temporal_annotations))

            for video_file in sorted(scene_path.glob("*.avi")) + sorted(scene_path.glob("*.mp4")):
                temporal_annotations = self._load_frame_annotations(video_file)
                label = 1 if temporal_annotations is not None else 0
                self._videos.append((video_file, label, temporal_annotations))

    def _load_frame_annotations(self, video_path: Path) -> np.ndarray[Any, Any] | None:
        """Load frame-level annotations for a video.

        Args:
            video_path: Path to video file or directory

        Returns:
            Array of anomaly frame indices or None
        """
        annotation_file = video_path.with_suffix(".txt")
        if video_path.is_dir():
            annotation_file = video_path / "annotations.txt"

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
                        elif len(parts) == 1:
                            frame_idx = int(parts[0])
                            annotations.append((frame_idx, frame_idx))
                return np.array(annotations) if annotations else None
            except (OSError, ValueError):
                return None

        return None

    def get_scene_statistics(self) -> dict[str, Any]:
        """Get statistics about scenes in the dataset.

        Returns:
            Dict with scene-level statistics
        """
        by_scene: dict[str, dict[str, int]] = {}
        stats: dict[str, Any] = {
            "total_videos": len(self._videos),
            "normal_videos": sum(1 for _, label, _ in self._videos if label == 0),
            "anomaly_videos": sum(1 for _, label, _ in self._videos if label == 1),
            "by_scene": by_scene,
        }

        for video_path, label, _ in self._videos:
            scene_id = video_path.parent.name
            if scene_id not in by_scene:
                by_scene[scene_id] = {"normal": 0, "anomaly": 0}
            if label == 0:
                by_scene[scene_id]["normal"] += 1
            else:
                by_scene[scene_id]["anomaly"] += 1

        return stats

    @classmethod
    def get_num_scenes(cls) -> int:
        """Get number of scenes in the dataset.

        Returns:
            Number of scenes
        """
        return cls.NUM_SCENES

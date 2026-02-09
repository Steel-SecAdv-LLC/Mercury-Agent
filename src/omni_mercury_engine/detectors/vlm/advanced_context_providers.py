"""
Mercury Agent
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
Advanced Context Providers for VLM-based Anomaly Detection.

Extends the base context provider system with:
- Semantic context: Scene-level understanding and object relationships
- Frequency context: Spectral analysis for periodic anomaly detection
- Appearance context: Color, texture, and visual statistics

These providers enhance VLM prompts with rich contextual information
to improve anomaly detection precision.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .context_providers import BaseContextProvider, ContextInfo

logger = logging.getLogger(__name__)


@dataclass
class SemanticFeatures:
    """Container for extracted semantic features."""

    scene_type: str
    complexity_score: float
    edge_density: float
    texture_uniformity: float
    object_count_estimate: int
    dominant_orientations: list[float]
    symmetry_score: float
    clutter_score: float


@dataclass
class FrequencyFeatures:
    """Container for frequency-domain features."""

    dominant_frequencies: list[tuple[float, float]]  # (frequency, magnitude)
    periodic_score: float
    noise_level: float
    spectral_centroid: float
    spectral_spread: float
    temporal_periodicity: float | None = None
    flicker_detected: bool = False


@dataclass
class AppearanceFeatures:
    """Container for appearance-based features."""

    color_histogram: np.ndarray
    dominant_colors: list[tuple[tuple[int, int, int], float]]
    color_variance: float
    brightness_mean: float
    brightness_std: float
    contrast: float
    saturation_mean: float
    texture_energy: float
    texture_entropy: float
    gradient_magnitude_mean: float


class SemanticContextProvider(BaseContextProvider):
    """
    Semantic context provider for scene-level understanding.

    Extracts high-level semantic features without requiring
    deep learning models - uses classical CV techniques for
    fast, interpretable scene analysis.

    Features extracted:
    - Scene complexity (edge density, texture uniformity)
    - Object count estimation (connected components)
    - Spatial organization (symmetry, clutter)
    - Dominant orientations (gradient analysis)
    """

    def __init__(
        self,
        complexity_threshold: float = 0.5,
        edge_kernel_size: int = 3,
        random_state: int | None = None,
    ):
        """
        Initialize semantic context provider.

        Args:
            complexity_threshold: Threshold for complex vs simple scenes
            edge_kernel_size: Kernel size for edge detection
            random_state: Seed for reproducible random sampling
        """
        self.complexity_threshold = complexity_threshold
        self.edge_kernel_size = edge_kernel_size
        self.rng = np.random.default_rng(random_state)

        # Scene type thresholds
        self._scene_thresholds = {
            "sparse": (0.0, 0.2),
            "moderate": (0.2, 0.5),
            "dense": (0.5, 0.8),
            "cluttered": (0.8, 1.0),
        }

    def extract_context(
        self,
        frames: np.ndarray | torch.Tensor,
        **kwargs: Any,
    ) -> ContextInfo:
        """
        Extract semantic context from input frames.

        Args:
            frames: Input frames [T, C, H, W] or [C, H, W]

        Returns:
            Semantic context information
        """
        if isinstance(frames, torch.Tensor):
            frames = frames.cpu().numpy()

        if frames.ndim == 3:
            frames = frames[np.newaxis, ...]

        # Use middle frame for semantic analysis
        mid_idx = len(frames) // 2
        frame = frames[mid_idx]

        # Extract features
        features = self._extract_semantic_features(frame)

        # Build description
        description = self._build_semantic_description(features)

        return ContextInfo(
            context_type="semantic",
            description=description,
            features=np.array(
                [
                    features.complexity_score,
                    features.edge_density,
                    features.texture_uniformity,
                    features.object_count_estimate,
                    features.symmetry_score,
                    features.clutter_score,
                ]
            ),
            metadata={
                "scene_type": features.scene_type,
                "complexity_score": features.complexity_score,
                "object_count_estimate": features.object_count_estimate,
                "dominant_orientations": features.dominant_orientations,
            },
        )

    def _extract_semantic_features(self, frame: np.ndarray) -> SemanticFeatures:
        """Extract semantic features from a single frame."""
        # Convert to grayscale if needed
        if frame.ndim == 3 and frame.shape[0] in [1, 3]:
            if frame.shape[0] == 3:
                gray = 0.299 * frame[0] + 0.587 * frame[1] + 0.114 * frame[2]
            else:
                gray = frame[0]
        else:
            gray = frame

        # Normalize to 0-1 range
        if gray.max() > 1.0:
            gray = gray / 255.0

        # Edge detection using Sobel-like gradients
        edge_density, grad_x, grad_y = self._compute_edge_density(gray)

        # Texture uniformity
        texture_uniformity = self._compute_texture_uniformity(gray)

        # Object count estimation
        object_count = self._estimate_object_count(gray, edge_density)

        # Dominant orientations from gradients
        orientations = self._compute_dominant_orientations(grad_x, grad_y)

        # Symmetry score
        symmetry = self._compute_symmetry(gray)

        # Clutter score (combination of edge density and object count)
        clutter = self._compute_clutter_score(edge_density, object_count, texture_uniformity)

        # Complexity score (overall scene complexity)
        complexity = 0.4 * edge_density + 0.3 * clutter + 0.3 * (1 - texture_uniformity)

        # Determine scene type
        scene_type = self._classify_scene(complexity)

        return SemanticFeatures(
            scene_type=scene_type,
            complexity_score=float(complexity),
            edge_density=float(edge_density),
            texture_uniformity=float(texture_uniformity),
            object_count_estimate=object_count,
            dominant_orientations=orientations,
            symmetry_score=float(symmetry),
            clutter_score=float(clutter),
        )

    def _compute_edge_density(self, gray: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        """Compute edge density using gradient magnitude."""
        # Sobel-like kernels
        k = self.edge_kernel_size
        pad = k // 2

        # Pad image
        padded = np.pad(gray, pad, mode="reflect")

        # Simple gradient computation
        grad_x = np.zeros_like(gray)
        grad_y = np.zeros_like(gray)

        for i in range(gray.shape[0]):
            for j in range(gray.shape[1]):
                # Central difference
                grad_x[i, j] = (padded[i + pad, j + pad + 1] - padded[i + pad, j + pad - 1]) / 2
                grad_y[i, j] = (padded[i + pad + 1, j + pad] - padded[i + pad - 1, j + pad]) / 2

        # Gradient magnitude
        mag = np.sqrt(grad_x**2 + grad_y**2)

        # Edge density is the mean gradient magnitude normalized
        density = mag.mean() / (mag.max() + 1e-8)

        return density, grad_x, grad_y

    def _compute_texture_uniformity(self, gray: np.ndarray) -> float:
        """Compute texture uniformity using local variance."""
        # Compute local variance in 8x8 blocks
        h, w = gray.shape
        block_size = 8
        variances = []

        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = gray[i : i + block_size, j : j + block_size]
                variances.append(block.var())

        if not variances:
            return 0.5

        # Uniformity is inverse of variance of variances
        var_of_var = np.var(variances)
        uniformity = 1.0 / (1.0 + var_of_var * 100)

        return float(uniformity)

    def _estimate_object_count(self, gray: np.ndarray, edge_density: float) -> int:
        """Estimate object count using connected components on edges."""
        # Simple thresholding based on edge density
        threshold = 0.1 + edge_density * 0.2

        # Compute gradient magnitude
        grad_x = np.diff(gray, axis=1, prepend=gray[:, :1])
        grad_y = np.diff(gray, axis=0, prepend=gray[:1, :])
        mag = np.sqrt(grad_x**2 + grad_y**2)

        # Threshold to binary
        binary = (mag > threshold).astype(np.uint8)

        # Simple connected components (4-connectivity)
        labels = np.zeros_like(binary, dtype=np.int32)
        current_label = 0

        for i in range(binary.shape[0]):
            for j in range(binary.shape[1]):
                if binary[i, j] == 1 and labels[i, j] == 0:
                    current_label += 1
                    self._flood_fill(binary, labels, i, j, current_label)

        # Filter small components (noise)
        unique, counts = np.unique(labels, return_counts=True)
        min_size = binary.size * 0.001  # At least 0.1% of image
        significant_objects = sum(1 for c in counts[1:] if c > min_size)

        return min(significant_objects, 50)  # Cap at 50

    def _flood_fill(
        self,
        binary: np.ndarray,
        labels: np.ndarray,
        start_i: int,
        start_j: int,
        label: int,
    ) -> None:
        """Simple flood fill for connected components."""
        stack = [(start_i, start_j)]
        h, w = binary.shape

        while stack:
            i, j = stack.pop()
            if i < 0 or i >= h or j < 0 or j >= w:
                continue
            if binary[i, j] == 0 or labels[i, j] != 0:
                continue

            labels[i, j] = label
            stack.extend([(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)])

    def _compute_dominant_orientations(
        self, grad_x: np.ndarray, grad_y: np.ndarray, num_bins: int = 8
    ) -> list[float]:
        """Compute dominant gradient orientations."""
        # Compute angles
        angles = np.arctan2(grad_y, grad_x)
        magnitudes = np.sqrt(grad_x**2 + grad_y**2)

        # Create histogram
        bins = np.linspace(-np.pi, np.pi, num_bins + 1)
        hist, _ = np.histogram(angles, bins=bins, weights=magnitudes)

        # Find top 3 orientations
        top_indices = np.argsort(hist)[-3:][::-1]
        dominant = []
        for idx in top_indices:
            if hist[idx] > hist.sum() * 0.1:  # At least 10% of total
                angle_deg = (bins[idx] + bins[idx + 1]) / 2 * 180 / np.pi
                dominant.append(float(angle_deg))

        return dominant

    def _compute_symmetry(self, gray: np.ndarray) -> float:
        """Compute bilateral symmetry score."""
        h, w = gray.shape

        # Horizontal symmetry
        left = gray[:, : w // 2]
        right = gray[:, w // 2 : w // 2 + left.shape[1]][:, ::-1]
        h_sym = 1.0 - np.abs(left - right).mean()

        # Vertical symmetry
        top = gray[: h // 2, :]
        bottom = gray[h // 2 : h // 2 + top.shape[0], :][::-1, :]
        v_sym = 1.0 - np.abs(top - bottom).mean()

        return float((h_sym + v_sym) / 2)

    def _compute_clutter_score(
        self, edge_density: float, object_count: int, texture_uniformity: float
    ) -> float:
        """Compute scene clutter score."""
        # Normalize object count (assume 20+ is very cluttered)
        obj_score = min(object_count / 20.0, 1.0)

        # Combine factors
        clutter = 0.4 * edge_density + 0.4 * obj_score + 0.2 * (1 - texture_uniformity)

        return float(clutter)

    def _classify_scene(self, complexity: float) -> str:
        """Classify scene type based on complexity."""
        for scene_type, (low, high) in self._scene_thresholds.items():
            if low <= complexity < high:
                return scene_type
        return "cluttered"

    def _build_semantic_description(self, features: SemanticFeatures) -> str:
        """Build natural language semantic description."""
        parts = []

        # Scene type
        type_descriptions = {
            "sparse": "The scene is relatively sparse with few visual elements.",
            "moderate": "The scene has a moderate amount of visual content.",
            "dense": "The scene is visually dense with many elements.",
            "cluttered": "The scene is highly cluttered with numerous overlapping elements.",
        }
        parts.append(type_descriptions.get(features.scene_type, ""))

        # Object count
        if features.object_count_estimate <= 3:
            parts.append(
                f"Approximately {features.object_count_estimate} distinct objects are visible."
            )
        elif features.object_count_estimate <= 10:
            parts.append(f"Several objects ({features.object_count_estimate}) are present.")
        else:
            parts.append(
                f"Many objects (approximately {features.object_count_estimate}) populate the scene."
            )

        # Symmetry
        if features.symmetry_score > 0.7:
            parts.append("The composition shows strong symmetry.")
        elif features.symmetry_score < 0.3:
            parts.append("The layout is asymmetric.")

        # Dominant orientations
        if features.dominant_orientations:
            if len(features.dominant_orientations) == 1:
                angle = features.dominant_orientations[0]
                if -10 < angle < 10 or abs(angle) > 170:
                    parts.append("Horizontal structures dominate.")
                elif 80 < abs(angle) < 100:
                    parts.append("Vertical structures dominate.")
                else:
                    parts.append(f"Diagonal structures at ~{angle:.0f} are prominent.")

        return " ".join(parts)

    def format_context_prompt(self, context: ContextInfo) -> str:
        """Format semantic context as prompt addition."""
        return f"\n[Semantic Context: {context.description}]"


class FrequencyContextProvider(BaseContextProvider):
    """
    Frequency-domain context provider for periodic pattern detection.

    Analyzes spectral characteristics to detect:
    - Periodic patterns (stripes, grids, textures)
    - Temporal periodicity (flickering, oscillations)
    - Noise levels and spectral distribution

    Useful for detecting anomalies like:
    - Machine vibrations
    - Flickering lights
    - Periodic interference patterns
    - Regular vs irregular motion
    """

    def __init__(
        self,
        frequency_bins: int = 32,
        periodicity_threshold: float = 0.3,
        flicker_threshold: float = 0.2,
    ):
        """
        Initialize frequency context provider.

        Args:
            frequency_bins: Number of frequency bins for analysis
            periodicity_threshold: Threshold for detecting periodic content
            flicker_threshold: Threshold for detecting temporal flicker
        """
        self.frequency_bins = frequency_bins
        self.periodicity_threshold = periodicity_threshold
        self.flicker_threshold = flicker_threshold

    def extract_context(
        self,
        frames: np.ndarray | torch.Tensor,
        **kwargs: Any,
    ) -> ContextInfo:
        """
        Extract frequency-domain context.

        Args:
            frames: Input frames [T, C, H, W] or [C, H, W]

        Returns:
            Frequency context information
        """
        if isinstance(frames, torch.Tensor):
            frames = frames.cpu().numpy()

        if frames.ndim == 3:
            frames = frames[np.newaxis, ...]

        # Extract frequency features
        features = self._extract_frequency_features(frames)

        # Build description
        description = self._build_frequency_description(features)

        # Prepare feature vector
        feature_vector = np.array(
            [
                features.periodic_score,
                features.noise_level,
                features.spectral_centroid,
                features.spectral_spread,
                features.temporal_periodicity or 0.0,
                float(features.flicker_detected),
            ]
        )

        return ContextInfo(
            context_type="frequency",
            description=description,
            features=feature_vector,
            metadata={
                "dominant_frequencies": features.dominant_frequencies,
                "periodic_score": features.periodic_score,
                "flicker_detected": features.flicker_detected,
                "temporal_periodicity": features.temporal_periodicity,
            },
        )

    def _extract_frequency_features(self, frames: np.ndarray) -> FrequencyFeatures:
        """Extract frequency-domain features."""
        t, c, h, w = frames.shape

        # Convert to grayscale
        if c == 3:
            gray_frames = 0.299 * frames[:, 0] + 0.587 * frames[:, 1] + 0.114 * frames[:, 2]
        else:
            gray_frames = frames[:, 0]

        # Spatial frequency analysis (on middle frame)
        mid_frame = gray_frames[t // 2]
        spatial_features = self._analyze_spatial_frequency(mid_frame)

        # Temporal frequency analysis (if multiple frames)
        temporal_periodicity = None
        flicker_detected = False
        if t > 1:
            temporal_periodicity, flicker_detected = self._analyze_temporal_frequency(gray_frames)

        return FrequencyFeatures(
            dominant_frequencies=spatial_features["dominant_frequencies"],
            periodic_score=spatial_features["periodic_score"],
            noise_level=spatial_features["noise_level"],
            spectral_centroid=spatial_features["spectral_centroid"],
            spectral_spread=spatial_features["spectral_spread"],
            temporal_periodicity=temporal_periodicity,
            flicker_detected=flicker_detected,
        )

    def _analyze_spatial_frequency(self, frame: np.ndarray) -> dict[str, Any]:
        """Analyze spatial frequency content using FFT."""
        # Normalize
        if frame.max() > 1.0:
            frame = frame / 255.0

        # 2D FFT
        fft = np.fft.fft2(frame)
        fft_shifted = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shifted)

        h, w = frame.shape

        # Create frequency coordinate grid
        y_freq = np.fft.fftshift(np.fft.fftfreq(h))
        x_freq = np.fft.fftshift(np.fft.fftfreq(w))
        freq_grid = np.sqrt(y_freq[:, np.newaxis] ** 2 + x_freq[np.newaxis, :] ** 2)

        # Radial averaging for power spectrum
        freq_bins = np.linspace(0, 0.5, self.frequency_bins + 1)
        power_spectrum = []

        for i in range(self.frequency_bins):
            mask = (freq_grid >= freq_bins[i]) & (freq_grid < freq_bins[i + 1])
            if mask.sum() > 0:
                power_spectrum.append(magnitude[mask].mean())
            else:
                power_spectrum.append(0)

        power_spectrum = np.array(power_spectrum)

        # Find dominant frequencies (peaks)
        # Exclude DC component (bin 0) from mean calculation for peak detection
        # as it typically dominates and masks periodic patterns
        ac_mean = power_spectrum[1:].mean() if len(power_spectrum) > 1 else power_spectrum.mean()
        dominant_frequencies = []
        for i in range(1, len(power_spectrum) - 1):
            if (
                power_spectrum[i] > power_spectrum[i - 1]
                and power_spectrum[i] > power_spectrum[i + 1]
                and power_spectrum[i] > ac_mean * 1.5
            ):
                freq = (freq_bins[i] + freq_bins[i + 1]) / 2
                dominant_frequencies.append((float(freq), float(power_spectrum[i])))

        # Sort by magnitude
        dominant_frequencies.sort(key=lambda x: x[1], reverse=True)
        dominant_frequencies = dominant_frequencies[:5]

        # Periodic score (ratio of dominant peaks to AC power, excluding DC)
        # This provides a more meaningful measure of periodicity
        total_power = power_spectrum.sum()
        ac_power = power_spectrum[1:].sum() if len(power_spectrum) > 1 else total_power
        peak_power = sum(p[1] for p in dominant_frequencies[:3])
        periodic_score = float(peak_power / (ac_power + 1e-8))

        # Noise level (high frequency content ratio)
        high_freq_idx = int(self.frequency_bins * 0.7)
        noise_level = float(power_spectrum[high_freq_idx:].sum() / (total_power + 1e-8))

        # Spectral centroid
        freq_centers = (freq_bins[:-1] + freq_bins[1:]) / 2
        spectral_centroid = float(np.sum(freq_centers * power_spectrum) / (total_power + 1e-8))

        # Spectral spread
        spectral_spread = float(
            np.sqrt(
                np.sum(((freq_centers - spectral_centroid) ** 2) * power_spectrum)
                / (total_power + 1e-8)
            )
        )

        return {
            "dominant_frequencies": dominant_frequencies,
            "periodic_score": periodic_score,
            "noise_level": noise_level,
            "spectral_centroid": spectral_centroid,
            "spectral_spread": spectral_spread,
        }

    def _analyze_temporal_frequency(self, gray_frames: np.ndarray) -> tuple[float | None, bool]:
        """Analyze temporal frequency content."""
        t = len(gray_frames)

        if t < 4:
            return None, False

        # Compute frame-to-frame differences
        diffs = np.diff(gray_frames.astype(float), axis=0)
        mean_diffs = diffs.mean(axis=(1, 2))

        # Temporal FFT
        temporal_fft = np.fft.fft(mean_diffs)
        temporal_mag = np.abs(temporal_fft[: len(temporal_fft) // 2])

        if len(temporal_mag) < 2:
            return None, False

        # Find dominant temporal frequency
        peak_idx = np.argmax(temporal_mag[1:]) + 1  # Skip DC

        # Periodicity score
        total_power = temporal_mag.sum()
        peak_power = temporal_mag[peak_idx]
        periodicity = float(peak_power / (total_power + 1e-8))

        # Detect flicker (high temporal variation)
        flicker = float(mean_diffs.std()) > self.flicker_threshold

        return periodicity if periodicity > self.periodicity_threshold else None, flicker

    def _build_frequency_description(self, features: FrequencyFeatures) -> str:
        """Build natural language frequency description."""
        parts = []

        # Periodic content
        if features.periodic_score > 0.5:
            parts.append("Strong periodic patterns detected in the image.")
        elif features.periodic_score > 0.2:
            parts.append("Some periodic/repetitive structures visible.")
        else:
            parts.append("Image has aperiodic, natural appearance.")

        # Dominant frequencies
        if features.dominant_frequencies:
            freq = features.dominant_frequencies[0][0]
            if freq < 0.1:
                parts.append("Low-frequency (large-scale) features dominate.")
            elif freq > 0.3:
                parts.append("High-frequency (fine detail) content is prominent.")

        # Noise level
        if features.noise_level > 0.3:
            parts.append("High noise or fine texture present.")
        elif features.noise_level < 0.1:
            parts.append("Image appears clean with smooth regions.")

        # Temporal features
        if features.flicker_detected:
            parts.append("Temporal flickering or rapid changes detected.")
        if features.temporal_periodicity:
            parts.append(f"Periodic motion with score {features.temporal_periodicity:.2f}.")

        return " ".join(parts)

    def format_context_prompt(self, context: ContextInfo) -> str:
        """Format frequency context as prompt addition."""
        return f"\n[Frequency Analysis: {context.description}]"


class AppearanceContextProvider(BaseContextProvider):
    """
    Appearance context provider for color and texture analysis.

    Extracts visual appearance features:
    - Color distribution and dominant colors
    - Brightness and contrast statistics
    - Texture characteristics (energy, entropy)

    Useful for detecting:
    - Color anomalies (unusual tints, discoloration)
    - Lighting anomalies (over/under exposure)
    - Surface defects (texture irregularities)
    """

    def __init__(
        self,
        color_bins: int = 16,
        num_dominant_colors: int = 5,
    ):
        """
        Initialize appearance context provider.

        Args:
            color_bins: Number of bins per color channel
            num_dominant_colors: Number of dominant colors to extract
        """
        self.color_bins = color_bins
        self.num_dominant_colors = num_dominant_colors

    def extract_context(
        self,
        frames: np.ndarray | torch.Tensor,
        **kwargs: Any,
    ) -> ContextInfo:
        """
        Extract appearance context.

        Args:
            frames: Input frames [T, C, H, W] or [C, H, W]

        Returns:
            Appearance context information
        """
        if isinstance(frames, torch.Tensor):
            frames = frames.cpu().numpy()

        if frames.ndim == 3:
            frames = frames[np.newaxis, ...]

        # Use middle frame
        mid_idx = len(frames) // 2
        frame = frames[mid_idx]

        # Extract features
        features = self._extract_appearance_features(frame)

        # Build description
        description = self._build_appearance_description(features)

        # Feature vector
        feature_vector = np.concatenate(
            [
                [features.brightness_mean, features.brightness_std],
                [features.contrast, features.saturation_mean],
                [features.texture_energy, features.texture_entropy],
                [features.gradient_magnitude_mean, features.color_variance],
            ]
        )

        return ContextInfo(
            context_type="appearance",
            description=description,
            features=feature_vector,
            metadata={
                "dominant_colors": features.dominant_colors,
                "brightness_mean": features.brightness_mean,
                "contrast": features.contrast,
                "texture_entropy": features.texture_entropy,
            },
        )

    def _extract_appearance_features(self, frame: np.ndarray) -> AppearanceFeatures:
        """Extract appearance features from frame."""
        c, h, w = frame.shape

        # Normalize to 0-255 range for color analysis
        if frame.max() <= 1.0:
            frame_uint8 = (frame * 255).astype(np.uint8)
        else:
            frame_uint8 = frame.astype(np.uint8)

        # Color histogram
        color_histogram = self._compute_color_histogram(frame_uint8)

        # Dominant colors
        dominant_colors = self._find_dominant_colors(frame_uint8)

        # Color variance
        color_variance = self._compute_color_variance(frame_uint8)

        # Brightness statistics
        if c == 3:
            gray = 0.299 * frame[0] + 0.587 * frame[1] + 0.114 * frame[2]
        else:
            gray = frame[0]

        if gray.max() > 1.0:
            gray = gray / 255.0

        brightness_mean = float(gray.mean())
        brightness_std = float(gray.std())

        # Contrast (using percentiles)
        p5, p95 = np.percentile(gray, [5, 95])
        contrast = float(p95 - p5)

        # Saturation (if color image)
        if c == 3:
            saturation = self._compute_saturation(frame_uint8)
        else:
            saturation = 0.0

        # Texture features
        texture_energy, texture_entropy = self._compute_texture_features(gray)

        # Gradient magnitude
        grad_x = np.diff(gray, axis=1, prepend=gray[:, :1])
        grad_y = np.diff(gray, axis=0, prepend=gray[:1, :])
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        gradient_magnitude_mean = float(grad_mag.mean())

        return AppearanceFeatures(
            color_histogram=color_histogram,
            dominant_colors=dominant_colors,
            color_variance=color_variance,
            brightness_mean=brightness_mean,
            brightness_std=brightness_std,
            contrast=contrast,
            saturation_mean=saturation,
            texture_energy=texture_energy,
            texture_entropy=texture_entropy,
            gradient_magnitude_mean=gradient_magnitude_mean,
        )

    def _compute_color_histogram(self, frame: np.ndarray) -> np.ndarray:
        """Compute 3D color histogram."""
        c, h, w = frame.shape

        if c == 3:
            # Quantize colors
            quantized = (frame // (256 // self.color_bins)).astype(np.int32)

            # Build histogram
            hist = np.zeros((self.color_bins, self.color_bins, self.color_bins))
            for i in range(h):
                for j in range(w):
                    r, g, b = quantized[:, i, j]
                    r = min(r, self.color_bins - 1)
                    g = min(g, self.color_bins - 1)
                    b = min(b, self.color_bins - 1)
                    hist[r, g, b] += 1

            # Normalize
            hist = hist / (h * w)
            return hist.flatten()
        else:
            # Grayscale histogram
            hist, _ = np.histogram(frame.flatten(), bins=self.color_bins, range=(0, 256))
            return hist / hist.sum()

    def _find_dominant_colors(self, frame: np.ndarray) -> list[tuple[tuple[int, int, int], float]]:
        """Find dominant colors using simple binning."""
        c, h, w = frame.shape

        if c != 3:
            # Grayscale
            mean_val = int(frame.mean())
            return [((mean_val, mean_val, mean_val), 1.0)]

        # Quantize to fewer colors
        bin_size = 32
        quantized = (frame // bin_size) * bin_size + bin_size // 2

        # Count color occurrences
        color_counts: dict[tuple[int, int, int], int] = {}
        for i in range(h):
            for j in range(w):
                color = tuple(quantized[:, i, j].tolist())
                color_counts[color] = color_counts.get(color, 0) + 1

        # Sort by count
        total = h * w
        sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)

        dominant = []
        for color, count in sorted_colors[: self.num_dominant_colors]:
            dominant.append((color, count / total))

        return dominant

    def _compute_color_variance(self, frame: np.ndarray) -> float:
        """Compute variance across color channels."""
        c, h, w = frame.shape

        if c == 3:
            # Variance of each channel
            var_r = frame[0].var()
            var_g = frame[1].var()
            var_b = frame[2].var()
            return float((var_r + var_g + var_b) / 3)
        else:
            return float(frame.var())

    def _compute_saturation(self, frame: np.ndarray) -> float:
        """Compute mean saturation for color image."""
        # Simple saturation: (max - min) / max for each pixel
        c_max = frame.max(axis=0)
        c_min = frame.min(axis=0)

        # Avoid division by zero
        saturation = np.where(c_max > 0, (c_max - c_min) / (c_max + 1e-8), 0)

        return float(saturation.mean())

    def _compute_texture_features(self, gray: np.ndarray) -> tuple[float, float]:
        """Compute texture energy and entropy using GLCM-like features."""
        # Quantize to fewer levels
        levels = 8
        quantized = (gray * (levels - 1)).astype(np.int32)
        quantized = np.clip(quantized, 0, levels - 1)

        # Simple co-occurrence (horizontal pairs)
        h, w = quantized.shape
        glcm = np.zeros((levels, levels))

        for i in range(h):
            for j in range(w - 1):
                glcm[quantized[i, j], quantized[i, j + 1]] += 1

        # Normalize
        glcm = glcm / (glcm.sum() + 1e-8)

        # Energy (Angular Second Moment)
        energy = float((glcm**2).sum())

        # Entropy
        glcm_nonzero = glcm[glcm > 0]
        entropy = float(-np.sum(glcm_nonzero * np.log2(glcm_nonzero)))

        return energy, entropy

    def _build_appearance_description(self, features: AppearanceFeatures) -> str:
        """Build natural language appearance description."""
        parts = []

        # Brightness
        if features.brightness_mean < 0.3:
            parts.append("The image is relatively dark.")
        elif features.brightness_mean > 0.7:
            parts.append("The image is brightly lit.")
        else:
            parts.append("The image has moderate brightness.")

        # Contrast
        if features.contrast > 0.6:
            parts.append("High contrast with strong shadows and highlights.")
        elif features.contrast < 0.2:
            parts.append("Low contrast, possibly hazy or foggy.")

        # Dominant colors
        if features.dominant_colors:
            top_color, coverage = features.dominant_colors[0]
            if coverage > 0.3:
                r, g, b = top_color
                color_name = self._rgb_to_name(r, g, b)
                parts.append(f"Dominant color: {color_name} ({coverage:.0%} coverage).")

        # Saturation
        if features.saturation_mean > 0.5:
            parts.append("Colors are vivid and saturated.")
        elif features.saturation_mean < 0.2:
            parts.append("Colors are muted or desaturated.")

        # Texture
        if features.texture_entropy > 3.0:
            parts.append("Complex, varied texture throughout.")
        elif features.texture_entropy < 1.5:
            parts.append("Uniform, smooth surface texture.")

        return " ".join(parts)

    def _rgb_to_name(self, r: int, g: int, b: int) -> str:
        """Convert RGB to approximate color name."""
        # Simple color naming
        if max(r, g, b) < 50:
            return "black"
        if min(r, g, b) > 200:
            return "white"
        if r > g + 50 and r > b + 50:
            return "red"
        if g > r + 50 and g > b + 50:
            return "green"
        if b > r + 50 and b > g + 50:
            return "blue"
        if r > 200 and g > 200 and b < 100:
            return "yellow"
        if r > 200 and g < 150 and b > 200:
            return "magenta"
        if r < 100 and g > 200 and b > 200:
            return "cyan"
        if abs(r - g) < 30 and abs(g - b) < 30:
            if r > 150:
                return "light gray"
            return "gray"
        return "mixed"

    def format_context_prompt(self, context: ContextInfo) -> str:
        """Format appearance context as prompt addition."""
        return f"\n[Appearance Analysis: {context.description}]"


class EnhancedCombinedContextProvider:
    """
    Enhanced combined context provider with all context types.

    Integrates:
    - Position context (spatial awareness)
    - Temporal context (motion analysis)
    - Semantic context (scene understanding)
    - Frequency context (spectral analysis)
    - Appearance context (color/texture)
    """

    def __init__(
        self,
        enable_position: bool = True,
        enable_temporal: bool = True,
        enable_semantic: bool = True,
        enable_frequency: bool = True,
        enable_appearance: bool = True,
        random_state: int | None = None,
    ):
        """
        Initialize enhanced combined provider.

        Args:
            enable_position: Enable position context
            enable_temporal: Enable temporal context
            enable_semantic: Enable semantic context
            enable_frequency: Enable frequency context
            enable_appearance: Enable appearance context
            random_state: Seed for reproducible results
        """
        from .context_providers import PositionContextProvider, TemporalContextProvider

        self.providers: dict[str, BaseContextProvider] = {}

        if enable_position:
            self.providers["position"] = PositionContextProvider()

        if enable_temporal:
            self.providers["temporal"] = TemporalContextProvider()

        if enable_semantic:
            self.providers["semantic"] = SemanticContextProvider(random_state=random_state)

        if enable_frequency:
            self.providers["frequency"] = FrequencyContextProvider()

        if enable_appearance:
            self.providers["appearance"] = AppearanceContextProvider()

    def extract_all_context(
        self,
        frames: np.ndarray | torch.Tensor,
        context_types: list[str] | None = None,
    ) -> dict[str, ContextInfo]:
        """
        Extract specified context types.

        Args:
            frames: Input frames
            context_types: List of context types to extract (None = all)

        Returns:
            Dict mapping context type to context info
        """
        contexts = {}

        types_to_extract = context_types or list(self.providers.keys())

        for ctx_type in types_to_extract:
            if ctx_type in self.providers:
                try:
                    contexts[ctx_type] = self.providers[ctx_type].extract_context(frames)
                except Exception as e:
                    logger.warning(f"Failed to extract {ctx_type} context: {e}")

        return contexts

    def format_combined_prompt(
        self,
        contexts: dict[str, ContextInfo],
        priority_order: list[str] | None = None,
    ) -> str:
        """
        Format all contexts as prompt addition.

        Args:
            contexts: Dict of context info
            priority_order: Order of context types (None = default)

        Returns:
            Combined context string for prompt
        """
        if priority_order is None:
            priority_order = ["semantic", "position", "temporal", "appearance", "frequency"]

        parts = []

        for ctx_type in priority_order:
            if ctx_type in contexts and ctx_type in self.providers:
                parts.append(self.providers[ctx_type].format_context_prompt(contexts[ctx_type]))

        return "\n".join(parts)

    def get_context_summary(self, contexts: dict[str, ContextInfo]) -> dict[str, Any]:
        """
        Get a structured summary of all contexts.

        Args:
            contexts: Dict of context info

        Returns:
            Summary dictionary with key metrics
        """
        summary: dict[str, Any] = {}

        for ctx_type, ctx_info in contexts.items():
            summary[ctx_type] = {
                "description": ctx_info.description,
                "metadata": ctx_info.metadata,
            }

        return summary

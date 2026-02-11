"""
Fingerprint Recognition Module for Mercury Agent Biometric System.

Implements minutiae-based fingerprint matching with ridge flow analysis
and liveness detection via sweat pore analysis.

References:
- Maltoni et al. (2009): Handbook of Fingerprint Recognition
- Jain et al. (1997): On-line Fingerprint Verification
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class MinutiaeType(Enum):
    """Type of fingerprint minutiae."""

    RIDGE_ENDING = auto()
    BIFURCATION = auto()
    SHORT_RIDGE = auto()
    ISLAND = auto()
    SPUR = auto()
    CROSSOVER = auto()


class SingularityType(Enum):
    """Type of fingerprint singularity (core/delta)."""

    CORE_LOOP = auto()
    CORE_WHORL = auto()
    DELTA = auto()


@dataclass
class Minutia:
    """A single fingerprint minutia point."""

    x: float
    y: float
    orientation: float
    type: MinutiaeType
    quality: float = 1.0
    ridge_count: int = 0

    def distance_to(self, other: Minutia) -> float:
        """Euclidean distance to another minutia."""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def angle_difference(self, other: Minutia) -> float:
        """Absolute angle difference to another minutia."""
        diff = abs(self.orientation - other.orientation)
        return min(diff, 2 * math.pi - diff)


@dataclass
class Singularity:
    """A fingerprint singularity point (core or delta)."""

    x: float
    y: float
    type: SingularityType
    orientation: float = 0.0


@dataclass
class FingerprintFeatures:
    """Extracted fingerprint features."""

    minutiae: list[Minutia]
    singularities: list[Singularity]
    orientation_field: np.ndarray
    ridge_frequency: np.ndarray
    quality_map: np.ndarray
    enhanced_image: np.ndarray | None = None
    overall_quality: float = 0.0


@dataclass
class FingerprintMatchResult:
    """Result of fingerprint matching."""

    match_score: float
    matched_minutiae: int
    total_probe_minutiae: int
    total_gallery_minutiae: int
    is_match: bool
    confidence: float
    transformation: tuple[float, float, float] | None = None


@dataclass
class FingerprintLivenessResult:
    """Result of fingerprint liveness detection."""

    is_live: bool
    confidence: float
    pore_score: float
    perspiration_score: float
    elasticity_score: float
    details: dict[str, Any] = field(default_factory=dict)


class OrientationFieldEstimator:
    """
    Estimate local ridge orientation field.

    Uses gradient-based method with block averaging.
    """

    def __init__(self, block_size: int = 16) -> None:
        """Initialize the estimator."""
        self._block_size = block_size

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate orientation field from fingerprint image.

        Args:
            image: Grayscale fingerprint image

        Returns:
            Orientation field (radians) at block resolution
        """
        if image.ndim == 3:
            image = np.mean(image, axis=2)

        image = image.astype(np.float64)
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)

        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8.0
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 8.0

        gx = self._convolve2d(image, sobel_x)
        gy = self._convolve2d(image, sobel_y)

        gxx = gx * gx
        gyy = gy * gy
        gxy = gx * gy

        h, w = image.shape
        block_h = h // self._block_size
        block_w = w // self._block_size
        orientation = np.zeros((block_h, block_w))

        for i in range(block_h):
            for j in range(block_w):
                y1 = i * self._block_size
                y2 = (i + 1) * self._block_size
                x1 = j * self._block_size
                x2 = (j + 1) * self._block_size

                vx = 2 * np.sum(gxy[y1:y2, x1:x2])
                vy = np.sum(gxx[y1:y2, x1:x2] - gyy[y1:y2, x1:x2])

                orientation[i, j] = 0.5 * np.arctan2(vx, vy)

        return orientation

    def _convolve2d(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """2D convolution."""
        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2

        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
        result = np.zeros_like(image)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                result[i, j] = np.sum(padded[i : i + kh, j : j + kw] * kernel)

        return result


class RidgeFrequencyEstimator:
    """
    Estimate local ridge frequency.

    Uses projection-based method along ridge direction.
    """

    def __init__(
        self,
        block_size: int = 32,
        window_size: int = 64,
    ) -> None:
        """Initialize the estimator."""
        self._block_size = block_size
        self._window_size = window_size

    def estimate(
        self,
        image: np.ndarray,
        orientation: np.ndarray,
    ) -> np.ndarray:
        """
        Estimate ridge frequency from image and orientation.

        Args:
            image: Grayscale fingerprint image
            orientation: Orientation field

        Returns:
            Ridge frequency map
        """
        if image.ndim == 3:
            image = np.mean(image, axis=2)

        h, w = image.shape
        oh, ow = orientation.shape
        frequency = np.zeros_like(orientation)

        for i in range(oh):
            for j in range(ow):
                cy = min(int((i + 0.5) * self._block_size), h - 1)
                cx = min(int((j + 0.5) * self._block_size), w - 1)

                y1 = max(0, cy - self._window_size // 2)
                y2 = min(h, cy + self._window_size // 2)
                x1 = max(0, cx - self._window_size // 2)
                x2 = min(w, cx + self._window_size // 2)

                block = image[y1:y2, x1:x2]
                angle = orientation[i, j]

                projection = self._project_along_direction(block, angle)
                frequency[i, j] = self._estimate_frequency_from_projection(projection)

        return frequency

    def _project_along_direction(
        self,
        block: np.ndarray,
        angle: float,
    ) -> np.ndarray:
        """Project block along perpendicular to ridge direction."""
        h, w = block.shape
        projection = np.zeros(w)

        cos_a = np.cos(angle + np.pi / 2)
        sin_a = np.sin(angle + np.pi / 2)

        for j in range(w):
            values = []
            for i in range(h):
                x = int(w / 2 + (j - w / 2) * cos_a - (i - h / 2) * sin_a)
                y = int(h / 2 + (j - w / 2) * sin_a + (i - h / 2) * cos_a)

                if 0 <= x < w and 0 <= y < h:
                    values.append(block[y, x])

            if values:
                projection[j] = np.mean(values)

        return projection

    def _estimate_frequency_from_projection(self, projection: np.ndarray) -> float:
        """Estimate frequency from projection profile."""
        projection = projection - np.mean(projection)

        if np.std(projection) < 0.01:
            return 0.1

        crossings = []
        for i in range(1, len(projection)):
            if projection[i - 1] * projection[i] < 0:
                crossings.append(i)

        if len(crossings) < 2:
            return 0.1

        periods = []
        for i in range(0, len(crossings) - 1, 2):
            if i + 2 < len(crossings):
                periods.append(crossings[i + 2] - crossings[i])

        if not periods:
            return 0.1

        avg_period = np.mean(periods)
        frequency = 1.0 / max(float(avg_period), 1.0)  # type: ignore[operator, unused-ignore]

        return float(min(0.5, max(0.05, frequency)))


class GaborEnhancer:
    """
    Enhance fingerprint using Gabor filters.

    Applies contextual Gabor filters based on local orientation and frequency.
    """

    def __init__(self, kernel_size: int = 25) -> None:
        """Initialize the enhancer."""
        self._kernel_size = kernel_size
        self._filter_cache: dict[tuple[float, float], np.ndarray] = {}

    def enhance(
        self,
        image: np.ndarray,
        orientation: np.ndarray,
        frequency: np.ndarray,
        quality_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Enhance fingerprint image using Gabor filters.

        Args:
            image: Input fingerprint image
            orientation: Local orientation field
            frequency: Local frequency field
            quality_mask: Optional mask for valid regions

        Returns:
            Enhanced fingerprint image
        """
        if image.ndim == 3:
            image = np.mean(image, axis=2)

        image = image.astype(np.float64)
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)

        h, w = image.shape
        enhanced = np.zeros_like(image)
        oh, ow = orientation.shape
        block_h = h // oh
        block_w = w // ow

        for i in range(oh):
            for j in range(ow):
                y1 = i * block_h
                y2 = min((i + 1) * block_h, h)
                x1 = j * block_w
                x2 = min((j + 1) * block_w, w)

                angle = orientation[i, j]
                freq = frequency[i, j]

                gabor = self._get_gabor_filter(angle, freq)
                block = image[
                    max(0, y1 - self._kernel_size // 2) : min(h, y2 + self._kernel_size // 2),
                    max(0, x1 - self._kernel_size // 2) : min(w, x2 + self._kernel_size // 2),
                ]

                filtered = self._apply_filter(block, gabor)

                cy, cx = filtered.shape[0] // 2, filtered.shape[1] // 2
                bh, bw = y2 - y1, x2 - x1

                start_y = max(0, cy - bh // 2)
                start_x = max(0, cx - bw // 2)
                end_y = min(filtered.shape[0], start_y + bh)
                end_x = min(filtered.shape[1], start_x + bw)

                extracted = filtered[start_y:end_y, start_x:end_x]

                target_h = min(bh, extracted.shape[0])
                target_w = min(bw, extracted.shape[1])
                enhanced[y1 : y1 + target_h, x1 : x1 + target_w] = extracted[:target_h, :target_w]

        enhanced = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min() + 1e-8)
        return enhanced

    def _get_gabor_filter(self, angle: float, frequency: float) -> np.ndarray:
        """Get or create Gabor filter for given parameters."""
        key = (round(angle, 2), round(frequency, 3))
        if key in self._filter_cache:
            return self._filter_cache[key]

        size = self._kernel_size
        half = size // 2

        x = np.arange(-half, half + 1)
        y = np.arange(-half, half + 1)
        xx, yy = np.meshgrid(x, y)

        x_theta = xx * np.cos(angle) + yy * np.sin(angle)
        y_theta = -xx * np.sin(angle) + yy * np.cos(angle)

        sigma_x = 4.0
        sigma_y = 4.0

        gaussian = np.exp(-0.5 * (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2))
        sinusoid = np.cos(2 * np.pi * frequency * x_theta)

        gabor = gaussian * sinusoid
        gabor = gabor - np.mean(gabor)
        gabor = gabor / (np.sum(np.abs(gabor)) + 1e-8)

        self._filter_cache[key] = gabor
        return gabor

    def _apply_filter(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Apply filter using FFT convolution."""
        pad_h = kernel.shape[0] // 2
        pad_w = kernel.shape[1] // 2
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")

        result = np.zeros_like(padded)
        kh, kw = kernel.shape

        for i in range(pad_h, padded.shape[0] - pad_h):
            for j in range(pad_w, padded.shape[1] - pad_w):
                result[i, j] = np.sum(
                    padded[i - pad_h : i + pad_h + 1, j - pad_w : j + pad_w + 1] * kernel
                )

        return result[pad_h:-pad_h, pad_w:-pad_w]


class MinutiaeExtractor:
    """
    Extract minutiae from enhanced fingerprint image.

    Uses crossing number method on thinned ridges.
    """

    def __init__(
        self,
        quality_threshold: float = 0.3,
        border_margin: int = 10,
    ) -> None:
        """Initialize the extractor."""
        self._quality_threshold = quality_threshold
        self._border_margin = border_margin

    def extract(
        self,
        enhanced_image: np.ndarray,
        orientation: np.ndarray,
        quality_map: np.ndarray | None = None,
    ) -> list[Minutia]:
        """
        Extract minutiae from enhanced fingerprint.

        Args:
            enhanced_image: Enhanced fingerprint image
            orientation: Local orientation field
            quality_map: Optional quality map

        Returns:
            List of extracted minutiae
        """
        binary = self._binarize(enhanced_image)
        skeleton = self._thin(binary)
        minutiae = self._find_minutiae(skeleton, orientation)
        minutiae = self._filter_minutiae(minutiae, enhanced_image.shape, quality_map)

        return minutiae

    def _binarize(self, image: np.ndarray) -> np.ndarray:
        """Binarize the image using adaptive thresholding."""
        threshold = np.mean(image)
        binary = (image < threshold).astype(np.uint8)
        return binary

    def _thin(self, binary: np.ndarray) -> np.ndarray:
        """Morphological thinning to get ridge skeleton."""
        skeleton = binary.copy()
        changed = True

        while changed:
            changed = False

            for step in range(2):
                marker = np.zeros_like(skeleton)

                for i in range(1, skeleton.shape[0] - 1):
                    for j in range(1, skeleton.shape[1] - 1):
                        if skeleton[i, j] == 0:
                            continue

                        neighbors = [
                            skeleton[i - 1, j],
                            skeleton[i - 1, j + 1],
                            skeleton[i, j + 1],
                            skeleton[i + 1, j + 1],
                            skeleton[i + 1, j],
                            skeleton[i + 1, j - 1],
                            skeleton[i, j - 1],
                            skeleton[i - 1, j - 1],
                        ]

                        b = sum(neighbors)
                        if b < 2 or b > 6:
                            continue

                        a = 0
                        for k in range(8):
                            if neighbors[k] == 0 and neighbors[(k + 1) % 8] == 1:
                                a += 1

                        if a != 1:
                            continue

                        if step == 0:
                            if neighbors[0] * neighbors[2] * neighbors[4] != 0:
                                continue
                            if neighbors[2] * neighbors[4] * neighbors[6] != 0:
                                continue
                        else:
                            if neighbors[0] * neighbors[2] * neighbors[6] != 0:
                                continue
                            if neighbors[0] * neighbors[4] * neighbors[6] != 0:
                                continue

                        marker[i, j] = 1
                        changed = True

                skeleton = skeleton - marker

        return skeleton

    def _find_minutiae(
        self,
        skeleton: np.ndarray,
        orientation: np.ndarray,
    ) -> list[Minutia]:
        """Find minutiae using crossing number."""
        minutiae = []
        h, w = skeleton.shape
        oh, ow = orientation.shape

        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if skeleton[i, j] == 0:
                    continue

                neighbors = [
                    skeleton[i - 1, j],
                    skeleton[i - 1, j + 1],
                    skeleton[i, j + 1],
                    skeleton[i + 1, j + 1],
                    skeleton[i + 1, j],
                    skeleton[i + 1, j - 1],
                    skeleton[i, j - 1],
                    skeleton[i - 1, j - 1],
                ]

                cn = 0
                for k in range(8):
                    cn += abs(neighbors[k] - neighbors[(k + 1) % 8])
                cn = cn // 2

                if cn == 1:
                    minutiae_type = MinutiaeType.RIDGE_ENDING
                elif cn == 3:
                    minutiae_type = MinutiaeType.BIFURCATION
                else:
                    continue

                oi = min(int(i / (h / oh)), oh - 1)
                oj = min(int(j / (w / ow)), ow - 1)
                angle = orientation[oi, oj]

                minutiae.append(
                    Minutia(
                        x=float(j),
                        y=float(i),
                        orientation=angle,
                        type=minutiae_type,
                        quality=1.0,
                    )
                )

        return minutiae

    def _filter_minutiae(
        self,
        minutiae: list[Minutia],
        image_shape: tuple[int, int],
        quality_map: np.ndarray | None,
    ) -> list[Minutia]:
        """Filter spurious minutiae."""
        h, w = image_shape
        filtered = []

        for m in minutiae:
            if m.x < self._border_margin or m.x >= w - self._border_margin:
                continue
            if m.y < self._border_margin or m.y >= h - self._border_margin:
                continue

            if quality_map is not None:
                qh, qw = quality_map.shape
                qi = int(m.y / (h / qh))
                qj = int(m.x / (w / qw))
                qi = min(qi, qh - 1)
                qj = min(qj, qw - 1)

                if quality_map[qi, qj] < self._quality_threshold:
                    continue

            filtered.append(m)

        return self._remove_close_minutiae(filtered, min_distance=10.0)

    def _remove_close_minutiae(
        self,
        minutiae: list[Minutia],
        min_distance: float,
    ) -> list[Minutia]:
        """Remove minutiae that are too close together."""
        if len(minutiae) <= 1:
            return minutiae

        filtered = []
        used = [False] * len(minutiae)

        for i, m1 in enumerate(minutiae):
            if used[i]:
                continue

            cluster = [m1]
            for j, m2 in enumerate(minutiae[i + 1 :], i + 1):
                if not used[j] and m1.distance_to(m2) < min_distance:
                    cluster.append(m2)
                    used[j] = True

            best = max(cluster, key=lambda m: m.quality)
            filtered.append(best)

        return filtered


class FingerprintMatcher:
    """
    Match fingerprints using minutiae comparison.

    Implements Bozorth3-like algorithm with spatial and angular tolerances.
    """

    def __init__(
        self,
        distance_tolerance: float = 15.0,
        angle_tolerance: float = 0.25,
        min_matched_minutiae: int = 8,
        match_threshold: float = 40.0,
    ) -> None:
        """Initialize the matcher."""
        self._dist_tol = distance_tolerance
        self._angle_tol = angle_tolerance
        self._min_matched = min_matched_minutiae
        self._threshold = match_threshold

    def match(
        self,
        probe: FingerprintFeatures,
        gallery: FingerprintFeatures,
    ) -> FingerprintMatchResult:
        """
        Match probe fingerprint against gallery.

        Args:
            probe: Probe fingerprint features
            gallery: Gallery fingerprint features

        Returns:
            Match result with score and confidence
        """
        probe_minutiae = probe.minutiae
        gallery_minutiae = gallery.minutiae

        if len(probe_minutiae) < self._min_matched or len(gallery_minutiae) < self._min_matched:
            return FingerprintMatchResult(
                match_score=0.0,
                matched_minutiae=0,
                total_probe_minutiae=len(probe_minutiae),
                total_gallery_minutiae=len(gallery_minutiae),
                is_match=False,
                confidence=0.0,
            )

        best_score = 0.0
        best_matched = 0
        best_transform = None

        for pi, pm in enumerate(probe_minutiae[:20]):
            for gi, gm in enumerate(gallery_minutiae[:20]):
                dx = gm.x - pm.x
                dy = gm.y - pm.y
                da = gm.orientation - pm.orientation

                matched = self._count_matched_minutiae(probe_minutiae, gallery_minutiae, dx, dy, da)

                score = self._compute_score(matched, len(probe_minutiae), len(gallery_minutiae))

                if score > best_score:
                    best_score = score
                    best_matched = matched
                    best_transform = (dx, dy, da)

        is_match = best_score >= self._threshold
        confidence = (
            min(1.0, best_score / 100.0) if is_match else best_score / self._threshold * 0.5
        )

        return FingerprintMatchResult(
            match_score=best_score,
            matched_minutiae=best_matched,
            total_probe_minutiae=len(probe_minutiae),
            total_gallery_minutiae=len(gallery_minutiae),
            is_match=is_match,
            confidence=confidence,
            transformation=best_transform,
        )

    def _count_matched_minutiae(
        self,
        probe: list[Minutia],
        gallery: list[Minutia],
        dx: float,
        dy: float,
        da: float,
    ) -> int:
        """Count matched minutiae under given transformation."""
        matched = 0
        gallery_used = [False] * len(gallery)

        for pm in probe:
            px_transformed = pm.x + dx
            py_transformed = pm.y + dy
            pa_transformed = pm.orientation + da

            for gi, gm in enumerate(gallery):
                if gallery_used[gi]:
                    continue

                dist = math.sqrt((px_transformed - gm.x) ** 2 + (py_transformed - gm.y) ** 2)
                if dist > self._dist_tol:
                    continue

                angle_diff = abs(pa_transformed - gm.orientation)
                angle_diff = min(angle_diff, 2 * math.pi - angle_diff)
                if angle_diff > self._angle_tol:
                    continue

                matched += 1
                gallery_used[gi] = True
                break

        return matched

    def _compute_score(
        self,
        matched: int,
        probe_count: int,
        gallery_count: int,
    ) -> float:
        """Compute match score from matched count."""
        if probe_count == 0 or gallery_count == 0:
            return 0.0

        score = (matched**2) / (probe_count * gallery_count) * 100
        return score


class FingerprintLivenessDetector:
    """
    Detect fingerprint presentation attacks.

    Analyzes sweat pores, perspiration patterns, and skin elasticity.
    """

    def __init__(
        self,
        pore_threshold: float = 0.5,
        perspiration_threshold: float = 0.4,
        elasticity_threshold: float = 0.5,
    ) -> None:
        """Initialize the liveness detector."""
        self._pore_threshold = pore_threshold
        self._perspiration_threshold = perspiration_threshold
        self._elasticity_threshold = elasticity_threshold

    def detect(
        self,
        images: list[np.ndarray],
        features: FingerprintFeatures | None = None,
    ) -> FingerprintLivenessResult:
        """
        Detect liveness from fingerprint images.

        Args:
            images: Sequence of fingerprint images
            features: Pre-extracted features (optional)

        Returns:
            Liveness result with confidence scores
        """
        if len(images) < 1:
            return FingerprintLivenessResult(
                is_live=False,
                confidence=0.0,
                pore_score=0.0,
                perspiration_score=0.0,
                elasticity_score=0.0,
                details={"error": "No images provided"},
            )

        pore_score = self._analyze_sweat_pores(images[0])
        perspiration_score = self._analyze_perspiration(images)
        elasticity_score = self._analyze_elasticity(images)

        pore_live = pore_score > self._pore_threshold
        perspiration_live = perspiration_score > self._perspiration_threshold
        elasticity_live = elasticity_score > self._elasticity_threshold

        is_live = pore_live and (perspiration_live or elasticity_live)
        confidence = (pore_score + perspiration_score + elasticity_score) / 3.0

        return FingerprintLivenessResult(
            is_live=is_live,
            confidence=confidence,
            pore_score=pore_score,
            perspiration_score=perspiration_score,
            elasticity_score=elasticity_score,
            details={
                "pore_live": pore_live,
                "perspiration_live": perspiration_live,
                "elasticity_live": elasticity_live,
            },
        )

    def _analyze_sweat_pores(self, image: np.ndarray) -> float:
        """Analyze sweat pore presence and distribution."""
        if image.ndim == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image.astype(np.float64)

        gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)

        laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        edges = self._convolve2d(gray, laplacian)

        threshold = np.percentile(np.abs(edges), 95)
        high_freq = np.abs(edges) > threshold

        pore_density = np.sum(high_freq) / high_freq.size
        expected_density = 0.03

        if pore_density < expected_density / 5:
            return 0.2

        score = min(1.0, pore_density / expected_density)
        return float(score)

    def _analyze_perspiration(self, images: list[np.ndarray]) -> float:
        """Analyze perspiration changes over time."""
        if len(images) < 2:
            return 0.5

        intensities = []
        for img in images:
            if img.ndim == 3:
                gray = np.mean(img, axis=2)
            else:
                gray = img
            intensities.append(np.mean(gray))

        variation = np.std(intensities) / (np.mean(intensities) + 1e-8)

        if variation < 0.001:
            return 0.2

        score = min(1.0, variation / 0.02)
        return float(score)

    def _analyze_elasticity(self, images: list[np.ndarray]) -> float:
        """Analyze skin elasticity from pressure variations."""
        if len(images) < 2:
            return 0.5

        contrasts = []
        for img in images:
            if img.ndim == 3:
                gray = np.mean(img, axis=2)
            else:
                gray = img
            contrasts.append(np.std(gray))

        variation = np.std(contrasts) / (np.mean(contrasts) + 1e-8)

        if variation < 0.001:
            return 0.2

        score = min(1.0, variation / 0.05)
        return float(score)

    def _convolve2d(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """2D convolution."""
        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2

        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
        result = np.zeros_like(image)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                result[i, j] = np.sum(padded[i : i + kh, j : j + kw] * kernel)

        return result


class FingerprintRecognizer:
    """
    Complete fingerprint recognition system.

    Integrates enhancement, minutiae extraction, matching, and liveness detection.
    """

    def __init__(
        self,
        match_threshold: float = 40.0,
        liveness_required: bool = True,
    ) -> None:
        """Initialize the fingerprint recognizer."""
        self._orientation_estimator = OrientationFieldEstimator()
        self._frequency_estimator = RidgeFrequencyEstimator()
        self._enhancer = GaborEnhancer()
        self._minutiae_extractor = MinutiaeExtractor()
        self._matcher = FingerprintMatcher(match_threshold=match_threshold)
        self._liveness_detector = FingerprintLivenessDetector()
        self._liveness_required = liveness_required

    def extract_features(self, image: np.ndarray) -> FingerprintFeatures:
        """
        Extract fingerprint features from image.

        Args:
            image: Fingerprint image

        Returns:
            FingerprintFeatures containing minutiae and metadata
        """
        if image.ndim == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image.astype(np.float64)

        gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)

        orientation = self._orientation_estimator.estimate(gray)
        frequency = self._frequency_estimator.estimate(gray, orientation)
        enhanced = self._enhancer.enhance(gray, orientation, frequency)

        quality_map = self._compute_quality_map(gray, orientation)
        minutiae = self._minutiae_extractor.extract(enhanced, orientation, quality_map)

        singularities = self._detect_singularities(orientation)
        overall_quality = self._compute_overall_quality(quality_map, len(minutiae))

        return FingerprintFeatures(
            minutiae=minutiae,
            singularities=singularities,
            orientation_field=orientation,
            ridge_frequency=frequency,
            quality_map=quality_map,
            enhanced_image=enhanced,
            overall_quality=overall_quality,
        )

    def verify(
        self,
        probe_image: np.ndarray,
        enrolled_features: FingerprintFeatures,
        liveness_images: list[np.ndarray] | None = None,
    ) -> tuple[FingerprintMatchResult, FingerprintLivenessResult | None]:
        """
        Verify a fingerprint against enrolled features.

        Args:
            probe_image: Probe fingerprint image
            enrolled_features: Enrolled fingerprint features
            liveness_images: Additional images for liveness detection

        Returns:
            Tuple of (match_result, liveness_result)
        """
        probe_features = self.extract_features(probe_image)
        match_result = self._matcher.match(probe_features, enrolled_features)

        liveness_result = None
        if self._liveness_required:
            images = [probe_image]
            if liveness_images:
                images.extend(liveness_images)

            liveness_result = self._liveness_detector.detect(images, probe_features)

            if not liveness_result.is_live:
                match_result = FingerprintMatchResult(
                    match_score=0.0,
                    matched_minutiae=0,
                    total_probe_minutiae=match_result.total_probe_minutiae,
                    total_gallery_minutiae=match_result.total_gallery_minutiae,
                    is_match=False,
                    confidence=0.0,
                )

        return match_result, liveness_result

    def _compute_quality_map(
        self,
        image: np.ndarray,
        orientation: np.ndarray,
    ) -> np.ndarray:
        """Compute local quality map."""
        oh, ow = orientation.shape
        quality = np.zeros((oh, ow))

        h, w = image.shape
        block_h = h // oh
        block_w = w // ow

        for i in range(oh):
            for j in range(ow):
                y1 = i * block_h
                y2 = (i + 1) * block_h
                x1 = j * block_w
                x2 = (j + 1) * block_w

                block = image[y1:y2, x1:x2]
                contrast = np.std(block)
                coherence = self._compute_coherence(orientation, i, j)

                quality[i, j] = 0.5 * min(1.0, contrast * 5) + 0.5 * coherence

        return quality

    def _compute_coherence(
        self,
        orientation: np.ndarray,
        i: int,
        j: int,
    ) -> float:
        """Compute orientation coherence around a block."""
        oh, ow = orientation.shape
        angles = []

        for di in range(-1, 2):
            for dj in range(-1, 2):
                ni, nj = i + di, j + dj
                if 0 <= ni < oh and 0 <= nj < ow:
                    angles.append(orientation[ni, nj])

        if len(angles) < 2:
            return 1.0

        cos_sum = sum(np.cos(2 * a) for a in angles)
        sin_sum = sum(np.sin(2 * a) for a in angles)

        coherence = np.sqrt(cos_sum**2 + sin_sum**2) / len(angles)
        return float(coherence)

    def _detect_singularities(self, orientation: np.ndarray) -> list[Singularity]:
        """Detect singular points (cores and deltas)."""
        singularities = []
        oh, ow = orientation.shape

        for i in range(1, oh - 1):
            for j in range(1, ow - 1):
                angles = [
                    orientation[i - 1, j - 1],
                    orientation[i - 1, j],
                    orientation[i - 1, j + 1],
                    orientation[i, j + 1],
                    orientation[i + 1, j + 1],
                    orientation[i + 1, j],
                    orientation[i + 1, j - 1],
                    orientation[i, j - 1],
                ]

                poincare = 0.0
                for k in range(8):
                    diff = angles[(k + 1) % 8] - angles[k]
                    while diff > np.pi / 2:
                        diff -= np.pi
                    while diff < -np.pi / 2:
                        diff += np.pi
                    poincare += diff

                poincare /= np.pi

                if abs(poincare - 1) < 0.3:
                    singularities.append(
                        Singularity(
                            x=float(j),
                            y=float(i),
                            type=SingularityType.CORE_LOOP,
                            orientation=orientation[i, j],
                        )
                    )
                elif abs(poincare + 1) < 0.3:
                    singularities.append(
                        Singularity(
                            x=float(j),
                            y=float(i),
                            type=SingularityType.DELTA,
                            orientation=orientation[i, j],
                        )
                    )

        return singularities

    def _compute_overall_quality(
        self,
        quality_map: np.ndarray,
        minutiae_count: int,
    ) -> float:
        """Compute overall fingerprint quality."""
        avg_quality = np.mean(quality_map)
        minutiae_factor = min(1.0, minutiae_count / 30.0)

        return float(0.6 * avg_quality + 0.4 * minutiae_factor)

"""
Iris Recognition Module for Mercury Agent Biometric System.

Implements the Daugman IrisCode algorithm for iris recognition with
liveness detection based on pupil dynamics and specular reflection analysis.

References:
- Daugman (2004): How Iris Recognition Works
- Daugman (1993): High confidence visual recognition by a test of statistical independence
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class IrisFeatures:
    """Extracted iris features."""

    iris_code: np.ndarray
    mask: np.ndarray
    pupil_center: tuple[float, float]
    pupil_radius: float
    iris_center: tuple[float, float]
    iris_radius: float
    quality_score: float
    normalized_iris: np.ndarray | None = None


@dataclass
class IrisMatchResult:
    """Result of iris matching."""

    hamming_distance: float
    match_score: float
    is_match: bool
    bits_compared: int
    confidence: float


@dataclass
class LivenessResult:
    """Result of liveness detection."""

    is_live: bool
    confidence: float
    pupil_response: float
    specular_reflection_score: float
    texture_authenticity: float
    details: dict[str, Any] = field(default_factory=dict)


class GaborFilter:
    """
    2D Gabor filter bank for iris texture analysis.

    Gabor filters are bandpass filters that capture texture information
    at specific orientations and frequencies.
    """

    def __init__(
        self,
        kernel_size: int = 31,
        num_orientations: int = 8,
        num_scales: int = 4,
        sigma: float = 3.0,
        wavelength_base: float = 4.0,
    ) -> None:
        """Initialize the Gabor filter bank."""
        self._kernel_size = kernel_size
        self._num_orientations = num_orientations
        self._num_scales = num_scales
        self._sigma = sigma
        self._wavelength_base = wavelength_base
        self._filters = self._create_filter_bank()

    def _create_filter_bank(self) -> list[np.ndarray]:
        """Create the Gabor filter bank."""
        filters = []
        half_size = self._kernel_size // 2

        for scale in range(self._num_scales):
            wavelength = self._wavelength_base * (2**scale)
            sigma = self._sigma * (2 ** (scale / 2))

            for orientation in range(self._num_orientations):
                theta = orientation * np.pi / self._num_orientations

                x = np.arange(-half_size, half_size + 1)
                y = np.arange(-half_size, half_size + 1)
                xx, yy = np.meshgrid(x, y)

                x_theta = xx * np.cos(theta) + yy * np.sin(theta)
                y_theta = -xx * np.sin(theta) + yy * np.cos(theta)

                gaussian = np.exp(-(x_theta**2 + y_theta**2) / (2 * sigma**2))
                sinusoid = np.exp(2j * np.pi * x_theta / wavelength)

                gabor = gaussian * sinusoid
                gabor = gabor / np.sqrt(np.sum(np.abs(gabor) ** 2))
                filters.append(gabor)

        return filters

    def apply(self, image: np.ndarray) -> list[np.ndarray]:
        """Apply all filters to an image."""
        responses = []

        for gabor_filter in self._filters:
            response = self._convolve2d(image, gabor_filter)
            responses.append(response)

        return responses

    def _convolve2d(
        self,
        image: np.ndarray,
        kernel: np.ndarray,
    ) -> np.ndarray:
        """2D convolution using FFT."""
        image_fft = np.fft.fft2(image, s=image.shape)
        kernel_padded = np.zeros_like(image, dtype=complex)

        kh, kw = kernel.shape
        kernel_padded[:kh, :kw] = kernel
        kernel_padded = np.roll(kernel_padded, -kh // 2, axis=0)
        kernel_padded = np.roll(kernel_padded, -kw // 2, axis=1)

        kernel_fft = np.fft.fft2(kernel_padded)
        result = np.fft.ifft2(image_fft * kernel_fft)

        return result


class IrisSegmenter:
    """
    Iris segmentation using integro-differential operator.

    Localizes the pupil and iris boundaries in an eye image.
    """

    def __init__(
        self,
        pupil_radius_range: tuple[int, int] = (20, 80),
        iris_radius_range: tuple[int, int] = (60, 150),
        search_step: int = 2,
    ) -> None:
        """Initialize the segmenter."""
        self._pupil_range = pupil_radius_range
        self._iris_range = iris_radius_range
        self._search_step = search_step

    def segment(
        self,
        image: np.ndarray,
    ) -> tuple[tuple[float, float], float, tuple[float, float], float]:
        """
        Segment iris from eye image.

        Returns:
            Tuple of (pupil_center, pupil_radius, iris_center, iris_radius)
        """
        if image.ndim == 3:
            image = np.mean(image, axis=2)

        image = image.astype(np.float64)
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)

        pupil_center, pupil_radius = self._find_circle(
            image,
            self._pupil_range,
            dark_circle=True,
        )

        search_center = pupil_center
        iris_center, iris_radius = self._find_circle(
            image,
            self._iris_range,
            dark_circle=False,
            center_hint=search_center,
        )

        return pupil_center, pupil_radius, iris_center, iris_radius

    def _find_circle(
        self,
        image: np.ndarray,
        radius_range: tuple[int, int],
        dark_circle: bool = True,
        center_hint: tuple[float, float] | None = None,
    ) -> tuple[tuple[float, float], float]:
        """
        Find a circle using the integro-differential operator.

        The operator finds the maximum of the circular integral gradient.
        """
        h, w = image.shape
        best_score = -np.inf
        best_center = (h / 2, w / 2)
        best_radius = (radius_range[0] + radius_range[1]) / 2

        if center_hint is not None:
            cy_range = range(
                max(0, int(center_hint[0] - 30)),
                min(h, int(center_hint[0] + 30)),
                self._search_step,
            )
            cx_range = range(
                max(0, int(center_hint[1] - 30)),
                min(w, int(center_hint[1] + 30)),
                self._search_step,
            )
        else:
            margin = radius_range[1]
            cy_range = range(margin, h - margin, self._search_step)
            cx_range = range(margin, w - margin, self._search_step)

        for cy in cy_range:
            for cx in cx_range:
                for r in range(radius_range[0], radius_range[1], self._search_step):
                    score = self._circle_score(image, (cy, cx), r, dark_circle)
                    if score > best_score:
                        best_score = score
                        best_center = (float(cy), float(cx))
                        best_radius = float(r)

        return best_center, best_radius

    def _circle_score(
        self,
        image: np.ndarray,
        center: tuple[int, int],
        radius: int,
        dark_circle: bool,
    ) -> float:
        """Compute score for a candidate circle."""
        cy, cx = center
        n_points = max(16, int(2 * np.pi * radius / 4))
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

        inner_values = []
        outer_values = []

        for angle in angles:
            inner_r = max(1, radius - 2)
            outer_r = radius + 2

            iy = int(cy + inner_r * np.sin(angle))
            ix = int(cx + inner_r * np.cos(angle))
            oy = int(cy + outer_r * np.sin(angle))
            ox = int(cx + outer_r * np.cos(angle))

            h, w = image.shape
            if 0 <= iy < h and 0 <= ix < w:
                inner_values.append(image[iy, ix])
            if 0 <= oy < h and 0 <= ox < w:
                outer_values.append(image[oy, ox])

        if not inner_values or not outer_values:
            return float(-np.inf)

        inner_mean = np.mean(inner_values)
        outer_mean = np.mean(outer_values)

        if dark_circle:
            return float(outer_mean - inner_mean)
        else:
            return float(np.abs(outer_mean - inner_mean))


class IrisNormalizer:
    """
    Rubber sheet normalization (Daugman's method).

    Maps the iris region to a fixed-size rectangular representation.
    """

    def __init__(
        self,
        angular_resolution: int = 256,
        radial_resolution: int = 64,
    ) -> None:
        """Initialize the normalizer."""
        self._angular_res = angular_resolution
        self._radial_res = radial_resolution

    def normalize(
        self,
        image: np.ndarray,
        pupil_center: tuple[float, float],
        pupil_radius: float,
        iris_center: tuple[float, float],
        iris_radius: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Normalize iris region to rectangular coordinates.

        Returns:
            Tuple of (normalized_iris, mask)
        """
        if image.ndim == 3:
            image = np.mean(image, axis=2)

        normalized = np.zeros((self._radial_res, self._angular_res))
        mask = np.ones((self._radial_res, self._angular_res), dtype=bool)

        h, w = image.shape
        angles = np.linspace(0, 2 * np.pi, self._angular_res, endpoint=False)
        radii = np.linspace(0, 1, self._radial_res)

        for i, r_norm in enumerate(radii):
            for j, theta in enumerate(angles):
                x_pupil = pupil_center[1] + pupil_radius * np.cos(theta)
                y_pupil = pupil_center[0] + pupil_radius * np.sin(theta)

                x_iris = iris_center[1] + iris_radius * np.cos(theta)
                y_iris = iris_center[0] + iris_radius * np.sin(theta)

                x = int(x_pupil + r_norm * (x_iris - x_pupil))
                y = int(y_pupil + r_norm * (y_iris - y_pupil))

                if 0 <= x < w and 0 <= y < h:
                    normalized[i, j] = image[y, x]
                else:
                    mask[i, j] = False

        return normalized, mask


class IrisEncoder:
    """
    Encode normalized iris to binary IrisCode.

    Uses Gabor wavelets to extract phase information.
    """

    def __init__(
        self,
        num_filters: int = 8,
        kernel_size: int = 31,
    ) -> None:
        """Initialize the encoder."""
        self._gabor = GaborFilter(
            kernel_size=kernel_size,
            num_orientations=num_filters,
            num_scales=1,
        )
        self._num_filters = num_filters

    def encode(
        self,
        normalized_iris: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Encode normalized iris to binary code.

        Returns:
            Tuple of (iris_code, code_mask)
        """
        responses = self._gabor.apply(normalized_iris)
        h, w = normalized_iris.shape
        code_bits = []
        code_mask = []

        for response in responses:
            real_bits = (np.real(response) >= 0).astype(np.uint8)
            imag_bits = (np.imag(response) >= 0).astype(np.uint8)

            code_bits.append(real_bits)
            code_bits.append(imag_bits)
            code_mask.append(mask)
            code_mask.append(mask)

        iris_code = np.stack(code_bits, axis=0)
        code_mask = np.stack(code_mask, axis=0)  # type: ignore[assignment]

        return iris_code, code_mask  # type: ignore[return-value]


class IrisMatcher:
    """
    Match iris codes using Hamming distance.

    Includes rotation compensation for different eye orientations.
    """

    def __init__(
        self,
        match_threshold: float = 0.32,
        rotation_shifts: int = 8,
    ) -> None:
        """Initialize the matcher."""
        self._threshold = match_threshold
        self._rotation_shifts = rotation_shifts

    def match(
        self,
        code1: np.ndarray,
        mask1: np.ndarray,
        code2: np.ndarray,
        mask2: np.ndarray,
    ) -> IrisMatchResult:
        """
        Match two iris codes.

        Returns match result with Hamming distance and confidence.
        """
        best_distance = 1.0
        best_bits = 0

        for shift in range(-self._rotation_shifts, self._rotation_shifts + 1):
            code2_shifted = np.roll(code2, shift, axis=2)
            mask2_shifted = np.roll(mask2, shift, axis=2)

            combined_mask = mask1 & mask2_shifted
            bits_compared = np.sum(combined_mask)

            if bits_compared < 100:
                continue

            xor_result = np.logical_xor(code1, code2_shifted)
            masked_xor = xor_result & combined_mask
            distance = np.sum(masked_xor) / bits_compared

            if distance < best_distance:
                best_distance = distance
                best_bits = bits_compared

        is_match = best_distance < self._threshold
        confidence = 1.0 - (best_distance / self._threshold) if is_match else 0.0
        match_score = 1.0 - best_distance

        return IrisMatchResult(
            hamming_distance=best_distance,
            match_score=match_score,
            is_match=is_match,
            bits_compared=best_bits,
            confidence=confidence,
        )


class IrisLivenessDetector:
    """
    Detect presentation attacks on iris recognition systems.

    Analyzes pupil dynamics, specular reflections, and texture authenticity.
    """

    def __init__(
        self,
        pupil_response_threshold: float = 0.15,
        reflection_threshold: float = 0.5,
        texture_threshold: float = 0.6,
    ) -> None:
        """Initialize the liveness detector."""
        self._pupil_threshold = pupil_response_threshold
        self._reflection_threshold = reflection_threshold
        self._texture_threshold = texture_threshold

    def detect(
        self,
        images: list[np.ndarray],
        pupil_radii: list[float] | None = None,
    ) -> LivenessResult:
        """
        Detect liveness from a sequence of iris images.

        Args:
            images: Sequence of eye images (for pupil dynamics)
            pupil_radii: Pre-computed pupil radii (optional)

        Returns:
            LivenessResult with confidence scores
        """
        if len(images) < 2:
            return LivenessResult(
                is_live=False,
                confidence=0.0,
                pupil_response=0.0,
                specular_reflection_score=0.0,
                texture_authenticity=0.0,
                details={"error": "Insufficient images for liveness detection"},
            )

        pupil_response = self._analyze_pupil_dynamics(images, pupil_radii)
        reflection_score = self._analyze_specular_reflections(images[0])
        texture_score = self._analyze_texture_authenticity(images[0])

        pupil_live = pupil_response > self._pupil_threshold
        reflection_live = reflection_score > self._reflection_threshold
        texture_live = texture_score > self._texture_threshold

        is_live = pupil_live and reflection_live and texture_live
        confidence = (pupil_response + reflection_score + texture_score) / 3.0

        return LivenessResult(
            is_live=is_live,
            confidence=confidence,
            pupil_response=pupil_response,
            specular_reflection_score=reflection_score,
            texture_authenticity=texture_score,
            details={
                "pupil_live": pupil_live,
                "reflection_live": reflection_live,
                "texture_live": texture_live,
            },
        )

    def _analyze_pupil_dynamics(
        self,
        images: list[np.ndarray],
        pupil_radii: list[float] | None,
    ) -> float:
        """Analyze pupil light response dynamics."""
        if pupil_radii is not None and len(pupil_radii) >= 2:
            radii = np.array(pupil_radii)
        else:
            radii = []  # type: ignore[assignment]
            segmenter = IrisSegmenter()
            for img in images:
                try:
                    _, pupil_r, _, _ = segmenter.segment(img)
                    radii.append(pupil_r)  # type: ignore[attr-defined]
                except Exception:
                    pass

            if len(radii) < 2:
                return 0.0

            radii = np.array(radii)

        variation = np.std(radii) / (np.mean(radii) + 1e-8)
        response_score = min(1.0, variation / 0.2)

        return float(response_score)

    def _analyze_specular_reflections(self, image: np.ndarray) -> float:
        """Analyze specular reflection patterns for authenticity."""
        if image.ndim == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image.copy()

        threshold = np.percentile(gray, 99)
        bright_spots = gray > threshold

        num_spots = np.sum(bright_spots)
        total_pixels = gray.size

        spot_ratio = num_spots / total_pixels
        expected_ratio = 0.005

        if spot_ratio < expected_ratio / 10:
            return 0.2
        elif spot_ratio > expected_ratio * 10:
            return 0.3

        return float(min(1.0, 1.0 - abs(spot_ratio - expected_ratio) / expected_ratio))

    def _analyze_texture_authenticity(self, image: np.ndarray) -> float:
        """Analyze iris texture for authenticity markers."""
        if image.ndim == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image.copy()

        gray = gray.astype(np.float64)
        gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)

        gx = np.diff(gray, axis=1)
        gy = np.diff(gray, axis=0)

        gradient_variance = (np.var(gx) + np.var(gy)) / 2
        expected_variance = 0.01

        if gradient_variance < expected_variance / 10:
            return 0.2

        fft = np.fft.fft2(gray)
        magnitude = np.abs(fft)
        magnitude = np.fft.fftshift(magnitude)

        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        low_freq = magnitude[cy - 10 : cy + 10, cx - 10 : cx + 10].sum()
        high_freq = magnitude.sum() - low_freq

        freq_ratio = high_freq / (low_freq + 1e-8)
        expected_ratio = 10.0

        if freq_ratio < expected_ratio / 5:
            return 0.3

        return float(min(1.0, gradient_variance / expected_variance * 0.5 + 0.5))


class IrisRecognizer:
    """
    Complete iris recognition system.

    Integrates segmentation, normalization, encoding, matching, and liveness detection.
    """

    def __init__(
        self,
        match_threshold: float = 0.32,
        liveness_required: bool = True,
    ) -> None:
        """Initialize the iris recognizer."""
        self._segmenter = IrisSegmenter()
        self._normalizer = IrisNormalizer()
        self._encoder = IrisEncoder()
        self._matcher = IrisMatcher(match_threshold)
        self._liveness_detector = IrisLivenessDetector()
        self._liveness_required = liveness_required

    def extract_features(self, image: np.ndarray) -> IrisFeatures:
        """
        Extract iris features from an eye image.

        Args:
            image: Eye image (grayscale or RGB)

        Returns:
            IrisFeatures containing iris code and metadata
        """
        pupil_center, pupil_radius, iris_center, iris_radius = self._segmenter.segment(image)

        normalized, mask = self._normalizer.normalize(
            image, pupil_center, pupil_radius, iris_center, iris_radius
        )

        iris_code, code_mask = self._encoder.encode(normalized, mask)

        quality_score = self._compute_quality(normalized, mask)

        return IrisFeatures(
            iris_code=iris_code,
            mask=code_mask,
            pupil_center=pupil_center,
            pupil_radius=pupil_radius,
            iris_center=iris_center,
            iris_radius=iris_radius,
            quality_score=quality_score,
            normalized_iris=normalized,
        )

    def verify(
        self,
        probe_image: np.ndarray,
        enrolled_features: IrisFeatures,
        liveness_images: list[np.ndarray] | None = None,
    ) -> tuple[IrisMatchResult, LivenessResult | None]:
        """
        Verify an iris against enrolled features.

        Args:
            probe_image: Probe iris image
            enrolled_features: Enrolled iris features
            liveness_images: Additional images for liveness detection

        Returns:
            Tuple of (match_result, liveness_result)
        """
        probe_features = self.extract_features(probe_image)

        match_result = self._matcher.match(
            probe_features.iris_code,
            probe_features.mask,
            enrolled_features.iris_code,
            enrolled_features.mask,
        )

        liveness_result = None
        if self._liveness_required:
            images = [probe_image]
            if liveness_images:
                images.extend(liveness_images)

            liveness_result = self._liveness_detector.detect(images)

            if not liveness_result.is_live:
                match_result = IrisMatchResult(
                    hamming_distance=match_result.hamming_distance,
                    match_score=0.0,
                    is_match=False,
                    bits_compared=match_result.bits_compared,
                    confidence=0.0,
                )

        return match_result, liveness_result

    def _compute_quality(
        self,
        normalized_iris: np.ndarray,
        mask: np.ndarray,
    ) -> float:
        """Compute iris image quality score."""
        usable_ratio = np.sum(mask) / mask.size

        gradient_x = np.diff(normalized_iris, axis=1)
        gradient_y = np.diff(normalized_iris, axis=0)
        sharpness = np.mean(np.abs(gradient_x)) + np.mean(np.abs(gradient_y))

        contrast = np.std(normalized_iris[mask])

        quality = 0.4 * usable_ratio + 0.3 * min(1.0, sharpness * 10) + 0.3 * min(1.0, contrast * 5)  # type: ignore[operator]

        return float(quality)

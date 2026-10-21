# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Visual anomaly analyser: learn normal appearance, then detect, localise, classify and explain.

:class:`VisualAnomalyAnalyzer` is fitted on normal images of a domain (faces
from one capture station, a product line, a camera view) and then scrutinises
any image of that domain:

* **Detect across image types.** Grayscale, RGB/RGBA, 8/16-bit, float
  radiometric and multispectral inputs are all accepted through
  :func:`~omni_mercury_engine.vision.imaging.to_canonical`, with the value
  mapping recorded.
* **Learn from normal data.** Interpretable patch features
  (:mod:`~omni_mercury_engine.vision.features`) of the normal images build a
  greedy k-centre coreset memory bank (PatchCore's coreset selection, Roth et
  al., 2022), a Ledoit-Wolf-shrunk Gaussian (Mahalanobis scoring, as in
  PaDiM, Defard et al., 2020), and — for aligned inputs such as cropped faces —
  an optional per-position Gaussian.
* **Calibrated decisions.** A held-out normal split sets the image threshold by
  split conformal prediction: the reported ``p_value`` is
  ``(1 + #{calibration scores >= s}) / (n + 1)``, so flagging when
  ``p_value <= alpha`` keeps the false-alarm rate at most ``alpha`` for images
  exchangeable with the normal set. When the calibration set is too small to
  reach ``alpha`` the report says so instead of pretending.
* **Localise precisely.** Patch scores are spread back over each patch's
  pixels (coverage-averaged), smoothed, and segmented by hysteresis: regions
  are seeded where the map exceeds the calibrated image threshold and grown
  over pixels above a calibrated pixel threshold.
* **Recognise and classify.** Each region's feature deviations, in robust
  z-units of the normal model, are mapped to named anomaly types (occlusion,
  blur, noise, exposure, colour shift, foreign texture, structural change,
  invalid pixels, splice candidate, copy-move duplicate), each with its
  evidence. Labelled examples can also teach domain-specific types
  (:meth:`VisualAnomalyAnalyzer.learn_anomaly_types`).
* **Separate content from capture conditions.** Illumination-relative features
  keep global lighting changes out of the content score; the measured
  environment is compared with the envelope of the normal set, and by default
  the content decision is withheld (``ThreeState.UNAVAILABLE``) when capture
  conditions fall outside it.
* **Interpretable, real-time or batch.** Every result is a
  :class:`VisualAnomalyReport` with a pixel map, regions, per-feature
  evidence, forensic findings, environment, provenance and timing.
  :class:`StreamingVisualMonitor` adds frame persistence and latency tracking.
* **Optional deep evidence.** A fitted deep detector (for example
  :class:`~omni_mercury_engine.detectors.visual.PatchCoreDetector`) can be
  fused as an additional, calibrated map channel.
"""

from __future__ import annotations

import json
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from omni_mercury_engine.verifiers.three_state import ThreeState
from omni_mercury_engine.vision.features import PatchFeatures, extract_patch_features
from omni_mercury_engine.vision.forensics import (
    copy_move_detection,
    jpeg_grid_analysis,
    noise_inconsistency,
    periodic_artifact_analysis,
)
from omni_mercury_engine.vision.imaging import (
    ENVIRONMENT_FIELDS,
    CanonicalImage,
    EnvironmentReport,
    assess_environment,
    to_canonical,
)

Array = np.ndarray[Any, Any]
DeepMapProvider = Callable[[Array], Array]

MODEL_SCHEMA = "mercury_visual_normal_model/v2"

#: Region-level anomaly types and what each one means.
ANOMALY_TYPES: dict[str, str] = {
    "occlusion_or_missing": "region is flat: structure, gradients and fine detail collapsed",
    "blur_or_defocus": "fine detail lost while coarse structure remains",
    "noise_or_sensor_corruption": "excess high-frequency noise",
    "exposure_anomaly": "region much brighter or darker than normal, structure intact",
    "color_shift": "colour departs from normal while texture is intact",
    "foreign_texture_or_object": "micro-texture unlike anything in the normal set",
    "structural_change": "edge geometry (orientation) departs from normal",
    "invalid_pixels": "input contained non-finite values here",
    "splice_candidate": "sensor-noise level inconsistent with the rest of the image",
    "copy_move_duplicate": "region duplicates another part of the image",
    "unclassified_deviation": "deviation without a dominant known signature",
}

#: Physical signature of each content anomaly type: conditions on a region's
#: robust z-deviations from the normal model. ``texture_divergence`` (LBP
#: chi-square, gain-invariant) separates *structure preserved* (exposure,
#: colour) from *structure destroyed* (occlusion, blur, noise, foreign matter);
#: band energies and the noise residual separate how it was destroyed.
TYPE_SIGNATURES: dict[str, tuple[tuple[str, str, float], ...]] = {
    "color_shift": (("color_opponent", ">=", 5.0), ("texture_divergence", "<=", 10.0)),
    "exposure_anomaly": (
        ("abs_intensity_mean", ">=", 3.0),
        ("texture_divergence", "<=", 15.0),
        ("color_opponent", "<=", 5.0),
    ),
    "structural_change": (
        ("orientation_divergence", ">=", 20.0),
        ("orientation_minus_texture", ">=", 0.0),
    ),
    "noise_or_sensor_corruption": (
        ("noise_residual", ">=", 5.0),
        ("frequency_fine", ">=", 3.0),
        ("geometry_or_color_change", "<=", 10.0),
        ("abs_intensity_mean", "<=", 3.0),
    ),
    "foreign_texture_or_object": (
        ("texture_divergence", ">=", 10.0),
        ("frequency_fine", ">=", -3.0),
        ("geometry_or_color_change", ">=", 10.0),
    ),
    "blur_or_defocus": (
        ("frequency_fine", "<=", -5.0),
        ("frequency_mid", "<=", -5.0),
        ("abs_intensity_mean", "<=", 3.0),
    ),
    "occlusion_or_missing": (
        ("frequency_fine", "<=", -5.0),
        ("noise_residual", "<=", -3.0),
        ("texture_divergence", ">=", 10.0),
        ("abs_intensity_mean", ">=", 3.0),
    ),
}

#: Labels attached from definitive input or forensic evidence; reported as
#: secondary labels unless no content signature applies.
_FORENSIC_TYPES = ("invalid_pixels", "copy_move_duplicate", "splice_candidate")

_FORENSIC_STATS = ("noise_inconsistency_z", "blockiness", "periodic_peak_excess")

#: Median patch-level ``lattice_any_axis`` of the normal images above which the
#: domain is treated as aperiodic and the lattice group is switched off. A
#: texture without translation symmetry has no symmetry to break, so the
#: group would only add normal-image variance. Measured on MVTec AD textures at
#: 256 px (finest scale): grid 0.05, carpet 0.22-0.27 and a synthetic weave
#: 0.03 are periodic; leather 0.66-0.72, wood 0.68-0.73 and tile 0.83 are not.
LATTICE_APERIODIC_RESIDUAL = 0.4


@dataclass
class AnalyzerConfig:
    """Configuration of :class:`VisualAnomalyAnalyzer`.

    Attributes:
        patch_size: Patch side in pixels.
        stride: Patch step in pixels.
        scales: Feature pyramid scales.
        resize_to: Optional ``(height, width)`` every image is resized to;
            required for ``positional_model``.
        illumination_invariant: Keep global lighting changes out of content
            features (they are still measured by the environment check).
        positional_model: Also fit a per-patch-position Gaussian (only for
            aligned inputs such as cropped faces or fixed-camera views).
        alpha: Target image-level false-alarm rate for the conformal decision.
        pixel_alpha: Tail mass of normal pixel scores above the region-growing
            threshold.
        max_memory: Memory-bank size cap (coreset size).
        n_neighbors: Neighbours averaged for the kNN patch score.
        smoothing_sigma: Gaussian smoothing of the pixel map, in pixels.
        min_region_area: Smallest reported region, in pixels (default: a
            quarter patch).
        forensics: Run splice / JPEG-grid / periodic-artifact checks.
        copy_move: Also run copy-move detection (thorough; ~0.3 s per 256 px image).
        environment_z_limit: Robust z beyond which a capture condition is
            outside the normal envelope.
        environment_policy: ``"withhold"`` (content decision becomes
            UNAVAILABLE outside the envelope) or ``"report"`` (decide anyway,
            flagging the conditions).
        calibration_fraction: Share of the fit images held out for calibration
            when no explicit calibration set is given.
        type_simulations: Synthetic anomalies per type generated on the
            domain's own training images to learn domain-adapted type
            prototypes; ``0`` falls back to the fixed physical-signature rules.
        open_set_factor: A region farther than this multiple of a type's
            typical (99th-percentile) prototype distance is reported as
            ``unclassified_deviation`` rather than forced into a known type.
        seed: Seed for the train/calibration split, coreset start and synthesis.
    """

    patch_size: int = 16
    stride: int = 8
    scales: tuple[float, ...] = (1.0, 0.5)
    resize_to: tuple[int, int] | None = None
    illumination_invariant: bool = True
    positional_model: bool = False
    alpha: float = 0.05
    pixel_alpha: float = 0.002
    max_memory: int = 4096
    n_neighbors: int = 3
    smoothing_sigma: float = 4.0
    min_region_area: int | None = None
    forensics: bool = True
    copy_move: bool = False
    environment_z_limit: float = 6.0
    environment_policy: str = "withhold"
    calibration_fraction: float = 0.4
    type_simulations: int = 12
    open_set_factor: float = 3.0
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary of the configuration."""
        data = asdict(self)
        data["scales"] = list(self.scales)
        data["resize_to"] = None if self.resize_to is None else list(self.resize_to)
        return data

    @classmethod
    def fast(cls, **overrides: Any) -> AnalyzerConfig:
        """Return a real-time profile: one coarse scale, a sparser grid, cheap forensics only."""
        base: dict[str, Any] = {
            "scales": (0.5,),
            "stride": 16,
            "max_memory": 2048,
            "copy_move": False,
        }
        base.update(overrides)
        return cls(**base)


@dataclass
class AnomalyRegion:
    """One localised anomaly.

    Attributes:
        box: ``(x0, y0, x1, y1)`` pixel bounding box (exclusive upper bounds).
        area: Region area in pixels.
        area_fraction: Region area over image area.
        centroid: ``(x, y)`` centre of mass of the region.
        peak: ``(x, y)`` of the highest map value in the region.
        peak_score: Highest map value (1.0 is the calibrated image threshold).
        mean_score: Mean map value over the region.
        anomaly_type: The best-scoring entry of :data:`ANOMALY_TYPES`.
        type_scores: Score in ``[0, 1]`` of every type that applies.
        evidence: Human-readable statements behind the classification.
        feature_deviation: Robust z of each named feature over the region.
        learned_type: Nearest learned type, when types were taught.
        learned_confidence: Softmax confidence of ``learned_type``.
    """

    box: tuple[int, int, int, int]
    area: int
    area_fraction: float
    centroid: tuple[float, float]
    peak: tuple[int, int]
    peak_score: float
    mean_score: float
    anomaly_type: str
    type_scores: dict[str, float]
    evidence: list[str]
    feature_deviation: dict[str, float]
    learned_type: str | None = None
    learned_confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary."""
        return asdict(self)


@dataclass
class VisualAnomalyReport:
    """The complete, interpretable result of analysing one image.

    Attributes:
        state: ``GROUNDED`` when a calibrated content decision was reached,
            ``UNAVAILABLE`` when it was withheld (reason given).
        reason: Why the state is what it is.
        is_anomalous: The calibrated content decision (``None`` when
            withheld). Its false-alarm guarantee covers content only; forensic
            evidence is reported separately in ``manipulation_suspected``.
        image_score: Image score in threshold units (1.0 = calibrated threshold).
        p_value: Conformal p-value against the normal calibration scores.
        alpha: Decision level.
        calibration_size: Number of calibration images behind ``p_value``.
        anomaly_map: ``(H, W)`` float32 map in threshold units.
        regions: Localised anomalies, strongest first.
        forensics: Splice / JPEG / copy-move / periodic findings, with robust
            z against the normal set where calibrated.
        environment: Measured capture conditions.
        environment_z: Robust z of each condition against the normal envelope.
        environment_in_envelope: Whether every condition is inside the envelope.
        provenance: How the input was mapped to canonical form.
        explanation: Ordered, human-readable findings.
        timing_ms: Wall time per stage.
        manipulation_suspected: Whether any forensic screen fired (robust
            z > 6 against the normal set, or a copy-move duplication). A
            screening result with stated thresholds, not a calibrated decision.
        forensic_alerts: The forensic findings behind ``manipulation_suspected``.
        deep_evidence: Deep-detector channel summary, if one is attached.
    """

    state: ThreeState
    reason: str
    is_anomalous: bool | None
    image_score: float
    p_value: float | None
    alpha: float
    calibration_size: int
    anomaly_map: Array
    regions: list[AnomalyRegion]
    forensics: dict[str, Any]
    environment: EnvironmentReport
    environment_z: dict[str, float]
    environment_in_envelope: bool
    provenance: dict[str, Any]
    explanation: list[str]
    timing_ms: dict[str, float]
    manipulation_suspected: bool = False
    forensic_alerts: list[str] = field(default_factory=list)
    deep_evidence: dict[str, Any] | None = None

    def to_dict(self, include_map: bool = False) -> dict[str, Any]:
        """Return a JSON-safe dictionary (the pixel map only when requested)."""
        data: dict[str, Any] = {
            "state": self.state.value,
            "reason": self.reason,
            "is_anomalous": self.is_anomalous,
            "image_score": self.image_score,
            "p_value": self.p_value,
            "alpha": self.alpha,
            "calibration_size": self.calibration_size,
            "regions": [r.to_dict() for r in self.regions],
            "forensics": _json_safe(self.forensics),
            "environment": self.environment.to_dict(),
            "environment_z": self.environment_z,
            "environment_in_envelope": self.environment_in_envelope,
            "provenance": self.provenance,
            "explanation": self.explanation,
            "timing_ms": self.timing_ms,
            "manipulation_suspected": self.manipulation_suspected,
            "forensic_alerts": self.forensic_alerts,
            "deep_evidence": self.deep_evidence,
        }
        if include_map:
            data["anomaly_map"] = self.anomaly_map.tolist()
        return data


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _ledoit_wolf(x: Array) -> tuple[Array, Array]:
    """Ledoit-Wolf (2004) shrinkage covariance; returns ``(mean, precision)``."""
    n, d = x.shape
    mean = x.mean(axis=0)
    centered = x - mean
    sample = centered.T @ centered / n
    mu = float(np.trace(sample)) / d
    target = mu * np.eye(d)
    delta = float(np.sum((sample - target) ** 2)) / d
    beta_terms = np.einsum("ni,nj->nij", centered, centered) if n * d * d <= 4e7 else None
    if beta_terms is not None:
        beta = float(np.sum((beta_terms - sample) ** 2)) / (n * n * d)
    else:
        sq = centered**2
        beta = float(np.sum(sq.T @ sq) / n - np.sum(sample**2)) / (n * d)
    shrinkage = min(max(beta, 0.0), delta) / max(delta, 1e-12)
    covariance = shrinkage * target + (1.0 - shrinkage) * sample
    precision = np.linalg.pinv(covariance + 1e-6 * np.eye(d))
    return mean, precision


def _ledoit_wolf_on(x: Array, active: Array) -> tuple[Array, Array]:
    """Ledoit-Wolf over the ``active`` columns only.

    Switched-off feature groups carry a constant zero z; including them would
    still shift the shrinkage target (the mean variance) and so the precision
    of every active feature. Their rows and columns of the returned precision
    are zero, so they contribute nothing to a Mahalanobis distance.
    """
    if active.all():
        return _ledoit_wolf(x)
    dim = x.shape[1]
    mean, precision = np.zeros(dim), np.zeros((dim, dim))
    mean[active], precision[np.ix_(active, active)] = _ledoit_wolf(x[:, active])
    return mean, precision


def _mahalanobis(x: Array, mean: Array, precision: Array) -> Array:
    centered = x - mean
    return np.asarray(
        np.sqrt(np.maximum(np.einsum("ni,ij,nj->n", centered, precision, centered), 0.0))
    )


def _greedy_coreset(x: Array, size: int, seed: int) -> Array:
    """Greedy k-centre selection on a random projection (PatchCore's coreset)."""
    n = x.shape[0]
    if n <= size:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    dim = min(x.shape[1], 24)
    projection = rng.normal(size=(x.shape[1], dim)) / np.sqrt(dim)
    z = x @ projection
    selected = np.empty(size, dtype=np.int64)
    selected[0] = int(rng.integers(n))
    distances = np.sum((z - z[selected[0]]) ** 2, axis=1)
    for i in range(1, size):
        nxt = int(np.argmax(distances))
        selected[i] = nxt
        distances = np.minimum(distances, np.sum((z - z[nxt]) ** 2, axis=1))
    return selected


def _knn_mean_distance(queries: Array, bank: Array, k: int, chunk: int = 1024) -> Array:
    bank_sq = np.sum(bank * bank, axis=1)
    out = np.empty(queries.shape[0])
    k = min(k, bank.shape[0])
    for start in range(0, queries.shape[0], chunk):
        q = queries[start : start + chunk]
        d2 = np.maximum(np.sum(q * q, axis=1)[:, np.newaxis] + bank_sq - 2.0 * q @ bank.T, 0.0)
        nearest = np.partition(d2, k - 1, axis=1)[:, :k]
        out[start : start + chunk] = np.sqrt(nearest).mean(axis=1)
    return out


def _robust_z(values: Array, median: Array, scale: Array) -> Array:
    return (values - median) / scale


def _chi2(hist: Array, reference: Array) -> Array:
    return np.asarray(0.5 * np.sum((hist - reference) ** 2 / (hist + reference + 1e-9), axis=-1))


class VisualAnomalyAnalyzer:
    """Learn the normal appearance of a domain; detect, localise, classify and explain deviations.

    Example:
        analyzer = VisualAnomalyAnalyzer().fit(normal_images)
        report = analyzer.analyze(test_image)
        if report.is_anomalous:
            for region in report.regions:
                print(region.anomaly_type, region.box, region.evidence)
    """

    def __init__(self, config: AnalyzerConfig | None = None) -> None:
        """Create an unfitted analyser with ``config`` (defaults if omitted)."""
        self.config = config or AnalyzerConfig()
        if self.config.environment_policy not in ("withhold", "report"):
            raise ValueError("environment_policy must be 'withhold' or 'report'")
        self._fitted = False
        self._deep: DeepMapProvider | None = None
        self._deep_name: str | None = None
        self._learned_types: dict[str, Array] = {}
        self._learned_scale: Array | None = None
        self.learned_type_loo_accuracy: float | None = None
        self._lattice_periodicity: float | None = None
        self._lattice_active = False
        self._skip_lattice = False

    # ------------------------------------------------------------------ input
    def _prepare(self, image: Any) -> CanonicalImage:
        canonical = to_canonical(image)
        target = self.config.resize_to
        if target is None or canonical.shape == tuple(target):
            return canonical
        height, width = canonical.shape
        zoom = (target[0] / height, target[1] / width)
        pixels = np.clip(
            ndimage.zoom(canonical.pixels.astype(np.float64), (*zoom, 1), order=1), 0.0, 1.0
        )[: target[0], : target[1]]
        luminance = np.clip(
            ndimage.zoom(canonical.luminance.astype(np.float64), zoom, order=1), 0.0, 1.0
        )[: target[0], : target[1]]
        invalid = ndimage.zoom(canonical.invalid_mask.astype(np.float64), zoom, order=0) > 0.5
        return CanonicalImage(
            pixels=pixels.astype(np.float32),
            luminance=luminance.astype(np.float32),
            invalid_mask=invalid[: target[0], : target[1]],
            provenance=canonical.provenance,
        )

    def _features(self, image: CanonicalImage) -> PatchFeatures:
        cfg = self.config
        feats = extract_patch_features(
            image,
            patch_size=cfg.patch_size,
            stride=cfg.stride,
            scales=cfg.scales,
            illumination_invariant=cfg.illumination_invariant,
            lattice=not self._skip_lattice,
        )
        fitted_names = getattr(self, "_names", None)
        if fitted_names is not None and feats.names != fitted_names:
            fitted_colour = any(n.endswith("color_saturation") for n in fitted_names)
            raise ValueError(
                f"image layout {image.provenance.layout!r} does not match the normal model "
                f"(fitted on {'colour' if fitted_colour else 'single-channel'} images); fit one "
                "analyser per image type"
            )
        return feats

    def _forensic_stats(self, image: CanonicalImage) -> dict[str, Any]:
        lum = image.luminance
        result: dict[str, Any] = {}
        if self.config.forensics:
            result["noise_inconsistency"] = noise_inconsistency(lum)
            result["jpeg_grid"] = jpeg_grid_analysis(lum)
            result["periodic_artifacts"] = periodic_artifact_analysis(lum)
        if self.config.copy_move:
            result["copy_move"] = copy_move_detection(lum)
        return result

    @staticmethod
    def _forensic_vector(stats: dict[str, Any]) -> dict[str, float]:
        out: dict[str, float] = {}
        if "noise_inconsistency" in stats:
            out["noise_inconsistency_z"] = float(stats["noise_inconsistency"]["max_abs_z"])
            out["blockiness"] = float(stats["jpeg_grid"]["blockiness"])
            out["periodic_peak_excess"] = float(stats["periodic_artifacts"]["peak_excess"])
        return out

    # -------------------------------------------------------------- deep evidence
    def attach_deep_evidence(self, provider: DeepMapProvider, name: str = "deep") -> None:
        """Fuse a deep anomaly-map provider as an extra channel (attach before :meth:`fit`).

        Args:
            provider: Callable mapping a ``(N, H, W, 3)`` float batch in
                ``[0, 1]`` to ``(N, H', W')`` anomaly maps (for example a fitted
                PatchCore detector's ``detect(...)["anomaly_maps"]``).
            name: Label used in reports.
        """
        if self._fitted:
            raise RuntimeError("attach deep evidence before fit so it is calibrated")
        self._deep = provider
        self._deep_name = name

    def _deep_map(self, image: CanonicalImage) -> Array | None:
        if self._deep is None:
            return None
        rgb = (
            image.pixels
            if image.pixels.shape[2] == 3
            else np.repeat(image.luminance[..., None], 3, 2)
        )
        raw = np.asarray(self._deep(rgb[np.newaxis].astype(np.float32)), dtype=np.float64)
        raw = raw.reshape(raw.shape[-2:]) if raw.ndim == 3 else raw
        height, width = image.shape
        return np.asarray(
            ndimage.zoom(raw, (height / raw.shape[0], width / raw.shape[1]), order=1)[
                :height, :width
            ]
        )

    # -------------------------------------------------------------------- fit
    def fit(
        self,
        normal_images: Sequence[Any],
        calibration_images: Sequence[Any] | None = None,
    ) -> VisualAnomalyAnalyzer:
        """Learn the normal appearance and calibrate the decision thresholds.

        Args:
            normal_images: Normal (defect-free) images of the domain.
            calibration_images: Optional separate normal images for
                calibration; otherwise ``calibration_fraction`` of
                ``normal_images`` is held out (never used to build the model).

        Returns:
            ``self``.

        Raises:
            ValueError: fewer than 4 images in total, or mismatched sizes with
                ``positional_model`` and no ``resize_to``.
        """
        cfg = self.config
        images = [self._prepare(img) for img in normal_images]
        self._skip_lattice = False
        if calibration_images is None:
            if len(images) < 4:
                raise ValueError("at least 4 normal images are required (train + calibration)")
            rng = np.random.default_rng(cfg.seed)
            order = rng.permutation(len(images))
            n_cal = max(2, round(len(images) * cfg.calibration_fraction))
            calibration = [images[i] for i in order[:n_cal]]
            train = [images[i] for i in order[n_cal:]]
        else:
            train = images
            calibration = [self._prepare(img) for img in calibration_images]
            if len(train) < 2 or len(calibration) < 2:
                raise ValueError("at least 2 training and 2 calibration images are required")

        if cfg.positional_model and len({img.shape for img in train + calibration}) != 1:
            raise ValueError("positional_model needs equally sized images; set resize_to")

        train_features = [self._features(img) for img in train]
        sample = train_features[0]
        if any(f.names != sample.names for f in train_features):
            raise ValueError(
                "normal images mix colour and single-channel layouts; fit one analyser per image type"
            )
        self._names = sample.names
        self._base_names = sample.base_names
        self._groups = sample.groups
        stacked = np.concatenate([f.flat for f in train_features]).astype(np.float64)

        self._median = np.median(stacked, axis=0)
        mad = np.median(np.abs(stacked - self._median), axis=0) * 1.4826
        spread = np.std(stacked, axis=0)
        self._scale = np.maximum(np.where(mad > 1e-9, mad, spread), 1e-3)
        self._lattice_periodicity, self._lattice_active = self._lattice_relevance(
            stacked, sample.base_names, sample.scales
        )
        if not self._lattice_active:
            # An infinite scale makes every lattice z exactly 0 in every
            # downstream model (memory bank, Gaussian, classification), and
            # the group is no longer computed for any later image.
            lattice = np.array([g == "lattice" for g in sample.groups])
            self._scale = np.where(lattice, np.inf, self._scale)
            self._skip_lattice = True
        active = np.isfinite(self._scale)
        standardized = _robust_z(stacked, self._median, self._scale)

        bank_index = _greedy_coreset(standardized, cfg.max_memory, cfg.seed)
        self._bank = standardized[bank_index]
        self._gauss_mean, self._gauss_precision = _ledoit_wolf_on(standardized, active)

        self._positional: tuple[Array, Array] | None = None
        if cfg.positional_model:
            grid_shape = sample.values.shape[:2]
            per_image = np.stack(
                [
                    _robust_z(f.flat.astype(np.float64), self._median, self._scale)
                    for f in train_features
                ]
            )
            means = np.empty((per_image.shape[1], per_image.shape[2]))
            precisions = np.empty((per_image.shape[1], per_image.shape[2], per_image.shape[2]))
            for p in range(per_image.shape[1]):
                means[p], precisions[p] = _ledoit_wolf_on(per_image[:, p, :], active)
            self._positional = (means, precisions)
            self._grid_shape = grid_shape

        self._texture_index = np.array(
            [i for i, n in enumerate(self._base_names) if n.startswith("texture_")]
        )
        self._orientation_index = np.array(
            [i for i, n in enumerate(self._base_names) if n.startswith("orientation_")]
        )
        self._texture_reference = stacked[:, self._texture_index].mean(axis=0)
        self._orientation_reference = stacked[:, self._orientation_index].mean(axis=0)
        tex_div = self._divergence(stacked, self._texture_index, self._texture_reference)
        ori_div = self._divergence(stacked, self._orientation_index, self._orientation_reference)
        self._div_stats = {
            "texture": (
                float(np.median(tex_div)),
                float(np.median(np.abs(tex_div - np.median(tex_div))) * 1.4826 + 1e-6),
            ),
            "orientation": (
                float(np.median(ori_div)),
                float(np.median(np.abs(ori_div - np.median(ori_div))) * 1.4826 + 1e-6),
            ),
        }

        # Calibration: a small first split (at most 3 images — thousands of
        # patches each) normalises the score components and sets the pixel
        # threshold; the remaining, disjoint images set the conformal image
        # threshold, so calibration and test scores stay exchangeable.
        n_norm = max(1, min(3, len(calibration) // 3))
        norm_set, conformal_set = calibration[:n_norm], calibration[n_norm:]
        if not conformal_set:
            conformal_set = norm_set
        self._component_norm: dict[str, tuple[float, float]] = {}
        components = [self._raw_components(img) for img in norm_set]
        for key in components[0]:
            values = np.concatenate([c[key].ravel() for c in components])
            low, high = np.quantile(values, [0.5, 0.99])
            self._component_norm[key] = (float(low), float(max(high - low, 1e-6)))

        norm_maps = [self._fused_map(img)[0] for img in norm_set]
        pixel_values = np.concatenate([m.ravel() for m in norm_maps])
        self._pixel_threshold_raw = float(np.quantile(pixel_values, 1.0 - cfg.pixel_alpha))

        cal_scores = np.array([float(self._fused_map(img)[0].max()) for img in conformal_set])
        self._calibration_scores = np.sort(cal_scores)
        n = cal_scores.size
        rank = int(np.ceil((n + 1) * (1.0 - cfg.alpha)))
        self._image_threshold_raw = (
            float(self._calibration_scores[rank - 1]) if rank <= n else float(cal_scores.max())
        )
        self._calibration_sufficient = rank <= n
        if self._pixel_threshold_raw >= self._image_threshold_raw:
            self._pixel_threshold_raw = 0.5 * self._image_threshold_raw

        environments = np.stack([assess_environment(img).vector() for img in images])
        self._env_median = np.median(environments, axis=0)
        env_mad = np.median(np.abs(environments - self._env_median), axis=0) * 1.4826
        env_floor = np.array([0.02, 0.01, 0.05, 0.05, 0.01, 0.01, 0.02])
        self._env_scale = np.maximum(env_mad, env_floor)

        forensic = [self._forensic_vector(self._forensic_stats(img)) for img in images]
        self._forensic_norm: dict[str, tuple[float, float]] = {}
        for key in _FORENSIC_STATS:
            vals = np.array([f[key] for f in forensic if key in f])
            if vals.size:
                med = float(np.median(vals))
                scale = float(np.median(np.abs(vals - med)) * 1.4826)
                self._forensic_norm[key] = (med, max(scale, 0.05 * abs(med) + 1e-3))

        self._n_train, self._n_calibration = len(train), int(n)
        self._prototypes: dict[str, Any] | None = None
        if cfg.type_simulations > 0:
            self._prototypes = self._learn_type_prototypes(train) or None
        self._fitted = True
        return self

    @staticmethod
    def _compress(vector: Array) -> Array:
        """Signed log compression of z-deviations: comparable across domains of different variability."""
        return np.sign(vector) * np.log1p(np.abs(vector))

    def _learn_type_prototypes(self, train: Sequence[CanonicalImage]) -> dict[str, Any]:
        """Synthesise every anomaly type on training images and fit per-variant Gaussians.

        Synthetic anomalies go through the same scoring and segmentation as
        real inputs, and the deviation is read over the *detected* region that
        overlaps the synthetic mask — so prototypes see exactly what
        inference sees. Two-polarity types (darker / brighter) get one
        prototype per polarity. Each variant's signed-log-compressed deviation
        vectors get a mean and diagonal variance; classification is the
        Gaussian posterior under equal priors (summed per type), and each
        variant's 99th-percentile normalised distance sets its open-set bound.
        Synthetic anomalies too weak to be detected are skipped: inference
        could never classify them either.
        """
        from omni_mercury_engine.vision.simulation import (
            POLAR_TYPES,
            SIMULATED_TYPES,
            simulate_anomaly,
        )

        rng = np.random.default_rng(self.config.seed + 17)
        variants: list[tuple[str, str | None]] = []
        for anomaly_type in SIMULATED_TYPES:
            if anomaly_type in POLAR_TYPES:
                variants += [(anomaly_type, "dark"), (anomaly_type, "bright")]
            else:
                variants.append((anomaly_type, None))
        samples: dict[tuple[str, str | None], list[Array]] = {v: [] for v in variants}
        attempts = 0
        for variant in variants:
            anomaly_type, polarity = variant
            tries = 0
            while (
                len(samples[variant]) < self.config.type_simulations
                and tries < 3 * self.config.type_simulations
            ):
                base = train[attempts % len(train)]
                attempts += 1
                tries += 1
                pixels, truth = simulate_anomaly(base.pixels, anomaly_type, rng, polarity=polarity)
                canonical = to_canonical(pixels.astype(np.float32))
                fused, _ = self._fused_map(canonical)
                score_map = fused / self._image_threshold_raw
                best_mask, best_overlap = None, 0
                for mask, _peak, _index in self._segment(score_map):
                    overlap = int(np.sum(mask & truth))
                    if overlap > best_overlap:
                        best_mask, best_overlap = mask, overlap
                if best_mask is None:
                    continue
                _, vector = self._region_deviation(best_mask)
                samples[variant].append(self._compress(vector))
        kept = [v for v in variants if len(samples[v]) >= 3]
        if len({v[0] for v in kept}) < 2:
            return {}
        matrices = [np.stack(samples[v]) for v in kept]
        means = np.stack([m.mean(axis=0) for m in matrices])
        # Pooled (shared) diagonal within-variant variance: stable with few
        # samples per variant, and the posterior then ranks by distance alone.
        within = np.concatenate([m - mu for m, mu in zip(matrices, means, strict=True)])
        dof = max(within.shape[0] - len(kept), 1)
        pooled_var = np.maximum(np.sum(within**2, axis=0) / dof, 1e-3)
        # Open-set bound from leave-one-out distances: how far a *new* sample of
        # a variant typically falls from that variant's centroid.
        loo_q95 = []
        for matrix in matrices:
            n_v = matrix.shape[0]
            loo = []
            for i in range(n_v):
                centroid = (matrix.sum(axis=0) - matrix[i]) / (n_v - 1)
                loo.append(float(np.sum((matrix[i] - centroid) ** 2 / pooled_var)))
            loo_q95.append(float(np.quantile(loo, 0.95)))
        return {
            "types": [v[0] for v in kept],
            "variants": [f"{v[0]}:{v[1]}" if v[1] else v[0] for v in kept],
            "means": means,
            "pooled_var": pooled_var,
            "loo_q95": np.array(loo_q95),
            "samples_per_variant": [len(samples[v]) for v in kept],
        }

    def _prototype_posterior(self, vector: Array) -> tuple[dict[str, float], str, float, float]:
        """Return ``(posterior, best type, normalised distance, open-set bound)`` for one region."""
        assert self._prototypes is not None
        x = self._compress(vector)
        means = self._prototypes["means"]
        d2 = np.sum((x - means) ** 2 / self._prototypes["pooled_var"], axis=1)
        log_like = -0.5 * (d2 - d2.min())
        posterior = np.exp(log_like) / np.exp(log_like).sum()
        best = int(np.argmax(posterior))
        types = self._prototypes["types"]
        # Floor the bound at the chi-square 99th percentile for this
        # dimensionality: very consistent synthetic variants would otherwise
        # get a bound tighter than sampling noise alone justifies.
        from scipy.stats import chi2

        floor = float(chi2.ppf(0.99, df=x.size))
        bound = float(self.config.open_set_factor * max(self._prototypes["loo_q95"][best], floor))
        by_type: dict[str, float] = {}
        for anomaly_type, p in zip(types, posterior, strict=True):
            by_type[anomaly_type] = by_type.get(anomaly_type, 0.0) + float(p)
        return by_type, types[best], float(d2[best]), bound

    def recalibrate(
        self, normal_images: Sequence[Any], update_environment: bool = True
    ) -> VisualAnomalyAnalyzer:
        """Re-anchor the conformal decision on fresh normal images from the deployment session.

        The conformal false-alarm guarantee holds for images exchangeable with
        the calibration set. A new camera session, lens or line can shift
        normal appearance slightly (measured on MVTec AD: carpet's good test
        images separate from its good training images at AUROC 0.66 under
        this analyser), which inflates false alarms. Recalibrating on a few
        in-session normal images restores the guarantee for that session
        without refitting the learned normal model.

        Args:
            normal_images: Normal images from the deployment session (at least 2).
            update_environment: Also widen the capture-condition envelope
                with these images.

        Returns:
            ``self``.
        """
        self._require_fitted()
        prepared = [self._prepare(img) for img in normal_images]
        if len(prepared) < 2:
            raise ValueError("recalibration needs at least 2 normal images")
        scores = np.sort(np.array([float(self._fused_map(img)[0].max()) for img in prepared]))
        n = scores.size
        rank = int(np.ceil((n + 1) * (1.0 - self.config.alpha)))
        self._calibration_scores = scores
        self._image_threshold_raw = float(scores[rank - 1]) if rank <= n else float(scores.max())
        self._calibration_sufficient = rank <= n
        self._n_calibration = n
        if self._pixel_threshold_raw >= self._image_threshold_raw:
            self._pixel_threshold_raw = 0.5 * self._image_threshold_raw
        if update_environment:
            env = np.stack([assess_environment(img).vector() for img in prepared])
            self._env_median = np.median(np.vstack([self._env_median, env]), axis=0)
        return self

    @property
    def is_fitted(self) -> bool:
        """Return True once :meth:`fit` (or :meth:`load`) has run."""
        return self._fitted

    @staticmethod
    def _lattice_relevance(
        stacked: Array, base_names: tuple[str, ...], scales: tuple[float, ...]
    ) -> tuple[float | None, bool]:
        """Decide from the normal data whether the domain has a lattice worth modelling.

        Returns:
            ``(periodicity, active)``: the median patch ``lattice_any_axis`` at
            the finest scale (``None`` without lattice features) and whether it
            is at most :data:`LATTICE_APERIODIC_RESIDUAL`.
        """
        candidates = [i for i, name in enumerate(base_names) if name == "lattice_any_axis"]
        if not candidates:
            return None, False
        finest = max(candidates, key=lambda i: scales[i])
        periodicity = float(np.median(stacked[:, finest]))
        return periodicity, periodicity <= LATTICE_APERIODIC_RESIDUAL

    @property
    def lattice_active(self) -> bool:
        """Return whether the lattice-symmetry features are used for this domain.

        Decided at :meth:`fit` from the normal images: on an aperiodic texture
        (``lattice_periodicity`` above :data:`LATTICE_APERIODIC_RESIDUAL`) the
        group carries no symmetry and is switched off.
        """
        return bool(self._fitted and self._lattice_active)

    @property
    def lattice_periodicity(self) -> float | None:
        """Return the measured median patch lattice residual of the normal images."""
        return self._lattice_periodicity if self._fitted else None

    @property
    def calibration_sufficient(self) -> bool:
        """Return whether the calibration set is large enough to reach ``alpha``."""
        return bool(self._fitted and self._calibration_sufficient)

    def _divergence(self, values: Array, index: Array, reference: Array) -> Array:
        hist = values[:, index]
        hist = hist / (hist.sum(axis=1, keepdims=True) + 1e-9)
        ref = reference / (reference.sum() + 1e-9)
        return _chi2(hist, ref)

    # ---------------------------------------------------------------- scoring
    def _raw_components(self, image: CanonicalImage) -> dict[str, Array]:
        feats = self._features(image)
        grid_shape = feats.values.shape[:2]
        z = _robust_z(feats.flat.astype(np.float64), self._median, self._scale)
        components = {
            "knn": _knn_mean_distance(z, self._bank, self.config.n_neighbors).reshape(grid_shape),
            "mahalanobis": _mahalanobis(z, self._gauss_mean, self._gauss_precision).reshape(
                grid_shape
            ),
        }
        if self._positional is not None:
            means, precisions = self._positional
            centered = z - means
            components["positional"] = np.sqrt(
                np.maximum(np.einsum("pi,pij,pj->p", centered, precisions, centered), 0.0)
            ).reshape(grid_shape)
        deep = self._deep_map(image)
        if deep is not None:
            components["deep"] = deep
        self._last_features = feats
        self._last_z = z.reshape(*grid_shape, -1)
        return components

    def _patch_to_pixels(self, grid_scores: Array, feats: PatchFeatures) -> Array:
        grid = feats.grid
        total = np.zeros((grid.height, grid.width))
        count = np.zeros((grid.height, grid.width))
        size = grid.patch_size
        for i, y in enumerate(grid.rows):
            for j, x in enumerate(grid.cols):
                total[y : y + size, x : x + size] += grid_scores[i, j]
                count[y : y + size, x : x + size] += 1.0
        return total / np.maximum(count, 1.0)

    def _fused_map(self, image: CanonicalImage) -> tuple[Array, dict[str, Array]]:
        components = self._raw_components(image)
        feats = self._last_features
        pixel_maps: dict[str, Array] = {}
        normalised: list[Array] = []
        for key, values in components.items():
            if key == "deep":
                pixel = values
            else:
                pixel = self._patch_to_pixels(values, feats)
            pixel_maps[key] = pixel
            if hasattr(self, "_component_norm") and key in self._component_norm:
                low, span = self._component_norm[key]
                normalised.append((pixel - low) / span)
            else:
                normalised.append(pixel)
        fused = np.mean(np.stack(normalised), axis=0)
        if self.config.smoothing_sigma > 0:
            fused = ndimage.gaussian_filter(fused, self.config.smoothing_sigma, mode="reflect")
        return fused, pixel_maps

    # ----------------------------------------------------------- classification
    def _region_deviation(self, region_mask: Array) -> tuple[dict[str, float], Array]:
        feats = self._last_features
        z = self._last_z
        cy, cx = feats.grid.centers()
        inside = region_mask[
            np.clip(cy.astype(int), 0, region_mask.shape[0] - 1)[:, np.newaxis],
            np.clip(cx.astype(int), 0, region_mask.shape[1] - 1)[np.newaxis, :],
        ]
        if not np.any(inside):
            peak = np.unravel_index(int(np.argmax(region_mask)), region_mask.shape)
            gi = int(np.argmin(np.abs(cy - peak[0])))
            gj = int(np.argmin(np.abs(cx - peak[1])))
            inside = np.zeros(z.shape[:2], dtype=bool)
            inside[gi, gj] = True
        zr = z[inside].mean(axis=0)
        raw = feats.values[inside].astype(np.float64)

        by_base: dict[str, list[float]] = {}
        for value, base in zip(zr, self._base_names, strict=True):
            by_base.setdefault(base, []).append(float(value))
        deviation = {name: float(np.mean(vals)) for name, vals in by_base.items()}

        tex_div = float(
            np.mean(self._divergence(raw, self._texture_index, self._texture_reference))
        )
        ori_div = float(
            np.mean(self._divergence(raw, self._orientation_index, self._orientation_reference))
        )
        med, scale = self._div_stats["texture"]
        deviation["texture_divergence"] = (tex_div - med) / scale
        med, scale = self._div_stats["orientation"]
        deviation["orientation_divergence"] = (ori_div - med) / scale
        for prefix in ("texture_lbp", "orientation_bin"):
            for key in [k for k in deviation if k.startswith(prefix)]:
                del deviation[key]
        vector = np.array([deviation[k] for k in sorted(deviation)])
        return deviation, vector

    @staticmethod
    def _signature_inputs(dev: dict[str, float]) -> dict[str, float]:
        get = dev.get
        tex = get("texture_divergence", 0.0)
        ori = get("orientation_divergence", 0.0)
        color = max(abs(get("color_red_green", 0.0)), abs(get("color_yellow_blue", 0.0)))
        return {
            "abs_intensity_mean": abs(get("intensity_mean", 0.0)),
            "frequency_fine": get("frequency_fine", 0.0),
            "frequency_mid": get("frequency_mid", 0.0),
            "noise_residual": get("noise_residual", 0.0),
            "texture_divergence": tex,
            "orientation_divergence": ori,
            "orientation_minus_texture": ori - tex,
            "color_opponent": color,
            "geometry_or_color_change": max(ori, color),
        }

    @classmethod
    def _rule_types(cls, dev: dict[str, float]) -> tuple[dict[str, float], list[str]]:
        """Score every content anomaly type by its physical signature (soft AND).

        Each type in :data:`TYPE_SIGNATURES` is a conjunction of conditions on
        region deviations in robust z-units of the normal model. A condition's
        margin is its distance past the bound in units of
        ``max(1, 0.2 * |bound|)``; the type's score is the sigmoid of its
        weakest margin, so a type scores above 0.5 only when every condition
        holds. The evidence lists each satisfied type's measured conditions.
        """
        inputs = cls._signature_inputs(dev)
        scores: dict[str, float] = {}
        evidence: list[str] = []
        for anomaly_type, conditions in TYPE_SIGNATURES.items():
            margins = []
            for name, op, bound in conditions:
                unit = max(1.0, 0.2 * abs(bound))
                value = inputs[name]
                margins.append((value - bound) / unit if op == ">=" else (bound - value) / unit)
            score = _sigmoid(min(margins))
            if score >= 0.5:
                scores[anomaly_type] = score
                evidence.append(
                    f"{anomaly_type}: "
                    + ", ".join(
                        f"{name} {inputs[name]:+.1f}σ ({op} {bound:g})"
                        for name, op, bound in conditions
                    )
                )
        return scores, evidence

    def _prototype_types(
        self, deviation: dict[str, float], vector: Array
    ) -> tuple[dict[str, float], list[str]]:
        """Classify a region against the domain-adapted synthetic prototypes (open-set)."""
        posterior, best, distance, bound = self._prototype_posterior(vector)
        top = sorted(deviation.items(), key=lambda kv: -abs(kv[1]))[:4]
        summary = ", ".join(f"{k} {v:+.1f}σ" for k, v in top)
        if distance > bound:
            return {}, [
                f"no known type fits (closest: {best}, distance {distance:.1f} > open-set bound "
                f"{bound:.1f}); largest deviations: {summary}"
            ]
        scores = {t: p for t, p in posterior.items() if p >= 0.05}
        return scores, [
            f"{best}: matches this domain's synthetic {best} prototype "
            f"(posterior {posterior[best]:.2f}, distance {distance:.1f} within bound {bound:.1f})",
            f"largest deviations: {summary}",
        ]

    def learn_anomaly_types(self, examples: Sequence[tuple[Any, Array, str]]) -> float | None:
        """Teach domain-specific anomaly types from labelled examples.

        Each example is ``(image, mask, label)`` with a boolean ``mask`` marking
        the anomalous pixels. Regions are described by the same robust feature
        deviations the rule taxonomy uses and classified by nearest centroid in
        pooled-scale z space. Leave-one-out accuracy over the examples is
        returned (``None`` with fewer than two labels) and stored as
        :attr:`learned_type_loo_accuracy`.
        """
        self._require_fitted()
        vectors: list[Array] = []
        labels: list[str] = []
        for image, mask, label in examples:
            canonical = self._prepare(image)
            mask_arr = np.asarray(mask, dtype=bool)
            if mask_arr.shape != canonical.shape:
                mask_arr = (
                    ndimage.zoom(
                        mask_arr.astype(float),
                        (
                            canonical.shape[0] / mask_arr.shape[0],
                            canonical.shape[1] / mask_arr.shape[1],
                        ),
                        order=0,
                    )
                    > 0.5
                )
            self._raw_components(canonical)
            _, vector = self._region_deviation(mask_arr)
            vectors.append(vector)
            labels.append(label)
        matrix = np.stack(vectors)
        scale = np.maximum(matrix.std(axis=0), 1e-3)
        self._learned_scale = scale
        unique = sorted(set(labels))
        self._learned_types = {
            lab: matrix[[i for i, x in enumerate(labels) if x == lab]].mean(axis=0)
            for lab in unique
        }
        if len(unique) < 2:
            self.learned_type_loo_accuracy = None
            return None
        correct = 0
        for i, vec in enumerate(matrix):
            others = [j for j in range(len(labels)) if j != i]
            centroids = {}
            for lab in unique:
                members = [j for j in others if labels[j] == lab]
                if members:
                    centroids[lab] = matrix[members].mean(axis=0)
            predicted = min(
                centroids, key=lambda lab: float(np.sum(((vec - centroids[lab]) / scale) ** 2))
            )
            correct += predicted == labels[i]
        self.learned_type_loo_accuracy = correct / len(labels)
        return self.learned_type_loo_accuracy

    def _learned_prediction(self, vector: Array) -> tuple[str | None, float | None]:
        if not self._learned_types or self._learned_scale is None:
            return None, None
        names = sorted(self._learned_types)
        d2 = np.array(
            [
                float(np.sum(((vector - self._learned_types[n]) / self._learned_scale) ** 2))
                for n in names
            ]
        )
        weights = np.exp(-(d2 - d2.min()) / 2.0)
        probs = weights / weights.sum()
        best = int(np.argmax(probs))
        return names[best], float(probs[best])

    # ----------------------------------------------------------- localisation
    def _segment(self, score_map: Array, seed_level: float = 1.0) -> list[tuple[Array, float, int]]:
        """Segment a threshold-unit map into regions: ``(mask, peak score, flat peak index)``.

        Components are seeded where the map reaches ``seed_level`` (1.0 = the
        calibrated image threshold) and bounded by the calibrated pixel
        threshold; each region's extent is the half-maximum contour around its
        peak, so the smoothing halo of a strong defect — and normal pixels
        merely near it — stays out of the region.
        """
        cfg = self.config
        min_area = cfg.min_region_area or max(4, (cfg.patch_size * cfg.patch_size) // 4)
        pixel_thr = self._pixel_threshold_raw / self._image_threshold_raw
        labels, count = ndimage.label(score_map > pixel_thr)
        found: list[tuple[Array, float, int]] = []
        for index in range(1, count + 1):
            component = labels == index
            peak_score = float(score_map[component].max())
            if peak_score < seed_level:
                continue
            peak_index = int(np.argmax(np.where(component, score_map, -np.inf)))
            sub_labels, _ = ndimage.label(component & (score_map >= 0.5 * peak_score))
            mask = sub_labels == sub_labels.flat[peak_index]
            if int(mask.sum()) >= min_area:
                found.append((mask, peak_score, peak_index))
        return found

    def _regions(
        self,
        score_map: Array,
        image: CanonicalImage,
        forensics: dict[str, Any],
    ) -> list[AnomalyRegion]:
        regions: list[AnomalyRegion] = []
        height, width = score_map.shape
        for mask, peak_score, peak_flat in self._segment(score_map):
            area = int(mask.sum())
            ys, xs = np.nonzero(mask)
            py, px = divmod(peak_flat, width)
            deviation, vector = self._region_deviation(mask)
            if self._prototypes is not None:
                type_scores, evidence = self._prototype_types(deviation, vector)
            else:
                type_scores, evidence = self._rule_types(deviation)
            if image.invalid_mask[mask].any():
                type_scores["invalid_pixels"] = 1.0
                evidence.append(f"{int(image.invalid_mask[mask].sum())} non-finite input pixels")
            box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            for blk in forensics.get("noise_inconsistency", {}).get("flagged", []):
                overlap = _box_overlap(box, blk["box"])
                if overlap > 0.25:
                    type_scores["splice_candidate"] = max(
                        type_scores.get("splice_candidate", 0.0), _sigmoid(abs(blk["z"]) - 4.0)
                    )
                    evidence.append(f"noise level {blk['z']:+.1f}σ vs the rest of the image")
                    break
            for cluster in forensics.get("copy_move", {}).get("clusters", []):
                for key in ("source_box", "target_box"):
                    if _box_overlap(box, tuple(cluster[key])) > 0.25:
                        type_scores["copy_move_duplicate"] = 1.0
                        evidence.append(
                            f"duplicates the region at {tuple(cluster['source_box' if key == 'target_box' else 'target_box'])}"
                        )
                        break
            content = {k: v for k, v in type_scores.items() if k not in _FORENSIC_TYPES}
            # Content types only appear in ``content`` when they apply: rule
            # scores are pre-filtered at 0.5, and prototype posteriors are
            # empty when the open-set bound rejects — so the argmax is the
            # decision even when the posterior is shared between two
            # physically related types (a dark flat fill is both).
            if "invalid_pixels" in type_scores:
                anomaly_type = "invalid_pixels"
            elif content:
                anomaly_type = max(content, key=lambda k: content[k])
            elif "copy_move_duplicate" in type_scores:
                anomaly_type = "copy_move_duplicate"
            elif type_scores.get("splice_candidate", 0.0) >= 0.5:
                anomaly_type = "splice_candidate"
            else:
                anomaly_type = "unclassified_deviation"
            if anomaly_type == "unclassified_deviation":
                top = sorted(deviation.items(), key=lambda kv: -abs(kv[1]))[:3]
                evidence.append(
                    "largest deviations: " + ", ".join(f"{k} {v:+.1f}σ" for k, v in top)
                )
            learned, confidence = self._learned_prediction(vector)
            regions.append(
                AnomalyRegion(
                    box=box,
                    area=area,
                    area_fraction=area / float(height * width),
                    centroid=(float(xs.mean()), float(ys.mean())),
                    peak=(int(px), int(py)),
                    peak_score=peak_score,
                    mean_score=float(score_map[mask].mean()),
                    anomaly_type=anomaly_type,
                    type_scores={k: round(v, 4) for k, v in type_scores.items()},
                    evidence=evidence,
                    feature_deviation={k: round(v, 3) for k, v in deviation.items()},
                    learned_type=learned,
                    learned_confidence=confidence,
                )
            )
        regions.sort(key=lambda r: -r.peak_score)
        return regions

    # ---------------------------------------------------------------- analyse
    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "VisualAnomalyAnalyzer is not fitted: call fit(normal_images) or load(path) first"
            )

    def analyze(self, image: Any) -> VisualAnomalyReport:
        """Scrutinise one image against the learned normal model.

        Args:
            image: Any input :func:`~omni_mercury_engine.vision.imaging.to_canonical` accepts.

        Returns:
            The full report.
        """
        self._require_fitted()
        cfg = self.config
        timing: dict[str, float] = {}
        t0 = time.perf_counter()
        canonical = self._prepare(image)
        timing["prepare"] = (time.perf_counter() - t0) * 1e3

        t = time.perf_counter()
        fused, pixel_maps = self._fused_map(canonical)
        score_map = (fused / self._image_threshold_raw).astype(np.float32)
        image_score = float(score_map.max())
        raw_score = float(fused.max())
        timing["score_map"] = (time.perf_counter() - t) * 1e3

        t = time.perf_counter()
        forensics = self._forensic_stats(canonical)
        forensic_z: dict[str, float] = {}
        for key, value in self._forensic_vector(forensics).items():
            if key in self._forensic_norm:
                med, scale = self._forensic_norm[key]
                forensic_z[key] = (value - med) / scale
        forensics["z_vs_normal"] = forensic_z
        timing["forensics"] = (time.perf_counter() - t) * 1e3

        t = time.perf_counter()
        regions = self._regions(score_map, canonical, forensics)
        timing["localization"] = (time.perf_counter() - t) * 1e3

        environment = assess_environment(canonical)
        env_z_vec = (environment.vector() - self._env_median) / self._env_scale
        env_z = {
            name: round(float(v), 3) for name, v in zip(ENVIRONMENT_FIELDS, env_z_vec, strict=True)
        }
        out_of_envelope = [name for name, v in env_z.items() if abs(v) > cfg.environment_z_limit]
        in_envelope = not out_of_envelope

        n = self._calibration_scores.size
        p_value = float((1 + np.sum(self._calibration_scores >= raw_score)) / (n + 1))
        content_anomalous = p_value <= cfg.alpha
        forensic_flags = [f"{key} {z:+.1f}σ vs normal" for key, z in forensic_z.items() if z > 6.0]
        if forensics.get("copy_move", {}).get("detected"):
            forensic_flags.append("copy-move duplication detected")
        if (
            forensics.get("jpeg_grid", {}).get("grid_misaligned")
            and forensic_z.get("blockiness", 0.0) > 3.0
        ):
            forensic_flags.append("JPEG grid present at a non-zero phase (crop or recompression)")

        explanation: list[str] = []
        if not in_envelope and cfg.environment_policy == "withhold":
            state = ThreeState.UNAVAILABLE
            is_anomalous: bool | None = None
            reason = (
                "capture conditions outside the calibrated envelope ("
                + ", ".join(f"{k} {env_z[k]:+.1f}σ" for k in out_of_envelope)
                + "); content decision withheld — re-capture within normal conditions to decide"
            )
            explanation.append(reason)
        elif not self._calibration_sufficient:
            state = ThreeState.UNAVAILABLE
            is_anomalous = None
            reason = (
                f"{n} calibration images cannot certify alpha={cfg.alpha} "
                f"(the smallest reachable p-value is {1 / (n + 1):.3f}); "
                "add normal calibration images to decide"
            )
            explanation.append(reason)
        else:
            state = ThreeState.GROUNDED
            is_anomalous = bool(content_anomalous)
            reason = f"conformal p={p_value:.4f} vs alpha={cfg.alpha} over {n} calibration images"
        if content_anomalous:
            explanation.append(
                f"content score {image_score:.2f}× the calibrated threshold (p={p_value:.4f})"
            )
        for region in regions[:5]:
            explanation.append(
                f"{region.anomaly_type} at {region.box} (peak {region.peak_score:.2f}×): "
                + "; ".join(region.evidence[:2])
            )
        explanation.extend(forensic_flags)
        if not in_envelope:
            explanation.append(
                "unusual capture conditions: "
                + ", ".join(f"{k} {env_z[k]:+.1f}σ" for k in out_of_envelope)
            )

        deep_summary = None
        if "deep" in pixel_maps:
            deep_summary = {
                "name": self._deep_name,
                "max_raw": float(pixel_maps["deep"].max()),
                "fused": True,
            }
        timing["total"] = (time.perf_counter() - t0) * 1e3
        return VisualAnomalyReport(
            state=state,
            reason=reason,
            is_anomalous=is_anomalous,
            image_score=image_score,
            p_value=p_value,
            alpha=cfg.alpha,
            calibration_size=n,
            anomaly_map=score_map,
            regions=regions,
            forensics=forensics,
            environment=environment,
            environment_z=env_z,
            environment_in_envelope=in_envelope,
            provenance=canonical.provenance.to_dict(),
            explanation=explanation,
            timing_ms={k: round(v, 2) for k, v in timing.items()},
            manipulation_suspected=bool(forensic_flags),
            forensic_alerts=forensic_flags,
            deep_evidence=deep_summary,
        )

    def analyze_batch(self, images: Sequence[Any]) -> list[VisualAnomalyReport]:
        """Analyse many images; each report is independent of batch order."""
        return [self.analyze(image) for image in images]

    def score(self, image: Any) -> float:
        """Return the raw image-level content statistic that the conformal rule ranks.

        Cheaper than :meth:`analyze` (no localisation, forensics or report)
        and suited to batch ranking such as AUROC evaluation.
        """
        self._require_fitted()
        return float(self._fused_map(self._prepare(image))[0].max())

    # -------------------------------------------------------------- persistence
    def save(self, path: str | Path) -> Path:
        """Persist the fitted normal model to ``path`` (``.npz``, no pickle).

        Learned anomaly types are saved; a deep-evidence provider is not (it is
        code) and must be re-attached and re-fitted.
        """
        self._require_fitted()
        if self._deep is not None:
            raise RuntimeError("models with deep evidence cannot be saved; refit after loading")
        meta = {
            "schema": MODEL_SCHEMA,
            "config": self.config.to_dict(),
            "names": list(self._names),
            "base_names": list(self._base_names),
            "groups": list(self._groups),
            "component_norm": self._component_norm,
            "div_stats": self._div_stats,
            "forensic_norm": self._forensic_norm,
            "pixel_threshold_raw": self._pixel_threshold_raw,
            "image_threshold_raw": self._image_threshold_raw,
            "calibration_sufficient": self._calibration_sufficient,
            "lattice_periodicity": self._lattice_periodicity,
            "lattice_active": self._lattice_active,
            "n_train": self._n_train,
            "learned_labels": sorted(self._learned_types),
            "learned_type_loo_accuracy": self.learned_type_loo_accuracy,
        }
        arrays: dict[str, Array] = {
            "median": self._median,
            "scale": self._scale,
            "bank": self._bank,
            "gauss_mean": self._gauss_mean,
            "gauss_precision": self._gauss_precision,
            "texture_index": self._texture_index,
            "orientation_index": self._orientation_index,
            "texture_reference": self._texture_reference,
            "orientation_reference": self._orientation_reference,
            "calibration_scores": self._calibration_scores,
            "env_median": self._env_median,
            "env_scale": self._env_scale,
        }
        if self._positional is not None:
            arrays["positional_means"], arrays["positional_precisions"] = self._positional
        if self._learned_types and self._learned_scale is not None:
            arrays["learned_centroids"] = np.stack(
                [self._learned_types[k] for k in sorted(self._learned_types)]
            )
            arrays["learned_scale"] = self._learned_scale
        if self._prototypes:
            meta["prototype_types"] = list(self._prototypes["types"])
            meta["prototype_variants"] = list(self._prototypes["variants"])
            meta["prototype_samples"] = list(self._prototypes["samples_per_variant"])
            arrays["prototype_means"] = self._prototypes["means"]
            arrays["prototype_pooled_var"] = self._prototypes["pooled_var"]
            arrays["prototype_loo_q95"] = self._prototypes["loo_q95"]
        target = Path(path)
        if target.suffix != ".npz":
            target = target.with_suffix(".npz")
        payload: dict[str, Any] = {"meta": np.array(json.dumps(meta)), **arrays}
        np.savez_compressed(target, **payload)
        return target

    @classmethod
    def load(cls, path: str | Path) -> VisualAnomalyAnalyzer:
        """Load a model written by :meth:`save` (``allow_pickle=False``)."""
        with np.load(Path(path), allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            if meta.get("schema") != MODEL_SCHEMA:
                raise ValueError(f"unsupported visual model schema: {meta.get('schema')!r}")
            cfg = dict(meta["config"])
            cfg["scales"] = tuple(cfg["scales"])
            cfg["resize_to"] = None if cfg["resize_to"] is None else tuple(cfg["resize_to"])
            analyzer = cls(AnalyzerConfig(**cfg))
            analyzer._names = tuple(meta["names"])
            analyzer._base_names = tuple(meta["base_names"])
            analyzer._groups = tuple(meta["groups"])
            analyzer._median = data["median"]
            analyzer._scale = data["scale"]
            analyzer._bank = data["bank"]
            analyzer._gauss_mean = data["gauss_mean"]
            analyzer._gauss_precision = data["gauss_precision"]
            analyzer._texture_index = data["texture_index"]
            analyzer._orientation_index = data["orientation_index"]
            analyzer._texture_reference = data["texture_reference"]
            analyzer._orientation_reference = data["orientation_reference"]
            analyzer._calibration_scores = data["calibration_scores"]
            analyzer._env_median = data["env_median"]
            analyzer._env_scale = data["env_scale"]
            analyzer._positional = (
                (data["positional_means"], data["positional_precisions"])
                if "positional_means" in data
                else None
            )
            if "learned_centroids" in data:
                labels = meta["learned_labels"]
                analyzer._learned_types = dict(zip(labels, data["learned_centroids"], strict=True))
                analyzer._learned_scale = data["learned_scale"]
            analyzer._prototypes = None
            if "prototype_means" in data:
                analyzer._prototypes = {
                    "types": list(meta["prototype_types"]),
                    "variants": list(meta["prototype_variants"]),
                    "samples_per_variant": list(meta["prototype_samples"]),
                    "means": data["prototype_means"],
                    "pooled_var": data["prototype_pooled_var"],
                    "loo_q95": data["prototype_loo_q95"],
                }
        analyzer._component_norm = {
            k: (float(v[0]), float(v[1])) for k, v in meta["component_norm"].items()
        }
        analyzer._div_stats = {k: (float(v[0]), float(v[1])) for k, v in meta["div_stats"].items()}
        analyzer._forensic_norm = {
            k: (float(v[0]), float(v[1])) for k, v in meta["forensic_norm"].items()
        }
        analyzer._pixel_threshold_raw = float(meta["pixel_threshold_raw"])
        analyzer._image_threshold_raw = float(meta["image_threshold_raw"])
        analyzer._calibration_sufficient = bool(meta["calibration_sufficient"])
        analyzer._lattice_periodicity = meta["lattice_periodicity"]
        analyzer._lattice_active = bool(meta["lattice_active"])
        analyzer._skip_lattice = not analyzer._lattice_active
        analyzer._n_train = int(meta["n_train"])
        analyzer._n_calibration = int(analyzer._calibration_scores.size)
        analyzer.learned_type_loo_accuracy = meta.get("learned_type_loo_accuracy")
        analyzer._fitted = True
        return analyzer


def _box_overlap(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    """Return the intersection area over the smaller box's area."""
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / smaller if smaller > 0 else 0.0


def _box_iou(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class StreamEvent:
    """One processed frame of a :class:`StreamingVisualMonitor`.

    Attributes:
        frame_index: Zero-based frame counter.
        report: The frame's analysis.
        persistent_alert: True when an anomalous region has persisted (overlapping
            by IoU >= 0.3) for ``persistence`` consecutive frames.
        latency_ms: Wall time for this frame.
        latency_p50_ms: Rolling median latency.
        latency_p95_ms: Rolling 95th-percentile latency.
        over_budget: Whether this frame exceeded the latency budget.
    """

    frame_index: int
    report: VisualAnomalyReport
    persistent_alert: bool
    latency_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    over_budget: bool


@dataclass
class StreamingVisualMonitor:
    """Real-time wrapper: frame-to-frame persistence and latency accounting.

    A single noisy frame should not page anyone; a region that stays anomalous
    across ``persistence`` consecutive frames should. Latency is tracked over a
    rolling window and compared with ``latency_budget_ms``; when
    ``drop_forensics_over_budget`` is set, forensic checks are switched off
    after two consecutive over-budget frames (content scoring is kept).

    Attributes:
        analyzer: A fitted analyser.
        persistence: Consecutive frames a region must persist to alert.
        latency_budget_ms: Per-frame budget, or ``None``.
        drop_forensics_over_budget: Shed forensics when over budget.
        window: Frames kept for latency statistics.
    """

    analyzer: VisualAnomalyAnalyzer
    persistence: int = 2
    latency_budget_ms: float | None = None
    drop_forensics_over_budget: bool = True
    window: int = 120
    _frame: int = 0
    _history: deque[list[tuple[int, int, int, int]]] = field(default_factory=deque)
    _latencies: deque[float] = field(default_factory=deque)
    _over_budget_streak: int = 0
    forensics_shed: bool = False

    def process(self, frame: Any) -> StreamEvent:
        """Analyse one frame and update persistence and latency state."""
        start = time.perf_counter()
        report = self.analyzer.analyze(frame)
        latency = (time.perf_counter() - start) * 1e3
        self._latencies.append(latency)
        while len(self._latencies) > self.window:
            self._latencies.popleft()

        boxes = [r.box for r in report.regions] if report.is_anomalous else []
        self._history.append(boxes)
        while len(self._history) > self.persistence:
            self._history.popleft()
        persistent = False
        if len(self._history) == self.persistence and boxes:
            for box in boxes:
                if all(
                    any(_box_iou(box, other) >= 0.3 for other in frames)
                    for frames in list(self._history)[:-1]
                ):
                    persistent = True
                    break

        over = self.latency_budget_ms is not None and latency > self.latency_budget_ms
        self._over_budget_streak = self._over_budget_streak + 1 if over else 0
        if (
            self.drop_forensics_over_budget
            and self._over_budget_streak >= 2
            and self.analyzer.config.forensics
        ):
            self.analyzer.config.forensics = False
            self.forensics_shed = True

        latencies = np.array(self._latencies)
        event = StreamEvent(
            frame_index=self._frame,
            report=report,
            persistent_alert=persistent,
            latency_ms=latency,
            latency_p50_ms=float(np.percentile(latencies, 50)),
            latency_p95_ms=float(np.percentile(latencies, 95)),
            over_budget=bool(over),
        )
        self._frame += 1
        return event


__all__ = [
    "ANOMALY_TYPES",
    "MODEL_SCHEMA",
    "AnalyzerConfig",
    "AnomalyRegion",
    "StreamEvent",
    "StreamingVisualMonitor",
    "VisualAnomalyAnalyzer",
    "VisualAnomalyReport",
]

# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Advanced Biometric Processing Engine for Mercury Agent.

Provides enhanced biometric capabilities including:
- Neural-symbolic fusion for biometric matching
- Heuristic age progression (brightness/contrast/wrinkle-noise filters)
- Multi-modal biometric analysis (face, attributes, emotions)
- MZSS (Multi-Zone Similarity Score) computation

Key Features:
- Transformer-based neural-symbolic fusion (8-head attention)
- FaceNet/DeepFace integration; absent backends fail loudly, never silently
- Seeded, deterministic polynomial age filters (visual heuristics; no
  validated accuracy claim is made for them)
- Discrete Gaussian age-uncertainty kernel (legacy "quantum" naming)

References:
    - FaceNet: A Unified Embedding for Face Recognition (Schroff et al., 2015)
    - DeepFace: Closing the Gap to Human-Level Performance (Taigman et al., 2014)
    - Age Progression/Regression by Conditional Adversarial Autoencoder (Zhang et al., 2017)

Designed for humanitarian missing persons applications.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from omni_mercury_engine._compat import preload_triton_before_tensorflow

logger = logging.getLogger(__name__)

TORCH_AVAILABLE = False
try:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not available, advanced biometric features limited")

# Optional-dep imports below catch ``Exception`` (not just
# ``ImportError``) because each of these packages can fail at import
# time with non-``ImportError`` types — e.g. ``deepface`` transitively
# imports ``retinaface`` which raises ``ValueError`` if ``tf-keras``
# is missing on tensorflow >=2.16; ``facenet_pytorch`` and ``cv2``
# raise ``OSError`` / ``RuntimeError`` on missing native libraries
# or CUDA stack mismatches; ``face_recognition`` raises
# ``RuntimeError`` if its dlib backend was built against a different
# numpy ABI.  Letting any of those escape would break callers that
# legitimately want to discover the optional dep is unusable on this
# host and fall back to simulated embeddings.  ``BaseException`` is
# deliberately not caught (``KeyboardInterrupt`` / ``SystemExit``
# keep propagating).
DEEPFACE_AVAILABLE = False

# deepface pulls TensorFlow into the process; triton (torch's compiler
# backend) segfaults if it loads after TensorFlow's LLVM, so bind it first.
preload_triton_before_tensorflow()

try:
    from deepface import DeepFace

    DEEPFACE_AVAILABLE = True
except Exception as _deepface_exc:
    logger.debug(
        "DeepFace not available (%s: %s), using fallback embeddings",
        type(_deepface_exc).__name__,
        _deepface_exc,
    )

FACE_RECOGNITION_AVAILABLE = False
try:
    import face_recognition

    FACE_RECOGNITION_AVAILABLE = True
except Exception as _face_recognition_exc:
    logger.debug(
        "face_recognition not available (%s: %s)",
        type(_face_recognition_exc).__name__,
        _face_recognition_exc,
    )

# OpenCV is a core dependency (pyproject ``[project] dependencies``), not an
# optional extra, so this is a hard import. It was optional, and the face path
# paid for it twice: ``apply_polynomial_age_filter`` returned its input
# unchanged, and ``detect_and_align_face`` reported "Face detection failed"
# when the truth was that the decoder was missing. OpenCV supplies the
# decoding, colour conversion, resize and blur this module needs, and -- on
# the 4.x line -- the Haar cascade that detects a face without PyTorch.
import cv2

FACENET_AVAILABLE = False
try:
    from facenet_pytorch import MTCNN, InceptionResnetV1

    FACENET_AVAILABLE = True
except Exception:
    logger.debug("FaceNet-PyTorch not available")


class MatchCategory(Enum):
    """Biometric match categories based on MZSS score."""

    PRIMARY = "primary"
    SECONDARY = "secondary"
    UNDETERMINED = "undetermined"


@dataclass
class BiometricResult:
    """Result from biometric analysis."""

    success: bool
    embedding: np.ndarray[Any, Any] | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    similarity: float = 0.0
    mzss_score: float = 0.0
    match_category: MatchCategory = MatchCategory.UNDETERMINED
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgeProgressionResult:
    """Result from age progression."""

    success: bool
    original_face: np.ndarray[Any, Any] | None = None
    progressed_face: np.ndarray[Any, Any] | None = None
    age_delta: int = 0
    # Embedding similarity between original and progressed face; None when no
    # real embedding backend was available (never a fabricated number).
    similarity: float | None = None
    original_embedding: np.ndarray[Any, Any] | None = None
    progressed_embedding: np.ndarray[Any, Any] | None = None
    quantum_factor: float = 1.2
    message: str = ""
    # How the progression was produced: the geometric growth model actually
    # applied (or why none was), and the texture model used.
    geometric_model: dict[str, Any] | None = None
    texture_model: str | None = None


#: Time constant (years) of the assumed craniofacial growth schedule
#: ``g(age) = 1 - exp(-age / tau)``: 95% of facial growth complete by ~18.
#: A stated modelling assumption, not a fit to anthropometric data.
CRANIOFACIAL_GROWTH_TAU_YEARS = 6.0

#: Cardioidal strain parameter spanning infancy to adulthood under the schedule.
CRANIOFACIAL_K_RANGE = 0.25


def craniofacial_growth_fraction(age: float) -> float:
    """Fraction of craniofacial growth completed by ``age`` under the stated schedule."""
    return float(1.0 - np.exp(-max(age, 0.0) / CRANIOFACIAL_GROWTH_TAU_YEARS))


def cardioidal_strain(
    face: np.ndarray[Any, Any],
    k: float,
    origin: tuple[float, float] = (0.42, 0.5),
) -> np.ndarray[Any, Any]:
    """Apply the cardioidal-strain growth transform to an aligned face crop.

    The revised cardioidal strain (Todd, Mark, Shaw & Pittenger, 1980; Mark &
    Todd, 1983) models craniofacial growth as a radial deformation about a
    point near the centre of the head: in polar coordinates ``(R, theta)``
    with ``theta`` measured from the upward vertical, ``R' = R (1 + k (1 -
    cos theta))``. The crown (theta = 0) does not move while the lower face
    (theta near pi) elongates by up to ``1 + 2k`` — the characteristic change
    from a child's to an adult's facial proportions. ``k > 0`` ages the face
    forward, ``k < 0`` rejuvenates it.

    Args:
        face: ``(H, W)`` or ``(H, W, C)`` aligned face crop.
        k: Strain parameter (the difference between target and source levels).
        origin: Strain origin as ``(y, x)`` fractions of the crop (default: on
            the vertical midline, 42% from the top — the head centre in a
            standard face crop).

    Returns:
        The warped face, same shape and dtype.
    """
    from scipy import ndimage

    image = np.asarray(face)
    height, width = image.shape[:2]
    oy, ox = origin[0] * height, origin[1] * width
    yy, xx = np.indices((height, width), dtype=np.float64)
    dy, dx = yy - oy, xx - ox
    radius = np.hypot(dy, dx)
    cos_theta = np.where(radius > 0, -dy / np.maximum(radius, 1e-9), 1.0)
    # Inverse map: an output pixel at radius R' samples the source at R' / s.
    stretch = 1.0 + k * (1.0 - cos_theta)
    src_y = oy + dy / stretch
    src_x = ox + dx / stretch
    work = image.astype(np.float64)
    # Cubic resampling: bilinear interpolation acts as a low-pass filter and
    # visibly softens skin detail (measured by assess_progression's
    # sharpness ratio); ringing overshoot is clipped to the input's range.
    if work.ndim == 2:
        warped = ndimage.map_coordinates(work, [src_y, src_x], order=3, mode="nearest")
    else:
        warped = np.stack(
            [
                ndimage.map_coordinates(work[..., c], [src_y, src_x], order=3, mode="nearest")
                for c in range(work.shape[2])
            ],
            axis=-1,
        )
    warped = np.clip(warped, float(work.min()), float(work.max()))
    if np.issubdtype(image.dtype, np.integer):
        return np.round(warped).astype(image.dtype)
    return warped.astype(image.dtype)


def structural_similarity(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any]) -> float:
    """Mean SSIM (Wang, Bovik, Sheikh & Simoncelli, 2004) of two images' luminance.

    Gaussian-windowed (sigma 1.5) with the standard constants for a ``[0, 1]``
    dynamic range; colour images are compared on Rec. 709 luma.
    """
    from scipy import ndimage

    def _luma(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        arr = np.asarray(x, dtype=np.float64)
        if np.issubdtype(np.asarray(x).dtype, np.integer):
            arr = arr / 255.0
        if arr.ndim == 3 and arr.shape[2] >= 3:
            arr = arr[..., :3] @ np.array([0.2126, 0.7152, 0.0722])
        elif arr.ndim == 3:
            arr = arr[..., 0]
        return arr

    x, y = _luma(a), _luma(b)
    c1, c2 = 0.01**2, 0.03**2
    mu_x = ndimage.gaussian_filter(x, 1.5)
    mu_y = ndimage.gaussian_filter(y, 1.5)
    sxx = ndimage.gaussian_filter(x * x, 1.5) - mu_x**2
    syy = ndimage.gaussian_filter(y * y, 1.5) - mu_y**2
    sxy = ndimage.gaussian_filter(x * y, 1.5) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (sxx + syy + c2)
    )
    return float(ssim_map.mean())


class BiometricFusion:
    """Transformer-based neural-symbolic fusion for biometric matching.

    Uses 8-head self-attention to combine neural embeddings with symbolic constraint satisfaction
    scores.
    """

    def __init__(self, dim: int = 512, device: str = "cpu") -> None:
        """Initialize the instance."""
        self.dim = dim
        self.device = device

        if TORCH_AVAILABLE:
            self.transformer = nn.TransformerEncoderLayer(d_model=dim, nhead=8, batch_first=True)
            self.transformer = self.transformer.to(device)
        else:
            self.transformer = None  # type: ignore[assignment, unused-ignore]

    def forward(
        self, neural_emb: np.ndarray[Any, Any], symbolic_score: float = 1.0
    ) -> np.ndarray[Any, Any]:
        """Fuse neural embedding with symbolic constraint score.

        Args:
            neural_emb: Neural embedding vector
            symbolic_score: Symbolic constraint satisfaction (0-1)

        Returns:
            Fused embedding
        """
        if not TORCH_AVAILABLE or self.transformer is None:
            return neural_emb * symbolic_score

        neural_emb = np.asarray(neural_emb, dtype=np.float32)

        if neural_emb.ndim == 1:
            neural_emb = neural_emb.reshape(1, -1)

        current_dim = neural_emb.shape[-1]
        if current_dim != self.dim:
            if current_dim < self.dim:
                padding = np.zeros((neural_emb.shape[0], self.dim - current_dim), dtype=np.float32)
                neural_emb = np.concatenate([neural_emb, padding], axis=1)
            else:
                neural_emb = neural_emb[:, : self.dim]

        tensor = torch.from_numpy(neural_emb).to(self.device)

        with torch.no_grad():
            fused = self.transformer(tensor)

        fused = fused * symbolic_score

        return fused.cpu().numpy()


class MultimodalBiometricEngine:
    """Advanced biometric processing engine with neural-symbolic fusion.

    Provides face detection, feature extraction, attribute analysis,
    and multi-zone similarity scoring for missing persons applications.

    Example:
        engine = MultimodalBiometricEngine()
        result = engine.analyze_image("photo.jpg")
        print(f"Age: {result.attributes.get('age')}")
    """

    def __init__(self, device: str = "cpu", seed: int | None = None) -> None:
        """Initialize advanced biometric engine.

        Args:
            device: Torch device string ("cpu" / "cuda")
            seed: Optional seed for the per-instance ``Generator`` driving
                the simulated-embedding fallbacks used when DeepFace /
                face_recognition are unavailable. ``None`` (default) uses
                an OS-seeded ``Generator`` — same effective behavior as
                before.
        """
        self.device = device
        self.fusion_model = BiometricFusion(dim=512, device=device)

        self.mzss_alpha = 0.5
        self.mzss_beta = 0.3
        self.mzss_gamma = 0.2

        self.target_embedding_size = 128
        self._rng: np.random.Generator = np.random.default_rng(seed)

        logger.info(
            f"MultimodalBiometricEngine initialized (device={device}, "
            f"deepface={DEEPFACE_AVAILABLE}, facenet={FACENET_AVAILABLE})"
        )

    def extract_features(self, image_path: str) -> np.ndarray[Any, Any] | None:
        """Extract facial features from image.

        Uses DeepFace (FaceNet) as primary, face_recognition as fallback.

        Args:
            image_path: Path to image file

        Returns:
            Feature embedding or None if extraction fails
        """
        try:
            if DEEPFACE_AVAILABLE:
                embedding = DeepFace.represent(
                    img_path=image_path,
                    model_name="Facenet",
                    enforce_detection=False,
                )
                if isinstance(embedding, list) and len(embedding) > 0:
                    return np.array(embedding[0]["embedding"], dtype=np.float32)

            if FACE_RECOGNITION_AVAILABLE:
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    return np.array(encodings[0], dtype=np.float32)

            # No embedding backend -> None. process_biometric already handles
            # None as an honest extraction failure; a random stand-in made the
            # downstream "similarity" a comparison of noise with noise.
            logger.warning("No embedding backend available - features not extracted")
            return None

        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return None

    def extract_features_from_array(
        self, image_data: np.ndarray[Any, Any]
    ) -> np.ndarray[Any, Any] | None:
        """Extract features from numpy array image data.

        Args:
            image_data: Image as numpy array

        Returns:
            Feature embedding or None
        """
        try:
            if DEEPFACE_AVAILABLE:
                embedding = DeepFace.represent(
                    img_path=image_data,
                    model_name="Facenet",
                    enforce_detection=False,
                )
                if isinstance(embedding, list) and len(embedding) > 0:
                    return np.array(embedding[0]["embedding"], dtype=np.float32)

            logger.warning("No embedding backend available for array input")
            return None

        except Exception as e:
            logger.error(f"Array feature extraction error: {e}")
            return None

    def analyze_attributes(self, image_path: str) -> dict[str, Any]:
        """Analyze facial attributes (age, gender, emotion).

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with age, gender, emotion analysis
        """
        try:
            if DEEPFACE_AVAILABLE:
                analysis = DeepFace.analyze(
                    img_path=image_path,
                    actions=["age", "gender", "emotion"],
                    enforce_detection=False,
                )
                if isinstance(analysis, list) and len(analysis) > 0:
                    result: dict[str, Any] = analysis[0]
                    result["measured"] = True
                    return result

            # No backend -> no attributes. Fabricated placeholders (age 30,
            # 50/50 gender) used to flow into age-proximity and inflate the
            # MZSS match score; an unmeasured record is reported instead.
            logger.warning("Attributes not measured (DeepFace not available)")
            return {"measured": False, "reason": "DeepFace not available"}

        except Exception as e:
            logger.error(f"Attribute analysis error: {e}")
            return {}

    def compute_mzss(
        self,
        biometric_match: float,
        symbolic_match: float = 0.8,
        age_proximity: float = 0.9,
    ) -> float:
        """Compute Multi-Zone Similarity Score (MZSS).

        MZSS = α*B + β*S + γ*A
        where B=biometric, S=symbolic, A=age proximity

        Args:
            biometric_match: Biometric similarity (0-1)
            symbolic_match: Symbolic constraint satisfaction (0-1)
            age_proximity: Age proximity score (0-1)

        Returns:
            MZSS score (0-1)
        """
        mzss = (
            self.mzss_alpha * biometric_match
            + self.mzss_beta * symbolic_match
            + self.mzss_gamma * age_proximity
        )
        return float(np.clip(mzss, 0.0, 1.0))

    def categorize_match(self, mzss: float) -> MatchCategory:
        """Categorize match based on MZSS score.

        Args:
            mzss: Multi-Zone Similarity Score

        Returns:
            MatchCategory (PRIMARY, SECONDARY, or UNDETERMINED)
        """
        if mzss >= 0.90:
            return MatchCategory.PRIMARY
        elif mzss >= 0.85:
            return MatchCategory.SECONDARY
        return MatchCategory.UNDETERMINED

    def match_faces(self, image1_path: str, image2_path: str) -> BiometricResult:
        """Match two faces and compute similarity scores.

        Args:
            image1_path: Path to first image
            image2_path: Path to second image

        Returns:
            BiometricResult with match scores
        """
        features1 = self.extract_features(image1_path)
        features2 = self.extract_features(image2_path)

        if features1 is None or features2 is None:
            return BiometricResult(
                success=False,
                metadata={"message": "Feature extraction failed"},
            )

        similarity = self._compute_similarity(features1, features2)

        analysis1 = self.analyze_attributes(image1_path)
        analysis2 = self.analyze_attributes(image2_path)

        # Age proximity is a measurement: compute it only when both ages were
        # actually measured; otherwise use the documented neutral default and
        # say so, instead of deriving perfect proximity from fabricated ages.
        age1 = analysis1.get("age")
        age2 = analysis2.get("age")
        if isinstance(age1, (int, float)) and isinstance(age2, (int, float)):
            ages_measured = True
            age_proximity = self._compute_age_proximity(float(age1), float(age2))
        else:
            ages_measured = False
            age_proximity = 0.9  # compute_mzss's documented neutral default

        mzss = self.compute_mzss(similarity, 0.8, age_proximity)
        category = self.categorize_match(mzss)

        return BiometricResult(
            success=True,
            embedding=features1,
            attributes={"image1": analysis1, "image2": analysis2},
            similarity=similarity,
            mzss_score=mzss,
            match_category=category,
            metadata={
                "age_proximity": age_proximity,
                "age_proximity_measured": ages_measured,
                "biometric_similarity": similarity,
            },
        )

    def fuse_with_symbolic(
        self, image_path: str, symbolic_data: dict[str, Any]
    ) -> np.ndarray[Any, Any]:
        """Fuse biometric features with symbolic constraint data.

        Args:
            image_path: Path to image
            symbolic_data: Dictionary with symbolic scores (e.g., 'csdm')

        Returns:
            Fused feature embedding
        """
        features = self.extract_features(image_path)
        if features is None:
            raise ValueError(
                f"feature extraction failed for {image_path!r}; refusing to fuse "
                "random stand-in features with symbolic data"
            )

        symbolic_score = symbolic_data.get("csdm", 0.8)
        fused = self.fusion_model.forward(features, symbolic_score)

        return fused

    def _compute_similarity(
        self, features1: np.ndarray[Any, Any], features2: np.ndarray[Any, Any]
    ) -> float:
        """Compute cosine similarity between feature vectors."""
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        features1_norm = features1 / norm1
        features2_norm = features2 / norm2

        similarity = np.dot(features1_norm, features2_norm)
        return float((similarity + 1) / 2)

    def _compute_age_proximity(self, age1: float, age2: float) -> float:
        """Compute age proximity score."""
        age_diff = abs(age1 - age2)
        proximity = max(0.0, 1.0 - (age_diff / 50.0))
        return float(proximity)


class FaceDetectorUnavailableError(RuntimeError):
    """No face detector is installed, so no image was ever examined.

    Distinct from ``detect_and_align_face`` returning ``None``, which means a
    detector ran and found no face. Keeping the two apart is what stops
    "Face detection failed" from being reported for a missing dependency --
    the misdirection the removed ``CV2_AVAILABLE`` guard used to produce.
    """


def face_detector_requirement() -> str:
    """Why no face detector is available, and what supplies one.

    Mercury has two: MTCNN from ``facenet_pytorch`` (needs the ``[ml]`` extra
    for PyTorch), and OpenCV's Haar cascade. OpenCV 5.0 moved
    ``CascadeClassifier`` and the bundled ``cv2.data.haarcascades`` into
    ``opencv_contrib``, so the ``-headless`` 5.x wheel the core pin admits
    carries neither -- a core-only install on 5.x therefore has *no* detector,
    and the pin's ``<6`` ceiling is deliberate (see pyproject). Saying which
    one to install beats returning ``None`` and letting the caller report "no
    face in the image".
    """
    cascade = getattr(cv2, "CascadeClassifier", None) is not None
    bundled = getattr(getattr(cv2, "data", None), "haarcascades", None) is not None
    return (
        "no face detector is available in this install: MTCNN needs PyTorch "
        '(pip install "mercury-agent-global[ml]"), and this OpenCV '
        f"({cv2.__version__}) provides CascadeClassifier={cascade}, bundled "
        f"cascades={bundled} -- OpenCV 5.x moved both into opencv_contrib. "
        "Install the [ml] extra, or supply the cascade XMLs (see "
        "docs/INSTALLATION.md)."
    )


def _gaussian_blur_5x5(image: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Blur with OpenCV's 5x5 ``sigma=0`` kernel (sigma 1.1, derived by OpenCV).

    One implementation, because OpenCV is core. The previous arrangement --
    skip the blur, and with it the whole filter, when OpenCV was absent -- made
    an unprocessed face pass for an age-progressed one. A SciPy stand-in would
    cure the silence but leave the pixels an operator gets depending on which
    wheels they installed, which is the wrong property for a face a search may
    be run against.
    """
    work = np.clip(image, 0, 255).astype(np.uint8)
    return np.asarray(cv2.GaussianBlur(work, (5, 5), 0)).astype(float)


class AgeProgressionEngine:
    """Age-progression engine: published growth geometry, deterministic texture, audited output.

    * **Geometry** — for juvenile age spans with a known source age, the
      cardioidal-strain craniofacial growth model (:func:`cardioidal_strain`)
      reshapes the face as it grows; adults get no geometric change.
    * **Texture** — the default ``"detail"`` model amplifies (aging) or
      attenuates (rejuvenation) the face's own skin-fold band
      deterministically; the seeded ``"legacy_noise"`` brightness/contrast/
      noise filter (:meth:`apply_polynomial_age_filter`) remains available.
    * **Audit** — :meth:`assess_progression` measures structural similarity,
      injected noise and lost detail, and can run the visual anomaly analyser
      on the output.

    The growth schedule and texture model are stated heuristics: no validated
    accuracy claim is made for the progression itself, and the reported
    embedding similarity is ``None`` whenever no real embedding backend is
    available.

    Example:
        engine = AgeProgressionEngine()
        result = engine.progress_age("photo.jpg", target_age_delta=10)
        if result.success:
            print(f"Similarity: {result.similarity}")
    """

    def __init__(self, device: str = "cpu", seed: int | None = None) -> None:
        """Initialize age-progression engine.

        Args:
            device: Torch device string ("cpu" / "cuda")
            seed: Optional seed for the per-instance ``Generator`` driving
                the simulated FaceNet-embedding fallback and the
                wrinkle / noise perturbations applied during age
                progression. ``None`` (default) uses an OS-seeded
                ``Generator`` — same effective behavior as before.
        """
        self.device = device
        self.quantum_factor = 1.2
        self.golden_ratio = 0.618
        self._rng: np.random.Generator = np.random.default_rng(seed)

        if FACENET_AVAILABLE:
            self.face_detector = MTCNN(device=device)
            self.facenet = InceptionResnetV1(pretrained="vggface2").eval()
            if TORCH_AVAILABLE:
                self.facenet = self.facenet.to(device)
        else:
            self.face_detector = None
            self.facenet = None

        logger.info(
            f"AgeProgressionEngine initialized (device={device}, " f"facenet={FACENET_AVAILABLE})"
        )

    def detect_and_align_face(self, image_path: str) -> np.ndarray[Any, Any] | None:
        """Detect and align face from image.

        Args:
            image_path: Path to image file

        Returns:
            Aligned face as numpy array (160x160) or None
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self.face_detector is not None and TORCH_AVAILABLE:
                boxes, _ = self.face_detector.detect(image_rgb)
                if boxes is not None and len(boxes) > 0:
                    box = boxes[0].astype(int)
                    x1, y1, x2, y2 = box
                    face = image_rgb[y1:y2, x1:x2]
                    face = cv2.resize(face, (160, 160))
                    return face

            # Haar-cascade fallback. OpenCV 5.0 relocated ``CascadeClassifier``
            # and the bundled ``cv2.data.haarcascades`` into ``opencv_contrib``,
            # so they are absent from the ``-headless`` 5.x wheel. Feature-detect
            # both via ``getattr`` (this also keeps the blocking mypy gate green
            # across the 4.x and 5.x stubs, which differ on these symbols) and
            # skip the fallback cleanly when unavailable — the DNN detector above
            # stays the primary path.
            cascade_factory = getattr(cv2, "CascadeClassifier", None)
            # ``cv2.data`` can exist while ``cv2.data.haarcascades`` does not
            # (the 5.x headless/contrib split), so feature-detect the bundled
            # cascade directory itself, not just the ``data`` module.
            haarcascades_dir = getattr(getattr(cv2, "data", None), "haarcascades", None)
            if cascade_factory is not None and haarcascades_dir is not None:
                # ``os.path.join`` normalizes the separator: ``haarcascades_dir``
                # is not guaranteed to carry a trailing slash across builds.
                face_cascade = cascade_factory(
                    os.path.join(haarcascades_dir, "haarcascade_frontalface_default.xml")
                )
                # A missing/unreadable cascade XML yields an empty classifier
                # rather than raising; run the fallback only when it loaded.
                if not face_cascade.empty():
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                    if len(faces) > 0:
                        x, y, w, h = faces[0]
                        face = image_rgb[y : y + h, x : x + w]
                        face = cv2.resize(face, (160, 160))
                        return face
                    return None

            # Neither detector exists in this install, so nothing looked at the
            # image. Returning ``None`` here would be read by ``progress_age``
            # as "Face detection failed" -- the same wrong reason the removed
            # ``CV2_AVAILABLE`` guard used to give. ``None`` is reserved for
            # "a detector ran and found no face"; this is a configuration
            # statement, and ``FaceDetectorUnavailableError`` carries it past
            # the handler below so the caller reports the real cause.
            if self.face_detector is None or not TORCH_AVAILABLE:
                raise FaceDetectorUnavailableError(face_detector_requirement())

            return None

        except FaceDetectorUnavailableError:
            # A configuration fact, not a detection outcome: let it through.
            raise
        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return None

    def extract_facenet_embedding(self, face: np.ndarray[Any, Any]) -> np.ndarray[Any, Any] | None:
        """Extract 512-dimensional FaceNet embedding.

        Args:
            face: Aligned face image (160x160)

        Returns:
            512-dim FaceNet embedding, or ``None`` when no real embedding
            backend is available — a random stand-in would make every
            downstream similarity a fabricated number.
        """
        try:
            if self.facenet is None or not TORCH_AVAILABLE:
                return None

            face_tensor = torch.from_numpy(face).permute(2, 0, 1).float() / 255.0
            face_tensor = face_tensor.unsqueeze(0).to(self.device)

            with torch.no_grad():
                embedding = self.facenet(face_tensor)

            return embedding.cpu().numpy().flatten()

        except Exception as e:
            logger.error(f"Embedding extraction error: {e}")
            return None

    def apply_polynomial_age_filter(
        self, face: np.ndarray[Any, Any], age_delta: int
    ) -> np.ndarray[Any, Any]:
        """Apply seeded polynomial brightness/contrast/noise filters.

        A visual heuristic (no validated accuracy claim): aging darkens and
        sharpens contrast and adds wrinkle-scale noise; rejuvenation brightens,
        softens contrast and smooths.

        Args:
            face: Input face image
            age_delta: Years to progress (+forward, -backward)

        Returns:
            Age-progressed face image
        """
        aged = face.astype(float)

        if age_delta > 0:
            brightness_adj = 1.0 - (0.02 * age_delta)
            contrast_adj = 1.0 + (0.01 * age_delta)

            aged = aged * brightness_adj
            aged = np.clip((aged - 127.5) * contrast_adj + 127.5, 0, 255)

            wrinkle_intensity = min(age_delta * 0.5, 20)
            noise = self._rng.standard_normal(face.shape) * wrinkle_intensity
            aged = aged + noise

        else:
            age_delta_abs = abs(age_delta)
            brightness_adj = 1.0 + (0.02 * age_delta_abs)
            contrast_adj = 1.0 - (0.01 * age_delta_abs)

            aged = aged * brightness_adj
            aged = np.clip((aged - 127.5) * contrast_adj + 127.5, 0, 255)

            aged = _gaussian_blur_5x5(aged)

        # quantum_factor is a legacy-named seeded-noise amplitude.
        seeded_noise = self._rng.standard_normal(face.shape) * self.quantum_factor
        aged = aged + seeded_noise

        return np.clip(aged, 0, 255).astype(np.uint8)

    def apply_detail_age_filter(
        self, face: np.ndarray[Any, Any], age_delta: int
    ) -> np.ndarray[Any, Any]:
        """Deterministic skin-texture aging that works on detail the face already has.

        Aging amplifies the face's own *mid-frequency* band (difference of
        Gaussians, sigma 1 to 3 — the scale of skin folds and creases), with a
        gain growing with the years and capped, and applies a mild tonal
        change; rejuvenation attenuates that band and slightly lifts tone. The
        finest band, where sensor noise lives, is left untouched, and no noise
        is added — the same face always yields the same result and nothing
        synthetic is passed off as wrinkles. A visual heuristic: no validated
        accuracy claim is made for it.

        Args:
            face: Aligned face (``uint8`` or float in ``[0, 1]``).
            age_delta: Years to progress (+forward, -backward).

        Returns:
            The progressed face, same dtype.
        """
        from scipy import ndimage

        image = np.asarray(face)
        scale = 255.0 if np.issubdtype(image.dtype, np.integer) else 1.0
        work = image.astype(np.float64) / scale

        def _blur(sigma: float) -> np.ndarray[Any, Any]:
            return np.asarray(
                ndimage.gaussian_filter(work, (sigma, sigma, 0) if work.ndim == 3 else sigma)
            )

        band = _blur(1.0) - _blur(3.0)
        years = float(abs(age_delta))
        if age_delta > 0:
            gain = 1.0 + min(0.06 * years, 1.5)
            tone = 1.0 - min(0.008 * years, 0.2)
        else:
            gain = max(1.0 - 0.06 * years, 0.2)
            tone = 1.0 + min(0.006 * years, 0.12)
        out = np.clip((work + (gain - 1.0) * band) * tone, 0.0, 1.0)
        if scale != 1.0:
            return np.round(out * scale).astype(image.dtype)
        return out.astype(image.dtype)

    def progress_face(
        self,
        face: np.ndarray[Any, Any],
        target_age_delta: int,
        source_age: float | None = None,
        texture_model: str = "detail",
    ) -> AgeProgressionResult:
        """Progress an already-aligned face crop by ``target_age_delta`` years.

        Two stages, each reported in the result:

        * **Geometry** — when ``source_age`` is given and the age span overlaps
          the growth years, the cardioidal-strain craniofacial growth model is
          applied with ``k = K_RANGE * (g(target) - g(source))`` under the
          stated growth schedule (:func:`craniofacial_growth_fraction`). Adult
          faces get no geometric change (the model's growth has saturated);
          without ``source_age`` none is applied and the result says why.
        * **Texture** — ``"detail"`` (deterministic, default) or
          ``"legacy_noise"`` (the seeded brightness/contrast/noise filter).

        Args:
            face: Aligned face crop (``uint8`` or float in ``[0, 1]``).
            target_age_delta: Years to progress (+forward, -backward).
            source_age: Age of the person in ``face``, if known.
            texture_model: ``"detail"`` or ``"legacy_noise"``.

        Returns:
            The progression result with its geometric and texture provenance.
        """
        if texture_model not in ("detail", "legacy_noise"):
            raise ValueError("texture_model must be 'detail' or 'legacy_noise'")
        geometric: dict[str, Any]
        progressed = np.asarray(face)
        if source_age is None:
            geometric = {"applied": False, "reason": "source_age not provided"}
        else:
            target_age = max(source_age + target_age_delta, 0.0)
            k = CRANIOFACIAL_K_RANGE * (
                craniofacial_growth_fraction(target_age) - craniofacial_growth_fraction(source_age)
            )
            if abs(k) < 1e-3:
                geometric = {
                    "applied": False,
                    "reason": "no craniofacial growth across this age span under the model",
                    "k": k,
                }
            else:
                progressed = cardioidal_strain(progressed, k)
                geometric = {
                    "applied": True,
                    "model": "cardioidal strain (Todd, Mark, Shaw & Pittenger, 1980)",
                    "k": round(k, 5),
                    "source_age": source_age,
                    "target_age": target_age,
                    "schedule": f"g(age) = 1 - exp(-age / {CRANIOFACIAL_GROWTH_TAU_YEARS:g}) "
                    "(stated assumption, not fitted to anthropometric data)",
                }
        if texture_model == "detail":
            progressed = self.apply_detail_age_filter(progressed, target_age_delta)
        else:
            progressed = self.apply_polynomial_age_filter(progressed, target_age_delta)

        original_embedding = self.extract_facenet_embedding(np.asarray(face))
        progressed_embedding = self.extract_facenet_embedding(progressed)
        # Similarity is a measurement: report it only when both embeddings
        # came from a real backend.
        similarity = (
            self._compute_similarity(original_embedding, progressed_embedding)
            if original_embedding is not None and progressed_embedding is not None
            else None
        )
        return AgeProgressionResult(
            success=True,
            original_face=np.asarray(face),
            progressed_face=progressed,
            age_delta=target_age_delta,
            similarity=similarity,
            original_embedding=original_embedding,
            progressed_embedding=progressed_embedding,
            quantum_factor=self.quantum_factor,
            geometric_model=geometric,
            texture_model=texture_model,
        )

    def progress_age(
        self,
        image_path: str,
        target_age_delta: int,
        source_age: float | None = None,
        texture_model: str = "detail",
    ) -> AgeProgressionResult:
        """Detect and align the face in ``image_path``, then :meth:`progress_face` it.

        Args:
            image_path: Path to input image
            target_age_delta: Years to progress (+forward, -backward)
            source_age: Age of the person in the image, if known (enables the
                craniofacial growth model for juvenile spans).
            texture_model: ``"detail"`` (default) or ``"legacy_noise"``.

        Returns:
            AgeProgressionResult with progressed image and metadata
        """
        try:
            face = self.detect_and_align_face(image_path)
            if face is None:
                return AgeProgressionResult(success=False, message="Face detection failed")
            return self.progress_face(
                face, target_age_delta, source_age=source_age, texture_model=texture_model
            )
        except Exception as e:
            logger.error(f"Age progression error: {e}")
            return AgeProgressionResult(success=False, message=str(e))

    @staticmethod
    def assess_progression(
        result: AgeProgressionResult, analyzer: Any | None = None
    ) -> dict[str, Any]:
        """Quality-control a progression: how faithful and how clean is the output?

        Measures, on the original vs. progressed face:

        * ``structural_similarity`` — mean SSIM (identity-bearing structure kept);
        * ``noise_ratio`` — texture-robust noise estimate, progressed / original
          (above 1.5 means the progression injected noise);
        * ``sharpness_ratio`` — median-tile Laplacian variance ratio (below 0.5
          means detail was destroyed).

        With a fitted :class:`~omni_mercury_engine.vision.VisualAnomalyAnalyzer`
        (for example one fitted on genuine faces from the same capture
        conditions), the progressed face is also analysed for localised
        artefacts. Thresholds are screening values, stated in the output.

        Returns:
            A JSON-safe dictionary of the measurements and flags.
        """
        if not result.success or result.original_face is None or result.progressed_face is None:
            return {"assessed": False, "reason": result.message or "progression did not succeed"}
        from omni_mercury_engine.vision.imaging import assess_environment, to_canonical

        original = to_canonical(result.original_face)
        progressed = to_canonical(result.progressed_face)
        env_o, env_p = assess_environment(original), assess_environment(progressed)
        noise_ratio = env_p.noise_sigma / max(env_o.noise_sigma, 1e-6)
        sharpness_ratio = env_p.sharpness / max(env_o.sharpness, 1e-12)
        ssim = structural_similarity(result.original_face, result.progressed_face)
        flags = []
        if noise_ratio > 1.5:
            flags.append("noise_injected")
        if sharpness_ratio < 0.5:
            flags.append("detail_lost")
        if ssim < 0.6:
            flags.append("structure_not_preserved")
        report: dict[str, Any] = {
            "assessed": True,
            "structural_similarity": round(ssim, 4),
            "noise_ratio": round(float(noise_ratio), 3),
            "sharpness_ratio": round(float(sharpness_ratio), 3),
            "flags": flags,
            "thresholds": {
                "noise_ratio": 1.5,
                "sharpness_ratio": 0.5,
                "structural_similarity": 0.6,
            },
            "geometric_model": result.geometric_model,
            "texture_model": result.texture_model,
        }
        if analyzer is not None:
            visual = analyzer.analyze(result.progressed_face)
            report["visual_analysis"] = visual.to_dict()
        return report

    def create_age_timeline(
        self,
        image_path: str,
        age_range: tuple[int, int] = (-10, 10),
        step: int = 5,
    ) -> list[AgeProgressionResult]:
        """Create timeline of age-progressed images.

        Args:
            image_path: Source image path
            age_range: (min_delta, max_delta)
            step: Year increment

        Returns:
            List of AgeProgressionResult for each age delta
        """
        timeline = []
        for age_delta in range(age_range[0], age_range[1] + 1, step):
            result = self.progress_age(image_path, age_delta)
            if result.success:
                timeline.append(result)
        return timeline

    def _compute_similarity(
        self, emb1: np.ndarray[Any, Any] | None, emb2: np.ndarray[Any, Any] | None
    ) -> float:
        """Compute cosine similarity between embeddings."""
        if emb1 is None or emb2 is None:
            return 0.0

        try:
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            emb1_norm = emb1 / norm1
            emb2_norm = emb2 / norm2

            similarity = np.dot(emb1_norm, emb2_norm)
            return float((similarity + 1) / 2)

        except Exception:
            return 0.0


class QuantumAgeVariant:
    """Age-uncertainty helpers (legacy "quantum" naming; nothing quantum runs).

    ``apply_quantum_uncertainty`` is a plain scalar widening factor and
    ``compute_age_probability_distribution`` is a discrete Gaussian kernel
    over an age window.
    """

    @staticmethod
    def apply_quantum_uncertainty(age_estimate: float, uncertainty: float = 1.2) -> float:
        """Widen an age estimate by a scalar factor (legacy-named helper)."""
        return age_estimate * uncertainty

    @staticmethod
    def compute_age_probability_distribution(
        base_age: int, quantum_factor: float = 1.2
    ) -> dict[int, float]:
        """Generate probability distribution for age range.

        Args:
            base_age: Estimated base age
            quantum_factor: Quantum uncertainty factor

        Returns:
            Dictionary mapping ages to probabilities
        """
        ages = range(max(0, base_age - 10), base_age + 11)
        probs = {}

        for age in ages:
            distance = abs(age - base_age)
            prob = np.exp(-(distance**2) / (2 * quantum_factor**2))
            probs[age] = prob

        total = sum(probs.values())
        return {age: prob / total for age, prob in probs.items()}


__all__ = [
    "DEEPFACE_AVAILABLE",
    "FACENET_AVAILABLE",
    "FACE_RECOGNITION_AVAILABLE",
    "AgeProgressionEngine",
    "AgeProgressionResult",
    "BiometricFusion",
    "BiometricResult",
    "MatchCategory",
    "MultimodalBiometricEngine",
    "QuantumAgeVariant",
]

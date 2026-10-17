# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Biometric anomaly detection model."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment, unused-ignore]
    TORCH_AVAILABLE = False

from omni_mercury_engine._compat import preload_triton_before_tensorflow

# deepface pulls TensorFlow into the process (1.8 s of the 2.0 s it took to
# construct an engine, plus a stream of CUDA probe messages), and only image
# inputs use it. ``DeepFace`` therefore stays a module-level name, patchable
# like any other, that starts out unresolved and is filled by ``_deepface()``
# the first time an image needs it.
_UNRESOLVED: Any = object()
DeepFace: Any = _UNRESOLVED


def _deepface() -> Any:
    """Return the ``DeepFace`` class, importing it on first use (``None`` if unusable)."""
    global DeepFace
    if DeepFace is _UNRESOLVED:
        # triton (torch's compiler backend) segfaults if it loads after
        # TensorFlow's LLVM, so bind it first.
        preload_triton_before_tensorflow()
        try:
            from deepface import DeepFace as loaded
        except Exception:
            # ``except Exception`` (not just ``(ImportError, ValueError)``)
            # because the deepface -> retinaface -> tensorflow import chain
            # can raise ``OSError`` (missing CUDA runtime), ``RuntimeError``
            # (tf version mismatch), or ``AttributeError`` (stale transitive
            # deps) in addition to the known ``ValueError`` from the
            # tf-keras check.  ``BaseException`` is deliberately not caught.
            loaded = None
        DeepFace = loaded
    return DeepFace


def __getattr__(name: str) -> Any:
    """Resolve ``DEEPFACE_AVAILABLE`` when it is read (importing DeepFace if needed)."""
    if name == "DEEPFACE_AVAILABLE":
        return _deepface() is not None
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Flipped once, on the first feature-extraction call that has to fall back to
# the simulated harmonic path because DeepFace is unavailable.  That fallback
# is deterministic but is *not* a face-identity embedding, so we surface a
# single WARNING rather than let callers silently mistake degraded signal for
# real recognition (``biometric_multimodal.py`` logs the equivalent per call).
_SIMULATED_FALLBACK_WARNED = False


def _warn_simulated_biometric_fallback() -> None:
    """Warn once when face features fall back to the simulated path.

    Fires the first time ``extract_features`` uses the harmonic fallback
    because DeepFace is unavailable, so the degraded mode is never silent.
    """
    global _SIMULATED_FALLBACK_WARNED
    if not _SIMULATED_FALLBACK_WARNED:
        _SIMULATED_FALLBACK_WARNED = True
        logger.warning(
            "DeepFace unavailable -- biometric features are using the simulated "
            "harmonic fallback, which is NOT a face-identity embedding and must "
            "not be relied on for recognition or matching. Install the 'face' "
            "extra on Python < 3.14 for real embeddings."
        )


class HarmonicDecomposer:
    """Simple harmonic decomposition using FFT for biometric feature analysis."""

    def decompose(self, signal: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Decompose signal into harmonic components using FFT."""
        if signal.ndim == 1:
            signal = signal.reshape(1, -1)
        fft_result = np.fft.fft(signal, axis=1)
        return np.abs(fft_result)


class FourierAnalyzer:
    """Fourier analysis for frequency-domain biometric features."""

    def analyze(self, data: np.ndarray[Any, Any]) -> dict[str, np.ndarray[Any, Any]]:
        """Analyze frequency components of biometric data."""
        if data.ndim == 1:
            data = data.reshape(1, -1)
        fft_result = np.fft.fft(data, axis=1)
        power_spectrum = np.abs(fft_result) ** 2
        return {
            "frequencies": np.fft.fftfreq(data.shape[1]),
            "power_spectrum": power_spectrum,
            "phase": np.angle(fft_result),
        }


class BiometricAnomalyModel:
    """Biometric anomaly detection for facial recognition and analysis."""

    def __init__(self, config: dict[str, Any] | None = None, **kwargs: Any) -> None:
        """Initialize the instance."""
        self.config = config or {}
        self.model_name = self.config.get("model_name", "Facenet")
        self.use_harmonic_features = self.config.get("use_harmonic_features", True)
        self.harmonic_decomposer = HarmonicDecomposer()
        self.fourier_analyzer = FourierAnalyzer()
        self.target_embedding_size = 128
        self.visual_analyzer: Any = None

    def fit_normal(self, images: list[Any], config: Any = None) -> BiometricAnomalyModel:
        """Learn the normal appearance of genuine captures for visual anomaly analysis.

        Fits a :class:`~omni_mercury_engine.vision.VisualAnomalyAnalyzer` on
        ``images`` (paths or arrays of genuine, well-captured biometric images
        from the deployment's capture conditions — for example aligned face
        crops from one enrolment station). Afterwards :meth:`predict` on a
        single image returns a calibrated, localised, classified and explained
        visual anomaly analysis.

        Args:
            images: Normal images (at least 4).
            config: Optional :class:`~omni_mercury_engine.vision.AnalyzerConfig`.

        Returns:
            ``self``.
        """
        from omni_mercury_engine.vision import AnalyzerConfig, VisualAnomalyAnalyzer

        analyzer = VisualAnomalyAnalyzer(config or AnalyzerConfig())
        self.visual_analyzer = analyzer.fit([self._resolve_image_path(img) for img in images])
        return self

    def _visual_assessment(self, image: Any) -> dict[str, Any]:
        """Visual anomaly analysis of one image (calibrated when a normal model is fitted).

        With a fitted normal model the full report is returned and the
        pipeline-facing score is ``1 - p_value`` of the conformal content
        decision (neutral 0.5 when the decision is withheld). Without one, no
        content verdict is invented: capture conditions and forensic
        screening are reported, with presentation-attack cues (screen-replay
        moiré / print halftone peaks, spliced noise levels) flagged at stated
        absolute thresholds.
        """
        analyzer = self.visual_analyzer
        if analyzer is not None and analyzer.is_fitted:
            report = analyzer.analyze(image)
            if report.is_anomalous is None or report.p_value is None:
                score = 0.5
            else:
                score = 1.0 - float(report.p_value)
            return {
                "anomaly_scores": np.array([score], dtype=np.float32),
                "anomaly_score_basis": "1 - conformal p-value of the visual anomaly analysis",
                "visual_analysis": report.to_dict(),
            }

        from omni_mercury_engine.vision.forensics import (
            jpeg_grid_analysis,
            noise_inconsistency,
            periodic_artifact_analysis,
        )
        from omni_mercury_engine.vision.imaging import assess_environment, to_canonical

        try:
            canonical = to_canonical(image)
        except ValueError as exc:
            return {
                "anomaly_scores": np.array([0.5], dtype=np.float32),
                "anomaly_score_basis": "unavailable",
                "visual_analysis": {"state": "unavailable", "reason": str(exc)},
            }
        forensics = {
            "noise_inconsistency": noise_inconsistency(canonical.luminance),
            "jpeg_grid": jpeg_grid_analysis(canonical.luminance),
            "periodic_artifacts": periodic_artifact_analysis(canonical.luminance),
        }
        cues = []
        if forensics["periodic_artifacts"]["peak_excess"] > 3.5:
            cues.append(
                "isolated spectral peaks (excess "
                f"{forensics['periodic_artifacts']['peak_excess']:.1f} > 3.5): "
                "screen-replay moire or print halftone"
            )
        if forensics["noise_inconsistency"]["max_abs_z"] > 6.0:
            cues.append(
                "sensor-noise level inconsistent across the image "
                f"(|z| {forensics['noise_inconsistency']['max_abs_z']:.1f} > 6): splice or retouch"
            )
        if forensics["jpeg_grid"]["grid_misaligned"]:
            cues.append("JPEG grid at a non-zero phase: cropped or recompressed after editing")
        return {
            "anomaly_scores": np.array([0.5], dtype=np.float32),
            "anomaly_score_basis": "unavailable (no normal model fitted)",
            "visual_analysis": {
                "state": "unavailable",
                "reason": "no normal model fitted: call fit_normal(normal_images) for a "
                "calibrated content anomaly analysis",
                "environment": assess_environment(canonical).to_dict(),
                "forensics": forensics,
                "presentation_attack_cues": cues,
                "cue_thresholds": "absolute screening values, uncalibrated for this domain",
                "provenance": canonical.provenance.to_dict(),
            },
        }

    @staticmethod
    def _is_image_input(data: object) -> bool:
        """Return True only for inputs DeepFace can actually process.

        DeepFace accepts an image path or an array shaped like an image
        (H x W x C, or a batch N x H x W x C, with C in {1, 3, 4}). The
        fusion pipeline routes 2-D tabular feature matrices through every
        model, including this one; without this guard those tabular arrays
        reach ``DeepFace.represent`` / ``DeepFace.analyze``, triggering a
        ~92 MB model download and a stream of "DeepFace ... failed"
        warnings before falling back. Restricting DeepFace to genuine
        image input keeps the tabular detect path on the harmonic
        fallback silently.
        """
        if isinstance(data, str):
            return True
        if isinstance(data, np.ndarray):
            if data.ndim == 3 and data.shape[-1] in (1, 3, 4):
                return True
            if data.ndim == 4 and data.shape[-1] in (1, 3, 4):
                return True
        return False

    @staticmethod
    def _resolve_image_path(data: Any) -> Any:
        """Load a path string into an RGB pixel array; pass everything else through.

        Every downstream branch needs pixels (DeepFace calls here receive the
        ndarray-coerced input, and the harmonic fallback is numeric-only), but
        the old ``np.array(path_string)`` coercion produced a 0-d string array
        no branch could process — the documented "path or array" contract
        never actually worked for paths.

        Raises:
            ValueError: the path cannot be loaded (missing OpenCV, unreadable
                or nonexistent file), naming the exact blocker.
        """
        if not isinstance(data, str):
            return data
        try:
            import cv2
        except ImportError as exc:
            raise ValueError(
                f"image path {data!r} needs OpenCV to load; install opencv-python "
                "or pass a pixel array"
            ) from exc
        loaded = cv2.imread(data)
        if loaded is None:
            raise ValueError(f"image could not be loaded from path: {data!r}")
        return loaded[..., ::-1]  # BGR -> RGB

    def _extract_harmonic_features(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Extract features using harmonic decomposition and Fourier analysis."""
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        if data.ndim == 0 or data.dtype.kind not in "fiub":
            raise ValueError(
                "harmonic feature extraction needs a numeric array; got "
                f"ndim={data.ndim}, dtype={data.dtype} - load image paths into "
                "pixel arrays before extraction"
            )

        if data.ndim == 3:
            data = data.reshape(data.shape[0], -1)
        elif data.ndim == 1:
            data = data.reshape(1, -1)

        batch_size = data.shape[0]

        if not self.use_harmonic_features or self.config.get("use_harmonic_features") is False:
            return np.zeros((batch_size, self.target_embedding_size), dtype=np.float32)

        harmonic = self.harmonic_decomposer.decompose(data)
        fourier_result = self.fourier_analyzer.analyze(data)

        power_spectrum = fourier_result["power_spectrum"]
        phase = fourier_result["phase"]

        def leading(block: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
            """First 32 columns of ``block``, zero-padded when narrower."""
            if block.shape[1] >= 32:
                return block[:, :32]
            return np.pad(block, ((0, 0), (0, 32 - block.shape[1])))

        def summary(block: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
            """Mean, standard deviation, maximum and minimum of every row."""
            return np.stack(
                [block.mean(axis=1), block.std(axis=1), block.max(axis=1), block.min(axis=1)],
                axis=1,
            )

        features = np.concatenate(
            [
                leading(harmonic),
                leading(power_spectrum),
                leading(phase),
                summary(harmonic),
                summary(power_spectrum),
                np.zeros((batch_size, 24)),
            ],
            axis=1,
        )
        return features[:, : self.target_embedding_size].astype(np.float32)

    def _normalize_embedding_size(
        self, embedding: np.ndarray[Any, Any] | torch.Tensor
    ) -> np.ndarray[Any, Any] | torch.Tensor:
        """Normalize embedding to target size (128 features)."""
        if TORCH_AVAILABLE and isinstance(embedding, torch.Tensor):
            is_torch = True
            embedding_np = embedding.detach().cpu().numpy()
        else:
            is_torch = False
            embedding_np = np.array(embedding)

        if embedding_np.ndim == 1:
            embedding_np = embedding_np.reshape(1, -1)

        batch_size, current_size = embedding_np.shape

        if current_size == self.target_embedding_size:
            normalized = embedding_np
        elif current_size > self.target_embedding_size:
            normalized = embedding_np[:, : self.target_embedding_size]
        else:
            padding = np.zeros(
                (batch_size, self.target_embedding_size - current_size), dtype=embedding_np.dtype
            )
            normalized = np.concatenate([embedding_np, padding], axis=1)

        if is_torch and TORCH_AVAILABLE:
            return torch.from_numpy(normalized)
        return normalized

    def extract_features(
        self, data: str | np.ndarray[Any, Any] | dict[str, Any]
    ) -> np.ndarray[Any, Any] | torch.Tensor:
        """Extract biometric features from image data."""
        if isinstance(data, dict):
            data = data["reference"] if "reference" in data else np.array(next(iter(data.values())))

        data = self._resolve_image_path(data)

        image_input = self._is_image_input(data)

        if not isinstance(data, np.ndarray):
            data = np.array(data)

        if data.ndim == 1:
            data = data.reshape(1, -1)

        deepface = _deepface() if image_input else None
        if deepface is not None:
            try:
                result = deepface.represent(
                    data, model_name=self.model_name, enforce_detection=False
                )
                if isinstance(result, list) and len(result) > 0:
                    embeddings = [r["embedding"] for r in result]
                    features = np.array(embeddings, dtype=np.float32)
                else:
                    features = self._extract_harmonic_features(data)
            except Exception as e:
                logger.debug("DeepFace feature extraction failed, using harmonic fallback: %s", e)
                features = self._extract_harmonic_features(data)
        else:
            if image_input:
                _warn_simulated_biometric_fallback()
            features = self._extract_harmonic_features(data)

        features = self._normalize_embedding_size(features)  # type: ignore[assignment, unused-ignore]

        if TORCH_AVAILABLE:
            if isinstance(features, np.ndarray):
                return torch.from_numpy(features)
            return features
        return features

    def _embedding_vector(
        self, image: str | np.ndarray[Any, Any]
    ) -> tuple[np.ndarray[Any, Any] | None, str]:
        """Return a 1-D embedding for one image plus the backend that ACTUALLY ran.

        The backend label is decided by what produced the vector, not by what
        was available: ``"deepface"`` only when the DeepFace representation
        succeeded, ``"harmonic_fallback"`` whenever the deterministic
        harmonic/Fourier path produced it (including after a DeepFace failure).
        One embedding (the first) is used when a backend returns several, and
        the label says so — a multi-face image never concatenates into an
        incomparable vector. ``(None, reason)`` names every unmeasurable case.
        """
        try:
            image = self._resolve_image_path(image)
        except ValueError as exc:
            return None, str(exc)

        features: np.ndarray[Any, Any] | None = None
        backend = "harmonic_fallback"
        if self._is_image_input(image):
            deepface = _deepface()
            if deepface is None:
                _warn_simulated_biometric_fallback()
            else:
                try:
                    represented = deepface.represent(
                        image, model_name=self.model_name, enforce_detection=False
                    )
                    if isinstance(represented, list) and represented:
                        features = np.array([r["embedding"] for r in represented], dtype=np.float32)
                        backend = "deepface"
                except Exception as exc:
                    logger.debug("DeepFace representation failed, using harmonic fallback: %s", exc)

        if features is None:
            try:
                array = np.asarray(image)
                if self._is_image_input(array):
                    # One image (or one per batch item) is ONE sample: flatten
                    # the pixels so the harmonic features describe the whole
                    # image, never each pixel row as a separate "embedding".
                    array = (
                        array.reshape(1, -1)
                        if array.ndim == 3
                        else array.reshape(array.shape[0], -1)
                    )
                features = np.asarray(self._extract_harmonic_features(array))
            except ValueError as exc:
                return None, str(exc)

        normalized = self._normalize_embedding_size(features)
        if TORCH_AVAILABLE and isinstance(normalized, torch.Tensor):
            normalized = normalized.detach().cpu().numpy()
        matrix = np.asarray(normalized, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.shape[0] > 1:
            backend += f" (first of {matrix.shape[0]} embeddings)"
        vector = matrix[0]
        if float(np.linalg.norm(vector)) < 1e-12:
            return None, "embedding is all zeros (no measurable features for this input)"
        return vector, backend

    @staticmethod
    def _cosine_match_score(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any]) -> float:
        """Cosine similarity mapped from ``[-1, 1]`` to a ``[0, 1]`` match score."""
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        return float(np.clip((1.0 + cos) / 2.0, 0.0, 1.0))

    def _predict_match(
        self,
        reference: str | np.ndarray[Any, Any],
        test: str | np.ndarray[Any, Any],
        enable_age_progression: bool = False,
    ) -> dict[str, Any]:
        """Compare a reference and a test image and return a real match score.

        The score is the cosine similarity of the two embeddings mapped to
        ``[0, 1]``. With DeepFace installed the embeddings are FaceNet
        representations; otherwise the model's deterministic harmonic/Fourier
        features are used and the result is appearance-level spectral
        similarity, **not** identity verification — the ``embedding_backend``
        and ``verification_note`` fields state which grade was produced.

        With ``enable_age_progression`` and a reference given as an image
        path, seeded age-progressed variants of the reference are also matched
        against the test image and the best score is reported as
        ``match_score_age_robust`` (useful when the two images may be years
        apart). When the progression backend is unavailable the result says
        so explicitly instead of silently skipping.
        """
        ref_vec, ref_info = self._embedding_vector(reference)
        test_vec, test_info = self._embedding_vector(test)
        if ref_vec is None or test_vec is None:
            reason = ref_info if ref_vec is None else test_info
            return {
                "model_type": "biometric",
                "match_score": None,
                "anomaly_scores": np.array([0.5], dtype=np.float32),
                "error": f"match unavailable: {reason}",
            }

        match_score = self._cosine_match_score(ref_vec, test_vec)
        backend = ref_info if ref_info == test_info else f"{ref_info}+{test_info}"
        result: dict[str, Any] = {
            "model_type": "biometric",
            "match_score": match_score,
            # Pipeline-facing anomaly: a poor match is the anomalous outcome.
            "anomaly_scores": np.array([1.0 - match_score], dtype=np.float32),
            "embedding_backend": backend,
            "verification_note": (
                "FaceNet embedding similarity"
                if backend == "deepface"
                else "harmonic-feature similarity (appearance-level, not identity verification)"
            ),
        }

        if enable_age_progression:
            result["age_progression"] = self._age_robust_match(reference, test_vec, match_score)
            applied = result["age_progression"].get("applied", False)
            if applied:
                result["match_score_age_robust"] = result["age_progression"]["best_score"]

        return result

    def _age_robust_match(
        self,
        reference: str | np.ndarray[Any, Any],
        test_vec: np.ndarray[Any, Any],
        baseline_score: float,
        age_deltas: tuple[int, ...] = (-10, -5, 5, 10),
    ) -> dict[str, Any]:
        """Match seeded age-progressed reference variants against the test embedding.

        Returns an ``applied: False`` record naming the exact blocker when the
        progression path cannot run (array reference, missing OpenCV, failed
        face detection) — the caller never receives a silently absent result.
        """
        if not isinstance(reference, str):
            return {
                "applied": False,
                "reason": "age progression needs the reference as an image path",
            }
        try:
            from omni_mercury_engine.models.biometric_multimodal import AgeProgressionEngine
        except ImportError as exc:
            return {"applied": False, "reason": f"age-progression engine unavailable: {exc}"}

        engine = AgeProgressionEngine(seed=0)  # deterministic perturbations
        per_delta: dict[str, float] = {}
        for delta in age_deltas:
            progressed = engine.progress_age(reference, target_age_delta=delta)
            if not progressed.success or progressed.progressed_face is None:
                return {
                    "applied": False,
                    "reason": f"age progression failed at delta {delta}: {progressed.message or 'face detection failed'}",
                }
            variant_vec, info = self._embedding_vector(progressed.progressed_face)
            if variant_vec is None:
                return {"applied": False, "reason": f"progressed-variant embedding failed: {info}"}
            per_delta[str(delta)] = self._cosine_match_score(variant_vec, test_vec)

        return {
            "applied": True,
            "baseline_score": baseline_score,
            "scores_by_delta": per_delta,
            "best_score": max(baseline_score, *per_delta.values()),
            "note": (
                "heuristic polynomial age filters (seeded, deterministic); "
                "no validated accuracy claim is made for the progression itself"
            ),
        }

    def predict(self, data: np.ndarray[Any, Any] | dict[str, Any]) -> dict[str, Any]:
        """Analyse biometric input: a verification pair, one image, or a tabular feature matrix.

        * ``{"reference": ..., "test": ...}`` — reference-vs-test match
          (:meth:`_predict_match`), optionally age-robust.
        * One image (path or pixel array) — visual anomaly analysis
          (:meth:`_visual_assessment`: calibrated, localised, classified and
          explained once :meth:`fit_normal` has run; capture-condition and
          forensic screening otherwise), plus face attributes **only when
          DeepFace measured them** (``attributes_measured``).
        * A tabular matrix (the fusion pipeline's route) — the harmonic
          spectral-deviation score; no face attributes are claimed for it.
        """
        if data is None:
            return {
                "model_type": "biometric",
                "anomaly_scores": np.array([0.5], dtype=np.float32),
                "attributes_measured": False,
                "error": "No input data provided",
            }

        if isinstance(data, dict):
            if "reference" in data and "test" in data:
                # Two-image verification: compute a real reference-vs-test
                # match instead of silently analysing the reference alone.
                return self._predict_match(
                    data["reference"],
                    data["test"],
                    enable_age_progression=bool(data.get("enable_age_progression", False)),
                )
            if "reference" in data:
                data = data["reference"]
            elif len(data) > 0:
                data = next(iter(data.values()))
            else:
                data = np.array([])

        if isinstance(data, str):
            try:
                data = self._resolve_image_path(data)
            except ValueError as exc:
                return {
                    "model_type": "biometric",
                    "anomaly_scores": np.array([0.5], dtype=np.float32),
                    "error": str(exc),
                }

        image_input = self._is_image_input(data)

        if not isinstance(data, np.ndarray):
            data = np.array(data)

        if image_input:
            result: dict[str, Any] = {"model_type": "biometric"}
            result.update(self._visual_assessment(data))
            result["attributes_measured"] = False
            deepface = _deepface()
            if deepface is not None:
                try:
                    analysis = deepface.analyze(
                        data, actions=["age", "gender", "emotion"], enforce_detection=False
                    )
                    if isinstance(analysis, list) and len(analysis) > 0:
                        analysis = analysis[0]
                    gender = analysis.get("gender", {})
                    emotion = analysis.get("emotion", {})
                    if "age" in analysis:
                        result["age"] = analysis["age"]
                    if gender:
                        result["gender_confidence"] = float(max(gender.values()))
                    if emotion:
                        result["emotion_confidence"] = float(max(emotion.values()))
                    result["attributes_measured"] = any(
                        key in result for key in ("age", "gender_confidence", "emotion_confidence")
                    )
                except Exception as e:
                    logger.warning("DeepFace biometric analysis failed: %s", e)
                    result["error"] = f"Analysis failed: {e}"
            return result

        features = self.extract_features(data)
        if TORCH_AVAILABLE and isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()

        anomaly_score = np.mean(np.abs(features - 0.5))  # type: ignore[assignment, unused-ignore]

        return {
            "model_type": "biometric",
            "anomaly_scores": np.array([anomaly_score], dtype=np.float32),
            "anomaly_score_basis": "harmonic spectral deviation (tabular route; not a visual anomaly score)",
            "attributes_measured": False,
        }

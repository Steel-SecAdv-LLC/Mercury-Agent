# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Deterministic, offline concrete VLM detector.

:class:`StatisticalVLMDetector` is a *concrete* implementation of the
:class:`~omni_mercury_engine.detectors.vlm.base_vlm.BaseVLMDetector` contract
that runs **fully offline** with **no network and no model download**. It
exists so the public ``detectors.vlm`` surface ships at least one
instantiable, wired-and-tested detector — the base class stays a genuine
ABC, but it is no longer the *only* thing on the public path.

Why this is a surrogate, not a replacement
-------------------------------------------
The production VLM backends (AnyAnomaly / LAVAD / BLIP, see
``lvlm_backends.py`` and ``blip_vlm.py``) require the ``transformers``
package and a HuggingFace Hub download of a multi-billion-parameter
Large Vision-Language Model. Those are external dependencies that are not
available in every environment (``transformers`` is optional and the model
weights are network-gated behind ``SafeHFLoader``'s revision pin). Rather
than retire the VLM surface or leave only an un-instantiable ABC, this
detector preserves the public contract with a deterministic computer-vision
heuristic: per-frame texture/intensity salience statistics drive the anomaly
score, and the visual-question-answering control flow (`_create_prompt` →
score → `_parse_response`) is exercised for real.

Remediation plan to graduate to a trained VLM
----------------------------------------------
1. ``pip install 'omni-mercury-engine[vlm]'`` (pulls ``transformers`` and
   ``accelerate`` -- both required by the transformers-backed VLM backends).
2. Use :class:`~omni_mercury_engine.detectors.vlm.blip_vlm.BLIPVLMDetector`
   or :func:`~omni_mercury_engine.detectors.vlm.lvlm_backends.get_lvlm_backend`
   with an explicit ``revision=`` SHA pin.
3. Keep this detector as the offline / CI / smoke-test fallback; it is not a
   substitute for a real LVLM's semantic understanding.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from omni_mercury_engine.detectors.provenance import DecisionSource
from omni_mercury_engine.detectors.vlm.base_vlm import BaseVLMDetector, VLMConfig

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Salience-score logistic calibration. A perfectly flat frame has
# ``raw == 0`` and must map *below* the default 0.5 anomaly threshold; a
# high-texture / high-variance frame must map well above it. These constants
# are deterministic and unit-pinned by ``tests/test_vlm_detectors.py``.
_SCORE_CENTER = 0.15
_SCORE_SLOPE = 8.0
_FEATURE_DIM = 8

# Parse patterns for ``_parse_response`` — deterministic, no model required.
# The leading-answer pattern is checked first, and the negative token set
# before the affirmative one: a response that *begins* with a negation
# ("No anomaly detected") embeds affirmative words ("detected") and must not
# parse as an anomaly (B6 audit cure).
_ANSWER_RE = re.compile(r"^\s*(?:answer\s*[:=]?\s*)?(yes|no)\b", re.IGNORECASE)
_YES_RE = re.compile(r"\b(yes|anomaly|anomalous|abnormal|detected|present)\b", re.IGNORECASE)
_NO_RE = re.compile(r"\b(no|normal|nominal|absent|none)\b", re.IGNORECASE)
_CONF_RE = re.compile(r"confidence[:=\s]+([0-9]*\.?[0-9]+)", re.IGNORECASE)


class StatisticalVLMDetector(BaseVLMDetector):
    """Offline, deterministic statistical texture heuristic (not a language model).

    Implements all five :class:`BaseVLMDetector` contract methods using
    reproducible image statistics instead of a Large Vision-Language Model:
    the anomaly score is a fixed logistic map of per-frame texture/intensity
    salience, and ``anomaly_description`` does **not** influence the score —
    there is no semantic understanding here.  Intended as the offline
    default / CI fallback for the ``detectors.vlm`` surface; production
    deployments should use a ``transformers``-backed backend (see module
    docstring).

    Decision calibration: unfitted, frames are flagged by the hand-set
    ``confidence_threshold`` cut on the salience score (texture-rich but
    perfectly normal imagery then flags — the score constants are unit
    pins, not a calibration).  Calling :meth:`fit` with reference frames
    calibrates the decision: the fitted interval holds the central
    ``1 - contamination`` mass of the reference salience scores and frames
    scoring outside it (in either direction) are flagged.
    """

    decision_source = DecisionSource(
        kind="experimental",
        description=(
            "Statistical texture heuristic, not a vision-language model: score = "
            "sigmoid(8 x (raw - 0.15)) with raw = pixel std + 0.5 x (x-gradient + y-gradient) "
            "+ 0.3 x brightness; anomaly_description does not influence the score. Unfitted, "
            "is_anomaly cuts at confidence_threshold (default 0.5) — a hand-set cut under "
            "which textured normal imagery flags. fit() calibrates the decision to a central "
            "1 - contamination quantile interval of the reference scores; frames outside it "
            "flag. The score constants themselves remain unit-pinned, not fitted."
        ),
    )

    def __init__(self, config: VLMConfig | dict[str, Any] | None = None) -> None:
        """Initialize the detector and its (sentinel) statistical model.

        Args:
            config: Detector configuration. ``model_type`` is ignored — this
                detector never loads an LVLM.
        """
        super().__init__(config)
        # Decision interval calibrated by ``fit`` (None = unfitted: the
        # hand-set confidence_threshold cut applies).
        self._fitted_score_interval: tuple[float, float] | None = None
        # Eagerly build the (trivial) statistical "model" so ``.model`` /
        # ``.processor`` are non-None and never trigger a network load.
        self._initialize_model()

    def fit(self, data: NDArray[Any] | torch.Tensor) -> StatisticalVLMDetector:
        """Calibrate the salience decision interval from reference frames.

        The score formula's constants stay fixed, but the *decision* becomes
        calibrated: the fitted interval holds the central
        ``1 - contamination`` mass (default 0.95) of the reference frames'
        salience scores, and :meth:`detect` flags frames scoring outside it
        in either direction — so texture-rich normal scenes stop flagging,
        and frames meaningfully smoother *or* busier than the reference do.

        The interval's ends are actual reference scores, taken at
        finite-sample ranks (the ``alpha / 2`` quantile rounded DOWN to an
        order statistic, the ``1 - alpha / 2`` quantile rounded UP), so at
        most ``alpha`` of the reference frames fall outside it.  An
        interpolated quantile lies strictly inside ``[min, max]`` and so
        always flagged both reference extremes, however small the set (2 of
        2 frames, 2 of 5, 2 of 10).  With fewer than ``2 / alpha`` frames
        (40 at the default) no frame can be spared for either tail and the
        interval is the reference minimum and maximum.

        Args:
            data: Reference (normal) images/video, any supported layout.

        Returns:
            Self for method chaining.
        """
        scores = self._salience_scores(self.extract_features(data)).detach().cpu().numpy()
        alpha = float(self._contamination)
        if scores.size * alpha < 2.0:
            self._fitted_score_interval = (float(scores.min()), float(scores.max()))
        else:
            self._fitted_score_interval = (
                float(np.quantile(scores, alpha / 2, method="lower")),
                float(np.quantile(scores, 1 - alpha / 2, method="higher")),
            )
        self._is_fitted = True
        return self

    # ------------------------------------------------------------------ #
    # Contract method 1/5
    # ------------------------------------------------------------------ #
    def _initialize_model(self) -> None:
        """Build the deterministic statistical model (no network/download)."""
        # The "model" is the statistics engine itself; a non-None sentinel
        # keeps the ``model`` / ``processor`` properties from recursing into
        # a load that would otherwise hit the network.
        self._model = {"kind": "statistical", "feature_dim": _FEATURE_DIM}
        self._processor = {"kind": "statistical"}

    # ------------------------------------------------------------------ #
    # Contract method 2/5
    # ------------------------------------------------------------------ #
    def _create_prompt(
        self,
        anomaly_description: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Create a deterministic VQA prompt string.

        Args:
            anomaly_description: Description of the anomaly to detect.
            context: Optional context information; its keys are appended
                deterministically (sorted) so the prompt is reproducible.

        Returns:
            A formatted, deterministic prompt string.
        """
        prompt = (
            f"Question: Does this frame contain {anomaly_description}? "
            "Answer 'yes' or 'no' and give a confidence in [0, 1]."
        )
        if context:
            hints = ", ".join(f"{k}={context[k]}" for k in sorted(context))
            prompt = f"{prompt} Context: {hints}."
        return prompt

    # ------------------------------------------------------------------ #
    # Contract method 3/5
    # ------------------------------------------------------------------ #
    def _parse_response(self, response: str) -> tuple[bool, float, str]:
        """Parse a (yes/no + confidence) response deterministically.

        Args:
            response: Model response text (here, a synthesised VQA answer).

        Returns:
            Tuple of ``(is_anomaly, confidence, explanation)``.
        """
        conf_match = _CONF_RE.search(response)
        confidence = float(conf_match.group(1)) if conf_match else 0.0
        confidence = max(0.0, min(1.0, confidence))

        # Decide anomaly from an anchored leading yes/no first; then from
        # negative before affirmative tokens (a leading negation such as
        # "No anomaly detected" embeds affirmative words and used to parse
        # as an anomaly); last, fall back to the parsed confidence vs the
        # configured threshold.
        answer_match = _ANSWER_RE.match(response)
        if answer_match:
            is_anomaly = answer_match.group(1).lower() == "yes"
        elif _NO_RE.search(response):
            is_anomaly = False
        elif _YES_RE.search(response):
            is_anomaly = True
        else:
            is_anomaly = confidence >= self.vlm_config.confidence_threshold

        explanation = response.strip()
        return is_anomaly, confidence, explanation

    # ------------------------------------------------------------------ #
    # Frame statistics (the deterministic "vision" stage)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_nchw(data: NDArray[Any] | torch.Tensor) -> torch.Tensor:
        """Normalise arbitrary image/video input to a float ``[N, C, H, W]`` tensor.

        Accepts numpy or torch input in ``[H, W]``, ``[C, H, W]``,
        ``[H, W, C]``, ``[N, C, H, W]`` or ``[N, H, W, C]`` layouts and pixel
        values in either ``[0, 1]`` or ``[0, 255]``. Output is scaled to
        ``[0, 1]``.
        """
        tensor = torch.as_tensor(np.asarray(data) if not isinstance(data, torch.Tensor) else data)
        tensor = tensor.float()

        if tensor.ndim == 2:  # [H, W] grayscale frame
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.ndim == 3:
            # [C, H, W] if a small leading channel dim, else [H, W, C].
            if tensor.shape[0] in (1, 3) and tensor.shape[-1] not in (1, 3):
                tensor = tensor.unsqueeze(0)
            elif tensor.shape[-1] in (1, 3):
                tensor = tensor.permute(2, 0, 1).unsqueeze(0)
            else:  # ambiguous square single-channel stack -> treat as [N, H, W]
                tensor = tensor.unsqueeze(1)
        elif tensor.ndim == 4:
            # Channel-last [N, H, W, C] -> [N, C, H, W].
            if tensor.shape[-1] in (1, 3) and tensor.shape[1] not in (1, 3):
                tensor = tensor.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Unsupported input rank {tensor.ndim}; expected 2-4 dims")

        if tensor.max() > 1.5:  # values look like [0, 255]
            tensor = tensor / 255.0
        return tensor.clamp(0.0, 1.0)

    def extract_features(self, data: NDArray[Any] | torch.Tensor) -> torch.Tensor:
        """Extract a deterministic ``[N, 8]`` salience-statistics feature tensor.

        The eight per-frame features are luma mean, luma std, horizontal and
        vertical gradient energy, 10th/90th intensity percentiles, and bright/
        dark pixel fractions. These are the fusion-pipeline features.

        Args:
            data: Input images/video, any supported layout (see ``_to_nchw``).

        Returns:
            ``[N, 8]`` float tensor of per-frame statistics.
        """
        frames = self._to_nchw(data)
        luma = frames.mean(dim=1)  # [N, H, W] channel-averaged intensity
        n = luma.shape[0]
        flat = luma.reshape(n, -1)

        mean = flat.mean(dim=1)
        std = flat.std(dim=1, unbiased=False)
        # Spatial gradient energy (texture/edges).
        grad_x = (luma[:, :, 1:] - luma[:, :, :-1]).abs().reshape(n, -1).mean(dim=1)
        grad_y = (luma[:, 1:, :] - luma[:, :-1, :]).abs().reshape(n, -1).mean(dim=1)
        p10 = torch.quantile(flat, 0.10, dim=1)
        p90 = torch.quantile(flat, 0.90, dim=1)
        bright = (flat > 0.7).float().mean(dim=1)
        dark = (flat < 0.3).float().mean(dim=1)

        return torch.stack([mean, std, grad_x, grad_y, p10, p90, bright, dark], dim=1)

    def _salience_scores(self, features: torch.Tensor) -> torch.Tensor:
        """Map per-frame statistics to a calibrated anomaly score in ``[0, 1]``.

        A flat (zero-variance) frame maps below 0.5; a high-texture / high-
        variance frame maps well above it.
        """
        std, grad_x, grad_y, bright = (
            features[:, 1],
            features[:, 2],
            features[:, 3],
            features[:, 6],
        )
        raw = std + 0.5 * (grad_x + grad_y) + 0.3 * bright
        return torch.sigmoid(_SCORE_SLOPE * (raw - _SCORE_CENTER))

    # ------------------------------------------------------------------ #
    # Contract method 4/5
    # ------------------------------------------------------------------ #
    def detect(self, data: NDArray[Any] | torch.Tensor) -> dict[str, Any]:
        """Detect anomalies via deterministic salience statistics + VQA control flow.

        Args:
            data: Images ``[N, C, H, W]`` or video frames ``[T, C, H, W]``.

        Returns:
            Dict with ``scores`` (``[N]``), ``is_anomaly`` (``[N]`` bool),
            ``explanations`` (``list[str]``), and ``features`` (``[N, 8]``
            tensor) — the :class:`BaseVLMDetector` contract.
        """
        features = self.extract_features(data)
        scores = self._salience_scores(features)
        threshold = self.vlm_config.confidence_threshold
        interval = self._fitted_score_interval

        # The prompt is created (and surfaced in the explanation) so the VQA
        # control flow is real, but it is NOT fed to ``_parse_response`` — the
        # question text itself names the anomaly, which would bias the parser.
        prompt = self._create_prompt(self.vlm_config.anomaly_description)
        explanations: list[str] = []
        flags: list[bool] = []
        for score in scores.tolist():
            if interval is not None:
                # Calibrated decision (see ``fit``): outside the fitted
                # central-quantile interval of the reference scores.
                anomalous = score < interval[0] or score > interval[1]
                range_text = "calibrated"
            else:
                anomalous = score >= threshold
                range_text = "nominal"
            verdict = "yes" if anomalous else "no"
            # Synthesise *only the answer* the VQA stage would have produced,
            # then parse it for real so ``_parse_response`` is on the live path.
            answer = (
                f"Answer: {verdict}. Confidence: {score:.4f}. "
                f"Frame salience {'is outside' if anomalous else 'is within'} "
                f"the {range_text} range."
            )
            is_anomaly, _, parsed = self._parse_response(answer)
            flags.append(is_anomaly)
            explanations.append(f"{prompt} -> {parsed}")

        return {
            "scores": scores.detach().cpu().numpy(),
            "is_anomaly": np.asarray(flags, dtype=bool),
            "explanations": explanations,
            "features": features,
        }

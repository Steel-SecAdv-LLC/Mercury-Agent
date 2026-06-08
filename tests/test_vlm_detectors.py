# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for Vision-Language Model (VLM) anomaly detectors.

Tests AnyAnomaly and LAVAD zero-shot/training-free detectors.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.vlm


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestAnyAnomalyDetector:
    """Tests for AnyAnomaly zero-shot VAD detector."""

    def test_anyanomaly_initialization(self) -> None:
        """Test AnyAnomaly can be initialized with default config."""
        from omni_mercury_engine.detectors.vlm import AnyAnomalyDetector

        detector = AnyAnomalyDetector()
        assert detector is not None
        assert detector.config is not None

    def test_anyanomaly_config(self) -> None:
        """Test AnyAnomaly with custom config."""
        from omni_mercury_engine.detectors.vlm import AnyAnomalyDetector
        from omni_mercury_engine.detectors.vlm.anyanomaly import AnyAnomalyConfig

        config = AnyAnomalyConfig(
            context_window=4,
            enable_positional_context=True,
            enable_temporal_context=False,
        )
        detector = AnyAnomalyDetector(config=config)
        cfg = cast("AnyAnomalyConfig", detector.config)
        assert cfg.context_window == 4
        assert cfg.enable_positional_context is True

    def test_anyanomaly_set_anomaly_definition(self) -> None:
        """Test setting custom anomaly definition."""
        from omni_mercury_engine.detectors.vlm import AnyAnomalyDetector

        detector = AnyAnomalyDetector()
        detector.set_anomaly_definition("A person falling down or collapsing on the ground")
        assert detector.anomaly_definition is not None

    def test_anyanomaly_set_reference_normal(self, sample_image_batch: Any) -> None:
        """Test setting reference normal frames."""
        from omni_mercury_engine.detectors.vlm import AnyAnomalyDetector

        detector = AnyAnomalyDetector()
        # Convert to list of frames
        frames = [sample_image_batch[i] for i in range(sample_image_batch.shape[0])]
        detector.set_reference_normal(frames)
        assert len(detector.reference_frames) > 0

    def test_anyanomaly_mock_backend_factory_hard_fails(self) -> None:
        """Failure-mode coverage on the detect-path:
        ``MockLVLMBackend`` is intentionally a hard-fail (Phase 2 audit
        cure: silent mock degradation is not permitted in production).
        Pin that contract — any path that reaches
        ``MockLVLMBackend.initialize()`` must raise
        ``NotImplementedError`` rather than producing fabricated
        outputs that could be mistaken for real detections.

        This used to be exercised indirectly via the deleted
        ``test_anyanomaly_detect_mock``, which constructed an
        ``AnyAnomalyDetector`` with the now-removed ``backend="mock"``
        config field and called ``detector.detect(...)``.  That test
        was permanently skipped because the failure mode is
        intentional, but dropping it left the file with no
        AnyAnomaly detect-path coverage.  This smoke replaces that
        coverage at the factory boundary, where the contract actually
        lives, without requiring a real VLM checkpoint or a CUDA
        runner."""
        from omni_mercury_engine.detectors.vlm.lvlm_backends import (
            MockLVLMBackend,
            get_lvlm_backend,
        )

        backend = get_lvlm_backend(model_type="mock")
        assert isinstance(backend, MockLVLMBackend)

        with pytest.raises(NotImplementedError, match="cannot be used in production"):
            backend.initialize()


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestLAVADDetector:
    """Tests for LAVAD training-free LLM-based detector."""

    def test_lavad_initialization(self) -> None:
        """Test LAVAD can be initialized with default config."""
        from omni_mercury_engine.detectors.vlm import LAVADDetector

        detector = LAVADDetector()
        assert detector is not None
        assert detector.config is not None

    def test_lavad_config(self) -> None:
        """Test LAVAD with custom config."""
        from omni_mercury_engine.detectors.vlm import LAVADDetector
        from omni_mercury_engine.detectors.vlm.lavad import LAVADConfig

        config = LAVADConfig(
            llm_model="mock",
            vlm_model="mock",
            sampling_fps=1.0,
            temporal_window=8,
        )
        detector = LAVADDetector(config=config)
        cfg = cast("LAVADConfig", detector.config)
        assert cfg.sampling_fps == 1.0
        assert cfg.temporal_window == 8

    def test_lavad_set_scene_context(self) -> None:
        """Test setting scene context."""
        from omni_mercury_engine.detectors.vlm import LAVADDetector

        detector = LAVADDetector()
        detector.set_scene_context(
            scene_description="A busy shopping mall",
            expected_activities=["shopping", "walking", "talking"],
            anomaly_types=["theft", "fighting", "falling"],
        )
        assert detector.scene_context is not None

    def test_lavad_detect_video_mock_hard_fails(self, sample_video_frames: Any) -> None:
        """``LAVADDetector(config=LAVADConfig(llm_model="mock"))`` must
        raise when ``detect_video`` is called.

        Phase 2 audit cure (commit 4d29bf1): ``MockLVLMBackend.initialize``
        and ``MockLLMAdapter.__init__`` are hard-fails — silent mock
        degradation is not permitted in production.  This test pins the
        positive contract: a stale caller that wires up a "mock" LVLM /
        LLM in production must trip ``NotImplementedError``.
        """
        from omni_mercury_engine.detectors.vlm import LAVADDetector
        from omni_mercury_engine.detectors.vlm.lavad import LAVADConfig

        config = LAVADConfig(llm_model="mock", vlm_model="mock")
        detector = LAVADDetector(config=config)

        video = torch.stack(sample_video_frames)
        with pytest.raises(NotImplementedError, match="cannot be used in production"):
            detector.detect_video(video)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestBaseVLMDetector:
    """Contract tests for the experimental VLM detector base (ROADMAP #3).

    The base is a genuine ABC: the five contract methods are
    ``@abstractmethod`` (no ``NotImplementedError`` stub on the public path),
    so the base cannot be instantiated directly.
    """

    def test_base_is_abstract_not_instantiable(self) -> None:
        """Direct instantiation raises TypeError, not a runtime NotImplementedError."""
        from omni_mercury_engine.detectors.vlm import BaseVLMDetector

        with pytest.raises(TypeError):
            BaseVLMDetector()  # type: ignore[abstract]

    def test_contract_methods_are_abstract(self) -> None:
        """The five contract methods are declared abstract."""
        from omni_mercury_engine.detectors.vlm.base_vlm import BaseVLMDetector

        assert BaseVLMDetector.__abstractmethods__ == frozenset(
            {
                "_initialize_model",
                "_create_prompt",
                "_parse_response",
                "detect",
                "extract_features",
            }
        )

    def test_frame_sampling(self) -> None:
        """The concrete ``_sample_frames`` helper works on a concrete subclass."""
        from omni_mercury_engine.detectors.vlm.base_vlm import BaseVLMDetector

        class _Concrete(BaseVLMDetector):
            def _initialize_model(self) -> None: ...

            def _create_prompt(
                self, anomaly_description: str, context: dict[str, Any] | None = None
            ) -> str:
                return ""

            def _parse_response(self, response: str) -> tuple[bool, float, str]:
                return (False, 0.0, "")

            def detect(self, data: np.ndarray[Any, Any] | torch.Tensor) -> dict[str, Any]:
                return {}

            def extract_features(self, data: np.ndarray[Any, Any] | torch.Tensor) -> torch.Tensor:
                return torch.zeros(1)

        detector = _Concrete()
        # Create mock video [T, H, W, C]
        video = torch.randn(30, 224, 224, 3)
        sampled = detector._sample_frames(video, n_frames=8)
        assert len(sampled) == 8


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestStatisticalVLMDetector:
    """Behavioural tests for the offline concrete VLM detector (un-retires #3).

    ``StatisticalVLMDetector`` is the first *instantiable* detector on the
    public ``detectors.vlm`` path: a deterministic, network-free surrogate
    that implements all five contract methods for real. These tests assert
    behaviour (contract shape, salience ordering, determinism), not stubs.
    """

    def _make(self) -> Any:
        from omni_mercury_engine.detectors.vlm import StatisticalVLMDetector

        return StatisticalVLMDetector()

    def test_offline_instantiation_no_network(self) -> None:
        """It instantiates offline and exposes a non-None statistical model."""
        det = self._make()
        # ``.model`` / ``.processor`` must NOT trigger a network load.
        assert det.model is not None
        assert det.processor is not None
        assert det._is_fitted is True

    def test_detect_returns_full_contract(self) -> None:
        """``detect`` returns the four documented keys with correct types/shapes."""
        det = self._make()
        frames = torch.rand(5, 3, 32, 32)
        out = det.detect(frames)
        assert set(out) >= {"scores", "is_anomaly", "explanations", "features"}
        assert out["scores"].shape == (5,)
        assert out["is_anomaly"].shape == (5,)
        assert out["is_anomaly"].dtype == bool
        assert len(out["explanations"]) == 5
        assert out["features"].shape == (5, 8)

    def test_scores_in_unit_interval(self) -> None:
        """All anomaly scores lie in [0, 1]."""
        det = self._make()
        scores = det.detect(torch.rand(8, 3, 24, 24))["scores"]
        assert float(scores.min()) >= 0.0
        assert float(scores.max()) <= 1.0

    def test_flat_frame_is_not_anomalous(self) -> None:
        """A constant (zero-texture) frame scores below threshold -> not anomalous."""
        det = self._make()
        flat = torch.full((1, 3, 32, 32), 0.5)
        out = det.detect(flat)
        assert float(out["scores"][0]) < 0.5
        assert bool(out["is_anomaly"][0]) is False

    def test_textured_frame_scores_higher_than_flat(self) -> None:
        """The real signal: a high-variance frame is more anomalous than a flat one."""
        det = self._make()
        torch.manual_seed(0)
        flat = torch.full((1, 3, 32, 32), 0.5)
        noisy = torch.rand(1, 3, 32, 32)
        flat_score = float(det.detect(flat)["scores"][0])
        noisy_score = float(det.detect(noisy)["scores"][0])
        assert noisy_score > flat_score
        assert noisy_score >= 0.5  # textured frame crosses the anomaly threshold

    def test_determinism(self) -> None:
        """Identical input yields byte-identical scores and features (no RNG)."""
        det = self._make()
        torch.manual_seed(1)
        data = torch.rand(4, 3, 20, 20)
        a = det.detect(data)
        b = det.detect(data)
        assert np.allclose(a["scores"], b["scores"])
        assert torch.allclose(a["features"], b["features"])

    def test_extract_features_shape_and_finiteness(self) -> None:
        """Feature extraction returns a finite ``[N, 8]`` tensor."""
        det = self._make()
        feats = det.extract_features(torch.rand(3, 3, 16, 16))
        assert feats.shape == (3, 8)
        assert torch.isfinite(feats).all()

    def test_uint8_range_input_is_normalised(self) -> None:
        """Inputs in [0, 255] are normalised, not treated as out-of-range."""
        det = self._make()
        frame = torch.randint(0, 256, (1, 3, 16, 16)).float()
        out = det.detect(frame)
        assert 0.0 <= float(out["scores"][0]) <= 1.0

    def test_channel_last_video_input(self) -> None:
        """A channel-last video tensor ``[T, H, W, C]`` is handled."""
        det = self._make()
        video = torch.rand(6, 28, 28, 3)  # [T, H, W, C]
        out = det.detect(video)
        assert out["scores"].shape == (6,)

    def test_create_prompt_is_deterministic_and_uses_context(self) -> None:
        """Prompt construction is reproducible and folds in sorted context keys."""
        det = self._make()
        p1 = det._create_prompt("a falling person", {"zone": "B", "cam": 3})
        p2 = det._create_prompt("a falling person", {"cam": 3, "zone": "B"})
        assert p1 == p2  # key order does not matter
        assert "a falling person" in p1
        assert "cam=3" in p1 and "zone=B" in p1

    def test_parse_response_extracts_decision_and_confidence(self) -> None:
        """The response parser recovers (is_anomaly, confidence, explanation)."""
        det = self._make()
        yes = det._parse_response("Answer: yes. Confidence: 0.83. Salience exceeds range.")
        no = det._parse_response("Answer: no. Confidence: 0.12. Within nominal range.")
        assert yes[0] is True and abs(yes[1] - 0.83) < 1e-9
        assert no[0] is False and abs(no[1] - 0.12) < 1e-9


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestContextProviders:
    """Tests for context extraction utilities."""

    def test_positional_context_extractor(self, sample_image: Any) -> None:
        """Test positional context extraction."""
        from omni_mercury_engine.detectors.vlm.context_providers import PositionalContextExtractor

        extractor = PositionalContextExtractor()
        context = extractor.extract(sample_image)
        assert isinstance(context, dict)

    def test_temporal_context_extractor(self, sample_video_frames: Any) -> None:
        """Test temporal context extraction."""
        from omni_mercury_engine.detectors.vlm.context_providers import TemporalContextExtractor

        extractor = TemporalContextExtractor(window_size=4)
        context = extractor.extract(sample_video_frames)
        assert isinstance(context, dict)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestLVLMBackends:
    """Tests for LVLM backend implementations."""

    def test_mock_backend_hard_fails_on_use(self) -> None:
        """``MockLVLMBackend`` must raise ``NotImplementedError`` the
        moment its model is used.

        Phase 2 audit cure (commit 4d29bf1): ``MockLVLMBackend`` may be
        constructed (so the registry / factory layout stays uniform)
        but ``initialize`` is a hard-fail, which means any real call
        — VQA, generation, feature extraction — short-circuits to
        ``NotImplementedError`` rather than silently returning fake
        scores.  This test pins that contract.
        """
        from omni_mercury_engine.detectors.vlm.lvlm_backends import (
            MockLVLMBackend,
            get_lvlm_backend,
        )

        backend = get_lvlm_backend("mock")
        # ``isinstance`` both narrows the type for mypy and pins the runtime
        # contract -- if the factory wiring ever regresses to a different
        # backend, this assertion fails with a clear message instead of
        # producing an AttributeError on the missing ``vqa`` method.
        assert isinstance(backend, MockLVLMBackend)

        with pytest.raises(NotImplementedError, match="cannot be used in production"):
            backend.vqa(
                image=torch.randn(3, 224, 224),
                question="What is in this image?",
            )

    def test_backend_factory_rejects_unknown_model_type(self) -> None:
        """``get_lvlm_backend("unknown")`` must raise ``ValueError``.

        Phase 2 audit cure: the legacy fall-through to
        ``MockLVLMBackend`` for an unknown ``model_type`` masked
        configuration errors and silently routed production traffic
        through the mock backend (whose internal methods are all
        hard-fail stubs).  The factory now raises at configuration
        time, surfacing the typo / wiring error immediately.
        """
        from omni_mercury_engine.detectors.vlm.lvlm_backends import get_lvlm_backend

        with pytest.raises(ValueError, match="Unknown LVLM model_type"):
            get_lvlm_backend("unknown")

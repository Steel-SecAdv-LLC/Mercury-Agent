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

from typing import cast

"""
Tests for Vision-Language Model (VLM) anomaly detectors.

Tests AnyAnomaly and LAVAD zero-shot/training-free detectors.
"""

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

    def test_anyanomaly_set_reference_normal(self, sample_image_batch) -> None:
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

    def test_lavad_detect_video_mock_hard_fails(self, sample_video_frames) -> None:
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
    """Tests for base VLM detector class."""

    def test_base_vlm_initialization(self) -> None:
        """Test BaseVLMDetector can be initialized."""
        from omni_mercury_engine.detectors.vlm import BaseVLMDetector

        detector = BaseVLMDetector()
        assert detector is not None

    def test_frame_sampling(self) -> None:
        """Test frame sampling functionality."""
        from omni_mercury_engine.detectors.vlm.base_vlm import BaseVLMDetector

        detector = BaseVLMDetector()

        # Create mock video [T, H, W, C]
        video = torch.randn(30, 224, 224, 3)

        # Sample frames
        sampled = detector._sample_frames(video, n_frames=8)
        assert len(sampled) == 8


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestContextProviders:
    """Tests for context extraction utilities."""

    def test_positional_context_extractor(self, sample_image) -> None:
        """Test positional context extraction."""
        from omni_mercury_engine.detectors.vlm.context_providers import PositionalContextExtractor

        extractor = PositionalContextExtractor()
        context = extractor.extract(sample_image)
        assert isinstance(context, dict)

    def test_temporal_context_extractor(self, sample_video_frames) -> None:
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

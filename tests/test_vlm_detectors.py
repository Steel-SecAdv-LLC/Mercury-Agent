"""
OMNI ♱ AVA (O♱A)
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

    def test_anyanomaly_initialization(self):
        """Test AnyAnomaly can be initialized with default config."""
        from omni_anomaly_engine.detectors.vlm import AnyAnomalyDetector

        detector = AnyAnomalyDetector()
        assert detector is not None
        assert detector.config is not None

    def test_anyanomaly_config(self):
        """Test AnyAnomaly with custom config."""
        from omni_anomaly_engine.detectors.vlm import AnyAnomalyDetector
        from omni_anomaly_engine.detectors.vlm.anyanomaly import AnyAnomalyConfig

        config = AnyAnomalyConfig(
            backend="mock",
            context_window=4,
            enable_positional_context=True,
            enable_temporal_context=False,
        )
        detector = AnyAnomalyDetector(config=config)
        assert detector.config.context_window == 4
        assert detector.config.enable_positional_context is True

    def test_anyanomaly_set_anomaly_definition(self):
        """Test setting custom anomaly definition."""
        from omni_anomaly_engine.detectors.vlm import AnyAnomalyDetector

        detector = AnyAnomalyDetector()
        detector.set_anomaly_definition("A person falling down or collapsing on the ground")
        assert detector.anomaly_definition is not None

    def test_anyanomaly_set_reference_normal(self, sample_image_batch):
        """Test setting reference normal frames."""
        from omni_anomaly_engine.detectors.vlm import AnyAnomalyDetector

        detector = AnyAnomalyDetector()
        # Convert to list of frames
        frames = [sample_image_batch[i] for i in range(sample_image_batch.shape[0])]
        detector.set_reference_normal(frames)
        assert len(detector.reference_frames) > 0

    def test_anyanomaly_detect_mock(self, sample_image):
        """Test AnyAnomaly detection with mock backend."""
        from omni_anomaly_engine.detectors.vlm import AnyAnomalyDetector
        from omni_anomaly_engine.detectors.vlm.anyanomaly import AnyAnomalyConfig

        config = AnyAnomalyConfig(backend="mock")
        detector = AnyAnomalyDetector(config=config)
        detector.set_anomaly_definition("A person running")

        result = detector.detect(sample_image)
        assert "scores" in result
        assert "reasoning" in result
        assert "is_anomaly" in result


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestLAVADDetector:
    """Tests for LAVAD training-free LLM-based detector."""

    def test_lavad_initialization(self):
        """Test LAVAD can be initialized with default config."""
        from omni_anomaly_engine.detectors.vlm import LAVADDetector

        detector = LAVADDetector()
        assert detector is not None
        assert detector.config is not None

    def test_lavad_config(self):
        """Test LAVAD with custom config."""
        from omni_anomaly_engine.detectors.vlm import LAVADDetector
        from omni_anomaly_engine.detectors.vlm.lavad import LAVADConfig

        config = LAVADConfig(
            llm_model="mock",
            vlm_model="mock",
            sampling_fps=1.0,
            temporal_window=8,
        )
        detector = LAVADDetector(config=config)
        assert detector.config.sampling_fps == 1.0
        assert detector.config.temporal_window == 8

    def test_lavad_set_scene_context(self):
        """Test setting scene context."""
        from omni_anomaly_engine.detectors.vlm import LAVADDetector

        detector = LAVADDetector()
        detector.set_scene_context(
            scene_description="A busy shopping mall",
            expected_activities=["shopping", "walking", "talking"],
            anomaly_types=["theft", "fighting", "falling"],
        )
        assert detector.scene_context is not None

    def test_lavad_detect_video_mock(self, sample_video_frames):
        """Test LAVAD video detection with mock backend."""
        from omni_anomaly_engine.detectors.vlm import LAVADDetector
        from omni_anomaly_engine.detectors.vlm.lavad import LAVADConfig

        config = LAVADConfig(llm_model="mock", vlm_model="mock")
        detector = LAVADDetector(config=config)

        # Stack frames into video tensor
        video = torch.stack(sample_video_frames)
        result = detector.detect_video(video)

        assert "scores" in result
        assert "frame_scores" in result
        assert "captions" in result


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestBaseVLMDetector:
    """Tests for base VLM detector class."""

    def test_base_vlm_initialization(self):
        """Test BaseVLMDetector can be initialized."""
        from omni_anomaly_engine.detectors.vlm import BaseVLMDetector

        detector = BaseVLMDetector()
        assert detector is not None

    def test_frame_sampling(self):
        """Test frame sampling functionality."""
        from omni_anomaly_engine.detectors.vlm.base_vlm import BaseVLMDetector

        detector = BaseVLMDetector()

        # Create mock video [T, H, W, C]
        video = torch.randn(30, 224, 224, 3)

        # Sample frames
        sampled = detector._sample_frames(video, n_frames=8)
        assert len(sampled) == 8


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestContextProviders:
    """Tests for context extraction utilities."""

    def test_positional_context_extractor(self, sample_image):
        """Test positional context extraction."""
        from omni_anomaly_engine.detectors.vlm.context_providers import (
            PositionalContextExtractor,
        )

        extractor = PositionalContextExtractor()
        context = extractor.extract(sample_image)
        assert isinstance(context, dict)

    def test_temporal_context_extractor(self, sample_video_frames):
        """Test temporal context extraction."""
        from omni_anomaly_engine.detectors.vlm.context_providers import (
            TemporalContextExtractor,
        )

        extractor = TemporalContextExtractor(window_size=4)
        context = extractor.extract(sample_video_frames)
        assert isinstance(context, dict)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestLVLMBackends:
    """Tests for LVLM backend implementations."""

    def test_mock_backend(self):
        """Test mock LVLM backend."""
        from omni_anomaly_engine.detectors.vlm.lvlm_backends import get_lvlm_backend

        backend = get_lvlm_backend("mock")
        assert backend is not None

        # Test VQA
        response = backend.vqa(
            image=torch.randn(3, 224, 224),
            question="What is in this image?",
        )
        assert isinstance(response, str)

    def test_backend_factory(self):
        """Test backend factory function."""
        from omni_anomaly_engine.detectors.vlm.lvlm_backends import get_lvlm_backend

        # Test that unsupported backends raise error or return mock
        backend = get_lvlm_backend("unknown")
        assert backend is not None  # Falls back to mock

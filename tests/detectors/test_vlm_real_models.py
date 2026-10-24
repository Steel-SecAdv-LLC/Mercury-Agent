# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The video detectors on their real, pinned models, on real photographs.

No stand-in anywhere: LAVAD captions frames with the pinned BLIP captioner
and judges each caption with the pinned Qwen2.5-0.5B-Instruct reasoner;
AnyAnomaly answers its visual question with the pinned BLIP VQA model.  The
clips are real CC0 photographs (``tests/fixtures/visual``): a coffee cup on
every frame, with a cat (scikit-image's "chelsea") as the anomaly in one.

Measured before this tier existed, LAVAD's real default never flagged a
frame: its reasoning prompt went to the captioner, which returns ``""`` for
a text-only prompt, and every window scored 0.  Its scores were also divided
by the clip maximum, which would have flagged the top frame of every clip.

The tier runs offline (``MERCURY_OFFLINE=1`` forces cache-only loads) and is
skipped when the pinned weights are not cached -- unless
``MERCURY_REQUIRES_HF_MODELS=1`` declares it must run, in which case the
session aborts instead (``tests/conftest.py``).  Populate the cache with
``python scripts/prefetch_pinned_models.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
Image = pytest.importorskip("PIL.Image")

from tests.hf_model_cache import missing_pinned_models

_MISSING = missing_pinned_models()
pytestmark = pytest.mark.skipif(
    bool(_MISSING),
    reason=f"pinned models not cached ({', '.join(_MISSING)}); "
    "run scripts/prefetch_pinned_models.py",
)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "visual"
CAT_FRAME = 5


def _frame(name: str) -> np.ndarray[Any, Any]:
    with Image.open(FIXTURES / name) as image:
        rgb = image.convert("RGB").resize((224, 224))
    return np.asarray(rgb, dtype=np.float32).transpose(2, 0, 1) / 255.0


@pytest.fixture(scope="module")
def clips() -> dict[str, np.ndarray[Any, Any]]:
    coffee, cat = _frame("coffee.png"), _frame("chelsea.png")
    frames = [coffee] * 8
    with_cat = list(frames)
    with_cat[CAT_FRAME] = cat
    return {"cat": np.stack(with_cat), "normal": np.stack(frames)}


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every load resolves from the local cache: no network in this tier."""
    monkeypatch.setenv("MERCURY_OFFLINE", "1")


@pytest.fixture(scope="module")
def lavad() -> Any:
    from omni_mercury_engine.detectors.vlm import LAVADDetector

    detector = LAVADDetector()
    detector.set_anomaly_description("a cat")
    return detector


def test_lavad_localises_the_real_anomalous_frame(
    lavad: Any, clips: dict[str, np.ndarray[Any, Any]]
) -> None:
    result = lavad.detect(clips["cat"])
    assert "cat" in result["captions"][CAT_FRAME]
    assert result["anomaly_frames"] == [CAT_FRAME]
    scores = np.asarray(result["scores"])
    assert scores[CAT_FRAME] > 0.9
    assert np.delete(scores, CAT_FRAME).max() < 0.5


def test_lavad_flags_nothing_in_a_normal_clip(
    lavad: Any, clips: dict[str, np.ndarray[Any, Any]]
) -> None:
    """No clip-maximum normalisation: an all-normal clip has no anomalous frame."""
    result = lavad.detect(clips["normal"])
    assert result["anomaly_frames"] == []
    assert np.asarray(result["scores"]).max() < 0.5


def test_lavad_reasoner_is_a_language_model_not_the_captioner(lavad: Any) -> None:
    from omni_mercury_engine.detectors.vlm.lvlm_backends import HFTextReasoningBackend

    assert isinstance(lavad.reasoner, HFTextReasoningBackend)
    assert lavad.reasoner is not lavad.caption_model
    with pytest.raises(ValueError, match="text-only"):
        lavad.reasoner.generate([np.zeros((3, 8, 8))], "describe")


@pytest.fixture(scope="module")
def anyanomaly() -> Any:
    from omni_mercury_engine.detectors.vlm import AnyAnomalyDetector
    from omni_mercury_engine.detectors.vlm.anyanomaly import AnyAnomalyConfig
    from omni_mercury_engine.detectors.vlm.base_vlm import LVLMType

    detector = AnyAnomalyDetector(
        AnyAnomalyConfig(
            model_type=LVLMType.BLIP_VQA,
            model_name="Salesforce/blip-vqa-base",
            segment_length=2,
            segment_overlap=1,
        )
    )
    detector.set_anomaly_definition("a cat")
    return detector


def test_anyanomaly_on_blip_vqa_peaks_at_the_real_anomaly(
    anyanomaly: Any, clips: dict[str, np.ndarray[Any, Any]]
) -> None:
    """Segments are judged as units, so the cat's 2-frame segments are flagged."""
    result = anyanomaly.detect(clips["cat"])
    scores = np.asarray(result["scores"])
    assert int(np.argmax(scores)) == CAT_FRAME
    assert scores[CAT_FRAME] > 0.9
    assert set(result["anomaly_frames"]) <= {CAT_FRAME - 1, CAT_FRAME, CAT_FRAME + 1}
    assert CAT_FRAME in result["anomaly_frames"]


def test_anyanomaly_flags_nothing_in_a_normal_clip(
    anyanomaly: Any, clips: dict[str, np.ndarray[Any, Any]]
) -> None:
    result = anyanomaly.detect(clips["normal"])
    assert result["anomaly_frames"] == []

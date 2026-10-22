# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The five visual anomaly detectors, driven end to end on real images.

Data: real CC0 photographs (``tests/fixtures/visual``, recorded and
SHA-256-verified by ``scripts/record_visual_fixtures.py``).  Normal images are
crops of a real brick wall; a defect is a patch of real gravel pasted into a
brick crop -- the CutPaste protocol (Li et al., CVPR 2021) for foreign-texture
defects.  Every detector is fitted on brick only and must:

* rank defect crops above held-out clean crops (AUROC bar per detector, set
  below what was measured at these CI budgets);
* make an informative *decision*: before calibration existed every detector
  compared its raw, unbounded score with a fixed ``threshold=0.5``, so on real
  images PatchCore, PaDiM and CFlow flagged every image (scores ~3, ~23,
  ~180) and STFPM and Reverse Distillation flagged none (scores ~0.01, ~0.2);
* calibrate its threshold on held-out normals by split conformal prediction
  and record the achieved false-alarm bound;
* survive ``save`` / ``load`` with identical scores and decision -- before,
  PatchCore and PaDiM dropped their fitted model (plain attributes outside
  ``state_dict``), Reverse Distillation and CFlow had no modules to load into,
  and every subclass config failed to rebuild.

Budgets: 64x64 inputs at native resolution and 20-100 epochs for the trained
detectors keep each fit within seconds on CPU.  CFlow is also scored by its
full per-pixel likelihood here: before, it scored only the prior term
``0.5 * ||z||^2`` and dropped the flow's log-determinant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
Image = pytest.importorskip("PIL.Image")

from omni_mercury_engine.core.exceptions import DataException
from omni_mercury_engine.detectors import visual

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "visual"
SIZE = 64
N_TRAIN = 30

#: detector -> (class, config, AUROC bar). Measured on this data at these
#: settings: PatchCore 0.993, PaDiM 0.993, STFPM 0.986, Reverse Distillation
#: 0.944, CFlow 0.750.  CFlow is the weakest here and least stable in training
#: budget (0.674 at 150 epochs); its bar pins what it measurably achieves.
DETECTORS: dict[str, tuple[Any, dict[str, Any], float]] = {
    "patchcore": (visual.PatchCoreDetector, {}, 0.95),
    "padim": (visual.PaDiMDetector, {}, 0.95),
    "reverse_distillation": (visual.ReverseDistillationDetector, {"num_epochs": 20}, 0.90),
    "stfpm": (visual.STFPMDetector, {"num_epochs": 60}, 0.90),
    "cflow": (visual.CFlowDetector, {"num_epochs": 100}, 0.70),
}


def _rgb(name: str) -> np.ndarray[Any, Any]:
    with Image.open(FIXTURES / name) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _crops(source: np.ndarray[Any, Any], rng: np.random.Generator, n: int) -> np.ndarray[Any, Any]:
    ys = rng.integers(0, source.shape[0] - SIZE, size=n)
    xs = rng.integers(0, source.shape[1] - SIZE, size=n)
    return np.stack([source[y : y + SIZE, x : x + SIZE] for y, x in zip(ys, xs, strict=True)])


@pytest.fixture(scope="module")
def data() -> dict[str, np.ndarray[Any, Any]]:
    """Train / clean / defect batches, channel-first, deterministic."""
    brick, gravel = _rgb("brick.png"), _rgb("gravel.png")
    rng = np.random.default_rng(0)
    train = _crops(brick, rng, N_TRAIN)
    clean = _crops(brick, rng, 12)
    defect = _crops(brick, rng, 12)
    patch = SIZE // 4
    foreign = _crops(gravel, rng, 12)
    for i in range(len(defect)):
        y, x = rng.integers(0, SIZE - patch, size=2)
        defect[i, y : y + patch, x : x + patch] = foreign[i, :patch, :patch]

    def chw(a: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return np.ascontiguousarray(a.transpose(0, 3, 1, 2))

    return {"train": chw(train), "clean": chw(clean), "defect": chw(defect)}


def _auroc(positive: np.ndarray[Any, Any], negative: np.ndarray[Any, Any]) -> float:
    pos, neg = np.asarray(positive, float)[:, None], np.asarray(negative, float)[None, :]
    return float(((pos > neg).sum() + 0.5 * (pos == neg).sum()) / (pos.size * neg.size))


def _fitted(name: str, data: dict[str, np.ndarray[Any, Any]], **overrides: Any) -> Any:
    cls, config, _ = DETECTORS[name]
    torch.manual_seed(0)
    detector = cls({"image_size": (SIZE, SIZE), **config, **overrides})
    return detector.fit(data["train"])


@pytest.fixture(scope="module")
def fitted(data: dict[str, np.ndarray[Any, Any]]) -> dict[str, Any]:
    return {name: _fitted(name, data) for name in DETECTORS}


@pytest.mark.parametrize("name", sorted(DETECTORS))
def test_real_defects_rank_above_held_out_clean_images(
    name: str, data: dict[str, np.ndarray[Any, Any]], fitted: dict[str, Any]
) -> None:
    detector = fitted[name]
    clean, defect = detector.detect(data["clean"]), detector.detect(data["defect"])
    assert _auroc(defect["scores"], clean["scores"]) >= DETECTORS[name][2]


@pytest.mark.parametrize("name", sorted(DETECTORS))
def test_calibrated_decision_is_informative(
    name: str, data: dict[str, np.ndarray[Any, Any]], fitted: dict[str, Any]
) -> None:
    """Held-out clean images mostly pass; defects are flagged more often."""
    detector = fitted[name]
    clean = np.asarray(detector.detect(data["clean"])["is_anomaly"], bool)
    defect = np.asarray(detector.detect(data["defect"])["is_anomaly"], bool)
    assert clean.mean() <= 0.5
    assert defect.mean() > clean.mean()


@pytest.mark.parametrize("name", sorted(set(DETECTORS) - {"cflow"}))
def test_fixed_half_threshold_was_uninformative(
    name: str, data: dict[str, np.ndarray[Any, Any]]
) -> None:
    """The pre-calibration default (raw score > 0.5) decides the same on every image.

    CFlow is excluded because this change also replaced its score (full
    per-dimension NLL instead of the prior term, whose values of ~180 made the
    old 0.5 flag everything); the new scale crossing 0.5 is coincidence, which
    is the point of calibrating on the detector's own scale.
    """
    detector = _fitted(name, data, threshold=0.5)
    flags = np.concatenate(
        [
            np.asarray(detector.detect(data["clean"])["is_anomaly"], bool),
            np.asarray(detector.detect(data["defect"])["is_anomaly"], bool),
        ]
    )
    assert flags.all() or not flags.any()
    assert detector.calibration == {"method": "fixed", "threshold": 0.5}


@pytest.mark.parametrize("name", sorted(DETECTORS))
def test_split_conformal_calibration_is_recorded(name: str, fitted: dict[str, Any]) -> None:
    calibration = fitted[name].calibration
    n_cal = round(N_TRAIN * 0.2)
    assert calibration["method"] == "split_conformal"
    assert calibration["n_calibration"] == n_cal
    # k = min(ceil((n + 1)(1 - 0.05)), n) = n, so the bound is 1 / (n + 1).
    assert calibration["false_alarm_bound"] == pytest.approx(1.0 / (n_cal + 1))
    assert fitted[name].threshold == calibration["threshold"]


@pytest.mark.parametrize("name", sorted(DETECTORS))
def test_save_load_round_trip_is_exact(
    name: str, data: dict[str, np.ndarray[Any, Any]], fitted: dict[str, Any], tmp_path: Path
) -> None:
    detector = fitted[name]
    path = tmp_path / f"{name}.pt"
    detector.save(str(path))
    cls, config, _ = DETECTORS[name]
    torch.manual_seed(123)  # reload must not depend on the RNG state
    reloaded = cls({"image_size": (SIZE, SIZE), **config}).load(str(path))
    before, after = detector.detect(data["defect"]), reloaded.detect(data["defect"])
    np.testing.assert_allclose(after["scores"], before["scores"], rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(after["is_anomaly"], before["is_anomaly"])
    assert reloaded.calibration == detector.calibration


@pytest.mark.parametrize("name", sorted(DETECTORS))
def test_registry_harvest_on_real_images(
    name: str, data: dict[str, np.ndarray[Any, Any]], fitted: dict[str, Any]
) -> None:
    features = fitted[name].extract_features(data["defect"])
    assert features.shape[0] == len(data["defect"])
    assert torch.isfinite(torch.as_tensor(features)).all()


def test_calibration_needs_two_images_unless_threshold_is_fixed(
    data: dict[str, np.ndarray[Any, Any]],
) -> None:
    one = data["train"][:1]
    with pytest.raises(DataException, match="at least 2"):
        visual.PatchCoreDetector({"image_size": (SIZE, SIZE)}).fit(one)
    fixed = visual.PatchCoreDetector({"image_size": (SIZE, SIZE), "threshold": 3.0}).fit(one)
    assert fixed.threshold == 3.0

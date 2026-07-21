# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the centralized ``safe_torch_load`` checkpoint wrapper.

The wrapper is Mercury's single sanctioned ``torch.load`` entry point; it
hard-pins ``weights_only=True`` so a checkpoint's pickle stream can never
execute arbitrary code (the RCE-class threat). The path-validation and
policy checks run *before* torch is imported, so they are tested without
the ``[ml]`` extra; the actual load / malicious-payload rejection is
gated on torch.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from omni_mercury_engine.security.safe_torch import (
    DEFAULT_MAX_CHECKPOINT_BYTES,
    UnsafeCheckpointError,
    safe_torch_load,
)

if TYPE_CHECKING:
    pass

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = REPO_ROOT / "src" / "omni_mercury_engine" / "models" / "checkpoints"
SIGMA_WEIGHTS = (
    REPO_ROOT / "src" / "omni_mercury_engine" / "security" / "sigma_immutable_weights.pt"
)


# ---------------------------------------------------------------------------
# Policy / validation checks — these run before the lazy ``import torch`` and
# so must hold even in a torch-free install.
# ---------------------------------------------------------------------------
class TestPolicyNoTorch:
    def test_weights_only_false_is_refused(self, tmp_path: Path) -> None:
        f = tmp_path / "x.pt"
        f.write_bytes(b"PK\x03\x04dummy")
        with pytest.raises(UnsafeCheckpointError, match="weights_only=False"):
            safe_torch_load(f, weights_only=False)

    def test_weights_only_none_is_refused(self, tmp_path: Path) -> None:
        f = tmp_path / "x.pt"
        f.write_bytes(b"dummy")
        with pytest.raises(UnsafeCheckpointError):
            safe_torch_load(f, weights_only=None)  # type: ignore[arg-type]

    def test_custom_pickle_module_is_refused(self, tmp_path: Path) -> None:
        import pickle as _pickle

        f = tmp_path / "x.pt"
        f.write_bytes(b"dummy")
        with pytest.raises(UnsafeCheckpointError, match="pickle_module"):
            safe_torch_load(f, pickle_module=_pickle)

    def test_missing_path_refused(self) -> None:
        with pytest.raises(UnsafeCheckpointError, match="does not exist"):
            safe_torch_load("/nonexistent/checkpoint.pt")

    def test_directory_refused(self, tmp_path: Path) -> None:
        with pytest.raises(UnsafeCheckpointError, match="not a regular file"):
            safe_torch_load(tmp_path)

    def test_empty_file_refused(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.pt"
        f.write_bytes(b"")
        with pytest.raises(UnsafeCheckpointError, match="empty"):
            safe_torch_load(f)

    def test_size_ceiling_refused(self, tmp_path: Path) -> None:
        f = tmp_path / "big.pt"
        f.write_bytes(b"x" * 4096)
        with pytest.raises(UnsafeCheckpointError, match="size ceiling"):
            safe_torch_load(f, max_bytes=1024)

    def test_default_ceiling_is_two_gib(self) -> None:
        assert DEFAULT_MAX_CHECKPOINT_BYTES == 2 * 1024 * 1024 * 1024

    def test_error_type_is_valueerror_subclass(self) -> None:
        # Call sites that catch ValueError/Exception must still catch us.
        assert issubclass(UnsafeCheckpointError, ValueError)


# ---------------------------------------------------------------------------
# Real load behavior — requires torch.
# ---------------------------------------------------------------------------
@pytest.mark.ml
class TestLoadWithTorch:
    def test_loads_shipped_checkpoint(self) -> None:
        torch = pytest.importorskip("torch")
        assert torch is not None
        path = CHECKPOINT_DIR / "solar_storm_geomag.pt"
        if not path.is_file():
            pytest.skip("shipped checkpoint not present")
        payload = safe_torch_load(path)
        assert payload is not None

    def test_loads_sigma_weights(self) -> None:
        pytest.importorskip("torch")
        if not SIGMA_WEIGHTS.is_file():
            pytest.skip("sigma weights not present")
        payload = safe_torch_load(SIGMA_WEIGHTS)
        assert payload is not None

    def test_round_trip_state_dict(self, tmp_path: Path) -> None:
        torch = pytest.importorskip("torch")
        state = {"w": torch.ones(3), "b": torch.zeros(2), "epoch": 7, "lr": 0.01}
        p = tmp_path / "ckpt.pt"
        torch.save(state, p)
        loaded = safe_torch_load(p)
        assert loaded["epoch"] == 7
        assert loaded["lr"] == pytest.approx(0.01)
        assert torch.equal(loaded["w"], torch.ones(3))

    def test_map_location_forwarded(self, tmp_path: Path) -> None:
        torch = pytest.importorskip("torch")
        p = tmp_path / "ckpt.pt"
        torch.save({"w": torch.ones(2)}, p)
        loaded = safe_torch_load(p, map_location="cpu")
        assert loaded["w"].device.type == "cpu"

    def test_stream_input_is_accepted(self, tmp_path: Path) -> None:
        torch = pytest.importorskip("torch")
        import io

        p = tmp_path / "ckpt.pt"
        torch.save({"v": torch.tensor([1.0, 2.0])}, p)
        with io.BytesIO(p.read_bytes()) as stream:
            loaded = safe_torch_load(stream)
        assert torch.equal(loaded["v"], torch.tensor([1.0, 2.0]))

    def test_malicious_reduce_payload_is_refused_and_not_executed(self, tmp_path: Path) -> None:
        """The core security property: an RCE ``__reduce__`` never runs."""
        torch = pytest.importorskip("torch")
        marker = tmp_path / "PWNED"

        class Evil:
            def __reduce__(self):  # type: ignore[no-untyped-def]
                return (os.system, (f"touch {marker}",))

        p = tmp_path / "evil.pt"
        # torch.save writes a full pickle; the wrapper's weights_only=True must
        # refuse it on load.
        torch.save({"payload": Evil()}, p)
        with pytest.raises(UnsafeCheckpointError):
            safe_torch_load(p)
        assert not marker.exists(), "RCE payload executed — weights_only bypassed!"

    def test_representative_call_site_checkpoint_paths(self) -> None:
        """A real call site (shipped-checkpoint loader) routes through the wrapper."""
        pytest.importorskip("torch")
        from omni_mercury_engine.models import checkpoint_paths

        # The loader resolves a shipped checkpoint and loads it safely; if the
        # checkpoint is absent in this environment, the loader raises a clear
        # error rather than falling back to an unsafe load.
        try:
            payload, provenance = checkpoint_paths.load_shipped_checkpoint("solar_storm_geomag")
        except (FileNotFoundError, RuntimeError, KeyError):
            pytest.skip("shipped checkpoint not resolvable in this environment")
        assert payload is not None
        assert provenance is not None

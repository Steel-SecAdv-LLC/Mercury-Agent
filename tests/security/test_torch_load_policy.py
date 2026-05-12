"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Tests for the ``torch.load`` policy: every callsite defaults to
``weights_only=True`` and falls back to ``weights_only=False`` only via
an explicit ``allow_unsafe=True`` caller argument.  The unsafe branch
must also emit a ``logger.warning`` so that any forensic timeline can
attribute the relaxation back to a specific caller.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# lstm_ae.AnomalyDetector.load
# ---------------------------------------------------------------------------


def test_lstm_ae_load_defaults_to_weights_only(tmp_path):
    """Default path: weights_only=True is used, no unsafe fallback."""
    from omni_mercury_engine.models.lstm_ae import AnomalyDetector

    det = AnomalyDetector(input_dim=4, hidden_dim=8, latent_dim=2, seq_len=3, device="cpu")
    path = tmp_path / "model.pt"
    det.save(str(path))

    loaded = AnomalyDetector.load(str(path), device="cpu")
    assert isinstance(loaded, AnomalyDetector)


def test_lstm_ae_load_raises_runtimeerror_without_allow_unsafe(tmp_path):
    """A checkpoint that needs unsafe load must NOT silently fall through."""
    from omni_mercury_engine.models import lstm_ae as lstm_mod
    from omni_mercury_engine.models.lstm_ae import AnomalyDetector

    path = tmp_path / "model.pt"
    path.write_bytes(b"not a real checkpoint")

    with patch.object(lstm_mod.torch, "load", side_effect=RuntimeError("needs unsafe")):
        with pytest.raises(RuntimeError, match="weights_only=True"):
            AnomalyDetector.load(str(path), device="cpu", allow_unsafe=False)


def test_lstm_ae_load_warns_on_allow_unsafe(tmp_path, caplog):
    """The unsafe branch must emit a logger.warning that names the path."""
    from omni_mercury_engine.models import lstm_ae as lstm_mod
    from omni_mercury_engine.models.lstm_ae import AnomalyDetector

    path = tmp_path / "model.pt"
    path.write_bytes(b"not a real checkpoint")

    fake_checkpoint = {
        "input_dim": 4,
        "hidden_dim": 8,
        "latent_dim": 2,
        "seq_len": 3,
        "model_state": AnomalyDetector(
            input_dim=4, hidden_dim=8, latent_dim=2, seq_len=3, device="cpu"
        ).model.state_dict(),
        "threshold": 0.5,
    }

    safe_call_count = {"n": 0}

    def fake_torch_load(p, *, map_location, weights_only):
        if weights_only:
            safe_call_count["n"] += 1
            raise RuntimeError("simulated unsafe-required failure")
        return fake_checkpoint

    with patch.object(lstm_mod.torch, "load", side_effect=fake_torch_load), caplog.at_level("WARNING"):
        AnomalyDetector.load(str(path), device="cpu", allow_unsafe=True)

    assert safe_call_count["n"] == 1, "safe load must be attempted first"
    assert any("allow_unsafe" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# Source-level invariant: every torch.load callsite in src/ that targets a
# disk path includes weights_only=True.  This guards against future
# regressions where someone reintroduces a bare torch.load(path).
# ---------------------------------------------------------------------------


def test_every_torch_load_in_src_pins_weights_only():
    """Static check: no torch.load callsite in src/ omits weights_only."""
    import pathlib
    import re

    src = pathlib.Path(__file__).resolve().parent.parent.parent / "src"
    pattern = re.compile(r"torch\.load\s*\(")

    offenders: list[str] = []
    for py in src.rglob("*.py"):
        text = py.read_text()
        # Grab each torch.load(...) invocation along with the chars up to its
        # matching close paren.  We don't need a real parser — we just look at
        # the rough call body and check for ``weights_only`` mentioned within.
        i = 0
        while True:
            m = pattern.search(text, i)
            if not m:
                break
            start = m.end()
            depth = 1
            j = start
            while j < len(text) and depth:
                ch = text[j]
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                j += 1
            body = text[start:j]
            if "weights_only" not in body:
                offenders.append(f"{py.relative_to(src)}:{text.count(chr(10), 0, m.start()) + 1}")
            i = j

    assert not offenders, (
        "torch.load callsites without weights_only: " + ", ".join(offenders)
    )

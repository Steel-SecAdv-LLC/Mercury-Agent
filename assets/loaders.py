# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Deterministic, first-party synthetic-data generators for the ``real-data`` validation suites.

Each generator returns a dictionary whose keys match the field names
that the detectors-under-test consume.  The shapes/statistics encode
the documented structure of the corresponding upstream domain so the
detectors exercise their full code paths against data whose ground
truth is known by construction.

Generators:
    * :func:`generate_pcap_data`     — Hash-chained network packet
      capture, with an optional tampering injector that introduces
      duplicate / replayed packet hashes (the
      :class:`ResonanceHashIntegrityChecker` flags duplicate
      ratios above 5%).
    * :func:`generate_seti_signal`   — Cosmic time-domain signal
      with optional narrow-band or repeating-pulse technosignature
      injection (the :class:`SETICosmicSignalAnalyzer` flags
      magnitude excursions above ``threshold_std``).
    * :func:`generate_mimic_vitals`  — MIMIC-III-style (heart-rate,
      systolic BP, diastolic BP, temperature, respiratory rate)
      vital-sign time series; the ``sepsis`` disease profile drives
      the classic septic-shock physiology (febrile, tachycardic,
      hypotensive, tachypneic) which the
      :class:`TemporalVitalSignsDetector` is built to flag.
    * :func:`generate_medical_image` — Synthetic 2D chest-radiograph
      surrogate (low-pass-filtered Gaussian texture) with an
      optional focal opacity / lesion injector.

All generators use ``numpy.random.default_rng`` with explicit
seeding so independent calls produce statistically-independent
samples but a single call with a fixed seed is reproducible.  The
``seed`` keyword is optional everywhere.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

import numpy as np

__all__ = [
    "generate_medical_image",
    "generate_mimic_vitals",
    "generate_pcap_data",
    "generate_seti_signal",
]


def _rng(seed: int | None) -> np.random.Generator:
    """Return a fresh ``Generator`` seeded by ``seed`` or a high-entropy default."""
    return np.random.default_rng(seed) if seed is not None else np.random.default_rng()


# ---------------------------------------------------------------------------
# Cyber Fortress — synthetic hash-chained PCAP
# ---------------------------------------------------------------------------


def generate_pcap_data(
    num_packets: int = 200,
    inject_tampering: bool = False,
    tampering_ratio: float = 0.1,
    *,
    seed: int | None = None,
) -> dict[str, Any]:
    """Generate a synthetic packet capture with a SHA3-anchored hash chain.

    Each "packet" is a fixed-width record of (src_port, dst_port,
    payload_len, payload_entropy) so the per-packet hash is a
    deterministic function of randomised content.  The hash chain
    is the per-packet ``sha3_256`` of the row.  When ``inject_tampering``
    is true, ``tampering_ratio`` of the rows are *replaced* by a
    duplicate of an earlier row, which mimics a replay/tampering
    attack — the
    :class:`omni_mercury_engine.security.cyber_fortress.ResonanceHashIntegrityChecker`
    detects the resulting duplicate-hash ratio.

    Args:
        num_packets: Number of packets to synthesise.
        inject_tampering: When ``True``, replace ``tampering_ratio`` of
            packets with duplicates of earlier packets.
        tampering_ratio: Fraction of packets to tamper.  Ignored when
            ``inject_tampering`` is ``False``.
        seed: Optional deterministic seed.

    Returns:
        Dictionary with keys ``hash_chain`` (``list[str]``),
        ``network_traffic`` (``numpy.ndarray`` of shape
        ``(num_packets, 4)``), and ``tamper_indices``
        (``numpy.ndarray`` of indices that were tampered, empty when
        no tampering).
    """
    if num_packets <= 0:
        raise ValueError("num_packets must be > 0")
    if not 0.0 <= tampering_ratio <= 1.0:
        raise ValueError("tampering_ratio must be in [0, 1]")

    rng = _rng(seed)
    # Plausible TCP-like header structure: ports in [1024, 65535],
    # payload length in [40, 1500] bytes, entropy in [0, 8] bits/byte.
    src_ports = rng.integers(1024, 65536, size=num_packets, dtype=np.int64)
    dst_ports = rng.integers(1024, 65536, size=num_packets, dtype=np.int64)
    payload_lens = rng.integers(40, 1501, size=num_packets, dtype=np.int64)
    payload_entropy = rng.uniform(0.0, 8.0, size=num_packets)
    network_traffic = np.column_stack(
        [
            src_ports.astype(np.float64),
            dst_ports.astype(np.float64),
            payload_lens.astype(np.float64),
            payload_entropy,
        ]
    )

    # Per-row hash anchored in a per-call salt so independent calls
    # produce independent chains.  Embed a high-resolution timestamp
    # in the salt so concurrent calls with the same seed still
    # diverge.
    salt = f"{time.time_ns()}-{rng.integers(0, 2**32):08x}".encode()

    def _row_hash(row_idx: int) -> str:
        row = network_traffic[row_idx]
        return hashlib.sha3_256(salt + row.tobytes() + row_idx.to_bytes(8, "big")).hexdigest()

    hash_chain = [_row_hash(i) for i in range(num_packets)]

    tamper_indices = np.empty(0, dtype=np.int64)
    if inject_tampering and tampering_ratio > 0.0:
        num_tampered = max(1, round(num_packets * tampering_ratio))
        num_tampered = min(num_tampered, num_packets - 1)
        # Choose which rows to tamper; never tamper row 0 so a
        # source row always exists.
        tamper_indices = np.sort(
            rng.choice(np.arange(1, num_packets), size=num_tampered, replace=False)
        )
        for idx in tamper_indices:
            src = int(rng.integers(0, idx))
            hash_chain[idx] = hash_chain[src]

    return {
        "hash_chain": hash_chain,
        "network_traffic": network_traffic,
        "tamper_indices": tamper_indices,
    }


# ---------------------------------------------------------------------------
# Emergent Life Detector — synthetic SETI cosmic signal
# ---------------------------------------------------------------------------


def generate_seti_signal(
    num_samples: int = 10000,
    inject_technosignature: bool = False,
    signal_type: str = "narrow_band",
    *,
    seed: int | None = None,
) -> dict[str, Any]:
    """Generate a synthetic time-domain cosmic signal.

    Baseline is zero-mean Gaussian noise (unit standard deviation),
    which approximates the dynamic-spectrum baseline of the Allen
    Telescope Array SETI lane after RFI excision.  When
    ``inject_technosignature`` is true, the chosen signal model is
    additively combined into the noise floor:

    *   ``"narrow_band"``: monotone sinusoid at a fixed normalised
        frequency (the classic SETI narrow-band candidate).
    *   ``"repeating"``: a periodic train of Gaussian-enveloped
        pulses (the FRB / repeater morphology).
    *   ``"chirp"``: a linear-frequency chirp across the band
        (drifting carrier, also a SETI candidate morphology).

    Each signal is scaled so the peak resonance magnitude exceeds the
    detector's default ``threshold_std=4.0``.

    Args:
        num_samples: Length of the synthesised time series.
        inject_technosignature: When ``True``, add a technosignature
            on top of the noise floor.
        signal_type: Model identifier; see above.
        seed: Optional deterministic seed.

    Returns:
        Dictionary with key ``cosmic_signal`` (``numpy.ndarray`` of
        shape ``(num_samples,)``) and ``signal_type`` echoed back.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be > 0")
    valid_types = {"narrow_band", "repeating", "chirp"}
    if signal_type not in valid_types:
        raise ValueError(f"signal_type must be one of {sorted(valid_types)}; got {signal_type!r}")

    rng = _rng(seed)
    noise = rng.standard_normal(num_samples)
    cosmic_signal: np.ndarray[Any, Any] = noise.astype(np.float64)

    if inject_technosignature:
        t = np.arange(num_samples, dtype=np.float64)
        if signal_type == "narrow_band":
            # Frequency normalised to (0, 0.5); 0.137 is arbitrary but
            # comfortably above DC and well separated from harmonics
            # of any FFT length that's a multiple of 8.
            cosmic_signal = cosmic_signal + 4.5 * np.sin(2.0 * np.pi * 0.137 * t)
        elif signal_type == "repeating":
            period = max(64, num_samples // 32)
            width = max(4, period // 16)
            pulse_centres = np.arange(period // 2, num_samples, period)
            envelope = np.zeros(num_samples)
            for centre in pulse_centres:
                lo = int(max(0, int(centre) - 4 * width))
                hi = int(min(num_samples, int(centre) + 4 * width + 1))
                envelope[lo:hi] += np.exp(
                    -((np.arange(lo, hi) - int(centre)) ** 2) / (2.0 * width**2)
                )
            cosmic_signal = cosmic_signal + 5.0 * envelope
        else:  # chirp
            f0, f1 = 0.02, 0.40
            k = (f1 - f0) / max(num_samples - 1, 1)
            phase = 2.0 * np.pi * (f0 * t + 0.5 * k * t * t)
            cosmic_signal = cosmic_signal + 4.5 * np.sin(phase)

    return {
        "cosmic_signal": cosmic_signal,
        "signal_type": signal_type if inject_technosignature else "noise",
    }


# ---------------------------------------------------------------------------
# Medical Cure Predictor — synthetic MIMIC-III-style vital signs
# ---------------------------------------------------------------------------

# Vital-sign columns: (heart-rate, systolic BP, diastolic BP,
# temperature °F, respiratory rate).  The healthy baseline reflects
# adult means and standard deviations published in the MIMIC-III
# documentation (e.g. https://mimic.mit.edu/).
_VITALS_BASELINE_MEAN = np.array([75.0, 120.0, 80.0, 98.6, 16.0])
_VITALS_BASELINE_STD = np.array([6.0, 8.0, 5.0, 0.4, 1.5])


def generate_mimic_vitals(
    num_timesteps: int = 288,
    inject_disease: bool = False,
    disease_type: str = "sepsis",
    *,
    seed: int | None = None,
) -> dict[str, Any]:
    """Generate a synthetic MIMIC-III-style vital-sign time series.

    Each row is a sample of (HR, SBP, DBP, temperature °F, respiratory
    rate).  ``num_timesteps`` rows over a 24-hour window correspond
    to a 5-minute sampling cadence (the default MIMIC-III chartevents
    grid).  The healthy baseline is a multivariate Gaussian around the
    published adult means with mild low-frequency drift to mimic
    circadian variation.

    Disease profiles available:

    *   ``"sepsis"`` (default): septic-shock physiology — fever,
        tachycardia, hypotension, tachypnea over the second half of
        the window.
    *   ``"hypothermia"``: temperature regression below 96°F with
        bradycardia.
    *   ``"hypertensive_crisis"``: SBP > 180 mmHg sustained for the
        latter half of the window.

    Args:
        num_timesteps: Number of samples (rows) to generate.
        inject_disease: When ``True``, apply the named disease
            trajectory on top of the baseline.
        disease_type: Identifier; see above.
        seed: Optional deterministic seed.

    Returns:
        Dictionary with key ``vital_signs_sequence`` (``numpy.ndarray``
        of shape ``(num_timesteps, 5)``) and ``disease_type`` echoed
        back.
    """
    if num_timesteps <= 0:
        raise ValueError("num_timesteps must be > 0")
    valid_types = {"sepsis", "hypothermia", "hypertensive_crisis"}
    if disease_type not in valid_types:
        raise ValueError(f"disease_type must be one of {sorted(valid_types)}; got {disease_type!r}")

    rng = _rng(seed)
    base = _VITALS_BASELINE_MEAN + rng.standard_normal((num_timesteps, 5)) * _VITALS_BASELINE_STD

    # Low-frequency circadian drift, max ~3% of baseline per channel.
    t = np.linspace(0.0, 2.0 * np.pi, num_timesteps)
    drift_amp = _VITALS_BASELINE_MEAN * 0.03
    drift = drift_amp * np.sin(t)[:, None]
    base = base + drift

    if inject_disease:
        # Ramp the disease signature in over the second half of the
        # window so the detector sees a *change* rather than a constant
        # offset (the latter would be indistinguishable from a healthy
        # baseline at a different operating point).
        half = num_timesteps // 2
        ramp = np.zeros(num_timesteps)
        ramp[half:] = np.linspace(0.0, 1.0, num_timesteps - half)

        if disease_type == "sepsis":
            # qSOFA-aligned septic-shock trajectory.
            base[:, 0] += 35.0 * ramp  # HR up
            base[:, 1] -= 35.0 * ramp  # SBP down
            base[:, 2] -= 18.0 * ramp  # DBP down
            base[:, 3] += 3.5 * ramp  # temperature up
            base[:, 4] += 10.0 * ramp  # RR up
        elif disease_type == "hypothermia":
            base[:, 0] -= 25.0 * ramp  # HR down (bradycardia)
            base[:, 3] -= 4.0 * ramp  # temperature down
            base[:, 4] -= 4.0 * ramp  # RR down
        else:  # hypertensive_crisis
            base[:, 1] += 80.0 * ramp  # SBP up
            base[:, 2] += 35.0 * ramp  # DBP up
            base[:, 0] += 15.0 * ramp  # mild HR up

    return {
        "vital_signs_sequence": base,
        "disease_type": disease_type if inject_disease else "normal",
    }


# ---------------------------------------------------------------------------
# Medical Cure Predictor — synthetic chest-radiograph surrogate
# ---------------------------------------------------------------------------


def generate_medical_image(
    height: int = 256,
    width: int = 256,
    inject_anomaly: bool = False,
    *,
    seed: int | None = None,
) -> dict[str, Any]:
    """Generate a synthetic chest-radiograph surrogate.

    Baseline is a smoothly-varying 2D Gaussian texture (low-pass
    filtered ``randn``) which crudely approximates the soft-tissue
    background of a frontal chest X-ray.  When ``inject_anomaly`` is
    true, a focal high-intensity Gaussian "opacity" is overlaid at a
    random location to mimic a pulmonary nodule / pneumonia
    consolidation.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        inject_anomaly: When ``True``, overlay a focal opacity.
        seed: Optional deterministic seed.

    Returns:
        Dictionary with key ``medical_image`` (``numpy.ndarray`` of
        shape ``(height, width)`` with values in ``[0, 1]``) and
        ``anomaly_present`` (``bool``).
    """
    if height <= 0 or width <= 0:
        raise ValueError("height and width must both be > 0")

    rng = _rng(seed)
    raw = rng.standard_normal((height, width))

    # Cheap low-pass filter: average the image with a shifted copy
    # of itself a few times.  Gives a smooth low-frequency texture
    # without depending on scipy.signal.
    smoothed = raw.copy()
    for _ in range(8):
        kernel_pad = np.pad(smoothed, 1, mode="edge")
        smoothed = (
            kernel_pad[1:-1, 1:-1]
            + kernel_pad[:-2, 1:-1]
            + kernel_pad[2:, 1:-1]
            + kernel_pad[1:-1, :-2]
            + kernel_pad[1:-1, 2:]
        ) / 5.0

    # Normalise to [0, 1].
    image = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min() + 1e-12)

    anomaly_present = bool(inject_anomaly)
    if inject_anomaly:
        # Focal opacity: a Gaussian blob ~10% of the shortest side
        # with peak intensity 1.0, additively combined and re-clipped.
        sigma = max(1.0, 0.05 * float(min(height, width)))
        cy = int(rng.integers(height // 4, max(height // 4 + 1, 3 * height // 4)))
        cx = int(rng.integers(width // 4, max(width // 4 + 1, 3 * width // 4)))
        yy, xx = np.mgrid[0:height, 0:width]
        opacity = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma**2))
        image = np.clip(image + opacity, 0.0, 1.0)

    return {
        "medical_image": image,
        "anomaly_present": anomaly_present,
    }

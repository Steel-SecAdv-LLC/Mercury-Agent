#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Score ``rf_vital_signs`` against contact reference sensors on real recordings.

Default data: Schellenberger et al. (2020), "A dataset of clinically recorded
radar vital signs with synchronised reference sensor signals", Sci. Data 7,
291 (CC BY 4.0, figshare 10.6084/m9.figshare.12186516): 30 healthy adults,
24 GHz CW Six-Port radar I/Q at 2 kHz, synchronised two-lead ECG (2 kHz)
and thoracic impedance (100 Hz). ``--fetch`` reads only the requested
scenario's ``.mat`` members out of the three zip archives with HTTP range
requests (about 42 MB per subject for Resting); zipfile checks each
member's CRC-32.

Your own hardware: ``--npz`` files holding ``fs`` and one RF input
(``radar_i`` + ``radar_q`` + ``wavelength_m``, ``radar_displacement_m``,
``radar_phase_rad`` + ``wavelength_m``, or ``csi_amplitude`` and/or
``csi_phase``) plus at least one synchronised reference: ``ecg`` + ``fs_ecg``
or ``ppg`` + ``fs_ppg`` (pulse oximeter) for heart rate, ``resp`` +
``fs_resp`` (impedance or belt) for breathing. Optional ``subject``.

Radar I/Q are demodulated twice -- with ellipse correction
(``radar_iq.demodulate_iq``) and with raw arctangent -- so the front end's
effect is measured, not assumed.

Usage::

    python benchmarks/rf_vitals_reference_validation.py --fetch
    python benchmarks/rf_vitals_reference_validation.py --data-dir DIR
    python benchmarks/rf_vitals_reference_validation.py --npz rec1.npz rec2.npz
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import shutil
import sys
import time
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import (
    io as sio,
    signal,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from omni_mercury_engine.biometric import vital_reference as vr
from omni_mercury_engine.biometric.radar_iq import demodulate_iq, fit_iq_ellipse
from omni_mercury_engine.biometric.rf_vital_signs import track_vital_signs

RESULTS_PATH = REPO_ROOT / "benchmarks" / "results" / "rf_vitals_reference_validation.json"
DEFAULT_CACHE = Path.home() / ".cache" / "mercury" / "schellenberger2020"
FIGSHARE_ZIPS = (
    "https://ndownloader.figshare.com/files/22515785",
    "https://ndownloader.figshare.com/files/22516136",
    "https://ndownloader.figshare.com/files/22521941",
)
#: Six-Port radar carrier of the dataset.
SCHELLENBERGER_WAVELENGTH_M = 299_792_458.0 / 24e9
ANALYSIS_FS = 100.0
HEART_TOLERANCE_BPM = 2.0
BREATH_TOLERANCE_BPM = 1.0


class _RangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file through HTTP range requests."""

    def __init__(self, url: str) -> None:
        self.url, self.pos = url, 0
        with self._get(0, 0) as response:
            self.size = int(response.headers["Content-Range"].rsplit("/", 1)[1])

    def _get(self, first: int, last: int) -> Any:
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={first}-{last}"})
        return urllib.request.urlopen(request, timeout=600)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, buffer: Any) -> int:
        n = min(len(buffer), self.size - self.pos)
        if n <= 0:
            return 0
        with self._get(self.pos, self.pos + n - 1) as response:
            data = response.read()
        buffer[: len(data)] = data
        self.pos += len(data)
        return len(data)


def fetch_schellenberger(cache: Path, scenario: str) -> list[Path]:
    """Download every subject's ``scenario`` recording into ``cache`` (idempotent)."""
    cache.mkdir(parents=True, exist_ok=True)
    paths = []
    for url in FIGSHARE_ZIPS:
        archive = zipfile.ZipFile(io.BufferedReader(_RangeFile(url), buffer_size=8 << 20))
        for info in archive.infolist():
            if not info.filename.endswith(f"_{scenario}.mat"):
                continue
            dest = cache / Path(info.filename).name
            if not (dest.exists() and dest.stat().st_size == info.file_size):
                part = dest.with_suffix(".part")
                with archive.open(info) as src, part.open("wb") as dst:
                    shutil.copyfileobj(src, dst, 8 << 20)
                part.replace(dest)
                print(f"fetched {dest.name}", file=sys.stderr)
            paths.append(dest)
    return sorted(paths)


@dataclass
class Recording:
    """One synchronised RF + reference recording."""

    subject: str
    source: Path
    fs: float
    rf: dict[str, Any]
    ecg: list[np.ndarray] = field(default_factory=list)
    fs_ecg: float = 0.0
    ppg: np.ndarray | None = None
    fs_ppg: float = 0.0
    resp: np.ndarray | None = None
    fs_resp: float = 0.0


def _scalar(value: Any) -> float:
    return float(np.asarray(value).item())


def _decimate(x: np.ndarray, fs: float) -> tuple[np.ndarray, float]:
    factor = round(fs / ANALYSIS_FS)
    if factor <= 1 or abs(fs / factor - ANALYSIS_FS) > 1e-6:
        return x, fs
    out = x
    for step in (f for f in (4, 5, 2, 3) for _ in range(8)):
        if factor % step == 0:
            out, factor = signal.decimate(out, step, zero_phase=True), factor // step
        if factor == 1:
            break
    if factor != 1:
        out = signal.decimate(out, factor, ftype="fir", zero_phase=True)
    return np.asarray(out, dtype=np.float64), ANALYSIS_FS


def load_schellenberger(path: Path) -> Recording:
    """Load one Schellenberger et al. (2020) ``.mat`` recording."""
    m = sio.loadmat(path)
    fs_radar = _scalar(m["fs_radar"])
    i, fs = _decimate(m["radar_i"].ravel().astype(np.float64), fs_radar)
    q, _ = _decimate(m["radar_q"].ravel().astype(np.float64), fs_radar)
    return Recording(
        subject=path.name.split("_", 1)[0],
        source=path,
        fs=fs,
        rf={"radar_i": i, "radar_q": q, "wavelength_m": SCHELLENBERGER_WAVELENGTH_M},
        ecg=[m["tfm_ecg1"].ravel(), m["tfm_ecg2"].ravel()],
        fs_ecg=_scalar(m["fs_ecg"]),
        resp=m["tfm_z0"].ravel(),
        fs_resp=_scalar(m["fs_z0"]),
    )


_RF_KEYS = (
    "radar_i",
    "radar_q",
    "wavelength_m",
    "radar_displacement_m",
    "radar_phase_rad",
    "csi_amplitude",
    "csi_phase",
    "subcarrier_indices",
)


def load_npz(path: Path) -> Recording:
    """Load an operator recording (keys in the module docstring)."""
    with np.load(path, allow_pickle=False) as z:
        keys = set(z.files)
        if "fs" not in keys:
            raise ValueError(f"{path}: missing 'fs'")
        rf = {k: (_scalar(z[k]) if k == "wavelength_m" else z[k]) for k in _RF_KEYS if k in keys}
        if not (
            {"radar_i", "radar_q"} <= keys
            or keys & {"radar_displacement_m", "radar_phase_rad", "csi_amplitude", "csi_phase"}
        ):
            raise ValueError(f"{path}: no RF input")
        if not keys & {"ecg", "ppg", "resp"}:
            raise ValueError(f"{path}: no reference signal (ecg, ppg or resp)")
        return Recording(
            subject=str(z["subject"]) if "subject" in keys else path.stem,
            source=path,
            fs=_scalar(z["fs"]),
            rf=rf,
            ecg=[z["ecg"]] if "ecg" in keys else [],
            fs_ecg=_scalar(z["fs_ecg"]) if "ecg" in keys else 0.0,
            ppg=z["ppg"] if "ppg" in keys else None,
            fs_ppg=_scalar(z["fs_ppg"]) if "ppg" in keys else 0.0,
            resp=z["resp"] if "resp" in keys else None,
            fs_resp=_scalar(z["fs_resp"]) if "resp" in keys else 0.0,
        )


def _heart_events(rec: Recording) -> tuple[np.ndarray | None, str | None]:
    """Heart reference event times from the cleanest ECG lead, else the PPG."""
    best: tuple[float, np.ndarray, str] | None = None
    for k, lead in enumerate(rec.ecg):
        times = vr.detect_ecg_r_peaks(lead, rec.fs_ecg) / rec.fs_ecg
        quality = vr.valid_intervals(times, vr.HEART_INTERVALS)[1].size / max(1, times.size - 1)
        if best is None or quality > best[0]:
            best = (quality, times, f"ecg{k + 1}")
    if best is not None:
        return best[1], best[2]
    if rec.ppg is not None:
        return vr.detect_pulse_peaks(rec.ppg, rec.fs_ppg) / rec.fs_ppg, "ppg"
    return None, None


def _front_ends(rec: Recording) -> dict[str, dict[str, Any]]:
    """RF inputs to score: I/Q both ways, anything else as given."""
    rf = rec.rf
    if "radar_i" in rf:
        ellipse = fit_iq_ellipse(rf["radar_i"], rf["radar_q"])
        wavelength = rf["wavelength_m"]
        raw = np.unwrap(np.arctan2(rf["radar_q"], rf["radar_i"]))
        return {
            "ellipse": {
                "radar_displacement_m": demodulate_iq(
                    rf["radar_i"], rf["radar_q"], wavelength_m=wavelength, ellipse=ellipse
                ),
                "_ellipse": {"residual": ellipse.residual, "arc_rad": ellipse.arc_rad},
            },
            "raw_arctangent": {"radar_displacement_m": (raw - raw[0]) * wavelength / (4 * np.pi)},
        }
    return {"as_given": dict(rf)}


def score(rec: Recording, window_s: float, step_s: float) -> dict[str, Any]:
    """Per-window estimates and references for every front end of one recording."""
    heart_times, heart_source = _heart_events(rec)
    breath_times = (
        vr.detect_breaths(rec.resp, rec.fs_resp) / rec.fs_resp if rec.resp is not None else None
    )
    out: dict[str, Any] = {
        "subject": rec.subject,
        "heart_reference": heart_source,
        "front_ends": {},
    }
    for name, inputs in _front_ends(rec).items():
        extra = inputs.pop("_ellipse", None)
        track = track_vital_signs(
            fs=rec.fs, consent_confirmed=True, window_s=window_s, step_s=step_s, **inputs
        )
        nan = np.full(track.window_start_s.shape, np.nan)
        hr_ref = (
            vr.windowed_rates(
                heart_times,
                track.window_start_s,
                track.window_end_s,
                vr.HEART_INTERVALS,
                min_coverage=0.7,
            )
            if heart_times is not None
            else nan
        )
        br_ref = (
            vr.windowed_rates(
                breath_times,
                track.window_start_s,
                track.window_end_s,
                vr.BREATH_INTERVALS,
                min_coverage=0.5,
            )
            if breath_times is not None
            else nan
        )
        out["front_ends"][name] = {
            "ellipse": extra,
            "heart": list(zip(track.heart_bpm.tolist(), hr_ref.tolist(), strict=True)),
            "respiration": list(zip(track.respiration_bpm.tolist(), br_ref.tolist(), strict=True)),
            "heart_reasons": Counter(str(r) for r in track.heart_reasons if r is not None),
            "respiration_reasons": Counter(
                str(r) for r in track.respiration_reasons if r is not None
            ),
        }
    return out


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarise(scored: list[dict[str, Any]], resamples: int, seed: int) -> dict[str, Any]:
    """Pooled agreement per front end and vital sign, plus per-subject figures."""
    names = sorted({n for s in scored for n in s["front_ends"]})
    summary: dict[str, Any] = {}
    for name in names:
        entry: dict[str, Any] = {}
        for vital, tol in (("heart", HEART_TOLERANCE_BPM), ("respiration", BREATH_TOLERANCE_BPM)):
            rows = [
                (s["subject"], e, r)
                for s in scored
                if name in s["front_ends"]
                for e, r in s["front_ends"][name][vital]
            ]
            pooled = vr.bootstrap_by_subject(rows, tolerance=tol, resamples=resamples, seed=seed)
            reasons: Counter[str] = Counter()
            for s in scored:
                if name in s["front_ends"]:
                    reasons.update(s["front_ends"][name][f"{vital}_reasons"])
            pooled["abstentions"] = dict(reasons.most_common())
            pooled["per_subject"] = {
                s["subject"]: vr.agreement(
                    [e for e, _ in s["front_ends"][name][vital]],
                    [r for _, r in s["front_ends"][name][vital]],
                    tolerance=tol,
                )
                for s in scored
                if name in s["front_ends"]
            }
            entry[vital] = pooled
        summary[name] = entry
    return summary


def _finite(obj: Any) -> Any:
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    return obj


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--fetch", action="store_true", help="download the public dataset")
    source.add_argument("--data-dir", type=Path, help="directory of Schellenberger .mat files")
    source.add_argument("--npz", type=Path, nargs="+", help="operator recordings")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--scenario", default="Resting")
    parser.add_argument("--window-s", type=float, default=30.0)
    parser.add_argument("--step-s", type=float, default=30.0)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--out", type=Path, default=RESULTS_PATH)
    args = parser.parse_args(argv)
    logging.disable(logging.WARNING)

    started = time.perf_counter()
    if args.npz:
        recordings = [load_npz(p) for p in args.npz]
        dataset = "operator npz"
    else:
        directory = args.cache if args.fetch else args.data_dir
        paths = (
            fetch_schellenberger(directory, args.scenario)
            if args.fetch
            else sorted(directory.glob(f"*_{args.scenario}.mat"))
        )
        if not paths:
            print(f"error: no *_{args.scenario}.mat files in {directory}", file=sys.stderr)
            return 2
        recordings = [load_schellenberger(p) for p in paths]
        dataset = f"Schellenberger et al. 2020 ({args.scenario})"

    scored = []
    for rec in recordings:
        scored.append(score(rec, args.window_s, args.step_s))
        print(f"scored {rec.subject}", file=sys.stderr)
    document = {
        "schema": "mercury.benchmarks.rf_vitals_reference_validation/v1",
        "dataset": dataset,
        "design": {
            "window_s": args.window_s,
            "step_s": args.step_s,
            "analysis_fs": ANALYSIS_FS,
            "heart_tolerance_bpm": HEART_TOLERANCE_BPM,
            "breath_tolerance_bpm": BREATH_TOLERANCE_BPM,
            "resamples": args.resamples,
            "seed": args.seed,
        },
        "inputs": {rec.source.name: _sha256(rec.source) for rec in recordings},
        "heart_reference": {s["subject"]: s["heart_reference"] for s in scored},
        "ellipse_fit": {
            s["subject"]: s["front_ends"]["ellipse"]["ellipse"]
            for s in scored
            if "ellipse" in s["front_ends"]
        },
        "results": summarise(scored, args.resamples, args.seed),
        "elapsed_s": round(time.perf_counter() - started, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(_finite(document), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for name, entry in document["results"].items():
        for vital in ("heart", "respiration"):
            r = entry[vital]
            print(
                f"{name:<15} {vital:<11} coverage {r['coverage']:.2f} {r['coverage_ci95']}  "
                f"MAE {r['mae']:.2f} {r['mae_ci95']}  bias {r['bias']:+.2f}  "
                f"LoA [{r['loa_low']:.2f}, {r['loa_high']:.2f}]  n={r['reported']}/{r['reference_windows']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

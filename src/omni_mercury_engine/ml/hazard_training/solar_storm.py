# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Train the GeomagneticStormPredictor on real OMNI2 solar wind + observed Kp.

Data source (hook #11, ``SolarStormDetector.load_neural_weights``):

* **NASA SPDF OMNI2 hourly archive**
  (``https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_YYYY.dat``) --
  real multi-spacecraft L1 solar wind / IMF measurements paired with the real
  *observed* planetary Kp index for every hour since 1963. This gives exact
  feature/label pairs, unlike event lists (e.g. NASA DONKI GST) which carry
  labels but no aligned upstream solar wind observations; DONKI remains
  documented as an alternative label source in ``docs/``.
* **GFZ Potsdam Kp service** (``https://kp.gfz-potsdam.de/app/json/``) -- the
  authoritative Kp producer, used as an integrity cross-check that our OMNI2
  column parsing recovers the same Kp values for a fixed storm window
  (2023-04-20..30, which contains the 2023-04-23 G4 storm).

Task: nowcast Kp (and storm probability, Kp >= 5) from the same instantaneous
solar wind / IMF observations the detector's deterministic Boyle-index physics
fallback consumes. The merit gate compares the trained network against that
physics fallback *through the public detector API* on held-out test years.

Temporal split (never random -- Kp autocorrelates over days and solar-cycle
years): train 2005-2018, validation 2019-2021, test 2022-2024. The test span
covers the ascent of solar cycle 25 including the 2024-05-10 G5 superstorm.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from omni_mercury_engine.ml.hazard_training.common import (
    EvaluationOutcome,
    PipelineContext,
    TemporalSplit,
    binary_auc,
    cached_fetch,
    candidate_paths,
    save_candidate,
    save_evaluation,
    seed_everything,
    sha256_file,
    ship_checkpoint,
)
from omni_mercury_engine.ml.hazard_training.features import (
    GEOMAG_FEATURE_DIM,
    GEOMAG_FEATURE_NAMES,
    GEOMAG_FEATURE_SPEC_VERSION,
    build_geomag_feature_vector,
)

logger = logging.getLogger(__name__)

HOOK_NAME = "solar_storm_geomag"
CHECKPOINT_NAME = "solar_storm_geomag"

OMNI2_URL_TEMPLATE = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_{year}.dat"
GFZ_KP_URL = (
    "https://kp.gfz.de/app/json/?start=2023-04-20T00:00:00Z" "&end=2023-04-30T00:00:00Z&index=Kp"
)

SPLIT = TemporalSplit(
    train_years=tuple(range(2005, 2019)),
    val_years=(2019, 2020, 2021),
    test_years=(2022, 2023, 2024),
)

# OMNI2 whitespace-token indices (0-based) per the official format description
# (https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2.text).
_COL_YEAR = 0
_COL_DOY = 1
_COL_HOUR = 2
_COL_BMAG = 8  # word 9: field magnitude average |B|, fill 999.9
_COL_BY_GSM = 15  # word 16, fill 999.9
_COL_BZ_GSM = 16  # word 17, fill 999.9
_COL_TEMPERATURE = 22  # word 23, fill 9999999.
_COL_DENSITY = 23  # word 24, fill 999.9
_COL_SPEED = 24  # word 25, fill 9999.
_COL_PRESSURE = 28  # word 29, fill 99.99
_COL_KP = 38  # word 39: Kp * 10 as integer, fill 99

_FILL_ATOL = 1e-6


@dataclass
class Omni2Hourly:
    """Parsed OMNI2 hourly records (NaN where the archive holds fill values)."""

    year: np.ndarray
    doy: np.ndarray
    hour: np.ndarray
    bmag: np.ndarray
    by_gsm: np.ndarray
    bz_gsm: np.ndarray
    temperature: np.ndarray
    density: np.ndarray
    speed: np.ndarray
    pressure: np.ndarray
    kp: np.ndarray


def _nanfill(values: np.ndarray, fill_value: float) -> np.ndarray:
    """Replace the archive's documented fill value with NaN."""
    out = values.astype(np.float64, copy=True)
    out[np.isclose(out, fill_value, rtol=0.0, atol=max(_FILL_ATOL, abs(fill_value) * 1e-6))] = (
        np.nan
    )
    return out


def parse_omni2(paths: list[Any]) -> Omni2Hourly:
    """Parse OMNI2 yearly ``.dat`` files into hourly arrays.

    Args:
        paths: Paths of cached ``omni2_YYYY.dat`` files.

    Returns:
        Concatenated hourly records, chronological, fills mapped to NaN.

    Raises:
        ValueError: If a row has fewer tokens than the Kp column requires
            (would mean the format assumption is wrong -- fail loud).
    """
    rows = []
    for path in paths:
        data = np.loadtxt(path, dtype=np.float64, ndmin=2)
        if data.shape[1] <= _COL_KP:
            raise ValueError(
                f"{path}: OMNI2 row has {data.shape[1]} columns, expected > {_COL_KP}; "
                "format description may have changed -- refusing to guess"
            )
        rows.append(data)
    data = np.vstack(rows)
    return Omni2Hourly(
        year=data[:, _COL_YEAR].astype(np.int64),
        doy=data[:, _COL_DOY].astype(np.int64),
        hour=data[:, _COL_HOUR].astype(np.int64),
        bmag=_nanfill(data[:, _COL_BMAG], 999.9),
        by_gsm=_nanfill(data[:, _COL_BY_GSM], 999.9),
        bz_gsm=_nanfill(data[:, _COL_BZ_GSM], 999.9),
        temperature=_nanfill(data[:, _COL_TEMPERATURE], 9999999.0),
        density=_nanfill(data[:, _COL_DENSITY], 999.9),
        speed=_nanfill(data[:, _COL_SPEED], 9999.0),
        pressure=_nanfill(data[:, _COL_PRESSURE], 99.99),
        kp=_nanfill(data[:, _COL_KP], 99.0) / 10.0,
    )


def _crosscheck_kp_against_gfz(omni: Omni2Hourly, gfz_path: Any) -> dict[str, float]:
    """Verify OMNI2 Kp parsing against the authoritative GFZ Kp service.

    GFZ reports Kp in exact thirds (e.g. 8.333); OMNI2 stores the standard
    one-decimal encodings (83 -> 8.3), so agreement within 0.06 after
    rounding GFZ to one decimal proves the column/scale decoding is right.

    Raises:
        RuntimeError: On any 3-hour bin disagreement > 0.06 -- a parsing bug
            here would silently corrupt every training label.
    """
    payload = json.loads(gfz_path.read_text())
    gfz_kp = np.asarray(payload["Kp"], dtype=np.float64)
    gfz_times = payload["datetime"]

    checked = 0
    worst = 0.0
    for t_iso, kp_ref in zip(gfz_times, gfz_kp, strict=True):
        year = int(t_iso[0:4])
        month, day, hour = int(t_iso[5:7]), int(t_iso[8:10]), int(t_iso[11:13])
        doy = _dt.date(year, month, day).timetuple().tm_yday
        mask = (omni.year == year) & (omni.doy == doy) & (omni.hour == hour)
        if not mask.any():
            continue
        kp_omni = float(omni.kp[mask][0])
        if np.isnan(kp_omni):
            continue
        diff = abs(kp_omni - round(kp_ref, 1))
        worst = max(worst, diff)
        checked += 1
        if diff > 0.06:
            raise RuntimeError(
                f"OMNI2 Kp parsing cross-check FAILED at {t_iso}: parsed {kp_omni} vs "
                f"GFZ {kp_ref:.3f}. The Kp column/scale decoding is wrong; refusing to "
                "train on corrupted labels."
            )
    if checked < 40:
        raise RuntimeError(
            f"OMNI2/GFZ Kp cross-check only matched {checked} bins (expected ~80 for a "
            "10-day window); time alignment is broken."
        )
    logger.info("OMNI2 Kp cross-check vs GFZ: %d bins, worst |diff|=%.3f", checked, worst)
    return {"bins_checked": float(checked), "worst_abs_diff": worst}


def fetch(ctx: PipelineContext) -> dict[str, Any]:
    """Download and integrity-check all OMNI2 years needed by the split.

    Returns:
        Manifest with per-file URLs and SHA-256 digests plus the GFZ
        cross-check summary.
    """
    omni_dir = ctx.data_dir / "omni2"
    sources: list[dict[str, Any]] = []
    paths = []
    for year in SPLIT.all_years:
        url = OMNI2_URL_TEMPLATE.format(year=year)
        path = cached_fetch(url, omni_dir / f"omni2_{year}.dat")
        paths.append(path)
        sources.append(
            {
                "url": url,
                "sha256": sha256_file(path),
                "description": f"NASA SPDF OMNI2 hourly solar wind/IMF/Kp, {year}",
            }
        )
    gfz_path = cached_fetch(GFZ_KP_URL, omni_dir / "gfz_kp_crosscheck.json")
    sources.append(
        {
            "url": GFZ_KP_URL,
            "sha256": sha256_file(gfz_path),
            "description": "GFZ Potsdam definitive Kp, 2023-04-20..30 (parser cross-check)",
        }
    )
    omni = parse_omni2([p for p in paths if "2023" in p.name])
    crosscheck = _crosscheck_kp_against_gfz(omni, gfz_path)

    manifest = {"hook": HOOK_NAME, "sources": sources, "kp_crosscheck": crosscheck}
    manifest_path = omni_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    logger.info("fetch complete: %d OMNI2 years cached under %s", len(paths), omni_dir)
    return manifest


@dataclass
class GeomagDataset:
    """Feature/label matrices with per-sample year for temporal splitting."""

    features: np.ndarray
    kp: np.ndarray
    storm: np.ndarray
    years: np.ndarray
    raw_fields: list[dict[str, float]]
    feature_fill: dict[str, float]
    feature_mean: np.ndarray
    feature_std: np.ndarray


def _load_omni_for_split(ctx: PipelineContext) -> Omni2Hourly:
    """Load every cached OMNI2 year of the split, failing loud on gaps."""
    omni_dir = ctx.data_dir / "omni2"
    paths = []
    for year in SPLIT.all_years:
        path = omni_dir / f"omni2_{year}.dat"
        if not path.exists():
            raise FileNotFoundError(f"missing OMNI2 cache file {path}; run the --fetch stage first")
        paths.append(path)
    return parse_omni2(paths)


def build_dataset(ctx: PipelineContext) -> GeomagDataset:
    """Assemble the feature/label dataset from cached OMNI2 files.

    Rows require finite speed, Bz and Kp (the two primary drivers and the
    label). Optional fields are filled with medians computed from the TRAIN
    years only, and standardization statistics likewise come from the train
    years only -- no leakage from val/test.
    """
    omni = _load_omni_for_split(ctx)
    valid = np.isfinite(omni.speed) & np.isfinite(omni.bz_gsm) & np.isfinite(omni.kp)
    idx = np.flatnonzero(valid)
    if ctx.limit_samples is not None:
        idx = idx[: ctx.limit_samples]

    years = omni.year[idx]
    train_mask, _, _ = SPLIT.masks(years)
    if not train_mask.any():
        raise RuntimeError("no training rows found in the OMNI2 cache; cannot proceed")

    def _train_median(values: np.ndarray) -> float:
        observed = values[idx][train_mask]
        observed = observed[np.isfinite(observed)]
        if observed.size == 0:
            raise RuntimeError("optional field has zero observed training values")
        return float(np.median(observed))

    fill = {
        "by_imf_nt": _train_median(omni.by_gsm),
        "imf_magnitude_nt": _train_median(omni.bmag),
        "proton_density_p_cm3": _train_median(omni.density),
        "proton_temperature_k": _train_median(omni.temperature),
        "flow_pressure_npa": _train_median(omni.pressure),
    }

    def _opt(values: np.ndarray, i: int) -> float | None:
        v = float(values[i])
        return v if np.isfinite(v) else None

    raw_fields: list[dict[str, float]] = []
    feats = np.zeros((idx.size, GEOMAG_FEATURE_DIM), dtype=np.float32)
    for row, i in enumerate(idx):
        fields: dict[str, float] = {
            "solar_wind_speed_km_s": float(omni.speed[i]),
            "bz_imf_nt": float(omni.bz_gsm[i]),
        }
        for key, values in (
            ("by_imf_nt", omni.by_gsm),
            ("imf_magnitude_nt", omni.bmag),
            ("proton_density_p_cm3", omni.density),
            ("proton_temperature_k", omni.temperature),
            ("flow_pressure_npa", omni.pressure),
        ):
            v = _opt(values, int(i))
            if v is not None:
                fields[key] = v
        raw_fields.append(fields)
        feats[row] = build_geomag_feature_vector(fields, fill=fill)

    kp = omni.kp[idx].astype(np.float32)
    mean = feats[train_mask].mean(axis=0)
    std = feats[train_mask].std(axis=0)
    std[std < 1e-6] = 1.0

    return GeomagDataset(
        features=feats,
        kp=kp,
        storm=(kp >= 5.0).astype(np.float32),
        years=years,
        raw_fields=raw_fields,
        feature_fill=fill,
        feature_mean=mean.astype(np.float32),
        feature_std=std.astype(np.float32),
    )


def train(ctx: PipelineContext) -> dict[str, Any]:
    """Train the predictor with early stopping on validation Kp MAE.

    The Kp head is optimized on its pre-clamp linear output (the ``forward``
    clamp to [0, 9] would zero gradients whenever the head starts outside the
    range); inference through ``forward`` is unchanged.

    Returns:
        Training record (epochs run, best validation MAE, sample counts).
    """
    from omni_mercury_engine.space.solar_storm_detector import GeomagneticStormPredictor

    rng = seed_everything(ctx.seed)
    ds = build_dataset(ctx)
    train_mask, val_mask, _ = SPLIT.masks(ds.years)
    x = (ds.features - ds.feature_mean) / ds.feature_std
    x_train = torch.from_numpy(x[train_mask])
    y_kp_train = torch.from_numpy(ds.kp[train_mask])
    y_storm_train = torch.from_numpy(ds.storm[train_mask])
    x_val = torch.from_numpy(x[val_mask])
    y_kp_val = torch.from_numpy(ds.kp[val_mask])

    model = GeomagneticStormPredictor(input_dim=GEOMAG_FEATURE_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    bce = torch.nn.BCELoss()

    storm_frac = float(y_storm_train.mean().item())
    logger.info(
        "training on %d rows (%.2f%% storm), validating on %d rows",
        x_train.shape[0],
        100 * storm_frac,
        x_val.shape[0],
    )

    batch_size = 512
    best_val_mae = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience, bad_epochs = 8, 0
    epochs_run = 0

    for epoch in range(ctx.max_epochs):
        epochs_run = epoch + 1
        model.train()
        perm = torch.from_numpy(rng.permutation(x_train.shape[0]))
        epoch_loss = 0.0
        for start in range(0, x_train.shape[0], batch_size):
            batch_idx = perm[start : start + batch_size]
            if batch_idx.shape[0] < 2:
                continue  # BatchNorm needs >1 sample
            xb = x_train[batch_idx]
            features = model.feature_fusion(xb)
            kp_raw = model.kp_predictor(features).squeeze(-1)
            storm_prob = model.storm_predictor(features).squeeze(-1)
            loss = torch.nn.functional.mse_loss(kp_raw, y_kp_train[batch_idx]) + bce(
                storm_prob.clamp(1e-6, 1 - 1e-6), y_storm_train[batch_idx]
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * batch_idx.shape[0]

        model.eval()
        with torch.no_grad():
            _, kp_pred = model(x_val)
            val_mae = float((kp_pred.squeeze(-1) - y_kp_val).abs().mean().item())
        logger.info(
            "epoch %d: train loss %.4f, val Kp MAE %.4f",
            epoch + 1,
            epoch_loss / x_train.shape[0],
            val_mae,
        )
        if val_mae < best_val_mae - 1e-4:
            best_val_mae = val_mae
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                logger.info("early stop at epoch %d (patience %d)", epoch + 1, patience)
                break

    if best_state is None:
        raise RuntimeError("training produced no finite validation MAE; refusing to save")
    model.load_state_dict(best_state)

    operating_point = _select_operating_point(model, ds, x_val=x_val, val_mask=val_mask)

    record = {
        "seed": ctx.seed,
        "epochs_run": epochs_run,
        "best_val_kp_mae": best_val_mae,
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "train_years": list(SPLIT.train_years),
        "val_years": list(SPLIT.val_years),
        "train_storm_fraction": storm_frac,
        "operating_point": operating_point,
    }
    payload: dict[str, Any] = {
        "geomag_predictor": model.state_dict(),
        "feature_spec": GEOMAG_FEATURE_SPEC_VERSION,
        "feature_names": list(GEOMAG_FEATURE_NAMES),
        "feature_mean": ds.feature_mean.tolist(),
        "feature_std": ds.feature_std.tolist(),
        "feature_fill": ds.feature_fill,
        "operating_point": operating_point,
    }
    save_candidate(ctx.data_dir, HOOK_NAME, payload, record)
    return record


def _boyle_kp(fields: dict[str, float]) -> float:
    """Boyle-index Kp for one observation (parity with the detector physics).

    Mirrors ``SolarStormDetector._predict_geomagnetic_storm_physics`` exactly
    (Boyle et al. 1997 polar-cap potential, empirical log map to Kp); used
    only to compute the physics recall floor for operating-point selection on
    the validation years. The evaluate stage still measures physics through
    the public detector API.
    """
    v = float(fields.get("solar_wind_speed_km_s", 400.0))
    bz = float(fields.get("bz_imf_nt", 0.0))
    by = float(fields.get("by_imf_nt", 0.0))
    b_transverse = float(np.hypot(by, bz))
    clock_angle = float(np.arctan2(abs(by), bz))
    coupling = np.sin(clock_angle / 2.0) ** 3
    boyle_kv = 1e-4 * v**2 + 11.7 * b_transverse * coupling
    return float(np.clip(8.93 * np.log10(max(boyle_kv, 1e-9)) - 12.55, 0.0, 9.0))


def _select_operating_point(
    model: Any,
    ds: GeomagDataset,
    *,
    x_val: torch.Tensor,
    val_mask: np.ndarray,
) -> dict[str, Any]:
    """Choose the storm-onset threshold for the storm-probability head.

    Policy (documented for owner ratification): on the VALIDATION years
    only, require the dual-rule decision ``(kp_pred >= 5) OR
    (storm_prob >= tau)`` to reach a storm recall of at least
    ``max(physics validation recall, 0.55)`` AND a false-alarm rate of at
    most ``0.8 * physics validation FAR`` (the 20% headroom guards the
    val->test distribution shift the ship gate's hard FAR constraint does
    not forgive); among feasible thresholds pick the one maximizing CSI
    (ties -> higher tau, i.e. fewer false alarms). Both selection targets
    mirror the ship gate's secondary constraints — an operating point chosen
    against only one of them can win recall while regressing FAR by a
    rounding error and be refused (the first candidate did exactly that:
    test FAR 3.183% vs physics 3.139%). This machinery exists because the
    MSE-trained Kp point estimate regresses toward the mean on a ~3%-storm
    dataset, so thresholding it at Kp>=5 halves recall versus physics even
    though its ranking (AUC) is far better; the BCE-trained storm head
    carries that ranking and must drive the onset decision.

    Returns:
        Operating-point record stored in the checkpoint payload and the
        provenance sidecar (threshold, policy, and the validation-year
        recall/FAR/CSI for both the learned dual rule and physics).
    """
    val_idx = np.flatnonzero(val_mask)
    storm_true = ds.storm[val_mask].astype(bool)
    if not storm_true.any() or storm_true.all():
        raise RuntimeError(
            "validation years contain a single class; cannot select an " "operating point honestly"
        )

    model.eval()
    with torch.no_grad():
        storm_prob_t, kp_pred_t = model(x_val)
    storm_prob = storm_prob_t.squeeze(-1).numpy().astype(np.float64)
    kp_pred = kp_pred_t.squeeze(-1).numpy().astype(np.float64)

    kp_phys = np.array([_boyle_kp(ds.raw_fields[i]) for i in val_idx])
    phys_detect = kp_phys >= 5.0
    physics_recall = float(np.mean(phys_detect[storm_true]))
    physics_far = float(np.mean(phys_detect[~storm_true]))

    recall_floor = max(physics_recall, 0.55)
    far_ceiling = 0.8 * physics_far
    kp_detect = kp_pred >= 5.0

    def _dual_metrics(tau: float) -> tuple[float, float, float]:
        detect = kp_detect | (storm_prob >= tau)
        tp = float(np.sum(detect & storm_true))
        fn = float(np.sum(~detect & storm_true))
        fp = float(np.sum(detect & ~storm_true))
        recall = tp / max(tp + fn, 1.0)
        far = fp / max(float(np.sum(~storm_true)), 1.0)
        csi = tp / max(tp + fn + fp, 1.0)
        return recall, far, csi

    taus = np.unique(np.quantile(storm_prob, np.linspace(0.0, 1.0, 513)))
    best: dict[str, Any] | None = None
    fallback: dict[str, Any] | None = None
    for tau in taus:
        recall, far, csi = _dual_metrics(float(tau))
        entry = {
            "storm_prob_threshold": float(tau),
            "val_recall": recall,
            "val_far": far,
            "val_csi": csi,
        }
        # Fallback if no threshold satisfies both floors: the most
        # conservative feasible-on-FAR point with the best recall (a
        # recall-maximizing fallback that blows the FAR ceiling would be
        # selecting a point the ship gate is guaranteed to refuse).
        if far <= far_ceiling and (fallback is None or recall > fallback["val_recall"]):
            fallback = entry
        if (
            recall >= recall_floor
            and far <= far_ceiling
            and (
                best is None
                or csi > best["val_csi"]
                or (csi == best["val_csi"] and tau > best["storm_prob_threshold"])
            )
        ):
            best = entry
    floor_met = best is not None
    chosen = best if best is not None else fallback
    if chosen is None:
        raise RuntimeError(
            "no operating point satisfies even the FAR ceiling on validation; "
            "the storm head is not usable for onset decisions -- refusing to "
            "record a doomed operating point"
        )
    return {
        **chosen,
        "policy": "dual-rule (kp_pred>=5 OR storm_prob>=tau); tau maximizes val CSI "
        "subject to val recall >= max(physics val recall, 0.55) AND "
        "val FAR <= 0.8 * physics val FAR",
        "recall_floor": recall_floor,
        "recall_floor_met": floor_met,
        "far_ceiling": far_ceiling,
        "val_recall_physics": physics_recall,
        "val_far_physics": physics_far,
    }


def _g_bucket(kp: float) -> str:
    """NOAA G-scale bucket for a Kp value (matches the detector's mapping)."""
    if kp >= 9:
        return "extreme"
    if kp >= 8:
        return "severe"
    if kp >= 7:
        return "strong"
    if kp >= 6:
        return "moderate"
    if kp >= 5:
        return "minor"
    return "none"


def evaluate(ctx: PipelineContext) -> EvaluationOutcome:
    """Compare learned vs physics through the public detector API.

    Both paths receive the *identical* held-out cases: the raw solar wind /
    IMF observations for every test-year hour. Physics is the detector's
    deterministic Boyle-index fallback; learned is the same detector after
    ``load_neural_weights`` on the candidate checkpoint.

    Returns:
        The evaluation outcome (primary metric: Kp MAE, lower is better).
    """
    from omni_mercury_engine.space.solar_storm_detector import SolarStormDetector

    ds = build_dataset(ctx)
    _, _, test_mask = SPLIT.masks(ds.years)
    test_idx = np.flatnonzero(test_mask)
    if test_idx.size == 0:
        raise RuntimeError("no test rows found; cannot evaluate")

    cand_path, _ = candidate_paths(ctx.data_dir, HOOK_NAME)
    if not cand_path.exists():
        raise FileNotFoundError(f"no candidate checkpoint at {cand_path}; run --train first")

    physics_det = SolarStormDetector(
        enable_flare_detection=False, enable_cme_tracking=False, enable_geomag_prediction=True
    )
    learned_det = SolarStormDetector(
        enable_flare_detection=False, enable_cme_tracking=False, enable_geomag_prediction=True
    )
    learned_det.load_neural_weights(str(cand_path))

    kp_true = ds.kp[test_idx]
    storm_true = ds.storm[test_idx]
    results: dict[str, dict[str, list[float]]] = {
        "physics": {"kp": [], "conf": []},
        "learned": {"kp": [], "conf": []},
    }
    buckets: dict[str, list[str]] = {"physics": [], "learned": []}
    detected: dict[str, list[bool]] = {"physics": [], "learned": []}
    for i in test_idx:
        case = {"magnetosphere_data": dict(ds.raw_fields[i])}
        for label, det in (("physics", physics_det), ("learned", learned_det)):
            out = det.predict_solar_storm(case)
            if out.kp_index is None or not np.isfinite(out.kp_index):
                raise RuntimeError(f"{label} path returned non-finite Kp for case {i}")
            results[label]["kp"].append(float(out.kp_index))
            results[label]["conf"].append(float(out.confidence))
            buckets[label].append(out.geomagnetic_storm_level)
            # The deployed storm-onset decision is the emitted level: physics
            # thresholds its Kp at 5; the learned path applies the dual rule
            # (regressed Kp OR storm-probability >= ratified threshold).
            detected[label].append(out.geomagnetic_storm_level != "none")

    bucket_true = [_g_bucket(float(k)) for k in kp_true]

    def _metrics(label: str) -> dict[str, float]:
        kp_pred = np.asarray(results[label]["kp"])
        conf = np.asarray(results[label]["conf"])
        bucket_pred = buckets[label]
        detect = np.asarray(detected[label], dtype=bool)
        is_storm = storm_true == 1.0
        tp = float(np.sum(detect & is_storm))
        fn = float(np.sum(~detect & is_storm))
        fp = float(np.sum(detect & ~is_storm))
        return {
            "kp_mae": float(np.mean(np.abs(kp_pred - kp_true))),
            "kp_rmse": float(np.sqrt(np.mean((kp_pred - kp_true) ** 2))),
            "g_bucket_accuracy": float(
                np.mean([p == t for p, t in zip(bucket_pred, bucket_true, strict=True)])
            ),
            "storm_auc": binary_auc(storm_true, conf),
            "storm_recall_kp5": float(
                np.mean(kp_pred[is_storm] >= 5.0) if is_storm.any() else np.nan
            ),
            "false_alarm_rate_kp5": float(np.mean(kp_pred[~is_storm] >= 5.0)),
            "storm_recall_op": float(tp / max(tp + fn, 1.0)),
            "false_alarm_rate_op": float(fp / max(float(np.sum(~is_storm)), 1.0)),
            "storm_csi_op": float(tp / max(tp + fn + fp, 1.0)),
        }

    outcome = EvaluationOutcome(
        hook=HOOK_NAME,
        primary_metric="kp_mae",
        higher_is_better=False,
        learned=_metrics("learned"),
        physics=_metrics("physics"),
        n_test_samples=int(test_idx.size),
        test_years=SPLIT.test_years,
        extras={
            "test_storm_fraction": float(storm_true.mean()),
            "comparison": "identical held-out OMNI2 hours through "
            "SolarStormDetector.predict_solar_storm, physics fallback vs loaded checkpoint",
            "operating_point": "learned storm onset uses the dual rule carried by the "
            "checkpoint (see payload['operating_point']); physics onset is its Kp>=5 "
            "threshold — each path is scored on its own deployed decision rule",
        },
        constraints=[
            {
                "metric": "storm_recall_op",
                "higher_is_better": True,
                "description": "storm recall at the deployed operating point must not "
                "regress below physics (the first shipped checkpoint halved it)",
            },
            {
                "metric": "false_alarm_rate_op",
                "higher_is_better": False,
                "description": "false-alarm rate at the deployed operating point must "
                "not exceed physics",
            },
            {
                "metric": "storm_auc",
                "higher_is_better": True,
                "description": "storm-hour ranking quality must not regress",
            },
        ],
    )
    save_evaluation(ctx.data_dir, outcome)
    logger.info(
        "evaluation: learned Kp MAE %.4f vs physics %.4f on %d held-out hours (%s)",
        outcome.learned["kp_mae"],
        outcome.physics["kp_mae"],
        outcome.n_test_samples,
        "LEARNED WINS" if outcome.learned_beats_physics else "PHYSICS WINS",
    )
    return outcome


def ship(ctx: PipelineContext) -> tuple[Any, Any]:
    """Promote the candidate through the merit gate (may refuse loudly)."""
    from omni_mercury_engine.ml.hazard_training.common import load_evaluation

    outcome = load_evaluation(ctx.data_dir, HOOK_NAME)
    manifest_path = ctx.data_dir / "omni2" / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing fetch manifest {manifest_path}; run --fetch first")
    manifest = json.loads(manifest_path.read_text())
    return ship_checkpoint(
        hook=HOOK_NAME,
        checkpoint_name=CHECKPOINT_NAME,
        data_dir=ctx.data_dir,
        outcome=outcome,
        data_sources=manifest["sources"],
        seed=ctx.seed,
        out_dir=ctx.ship_dir,
    )

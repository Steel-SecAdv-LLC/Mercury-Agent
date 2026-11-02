# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Detection quality of the served routes under adversarial conditions, measured.

Wide scope is only worth something if the detectors hold up when the data is
hostile. This suite drives the routes ``mercury-agent serve`` exposes (the
univariate z-score and IQR, the multivariate route, the fusion route and the
detector-tier ensemble) through scenarios built to defeat a naive detector,
and writes what it measures to ``served_routes_results.json`` beside this
file. ``tests/research/test_served_routes_ledger.py`` fails when the committed
ledger no longer matches what the code produces, and pins the floors the
routes must clear. Everything is seeded; a run is reproducible to the digit.

Scenarios (each a 1-D series of ``N`` points, plus a 64-feature matrix for
the width cases):

* ``point_spike``         a lone 8-sigma spike (the easy case, the baseline)
* ``slow_ramp``           a linear drift through the last fifth of the series
                          (the anomaly is a trend, not a point)
* ``level_shift``         the mean steps by 4 sigma for the last third
* ``seasonal_masking``    a strong sinusoid with a spike hidden at a crest,
                          its size the seasonal amplitude itself
* ``variance_burst``      a stretch of tripled variance with a point anomaly
                          inside it
* ``bounded_evasion``     an attacker who spends a fixed L2 budget spreading
                          an 8-sigma spike over its five neighbours, so no
                          single point stands out as much
* ``distribution_shift``  the first half Gaussian, the second half Student-t
                          with 3 degrees of freedom (heavy tails), anomalies
                          planted in the second half only
* ``wide_noise``          a 300 x 64 noise matrix (no anomaly): the false-
                          positive rate at width

Three questions are answered per route:

1. **Shape** -- AUROC of the route's scores against the planted labels, and
   recall at the route's own flags, per scenario.
2. **Rate independence** -- the tier ensemble flags the top ``contamination``
   share by construction; the sweep records the flag rate against the
   parameter on data whose true anomaly fraction is 5%, next to the
   conformal flags (``conformal_alpha``), whose rate is evidence-controlled.
3. **Attribution** -- each tier member's own AUROC, and the ensemble's AUROC
   with the member left out, so the registry can claim only what the data
   supports.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from omni_mercury_engine.api import fusion_weights
from omni_mercury_engine.api.routes.detection import FusionRequest, detect_fusion
from omni_mercury_engine.api.server import (
    MultivariateRequest,
    UnivariateRequest,
    detect_multivariate,
    detect_univariate,
)
from omni_mercury_engine.detectors.detection_tier import run_tier_ensemble

LEDGER = _HERE / "served_routes_results.json"
N = 300
SEEDS = (0, 1)
SENSITIVITY = 0.5
CONTAMINATIONS = (0.01, 0.05, 0.10, 0.20)
TRUE_FRACTION = 0.05


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def auroc(scores: np.ndarray[Any, Any], labels: np.ndarray[Any, Any]) -> float:
    """Area under the ROC curve by the Mann-Whitney statistic (ties at 0.5)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def recall(flags: Any, labels: np.ndarray[Any, Any]) -> float:
    flags = np.asarray(flags).astype(bool)
    labels = np.asarray(labels).astype(bool)
    return float(flags[labels].mean()) if labels.any() else float("nan")


def false_positive_rate(flags: Any, labels: np.ndarray[Any, Any]) -> float:
    flags = np.asarray(flags).astype(bool)
    labels = np.asarray(labels).astype(bool)
    return float(flags[~labels].mean()) if (~labels).any() else float("nan")


def _r(x: float) -> float:
    return float(round(x, 4))


# ---------------------------------------------------------------------------
# scenarios
# ---------------------------------------------------------------------------


def point_spike(rng: np.random.Generator) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    x = rng.standard_normal(N)
    labels = np.zeros(N, dtype=bool)
    at = int(rng.integers(20, N - 20))
    x[at] += 8.0
    labels[at] = True
    return x, labels


def slow_ramp(rng: np.random.Generator) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    x = rng.standard_normal(N)
    labels = np.zeros(N, dtype=bool)
    start = N - N // 5
    x[start:] += np.linspace(0.0, 6.0, N - start)
    labels[start + (N - start) // 2 :] = True  # the second half of the ramp is unmistakably away
    return x, labels


def level_shift(rng: np.random.Generator) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    # The shifted regime is the last third: a minority, so "anomalous" is
    # well defined for an unsupervised detector (a shift covering most of
    # the series would make the pre-shift rows the departure).
    x = rng.standard_normal(N)
    labels = np.zeros(N, dtype=bool)
    at = 2 * N // 3
    x[at:] += 4.0
    labels[at:] = True
    return x, labels


def seasonal_masking(
    rng: np.random.Generator,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    t = np.arange(N)
    season = 3.0 * np.sin(2 * np.pi * t / 50.0)
    x = season + 0.5 * rng.standard_normal(N)
    labels = np.zeros(N, dtype=bool)
    crest = int(np.argmax(season[100:200])) + 100  # a crest: the spike rides the season
    x[crest] += 3.0  # one seasonal amplitude: invisible to a global z, plain to a local one
    labels[crest] = True
    return x, labels


def variance_burst(
    rng: np.random.Generator,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    x = rng.standard_normal(N)
    labels = np.zeros(N, dtype=bool)
    x[120:180] *= 3.0
    at = 150
    x[at] = 12.0  # inside the burst: 4 burst-sigma, 12 global-sigma
    labels[at] = True
    return x, labels


def bounded_evasion(
    rng: np.random.Generator,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    x = rng.standard_normal(N)
    labels = np.zeros(N, dtype=bool)
    at = int(rng.integers(20, N - 20))
    # The attacker has the spike's energy (8 sigma) and spreads it over the
    # point and its two neighbours on each side with a fixed L2 budget: each
    # of the five carries 8 / sqrt(5) = 3.58 sigma.
    spread = 8.0 / np.sqrt(5.0)
    x[at - 2 : at + 3] += spread
    labels[at - 2 : at + 3] = True
    return x, labels


def distribution_shift(
    rng: np.random.Generator,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    x = np.concatenate([rng.standard_normal(N // 2), rng.standard_t(3, N - N // 2) / np.sqrt(3.0)])
    labels = np.zeros(N, dtype=bool)
    for at in (N // 2 + 40, N // 2 + 110):
        x[at] += 7.0
        labels[at] = True
    return x, labels


SCENARIOS = {
    "point_spike": point_spike,
    "slow_ramp": slow_ramp,
    "level_shift": level_shift,
    "seasonal_masking": seasonal_masking,
    "variance_burst": variance_burst,
    "bounded_evasion": bounded_evasion,
    "distribution_shift": distribution_shift,
}


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------


def _univariate(x: np.ndarray[Any, Any], method: str) -> tuple[np.ndarray[Any, Any], Any]:
    request = UnivariateRequest(data=x.tolist(), sensitivity=SENSITIVITY, method=method)
    response = asyncio.run(detect_univariate(request))
    return np.asarray(response.scores, dtype=float), response.anomalies


def _multivariate(matrix: np.ndarray[Any, Any]) -> tuple[np.ndarray[Any, Any], Any]:
    request = MultivariateRequest(data=matrix.tolist(), sensitivity=SENSITIVITY)
    response = asyncio.run(detect_multivariate(request))
    return np.asarray(response.scores, dtype=float), response.anomalies


def _fusion_rows(matrix: np.ndarray[Any, Any]) -> tuple[np.ndarray[Any, Any], Any]:
    """Per-row fused scores as the batch job computes them: residuals, every component, decisive rule."""
    residual, _ = fusion_weights.deseasonalize(matrix)
    location, scale = fusion_weights.column_statistics(residual)
    parts = fusion_weights.row_components(
        residual, location, scale, dof=fusion_weights.null_dof(matrix.shape[0])
    )
    names = ["statistical", "temporal", "collective", "dimensional"]
    scores = np.asarray(
        [
            fusion_weights.fuse_calibrated({n: float(parts[n][i]) for n in names}, names)
            for i in range(matrix.shape[0])
        ],
        dtype=float,
    )
    threshold = 2.0 + (1.0 - SENSITIVITY) * 3.0
    return scores, scores > threshold


def _fusion_series(matrix: np.ndarray[Any, Any]) -> dict[str, Any]:
    request = FusionRequest(
        data=matrix.tolist(),
        detectors=["statistical", "temporal", "collective", "dimensional", "spectral"],
        sensitivity=SENSITIVITY,
    )
    response = asyncio.run(detect_fusion(request, None))
    return {"is_anomaly": bool(response.is_anomaly), "fused_score": _r(response.fused_score)}


def _tier(x: np.ndarray[Any, Any], contamination: float, alpha: float | None) -> dict[str, Any]:
    return run_tier_ensemble(
        x,
        contamination=contamination,
        method="average",
        include_attribution=True,
        conformal_alpha=alpha,
    )


# ---------------------------------------------------------------------------
# the campaign
# ---------------------------------------------------------------------------


def shape_results() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, make in SCENARIOS.items():
        per_route: dict[str, list[dict[str, float]]] = {}
        series_flags: list[bool] = []
        for seed in SEEDS:
            x, labels = make(np.random.default_rng(seed))
            matrix = x.reshape(-1, 1)
            routes: dict[str, tuple[np.ndarray[Any, Any], Any]] = {
                "univariate_zscore": _univariate(x, "zscore"),
                "univariate_iqr": _univariate(x, "iqr"),
                "multivariate": _multivariate(matrix),
                "fusion_rows": _fusion_rows(matrix),
            }
            tier = _tier(x, 0.05, None)
            routes["tier_average"] = (np.asarray(tier["scores"], dtype=float), tier["flags"])
            for route, (scores, flags) in routes.items():
                per_route.setdefault(route, []).append(
                    {
                        "auroc": _r(auroc(scores, labels)),
                        "recall": _r(recall(flags, labels)),
                        "fpr": _r(false_positive_rate(flags, labels)),
                    }
                )
            series_flags.append(_fusion_series(matrix)["is_anomaly"])
        out[name] = {
            route: {
                "auroc": _r(float(np.mean([m["auroc"] for m in runs]))),
                "recall": _r(float(np.mean([m["recall"] for m in runs]))),
                "fpr": _r(float(np.mean([m["fpr"] for m in runs]))),
            }
            for route, runs in per_route.items()
        }
        out[name]["fusion_series_flagged"] = int(sum(series_flags))
        out[name]["runs"] = len(SEEDS)
    return out


def width_results() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for width in (1, 8, 64):
        flagged = 0
        rows = 0
        series_flags = 0
        for seed in SEEDS:
            noise = np.random.default_rng(100 + seed).standard_normal((N, width))
            _, flags = _multivariate(noise)
            flagged += int(np.sum(flags))
            rows += N
            series_flags += int(_fusion_series(noise)["is_anomaly"])
        out[str(width)] = {
            "multivariate_fpr": _r(flagged / rows),
            "fusion_series_flagged": series_flags,
            "runs": len(SEEDS),
        }
    return out


def rate_results() -> dict[str, Any]:
    """Flag rate against the contamination parameter on data with a 5% true anomaly share."""
    rng = np.random.default_rng(7)
    x = rng.standard_normal(N)
    labels = np.zeros(N, dtype=bool)
    positions = rng.choice(np.arange(10, N - 10), size=int(TRUE_FRACTION * N), replace=False)
    x[positions] += rng.uniform(5.0, 9.0, size=positions.size)
    labels[positions] = True
    sweep: dict[str, Any] = {}
    for c in CONTAMINATIONS:
        tier = _tier(x, c, 0.05)
        sweep[str(c)] = {
            "flag_rate": _r(float(np.mean(tier["flags"]))),
            "recall": _r(recall(tier["flags"], labels)),
            "fpr": _r(false_positive_rate(tier["flags"], labels)),
            "conformal_flag_rate": _r(float(np.mean(tier["conformal_flags"]))),
            "conformal_fpr": _r(false_positive_rate(tier["conformal_flags"], labels)),
        }
    rates = np.array([sweep[str(c)]["flag_rate"] for c in CONTAMINATIONS])
    params = np.array(CONTAMINATIONS)
    slope = float(np.polyfit(params, rates, 1)[0])
    return {
        "true_fraction": TRUE_FRACTION,
        "sweep": sweep,
        # 1.0 means the flag count is the parameter; 0.0 means it is the data.
        "flag_rate_slope_vs_contamination": _r(slope),
    }


def attribution_results() -> dict[str, Any]:
    """Each member's own AUROC and the ensemble's AUROC without it, over the shape scenarios."""
    own: dict[str, list[float]] = {}
    without: dict[str, list[float]] = {}
    full: list[float] = []
    for name, make in SCENARIOS.items():
        x, labels = make(np.random.default_rng(0))
        tier = _tier(x, 0.05, None)
        members = list(tier["detector_names"])
        matrix = np.asarray(tier["per_detector_scores"], dtype=float)
        ensemble = matrix.mean(axis=1)
        full.append(auroc(ensemble, labels))
        for j, member in enumerate(members):
            own.setdefault(member, []).append(auroc(matrix[:, j], labels))
            rest = np.delete(matrix, j, axis=1).mean(axis=1)
            without.setdefault(member, []).append(auroc(rest, labels))
    full_mean = float(np.nanmean(full))
    members_out = {
        member: {
            "own_auroc": _r(float(np.nanmean(own[member]))),
            "ensemble_without": _r(float(np.nanmean(without[member]))),
            # Positive: the ensemble is better with the member than without it.
            "marginal_gain": _r(full_mean - float(np.nanmean(without[member]))),
        }
        for member in own
    }
    return {"ensemble_auroc": _r(full_mean), "members": members_out}


#: Every file whose content decides the numbers; the ledger records each one's
#: git blob hash, and the freshness test recomputes them, so a change to any of
#: them forces this suite to be re-run before the ledger is believed.
INPUTS = (
    "research/adversarial/served_routes.py",
    "src/omni_mercury_engine/api/fusion_weights.py",
    "src/omni_mercury_engine/api/routes/detection.py",
    "src/omni_mercury_engine/api/routes/batch.py",
    "src/omni_mercury_engine/api/server.py",
    "src/omni_mercury_engine/detectors/detection_tier.py",
)


def blob_hash(relative: str) -> str:
    """The git blob id of a repository file, computed without git."""
    content = (_REPO / relative).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(content) + content).hexdigest()  # noqa: S324 - git id


def provenance() -> dict[str, Any]:
    blob = blob_hash("src/omni_mercury_engine/api/fusion_weights.py")
    git = shutil.which("git")
    commit = ""
    if git is not None:
        try:
            commit = subprocess.run(  # noqa: S603 - fixed argv, no shell
                [git, "rev-parse", "HEAD"],
                cwd=_REPO,
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
        except OSError:
            commit = ""
    return {
        "kernel_blob": blob,
        "inputs": {relative: blob_hash(relative) for relative in INPUTS},
        "commit": commit or None,
        "n": N,
        "seeds": list(SEEDS),
    }


def build() -> dict[str, Any]:
    return {
        "provenance": provenance(),
        "shape": shape_results(),
        "width": width_results(),
        "rate_independence": rate_results(),
        "attribution": attribution_results(),
    }


def comparable(ledger: dict[str, Any]) -> dict[str, Any]:
    """The ledger without the fields that change between identical runs."""
    out = json.loads(json.dumps(ledger))
    out.get("provenance", {}).pop("commit", None)
    return out


DOC = _REPO / "docs" / "DETECTION_QUALITY.md"


def render_doc(ledger: dict[str, Any]) -> str:
    """The measured tables as a page, regenerated from the ledger, never typed."""
    lines = [
        "# Detection quality of the served routes under adversarial conditions",
        "",
        "Measured, not asserted: every number here is produced by",
        "`python research/adversarial/served_routes.py` from seeded scenarios built to",
        "defeat a naive detector, and `tests/research/test_served_routes_ledger.py`",
        "fails when the committed ledger (`research/adversarial/served_routes_results.json`)",
        "no longer matches what the code produces. Run the script to refresh both.",
        "",
        f"Series length {ledger['provenance']['n']}, seeds {ledger['provenance']['seeds']}; "
        f"kernel blob `{ledger['provenance']['kernel_blob'][:12]}`.",
        "",
        "## Shape: AUROC / recall at the route's flags / false-positive rate",
        "",
    ]
    routes = ["univariate_zscore", "univariate_iqr", "multivariate", "fusion_rows", "tier_average"]
    lines.append("| Scenario | " + " | ".join(routes) + " | fusion series verdict |")
    lines.append("|---|" + "---|" * (len(routes) + 1))
    for name, row in ledger["shape"].items():
        cells = [
            f"{row[r]['auroc']:.3f} / {row[r]['recall']:.2f} / {row[r]['fpr']:.3f}" for r in routes
        ]
        lines.append(
            f"| `{name}` | "
            + " | ".join(cells)
            + f" | {row['fusion_series_flagged']} of {row['runs']} flagged |"
        )
    lines += [
        "",
        "Scenarios: a lone 8-sigma spike; a linear ramp through the last fifth (labels on",
        "its second half); a 4-sigma level shift over the last third; a spike of one",
        "seasonal amplitude on a crest of a strong season; a point anomaly inside a stretch",
        "of tripled variance; an 8-sigma spike spread by an attacker over five rows;",
        "Gaussian first half, heavy-tailed (t3) second half with two planted spikes.",
        "",
        "## Width: false positives on pure noise",
        "",
        "| Width | multivariate per-row FPR | fusion series flagged |",
        "|---|---|---|",
    ]
    for width, row in ledger["width"].items():
        lines.append(
            f"| {width} | {row['multivariate_fpr']:.4f} | {row['fusion_series_flagged']} of {row['runs']} |"
        )
    rate = ledger["rate_independence"]
    lines += [
        "",
        "The nominal per-row rate at the default sensitivity is 0.00047 (two-sided z 3.5).",
        "",
        "## Rate independence: the detector tier's flags against `contamination`",
        "",
        f"True anomaly fraction {rate['true_fraction']:.2f}. The flag rate's slope against the",
        f"parameter is **{rate['flag_rate_slope_vs_contamination']:.2f}** (1.0: the flag count is",
        "the parameter, by construction; the tier thresholds at the contamination quantile).",
        "The conformal flags (`conformal_alpha` = 0.05) are evidence-controlled: their rate",
        "does not move with the parameter.",
        "",
        "| contamination | flag rate | recall | FPR | conformal flag rate | conformal FPR |",
        "|---|---|---|---|---|---|",
    ]
    for c, row in rate["sweep"].items():
        lines.append(
            f"| {c} | {row['flag_rate']:.3f} | {row['recall']:.3f} | {row['fpr']:.3f} | "
            f"{row['conformal_flag_rate']:.3f} | {row['conformal_fpr']:.3f} |"
        )
    attribution = ledger["attribution"]
    lines += [
        "",
        "## Attribution: what each tier member adds over the rest",
        "",
        f"Ensemble (average of all members) AUROC over the shape scenarios: "
        f"**{attribution['ensemble_auroc']:.3f}**. A negative marginal gain means the",
        "ensemble scores higher without the member on these shapes.",
        "",
        "| Member | own AUROC | ensemble without it | marginal gain |",
        "|---|---|---|---|",
    ]
    for member, row in sorted(
        attribution["members"].items(), key=lambda kv: -kv[1]["marginal_gain"]
    ):
        lines.append(
            f"| `{member}` | {row['own_auroc']:.3f} | {row['ensemble_without']:.3f} | "
            f"{row['marginal_gain']:+.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--check", action="store_true", help="exit 1 if the ledger or page is stale"
    )
    args = parser.parse_args(argv)
    fresh = build()
    if args.check:
        current = json.loads(LEDGER.read_text(encoding="utf-8")) if LEDGER.exists() else {}
        stale = comparable(current) != comparable(fresh)
        page_stale = not DOC.exists() or DOC.read_text(encoding="utf-8") != render_doc(current)
        if stale or page_stale:
            what = " and ".join(
                n for n, flag in ((LEDGER.name, stale), (DOC.name, page_stale)) if flag
            )
            print(f"STALE: {what} differ from what the code produces; re-run this script.")
            return 1
        print("ledger and page current")
        return 0
    LEDGER.write_text(json.dumps(fresh, indent=1) + "\n", encoding="utf-8")
    DOC.write_text(render_doc(fresh), encoding="utf-8")
    print(f"wrote {LEDGER.relative_to(_REPO)} and {DOC.relative_to(_REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

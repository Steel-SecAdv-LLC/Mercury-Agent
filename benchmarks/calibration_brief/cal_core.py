"""Independent reimplementation of the Mercury calibration-alignment core.

Everything here is rebuilt *from the formulas in the validation brief* -- it does
not import the repo's calibration code except where a claim is explicitly about
Mercury's own classes (see ``load_mercury_calibrators``).  Independent
reimplementation IS the validation (brief, Purpose clause).

Operating rules honoured here:
  R5 determinism  -- all randomness flows through ``np.random.default_rng(seed)``.
  R3 no magic     -- constants used by a claim are passed in, never hidden.
  Metric defs     -- implemented exactly as Part 1.0 of the brief specifies.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types
from dataclasses import dataclass

import numpy as np
from scipy import optimize
from scipy.stats import rankdata

# Brief 1.0: clip all scores to this range.  Coarser clipping (1e-6) creates
# float ties and perturbs AUROC; at 1e-12 the perturbation is exactly 0.
CLIP_LO = 1e-12
CLIP_HI = 1.0 - 1e-12

# Golden ratio (used only by the Phi-related claims/experiments, never hidden).
PHI = (1.0 + np.sqrt(5.0)) / 2.0


ADBENCH_BASE = ("https://raw.githubusercontent.com/Minqi824/ADBench/main/"
                "adbench/datasets/Classical/")
ADBENCH = ["6_cardio", "23_mammography", "38_thyroid", "31_satimage-2",
           "28_pendigits", "30_satellite"]


def ensure_datasets(out: str = "data") -> None:
    """Download the six ADBench datasets if absent (retry w/ exponential backoff)."""
    import time
    import urllib.request

    p = pathlib.Path(out)
    p.mkdir(exist_ok=True)
    for nm in ADBENCH:
        dst = p / f"{nm}.npz"
        if dst.exists():
            continue
        for attempt in range(4):
            try:
                with urllib.request.urlopen(ADBENCH_BASE + nm + ".npz", timeout=60) as r:
                    dst.write_bytes(r.read())
                break
            except Exception:  # noqa: BLE001 - network retry
                time.sleep(2 ** attempt)


def clip01(p: np.ndarray) -> np.ndarray:
    return np.clip(p, CLIP_LO, CLIP_HI)


def logit(p: np.ndarray | float) -> np.ndarray:
    p = clip01(np.asarray(p, dtype=float))
    return np.log(p / (1.0 - p))


def sigmoid(z: np.ndarray | float) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


# --------------------------------------------------------------------------- #
# Metrics (Brief 1.0 -- implement exactly).
# --------------------------------------------------------------------------- #
def auroc(y: np.ndarray, p: np.ndarray) -> float:
    """Mann-Whitney AUROC with average ranks for ties."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    n1 = float(np.sum(y == 1))
    n0 = float(np.sum(y == 0))
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = rankdata(p, method="average")
    sum_ranks_pos = float(np.sum(ranks[y == 1]))
    return (sum_ranks_pos - n1 * (n1 + 1.0) / 2.0) / (n1 * n0)


def brier(y: np.ndarray, p: np.ndarray) -> float:
    """Mean (p - y)^2."""
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def nll(y: np.ndarray, p: np.ndarray) -> float:
    """Mean negative log-likelihood (binary)."""
    p = clip01(np.asarray(p, dtype=float))
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error, 15 equal-mass (quantile) bins.

    sum_b w_b * |mean(p_b) - mean(y_b)|.  Equal mass realised by splitting the
    score-sorted samples into ``n_bins`` contiguous chunks (array_split), which
    is exact equal-mass up to +/-1 and is invariant to score ties at chunk
    interiors.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n == 0:
        return float("nan")
    order = np.argsort(p, kind="stable")
    total = 0.0
    for chunk in np.array_split(order, n_bins):
        if len(chunk) == 0:
            continue
        w = len(chunk) / n
        total += w * abs(float(np.mean(p[chunk])) - float(np.mean(y[chunk])))
    return total


def net_benefit(y: np.ndarray, p: np.ndarray, t: float) -> float:
    """NB(t) = TP/n - (FP/n) * t/(1-t), decision = 1[p >= t]."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    n = len(y)
    pos = p >= t
    tp = float(np.sum(pos & (y == 1)))
    fp = float(np.sum(pos & (y == 0)))
    return tp / n - (fp / n) * (t / (1.0 - t))


def net_benefit_treat_all(prevalence: float, t: float) -> float:
    """Treat-all reference NB_all(t) = pi - (1-pi) * t/(1-t)."""
    return prevalence - (1.0 - prevalence) * (t / (1.0 - t))


NB_GRID = (0.05, 0.10, 0.15, 0.20)


# --------------------------------------------------------------------------- #
# Beta calibration (Brief 1.1).
#   c_theta(s) = sigmoid(a*ln s - b*ln(1-s) + c),  a,b >= 0,  theta_id=(1,1,0)
#   fit: min  mean NLL + rho * ||theta - theta_id||^2,  rho = 1e-3   (L-BFGS-B)
# --------------------------------------------------------------------------- #
@dataclass
class BetaCalibrator:
    a: float = 1.0
    b: float = 1.0
    c: float = 0.0
    rho: float = 1e-3
    fitted: bool = False

    @staticmethod
    def _features(s: np.ndarray) -> np.ndarray:
        s = clip01(np.asarray(s, dtype=float))
        return np.column_stack([np.log(s), -np.log(1.0 - s), np.ones_like(s)])

    def fit(self, s: np.ndarray, y: np.ndarray) -> "BetaCalibrator":
        x = self._features(s)
        y = np.asarray(y, dtype=float)
        n = len(y)
        theta_id = np.array([1.0, 1.0, 0.0])
        rho = self.rho

        def obj(theta: np.ndarray) -> tuple[float, np.ndarray]:
            z = x @ theta
            p = sigmoid(z)
            pc = clip01(p)
            loss = -np.mean(y * np.log(pc) + (1.0 - y) * np.log(1.0 - pc))
            loss += rho * float(np.sum((theta - theta_id) ** 2))
            grad = x.T @ (p - y) / n + 2.0 * rho * (theta - theta_id)
            return float(loss), grad

        res = optimize.minimize(
            obj,
            x0=theta_id.copy(),
            method="L-BFGS-B",
            jac=True,
            bounds=[(0.0, None), (0.0, None), (None, None)],
            options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-10},
        )
        self.a, self.b, self.c = (float(v) for v in res.x)
        self.fitted = True
        return self

    def calibrate(self, s: np.ndarray) -> np.ndarray:
        x = self._features(s)
        return sigmoid(x @ np.array([self.a, self.b, self.c]))


def prevalence_shift(p: np.ndarray, pi_src: float, pi_tgt: float) -> np.ndarray:
    """Label-shift adjustment (frozen score): logit(p') = logit(p) + logit(pi_tgt) - logit(pi_src)."""
    return sigmoid(logit(p) + logit(pi_tgt) - logit(pi_src))


# --------------------------------------------------------------------------- #
# Non-parametric remaps used by V8 / X-items.
# --------------------------------------------------------------------------- #
def conditional_mean_remap(
    s_cal: np.ndarray, y_cal: np.ndarray, s_eval: np.ndarray, n_bins: int = 50
) -> np.ndarray:
    """Equal-mass binned conditional-mean remap (non-monotone).

    Fit bin edges + per-bin mean(y) on the cal set; map eval scores through the
    bin they fall into.  This is the V8 'recovery' map (can invert a bijective
    but non-monotone scramble that monotone calibrators cannot).
    """
    s_cal = np.asarray(s_cal, dtype=float)
    y_cal = np.asarray(y_cal, dtype=float)
    order = np.argsort(s_cal, kind="stable")
    chunks = np.array_split(order, n_bins)
    edges = [-np.inf]
    means = []
    for ch in chunks:
        if len(ch) == 0:
            continue
        means.append(float(np.mean(y_cal[ch])))
        edges.append(float(s_cal[ch[-1]]))
    edges[-1] = np.inf
    edges_arr = np.array(edges)
    means_arr = np.array(means)
    idx = np.clip(np.searchsorted(edges_arr, np.asarray(s_eval, dtype=float), side="right") - 1,
                  0, len(means_arr) - 1)
    return means_arr[idx]


# --------------------------------------------------------------------------- #
# Synthetic world (Brief 1.2).
# --------------------------------------------------------------------------- #
@dataclass
class SynthWorld:
    z: np.ndarray
    y: np.ndarray
    s: np.ndarray          # deployed (miscalibrated) score
    c_star: np.ndarray     # oracle conditional probability c*(z)


def make_synth(
    n: int,
    pi: float = 0.15,
    gamma: float = 2.2,
    delta: float = 0.8,
    pi_score: float | None = None,
    seed: int = 7,
) -> SynthWorld:
    """z|y=1 ~ N(+1,1); z|y=0 ~ N(-1,1).  Oracle c*(z)=sigmoid(2z+logit pi).

    Deployed score s = sigmoid(gamma*(2z + logit pi_score) + delta).  pi_score
    is the *training* prevalence frozen into the score; if None it equals pi
    (the score was trained in this environment).
    """
    if pi_score is None:
        pi_score = pi
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < pi).astype(int)
    z = rng.normal(loc=np.where(y == 1, 1.0, -1.0), scale=1.0)
    lpi = float(np.log(pi / (1.0 - pi)))
    lpis = float(np.log(pi_score / (1.0 - pi_score)))
    c_star = sigmoid(2.0 * z + lpi)
    s = sigmoid(gamma * (2.0 * z + lpis) + delta)
    return SynthWorld(z=z, y=y, s=clip01(s), c_star=clip01(c_star))


def oracle_brier_bound(c_star: np.ndarray) -> float:
    """E[c*(1-c*)] -- the irreducible Brier floor (G4 ceiling)."""
    return float(np.mean(c_star * (1.0 - c_star)))


# --------------------------------------------------------------------------- #
# Loader for Mercury's own calibrators (Brief 1.1 import recipe).
# The package __init__ enforces a mandatory PQC gate that raises on import, so
# we register stub parent packages and load the two modules by file path.
# --------------------------------------------------------------------------- #
def load_mercury_calibrators(repo_root: str | pathlib.Path | None = None):
    """Return the Mercury calibration module (PlattScaling, IsotonicCalibration,
    TemperatureScaling, CalibrationEnsemble) loaded standalone."""
    if repo_root is None:
        repo_root = pathlib.Path(__file__).resolve().parents[2]
    root = pathlib.Path(repo_root) / "src" / "omni_mercury_engine"
    for name in ["omni_mercury_engine", "omni_mercury_engine.ml", "omni_mercury_engine.core"]:
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = []  # mark as package
            sys.modules[name] = mod

    def _load(modname: str, relpath: str):
        spec = importlib.util.spec_from_file_location(modname, root / relpath)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[modname] = mod
        spec.loader.exec_module(mod)
        return mod

    mlmod = _load("omni_mercury_engine.ml.mercury_ml", "ml/mercury_ml.py")
    sys.modules["omni_mercury_engine.ml"].mercury_ml = mlmod  # type: ignore[attr-defined]
    return _load("omni_mercury_engine.core.calibration", "core/calibration.py")

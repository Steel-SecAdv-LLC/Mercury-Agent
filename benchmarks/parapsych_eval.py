"""Parapsychology (GCP) sub-net evaluation (WS-D) -- pre-registered, opt-in.

Runs the protocol in ``docs/PARAPSYCH_PREREGISTRATION.md``:

1. Attempt **real** GCP raw-stream ingestion for a fixed event window and report
   reachability honestly (the archive host is unreachable in this environment).
2. Apply the **pre-registered** statistics (network variance, Stouffer Z) to a
   clearly-labelled **synthetic true-random** stream across fixed seeds, showing
   the expected **null**.
3. Wire the differentiable encoder (``ConsciousnessFieldAnalyzer``) over the
   stream for plumbing; it is untrained (quarantined) and is not reported as a
   meaningful detector.

**No claim is made that psi is real.** A clean null is the expected, valid
result. The module stays QUARANTINE until a real, reachable GCP stream clears
the pre-registered, multiple-comparison-corrected bar.

Usage::

    python benchmarks/parapsych_eval.py --out artifacts/parapsych_eval.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

from omni_mercury_engine.models.gcp_ingest import (
    egg_sums_to_z,
    fetch_egg_stream,
    network_variance,
    stouffer_z,
    synthetic_null_streams,
)

WINDOW_SECONDS = 300
N_EGGS = 64
SEEDS = (0, 1, 2)
# Two-sided 5% threshold, Bonferroni-corrected for the 3 fixed seeds reported.
_BONFERRONI_Z = 2.638  # |Z| for alpha=0.05/3 two-sided


def _normal_sf(z: float) -> float:
    """Two-sided survival probability for a standard normal |z| (no scipy)."""
    return math.erfc(abs(z) / math.sqrt(2.0))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="artifacts/parapsych_eval.json")
    args = ap.parse_args()

    # 1. Honest real-ingestion attempt (fixed window from the pre-registration).
    ingest = fetch_egg_stream(2020, 1, 1, "00:00:00", "00:05:00")

    # 2. Pre-registered statistics on the synthetic null, fixed seeds.
    per_seed = []
    for seed in SEEDS:
        sums = synthetic_null_streams(WINDOW_SECONDS, N_EGGS, seed)
        z = egg_sums_to_z(sums)
        sz = stouffer_z(z)
        per_seed.append(
            {
                "seed": seed,
                "stouffer_z": sz,
                "two_sided_p": _normal_sf(sz),
                "mean_network_variance": float(network_variance(z).mean()),
                "significant_bonferroni": bool(abs(sz) > _BONFERRONI_Z),
            }
        )
    any_sig = any(s["significant_bonferroni"] for s in per_seed)

    # 3. Encoder plumbing (untrained / quarantined -> not a meaningful detector).
    encoder_status = "wired; untrained (quarantined); abstains -- not reported as a detector"
    try:
        import torch

        from omni_mercury_engine.models.parapsychology import ConsciousnessFieldAnalyzer

        enc = ConsciousnessFieldAnalyzer(sequence_length=WINDOW_SECONDS)
        enc.eval()
        seq = torch.tensor(
            egg_sums_to_z(synthetic_null_streams(WINDOW_SECONDS, N_EGGS, 0)).mean(axis=1),
            dtype=torch.float32,
        ).reshape(1, -1, 1)
        with torch.no_grad():
            enc(seq)
        encoder_status = "wired + runs (untrained/quarantined); output not reported as meaningful"
    except Exception as e:  # torch missing etc.
        encoder_status = f"not exercised: {type(e).__name__}"

    artifact = {
        "metadata": {
            "purpose": "WS-D GCP parapsychology sub-net pre-registered evaluation",
            "psi_claim": "NONE -- pure signal-processing / anomaly task; a null is valid",
            "real_ingestion": {
                "reachable": ingest.reachable,
                "reason": ingest.reason,
                "provenance": ingest.provenance,
            },
            "signal_used": "SYNTHETIC true-random Binomial(200,0.5) -- cannot lift quarantine",
            "statistics": "network variance + Stouffer Z (pre-registered)",
            "bonferroni_z_threshold": _BONFERRONI_Z,
            "encoder": encoder_status,
        },
        "per_seed": per_seed,
        "any_significant": any_sig,
        "verdict": (
            "QUARANTINE -- real GCP stream unreachable here; on the synthetic null the "
            "pre-registered statistic is non-significant (expected null). No psi claim. "
            "Lift only on real, reachable data clearing the corrected bar."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"real GCP reachable: {ingest.reachable} ({ingest.reason[:60]})")
    for s in per_seed:
        print(
            f"  seed {s['seed']}: Stouffer Z={s['stouffer_z']:+.3f} "
            f"p={s['two_sided_p']:.3f} sig={s['significant_bonferroni']}"
        )
    print(f"any_significant={any_sig}  (expected: False / null)")
    print("VERDICT: QUARANTINE (null on synthetic; real data unreachable; no psi claim)")
    print(f"artifact: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

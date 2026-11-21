# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""One decision, three answers, and a challenge you can run yourself.

This is ``mercury-agent demo`` run from a checkout, plus the recipe that
builds the window it scores.  The demonstration itself lives in the package
(:mod:`omni_mercury_engine.demo`) so it works from an installed wheel; see
that module for what it shows and why the honest outcome is a *defer*.

The window (``src/omni_mercury_engine/demo/auditable_window.csv``, 80 rows x
16 columns) is built here from the data the bundled checkpoint was trained on
(``scripts/train_default_fusion.build_dataset``):

* its ordinary rows come from one of that data's four regimes;
* column X0 was made to lead column X1 by one step (coefficient 0.7);
* rows 40-43 are four of that data's labelled anomalies.

Usage::

    python examples/auditable_decision.py              # the demonstration (same as: mercury-agent demo)
    python examples/auditable_decision.py --out DIR    # keep decision.json and window.csv in DIR
    python examples/auditable_decision.py --json       # the same, machine-readable
    python examples/auditable_decision.py --rebuild-window   # rebuild the committed window

Exit status 0 when the decision's findings hold on replay and the tampered
copy fails; 1 otherwise.  Needs the full agent (``pip install -e ".[pqc,ml]"``).
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]

#: How the window is made (see :func:`build_window`).
SEED = 0


def build_window() -> np.ndarray[Any, Any]:
    """Rebuild the committed window from the bundled checkpoint's training data.

    Returns:
        The 80 x 16 window, rounded to four decimals as committed.
    """
    sys.path.insert(0, str(REPO))
    from omni_mercury_engine.demo import PLANTED, ROWS
    from scripts.train_default_fusion import build_dataset

    features, labels = build_dataset()
    normals = features[labels == 0].astype(float)
    anomalies = features[labels == 1].astype(float)
    # One regime: the ordinary rows near the first one (the regimes sit about
    # 22 apart in this space and are about 6 across).
    regime = normals[np.linalg.norm(normals - normals[0], axis=1) < 8.0]
    rng = np.random.default_rng(SEED)
    window = regime[rng.choice(len(regime), size=ROWS, replace=False)].copy()
    window[1:, 1] += PLANTED["coefficient"] * (window[:-1, 0] - regime[:, 0].mean())
    rows = list(PLANTED["anomaly_rows"])
    window[rows] = anomalies[rng.choice(len(anomalies), size=len(rows), replace=False)]
    rounded: np.ndarray[Any, Any] = np.round(window, 4)
    return rounded


def main(argv: list[str] | None = None) -> int:
    """Run the demonstration, or rebuild its window."""
    from omni_mercury_engine import demo

    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("mercury-demo"),
        help="keep decision.json and window.csv in this directory",
    )
    parser.add_argument(
        "--json", action="store_true", help="print the answers, record and challenge as JSON"
    )
    parser.add_argument(
        "--rebuild-window",
        action="store_true",
        help="rebuild the committed window from the training data and exit",
    )
    args = parser.parse_args(argv)
    if args.rebuild_window:
        np.savetxt(demo.WINDOW_PATH, build_window(), delimiter=",", fmt="%.4f")
        print(f"wrote {demo.WINDOW_PATH.relative_to(REPO)}")
        return 0

    warnings.simplefilter("ignore")
    logging.disable(logging.WARNING)
    status, text = demo.run(args.out, as_json=args.json)
    print(text)
    return status


if __name__ == "__main__":
    raise SystemExit(main())

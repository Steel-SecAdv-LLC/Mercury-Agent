# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Sequential change detection with a conformal test martingale. (moved; shim).

The implementation now lives in :mod:`omni_mercury_engine.detectors.conformal_martingale`; this shim re-exports it
so released ``omni_mercury_engine.core.conformal_martingale`` imports keep
working unchanged. New code should import from the new location.

Tests a stream of nonconformity scores (higher is stranger) online for a rise
away from a reference sample, with no model of either distribution:

- p-values: the smoothed conformal p-value p_n = (#{a in bag: a > a_n} +
  U_n·(#{a in bag: a = a_n} + 1)) / (|bag| + 1) with U_n uniform on [0, 1],
  where the bag holds the reference scores and every earlier stream score; a_n
  then joins the bag. Under exchangeability these p-values are independent and
  uniform, ties included (Vovk, Nouretdinov & Gammerman 2003, "Testing
  exchangeability on-line", ICML; Vovk, Gammerman & Shafer 2022, *Algorithmic
  Learning in a Random World*, 2nd ed.). Without U_n, tied scores can make
  p_n smaller than uniform given the past, and the guarantee below fails.
- Betting: the one-sided Simple Jumper (Vovk et al. 2021, "Retrain or not
  retrain: conformal test martingales for change-point detection", PMLR 152).
  Capital is split between f(p) = 1.5 - p, a bet on small p-values (high
  scores), and f(p) = 1, no bet; each step a fraction J of it is spread evenly
  over both, so it can move to the bet when a change starts. Each step's factor
  is fixed before p_n is seen and integrates to 1 over [0, 1], so the capital
  S_n is a test martingale.
- Alarm: the Shiryaev-Roberts statistic R_n = Σ_{k<n} S_n/S_k, the sum over k
  of the capital the same bets gained after observation k, computed as
  R_n = (S_n/S_{n-1})(R_{n-1} + 1). R_n - n is a martingale under
  exchangeability, so by optional stopping the mean time from a reset to an
  alarm R_n >= c is at least c observations, whatever the score distribution.
  Since S_n/S_{n-1} <= 1.5, an alarm overshoots c by little (R_n < 1.5(c + 1)),
  and the bound is nearly tight in simulation (see the tests).
  With known pre- and post-change densities this procedure is optimal for that
  constraint in repeated monitoring (Pollak & Tartakovsky 2009, *Stat. Sin.*
  19:1729); here the betting function stands in for the unknown likelihood
  ratio.

Limitations: the null is exchangeability, so drift, seasonality or
autocorrelation in the scores also raise the statistic. The bag grows by one
score per observation. After a change the bag absorbs post-change scores and
the p-values return toward uniform, so an alarm calls for a refit of the
reference, not for continued monitoring against it.
"""

from __future__ import annotations

from omni_mercury_engine.detectors.conformal_martingale import (
    ChangeStatus,
    ConformalChangeDetector,
)

__all__ = [
    "ChangeStatus",
    "ConformalChangeDetector",
]

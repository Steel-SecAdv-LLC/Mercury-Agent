# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""One-sample-at-a-time detection through the engine.

Before this module, nothing took live data one sample at a time *through
the engine*: the streaming components (``streaming.streaming_detector``,
``detectors.detection_tier.TierStreamingScorer``) each drive a single
detector, while the engine's own surfaces (``detect``,
``detect_with_fusion``) are batch calls with no state between them.

:class:`EngineStreamSession` closes that gap: it holds a rolling window,
fits its own copies of the engine's enabled detectors once at warm-up (and
again every ``refit_interval`` samples, to track drift), and routes every
new sample through :meth:`OmniMercuryEngine.detect` -- the gated boundary,
so the harm-uplift gate applies to streaming verdicts exactly as to batch
ones. Detector state persists across samples; nothing is re-fit per record.

Three properties hold on every tick. The engine's shared state is never
written: the session's copies reach ``detect`` as per-call
``detector_overrides``. A sample never helps define the reference it is
scored against: a fit runs on the window as it stood *before* the sample
arrived. And a verdict is the newest sample's own: a detector that reports
only a window-level result is marked ``point_resolved: False`` and kept out
of the verdict, instead of having the window's statistic presented as the
sample's.

Drift is acted on, not only counted. When the online drift monitor reports a
change, the session drops the samples that predate it from its window and,
on the next tick, refits its detectors on what remains -- the post-drift
samples -- instead of scoring the new regime against the old one until the
next interval refit comes round. The tick says so (``drift_detected``,
``refit_reason``).

Example::

    engine = OmniMercuryEngine()
    session = engine.stream_session(detector_types=["statistical"])
    for value in live_feed:
        verdict = session.observe(value)
        if verdict["is_anomaly"]:
            alert(verdict)
"""

from __future__ import annotations

import copy
import itertools
import logging
import math
from collections import deque
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from omni_mercury_engine.engine import OmniMercuryEngine

logger = logging.getLogger(__name__)


class EngineStreamSession:
    """Stateful one-sample-at-a-time detection over an engine's detectors.

    The session keeps the most recent ``window_size`` samples. Each
    :meth:`observe` call appends one sample, and once ``min_samples`` have
    arrived, scores the window through ``engine.detect`` and reports the
    newest sample's verdict. Detectors are fitted on the warm-up window
    once, then re-fitted every ``refit_interval`` samples; between refits
    each call only scores, so per-sample cost is one detect pass, never a
    retrain. Every fit uses the samples that preceded the one being scored
    (at least two of them), never that sample itself.

    A drift the monitor detects triggers a refit of its own, ahead of the
    interval: the samples older than the detected change are dropped from
    the window, and on the next tick the detectors are refitted on the
    post-drift samples (the sample being scored excluded, as always). That
    first post-drift reference is short -- the monitor reports a change a
    dozen or so samples in, and its cut can leave a boundary sample of the
    old regime -- so it is consolidated: the detectors are refitted again
    each time the post-drift window has doubled, until the window is full.
    The monitor needs no reset for this -- ADWIN keeps only the post-change
    part of its own window when it reports a drift, so it is already
    watching the new regime. ``refit_reason`` on a tick is ``"warmup"``,
    ``"interval"``, ``"drift"`` or ``None`` (no fit on that tick).

    Only detectors that return per-row output aligned with the window give a
    verdict about the newest sample. One that returns a window-level result
    only is reported under ``per_detector`` with ``point_resolved: False``,
    its ``score`` / ``is_anomaly`` are ``None`` and its window statistic is
    kept under ``window_score`` / ``window_is_anomaly``; it takes no part in
    the tick's ``is_anomaly`` or ``score``. A tick on which no detector is
    point-resolved is an explicit abstention (``status:
    "not_point_resolved"``, ``is_anomaly: None``), not a clear.

    A non-finite sample is treated as unmeasured: it is excluded from the
    window, counted in ``samples_unmeasured``, and the returned verdict is
    marked ``partial`` -- it never poisons the window and never becomes a
    silent all-clear.

    Every verdict also carries ``churn``: the fraction of verdict flips (the
    newest sample's ``is_anomaly`` differing from the previous point-resolved
    tick's) among the last ``churn_window`` consecutive point-resolved verdict
    pairs. A stream that flips every tick is operationally unstable even when
    each tick is individually defensible, and nothing measured it. ``churn``
    is ``None`` until two point-resolved verdicts exist; warmup, unmeasured
    and not-point-resolved ticks never enter the history (they report the
    churn as it stands).
    """

    def __init__(
        self,
        engine: OmniMercuryEngine,
        *,
        detector_types: list[str] | None = None,
        window_size: int = 256,
        min_samples: int = 32,
        refit_interval: int = 128,
        monitor_drift: bool = True,
        churn_window: int = 32,
    ) -> None:
        """Initialize the session.

        Args:
            engine: The engine whose enabled detectors score the stream.
            detector_types: Detector names to run (default: all enabled).
            window_size: Rolling-window length scored on each sample (>= 2).
            min_samples: Samples to buffer before the first verdict (>= 2,
                <= window_size).
            refit_interval: Samples between detector refits (>= 1).
            monitor_drift: Run the adaptive-windowing (ADWIN, Bifet &
                Gavalda 2007) online drift monitor over the measured stream;
                each verdict then carries a ``drift`` record, a detected
                change is counted and named, and it triggers a refit of the
                session's detectors on the post-drift samples at the next
                tick.  Previously the online drift detector was built but
                reachable from no live path.
            churn_window: Number of consecutive point-resolved verdict pairs
                the per-tick ``churn`` fraction is computed over (>= 1).

        Raises:
            ValueError: If a window/sample/interval parameter is out of range.
        """
        if window_size < 2:
            raise ValueError(f"window_size must be >= 2, got {window_size}")
        if not 2 <= min_samples <= window_size:
            raise ValueError(
                f"min_samples must be in [2, window_size={window_size}], got {min_samples}"
            )
        if refit_interval < 1:
            raise ValueError(f"refit_interval must be >= 1, got {refit_interval}")
        if churn_window < 1:
            raise ValueError(f"churn_window must be >= 1, got {churn_window}")
        self._engine = engine
        self._detector_types = list(detector_types) if detector_types is not None else None
        # The session operates on its OWN detector instances, never the
        # engine's shared ones: warm-up/refit fits them and each sample is
        # scored through them, so the engine's trained state is untouched and
        # two sessions on one engine cannot overwrite each other's fits. The
        # names are resolved once, here, against the engine's detectors.
        names = self._detector_types if self._detector_types is not None else list(engine.detectors)
        self._names: list[str] = list(dict.fromkeys(names))
        self._detectors: dict[str, Any] = {}
        for name in self._names:
            source = engine.detectors.get(name)
            if source is None:
                raise KeyError(f"engine has no detector named {name!r}")
            self._detectors[name] = self._private_copy(name, source)
        self._window: deque[np.ndarray[Any, Any]] = deque(maxlen=window_size)
        self._min_samples = min_samples
        self._refit_interval = refit_interval
        self._samples_seen = 0
        self._samples_unmeasured = 0
        self._samples_since_fit = 0
        self._fitted = False
        # A drift was detected and the detectors have not been refitted on
        # the post-drift window yet (done on the next tick).
        self._drift_refit_pending = False
        # Window length at which the short post-drift reference is refitted
        # again (it doubles until the window is full); None outside that phase.
        self._drift_consolidate_at: int | None = None
        self._feature_dim: int | None = None
        # The last churn_window + 1 point-resolved verdicts (churn_window
        # consecutive pairs). Only "ok" ticks enter; warmup, unmeasured and
        # not-point-resolved ticks are excluded by construction.
        self._churn_history: deque[bool] = deque(maxlen=churn_window + 1)
        self._drift_monitor: Any = None
        self._drift_events = 0
        if monitor_drift:
            from omni_mercury_engine.detectors.drift import OnlineDriftDetector

            self._drift_monitor = OnlineDriftDetector(
                max_window_size=max(window_size * 4, 200),
                min_window_size=max(min_samples, 50),
            )

    @property
    def samples_seen(self) -> int:
        """Total samples offered to :meth:`observe` (measured or not)."""
        return self._samples_seen

    @property
    def samples_unmeasured(self) -> int:
        """Samples excluded as non-finite (reported, never scored)."""
        return self._samples_unmeasured

    def _churn(self) -> float | None:
        """Flip fraction over the retained point-resolved verdict pairs.

        ``None`` while fewer than two point-resolved verdicts exist (no pair
        to compare yet) -- never a fabricated 0.0 stability claim.
        """
        if len(self._churn_history) < 2:
            return None
        verdicts = list(self._churn_history)
        flips = sum(1 for prev, cur in itertools.pairwise(verdicts) if prev != cur)
        return flips / (len(verdicts) - 1)

    def _coerce(self, sample: Any) -> np.ndarray[Any, Any] | None:
        """One sample as a finite 1-D float vector, or None if unmeasured."""
        array = np.asarray(sample, dtype=np.float64).reshape(-1)
        if array.size == 0 or not np.all(np.isfinite(array)):
            return None
        if self._feature_dim is None:
            self._feature_dim = int(array.size)
        elif array.size != self._feature_dim:
            raise ValueError(
                f"sample has {array.size} features; this stream carries {self._feature_dim}"
            )
        return array

    @staticmethod
    def _private_copy(name: str, source: Any) -> Any:
        """A detector instance of the session's own, sharing nothing with ``source``.

        A deep copy when the detector can be copied: it carries the
        configuration and every attribute the caller set, and the session
        refits it on its own window regardless. A detector holding a lock or
        another uncopyable resource is built afresh from its class and a copy
        of its configuration instead -- the fitted state it loses is state
        the session replaces at warm-up, but attributes set on the instance
        after construction are not carried, which is logged.

        Raises:
            TypeError: If the detector can be neither copied nor rebuilt from
                its configuration; a session will not share the engine's
                instance in its place.
        """
        try:
            return copy.deepcopy(source)
        except Exception as copy_error:
            try:
                rebuilt = type(source)(copy.deepcopy(source.config))
            except Exception as rebuild_error:
                raise TypeError(
                    f"detector {name!r} ({type(source).__name__}) cannot be given to a "
                    f"stream session: it cannot be copied ({copy_error}) or rebuilt from "
                    f"its configuration ({rebuild_error}), and a session does not share "
                    "the engine's own instance"
                ) from rebuild_error
            logger.warning(
                "Stream session: detector %r (%s) cannot be copied (%s); the session uses "
                "a new instance built from its configuration. Attributes set on the "
                "engine's instance after construction are not carried over.",
                name,
                type(source).__name__,
                copy_error,
            )
            return rebuilt

    def _fit_window(self, window: np.ndarray[Any, Any]) -> None:
        """Fit the session's own detectors on the current window.

        Only the session's private detector copies are fitted; the engine's
        shared instances are never touched.
        """
        for detector in self._detectors.values():
            detector.fit(window)
        self._fitted = True
        self._samples_since_fit = 0

    def _newest_sample_view(self, name: str, result: dict[str, Any], n_rows: int) -> dict[str, Any]:
        """One detector's contribution to the newest sample's verdict.

        A detector speaks about the newest sample only through per-row output
        aligned with the window: its ``scores[-1]`` and, when ``is_anomaly``
        is per-row too, its ``is_anomaly[-1]`` (otherwise the sample's score
        against the detector's own threshold). A detector that reports a
        window-level result only -- no per-row scores of the window's length
        -- says nothing about the newest sample in particular. Its batch
        statistic (``anomaly_score``, often the window maximum) and its
        ``any()``-style flag used to be attributed to the newest sample, so
        one anomaly anywhere in the window flagged every tick until it
        scrolled out. Such a detector is returned ``point_resolved: False``
        with ``score`` / ``is_anomaly`` of ``None`` and its window-level
        values under ``window_score`` / ``window_is_anomaly``.
        """
        from omni_mercury_engine.core.base import BaseDetector

        def aligned(value: Any) -> np.ndarray[Any, Any] | None:
            if value is None or isinstance(value, (bool, int, float, str)):
                return None
            try:
                rows = np.asarray(value).reshape(-1)
            except (TypeError, ValueError):
                return None
            return rows if rows.size == n_rows else None

        raw_flags = result.get("is_anomaly", False)
        scores = aligned(result.get("scores"))
        if scores is None:
            window_score = result.get("anomaly_score", result.get("anomaly_prob"))
            if window_score is None and result.get("scores") is not None:
                misaligned = np.asarray(result["scores"], dtype=np.float64).reshape(-1)
                window_score = float(misaligned.max()) if misaligned.size else None
            return {
                "point_resolved": False,
                "score": None,
                "is_anomaly": None,
                "uncertainty": None,
                "window_score": (
                    None if window_score is None else float(np.clip(window_score, 0.0, 1.0))
                ),
                "window_is_anomaly": bool(np.any(np.asarray(raw_flags))),
            }

        last_score = float(np.clip(scores[-1], 0.0, 1.0))
        threshold = self._detectors[name].get_effective_threshold()
        flags = aligned(raw_flags)
        is_anomaly = bool(flags[-1]) if flags is not None else last_score > threshold
        uncertainty = BaseDetector.uncertainty_from_result(
            {**result, "scores": [last_score]}, threshold=threshold
        )
        return {
            "point_resolved": True,
            "score": last_score,
            "is_anomaly": bool(is_anomaly),
            "uncertainty": float(uncertainty),
        }

    def observe(self, sample: Any, timestamp: Any = None) -> dict[str, Any]:
        """Feed one sample; return the newest sample's verdict.

        Args:
            sample: One observation -- a scalar or a fixed-length vector.
            timestamp: Optional timestamp echoed back in the verdict.

        Returns:
            Verdict dict: ``status`` ("warming_up", "ok", "unmeasured" or
            "not_point_resolved"), ``partial`` (True while warming up, when
            this sample was unmeasured, or when no detector gave a verdict
            about this sample), ``is_anomaly``/``score``/``per_detector``/
            ``uncertainty`` once warmed up, ``churn`` (verdict-flip fraction
            over the last ``churn_window`` point-resolved pairs; ``None``
            until two resolved verdicts), plus the session counters.

        Raises:
            ValueError: On a sample whose dimensionality contradicts the
                stream's.
        """
        self._samples_seen += 1
        array = self._coerce(sample)
        base: dict[str, Any] = {
            "timestamp": timestamp,
            "samples_seen": self._samples_seen,
            "samples_unmeasured": self._samples_unmeasured,
            "window_fill": len(self._window),
            # Verdict-flip fraction over the last point-resolved ticks; a
            # tick that resolves a verdict below overwrites this with the
            # value that includes itself. None until two resolved verdicts.
            "churn": self._churn(),
        }
        if array is None:
            self._samples_unmeasured += 1
            base["samples_unmeasured"] = self._samples_unmeasured
            # Unmeasured is reported, never scored and never an all-clear.
            return {**base, "status": "unmeasured", "partial": True, "is_anomaly": None}

        # Fit BEFORE the sample joins the window: the reference a sample is
        # scored against must not contain that sample, or an anomaly helps
        # define the "normal" it is then compared with. The reference is the
        # window as it stood when the sample arrived. ``min_samples`` gates
        # the first verdict; once fitted, the session keeps scoring even
        # while a drift has shortened the window.
        will_score = self._fitted or len(self._window) + 1 >= self._min_samples
        refit_reason: str | None = None
        if will_score and len(self._window) >= 2:
            if not self._fitted:
                refit_reason = "warmup"
            elif self._drift_refit_pending or (
                self._drift_consolidate_at is not None
                and len(self._window) >= self._drift_consolidate_at
            ):
                refit_reason = "drift"
            elif self._samples_since_fit >= self._refit_interval:
                refit_reason = "interval"
        if refit_reason is not None:
            self._fit_window(np.vstack(self._window))
            self._drift_refit_pending = False
            capacity = self._window.maxlen or len(self._window)
            self._drift_consolidate_at = (
                min(2 * len(self._window), capacity)
                if refit_reason == "drift" and len(self._window) < capacity
                else None
            )
        base["refit_reason"] = refit_reason

        self._window.append(array)
        if self._drift_monitor is not None:
            drift_result = self._drift_monitor.update(array)
            if drift_result is not None:
                self._drift_events += 1
                # Act on it. The monitor has cut its own window to the
                # samples since the change; keep the same samples here, so
                # the next tick's fit -- and every score until then -- is
                # about the new regime, not a blend of the two.
                since_change = int(drift_result.details.get("window_size", 0))
                if 2 <= since_change < len(self._window):
                    recent = list(self._window)[-since_change:]
                    self._window.clear()
                    self._window.extend(recent)
                self._drift_refit_pending = True
            base["drift"] = {
                "detected": drift_result is not None,
                "events": self._drift_events,
                "message": None if drift_result is None else drift_result.message,
            }
            base["drift_detected"] = drift_result is not None
        base["window_fill"] = len(self._window)
        if not will_score or not self._fitted:
            return {**base, "status": "warming_up", "partial": True, "is_anomaly": None}

        window = np.vstack(self._window)
        self._samples_since_fit += 1
        n_rows = window.shape[0]

        # Score through the session's own detector copies, handed to the
        # gated boundary as per-call overrides: the engine's detectors are
        # never swapped, so no other caller of this engine -- and no other
        # session -- can observe or inherit them.
        detection = self._engine.detect(
            window, detector_types=self._names, detector_overrides=self._detectors
        )
        per_detector: dict[str, dict[str, Any]] = {}
        for name, result in detection["detectors"].items():
            per_detector[name] = self._newest_sample_view(name, result, n_rows)

        resolved = [name for name, entry in per_detector.items() if entry["point_resolved"]]
        if not resolved:
            # Nothing spoke about this sample. A window-level statistic is
            # not its verdict, and "no verdict" is not "clear".
            return {
                **base,
                "status": "not_point_resolved",
                "partial": True,
                "is_anomaly": None,
                "score": None,
                "per_detector": per_detector,
                "uncertainty": 1.0,
            }

        scores = [per_detector[name]["score"] for name in resolved]
        score = float(np.mean(scores))
        certainty_weights = [1.0 - per_detector[name]["uncertainty"] for name in resolved]
        if sum(certainty_weights) > 0.0:
            # Certainty-weighted mean: a detector shrugging at its decision
            # boundary moves the stream verdict less than a certain one.
            score = float(np.average(scores, weights=np.asarray(certainty_weights) + 1e-9))
        is_anomaly = any(per_detector[name]["is_anomaly"] for name in resolved)
        if math.isnan(score):
            raise RuntimeError("stream verdict produced a NaN score; refusing to serve it")
        # This tick resolved a verdict about its own sample: it enters the
        # churn history, and the tick reports the churn including itself.
        self._churn_history.append(bool(is_anomaly))
        return {
            **base,
            "status": "ok",
            "partial": False,
            "is_anomaly": is_anomaly,
            "score": score,
            "per_detector": per_detector,
            "uncertainty": float(np.mean([per_detector[name]["uncertainty"] for name in resolved])),
            "churn": self._churn(),
        }

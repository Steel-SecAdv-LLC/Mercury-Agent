# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Reproducible training pipeline for the hazard-detector neural hooks.

Submodules (imported lazily by the registry so that inference-side imports of
:mod:`.features` never drag in training dependencies):

* :mod:`.common` -- caching, provenance, temporal splits, the merit gate.
* :mod:`.features` -- canonical feature specs shared with the detectors.
* :mod:`.solar_storm` -- hook "solar_storm" (GeomagneticStormPredictor,
  OMNI2 solar wind + observed Kp; the category-(a) pipeline).
* :mod:`.registry` -- one entry per ``load_neural_weights`` hook; the
  category (b)/(c) hooks fail loud with their documented data requirement.

Entry point: ``scripts/train_hazard_checkpoints.py``.
"""

# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for pandemic loader multi-pathogen expansion."""

from __future__ import annotations

import logging
import unittest
from unittest.mock import patch

from omni_mercury_engine.loaders.pandemic_loader import (
    _EVENT_CATALOG,
    PandemicLoader,
)

logger = logging.getLogger(__name__)


class TestPandemicExpansion(unittest.TestCase):
    """Test suite for the multi-pathogen pandemic loader expansion."""

    def setUp(self) -> None:
        self.loader = PandemicLoader()

    # ----------------------------------------------------------------
    # Test 1: pathogen classes complete
    # ----------------------------------------------------------------
    def test_pathogen_classes_complete(self) -> None:
        """All 6 pathogen classes must be present."""
        classes = self.loader.list_pathogen_classes()
        expected = {
            "virus",
            "bacteria",
            "fungus",
            "parasite",
            "prion",
            "biosurveillance",
        }
        self.assertEqual(set(classes), expected)

    # ----------------------------------------------------------------
    # Test 2: existing COVID events preserved
    # ----------------------------------------------------------------
    def test_existing_covid_events_preserved(self) -> None:
        """covid_usa_wave1 and other original COVID events still exist."""
        event_ids = set(_EVENT_CATALOG.keys())
        self.assertIn("covid_usa_wave1", event_ids)
        self.assertIn("covid_italy_wave1", event_ids)
        self.assertIn("covid_india_delta", event_ids)
        self.assertIn("ebola_2014", event_ids)
        self.assertIn("mpox_2022", event_ids)

    # ----------------------------------------------------------------
    # Test 3: every event has pathogen_class
    # ----------------------------------------------------------------
    def test_event_catalog_has_pathogen_class(self) -> None:
        """Every event in the catalog must have a pathogen_class field."""
        for event_id, meta in _EVENT_CATALOG.items():
            self.assertIn(
                "pathogen_class",
                meta,
                f"Event '{event_id}' missing pathogen_class",
            )

    # ----------------------------------------------------------------
    # Test 4: prion warning logged
    # ----------------------------------------------------------------
    def test_prion_warning_logged(self) -> None:
        """Prion events must emit a density warning via logging."""
        with self.assertLogs(
            "omni_mercury_engine.loaders.pandemic_loader",
            level="WARNING",
        ) as cm:
            self.loader.get_ground_truth("cjd_us_surveillance")

        # Check that a PRION or stub warning was emitted
        log_text = " ".join(cm.output)
        self.assertTrue(
            "stub" in log_text.lower() or "prion" in log_text.lower(),
            f"Expected prion/stub warning in logs, got: {log_text}",
        )

    # ----------------------------------------------------------------
    # Test 5: list_events includes all classes
    # ----------------------------------------------------------------
    def test_list_events_includes_all_classes(self) -> None:
        """list_events() must return events from all 6 pathogen classes."""
        events = self.loader.list_events()
        event_ids = {e["event_id"] for e in events}

        # Check at least one event per class
        classes_found: set[str] = set()
        for eid in event_ids:
            meta = _EVENT_CATALOG[eid]
            classes_found.add(meta["pathogen_class"])

        expected = {
            "virus",
            "bacteria",
            "fungus",
            "parasite",
            "prion",
            "biosurveillance",
        }
        self.assertEqual(
            classes_found,
            expected,
            f"Missing classes: {expected - classes_found}",
        )

    # ----------------------------------------------------------------
    # Test 6: WHO GHO graceful failure
    # ----------------------------------------------------------------
    def test_who_gho_graceful_failure(self) -> None:
        """Mock WHO API returning error; verify loader returns empty."""
        with patch.object(
            self.loader,
            "_fetch_json",
            side_effect=ConnectionError("Mocked 500"),
        ):
            df = self.loader._fetch_who_gho(
                indicator="WHS3_41",
                country="YEM",
                start_date="2017-01-01",
                end_date="2018-12-31",
            )
            self.assertTrue(df.empty)

    # ----------------------------------------------------------------
    # Test 7: stub events return empty DataFrame
    # ----------------------------------------------------------------
    def test_stub_events_return_empty(self) -> None:
        """Stub events (fungus, prion, parasite) return empty DataFrame."""
        stub_events = [
            "candida_auris_us_tracking",
            "cjd_us_surveillance",
            "malaria_subsaharan_2019_2022",
        ]
        for eid in stub_events:
            df = self.loader.fetch_historical(eid)
            self.assertTrue(
                df.empty,
                f"Stub event '{eid}' should return empty DataFrame",
            )


if __name__ == "__main__":
    unittest.main()

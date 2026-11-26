# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
r"""Coverage-guided fuzzing of the watch's external CAP/XML parsers (Atheris).

``parse_cap`` and ``CAPAlertGenerator.validate_cap_xml`` read XML from
upstream alerting authorities (WMO, GDACS, national CAP feeds). The
Hypothesis properties in ``test_fuzz_cap_parser.py`` pin their error contract
over a fixed input grammar; this harness drives the same contract with
libFuzzer's coverage feedback, so inputs evolve toward unexplored parser
branches instead of being drawn from a fixed distribution:

* ``parse_cap`` raises only ``ValueError`` or returns a ``dict``;
* ``validate_cap_xml`` never raises and returns a ``bool``.

Any other exception, or a wrong return type, is a crash: libFuzzer stops, and
writes the reproducing input to ``crash-<sha1>``.

Run (CPython >= 3.12 on Linux x86_64; ``.github/workflows/fuzz.yml``), seeded
with the real CAP alerts under ``tests/watch/data`` and the CAP/XML token
dictionary beside this file::

    mkdir -p corpus && cp tests/watch/data/*.cap.xml corpus/
    PYTHONPATH=src python tests/fuzz/atheris_cap_parser.py corpus \\
        -dict=tests/fuzz/cap.dict -max_total_time=60

Replay a crash with ``python tests/fuzz/atheris_cap_parser.py crash-<sha1>``.
"""

from __future__ import annotations

import sys

import atheris

with atheris.instrument_imports(include=["omni_mercury_engine"]):
    from omni_mercury_engine.alerting.cap_generator import CAPAlertGenerator
    from omni_mercury_engine.watch.world import parse_cap

#: Upper bound on one decoded document, in characters. CAP alerts are a few
#: kilobytes; a bound keeps each execution fast enough to explore widely.
MAX_DOCUMENT_CHARS = 8192


def check_document(document: str) -> None:
    """Assert the parsers' error contract on one document.

    Args:
        document: Arbitrary text presented as a CAP alert.

    Raises:
        AssertionError: A parser broke its contract.
    """
    try:
        parsed = parse_cap(document)
    except ValueError:
        pass  # the one documented failure mode
    else:
        if not isinstance(parsed, dict):
            raise AssertionError(f"parse_cap returned {type(parsed).__name__}, not dict")
    verdict = CAPAlertGenerator.validate_cap_xml(document)
    if not isinstance(verdict, bool):
        raise AssertionError(f"validate_cap_xml returned {type(verdict).__name__}, not bool")


def fuzz_one_input(data: bytes) -> None:
    """libFuzzer entry point: decode the bytes and check both parsers.

    The bytes are decoded as UTF-8 (invalid sequences replaced) rather than
    through ``FuzzedDataProvider``, so a seed file is the exact document the
    parsers see and mutations stay close to real alerts.

    Args:
        data: Fuzzer-generated input.
    """
    check_document(data.decode("utf-8", errors="replace")[:MAX_DOCUMENT_CHARS])


def main() -> None:
    """Hand control to libFuzzer with the command-line options."""
    atheris.Setup(sys.argv, fuzz_one_input)
    atheris.Fuzz()


if __name__ == "__main__":
    main()

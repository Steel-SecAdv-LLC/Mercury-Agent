#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Record real CC0 images for the visual and video detector tests (tests/fixtures/visual).

Downloads four photographs from scikit-image's bundled sample data at a pinned
commit, byte-verbatim, and checks each against the SHA-256 that scikit-image's
own data registry (``skimage/data/_registry.py``) publishes for it:

* ``brick.png``: a brick wall (CC0Textures "Bricks25", CC0);
* ``gravel.png``: gravel (CC0Textures "Gravel04", CC0);
* ``chelsea.png``: a cat (CC0, photographer Stefan van der Walt);
* ``coffee.png``: a coffee cup (CC0, photographer Rachel Michetti).

The licences are those stated in scikit-image's ``skimage/data/_fetchers.py``
at the same commit.  Real images matter here: a visual anomaly detector that
separates synthetic squares from sinusoids proves little; one that separates a
real foreign-texture defect (gravel pasted into brick, the CutPaste protocol of
Li et al., CVPR 2021) from held-out real brick does, and a vision-language
model asked a question about a real photograph is asked a real question.

Writes the PNGs and ``PROVENANCE.json`` (URLs, licences, SHA-256).

Usage::

    python scripts/record_visual_fixtures.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

#: scikit-image v0.24.0 (annotated tag 8be71b82) dereferenced to its commit.
SKIMAGE_COMMIT = "a160f384523ac70173d069954b7084bee79fb68e"
BASE = f"https://raw.githubusercontent.com/scikit-image/scikit-image/{SKIMAGE_COMMIT}/skimage/data/"
FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "visual"

#: name -> (SHA-256 published in skimage/data/_registry.py, licence note).
IMAGES: dict[str, tuple[str, str]] = {
    "brick.png": (
        "7966caf324f6ba843118d98f7a07746d22f6a343430add0233eca5f6eaaa8fcf",
        "CC0 (CC0Textures Bricks25)",
    ),
    "gravel.png": (
        "c48615b451bf1e606fbd72c0aa9f8cc0f068ab7111ef7d93bb9b0f2586440c12",
        "CC0 (CC0Textures Gravel04)",
    ),
    "chelsea.png": (
        "596aa1e7cb875eb79f437e310381d26b338a81c2da23439704a73c4651e8c4bb",
        "CC0 by the photographer (Stefan van der Walt)",
    ),
    "coffee.png": (
        "cc02f8ca188b167c775a7101b5d767d1e71792cf762c33d6fa15a4599b5a8de7",
        "CC0 by the photographer (Rachel Michetti)",
    ),
}


def _get(name: str) -> bytes:
    url = BASE + name
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - fixed https host
        return bytes(response.read())


def main() -> int:
    """Download, verify and write every image plus PROVENANCE.json."""
    FIXTURES.mkdir(parents=True, exist_ok=True)
    provenance: dict[str, object] = {
        "description": (
            "Four real photographs from scikit-image's bundled sample data, byte-verbatim, "
            "each verified against the SHA-256 in scikit-image's data registry."
        ),
        "source": f"https://github.com/scikit-image/scikit-image at commit {SKIMAGE_COMMIT}",
        "licence_source": "skimage/data/_fetchers.py at the same commit",
        "recorder": "scripts/record_visual_fixtures.py",
        "recorded_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": {},
    }
    files: dict[str, dict[str, str]] = {}
    for name, (expected, licence) in IMAGES.items():
        payload = _get(name)
        digest = hashlib.sha256(payload).hexdigest()
        if digest != expected:
            print(f"{name}: SHA-256 {digest} != registry {expected}", file=sys.stderr)
            return 1
        (FIXTURES / name).write_bytes(payload)
        files[name] = {"url": BASE + name, "sha256": digest, "licence": licence}
    provenance["files"] = files
    (FIXTURES / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"wrote {len(files)} images to {FIXTURES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

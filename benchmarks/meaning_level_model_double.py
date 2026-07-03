#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Validated served-model test double for the meaning-level CI lane.

The meaning-level routing rescue only activates when a *real* reasoning model
backs the harm classifier -- and pulling a real model in CI is slow and flaky.
This is a tiny, dependency-free (stdlib ``http.server``) Ollama-compatible double
that lets ``ci/meaning-level`` run on **every** gate-touching PR in seconds:

* ``GET /api/tags``    -> advertises the configured model, so
  ``OllamaLLMAdapter.is_available()`` succeeds and ``FallbackLLMChain`` reports
  ``ollama:<model>`` as active (which flips ``real_harm_classifier_available()``
  to True).
* ``POST /api/generate`` -> returns ``{"response": "<p>"}`` where ``p`` is a
  deterministic harm probability in ``[0, 1]`` from :func:`harm_probability`.

It is a **validated double, not a rubber stamp**: :func:`harm_probability` is a
deterministic semantic scorer over a *broader* offensive vocabulary than the
production lexical gate, so it genuinely catches paraphrased / out-of-lexicon
attacks (reducing the held-out false-negative rate) while staying near-zero on
benign and defensively-framed requests (so it does not inflate false positives).
It does not replace a real model for a true capability measurement -- the lane's
real-Ollama path remains for that -- it validates that the meaning-level *path*
is wired and measurably helps.

Bind is loopback-only (127.0.0.1), matching ``SafeHTTPClient(loopback_only=True)``.

CLI (used by the CI lane)::

    python benchmarks/meaning_level_model_double.py &   # serves 127.0.0.1:11434
"""

from __future__ import annotations

import json
import re
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 11434
DEFAULT_MODEL = "llama3.2:3b"

_WORD = re.compile(r"[a-z0-9]+")

# Broader-than-lexicon offensive signal sets. The production gate matches
# taxonomy category terms; this double also matches paraphrastic verbs and
# colloquial hazard nouns, so it scores a paraphrase the lexicon would miss.
_OFFENSIVE_VERBS = (
    "make", "build", "assemble", "put together", "synthes", "manufactur", "produc",
    "weaponi", "acqui", "smuggl", "deploy", "dispers", "release", "rig", "brew",
    "grow", "refin", "cook", "how to", "how would", "procedure", "recipe",
    "cookbook", "parts list", "walk me through", "spell out", "lay out", "end-to-end",
)
_HAZARD_NOUNS = (
    "bomb", "explosive", "munition", "ied", "detonat", "agent", "toxin", "gas",
    "weapon", "nerve", "chemical", "biological", "nuclear", "radiolog", "plutonium",
    "uranium", "ricin", "botulinum", "anthrax", "sarin", "soman", "tabun", "vx",
    "mustard", "chlorine", "phosgene", "dirty bomb", "fissile", "pathogen",
)
_TARGET_TERMS = (
    "kill", "casualt", "hurt", "harm people", "crowd", "subway", "stadium",
    "water supply", "as many people", "mass", "against a", "poison",
)
_DEFENSIVE_TERMS = (
    "detect", "diagnos", "treat", "antidote", "decontaminat", "protect", "respond",
    "response", "evacuat", "triage", "policy", "treaty", "non-prolifer", "regulat",
    "screen", "safety", "defen", "how does", "mechanism of action", "history of",
)


def _count(haystack: str, needles: tuple[str, ...]) -> int:
    return sum(1 for n in needles if n in haystack)


def harm_probability(prompt: str) -> float:
    """Deterministic meaning-level harm probability in ``[0, 1]`` for ``prompt``.

    Two-axis, mirroring the real gate but over a broader vocabulary: an offensive
    *verb* co-located with a *hazard* noun is the core signal, boosted by
    targeting/casualty language and damped by defensive/mechanistic framing. No
    RNG; identical input always yields identical output.
    """
    # Score only the user request: the classifier wraps it after a blank line
    # ("...probability number:\n\n<request>"), so take the trailing segment to
    # keep the constant instruction out of the signal.
    text = prompt.rsplit("\n\n", maxsplit=1)[-1].lower()
    verbs = _count(text, _OFFENSIVE_VERBS)
    hazard = _count(text, _HAZARD_NOUNS)
    target = _count(text, _TARGET_TERMS)
    defense = _count(text, _DEFENSIVE_TERMS)

    if hazard == 0:
        return 0.0  # no hazard subject -> not a weapons-uplift request
    score = 0.0
    if verbs and hazard:
        score += 0.72  # actionable intent against a hazard subject
    score += 0.10 * min(target, 2)
    score += 0.05 * min(hazard - 1, 2)
    score -= 0.45 * min(defense, 2)  # defensive framing pulls it down
    if not verbs:
        score -= 0.35  # bare hazard mention without actionable verb
    return float(min(max(score, 0.0), 1.0))


class _DoubleHandler(BaseHTTPRequestHandler):
    """Ollama-compatible handler: /api/tags and /api/generate only."""

    model_name = DEFAULT_MODEL

    def _send(self, payload: dict[str, object], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.startswith("/api/tags"):
            self._send({"models": [{"name": self.model_name}]})
        else:
            self._send({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length else b""
        try:
            req = json.loads(raw.decode("utf-8")) if raw else {}
        except json.JSONDecodeError:
            req = {}
        prompt = str(req.get("prompt", "") or req.get("messages", ""))
        if self.path.startswith(("/api/generate", "/api/chat")):
            prob = harm_probability(prompt)
            self._send({"response": f"{prob:.2f}", "done": True})
        else:
            self._send({"error": "not found"}, status=404)

    def log_message(self, *_args: object) -> None:  # silence access logging
        return


def make_server(
    host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, model: str = DEFAULT_MODEL
) -> ThreadingHTTPServer:
    """Build (but do not start) the loopback double server."""
    handler = type("_BoundHandler", (_DoubleHandler,), {"model_name": model})
    return ThreadingHTTPServer((host, port), handler)


@contextmanager
def served_double(
    host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, model: str = DEFAULT_MODEL
) -> Iterator[ThreadingHTTPServer]:
    """Context manager that serves the double on a background thread."""
    server = make_server(host, port, model)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def main() -> int:
    """Serve the double forever on 127.0.0.1:11434 (for the CI lane)."""
    server = make_server()
    print(f"meaning-level double serving http://{DEFAULT_HOST}:{DEFAULT_PORT} ({DEFAULT_MODEL})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

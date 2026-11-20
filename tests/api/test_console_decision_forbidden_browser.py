# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""An ethical-gate 403 and a decision-bearing response render in the open console.

The terminal told an operator whose request the harm-uplift gate refused (403)
to create an API key, rendered an object ``explanation`` as ``[object
Object]`` and collapsed a ``decision`` record to nothing. This opens the real
served page (``/console`` plus its static modules, byte for byte from the
repository) in headless Chromium against a canned engine: a stub server in
this process answers ``/health``, serves the console, and returns scripted
detection responses, so the test is deterministic and fully offline. The
driver runs the console's own command path (``logic.runLive``) and reads the
terminal state:

* a 403 whose payload carries ``detail`` and an ``ethics`` object renders the
  refusal with check / score / threshold / ruleset, and never the API-key
  advice;
* a 200 whose payload carries a ``decision`` record and an object
  ``explanation`` renders state, disposition, abstention, calibration,
  coverage, reasons and caveats, and the explanation's method, top
  attributions and faithfulness -- no ``[object Object]`` anywhere.

Node with Playwright and a launchable Chromium are required, exactly as for
``test_console_watch_alert_browser.py``; where they are absent the test skips
(CI's full-suite lane sets ``MERCURY_REQUIRES_BROWSER=1``, making that a
failure there). The pure-function halves of these behaviors are covered
browser-free in ``tests/api/test_console_response_rendering.py``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from tests.browser_driver import node_env, playwright_available

REPO = Path(__file__).resolve().parents[2]
ASSETS = REPO / "src" / "omni_mercury_engine" / "api" / "frontend_assets"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(
    not playwright_available(),
    reason="Node with Playwright is not installed: the console is opened in headless Chromium",
)

#: What the engine will put on an ethical-gate 403 (detail is today's; ethics
#: is the object the API adds -- the console renders what is present).
FORBIDDEN = {
    "detail": "Request refused by the harm-uplift decision gate (check=harm_uplift).",
    "ethics": {
        "check": "harm_uplift",
        "score": 0.97,
        "threshold": 0.5,
        "ruleset_version": "2026.3",
        "details": "aggregate uplift pattern in the request text",
    },
}

#: A detection response carrying the decision record (decision/record.py
#: ``to_dict`` shape) and the flagship's Integrated-Gradients explanation dict.
DECIDED = {
    "scores": [0.12, 0.2, 0.91],
    "is_anomaly": [False, False, True],
    "threshold": 0.5,
    "method": "multivariate",
    "decision": {
        "state": "GROUNDED",
        "disposition": "act",
        "decision_label": 1,
        "abstained": False,
        "anomaly_prob": 0.91,
        "threshold": 0.5,
        "decision_confidence": 0.9,
        "coverage": 0.9,
        "calibrated": True,
        "severity": 0.8,
        "reasons": ["calibrated probability 0.910 cleared the threshold 0.500"],
        "caveats": ["ethical gate not run on this payload"],
    },
    "explanation": {
        "explanation_type": "integrated_gradients",
        "feature_importances": [
            {"feature": "f3", "importance": 0.42},
            {"feature": "f1", "importance": -0.17},
        ],
        "faithfulness_scores": {"comprehensiveness": 0.81},
    },
}

_CTYPES = {
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".png": "image/png",
    ".html": "text/html; charset=utf-8",
}


class _CannedEngine(BaseHTTPRequestHandler):
    """Serves the real console files and scripted detection answers."""

    def log_message(self, *args: object) -> None:
        pass

    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: object) -> None:
        self._send(status, json.dumps(payload).encode(), "application/json")

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/console":
            page = (ASSETS / "pages" / "console.html").read_bytes()
            self._send(200, page, _CTYPES[".html"])
            return
        if path.startswith("/static/console/"):
            name = path.rsplit("/", 1)[1]
            file = ASSETS / "static" / "console" / name
            if file.is_file():
                self._send(
                    200, file.read_bytes(), _CTYPES.get(file.suffix, "application/octet-stream")
                )
            else:
                self._json(404, {"detail": "not here"})
            return
        if path == "/health":
            self._json(200, {"status": "healthy"})
            return
        # Everything else the console probes at boot (voice status, watch,
        # alerts stream, backends, ...) gets an empty, well-formed answer.
        self._json(200, {})

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        path = self.path.split("?", 1)[0]
        if path == "/api/v1/detect/fusion":
            self._json(403, FORBIDDEN)
            return
        if path == "/api/v1/detect/multivariate":
            self._json(200, DECIDED)
            return
        self._json(200, {})


_DRIVER = """
'use strict';
const { chromium } = require('playwright');
const BASE = process.env.BASE;
let failed = 0;
function check(ok, what, detail) {
  if (!ok) { failed += 1; }
  console.log((ok ? 'ok   ' : 'FAIL ') + what + (detail ? ' :: ' + detail : ''));
}
(async () => {
  const browser = await chromium.launch(process.env.CHROMIUM ? { executablePath: process.env.CHROMIUM } : {});
  const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
  const pageErrors = [];
  page.on('pageerror', (error) => pageErrors.push(error.message));

  await page.goto(BASE + '/console');
  await page.waitForFunction(() => window.MercuryConsoleApp && window.MercuryConsoleApp.logic);
  await page.getByRole('button', { name: /enter console/i }).click();
  await page.waitForFunction(() => window.MercuryConsoleApp.logic.live(), null, { timeout: 60000 });

  const terminal = () => page.evaluate(() =>
    window.MercuryConsoleApp.logic.state.lines.map((l) => JSON.stringify(l)).join('\\n'));

  // An ethical-gate 403: the console's own live command path, canned answer.
  await page.evaluate(async () => {
    const L = window.MercuryConsoleApp.logic, raw = 'mercury-agent detect --detector fusion';
    await L.runLive(raw, L.parse(raw), { errors: [], r: { vals: { '--detector': 'fusion' } } });
  });
  const t1 = await terminal();
  check(t1.includes('Request refused by the harm-uplift decision gate'), 'the 403 renders the server detail');
  check(t1.includes('check · harm_uplift'), 'the ethics check is named', t1.slice(-600));
  check(t1.includes('score · 0.970') && t1.includes('threshold · 0.500'), 'score and threshold are shown');
  check(t1.includes('ruleset · 2026.3'), 'the ruleset version is shown');
  check(t1.includes('aggregate uplift pattern'), 'the ethics details are shown');
  check(!t1.includes('Create an API key'), 'it does not tell the operator to create an API key');

  // A 200 carrying a decision record and an object explanation.
  await page.evaluate(async () => {
    const L = window.MercuryConsoleApp.logic, raw = 'mercury-agent detect';
    await L.runLive(raw, L.parse(raw), { errors: [], r: { vals: {} } });
  });
  const t2 = await terminal();
  check(t2.includes('decision GROUNDED · act'), 'the decision state and disposition are shown', t2.slice(-600));
  check(t2.includes('abstained false') && t2.includes('calibrated true') && t2.includes('coverage 90.0%'),
    'abstention, calibration and coverage are shown');
  check(t2.includes('reason · calibrated probability 0.910 cleared the threshold 0.500'), 'the reasons are shown');
  check(t2.includes('caveat · ethical gate not run on this payload'), 'the caveats are shown');
  check(t2.includes('integrated_gradients') && t2.includes('f3: 0.420') && t2.includes('f1: -0.170'),
    'the IG explanation names its method and top attributions');
  check(t2.includes('faithfulness comprehensiveness 0.810'), 'the faithfulness score is shown');
  check(!t2.includes('[object Object]'), 'no [object Object] anywhere in the terminal');
  check(pageErrors.length === 0, 'no page errors', pageErrors.join(' | '));

  await browser.close();
  console.log(failed === 0 ? 'console decision/forbidden: all checks passed'
    : 'console decision/forbidden: ' + failed + ' check(s) FAILED');
  process.exit(failed === 0 ? 0 : 1);
})().catch((error) => { console.error('FAILED', error); process.exit(1); });
"""


def test_forbidden_and_decision_responses_render_in_the_open_console(tmp_path: Path) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _CannedEngine)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    driver = tmp_path / "console_decision_forbidden_check.js"
    driver.write_text(_DRIVER, encoding="utf-8")
    try:
        check = subprocess.run(
            [str(NODE), str(driver)],
            capture_output=True,
            text=True,
            timeout=240,
            check=False,
            env={**node_env(), "BASE": f"http://127.0.0.1:{port}"},
        )
        report = check.stdout + check.stderr
        assert check.returncode == 0, report
        assert "console decision/forbidden: all checks passed" in report
        assert "FAIL" not in report
    finally:
        server.shutdown()
        thread.join(timeout=10)
        server.server_close()

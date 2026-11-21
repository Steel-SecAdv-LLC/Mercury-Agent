# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""REVIEW A FILE in the open console: a real file, a real browser, the real engine.

Headless Chromium opens the served console (its static modules byte for byte
from the repository), a committed sample is chosen in the page's own file
picker, and the request the page sends is answered by the real review library
(``omni_mercury_engine.review.review_bytes`` with the shipped checkpoint)
behind a small in-process HTTP server.  Nothing is canned: the bytes the
server receives are checked against the file on disk, and the decision the
page shows is checked against the one the engine returned.

* the picker reads the file and posts it, base64 in JSON, to
  ``/api/v1/review/inline``;
* the result lands in RESULTS as a REVIEW with the decision in its headline,
  the three answers, and the decision record;
* a file the server refuses (compressed audio, 415) is reported in the
  terminal with the server's reason, and no result is invented.

Node with Playwright and a launchable Chromium are required, as for the other
console browser tests; where they are absent the test skips (CI's full-suite
lane sets ``MERCURY_REQUIRES_BROWSER=1``, making that a failure there).
"""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar

import pytest

pytest.importorskip("torch")
pytest.importorskip("PIL")

from omni_mercury_engine import review
from tests.browser_driver import node_env, playwright_available

REPO = Path(__file__).resolve().parents[2]
ASSETS = REPO / "src" / "omni_mercury_engine" / "api" / "frontend_assets"
SAMPLE = REPO / "examples" / "data" / "review" / "sample_document.txt"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(
    not playwright_available(),
    reason="Node with Playwright is not installed: the console is opened in headless Chromium",
)

_CTYPES = {
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".png": "image/png",
    ".html": "text/html; charset=utf-8",
}


class _ReviewingEngine(BaseHTTPRequestHandler):
    """Serves the real console files and answers a review with the real library."""

    received: ClassVar[list[dict[str, Any]]] = []
    answered: ClassVar[list[dict[str, Any]]] = []

    def log_message(self, *args: object) -> None:
        pass

    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: object) -> None:
        self._send(status, json.dumps(payload, default=str).encode(), "application/json")

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/console":
            self._send(200, (ASSETS / "pages" / "console.html").read_bytes(), _CTYPES[".html"])
        elif path.startswith("/static/console/"):
            file = ASSETS / "static" / "console" / path.rsplit("/", 1)[1]
            if file.is_file():
                self._send(
                    200, file.read_bytes(), _CTYPES.get(file.suffix, "application/octet-stream")
                )
            else:
                self._json(404, {"detail": "not here"})
        elif path == "/health":
            self._json(200, {"status": "healthy"})
        else:
            self._json(200, {})

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if self.path.split("?", 1)[0] != "/api/v1/review/inline":
            self._json(200, {})
            return
        request = json.loads(body)
        data = base64.b64decode(request["content_base64"], validate=True)
        self.received.append(
            {"filename": request["filename"], "sha256": hashlib.sha256(data).hexdigest()}
        )
        try:
            result = review.review_bytes(data, request["filename"])
        except review.ReviewError as refused:
            self._json(refused.status, {"detail": str(refused)})
            return
        self.answered.append(result)
        self._json(200, result)


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

  const picker = page.locator('input[type="file"][accept*=".wav"]');
  check(await picker.count() === 1, 'the page has one REVIEW A FILE picker');
  check(await page.getByText('REVIEW A FILE').count() >= 1, 'the picker is labelled REVIEW A FILE');

  // A real file, chosen in the page's own picker.
  await picker.setInputFiles(process.env.SAMPLE);
  await page.waitForFunction(() => window.MercuryConsoleApp.logic.state.results.some((x) => x.kind === 'REVIEW'),
    null, { timeout: 180000 });
  const shown = await page.evaluate(() => {
    const L = window.MercuryConsoleApp.logic, r = L.state.results.find((x) => x.kind === 'REVIEW');
    return { route: r.route, status: r.status, http: r.http, label: r.label, headline: r.sum.headline,
      kv: r.sum.kv, body: r.body, decision: r.data.decision, sha: r.data.review.sha256,
      terminal: L.state.lines.map((l) => JSON.stringify(l)).join('\\n'), ws: L.state.ws };
  });
  console.log('SHOWN ' + JSON.stringify({ headline: shown.headline, sha: shown.sha, decision: shown.decision }));
  check(shown.route === 'POST /api/v1/review/inline' && shown.status === 'ok' && shown.http === 200,
    'the review is kept in RESULTS with its route', shown.route + ' ' + shown.status);
  check(/^(ABSTAINED|ANOMALOUS SEGMENT|NO ANOMALOUS SEGMENT) · \\d+ of \\d+ segments flagged · document$/.test(shown.headline),
    'the headline states the decision', shown.headline);
  const keys = shown.kv.map((kv) => kv[0]);
  check(['ANSWER 1 · WHY THIS SEGMENT', 'ANSWER 2 · HOW SURE', 'ANSWER 3 · THE RECORD'].every((k) => keys.includes(k)),
    'the three answers are shown', keys.join(' | ').slice(0, 400));
  check(keys.includes('certificate scope') && keys.includes('1 · WHY THIS ROW'),
    'the certificate scope and the decision panel are shown');
  check(shown.body && /bytes, base64\\)$/.test(shown.body.content_base64),
    'the kept request does not hold the file a second time', JSON.stringify(shown.body));
  check(shown.terminal.includes('review · ' + shown.headline), 'the terminal reports the review');
  check(shown.ws === 'res', 'the RESULTS workspace is opened');
  const text = shown.headline + shown.kv.map((kv) => kv.join(' ')).join(' ');
  check(!/\\[object Object\\]|undefined|NaN/.test(text), 'nothing renders as a placeholder');

  // A file the server refuses: reported with the reason, nothing invented. The console is
  // now showing RESULTS (the picker lives in the workspace it left), so the same handler the
  // picker calls is given the file directly.
  await page.evaluate(() => {
    const bytes = new Uint8Array(69); bytes.set([0x49, 0x44, 0x33, 4, 0]);
    window.MercuryConsoleApp.logic.reviewFile(new File([bytes], 'voice.mp3', { type: 'audio/mpeg' }));
  });
  await page.waitForFunction(() => window.MercuryConsoleApp.logic.state.results.filter((x) => x.kind === 'REVIEW').length === 2,
    null, { timeout: 60000 });
  const refused = await page.evaluate(() => {
    const L = window.MercuryConsoleApp.logic, all = L.state.results.filter((x) => x.kind === 'REVIEW');
    const r = all.find((x) => x.status === 'error');
    return { found: !!r, http: r && r.http, headline: r && r.sum.headline, kv: r ? r.sum.kv : null,
      terminal: L.state.lines.map((l) => JSON.stringify(l)).join('\\n') };
  });
  check(refused.found && refused.http === 415, 'the refusal is kept as an error with its status', JSON.stringify(refused.http));
  check(/not reviewed · 415 · .*compressed/.test(refused.headline || ''), 'the server\\'s reason is shown', refused.headline);
  check(Array.isArray(refused.kv) && refused.kv.length === 0, 'no review is invented for it');
  check(refused.terminal.includes('not reviewed · 415'), 'the terminal reports the refusal');
  check(pageErrors.length === 0, 'no page errors', pageErrors.join(' | '));

  await browser.close();
  console.log(failed === 0 ? 'console review: all checks passed' : 'console review: ' + failed + ' check(s) FAILED');
  process.exit(failed === 0 ? 0 : 1);
})().catch((error) => { console.error('FAILED', error); process.exit(1); });
"""


def test_a_file_chosen_in_the_console_is_reviewed_by_the_engine(tmp_path: Path) -> None:
    _ReviewingEngine.received.clear()
    _ReviewingEngine.answered.clear()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ReviewingEngine)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    driver = tmp_path / "console_review_check.js"
    driver.write_text(_DRIVER, encoding="utf-8")
    try:
        check = subprocess.run(
            [str(NODE), str(driver)],
            capture_output=True,
            text=True,
            timeout=420,
            check=False,
            env={**node_env(), "BASE": f"http://127.0.0.1:{port}", "SAMPLE": str(SAMPLE)},
        )
        report = check.stdout + check.stderr
        assert check.returncode == 0, report
        assert "console review: all checks passed" in report
        assert "FAIL" not in report
    finally:
        server.shutdown()
        thread.join(timeout=10)
        server.server_close()

    # The bytes the page sent are the file's, and the decision it showed is the engine's.
    digest = hashlib.sha256(SAMPLE.read_bytes()).hexdigest()
    assert _ReviewingEngine.received[0] == {"filename": SAMPLE.name, "sha256": digest}
    assert _ReviewingEngine.received[1]["filename"] == "voice.mp3"
    shown = json.loads(next(line for line in report.splitlines() if line.startswith("SHOWN "))[6:])
    answered = _ReviewingEngine.answered[0]
    assert shown["sha"] == digest == answered["review"]["sha256"]
    assert shown["decision"] == json.loads(json.dumps(answered["decision"], default=str))

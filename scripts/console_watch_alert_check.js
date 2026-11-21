// Copyright (C) 2025 Steel Security Advisors LLC
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Mercury Console: a finding the watch reports reaches an open console.
//
// Opens the served console (/console) in headless Chromium, enters it as a
// person does, waits until its alert stream is connected, then asks the
// server for a finding (by creating the file named in TRIGGER, which
// tests/api/console_alert_server.py waits for) and watches it arrive:
//
//   * the alert banner is drawn, is the topmost thing where it sits (not
//     under the splash or a dialog), and carries the headline and what to do;
//   * a screen reader is told (the assertive live region);
//   * the terminal carries the alert row and the agent panel the entry;
//   * dismissing the banner removes it and leaves the record;
//   * GET /api/v1/watch on the same server shows the finding as standing.
//
// Usage (the server must serve the console; MERCURY_FRONTEND_ENABLED=true):
//   BASE=http://127.0.0.1:8765 TRIGGER=/tmp/x/report node scripts/console_watch_alert_check.js
//   OUT=/tmp/shots adds a screenshot of the banner. Exit status 0 when every
//   check passed; each check prints one "ok" or "FAIL" line.
'use strict';
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

const BASE = process.env.BASE || 'http://127.0.0.1:8765';
const TRIGGER = process.env.TRIGGER;
const OUT = process.env.OUT;
const HEADLINE = process.env.HEADLINE || 'Major flooding: St. Johns River near Cocoa, Florida';
const ACTION = process.env.ACTION || "turn around, don't drown";

let failed = 0;
function check(ok, what, detail) {
  if (!ok) { failed += 1; }
  console.log((ok ? 'ok   ' : 'FAIL ') + what + (detail ? ' :: ' + detail : ''));
}

(async () => {
  if (!TRIGGER) { throw new Error('TRIGGER (a file path the server waits for) is required'); }
  const browser = await chromium.launch(process.env.CHROMIUM ? { executablePath: process.env.CHROMIUM } : {});
  const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
  const pageErrors = [];
  page.on('pageerror', (error) => pageErrors.push(error.message));

  await page.goto(BASE + '/console');
  await page.waitForFunction(() => window.MercuryConsoleApp && window.MercuryConsoleApp.logic);
  // The console opens on its splash; a person enters it.
  await page.getByRole('button', { name: /enter console/i }).click();
  await page.waitForFunction(() => {
    const logic = window.MercuryConsoleApp.logic;
    return logic.live() && logic.state.alerts && logic.state.alerts.st === 'live';
  }, null, { timeout: 60000 });
  check(!(await page.evaluate(() => window.MercuryConsoleApp.logic.state.alertBanner)),
    'no banner before anything is reported');

  fs.writeFileSync(TRIGGER, 'report\n');
  await page.waitForFunction(() => window.MercuryConsoleApp.logic.state.alertBanner, null, { timeout: 60000 });

  // What a person sees: every leaf that carries the headline, and whether it
  // is the topmost element at its own centre.
  const seen = await page.evaluate((headline) => {
    const leaves = Array.from(document.querySelectorAll('#mercury-console *'))
      .filter((el) => el.children.length === 0 && (el.textContent || '').includes(headline));
    return leaves.map((el) => {
      const box = el.getBoundingClientRect();
      const top = document.elementFromPoint(box.left + Math.min(box.width / 2, 40), box.top + box.height / 2);
      const region = el.closest('[role="alert"], [role="status"], [aria-live]');
      return {
        text: el.textContent,
        onTop: !!top && (top === el || el.contains(top) || top.contains(el)),
        inView: box.width > 0 && box.height > 0 && box.top >= 0 && box.bottom <= window.innerHeight,
        role: region ? (region.getAttribute('role') || 'aria-live=' + region.getAttribute('aria-live')) : null,
      };
    });
  }, HEADLINE);
  const banner = seen.find((node) => node.onTop && node.inView && node.text.startsWith(HEADLINE));
  check(!!banner, 'the banner is drawn, in view and on top', JSON.stringify(seen.map((n) => [n.onTop, n.inView, n.role])));
  check(!!banner && banner.text.includes(ACTION), 'the banner says what to do');
  check(seen.some((node) => node.text.startsWith('Mercury alert: ' + HEADLINE) && /assertive/.test(node.role || '')),
    'a screen reader is told (assertive live region)');
  check(seen.some((node) => node.text.includes('◆ Mercury · watch severe') && node.text.includes(HEADLINE)),
    'the terminal carries the alert row');

  const state = await page.evaluate(() => {
    const s = window.MercuryConsoleApp.logic.state;
    return {
      chat: s.chat.filter((entry) => entry.proactive).map((entry) => ({ steps: entry.steps, text: entry.text })),
      results: (s.results || []).filter((r) => /ALERT/.test(r.kind || '')).map((r) => r.label),
    };
  });
  check(state.chat.length === 1 && state.chat[0].text.startsWith(HEADLINE) &&
    /^proactive · watch severe · severity 1\.00 · watch:river_gauges/.test(state.chat[0].steps[0]),
  'the agent panel has the entry, once', JSON.stringify(state.chat.map((c) => c.steps)));
  check(state.results.length === 1 && state.results[0] === 'Alert · watch severe', 'it is in the results', JSON.stringify(state.results));
  if (OUT) { fs.mkdirSync(OUT, { recursive: true }); await page.screenshot({ path: path.join(OUT, 'console-watch-alert.png') }); }

  // Dismissing the banner removes it and leaves the record.
  const dismiss = page.getByRole('button', { name: /dismiss/i }).first();
  check(await dismiss.count() === 1, 'the banner can be dismissed by a named control');
  if (await dismiss.count()) {
    await dismiss.click();
    await page.waitForFunction(() => !window.MercuryConsoleApp.logic.state.alertBanner, null, { timeout: 10000 });
    const after = await page.evaluate((headline) => Array.from(document.querySelectorAll('#mercury-console *'))
      .filter((el) => el.children.length === 0 && (el.textContent || '').includes(headline)).length, HEADLINE);
    check(after >= 1, 'the record stays after the banner is dismissed', String(after));
  }

  const recorded = await page.evaluate(async () => (await fetch('/api/v1/watch?open_only=true')).json());
  const standing = (recorded.findings || []).filter((finding) => !finding.ended);
  check(standing.length === 1 && standing[0].headline === HEADLINE && String(standing[0].action).includes(ACTION),
    'GET /api/v1/watch shows it standing, with what to do', JSON.stringify(standing.map((f) => f.headline)));
  check(pageErrors.length === 0, 'no page errors', pageErrors.join(' | '));

  await browser.close();
  console.log(failed === 0 ? 'console watch alert: all checks passed' : 'console watch alert: ' + failed + ' check(s) FAILED');
  process.exit(failed === 0 ? 0 : 1);
})().catch((error) => { console.error('FAILED', error); process.exit(1); });

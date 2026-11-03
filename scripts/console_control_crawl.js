// Copyright (C) 2025 Steel Security Advisors LLC
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Mercury Console control crawl: find the controls that do nothing when clicked.
//
// Drives every view of the served console (/console) against a live engine in
// headless Chromium and clicks every interactive control it can reach: buttons,
// ARIA roles (tab, menuitem, option, checkbox, switch, radio), links, summaries,
// labels, selects, and any element styled cursor:pointer that is not already one
// of those. Before each click the console's state is restored to the view's
// baseline through the mounted logic instance, so a click is judged on its own;
// a control that a click reveals is queued and clicked in turn, to depth 2.
//
// A click "had an effect" when any of these changed: the DOM under #mercury-console
// (a MutationObserver), a key of the logic state (identity), the value or checked
// state of a form control, a network request other than /health and the alert
// stream, the clipboard, a download, a window.open or the page URL. A control with
// no effect is reported as
//   no-op (already active)   it carried aria-selected/checked/pressed/current="true"
//   file-picker              its label opens a file chooser, which headless cannot show
//   DEAD                     nothing observable happened
//   unreachable / path-broken  it could not be found again after the state restore
//
// Usage (the server must serve the console; MERCURY_FRONTEND_ENABLED=true):
//   BASE=http://127.0.0.1:8765 OUT=/tmp/crawl MAX_PER_VIEW=300 node scripts/console_control_crawl.js
//   VIEWS="Session,Engine" node scripts/console_control_crawl.js      # a subset of views
//   DEPTH=1 node scripts/console_control_crawl.js                     # do not follow revealed controls
// Writes OUT/crawl_audit.json ({results: {view: [rows]}, errors}) and prints one
// summary line per view. Exit status is 0 when the crawl completed; read the
// DEAD rows, replay each one by hand, and fix or explain it (docs/design/
// mercury_console/README.md, section 21). Requires Node with the playwright
// package and a Chromium it can launch (`npm i playwright && npx playwright
// install chromium`); CI does not run it, it is a manual audit tool.
//
// Honest limits: "no effect" is a lead, not a verdict. The crawl of
// 2026-11-02 reported 16 DEAD rows after the fixes on this branch; 14 replayed
// to a control whose state the click would set again (a chip already removed,
// a command already loaded, a dialog already open, a send button with an
// empty draft) and 2 exposed missing ARIA state, which were fixed. Budget
// about five minutes per view at MAX_PER_VIEW=300.
const { chromium } = require('playwright');
const fs = require('fs');
const BASE = process.env.BASE || 'http://127.0.0.1:8765', OUT = process.env.OUT || '.';
const NAV = ["Terminal","Agent","Mass Run","Results","Captures","Record session","Commands","Runs","Detectors","Domains","Physics","Tools","AMA","FINDΩYOU™","MCP tools","AI vendors","API routes","Session","Diagnostics","Engine"];
const SEL = 'button, [role=button], [role=tab], [role=menuitem], [role=option], [role=checkbox], [role=switch], [role=radio], input[type=checkbox], input[type=radio], a[href], summary, label[for], select';
const MAX_PER_VIEW = +(process.env.MAX_PER_VIEW || 420), DEPTH = +(process.env.DEPTH || 2);
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1700, height: 1100 } });
  const net = []; page.on('request', (r) => net.push(r.method() + ' ' + r.url()));
  const errors = []; page.on('pageerror', (e) => errors.push(String(e.message || e)));
  page.on('dialog', (d) => d.dismiss().catch(() => {}));
  await page.goto(BASE + '/console', { waitUntil: 'load' });
  await page.waitForFunction(() => document.getElementById('mercury-console') && document.getElementById('mercury-console').dataset.ready === 'true', null, { timeout: 60000 });
  await page.waitForTimeout(1500);
  { const b = page.getByRole('button', { name: /ENTER CONSOLE/ }); if (await b.count()) { await b.first().click({ timeout: 3000 }); await page.waitForTimeout(1200); } }
  await page.evaluate(() => {
    window.__fx = { clip: 0, dl: 0, open: 0, muts: 0 };
    const w = navigator.clipboard && navigator.clipboard.writeText;
    if (navigator.clipboard) navigator.clipboard.writeText = function () { window.__fx.clip++; return w ? w.apply(navigator.clipboard, arguments).catch(() => {}) : Promise.resolve(); };
    const co = URL.createObjectURL; URL.createObjectURL = function () { window.__fx.dl++; return co.apply(URL, arguments); };
    window.open = function () { window.__fx.open++; return null; };
    new MutationObserver((ms) => { window.__fx.muts += ms.length; }).observe(document.getElementById('mercury-console'), { subtree: true, childList: true, attributes: true, characterData: true });
    window.__txt = (el) => (el.getAttribute('aria-label') || el.innerText || el.value || el.title || '').trim().replace(/\s+/g, ' ').slice(0, 70);
    window.__pool = (SEL) => {
      const base = Array.from(document.querySelectorAll(SEL));
      const ptr = Array.from(document.querySelectorAll('#mercury-console *')).filter((el) => getComputedStyle(el).cursor === 'pointer' && !el.closest(SEL) && !el.querySelector(SEL) && el.getBoundingClientRect().width >= 8 && el.getBoundingClientRect().height >= 8).map((el) => { el.__ptr = true; return el; });
      return base.concat(ptr);
    };
    window.__desc = (SEL) => { const seen = {}; return window.__pool(SEL).filter((el) => { const r = el.getBoundingClientRect(); return r.width > 0 && r.height > 0 && !(el.disabled || el.getAttribute('aria-disabled') === 'true'); }).map((el) => { const tag = el.tagName.toLowerCase() + (el.__ptr ? '.ptr' : ''), text = window.__txt(el), k = tag + '|' + text, occ = seen[k] || 0; seen[k] = occ + 1; const r = el.getBoundingClientRect(); return { tag, text, occ, role: el.getAttribute('role') || '', type: el.getAttribute('type') || '', x: Math.round(r.x), y: Math.round(r.y), active: el.getAttribute('aria-selected') === 'true' || el.getAttribute('aria-checked') === 'true' || !!el.getAttribute('aria-current') || el.getAttribute('aria-pressed') === 'true' }; }); };
    window.__find = (SEL, m) => { const pool = window.__pool(SEL).filter((el) => (el.tagName.toLowerCase() + (el.__ptr ? '.ptr' : '')) === m.tag && window.__txt(el) === m.text && el.getBoundingClientRect().width > 0); return pool[Math.min(m.occ || 0, pool.length - 1)] || null; };
    window.__click = (SEL, m) => { const el = window.__find(SEL, m); if (!el) return 'missing'; try { el.scrollIntoView({ block: 'nearest' }); el.click(); return 'ok'; } catch (e) { return 'error: ' + e; } };
    window.__form = () => JSON.stringify(Array.from(document.querySelectorAll('input,select,textarea')).map((e) => (e.type === 'checkbox' || e.type === 'radio') ? e.checked : e.value));
    window.__mark = () => { window.__B = Object.assign({}, window.MercuryConsoleApp.logic.state); window.__F = window.__form(); };
    window.__changed = () => { const A = window.MercuryConsoleApp.logic.state, B = window.__B || {}; const keys = Object.keys(A).filter((k) => A[k] !== B[k]).concat(Object.keys(B).filter((k) => !(k in A))); if (window.__form() !== window.__F) keys.push('<form>'); return keys; };
    window.__snap = () => { const L = window.MercuryConsoleApp.logic; window.__S = Object.assign({}, L.state); };
    window.__restore = () => { const L = window.MercuryConsoleApp.logic; const cur = Object.keys(L.state), patch = Object.assign({}, window.__S); cur.forEach((k) => { if (!(k in patch)) patch[k] = undefined; }); L.setState(patch); };
  });
  const navTo = async (label) => {
    const esc = label.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const el = page.getByRole('button', { name: new RegExp('^\\S{0,2}\\s*' + esc + '(\\s+\\d+)?$') }).first();
    if (!(await el.count())) return false;
    try { await el.click({ timeout: 2000 }); } catch (e) { return false; }
    await page.waitForTimeout(350); return true;
  };
  const key = (m) => m.tag + '|' + m.text + '|' + m.occ;
  const isNav = (m) => m.x < 250 && NAV.some((n) => m.text.includes(n)) || /ENTER CONSOLE|Collapse navigation|Expand navigation/.test(m.text);
  const results = {};
  for (const view of (process.env.VIEWS ? NAV.filter((v) => process.env.VIEWS.split(",").includes(v)) : NAV)) {
    if (!(await navTo(view))) { results[view] = { error: 'nav item not found' }; continue; }
    await page.evaluate(() => window.__snap());
    const base = (await page.evaluate((SEL) => window.__desc(SEL), SEL)).filter((m) => !isNav(m));
    const seen = new Set(base.map(key)); const queue = base.map((m) => ({ path: [], m })); const rows = [];
    while (queue.length && rows.length < MAX_PER_VIEW) {
      const { path, m } = queue.shift();
      await page.evaluate(() => window.__restore()); await page.waitForTimeout(250);
      let reach = 'ok';
      for (const p of path) { reach = await page.evaluate(([SEL, p]) => window.__click(SEL, p), [SEL, p]); if (reach !== 'ok') break; await page.waitForTimeout(300); }
      if (reach !== 'ok') { rows.push(Object.assign({}, m, { path: path.map((p) => p.text), verdict: 'path-broken' })); continue; }
      await page.evaluate(() => { window.__fx.clip = 0; window.__fx.dl = 0; window.__fx.open = 0; window.__fx.muts = 0; window.__mark(); });
      const n0 = net.length, e0 = errors.length, url0 = page.url();
      const r = await page.evaluate(([SEL, m]) => window.__click(SEL, m), [SEL, m]);
      if (r !== 'ok') { rows.push(Object.assign({}, m, { path: path.map((p) => p.text), verdict: 'unreachable: ' + r })); continue; }
      await page.waitForTimeout(450);
      const fx = await page.evaluate(() => window.__fx); const changed = await page.evaluate(() => window.__changed());
      const dnet = net.slice(n0).filter((u) => !/\/health$|\/api\/v1\/voice\/alerts/.test(u));
      const effect = fx.muts > 0 || changed.length || dnet.length || fx.clip || fx.dl || fx.open || page.url() !== url0;
      const verdict = effect ? 'effect' : m.active ? 'no-op (already active)' : /IMPORT|LOAD FILE|CHOOSE FILE|\bBROWSE\b/.test(m.text) ? 'file-picker (no effect headless)' : 'DEAD';
      rows.push(Object.assign({}, m, { path: path.map((p) => p.text), verdict, muts: fx.muts, changed: changed.slice(0, 8), net: dnet.slice(0, 3), clip: fx.clip, dl: fx.dl, errors: errors.slice(e0) }));
      if (effect && path.length < DEPTH) {
        const now = (await page.evaluate((SEL) => window.__desc(SEL), SEL)).filter((x) => !isNav(x));
        for (const x of now) { const k = key(x); if (!seen.has(k)) { seen.add(k); queue.push({ path: path.concat([m]), m: x }); } }
      }
      await page.keyboard.press('Escape').catch(() => {});
    }
    results[view] = rows;
    const c = (v) => rows.filter((r) => r.verdict.startsWith(v)).length;
    console.log(`${view}: ${rows.length} controls (${queue.length} unvisited) · dead ${c('DEAD')} · no-op ${c('no-op')} · unreachable ${c('unreachable') + c('path-broken')}`);
    fs.writeFileSync(OUT + '/crawl_audit.json', JSON.stringify({ results, errors }, null, 1));
  }
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });

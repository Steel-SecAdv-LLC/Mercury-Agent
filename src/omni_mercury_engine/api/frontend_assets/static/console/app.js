/* Copyright (C) 2025 Steel Security Advisors LLC
   SPDX-License-Identifier: AGPL-3.0-or-later
   Mercury Console application. Rebuilt in plain JavaScript from the design prototype
   (docs/design/mercury_console/design/Mercury CLI.dc.html): the DOM is built once and updated in place, every
   listener is attached with addEventListener, dynamic geometry goes through the CSSOM, and every string that came
   from the server or a file is written with textContent. Chart SVG comes from viz.js, which emits no inline style. */
(function () {
  "use strict";

  const CAT = window.MERCURY_CLI;
  const A = window.MercuryCLIAPI;
  const V = window.MercuryCLIViz;
  const M = window.MercuryCLIMass;
  const G = window.MercuryCLIAgent;
  const K = window.MercuryCLICapture;
  const root = document.getElementById("mercury-console");
  if (!root || !CAT || !A) return;

  /* ================================================================== utilities */

  const $ = (id) => document.getElementById(id);
  function h(tag, attrs, ...kids) {
    const el = document.createElement(tag);
    if (attrs) {
      for (const [k, v] of Object.entries(attrs)) {
        if (v == null || v === false) continue;
        if (k === "class") el.className = v;
        else if (k === "text") el.textContent = v;
        else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
        else if (k === "dataset") Object.assign(el.dataset, v);
        else el.setAttribute(k, v === true ? "" : String(v));
      }
    }
    for (const kid of kids.flat(Infinity)) {
      if (kid == null || kid === false) continue;
      el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
    }
    return el;
  }
  const svgNode = (markup) => { const box = h("div", { class: "mc-chartbox" }); box.innerHTML = markup; return box; }; // viz.js output only
  const clear = (el) => { while (el.firstChild) el.removeChild(el.firstChild); return el; };
  const shq = (x) => (/^[\w./:@%+=,-]+$/.test(String(x)) ? String(x) : "'" + String(x).replace(/'/g, "'\\''") + "'");
  const f4 = (x) => (x == null || !isFinite(x) ? "—" : Math.abs(x) >= 1e5 || (Math.abs(x) < 1e-3 && x !== 0) ? Number(x).toExponential(3) : String(+Number(x).toFixed(4)));
  const nowIso = () => new Date().toISOString();
  const uid = (p) => p + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
  const plural = (n, w) => n + " " + w + (n === 1 ? "" : "s");
  function announce(text) { const el = $("mc-live"); el.textContent = ""; setTimeout(() => { el.textContent = text; }, 30); }
  let toastTimer = 0;
  function toast(text) {
    const old = root.querySelector(".mc-toast"); if (old) old.remove();
    const el = h("div", { class: "mc-toast", role: "status", text }); root.append(el);
    clearTimeout(toastTimer); toastTimer = setTimeout(() => el.remove(), 3200);
  }
  function copy(text) {
    const done = () => toast("Copied to the clipboard");
    try { navigator.clipboard.writeText(String(text)).then(done, () => toast("Copy failed: select and copy by hand")); } catch (e) { toast("Copy failed: select and copy by hand"); }
  }
  function download(name, mime, data) {
    const blob = data instanceof Blob ? data : new Blob([data], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = h("a", { href: url, download: name }); document.body.append(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 4000);
  }
  // A CSV cell that starts with = + - @ (or a control) is a formula to a spreadsheet; neutralise it (OWASP CSV injection).
  const csvSafe = (v) => { if (typeof v !== "string") return v; return /^[=+\-@\t\r]/.test(v) ? "'" + v : v; };

  /* ================================================================== persistence */

  const LS_STATE = "mercury_cli_state_v1";
  const LS_SETTINGS = "mercury-settings";
  function readJson(key, fallback) { try { const v = JSON.parse(localStorage.getItem(key) || "null"); return v == null ? fallback : v; } catch (e) { return fallback; } }
  function writeJson(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); return true; } catch (e) { return false; } }

  /* ================================================================== catalog indexes */

  const commands = CAT.commands.filter((c) => c.path);
  const cmdBy = (path) => commands.find((c) => c.path === path) || null;
  const childrenOf = (path) => commands.filter((c) => { const p = c.path.split(" "); return p.length === (path ? path.split(" ").length + 1 : 1) && (!path || c.path.startsWith(path + " ")); });
  const routeKeys = new Set(Object.keys(CAT.schemas || {}));
  const apiPaths = new Set((CAT.api || []).map((r) => r.m + " " + r.path));
  const VOCAB = CAT.vocab || {};
  const LONG_RUNNING = new Set(["serve", "mcp", "stream", "train", "tune", "build-features"]);

  /* ================================================================== state */

  const saved = readJson(LS_STATE, {});
  const state = {
    view: "terminal",
    sel: typeof saved.sel === "string" && cmdBy(saved.sel) ? saved.sel : "detect",
    vals: saved.vals && typeof saved.vals === "object" ? saved.vals : {},
    history: Array.isArray(saved.history) ? saved.history.slice(-200) : [],
    domain: typeof saved.domain === "string" ? saved.domain : "",
    samples: saved.samples || "",
    results: Array.isArray(saved.results) ? saved.results.slice(0, 30) : [],
    resultSel: null,
    chartSel: {},
    catSel: Array.isArray(saved.catSel) ? saved.catSel : [],
    capTag: saved.capTag || "review",
    capPad: saved.capPad == null ? 2 : saved.capPad,
    capView: "caps", capSel: null, recSel: null, capFlash: null,
    caps: [], recs: [], rec: null, recNote: "", recDiscardArm: false,
    data: null,
    panel: false, panelTab: "chat", ref: null,
    chat: [], chatDomain: null, chatOffline: false, sessionId: "console-" + Math.random().toString(36).slice(2, 10),
    health: null, voice: null,
    filters: {},
    optsOpen: false,
    introClosed: !!saved.introClosed,
    mass: { sel: Array.isArray(saved.massSel) ? saved.massSel : ["uni-z", "uni-iqr", "multi"], out: null, running: false, batch: [] },
    running: null,
  };
  const loadedCaps = K ? K.load() : { caps: [], recs: [] };
  state.caps = loadedCaps.caps; state.recs = loadedCaps.recs;

  let saveTimer = 0;
  function persist() {
    clearTimeout(saveTimer);
    saveTimer = setTimeout(() => {
      const results = state.results.slice(0, 30).map((r) => {
        const big = r.ds && r.ds.matrix && r.ds.matrix.length * (r.ds.cols || 1) > 20000;
        return big ? Object.assign({}, r, { ds: Object.assign({}, r.ds, { matrix: null, trimmed: true }) }) : r;
      });
      const body = { sel: state.sel, vals: state.vals, history: state.history.slice(-200), domain: state.domain, samples: state.samples, results, catSel: state.catSel, capTag: state.capTag, capPad: state.capPad, introClosed: state.introClosed, massSel: state.mass.sel };
      if (!writeJson(LS_STATE, body)) writeJson(LS_STATE, Object.assign(body, { results: results.slice(0, 5) }));
      if (K) K.save(C);
    }, 250);
  }

  /* The component handle capture.js drives (it was written against the prototype's component). */
  const C = {
    get state() { return state; },
    A,
    _capsLoaded: true,
    setState(patch) {
      const p = typeof patch === "function" ? patch(state) : patch;
      if (!p) return;
      if (p.ws === "caps") { delete p.ws; p.view = "captures"; }
      Object.assign(state, p);
      persist(); render();
    },
    append(rows) { rows.forEach((r) => termRow(r)); },
    row(text, color) { return { text, color }; },
    csvSafe,
    download,
    cmdBy,
    live: () => !!(state.health && state.health.ok),
    recEv(k, x) { if (!state.rec) return; state.rec.events.push({ t: Date.now() - state.rec.t0, k, x: String(x).slice(0, 4000) }); },
  };

  /* ================================================================== command line */

  function tokenize(line) {
    const out = []; let cur = "", q = null, any = false;
    for (let i = 0; i < line.length; i++) {
      const ch = line[i];
      if (q) { if (ch === q) q = null; else if (ch === "\\" && q === '"' && i + 1 < line.length) cur += line[++i]; else cur += ch; continue; }
      if (ch === "'" || ch === '"') { q = ch; any = true; continue; }
      if (ch === "\\" && i + 1 < line.length) { cur += line[++i]; any = true; continue; }
      if (/\s/.test(ch)) { if (cur || any) { out.push(cur); cur = ""; any = false; } continue; }
      cur += ch; any = true;
    }
    if (cur || any) out.push(cur);
    return { tokens: out, open: !!q };
  }
  const optLabel = (o) => o.flags.filter((f) => !f.startsWith("--no-")).map((f) => "'" + f + "'").join(" / ");
  function build(path, vals) {
    const c = cmdBy(path); if (!c) return "mercury-agent " + path;
    const parts = ["mercury-agent", path];
    const v = vals || {};
    (v._args || []).forEach((a) => { if (a !== "" && a != null) parts.push(shq(a)); });
    c.opts.forEach((o) => {
      const val = v[o.k];
      if (val == null || val === "" || val === false) return;
      if (o.flag) { if (val === true) parts.push(o.k); return; }
      (o.multiple && Array.isArray(val) ? val : [val]).forEach((x) => parts.push(o.k, shq(x)));
    });
    return parts.join(" ");
  }
  function parse(line) {
    const { tokens, open } = tokenize(line.trim());
    const errs = [];
    if (open) errs.push("No closing quotation");
    let t = tokens.slice();
    if (t[0] === "mercury-agent") t = t.slice(1);
    let path = "", i = 0;
    while (i < t.length) { const next = path ? path + " " + t[i] : t[i]; if (cmdBy(next)) { path = next; i++; } else break; }
    const c = cmdBy(path);
    if (!c) return { path: "", cmd: null, vals: {}, errs: t.length ? ["No such command '" + (t[0] || "") + "'."] : [], rest: t };
    const vals = { _args: [] };
    const byFlag = new Map(); c.opts.forEach((o) => o.flags.forEach((f) => byFlag.set(f, o)));
    for (; i < t.length; i++) {
      let tok = t[i], inline = null;
      if (tok.startsWith("--") && tok.includes("=")) { inline = tok.slice(tok.indexOf("=") + 1); tok = tok.slice(0, tok.indexOf("=")); }
      if (tok === "--help") { vals._help = true; continue; }
      if (tok.startsWith("-") && tok.length > 1 && !/^-\d/.test(tok)) {
        const o = byFlag.get(tok);
        if (!o) { errs.push("No such option: " + tok); continue; }
        if (o.flag) { vals[o.k] = !tok.startsWith("--no-"); continue; }
        const val = inline != null ? inline : t[++i];
        if (val == null) { errs.push("Option '" + tok + "' requires an argument."); continue; }
        if (o.multiple) (vals[o.k] = Array.isArray(vals[o.k]) ? vals[o.k] : []).push(val); else vals[o.k] = val;
      } else if (c.kind === "group" && !c.args) {
        errs.push("No such command '" + tok + "'.");
      } else if (!c.args || vals._args.length >= c.args.reduce((n, a) => n + (a.nargs === -1 ? Infinity : a.nargs || 1), 0)) {
        errs.push("Got unexpected extra argument (" + tok + ")");
      } else vals._args.push(tok);
    }
    errs.push(...validate(c, vals, false));
    return { path, cmd: c, vals, errs, rest: [] };
  }
  function validate(c, vals, requireAll) {
    const errs = [];
    c.opts.forEach((o) => {
      const v = vals[o.k];
      if (v == null || v === "") { if (requireAll && o.req) errs.push("Missing option " + optLabel(o) + "."); return; }
      if (o.flag) return;
      (Array.isArray(v) ? v : [v]).forEach((x) => {
        if (o.choices && !o.choices.includes(String(x))) errs.push("Invalid value for " + optLabel(o) + ": '" + x + "' is not one of " + o.choices.map((q) => "'" + q + "'").join(", ") + ".");
        else if (o.type === "float" && !isFinite(parseFloat(x))) errs.push("Invalid value for " + optLabel(o) + ": '" + x + "' is not a valid float.");
        else if (o.type === "int" && !/^-?\d+$/.test(String(x))) errs.push("Invalid value for " + optLabel(o) + ": '" + x + "' is not a valid integer.");
      });
    });
    (c.args || []).forEach((a, k) => { if (requireAll && a.req && !(vals._args || [])[k]) errs.push("Missing argument '" + String(a.name).toUpperCase() + "'."); });
    return errs;
  }
  function completions(line) {
    const { tokens } = tokenize(line);
    const endsSpace = /\s$/.test(line);
    let t = tokens.slice(); if (t[0] === "mercury-agent") t = t.slice(1);
    const partial = endsSpace ? "" : t.pop() || "";
    let path = ""; for (const tok of t) { const nx = path ? path + " " + tok : tok; if (cmdBy(nx)) path = nx; else break; }
    const c = cmdBy(path);
    const prev = t[t.length - 1];
    if (c && prev && prev.startsWith("-")) {
      const o = c.opts.find((x) => x.flags.includes(prev));
      if (o && !o.flag) {
        const choices = o.choices || vocabValues(c.path + " " + o.k);
        if (choices.length) return choices.filter((x) => x.startsWith(partial));
      }
    }
    if (partial.startsWith("-") && c) return c.opts.flatMap((o) => o.flags).filter((f) => f.startsWith(partial));
    const subs = childrenOf(path).map((x) => x.path.split(" ").pop()).filter((x) => x.startsWith(partial));
    if (subs.length) return subs;
    if (c && !partial) return c.opts.filter((o) => state.vals[c.path] == null || state.vals[c.path][o.k] == null).map((o) => o.k);
    return [];
  }
  function vocabValues(key) { const v = VOCAB[key]; return v ? v.groups.flatMap((g) => g.items.map((x) => x[0])) : []; }

  /* ================================================================== datasets */

  function parseDataset(name, text) {
    const trimmed = text.trim();
    let rows = [], names = [];
    if (/^[[{]/.test(trimmed)) {
      let j; try { j = JSON.parse(trimmed); } catch (e) { return { error: "Not valid JSON: " + e.message }; }
      if (j && !Array.isArray(j) && Array.isArray(j.data)) j = j.data;
      if (!Array.isArray(j) || !j.length) return { error: "Expected a JSON array (numbers, rows or records)." };
      if (j.every((x) => typeof x === "number")) { rows = j.map((x) => [x]); names = ["value"]; }
      else if (j.every(Array.isArray)) { rows = j.map((r) => r.map(Number)); names = rows[0].map((_, k) => "col" + (k + 1)); }
      else if (j.every((x) => x && typeof x === "object")) {
        names = Object.keys(j[0]).filter((k) => j.every((r) => isFinite(Number(r[k])) && r[k] !== null && r[k] !== ""));
        rows = j.map((r) => names.map((k) => Number(r[k])));
      } else return { error: "Mixed JSON array: use numbers, equal-length rows, or records." };
    } else {
      const lines = trimmed.split(/\r?\n/).filter((l) => l.trim());
      const delim = lines[0].includes("\t") ? "\t" : lines[0].includes(";") && !lines[0].includes(",") ? ";" : ",";
      const split = (l) => l.split(delim).map((x) => x.trim().replace(/^"|"$/g, ""));
      let head = split(lines[0]);
      const isHeader = head.some((x) => x !== "" && !isFinite(Number(x)));
      const body = (isHeader ? lines.slice(1) : lines).map(split);
      if (!isHeader) head = head.map((_, k) => "col" + (k + 1));
      const keep = head.map((_, k) => body.length > 0 && body.every((r) => r[k] !== undefined && r[k] !== "" && isFinite(Number(r[k]))));
      names = head.filter((_, k) => keep[k]);
      rows = body.map((r) => r.filter((_, k) => keep[k]).map(Number));
    }
    rows = rows.filter((r) => r.length === names.length && r.every((x) => isFinite(x)));
    if (!names.length || !rows.length) return { error: "No numeric columns found. Use a CSV with a header row, or a JSON array of numbers, rows or records." };
    return { name, rows: rows.length, cols: names.length, names, matrix: rows, total: rows.length };
  }
  function pickFile(accept) {
    return new Promise((ok) => {
      const inp = h("input", { type: "file", accept: accept || ".csv,.json,.tsv,.txt", class: "mc-sr", "aria-hidden": "true" });
      inp.addEventListener("change", () => { const f = inp.files && inp.files[0]; inp.remove(); ok(f || null); });
      document.body.append(inp); inp.click();
    });
  }
  async function loadDataset() {
    const f = await pickFile();
    if (!f) return;
    if (f.size > 25 * 1024 * 1024) { termRow({ text: "That file is " + (f.size / 1048576).toFixed(1) + " MB; the console reads files up to 25 MB.", k: "err" }); return; }
    const d = parseDataset(f.name, await f.text());
    if (d.error) { termRow({ text: f.name + ": " + d.error, k: "err" }); return; }
    state.data = d;
    pushResult({ kind: "DATASET", label: "Dataset · " + d.name, route: "local file · " + d.rows + " × " + d.cols, status: "ok", http: null, ms: null, body: null, data: { dataset: true, rows: d.rows, cols: d.cols, columns: d.names }, sum: { headline: d.rows + " rows × " + d.cols + " numeric columns", kv: [["columns", d.names.slice(0, 12).join(", ") + (d.names.length > 12 ? " …" : "")]] }, ds: d });
    termRow({ text: "loaded " + d.name + " · " + d.rows + " rows × " + d.cols + " numeric columns (" + d.names.slice(0, 6).join(", ") + (d.cols > 6 ? ", …" : "") + ")", k: "ok" });
    const c = cmdBy(state.sel);
    if (c && c.opts.some((o) => o.k === "--input")) setVal(state.sel, "--input", d.name);
    C.recEv("out", "loaded " + d.name);
    renderNav();
  }
  const series = (d, col) => d.matrix.map((r) => r[Math.max(0, Math.min(d.cols - 1, col || 0))]);

  /* ================================================================== results */

  function pushResult(r) {
    const rec = Object.assign({ id: uid("r"), at: nowIso() }, r);
    if (!rec.sum && rec.data && M) rec.sum = M.summarize(rec.data);
    if (rec.ds && rec.ds.matrix === undefined && state.data) rec.ds = state.data;
    state.results.unshift(rec);
    state.results = state.results.slice(0, 30);
    state.resultSel = rec.id;
    C.recEv("result", rec.label + " · " + ((rec.sum && rec.sum.headline) || rec.status));
    persist(); renderNav();
    return rec;
  }
  const dsMeta = (d) => (d ? { name: d.name, rows: d.rows, cols: d.cols, names: d.names, matrix: d.matrix, total: d.total } : null);

  /* ================================================================== terminal */

  const termOut = $("mc-term-out");
  const promptIn = $("mc-prompt-in");
  const lines = [];
  function termRow(r) {
    const row = typeof r === "string" ? { text: r } : r;
    const el = row.act
      ? h("button", { class: "mc-line mc-line-act", type: "button", text: row.text, onclick: row.act })
      : h("p", { class: "mc-line", text: row.text || " " });
    if (row.k) el.dataset.k = row.k;
    if (row.color) el.style.color = row.color;
    termOut.append(el); lines.push(row);
    while (termOut.childNodes.length > 2000) { termOut.removeChild(termOut.firstChild); lines.shift(); }
    termOut.scrollTop = termOut.scrollHeight;
    if (row.k !== "cmd" && row.text) C.recEv("out", row.text);
  }
  function termRows(list) { list.forEach(termRow); }
  function banner() {
    if (state.introClosed) return;
    const close = h("button", { class: "mc-iconbtn mc-banner-x", type: "button", title: "Close", text: "×" });
    const box = h("div", { class: "mc-banner" },
      close,
      h("h2", { text: "MERCURY CONSOLE · " + CAT.version }),
      h("p", { class: "mc-help", text: CAT.nCommands + " commands, " + CAT.detectors.length + " detectors, " + CAT.loaders.length + " loaders, " + CAT.mcp.length + " MCP tools and " + CAT.api.length + " API routes, read from the engine's own code. Pick a command, fill its options (the prompt follows), and press Enter. Commands with an API route run against this server; the rest are prepared for a shell." }),
      h("p", { class: "mc-help mc-muted", text: "Load a CSV or JSON file with LOAD FILE, then run detect. :help lists console commands. Ctrl-K searches everything." }));
    close.addEventListener("click", () => { state.introClosed = true; persist(); box.remove(); });
    termOut.append(box);
  }
  function helpRows() {
    termRows([
      { text: "CONSOLE COMMANDS", k: "head" },
      { text: "  :help                     this list" },
      { text: "  :clear                    clear the output (Ctrl-L)" },
      { text: "  :load                     load a CSV/JSON dataset from your computer" },
      { text: "  :rec [start [title]|stop|discard]   record this session (asciicast v2 + Markdown)" },
      { text: "  :mark <note>              add a mark to the recording" },
      { text: "  :capture <from> [to] [anomaly|normal|review|evidence] [label]" },
      { text: "  :diag                     run the console self-test" },
      { text: "  :health                   probe the engine" },
      { text: "Anything else that is not a command is offered to Mercury Agent.", k: "dim" },
      { text: "" },
    ]);
  }

  /* options drawer */
  const optsBox = $("mc-opts");
  function setVal(path, k, v) {
    const vals = (state.vals[path] = Object.assign({}, state.vals[path] || {}));
    if (v === "" || v == null || v === false) delete vals[k]; else vals[k] = v;
    syncPrompt(); renderOpts(); persist();
  }
  function syncPrompt() { if (document.activeElement !== promptIn || promptIn.dataset.fromForm) promptIn.value = build(state.sel, state.vals[state.sel]); delete promptIn.dataset.fromForm; renderSuggest(); }
  function selectCommand(path, opts) {
    if (!cmdBy(path)) return;
    state.sel = path;
    if (!opts || !opts.keepPrompt) promptIn.value = build(path, state.vals[path]);
    state.optsOpen = true;
    renderOpts(); renderSuggest(); persist();
    if (state.view !== "terminal") go("terminal");
  }
  function renderOpts() {
    clear(optsBox);
    const c = cmdBy(state.sel); if (!c) return;
    const vals = state.vals[c.path] || {};
    const set = c.opts.filter((o) => vals[o.k] != null && vals[o.k] !== "").length;
    const missing = validate(c, vals, true).filter((e) => e.startsWith("Missing"));
    const bar = h("button", { class: "mc-opts-bar", type: "button", "aria-expanded": state.optsOpen ? "true" : "false", "aria-controls": "mc-opts-body" },
      h("span", { text: (state.optsOpen ? "▾" : "▸") + " OPTIONS" }),
      h("span", { class: "mc-muted", text: c.path + " · " + plural(c.opts.length, "option") + " · " + set + " set" }),
      missing.length ? h("span", { class: "mc-warn", text: missing.length + " required" }) : null,
      h("span", { class: "mc-spacer" }));
    bar.addEventListener("click", () => { state.optsOpen = !state.optsOpen; renderOpts(); });
    optsBox.append(bar);
    if (!state.optsOpen) return;
    const grip = h("div", { class: "mc-grip", role: "separator", "aria-orientation": "horizontal", "aria-label": "Resize options", tabindex: "0" });
    const body = h("div", { class: "mc-opts-body", id: "mc-opts-body" });
    if (optsBox.dataset.height) { body.dataset.dragged = "1"; body.style.height = optsBox.dataset.height + "px"; }
    dragResize(grip, body);
    const head = h("div", { class: "mc-row" },
      h("span", { class: "mc-help", text: c.doc || "" }),
      h("span", { class: "mc-spacer" }),
      c.opts.some((o) => o.k === "--input" || o.k === "--data") ? h("button", { class: "mc-btn", type: "button", text: state.data ? "File · " + state.data.name : "Load file", onclick: loadDataset }) : null,
      set ? h("button", { class: "mc-btn", type: "button", text: "Reset", onclick: () => { state.vals[c.path] = {}; syncPrompt(); renderOpts(); persist(); } }) : null);
    body.append(head);
    if (c.ex && c.ex.length) {
      body.append(h("div", { class: "mc-row" }, h("span", { class: "mc-label", text: "Examples" }),
        c.ex.map((ex) => h("button", { class: "mc-chip", type: "button", text: ex, onclick: () => { const p = parse(ex); if (p.cmd) { state.vals[p.path] = p.vals; selectCommand(p.path); } } }))));
    }
    if (c.args) c.args.forEach((a, k) => body.append(argRow(c, a, k)));
    const ordered = c.opts.slice().sort((a, b) => (b.req || 0) - (a.req || 0));
    ordered.forEach((o) => body.append(optRow(c, o, vals)));
    if (!c.opts.length && !c.args) body.append(h("p", { class: "mc-muted", text: c.kind === "group" ? "A command group. Its sub-commands: " + childrenOf(c.path).map((x) => x.path.split(" ").pop()).join(", ") : "No options." }));
    if (c.kind === "group") body.append(h("div", { class: "mc-row" }, childrenOf(c.path).map((x) => h("button", { class: "mc-chip", type: "button", text: x.path.split(" ").pop(), title: x.doc, onclick: () => selectCommand(x.path) }))));
    optsBox.append(grip, body);
  }
  function dragResize(grip, body) {
    let y0 = 0, h0 = 0;
    const move = (e) => { const hgt = Math.max(80, Math.min(window.innerHeight * 0.7, h0 - (e.clientY - y0))); body.style.height = hgt + "px"; body.dataset.dragged = "1"; optsBox.dataset.height = String(Math.round(hgt)); };
    const up = () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); };
    grip.addEventListener("pointerdown", (e) => { y0 = e.clientY; h0 = body.getBoundingClientRect().height; window.addEventListener("pointermove", move); window.addEventListener("pointerup", up); e.preventDefault(); });
    grip.addEventListener("keydown", (e) => { if (e.key !== "ArrowUp" && e.key !== "ArrowDown") return; const cur = body.getBoundingClientRect().height; const hgt = Math.max(80, cur + (e.key === "ArrowUp" ? 24 : -24)); body.style.height = hgt + "px"; body.dataset.dragged = "1"; optsBox.dataset.height = String(Math.round(hgt)); e.preventDefault(); });
  }
  function argRow(c, a, k) {
    const vals = state.vals[c.path] || {}; const args = (vals._args || []).slice();
    const id = "mc-arg-" + k;
    const inp = h("input", { id, class: "mc-input", type: "text", value: args[k] || "", spellcheck: "false" });
    inp.addEventListener("input", () => { args[k] = inp.value; const v = Object.assign({}, state.vals[c.path] || {}, { _args: args }); state.vals[c.path] = v; syncPrompt(); persist(); });
    const ctl = h("div", { class: "mc-opt-ctl" }, inp, a.choices ? h("div", { class: "mc-row" }, a.choices.map((x) => h("button", { class: "mc-chip", type: "button", "aria-pressed": args[k] === x ? "true" : "false", text: x, onclick: () => { args[k] = x; state.vals[c.path] = Object.assign({}, vals, { _args: args }); syncPrompt(); renderOpts(); } }))) : null);
    return h("div", { class: "mc-opt", dataset: { state: args[k] ? "set" : a.req ? "missing" : "" } },
      h("div", null, h("div", { class: "mc-opt-name" }, h("label", { for: id, text: String(a.name).toUpperCase() }), a.req ? h("span", { class: "mc-tag", dataset: { k: "req" }, text: "required" }) : null, h("span", { class: "mc-tag", text: "argument" }))), ctl);
  }
  function optRow(c, o, vals) {
    const v = vals[o.k];
    const id = "mc-opt-" + o.k.replace(/[^\w]/g, "");
    const vocabKey = c.path + " " + o.k, vocab = VOCAB[vocabKey];
    let ctl;
    if (o.flag) {
      ctl = h("div", { class: "mc-switch", role: "group", "aria-label": o.k },
        h("button", { type: "button", "aria-pressed": v ? "false" : "true", text: "OFF", onclick: () => setVal(c.path, o.k, false) }),
        h("button", { type: "button", "aria-pressed": v ? "true" : "false", text: "ON", onclick: () => setVal(c.path, o.k, true) }));
    } else if (o.choices) {
      ctl = h("div", { class: "mc-row", role: "radiogroup", "aria-label": o.k }, o.choices.map((x) => h("button", { class: "mc-chip", type: "button", role: "radio", "aria-checked": String(v) === x ? "true" : "false", text: x + (o.def === x ? " ·default" : ""), onclick: () => setVal(c.path, o.k, String(v) === x ? "" : x) })));
    } else {
      const inp = h("input", { id, class: "mc-input", type: "text", spellcheck: "false", value: v == null ? "" : Array.isArray(v) ? v.join(",") : String(v), placeholder: o.def != null && o.def !== "" ? "default " + o.def : o.type || "value" });
      inp.addEventListener("input", () => { const vv = state.vals[c.path] = Object.assign({}, state.vals[c.path] || {}); if (inp.value === "") delete vv[o.k]; else vv[o.k] = o.multiple ? inp.value.split(",").map((x) => x.trim()).filter(Boolean) : inp.value; syncPrompt(); persist(); });
      inp.addEventListener("change", () => renderOpts());
      const kids = [inp];
      if (vocab) {
        kids.push(h("div", { class: "mc-row" }, vocab.groups.flatMap((g) => g.items.map((it) => h("button", { class: "mc-chip", type: "button", title: it[1] || "", "aria-pressed": String(v) === it[0] ? "true" : "false", onclick: () => setVal(c.path, o.k, it[0]) }, it[0], it[1] ? h("span", { class: "mc-chip-note", text: it[1] }) : null)))));
        if (vocab.closed && v && !vocabValues(vocabKey).includes(String(v))) {
          const near = (vocab.near || {})[String(v).toLowerCase()];
          kids.push(h("div", { class: "mc-opt-note" }, "Mercury will use " + vocab.fallback + ": " + String(v) + " is not a voice domain. ",
            near ? h("button", { class: "mc-btn", type: "button", text: "Use " + near[0], title: near[1], onclick: () => setVal(c.path, o.k, near[0]) }) : null));
        }
        if (!v && vocab.unset) kids.push(h("span", { class: "mc-muted", text: vocab.unset }));
      }
      ctl = h("div", { class: "mc-opt-ctl" }, kids);
    }
    const st = v != null && v !== "" && v !== false ? "set" : o.req ? "missing" : "";
    return h("div", { class: "mc-opt", dataset: { state: st } },
      h("div", null,
        h("div", { class: "mc-opt-name" }, o.flag || o.choices ? h("span", { text: o.k }) : h("label", { for: id, text: o.k }), h("span", { class: "mc-opt-flags", text: o.flags.filter((f) => f !== o.k).join(" ") }), o.req ? h("span", { class: "mc-tag", dataset: { k: "req" }, text: "required" }) : null, o.type ? h("span", { class: "mc-tag", text: o.type }) : null, o.multiple ? h("span", { class: "mc-tag", text: "repeatable" }) : null, o.env ? h("span", { class: "mc-tag", text: "env " + o.env }) : null),
        o.help ? h("div", { class: "mc-help", text: o.help }) : null,
        vocab && vocab.note ? h("div", { class: "mc-help mc-muted", text: vocab.note }) : null),
      o.flag || o.choices ? h("div", { class: "mc-opt-ctl" }, ctl) : ctl);
  }

  /* prompt behaviour */
  const hist = { i: -1, draft: "" };
  let searchMode = null;
  function renderSuggest() {
    const box = clear($("mc-suggest"));
    if (searchMode) {
      const q = searchMode.q;
      const hit = state.history.slice().reverse().find((x) => x.includes(q));
      box.append(h("span", { class: "mc-hint", text: "reverse-i-search '" + q + "': " + (hit || "no match") + " · Enter accepts · Esc cancels" }));
      return;
    }
    const line = promptIn.value;
    if (!line.trim() || line.trim().startsWith(":")) return;
    const p = parse(line);
    if (p.errs.length && line.trim() !== "mercury-agent") box.append(h("span", { class: "mc-hint mc-warn", text: p.errs[0] }));
    else if (p.cmd) box.append(h("span", { class: "mc-hint", text: p.cmd.doc || "" }));
  }
  function fromInput() {
    const line = promptIn.value;
    if (line.trim().startsWith(":")) { renderSuggest(); return; }
    const p = parse(line);
    if (p.cmd) {
      const changed = p.path !== state.sel;
      state.sel = p.path; state.vals[p.path] = p.vals;
      promptIn.dataset.fromForm = "";
      delete promptIn.dataset.fromForm;
      if (changed) state.optsOpen = true;
      renderOpts(); persist();
    }
    renderSuggest();
  }
  promptIn.addEventListener("input", () => { if (searchMode) { searchMode.q = promptIn.value; renderSuggest(); return; } hist.i = -1; fromInput(); });
  promptIn.addEventListener("keydown", (e) => {
    if (searchMode) {
      if (e.key === "Escape") { promptIn.value = searchMode.saved; searchMode = null; renderSuggest(); e.preventDefault(); return; }
      if (e.key === "Enter" || (e.ctrlKey && e.key.toLowerCase() === "r")) { const q = searchMode.q; const hit = state.history.slice().reverse().find((x) => x.includes(q)); promptIn.value = hit || searchMode.saved; searchMode = null; fromInput(); e.preventDefault(); return; }
      return;
    }
    if (e.key === "Tab") {
      const cands = completions(promptIn.value);
      if (cands.length) {
        e.preventDefault();
        const { tokens } = tokenize(promptIn.value); const endsSpace = /\s$/.test(promptIn.value);
        const partial = endsSpace ? "" : tokens[tokens.length - 1] || "";
        const common = cands.reduce((a, b) => { let k = 0; while (k < a.length && k < b.length && a[k] === b[k]) k++; return a.slice(0, k); });
        const pick = cands.length === 1 ? cands[0] + " " : common.length > partial.length ? common : null;
        if (pick) promptIn.value = promptIn.value.slice(0, promptIn.value.length - partial.length) + pick;
        else { const box = clear($("mc-suggest")); cands.slice(0, 24).forEach((x) => box.append(h("button", { class: "mc-chip", type: "button", text: x, onclick: () => { promptIn.value = promptIn.value.slice(0, promptIn.value.length - partial.length) + x + " "; promptIn.focus(); fromInput(); } }))); return; }
        fromInput();
      }
      return;
    }
    if (e.key === "ArrowUp" || e.key === "ArrowDown") {
      if (!state.history.length) return;
      e.preventDefault();
      if (hist.i === -1) hist.draft = promptIn.value;
      hist.i = e.key === "ArrowUp" ? Math.min(state.history.length - 1, hist.i + 1) : hist.i - 1;
      promptIn.value = hist.i < 0 ? hist.draft : state.history[state.history.length - 1 - hist.i];
      if (hist.i < 0) hist.i = -1;
      fromInput(); return;
    }
    if (!e.ctrlKey) return;
    const key = e.key.toLowerCase();
    if (key === "l") { e.preventDefault(); clear(termOut); lines.length = 0; }
    else if (key === "u") { e.preventDefault(); promptIn.value = ""; fromInput(); }
    else if (key === "c" && !window.getSelection().toString()) { if (state.running) { state.running.abort(); termRow({ text: "^C cancelled", k: "warn" }); } else { termRow({ text: "$ " + promptIn.value + "^C", k: "dim" }); promptIn.value = ""; } e.preventDefault(); }
    else if (key === "r") { e.preventDefault(); searchMode = { q: "", saved: promptIn.value }; promptIn.value = ""; renderSuggest(); }
    else if (key === "f") { e.preventDefault(); const q = prompt("Find in output"); if (q) findInOutput(q); }
  });
  function findInOutput(q) {
    const hits = Array.from(termOut.querySelectorAll(".mc-line")).filter((el) => el.textContent.toLowerCase().includes(q.toLowerCase()));
    if (!hits.length) { toast("No match for '" + q + "'"); return; }
    hits[hits.length - 1].scrollIntoView({ block: "center" }); hits[hits.length - 1].focus && hits[hits.length - 1].focus();
    toast(plural(hits.length, "line") + " match '" + q + "'");
  }
  $("mc-prompt").addEventListener("submit", (e) => { e.preventDefault(); submitLine(promptIn.value); });

  function submitLine(raw) {
    const line = raw.trim();
    if (!line) return;
    if (state.history[state.history.length - 1] !== line) state.history.push(line);
    hist.i = -1; persist();
    C.recEv("cmd", line);
    if (line.startsWith(":")) { consoleCommand(line); promptIn.value = build(state.sel, state.vals[state.sel]); return; }
    const p = parse(line);
    if (!p.cmd) {
      termRow({ text: "$ " + line, k: "cmd" });
      if (/\s/.test(line) || /\?$/.test(line)) { termRow({ text: "Not a command. Asking Mercury Agent…", k: "dim" }); openPanel("chat"); sendChat(line); }
      else termRow({ text: p.errs[0] || "No such command.", k: "err" });
      termRow("");
      return;
    }
    termRow({ text: "$ " + (p.errs.length ? line : build(p.path, p.vals)), k: "cmd" });
    if (p.vals._help) { helpFor(p.cmd); return; }
    // click stops at the first usage error and reports only that one; so does the console.
    const err = p.errs[0] || validate(p.cmd, p.vals, true)[0];
    if (err) { termRow({ text: "Usage: mercury-agent " + p.path + " [OPTIONS]", k: "dim" }); termRow({ text: "Try 'mercury-agent " + p.path + " --help' for help.", k: "dim" }); termRow({ text: "Error: " + err, k: "err" }); termRow(""); return; }
    state.sel = p.path; state.vals[p.path] = p.vals; persist(); renderOpts();
    execute(p.cmd, p.vals);
  }
  function helpFor(c) {
    termRow({ text: "Usage: mercury-agent " + c.path + (c.kind === "group" ? " COMMAND" : "") + " [OPTIONS]", k: "head" });
    if (c.doc) termRow("  " + c.doc);
    c.opts.forEach((o) => termRow("  " + o.flags.join(", ").padEnd(28) + " " + (o.help || "") + (o.req ? " [required]" : "") + (o.def != null && o.def !== "" ? " [default: " + o.def + "]" : "")));
    if (c.kind === "group") childrenOf(c.path).forEach((x) => termRow("  " + x.path.split(" ").pop().padEnd(28) + " " + (x.doc || "")));
    termRow("");
  }
  function consoleCommand(line) {
    const t = line.trim().split(/\s+/), cmd = t[0].slice(1).toLowerCase();
    if (cmd === "rec" || cmd === "mark" || cmd === "capture") { if (K) K.recCommand(C, line); render(); return; }
    termRow({ text: "$ " + line, k: "cmd" });
    if (cmd === "help") helpRows();
    else if (cmd === "clear") { clear(termOut); lines.length = 0; }
    else if (cmd === "load") loadDataset();
    else if (cmd === "diag") { go("diag"); runDiag(); }
    else if (cmd === "health") probe(true);
    else { termRow({ text: "Unknown console command :" + cmd + ". :help lists them.", k: "err" }); termRow(""); }
  }

  /* ================================================================== execution */

  // CLI option -> request field, for the commands with an API route. Values convert by the field's declared type.
  const OPT_FIELD = {
    "tier-detect": { "--method": "method", "--subset": "subset", "--contamination": "contamination", "--conformal-alpha": "conformal_alpha", "--attribution": "include_attribution", "--counterfactual": "include_counterfactual", "--cf-index": "counterfactual_index", "--cf-method": "counterfactual_method" },
    rca: { "--top-k": "top_k", "--node-names": "node_names" },
    "hazard-viz": { "--detector": "hazard", "--format": "format" },
  };
  function convert(field, raw) {
    if (raw == null) return raw;
    switch (field.type) {
      case "number": return parseFloat(raw);
      case "integer": return parseInt(raw, 10);
      case "boolean": return raw === true || raw === "true";
      case "string[]": return Array.isArray(raw) ? raw : String(raw).split(",").map((x) => x.trim()).filter(Boolean);
      default: return raw;
    }
  }
  function routeFor(c, vals) {
    const fn = A.CLI_TO_ROUTE[c.path];
    return fn ? fn(vals) : null;
  }
  function buildBody(c, vals, route) {
    const schema = A.SCHEMAS[route] || {};
    const d = state.data;
    const body = {};
    const notes = [];
    if (c.path === "hazard-viz") {
      const hz = vals["--detector"];
      if (!hz) return { error: "Choose --detector (the hazard to render)." };
      if (vals["--format"] === "html") return { error: "--format html is CLI-only; the API renders png or geojson." };
      const cap = M && M.CAPS.find((x) => x.id === "haz-" + hz);
      if (!cap) return { error: "No console builder for hazard " + hz + "." };
      const why = cap.block ? cap.block(d) : null;
      if (why) return { error: hz + " " + why + "." };
      Object.assign(body, cap.build(d, {}));
      if (vals["--format"]) body.format = vals["--format"];
      return { body, notes: ["arrays built from " + (d ? d.name : "the built-in sample")] };
    }
    if (!d) return { error: "Load the data file first (LOAD FILE or :load). The browser cannot read " + (vals["--input"] || "a path") + " from disk." };
    const col = 0;
    Object.entries(schema).forEach(([k, f]) => {
      if (k === "data" || k === "observations") body[k] = f.type === "number[]" ? series(d, col) : d.matrix;
    });
    if (route.endsWith("/multivariate")) body.features = d.names.slice();
    if (route.endsWith("/fusion")) { body.return_contributions = true; }
    const map = OPT_FIELD[c.path] || {};
    Object.entries(map).forEach(([flag, field]) => { if (vals[flag] != null && vals[flag] !== "" && schema[field]) body[field] = convert(schema[field], vals[flag]); });
    if (c.path === "rca" && (vals["--train"] || vals["--adjacency"])) notes.push("--train/--adjacency are files the browser cannot read; the server learns the graph from the observations");
    if (c.path.startsWith("physics ")) notes.push("served by " + route + "; the CLI's " + c.path + " engine runs locally only");
    if (series(d, col).length !== d.rows) notes.push("first column used as the series");
    if (d.cols > 1 && schema.data && schema.data.type === "number[]") notes.push("series = column " + d.names[col]);
    return { body, notes };
  }
  async function execute(c, vals) {
    const top = c.path.split(" ")[0];
    if (c.path === "voice") {
      state.chatDomain = vals["--domain"] || null; state.chatOffline = !!vals["--offline"];
      openPanel("chat");
      termRow({ text: "Conversation open on the right · domain " + (state.chatDomain || "not set (Mercury uses general)") + (state.chatOffline ? " · offline (template replies)" : ""), k: "info" });
      if (vals["--model"]) termRow({ text: "--model selects the local Ollama model of a shell session; the server's voice uses its own configuration.", k: "dim" });
      termRow("");
      return;
    }
    const route = routeFor(c, vals);
    const writes = c.opts.some((o) => o.k === "--output" && vals["--output"]);
    if (LONG_RUNNING.has(top) || !route || writes) {
      const why = LONG_RUNNING.has(top) ? "long-running or training command" : writes ? "writes a file (-o/--output)" : "has no API route";
      const cmdline = build(c.path, vals);
      termRow({ text: "Prepared · " + why + ", so it runs in your shell:", k: "warn" });
      termRow({ text: "  " + cmdline, k: "info", act: () => copy(cmdline) });
      termRow({ text: "  (click the line to copy)", k: "dim" });
      termRow("");
      return;
    }
    if (!routeKeys.has(route)) { termRow({ text: "This server does not serve " + route + ".", k: "err" }); termRow(""); return; }
    const built = buildBody(c, vals, route);
    if (built.error) { termRow({ text: built.error, k: "err" }); termRow(""); return; }
    (built.notes || []).forEach((n) => termRow({ text: "note · " + n, k: "dim" }));
    await callRoute(route, built.body, { label: c.path + (state.data ? " · " + state.data.name : ""), kind: "CLI" });
  }
  async function callRoute(route, body, meta) {
    const [method, path] = route.split(" ");
    const ctrl = new AbortController();
    state.running = ctrl; setRun("running", method + " " + path);
    termRow({ text: "→ " + method + " " + path, k: "dim" });
    const r = method === "GET" ? await A.get(path, { signal: ctrl.signal }) : await A.post(path, body, { signal: ctrl.signal });
    state.running = null;
    const rec = pushResult({ kind: (meta && meta.kind) || "API", label: (meta && meta.label) || route, route, status: r.ok ? "ok" : "error", http: r.status, ms: r.meta ? r.meta.ms : null, corr: r.meta ? r.meta.correlationId : null, body, data: r.ok ? r.data : r.data || null, error: r.ok ? null : r.error, ds: dsMeta(state.data) });
    if (r.ok) {
      setRun("ok", "HTTP " + r.status + " · " + (r.meta ? r.meta.ms + " ms" : ""));
      const sum = rec.sum || {};
      termRow({ text: "← HTTP " + r.status + " · " + (r.meta ? r.meta.ms : "?") + " ms" + (r.meta && r.meta.correlationId ? " · " + r.meta.correlationId : ""), k: "ok" });
      if (sum.headline) termRow({ text: "  " + sum.headline, k: "head" });
      (sum.kv || []).slice(0, 6).forEach(([k, v]) => termRow("  " + String(k).padEnd(18) + " " + v));
      termRow({ text: "  open in Results →", k: "info", act: () => { state.resultSel = rec.id; go("results"); } });
    } else {
      setRun("error", r.kind === "ratelimit" ? "rate limited" : "HTTP " + (r.status || r.kind));
      termRow({ text: "← " + (r.status ? "HTTP " + r.status : r.kind) + " · " + r.error, k: "err" });
      if (r.kind === "auth") termRow({ text: "  This route needs an API key: open the Engine pill and set X-API-Key.", k: "warn" });
      if (r.kind === "ratelimit") { const b = A.budget(); termRow({ text: "  Rate limited; retry " + (b.retryUntil ? "in " + Math.max(1, Math.round((b.retryUntil - Date.now()) / 1000)) + " s" : "later") + ".", k: "warn" }); }
    }
    termRow("");
    return rec;
  }

  /* ================================================================== engine status */

  function setRun(st, text) { const p = $("mc-run-pill"); p.dataset.state = st; $("mc-run-text").textContent = st === "running" ? "Running · " + text : st === "ok" ? "Done · " + text : st === "error" ? "Error · " + text : "Idle"; }
  async function probe(verbose) {
    const r = await A.health();
    state.health = { ok: r.ok, status: r.status, data: r.data, at: Date.now(), error: r.error };
    const pill = $("mc-conn");
    pill.dataset.state = r.ok ? "live" : "down";
    $("mc-conn-text").textContent = r.ok ? "Engine · " + ((r.data && r.data.status) || "live") : "Engine · unreachable";
    if (verbose) { termRow({ text: r.ok ? "engine " + location.origin + " · HTTP " + r.status + " · " + JSON.stringify(r.data).slice(0, 200) : "engine unreachable · " + r.error, k: r.ok ? "ok" : "err" }); termRow(""); }
    const v = await A.voiceStatus();
    state.voice = v.ok ? v.data : null;
    renderMode();
    if (state.view === "engine") render();
    return r;
  }

  /* ================================================================== agent conversation */

  const chatLog = $("mc-chat-log");
  function renderMode() {
    const sanitized = (VOCAB["voice --domain"] && state.chatDomain && !vocabValues("voice --domain").includes(state.chatDomain));
    const d = state.chatDomain || state.domainForChat || null;
    $("mc-mode").textContent = (state.chatOffline ? "offline · voice --offline · template replies" : (state.health && state.health.ok ? "live" : "engine unreachable") + " · /api/v1/voice") + " · domain " + (d || "not set (general)") + (sanitized ? " → general (not a voice domain)" : "");
  }
  function chatMsg(m) {
    const el = h("div", { class: "mc-msg", dataset: { role: m.role } });
    if (m.steps && m.steps.length) el.append(h("ul", { class: "mc-steps" }, m.steps.map((s) => h("li", { text: s }))));
    el.append(document.createTextNode(m.text || ""));
    if (m.meta) el.append(h("span", { class: "mc-msg-meta", text: m.meta }));
    if (m.follow && m.follow.length) el.append(h("div", { class: "mc-row" }, m.follow.map((f) => h("button", { class: "mc-chip", type: "button", text: f, onclick: () => sendChat(f) }))));
    chatLog.append(el);
    chatLog.parentElement.scrollTop = chatLog.parentElement.scrollHeight;
  }
  async function sendChat(text) {
    const msg = String(text || "").trim(); if (!msg) return;
    state.chat.push({ role: "user", text: msg }); chatMsg({ role: "user", text: msg });
    C.recEv("cmd", "agent: " + msg);
    if (state.chatOffline) { const m = { role: "agent", text: "Offline mode answers from templates in a local shell (mercury-agent voice --offline). Remove --offline to talk to this server's voice." }; state.chat.push(m); chatMsg(m); return; }
    const r = await A.speak(msg, state.chatDomain || null, state.sessionId);
    if (!r.ok) { const m = { role: "error", text: (r.status ? "HTTP " + r.status + " · " : "") + r.error }; state.chat.push(m); chatMsg(m); return; }
    const d = r.data || {};
    const meta = ["intent " + (d.intent || "unknown"), typeof d.confidence === "number" ? "confidence " + d.confidence.toFixed(2) : null, d.search_performed ? "searched " + (d.search_results_count || 0) : null, Array.isArray(d.sources) && d.sources.length ? plural(d.sources.length, "source") : null, r.meta && r.meta.ms != null ? r.meta.ms + " ms" : null].filter(Boolean).join(" · ");
    const m = { role: "agent", text: (d.message || "(no reply)") + (d.uncertainty_note ? "\n\n" + d.uncertainty_note : ""), steps: d.reasoning_summary ? [d.reasoning_summary] : null, meta, follow: Array.isArray(d.suggested_follow_ups) ? d.suggested_follow_ups.slice(0, 4) : null };
    state.chat.push(m); chatMsg(m);
    C.recEv("out", "agent: " + (d.message || ""));
  }
  $("mc-composer").addEventListener("submit", (e) => { e.preventDefault(); const inp = $("mcli-chat-input"); const t = inp.value; inp.value = ""; openPanel("chat"); sendChat(t); });

  /* ================================================================== panel */

  const body = $("mc-body");
  function openPanel(tab) {
    state.panel = true; state.panelTab = tab || state.panelTab;
    renderPanel();
  }
  function renderPanel() {
    body.dataset.panel = state.panel ? "open" : "closed";
    $("mc-chat-toggle").setAttribute("aria-pressed", state.panel ? "true" : "false");
    const chat = state.panelTab === "chat";
    $("mc-tab-chat").setAttribute("aria-selected", chat ? "true" : "false");
    $("mc-tab-ref").setAttribute("aria-selected", chat ? "false" : "true");
    $("mc-panel-chat").hidden = !chat; $("mc-panel-ref").hidden = chat; $("mc-composer").hidden = !chat;
    if (!chat) renderRef();
    renderMode();
  }
  $("mc-chat-toggle").addEventListener("click", () => { state.panel = !state.panel; renderPanel(); if (state.panel) $("mcli-chat-input").focus(); });
  $("mc-panel-close").addEventListener("click", () => { state.panel = false; renderPanel(); });
  $("mc-tab-chat").addEventListener("click", () => openPanel("chat"));
  $("mc-tab-ref").addEventListener("click", () => openPanel("ref"));
  function showRef(kind, item) { state.ref = { kind, item }; openPanel("ref"); }
  function kv(pairs) { return h("dl", { class: "mc-kv" }, pairs.filter((p) => p && p[1] != null && p[1] !== "").flatMap(([k, v]) => [h("dt", { text: k }), h("dd", null, v instanceof Node ? v : String(v))])); }
  function codeBlock(text) {
    const pre = h("pre", { class: "mc-pre", text });
    return h("div", null, pre, h("button", { class: "mc-btn", type: "button", text: "Copy", onclick: () => copy(text) }));
  }
  function renderRef() {
    const box = clear($("mc-panel-ref"));
    const r = state.ref;
    if (!r) { box.append(h("p", { class: "mc-muted", text: "Select a catalog item to see how to run it." })); return; }
    const it = r.item;
    if (r.kind === "detector") {
      box.append(h("h2", { class: "mc-h1", text: it.name }), h("p", { class: "mc-help", text: it.desc }),
        kv([["class", it.c], ["module", it.mod], ["family", it.fam], ["modality", it.modality], ["needs fit", it.fit ? "yes" : "no"], ["seams", (it.seams || []).join(", ")], ["wraps", it.wraps], ["covered by", it.coveredBy]]));
      if (it.ont) {
        box.append(h("h3", { class: "mc-label mc-section", text: "Ontology" }), kv([["detects", it.ont.detects.join(", ") || "—"], ["cannot detect", it.ont.cannot.join(", ") || "—"]]));
        if (it.ont.validated.length) box.append(h("ul", { class: "mc-steps" }, it.ont.validated.map((v) => h("li", { text: (v.d || "") + " · " + (v.n || "") + (v.e ? " · " + v.e : "") }))));
        else box.append(h("p", { class: "mc-help mc-warn", text: it.ont.note || "No validation evidence recorded." }));
      }
      // The manifest's own module and class (core/detector_registry.py DETECTOR_MANIFEST).
      const py = "from " + it.mod + " import " + it.c + "\n\ndetector = " + it.c + "()  # constructor options: help(" + it.c + ")\n" + (it.fit ? "detector.fit(train)  # this detector needs a fit before scoring\n" : "");
      const tier = CAT.tier || { paradigms: {}, torch: [] };
      const inTier = Object.values(tier.paradigms).some((names) => names.includes(it.name)) || tier.torch.includes(it.name);
      box.append(h("h3", { class: "mc-label mc-section", text: "Run it" }), codeBlock(inTier ? "mercury-agent tier-detect -i data.csv --subset " + it.name + (tier.torch.includes(it.name) ? "  # needs torch" : "") : py));
    } else if (r.kind === "command") {
      box.append(h("h2", { class: "mc-h1", text: it.path }), h("p", { class: "mc-help", text: it.doc }), kv([["section", it.section], ["options", it.opts.length], ["API route", routeFor(it, {}) || "none (shell)"]]),
        h("button", { class: "mc-btn mc-btn-primary", type: "button", text: "Open in terminal", onclick: () => selectCommand(it.path) }));
    } else if (r.kind === "mcp") {
      const schema = (CAT.mcpSchemas || {})[it.n] || {};
      const args = A.body(schema, {}, { requiredOnly: true });
      box.append(h("h2", { class: "mc-h1", text: it.n }), h("p", { class: "mc-help", text: it.d }), fieldTable(schema), h("h3", { class: "mc-label mc-section", text: "JSON-RPC over stdio (mercury-agent mcp)" }), codeBlock(A.mcpCall(it.n, args)));
    } else if (r.kind === "api") {
      const key = it.m + " " + it.path, schema = A.withParams(it.path ? key : key, A.SCHEMAS[key] || {});
      const b = it.m === "GET" ? null : A.body(schema, {}, { requiredOnly: false });
      box.append(h("h2", { class: "mc-h1", text: key }), h("p", { class: "mc-help", text: it.s || "" }), fieldTable(schema), h("h3", { class: "mc-label mc-section", text: "curl" }), codeBlock(A.curl(key, b, {}, schema)),
        it.m === "GET" && !A.pathParams(key).length ? h("button", { class: "mc-btn mc-btn-primary", type: "button", text: "Call", onclick: () => callRoute(key, null, { label: key, kind: "API" }) }) : null,
        it.m !== "GET" && A.SCHEMAS[key] ? h("button", { class: "mc-btn mc-btn-primary", type: "button", text: "Open request form", onclick: () => { state.apiForm = key; go("api"); } }) : null);
    } else if (r.kind === "run") {
      const flags = (it.a || []).map((a) => a.f[0] + (a.flag ? "" : " <" + (a.type || "value") + ">") + (a.def != null ? "  # default " + a.def : "")).join("\n");
      box.append(h("h2", { class: "mc-h1", text: it.p }), h("p", { class: "mc-help", text: it.s }), kv([["category", it.cat], ["group", it.g]]), codeBlock("python " + it.p + (flags ? "\n# flags:\n" + flags.split("\n").map((x) => "#   " + x).join("\n") : "")));
    } else if (r.kind === "library") {
      box.append(h("h2", { class: "mc-h1", text: it.c }), h("p", { class: "mc-help", text: it.what }), kv([["module", it.mod], ["call", it.call], ["family", it.fam], ["reached through", it.via || "Python only"], ["needs torch", it.torch ? "yes" : "no"]]), codeBlock((it.code || []).join("\n")));
    } else if (r.kind === "loader") {
      box.append(h("h2", { class: "mc-h1", text: it.id }), kv([["class", it.cls], ["file", it.file]]), codeBlock("from omni_mercury_engine.loaders import " + it.cls + "\n\nloader = " + it.cls + "()\n"),
        h("button", { class: "mc-btn", type: "button", text: "Use as CONTEXT domain", onclick: () => { state.domain = it.id; persist(); toast("CONTEXT domain · " + it.id); } }));
    } else if (r.kind === "tool") {
      box.append(h("h2", { class: "mc-h1", text: it.name }), kv([["group", it.group]]), codeBlock("mercury-agent tool " + it.name + " --help"), h("button", { class: "mc-btn mc-btn-primary", type: "button", text: "Open in terminal", onclick: () => { state.vals.tool = { _args: [it.name] }; selectCommand("tool"); } }));
    }
  }
  function fieldTable(schema) {
    const keys = Object.keys(schema || {});
    if (!keys.length) return h("p", { class: "mc-muted", text: "No request fields." });
    return h("table", { class: "mc-table" }, h("thead", null, h("tr", null, ["field", "type", "notes"].map((x) => h("th", { text: x })))),
      h("tbody", null, keys.map((k) => { const f = schema[k]; return h("tr", null, h("td", { text: k + (f.req ? " *" : "") }), h("td", { text: f.type + (f.enum ? " · " + f.enum.join("|") : "") }), h("td", { text: [f.desc, f.def !== undefined ? "default " + JSON.stringify(f.def) : null, f.min != null ? "min " + f.min : null, f.max != null ? "max " + f.max : null, f.p ? "path" : f.q ? "query" : null].filter(Boolean).join(" · ") })); })));
  }

  /* ================================================================== navigation */

  const NAV = [
    ["WORK", [["terminal", "$", "Terminal"], ["agent", "@", "Agent"], ["mass", "Σ", "Mass Run"], ["results", "◧", "Results"], ["captures", "⊡", "Captures"], ["record", "●", "Record session"]]],
    ["CATALOG", [["commands", "›", "Commands"], ["runs", "▸", "Runs"], ["detectors", "◎", "Detectors"], ["domains", "◇", "Domains"], ["physics", "φ", "Physics"], ["tools", "⌗", "Tools"]]],
    ["INTERFACES", [["mcp", "⇄", "MCP tools"], ["api", "{}", "API routes"]]],
    ["SESSION", [["session", "⋯", "Session"], ["diag", "√", "Diagnostics"], ["engine", "●", "Engine"]]],
  ];
  function navCount(id) {
    switch (id) {
      case "results": return state.results.length || "";
      case "captures": return (state.caps.length + state.recs.length) || "";
      case "record": return state.rec ? "REC" : "";
      case "commands": return CAT.nCommands;
      case "runs": return CAT.runs.length;
      case "detectors": return CAT.detectors.length + CAT.library.length;
      case "domains": return CAT.loaders.length + CAT.hazards.length;
      case "tools": return CAT.nTools;
      case "mcp": return CAT.mcp.length;
      case "api": return CAT.api.length;
      default: return "";
    }
  }
  function renderNav() {
    const nav = clear($("mc-nav"));
    NAV.forEach(([group, items]) => {
      const g = h("div", { class: "mc-navgroup", role: "group", "aria-label": group },
        h("div", { class: "mc-navhead", text: group + (group === "CATALOG" && state.catSel.length ? " · " + state.catSel.length + " selected" : "") }));
      items.forEach(([id, glyph, label]) => {
        const n = navCount(id);
        const btn = h("button", { class: "mc-navitem", type: "button", "aria-current": state.view === id ? "page" : null },
          h("span", { class: "mc-navglyph", "aria-hidden": "true", text: glyph }), h("span", { text: label }), h("span", { class: "mc-navcount", dataset: id === "record" && state.rec ? { rec: "1" } : null, text: String(n) }));
        btn.addEventListener("click", () => { if (id === "record") { if (K) K.recCommand(C, state.rec ? ":rec stop" : ":rec start"); render(); return; } go(id); });
        g.append(btn);
      });
      nav.append(g);
    });
  }
  function go(view) {
    state.view = view;
    render();
    const target = view === "terminal" ? promptIn : $("mc-main");
    if (target && target.focus) target.focus({ preventScroll: true });
  }

  /* ================================================================== views */

  const viewEl = $("mc-view");
  const termEl = $("mc-view-term");
  const scrolls = {};
  function render() {
    renderNav();
    const term = state.view === "terminal";
    termEl.hidden = !term; viewEl.hidden = term;
    if (!term) {
      const prev = viewEl.dataset.view;
      if (prev) scrolls[prev] = viewEl.scrollTop;
      clear(viewEl); viewEl.dataset.view = state.view;
      const fn = VIEWS[state.view] || VIEWS.terminal;
      try { fn(viewEl); } catch (e) { viewEl.append(h("p", { class: "mc-bad", text: "This view failed to render: " + e.message })); }
      viewEl.scrollTop = scrolls[state.view] || 0;
    }
    if (state.panel && state.panelTab === "ref") renderRef();
  }
  function heading(el, title, lede) { el.append(h("h1", { class: "mc-h1", text: title })); if (lede) el.append(h("p", { class: "mc-lede", text: lede })); }
  function filterBox(el, key, placeholder) {
    const id = "mc-filter-" + key;
    const inp = h("input", { id, class: "mc-input", type: "search", value: state.filters[key] || "", placeholder: placeholder || "Filter" });
    inp.addEventListener("input", () => { state.filters[key] = inp.value; const pos = inp.selectionStart; render(); const again = $(id); if (again) { again.focus(); again.setSelectionRange(pos, pos); } });
    el.append(h("div", { class: "mc-filter" }, h("label", { for: id, class: "mc-label", text: "Filter" }), inp));
    return (state.filters[key] || "").toLowerCase();
  }
  const match = (q, ...fields) => !q || fields.some((f) => String(f || "").toLowerCase().includes(q));
  function groupedList(items, groupOf, renderItem) {
    const ul = h("ul", { class: "mc-list" });
    let last = null;
    items.forEach((it) => { const g = groupOf(it); if (g !== last) { ul.append(h("li", { class: "mc-listgroup", text: g })); last = g; } ul.append(h("li", null, renderItem(it))); });
    if (!items.length) ul.append(h("li", { class: "mc-empty", text: "Nothing matches the filter." }));
    return ul;
  }
  function itemButton(title, sub, opts) {
    const o = opts || {};
    const b = h("button", { class: "mc-item", type: "button", "aria-current": o.current ? "true" : null },
      o.check != null ? h("input", { type: "checkbox", class: "mc-check", checked: o.check ? "" : null, "aria-label": "Select " + title, onclick: (e) => { e.stopPropagation(); o.onCheck && o.onCheck(e.target.checked); } }) : h("span", { class: "mc-item-dot", dataset: o.dot ? { k: o.dot } : null }),
      h("span", { class: "mc-item-t", text: title }), h("span", { class: "mc-muted", text: o.right || "" }), sub ? h("span", { class: "mc-item-s", text: sub }) : null);
    if (o.onClick) b.addEventListener("click", o.onClick);
    return b;
  }
  const catKey = (kind, id) => kind + ":" + id;
  function toggleCat(key, on) { const s = new Set(state.catSel); if (on) s.add(key); else s.delete(key); state.catSel = Array.from(s); persist(); renderNav(); }

  const VIEWS = {
    agent(el) {
      heading(el, "Agent", "Mercury Agent's capabilities and the Python that runs each one. The conversation is on the right (POST /api/v1/voice/speak).");
      el.append(h("div", { class: "mc-row" }, h("button", { class: "mc-btn mc-btn-primary", type: "button", text: "Open conversation", onclick: () => { openPanel("chat"); $("mcli-chat-input").focus(); } })));
      if (!G) return;
      const groups = new Map();
      (G.CAPS || []).forEach((c) => { const g = c.g || c.group || "Capabilities"; if (!groups.has(g)) groups.set(g, []); groups.get(g).push(c); });
      groups.forEach((caps, g) => {
        el.append(h("h2", { class: "mc-label mc-section", text: g }));
        el.append(h("div", { class: "mc-cards" }, caps.map((c) => {
          let py = ""; try { py = G.python ? G.python(c, {}) : ""; } catch (e) { py = ""; }
          return h("article", { class: "mc-card" }, h("h3", { text: c.t || c.title || c.id }), h("p", { class: "mc-help", text: c.what || "" }),
            kv([["module", c.m], ["classes", c.cls], ["reach", c.reach], ["environment", c.env]]),
            (c.st || []).length ? h("ul", { class: "mc-steps" }, c.st.map((s) => h("li", { text: s[0] + " · " + s[1] }))) : null,
            py ? codeBlock(py) : null);
        })));
      });
    },
    mass(el) {
      heading(el, "Mass Run", "Run many detection routes on one dataset and compare them. Uses " + (state.data ? state.data.name + " (" + state.data.rows + " rows)" : "the built-in 16-point sample; load a file for your own data") + ".");
      const caps = M ? M.CAPS : [];
      const sel = new Set(state.mass.sel);
      el.append(h("div", { class: "mc-row" },
        h("button", { class: "mc-btn", type: "button", text: state.data ? "File · " + state.data.name : "Load file", onclick: async () => { await loadDataset(); render(); } }),
        h("button", { class: "mc-btn", type: "button", text: "Recommended", onclick: () => { state.mass.sel = caps.filter((c) => c.rec).map((c) => c.id); persist(); render(); } }),
        h("button", { class: "mc-btn", type: "button", text: "None", onclick: () => { state.mass.sel = []; persist(); render(); } }),
        h("span", { class: "mc-spacer" }),
        h("button", { class: "mc-btn", type: "button", text: "Script (.sh)", disabled: sel.size ? null : "", onclick: () => download("mercury-mass-run.sh", "text/x-shellscript", M.script(A, caps.filter((c) => sel.has(c.id)), state.data, { domain: state.domain })) }),
        state.mass.running ? h("button", { class: "mc-btn mc-btn-danger", type: "button", text: "Cancel", onclick: () => { state.mass.abort && state.mass.abort.abort(); } })
          : h("button", { id: "mc-mass-run", class: "mc-btn mc-btn-primary", type: "button", text: "Run " + sel.size, disabled: sel.size ? null : "", onclick: runMass })));
      const outBy = new Map(((state.mass.out || [])).map((o) => [o.id, o]));
      const groups = [...new Set(caps.map((c) => c.group))];
      el.append(h("div", { class: "mc-split" },
        groupedList(caps, (c) => c.group, (c) => {
          const o = outBy.get(c.id); const why = c.block ? c.block(state.data) : null;
          return itemButton(c.label, o ? (o.status + (o.http ? " · HTTP " + o.http : "") + (o.ms != null ? " · " + o.ms + " ms" : "") + (o.error ? " · " + o.error : "") + (o.sum && o.sum.headline ? " · " + o.sum.headline : "")) : why || c.route, { check: sel.has(c.id), onCheck: (on) => { if (on) sel.add(c.id); else sel.delete(c.id); state.mass.sel = Array.from(sel); persist(); render(); }, right: o ? o.status : "" });
        }),
        h("div", null, massSummary())));
      void groups;
    },
    results(el) {
      heading(el, "Results", "Every response this session received, newest first. Charts are chosen for the data's shape.");
      if (!state.results.length) { el.append(h("div", { class: "mc-empty", text: "No results yet. Run a command in the Terminal, or call a route from API routes." })); return; }
      const cur = state.results.find((r) => r.id === state.resultSel) || state.results[0];
      el.append(h("div", { class: "mc-split" },
        groupedList(state.results, () => "Responses", (r) => itemButton(r.label, (r.route || "") + (r.http ? " · HTTP " + r.http : "") + " · " + String(r.at).slice(11, 19) + " UTC", { current: r === cur, dot: r.status === "ok" ? "ok" : "error", right: r.sum && r.sum.headline ? r.sum.headline : "", onClick: () => { state.resultSel = r.id; render(); } })),
        resultDetail(cur)));
    },
    captures(el) {
      heading(el, "Captures", "Sections of results saved with provenance and a SHA-256 of their data, and recorded sessions. Labels files feed tier-detect/train --labels.");
      el.append(h("div", { class: "mc-row", role: "tablist" },
        h("button", { class: "mc-tab", type: "button", role: "tab", "aria-selected": state.capView === "caps" ? "true" : "false", text: "Captures · " + state.caps.length, onclick: () => { state.capView = "caps"; render(); } }),
        h("button", { class: "mc-tab", type: "button", role: "tab", "aria-selected": state.capView === "recs" ? "true" : "false", text: "Recordings · " + state.recs.length, onclick: () => { state.capView = "recs"; render(); } }),
        h("span", { class: "mc-spacer" }),
        h("button", { class: "mc-btn", type: "button", text: "Evidence pack (.zip)", disabled: state.caps.length || state.recs.length ? null : "", onclick: () => { const r = K.pack(C); if (r) toast("Evidence pack · " + r.files.length + " files · " + r.bytes + " bytes"); } })));
      if (state.capView === "caps") {
        if (!state.caps.length) { el.append(h("div", { class: "mc-empty", text: "No captures yet. Open a result, select samples on the strip and press CAPTURE, or :capture <from> <to> in the Terminal." })); return; }
        const cur = state.caps.find((c) => c.id === state.capSel) || state.caps[0];
        el.append(h("div", { class: "mc-split" }, groupedList(state.caps, () => "Captures", (c) => itemButton(c.label, (c.tag || "").toUpperCase() + " · samples " + c.from + "–" + c.to + " · " + c.n + " rows" + (c.flags ? " · " + c.flagged + " flagged" : ""), { current: c === cur, dot: c.tag, onClick: () => { state.capSel = c.id; render(); } })), capDetail(cur)));
      } else {
        if (!state.recs.length) { el.append(h("div", { class: "mc-empty", text: "No recordings. Start one with Record session in the nav or :rec start." })); return; }
        const cur = state.recs.find((r) => r.id === state.recSel) || state.recs[0];
        el.append(h("div", { class: "mc-split" }, groupedList(state.recs, () => "Recordings", (r) => itemButton(r.title, K.clock(r.dur) + " · " + plural((r.events || []).length, "event"), { current: r === cur, dot: "info", onClick: () => { state.recSel = r.id; render(); } })), recDetail(cur)));
      }
    },
    commands(el) {
      heading(el, "Commands", CAT.nCommands + " mercury-agent commands, read from the click group in cli.py.");
      const q = filterBox(el, "commands", "Filter commands");
      const items = commands.filter((c) => match(q, c.path, c.doc, c.section)).sort((a, b) => CAT.sections.indexOf(a.section) - CAT.sections.indexOf(b.section) || a.path.localeCompare(b.path));
      el.append(groupedList(items, (c) => c.section || "Other", (c) => itemButton(c.path, c.doc, { check: state.catSel.includes(catKey("cmd", c.path)), onCheck: (on) => toggleCat(catKey("cmd", c.path), on), right: routeFor(c, {}) ? "API" : c.kind === "group" ? "group" : "", onClick: () => { showRef("command", c); } })));
    },
    runs(el) {
      heading(el, "Runs", CAT.runs.length + " benchmarks, scripts and examples. Parsed, never executed: run them from a shell.");
      const q = filterBox(el, "runs", "Filter runs");
      const items = CAT.runs.filter((r) => match(q, r.p, r.s, r.cat)).sort((a, b) => a.cat.localeCompare(b.cat) || a.p.localeCompare(b.p));
      el.append(groupedList(items, (r) => r.cat, (r) => itemButton(r.p, r.s, { check: state.catSel.includes(catKey("run", r.p)), onCheck: (on) => toggleCat(catKey("run", r.p), on), right: r.a ? plural(r.a.length, "flag") : "", onClick: () => showRef("run", r) })));
      void 0;
    },
    detectors(el) {
      heading(el, "Detectors", CAT.detectors.length + " registry detectors (core/detector_registry.py) with their ontology, plus " + CAT.library.length + " library classes reachable from Python.");
      const q = filterBox(el, "detectors", "Filter by name, family, modality or anomaly class");
      const items = CAT.detectors.filter((d) => match(q, d.name, d.c, d.fam, d.modality, d.desc, d.ont && d.ont.detects.join(" "))).sort((a, b) => a.fam.localeCompare(b.fam) || a.name.localeCompare(b.name));
      el.append(groupedList(items, (d) => d.fam, (d) => itemButton(d.name, d.desc + " · " + d.modality, { check: state.catSel.includes(catKey("det", d.name)), onCheck: (on) => toggleCat(catKey("det", d.name), on), right: d.ont && d.ont.validated.length ? "validated" : "", onClick: () => showRef("detector", d) })));
      const lib = CAT.library.filter((l) => match(q, l.c, l.what, l.fam));
      if (lib.length) { el.append(h("h2", { class: "mc-label mc-section", text: "Library classes" })); el.append(groupedList(lib, (l) => l.fam, (l) => itemButton(l.c, l.call, { right: l.torch ? "torch" : "", onClick: () => showRef("library", l) }))); }
      const preds = CAT.predictors || [];
      if (preds.length && !q) { el.append(h("h2", { class: "mc-label mc-section", text: "Predictors" })); el.append(groupedList(preds, () => "Conformal & probabilistic", (p) => itemButton(p.c, p.mod + (p.cli ? " · " + p.cli : ""), {}))); }
    },
    domains(el) {
      heading(el, "Domains", CAT.loaders.length + " data loaders and " + CAT.hazards.length + " hazard visualisers. CONTEXT domain: " + (state.domain || "not set") + ".");
      const q = filterBox(el, "domains", "Filter domains");
      el.append(groupedList(CAT.loaders.filter((l) => match(q, l.id, l.cls)), () => "Loaders", (l) => itemButton(l.id, l.cls + " · " + l.file, { current: state.domain === l.id, check: state.catSel.includes(catKey("dom", l.id)), onCheck: (on) => toggleCat(catKey("dom", l.id), on), onClick: () => showRef("loader", l) })));
      el.append(h("h2", { class: "mc-label mc-section", text: "Hazards · POST /api/v1/hazard/visualize" }));
      el.append(h("div", { class: "mc-row" }, CAT.hazards.filter((x) => match(q, x)).map((x) => h("button", { class: "mc-chip", type: "button", text: x, onclick: () => { state.vals["hazard-viz"] = { "--detector": x, "--output": "out.png" }; selectCommand("hazard-viz"); } }))));
    },
    physics(el) {
      heading(el, "Physics", "The physics command group. Spectral, dynamics and UI/UX analyses run locally in a shell; the console approximates them with the matching API route and says so on every run.");
      el.append(groupedList(commands.filter((c) => c.path.startsWith("physics")), () => "physics", (c) => itemButton(c.path, c.doc, { right: routeFor(c, {}) || "", onClick: () => selectCommand(c.path) })));
    },
    tools(el) {
      heading(el, "Tools", CAT.nTools + " operator tools (tools/__init__.py), run with mercury-agent tool <name>.");
      const q = filterBox(el, "tools", "Filter tools");
      const items = CAT.tools.flatMap(([g, names]) => names.map((n) => ({ name: n, group: g }))).filter((t) => match(q, t.name, t.group));
      el.append(groupedList(items, (t) => t.group, (t) => itemButton(t.name, "", { check: state.catSel.includes(catKey("tool", t.name)), onCheck: (on) => toggleCat(catKey("tool", t.name), on), onClick: () => showRef("tool", t) })));
    },
    mcp(el) {
      heading(el, "MCP tools", CAT.mcp.length + " tools served by mercury-agent mcp (JSON-RPC over stdio). Select some to export a script.");
      const q = filterBox(el, "mcp", "Filter tools");
      const items = CAT.mcp.filter((t) => match(q, t.n, t.d));
      const picked = items.filter((t) => state.catSel.includes(catKey("mcp", t.n)));
      el.append(h("div", { class: "mc-row" }, h("button", { class: "mc-btn", type: "button", text: "Script for " + plural(picked.length, "tool") + " (.sh)", disabled: picked.length ? null : "", onclick: () => download("mercury-mcp.sh", "text/x-shellscript", M.mcpScript(A, picked.map((t) => t.n), (n) => A.body((CAT.mcpSchemas || {})[n] || {}, {}, { requiredOnly: true }))) })));
      el.append(groupedList(items, () => "Tools", (t) => itemButton(t.n, t.d, { check: state.catSel.includes(catKey("mcp", t.n)), onCheck: (on) => { toggleCat(catKey("mcp", t.n), on); render(); }, onClick: () => showRef("mcp", t) })));
    },
    api(el) {
      heading(el, "API routes", CAT.api.length + " operations from this server's OpenAPI document. Account and model-registry routes are hidden unless shown.");
      const q = filterBox(el, "api", "Filter routes");
      const showAll = !!state.apiAll;
      el.append(h("div", { class: "mc-row" }, h("label", { class: "mc-row" }, h("input", { type: "checkbox", checked: showAll ? "" : null, onchange: (e) => { state.apiAll = e.target.checked; render(); } }), h("span", { class: "mc-help", text: "Show account and model-registry routes" }))));
      if (state.apiForm) el.append(apiForm(state.apiForm));
      const items = CAT.api.filter((r) => (showAll || !/^(Accounts|Models)$/.test(r.g)) && match(q, r.path, r.m, r.s, r.g));
      el.append(groupedList(items, (r) => r.g, (r) => itemButton(r.m + " " + r.path, r.s || "", { onClick: () => showRef("api", r) })));
    },
    session(el) {
      heading(el, "Session", "What this browser keeps for you, and exports.");
      const st = { history: state.history.length, results: state.results.length, captures: state.caps.length, recordings: state.recs.length };
      el.append(kv([["origin", location.origin], ["history", st.history + " commands"], ["results", st.results + " (last 30 kept)"], ["captures", st.captures + " (up to 200)"], ["recordings", st.recordings + " (up to 10)"], ["CONTEXT domain", state.domain || "not set"], ["session id", state.sessionId]]));
      el.append(h("div", { class: "mc-row" },
        h("button", { class: "mc-btn", type: "button", text: "Export session (.json)", onclick: () => download("mercury-console-session.json", "application/json", JSON.stringify({ exported_at: nowIso(), origin: location.origin, history: state.history, results: state.results, captures: state.caps, recordings: state.recs }, null, 2)) }),
        h("button", { class: "mc-btn", type: "button", text: "Import MCP results (.jsonl)", onclick: importMcp }),
        h("button", { class: "mc-btn mc-btn-danger", type: "button", text: "Clear history", onclick: () => { state.history = []; persist(); render(); } }),
        h("button", { class: "mc-btn mc-btn-danger", type: "button", text: "Clear results", onclick: () => { state.results = []; persist(); render(); } })));
    },
    diag(el) {
      heading(el, "Diagnostics", "Checks that the console's catalog, client and charts agree with this server. :diag runs them from the Terminal.");
      el.append(h("div", { class: "mc-row" }, h("button", { class: "mc-btn mc-btn-primary", type: "button", text: "Run self-test", onclick: runDiag })));
      const d = state.diag;
      if (!d) return;
      el.append(h("p", { class: d.failed ? "mc-bad" : "mc-good", text: d.passed + " passed · " + d.failed + " failed · " + d.ms + " ms" }));
      el.append(h("table", { class: "mc-table" }, h("tbody", null, d.checks.map((c) => h("tr", null, h("td", { class: c.ok ? "mc-good" : "mc-bad", text: c.ok ? "PASS" : "FAIL" }), h("td", { text: c.name }), h("td", { class: "mc-muted", text: c.detail || "" }))))));
    },
    engine(el) {
      heading(el, "Engine", "The server this page came from.");
      const hs = state.health;
      el.append(kv([["origin", location.origin], ["health", hs ? (hs.ok ? "HTTP " + hs.status : "unreachable · " + (hs.error || "")) : "not probed"], ["checked", hs ? new Date(hs.at).toISOString().slice(11, 19) + " UTC" : "—"], ["catalog", "catalog.js · version " + CAT.version]]));
      if (hs && hs.data) el.append(h("pre", { class: "mc-pre", text: JSON.stringify(hs.data, null, 2) }));
      const b = A.budget();
      el.append(h("h2", { class: "mc-label mc-section", text: "Rate-limit budget" }), kv([["limit", b.limit], ["remaining", b.remaining], ["resets", b.resetAt ? new Date(b.resetAt).toISOString().slice(11, 19) + " UTC" : "—"], ["calls", b.calls], ["429s", b.limited]]));
      if (state.voice) el.append(h("h2", { class: "mc-label mc-section", text: "Voice" }), h("pre", { class: "mc-pre", text: JSON.stringify(state.voice, null, 2) }));
      el.append(h("button", { class: "mc-btn mc-btn-primary", type: "button", text: "Probe", onclick: () => probe(false) }));
    },
  };
  VIEWS.record = VIEWS.captures;
  VIEWS.terminal = () => {};

  function massSummary() {
    const out = state.mass.out;
    if (!out) return h("div", { class: "mc-empty", text: "Select capabilities and press Run. Each one is a separate call to this server; results land in Results." });
    const n = state.data ? state.data.rows : 16;
    const cons = M.consensus(out, n);
    const ok = out.filter((o) => o.status === "ok").length;
    const box = h("div", null, h("h2", { class: "mc-label", text: "Consensus" }), kv([["routes", ok + " ok of " + out.length], ["voters", cons.voters + " per-point routes"], ["majority-flagged samples", cons.majority]]));
    if (cons.top.length) {
      const head = h("thead", null, h("tr", null, h("th", { text: "sample" }), h("th", { text: "votes" })));
      const rows = cons.top.map((t) => h("tr", { dataset: t.v > cons.voters / 2 ? { flag: "1" } : null }, h("td", { text: t.i }), h("td", { text: t.v + " / " + cons.voters })));
      box.append(h("table", { class: "mc-table" }, head, h("tbody", null, rows)));
    }
    if (V && cons.voters) { const svg = V.render("consensus", { consensus: cons, votes: cons.votes, voters: cons.voters }, null, null); if (svg) box.append(svgNode(svg)); }
    return box;
  }
  async function runMass() {
    const caps = M.CAPS.filter((c) => state.mass.sel.includes(c.id));
    if (!caps.length) return;
    const ctrl = new AbortController();
    state.mass.running = true; state.mass.abort = ctrl; setRun("running", "mass run · " + caps.length);
    render();
    const ds = state.data ? { name: state.data.name, rows: state.data.rows, cols: state.data.cols, matrix: state.data.matrix } : null;
    const out = await M.run(A, caps, ds, { domain: state.domain || null, signal: ctrl.signal, cancelled: () => ctrl.signal.aborted }, (o) => { state.mass.out = o; if (state.view === "mass") render(); }, 2);
    state.mass.running = false; state.mass.out = out;
    out.filter((o) => o.status === "ok" || o.status === "error").forEach((o) => pushResult({ kind: "MASS", label: o.label, route: o.route, status: o.status === "ok" ? "ok" : "error", http: o.http, ms: o.ms, corr: o.corr, body: o.body, data: o.data || null, error: o.error || null, sum: o.sum, ds: dsMeta(state.data) }));
    setRun(out.every((o) => o.status === "ok" || o.status === "skipped") ? "ok" : "error", "mass run · " + out.filter((o) => o.status === "ok").length + "/" + out.length);
    announce("Mass run finished");
    render();
  }

  function resultDetail(r) {
    const box = h("div", null);
    const sum = r.sum || {};
    box.append(h("h2", { class: "mc-h1", text: r.label }), sum.headline ? h("p", { class: "mc-lede", text: sum.headline }) : null);
    box.append(kv([["route", r.route], ["status", r.status + (r.http ? " · HTTP " + r.http : "")], ["time", r.ms != null ? r.ms + " ms" : null], ["correlation id", r.corr], ["received", String(r.at).replace("T", " ").slice(0, 19) + " UTC"], ["dataset", r.ds ? r.ds.name + " · " + r.ds.rows + " × " + r.ds.cols : null], ["error", r.error]].concat((sum.kv || []).map((p) => [p[0], p[1]]))));
    if (sum.image) box.append(h("img", { class: "mc-img", src: sum.image, alt: "Server image for " + r.label }));
    if (sum.flags) { const idx = sum.flags.map((f, i) => (f ? i : -1)).filter((i) => i >= 0); if (idx.length) box.append(h("p", { class: "mc-help" }, h("span", { class: "mc-bad", text: plural(idx.length, "flagged sample") + ": " }), idx.slice(0, 40).join(", ") + (idx.length > 40 ? " …" : ""))); }
    if (V && r.data && r.status === "ok") {
      const av = V.available(r.data, r.body, r.ds);
      if (av.length) {
        const chosen = new Set(state.chartSel[r.id] || av.filter((c) => c.rec).map((c) => c.id));
        box.append(h("h3", { class: "mc-label mc-section", text: "Charts · " + av.length + " fit this data" }));
        box.append(h("div", { class: "mc-row" }, av.map((c) => h("button", { class: "mc-chip", type: "button", "aria-pressed": chosen.has(c.id) ? "true" : "false", text: c.label + (c.rec ? " ·" : ""), onclick: () => { if (chosen.has(c.id)) chosen.delete(c.id); else chosen.add(c.id); state.chartSel[r.id] = Array.from(chosen); render(); } })),
          h("button", { class: "mc-chip", type: "button", text: "all", onclick: () => { state.chartSel[r.id] = av.map((c) => c.id); render(); } }),
          h("button", { class: "mc-chip", type: "button", text: "none", onclick: () => { state.chartSel[r.id] = []; render(); } })));
        const grid = h("div", { class: "mc-charts" });
        av.filter((c) => chosen.has(c.id)).forEach((c) => { const svg = V.render(c.id, r.data, r.body, r.ds); if (svg) grid.append(svgNode(svg)); });
        box.append(grid);
      }
    }
    box.append(h("div", { class: "mc-row" },
      r.data ? h("button", { class: "mc-btn", type: "button", text: "JSON", onclick: () => download(slug(r.label) + ".json", "application/json", JSON.stringify({ route: r.route, request: r.body, response: r.data }, null, 2)) }) : null,
      r.data && V ? h("button", { class: "mc-btn", type: "button", text: "CSV", onclick: () => { const csv = V.csv(r.data); if (csv) download(slug(r.label) + ".csv", "text/csv", csv); else toast("No tabular data in this response"); } }) : null,
      r.data && V ? h("button", { class: "mc-btn", type: "button", text: "HTML report", onclick: () => download(slug(r.label) + ".html", "text/html", V.report(r, state.chartSel[r.id] || V.available(r.data, r.body, r.ds).filter((c) => c.rec).map((c) => c.id))) }) : null,
      r.data && G && G.narratePayload ? h("button", { class: "mc-btn", type: "button", text: "Ask Mercury to explain", onclick: () => narrate(r) }) : null));
    if (K && r.status === "ok") box.append(capturePanel(r));
    box.append(h("h3", { class: "mc-label mc-section", text: "Raw response" }), h("pre", { class: "mc-pre", text: JSON.stringify(r.data, null, 2) || "null" }));
    return box;
  }
  const slug = (t) => String(t || "result").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 48) || "result";
  async function narrate(r) {
    openPanel("chat");
    let payload; try { payload = G.narratePayload(r.data); } catch (e) { payload = r.data; }
    const res = await A.narrate(payload, state.chatDomain || null, "normal");
    const m = res.ok ? { role: "agent", text: (res.data && (res.data.narrative || res.data.message || res.data.text)) || JSON.stringify(res.data), meta: "narrate · " + (res.meta ? res.meta.ms + " ms" : "") } : { role: "error", text: "narrate failed · " + res.error };
    state.chat.push(m); chatMsg(m);
  }
  function capturePanel(r) {
    const src = K.source(r, r.ds ? r.ds.total || r.ds.rows : null);
    const box = h("div", null, h("h3", { class: "mc-label mc-section", text: "Capture · " + src.n + " samples" }));
    if (!src.n) { box.append(h("p", { class: "mc-muted", text: "Nothing sample-shaped to capture in this response." })); return box; }
    const selKey = "cap:" + r.id; const sel = state[selKey] || { from: 0, to: src.n - 1 };
    const strip = h("div", { class: "mc-strip", role: "slider", tabindex: "0", "aria-label": "Select samples to capture", "aria-valuemin": "0", "aria-valuemax": String(src.n - 1), "aria-valuenow": String(sel.from), "aria-valuetext": "samples " + sel.from + " to " + sel.to });
    const scores = src.scores || (src.rows ? src.rows.map((x) => x[0]) : null);
    const max = scores ? Math.max(...scores.map((x) => Math.abs(x || 0)), 1e-9) : 1;
    const step = Math.max(1, Math.ceil(src.n / 400));
    for (let i = 0; i < src.n; i += step) {
      const bar = h("span", { class: "mc-strip-bar" });
      bar.style.left = (100 * i) / src.n + "%"; bar.style.width = Math.max(0.25, (100 * step) / src.n) + "%";
      bar.style.height = scores ? Math.max(4, (100 * Math.abs(scores[i] || 0)) / max) + "%" : "40%";
      if (src.flags && src.flags.slice(i, i + step).some(Boolean)) bar.dataset.flag = "1";
      strip.append(bar);
    }
    const shade = h("span", { class: "mc-strip-sel" }); strip.append(shade);
    const place = () => { shade.style.left = (100 * sel.from) / src.n + "%"; shade.style.width = (100 * (sel.to - sel.from + 1)) / src.n + "%"; };
    place();
    const fromIn = h("input", { id: "mc-cap-from-" + r.id, class: "mc-input", type: "number", min: "0", max: String(src.n - 1), value: String(sel.from) });
    const toIn = h("input", { id: "mc-cap-to-" + r.id, class: "mc-input", type: "number", min: "0", max: String(src.n - 1), value: String(sel.to) });
    const set = (a, b) => { sel.from = Math.max(0, Math.min(src.n - 1, Math.min(a, b))); sel.to = Math.max(0, Math.min(src.n - 1, Math.max(a, b))); state[selKey] = sel; fromIn.value = sel.from; toIn.value = sel.to; strip.setAttribute("aria-valuenow", String(sel.from)); strip.setAttribute("aria-valuetext", "samples " + sel.from + " to " + sel.to); place(); };
    const at = (e) => { const rc = strip.getBoundingClientRect(); return Math.floor(((e.clientX - rc.left) / rc.width) * src.n); };
    strip.addEventListener("pointerdown", (e) => { const a = at(e); set(a, a); strip.setPointerCapture(e.pointerId); const mv = (ev) => set(a, at(ev)); const up = () => { strip.removeEventListener("pointermove", mv); strip.removeEventListener("pointerup", up); }; strip.addEventListener("pointermove", mv); strip.addEventListener("pointerup", up); });
    strip.addEventListener("keydown", (e) => { const d = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0; if (!d) return; e.preventDefault(); if (e.shiftKey) set(sel.from, sel.to + d); else set(sel.from + d, sel.to + d); });
    fromIn.addEventListener("change", () => set(+fromIn.value, sel.to)); toIn.addEventListener("change", () => set(sel.from, +toIn.value));
    const labelIn = h("input", { id: "mc-cap-label-" + r.id, class: "mc-input", type: "text", placeholder: "label (optional)" });
    const tags = h("div", { class: "mc-row", role: "radiogroup", "aria-label": "Tag" }, K.TAGS.map((t) => h("button", { class: "mc-chip", type: "button", role: "radio", "aria-checked": state.capTag === t[0] ? "true" : "false", text: t[1], onclick: () => { state.capTag = t[0]; persist(); render(); } })));
    const pads = h("div", { class: "mc-row", role: "radiogroup", "aria-label": "Padding" }, h("span", { class: "mc-label", text: "padding" }), [0, 2, 5, 10].map((p) => h("button", { class: "mc-chip", type: "button", role: "radio", "aria-checked": state.capPad === p ? "true" : "false", text: "±" + p, onclick: () => { state.capPad = p; persist(); render(); } })));
    const firstFlag = src.flags ? src.flags.indexOf(true) : -1;
    box.append(strip,
      h("div", { class: "mc-row" }, h("label", { for: fromIn.id, class: "mc-label", text: "from" }), fromIn, h("label", { for: toIn.id, class: "mc-label", text: "to" }), toIn,
        h("button", { class: "mc-chip", type: "button", text: "Whole", onclick: () => set(0, src.n - 1) }),
        src.flags ? h("button", { class: "mc-chip", type: "button", text: "Flagged span", onclick: () => { const li = src.flags.lastIndexOf(true); if (firstFlag >= 0) set(firstFlag, li); } }) : null,
        src.flags ? h("button", { class: "mc-chip", type: "button", text: "Next event", onclick: () => { const rs = K.runs(src.flags, state.capPad, src.n); const nx = rs.find((x) => x[0] > sel.from) || rs[0]; if (nx) set(nx[0], nx[1]); } }) : null),
      tags, pads,
      h("div", { class: "mc-row" }, h("label", { for: labelIn.id, class: "mc-sr", text: "Capture label" }), labelIn,
        h("button", { class: "mc-btn mc-btn-primary", type: "button", text: "Capture", onclick: () => { const c = K.captureRange(C, r, sel.from, sel.to, { tag: state.capTag, label: labelIn.value }); if (c) toast("Captured · " + c.n + " rows · SHA-256 " + c.sha256.slice(0, 12) + "…"); } }),
        src.flags ? h("button", { class: "mc-btn", type: "button", text: "Capture all flagged events", onclick: () => { const out = K.captureEvents(C, r); toast(out.length ? "Captured " + plural(out.length, "event") : "No flagged events"); } }) : null));
    if (state.capFlash && state.capFlash.rid === r.id) box.append(h("p", { class: "mc-good", text: state.capFlash.text }));
    return box;
  }
  function capDetail(c) {
    const src = c.src || {};
    const box = h("div", null);
    const title = h("input", { id: "mc-cap-title", class: "mc-input", type: "text", value: c.label });
    title.addEventListener("change", () => K.update(C, c.id, { label: title.value }));
    const note = h("textarea", { id: "mc-cap-note", class: "mc-textarea", rows: "3", text: c.note || "" });
    note.addEventListener("change", () => K.update(C, c.id, { note: note.value }));
    box.append(h("label", { for: "mc-cap-title", class: "mc-label", text: "Title" }), title,
      h("div", { class: "mc-row", role: "radiogroup", "aria-label": "Tag" }, K.TAGS.map((t) => h("button", { class: "mc-chip", type: "button", role: "radio", "aria-checked": c.tag === t[0] ? "true" : "false", text: t[1], onclick: () => K.update(C, c.id, { tag: t[0] }) }))),
      h("label", { for: "mc-cap-note", class: "mc-label", text: "Note" }), note,
      kv([["samples", c.from + "–" + c.to + " · " + c.n + " rows" + (src.n ? " of " + src.n : "")], ["flagged", c.flags ? c.flagged + (c.firstFlag != null ? " · first " + c.firstFlag + " · last " + c.lastFlag : "") : null], ["score", c.scoreStats ? "max " + f4(c.scoreStats.max) + " · mean " + f4(c.scoreStats.mean) : null], ["source", src.label], ["route", (src.route || "—") + (src.http ? " · HTTP " + src.http : "")], ["received", src.receivedAt], ["dataset", src.dataset || "request body or built-in sample"], ["captured", c.at], ["SHA-256", c.sha256]]));
    if ((c.colStats || []).length) box.append(h("table", { class: "mc-table" }, h("thead", null, h("tr", null, ["column", "n", "mean", "sd", "min", "max"].map((x) => h("th", { text: x })))), h("tbody", null, c.colStats.slice(0, 12).map((s) => h("tr", null, [s.name, s.n, f4(s.mean), f4(s.sd), f4(s.min), f4(s.max)].map((x) => h("td", { text: String(x) })))))));
    const groups = K.labelGroups(state.caps).filter((g) => g.ids.includes(c.id));
    const g = groups[0];
    box.append(h("div", { class: "mc-row" },
      h("button", { class: "mc-btn", type: "button", text: "CSV", onclick: () => download(slug(c.label) + ".csv", "text/csv", K.toCsv(c, csvSafe)) }),
      h("button", { class: "mc-btn", type: "button", text: "JSON", onclick: () => download(slug(c.label) + ".json", "application/json", JSON.stringify(K.capJson(c), null, 2)) }),
      h("button", { class: "mc-btn", type: "button", text: "Markdown", onclick: () => download(slug(c.label) + ".md", "text/markdown", K.capMarkdown(c)) }),
      g ? h("button", { class: "mc-btn", type: "button", text: "Labels file", onclick: () => download(K.labelsName(g), "text/csv", K.labelsCsv(g.y)) }) : null,
      g ? h("button", { class: "mc-btn", type: "button", text: "Load tier-detect --labels", onclick: () => { const line = K.cmdWithLabels(C, "tier-detect", g, ""); promptIn.value = line; go("terminal"); fromInput(); } }) : null,
      h("button", { class: "mc-btn mc-btn-danger", type: "button", text: state.capDelArm === c.id ? "Confirm delete" : "Delete", onclick: () => { if (state.capDelArm === c.id) { state.caps = state.caps.filter((x) => x.id !== c.id); state.capDelArm = null; persist(); render(); } else { state.capDelArm = c.id; render(); setTimeout(() => { if (state.capDelArm === c.id) { state.capDelArm = null; render(); } }, 3000); } } })));
    if (c.rows) box.append(h("h3", { class: "mc-label mc-section", text: "Rows (first 300)" }), h("table", { class: "mc-table" }, h("thead", null, h("tr", null, ["#"].concat(c.names, c.scores ? ["score"] : [], c.flags ? ["flag"] : []).map((x) => h("th", { text: x })))),
      h("tbody", null, c.rows.slice(0, 300).map((row, i) => h("tr", { dataset: c.flags && c.flags[i] ? { flag: "1" } : null }, [c.from + i].concat(row.map(f4), c.scores ? [f4(c.scores[i])] : [], c.flags ? [c.flags[i] ? "1" : "0"] : []).map((x) => h("td", { text: String(x) })))))));
    return box;
  }
  function recDetail(r) {
    const box = h("div", null, h("h2", { class: "mc-h1", text: r.title }));
    box.append(kv([["started", r.startedAt], ["ended", r.endedAt], ["duration", K.clock(r.dur)], ["events", (r.events || []).length], ["counts", K.countText ? K.countText(r) : ""], ["engine", r.engine], ["dataset", r.dataset]]));
    box.append(h("div", { class: "mc-row" },
      h("button", { class: "mc-btn", type: "button", text: ".cast", onclick: () => download(slug(r.title) + ".cast", "application/x-asciicast", K.cast(r)) }),
      h("button", { class: "mc-btn", type: "button", text: ".md", onclick: () => download(slug(r.title) + ".md", "text/markdown", K.markdown(r)) }),
      h("button", { class: "mc-btn", type: "button", text: ".json", onclick: () => download(slug(r.title) + ".json", "application/json", JSON.stringify(r, null, 2)) }),
      h("button", { class: "mc-btn", type: "button", text: "Replay .sh", onclick: () => download(slug(r.title) + ".sh", "text/x-shellscript", K.replaySh(r)) }),
      h("button", { class: "mc-btn mc-btn-danger", type: "button", text: "Delete", onclick: () => { state.recs = state.recs.filter((x) => x.id !== r.id); persist(); render(); } })));
    box.append(h("table", { class: "mc-table" }, h("tbody", null, (r.events || []).slice(0, 500).map((e) => h("tr", null, h("td", { class: "mc-muted", text: K.clock(e.t) }), h("td", { text: e.k }), h("td", { text: e.x }))))));
    return box;
  }
  function apiForm(key) {
    const schema = A.withParams(key, A.SCHEMAS[key] || {});
    const body0 = A.body(schema, {}, { requiredOnly: false });
    const ta = h("textarea", { id: "mc-api-body", class: "mc-textarea", rows: "10", spellcheck: "false", text: JSON.stringify(body0, null, 2) });
    const params = A.pathParams(key).concat(Object.keys(schema).filter((k) => schema[k].q));
    const inputs = {};
    const box = h("div", { class: "mc-card" }, h("div", { class: "mc-row" }, h("h3", { text: key }), h("span", { class: "mc-spacer" }), h("button", { class: "mc-iconbtn", type: "button", title: "Close form", text: "×", onclick: () => { state.apiForm = null; render(); } })), fieldTable(schema));
    params.forEach((p) => { const id = "mc-api-p-" + p; inputs[p] = h("input", { id, class: "mc-input", type: "text" }); box.append(h("div", { class: "mc-field" }, h("label", { for: id, class: "mc-label", text: p }), inputs[p])); });
    const status = h("p", { class: "mc-help", "aria-live": "polite" });
    box.append(h("label", { for: "mc-api-body", class: "mc-label", text: "Request body (JSON)" }), ta, status,
      h("div", { class: "mc-row" },
        h("button", { class: "mc-btn mc-btn-primary", type: "button", text: "Send", onclick: async () => {
          let b; try { b = ta.value.trim() ? JSON.parse(ta.value) : null; } catch (e) { status.textContent = "Body is not valid JSON: " + e.message; return; }
          const vals = {}; params.forEach((p) => { vals[p] = inputs[p].value; });
          const miss = A.missing(schema, b || {}, vals); if (miss.length) { status.textContent = "Missing: " + miss.join(", "); return; }
          const method = key.split(" ")[0], path = A.url(key, vals, schema);
          status.textContent = "Sending…";
          const r = method === "GET" ? await A.get(path) : method === "PUT" ? await A.put(path, b) : method === "DELETE" ? await A.del(path) : await A.post(path, b);
          const rec = pushResult({ kind: "API", label: key, route: key, status: r.ok ? "ok" : "error", http: r.status, ms: r.meta ? r.meta.ms : null, corr: r.meta ? r.meta.correlationId : null, body: b, data: r.data, error: r.ok ? null : r.error, ds: dsMeta(state.data) });
          status.textContent = r.ok ? "HTTP " + r.status + " · " + (r.meta ? r.meta.ms + " ms" : "") + " · saved to Results" : "HTTP " + (r.status || r.kind) + " · " + r.error;
          state.resultSel = rec.id;
        } }),
        h("button", { class: "mc-btn", type: "button", text: "Copy curl", onclick: () => { let b = null; try { b = JSON.parse(ta.value); } catch (e) { b = null; } copy(A.curl(key, b, {}, schema)); } })));
    return box;
  }
  async function importMcp() {
    const f = await pickFile(".jsonl,.json,.txt");
    if (!f) return;
    const picked = CAT.mcp.filter((t) => state.catSel.includes(catKey("mcp", t.n))).map((t) => t.n);
    const rows = M.mcpImport(await f.text(), picked.length ? picked : CAT.mcp.map((t) => t.n));
    rows.forEach((r) => pushResult({ kind: "MCP", label: r.label, route: r.route, status: r.status, http: null, ms: null, body: null, data: r.data, ds: null }));
    toast("Imported " + plural(rows.length, "MCP result"));
    render();
  }

  /* ================================================================== self-test */

  async function runDiag() {
    const t0 = performance.now(), checks = [];
    const ck = (name, ok, detail) => checks.push({ name, ok: !!ok, detail: detail || "" });
    ck("catalog loaded", CAT && CAT.commands.length > 0, CAT.commands.length + " commands");
    ck("every command has a section", commands.every((c) => c.section), "");
    Object.entries(A.CLI_TO_ROUTE).forEach(([path, fn]) => { const route = fn({}); ck("route for " + path + " is in the catalog", cmdBy(path) && routeKeys.has(route) && apiPaths.has(route), route); });
    commands.slice(0, 400).forEach((c) => { const vals = {}; c.opts.forEach((o) => { if (o.choices) vals[o.k] = o.choices[0]; else if (o.flag) vals[o.k] = true; else if (o.type === "float") vals[o.k] = "0.5"; else if (o.type === "int") vals[o.k] = "3"; else vals[o.k] = "x y"; }); const line = build(c.path, vals), p = parse(line); ck("build → parse round-trip · " + c.path, p.path === c.path && c.opts.every((o) => String(p.vals[o.k]) === String(vals[o.k])), p.errs[0] || ""); });
    const pd = parse("mercury-agent detect --bogus 1"); ck("unknown option is reported in click's words", pd.errs.includes("No such option: --bogus"), pd.errs[0]);
    const pc = parse("mercury-agent tier-detect -i a.csv --method nope"); ck("invalid choice is reported", pc.errs.some((e) => e.includes("is not one of")), pc.errs[0]);
    const n = 64, sq = Array.from({ length: n }, (_, i) => Math.sin(i / 4) + (i === 40 ? 6 : 0));
    const sample = { scores: sq.map(Math.abs), anomalies: sq.map((v) => Math.abs(v) > 3), threshold: 2.5 };
    let charts = 0, styled = 0;
    V.CHARTS.forEach((c) => { const out = V.render(c.id, sample, { data: sq }, null); if (out) { charts++; if (/\sstyle\s*=|<style/.test(out)) styled++; } });
    ck("charts render without inline styles (CSP)", charts > 0 && styled === 0, charts + " rendered · " + styled + " with style");
    if (K) { const hash = K.sha256("abc"); ck("SHA-256 FIPS 180-4 vector ('abc')", hash === "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", hash.slice(0, 16)); ck("CRC-32 IEEE vector ('123456789')", (K.crc32(new TextEncoder().encode("123456789")) >>> 0) === 0xcbf43926, ""); }
    const hr = await A.health(); ck("GET /health", hr.ok, hr.ok ? "HTTP " + hr.status : hr.error);
    const oa = await A.openapi();
    if (oa.ok && oa.data && oa.data.paths) {
      const live = new Set(); Object.entries(oa.data.paths).forEach(([p, ops]) => Object.keys(ops).forEach((m) => live.add(m.toUpperCase() + " " + p)));
      const stale = [...apiPaths].filter((k) => !live.has(k));
      ck("catalog API routes all served by this server", !stale.length, stale.length ? stale.slice(0, 4).join(", ") : apiPaths.size + " routes");
    } else ck("GET /openapi.json", false, oa.error || "not served");
    const vs = await A.voiceStatus(); ck("GET /api/v1/voice/status", vs.ok, vs.ok ? "HTTP " + vs.status : vs.error);
    state.diag = { checks, passed: checks.filter((c) => c.ok).length, failed: checks.filter((c) => !c.ok).length, ms: Math.round(performance.now() - t0) };
    termRow({ text: "self-test · " + state.diag.passed + " passed · " + state.diag.failed + " failed · " + state.diag.ms + " ms", k: state.diag.failed ? "err" : "ok" });
    checks.filter((c) => !c.ok).slice(0, 10).forEach((c) => termRow({ text: "  FAIL " + c.name + (c.detail ? " · " + c.detail : ""), k: "err" }));
    termRow("");
    window.__mercuryConsoleDiag = state.diag;
    if (state.view === "diag") render();
    return state.diag;
  }

  /* ================================================================== palette */

  const palette = $("mc-palette"), palIn = $("mc-palette-in"), palList = $("mc-palette-list");
  let palItems = [], palIdx = 0;
  function paletteIndex() {
    return [].concat(
      NAV.flatMap(([, items]) => items.map(([id, , label]) => ({ kind: "view", t: label, s: "open", go: () => go(id) }))),
      commands.map((c) => ({ kind: "command", t: c.path, s: c.doc, go: () => selectCommand(c.path) })),
      CAT.detectors.map((d) => ({ kind: "detector", t: d.name, s: d.desc + " · " + d.fam, go: () => { go("detectors"); showRef("detector", d); } })),
      CAT.library.map((l) => ({ kind: "library", t: l.c, s: l.call, go: () => { go("detectors"); showRef("library", l); } })),
      CAT.loaders.map((l) => ({ kind: "domain", t: l.id, s: l.cls, go: () => { go("domains"); showRef("loader", l); } })),
      CAT.tools.flatMap(([g, names]) => names.map((n) => ({ kind: "tool", t: n, s: g, go: () => { go("tools"); showRef("tool", { name: n, group: g }); } }))),
      CAT.mcp.map((t) => ({ kind: "mcp", t: t.n, s: t.d, go: () => { go("mcp"); showRef("mcp", t); } })),
      CAT.api.map((r) => ({ kind: "api", t: r.m + " " + r.path, s: r.s || r.g, go: () => { go("api"); showRef("api", r); } })),
      CAT.runs.map((r) => ({ kind: "run", t: r.p, s: r.s, go: () => { go("runs"); showRef("run", r); } })));
  }
  let PAL = null;
  function score(q, it) {
    const t = it.t.toLowerCase(), s = (it.s || "").toLowerCase();
    if (!q) return 1;
    if (t === q) return 100; if (t.startsWith(q)) return 80; if (t.includes(q)) return 60 - t.indexOf(q) * 0.1;
    let k = 0; for (const ch of t) if (ch === q[k]) k++; if (k === q.length) return 30;
    return s.includes(q) ? 10 : 0;
  }
  function renderPalette() {
    const q = palIn.value.trim().toLowerCase();
    PAL = PAL || paletteIndex();
    palItems = PAL.map((it) => [score(q, it), it]).filter((x) => x[0] > 0).sort((a, b) => b[0] - a[0]).slice(0, 60).map((x) => x[1]);
    palIdx = Math.min(palIdx, Math.max(0, palItems.length - 1));
    clear(palList);
    palItems.forEach((it, i) => {
      const li = h("li", { class: "mc-palette-opt", role: "option", id: "mc-pal-" + i, "aria-selected": i === palIdx ? "true" : "false" }, h("span", { class: "mc-palette-kind", text: it.kind }), h("span", null, it.t, h("span", { class: "mc-palette-sub", text: it.s || "" })));
      li.addEventListener("click", () => { palette.close(); it.go(); });
      li.addEventListener("mousemove", () => { if (palIdx !== i) { palIdx = i; renderPalette(); } });
      palList.append(li);
    });
    palIn.setAttribute("aria-activedescendant", palItems.length ? "mc-pal-" + palIdx : "");
    if (!palItems.length) palList.append(h("li", { class: "mc-empty", text: "No match." }));
  }
  function openPalette() { palIn.value = ""; palIdx = 0; renderPalette(); palette.showModal(); palIn.focus(); }
  palIn.addEventListener("input", () => { palIdx = 0; renderPalette(); });
  palIn.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") { e.preventDefault(); palIdx = (palIdx + (e.key === "ArrowDown" ? 1 : -1) + palItems.length) % Math.max(1, palItems.length); renderPalette(); const el = $("mc-pal-" + palIdx); if (el) el.scrollIntoView({ block: "nearest" }); }
    else if (e.key === "Enter") { e.preventDefault(); const it = palItems[palIdx]; if (it) { palette.close(); it.go(); } }
  });
  $("mc-search").addEventListener("click", openPalette);
  // Escape and a click on the backdrop close every dialog. The native cancel is not relied on alone: Chrome's
  // close-watcher rules can ignore Escape for a dialog opened from a keyboard shortcut.
  ["mc-palette", "mc-conn-dialog", "mc-settings-dialog"].forEach((id) => {
    const dlg = $(id);
    dlg.addEventListener("keydown", (e) => { if (e.key === "Escape") { e.preventDefault(); dlg.close(); } });
    dlg.addEventListener("click", (e) => { if (e.target === dlg) { const r = dlg.getBoundingClientRect(); if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) dlg.close(); } });
  });
  palette.addEventListener("close", () => { if (state.view === "terminal") promptIn.focus(); });
  document.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") { e.preventDefault(); if (palette.open) palette.close(); else openPalette(); }
  });

  /* ================================================================== dialogs: connection, settings */

  const connDlg = $("mc-conn-dialog");
  $("mc-conn").addEventListener("click", () => {
    const cfg = A.cfg; $("mc-conn-origin").textContent = location.origin; $("mc-conn-key").value = cfg.apiKey || ""; $("mc-conn-remember").checked = !!cfg.remember; $("mc-conn-timeout").value = cfg.timeoutMs;
    $("mc-conn-status").textContent = state.health ? (state.health.ok ? "Engine answered HTTP " + state.health.status : "Engine unreachable: " + state.health.error) : "";
    connDlg.showModal();
  });
  $("mc-conn-save").addEventListener("click", async () => {
    A.set({ apiKey: $("mc-conn-key").value.trim(), remember: $("mc-conn-remember").checked, timeoutMs: Math.max(1000, Math.min(600000, +$("mc-conn-timeout").value || 15000)) });
    $("mc-conn-status").textContent = "Probing…";
    const r = await probe(false);
    $("mc-conn-status").textContent = r.ok ? "Saved · engine answered HTTP " + r.status : "Saved · engine unreachable: " + r.error;
  });
  const settings = Object.assign({ theme: "dark", contrast: false, motion: false, focus: false, scale: "1" }, readJson(LS_SETTINGS, {}));
  function applySettings() {
    const html = document.documentElement;
    const theme = settings.theme === "system" ? (window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark") : settings.theme;
    html.dataset.theme = theme;
    html.classList.toggle("mc-contrast", !!settings.contrast);
    html.classList.toggle("mn-reduce-motion", !!settings.motion);
    html.classList.toggle("mc-focus", !!settings.focus);
    html.style.setProperty("--scale", String(+settings.scale || 1));
    writeJson(LS_SETTINGS, settings);
  }
  $("mc-settings").addEventListener("click", () => { $("mc-set-theme").value = settings.theme; $("mc-set-contrast").checked = !!settings.contrast; $("mc-set-motion").checked = !!settings.motion; $("mc-set-focus").checked = !!settings.focus; $("mc-set-scale").value = String(settings.scale); $("mc-settings-dialog").showModal(); });
  [["mc-set-theme", "theme", "value"], ["mc-set-contrast", "contrast", "checked"], ["mc-set-motion", "motion", "checked"], ["mc-set-focus", "focus", "checked"], ["mc-set-scale", "scale", "value"]].forEach(([id, key, prop]) => $(id).addEventListener("change", (e) => { settings[key] = e.target[prop]; applySettings(); }));
  applySettings();

  /* ================================================================== boot */

  function boot() {
    $("mc-version").textContent = "v" + CAT.version;
    $("mc-boot-version").textContent = "v" + CAT.version;
    const steps = [["catalog · " + CAT.nCommands + " commands · " + CAT.detectors.length + " detectors", 0.35], ["client · same-origin " + location.origin, 0.6], ["engine · probing /health", 0.85]];
    const fill = $("mc-boot-fill"), status = $("mc-boot-status");
    let k = 0;
    const tick = () => { if (k < steps.length) { status.textContent = steps[k][0]; fill.style.width = steps[k][1] * 100 + "%"; k++; setTimeout(tick, 160); } };
    tick();
    let entered = false;
    const enter = () => { if (entered) return; entered = true; const b = $("mc-boot"); b.dataset.leaving = "1"; setTimeout(() => { b.hidden = true; root.dataset.ready = "true"; promptIn.focus(); }, 240); };
    $("mc-boot-enter").addEventListener("click", enter);
    $("mc-boot").addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === "Escape") enter(); });
    try { if (sessionStorage.getItem("mercury_cli_booted")) setTimeout(enter, 0); sessionStorage.setItem("mercury_cli_booted", "1"); } catch (e) { /* no session storage */ }
    $("mc-boot-enter").focus();
    probe(false).then((r) => { fill.style.width = "100%"; status.textContent = r.ok ? "engine · live · " + location.origin : "engine · unreachable (the console still prepares commands)"; });
  }

  window.addEventListener("beforeunload", () => { if (state.rec && K) K.recStop(C, true); });

  /* ================================================================== start */

  banner();
  termRows([{ text: "Mercury Console · " + CAT.nCommands + " commands · type :help", k: "dim" }, { text: "" }]);
  renderNav();
  promptIn.value = build(state.sel, state.vals[state.sel]);
  renderOpts(); renderSuggest(); renderPanel(); render();
  boot();
  window.MercuryConsole = { state, parse, build, validate, completions, runDiag, submitLine };
})();

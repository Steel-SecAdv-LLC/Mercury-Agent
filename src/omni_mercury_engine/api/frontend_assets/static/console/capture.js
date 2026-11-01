/* Copyright (C) 2025 Steel Security Advisors LLC
   SPDX-License-Identifier: AGPL-3.0-or-later
   Ported from docs/design/mercury_console/design/mercury-cli-capture.js (template value builders removed; app.js renders the DOM). */
/* Design notes: Mercury CLI — captures (sections of data with provenance) and session recordings.
   Pure helpers plus UI value builders the console component (C) delegates to.
   SHA-256 per FIPS 180-4 · CRC-32 (IEEE 802.3) · ZIP stored entries (PKWARE APPNOTE) · asciicast v2.
   Labels files match cli.py _load_labels: one 0/1 value per row, header row skipped. */
(function () {
  "use strict";
  const KEY = "mercury_cli_capture_v1";
  const TAGS = [["anomaly", "ANOMALY", "var(--anomaly)"], ["normal", "NORMAL", "var(--normal)"], ["review", "REVIEW", "var(--warning)"], ["evidence", "EVIDENCE", "var(--mercury-300)"]];
  const TAGC = {}, TAGN = {}; TAGS.forEach((t) => { TAGC[t[0]] = t[2]; TAGN[t[0]] = t[1]; });
  const fin = (x) => typeof x === "number" && isFinite(x);
  const num = (x) => { const v = typeof x === "number" ? x : x == null || x === "" ? NaN : +x; return isFinite(v) ? v : null; };
  const f4 = (x) => (x == null || !isFinite(x) ? "—" : Math.abs(x) >= 1e5 || (Math.abs(x) < 1e-3 && x !== 0) ? Number(x).toExponential(3) : String(+Number(x).toFixed(4)));
  const pad2 = (v) => String(v).padStart(2, "0");
  const clock = (ms) => { const t = Math.max(0, Math.floor((ms || 0) / 1000)), h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), s = t % 60; return (h ? h + ":" + pad2(m) : pad2(m)) + ":" + pad2(s); };
  const slug = (t) => String(t || "").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 48) || "item";
  const utc = (iso) => (iso ? String(iso).replace("T", " ").slice(0, 19) + " UTC" : "—");
  const enc = new TextEncoder(), dec = new TextDecoder();
  const utf8 = (s) => (s instanceof Uint8Array ? s : enc.encode(String(s)));
  const copy = (t) => { try { navigator.clipboard.writeText(String(t)); } catch (e) {} };
  const shq = (x) => (/^[\w./:@%+=,-]+$/.test(String(x)) ? String(x) : "'" + String(x).replace(/'/g, "'\\''") + "'");

  /* ---------- statistics and sections ---------- */
  function stats(xs) {
    const v = (xs || []).filter(fin); if (!v.length) return null;
    const n = v.length, mean = v.reduce((a, b) => a + b, 0) / n, sd = n > 1 ? Math.sqrt(v.reduce((a, b) => a + (b - mean) * (b - mean), 0) / (n - 1)) : 0, s = v.slice().sort((a, b) => a - b);
    const q = (p) => { const i = (s.length - 1) * p, lo = Math.floor(i), hi = Math.ceil(i); return s[lo] + (s[hi] - s[lo]) * (i - lo); };
    return { n, mean, sd, min: s[0], max: s[s.length - 1], median: q(0.5), p05: q(0.05), p95: q(0.95) };
  }
  function source(res, total) {
    const sm = (res && res.sum) || {}, ds = res && res.ds, body = res && res.body;
    let rows = null, names = [];
    if (ds && Array.isArray(ds.matrix) && ds.matrix.length) { rows = ds.matrix; names = (ds.names || []).slice(); }
    else if (body) { const d = Array.isArray(body.data) ? body.data : Array.isArray(body.observations) ? body.observations : null; if (d && d.length) rows = d.map((r) => (Array.isArray(r) ? r : r && typeof r === "object" ? Object.values(r) : [r])); }
    const sc = Array.isArray(sm.scores) ? sm.scores : null, fl = Array.isArray(sm.flags) ? sm.flags : null;
    const n = Math.max(rows ? rows.length : 0, sc ? sc.length : 0, fl ? fl.length : 0);
    const okRows = rows && rows.length === n, cols = okRows ? rows.reduce((a, r) => Math.max(a, r.length), 0) : 0;
    names = Array.from({ length: cols }, (_, j) => (names[j] != null && String(names[j]).trim() ? String(names[j]) : "col" + (j + 1)));
    return { n, rows: okRows ? rows : null, names, scores: sc && sc.length === n ? sc : null, flags: fl && fl.length === n ? fl.map(Boolean) : null, total: Math.max(n, total || 0) };
  }
  function slice(src, from, to) {
    const n = src.n; if (!n) return { from: 0, to: -1, n: 0, rows: null, names: [], scores: null, flags: null, colStats: [], scoreStats: null, flagged: 0, firstFlag: null, lastFlag: null };
    let a = Math.round(+from), b = Math.round(+to); if (!isFinite(a)) a = 0; if (!isFinite(b)) b = n - 1; if (a > b) { const t = a; a = b; b = t; }
    a = Math.max(0, Math.min(n - 1, a)); b = Math.max(0, Math.min(n - 1, b));
    const rows = src.rows ? src.rows.slice(a, b + 1).map((r) => src.names.map((_, j) => num(r[j]))) : null;
    const scores = src.scores ? src.scores.slice(a, b + 1).map(num) : null, flags = src.flags ? src.flags.slice(a, b + 1) : null;
    const colStats = rows ? src.names.map((nm, j) => Object.assign({ name: nm }, stats(rows.map((r) => r[j])) || { n: 0 })) : [];
    const fi = flags ? flags.indexOf(true) : -1, li = flags ? flags.lastIndexOf(true) : -1;
    return { from: a, to: b, n: b - a + 1, rows, names: rows ? src.names.slice() : [], scores, flags, colStats, scoreStats: scores ? stats(scores) : null, flagged: flags ? flags.filter(Boolean).length : 0, firstFlag: fi >= 0 ? a + fi : null, lastFlag: li >= 0 ? a + li : null };
  }
  function runs(flags, pad, n) {
    const out = []; if (!flags) return out; const p = Math.max(0, pad | 0); let i = 0;
    while (i < flags.length) { if (flags[i]) { let j = i; while (j + 1 < flags.length && flags[j + 1]) j++; out.push([Math.max(0, i - p), Math.min(n - 1, j + p)]); i = j + 1; } else i++; }
    const m = []; out.forEach((r) => { const last = m[m.length - 1]; if (last && r[0] <= last[1] + 1) last[1] = Math.max(last[1], r[1]); else m.push(r.slice()); });
    return m;
  }
  function labels(caps, total) { const y = new Array(Math.max(0, total | 0)).fill(0); (caps || []).filter((c) => c.tag === "anomaly").forEach((c) => { for (let i = Math.max(0, c.from); i <= c.to && i < y.length; i++) y[i] = 1; }); return y; }
  function labelGroups(caps) {
    const m = new Map();
    (caps || []).forEach((c) => { const s = c.src || {}, total = Math.max(s.total || 0, s.n || 0, (c.to | 0) + 1), key = s.dataset ? "ds:" + s.dataset : "res:" + s.resultId; let g = m.get(key); if (!g) { g = { key, dataset: s.dataset || null, total, ids: [], caps: [] }; m.set(key, g); } g.total = Math.max(g.total, total); g.ids.push(c.id); g.caps.push(c); });
    return Array.from(m.values()).map((g) => Object.assign(g, { y: labels(g.caps, g.total) }));
  }
  const labelsCsv = (y) => "label\n" + y.join("\n") + "\n";
  const labelsName = (g) => String(g && g.dataset ? g.dataset.replace(/\.[^./]+$/, "") : "data").replace(/[^\w.-]+/g, "-") + ".labels.csv";
  function toCsv(sl, safe) {
    const sf = safe || ((v) => (v == null ? "" : String(v)));
    const head = ["index"].concat(sl.names || []).concat(sl.scores ? ["score"] : []).concat(sl.flags ? ["is_anomaly"] : []);
    const L = [head.map(sf).join(",")];
    for (let k = 0; k < sl.n; k++) { const r = [sl.from + k].concat(sl.rows ? sl.rows[k] : []).concat(sl.scores ? [sl.scores[k]] : []).concat(sl.flags ? [sl.flags[k] ? 1 : 0] : []); L.push(r.map((v) => (v == null ? "" : sf(v))).join(",")); }
    return L.join("\n") + "\n";
  }

  /* ---------- SHA-256, CRC-32, ZIP ---------- */
  const K = new Uint32Array([0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2]);
  function sha256(input) {
    const bytes = utf8(input), l = bytes.length, nb = ((l + 9 + 63) >> 6) << 6, m = new Uint8Array(nb); m.set(bytes); m[l] = 0x80;
    const dv = new DataView(m.buffer), bits = l * 8; dv.setUint32(nb - 4, bits >>> 0); dv.setUint32(nb - 8, Math.floor(bits / 4294967296));
    const H = new Uint32Array([0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]), W = new Uint32Array(64);
    const r = (x, n) => (x >>> n) | (x << (32 - n));
    for (let o = 0; o < nb; o += 64) {
      for (let i = 0; i < 16; i++) W[i] = dv.getUint32(o + i * 4);
      for (let i = 16; i < 64; i++) { const x = W[i - 15], y = W[i - 2], s0 = r(x, 7) ^ r(x, 18) ^ (x >>> 3), s1 = r(y, 17) ^ r(y, 19) ^ (y >>> 10); W[i] = (W[i - 16] + s0 + W[i - 7] + s1) >>> 0; }
      let a = H[0], b = H[1], c = H[2], d = H[3], e = H[4], f = H[5], g = H[6], h = H[7];
      for (let i = 0; i < 64; i++) { const S1 = r(e, 6) ^ r(e, 11) ^ r(e, 25), ch = (e & f) ^ (~e & g), t1 = (h + S1 + ch + K[i] + W[i]) >>> 0, S0 = r(a, 2) ^ r(a, 13) ^ r(a, 22), mj = (a & b) ^ (a & c) ^ (b & c), t2 = (S0 + mj) >>> 0; h = g; g = f; f = e; e = (d + t1) >>> 0; d = c; c = b; b = a; a = (t1 + t2) >>> 0; }
      H[0] = (H[0] + a) >>> 0; H[1] = (H[1] + b) >>> 0; H[2] = (H[2] + c) >>> 0; H[3] = (H[3] + d) >>> 0; H[4] = (H[4] + e) >>> 0; H[5] = (H[5] + f) >>> 0; H[6] = (H[6] + g) >>> 0; H[7] = (H[7] + h) >>> 0;
    }
    return Array.from(H).map((x) => x.toString(16).padStart(8, "0")).join("");
  }
  let CRC = null;
  function crc32(u8) {
    if (!CRC) { CRC = new Uint32Array(256); for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; CRC[n] = c >>> 0; } }
    let c = 0xffffffff; for (let i = 0; i < u8.length; i++) c = CRC[(c ^ u8[i]) & 0xff] ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0;
  }
  function zip(files) {
    const now = new Date(), dt = ((now.getHours() << 11) | (now.getMinutes() << 5) | (now.getSeconds() >> 1)) & 0xffff, dd = (((now.getFullYear() - 1980) << 9) | ((now.getMonth() + 1) << 5) | now.getDate()) & 0xffff;
    const parts = [], cd = []; let off = 0;
    files.forEach((f) => {
      const name = utf8(f.name), data = utf8(f.text), crc = crc32(data);
      const lh = new DataView(new ArrayBuffer(30)); lh.setUint32(0, 0x04034b50, true); lh.setUint16(4, 20, true); lh.setUint16(6, 0x0800, true); lh.setUint16(8, 0, true); lh.setUint16(10, dt, true); lh.setUint16(12, dd, true); lh.setUint32(14, crc, true); lh.setUint32(18, data.length, true); lh.setUint32(22, data.length, true); lh.setUint16(26, name.length, true); lh.setUint16(28, 0, true);
      parts.push(new Uint8Array(lh.buffer), name, data);
      const ch = new DataView(new ArrayBuffer(46)); ch.setUint32(0, 0x02014b50, true); ch.setUint16(4, 20, true); ch.setUint16(6, 20, true); ch.setUint16(8, 0x0800, true); ch.setUint16(10, 0, true); ch.setUint16(12, dt, true); ch.setUint16(14, dd, true); ch.setUint32(16, crc, true); ch.setUint32(20, data.length, true); ch.setUint32(24, data.length, true); ch.setUint16(28, name.length, true); ch.setUint16(30, 0, true); ch.setUint16(32, 0, true); ch.setUint16(34, 0, true); ch.setUint16(36, 0, true); ch.setUint32(38, 0, true); ch.setUint32(42, off, true);
      cd.push(new Uint8Array(ch.buffer), name);
      off += 30 + name.length + data.length;
    });
    const cdSize = cd.reduce((a, p) => a + p.length, 0), eo = new DataView(new ArrayBuffer(22));
    eo.setUint32(0, 0x06054b50, true); eo.setUint16(4, 0, true); eo.setUint16(6, 0, true); eo.setUint16(8, files.length, true); eo.setUint16(10, files.length, true); eo.setUint32(12, cdSize, true); eo.setUint32(16, off, true); eo.setUint16(20, 0, true);
    const all = parts.concat(cd, [new Uint8Array(eo.buffer)]), out = new Uint8Array(all.reduce((a, p) => a + p.length, 0)); let p = 0; all.forEach((x) => { out.set(x, p); p += x.length; });
    return out;
  }
  function unzip(u8) {
    const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength); let e = u8.length - 22; while (e >= 0 && dv.getUint32(e, true) !== 0x06054b50) e--; if (e < 0) return null;
    const cnt = dv.getUint16(e + 10, true), out = []; let p = dv.getUint32(e + 16, true);
    for (let i = 0; i < cnt; i++) {
      if (dv.getUint32(p, true) !== 0x02014b50) return null;
      const crc = dv.getUint32(p + 16, true), size = dv.getUint32(p + 24, true), nl = dv.getUint16(p + 28, true), el = dv.getUint16(p + 30, true), cl = dv.getUint16(p + 32, true), lo = dv.getUint32(p + 42, true);
      const name = dec.decode(u8.subarray(p + 46, p + 46 + nl)), lnl = dv.getUint16(lo + 26, true), lel = dv.getUint16(lo + 28, true), data = u8.subarray(lo + 30 + lnl + lel, lo + 30 + lnl + lel + size);
      out.push({ name, size, crcOk: crc32(data) === crc, localSig: dv.getUint32(lo, true) === 0x04034b50, text: dec.decode(data) }); p += 46 + nl + el + cl;
    }
    return out;
  }

  /* ---------- recording exports ---------- */
  function counts(r) { const c = { cmd: 0, out: 0, result: 0, capture: 0, mark: 0, user: 0, agent: 0 }; ((r && r.events) || []).forEach((e) => { if (e.k in c) c[e.k]++; }); return c; }
  function countText(r) { const c = counts(r); return c.cmd + " commands · " + c.result + " results · " + c.capture + " captures · " + c.mark + " marks · " + ((r && r.events) || []).length + " events"; }
  function cast(rec) {
    const ESC = String.fromCharCode(27), head = { version: 2, width: 120, height: 36, timestamp: Math.floor((rec.t0 || Date.parse(rec.startedAt) || 0) / 1000), title: String(rec.title || "Mercury CLI session"), env: { SHELL: "/bin/bash", TERM: "xterm-256color" } };
    const L = [JSON.stringify(head)]; let last = 0;
    const ev = (t, k, x) => { const tt = Math.max(last, Math.round(t || 0) / 1000); last = tt; L.push(JSON.stringify([+tt.toFixed(3), k, x])); };
    (rec.events || []).forEach((e) => { const x = String(e.x == null ? "" : e.x).replace(/\r?\n/g, "\r\n");
      if (e.k === "out") ev(e.t, "o", x + "\r\n");
      else if (e.k === "cmd") ev(e.t, "i", x + "\r");
      else if (e.k === "mark") ev(e.t, "m", x);
      else if (e.k === "result") ev(e.t, "o", ESC + "[36m[result] " + x + ESC + "[0m\r\n");
      else if (e.k === "capture") ev(e.t, "o", ESC + "[33m[capture] " + x + ESC + "[0m\r\n");
      else if (e.k === "user" || e.k === "agent") ev(e.t, "o", ESC + "[35m[" + (e.k === "user" ? "you → mercury" : "mercury") + "] " + x + ESC + "[0m\r\n"); });
    return L.join("\n") + "\n";
  }
  function markdown(rec) {
    const L = ["# " + String(rec.title || "Mercury CLI session").replace(/\n/g, " "), "", "- Recorded: " + utc(rec.startedAt) + " → " + utc(rec.endedAt) + " (" + clock(rec.dur) + ")", "- Engine: " + (rec.engine || "—"), "- Dataset: " + (rec.dataset || "none loaded"), "- " + countText(rec), "", "## Timeline", ""];
    let lastCmd = null, buf = []; const flush = () => { if (buf.length) { L.push("````text"); buf.forEach((x) => L.push(x)); L.push("````", ""); buf = []; } };
    (rec.events || []).forEach((e) => { const x = String(e.x == null ? "" : e.x), t = "**+" + clock(e.t) + "**";
      if (e.k === "out") { if (lastCmd != null && x === "$ " + lastCmd) { lastCmd = null; return; } buf.push(x); return; } flush();
      if (e.k === "cmd") lastCmd = x;
      if (e.k === "cmd") L.push(t + " · `" + x.replace(/`/g, "'") + "`", "");
      else if (e.k === "mark") L.push(t + " · **MARK** · " + x, "");
      else if (e.k === "result") L.push(t + " · result · " + x, "");
      else if (e.k === "capture") L.push(t + " · capture · " + x, "");
      else if (e.k === "user") L.push(t + " · you → Mercury · " + x.replace(/\s+/g, " "), "");
      else if (e.k === "agent") L.push(t + " · Mercury · " + x.replace(/\s+/g, " "), ""); });
    flush(); return L.join("\n");
  }
  function replaySh(rec) {
    const cmds = (rec.events || []).filter((e) => e.k === "cmd" && /^(mercury-agent|python3?|streamlit)\b/.test(String(e.x))).map((e) => String(e.x));
    return ["#!/usr/bin/env bash", "# Replay of recording: " + String(rec.title || "").replace(/\n/g, " "), "# Recorded " + (rec.startedAt || "") + " · " + cmds.length + " commands", "set -uo pipefail", ""].concat(cmds.map((c, i) => "echo '▸ [" + (i + 1) + "/" + cmds.length + "]'; " + c + " || echo '✗ step " + (i + 1) + " failed'")).join("\n") + "\n";
  }
  function capJson(c) {
    return { id: c.id, label: c.label, tag: c.tag, note: c.note || "", captured_at: c.at, range: [c.from, c.to], rows: c.n, flagged: c.flagged, first_flag: c.firstFlag, last_flag: c.lastFlag, columns: c.names, column_stats: c.colStats, score_stats: c.scoreStats, source: c.src, data_sha256_at_capture: c.sha256, data_bytes: c.bytes, data: { rows: c.rows, scores: c.scores, flags: c.flags } };
  }
  function capMarkdown(c) {
    const s = c.src || {}, L = ["### " + c.label, "", "- Tag: " + (TAGN[c.tag] || c.tag) + " · samples " + c.from + "–" + c.to + " (" + c.n + " rows)" + (c.flags ? " · " + c.flagged + " flagged" : ""), "- Source: " + (s.label || "—") + " · " + (s.route || "—") + (s.http != null ? " · HTTP " + s.http : "") + " · received " + utc(s.receivedAt) + (s.simulated ? " · SIMULATED engine" : ""), "- Dataset: " + (s.dataset || "—")];
    if (c.scoreStats) L.push("- Score: max " + f4(c.scoreStats.max) + " · mean " + f4(c.scoreStats.mean));
    (c.colStats || []).slice(0, 8).forEach((x) => { if (x.n) L.push("- " + x.name + ": mean " + f4(x.mean) + " ± " + f4(x.sd) + " [" + f4(x.min) + " – " + f4(x.max) + "]"); });
    L.push("- Data SHA-256: `" + c.sha256 + "`"); if (c.note) L.push("- Note: " + String(c.note).replace(/\n+/g, " ")); L.push("- Captured: " + utc(c.at));
    return L.join("\n") + "\n";
  }

  /* ---------- persistence ---------- */
  function load() { try { const p = JSON.parse(localStorage.getItem(KEY) || "null"); return p && typeof p === "object" ? { caps: Array.isArray(p.caps) ? p.caps : [], recs: Array.isArray(p.recs) ? p.recs : [] } : { caps: [], recs: [] }; } catch (e) { return { caps: [], recs: [] }; } }
  function save(C, force, over) {
    if (!C._capsLoaded) return false;
    const s = Object.assign({ caps: C.state.caps || [], recs: C.state.recs || [] }, over || {});
    if (!force && s.caps === C._capsSaved && s.recs === C._recsSaved) return null;
    const put = (caps, recs) => localStorage.setItem(KEY, JSON.stringify({ v: 1, caps, recs }));
    try { put(s.caps, s.recs); C._capsSaved = s.caps; C._recsSaved = s.recs; return true; }
    catch (e) { try { put(s.caps.slice(0, 40), s.recs.slice(0, 2).map((r) => Object.assign({}, r, { events: (r.events || []).slice(0, 3000) }))); C._capsSaved = s.caps; C._recsSaved = s.recs; return "trimmed"; } catch (e2) { return false; } }
  }

  /* ---------- actions ---------- */
  function totalFor(C, res) { const d = C.state.data; return d && res && res.ds && d.name === res.ds.name ? d.total || d.rows : null; }
  function captureRange(C, res, a, b, opts) {
    if (!res) return null; const o = opts || {}, src = source(res, totalFor(C, res)); if (src.n < 1) return null;
    const sl = slice(src, a, b), csv = toCsv(sl, (v) => C.csvSafe(v)), sim = !!((res.data && res.data._simulated) || /SIMULATED/.test(res.kind || ""));
    const cap = Object.assign({ id: "c" + Date.now().toString(36) + Math.random().toString(36).slice(2, 6), at: new Date().toISOString(), label: (o.label && String(o.label).trim()) || res.label + " · samples " + sl.from + "–" + sl.to, tag: TAGC[o.tag] ? o.tag : "review", note: o.note || "", sha256: sha256(csv), bytes: utf8(csv).length,
      src: { resultId: res.id, label: res.label, kind: res.kind || "", route: res.route || "", http: res.http == null ? null : res.http, ms: res.ms == null ? null : res.ms, receivedAt: res.at || null, dataset: res.ds ? res.ds.name : null, simulated: sim, n: src.n, total: src.total } }, sl);
    C.setState((p) => ({ caps: [cap].concat(p.caps || []).slice(0, 200), capSel: cap.id, capFlash: o.quiet ? p.capFlash : { rid: res.id, id: cap.id, text: "Saved · " + cap.label + " · " + cap.n + " rows" + (cap.flags ? " · " + cap.flagged + " flagged" : "") + " · SHA-256 " + cap.sha256.slice(0, 12) + "…" } }));
    if (C.recEv) C.recEv("capture", cap.label + " · " + cap.n + " rows" + (cap.flags ? " · " + cap.flagged + " flagged" : "") + " · sha256 " + cap.sha256.slice(0, 16));
    return cap;
  }
  function captureEvents(C, res) {
    const src = source(res, totalFor(C, res)), pad = C.state.capPad == null ? 2 : C.state.capPad, rs = runs(src.flags, pad, src.n), tag = C.state.capTag || "review", out = [];
    rs.forEach((r, i) => out.push(captureRange(C, res, r[0], r[1], { tag, label: res.label + " · event " + (i + 1) + "/" + rs.length + " · samples " + r[0] + "–" + r[1], quiet: true })));
    if (out.length) C.setState({ capFlash: { rid: res.id, id: out[0].id, text: "Saved " + out.length + " flagged event" + (out.length === 1 ? "" : "s") + " · ±" + pad + " samples each · tag " + (TAGN[tag] || tag) } });
    return out;
  }
  function update(C, id, patch) { C.setState((p) => ({ caps: (p.caps || []).map((x) => (x.id === id ? Object.assign({}, x, patch) : x)) })); }
  function updateRec(C, id, patch) { C.setState((p) => ({ recs: (p.recs || []).map((x) => (x.id === id ? Object.assign({}, x, patch) : x)) })); }
  function cmdWithLabels(C, path, g, prefix) {
    const c = C.cmdBy ? C.cmdBy(path) : null, opts = (c && c.opts) || [], inF = opts.find((o) => o.k === "--input") || opts.find((o) => o.k === "--data"), lab = opts.find((o) => o.k === "--labels");
    return "mercury-agent " + path + " " + (inF ? inF.k : "--input") + " " + shq((g && g.dataset) || "data.csv") + " " + (lab ? lab.k : "--labels") + " " + shq((prefix || "") + labelsName(g));
  }
  function recStart(C, title) {
    const s = C.state; if (s.rec) return s.rec; const now = new Date(), base = C.A && C.A.cfg ? C.A.cfg.baseUrl : "";
    const R = { id: "rec" + now.getTime().toString(36), title: (title && String(title).trim()) || "Session " + now.toISOString().slice(0, 16).replace("T", " ") + " UTC", t0: now.getTime(), startedAt: now.toISOString(), engine: C.live && C.live() ? base + (/^sim:/.test(base || "") ? " (simulated engine)" : "") : "browser preview (no engine)", dataset: s.data ? s.data.name + " · " + s.data.rows + " × " + s.data.cols : null, events: [{ t: 0, k: "mark", x: "recording started" }] };
    C.setState({ rec: R, recNote: "", recDiscardArm: false });
    C.append([C.row("● recording · " + R.title + " · commands, output, results, captures and your marks are kept until you stop", "var(--anomaly-soft)"), C.row("")]);
    return R;
  }
  function recStop(C, silent) {
    const R = C.state.rec; if (!R) return null; const t = Date.now() - R.t0;
    const done = Object.assign({}, R, { endedAt: new Date().toISOString(), dur: t, events: R.events.concat([{ t, k: "mark", x: silent ? "recording ended (page closed or reloaded)" : "recording stopped" }]) }), recs = [done].concat(C.state.recs || []).slice(0, 10);
    if (silent) { save(C, true, { recs }); return done; }
    C.setState({ rec: null, recs, recSel: done.id, recNote: "", recDiscardArm: false });
    C.append([Object.assign(C.row("■ recording saved · " + done.title + " · " + clock(done.dur) + " · " + done.events.length + " events · open Captures → Recordings to export", "var(--anomaly-soft)"), { act: () => C.setState({ ws: "caps", capView: "recs", recSel: done.id }) }), C.row("")]);
    return done;
  }
  function recDiscard(C, now) {
    if (!C.state.rec) return;
    if (!now && !C.state.recDiscardArm) { C.setState({ recDiscardArm: true }); clearTimeout(C._rda); C._rda = setTimeout(() => C.setState({ recDiscardArm: false }), 3000); return; }
    C.setState({ rec: null, recDiscardArm: false, recNote: "" }); C.append([C.row("○ recording discarded", "var(--fg-3)"), C.row("")]);
  }
  function recMark(C, text) {
    const R = C.state.rec; if (!R) return null; const x = String(text || "").trim() || "mark " + R.events.filter((e) => e.k === "mark").length;
    if (C.recEv) C.recEv("mark", x); C.setState({ recNote: "" }); return x;
  }
  function recCommand(C, raw) {
    const t = raw.trim().split(/\s+/), cmd = t[0].slice(1).toLowerCase(), rest = raw.trim().slice(t[0].length).trim(), W = "var(--warning)";
    C.append([C.row("$ " + raw, "var(--normal)")]);
    if (cmd === "mark") { if (!C.state.rec) { C.append([C.row("No recording in progress. Start one with :rec", W), C.row("")]); return; } recMark(C, rest); C.append([C.row("▸ marked +" + clock(Date.now() - C.state.rec.t0), "var(--fg-3)"), C.row("")]); return; }
    if (cmd === "capture") {
      const res = (C.state.results || [])[0], a = parseInt(t[1], 10), b = parseInt(t[2], 10), tg = (t[3] || "").toLowerCase(), hasTag = !!TAGC[tg], label = t.slice(hasTag ? 4 : isFinite(b) ? 3 : 2).join(" ");
      if (!res) { C.append([C.row("No result to capture from. Run a command or load a file first.", W), C.row("")]); return; }
      if (!isFinite(a)) { C.append([C.row("usage: :capture <from> [to] [anomaly|normal|review|evidence] [label…] · captures from the newest result", W), C.row("")]); return; }
      const cap = captureRange(C, res, a, isFinite(b) ? b : a, { tag: hasTag ? tg : C.state.capTag || "review", label: label || undefined });
      C.append([Object.assign(C.row("● captured " + cap.label + " · " + cap.n + " rows · sha256 " + cap.sha256.slice(0, 16) + "…", "var(--normal)"), { act: () => C.setState({ ws: "caps", capView: "caps", capSel: cap.id }) }), C.row("")]); return;
    }
    const sub = (t[1] || (C.state.rec ? "stop" : "start")).toLowerCase(), title = t.slice(2).join(" ");
    if (sub === "start") recStart(C, title); else if (sub === "stop") recStop(C, false); else if (sub === "discard") recDiscard(C, true);
    else C.append([C.row("usage: :rec [start [title] | stop | discard] · :mark <note> · :capture <from> [to] [tag] [label]", W), C.row("")]);
  }
  function pack(C) {
    const s = C.state, caps = (s.caps || []).slice().reverse(), recs = (s.recs || []).slice().reverse(); if (!caps.length && !recs.length) return null;
    const files = [], safe = (v) => C.csvSafe(v), exportedAt = new Date().toISOString();
    const capMeta = caps.map((c, i) => { const file = "captures/" + String(i + 1).padStart(3, "0") + "-" + slug(c.label) + ".csv", csv = toCsv(c, safe); files.push({ name: file, text: csv }); const j = capJson(c); delete j.data; return Object.assign(j, { file, data_sha256_now: sha256(csv) }); });
    const groups = labelGroups(s.caps || []).filter((g) => g.y.some((v) => v === 1));
    const labMeta = groups.map((g) => { const file = "labels/" + labelsName(g); files.push({ name: file, text: labelsCsv(g.y) }); return { file, dataset: g.dataset, samples: g.total, anomalous: g.y.filter((v) => v === 1).length, from_captures: g.ids }; });
    const recMeta = recs.map((r, i) => { const base = "recordings/" + String(i + 1).padStart(3, "0") + "-" + slug(r.title); files.push({ name: base + ".cast", text: cast(r) }, { name: base + ".md", text: markdown(r) }); return { title: r.title, started_at: r.startedAt, ended_at: r.endedAt, duration_ms: r.dur, events: (r.events || []).length, files: [base + ".cast", base + ".md"] }; });
    const engine = C.A && C.A.cfg ? C.A.cfg.baseUrl || "no engine (browser preview)" : "—";
    files.push({ name: "captures.json", text: JSON.stringify({ format: "mercury-cli.evidence/1", exported_at: exportedAt, engine, captures: capMeta, labels: labMeta, recordings: recMeta }, null, 2) + "\n" });
    const use = groups.length ? ["", "Use the labels (1 = inside an ANOMALY capture, 0 = everything else)", "  " + cmdWithLabels(C, "tier-detect", groups[0], "labels/"), "  " + cmdWithLabels(C, "train", groups[0], "labels/"), "  The label count must equal the dataset's row count (cli.py checks this)."] : [];
    files.push({ name: "README.txt", text: ["Mercury CLI · evidence pack", "Exported " + utc(exportedAt) + " · engine " + engine, "", "Contents", "  captures/NNN-*.csv    one file per captured section: index, columns, score, is_anomaly", "  captures.json         label, tag, note, statistics, source result, route and time for each capture", "  labels/*.labels.csv   0/1 label vectors for tier-detect -l, train -l, tune -l and build-features -l", "  recordings/*.cast     session recordings in asciicast v2 (asciinema play file.cast)", "  recordings/*.md       the same sessions as a Markdown lab notebook", "  SHA256SUMS            SHA-256 of every file above", "", "Verify", "  sha256sum -c SHA256SUMS            (macOS: shasum -a 256 -c SHA256SUMS)", "  data_sha256_at_capture in captures.json is the SHA-256 of each CSV when it was captured.", "  It matches SHA256SUMS while the data is unchanged."].concat(use, ["", "Captures marked simulated came from the in-browser simulated engine, not Mercury's models."]).join("\n") + "\n" });
    files.push({ name: "SHA256SUMS", text: files.map((f) => sha256(f.text) + "  " + f.name).join("\n") + "\n" });
    const u8 = zip(files); C.download("mercury-evidence-" + exportedAt.slice(0, 19).replace(/[:T]/g, "-") + ".zip", "application/zip", u8);
    if (C.recEv) C.recEv("mark", "exported evidence pack · " + files.length + " files");
    return { files: files.map((f) => f.name), bytes: u8.length };
  }

  window.MercuryCLICapture = { KEY, TAGS, stats, source, slice, runs, labels, labelGroups, labelsCsv, labelsName, toCsv, sha256, crc32, zip, unzip, cast, markdown, replaySh, capJson, capMarkdown, counts, countText, clock, load, save,
    captureRange, captureEvents, recStart, recStop, recDiscard, recMark, recCommand, pack, cmdWithLabels };
})();

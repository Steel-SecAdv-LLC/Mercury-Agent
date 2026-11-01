/* Copyright (C) 2025 Steel Security Advisors LLC
   SPDX-License-Identifier: AGPL-3.0-or-later
   Mercury Console HTTP client. Ported from docs/design/mercury_console/design/mercury-cli-api.js with the changes the
   page's Content-Security-Policy requires: same-origin only (connect-src 'self'), so there is no base-URL field and
   no simulated engine; request schemas come from the generated catalog (app.openapi() and the MCP tool specs) rather
   than a hand-written table. The optional X-API-Key header is kept. */
(function () {
  "use strict";
  if (window.MercuryCLIAPI) return;
  const CAT = window.MERCURY_CLI || {};
  const LS = "mercury_cli_conn_v1";
  const cfg = Object.assign({ apiKey: "", remember: false, timeoutMs: 15000 }, read());
  delete cfg.baseUrl; // the console only talks to the origin that served it

  // X-RateLimit-* on every reply and Retry-After on 429 (exposed through CORS by api/server.py).
  const budget = { limit: null, remaining: null, resetAt: null, retryUntil: null, at: null, calls: 0, limited: 0 };
  function noteMeta(status, meta) {
    if (!meta) return;
    budget.calls++; budget.at = Date.now();
    const rl = meta.rateLimit || {};
    if (rl.limit != null && isFinite(+rl.limit)) budget.limit = +rl.limit;
    if (rl.remaining != null && isFinite(+rl.remaining)) budget.remaining = +rl.remaining;
    if (rl.reset != null && isFinite(+rl.reset)) budget.resetAt = +rl.reset > 1e9 ? +rl.reset * 1000 : Date.now() + +rl.reset * 1000;
    if (status === 429) {
      budget.limited++;
      const ra = meta.retryAfter != null ? parseFloat(meta.retryAfter) : NaN;
      budget.retryUntil = Date.now() + (isNaN(ra) ? 1000 : Math.max(1, ra) * 1000);
    } else if (status < 400 && budget.retryUntil && Date.now() >= budget.retryUntil) budget.retryUntil = null;
  }
  const PROBE_TIMEOUT = 2500;
  function read() {
    let o = {};
    try { o = JSON.parse(localStorage.getItem(LS) || "{}") || {}; } catch (e) { o = {}; }
    try { const k = sessionStorage.getItem(LS + "_key"); if (k && !o.apiKey) o.apiKey = k; } catch (e) { /* storage blocked */ }
    return o;
  }
  function save() {
    try {
      localStorage.setItem(LS, JSON.stringify({ apiKey: cfg.remember ? cfg.apiKey : "", remember: !!cfg.remember, timeoutMs: cfg.timeoutMs }));
      sessionStorage.setItem(LS + "_key", cfg.apiKey || "");
    } catch (e) { /* storage blocked: the key lives for this page only */ }
  }
  const headers = (json) => { const h = { Accept: "application/json" }; if (json) h["Content-Type"] = "application/json"; if (cfg.apiKey) h["X-API-Key"] = cfg.apiKey; return h; };
  function announce(detail) { try { window.dispatchEvent(new CustomEvent("mcli:call", { detail })); } catch (e) { /* no window */ } }

  async function call(method, path, body, opts) {
    const max = opts && opts.retries != null ? opts.retries : 2;
    let r, n = 0;
    announce({ phase: "start", method, path });
    for (let attempt = 0; attempt <= max; attempt++) {
      r = await call1(method, path, body, opts); n = attempt + 1;
      if (r && r.meta) noteMeta(r.status, r.meta);
      if (r.ok || !(r.status === 429 || r.status === 503) || attempt === max || (opts && opts.signal && opts.signal.aborted)) break;
      const ra = r.meta && r.meta.retryAfter != null ? parseFloat(r.meta.retryAfter) : NaN;
      await new Promise((ok) => setTimeout(ok, Math.min(10000, isNaN(ra) ? 500 * Math.pow(2, attempt) : ra * 1000)));
    }
    if (r && r.meta) r.meta.attempts = n;
    announce({ phase: "end", method, path, ok: !!(r && r.ok), status: r && r.status, data: r && r.data });
    return r;
  }
  async function call1(method, path, body, opts) {
    if (typeof path !== "string" || path.charAt(0) !== "/" || path.charAt(1) === "/") return { ok: false, status: 0, kind: "config", error: "Only same-origin paths are allowed" };
    const ctrl = new AbortController();
    const to = setTimeout(() => ctrl.abort(), path === "/health" ? PROBE_TIMEOUT : cfg.timeoutMs);
    const t0 = performance.now();
    let userAbort = false;
    if (opts && opts.signal) {
      if (opts.signal.aborted) { clearTimeout(to); return { ok: false, status: 0, kind: "cancelled", error: "Cancelled" }; }
      opts.signal.addEventListener("abort", () => { userAbort = true; ctrl.abort(); }, { once: true });
    }
    try {
      const res = await fetch(path, { method, headers: headers(body != null), body: body != null ? JSON.stringify(body) : undefined, signal: ctrl.signal, credentials: "same-origin" });
      let data = null;
      const ct = res.headers.get("content-type") || "";
      if (/^image\//.test(ct)) {
        const blob = await res.blob();
        data = await new Promise((ok) => { const fr = new FileReader(); fr.onload = () => ok({ image: fr.result, content_type: ct, bytes: blob.size }); fr.readAsDataURL(blob); });
      } else {
        const text = await res.text();
        try { data = text ? JSON.parse(text) : null; } catch (e) { data = text; }
      }
      clearTimeout(to);
      const meta = { ms: Math.round(performance.now() - t0), correlationId: res.headers.get("X-Correlation-ID"), rateLimit: { limit: res.headers.get("X-RateLimit-Limit"), remaining: res.headers.get("X-RateLimit-Remaining"), reset: res.headers.get("X-RateLimit-Reset") }, retryAfter: res.headers.get("Retry-After") };
      if (!res.ok) {
        const detail = data && typeof data === "object" ? data.detail || data.message || data.error : null;
        const msg = Array.isArray(detail) ? detail.map((d) => (d && d.loc ? d.loc.slice(1).join(".") + ": " : "") + ((d && d.msg) || String(d))).join("; ") : detail;
        return { ok: false, status: res.status, kind: res.status === 401 || res.status === 403 ? "auth" : res.status === 429 ? "ratelimit" : res.status === 422 ? "validation" : "http", error: msg || res.statusText || "HTTP " + res.status, data, meta };
      }
      return { ok: true, status: res.status, data, meta };
    } catch (e) {
      clearTimeout(to);
      if (userAbort) return { ok: false, status: 0, kind: "cancelled", error: "Cancelled" };
      return { ok: false, status: 0, kind: e.name === "AbortError" ? "timeout" : "network", error: e.name === "AbortError" ? "Timed out after " + (path === "/health" ? PROBE_TIMEOUT : cfg.timeoutMs) + " ms" : "The server did not answer" };
    }
  }

  /* CAP 1.2 mappings: alerting/cap_generator.py */
  const CAP = {
    severity: (s) => (s >= 0.9 ? "Extreme" : s >= 0.7 ? "Severe" : s >= 0.5 ? "Moderate" : s >= 0.3 ? "Minor" : "Unknown"),
    certainty: (s) => (s >= 0.9 ? "Observed" : s >= 0.7 ? "Likely" : s >= 0.5 ? "Possible" : "Unlikely"),
    urgency: (s) => (s >= 0.9 ? "Immediate" : s >= 0.7 ? "Expected" : s >= 0.5 ? "Future" : "Unknown"),
  };
  /* api/server.py _classify_severity(score, threshold): ratio < 1.5 low, < 2.5 medium, < 4.0 high, else critical */
  const SEVERITY = (score, thr) => { const r = thr > 0 ? score / thr : score; return r >= 4 ? "critical" : r >= 2.5 ? "high" : r >= 1.5 ? "medium" : "low"; };

  /* Request schemas (field -> {type, def, min, max, enum, req, desc, p, q, raw}), generated. */
  const SCHEMAS = CAT.schemas || {};
  const MCP = CAT.mcpSchemas || {};
  /* CLI command -> the API route that does the same work when the console runs it. */
  const CLI_TO_ROUTE = {
    detect: (vals) => (vals["--detector"] === "fusion" ? "POST /api/v1/detect/fusion" : "POST /api/v1/detect/multivariate"),
    "tier-detect": () => "POST /api/v1/detect/tier",
    rca: () => "POST /api/v1/detect/rca",
    "hazard-viz": () => "POST /api/v1/hazard/visualize",
    "physics spectral": () => "POST /api/v1/detect/three-r",
    "physics dynamics": () => "POST /api/v1/detect/univariate",
    "physics uiux": () => "POST /api/v1/detect/multivariate",
    "physics integrated": () => "POST /api/v1/detect/fusion",
  };
  const STR = { message: "status", email: "you@example.com", name: "my-key", query: "anomaly detection on power-grid telemetry", question: "What does the ethics gate check?", text: "Grid load spiked to 3x baseline at 03:00 UTC.", action: "share an anonymised anomaly report with the grid operator", title: "Incident summary", subject_pseudonym: "subject-001", hazard: "earthquake", stream: "default", subject_id: "subject-001", domain: "infrastructure" };
  function sample(field, key) {
    if (field.def !== undefined) return field.def;
    if (field.type === "string" && key && STR[key] !== undefined) return STR[key];
    switch (field.type) {
      case "number[]": return [0.4, 0.5, 0.45, 10.0, 0.5];
      case "number[][]": return [[0.3, 0.2], [0.4, 0.2], [4.0, 0.2], [0.3, 0.2]];
      case "integer[]": return [0, 0, 1, 0];
      case "string[]": return ["f0", "f1"];
      case "object[]": return [{}];
      case "object": return {};
      case "boolean": return false;
      case "integer": return field.min != null ? field.min : 1;
      case "number": return field.min != null && field.max != null ? (field.min + field.max) / 2 : 0.5;
      case "enum": return field.enum[0];
      default: return key ? "<" + key + ">" : "";
    }
  }
  function body(schema, vals, opts) {
    const rk = Object.keys(schema).find((k) => schema[k].raw);
    if (rk) { const v = vals && vals[rk]; return v && typeof v === "object" ? v : {}; }
    const o = {}, only = opts && opts.requiredOnly;
    Object.keys(schema).forEach((k) => {
      const f = schema[k]; if (f.p || f.q) return;
      const v = vals && vals[k];
      if (v !== undefined && v !== "" && v !== null) o[k] = v;
      else if (f.req || (!only && f.def !== undefined)) o[k] = sample(f, k);
    });
    return o;
  }
  function missing(schema, b, vals) {
    if (Object.keys(schema).some((k) => schema[k].raw)) return b && Object.keys(b).length ? [] : Object.keys(schema).filter((k) => schema[k].raw);
    return Object.keys(schema).filter((k) => {
      const f = schema[k]; if (!f.req) return false;
      const v = f.p || f.q ? (vals || {})[k] : b[k];
      return v === undefined || v === "" || (Array.isArray(v) && !v.length);
    });
  }
  const pathParams = (route) => [...route.matchAll(/\{(\w+)\}/g)].map((m) => m[1]);
  function withParams(route, schema) { const sc = Object.assign({}, schema || {}); pathParams(route).forEach((k) => { if (!sc[k]) sc[k] = { type: "string", req: 1, p: 1, desc: "path parameter" }; }); return sc; }
  function url(route, vals, schema) {
    const path = route.split(" ")[1];
    const u = path.replace(/\{(\w+)\}/g, (m, k) => (vals && vals[k] != null && vals[k] !== "" ? encodeURIComponent(vals[k]) : "{" + k + "}"));
    const qs = Object.keys(schema || {}).filter((k) => schema[k].q && vals && vals[k] != null && vals[k] !== "").map((k) => encodeURIComponent(k) + "=" + encodeURIComponent(vals[k]));
    return u + (qs.length ? "?" + qs.join("&") : "");
  }
  const origin = () => (typeof location !== "undefined" && location.origin && location.origin !== "null" ? location.origin : "http://localhost:8000");

  window.MercuryCLIAPI = {
    get cfg() { return Object.assign({ baseUrl: origin() }, cfg); },
    budget: () => Object.assign({}, budget),
    set(o) { ["apiKey", "remember", "timeoutMs"].forEach((k) => { if (o && o[k] !== undefined) cfg[k] = o[k]; }); save(); },
    connected: () => true,
    simulated: () => false,
    health: () => call("GET", "/health"),
    openapi: () => call("GET", "/openapi.json"),
    post: (path, b, o) => call("POST", path, b, o),
    get: (path, o) => call("GET", path, null, o),
    put: (path, b, o) => call("PUT", path, b, o),
    del: (path, o) => call("DELETE", path, null, o),
    speak: (message, domain, sessionId) => call("POST", "/api/v1/voice/speak", { message, domain: domain || null, session_id: sessionId || null }),
    narrate: (detection, domain, verbosity) => call("POST", "/api/v1/voice/narrate", { detection_result: detection, domain: domain || null, verbosity: verbosity || "normal" }),
    greet: (domain) => call("GET", "/api/v1/voice/greet" + (domain ? "?domain=" + encodeURIComponent(domain) : "")),
    voiceStatus: () => call("GET", "/api/v1/voice/status"),
    voiceHistory: (limit, sessionId) => call("GET", "/api/v1/voice/history?limit=" + Math.max(1, Math.min(200, +limit || 20)) + (sessionId ? "&session_id=" + encodeURIComponent(sessionId) : "")),
    alertsUrl: () => null, // api/voice.py has no WebSocket route
    batchSubmit: (b) => call("POST", "/api/v1/batch/detect", b),
    batchStatus: (id) => call("GET", "/api/v1/batch/jobs/" + encodeURIComponent(id)),
    batchResults: (id, offset, limit) => call("GET", "/api/v1/batch/jobs/" + encodeURIComponent(id) + "/results" + (offset != null ? "?offset=" + offset + "&limit=" + (limit || 1000) : "")),
    batchCancel: (id) => call("DELETE", "/api/v1/batch/jobs/" + encodeURIComponent(id)),
    SCHEMAS, MCP, CLI_TO_ROUTE, CAP, SEVERITY, sample, body, missing, pathParams, withParams, url,
    curl(route, b, vals, schema) {
      const m = route.split(" ")[0], p = url(route, vals, schema);
      const h = cfg.apiKey ? " \\\n  -H 'X-API-Key: $MERCURY_API_KEY'" : "";
      return "curl -X " + m + " " + origin() + p + h + (b ? " \\\n  -H 'Content-Type: application/json' \\\n  -d '" + JSON.stringify(b).replace(/'/g, "'\\''") + "'" : "");
    },
    mcpCall: (name, args) => JSON.stringify({ jsonrpc: "2.0", id: 1, method: "tools/call", params: { name, arguments: args } }, null, 2),
  };
})();

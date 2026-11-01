/* Mercury CLI — typed HTTP client + request schemas for mercury-agent serve.
   Grounded in main (synced 2026-10-29; interface files identical to steel/version-cov7): api/server.py, api/routes/{detection,batch,voice,hazard}.py,
   api/health.py, mcp_server.py, alerting/cap_generator.py. Plain JS, no build step. */
(function () {
  // The dc runtime re-mounts every page <script>, so this file executes twice. The client owns the
  // connection state (cfg): a second instance would leave the console and window.MercuryCLIAPI holding
  // different configs (mass-run gates read the global; the console binds this.A once at init).
  if (window.MercuryCLIAPI) return;
  const LS = "mercury_cli_conn_v1";
  const cfg = Object.assign({ baseUrl: "", apiKey: "", remember: false, timeoutMs: 12000 }, read());
  // The engine answers every call with X-RateLimit-Limit/Remaining/Reset and, on 429, Retry-After. Keep the latest so the
  // console can show the budget and a countdown instead of a silent pause (the budget is per server, not per route).
  const budget = { limit: null, remaining: null, resetAt: null, retryUntil: null, at: null, calls: 0, limited: 0 };
  function noteMeta(status, meta) { if (!meta) return; budget.calls++; budget.at = Date.now(); const rl = meta.rateLimit || {}; if (rl.limit != null && isFinite(+rl.limit)) budget.limit = +rl.limit; if (rl.remaining != null && isFinite(+rl.remaining)) budget.remaining = +rl.remaining; if (rl.reset != null && isFinite(+rl.reset)) budget.resetAt = +rl.reset > 1e9 ? +rl.reset * 1000 : Date.now() + +rl.reset * 1000; if (status === 429) { budget.limited++; const ra = meta.retryAfter != null ? parseFloat(meta.retryAfter) : NaN; budget.retryUntil = Date.now() + (isNaN(ra) ? 1000 : Math.max(1, ra) * 1000); } else if (status < 400 && budget.retryUntil && Date.now() >= budget.retryUntil) budget.retryUntil = null; }
  const PROBE_TIMEOUT = 2500;
  function read() { let o = {}; try { o = JSON.parse(localStorage.getItem(LS) || "{}"); } catch (e) {} try { const k = sessionStorage.getItem(LS + "_key"); if (k && !o.apiKey) o.apiKey = k; } catch (e) {} return o; }
  function save() { try { localStorage.setItem(LS, JSON.stringify({ baseUrl: cfg.baseUrl, apiKey: cfg.remember ? cfg.apiKey : "", remember: !!cfg.remember, timeoutMs: cfg.timeoutMs })); sessionStorage.setItem(LS + "_key", cfg.apiKey || ""); } catch (e) {} }
  const base = () => cfg.baseUrl.replace(/\/+$/, "");
  const headers = (json) => { const h = {}; if (json) h["Content-Type"] = "application/json"; if (cfg.apiKey) h["X-API-Key"] = cfg.apiKey; return h; };

  // The agent figure reads these: activity from start and end, the signal and the ripple from the reply.
  function announce(detail) { try { window.dispatchEvent(new CustomEvent("mcli:call", { detail: detail })); } catch (e) { /* no window in a non-browser host */ } }
  async function call(method, path, body, opts) {
    const max = (opts && opts.retries != null) ? opts.retries : 2; let r, n = 0;
    announce({ phase: "start", method: method, path: path });
    for (let attempt = 0; attempt <= max; attempt++) {
      r = await call1(method, path, body, opts); n = attempt + 1; if (r && r.meta) noteMeta(r.status, r.meta);
      if (r.ok || !(r.status === 429 || r.status === 503) || attempt === max || (opts && opts.signal && opts.signal.aborted)) break;
      const ra = r.meta && r.meta.retryAfter != null ? parseFloat(r.meta.retryAfter) : NaN; const wait = Math.min(10000, isNaN(ra) ? 500 * Math.pow(2, attempt) : ra * 1000);
      await new Promise((ok) => setTimeout(ok, wait));
    }
    if (r && r.meta) r.meta.attempts = n;
    announce({ phase: "end", method: method, path: path, ok: !!(r && r.ok), status: r && r.status, data: r && r.data });
    return r;
  }
  async function call1(method, path, body, opts) {
    if (!cfg.baseUrl) return { ok: false, status: 0, kind: "offline", error: "No server configured" };
    if (!/^(https?:\/\/|sim:\/\/)/i.test(cfg.baseUrl)) return { ok: false, status: 0, kind: "config", error: "Base URL must start with http://, https:// (or sim:// for the simulated engine)" };
    if (cfg.baseUrl.startsWith("sim://") && window.MercuryCLISim) { if (opts && opts.signal && opts.signal.aborted) return { ok: false, status: 0, kind: "cancelled", error: "Cancelled" }; const s0 = performance.now(); const r = await window.MercuryCLISim.handle(method, path, body, headers(body != null)); const meta = { ms: Math.round(performance.now() - s0), correlationId: "sim-" + Math.random().toString(36).slice(2, 10), simulated: true, rateLimit: {}, retryAfter: r.retryAfter != null ? String(r.retryAfter) : null }; const data = r.image ? { image: r.image, content_type: "image/png", _simulated: true } : r.body; if (r.status >= 400) return { ok: false, status: r.status, kind: r.status === 401 || r.status === 403 ? "auth" : r.status === 429 ? "ratelimit" : r.status === 422 ? "validation" : "http", error: (data && (data.message || data.error)) || "error", data, meta }; return { ok: true, status: r.status, data, meta }; }
    const ctrl = new AbortController(); const to = setTimeout(() => ctrl.abort(), path === "/health" ? PROBE_TIMEOUT : cfg.timeoutMs); const t0 = performance.now(); let userAbort = false; if (opts && opts.signal) { if (opts.signal.aborted) return { ok: false, status: 0, kind: "cancelled", error: "Cancelled" }; opts.signal.addEventListener("abort", () => { userAbort = true; ctrl.abort(); }, { once: true }); }
    try {
      const res = await fetch(base() + path, { method, headers: headers(body != null), body: body != null ? JSON.stringify(body) : undefined, signal: ctrl.signal });
      clearTimeout(to);
      let data = null; const ct = res.headers.get("content-type") || ""; if (/^image\//.test(ct)) { const blob = await res.blob(); data = await new Promise((ok) => { const fr = new FileReader(); fr.onload = () => ok({ image: fr.result, content_type: ct, bytes: blob.size }); fr.readAsDataURL(blob); }); } else { const text = await res.text(); try { data = text ? JSON.parse(text) : null; } catch (e) { data = text; } }
      const meta = { ms: Math.round(performance.now() - t0), correlationId: res.headers.get("X-Correlation-ID"), rateLimit: { limit: res.headers.get("X-RateLimit-Limit"), remaining: res.headers.get("X-RateLimit-Remaining"), reset: res.headers.get("X-RateLimit-Reset") }, retryAfter: res.headers.get("Retry-After") };
      if (!res.ok) return { ok: false, status: res.status, kind: res.status === 401 || res.status === 403 ? "auth" : res.status === 429 ? "ratelimit" : res.status === 422 ? "validation" : "http", error: (data && (data.detail || data.message)) || res.statusText, data, meta };
      return { ok: true, status: res.status, data, meta };
    } catch (e) {
      clearTimeout(to);
      if (userAbort) return { ok: false, status: 0, kind: "cancelled", error: "Cancelled" };
      return { ok: false, status: 0, kind: e.name === "AbortError" ? "timeout" : "network", error: e.name === "AbortError" ? "Timed out after " + cfg.timeoutMs + " ms" : "Unreachable or blocked by CORS (set MERCURY_CORS_ORIGINS on the server)" };
    }
  }

  /* CAP 1.2 mappings — alerting/cap_generator.py */
  const CAP = {
    severity: (s) => (s >= 0.9 ? "Extreme" : s >= 0.7 ? "Severe" : s >= 0.5 ? "Moderate" : s >= 0.3 ? "Minor" : "Unknown"),
    certainty: (s) => (s >= 0.9 ? "Observed" : s >= 0.7 ? "Likely" : s >= 0.5 ? "Possible" : "Unlikely"),
    urgency: (s) => (s >= 0.9 ? "Immediate" : s >= 0.7 ? "Expected" : s >= 0.5 ? "Future" : "Unknown"),
    category: { earthquake: "Geo", tsunami: "Geo", hurricane: "Met", tornado: "Met", flood: "Met", wildfire: "Fire", volcanic: "Geo", landslide: "Geo", sepsis: "Health", pandemic: "Health", financial: "Other", energy: "Infra", marine: "Env", network_security: "Security", fema: "Other" },
    event: { earthquake: "Earthquake", tsunami: "Tsunami", hurricane: "Hurricane", tornado: "Tornado", flood: "Flood", wildfire: "Wildfire", volcanic: "Volcanic Eruption", landslide: "Landslide", sepsis: "Public Health Emergency", pandemic: "Pandemic", financial: "Financial Crisis", energy: "Power Grid Emergency", marine: "Marine Environmental Emergency", network_security: "Cyber Security Incident", fema: "Disaster Declaration" },
  };
  /* api/server.py SeverityLevel via _classify_severity(score, threshold) ratio bands */
  const SEVERITY = (score, thr) => { const r = thr > 0 ? score / thr : score; return r >= 3 ? "critical" : r >= 2 ? "high" : r >= 1.5 ? "medium" : "low"; };

  /* Request schemas: field → {type, def, min, max, enum, req, desc}. Used for forms, curl bodies and MCP payloads. */
  const F = (type, o) => Object.assign({ type }, o || {});
  const SCHEMAS = {
    "POST /api/v1/detect/univariate": { data: F("number[]", { req: 1, desc: "Time series, ≥ 3 points" }), sensitivity: F("number", { def: 0.5, min: 0, max: 1 }), method: F("enum", { def: "zscore", enum: ["zscore", "iqr"] }) },
    "POST /api/v1/detect/multivariate": { data: F("number[][]", { req: 1, desc: "[n_samples, n_features]" }), features: F("string[]"), sensitivity: F("number", { def: 0.5, min: 0, max: 1 }), correlate_features: F("boolean", { def: true }) },
    "POST /api/v1/detect/neurosymbolic": { data: F("object[]", { req: 1, desc: "List of records" }), sensitivity: F("number", { def: 0.5, min: 0, max: 1 }), fusion_strategy: F("enum", { def: "confidence_weighted", enum: ["weighted_average", "attention", "gated", "confidence_weighted"] }), include_explanations: F("boolean", { def: true }) },
    "POST /api/v1/detect/fusion": { data: F("number[]|number[][]", { req: 1 }), detectors: F("string[]", { def: ["statistical", "temporal", "dimensional"] }), sensitivity: F("number", { def: 0.5, min: 0, max: 1 }), return_contributions: F("boolean", { def: true }) },
    "POST /api/v1/detect/three-r": { data: F("number[]", { req: 1 }), recursion_depth: F("integer", { def: 5, min: 1, max: 10 }), harmonic_bands: F("integer", { def: 8, min: 1, max: 32 }), ethical_threshold: F("number", { def: 0.96, min: 0.9, max: 0.99, desc: "Score-scaling base, not an ethics control" }) },
    "POST /api/v1/detect/tier": { data: F("number[]", { req: 1, desc: "1-D series" }), labels: F("integer[]"), subset: F("string[]"), method: F("enum", { enum: ["stacking", "bma", "average", "consensus"] }), contamination: F("number", { def: 0.05, min: 0, max: 0.5 }), conformal_alpha: F("number", { min: 0, max: 1 }), include_counterfactual: F("boolean", { def: false }), counterfactual_index: F("integer"), counterfactual_method: F("enum", { def: "prototype", enum: ["wachter", "dice", "growing_spheres", "prototype", "genetic"] }), include_attribution: F("boolean", { def: false }) },
    "POST /api/v1/detect/flagship": { data: F("number[][]", { req: 1, desc: "Feature matrix" }), domain: F("string", { desc: "GOSNN threshold tuning, e.g. medical" }), explain: F("boolean", { def: false }), gdpr_report: F("boolean", { def: false }), subject_id: F("string") },
    "POST /api/v1/detect/rca": { observations: F("number[][]", { req: 1, desc: "Rows × nodes; last row is the anomaly" }), adjacency: F("number[][]"), train: F("number[][]"), top_k: F("integer", { min: 1 }), node_names: F("string[]") },
    "POST /api/v1/batch/detect": { data: F("number[][]", { req: 1 }), method: F("enum", { def: "univariate", enum: ["univariate", "multivariate", "fusion", "statistical", "temporal", "neurosymbolic"] }), sensitivity: F("number", { def: 0.5, min: 0, max: 1 }), feature_names: F("string[]"), chunk_size: F("integer", { def: 1000, min: 1, max: 10000 }), priority: F("integer", { def: 5, min: 1, max: 10 }), callback_url: F("string"), metadata: F("object") },
    "POST /api/v1/hazard/visualize": { hazard: F("enum", { req: 1, enum: ["earthquake", "tsunami", "meteor", "wildfire", "tornado", "hurricane", "volcanic", "landslide", "schumann"] }), format: F("enum", { def: "png", enum: ["png", "geojson"] }), diagnostics: F("object", { desc: "HazardDiagnostics payload" }), arrays: F("object"), params: F("object"), geotransform: F("object") },
    "POST /api/v1/voice/speak": { message: F("string", { req: 1 }), domain: F("string"), session_id: F("string") },
    "POST /api/v1/voice/narrate": { detection_result: F("object", { req: 1 }), domain: F("string"), verbosity: F("enum", { def: "normal", enum: ["brief", "normal", "detailed"] }) },
    "POST /api/v1/auth/login": { email: F("string", { req: 1, max: 320 }), password: F("string", { req: 1 }), totp_code: F("string", { desc: "6-digit TOTP when 2FA is on" }), recovery_code: F("string"), remember_me: F("boolean", { def: true }) },
    "POST /api/v1/auth/register": { email: F("string", { req: 1 }), password: F("string", { req: 1, desc: "≥ 8 chars" }) },
    "POST /api/v1/auth/api-keys": { name: F("string", { req: 1 }), expires_in_days: F("integer", { min: 1, max: 3650 }), permissions: F("string[]") },
    "POST /api/v1/models": { name: F("string", { req: 1, desc: "1–100 chars" }), model_type: F("enum", { def: "fusion", enum: ["fusion", "statistical", "temporal", "neurosymbolic", "lstm_autoencoder", "isolation_forest", "custom"] }), description: F("string", { desc: "≤ 500 chars" }), tags: F("string[]", { desc: "≤ 10 tags" }), metadata: F("object") },
    "POST /api/v1/models/{model_id}/deploy": { version_number: F("string", { req: 1 }), replicas: F("integer", { def: 1, min: 1, max: 10 }), resources: F("object", { desc: "cpu, memory, gpu" }), canary_percentage: F("integer", { def: 0, min: 0, max: 100 }) },
    "POST /api/v1/models/{model_id}/versions": { framework: F("enum", { def: "pytorch", enum: ["pytorch", "onnx", "tensorflow", "sklearn", "custom"] }), description: F("string", { desc: "≤ 500 chars" }), config: F("object"), set_as_default: F("boolean", { def: false }) },
    "PUT /api/v1/models/{model_id}/versions/{version_number}/default": {},
    "PUT /api/v1/models/{model_id}/versions/{version_number}/metrics": { accuracy: F("number", { min: 0, max: 1 }), precision: F("number", { min: 0, max: 1 }), recall: F("number", { min: 0, max: 1 }), f1_score: F("number", { min: 0, max: 1 }), auc_roc: F("number", { min: 0, max: 1 }), latency_p50_ms: F("number", { min: 0 }), latency_p99_ms: F("number", { min: 0 }), throughput_rps: F("number", { min: 0 }) },
    "PUT /api/v1/models/{model_id}/status": { status: F("enum", { req: 1, q: 1, enum: ["draft", "staged", "deployed", "deprecated", "archived"], desc: "query parameter ?status=" }) },
    "POST /api/v1/assessment/adjudication/cases": { body: F("object", { req: 1, raw: 1, desc: "Free-form JSON; the assessment service validates it. Its split is fixed by its id." }) },
    "POST /api/v1/assessment/adjudication/outcomes": { body: F("object", { req: 1, raw: 1, desc: "Free-form JSON: what happened in a case" }) },
    "POST /api/v1/assessment/adjudication/ratings": { body: F("object", { req: 1, raw: 1, desc: "Blind rating; recorded under your principal id and the server clock" }) },
    "POST /api/v1/assessment/adjudication/decisions": { body: F("object", { req: 1, raw: 1, desc: "Needs the adjudicator or admin role" }) },
    "POST /api/v1/assessment/adjudication/export": { split: F("enum", { def: "heldout", enum: ["heldout", "training"] }) },
    "POST /api/v1/auth/verify-email": { token: F("string", { req: 1 }) },
    "POST /api/v1/auth/resend-verification": { email: F("string", { req: 1 }) },
    "POST /api/v1/auth/logout": {},
    "POST /api/v1/auth/password-reset/request": { email: F("string", { req: 1 }) },
    "POST /api/v1/auth/password-reset/confirm": { token: F("string", { req: 1 }), new_password: F("string", { req: 1 }) },
    "POST /api/v1/auth/password/change": { current_password: F("string", { req: 1 }), new_password: F("string", { req: 1 }) },
    "POST /api/v1/auth/email-change/request": { new_email: F("string", { req: 1 }), current_password: F("string", { req: 1 }) },
    "POST /api/v1/auth/email-change/confirm": { token: F("string", { req: 1 }) },
    "POST /api/v1/auth/account/delete": { current_password: F("string", { req: 1 }) },
    "POST /api/v1/auth/2fa/enroll": {},
    "POST /api/v1/auth/2fa/confirm": { code: F("string", { req: 1, desc: "6-digit code from your authenticator" }) },
    "POST /api/v1/auth/2fa/disable": {},
    "POST /api/v1/auth/2fa/recovery-codes": { current_password: F("string", { req: 1 }) },
    "POST /api/v1/assessment/statements/verify": { statements: F("object[]", { req: 1 }), evidence: F("object[]"), config: F("object") },
    "POST /api/v1/assessment/intel/screen": { intel_reports: F("object", { req: 1 }), history: F("object") },
    "POST /api/v1/assessment/intel/fuse": { intel_reports: F("object", { req: 1 }), temporal_context: F("object[]"), history: F("object") },
    "POST /api/v1/assessment/concealed-information-test": { subject_pseudonym: F("string", { req: 1 }), administered_at: F("number", { req: 1 }), purpose: F("string", { req: 1 }), consent: F("object", { req: 1 }), questions: F("object[]", { req: 1, desc: "question_id, probe, irrelevants, responses" }), series: F("object"), windows: F("object") },
    "POST /api/v1/assessment/vital-signs": { purpose: F("string", { req: 1 }), lawful_basis: F("string", { req: 1 }), fs: F("number", { req: 1, desc: "Sample rate Hz" }), recorded_at: F("number", { req: 1 }), mode: F("enum", { def: "estimate", enum: ["estimate", "track"] }), consent: F("object"), radar_displacement_m: F("number[]"), csi_amplitude: F("number[][]") },
  };
  /* MCP tool input schemas — mcp_server.py */
  const MCP = {
    mercury_detect_anomaly: { data: F("number[][]", { req: 1 }) },
    mercury_detect_fusion: { data: F("number[][]", { req: 1 }), domain: F("string"), gdpr_report: F("boolean"), subject_id: F("string") },
    mercury_tier_detect: { data: F("number[]", { req: 1 }), labels: F("integer[]"), subset: F("string[]"), method: F("enum", { enum: ["stacking", "bma", "average", "consensus"] }), contamination: F("number", { def: 0.05, min: 0, max: 0.5 }), conformal_alpha: F("number"), include_attribution: F("boolean"), include_counterfactual: F("boolean"), counterfactual_index: F("integer"), counterfactual_method: F("enum", { enum: ["wachter", "dice", "growing_spheres", "prototype", "genetic"] }) },
    mercury_localize_root_cause: { observations: F("number[][]", { req: 1 }), adjacency: F("number[][]"), train: F("number[][]"), top_k: F("integer"), node_names: F("string[]") },
    mercury_hazard_visualize: { hazard: F("enum", { enum: ["earthquake", "tsunami", "meteor", "wildfire", "tornado", "hurricane", "volcanic", "landslide", "schumann"] }), format: F("enum", { enum: ["png", "geojson"] }), diagnostics: F("object"), arrays: F("object"), params: F("object"), geotransform: F("object") },
    mercury_score_ethics: { action: F("string", { req: 1 }), context: F("object") },
    mercury_research: { query: F("string", { req: 1 }), max_sources: F("integer", { min: 1, max: 10 }) },
    mercury_answer: { question: F("string", { req: 1 }), max_sources: F("integer", { min: 1, max: 10 }) },
    mercury_write_document: { title: F("string", { req: 1 }), sections: F("object[]", { req: 1, desc: "[{heading, body}]" }), format: F("enum", { enum: ["markdown", "html", "text"] }) },
    mercury_calibrate_confidence: { scores: F("number[]", { req: 1 }), labels: F("integer[]", { req: 1 }), method: F("string") },
    mercury_verify_claims: { text: F("string", { req: 1 }) },
    mercury_check_provenance: { text: F("string", { req: 1 }), sources: F("string[]"), verified: F("boolean") },
    mercury_self_consistency: { answers: F("string[]", { req: 1 }), prob: F("number") },
    mercury_value_metrics: { stream: F("string") },
    mercury_verify_statements: { statements: F("object[]", { req: 1 }), evidence: F("object[]"), config: F("object"), source_reliability: F("object"), verifiability_reference: F("object[]") },
    mercury_screen_intel_reports: { intel_reports: F("object", { req: 1 }), history: F("object") },
    mercury_fuse_intelligence: { intel_reports: F("object", { req: 1 }), temporal_context: F("object[]"), history: F("object") },
    mercury_concealed_information_test: { subject_pseudonym: F("string", { req: 1 }), administered_at: F("number", { req: 1 }), purpose: F("string", { req: 1 }), consent: F("object", { req: 1 }), questions: F("object[]", { req: 1 }), series: F("object", { req: 1 }), windows: F("object"), config: F("object"), probe_details_publicly_disclosed: F("boolean") },
    mercury_estimate_vital_signs: { purpose: F("string", { req: 1 }), lawful_basis: F("string", { req: 1 }), fs: F("number", { req: 1 }), recorded_at: F("number", { req: 1 }), mode: F("enum", { enum: ["estimate", "track"] }), subject_pseudonym: F("string"), consent: F("object"), attestation: F("object"), radar_displacement_m: F("number[]"), radar_phase_rad: F("number[]"), wavelength_m: F("number"), csi_amplitude: F("number[][]"), csi_phase: F("number[][]"), include_waveform: F("boolean") },
    mercury_adjudication_status: {},
  };
  /* CLI command → API route that performs the same work when a server is connected */
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
  const STR = { message: "status", email: "you@example.com", password: "<your-password>", name: "my-key", model_type: "<model_type>", query: "anomaly detection on power-grid telemetry", question: "What does the ethics gate check?", text: "Grid load spiked to 3x baseline at 03:00 UTC.", action: "share an anonymised anomaly report with the grid operator", title: "Incident summary", subject_pseudonym: "subject-001", purpose: "<purpose>", lawful_basis: "<lawful_basis>", hazard: "earthquake", stream: "default", subject_id: "subject-001", domain: "network_security", token: "<token-from-email>", new_password: "<new-password>", current_password: "<current-password>", code: "<6-digit-code>", new_email: "new@example.com", version_number: "<version_number>", totp_code: "", recovery_code: "" };
  function sample(field, key) {
    if (field.def !== undefined) return field.def;
    if (field.type === "string" && key && STR[key] !== undefined) return STR[key];
    switch (field.type) {
      case "number[]": return [0.4, 0.5, 0.45, 10.0, 0.5];
      case "number[][]": case "number[]|number[][]": return [[0.3, 0.2], [0.4, 0.2], [4.0, 0.2], [0.3, 0.2]];
      case "integer[]": return [0, 0, 1, 0];
      case "string[]": return ["f0", "f1"];
      case "object[]": return [{}];
      case "object": return {};
      case "boolean": return false;
      case "integer": return 1;
      case "number": return 0.5;
      case "enum": return field.enum[0];
      default: return key ? "<" + key + ">" : "";
    }
  }
  function body(schema, vals, opts) {
    const rk = Object.keys(schema).find((k) => schema[k].raw); if (rk) { const v = vals && vals[rk]; return v && typeof v === "object" ? v : {}; }
    const o = {}; const only = opts && opts.requiredOnly;
    Object.keys(schema).forEach((k) => { const f = schema[k]; if (f.p || f.q) return; const v = vals && vals[k]; if (v !== undefined && v !== "" && v !== null) o[k] = v; else if (f.req || (!only && f.def !== undefined)) o[k] = sample(f, k); });
    return o;
  }
  function missing(schema, b, vals) { if (Object.keys(schema).some((k) => schema[k].raw)) return b && Object.keys(b).length ? [] : Object.keys(schema).filter((k) => schema[k].raw); return Object.keys(schema).filter((k) => { const f = schema[k]; if (!f.req) return false; const v = f.p || f.q ? (vals || {})[k] : b[k]; return v === undefined || v === "" || (Array.isArray(v) && !v.length); }); }
  function pathParams(route) { return [...route.matchAll(/\{(\w+)\}/g)].map((m) => m[1]); }
  function withParams(route, schema) { const sc = Object.assign({}, schema || {}); pathParams(route).forEach((k) => { if (!sc[k]) sc[k] = F("string", { req: 1, p: 1, desc: "path parameter" }); }); return sc; }
  function url(route, vals, schema) { const path = route.split(" ")[1]; let u = path.replace(/\{(\w+)\}/g, (m, k) => (vals && vals[k] != null && vals[k] !== "" ? encodeURIComponent(vals[k]) : "{" + k + "}")); const qs = Object.keys(schema || {}).filter((k) => schema[k].q && vals && vals[k] != null && vals[k] !== "").map((k) => encodeURIComponent(k) + "=" + encodeURIComponent(vals[k])); return u + (qs.length ? "?" + qs.join("&") : ""); }

  window.MercuryCLIAPI = {
    get cfg() { return Object.assign({}, cfg); },
    budget: () => Object.assign({}, budget),
    set(o) { Object.assign(cfg, o); save(); },
    connected: () => !!cfg.baseUrl,
    simulated: () => cfg.baseUrl.startsWith("sim://"),
    health: () => call("GET", "/health"),
    async probeUrl(url) { const saved = cfg.baseUrl; cfg.baseUrl = url; try { return await call("GET", "/health"); } finally { cfg.baseUrl = saved; } },
    openapi: () => call("GET", "/openapi.json"),
    post: (path, b, o) => call("POST", path, b, o),
    get: (path) => call("GET", path),
    del: (path) => call("DELETE", path),
    speak: (message, domain, session_id, opts) => call("POST", "/api/v1/voice/speak", Object.assign({ message, domain: domain || null, session_id: session_id || null }, opts && opts.tone ? { tone: opts.tone } : {}, opts && opts.verbosity ? { verbosity: opts.verbosity } : {})),
    alertsUrl() { return null; /* api/voice.py has no WebSocket route on main; the dock shows alerts as absent instead of opening a socket that cannot connect */ },
    narrate: (detection_result, domain, verbosity) => call("POST", "/api/v1/voice/narrate", { detection_result, domain: domain || null, verbosity: verbosity || "normal" }),
    greet: (domain) => call("GET", "/api/v1/voice/greet" + (domain ? "?domain=" + encodeURIComponent(domain) : "")),
    voiceStatus: () => call("GET", "/api/v1/voice/status"),
    voiceHistory: (limit, session_id) => call("GET", "/api/v1/voice/history?limit=" + Math.max(1, Math.min(200, +limit || 20)) + (session_id ? "&session_id=" + encodeURIComponent(session_id) : "")),
    agentNotice: () => call("GET", "/agent-notice"),
    securityTxt: () => call("GET", "/.well-known/security.txt"),
    sourceOffer: () => call("GET", "/source"),
    batchSubmit: (b) => call("POST", "/api/v1/batch/detect", b),
    batchStatus: (id) => call("GET", "/api/v1/batch/jobs/" + encodeURIComponent(id)),
    batchResults: (id, offset, limit) => call("GET", "/api/v1/batch/jobs/" + encodeURIComponent(id) + "/results" + (offset != null ? "?offset=" + offset + "&limit=" + (limit || 1000) : "")),
    batchCancel: (id) => call("DELETE", "/api/v1/batch/jobs/" + encodeURIComponent(id)),
    SCHEMAS, MCP, CLI_TO_ROUTE, CAP, SEVERITY, sample, body, missing, pathParams, withParams, url,
    put: (path, b) => call("PUT", path, b),
    curl(route, b, vals, schema) { const m = route.split(" ")[0], p = url(route, vals, schema); const h = cfg.apiKey ? " \\\n  -H 'X-API-Key: " + cfg.apiKey.slice(0, 4) + "…'" : ""; return "curl -X " + m + " " + (base() || "http://localhost:8000") + p + h + (b ? " \\\n  -H 'Content-Type: application/json' \\\n  -d '" + JSON.stringify(b) + "'" : ""); },
    mcpCall(name, args) { return JSON.stringify({ jsonrpc: "2.0", id: 1, method: "tools/call", params: { name, arguments: args } }, null, 2); },
  };
})();

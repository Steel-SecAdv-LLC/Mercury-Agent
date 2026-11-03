/* Mercury CLI — simulated engine. An in-browser stand-in for `mercury-agent serve` that answers
   the routes the console uses with response shapes taken from main's pydantic models
   (api/server.py, api/routes/detection.py, batch.py, hazard.py, assessment.py, api/voice.py).
   Numbers are computed from the request data with simple, deterministic statistics. They are
   NOT Mercury's models: every response carries "_simulated": true and the console labels it. */
(function () {
  const VERSION = "3.0.0";
  const SIMCFG = { fast: false, rateLimitNext: 0 };
  const t0 = Date.now();
  const num = (a) => (Array.isArray(a) ? a.map(Number).filter((x) => isFinite(x)) : []);
  const col0 = (d) => (Array.isArray(d) ? d.map((r) => (Array.isArray(r) ? +r[0] : +r)) : []);
  const med = (a) => { const b = a.slice().sort((x, y) => x - y), m = b.length >> 1; return b.length ? (b.length % 2 ? b[m] : (b[m - 1] + b[m]) / 2) : 0; };
  const mean = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0);
  const sd = (a) => { const m = mean(a); return Math.sqrt(mean(a.map((x) => (x - m) * (x - m)))) || 1e-9; };
  const clip = (x) => Math.max(0, Math.min(1, x));
  const robust = (a) => { const m = med(a), mad = med(a.map((x) => Math.abs(x - m))) || 1e-9; return a.map((x) => Math.abs(x - m) / (1.4826 * mad)); };
  const zs = (a) => { const m = mean(a), s = sd(a); return a.map((x) => Math.abs(x - m) / s); };
  const iqr = (a) => { const b = a.slice().sort((x, y) => x - y), q = (p) => b[Math.min(b.length - 1, Math.floor(p * (b.length - 1)))], q1 = q(0.25), q3 = q(0.75), r = q3 - q1 || 1e-9; return a.map((x) => (x < q1 ? (q1 - x) / r : x > q3 ? (x - q3) / r : 0)); };
  const isoish = (a) => { const r = robust(a); return r.map((v, i) => v * 0.8 + (i > 0 ? Math.abs(a[i] - a[i - 1]) / (sd(a) * 4) : 0)); };
  const toScores = (raw, k) => raw.map((v) => +clip(v / k).toFixed(4));
  const thrFor = (sens) => +(0.75 - 0.4 * (sens == null ? 0.5 : sens)).toFixed(3);
  const ERR = (status, error, message) => ({ status, body: { error, message, details: null, request_id: null } });
  const need = (b, k) => (b == null || b[k] == null || (Array.isArray(b[k]) && !b[k].length) ? ERR(422, "validation_error", "field '" + k + "' is required") : null);

  function univariate(b) {
    const e = need(b, "data"); if (e) return e; const x = num(b.data); if (x.length < 3) return ERR(422, "validation_error", "data must have at least 3 points");
    const method = b.method || "zscore", raw = method === "iqr" ? iqr(x) : method === "isolation_forest" ? isoish(x) : zs(x);
    const scores = toScores(raw, method === "iqr" ? 3 : 4), threshold = thrFor(b.sensitivity), anomalies = scores.map((s) => s >= threshold);
    return { status: 200, body: { anomalies, scores, anomaly_points: anomalies.map((a, i) => (a ? { index: i, value: x[i], score: scores[i], severity: scores[i] / threshold >= 3 ? "critical" : scores[i] / threshold >= 2 ? "high" : scores[i] / threshold >= 1.5 ? "medium" : "low" } : null)).filter(Boolean), method, threshold, summary: { n_points: x.length, n_anomalies: anomalies.filter(Boolean).length, mean: +mean(x).toFixed(4), std: +sd(x).toFixed(4) } } };
  }
  function multivariate(b) {
    const e = need(b, "data"); if (e) return e; const M = b.data.map((r) => (Array.isArray(r) ? r.map(Number) : [+r])), d = M[0].length;
    const cols = Array.from({ length: d }, (_, j) => robust(M.map((r) => r[j])));
    const scores = M.map((_, i) => +clip(Math.sqrt(cols.reduce((a, c) => a + c[i] * c[i], 0) / d) / 4).toFixed(4)), threshold = thrFor(b.sensitivity);
    const features = b.features && b.features.length === d ? b.features : Array.from({ length: d }, (_, j) => "f" + j);
    return { status: 200, body: { anomalies: scores.map((s) => s >= threshold), scores, feature_contributions: Object.fromEntries(features.map((f, j) => [f, cols[j].map((v) => +v.toFixed(3))])), method: "robust_mahalanobis", threshold, features, summary: { n_samples: M.length, n_features: d } } };
  }
  function fusion(b) {
    const e = need(b, "data"); if (e) return e; const x = col0(b.data), dets = b.detectors && b.detectors.length ? b.detectors : ["statistical", "temporal", "dimensional"];
    const per = { statistical: Math.max(...zs(x)) / 4, temporal: Math.max(...x.map((v, i) => (i ? Math.abs(v - x[i - 1]) : 0))) / (sd(x) * 4), dimensional: Math.max(...robust(x)) / 5 };
    const detector_scores = Object.fromEntries(dets.map((k) => [k, +clip(per[k] != null ? per[k] : 0.3).toFixed(4)])), vals = Object.values(detector_scores), fused = +mean(vals).toFixed(4), threshold = thrFor(b.sensitivity);
    return { status: 200, body: { is_anomaly: fused >= threshold, fused_score: fused, confidence: +clip(0.5 + Math.abs(fused - threshold)).toFixed(3), detector_scores, detector_contributions: Object.fromEntries(dets.map((k) => [k, +(detector_scores[k] / (vals.reduce((a, v) => a + v, 0) || 1)).toFixed(3)])), threshold, explanation: "Mean of " + dets.length + " detector scores vs threshold " + threshold + " (simulated)." } };
  }
  function threeR(b) {
    const e = need(b, "data"); if (e) return e; const x = num(b.data), n = x.length, bands = b.harmonic_bands || 8;
    const harm = Array.from({ length: Math.min(bands, Math.floor(n / 2)) }, (_, k) => { let re = 0, im = 0; x.forEach((v, t) => { re += v * Math.cos((2 * Math.PI * (k + 1) * t) / n); im -= v * Math.sin((2 * Math.PI * (k + 1) * t) / n); }); return +(Math.sqrt(re * re + im * im) / n).toFixed(4); });
    const rec = clip(Math.max(...zs(x)) / 4), res = clip((Math.max(...harm, 0) / (mean(harm) || 1)) / 8), opt = clip(1 - sd(x) / (Math.abs(mean(x)) + sd(x)));
    const w = { recursion: 0.382, resonance: 0.382, refactoring: 0.236 }, eth = +Math.pow(b.ethical_threshold || 0.96, 1.618).toFixed(4), fs = +((w.recursion * rec + w.resonance * res + w.refactoring * opt) * eth).toFixed(4);
    return { status: 200, body: { fusion_score: fs, recursion_score: +rec.toFixed(4), resonance_score: +res.toFixed(4), optimization_score: +opt.toFixed(4), ethical_scaling: eth, lyapunov_bound: 0.25, is_stable: fs < 0.75, weights: w, harmonic_analysis: { bands: harm } } };
  }
  function neuro(b) {
    const e = need(b, "data"); if (e) return e; const rows = b.data, keys = Object.keys(rows[0] || {}), x = rows.map((r) => +r[keys[0]]);
    const sc = robust(x).map((v) => +clip(v / 4).toFixed(4)), overall = +Math.max(...sc).toFixed(4), thr = thrFor(b.sensitivity), fired = sc.filter((s) => s >= thr).length;
    return { status: 200, body: { overall_score: overall, overall_confidence: +clip(0.55 + overall / 3).toFixed(3), neural_contribution: 0.6, symbolic_contribution: 0.4, patterns_detected: fired, rules_fired: fired ? 2 : 0, anomaly_scores: sc.map((s, i) => ({ index: i, score: s, is_anomaly: s >= thr })), decision: overall >= thr ? "flag" : "pass", explanation: b.include_explanations === false ? "" : fired + " record(s) exceed the robust-deviation rule at " + thr + " (simulated).", audit_trail: [{ step: "neural", strategy: b.fusion_strategy || "confidence_weighted" }, { step: "symbolic", rule: "robust_deviation" }] } };
  }
  const TIER_NAMES = ["spectral_residual", "bocpd", "hawkes", "particle_filter", "imm", "digital_twin", "spot_evt", "gaussian_process", "survival", "energy_based", "deep_svdd", "echo_state", "spiking", "rca", "deeplog_sequence", "frequent_pattern", "srcnn", "diffusion_ad"];
  function tier(b) {
    const e = need(b, "data"); if (e) return e; const x = num(b.data), subset = b.subset && b.subset.length ? b.subset : TIER_NAMES.slice(0, 16);
    const bad = subset.filter((s) => !TIER_NAMES.includes(s)); if (bad.length) return ERR(400, "bad_request", "unknown tier detector(s): " + bad.join(", "));
    if (subset.some((s) => s === "srcnn" || s === "diffusion_ad")) return ERR(400, "bad_request", "srcnn/diffusion_ad need torch on the server (simulated engine reports this as unavailable)");
    const base = robust(x), d1 = x.map((v, i) => (i ? Math.abs(v - x[i - 1]) : 0)), members = {};
    subset.forEach((nm, j) => { const mix = (j % 3) / 3; members[nm] = base.map((v, i) => +clip((v * (1 - mix) + (d1[i] / (sd(x) || 1)) * mix) / 4).toFixed(4)); });
    const scores = x.map((_, i) => +mean(subset.map((nm) => members[nm][i])).toFixed(4)), c = b.contamination == null ? 0.05 : b.contamination, k = Math.max(1, Math.round(x.length * c));
    const threshold = scores.slice().sort((p, q) => q - p)[Math.min(k, scores.length) - 1], flags = scores.map((s) => (s >= threshold ? 1 : 0));
    const out = { scores, flags, threshold, method: b.method || "average", members: subset, n_points: x.length, n_flagged: flags.reduce((a, f) => a + f, 0) };
    if (b.include_attribution) out.attribution = Object.fromEntries(subset.map((nm) => [nm, +mean(members[nm]).toFixed(4)]));
    if (b.conformal_alpha != null) out.conformal = { alpha: b.conformal_alpha, threshold };
    return { status: 200, body: out };
  }
  function flagship(b) { const e = need(b, "data"); if (e) return e; const r = multivariate({ data: b.data, sensitivity: 0.5 }).body; const prob = r.scores; return { status: 200, body: { anomaly_prob: prob, is_anomaly: prob.map((p) => p >= 0.5), threshold_used: 0.5, threshold_source: b.domain ? "domain:" + b.domain : "model_default", domain: b.domain || null, decision: { disposition: prob.some((p) => p >= 0.5) ? "recommend" : "no_action" }, explanation: b.explain ? "Domain-tuned threshold 0.5 (simulated GOSNN)." : undefined } }; }
  function rca(b) { const e = need(b, "observations"); if (e) return e; const O = b.observations; if (O.length < 2 || !Array.isArray(O[0]) || O[0].length < 2) return ERR(422, "validation_error", "observations must be rows × ≥2 nodes"); const d = O[0].length, names = b.node_names && b.node_names.length === d ? b.node_names : Array.from({ length: d }, (_, j) => "node_" + j); const last = O[O.length - 1], hist = O.slice(0, -1); const z = names.map((nm, j) => { const c = hist.map((r) => +r[j]); return { node: nm, score: +clip(Math.abs(last[j] - mean(c)) / (sd(c) * 4)).toFixed(4) }; }).sort((p, q) => q.score - p.score); return { status: 200, body: { root_causes: z.slice(0, b.top_k || 3), ranking: z.map((x) => x.node), n_nodes: d } }; }
  function hazard(b) {
    const KNOWN = ["earthquake", "tsunami", "meteor", "wildfire", "tornado", "hurricane", "volcanic", "landslide", "schumann"];
    if (!b || !KNOWN.includes(b.hazard)) return ERR(400, "bad_request", "unknown hazard " + JSON.stringify(b && b.hazard) + "; expected one of " + KNOWN.join(", "));
    const arrs = b.arrays || {}, key = Object.keys(arrs)[0]; if (!key) return ERR(400, "bad_request", "no input arrays for hazard " + b.hazard);
    let x = arrs[key]; if (Array.isArray(x) && Array.isArray(x[0])) x = x.map((r) => mean(r.map(Number))); x = num(x);
    const sc = robust(x), peak = sc.indexOf(Math.max(...sc));
    if (b.format === "geojson") return { status: 200, body: { type: "FeatureCollection", features: [{ type: "Feature", geometry: { type: "Point", coordinates: [0, 0] }, properties: { hazard: b.hazard, peak_index: peak, peak_score: +clip(sc[peak] / 5).toFixed(3), simulated: true } }], _simulated: true } };
    const W = 640, Hh = 220, cv = document.createElement("canvas"); cv.width = W; cv.height = Hh; const g = cv.getContext("2d");
    g.fillStyle = "#06080B"; g.fillRect(0, 0, W, Hh); g.strokeStyle = "rgba(0,188,212,0.08)"; for (let i = 0; i < W; i += 40) { g.beginPath(); g.moveTo(i, 0); g.lineTo(i, Hh); g.stroke(); }
    const mn = Math.min(...x), mx = Math.max(...x), Y = (v) => Hh - 30 - ((v - mn) / (mx - mn || 1)) * (Hh - 60), X = (i) => 20 + (i / Math.max(1, x.length - 1)) * (W - 40);
    g.strokeStyle = "#00BCD4"; g.lineWidth = 1.5; g.beginPath(); x.forEach((v, i) => (i ? g.lineTo(X(i), Y(v)) : g.moveTo(X(i), Y(v)))); g.stroke();
    x.forEach((v, i) => { if (sc[i] > 3.5) { g.fillStyle = "#FF4D52"; g.beginPath(); g.arc(X(i), Y(v), 4, 0, 7); g.fill(); } });
    g.fillStyle = "#8FA3B8"; g.font = "12px monospace"; g.fillText(b.hazard.toUpperCase() + " · " + key + " · " + x.length + " samples · SIMULATED ENGINE", 20, 18);
    return { status: 200, image: cv.toDataURL("image/png") };
  }
  const turns = [], detections = [];
  // Every detection the simulated engine answers is remembered here, so the simulated voice can say what it
  // has seen -- the same wiring mercury-agent serve has (record_detection mirrors into the voice retriever).
  function remember(path, body, r) {
    if (!r || r.status !== 200 || !r.body) return; const out = r.body, flags = Array.isArray(out.anomalies) ? out.anomalies : Array.isArray(out.results) ? out.results.map((x) => !!(x.is_anomaly ?? x.anomaly)) : null;
    const n = Array.isArray(body && body.data) ? body.data.length : (flags ? flags.length : null), flagged = flags ? flags.filter(Boolean).length : (out.summary && out.summary.n_anomalies != null ? out.summary.n_anomalies : null);
    const scores = Array.isArray(out.scores) ? out.scores.filter((v) => typeof v === "number") : [];
    detections.push({ route: path.replace("/api/v1/detect/", ""), n, flagged, top: scores.length ? +Math.max.apply(null, scores).toFixed(3) : null, domain: body && body.domain || null, at: Date.now() / 1000 }); if (detections.length > 100) detections.shift();
  }
  function turn(speaker, content, type) { turns.push({ turn_id: "turn_" + (turns.length + 1), type: type || "query", speaker, content: String(content), timestamp: Date.now() / 1000 }); if (turns.length > 200) turns.shift(); }
  function speak(b) {
    const e = need(b, "message"); if (e) return e; const q = String(b.message).toLowerCase();
    turn("user", b.message);
    const R = (message, intent, conf, sources, extra) => { turn("mercury", message); return { status: 200, body: Object.assign({ message, confidence: conf, intent, sources: sources || [], timestamp: Date.now() / 1000, metadata: { session_id: b.session_id || null, domain: b.domain || null }, reasoning_summary: null, search_performed: false, search_results_count: 0, suggested_follow_ups: [], uncertainty_note: null, response_time_ms: +(1 + Math.random() * 3).toFixed(1) }, extra || {}, { _simulated: true }) }; };
    const ago = (t) => { const s = Math.max(0, Math.round(Date.now() / 1000 - t)); return s < 60 ? s + " s ago" : Math.round(s / 60) + " min ago"; };
    const recent = detections.slice(-5).reverse(), flaggedTotal = detections.reduce((a, d) => a + (d.flagged || 0), 0);
    const describe = (d) => d.route + " on " + (d.n != null ? d.n + " samples" : "a series") + " · " + (d.flagged != null ? d.flagged + " flagged" : "no per-sample flags") + (d.top != null ? " · top score " + d.top : "") + (d.domain ? " · " + d.domain : "") + " · " + ago(d.at);
    if (/status|health|running|operational/.test(q)) return R("Mercury Agent operational (simulated engine). Detection, tier, hazard and assessment routes are answering. This session: " + detections.length + " detection run" + (detections.length === 1 ? "" : "s") + ", " + flaggedTotal + " sample" + (flaggedTotal === 1 ? "" : "s") + " flagged, " + Object.keys(jobs).length + " batch job" + (Object.keys(jobs).length === 1 ? "" : "s") + ".", "status", 0.9, ["system_status", "detection_log"], { reasoning_summary: "Counted the runs and batch jobs this simulated engine has answered since the page loaded.", suggested_follow_ups: ["What anomalies have you detected?", "Which detector flagged the most samples?", "How do I bring my own data in?"] });
    if (/help|how do you work|what can you do|guide me/.test(q)) return R("I can help with anomaly detection, pattern analysis, and STEAM exploration. You can ask me about: (1) Detection results and their meaning, (2) Historical patterns and trends, (3) Knowledge graph relationships, (4) System status and health. I provide transparent, evidence-backed responses with explicit confidence levels.", "help", 1.0);
    if (/which detector|most samples|most flagged/.test(q) && detections.length) { const by = {}; detections.forEach((d) => { by[d.route] = (by[d.route] || 0) + (d.flagged || 0); }); const top = Object.entries(by).sort((a, c) => c[1] - a[1])[0]; return R(top[0] + " flagged the most samples this session (simulated engine): " + top[1] + " across " + detections.filter((d) => d.route === top[0]).length + " run" + (detections.filter((d) => d.route === top[0]).length === 1 ? "" : "s") + ".", "search_patterns", 0.8, ["detection_log"], { search_performed: true, search_results_count: detections.length, reasoning_summary: "Summed per-sample flags by route over the detection log.", suggested_follow_ups: ["What anomalies have you detected?", "Explain the selected sample"] }); }
    if (/detect|anomal|scan|recent|event|flag/.test(q)) {
      if (!recent.length) return R("No detections yet this session (simulated engine). Send a series to /api/v1/detect/univariate, press RUN on a detect command, or run a Mass Run to compare detectors on it; I will remember every result.", "search_patterns", 0.6, ["detection_log"], { search_performed: true, search_results_count: 0, uncertainty_note: "Simulated engine: results are computed in this browser and are not Mercury's models.", suggested_follow_ups: ["How do I collect data?", "What should I run next?"] });
      return R(detections.length + " detection run" + (detections.length === 1 ? "" : "s") + " this session, " + flaggedTotal + " sample" + (flaggedTotal === 1 ? "" : "s") + " flagged (simulated engine). Most recent: " + recent.map(describe).join("; ") + ".", "search_patterns", 0.8, ["detection_log"], { search_performed: true, search_results_count: detections.length, reasoning_summary: "Read the last " + recent.length + " entries of the detection log, newest first.", uncertainty_note: "Simulated engine: scores are z-score style and relative to each dataset.", suggested_follow_ups: ["Which detector flagged the most samples?", "Explain the selected sample", "Run the tier ensemble on this data"] });
    }
    if (/who are you|what are you|your name|introduce/.test(q)) return R("I am the Mercury Agent voice of this simulated engine: a browser stand-in for mercury-agent serve that answers from what this page has run. Connect a real server for Mercury's own voice and models.", "greeting", 0.95, ["system_status"], { suggested_follow_ups: ["What is your status?", "What can you do?"] });
    return R("No matching events, patterns or facts in memory for that query (simulated engine). I can report status, what has been detected this session, which detector flagged the most, and how to bring data in.", "unknown", 0.3, [], { search_performed: true, search_results_count: 0, suggested_follow_ups: ["What is your status?", "What anomalies have you detected?", "What can you do?"] });
  }
  function narrate(b) {
    const e = need(b, "detection_result"); if (e) return e; const d = b.detection_result || {}, v = b.verbosity || "normal";
    const det = !!(d.anomaly_detected ?? d.is_anomaly), score = +(d.anomaly_score ?? d.anomaly_prob ?? 0) || 0, conf = +(d.confidence ?? 0.5) || 0.5;
    const level = conf >= 0.9 ? "very_high" : conf >= 0.75 ? "high" : conf >= 0.5 ? "moderate" : conf >= 0.25 ? "low" : "very_low";
    const summary = det ? "Anomaly detected with score " + score.toFixed(2) : "No anomaly detected";
    const chain = (d.reasoning_chain || []).map(String), causes = (d.causal_factors || []).map(String);
    const detail = v === "brief" ? summary + "." : (det ? "The analysis indicates an anomaly with a score of " + score.toFixed(2) + "." : "The data appears normal within expected parameters.") + (chain.length ? " Reasoning: " + chain.join("; ") + "." : "") + (causes.length && v === "detailed" ? " Contributing factors: " + causes.join(", ") + "." : "") + (d.flagged != null ? " " + d.flagged + " of " + d.samples + " samples were flagged." : "");
    const recs = (d.recommendations && d.recommendations.length ? d.recommendations : det ? ["Review the flagged data", "Check related systems"] : []).map(String);
    return { status: 200, body: { summary, detailed_explanation: detail, recommendations: recs, confidence_statement: "Confidence: " + Math.round(conf * 100) + "%. " + (conf > 0.8 ? "High confidence in this assessment." : "Moderate confidence - additional verification recommended."), severity: level, _simulated: true } };
  }
  const jobs = {};
  function batchSubmit(b, auth) { if (!auth) return ERR(401, "unauthorized", "API key required"); const e = need(b, "data"); if (e) return e; const id = "sim-job-" + Math.random().toString(36).slice(2, 10); jobs[id] = { id, method: b.method || "univariate", data: b.data, created: Date.now(), cancelled: false }; return { status: 202, body: { job_id: id, status: "pending", message: "Job accepted (simulated)", estimated_duration_seconds: 3, poll_url: "/api/v1/batch/jobs/" + id } }; }
  function jobState(j) { const el = SIMCFG.fast ? 99 : (Date.now() - j.created) / 1000, total = j.data.length, done = j.cancelled ? Math.min(total, Math.floor(el * total / 3)) : Math.min(total, Math.floor((el / 3) * total)); const status = j.cancelled ? "cancelled" : el < 0.6 ? "pending" : done < total ? "running" : "completed"; return { job_id: j.id, status, method: j.method, total_items: total, processed_items: done, failed_items: 0, progress_percentage: +((done / total) * 100).toFixed(1), created_at: new Date(j.created).toISOString(), started_at: el >= 0.6 ? new Date(j.created + 600).toISOString() : null, completed_at: status === "completed" ? new Date(j.created + 3000).toISOString() : null, error_message: null, metadata: { simulated: true } }; }
  function batchResults(j, q) { const off = +(q && q.offset) || 0, lim = +(q && q.limit) || 1000; const x = col0(j.data), r = univariate({ data: x.length >= 3 ? x : x.concat([0, 0, 0]), sensitivity: 0.5, method: "zscore" }).body; const results = x.map((v, i) => ({ index: i, value: v, score: r.scores[i], is_anomaly: r.anomalies[i] })); const page = results.slice(off, off + lim); return { job_id: j.id, status: "completed", total_results: results.length, offset: off, limit: lim, results: page, has_more: off + lim < results.length }; }
  function assessment(path, b, auth) {
    if (!auth) return ERR(401, "unauthorized", "credential required");
    if (path.endsWith("/statements/verify")) { if (!b || !b.statements) return ERR(422, "validation_error", "statements required"); const ev = b.evidence || []; return { status: 200, body: { statements: b.statements.map((s) => ({ statement_id: s.statement_id, corroboration: { log_odds: +(ev.filter((e) => e.stance === "supports").length * 1.1).toFixed(2), independent_clusters: new Set(ev.map((e) => e.source_id)).size }, contradictions: [], consistency: { speaker: s.speaker, indicators: [] } })), _simulated: true } }; }
    if (path.endsWith("/intel/screen")) { const r = (b && b.intel_reports) || {}; const ids = Object.values(r).map((x) => x.report_id).filter(Boolean); return { status: 200, body: { assessment: { independence: Object.values(r).map((x) => ({ report_id: x.report_id, source_id: x.source_id, independent: !(x.derived_from && x.derived_from.some((d) => ids.includes(d))) })) }, _simulated: true } }; }
    if (path.endsWith("/intel/fuse")) { const r = (b && b.intel_reports) || {}; const vals = Object.values(r); return { status: 200, body: { decides_with: "evidence", fused: { confidence: +mean(vals.map((x) => +x.confidence || 0)).toFixed(3), threat_score: +mean(vals.map((x) => +x.threat_score || 0)).toFixed(3) }, _simulated: true } }; }
    return ERR(501, "not_simulated", "This assessment route needs a real engine (consent and policy gates).");
  }
  async function handle(method, fullPath, body, headers) {
    if (!SIMCFG.fast) await new Promise((r) => setTimeout(r, 40 + Math.random() * 80));
    const path = fullPath.split("?")[0], auth = !!(headers && headers["X-API-Key"]);
    if (SIMCFG.rateLimitNext > 0 && path !== "/health") { SIMCFG.rateLimitNext--; return { status: 429, retryAfter: 0, body: { error: "rate_limited", message: "Too many requests (simulated)", details: null, request_id: null, _simulated: true } }; }
    const map = {
      "GET /health": () => ({ status: 200, body: { status: "healthy", version: VERSION, uptime_seconds: Math.round((Date.now() - t0) / 1000), _simulated: true } }),
      "GET /api/v1/voice/status": () => ({ status: 200, body: { status: "operational", message: "Mercury voice interface is active and ready for conversation.", components: { voice_interface: "operational", narrative_engine: "operational", conversation_history: turns.length }, statistics: { conversation_turns: turns.length }, _simulated: true } }),
      "GET /api/v1/voice/greet": () => ({ status: 200, body: { message: "Mercury Agent online (simulated engine). Ask about a detection, a pattern, or system status.", capabilities: ["Natural language conversation", "Anomaly detection analysis", "System status reporting", "Detection result explanation", "Historical event lookup"], _simulated: true } }),
      "GET /api/v1/voice/history": () => { const lim = Math.max(1, +(new URLSearchParams(fullPath.split("?")[1] || "").get("limit")) || 10); return { status: 200, body: { history: turns.slice(-lim), total_turns: turns.length, _simulated: true } }; },
      "GET /agent-notice": () => ({ status: 200, body: Object.assign({}, window.MercuryCLIAgent ? window.MercuryCLIAgent.NOTICE : { binding: false, authority: "none" }, { _simulated: true }) }),
      "GET /.well-known/security.txt": () => ({ status: 200, body: "# Mercury Agent security.txt (RFC 9116), generated by the running service.\nContact: mailto:steel.sa.llc@mercuryagent.global\nContact: https://github.com/Steel-SecAdv-LLC/Mercury-Agent/blob/main/SECURITY.md#reporting-a-vulnerability\nExpires: " + new Date(Date.now() + 180 * 864e5).toISOString().slice(0, 19) + "Z\nPolicy: https://github.com/Steel-SecAdv-LLC/Mercury-Agent/blob/main/SECURITY.md\nPolicy: https://github.com/Steel-SecAdv-LLC/Mercury-Agent/blob/main/docs/AGENT_NOTICE.md\nPreferred-Languages: en\n" }),
      "POST /api/v1/voice/speak": () => speak(body), "POST /api/v1/voice/narrate": () => narrate(body),
      "POST /api/v1/detect/univariate": () => univariate(body), "POST /api/v1/detect/multivariate": () => multivariate(body),
      "POST /api/v1/detect/fusion": () => fusion(body), "POST /api/v1/detect/three-r": () => threeR(body), "POST /api/v1/detect/neurosymbolic": () => neuro(body),
      "POST /api/v1/detect/tier": () => tier(body), "POST /api/v1/detect/flagship": () => flagship(body), "POST /api/v1/detect/rca": () => rca(body),
      "POST /api/v1/hazard/visualize": () => hazard(body), "POST /api/v1/batch/detect": () => batchSubmit(body, auth),
    };
    let r = map[method + " " + path] ? map[method + " " + path]() : null;
    if (!r && path.startsWith("/api/v1/batch/jobs/")) { const parts = path.split("/"), id = parts[5], j = jobs[id]; if (!j) r = ERR(404, "not_found", "job " + id + " not found"); else if (method === "DELETE") { j.cancelled = true; r = { status: 200, body: jobState(j) }; } else if (parts[6] === "results") { const qs = Object.fromEntries(new URLSearchParams(fullPath.split("?")[1] || "")); r = { status: 200, body: batchResults(j, qs) }; } else r = { status: 200, body: jobState(j) }; }
    if (!r && path.startsWith("/api/v1/assessment/") && method === "POST") r = assessment(path, body, auth);
    if (!r) r = ERR(501, "not_simulated", method + " " + path + " is not part of the simulated engine. Connect a real mercury-agent serve.");
    if (method === "POST" && path.startsWith("/api/v1/detect/")) remember(path, body, r);
    if (r.body && typeof r.body === "object" && !Array.isArray(r.body) && r.body._simulated === undefined) r.body._simulated = true;
    return r;
  }
  window.MercuryCLISim = { handle, URL: "sim://mercury-agent", TIER_NAMES, cfg: SIMCFG };
})();

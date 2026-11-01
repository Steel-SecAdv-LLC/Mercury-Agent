/* Mercury CLI — Mass Run planner/executor and response summariser.
   Capabilities map 1:1 to routes on main (api/server.py, api/routes/detection.py, batch.py, voice.py).
   Hazard, assessment and account routes need domain-specific payloads and stay individual (REF forms). */
(function () {
  const SAMPLE = [0.42, 0.47, 0.44, 0.46, 0.43, 0.45, 3.9, 0.44, 0.46, 0.41, 0.45, 0.47, 0.43, 0.44, 2.8, 0.46];
  const series = (d) => (d ? d.matrix.map((r) => r[0]) : SAMPLE.slice());
  const matrix = (d) => (d ? d.matrix : SAMPLE.map((v, i) => [v, Math.sin(i / 2) * 0.1 + 0.5]));
  const records = (d) => matrix(d).map((r) => Object.fromEntries(r.map((v, j) => ["f" + j, v])));
  const C = (id, label, group, route, build, opts) => Object.assign({ id, label, group, route, build }, opts || {});
  const CAPS = [
    C("uni-z", "Univariate · z-score", "Detection", "POST /api/v1/detect/univariate", (d) => ({ data: series(d), sensitivity: 0.5, method: "zscore" }), { rec: 1, per: 1 }),
    C("uni-iqr", "Univariate · IQR", "Detection", "POST /api/v1/detect/univariate", (d) => ({ data: series(d), sensitivity: 0.5, method: "iqr" }), { rec: 1, per: 1 }),
    C("uni-if", "Univariate · model-based (multivariate route)", "Detection", "POST /api/v1/detect/multivariate", (d) => ({ data: series(d).map((v) => [v]), sensitivity: 0.5 }), { rec: 1, per: 1 }),
    C("multi", "Multivariate · correlated", "Detection", "POST /api/v1/detect/multivariate", (d) => ({ data: matrix(d), sensitivity: 0.5, correlate_features: true }), { rec: 1, per: 1 }),
    C("fusion", "Fusion · statistical + temporal + dimensional", "Detection", "POST /api/v1/detect/fusion", (d) => ({ data: matrix(d), detectors: ["statistical", "temporal", "dimensional"], sensitivity: 0.5, return_contributions: true }), { rec: 1 }),
    C("three-r", "3R · recursion · resonance · refactoring", "Detection", "POST /api/v1/detect/three-r", (d) => ({ data: series(d), recursion_depth: 5, harmonic_bands: 8 }), { rec: 1 }),
    C("neuro", "Neuro-symbolic · confidence-weighted", "Detection", "POST /api/v1/detect/neurosymbolic", (d) => ({ data: records(d), sensitivity: 0.5, fusion_strategy: "confidence_weighted", include_explanations: true }), { rec: 1 }),
    C("tier-stack", "Tier ensemble · stacking", "Ensembles", "POST /api/v1/detect/tier", (d) => ({ data: series(d), method: "stacking", contamination: 0.1 }), { per: 1, block: () => "stacking requires per-point labels — run it from the API form with a labels array" }),
    C("tier-bma", "Tier ensemble · Bayesian model averaging", "Ensembles", "POST /api/v1/detect/tier", (d) => ({ data: series(d), method: "bma", contamination: 0.1 }), { per: 1 }),
    C("tier-avg", "Tier ensemble · average", "Ensembles", "POST /api/v1/detect/tier", (d) => ({ data: series(d), method: "average", contamination: 0.1 }), { per: 1 }),
    C("tier-cons", "Tier ensemble · consensus", "Ensembles", "POST /api/v1/detect/tier", (d) => ({ data: series(d), method: "consensus", contamination: 0.1 }), { per: 1 }),
    C("tier-conf", "Tier · conformal α 0.1 + attribution", "Ensembles", "POST /api/v1/detect/tier", (d) => ({ data: series(d), contamination: 0.1, conformal_alpha: 0.1, include_attribution: true }), { per: 1 }),
    C("flagship", "Flagship · GOSNN domain-tuned", "Ensembles", "POST /api/v1/detect/flagship", (d, ctx) => ({ data: matrix(d), domain: ctx.domain || null, explain: true }), { rec: 1 }),
    C("rca", "Root cause · last row as the anomaly", "Explain", "POST /api/v1/detect/rca", (d) => ({ observations: matrix(d), top_k: 3 }), { needsCols: 2 }),
  ];
  const TIER = { temporal_streaming: ["spectral_residual", "bocpd", "hawkes"], state_space: ["particle_filter", "imm", "digital_twin"], probabilistic: ["spot_evt", "gaussian_process", "survival"], generative: ["energy_based", "deep_svdd"], neuromorphic: ["echo_state", "spiking"], systems: ["rca", "deeplog_sequence", "frequent_pattern"] };
  const TORCH_TIER = ["srcnn", "diffusion_ad"];
  Object.entries(TIER).forEach(([par, names]) => names.forEach((n) => CAPS.push(C("tier-" + n, n + " · " + par.replace(/_/g, " "), "Tier detectors (individual)", "POST /api/v1/detect/tier", (d) => ({ data: series(d), subset: [n], contamination: 0.1 }), { per: 1 }))));
  TORCH_TIER.forEach((n) => CAPS.push(C("tier-" + n, n + " · needs torch on the server", "Tier detectors (individual)", "POST /api/v1/detect/tier", (d) => ({ data: series(d), subset: [n], contamination: 0.1 }), { per: 1 })));
  const H = (id, label, arrays, need) => C("haz-" + id, label, "Hazards · PNG", "POST /api/v1/hazard/visualize", (d) => ({ hazard: id, format: "png", arrays: arrays(d), params: {} }), { block: need || null, img: 1 });
  CAPS.push(
    H("earthquake", "Earthquake · waveform", (d) => ({ series: series(d) }), (d) => ((d ? d.matrix.length : 16) < 550 ? "needs ≥550 samples (one 5.5 s STA/LTA window at 100 Hz)" : null)),
    H("tsunami", "Tsunami · sea-level series", (d) => ({ series: series(d) })),
    H("schumann", "Schumann resonance · series", (d) => ({ series: series(d) })),
    H("meteor", "Meteor · radar returns", (d) => ({ radar_series: series(d) })),
    H("tornado", "Tornado · Doppler sweeps × 64 gates", (d) => ({ radar_sequence: matrix(d) }), (d) => (!d || d.cols !== 64 ? "needs a (sweeps × 64) matrix" : null)),
    H("volcanic", "Volcanic · seismic (timesteps × 32)", (d) => ({ seismic_sequence: matrix(d) }), (d) => (!d || d.cols !== 32 ? "needs a (timesteps × 32) matrix" : null)),
    H("landslide", "Landslide · 64 slope features", (d) => ({ slope_features: matrix(d)[0] }), (d) => (!d || d.cols !== 64 ? "needs one row of 64 features" : null)),
    H("hurricane", "Hurricane · wind_u + wind_v grids", () => ({}), () => "needs wind_u and wind_v 2-D grids (use API → hazard form)"),
    H("wildfire", "Wildfire · thermal image (3 × H × W)", () => ({}), () => "needs a 3-channel thermal image (use API → hazard form)")
  );
  const STMT = { statements: [{ statement_id: "s1", speaker: "w1", made_at: 100, claims: [{ subject: "w1", predicate: "location", value: "depot", interval: [0, 50] }], details: [{ text: "saw a red van leave", verifiable: true }] }], evidence: [0, 1].map((i) => ({ evidence_id: "e" + i, target: { subject: "w1", predicate: "location" }, stance: "supports", reliability: 0.9, source_id: "cam" + i, time: [0, 50] })) };
  const INTEL = { open_source: { report_id: "r1", source_id: "A", text: "convoy at the crossing" }, signals: { report_id: "r2", source_id: "B", derived_from: ["r1"], text: "convoy" } };
  CAPS.push(
    C("asm-stmt", "Statements · corroboration & contradictions", "Assessment · needs detect permission", "POST /api/v1/assessment/statements/verify", () => JSON.parse(JSON.stringify(STMT)), { block: () => (window.MercuryCLIAPI && window.MercuryCLIAPI.cfg && window.MercuryCLIAPI.cfg.apiKey ? null : "needs an API key with the detect permission — set it in the server connection dialog"), fixture: "tests/api/test_assessment_routes.py" }),
    C("asm-screen", "Intel screening · source independence", "Assessment · needs detect permission", "POST /api/v1/assessment/intel/screen", () => ({ intel_reports: JSON.parse(JSON.stringify(INTEL)) }), { block: () => (window.MercuryCLIAPI && window.MercuryCLIAPI.cfg && window.MercuryCLIAPI.cfg.apiKey ? null : "needs an API key with the detect permission — set it in the server connection dialog"), fixture: "tests/api/test_assessment_routes.py" }),
    C("asm-fuse", "Intel fusion · evidence-weighted", "Assessment · needs detect permission", "POST /api/v1/assessment/intel/fuse", () => ({ intel_reports: { open_source: { confidence: 0.8, threat_score: 0.6 } } }), { block: () => (window.MercuryCLIAPI && window.MercuryCLIAPI.cfg && window.MercuryCLIAPI.cfg.apiKey ? null : "needs an API key with the detect permission — set it in the server connection dialog"), fixture: "tests/api/test_assessment_routes.py" }),
    C("agent-speak", "Agent status · voice", "Agent", "POST /api/v1/voice/speak", (d, ctx) => ({ message: "status", domain: ctx.domain || null, session_id: null }), { rec: 1 })
  );
  const BATCH = ["univariate", "multivariate", "fusion", "statistical", "temporal", "neurosymbolic"];

  const arr = (x) => (Array.isArray(x) ? x : null);
  function summarize(data) {
    const d = data || {}, kv = [];
    const scores = arr(d.scores) || (arr(d.anomaly_scores) && d.anomaly_scores.every((x) => typeof x === "number") ? d.anomaly_scores : null) || (arr(d.anomaly_scores) ? d.anomaly_scores.map((x) => x && (x.score ?? x.anomaly_score ?? x.value)).filter((x) => typeof x === "number") : null) || (arr(d.anomaly_prob) ? d.anomaly_prob : null);
    const flags = arr(d.anomalies) || arr(d.is_anomaly) || arr(d.flags) || (scores && typeof d.threshold === "number" ? scores.map((s) => s >= d.threshold) : null);
    let headline = "";
    if (flags) { const n = flags.filter(Boolean).length; headline = n + " of " + flags.length + " flagged"; }
    else if (typeof d.is_anomaly === "boolean") headline = d.is_anomaly ? "ANOMALY" : "NORMAL";
    else if (typeof d.fusion_score === "number") headline = "fusion " + d.fusion_score.toFixed(3) + (d.is_stable === false ? " · UNSTABLE" : "");
    else if (typeof d.overall_score === "number") headline = "score " + d.overall_score.toFixed(3) + (d.decision ? " · " + d.decision : "");
    else if (Array.isArray(d.root_causes) && d.root_causes.length) headline = "top cause " + (d.root_causes[0].node || d.root_causes[0].name || "?") + (d.root_causes[0].score != null ? " · " + (+d.root_causes[0].score).toFixed(3) : "");
    else if (d.image) headline = (d.content_type || "image") + (d.bytes ? " · " + Math.round(d.bytes / 1024) + " KB" : "");
    else if (d.type === "FeatureCollection") headline = (d.features || []).length + " GeoJSON features";
    else if (d.assessment && Array.isArray(d.assessment.independence)) headline = d.assessment.independence.filter((r) => r.independent).length + " of " + d.assessment.independence.length + " reports independent";
    else if (d.decides_with) headline = "decides with " + d.decides_with + (d.fused && d.fused.threat_score != null ? " · threat " + d.fused.threat_score : "");
    else if (Array.isArray(d.statements)) headline = d.statements.length + " statement" + (d.statements.length === 1 ? "" : "s") + " assessed" + (d.statements[0] && d.statements[0].corroboration ? " · log-odds " + d.statements[0].corroboration.log_odds : "");
    else if (typeof d.message === "string") headline = d.message.length > 90 ? d.message.slice(0, 88) + "…" : d.message;
    else if (d.status) headline = String(d.status);
    [["fused_score", 3], ["confidence", 3], ["threshold", 3], ["overall_score", 3], ["overall_confidence", 3], ["fusion_score", 3], ["recursion_score", 3], ["resonance_score", 3], ["lyapunov_bound", 3], ["neural_contribution", 3], ["symbolic_contribution", 3]].forEach(([k, p]) => { if (typeof d[k] === "number") kv.push([k, d[k].toFixed(p)]); });
    ["method", "decision", "is_stable", "patterns_detected", "rules_fired", "threshold_source"].forEach((k) => { if (d[k] != null && typeof d[k] !== "object") kv.push([k, String(d[k])]); });
    if (d.decision && typeof d.decision === "object") kv.push(["decision", d.decision.disposition || JSON.stringify(d.decision).slice(0, 80)]);
    if (d.detector_scores && typeof d.detector_scores === "object") Object.entries(d.detector_scores).slice(0, 6).forEach(([k, v]) => kv.push(["· " + k, typeof v === "number" ? v.toFixed(3) : String(v)]));
    const rc = arr(d.root_causes) || arr(d.ranking) || arr(d.top_k) || arr(d.causes);
    if (rc) rc.slice(0, 5).forEach((x, i) => kv.push(["cause " + (i + 1), typeof x === "object" ? (x.node || x.name || x.feature || JSON.stringify(x)).toString().slice(0, 60) + (x.score != null ? " · " + (+x.score).toFixed(3) : "") : String(x)]));
    if (d.summary && typeof d.summary === "object") Object.entries(d.summary).slice(0, 4).forEach(([k, v]) => kv.push([k, typeof v === "object" ? JSON.stringify(v).slice(0, 60) : String(v)]));
    if (d.explanation) kv.push(["explanation", String(d.explanation).slice(0, 220)]);
    const image = d.image_base64 || d.png_base64 || d.image || null;
    return { scores, flags, headline: headline || "response received", kv, image: image && typeof image === "string" ? (image.startsWith("data:") ? image : "data:image/png;base64," + image) : null, geojson: d.type === "FeatureCollection" ? (d.features || []).length + " features" : null };
  }
  function consensus(results, n) {
    const per = results.filter((r) => r.sum && r.sum.flags && r.sum.flags.length === n);
    const votes = Array.from({ length: n }, (_, i) => per.reduce((a, r) => a + (r.sum.flags[i] ? 1 : 0), 0));
    const top = votes.map((v, i) => ({ i, v })).filter((x) => x.v > 0).sort((a, b) => b.v - a.v || a.i - b.i).slice(0, 10);
    return { voters: per.length, votes, top, majority: votes.filter((v) => per.length && v > per.length / 2).length };
  }
  async function run(A, caps, dataset, ctx, onStep, conc) {
    const out = caps.map((c) => ({ id: c.id, label: c.label, route: c.route, status: "queued" }));
    const queue = caps.map((_, i) => i), N = Math.max(1, Math.min(conc || 2, 4)), RL_MAX = 6;
    const worker = async () => { while (queue.length) { if (ctx.cancelled && ctx.cancelled()) return; const k = queue.shift(), c = caps[k], o = out[k]; const why = c.block ? c.block(dataset) : null; if (why) { o.status = "skipped"; o.error = why; onStep(out.slice()); continue; } o.status = "running"; onStep(out.slice()); const body = c.build(dataset, ctx); o.body = body; const r = await A.post(c.route.split(" ")[1], body, { signal: ctx.signal }); o.ms = r.meta ? r.meta.ms : null; o.http = r.status; o.corr = (r.meta && r.meta.correlationId) || null;
      if (!r.ok && r.kind === "ratelimit" && (o.rl = (o.rl || 0) + 1) <= RL_MAX && !(ctx.cancelled && ctx.cancelled())) {
        // The engine's token bucket (burst 20, 100/min) is smaller than a full mass run and the client's own
        // 3 attempts can all land inside the drained window. Honour Retry-After, park this capability at the
        // tail of the queue and slow this worker down instead of recording a false failure.
        const ra = r.meta && r.meta.retryAfter != null ? parseFloat(r.meta.retryAfter) : NaN; const wait = Math.min(15000, (isNaN(ra) ? 1000 : Math.max(1000, ra * 1000)) * o.rl);
        o.status = "queued"; o.error = "rate limited · retry " + o.rl + " of " + RL_MAX + " in " + Math.round(wait / 1000) + " s"; queue.push(k); onStep(out.slice());
        await new Promise((ok) => setTimeout(ok, wait)); continue;
      }
      if (r.ok) { o.status = "ok"; delete o.error; o.data = r.data; o.sum = summarize(r.data); } else { o.status = r.kind === "cancelled" ? "cancelled" : "error"; o.error = (r.kind || "") + " · " + (typeof r.error === "string" ? r.error : JSON.stringify(r.error)); } if (r.meta && r.meta.attempts > 1) o.attempts = r.meta.attempts; onStep(out.slice()); } };
    await Promise.all(Array.from({ length: N }, worker));
    return out;
  }
  async function batch(A, methods, dataset, onStep, cancelled, pollMs) {
    const rows = dataset ? dataset.matrix : SAMPLE.map((v) => [v]); const out = methods.map((m) => ({ id: "batch-" + m, label: "Batch job · " + m, route: "POST /api/v1/batch/detect", status: "queued" }));
    for (let i = 0; i < methods.length; i++) { const o = out[i]; o.status = "submitting"; onStep(out.slice()); const sub = await A.batchSubmit({ data: rows, method: methods[i], sensitivity: 0.5, chunk_size: 1000, priority: 5 }); if (!sub.ok) { o.status = "error"; o.http = sub.status; o.error = (sub.kind || "") + " · " + sub.error; onStep(out.slice()); continue; } o.job = sub.data.job_id; o.status = "pending"; onStep(out.slice());
      for (let t = 0; t < 120; t++) { if (cancelled && cancelled()) { await A.batchCancel(o.job); o.status = "cancelled"; break; } if (pollMs !== 0) await new Promise((r) => setTimeout(r, pollMs || 1000)); const st = await A.batchStatus(o.job); if (!st.ok) { o.status = "error"; o.error = st.error; break; } o.progress = st.data.progress_percentage; o.status = st.data.status; onStep(out.slice()); if (["completed", "failed", "cancelled", "partial"].includes(st.data.status)) break; }
      if (o.status === "completed" || o.status === "partial") { let rs = await A.batchResults(o.job, 0, 1000); if (rs.ok) { const all = (rs.data.results || []).slice(); let off = all.length, pages = 1; while (rs.ok && rs.data.has_more && pages < 500) { rs = await A.batchResults(o.job, off, 1000); if (!rs.ok) break; all.push.apply(all, rs.data.results || []); off = all.length; pages++; o.status = "fetching " + pages + " pages"; onStep(out.slice()); } rs = { ok: true, data: Object.assign({}, rs.data, { results: all, total_results: all.length, has_more: false, pages_fetched: pages }) }; o.data = rs.data; const flat = (rs.data.results || []).map((x) => x && (x.is_anomaly ?? (Array.isArray(x.anomalies) ? x.anomalies[0] : null))); o.sum = { flags: flat.every((x) => typeof x === "boolean") ? flat : null, scores: null, headline: rs.data.total_results + " results · " + o.status, kv: [["job_id", o.job], ["status", o.status], ["results", String(rs.data.total_results)]] }; o.status = "ok"; } }
      onStep(out.slice()); }
    return out;
  }
  function script(A, caps, dataset, ctx) {
    const base = (A.cfg.baseUrl || "http://127.0.0.1:8000").replace(/\/+$/, ""), key = A.cfg.apiKey ? ' -H "X-API-Key: $MERCURY_API_KEY"' : "";
    const L = ["#!/usr/bin/env bash", "# Mercury Agent · mass run · generated by Mercury CLI console", "# Requires: mercury-agent serve (pip install \"mercury-agent-global[api]\"), curl, jq", "set -euo pipefail", 'BASE="${MERCURY_BASE:-' + base + '}"', "mkdir -p mercury-mass-run && cd mercury-mass-run", ""];
    caps.forEach((c) => { const path = c.route.split(" ")[1]; L.push("# " + c.label, "cat > " + c.id + ".req.json <<'JSON'", JSON.stringify(c.build(dataset, ctx)), "JSON", 'curl -sS -X POST "$BASE' + path + '"' + key + ' -H "Content-Type: application/json" -d @' + c.id + ".req.json -o " + c.id + ".res.json -w '" + c.id + " %{http_code} %{time_total}s\\n'", ""); });
    L.push("jq -s 'map({file: input_filename})' *.res.json >/dev/null 2>&1 || true", 'echo "Responses saved in $(pwd)"');
    return L.join("\n");
  }
  function mcpScript(A, tools, argsFor) {
    const lines = [{ jsonrpc: "2.0", id: 0, method: "initialize", params: { protocolVersion: "2024-11-05", capabilities: {}, clientInfo: { name: "mercury-cli-console", version: "1" } } }, { jsonrpc: "2.0", method: "notifications/initialized" }].concat(tools.map((t, i) => ({ jsonrpc: "2.0", id: i + 1, method: "tools/call", params: { name: t, arguments: argsFor(t) } })));
    return ["#!/usr/bin/env bash", "# Mercury Agent · MCP tools over stdio · generated by Mercury CLI console", "# Requires: pip install mercury-agent-global (mercury-agent on PATH)", "set -euo pipefail", "cat > mcp-requests.jsonl <<'JSONL'"].concat(lines.map((l) => JSON.stringify(l)), ["JSONL", '{ cat mcp-requests.jsonl; sleep "$' + '{MCP_WAIT:-20}"; } | mercury-agent mcp > mcp-results.jsonl', 'echo "Responses in $(pwd)/mcp-results.jsonl. Load it with Session > Import MCP results."']).join("\n");
  }
  function mcpImport(text, tools) { const out = []; text.split(/\r?\n/).forEach((l) => { if (!l.trim()) return; let j; try { j = JSON.parse(l); } catch (e) { return; } if (!j || j.id == null || j.id === 0) return; const name = tools[j.id - 1] || "tool #" + j.id; let data = j.result || { error: j.error }; const txt = j.result && Array.isArray(j.result.content) ? j.result.content.map((c) => c.text || "").join("\n") : null; if (txt) { try { data = JSON.parse(txt); } catch (e) { data = { text: txt }; } } out.push({ label: name, route: "MCP tools/call " + name, status: j.error || (j.result && j.result.isError) ? "error" : "ok", data }); }); return out; }
  window.MercuryCLIMass = { CAPS, BATCH, TIER, TORCH_TIER, SAMPLE, summarize, consensus, run, batch, script, mcpScript, mcpImport, series, matrix };
})();

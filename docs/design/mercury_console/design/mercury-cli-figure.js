/* Mercury Agent presence. A luminous figure drawn on a canvas. Its layout is the traced reference (mercury-cli-figure-trace.js);
   its look and motion are generated here. Each motion is tied to something the console is doing:
     activity   requests in flight. api.js dispatches mcli:call start and end for every call; the figure speeds up and brightens.
     signal     the last reply's confidence, or the last detection's flagged share. It sets how bright the core burns.
     pointer    while the pointer is over the figure, nearby particles lean toward it on springs; the head turns to follow.
     shockwave  each finished request sends a ring out from the core. Gold on success, red on failure.
     state      engine status. Offline and unreachable dim the figure; simulated is labelled as simulated.
     click      focuses the chat input, so the figure is also the way into the agent.
   Ambient drift (flow along the threads, twinkling stars) runs on top and stops under reduced motion. */
(function () {
  "use strict";

  var IMG_W = 495, IMG_H = 1198;
  var CV = { el: null, ctx: null, w: 0, h: 0, dpr: 1, cap: null, bound: null, P: null, stars: null, layout: null };
  var S = { inflight: 0, activity: 0, signal: 0.5, haveSignal: false, lastEnd: 0, lastCall: null, pulses: [], healthOk: null,
    ptr: null, focus: null, lastDraw: 0, lastCap: 0, lastKey: "", now: 0 };

  function reduced() {
    try {
      return document.documentElement.classList.contains("mn-reduce-motion") ||
        !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
    } catch (e) { return false; }
  }
  function status() {
    var A = window.MercuryCLIAPI;
    if (!A || !A.connected || !A.connected()) return "offline";
    if (A.simulated && A.simulated()) return "simulated";
    return S.healthOk === false ? "unreachable" : "live";
  }
  function clamp(v, a, b) { return v < a ? a : v > b ? b : v; }

  function decode(b64) {
    var bin = atob(b64), u = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
    return u;
  }
  function buildSeeds() {
    var T = window.MERCURY_FIGURE_TRACE;
    if (!T) return null;
    var u = decode(T.seeds), n = Math.floor(u.length / 8);
    var P = { n: n, hx: new Float32Array(n), hy: new Float32Array(n), tx: new Float32Array(n), ty: new Float32Array(n),
      ridge: new Float32Array(n), bright: new Float32Array(n), kind: new Uint8Array(n), ph: new Float32Array(n),
      dx: new Float32Array(n), dy: new Float32Array(n), vx: new Float32Array(n), vy: new Float32Array(n) };
    for (var i = 0; i < n; i++) {
      var o = i * 8;
      P.hx[i] = (u[o] | (u[o + 1] << 8)) / 65535;
      P.hy[i] = (u[o + 2] | (u[o + 3] << 8)) / 65535;
      var a = u[o + 4] / 255 * Math.PI * 2;
      P.tx[i] = Math.cos(a); P.ty[i] = Math.sin(a);
      P.ridge[i] = u[o + 5] / 255; P.bright[i] = u[o + 6] / 255; P.kind[i] = u[o + 7];
      P.ph[i] = (i * 2.399963) % (Math.PI * 2);
    }
    var sb = decode(T.starBytes || ""), sn = Math.floor(sb.length / 5), stars = [];
    for (var k = 0; k < sn; k++) {
      var q = k * 5;
      stars.push({ x: (sb[q] | (sb[q + 1] << 8)) / 65535, y: (sb[q + 2] | (sb[q + 3] << 8)) / 65535, m: sb[q + 4] / 255, ph: (k * 1.618) % 6.283 });
    }
    // procedural faint stars so the field is never sparse
    var seed = 9176; function r() { seed = (seed * 16807) % 2147483647; return seed / 2147483647; }
    for (var j = 0; j < 140; j++) stars.push({ x: r(), y: r(), m: 0.15 + r() * 0.35, ph: r() * 6.283 });
    CV.stars = stars;
    // the continuous threads: count8, brightness8, then (x16, y16) per point
    if (T.lines) {
      var lb = decode(T.lines), o2 = 0, lens = [], starts = [], brs = [], tot = 0, rawx = [], rawy = [];
      while (o2 + 2 <= lb.length) {
        var cnt = lb[o2], br = lb[o2 + 1] / 255; o2 += 2;
        if (o2 + cnt * 4 > lb.length) break;
        lens.push(cnt); starts.push(tot); brs.push(br);
        for (var z = 0; z < cnt; z++) { rawx.push((lb[o2] | (lb[o2 + 1] << 8)) / 65535); rawy.push((lb[o2 + 2] | (lb[o2 + 3] << 8)) / 65535); o2 += 4; }
        tot += cnt;
      }
      var LN = { count: lens.length, total: tot, len: Uint8Array.from(lens), start: Int32Array.from(starts), br: Float32Array.from(brs),
        hx: Float32Array.from(rawx), hy: Float32Array.from(rawy), nx: new Float32Array(tot), ny: new Float32Array(tot),
        dx: new Float32Array(tot), dy: new Float32Array(tot), vx: new Float32Array(tot), vy: new Float32Array(tot),
        px: new Float32Array(tot), py: new Float32Array(tot), ph: new Float32Array(lens.length) };
      for (var li0 = 0; li0 < LN.count; li0++) {
        LN.ph[li0] = (li0 * 2.399963) % 6.283;
        var s0 = LN.start[li0], n0 = LN.len[li0];
        for (var k0 = 0; k0 < n0; k0++) {
          var a0 = s0 + Math.max(0, k0 - 1), b0 = s0 + Math.min(n0 - 1, k0 + 1);
          var tx0 = (LN.hx[b0] - LN.hx[a0]) * IMG_W, ty0 = (LN.hy[b0] - LN.hy[a0]) * IMG_H, tl = Math.sqrt(tx0 * tx0 + ty0 * ty0) || 1;
          LN.nx[s0 + k0] = -ty0 / tl; LN.ny[s0 + k0] = tx0 / tl;
        }
      }
      CV.LN = LN;
    }
    return P;
  }

  function layout() {
    var w = CV.w, h = CV.h, s = Math.min(h * 0.95 / IMG_H, w * 0.95 / IMG_W);
    CV.layout = { s: s, x0: (w - IMG_W * s) / 2, y0: (h - IMG_H * s) / 2 };
    var L = CV.layout;
    var core = window.MERCURY_FIGURE_TRACE && window.MERCURY_FIGURE_TRACE.core ? window.MERCURY_FIGURE_TRACE.core : [0.5, 0.49];
    L.cx = L.x0 + core[0] * IMG_W * s; L.cy = L.y0 + core[1] * IMG_H * s;
    L.hx = L.x0 + 0.5 * IMG_W * s; L.hy = L.y0 + 0.27 * IMG_H * s;
  }

  function resize() {
    var cv = CV.el, dpr = Math.min(2, window.devicePixelRatio || 1);
    var w = cv.clientWidth, h = cv.clientHeight;
    if (!w || !h) return false;
    if (w !== CV.w || h !== CV.h || dpr !== CV.dpr) {
      CV.w = w; CV.h = h; CV.dpr = dpr;
      cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
      CV.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      layout();
    }
    return true;
  }

  var BUCKETS = 8, COLORS = [[255, 198, 112], [250, 212, 146], [255, 244, 214]];
  function drawFrame(t, nowIn) {
    var now = Math.max(nowIn, performance.now());
    var ctx = CV.ctx, w = CV.w, h = CV.h, P = CV.P, L = CV.layout, s = L.s;
    var act = S.activity, sig = S.haveSignal ? S.signal : 0.5, live = status();
    var dim = (live === "offline" || live === "unreachable") ? 0.6 : 1;
    var ptr = null;
    if (S.ptr && CV.el) { var rc = CV.el.getBoundingClientRect(); ptr = { x: S.ptr.x - rc.left, y: S.ptr.y - rc.top }; }
    var over = ptr && ptr.x >= 0 && ptr.y >= 0 && ptr.x <= w && ptr.y <= h;

    // background: the reference's night, deep blue with a warm glow behind the body
    var bgG = ctx.createLinearGradient(0, 0, 0, h);
    bgG.addColorStop(0, "rgb(2,8,30)"); bgG.addColorStop(0.5, "rgb(7,24,70)"); bgG.addColorStop(1, "rgb(2,7,24)");
    ctx.globalCompositeOperation = "source-over"; ctx.globalAlpha = 1; ctx.fillStyle = bgG; ctx.fillRect(0, 0, w, h);
    var nb = ctx.createRadialGradient(L.hx, (L.hy + L.cy) / 2, 0, L.hx, (L.hy + L.cy) / 2, s * 520);
    nb.addColorStop(0, "rgba(46,96,200," + (0.26 * dim).toFixed(3) + ")"); nb.addColorStop(0.55, "rgba(30,70,160," + (0.12 * dim).toFixed(3) + ")"); nb.addColorStop(1, "rgba(30,70,160,0)");
    ctx.fillStyle = nb; ctx.fillRect(0, 0, w, h);
    for (var si = 0; si < CV.stars.length; si++) {
      var st = CV.stars[si], tw = 0.55 + 0.45 * Math.sin(t * (0.6 + si % 5 * 0.2) + st.ph);
      ctx.fillStyle = "rgba(240,244,255," + Math.min(1, (0.25 + st.m * 0.9) * tw * (0.6 + 0.4 * dim)).toFixed(3) + ")";
      var sz = 0.9 + st.m * 1.4;
      ctx.fillRect(st.x * w, st.y * h, sz, sz);
    }

    // shockwaves, one per finished request, expanding from the core
    var shocks = [];
    for (var pu = S.pulses.length - 1; pu >= 0; pu--) {
      var p0 = S.pulses[pu], age = Math.max(0, (now - p0.t) / 1300);
      if (age > 1) { S.pulses.splice(pu, 1); continue; }
      shocks.push({ r: (0.05 + age * 0.95) * IMG_H * s * 0.62, a: 1 - age, ok: p0.ok, k: 0.9 * (1 - age) });
    }

    // particles: each sits on its traced home, moves along the local thread, and is pulled by the pointer and the shocks
    var n = P.n, pr = 62 * s + 40 * act * s, flowAmp = (0.6 + 1.1 * act) * 7.5 * s, spd = 0.5 + act * 1.7;
    var BK = CV.bk || (CV.bk = (function () { var o = { bx0: [], by0: [], bx1: [], by1: [], dotX: [], dotY: [] }; for (var c = 0; c < 3 * BUCKETS; c++) { o.bx0.push([]); o.by0.push([]); o.bx1.push([]); o.by1.push([]); o.dotX.push([]); o.dotY.push([]); } return o; })());
    var bx0 = BK.bx0, by0 = BK.by0, bx1 = BK.bx1, by1 = BK.by1, dotX = BK.dotX, dotY = BK.dotY;
    for (var cc = 0; cc < 3 * BUCKETS; cc++) { bx0[cc].length = 0; by0[cc].length = 0; bx1[cc].length = 0; by1[cc].length = 0; dotX[cc].length = 0; dotY[cc].length = 0; }
    for (var i = 0; i < n; i++) {
      var hx = L.x0 + P.hx[i] * IMG_W * s, hy = L.y0 + P.hy[i] * IMG_H * s;
      var k = P.kind[i], rid = P.ridge[i];
      var f = Math.sin(t * spd + P.ph[i]) * flowAmp * (k === 0 ? 0.6 + 0.6 * rid : k === 1 ? 0.5 : 0.25);
      var tgx = P.tx[i] * f, tgy = P.ty[i] * f;
      if (k === 2) { var br = Math.sin(t * (1.1 + act) + P.ph[i] * 3) * 1.6 * s; tgx += Math.cos(P.ph[i]) * br; tgy += Math.sin(P.ph[i]) * br; }
      var ax = 0, ay = 0;
      if (over) {
        var ddx = ptr.x - (hx + P.dx[i]), ddy = ptr.y - (hy + P.dy[i]), dist = Math.sqrt(ddx * ddx + ddy * ddy);
        if (dist < pr && dist > 0.01) { var kk = Math.pow(1 - dist / pr, 2) * 0.55; ax += ddx * kk; ay += ddy * kk; }
      }
      for (var q = 0; q < shocks.length; q++) {
        var sh = shocks[q], rx = hx - L.cx, ry = hy - L.cy, rd = Math.sqrt(rx * rx + ry * ry) + 0.001;
        var band = Math.abs(rd - sh.r); if (band < 16 * s) { var pushK = (1 - band / (16 * s)) * sh.k * 3.2 * s; ax += rx / rd * pushK; ay += ry / rd * pushK; }
      }
      var tX = tgx + ax, tY = tgy + ay;
      P.vx[i] = (P.vx[i] + (tX - P.dx[i]) * 0.12) * 0.78;
      P.vy[i] = (P.vy[i] + (tY - P.dy[i]) * 0.12) * 0.78;
      P.dx[i] += P.vx[i]; P.dy[i] += P.vy[i];
      var X = hx + P.dx[i], Y = hy + P.dy[i];
      var al = (0.22 + 0.62 * P.bright[i] * (0.45 + 0.55 * sig)) * dim * (k === 2 ? 1.05 : k === 1 ? 0.7 : (CV.LN ? 0.42 : 0.95));
      var bkt = clamp(Math.floor(al * BUCKETS), 0, BUCKETS - 1);
      var col = k === 2 ? 2 : k === 1 ? 1 : 0, ci = col * BUCKETS + bkt;
      if (k === 2 || rid < 0.18) { dotX[ci].push(X); dotY[ci].push(Y); }
      else {
        var Lh = (9 + 16 * rid) * s * (0.85 + 0.35 * act);
        bx0[ci].push(X - P.tx[i] * Lh); by0[ci].push(Y - P.ty[i] * Lh); bx1[ci].push(X + P.tx[i] * Lh); by1[ci].push(Y + P.ty[i] * Lh);
      }
    }
    ctx.globalCompositeOperation = "source-over";
    for (var b2 = 0; b2 < 3 * BUCKETS; b2++) {
      var colI = Math.floor(b2 / BUCKETS), aB = ((b2 % BUCKETS) + 0.5) / BUCKETS, RGB = COLORS[colI];
      var rgba = "rgba(" + RGB[0] + "," + RGB[1] + "," + RGB[2] + "," + aB.toFixed(3) + ")";
      var sx = bx0[b2];
      if (sx.length) {
        ctx.lineCap = "round";
        ctx.globalCompositeOperation = "lighter"; ctx.strokeStyle = "rgba(96,150,255," + (aB * 0.16).toFixed(3) + ")"; ctx.lineWidth = Math.max(3, 7.5 * s + 1.5);
        ctx.beginPath();
        for (var g1 = 0; g1 < sx.length; g1++) { ctx.moveTo(bx0[b2][g1], by0[b2][g1]); ctx.lineTo(bx1[b2][g1], by1[b2][g1]); }
        ctx.stroke();
        ctx.globalCompositeOperation = "source-over"; ctx.strokeStyle = "rgba(" + RGB[0] + "," + RGB[1] + "," + RGB[2] + "," + (aB * 0.22).toFixed(3) + ")"; ctx.lineWidth = Math.max(2, 4.2 * s + 1);
        ctx.beginPath();
        for (var g2 = 0; g2 < sx.length; g2++) { ctx.moveTo(bx0[b2][g2], by0[b2][g2]); ctx.lineTo(bx1[b2][g2], by1[b2][g2]); }
        ctx.stroke();
        ctx.strokeStyle = rgba; ctx.lineWidth = Math.max(0.6, 0.9 * s + 0.35);
        ctx.beginPath();
        for (var m = 0; m < sx.length; m++) { ctx.moveTo(bx0[b2][m], by0[b2][m]); ctx.lineTo(bx1[b2][m], by1[b2][m]); }
        ctx.stroke();
      }
      var dx2 = dotX[b2];
      if (dx2.length) {
        var sz2 = colI === 2 ? 2.4 * s + 0.8 : colI === 1 ? 1.4 * s + 0.4 : 1.2 * s + 0.4;
        ctx.fillStyle = rgba; ctx.beginPath();
        for (var m2 = 0; m2 < dx2.length; m2++) ctx.rect(dotX[b2][m2] - sz2 / 2, dotY[b2][m2] - sz2 / 2, sz2, sz2);
        ctx.fill();
      }
    }

    // threads: the traced curves, drawn as smooth paths with a pulse of light running along each
    var LN = CV.LN;
    if (LN) {
      var swayAmp = (1.2 + 1.8 * act) * s, spdL = 0.5 + act;
      for (var li = 0; li < LN.count; li++) {
        var st0 = LN.start[li], lnn = LN.len[li], ph0 = LN.ph[li];
        for (var pj = 0; pj < lnn; pj++) {
          var q2 = st0 + pj, hx2 = L.x0 + LN.hx[q2] * IMG_W * s, hy2 = L.y0 + LN.hy[q2] * IMG_H * s;
          var sw = Math.sin(t * spdL + ph0 + pj * 0.45) * swayAmp;
          var tgx2 = sw * LN.nx[q2], tgy2 = sw * LN.ny[q2], ax2 = 0, ay2 = 0;
          if (over) {
            var ex = ptr.x - (hx2 + LN.dx[q2]), ey = ptr.y - (hy2 + LN.dy[q2]), ed = Math.sqrt(ex * ex + ey * ey);
            if (ed < pr && ed > 0.01) { var ek = Math.pow(1 - ed / pr, 2) * 0.55; ax2 += ex * ek; ay2 += ey * ek; }
          }
          for (var q3 = 0; q3 < shocks.length; q3++) {
            var sh3 = shocks[q3], rx3 = hx2 - L.cx, ry3 = hy2 - L.cy, rd3 = Math.sqrt(rx3 * rx3 + ry3 * ry3) + 0.001, bd3 = Math.abs(rd3 - sh3.r);
            if (bd3 < 16 * s) { var pk3 = (1 - bd3 / (16 * s)) * sh3.k * 3.2 * s; ax2 += rx3 / rd3 * pk3; ay2 += ry3 / rd3 * pk3; }
          }
          LN.vx[q2] = (LN.vx[q2] + (tgx2 + ax2 - LN.dx[q2]) * 0.12) * 0.78; LN.vy[q2] = (LN.vy[q2] + (tgy2 + ay2 - LN.dy[q2]) * 0.12) * 0.78;
          LN.dx[q2] += LN.vx[q2]; LN.dy[q2] += LN.vy[q2];
          LN.px[q2] = hx2 + LN.dx[q2]; LN.py[q2] = hy2 + LN.dy[q2];
        }
      }
      function tracePath(st, ln) {
        ctx.moveTo(LN.px[st], LN.py[st]);
        for (var k = 1; k < ln - 1; k++) { var a = st + k, b = a + 1; ctx.quadraticCurveTo(LN.px[a], LN.py[a], (LN.px[a] + LN.px[b]) / 2, (LN.py[a] + LN.py[b]) / 2); }
        ctx.lineTo(LN.px[st + ln - 1], LN.py[st + ln - 1]);
      }
      ctx.lineCap = "round"; ctx.lineJoin = "round";
      var passes = [["lighter", 0, 7.5, 0.10], ["source-over", 1, 3.2, 0.22], ["source-over", 2, 1.1, 1.0]];
      for (var pa = 0; pa < passes.length; pa++) {
        var P2 = passes[pa];
        ctx.globalCompositeOperation = P2[0];
        for (var bk = 0; bk < 4; bk++) {
          var base = (0.46 + 0.22 * bk) * (0.65 + 0.35 * sig) * dim, col2 = P2[1] === 0 ? "96,150,255" : P2[1] === 1 ? "255,196,110" : (bk >= 2 ? "255,236,176" : "255,214,140");
          ctx.strokeStyle = "rgba(" + col2 + "," + (clamp(base * P2[3] * (P2[1] === 0 ? 0.6 : 1) * (0.85 + 0.4 * act), 0, 1)).toFixed(3) + ")";
          ctx.lineWidth = Math.max(P2[1] === 2 ? 0.7 : 1.2, P2[2] * s + (P2[1] === 2 ? 0.3 : 1));
          ctx.beginPath();
          for (var lj = 0; lj < LN.count; lj++) { if (Math.min(3, Math.floor(LN.br[lj] * 4)) !== bk) continue; tracePath(LN.start[lj], LN.len[lj]); }
          ctx.stroke();
        }
      }
      // a bright pulse travels along every second thread; it quickens with activity
      ctx.globalCompositeOperation = "lighter";
      ctx.fillStyle = "rgba(255,244,216," + (0.55 * dim).toFixed(3) + ")"; ctx.beginPath();
      for (var lp = 0; lp < LN.count; lp += 3) {
        var uu = (t * (0.05 + 0.07 * ((lp % 7) / 7)) * (1 + 2.2 * act) + LN.ph[lp] / 6.283) % 1, fi = uu * (LN.len[lp] - 1), i0 = Math.floor(fi), fr = fi - i0, ia = LN.start[lp] + i0, ib = Math.min(LN.start[lp] + LN.len[lp] - 1, ia + 1);
        var ppx = LN.px[ia] + (LN.px[ib] - LN.px[ia]) * fr, ppy = LN.py[ia] + (LN.py[ib] - LN.py[ia]) * fr, rr0 = Math.max(0.3, 0.8 * s + 0.55);
        ctx.moveTo(ppx + rr0, ppy); ctx.arc(ppx, ppy, rr0, 0, 6.283);
      }
      ctx.fill();
      ctx.globalCompositeOperation = "source-over";
    }

    // light sources: the head halo and the core, both burning with the signal and activity
    var hg = ctx.createRadialGradient(L.hx, L.hy, 0, L.hx, L.hy, s * 150);
    hg.addColorStop(0, "rgba(255,222,160," + (0.14 + 0.08 * act).toFixed(3) + ")"); hg.addColorStop(0.6, "rgba(120,170,255," + (0.07 + 0.05 * act).toFixed(3) + ")"); hg.addColorStop(1, "rgba(100,150,255,0)");
    ctx.fillStyle = hg; ctx.fillRect(L.hx - s * 150, L.hy - s * 150, s * 300, s * 300);
    var coreI = (0.55 + 0.45 * sig) * dim, cg = ctx.createRadialGradient(L.cx, L.cy, 0, L.cx, L.cy, s * 124);
    cg.addColorStop(0, "rgba(255,252,232," + (coreI).toFixed(3) + ")"); cg.addColorStop(0.12, "rgba(255,236,170," + (coreI * 0.92).toFixed(3) + ")");
    cg.addColorStop(0.4, "rgba(255,196,120," + (coreI * 0.40).toFixed(3) + ")"); cg.addColorStop(0.7, "rgba(110,160,255," + (coreI * 0.14).toFixed(3) + ")"); cg.addColorStop(1, "rgba(90,140,255,0)");
    ctx.fillStyle = cg; ctx.fillRect(L.cx - s * 124, L.cy - s * 124, s * 248, s * 248);

    // rings of energy turning around the core while work is in flight
    ctx.globalCompositeOperation = "lighter";
    for (var ar = 0; ar < 3; ar++) {
      var rr = s * (46 + ar * 22), sp = t * (0.35 + ar * 0.2) * (1 + act * 1.6) * (ar % 2 ? 1 : -1);
      ctx.strokeStyle = "rgba(255,230,170," + ((0.10 + 0.32 * act + 0.12 * sig) * dim).toFixed(3) + ")"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.ellipse(L.cx, L.cy, rr, rr * 0.84, sp, 0.25, 4.6); ctx.stroke();
    }
    for (var sh2 = 0; sh2 < shocks.length; sh2++) {
      var z = shocks[sh2], colS = z.ok ? "255,230,150" : "255,96,96";
      ctx.strokeStyle = "rgba(" + (z.ok ? "170,205,255" : colS) + "," + (z.a * 0.75 * dim).toFixed(3) + ")"; ctx.lineWidth = 1.4;
      ctx.beginPath(); ctx.arc(L.cx, L.cy, z.r, 0, 6.283); ctx.stroke();
    }

    // hover: the figure reports its own state where the pointer rests on the core or the head
    ctx.globalCompositeOperation = "source-over";
    if (over && (Math.hypot(ptr.x - L.cx, ptr.y - L.cy) < 80 * s || Math.hypot(ptr.x - L.hx, ptr.y - L.hy) < 80 * s)) {
      var lines = [live.toUpperCase() + " · activity " + S.activity.toFixed(2) + " · signal " + (S.haveSignal ? S.signal.toFixed(2) : "none yet")];
      if (S.lastCall) lines.push(S.lastCall);
      lines.push("click to ask the agent");
      ctx.font = "10px ui-monospace, Menlo, Consolas, monospace";
      var tw2 = 0; lines.forEach(function (ln) { tw2 = Math.max(tw2, ctx.measureText(ln).width); });
      var bx = clamp(ptr.x + 14, 4, w - tw2 - 16), by = clamp(ptr.y + 14, 4, h - lines.length * 14 - 8);
      ctx.fillStyle = "rgba(4,12,34,0.82)"; ctx.fillRect(bx - 6, by - 4, tw2 + 12, lines.length * 14 + 6);
      ctx.strokeStyle = "rgba(255,214,128,0.5)"; ctx.lineWidth = 1; ctx.strokeRect(bx - 6.5, by - 4.5, tw2 + 13, lines.length * 14 + 7);
      ctx.fillStyle = "rgba(240,236,220,0.95)";
      lines.forEach(function (ln, li) { ctx.fillText(ln, bx, by + 10 + li * 14); });
    }
    if (live === "offline" || live === "unreachable") {
      ctx.fillStyle = "rgba(4,10,30,0.25)"; ctx.fillRect(0, 0, w, h);
    }
  }

  function caption(t) {
    if (!CV.cap || t - S.lastCap < 400) return;
    S.lastCap = t;
    var live = status(), label = { live: "LIVE", simulated: "SIMULATED ENGINE", offline: "NO ENGINE", unreachable: "ENGINE UNREACHABLE" }[live];
    var sig = S.haveSignal ? S.signal.toFixed(2) : "none yet";
    var text = label + " · activity " + S.activity.toFixed(2) + " · signal " + sig + " · in flight " + S.inflight;
    if (text !== S.lastKey) { CV.cap.textContent = text; S.lastKey = text; }
  }

  function bind(cv) {
    if (CV.bound === cv) return;
    CV.bound = cv;
    cv.addEventListener("click", function () {
      var input = document.querySelector('input[placeholder="Ask Mercury Agent…"]');
      if (input) input.focus();
    });
    cv.style.cursor = "pointer";
  }

  function frame(now) {
    var cv = document.getElementById("mcli-presence");
    if (cv) {
      if (cv !== CV.el) {
        CV.el = cv; CV.ctx = cv.getContext("2d"); CV.cap = document.getElementById("mcli-presence-cap");
        CV.w = 0; CV.P = CV.P || buildSeeds(); bind(cv);
      }
      if (CV.P && resize()) {
        var target = S.inflight > 0 ? 1 : (now - S.lastEnd < 1500 ? 0.35 : 0);
        S.activity += (target - S.activity) * 0.08;
        var still = reduced(), t = still ? 0 : now / 1000;
        var due = still ? now - S.lastDraw > 900 : now - S.lastDraw >= 33;
        if (due) { var t0f = performance.now(); try { drawFrame(t, now); } catch (err) { S.errors = (S.errors || 0) + 1; S.lastError = String(err && err.message || err); try { CV.ctx.globalCompositeOperation = 'source-over'; CV.ctx.globalAlpha = 1; } catch (e2) { /* context lost */ } } S.lastMs = performance.now() - t0f; S.lastDraw = now; }
        caption(now);
      }
    }
    requestAnimationFrame(frame);
  }

  window.addEventListener("mcli:call", function (ev) {
    var d = ev.detail || {}, now = performance.now();
    if (d.phase === "start") { S.inflight++; return; }
    if (d.phase !== "end") return;
    S.inflight = Math.max(0, S.inflight - 1); S.lastEnd = now;
    S.lastCall = (d.method || "") + " " + (d.path || "") + " · HTTP " + (d.status != null ? d.status : (d.ok ? 200 : "error"));
    S.pulses.push({ t: now, ok: !!d.ok }); if (S.pulses.length > 10) S.pulses.shift();
    if (d.path === "/health") S.healthOk = !!d.ok;
    if (d.ok && d.data && typeof d.data === "object") {
      var conf = typeof d.data.confidence === "number" ? d.data.confidence : null;
      var flags = Array.isArray(d.data.anomalies) ? d.data.anomalies : Array.isArray(d.data.is_anomaly) ? d.data.is_anomaly : null;
      if (conf != null) { S.signal = clamp(conf, 0, 1); S.haveSignal = true; }
      else if (flags && flags.length) { S.signal = flags.filter(Boolean).length / flags.length; S.haveSignal = true; }
      else if (typeof d.data.is_anomaly === "boolean") { S.signal = d.data.is_anomaly ? 1 : 0; S.haveSignal = true; }
    }
  });
  window.addEventListener("pointermove", function (e) { S.ptr = { x: e.clientX, y: e.clientY }; });
  document.addEventListener("mouseout", function (e) { if (!e.relatedTarget) S.ptr = null; });
  window.addEventListener("blur", function () { S.ptr = null; });

  window.MercuryPresence = {
    state: function () {
      return { status: status(), activity: +S.activity.toFixed(3), inflight: S.inflight, signal: S.haveSignal ? +S.signal.toFixed(3) : null,
        pulses: S.pulses.length, last: S.lastCall, seeds: CV.P ? CV.P.n : 0, stars: CV.stars ? CV.stars.length : 0, frameMs: S.lastMs ? +S.lastMs.toFixed(2) : null, errors: S.errors || 0, lastError: S.lastError || null };
    }
  };
  requestAnimationFrame(frame);
})();

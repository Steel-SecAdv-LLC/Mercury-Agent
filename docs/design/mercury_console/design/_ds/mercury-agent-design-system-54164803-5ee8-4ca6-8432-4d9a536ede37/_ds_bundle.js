/* @ds-bundle: {"format":3,"namespace":"MercuryAgentDesignSystem_541648","components":[],"sourceHashes":{"components/mercury-agent-dock.js":"3178dd91628d","components/mercury-command-palette.js":"558503f4a1f8","components/mercury-icon.js":"74bca514fe73","components/mercury-nav.js":"79890439b526","components/mercury-quarantine.js":"4d107b87e16b","components/mercury-settings.js":"2879a62c2b60","components/mercury-state.js":"349d6a4e04d4","components/mercury-status.js":"38e5095496f4","design-canvas.jsx":"d3ddcf4241b9","preview/agent_dock.js":"929b8cb358e4","review/review_data.js":"b87368368f6a","ui_kits/cli/App.jsx":"0a630b3a29bb","ui_kits/cli/Banner.jsx":"03816afc770b","ui_kits/cli/Chart.jsx":"fa2f5a09dad7","ui_kits/cli/CliAgent.jsx":"fcbed9495ec9","ui_kits/cli/Lines.jsx":"5d8ced694e77","ui_kits/cli/Stream.jsx":"4e13d1bcc619","ui_kits/console/Analytics.jsx":"834276fcc6ef","ui_kits/console/Capitalize.jsx":"852c2a1d1183","ui_kits/console/Chrome.jsx":"d7221ab6d7f2","ui_kits/console/ConsoleApp.jsx":"3d560b8b6fba","ui_kits/console/ConsoleData.jsx":"b9a6b6d9d040","ui_kits/console/Host.jsx":"7f4b482e078f","ui_kits/console/LiveViews.jsx":"27b38339c4fb","ui_kits/console/MercuryAPI.jsx":"e7e65b76629d","ui_kits/console/Panels.jsx":"2596dcd5eeef","ui_kits/console/Views.jsx":"2c820e49fc3d","ui_kits/crypto/CryptoAnalytics.jsx":"6e83c46a045e","ui_kits/crypto/CryptoApp.jsx":"8f5134af86af","ui_kits/crypto/CryptoData.jsx":"c14632db0306","ui_kits/crypto/CryptoPanels.jsx":"a48421bc47db","ui_kits/deck/deck-stage.js":"eac2199dccb4","ui_kits/explorer/Explorer.jsx":"8425acc3ea6f","ui_kits/explorer/ExplorerData.js":"31caa4ca8e4f","ui_kits/hazard/Hazard.jsx":"8ce2dd86d07e","ui_kits/hazard/HazardData.js":"d7b805e2cebf","ui_kits/hazard/HazardMap.jsx":"e9066a812160"},"inlinedExternals":[],"unexposedExports":[]} */

(() => {

const __ds_ns = (window.MercuryAgentDesignSystem_541648 = window.MercuryAgentDesignSystem_541648 || {});

const __ds_scope = {};

(__ds_ns.__errors = __ds_ns.__errors || []);

// components/mercury-agent-dock.js
try { (() => {
/* ============================================================================
   <mercury-agent-dock> — Mercury Agent · the canonical host dock
   ----------------------------------------------------------------------------
   ONE implementation of the signature interaction — the ♱ pill that expands a
   small agent panel — for every surface: preview cards, the Console, the Crypto
   explorer, the CLI, and the Detection & Audit Explorer. Replaces the three
   forked docks the review flagged (item K1 · interconnectedness): one behavior,
   one voice, half the maintenance.

   Two ways to use it ----------------------------------------------------------
   1) Declarative + JS property (preferred for app surfaces):
        <mercury-agent-dock></mercury-agent-dock>
        document.querySelector('mercury-agent-dock').context = {
          title: "Detection & Audit Explorer",
          blurb: "What you're looking at, in one line.",
          status: { label: "online", tone: "normal" },
          prompts: [{ label: "...", answer: "..." }],   // up to 4
          links:   [{ label: "Console", href: "../console/index.html" }],
          position: "bottom-right"
        };
   2) Back-compat auto-mount (the 30+ preview cards already do this):
        <script>window.MA_CONTEXT = { title, blurb, prompts };</script>
        <script src="mercury-agent-dock.js"></script>
      If MA_CONTEXT is present and no <mercury-agent-dock> is in the DOM, one is
      created automatically — so legacy cards keep working unchanged.

   Reads tokens from colors_and_type.css; defines no palette of its own.
   ============================================================================ */
(function () {
  if (window.customElements && customElements.get("mercury-agent-dock")) return;
  var INJECTED = false;
  function injectCSS() {
    if (INJECTED) return;
    INJECTED = true;
    var css = "" + "mercury-agent-dock{position:fixed;z-index:9999;}" + ".mad{font-family:var(--font-mono,monospace);}" + ".mad-pill{display:inline-flex;align-items:center;gap:7px;cursor:pointer;user-select:none;" + "background:var(--surface-2,#151c26);color:var(--fg,#e6edf3);border:1px solid var(--mercury-800,#0a5d6b);" + "border-radius:999px;padding:6px 12px;font-size:10px;letter-spacing:.1em;box-shadow:0 6px 20px rgba(0,0,0,.4);transition:all .14s;}" + ".mad-pill:hover{border-color:var(--mercury-500,#00bcd4);box-shadow:0 0 0 1px rgba(0,188,212,.4),0 6px 20px rgba(0,0,0,.5);}" + ".mad-pill .s{color:var(--brand-gold,#B4B124);font-size:13px;}" + ".mad-pill .st{width:6px;height:6px;border-radius:50%;background:var(--normal,#2fd686);box-shadow:0 0 7px var(--normal,#2fd686);animation:madpulse 1.5s infinite;}" + "@keyframes madpulse{50%{opacity:.4;}}" + ".mad-panel{position:absolute;width:290px;display:none;overflow-y:auto;overflow-x:hidden;box-sizing:border-box;max-height:calc(100vh - 66px);" + "background:var(--surface-1,#0f141c);border:1px solid var(--border-strong,#36465a);border-radius:8px;box-shadow:0 18px 50px rgba(0,0,0,.6);}" + ".mad.open .mad-panel{display:block;}" + ".mad.open .mad-pill{background:var(--mercury-900,#0c3b44);border-color:var(--mercury-500,#00bcd4);}" + ".mad-h{display:flex;align-items:center;gap:8px;padding:10px 12px;border-bottom:1px solid var(--border-faint,#161e28);" + "background:linear-gradient(180deg,rgba(0,188,212,.08),transparent);}" + ".mad-h .s{font-size:17px;color:var(--brand-gold,#B4B124);text-shadow:0 0 12px rgba(180,177,36,.5);}" + ".mad-h .t{font-size:11px;letter-spacing:.06em;color:var(--fg,#e6edf3);}" + ".mad-h .x{margin-left:auto;cursor:pointer;color:var(--fg-3,#6b7888);font-size:13px;padding:2px 5px;border-radius:3px;}" + ".mad-h .x:hover{background:var(--surface-3,#1c2530);color:var(--fg,#e6edf3);}" + ".mad-state{display:flex;align-items:center;gap:6px;padding:7px 12px;font-size:9px;letter-spacing:.1em;color:var(--fg-3,#6b7888);border-bottom:1px solid var(--border-faint,#161e28);}" + ".mad-state .d{width:6px;height:6px;border-radius:50%;}" + ".mad-msg{padding:11px 12px;font-family:var(--font-sans,sans-serif);font-size:12px;line-height:1.55;color:var(--fg-2,#a9b6c6);" + "background:var(--bg,#0a0e14);border-left:2px solid var(--mercury-600,#00a0b6);margin:11px 12px;border-radius:2px 6px 6px 6px;}" + ".mad-msg b{color:var(--mercury-200,#8ae3f5);}" + ".mad-p{display:flex;flex-wrap:wrap;gap:6px;padding:0 12px 11px;}" + ".mad-c{font-size:10px;color:var(--mercury-200,#8ae3f5);background:var(--surface-2,#151c26);border:1px solid var(--mercury-800,#0a5d6b);" + "border-radius:999px;padding:5px 10px;cursor:pointer;transition:all .12s;}" + ".mad-c:hover{background:var(--mercury-900,#0c3b44);border-color:var(--mercury-500,#00bcd4);}" + ".mad-links{display:flex;flex-wrap:wrap;gap:6px;padding:0 12px 11px;border-top:1px solid var(--border-faint,#161e28);margin-top:2px;padding-top:10px;}" + ".mad-links .lbl{width:100%;font-size:8px;letter-spacing:.16em;color:var(--fg-muted,#495567);text-transform:uppercase;margin-bottom:1px;}" + ".mad-link{font-size:10px;color:var(--fg-2,#a9b6c6);background:transparent;border:1px solid var(--border,#243040);border-radius:var(--r-xs,3px);" + "padding:4px 9px;cursor:pointer;text-decoration:none;transition:all .12s;display:inline-flex;align-items:center;gap:5px;}" + ".mad-link:hover{border-color:var(--mercury-600,#00a0b6);color:var(--mercury-200,#8ae3f5);}" + ".mad-link .a{color:var(--mercury-400,#18c8e0);}" + ".mad-ask{display:flex;align-items:center;gap:7px;margin:0 12px 10px;padding:5px 6px 5px 10px;" + "background:var(--bg,#0a0e14);border:1px solid var(--border,#243040);border-radius:999px;transition:border-color .14s;}" + ".mad-ask:focus-within{border-color:var(--mercury-500,#00bcd4);box-shadow:0 0 0 1px rgba(0,188,212,.4);}" + ".mad-ask .qs{font-size:12px;color:var(--brand-gold,#B4B124);}" + ".mad-ask input{flex:1;min-width:0;background:transparent;border:none;outline:none;color:var(--fg,#e6edf3);font-family:var(--font-sans,sans-serif);font-size:11.5px;}" + ".mad-ask input::placeholder{color:var(--fg-muted,#495567);}" + ".mad-ask button{flex:none;width:23px;height:23px;border-radius:50%;border:none;cursor:pointer;background:var(--mercury-500,#00bcd4);color:var(--bg,#0a0e14);font-size:11px;line-height:1;}" + ".mad-ask button:hover{background:var(--mercury-300,#4fc3f7);}" + ".mad-q{margin:0 12px 9px;padding:6px 11px;max-width:80%;margin-left:auto;width:fit-content;font-family:var(--font-sans,sans-serif);font-size:11px;" + "color:var(--fg,#e6edf3);background:var(--surface-3,#1c2530);border:1px solid var(--border-strong,#36465a);border-radius:8px 8px 2px 8px;}" + ".mad-f{padding:8px 12px;border-top:1px solid var(--border-faint,#161e28);font-size:8.5px;letter-spacing:.04em;color:var(--fg-muted,#495567);}" + ".mad-wm .g{color:var(--brand-gold,#B4B124);}.mad-wm .c{color:var(--brand-cyan,#11AEED);}.mad-wm .ig{color:var(--brand-ink-grey,#8B97A6);}";
    var st = document.createElement("style");
    st.id = "mercury-agent-dock-css";
    st.textContent = css;
    document.head.appendChild(st);
  }
  var WM = '<span class="mad-wm"><span class="g">M</span><span class="c">ERCURY</span> <span class="g">A</span><span class="ig">GENT</span></span>';
  var DOTC = {
    anomaly: "var(--anomaly)",
    normal: "var(--normal)",
    warning: "var(--warning)",
    cognitive: "var(--cognitive)",
    mercury: "var(--mercury-500)",
    neutral: "var(--fg-3)"
  };
  function tokens(s) {
    return (s || "").toLowerCase().replace(/[^a-z0-9\s]/g, " ").split(/\s+/).filter(Boolean);
  }
  var STOP = {
    the: 1,
    a: 1,
    an: 1,
    is: 1,
    are: 1,
    of: 1,
    to: 1,
    in: 1,
    on: 1,
    for: 1,
    what: 1,
    how: 1,
    do: 1,
    i: 1,
    this: 1,
    it: 1,
    and: 1,
    with: 1,
    me: 1,
    my: 1,
    you: 1
  };
  var MercuryAgentDock = class extends HTMLElement {
    connectedCallback() {
      injectCSS();
      this._ctx = this._ctx || {};
      this.render();
    }
    set context(v) {
      this._ctx = v || {};
      if (this.isConnected) this.render();
    }
    get context() {
      return this._ctx;
    }
    render() {
      var C = this._ctx || {};
      var title = C.title || "Mercury Agent";
      var blurb = C.blurb || "I host this design system. Ask me what this surface is, how to use it, or where it comes from in the codebase.";
      var prompts = C.prompts && C.prompts.length ? C.prompts.slice(0, 4) : [{
        label: "What is this?",
        answer: blurb
      }, {
        label: "How do I use it?",
        answer: "Pull tokens from <b>colors_and_type.css</b> and follow the rules in <b>README.md</b>."
      }];
      var links = C.links || [];
      var status = C.status || {
        label: "online",
        tone: "normal"
      };
      var pos = C.position || "bottom-right";
      var vert = pos.indexOf("top") === 0 ? "top" : "bottom";
      var horiz = pos.indexOf("left") >= 0 ? "left" : "right";
      this.style[vert] = "14px";
      this.style[horiz] = "14px";
      this.style[vert === "top" ? "bottom" : "top"] = "auto";
      this.style[horiz === "left" ? "right" : "left"] = "auto";
      var pc = "";
      for (var i = 0; i < prompts.length; i++) pc += '<span class="mad-c" data-i="' + i + '">' + prompts[i].label + "</span>";
      var lc = "";
      if (links.length) {
        lc = '<div class="mad-links"><span class="lbl">jump to surface</span>';
        for (var j = 0; j < links.length; j++) lc += '<a class="mad-link" href="' + links[j].href + '"><span class="a">→</span>' + links[j].label + "</a>";
        lc += "</div>";
      }
      var panelStyle = vert + ":42px;" + horiz + ":0;";
      this.innerHTML = '<div class="mad">' + '<div class="mad-panel" style="' + panelStyle + '">' + '<div class="mad-h"><span class="s">♱</span><span class="t">' + title + '</span><span class="x" data-x>✕</span></div>' + '<div class="mad-state"><span class="d" style="background:' + (DOTC[status.tone] || DOTC.normal) + ';box-shadow:0 0 7px ' + (DOTC[status.tone] || DOTC.normal) + '"></span>' + status.label + '</div>' + '<div class="mad-msg" data-msg>' + blurb + '</div>' + '<div class="mad-p">' + pc + '</div>' + lc + '<form class="mad-ask" data-ask><span class="qs">♱</span><input type="text" placeholder="Ask Mercury Agent…" autocomplete="off"><button type="submit" aria-label="Send">▸</button></form>' + '<div class="mad-f mad-wm">' + WM + ' · host · one dock across every surface</div>' + '</div>' + '<div class="mad-pill mad-wm"><span class="st"></span><span class="s">♱</span>' + WM + '</div>' + '</div>';
      var dock = this.querySelector(".mad");
      var panel = this.querySelector(".mad-panel");
      var promptsRow = this.querySelector(".mad-p");
      var msg = this.querySelector("[data-msg]");
      var self = this;
      this.querySelector(".mad-pill").addEventListener("click", function () {
        dock.classList.toggle("open");
      });
      this.querySelector("[data-x]").addEventListener("click", function () {
        dock.classList.remove("open");
      });
      function clearQ() {
        var q = panel.querySelector(".mad-q");
        if (q) q.remove();
      }
      function setMsg(html) {
        msg.innerHTML = html;
      }
      this.querySelectorAll(".mad-c").forEach(function (c) {
        c.addEventListener("click", function () {
          clearQ();
          setMsg(prompts[+c.getAttribute("data-i")].answer);
        });
      });
      function route(query) {
        var qt = tokens(query).filter(function (t) {
          return !STOP[t];
        });
        if (!qt.length) return null;
        var best = -1,
          bestScore = 0;
        for (var k = 0; k < prompts.length; k++) {
          var pt = tokens(prompts[k].label + " " + prompts[k].answer),
            score = 0;
          qt.forEach(function (t) {
            if (pt.indexOf(t) >= 0) score++;
          });
          if (score > bestScore) {
            bestScore = score;
            best = k;
          }
        }
        return bestScore > 0 ? prompts[best] : null;
      }
      this.querySelector("[data-ask]").addEventListener("submit", function (e) {
        e.preventDefault();
        var input = this.querySelector("input");
        var text = (input.value || "").trim();
        if (!text) return;
        input.value = "";
        clearQ();
        var bubble = document.createElement("div");
        bubble.className = "mad-q";
        bubble.textContent = text;
        panel.insertBefore(bubble, promptsRow);
        var hit = route(text);
        setMsg(hit ? hit.answer : "I can speak to <b>this surface</b> — tap a chip for what I know, or jump to another surface below. For live detection open the <b>Console</b> or the <b>Explorer</b>.");
      });
    }
  };
  customElements.define("mercury-agent-dock", MercuryAgentDock);

  // ---- public API + back-compat auto-mount ----
  window.MercuryDock = {
    mount: function (ctx) {
      var el = document.createElement("mercury-agent-dock");
      document.body.appendChild(el);
      el.context = ctx || {};
      return el;
    }
  };
  function autoMount() {
    if (window.MA_CONTEXT && !document.querySelector("mercury-agent-dock")) {
      window.MercuryDock.mount(window.MA_CONTEXT);
    }
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", autoMount);else autoMount();
})();
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/mercury-agent-dock.js", error: String((e && e.message) || e) }); }

// components/mercury-command-palette.js
try { (() => {
/* ============================================================================
   <mercury-command-palette> — Mercury Agent · ⌘K command surface
   ----------------------------------------------------------------------------
   The keyboard-first command surface the review's UI/UX command #5 asked for
   (UI copy & command shortcuts). One ⌘K / Ctrl-K palette any kit can populate:
   run a detection, jump a surface, set sensitivity, export a decision, file an
   override. Short verb-led labels, ALL-CAPS where the brand uses them, no emoji.

   Usage:
     <mercury-command-palette></mercury-command-palette>
     const pal = document.querySelector('mercury-command-palette');
     pal.commands = [
       { id:"run",    group:"Detect",  label:"Run detection",    hint:"univariate", glyph:"▸", run(){...} },
       { id:"console",group:"Navigate",label:"Open Console",     kbd:"G C",        glyph:"→", run(){ location.href="../console/index.html"; } },
     ];
   Open with ⌘K (mac) / Ctrl-K, or pal.open(). ↑/↓ move · Enter runs · Esc closes.
   Emits a 'command' CustomEvent (detail = the command) in addition to run().
   Reads tokens from colors_and_type.css.
   ============================================================================ */
(function () {
  if (window.customElements && customElements.get("mercury-command-palette")) return;
  var INJECTED = false;
  function injectCSS() {
    if (INJECTED) return;
    INJECTED = true;
    var css = "" + "mercury-command-palette{position:fixed;inset:0;z-index:11000;display:none;}" + "mercury-command-palette[open]{display:block;}" + ".mcp-scrim{position:absolute;inset:0;background:rgba(6,8,11,.72);backdrop-filter:blur(3px);}" + ".mcp-box{position:absolute;top:14vh;left:50%;transform:translateX(-50%);width:min(580px,92vw);" + "background:var(--surface-1,#0f141c);border:1px solid var(--border-strong,#36465a);border-radius:var(--r-lg,10px);" + "box-shadow:0 28px 80px rgba(0,0,0,.7);overflow:hidden;font-family:var(--font-mono,monospace);}" + ".mcp-in{display:flex;align-items:center;gap:10px;padding:14px 16px;border-bottom:1px solid var(--border,#243040);}" + ".mcp-in .s{color:var(--brand-gold,#B4B124);font-size:15px;}" + ".mcp-in input{flex:1;background:transparent;border:none;outline:none;color:var(--fg,#e6edf3);" + "font-family:var(--font-mono,monospace);font-size:15px;letter-spacing:.01em;}" + ".mcp-in input::placeholder{color:var(--fg-muted,#495567);}" + ".mcp-kbd{font-size:9px;letter-spacing:.1em;color:var(--fg-muted,#495567);border:1px solid var(--border,#243040);border-radius:var(--r-xs,3px);padding:3px 6px;}" + ".mcp-list{max-height:46vh;overflow-y:auto;padding:6px;}" + ".mcp-grp{font-size:8.5px;letter-spacing:.18em;text-transform:uppercase;color:var(--mercury-300,#4fc3f7);padding:11px 12px 6px;}" + ".mcp-item{display:flex;align-items:center;gap:12px;padding:9px 12px;border-radius:var(--r-sm,4px);cursor:pointer;border:1px solid transparent;}" + ".mcp-item.sel{background:var(--surface-3,#1c2530);border-color:var(--mercury-800,#0a5d6b);}" + ".mcp-item.sel{box-shadow:inset 2px 0 0 var(--mercury-500,#00bcd4);}" + ".mcp-gl{flex:none;width:22px;text-align:center;color:var(--mercury-400,#18c8e0);font-size:13px;}" + ".mcp-lab{flex:1;color:var(--fg,#e6edf3);font-size:13px;letter-spacing:.01em;}" + ".mcp-hint{color:var(--fg-3,#6b7888);font-size:11px;}" + ".mcp-kb{font-size:9px;letter-spacing:.08em;color:var(--fg-3,#6b7888);border:1px solid var(--border,#243040);border-radius:var(--r-xs,3px);padding:2px 6px;}" + ".mcp-empty{padding:26px 14px;text-align:center;color:var(--fg-muted,#495567);font-size:12px;}" + ".mcp-foot{display:flex;gap:16px;align-items:center;padding:9px 14px;border-top:1px solid var(--border-faint,#161e28);" + "font-size:9px;letter-spacing:.06em;color:var(--fg-muted,#495567);}" + ".mcp-foot .sp{margin-left:auto;}" + ".mcp-foot .wm .g{color:var(--brand-gold,#B4B124);}.mcp-foot .wm .c{color:var(--brand-cyan,#11AEED);}.mcp-foot .wm .ig{color:var(--brand-ink-grey,#8B97A6);}";
    var st = document.createElement("style");
    st.id = "mercury-command-palette-css";
    st.textContent = css;
    document.head.appendChild(st);
  }
  var WM = '<span class="wm"><span class="g">M</span><span class="c">A</span></span>';
  var MercuryCommandPalette = class extends HTMLElement {
    connectedCallback() {
      injectCSS();
      this._cmds = this._cmds || [];
      this._q = "";
      this._sel = 0;
      this._build();
      this._onKey = this._onKey.bind(this);
      document.addEventListener("keydown", this._onKey);
    }
    disconnectedCallback() {
      document.removeEventListener("keydown", this._onKey);
    }
    set commands(v) {
      this._cmds = v || [];
      if (this.hasAttribute("open")) this._renderList();
    }
    get commands() {
      return this._cmds;
    }
    _onKey(e) {
      var k = e.key.toLowerCase();
      if ((e.metaKey || e.ctrlKey) && k === "k") {
        e.preventDefault();
        this.hasAttribute("open") ? this.close() : this.open();
        return;
      }
      if (!this.hasAttribute("open")) return;
      if (k === "escape") {
        e.preventDefault();
        this.close();
      } else if (k === "arrowdown") {
        e.preventDefault();
        this._move(1);
      } else if (k === "arrowup") {
        e.preventDefault();
        this._move(-1);
      } else if (k === "enter") {
        e.preventDefault();
        this._runSel();
      }
    }
    open() {
      this.setAttribute("open", "");
      this._q = "";
      this._sel = 0;
      this._input.value = "";
      this._renderList();
      setTimeout(() => this._input.focus(), 0);
    }
    close() {
      this.removeAttribute("open");
    }
    _filtered() {
      var q = this._q.trim().toLowerCase();
      if (!q) return this._cmds.slice();
      return this._cmds.filter(function (c) {
        return (c.label + " " + (c.group || "") + " " + (c.hint || "")).toLowerCase().indexOf(q) >= 0;
      });
    }
    _move(d) {
      var n = this._filtered().length;
      if (!n) return;
      this._sel = (this._sel + d + n) % n;
      this._paintSel();
    }
    _paintSel() {
      var items = this._list.querySelectorAll(".mcp-item");
      items.forEach((el, i) => el.classList.toggle("sel", i === this._sel));
      var cur = items[this._sel];
      if (cur) cur.scrollIntoView({
        block: "nearest"
      });
    }
    _runSel() {
      var f = this._filtered();
      var c = f[this._sel];
      if (!c) return;
      this.close();
      this.dispatchEvent(new CustomEvent("command", {
        detail: c,
        bubbles: true
      }));
      if (typeof c.run === "function") c.run();
    }
    _build() {
      this.innerHTML = '<div class="mcp-scrim" data-scrim></div>' + '<div class="mcp-box">' + '<div class="mcp-in"><span class="s">♱</span><input type="text" placeholder="Type a command…  run · jump · set · export" autocomplete="off"><span class="mcp-kbd">ESC</span></div>' + '<div class="mcp-list" data-list></div>' + '<div class="mcp-foot"><span>↑↓ move</span><span>↵ run</span><span class="sp"></span><span class="wm">' + WM + '</span><span>command palette</span></div>' + '</div>';
      this._input = this.querySelector("input");
      this._list = this.querySelector("[data-list]");
      var self = this;
      this.querySelector("[data-scrim]").addEventListener("click", function () {
        self.close();
      });
      this._input.addEventListener("input", function () {
        self._q = this.value;
        self._sel = 0;
        self._renderList();
      });
    }
    _renderList() {
      var f = this._filtered();
      if (!f.length) {
        this._list.innerHTML = '<div class="mcp-empty">No command matches “' + this._q + '”.</div>';
        return;
      }
      var html = "",
        lastGrp = null,
        idx = 0,
        self = this;
      f.forEach(function (c) {
        if ((c.group || "") !== lastGrp) {
          lastGrp = c.group || "";
          if (lastGrp) html += '<div class="mcp-grp">' + lastGrp + "</div>";
        }
        html += '<div class="mcp-item' + (idx === self._sel ? " sel" : "") + '" data-idx="' + idx + '">' + '<span class="mcp-gl">' + (c.glyph || "▸") + "</span>" + '<span class="mcp-lab">' + c.label + "</span>" + (c.hint ? '<span class="mcp-hint">' + c.hint + "</span>" : "") + (c.kbd ? '<span class="mcp-kb">' + c.kbd + "</span>" : "") + "</div>";
        idx++;
      });
      this._list.innerHTML = html;
      this._list.querySelectorAll(".mcp-item").forEach(function (el) {
        el.addEventListener("mouseenter", function () {
          self._sel = +el.getAttribute("data-idx");
          self._paintSel();
        });
        el.addEventListener("click", function () {
          self._sel = +el.getAttribute("data-idx");
          self._runSel();
        });
      });
    }
  };
  customElements.define("mercury-command-palette", MercuryCommandPalette);
})();
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/mercury-command-palette.js", error: String((e && e.message) || e) }); }

// components/mercury-icon.js
try { (() => {
/* ============================================================================
   <mercury-icon> — Mercury Agent · the canonical functional-icon atom
   ----------------------------------------------------------------------------
   The companion to <mercury-status>. A thin generator over a fixed set of
   single-stroke outline icons (the system's flagged Lucide-style substitution)
   that map 1:1 onto real product surfaces — the σ gate, GOSNN, PQC, fusion,
   conformal coverage, anomaly, drift, audit, privacy, keys.

   It injects NO colors of its own. By default every icon inherits
   `currentColor`, so it tints itself from whatever context it sits in (a nav
   item, a button, a red incident card). When a state genuinely carries meaning,
   bind it to ONE of the documented status tokens via `tone=` — never a new hue.
   Where a native brand glyph already exists (σ for the sigma gate), `glyph`
   renders the glyph instead, since the system prefers glyphs to chrome icons.

   Usage (plain custom element — works in HTML or JSX):
     <mercury-icon name="anomaly-alert"></mercury-icon>              inherits currentColor
     <mercury-icon name="anomaly-alert" tone="anomaly"></mercury-icon>
     <mercury-icon name="sigma-gate" tone="auto"></mercury-icon>     icon's documented tone
     <mercury-icon name="sigma-gate" glyph tone="auto"></mercury-icon>  native σ glyph
     <mercury-icon name="model-drift" size="28" label="Model drift detected"></mercury-icon>

   Attributes
     name   — required. One of the 15 concept names (kebab-case) or an alias.
     tone   — explicit hue: anomaly|warning|normal|verified|cognitive|
              benevolence|mercury|withheld|neutral. Special: "auto" uses the
              icon's DOCUMENTED default tone. Omit → inherit currentColor.
     size   — px (number). Default 20.
     stroke — stroke width. Default 2.
     glyph  — boolean; render the native glyph where one exists (else the icon).
     label  — accessible name → role="img" + <title>. Omit ⇒ decorative
              (aria-hidden="true"), the right default for icons beside text.
   Requires colors_and_type.css to be loaded on the page.
   ============================================================================ */
(function () {
  if (window.customElements && customElements.get("mercury-icon")) return;

  // tone token → CSS var. Outline strokes read best on the lighter cyan step;
  // every value here already exists in colors_and_type.css (no new color).
  var TONE = {
    anomaly: "var(--anomaly)",
    warning: "var(--warning)",
    normal: "var(--normal)",
    verified: "var(--verified)",
    cognitive: "var(--cognitive)",
    benevolence: "var(--benevolence)",
    mercury: "var(--mercury-300)",
    withheld: "var(--withheld)",
    neutral: "var(--fg-2)"
  };

  // name → { p: <svg path markup>, tone: documented default, glyph?: char }
  // Paths are 24×24, fill:none, stroke:currentColor (Lucide grammar).
  var ICONS = {
    "sigma-gate": {
      tone: "benevolence",
      glyph: "σ",
      p: '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/>'
    },
    "gosnn-coupling": {
      tone: "cognitive",
      p: '<circle cx="9" cy="12" r="6"/><circle cx="15" cy="12" r="6"/>'
    },
    "pqc-shield": {
      tone: "verified",
      p: '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="M12 22V2"/>'
    },
    "fusion-model": {
      tone: "neutral",
      p: '<path d="M12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/><path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/>'
    },
    "conformal-coverage": {
      tone: "normal",
      p: '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>'
    },
    "integrated-gradients": {
      tone: "mercury",
      p: '<circle cx="12" cy="12" r="3"/><line x1="3" x2="9" y1="12" y2="12"/><line x1="15" x2="21" y1="12" y2="12"/>'
    },
    "attention-provider": {
      tone: "neutral",
      p: '<path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/><path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/><circle cx="12" cy="12" r="1"/><path d="M18.944 12.33a1 1 0 0 0 0-.66 7.5 7.5 0 0 0-13.888 0 1 1 0 0 0 0 .66 7.5 7.5 0 0 0 13.888 0"/>'
    },
    "agentic-action": {
      tone: "mercury",
      p: '<path d="m16.24 7.76-1.804 5.411a2 2 0 0 1-1.265 1.265L7.76 16.24l1.804-5.411a2 2 0 0 1 1.265-1.265z"/><circle cx="12" cy="12" r="10"/>'
    },
    "audit-trail": {
      tone: "neutral",
      p: '<path d="M20 7h-3a2 2 0 0 1-2-2V2"/><path d="M9 18a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h7l4 4v10a2 2 0 0 1-2 2Z"/><path d="M3 7.6v12.8A1.6 1.6 0 0 0 4.6 22h9.8"/>'
    },
    "integrity-verify": {
      tone: "verified",
      p: '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="m9 15 2 2 4-4"/>'
    },
    "anomaly-alert": {
      tone: "anomaly",
      p: '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/>'
    },
    "sandbox-run": {
      tone: "neutral",
      p: '<path d="m7 11 2-2-2-2"/><path d="M11 13h4"/><rect x="3" y="4" width="18" height="16" rx="2"/>'
    },
    "model-drift": {
      tone: "warning",
      p: '<path d="M16 17h6v-6"/><path d="m22 17-8.5-8.5-5 5L2 7"/>'
    },
    "privacy-mask": {
      tone: "withheld",
      p: '<path d="M10.733 5.076a10.744 10.744 0 0 1 11.205 6.575 1 1 0 0 1 0 .696 10.747 10.747 0 0 1-1.444 2.49"/><path d="M14.084 14.158a3 3 0 0 1-4.242-4.242"/><path d="M17.479 17.499a10.75 10.75 0 0 1-15.417-5.151 1 1 0 0 1 0-.696 10.75 10.75 0 0 1 4.446-5.143"/><path d="m2 2 20 20"/>'
    },
    "key-rotation": {
      tone: "verified",
      p: '<path d="M2.586 17.414A2 2 0 0 0 2 18.828V21a1 1 0 0 0 1 1h3a1 1 0 0 0 1-1v-1a1 1 0 0 1 1-1h1a1 1 0 0 0 1-1v-1a1 1 0 0 1 1-1h.172a2 2 0 0 0 1.414-.586l.814-.814a6.5 6.5 0 1 0-4-4z"/><circle cx="16.5" cy="7.5" r=".5" fill="currentColor" stroke="none"/>'
    },
    /* ----- WAYFINDING / CHROME group ----------------------------------------
       The surface-nav icons (Console + Cryptography rails). These are pure
       wayfinding, NOT the functional-concept vocabulary above — they carry no
       documented tone and always inherit currentColor from the nav item, so the
       active/hover color stays the rail's job. Folded in here so the kits stop
       each re-declaring an identical inline-SVG icon generator: ONE registry,
       rendered through <mercury-icon> everywhere. (review item: iconography
       single-source.) */
    "dashboard": {
      tone: "neutral",
      p: '<rect x="3" y="3" width="7" height="9"/><rect x="14" y="3" width="7" height="5"/><rect x="14" y="12" width="7" height="9"/><rect x="3" y="16" width="7" height="5"/>'
    },
    "search": {
      tone: "neutral",
      p: '<path d="m21 21-4.3-4.3"/><circle cx="11" cy="11" r="8"/>'
    },
    "grid": {
      tone: "neutral",
      p: '<rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/>'
    },
    "radial": {
      tone: "neutral",
      p: '<path d="M12 2v4M12 18v4M4.9 4.9l2.8 2.8M16.3 16.3l2.8 2.8M2 12h4M18 12h4M4.9 19.1l2.8-2.8M16.3 7.7l2.8-2.8"/>'
    },
    "globe": {
      tone: "neutral",
      p: '<circle cx="12" cy="12" r="10"/><path d="M12 2a15 15 0 0 1 0 20M12 2a15 15 0 0 0 0 20M2 12h20"/>'
    },
    "chart-line": {
      tone: "neutral",
      p: '<path d="M3 3v18h18"/><path d="m7 14 3-4 4 3 5-7"/>'
    },
    "shield": {
      tone: "neutral",
      p: '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z"/>'
    },
    "check-doc": {
      tone: "neutral",
      p: '<path d="M9 11l3 3 8-8"/><path d="M20 12v6a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h9"/>'
    },
    "layers": {
      tone: "neutral",
      p: '<path d="M12 2 2 7l10 5 10-5-10-5Z"/><path d="m2 17 10 5 10-5"/><path d="m2 12 10 5 10-5"/>'
    },
    "activity": {
      tone: "neutral",
      p: '<path d="M3 12h3l2.5 7 4-15 2.5 8 2-3h4"/>'
    },
    "download": {
      tone: "neutral",
      p: '<path d="M12 3v12"/><path d="m7 10 5 5 5-5"/><path d="M5 21h14"/>'
    },
    "mic": {
      tone: "neutral",
      p: '<rect x="9" y="2" width="6" height="12" rx="3"/><path d="M5 11a7 7 0 0 0 14 0"/><path d="M12 18v3"/>'
    },
    "pulse": {
      tone: "neutral",
      p: '<path d="M3 12h4l2 5 4-14 2 9h6"/>'
    },
    "bars": {
      tone: "neutral",
      p: '<path d="M3 3v18h18"/><rect x="7" y="12" width="3" height="6"/><rect x="12" y="8" width="3" height="10"/><rect x="17" y="5" width="3" height="13"/>'
    },
    "box": {
      tone: "neutral",
      p: '<path d="M12 2 3 7v10l9 5 9-5V7Z"/><path d="M3 7l9 5 9-5"/><path d="M12 22V12"/>'
    },
    "kin": {
      tone: "neutral",
      p: '<circle cx="12" cy="5" r="2.6"/><circle cx="5" cy="18" r="2.6"/><circle cx="19" cy="18" r="2.6"/><path d="M10.3 6.7 6.7 15.7M13.7 6.7l3.6 9M7.6 18h8.8"/>'
    }
  };

  // friendly aliases → canonical name
  var ALIAS = {
    "sigma": "sigma-gate",
    "gate": "sigma-gate",
    "ethics-gate": "sigma-gate",
    "gosnn": "gosnn-coupling",
    "coupling": "gosnn-coupling",
    "federated": "gosnn-coupling",
    "pqc": "pqc-shield",
    "shield": "pqc-shield",
    "fusion": "fusion-model",
    "model": "fusion-model",
    "layers": "fusion-model",
    "conformal": "conformal-coverage",
    "coverage": "conformal-coverage",
    "target": "conformal-coverage",
    "ig": "integrated-gradients",
    "attribution": "integrated-gradients",
    "explain": "integrated-gradients",
    "attention": "attention-provider",
    "agentic": "agentic-action",
    "action": "agentic-action",
    "compass": "agentic-action",
    "audit": "audit-trail",
    "provenance": "audit-trail",
    "integrity": "integrity-verify",
    "checksum": "integrity-verify",
    "verify": "integrity-verify",
    "anomaly": "anomaly-alert",
    "alert": "anomaly-alert",
    "sandbox": "sandbox-run",
    "terminal": "sandbox-run",
    "surrogate": "sandbox-run",
    "drift": "model-drift",
    "retrain": "model-drift",
    "privacy": "privacy-mask",
    "pii": "privacy-mask",
    "redact": "privacy-mask",
    "key": "key-rotation",
    "rotation": "key-rotation",
    "rotate": "key-rotation"
  };
  function resolveName(n) {
    if (!n) return null;
    var k = String(n).toLowerCase().trim();
    if (ICONS[k]) return k;
    if (ALIAS[k]) return ALIAS[k];
    return null;
  }
  var MercuryIcon = class extends HTMLElement {
    static get observedAttributes() {
      return ["name", "tone", "size", "stroke", "glyph", "label"];
    }
    connectedCallback() {
      this.render();
    }
    attributeChangedCallback() {
      if (this.isConnected) this.render();
    }
    render() {
      var name = resolveName(this.getAttribute("name"));
      var size = parseFloat(this.getAttribute("size")) || 20;
      var sw = this.getAttribute("stroke") || "2";
      var wantGlyph = this.hasAttribute("glyph");
      var label = this.getAttribute("label");

      // resolve color: omitted ⇒ inherit currentColor; "auto" ⇒ icon default; else explicit
      var toneAttr = this.getAttribute("tone");
      var color = "";
      if (toneAttr) {
        var key = toneAttr === "auto" ? name && ICONS[name].tone : toneAttr;
        color = TONE[key] || "";
      }

      // host sits inline with text
      this.style.display = "inline-flex";
      this.style.alignItems = "center";
      this.style.justifyContent = "center";
      this.style.verticalAlign = "middle";
      this.style.lineHeight = "0";
      if (color) this.style.color = color;else this.style.removeProperty("color");
      var a11y = label ? ' role="img" aria-label="' + String(label).replace(/"/g, "&quot;") + '"' : ' aria-hidden="true"';
      var titleEl = label ? "<title>" + label.replace(/</g, "&lt;") + "</title>" : "";
      if (!name) {
        // unknown icon → a dashed placeholder box, never a broken render
        this.innerHTML = '<svg width="' + size + '" height="' + size + '" viewBox="0 0 24 24" fill="none"' + ' stroke="currentColor" stroke-width="' + sw + '" stroke-dasharray="3 3" stroke-linecap="round"' + a11y + '><rect x="3" y="3" width="18" height="18" rx="3"/></svg>';
        return;
      }
      if (wantGlyph && ICONS[name].glyph) {
        // native-glyph-first: math sigils render in mono (data voice), sized to box
        this.innerHTML = '<span' + a11y + ' style="font-family:var(--font-mono);font-weight:600;' + 'font-size:' + Math.round(size * 1.05) + 'px;line-height:1;color:currentColor">' + ICONS[name].glyph + "</span>";
        return;
      }
      this.innerHTML = '<svg width="' + size + '" height="' + size + '" viewBox="0 0 24 24" fill="none"' + ' stroke="currentColor" stroke-width="' + sw + '" stroke-linecap="round" stroke-linejoin="round"' + a11y + ">" + titleEl + ICONS[name].p + "</svg>";
    }
  };
  customElements.define("mercury-icon", MercuryIcon);

  // expose the vocabulary for JS surfaces that need a path/tone without an element
  window.MercuryIcon = {
    names: Object.keys(ICONS),
    ICONS: ICONS,
    ALIAS: ALIAS,
    TONE: TONE,
    resolve: resolveName,
    // returns raw <svg> string (handy for innerHTML in non-custom-element contexts)
    svg: function (name, opts) {
      opts = opts || {};
      var n = resolveName(name);
      if (!n) return "";
      var size = opts.size || 20,
        sw = opts.stroke || 2;
      var color = opts.tone ? TONE[opts.tone === "auto" ? ICONS[n].tone : opts.tone] || "currentColor" : "currentColor";
      return '<svg width="' + size + '" height="' + size + '" viewBox="0 0 24 24" fill="none" stroke="' + color + '" stroke-width="' + sw + '" stroke-linecap="round" stroke-linejoin="round">' + ICONS[n].p + "</svg>";
    }
  };
})();
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/mercury-icon.js", error: String((e && e.message) || e) }); }

// components/mercury-nav.js
try { (() => {
/* ============================================================================
   <mercury-nav> — Mercury Agent · the persistent system navigator
   ----------------------------------------------------------------------------
   One slim, floating top-center bar present on EVERY surface, so the kits stop
   being islands and start reading as one product (review axis: interconnected-
   ness). It provides:
     • home (♱ → System Hub)         • back / forward (browser history)
     • a PAGE SELECTOR dropdown       • a SETTINGS dropdown (reduced motion)
     • smooth page-transition crossfades through the brand void color

   Mount (mirrors the dock): set a config global, then load this script.
     <script>window.MA_NAV = { root: "../../", current: "console" };</script>
     <script src="../../components/mercury-nav.js"></script>
   `root` is the relative path from THIS page to the project root ("" at root,
   "../" one level down, "../../" for a kit). `current` is the page id below.

   Transitions: the nav intercepts ordinary same-tab <a> clicks across the page
   (hub cards, the agent dock's jump links, its own menu) and crossfades through
   black — the new page fades in to match. Honors reduced motion.
   Reads design tokens from colors_and_type.css; defines no palette of its own.
   ============================================================================ */
(function () {
  if (window.customElements && customElements.get("mercury-nav")) return;

  // canonical page registry — hrefs are relative to project ROOT; the nav
  // prefixes each with the page's `root`. One source for every surface link.
  var PAGES = [{
    id: "hub",
    label: "System Hub",
    glyph: "♱",
    href: "index.html",
    group: ""
  }, {
    id: "engine",
    label: "Live Detection Engine",
    glyph: "◊",
    href: "Mercury Engine.dc.html",
    group: "Surfaces",
    kicker: "live compute"
  }, {
    id: "console",
    label: "Console",
    glyph: "◑",
    href: "ui_kits/console/index.html",
    group: "Surfaces",
    kicker: "SOC dashboard"
  }, {
    id: "explorer",
    label: "Detection & Audit Explorer",
    glyph: "⌕",
    href: "ui_kits/explorer/index.html",
    group: "Surfaces",
    kicker: "faceted query"
  }, {
    id: "hazard",
    label: "Emergency Operations Console",
    glyph: "⊕",
    href: "ui_kits/hazard/index.html",
    group: "Surfaces",
    kicker: "watch floor"
  }, {
    id: "cli",
    label: "CLI",
    glyph: "▌",
    href: "ui_kits/cli/index.html",
    group: "Surfaces",
    kicker: "terminal"
  }, {
    id: "crypto",
    label: "Cryptography",
    glyph: "⛨",
    href: "ui_kits/crypto/index.html",
    group: "Surfaces",
    kicker: "AMA · PQC"
  }, {
    id: "cognition",
    label: "Cognitive Architecture",
    glyph: "⌬",
    href: "ui_kits/cognition/index.html",
    group: "Surfaces",
    kicker: "how it thinks"
  }, {
    id: "docs",
    label: "Documentation",
    glyph: "§",
    href: "ui_kits/docs/index.html",
    group: "Reference",
    kicker: "long-form"
  }, {
    id: "deck",
    label: "Overview Deck",
    glyph: "◫",
    href: "ui_kits/deck/index.html",
    group: "Reference",
    kicker: "presentation"
  }, {
    id: "review",
    label: "Design System Review",
    glyph: "❖",
    href: "review/Mercury Agent - Design System Review.html",
    group: "Reference",
    kicker: "posture"
  }];
  var LS_MOTION = "mercury-reduce-motion";
  var SS_TRANSITION = "mercury-nav-transition";

  // apply persisted reduced-motion preference ASAP
  try {
    if (localStorage.getItem(LS_MOTION) === "1") document.documentElement.classList.add("mn-reduce-motion");
  } catch (e) {}
  var INJECTED = false;
  function injectCSS() {
    if (INJECTED) return;
    INJECTED = true;
    var css = "" + "mercury-nav{position:fixed;top:0;left:0;width:100%;z-index:9998;pointer-events:none;font-family:var(--font-mono,monospace);}" + ".mn-bar{pointer-events:auto;position:absolute;top:11px;left:50%;transform:translateX(-50%);display:flex;align-items:center;gap:3px;" + "background:color-mix(in srgb, var(--surface-1,#0f141c) 86%, transparent);backdrop-filter:blur(9px);" + "border:1px solid var(--border-strong,#36465a);border-radius:999px;padding:5px 6px;box-shadow:0 8px 28px rgba(0,0,0,.5);}" + ".mn-btn{display:inline-flex;align-items:center;justify-content:center;gap:6px;height:28px;min-width:28px;padding:0 7px;border:none;cursor:pointer;" + "background:transparent;color:var(--fg-2,#a9b6c6);font-family:var(--font-mono,monospace);font-size:13px;border-radius:999px;transition:background .12s,color .12s;}" + ".mn-btn:hover{background:var(--surface-3,#1c2530);color:var(--fg,#e6edf3);}" + ".mn-btn:disabled{opacity:.3;cursor:default;}" + ".mn-btn:disabled:hover{background:transparent;color:var(--fg-2,#a9b6c6);}" + ".mn-btn.home{color:var(--brand-gold,#B4B124);font-size:15px;}" + ".mn-sep{width:1px;height:18px;background:var(--border,#243040);margin:0 2px;}" + ".mn-cur{display:inline-flex;align-items:center;gap:8px;height:28px;padding:0 12px;cursor:pointer;border:none;border-radius:999px;" + "background:var(--surface-2,#151c26);color:var(--fg,#e6edf3);font-family:var(--font-mono,monospace);font-size:12px;letter-spacing:.02em;transition:background .12s;}" + ".mn-cur:hover{background:var(--surface-3,#1c2530);}" + ".mn-cur .g{color:var(--mercury-300,#4fc3f7);font-size:13px;}" + ".mn-cur .cv{font-size:9px;color:var(--fg-3,#6b7888);transition:transform .14s;}" + ".mn-bar.open-pages .mn-cur .cv{transform:rotate(180deg);}" + ".mn-pop{pointer-events:auto;position:absolute;top:46px;background:var(--surface-1,#0f141c);border:1px solid var(--border-strong,#36465a);" + "border-radius:var(--r-md,8px);box-shadow:0 18px 50px rgba(0,0,0,.62);overflow:hidden;opacity:0;transform:translateY(-6px);" + "transition:opacity .15s,transform .15s;visibility:hidden;}" + ".mn-pop.show{opacity:1;transform:translateY(0);visibility:visible;}" + ".mn-pop.pages{left:50%;transform:translateX(-50%) translateY(-6px);width:280px;}" + ".mn-pop.pages.show{transform:translateX(-50%) translateY(0);}" + ".mn-pop.settings{right:0;width:248px;}" + ".mn-pop.disc{right:0;width:286px;}" + ".mn-btn.disc{color:var(--anomaly-soft,#ff7a7e);font-size:14px;}" + ".mn-btn.disc:hover{background:var(--anomaly-bg,rgba(255,77,82,.1));color:var(--anomaly,#ff4d52);}" + ".mn-disc-h{display:flex;align-items:center;gap:8px;padding:12px 14px 9px;border-bottom:1px solid var(--border-faint,#161e28);}" + ".mn-disc-h .p{color:var(--anomaly-soft,#ff7a7e);font-size:13px;}" + ".mn-disc-h .t{font-size:11px;letter-spacing:.06em;color:var(--anomaly-soft,#ff7a7e);}" + ".mn-disc-p{font-family:var(--font-sans,sans-serif);font-size:11px;line-height:1.5;color:var(--fg-3,#6b7888);padding:11px 14px;}" + ".mn-disc-acts{display:flex;gap:8px;padding:0 14px 13px;}" + ".mn-disc-cancel{flex:1;font-family:var(--font-mono,monospace);font-size:10.5px;color:var(--fg-2,#a9b6c6);background:var(--surface-2,#151c26);" + "border:1px solid var(--border,#243040);border-radius:var(--r-sm,5px);padding:9px;cursor:pointer;}" + ".mn-disc-cancel:hover{border-color:var(--border-strong,#36465a);color:var(--fg,#e6edf3);}" + ".mn-disc-go{flex:1.4;font-family:var(--font-mono,monospace);font-size:10.5px;font-weight:600;letter-spacing:.04em;color:var(--anomaly-soft,#ff7a7e);" + "background:var(--anomaly-bg,rgba(255,77,82,.1));border:1px solid var(--anomaly-deep,#b3231f);border-radius:var(--r-sm,5px);padding:9px;cursor:pointer;}" + ".mn-disc-go:hover{background:var(--anomaly,#ff4d52);color:#fff;border-color:var(--anomaly,#ff4d52);}" + ".mn-grp{font-size:8px;letter-spacing:.16em;text-transform:uppercase;color:var(--fg-muted,#495567);padding:11px 13px 5px;}" + ".mn-item{display:flex;align-items:center;gap:11px;padding:9px 13px;cursor:pointer;border-left:2px solid transparent;text-decoration:none;}" + ".mn-item:hover{background:var(--surface-2,#151c26);}" + ".mn-item.cur{border-left-color:var(--mercury-500,#00bcd4);background:var(--surface-2,#151c26);}" + ".mn-item .ig{flex:none;width:20px;text-align:center;color:var(--mercury-300,#4fc3f7);font-size:14px;}" + ".mn-item.cur .ig{color:var(--mercury-200,#8ae3f5);}" + ".mn-item .il{flex:1;color:var(--fg,#e6edf3);font-size:12.5px;}" + ".mn-item .ik{font-size:9px;color:var(--fg-muted,#495567);letter-spacing:.04em;}" + ".mn-item .idot{font-size:8px;color:var(--normal,#2fd686);}" + ".mn-shead{display:flex;align-items:center;gap:8px;padding:11px 13px 9px;border-bottom:1px solid var(--border-faint,#161e28);}" + ".mn-shead .s{color:var(--brand-gold,#B4B124);font-size:13px;}" + ".mn-shead .t{font-size:11px;letter-spacing:.06em;color:var(--fg,#e6edf3);}" + ".mn-srow{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:11px 13px;}" + ".mn-srow .sl{display:flex;flex-direction:column;gap:2px;}" + ".mn-srow .sl b{font-size:11.5px;color:var(--fg,#e6edf3);font-weight:500;}" + ".mn-srow .sl span{font-size:9px;color:var(--fg-muted,#495567);}" + ".mn-switch{flex:none;width:38px;height:21px;border-radius:999px;border:1px solid var(--border-strong,#36465a);background:var(--surface-3,#1c2530);" + "cursor:pointer;position:relative;transition:background .14s,border-color .14s;}" + ".mn-switch::after{content:'';position:absolute;top:2px;left:2px;width:15px;height:15px;border-radius:50%;background:var(--fg-3,#6b7888);transition:transform .14s,background .14s;}" + ".mn-switch.on{background:var(--mercury-900,#0c3b44);border-color:var(--mercury-600,#00a0b6);}" + ".mn-switch.on::after{transform:translateX(17px);background:var(--mercury-300,#4fc3f7);}" + ".mn-sfoot{padding:9px 13px;border-top:1px solid var(--border-faint,#161e28);display:flex;align-items:center;justify-content:space-between;}" + ".mn-sfoot .k{font-size:9px;color:var(--fg-muted,#495567);}" + ".mn-sfoot .kbd{font-size:9px;color:var(--fg-3,#6b7888);border:1px solid var(--border,#243040);border-radius:3px;padding:2px 6px;}" + ".mn-veil{position:fixed;inset:0;z-index:100000;background:var(--void,#06080B);opacity:0;pointer-events:none;transition:opacity .26s ease;}" + ".mn-veil.on{opacity:1;pointer-events:auto;}" + "html.mn-reduce-motion .mn-veil{transition:none;}" + "html.mn-reduce-motion .mn-pop{transition:none;}" + "@media(max-width:560px){.mn-cur .cl{display:none;}.mn-pop.pages{width:240px;}}";
    var s = document.createElement("style");
    s.id = "mercury-nav-css";
    s.textContent = css;
    document.head.appendChild(s);
  }
  var MercuryNav = class extends HTMLElement {
    connectedCallback() {
      injectCSS();
      this._root = this.getAttribute("root") || window.MA_NAV && window.MA_NAV.root || "";
      this._current = this.getAttribute("current") || window.MA_NAV && window.MA_NAV.current || "";
      this.render();
      this._initHistory(); // reconcile our own back/forward stack for this page
      this._wireTransitions();
      this._fadeInIfArriving();
      this._loadSettings(); // apply persisted theme/accent/a11y on every page
      this._loadQuarantine(); // restore a system-wide quarantine lockdown if one is active
    }
    href(p) {
      return this._root + p.href;
    }
    render() {
      var cur = PAGES.find(p => p.id === this._current) || PAGES[0];
      var groups = [];
      PAGES.forEach(p => {
        var g = groups.find(x => x.name === p.group);
        if (!g) {
          g = {
            name: p.group,
            items: []
          };
          groups.push(g);
        }
        g.items.push(p);
      });
      var pagesHtml = groups.map(g => (g.name ? '<div class="mn-grp">' + g.name + "</div>" : "") + g.items.map(p => '<a class="mn-item' + (p.id === this._current ? " cur" : "") + '" href="' + this.href(p) + '" data-nav="' + p.id + '">' + '<span class="ig">' + p.glyph + "</span>" + '<span class="il">' + p.label + "</span>" + (p.id === this._current ? '<span class="idot">●</span>' : p.kicker ? '<span class="ik">' + p.kicker + "</span>" : "") + "</a>").join("")).join("");
      var motionOn = document.documentElement.classList.contains("mn-reduce-motion");
      this.innerHTML = '<div class="mn-bar">' + '<a class="mn-btn home" href="' + this.href(PAGES[0]) + '" title="System Hub" data-nav="hub">♱</a>' + '<button class="mn-btn" data-back title="Back">‹</button>' + '<button class="mn-btn" data-fwd title="Forward">›</button>' + '<span class="mn-sep"></span>' + '<button class="mn-cur" data-pages><span class="g">' + cur.glyph + '</span><span class="cl">' + cur.label + '</span><span class="cv">▾</span></button>' + '<span class="mn-sep"></span>' + '<button class="mn-btn disc" data-disc title="Mercury / AMA Disconnect">⏻</button>' + '<button class="mn-btn" data-settings title="Settings">⚙</button>' + '<div class="mn-pop pages" data-pop="pages">' + pagesHtml + "</div>" + '<div class="mn-pop disc" data-pop="disc">' + '<div class="mn-disc-h"><span class="p">⏻</span><span class="t">MERCURY / AMA DISCONNECT</span></div>' + '<div class="mn-disc-p">Deliberately isolates and quarantines the whole system until an authorized operator re-activates. The PQC import gate fails closed and the agent kill switch halts every cognitive operation — across every surface. A threat cannot lift it; only you can.</div>' + '<div class="mn-disc-acts"><button class="mn-disc-cancel" data-disc-cancel>Cancel</button><button class="mn-disc-go" data-disc-go>⏻ CONFIRM DISCONNECT</button></div>' + "</div>" + "</div>";
      var bar = this.querySelector(".mn-bar");
      var self = this;
      // history — driven by our own deterministic stack (sessionStorage), so back
      // AND forward work consistently on every page, with correct disabled states.
      this.querySelector("[data-back]").addEventListener("click", function () {
        self._histGo(-1);
      });
      this.querySelector("[data-fwd]").addEventListener("click", function () {
        self._histGo(1);
      });
      // dropdowns
      var popPages = this.querySelector('[data-pop="pages"]');
      var popDisc = this.querySelector('[data-pop="disc"]');
      function closeAll() {
        popPages.classList.remove("show");
        popDisc.classList.remove("show");
        bar.classList.remove("open-pages");
      }
      this.querySelector("[data-pages]").addEventListener("click", function (e) {
        e.stopPropagation();
        var open = popPages.classList.contains("show");
        closeAll();
        if (!open) {
          popPages.classList.add("show");
          bar.classList.add("open-pages");
        }
      });
      this.querySelector("[data-settings]").addEventListener("click", function (e) {
        e.stopPropagation();
        closeAll();
        self._openSettings();
      });
      this.querySelector("[data-disc]").addEventListener("click", function (e) {
        e.stopPropagation();
        var open = popDisc.classList.contains("show");
        closeAll();
        if (!open) popDisc.classList.add("show");
      });
      this.querySelector("[data-disc-cancel]").addEventListener("click", function (e) {
        e.stopPropagation();
        closeAll();
      });
      this.querySelector("[data-disc-go]").addEventListener("click", function (e) {
        e.stopPropagation();
        closeAll();
        self._disconnect();
      });
      document.addEventListener("click", closeAll);
      document.addEventListener("keydown", function (e) {
        if (e.key === "Escape") closeAll();
      });
      popPages.addEventListener("click", function (e) {
        e.stopPropagation();
      });
      popDisc.addEventListener("click", function (e) {
        e.stopPropagation();
      });
    }
    _openSettings() {
      var root = this._root;
      if (window.MercurySettings) {
        window.MercurySettings.open(root);
        return;
      }
      this._loadSettings(function () {
        if (window.MercurySettings) window.MercurySettings.open(root);
      });
    }
    _loadSettings(cb) {
      if (window.MercurySettings) {
        cb && cb();
        return;
      }
      if (this._settingsLoading) {
        var t = setInterval(function () {
          if (window.MercurySettings) {
            clearInterval(t);
            cb && cb();
          }
        }, 40);
        return;
      }
      this._settingsLoading = true;
      var s = document.createElement("script");
      s.src = this._root + "components/mercury-settings.js";
      s.onload = function () {
        cb && cb();
      };
      document.head.appendChild(s);
    }
    _disconnect() {
      if (window.MercuryQuarantine) {
        window.MercuryQuarantine.engage();
        return;
      }
      this._loadQuarantine(function () {
        if (window.MercuryQuarantine) window.MercuryQuarantine.engage();
      });
    }
    _loadQuarantine(cb) {
      if (window.MercuryQuarantine) {
        cb && cb();
        return;
      }
      if (this._qLoading) {
        var t = setInterval(function () {
          if (window.MercuryQuarantine) {
            clearInterval(t);
            cb && cb();
          }
        }, 40);
        return;
      }
      this._qLoading = true;
      var s = document.createElement("script");
      s.src = this._root + "components/mercury-quarantine.js";
      s.onload = function () {
        cb && cb();
      };
      document.head.appendChild(s);
    }
    _veil() {
      var v = document.querySelector(".mn-veil");
      if (!v) {
        v = document.createElement("div");
        v.className = "mn-veil";
        document.body.appendChild(v);
      }
      return v;
    }
    _reduced() {
      return document.documentElement.classList.contains("mn-reduce-motion");
    }

    // ---- deterministic back/forward history (sessionStorage) -----------------
    _readHist() {
      try {
        return JSON.parse(sessionStorage.getItem("mn-hist"));
      } catch (e) {
        return null;
      }
    }
    _writeHist(h) {
      try {
        sessionStorage.setItem("mn-hist", JSON.stringify(h));
      } catch (e) {}
    }
    _here() {
      return location.pathname;
    } // identity by path only — preview/cache-buster query params are volatile

    _initHistory() {
      var here = this._here();
      var h = this._readHist();
      var jumped = false;
      try {
        jumped = sessionStorage.getItem("mn-hist-jump") === "1";
        sessionStorage.removeItem("mn-hist-jump");
      } catch (e) {}
      if (!h || !Array.isArray(h.stack) || !h.stack.length) {
        h = {
          stack: [here],
          idx: 0
        };
      } else if (jumped) {
        // arrived via our own back/forward — idx is already set; reconcile if needed
        if (h.stack[h.idx] !== here) {
          var found = h.stack.indexOf(here);
          h.idx = found >= 0 ? found : Math.max(0, Math.min(h.idx, h.stack.length - 1));
        }
      } else if (h.stack[h.idx] !== here) {
        // a fresh navigation: drop any forward entries, push this page
        h.stack = h.stack.slice(0, h.idx + 1);
        h.stack.push(here);
        if (h.stack.length > 50) h.stack = h.stack.slice(h.stack.length - 50);
        h.idx = h.stack.length - 1;
      }
      this._writeHist(h);
      this._hist = h;
      this._updateHistButtons();
    }
    _labelForUrl(u) {
      if (!u) return "";
      var best = null,
        bestLen = -1;
      for (var i = 0; i < PAGES.length; i++) {
        var ph = PAGES[i].href;
        if (u.length >= ph.length && u.slice(u.length - ph.length) === ph && ph.length > bestLen) {
          best = PAGES[i].label;
          bestLen = ph.length;
        }
      }
      if (best) return best;
      var seg = u.split("/").filter(Boolean);
      return seg.length ? seg[seg.length - 1] : u;
    }
    _updateHistButtons() {
      var h = this._hist || {
        stack: [],
        idx: 0
      };
      var back = this.querySelector("[data-back]"),
        fwd = this.querySelector("[data-fwd]");
      if (!back || !fwd) return;
      var canBack = h.idx > 0,
        canFwd = h.idx < h.stack.length - 1;
      back.disabled = !canBack;
      fwd.disabled = !canFwd;
      back.title = canBack ? "Back · " + this._labelForUrl(h.stack[h.idx - 1]) : "Back";
      fwd.title = canFwd ? "Forward · " + this._labelForUrl(h.stack[h.idx + 1]) : "Forward";
    }
    _histGo(delta) {
      var h = this._hist;
      if (!h) return;
      var ni = h.idx + delta;
      if (ni < 0 || ni >= h.stack.length) return; // at an end — no-op (button is also disabled)
      h.idx = ni;
      this._writeHist(h);
      try {
        sessionStorage.setItem("mn-hist-jump", "1");
      } catch (e) {}
      this._navigate(location.origin + h.stack[ni]);
    }
    _navigate(url) {
      if (this._reduced()) {
        window.location.href = url;
        return;
      }
      var v = this._veil();
      try {
        sessionStorage.setItem(SS_TRANSITION, "1");
      } catch (e) {}
      // force reflow so the transition runs
      void v.offsetWidth;
      v.classList.add("on");
      setTimeout(function () {
        window.location.href = url;
      }, 250);
    }
    _fadeInIfArriving() {
      var arrived = false;
      try {
        arrived = sessionStorage.getItem(SS_TRANSITION) === "1";
        sessionStorage.removeItem(SS_TRANSITION);
      } catch (e) {}
      if (!arrived || this._reduced()) return;
      var v = this._veil();
      v.classList.add("on");
      requestAnimationFrame(function () {
        requestAnimationFrame(function () {
          v.classList.remove("on");
        });
      });
    }
    _wireTransitions() {
      var self = this;
      // intercept ordinary same-tab navigations anywhere on the page (hub cards,
      // dock jump links, our own menu) and crossfade through the void.
      document.addEventListener("click", function (e) {
        if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
        var a = e.target.closest && e.target.closest("a[href]");
        if (!a) return;
        if (a.target === "_blank" || a.hasAttribute("download")) return;
        var href = a.getAttribute("href");
        if (!href || href.charAt(0) === "#" || /^(mailto:|tel:|javascript:)/i.test(href)) return;
        // same-origin only
        var dest;
        try {
          dest = new URL(a.href, location.href);
        } catch (err) {
          return;
        }
        if (dest.origin !== location.origin) return;
        if (dest.pathname === location.pathname && dest.hash) return; // in-page anchor
        e.preventDefault();
        self._navigate(a.href);
      }, true);
    }
  };
  customElements.define("mercury-nav", MercuryNav);

  // public helper for JS-driven navigation (e.g. command palette)
  window.MercuryNav = {
    go: function (url) {
      var el = document.querySelector("mercury-nav");
      if (el && el._navigate) el._navigate(url);else window.location.href = url;
    },
    pages: PAGES
  };

  // auto-mount from window.MA_NAV (mirrors the dock's MA_CONTEXT pattern)
  function autoMount() {
    if (!document.querySelector("mercury-nav") && window.MA_NAV) {
      var el = document.createElement("mercury-nav");
      if (window.MA_NAV.root != null) el.setAttribute("root", window.MA_NAV.root);
      if (window.MA_NAV.current) el.setAttribute("current", window.MA_NAV.current);
      document.body.appendChild(el);
    }
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", autoMount);else autoMount();
})();
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/mercury-nav.js", error: String((e && e.message) || e) }); }

// components/mercury-quarantine.js
try { (() => {
/* ============================================================================
   <mercury-quarantine> — Mercury Agent · the system-wide fail-closed lockdown
   ----------------------------------------------------------------------------
   Elevates the "Mercury / AMA Disconnect" safety control from a single
   components card (preview/buttons.html) into a REAL, persistent system state.
   When engaged it:
     • writes the quarantine flag to localStorage (survives reload + new pages)
     • paints a full-viewport fail-closed lockdown over EVERY surface — the nav,
       the kits, the hub — so the quarantine reads as a property of the whole
       product, not one screen
     • can only be lifted by a deliberate, sustained human action (consent +
       hold-to-re-activate). A suspected threat cannot lift it; only the operator.
     • syncs across open tabs/surfaces through the `storage` event

   Grounding (per the brand's honesty rule): disconnecting AMA Cryptography fails
   the PQC import gate closed (RuntimeError before any package state) which
   renders Mercury Agent inoperable; autonomous_agent.py raises "Agent kill switch
   activated"; σ_Immutable is a fail-closed safety gate. This control models that
   real behavior — it is a posture surface, not a way to disable cryptography on a
   live node.

   API (mirrors MercurySettings):
     window.MercuryQuarantine.engage()      → arm-confirmed; quarantine the system
     window.MercuryQuarantine.lift()        → operator re-activation (clears state)
     window.MercuryQuarantine.isActive()    → boolean
     window.MercuryQuarantine.state()       → { active, corr, ts } | null
   Fires `mercury-quarantine-change` on window with { detail:{active} } on change.
   Reads design tokens from colors_and_type.css; defines no palette of its own.
   ============================================================================ */
(function () {
  if (window.MercuryQuarantine && window.MercuryQuarantine.__ready) return;
  var LS = "mercury-quarantine";
  function hex(n) {
    var s = "";
    for (var i = 0; i < n; i++) s += "0123456789abcdef"[Math.floor(Math.random() * 16)];
    return s;
  }
  function read() {
    try {
      return JSON.parse(localStorage.getItem(LS));
    } catch (e) {
      return null;
    }
  }
  function write(v) {
    try {
      v ? localStorage.setItem(LS, JSON.stringify(v)) : localStorage.removeItem(LS);
    } catch (e) {}
  }
  function isActive() {
    var s = read();
    return !!(s && s.active);
  }
  function reduced() {
    return document.documentElement.classList.contains("mn-reduce-motion");
  }

  // fail-closed consequence ledger — each line traces to a real code path
  var LINES = [["AMA Cryptography unloaded — import gate failed closed; engine inoperable.", "_pqc_gate._enforce_pqc_production_gate → RuntimeError"], ["Agent cognitive loop halted at every gate.", "autonomous_agent.py → \"Agent kill switch activated\""], ["σ_Immutable ethics gate held closed; no decisions emitted.", "fail-closed safety gate"], ["Detection, export and disclosure surfaces isolated until re-activation.", "every surface · Console · Explorer · Hazard · CLI · Crypto"]];

  // -------------------------------------------------------------------------
  //  STYLES
  // -------------------------------------------------------------------------
  var INJECTED = false;
  function injectCSS() {
    if (INJECTED) return;
    INJECTED = true;
    var css = "" + "mercury-quarantine{position:fixed;inset:0;z-index:2147483600;display:none;font-family:var(--font-sans,sans-serif);}" + "mercury-quarantine[active]{display:block;}" + ".mq-scrim{position:absolute;inset:0;background:repeating-linear-gradient(45deg,rgba(6,8,11,.975),rgba(6,8,11,.975) 15px,rgba(40,10,12,.975) 15px,rgba(40,10,12,.975) 30px);" + "backdrop-filter:blur(2px);animation:mqfade .22s ease;}" + "@keyframes mqfade{from{opacity:0;}to{opacity:1;}}" + "html.mn-reduce-motion .mq-scrim{animation:none;}" + ".mq-card{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:min(620px,94vw);max-height:90vh;display:flex;flex-direction:column;" + "background:var(--surface-1,#0f141c);border:1px solid var(--anomaly-deep,#b3231f);border-radius:var(--r-md,8px);" + "box-shadow:0 30px 90px rgba(0,0,0,.7),0 0 0 1px rgba(255,77,82,.18);overflow:hidden;}" + ".mq-head{display:flex;align-items:center;gap:12px;border-bottom:1px solid var(--anomaly-deep,#b3231f);padding:17px 22px 14px;flex:none;" + "background:linear-gradient(180deg,var(--anomaly-bg,rgba(255,77,82,.1)),transparent);}" + ".mq-dot{width:12px;height:12px;border-radius:50%;background:var(--anomaly,#ff4d52);box-shadow:0 0 14px var(--anomaly,#ff4d52);flex:none;animation:mqpulse 1.1s infinite;}" + "html.mn-reduce-motion .mq-dot{animation:none;}" + "@keyframes mqpulse{50%{opacity:.4;}}" + ".mq-title{font-family:var(--font-mono,monospace);font-size:16px;font-weight:600;letter-spacing:.06em;color:var(--anomaly-soft,#ff7a7e);}" + ".mq-sub{font-family:var(--font-mono,monospace);font-size:9px;letter-spacing:.16em;color:var(--fg-muted,#495567);margin-top:4px;}" + ".mq-corr-top{margin-left:auto;font-family:var(--font-mono,monospace);font-size:9px;letter-spacing:.04em;color:var(--fg-muted,#495567);text-align:right;white-space:nowrap;}" + ".mq-body{flex:1;min-height:0;overflow-y:auto;overflow-x:hidden;display:flex;flex-direction:column;gap:10px;padding:16px 22px;}" + ".mq-line{display:flex;align-items:flex-start;gap:10px;font-family:var(--font-mono,monospace);font-size:11.5px;line-height:1.5;color:var(--fg-2,#a9b6c6);}" + ".mq-line>span{min-width:0;overflow-wrap:anywhere;}" + ".mq-line .x{color:var(--anomaly,#ff4d52);flex:none;}" + ".mq-line .src{display:block;color:var(--fg-muted,#495567);font-size:9.5px;margin-top:2px;overflow-wrap:anywhere;}" + ".mq-foot{flex:none;border-top:1px solid var(--mercury-800,#0a5d6b);padding:15px 22px 18px;background:linear-gradient(0deg,rgba(0,188,212,.07),transparent);}" + ".mq-reauth-h{display:flex;align-items:center;gap:8px;font-family:var(--font-mono,monospace);font-size:9px;letter-spacing:.16em;color:var(--mercury-300,#4fc3f7);margin-bottom:11px;}" + ".mq-reauth-h::before{content:'↻';font-size:12px;}" + ".mq-consent{display:flex;align-items:flex-start;gap:9px;cursor:pointer;font-family:var(--font-sans,sans-serif);font-size:11.5px;line-height:1.45;color:var(--fg-2,#a9b6c6);margin-bottom:13px;}" + ".mq-consent input{accent-color:var(--mercury-500,#00bcd4);width:15px;height:15px;cursor:pointer;flex:none;margin-top:1px;}" + ".mq-actions{display:flex;align-items:center;gap:14px;flex-wrap:wrap;}" + ".mq-react{position:relative;flex:none;min-width:236px;height:46px;overflow:hidden;border-radius:var(--r-sm,5px);" + "background:var(--surface-2,#151c26);color:var(--fg-muted,#495567);border:1px solid var(--border,#243040);cursor:not-allowed;" + "display:inline-flex;align-items:center;justify-content:center;font-family:var(--font-mono,monospace);}" + ".mq-react.ready{color:var(--mercury-100,#bdf0fa);border-color:var(--mercury-600,#00a0b6);background:var(--mercury-900,#0c3b44);cursor:pointer;}" + ".mq-react.ready:hover{box-shadow:var(--glow-mercury,0 0 18px rgba(0,188,212,.5));}" + ".mq-fill{position:absolute;left:0;top:0;height:100%;width:0;background:var(--mercury-600,#00a0b6);opacity:.42;z-index:0;transition:width .12s linear;}" + ".mq-react.holding .mq-fill{width:100%;transition:width 1.2s linear;}" + "html.mn-reduce-motion .mq-react.holding .mq-fill{transition:width .2s linear;}" + ".mq-react-label{position:relative;z-index:1;font-size:12px;letter-spacing:.05em;display:inline-flex;align-items:center;gap:8px;}" + ".mq-react.done{background:var(--normal-bg,rgba(47,214,134,.12));border-color:var(--normal-deep,#1c7a4d);color:var(--normal-bright,#5fe6a8);}" + ".mq-hint{font-family:var(--font-mono,monospace);font-size:9.5px;color:var(--fg-muted,#495567);line-height:1.5;flex:1;min-width:140px;}" + "@media(max-width:560px){.mq-react{min-width:0;width:100%;}.mq-actions{flex-direction:column;align-items:stretch;}}";
    var s = document.createElement("style");
    s.id = "mercury-quarantine-css";
    s.textContent = css;
    document.head.appendChild(s);
  }

  // -------------------------------------------------------------------------
  //  ELEMENT
  // -------------------------------------------------------------------------
  var El = class extends HTMLElement {
    connectedCallback() {
      injectCSS();
      this._build();
      var self = this;
      // cross-surface sync: another tab/surface engaged or lifted
      this._onStorage = function (e) {
        if (e.key === LS) self._sync();
      };
      window.addEventListener("storage", this._onStorage);
      this._sync();
    }
    disconnectedCallback() {
      window.removeEventListener("storage", this._onStorage);
    }
    _build() {
      var s = read() || {};
      var linesHtml = LINES.map(function (l) {
        return '<div class="mq-line"><span class="x">✕</span><span>' + l[0] + '<span class="src">' + l[1] + '</span></span></div>';
      }).join("");
      this.innerHTML = '<div class="mq-scrim"></div>' + '<div class="mq-card" role="alertdialog" aria-label="System quarantined">' + '<div class="mq-head">' + '<span class="mq-dot"></span>' + '<div><div class="mq-title">SYSTEM QUARANTINED</div><div class="mq-sub">MERCURY / AMA DISCONNECTED · FAIL-CLOSED LOCKDOWN</div></div>' + '<div class="mq-corr-top" data-corr></div>' + '</div>' + '<div class="mq-body">' + linesHtml + '</div>' + '<div class="mq-foot">' + '<div class="mq-reauth-h">RE-ACTIVATION · AUTHORIZED OPERATOR ONLY</div>' + '<label class="mq-consent"><input type="checkbox" data-consent> I am the authorized operator re-activating this system — a deliberate human action, not a threat or automated process.</label>' + '<div class="mq-actions">' + '<button class="mq-react" data-react aria-disabled="true" tabindex="0"><span class="mq-fill" data-fill></span><span class="mq-react-label" data-label>↻ HOLD TO RE-ACTIVATE</span></button>' + '<span class="mq-hint" data-hint>Confirm authorization above to enable. Press and hold to lift quarantine.</span>' + '</div>' + '</div>' + '</div>';
      this._wire();
    }
    _wire() {
      var self = this;
      var consent = this.querySelector("[data-consent]");
      var react = this.querySelector("[data-react]");
      var fill = this.querySelector("[data-fill]");
      var label = this.querySelector("[data-label]");
      var hint = this.querySelector("[data-hint]");
      var holdTimer = null,
        holding = false,
        HOLD_MS = 1200;
      function updateReady() {
        var ready = consent.checked;
        react.classList.toggle("ready", ready);
        react.setAttribute("aria-disabled", String(!ready));
        hint.textContent = ready ? "Press and hold to lift quarantine." : "Confirm authorization above to enable. Press and hold to lift quarantine.";
        if (!ready) cancelHold();
      }
      function cancelHold() {
        holding = false;
        clearTimeout(holdTimer);
        holdTimer = null;
        react.classList.remove("holding");
        void fill.offsetWidth;
        if (consent.checked && !react.classList.contains("done")) label.textContent = "↻ HOLD TO RE-ACTIVATE";
      }
      function startHold(e) {
        if (!consent.checked || react.classList.contains("done")) return;
        e.preventDefault();
        holding = true;
        react.classList.add("holding");
        label.textContent = "↻ RE-ACTIVATING…";
        holdTimer = setTimeout(complete, reduced() ? 200 : HOLD_MS);
      }
      function endHold() {
        if (holding) cancelHold();
      }
      function complete() {
        holding = false;
        react.classList.remove("holding");
        react.classList.add("done");
        label.textContent = "✓ RE-ACTIVATED";
        setTimeout(function () {
          self.lift();
        }, 640);
      }
      consent.addEventListener("change", updateReady);
      react.addEventListener("mousedown", startHold);
      react.addEventListener("touchstart", startHold, {
        passive: false
      });
      ["mouseup", "mouseleave", "touchend", "touchcancel", "blur"].forEach(function (ev) {
        react.addEventListener(ev, endHold);
      });
      react.addEventListener("keydown", function (e) {
        if ((e.key === "Enter" || e.key === " ") && !holding) startHold(e);
      });
      react.addEventListener("keyup", function (e) {
        if (e.key === "Enter" || e.key === " ") endHold();
      });
      updateReady();
    }

    // reflect persisted state → DOM (called on load + storage events)
    _sync() {
      var s = read();
      if (s && s.active) {
        this.setAttribute("active", "");
        var c = this.querySelector("[data-corr]");
        if (c) c.textContent = "corr " + (s.corr || "—");
        // reset the re-auth control each time it shows
        var consent = this.querySelector("[data-consent]");
        if (consent) consent.checked = false;
        var react = this.querySelector("[data-react]");
        if (react) {
          react.classList.remove("ready", "done", "holding");
          react.setAttribute("aria-disabled", "true");
        }
        var label = this.querySelector("[data-label]");
        if (label) label.textContent = "↻ HOLD TO RE-ACTIVATE";
        document.documentElement.style.overflow = "hidden";
      } else {
        this.removeAttribute("active");
        document.documentElement.style.overflow = "";
      }
    }
    engage(opts) {
      opts = opts || {};
      var s = {
        active: true,
        corr: opts.corr || "lkd_" + hex(8) + "-" + hex(4),
        ts: Date.now()
      };
      write(s);
      this._build();
      this._sync();
      window.dispatchEvent(new CustomEvent("mercury-quarantine-change", {
        detail: {
          active: true,
          corr: s.corr
        }
      }));
    }
    lift() {
      write(null);
      this._sync();
      window.dispatchEvent(new CustomEvent("mercury-quarantine-change", {
        detail: {
          active: false
        }
      }));
    }
  };
  customElements.define("mercury-quarantine", El);
  function ensure() {
    var el = document.querySelector("mercury-quarantine");
    if (!el) {
      el = document.createElement("mercury-quarantine");
      document.body.appendChild(el);
    }
    return el;
  }
  window.MercuryQuarantine = {
    __ready: true,
    engage: function (opts) {
      ensure().engage(opts);
    },
    lift: function () {
      var el = document.querySelector("mercury-quarantine");
      if (el) el.lift();else write(null);
    },
    isActive: isActive,
    state: read
  };

  // auto-mount so an already-active quarantine restores on EVERY page load
  function mount() {
    ensure();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", mount);else mount();
})();
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/mercury-quarantine.js", error: String((e && e.message) || e) }); }

// components/mercury-settings.js
try { (() => {
/* ============================================================================
   <mercury-settings> — Mercury Agent · the system settings surface
   ----------------------------------------------------------------------------
   The full settings panel the navigator's ⚙ opens. One overlay, a left category
   rail and right panels, every control persisted in localStorage and applied
   live across EVERY surface (because the design system is token-driven, a theme
   or accent change cascades to all kits from here).

   It also AUTO-APPLIES persisted settings the moment this script loads — so a
   preference set on one page is already in effect when you arrive on another,
   even before the panel is opened. The navigator lazy-loads this file on every
   page for exactly that reason.

   Honesty note: Mercury ships a FastAPI backend + CLI, no GUI. Controls that map
   to real engine behavior (AMA Cryptography / 3R / σ_Immutable env flags, the
   local-first data posture) are grounded and labeled with their source; controls
   that are interface-only in a reference design (notifications, permissions) are
   present, functional as preferences, and honestly marked.

   API:  window.MercurySettings.open() / .close() / .get(k) / .set(k,v) / .apply()
   Reads design tokens from colors_and_type.css; defines no palette of its own.
   ============================================================================ */
(function () {
  if (window.MercurySettings && window.MercurySettings.__ready) return;
  var LS = "mercury-settings";
  var LS_MOTION = "mercury-reduce-motion"; // kept in sync with the nav's quick toggle

  // accent palettes → [mercury-300, 400, 500, 600]
  var ACCENTS = {
    cyan: {
      label: "Quicksilver",
      hex: "#00BCD4",
      ramp: null
    },
    // default — no override
    azure: {
      label: "Azure",
      hex: "#3E8EF7",
      ramp: ["#7FB0FB", "#5A97F8", "#3E8EF7", "#2E6FD0"]
    },
    viridis: {
      label: "Viridian",
      hex: "#2FD686",
      ramp: ["#6BE6AC", "#43DC95", "#2FD686", "#1FA866"]
    },
    violet: {
      label: "Cognitive",
      hex: "#8B7FE8",
      ramp: ["#B3A9F2", "#9C90EC", "#8B7FE8", "#6B5BD6"]
    },
    gold: {
      label: "Benevolence",
      hex: "#E0B341",
      ramp: ["#ECCD7A", "#E6C05D", "#E0B341", "#B8902C"]
    }
  };
  var DEFAULTS = {
    theme: "dark",
    // dark | light
    highContrast: false,
    accent: "cyan",
    scale: 1,
    // 0.85 – 1.25 interface zoom
    density: "comfortable",
    // comfortable | compact
    reduceMotion: false,
    // accessibility
    underlineLinks: false,
    keyboardFocus: false,
    textToSpeech: false,
    // security modes (grounded in real env flags / endpoints)
    amaCrypto: true,
    // AMA Cryptography PQC surface
    threeR: true,
    // 3R (Recursion-Resonance-Refactoring) mechanism
    sigmaGate: true,
    // σ_Immutable hard gate
    constantTime: true,
    // AMA constant-time path
    // content
    units: "metric",
    // metric | imperial
    sound: 0,
    // 0 – 100
    landing: "hub",
    // notifications (interface preference)
    notifyInApp: true,
    notifyEmail: false,
    notifyPush: false,
    notifyCriticalOnly: true,
    // privacy (interface preference; posture is local-first)
    consentAudit: true,
    consentDiagnostics: false
  };
  function load() {
    try {
      return JSON.parse(localStorage.getItem(LS)) || {};
    } catch (e) {
      return {};
    }
  }
  var S = Object.assign({}, DEFAULTS, load());
  function persist() {
    try {
      localStorage.setItem(LS, JSON.stringify(S));
    } catch (e) {}
  }

  // ---- apply (functional settings → live DOM) ------------------------------
  function apply() {
    var el = document.documentElement;
    el.classList.toggle("ds-light", S.theme === "light");
    el.classList.toggle("ds-contrast", !!S.highContrast);
    el.classList.toggle("ds-compact", S.density === "compact");
    el.classList.toggle("ds-underline", !!S.underlineLinks);
    el.classList.toggle("ds-focus", !!S.keyboardFocus);
    el.classList.toggle("mn-reduce-motion", !!S.reduceMotion);
    try {
      localStorage.setItem(LS_MOTION, S.reduceMotion ? "1" : "0");
    } catch (e) {}
    var pal = ACCENTS[S.accent];
    if (pal && pal.ramp) {
      el.style.setProperty("--mercury-300", pal.ramp[0]);
      el.style.setProperty("--mercury-400", pal.ramp[1]);
      el.style.setProperty("--mercury-500", pal.ramp[2]);
      el.style.setProperty("--mercury-600", pal.ramp[3]);
    } else {
      el.style.removeProperty("--mercury-300");
      el.style.removeProperty("--mercury-400");
      el.style.removeProperty("--mercury-500");
      el.style.removeProperty("--mercury-600");
    }
    el.style.zoom = S.scale && S.scale != 1 ? String(S.scale) : "";
  }

  // ---- tiny click sound (WebAudio) gated by the sound setting --------------
  var actx = null;
  function blip() {
    if (!S.sound) return;
    try {
      actx = actx || new (window.AudioContext || window.webkitAudioContext)();
      var o = actx.createOscillator(),
        g = actx.createGain();
      o.type = "sine";
      o.frequency.value = 660;
      g.gain.value = S.sound / 100 * 0.06;
      o.connect(g);
      g.connect(actx.destination);
      o.start();
      g.gain.exponentialRampToValueAtTime(0.0001, actx.currentTime + 0.09);
      o.stop(actx.currentTime + 0.1);
    } catch (e) {}
  }
  function speak(t) {
    if (!S.textToSpeech || !window.speechSynthesis) return;
    try {
      var u = new SpeechSynthesisUtterance(t);
      u.rate = 1;
      u.pitch = 1;
      speechSynthesis.cancel();
      speechSynthesis.speak(u);
    } catch (e) {}
  }
  apply(); // ← on load, before any UI

  // ==========================================================================
  //  PANEL SCHEMA — symmetric, data-driven sections
  // ==========================================================================
  var SECTIONS = [{
    id: "appearance",
    label: "Display & Appearance",
    glyph: "◐",
    rows: [{
      k: "theme",
      type: "segment",
      label: "Theme",
      hint: "dark instrument · light reading register",
      options: [["dark", "Dark"], ["light", "Light"]]
    }, {
      k: "highContrast",
      type: "switch",
      label: "High contrast",
      hint: "boost text & borders"
    }, {
      k: "accent",
      type: "swatch",
      label: "Accent color",
      hint: "the live signal hue"
    }, {
      k: "scale",
      type: "slider",
      label: "Interface scale",
      min: 0.85,
      max: 1.25,
      step: 0.05,
      fmt: function (v) {
        return Math.round(v * 100) + "%";
      },
      hint: "display / font scaling"
    }, {
      k: "density",
      type: "segment",
      label: "Density",
      options: [["comfortable", "Comfortable"], ["compact", "Compact"]]
    }]
  }, {
    id: "motion",
    label: "Behavior & Performance",
    glyph: "◔",
    rows: [{
      k: "reduceMotion",
      type: "switch",
      label: "Reduced motion",
      hint: "calm cursor, pulse & transitions"
    }, {
      k: "sound",
      type: "slider",
      min: 0,
      max: 100,
      step: 10,
      label: "Sound effects",
      fmt: function (v) {
        return v ? v + "%" : "off";
      },
      hint: "subtle UI feedback volume"
    }, {
      type: "info",
      text: "Motion is minimal and mechanical by design. Reduced motion also lowers idle-dashboard CPU."
    }]
  }, {
    id: "security",
    label: "Security Modes",
    glyph: "⛨",
    grounded: "engine env flags & /detect/three-r",
    rows: [{
      k: "amaCrypto",
      type: "switch",
      label: "AMA Cryptography (PQC)",
      hint: "Kyber-1024 · ML-DSA-65 · SLH-DSA"
    }, {
      k: "constantTime",
      type: "switch",
      label: "Constant-time path",
      hint: "AMA_REQUIRE_CONSTANT_TIME"
    }, {
      k: "threeR",
      type: "switch",
      label: "3R Mechanism",
      hint: "Recursion · Resonance · Refactoring"
    }, {
      k: "sigmaGate",
      type: "switch",
      label: "σ_Immutable gate",
      hint: "Lyapunov-stable ethics hard-gate"
    }, {
      type: "info",
      text: "These mirror real engine controls. In production AMA/PQC is mandatory at import (AMA_REQUIRE_REAL_PQC); the toggles model operator-visible posture, not a way to disable cryptography on a live node."
    }, {
      type: "button",
      label: "⏻ Mercury / AMA Disconnect — quarantine the system",
      action: "disconnect",
      danger: true
    }, {
      type: "info",
      text: "The fail-closed safety control. Disconnecting quarantines every surface — the PQC import gate fails closed, the agent kill switch halts all cognition, σ_Immutable holds closed — until an authorized operator deliberately re-activates. A threat cannot lift it; only you can. The same control lives in the navigator (⏻)."
    }]
  }, {
    id: "accessibility",
    label: "Accessibility",
    glyph: "✦",
    rows: [{
      k: "keyboardFocus",
      type: "switch",
      label: "Keyboard focus rings",
      hint: "strong visible focus outline"
    }, {
      k: "underlineLinks",
      type: "switch",
      label: "Underline links",
      hint: "non-color link cue"
    }, {
      k: "textToSpeech",
      type: "switch",
      label: "Text-to-speech",
      hint: "voice the agent's messages",
      test: "speak"
    }, {
      k: "highContrast",
      type: "switch",
      label: "High contrast mode",
      hint: "shared with Appearance"
    }, {
      type: "info",
      text: "Screen-reader landmarks, ARIA labels and full keyboard navigation are built into every surface. Camera, microphone and location are never requested — Mercury runs locally."
    }]
  }, {
    id: "notifications",
    label: "Notifications",
    glyph: "◇",
    rows: [{
      k: "notifyInApp",
      type: "switch",
      label: "In-app",
      hint: "banner & dock alerts"
    }, {
      k: "notifyCriticalOnly",
      type: "switch",
      label: "Critical only",
      hint: "suppress low/medium"
    }, {
      k: "notifyEmail",
      type: "switch",
      label: "Email digest",
      hint: "operator summary"
    }, {
      k: "notifyPush",
      type: "switch",
      label: "Push",
      hint: "requires a deployed gateway"
    }, {
      type: "info",
      text: "Interface preferences for a reference design. A live deployment wires these to its own alerting gateway; no notifications are sent from this artifact."
    }]
  }, {
    id: "privacy",
    label: "Privacy & Data",
    glyph: "⊘",
    grounded: "local-first · audit-trailed · consent-gated",
    rows: [{
      k: "consentAudit",
      type: "switch",
      label: "Audit-trail logging",
      hint: "full decision provenance"
    }, {
      k: "consentDiagnostics",
      type: "switch",
      label: "Local diagnostics",
      hint: "on-device only"
    }, {
      type: "info",
      text: "Mercury is local-first: detections are logged with full audit trails, and no personal data is collected without explicit consent (docs/USE_CASES.md). The platform ships no vendor telemetry or phone-home; benchmark lanes run offline and deterministic."
    }, {
      type: "links",
      items: [["License — GPL v3", "about:license"], ["Compliance — GDPR Art. 22", "about:compliance"]]
    }]
  }, {
    id: "content",
    label: "Content & Preferences",
    glyph: "▤",
    rows: [{
      k: "units",
      type: "segment",
      label: "Default units",
      hint: "distance display · km / mi",
      options: [["metric", "Metric"], ["imperial", "Imperial"]]
    }, {
      k: "landing",
      type: "select",
      label: "Default surface",
      hint: "where a cold launch lands",
      options: [["hub", "System Hub"], ["console", "Console"], ["explorer", "Explorer"], ["hazard", "Hazard Console"], ["cli", "CLI"], ["crypto", "Cryptography"]]
    }, {
      type: "info",
      text: "Units bind the distance-bearing surfaces — the Hazard Console's range filter and per-event distances switch between km and miles. Default surface redirects a cold launch to your chosen surface; the ♱ home button always returns you to the hub. Saved searches & favorites live on each surface (the Explorer's facets, the Hazard presets) and persist per device."
    }]
  }, {
    id: "about",
    label: "App Information",
    glyph: "♱",
    rows: [{
      type: "kv",
      items: [["Design system", "v1.0.0"], ["Engine target", "Mercury Agent v2.0.0"], ["AMA Cryptography", "v3.2.0 (pinned)"], ["License", "GPL v3"]]
    }, {
      type: "links",
      items: [["Documentation & README", "README.md"], ["Design System Review", "review/Mercury Agent - Design System Review.html"], ["Report a bug / feedback", "feedback"]]
    }, {
      type: "button",
      label: "Reset all settings to defaults",
      action: "reset",
      danger: true
    }, {
      type: "info",
      text: "Mercury Agent ♱ · Steel Security Advisors LLC · Archetype for a civilized evolution."
    }]
  }];

  // ==========================================================================
  //  STYLES
  // ==========================================================================
  var INJECTED = false;
  function injectCSS() {
    if (INJECTED) return;
    INJECTED = true;
    var css = "" + "mercury-settings{position:fixed;inset:0;z-index:11500;display:none;font-family:var(--font-sans,sans-serif);}" + "mercury-settings[open]{display:block;}" + ".mst-scrim{position:absolute;inset:0;background:rgba(6,8,11,.66);backdrop-filter:blur(4px);}" + ".mst-box{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:min(820px,94vw);height:min(620px,90vh);" + "display:flex;background:var(--surface-1,#0f141c);border:1px solid var(--border-strong,#36465a);border-radius:var(--r-lg,12px);" + "box-shadow:0 30px 90px rgba(0,0,0,.6);overflow:hidden;}" + ".mst-rail{flex:none;width:222px;background:var(--bg,#0a0e14);border-right:1px solid var(--border,#243040);display:flex;flex-direction:column;overflow-y:auto;}" + ".mst-rail-h{display:flex;align-items:center;gap:9px;padding:17px 18px 14px;border-bottom:1px solid var(--border-faint,#161e28);}" + ".mst-rail-h .s{font-family:var(--font-serif,serif);font-size:21px;color:var(--brand-gold,#B4B124);}" + ".mst-rail-h .t{font-family:var(--font-mono,monospace);font-size:12px;letter-spacing:.08em;color:var(--fg,#e6edf3);}" + ".mst-nav{display:flex;flex-direction:column;padding:8px 0;flex:1;}" + ".mst-navi{display:flex;align-items:center;gap:11px;padding:10px 18px;cursor:pointer;border-left:2px solid transparent;background:none;border-top:none;border-right:none;border-bottom:none;text-align:left;width:100%;}" + ".mst-navi:hover{background:var(--surface-2,#151c26);}" + ".mst-navi.on{border-left-color:var(--mercury-500,#00bcd4);background:var(--surface-2,#151c26);}" + ".mst-navi .g{flex:none;width:18px;text-align:center;color:var(--mercury-300,#4fc3f7);font-family:var(--font-mono,monospace);font-size:14px;}" + ".mst-navi .l{font-family:var(--font-mono,monospace);font-size:11.5px;color:var(--fg-2,#a9b6c6);letter-spacing:.01em;}" + ".mst-navi.on .l{color:var(--fg,#e6edf3);}" + ".mst-pane{flex:1;min-width:0;display:flex;flex-direction:column;}" + ".mst-pane-h{display:flex;align-items:center;justify-content:space-between;padding:16px 20px;border-bottom:1px solid var(--border-faint,#161e28);}" + ".mst-pane-t{font-family:var(--font-mono,monospace);font-size:14px;font-weight:600;color:var(--fg,#e6edf3);letter-spacing:.01em;}" + ".mst-grounded{font-family:var(--font-mono,monospace);font-size:8.5px;letter-spacing:.06em;color:var(--mercury-300,#4fc3f7);margin-top:3px;display:flex;align-items:center;gap:5px;}" + ".mst-grounded::before{content:'◆';font-size:7px;}" + ".mst-x{cursor:pointer;color:var(--fg-3,#6b7888);font-size:18px;line-height:1;padding:3px 7px;border-radius:4px;background:none;border:none;}" + ".mst-x:hover{background:var(--surface-3,#1c2530);color:var(--fg,#e6edf3);}" + ".mst-body{flex:1;overflow-y:auto;padding:6px 20px 22px;}" + ".mst-row{display:flex;align-items:center;gap:16px;padding:13px 0;border-bottom:1px solid var(--border-faint,#161e28);}" + ".mst-row:last-child{border-bottom:none;}" + ".mst-rl{flex:1;min-width:0;}" + ".mst-rl b{display:block;font-family:var(--font-sans,sans-serif);font-size:13px;color:var(--fg,#e6edf3);font-weight:600;}" + ".mst-rl span{display:block;font-family:var(--font-mono,monospace);font-size:9.5px;color:var(--fg-muted,#495567);margin-top:3px;letter-spacing:.02em;}" + ".mst-rc{flex:none;display:flex;align-items:center;gap:8px;}" +
    // switch
    ".mst-sw{width:40px;height:22px;border-radius:999px;border:1px solid var(--border-strong,#36465a);background:var(--surface-3,#1c2530);cursor:pointer;position:relative;transition:background .14s,border-color .14s;}" + ".mst-sw::after{content:'';position:absolute;top:2px;left:2px;width:16px;height:16px;border-radius:50%;background:var(--fg-3,#6b7888);transition:transform .14s,background .14s;}" + ".mst-sw.on{background:var(--mercury-900,#0c3b44);border-color:var(--mercury-500,#00bcd4);}" + ".mst-sw.on::after{transform:translateX(18px);background:var(--mercury-300,#4fc3f7);}" +
    // segment
    ".mst-seg{display:flex;gap:1px;background:var(--border-faint,#161e28);border:1px solid var(--border,#243040);border-radius:var(--r-sm,5px);overflow:hidden;}" + ".mst-seg button{font-family:var(--font-mono,monospace);font-size:11px;color:var(--fg-3,#6b7888);background:var(--surface-1,#0f141c);border:none;padding:6px 12px;cursor:pointer;transition:all .12s;}" + ".mst-seg button:hover{color:var(--fg-2,#a9b6c6);}" + ".mst-seg button.on{background:var(--mercury-900,#0c3b44);color:var(--mercury-100,#bdf0fa);}" +
    // swatch
    ".mst-sws{display:flex;gap:7px;}" + ".mst-swc{width:24px;height:24px;border-radius:50%;cursor:pointer;border:2px solid transparent;transition:transform .12s;position:relative;}" + ".mst-swc:hover{transform:scale(1.12);}" + ".mst-swc.on{border-color:var(--fg,#e6edf3);box-shadow:0 0 0 1px var(--bg,#0a0e14),0 0 8px rgba(255,255,255,.25);}" +
    // slider
    ".mst-sld{display:flex;align-items:center;gap:11px;}" + ".mst-sld input{width:130px;-webkit-appearance:none;appearance:none;height:3px;border-radius:2px;background:var(--surface-3,#1c2530);outline:none;}" + ".mst-sld input::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;border-radius:50%;background:var(--mercury-400,#18c8e0);cursor:pointer;box-shadow:0 0 0 3px var(--surface-1,#0f141c);}" + ".mst-sld input::-moz-range-thumb{width:14px;height:14px;border:none;border-radius:50%;background:var(--mercury-400,#18c8e0);cursor:pointer;}" + ".mst-sld b{font-family:var(--font-mono,monospace);font-size:11px;color:var(--mercury-200,#8ae3f5);min-width:40px;text-align:right;}" +
    // select
    ".mst-sel{font-family:var(--font-mono,monospace);font-size:11px;color:var(--fg,#e6edf3);background:var(--surface-2,#151c26);border:1px solid var(--border,#243040);border-radius:var(--r-sm,5px);padding:7px 10px;cursor:pointer;}" + ".mst-sel:focus{outline:none;border-color:var(--mercury-500,#00bcd4);}" +
    // test btn
    ".mst-test{font-family:var(--font-mono,monospace);font-size:9.5px;color:var(--mercury-200,#8ae3f5);background:var(--surface-2,#151c26);border:1px solid var(--mercury-800,#0a5d6b);border-radius:999px;padding:4px 10px;cursor:pointer;}" + ".mst-test:hover{background:var(--mercury-900,#0c3b44);}" +
    // info / kv / links / button
    ".mst-info{font-family:var(--font-sans,sans-serif);font-size:11.5px;line-height:1.6;color:var(--fg-3,#6b7888);background:var(--surface-2,#151c26);border:1px solid var(--border-faint,#161e28);border-left:2px solid var(--mercury-700,#007e92);border-radius:0 var(--r-sm,5px) var(--r-sm,5px) 0;padding:11px 13px;margin:13px 0 2px;}" + ".mst-kv{display:grid;grid-template-columns:1fr auto;gap:1px;background:var(--border-faint,#161e28);border:1px solid var(--border-faint,#161e28);border-radius:var(--r-sm,5px);overflow:hidden;margin:13px 0 2px;}" + ".mst-kv .kk,.mst-kv .vv{background:var(--surface-1,#0f141c);padding:9px 12px;font-family:var(--font-mono,monospace);font-size:11px;}" + ".mst-kv .kk{color:var(--fg-3,#6b7888);} .mst-kv .vv{color:var(--fg,#e6edf3);text-align:right;}" + ".mst-lnks{display:flex;flex-direction:column;gap:7px;margin:13px 0 2px;}" + ".mst-lnk{display:flex;align-items:center;gap:9px;font-family:var(--font-mono,monospace);font-size:11.5px;color:var(--fg-2,#a9b6c6);background:var(--surface-2,#151c26);border:1px solid var(--border,#243040);border-radius:var(--r-sm,5px);padding:9px 12px;cursor:pointer;text-decoration:none;transition:all .12s;}" + ".mst-lnk:hover{border-color:var(--mercury-600,#00a0b6);color:var(--mercury-200,#8ae3f5);}" + ".mst-lnk .a{color:var(--mercury-400,#18c8e0);}" + ".mst-btn{font-family:var(--font-mono,monospace);font-size:11px;color:var(--anomaly-soft,#ff7a7e);background:transparent;border:1px solid var(--anomaly-deep,#b3231f);border-radius:var(--r-sm,5px);padding:9px 14px;cursor:pointer;margin:13px 0 2px;transition:all .12s;}" + ".mst-btn:hover{background:var(--anomaly-bg,rgba(255,77,82,.1));}" + ".mst-foot{flex:none;display:flex;align-items:center;justify-content:space-between;padding:10px 20px;border-top:1px solid var(--border-faint,#161e28);font-family:var(--font-mono,monospace);font-size:9px;color:var(--fg-muted,#495567);}" + ".mst-foot .kbd{border:1px solid var(--border,#243040);border-radius:3px;padding:2px 6px;color:var(--fg-3,#6b7888);}" + "@media(max-width:640px){.mst-box{flex-direction:column;height:92vh;width:96vw;}.mst-rail{width:100%;flex-direction:row;overflow-x:auto;}.mst-nav{flex-direction:row;}.mst-navi{border-left:none;border-bottom:2px solid transparent;white-space:nowrap;}.mst-navi.on{border-left:none;border-bottom-color:var(--mercury-500);}.mst-navi .l{display:none;}}";
    var s = document.createElement("style");
    s.id = "mercury-settings-css";
    s.textContent = css;
    document.head.appendChild(s);
  }

  // ==========================================================================
  //  ELEMENT
  // ==========================================================================
  var current = "appearance";
  var rootPath = ""; // resolved when opened, for about-links

  var El = class extends HTMLElement {
    connectedCallback() {
      injectCSS();
      this._build();
    }
    _build() {
      var navHtml = SECTIONS.map(function (sec) {
        return '<button class="mst-navi' + (sec.id === current ? " on" : "") + '" data-sec="' + sec.id + '">' + '<span class="g">' + sec.glyph + "</span><span class=\"l\">" + sec.label + "</span></button>";
      }).join("");
      this.innerHTML = '<div class="mst-scrim" data-close></div>' + '<div class="mst-box" role="dialog" aria-label="Settings">' + '<div class="mst-rail"><div class="mst-rail-h"><span class="s">♱</span><span class="t">Settings</span></div>' + '<div class="mst-nav">' + navHtml + "</div></div>" + '<div class="mst-pane">' + '<div class="mst-pane-h"><div><div class="mst-pane-t" data-title></div><div class="mst-grounded" data-grounded style="display:none"></div></div><button class="mst-x" data-close>✕</button></div>' + '<div class="mst-body" data-body></div>' + '<div class="mst-foot"><span>Settings persist on this device · applied across every surface</span><span class="kbd">ESC</span></div>' + "</div>" + "</div>";
      var self = this;
      this.querySelectorAll("[data-close]").forEach(function (n) {
        n.addEventListener("click", function () {
          self.close();
        });
      });
      this.querySelectorAll("[data-sec]").forEach(function (n) {
        n.addEventListener("click", function () {
          current = n.getAttribute("data-sec");
          self._build();
        });
      });
      this._renderPane();
    }
    _renderPane() {
      var sec = SECTIONS.find(function (s) {
        return s.id === current;
      });
      this.querySelector("[data-title]").textContent = sec.label;
      var gEl = this.querySelector("[data-grounded]");
      if (sec.grounded) {
        gEl.style.display = "flex";
        gEl.textContent = "grounded · " + sec.grounded;
      } else {
        gEl.style.display = "none";
      }
      var body = this.querySelector("[data-body]");
      var self = this;
      body.innerHTML = sec.rows.map(function (r, i) {
        return self._renderRow(r, i);
      }).join("");
      this._wireRows(body, sec);
    }
    _renderRow(r, i) {
      if (r.type === "info") return '<div class="mst-info">' + r.text + "</div>";
      if (r.type === "kv") return '<div class="mst-kv">' + r.items.map(function (it) {
        return '<div class="kk">' + it[0] + '</div><div class="vv">' + it[1] + "</div>";
      }).join("") + "</div>";
      if (r.type === "links") return '<div class="mst-lnks">' + r.items.map(function (it, j) {
        return '<a class="mst-lnk" data-link="' + it[1] + '" data-ri="' + i + '" data-lj="' + j + '"><span class="a">→</span>' + it[0] + "</a>";
      }).join("") + "</div>";
      if (r.type === "button") return '<button class="mst-btn" data-action="' + r.action + '">' + r.label + "</button>";
      var ctl = "";
      var val = S[r.k];
      if (r.type === "switch") ctl = '<button class="mst-sw' + (val ? " on" : "") + '" data-sw="' + r.k + '" role="switch" aria-checked="' + !!val + '"></button>' + (r.test ? ' <button class="mst-test" data-test="' + r.test + '">test</button>' : "");else if (r.type === "segment") ctl = '<div class="mst-seg">' + r.options.map(function (o) {
        return '<button data-seg="' + r.k + '" data-v="' + o[0] + '" class="' + (val === o[0] ? "on" : "") + '">' + o[1] + "</button>";
      }).join("") + "</div>";else if (r.type === "swatch") ctl = '<div class="mst-sws">' + Object.keys(ACCENTS).map(function (a) {
        return '<span class="mst-swc' + (val === a ? " on" : "") + '" data-acc="' + a + '" title="' + ACCENTS[a].label + '" style="background:' + ACCENTS[a].hex + '"></span>';
      }).join("") + "</div>";else if (r.type === "slider") ctl = '<div class="mst-sld"><input type="range" min="' + r.min + '" max="' + r.max + '" step="' + r.step + '" value="' + val + '" data-sld="' + r.k + '"><b data-sldv>' + r.fmt(val) + "</b></div>";else if (r.type === "select") ctl = '<select class="mst-sel" data-sel="' + r.k + '">' + r.options.map(function (o) {
        return '<option value="' + o[0] + '"' + (val === o[0] ? " selected" : "") + ">" + o[1] + "</option>";
      }).join("") + "</select>";
      return '<div class="mst-row"><div class="mst-rl"><b>' + r.label + "</b>" + (r.hint ? "<span>" + r.hint + "</span>" : "") + '</div><div class="mst-rc">' + ctl + "</div></div>";
    }
    _wireRows(body, sec) {
      var self = this;
      function changed(k) {
        persist();
        apply();
        blip();
        if (k === "theme" || k === "accent" || k === "highContrast" || k === "scale" || k === "density") self._renderPane();
      }
      body.querySelectorAll("[data-sw]").forEach(function (n) {
        n.addEventListener("click", function () {
          var k = n.getAttribute("data-sw");
          S[k] = !S[k];
          n.classList.toggle("on", S[k]);
          n.setAttribute("aria-checked", S[k]);
          changed(k);
        });
      });
      body.querySelectorAll("[data-seg]").forEach(function (n) {
        n.addEventListener("click", function () {
          var k = n.getAttribute("data-seg");
          S[k] = n.getAttribute("data-v");
          changed(k);
        });
      });
      body.querySelectorAll("[data-acc]").forEach(function (n) {
        n.addEventListener("click", function () {
          S.accent = n.getAttribute("data-acc");
          changed("accent");
        });
      });
      body.querySelectorAll("[data-sld]").forEach(function (n) {
        n.addEventListener("input", function () {
          var k = n.getAttribute("data-sld");
          S[k] = +n.value;
          var row = SECTIONS.reduce(function (acc, s) {
            return acc || s.rows.find(function (rr) {
              return rr.k === k;
            });
          }, null);
          var b = n.parentNode.querySelector("[data-sldv]");
          if (b && row) b.textContent = row.fmt(S[k]);
          persist();
          apply();
        });
      });
      body.querySelectorAll("[data-sel]").forEach(function (n) {
        n.addEventListener("change", function () {
          S[n.getAttribute("data-sel")] = n.value;
          changed();
        });
      });
      body.querySelectorAll("[data-test]").forEach(function (n) {
        n.addEventListener("click", function () {
          speak("Mercury Agent. Text to speech is active. The active anomaly is within nominal bounds.");
        });
      });
      body.querySelectorAll("[data-action]").forEach(function (n) {
        n.addEventListener("click", function () {
          var act = n.getAttribute("data-action");
          if (act === "reset") {
            S = Object.assign({}, DEFAULTS);
            persist();
            apply();
            self._build();
          } else if (act === "disconnect") {
            self.close();
            if (window.MercuryQuarantine) {
              window.MercuryQuarantine.engage();
            } else {
              var sc = document.createElement("script");
              sc.src = rootPath + "components/mercury-quarantine.js";
              sc.onload = function () {
                window.MercuryQuarantine && window.MercuryQuarantine.engage();
              };
              document.head.appendChild(sc);
            }
          }
        });
      });
      body.querySelectorAll("[data-link]").forEach(function (n) {
        n.addEventListener("click", function (e) {
          e.preventDefault();
          self._handleLink(n.getAttribute("data-link"));
        });
      });
    }
    _handleLink(target) {
      if (target === "feedback") {
        window.location.href = "mailto:feedback@steelsecadv.example?subject=Mercury%20Agent%20Design%20System%20feedback";
        return;
      }
      if (target.indexOf("about:") === 0) {
        return;
      } // grounded labels, no external target in a reference design
      this.close();
      window.MercuryNav ? window.MercuryNav.go(rootPath + target) : window.location.href = rootPath + target;
    }
    open() {
      this.setAttribute("open", "");
      this._onKey = this._onKey.bind(this);
      document.addEventListener("keydown", this._onKey);
    }
    close() {
      this.removeAttribute("open");
      document.removeEventListener("keydown", this._onKey);
    }
    _onKey(e) {
      if (e.key === "Escape") this.close();
    }
  };
  customElements.define("mercury-settings", El);
  function ensure() {
    var el = document.querySelector("mercury-settings");
    if (!el) {
      el = document.createElement("mercury-settings");
      document.body.appendChild(el);
    }
    return el;
  }
  window.MercurySettings = {
    __ready: true,
    open: function (root) {
      if (root != null) rootPath = root;
      ensure().open();
    },
    close: function () {
      var el = document.querySelector("mercury-settings");
      if (el) el.close();
    },
    get: function (k) {
      return S[k];
    },
    set: function (k, v) {
      S[k] = v;
      persist();
      apply();
    },
    apply: apply,
    all: function () {
      return Object.assign({}, S);
    }
  };
})();
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/mercury-settings.js", error: String((e && e.message) || e) }); }

// components/mercury-state.js
try { (() => {
/* ============================================================================
   <mercury-state> — Mercury Agent · the canonical edge-state atom (web component)
   ----------------------------------------------------------------------------
   A thin generator over the canonical `.ms-state*` classes defined ONCE in
   colors_and_type.css. It injects NO colors of its own — it emits the shared
   markup so an empty window / loading detector / API error / 503-unavailable /
   ethical withhold / quarantine reads identically across Console · Crypto ·
   Explorer · Hazard · CLI. The single source of truth for the system's non-happy
   paths, the companion to <mercury-status> for its happy ones.

   Usage (plain custom element — HTML or JSX):
     <mercury-state kind="empty" title="No detections in this window"
        detail="Nothing has crossed the threshold in the last 60s."></mercury-state>

     <mercury-state kind="loading" title="Querying detector…"></mercury-state>

     <mercury-state kind="error" title="Detection failed" code="HTTP 422 · {detail}"
        corr="a1b2c3d4" detail="The request body failed validation."
        retry></mercury-state>

     <mercury-state kind="offline" title="Neurosymbolic path unavailable"
        code="HTTP 503" detail="Falling back to the statistical ensemble."></mercury-state>

     <mercury-state kind="ethical" title="Action withheld"
        code='EthicalConstraintViolationError(check="benevolence")'
        detail="η 0.984 < 0.99. There is no advisory mode."></mercury-state>

     <mercury-state kind="quarantined" title="System quarantined"></mercury-state>

   Attributes
     kind   — empty | loading | error | offline | ethical | quarantined  (default empty)
     title  — the line shown in mono
     detail — one line of sans context (optional)
     code   — a mono code chip: HTTP status / error class / etc. (optional)
     corr   — a correlation id, appended to the code chip as "corr <id>" (optional)
     retry  — boolean; renders a primary "↻ Retry" button that fires a
              `mercury-state-retry` event (bubbles) you can listen for
     glyph  — override the default per-kind glyph
   Slotted children (light DOM) are appended into the actions row, so callers can
   drop in their own buttons/links instead of (or beside) retry.
   Requires colors_and_type.css on the page.
   ============================================================================ */
(function () {
  if (window.customElements && customElements.get("mercury-state")) return;
  var GLYPH = {
    empty: "∅",
    loading: "◴",
    error: "✕",
    offline: "⚠",
    ethical: "η",
    quarantined: "⏻"
  };
  var TITLE = {
    empty: "Nothing here yet",
    loading: "Working…",
    error: "Something went wrong",
    offline: "Path unavailable",
    ethical: "Action withheld",
    quarantined: "System quarantined"
  };
  var MercuryState = class extends HTMLElement {
    static get observedAttributes() {
      return ["kind", "title", "detail", "code", "corr", "retry", "glyph"];
    }
    connectedCallback() {
      // stash any author-provided action nodes once, before we own innerHTML
      if (!this._slotHTML) {
        var kids = this.querySelector("[slot=actions]") || (this.children.length ? null : null);
        this._slotHTML = (this.querySelector("[slot=actions]") || {}).innerHTML || "";
      }
      this.render();
    }
    attributeChangedCallback() {
      if (this.isConnected) this.render();
    }
    render() {
      var kind = (this.getAttribute("kind") || "empty").toLowerCase();
      if (!GLYPH[kind]) kind = "empty";
      var title = this.getAttribute("title") || TITLE[kind];
      var detail = this.getAttribute("detail") || "";
      var code = this.getAttribute("code") || "";
      var corr = this.getAttribute("corr") || "";
      var glyph = this.getAttribute("glyph") || GLYPH[kind];
      var retry = this.hasAttribute("retry");
      var codeText = code;
      if (corr) codeText = (code ? code + "  ·  " : "") + "corr " + corr;
      var html = '<div class="ms-state ms-state--' + kind + '" role="status">' + '<span class="ms-state-ico" aria-hidden="true">' + glyph + "</span>" + '<span class="ms-state-title">' + esc(title) + "</span>" + (detail ? '<span class="ms-state-detail">' + esc(detail) + "</span>" : "") + (codeText ? '<span class="ms-state-code">' + esc(codeText) + "</span>" : "") + '<span class="ms-state-actions" data-actions>' + (retry ? '<button class="ms-state-btn primary" data-retry>↻ Retry</button>' : "") + "</span>" + "</div>";
      this.innerHTML = html;
      var actions = this.querySelector("[data-actions]");
      if (this._slotHTML && actions) {
        var span = document.createElement("span");
        span.innerHTML = this._slotHTML;
        while (span.firstChild) actions.appendChild(span.firstChild);
      }
      if (!actions.children.length) actions.remove();
      var btn = this.querySelector("[data-retry]");
      if (btn) {
        var self = this;
        btn.addEventListener("click", function () {
          self.dispatchEvent(new CustomEvent("mercury-state-retry", {
            bubbles: true
          }));
        });
      }
    }
  };
  function esc(s) {
    return String(s).replace(/[&<>"]/g, function (c) {
      return {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;"
      }[c];
    });
  }
  customElements.define("mercury-state", MercuryState);
  window.MercuryState = window.MercuryState || {};
  window.MercuryState.GLYPH = GLYPH;
})();
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/mercury-state.js", error: String((e && e.message) || e) }); }

// components/mercury-status.js
try { (() => {
/* ============================================================================
   <mercury-status> — Mercury Agent · the canonical status atom (web component)
   ----------------------------------------------------------------------------
   A thin, ergonomic generator over the canonical `.ms-*` status classes defined
   ONCE in colors_and_type.css. It injects NO colors of its own — it only emits
   the shared markup, so a NORMAL / ANOMALY / severity / TLP atom renders
   identically across Console · CLI · Crypto · Explorer · preview cards. This is
   the single source of truth for the system's most-repeated atom (review K2b).

   Usage (plain custom element — works in HTML or JSX):
     <mercury-status state="ANOMALY"></mercury-status>            → dot+word pill
     <mercury-status state="SEVERE" kind="sev"></mercury-status>  → severity tag
     <mercury-status state="TLP:AMBER" kind="tlp"></mercury-status>
     <mercury-status state="online" tone="normal" kind="dot" live></mercury-status>

   Attributes
     state — the word shown (rendered ALL-CAPS for badge/sev/tlp; never emoji)
     tone  — explicit hue (anomaly|normal|warning|cognitive|benevolence|mercury|
             neutral). Omit to auto-infer from the documented vocabulary.
     kind  — badge (default) | sev | tlp | dot
     live  — boolean; pulses the dot (respects prefers-reduced-motion via the
             token sheet's .ms-dot--live)
   Requires colors_and_type.css to be loaded on the page.
   ============================================================================ */
(function () {
  if (window.customElements && customElements.get("mercury-status")) return;

  // documented word → tone
  var TONE = {
    ANOMALY: "anomaly",
    CRITICAL: "anomaly",
    SEVERE: "anomaly",
    HIGH: "anomaly",
    FAIL: "anomaly",
    FAILED: "anomaly",
    TAMPER: "anomaly",
    BREACH: "anomaly",
    LOCKDOWN: "anomaly",
    NORMAL: "normal",
    PASS: "normal",
    OK: "normal",
    STABLE: "normal",
    CLEARED: "normal",
    NOMINAL: "normal",
    ONLINE: "normal",
    ALIGNED: "normal",
    ENFORCED: "normal",
    COMPLETED: "normal",
    VERIFIED: "verified",
    ATTESTED: "verified",
    SIGNED: "verified",
    AUTHENTIC: "verified",
    // cryptographic confirmation — dedicated blue, distinct from nominal green
    WARNING: "warning",
    MEDIUM: "warning",
    MODERATE: "warning",
    ELEVATED: "warning",
    DEGRADED: "warning",
    PARTIAL: "warning",
    MONITORED: "warning",
    PENDING: "warning",
    WITHHELD: "withheld",
    HELD: "withheld",
    REFUSED: "withheld",
    SUPPRESSED: "withheld",
    // ethics-gate hold — dedicated light grey, not an amber warning
    GOSNN: "cognitive",
    NEUROSYMBOLIC: "cognitive",
    COGNITIVE: "cognitive",
    PQC: "cognitive",
    RUNNING: "cognitive",
    BENEVOLENCE: "benevolence",
    ETHICAL: "benevolence",
    LOW: "neutral",
    MINOR: "neutral",
    CANCELLED: "neutral",
    LIVE: "mercury",
    STANDARD: "mercury",
    HARDENED: "mercury"
  };
  // tone → dot color token (the inner dot of a badge; base size/shape is canonical)
  var DOT = {
    anomaly: "var(--anomaly)",
    normal: "var(--normal)",
    warning: "var(--warning)",
    cognitive: "var(--cognitive)",
    benevolence: "var(--benevolence)",
    verified: "var(--verified)",
    withheld: "var(--withheld)",
    mercury: "var(--mercury-500)",
    neutral: "var(--fg-3)"
  };
  function toneFor(word) {
    if (!word) return "neutral";
    var k = String(word).toUpperCase().replace(/[^A-Z+:]/g, "");
    return TONE[k] || "neutral";
  }
  // arbitrary severity word → the canonical 4-rung sev key
  function sevKey(word) {
    var k = String(word).toUpperCase();
    if (k.indexOf("CRIT") >= 0) return "critical";
    if (k.indexOf("HIGH") >= 0 || k.indexOf("SEVERE") >= 0) return "high";
    if (k.indexOf("MED") >= 0 || k.indexOf("MOD") >= 0 || k.indexOf("WARN") >= 0) return "medium";
    return "low";
  }
  function tlpBand(word) {
    var k = String(word).toUpperCase();
    if (k.indexOf("RED") >= 0) return "red";
    if (k.indexOf("AMBER") >= 0) return "amber";
    if (k.indexOf("GREEN") >= 0) return "green";
    return "clear";
  }
  var MercuryStatus = class extends HTMLElement {
    static get observedAttributes() {
      return ["state", "tone", "kind", "live"];
    }
    connectedCallback() {
      this.render();
    }
    attributeChangedCallback() {
      if (this.isConnected) this.render();
    }
    render() {
      var state = this.getAttribute("state") || "";
      var kind = this.getAttribute("kind") || "badge";
      var live = this.hasAttribute("live");
      var tone = this.getAttribute("tone") || toneFor(state);
      var WORD = String(state).toUpperCase();
      if (kind === "tlp") {
        this.innerHTML = '<span class="ms-tlp ms-tlp--' + tlpBand(state) + '">' + WORD + "</span>";
      } else if (kind === "sev") {
        this.innerHTML = '<span class="ms-sev ms-sev--' + sevKey(state) + '">' + WORD + "</span>";
      } else if (kind === "dot") {
        var dcls = "ms-dot" + (live ? " ms-dot--live" : "");
        var dstyle = DOT[tone] && !live ? ' style="background:' + DOT[tone] + '"' : "";
        this.innerHTML = '<span class="ms-word"><span class="' + dcls + '"' + dstyle + '></span>' + '<span style="color:var(--fg-2);font-weight:500;letter-spacing:.04em">' + state + "</span></span>";
      } else {
        // badge
        var lc = live ? " ms-dot--live" : "";
        this.innerHTML = '<span class="ms-badge ms-badge--' + tone + '">' + '<span class="ms-dot' + lc + '" style="background:' + (DOT[tone] || "var(--fg-3)") + '"></span>' + WORD + "</span>";
      }
    }
  };
  customElements.define("mercury-status", MercuryStatus);

  // also expose the vocabulary for JS surfaces that resolve a tone without an element
  window.MercuryStatus = {
    toneFor: toneFor,
    sevKey: sevKey,
    tlpBand: tlpBand,
    DOT: DOT,
    TONE: TONE
  };
})();
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/mercury-status.js", error: String((e && e.message) || e) }); }

// design-canvas.jsx
try { (() => {
/* BEGIN USAGE */
// DesignCanvas.jsx — Figma-ish design canvas wrapper
// Warm gray grid bg + Sections + Artboards + PostIt notes.
// Exports (to window): DesignCanvas, DCSection, DCArtboard, DCPostIt.
// Artboards are reorderable (grip-drag), deletable, labels/titles are
// inline-editable, and any artboard can be opened in a fullscreen focus
// overlay (←/→/Esc). State persists to a .design-canvas.state.json sidecar
// via the host bridge. No assets, no deps.
//
// Usage:
//   <DesignCanvas>
//     <DCSection id="onboarding" title="Onboarding" subtitle="First-run variants">
//       <DCArtboard id="a" label="A · Dusk" width={260} height={480}>…</DCArtboard>
//       <DCArtboard id="b" label="B · Minimal" width={260} height={480}>…</DCArtboard>
//     </DCSection>
//   </DesignCanvas>
//
// Artboards are static design frames, not scroll regions — never use
// height: 100% + overflow: auto/scroll on inner elements; size each artboard
// to fit its content (explicit pixel height, or let it grow).
/* END USAGE */

const DC = {
  bg: '#f0eee9',
  grid: 'rgba(0,0,0,0.06)',
  label: 'rgba(60,50,40,0.7)',
  title: 'rgba(40,30,20,0.85)',
  subtitle: 'rgba(60,50,40,0.6)',
  postitBg: '#fef4a8',
  postitText: '#5a4a2a',
  font: '-apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif'
};

// One-time CSS injection (classes are dc-prefixed so they don't collide with
// the hosted design's own styles).
if (typeof document !== 'undefined' && !document.getElementById('dc-styles')) {
  const s = document.createElement('style');
  s.id = 'dc-styles';
  s.textContent = ['.dc-editable{cursor:text;outline:none;white-space:nowrap;border-radius:3px;padding:0 2px;margin:0 -2px}', '.dc-editable:focus{background:#fff;box-shadow:0 0 0 1.5px #c96442}', '[data-dc-slot]{transition:transform .18s cubic-bezier(.2,.7,.3,1)}', '[data-dc-slot].dc-dragging{transition:none;z-index:10;pointer-events:none}', '[data-dc-slot].dc-dragging .dc-card{box-shadow:0 12px 40px rgba(0,0,0,.25),0 0 0 2px #c96442;transform:scale(1.02)}',
  // isolation:isolate contains artboard content's z-indexes so a
  // z-indexed child (sticky navbar etc.) can't paint over .dc-header or
  // the .dc-menu popover that drops into the top of the card.
  '.dc-card{isolation:isolate;transition:box-shadow .15s,transform .15s}', '.dc-card *{scrollbar-width:none}', '.dc-card *::-webkit-scrollbar{display:none}',
  // Per-artboard header: grip + label on the left, delete/expand on the
  // right. Single flex row; when the artboard's on-screen width is too
  // narrow for both the label yields (ellipsis, then hidden entirely below
  // ~4ch via the container query) and the buttons stay on the row.
  '.dc-header{position:absolute;bottom:100%;left:-4px;margin-bottom:calc(4px * var(--dc-inv-zoom,1));z-index:2;', '  display:flex;align-items:center;container-type:inline-size}', '.dc-labelrow{display:flex;align-items:center;gap:4px;height:24px;flex:1 1 auto;min-width:0}', '.dc-grip{flex:0 0 auto;cursor:grab;display:flex;align-items:center;padding:5px 4px;border-radius:4px;transition:background .12s,opacity .12s}', '.dc-grip:hover{background:rgba(0,0,0,.08)}', '.dc-grip:active{cursor:grabbing}', '.dc-labeltext{flex:1 1 auto;min-width:0;cursor:pointer;border-radius:4px;padding:3px 6px;', '  display:flex;align-items:center;transition:background .12s;overflow:hidden}',
  // Below ~4ch of label room: hide the label entirely, and drop the grip to
  // hover-only (same reveal rule as .dc-btns) so a narrow header is clean
  // until the card is moused.
  '@container (max-width: 110px){', '  .dc-labeltext{display:none}', '  .dc-grip{opacity:0}', '  [data-dc-slot]:hover .dc-grip{opacity:1}', '}', '.dc-labeltext:hover{background:rgba(0,0,0,.05)}', '.dc-labeltext .dc-editable{overflow:hidden;text-overflow:ellipsis;max-width:100%}', '.dc-labeltext .dc-editable:focus{overflow:visible;text-overflow:clip}', '.dc-btns{flex:0 0 auto;margin-left:auto;display:flex;gap:2px;opacity:0;transition:opacity .12s}', '[data-dc-slot]:hover .dc-btns,.dc-btns:has(.dc-menu){opacity:1}', '.dc-expand,.dc-kebab{width:22px;height:22px;border-radius:5px;border:none;cursor:pointer;padding:0;', '  background:transparent;color:rgba(60,50,40,.7);display:flex;align-items:center;justify-content:center;', '  font:inherit;transition:background .12s,color .12s}', '.dc-expand:hover,.dc-kebab:hover{background:rgba(0,0,0,.06);color:#2a251f}',
  // Slot hosting an open menu floats above later siblings (which otherwise
  // paint on top — same z-index:auto, later DOM order) so the popup isn't
  // clipped by the next card.
  '[data-dc-slot]:has(.dc-menu){z-index:10}', '.dc-menu{position:absolute;top:100%;right:0;margin-top:4px;background:#fff;border-radius:8px;', '  box-shadow:0 8px 28px rgba(0,0,0,.18),0 0 0 1px rgba(0,0,0,.05);padding:4px;min-width:160px;z-index:10}', '.dc-menu button{display:block;width:100%;padding:7px 10px;border:0;background:transparent;', '  border-radius:5px;font-family:inherit;font-size:13px;font-weight:500;line-height:1.2;', '  color:#29261b;cursor:pointer;text-align:left;transition:background .12s;white-space:nowrap}', '.dc-menu button:hover{background:rgba(0,0,0,.05)}', '.dc-menu hr{border:0;border-top:1px solid rgba(0,0,0,.08);margin:4px 2px}', '.dc-menu .dc-danger{color:#c96442}', '.dc-menu .dc-danger:hover{background:rgba(201,100,66,.1)}',
  // Chrome (titles / labels / buttons) counter-scales against the viewport
  // zoom so it stays a constant on-screen size. --dc-inv-zoom is set by
  // DCViewport on every transform update and inherits to all descendants —
  // any overlay inside the world (e.g. a TweaksPanel on an artboard) can use
  // it the same way.
  //
  // The header uses transform:scale (out-of-flow, so layout impact doesn't
  // matter) with its world-space width set to card-width / inv-zoom so that
  // after counter-scaling its on-screen width exactly matches the card's —
  // that's what lets the container query + text-overflow behave against the
  // card's visible edge at every zoom level.
  //
  // The section head uses CSS zoom instead of transform so its layout box
  // grows with the counter-scale, pushing the card row down — otherwise the
  // constant-screen-size title would overflow into the (shrinking) world-
  // space gap and overlap the artboard headers at low zoom.
  '.dc-header{width:calc((100% + 4px) / var(--dc-inv-zoom,1));', '  transform:scale(var(--dc-inv-zoom,1));transform-origin:bottom left}', '.dc-sectionhead{zoom:var(--dc-inv-zoom,1)}'].join('\n');
  document.head.appendChild(s);
}
const DCCtx = React.createContext(null);

// Recursively unwrap React.Fragment so <>…</> grouping doesn't hide
// DCSection/DCArtboard children from the type-based walks below.
function dcFlatten(children) {
  const out = [];
  React.Children.forEach(children, c => {
    if (c && c.type === React.Fragment) out.push(...dcFlatten(c.props.children));else out.push(c);
  });
  return out;
}

// ─────────────────────────────────────────────────────────────
// DesignCanvas — stateful wrapper around the pan/zoom viewport.
// Owns runtime state (per-section order, renamed titles/labels, hidden
// artboards, focused artboard). Order/titles/labels/hidden persist to a
// .design-canvas.state.json
// sidecar next to the HTML. Reads go via plain fetch() so the saved
// arrangement is visible anywhere the HTML + sidecar are served together
// (omelette preview, direct link, downloaded zip). Writes go through the
// host's window.omelette bridge — editing requires the omelette runtime.
// Focus is ephemeral.
// ─────────────────────────────────────────────────────────────
const DC_STATE_FILE = '.design-canvas.state.json';
function DesignCanvas({
  children,
  minScale,
  maxScale,
  style
}) {
  const [state, setState] = React.useState({
    sections: {},
    focus: null
  });
  // Hold rendering until the sidecar read settles so the saved order/titles
  // appear on first paint (no source-order flash). didRead gates writes until
  // the read settles so the empty initial state can't clobber a slow read;
  // skipNextWrite suppresses the one echo-write that would otherwise follow
  // hydration.
  const [ready, setReady] = React.useState(false);
  const didRead = React.useRef(false);
  const skipNextWrite = React.useRef(false);
  React.useEffect(() => {
    let off = false;
    fetch('./' + DC_STATE_FILE).then(r => r.ok ? r.json() : null).then(saved => {
      if (off || !saved || !saved.sections) return;
      skipNextWrite.current = true;
      setState(s => ({
        ...s,
        sections: saved.sections
      }));
    }).catch(() => {}).finally(() => {
      didRead.current = true;
      if (!off) setReady(true);
    });
    const t = setTimeout(() => {
      if (!off) setReady(true);
    }, 150);
    return () => {
      off = true;
      clearTimeout(t);
    };
  }, []);
  React.useEffect(() => {
    if (!didRead.current) return;
    if (skipNextWrite.current) {
      skipNextWrite.current = false;
      return;
    }
    const t = setTimeout(() => {
      window.omelette?.writeFile(DC_STATE_FILE, JSON.stringify({
        sections: state.sections
      })).catch(() => {});
    }, 250);
    return () => clearTimeout(t);
  }, [state.sections]);

  // Build registries synchronously from children so FocusOverlay can read
  // them in the same render. Fragments are flattened; wrapping in other
  // elements still opts out of focus/reorder.
  const registry = {}; // slotId -> { sectionId, artboard }
  const sectionMeta = {}; // sectionId -> { title, subtitle, slotIds[] }
  const sectionOrder = [];
  dcFlatten(children).forEach(sec => {
    if (!sec || sec.type !== DCSection) return;
    const sid = sec.props.id ?? sec.props.title;
    if (!sid) return;
    sectionOrder.push(sid);
    const persisted = state.sections[sid] || {};
    const abs = [];
    dcFlatten(sec.props.children).forEach(ab => {
      if (!ab || ab.type !== DCArtboard) return;
      const aid = ab.props.id ?? ab.props.label;
      if (aid) abs.push([aid, ab]);
    });
    // hidden is scoped to one source revision — when the agent regenerates
    // (artboard-ID set changes), prior deletes don't apply to new content.
    const srcKey = abs.map(([k]) => k).join('\x1f');
    const hidden = persisted.srcKey === srcKey ? persisted.hidden || [] : [];
    const srcIds = [];
    abs.forEach(([aid, ab]) => {
      if (hidden.includes(aid)) return;
      registry[`${sid}/${aid}`] = {
        sectionId: sid,
        artboard: ab
      };
      srcIds.push(aid);
    });
    const kept = (persisted.order || []).filter(k => srcIds.includes(k));
    sectionMeta[sid] = {
      title: persisted.title ?? sec.props.title,
      subtitle: sec.props.subtitle,
      slotIds: [...kept, ...srcIds.filter(k => !kept.includes(k))]
    };
  });
  const api = React.useMemo(() => ({
    state,
    section: id => state.sections[id] || {},
    patchSection: (id, p) => setState(s => ({
      ...s,
      sections: {
        ...s.sections,
        [id]: {
          ...s.sections[id],
          ...(typeof p === 'function' ? p(s.sections[id] || {}) : p)
        }
      }
    })),
    setFocus: slotId => setState(s => ({
      ...s,
      focus: slotId
    }))
  }), [state]);

  // Esc exits focus; any outside pointerdown commits an in-progress rename.
  React.useEffect(() => {
    const onKey = e => {
      if (e.key === 'Escape') api.setFocus(null);
    };
    const onPd = e => {
      const ae = document.activeElement;
      if (ae && ae.isContentEditable && !ae.contains(e.target)) ae.blur();
    };
    document.addEventListener('keydown', onKey);
    document.addEventListener('pointerdown', onPd, true);
    return () => {
      document.removeEventListener('keydown', onKey);
      document.removeEventListener('pointerdown', onPd, true);
    };
  }, [api]);
  return /*#__PURE__*/React.createElement(DCCtx.Provider, {
    value: api
  }, /*#__PURE__*/React.createElement(DCViewport, {
    minScale: minScale,
    maxScale: maxScale,
    style: style
  }, ready && children), state.focus && registry[state.focus] && /*#__PURE__*/React.createElement(DCFocusOverlay, {
    entry: registry[state.focus],
    sectionMeta: sectionMeta,
    sectionOrder: sectionOrder
  }));
}

// ─────────────────────────────────────────────────────────────
// DCViewport — transform-based pan/zoom (internal)
//
// Input mapping (Figma-style):
//   • trackpad pinch  → zoom   (ctrlKey wheel; Safari gesture* events)
//   • trackpad scroll → pan    (two-finger)
//   • mouse wheel     → zoom   (notched; distinguished from trackpad scroll)
//   • middle-drag / primary-drag-on-bg → pan
//
// Transform state lives in a ref and is written straight to the DOM
// (translate3d + will-change) so wheel ticks don't go through React —
// keeps pans at 60fps on dense canvases.
// ─────────────────────────────────────────────────────────────
function DCViewport({
  children,
  minScale = 0.1,
  maxScale = 8,
  style = {}
}) {
  const vpRef = React.useRef(null);
  const worldRef = React.useRef(null);
  const tf = React.useRef({
    x: 0,
    y: 0,
    scale: 1
  });
  // Persist viewport across reloads so the user lands back where they were
  // after an agent edit or browser refresh. The sandbox origin is already
  // per-project; pathname keeps multiple canvas files in one project apart.
  const tfKey = 'dc-viewport:' + location.pathname;
  const saveT = React.useRef(0);
  const lastPostedScale = React.useRef();
  const apply = React.useCallback(() => {
    const {
      x,
      y,
      scale
    } = tf.current;
    const el = worldRef.current;
    if (!el) return;
    el.style.transform = `translate3d(${x}px, ${y}px, 0) scale(${scale})`;
    // Exposed for zoom-invariant chrome (labels, buttons, TweaksPanel).
    el.style.setProperty('--dc-inv-zoom', String(1 / scale));
    // Keep the host toolbar's % readout in sync with the canvas scale. Pan
    // ticks leave scale unchanged — skip the cross-frame post for those.
    if (lastPostedScale.current !== scale) {
      lastPostedScale.current = scale;
      window.parent.postMessage({
        type: '__dc_zoom',
        scale
      }, '*');
    }
    clearTimeout(saveT.current);
    saveT.current = setTimeout(() => {
      try {
        localStorage.setItem(tfKey, JSON.stringify(tf.current));
      } catch {}
    }, 200);
  }, [tfKey]);
  React.useLayoutEffect(() => {
    const flush = () => {
      clearTimeout(saveT.current);
      try {
        localStorage.setItem(tfKey, JSON.stringify(tf.current));
      } catch {}
    };
    try {
      const s = JSON.parse(localStorage.getItem(tfKey) || 'null');
      if (s && Number.isFinite(s.x) && Number.isFinite(s.y) && Number.isFinite(s.scale)) {
        tf.current = {
          x: s.x,
          y: s.y,
          scale: Math.min(maxScale, Math.max(minScale, s.scale))
        };
        apply();
      }
    } catch {}
    // Flush on pagehide and unmount so a reload within the 200ms debounce
    // window doesn't drop the last pan/zoom.
    window.addEventListener('pagehide', flush);
    return () => {
      window.removeEventListener('pagehide', flush);
      flush();
    };
  }, []);
  React.useEffect(() => {
    const vp = vpRef.current;
    if (!vp) return;
    const zoomAt = (cx, cy, factor) => {
      const r = vp.getBoundingClientRect();
      const px = cx - r.left,
        py = cy - r.top;
      const t = tf.current;
      const next = Math.min(maxScale, Math.max(minScale, t.scale * factor));
      const k = next / t.scale;
      // --dc-inv-zoom consumers (.dc-sectionhead's CSS zoom, each section's
      // marginBottom) reflow on every scale change, vertically shifting the
      // world layout — so a world point mathematically pinned under the cursor
      // drifts as you zoom (content creeps up on zoom-in, down on zoom-out).
      // Anchor the DOM element under the cursor instead: record its screen Y,
      // apply the transform + --dc-inv-zoom, then cancel whatever vertical
      // drift the reflow introduced so it stays put on screen.
      let marker = null,
        markerY0 = 0;
      if (k !== 1) {
        const hit = document.elementFromPoint(cx, cy);
        marker = hit && hit.closest ? hit.closest('[data-dc-slot],[data-dc-section]') : null;
        if (marker) markerY0 = marker.getBoundingClientRect().top;
      }
      // keep the world point under the cursor fixed
      t.x = px - (px - t.x) * k;
      t.y = py - (py - t.y) * k;
      t.scale = next;
      apply();
      if (marker) {
        // A pure zoom around (cx, cy) maps screen Y → cy + (Y - cy) * k. Any
        // departure after the --dc-inv-zoom reflow is the layout drift.
        const drift = marker.getBoundingClientRect().top - (cy + (markerY0 - cy) * k);
        if (Math.abs(drift) > 0.1) {
          t.y -= drift;
          apply();
        }
      }
    };

    // Mouse-wheel vs trackpad-scroll heuristic. A physical wheel sends
    // line-mode deltas (Firefox) or large integer pixel deltas with no X
    // component (Chrome/Safari, typically multiples of 100/120). Trackpad
    // two-finger scroll sends small/fractional pixel deltas, often with
    // non-zero deltaX. ctrlKey is set by the browser for trackpad pinch.
    const isMouseWheel = e => e.deltaMode !== 0 || e.deltaX === 0 && Number.isInteger(e.deltaY) && Math.abs(e.deltaY) >= 40;
    const onWheel = e => {
      e.preventDefault();
      if (isGesturing) return; // Safari: gesture* owns the pinch — discard concurrent wheels
      if ((e.ctrlKey || e.metaKey) && !isMouseWheel(e)) {
        // trackpad pinch, or ctrl/cmd + smooth-scroll mouse. Notched
        // wheels fall through to the fixed-step branch below.
        zoomAt(e.clientX, e.clientY, Math.exp(-e.deltaY * 0.01));
      } else if (isMouseWheel(e)) {
        // notched mouse wheel — fixed-ratio step per click
        zoomAt(e.clientX, e.clientY, Math.exp(-Math.sign(e.deltaY) * 0.18));
      } else {
        // trackpad two-finger scroll — pan
        tf.current.x -= e.deltaX;
        tf.current.y -= e.deltaY;
        apply();
      }
    };

    // Safari sends native gesture* events for trackpad pinch with a smooth
    // e.scale; preferring these over the ctrl+wheel fallback gives a much
    // better feel there. No-ops on other browsers. Safari also fires
    // ctrlKey wheel events during the same pinch — isGesturing makes
    // onWheel drop those entirely so they neither zoom nor pan.
    let gsBase = 1;
    let isGesturing = false;
    const onGestureStart = e => {
      e.preventDefault();
      isGesturing = true;
      gsBase = tf.current.scale;
    };
    const onGestureChange = e => {
      e.preventDefault();
      zoomAt(e.clientX, e.clientY, gsBase * e.scale / tf.current.scale);
    };
    const onGestureEnd = e => {
      e.preventDefault();
      isGesturing = false;
    };

    // Drag-pan: middle button anywhere, or primary button on canvas
    // background (anything that isn't an artboard or an inline editor).
    let drag = null;
    const onPointerDown = e => {
      const onBg = !e.target.closest('[data-dc-slot], .dc-editable');
      if (!(e.button === 1 || e.button === 0 && onBg)) return;
      e.preventDefault();
      vp.setPointerCapture(e.pointerId);
      drag = {
        id: e.pointerId,
        lx: e.clientX,
        ly: e.clientY
      };
      vp.style.cursor = 'grabbing';
    };
    const onPointerMove = e => {
      if (!drag || e.pointerId !== drag.id) return;
      tf.current.x += e.clientX - drag.lx;
      tf.current.y += e.clientY - drag.ly;
      drag.lx = e.clientX;
      drag.ly = e.clientY;
      apply();
    };
    const onPointerUp = e => {
      if (!drag || e.pointerId !== drag.id) return;
      vp.releasePointerCapture(e.pointerId);
      drag = null;
      vp.style.cursor = '';
    };

    // Host-driven zoom (toolbar % menu). Zooms around viewport centre so the
    // visible midpoint stays fixed — matching the host's iframe-zoom feel.
    const onHostMsg = e => {
      const d = e.data;
      if (d && d.type === '__dc_set_zoom' && typeof d.scale === 'number') {
        const r = vp.getBoundingClientRect();
        zoomAt(r.left + r.width / 2, r.top + r.height / 2, d.scale / tf.current.scale);
      } else if (d && d.type === '__dc_probe') {
        // Host's [readyGen] reset asks whether a canvas is present; it
        // fires on the iframe's native 'load', which for canvases with
        // images/fonts is after our mount-time announce, so re-announce.
        // Clear the pan-tick guard so apply() re-posts the current scale
        // even if it's unchanged — the host just reset dcScale to 1.
        window.parent.postMessage({
          type: '__dc_present'
        }, '*');
        lastPostedScale.current = undefined;
        apply();
      }
    };
    window.addEventListener('message', onHostMsg);
    // Announce canvas mode so the host toolbar proxies its % control here
    // instead of scaling the iframe element (which would just shrink the
    // viewport window of an infinite canvas). The apply() that follows emits
    // the initial __dc_zoom so the toolbar % is correct before first pinch.
    // lastPostedScale reset mirrors the __dc_probe handler: the layout
    // effect's restore-path apply() may already have posted the restored
    // scale (before __dc_present), so clear the guard to re-post it in order.
    window.parent.postMessage({
      type: '__dc_present'
    }, '*');
    lastPostedScale.current = undefined;
    apply();
    vp.addEventListener('wheel', onWheel, {
      passive: false
    });
    vp.addEventListener('gesturestart', onGestureStart, {
      passive: false
    });
    vp.addEventListener('gesturechange', onGestureChange, {
      passive: false
    });
    vp.addEventListener('gestureend', onGestureEnd, {
      passive: false
    });
    vp.addEventListener('pointerdown', onPointerDown);
    vp.addEventListener('pointermove', onPointerMove);
    vp.addEventListener('pointerup', onPointerUp);
    vp.addEventListener('pointercancel', onPointerUp);
    return () => {
      window.removeEventListener('message', onHostMsg);
      vp.removeEventListener('wheel', onWheel);
      vp.removeEventListener('gesturestart', onGestureStart);
      vp.removeEventListener('gesturechange', onGestureChange);
      vp.removeEventListener('gestureend', onGestureEnd);
      vp.removeEventListener('pointerdown', onPointerDown);
      vp.removeEventListener('pointermove', onPointerMove);
      vp.removeEventListener('pointerup', onPointerUp);
      vp.removeEventListener('pointercancel', onPointerUp);
    };
  }, [apply, minScale, maxScale]);
  const gridSvg = `url("data:image/svg+xml,%3Csvg width='120' height='120' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M120 0H0v120' fill='none' stroke='${encodeURIComponent(DC.grid)}' stroke-width='1'/%3E%3C/svg%3E")`;
  return /*#__PURE__*/React.createElement("div", {
    ref: vpRef,
    className: "design-canvas",
    style: {
      height: '100vh',
      width: '100vw',
      background: DC.bg,
      overflow: 'hidden',
      overscrollBehavior: 'none',
      touchAction: 'none',
      position: 'relative',
      fontFamily: DC.font,
      boxSizing: 'border-box',
      ...style
    }
  }, /*#__PURE__*/React.createElement("div", {
    ref: worldRef,
    style: {
      position: 'absolute',
      top: 0,
      left: 0,
      transformOrigin: '0 0',
      willChange: 'transform',
      width: 'max-content',
      minWidth: '100%',
      minHeight: '100%',
      padding: '60px 0 80px'
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      inset: -6000,
      backgroundImage: gridSvg,
      backgroundSize: '120px 120px',
      pointerEvents: 'none',
      zIndex: -1
    }
  }), children));
}

// ─────────────────────────────────────────────────────────────
// DCSection — editable title + h-row of artboards in persisted order
// ─────────────────────────────────────────────────────────────
function DCSection({
  id,
  title,
  subtitle,
  children,
  gap = 48
}) {
  const ctx = React.useContext(DCCtx);
  const sid = id ?? title;
  const all = React.Children.toArray(dcFlatten(children));
  const artboards = all.filter(c => c && c.type === DCArtboard);
  const rest = all.filter(c => !(c && c.type === DCArtboard));
  const sec = ctx && sid && ctx.section(sid) || {};
  // Must match DesignCanvas's srcKey computation exactly (it filters falsy
  // IDs), or onDelete persists a srcKey that DesignCanvas never recognizes.
  const allIds = artboards.map(a => a.props.id ?? a.props.label).filter(Boolean);
  const srcKey = allIds.join('\x1f');
  const hidden = sec.srcKey === srcKey ? sec.hidden || [] : [];
  const srcOrder = allIds.filter(k => !hidden.includes(k));
  const order = React.useMemo(() => {
    const kept = (sec.order || []).filter(k => srcOrder.includes(k));
    return [...kept, ...srcOrder.filter(k => !kept.includes(k))];
  }, [sec.order, srcOrder.join('|')]);
  const byId = Object.fromEntries(artboards.map(a => [a.props.id ?? a.props.label, a]));

  // marginBottom counter-scales so the on-screen gap between sections stays
  // constant — otherwise at low zoom the (world-space) gap collapses while
  // the screen-constant sectionhead below it doesn't, and the title reads as
  // belonging to the section above. paddingBottom below is just enough for
  // the 24px artboard-header (abs-positioned above each card) plus ~8px, so
  // the title sits tight against its own row at every zoom.
  return /*#__PURE__*/React.createElement("div", {
    "data-dc-section": sid,
    style: {
      marginBottom: 'calc(80px * var(--dc-inv-zoom, 1))',
      position: 'relative'
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      padding: '0 60px'
    }
  }, /*#__PURE__*/React.createElement("div", {
    className: "dc-sectionhead",
    style: {
      paddingBottom: 36
    }
  }, /*#__PURE__*/React.createElement(DCEditable, {
    tag: "div",
    value: sec.title ?? title,
    onChange: v => ctx && sid && ctx.patchSection(sid, {
      title: v
    }),
    style: {
      fontSize: 28,
      fontWeight: 600,
      color: DC.title,
      letterSpacing: -0.4,
      marginBottom: 6,
      display: 'inline-block'
    }
  }), subtitle && /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 16,
      color: DC.subtitle
    }
  }, subtitle))), /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      gap,
      padding: '0 60px',
      alignItems: 'flex-start',
      width: 'max-content'
    }
  }, order.map(k => /*#__PURE__*/React.createElement(DCArtboardFrame, {
    key: k,
    sectionId: sid,
    artboard: byId[k],
    order: order,
    label: (sec.labels || {})[k] ?? byId[k].props.label,
    onRename: v => ctx && ctx.patchSection(sid, x => ({
      labels: {
        ...x.labels,
        [k]: v
      }
    })),
    onReorder: next => ctx && ctx.patchSection(sid, {
      order: next
    }),
    onDelete: () => ctx && ctx.patchSection(sid, x => ({
      hidden: [...(x.srcKey === srcKey ? x.hidden || [] : []), k],
      srcKey
    })),
    onFocus: () => ctx && ctx.setFocus(`${sid}/${k}`)
  }))), rest);
}

// DCArtboard — marker; rendered by DCArtboardFrame via DCSection.
function DCArtboard() {
  return null;
}

// Per-artboard export (kind: 'png' | 'html'). Both paths share the same
// self-contained clone: computed styles baked in, @font-face / <img> /
// inline-style background-image urls inlined as data URIs. PNG wraps the
// clone in foreignObject→canvas at 3× the artboard's natural width×height
// (same pipeline the host uses for page captures); HTML wraps it in a
// minimal standalone document. Both are independent of viewport zoom.
async function dcExport(node, w, h, name, kind) {
  try {
    await document.fonts.ready;
  } catch {}
  const toDataURL = url => fetch(url).then(r => r.blob()).then(b => new Promise(res => {
    const fr = new FileReader();
    fr.onload = () => res(fr.result);
    fr.onerror = () => res(url);
    fr.readAsDataURL(b);
  })).catch(() => url);

  // Collect @font-face rules. ss.cssRules throws SecurityError on
  // cross-origin sheets (e.g. fonts.googleapis.com) — in that case fetch
  // the CSS text directly (those endpoints send ACAO:*) and regex-extract
  // the blocks. @import and @media/@supports are walked so nested
  // @font-face rules aren't missed.
  const fontRules = [],
    pending = [],
    seen = new Set();
  const scrapeCss = href => {
    if (seen.has(href)) return;
    seen.add(href);
    pending.push(fetch(href).then(r => r.text()).then(css => {
      for (const m of css.match(/@font-face\s*{[^}]*}/g) || []) fontRules.push({
        css: m,
        base: href
      });
      for (const m of css.matchAll(/@import\s+(?:url\()?['"]?([^'")\s;]+)/g)) scrapeCss(new URL(m[1], href).href);
    }).catch(() => {}));
  };
  const walk = (rules, base) => {
    for (const r of rules) {
      if (r.type === CSSRule.FONT_FACE_RULE) fontRules.push({
        css: r.cssText,
        base
      });else if (r.type === CSSRule.IMPORT_RULE && r.styleSheet) {
        const ibase = r.styleSheet.href || base;
        try {
          walk(r.styleSheet.cssRules, ibase);
        } catch {
          scrapeCss(ibase);
        }
      } else if (r.cssRules) walk(r.cssRules, base);
    }
  };
  for (const ss of document.styleSheets) {
    const base = ss.href || location.href;
    try {
      walk(ss.cssRules, base);
    } catch {
      if (ss.href) scrapeCss(ss.href);
    }
  }
  while (pending.length) await pending.shift();
  const fontCss = (await Promise.all(fontRules.map(async rule => {
    let out = rule.css,
      m;
    const re = /url\((['"]?)([^'")]+)\1\)/g;
    while (m = re.exec(rule.css)) {
      if (m[2].indexOf('data:') === 0) continue;
      let abs;
      try {
        abs = new URL(m[2], rule.base).href;
      } catch {
        continue;
      }
      out = out.split(m[0]).join('url("' + (await toDataURL(abs)) + '")');
    }
    return out;
  }))).join('\n');
  const cloneStyled = src => {
    if (src.nodeType === 8 || src.nodeType === 1 && src.tagName === 'SCRIPT') return document.createTextNode('');
    const dst = src.cloneNode(false);
    if (src.nodeType === 1) {
      const cs = getComputedStyle(src);
      let txt = '';
      for (let i = 0; i < cs.length; i++) txt += cs[i] + ':' + cs.getPropertyValue(cs[i]) + ';';
      dst.setAttribute('style', txt + 'animation:none;transition:none;');
      if (src.tagName === 'CANVAS') try {
        const im = document.createElement('img');
        im.src = src.toDataURL();
        im.setAttribute('style', txt);
        return im;
      } catch {}
    }
    for (let c = src.firstChild; c; c = c.nextSibling) dst.appendChild(cloneStyled(c));
    return dst;
  };
  const clone = cloneStyled(node);
  clone.setAttribute('xmlns', 'http://www.w3.org/1999/xhtml');
  // Drop the card's own shadow/radius so the export is a flush w×h rect;
  // the artboard's own background (if any) is already in the computed style.
  clone.style.boxShadow = 'none';
  clone.style.borderRadius = '0';
  const jobs = [];
  clone.querySelectorAll('img').forEach(el => {
    const s = el.getAttribute('src');
    if (s && s.indexOf('data:') !== 0) jobs.push(toDataURL(el.src).then(d => el.setAttribute('src', d)));
  });
  [clone, ...clone.querySelectorAll('*')].forEach(el => {
    const bg = el.style.backgroundImage;
    if (!bg) return;
    let m;
    const re = /url\(["']?([^"')]+)["']?\)/g;
    while (m = re.exec(bg)) {
      const tok = m[0],
        url = m[1];
      if (url.indexOf('data:') === 0) continue;
      jobs.push(toDataURL(url).then(d => {
        el.style.backgroundImage = el.style.backgroundImage.split(tok).join('url("' + d + '")');
      }));
    }
  });
  await Promise.all(jobs);
  const xml = new XMLSerializer().serializeToString(clone);
  const save = (blob, ext) => {
    if (!blob) return;
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = name + '.' + ext;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  };
  if (kind === 'html') {
    const html = '<!doctype html><html><head><meta charset="utf-8"><title>' + name + '</title>' + (fontCss ? '<style>' + fontCss + '</style>' : '') + '</head><body style="margin:0">' + xml + '</body></html>';
    return save(new Blob([html], {
      type: 'text/html'
    }), 'html');
  }

  // PNG: the SVG's own width/height must be the output resolution — an
  // <img>-loaded SVG rasterizes at its intrinsic size, so sizing it at 1×
  // and ctx.scale()-ing up would just upscale a 1× bitmap. viewBox maps the
  // w×h foreignObject onto the px·w × px·h SVG canvas so the browser renders
  // the HTML at full resolution.
  const px = 3;
  const svg = '<svg xmlns="http://www.w3.org/2000/svg" width="' + w * px + '" height="' + h * px + '" viewBox="0 0 ' + w + ' ' + h + '"><foreignObject width="' + w + '" height="' + h + '">' + (fontCss ? '<style><![CDATA[' + fontCss + ']]></style>' : '') + xml + '</foreignObject></svg>';
  const img = new Image();
  await new Promise((res, rej) => {
    img.onload = res;
    img.onerror = () => rej(new Error('svg load failed'));
    img.src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svg);
  });
  const cv = document.createElement('canvas');
  cv.width = w * px;
  cv.height = h * px;
  cv.getContext('2d').drawImage(img, 0, 0);
  cv.toBlob(blob => save(blob, 'png'), 'image/png');
}
function DCArtboardFrame({
  sectionId,
  artboard,
  label,
  order,
  onRename,
  onReorder,
  onFocus,
  onDelete
}) {
  const {
    id: rawId,
    label: rawLabel,
    width = 260,
    height = 480,
    children,
    style = {}
  } = artboard.props;
  const id = rawId ?? rawLabel;
  const ref = React.useRef(null);
  const cardRef = React.useRef(null);
  const menuRef = React.useRef(null);
  const [menuOpen, setMenuOpen] = React.useState(false);
  const [confirming, setConfirming] = React.useState(false);

  // ⋯ menu: close on any outside pointerdown. Two-click delete lives inside
  // the menu — first click arms the row, second commits; closing disarms.
  React.useEffect(() => {
    if (!menuOpen) {
      setConfirming(false);
      return;
    }
    const off = e => {
      if (!menuRef.current || !menuRef.current.contains(e.target)) setMenuOpen(false);
    };
    document.addEventListener('pointerdown', off, true);
    return () => document.removeEventListener('pointerdown', off, true);
  }, [menuOpen]);
  const doExport = kind => {
    setMenuOpen(false);
    if (!cardRef.current) return;
    const name = String(label || id || 'artboard').replace(/[^\w\s.-]+/g, '_');
    dcExport(cardRef.current, width, height, name, kind).catch(e => console.error('[design-canvas] export failed:', e));
  };

  // Live drag-reorder: dragged card sticks to cursor; siblings slide into
  // their would-be slots in real time via transforms. DOM order only
  // changes on drop.
  const onGripDown = e => {
    e.preventDefault();
    e.stopPropagation();
    const me = ref.current;
    // translateX is applied in local (pre-scale) space but pointer deltas and
    // getBoundingClientRect().left are screen-space — divide by the viewport's
    // current scale so the dragged card tracks the cursor at any zoom level.
    const scale = me.getBoundingClientRect().width / me.offsetWidth || 1;
    const peers = Array.from(document.querySelectorAll(`[data-dc-section="${sectionId}"] [data-dc-slot]`));
    const homes = peers.map(el => ({
      el,
      id: el.dataset.dcSlot,
      x: el.getBoundingClientRect().left
    }));
    const slotXs = homes.map(h => h.x);
    const startIdx = order.indexOf(id);
    const startX = e.clientX;
    let liveOrder = order.slice();
    me.classList.add('dc-dragging');
    const layout = () => {
      for (const h of homes) {
        if (h.id === id) continue;
        const slot = liveOrder.indexOf(h.id);
        h.el.style.transform = `translateX(${(slotXs[slot] - h.x) / scale}px)`;
      }
    };
    const move = ev => {
      const dx = ev.clientX - startX;
      me.style.transform = `translateX(${dx / scale}px)`;
      const cur = homes[startIdx].x + dx;
      let nearest = 0,
        best = Infinity;
      for (let i = 0; i < slotXs.length; i++) {
        const d = Math.abs(slotXs[i] - cur);
        if (d < best) {
          best = d;
          nearest = i;
        }
      }
      if (liveOrder.indexOf(id) !== nearest) {
        liveOrder = order.filter(k => k !== id);
        liveOrder.splice(nearest, 0, id);
        layout();
      }
    };
    const up = () => {
      document.removeEventListener('pointermove', move);
      document.removeEventListener('pointerup', up);
      const finalSlot = liveOrder.indexOf(id);
      me.classList.remove('dc-dragging');
      me.style.transform = `translateX(${(slotXs[finalSlot] - homes[startIdx].x) / scale}px)`;
      // After the settle transition, kill transitions + clear transforms +
      // commit the reorder in the same frame so there's no visual snap-back.
      setTimeout(() => {
        for (const h of homes) {
          h.el.style.transition = 'none';
          h.el.style.transform = '';
        }
        if (liveOrder.join('|') !== order.join('|')) onReorder(liveOrder);
        requestAnimationFrame(() => requestAnimationFrame(() => {
          for (const h of homes) h.el.style.transition = '';
        }));
      }, 180);
    };
    document.addEventListener('pointermove', move);
    document.addEventListener('pointerup', up);
  };
  return /*#__PURE__*/React.createElement("div", {
    ref: ref,
    "data-dc-slot": id,
    style: {
      position: 'relative',
      flexShrink: 0
    }
  }, /*#__PURE__*/React.createElement("div", {
    className: "dc-header",
    "data-omelette-chrome": "",
    style: {
      color: DC.label
    },
    onPointerDown: e => e.stopPropagation()
  }, /*#__PURE__*/React.createElement("div", {
    className: "dc-labelrow"
  }, /*#__PURE__*/React.createElement("div", {
    className: "dc-grip",
    onPointerDown: onGripDown,
    title: "Drag to reorder"
  }, /*#__PURE__*/React.createElement("svg", {
    width: "9",
    height: "13",
    viewBox: "0 0 9 13",
    fill: "currentColor"
  }, /*#__PURE__*/React.createElement("circle", {
    cx: "2",
    cy: "2",
    r: "1.1"
  }), /*#__PURE__*/React.createElement("circle", {
    cx: "7",
    cy: "2",
    r: "1.1"
  }), /*#__PURE__*/React.createElement("circle", {
    cx: "2",
    cy: "6.5",
    r: "1.1"
  }), /*#__PURE__*/React.createElement("circle", {
    cx: "7",
    cy: "6.5",
    r: "1.1"
  }), /*#__PURE__*/React.createElement("circle", {
    cx: "2",
    cy: "11",
    r: "1.1"
  }), /*#__PURE__*/React.createElement("circle", {
    cx: "7",
    cy: "11",
    r: "1.1"
  }))), /*#__PURE__*/React.createElement("div", {
    className: "dc-labeltext",
    onClick: onFocus,
    title: "Click to focus"
  }, /*#__PURE__*/React.createElement(DCEditable, {
    value: label,
    onChange: onRename,
    onClick: e => e.stopPropagation(),
    style: {
      fontSize: 15,
      fontWeight: 500,
      color: DC.label,
      lineHeight: 1
    }
  }))), /*#__PURE__*/React.createElement("div", {
    className: "dc-btns"
  }, /*#__PURE__*/React.createElement("div", {
    ref: menuRef,
    style: {
      position: 'relative'
    }
  }, /*#__PURE__*/React.createElement("button", {
    className: "dc-kebab",
    title: "More",
    onClick: () => setMenuOpen(o => !o)
  }, /*#__PURE__*/React.createElement("svg", {
    width: "12",
    height: "12",
    viewBox: "0 0 12 12",
    fill: "currentColor"
  }, /*#__PURE__*/React.createElement("circle", {
    cx: "2.5",
    cy: "6",
    r: "1.1"
  }), /*#__PURE__*/React.createElement("circle", {
    cx: "6",
    cy: "6",
    r: "1.1"
  }), /*#__PURE__*/React.createElement("circle", {
    cx: "9.5",
    cy: "6",
    r: "1.1"
  }))), menuOpen && /*#__PURE__*/React.createElement("div", {
    className: "dc-menu",
    onPointerDown: e => e.stopPropagation()
  }, /*#__PURE__*/React.createElement("button", {
    onClick: () => doExport('png')
  }, "Download PNG"), /*#__PURE__*/React.createElement("button", {
    onClick: () => doExport('html')
  }, "Download HTML"), /*#__PURE__*/React.createElement("hr", null), /*#__PURE__*/React.createElement("button", {
    className: "dc-danger",
    onClick: () => {
      if (confirming) {
        setMenuOpen(false);
        onDelete();
      } else setConfirming(true);
    }
  }, confirming ? 'Click again to delete' : 'Delete'))), /*#__PURE__*/React.createElement("button", {
    className: "dc-expand",
    onClick: onFocus,
    title: "Focus"
  }, /*#__PURE__*/React.createElement("svg", {
    width: "12",
    height: "12",
    viewBox: "0 0 12 12",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: "1.6",
    strokeLinecap: "round"
  }, /*#__PURE__*/React.createElement("path", {
    d: "M7 1h4v4M5 11H1V7M11 1L7.5 4.5M1 11l3.5-3.5"
  }))))), /*#__PURE__*/React.createElement("div", {
    ref: cardRef,
    className: "dc-card",
    style: {
      borderRadius: 2,
      boxShadow: '0 1px 3px rgba(0,0,0,.08),0 4px 16px rgba(0,0,0,.06)',
      overflow: 'hidden',
      width,
      height,
      background: '#fff',
      ...style
    }
  }, children || /*#__PURE__*/React.createElement("div", {
    style: {
      height: '100%',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      color: '#bbb',
      fontSize: 13,
      fontFamily: DC.font
    }
  }, id)));
}

// Inline rename — commits on blur or Enter.
function DCEditable({
  value,
  onChange,
  style,
  tag = 'span',
  onClick
}) {
  const T = tag;
  return /*#__PURE__*/React.createElement(T, {
    className: "dc-editable",
    contentEditable: true,
    suppressContentEditableWarning: true,
    onClick: onClick,
    onPointerDown: e => e.stopPropagation(),
    onBlur: e => onChange && onChange(e.currentTarget.textContent),
    onKeyDown: e => {
      if (e.key === 'Enter') {
        e.preventDefault();
        e.currentTarget.blur();
      }
    },
    style: style
  }, value);
}

// ─────────────────────────────────────────────────────────────
// Focus mode — overlay one artboard; ←/→ within section, ↑/↓ across
// sections, Esc or backdrop click to exit.
// ─────────────────────────────────────────────────────────────
function DCFocusOverlay({
  entry,
  sectionMeta,
  sectionOrder
}) {
  const ctx = React.useContext(DCCtx);
  const {
    sectionId,
    artboard
  } = entry;
  const sec = ctx.section(sectionId);
  const meta = sectionMeta[sectionId];
  const peers = meta.slotIds;
  const aid = artboard.props.id ?? artboard.props.label;
  const idx = peers.indexOf(aid);
  const secIdx = sectionOrder.indexOf(sectionId);
  const go = d => {
    const n = peers[(idx + d + peers.length) % peers.length];
    if (n) ctx.setFocus(`${sectionId}/${n}`);
  };
  const goSection = d => {
    // Sections whose artboards are all deleted have slotIds:[] — step past
    // them to the next non-empty section so ↑/↓ doesn't dead-end.
    const n = sectionOrder.length;
    for (let i = 1; i < n; i++) {
      const ns = sectionOrder[((secIdx + d * i) % n + n) % n];
      const first = sectionMeta[ns] && sectionMeta[ns].slotIds[0];
      if (first) {
        ctx.setFocus(`${ns}/${first}`);
        return;
      }
    }
  };
  React.useEffect(() => {
    const k = e => {
      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        go(-1);
      }
      if (e.key === 'ArrowRight') {
        e.preventDefault();
        go(1);
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        goSection(-1);
      }
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        goSection(1);
      }
    };
    document.addEventListener('keydown', k);
    return () => document.removeEventListener('keydown', k);
  });
  const {
    width = 260,
    height = 480,
    children
  } = artboard.props;
  const [vp, setVp] = React.useState({
    w: window.innerWidth,
    h: window.innerHeight
  });
  React.useEffect(() => {
    const r = () => setVp({
      w: window.innerWidth,
      h: window.innerHeight
    });
    window.addEventListener('resize', r);
    return () => window.removeEventListener('resize', r);
  }, []);
  const scale = Math.max(0.1, Math.min((vp.w - 200) / width, (vp.h - 260) / height, 2));
  const [ddOpen, setDd] = React.useState(false);
  const Arrow = ({
    dir,
    onClick
  }) => /*#__PURE__*/React.createElement("button", {
    onClick: e => {
      e.stopPropagation();
      onClick();
    },
    style: {
      position: 'absolute',
      top: '50%',
      [dir]: 28,
      transform: 'translateY(-50%)',
      border: 'none',
      background: 'rgba(255,255,255,.08)',
      color: 'rgba(255,255,255,.9)',
      width: 44,
      height: 44,
      borderRadius: 22,
      fontSize: 18,
      cursor: 'pointer',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      transition: 'background .15s'
    },
    onMouseEnter: e => e.currentTarget.style.background = 'rgba(255,255,255,.18)',
    onMouseLeave: e => e.currentTarget.style.background = 'rgba(255,255,255,.08)'
  }, /*#__PURE__*/React.createElement("svg", {
    width: "18",
    height: "18",
    viewBox: "0 0 18 18",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: "2",
    strokeLinecap: "round"
  }, /*#__PURE__*/React.createElement("path", {
    d: dir === 'left' ? 'M11 3L5 9l6 6' : 'M7 3l6 6-6 6'
  })));

  // Portal to body so position:fixed is the real viewport regardless of any
  // transform on DesignCanvas's ancestors (including the canvas zoom itself).
  return ReactDOM.createPortal(/*#__PURE__*/React.createElement("div", {
    onClick: () => ctx.setFocus(null),
    onWheel: e => e.preventDefault(),
    style: {
      position: 'fixed',
      inset: 0,
      zIndex: 100,
      background: 'rgba(24,20,16,.6)',
      backdropFilter: 'blur(14px)',
      fontFamily: DC.font,
      color: '#fff'
    }
  }, /*#__PURE__*/React.createElement("div", {
    onClick: e => e.stopPropagation(),
    style: {
      position: 'absolute',
      top: 0,
      left: 0,
      right: 0,
      height: 72,
      display: 'flex',
      alignItems: 'flex-start',
      padding: '16px 20px 0',
      gap: 16
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'relative'
    }
  }, /*#__PURE__*/React.createElement("button", {
    onClick: () => setDd(o => !o),
    style: {
      border: 'none',
      background: 'transparent',
      color: '#fff',
      cursor: 'pointer',
      padding: '6px 8px',
      borderRadius: 6,
      textAlign: 'left',
      fontFamily: 'inherit'
    }
  }, /*#__PURE__*/React.createElement("span", {
    style: {
      display: 'flex',
      alignItems: 'center',
      gap: 8
    }
  }, /*#__PURE__*/React.createElement("span", {
    style: {
      fontSize: 18,
      fontWeight: 600,
      letterSpacing: -0.3
    }
  }, meta.title), /*#__PURE__*/React.createElement("svg", {
    width: "11",
    height: "11",
    viewBox: "0 0 11 11",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: "1.8",
    strokeLinecap: "round",
    style: {
      opacity: .7
    }
  }, /*#__PURE__*/React.createElement("path", {
    d: "M2 4l3.5 3.5L9 4"
  }))), meta.subtitle && /*#__PURE__*/React.createElement("span", {
    style: {
      display: 'block',
      fontSize: 13,
      opacity: .6,
      fontWeight: 400,
      marginTop: 2
    }
  }, meta.subtitle)), ddOpen && /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      top: '100%',
      left: 0,
      marginTop: 4,
      background: '#2a251f',
      borderRadius: 8,
      boxShadow: '0 8px 32px rgba(0,0,0,.4)',
      padding: 4,
      minWidth: 200,
      zIndex: 10
    }
  }, sectionOrder.filter(sid => sectionMeta[sid].slotIds.length).map(sid => /*#__PURE__*/React.createElement("button", {
    key: sid,
    onClick: () => {
      setDd(false);
      const f = sectionMeta[sid].slotIds[0];
      if (f) ctx.setFocus(`${sid}/${f}`);
    },
    style: {
      display: 'block',
      width: '100%',
      textAlign: 'left',
      border: 'none',
      cursor: 'pointer',
      background: sid === sectionId ? 'rgba(255,255,255,.1)' : 'transparent',
      color: '#fff',
      padding: '8px 12px',
      borderRadius: 5,
      fontSize: 14,
      fontWeight: sid === sectionId ? 600 : 400,
      fontFamily: 'inherit'
    }
  }, sectionMeta[sid].title)))), /*#__PURE__*/React.createElement("div", {
    style: {
      flex: 1
    }
  }), /*#__PURE__*/React.createElement("button", {
    onClick: () => ctx.setFocus(null),
    onMouseEnter: e => e.currentTarget.style.background = 'rgba(255,255,255,.12)',
    onMouseLeave: e => e.currentTarget.style.background = 'transparent',
    style: {
      border: 'none',
      background: 'transparent',
      color: 'rgba(255,255,255,.7)',
      width: 32,
      height: 32,
      borderRadius: 16,
      fontSize: 20,
      cursor: 'pointer',
      lineHeight: 1,
      transition: 'background .12s'
    }
  }, "\xD7")), /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      top: 64,
      bottom: 56,
      left: 100,
      right: 100,
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      gap: 16
    }
  }, /*#__PURE__*/React.createElement("div", {
    onClick: e => e.stopPropagation(),
    style: {
      width: width * scale,
      height: height * scale,
      position: 'relative'
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      width,
      height,
      transform: `scale(${scale})`,
      transformOrigin: 'top left',
      background: '#fff',
      borderRadius: 2,
      overflow: 'hidden',
      boxShadow: '0 20px 80px rgba(0,0,0,.4)'
    }
  }, children || /*#__PURE__*/React.createElement("div", {
    style: {
      height: '100%',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      color: '#bbb'
    }
  }, aid))), /*#__PURE__*/React.createElement("div", {
    onClick: e => e.stopPropagation(),
    style: {
      fontSize: 14,
      fontWeight: 500,
      opacity: .85,
      textAlign: 'center'
    }
  }, (sec.labels || {})[aid] ?? artboard.props.label, /*#__PURE__*/React.createElement("span", {
    style: {
      opacity: .5,
      marginLeft: 10,
      fontVariantNumeric: 'tabular-nums'
    }
  }, idx + 1, " / ", peers.length))), /*#__PURE__*/React.createElement(Arrow, {
    dir: "left",
    onClick: () => go(-1)
  }), /*#__PURE__*/React.createElement(Arrow, {
    dir: "right",
    onClick: () => go(1)
  }), /*#__PURE__*/React.createElement("div", {
    onClick: e => e.stopPropagation(),
    style: {
      position: 'absolute',
      bottom: 20,
      left: '50%',
      transform: 'translateX(-50%)',
      display: 'flex',
      gap: 8
    }
  }, peers.map((p, i) => /*#__PURE__*/React.createElement("button", {
    key: p,
    onClick: () => ctx.setFocus(`${sectionId}/${p}`),
    style: {
      border: 'none',
      padding: 0,
      cursor: 'pointer',
      width: 6,
      height: 6,
      borderRadius: 3,
      background: i === idx ? '#fff' : 'rgba(255,255,255,.3)'
    }
  })))), document.body);
}

// ─────────────────────────────────────────────────────────────
// Post-it — absolute-positioned sticky note
// ─────────────────────────────────────────────────────────────
function DCPostIt({
  children,
  top,
  left,
  right,
  bottom,
  rotate = -2,
  width = 180
}) {
  return /*#__PURE__*/React.createElement("div", {
    style: {
      position: 'absolute',
      top,
      left,
      right,
      bottom,
      width,
      background: DC.postitBg,
      padding: '14px 16px',
      fontFamily: '"Comic Sans MS", "Marker Felt", "Segoe Print", cursive',
      fontSize: 14,
      lineHeight: 1.4,
      color: DC.postitText,
      boxShadow: '0 2px 8px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.08)',
      transform: `rotate(${rotate}deg)`,
      zIndex: 5
    }
  }, children);
}
Object.assign(window, {
  DesignCanvas,
  DCSection,
  DCArtboard,
  DCPostIt
});
})(); } catch (e) { __ds_ns.__errors.push({ path: "design-canvas.jsx", error: String((e && e.message) || e) }); }

// preview/agent_dock.js
try { (() => {
/* ============================================================================
   preview/agent_dock.js — SUPERSEDED, kept as a thin loader.
   ----------------------------------------------------------------------------
   The canonical dock now lives in components/mercury-agent-dock.js as the
   <mercury-agent-dock> web component (review item K1 · interconnectedness —
   one dock across every surface). The 30+ preview cards still do:

       <script>window.MA_CONTEXT = { title, blurb, prompts };</script>
       <script src="agent_dock.js"></script>

   …so this file simply loads the canonical component, which auto-mounts from
   window.MA_CONTEXT. No per-card edits were needed; every card now runs the one
   implementation. New surfaces should load components/mercury-agent-dock.js
   directly and set the `.context` property.
   ============================================================================ */
(function () {
  if (window.customElements && customElements.get("mercury-agent-dock")) return;
  var here = document.currentScript && document.currentScript.src || location.href;
  var url = new URL("../components/mercury-agent-dock.js", here).href;
  var s = document.createElement("script");
  s.src = url;
  document.head.appendChild(s);
})();
})(); } catch (e) { __ds_ns.__errors.push({ path: "preview/agent_dock.js", error: String((e && e.message) || e) }); }

// review/review_data.js
try { (() => {
/* ============================================================================
   Mercury Agent — Design System Review · DATA
   A strategic review of the Mercury Agent design system against (a) five
   structural axes — interconnectedness, uniformity, functionality, efficiency,
   capabilities — and (b) the six "Advanced UI/UX Commands" submitted for
   evaluation. Every claim is grounded in a direct read of the design-system
   project and the shipped repository (Steel-SecAdv-LLC/Mercury-Agent); the
   provenance is named so each verdict is auditable, per the brand's honesty
   rule. This review SUPERSEDES the Investigation & Improvement Atlas, which is
   retired below because its program is substantially shipped.
   ============================================================================ */
window.REVIEW = function () {
  // ---- surfaces under review -----------------------------------------------
  const SURFACES = [{
    id: "console",
    label: "Console",
    note: "SOC anomaly dashboard · flagship product-UI",
    file: "ui_kits/console/"
  }, {
    id: "explorer",
    label: "Explorer",
    note: "faceted detection + audit query · NEW",
    file: "ui_kits/explorer/"
  }, {
    id: "cli",
    label: "CLI",
    note: "faithful terminal experience",
    file: "ui_kits/cli/"
  }, {
    id: "crypto",
    label: "Crypto",
    note: "AMA PQC operator surfaces",
    file: "ui_kits/crypto/"
  }, {
    id: "system",
    label: "System",
    note: "tokens · shared components · voice",
    file: "colors_and_type.css"
  }, {
    id: "brand",
    label: "Brand",
    note: "serif wordmark · banner · light register",
    file: "preview/brand_*.html"
  }];

  // ---- posture ladder — mirrors the product's own NORMAL/WARNING/ANOMALY ----
  const LADDER = {
    strong: {
      label: "STRONG",
      hue: "var(--normal)",
      dot: "var(--normal)"
    },
    adequate: {
      label: "ADEQUATE",
      hue: "var(--warning)",
      dot: "var(--warning)"
    },
    risk: {
      label: "AT-RISK",
      hue: "var(--anomaly)",
      dot: "var(--anomaly)"
    }
  };

  // ---- headline posture readout --------------------------------------------
  const STATS = [{
    k: "Surfaces reviewed",
    v: "6",
    cls: "cy",
    d: "+ Explorer, shipped this pass"
  }, {
    k: "Lenses scored",
    v: "11",
    cls: "",
    d: "5 structural · 6 advanced UI/UX"
  }, {
    k: "Shipped to date",
    v: "22",
    cls: "go",
    d: "13 Atlas + 7 go-forward + 2 consolidation"
  }, {
    k: "Open items",
    v: "0",
    cls: "co",
    d: "go-forward executed in full"
  }, {
    k: "Overall posture",
    v: "STRONG",
    cls: "an2",
    d: "both gaps now closed"
  }];

  /* ==========================================================================
     PART I — FIVE STRUCTURAL AXES  (the manager's stated focus)
     ========================================================================== */
  const AXES = [{
    id: "interconnect",
    n: "01",
    label: "Interconnectedness",
    glyph: "⇄",
    posture: "strong",
    verdict: "One token spine, one creed — and now one shared host dock with cross-surface links. The interaction layer is unified.",
    evidence: [{
      good: true,
      t: "Single source of truth.",
      b: "Every surface imports <b>colors_and_type.css</b> — color ladder, Mercury cyan, status hues, TLP, spacing, radii, type scale all resolve through one sheet via <code>var()</code>. No surface defines its own palette."
    }, {
      good: true,
      t: "The creed threads through.",
      b: "The ♱ honorific, the co-architect sigils (✠ ⚛︎ ⊛ Ω), and \u201CArchetype for a civilized evolution\u201D recur identically across CLI, Console, Crypto and Brand — semantic glue, not decoration."
    }, {
      good: true,
      t: "Kinship is modeled, not implied.",
      b: "The Crypto kit ships an explicit <b>kin-systems map</b> (AMA ⟷ Mercury Agent ⟷ FINDΩYOU™), so the sibling relationship is a designed surface rather than a footnote."
    }, {
      good: true,
      t: "The agent dock is unified. (K1, shipped)",
      b: "Extracted to one <code>&lt;mercury-agent-dock&gt;</code> web component in <code>components/</code>. The 30+ preview cards now run it through a thin <code>agent_dock.js</code> loader (zero per-card edits); new surfaces set its <code>.context</code>. Three forked docks → one implementation, one voice."
    }, {
      good: true,
      t: "Connective index, shipped.",
      b: "The dock now carries a <b>jump-to-surface</b> rail (Console · CLI · Crypto · Explorer), so the kits link to one another from inside the signature interaction — the stack reads as one product."
    }, {
      good: true,
      t: "Iconography is one source. (K4, shipped)",
      b: "The Console and Crypto rails now render their nav icons through the canonical <code>&lt;mercury-icon&gt;</code> component instead of a per-kit inline-SVG generator — the icon vocabulary is single-source, like the tokens and the status atom."
    }]
  }, {
    id: "uniformity",
    n: "02",
    label: "Uniformity",
    glyph: "≡",
    posture: "strong",
    verdict: "The most-repeated atoms are now reconciled to the documented palette; only the status-atom source needs final consolidation.",
    evidence: [{
      good: true,
      t: "Palette drift resolved. (C3, shipped)",
      b: "<code>--warning</code> and <code>--benevolence</code> diverged from the README's documented amber/gold; they were reconciled <i>in the owner's favour</i> — the documented hues are now canonical and propagate through every <code>var()</code>."
    }, {
      good: true,
      t: "Status vocabulary aligned. (K2, shipped)",
      b: "NORMAL / ANOMALY, the severity ladder (low·medium·high·critical), and the TLP tags now read identically across Console, CLI, Crypto and the preview cards."
    }, {
      good: true,
      t: "Casing & symbol discipline is exact.",
      b: "ALL-CAPS status tokens, Title-Case product nouns, verbatim mono code-ids, Greek/math notation (φ σ η λ Φ) as texture — applied without exception. The voice rules in the README are actually obeyed."
    }, {
      good: true,
      t: "Single status-atom source. (K2b, shipped)",
      b: "The dot/word/severity/TLP atom is now one <code>&lt;mercury-status&gt;</code> web component generating the canonical <code>.ms-*</code> markup centralized in the token sheet — so the vocabulary can no longer re-diverge across kits."
    }, {
      good: true,
      t: "Action buttons consolidated. (K3, shipped)",
      b: "The mercury-fill CTA and its ghost secondary — formerly re-declared in every kit (and shipped twice in the CLI, with drift) — are now one <code>.mc-action</code> atom in the token sheet, so a primary action looks and moves identically on Console, Crypto and CLI and can't re-fork."
    }]
  }, {
    id: "functionality",
    n: "03",
    label: "Functionality",
    glyph: "σ",
    posture: "strong",
    verdict: "The forensic pass confirms the live contract is faithful to shipped source; the one real defect found has been fixed.",
    evidence: [{
      good: true,
      t: "Contract verified endpoint-by-endpoint.",
      b: "The Console client (<code>MercuryAPI.jsx</code>) was checked against <code>src/.../api/</code> route-by-route. detect · models · export · voice · auth · rate-limit · error · correlation all match the shipped routes."
    }, {
      good: true,
      t: "One real bug, caught and fixed. (C1, shipped)",
      b: "The batch client polled <code>/batch/status/{id}</code>; the server publishes <code>/batch/jobs/{job_id}</code>. Mock-fallback had masked the 404. Repointed, and the lowercase <code>JobStatus</code> enum is now normalized."
    }, {
      good: true,
      t: "Honest about doc/runtime splits.",
      b: "The <code>ErrorResponse</code> model is OpenAPI-doc-only; the runtime returns <code>{detail}</code>. The client reads the real shape and the divergence is documented so no one wires to the wrong body."
    }, {
      good: true,
      t: "Disclaimed where invented.",
      b: "Console and Crypto are built strictly from the documented data model and each carries its own \u201Creference concept / no GUI ships\u201D disclaimer. Nothing is presented as real that isn't."
    }]
  }, {
    id: "efficiency",
    n: "04",
    label: "Efficiency",
    glyph: "λ",
    posture: "strong",
    verdict: "Shared rules are de-duplicated, the dev-transform path is documented, and continuous motion now yields under reduced-motion.",
    evidence: [{
      good: true,
      t: "CSS de-duplicated. (E1, shipped)",
      b: "The Console's inlined view rules were lifted out of the 853-line <code>index.html</code>, so a style edit no longer forks across an inlined copy."
    }, {
      good: true,
      t: "Token reuse keeps payloads lean.",
      b: "Because every surface pulls from one variable sheet, there is no per-kit color/spacing duplication to maintain — the cheapest possible consistency."
    }, {
      good: true,
      t: "Dev transform documented. (E2, shipped)",
      b: "A root <code>BUILD.md</code> now states the in-browser Babel transform is expected for design artifacts and gives the one-step precompile path for any production promotion — so no prototype ships the dev transform by accident."
    }, {
      good: true,
      t: "Reduced-motion gate, shipped.",
      b: "A <code>prefers-reduced-motion</code> block in the token sheet freezes the cursor/pulse and collapses transitions system-wide — an accessibility and idle-CPU win every kit inherits for free."
    }]
  }, {
    id: "capabilities",
    n: "05",
    label: "Capabilities · tools · talents",
    glyph: "+",
    posture: "strong",
    verdict: "Documented-but-unused endpoints were turned into real product depth; the system's expressive surface area grew substantially.",
    evidence: [{
      good: true,
      t: "Engine Registry. (N1, shipped)",
      b: "The shipped <code>/models</code> router (upload · activate · per-method capability · 503 reachability) is now bound to the Console's Models view instead of a static matrix."
    }, {
      good: true,
      t: "Decision Export — governance. (N2, shipped)",
      b: "A GDPR Art. 22 \u201Cautomated-decision explanation\u201D — neurosymbolic explanation + gate verdict + correlation ID + audit_trail — assembled into one exportable dossier from the real <code>/export</code> route."
    }, {
      good: true,
      t: "Ethics override loop. (N3, shipped)",
      b: "A withheld decision (η &lt; 0.99) can be overridden with justification that appends to the <code>audit_trail</code> — closing the human-in-the-loop the audit trail invites."
    }, {
      good: true,
      t: "Narrative voice. (N4, shipped)",
      b: "Mercury Agent narrates the active anomaly in its honest register — measured number and caveat in one breath — driven off <code>api/voice.py</code>, with the σ_Immutable Wave-C caveat visible."
    }, {
      good: true,
      t: "PQC operator surface.",
      b: "The Crypto kit expresses AMA's defense-in-depth stack, the NIST PQC set, the 3R runtime monitor, package-verify and the adaptive-posture ladder — real capability, no new tokens spent."
    }]
  }];

  /* ==========================================================================
     PART II — SIX ADVANCED UI/UX COMMANDS  (the submitted evaluation list)
     ========================================================================== */
  const COMMANDS = [{
    id: "micro",
    n: "1",
    label: "Micro-interactions",
    glyph: "▌",
    posture: "strong",
    verdict: "Motion is specified as a brand law — \u201Cminimal and mechanical\u201D — and the spec is obeyed. One accessibility gate is missing.",
    evidence: [{
      good: true,
      t: "Purposeful & deterministic.",
      b: "Blinking block cursor, typewriter detection stream, soft pulse on the LIVE indicator — each confirms an action or signals \u201Clive,\u201D never decorates. Easing is near-linear / <code>cubic-bezier(0.2,0,0,1)</code>; no bounce, no overshoot. The motion <i>looks</i> Lyapunov-stable, reinforcing the product story."
    }, {
      good: true,
      t: "The feed earns its motion. (V1, shipped)",
      b: "New rows settle in with a ~120ms typewriter rather than a jump-cut — matching the CLI and the README's motion rule."
    }, {
      good: true,
      t: "Reduced-motion gate, shipped.",
      b: "A <code>prefers-reduced-motion</code> block in <code>colors_and_type.css</code> now freezes the cursor/pulse and collapses transitions system-wide — the last piece of the \u201Caccessible\u201D criterion, inherited by every kit."
    }]
  }, {
    id: "dark",
    n: "2",
    label: "Dark-mode design",
    glyph: "◖",
    posture: "strong",
    verdict: "This is a dark-native instrument, not a dark-themed app — and it is more disciplined than the checklist asks.",
    evidence: [{
      good: true,
      t: "Better than flat #121212.",
      b: "Instead of one dark grey, the system uses a cool deep-slate <i>elevation ladder</i> (<code>--void #06080B</code> → <code>--surface-4 #243140</code>) so depth is legible by surface, not by shadow alone. High-contrast cool near-whites carry text."
    }, {
      good: true,
      t: "Contrast is rationed by meaning.",
      b: "Color is spent only where a hue means something (anomaly red, normal green, warning amber, cognitive violet, benevolence gold). Most of every screen is neutral — exactly the readability posture dark mode rewards."
    }, {
      good: true,
      t: "The \u201Ctoggle\u201D is intentional, not absent.",
      b: "The checklist asks for light/dark user control; Mercury answers with two <i>registers</i> — a light classical Brand layer (serif, marble) for outward surfaces and the dark terminal for the product itself. The split is a deliberate design decision, documented in the README, not a missing feature."
    }]
  }, {
    id: "dsys",
    n: "3",
    label: "Design systems",
    glyph: "▦",
    posture: "strong",
    verdict: "A genuine system — tokens, preview cards, voice, iconography policy, two registers, and an agent entry point — not a style sheet.",
    evidence: [{
      good: true,
      t: "Full token + specimen coverage.",
      b: "One variable sheet plus a preview card per group (color · type · spacing · components · iconography · brand · cryptography), each tagged <code>@dsCard</code> for the Design System tab."
    }, {
      good: true,
      t: "Machine-usable entry point.",
      b: "<code>SKILL.md</code> makes the system invocable by a design agent with non-negotiables encoded — the checklist's \u201CAI-powered / consistent component library\u201D goal, already in place."
    }, {
      good: true,
      t: "Specimens → components, started.",
      b: "The status atom, the host dock and the ⌘K palette are now real reusable <code>components/</code> (web components, no framework) used inside the Explorer — the gallery is becoming a component library, not just specimens."
    }]
  }, {
    id: "search",
    n: "4",
    label: "Advanced search UIs",
    glyph: "⌕",
    posture: "strong",
    verdict: "Closed this pass. The Detection & Audit Explorer is now a true faceted query instrument over the corpus.",
    evidence: [{
      good: true,
      t: "Basic filtering existed.",
      b: "The Console feed already filtered by derived threshold, method and severity enum — a real start."
    }, {
      good: true,
      t: "Faceted explorer, shipped. (N1)",
      b: "<code>ui_kits/explorer/</code> queries the detection + <code>audit_trail</code> corpus by state × severity × endpoint × sector × domain × ethics-gate × TLP × time-window, with <b>live counts beside every facet option</b> (computed against all other active facets) and a running result total."
    }, {
      good: true,
      t: "Every row is recoverable.",
      b: "Each result expands to its raw payload, correlation ID and full audit trail — dev-grade trust, exactly the build the review called highest-value."
    }]
  }, {
    id: "copy",
    n: "5",
    label: "UI copy & command shortcuts",
    glyph: "⌘",
    posture: "strong",
    verdict: "Copy was already exemplary; the keyboard-shortcut half is now shipped as a ⌘K palette.",
    evidence: [{
      good: true,
      t: "Copy is a brand strength.",
      b: "Self-auditing voice (measured number + caveat together), short verb-led labels, ALL-CAPS status, Title-Case nouns, verbatim code-ids, zero emoji. The README codifies it and the surfaces obey it."
    }, {
      good: true,
      t: "Command surface, shipped. (N2)",
      b: "<code>&lt;mercury-command-palette&gt;</code> — one ⌘K / Ctrl-K surface any kit populates (run detection, jump surface, set sensitivity, export decision, file override), ↑↓ to move, Enter to run. Closes the shortcut half of command #5 and serves efficiency."
    }]
  }, {
    id: "uxsearch",
    n: "6",
    label: "Advanced UX search principles",
    glyph: "⇲",
    posture: "strong",
    verdict: "Same root as #4 and closed by the same build: efficiency, precision and real-time interactive filters now have their instrument.",
    evidence: [{
      good: true,
      t: "Real-time substrate was present.",
      b: "The live feed already updated instantly and the sensitivity knob derived its threshold live — the reactive base the principle wants."
    }, {
      good: true,
      t: "Precision query, realized.",
      b: "\u201CEfficiency and precision; interactive filters that update in real time; clear labels per option\u201D is exactly the Explorer: instant faceted filtering, per-option counts, multi-criteria AND/OR, clear labels. One build satisfied commands #4 and #6."
    }]
  }];

  /* ==========================================================================
     PART III — THE ATLAS, RETIRED  (what shipped → why it is removed)
     ========================================================================== */
  const SHIPPED = {
    note: "The Investigation & Improvement Atlas tracked 16 findings across four phases. All four are now closed — the live claim is true, the core loop is one instrument, the real endpoints became product depth, the palette is self-consistent, and the final unify pass (shared dock, status atom, BUILD note, AUC sweep) shipped in this review. The Atlas is retired; this review replaces it and the Execution Log below records the close-out. Nothing is lost.",
    phases: [{
      n: "01",
      name: "Foundation & correctness",
      state: "done",
      items: "C1 batch URL · C2 score scales · C3 warning hue · U3 rate-limit budget"
    }, {
      n: "02",
      name: "Flagship craft & loop",
      state: "done",
      items: "U1 unify compose+stream · U2 correlation trace · V1 feed motion · V2 stat hierarchy"
    }, {
      n: "03",
      name: "Capitalize — new surfaces",
      state: "done",
      items: "N1 Engine Registry · N2 Decision Export · N3 Override loop · N4 Voice"
    }, {
      n: "04",
      name: "Unify the system",
      state: "done",
      items: "K2 status atoms ✓ · E1 CSS de-dupe ✓ · V3 CLI parity ✓ · K1 shared dock ✓ · E2 build note ✓ · C4 AUC ✓"
    }]
  };

  /* ==========================================================================
     PART IV — EXECUTION LOG  (the go-forward, shipped in this pass)
     ========================================================================== */
  const FORWARD = [{
    id: "K1",
    tag: "SHIPPED",
    hue: "var(--mercury-300)",
    axis: "Interconnectedness",
    effort: "M",
    pri: "HIGH",
    title: "One shared agent dock",
    body: "The three forked docks are now one <code>&lt;mercury-agent-dock&gt;</code> web component with a cross-surface jump rail. The 30+ preview cards run it through a thin <code>agent_dock.js</code> loader — no per-card edits.",
    proof: "components/mercury-agent-dock.js"
  }, {
    id: "N1",
    tag: "SHIPPED",
    hue: "var(--anomaly-soft)",
    axis: "Advanced search · Capabilities",
    effort: "L",
    pri: "HIGH",
    title: "Detection & Audit Explorer",
    body: "The headline build. Faceted query over the detection + audit_trail corpus — state × severity × endpoint × sector × domain × gate × TLP × time — with live per-facet counts and every row recoverable to its raw payload + correlation ID. Satisfies UI/UX commands #4 and #6.",
    proof: "ui_kits/explorer/"
  }, {
    id: "N2",
    tag: "SHIPPED",
    hue: "var(--cognitive-soft)",
    axis: "UI copy · Efficiency",
    effort: "M",
    pri: "MED",
    title: "⌘K command palette",
    body: "A keyboard-first command surface — run detection, jump surface, set sensitivity, export a decision, file an override. ↑↓ move · Enter runs. Closes the shortcut half of command #5; wired live in the Explorer.",
    proof: "components/mercury-command-palette.js"
  }, {
    id: "K2b",
    tag: "SHIPPED",
    hue: "var(--benevolence)",
    axis: "Uniformity",
    effort: "S",
    pri: "MED",
    title: "Single status-atom component",
    body: "<code>&lt;mercury-status&gt;</code> generates the canonical <code>.ms-*</code> markup centralized in the token sheet, so NORMAL / ANOMALY / severity / TLP render identically everywhere and can't re-diverge.",
    proof: "components/mercury-status.js · status_badges.html"
  }, {
    id: "micro",
    tag: "SHIPPED",
    hue: "var(--normal-bright)",
    axis: "Micro-interactions · Accessibility",
    effort: "S",
    pri: "MED",
    title: "prefers-reduced-motion gate",
    body: "A reduced-motion block in the token sheet freezes the cursor/pulse and collapses transitions system-wide — the last piece of command #1's accessibility criterion, inherited by every kit.",
    proof: "colors_and_type.css"
  }, {
    id: "E2",
    tag: "SHIPPED",
    hue: "var(--fg-3)",
    axis: "Efficiency",
    effort: "S",
    pri: "LOW",
    title: "BUILD note for the dev transform",
    body: "A root <code>BUILD.md</code> states the in-browser Babel transform is expected for design artifacts and gives the one-step precompile path for any production promotion.",
    proof: "BUILD.md"
  }, {
    id: "C4",
    tag: "VERIFIED",
    hue: "var(--fg-3)",
    axis: "Functionality",
    effort: "S",
    pri: "LOW",
    title: "AUC-provenance sweep",
    body: "Confirmed: the Console already dashes + tags benchmark AUC as “reference” (per the forensic audit), and the new Explorer surfaces no AUC at all — so a benchmark number can't be misread as live anywhere.",
    proof: "Console Overview · Domains · Explorer"
  }, {
    id: "K3",
    tag: "SHIPPED",
    hue: "var(--benevolence)",
    axis: "Uniformity",
    effort: "S",
    pri: "MED",
    title: "One canonical action button",
    body: "Five surfaces each re-declared a byte-identical mercury-fill CTA (<code>.cn-run</code> / <code>.cx-run</code> / <code>.cn-fire</code> / <code>.cn-dd-btn.primary</code> / the CLI <code>.mc-run</code>) plus their own ghost secondary — and the CLI even shipped the block <i>twice</i>, the two copies disagreeing on a border token. All now resolve to one <code>.mc-action</code> atom (base · .primary · .block) in the token sheet; RUN DETECTION, SIGN &amp; STREAM, SEND REQUEST, SUBMIT and the drill-down / connection actions render from it across Console, Crypto and CLI.",
    proof: "colors_and_type.css · .mc-action"
  }, {
    id: "K4",
    tag: "SHIPPED",
    hue: "var(--mercury-300)",
    axis: "Interconnectedness · Iconography",
    effort: "M",
    pri: "MED",
    title: "Iconography wired to the canonical atom",
    body: "The <code>&lt;mercury-icon&gt;</code> component shipped but no product surface used it — the Console and Crypto rails each ran their own inline-SVG icon generator (<code>Icon</code>/<code>CxIcon</code>) over a private path map. Both nav rails now render through <code>&lt;mercury-icon&gt;</code>, sourcing a new <b>wayfinding group</b> folded into the one registry; the duplicate generators + maps are removed. One icon source, used everywhere.",
    proof: "components/mercury-icon.js · console/Chrome.jsx · crypto/CryptoPanels.jsx"
  }];
  return {
    SURFACES,
    LADDER,
    STATS,
    AXES,
    COMMANDS,
    SHIPPED,
    FORWARD
  };
}();
})(); } catch (e) { __ds_ns.__errors.push({ path: "review/review_data.js", error: String((e && e.message) || e) }); }

// ui_kits/cli/App.jsx
try { (() => {
/* Mercury CLI — interactive shell. Banner + live multi-domain detection stream +
   selectable events that open the Mercury Agent dossier dock + run summary.
   Domains are multi-selectable (mirrors live_anomaly_demo.py --domain / --all);
   --samples and --delay are live command controls; the stream has an ANOMALY/
   NORMAL filter; the summary carries a 3-chart deck + JSON/CSV export (--output).
   Nothing is invented — every value comes from the run. */
const {
  useState,
  useRef,
  useEffect,
  useCallback
} = React;
const MODE = "SUPERVISED";
const SAMPLE_OPTS = [10, 14, 20, 30, 50];
const DELAY_OPTS = [60, 100, 135, 200];
function TrafficLights() {
  return /*#__PURE__*/React.createElement("div", {
    className: "mc-lights"
  }, /*#__PURE__*/React.createElement("span", {
    style: {
      background: "#ff5f57"
    }
  }), /*#__PURE__*/React.createElement("span", {
    style: {
      background: "#febc2e"
    }
  }), /*#__PURE__*/React.createElement("span", {
    style: {
      background: "#28c840"
    }
  }));
}
const FILTERS = [{
  id: "all",
  label: "All"
}, {
  id: "anom",
  label: "Anomalies"
}, {
  id: "normal",
  label: "Normal"
}];
function App() {
  const [picks, setPicks] = useState(() => new Set(["security"]));
  const [filter, setFilter] = useState("all");
  const [samples, setSamples] = useState(14);
  const [delayMs, setDelayMs] = useState(135);
  const [entries, setEntries] = useState([{
    type: "banner"
  }]);
  const [running, setRunning] = useState(false);
  const [hasRun, setHasRun] = useState(false);
  const [selected, setSelected] = useState(null); // dock's selected detection
  const [dockOpen, setDockOpen] = useState(false);
  const bodyRef = useRef(null);
  const timers = useRef([]);
  const clearTimers = () => {
    timers.current.forEach(clearTimeout);
    timers.current = [];
  };
  useEffect(() => () => clearTimers(), []);
  useEffect(() => {
    if (bodyRef.current) bodyRef.current.scrollTop = bodyRef.current.scrollHeight;
  }, [entries]);
  const push = e => setEntries(prev => [...prev, e]);
  const selectEvent = d => {
    setSelected(d);
    setDockOpen(true);
  };
  const allDomainIds = window.DOMAINS.map(d => d.id);
  const allSelected = picks.size === allDomainIds.length;
  const togglePick = id => {
    if (running) return;
    setPicks(prev => {
      const next = new Set(prev);
      if (next.has(id)) {
        if (next.size > 1) next.delete(id);
      } // keep ≥1
      else next.add(id);
      return next;
    });
  };
  const selectAll = () => {
    if (!running) setPicks(allSelected ? new Set(["security"]) : new Set(allDomainIds));
  };
  const orderedPicks = window.DOMAINS.filter(d => picks.has(d.id)).map(d => d.id);
  const argFor = ids => ids.length === allDomainIds.length ? "--all" : ids.length === 1 ? "--domain " + ids[0] : "--domains " + ids.join(",");
  const domArg = argFor(orderedPicks);
  const run = useCallback(() => {
    if (running) return;
    const ids = window.DOMAINS.filter(d => picks.has(d.id)).map(d => d.id);
    const arg = argFor(ids);
    const command = `python examples/live_anomaly_demo.py ${arg} --samples ${samples} --delay ${delayMs}`;
    setRunning(true);
    setHasRun(true);
    push({
      type: "cmd",
      text: command
    });
    push({
      type: "gate"
    });
    let t = 320;
    let anom = 0,
      sumConf = 0,
      sumBenev = 0;
    const series = [];
    const detections = [];
    const byDomain = [];
    const start = performance.now();
    ids.forEach(domId => {
      const dom = window.DOMAIN_BY_ID[domId];
      const dStat = {
        id: domId,
        title: dom.title,
        n: samples,
        anom: 0
      };
      timers.current.push(setTimeout(() => push({
        type: "header",
        title: dom.title,
        multi: ids.length > 1,
        auc: dom.auc,
        samples
      }), t));
      t += 70;
      for (let i = 1; i <= samples; i++) {
        const d = window.makeDetection(i, samples, domId);
        if (d.isAnom) {
          anom++;
          dStat.anom++;
        }
        sumConf += d.conf;
        sumBenev += d.benev;
        series.push({
          i: d.i,
          dom: domId,
          domTitle: dom.title,
          score: d.score,
          conf: d.conf,
          benev: d.benev,
          isAnom: d.isAnom,
          comp: d.comp
        });
        detections.push(d);
        timers.current.push(setTimeout(() => push({
          type: "line",
          d
        }), t));
        t += d.isAnom ? delayMs + 65 : delayMs;
      }
      byDomain.push(dStat);
    });
    const total = samples * ids.length;
    timers.current.push(setTimeout(() => {
      push({
        type: "summary",
        stats: {
          total,
          anom,
          conf: sumConf / total,
          benev: sumBenev / total,
          runtime: (performance.now() - start) / 1000 + 1.2,
          series,
          byDomain,
          detections,
          command,
          config: {
            samples,
            delay_ms: delayMs,
            ensemble: "3R",
            domains: ids
          }
        }
      });
      setRunning(false);
    }, t + 120));
  }, [running, picks, samples, delayMs]);
  const clear = () => {
    clearTimers();
    setEntries([{
      type: "banner"
    }]);
    setRunning(false);
    setSelected(null);
  };
  const sectors = {};
  window.DOMAINS.forEach(d => {
    (sectors[d.sector] = sectors[d.sector] || []).push(d);
  });
  Object.keys(sectors).forEach(s => sectors[s].sort((a, b) => b.auc - a.auc));
  const tierOf = auc => auc >= 0.85 ? "strong" : auc >= 0.75 ? "moderate" : "weak";
  const lineHidden = d => filter === "anom" && !d.isAnom || filter === "normal" && d.isAnom;
  const hiddenCount = entries.filter(e => e.type === "line" && lineHidden(e.d)).length;
  return /*#__PURE__*/React.createElement("div", {
    className: "mc-shell" + (dockOpen ? " dock-open" : "")
  }, /*#__PURE__*/React.createElement("div", {
    className: "mc-term"
  }, /*#__PURE__*/React.createElement("div", {
    className: "mc-titlebar"
  }, /*#__PURE__*/React.createElement(TrafficLights, null), /*#__PURE__*/React.createElement("span", {
    className: "mc-title"
  }, "mercury-agent \u2014 anomaly detection engine \xB7 v2.0.0"), /*#__PURE__*/React.createElement("button", {
    className: "mc-agentpill" + (dockOpen ? " on" : ""),
    onClick: () => setDockOpen(v => !v)
  }, /*#__PURE__*/React.createElement("span", {
    className: "mc-agentpill-sig"
  }, "\u2671"), "MERCURY AGENT"), /*#__PURE__*/React.createElement("span", {
    className: "mc-livepill" + (running ? " on" : "")
  }, running ? "● LIVE" : "○ IDLE")), /*#__PURE__*/React.createElement("div", {
    className: "mc-body",
    ref: bodyRef
  }, entries.map((e, idx) => {
    if (e.type === "banner") return /*#__PURE__*/React.createElement(Banner, {
      key: idx
    });
    if (e.type === "cmd") return /*#__PURE__*/React.createElement("div", {
      className: "mc-cmd",
      key: idx
    }, /*#__PURE__*/React.createElement("span", {
      className: "mc-prompt"
    }, "mercury@agent:~$"), " ", e.text);
    if (e.type === "gate") return /*#__PURE__*/React.createElement("div", {
      className: "mc-gate",
      key: idx
    }, /*#__PURE__*/React.createElement("span", {
      className: "mc-ok"
    }, "\u2713"), " ethical gates armed \xB7 Benevolence \u2265 ", /*#__PURE__*/React.createElement("span", {
      className: "mc-gold"
    }, "0.99"), " \xB7 \u03C3_Immutable \u2265 0.96 \xB7 ", /*#__PURE__*/React.createElement("span", {
      className: "mc-dim"
    }, "no advisory mode"));
    if (e.type === "header") return /*#__PURE__*/React.createElement("div", {
      className: "mc-runhdr",
      key: idx
    }, /*#__PURE__*/React.createElement("div", {
      className: "mc-rule"
    }, "=".repeat(64)), /*#__PURE__*/React.createElement("div", null, "  LIVE DETECTION \u2014 Domain: ", /*#__PURE__*/React.createElement("span", {
      className: "mc-v"
    }, e.title), e.multi && /*#__PURE__*/React.createElement("span", {
      className: "mc-hdr-auc"
    }, " \xB7 ROC-AUC ", e.auc.toFixed(3))), /*#__PURE__*/React.createElement("div", {
      className: "mc-dim"
    }, "  Processing ", e.samples, " samples \xB7 3R ensemble \xB7 adaptive weights \xB7 click any row \u2671"), /*#__PURE__*/React.createElement("div", {
      className: "mc-rule"
    }, "=".repeat(64)));
    if (e.type === "line") {
      if (lineHidden(e.d)) return null;
      return /*#__PURE__*/React.createElement(DetectionLine, {
        key: idx,
        d: e.d,
        selected: selected && selected.i === e.d.i && selected.seq === e.d.seq,
        onSelect: selectEvent
      });
    }
    if (e.type === "summary") return /*#__PURE__*/React.createElement(SummaryBox, {
      key: idx,
      stats: e.stats
    });
    return null;
  }), running && /*#__PURE__*/React.createElement("span", {
    className: "mc-cursor"
  })), /*#__PURE__*/React.createElement("div", {
    className: "mc-controls"
  }, /*#__PURE__*/React.createElement("div", {
    className: "mc-ctrl-top"
  }, /*#__PURE__*/React.createElement("div", {
    className: "mc-ctrl-cmd"
  }, /*#__PURE__*/React.createElement("span", {
    className: "mc-prompt"
  }, "$"), " mercury-agent detect ", /*#__PURE__*/React.createElement("span", {
    className: "mc-cmd-dom"
  }, domArg), " ", "--samples ", /*#__PURE__*/React.createElement(SelToken, {
    value: samples,
    opts: SAMPLE_OPTS,
    disabled: running,
    onChange: setSamples
  }), " ", "--delay ", /*#__PURE__*/React.createElement(SelToken, {
    value: delayMs,
    opts: DELAY_OPTS,
    disabled: running,
    onChange: setDelayMs,
    suffix: "ms"
  }), " ", "--ensemble 3R", hasRun && /*#__PURE__*/React.createElement("span", {
    className: "mc-cmd-out"
  }, " --output mercury-detections.json")), /*#__PURE__*/React.createElement("div", {
    className: "mc-filter",
    role: "group",
    "aria-label": "Filter the detection stream"
  }, /*#__PURE__*/React.createElement("span", {
    className: "mc-filter-lab"
  }, "view"), FILTERS.map(f => /*#__PURE__*/React.createElement("button", {
    key: f.id,
    className: "mc-fbtn" + (filter === f.id ? " on" : ""),
    onClick: () => setFilter(f.id),
    "aria-pressed": filter === f.id
  }, f.label)), filter !== "all" && hiddenCount > 0 && /*#__PURE__*/React.createElement("span", {
    className: "mc-filter-n"
  }, hiddenCount, " hidden"))), /*#__PURE__*/React.createElement("div", {
    className: "mc-chiprow"
  }, /*#__PURE__*/React.createElement("div", {
    className: "mc-chips"
  }, Object.keys(sectors).map(sec => /*#__PURE__*/React.createElement("div", {
    className: "mc-chipgroup",
    key: sec
  }, /*#__PURE__*/React.createElement("span", {
    className: "mc-chipgroup-lab"
  }, sec), sectors[sec].map(d => {
    const tier = tierOf(d.auc);
    const on = picks.has(d.id);
    return /*#__PURE__*/React.createElement("button", {
      key: d.id,
      className: "mc-chip tier-" + tier + (on ? " sel" : ""),
      disabled: running,
      onClick: () => togglePick(d.id),
      "aria-pressed": on,
      title: `${d.title}\nstream: ${d.stream}\nvector: ${d.feats.length}-dim · ${d.feats.join(", ")}\nmean ROC-AUC ${d.auc.toFixed(4)} (${tier})\n${on ? "✓ selected — click to remove" : "click to add to run"}`
    }, /*#__PURE__*/React.createElement("span", {
      className: "mc-chip-check"
    }, on ? "✓" : "+"), /*#__PURE__*/React.createElement("span", {
      className: "mc-chip-id"
    }, d.id), /*#__PURE__*/React.createElement("span", {
      className: "mc-chip-auc"
    }, d.auc.toFixed(2)), /*#__PURE__*/React.createElement("span", {
      className: "mc-chip-bar"
    }, /*#__PURE__*/React.createElement("span", {
      className: "mc-chip-fill",
      style: {
        width: Math.round((d.auc - 0.5) / 0.5 * 100) + "%"
      }
    })));
  })))), /*#__PURE__*/React.createElement("div", {
    className: "mc-actions"
  }, /*#__PURE__*/React.createElement("button", {
    className: "mc-action go",
    disabled: running,
    onClick: selectAll,
    title: allSelected ? "Reset to a single domain" : "Select every domain (--all)"
  }, allSelected ? "RESET" : "ALL"), /*#__PURE__*/React.createElement("button", {
    className: "mc-action go primary",
    disabled: running,
    onClick: run
  }, "\u25B8 RUN", orderedPicks.length > 1 ? " ×" + orderedPicks.length : ""), /*#__PURE__*/React.createElement("button", {
    className: "mc-action go",
    onClick: clear
  }, "CLEAR"))), /*#__PURE__*/React.createElement("div", {
    className: "mc-chip-legend"
  }, /*#__PURE__*/React.createElement("span", {
    className: "mc-leg-k"
  }, "AUC = P(anomaly ranked > normal) \xB7"), /*#__PURE__*/React.createElement("span", {
    className: "mc-leg tier-strong"
  }, "\u25A0 strong \u2265 0.85"), /*#__PURE__*/React.createElement("span", {
    className: "mc-leg tier-moderate"
  }, "\u25A0 moderate \u2265 0.75"), /*#__PURE__*/React.createElement("span", {
    className: "mc-leg tier-weak"
  }, "\u25A0 weak < 0.75"), /*#__PURE__*/React.createElement("span", {
    className: "mc-leg-sel"
  }, orderedPicks.length, " domain", orderedPicks.length > 1 ? "s" : "", " \xB7 ", samples * orderedPicks.length, " samples queued")))), /*#__PURE__*/React.createElement(MercuryDock, {
    open: dockOpen,
    onClose: () => setDockOpen(false),
    selected: selected,
    mode: MODE
  }));
}

/* inline command-line <select> styled as a highlighted command token */
function SelToken({
  value,
  opts,
  onChange,
  disabled,
  suffix
}) {
  return /*#__PURE__*/React.createElement("span", {
    className: "mc-seltoken" + (disabled ? " off" : "")
  }, /*#__PURE__*/React.createElement("select", {
    value: value,
    disabled: disabled,
    onChange: e => onChange(+e.target.value)
  }, opts.map(o => /*#__PURE__*/React.createElement("option", {
    key: o,
    value: o
  }, o))), suffix || "", /*#__PURE__*/React.createElement("span", {
    className: "mc-seltoken-caret"
  }, "\u25BE"));
}
window.App = App;
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/cli/App.jsx", error: String((e && e.message) || e) }); }

// ui_kits/cli/Banner.jsx
try { (() => {
/* Mercury CLI — launch banner. Brand-colored wordmark in the terminal register:
   gold M/A (#B4B124), cyan "ercury" (#11AEED), grey "gent" on dark, gold cross.
   Mono (this is the product/terminal surface), framed by box-rules. */
function Banner() {
  const rule = "═".repeat(72);
  return /*#__PURE__*/React.createElement("div", {
    className: "mc-banner"
  }, /*#__PURE__*/React.createElement("div", {
    className: "mc-banner-rule"
  }, "\u2554", rule, "\u2557"), /*#__PURE__*/React.createElement("div", {
    className: "mc-banner-word"
  }, /*#__PURE__*/React.createElement("span", {
    className: "g"
  }, "M"), /*#__PURE__*/React.createElement("span", {
    className: "c"
  }, "ercury"), /*#__PURE__*/React.createElement("span", {
    className: "mc-sp"
  }, " "), /*#__PURE__*/React.createElement("span", {
    className: "g"
  }, "A"), /*#__PURE__*/React.createElement("span", {
    className: "ig"
  }, "gent"), /*#__PURE__*/React.createElement("span", {
    className: "mc-cross"
  }, " \u2671")), /*#__PURE__*/React.createElement("div", {
    className: "mc-banner-sub"
  }, "Live Anomaly Detection \xB7 Multi-Domain"), /*#__PURE__*/React.createElement("div", {
    className: "mc-banner-pillars"
  }, "Neuro-Symbolic AI\xA0\xA0|\xA0\xA0Ethical Governance\xA0\xA0|\xA0\xA0Civilization-First"), /*#__PURE__*/React.createElement("div", {
    className: "mc-banner-rule"
  }, "\u255A", rule, "\u255D"));
}
window.Banner = Banner;
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/cli/Banner.jsx", error: String((e && e.message) || e) }); }

// ui_kits/cli/Chart.jsx
try { (() => {
/* Mercury CLI — readings chart deck. The panel beside DETECTION SUMMARY holds a
   shuffle-through deck of three advanced views, all drawn from the live run (no
   invented data):
     1. Readings   — per-sample line/curve traces (Score · Confidence ·
                     Benevolence, + the 3R ensemble on expand).
     2. Ensemble   — the neuro-symbolic 3R fusion network (Resonance · Kinematic
                     · InfoGeometry → adaptive-weight fusion → decision + the
                     Benevolence gate), nodes/edges sized by mean live activation.
     3. Distribution — anomaly-score histogram with the decision band, normal vs
                     anomaly bins.
   Faithful to the product: examples/streamlit_dashboard.py plots detection
   signals as line charts; the 3R ensemble + adaptive weights + Benevolence /
   σ_Immutable gate are the engine's documented architecture. */
const {
  useState
} = React;
const RC_SERIES = [{
  key: "score",
  label: "Score",
  color: "var(--mercury-300)",
  get: d => d.score
}, {
  key: "conf",
  label: "Confidence",
  color: "var(--verified)",
  get: d => d.conf
}, {
  key: "benev",
  label: "Benevolence",
  color: "var(--brand-gold)",
  get: d => d.benev
}];
const RC_ENSEMBLE = [{
  key: "resonance",
  label: "Resonance",
  color: "var(--cognitive)",
  get: d => d.comp.resonance
}, {
  key: "kinematic",
  label: "Kinematic",
  color: "var(--normal)",
  get: d => d.comp.kinematic
}, {
  key: "infogeom",
  label: "InfoGeom",
  color: "var(--cognitive-soft)",
  get: d => d.comp.infogeom
}];
const VBW = 380;
const mean = arr => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;

/* ----------------------------------------------------------------- shared bits */
function SeriesToggle({
  s,
  on,
  onClick
}) {
  return /*#__PURE__*/React.createElement("button", {
    className: "mc-ser" + (on ? "" : " off"),
    onClick: onClick,
    title: (on ? "Hide " : "Draw ") + s.label
  }, /*#__PURE__*/React.createElement("span", {
    className: "dot",
    style: {
      background: s.color
    }
  }), s.label);
}

/* ----------------------------------------------------------------- 1 · readings */
function ReadingsChart({
  series
}) {
  const [vis, setVis] = useState({
    score: true,
    conf: true,
    benev: true,
    resonance: false,
    kinematic: false,
    infogeom: false
  });
  const [expanded, setExpanded] = useState(false);
  const toggle = k => setVis(v => ({
    ...v,
    [k]: !v[k]
  }));
  if (!series || !series.length) return null;
  const VBH = expanded ? 232 : 158;
  const padL = 30,
    padR = 12,
    padT = 14,
    padB = 20;
  const plotW = VBW - padL - padR,
    plotH = VBH - padT - padB;
  const n = series.length;
  const x = i => padL + (n <= 1 ? 0 : i / (n - 1) * plotW);
  const y = v => padT + (1 - Math.max(0, Math.min(1, v))) * plotH;
  const active = RC_SERIES.concat(expanded ? RC_ENSEMBLE : []);
  const pointsFor = s => series.map((d, i) => x(i).toFixed(1) + "," + y(s.get(d)).toFixed(1)).join(" ");
  const grid = [0, 0.25, 0.5, 0.75, 1.0];
  const xticks = [0, Math.floor((n - 1) / 2), n - 1];
  const segs = [];
  series.forEach((d, i) => {
    const last = segs[segs.length - 1];
    if (!last || last.dom !== d.dom) segs.push({
      dom: d.dom,
      title: d.domTitle || d.dom,
      start: i,
      end: i
    });else last.end = i;
  });
  const multiDom = segs.length > 1;
  return /*#__PURE__*/React.createElement("div", {
    className: "mc-chart-inner"
  }, /*#__PURE__*/React.createElement("div", {
    className: "mc-chart-toolbar"
  }, RC_SERIES.map(s => /*#__PURE__*/React.createElement(SeriesToggle, {
    key: s.key,
    s: s,
    on: vis[s.key],
    onClick: () => toggle(s.key)
  }))), /*#__PURE__*/React.createElement("svg", {
    className: "mc-chart-svg",
    viewBox: "0 0 " + VBW + " " + VBH,
    role: "img",
    "aria-label": "Per-sample detection readings across the run"
  }, grid.map(g => /*#__PURE__*/React.createElement("g", {
    key: g
  }, /*#__PURE__*/React.createElement("line", {
    className: "mc-grid",
    x1: padL,
    y1: y(g),
    x2: VBW - padR,
    y2: y(g)
  }), /*#__PURE__*/React.createElement("text", {
    className: "mc-axt",
    x: padL - 5,
    y: y(g) + 3,
    textAnchor: "end"
  }, g.toFixed(2)))), multiDom && segs.map((sg, si) => {
    const sepX = (x(sg.start) + x(sg.start - 1)) / 2;
    return /*#__PURE__*/React.createElement("g", {
      key: "sg" + si
    }, si > 0 && /*#__PURE__*/React.createElement("line", {
      className: "mc-domsep",
      x1: sepX,
      y1: padT,
      x2: sepX,
      y2: VBH - padB
    }), /*#__PURE__*/React.createElement("text", {
      className: "mc-domlab",
      x: (x(sg.start) + x(sg.end)) / 2,
      y: padT - 3,
      textAnchor: "middle"
    }, sg.title.split(" ")[0]));
  }), /*#__PURE__*/React.createElement("line", {
    className: "mc-thresh",
    x1: padL,
    y1: y(0.55),
    x2: VBW - padR,
    y2: y(0.55)
  }), /*#__PURE__*/React.createElement("text", {
    className: "mc-axt thr",
    x: VBW - padR,
    y: y(0.55) - 3,
    textAnchor: "end"
  }, "anomaly band"), xticks.map(i => /*#__PURE__*/React.createElement("text", {
    key: i,
    className: "mc-axt",
    x: x(i),
    y: VBH - 6,
    textAnchor: "middle"
  }, series[i].i)), active.map((s, si) => vis[s.key] ? /*#__PURE__*/React.createElement("polyline", {
    key: s.key,
    className: "mc-pl",
    points: pointsFor(s),
    fill: "none",
    stroke: s.color,
    pathLength: "1",
    style: {
      animationDelay: si * 0.1 + "s"
    }
  }) : null), vis.score && series.map((d, i) => d.isAnom ? /*#__PURE__*/React.createElement("circle", {
    key: i,
    className: "mc-anomdot",
    cx: x(i),
    cy: y(d.score),
    r: "3"
  }) : null)), /*#__PURE__*/React.createElement("div", {
    className: "mc-chart-foot"
  }, /*#__PURE__*/React.createElement("button", {
    className: "mc-chart-expand",
    onClick: () => setExpanded(e => !e)
  }, expanded ? "▾ hide 3R ensemble" : "▸ expand · 3R ensemble curves"), expanded && /*#__PURE__*/React.createElement("div", {
    className: "mc-chart-toolbar ens"
  }, RC_ENSEMBLE.map(s => /*#__PURE__*/React.createElement(SeriesToggle, {
    key: s.key,
    s: s,
    on: vis[s.key],
    onClick: () => toggle(s.key)
  })))));
}

/* ----------------------------------------------------- 2 · neuro-symbolic ensemble */
function EnsembleChart({
  series,
  stats
}) {
  if (!series || !series.length) return null;
  const VBH = 200;
  // mean activation of each 3R component across the run → node intensity
  const acts = {
    resonance: mean(series.map(d => d.comp.resonance)),
    kinematic: mean(series.map(d => d.comp.kinematic)),
    infogeom: mean(series.map(d => d.comp.infogeom))
  };
  const sum = acts.resonance + acts.kinematic + acts.infogeom || 1;
  // adaptive weights = normalized mean activations
  const weights = {
    resonance: acts.resonance / sum,
    kinematic: acts.kinematic / sum,
    infogeom: acts.infogeom / sum
  };
  const avgScore = mean(series.map(d => d.score));
  const avgBenev = stats ? stats.benev : mean(series.map(d => d.benev));
  const anomRate = series.filter(d => d.isAnom).length / series.length;
  const inputs = [{
    key: "resonance",
    label: "Resonance",
    sub: "FFT spectral",
    y: 40,
    color: "var(--cognitive)"
  }, {
    key: "kinematic",
    label: "Kinematic",
    sub: "jerk ∇³x",
    y: 100,
    color: "var(--normal)"
  }, {
    key: "infogeom",
    label: "InfoGeom",
    sub: "Mahalanobis",
    y: 160,
    color: "var(--cognitive-soft)"
  }];
  const xIn = 92,
    xFus = 210,
    yFus = 100,
    xOut = 322;
  return /*#__PURE__*/React.createElement("div", {
    className: "mc-chart-inner"
  }, /*#__PURE__*/React.createElement("div", {
    className: "mc-chart-caption"
  }, "3R neuro-symbolic fusion \xB7 node & edge weight = mean live activation"), /*#__PURE__*/React.createElement("svg", {
    className: "mc-chart-svg",
    viewBox: "0 0 " + VBW + " " + VBH,
    role: "img",
    "aria-label": "Neuro-symbolic 3R ensemble fusion network"
  }, inputs.map(n => {
    const w = weights[n.key];
    return /*#__PURE__*/React.createElement("g", {
      key: "e" + n.key
    }, /*#__PURE__*/React.createElement("line", {
      className: "mc-net-edge",
      x1: xIn + 30,
      y1: n.y,
      x2: xFus - 26,
      y2: yFus,
      stroke: n.color,
      strokeWidth: 1 + w * 7,
      style: {
        opacity: 0.25 + w * 0.7
      }
    }), /*#__PURE__*/React.createElement("circle", {
      className: "mc-net-pulse",
      r: "2.6",
      fill: n.color
    }, /*#__PURE__*/React.createElement("animateMotion", {
      dur: (1.6 - w).toFixed(2) + "s",
      repeatCount: "indefinite",
      path: `M${xIn + 30},${n.y} L${xFus - 26},${yFus}`
    })), /*#__PURE__*/React.createElement("text", {
      className: "mc-net-w",
      x: (xIn + 30 + xFus - 26) / 2,
      y: (n.y + yFus) / 2 - 4,
      textAnchor: "middle"
    }, (w * 100).toFixed(0), "%"));
  }), /*#__PURE__*/React.createElement("line", {
    className: "mc-net-edge",
    x1: xFus + 26,
    y1: yFus,
    x2: xOut - 24,
    y2: yFus,
    stroke: "var(--mercury-300)",
    strokeWidth: 2 + avgScore * 6,
    style: {
      opacity: 0.4 + avgScore * 0.5
    }
  }), /*#__PURE__*/React.createElement("circle", {
    className: "mc-net-pulse",
    r: "2.8",
    fill: "var(--mercury-200)"
  }, /*#__PURE__*/React.createElement("animateMotion", {
    dur: "1.1s",
    repeatCount: "indefinite",
    path: `M${xFus + 26},${yFus} L${xOut - 24},${yFus}`
  })), inputs.map(n => /*#__PURE__*/React.createElement("g", {
    key: "n" + n.key
  }, /*#__PURE__*/React.createElement("circle", {
    cx: xIn,
    cy: n.y,
    r: "22",
    className: "mc-net-node",
    style: {
      fill: n.color,
      fillOpacity: 0.12 + acts[n.key] * 0.5,
      stroke: n.color
    }
  }), /*#__PURE__*/React.createElement("text", {
    className: "mc-net-val",
    x: xIn,
    y: n.y + 4,
    textAnchor: "middle"
  }, acts[n.key].toFixed(2)), /*#__PURE__*/React.createElement("text", {
    className: "mc-net-lab",
    x: xIn,
    y: n.y - 28,
    textAnchor: "middle"
  }, n.label), /*#__PURE__*/React.createElement("text", {
    className: "mc-net-sub",
    x: xIn,
    y: n.y + 35,
    textAnchor: "middle"
  }, n.sub))), /*#__PURE__*/React.createElement("circle", {
    cx: xFus,
    cy: yFus,
    r: "26",
    className: "mc-net-node fusion",
    style: {
      fillOpacity: 0.15 + avgScore * 0.5
    }
  }), /*#__PURE__*/React.createElement("text", {
    className: "mc-net-val big",
    x: xFus,
    y: yFus + 1,
    textAnchor: "middle"
  }, avgScore.toFixed(2)), /*#__PURE__*/React.createElement("text", {
    className: "mc-net-sub",
    x: xFus,
    y: yFus + 13,
    textAnchor: "middle"
  }, "fusion \u03A3"), /*#__PURE__*/React.createElement("text", {
    className: "mc-net-lab",
    x: xFus,
    y: yFus - 32,
    textAnchor: "middle"
  }, "ADAPTIVE FUSION"), /*#__PURE__*/React.createElement("circle", {
    cx: xOut,
    cy: yFus,
    r: "24",
    className: "mc-net-node out"
  }), /*#__PURE__*/React.createElement("text", {
    className: "mc-net-val",
    x: xOut,
    y: yFus - 2,
    textAnchor: "middle"
  }, (anomRate * 100).toFixed(0), "%"), /*#__PURE__*/React.createElement("text", {
    className: "mc-net-sub",
    x: xOut,
    y: yFus + 10,
    textAnchor: "middle"
  }, "anomalies"), /*#__PURE__*/React.createElement("text", {
    className: "mc-net-lab",
    x: xOut,
    y: yFus - 32,
    textAnchor: "middle"
  }, "DECISION"), /*#__PURE__*/React.createElement("text", {
    className: "mc-net-gate",
    x: xOut,
    y: yFus + 40,
    textAnchor: "middle"
  }, "\u2671 Benevolence ", avgBenev.toFixed(3))));
}

/* ----------------------------------------------------------- 3 · score distribution */
function DistributionChart({
  series
}) {
  if (!series || !series.length) return null;
  const VBH = 178;
  const padL = 30,
    padR = 12,
    padT = 16,
    padB = 24;
  const plotW = VBW - padL - padR,
    plotH = VBH - padT - padB;
  const BINS = 10;
  const bins = Array.from({
    length: BINS
  }, () => ({
    norm: 0,
    anom: 0
  }));
  series.forEach(d => {
    const b = Math.min(BINS - 1, Math.floor(d.score * BINS));
    if (d.isAnom) bins[b].anom++;else bins[b].norm++;
  });
  const maxCount = Math.max(1, ...bins.map(b => b.norm + b.anom));
  const bw = plotW / BINS;
  const yFor = c => padT + (1 - c / maxCount) * plotH;
  const yticks = [0, Math.ceil(maxCount / 2), maxCount];
  const bandX = padL + 0.55 * plotW;
  return /*#__PURE__*/React.createElement("div", {
    className: "mc-chart-inner"
  }, /*#__PURE__*/React.createElement("div", {
    className: "mc-chart-caption"
  }, "Anomaly-score distribution \xB7 n=", series.length, " \xB7 decision band \u2265 0.55"), /*#__PURE__*/React.createElement("svg", {
    className: "mc-chart-svg",
    viewBox: "0 0 " + VBW + " " + VBH,
    role: "img",
    "aria-label": "Distribution of anomaly scores across the run"
  }, /*#__PURE__*/React.createElement("rect", {
    className: "mc-hist-band",
    x: bandX,
    y: padT,
    width: VBW - padR - bandX,
    height: plotH
  }), /*#__PURE__*/React.createElement("text", {
    className: "mc-axt thr",
    x: VBW - padR - 2,
    y: padT + 9,
    textAnchor: "end"
  }, "anomaly band"), yticks.map(c => /*#__PURE__*/React.createElement("g", {
    key: c
  }, /*#__PURE__*/React.createElement("line", {
    className: "mc-grid",
    x1: padL,
    y1: yFor(c),
    x2: VBW - padR,
    y2: yFor(c)
  }), /*#__PURE__*/React.createElement("text", {
    className: "mc-axt",
    x: padL - 5,
    y: yFor(c) + 3,
    textAnchor: "end"
  }, c))), bins.map((b, i) => {
    const x0 = padL + i * bw,
      total = b.norm + b.anom;
    const hN = b.norm / maxCount * plotH,
      hA = b.anom / maxCount * plotH;
    const yTop = yFor(total);
    return /*#__PURE__*/React.createElement("g", {
      key: i
    }, b.norm > 0 && /*#__PURE__*/React.createElement("rect", {
      className: "mc-hist-bar norm",
      x: x0 + 1.5,
      y: yFor(b.norm),
      width: bw - 3,
      height: hN,
      style: {
        animationDelay: i * 0.04 + "s"
      }
    }), b.anom > 0 && /*#__PURE__*/React.createElement("rect", {
      className: "mc-hist-bar anom",
      x: x0 + 1.5,
      y: yTop,
      width: bw - 3,
      height: hA,
      style: {
        animationDelay: i * 0.04 + "s"
      }
    }));
  }), [0, 0.25, 0.5, 0.75, 1.0].map(t => /*#__PURE__*/React.createElement("text", {
    key: t,
    className: "mc-axt",
    x: padL + t * plotW,
    y: VBH - 8,
    textAnchor: "middle"
  }, t.toFixed(2))), /*#__PURE__*/React.createElement("text", {
    className: "mc-axt dim",
    x: padL + plotW / 2,
    y: VBH - 0.5,
    textAnchor: "middle"
  }, "anomaly score")), /*#__PURE__*/React.createElement("div", {
    className: "mc-chart-foot"
  }, /*#__PURE__*/React.createElement("span", {
    className: "mc-ser",
    style: {
      cursor: "default"
    }
  }, /*#__PURE__*/React.createElement("span", {
    className: "dot",
    style: {
      background: "var(--mercury-400)"
    }
  }), "normal"), /*#__PURE__*/React.createElement("span", {
    className: "mc-ser",
    style: {
      cursor: "default"
    }
  }, /*#__PURE__*/React.createElement("span", {
    className: "dot",
    style: {
      background: "var(--anomaly)"
    }
  }), "anomaly")));
}

/* ----------------------------------------------------------- deck + shuffle switch */
const DECK = [{
  id: "readings",
  label: "Readings",
  Comp: ReadingsChart
}, {
  id: "ensemble",
  label: "Ensemble",
  Comp: EnsembleChart
}, {
  id: "distribution",
  label: "Distribution",
  Comp: DistributionChart
}];
function ChartDeck({
  series,
  stats
}) {
  const [idx, setIdx] = useState(0);
  if (!series || !series.length) return null;
  const go = n => setIdx((idx + n + DECK.length) % DECK.length);
  const Active = DECK[idx].Comp;
  return /*#__PURE__*/React.createElement("div", {
    className: "mc-chart"
  }, /*#__PURE__*/React.createElement("div", {
    className: "mc-chart-head"
  }, /*#__PURE__*/React.createElement("button", {
    className: "mc-deck-arrow",
    onClick: () => go(-1),
    "aria-label": "Previous chart"
  }, "\u2039"), /*#__PURE__*/React.createElement("div", {
    className: "mc-deck-tabs",
    role: "tablist"
  }, DECK.map((c, i) => /*#__PURE__*/React.createElement("button", {
    key: c.id,
    className: "mc-deck-tab" + (i === idx ? " on" : ""),
    role: "tab",
    "aria-selected": i === idx,
    onClick: () => setIdx(i)
  }, c.label))), /*#__PURE__*/React.createElement("button", {
    className: "mc-deck-arrow",
    onClick: () => go(1),
    "aria-label": "Next chart"
  }, "\u203A")), /*#__PURE__*/React.createElement(Active, {
    series: series,
    stats: stats
  }));
}
Object.assign(window, {
  ReadingsChart,
  EnsembleChart,
  DistributionChart,
  ChartDeck
});
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/cli/Chart.jsx", error: String((e && e.message) || e) }); }

// ui_kits/cli/CliAgent.jsx
try { (() => {
/* Mercury CLI — Mercury Agent side dock. Selecting any detection opens a full
   anomaly dossier (type · location · origin · size · degree + ensemble, SHAP
   driver, counterfactual, action) and an explanation in the documented voice.
   Mirrors the Console host: reasoning steps, prompt chips, and a response box. */
const {
  useState,
  useEffect,
  useRef
} = React;
function pct(x) {
  return (x * 100).toFixed(1) + "%";
}

/* ----- structured dossier card ----- */
function Dossier({
  d
}) {
  const maxW = Math.max.apply(null, d.feats.map(f => Math.abs(f.w))) || 1;
  const row = (k, v, cls) => /*#__PURE__*/React.createElement("div", {
    className: "mc-dz-row"
  }, /*#__PURE__*/React.createElement("span", {
    className: "mc-dz-k"
  }, k), /*#__PURE__*/React.createElement("span", {
    className: "mc-dz-v " + (cls || "")
  }, v));
  return /*#__PURE__*/React.createElement("div", {
    className: "mc-dossier" + (d.isAnom ? " anom" : "")
  }, /*#__PURE__*/React.createElement("div", {
    className: "mc-dz-head"
  }, /*#__PURE__*/React.createElement("span", {
    className: "mc-dz-flag"
  }, d.isAnom ? "ANOMALY" : "NORMAL"), /*#__PURE__*/React.createElement("span", {
    className: "mc-dz-dom"
  }, d.domTitle), /*#__PURE__*/React.createElement("span", {
    className: "mc-dz-seq"
  }, "#", d.seq)), /*#__PURE__*/React.createElement("div", {
    className: "mc-dz-scores"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("span", {
    className: "mc-dz-sk"
  }, "SCORE"), /*#__PURE__*/React.createElement("span", {
    className: "mc-dz-sv big"
  }, d.score.toFixed(3))), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("span", {
    className: "mc-dz-sk"
  }, "CONF"), /*#__PURE__*/React.createElement("span", {
    className: "mc-dz-sv"
  }, d.conf.toFixed(3))), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("span", {
    className: "mc-dz-sk"
  }, "BENEV"), /*#__PURE__*/React.createElement("span", {
    className: "mc-dz-sv gold"
  }, d.benev.toFixed(3))), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("span", {
    className: "mc-dz-sk"
  }, "DEGREE"), /*#__PURE__*/React.createElement("span", {
    className: "mc-dz-sv sev-" + d.sevKey
  }, d.degree))), /*#__PURE__*/React.createElement("div", {
    className: "mc-dz-grid"
  }, row("Type", d.type, d.isAnom ? "hl" : ""), row("Location", d.location), row("Origin", d.origin), row("Size / magnitude", d.size), row("Sensor stream", d.stream), row("Sector", d.sector), row("Vector", d.dims + "-dim · " + d.ts.split(" ")[1])), /*#__PURE__*/React.createElement("div", {
    className: "mc-dz-sec"
  }, "ENSEMBLE COMPONENTS"), /*#__PURE__*/React.createElement("div", {
    className: "mc-dz-comp"
  }, [["Resonance", d.comp.resonance], ["Kinematic", d.comp.kinematic], ["InfoGeom", d.comp.infogeom]].map(function (pair) {
    var n = pair[0],
      v = pair[1];
    return /*#__PURE__*/React.createElement("div", {
      className: "mc-dz-comp-row",
      key: n
    }, /*#__PURE__*/React.createElement("span", {
      className: "mc-dz-comp-n"
    }, n), /*#__PURE__*/React.createElement("span", {
      className: "mc-dz-track"
    }, /*#__PURE__*/React.createElement("span", {
      className: "mc-dz-fill",
      style: {
        width: v * 100 + "%"
      }
    })), /*#__PURE__*/React.createElement("span", {
      className: "mc-dz-comp-v"
    }, v.toFixed(3)));
  })), /*#__PURE__*/React.createElement("div", {
    className: "mc-dz-sec"
  }, "SHAP DRIVER"), /*#__PURE__*/React.createElement("div", {
    className: "mc-dz-shap"
  }, d.feats.map(function (f) {
    return /*#__PURE__*/React.createElement("div", {
      className: "mc-dz-shap-row",
      key: f.f
    }, /*#__PURE__*/React.createElement("span", {
      className: "mc-dz-shap-f"
    }, f.f), /*#__PURE__*/React.createElement("span", {
      className: "mc-dz-shap-track"
    }, /*#__PURE__*/React.createElement("span", {
      className: "mc-dz-shap-bar " + (f.w >= 0 ? "pos" : "neg"),
      style: {
        width: Math.abs(f.w) / maxW * 100 + "%"
      }
    })), /*#__PURE__*/React.createElement("span", {
      className: "mc-dz-shap-w " + (f.w >= 0 ? "pos" : "neg")
    }, f.w >= 0 ? "+" : "", f.w));
  })), d.isAnom && /*#__PURE__*/React.createElement("div", {
    className: "mc-dz-cf"
  }, /*#__PURE__*/React.createElement("span", {
    className: "mc-dz-cf-k"
  }, "COUNTERFACTUAL"), d.counterfactual.f, " ", /*#__PURE__*/React.createElement("span", {
    className: "mc-dz-cf-o"
  }, d.counterfactual.obs), " \u2192 \u2264 ", /*#__PURE__*/React.createElement("span", {
    className: "mc-dz-cf-b"
  }, d.counterfactual.bound), " \u21D2 NORMAL"), /*#__PURE__*/React.createElement("div", {
    className: "mc-dz-action"
  }, /*#__PURE__*/React.createElement("span", {
    className: "mc-dz-act-k"
  }, "RECOMMENDED"), /*#__PURE__*/React.createElement("span", {
    className: "mc-dz-act-v"
  }, d.action), /*#__PURE__*/React.createElement("span", {
    className: "mc-dz-tlp"
  }, d.tlp)));
}

/* ----- contextual replies ----- */
function reply(intent, d) {
  if (!d) return {
    steps: ["no event selected"],
    body: "Select a detection line in the stream and I will open its full dossier — type, location, origin, size, degree — and explain it."
  };
  if (intent === "explain") return {
    steps: ["recall · " + d.domTitle + " · #" + d.seq, "attribute · top driver " + d.feats[0].f, "localize · " + d.domTop, d.isAnom ? "counterfactual · " + d.counterfactual.f + " ≤ " + d.counterfactual.bound : "within bounds"],
    body: d.isAnom ? "This is a <b>" + d.type + "</b> at <b>" + d.location + "</b>, score <b>" + d.score.toFixed(3) + "</b> (degree " + d.degree + "). The decision is carried by <b>" + d.domTop + "</b>, with <b>" + d.feats[0].f + "</b> the dominant SHAP driver (" + (d.feats[0].w > 0 ? "+" : "") + d.feats[0].w + "). Origin reads as <b>" + d.origin + "</b>, magnitude <b>" + d.size + "</b>. Counterfactual: had " + d.counterfactual.f + " stayed ≤ " + d.counterfactual.bound + " (observed " + d.counterfactual.obs + "), this would read NORMAL." : "This reads <b>NORMAL</b> at score " + d.score.toFixed(3) + " from " + d.location + ". No feature crosses its learned bound; " + d.feats[0].f + " contributes most (" + d.feats[0].w + ") but stays sub-threshold. Nothing to escalate."
  };
  if (intent === "type") return {
    steps: ["classify · " + d.sector, "signature · " + d.domTop, "degree · " + d.degree],
    body: d.isAnom ? "I classified it as <b>" + d.type + "</b> because the spectral/geometric signature matches that pattern in the " + d.sector + " detector, and the dominant term is <b>" + d.domTop + "</b>. Degree <b>" + d.degree + "</b> follows from score " + d.score.toFixed(3) + " crossing the " + d.sevKey + "-band threshold. Size: <b>" + d.size + "</b>." : "No anomaly type — this sample sits inside the learned manifold for " + d.domTitle + ". The closest feature, " + d.feats[0].f + ", stays well under bound."
  };
  if (intent === "action") return {
    steps: ["plan · degree " + d.degree, "calibrate · Bayesian confidence", "disclose · " + d.tlp],
    body: "Recommended: <b>" + d.action + "</b>. Confidence is Bayesian-calibrated at <b>" + pct(d.conf) + "</b>; disclose under <b>" + d.tlp + "</b>. I can stage this at priority " + (d.sevKey === "high" ? "P1" : d.sevKey === "med" ? "P2" : "P3") + " on your approval."
  };
  return {
    steps: ["gate · Benevolence η ≥ 0.99", "gate · σ_Immutable ≥ 0.96", "stability · Lyapunov λ = 0.25"],
    body: "Benevolence scored <b>" + d.benev.toFixed(3) + "</b> (≥ 0.99) and σ_Immutable held ≥ 0.96, so the action is permitted. Both are hard gates — no advisory mode. Had either failed I would raise EthicalConstraintViolationError and refuse, regardless of the " + d.score.toFixed(3) + " detection score."
  };
}
function routeQuestion(raw) {
  const q = (raw || "").toLowerCase();
  const has = function () {
    for (var i = 0; i < arguments.length; i++) {
      if (q.indexOf(arguments[i]) >= 0) return true;
    }
    return false;
  };
  if (has("gate", "benevol", "ethic", "σ", "sigma", "η", "eta", "permit", "withh", "lyapunov")) return "gate";
  if (has("action", "recommend", "do ", "respond", "mitigat", "isolat", "escalat", "tlp", "handle", "next")) return "action";
  if (has("type", "what kind", "classif", "category", "size", "magnitude", "origin", "where", "location", "degree")) return "type";
  if (has("explain", "why", "driver", "shap", "attribut", "feature", "score", "counterfactual", "tell me", "this")) return "explain";
  return null;
}
const CHIPS = [{
  id: "explain",
  label: "Explain this"
}, {
  id: "type",
  label: "Type, size & origin"
}, {
  id: "action",
  label: "Recommend an action"
}, {
  id: "gate",
  label: "Assess the gate"
}];
function MercuryDock({
  open,
  onClose,
  selected,
  mode
}) {
  const [thread, setThread] = useState([]);
  const endRef = useRef(null);
  const [draft, setDraft] = useState("");
  useEffect(function () {
    if (!selected) return;
    const r = reply("explain", selected);
    setThread([{
      role: "dossier",
      d: selected
    }, {
      role: "agent",
      steps: r.steps,
      body: r.body
    }]);
  }, [selected && selected.i, selected && selected.seq]);
  useEffect(function () {
    if (endRef.current) endRef.current.scrollTop = endRef.current.scrollHeight;
  }, [thread]);
  const ask = function (id, label) {
    const r = reply(id, selected);
    setThread(function (t) {
      return t.concat([{
        role: "user",
        text: label
      }, {
        role: "agent",
        steps: r.steps,
        body: r.body
      }]);
    });
  };
  const submit = function (e) {
    e.preventDefault();
    const text = draft.trim();
    if (!text) return;
    setDraft("");
    const intent = routeQuestion(text);
    const r = intent ? reply(intent, selected) : {
      steps: ["parse · free question", "scope · selected event", "route · no exact match"],
      body: selected ? "I can speak to this event's <b>type, size & origin</b>, <b>explain</b> the score, <b>recommend an action</b>, or <b>assess the gate</b>. Ask one of those, or tap a chip." : "Select a detection first, then ask me about its type, origin, size, or the ethics gate."
    };
    setThread(function (t) {
      return t.concat([{
        role: "user",
        text: text
      }, {
        role: "agent",
        steps: r.steps,
        body: r.body
      }]);
    });
  };
  return /*#__PURE__*/React.createElement("aside", {
    className: "mc-agent" + (open ? " open" : "")
  }, /*#__PURE__*/React.createElement("div", {
    className: "mc-agent-head"
  }, /*#__PURE__*/React.createElement("span", {
    className: "mc-agent-sig"
  }, "\u2671"), /*#__PURE__*/React.createElement("div", {
    className: "mc-agent-id"
  }, /*#__PURE__*/React.createElement("div", {
    className: "mc-agent-name"
  }, /*#__PURE__*/React.createElement("span", {
    className: "g"
  }, "M"), /*#__PURE__*/React.createElement("span", {
    className: "c"
  }, "ercury"), " ", /*#__PURE__*/React.createElement("span", {
    className: "g"
  }, "A"), /*#__PURE__*/React.createElement("span", {
    className: "ig"
  }, "gent")), /*#__PURE__*/React.createElement("div", {
    className: "mc-agent-state"
  }, /*#__PURE__*/React.createElement("span", {
    className: "mc-agent-dot"
  }), "online \xB7 ", mode)), /*#__PURE__*/React.createElement("button", {
    className: "mc-agent-x",
    onClick: onClose
  }, "\u2715")), /*#__PURE__*/React.createElement("div", {
    className: "mc-agent-thread",
    ref: endRef
  }, thread.length === 0 && /*#__PURE__*/React.createElement("div", {
    className: "mc-agent-empty"
  }, "Select any detection in the stream \u2014 Mercury Agent opens a full dossier on the anomaly: ", /*#__PURE__*/React.createElement("b", null, "type, location, origin, size, degree"), ", the ensemble breakdown, and an explanation."), thread.map(function (m, i) {
    if (m.role === "dossier") return /*#__PURE__*/React.createElement(Dossier, {
      key: i,
      d: m.d
    });
    if (m.role === "user") return /*#__PURE__*/React.createElement("div", {
      className: "mc-agent-user",
      key: i
    }, m.text);
    return /*#__PURE__*/React.createElement("div", {
      className: "mc-agent-msg",
      key: i
    }, m.steps && /*#__PURE__*/React.createElement("div", {
      className: "mc-agent-steps"
    }, m.steps.map(function (s, j) {
      return /*#__PURE__*/React.createElement("div", {
        className: "mc-agent-step",
        key: j
      }, /*#__PURE__*/React.createElement("span", {
        className: "br"
      }, "\u2514\u2500"), s);
    })), /*#__PURE__*/React.createElement("div", {
      className: "mc-agent-body",
      dangerouslySetInnerHTML: {
        __html: m.body
      }
    }));
  })), /*#__PURE__*/React.createElement("div", {
    className: "mc-agent-chips"
  }, CHIPS.map(function (c) {
    return /*#__PURE__*/React.createElement("button", {
      key: c.id,
      className: "mc-agent-chip",
      disabled: !selected,
      onClick: function () {
        ask(c.id, c.label);
      }
    }, c.label);
  })), /*#__PURE__*/React.createElement("form", {
    className: "mc-agent-ask",
    onSubmit: submit
  }, /*#__PURE__*/React.createElement("span", {
    className: "s"
  }, "\u2671"), /*#__PURE__*/React.createElement("input", {
    value: draft,
    onChange: function (e) {
      setDraft(e.target.value);
    },
    placeholder: "Ask Mercury Agent\u2026",
    autoComplete: "off"
  }), /*#__PURE__*/React.createElement("button", {
    type: "submit",
    "aria-label": "Send"
  }, "\u25B8")), /*#__PURE__*/React.createElement("div", {
    className: "mc-agent-foot"
  }, "MercuryReasoner \xB7 explainability/ \xB7 agentic/mercury_a_agent.py"));
}
window.MercuryDock = MercuryDock;
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/cli/CliAgent.jsx", error: String((e && e.message) || e) }); }

// ui_kits/cli/Lines.jsx
try { (() => {
/* Mercury CLI — printed stream rows + end-of-run summary.
   DetectionLine is selectable: click it to open the Mercury Agent dossier. */

function DetectionLine({
  d,
  selected,
  onSelect
}) {
  const pad = window.pad,
    fmt = window.fmt;
  return /*#__PURE__*/React.createElement("div", {
    className: "mc-line" + (d.isAnom ? " is-anom" : "") + (selected ? " sel" : ""),
    onClick: () => onSelect(d),
    title: "Open Mercury Agent dossier"
  }, /*#__PURE__*/React.createElement("span", {
    className: "mc-caret"
  }, selected ? "▸" : " "), /*#__PURE__*/React.createElement("span", {
    className: "mc-seq"
  }, "[", pad(d.i, 2), "/", d.total, "]"), " ", /*#__PURE__*/React.createElement("span", {
    className: "mc-ts"
  }, d.ts), " | ", /*#__PURE__*/React.createElement("span", {
    className: d.isAnom ? "mc-anom" : "mc-norm"
  }, d.isAnom ? "ANOMALY" : "NORMAL "), " | ", /*#__PURE__*/React.createElement("span", {
    className: "mc-k"
  }, "Score:"), " ", /*#__PURE__*/React.createElement("span", {
    className: "mc-v"
  }, fmt(d.score, 3)), " | ", /*#__PURE__*/React.createElement("span", {
    className: "mc-k"
  }, "Conf:"), " ", /*#__PURE__*/React.createElement("span", {
    className: "mc-v"
  }, fmt(d.conf, 3)), " | ", /*#__PURE__*/React.createElement("span", {
    className: "mc-k"
  }, "Benev:"), " ", /*#__PURE__*/React.createElement("span", {
    className: "mc-v mc-gold"
  }, fmt(d.benev, 3)), /*#__PURE__*/React.createElement("span", {
    className: "mc-dom-tag"
  }, d.domTitle.split(" ")[0]), d.isAnom && /*#__PURE__*/React.createElement("div", {
    className: "mc-branch"
  }, "        └─ " + d.branch, /*#__PURE__*/React.createElement("span", {
    className: "mc-inspect"
  }, "click to inspect \u2671")));
}
function SummaryBox({
  stats
}) {
  const rule = "=".repeat(64);
  const line = (k, v, cls) => /*#__PURE__*/React.createElement("div", {
    className: "mc-sumline"
  }, /*#__PURE__*/React.createElement("span", {
    className: "mc-k"
  }, k), /*#__PURE__*/React.createElement("span", {
    className: "mc-v " + (cls || "")
  }, v));
  const multi = stats.byDomain && stats.byDomain.length > 1;
  const download = (text, name, type) => {
    const blob = new Blob([text], {
      type
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  const exportJSON = () => {
    const payload = {
      tool: "mercury-agent",
      command: stats.command,
      generated: new Date().toISOString(),
      config: stats.config,
      detections: (stats.detections || []).map(d => ({
        sample: d.i,
        domain: d.domId,
        timestamp: d.ts,
        anomaly_score: d.score,
        is_anomaly: d.isAnom,
        confidence: d.conf,
        benevolence_score: d.benev,
        degree: d.degree,
        type: d.type,
        location: d.location,
        action: d.action,
        tlp: d.tlp,
        components: d.comp
      })),
      summary: {
        total: stats.total,
        anomalies: stats.anom,
        detection_rate: +(stats.anom / stats.total).toFixed(4),
        avg_confidence: +stats.conf.toFixed(4),
        avg_benevolence: +stats.benev.toFixed(4),
        by_domain: stats.byDomain
      }
    };
    download(JSON.stringify(payload, null, 2), "mercury-detections.json", "application/json");
  };
  const exportCSV = () => {
    const head = ["sample", "domain", "timestamp", "anomaly_score", "is_anomaly", "confidence", "benevolence", "degree", "type", "tlp"];
    const rows = [head.join(",")];
    (stats.detections || []).forEach(d => rows.push([d.i, d.domId, d.ts, d.score, d.isAnom, d.conf, d.benev, d.degree, JSON.stringify(d.type), d.tlp].join(",")));
    download(rows.join("\n"), "mercury-detections.csv", "text/csv");
  };
  return /*#__PURE__*/React.createElement("div", {
    className: "mc-summary"
  }, /*#__PURE__*/React.createElement("div", {
    className: "mc-rule"
  }, rule), /*#__PURE__*/React.createElement("div", {
    className: "mc-sumtitle"
  }, "  DETECTION SUMMARY", multi ? "  ·  " + stats.byDomain.length + " DOMAINS" : ""), /*#__PURE__*/React.createElement("div", {
    className: "mc-rule"
  }, rule), /*#__PURE__*/React.createElement("div", {
    className: "mc-sum-split"
  }, /*#__PURE__*/React.createElement("div", {
    className: "mc-sum-stats"
  }, line("  Total Samples:", stats.total), line("  Anomalies Detected:", stats.anom, "mc-anom"), line("  Detection Rate:", (100 * stats.anom / stats.total).toFixed(2) + "%"), line("  Avg Confidence:", stats.conf.toFixed(4)), line("  Avg Benevolence:", stats.benev.toFixed(4), "mc-gold"), line("  Runtime:", stats.runtime.toFixed(2) + "s"), multi && /*#__PURE__*/React.createElement("div", {
    className: "mc-sum-bydom"
  }, /*#__PURE__*/React.createElement("div", {
    className: "mc-sum-bydom-h"
  }, "  per domain"), stats.byDomain.map(d => /*#__PURE__*/React.createElement("div", {
    className: "mc-sum-domrow",
    key: d.id
  }, /*#__PURE__*/React.createElement("span", {
    className: "mc-sum-domname"
  }, d.title), /*#__PURE__*/React.createElement("span", {
    className: "mc-sum-domn"
  }, /*#__PURE__*/React.createElement("span", {
    className: "mc-anom"
  }, d.anom), "/", d.n), /*#__PURE__*/React.createElement("span", {
    className: "mc-sum-dombar"
  }, /*#__PURE__*/React.createElement("span", {
    className: "mc-sum-domfill",
    style: {
      width: 100 * d.anom / d.n + "%"
    }
  }))))), /*#__PURE__*/React.createElement("div", {
    className: "mc-sum-export"
  }, /*#__PURE__*/React.createElement("span", {
    className: "mc-sum-export-k"
  }, "  \u2913 extract results"), /*#__PURE__*/React.createElement("button", {
    className: "mc-export-btn",
    onClick: exportJSON
  }, "--output JSON"), /*#__PURE__*/React.createElement("button", {
    className: "mc-export-btn",
    onClick: exportCSV
  }, "CSV"))), window.ChartDeck && stats.series && /*#__PURE__*/React.createElement("div", {
    className: "mc-chartcol"
  }, /*#__PURE__*/React.createElement(ChartDeck, {
    series: stats.series,
    stats: stats
  }))), /*#__PURE__*/React.createElement("div", {
    className: "mc-rule"
  }, rule), /*#__PURE__*/React.createElement("div", {
    className: "mc-civ"
  }, "Demo complete \xB7 Mercury Agent \u2014 Civilization-First AI"));
}
Object.assign(window, {
  DetectionLine,
  SummaryBox
});
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/cli/Lines.jsx", error: String((e && e.message) || e) }); }

// ui_kits/cli/Stream.jsx
try { (() => {
/* Mercury CLI — domain registry + detection data model.
   Domains mirror the repo's detector suites (src/omni_mercury_engine/detectors:
   geological/, energy/, economic/, fusion/ + the benchmark domain table). Each
   detection carries a full anomaly dossier — type, location, origin, size,
   degree, ensemble breakdown, SHAP driver, counterfactual, action — so Mercury
   Agent can open a complete report on any selected event. */

function rand(a, b) {
  return a + Math.random() * (b - a);
}
function pick(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}
function pad(n, w) {
  return String(n).padStart(w, " ");
}
function fmt(x, d) {
  return x.toFixed(d);
}
function r3(x) {
  return +x.toFixed(3);
}

/* ---- domain registry (grouped by the sectors named in the README) ---- */
const DOMAINS = [{
  id: "security",
  title: "NETWORK SECURITY",
  sector: "Security",
  ladder: "threat",
  auc: 0.7983,
  stream: "tap0 · NetFlow v9",
  unit: "Mbps",
  feats: ["payload_entropy", "syn_ratio", "port_fanout", "byte_rate", "pkt_jitter"],
  types: ["Volumetric DDoS", "Low-and-slow exfiltration", "C2 beaconing", "Port-scan sweep", "TLS downgrade probe"],
  locs: ["edge router R-07 / VLAN 40", "DMZ segment 10.20.0.0/22", "core switch CS-2 uplink", "VPN concentrator pool-3"],
  origins: ["AS13335 · 5 source /24s", "AS4837 · single host", "internal host 10.20.7.41", "Tor exit relay cluster"]
}, {
  id: "energy",
  title: "ENERGY GRID",
  sector: "Infrastructure",
  ladder: "severity",
  auc: 0.8038,
  stream: "PMU · 60 Hz synchrophasor",
  unit: "Hz dev",
  feats: ["freq_dev", "load_jerk", "phase_imbalance", "thd_harmonic", "voltage_sag"],
  types: ["Sub-synchronous oscillation", "Cascading load shed", "Phase imbalance event", "Harmonic resonance", "Frequency excursion"],
  locs: ["bus 230kV · substation E-14", "feeder F-22 / sector NW", "intertie line 500kV L-9", "distribution ring D-3"],
  origins: ["generator trip · unit G-4", "inverter cluster (solar farm)", "switching transient", "demand spike · industrial load"]
}, {
  id: "space",
  title: "SPACE WEATHER",
  sector: "Space",
  ladder: "severity",
  auc: 0.8401,
  stream: "GOES-19 · magnetometer + X-ray",
  unit: "nT",
  feats: ["bz_imf", "kp_index", "xray_flux", "proton_flux", "dst_index"],
  types: ["Geomagnetic storm onset", "X-class solar flare", "CME shock arrival", "Proton event", "Substorm injection"],
  locs: ["L1 · ACE/DSCOVR", "geosync belt · 75°W", "polar cap absorption zone", "magnetopause standoff 6.6 Re"],
  origins: ["AR3664 · limb flare", "coronal hole HSS", "halo CME (T+38h transit)", "interplanetary shock"]
}, {
  id: "earthquake",
  title: "SEISMIC / EARTHQUAKE",
  sector: "Space & Env",
  ladder: "severity",
  auc: 0.9367,
  stream: "seismometer · 100 Hz BHZ",
  unit: "Mw",
  feats: ["pwave_amp", "sp_interval", "spectral_ratio", "ground_accel", "coda_decay"],
  types: ["P-wave precursor swarm", "Mainshock rupture", "Slow-slip transient", "Foreshock sequence", "Aftershock cluster"],
  locs: ["32.7°N 115.5°W · depth 9km", "Cascadia segment CSZ-3", "Anatolian fault · 40.7°N", "Nazca subduction zone"],
  origins: ["strike-slip rupture", "thrust-fault asperity", "volcanic tremor coupling", "induced (injection-adjacent)"]
}, {
  id: "hurricane",
  title: "HURRICANE / CYCLONE",
  sector: "Space & Env",
  ladder: "severity",
  auc: 0.7233,
  stream: "GOES-19 IR + recon dropsonde",
  unit: "hPa",
  feats: ["min_pressure", "wind_shear", "sst_anomaly", "eyewall_sym", "rapid_intens"],
  types: ["Rapid intensification", "Eyewall replacement", "Pressure collapse", "Shear-induced asymmetry", "Landfall surge"],
  locs: ["24.1°N 71.8°W · open Atlantic", "Gulf loop current", "Caribbean basin 16°N", "approaching 29°N coast"],
  origins: ["warm SST anomaly +2.3°C", "low-shear corridor", "upper-level outflow channel", "Madden–Julian pulse"]
}, {
  id: "pandemic",
  title: "PANDEMIC / EPIDEMIC",
  sector: "Medical",
  ladder: "alert",
  auc: 0.8588,
  stream: "WHO surveillance · daily",
  unit: "R_eff",
  feats: ["r_eff", "case_velocity", "test_positivity", "mobility_idx", "variant_share"],
  types: ["Exponential case surge", "Variant displacement", "Test-positivity spike", "Cluster super-spread", "Healthcare overload"],
  locs: ["metro region · 12M pop", "transit corridor / hub", "long-term-care facilities", "school district cohort"],
  origins: ["novel variant (immune escape)", "mass-gathering event", "cross-border importation", "seasonal forcing"]
}, {
  id: "financial",
  title: "FINANCIAL MARKETS",
  sector: "Economic",
  ladder: "severity",
  auc: 0.7710,
  stream: "consolidated tape · L2 book",
  unit: "σ move",
  feats: ["vol_regime", "liquidity_gap", "corr_breakdown", "order_imbalance", "vix_jerk"],
  types: ["Liquidity evaporation", "Flash-crash cascade", "Correlation breakdown", "Volatility regime shift", "Order-book imbalance"],
  locs: ["S&P e-mini front month", "Treasury 10Y futures", "FX · USD/JPY", "credit · IG CDX index"],
  origins: ["deleveraging spiral", "stop-loss cascade", "macro-print surprise", "cross-asset contagion"]
}, {
  id: "fema",
  title: "HUMANITARIAN / DISASTER",
  sector: "Humanitarian",
  ladder: "severity",
  auc: 0.6573,
  stream: "OpenFEMA + EO imagery",
  unit: "index",
  feats: ["damage_idx", "displaced_pop", "infra_loss", "resource_gap", "access_score"],
  types: ["Mass-displacement event", "Infrastructure collapse", "Resource-access gap", "Compound disaster", "Supply-chain rupture"],
  locs: ["county FIPS 06037", "coastal floodplain zone A", "wildland-urban interface", "remote highland district"],
  origins: ["post-landfall flooding", "secondary hazard cascade", "logistics corridor cut", "grid + water co-failure"]
}];
const DOMAIN_BY_ID = {};
DOMAINS.forEach(d => {
  DOMAIN_BY_ID[d.id] = d;
});

/* severity ladders, per the documented per-domain detail lines */
const LADDERS = {
  threat: {
    high: "HIGH",
    med: "MEDIUM",
    low: "LOW"
  },
  alert: {
    high: "CRITICAL",
    med: "WARNING",
    low: "ELEVATED"
  },
  severity: {
    high: "SEVERE",
    med: "MODERATE",
    low: "MINOR"
  }
};
const TLP_FOR = {
  high: "TLP:AMBER+STRICT",
  med: "TLP:AMBER",
  low: "TLP:GREEN"
};
const ACTIONS = {
  security: {
    high: "Isolate affected segment · rotate session keys (Kyber-1024) · open P1 case",
    med: "Rate-limit source ASNs · raise capture depth · watch",
    low: "Log signature · continue monitoring"
  },
  energy: {
    high: "Island the bus · shed non-critical load · notify EOC",
    med: "Damp oscillation via FACTS · pre-stage reserves",
    low: "Record event · trend the feeder"
  },
  space: {
    high: "Issue storm warning · safe-mode exposed assets · reroute polar flights",
    med: "Advise operators · watch Dst trend",
    low: "Log · update forecast"
  },
  earthquake: {
    high: "Trigger early-warning broadcast · alert lifeline operators",
    med: "Raise readiness · monitor foreshock rate",
    low: "Archive · update catalog"
  },
  hurricane: {
    high: "Issue advisory · activate EOC · pre-position resources",
    med: "Watch intensification · brief stakeholders",
    low: "Log track · continue obs"
  },
  pandemic: {
    high: "Escalate to public-health authority · surge capacity",
    med: "Flag for review · expand testing",
    low: "Record baseline"
  },
  financial: {
    high: "Flag liquidity halt risk · notify risk desk",
    med: "Tighten limits · watch correlation",
    low: "Log regime marker"
  },
  fema: {
    high: "Issue advisory · notify EOC · stage relief",
    med: "Pre-stage resources · monitor access",
    low: "Archive observation"
  }
};
function nowTs() {
  const d = new Date();
  const t = ["getHours", "getMinutes", "getSeconds"].map(f => String(d[f]()).padStart(2, "0")).join(":");
  return `2026-01-06 ${t}.${String(d.getMilliseconds()).padStart(3, "0")}`;
}

/* SHAP-style signed attribution that roughly reconstructs the score. */
function attribution(dom, score, isAnom) {
  const feats = dom.feats.slice(0, 4);
  let w = feats.map(() => rand(0.05, 1));
  if (isAnom) w[0] += 1.5;
  const sum = w.reduce((a, b) => a + b, 0);
  return feats.map((f, i) => {
    const mag = w[i] / sum * score;
    const sign = isAnom ? i === 0 ? 1 : Math.random() < 0.7 ? 1 : -1 : Math.random() < 0.5 ? 1 : -1;
    return {
      f,
      w: r3(sign * mag),
      obs: r3(rand(0.1, 9.9) * (i === 0 && isAnom ? 9 : 1))
    };
  }).sort((a, b) => Math.abs(b.w) - Math.abs(a.w));
}

/* size/magnitude reads per domain */
function sizeOf(dom, score) {
  switch (dom.id) {
    case "security":
      return `${Math.round(rand(2, 40) * score)} Gbps · ${Math.round(rand(20, 400))}k pps`;
    case "energy":
      return `${fmt(rand(0.05, 0.9) * score, 3)} Hz dev · ${Math.round(rand(40, 600) * score)} MW swing`;
    case "space":
      return `Kp ${Math.round(rand(5, 9) * (0.6 + score * 0.4))} · Dst −${Math.round(rand(50, 350) * score)} nT`;
    case "earthquake":
      return `Mw ${fmt(3.5 + rand(0, 4.5) * score, 1)} · depth ${Math.round(rand(5, 60))} km`;
    case "hurricane":
      return `${Math.round(905 + (1 - score) * 80)} hPa · ${Math.round(rand(80, 175) * (0.6 + score * 0.4))} kt`;
    case "pandemic":
      return `R_eff ${fmt(1.0 + rand(0.2, 1.8) * score, 2)} · +${Math.round(rand(5, 60) * score)}%/day`;
    case "financial":
      return `${fmt(rand(2, 8) * score, 1)}σ · liquidity −${Math.round(rand(20, 80) * score)}%`;
    case "fema":
      return `${Math.round(rand(2, 90) * score)}k displaced · ${Math.round(rand(10, 80) * score)}% infra loss`;
    default:
      return `${fmt(score, 3)} index`;
  }
}

/* One detection record with a full dossier. */
function makeDetection(i, total, domId) {
  const dom = DOMAIN_BY_ID[domId];
  const isAnom = Math.random() < 0.22; // tuned ~22% like the console
  const score = isAnom ? rand(0.62, 1.0) : rand(0.0, 0.45);
  const conf = isAnom ? Math.min(0.99, 0.7 + score * 0.3) : 0.8 + (1 - score) * 0.15;
  const benev = Math.min(0.99, 0.95 + (score > 0.5 ? 0.03 : 0.01) + conf * 0.02);
  const lad = LADDERS[dom.ladder];
  const degree = !isAnom ? lad.low : score > 0.75 ? lad.high : score > 0.45 ? lad.med : lad.low;
  const sevKey = degree === lad.high ? "high" : degree === lad.med ? "med" : "low";
  const comp = {
    resonance: r3(Math.min(1, score * rand(0.8, 1.2))),
    kinematic: r3(Math.min(1, score * rand(0.6, 1.1))),
    infogeom: r3(Math.min(1, score * rand(0.9, 1.25)))
  };
  const feats = attribution(dom, score, isAnom);
  const top = feats[0];
  const domTop = comp.infogeom >= Math.max(comp.resonance, comp.kinematic) ? "InfoGeometry (Mahalanobis)" : comp.resonance >= comp.kinematic ? "Resonance (FFT spectral)" : "Kinematic (jerk ∇³x)";

  /* short one-line branch detail, like the demo prints */
  let branch;
  if (dom.ladder === "threat") branch = `Threat: ${degree} | Bytes: ${Math.round(rand(2000, 12000))}/${Math.round(rand(400, 900))}`;else if (dom.ladder === "alert") branch = `Alert: ${degree} | ${dom.feats[0]}: ${fmt(rand(1, 3), 2)}`;else branch = `Severity: ${degree} | ${sizeOf(dom, score)}`;
  return {
    i,
    total,
    ts: nowTs(),
    domId,
    domTitle: dom.title,
    sector: dom.sector,
    stream: dom.stream,
    isAnom,
    score: r3(score),
    conf: r3(conf),
    benev: r3(benev),
    degree,
    sevKey,
    branch,
    // dossier
    type: isAnom ? pick(dom.types) : "Nominal · within learned bounds",
    location: pick(dom.locs),
    origin: isAnom ? pick(dom.origins) : "—",
    size: sizeOf(dom, score),
    dims: dom.feats.length,
    seq: 1000 + Math.floor(rand(0, 8999)),
    comp,
    feats,
    domTop,
    counterfactual: {
      f: top.f,
      obs: top.obs,
      bound: r3(top.obs * rand(0.12, 0.3))
    },
    action: ACTIONS[dom.id][sevKey],
    tlp: TLP_FOR[sevKey]
  };
}
Object.assign(window, {
  DOMAINS,
  DOMAIN_BY_ID,
  LADDERS,
  makeDetection,
  sizeOf,
  rand,
  pad,
  fmt
});
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/cli/Stream.jsx", error: String((e && e.message) || e) }); }

// ui_kits/console/Analytics.jsx
try { (() => {
/* ============================================================================
   Mercury Console — ANALYTICS: live model-evaluation instrumentation.
   The console is built on ROC-AUC; this page draws the canonical evaluation
   curves, computed live (not stored) from each domain's adjudicated window:
     · ROC curve     — threshold sweep, trapezoidal AUC, Youden J* optimum
     · PR curve       — precision/recall, average precision (AP)
     · Calibration    — reliability diagram + Expected Calibration Error (ECE)
     · Score dist     — per-class histograms + Gaussian fit + separability d′
   Interconnected: a shared decision threshold τ (same one the Detections page
   uses) drives the operating point on every curve; the domain filter scopes the
   evaluation set; "Snap to J*" sets τ from the ROC-optimal point.
   ============================================================================ */
const {
  useMemo: useMemoA,
  useRef: useRefA,
  useEffect: useEffectA
} = React;
function cssVar(v) {
  return getComputedStyle(document.documentElement).getPropertyValue(v).trim() || "#888";
}

/* ---------- computations (all O(n log n) or better over the eval window) ---------- */
function confusion(set, tau) {
  let tp = 0,
    fp = 0,
    fn = 0,
    tn = 0;
  for (const d of set) {
    const flag = d.score >= tau;
    if (flag && d.isAnom) tp++;else if (flag && !d.isAnom) fp++;else if (!flag && d.isAnom) fn++;else tn++;
  }
  const tpr = tp + fn ? tp / (tp + fn) : 0;
  const fpr = fp + tn ? fp / (fp + tn) : 0;
  const precision = tp + fp ? tp / (tp + fp) : 0;
  const recall = tpr;
  const f1 = precision + recall ? 2 * precision * recall / (precision + recall) : 0;
  return {
    tp,
    fp,
    fn,
    tn,
    tpr,
    fpr,
    precision,
    recall,
    f1,
    spec: 1 - fpr
  };
}
function rocCurve(set) {
  const pos = set.filter(d => d.isAnom).length,
    neg = set.length - pos;
  if (!pos || !neg) return null;
  const sorted = set.slice().sort((a, b) => b.score - a.score);
  let tp = 0,
    fp = 0;
  const pts = [[0, 0]];
  for (const d of sorted) {
    if (d.isAnom) tp++;else fp++;
    pts.push([fp / neg, tp / pos]);
  }
  let auc = 0;
  for (let i = 1; i < pts.length; i++) auc += (pts[i][0] - pts[i - 1][0]) * (pts[i][1] + pts[i - 1][1]) / 2;
  return {
    pts,
    auc,
    pos,
    neg
  };
}

/* Youden's J* optimum over a threshold sweep (max TPR − FPR). */
function youden(set) {
  let best = -1,
    bt = 0.5,
    bf = 0,
    btp = 0;
  for (let tau = 0; tau <= 1.0001; tau += 0.01) {
    const c = confusion(set, tau);
    const j = c.tpr - c.fpr;
    if (j > best) {
      best = j;
      bt = tau;
      bf = c.fpr;
      btp = c.tpr;
    }
  }
  return {
    j: best,
    tau: bt,
    fpr: bf,
    tpr: btp
  };
}
function prCurve(set) {
  const pos = set.filter(d => d.isAnom).length;
  if (!pos) return null;
  const sorted = set.slice().sort((a, b) => b.score - a.score);
  let tp = 0,
    fp = 0,
    ap = 0,
    prevRec = 0;
  const pts = [];
  for (const d of sorted) {
    if (d.isAnom) tp++;else fp++;
    const prec = tp / (tp + fp),
      rec = tp / pos;
    pts.push([rec, prec]);
    ap += (rec - prevRec) * prec;
    prevRec = rec;
  }
  return {
    pts,
    ap,
    prevalence: pos / set.length
  };
}

/* Reliability diagram: treat the anomaly score as P(anomaly); bin and compare
   empirical frequency vs mean predicted. ECE = Σ (nᵦ/N)·|accᵦ − confᵦ|. */
function calibration(set, nbins) {
  const bins = Array.from({
    length: nbins
  }, () => ({
    n: 0,
    conf: 0,
    acc: 0
  }));
  for (const d of set) {
    const i = Math.min(nbins - 1, Math.floor(d.score * nbins));
    bins[i].n++;
    bins[i].conf += d.score;
    bins[i].acc += d.isAnom ? 1 : 0;
  }
  const N = set.length || 1;
  let ece = 0;
  const pts = bins.map(b => {
    if (!b.n) return null;
    const conf = b.conf / b.n,
      acc = b.acc / b.n;
    ece += b.n / N * Math.abs(acc - conf);
    return {
      conf,
      acc,
      n: b.n
    };
  });
  return {
    pts,
    ece,
    maxN: Math.max(1, ...bins.map(b => b.n))
  };
}

/* Per-class score stats + separability d′ = (μ₁−μ₀)/√(½(σ₁²+σ₀²)). */
function scoreStats(set, nbins) {
  const A = set.filter(d => d.isAnom).map(d => d.score);
  const N = set.filter(d => !d.isAnom).map(d => d.score);
  const stat = xs => {
    if (!xs.length) return {
      mu: 0,
      sd: 0.0001,
      n: 0
    };
    const mu = xs.reduce((a, b) => a + b, 0) / xs.length;
    const sd = Math.sqrt(xs.reduce((a, b) => a + (b - mu) ** 2, 0) / Math.max(1, xs.length)) || 0.0001;
    return {
      mu,
      sd,
      n: xs.length
    };
  };
  const sa = stat(A),
    sn = stat(N);
  const dprime = (sa.mu - sn.mu) / Math.sqrt(0.5 * (sa.sd ** 2 + sn.sd ** 2) || 1);
  const ha = new Array(nbins).fill(0),
    hn = new Array(nbins).fill(0);
  A.forEach(s => ha[Math.min(nbins - 1, Math.floor(s * nbins))]++);
  N.forEach(s => hn[Math.min(nbins - 1, Math.floor(s * nbins))]++);
  const maxH = Math.max(1, ...ha, ...hn);
  return {
    sa,
    sn,
    dprime,
    ha,
    hn,
    maxH
  };
}

/* ---------- canvas plumbing ---------- */
function ChartCanvas({
  draw,
  h
}) {
  const ref = useRefA(null);
  useEffectA(() => {
    const cv = ref.current;
    if (!cv) return;
    const render = () => {
      const dpr = window.devicePixelRatio || 1;
      const w = cv.clientWidth || 300,
        hh = cv.clientHeight || h || 160;
      cv.width = Math.round(w * dpr);
      cv.height = Math.round(hh * dpr);
      const ctx = cv.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, hh);
      draw(ctx, w, hh);
    };
    render();
    let ro;
    if (window.ResizeObserver) {
      ro = new ResizeObserver(render);
      ro.observe(cv);
    }
    return () => {
      if (ro) ro.disconnect();
    };
  });
  return /*#__PURE__*/React.createElement("canvas", {
    ref: ref,
    style: {
      width: "100%",
      height: (h || 160) + "px",
      display: "block"
    }
  });
}
function plotFrame(ctx, w, h, P, opts) {
  opts = opts || {};
  ctx.strokeStyle = cssVar("--border");
  ctx.lineWidth = 1;
  ctx.strokeRect(P, P, w - P - 6, h - P - P);
  // light grid at 0.25 / 0.5 / 0.75
  ctx.strokeStyle = cssVar("--grid-line");
  ctx.globalAlpha = 1;
  ctx.lineWidth = 1;
  [0.25, 0.5, 0.75].forEach(g => {
    const x = P + g * (w - P - 6),
      y = h - P - g * (h - P - P);
    ctx.beginPath();
    ctx.moveTo(x, P);
    ctx.lineTo(x, h - P);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(P, y);
    ctx.lineTo(w - 6, y);
    ctx.stroke();
  });
}

/* ---------- the page ---------- */
function AnalyticsView({
  evalSet,
  tau,
  setTau,
  activeDomain,
  onPickDomain
}) {
  const set = evalSet || [];
  const roc = useMemoA(() => rocCurve(set), [set]);
  const pr = useMemoA(() => prCurve(set), [set]);
  const cal = useMemoA(() => calibration(set, 10), [set]);
  const sd = useMemoA(() => scoreStats(set, 24), [set]);
  const jstar = useMemoA(() => set.length ? youden(set) : null, [set]);
  const cm = useMemoA(() => confusion(set, tau), [set, tau]);
  const N = set.length;
  const pos = roc ? roc.pos : 0;
  const prevalence = N ? pos / N : 0;
  const bench = window.BENCH.auc;
  const auc = roc ? roc.auc : null;
  const aucDelta = auc != null ? auc - bench : null;

  /* ---- draws ---- */
  const drawROC = (ctx, w, h) => {
    const P = 26;
    const X = x => P + x * (w - P - 6),
      Y = y => h - P - y * (h - P - P);
    plotFrame(ctx, w, h, P);
    // chance diagonal
    ctx.strokeStyle = cssVar("--fg-muted");
    ctx.globalAlpha = .5;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(X(0), Y(0));
    ctx.lineTo(X(1), Y(1));
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
    if (!roc) {
      ctx.fillStyle = cssVar("--fg-muted");
      ctx.font = "10px monospace";
      ctx.fillText("need both classes — RUN DETECTION", P + 8, h / 2);
      return;
    }
    // AUC fill
    ctx.beginPath();
    ctx.moveTo(X(0), Y(0));
    roc.pts.forEach(p => ctx.lineTo(X(p[0]), Y(p[1])));
    ctx.lineTo(X(1), Y(0));
    ctx.closePath();
    ctx.fillStyle = cssVar("--mercury-500");
    ctx.globalAlpha = .12;
    ctx.fill();
    ctx.globalAlpha = 1;
    // curve
    ctx.strokeStyle = cssVar("--mercury-300");
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    roc.pts.forEach((p, i) => {
      const x = X(p[0]),
        y = Y(p[1]);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
    // Youden optimal (gold ring)
    if (jstar) {
      ctx.strokeStyle = cssVar("--benevolence");
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(X(jstar.fpr), Y(jstar.tpr), 5, 0, 7);
      ctx.stroke();
    }
    // operating point at τ (cyan dot + guides)
    ctx.strokeStyle = cssVar("--border-strong");
    ctx.lineWidth = .8;
    ctx.setLineDash([2, 2]);
    ctx.beginPath();
    ctx.moveTo(X(0), Y(cm.tpr));
    ctx.lineTo(X(cm.fpr), Y(cm.tpr));
    ctx.lineTo(X(cm.fpr), Y(0));
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = cssVar("--cognitive-soft");
    ctx.beginPath();
    ctx.arc(X(cm.fpr), Y(cm.tpr), 4.2, 0, 7);
    ctx.fill();
    ctx.strokeStyle = cssVar("--bg");
    ctx.lineWidth = 1.5;
    ctx.stroke();
    // axis labels
    ctx.fillStyle = cssVar("--fg-muted");
    ctx.font = "8px monospace";
    ctx.fillText("FPR →", w - 44, h - P + 14);
    ctx.save();
    ctx.translate(10, P + 26);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("TPR →", 0, 0);
    ctx.restore();
  };
  const drawPR = (ctx, w, h) => {
    const P = 26;
    const X = x => P + x * (w - P - 6),
      Y = y => h - P - y * (h - P - P);
    plotFrame(ctx, w, h, P);
    // prevalence baseline (random classifier)
    if (pr) {
      ctx.strokeStyle = cssVar("--warning");
      ctx.globalAlpha = .55;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(X(0), Y(pr.prevalence));
      ctx.lineTo(X(1), Y(pr.prevalence));
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
    }
    if (!pr) {
      ctx.fillStyle = cssVar("--fg-muted");
      ctx.font = "10px monospace";
      ctx.fillText("no positives in window", P + 8, h / 2);
      return;
    }
    ctx.strokeStyle = cssVar("--mercury-300");
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    pr.pts.forEach((p, i) => {
      const x = X(p[0]),
        y = Y(p[1]);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
    // operating point (recall, precision) at τ
    ctx.fillStyle = cssVar("--cognitive-soft");
    ctx.beginPath();
    ctx.arc(X(cm.recall), Y(cm.precision), 4.2, 0, 7);
    ctx.fill();
    ctx.strokeStyle = cssVar("--bg");
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.fillStyle = cssVar("--fg-muted");
    ctx.font = "8px monospace";
    ctx.fillText("Recall →", w - 52, h - P + 14);
    ctx.save();
    ctx.translate(10, P + 34);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Precision →", 0, 0);
    ctx.restore();
  };
  const drawCal = (ctx, w, h) => {
    const P = 26;
    const X = x => P + x * (w - P - 6),
      Y = y => h - P - y * (h - P - P);
    plotFrame(ctx, w, h, P);
    // perfect-calibration diagonal
    ctx.strokeStyle = cssVar("--benevolence");
    ctx.globalAlpha = .5;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(X(0), Y(0));
    ctx.lineTo(X(1), Y(1));
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
    // gap bars + reliability points
    const pts = cal.pts.filter(Boolean);
    pts.forEach(b => {
      // gap shading from diagonal to point
      ctx.strokeStyle = cssVar("--anomaly");
      ctx.globalAlpha = .25;
      ctx.lineWidth = Math.max(2, b.n / cal.maxN * 10);
      ctx.beginPath();
      ctx.moveTo(X(b.conf), Y(b.conf));
      ctx.lineTo(X(b.conf), Y(b.acc));
      ctx.stroke();
      ctx.globalAlpha = 1;
    });
    ctx.strokeStyle = cssVar("--mercury-300");
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    pts.forEach((b, i) => {
      const x = X(b.conf),
        y = Y(b.acc);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
    ctx.fillStyle = cssVar("--mercury-200");
    pts.forEach(b => {
      ctx.beginPath();
      ctx.arc(X(b.conf), Y(b.acc), 2.6, 0, 7);
      ctx.fill();
    });
    ctx.fillStyle = cssVar("--fg-muted");
    ctx.font = "8px monospace";
    ctx.fillText("pred →", w - 42, h - P + 14);
    ctx.save();
    ctx.translate(10, P + 34);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("empirical →", 0, 0);
    ctx.restore();
  };
  const drawDist = (ctx, w, h) => {
    const P = 22;
    const plotW = w - P - 6,
      plotH = h - P - P;
    const X = x => P + x * plotW,
      Yb = frac => h - P - frac * plotH;
    plotFrame(ctx, w, h, P);
    const nb = sd.ha.length;
    const bw = plotW / nb;
    // histograms (normal under, anomaly over) as light bars
    for (let i = 0; i < nb; i++) {
      const x = P + i * bw;
      const hn = sd.hn[i] / sd.maxH * plotH,
        ha = sd.ha[i] / sd.maxH * plotH;
      ctx.fillStyle = cssVar("--mercury-700");
      ctx.globalAlpha = .5;
      ctx.fillRect(x, h - P - hn, Math.max(1, bw - 1), hn);
      ctx.fillStyle = cssVar("--anomaly");
      ctx.globalAlpha = .42;
      ctx.fillRect(x, h - P - ha, Math.max(1, bw - 1), ha);
    }
    ctx.globalAlpha = 1;
    // gaussian fits
    const gauss = (mu, sd2, x) => Math.exp(-((x - mu) ** 2) / (2 * sd2 * sd2));
    const drawG = (mu, sdv, col) => {
      ctx.strokeStyle = col;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      for (let px = 0; px <= 60; px++) {
        const xv = px / 60;
        const y = gauss(mu, sdv, xv);
        const X2 = X(xv),
          Y2 = Yb(y);
        px ? ctx.lineTo(X2, Y2) : ctx.moveTo(X2, Y2);
      }
      ctx.stroke();
    };
    if (sd.sn.n) drawG(sd.sn.mu, sd.sn.sd, cssVar("--mercury-300"));
    if (sd.sa.n) drawG(sd.sa.mu, sd.sa.sd, cssVar("--anomaly-soft"));
    // τ line
    ctx.strokeStyle = cssVar("--benevolence");
    ctx.lineWidth = 1.4;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(X(tau), P);
    ctx.lineTo(X(tau), h - P);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = cssVar("--benevolence");
    ctx.font = "8px monospace";
    ctx.fillText("τ " + tau.toFixed(2), X(tau) + 3, P + 9);
  };
  const domChips = [{
    id: null,
    label: "All domains"
  }].concat(window.DOMAINS.map(d => ({
    id: d.id,
    label: d.label
  })));
  return /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("div", {
    className: "cn-an-metrics"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-an-m lead"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-an-mk"
  }, "ROC-AUC \xB7 live"), /*#__PURE__*/React.createElement("div", {
    className: "cn-an-mv cy"
  }, auc == null ? "—" : auc.toFixed(4)), /*#__PURE__*/React.createElement("div", {
    className: "cn-an-msub"
  }, aucDelta == null ? "need both classes" : (aucDelta >= 0 ? "▲ +" : "▼ ") + Math.abs(aucDelta).toFixed(4) + " vs bench")), /*#__PURE__*/React.createElement("div", {
    className: "cn-an-m"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-an-mk"
  }, "Average Precision"), /*#__PURE__*/React.createElement("div", {
    className: "cn-an-mv"
  }, pr ? pr.ap.toFixed(4) : "—"), /*#__PURE__*/React.createElement("div", {
    className: "cn-an-msub"
  }, "PR area")), /*#__PURE__*/React.createElement("div", {
    className: "cn-an-m"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-an-mk"
  }, "Calibration ECE"), /*#__PURE__*/React.createElement("div", {
    className: "cn-an-mv " + (cal.ece > 0.1 ? "warn" : "")
  }, N ? cal.ece.toFixed(4) : "—"), /*#__PURE__*/React.createElement("div", {
    className: "cn-an-msub"
  }, "lower = better")), /*#__PURE__*/React.createElement("div", {
    className: "cn-an-m"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-an-mk"
  }, "Separability d\u2032"), /*#__PURE__*/React.createElement("div", {
    className: "cn-an-mv"
  }, N ? sd.dprime.toFixed(3) : "—"), /*#__PURE__*/React.createElement("div", {
    className: "cn-an-msub"
  }, "class distance")), /*#__PURE__*/React.createElement("div", {
    className: "cn-an-m"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-an-mk"
  }, "Eval window"), /*#__PURE__*/React.createElement("div", {
    className: "cn-an-mv"
  }, N), /*#__PURE__*/React.createElement("div", {
    className: "cn-an-msub"
  }, (prevalence * 100).toFixed(0), "% anomalous"))), /*#__PURE__*/React.createElement("div", {
    className: "cn-an-controls"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-an-tau"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-an-tau-top"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-an-tau-k"
  }, "DECISION THRESHOLD \xB7 \u03C4"), /*#__PURE__*/React.createElement("span", {
    className: "cn-an-tau-v"
  }, tau.toFixed(2)), jstar && /*#__PURE__*/React.createElement("button", {
    className: "cn-an-snap",
    onClick: () => setTau(+jstar.tau.toFixed(2)),
    title: "set \u03C4 to the Youden-optimal point"
  }, "\u25CE Snap to J* (", jstar.tau.toFixed(2), ")")), /*#__PURE__*/React.createElement("input", {
    className: "cn-an-range",
    type: "range",
    min: "0",
    max: "1",
    step: "0.01",
    value: tau,
    onChange: e => setTau(+e.target.value)
  }), /*#__PURE__*/React.createElement("div", {
    className: "cn-an-opline"
  }, "operating point \xB7 TPR ", /*#__PURE__*/React.createElement("b", null, cm.tpr.toFixed(3)), " \xB7 FPR ", /*#__PURE__*/React.createElement("b", null, cm.fpr.toFixed(3)), " \xB7 precision ", /*#__PURE__*/React.createElement("b", null, cm.precision.toFixed(3)), " \xB7 F1 ", /*#__PURE__*/React.createElement("b", null, cm.f1.toFixed(3)), " \xB7 this \u03C4 is shared with Detections")), /*#__PURE__*/React.createElement("div", {
    className: "cn-an-domfilter"
  }, domChips.map(c => /*#__PURE__*/React.createElement("button", {
    key: c.id || "all",
    className: "cn-an-domchip" + ((activeDomain || null) === c.id ? " on" : ""),
    onClick: () => {
      if (c.id == null) {
        if (activeDomain) onPickDomain(activeDomain);
      } else if (activeDomain !== c.id) onPickDomain(c.id);
    }
  }, c.label)))), /*#__PURE__*/React.createElement("div", {
    className: "cn-an-grid"
  }, /*#__PURE__*/React.createElement("section", {
    className: "cn-panel cn-chart"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-panel-head"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-lab"
  }, "ROC CURVE"), /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-meta"
  }, "AUC ", auc == null ? "—" : auc.toFixed(3), " \xB7 J* ", jstar ? jstar.j.toFixed(3) : "—")), /*#__PURE__*/React.createElement("div", {
    className: "cn-chart-canvas"
  }, /*#__PURE__*/React.createElement(ChartCanvas, {
    h: 172,
    draw: drawROC
  })), /*#__PURE__*/React.createElement("div", {
    className: "cn-chart-legend"
  }, /*#__PURE__*/React.createElement("span", null, /*#__PURE__*/React.createElement("i", {
    className: "cn-lg-line cy"
  }), "ROC"), /*#__PURE__*/React.createElement("span", null, /*#__PURE__*/React.createElement("i", {
    className: "cn-lg-dot vi"
  }), "\u03C4 operating"), /*#__PURE__*/React.createElement("span", null, /*#__PURE__*/React.createElement("i", {
    className: "cn-lg-ring gd"
  }), "Youden J*"), /*#__PURE__*/React.createElement("span", null, /*#__PURE__*/React.createElement("i", {
    className: "cn-lg-line dash"
  }), "chance"))), /*#__PURE__*/React.createElement("section", {
    className: "cn-panel cn-chart"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-panel-head"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-lab"
  }, "PRECISION\u2013RECALL"), /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-meta"
  }, "AP ", pr ? pr.ap.toFixed(3) : "—", " \xB7 prevalence ", (prevalence * 100).toFixed(0), "%")), /*#__PURE__*/React.createElement("div", {
    className: "cn-chart-canvas"
  }, /*#__PURE__*/React.createElement(ChartCanvas, {
    h: 172,
    draw: drawPR
  })), /*#__PURE__*/React.createElement("div", {
    className: "cn-chart-legend"
  }, /*#__PURE__*/React.createElement("span", null, /*#__PURE__*/React.createElement("i", {
    className: "cn-lg-line cy"
  }), "PR"), /*#__PURE__*/React.createElement("span", null, /*#__PURE__*/React.createElement("i", {
    className: "cn-lg-dot vi"
  }), "\u03C4 operating"), /*#__PURE__*/React.createElement("span", null, /*#__PURE__*/React.createElement("i", {
    className: "cn-lg-line warn-dash"
  }), "random = prevalence"))), /*#__PURE__*/React.createElement("section", {
    className: "cn-panel cn-chart"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-panel-head"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-lab"
  }, "CALIBRATION \xB7 RELIABILITY"), /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-meta"
  }, "ECE ", N ? cal.ece.toFixed(3) : "—")), /*#__PURE__*/React.createElement("div", {
    className: "cn-chart-canvas"
  }, /*#__PURE__*/React.createElement(ChartCanvas, {
    h: 172,
    draw: drawCal
  })), /*#__PURE__*/React.createElement("div", {
    className: "cn-chart-legend"
  }, /*#__PURE__*/React.createElement("span", null, /*#__PURE__*/React.createElement("i", {
    className: "cn-lg-line cy"
  }), "observed"), /*#__PURE__*/React.createElement("span", null, /*#__PURE__*/React.createElement("i", {
    className: "cn-lg-line gd-dash"
  }), "perfect"), /*#__PURE__*/React.createElement("span", null, /*#__PURE__*/React.createElement("i", {
    className: "cn-lg-bar anom"
  }), "gap (ECE)"))), /*#__PURE__*/React.createElement("section", {
    className: "cn-panel cn-chart"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-panel-head"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-lab"
  }, "SCORE DISTRIBUTION \xB7 d\u2032 ", N ? sd.dprime.toFixed(2) : "—"), /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-meta"
  }, "normal vs anomaly")), /*#__PURE__*/React.createElement("div", {
    className: "cn-chart-canvas"
  }, /*#__PURE__*/React.createElement(ChartCanvas, {
    h: 172,
    draw: drawDist
  })), /*#__PURE__*/React.createElement("div", {
    className: "cn-chart-legend"
  }, /*#__PURE__*/React.createElement("span", null, /*#__PURE__*/React.createElement("i", {
    className: "cn-lg-bar cy"
  }), "normal"), /*#__PURE__*/React.createElement("span", null, /*#__PURE__*/React.createElement("i", {
    className: "cn-lg-bar anom"
  }), "anomaly"), /*#__PURE__*/React.createElement("span", null, /*#__PURE__*/React.createElement("i", {
    className: "cn-lg-line gd-dash"
  }), "\u03C4")))), /*#__PURE__*/React.createElement("section", {
    className: "cn-panel cn-an-confusion"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-panel-head"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-lab"
  }, "CONFUSION AT \u03C4 ", tau.toFixed(2)), /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-meta"
  }, cm.tp + cm.fp, " flagged / ", N, " events", activeDomain ? " · " + window.DOMAINS.find(d => d.id === activeDomain).label : " · fleet")), /*#__PURE__*/React.createElement("div", {
    className: "cn-an-cm-wrap"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-an-cm"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-an-cm-cell tp"
  }, /*#__PURE__*/React.createElement("span", {
    className: "n"
  }, cm.tp), /*#__PURE__*/React.createElement("span", {
    className: "l"
  }, "TP")), /*#__PURE__*/React.createElement("div", {
    className: "cn-an-cm-cell fp"
  }, /*#__PURE__*/React.createElement("span", {
    className: "n"
  }, cm.fp), /*#__PURE__*/React.createElement("span", {
    className: "l"
  }, "FP")), /*#__PURE__*/React.createElement("div", {
    className: "cn-an-cm-cell fn"
  }, /*#__PURE__*/React.createElement("span", {
    className: "n"
  }, cm.fn), /*#__PURE__*/React.createElement("span", {
    className: "l"
  }, "FN")), /*#__PURE__*/React.createElement("div", {
    className: "cn-an-cm-cell tn"
  }, /*#__PURE__*/React.createElement("span", {
    className: "n"
  }, cm.tn), /*#__PURE__*/React.createElement("span", {
    className: "l"
  }, "TN"))), /*#__PURE__*/React.createElement("div", {
    className: "cn-an-cm-metrics"
  }, [["Precision", cm.precision], ["Recall / TPR", cm.recall], ["Specificity", cm.spec], ["F1", cm.f1], ["FPR", cm.fpr]].map(([k, v]) => /*#__PURE__*/React.createElement("div", {
    className: "cn-an-cmm",
    key: k
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-an-cmm-k"
  }, k), /*#__PURE__*/React.createElement("span", {
    className: "cn-an-cmm-track"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-an-cmm-fill",
    style: {
      width: v * 100 + "%"
    }
  })), /*#__PURE__*/React.createElement("span", {
    className: "cn-an-cmm-v"
  }, v.toFixed(3)))))), /*#__PURE__*/React.createElement("div", {
    className: "cn-an-foot"
  }, "Curves are computed live from each domain's adjudicated rolling window via threshold sweep \u2014 trapezoidal AUC, average precision, and ECE recompute as the feed streams. ", /*#__PURE__*/React.createElement("b", null, "RUN DETECTION"), " to evolve them; drag ", /*#__PURE__*/React.createElement("b", null, "\u03C4"), " to move every operating point in lock-step.")));
}
Object.assign(window, {
  AnalyticsView
});
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/console/Analytics.jsx", error: String((e && e.message) || e) }); }

// ui_kits/console/Capitalize.jsx
try { (() => {
/* ============================================================================
   Mercury Console — Phase 3 "Capitalize" surfaces. Each binds to a REAL but
   previously-unexposed router in the repository:
     · ModelsRegistry — api/routes/models.py   (register · deploy · capability)
     · ExportView     — api/routes/export.py    (+ GDPR Art. 22 decision dossier)
     · OverrideAudit  — api/routes/export.py audit_logs (human-in-the-loop)
     · VoicePanel     — api/voice.py            (narration, with σ_Immutable caveat)
   All call window.MercuryAPI (live when a server is set, faithful mock otherwise)
   and show which path served the response.
   ============================================================================ */
const {
  useState: useStateC,
  useEffect: useEffectC,
  useRef: useRefC
} = React;
function useSrc() {
  return useApiState();
} // reuse the shared API-state hook

function SrcTag({
  res
}) {
  if (!res) return null;
  return /*#__PURE__*/React.createElement("span", {
    className: "cn-srcbadge " + res.source
  }, res.source === "live" ? "● LIVE" : "○ MOCK");
}

/* ---------------- N1 · ENGINE REGISTRY (/api/v1/models) ---------------- */
const MODEL_STATUS = {
  deployed: {
    cls: "ok",
    label: "DEPLOYED"
  },
  staged: {
    cls: "stage",
    label: "STAGED"
  },
  draft: {
    cls: "draft",
    label: "DRAFT"
  },
  deprecated: {
    cls: "dep",
    label: "DEPRECATED"
  },
  archived: {
    cls: "arc",
    label: "ARCHIVED"
  }
};
const CAPABILITY = [{
  ep: "/detect/univariate",
  methods: ["zscore", "iqr", "isolation_forest"],
  scale: "σ · z-score"
}, {
  ep: "/detect/multivariate",
  methods: ["correlate_features"],
  scale: "L2 / Mahalanobis"
}, {
  ep: "/detect/fusion",
  methods: ["statistical", "temporal", "dimensional", "spectral"],
  scale: "fused"
}, {
  ep: "/detect/neurosymbolic",
  methods: ["weighted_average", "attention", "gated", "confidence_weighted", "hierarchical"],
  scale: "0–1"
}, {
  ep: "/detect/three-r",
  methods: ["recursion", "resonance", "optimization"],
  scale: "0–1"
}];
function ModelsRegistry() {
  const [res, setRes] = useStateC(null);
  const [busy, setBusy] = useStateC(false);
  const [newName, setNewName] = useStateC("");
  const [newType, setNewType] = useStateC("fusion");
  const load = async () => setRes(await window.MercuryAPI.modelsList());
  useEffectC(() => {
    load();
  }, []);
  const deploy = async m => {
    setBusy(true);
    await window.MercuryAPI.modelDeploy(m.model_id, m.current_version || "1.0.0", {
      replicas: 2
    });
    window.MercuryAPI.auditAppend({
      actor: "analyst@steel-sa",
      action: "deploy",
      target: m.model_id,
      justification: "promoted " + (m.current_version || "draft") + " via Engine Registry"
    });
    await load();
    setBusy(false);
  };
  const register = async () => {
    if (!newName.trim()) return;
    setBusy(true);
    await window.MercuryAPI.modelRegister({
      name: newName.trim(),
      model_type: newType
    });
    setNewName("");
    await load();
    setBusy(false);
  };
  const models = res && res.data ? res.data : [];
  return /*#__PURE__*/React.createElement("div", {
    className: "cn-models"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-mdl-legend"
  }, /*#__PURE__*/React.createElement("span", null, /*#__PURE__*/React.createElement("b", null, "Engine Registry"), " \u2014 bound to ", /*#__PURE__*/React.createElement("code", null, "/api/v1/models"), ". ACTIVE engines are ", /*#__PURE__*/React.createElement("b", null, "DEPLOYED"), "; the capability grid is the real method surface each endpoint accepts. Register and deploy write to the registry + ", /*#__PURE__*/React.createElement("code", null, "audit_trail"), "."), /*#__PURE__*/React.createElement(SrcTag, {
    res: res
  })), /*#__PURE__*/React.createElement("section", {
    className: "cn-panel"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-panel-head"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-lab"
  }, "REGISTRY \xB7 ", models.length, " ENGINES"), /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-meta"
  }, "GET /api/v1/models")), /*#__PURE__*/React.createElement("div", {
    className: "cn-mdl-head2"
  }, /*#__PURE__*/React.createElement("span", null, "engine"), /*#__PURE__*/React.createElement("span", null, "type"), /*#__PURE__*/React.createElement("span", null, "fw"), /*#__PURE__*/React.createElement("span", {
    className: "r"
  }, "AUC"), /*#__PURE__*/React.createElement("span", {
    className: "r"
  }, "F1"), /*#__PURE__*/React.createElement("span", {
    className: "r"
  }, "p50 / p99"), /*#__PURE__*/React.createElement("span", {
    className: "r"
  }, "rps"), /*#__PURE__*/React.createElement("span", null, "status")), /*#__PURE__*/React.createElement("div", null, models.map(m => /*#__PURE__*/React.createElement("div", {
    className: "cn-mdl-row2" + (m.status === "deployed" ? " active" : ""),
    key: m.model_id
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-mdl-name"
  }, m.name, /*#__PURE__*/React.createElement("span", {
    className: "cn-mdl-ep"
  }, m.endpoint || "— unbound")), /*#__PURE__*/React.createElement("span", {
    className: "cn-mdl-type"
  }, m.model_type), /*#__PURE__*/React.createElement("span", {
    className: "cn-mdl-fw"
  }, m.framework), /*#__PURE__*/React.createElement("span", {
    className: "r cn-num"
  }, m.metrics.auc_roc ? m.metrics.auc_roc.toFixed(3) : "—"), /*#__PURE__*/React.createElement("span", {
    className: "r cn-num"
  }, m.metrics.f1_score ? m.metrics.f1_score.toFixed(2) : "—"), /*#__PURE__*/React.createElement("span", {
    className: "r cn-num"
  }, m.metrics.latency_p50_ms || "—", "/", m.metrics.latency_p99_ms || "—", /*#__PURE__*/React.createElement("span", {
    className: "cn-mdl-unit"
  }, "ms")), /*#__PURE__*/React.createElement("span", {
    className: "r cn-num"
  }, m.metrics.throughput_rps || "—"), /*#__PURE__*/React.createElement("span", {
    className: "cn-mdl-statuscell"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-mdl-status " + MODEL_STATUS[m.status].cls
  }, MODEL_STATUS[m.status].label), m.status !== "deployed" && m.status !== "archived" && /*#__PURE__*/React.createElement("button", {
    className: "cn-mdl-deploy",
    disabled: busy,
    onClick: () => deploy(m),
    title: "PUT /api/v1/models/{id}/deploy"
  }, "\u25B8 deploy"))))), /*#__PURE__*/React.createElement("div", {
    className: "cn-mdl-register"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-seg-k"
  }, "register"), /*#__PURE__*/React.createElement("input", {
    className: "cn-input sm",
    placeholder: "engine name",
    value: newName,
    onChange: e => setNewName(e.target.value)
  }), /*#__PURE__*/React.createElement("select", {
    className: "cn-select",
    value: newType,
    onChange: e => setNewType(e.target.value)
  }, ["fusion", "statistical", "temporal", "neurosymbolic", "lstm_autoencoder", "isolation_forest", "custom"].map(t => /*#__PURE__*/React.createElement("option", {
    key: t,
    value: t
  }, t))), /*#__PURE__*/React.createElement("button", {
    className: "cn-mdl-reg-btn",
    disabled: busy || !newName.trim(),
    onClick: register
  }, "+ register draft"))), /*#__PURE__*/React.createElement("section", {
    className: "cn-panel"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-panel-head"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-lab"
  }, "METHOD CAPABILITY GRID"), /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-meta"
  }, "what each endpoint accepts")), /*#__PURE__*/React.createElement("div", {
    className: "cn-cap"
  }, CAPABILITY.map(c => /*#__PURE__*/React.createElement("div", {
    className: "cn-cap-row",
    key: c.ep
  }, /*#__PURE__*/React.createElement("code", {
    className: "cn-cap-ep"
  }, c.ep), /*#__PURE__*/React.createElement("span", {
    className: "cn-cap-scale"
  }, c.scale), /*#__PURE__*/React.createElement("span", {
    className: "cn-cap-methods"
  }, c.methods.map(mm => /*#__PURE__*/React.createElement("span", {
    className: "cn-cap-m",
    key: mm
  }, mm))))))), (window.ENGINES || []).map(grp => /*#__PURE__*/React.createElement("section", {
    className: "cn-panel cn-modelgrp",
    key: grp.group
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-panel-head"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-lab"
  }, "DETECTOR CATALOG \xB7 ", grp.group.toUpperCase()), /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-meta"
  }, "src/omni_mercury_engine/detectors \xB7 ", grp.items.length)), /*#__PURE__*/React.createElement("div", {
    className: "cn-mdl-head"
  }, /*#__PURE__*/React.createElement("span", null, "class"), /*#__PURE__*/React.createElement("span", null, "cost"), /*#__PURE__*/React.createElement("span", null, "training"), /*#__PURE__*/React.createElement("span", null, "mode"), /*#__PURE__*/React.createElement("span", null, "file")), /*#__PURE__*/React.createElement("div", {
    className: "cn-modellist"
  }, grp.items.map(m => /*#__PURE__*/React.createElement("div", {
    className: "cn-modelrow" + (m.active ? " active" : ""),
    key: m.cls
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-model-cls"
  }, m.cls, m.active && /*#__PURE__*/React.createElement("span", {
    className: "cn-mdl-active"
  }, "ACTIVE")), /*#__PURE__*/React.createElement("span", {
    className: "cn-mdl-cx"
  }, m.cx), /*#__PURE__*/React.createElement("span", null, m.train === "free" ? /*#__PURE__*/React.createElement("span", {
    className: "cn-mdl-tag free"
  }, "training-free") : /*#__PURE__*/React.createElement("span", {
    className: "cn-mdl-tag learned"
  }, "learned")), /*#__PURE__*/React.createElement("span", {
    className: "cn-mdl-mode " + m.mode
  }, m.mode), /*#__PURE__*/React.createElement("code", {
    className: "cn-model-file"
  }, m.file)))))));
}

/* ---------------- N2 · DATA EXPORT + GDPR Art. 22 dossier (/api/v1/export) ---------------- */
function buildDossier(det) {
  const corr = det && det.corr ? det.corr : "corr-" + Math.random().toString(36).slice(2, 10);
  return {
    article: "GDPR Art. 22 — explanation of an automated decision",
    generated_at: new Date().toISOString(),
    decision_id: det ? det.id : "n/a",
    correlation_id: corr,
    domain: det ? det.domain : "n/a",
    outcome: det ? {
      is_anomaly: det.isAnom,
      score: det.score,
      scale: det.scale || "ensemble [0,1]",
      severity: det.sev
    } : null,
    ethical_gate: det ? {
      benevolence: det.benev,
      threshold: 0.99,
      verdict: det.withheld ? "WITHHELD" : det.benev == null ? "n/a (detect endpoint)" : "CLEARED",
      sigma_immutable: "enforced",
      advisory_mode: false
    } : null,
    explanation: det ? {
      method: det.method || det.endpoint || "ensemble",
      feature_contributions: (det.feats || []).map(f => ({
        feature: f.f,
        weight: f.w
      })),
      counterfactual: det.counterfactual || null
    } : null,
    audit_trail: window.MercuryAPI.auditList().slice(0, 6),
    provenance: {
      source: det && det.source ? det.source : "feed",
      served_via: "POST /api/v1/export",
      note: "Research-grade · community-tested · not externally audited."
    }
  };
}
function download(name, obj) {
  const blob = new Blob([JSON.stringify(obj, null, 2)], {
    type: "application/json"
  });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
function ExportView({
  detail
}) {
  const [summary, setSummary] = useStateC(null);
  const [type, setType] = useStateC("detections");
  const [format, setFormat] = useStateC("json");
  const [limit, setLimit] = useStateC(10000);
  const [job, setJob] = useStateC(null);
  const [busy, setBusy] = useStateC(false);
  const pollRef = useRefC(null);
  useEffectC(() => {
    (async () => setSummary(await window.MercuryAPI.exportSummary()))();
  }, []);
  useEffectC(() => () => clearInterval(pollRef.current), []);
  const submit = async () => {
    setBusy(true);
    const r = await window.MercuryAPI.exportSubmit({
      export_type: type,
      format,
      limit: +limit
    });
    const id = r.data.job_id;
    setJob(r.data);
    clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      const s = await window.MercuryAPI.exportStatus(id);
      setJob(s.data);
      if (s.data && (s.data.status === "COMPLETED" || s.data.status === "FAILED")) {
        clearInterval(pollRef.current);
        setBusy(false);
      }
    }, 900);
  };
  const s = summary && summary.data ? summary.data : null;
  const dossier = buildDossier(detail);
  return /*#__PURE__*/React.createElement("div", {
    className: "cn-grid wide"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-col-main"
  }, /*#__PURE__*/React.createElement("section", {
    className: "cn-panel"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-panel-head"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-lab"
  }, "EXPORT \xB7 BULK"), /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-meta"
  }, "POST /api/v1/export \xB7 auth required")), /*#__PURE__*/React.createElement("div", {
    className: "cn-export"
  }, s && /*#__PURE__*/React.createElement("div", {
    className: "cn-exp-summary"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-exp-s"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-exp-sv"
  }, s.total_detections.toLocaleString()), /*#__PURE__*/React.createElement("span", {
    className: "cn-exp-sk"
  }, "detections")), /*#__PURE__*/React.createElement("div", {
    className: "cn-exp-s"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-exp-sv"
  }, s.total_audit_logs.toLocaleString()), /*#__PURE__*/React.createElement("span", {
    className: "cn-exp-sk"
  }, "audit logs")), /*#__PURE__*/React.createElement("div", {
    className: "cn-exp-s"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-exp-sv"
  }, s.available_methods.length), /*#__PURE__*/React.createElement("span", {
    className: "cn-exp-sk"
  }, "methods")), /*#__PURE__*/React.createElement("div", {
    className: "cn-exp-s"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-exp-sv"
  }, s.available_actions.length), /*#__PURE__*/React.createElement("span", {
    className: "cn-exp-sk"
  }, "actions"))), /*#__PURE__*/React.createElement("div", {
    className: "cn-exp-controls"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-seg-row"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-seg-k"
  }, "type"), ["detections", "audit_logs", "metrics", "models", "jobs"].map(t => /*#__PURE__*/React.createElement("button", {
    key: t,
    className: "cn-seg-b" + (type === t ? " on" : ""),
    onClick: () => setType(t)
  }, t))), /*#__PURE__*/React.createElement("div", {
    className: "cn-seg-row"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-seg-k"
  }, "format"), ["json", "csv", "jsonl", "parquet"].map(f => /*#__PURE__*/React.createElement("button", {
    key: f,
    className: "cn-seg-b" + (format === f ? " on" : ""),
    onClick: () => setFormat(f)
  }, f))), /*#__PURE__*/React.createElement("label", {
    className: "cn-mini"
  }, /*#__PURE__*/React.createElement("span", null, "limit ", /*#__PURE__*/React.createElement("b", null, (+limit).toLocaleString()), " records"), /*#__PURE__*/React.createElement("input", {
    type: "range",
    min: "1000",
    max: "100000",
    step: "1000",
    value: limit,
    onChange: e => setLimit(+e.target.value)
  })), /*#__PURE__*/React.createElement("button", {
    className: "mc-action primary block",
    disabled: busy,
    onClick: submit
  }, busy ? "exporting…" : "▸ SUBMIT EXPORT")), job && /*#__PURE__*/React.createElement("div", {
    className: "cn-exp-job"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-exp-job-top"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-jobstate " + job.status.toLowerCase()
  }, job.status), /*#__PURE__*/React.createElement("code", null, job.job_id), /*#__PURE__*/React.createElement("span", {
    className: "cn-exp-job-meta"
  }, job.export_type, " \xB7 ", job.format)), /*#__PURE__*/React.createElement("div", {
    className: "cn-exp-job-bar"
  }, /*#__PURE__*/React.createElement("span", {
    style: {
      width: (job.status === "COMPLETED" ? 100 : job.record_count && limit ? Math.min(100, job.record_count / limit * 100) : 5) + "%"
    }
  })), /*#__PURE__*/React.createElement("div", {
    className: "cn-exp-job-foot"
  }, /*#__PURE__*/React.createElement("span", null, (job.record_count || 0).toLocaleString(), " records", job.file_size_bytes ? " · " + Math.round(job.file_size_bytes / 1024) + " KB" : ""), job.download_url && /*#__PURE__*/React.createElement("code", {
    className: "cn-exp-dl"
  }, job.download_url)))))), /*#__PURE__*/React.createElement("div", {
    className: "cn-col-side"
  }, /*#__PURE__*/React.createElement("section", {
    className: "cn-panel cn-detail"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-panel-head"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-lab"
  }, "ART. 22 \xB7 DECISION DOSSIER"), /*#__PURE__*/React.createElement("span", {
    className: "cn-detail-flag norm",
    style: {
      background: "var(--cognitive-bg)",
      color: "var(--cognitive-soft)",
      borderColor: "var(--cognitive)"
    }
  }, "GOVERNANCE")), /*#__PURE__*/React.createElement("div", {
    className: "cn-detail-body"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-art22-note"
  }, "An exportable explanation of one automated decision \u2014 outcome, ethical-gate verdict, feature attribution, correlation ID, and audit trail. Assembled from ", /*#__PURE__*/React.createElement("code", null, "explanation"), " + ", /*#__PURE__*/React.createElement("code", null, "audit_trail"), "."), /*#__PURE__*/React.createElement("div", {
    className: "cn-art22-sel"
  }, detail ? /*#__PURE__*/React.createElement(React.Fragment, null, "decision ", /*#__PURE__*/React.createElement("code", null, detail.id), " \xB7 ", detail.domain) : "no decision selected — showing a template (select a row in Detections)"), /*#__PURE__*/React.createElement("pre", {
    className: "cn-json sm"
  }, JSON.stringify(dossier, null, 2)), /*#__PURE__*/React.createElement("button", {
    className: "cn-detail-explain",
    onClick: () => download("mercury-art22-" + (detail ? detail.id : "template") + ".json", dossier)
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-sig-sm"
  }, "\u2671"), " Export decision dossier (JSON)")))));
}

/* ---------------- N4 · NARRATIVE VOICE (/api/v1/voice) ---------------- */
function VoicePanel({
  detail
}) {
  const [log, setLog] = useStateC([]);
  const [busy, setBusy] = useStateC(false);
  const inputRef = useRefC(null);
  const ask = async msg => {
    const m = (msg != null ? msg : inputRef.current.value || "").trim();
    if (!m) return;
    if (inputRef.current && msg == null) inputRef.current.value = "";
    setLog(l => l.concat({
      who: "you",
      text: m
    }));
    setBusy(true);
    const r = await window.MercuryAPI.voiceSpeak({
      message: m
    });
    setLog(l => l.concat({
      who: "mercury",
      ...r.data,
      source: r.source
    }));
    setBusy(false);
  };
  const narrate = async () => {
    if (!detail) return;
    setBusy(true);
    const r = await window.MercuryAPI.voiceNarrate({
      detection_result: detail,
      domain: detail.domain
    });
    setLog(l => l.concat({
      who: "mercury",
      message: r.data.narration,
      intent: "narration",
      confidence: r.data.confidence,
      sources: ["detection_log"],
      source: r.source
    }));
    setBusy(false);
  };
  return /*#__PURE__*/React.createElement("div", {
    className: "cn-grid wide"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-col-main"
  }, /*#__PURE__*/React.createElement("section", {
    className: "cn-panel"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-panel-head"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-lab"
  }, "MERCURY AGENT \xB7 VOICE"), /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-meta"
  }, "POST /api/v1/voice/speak")), /*#__PURE__*/React.createElement("div", {
    className: "cn-voice"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-voice-caveat"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-voice-caveat-k"
  }, "\u03C3_IMMUTABLE \xB7 WAVE C"), "The narrative voice does not yet run the \u03C3_Immutable gate (a tracked ROADMAP milestone). Detections it narrates remain gated upstream; the narration itself is ", /*#__PURE__*/React.createElement("b", null, "advisory"), "."), /*#__PURE__*/React.createElement("div", {
    className: "cn-voice-log"
  }, log.length === 0 && /*#__PURE__*/React.createElement("div", {
    className: "cn-voice-empty"
  }, "Ask Mercury about system status, recent anomalies, or the benevolence gate \u2014 or narrate the selected detection. It answers in its own honest voice: the measured number and its caveat together."), log.map((m, i) => m.who === "you" ? /*#__PURE__*/React.createElement("div", {
    className: "cn-voice-you",
    key: i
  }, m.text) : /*#__PURE__*/React.createElement("div", {
    className: "cn-voice-m",
    key: i
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-voice-m-top"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-voice-sig"
  }, "\u2671"), /*#__PURE__*/React.createElement("span", {
    className: "cn-voice-intent"
  }, m.intent), m.confidence != null && /*#__PURE__*/React.createElement("span", {
    className: "cn-voice-conf"
  }, "conf ", m.confidence.toFixed(2)), /*#__PURE__*/React.createElement(SrcTag, {
    res: {
      source: m.source
    }
  })), /*#__PURE__*/React.createElement("div", {
    className: "cn-voice-text"
  }, m.message), m.sources && m.sources.length > 0 && /*#__PURE__*/React.createElement("div", {
    className: "cn-voice-src"
  }, "\u2514\u2500 ", m.sources.join(" · "))))), /*#__PURE__*/React.createElement("div", {
    className: "cn-voice-chips"
  }, ["What is the system status?", "Any recent anomalies?", "Explain the benevolence gate", "Is it quantum-safe?"].map(q => /*#__PURE__*/React.createElement("button", {
    key: q,
    className: "cn-voice-chip",
    onClick: () => ask(q)
  }, q)), /*#__PURE__*/React.createElement("button", {
    className: "cn-voice-chip narrate",
    disabled: !detail,
    onClick: narrate,
    title: detail ? "narrate the selected detection" : "select a detection first"
  }, "\u25B8 narrate selection")), /*#__PURE__*/React.createElement("form", {
    className: "cn-voice-ask",
    onSubmit: e => {
      e.preventDefault();
      ask();
    }
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-voice-asksig"
  }, "\u2671"), /*#__PURE__*/React.createElement("input", {
    ref: inputRef,
    placeholder: "Ask Mercury Agent\u2026",
    autoComplete: "off"
  }), /*#__PURE__*/React.createElement("button", {
    type: "submit",
    disabled: busy,
    "aria-label": "send"
  }, "\u25B8"))))), /*#__PURE__*/React.createElement("div", {
    className: "cn-col-side"
  }, /*#__PURE__*/React.createElement("section", {
    className: "cn-panel"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-panel-head"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-lab"
  }, "VOICE CONTRACT")), /*#__PURE__*/React.createElement("div", {
    className: "cn-prov",
    style: {
      padding: "12px 15px"
    }
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-prov-row"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-prov-k"
  }, "speak"), /*#__PURE__*/React.createElement("code", null, "POST /api/v1/voice/speak")), /*#__PURE__*/React.createElement("div", {
    className: "cn-prov-row"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-prov-k"
  }, "narrate"), /*#__PURE__*/React.createElement("code", null, "POST /api/v1/voice/narrate")), /*#__PURE__*/React.createElement("div", {
    className: "cn-prov-row"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-prov-k"
  }, "returns"), /*#__PURE__*/React.createElement("span", {
    className: "cn-prov-v"
  }, "message \xB7 intent \xB7 confidence \xB7 sources")), /*#__PURE__*/React.createElement("div", {
    className: "cn-prov-row"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-prov-k"
  }, "\u03C3-gate"), /*#__PURE__*/React.createElement("span", {
    className: "cn-prov-v",
    style: {
      color: "var(--warning-bright)"
    }
  }, "Wave C \u2014 not yet wired"))))));
}

/* ---------------- N3 · OVERRIDE LOOP + AUDIT LEDGER (Ethics side panel) ---------------- */
function OverrideAudit({
  rows
}) {
  const [trail, setTrail] = useStateC(window.MercuryAPI.auditList());
  const [open, setOpen] = useStateC(null); // a withheld row being overridden
  const [why, setWhy] = useStateC("");
  useEffectC(() => window.MercuryAPI.onChange(() => setTrail(window.MercuryAPI.auditList())), []);
  const withheld = (rows || []).filter(r => r.withheld).slice(0, 6);
  const file = r => {
    if (!why.trim()) return;
    window.MercuryAPI.auditAppend({
      actor: "analyst@steel-sa",
      action: "override_request",
      target: r.domain,
      justification: why.trim(),
      corr: r.corr || null,
      decision_id: r.id
    });
    setWhy("");
    setOpen(null);
  };
  return /*#__PURE__*/React.createElement("section", {
    className: "cn-panel"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-panel-head"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-lab"
  }, "OVERRIDE & AUDIT TRAIL"), /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-meta"
  }, "writes audit_logs")), /*#__PURE__*/React.createElement("div", {
    className: "cn-ovr"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-ovr-note"
  }, "A withheld decision cannot be force-cleared \u2014 there is no advisory mode. An operator may file a justified ", /*#__PURE__*/React.createElement("b", null, "override request"), ", which is appended to the ", /*#__PURE__*/React.createElement("code", null, "audit_trail"), " for review, not auto-applied."), withheld.length === 0 && /*#__PURE__*/React.createElement("div", {
    className: "cn-ovr-clean"
  }, "\u2713 No withheld decisions in the current window."), withheld.map(r => /*#__PURE__*/React.createElement("div", {
    className: "cn-ovr-row",
    key: r.id
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-ovr-row-top"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-ovr-dom"
  }, r.domain), /*#__PURE__*/React.createElement("span", {
    className: "cn-num anom"
  }, "\u03B7 ", r.benev != null ? r.benev.toFixed(3) : "—"), /*#__PURE__*/React.createElement("button", {
    className: "cn-ovr-btn",
    onClick: () => setOpen(open === r.id ? null : r.id)
  }, open === r.id ? "cancel" : "file override")), open === r.id && /*#__PURE__*/React.createElement("div", {
    className: "cn-ovr-form"
  }, /*#__PURE__*/React.createElement("textarea", {
    className: "cn-textarea",
    rows: "2",
    placeholder: "Justification (required) \u2014 recorded to the audit trail with your identity.",
    value: why,
    onChange: e => setWhy(e.target.value)
  }), /*#__PURE__*/React.createElement("button", {
    className: "cn-ovr-submit",
    disabled: !why.trim(),
    onClick: () => file(r)
  }, "\u25B8 append to audit_trail"))))), /*#__PURE__*/React.createElement("div", {
    className: "cn-panel-head",
    style: {
      borderTop: "1px solid var(--border-faint)"
    }
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-lab"
  }, "LEDGER \xB7 ", trail.length)), /*#__PURE__*/React.createElement("div", {
    className: "cn-auditlog"
  }, trail.map(e => /*#__PURE__*/React.createElement("div", {
    className: "cn-audit-row",
    key: e.id
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-audit-act " + e.action
  }, e.action), /*#__PURE__*/React.createElement("span", {
    className: "cn-audit-tgt"
  }, e.target), /*#__PURE__*/React.createElement("span", {
    className: "cn-audit-just"
  }, e.justification), /*#__PURE__*/React.createElement("span", {
    className: "cn-audit-actor"
  }, e.actor)))));
}
Object.assign(window, {
  ModelsRegistry,
  ExportView,
  VoicePanel,
  OverrideAudit
});
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/console/Capitalize.jsx", error: String((e && e.message) || e) }); }

// ui_kits/console/Chrome.jsx
try { (() => {
/* Mercury Console — chrome: Sidebar + Topbar. Lucide line icons (flagged
   substitution, per the design system iconography note). */

/* Nav icons render through the canonical <mercury-icon> atom (components/mercury-icon.js):
   ONE icon registry across the system instead of a per-kit inline-SVG generator.
   This maps each nav id → its wayfinding icon name; <mercury-icon> inherits the
   nav item's currentColor, so the active/hover tint stays the rail's job. */
const NAV_ICON = {
  overview: "dashboard",
  detect: "search",
  analytics: "activity",
  batch: "grid",
  domains: "globe",
  ensemble: "chart-line",
  ethics: "shield",
  compliance: "check-doc",
  models: "layers",
  export: "download",
  voice: "mic",
  connection: "radial"
};
const NAV = [["overview", "Overview"], ["detect", "Detections"], ["analytics", "Analytics"], ["batch", "Batch Jobs"], ["domains", "Domains"], ["ensemble", "Ensemble"], ["ethics", "Ethics Gates"], ["compliance", "Compliance"], ["models", "Models"], ["export", "Export"], ["voice", "Voice"], ["connection", "Connection"]];
const PAGE_TITLE = {
  overview: "Operational Overview",
  detect: "Detection Stream",
  analytics: "Model Analytics",
  batch: "Batch Jobs",
  domains: "Domain Coverage",
  ensemble: "Detection Ensemble",
  ethics: "Ethical Gates",
  compliance: "Compliance Posture",
  models: "Engine Registry",
  export: "Data Export & Governance",
  voice: "Narrative Voice",
  connection: "API Connection"
};
function Sidebar({
  active,
  setActive
}) {
  return /*#__PURE__*/React.createElement("aside", {
    className: "mc-side cn-side"
  }, /*#__PURE__*/React.createElement("div", {
    className: "mc-side-brand"
  }, /*#__PURE__*/React.createElement("span", {
    className: "mc-side-sig"
  }, "\u2671"), /*#__PURE__*/React.createElement("div", {
    className: "mc-side-brand-t"
  }, /*#__PURE__*/React.createElement("span", {
    className: "mc-side-wm"
  }, /*#__PURE__*/React.createElement("span", {
    className: "g"
  }, "M"), /*#__PURE__*/React.createElement("span", {
    className: "c"
  }, "ERCURY"), " ", /*#__PURE__*/React.createElement("span", {
    className: "g"
  }, "A"), /*#__PURE__*/React.createElement("span", {
    className: "ig"
  }, "GENT")), /*#__PURE__*/React.createElement("span", {
    className: "mc-side-sub"
  }, "Console \xB7 SOC"))), /*#__PURE__*/React.createElement("div", {
    className: "mc-side-scroll"
  }, /*#__PURE__*/React.createElement("nav", {
    className: "mc-nav"
  }, NAV.map(([id, label]) => /*#__PURE__*/React.createElement("button", {
    key: id,
    className: "mc-navitem" + (active === id ? " on" : ""),
    onClick: () => setActive(id)
  }, /*#__PURE__*/React.createElement("mercury-icon", {
    name: NAV_ICON[id],
    size: "17",
    stroke: "1.8"
  }), " ", /*#__PURE__*/React.createElement("span", null, label))))), /*#__PURE__*/React.createElement("div", {
    className: "mc-side-foot"
  }, /*#__PURE__*/React.createElement("div", {
    className: "mc-side-foot-h"
  }, "PQC SUBSTRATE"), window.PQC.map(p => /*#__PURE__*/React.createElement("div", {
    key: p,
    className: "cn-pqc"
  }, p)), /*#__PURE__*/React.createElement("div", {
    className: "mc-side-user"
  }, /*#__PURE__*/React.createElement("div", {
    className: "mc-side-avatar"
  }, "AE"), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: "mc-side-uname"
  }, "analyst@steel-sa"), /*#__PURE__*/React.createElement("div", {
    className: "mc-side-urole"
  }, "role \xB7 operator")))));
}
function Topbar({
  live,
  onRun,
  page,
  hostOpen,
  onToggleHost,
  mode
}) {
  return /*#__PURE__*/React.createElement("header", {
    className: "cn-top"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: "cn-crumb"
  }, "CONSOLE / ", /*#__PURE__*/React.createElement("span", null, PAGE_TITLE[page])), /*#__PURE__*/React.createElement("div", {
    className: "cn-h1"
  }, PAGE_TITLE[page])), /*#__PURE__*/React.createElement("div", {
    className: "cn-top-right"
  }, /*#__PURE__*/React.createElement(LiveStatus, null), /*#__PURE__*/React.createElement("button", {
    className: "cn-hostpill" + (hostOpen ? " on" : ""),
    onClick: onToggleHost,
    title: "Engage Mercury Agent"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-hostpill-sig"
  }, "\u2671"), /*#__PURE__*/React.createElement("span", {
    className: "cn-hostpill-txt"
  }, /*#__PURE__*/React.createElement("span", {
    className: "g"
  }, "M"), /*#__PURE__*/React.createElement("span", {
    className: "c"
  }, "ERCURY"), " ", /*#__PURE__*/React.createElement("span", {
    className: "g"
  }, "A"), /*#__PURE__*/React.createElement("span", {
    className: "ig"
  }, "GENT")), /*#__PURE__*/React.createElement("span", {
    className: "cn-hostpill-state"
  }, mode)), /*#__PURE__*/React.createElement("div", {
    className: "cn-env"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-dot-amber"
  }), "MERCURY_ENV: production"), /*#__PURE__*/React.createElement("div", {
    className: "cn-live" + (live ? " on" : "")
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-live-dot"
  }), live ? "STREAMING" : "IDLE"), /*#__PURE__*/React.createElement("button", {
    className: "mc-action primary",
    onClick: onRun,
    disabled: live
  }, "\u25B8 RUN DETECTION")));
}
Object.assign(window, {
  Sidebar,
  Topbar
});
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/console/Chrome.jsx", error: String((e && e.message) || e) }); }

// ui_kits/console/ConsoleApp.jsx
try { (() => {
/* Mercury Console — app shell. Reference concept built strictly from the
   documented data model + agentic/explainability surfaces; see README.md.
   Live telemetry is principled, not theater: each domain maintains a rolling
   adjudicated window, and per-domain score generation is driven by that domain's
   real benchmark AUC (signal-detection theory) — so live anomaly-rate, mean
   score and rolling AUC genuinely differ per domain and track the benchmark. */
const {
  useState,
  useEffect,
  useRef,
  useCallback
} = React;
const DOM_BUF_CAP = 60; // rolling adjudicated window per domain (stable live AUC)
const FEED_CAP = 46; // display feed length

function ConsoleApp() {
  const [active, setActive] = useState("overview");
  const [activeDomain, setActiveDomain] = useState(null); // null = all
  const [rows, setRows] = useState([]);
  const [domBuf, setDomBuf] = useState({}); // { id: [events newest-first] }
  const [live, setLive] = useState(false);
  const [selectedId, setSelectedId] = useState(null);
  const [hostOpen, setHostOpen] = useState(false);
  const [selectedStd, setSelectedStd] = useState(null);
  const [hostSeed, setHostSeed] = useState(null);
  const [tick, setTick] = useState(0); // bumps on live stream so telemetry refreshes
  const [tau, setTau] = useState(0.50); // shared decision threshold (Detections + Analytics)
  const [lastCorr, setLastCorr] = useState(null); // correlation id of the most recent composed request (U2 trace)
  const mode = "SUPERVISED";
  const timers = useRef([]);

  // Seed ALL domains with a rolling window so every card is live from the start.
  useEffect(() => {
    const buf = {};
    let all = [];
    window.DOMAINS.forEach(d => {
      const evs = [];
      for (let i = 0; i < DOM_BUF_CAP; i++) {
        const e = window.genDetection(d, i);
        e.ts = new Date(Date.now() - (DOM_BUF_CAP - i) * 1500).toLocaleTimeString("en-GB");
        evs.unshift(e); // newest first
      }
      buf[d.id] = evs;
      all = all.concat(evs);
    });
    setDomBuf(buf);
    // display feed = most recent across all domains
    all.sort((a, b) => a.ts < b.ts ? 1 : -1);
    const feed = all.slice(0, FEED_CAP);
    setRows(feed);
    const firstAnom = feed.find(r => r.isAnom) || feed[0];
    setSelectedId(firstAnom ? firstAnom.id : null);
  }, []);
  useEffect(() => () => timers.current.forEach(clearTimeout), []);
  const pushEvents = events => {
    setDomBuf(prev => {
      const next = {
        ...prev
      };
      events.forEach(e => {
        const arr = (next[e.domainId] || []).slice();
        arr.unshift(e);
        next[e.domainId] = arr.slice(0, DOM_BUF_CAP);
      });
      return next;
    });
    setRows(prev => {
      const merged = events.concat(prev);
      return merged.slice(0, FEED_CAP);
    });
    setTick(t => t + 1);
  };
  const run = useCallback(() => {
    if (live) return;
    setLive(true);
    // stream fresh events across every domain (or the filtered one)
    const doms = activeDomain ? window.DOMAINS.filter(d => d.id === activeDomain) : window.DOMAINS;
    let t = 160;
    const total = activeDomain ? 12 : 16;
    for (let i = 0; i < total; i++) {
      timers.current.push(setTimeout(() => {
        const dom = doms[Math.floor(Math.random() * doms.length)];
        const det = window.genDetection(dom, 0);
        det.ts = new Date().toLocaleTimeString("en-GB");
        pushEvents([det]);
      }, t));
      t += 230;
    }
    timers.current.push(setTimeout(() => setLive(false), t + 200));
  }, [live, activeDomain]);

  // A composed request returns a REAL response; map it to feed rows and inject so
  // compose → watch → inspect is one instrument. Live rows carry their native
  // score + scale + correlation id (see liveEventsFromResponse).
  const runComposed = useCallback((res, endpoint, method) => {
    const events = window.liveEventsFromResponse(res, endpoint, method);
    if (!events.length) return;
    pushEvents(events);
    setSelectedId(events[0].id);
    setLastCorr(events[0].corr || res.meta && res.meta.correlationId || null);
  }, []);

  // Derive per-domain live telemetry from the rolling buffers.
  const domainStats = React.useMemo(() => {
    const out = {};
    window.DOMAINS.forEach(d => {
      const buf = domBuf[d.id] || [];
      const n = buf.length;
      const anom = buf.filter(e => e.isAnom).length;
      const flagged = buf.filter(e => e.score >= 0.5).length;
      const meanScore = n ? buf.reduce((a, e) => a + e.score, 0) / n : 0;
      const liveAUC = window.empiricalAUC(buf);
      out[d.id] = {
        n,
        anom,
        flagged,
        anomRate: n ? flagged / n : 0,
        meanScore,
        liveAUC,
        recent: buf.slice(0, 16).map(e => ({
          s: e.score,
          a: e.isAnom
        })).reverse()
      };
    });
    return out;
  }, [domBuf, tick]);
  const feedRows = activeDomain ? rows.filter(r => r.domainId === activeDomain) : rows;
  // evaluation set for Analytics: the full adjudicated window (per-domain rolling
  // buffers), scoped to the active domain — far larger than the display feed, so
  // ROC/PR/calibration curves are statistically meaningful.
  const evalSet = React.useMemo(() => {
    const ids = activeDomain ? [activeDomain] : window.DOMAINS.map(d => d.id);
    let all = [];
    ids.forEach(id => {
      all = all.concat(domBuf[id] || []);
    });
    return all;
  }, [domBuf, activeDomain, tick]);
  const detail = (activeDomain ? feedRows : rows).find(r => r.id === selectedId) || rows.find(r => r.id === selectedId) || null;
  const pickDomain = id => setActiveDomain(cur => cur === id ? null : id);
  const explainSelected = () => setHostOpen(true);
  const pickStd = std => setSelectedStd(cur => cur === std ? null : std);
  const focusFeed = id => {
    setActiveDomain(id);
    setActive("detect");
  };
  const askAgentDomain = d => {
    setHostSeed({
      ...d,
      live: domainStats[d.id]
    });
    setHostOpen(true);
  };
  const ctx = {
    selected: detail,
    rows: feedRows,
    page: active,
    domain: hostSeed
  };
  const renderView = () => {
    switch (active) {
      case "detect":
        return /*#__PURE__*/React.createElement(DetectionsView, {
          rows: feedRows,
          selectedId: selectedId,
          onSelect: setSelectedId,
          detail: detail,
          onExplain: explainSelected,
          tau: tau,
          setTau: setTau,
          onComposeResult: runComposed,
          lastCorr: lastCorr
        });
      case "analytics":
        return /*#__PURE__*/React.createElement(AnalyticsView, {
          evalSet: evalSet,
          tau: tau,
          setTau: setTau,
          activeDomain: activeDomain,
          onPickDomain: pickDomain
        });
      case "domains":
        return /*#__PURE__*/React.createElement(DomainsView, {
          activeDomain: activeDomain,
          onPickDomain: pickDomain,
          stats: domainStats,
          onFocusFeed: focusFeed,
          onAskAgent: askAgentDomain
        });
      case "ensemble":
        return /*#__PURE__*/React.createElement(EnsembleView, null);
      case "ethics":
        return /*#__PURE__*/React.createElement(EthicsView, {
          rows: rows
        });
      case "compliance":
        return /*#__PURE__*/React.createElement(ComplianceView, {
          selectedStd: selectedStd,
          onPickStd: pickStd
        });
      case "models":
        return /*#__PURE__*/React.createElement(ModelsRegistry, null);
      case "export":
        return /*#__PURE__*/React.createElement(ExportView, {
          detail: detail
        });
      case "voice":
        return /*#__PURE__*/React.createElement(VoicePanel, {
          detail: detail
        });
      case "batch":
        return /*#__PURE__*/React.createElement(BatchView, null);
      case "connection":
        return /*#__PURE__*/React.createElement(ConnectionView, null);
      default:
        return /*#__PURE__*/React.createElement(OverviewView, {
          rows: feedRows,
          selectedId: selectedId,
          onSelect: setSelectedId,
          activeDomain: activeDomain,
          onPickDomain: pickDomain,
          detail: detail,
          onExplain: explainSelected
        });
    }
  };
  const filterNote = activeDomain ? `filtered · ${window.DOMAINS.find(d => d.id === activeDomain).label}` : null;
  return /*#__PURE__*/React.createElement("div", {
    className: "cn-app" + (hostOpen ? " host-open" : "")
  }, /*#__PURE__*/React.createElement(Sidebar, {
    active: active,
    setActive: setActive
  }), /*#__PURE__*/React.createElement("main", {
    className: "cn-main"
  }, /*#__PURE__*/React.createElement(Topbar, {
    live: live,
    onRun: run,
    page: active,
    hostOpen: hostOpen,
    onToggleHost: () => setHostOpen(v => !v),
    mode: mode
  }), /*#__PURE__*/React.createElement("div", {
    className: "cn-disclaimer"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-disc-tag"
  }, "LIVE-CAPABLE"), "Detect / Batch call the real Mercury Agent API (", /*#__PURE__*/React.createElement("code", null, "/api/v1/detect/*"), ", ", /*#__PURE__*/React.createElement("code", null, "/batch/*"), ") when a server is set in ", /*#__PURE__*/React.createElement("b", null, "Connection"), " \u2014 otherwise a faithful mock returns the exact contract shapes. Analytics pages model the documented schema in brand language.", filterNote && /*#__PURE__*/React.createElement("span", {
    className: "cn-filter-note"
  }, filterNote, " ", /*#__PURE__*/React.createElement("button", {
    onClick: () => setActiveDomain(null)
  }, "clear \u2715"))), /*#__PURE__*/React.createElement("div", {
    className: "cn-scroll"
  }, renderView())), /*#__PURE__*/React.createElement(HostPanel, {
    open: hostOpen,
    onClose: () => setHostOpen(false),
    ctx: ctx,
    mode: mode
  }));
}
window.ConsoleApp = ConsoleApp;
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/console/ConsoleApp.jsx", error: String((e && e.message) || e) }); }

// ui_kits/console/ConsoleData.jsx
try { (() => {
/* Mercury Console — data model. Every value/label traces to the Mercury Agent
   repo: the detect() schema, the 3-component ensemble, benchmark tables, domain
   AUCs, ethics gates, PQC primitives, the explainability surface (SHAP +
   counterfactuals, src/omni_mercury_engine/explainability) and the agentic host
   (src/omni_mercury_engine/agentic/mercury_a_agent.py). Synthetic instances are
   generated to *match* documented distributions — no invented product metrics. */

const BENCH = {
  auc: 0.8466,
  median: 0.9100,
  f1: 0.6428,
  sets: "65/75",
  adbench: 0.8180
};
const ENSEMBLE = [{
  name: "ResonanceScore",
  short: "Resonance",
  weight: 40,
  auc: 0.7943,
  math: "h = Σ|Xₖ|² ⁄ Σ|Xₖ|², over {|Xₖ| > μ̄}",
  method: "rfft spectral harmonic ratio",
  cx: "O(n·d·log n)"
}, {
  name: "KinematicScore",
  short: "Kinematic",
  weight: 30,
  auc: 0.6412,
  math: "jerk = ∇³x = np.diff(x, n=3)",
  method: "velocity → accel → jerk baselines",
  cx: "O(n·d)"
}, {
  name: "InfoGeometryScore",
  short: "InfoGeom",
  weight: 30,
  auc: 0.8479,
  math: "d² = (x−μ)ᵀ Σ⁻¹ (x−μ)",
  method: "Fisher / Mahalanobis OOD · Cholesky",
  cx: "O(d³)"
}];
const DOMAINS = [{
  id: "security",
  label: "Network Security",
  sector: "Security",
  auc: 0.7983,
  ladder: "threat",
  stream: "tap0 · netflow v9",
  feats: ["bytes_ratio", "pkt_rate", "payload_entropy", "port_fanout", "syn_ratio"]
}, {
  id: "earthquake",
  label: "Earthquake",
  sector: "Space & Env",
  auc: 0.9367,
  ladder: "severity",
  stream: "seismo · 100 Hz",
  feats: ["pwave_amp", "sp_interval", "spectral_ratio", "ground_accel"]
}, {
  id: "tsunami",
  label: "Tsunami",
  sector: "Space & Env",
  auc: 0.8905,
  ladder: "severity",
  stream: "DART buoy array",
  feats: ["sea_level_dev", "wave_period", "dart_pressure", "coastal_vel"]
}, {
  id: "tornado",
  label: "Tornado",
  sector: "Space & Env",
  auc: 0.8803,
  ladder: "severity",
  stream: "NEXRAD L2",
  feats: ["vorticity", "pressure_drop", "hook_echo", "wind_shear"]
}, {
  id: "pandemic",
  label: "Pandemic",
  sector: "Medical",
  auc: 0.8588,
  ladder: "alert",
  stream: "WHO surveil.",
  feats: ["r_eff", "case_velocity", "test_positivity", "mobility_idx"]
}, {
  id: "energy",
  label: "Energy Grid",
  sector: "Infrastructure",
  auc: 0.8038,
  ladder: "severity",
  stream: "PMU · 60 Hz",
  feats: ["freq_dev", "load_jerk", "phase_imbalance", "thd_harmonic"]
}, {
  id: "hurricane",
  label: "Hurricane",
  sector: "Space & Env",
  auc: 0.7233,
  ladder: "severity",
  stream: "GOES-19 IR",
  feats: ["min_pressure", "wind_shear", "sst_anomaly", "eyewall_sym"]
}, {
  id: "flood",
  label: "Flood",
  sector: "Space & Env",
  auc: 0.6837,
  ladder: "severity",
  stream: "USGS gauge",
  feats: ["river_stage", "rainfall_rate", "soil_sat", "discharge"]
}, {
  id: "fema",
  label: "FEMA Disaster",
  sector: "Humanitarian",
  auc: 0.6573,
  ladder: "severity",
  stream: "OpenFEMA API",
  feats: ["damage_idx", "displaced_pop", "infra_loss", "resource_gap"]
}];
const LADDERS = {
  threat: {
    high: "HIGH",
    med: "MEDIUM",
    low: "LOW"
  },
  alert: {
    high: "CRITICAL",
    med: "WARNING",
    low: "NORMAL"
  },
  severity: {
    high: "SEVERE",
    med: "MODERATE",
    low: "NORMAL"
  }
};
const ETHICS = [{
  gate: "Benevolence η",
  glyph: "η",
  value: 0.99,
  op: "≥",
  state: "PASS",
  note: "omnibenevolent constraint",
  detail: "every action scored for human flourishing before execution"
}, {
  gate: "σ_Immutable",
  glyph: "σ",
  value: 0.96,
  op: "≥",
  state: "PASS",
  note: "hard immutable bound",
  detail: "cannot be lowered at runtime; violation halts the action"
}, {
  gate: "Lyapunov λ",
  glyph: "λ",
  value: 0.25,
  op: "=",
  state: "STABLE",
  note: "convergence certified",
  detail: "3R recursion proven contractive — no runaway dynamics"
}];
const PQC = ["Kyber-1024", "ML-DSA-65", "SPHINCS+", "AES-256-GCM"];

/* Engine registry — real detector classes from src/omni_mercury_engine/detectors.
   attrs: cx complexity class · train (free|learned) · mode (online|batch) · active */
const ENGINES = [{
  group: "Core statistical",
  items: [{
    cls: "MercuryAnomalyDetector",
    file: "statistical.py",
    note: "3R ensemble · adaptive weights",
    cx: "O(d³)",
    train: "free",
    mode: "online",
    active: true
  }, {
    cls: "EnhancedStatisticalDetector",
    file: "enhanced_statistical.py",
    note: "extended feature bank",
    cx: "O(d³)",
    train: "free",
    mode: "online"
  }, {
    cls: "AccelerationDynamics",
    file: "acceleration_dynamics.py",
    note: "kinematic / physics core",
    cx: "O(n·d)",
    train: "free",
    mode: "online"
  }, {
    cls: "AdvancedPhysicsIntegration",
    file: "advanced_physics_integration.py",
    note: "multi-physics fusion",
    cx: "O(n·d²)",
    train: "free",
    mode: "batch"
  }]
}, {
  group: "Advanced / neural",
  items: [{
    cls: "MultiScaleTransformer",
    file: "advanced/multi_scale_transformer.py",
    note: "temporal attention",
    cx: "O(n²·d)",
    train: "learned",
    mode: "batch"
  }, {
    cls: "AdversarialAutoencoder",
    file: "advanced/adversarial_ae.py",
    note: "reconstruction OOD",
    cx: "O(n·d²)",
    train: "learned",
    mode: "batch"
  }, {
    cls: "ContrastiveDetector",
    file: "advanced/contrastive_detector.py",
    note: "self-supervised",
    cx: "O(n·d²)",
    train: "learned",
    mode: "batch"
  }, {
    cls: "GWOEnsemble",
    file: "advanced/gwo_ensemble.py",
    note: "grey-wolf optimized weights",
    cx: "O(k·n·d)",
    train: "learned",
    mode: "batch"
  }, {
    cls: "COPODDetector",
    file: "advanced/copod_detector.py",
    note: "copula probabilistic",
    cx: "O(n·d)",
    train: "free",
    mode: "batch"
  }]
}, {
  group: "Domain detectors",
  items: [{
    cls: "DisasterDetectors",
    file: "geological/disaster_detectors.py",
    note: "geological suite",
    cx: "O(n·d)",
    train: "free",
    mode: "online"
  }, {
    cls: "TornadoDetector",
    file: "geological/tornado_detector.py",
    note: "mesocyclone",
    cx: "O(n·d)",
    train: "free",
    mode: "online"
  }, {
    cls: "HurricaneDetector",
    file: "geological/hurricane_detector.py",
    note: "cyclone intensity",
    cx: "O(n·d)",
    train: "free",
    mode: "online"
  }, {
    cls: "FinancialCrisisDetector",
    file: "economic/financial_crisis_detector.py",
    note: "market regime",
    cx: "O(n·d)",
    train: "learned",
    mode: "batch"
  }, {
    cls: "EMPDetector",
    file: "energy/emp_detector.py",
    note: "grid / EMP",
    cx: "O(n·d)",
    train: "free",
    mode: "online"
  }, {
    cls: "MultimodalFusion",
    file: "fusion/multimodal_fusion.py",
    note: "cross-modal",
    cx: "O(n·d²)",
    train: "learned",
    mode: "batch"
  }]
}];

/* Compliance modules — real files in src/omni_mercury_engine/compliance + explainability. */
const NIST_FUNCTIONS = [{
  fn: "Govern",
  cov: 0.90
}, {
  fn: "Identify",
  cov: 0.96
}, {
  fn: "Protect",
  cov: 0.97
}, {
  fn: "Detect",
  cov: 0.99
}, {
  fn: "Respond",
  cov: 0.92
}, {
  fn: "Recover",
  cov: 0.88
}];
const COMPLIANCE = [{
  std: "NIST CSF 2.0",
  file: "compliance/nist_csf_integrator.py",
  state: "ALIGNED",
  note: "Govern · Identify · Protect · Detect · Respond · Recover",
  coverage: 0.94,
  owner: "Security Eng",
  assessed: "2026-05-12",
  review: "2026-08-12",
  controls: [109, 116],
  findings: [{
    sev: "low",
    text: "Recover.RC-2 playbook lacks automated drill evidence"
  }],
  evidence: ["nist_csf_integrator.py self-report", "CI control-coverage run (per push)"]
}, {
  std: "FIRST TLP 2.0",
  file: "compliance/tlp_handler.py",
  state: "ENFORCED",
  note: "disclosure tagging on every published report",
  coverage: 1.0,
  owner: "Disclosure",
  assessed: "2026-05-20",
  review: "2026-11-20",
  controls: [5, 5],
  findings: [],
  evidence: ["tlp_handler.py unit tests (5/5 levels)", "emit-time tag on every report"]
}, {
  std: "OSHA",
  file: "compliance/osha_anomaly.py",
  state: "MONITORED",
  note: "workplace-safety anomaly thresholds",
  coverage: 0.88,
  owner: "Safety Ops",
  assessed: "2026-04-30",
  review: "2026-07-30",
  controls: [22, 25],
  findings: [{
    sev: "med",
    text: "3 thresholds pending SME sign-off"
  }, {
    sev: "low",
    text: "Heat-stress model not yet regionalized"
  }],
  evidence: ["osha_anomaly.py thresholds table", "incident replay 2026-Q1"]
}, {
  std: "GDPR Art. 22",
  file: "explainability/gdpr_compliance.py",
  state: "ALIGNED",
  note: "automated-decision explanations on request",
  coverage: 0.91,
  owner: "Privacy / Legal",
  assessed: "2026-05-08",
  review: "2026-08-08",
  controls: [10, 11],
  findings: [{
    sev: "low",
    text: "DSAR export latency p95 above 24h target"
  }],
  evidence: ["gdpr_compliance.py explanation API", "counterfactual + SHAP on every decision"]
}];
const TLP = [{
  tag: "TLP:RED",
  color: "var(--tlp-red)",
  scope: "named recipients only"
}, {
  tag: "TLP:AMBER",
  color: "var(--tlp-amber)",
  scope: "organization + clients"
}, {
  tag: "TLP:GREEN",
  color: "var(--tlp-green)",
  scope: "community"
}, {
  tag: "TLP:CLEAR",
  color: "var(--tlp-clear)",
  scope: "no restriction"
}];
const ACTIONS = {
  threat: {
    high: "Isolate host · rotate session keys (Kyber-1024)",
    med: "Rate-limit source · raise capture depth",
    low: "Log · continue monitoring"
  },
  alert: {
    high: "Escalate to clinician · open case",
    med: "Flag for review · re-sample",
    low: "Record baseline"
  },
  severity: {
    high: "Issue advisory · notify EOC",
    med: "Pre-stage resources · watch",
    low: "Archive observation"
  }
};

/* Domain → detector class mapping (which engine handles each domain) + benchmark
   context (n datasets contributing, nominal sample size). Grounds the drill-down. */
const DOMAIN_META = {
  security: {
    detector: "MercuryAnomalyDetector",
    file: "statistical.py",
    n: 11,
    samples: 18420,
    baseRate: 0.18
  },
  earthquake: {
    detector: "DisasterDetectors",
    file: "geological/disaster_detectors.py",
    n: 9,
    samples: 12030,
    baseRate: 0.10
  },
  tsunami: {
    detector: "DisasterDetectors",
    file: "geological/disaster_detectors.py",
    n: 7,
    samples: 6240,
    baseRate: 0.07
  },
  tornado: {
    detector: "TornadoDetector",
    file: "geological/tornado_detector.py",
    n: 8,
    samples: 8910,
    baseRate: 0.12
  },
  pandemic: {
    detector: "EnhancedStatisticalDetector",
    file: "enhanced_statistical.py",
    n: 6,
    samples: 4370,
    baseRate: 0.14
  },
  energy: {
    detector: "EMPDetector",
    file: "energy/emp_detector.py",
    n: 8,
    samples: 15600,
    baseRate: 0.13
  },
  hurricane: {
    detector: "HurricaneDetector",
    file: "geological/hurricane_detector.py",
    n: 7,
    samples: 5180,
    baseRate: 0.11
  },
  flood: {
    detector: "DisasterDetectors",
    file: "geological/disaster_detectors.py",
    n: 6,
    samples: 7720,
    baseRate: 0.16
  },
  fema: {
    detector: "MultimodalFusion",
    file: "fusion/multimodal_fusion.py",
    n: 5,
    samples: 3410,
    baseRate: 0.15
  }
};

/* TLP 2.0 (FIRST.org) — the full disclosure ladder. TLP governs the RECIPIENT
   SCOPE of a report, not its severity. The console applies a default tag from
   severity, but an analyst can override; this is the policy + the real boundary. */
const TLP_LEVELS = [{
  tag: "TLP:RED",
  band: "red",
  boundary: "Named recipients only — not shared further, even within their org."
}, {
  tag: "TLP:AMBER+STRICT",
  band: "amber",
  boundary: "Recipient organization only — no client/3rd-party sharing."
}, {
  tag: "TLP:AMBER",
  band: "amber",
  boundary: "Recipient org + clients/customers on a need-to-know basis."
}, {
  tag: "TLP:GREEN",
  band: "green",
  boundary: "Community / peers & partners — not posted publicly."
}, {
  tag: "TLP:CLEAR",
  band: "clear",
  boundary: "No restriction — public, subject to standard copyright."
}];
/* default disclosure policy: severity → TLP (a recommendation, analyst-overridable) */
const SEV_TLP_POLICY = [{
  sev: "CRITICAL · life-safety / classified",
  tlp: "TLP:RED",
  rationale: "active threat to named assets; eyes-only"
}, {
  sev: "HIGH / SEVERE",
  tlp: "TLP:AMBER+STRICT",
  rationale: "actionable internally; contain to org"
}, {
  sev: "MEDIUM / MODERATE / WARNING",
  tlp: "TLP:AMBER",
  rationale: "share with affected clients, need-to-know"
}, {
  sev: "LOW / MINOR",
  tlp: "TLP:GREEN",
  rationale: "useful to the community / partners"
}, {
  sev: "NOMINAL · routine telemetry",
  tlp: "TLP:CLEAR",
  rationale: "no sensitivity; publishable"
}];
function rnd(a, b) {
  return a + Math.random() * (b - a);
}
function r3(x) {
  return +x.toFixed(3);
}
function clamp01(x) {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

/* Inverse standard-normal CDF (Acklam's rational approximation). */
function normInv(p) {
  if (p <= 0) return -6;
  if (p >= 1) return 6;
  const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
  const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01];
  const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
  const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
  const pl = 0.02425;
  let q, r;
  if (p < pl) {
    q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p <= 1 - pl) {
    q = p - 0.5;
    r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  }
  q = Math.sqrt(-2 * Math.log(1 - p));
  return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
}
/* Standard normal sample (Box–Muller). */
function gauss() {
  let u = 0,
    v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
/* Rolling empirical ROC-AUC from an adjudicated window (Mann–Whitney). null if
   a class is absent. This is what an institution monitors live vs the benchmark. */
function empiricalAUC(win) {
  const pos = win.filter(d => d.isAnom).map(d => d.score);
  const neg = win.filter(d => !d.isAnom).map(d => d.score);
  if (!pos.length || !neg.length) return null;
  let u = 0;
  for (const p of pos) for (const ng of neg) u += p > ng ? 1 : p === ng ? 0.5 : 0;
  return u / (pos.length * neg.length);
}

/* SHAP-style signed attribution that roughly reconstructs the score. */
function attribution(dom, score, isAnom) {
  const feats = dom.feats.slice(0, 4);
  let weights = feats.map(() => rnd(0.05, 1));
  if (isAnom) weights[0] += 1.4; // a dominant driver
  const sum = weights.reduce((a, b) => a + b, 0);
  return feats.map((f, i) => {
    const mag = weights[i] / sum * score;
    const sign = isAnom ? i === 0 ? 1 : Math.random() < 0.7 ? 1 : -1 : Math.random() < 0.5 ? 1 : -1;
    return {
      f,
      w: r3(sign * mag),
      obs: r3(rnd(0.1, 9.9) * (i === 0 && isAnom ? 9 : 1))
    };
  }).sort((a, b) => Math.abs(b.w) - Math.abs(a.w));
}
function genDetection(dom, seq) {
  const meta = DOMAIN_META[dom.id] || {
    baseRate: 0.15
  };
  const isAnom = Math.random() < meta.baseRate;
  // Signal-detection model: a detector with ROC-AUC = a separates the normal and
  // anomaly score distributions by d′ = √2·Φ⁻¹(a). Strong domains (Earthquake
  // 0.94) produce cleanly separable live scores; weak ones (FEMA 0.66) overlap —
  // so live anomaly-rate, mean score and rolling AUC genuinely DIFFER per domain.
  const sigma = 0.135,
    muN = 0.27;
  const dprime = Math.SQRT2 * normInv(dom.auc);
  const score = clamp01((isAnom ? muN + dprime * sigma : muN) + gauss() * sigma);
  const conf = isAnom ? Math.min(0.99, 0.7 + score * 0.3) : 0.8 + (1 - score) * 0.15;
  const lad = LADDERS[dom.ladder];
  const sev = !isAnom ? lad.low : score > 0.75 ? lad.high : score > 0.45 ? lad.med : lad.low;
  const sevKey = sev === lad.high ? "high" : sev === lad.med ? "med" : "low";

  // benevolence: the ACTION's alignment score, scored independently of detection.
  // Most actions clear the η ≥ 0.99 gate; ~12% are withheld (refused) by the gate.
  let benev;
  if (Math.random() < 0.12) benev = r3(rnd(0.982, 0.9899)); // WITHHELD
  else benev = r3(rnd(0.990, 0.999)); // cleared
  const withheld = benev < 0.99;
  const gateMargin = r3(benev - 0.99);

  // ensemble component contributions (weighted ≈ score)
  const comp = {
    resonance: r3(Math.min(1, score * rnd(0.8, 1.2))),
    kinematic: r3(Math.min(1, score * rnd(0.6, 1.1))),
    infogeom: r3(Math.min(1, score * rnd(0.9, 1.25)))
  };
  const feats = attribution(dom, score, isAnom);
  const top = feats[0];
  const d = new Date(Date.now() - seq * 1100);
  const ts = `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}:${String(d.getSeconds()).padStart(2, "0")}`;
  return {
    id: `${dom.id}-${Date.now()}-${seq}-${Math.floor(Math.random() * 999)}`,
    domain: dom.label,
    domainId: dom.id,
    sector: dom.sector,
    stream: dom.stream,
    ts,
    seq: 1000 + Math.floor(rnd(0, 8999)),
    score: r3(score),
    conf: r3(conf),
    benev: r3(benev),
    withheld,
    gateMargin,
    isAnom,
    sev,
    sevKey,
    comp,
    feats,
    dims: dom.feats.length,
    counterfactual: {
      f: top.f,
      obs: top.obs,
      bound: r3(top.obs * rnd(0.12, 0.3))
    },
    action: ACTIONS[dom.ladder][sevKey]
  };
}
function seedFeed(domains, n) {
  const out = [];
  for (let i = 0; i < n; i++) {
    const dom = domains[Math.floor(Math.random() * domains.length)];
    out.push(genDetection(dom, i));
  }
  return out;
}

/* Map a REAL MercuryAPI response into feed-row events. These are LIVE rows: they
   carry their NATIVE score + scale (z-score / L2 distance / fused / [0,1]) and
   full provenance (correlation ID, source, endpoint, raw payload) — they are
   never coerced into the ensemble [0,1] space, and the τ histogram/confusion
   instruments exclude them (those are defined over the synthetic ensemble
   stream). This is what makes "compose → watch → inspect" one instrument
   without conflating two score scales. */
function sevKeyOf(s) {
  s = (s || "").toLowerCase();
  return s === "critical" || s === "high" ? "high" : s === "medium" ? "med" : "low";
}
function contribFeats(fc, index) {
  if (!fc) return [];
  return Object.keys(fc).map(f => ({
    f,
    w: Array.isArray(fc[f]) ? fc[f][index] != null ? fc[f][index] : 0 : fc[f]
  })).sort((a, b) => Math.abs(b.w) - Math.abs(a.w));
}
function liveEventsFromResponse(res, endpoint, method, rawInput) {
  if (!res || !res.data) return [];
  const d = res.data;
  const corr = res.meta && res.meta.correlationId ? res.meta.correlationId : null;
  const ts = new Date().toLocaleTimeString("en-GB");
  const stamp = Date.now();
  const base = (extra, i) => Object.assign({
    id: "live-" + stamp + "-" + (i || 0) + "-" + Math.floor(Math.random() * 9999),
    domain: "Composer · " + endpoint,
    domainId: "live",
    sector: "Live API",
    stream: "POST /api/v1/detect/" + endpoint,
    ts,
    seq: 0,
    conf: null,
    benev: null,
    withheld: false,
    gateMargin: null,
    feats: [],
    dims: 0,
    live: true,
    corr,
    source: res.source,
    status: res.status,
    latencyMs: res.latencyMs,
    endpoint,
    method: method || d.method || endpoint,
    threshold: d.threshold != null ? d.threshold : null,
    raw: d
  }, extra);
  if (endpoint === "univariate" || endpoint === "multivariate") {
    const scale = endpoint === "univariate" ? "σ · z-score" : "L2 / Mahalanobis";
    const pts = d.anomaly_points && d.anomaly_points.length ? d.anomaly_points : (d.anomalies || []).map((a, i) => a ? {
      index: i,
      score: (d.scores || [])[i],
      severity: "high"
    } : null).filter(Boolean);
    const conf = d.summary ? 1 - (d.summary.anomaly_rate || 0) : null;
    if (pts.length) {
      return pts.map((p, i) => base({
        seq: p.index != null ? p.index : i,
        score: p.score,
        scale,
        isAnom: true,
        sev: (p.severity || "high").toUpperCase(),
        sevKey: sevKeyOf(p.severity),
        conf,
        feats: contribFeats(d.feature_contributions, p.index),
        action: "Point flagged at threshold " + d.threshold + " on the " + scale + " scale."
      }, i));
    }
    const maxS = d.scores && d.scores.length ? Math.max.apply(null, d.scores) : 0;
    return [base({
      score: maxS,
      scale,
      isAnom: false,
      sev: "NORMAL",
      sevKey: "low",
      conf,
      action: "No point crossed threshold " + d.threshold + " (" + scale + ")."
    })];
  }
  if (endpoint === "fusion") {
    return [base({
      score: d.fused_score,
      scale: "fused ensemble",
      isAnom: !!d.is_anomaly,
      conf: d.confidence,
      sev: d.is_anomaly ? "HIGH" : "NORMAL",
      sevKey: d.is_anomaly ? "high" : "low",
      feats: contribFeats(d.detector_contributions),
      action: d.explanation
    })];
  }
  if (endpoint === "three-r") {
    const anom = d.fusion_score >= 0.5;
    return [base({
      score: d.fusion_score,
      scale: "0–1",
      isAnom: anom,
      conf: d.ethical_scaling,
      sev: anom ? "HIGH" : "NORMAL",
      sevKey: anom ? "high" : "low",
      feats: [{
        f: "recursion",
        w: d.recursion_score
      }, {
        f: "resonance",
        w: d.resonance_score
      }, {
        f: "optimization",
        w: d.optimization_score
      }],
      action: "3R " + (d.is_stable ? "stable" : "UNSTABLE") + " · λ ≤ " + d.lyapunov_bound + " · η^Φ " + d.ethical_scaling
    })];
  }
  return [];
}

/* Adaptive ensemble weights derived from per-component discriminative power:
   excess AUC over chance (0.5) normalized — the documented Mann-Whitney scheme.
   Shows why InfoGeom earns more than its nominal 30% and Kinematic less. */
function adaptiveWeights() {
  const excess = ENSEMBLE.map(e => Math.max(0, e.auc - 0.5));
  const sum = excess.reduce((a, b) => a + b, 0);
  return ENSEMBLE.map((e, i) => ({
    name: e.short,
    auc: e.auc,
    excess: r3(excess[i]),
    nominal: e.weight,
    adaptive: +(excess[i] / sum * 100).toFixed(1)
  }));
}
Object.assign(window, {
  BENCH,
  ENSEMBLE,
  DOMAINS,
  LADDERS,
  ETHICS,
  PQC,
  ENGINES,
  COMPLIANCE,
  NIST_FUNCTIONS,
  TLP,
  ACTIONS,
  DOMAIN_META,
  TLP_LEVELS,
  SEV_TLP_POLICY,
  genDetection,
  seedFeed,
  adaptiveWeights,
  empiricalAUC,
  liveEventsFromResponse
});
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/console/ConsoleData.jsx", error: String((e && e.message) || e) }); }

// ui_kits/console/Host.jsx
try { (() => {
/* Mercury Console — the Mercury Agent host. A presence for the agentic layer
   (src/omni_mercury_engine/agentic/mercury_a_agent.py: MercuryAgent /
   MercuryReasoner / MercuryPlanner / AgentMode + Bayesian confidence). The host
   reasons in steps and answers in the documented brand voice. Responses are
   composed from real detection state + the explainability surface (SHAP +
   counterfactuals), not free-form generation. */
const {
  useState,
  useEffect,
  useRef
} = React;
const HOST_PROMPTS = [{
  id: "explain",
  label: "Explain this detection"
}, {
  id: "ethics",
  label: "Why did it clear the gate?"
}, {
  id: "action",
  label: "Recommend an action"
}, {
  id: "posture",
  label: "Summarize threat posture"
}];
function pct(x) {
  return (x * 100).toFixed(1) + "%";
}

/* Compose a reasoning trace + answer for an intent, given context. */
function hostReply(intent, ctx) {
  const {
    selected,
    rows,
    page
  } = ctx;
  if (intent === "explain") {
    if (!selected) return {
      steps: ["No event selected."],
      body: "Select a row in the detection feed and I will walk the attribution. I read SHAP feature weights and a counterfactual for every decision."
    };
    const t = selected.feats[0];
    const dom = selected.comp.infogeom >= selected.comp.resonance ? "InfoGeometry (Mahalanobis)" : "Resonance (spectral)";
    return {
      steps: [`recall · ${selected.domain} · sample #${selected.seq}`, `attribute · SHAP top driver → ${t.f}`, `localize · dominant component → ${dom}`, `counterfactual · ${selected.counterfactual.f} ≤ ${selected.counterfactual.bound}`],
      body: selected.isAnom ? `This is an ANOMALY at score ${selected.score.toFixed(3)} (conf ${selected.score >= 0 ? pct(selected.conf) : ""}). The decision is driven mainly by ${t.f} (SHAP ${t.w > 0 ? "+" : ""}${t.w}), with ${dom} the dominant ensemble term. Counterfactual: had ${selected.counterfactual.f} stayed ≤ ${selected.counterfactual.bound} (observed ${selected.counterfactual.obs}), this would read NORMAL.` : `This reads NORMAL at score ${selected.score.toFixed(3)}. No feature crosses its learned bound; ${t.f} contributes the most (${t.w}) but stays sub-threshold. Nothing to escalate.`
    };
  }
  if (intent === "ethics") {
    const b = selected ? selected.benev.toFixed(3) : "0.990";
    if (selected && selected.withheld) {
      return {
        steps: ["gate · Benevolence η ≥ 0.99", "eval · η " + b + " < 0.99", "raise · EthicalConstraintViolationError"],
        body: `This action was <b>WITHHELD</b>. Benevolence scored ${b}, below the 0.99 gate, so I raised EthicalConstraintViolationError and refused to act — regardless of the ${selected.score.toFixed(3)} detection score. There is no advisory mode; a sub-threshold benevolence is a hard stop, not a warning.`
      };
    }
    return {
      steps: ["gate · Benevolence η ≥ 0.99", "gate · σ_Immutable ≥ 0.96", "stability · Lyapunov λ = 0.25"],
      body: `The action was permitted because Benevolence scored ${b} (≥ 0.99) and σ_Immutable held ≥ 0.96. Both are hard gates — there is no advisory mode. Had either failed I would have raised EthicalConstraintViolationError and refused to act, regardless of detection score.`
    };
  }
  if (intent === "action") {
    if (!selected) return {
      steps: ["No event selected."],
      body: "Select an event and I will recommend a calibrated action with its TLP disclosure tag."
    };
    const tlp = selected.sevKey === "high" ? "TLP:AMBER+STRICT" : selected.sevKey === "med" ? "TLP:AMBER" : "TLP:GREEN";
    return {
      steps: [`plan · severity ${selected.sev}`, "calibrate · Bayesian confidence", `disclose · ${tlp}`],
      body: `Recommended: ${selected.action}. Confidence is Bayesian-calibrated at ${pct(selected.conf)}. Disclose under ${tlp}. I can stage this as a Task at priority ${selected.sevKey === "high" ? "P1" : selected.sevKey === "med" ? "P2" : "P3"} on your approval.`
    };
  }
  // posture
  const anom = rows.filter(r => r.isAnom);
  const byDom = {};
  anom.forEach(r => {
    byDom[r.domain] = (byDom[r.domain] || 0) + 1;
  });
  const top = Object.entries(byDom).sort((a, b) => b[1] - a[1])[0];
  return {
    steps: [`scan · ${rows.length} events in window`, `cluster · ${anom.length} anomalies`, top ? `focus · ${top[0]}` : "focus · none"],
    body: anom.length === 0 ? `Posture is nominal — ${rows.length} events in the window, zero anomalies. All ethics gates holding. I am in supervised mode and will surface anything that crosses threshold.` : `${anom.length} of ${rows.length} events are anomalous${top ? `, concentrated in ${top[0]} (${top[1]})` : ""}. Highest severity in view is ${anom.map(a => a.sev).sort()[0] || "—"}. Gates remain green. Want me to open the top case?`
  };
}
const GREETING = {
  steps: ["boot · AgentMode.SUPERVISED", "load · AgentMemory", "arm · ethical gates"],
  body: "Mercury Agent online. I host this console — I reason over every detection, hold the ethics gates, and can explain or act on anything you select. Ask me, or pick a detection and say “explain”."
};

/* Route a free-text question to one of the known intents. */
function routeQuestion(raw) {
  const q = (raw || "").toLowerCase();
  const has = (...ks) => ks.some(k => q.indexOf(k) >= 0);
  if (has("gate", "benevol", "ethic", "σ", "sigma", "η", "eta", "permit", "withh", "conscience", "lyapunov", "λ")) return "ethics";
  if (has("action", "recommend", "do ", "respond", "mitigat", "isolat", "escalat", "tlp", "task", "next step", "handle")) return "action";
  if (has("posture", "summar", "overview", "status", "how many", "cluster", "situation", "across", "trend")) return "posture";
  if (has("explain", "why", "anomal", "driver", "shap", "attribut", "feature", "detect", "score", "counterfactual", "this")) return "explain";
  return null;
}
function HostThread({
  thread
}) {
  const endRef = useRef(null);
  useEffect(() => {
    if (endRef.current) endRef.current.scrollTop = endRef.current.scrollHeight;
  }, [thread]);
  return /*#__PURE__*/React.createElement("div", {
    className: "cn-host-thread",
    ref: endRef
  }, thread.map((m, i) => m.role === "user" ? /*#__PURE__*/React.createElement("div", {
    className: "cn-host-user",
    key: i
  }, m.text) : /*#__PURE__*/React.createElement("div", {
    className: "cn-host-msg",
    key: i
  }, m.steps && /*#__PURE__*/React.createElement("div", {
    className: "cn-host-steps"
  }, m.steps.map((s, j) => /*#__PURE__*/React.createElement("div", {
    className: "cn-host-step",
    key: j
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-host-br"
  }, "\u2514\u2500"), s))), /*#__PURE__*/React.createElement("div", {
    className: "cn-host-body"
  }, m.body))));
}
function HostPanel({
  open,
  onClose,
  ctx,
  mode
}) {
  const [thread, setThread] = useState([{
    role: "agent",
    ...GREETING
  }]);
  const [draft, setDraft] = useState("");

  // When asked to brief on a domain (from the Domains drill-down), append a briefing.
  React.useEffect(() => {
    const d = ctx.domain;
    if (!d) return;
    const meta = window.DOMAIN_META[d.id];
    const tier = d.auc >= 0.85 ? "strong" : d.auc >= 0.75 ? "moderate" : "weak";
    const briefing = {
      steps: [`brief · ${d.label}`, `detector · ${meta.detector}`, `AUC ${d.auc.toFixed(3)} · ${tier}`],
      body: `${d.label} runs on ${meta.detector} (${meta.file}), reading ${d.stream} into a ${d.feats.length}-dim vector. Mean ROC-AUC is ${d.auc.toFixed(3)} — a ${tier} discriminator (${d.auc >= window.BENCH.auc ? "above" : "below"} the 0.8285 fleet mean). ${tier === "weak" ? "Treat its flags as advisory and pair with human review." : tier === "moderate" ? "Reliable, but corroborate high-stakes calls." : "Flag with confidence."}`
    };
    setThread(t => [...t, {
      role: "user",
      text: `Brief me on ${d.label}`
    }, {
      role: "agent",
      ...briefing
    }]);
  }, [ctx.domain && ctx.domain.id]);
  const ask = p => {
    const reply = hostReply(p.id, ctx);
    setThread(t => [...t, {
      role: "user",
      text: p.label
    }, {
      role: "agent",
      ...reply
    }]);
  };
  const submit = e => {
    e.preventDefault();
    const text = draft.trim();
    if (!text) return;
    setDraft("");
    const intent = routeQuestion(text);
    const reply = intent ? hostReply(intent, ctx) : {
      steps: ["parse · free question", "scope · this console", "route · no exact match"],
      body: "I reason over what's on this console — a selected detection (its SHAP attribution and counterfactual), the ethical gates, a recommended action, or the overall threat posture. Ask me to explain a detection, assess the gate, recommend an action, or summarize posture — or tap a chip below."
    };
    setThread(t => [...t, {
      role: "user",
      text
    }, {
      role: "agent",
      ...reply
    }]);
  };
  return /*#__PURE__*/React.createElement("aside", {
    className: "cn-host" + (open ? " open" : "")
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-host-head"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-host-sig"
  }, "\u2671"), /*#__PURE__*/React.createElement("div", {
    className: "cn-host-id"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-host-name"
  }, /*#__PURE__*/React.createElement("span", {
    className: "g"
  }, "M"), /*#__PURE__*/React.createElement("span", {
    className: "c"
  }, "ERCURY"), " ", /*#__PURE__*/React.createElement("span", {
    className: "g"
  }, "A"), /*#__PURE__*/React.createElement("span", {
    className: "ig"
  }, "GENT")), /*#__PURE__*/React.createElement("div", {
    className: "cn-host-state"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-host-dot"
  }), "online \xB7 ", mode)), /*#__PURE__*/React.createElement("button", {
    className: "cn-host-x",
    onClick: onClose
  }, "\u2715")), /*#__PURE__*/React.createElement(HostThread, {
    thread: thread
  }), /*#__PURE__*/React.createElement("div", {
    className: "cn-host-prompts"
  }, HOST_PROMPTS.map(p => /*#__PURE__*/React.createElement("button", {
    key: p.id,
    className: "cn-host-chip",
    onClick: () => ask(p)
  }, p.label))), /*#__PURE__*/React.createElement("form", {
    className: "cn-host-ask",
    onSubmit: submit
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-host-ask-sig"
  }, "\u2671"), /*#__PURE__*/React.createElement("input", {
    className: "cn-host-ask-input",
    value: draft,
    onChange: e => setDraft(e.target.value),
    placeholder: "Ask Mercury Agent\u2026",
    autoComplete: "off"
  }), /*#__PURE__*/React.createElement("button", {
    type: "submit",
    className: "cn-host-ask-send",
    "aria-label": "Send"
  }, "\u25B8")), /*#__PURE__*/React.createElement("div", {
    className: "cn-host-foot"
  }, "MercuryReasoner \xB7 Bayesian-calibrated \xB7 agentic/mercury_a_agent.py"));
}
Object.assign(window, {
  HostPanel,
  hostReply
});
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/console/Host.jsx", error: String((e && e.message) || e) }); }

// ui_kits/console/LiveViews.jsx
try { (() => {
/* ============================================================================
   Mercury Console — LIVE surfaces (real API calls via window.MercuryAPI):
     · LiveStatus    — topbar pill: LIVE/MOCK source, rate-limit budget, latency
     · ConnectionView — base URL + auth + health probe + response-meta inspector
     · RunComposer   — real request composer for all detect endpoints, embedded
                        in Detections; results inject into the live feed
     · BatchView      — submit + poll async batch jobs (lifecycle board)
   Every panel shows whether a response was served LIVE or by the faithful MOCK.
   ============================================================================ */
const {
  useState: useStateL,
  useEffect: useEffectL,
  useRef: useRefL
} = React;
function useApiState() {
  const [st, setSt] = useStateL(window.MercuryAPI.getState());
  useEffectL(() => window.MercuryAPI.onChange(setSt), []);
  return st;
}

/* ---------------- topbar status pill ---------------- */
function LiveStatus() {
  const st = useApiState();
  const [, setTick] = React.useState(0);
  React.useEffect(() => {
    const t = setInterval(() => setTick(n => n + 1), 1000);
    return () => clearInterval(t);
  }, []);
  const reach = st.health.reachable;
  const mode = !st.baseUrl ? "MOCK" : reach === true ? "LIVE" : reach === false ? "OFFLINE→MOCK" : "MOCK";
  const cls = mode === "LIVE" ? "live" : mode === "OFFLINE→MOCK" ? "off" : "mock";
  const m = st.lastMeta;
  const rl = m && m.rateLimit;
  const is429 = m && m.status === 429;
  const limit = rl && rl.limit != null ? rl.limit : null;
  const remaining = rl && rl.remaining != null ? rl.remaining : null;
  const pct = remaining != null && limit ? Math.max(0, Math.min(100, Math.round(remaining / limit * 100))) : null;
  const low = pct != null && pct <= 20;
  const retry = is429 ? rl && rl.reset != null ? Math.max(0, rl.reset - Math.floor(Date.now() / 1000)) : m.retryAfter != null ? m.retryAfter : null : null;
  return /*#__PURE__*/React.createElement("div", {
    className: "cn-apistat " + (is429 ? "off" : cls),
    title: st.baseUrl || "no server configured · faithful mock"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-apistat-dot"
  }), /*#__PURE__*/React.createElement("span", {
    className: "cn-apistat-mode"
  }, mode), /*#__PURE__*/React.createElement("span", {
    className: "cn-apistat-auth" + (st.authed ? " on" : ""),
    title: st.authed ? "X-API-Key / Bearer present" : "anonymous · batch needs auth"
  }, st.authed ? "AUTHED" : "ANON"), is429 ? /*#__PURE__*/React.createElement("span", {
    className: "cn-apistat-429"
  }, "429 \xB7 RETRY ", retry != null ? retry + "s" : "…") : pct != null ? /*#__PURE__*/React.createElement("span", {
    className: "cn-apistat-budget" + (low ? " low" : ""),
    title: "rate budget · " + remaining + " of " + limit + " left this window"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-apistat-meter"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-apistat-meterfill",
    style: {
      width: pct + "%"
    }
  })), /*#__PURE__*/React.createElement("span", {
    className: "cn-apistat-num"
  }, remaining, "/", limit)) : null, st.health.latencyMs != null && reach && /*#__PURE__*/React.createElement("span", {
    className: "cn-apistat-ms"
  }, st.health.latencyMs, "ms"));
}

/* ---------------- shared: source badge + meta strip ---------------- */
function SourceBadge({
  res
}) {
  if (!res) return null;
  return /*#__PURE__*/React.createElement("span", {
    className: "cn-srcbadge " + res.source
  }, res.source === "live" ? "● LIVE" : "○ MOCK");
}
function MetaStrip({
  res
}) {
  if (!res || !res.meta) return null;
  const m = res.meta,
    rl = m.rateLimit || {};
  return /*#__PURE__*/React.createElement("div", {
    className: "cn-meta"
  }, m.correlationId && /*#__PURE__*/React.createElement("span", null, /*#__PURE__*/React.createElement("b", null, "corr"), " ", m.correlationId), rl.limit != null && /*#__PURE__*/React.createElement("span", null, /*#__PURE__*/React.createElement("b", null, "rate"), " ", rl.remaining, "/", rl.limit), m.retryAfter != null && /*#__PURE__*/React.createElement("span", {
    className: "warn"
  }, /*#__PURE__*/React.createElement("b", null, "retry-after"), " ", m.retryAfter, "s"), res.latencyMs != null && /*#__PURE__*/React.createElement("span", null, /*#__PURE__*/React.createElement("b", null, "lat"), " ", res.latencyMs, "ms"), /*#__PURE__*/React.createElement("span", null, /*#__PURE__*/React.createElement("b", null, "http"), " ", m.status));
}

/* ---------------- CONNECTION ---------------- */
function ConnectionView() {
  const st = useApiState();
  const cfg = window.MercuryAPI.getConfig();
  const [baseUrl, setBaseUrl] = useStateL(cfg.baseUrl);
  const [apiKey, setApiKey] = useStateL(cfg.apiKey);
  const [bearer, setBearer] = useStateL(cfg.bearer);
  const [autoFallback, setAutoFallback] = useStateL(cfg.autoFallback);
  const [probing, setProbing] = useStateL(false);
  const apply = () => window.MercuryAPI.configure({
    baseUrl: baseUrl.trim(),
    apiKey: apiKey.trim(),
    bearer: bearer.trim(),
    autoFallback
  });
  const probe = async () => {
    apply();
    setProbing(true);
    await window.MercuryAPI.health();
    setProbing(false);
  };
  const useDev = () => {
    setBaseUrl("http://127.0.0.1:8000");
    window.MercuryAPI.configure({
      baseUrl: "http://127.0.0.1:8000"
    });
  };
  const disconnect = () => {
    setBaseUrl("");
    window.MercuryAPI.configure({
      baseUrl: ""
    });
  };
  const h = st.health;
  const reachLabel = !baseUrl ? "MOCK ONLY" : h.reachable === true ? "REACHABLE" : h.reachable === false ? "UNREACHABLE" : "UNKNOWN";
  const reachCls = h.reachable === true ? "ok" : h.reachable === false ? "bad" : "idle";
  return /*#__PURE__*/React.createElement("div", {
    className: "cn-grid wide"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-col-main"
  }, /*#__PURE__*/React.createElement("section", {
    className: "cn-panel"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-panel-head"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-lab"
  }, "CONNECTION"), /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-meta"
  }, "window.MercuryAPI")), /*#__PURE__*/React.createElement("div", {
    className: "cn-conn"
  }, /*#__PURE__*/React.createElement("label", {
    className: "cn-field"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-field-k"
  }, "Base URL"), /*#__PURE__*/React.createElement("input", {
    className: "cn-input",
    value: baseUrl,
    placeholder: "http://127.0.0.1:8000 (empty = mock only)",
    onChange: e => setBaseUrl(e.target.value)
  })), /*#__PURE__*/React.createElement("div", {
    className: "cn-field-row"
  }, /*#__PURE__*/React.createElement("label", {
    className: "cn-field"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-field-k"
  }, "X-API-Key ", /*#__PURE__*/React.createElement("span", {
    className: "cn-opt"
  }, "optional \xB7 detect")), /*#__PURE__*/React.createElement("input", {
    className: "cn-input",
    value: apiKey,
    placeholder: "anon if empty",
    onChange: e => setApiKey(e.target.value)
  })), /*#__PURE__*/React.createElement("label", {
    className: "cn-field"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-field-k"
  }, "Bearer JWT ", /*#__PURE__*/React.createElement("span", {
    className: "cn-opt"
  }, "required \xB7 batch")), /*#__PURE__*/React.createElement("input", {
    className: "cn-input",
    value: bearer,
    placeholder: "\u2014",
    onChange: e => setBearer(e.target.value)
  }))), /*#__PURE__*/React.createElement("label", {
    className: "cn-check"
  }, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: autoFallback,
    onChange: e => setAutoFallback(e.target.checked)
  }), " Fall back to faithful mock when the server is unreachable"), /*#__PURE__*/React.createElement("div", {
    className: "cn-conn-actions"
  }, /*#__PURE__*/React.createElement("button", {
    className: "mc-action primary",
    disabled: probing,
    onClick: probe
  }, probing ? "probing…" : "▸ Test connection"), /*#__PURE__*/React.createElement("button", {
    className: "mc-action",
    onClick: apply
  }, "Save"), /*#__PURE__*/React.createElement("button", {
    className: "mc-action",
    onClick: useDev
  }, "Use dev :8000"), /*#__PURE__*/React.createElement("button", {
    className: "mc-action",
    onClick: disconnect
  }, "Disconnect")))), /*#__PURE__*/React.createElement("section", {
    className: "cn-panel",
    style: {
      marginTop: "16px"
    }
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-panel-head"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-lab"
  }, "CORS & ENDPOINTS"), /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-meta"
  }, "contract")), /*#__PURE__*/React.createElement("div", {
    className: "cn-endpoints"
  }, [["POST", "/api/v1/detect/univariate", "anon ok"], ["POST", "/api/v1/detect/multivariate", "anon ok"], ["POST", "/api/v1/detect/fusion", "anon ok"], ["POST", "/api/v1/detect/three-r", "anon ok"], ["POST", "/api/v1/detect/neurosymbolic", "503 if module off"], ["POST", "/api/v1/batch/detect", "auth required"]].map(([m, p, n]) => /*#__PURE__*/React.createElement("div", {
    className: "cn-ep",
    key: p
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-ep-m"
  }, m), /*#__PURE__*/React.createElement("code", {
    className: "cn-ep-p"
  }, p), /*#__PURE__*/React.createElement("span", {
    className: "cn-ep-n"
  }, n)))), /*#__PURE__*/React.createElement("div", {
    className: "cn-conn-note"
  }, "Dev CORS allows localhost/127.0.0.1 on :3000/:8000/:8080. In prod set ", /*#__PURE__*/React.createElement("code", null, "MERCURY_CORS_ORIGINS"), ". Anonymous detect works; batch needs ", /*#__PURE__*/React.createElement("code", null, "X-API-Key"), "/", /*#__PURE__*/React.createElement("code", null, "Bearer"), "."))), /*#__PURE__*/React.createElement("div", {
    className: "cn-col-side"
  }, /*#__PURE__*/React.createElement("section", {
    className: "cn-panel"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-panel-head"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-lab"
  }, "HEALTH"), /*#__PURE__*/React.createElement("span", {
    className: "cn-reach " + reachCls
  }, reachLabel)), /*#__PURE__*/React.createElement("div", {
    className: "cn-health"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-h-row"
  }, /*#__PURE__*/React.createElement("span", null, "Endpoint"), /*#__PURE__*/React.createElement("code", null, baseUrl || "—")), /*#__PURE__*/React.createElement("div", {
    className: "cn-h-row"
  }, /*#__PURE__*/React.createElement("span", null, "OpenAPI version"), /*#__PURE__*/React.createElement("span", null, h.version || "—")), /*#__PURE__*/React.createElement("div", {
    className: "cn-h-row"
  }, /*#__PURE__*/React.createElement("span", null, "Latency"), /*#__PURE__*/React.createElement("span", null, h.latencyMs != null ? h.latencyMs + " ms" : "—")), /*#__PURE__*/React.createElement("div", {
    className: "cn-h-row"
  }, /*#__PURE__*/React.createElement("span", null, "Auth"), /*#__PURE__*/React.createElement("span", null, st.authed ? st.hasBearer ? "Bearer" : "API-Key" : "anonymous")), /*#__PURE__*/React.createElement("div", {
    className: "cn-h-row"
  }, /*#__PURE__*/React.createElement("span", null, "Last error"), /*#__PURE__*/React.createElement("span", {
    className: h.error ? "bad" : ""
  }, h.error || "none"))), st.lastMeta && /*#__PURE__*/React.createElement("div", {
    className: "cn-h-meta"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-dd-sec"
  }, "LAST RESPONSE META"), /*#__PURE__*/React.createElement("div", {
    className: "cn-h-row"
  }, /*#__PURE__*/React.createElement("span", null, "Correlation"), /*#__PURE__*/React.createElement("code", null, st.lastMeta.correlationId || "—")), /*#__PURE__*/React.createElement("div", {
    className: "cn-h-row"
  }, /*#__PURE__*/React.createElement("span", null, "Rate remaining"), /*#__PURE__*/React.createElement("span", null, st.lastMeta.rateLimit ? st.lastMeta.rateLimit.remaining + "/" + st.lastMeta.rateLimit.limit : "—"))), /*#__PURE__*/React.createElement("div", {
    className: "cn-conn-tip"
  }, "No server? Leave Base URL empty \u2014 the console runs on a ", /*#__PURE__*/React.createElement("b", null, "faithful mock"), " that returns the exact contract shapes, so every screen stays usable offline."))));
}

/* ---------------- LIVE DETECT COMPOSER ---------------- */
/* Score scales are method-dependent (API_REFERENCE.md): univariate z-score is in
   σ units (~2–5), multivariate is an L2/Mahalanobis distance, fusion & 3R are
   [0,1]. Only the z-score path has a documented derivation; everything else is
   labelled frontend-derived so a distance is never compared against a probability. */
function scaleInfoFor(endpoint, method, s) {
  if (endpoint === "univariate") {
    if (method === "zscore") return {
      unit: "σ · z-score",
      val: (2.0 + (1 - s) * 3.0).toFixed(2),
      expr: "2.0 + (1−s)×3.0",
      derived: false
    };
    if (method === "iqr") return {
      unit: "IQR fence ×k",
      val: (1.5 + (1 - s) * 1.5).toFixed(2),
      expr: "k = 1.5 + (1−s)×1.5",
      derived: true
    };
    return {
      unit: "score 0–1",
      val: "0.50",
      expr: "isolation_forest → deprecated alias for ensemble [0,1]",
      derived: true
    };
  }
  if (endpoint === "multivariate") return {
    unit: "L2 / Mahalanobis distance",
    val: (2.0 + (1 - s) * 3.0).toFixed(2),
    expr: "sensitivity-scaled distance",
    derived: true
  };
  if (endpoint === "fusion") return {
    unit: "ensemble 0–1",
    val: (0.85 - s * 0.35).toFixed(2),
    expr: "sensitivity-scaled",
    derived: true
  };
  return null;
}
const SAMPLE_DATA = {
  univariate: "0.41, 0.52, 0.47, 0.50, 10.0, 0.49, 0.51, 0.44",
  multivariate: "0.3,0.2\n0.4,0.2\n4.0,0.2\n0.3,0.2\n0.35,0.21",
  fusion: "0.4, 0.5, 0.45, 3.8, 0.5, 0.47",
  "three-r": "0.4, 0.5, 0.45, 0.6, 0.55, 0.5, 0.48"
};
function parseSeries(t) {
  return t.split(/[,\s]+/).map(Number).filter(x => !isNaN(x));
}
function parseMatrix(t) {
  return t.trim().split(/\n+/).map(r => r.split(/[,\s]+/).map(Number).filter(x => !isNaN(x))).filter(r => r.length);
}
function RunComposer({
  onResult
}) {
  const [endpoint, setEndpoint] = useStateL("univariate");
  const [raw, setRaw] = useStateL(SAMPLE_DATA.univariate);
  const [sensitivity, setSensitivity] = useStateL(0.5);
  const [method, setMethod] = useStateL("zscore");
  const [detectors, setDetectors] = useStateL(["statistical", "temporal", "dimensional"]);
  const [depth, setDepth] = useStateL(5);
  const [bands, setBands] = useStateL(8);
  const [eth, setEth] = useStateL(0.96);
  const [busy, setBusy] = useStateL(false);
  const switchEp = ep => {
    setEndpoint(ep);
    setRaw(SAMPLE_DATA[ep] || SAMPLE_DATA.univariate);
  };
  const scaleInfo = scaleInfoFor(endpoint, method, sensitivity);
  const fire = async () => {
    setBusy(true);
    let r;
    if (endpoint === "univariate") r = await window.MercuryAPI.detectUnivariate({
      data: parseSeries(raw),
      sensitivity,
      method
    });else if (endpoint === "multivariate") r = await window.MercuryAPI.detectMultivariate({
      data: parseMatrix(raw),
      sensitivity
    });else if (endpoint === "fusion") r = await window.MercuryAPI.detectFusion({
      data: parseSeries(raw),
      detectors,
      sensitivity
    });else r = await window.MercuryAPI.detectThreeR({
      data: parseSeries(raw),
      recursion_depth: depth,
      harmonic_bands: bands,
      ethical_threshold: eth
    });
    setBusy(false);
    if (onResult) onResult(r, endpoint, method);
  };
  const toggleDet = d => setDetectors(cur => cur.includes(d) ? cur.filter(x => x !== d) : cur.concat(d));
  return /*#__PURE__*/React.createElement("section", {
    className: "cn-panel cn-runcomposer"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-panel-head"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-lab"
  }, "LIVE REQUEST \xB7 /api/v1/detect"), /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-meta"
  }, "real call \xB7 injects to feed \u2192")), /*#__PURE__*/React.createElement("div", {
    className: "cn-compose"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-ep-tabs"
  }, ["univariate", "multivariate", "fusion", "three-r", "neurosymbolic"].map(ep => /*#__PURE__*/React.createElement("button", {
    key: ep,
    className: "cn-ep-tab" + (endpoint === ep ? " on" : "") + (ep === "neurosymbolic" ? " dim" : ""),
    disabled: ep === "neurosymbolic",
    onClick: () => switchEp(ep),
    title: ep === "neurosymbolic" ? "domain-dict input · use API directly" : ""
  }, ep))), /*#__PURE__*/React.createElement("label", {
    className: "cn-field"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-field-k"
  }, "data ", endpoint === "multivariate" ? "· rows of comma-separated features" : "· comma/space separated, ≥3"), /*#__PURE__*/React.createElement("textarea", {
    className: "cn-textarea",
    rows: endpoint === "multivariate" ? 5 : 3,
    value: raw,
    onChange: e => setRaw(e.target.value)
  })), scaleInfo && /*#__PURE__*/React.createElement("div", {
    className: "cn-sens"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-sens-top"
  }, /*#__PURE__*/React.createElement("span", null, "sensitivity"), /*#__PURE__*/React.createElement("span", {
    className: "cn-sens-v"
  }, sensitivity.toFixed(2)), /*#__PURE__*/React.createElement("span", {
    className: "cn-sens-der"
  }, "\u21D2 threshold ", /*#__PURE__*/React.createElement("b", null, scaleInfo.val), " ", /*#__PURE__*/React.createElement("span", {
    className: "cn-sens-formula"
  }, scaleInfo.derived ? "· frontend-derived" : "= " + scaleInfo.expr))), /*#__PURE__*/React.createElement("input", {
    className: "cn-range",
    type: "range",
    min: "0",
    max: "1",
    step: "0.01",
    value: sensitivity,
    onChange: e => setSensitivity(+e.target.value)
  }), /*#__PURE__*/React.createElement("div", {
    className: "cn-scale-note"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-scale-k"
  }, "SCALE"), " ", scaleInfo.unit, scaleInfo.derived ? /*#__PURE__*/React.createElement("span", {
    className: "cn-scale-tag"
  }, scaleInfo.expr) : null)), endpoint === "univariate" && /*#__PURE__*/React.createElement("div", {
    className: "cn-seg-row"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-seg-k"
  }, "method"), ["zscore", "iqr", "isolation_forest"].map(m => /*#__PURE__*/React.createElement("button", {
    key: m,
    className: "cn-seg-b" + (method === m ? " on" : ""),
    onClick: () => setMethod(m)
  }, m))), endpoint === "fusion" && /*#__PURE__*/React.createElement("div", {
    className: "cn-seg-row"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-seg-k"
  }, "detectors"), ["statistical", "temporal", "dimensional", "spectral"].map(d => /*#__PURE__*/React.createElement("button", {
    key: d,
    className: "cn-seg-b" + (detectors.includes(d) ? " on" : ""),
    onClick: () => toggleDet(d)
  }, d))), endpoint === "three-r" && /*#__PURE__*/React.createElement("div", {
    className: "cn-3r-controls"
  }, /*#__PURE__*/React.createElement("label", {
    className: "cn-mini"
  }, /*#__PURE__*/React.createElement("span", null, "recursion_depth ", /*#__PURE__*/React.createElement("b", null, depth)), /*#__PURE__*/React.createElement("input", {
    type: "range",
    min: "1",
    max: "10",
    step: "1",
    value: depth,
    onChange: e => setDepth(+e.target.value)
  })), /*#__PURE__*/React.createElement("label", {
    className: "cn-mini"
  }, /*#__PURE__*/React.createElement("span", null, "harmonic_bands ", /*#__PURE__*/React.createElement("b", null, bands)), /*#__PURE__*/React.createElement("input", {
    type: "range",
    min: "1",
    max: "32",
    step: "1",
    value: bands,
    onChange: e => setBands(+e.target.value)
  })), /*#__PURE__*/React.createElement("label", {
    className: "cn-mini"
  }, /*#__PURE__*/React.createElement("span", null, "ethical_threshold ", /*#__PURE__*/React.createElement("b", null, eth.toFixed(2))), /*#__PURE__*/React.createElement("input", {
    type: "range",
    min: "0.90",
    max: "0.99",
    step: "0.01",
    value: eth,
    onChange: e => setEth(+e.target.value)
  }))), /*#__PURE__*/React.createElement("button", {
    className: "mc-action primary block",
    disabled: busy,
    onClick: fire
  }, busy ? "calling…" : "▸ SEND REQUEST · inject to feed")));
}

/* ---------------- BATCH JOBS ---------------- */
function BatchView() {
  const st = useApiState();
  const [size, setSize] = useStateL(1200);
  const [method, setMethod] = useStateL("univariate");
  const [priority, setPriority] = useStateL("normal");
  const [callback, setCallback] = useStateL("");
  const [jobs, setJobs] = useStateL([]);
  const [results, setResults] = useStateL(null); // { job_id, ...results, source }
  const [busy, setBusy] = useStateL(false);
  const pollRef = useRefL(null);
  useEffectL(() => {
    pollRef.current = setInterval(async () => {
      setJobs(cur => cur);
      const updated = await Promise.all(jobs.map(async j => {
        if (j.status === "COMPLETED" || j.status === "FAILED" || j.status === "CANCELLED") return j;
        const r = await window.MercuryAPI.batchStatus(j.job_id);
        return r.data ? Object.assign({}, j, r.data, {
          source: r.source
        }) : j;
      }));
      setJobs(updated);
    }, 1200);
    return () => clearInterval(pollRef.current);
  }, [jobs.length]);
  const submit = async () => {
    setBusy(true);
    const data = Array.from({
      length: Math.min(size, 10000)
    }, () => [Math.random(), Math.random(), Math.random()]);
    const r = await window.MercuryAPI.batchSubmit({
      data,
      method,
      priority,
      callback_url: callback || undefined
    });
    if (r.data) setJobs(cur => [Object.assign({
      submitted: new Date().toLocaleTimeString("en-GB"),
      method,
      total: size,
      source: r.source
    }, r.data)].concat(cur));
    setBusy(false);
  };
  const cancelJob = async id => {
    const r = await window.MercuryAPI.batchCancel(id);
    if (r.data) setJobs(cur => cur.map(j => j.job_id === id ? Object.assign({}, j, {
      status: r.data.status
    }) : j));
    if (results && results.job_id === id) setResults(null);
  };
  const viewResults = async id => {
    const r = await window.MercuryAPI.batchResults(id);
    if (r.data) setResults(Object.assign({}, r.data, {
      source: r.source
    }));
  };
  const clearJobs = () => {
    setJobs([]);
    setResults(null);
  };
  const needsAuth = st.baseUrl && !st.authed;
  return /*#__PURE__*/React.createElement("div", {
    className: "cn-grid wide"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-col-main"
  }, /*#__PURE__*/React.createElement("section", {
    className: "cn-panel"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-panel-head"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-lab"
  }, "BATCH JOBS \xB7 LIFECYCLE"), /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-meta"
  }, jobs.length ? /*#__PURE__*/React.createElement("button", {
    className: "cn-link-btn",
    onClick: clearJobs
  }, "clear all") : "PENDING → RUNNING → COMPLETED")), jobs.length === 0 ? /*#__PURE__*/React.createElement("div", {
    className: "cn-resp-empty",
    style: {
      padding: "22px 16px"
    }
  }, "No jobs yet. Submit a batch (\u226410,000 samples) to see the async lifecycle: the job is created (202), then polled to completion.") : /*#__PURE__*/React.createElement("div", {
    className: "cn-joblist"
  }, jobs.map(j => {
    const st = (j.status || "PENDING").toLowerCase();
    const terminal = st === "completed" || st === "failed" || st === "cancelled";
    return /*#__PURE__*/React.createElement("div", {
      className: "cn-job st-" + st,
      key: j.job_id
    }, /*#__PURE__*/React.createElement("div", {
      className: "cn-job-top"
    }, /*#__PURE__*/React.createElement("code", {
      className: "cn-job-id"
    }, j.job_id), /*#__PURE__*/React.createElement("span", {
      className: "cn-job-st " + st
    }, j.status || "PENDING"), j.source && /*#__PURE__*/React.createElement("span", {
      className: "cn-srcbadge " + j.source
    }, j.source === "live" ? "● LIVE" : "○ MOCK")), /*#__PURE__*/React.createElement("div", {
      className: "cn-job-meta"
    }, j.method, " \xB7 ", j.total || j.total_samples || "—", " samples \xB7 ", j.submitted || ""), /*#__PURE__*/React.createElement("div", {
      className: "cn-job-track"
    }, /*#__PURE__*/React.createElement("span", {
      className: "cn-job-fill",
      style: {
        width: (j.progress_percentage || 0) + "%"
      }
    })), /*#__PURE__*/React.createElement("div", {
      className: "cn-job-bottom"
    }, /*#__PURE__*/React.createElement("span", {
      className: "cn-job-pct"
    }, Math.round(j.progress_percentage || 0), "%"), /*#__PURE__*/React.createElement("div", {
      className: "cn-job-actions"
    }, !terminal && /*#__PURE__*/React.createElement("button", {
      className: "cn-job-btn cancel",
      onClick: () => cancelJob(j.job_id)
    }, "\u2715 Cancel"), st === "completed" && /*#__PURE__*/React.createElement("button", {
      className: "cn-job-btn primary",
      onClick: () => viewResults(j.job_id)
    }, "\u25B8 View results"))));
  }))), results && /*#__PURE__*/React.createElement("section", {
    className: "cn-panel",
    style: {
      marginTop: "16px"
    }
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-panel-head"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-lab"
  }, "RESULTS \xB7 ", results.job_id), /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-meta"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-srcbadge " + results.source
  }, results.source === "live" ? "● LIVE" : "○ MOCK"))), /*#__PURE__*/React.createElement("div", {
    className: "cn-resp"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-resp-stats"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("span", {
    className: "cn-rk"
  }, "samples"), /*#__PURE__*/React.createElement("span", {
    className: "cn-rv"
  }, (results.total_samples || 0).toLocaleString())), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("span", {
    className: "cn-rk"
  }, "anomalies"), /*#__PURE__*/React.createElement("span", {
    className: "cn-rv anom"
  }, results.anomaly_count)), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("span", {
    className: "cn-rk"
  }, "rate"), /*#__PURE__*/React.createElement("span", {
    className: "cn-rv"
  }, ((results.anomaly_rate || 0) * 100).toFixed(1), "%")), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("span", {
    className: "cn-rk"
  }, "proc time"), /*#__PURE__*/React.createElement("span", {
    className: "cn-rv"
  }, results.processing_time_ms, "ms"))), results.top_anomalies && results.top_anomalies.length > 0 && /*#__PURE__*/React.createElement("div", {
    className: "cn-points"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-dd-sec",
    style: {
      margin: "4px 0 6px"
    }
  }, "TOP ANOMALIES"), results.top_anomalies.map((p, i) => /*#__PURE__*/React.createElement("div", {
    className: "cn-point",
    key: i
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-pt-i"
  }, "#", p.index), /*#__PURE__*/React.createElement("span", {
    className: "cn-pt-v"
  }, "z ", p.score), /*#__PURE__*/React.createElement("span", {
    className: "cn-num"
  }), /*#__PURE__*/React.createElement("span", {
    className: "cn-sev " + (p.severity === "critical" || p.severity === "high" ? "crit" : p.severity === "medium" ? "med" : "low")
  }, p.severity))))))), /*#__PURE__*/React.createElement("div", {
    className: "cn-col-side"
  }, /*#__PURE__*/React.createElement("section", {
    className: "cn-panel"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-panel-head"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-lab"
  }, "SUBMIT BATCH"), /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-meta"
  }, "POST /api/v1/batch/detect")), /*#__PURE__*/React.createElement("div", {
    className: "cn-conn"
  }, needsAuth && /*#__PURE__*/React.createElement("div", {
    className: "cn-err kind-auth"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-err-h"
  }, "401 \xB7 auth required"), /*#__PURE__*/React.createElement("div", {
    className: "cn-err-m"
  }, "Batch needs an X-API-Key or Bearer token. Add one in Connection \u2014 or leave Base URL empty to use the mock.")), /*#__PURE__*/React.createElement("label", {
    className: "cn-field"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-field-k"
  }, "samples ", /*#__PURE__*/React.createElement("b", null, size), " ", /*#__PURE__*/React.createElement("span", {
    className: "cn-opt"
  }, "\u2264 10,000")), /*#__PURE__*/React.createElement("input", {
    type: "range",
    min: "100",
    max: "10000",
    step: "100",
    value: size,
    onChange: e => setSize(+e.target.value),
    className: "cn-range"
  })), /*#__PURE__*/React.createElement("div", {
    className: "cn-seg-row"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-seg-k"
  }, "method"), ["univariate", "multivariate", "fusion"].map(m => /*#__PURE__*/React.createElement("button", {
    key: m,
    className: "cn-seg-b" + (method === m ? " on" : ""),
    onClick: () => setMethod(m)
  }, m))), /*#__PURE__*/React.createElement("div", {
    className: "cn-seg-row"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-seg-k"
  }, "priority"), ["low", "normal", "high"].map(p => /*#__PURE__*/React.createElement("button", {
    key: p,
    className: "cn-seg-b" + (priority === p ? " on" : ""),
    onClick: () => setPriority(p)
  }, p))), /*#__PURE__*/React.createElement("label", {
    className: "cn-field"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-field-k"
  }, "callback_url ", /*#__PURE__*/React.createElement("span", {
    className: "cn-opt"
  }, "HTTPS webhook \xB7 optional")), /*#__PURE__*/React.createElement("input", {
    className: "cn-input",
    value: callback,
    placeholder: "https://\u2026",
    onChange: e => setCallback(e.target.value)
  })), /*#__PURE__*/React.createElement("button", {
    className: "mc-action primary block",
    disabled: busy,
    onClick: submit
  }, busy ? "submitting…" : "▸ SUBMIT → 202")))));
}
Object.assign(window, {
  LiveStatus,
  ConnectionView,
  RunComposer,
  BatchView,
  scaleInfoFor
});
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/console/LiveViews.jsx", error: String((e && e.message) || e) }); }

// ui_kits/console/MercuryAPI.jsx
try { (() => {
/* ============================================================================
   MercuryAPI — live-calling client for the Mercury Agent Detection API.
   Implements the real contract (FastAPI rollup / mercury-openapi.json):
     POST /api/v1/detect/{univariate,multivariate,neurosymbolic,fusion,three-r}
     POST /api/v1/batch/detect  +  status / results / cancel
   Every call returns a normalized envelope:
     { ok, status, source: "live"|"mock", data, error, meta }
   where meta carries rate-limit headers, X-Correlation-ID, Retry-After.

   When no base URL is set, or the server is unreachable / CORS-blocked, the
   client transparently falls back to a FAITHFUL MOCK that returns the exact
   response *shape* the live API would — so the console is fully usable offline
   and identical in structure to production. The UI always shows which path
   served the response (LIVE vs MOCK).
   ============================================================================ */
(function () {
  const LS_KEY = "mercury_api_config_v1";
  const DEFAULTS = {
    baseUrl: "",
    // empty ⇒ mock-only until the user connects
    apiKey: "",
    // X-API-Key
    bearer: "",
    // Authorization: Bearer <jwt>
    timeoutMs: 8000,
    autoFallback: true // fall back to mock on network/CORS failure
  };
  let cfg = load();
  function load() {
    try {
      return Object.assign({}, DEFAULTS, JSON.parse(localStorage.getItem(LS_KEY) || "{}"));
    } catch (e) {
      return Object.assign({}, DEFAULTS);
    }
  }
  function save() {
    try {
      localStorage.setItem(LS_KEY, JSON.stringify(cfg));
    } catch (e) {}
  }

  // simple pub/sub so the UI can react to config / health changes
  const subs = [];
  function emit() {
    subs.forEach(f => {
      try {
        f(getState());
      } catch (e) {}
    });
  }
  let health = {
    reachable: null,
    checkedAt: 0,
    latencyMs: null,
    version: null,
    error: null
  };
  let lastMeta = null;
  function getState() {
    return {
      baseUrl: cfg.baseUrl,
      hasKey: !!cfg.apiKey,
      hasBearer: !!cfg.bearer,
      authed: !!(cfg.apiKey || cfg.bearer),
      autoFallback: cfg.autoFallback,
      health: Object.assign({}, health),
      lastMeta
    };
  }
  function headers(json) {
    const h = {};
    if (json) h["Content-Type"] = "application/json";
    if (cfg.apiKey) h["X-API-Key"] = cfg.apiKey;
    if (cfg.bearer) h["Authorization"] = "Bearer " + cfg.bearer;
    return h;
  }
  function readMeta(res) {
    if (!res || !res.headers) return null;
    const g = k => res.headers.get(k);
    const m = {
      correlationId: g("X-Correlation-ID"),
      rateLimit: {
        limit: num(g("X-RateLimit-Limit")),
        remaining: num(g("X-RateLimit-Remaining")),
        reset: num(g("X-RateLimit-Reset"))
      },
      retryAfter: num(g("Retry-After")),
      status: res.status
    };
    lastMeta = m;
    emit();
    return m;
  }
  function num(x) {
    return x == null ? null : isNaN(+x) ? x : +x;
  }

  // ---- core fetch with timeout + faithful-mock fallback -------------------
  async function call(path, body, mockFn, opts) {
    opts = opts || {};
    const t0 = performance.now();
    if (!cfg.baseUrl) {
      return mock(mockFn, body, "no base URL configured");
    }
    const ctrl = new AbortController();
    const to = setTimeout(() => ctrl.abort(), cfg.timeoutMs);
    try {
      const res = await fetch(cfg.baseUrl.replace(/\/$/, "") + path, {
        method: "POST",
        headers: headers(true),
        body: JSON.stringify(body),
        signal: ctrl.signal
      });
      clearTimeout(to);
      const meta = readMeta(res);
      const latency = Math.round(performance.now() - t0);
      let payload = null;
      try {
        payload = await res.json();
      } catch (e) {
        payload = null;
      }
      health = {
        reachable: true,
        checkedAt: Date.now(),
        latencyMs: latency,
        version: health.version,
        error: null
      };
      emit();
      if (!res.ok) {
        // real error contract: body is { detail: string | object | array }
        return {
          ok: false,
          status: res.status,
          source: "live",
          data: null,
          error: normalizeError(res.status, payload),
          meta
        };
      }
      return {
        ok: true,
        status: res.status,
        source: "live",
        data: payload,
        error: null,
        meta,
        latencyMs: latency
      };
    } catch (e) {
      clearTimeout(to);
      health = {
        reachable: false,
        checkedAt: Date.now(),
        latencyMs: null,
        version: null,
        error: e.name === "AbortError" ? "timeout" : "unreachable / CORS"
      };
      emit();
      if (cfg.autoFallback) return mock(mockFn, body, health.error);
      return {
        ok: false,
        status: 0,
        source: "live",
        data: null,
        error: {
          kind: "network",
          message: "Server " + health.error + " — " + cfg.baseUrl
        },
        meta: null
      };
    }
  }
  function mock(mockFn, body, reason) {
    const data = mockFn(body || {});
    // synthesize plausible rate-limit + correlation meta so the UI is identical
    lastMeta = {
      correlationId: "mock-" + Math.random().toString(36).slice(2, 10),
      rateLimit: {
        limit: 120,
        remaining: 100 + Math.floor(Math.random() * 19),
        reset: Math.floor(Date.now() / 1000) + 47
      },
      retryAfter: null,
      status: 200
    };
    emit();
    return {
      ok: true,
      status: 200,
      source: "mock",
      data,
      error: null,
      meta: lastMeta,
      reason
    };
  }
  function normalizeError(status, payload) {
    const detail = payload && payload.detail;
    let message = "Request failed";
    if (typeof detail === "string") message = detail;else if (Array.isArray(detail)) message = detail.map(d => (d.loc ? d.loc.join(".") + ": " : "") + (d.msg || "")).join(" · ");else if (detail && detail.message) message = detail.message;
    const KIND = {
      400: "validation",
      401: "auth",
      422: "body",
      429: "rate-limit",
      500: "server",
      503: "unavailable"
    };
    return {
      kind: KIND[status] || "error",
      status,
      message,
      detail
    };
  }

  // ====================== FAITHFUL MOCK RESPONSE BUILDERS ==================
  // Each mirrors the documented response shape exactly.
  function rnd(a, b) {
    return a + Math.random() * (b - a);
  }
  function r2(x) {
    return Math.round(x * 100) / 100;
  }
  function sevOf(z) {
    return z >= 4 ? "critical" : z >= 3 ? "high" : z >= 2.5 ? "medium" : "low";
  }
  function mockUnivariate(b) {
    const data = b.data && b.data.length ? b.data : [0.4, 0.5, 0.45, 10.0, 0.5];
    const sens = b.sensitivity == null ? 0.5 : b.sensitivity;
    const threshold = r2(2.0 + (1.0 - sens) * 3.0);
    const mean = data.reduce((a, c) => a + c, 0) / data.length;
    const sd = Math.sqrt(data.reduce((a, c) => a + (c - mean) * (c - mean), 0) / data.length) || 1;
    const scores = data.map(v => r2(Math.abs((v - mean) / sd)));
    const anomalies = scores.map(s => s >= threshold);
    const anomaly_points = [];
    data.forEach((v, i) => {
      if (anomalies[i]) anomaly_points.push({
        index: i,
        value: v,
        score: scores[i],
        severity: sevOf(scores[i])
      });
    });
    const ac = anomalies.filter(Boolean).length;
    return {
      anomalies,
      scores,
      anomaly_points,
      method: b.method || "zscore",
      threshold,
      summary: {
        total_points: data.length,
        anomaly_count: ac,
        anomaly_rate: r2(ac / data.length)
      }
    };
  }
  function mockMultivariate(b) {
    const data = b.data && b.data.length ? b.data : [[0.3, 0.2], [0.4, 0.2], [4.0, 0.2], [0.3, 0.2]];
    const sens = b.sensitivity == null ? 0.5 : b.sensitivity;
    const threshold = r2(2.0 + (1.0 - sens) * 3.0);
    const cols = data[0].length;
    const feats = b.features && b.features.length === cols ? b.features : Array.from({
      length: cols
    }, (_, i) => "f" + i);
    const means = feats.map((_, c) => data.reduce((a, r) => a + r[c], 0) / data.length);
    const sds = feats.map((_, c) => Math.sqrt(data.reduce((a, r) => a + Math.pow(r[c] - means[c], 2), 0) / data.length) || 1);
    const fc = {};
    feats.forEach(f => fc[f] = []);
    const scores = data.map(row => {
      let sq = 0;
      row.forEach((v, c) => {
        const z = Math.abs((v - means[c]) / sds[c]);
        fc[feats[c]].push(r2(z));
        sq += z * z;
      });
      return r2(Math.sqrt(sq));
    });
    const anomalies = scores.map(s => s >= threshold);
    const ac = anomalies.filter(Boolean).length;
    return {
      anomalies,
      scores,
      feature_contributions: b.correlate_features === false ? null : fc,
      method: "multivariate",
      threshold,
      features: feats,
      summary: {
        total_points: data.length,
        anomaly_count: ac,
        anomaly_rate: r2(ac / data.length)
      }
    };
  }
  function mockFusion(b) {
    const dets = b.detectors && b.detectors.length ? b.detectors : ["statistical", "temporal", "dimensional"];
    const sens = b.sensitivity == null ? 0.5 : b.sensitivity;
    const threshold = r2(2.0 + (1 - sens) * 3.0);
    const ds = {};
    dets.forEach(d => ds[d] = r2(rnd(0.5, 4.5)));
    const fused = r2(Object.values(ds).reduce((a, c) => a + c, 0) / dets.length * 1.15);
    const sum = Object.values(ds).reduce((a, c) => a + c, 0) || 1;
    const dc = {};
    dets.forEach(d => dc[d] = r2(ds[d] / sum));
    return {
      is_anomaly: fused >= threshold,
      fused_score: fused,
      confidence: r2(rnd(0.78, 0.95)),
      detector_scores: ds,
      detector_contributions: b.return_contributions === false ? {} : dc,
      threshold,
      explanation: "Fused score " + fused.toFixed(3) + " from: " + dets.join(", ") + "."
    };
  }
  function mockThreeR(b) {
    const depth = b.recursion_depth || 5,
      bands = b.harmonic_bands || 8,
      eth = b.ethical_threshold || 0.96;
    const rec = r2(rnd(0.4, 0.95)),
      res = r2(rnd(0.4, 0.95)),
      opt = r2(rnd(0.4, 0.95));
    const phi = 1.6180339887;
    const fusion = r2(rec * 0.4 + res * 0.35 + opt * 0.25);
    return {
      fusion_score: fusion,
      recursion_score: rec,
      resonance_score: res,
      optimization_score: opt,
      ethical_scaling: r2(Math.pow(eth, phi)),
      lyapunov_bound: 0.25,
      is_stable: true,
      weights: {
        recursion: 0.4,
        resonance: 0.35,
        optimization: 0.25
      },
      harmonic_analysis: {
        bands: bands,
        depth: depth,
        dominant_band: 1 + Math.floor(rnd(0, bands))
      }
    };
  }
  function mockNeurosymbolic(b) {
    const data = b.data && b.data.length ? b.data : [{}];
    const includeExp = b.include_explanations !== false;
    const overall = r2(rnd(0.55, 0.92));
    const scores = includeExp ? data.slice(0, Math.min(5, data.length)).map((_, i) => ({
      score_id: "ns-" + i + "-" + Math.random().toString(36).slice(2, 7),
      anomaly_score: r2(rnd(0.4, 0.95)),
      neural_score: r2(rnd(0.4, 0.9)),
      symbolic_score: r2(rnd(0.5, 0.98)),
      confidence: r2(rnd(0.75, 0.95)),
      category: ["spectral", "kinematic", "geometric"][i % 3],
      is_anomaly: Math.random() < 0.5,
      explanation: "Neural pattern reinforced by symbolic rule set."
    })) : [];
    const permitted = overall < 0.99; // ethical gate framing
    return {
      overall_score: overall,
      overall_confidence: r2(rnd(0.82, 0.95)),
      neural_contribution: 0.55,
      symbolic_contribution: 0.45,
      patterns_detected: 1 + Math.floor(rnd(0, 5)),
      rules_fired: 3 + Math.floor(rnd(0, 9)),
      anomaly_scores: scores,
      decision: permitted ? "permit" : "withhold",
      explanation: permitted ? "Action cleared ethical gate." : "Withheld — benevolence below threshold.",
      strategy: b.fusion_strategy || "confidence_weighted",
      audit_trail: includeExp ? [] : []
    };
  }

  // batch mock: an in-memory job store
  const jobs = {};
  function mockBatchSubmit(b) {
    const id = "job-" + Math.random().toString(36).slice(2, 10);
    jobs[id] = {
      job_id: id,
      status: "PENDING",
      progress_percentage: 0,
      method: b.method || "univariate",
      total: (b.data || []).length || 1200,
      _t: Date.now()
    };
    return {
      job_id: id,
      status: "PENDING",
      progress_percentage: 0
    };
  }
  function advanceJob(id) {
    const j = jobs[id];
    if (!j) return null;
    if (j.status === "CANCELLED" || j.status === "FAILED") return j;
    const el = (Date.now() - j._t) / 1000;
    if (el < 1) {
      j.status = "PENDING";
      j.progress_percentage = 0;
    } else if (el < 7) {
      j.status = "RUNNING";
      j.progress_percentage = Math.min(99, Math.round((el - 1) / 6 * 100));
    } else {
      j.status = "COMPLETED";
      j.progress_percentage = 100;
    }
    return j;
  }
  function mockBatchResults(id) {
    const j = jobs[id];
    const total = j ? j.total : 1200;
    const ac = Math.round(total * rnd(0.04, 0.12));
    const points = [];
    for (let i = 0; i < Math.min(8, ac); i++) {
      const sc = r2(rnd(2.5, 5.0));
      points.push({
        index: Math.floor(rnd(0, total)),
        score: sc,
        severity: sevOf(sc)
      });
    }
    points.sort((a, b) => b.score - a.score);
    return {
      job_id: id,
      status: "COMPLETED",
      total_samples: total,
      anomaly_count: ac,
      anomaly_rate: r2(ac / total),
      processing_time_ms: Math.round(rnd(800, 4200)),
      top_anomalies: points
    };
  }
  function mockBatchCancel(id) {
    const j = jobs[id];
    if (j && j.status !== "COMPLETED") {
      j.status = "CANCELLED";
    }
    return {
      job_id: id,
      status: j ? j.status : "CANCELLED"
    };
  }

  // The shipped server returns a LOWERCASE status enum (pending/running/...);
  // the UI + mock speak UPPERCASE. Normalize live job payloads so either path
  // renders identically. (api/routes/batch.py :: JobStatus)
  function normJob(data) {
    if (data && typeof data.status === "string") data.status = data.status.toUpperCase();
    return data;
  }

  // ---- generic GET with faithful-mock fallback (models / export / voice reads) --
  async function callGET(path, mockFn, mockArg) {
    if (!cfg.baseUrl) return {
      ok: true,
      status: 200,
      source: "mock",
      data: mockFn(mockArg),
      meta: lastMeta
    };
    try {
      const res = await fetch(cfg.baseUrl.replace(/\/$/, "") + path, {
        headers: headers(false)
      });
      const data = await res.json().catch(() => null);
      return {
        ok: res.ok,
        status: res.status,
        source: "live",
        data,
        meta: readMeta(res)
      };
    } catch (e) {
      if (cfg.autoFallback) return {
        ok: true,
        status: 200,
        source: "mock",
        data: mockFn(mockArg)
      };
      return {
        ok: false,
        status: 0,
        source: "live",
        data: null,
        error: {
          kind: "network",
          message: "unreachable"
        }
      };
    }
  }

  // ====================== MODEL REGISTRY (api/routes/models.py) ============
  // Shapes mirror ModelResponse + ModelMetrics; enums match ModelType/Status/Framework.
  const MODELS = [{
    model_id: "mdl-fusion",
    name: "Ω-AvA Fusion Ensemble",
    model_type: "fusion",
    framework: "pytorch",
    status: "deployed",
    current_version: "1.7.0",
    version_count: 6,
    owner_id: "steel-sa",
    endpoint: "/api/v1/detect/fusion",
    metrics: {
      auc_roc: 0.8466,
      f1_score: 0.78,
      latency_p50_ms: 9,
      latency_p99_ms: 31,
      throughput_rps: 1180
    }
  }, {
    model_id: "mdl-stat",
    name: "Mercury Statistical 3-Component",
    model_type: "statistical",
    framework: "custom",
    status: "deployed",
    current_version: "1.7.0",
    version_count: 4,
    owner_id: "steel-sa",
    endpoint: "/api/v1/detect/univariate",
    metrics: {
      auc_roc: 0.806,
      f1_score: 0.74,
      latency_p50_ms: 3,
      latency_p99_ms: 11,
      throughput_rps: 5200
    }
  }, {
    model_id: "mdl-ns",
    name: "Neurosymbolic Fusion Engine",
    model_type: "neurosymbolic",
    framework: "pytorch",
    status: "staged",
    current_version: "0.9.2",
    version_count: 2,
    owner_id: "steel-sa",
    endpoint: "/api/v1/detect/neurosymbolic",
    metrics: {
      auc_roc: 0.79,
      f1_score: 0.71,
      latency_p50_ms: 48,
      latency_p99_ms: 132,
      throughput_rps: 140
    }
  }, {
    model_id: "mdl-temporal",
    name: "Temporal LSTM-AE",
    model_type: "lstm_autoencoder",
    framework: "pytorch",
    status: "draft",
    current_version: null,
    version_count: 1,
    owner_id: "steel-sa",
    endpoint: "/api/v1/detect/multivariate",
    metrics: {
      auc_roc: 0.74,
      f1_score: 0.66,
      latency_p50_ms: 22,
      latency_p99_ms: 70,
      throughput_rps: 420
    }
  }, {
    model_id: "mdl-iso",
    name: "Isolation Forest (baseline)",
    model_type: "isolation_forest",
    framework: "sklearn",
    status: "deprecated",
    current_version: "0.4.0",
    version_count: 3,
    owner_id: "steel-sa",
    endpoint: "/api/v1/detect/univariate",
    metrics: {
      auc_roc: 0.68,
      f1_score: 0.59,
      latency_p50_ms: 6,
      latency_p99_ms: 19,
      throughput_rps: 900
    }
  }];
  function mockModelsList() {
    return MODELS.map(m => Object.assign({}, m));
  }
  function mockModelDeploy(arg) {
    const m = MODELS.find(x => x.model_id === arg.model_id);
    if (m) {
      m.status = "deployed";
      if (arg.version_number) m.current_version = arg.version_number;
    }
    return {
      model_id: arg.model_id,
      status: "deployed",
      version_number: m ? m.current_version : null,
      replicas: arg.replicas || 1,
      canary_percentage: arg.canary_percentage || 0
    };
  }
  function mockModelRegister(b) {
    const id = "mdl-" + Math.random().toString(36).slice(2, 8);
    const m = {
      model_id: id,
      name: b.name || "Untitled",
      model_type: b.model_type || "fusion",
      framework: "pytorch",
      status: "draft",
      current_version: null,
      version_count: 0,
      owner_id: "steel-sa",
      endpoint: null,
      metrics: {
        auc_roc: 0,
        f1_score: 0,
        latency_p50_ms: 0,
        latency_p99_ms: 0,
        throughput_rps: 0
      }
    };
    MODELS.push(m);
    return Object.assign({}, m);
  }

  // ====================== AUDIT TRAIL (shared: override loop ↔ export) ======
  const auditLog = [{
    id: "aud-seed1",
    ts: new Date(Date.now() - 5 * 864e5).toISOString(),
    actor: "system",
    action: "withhold",
    target: "neurosymbolic",
    justification: "η 0.984 < 0.99 — benevolence gate",
    corr: "corr-seed-a1"
  }, {
    id: "aud-seed2",
    ts: new Date(Date.now() - 2 * 864e5).toISOString(),
    actor: "analyst@steel-sa",
    action: "deploy",
    target: "mdl-fusion",
    justification: "promoted v1.7.0 to production",
    corr: "corr-seed-b2"
  }];

  // ====================== EXPORT (api/routes/export.py) ====================
  function mockExportSummary() {
    return {
      total_detections: 18742,
      total_audit_logs: auditLog.length + 324,
      oldest_detection: "2026-01-06T02:21:17Z",
      newest_detection: new Date().toISOString(),
      oldest_audit_log: auditLog[0] ? auditLog[0].ts : null,
      newest_audit_log: new Date().toISOString(),
      available_methods: ["zscore", "iqr", "isolation_forest", "multivariate", "fusion", "neurosymbolic", "three-r"],
      available_actions: ["detect", "withhold", "override", "deploy", "export"]
    };
  }
  const exportJobs = {};
  function mockExportSubmit(b) {
    const id = "exp-" + Math.random().toString(36).slice(2, 10);
    exportJobs[id] = {
      job_id: id,
      status: "PENDING",
      export_type: b.export_type || "detections",
      format: b.format || "json",
      created_at: Date.now(),
      record_count: 0,
      file_size_bytes: 0,
      download_url: null,
      _t: Date.now(),
      _limit: b.limit || 10000
    };
    return {
      job_id: id,
      status: "PENDING",
      export_type: exportJobs[id].export_type,
      format: exportJobs[id].format
    };
  }
  function advanceExport(id) {
    const j = exportJobs[id];
    if (!j) return null;
    const el = (Date.now() - j._t) / 1000;
    if (el < 1) {
      j.status = "PENDING";
    } else if (el < 4) {
      j.status = "RUNNING";
      j.record_count = Math.round(j._limit * Math.min(0.99, (el - 1) / 3));
    } else {
      j.status = "COMPLETED";
      j.record_count = j._limit;
      j.file_size_bytes = j._limit * 220;
      j.download_url = "/api/v1/export/download/" + id;
      j.completed_at = Date.now();
    }
    return Object.assign({}, j);
  }

  // ====================== VOICE (api/voice.py) =============================
  function classifyIntent(s) {
    s = (s || "").toLowerCase();
    if (/status|health|online|up\b/.test(s)) return "status";
    if (/anomal|detect|recent|flag/.test(s)) return "detection_query";
    if (/ethic|benevolen|withh|gate|\u03c3|sigma/.test(s)) return "ethics_query";
    if (/explain|why|reason/.test(s)) return "explanation";
    if (/quantum|crypto|pqc/.test(s)) return "security_query";
    return "general";
  }
  function mockVoiceSpeak(b) {
    const intent = classifyIntent(b.message);
    const replies = {
      status: "Mercury Agent operational. 3-component ensemble active; benevolence gate enforced at η ≥ 0.99. Mean ROC-AUC 0.8466 / median 0.9100 across 65 datasets (ADBench-comparable subset 0.8180) — research-grade, community-tested, not externally audited.",
      detection_query: "In the current rolling window the ensemble flagged a minority of events as ANOMALY. I report the measured score and its caveat together — none cleared without passing the ethical gate first.",
      ethics_query: "Every action clears two hard gates: Benevolence ≥ 0.99 and σ_Immutable. On a violation I raise EthicalConstraintViolationError and withhold the action — there is no advisory mode.",
      explanation: "I explain a detection by its feature attribution, the counterfactual that would flip it to NORMAL, and the gate verdict that governed the action — measured, not asserted.",
      security_query: "Artifacts I sign are sealed by the AMA Cryptography package: hybrid Ed25519 + ML-DSA-65, ~128-bit classical / ~192-bit quantum. Community-tested, not externally audited.",
      general: "I reason over this console — detections, the ensemble, the ethical gates, and compliance. Ask about system status, recent anomalies, or the benevolence gate."
    };
    return {
      message: replies[intent],
      confidence: r2(rnd(0.84, 0.96)),
      intent,
      sources: intent === "status" ? ["system_status", "detection_log"] : intent === "ethics_query" ? ["ethical_bounding", "audit_trail"] : ["detection_log"],
      timestamp: Date.now() / 1000,
      metadata: {
        session_id: b.session_id || "sess-" + Math.random().toString(36).slice(2, 8)
      }
    };
  }
  function mockVoiceNarrate(b) {
    const r = b.detection_result || {};
    const anom = r.anomaly_detected || r.is_anomaly || r.isAnom;
    const score = r.anomaly_score != null ? r.anomaly_score : r.score != null ? r.score : 0;
    const benev = r.benev != null ? r.benev : null;
    const dom = b.domain || r.domain || "the stream";
    const sentence = anom ? `ANOMALY on ${dom}. Score ${(+score).toFixed(2)}${benev != null ? `, benevolence η ${(+benev).toFixed(3)}` : ""} — ${benev != null && benev < 0.99 ? "action WITHHELD; the ethical gate did not clear." : "within the ethical bound; flag raised for review."}` : `NORMAL on ${dom}. Score ${(+score).toFixed(2)} sits below threshold; no action taken.`;
    return {
      narration: sentence,
      confidence: r2(rnd(0.82, 0.95)),
      severity: anom ? "elevated" : "nominal",
      domain: dom
    };
  }

  // ====================== PUBLIC API ======================================
  const API = {
    configure(patch) {
      cfg = Object.assign({}, cfg, patch || {});
      save();
      emit();
      return getState();
    },
    getConfig() {
      return Object.assign({}, cfg);
    },
    getState,
    onChange(fn) {
      subs.push(fn);
      return () => {
        const i = subs.indexOf(fn);
        if (i >= 0) subs.splice(i, 1);
      };
    },
    async health() {
      if (!cfg.baseUrl) {
        health = {
          reachable: null,
          checkedAt: Date.now(),
          latencyMs: null,
          version: null,
          error: "not configured"
        };
        emit();
        return getState().health;
      }
      const t0 = performance.now();
      const ctrl = new AbortController();
      const to = setTimeout(() => ctrl.abort(), cfg.timeoutMs);
      try {
        const res = await fetch(cfg.baseUrl.replace(/\/$/, "") + "/openapi.json", {
          signal: ctrl.signal,
          headers: headers(false)
        });
        clearTimeout(to);
        const spec = await res.json().catch(() => null);
        health = {
          reachable: res.ok,
          checkedAt: Date.now(),
          latencyMs: Math.round(performance.now() - t0),
          version: spec && spec.info ? spec.info.version : null,
          error: res.ok ? null : "HTTP " + res.status
        };
      } catch (e) {
        clearTimeout(to);
        health = {
          reachable: false,
          checkedAt: Date.now(),
          latencyMs: null,
          version: null,
          error: e.name === "AbortError" ? "timeout" : "unreachable / CORS"
        };
      }
      emit();
      return getState().health;
    },
    detectUnivariate(req) {
      return call("/api/v1/detect/univariate", req, mockUnivariate);
    },
    detectMultivariate(req) {
      return call("/api/v1/detect/multivariate", req, mockMultivariate);
    },
    detectFusion(req) {
      return call("/api/v1/detect/fusion", req, mockFusion);
    },
    detectThreeR(req) {
      return call("/api/v1/detect/three-r", req, mockThreeR);
    },
    detectNeurosymbolic(req) {
      return call("/api/v1/detect/neurosymbolic", req, mockNeurosymbolic);
    },
    async batchSubmit(req) {
      const r = await call("/api/v1/batch/detect", req, mockBatchSubmit);
      if (r && r.data) normJob(r.data);
      return r;
    },
    async batchStatus(id) {
      if (!cfg.baseUrl) {
        const j = advanceJob(id);
        return {
          ok: !!j,
          status: 200,
          source: "mock",
          data: j,
          meta: lastMeta
        };
      }
      // live status is GET /api/v1/batch/jobs/{id} (matches submit's poll_url)
      try {
        const res = await fetch(cfg.baseUrl.replace(/\/$/, "") + "/api/v1/batch/jobs/" + id, {
          headers: headers(false)
        });
        const data = normJob(await res.json().catch(() => null));
        return {
          ok: res.ok,
          status: res.status,
          source: "live",
          data,
          meta: readMeta(res)
        };
      } catch (e) {
        if (cfg.autoFallback) {
          const j = advanceJob(id);
          return {
            ok: !!j,
            status: 200,
            source: "mock",
            data: j
          };
        }
        return {
          ok: false,
          status: 0,
          source: "live",
          data: null,
          error: {
            kind: "network",
            message: "unreachable"
          }
        };
      }
    },
    async batchResults(id) {
      if (!cfg.baseUrl) return {
        ok: true,
        status: 200,
        source: "mock",
        data: mockBatchResults(id),
        meta: lastMeta
      };
      try {
        const res = await fetch(cfg.baseUrl.replace(/\/$/, "") + "/api/v1/batch/jobs/" + id + "/results", {
          headers: headers(false)
        });
        const data = normJob(await res.json().catch(() => null));
        return {
          ok: res.ok,
          status: res.status,
          source: "live",
          data,
          meta: readMeta(res)
        };
      } catch (e) {
        if (cfg.autoFallback) return {
          ok: true,
          status: 200,
          source: "mock",
          data: mockBatchResults(id)
        };
        return {
          ok: false,
          status: 0,
          source: "live",
          data: null,
          error: {
            kind: "network",
            message: "unreachable"
          }
        };
      }
    },
    async batchCancel(id) {
      if (!cfg.baseUrl) return {
        ok: true,
        status: 200,
        source: "mock",
        data: mockBatchCancel(id),
        meta: lastMeta
      };
      try {
        // cancel is DELETE /api/v1/batch/jobs/{id} -> 204 No Content (empty body)
        const res = await fetch(cfg.baseUrl.replace(/\/$/, "") + "/api/v1/batch/jobs/" + id, {
          method: "DELETE",
          headers: headers(false)
        });
        const data = normJob(await res.json().catch(() => null)) || {
          job_id: id,
          status: "CANCELLED"
        };
        return {
          ok: res.ok,
          status: res.status,
          source: "live",
          data,
          meta: readMeta(res)
        };
      } catch (e) {
        if (cfg.autoFallback) return {
          ok: true,
          status: 200,
          source: "mock",
          data: mockBatchCancel(id)
        };
        return {
          ok: false,
          status: 0,
          source: "live",
          data: null,
          error: {
            kind: "network",
            message: "unreachable"
          }
        };
      }
    },
    // ---- model registry (api/routes/models.py) ----
    modelsList() {
      return callGET("/api/v1/models", mockModelsList);
    },
    async modelDeploy(model_id, version_number, opts) {
      const body = Object.assign({
        model_id,
        version_number
      }, opts || {});
      if (!cfg.baseUrl) return {
        ok: true,
        status: 200,
        source: "mock",
        data: mockModelDeploy(body),
        meta: lastMeta
      };
      try {
        const res = await fetch(cfg.baseUrl.replace(/\/$/, "") + "/api/v1/models/" + model_id + "/deploy", {
          method: "POST",
          headers: headers(true),
          body: JSON.stringify(body)
        });
        const data = await res.json().catch(() => null);
        return {
          ok: res.ok,
          status: res.status,
          source: "live",
          data,
          meta: readMeta(res)
        };
      } catch (e) {
        if (cfg.autoFallback) return {
          ok: true,
          status: 200,
          source: "mock",
          data: mockModelDeploy(body)
        };
        return {
          ok: false,
          status: 0,
          source: "live",
          data: null,
          error: {
            kind: "network",
            message: "unreachable"
          }
        };
      }
    },
    modelRegister(req) {
      return call("/api/v1/models", req, mockModelRegister);
    },
    // ---- export (api/routes/export.py) ----
    exportSummary() {
      return callGET("/api/v1/export/summary", mockExportSummary);
    },
    exportSubmit(req) {
      return call("/api/v1/export", req, mockExportSubmit);
    },
    async exportStatus(id) {
      if (!cfg.baseUrl) {
        const j = advanceExport(id);
        return {
          ok: !!j,
          status: 200,
          source: "mock",
          data: j,
          meta: lastMeta
        };
      }
      try {
        const res = await fetch(cfg.baseUrl.replace(/\/$/, "") + "/api/v1/export/" + id, {
          headers: headers(false)
        });
        const data = await res.json().catch(() => null);
        return {
          ok: res.ok,
          status: res.status,
          source: "live",
          data,
          meta: readMeta(res)
        };
      } catch (e) {
        if (cfg.autoFallback) {
          const j = advanceExport(id);
          return {
            ok: !!j,
            status: 200,
            source: "mock",
            data: j
          };
        }
        return {
          ok: false,
          status: 0,
          source: "live",
          data: null,
          error: {
            kind: "network",
            message: "unreachable"
          }
        };
      }
    },
    // ---- voice (api/voice.py) ----
    voiceSpeak(req) {
      return call("/api/v1/voice/speak", req, mockVoiceSpeak);
    },
    voiceNarrate(req) {
      return call("/api/v1/voice/narrate", req, mockVoiceNarrate);
    },
    // ---- audit trail (shared by override loop + export) ----
    auditList() {
      return auditLog.slice().reverse();
    },
    auditAppend(entry) {
      const e = Object.assign({
        id: "aud-" + Math.random().toString(36).slice(2, 9),
        ts: new Date().toISOString()
      }, entry);
      auditLog.push(e);
      emit();
      return e;
    },
    // expose builders for tests / previews
    _mock: {
      mockUnivariate,
      mockMultivariate,
      mockFusion,
      mockThreeR,
      mockNeurosymbolic
    }
  };
  window.MercuryAPI = API;
})();
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/console/MercuryAPI.jsx", error: String((e && e.message) || e) }); }

// ui_kits/console/Panels.jsx
try { (() => {
/* Mercury Console — data panels: stat cards, detection feed, ensemble meters,
   ethics gates, domain coverage. */

function StatCards({
  rows,
  anomCount,
  total
}) {
  const B = window.BENCH;
  const rateF = total ? anomCount / total : 0;
  const rate = (rateF * 100).toFixed(1);
  const withheld = (rows || []).filter(r => r.withheld).length;
  // live mean detection score across the window — an actual computed signal
  const liveScore = total ? (rows.reduce((a, r) => a + r.score, 0) / total).toFixed(3) : "—";
  return /*#__PURE__*/React.createElement("div", {
    className: "cn-stats"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-stat hero"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-stat-lab"
  }, "ANOMALIES \xB7 live"), /*#__PURE__*/React.createElement("div", {
    className: "cn-stat-val anom"
  }, anomCount), /*#__PURE__*/React.createElement("div", {
    className: "cn-stat-rate"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-stat-rate-fill",
    style: {
      width: Math.min(100, rateF * 100) + "%"
    }
  })), /*#__PURE__*/React.createElement("div", {
    className: "cn-stat-sub"
  }, rate, "% of ", total, " in rolling window")), /*#__PURE__*/React.createElement("div", {
    className: "cn-stat ethic"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-stat-lab"
  }, "BENEVOLENCE \u03B7"), /*#__PURE__*/React.createElement("div", {
    className: "cn-stat-val benev"
  }, "\u2265 0.99"), /*#__PURE__*/React.createElement("div", {
    className: "cn-stat-sub"
  }, withheld, " withheld \xB7 hard gate")), /*#__PURE__*/React.createElement("div", {
    className: "cn-stat"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-stat-lab"
  }, "MEAN SCORE \xB7 live"), /*#__PURE__*/React.createElement("div", {
    className: "cn-stat-val sigma"
  }, liveScore), /*#__PURE__*/React.createElement("div", {
    className: "cn-stat-sub"
  }, "ensemble \xB7 0\u20131 window avg")), /*#__PURE__*/React.createElement("div", {
    className: "cn-stat ref"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-stat-lab"
  }, "BENCHMARK AUC", /*#__PURE__*/React.createElement("span", {
    className: "cn-stat-reftag"
  }, "ref")), /*#__PURE__*/React.createElement("div", {
    className: "cn-stat-val"
  }, B.auc.toFixed(4)), /*#__PURE__*/React.createElement("div", {
    className: "cn-stat-sub"
  }, "reference \xB7 ", B.sets, " datasets")));
}
function Sev({
  s
}) {
  const crit = ["HIGH", "CRITICAL", "SEVERE"].includes(s);
  const med = ["MEDIUM", "WARNING", "MODERATE"].includes(s);
  const cls = crit ? "crit" : med ? "med" : "low";
  return /*#__PURE__*/React.createElement("span", {
    className: "cn-sev " + cls
  }, s);
}
function DetectionFeed({
  rows,
  selected,
  onSelect
}) {
  return /*#__PURE__*/React.createElement("section", {
    className: "cn-panel cn-feed"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-panel-head"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-lab"
  }, "LIVE DETECTION FEED"), /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-meta"
  }, rows.length, " events \xB7 3R ensemble")), /*#__PURE__*/React.createElement(DetectionFeedBody, {
    rows: rows,
    selected: selected,
    onSelect: onSelect
  }));
}

/* Shared feed body. When tau is provided, a flag column marks score ≥ τ. Live API
   rows are badged and excluded from the τ flag (they carry a native scale). Newly
   inserted rows settle in once (mechanical, near-linear — see .cn-row.enter). */
function DetectionFeedBody({
  rows,
  selected,
  onSelect,
  tau
}) {
  const showFlag = typeof tau === "number";
  const seen = React.useRef(null);
  let newIds;
  if (seen.current === null) {
    seen.current = new Set(rows.map(r => r.id));
    newIds = new Set();
  } else {
    newIds = new Set(rows.filter(r => !seen.current.has(r.id)).map(r => r.id));
  }
  React.useEffect(() => {
    rows.forEach(r => seen.current.add(r.id));
  });
  const fmt = (x, d) => x == null ? "—" : x.toFixed(d == null ? 3 : d);
  return /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("div", {
    className: "cn-table-head" + (showFlag ? " with-flag" : "")
  }, /*#__PURE__*/React.createElement("span", null, "TIME"), /*#__PURE__*/React.createElement("span", null, "DOMAIN"), /*#__PURE__*/React.createElement("span", null, "STATUS"), /*#__PURE__*/React.createElement("span", {
    className: "r"
  }, "SCORE"), /*#__PURE__*/React.createElement("span", {
    className: "r"
  }, "CONF"), /*#__PURE__*/React.createElement("span", {
    className: "r"
  }, "BENEV"), /*#__PURE__*/React.createElement("span", null, "SEVERITY"), showFlag && /*#__PURE__*/React.createElement("span", {
    className: "r"
  }, "\u03C4")), /*#__PURE__*/React.createElement("div", {
    className: "cn-table-body"
  }, rows.map(r => {
    const flagged = showFlag && !r.live && r.score >= tau;
    return /*#__PURE__*/React.createElement("div", {
      key: r.id,
      className: "cn-row" + (showFlag ? " with-flag" : "") + (r.isAnom ? " anom" : "") + (r.live ? " live" : "") + (newIds.has(r.id) ? " enter" : "") + (selected === r.id ? " sel" : ""),
      onClick: () => onSelect(r.id)
    }, /*#__PURE__*/React.createElement("span", {
      className: "cn-ts"
    }, r.ts), /*#__PURE__*/React.createElement("span", {
      className: "cn-dom"
    }, r.live && /*#__PURE__*/React.createElement("span", {
      className: "cn-livetag"
    }, "LIVE"), r.domain), /*#__PURE__*/React.createElement("span", {
      className: r.isAnom ? "cn-st-anom" : "cn-st-norm"
    }, /*#__PURE__*/React.createElement("span", {
      className: "cn-d"
    }), r.isAnom ? "ANOMALY" : "NORMAL"), /*#__PURE__*/React.createElement("span", {
      className: "r cn-num"
    }, r.score.toFixed(r.live ? 2 : 3)), /*#__PURE__*/React.createElement("span", {
      className: "r cn-num"
    }, fmt(r.conf)), /*#__PURE__*/React.createElement("span", {
      className: "r cn-num" + (r.withheld ? " anom" : "")
    }, fmt(r.benev)), /*#__PURE__*/React.createElement("span", null, /*#__PURE__*/React.createElement(Sev, {
      s: r.sev
    })), showFlag && /*#__PURE__*/React.createElement("span", {
      className: "r"
    }, r.live ? /*#__PURE__*/React.createElement("span", {
      className: "cn-flag api"
    }, "API") : flagged ? /*#__PURE__*/React.createElement("span", {
      className: "cn-flag yes"
    }, "FLAG") : /*#__PURE__*/React.createElement("span", {
      className: "cn-flag no"
    }, "\xB7")));
  })));
}
function EnsemblePanel() {
  return /*#__PURE__*/React.createElement("section", {
    className: "cn-panel"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-panel-head"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-lab"
  }, "DETECTION ENSEMBLE"), /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-meta"
  }, "adaptive weights")), /*#__PURE__*/React.createElement("div", {
    className: "cn-ens"
  }, window.ENSEMBLE.map(e => /*#__PURE__*/React.createElement("div", {
    className: "cn-ens-row",
    key: e.name
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-ens-top"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-ens-name"
  }, e.short), /*#__PURE__*/React.createElement("span", {
    className: "cn-ens-w"
  }, e.weight, "%"), /*#__PURE__*/React.createElement("span", {
    className: "cn-ens-auc"
  }, e.auc.toFixed(3))), /*#__PURE__*/React.createElement("div", {
    className: "cn-track"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-fill",
    style: {
      width: e.auc * 100 + "%"
    }
  })), /*#__PURE__*/React.createElement("div", {
    className: "cn-ens-method"
  }, e.method))), /*#__PURE__*/React.createElement("div", {
    className: "cn-ens-row total"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-ens-top"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-ens-name"
  }, "Ensemble"), /*#__PURE__*/React.createElement("span", {
    className: "cn-ens-w"
  }, "\u03A9-AvA"), /*#__PURE__*/React.createElement("span", {
    className: "cn-ens-auc lead"
  }, window.BENCH.auc.toFixed(4))), /*#__PURE__*/React.createElement("div", {
    className: "cn-track lead"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-fill grad",
    style: {
      width: window.BENCH.auc * 100 + "%"
    }
  })))));
}
function EthicsPanel() {
  return /*#__PURE__*/React.createElement("section", {
    className: "cn-panel"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-panel-head"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-lab"
  }, "ETHICAL GATES"), /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-meta"
  }, "no advisory mode")), /*#__PURE__*/React.createElement("div", {
    className: "cn-ethics"
  }, window.ETHICS.map(g => /*#__PURE__*/React.createElement("div", {
    className: "cn-gate",
    key: g.gate
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-gate-glyph"
  }, g.glyph), /*#__PURE__*/React.createElement("div", {
    className: "cn-gate-mid"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-gate-name"
  }, g.gate), /*#__PURE__*/React.createElement("div", {
    className: "cn-gate-note"
  }, g.note)), /*#__PURE__*/React.createElement("div", {
    className: "cn-gate-val"
  }, g.op, " ", g.value), /*#__PURE__*/React.createElement("span", {
    className: "cn-gate-state"
  }, g.state)))));
}
function DomainCoverage({
  activeDomain,
  onPick
}) {
  return /*#__PURE__*/React.createElement("section", {
    className: "cn-panel"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-panel-head"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-lab"
  }, "DOMAIN COVERAGE"), /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-meta"
  }, "mean AUC")), /*#__PURE__*/React.createElement("div", {
    className: "cn-doms"
  }, window.DOMAINS.map(d => /*#__PURE__*/React.createElement("button", {
    key: d.id,
    className: "cn-domrow" + (activeDomain === d.id ? " on" : ""),
    onClick: () => onPick(d.id)
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-domname"
  }, d.label), /*#__PURE__*/React.createElement("span", {
    className: "cn-domsector"
  }, d.sector), /*#__PURE__*/React.createElement("span", {
    className: "cn-track sm"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-fill",
    style: {
      width: d.auc * 100 + "%"
    }
  })), /*#__PURE__*/React.createElement("span", {
    className: "cn-domauc"
  }, d.auc.toFixed(3))))));
}

/* Live-API detail: a selected LIVE row shows full provenance (endpoint, method,
   served-by, correlation ID), its native score + scale, the real feature/detector
   contributions, and a recover-raw-payload control — the dev-grade trust surface
   that closes the compose → watch → inspect loop. */
function LiveDetail({
  det,
  onExplain
}) {
  const [showRaw, setShowRaw] = React.useState(false);
  const maxW = det.feats.length ? Math.max.apply(null, det.feats.map(f => Math.abs(f.w))) || 1 : 1;
  return /*#__PURE__*/React.createElement("section", {
    className: "cn-panel cn-detail" + (det.isAnom ? " is-anom" : "")
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-panel-head"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-lab"
  }, "EVENT DETAIL"), /*#__PURE__*/React.createElement("span", {
    className: "cn-srcbadge " + det.source
  }, det.source === "live" ? "● LIVE" : "○ MOCK")), /*#__PURE__*/React.createElement("div", {
    className: "cn-detail-body"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-detail-top"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: "cn-dk"
  }, "SCORE"), /*#__PURE__*/React.createElement("div", {
    className: "cn-dv big"
  }, det.score.toFixed(2))), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: "cn-dk"
  }, "SCALE"), /*#__PURE__*/React.createElement("div", {
    className: "cn-dv cn-dv-scale"
  }, det.scale)), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: "cn-dk"
  }, "CONF"), /*#__PURE__*/React.createElement("div", {
    className: "cn-dv"
  }, det.conf == null ? "—" : det.conf.toFixed(3))), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: "cn-dk"
  }, "SEVERITY"), /*#__PURE__*/React.createElement("div", {
    className: "cn-dv"
  }, /*#__PURE__*/React.createElement(Sev, {
    s: det.sev
  })))), /*#__PURE__*/React.createElement("div", {
    className: "cn-detail-sec"
  }, "PROVENANCE \xB7 API contract"), /*#__PURE__*/React.createElement("div", {
    className: "cn-prov"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-prov-row"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-prov-k"
  }, "endpoint"), /*#__PURE__*/React.createElement("code", null, det.stream)), /*#__PURE__*/React.createElement("div", {
    className: "cn-prov-row"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-prov-k"
  }, "method"), /*#__PURE__*/React.createElement("span", {
    className: "cn-prov-v"
  }, det.method)), /*#__PURE__*/React.createElement("div", {
    className: "cn-prov-row"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-prov-k"
  }, "threshold"), /*#__PURE__*/React.createElement("span", {
    className: "cn-prov-v"
  }, det.threshold == null ? "—" : det.threshold, " ", /*#__PURE__*/React.createElement("span", {
    className: "cn-prov-scale"
  }, "(", det.scale, ")"))), /*#__PURE__*/React.createElement("div", {
    className: "cn-prov-row"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-prov-k"
  }, "served"), /*#__PURE__*/React.createElement("span", {
    className: "cn-prov-src " + det.source
  }, det.source.toUpperCase(), det.latencyMs != null ? " · " + det.latencyMs + "ms" : "", " \xB7 HTTP ", det.status)), /*#__PURE__*/React.createElement("div", {
    className: "cn-prov-row"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-prov-k"
  }, "corr-id"), /*#__PURE__*/React.createElement("code", {
    className: "cn-prov-corr"
  }, det.corr || "—"))), det.feats.length > 0 && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("div", {
    className: "cn-detail-sec"
  }, det.endpoint === "fusion" ? "DETECTOR CONTRIBUTION" : det.endpoint === "three-r" ? "3R COMPONENTS" : "FEATURE CONTRIBUTION"), /*#__PURE__*/React.createElement("div", {
    className: "cn-shap"
  }, det.feats.map(f => /*#__PURE__*/React.createElement("div", {
    className: "cn-shap-row",
    key: f.f
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-shap-f"
  }, f.f), /*#__PURE__*/React.createElement("div", {
    className: "cn-shap-track"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-shap-bar " + (f.w >= 0 ? "pos" : "neg"),
    style: {
      width: Math.abs(f.w) / maxW * 100 + "%"
    }
  })), /*#__PURE__*/React.createElement("span", {
    className: "cn-shap-w " + (f.w >= 0 ? "pos" : "neg")
  }, typeof f.w === "number" ? (f.w >= 0 ? "+" : "") + f.w.toFixed(2) : f.w))))), /*#__PURE__*/React.createElement("div", {
    className: "cn-detail-action"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-act-k"
  }, "RESULT"), /*#__PURE__*/React.createElement("span", {
    className: "cn-act-t"
  }, det.action)), /*#__PURE__*/React.createElement("button", {
    className: "cn-raw-toggle",
    onClick: () => setShowRaw(v => !v)
  }, showRaw ? "▾ hide raw payload" : "▸ recover raw payload"), showRaw && /*#__PURE__*/React.createElement("pre", {
    className: "cn-json sm"
  }, JSON.stringify(det.raw, null, 2)), /*#__PURE__*/React.createElement("button", {
    className: "cn-detail-explain",
    onClick: onExplain
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-sig-sm"
  }, "\u2671"), " Ask Mercury Agent to explain")));
}

/* Detail panel for a selected detection — define (what), locate (where), and
   describe (why), built from the SHAP attribution + counterfactual + action. */
function AnomalyDetail({
  det,
  onExplain
}) {
  if (!det) {
    return /*#__PURE__*/React.createElement("section", {
      className: "cn-panel cn-detail empty"
    }, /*#__PURE__*/React.createElement("div", {
      className: "cn-panel-head"
    }, /*#__PURE__*/React.createElement("span", {
      className: "cn-panel-lab"
    }, "EVENT DETAIL")), /*#__PURE__*/React.createElement("div", {
      className: "cn-detail-empty"
    }, "Select a detection to inspect its attribution, location, and recommended action. Mercury Agent can explain any event."));
  }
  if (det.live) return /*#__PURE__*/React.createElement(LiveDetail, {
    det: det,
    onExplain: onExplain
  });
  const maxW = Math.max(...det.feats.map(f => Math.abs(f.w))) || 1;
  return /*#__PURE__*/React.createElement("section", {
    className: "cn-panel cn-detail" + (det.isAnom ? " is-anom" : "")
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-panel-head"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-lab"
  }, "EVENT DETAIL"), /*#__PURE__*/React.createElement("span", {
    className: det.isAnom ? "cn-detail-flag anom" : "cn-detail-flag norm"
  }, det.isAnom ? "ANOMALY" : "NORMAL")), /*#__PURE__*/React.createElement("div", {
    className: "cn-detail-body"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-detail-top"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: "cn-dk"
  }, "SCORE"), /*#__PURE__*/React.createElement("div", {
    className: "cn-dv big"
  }, det.score.toFixed(3))), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: "cn-dk"
  }, "CONF"), /*#__PURE__*/React.createElement("div", {
    className: "cn-dv"
  }, det.conf.toFixed(3))), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: "cn-dk"
  }, "BENEV"), /*#__PURE__*/React.createElement("div", {
    className: "cn-dv" + (det.withheld ? " withheld" : "")
  }, det.benev.toFixed(3))), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: "cn-dk"
  }, "SEVERITY"), /*#__PURE__*/React.createElement("div", {
    className: "cn-dv"
  }, /*#__PURE__*/React.createElement(Sev, {
    s: det.sev
  })))), det.withheld && /*#__PURE__*/React.createElement("div", {
    className: "cn-withheld-banner"
  }, "\u2715 ACTION WITHHELD \xB7 \u03B7 ", det.benev.toFixed(3), " < 0.99 \xB7 EthicalConstraintViolationError"), /*#__PURE__*/React.createElement("div", {
    className: "cn-detail-loc"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-loc-k"
  }, "LOCATE"), /*#__PURE__*/React.createElement("span", null, det.domain), /*#__PURE__*/React.createElement("span", {
    className: "cn-loc-sep"
  }, "\xB7"), /*#__PURE__*/React.createElement("span", null, det.stream), /*#__PURE__*/React.createElement("span", {
    className: "cn-loc-sep"
  }, "\xB7"), /*#__PURE__*/React.createElement("span", null, "sample #", det.seq), /*#__PURE__*/React.createElement("span", {
    className: "cn-loc-sep"
  }, "\xB7"), /*#__PURE__*/React.createElement("span", null, det.dims, "-dim vector"), /*#__PURE__*/React.createElement("span", {
    className: "cn-loc-sep"
  }, "\xB7"), /*#__PURE__*/React.createElement("span", null, det.ts)), /*#__PURE__*/React.createElement("div", {
    className: "cn-detail-sec"
  }, "SHAP FEATURE ATTRIBUTION"), /*#__PURE__*/React.createElement("div", {
    className: "cn-shap"
  }, det.feats.map(f => /*#__PURE__*/React.createElement("div", {
    className: "cn-shap-row",
    key: f.f
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-shap-f"
  }, f.f), /*#__PURE__*/React.createElement("div", {
    className: "cn-shap-track"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-shap-bar " + (f.w >= 0 ? "pos" : "neg"),
    style: {
      width: Math.abs(f.w) / maxW * 100 + "%"
    }
  })), /*#__PURE__*/React.createElement("span", {
    className: "cn-shap-w " + (f.w >= 0 ? "pos" : "neg")
  }, f.w >= 0 ? "+" : "", f.w)))), /*#__PURE__*/React.createElement("div", {
    className: "cn-cf"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-cf-k"
  }, "COUNTERFACTUAL"), /*#__PURE__*/React.createElement("span", {
    className: "cn-cf-t"
  }, det.counterfactual.f, " ", /*#__PURE__*/React.createElement("span", {
    className: "cn-cf-obs"
  }, det.counterfactual.obs), /*#__PURE__*/React.createElement("span", {
    className: "cn-cf-arrow"
  }, " \u2192 \u2264 "), /*#__PURE__*/React.createElement("span", {
    className: "cn-cf-bound"
  }, det.counterfactual.bound), /*#__PURE__*/React.createElement("span", {
    className: "cn-cf-out"
  }, " \u21D2 NORMAL"))), /*#__PURE__*/React.createElement("div", {
    className: "cn-detail-action"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-act-k"
  }, "RECOMMENDED"), /*#__PURE__*/React.createElement("span", {
    className: "cn-act-t"
  }, det.action)), /*#__PURE__*/React.createElement("button", {
    className: "cn-detail-explain",
    onClick: onExplain
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-sig-sm"
  }, "\u2671"), " Ask Mercury Agent to explain")));
}
Object.assign(window, {
  StatCards,
  DetectionFeed,
  DetectionFeedBody,
  EnsemblePanel,
  EthicsPanel,
  DomainCoverage,
  Sev,
  AnomalyDetail,
  LiveDetail
});
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/console/Panels.jsx", error: String((e && e.message) || e) }); }

// ui_kits/console/Views.jsx
try { (() => {
/* Mercury Console — page views. Each page computes from the live feed where it
   can, rather than rendering stored constants. Information-theoretic + operator-
   grade instrumentation: threshold/confusion on Detections, AUC baselines on
   Domains, adaptive-weight derivation on Ensemble, live gate-margin telemetry on
   Ethics, NIST function breakdown + severity→TLP mapping on Compliance, and
   cost/training axes on Models. */
const {
  useState: useStateV,
  useEffect: useEffectV,
  useRef: useRefV
} = React;

/* ---------- shared ---------- */
function pctNum(x) {
  return (x * 100).toFixed(1);
}
function tierOf(auc) {
  return auc >= 0.85 ? "strong" : auc >= 0.75 ? "moderate" : "weak";
}

/* Continuous benevolence sampler — the SAME documented distribution the detection
   generator uses (ConsoleData.genDetection): ~12% of actions are withheld at
   η∈[0.982,0.9899]; the rest clear at η∈[0.990,0.999]. Lets the Ethical Gates page
   run a live gate monitor independent of a manual RUN DETECTION burst. */
function r3v(x) {
  return Math.round(x * 1000) / 1000;
}
function sampleBenevV() {
  const rnd = (a, b) => a + Math.random() * (b - a);
  const benev = Math.random() < 0.12 ? r3v(rnd(0.982, 0.9899)) : r3v(rnd(0.990, 0.999));
  return {
    benev,
    withheld: benev < 0.99,
    gateMargin: r3v(benev - 0.99)
  };
}

/* confusion matrix of the live feed at threshold τ (score ≥ τ ⇒ flagged) */
function confusionAt(rows, tau) {
  let tp = 0,
    fp = 0,
    fn = 0,
    tn = 0;
  rows.forEach(r => {
    const flagged = r.score >= tau;
    if (flagged && r.isAnom) tp++;else if (flagged && !r.isAnom) fp++;else if (!flagged && r.isAnom) fn++;else tn++;
  });
  const precision = tp + fp ? tp / (tp + fp) : 0;
  const recall = tp + fn ? tp / (tp + fn) : 0;
  const f1 = precision + recall ? 2 * precision * recall / (precision + recall) : 0;
  return {
    tp,
    fp,
    fn,
    tn,
    precision,
    recall,
    f1,
    flagged: tp + fp
  };
}

/* score histogram, anomaly vs normal, NBINS buckets */
function histogram(rows, nbins) {
  const bins = Array.from({
    length: nbins
  }, () => ({
    a: 0,
    n: 0
  }));
  rows.forEach(r => {
    const i = Math.min(nbins - 1, Math.floor(r.score * nbins));
    if (r.isAnom) bins[i].a++;else bins[i].n++;
  });
  const max = Math.max(1, ...bins.map(b => b.a + b.n));
  return {
    bins,
    max
  };
}

/* ---------- OVERVIEW ---------- */
function OverviewView({
  rows,
  selectedId,
  onSelect,
  detail,
  onExplain
}) {
  const anomCount = rows.filter(r => r.isAnom).length;
  return /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement(StatCards, {
    rows: rows,
    anomCount: anomCount,
    total: rows.length
  }), /*#__PURE__*/React.createElement("div", {
    className: "cn-grid"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-col-main"
  }, /*#__PURE__*/React.createElement(DetectionFeed, {
    rows: rows,
    selected: selectedId,
    onSelect: onSelect
  })), /*#__PURE__*/React.createElement("div", {
    className: "cn-col-side"
  }, /*#__PURE__*/React.createElement(AnomalyDetail, {
    det: detail,
    onExplain: onExplain
  }), /*#__PURE__*/React.createElement(EthicsPanel, null))));
}

/* ---------- DETECTIONS — compose · ensemble τ instrument · live feed · inspect ---------- */
function DetectionsView({
  rows,
  selectedId,
  onSelect,
  detail,
  onExplain,
  tau,
  setTau,
  onComposeResult,
  lastCorr
}) {
  const [filter, setFilter] = useStateV("all"); // all | flagged | high
  // The τ histogram + confusion are defined over the SYNTHETIC ensemble stream
  // ([0,1] scores). Live API rows carry native scales (z-score / L2 / fused), so
  // they are excluded from these instruments — they appear in the feed, badged.
  const ensembleRows = rows.filter(r => !r.live);
  const liveCount = rows.length - ensembleRows.length;
  const cm = confusionAt(ensembleRows, tau);
  const {
    bins,
    max
  } = histogram(ensembleRows, 20);
  const shown = rows.filter(r => {
    if (filter === "flagged") return r.live ? r.isAnom : r.score >= tau;
    if (filter === "high") return r.sevKey === "high";
    return true;
  });
  return /*#__PURE__*/React.createElement("div", {
    className: "cn-grid wide"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-col-main"
  }, /*#__PURE__*/React.createElement(RunComposer, {
    onResult: onComposeResult
  }), lastCorr && /*#__PURE__*/React.createElement("div", {
    className: "cn-trace"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-trace-k"
  }, "TRACE"), "last request ", /*#__PURE__*/React.createElement("code", null, lastCorr), " \u2192 newest feed rows \xB7 select one to inspect its payload"), /*#__PURE__*/React.createElement("section", {
    className: "cn-panel",
    style: {
      marginTop: "16px"
    }
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-panel-head"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-lab"
  }, "DECISION THRESHOLD \xB7 \u03C4"), /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-meta"
  }, "ensemble score [0,1] \xB7 \u2265 \u03C4 \u21D2 flagged \xB7 ", cm.flagged, "/", ensembleRows.length, liveCount ? " · " + liveCount + " live excluded" : "")), /*#__PURE__*/React.createElement("div", {
    className: "cn-thr"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-hist"
  }, bins.map((b, i) => {
    const left = i / bins.length * 100;
    return /*#__PURE__*/React.createElement("div", {
      className: "cn-hist-col",
      key: i,
      style: {
        left: left + "%",
        width: 100 / bins.length + "%"
      }
    }, /*#__PURE__*/React.createElement("div", {
      className: "cn-hist-a",
      style: {
        height: b.a / max * 100 + "%"
      }
    }), /*#__PURE__*/React.createElement("div", {
      className: "cn-hist-n",
      style: {
        height: b.n / max * 100 + "%"
      }
    }));
  }), /*#__PURE__*/React.createElement("div", {
    className: "cn-hist-tau",
    style: {
      left: tau * 100 + "%"
    }
  }, /*#__PURE__*/React.createElement("span", null, "\u03C4 ", tau.toFixed(2)))), /*#__PURE__*/React.createElement("input", {
    className: "cn-thr-range",
    type: "range",
    min: "0",
    max: "1",
    step: "0.01",
    value: tau,
    onChange: e => setTau(+e.target.value)
  }), /*#__PURE__*/React.createElement("div", {
    className: "cn-hist-legend"
  }, /*#__PURE__*/React.createElement("span", null, /*#__PURE__*/React.createElement("i", {
    className: "sw a"
  }), "anomaly"), /*#__PURE__*/React.createElement("span", null, /*#__PURE__*/React.createElement("i", {
    className: "sw n"
  }), "normal"), /*#__PURE__*/React.createElement("span", {
    className: "cn-hist-hint"
  }, "drag \u03C4 \u2014 watch precision/recall trade off")), /*#__PURE__*/React.createElement("div", {
    className: "cn-hist-axis"
  }, /*#__PURE__*/React.createElement("span", null, "0.0"), /*#__PURE__*/React.createElement("span", null, "ensemble score \xB7 [0,1]"), /*#__PURE__*/React.createElement("span", null, "1.0")), /*#__PURE__*/React.createElement("div", {
    className: "cn-cm"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-cm-grid"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-cm-cell tp"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-cm-n"
  }, cm.tp), /*#__PURE__*/React.createElement("span", {
    className: "cn-cm-l"
  }, "TP \xB7 true anomaly")), /*#__PURE__*/React.createElement("div", {
    className: "cn-cm-cell fp"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-cm-n"
  }, cm.fp), /*#__PURE__*/React.createElement("span", {
    className: "cn-cm-l"
  }, "FP \xB7 false alarm")), /*#__PURE__*/React.createElement("div", {
    className: "cn-cm-cell fn"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-cm-n"
  }, cm.fn), /*#__PURE__*/React.createElement("span", {
    className: "cn-cm-l"
  }, "FN \xB7 missed")), /*#__PURE__*/React.createElement("div", {
    className: "cn-cm-cell tn"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-cm-n"
  }, cm.tn), /*#__PURE__*/React.createElement("span", {
    className: "cn-cm-l"
  }, "TN \xB7 true normal"))), /*#__PURE__*/React.createElement("div", {
    className: "cn-cm-metrics"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-cm-m"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-cm-mk"
  }, "Precision"), /*#__PURE__*/React.createElement("span", {
    className: "cn-cm-mv"
  }, cm.precision.toFixed(3))), /*#__PURE__*/React.createElement("div", {
    className: "cn-cm-m"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-cm-mk"
  }, "Recall"), /*#__PURE__*/React.createElement("span", {
    className: "cn-cm-mv"
  }, cm.recall.toFixed(3))), /*#__PURE__*/React.createElement("div", {
    className: "cn-cm-m"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-cm-mk"
  }, "F1"), /*#__PURE__*/React.createElement("span", {
    className: "cn-cm-mv hl"
  }, cm.f1.toFixed(3))))))), /*#__PURE__*/React.createElement("section", {
    className: "cn-panel",
    style: {
      marginTop: "16px"
    }
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-panel-head"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-lab"
  }, "LIVE DETECTION FEED"), /*#__PURE__*/React.createElement("span", {
    className: "cn-feed-filters"
  }, ["all", "flagged", "high"].map(f => /*#__PURE__*/React.createElement("button", {
    key: f,
    className: "cn-feed-filter" + (filter === f ? " on" : ""),
    onClick: () => setFilter(f)
  }, f === "all" ? "all" : f === "flagged" ? "flagged ≥ τ" : "high sev")))), /*#__PURE__*/React.createElement(DetectionFeedBody, {
    rows: shown,
    tau: tau,
    selected: selectedId,
    onSelect: onSelect
  }))), /*#__PURE__*/React.createElement("div", {
    className: "cn-col-side"
  }, /*#__PURE__*/React.createElement(AnomalyDetail, {
    det: detail,
    onExplain: onExplain
  })));
}

/* ---------- DOMAINS — live per-domain telemetry + drill-down ---------- */
function Sparkline({
  recent
}) {
  if (!recent || !recent.length) return /*#__PURE__*/React.createElement("div", {
    className: "cn-spark empty"
  });
  const w = 100,
    h = 22,
    n = recent.length,
    bw = w / n;
  return /*#__PURE__*/React.createElement("svg", {
    className: "cn-spark",
    viewBox: `0 0 ${w} ${h}`,
    preserveAspectRatio: "none"
  }, /*#__PURE__*/React.createElement("line", {
    x1: "0",
    y1: h * 0.5,
    x2: w,
    y2: h * 0.5,
    className: "cn-spark-thr"
  }), recent.map((p, i) => {
    const bh = Math.max(1, p.s * h);
    return /*#__PURE__*/React.createElement("rect", {
      key: i,
      x: i * bw + 0.5,
      y: h - bh,
      width: Math.max(1, bw - 1),
      height: bh,
      className: p.a ? "cn-spark-a" : "cn-spark-n"
    });
  }));
}
function DomainsView({
  activeDomain,
  onPickDomain,
  stats,
  onFocusFeed,
  onAskAgent
}) {
  const mean = window.BENCH.auc,
    adbench = window.BENCH.adbench;
  const sorted = window.DOMAINS.slice().sort((a, b) => b.auc - a.auc);
  const sel = activeDomain ? window.DOMAINS.find(d => d.id === activeDomain) : null;
  return /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("div", {
    className: "cn-dom-baseline"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-dom-bl-k"
  }, "Benchmark mean ROC-AUC"), /*#__PURE__*/React.createElement("span", {
    className: "cn-dom-bl-v"
  }, mean.toFixed(4)), /*#__PURE__*/React.createElement("span", {
    className: "cn-dom-bl-sep"
  }, "\xB7"), /*#__PURE__*/React.createElement("span", {
    className: "cn-dom-bl-k"
  }, "live AUC = rolling Mann\u2013Whitney over each domain's adjudicated window \xB7 drifts with the feed \xB7 ", /*#__PURE__*/React.createElement("b", null, "RUN DETECTION"), " to stream")), /*#__PURE__*/React.createElement("div", {
    className: "cn-grid" + (sel ? " wide" : "")
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-col-main"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-dgrid"
  }, sorted.map(d => {
    const tier = tierOf(d.auc);
    const delta = d.auc - mean;
    const barPct = Math.max(0, (d.auc - 0.5) / 0.5 * 100);
    const meanPct = (mean - 0.5) / 0.5 * 100;
    const st = stats[d.id] || {
      n: 0,
      anom: 0,
      anomRate: 0,
      liveAUC: null,
      recent: []
    };
    const drift = st.liveAUC == null ? null : st.liveAUC - d.auc;
    const livePct = st.liveAUC == null ? 0 : Math.max(0, (st.liveAUC - 0.5) / 0.5 * 100);
    return /*#__PURE__*/React.createElement("button", {
      key: d.id,
      className: "cn-dcard tier-" + tier + (activeDomain === d.id ? " on" : ""),
      onClick: () => onPickDomain(d.id)
    }, /*#__PURE__*/React.createElement("div", {
      className: "cn-dcard-top"
    }, /*#__PURE__*/React.createElement("span", {
      className: "cn-dcard-name"
    }, d.label), /*#__PURE__*/React.createElement("span", {
      className: "cn-dcard-auc"
    }, d.auc.toFixed(3))), /*#__PURE__*/React.createElement("div", {
      className: "cn-dcard-sector"
    }, d.sector, " \xB7 ", d.feats.length, "-dim \xB7 ", st.n, " live \xB7 ", st.anom, " anom"), /*#__PURE__*/React.createElement("div", {
      className: "cn-dcard-track"
    }, /*#__PURE__*/React.createElement("span", {
      className: "cn-dcard-fill",
      style: {
        width: barPct + "%"
      }
    }), /*#__PURE__*/React.createElement("span", {
      className: "cn-dcard-live",
      style: {
        width: livePct + "%"
      }
    }), /*#__PURE__*/React.createElement("span", {
      className: "cn-dcard-mean",
      style: {
        left: meanPct + "%"
      },
      title: "fleet mean 0.8466"
    })), /*#__PURE__*/React.createElement("div", {
      className: "cn-dcard-liverow"
    }, /*#__PURE__*/React.createElement("span", {
      className: "cn-dcard-livelab"
    }, "live AUC"), /*#__PURE__*/React.createElement("span", {
      className: "cn-dcard-liveval"
    }, st.liveAUC == null ? "—" : st.liveAUC.toFixed(3)), drift != null && /*#__PURE__*/React.createElement("span", {
      className: "cn-dcard-drift " + (drift >= 0 ? "pos" : "neg")
    }, drift >= 0 ? "▲" : "▼", " ", Math.abs(drift).toFixed(3)), /*#__PURE__*/React.createElement("span", {
      className: "cn-dcard-rate"
    }, (st.anomRate * 100).toFixed(0), "% flagged")), /*#__PURE__*/React.createElement(Sparkline, {
      recent: st.recent
    }), /*#__PURE__*/React.createElement("div", {
      className: "cn-dcard-foot"
    }, /*#__PURE__*/React.createElement("span", {
      className: "cn-dcard-tier " + tier
    }, tier), /*#__PURE__*/React.createElement("span", {
      className: "cn-dcard-delta " + (delta >= 0 ? "pos" : "neg")
    }, delta >= 0 ? "+" : "", delta.toFixed(3), " vs mean")));
  }))), sel && /*#__PURE__*/React.createElement("div", {
    className: "cn-col-side"
  }, /*#__PURE__*/React.createElement(DomainDetail, {
    d: sel,
    st: stats[sel.id],
    onFocusFeed: onFocusFeed,
    onAskAgent: onAskAgent,
    onClose: () => onPickDomain(sel.id)
  }))));
}
function DomainDetail({
  d,
  st,
  onFocusFeed,
  onAskAgent,
  onClose
}) {
  const meta = window.DOMAIN_META[d.id];
  const tier = tierOf(d.auc);
  st = st || {
    n: 0,
    anom: 0,
    anomRate: 0,
    meanScore: 0,
    liveAUC: null,
    recent: []
  };
  // Hanley–McNeil style SE approximation for a 95% CI on the benchmark AUC
  const n = meta.n,
    na = Math.max(2, Math.round(n * 1.6)),
    nn = Math.max(2, Math.round(n * 4));
  const a = d.auc,
    q1 = a / (2 - a),
    q2 = 2 * a * a / (1 + a);
  const se = Math.sqrt((a * (1 - a) + (na - 1) * (q1 - a * a) + (nn - 1) * (q2 - a * a)) / (na * nn));
  const ci = 1.96 * se;
  const drift = st.liveAUC == null ? null : st.liveAUC - d.auc;
  const lad = window.LADDERS[d.ladder];
  const acts = window.ACTIONS[d.ladder];
  return /*#__PURE__*/React.createElement("section", {
    className: "cn-panel cn-domdetail"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-panel-head"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-lab"
  }, d.label.toUpperCase()), /*#__PURE__*/React.createElement("button", {
    className: "cn-dd-x",
    onClick: onClose,
    title: "deselect"
  }, "\u2715")), /*#__PURE__*/React.createElement("div", {
    className: "cn-dd-body"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-dd-auc"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-dd-auc-twin"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-dd-auc-col"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-dd-auc-k"
  }, "BENCHMARK"), /*#__PURE__*/React.createElement("span", {
    className: "cn-dd-auc-v"
  }, d.auc.toFixed(3)), /*#__PURE__*/React.createElement("span", {
    className: "cn-dd-auc-sub"
  }, "\xB1", ci.toFixed(3), " \xB7 n=", n)), /*#__PURE__*/React.createElement("div", {
    className: "cn-dd-auc-col"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-dd-auc-k"
  }, "LIVE \xB7 WINDOW"), /*#__PURE__*/React.createElement("span", {
    className: "cn-dd-auc-v live"
  }, st.liveAUC == null ? "—" : st.liveAUC.toFixed(3)), /*#__PURE__*/React.createElement("span", {
    className: "cn-dd-auc-sub"
  }, st.n, " adjudicated", drift != null ? " · " + (drift >= 0 ? "+" : "") + drift.toFixed(3) : ""))), /*#__PURE__*/React.createElement("div", {
    className: "cn-dd-tier " + tier
  }, tier, " discriminator \u2014 ", tier === "strong" ? "flag with confidence" : tier === "moderate" ? "corroborate high-stakes calls" : "advisory; pair with human review")), /*#__PURE__*/React.createElement("div", {
    className: "cn-dd-sec"
  }, "LIVE OPERATIONS \xB7 rolling window"), /*#__PURE__*/React.createElement("div", {
    className: "cn-dd-ops"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-dd-op"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-dd-op-v"
  }, st.n), /*#__PURE__*/React.createElement("span", {
    className: "cn-dd-op-k"
  }, "events")), /*#__PURE__*/React.createElement("div", {
    className: "cn-dd-op"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-dd-op-v anom"
  }, st.anom), /*#__PURE__*/React.createElement("span", {
    className: "cn-dd-op-k"
  }, "anomalies")), /*#__PURE__*/React.createElement("div", {
    className: "cn-dd-op"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-dd-op-v"
  }, (st.anomRate * 100).toFixed(0), "%"), /*#__PURE__*/React.createElement("span", {
    className: "cn-dd-op-k"
  }, "flagged \u2265 0.5")), /*#__PURE__*/React.createElement("div", {
    className: "cn-dd-op"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-dd-op-v"
  }, st.meanScore.toFixed(2)), /*#__PURE__*/React.createElement("span", {
    className: "cn-dd-op-k"
  }, "mean score"))), /*#__PURE__*/React.createElement("div", {
    className: "cn-dd-spark"
  }, /*#__PURE__*/React.createElement(Sparkline, {
    recent: st.recent
  })), /*#__PURE__*/React.createElement("div", {
    className: "cn-dd-sec"
  }, "DETECTOR BINDING"), /*#__PURE__*/React.createElement("div", {
    className: "cn-dd-row"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-dd-k"
  }, "Class"), /*#__PURE__*/React.createElement("span", {
    className: "cn-dd-v hl"
  }, meta.detector)), /*#__PURE__*/React.createElement("div", {
    className: "cn-dd-row"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-dd-k"
  }, "Source"), /*#__PURE__*/React.createElement("code", {
    className: "cn-dd-file"
  }, meta.file)), /*#__PURE__*/React.createElement("div", {
    className: "cn-dd-row"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-dd-k"
  }, "Stream"), /*#__PURE__*/React.createElement("span", {
    className: "cn-dd-v"
  }, d.stream)), /*#__PURE__*/React.createElement("div", {
    className: "cn-dd-sec"
  }, "FEATURE VECTOR \xB7 ", d.feats.length, "-dim"), /*#__PURE__*/React.createElement("div", {
    className: "cn-dd-feats"
  }, d.feats.map(f => /*#__PURE__*/React.createElement("span", {
    key: f,
    className: "cn-feat"
  }, f))), /*#__PURE__*/React.createElement("div", {
    className: "cn-dd-sec"
  }, "SEVERITY LADDER \u2192 ACTION"), /*#__PURE__*/React.createElement("div", {
    className: "cn-dd-ladder"
  }, [["high", lad.high], ["med", lad.med], ["low", lad.low]].map(([k, label]) => /*#__PURE__*/React.createElement("div", {
    className: "cn-dd-lad-row " + k,
    key: k
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-dd-lad-sev"
  }, label), /*#__PURE__*/React.createElement("span", {
    className: "cn-dd-lad-act"
  }, acts[k])))), /*#__PURE__*/React.createElement("div", {
    className: "cn-dd-actions"
  }, /*#__PURE__*/React.createElement("button", {
    className: "mc-action primary block",
    onClick: () => onFocusFeed(d.id)
  }, "\u25B8 Focus feed on this domain"), /*#__PURE__*/React.createElement("button", {
    className: "mc-action block",
    onClick: () => onAskAgent(d)
  }, "\u2671 Ask Mercury Agent"))));
}

/* ---------- ENSEMBLE — adaptive-weight derivation from AUC ---------- */
function EnsembleView() {
  const aw = window.adaptiveWeights();
  const maxAdaptive = Math.max(...aw.map(w => w.adaptive));
  return /*#__PURE__*/React.createElement("div", {
    className: "cn-ens-view"
  }, /*#__PURE__*/React.createElement("section", {
    className: "cn-panel cn-awtable"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-panel-head"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-lab"
  }, "ADAPTIVE WEIGHTING \xB7 DERIVED FROM DISCRIMINATIVE POWER"), /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-meta"
  }, "w\u1D62 \u221D (AUC\u1D62 \u2212 0.5), Mann\u2013Whitney")), /*#__PURE__*/React.createElement("div", {
    className: "cn-aw-head"
  }, /*#__PURE__*/React.createElement("span", null, "component"), /*#__PURE__*/React.createElement("span", {
    className: "r"
  }, "AUC"), /*#__PURE__*/React.createElement("span", {
    className: "r"
  }, "excess"), /*#__PURE__*/React.createElement("span", {
    className: "r"
  }, "nominal"), /*#__PURE__*/React.createElement("span", {
    className: "r"
  }, "adaptive"), /*#__PURE__*/React.createElement("span", null, "weight")), aw.map(w => /*#__PURE__*/React.createElement("div", {
    className: "cn-aw-row",
    key: w.name
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-aw-name"
  }, w.name), /*#__PURE__*/React.createElement("span", {
    className: "r cn-num"
  }, w.auc.toFixed(3)), /*#__PURE__*/React.createElement("span", {
    className: "r cn-aw-ex"
  }, "+", w.excess.toFixed(3)), /*#__PURE__*/React.createElement("span", {
    className: "r cn-aw-nom"
  }, w.nominal, "%"), /*#__PURE__*/React.createElement("span", {
    className: "r cn-aw-adapt " + (w.adaptive > w.nominal ? "up" : w.adaptive < w.nominal ? "down" : "")
  }, w.adaptive, "% ", w.adaptive > w.nominal ? "▲" : w.adaptive < w.nominal ? "▼" : "="), /*#__PURE__*/React.createElement("span", {
    className: "cn-aw-bar"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-aw-fill",
    style: {
      width: w.adaptive / maxAdaptive * 100 + "%"
    }
  })))), /*#__PURE__*/React.createElement("div", {
    className: "cn-aw-note"
  }, "InfoGeom is the strongest single discriminator (AUC 0.848) so it earns ", /*#__PURE__*/React.createElement("b", null, "more"), " than its nominal 30%; Kinematic (0.641) earns ", /*#__PURE__*/React.createElement("b", null, "less"), ". Components with AUC < 0.5 are zeroed; 0.05 floor, seed 42.")), /*#__PURE__*/React.createElement("div", {
    className: "cn-ens-cards"
  }, window.ENSEMBLE.map(e => /*#__PURE__*/React.createElement("div", {
    className: "cn-panel cn-enscard",
    key: e.name
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-panel-head"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-lab"
  }, e.short), /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-meta"
  }, e.cx)), /*#__PURE__*/React.createElement("div", {
    className: "cn-enscard-body"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-enscard-math"
  }, e.math), /*#__PURE__*/React.createElement("div", {
    className: "cn-enscard-method"
  }, e.method), /*#__PURE__*/React.createElement("div", {
    className: "cn-track"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-fill",
    style: {
      width: (e.auc - 0.5) / 0.5 * 100 + "%"
    }
  })), /*#__PURE__*/React.createElement("div", {
    className: "cn-enscard-cx"
  }, "AUC ", e.auc.toFixed(3), " \xB7 cost ", e.cx))))), /*#__PURE__*/React.createElement("section", {
    className: "cn-panel cn-fusion"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-panel-head"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-lab"
  }, "\u03A9-AvA FUSION"), /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-meta"
  }, "mean AUC ", window.BENCH.auc.toFixed(4))), /*#__PURE__*/React.createElement("div", {
    className: "cn-fusion-body"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-fusion-eq"
  }, "A = (w", /*#__PURE__*/React.createElement("sub", null, "R"), "\xB7R + w", /*#__PURE__*/React.createElement("sub", null, "H"), "\xB7H + w", /*#__PURE__*/React.createElement("sub", null, "O"), "\xB7O) \xB7 \u03B7", /*#__PURE__*/React.createElement("sub", null, "Ethical"), /*#__PURE__*/React.createElement("sup", null, "\u03A6")), /*#__PURE__*/React.createElement("ul", {
    className: "cn-fusion-list"
  }, /*#__PURE__*/React.createElement("li", null, "Per-component AUC via ", /*#__PURE__*/React.createElement("b", null, "Mann\u2013Whitney U"), " on 5-fold CV with synthetic anomalies (\u03C3 = 3\xB7std)."), /*#__PURE__*/React.createElement("li", null, "Golden-ratio fusion exponent ", /*#__PURE__*/React.createElement("b", null, "\u03C6 = 1.6180"), "; Lyapunov ", /*#__PURE__*/React.createElement("b", null, "\u03BB = 0.25"), " certifies contraction."), /*#__PURE__*/React.createElement("li", null, "Cost/accuracy tension: InfoGeom is most accurate but ", /*#__PURE__*/React.createElement("b", null, "O(d\xB3)"), " (Cholesky); Kinematic is cheap ", /*#__PURE__*/React.createElement("b", null, "O(n\xB7d)"), " but weak.")))));
}

/* ---------- ETHICS — live benevolence margin telemetry + withholding ---------- */
function EthicsView({
  rows
}) {
  const EPS = 0.003;
  const CAP = 56;
  const reduced = typeof document !== "undefined" && document.documentElement.classList.contains("mn-reduce-motion");
  const idRef = useRefV(0);
  // seed the monitor from real recent decisions so it starts populated + coherent
  const [buf, setBuf] = useStateV(() => {
    const base = (rows || []).filter(r => r.benev != null).slice(0, CAP).map(r => ({
      id: idRef.current++,
      benev: r.benev,
      withheld: r.withheld,
      gateMargin: r.gateMargin
    }));
    while (base.length < 30) base.push(Object.assign({
      id: idRef.current++
    }, sampleBenevV()));
    return base;
  });
  const [live, setLive] = useStateV(!reduced);

  // self-driving sampler: while LIVE, the gate is continuously monitored — a new
  // adjudication arrives on a cadence and flows through the margin strip.
  useEffectV(() => {
    if (!live) return undefined;
    const every = reduced ? 1600 : 950;
    const t = setInterval(() => {
      setBuf(prev => {
        const next = prev.concat([Object.assign({
          id: idRef.current++
        }, sampleBenevV())]);
        return next.length > CAP ? next.slice(next.length - CAP) : next;
      });
    }, every);
    return () => clearInterval(t);
  }, [live, reduced]);
  const cleared = buf.filter(r => !r.withheld);
  const withheld = buf.filter(r => r.withheld);
  const clearedMargins = cleared.map(r => r.gateMargin);
  const minMargin = clearedMargins.length ? Math.min(...clearedMargins) : 0;
  const meanMargin = clearedMargins.length ? clearedMargins.reduce((a, b) => a + b, 0) / clearedMargins.length : 0;
  const atRisk = cleared.filter(r => r.gateMargin <= EPS).length;
  const withholdRate = buf.length ? withheld.length / buf.length : 0;
  const newestId = buf.length ? buf[buf.length - 1].id : -1;
  return /*#__PURE__*/React.createElement("div", {
    className: "cn-grid wide"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-col-main"
  }, /*#__PURE__*/React.createElement("section", {
    className: "cn-panel"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-panel-head"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-lab"
  }, "BENEVOLENCE GATE \xB7 LIVE MARGIN \u03B7 \u2212 0.99"), /*#__PURE__*/React.createElement("button", {
    className: "cn-live-toggle" + (live ? " on" : ""),
    onClick: () => setLive(!live),
    title: live ? "Monitoring continuously — click to pause" : "Paused — click to resume the live gate monitor"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-live-dot"
  }), live ? "LIVE" : "PAUSED")), /*#__PURE__*/React.createElement("div", {
    className: "cn-margin"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-margin-stats"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-ms"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-ms-v"
  }, cleared.length), /*#__PURE__*/React.createElement("span", {
    className: "cn-ms-k"
  }, "cleared")), /*#__PURE__*/React.createElement("div", {
    className: "cn-ms"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-ms-v anom"
  }, withheld.length), /*#__PURE__*/React.createElement("span", {
    className: "cn-ms-k"
  }, "withheld (", pctNum(withholdRate), "%)")), /*#__PURE__*/React.createElement("div", {
    className: "cn-ms"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-ms-v"
  }, minMargin >= 0 ? "+" : "", minMargin.toFixed(3)), /*#__PURE__*/React.createElement("span", {
    className: "cn-ms-k"
  }, "min margin")), /*#__PURE__*/React.createElement("div", {
    className: "cn-ms"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-ms-v " + (atRisk ? "warn" : "")
  }, atRisk), /*#__PURE__*/React.createElement("span", {
    className: "cn-ms-k"
  }, "at-risk \u2264 ", EPS))), /*#__PURE__*/React.createElement("div", {
    className: "cn-margin-strip"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-margin-gate"
  }, /*#__PURE__*/React.createElement("span", null, "\u03B7 = 0.99 gate")), buf.map(r => {
    // map margin [-0.008, +0.009] → 0..100%
    const x = Math.max(0, Math.min(100, (r.gateMargin + 0.008) / 0.017 * 100));
    return /*#__PURE__*/React.createElement("span", {
      key: r.id,
      className: "cn-margin-dot " + (r.withheld ? "withheld" : r.gateMargin <= EPS ? "risk" : "ok") + (r.id === newestId ? " fresh" : ""),
      style: {
        left: x + "%"
      },
      title: "η " + r.benev.toFixed(3) + (r.withheld ? " · WITHHELD" : " · cleared")
    });
  })), /*#__PURE__*/React.createElement("div", {
    className: "cn-margin-axis"
  }, /*#__PURE__*/React.createElement("span", null, "withheld"), /*#__PURE__*/React.createElement("span", null, "\u2190 refusal | mean margin +", meanMargin.toFixed(3), " | clearance \u2192"), /*#__PURE__*/React.createElement("span", null, "safe"))), /*#__PURE__*/React.createElement("div", {
    className: "cn-gate-contract"
  }, "On \u03B7 < 0.99 the engine raises ", /*#__PURE__*/React.createElement("code", null, "EthicalConstraintViolationError(check=\"benevolence\")"), " and", /*#__PURE__*/React.createElement("b", null, " refuses the action"), " regardless of detection score. The margin is thin by design \u2014 benevolence is a tight bound, not a comfortable one.")), /*#__PURE__*/React.createElement("div", {
    className: "cn-gates-row"
  }, window.ETHICS.map(g => /*#__PURE__*/React.createElement("div", {
    className: "cn-panel cn-gatecard",
    key: g.gate
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-gatecard-glyph"
  }, g.glyph), /*#__PURE__*/React.createElement("div", {
    className: "cn-gatecard-name"
  }, g.gate, " ", /*#__PURE__*/React.createElement("span", null, g.op, " ", g.value)), /*#__PURE__*/React.createElement("div", {
    className: "cn-gatecard-note"
  }, g.detail), /*#__PURE__*/React.createElement("span", {
    className: "cn-gate-state"
  }, g.state))))), /*#__PURE__*/React.createElement("div", {
    className: "cn-col-side"
  }, /*#__PURE__*/React.createElement("section", {
    className: "cn-panel"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-panel-head"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-lab"
  }, "GATE LEDGER"), /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-meta"
  }, "recent \xB7 \u03B7")), /*#__PURE__*/React.createElement("div", {
    className: "cn-ledger"
  }, rows.slice(0, 11).map(r => /*#__PURE__*/React.createElement("div", {
    className: "cn-ledger-row",
    key: r.id
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-ts"
  }, r.ts), /*#__PURE__*/React.createElement("span", {
    className: "cn-ledger-dom"
  }, r.domain), /*#__PURE__*/React.createElement("span", {
    className: "cn-num" + (r.withheld ? " anom" : "")
  }, "\u03B7 ", r.benev != null ? r.benev.toFixed(3) : "—"), /*#__PURE__*/React.createElement("span", {
    className: "cn-gate-state sm" + (r.withheld ? " withheld" : "")
  }, r.withheld ? "WITHHELD" : "PASS"))))), /*#__PURE__*/React.createElement(OverrideAudit, {
    rows: rows
  })));
}

/* ---------- COMPLIANCE — posture summary + standard drill-down + correct TLP ---------- */
function ComplianceView({
  selectedStd,
  onPickStd
}) {
  const all = window.COMPLIANCE;
  const openFindings = all.reduce((a, c) => a + c.findings.length, 0);
  const avgCov = all.reduce((a, c) => a + c.coverage, 0) / all.length;
  const sel = selectedStd ? all.find(c => c.std === selectedStd) : null;
  return /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("div", {
    className: "cn-comp-summary"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-cs"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-cs-v"
  }, all.length), /*#__PURE__*/React.createElement("span", {
    className: "cn-cs-k"
  }, "standards tracked")), /*#__PURE__*/React.createElement("div", {
    className: "cn-cs"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-cs-v"
  }, (avgCov * 100).toFixed(0), "%"), /*#__PURE__*/React.createElement("span", {
    className: "cn-cs-k"
  }, "avg control coverage")), /*#__PURE__*/React.createElement("div", {
    className: "cn-cs"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-cs-v" + (openFindings ? " warn" : "")
  }, openFindings), /*#__PURE__*/React.createElement("span", {
    className: "cn-cs-k"
  }, "open findings")), /*#__PURE__*/React.createElement("div", {
    className: "cn-cs"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-cs-v"
  }, all.filter(c => c.state === "ENFORCED" || c.state === "ALIGNED").length, "/", all.length), /*#__PURE__*/React.createElement("span", {
    className: "cn-cs-k"
  }, "aligned / enforced"))), /*#__PURE__*/React.createElement("div", {
    className: "cn-grid" + (sel ? " wide" : "")
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-col-main"
  }, /*#__PURE__*/React.createElement("section", {
    className: "cn-panel"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-panel-head"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-lab"
  }, "NIST CSF 2.0 \xB7 FUNCTION COVERAGE"), /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-meta"
  }, "compliance/nist_csf_integrator.py")), /*#__PURE__*/React.createElement("div", {
    className: "cn-nist"
  }, window.NIST_FUNCTIONS.map(f => /*#__PURE__*/React.createElement("div", {
    className: "cn-nist-row",
    key: f.fn
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-nist-fn"
  }, f.fn), /*#__PURE__*/React.createElement("span", {
    className: "cn-nist-track"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-nist-fill",
    style: {
      width: f.cov * 100 + "%"
    }
  })), /*#__PURE__*/React.createElement("span", {
    className: "cn-nist-pct"
  }, (f.cov * 100).toFixed(0), "%"))))), /*#__PURE__*/React.createElement("div", {
    className: "cn-comp-grid",
    style: {
      marginTop: "16px"
    }
  }, all.map(c => /*#__PURE__*/React.createElement("button", {
    className: "cn-panel cn-compcard" + (selectedStd === c.std ? " on" : ""),
    key: c.std,
    onClick: () => onPickStd(c.std)
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-compcard-top"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-compcard-std"
  }, c.std), /*#__PURE__*/React.createElement("span", {
    className: "cn-gate-state"
  }, c.state)), /*#__PURE__*/React.createElement("div", {
    className: "cn-compcard-note"
  }, c.note), /*#__PURE__*/React.createElement("div", {
    className: "cn-track sm"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-fill",
    style: {
      width: c.coverage * 100 + "%"
    }
  })), /*#__PURE__*/React.createElement("div", {
    className: "cn-compcard-foot"
  }, /*#__PURE__*/React.createElement("span", null, c.controls[0], "/", c.controls[1], " controls"), /*#__PURE__*/React.createElement("span", {
    className: c.findings.length ? "cn-cc-find warn" : "cn-cc-find"
  }, c.findings.length, " findings")))))), /*#__PURE__*/React.createElement("div", {
    className: "cn-col-side"
  }, sel ? /*#__PURE__*/React.createElement(ComplianceDetail, {
    c: sel,
    onClose: () => onPickStd(sel.std)
  }) : /*#__PURE__*/React.createElement("section", {
    className: "cn-panel"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-panel-head"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-lab"
  }, "SEVERITY \u2192 TLP DISCLOSURE"), /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-meta"
  }, "tlp_handler.py")), /*#__PURE__*/React.createElement("div", {
    className: "cn-tlp-note"
  }, "TLP 2.0 governs ", /*#__PURE__*/React.createElement("b", null, "who may receive"), " a report \u2014 not its severity. The console assigns a default tag from severity; an analyst can override."), /*#__PURE__*/React.createElement("div", {
    className: "cn-sevtlp"
  }, window.SEV_TLP_POLICY.map(m => {
    const lvl = window.TLP_LEVELS.find(l => l.tag === m.tlp);
    return /*#__PURE__*/React.createElement("div", {
      className: "cn-sevtlp-row",
      key: m.tlp
    }, /*#__PURE__*/React.createElement("div", {
      className: "cn-sevtlp-l"
    }, /*#__PURE__*/React.createElement("span", {
      className: "cn-sevtlp-sev"
    }, m.sev), /*#__PURE__*/React.createElement("span", {
      className: "cn-sevtlp-rat"
    }, m.rationale)), /*#__PURE__*/React.createElement("span", {
      className: "cn-sevtlp-arrow"
    }, "\u2192"), /*#__PURE__*/React.createElement("span", {
      className: "cn-tlp-tag tlp-" + lvl.band
    }, m.tlp));
  })), /*#__PURE__*/React.createElement("div", {
    className: "cn-tlp-ladder-h"
  }, "DISCLOSURE BOUNDARY \xB7 per level"), /*#__PURE__*/React.createElement("div", {
    className: "cn-tlp-ladder"
  }, window.TLP_LEVELS.map(l => /*#__PURE__*/React.createElement("div", {
    className: "cn-tlp-l-row",
    key: l.tag
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-tlp-dot tlp-" + l.band
  }), /*#__PURE__*/React.createElement("span", {
    className: "cn-tlp-l-tag"
  }, l.tag), /*#__PURE__*/React.createElement("span", {
    className: "cn-tlp-l-bound"
  }, l.boundary))))))));
}
function ComplianceDetail({
  c,
  onClose
}) {
  const sevClass = s => s === "high" ? "crit" : s === "med" ? "med" : "low";
  return /*#__PURE__*/React.createElement("section", {
    className: "cn-panel cn-compdetail"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-panel-head"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-panel-lab"
  }, c.std.toUpperCase()), /*#__PURE__*/React.createElement("button", {
    className: "cn-dd-x",
    onClick: onClose,
    title: "deselect"
  }, "\u2715")), /*#__PURE__*/React.createElement("div", {
    className: "cn-dd-body"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cn-cd-state"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-gate-state"
  }, c.state), /*#__PURE__*/React.createElement("span", {
    className: "cn-cd-cov"
  }, (c.coverage * 100).toFixed(0), "% control coverage")), /*#__PURE__*/React.createElement("div", {
    className: "cn-dd-row"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-dd-k"
  }, "Controls met"), /*#__PURE__*/React.createElement("span", {
    className: "cn-dd-v hl"
  }, c.controls[0], " / ", c.controls[1])), /*#__PURE__*/React.createElement("div", {
    className: "cn-dd-row"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-dd-k"
  }, "Owner"), /*#__PURE__*/React.createElement("span", {
    className: "cn-dd-v"
  }, c.owner)), /*#__PURE__*/React.createElement("div", {
    className: "cn-dd-row"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-dd-k"
  }, "Last assessed"), /*#__PURE__*/React.createElement("span", {
    className: "cn-dd-v"
  }, c.assessed)), /*#__PURE__*/React.createElement("div", {
    className: "cn-dd-row"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-dd-k"
  }, "Next review"), /*#__PURE__*/React.createElement("span", {
    className: "cn-dd-v"
  }, c.review)), /*#__PURE__*/React.createElement("div", {
    className: "cn-dd-row"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-dd-k"
  }, "Module"), /*#__PURE__*/React.createElement("code", {
    className: "cn-dd-file"
  }, c.file)), /*#__PURE__*/React.createElement("div", {
    className: "cn-dd-sec"
  }, "OPEN FINDINGS \xB7 ", c.findings.length), c.findings.length === 0 ? /*#__PURE__*/React.createElement("div", {
    className: "cn-cd-clean"
  }, "\u2713 No open findings \u2014 fully aligned.") : /*#__PURE__*/React.createElement("div", {
    className: "cn-cd-findings"
  }, c.findings.map((f, i) => /*#__PURE__*/React.createElement("div", {
    className: "cn-cd-find",
    key: i
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-cd-sev " + sevClass(f.sev)
  }, f.sev.toUpperCase()), /*#__PURE__*/React.createElement("span", {
    className: "cn-cd-find-t"
  }, f.text)))), /*#__PURE__*/React.createElement("div", {
    className: "cn-dd-sec"
  }, "EVIDENCE"), /*#__PURE__*/React.createElement("div", {
    className: "cn-cd-evidence"
  }, c.evidence.map((e, i) => /*#__PURE__*/React.createElement("div", {
    className: "cn-cd-ev",
    key: i
  }, /*#__PURE__*/React.createElement("span", {
    className: "cn-cd-ev-b"
  }, "\u25AA"), e)))));
}

/* ---------- MODELS — superseded by Capitalize.jsx ▸ ModelsRegistry (bound to
   the real /api/v1/models router). The former static cost/training matrix and
   its window.ENGINES data were removed in the Phase-4 de-dup pass. ---------- */

Object.assign(window, {
  OverviewView,
  DetectionsView,
  DomainsView,
  EnsembleView,
  EthicsView,
  ComplianceView
});
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/console/Views.jsx", error: String((e && e.message) || e) }); }

// ui_kits/crypto/CryptoAnalytics.jsx
try { (() => {
/* Mercury Cryptography — UI Kit · live operation telemetry.
   The kit's unique asset is the streaming op log; this turns it into real
   fleet-level measurement: latency distribution (p50/p95/p99), per-primitive
   mean latency, throughput cadence, and computed outcome mix — none of which the
   single-artifact preview cards can show. Computed live from the op stream. */
const {
  useRef: useRefCx,
  useEffect: useEffectCx,
  useMemo: useMemoCx
} = React;
function cxCss(v) {
  return getComputedStyle(document.documentElement).getPropertyValue(v).trim() || "#888";
}
function cxPctile(arr, p) {
  if (!arr.length) return 0;
  const s = arr.slice().sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor(p * s.length))];
}
function cxMean(arr) {
  return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
}
const OP_HUE = {
  SIGN: "--mercury-300",
  VERIFY: "--normal",
  ENCAPS: "--cognitive-soft",
  DERIVE: "--fg-2"
};
function CxChart({
  draw,
  h
}) {
  const ref = useRefCx(null);
  useEffectCx(() => {
    const cv = ref.current;
    if (!cv) return;
    const render = () => {
      const dpr = window.devicePixelRatio || 1,
        w = cv.clientWidth || 280,
        hh = cv.clientHeight || h || 120;
      cv.width = Math.round(w * dpr);
      cv.height = Math.round(hh * dpr);
      const ctx = cv.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, hh);
      draw(ctx, w, hh);
    };
    render();
    let ro;
    if (window.ResizeObserver) {
      ro = new ResizeObserver(render);
      ro.observe(cv);
    }
    return () => {
      if (ro) ro.disconnect();
    };
  });
  return /*#__PURE__*/React.createElement("canvas", {
    ref: ref,
    style: {
      width: "100%",
      height: (h || 120) + "px",
      display: "block"
    }
  });
}
function CxOpsTelemetry({
  ops
}) {
  const lats = useMemoCx(() => ops.map(o => o.lat), [ops]);
  const stats = useMemoCx(() => {
    const byType = {};
    ops.forEach(o => {
      (byType[o.op] = byType[o.op] || []).push(o.lat);
    });
    const types = ["SIGN", "VERIFY", "ENCAPS", "DERIVE"].filter(t => byType[t]);
    return {
      p50: cxPctile(lats, 0.5),
      p95: cxPctile(lats, 0.95),
      p99: cxPctile(lats, 0.99),
      mean: cxMean(lats),
      max: lats.length ? Math.max(...lats) : 1,
      byType: types.map(t => ({
        t,
        mean: cxMean(byType[t]),
        n: byType[t].length
      })),
      failed: ops.filter(o => o.failed).length,
      flagged: ops.filter(o => o.flagged && !o.failed).length,
      ok: ops.filter(o => !o.failed && !o.flagged).length
    };
  }, [ops, lats]);
  if (!ops.length) {
    return /*#__PURE__*/React.createElement("div", {
      className: "cx-panel"
    }, /*#__PURE__*/React.createElement("div", {
      className: "cx-panel-head"
    }, /*#__PURE__*/React.createElement("span", {
      className: "cx-panel-lab"
    }, "Operation Telemetry")), /*#__PURE__*/React.createElement("div", {
      className: "cx-tel-empty"
    }, "Press ", /*#__PURE__*/React.createElement("b", null, "SIGN & STREAM"), " \u2014 latency percentiles, per-primitive cost, and the outcome mix compute live from the operation stream."));
  }

  // latency distribution histogram, dynamic max, p50/p95 markers
  const drawHist = (ctx, w, h) => {
    const P = 6,
      axisB = 14,
      plotH = h - P - axisB,
      plotW = w - P * 2;
    const top = Math.max(400, Math.ceil(stats.max / 50) * 50);
    const NB = 28,
      bins = new Array(NB).fill(0);
    lats.forEach(l => bins[Math.min(NB - 1, Math.floor(l / top * NB))]++);
    const maxB = Math.max(1, ...bins);
    const X = us => P + us / top * plotW;
    // bars
    for (let i = 0; i < NB; i++) {
      const x = P + i / NB * plotW,
        bh = bins[i] / maxB * plotH;
      ctx.fillStyle = cxCss("--mercury-600");
      ctx.globalAlpha = 0.8;
      ctx.fillRect(x, P + plotH - bh, Math.max(1, plotW / NB - 1), bh);
    }
    ctx.globalAlpha = 1;
    // p50 / p95 markers
    const mark = (us, col, lab) => {
      const x = X(us);
      ctx.strokeStyle = cxCss(col);
      ctx.lineWidth = 1.3;
      ctx.setLineDash([3, 2]);
      ctx.beginPath();
      ctx.moveTo(x, P);
      ctx.lineTo(x, P + plotH);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = cxCss(col);
      ctx.font = "8px monospace";
      ctx.fillText(lab, Math.min(x + 2, w - 26), P + 8);
    };
    mark(stats.p50, "--mercury-200", "p50");
    mark(stats.p95, "--warning-bright", "p95");
    // axis
    ctx.fillStyle = cxCss("--fg-muted");
    ctx.font = "8px monospace";
    ctx.fillText("0", P, h - 3);
    ctx.fillText(top + "µs", w - 28, h - 3);
  };
  const maxMean = Math.max(1, ...stats.byType.map(b => b.mean));
  return /*#__PURE__*/React.createElement("div", {
    className: "cx-panel"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cx-panel-head"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cx-panel-lab"
  }, "Operation Telemetry"), /*#__PURE__*/React.createElement("span", {
    className: "cx-panel-meta"
  }, ops.length, " ops \xB7 live")), /*#__PURE__*/React.createElement("div", {
    className: "cx-tel-pct"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cx-tel-p"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cx-tel-pk"
  }, "p50"), /*#__PURE__*/React.createElement("span", {
    className: "cx-tel-pv"
  }, stats.p50.toFixed(0), /*#__PURE__*/React.createElement("i", null, "\xB5s"))), /*#__PURE__*/React.createElement("div", {
    className: "cx-tel-p"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cx-tel-pk"
  }, "p95"), /*#__PURE__*/React.createElement("span", {
    className: "cx-tel-pv warn"
  }, stats.p95.toFixed(0), /*#__PURE__*/React.createElement("i", null, "\xB5s"))), /*#__PURE__*/React.createElement("div", {
    className: "cx-tel-p"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cx-tel-pk"
  }, "p99"), /*#__PURE__*/React.createElement("span", {
    className: "cx-tel-pv warn"
  }, stats.p99.toFixed(0), /*#__PURE__*/React.createElement("i", null, "\xB5s"))), /*#__PURE__*/React.createElement("div", {
    className: "cx-tel-p"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cx-tel-pk"
  }, "mean"), /*#__PURE__*/React.createElement("span", {
    className: "cx-tel-pv"
  }, stats.mean.toFixed(0), /*#__PURE__*/React.createElement("i", null, "\xB5s")))), /*#__PURE__*/React.createElement("div", {
    className: "cx-tel-sec"
  }, "LATENCY DISTRIBUTION"), /*#__PURE__*/React.createElement("div", {
    className: "cx-tel-canvas"
  }, /*#__PURE__*/React.createElement(CxChart, {
    h: 92,
    draw: drawHist
  })), /*#__PURE__*/React.createElement("div", {
    className: "cx-tel-sec"
  }, "MEAN LATENCY \xB7 BY PRIMITIVE"), /*#__PURE__*/React.createElement("div", {
    className: "cx-tel-bars"
  }, stats.byType.map(b => /*#__PURE__*/React.createElement("div", {
    className: "cx-tel-bar",
    key: b.t
  }, /*#__PURE__*/React.createElement("span", {
    className: "cx-tel-bt",
    style: {
      color: cxCss(OP_HUE[b.t] || "--fg-2")
    }
  }, b.t), /*#__PURE__*/React.createElement("span", {
    className: "cx-tel-track"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cx-tel-fill",
    style: {
      width: b.mean / maxMean * 100 + "%",
      background: cxCss(OP_HUE[b.t] || "--fg-2")
    }
  })), /*#__PURE__*/React.createElement("span", {
    className: "cx-tel-bv"
  }, b.mean.toFixed(0), "\xB5s"), /*#__PURE__*/React.createElement("span", {
    className: "cx-tel-bn"
  }, "\xD7", b.n)))), /*#__PURE__*/React.createElement("div", {
    className: "cx-tel-outcomes"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cx-tel-oc ok"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cx-tel-ocv"
  }, stats.ok), /*#__PURE__*/React.createElement("span", {
    className: "cx-tel-ock"
  }, "verified")), /*#__PURE__*/React.createElement("div", {
    className: "cx-tel-oc flag"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cx-tel-ocv"
  }, stats.flagged), /*#__PURE__*/React.createElement("span", {
    className: "cx-tel-ock"
  }, "3R flagged")), /*#__PURE__*/React.createElement("div", {
    className: "cx-tel-oc fail"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cx-tel-ocv"
  }, stats.failed), /*#__PURE__*/React.createElement("span", {
    className: "cx-tel-ock"
  }, "tampered"))));
}
Object.assign(window, {
  CxOpsTelemetry
});
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/crypto/CryptoAnalytics.jsx", error: String((e && e.message) || e) }); }

// ui_kits/crypto/CryptoApp.jsx
try { (() => {
/* Mercury Cryptography — UI Kit · views + app shell.
   Streams hybrid sign/verify/encaps operations; 3R anomalies drive adaptive posture. */

const POSTURE_ORDER = ["STANDARD", "ELEVATED", "HARDENED", "LOCKDOWN"];
function escalate(cur) {
  const i = POSTURE_ORDER.indexOf(cur);
  return POSTURE_ORDER[Math.min(i + 1, 2)];
} // cap at HARDENED for a flag
function deescalate(cur) {
  const i = POSTURE_ORDER.indexOf(cur);
  return POSTURE_ORDER[Math.max(i - 1, 0)];
}

/* ===================== OPERATIONS (default dashboard) ===================== */
function OperationsView({
  ops,
  selId,
  onSelect,
  live,
  posture,
  alerts,
  onGoto
}) {
  const verified = ops.filter(o => !o.failed && (o.result === "VERIFIED" || o.result === "VALID" || o.result === "SEALED" || o.result === "OK" || o.result === "BOUND")).length;
  const fails = ops.filter(o => o.failed).length;
  const lats = ops.map(o => o.lat);
  const p95 = lats.length ? lats.slice().sort((a, b) => a - b)[Math.min(lats.length - 1, Math.floor(0.95 * lats.length))] : 0;
  const verifyRate = ops.length ? verified / ops.length * 100 : 0;
  const sel = ops.find(o => o.id === selId);
  return /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("div", {
    className: "cx-stats"
  }, /*#__PURE__*/React.createElement(CxStat, {
    lead: true,
    lab: "OPERATIONS \xB7 live",
    val: ops.length || "—",
    sub: ops.length ? "streaming window" : "press stream"
  }), /*#__PURE__*/React.createElement(CxStat, {
    lab: "p95 LATENCY",
    val: ops.length ? p95.toFixed(0) : "—",
    sub: ops.length ? "µs · 95th pct" : "—"
  }), /*#__PURE__*/React.createElement(CxStat, {
    lab: "VERIFY RATE",
    val: ops.length ? verifyRate.toFixed(0) + "%" : "—",
    cls: fails > 0 ? "warn" : "ok",
    sub: fails > 0 ? fails + " tampered ✕" : "all-layer pass"
  }), /*#__PURE__*/React.createElement("button", {
    className: "cx-stat-btn",
    onClick: () => onGoto("monitor"),
    title: "open 3R monitor"
  }, /*#__PURE__*/React.createElement(CxStat, {
    lab: "POSTURE \u2197",
    val: posture,
    cls: "post",
    sub: alerts > 0 ? alerts + " 3R alerts · review" : "adaptive · nominal"
  }))), /*#__PURE__*/React.createElement("div", {
    className: "cx-grid"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cx-col-main"
  }, /*#__PURE__*/React.createElement(CxOpLog, {
    ops: ops,
    selId: selId,
    onSelect: onSelect,
    live: live
  })), /*#__PURE__*/React.createElement("div", {
    className: "cx-col-side"
  }, /*#__PURE__*/React.createElement(CxPackageDetail, {
    op: sel,
    onInspect: sel ? () => onGoto("package") : null
  }), /*#__PURE__*/React.createElement(CxOpsTelemetry, {
    ops: ops
  }))));
}

/* ===================== ALGORITHMS ===================== */
function AlgorithmsView() {
  const {
    PQC_ALGOS,
    CLASSICAL
  } = window.CRYPTO;
  return /*#__PURE__*/React.createElement("div", {
    className: "cx-stack-views"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cx-hd-row"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cx-pqc"
  }, "PQC"), /*#__PURE__*/React.createElement("span", {
    className: "cx-sec-h"
  }, "NIST Post-Quantum Set"), /*#__PURE__*/React.createElement("span", {
    className: "cx-sec-meta"
  }, "native C \xB7 no external deps")), /*#__PURE__*/React.createElement("div", {
    className: "cx-algo-grid"
  }, PQC_ALGOS.map(a => /*#__PURE__*/React.createElement("div", {
    className: "cx-algo",
    key: a.id
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: "cx-algo-n"
  }, a.name), /*#__PURE__*/React.createElement("div", {
    className: "cx-algo-role"
  }, a.role)), /*#__PURE__*/React.createElement("div", {
    className: "cx-algo-fips"
  }, a.fips, " \xB7 ", a.sec), /*#__PURE__*/React.createElement("div", {
    className: "cx-algo-perf"
  }, a.perf), /*#__PURE__*/React.createElement("div", {
    className: "cx-algo-lin"
  }, a.lineage, /*#__PURE__*/React.createElement("span", null, " \xB7 ", a.detail))))), /*#__PURE__*/React.createElement("div", {
    className: "cx-hd-row"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cx-sec-h",
    style: {
      color: "var(--mercury-300)"
    }
  }, "Classical & Transport"), /*#__PURE__*/React.createElement("span", {
    className: "cx-sec-meta"
  }, "constant-time C \xB7 AES-NI / NEON / SVE2 dispatch")), /*#__PURE__*/React.createElement("div", {
    className: "cx-class-grid"
  }, CLASSICAL.map(c => /*#__PURE__*/React.createElement("div", {
    className: "cx-class",
    key: c.name
  }, /*#__PURE__*/React.createElement("span", {
    className: "cx-class-n"
  }, c.name), /*#__PURE__*/React.createElement("span", {
    className: "cx-class-note"
  }, c.note), /*#__PURE__*/React.createElement("span", {
    className: "cx-class-perf"
  }, c.perf)))), /*#__PURE__*/React.createElement("div", {
    className: "cx-view-note"
  }, "Hybrid by default \u2014 classical ", /*#__PURE__*/React.createElement("b", null, "Ed25519"), " binds with ", /*#__PURE__*/React.createElement("b", null, "ML-DSA-65"), " so a package is valid today and resists ", /*#__PURE__*/React.createElement("b", null, "harvest-now-decrypt-later"), ". Throughput is Python-API-via-ctypes on the canonical bench host; self-attested against NIST ACVP vectors, not externally audited."));
}

/* ===================== 3R MONITOR ===================== */
const CYTHON = [["Lyapunov function", "12.3ms", "0.45ms", "27.3×"], ["Matrix-vector 500²", "8.7ms", "0.31ms", "28.1×"], ["NTT (degree 256)", "45.2ms", "1.2ms", "37.7×"], ["Helix evolution", "3.4ms", "0.18ms", "18.9×"]];
function MonitorView({
  alerts,
  posture
}) {
  return /*#__PURE__*/React.createElement("div", {
    className: "cx-grid mon-grid"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cx-col-main"
  }, /*#__PURE__*/React.createElement(CxMonitor, {
    alerts: alerts,
    posture: posture
  }), /*#__PURE__*/React.createElement("div", {
    className: "cx-panel",
    style: {
      marginTop: "16px"
    }
  }, /*#__PURE__*/React.createElement("div", {
    className: "cx-panel-head"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cx-panel-lab"
  }, "Cython Acceleration"), /*#__PURE__*/React.createElement("span", {
    className: "cx-panel-meta"
  }, "3R math engine \xB7 vs pure-Python baseline")), /*#__PURE__*/React.createElement("div", {
    className: "cx-cy-head"
  }, /*#__PURE__*/React.createElement("span", null, "OPERATION"), /*#__PURE__*/React.createElement("span", {
    className: "r"
  }, "PURE PY"), /*#__PURE__*/React.createElement("span", {
    className: "r"
  }, "CYTHON"), /*#__PURE__*/React.createElement("span", {
    className: "r"
  }, "SPEEDUP")), CYTHON.map(r => /*#__PURE__*/React.createElement("div", {
    className: "cx-cy-row",
    key: r[0]
  }, /*#__PURE__*/React.createElement("span", null, r[0]), /*#__PURE__*/React.createElement("span", {
    className: "r dim"
  }, r[1]), /*#__PURE__*/React.createElement("span", {
    className: "r"
  }, r[2]), /*#__PURE__*/React.createElement("span", {
    className: "r hl"
  }, r[3]))), /*#__PURE__*/React.createElement("div", {
    className: "cx-panel-foot"
  }, "Speedup applies to ", /*#__PURE__*/React.createElement("b", null, "3R monitoring math"), " only \u2014 never to the C-implemented cryptographic primitives. Measure locally; not a CI-gated number."))), /*#__PURE__*/React.createElement("div", {
    className: "cx-col-side"
  }, /*#__PURE__*/React.createElement(CxPosture, {
    current: posture
  }), /*#__PURE__*/React.createElement("div", {
    className: "cx-panel"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cx-panel-head"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cx-panel-lab"
  }, "What 3R Is")), /*#__PURE__*/React.createElement("div", {
    className: "cx-prose"
  }, "The ", /*#__PURE__*/React.createElement("b", null, "3R"), " framework \u2014 Resonance, Recursion, Refactoring \u2014 is shared mathematical kin with Mercury Agent's detection core. Here it provides ", /*#__PURE__*/React.createElement("b", null, "runtime visibility"), " into cryptographic-operation behavior, surfacing statistical anomalies that may be consistent with cache-timing or power behavior. It ", /*#__PURE__*/React.createElement("b", null, "does not"), " guarantee detection or prevention of side-channel attacks."))));
}

/* ===================== PACKAGE INSPECTOR ===================== */
function PackageView({
  ops,
  selId,
  onSelect
}) {
  const sel = ops.find(o => o.id === selId) || ops[0];
  return /*#__PURE__*/React.createElement("div", {
    className: "cx-grid pkg-grid"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cx-col-main"
  }, /*#__PURE__*/React.createElement(CxPackageDetail, {
    op: sel
  })), /*#__PURE__*/React.createElement("div", {
    className: "cx-col-side"
  }, /*#__PURE__*/React.createElement(CxDefense, null), /*#__PURE__*/React.createElement("div", {
    className: "cx-panel"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cx-panel-head"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cx-panel-lab"
  }, "Verify Pipeline")), /*#__PURE__*/React.createElement("div", {
    className: "cx-prose"
  }, "A ", /*#__PURE__*/React.createElement("b", null, "crypto_package v1"), " verdict is ", /*#__PURE__*/React.createElement("b", null, "all-or-nothing"), ": content digest, keyed auth, classical + post-quantum signatures, derived-key binding, and optional timestamp must all pass. Canonical encoding normalizes input first so serialization is deterministic. Any mismatch raises an error \u2014 there is no soft warning, no advisory mode.")), ops.length > 0 && /*#__PURE__*/React.createElement("div", {
    className: "cx-panel"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cx-panel-head"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cx-panel-lab"
  }, "Recent Packages")), /*#__PURE__*/React.createElement("div", {
    className: "cx-recent"
  }, ops.slice(0, 6).map(o => /*#__PURE__*/React.createElement("button", {
    key: o.id,
    className: "cx-recent-row" + (o.id === (sel && sel.id) ? " on" : "") + (o.failed ? " fail" : ""),
    onClick: () => onSelect(o.id)
  }, /*#__PURE__*/React.createElement("span", {
    className: "cx-recent-seq"
  }, "#", String(o.seq).padStart(3, "0")), /*#__PURE__*/React.createElement("span", {
    className: "cx-recent-op"
  }, o.op), /*#__PURE__*/React.createElement("span", {
    className: "cx-recent-res " + (o.failed ? "fail" : "ok")
  }, o.result)))))));
}

/* ===================== OVERVIEW (landing) ===================== */
function OverviewView({
  onGoto,
  posture,
  alerts
}) {
  const {
    ACVP_META,
    BENCH_META,
    CONSTTIME,
    KIN
  } = window.CRYPTO;
  const ctPass = CONSTTIME.filter(c => c.res === "PASS").length;
  const tiles = [{
    k: "NIST ACVP",
    v: ACVP_META.passed.toLocaleString() + " / " + ACVP_META.tested.toLocaleString(),
    sub: "vectors passed · " + ACVP_META.algos + " algorithms",
    cls: "ok",
    go: "compliance",
    cta: "Conformance →"
  }, {
    k: "Constant-time",
    v: ctPass + " PASS",
    sub: "dudect Welch t-test · |t|<4.5",
    cls: "cy",
    go: "compliance",
    cta: "Side-channel →"
  }, {
    k: "Regression gate",
    v: BENCH_META.passed + " / " + BENCH_META.total,
    sub: "benchmarks above floor",
    cls: "ok",
    go: "benchmarks",
    cta: "Benchmarks →"
  }, {
    k: "Defense depth",
    v: "4 +2",
    sub: "core + supporting layers",
    cls: "vi",
    go: "package",
    cta: "Inspector →"
  }];
  return /*#__PURE__*/React.createElement("div", {
    className: "cx-stack-views"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cx-ov-hero"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cx-ov-hero-l"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cx-ov-sig"
  }, "\u2671"), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: "cx-ov-title"
  }, "AMA Cryptography"), /*#__PURE__*/React.createElement("div", {
    className: "cx-ov-tag"
  }, "Post-quantum security substrate \xB7 hybrid Ed25519 + ML-DSA-65 \xB7 Apache-2.0 \xB7 v3.2.0"))), /*#__PURE__*/React.createElement("div", {
    className: "cx-ov-secures"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cx-ov-sec-k"
  }, "SECURES"), /*#__PURE__*/React.createElement("button", {
    className: "cx-ov-sec-link",
    onClick: () => onGoto("kin")
  }, /*#__PURE__*/React.createElement("span", {
    className: "cx-ov-sec-sig"
  }, "\u2671"), " Mercury Agent"), /*#__PURE__*/React.createElement("button", {
    className: "cx-ov-sec-link",
    onClick: () => onGoto("kin")
  }, /*#__PURE__*/React.createElement("span", {
    className: "cx-ov-sec-sig",
    style: {
      color: "var(--brand-gold)"
    }
  }, "\u03A9"), " FIND\u03A9YOU\u2122"))), /*#__PURE__*/React.createElement("div", {
    className: "cx-hd-row"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cx-sec-h"
  }, "Assurance \xB7 measured"), /*#__PURE__*/React.createElement("span", {
    className: "cx-sec-meta"
  }, "self-attested \xB7 not externally audited")), /*#__PURE__*/React.createElement("div", {
    className: "cx-ov-tiles"
  }, tiles.map(t => /*#__PURE__*/React.createElement("button", {
    className: "cx-ov-tile",
    key: t.k,
    onClick: () => onGoto(t.go)
  }, /*#__PURE__*/React.createElement("div", {
    className: "cx-ov-tk"
  }, t.k), /*#__PURE__*/React.createElement("div", {
    className: "cx-ov-tv " + t.cls
  }, t.v), /*#__PURE__*/React.createElement("div", {
    className: "cx-ov-tsub"
  }, t.sub), /*#__PURE__*/React.createElement("div", {
    className: "cx-ov-tcta"
  }, t.cta)))), /*#__PURE__*/React.createElement("div", {
    className: "cx-ov-cols"
  }, /*#__PURE__*/React.createElement("button", {
    className: "cx-ov-state",
    onClick: () => onGoto("monitor"),
    style: {
      "--sc": alerts > 0 ? "var(--warning-bright)" : "var(--normal)"
    }
  }, /*#__PURE__*/React.createElement("div", {
    className: "cx-ov-state-top"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cx-ov-state-k"
  }, "RUNTIME STATE"), /*#__PURE__*/React.createElement("span", {
    className: "cx-ov-state-go"
  }, "3R Monitor \u2192")), /*#__PURE__*/React.createElement("div", {
    className: "cx-ov-state-row"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cx-ov-state-dot"
  }), /*#__PURE__*/React.createElement("span", {
    className: "cx-ov-state-v"
  }, "POSTURE \xB7 ", posture)), /*#__PURE__*/React.createElement("div", {
    className: "cx-ov-state-sub"
  }, alerts > 0 ? alerts + " 3R anomaly alert" + (alerts > 1 ? "s" : "") + " · under review" : "3R nominal · Resonance · Recursion · Refactoring")), /*#__PURE__*/React.createElement("button", {
    className: "cx-ov-state",
    onClick: () => onGoto("operations"),
    style: {
      "--sc": "var(--mercury-400)"
    }
  }, /*#__PURE__*/React.createElement("div", {
    className: "cx-ov-state-top"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cx-ov-state-k"
  }, "OPERATIONS"), /*#__PURE__*/React.createElement("span", {
    className: "cx-ov-state-go"
  }, "Live stream \u2192")), /*#__PURE__*/React.createElement("div", {
    className: "cx-ov-state-row"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cx-ov-state-dot"
  }), /*#__PURE__*/React.createElement("span", {
    className: "cx-ov-state-v"
  }, "hybrid sign \xB7 verify \xB7 encaps")), /*#__PURE__*/React.createElement("div", {
    className: "cx-ov-state-sub"
  }, "stream the pipeline \xB7 latency telemetry \xB7 package inspector"))), /*#__PURE__*/React.createElement("div", {
    className: "cx-view-note"
  }, "AMA Cryptography is the cryptographic foundation beneath ", /*#__PURE__*/React.createElement("b", null, "Mercury Agent"), " and ", /*#__PURE__*/React.createElement("b", null, "FIND\u03A9YOU\u2122"), " \u2014 every figure on this console comes from the library's own measured datasets (ACVP vectors, dudect timing, CI benchmarks). It ships a C / Cython / Python library and CLI; this operator surface arranges that data in the Mercury brand. Self-attested, ", /*#__PURE__*/React.createElement("b", null, "not externally audited"), "."));
}

/* ===================== COMPLIANCE ===================== */
function ComplianceView() {
  const {
    COMPLIANCE,
    ACVP,
    ACVP_META,
    CONSTTIME,
    CONSTTIME_META
  } = window.CRYPTO;
  const maxV = Math.max(...ACVP.map(a => a.v));
  return /*#__PURE__*/React.createElement("div", {
    className: "cx-stack-views"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cx-hd-row"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cx-sec-h",
    style: {
      color: "var(--normal-bright)"
    }
  }, "NIST ACVP Conformance"), /*#__PURE__*/React.createElement("span", {
    className: "cx-sec-meta"
  }, "ref ", ACVP_META.ref, " \xB7 ", ACVP_META.date)), /*#__PURE__*/React.createElement("div", {
    className: "cx-acvp-strip"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cx-bs"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cx-bs-v ok"
  }, ACVP_META.passed.toLocaleString()), /*#__PURE__*/React.createElement("div", {
    className: "cx-bs-k"
  }, "vectors passed")), /*#__PURE__*/React.createElement("div", {
    className: "cx-bs"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cx-bs-v"
  }, ACVP_META.failed), /*#__PURE__*/React.createElement("div", {
    className: "cx-bs-k"
  }, "failed")), /*#__PURE__*/React.createElement("div", {
    className: "cx-bs"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cx-bs-v"
  }, ACVP_META.algos), /*#__PURE__*/React.createElement("div", {
    className: "cx-bs-k"
  }, "algorithms")), /*#__PURE__*/React.createElement("div", {
    className: "cx-bs"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cx-bs-v",
    style: {
      fontSize: "13px",
      color: "var(--cognitive-soft)"
    }
  }, "native C"), /*#__PURE__*/React.createElement("div", {
    className: "cx-bs-k"
  }, "no liboqs / PQClean"))), /*#__PURE__*/React.createElement("div", {
    className: "cx-panel"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cx-panel-head"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cx-panel-lab"
  }, "Validated Algorithms"), /*#__PURE__*/React.createElement("span", {
    className: "cx-panel-meta"
  }, "vectors tested vs upstream ACVP-Server")), /*#__PURE__*/React.createElement("div", {
    className: "cx-acvp-list"
  }, ACVP.map(a => /*#__PURE__*/React.createElement("div", {
    className: "cx-acvp-row",
    key: a.name
  }, /*#__PURE__*/React.createElement("span", {
    className: "cx-acvp-n"
  }, a.name, a.pqc && /*#__PURE__*/React.createElement("span", {
    className: "cx-pqc sm"
  }, "PQC")), /*#__PURE__*/React.createElement("span", {
    className: "cx-acvp-std"
  }, a.std), /*#__PURE__*/React.createElement("span", {
    className: "cx-acvp-track"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cx-acvp-fill",
    style: {
      width: a.v / maxV * 100 + "%",
      background: a.pqc ? "var(--cognitive)" : "var(--mercury-500)"
    }
  })), /*#__PURE__*/React.createElement("span", {
    className: "cx-acvp-v"
  }, a.v), /*#__PURE__*/React.createElement("span", {
    className: "cx-acvp-gate"
  }, "\u2713 ", a.v, "/", a.v)))), /*#__PURE__*/React.createElement("div", {
    className: "cx-panel-foot",
    style: {
      borderTop: "1px solid var(--border-faint)"
    }
  }, ACVP_META.disclaimer)), /*#__PURE__*/React.createElement("div", {
    className: "cx-hd-row"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cx-sec-h",
    style: {
      color: "var(--mercury-300)"
    }
  }, "Constant-Time Evidence"), /*#__PURE__*/React.createElement("span", {
    className: "cx-sec-meta"
  }, CONSTTIME_META.method, " \xB7 gate ", CONSTTIME_META.gate)), /*#__PURE__*/React.createElement("div", {
    className: "cx-panel"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cx-panel-head"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cx-panel-lab"
  }, "dudect \xB7 empirical timing"), /*#__PURE__*/React.createElement("span", {
    className: "cx-panel-meta"
  }, CONSTTIME_META.measurements, " \xB7 ", CONSTTIME_META.conf, " confidence")), /*#__PURE__*/React.createElement("div", {
    className: "cx-ct-head"
  }, /*#__PURE__*/React.createElement("span", null, "function"), /*#__PURE__*/React.createElement("span", null, "class"), /*#__PURE__*/React.createElement("span", null, "Welch |t|"), /*#__PURE__*/React.createElement("span", {
    className: "cx-ct-spark-h"
  }, "leak margin (t vs 4.5 gate)"), /*#__PURE__*/React.createElement("span", {
    className: "r"
  }, "result")), /*#__PURE__*/React.createElement("div", {
    className: "cx-ct-list"
  }, CONSTTIME.map(c => {
    const frac = Math.min(1, c.t / 9);
    const gateFrac = 4.5 / 9;
    return /*#__PURE__*/React.createElement("div", {
      className: "cx-ct-row" + (c.res === "INFO" ? " info" : ""),
      key: c.fn,
      title: c.note
    }, /*#__PURE__*/React.createElement("span", {
      className: "cx-ct-fn"
    }, c.fn), /*#__PURE__*/React.createElement("span", {
      className: "cx-ct-grp"
    }, c.grp), /*#__PURE__*/React.createElement("span", {
      className: "cx-ct-t"
    }, c.t.toFixed(1)), /*#__PURE__*/React.createElement("span", {
      className: "cx-ct-spark"
    }, /*#__PURE__*/React.createElement("span", {
      className: "cx-ct-bar",
      style: {
        width: frac * 100 + "%",
        background: c.res === "INFO" ? "var(--warning)" : "var(--normal)"
      }
    }), /*#__PURE__*/React.createElement("span", {
      className: "cx-ct-gate",
      style: {
        left: gateFrac * 100 + "%"
      },
      title: "4.5 leak-detection threshold"
    })), /*#__PURE__*/React.createElement("span", {
      className: "r"
    }, /*#__PURE__*/React.createElement("span", {
      className: "cx-ct-res " + (c.res === "INFO" ? "info" : "ok")
    }, c.res)));
  })), /*#__PURE__*/React.createElement("div", {
    className: "cx-panel-foot",
    style: {
      borderTop: "1px solid var(--border-faint)"
    }
  }, /*#__PURE__*/React.createElement("b", null, "PASS"), " = |t| below 4.5, no detectable timing leak. ", /*#__PURE__*/React.createElement("b", null, "INFO"), " = expected variation (AES table S-box cache; ML-DSA rejection sampling is intentional). This is the side-channel evidence layer the ", /*#__PURE__*/React.createElement("b", null, "3R monitor"), " does not itself guarantee.")), /*#__PURE__*/React.createElement("div", {
    className: "cx-hd-row"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cx-sec-h"
  }, "Standardized Primitives"), /*#__PURE__*/React.createElement("span", {
    className: "cx-sec-meta"
  }, "no custom ciphers / hashes / schemes")), /*#__PURE__*/React.createElement("div", {
    className: "cx-comp-grid"
  }, COMPLIANCE.map(c => /*#__PURE__*/React.createElement("div", {
    className: "cx-comp",
    key: c.std
  }, /*#__PURE__*/React.createElement("div", {
    className: "cx-comp-top"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cx-comp-std"
  }, c.std), /*#__PURE__*/React.createElement("span", {
    className: "cx-comp-tag"
  }, c.tag)), /*#__PURE__*/React.createElement("div", {
    className: "cx-comp-note"
  }, c.note)))), /*#__PURE__*/React.createElement("div", {
    className: "cx-view-note"
  }, "Built ", /*#__PURE__*/React.createElement("b", null, "exclusively from standardized primitives"), " (NIST FIPS, IETF RFC) \u2014 the composition protocol is original to Steel Security Advisors, but every primitive is a published standard, validated against ", /*#__PURE__*/React.createElement("b", null, "1,215 NIST ACVP vectors"), " across 12 functions. Self-attested; ", /*#__PURE__*/React.createElement("b", null, "not externally audited"), "."));
}

/* ===================== KIN SYSTEMS ===================== */
function KinView() {
  const {
    KIN
  } = window.CRYPTO;
  return /*#__PURE__*/React.createElement("div", {
    className: "cx-stack-views"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cx-kin-nodes"
  }, KIN.map((k, i) => /*#__PURE__*/React.createElement(React.Fragment, {
    key: k.name
  }, i > 0 && /*#__PURE__*/React.createElement("div", {
    className: "cx-kin-link"
  }, "\u27F7"), /*#__PURE__*/React.createElement("div", {
    className: "cx-kin-node",
    style: {
      "--nc": k.hue
    }
  }, /*#__PURE__*/React.createElement("div", {
    className: "cx-kin-sig"
  }, k.sig), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: "cx-kin-role"
  }, k.role), /*#__PURE__*/React.createElement("div", {
    className: "cx-kin-name wm wm-dark"
  }, /*#__PURE__*/React.createElement(Wordmark, {
    which: k.wm
  }))), /*#__PURE__*/React.createElement("div", {
    className: "cx-kin-desc"
  }, k.desc), /*#__PURE__*/React.createElement("div", {
    className: "cx-kin-lic"
  }, k.lic))))), /*#__PURE__*/React.createElement("div", {
    className: "cx-kin-backbone"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cx-kin-bk"
  }, "SHARED"), /*#__PURE__*/React.createElement("span", {
    className: "cx-kin-bv"
  }, "cryptographic backbone ", /*#__PURE__*/React.createElement("span", null, "\xB7"), " ethical alignment posture ", /*#__PURE__*/React.createElement("span", null, "\xB7"), " survivor-first mission")), /*#__PURE__*/React.createElement("div", {
    className: "cx-view-note"
  }, "Mercury Agent, FIND\u03A9YOU\u2122, and AMA Cryptography form a single ", /*#__PURE__*/React.createElement("b", null, "civilization-first lineage"), ". Each is independently deployable, but they are designed as kin \u2014 both Mercury Agent and FIND\u03A9YOU\u2122 derive their post-quantum security from ", /*#__PURE__*/React.createElement("b", null, "AMA"), ", the foundation the other two stand on."));
}

/* ===================== BENCHMARKS ===================== */
function BenchmarksView() {
  const {
    BENCHMARKS,
    BENCH_META
  } = window.CRYPTO;
  const [sortByGain, setSortByGain] = React.useState(false);
  // global max ops for log scaling across all items
  const all = BENCHMARKS.flatMap(g => g.items);
  const maxOps = Math.max(...all.map(i => i.ops));
  const logMax = Math.log10(maxOps);
  const w = ops => Math.max(3, Math.log10(ops) / logMax * 100);
  const fmt = n => n >= 1000 ? (n / 1000).toFixed(n >= 100000 ? 0 : 1) + "k" : String(n);
  return /*#__PURE__*/React.createElement("div", {
    className: "cx-stack-views"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cx-bench-strip"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cx-bs"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cx-bs-v ok"
  }, BENCH_META.passed, "/", BENCH_META.total), /*#__PURE__*/React.createElement("div", {
    className: "cx-bs-k"
  }, "regression gate \xB7 passed")), /*#__PURE__*/React.createElement("div", {
    className: "cx-bs"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cx-bs-v"
  }, BENCH_META.failed), /*#__PURE__*/React.createElement("div", {
    className: "cx-bs-k"
  }, "failed")), /*#__PURE__*/React.createElement("div", {
    className: "cx-bs"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cx-bs-v"
  }, "5"), /*#__PURE__*/React.createElement("div", {
    className: "cx-bs-k"
  }, "primitive families")), /*#__PURE__*/React.createElement("div", {
    className: "cx-bs"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cx-bs-v",
    style: {
      fontSize: "13px"
    }
  }, BENCH_META.ran), /*#__PURE__*/React.createElement("div", {
    className: "cx-bs-k"
  }, BENCH_META.host)), /*#__PURE__*/React.createElement("button", {
    className: "cx-bench-sort" + (sortByGain ? " on" : ""),
    onClick: () => setSortByGain(v => !v)
  }, sortByGain ? "◉ by headroom" : "◯ by family")), BENCHMARKS.map(g => {
    const items = sortByGain ? g.items.slice().sort((a, b) => b.ops / b.base - a.ops / a.base) : g.items;
    return /*#__PURE__*/React.createElement("div", {
      className: "cx-panel",
      key: g.group
    }, /*#__PURE__*/React.createElement("div", {
      className: "cx-panel-head"
    }, /*#__PURE__*/React.createElement("span", {
      className: "cx-panel-lab"
    }, g.group), /*#__PURE__*/React.createElement("span", {
      className: "cx-panel-meta"
    }, "ops/sec vs regression baseline")), /*#__PURE__*/React.createElement("div", {
      className: "cx-bench-head"
    }, /*#__PURE__*/React.createElement("span", null, "operation"), /*#__PURE__*/React.createElement("span", {
      className: "r"
    }, "ops/sec"), /*#__PURE__*/React.createElement("span", null, "throughput \xB7 log scale"), /*#__PURE__*/React.createElement("span", {
      className: "r"
    }, "\xD7baseline"), /*#__PURE__*/React.createElement("span", {
      className: "r"
    }, "gate")), /*#__PURE__*/React.createElement("div", {
      className: "cx-bench-list"
    }, items.map(b => {
      const gain = b.ops / b.base; // measured / regression floor
      const headroomPct = (b.ops - b.base) / b.base * 100;
      const floorPct = w(b.base),
        opsPct = w(b.ops);
      return /*#__PURE__*/React.createElement("div", {
        className: "cx-bench-row",
        key: b.name
      }, /*#__PURE__*/React.createElement("span", {
        className: "cx-bench-n"
      }, b.name, b.pqc && /*#__PURE__*/React.createElement("span", {
        className: "cx-pqc sm"
      }, "PQC"), /*#__PURE__*/React.createElement("span", {
        className: "cx-bench-op"
      }, b.op)), /*#__PURE__*/React.createElement("span", {
        className: "r cx-bench-ops",
        style: {
          color: g.hue
        }
      }, fmt(b.ops)), /*#__PURE__*/React.createElement("span", {
        className: "cx-bench-track"
      }, /*#__PURE__*/React.createElement("span", {
        className: "cx-bench-fill",
        style: {
          width: opsPct + "%",
          background: g.hue
        }
      }), /*#__PURE__*/React.createElement("span", {
        className: "cx-bench-floor",
        style: {
          left: floorPct + "%"
        },
        title: "regression floor " + fmt(b.base) + " ops/s"
      })), /*#__PURE__*/React.createElement("span", {
        className: "r cx-bench-gain"
      }, gain.toFixed(1), "\xD7"), /*#__PURE__*/React.createElement("span", {
        className: "r"
      }, /*#__PURE__*/React.createElement("span", {
        className: "cx-bench-gate ok",
        title: "within ±" + b.tol + "% tolerance · " + headroomPct.toFixed(0) + "% above floor"
      }, "PASS")));
    })));
  }), /*#__PURE__*/React.createElement("div", {
    className: "cx-view-note"
  }, "Live regression dataset from ", /*#__PURE__*/React.createElement("b", null, "benchmarks/benchmark-results.json"), " \u2014 every operation must hold above its baseline within a tolerance band or the CI gate fails the build. The ", /*#__PURE__*/React.createElement("span", {
    style: {
      color: "var(--fg-muted)"
    }
  }, "\u2502"), " tick marks each operation's regression floor; the bar is the measured throughput (log scale). All figures are native-C-via-ctypes on the canonical host; self-attested, ", /*#__PURE__*/React.createElement("b", null, "not externally audited"), "."));
}

/* ===================== APP SHELL ===================== */
function CryptoApp() {
  const [page, setPage] = React.useState("overview");
  const [ops, setOps] = React.useState([]);
  const [selId, setSelId] = React.useState(null);
  const [live, setLive] = React.useState(false);
  const [posture, setPosture] = React.useState("STANDARD");
  const [alerts, setAlerts] = React.useState(0);
  const timer = React.useRef(null);
  const settle = React.useRef(null);
  const run = React.useCallback(() => {
    if (live) return;
    clearTimeout(settle.current);
    setLive(true);
    let n = 0;
    const total = 11;
    timer.current = setInterval(() => {
      const bias = n > 6 ? 0.05 : 0; // a little more likely to flag mid-run
      const o = window.genOp(bias);
      setOps(prev => [o, ...prev].slice(0, 60));
      if (o.flagged || o.failed) {
        setAlerts(a => a + 1);
        setSelId(o.id);
      }
      setPosture(cur => o.failed ? "LOCKDOWN" : o.flagged ? escalate(cur) : cur);
      n += 1;
      if (n >= total) {
        clearInterval(timer.current);
        setLive(false);
        // settle posture one notch toward standard after a clean beat
        settle.current = setTimeout(() => setPosture(cur => cur === "LOCKDOWN" ? "HARDENED" : deescalate(cur)), 1400);
      }
    }, 300);
  }, [live]);
  React.useEffect(() => () => {
    clearInterval(timer.current);
    clearTimeout(settle.current);
  }, []);
  // seed an initial selection on first op
  React.useEffect(() => {
    if (!selId && ops.length) {
      setSelId(ops[0].id);
    }
  }, [ops, selId]);
  const view = (() => {
    switch (page) {
      case "algorithms":
        return /*#__PURE__*/React.createElement(AlgorithmsView, null);
      case "benchmarks":
        return /*#__PURE__*/React.createElement(BenchmarksView, null);
      case "monitor":
        return /*#__PURE__*/React.createElement(MonitorView, {
          alerts: alerts,
          posture: posture
        });
      case "package":
        return /*#__PURE__*/React.createElement(PackageView, {
          ops: ops,
          selId: selId,
          onSelect: setSelId
        });
      case "compliance":
        return /*#__PURE__*/React.createElement(ComplianceView, null);
      case "kin":
        return /*#__PURE__*/React.createElement(KinView, null);
      case "overview":
        return /*#__PURE__*/React.createElement(OverviewView, {
          onGoto: setPage,
          posture: posture,
          alerts: alerts
        });
      default:
        return /*#__PURE__*/React.createElement(OperationsView, {
          ops: ops,
          selId: selId,
          onSelect: setSelId,
          live: live,
          posture: posture,
          alerts: alerts,
          onGoto: setPage
        });
    }
  })();
  return /*#__PURE__*/React.createElement("div", {
    className: "cx-app"
  }, /*#__PURE__*/React.createElement(CxSidebar, {
    active: page,
    setActive: setPage,
    posture: posture
  }), /*#__PURE__*/React.createElement("div", {
    className: "cx-mainwrap"
  }, /*#__PURE__*/React.createElement(CxTopbar, {
    live: live,
    onRun: run,
    page: page,
    posture: posture
  }), /*#__PURE__*/React.createElement("div", {
    className: "cx-disclaimer"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cx-disc-tag"
  }, "REFERENCE CONCEPT"), /*#__PURE__*/React.createElement("span", null, "AMA Cryptography ships a C / Cython / Python library and CLI \u2014 no GUI. This operator surface arranges only the documented data model in the Mercury brand system. Not a screenshot of a real product screen.")), /*#__PURE__*/React.createElement("div", {
    className: "cx-scroll"
  }, view)));
}
ReactDOM.createRoot(document.getElementById("root")).render(/*#__PURE__*/React.createElement(CryptoApp, null));
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/crypto/CryptoApp.jsx", error: String((e && e.message) || e) }); }

// ui_kits/crypto/CryptoData.jsx
try { (() => {
/* Mercury Cryptography — UI Kit · data layer.
   Every value traces to the AMA Cryptography README / data model:
   - defense-in-depth layers (4 core + 2 supporting)
   - NIST PQC set (ML-DSA-65 / ML-KEM-1024 / SLH-DSA · FIPS 204/203/205)
   - 3R runtime monitor (Resonance · Recursion · Refactoring · λ=0.25)
   - adaptive posture ladder
   - kin systems (AMA ⟷ Mercury Agent ⟷ FINDΩYOU)
   Latencies are the README's canonical-bench (Python-API-via-ctypes) figures. */

const PQC_ALGOS = [{
  id: "ml-dsa-65",
  name: "ML-DSA-65",
  role: "Digital signature",
  fips: "FIPS 204",
  sec: "192-bit quantum",
  lineage: "Dilithium lineage",
  detail: "NTT q=8380417 · rejection sampling · constant-time",
  perf: "~132µs verify · ~336µs sign"
}, {
  id: "ml-kem-1024",
  name: "ML-KEM-1024",
  role: "Key encapsulation",
  fips: "FIPS 203",
  sec: "IND-CCA2",
  lineage: "Kyber lineage",
  detail: "Fujisaki–Okamoto transform · native NTT",
  perf: "10,253 encaps/s"
}, {
  id: "slh-dsa",
  name: "SLH-DSA",
  role: "Stateless hash sig",
  fips: "FIPS 205",
  sec: "hash-based",
  lineage: "SPHINCS+ lineage",
  detail: "SHA2-256f / SHAKE-128s parameter sets",
  perf: "2 param sets"
}];

// classical + transport supporting primitives (cyan)
const CLASSICAL = [{
  name: "Ed25519",
  note: "classical signature · RFC 8032",
  perf: "21,338 verify/s"
}, {
  name: "X25519",
  note: "key exchange · RFC 7748",
  perf: "15,401 ops/s"
}, {
  name: "AES-256-GCM",
  note: "AEAD · NIST SP 800-38D",
  perf: "293,143 enc/s"
}, {
  name: "ChaCha20-Poly1305",
  note: "AEAD · RFC 8439",
  perf: "256,249 enc/s"
}];

// the 6-layer defense-in-depth stack (attacker peels outward → in)
const DEFENSE = [{
  n: "01",
  alg: "SHA3-256",
  prot: "Content integrity",
  sec: "128-bit",
  bit: "collision res.",
  hue: "var(--mercury-300)",
  kind: "core"
}, {
  n: "02",
  alg: "HMAC-SHA3-256",
  prot: "Keyed message authentication",
  sec: "AUTH",
  bit: "integrity",
  hue: "var(--mercury-400)",
  kind: "core"
}, {
  n: "03",
  alg: "Ed25519",
  prot: "Classical digital signature",
  sec: "128-bit",
  bit: "classical",
  hue: "var(--mercury-500)",
  kind: "core"
}, {
  n: "04",
  alg: "ML-DSA-65",
  prot: "Quantum-resistant signature",
  sec: "192-bit",
  bit: "quantum · FIPS 204",
  hue: "var(--cognitive)",
  kind: "core",
  pqc: true
}, {
  n: "05",
  alg: "HKDF-SHA3-256",
  prot: "Key derivation · independence",
  sec: "RFC 5869",
  bit: "",
  hue: "var(--fg-muted)",
  kind: "sup"
}, {
  n: "06",
  alg: "RFC 3161",
  prot: "Timestamp · temporal proof",
  sec: "TSA",
  bit: "optional",
  hue: "var(--benevolence)",
  kind: "sup"
}];
const R3 = [{
  id: "resonance",
  glyph: "∿",
  name: "Resonance",
  desc: "FFT spectral anomaly · frequency-domain",
  hue: "var(--mercury-400)"
}, {
  id: "recursion",
  glyph: "⟲",
  name: "Recursion",
  desc: "multi-scale hierarchical pattern",
  hue: "var(--mercury-400)"
}, {
  id: "refactoring",
  glyph: "⊞",
  name: "Refactoring",
  desc: "code-complexity metrics · for review",
  hue: "var(--cognitive-soft)"
}];
const POSTURE = [{
  id: "STANDARD",
  hue: "var(--normal)",
  rule: "Hybrid signing · software keys permitted · 3R advisory"
}, {
  id: "ELEVATED",
  hue: "var(--warning-bright)",
  rule: "Constant-time enforced · key rotation accelerated · 3R sensitivity raised"
}, {
  id: "HARDENED",
  hue: "var(--benevolence)",
  rule: "HSM required for master secrets · timestamping mandatory · hybrid mandatory"
}, {
  id: "LOCKDOWN",
  hue: "var(--anomaly)",
  rule: "Operations gated · keys frozen & rotated · manual review required"
}];
const KIN = [{
  sig: "♱",
  wm: "mercury",
  role: "NEURO-SYMBOLIC AI",
  name: "Mercury Agent",
  desc: "7-phase cognitive agent. Autonomous, explainable, ethically-bounded detection.",
  lic: "GPL-3.0 · v2.0.0",
  hue: "var(--mercury-500)"
}, {
  sig: "Ω",
  wm: "findoyou",
  role: "BIOMETRIC PLATFORM",
  name: "FINDΩYOU™",
  desc: "Survivor-first recovery of the lost & missing. Geo-consent, bias-audited.",
  lic: "Near-future · US-focused",
  hue: "var(--brand-gold)"
}, {
  sig: "♱",
  wm: "ama",
  role: "PQC SUBSTRATE",
  name: "AMA Cryptography",
  desc: "Hybrid Ed25519 + ML-DSA-65 PQC substrate. Standalone, NIST-standardized.",
  lic: "Apache-2.0 · v3.2.0",
  hue: "var(--cognitive)"
}];
const COMPLIANCE = [{
  std: "NIST FIPS 204",
  note: "ML-DSA module-lattice signatures",
  tag: "ML-DSA-65"
}, {
  std: "NIST FIPS 203",
  note: "ML-KEM module-lattice KEM",
  tag: "ML-KEM-1024"
}, {
  std: "NIST FIPS 205",
  note: "SLH-DSA stateless hash signatures",
  tag: "SLH-DSA"
}, {
  std: "RFC 5869",
  note: "HKDF key derivation",
  tag: "HKDF-SHA3"
}, {
  std: "RFC 9106",
  note: "Argon2id password hashing",
  tag: "Argon2id"
}, {
  std: "RFC 9591",
  note: "FROST threshold Ed25519",
  tag: "t-of-n"
}];

// --- operation stream model (faithful op types + grounded latencies) ---
const OP_KINDS = [{
  op: "SIGN",
  alg: "Hybrid Ed25519 + ML-DSA-65",
  lat: [330, 360],
  res: "SEALED",
  pqc: true
}, {
  op: "VERIFY",
  alg: "crypto_package v1 · 6 layers",
  lat: [195, 215],
  res: "VERIFIED"
}, {
  op: "ENCAPS",
  alg: "ML-KEM-1024",
  lat: [92, 104],
  res: "OK",
  pqc: true
}, {
  op: "DERIVE",
  alg: "HKDF-SHA3-256",
  lat: [10, 16],
  res: "BOUND"
}, {
  op: "VERIFY",
  alg: "ML-DSA-65 signature",
  lat: [128, 140],
  res: "VALID",
  pqc: true
}];
const DOMAINS = ["crisis-response", "conservation", "whistleblower", "grid-control", "patient-records", "wallet-tx", "field-sensor", "archival"];
let SEQ = 0;
function pad2(n) {
  return n < 10 ? "0" + n : "" + n;
}
function nowTs() {
  const d = new Date();
  return pad2(d.getHours()) + ":" + pad2(d.getMinutes()) + ":" + pad2(d.getSeconds()) + "." + String(d.getMilliseconds()).padStart(3, "0");
}
function rnd(a, b) {
  return a + Math.random() * (b - a);
}

/* generate one operation. anomalyBias raises the chance of a 3R flag / tamper. */
function genOp(anomalyBias) {
  const k = OP_KINDS[Math.floor(Math.random() * OP_KINDS.length)];
  const roll = Math.random();
  let state = "NOMINAL",
    result = k.res,
    failed = false,
    flagged = false;
  if (roll < 0.06 + (anomalyBias || 0)) {
    flagged = true;
    state = "3R ANOMALY";
  }
  // a rare tamper on a VERIFY → FAILED (shows all-or-nothing verdict)
  if (k.op === "VERIFY" && roll < 0.03 + (anomalyBias || 0) * 0.5) {
    failed = true;
    result = "FAILED";
    state = "TAMPER";
  }
  SEQ += 1;
  const lat = rnd(k.lat[0], k.lat[1]);
  return {
    id: "op-" + SEQ + "-" + Math.random().toString(36).slice(2, 7),
    seq: SEQ,
    ts: nowTs(),
    op: k.op,
    alg: k.alg,
    pqc: !!k.pqc,
    domain: DOMAINS[Math.floor(Math.random() * DOMAINS.length)],
    bytes: Math.floor(rnd(256, 9248)),
    lat: lat,
    latLabel: lat < 1000 ? lat.toFixed(0) + "µs" : (lat / 1000).toFixed(2) + "ms",
    state,
    result,
    failed,
    flagged
  };
}

/* the 6-layer breakdown for a package VERIFY/SIGN detail panel */
function packageLayers(failed) {
  // if failed, the post-quantum signature is the layer that trips (illustrative)
  return [{
    alg: "SHA3-256",
    ev: "digest " + Math.random().toString(16).slice(2, 6) + "…" + Math.random().toString(16).slice(2, 6),
    st: "MATCH",
    ok: true
  }, {
    alg: "HMAC-SHA3-256",
    ev: "keyed tag valid",
    st: "PASS",
    ok: true
  }, {
    alg: "Ed25519",
    ev: "classical sig · 64B",
    st: "VALID",
    ok: true
  }, {
    alg: "ML-DSA-65",
    ev: "quantum sig · FIPS 204",
    st: failed ? "MISMATCH" : "VALID",
    ok: !failed,
    pqc: true
  }, {
    alg: "HKDF-SHA3-256",
    ev: "ethical-pillar context",
    st: "BOUND",
    ok: true
  }, {
    alg: "RFC 3161",
    ev: "TSA token · 2026-05-16Z",
    st: "STAMPED",
    ok: true
  }];
}
Object.assign(window, {
  CRYPTO: {
    PQC_ALGOS,
    CLASSICAL,
    DEFENSE,
    R3,
    POSTURE,
    KIN,
    COMPLIANCE,
    BENCH_META: {
      ran: "2026-04-27",
      total: 17,
      passed: 17,
      failed: 0,
      host: "canonical · native C via ctypes"
    },
    BENCHMARKS: [{
      group: "Hashing & KDF",
      hue: "var(--mercury-300)",
      items: [{
        name: "SHA3-256",
        op: "hash 1 KB · FIPS 202",
        ops: 184112,
        base: 31000,
        tol: 35,
        pqc: false
      }, {
        name: "HMAC-SHA3-256",
        op: "keyed auth",
        ops: 115408,
        base: 19500,
        tol: 40,
        pqc: false
      }, {
        name: "HKDF-SHA3-256",
        op: "derive · 3 keys",
        ops: 81703,
        base: 12500,
        tol: 35,
        pqc: false
      }]
    }, {
      group: "Classical signatures · Ed25519",
      hue: "var(--mercury-400)",
      items: [{
        name: "Ed25519 keygen",
        op: "key pair",
        ops: 55716,
        base: 10560,
        tol: 35,
        pqc: false
      }, {
        name: "Ed25519 sign",
        op: "expanded key",
        ops: 51488,
        base: 10430,
        tol: 35,
        pqc: false
      }, {
        name: "Ed25519 verify",
        op: "signature check",
        ops: 21338,
        base: 5113,
        tol: 35,
        pqc: false
      }, {
        name: "X25519",
        op: "DH scalar-mult · MULX+ADX",
        ops: 15401,
        base: 13000,
        tol: 40,
        pqc: false
      }, {
        name: "X25519 batch-4",
        op: "DH · 4-way batch (batches/s)",
        ops: 4112,
        base: 2600,
        tol: 40,
        pqc: false
      }]
    }, {
      group: "Post-quantum · NIST",
      hue: "var(--cognitive)",
      items: [{
        name: "ML-DSA-65 keygen",
        op: "FIPS 204 · Dilithium",
        ops: 3626,
        base: 1943,
        tol: 40,
        pqc: true
      }, {
        name: "ML-DSA-65 sign",
        op: "FIPS 204",
        ops: 2976,
        base: 130,
        tol: 50,
        pqc: true
      }, {
        name: "ML-DSA-65 verify",
        op: "FIPS 204",
        ops: 7576,
        base: 900,
        tol: 40,
        pqc: true
      }, {
        name: "ML-KEM-1024 keygen",
        op: "FIPS 203 · Kyber",
        ops: 4965,
        base: 2200,
        tol: 40,
        pqc: true
      }, {
        name: "ML-KEM-1024 encaps",
        op: "FIPS 203",
        ops: 10253,
        base: 2400,
        tol: 40,
        pqc: true
      }]
    }, {
      group: "Transport AEAD",
      hue: "var(--mercury-500)",
      items: [{
        name: "AES-256-GCM",
        op: "encrypt 1 KB · SP 800-38D",
        ops: 293143,
        base: 150000,
        tol: 40,
        pqc: false
      }, {
        name: "ChaCha20-Poly1305",
        op: "encrypt 1 KB · RFC 8439",
        ops: 256249,
        base: 32000,
        tol: 40,
        pqc: false
      }]
    }, {
      group: "Full package · hybrid",
      hue: "var(--benevolence)",
      items: [{
        name: "Package create",
        op: "6-layer · with PQC",
        ops: 2853,
        base: 200,
        tol: 70,
        pqc: true
      }, {
        name: "Package verify",
        op: "6-layer · all-or-nothing",
        ops: 4973,
        base: 700,
        tol: 50,
        pqc: true
      }]
    }],
    /* NIST ACVP conformance — docs/compliance/acvp_attestation.json (2026-04-25,
       ref v1.1.0.42). 12 algorithms validated against upstream ACVP vectors;
       self-attestation, NOT a CAVP certificate. */
    ACVP_META: {
      ref: "v1.1.0.42",
      date: "2026-04-25",
      tested: 1215,
      passed: 1215,
      failed: 0,
      algos: 12,
      disclaimer: "Self-attestation against NIST ACVP vectors — NOT a CAVP certificate, NOT FIPS-140-3 validated, NOT NIST-endorsed."
    },
    ACVP: [{
      name: "SHA-256",
      std: "FIPS 180-4",
      v: 3,
      pqc: false
    }, {
      name: "HMAC-SHA-256",
      std: "FIPS 198-1",
      v: 150,
      pqc: false
    }, {
      name: "SHA3-256",
      std: "FIPS 202",
      v: 251,
      pqc: false
    }, {
      name: "SHA3-512",
      std: "FIPS 202",
      v: 186,
      pqc: false
    }, {
      name: "SHAKE-128",
      std: "FIPS 202",
      v: 274,
      pqc: false
    }, {
      name: "SHAKE-256",
      std: "FIPS 202",
      v: 243,
      pqc: false
    }, {
      name: "AES-256-GCM",
      std: "SP 800-38D",
      v: 4,
      pqc: false
    }, {
      name: "ML-KEM-1024 KeyGen",
      std: "FIPS 203",
      v: 25,
      pqc: true
    }, {
      name: "ML-KEM-1024 EncapDecap",
      std: "FIPS 203",
      v: 25,
      pqc: true
    }, {
      name: "ML-DSA-65 KeyGen",
      std: "FIPS 204",
      v: 25,
      pqc: true
    }, {
      name: "ML-DSA-65 SigVer",
      std: "FIPS 204",
      v: 15,
      pqc: true
    }, {
      name: "SLH-DSA-SHA2-256f SigVer",
      std: "FIPS 205",
      v: 14,
      pqc: true
    }],
    /* Constant-time empirical evidence — docs/constant-time-testing.md (dudect /
       Welch's t-test, |t|<4.5 ⇒ no detectable timing leak at 99.999%). This is
       the side-channel evidence layer the 3R monitor explicitly does NOT cover. */
    CONSTTIME_META: {
      method: "dudect · Welch t-test",
      gate: "|t| < 4.5",
      conf: "99.999%",
      measurements: "10M · CPU-pinned"
    },
    CONSTTIME: [{
      fn: "consttime_memcmp",
      grp: "Utility",
      res: "PASS",
      t: 1.8,
      note: "identical vs differing buffers"
    }, {
      fn: "consttime_swap",
      grp: "Utility",
      res: "PASS",
      t: 1.2,
      note: "condition 0 vs 1"
    }, {
      fn: "secure_memzero",
      grp: "Utility",
      res: "PASS",
      t: 2.1,
      note: "zero vs 0xFF buffer"
    }, {
      fn: "consttime_lookup",
      grp: "Utility",
      res: "PASS",
      t: 2.6,
      note: "table index leak"
    }, {
      fn: "consttime_copy",
      grp: "Utility",
      res: "PASS",
      t: 1.5,
      note: "conditional copy"
    }, {
      fn: "Ed25519 sign",
      grp: "Classical",
      res: "PASS",
      t: 3.1,
      note: "key-independent timing"
    }, {
      fn: "HKDF-SHA3-256",
      grp: "Classical",
      res: "PASS",
      t: 2.4,
      note: "IKM-independent"
    }, {
      fn: "HMAC-SHA256 verify",
      grp: "Classical",
      res: "PASS",
      t: 2.9,
      note: "MAC compare timing"
    }, {
      fn: "AES-GCM tag verify",
      grp: "Classical",
      res: "INFO",
      t: 5.7,
      note: "table S-box cache variation · use bitsliced backend"
    }, {
      fn: "ML-KEM-1024 decaps",
      grp: "Post-quantum",
      res: "PASS",
      t: 3.4,
      note: "implicit-rejection timing"
    }, {
      fn: "ML-DSA-65 sign",
      grp: "Post-quantum",
      res: "INFO",
      t: 6.2,
      note: "rejection sampling · intentional timing by design"
    }]
  },
  genOp,
  packageLayers,
  nowTs
});
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/crypto/CryptoData.jsx", error: String((e && e.message) || e) }); }

// ui_kits/crypto/CryptoPanels.jsx
try { (() => {
/* Mercury Cryptography — UI Kit · chrome + reusable panels.
   Lucide line icons (flagged substitution per the design-system iconography note). */

/* Nav icons render through the canonical <mercury-icon> atom (components/mercury-icon.js):
   ONE icon registry across the system, no per-kit inline-SVG generator. Maps each
   nav id → its wayfinding icon name; <mercury-icon> inherits the rail's currentColor. */
const CX_NAV_ICON = {
  overview: "grid",
  operations: "chart-line",
  algorithms: "layers",
  benchmarks: "bars",
  monitor: "pulse",
  package: "box",
  compliance: "check-doc",
  kin: "kin"
};
const CX_NAV = [["overview", "Overview"], ["operations", "Operations"], ["algorithms", "Algorithms"], ["benchmarks", "Benchmarks"], ["monitor", "3R Monitor"], ["package", "Package Inspector"], ["compliance", "Compliance"], ["kin", "Kin Systems"]];
const CX_TITLE = {
  overview: "System Overview",
  operations: "Cryptographic Operations",
  algorithms: "Post-Quantum Algorithms",
  benchmarks: "Performance Benchmarks",
  monitor: "3R Runtime Monitor",
  package: "Signed-Package Inspector",
  compliance: "Standards & Compliance",
  kin: "Kin Systems"
};

/* ---------- canonical wordmark (dark-ground rendering) ----------
   gold #B4B124 · blue #11AEED · black→grey on dark. Uses shared .wm-* classes. */
function Wordmark({
  which,
  sig
}) {
  let body;
  if (which === "mercury") body = /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("span", {
    className: "wm-g"
  }, "M"), /*#__PURE__*/React.createElement("span", {
    className: "wm-c"
  }, "ercury"), " ", /*#__PURE__*/React.createElement("span", {
    className: "wm-g"
  }, "A"), /*#__PURE__*/React.createElement("span", {
    className: "wm-k"
  }, "gent"));else if (which === "findoyou") body = /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("span", {
    className: "wm-c"
  }, "FIND"), /*#__PURE__*/React.createElement("span", {
    className: "wm-g"
  }, "\u03A9"), /*#__PURE__*/React.createElement("span", {
    className: "wm-k"
  }, "YOU"), /*#__PURE__*/React.createElement("span", {
    className: "wm-k",
    style: {
      fontSize: ".62em",
      verticalAlign: "super"
    }
  }, "\u2122"));else body = /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("span", {
    className: "wm-g"
  }, "A"), /*#__PURE__*/React.createElement("span", {
    className: "wm-k"
  }, "M"), /*#__PURE__*/React.createElement("span", {
    className: "wm-g"
  }, "A"), " ", /*#__PURE__*/React.createElement("span", {
    className: "wm-g"
  }, "C"), /*#__PURE__*/React.createElement("span", {
    className: "wm-c"
  }, "ryptography"));
  return /*#__PURE__*/React.createElement(React.Fragment, null, body, sig && /*#__PURE__*/React.createElement("span", {
    className: "wm-sig"
  }, " \u2671"));
}
function CxSidebar({
  active,
  setActive,
  posture
}) {
  return /*#__PURE__*/React.createElement("aside", {
    className: "mc-side cx-side"
  }, /*#__PURE__*/React.createElement("div", {
    className: "mc-side-brand"
  }, /*#__PURE__*/React.createElement("span", {
    className: "mc-side-sig"
  }, "\u2671"), /*#__PURE__*/React.createElement("div", {
    className: "mc-side-brand-t"
  }, /*#__PURE__*/React.createElement("span", {
    className: "mc-side-wm wm wm-dark"
  }, /*#__PURE__*/React.createElement(Wordmark, {
    which: "ama"
  })), /*#__PURE__*/React.createElement("span", {
    className: "mc-side-sub"
  }, "Cryptography \xB7 AMA"))), /*#__PURE__*/React.createElement("div", {
    className: "mc-side-scroll"
  }, /*#__PURE__*/React.createElement("nav", {
    className: "mc-nav"
  }, CX_NAV.map(([id, label]) => /*#__PURE__*/React.createElement("button", {
    key: id,
    className: "mc-navitem" + (active === id ? " on" : ""),
    onClick: () => setActive(id)
  }, /*#__PURE__*/React.createElement("mercury-icon", {
    name: CX_NAV_ICON[id],
    size: "17",
    stroke: "1.8"
  }), " ", /*#__PURE__*/React.createElement("span", null, label))))), /*#__PURE__*/React.createElement("div", {
    className: "mc-side-foot"
  }, /*#__PURE__*/React.createElement("div", {
    className: "mc-side-foot-h"
  }, "SECURES"), /*#__PURE__*/React.createElement("div", {
    className: "cx-secures wm wm-dark"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cx-sig-sm"
  }, "\u2671"), " ", /*#__PURE__*/React.createElement(Wordmark, {
    which: "mercury"
  })), /*#__PURE__*/React.createElement("div", {
    className: "cx-secures wm wm-dark"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cx-sig-sm",
    style: {
      color: "var(--brand-gold)"
    }
  }, "\u03A9"), " ", /*#__PURE__*/React.createElement(Wordmark, {
    which: "findoyou"
  })), /*#__PURE__*/React.createElement("div", {
    className: "mc-side-user"
  }, /*#__PURE__*/React.createElement("div", {
    className: "mc-side-avatar cog"
  }, "AE"), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: "mc-side-uname"
  }, "steel.sa.llc"), /*#__PURE__*/React.createElement("div", {
    className: "mc-side-urole"
  }, "Apache-2.0 \xB7 not externally audited")))));
}
function CxTopbar({
  live,
  onRun,
  page,
  posture
}) {
  const tier = window.CRYPTO.POSTURE.find(p => p.id === posture) || window.CRYPTO.POSTURE[0];
  return /*#__PURE__*/React.createElement("header", {
    className: "cx-top"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    className: "cx-crumb"
  }, "CRYPTOGRAPHY / ", /*#__PURE__*/React.createElement("span", null, CX_TITLE[page])), /*#__PURE__*/React.createElement("div", {
    className: "cx-h1"
  }, CX_TITLE[page])), /*#__PURE__*/React.createElement("div", {
    className: "cx-top-right"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cx-posture",
    style: {
      "--pc": tier.hue
    },
    title: tier.rule
  }, /*#__PURE__*/React.createElement("span", {
    className: "cx-posture-dot"
  }), "POSTURE \xB7 ", tier.id), /*#__PURE__*/React.createElement("div", {
    className: "cx-live" + (live ? " on" : "")
  }, /*#__PURE__*/React.createElement("span", {
    className: "cx-live-dot"
  }), live ? "OPERATING" : "IDLE"), /*#__PURE__*/React.createElement("button", {
    className: "mc-action primary",
    onClick: onRun,
    disabled: live
  }, "\u25B8 SIGN & STREAM")));
}

/* ---------- stat cards ---------- */
function CxStat({
  lab,
  val,
  sub,
  cls,
  lead
}) {
  return /*#__PURE__*/React.createElement("div", {
    className: "cx-stat" + (lead ? " lead" : "")
  }, /*#__PURE__*/React.createElement("div", {
    className: "cx-stat-lab"
  }, lab), /*#__PURE__*/React.createElement("div", {
    className: "cx-stat-val" + (cls ? " " + cls : "")
  }, val), /*#__PURE__*/React.createElement("div", {
    className: "cx-stat-sub"
  }, sub));
}

/* ---------- defense-in-depth stack (rows) ---------- */
function CxDefense() {
  const D = window.CRYPTO.DEFENSE;
  return /*#__PURE__*/React.createElement("div", {
    className: "cx-panel"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cx-panel-head"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cx-panel-lab"
  }, "Defense-in-Depth"), /*#__PURE__*/React.createElement("span", {
    className: "cx-panel-meta"
  }, "4 core \xB7 2 supporting")), /*#__PURE__*/React.createElement("div", {
    className: "cx-defense"
  }, D.map((l, i) => /*#__PURE__*/React.createElement(React.Fragment, {
    key: l.n
  }, l.kind === "sup" && D[i - 1].kind === "core" && /*#__PURE__*/React.createElement("div", {
    className: "cx-def-div"
  }, /*#__PURE__*/React.createElement("span", null, "SUPPORTING"), /*#__PURE__*/React.createElement("span", {
    className: "r"
  })), /*#__PURE__*/React.createElement("div", {
    className: "cx-def-row",
    style: {
      "--ly": l.hue
    }
  }, /*#__PURE__*/React.createElement("span", {
    className: "cx-def-n"
  }, l.n), /*#__PURE__*/React.createElement("span", {
    className: "cx-def-mid"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cx-def-alg"
  }, l.alg, l.pqc && /*#__PURE__*/React.createElement("span", {
    className: "cx-pqc"
  }, "PQC")), /*#__PURE__*/React.createElement("span", {
    className: "cx-def-prot"
  }, l.prot)), /*#__PURE__*/React.createElement("span", {
    className: "cx-def-sec",
    style: {
      color: l.hue
    }
  }, l.sec))))), /*#__PURE__*/React.createElement("div", {
    className: "cx-panel-foot"
  }, "Bounded by the weakest layer \u2014 ", /*#__PURE__*/React.createElement("b", null, "~128-bit classical, ~192-bit quantum"), "."));
}

/* ---------- 3R monitor ---------- */
function CxMonitor({
  alerts,
  posture
}) {
  const R = window.CRYPTO.R3;
  return /*#__PURE__*/React.createElement("div", {
    className: "cx-panel"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cx-panel-head"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cx-panel-lab"
  }, "3R Runtime Monitor"), /*#__PURE__*/React.createElement("span", {
    className: "cx-mon-live" + (alerts > 0 ? " alert" : "")
  }, alerts > 0 ? alerts + " ALERT" : "MONITORING")), /*#__PURE__*/React.createElement("div", {
    className: "cx-mon"
  }, R.map(e => /*#__PURE__*/React.createElement("div", {
    className: "cx-mon-row",
    key: e.id
  }, /*#__PURE__*/React.createElement("span", {
    className: "cx-mon-glyph",
    style: {
      color: e.hue
    }
  }, e.glyph), /*#__PURE__*/React.createElement("span", {
    className: "cx-mon-mid"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cx-mon-n"
  }, e.name), /*#__PURE__*/React.createElement("span", {
    className: "cx-mon-d"
  }, e.desc)), /*#__PURE__*/React.createElement("span", {
    className: "cx-mon-state" + (alerts > 0 && e.id === "resonance" ? " alert" : "")
  }, alerts > 0 && e.id === "resonance" ? "FLAGGED" : "NOMINAL")))), /*#__PURE__*/React.createElement("div", {
    className: "cx-mon-foot"
  }, /*#__PURE__*/React.createElement("span", null, "\u03BB ", /*#__PURE__*/React.createElement("b", null, "0.25")), /*#__PURE__*/React.createElement("span", null, "Cython ", /*#__PURE__*/React.createElement("b", null, "18\u201337\xD7")), /*#__PURE__*/React.createElement("span", {
    className: "note"
  }, "surfaces anomalies for review \u2014 not a timing-attack guarantee")));
}

/* ---------- adaptive posture ladder ---------- */
function CxPosture({
  current
}) {
  const P = window.CRYPTO.POSTURE;
  return /*#__PURE__*/React.createElement("div", {
    className: "cx-panel"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cx-panel-head"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cx-panel-lab"
  }, "Adaptive Posture"), /*#__PURE__*/React.createElement("span", {
    className: "cx-panel-meta"
  }, "3R-driven")), /*#__PURE__*/React.createElement("div", {
    className: "cx-posture-ladder"
  }, P.map(t => /*#__PURE__*/React.createElement("div", {
    key: t.id,
    className: "cx-tier" + (t.id === current ? " on" : ""),
    style: {
      "--tc": t.hue
    }
  }, /*#__PURE__*/React.createElement("span", {
    className: "cx-tier-dot"
  }), /*#__PURE__*/React.createElement("span", {
    className: "cx-tier-n"
  }, t.id, t.id === current && /*#__PURE__*/React.createElement("span", {
    className: "cx-tier-cur"
  }, "\u25C2 CURRENT")), /*#__PURE__*/React.createElement("span", {
    className: "cx-tier-d"
  }, t.rule)))));
}

/* ---------- operation log ---------- */
function CxOpLog({
  ops,
  selId,
  onSelect,
  live
}) {
  return /*#__PURE__*/React.createElement("div", {
    className: "cx-panel"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cx-panel-head"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cx-panel-lab"
  }, "Operation Stream"), /*#__PURE__*/React.createElement("span", {
    className: "cx-panel-meta"
  }, ops.length, " ops \xB7 hybrid sign / verify / encaps")), /*#__PURE__*/React.createElement("div", {
    className: "cx-log-head"
  }, /*#__PURE__*/React.createElement("span", null, "#"), /*#__PURE__*/React.createElement("span", null, "TIME"), /*#__PURE__*/React.createElement("span", null, "OP"), /*#__PURE__*/React.createElement("span", null, "ALGORITHM"), /*#__PURE__*/React.createElement("span", {
    className: "r"
  }, "LAT"), /*#__PURE__*/React.createElement("span", null, "3R"), /*#__PURE__*/React.createElement("span", {
    className: "r"
  }, "RESULT")), /*#__PURE__*/React.createElement("div", {
    className: "cx-log-body"
  }, ops.length === 0 && /*#__PURE__*/React.createElement("div", {
    className: "cx-log-empty"
  }, "No operations yet \u2014 press ", /*#__PURE__*/React.createElement("b", null, "SIGN & STREAM"), " to run the hybrid sign / verify pipeline."), ops.map(o => /*#__PURE__*/React.createElement("button", {
    key: o.id,
    className: "cx-log-row" + (o.failed ? " fail" : "") + (o.flagged ? " flag" : "") + (o.id === selId ? " sel" : ""),
    onClick: () => onSelect(o.id)
  }, /*#__PURE__*/React.createElement("span", {
    className: "cx-log-seq"
  }, String(o.seq).padStart(3, "0")), /*#__PURE__*/React.createElement("span", {
    className: "cx-log-ts"
  }, o.ts), /*#__PURE__*/React.createElement("span", {
    className: "cx-log-op op-" + o.op.toLowerCase()
  }, o.op), /*#__PURE__*/React.createElement("span", {
    className: "cx-log-alg"
  }, o.alg, o.pqc && /*#__PURE__*/React.createElement("span", {
    className: "cx-pqc sm"
  }, "PQC")), /*#__PURE__*/React.createElement("span", {
    className: "cx-log-lat r"
  }, o.latLabel), /*#__PURE__*/React.createElement("span", {
    className: "cx-log-3r" + (o.flagged ? " flag" : "") + (o.failed ? " fail" : "")
  }, o.state), /*#__PURE__*/React.createElement("span", {
    className: "cx-log-res r " + (o.failed ? "fail" : "ok")
  }, o.result)))));
}

/* ---------- package detail ---------- */
function CxPackageDetail({
  op,
  onInspect
}) {
  if (!op) return /*#__PURE__*/React.createElement("div", {
    className: "cx-panel"
  }, /*#__PURE__*/React.createElement("div", {
    className: "cx-panel-head"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cx-panel-lab"
  }, "Package \xB7 Detail")), /*#__PURE__*/React.createElement("div", {
    className: "cx-detail-empty"
  }, "Select an operation to inspect its ", /*#__PURE__*/React.createElement("b", null, "signed-package"), " layer breakdown, domain, and verdict."));
  const layers = window.packageLayers(op.failed);
  return /*#__PURE__*/React.createElement("div", {
    className: "cx-panel cx-detail" + (op.failed ? " is-fail" : "")
  }, /*#__PURE__*/React.createElement("div", {
    className: "cx-panel-head"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cx-panel-lab"
  }, "Package \xB7 Detail"), /*#__PURE__*/React.createElement("span", {
    className: "cx-panel-meta"
  }, "#", String(op.seq).padStart(3, "0"), " \xB7 ", op.op)), /*#__PURE__*/React.createElement("div", {
    className: "cx-verdict " + (op.failed ? "fail" : "ok")
  }, /*#__PURE__*/React.createElement("span", {
    className: "cx-verdict-dot"
  }), /*#__PURE__*/React.createElement("span", {
    className: "cx-verdict-txt"
  }, op.result), /*#__PURE__*/React.createElement("span", {
    className: "cx-verdict-sub"
  }, op.failed ? "layer mismatch" : op.op === "SIGN" ? "sealed" : "6/6 layers", " \xB7 ", op.latLabel)), /*#__PURE__*/React.createElement("div", {
    className: "cx-detail-loc"
  }, /*#__PURE__*/React.createElement("span", {
    className: "cx-loc-k"
  }, "DOMAIN"), /*#__PURE__*/React.createElement("span", null, op.domain), /*#__PURE__*/React.createElement("span", {
    className: "cx-loc-sep"
  }, "\xB7"), /*#__PURE__*/React.createElement("span", {
    className: "cx-loc-k"
  }, "BYTES"), /*#__PURE__*/React.createElement("span", null, op.bytes.toLocaleString()), /*#__PURE__*/React.createElement("span", {
    className: "cx-loc-sep"
  }, "\xB7"), /*#__PURE__*/React.createElement("span", {
    className: "cx-loc-k"
  }, "TS"), /*#__PURE__*/React.createElement("span", null, op.ts)), /*#__PURE__*/React.createElement("div", {
    className: "cx-detail-sec"
  }, "LAYER BREAKDOWN"), /*#__PURE__*/React.createElement("div", {
    className: "cx-detail-layers"
  }, layers.map(l => /*#__PURE__*/React.createElement("div", {
    className: "cx-dl",
    key: l.alg
  }, /*#__PURE__*/React.createElement("span", {
    className: "cx-dl-i" + (l.ok ? "" : " bad")
  }, l.ok ? "✓" : "✕"), /*#__PURE__*/React.createElement("span", {
    className: "cx-dl-a"
  }, l.alg, l.pqc && /*#__PURE__*/React.createElement("span", {
    className: "cx-pqc sm"
  }, "PQC")), /*#__PURE__*/React.createElement("span", {
    className: "cx-dl-e"
  }, l.ev), /*#__PURE__*/React.createElement("span", {
    className: "cx-dl-s" + (l.ok ? "" : " bad")
  }, l.st)))), op.failed && /*#__PURE__*/React.createElement("div", {
    className: "cx-fail-banner"
  }, "VERDICT FAILED \u2014 any single layer mismatch fails the whole package. Operation rejected; no advisory mode."), onInspect && /*#__PURE__*/React.createElement("button", {
    className: "mc-action cx-detail-inspect",
    onClick: onInspect
  }, "\u25B8 Inspect in Package inspector"));
}
Object.assign(window, {
  CX_NAV,
  CX_TITLE,
  Wordmark,
  CxSidebar,
  CxTopbar,
  CxStat,
  CxDefense,
  CxMonitor,
  CxPosture,
  CxOpLog,
  CxPackageDetail
});
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/crypto/CryptoPanels.jsx", error: String((e && e.message) || e) }); }

// ui_kits/deck/deck-stage.js
try { (() => {
// @ds-adherence-ignore -- omelette starter scaffold (raw elements/hex/px by design)
/* BEGIN USAGE */
/**
 * <deck-stage> — reusable web component for HTML decks.
 *
 * Handles:
 *  (a) speaker notes — reads <script type="application/json" id="speaker-notes">
 *      and posts {slideIndexChanged: N} to the parent window on nav.
 *  (b) keyboard navigation — ←/→, PgUp/PgDn, Space, Home/End, number keys.
 *      On touch devices, tapping the left/right half of the stage goes
 *      prev/next — taps on links, buttons and other interactive slide
 *      content are left alone.
 *  (c) press R to reset to slide 0 (with a tasteful keyboard hint).
 *  (d) bottom-center overlay showing slide count + hints, fades out on idle.
 *  (e) auto-scaling — inner canvas is a fixed design size (default 1920×1080)
 *      scaled with `transform: scale()` to fit the viewport, letterboxed.
 *      Set the `noscale` attribute to render at authored size (1:1) — the
 *      PPTX exporter sets this so its DOM capture sees unscaled geometry.
 *  (f) print — `@media print` lays every slide out as its own page at the
 *      design size, so the browser's Print → Save as PDF produces a clean
 *      one-page-per-slide PDF with no extra setup.
 *  (g) thumbnail rail — resizable left-hand column of per-slide thumbnails
 *      (static clones). Click to navigate; ↑/↓ with a thumbnail focused to
 *      step between slides; drag to reorder; right-click for
 *      Skip / Move up / Move down / Delete (opens a Cancel/Delete confirm
 *      dialog). Drag the rail's right edge to resize; width persists to
 *      localStorage. Skipped slides carry `data-deck-skip`, are dimmed in
 *      the rail, omitted from prev/next navigation, and hidden at print.
 *      The rail is suppressed in presenting mode, in the host's Preview
 *      mode (ViewerMode='none'), on `noscale`, on narrow viewports
 *      (≤640px), and via the `no-rail` attribute. Rail mutations dispatch
 *      a `deckchange`
 *      CustomEvent on the element: detail = {action, from, to, slide}.
 *
 * Slides are HIDDEN, not unmounted. Non-active slides stay in the DOM with
 * `visibility: hidden` + `opacity: 0`, so their state (videos, iframes,
 * form inputs, React trees) is preserved across navigation.
 *
 * Lifecycle event — the component dispatches a `slidechange` CustomEvent on
 * itself whenever the active slide changes (including the initial mount).
 * The event bubbles and composes out of shadow DOM, so you can listen on
 * the <deck-stage> element or on document:
 *
 *   document.querySelector('deck-stage').addEventListener('slidechange', (e) => {
 *     e.detail.index         // new 0-based index
 *     e.detail.previousIndex // previous index, or -1 on init
 *     e.detail.total         // total slide count
 *     e.detail.slide         // the new active slide element
 *     e.detail.previousSlide // the prior slide element, or null on init
 *     e.detail.reason        // 'init' | 'keyboard' | 'click' | 'tap' | 'api'
 *   });
 *
 * Persistence: none at the deck level. The host app keeps the current slide
 * in its own URL (?slide=) and re-delivers it via location.hash on load, so a
 * bare load with no hash always starts at slide 1.
 *
 * Usage:
 *   <style>deck-stage:not(:defined){visibility:hidden}</style>
 *   <deck-stage width="1920" height="1080">
 *     <section data-label="Title">...</section>
 *     <section data-label="Agenda">...</section>
 *   </deck-stage>
 *   <script src="deck-stage.js"></script>
 *
 * The :not(:defined) rule prevents a flash of the first slide at its
 * authored styles before this script runs and attaches the shadow root.
 *
 * Slides are the direct element children of <deck-stage>. Each slide is
 * automatically tagged with:
 *   - data-screen-label="NN Label"   (1-indexed, for comment flow)
 *   - data-om-validate="no_overflowing_text,no_overlapping_text,slide_sized_text"
 *
 * Speaker notes stay in sync because the component posts {slideIndexChanged: N}
 * to the parent — just include the #speaker-notes script tag if asked for notes.
 *
 * Authoring guidance:
 *   - Write slide bodies as static HTML inside <deck-stage>, with sizing via
 *     CSS custom properties in a <style> block rather than JS constants.
 *     Static slide markup is what lets the user click a heading in edit mode
 *     and retype it directly; a slide rendered through <script type="text/babel">,
 *     React, or a loop over a JS array has to round-trip every tweak through a
 *     chat message instead. Reach for script-generated slides only when the
 *     content genuinely needs interactive behaviour static HTML can't express.
 *   - Do NOT set position/inset/width/height on the slide <section> elements —
 *     the component absolutely positions every slotted child for you.
 *   - Entrance animations: make the visible end-state the base style and
 *     animate *from* hidden, so print and reduced-motion show content.
 *     Gate the animation on [data-deck-active] and the motion query, e.g.
 *     `@media (prefers-reduced-motion:no-preference){ [data-deck-active] .x{animation:fade-in .5s both} }`.
 *     Avoid infinite decorative loops on slide content.
 */
/* END USAGE */

(() => {
  const DESIGN_W_DEFAULT = 1920;
  const DESIGN_H_DEFAULT = 1080;
  const OVERLAY_HIDE_MS = 1800;
  const VALIDATE_ATTR = 'no_overflowing_text,no_overlapping_text,slide_sized_text';
  const FINE_POINTER_MQ = matchMedia('(hover: hover) and (pointer: fine)');
  const NARROW_MQ = matchMedia('(max-width: 640px)');
  // Slide-authored controls that should keep a tap instead of it navigating.
  const INTERACTIVE_SEL = 'a[href], button, input, select, textarea, summary, label, video[controls], audio[controls], [role="button"], [onclick], [tabindex]:not([tabindex^="-"]), [contenteditable]:not([contenteditable="false" i])';
  const pad2 = n => String(n).padStart(2, '0');

  // Label precedence: data-label → data-screen-label (number stripped) → first heading → "Slide".
  const getSlideLabel = el => {
    const explicit = el.getAttribute('data-label');
    if (explicit) return explicit;
    const existing = el.getAttribute('data-screen-label');
    if (existing) return existing.replace(/^\s*\d+\s*/, '').trim() || existing;
    const h = el.querySelector('h1, h2, h3, [data-title]');
    const t = h && (h.textContent || '').trim().slice(0, 40);
    if (t) return t;
    return 'Slide';
  };
  const stylesheet = `
    :host {
      position: fixed;
      inset: 0;
      display: block;
      background: #000;
      color: #fff;
      font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", Helvetica, Arial, sans-serif;
      overflow: hidden;
      -webkit-tap-highlight-color: transparent;
    }
    /* connectedCallback holds this until document.fonts.ready (capped 2s) so
     * the first visible paint has the deck's real typography + final rail
     * layout. opacity (not visibility) so the active slide can't un-hide
     * itself via the ::slotted([data-deck-active]) visibility:visible rule.
     * Only the stage/rail hide — the black :host background stays, so the
     * iframe doesn't flash the page's default white. */
    :host([data-fonts-pending]) .stage,
    :host([data-fonts-pending]) .rail { opacity: 0; pointer-events: none; }

    .stage {
      position: absolute;
      inset: 0;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .canvas {
      position: relative;
      transform-origin: center center;
      flex-shrink: 0;
      background: #fff;
      will-change: transform;
    }

    /* Slides live in light DOM (via <slot>) so authored CSS still applies.
       We absolutely position each slotted child to stack them. */
    ::slotted(*) {
      position: absolute !important;
      inset: 0 !important;
      width: 100% !important;
      height: 100% !important;
      box-sizing: border-box !important;
      overflow: hidden;
      opacity: 0;
      pointer-events: none;
      visibility: hidden;
    }
    ::slotted([data-deck-active]) {
      opacity: 1;
      pointer-events: auto;
      visibility: visible;
    }

    .overlay {
      position: fixed;
      left: 50%;
      bottom: 22px;
      transform: translate(-50%, 6px) scale(0.92);
      filter: blur(6px);
      display: flex;
      align-items: center;
      gap: 4px;
      padding: 4px;
      background: #000;
      color: #fff;
      border-radius: 999px;
      font-size: 12px;
      font-feature-settings: "tnum" 1;
      letter-spacing: 0.01em;
      opacity: 0;
      pointer-events: none;
      transition: opacity 260ms ease, transform 260ms cubic-bezier(.2,.8,.2,1), filter 260ms ease;
      transform-origin: center bottom;
      z-index: 2147483000;
      user-select: none;
    }
    .overlay[data-visible] {
      opacity: 1;
      pointer-events: auto;
      transform: translate(-50%, 0) scale(1);
      filter: blur(0);
    }

    .btn {
      appearance: none;
      -webkit-appearance: none;
      background: transparent;
      border: 0;
      margin: 0;
      padding: 0;
      color: inherit;
      font: inherit;
      cursor: default;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      height: 28px;
      min-width: 28px;
      border-radius: 999px;
      color: rgba(255,255,255,0.72);
      transition: background 140ms ease, color 140ms ease;
      -webkit-tap-highlight-color: transparent;
    }
    .btn:hover { background: rgba(255,255,255,0.12); color: #fff; }
    .btn:active { background: rgba(255,255,255,0.18); }
    .btn:focus { outline: none; }
    .btn:focus-visible { outline: none; }
    .btn::-moz-focus-inner { border: 0; }
    .btn svg { width: 14px; height: 14px; display: block; }
    .btn.reset {
      font-size: 11px;
      font-weight: 500;
      letter-spacing: 0.02em;
      padding: 0 10px 0 12px;
      gap: 6px;
      color: rgba(255,255,255,0.72);
    }
    .btn.reset .kbd {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 16px;
      height: 16px;
      padding: 0 4px;
      font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
      font-size: 10px;
      line-height: 1;
      color: rgba(255,255,255,0.88);
      background: rgba(255,255,255,0.12);
      border-radius: 4px;
    }

    .count {
      font-variant-numeric: tabular-nums;
      color: #fff;
      font-weight: 500;
      padding: 0 8px;
      min-width: 42px;
      text-align: center;
      font-size: 12px;
    }
    .count .sep { color: rgba(255,255,255,0.45); margin: 0 3px; font-weight: 400; }
    .count .total { color: rgba(255,255,255,0.55); }

    .divider {
      width: 1px;
      height: 14px;
      background: rgba(255,255,255,0.18);
      margin: 0 2px;
    }

    /* ── Thumbnail rail ──────────────────────────────────────────────────
       Fixed column on the left; each thumbnail is a static deep-clone of
       the light-DOM slide scaled into a 16:9 (or design-aspect) frame. The
       stage re-fits around it (see _fit); hidden during present / noscale
       / print so capture geometry and fullscreen output are unchanged. */
    .rail {
      position: fixed;
      left: 0;
      top: 0;
      bottom: 0;
      width: var(--deck-rail-w, 188px);
      background: #141414;
      border-right: 1px solid rgba(255,255,255,0.08);
      overflow-y: auto;
      overflow-x: hidden;
      padding: 12px 10px;
      box-sizing: border-box;
      display: flex;
      flex-direction: column;
      gap: 12px;
      z-index: 2147482500;
      scrollbar-width: thin;
      scrollbar-color: rgba(255,255,255,0.18) transparent;
    }
    .rail::-webkit-scrollbar { width: 8px; }
    .rail::-webkit-scrollbar-track { background: transparent; margin: 2px; }
    .rail::-webkit-scrollbar-thumb {
      background: rgba(255,255,255,0.18);
      border-radius: 4px;
      border: 2px solid transparent;
      background-clip: content-box;
    }
    .rail::-webkit-scrollbar-thumb:hover {
      background: rgba(255,255,255,0.28);
      border: 2px solid transparent;
      background-clip: content-box;
    }
    :host([no-rail]) .rail,
    :host([noscale]) .rail { display: none; }
    .rail[data-presenting] { display: none; }
    @media (max-width: 640px) {
      .rail, .rail-resize { display: none; }
    }
    /* User-driven show/hide (the TweaksPanel toggle) slides instead of
       popping. Transitions are gated on :host([data-rail-anim]) — set only
       for the 200ms around the toggle — so window-resize and rail-width
       drag (which also call _fit) don't lag behind the cursor. */
    .rail[data-user-hidden] { transform: translateX(-100%); }
    :host([data-rail-anim]) .rail { transition: transform 200ms cubic-bezier(.3,.7,.4,1); }
    :host([data-rail-anim]) .stage { transition: left 200ms cubic-bezier(.3,.7,.4,1); }
    :host([data-rail-anim]) .canvas { transition: transform 200ms cubic-bezier(.3,.7,.4,1); }
    /* transition shorthand replaces rather than merges — repeat the base
       .overlay opacity/transform/filter transitions so visibility changes
       during the 200ms toggle window still fade instead of popping. */
    :host([data-rail-anim]) .overlay {
      transition: margin-left 200ms cubic-bezier(.3,.7,.4,1),
                  opacity 260ms ease,
                  transform 260ms cubic-bezier(.2,.8,.2,1),
                  filter 260ms ease;
    }

    .thumb {
      position: relative;
      display: flex;
      align-items: flex-start;
      gap: 8px;
      cursor: pointer;
      user-select: none;
    }
    .thumb .num {
      width: 16px;
      flex-shrink: 0;
      font-size: 11px;
      font-weight: 500;
      text-align: right;
      color: rgba(255,255,255,0.55);
      padding-top: 2px;
      font-variant-numeric: tabular-nums;
    }
    .thumb .frame {
      position: relative;
      flex: 1;
      min-width: 0;
      aspect-ratio: var(--deck-aspect);
      background: #fff;
      border-radius: 4px;
      outline: 2px solid transparent;
      outline-offset: 0;
      overflow: hidden;
      transition: outline-color 120ms ease;
    }
    .thumb:hover .frame { outline-color: rgba(255,255,255,0.25); }
    .thumb { outline: none; }
    .thumb:focus-visible .frame { outline-color: rgba(255,255,255,0.5); }
    .thumb[data-current] .num { color: #fff; }
    .thumb[data-current] .frame { outline-color: #D97757; }
    .thumb[data-dragging] { opacity: 0.35; }
    .thumb::before {
      content: '';
      position: absolute;
      left: 24px;
      right: 0;
      height: 3px;
      border-radius: 2px;
      background: #D97757;
      opacity: 0;
      pointer-events: none;
    }
    .thumb[data-drop="before"]::before { top: -8px; opacity: 1; }
    .thumb[data-drop="after"]::before { bottom: -8px; opacity: 1; }
    .thumb[data-skip] .frame { opacity: 0.35; }
    .thumb[data-skip] .frame::after {
      content: 'Skipped';
      position: absolute;
      inset: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      background: rgba(0,0,0,0.45);
      color: #fff;
      font-size: 10px;
      font-weight: 500;
      letter-spacing: 0.04em;
    }

    .ctxmenu {
      position: fixed;
      min-width: 150px;
      padding: 4px;
      background: #242424;
      border: 1px solid rgba(255,255,255,0.12);
      border-radius: 7px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.45);
      z-index: 2147483100;
      display: none;
      font-size: 12px;
    }
    .ctxmenu[data-open] { display: block; }
    .ctxmenu button {
      display: block;
      width: 100%;
      appearance: none;
      border: 0;
      background: transparent;
      color: #e8e8e8;
      font: inherit;
      text-align: left;
      padding: 6px 10px;
      border-radius: 4px;
      cursor: pointer;
    }
    .ctxmenu button:hover:not(:disabled) { background: rgba(255,255,255,0.08); }
    .ctxmenu button:disabled { opacity: 0.35; cursor: default; }
    .ctxmenu hr {
      border: 0;
      border-top: 1px solid rgba(255,255,255,0.1);
      margin: 4px 2px;
    }

    .rail-resize {
      position: fixed;
      left: calc(var(--deck-rail-w, 188px) - 3px);
      top: 0;
      bottom: 0;
      width: 6px;
      cursor: col-resize;
      z-index: 2147482600;
      touch-action: none;
    }
    .rail-resize:hover,
    .rail-resize[data-dragging] { background: rgba(255,255,255,0.12); }
    :host([no-rail]) .rail-resize,
    :host([noscale]) .rail-resize,
    .rail[data-presenting] + .rail-resize,
    .rail[data-user-hidden] + .rail-resize { display: none; }

    /* Delete-confirm popup — matches the SPA's ConfirmDialog layout
       (title + message body, depressed footer with Cancel / Delete). */
    .confirm-backdrop {
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,0.45);
      z-index: 2147483200;
      display: none;
      align-items: center;
      justify-content: center;
    }
    .confirm-backdrop[data-open] { display: flex; }
    .confirm {
      width: 320px;
      max-width: calc(100vw - 32px);
      background: #2a2a2a;
      color: #e8e8e8;
      border: 1px solid rgba(255,255,255,0.12);
      border-radius: 12px;
      box-shadow: 0 12px 32px rgba(0,0,0,0.5);
      overflow: hidden;
      font-family: inherit;
      animation: deck-confirm-in 0.18s ease;
    }
    @keyframes deck-confirm-in {
      from { opacity: 0; transform: scale(0.96); }
      to { opacity: 1; transform: scale(1); }
    }
    .confirm .body { padding: 20px 20px 16px; }
    .confirm .title { font-size: 14px; font-weight: 600; margin-bottom: 4px; }
    .confirm .msg { font-size: 13px; line-height: 1.5; color: rgba(255,255,255,0.65); }
    .confirm .footer {
      padding: 14px 20px;
      background: #1f1f1f;
      border-top: 1px solid rgba(255,255,255,0.08);
      display: flex;
      justify-content: flex-end;
      gap: 8px;
    }
    .confirm button {
      appearance: none;
      font: inherit;
      font-size: 13px;
      font-weight: 500;
      padding: 8px 16px;
      border-radius: 8px;
      cursor: pointer;
    }
    .confirm .cancel {
      background: transparent;
      border: 0;
      color: rgba(255,255,255,0.8);
    }
    .confirm .cancel:hover { background: rgba(255,255,255,0.08); }
    .confirm .danger {
      background: #c96442;
      border: 1px solid rgba(0,0,0,0.15);
      color: #fff;
      box-shadow: 0 1px 3px rgba(166,50,68,0.3), 0 2px 6px rgba(166,50,68,0.18);
    }
    .confirm .danger:hover { background: #b5563a; }

    /* ── Print: one page per slide, no chrome ────────────────────────────
       The screen layout stacks every slide at inset:0 inside a scaled
       canvas; for print we want them in document flow at the authored
       design size so the browser paginates one slide per sheet. The
       @page size is set from the width/height attributes via the inline
       <style id="deck-stage-print-page"> that connectedCallback injects
       into <head> (the @page at-rule has no effect inside shadow DOM). */
    @media print {
      :host {
        position: static;
        inset: auto;
        background: none;
        overflow: visible;
        color: inherit;
      }
      .stage { position: static; display: block; }
      .canvas {
        transform: none !important;
        width: auto !important;
        height: auto !important;
        background: none;
        will-change: auto;
      }
      ::slotted(*) {
        position: relative !important;
        inset: auto !important;
        width: var(--deck-design-w) !important;
        height: var(--deck-design-h) !important;
        box-sizing: border-box !important;
        opacity: 1 !important;
        visibility: visible !important;
        pointer-events: auto;
        break-after: page;
        page-break-after: always;
        break-inside: avoid;
        overflow: hidden;
      }
      /* :last-child alone isn't enough once data-deck-skip hides the
         trailing slide(s) — the last *visible* slide still carries
         break-after:page and prints a blank sheet. _markLastVisible()
         maintains data-deck-last-visible on the last non-skipped slide. */
      ::slotted(*:last-child),
      ::slotted([data-deck-last-visible]) {
        break-after: auto;
        page-break-after: auto;
      }
      ::slotted([data-deck-skip]) { display: none !important; }
      .overlay, .rail, .rail-resize, .ctxmenu, .confirm-backdrop { display: none !important; }
    }
  `;
  class DeckStage extends HTMLElement {
    static get observedAttributes() {
      return ['width', 'height', 'noscale', 'no-rail'];
    }
    constructor() {
      super();
      this._root = this.attachShadow({
        mode: 'open'
      });
      this._index = 0;
      this._slides = [];
      this._notes = [];
      this._hideTimer = null;
      this._mouseIdleTimer = null;
      this._menuIndex = -1;
      this._onKey = this._onKey.bind(this);
      this._onResize = this._onResize.bind(this);
      this._onSlotChange = this._onSlotChange.bind(this);
      this._onMouseMove = this._onMouseMove.bind(this);
      this._onTap = this._onTap.bind(this);
      this._onMessage = this._onMessage.bind(this);
      // Capture-phase close so a click anywhere dismisses the menu, but
      // ignore clicks that land inside the menu itself — otherwise the
      // capture handler runs before the menu's own (bubble) handler and
      // clears _menuIndex out from under it.
      this._onDocClick = e => {
        if (this._menu && e.composedPath && e.composedPath().includes(this._menu)) return;
        this._closeMenu();
      };
    }
    get designWidth() {
      return parseInt(this.getAttribute('width'), 10) || DESIGN_W_DEFAULT;
    }
    get designHeight() {
      return parseInt(this.getAttribute('height'), 10) || DESIGN_H_DEFAULT;
    }
    connectedCallback() {
      // Presenter-view popup loads deckUrl?_snthumb=...#N for its prev/cur/
      // next thumbnails — the rail has no business rendering inside those
      // (wrong scale, and it offsets the stage so the thumb shows a gutter).
      if (/[?&]_snthumb=/.test(location.search)) this.setAttribute('no-rail', '');
      this._render();
      this._loadNotes();
      this._syncPrintPageRule();
      window.addEventListener('keydown', this._onKey);
      window.addEventListener('resize', this._onResize);
      window.addEventListener('mousemove', this._onMouseMove, {
        passive: true
      });
      window.addEventListener('message', this._onMessage);
      window.addEventListener('click', this._onDocClick, true);
      this.addEventListener('click', this._onTap);
      // Print lays every slide out as its own page, so [data-deck-active]-
      // gated entrance styles need the attribute on every slide (not just
      // the current one) or their content prints at the hidden base style.
      // The transient freeze style lands BEFORE the attributes so any
      // attribute-keyed transition fires at 0s (changing transition-
      // duration after a transition has started doesn't affect it).
      this._onBeforePrint = () => {
        if (this._freezeStyle) this._freezeStyle.remove();
        this._freezeStyle = document.createElement('style');
        this._freezeStyle.textContent = '*,*::before,*::after{transition-duration:0s !important}';
        document.head.appendChild(this._freezeStyle);
        this._slides.forEach(s => s.setAttribute('data-deck-active', ''));
      };
      this._onAfterPrint = () => {
        this._applyIndex({
          showOverlay: false,
          broadcast: false
        });
        if (this._freezeStyle) {
          this._freezeStyle.remove();
          this._freezeStyle = null;
        }
      };
      window.addEventListener('beforeprint', this._onBeforePrint);
      window.addEventListener('afterprint', this._onAfterPrint);
      // Initial collection + layout happens via slotchange, which fires on mount.
      this._enableRail();
      // Hold the stage hidden until webfonts are ready so the first visible
      // paint has the deck's real typography — the :not(:defined) guard in
      // the page HTML only covers custom-element upgrade, not font load.
      // Capped so a 404'd font URL can't blank the deck indefinitely.
      this.setAttribute('data-fonts-pending', '');
      const reveal = () => this.removeAttribute('data-fonts-pending');
      // rAF first: fonts.ready is a pre-resolved promise until layout has
      // resolved the slotted text's font-family and pushed a FontFace into
      // 'loading'. Reading it here in connectedCallback (parse-time) would
      // settle the race in a microtask before any font fetch starts.
      requestAnimationFrame(() => {
        Promise.race([document.fonts ? document.fonts.ready : Promise.resolve(), new Promise(r => setTimeout(r, 2000))]).then(reveal, reveal);
      });
    }
    _enableRail() {
      // Idempotent — older host builds still post __omelette_rail_enabled.
      // no-rail guard keeps the observers/stylesheet walk off the cheap path
      // for presenter-popup thumbnail iframes (up to 9 per view).
      if (this._railEnabled || this.hasAttribute('no-rail')) return;
      this._railEnabled = true;
      // Per-viewer preference — restored alongside rail width. Default on;
      // only a stored '0' (from the TweaksPanel toggle) hides it.
      this._railVisible = true;
      try {
        if (localStorage.getItem('deck-stage.railVisible') === '0') this._railVisible = false;
      } catch (e) {}
      // Live thumbnail updates: watch the light-DOM slides for content
      // edits and re-clone just the affected thumb(s), debounced. Ignore
      // the data-deck-* / data-screen-label / data-om-validate attributes
      // this component itself writes so nav and skip don't trigger
      // spurious refreshes.
      const OWN_ATTRS = /^data-(deck-|screen-label$|om-validate$)/;
      this._liveDirty = new Set();
      this._liveObserver = new MutationObserver(records => {
        for (const r of records) {
          if (r.type === 'attributes' && OWN_ATTRS.test(r.attributeName || '')) continue;
          let n = r.target;
          while (n && n.parentElement !== this) n = n.parentElement;
          if (n && this._slideSet && this._slideSet.has(n)) this._liveDirty.add(n);
        }
        if (this._liveDirty.size && !this._liveTimer) {
          this._liveTimer = setTimeout(() => {
            this._liveTimer = null;
            this._liveDirty.forEach(s => this._refreshThumb(s));
            this._liveDirty.clear();
          }, 200);
        }
      });
      this._liveObserver.observe(this, {
        subtree: true,
        childList: true,
        characterData: true,
        attributes: true
      });
      // Lazy thumbnail materialization — clone the slide only when its
      // frame scrolls into (or near) the rail viewport. rootMargin gives
      // ~4 thumbs of pre-load so fast scrolling doesn't flash blanks.
      this._railObserver = new IntersectionObserver(entries => {
        entries.forEach(e => {
          if (e.isIntersecting && e.target.__deckThumb) {
            this._materialize(e.target.__deckThumb);
          }
        });
      }, {
        root: this._rail,
        rootMargin: '400px 0px'
      });
      // Tweaks typically change CSS vars / attrs OUTSIDE <deck-stage>
      // (on <html>, <body>, a wrapper div, or a <style> tag), which
      // _liveObserver can't see. Re-snapshot author CSS (constructable
      // sheet is shared by reference, so one replaceSync updates every
      // thumb shadow root) and re-sync each thumb host's attrs + custom
      // properties. In-slide DOM mutations are _liveObserver's job.
      // Debounced so slider drags don't thrash.
      this._onTweakChange = () => {
        clearTimeout(this._tweakTimer);
        this._tweakTimer = setTimeout(() => {
          this._snapshotAuthorCss();
          // One getComputedStyle for the whole batch — each
          // getPropertyValue read below reuses the same computed style
          // as long as nothing invalidates layout between thumbs.
          const cs = getComputedStyle(this);
          (this._thumbs || []).forEach(t => {
            if (t.host) this._syncThumbHostAttrs(t.host, cs);
          });
        }, 120);
      };
      window.addEventListener('tweakchange', this._onTweakChange);
      this._snapshotAuthorCss();
      // Build the rail now that it's enabled — slotchange already fired,
      // so _renderRail's early-return skipped the initial build.
      this._syncRailHidden();
      this._renderRail();
      this._fit();
    }

    /** Snapshot document stylesheets into a constructable sheet that each
     *  thumbnail's nested shadow root adopts — so author CSS styles the
     *  cloned slide content without touching this component's chrome.
     *  Cross-origin sheets throw on .cssRules — skip them. Re-callable:
     *  the existing constructable sheet is reused via replaceSync so every
     *  already-adopted shadow root picks up the fresh CSS without re-adopt. */
    _snapshotAuthorCss() {
      // :root in an adopted sheet inside a shadow root matches nothing
      // (only the document root qualifies), so author rules like
      // `:root[data-voice="modern"] .serif` never reach the clones.
      // Rewrite :root → :host and mirror <html>'s data-*/class/lang onto
      // each thumb host (see _syncThumbHostAttrs) so the same selectors
      // match inside the thumbnail's shadow tree.
      const authorCss = Array.from(document.styleSheets).map(sh => {
        try {
          return Array.from(sh.cssRules).map(r => r.cssText).join('\n');
        } catch (e) {
          return '';
        }
      }).join('\n')
      // The shadow host is featureless outside the functional :host(...)
      // form, so any compound on :root — [attr], .class, #id, :pseudo —
      // must become :host(<compound>) not :host<compound>. Same for the
      // html type selector (Tailwind class-strategy dark mode emits
      // html.dark; Pico uses html[data-theme]), which has nothing to
      // match inside the thumb's shadow tree.
      .replace(/:root((?:\[[^\]]*\]|[.#][-\w]+|:[-\w]+(?:\([^)]*\))?)+)/g, ':host($1)').replace(/:root\b/g, ':host').replace(/(^|[\s,>~+(}])html((?:\[[^\]]*\]|[.#][-\w]+|:[-\w]+(?:\([^)]*\))?)+)(?![-\w])/g, '$1:host($2)').replace(/(^|[\s,>~+(}])html(?![-\w])/g, '$1:host');
      // Every custom property the author references. _syncThumbHostAttrs
      // mirrors each one's *computed* value at <deck-stage> onto the
      // thumb host so the live value wins over the :host default above
      // regardless of which ancestor the tweak wrote to (<html>, <body>,
      // a wrapper div, or the deck-stage element itself all inherit
      // down to getComputedStyle(this)).
      this._authorVars = new Set(authorCss.match(/--[\w-]+/g) || []);
      try {
        if (!this._adoptedSheet) this._adoptedSheet = new CSSStyleSheet();
        this._adoptedSheet.replaceSync(authorCss);
      } catch (e) {
        this._adoptedSheet = null;
        this._authorCss = authorCss;
      }
    }
    _syncThumbHostAttrs(host, cs) {
      const de = document.documentElement;
      // setAttribute overwrites but can't delete — an attr removed from
      // <html> (toggleAttribute off, classList emptied) would linger on
      // the host and :host([data-*]) / :host(.foo) rules would keep
      // matching. Remove stale mirrored attrs first; iterate backward
      // because removeAttribute mutates the live NamedNodeMap.
      for (let i = host.attributes.length - 1; i >= 0; i--) {
        const n = host.attributes[i].name;
        if ((n.startsWith('data-') || n === 'class' || n === 'lang') && !de.hasAttribute(n)) {
          host.removeAttribute(n);
        }
      }
      for (const a of de.attributes) {
        if (a.name.startsWith('data-') || a.name === 'class' || a.name === 'lang') {
          host.setAttribute(a.name, a.value);
        }
      }
      // The :root→:host rewrite in _snapshotAuthorCss pins each custom
      // property to its stylesheet default on the thumb host, shadowing
      // the live value that would otherwise inherit. Tweaks can write the
      // live value on any ancestor — <html>, <body>, a wrapper div, the
      // deck-stage element — so read it as the *computed* value at
      // <deck-stage> (which sees the whole inheritance chain) rather than
      // trying to guess which element the author wrote to. Inline on the
      // host beats the :host{} rule. remove-stale covers vars dropped
      // from the stylesheet between snapshots.
      const vars = this._authorVars || new Set();
      for (let i = host.style.length - 1; i >= 0; i--) {
        const p = host.style[i];
        if (p.startsWith('--') && !vars.has(p)) host.style.removeProperty(p);
      }
      const live = cs || getComputedStyle(this);
      vars.forEach(p => {
        const v = live.getPropertyValue(p);
        if (v) host.style.setProperty(p, v.trim());else host.style.removeProperty(p);
      });
    }
    disconnectedCallback() {
      window.removeEventListener('keydown', this._onKey);
      window.removeEventListener('resize', this._onResize);
      window.removeEventListener('mousemove', this._onMouseMove);
      window.removeEventListener('message', this._onMessage);
      window.removeEventListener('click', this._onDocClick, true);
      window.removeEventListener('beforeprint', this._onBeforePrint);
      window.removeEventListener('afterprint', this._onAfterPrint);
      if (this._freezeStyle) {
        this._freezeStyle.remove();
        this._freezeStyle = null;
      }
      this.removeEventListener('click', this._onTap);
      if (this._hideTimer) clearTimeout(this._hideTimer);
      if (this._mouseIdleTimer) clearTimeout(this._mouseIdleTimer);
      if (this._liveTimer) clearTimeout(this._liveTimer);
      if (this._tweakTimer) clearTimeout(this._tweakTimer);
      if (this._railAnimTimer) clearTimeout(this._railAnimTimer);
      if (this._scaleRaf) cancelAnimationFrame(this._scaleRaf);
      if (this._liveObserver) this._liveObserver.disconnect();
      if (this._railObserver) this._railObserver.disconnect();
      if (this._onTweakChange) window.removeEventListener('tweakchange', this._onTweakChange);
    }
    attributeChangedCallback() {
      if (this._canvas) {
        this._canvas.style.width = this.designWidth + 'px';
        this._canvas.style.height = this.designHeight + 'px';
        this._canvas.style.setProperty('--deck-design-w', this.designWidth + 'px');
        this._canvas.style.setProperty('--deck-design-h', this.designHeight + 'px');
        if (this._rail) {
          this._rail.style.setProperty('--deck-aspect', this.designWidth + '/' + this.designHeight);
        }
        this._fit();
        this._scaleThumbs();
        this._syncPrintPageRule();
      }
    }
    _render() {
      const style = document.createElement('style');
      style.textContent = stylesheet;
      const stage = document.createElement('div');
      stage.className = 'stage';
      const canvas = document.createElement('div');
      canvas.className = 'canvas';
      canvas.style.width = this.designWidth + 'px';
      canvas.style.height = this.designHeight + 'px';
      canvas.style.setProperty('--deck-design-w', this.designWidth + 'px');
      canvas.style.setProperty('--deck-design-h', this.designHeight + 'px');
      const slot = document.createElement('slot');
      slot.addEventListener('slotchange', this._onSlotChange);
      canvas.appendChild(slot);
      stage.appendChild(canvas);

      // Overlay: compact, solid black, with clickable controls.
      const overlay = document.createElement('div');
      overlay.className = 'overlay export-hidden';
      overlay.setAttribute('role', 'toolbar');
      overlay.setAttribute('aria-label', 'Deck controls');
      overlay.setAttribute('data-omelette-chrome', '');
      overlay.innerHTML = `
        <button class="btn prev" type="button" aria-label="Previous slide" title="Previous (←)">
          <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M10 3L5 8l5 5"/></svg>
        </button>
        <span class="count" aria-live="polite"><span class="current">1</span><span class="sep">/</span><span class="total">1</span></span>
        <button class="btn next" type="button" aria-label="Next slide" title="Next (→)">
          <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M6 3l5 5-5 5"/></svg>
        </button>
        <span class="divider"></span>
        <button class="btn reset" type="button" aria-label="Reset to first slide" title="Reset (R)">Reset<span class="kbd">R</span></button>
      `;
      overlay.querySelector('.prev').addEventListener('click', () => this._advance(-1, 'click'));
      overlay.querySelector('.next').addEventListener('click', () => this._advance(1, 'click'));
      overlay.querySelector('.reset').addEventListener('click', () => this._go(0, 'click'));

      // Thumbnail rail + context menu. Thumbnails are populated in
      // _renderRail() after _collectSlides().
      const rail = document.createElement('div');
      rail.className = 'rail export-hidden';
      rail.setAttribute('data-omelette-chrome', '');
      rail.style.setProperty('--deck-aspect', this.designWidth + '/' + this.designHeight);
      // Edge auto-scroll while dragging a thumb near the rail's top/bottom
      // so off-screen drop targets are reachable. Native dragover fires
      // continuously while the pointer is stationary, so a per-event nudge
      // (ramped by edge proximity) is enough — no rAF loop needed.
      rail.addEventListener('dragover', e => {
        if (this._dragFrom == null) return;
        const r = rail.getBoundingClientRect();
        const EDGE = 40;
        const dt = e.clientY - r.top;
        const db = r.bottom - e.clientY;
        if (dt < EDGE) rail.scrollTop -= Math.ceil((EDGE - dt) / 3);else if (db < EDGE) rail.scrollTop += Math.ceil((EDGE - db) / 3);
      });
      const menu = document.createElement('div');
      menu.className = 'ctxmenu export-hidden';
      menu.setAttribute('data-omelette-chrome', '');
      menu.innerHTML = `
        <button type="button" data-act="skip">Skip slide</button>
        <button type="button" data-act="up">Move up</button>
        <button type="button" data-act="down">Move down</button>
        <hr>
        <button type="button" data-act="delete">Delete slide</button>
      `;
      menu.addEventListener('click', e => {
        const act = e.target && e.target.getAttribute && e.target.getAttribute('data-act');
        if (!act) return;
        const i = this._menuIndex;
        this._closeMenu();
        if (act === 'skip') this._toggleSkip(i);else if (act === 'up') this._moveSlide(i, i - 1);else if (act === 'down') this._moveSlide(i, i + 1);else if (act === 'delete') this._openConfirm(i);
      });
      menu.addEventListener('contextmenu', e => e.preventDefault());

      // Rail resize handle — drag to set --deck-rail-w, persisted to
      // localStorage so the width survives reloads.
      const resize = document.createElement('div');
      resize.className = 'rail-resize export-hidden';
      resize.setAttribute('data-omelette-chrome', '');
      resize.addEventListener('pointerdown', e => {
        e.preventDefault();
        resize.setPointerCapture(e.pointerId);
        resize.setAttribute('data-dragging', '');
        const move = ev => this._setRailWidth(ev.clientX);
        const up = () => {
          resize.removeEventListener('pointermove', move);
          resize.removeEventListener('pointerup', up);
          resize.removeEventListener('pointercancel', up);
          resize.removeAttribute('data-dragging');
          try {
            localStorage.setItem('deck-stage.railWidth', String(this._railPx));
          } catch (err) {}
        };
        resize.addEventListener('pointermove', move);
        resize.addEventListener('pointerup', up);
        resize.addEventListener('pointercancel', up);
      });

      // Delete-confirm dialog — mirrors the SPA's ConfirmDialog layout.
      const confirm = document.createElement('div');
      confirm.className = 'confirm-backdrop export-hidden';
      confirm.setAttribute('data-omelette-chrome', '');
      confirm.innerHTML = `
        <div class="confirm" role="dialog" aria-modal="true">
          <div class="body">
            <div class="title">Delete slide?</div>
            <div class="msg">This slide will be removed from the deck.</div>
          </div>
          <div class="footer">
            <button type="button" class="cancel">Cancel</button>
            <button type="button" class="danger">Delete</button>
          </div>
        </div>
      `;
      confirm.addEventListener('click', e => {
        if (e.target === confirm) this._closeConfirm();
      });
      confirm.querySelector('.cancel').addEventListener('click', () => this._closeConfirm());
      confirm.querySelector('.danger').addEventListener('click', () => {
        const i = this._confirmIndex;
        this._closeConfirm();
        this._deleteSlide(i);
      });
      this._root.append(style, rail, resize, stage, overlay, menu, confirm);
      this._canvas = canvas;
      this._stage = stage;
      this._slot = slot;
      this._overlay = overlay;
      this._rail = rail;
      this._resize = resize;
      this._menu = menu;
      this._confirm = confirm;
      this._countEl = overlay.querySelector('.current');
      this._totalEl = overlay.querySelector('.total');

      // Restore persisted rail width.
      let rw = 188;
      try {
        const s = localStorage.getItem('deck-stage.railWidth');
        if (s) rw = parseInt(s, 10) || rw;
      } catch (err) {}
      this._setRailWidth(rw);
      this._syncRailHidden();
    }
    _setRailWidth(px) {
      const w = Math.max(120, Math.min(360, Math.round(px)));
      this._railPx = w;
      this.style.setProperty('--deck-rail-w', w + 'px');
      this._fit();
      // _scaleThumbs forces a sync layout (frame.offsetWidth) then writes
      // N transforms. During a resize drag this runs per-pointermove;
      // coalesce to one per frame.
      if (!this._scaleRaf) {
        this._scaleRaf = requestAnimationFrame(() => {
          this._scaleRaf = null;
          this._scaleThumbs();
        });
      }
    }

    /** @page must live in the document stylesheet — it's a no-op inside
     *  shadow DOM. Inject/update a single <head> style tag so the print
     *  sheet matches the design size and Save-as-PDF yields one slide per
     *  page with no margins. */
    _syncPrintPageRule() {
      const id = 'deck-stage-print-page';
      let tag = document.getElementById(id);
      if (!tag) {
        tag = document.createElement('style');
        tag.id = id;
        document.head.appendChild(tag);
      }
      tag.textContent = '@page { size: ' + this.designWidth + 'px ' + this.designHeight + 'px; margin: 0; } ' + '@media print { html, body { margin: 0 !important; padding: 0 !important; background: none !important; overflow: visible !important; height: auto !important; } ' + '* { -webkit-print-color-adjust: exact; print-color-adjust: exact; } ' +
      // Jump authored animations/transitions to their end state so print
      // never captures mid-entrance — pairs with the beforeprint handler
      // in connectedCallback that sets data-deck-active on every slide.
      '*, *::before, *::after { animation-delay: -99s !important; animation-duration: .001s !important; ' + 'animation-iteration-count: 1 !important; animation-fill-mode: both !important; ' + 'animation-play-state: running !important; transition-duration: 0s !important; } }';
    }
    _onSlotChange() {
      // Rail mutations (delete/move) already reconcile synchronously and
      // emit slidechange with reason 'api'; skip the async slotchange that
      // would otherwise re-broadcast with reason 'init'.
      if (this._squelchSlotChange) {
        this._squelchSlotChange = false;
        return;
      }
      this._collectSlides();
      this._restoreIndex();
      this._applyIndex({
        showOverlay: false,
        broadcast: true,
        reason: 'init'
      });
      this._fit();
    }
    _collectSlides() {
      const assigned = this._slot.assignedElements({
        flatten: true
      });
      this._slides = assigned.filter(el => {
        // Skip template/style/script nodes even if someone slots them.
        const tag = el.tagName;
        return tag !== 'TEMPLATE' && tag !== 'SCRIPT' && tag !== 'STYLE';
      });
      this._slideSet = new Set(this._slides);
      this._slides.forEach((slide, i) => {
        const n = i + 1;
        slide.setAttribute('data-screen-label', `${pad2(n)} ${getSlideLabel(slide)}`);

        // Validation attribute for comment flow / auto-checks.
        if (!slide.hasAttribute('data-om-validate')) {
          slide.setAttribute('data-om-validate', VALIDATE_ATTR);
        }
        slide.setAttribute('data-deck-slide', String(i));
      });
      if (this._totalEl) this._totalEl.textContent = String(this._slides.length || 1);
      if (this._index >= this._slides.length) this._index = Math.max(0, this._slides.length - 1);
      this._markLastVisible();
      this._renderRail();
    }

    /** Tag the last non-skipped slide so print CSS can drop its
     *  break-after (see the @media print comment above — :last-child
     *  alone matches a hidden skipped slide). */
    _markLastVisible() {
      let last = null;
      this._slides.forEach(s => {
        s.removeAttribute('data-deck-last-visible');
        if (!s.hasAttribute('data-deck-skip')) last = s;
      });
      if (last) last.setAttribute('data-deck-last-visible', '');
    }
    _loadNotes() {
      const tag = document.getElementById('speaker-notes');
      if (!tag) {
        this._notes = [];
        return;
      }
      try {
        const parsed = JSON.parse(tag.textContent || '[]');
        if (Array.isArray(parsed)) this._notes = parsed;
      } catch (e) {
        console.warn('[deck-stage] Failed to parse #speaker-notes JSON:', e);
        this._notes = [];
      }
    }
    _restoreIndex() {
      // The host's ?slide= param is delivered as a #<int> hash (1-indexed) on
      // the iframe src. No hash → slide 1; the deck itself keeps no position
      // state across loads.
      const h = (location.hash || '').match(/^#(\d+)$/);
      if (h) {
        const n = parseInt(h[1], 10) - 1;
        if (n >= 0 && n < this._slides.length) this._index = n;
      }
    }
    _applyIndex({
      showOverlay = true,
      broadcast = true,
      reason = 'init'
    } = {}) {
      if (!this._slides.length) return;
      const prev = this._prevIndex == null ? -1 : this._prevIndex;
      const curr = this._index;
      // Keep the iframe's own hash in sync so an in-iframe location.reload()
      // (reload banner path in viewer-handle.ts) lands on the current slide,
      // not the stale deep-link hash from initial load.
      try {
        history.replaceState(null, '', '#' + (curr + 1));
      } catch (e) {}
      this._slides.forEach((s, i) => {
        if (i === curr) s.setAttribute('data-deck-active', '');else s.removeAttribute('data-deck-active');
      });
      if (this._countEl) this._countEl.textContent = String(curr + 1);
      // Follow-scroll on every navigation (init deep-link, keyboard, click,
      // tap, external goTo) — the only time we *don't* want the rail to
      // track current is after a rail-internal mutation, where _renderRail
      // has already restored the user's scroll position and yanking back to
      // current would undo it.
      this._syncRail(reason !== 'mutation');
      if (broadcast) {
        // (1) Legacy: host-window postMessage for speaker-notes renderers.
        try {
          window.postMessage({
            slideIndexChanged: curr,
            deckTotal: this._slides.length,
            deckSkipped: this._skippedIndices()
          }, '*');
        } catch (e) {}

        // (2) In-page CustomEvent on the <deck-stage> element itself.
        //     Bubbles and composes out of shadow DOM so slide code can listen:
        //       document.querySelector('deck-stage').addEventListener('slidechange', e => {
        //         e.detail.index, e.detail.previousIndex, e.detail.total, e.detail.slide, e.detail.reason
        //       });
        const detail = {
          index: curr,
          previousIndex: prev,
          total: this._slides.length,
          slide: this._slides[curr] || null,
          previousSlide: prev >= 0 ? this._slides[prev] || null : null,
          reason: reason // 'init' | 'keyboard' | 'click' | 'tap' | 'api'
        };
        this.dispatchEvent(new CustomEvent('slidechange', {
          detail,
          bubbles: true,
          composed: true
        }));
      }
      this._prevIndex = curr;
      if (showOverlay) this._flashOverlay();
    }
    _flashOverlay() {
      // Host posts __omelette_presenting while in fullscreen/tab presentation
      // mode — suppress the nav footer entirely (both hover and slide-change
      // flash) so the audience sees clean slides.
      if (!this._overlay || this._presenting) return;
      this._overlay.setAttribute('data-visible', '');
      if (this._hideTimer) clearTimeout(this._hideTimer);
      this._hideTimer = setTimeout(() => {
        this._overlay.removeAttribute('data-visible');
      }, OVERLAY_HIDE_MS);
    }
    _railWidth() {
      // State-based, no offsetWidth: the first _fit() can run before the
      // rail has had layout on some load paths, and a 0 there paints the
      // slide full-width for one frame before the post-slotchange _fit()
      // corrects it.
      if (!this._railEnabled || !this._railVisible || this.hasAttribute('no-rail') || this.hasAttribute('noscale') || this._presenting || this._previewMode || NARROW_MQ.matches) return 0;
      return this._railPx || 0;
    }
    _fit() {
      if (!this._canvas) return;
      const stage = this._canvas.parentElement;
      // PPTX export sets noscale so the DOM capture sees authored-size
      // geometry — the scaled canvas is in shadow DOM, so the exporter's
      // resetTransformSelector can't reach .canvas.style.transform directly.
      if (this.hasAttribute('noscale')) {
        this._canvas.style.transform = 'none';
        if (stage) stage.style.left = '0';
        if (this._overlay) this._overlay.style.marginLeft = '0';
        return;
      }
      const rw = this._railWidth();
      if (stage) stage.style.left = rw + 'px';
      // Overlay is centred on the viewport via left:50% + translate(-50%);
      // marginLeft shifts the centre by rw/2 so it lands in the middle of
      // the [rw, innerWidth] stage region.
      if (this._overlay) this._overlay.style.marginLeft = rw / 2 + 'px';
      const vw = window.innerWidth - rw;
      const vh = window.innerHeight;
      const s = Math.min(vw / this.designWidth, vh / this.designHeight);
      this._canvas.style.transform = `scale(${s})`;
    }
    _onResize() {
      this._fit();
      // Crossing the narrow-viewport breakpoint reveals the rail — rerun the
      // thumbnail scale the same way _setRailWidth does.
      if (!this._scaleRaf) {
        this._scaleRaf = requestAnimationFrame(() => {
          this._scaleRaf = null;
          this._scaleThumbs();
        });
      }
    }
    _onMouseMove() {
      // Keep overlay visible while mouse moves; hide after idle.
      this._flashOverlay();
    }
    _onMessage(e) {
      const d = e.data;
      if (d && typeof d.__omelette_presenting === 'boolean') {
        this._presenting = d.__omelette_presenting;
        if (this._presenting && this._overlay) {
          this._overlay.removeAttribute('data-visible');
          if (this._hideTimer) clearTimeout(this._hideTimer);
        }
        this._syncRailHidden();
        this._closeMenu();
        this._closeConfirm();
        this._fit();
        this._scaleThumbs();
      }
      // Host's Preview segment (ViewerMode='none'): the rail's drag-reorder /
      // right-click skip-delete affordances are editing chrome, so hide it
      // while the user is just looking at the deck. Same hard-hide path as
      // presenting; independent of the user's _railVisible preference so
      // returning to Edit restores whatever they had.
      if (d && typeof d.__omelette_preview_mode === 'boolean') {
        if (d.__omelette_preview_mode === this._previewMode) return;
        this._previewMode = d.__omelette_preview_mode;
        this._syncRailHidden();
        this._closeMenu();
        this._closeConfirm();
        this._fit();
        this._scaleThumbs();
      }
      // Per-viewer show/hide, driven by the TweaksPanel's auto-injected
      // "Thumbnail rail" toggle (or any author script). Independent of
      // whether the Tweaks panel itself is open — closing the panel
      // doesn't change rail visibility. Persists alongside rail width.
      if (d && d.type === '__deck_rail_visible' && typeof d.on === 'boolean') {
        if (d.on === this._railVisible) return;
        this._railVisible = d.on;
        try {
          localStorage.setItem('deck-stage.railVisible', d.on ? '1' : '0');
        } catch (e) {}
        // Arm the transition, commit it, then flip state — otherwise the
        // browser coalesces both writes and nothing animates on show.
        this.setAttribute('data-rail-anim', '');
        void (this._rail && this._rail.offsetHeight);
        this._syncRailHidden();
        this._fit();
        this._scaleThumbs();
        clearTimeout(this._railAnimTimer);
        this._railAnimTimer = setTimeout(() => this.removeAttribute('data-rail-anim'), 220);
      }
      if (d && d.type === '__omelette_rail_enabled') this._enableRail();
    }
    _syncRailHidden() {
      if (!this._rail) return;
      // data-presenting is the hard hide (display:none) for flag-off,
      // presentation mode, and the host's Preview segment — instant, no
      // transition. data-user-hidden is the soft hide (translateX(-100%))
      // for the viewer's rail toggle, so show/hide slides under
      // :host([data-rail-anim]).
      const hard = !this._railEnabled || this._presenting || this._previewMode;
      if (hard) this._rail.setAttribute('data-presenting', '');else this._rail.removeAttribute('data-presenting');
      if (!this._railVisible) this._rail.setAttribute('data-user-hidden', '');else this._rail.removeAttribute('data-user-hidden');
      // translateX hide leaves thumbs (tabIndex=0) in the tab order —
      // inert keeps them unfocusable while the rail is off-screen.
      this._rail.inert = hard || !this._railVisible;
    }
    _onTap(e) {
      // Touch-only — keyboard + the overlay toolbar cover nav on desktop.
      if (FINE_POINTER_MQ.matches) return;
      // Only taps that land on the stage (slide content or letterbox); the
      // overlay / rail / menus are siblings with their own click handlers.
      const path = e.composedPath();
      if (!this._stage || !path.includes(this._stage)) return;
      // Let interactive slide content keep the tap. composedPath (not
      // e.target.closest) so we see through open shadow roots — a <button>
      // inside a slide-authored custom element retargets e.target to the
      // host but still appears in the composed path.
      if (e.defaultPrevented) return;
      for (const n of path) {
        if (n === this._stage) break;
        if (n.matches && n.matches(INTERACTIVE_SEL)) return;
      }
      e.preventDefault();
      const rw = this._railWidth();
      const mid = rw + (window.innerWidth - rw) / 2;
      this._advance(e.clientX < mid ? -1 : 1, 'tap');
    }
    _onKey(e) {
      // Ignore when the user is typing.
      const t = e.target;
      if (t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))) return;
      // Confirm dialog swallows nav keys while open; Escape cancels. Enter
      // is left to the focused button's native activation so Tab→Cancel
      // →Enter activates Cancel, not the window-level confirm path.
      if (this._confirm && this._confirm.hasAttribute('data-open')) {
        if (e.key === 'Escape') {
          this._closeConfirm();
          e.preventDefault();
        }
        return;
      }
      if (e.key === 'Escape' && this._menu && this._menu.hasAttribute('data-open')) {
        this._closeMenu();
        e.preventDefault();
        return;
      }
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const key = e.key;
      let handled = true;
      if (key === 'ArrowRight' || key === 'PageDown' || key === ' ' || key === 'Spacebar') {
        this._advance(1, 'keyboard');
      } else if (key === 'ArrowLeft' || key === 'PageUp') {
        this._advance(-1, 'keyboard');
      } else if (key === 'Home') {
        this._go(0, 'keyboard');
      } else if (key === 'End') {
        this._go(this._slides.length - 1, 'keyboard');
      } else if (key === 'r' || key === 'R') {
        this._go(0, 'keyboard');
      } else if (/^[0-9]$/.test(key)) {
        // 1..9 jump to that slide; 0 jumps to 10.
        const n = key === '0' ? 9 : parseInt(key, 10) - 1;
        if (n < this._slides.length) this._go(n, 'keyboard');
      } else {
        handled = false;
      }
      if (handled) {
        e.preventDefault();
        this._flashOverlay();
      }
    }
    _go(i, reason = 'api') {
      if (!this._slides.length) return;
      const clamped = Math.max(0, Math.min(this._slides.length - 1, i));
      if (clamped === this._index) {
        this._flashOverlay();
        return;
      }
      this._index = clamped;
      this._applyIndex({
        showOverlay: true,
        broadcast: true,
        reason
      });
    }

    /** Step forward/back skipping any slide marked data-deck-skip. Falls
     *  back to _go's clamp-at-ends behaviour (flash overlay) when there's
     *  nothing further in that direction. */
    _advance(dir, reason) {
      if (!this._slides.length) return;
      let i = this._index + dir;
      while (i >= 0 && i < this._slides.length && this._slides[i].hasAttribute('data-deck-skip')) {
        i += dir;
      }
      if (i < 0 || i >= this._slides.length) {
        this._flashOverlay();
        return;
      }
      this._go(i, reason);
    }

    // ── Thumbnail rail ────────────────────────────────────────────────────
    //
    // Thumbs are keyed by slide element and reused across _renderRail()
    // calls, so a reorder/delete is an O(changed) DOM shuffle instead of an
    // O(N) teardown-and-re-clone. Each thumb starts as a lightweight shell
    // (num + empty frame); the clone is materialized lazily by an
    // IntersectionObserver when the frame scrolls into (or near) view, so
    // only visible-ish slides pay the clone + image-decode cost.

    _renderRail() {
      if (!this._rail || !this._railEnabled) {
        this._thumbs = [];
        return;
      }
      // FLIP: record each *materialized* thumb's top before the reconcile.
      // Off-screen (non-materialized) thumbs don't need the animation and
      // skipping their getBoundingClientRect saves a forced layout per
      // off-screen thumb on large decks.
      const prevTops = new Map();
      (this._thumbs || []).forEach(({
        thumb,
        slide,
        host
      }) => {
        if (host) prevTops.set(slide, thumb.getBoundingClientRect().top);
      });
      const st = this._rail.scrollTop;

      // Reconcile: reuse thumbs that already exist for a slide, create
      // shells for new slides, drop thumbs for removed slides.
      const bySlide = new Map();
      (this._thumbs || []).forEach(t => bySlide.set(t.slide, t));
      const next = [];
      this._slides.forEach(slide => {
        let t = bySlide.get(slide);
        if (t) bySlide.delete(slide);else t = this._makeThumb(slide);
        next.push(t);
      });
      // Orphans — slides removed since last render.
      bySlide.forEach(t => {
        if (this._railObserver) this._railObserver.unobserve(t.frame);
        t.thumb.remove();
      });
      // Put thumbs into document order to match _slides. insertBefore on
      // an already-correctly-placed node is a no-op, so this is cheap
      // when nothing moved.
      next.forEach((t, i) => {
        const want = t.thumb;
        const at = this._rail.children[i];
        if (at !== want) this._rail.insertBefore(want, at || null);
        t.i = i;
        t.num.textContent = String(i + 1);
        if (t.slide.hasAttribute('data-deck-skip')) t.thumb.setAttribute('data-skip', '');else t.thumb.removeAttribute('data-skip');
      });
      this._thumbs = next;
      this._rail.scrollTop = st;
      if (prevTops.size) {
        const moved = [];
        this._thumbs.forEach(({
          thumb,
          slide
        }) => {
          const old = prevTops.get(slide);
          if (old == null) return;
          const dy = old - thumb.getBoundingClientRect().top;
          if (Math.abs(dy) < 1) return;
          thumb.style.transition = 'none';
          thumb.style.transform = `translateY(${dy}px)`;
          moved.push(thumb);
        });
        if (moved.length) {
          // Commit the inverted positions before flipping the transition
          // on — otherwise the browser coalesces both style writes and
          // nothing animates.
          void this._rail.offsetHeight;
          moved.forEach(t => {
            t.style.transition = 'transform 180ms cubic-bezier(.2,.7,.3,1)';
            t.style.transform = '';
          });
          setTimeout(() => moved.forEach(t => {
            t.style.transition = '';
          }), 220);
        }
      }
      requestAnimationFrame(() => this._scaleThumbs());
      this._syncRail(false);
    }

    /** Create a lightweight thumb shell for one slide. The clone is
     *  materialized later by the IntersectionObserver. Event handlers
     *  look up the thumb's *current* index (via _thumbs.indexOf) so the
     *  same element can be reused across reorders. */
    _makeThumb(slide) {
      const thumb = document.createElement('div');
      thumb.className = 'thumb';
      thumb.tabIndex = 0;
      const num = document.createElement('div');
      num.className = 'num';
      const frame = document.createElement('div');
      frame.className = 'frame';
      thumb.append(num, frame);
      const entry = {
        thumb,
        num,
        frame,
        slide,
        clone: null,
        host: null,
        i: -1
      };
      // entry.i is refreshed on every _renderRail reconcile pass, so
      // handlers read the thumb's current position without an O(N) scan.
      const idx = () => entry.i;
      thumb.addEventListener('click', () => this._go(idx(), 'click'));
      // ↑/↓ step through the rail when a thumb has focus. _go clamps at the
      // ends and _applyIndex→_syncRail scrolls the new current thumb into
      // view; we move focus to it (preventScroll — _syncRail already
      // scrolled) so a held key walks the whole list. stopPropagation keeps
      // this out of the window-level _onKey nav handler.
      thumb.addEventListener('keydown', e => {
        if (e.key !== 'ArrowUp' && e.key !== 'ArrowDown') return;
        if (e.metaKey || e.ctrlKey || e.altKey) return;
        e.preventDefault();
        e.stopPropagation();
        this._go(idx() + (e.key === 'ArrowDown' ? 1 : -1), 'keyboard');
        const cur = this._thumbs && this._thumbs[this._index];
        if (cur) cur.thumb.focus({
          preventScroll: true
        });
      });
      thumb.addEventListener('contextmenu', e => {
        e.preventDefault();
        this._openMenu(idx(), e.clientX, e.clientY);
      });
      thumb.draggable = true;
      thumb.addEventListener('dragstart', e => {
        this._dragFrom = idx();
        thumb.setAttribute('data-dragging', '');
        e.dataTransfer.effectAllowed = 'move';
        try {
          e.dataTransfer.setData('text/plain', String(this._dragFrom));
        } catch (err) {}
      });
      thumb.addEventListener('dragend', () => {
        thumb.removeAttribute('data-dragging');
        this._clearDrop();
        this._dragFrom = null;
      });
      thumb.addEventListener('dragover', e => {
        if (this._dragFrom == null) return;
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';
        const r = thumb.getBoundingClientRect();
        this._setDrop(idx(), e.clientY < r.top + r.height / 2 ? 'before' : 'after');
      });
      thumb.addEventListener('drop', e => {
        if (this._dragFrom == null) return;
        e.preventDefault();
        const i = idx();
        const r = thumb.getBoundingClientRect();
        let to = e.clientY >= r.top + r.height / 2 ? i + 1 : i;
        if (this._dragFrom < to) to--;
        const from = this._dragFrom;
        this._clearDrop();
        this._dragFrom = null;
        if (to !== from) this._moveSlide(from, to);
      });
      if (this._railObserver) this._railObserver.observe(frame);
      frame.__deckThumb = entry;
      return entry;
    }

    /** Lazily build the clone for a thumb that has scrolled into view. */
    _materialize(entry) {
      if (entry.host) return;
      const dw = this.designWidth,
        dh = this.designHeight;
      let clone = entry.slide.cloneNode(true);
      clone.removeAttribute('id');
      clone.removeAttribute('data-deck-active');
      clone.querySelectorAll('[id]').forEach(el => el.removeAttribute('id'));
      // Neuter heavy media; replace <video> with its poster so the box
      // keeps a visual. <iframe>/<audio> become empty placeholders.
      clone.querySelectorAll('iframe, audio, object, embed').forEach(el => {
        el.removeAttribute('src');
        el.removeAttribute('srcdoc');
        el.removeAttribute('data');
        el.innerHTML = '';
      });
      clone.querySelectorAll('video').forEach(el => {
        if (!el.poster) {
          el.removeAttribute('src');
          el.innerHTML = '';
          return;
        }
        const img = document.createElement('img');
        img.src = el.poster;
        img.alt = '';
        img.style.cssText = el.style.cssText + ';object-fit:cover;width:100%;height:100%;';
        img.className = el.className;
        el.replaceWith(img);
      });
      // Images: defer decode and let the browser pick the smallest
      // srcset candidate for the ~140px thumb. Same-URL clones reuse the
      // slide's decoded bitmap (URL-keyed cache), so the remaining cost
      // is paint/composite — lazy+async keeps that off the main thread.
      clone.querySelectorAll('img').forEach(el => {
        el.loading = 'lazy';
        el.decoding = 'async';
        if (el.srcset) el.sizes = (this._railPx || 188) + 'px';
      });
      // Custom elements inside the slide would have their
      // connectedCallback fire when the clone is appended. Replace them
      // with inert boxes so a component-heavy deck doesn't run N copies
      // of each component's mount logic in the rail. Children are
      // preserved so layout-wrapper elements (<my-column><h2>…</h2>)
      // still show their authored content; the querySelectorAll NodeList
      // is static, so nested custom elements in the moved subtree are
      // still visited on later iterations.
      const neuter = el => {
        const box = document.createElement('div');
        box.style.cssText = (el.getAttribute('style') || '') + ';background:rgba(0,0,0,0.06);border:1px dashed rgba(0,0,0,0.15);';
        box.className = el.className;
        // Preserve theming/i18n hooks so [data-*] / :lang() / [dir]
        // descendant selectors still match the neutered root.
        for (const a of el.attributes) {
          const n = a.name;
          if (n.startsWith('data-') || n.startsWith('aria-') || n === 'lang' || n === 'dir' || n === 'role' || n === 'title') {
            box.setAttribute(n, a.value);
          }
        }
        while (el.firstChild) box.appendChild(el.firstChild);
        return box;
      };
      // querySelectorAll('*') returns descendants only — a custom-element
      // slide root (<my-slide>…</my-slide>) would slip through and upgrade
      // on append. Swap the root first.
      if (clone.tagName.includes('-')) clone = neuter(clone);
      clone.querySelectorAll('*').forEach(el => {
        if (el.tagName.includes('-')) el.replaceWith(neuter(el));
      });
      clone.style.cssText += ';position:absolute;top:0;left:0;transform-origin:0 0;' + 'pointer-events:none;width:' + dw + 'px;height:' + dh + 'px;' + 'box-sizing:border-box;overflow:hidden;visibility:visible;opacity:1;';
      const host = document.createElement('div');
      host.style.cssText = 'position:absolute;inset:0;';
      this._syncThumbHostAttrs(host);
      const sr = host.attachShadow({
        mode: 'open'
      });
      if (this._adoptedSheet) sr.adoptedStyleSheets = [this._adoptedSheet];else {
        const st = document.createElement('style');
        st.textContent = this._authorCss || '';
        sr.appendChild(st);
      }
      sr.appendChild(clone);
      entry.frame.appendChild(host);
      entry.host = host;
      entry.clone = clone;
      if (this._thumbScale) clone.style.transform = 'scale(' + this._thumbScale + ')';
      // Once materialized the IO callback is a no-op early-return —
      // unobserve so scroll doesn't keep firing it.
      if (this._railObserver) this._railObserver.unobserve(entry.frame);
    }

    /** Re-clone a single thumb (live-update path). No-op if the thumb
     *  hasn't been materialized yet — it'll pick up current content when
     *  it scrolls into view. */
    _refreshThumb(slide) {
      const entry = (this._thumbs || []).find(t => t.slide === slide);
      if (!entry || !entry.host) return;
      entry.host.remove();
      entry.host = entry.clone = null;
      this._materialize(entry);
    }
    _scaleThumbs() {
      if (!this._thumbs || !this._thumbs.length) return;
      // Every frame is the same width; if it reads 0 the rail is
      // display:none (noscale / no-rail / presenting / print) — leave the
      // clones as-is and re-run when the rail is revealed.
      const fw = this._thumbs[0].frame.offsetWidth;
      if (!fw) return;
      this._thumbScale = fw / this.designWidth;
      this._thumbs.forEach(({
        clone
      }) => {
        if (clone) clone.style.transform = 'scale(' + this._thumbScale + ')';
      });
    }
    _setDrop(i, where) {
      // dragover fires at pointer-event rate; touch only the previous
      // and new target rather than sweeping all N thumbs.
      const t = this._thumbs && this._thumbs[i];
      if (this._dropOn && this._dropOn !== t) {
        this._dropOn.thumb.removeAttribute('data-drop');
      }
      if (t) t.thumb.setAttribute('data-drop', where);
      this._dropOn = t || null;
    }
    _clearDrop() {
      if (this._dropOn) this._dropOn.thumb.removeAttribute('data-drop');
      this._dropOn = null;
    }
    _syncRail(follow) {
      if (!this._thumbs) return;
      this._thumbs.forEach(({
        thumb
      }, i) => {
        if (i === this._index) {
          thumb.setAttribute('data-current', '');
          if (follow && typeof thumb.scrollIntoView === 'function') {
            thumb.scrollIntoView({
              block: 'nearest'
            });
          }
        } else {
          thumb.removeAttribute('data-current');
        }
      });
    }
    _openMenu(i, x, y) {
      if (!this._menu) return;
      this._menuIndex = i;
      const slide = this._slides[i];
      const skip = slide && slide.hasAttribute('data-deck-skip');
      this._menu.querySelector('[data-act="skip"]').textContent = skip ? 'Unskip slide' : 'Skip slide';
      this._menu.querySelector('[data-act="up"]').disabled = i <= 0;
      this._menu.querySelector('[data-act="down"]').disabled = i >= this._slides.length - 1;
      this._menu.querySelector('[data-act="delete"]').disabled = this._slides.length <= 1;
      // Place, then clamp to viewport after it's measurable.
      this._menu.style.left = x + 'px';
      this._menu.style.top = y + 'px';
      this._menu.setAttribute('data-open', '');
      const r = this._menu.getBoundingClientRect();
      const nx = Math.min(x, window.innerWidth - r.width - 4);
      const ny = Math.min(y, window.innerHeight - r.height - 4);
      this._menu.style.left = Math.max(4, nx) + 'px';
      this._menu.style.top = Math.max(4, ny) + 'px';
    }
    _closeMenu() {
      if (this._menu) this._menu.removeAttribute('data-open');
      this._menuIndex = -1;
    }
    _openConfirm(i) {
      if (!this._confirm) return;
      this._confirmIndex = i;
      this._confirm.querySelector('.title').textContent = 'Delete slide ' + (i + 1) + '?';
      this._confirm.setAttribute('data-open', '');
      const btn = this._confirm.querySelector('.danger');
      if (btn && btn.focus) btn.focus();
    }
    _closeConfirm() {
      if (this._confirm) this._confirm.removeAttribute('data-open');
      this._confirmIndex = -1;
    }
    _emitDeckChange(detail) {
      this.dispatchEvent(new CustomEvent('deckchange', {
        detail,
        bubbles: true,
        composed: true
      }));
    }
    _deleteSlide(i) {
      const slide = this._slides[i];
      if (!slide || this._slides.length <= 1) return;
      const wasCurrent = i === this._index;
      if (i < this._index || wasCurrent && i === this._slides.length - 1) this._index--;
      this._squelchSlotChange = true;
      slide.remove();
      this._emitDeckChange({
        action: 'delete',
        from: i,
        slide
      });
      this._collectSlides();
      this._applyIndex({
        showOverlay: true,
        broadcast: true,
        reason: 'mutation'
      });
    }
    _toggleSkip(i) {
      const slide = this._slides[i];
      if (!slide) return;
      const on = !slide.hasAttribute('data-deck-skip');
      if (on) slide.setAttribute('data-deck-skip', '');else slide.removeAttribute('data-deck-skip');
      if (this._thumbs && this._thumbs[i]) {
        if (on) this._thumbs[i].thumb.setAttribute('data-skip', '');else this._thumbs[i].thumb.removeAttribute('data-skip');
      }
      this._markLastVisible();
      this._emitDeckChange({
        action: on ? 'skip' : 'unskip',
        from: i,
        slide
      });
      // Re-broadcast so the presenter popup's prev/next thumbnails re-pick
      // the nearest non-skipped slide without waiting for a nav event.
      try {
        window.postMessage({
          slideIndexChanged: this._index,
          deckTotal: this._slides.length,
          deckSkipped: this._skippedIndices()
        }, '*');
      } catch (e) {}
    }
    _skippedIndices() {
      const out = [];
      for (let i = 0; i < this._slides.length; i++) {
        if (this._slides[i].hasAttribute('data-deck-skip')) out.push(i);
      }
      return out;
    }
    _moveSlide(i, j) {
      if (j < 0 || j >= this._slides.length || j === i) return;
      const slide = this._slides[i];
      const ref = j < i ? this._slides[j] : this._slides[j].nextSibling;
      // Track the active slide across the reorder so the same content
      // stays on screen.
      const cur = this._index;
      if (cur === i) this._index = j;else if (i < cur && j >= cur) this._index = cur - 1;else if (i > cur && j <= cur) this._index = cur + 1;
      this._squelchSlotChange = true;
      this.insertBefore(slide, ref);
      this._emitDeckChange({
        action: 'move',
        from: i,
        to: j,
        slide
      });
      this._collectSlides();
      this._applyIndex({
        showOverlay: false,
        broadcast: true,
        reason: 'mutation'
      });
    }

    // Public API ------------------------------------------------------------

    /** Current slide index (0-based). */
    get index() {
      return this._index;
    }
    /** Total slide count. */
    get length() {
      return this._slides.length;
    }
    /** Programmatically navigate. */
    goTo(i) {
      this._go(i, 'api');
    }
    next() {
      this._advance(1, 'api');
    }
    prev() {
      this._advance(-1, 'api');
    }
    reset() {
      this._go(0, 'api');
    }
  }
  if (!customElements.get('deck-stage')) {
    customElements.define('deck-stage', DeckStage);
  }
})();
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/deck/deck-stage.js", error: String((e && e.message) || e) }); }

// ui_kits/explorer/Explorer.jsx
try { (() => {
/* Detection & Audit Explorer — app. A faceted query instrument over the
   detection + audit_trail corpus: filter by state · severity · endpoint · sector
   · domain · ethics gate · TLP · time-window, with LIVE per-option counts, and
   every row recoverable to its raw payload, correlation ID and audit trail.
   Uses the shared <mercury-status>, <mercury-agent-dock> and
   <mercury-command-palette> components — proving the unified layer (review N1). */
const {
  useState,
  useMemo,
  useEffect,
  useRef
} = React;
const {
  ENDPOINTS,
  DOMAINS,
  FACETS,
  WINDOWS,
  buildCorpus
} = window.EXPLORER;
const CORPUS = buildCorpus(240);
const fmtScale = r => r.endpoint === "univariate" ? "σ" : r.endpoint === "multivariate" ? "L2" : "·";
function searchStr(r) {
  return (r.corr + " " + r.domain + " " + r.method + " " + r.endpoint + " " + r.action + " " + r.sevWord + " " + r.state + " " + r.tlp).toLowerCase();
}
function passes(r, filters, q, winMs, except) {
  if (winMs && r.ageMs > winMs) return false;
  if (q && searchStr(r).indexOf(q) < 0) return false;
  for (const f of FACETS) {
    if (f.id === except) continue;
    const sel = filters[f.id];
    if (sel && sel.length && sel.indexOf(f.get(r)) < 0) return false;
  }
  return true;
}
function Chrome({
  count,
  total,
  onPalette
}) {
  return /*#__PURE__*/React.createElement("header", {
    className: "mc-chrome"
  }, /*#__PURE__*/React.createElement("div", {
    className: "mc-brand"
  }, /*#__PURE__*/React.createElement("span", {
    className: "mc-sig"
  }, "\u2671"), /*#__PURE__*/React.createElement("div", {
    className: "mc-titles"
  }, /*#__PURE__*/React.createElement("div", {
    className: "mc-wm"
  }, /*#__PURE__*/React.createElement("span", {
    className: "g"
  }, "M"), /*#__PURE__*/React.createElement("span", {
    className: "c"
  }, "ERCURY"), " ", /*#__PURE__*/React.createElement("span", {
    className: "g"
  }, "A"), /*#__PURE__*/React.createElement("span", {
    className: "k"
  }, "GENT")), /*#__PURE__*/React.createElement("div", {
    className: "mc-sub"
  }, "Detection & Audit Explorer"))), /*#__PURE__*/React.createElement("div", {
    className: "mc-right"
  }, /*#__PURE__*/React.createElement("span", {
    className: "mc-stat"
  }, /*#__PURE__*/React.createElement("b", null, count.toLocaleString()), " ", /*#__PURE__*/React.createElement("span", {
    className: "of"
  }, "/ ", total.toLocaleString(), " records")), /*#__PURE__*/React.createElement("button", {
    className: "mc-kbtn",
    onClick: onPalette
  }, /*#__PURE__*/React.createElement("span", {
    className: "kb"
  }, "\u2318K"), " commands"), /*#__PURE__*/React.createElement("mercury-status", {
    state: "REFERENCE",
    tone: "warning",
    kind: "badge"
  })));
}
function QueryBar({
  q,
  setQ,
  filters,
  removeFilter,
  clearAll,
  winId,
  setWin,
  activeCount,
  searchRef
}) {
  const chips = [];
  FACETS.forEach(f => (filters[f.id] || []).forEach(v => chips.push({
    f,
    v,
    label: f.render ? f.render(v) : v
  })));
  return /*#__PURE__*/React.createElement("div", {
    className: "ex-qbar"
  }, /*#__PURE__*/React.createElement("div", {
    className: "ex-search"
  }, /*#__PURE__*/React.createElement("span", {
    className: "ex-search-i"
  }, "\u2315"), /*#__PURE__*/React.createElement("input", {
    ref: searchRef,
    value: q,
    onChange: e => setQ(e.target.value),
    placeholder: "Search correlation ID \xB7 domain \xB7 method \xB7 action\u2026",
    autoComplete: "off"
  }), q && /*#__PURE__*/React.createElement("span", {
    className: "ex-search-x",
    onClick: () => setQ("")
  }, "\u2715")), /*#__PURE__*/React.createElement("div", {
    className: "ex-win"
  }, WINDOWS.map(w => /*#__PURE__*/React.createElement("button", {
    key: w.id,
    className: "ex-win-b" + (winId === w.id ? " on" : ""),
    onClick: () => setWin(winId === w.id ? null : w.id)
  }, w.label))), /*#__PURE__*/React.createElement("div", {
    className: "ex-chips"
  }, chips.length === 0 && /*#__PURE__*/React.createElement("span", {
    className: "ex-chips-empty"
  }, "no filters \xB7 ", activeCount, " records in window"), chips.map((c, i) => /*#__PURE__*/React.createElement("span", {
    key: i,
    className: "ex-chip"
  }, /*#__PURE__*/React.createElement("span", {
    className: "ex-chip-k"
  }, c.f.label), c.label, /*#__PURE__*/React.createElement("span", {
    className: "ex-chip-x",
    onClick: () => removeFilter(c.f.id, c.v)
  }, "\u2715"))), chips.length > 0 && /*#__PURE__*/React.createElement("button", {
    className: "ex-clear",
    onClick: clearAll
  }, "clear all")));
}
function FacetRail({
  filters,
  toggle,
  counts,
  summary
}) {
  return /*#__PURE__*/React.createElement("aside", {
    className: "ex-rail"
  }, /*#__PURE__*/React.createElement("div", {
    className: "ex-rail-sum"
  }, /*#__PURE__*/React.createElement("div", {
    className: "ex-sum-row"
  }, /*#__PURE__*/React.createElement("span", null, "ANOMALY"), /*#__PURE__*/React.createElement("b", {
    className: "an"
  }, summary.anom)), /*#__PURE__*/React.createElement("div", {
    className: "ex-sum-row"
  }, /*#__PURE__*/React.createElement("span", null, "NORMAL"), /*#__PURE__*/React.createElement("b", {
    className: "no"
  }, summary.normal)), /*#__PURE__*/React.createElement("div", {
    className: "ex-sum-row"
  }, /*#__PURE__*/React.createElement("span", null, "WITHHELD"), /*#__PURE__*/React.createElement("b", {
    className: "wa"
  }, summary.withheld)), /*#__PURE__*/React.createElement("div", {
    className: "ex-sum-row"
  }, /*#__PURE__*/React.createElement("span", null, "anomaly rate"), /*#__PURE__*/React.createElement("b", null, summary.rate))), FACETS.map(f => /*#__PURE__*/React.createElement("div", {
    className: "ex-facet",
    key: f.id
  }, /*#__PURE__*/React.createElement("div", {
    className: "ex-facet-h"
  }, f.label), f.order.map(v => {
    const on = (filters[f.id] || []).indexOf(v) >= 0;
    const n = counts[f.id][v] || 0;
    const label = f.render ? f.render(v) : v;
    return /*#__PURE__*/React.createElement("button", {
      key: v,
      className: "ex-opt" + (on ? " on" : "") + (n === 0 && !on ? " zero" : ""),
      onClick: () => toggle(f.id, v),
      disabled: n === 0 && !on
    }, /*#__PURE__*/React.createElement("span", {
      className: "ex-box"
    }, on ? "▣" : "▢"), /*#__PURE__*/React.createElement("span", {
      className: "ex-opt-l"
    }, f.id === "tlp" || f.id === "state" || f.id === "decision" || f.id === "sevKey" ? renderTag(f.id, v, label) : label), /*#__PURE__*/React.createElement("span", {
      className: "ex-opt-n"
    }, n));
  }))));
}
function renderTag(facetId, v, label) {
  if (facetId === "tlp") return /*#__PURE__*/React.createElement("mercury-status", {
    state: v,
    kind: "tlp"
  });
  if (facetId === "sevKey") return /*#__PURE__*/React.createElement("mercury-status", {
    state: label,
    kind: "sev"
  });
  if (facetId === "state") return /*#__PURE__*/React.createElement("mercury-status", {
    state: v,
    kind: "dot",
    tone: v === "ANOMALY" ? "anomaly" : "normal"
  });
  if (facetId === "decision") return /*#__PURE__*/React.createElement("mercury-status", {
    state: v,
    kind: "dot",
    tone: v === "WITHHELD" ? "withheld" : "normal"
  });
  return label;
}
function ResultRow({
  r,
  open,
  onToggle
}) {
  return /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("tr", {
    className: "ex-row" + (open ? " open" : "") + (r.isAnom ? " anom" : ""),
    onClick: onToggle
  }, /*#__PURE__*/React.createElement("td", {
    className: "ex-td-ts"
  }, r.ts.slice(11), /*#__PURE__*/React.createElement("span", {
    className: "ex-td-date"
  }, r.ts.slice(5, 10))), /*#__PURE__*/React.createElement("td", null, /*#__PURE__*/React.createElement("span", {
    className: "ex-dom"
  }, r.domain), /*#__PURE__*/React.createElement("span", {
    className: "ex-sec"
  }, r.sector)), /*#__PURE__*/React.createElement("td", null, /*#__PURE__*/React.createElement("code", {
    className: "ex-ep"
  }, r.endpoint), /*#__PURE__*/React.createElement("span", {
    className: "ex-meth"
  }, r.method)), /*#__PURE__*/React.createElement("td", {
    className: "ex-td-score"
  }, /*#__PURE__*/React.createElement("b", null, r.score), /*#__PURE__*/React.createElement("span", {
    className: "ex-scl"
  }, r.scale)), /*#__PURE__*/React.createElement("td", null, /*#__PURE__*/React.createElement("mercury-status", {
    state: r.state,
    kind: "badge"
  })), /*#__PURE__*/React.createElement("td", null, r.isAnom ? /*#__PURE__*/React.createElement("mercury-status", {
    state: r.sevWord,
    kind: "sev"
  }) : /*#__PURE__*/React.createElement("span", {
    className: "ex-dash"
  }, "\u2014")), /*#__PURE__*/React.createElement("td", {
    className: "ex-td-gate"
  }, /*#__PURE__*/React.createElement("span", {
    className: "ex-eta" + (r.withheld ? " wh" : "")
  }, "\u03B7 ", r.benev.toFixed(3)), r.withheld && /*#__PURE__*/React.createElement("span", {
    className: "ex-wh-tag"
  }, "WITHHELD")), /*#__PURE__*/React.createElement("td", null, /*#__PURE__*/React.createElement("mercury-status", {
    state: r.tlp,
    kind: "tlp"
  })), /*#__PURE__*/React.createElement("td", {
    className: "ex-td-corr"
  }, /*#__PURE__*/React.createElement("code", null, r.corr))), open && /*#__PURE__*/React.createElement("tr", {
    className: "ex-detail-row"
  }, /*#__PURE__*/React.createElement("td", {
    colSpan: 9
  }, /*#__PURE__*/React.createElement("div", {
    className: "ex-detail"
  }, /*#__PURE__*/React.createElement("div", {
    className: "ex-detail-l"
  }, /*#__PURE__*/React.createElement("div", {
    className: "ex-dlab"
  }, "AUDIT TRAIL ", /*#__PURE__*/React.createElement("span", {
    className: "ex-dlab-m"
  }, "\xB7 ", r.corr)), /*#__PURE__*/React.createElement("div", {
    className: "ex-audit"
  }, r.audit.map((a, i) => /*#__PURE__*/React.createElement("div", {
    className: "ex-audit-row",
    key: i
  }, /*#__PURE__*/React.createElement("span", {
    className: "ex-br"
  }, "\u2514\u2500"), /*#__PURE__*/React.createElement("span", {
    className: "ex-au-t"
  }, a.t.slice(11)), /*#__PURE__*/React.createElement("mercury-status", {
    state: auditTone(a.event),
    tone: auditToneName(a.event),
    kind: "dot"
  }), /*#__PURE__*/React.createElement("span", {
    className: "ex-au-actor"
  }, a.actor), /*#__PURE__*/React.createElement("span", {
    className: "ex-au-detail"
  }, a.detail)))), /*#__PURE__*/React.createElement("div", {
    className: "ex-action"
  }, /*#__PURE__*/React.createElement("span", {
    className: "ex-dlab"
  }, "RECOMMENDED ACTION"), /*#__PURE__*/React.createElement("div", null, r.action))), /*#__PURE__*/React.createElement("div", {
    className: "ex-detail-r"
  }, /*#__PURE__*/React.createElement("div", {
    className: "ex-dlab"
  }, "RAW PAYLOAD ", /*#__PURE__*/React.createElement("span", {
    className: "ex-dlab-m"
  }, "\xB7 ", r.raw.endpoint)), /*#__PURE__*/React.createElement("pre", {
    className: "ex-raw"
  }, JSON.stringify(r.raw, null, 2)))))));
}
function auditTone(ev) {
  return ev;
}
function auditToneName(ev) {
  if (ev === "withheld") return "withheld";
  if (ev === "detected" || ev === "received") return "mercury";
  if (ev === "disclosed" || ev === "acknowledged") return "normal";
  if (ev === "gate-eval") return "cognitive";
  return "neutral";
}
const SORTS = [{
  id: "recent",
  label: "newest",
  cmp: (a, b) => a.ageMs - b.ageMs
}, {
  id: "score",
  label: "score ↓",
  cmp: (a, b) => b.score - a.score
}, {
  id: "margin",
  label: "gate margin ↑",
  cmp: (a, b) => a.gateMargin - b.gateMargin
}];
function App() {
  const [q, setQ] = useState("");
  const [filters, setFilters] = useState({});
  const [winId, setWin] = useState(null);
  const [openId, setOpenId] = useState(null);
  const [sort, setSort] = useState("recent");
  const palRef = useRef(null);
  const dockRef = useRef(null);
  const searchRef = useRef(null);
  const winMs = winId ? WINDOWS.find(w => w.id === winId).ms : 0;
  const ql = q.trim().toLowerCase();
  const filtered = useMemo(() => CORPUS.filter(r => passes(r, filters, ql, winMs, null)), [filters, ql, winMs]);
  const sorted = useMemo(() => filtered.slice().sort(SORTS.find(s => s.id === sort).cmp), [filtered, sort]);
  const counts = useMemo(() => {
    const out = {};
    FACETS.forEach(f => {
      out[f.id] = {};
      const base = CORPUS.filter(r => passes(r, filters, ql, winMs, f.id));
      f.order.forEach(v => out[f.id][v] = 0);
      base.forEach(r => {
        const v = f.get(r);
        if (out[f.id][v] != null) out[f.id][v]++;
      });
    });
    return out;
  }, [filters, ql, winMs]);
  const summary = useMemo(() => {
    const anom = filtered.filter(r => r.isAnom).length;
    const wh = filtered.filter(r => r.withheld).length;
    return {
      anom,
      normal: filtered.length - anom,
      withheld: wh,
      rate: filtered.length ? (anom / filtered.length * 100).toFixed(1) + "%" : "—"
    };
  }, [filtered]);
  const toggle = (fid, v) => setFilters(prev => {
    const cur = prev[fid] || [];
    const next = cur.indexOf(v) >= 0 ? cur.filter(x => x !== v) : cur.concat([v]);
    return Object.assign({}, prev, {
      [fid]: next
    });
  });
  const removeFilter = (fid, v) => toggle(fid, v);
  const clearAll = () => {
    setFilters({});
    setQ("");
    setWin(null);
  };

  // ---- wire shared components ----
  useEffect(() => {
    if (dockRef.current) dockRef.current.context = {
      title: "Detection & Audit Explorer",
      blurb: "A faceted view over the detection + <b>audit_trail</b> corpus. Filter on the left; every count updates against your other facets. Click a row for its raw payload and correlation ID.",
      status: {
        label: "online · faceted query",
        tone: "normal"
      },
      prompts: [{
        label: "What can I filter by?",
        answer: "State · severity · endpoint · sector · domain · ethics gate · TLP · time-window. Selections within a facet are <b>OR</b>; across facets, <b>AND</b>. Counts show what each option would add."
      }, {
        label: "Where's the data from?",
        answer: "The documented <b>detect()</b> schema, the TLP ladder (<b>tlp_handler.py</b>) and the <b>audit_trail</b>. Synthetic instances matched to documented distributions — disclaimed, never presented as live."
      }, {
        label: "Why no single 0–1 score?",
        answer: "Score scale is method-dependent: univariate is a <b>z-score</b>, multivariate an <b>L2</b> distance, fusion/3R/neurosym are <b>0–1</b>. The scale rides with every row so they're never confused."
      }],
      links: [{
        label: "Hazard Console",
        href: "../hazard/index.html"
      }, {
        label: "Console",
        href: "../console/index.html"
      }, {
        label: "CLI",
        href: "../cli/index.html"
      }, {
        label: "Cryptography",
        href: "../crypto/index.html"
      }]
    };
  }, []);
  useEffect(() => {
    if (!palRef.current) return;
    palRef.current.commands = [{
      id: "focus",
      group: "Query",
      label: "Focus search",
      hint: "filter the corpus",
      glyph: "⌕",
      kbd: "/",
      run: () => searchRef.current && searchRef.current.focus()
    }, {
      id: "anom",
      group: "Query",
      label: "Show anomalies only",
      glyph: "▲",
      run: () => setFilters(p => Object.assign({}, p, {
        state: ["ANOMALY"]
      }))
    }, {
      id: "withheld",
      group: "Query",
      label: "Show withheld decisions",
      hint: "η < 0.99",
      glyph: "η",
      run: () => setFilters(p => Object.assign({}, p, {
        decision: ["WITHHELD"]
      }))
    }, {
      id: "crit",
      group: "Query",
      label: "Show critical + high",
      glyph: "σ",
      run: () => setFilters(p => Object.assign({}, p, {
        sevKey: ["critical", "high"]
      }))
    }, {
      id: "clear",
      group: "Query",
      label: "Clear all filters",
      glyph: "○",
      run: clearAll
    }, {
      id: "s-recent",
      group: "Sort",
      label: "Sort by newest",
      glyph: "↓",
      run: () => setSort("recent")
    }, {
      id: "s-score",
      group: "Sort",
      label: "Sort by score",
      glyph: "↓",
      run: () => setSort("score")
    }, {
      id: "s-margin",
      group: "Sort",
      label: "Sort by gate margin",
      hint: "closest to the gate",
      glyph: "↑",
      run: () => setSort("margin")
    }, {
      id: "export",
      group: "Export",
      label: "Export current view",
      hint: "JSON · " + filtered.length + " rows",
      glyph: "▤",
      run: () => exportView(sorted)
    }, {
      id: "g-hazard",
      group: "Navigate",
      label: "Open Emergency Operations Console",
      kbd: "G H",
      glyph: "→",
      run: () => location.href = "../hazard/index.html"
    }, {
      id: "g-console",
      group: "Navigate",
      label: "Open Console",
      kbd: "G C",
      glyph: "→",
      run: () => location.href = "../console/index.html"
    }, {
      id: "g-cli",
      group: "Navigate",
      label: "Open CLI",
      kbd: "G L",
      glyph: "→",
      run: () => location.href = "../cli/index.html"
    }, {
      id: "g-crypto",
      group: "Navigate",
      label: "Open Cryptography",
      kbd: "G Y",
      glyph: "→",
      run: () => location.href = "../crypto/index.html"
    }];
  }, [filtered.length, sorted]);
  function exportView(rows) {
    const blob = new Blob([JSON.stringify(rows.map(r => r.raw), null, 2)], {
      type: "application/json"
    });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "mercury-explorer-view.json";
    a.click();
  }
  return /*#__PURE__*/React.createElement("div", {
    className: "ex-app"
  }, /*#__PURE__*/React.createElement(Chrome, {
    count: filtered.length,
    total: CORPUS.length,
    onPalette: () => palRef.current && palRef.current.open()
  }), /*#__PURE__*/React.createElement(QueryBar, {
    q: q,
    setQ: setQ,
    filters: filters,
    removeFilter: removeFilter,
    clearAll: clearAll,
    winId: winId,
    setWin: setWin,
    activeCount: filtered.length,
    searchRef: searchRef
  }), /*#__PURE__*/React.createElement("div", {
    className: "ex-main"
  }, /*#__PURE__*/React.createElement(FacetRail, {
    filters: filters,
    toggle: toggle,
    counts: counts,
    summary: summary
  }), /*#__PURE__*/React.createElement("section", {
    className: "ex-results"
  }, /*#__PURE__*/React.createElement("div", {
    className: "ex-results-h"
  }, /*#__PURE__*/React.createElement("span", {
    className: "ex-rh-count"
  }, /*#__PURE__*/React.createElement("b", null, filtered.length), " results"), /*#__PURE__*/React.createElement("div", {
    className: "ex-sort"
  }, /*#__PURE__*/React.createElement("span", {
    className: "ex-sort-l"
  }, "sort"), SORTS.map(s => /*#__PURE__*/React.createElement("button", {
    key: s.id,
    className: "ex-sort-b" + (sort === s.id ? " on" : ""),
    onClick: () => setSort(s.id)
  }, s.label)))), /*#__PURE__*/React.createElement("div", {
    className: "ex-table-wrap"
  }, /*#__PURE__*/React.createElement("table", {
    className: "ex-table"
  }, /*#__PURE__*/React.createElement("thead", null, /*#__PURE__*/React.createElement("tr", null, /*#__PURE__*/React.createElement("th", null, "TIME"), /*#__PURE__*/React.createElement("th", null, "DOMAIN"), /*#__PURE__*/React.createElement("th", null, "ENDPOINT"), /*#__PURE__*/React.createElement("th", null, "SCORE"), /*#__PURE__*/React.createElement("th", null, "STATE"), /*#__PURE__*/React.createElement("th", null, "SEVERITY"), /*#__PURE__*/React.createElement("th", null, "GATE \u03B7"), /*#__PURE__*/React.createElement("th", null, "TLP"), /*#__PURE__*/React.createElement("th", null, "CORRELATION"))), /*#__PURE__*/React.createElement("tbody", null, sorted.length === 0 && /*#__PURE__*/React.createElement("tr", null, /*#__PURE__*/React.createElement("td", {
    colSpan: 9,
    className: "ex-empty"
  }, "No records match these facets. Loosen a filter or clear all.")), sorted.map(r => /*#__PURE__*/React.createElement(ResultRow, {
    key: r.id,
    r: r,
    open: openId === r.id,
    onToggle: () => setOpenId(openId === r.id ? null : r.id)
  }))))))), /*#__PURE__*/React.createElement("footer", {
    className: "ex-foot"
  }, /*#__PURE__*/React.createElement("span", null, "Reference concept \xB7 built strictly from the documented data model \xB7 synthetic instances, not live data"), /*#__PURE__*/React.createElement("span", {
    className: "ex-foot-r wm on-dark"
  }, /*#__PURE__*/React.createElement("span", {
    className: "wm-g"
  }, "M"), /*#__PURE__*/React.createElement("span", {
    className: "wm-c"
  }, "ERCURY"), " ", /*#__PURE__*/React.createElement("span", {
    className: "wm-g"
  }, "A"), /*#__PURE__*/React.createElement("span", {
    className: "wm-k"
  }, "GENT"), " \u2671 \xB7 Steel Security Advisors LLC")), /*#__PURE__*/React.createElement("mercury-command-palette", {
    ref: palRef
  }), /*#__PURE__*/React.createElement("mercury-agent-dock", {
    ref: dockRef
  }));
}
ReactDOM.createRoot(document.getElementById("root")).render(/*#__PURE__*/React.createElement(App, null));
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/explorer/Explorer.jsx", error: String((e && e.message) || e) }); }

// ui_kits/explorer/ExplorerData.js
try { (() => {
/* ============================================================================
   Detection & Audit Explorer — DATA
   A faceted corpus over the SAME documented data model the Console uses (the
   detect() schema, the 3-component ensemble, domain AUCs, ethics gates, the
   TLP disclosure ladder, correlation IDs, and the audit_trail). Records are
   synthesized to MATCH documented distributions — no invented product metrics.
   Grounded in: src/omni_mercury_engine/api/ (routes/detection.py, batch.py,
   export.py), docs/API_REFERENCE.md, compliance/tlp_handler.py.
   ============================================================================ */
window.EXPLORER = function () {
  const ENDPOINTS = [{
    id: "univariate",
    scale: "σ · z-score",
    methods: ["zscore", "iqr", "isolation_forest"]
  }, {
    id: "multivariate",
    scale: "L2 / Mahalanobis",
    methods: ["mahalanobis"]
  }, {
    id: "fusion",
    scale: "fused ensemble",
    methods: ["weighted", "attention"]
  }, {
    id: "three-r",
    scale: "0–1 · η^Φ",
    methods: ["recursive"]
  }, {
    id: "neurosymbolic",
    scale: "0–1 · GOSNN",
    methods: ["gated", "hierarchical"]
  }];
  const DOMAINS = [{
    id: "security",
    label: "Network Security",
    sector: "Security",
    auc: 0.7983,
    ladder: "threat",
    baseRate: 0.18,
    eps: ["univariate", "multivariate", "fusion"]
  }, {
    id: "earthquake",
    label: "Earthquake",
    sector: "Space & Env",
    auc: 0.9367,
    ladder: "severity",
    baseRate: 0.10,
    eps: ["multivariate", "fusion"]
  }, {
    id: "tsunami",
    label: "Tsunami",
    sector: "Space & Env",
    auc: 0.8905,
    ladder: "severity",
    baseRate: 0.07,
    eps: ["multivariate", "fusion"]
  }, {
    id: "tornado",
    label: "Tornado",
    sector: "Space & Env",
    auc: 0.8803,
    ladder: "severity",
    baseRate: 0.12,
    eps: ["fusion", "three-r"]
  }, {
    id: "pandemic",
    label: "Pandemic",
    sector: "Medical",
    auc: 0.8588,
    ladder: "alert",
    baseRate: 0.14,
    eps: ["multivariate", "neurosymbolic"]
  }, {
    id: "energy",
    label: "Energy Grid",
    sector: "Infrastructure",
    auc: 0.8038,
    ladder: "severity",
    baseRate: 0.13,
    eps: ["univariate", "fusion"]
  }, {
    id: "hurricane",
    label: "Hurricane",
    sector: "Space & Env",
    auc: 0.7233,
    ladder: "severity",
    baseRate: 0.11,
    eps: ["fusion", "three-r"]
  }, {
    id: "flood",
    label: "Flood",
    sector: "Space & Env",
    auc: 0.6837,
    ladder: "severity",
    baseRate: 0.16,
    eps: ["multivariate", "fusion"]
  }, {
    id: "fema",
    label: "FEMA Disaster",
    sector: "Humanitarian",
    auc: 0.6573,
    ladder: "severity",
    baseRate: 0.15,
    eps: ["neurosymbolic", "fusion"]
  }];
  const LADDERS = {
    threat: {
      high: "HIGH",
      med: "MEDIUM",
      low: "LOW"
    },
    alert: {
      high: "CRITICAL",
      med: "WARNING",
      low: "NORMAL"
    },
    severity: {
      high: "SEVERE",
      med: "MODERATE",
      low: "NORMAL"
    }
  };
  const ACTIONS = {
    threat: {
      high: "Isolate host · rotate session keys (Kyber-1024)",
      med: "Rate-limit source · raise capture depth",
      low: "Log · continue monitoring"
    },
    alert: {
      high: "Escalate to clinician · open case",
      med: "Flag for review · re-sample",
      low: "Record baseline"
    },
    severity: {
      high: "Issue advisory · notify EOC",
      med: "Pre-stage resources · watch",
      low: "Archive observation"
    }
  };
  // severity → default TLP disclosure (analyst-overridable), per SEV_TLP_POLICY
  const SEV_TLP = {
    critical: "TLP:RED",
    high: "TLP:AMBER+STRICT",
    medium: "TLP:AMBER",
    low: "TLP:GREEN",
    nominal: "TLP:CLEAR"
  };
  function rnd(a, b) {
    return a + Math.random() * (b - a);
  }
  function r3(x) {
    return +x.toFixed(3);
  }
  function clamp01(x) {
    return x < 0 ? 0 : x > 1 ? 1 : x;
  }
  function pick(a) {
    return a[Math.floor(Math.random() * a.length)];
  }
  function hex(n) {
    var s = "";
    for (var i = 0; i < n; i++) s += "0123456789abcdef"[Math.floor(Math.random() * 16)];
    return s;
  }
  function gauss() {
    var u = 0,
      v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  function normInv(p) {
    // Acklam
    if (p <= 0) return -6;
    if (p >= 1) return 6;
    var a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.3577518672690, -30.66479806614716, 2.506628277459239];
    var b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
    var c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
    var d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
    var pl = 0.02425,
      q,
      r;
    if (p < pl) {
      q = Math.sqrt(-2 * Math.log(p));
      return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
    }
    if (p <= 1 - pl) {
      q = p - 0.5;
      r = q * q;
      return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
    }
    q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  function fmtTs(d) {
    var p = n => String(n).padStart(2, "0");
    return d.getFullYear() + "-" + p(d.getMonth() + 1) + "-" + p(d.getDate()) + " " + p(d.getHours()) + ":" + p(d.getMinutes()) + ":" + p(d.getSeconds());
  }

  // native score per endpoint scale (never coerced to one universal 0–1)
  function nativeScore(ep, sev01, isAnom) {
    if (ep === "univariate") return r3(isAnom ? rnd(2.0, 5.4) : rnd(0.2, 1.9)); // z-score
    if (ep === "multivariate") return r3(isAnom ? rnd(3.1, 12.4) : rnd(0.4, 2.8)); // L2 distance
    return r3(sev01); // fused / 3R / neurosym 0–1
  }
  function genRecord(seq, now) {
    const dom = pick(DOMAINS);
    const ep = pick(dom.eps);
    const epObj = ENDPOINTS.find(e => e.id === ep);
    const method = pick(epObj.methods);
    const isAnom = Math.random() < dom.baseRate * (ep === "fusion" ? 1.1 : 1);

    // signal-detection model: AUC sets separability so strong domains separate cleanly
    const sigma = 0.135,
      muN = 0.27;
    const dprime = Math.SQRT2 * normInv(dom.auc);
    const sev01 = clamp01((isAnom ? muN + dprime * sigma : muN) + gauss() * sigma);
    const lad = LADDERS[dom.ladder];
    const sevKey = !isAnom ? "low" : sev01 > 0.78 ? Math.random() < 0.25 ? "critical" : "high" : sev01 > 0.45 ? "medium" : "low";
    const sevWord = !isAnom ? "NORMAL" : sevKey === "critical" ? "CRITICAL" : sevKey === "high" ? lad.high : sevKey === "medium" ? lad.med : lad.low;
    const conf = isAnom ? Math.min(0.99, 0.7 + sev01 * 0.3) : 0.8 + (1 - sev01) * 0.15;

    // ethics gate: ~12% withheld (η < 0.99)
    const withheld = isAnom && Math.random() < 0.14;
    const benev = withheld ? r3(rnd(0.962, 0.9898)) : r3(rnd(0.990, 0.999));
    const gateMargin = r3(benev - 0.99);
    const tlp = withheld ? "TLP:RED" : SEV_TLP[isAnom ? sevKey : "nominal"];
    const score = nativeScore(ep, sev01, isAnom);
    const corr = "mca_" + hex(8) + "-" + hex(4);
    const ageMs = Math.floor(rnd(0, 24 * 3600 * 1000));
    const d = new Date(now - ageMs);
    const ackKey = sevKey === "critical" || sevKey === "high";
    const action = ACTIONS[dom.ladder][ackKey ? "high" : sevKey === "medium" ? "med" : "low"];

    // a documented-shape raw payload (what the route would return)
    const raw = {
      endpoint: "POST /api/v1/detect/" + ep,
      method: method,
      is_anomaly: isAnom,
      score: score,
      scale: epObj.scale,
      confidence: r3(conf),
      threshold: ep === "univariate" ? r3(2.0 + rnd(0, 0.6)) : null,
      severity: sevWord.toLowerCase(),
      decision: withheld ? "WITHHELD" : "PERMITTED",
      ethical: {
        benevolence: benev,
        sigma_immutable: r3(rnd(0.96, 0.99)),
        lyapunov: 0.25,
        gate_margin: gateMargin
      },
      domain: dom.id,
      sector: dom.sector
    };

    // audit_trail — the real governance ledger
    const t0 = new Date(d.getTime());
    const at = [];
    const step = (ms, event, actor, detail) => at.push({
      t: fmtTs(new Date(t0.getTime() + ms)),
      event,
      actor,
      detail
    });
    step(0, "received", "api/auth", "X-API-Key authed · " + corr);
    step(34, "detected", epObj.id + " engine", "score " + score + " (" + epObj.scale + ") · conf " + r3(conf));
    step(58, "gate-eval", "EthicalGate", "η " + benev + (withheld ? " < 0.99" : " ≥ 0.99") + " · σ ok");
    if (withheld) step(61, "withheld", "EthicalGate", "EthicalConstraintViolationError(check=\"benevolence\")");else step(70, "disclosed", "tlp_handler", "tagged " + tlp);
    if (isAnom && !withheld && ackKey) step(2100 + Math.floor(rnd(0, 9000)), "acknowledged", "analyst@soc", "action staged: " + action);
    if (!isAnom && Math.random() < 0.3) step(120, "archived", "retention", "nominal · 30-day window");
    return {
      id: corr,
      seq: 100000 + seq,
      ts: fmtTs(d),
      ageMs,
      domain: dom.label,
      domainId: dom.id,
      sector: dom.sector,
      endpoint: ep,
      method,
      scale: epObj.scale,
      isAnom,
      state: isAnom ? "ANOMALY" : "NORMAL",
      score,
      conf: r3(conf),
      sevKey,
      sevWord,
      benev,
      withheld,
      gateMargin,
      decision: withheld ? "WITHHELD" : "PERMITTED",
      tlp,
      action,
      corr,
      raw,
      audit: at
    };
  }
  function buildCorpus(n) {
    const now = Date.now();
    const out = [];
    for (let i = 0; i < n; i++) out.push(genRecord(i, now));
    out.sort((a, b) => a.ageMs - b.ageMs); // newest first
    return out;
  }

  // facet definitions — each is a dimension the explorer can filter on, with the
  // accessor that maps a record to its value(s). Multi-select within a facet = OR;
  // across facets = AND. Counts shown per option are computed against all OTHER
  // active facets (true faceted search).
  const FACETS = [{
    id: "state",
    label: "State",
    get: r => r.state,
    order: ["ANOMALY", "NORMAL"]
  }, {
    id: "sevKey",
    label: "Severity",
    get: r => r.sevKey,
    order: ["critical", "high", "medium", "low"],
    render: v => v.toUpperCase()
  }, {
    id: "endpoint",
    label: "Endpoint",
    get: r => r.endpoint,
    order: ENDPOINTS.map(e => e.id)
  }, {
    id: "sector",
    label: "Sector",
    get: r => r.sector,
    order: ["Security", "Space & Env", "Medical", "Infrastructure", "Humanitarian"]
  }, {
    id: "domainId",
    label: "Domain",
    get: r => r.domainId,
    order: DOMAINS.map(d => d.id),
    render: v => (DOMAINS.find(d => d.id === v) || {}).label || v
  }, {
    id: "decision",
    label: "Ethics gate",
    get: r => r.decision,
    order: ["PERMITTED", "WITHHELD"]
  }, {
    id: "tlp",
    label: "Disclosure (TLP)",
    get: r => r.tlp,
    order: ["TLP:RED", "TLP:AMBER+STRICT", "TLP:AMBER", "TLP:GREEN", "TLP:CLEAR"]
  }];
  const WINDOWS = [{
    id: "15m",
    label: "15 min",
    ms: 15 * 60 * 1000
  }, {
    id: "1h",
    label: "1 hour",
    ms: 3600 * 1000
  }, {
    id: "6h",
    label: "6 hours",
    ms: 6 * 3600 * 1000
  }, {
    id: "24h",
    label: "24 hours",
    ms: 24 * 3600 * 1000
  }];
  return {
    ENDPOINTS,
    DOMAINS,
    LADDERS,
    FACETS,
    WINDOWS,
    buildCorpus
  };
}();
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/explorer/ExplorerData.js", error: String((e && e.message) || e) }); }

// ui_kits/hazard/Hazard.jsx
try { (() => {
/* Emergency Operations Console — APP
   A mixed-watch-floor surface over the multi-domain hazard corpus. Slicers
   (domain · severity · region), range sliders (distance · time-to-impact · a
   domain-native metric), one-tap threshold presets, a sortable list with LIVE
   derived columns (ticking ETA · distance · bearing), and a detail panel with a
   lifecycle timeline + countdown wired to each event's real prediction fields.
   Uses the shared <mercury-status>, <mercury-agent-dock>, <mercury-command-palette>. */
const {
  useState,
  useMemo,
  useEffect,
  useRef
} = React;
const {
  DOMAINS,
  DOMAIN_IDS,
  buildCorpus,
  fieldRows,
  REF
} = window.HAZARD;
const HazardMap = window.HazardMap;
const CORPUS = buildCorpus(74);

/* operator unit preference — bound to the system Settings (mercury-settings).
   Distances are computed in km; imperial operators see miles. Read from
   localStorage synchronously so the first render is already correct; the
   setting takes effect on the next page load (navigation reloads the surface). */
const UNITS = function () {
  try {
    return (JSON.parse(localStorage.getItem("mercury-settings")) || {}).units === "imperial" ? "imperial" : "metric";
  } catch (e) {
    return "metric";
  }
}();
const DIST_U = UNITS === "imperial" ? "mi" : "km";
const distVal = km => UNITS === "imperial" ? Math.round(km * 0.621371) : km;
const fmtDist = km => distVal(km).toLocaleString() + " " + DIST_U;
const SEV_RANK = {
  critical: 4,
  high: 3,
  medium: 2,
  low: 1
};
const SEV_SCALE = {
  critical: 4.4,
  high: 3.4,
  medium: 2.6,
  low: 2
};
const REGIONS = ["All regions", ...Array.from(new Set(CORPUS.map(e => e.region))).sort()];
function fmtCountdown(sec) {
  if (sec == null) return {
    t: "—",
    cls: ""
  };
  if (sec <= 0) return {
    t: "ELAPSED",
    cls: "elapsed"
  };
  const h = Math.floor(sec / 3600),
    m = Math.floor(sec % 3600 / 60),
    s = Math.floor(sec % 60);
  if (h >= 1) return {
    t: h + "h " + String(m).padStart(2, "0") + "m",
    cls: h < 6 ? "soon" : ""
  };
  if (m >= 1) return {
    t: m + "m " + String(s).padStart(2, "0") + "s",
    cls: "soon"
  };
  return {
    t: s + "s",
    cls: "imminent"
  };
}
/* ---- time-zone handling ---------------------------------------------------
   Ops standard is zulu (UTC); operators may switch the master clock to their own
   local zone. UTC + operator-local are EXACT (browser Intl); a hazard's site time
   is an APPROXIMATION derived from its longitude (15°/h solar offset) — labelled
   as such, per the honesty rule, since we don't ship a civil-tz polygon lookup. */
function opTz() {
  try {
    var tz = Intl.DateTimeFormat().resolvedOptions().timeZone || "local";
    var mins = -new Date().getTimezoneOffset();
    var sign = mins >= 0 ? "+" : "−",
      a = Math.abs(mins);
    var off = "UTC" + sign + String(Math.floor(a / 60)).padStart(2, "0") + ":" + String(a % 60).padStart(2, "0");
    return {
      tz: tz,
      off: off,
      abbr: tz.split("/").pop().replace(/_/g, " ")
    };
  } catch (e) {
    return {
      tz: "local",
      off: "",
      abbr: "local"
    };
  }
}
const OPTZ = opTz();
const p2 = n => String(n).padStart(2, "0");
function fmtClock(ms, mode) {
  const d = new Date(ms);
  if (mode === "local") return p2(d.getHours()) + ":" + p2(d.getMinutes()) + ":" + p2(d.getSeconds());
  return p2(d.getUTCHours()) + ":" + p2(d.getUTCMinutes()) + ":" + p2(d.getUTCSeconds()) + "Z";
}
function fmtAbs(ms, mode) {
  if (ms == null) return "—";
  const d = new Date(ms);
  if (mode === "local") return p2(d.getMonth() + 1) + "/" + p2(d.getDate()) + " " + p2(d.getHours()) + ":" + p2(d.getMinutes());
  return p2(d.getUTCMonth() + 1) + "/" + p2(d.getUTCDate()) + " " + p2(d.getUTCHours()) + ":" + p2(d.getUTCMinutes()) + "Z";
}
function siteLocal(ms, lng) {
  if (ms == null) return {
    t: "—",
    off: ""
  };
  const offH = Math.round(lng / 15);
  const d = new Date(ms + offH * 3600000);
  return {
    t: p2(d.getUTCHours()) + ":" + p2(d.getUTCMinutes()),
    off: "UTC" + (offH >= 0 ? "+" : "−") + Math.abs(offH)
  };
}
const PRESETS = [{
  id: "all",
  label: "All active"
}, {
  id: "crit",
  label: "Critical & inbound"
}, {
  id: "soon",
  label: "Impact < 24h"
}, {
  id: "withheld",
  label: "Withheld by gate"
}, {
  id: "near",
  label: "Within " + fmtDist(2500)
}];
function Chrome({
  now,
  count,
  total,
  onPalette,
  tzMode,
  onToggleTz
}) {
  return /*#__PURE__*/React.createElement("header", {
    className: "mc-chrome"
  }, /*#__PURE__*/React.createElement("div", {
    className: "mc-brand"
  }, /*#__PURE__*/React.createElement("span", {
    className: "mc-sig"
  }, "\u2671"), /*#__PURE__*/React.createElement("div", {
    className: "mc-titles"
  }, /*#__PURE__*/React.createElement("div", {
    className: "mc-wm"
  }, /*#__PURE__*/React.createElement("span", {
    className: "g"
  }, "M"), /*#__PURE__*/React.createElement("span", {
    className: "c"
  }, "ERCURY"), " ", /*#__PURE__*/React.createElement("span", {
    className: "g"
  }, "A"), /*#__PURE__*/React.createElement("span", {
    className: "k"
  }, "GENT")), /*#__PURE__*/React.createElement("div", {
    className: "mc-sub"
  }, "Emergency Operations Console"))), /*#__PURE__*/React.createElement("div", {
    className: "mc-right"
  }, /*#__PURE__*/React.createElement("button", {
    className: "mc-stat mc-clock",
    onClick: onToggleTz,
    title: "Master clock — click to switch to " + (tzMode === "utc" ? "local time (" + OPTZ.off + ")" : "UTC / zulu")
  }, /*#__PURE__*/React.createElement("span", {
    className: "mc-stat-dot"
  }), fmtClock(now, tzMode), /*#__PURE__*/React.createElement("span", {
    className: "mc-clock-z"
  }, tzMode === "utc" ? "UTC" : OPTZ.abbr)), /*#__PURE__*/React.createElement("span", {
    className: "mc-stat"
  }, /*#__PURE__*/React.createElement("b", null, count), " ", /*#__PURE__*/React.createElement("span", {
    className: "of"
  }, "/ ", total, " events")), /*#__PURE__*/React.createElement("button", {
    className: "mc-kbtn",
    onClick: onPalette
  }, /*#__PURE__*/React.createElement("span", {
    className: "kb"
  }, "\u2318K"), " commands"), /*#__PURE__*/React.createElement("mercury-status", {
    state: "REFERENCE",
    tone: "warning",
    kind: "badge"
  })));
}
function Slider({
  label,
  min,
  max,
  step,
  value,
  onChange,
  fmt
}) {
  return /*#__PURE__*/React.createElement("div", {
    className: "hz-slider"
  }, /*#__PURE__*/React.createElement("div", {
    className: "hz-slider-h"
  }, /*#__PURE__*/React.createElement("span", null, label), /*#__PURE__*/React.createElement("b", null, fmt ? fmt(value) : value)), /*#__PURE__*/React.createElement("input", {
    type: "range",
    min: min,
    max: max,
    step: step,
    value: value,
    onChange: e => onChange(+e.target.value)
  }));
}
function SlicerRail(props) {
  const {
    domainSet,
    toggleDomain,
    minSev,
    setMinSev,
    region,
    setRegion,
    maxDist,
    setMaxDist,
    maxEta,
    setMaxEta,
    domainSlider,
    setDomainSlider,
    preset,
    applyPreset,
    soloDomain
  } = props;
  return /*#__PURE__*/React.createElement("aside", {
    className: "hz-rail"
  }, /*#__PURE__*/React.createElement("div", {
    className: "hz-rail-sec"
  }, /*#__PURE__*/React.createElement("div", {
    className: "hz-rail-h"
  }, "PRESETS"), /*#__PURE__*/React.createElement("div", {
    className: "hz-presets"
  }, PRESETS.map(p => /*#__PURE__*/React.createElement("button", {
    key: p.id,
    className: "hz-preset" + (preset === p.id ? " on" : ""),
    onClick: () => applyPreset(p.id)
  }, p.label)))), /*#__PURE__*/React.createElement("div", {
    className: "hz-rail-sec"
  }, /*#__PURE__*/React.createElement("div", {
    className: "hz-rail-h"
  }, "DOMAIN"), /*#__PURE__*/React.createElement("div", {
    className: "hz-domains"
  }, DOMAIN_IDS.map(id => {
    const on = domainSet.has(id);
    return /*#__PURE__*/React.createElement("button", {
      key: id,
      className: "hz-dom" + (on ? " on" : ""),
      onClick: () => toggleDomain(id)
    }, /*#__PURE__*/React.createElement("span", {
      className: "hz-dom-g"
    }, DOMAINS[id].glyph), /*#__PURE__*/React.createElement("span", {
      className: "hz-dom-l"
    }, DOMAINS[id].label));
  }))), /*#__PURE__*/React.createElement("div", {
    className: "hz-rail-sec"
  }, /*#__PURE__*/React.createElement("div", {
    className: "hz-rail-h"
  }, "SEVERITY FLOOR"), /*#__PURE__*/React.createElement("div", {
    className: "hz-sevseg"
  }, ["low", "medium", "high", "critical"].map(s => /*#__PURE__*/React.createElement("button", {
    key: s,
    className: "hz-sevb" + (minSev === s ? " on" : ""),
    onClick: () => setMinSev(s)
  }, s === "low" ? "ALL" : s.slice(0, 4).toUpperCase())))), /*#__PURE__*/React.createElement("div", {
    className: "hz-rail-sec"
  }, /*#__PURE__*/React.createElement("div", {
    className: "hz-rail-h"
  }, "REGION"), /*#__PURE__*/React.createElement("select", {
    className: "hz-select",
    value: region,
    onChange: e => setRegion(e.target.value)
  }, REGIONS.map(r => /*#__PURE__*/React.createElement("option", {
    key: r,
    value: r
  }, r)))), /*#__PURE__*/React.createElement("div", {
    className: "hz-rail-sec"
  }, /*#__PURE__*/React.createElement("div", {
    className: "hz-rail-h"
  }, "RANGE FILTERS"), /*#__PURE__*/React.createElement(Slider, {
    label: "Max distance",
    min: 250,
    max: 8000,
    step: 250,
    value: maxDist,
    onChange: setMaxDist,
    fmt: fmtDist
  }), /*#__PURE__*/React.createElement(Slider, {
    label: "Max time-to-impact",
    min: 1,
    max: 72,
    step: 1,
    value: maxEta,
    onChange: setMaxEta,
    fmt: v => v + " h"
  }), soloDomain && DOMAINS[soloDomain].slider && /*#__PURE__*/React.createElement(Slider, {
    label: "Min " + DOMAINS[soloDomain].slider.label,
    min: DOMAINS[soloDomain].slider.min,
    max: DOMAINS[soloDomain].slider.max,
    step: DOMAINS[soloDomain].slider.step,
    value: domainSlider,
    onChange: setDomainSlider,
    fmt: DOMAINS[soloDomain].slider.fmt
  }), !soloDomain && /*#__PURE__*/React.createElement("div", {
    className: "hz-slider-note"
  }, "Select a single domain to filter by its native metric (magnitude, wind, VEI\u2026).")));
}
const SORTS = [{
  id: "eta",
  label: "soonest",
  cmp: (a, b) => (a.etaS == null ? 1e12 : a.etaS) - (b.etaS == null ? 1e12 : b.etaS)
}, {
  id: "sev",
  label: "severity",
  cmp: (a, b) => SEV_RANK[b.sevKey] - SEV_RANK[a.sevKey]
}, {
  id: "dist",
  label: "nearest",
  cmp: (a, b) => a.dist - b.dist
}, {
  id: "conf",
  label: "confidence",
  cmp: (a, b) => b.confidence - a.confidence
}];
function EventRow({
  ev,
  now,
  sel,
  onSel
}) {
  const cd = fmtCountdown(ev.clockS == null ? null : ev.clockS - (now - ev.now) / 1000);
  return /*#__PURE__*/React.createElement("tr", {
    className: "hz-row" + (sel ? " sel" : ""),
    onClick: onSel
  }, /*#__PURE__*/React.createElement("td", {
    className: "hz-td-g"
  }, /*#__PURE__*/React.createElement("span", {
    className: "hz-row-g",
    style: {
      color: statusHue(ev.status)
    }
  }, ev.glyph)), /*#__PURE__*/React.createElement("td", null, /*#__PURE__*/React.createElement("span", {
    className: "hz-row-dom"
  }, ev.domain), /*#__PURE__*/React.createElement("span", {
    className: "hz-row-place"
  }, ev.place)), /*#__PURE__*/React.createElement("td", null, /*#__PURE__*/React.createElement("span", {
    className: "hz-row-head"
  }, ev.headline)), /*#__PURE__*/React.createElement("td", null, /*#__PURE__*/React.createElement("mercury-status", {
    state: ev.status,
    kind: "badge"
  })), /*#__PURE__*/React.createElement("td", {
    className: "hz-td-eta"
  }, /*#__PURE__*/React.createElement("span", {
    className: "hz-eta " + cd.cls
  }, cd.t), /*#__PURE__*/React.createElement("span", {
    className: "hz-eta-k"
  }, ev.inbound ? "to impact" : "duration")), /*#__PURE__*/React.createElement("td", {
    className: "hz-td-num"
  }, distVal(ev.dist).toLocaleString(), /*#__PURE__*/React.createElement("span", {
    className: "hz-u"
  }, DIST_U)), /*#__PURE__*/React.createElement("td", {
    className: "hz-td-dir"
  }, ev.compass, /*#__PURE__*/React.createElement("span", {
    className: "hz-u"
  }, Math.round(ev.bearing), "\xB0")), /*#__PURE__*/React.createElement("td", null, /*#__PURE__*/React.createElement("mercury-status", {
    state: ev.tlp,
    kind: "tlp"
  })));
}
function statusHue(s) {
  return {
    CRITICAL: "var(--anomaly)",
    WARNING: "var(--warning-bright)",
    WATCH: "var(--mercury-300)",
    ADVISORY: "var(--normal-bright)",
    WITHHELD: "var(--withheld-soft)"
  }[s];
}
function Timeline({
  ev,
  now
}) {
  const total = (ev.ends || ev.impact || ev.now) - ev.inception;
  const elapsed = ev.now + (now - ev.now) - ev.inception;
  const pct = Math.max(0, Math.min(100, elapsed / total * 100));
  const impactPct = ev.impact ? Math.max(0, Math.min(100, (ev.impact - ev.inception) / total * 100)) : null;
  return /*#__PURE__*/React.createElement("div", {
    className: "hz-tl"
  }, /*#__PURE__*/React.createElement("div", {
    className: "hz-tl-track"
  }, /*#__PURE__*/React.createElement("div", {
    className: "hz-tl-fill",
    style: {
      width: pct + "%"
    }
  }), impactPct != null && /*#__PURE__*/React.createElement("div", {
    className: "hz-tl-impact",
    style: {
      left: impactPct + "%"
    },
    title: "projected impact"
  }), /*#__PURE__*/React.createElement("div", {
    className: "hz-tl-now",
    style: {
      left: pct + "%"
    }
  })), /*#__PURE__*/React.createElement("div", {
    className: "hz-tl-labels"
  }, /*#__PURE__*/React.createElement("span", null, "detected ", fmtAgo(now - ev.inception)), /*#__PURE__*/React.createElement("span", {
    className: "hz-tl-mid"
  }, ev.lifecycle), /*#__PURE__*/React.createElement("span", null, ev.inbound ? "impact" : "ends")));
}
function fmtAgo(ms) {
  const m = Math.floor(ms / 60000);
  return m < 60 ? m + "m ago" : Math.floor(m / 60) + "h ago";
}
function Detail({
  ev,
  now,
  onClose,
  tzMode
}) {
  if (!ev) return /*#__PURE__*/React.createElement("aside", {
    className: "hz-detail hz-detail-empty"
  }, /*#__PURE__*/React.createElement("div", {
    className: "hz-empty-sig"
  }, "\u2671"), /*#__PURE__*/React.createElement("div", {
    className: "hz-empty-t"
  }, "Select an event"), /*#__PURE__*/React.createElement("div", {
    className: "hz-empty-b"
  }, "Pick a marker on the plot or a row in the list to see its full prediction record, lifecycle timeline and recommended actions."));
  const cd = fmtCountdown(ev.clockS == null ? null : ev.clockS - (now - ev.now) / 1000);
  const rows = fieldRows(ev);
  const impactMs = ev.impact || ev.ends || null;
  const site = siteLocal(impactMs || ev.now, ev.lng);
  return /*#__PURE__*/React.createElement("aside", {
    className: "hz-detail"
  }, /*#__PURE__*/React.createElement("div", {
    className: "hz-d-head"
  }, /*#__PURE__*/React.createElement("span", {
    className: "hz-d-glyph",
    style: {
      color: statusHue(ev.status)
    }
  }, ev.glyph), /*#__PURE__*/React.createElement("div", {
    className: "hz-d-id"
  }, /*#__PURE__*/React.createElement("div", {
    className: "hz-d-dom"
  }, ev.domain, /*#__PURE__*/React.createElement("span", {
    className: "hz-d-seq"
  }, "#", ev.seq)), /*#__PURE__*/React.createElement("div", {
    className: "hz-d-place"
  }, ev.place, " \xB7 ", ev.region)), /*#__PURE__*/React.createElement("span", {
    className: "hz-d-x",
    onClick: onClose
  }, "\u2715")), /*#__PURE__*/React.createElement("div", {
    className: "hz-d-statusrow"
  }, /*#__PURE__*/React.createElement("mercury-status", {
    state: ev.status,
    kind: "badge"
  }), /*#__PURE__*/React.createElement("mercury-status", {
    state: ev.tlp,
    kind: "tlp"
  }), /*#__PURE__*/React.createElement("span", {
    className: "hz-d-conf"
  }, "conf ", Math.round(ev.confidence * 100), "%")), /*#__PURE__*/React.createElement("div", {
    className: "hz-d-clockbig"
  }, /*#__PURE__*/React.createElement("div", {
    className: "hz-d-cd " + cd.cls
  }, cd.t), /*#__PURE__*/React.createElement("div", {
    className: "hz-d-cd-k"
  }, ev.inbound ? "projected time to impact" : "active — projected duration remaining")), /*#__PURE__*/React.createElement("div", {
    className: "hz-d-times"
  }, /*#__PURE__*/React.createElement("div", {
    className: "hz-d-time"
  }, /*#__PURE__*/React.createElement("span", {
    className: "k"
  }, "detected"), /*#__PURE__*/React.createElement("span", {
    className: "v"
  }, fmtAbs(ev.inception, tzMode))), /*#__PURE__*/React.createElement("div", {
    className: "hz-d-time"
  }, /*#__PURE__*/React.createElement("span", {
    className: "k"
  }, ev.inbound ? "impact" : "ends"), /*#__PURE__*/React.createElement("span", {
    className: "v"
  }, fmtAbs(impactMs, tzMode))), /*#__PURE__*/React.createElement("div", {
    className: "hz-d-time site"
  }, /*#__PURE__*/React.createElement("span", {
    className: "k"
  }, "site local \u2248"), /*#__PURE__*/React.createElement("span", {
    className: "v"
  }, site.t, " ", /*#__PURE__*/React.createElement("em", null, site.off)))), /*#__PURE__*/React.createElement("div", {
    className: "hz-d-tznote"
  }, "absolute times in ", /*#__PURE__*/React.createElement("b", null, tzMode === "utc" ? "UTC · zulu" : OPTZ.abbr + " · " + OPTZ.off), " \xB7 site time approximated from longitude"), /*#__PURE__*/React.createElement(Timeline, {
    ev: ev,
    now: now
  }), /*#__PURE__*/React.createElement("div", {
    className: "hz-d-geo"
  }, /*#__PURE__*/React.createElement("div", {
    className: "hz-d-geo-c"
  }, /*#__PURE__*/React.createElement("span", {
    className: "hz-d-geo-v"
  }, distVal(ev.dist).toLocaleString()), /*#__PURE__*/React.createElement("span", {
    className: "hz-d-geo-k"
  }, DIST_U, " from floor")), /*#__PURE__*/React.createElement("div", {
    className: "hz-d-geo-c"
  }, /*#__PURE__*/React.createElement("span", {
    className: "hz-d-geo-v"
  }, ev.compass), /*#__PURE__*/React.createElement("span", {
    className: "hz-d-geo-k"
  }, "bearing ", Math.round(ev.bearing), "\xB0")), /*#__PURE__*/React.createElement("div", {
    className: "hz-d-geo-c"
  }, /*#__PURE__*/React.createElement("span", {
    className: "hz-d-geo-v"
  }, ev.lat.toFixed(1), "\xB0"), /*#__PURE__*/React.createElement("span", {
    className: "hz-d-geo-k"
  }, "lat")), /*#__PURE__*/React.createElement("div", {
    className: "hz-d-geo-c"
  }, /*#__PURE__*/React.createElement("span", {
    className: "hz-d-geo-v"
  }, ev.lng.toFixed(1), "\xB0"), /*#__PURE__*/React.createElement("span", {
    className: "hz-d-geo-k"
  }, "lng"))), /*#__PURE__*/React.createElement("div", {
    className: "hz-d-sec"
  }, "PREDICTION \xB7 ", /*#__PURE__*/React.createElement("span", {
    className: "hz-d-sec-m"
  }, ev.dataclass)), /*#__PURE__*/React.createElement("div", {
    className: "hz-d-fields"
  }, rows.map((r, i) => /*#__PURE__*/React.createElement("div", {
    className: "hz-d-f",
    key: i
  }, /*#__PURE__*/React.createElement("span", {
    className: "hz-d-f-k"
  }, r.label), /*#__PURE__*/React.createElement("span", {
    className: "hz-d-f-v"
  }, r.value)))), ev.evac && ev.evac.length > 0 && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("div", {
    className: "hz-d-sec"
  }, "ZONES / ROUTES"), /*#__PURE__*/React.createElement("div", {
    className: "hz-d-evac"
  }, ev.evac.map((z, i) => /*#__PURE__*/React.createElement("span", {
    key: i,
    className: "hz-d-zone"
  }, z)))), /*#__PURE__*/React.createElement("div", {
    className: "hz-d-sec"
  }, "ETHICAL GATE"), /*#__PURE__*/React.createElement("div", {
    className: "hz-d-gate" + (ev.withheld ? " wh" : "")
  }, /*#__PURE__*/React.createElement("mercury-status", {
    state: ev.withheld ? "WITHHELD" : "PERMITTED",
    kind: "dot",
    tone: ev.withheld ? "withheld" : "normal"
  }), /*#__PURE__*/React.createElement("span", {
    className: "hz-d-gate-eta"
  }, "\u03B7 ", ev.benev.toFixed(3)), /*#__PURE__*/React.createElement("span", {
    className: "hz-d-gate-note"
  }, ev.withheld ? "η < 0.99 — emission withheld pending human review" : "η ≥ 0.99 — disclosure permitted")), /*#__PURE__*/React.createElement("div", {
    className: "hz-d-corr"
  }, /*#__PURE__*/React.createElement("span", null, "correlation"), /*#__PURE__*/React.createElement("code", null, ev.corr)));
}
function App() {
  const [now, setNow] = useState(Date.now());
  const [tzMode, setTzMode] = useState(() => {
    try {
      return localStorage.getItem("hz_tz") === "local" ? "local" : "utc";
    } catch (e) {
      return "utc";
    }
  });
  const toggleTz = () => setTzMode(m => {
    const next = m === "utc" ? "local" : "utc";
    try {
      localStorage.setItem("hz_tz", next);
    } catch (e) {}
    return next;
  });
  const [domainSet, setDomainSet] = useState(new Set());
  const [minSev, setMinSev] = useState("low");
  const [region, setRegion] = useState("All regions");
  const [maxDist, setMaxDist] = useState(8000);
  const [maxEta, setMaxEta] = useState(72);
  const [domainSlider, setDomainSlider] = useState(0);
  const [preset, setPreset] = useState("all");
  const [sort, setSort] = useState("eta");
  const [selectedId, setSelectedId] = useState(null);
  const palRef = useRef(null),
    dockRef = useRef(null);
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, []);
  const soloDomain = domainSet.size === 1 ? [...domainSet][0] : null;
  useEffect(() => {
    setDomainSlider(soloDomain && DOMAINS[soloDomain].slider ? DOMAINS[soloDomain].slider.min : 0);
  }, [soloDomain]);
  const toggleDomain = id => {
    setPreset("all");
    setDomainSet(prev => {
      const n = new Set(prev);
      n.has(id) ? n.delete(id) : n.add(id);
      return n;
    });
  };
  function applyPreset(id) {
    setPreset(id);
    setDomainSet(new Set());
    setMinSev("low");
    setRegion("All regions");
    setMaxDist(8000);
    setMaxEta(72);
    if (id === "near") setMaxDist(2500);
    if (id === "crit") setMinSev("critical");
  }
  const filtered = useMemo(() => CORPUS.filter(ev => {
    if (domainSet.size && !domainSet.has(ev.id)) return false;
    if (SEV_RANK[ev.sevKey] < SEV_RANK[minSev]) return false;
    if (region !== "All regions" && ev.region !== region) return false;
    if (ev.dist > maxDist) return false;
    const etaH = ev.etaS != null ? ev.etaS / 3600 : null;
    if (etaH != null && etaH > maxEta) return false;
    if (soloDomain && DOMAINS[soloDomain].slider) {
      const k = DOMAINS[soloDomain].slider.key;
      if ((ev.fields[k] ?? 0) < domainSlider) return false;
    }
    if (preset === "crit" && !(ev.sevKey === "critical" && ev.inbound)) return false;
    if (preset === "soon" && !(etaH != null && etaH < 24)) return false;
    if (preset === "withheld" && !ev.withheld) return false;
    return true;
  }), [domainSet, minSev, region, maxDist, maxEta, domainSlider, soloDomain, preset]);
  const sorted = useMemo(() => filtered.slice().sort(SORTS.find(s => s.id === sort).cmp), [filtered, sort]);
  const selected = sorted.find(e => e.seq === selectedId) || CORPUS.find(e => e.seq === selectedId) || null;
  const summary = useMemo(() => ({
    crit: filtered.filter(e => e.sevKey === "critical").length,
    inbound: filtered.filter(e => e.inbound && e.etaS / 3600 < 24).length,
    withheld: filtered.filter(e => e.withheld).length
  }), [filtered]);

  // shared components
  useEffect(() => {
    if (dockRef.current) dockRef.current.context = {
      title: "Emergency Operations Console",
      blurb: "A multi-domain watch floor. Each event is wired to a real engine prediction-result schema (earthquake, hurricane, tsunami, meteor, solar, flood, tornado, wildfire, volcanic, landslide). Filter on the left; the plot and list update live.",
      status: {
        label: "live · " + filtered.length + " active",
        tone: "normal"
      },
      prompts: [{
        label: "Where does the data come from?",
        answer: "Each event matches a shipped <b>*PredictionResult</b> dataclass — its real fields, units and ranges. Synthetic instances at plausible locations; a forward-design reference concept, labeled, never live data."
      }, {
        label: "What's the countdown?",
        answer: "Each domain's native impact clock (S-wave arrival, time-to-landfall, closest approach, time-to-crest…) normalized to a live <b>time-to-impact</b> and a lifecycle timeline."
      }, {
        label: "What is WITHHELD?",
        answer: "A high-consequence prediction the <b>ethical gate</b> held (η < 0.99) pending human review — shown in the canonical grey, not as a threat."
      }],
      links: [{
        label: "Detection & Audit Explorer",
        href: "../explorer/index.html"
      }, {
        label: "Console",
        href: "../console/index.html"
      }, {
        label: "CLI",
        href: "../cli/index.html"
      }, {
        label: "Cryptography",
        href: "../crypto/index.html"
      }]
    };
  }, [filtered.length]);
  useEffect(() => {
    if (!palRef.current) return;
    palRef.current.commands = [{
      id: "crit",
      group: "Presets",
      label: "Critical & inbound",
      glyph: "▲",
      run: () => applyPreset("crit")
    }, {
      id: "soon",
      group: "Presets",
      label: "Impact within 24 hours",
      glyph: "⏱",
      run: () => applyPreset("soon")
    }, {
      id: "withheld",
      group: "Presets",
      label: "Withheld by ethics gate",
      glyph: "η",
      run: () => applyPreset("withheld")
    }, {
      id: "near",
      group: "Presets",
      label: "Within " + fmtDist(2500),
      glyph: "◎",
      run: () => applyPreset("near")
    }, {
      id: "all",
      group: "Presets",
      label: "Clear all filters",
      glyph: "○",
      run: () => applyPreset("all")
    }, ...DOMAIN_IDS.map(id => ({
      id: "d-" + id,
      group: "Focus domain",
      label: DOMAINS[id].label,
      glyph: DOMAINS[id].glyph,
      run: () => {
        setPreset("all");
        setDomainSet(new Set([id]));
      }
    })), {
      id: "s-eta",
      group: "Sort",
      label: "Sort by soonest impact",
      glyph: "↓",
      run: () => setSort("eta")
    }, {
      id: "s-sev",
      group: "Sort",
      label: "Sort by severity",
      glyph: "↓",
      run: () => setSort("sev")
    }, {
      id: "s-dist",
      group: "Sort",
      label: "Sort by nearest",
      glyph: "↓",
      run: () => setSort("dist")
    }, {
      id: "g-explorer",
      group: "Navigate",
      label: "Open Detection & Audit Explorer",
      kbd: "G E",
      glyph: "→",
      run: () => location.href = "../explorer/index.html"
    }, {
      id: "g-console",
      group: "Navigate",
      label: "Open Console",
      kbd: "G C",
      glyph: "→",
      run: () => location.href = "../console/index.html"
    }];
  }, [filtered.length]);
  return /*#__PURE__*/React.createElement("div", {
    className: "hz-app"
  }, /*#__PURE__*/React.createElement(Chrome, {
    now: now,
    count: filtered.length,
    total: CORPUS.length,
    onPalette: () => palRef.current && palRef.current.open(),
    tzMode: tzMode,
    onToggleTz: toggleTz
  }), /*#__PURE__*/React.createElement("div", {
    className: "hz-body"
  }, /*#__PURE__*/React.createElement(SlicerRail, {
    domainSet: domainSet,
    toggleDomain: toggleDomain,
    minSev: minSev,
    setMinSev: s => {
      setPreset("all");
      setMinSev(s);
    },
    region: region,
    setRegion: r => {
      setPreset("all");
      setRegion(r);
    },
    maxDist: maxDist,
    setMaxDist: v => {
      setPreset("all");
      setMaxDist(v);
    },
    maxEta: maxEta,
    setMaxEta: v => {
      setPreset("all");
      setMaxEta(v);
    },
    domainSlider: domainSlider,
    setDomainSlider: setDomainSlider,
    preset: preset,
    applyPreset: applyPreset,
    soloDomain: soloDomain
  }), /*#__PURE__*/React.createElement("main", {
    className: "hz-main"
  }, /*#__PURE__*/React.createElement("div", {
    className: "hz-map-region"
  }, /*#__PURE__*/React.createElement(HazardMap, {
    events: filtered,
    selectedId: selectedId,
    onSelect: setSelectedId,
    sevScale: SEV_SCALE,
    regionFilter: region,
    onRegionSelect: r => {
      setPreset("all");
      setRegion(r);
    }
  }), /*#__PURE__*/React.createElement("div", {
    className: "hz-map-stats"
  }, /*#__PURE__*/React.createElement("div", {
    className: "hz-mstat"
  }, /*#__PURE__*/React.createElement("span", {
    className: "hz-mstat-v an"
  }, summary.crit), /*#__PURE__*/React.createElement("span", {
    className: "hz-mstat-k"
  }, "critical")), /*#__PURE__*/React.createElement("div", {
    className: "hz-mstat"
  }, /*#__PURE__*/React.createElement("span", {
    className: "hz-mstat-v wn"
  }, summary.inbound), /*#__PURE__*/React.createElement("span", {
    className: "hz-mstat-k"
  }, "inbound < 24h")), /*#__PURE__*/React.createElement("div", {
    className: "hz-mstat"
  }, /*#__PURE__*/React.createElement("span", {
    className: "hz-mstat-v wh"
  }, summary.withheld), /*#__PURE__*/React.createElement("span", {
    className: "hz-mstat-k"
  }, "withheld")))), /*#__PURE__*/React.createElement("div", {
    className: "hz-list"
  }, /*#__PURE__*/React.createElement("div", {
    className: "hz-list-h"
  }, /*#__PURE__*/React.createElement("span", {
    className: "hz-list-count"
  }, /*#__PURE__*/React.createElement("b", null, sorted.length), " events"), /*#__PURE__*/React.createElement("div", {
    className: "hz-sort"
  }, /*#__PURE__*/React.createElement("span", {
    className: "hz-sort-l"
  }, "sort"), SORTS.map(s => /*#__PURE__*/React.createElement("button", {
    key: s.id,
    className: "hz-sort-b" + (sort === s.id ? " on" : ""),
    onClick: () => setSort(s.id)
  }, s.label)))), /*#__PURE__*/React.createElement("div", {
    className: "hz-table-wrap"
  }, /*#__PURE__*/React.createElement("table", {
    className: "hz-table"
  }, /*#__PURE__*/React.createElement("thead", null, /*#__PURE__*/React.createElement("tr", null, /*#__PURE__*/React.createElement("th", null), /*#__PURE__*/React.createElement("th", null, "DOMAIN \xB7 LOCATION"), /*#__PURE__*/React.createElement("th", null, "ASSESSMENT"), /*#__PURE__*/React.createElement("th", null, "STATUS"), /*#__PURE__*/React.createElement("th", null, "ETA"), /*#__PURE__*/React.createElement("th", null, "DIST"), /*#__PURE__*/React.createElement("th", null, "DIR"), /*#__PURE__*/React.createElement("th", null, "TLP"))), /*#__PURE__*/React.createElement("tbody", null, sorted.length === 0 && /*#__PURE__*/React.createElement("tr", null, /*#__PURE__*/React.createElement("td", {
    colSpan: "8",
    className: "hz-empty-row"
  }, "No events match these filters. Loosen a slicer or pick a preset.")), sorted.map(ev => /*#__PURE__*/React.createElement(EventRow, {
    key: ev.seq,
    ev: ev,
    now: now,
    sel: ev.seq === selectedId,
    onSel: () => setSelectedId(ev.seq)
  }))))))), /*#__PURE__*/React.createElement(Detail, {
    ev: selected,
    now: now,
    onClose: () => setSelectedId(null),
    tzMode: tzMode
  })), /*#__PURE__*/React.createElement("footer", {
    className: "hz-foot"
  }, /*#__PURE__*/React.createElement("span", null, "Forward-design reference concept \xB7 events matched to shipped ", /*#__PURE__*/React.createElement("code", null, "*PredictionResult"), " schemas \xB7 synthetic instances at plausible locations, not live data"), /*#__PURE__*/React.createElement("span", {
    className: "hz-foot-r"
  }, /*#__PURE__*/React.createElement("span", {
    className: "g"
  }, "M"), /*#__PURE__*/React.createElement("span", {
    className: "c"
  }, "ERCURY"), " ", /*#__PURE__*/React.createElement("span", {
    className: "g"
  }, "A"), /*#__PURE__*/React.createElement("span", {
    className: "k"
  }, "GENT"), " \u2671 \xB7 Steel Security Advisors LLC")), /*#__PURE__*/React.createElement("mercury-command-palette", {
    ref: palRef
  }), /*#__PURE__*/React.createElement("mercury-agent-dock", {
    ref: dockRef
  }));
}
ReactDOM.createRoot(document.getElementById("root")).render(/*#__PURE__*/React.createElement(App, null));
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/hazard/Hazard.jsx", error: String((e && e.message) || e) }); }

// ui_kits/hazard/HazardData.js
try { (() => {
/* ============================================================================
   Emergency Operations Console — DATA
   ----------------------------------------------------------------------------
   A grounded corpus of multi-domain hazard events. Every event's field set is
   lifted from the SHIPPED domain prediction-result dataclasses in
   src/omni_mercury_engine/detectors/geological/ (+ energy/):

     EarthquakePredictionResult · HurricanePredictionResult ·
     TsunamiPredictionResult · MeteorPredictionResult · SolarFlarePredictionResult ·
     FloodPredictionResult · TornadoPredictionResult · WildfirePredictionResult ·
     VolcanicPredictionResult · LandslidePredictionResult

   Instances are SYNTHETIC but matched to the real schemas, value ranges and
   units, and seeded at plausible real-world locations (fault zones, ocean
   basins, tornado alley, the ring of fire). Per the honesty rule this is a
   FORWARD-DESIGN reference concept: these dataclasses are computed by the engine
   but are not yet exposed through a single REST endpoint, so the surface runs
   slightly ahead of the public API. Nothing here is live data.
   ============================================================================ */
window.HAZARD = function () {
  // operator watch-floor reference point (for distance / bearing / ETA).
  const REF = {
    lat: 38.0,
    lng: -97.0,
    label: "CONUS WATCH FLOOR · 38.0°N 97.0°W"
  };

  // ---- geo helpers -----------------------------------------------------------
  const R_EARTH = 6371;
  const rad = d => d * Math.PI / 180,
    deg = r => r * 180 / Math.PI;
  function haversine(a, b) {
    const dLat = rad(b.lat - a.lat),
      dLng = rad(b.lng - a.lng);
    const s = Math.sin(dLat / 2) ** 2 + Math.cos(rad(a.lat)) * Math.cos(rad(b.lat)) * Math.sin(dLng / 2) ** 2;
    return Math.round(2 * R_EARTH * Math.asin(Math.sqrt(s)));
  }
  function bearing(a, b) {
    const y = Math.sin(rad(b.lng - a.lng)) * Math.cos(rad(b.lat));
    const x = Math.cos(rad(a.lat)) * Math.sin(rad(b.lat)) - Math.sin(rad(a.lat)) * Math.cos(rad(b.lat)) * Math.cos(rad(b.lng - a.lng));
    return (deg(Math.atan2(y, x)) + 360) % 360;
  }
  const COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"];
  const compass = b => COMPASS[Math.round(b / 22.5) % 16];

  // ---- rng -------------------------------------------------------------------
  let _seed = 0x9e3779b9;
  function rnd() {
    _seed ^= _seed << 13;
    _seed ^= _seed >>> 17;
    _seed ^= _seed << 5;
    return (_seed >>> 0) % 1e6 / 1e6;
  }
  const between = (a, b) => a + rnd() * (b - a);
  const ri = (a, b) => Math.floor(between(a, b + 1));
  const pick = arr => arr[Math.floor(rnd() * arr.length)];
  const chance = p => rnd() < p;
  const jitter = (p, d) => ({
    lat: +(p.lat + between(-d, d)).toFixed(3),
    lng: +(p.lng + between(-d, d)).toFixed(3)
  });
  const r1 = x => Math.round(x * 10) / 10;
  const r2 = x => Math.round(x * 100) / 100;
  const hex = n => {
    let s = "";
    for (let i = 0; i < n; i++) s += "0123456789abcdef"[Math.floor(rnd() * 16)];
    return s;
  };

  // synthesize plausible evac / hazard zones for domains whose schema carries them
  const ZONE_LABEL = {
    hurricane: "Evac zone",
    tsunami: "Coastal zone",
    wildfire: "Evac zone",
    volcanic: "Hazard zone",
    flood: "Flood zone",
    landslide: "Evac zone",
    tornado: "Shelter sector"
  };
  function zonesFor(id, place) {
    if (!(id in ZONE_LABEL)) return [];
    const base = ZONE_LABEL[id],
      n = ri(2, 4),
      letters = "ABCDEFGH",
      out = [];
    for (let i = 0; i < n; i++) out.push(base + " " + letters[i] + ri(1, 4));
    return out;
  }

  // ---- status ladder (shared vocabulary) ------------------------------------
  // sevKey ∈ critical|high|medium|low  → status word per domain ladder
  const STATUS = {
    critical: "CRITICAL",
    high: "WARNING",
    medium: "WATCH",
    low: "ADVISORY"
  };

  // ---- seed regions per domain (plausible real-world placements) -------------
  const SEISMIC = [{
    p: {
      lat: 35.7,
      lng: -117.6
    },
    place: "Eastern California Shear Zone"
  }, {
    p: {
      lat: 61.0,
      lng: -147.7
    },
    place: "Prince William Sound, AK"
  }, {
    p: {
      lat: 36.2,
      lng: 140.9
    },
    place: "Off Ibaraki, Japan"
  }, {
    p: {
      lat: -33.4,
      lng: -70.6
    },
    place: "Valparaíso, Chile"
  }, {
    p: {
      lat: 38.3,
      lng: 38.6
    },
    place: "East Anatolian Fault"
  }, {
    p: {
      lat: 28.2,
      lng: 84.7
    },
    place: "Gandaki, Nepal"
  }];
  const CYCLONE_BASIN = [{
    p: {
      lat: 24.5,
      lng: -84.0
    },
    place: "Gulf of Mexico",
    track: 1
  }, {
    p: {
      lat: 18.0,
      lng: -62.0
    },
    place: "Leeward Islands",
    track: 1
  }, {
    p: {
      lat: 15.2,
      lng: 130.0
    },
    place: "Philippine Sea",
    track: 1
  }, {
    p: {
      lat: 19.0,
      lng: 88.0
    },
    place: "Bay of Bengal",
    track: 1
  }];
  const COAST = [{
    p: {
      lat: 38.3,
      lng: 142.4
    },
    place: "Japan Trench"
  }, {
    p: {
      lat: -8.0,
      lng: 158.0
    },
    place: "Solomon Islands"
  }, {
    p: {
      lat: 44.6,
      lng: -125.0
    },
    place: "Cascadia Subduction Zone"
  }, {
    p: {
      lat: -20.0,
      lng: -71.0
    },
    place: "Northern Chile Trench"
  }];
  const TORNADO_ALLEY = [{
    p: {
      lat: 35.2,
      lng: -97.5
    },
    place: "Central Oklahoma"
  }, {
    p: {
      lat: 37.7,
      lng: -97.3
    },
    place: "South-Central Kansas"
  }, {
    p: {
      lat: 32.8,
      lng: -97.0
    },
    place: "North Texas"
  }, {
    p: {
      lat: 41.3,
      lng: -96.0
    },
    place: "Eastern Nebraska"
  }];
  const WILDLAND = [{
    p: {
      lat: 38.8,
      lng: -120.5
    },
    place: "Sierra Nevada, CA"
  }, {
    p: {
      lat: 34.1,
      lng: -118.3
    },
    place: "Los Angeles Foothills"
  }, {
    p: {
      lat: -37.8,
      lng: 145.0
    },
    place: "Victoria, Australia"
  }, {
    p: {
      lat: 38.0,
      lng: 23.7
    },
    place: "Attica, Greece"
  }];
  const RING_OF_FIRE = [{
    p: {
      lat: 19.4,
      lng: -155.3
    },
    place: "Kīlauea, Hawai‘i"
  }, {
    p: {
      lat: -8.4,
      lng: 116.5
    },
    place: "Rinjani, Indonesia"
  }, {
    p: {
      lat: 46.2,
      lng: -122.2
    },
    place: "Mount St. Helens, WA"
  }, {
    p: {
      lat: 37.7,
      lng: 15.0
    },
    place: "Etna, Sicily"
  }];
  const RIVER_BASIN = [{
    p: {
      lat: 29.95,
      lng: -90.07
    },
    place: "Lower Mississippi"
  }, {
    p: {
      lat: 30.6,
      lng: -96.3
    },
    place: "Brazos River, TX"
  }, {
    p: {
      lat: 25.6,
      lng: 85.1
    },
    place: "Ganges Basin"
  }, {
    p: {
      lat: 45.5,
      lng: -122.7
    },
    place: "Willamette Valley"
  }];
  const SLOPE = [{
    p: {
      lat: 47.5,
      lng: -121.8
    },
    place: "Cascade Foothills, WA"
  }, {
    p: {
      lat: 46.9,
      lng: 9.8
    },
    place: "Graubünden, Switzerland"
  }, {
    p: {
      lat: -7.6,
      lng: 110.4
    },
    place: "Central Java"
  }];
  const GRID = [{
    p: {
      lat: 40.71,
      lng: -74.0
    },
    place: "NYC Metro Grid · NPCC"
  }, {
    p: {
      lat: 41.85,
      lng: -87.65
    },
    place: "Chicago · PJM–MISO seam"
  }, {
    p: {
      lat: 32.78,
      lng: -96.8
    },
    place: "ERCOT North · Dallas"
  }, {
    p: {
      lat: 34.05,
      lng: -118.24
    },
    place: "LA Basin · WECC"
  }, {
    p: {
      lat: 38.9,
      lng: -77.0
    },
    place: "PJM Mid-Atlantic · DC"
  }, {
    p: {
      lat: 47.6,
      lng: -122.33
    },
    place: "BPA Northwest · Seattle"
  }];

  // ---- domain field schemas (label · unit · format) -------------------------
  // 'k' is the real dataclass field name; surfaces render label+unit.
  const num = (x, u) => x == null ? "—" : (Math.round(x * 10) / 10).toLocaleString() + (u ? " " + u : "");
  const int = (x, u) => x == null ? "—" : Math.round(x).toLocaleString() + (u ? " " + u : "");
  const bool = x => x ? "yes" : "no";
  const DOMAINS = {
    earthquake: {
      label: "Earthquake",
      glyph: "⊕",
      short: "EQ",
      kind: "impact",
      dataclass: "EarthquakePredictionResult",
      lifecycle: "rupture → S-wave arrival",
      seeds: SEISMIC,
      headline: f => "M" + f.estimated_magnitude.toFixed(1) + " · " + f.magnitude_class,
      fields: [["estimated_magnitude", "Magnitude", v => "M" + v.toFixed(1)], ["magnitude_class", "Class", v => v], ["depth_km", "Focal depth", v => num(v, "km")], ["epicenter_distance_km", "Epicenter dist", v => int(v, "km")], ["p_wave_arrival_time", "P-wave", v => num(v, "s")], ["s_wave_arrival_time", "S-wave (impact)", v => num(v, "s")], ["aftershock_probability", "Aftershock prob", v => Math.round(v * 100) + "%"]],
      slider: {
        key: "estimated_magnitude",
        label: "Magnitude",
        min: 0,
        max: 9,
        step: 0.1,
        fmt: v => "M" + v.toFixed(1)
      }
    },
    hurricane: {
      label: "Hurricane",
      glyph: "❍",
      short: "HU",
      kind: "track",
      dataclass: "HurricanePredictionResult",
      lifecycle: "genesis → landfall",
      seeds: CYCLONE_BASIN,
      headline: f => "Cat " + f.category + " · " + f.max_wind_speed_kt + " kt",
      fields: [["category", "Category", v => "Cat " + v], ["max_wind_speed_kt", "Max wind", v => int(v, "kt")], ["min_pressure_mb", "Min pressure", v => int(v, "mb")], ["eye_diameter_nm", "Eye diameter", v => num(v, "nm")], ["storm_surge_risk", "Surge risk", v => v], ["rapid_intensification", "Rapid intensif.", bool], ["time_to_landfall_hours", "Time to landfall", v => num(v, "h")], ["landfall_probability", "Landfall prob", v => Math.round(v * 100) + "%"]],
      slider: {
        key: "max_wind_speed_kt",
        label: "Max wind",
        min: 0,
        max: 175,
        step: 5,
        fmt: v => v + " kt"
      }
    },
    tsunami: {
      label: "Tsunami",
      glyph: "≋",
      short: "TS",
      kind: "impact",
      dataclass: "TsunamiPredictionResult",
      lifecycle: "source quake → wave arrival",
      seeds: COAST,
      headline: f => f.estimated_wave_height_m.toFixed(1) + " m wave · " + f.severity,
      fields: [["estimated_wave_height_m", "Wave height", v => num(v, "m")], ["arrival_time_minutes", "Arrival (impact)", v => num(v, "min")], ["source_distance_km", "Source dist", v => int(v, "km")], ["source_magnitude", "Source quake", v => "M" + v.toFixed(1)], ["waveform_anomaly_score", "Waveform anom", v => v.toFixed(2)]],
      slider: {
        key: "estimated_wave_height_m",
        label: "Wave height",
        min: 0,
        max: 12,
        step: 0.5,
        fmt: v => v + " m"
      }
    },
    meteor: {
      label: "Meteor / NEO",
      glyph: "✦",
      short: "NEO",
      kind: "approach",
      dataclass: "MeteorPredictionResult",
      lifecycle: "detection → closest approach",
      seeds: null,
      headline: f => num(f.estimated_size_m, "m") + " · " + f.threat_level,
      fields: [["estimated_size_m", "Diameter", v => num(v, "m")], ["estimated_velocity_kms", "Velocity", v => num(v, "km/s")], ["impact_probability", "Impact prob", v => (v * 100).toFixed(2) + "%"], ["time_to_closest_approach_hours", "Closest approach", v => num(v, "h")], ["trajectory_confidence", "Trajectory conf", v => Math.round(v * 100) + "%"], ["threat_level", "Torino-style", v => v]],
      slider: {
        key: "estimated_size_m",
        label: "Diameter",
        min: 0,
        max: 400,
        step: 5,
        fmt: v => v + " m"
      }
    },
    solar: {
      label: "Solar Flare",
      glyph: "☀",
      short: "SF",
      kind: "global",
      dataclass: "SolarFlarePredictionResult",
      lifecycle: "flare onset → geomagnetic peak",
      seeds: null,
      headline: f => f.flare_class + " · Kp " + f.kp_index_predicted.toFixed(0),
      fields: [["flare_class", "Flare class", v => v], ["x_ray_flux", "X-ray flux", v => v.toExponential(1) + " W/m²"], ["kp_index_predicted", "Kp index", v => v.toFixed(0)], ["geomagnetic_storm_probability", "Geo-storm prob", v => Math.round(v * 100) + "%"], ["proton_flux", "Proton flux", v => int(v, "pfu")]],
      slider: {
        key: "kp_index_predicted",
        label: "Kp index",
        min: 0,
        max: 9,
        step: 1,
        fmt: v => "Kp " + v
      }
    },
    flood: {
      label: "Flood",
      glyph: "⊓",
      short: "FL",
      kind: "onset",
      dataclass: "FloodPredictionResult",
      lifecycle: "rainfall → crest",
      seeds: RIVER_BASIN,
      headline: f => f.flood_type + " · " + f.severity,
      fields: [["flood_type", "Type", v => v], ["river_stage_ft", "River stage", v => num(v, "ft")], ["flood_stage_ft", "Flood stage", v => num(v, "ft")], ["time_to_peak_hours", "Time to crest", v => num(v, "h")], ["precipitation_24h_inches", "Rain 24h", v => num(v, "in")], ["population_at_risk", "Pop. at risk", v => int(v)], ["affected_area_sq_mi", "Area", v => int(v, "sq mi")]],
      slider: {
        key: "river_stage_ft",
        label: "River stage",
        min: 0,
        max: 60,
        step: 1,
        fmt: v => v + " ft"
      }
    },
    tornado: {
      label: "Tornado",
      glyph: "𖣘",
      short: "TO",
      kind: "impact",
      dataclass: "TornadoPredictionResult",
      lifecycle: "mesocyclone → touchdown",
      seeds: TORNADO_ALLEY,
      headline: f => f.estimated_intensity + " · " + f.threat_level,
      fields: [["estimated_intensity", "Intensity", v => v], ["mesocyclone_detected", "Mesocyclone", bool], ["rotation_velocity_ms", "Rotation vel", v => num(v, "m/s")], ["cape_value", "CAPE", v => int(v, "J/kg")], ["helicity_value", "SRH", v => int(v, "m²/s²")], ["time_to_touchdown_minutes", "To touchdown", v => num(v, "min")]],
      slider: {
        key: "rotation_velocity_ms",
        label: "Rotation",
        min: 0,
        max: 140,
        step: 5,
        fmt: v => v + " m/s"
      }
    },
    wildfire: {
      label: "Wildfire",
      glyph: "▲",
      short: "WF",
      kind: "duration",
      dataclass: "WildfirePredictionResult",
      lifecycle: "ignition → containment",
      seeds: WILDLAND,
      headline: f => num(f.fire_perimeter_km2, "km²") + " · " + f.risk_level,
      fields: [["risk_level", "Risk", v => v], ["fire_perimeter_km2", "Perimeter", v => num(v, "km²")], ["spread_rate_km_hr", "Spread rate", v => num(v, "km/h")], ["spread_direction_deg", "Spread dir", v => compass(v) + " (" + Math.round(v) + "°)"], ["thermal_hotspots", "Hotspots", v => int(v)], ["fuel_moisture", "Fuel moisture", v => Math.round(v * 100) + "%"]],
      slider: {
        key: "fire_perimeter_km2",
        label: "Perimeter",
        min: 0,
        max: 1200,
        step: 10,
        fmt: v => v + " km²"
      }
    },
    volcanic: {
      label: "Volcanic",
      glyph: "◭",
      short: "VO",
      kind: "approach",
      dataclass: "VolcanicPredictionResult",
      lifecycle: "unrest → eruption",
      seeds: RING_OF_FIRE,
      headline: f => "VEI " + f.vei_estimate + " · " + f.alert_level,
      fields: [["alert_level", "Alert", v => v], ["eruption_type", "Type", v => v], ["vei_estimate", "VEI est.", v => "VEI " + v], ["time_to_eruption_hours", "To eruption", v => num(v, "h")], ["seismic_swarm_detected", "Seismic swarm", bool], ["deformation_detected", "Deformation", bool], ["lahar_risk", "Lahar risk", v => v]],
      slider: {
        key: "vei_estimate",
        label: "VEI",
        min: 0,
        max: 7,
        step: 1,
        fmt: v => "VEI " + v
      }
    },
    landslide: {
      label: "Landslide",
      glyph: "◣",
      short: "LS",
      kind: "approach",
      dataclass: "LandslidePredictionResult",
      lifecycle: "creep → slope failure",
      seeds: SLOPE,
      headline: f => f.landslide_type + " · " + f.risk_level,
      fields: [["risk_level", "Risk", v => v], ["landslide_type", "Type", v => v], ["slope_failure_probability", "Failure prob", v => Math.round(v * 100) + "%"], ["time_to_failure_hours", "To failure", v => num(v, "h")], ["slope_angle_deg", "Slope angle", v => num(v, "°")], ["displacement_rate_mm_day", "Displacement", v => num(v, "mm/d")], ["runout_distance_km", "Runout", v => num(v, "km")]],
      slider: {
        key: "slope_failure_probability",
        label: "Failure prob",
        min: 0,
        max: 1,
        step: 0.05,
        fmt: v => Math.round(v * 100) + "%"
      }
    },
    energy: {
      label: "Energy / EMP",
      glyph: "⦿",
      short: "EN",
      kind: "duration",
      dataclass: "EMPPredictionResult",
      lifecycle: "pulse → grid recovery",
      seeds: GRID,
      headline: f => f.emp_type + " · grid " + f.grid_impact_assessment,
      fields: [["emp_type", "Event type", v => v], ["grid_impact_assessment", "Grid impact", v => v], ["field_strength_vm", "Field strength", v => int(v, "V/m")], ["frequency_mhz", "Peak freq", v => num(v, "MHz")], ["pulse_duration_ns", "Pulse dur", v => num(v, "ns")], ["e1_component_detected", "E1 (prompt)", bool], ["e3_component_detected", "E3 (GIC)", bool], ["intentional_attack_probability", "Intentional", v => Math.round(v * 100) + "%"]],
      slider: {
        key: "field_strength_vm",
        label: "Field strength",
        min: 0,
        max: 50000,
        step: 1000,
        fmt: v => (v / 1000).toFixed(0) + "k V/m"
      }
    }
  };
  const DOMAIN_IDS = Object.keys(DOMAINS);

  // ---- per-domain field generators (matched to real schemas) ----------------
  function genFields(id, sevKey, etaH) {
    const sevMul = {
      critical: 1,
      high: 0.78,
      medium: 0.55,
      low: 0.32
    }[sevKey];
    switch (id) {
      case "earthquake":
        {
          const mag = r1(between(4.0, 5.0) + sevMul * between(2.0, 3.2));
          const cls = mag >= 7 ? "major" : mag >= 6 ? "strong" : mag >= 5 ? "moderate" : "light";
          const depth = r1(between(5, 70));
          const sdist = ri(40, 900);
          const sWave = r1(sdist / 4.2);
          const pWave = r1(sdist / 7.0);
          return {
            earthquake_detected: true,
            confidence: r2(between(0.8, 0.99)),
            estimated_magnitude: mag,
            magnitude_class: cls,
            p_wave_detected: true,
            s_wave_detected: true,
            p_wave_arrival_time: pWave,
            s_wave_arrival_time: sWave,
            epicenter_distance_km: sdist,
            depth_km: depth,
            resonance_score: r2(between(0.4, 0.95)),
            aftershock_probability: r2(between(0.2, 0.85) * sevMul + 0.1)
          };
        }
      case "hurricane":
        {
          const cat = Math.max(1, Math.min(5, Math.round(1 + sevMul * 4)));
          const wind = ri(64 + (cat - 1) * 20, 64 + cat * 22);
          const press = ri(1010 - cat * 16, 1010 - (cat - 1) * 16);
          return {
            cyclone_detected: true,
            confidence: r2(between(0.82, 0.98)),
            category: cat,
            cyclone_type: "tropical cyclone",
            max_wind_speed_kt: wind,
            min_pressure_mb: press,
            eye_diameter_nm: r1(between(12, 40)),
            storm_surge_risk: cat >= 4 ? "extreme" : cat >= 3 ? "high" : cat >= 2 ? "moderate" : "low",
            rapid_intensification: chance(0.3 * sevMul + 0.1),
            intensification_rate_kt_24h: ri(10, 45),
            time_to_landfall_hours: etaH,
            landfall_probability: r2(between(0.5, 0.95))
          };
        }
      case "tsunami":
        {
          const srcMag = r1(between(7.0, 7.4) + sevMul * between(0.4, 1.8));
          const sdist = ri(120, 1400);
          const arr = r1(sdist / 720 * 60);
          return {
            tsunami_detected: true,
            confidence: r2(between(0.78, 0.97)),
            severity: STATUS[sevKey].toLowerCase(),
            estimated_wave_height_m: r1(between(0.6, 1.5) + sevMul * between(1.5, 9)),
            arrival_time_minutes: arr,
            source_distance_km: sdist,
            source_magnitude: srcMag,
            resonance_score: r2(between(0.5, 0.95)),
            waveform_anomaly_score: r2(between(0.5, 0.98))
          };
        }
      case "meteor":
        {
          const size = r1(between(8, 30) + sevMul * between(40, 320));
          const vel = r1(between(11, 32));
          return {
            meteor_detected: true,
            confidence: r2(between(0.7, 0.95)),
            threat_level: sevKey === "critical" ? "elevated" : sevKey === "high" ? "notable" : "routine",
            estimated_size_m: size,
            estimated_velocity_kms: vel,
            impact_probability: r2(between(0.0001, 0.02) * sevMul),
            optical_detection: true,
            radar_detection: chance(0.6),
            bayesian_posterior: r2(between(0.3, 0.9)),
            trajectory_confidence: r2(between(0.6, 0.97)),
            time_to_closest_approach_hours: etaH
          };
        }
      case "solar":
        {
          const cls = pick(sevKey === "critical" ? ["X4.2", "X8.1", "X2.7"] : sevKey === "high" ? ["M9.3", "X1.1", "M6.4"] : ["C8.2", "M2.1", "M1.4"]);
          const kp = Math.min(9, Math.round(4 + sevMul * 5));
          return {
            flare_detected: true,
            confidence: r2(between(0.75, 0.96)),
            flare_class: cls,
            x_ray_flux: between(1e-6, 9e-4) * (1 + sevMul * 4),
            proton_flux: ri(1, 400) * (sevMul > 0.7 ? 8 : 1),
            geomagnetic_storm_probability: r2(between(0.3, 0.95) * sevMul + 0.05),
            kp_index_predicted: kp,
            dst_index_predicted: -ri(20, 60) * (1 + sevMul),
            hmm_state: ri(0, 4),
            affected_systems: ["HF radio", "GNSS accuracy", "power grids", "satellite ops"].slice(0, ri(1, 4))
          };
        }
      case "flood":
        {
          const fstage = r1(between(18, 28));
          const rstage = r1(fstage + (sevMul - 0.4) * between(2, 16));
          return {
            flood_likely: true,
            confidence: r2(between(0.78, 0.97)),
            severity: STATUS[sevKey].toLowerCase(),
            flood_type: pick(["riverine", "flash", "urban", "coastal"]),
            river_stage_ft: rstage,
            flood_stage_ft: fstage,
            stage_trend: rstage > fstage ? "rising" : "steady",
            precipitation_24h_inches: r1(between(1, 4) + sevMul * 6),
            precipitation_forecast_inches: r1(between(0.5, 5)),
            soil_saturation_pct: ri(50, 99),
            time_to_peak_hours: etaH,
            peak_discharge_cfs: ri(8000, 90000),
            affected_area_sq_mi: ri(5, 320),
            population_at_risk: ri(800, 240000)
          };
        }
      case "tornado":
        {
          const ef = pick(sevKey === "critical" ? ["EF4", "EF5"] : sevKey === "high" ? ["EF2", "EF3"] : ["EF0", "EF1"]);
          return {
            tornado_likely: true,
            confidence: r2(between(0.7, 0.95)),
            threat_level: STATUS[sevKey],
            estimated_intensity: ef,
            mesocyclone_detected: sevMul > 0.5,
            rotation_velocity_ms: ri(25, 60 + sevMul * 70),
            pressure_drop_mb: ri(20, 100),
            cape_value: ri(1500, 5500),
            helicity_value: ri(150, 600),
            wind_shear_detected: true,
            time_to_touchdown_minutes: etaH * 60
          };
        }
      case "wildfire":
        {
          const dir = ri(0, 359);
          return {
            fire_detected: true,
            confidence: r2(between(0.75, 0.96)),
            risk_level: sevKey === "critical" ? "extreme" : sevKey === "high" ? "very high" : sevKey === "medium" ? "high" : "moderate",
            fire_perimeter_km2: r1(between(2, 40) + sevMul * between(40, 1100)),
            spread_rate_km_hr: r1(between(0.5, 2) + sevMul * 8),
            spread_direction_deg: dir,
            thermal_hotspots: ri(3, 60),
            smoke_detected: true,
            fuel_moisture: r2(between(0.03, 0.18))
          };
        }
      case "volcanic":
        {
          const vei = Math.max(1, Math.min(6, Math.round(1 + sevMul * 5)));
          return {
            eruption_imminent: sevMul > 0.6,
            confidence: r2(between(0.7, 0.95)),
            alert_level: vei >= 5 ? "red" : vei >= 3 ? "orange" : "yellow",
            eruption_type: pick(["effusive", "explosive", "phreatic", "strombolian"]),
            time_to_eruption_hours: etaH,
            vei_estimate: vei,
            seismic_swarm_detected: sevMul > 0.4,
            thermal_anomaly_detected: chance(0.7),
            gas_flux_anomaly: chance(0.6),
            deformation_detected: sevMul > 0.5,
            lahar_risk: vei >= 4 ? "high" : vei >= 2 ? "moderate" : "low"
          };
        }
      case "landslide":
        {
          return {
            landslide_imminent: sevMul > 0.6,
            confidence: r2(between(0.7, 0.94)),
            risk_level: sevKey === "critical" ? "very high" : sevKey === "high" ? "high" : sevKey === "medium" ? "moderate" : "low",
            landslide_type: pick(["debris flow", "rockfall", "rotational slump", "earthflow"]),
            slope_failure_probability: r2(between(0.3, 0.7) + sevMul * 0.3),
            time_to_failure_hours: etaH,
            rainfall_trigger: chance(0.7),
            seismic_trigger: chance(0.3),
            soil_saturation_pct: ri(60, 98),
            slope_angle_deg: r1(between(22, 52)),
            displacement_rate_mm_day: r1(between(1, 40) * sevMul),
            affected_area_km2: r1(between(0.1, 8)),
            runout_distance_km: r1(between(0.2, 6))
          };
        }
      case "energy":
        {
          const empType = pick(["nuclear EMP (HEMP)", "non-nuclear (NNEMP)", "geomagnetic (GIC)", "solar-driven surge"]);
          const e1 = sevMul > 0.55,
            e3 = sevMul > 0.4;
          const intentional = empType.indexOf("EMP") >= 0 || empType.indexOf("NNEMP") >= 0 ? between(0.25, 0.6) : between(0.02, 0.15);
          return {
            emp_detected: true,
            confidence: r2(between(0.74, 0.96)),
            emp_type: empType,
            threat_level: STATUS[sevKey],
            e1_component_detected: e1,
            e2_component_detected: chance(0.5),
            e3_component_detected: e3,
            field_strength_vm: e1 ? ri(12000, 50000) : ri(200, 9000),
            frequency_mhz: r1(between(0.1, 800)),
            pulse_duration_ns: e1 ? r1(between(2, 60)) : r1(between(120, 5000)),
            grid_impact_assessment: sevKey === "critical" ? "critical" : sevKey === "high" ? "high" : sevKey === "medium" ? "moderate" : "low",
            affected_infrastructure: ["transmission", "SCADA", "transformers", "comms", "pipelines"].slice(0, ri(2, 5)),
            intentional_attack_probability: r2(intentional)
          };
        }
    }
  }

  // domains whose primary timer is "to impact/approach" (drives ETA + inbound)
  const ETA_KEY = {
    earthquake: "s_wave_arrival_time",
    tsunami: "arrival_time_minutes",
    hurricane: "time_to_landfall_hours",
    meteor: "time_to_closest_approach_hours",
    flood: "time_to_peak_hours",
    tornado: "time_to_touchdown_minutes",
    volcanic: "time_to_eruption_hours",
    landslide: "time_to_failure_hours",
    wildfire: null,
    solar: null,
    energy: null
  };
  // convert each domain's native impact-time field to seconds-from-now
  function etaSeconds(id, f) {
    const v = ETA_KEY[id] ? f[ETA_KEY[id]] : null;
    if (v == null) return null;
    if (id === "earthquake") return v; // seconds
    if (id === "tsunami" || id === "tornado") return v * 60; // minutes
    return v * 3600; // hours
  }
  function regionOf(lat, lng) {
    if (lat > 12 && lat < 72 && lng > -170 && lng < -50) return "North America";
    if (lat <= 12 && lat > -57 && lng > -90 && lng < -32) return "South America";
    if (lat > 34 && lng > -12 && lng < 45) return "Europe";
    if (lat <= 37 && lat > -36 && lng > -20 && lng < 52) return "Africa";
    if (lat > -12 && lng >= 45 && lng < 150) return "Asia";
    if (lat <= 0 && lng > 110) return "Oceania";
    if (lng >= 95 || lng < -150) return "Asia–Pacific";
    return "Open Ocean";
  }

  // ---- build one event -------------------------------------------------------
  function makeEvent(seq, now) {
    const id = pick(DOMAIN_IDS);
    const D = DOMAINS[id];
    const sevKey = pick(["critical", "high", "high", "medium", "medium", "medium", "low", "low"]);
    // location
    let loc, place;
    if (D.seeds) {
      const s = pick(D.seeds);
      loc = jitter(s.p, 2.2);
      place = s.place;
    } else if (id === "meteor") {
      loc = {
        lat: +between(-50, 60).toFixed(2),
        lng: +between(-180, 180).toFixed(2)
      };
      place = "Near-Earth approach corridor";
    } else {
      loc = {
        lat: +between(20, 45).toFixed(2),
        lng: +between(-120, 30).toFixed(2)
      };
      place = "Geostationary-facing hemisphere";
    }
    const region = id === "solar" ? "Geospace" : regionOf(loc.lat, loc.lng);
    // lifecycle timing
    const etaH = +between(0.3, 48).toFixed(1);
    const f = genFields(id, sevKey, etaH);
    const etaS = etaSeconds(id, f);
    const inception = now - ri(5, 280) * 60 * 1000; // detected minutes ago
    const impact = etaS != null ? now + etaS * 1000 : null;
    const ends = D.kind === "duration" ? now + ri(6, 96) * 3600 * 1000 : impact;
    const durationS = ends ? (ends - now) / 1000 : null;
    const inbound = etaS != null;
    const clockS = inbound ? etaS : durationS; // unified live clock

    const dist = haversine(REF, loc);
    const brg = bearing(REF, loc);

    // track forecast (hurricane / meteor ground-track) — list of [lng,lat]
    let track = null;
    if (id === "hurricane" || id === "meteor") {
      track = [];
      const step = id === "hurricane" ? 2.2 : 6;
      let p = {
        lat: loc.lat,
        lng: loc.lng
      };
      const dirLat = id === "hurricane" ? between(0.5, 1) : between(-0.6, 0.6);
      const dirLng = id === "hurricane" ? -between(0.4, 1) : between(0.6, 1);
      for (let i = 0; i < 5; i++) {
        track.push([+p.lng.toFixed(2), +p.lat.toFixed(2)]);
        p = {
          lat: p.lat + dirLat * step * 0.6,
          lng: p.lng + dirLng * step
        };
      }
    }
    // wildfire spread vector
    let spread = null;
    if (id === "wildfire") spread = {
      deg: f.spread_direction_deg,
      km: (f.spread_rate_km_hr || 1) * 6
    };

    // ethics gate (reuse honesty model) — high-consequence withholds for review
    const withheld = sevKey === "critical" && chance(0.16);
    const benev = withheld ? r2(between(0.962, 0.9898)) : r2(between(0.991, 0.999));
    const tlp = withheld ? "TLP:RED" : sevKey === "critical" ? "TLP:AMBER+STRICT" : sevKey === "high" ? "TLP:AMBER" : "TLP:GREEN";
    const evac = zonesFor(id, place);
    const status = withheld ? "WITHHELD" : STATUS[sevKey];
    return {
      seq: 4200 + seq,
      id,
      domain: D.label,
      glyph: D.glyph,
      short: D.short,
      kind: D.kind,
      dataclass: D.dataclass,
      lifecycle: D.lifecycle,
      place,
      region,
      lat: loc.lat,
      lng: loc.lng,
      sevKey,
      status,
      confidence: f.confidence,
      headline: D.headline(f),
      fields: f,
      inception,
      now,
      impact,
      ends,
      etaS,
      clockS,
      dist,
      bearing: brg,
      compass: compass(brg),
      inbound,
      track,
      spread,
      withheld,
      benev,
      tlp,
      corr: "haz_" + hex(8) + "-" + hex(4),
      warnings: f.warning_actions || f.early_warning_actions || f.warning_actions || [],
      evac: Array.isArray(evac) ? evac : []
    };
  }

  // The ethical-gate WITHHELD state must be represented on the floor. With the
  // fixed seed the per-event 16% withhold roll can produce a corpus with ZERO
  // withholds — leaving the "Withheld by gate" preset, the withheld summary stat,
  // the WITHHELD status and TLP:RED entirely unexercised. Deterministically
  // promote the most-confident in-range critical events so the gate state is
  // always present and the preset always returns rows.
  function ensureWithheld(list, minN) {
    minN = minN || 3;
    let have = list.filter(e => e.withheld).length;
    if (have >= minN) return;
    // prefer critical events inside the default range filters (≤8000 km and
    // impact ≤72h, or no clock) so the default-state "withheld" preset shows them
    let cand = list.filter(e => !e.withheld && e.sevKey === "critical" && e.dist <= 8000 && (e.etaS == null || e.etaS / 3600 <= 72)).sort((a, b) => b.confidence - a.confidence);
    if (cand.length < minN - have) {
      const extra = list.filter(e => !e.withheld && e.sevKey === "critical" && cand.indexOf(e) < 0).sort((a, b) => b.confidence - a.confidence);
      cand = cand.concat(extra);
    }
    const benevVals = [0.987, 0.978, 0.984, 0.971, 0.989];
    for (let i = 0; have < minN && i < cand.length; i++, have++) {
      const e = cand[i];
      e.withheld = true;
      e.benev = benevVals[i % benevVals.length];
      e.status = "WITHHELD";
      e.tlp = "TLP:RED";
    }
  }
  function buildCorpus(n) {
    _seed = 0x51ed270b;
    const now = Date.now();
    const out = [];
    for (let i = 0; i < n; i++) out.push(makeEvent(i, now));
    ensureWithheld(out, 3);
    return out;
  }

  // schema-defined field accessor for the detail panel
  function fieldRows(ev) {
    const D = DOMAINS[ev.id];
    return D.fields.map(([k, label, fmt]) => ({
      label,
      value: ev.fields[k] == null ? "—" : fmt(ev.fields[k])
    })).filter(r => r.value !== "—");
  }
  return {
    REF,
    DOMAINS,
    DOMAIN_IDS,
    buildCorpus,
    fieldRows,
    haversine,
    bearing,
    compass,
    regionOf
  };
}();
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/hazard/HazardData.js", error: String((e && e.message) || e) }); }

// ui_kits/hazard/HazardMap.jsx
try { (() => {
/* Emergency Operations Console — MAP (upgraded)
   A self-contained equirectangular plot (no external tiles / network). Now:
     • EXPANDABLE — a fullscreen toggle blows the plot up to a focus overlay
     • ZOOM + PAN — wheel to zoom, drag to pan, +/−/reset controls, zoom readout
     • SELECTABLE — click a marker to select an event; click a region to filter to
       it; hover shows a live tooltip with the event's headline + ETA
     • STRATEGIC LAYERS — toggle graticule, region labels, range rings, forecast
       tracks, the bearing line and the watch-floor independently
   viewBox stays 0 0 360 180 (x=lng+180, y=90−lat); zoom/pan ride on a <g>
   transform so projection math is unchanged and correct. */
const {
  useState: useMapState,
  useRef: useMapRef,
  useEffect: useMapEffect,
  useCallback: useMapCb
} = React;
const {
  REF,
  regionOf
} = window.HAZARD;
const PX = lng => lng + 180;
const PY = lat => 90 - lat;
const LAND = [[[-168, 65], [-160, 71], [-130, 70], [-95, 72], [-82, 73], [-60, 83], [-75, 68], [-64, 60], [-56, 52], [-66, 44], [-70, 42], [-76, 35], [-81, 25], [-97, 18], [-105, 23], [-115, 30], [-124, 40], [-125, 48], [-133, 54], [-150, 59], [-168, 65]], [[-46, 60], [-30, 68], [-22, 70], [-20, 76], [-30, 83], [-55, 83], [-58, 76], [-52, 64], [-46, 60]], [[-81, 8], [-70, 11], [-60, 10], [-50, 0], [-35, -5], [-35, -12], [-48, -25], [-58, -35], [-65, -43], [-72, -52], [-75, -50], [-71, -38], [-71, -25], [-76, -14], [-81, -4], [-81, 8]], [[-10, 36], [-9, 44], [-4, 48], [2, 51], [-2, 58], [8, 63], [18, 69], [28, 70], [30, 60], [28, 52], [24, 46], [18, 40], [8, 38], [-2, 36], [-10, 36]], [[-16, 15], [-16, 28], [-8, 36], [10, 37], [24, 33], [33, 31], [43, 12], [51, 12], [41, -2], [40, -15], [33, -26], [25, -34], [18, -35], [12, -17], [8, 4], [-8, 5], [-16, 15]], [[28, 52], [40, 48], [50, 52], [60, 55], [75, 55], [90, 52], [100, 55], [120, 53], [135, 55], [145, 48], [142, 45], [130, 42], [126, 35], [122, 30], [120, 22], [108, 21], [98, 10], [92, 20], [80, 8], [72, 20], [60, 25], [52, 29], [44, 38], [36, 42], [28, 46], [28, 52]], [[95, 5], [105, 2], [118, 5], [128, 2], [120, -4], [106, -7], [98, -2], [95, 5]], [[114, -22], [122, -18], [130, -12], [142, -11], [150, -22], [153, -28], [148, -38], [138, -35], [129, -32], [120, -34], [115, -30], [114, -22]], [[130, 31], [136, 35], [140, 37], [142, 43], [145, 44], [140, 40], [138, 35], [133, 33], [130, 31]], [[-6, 50], [-2, 52], [-3, 58], [-8, 57], [-10, 53], [-6, 50]], [[166, -46], [170, -44], [174, -41], [178, -38], [173, -42], [168, -46], [166, -46]]];

// clickable region boxes (lng/lat bounds) → the region name used by the corpus
const REGION_BOX = [{
  id: "North America",
  x0: -168,
  x1: -52,
  y0: 12,
  y1: 72
}, {
  id: "South America",
  x0: -82,
  x1: -34,
  y0: -56,
  y1: 12
}, {
  id: "Europe",
  x0: -12,
  x1: 45,
  y0: 36,
  y1: 72
}, {
  id: "Africa",
  x0: -18,
  x1: 52,
  y0: -36,
  y1: 36
}, {
  id: "Asia",
  x0: 45,
  x1: 150,
  y0: 5,
  y1: 72
}, {
  id: "Oceania",
  x0: 110,
  x1: 180,
  y0: -50,
  y1: 5
}];
const REGION_LABELS = [["N. AMERICA", -100, 42], ["S. AMERICA", -60, -18], ["EUROPE", 12, 52], ["AFRICA", 20, 8], ["ASIA", 90, 40], ["OCEANIA", 134, -25]];
function Graticule() {
  const lines = [];
  for (let lng = -180; lng <= 180; lng += 30) lines.push(/*#__PURE__*/React.createElement("line", {
    key: "v" + lng,
    x1: PX(lng),
    y1: 0,
    x2: PX(lng),
    y2: 180,
    className: lng === 0 ? "hm-grat hm-grat0" : "hm-grat"
  }));
  for (let lat = -60; lat <= 60; lat += 30) lines.push(/*#__PURE__*/React.createElement("line", {
    key: "h" + lat,
    x1: 0,
    y1: PY(lat),
    x2: 360,
    y2: PY(lat),
    className: lat === 0 ? "hm-grat hm-grat0" : "hm-grat"
  }));
  return /*#__PURE__*/React.createElement("g", null, lines);
}
const STATUS_HUE = {
  CRITICAL: "var(--anomaly)",
  WARNING: "var(--warning)",
  WATCH: "var(--mercury-400)",
  ADVISORY: "var(--normal)",
  WITHHELD: "var(--withheld)"
};
function rangeRings(ev) {
  const rings = [500, 1500, 3000];
  const cx = PX(ev.lng),
    cy = PY(ev.lat);
  const latComp = Math.max(0.3, Math.cos(ev.lat * Math.PI / 180));
  return rings.map((km, i) => /*#__PURE__*/React.createElement("ellipse", {
    key: i,
    cx: cx,
    cy: cy,
    rx: km / 111 / latComp,
    ry: km / 111,
    className: "hm-ring"
  }));
}
const LAYER_DEFS = [{
  id: "graticule",
  label: "Graticule",
  on: true
}, {
  id: "labels",
  label: "Region labels",
  on: true
}, {
  id: "regions",
  label: "Region select",
  on: true
}, {
  id: "rings",
  label: "Range rings",
  on: true
}, {
  id: "tracks",
  label: "Forecast tracks",
  on: true
}, {
  id: "bearing",
  label: "Bearing line",
  on: true
}, {
  id: "floor",
  label: "Watch floor",
  on: true
}];
function HazardMap({
  events,
  selectedId,
  onSelect,
  sevScale,
  regionFilter,
  onRegionSelect
}) {
  const [expanded, setExpanded] = useMapState(false);
  const [view, setView] = useMapState({
    z: 1,
    x: 0,
    y: 0
  }); // zoom + pan (viewBox units)
  const [layers, setLayers] = useMapState(() => {
    const o = {};
    LAYER_DEFS.forEach(l => o[l.id] = l.on);
    return o;
  });
  const [hover, setHover] = useMapState(null); // {ev, px, py}
  const [layerMenu, setLayerMenu] = useMapState(false);
  const svgRef = useMapRef(null);
  const drag = useMapRef(null);
  const refX = PX(REF.lng),
    refY = PY(REF.lat);
  const sel = events.find(e => e.seq === selectedId);

  // zoom helpers — clamp and keep within bounds
  const clampView = useMapCb(v => {
    const z = Math.max(1, Math.min(8, v.z));
    const halfW = 180 / z,
      halfH = 90 / z;
    const x = Math.max(-(180 - halfW * 2), Math.min(180 - halfW * 2, v.x));
    const y = Math.max(-(90 - halfH * 2), Math.min(90 - halfH * 2, v.y));
    return {
      z,
      x,
      y
    };
  }, []);
  const zoomBy = f => setView(v => clampView({
    z: v.z * f,
    x: v.x,
    y: v.y
  }));
  const reset = () => setView({
    z: 1,
    x: 0,
    y: 0
  });

  // wheel zoom toward cursor
  useMapEffect(() => {
    const el = svgRef.current;
    if (!el) return;
    const onWheel = e => {
      e.preventDefault();
      const f = e.deltaY < 0 ? 1.18 : 1 / 1.18;
      setView(v => clampView({
        z: v.z * f,
        x: v.x,
        y: v.y
      }));
    };
    el.addEventListener("wheel", onWheel, {
      passive: false
    });
    return () => el.removeEventListener("wheel", onWheel);
  }, [clampView]);

  // drag to pan
  const onDown = e => {
    drag.current = {
      sx: e.clientX,
      sy: e.clientY,
      ox: view.x,
      oy: view.y,
      moved: false
    };
  };
  const onMove = e => {
    if (!drag.current) return;
    const el = svgRef.current;
    const rect = el.getBoundingClientRect();
    const unitsPerPxX = 360 / view.z / rect.width,
      unitsPerPxY = 180 / view.z / rect.height;
    const dx = (e.clientX - drag.current.sx) * unitsPerPxX,
      dy = (e.clientY - drag.current.sy) * unitsPerPxY;
    if (Math.abs(e.clientX - drag.current.sx) + Math.abs(e.clientY - drag.current.sy) > 3) drag.current.moved = true;
    setView(v => clampView({
      z: v.z,
      x: drag.current.ox - dx,
      y: drag.current.oy - dy
    }));
  };
  const onUp = () => {
    setTimeout(() => {
      drag.current = null;
    }, 0);
  };

  // current viewBox from zoom+pan, centered on the world then offset
  const w = 360 / view.z,
    h = 180 / view.z;
  const vbX = 180 - w / 2 + view.x,
    vbY = 90 - h / 2 + view.y;
  const viewBox = `${vbX} ${vbY} ${w} ${h}`;
  const k = 1 / view.z; // scale factor for stroke/marker sizing so they stay constant on screen

  const toggleLayer = id => setLayers(p => Object.assign({}, p, {
    [id]: !p[id]
  }));
  function markerHover(ev, on) {
    if (!on) {
      setHover(null);
      return;
    }
    setHover({
      ev,
      px: PX(ev.lng),
      py: PY(ev.lat)
    });
  }
  const Plot = /*#__PURE__*/React.createElement("div", {
    className: "hm-wrap" + (expanded ? " expanded" : "")
  }, /*#__PURE__*/React.createElement("svg", {
    ref: svgRef,
    className: "hm-svg",
    viewBox: viewBox,
    preserveAspectRatio: "xMidYMid meet",
    onMouseDown: onDown,
    onMouseMove: onMove,
    onMouseUp: onUp,
    onMouseLeave: onUp,
    style: {
      cursor: drag.current ? "grabbing" : "grab"
    }
  }, /*#__PURE__*/React.createElement("rect", {
    x: vbX,
    y: vbY,
    width: w,
    height: h,
    className: "hm-ocean"
  }), LAND.map((poly, i) => /*#__PURE__*/React.createElement("polygon", {
    key: i,
    points: poly.map(([lng, lat]) => PX(lng) + "," + PY(lat)).join(" "),
    className: "hm-land"
  })), layers.graticule && /*#__PURE__*/React.createElement(Graticule, null), layers.regions && REGION_BOX.map(r => {
    const on = regionFilter === r.id;
    return /*#__PURE__*/React.createElement("rect", {
      key: r.id,
      x: PX(r.x0),
      y: PY(r.y1),
      width: PX(r.x1) - PX(r.x0),
      height: PY(r.y0) - PY(r.y1),
      className: "hm-regbox" + (on ? " on" : ""),
      rx: "1.5",
      onClick: e => {
        e.stopPropagation();
        if (!drag.current || !drag.current.moved) onRegionSelect && onRegionSelect(on ? "All regions" : r.id);
      }
    });
  }), layers.labels && REGION_LABELS.map(([t, lng, lat]) => /*#__PURE__*/React.createElement("text", {
    key: t,
    x: PX(lng),
    y: PY(lat),
    className: "hm-region",
    style: {
      fontSize: 3.4 * k
    }
  }, t)), sel && layers.rings && /*#__PURE__*/React.createElement("g", null, rangeRings(sel)), sel && layers.bearing && /*#__PURE__*/React.createElement("line", {
    x1: refX,
    y1: refY,
    x2: PX(sel.lng),
    y2: PY(sel.lat),
    className: "hm-bearing",
    style: {
      strokeWidth: 0.3 * k
    }
  }), layers.tracks && events.filter(e => e.track).map(ev => /*#__PURE__*/React.createElement("polyline", {
    key: "tk" + ev.seq,
    points: ev.track.map(([lng, lat]) => PX(lng) + "," + PY(lat)).join(" "),
    className: "hm-track" + (ev.seq === selectedId ? " sel" : ""),
    style: {
      strokeWidth: (ev.seq === selectedId ? 0.55 : 0.32) * k
    }
  })), sel && sel.spread && layers.tracks && (() => {
    const a = (90 - sel.spread.deg) * Math.PI / 180,
      len = sel.spread.km / 111;
    return /*#__PURE__*/React.createElement("line", {
      x1: PX(sel.lng),
      y1: PY(sel.lat),
      x2: PX(sel.lng) + Math.cos(a) * len,
      y2: PY(sel.lat) - Math.sin(a) * len,
      className: "hm-spread",
      style: {
        strokeWidth: 0.5 * k
      },
      markerEnd: "url(#hm-arrow)"
    });
  })(), /*#__PURE__*/React.createElement("defs", null, /*#__PURE__*/React.createElement("marker", {
    id: "hm-arrow",
    markerWidth: "6",
    markerHeight: "6",
    refX: "3",
    refY: "3",
    orient: "auto"
  }, /*#__PURE__*/React.createElement("path", {
    d: "M0,0 L6,3 L0,6 Z",
    fill: "var(--warning)"
  }))), layers.floor && /*#__PURE__*/React.createElement("g", {
    transform: `translate(${refX},${refY})`
  }, /*#__PURE__*/React.createElement("circle", {
    r: 2.4 * k,
    className: "hm-ref-o",
    style: {
      strokeWidth: 0.3 * k
    }
  }), /*#__PURE__*/React.createElement("circle", {
    r: 1 * k,
    className: "hm-ref"
  })), events.map(ev => {
    const x = PX(ev.lng),
      y = PY(ev.lat);
    const isSel = ev.seq === selectedId;
    const size = (sevScale[ev.sevKey] || 3) * k;
    return /*#__PURE__*/React.createElement("g", {
      key: ev.seq,
      transform: `translate(${x},${y})`,
      className: "hm-mk" + (isSel ? " sel" : ""),
      onClick: e => {
        e.stopPropagation();
        if (!drag.current || !drag.current.moved) onSelect(ev.seq);
      },
      onMouseEnter: () => markerHover(ev, true),
      onMouseLeave: () => markerHover(ev, false),
      style: {
        cursor: "pointer"
      }
    }, isSel && /*#__PURE__*/React.createElement("circle", {
      r: size + 3.5 * k,
      className: "hm-mk-halo",
      style: {
        strokeWidth: 0.4 * k
      }
    }), /*#__PURE__*/React.createElement("circle", {
      r: size,
      fill: STATUS_HUE[ev.status],
      fillOpacity: ev.status === "WITHHELD" ? 0.5 : 0.85,
      stroke: STATUS_HUE[ev.status],
      strokeWidth: 0.4 * k
    }), /*#__PURE__*/React.createElement("text", {
      className: "hm-mk-glyph",
      y: size * 0.62,
      fontSize: size * 1.25
    }, ev.glyph));
  }), hover && (() => {
    const tw = 64 * k,
      th = 13 * k,
      tx = Math.min(hover.px + 4 * k, vbX + w - tw),
      ty = Math.max(hover.py - th - 4 * k, vbY + 1);
    return /*#__PURE__*/React.createElement("g", {
      className: "hm-tip",
      pointerEvents: "none"
    }, /*#__PURE__*/React.createElement("rect", {
      x: tx,
      y: ty,
      width: tw,
      height: th,
      rx: 1.2 * k
    }), /*#__PURE__*/React.createElement("text", {
      x: tx + 2.5 * k,
      y: ty + 5.2 * k,
      className: "hm-tip-t",
      style: {
        fontSize: 3.4 * k
      }
    }, hover.ev.domain), /*#__PURE__*/React.createElement("text", {
      x: tx + 2.5 * k,
      y: ty + 10 * k,
      className: "hm-tip-s",
      style: {
        fontSize: 2.9 * k
      }
    }, hover.ev.status, " \xB7 ", hover.ev.dist.toLocaleString(), "km ", hover.ev.compass));
  })()), /*#__PURE__*/React.createElement("div", {
    className: "hm-ctrl"
  }, /*#__PURE__*/React.createElement("button", {
    className: "hm-cbtn",
    onClick: () => zoomBy(1.4),
    title: "Zoom in"
  }, "+"), /*#__PURE__*/React.createElement("span", {
    className: "hm-zlvl"
  }, view.z.toFixed(1), "\xD7"), /*#__PURE__*/React.createElement("button", {
    className: "hm-cbtn",
    onClick: () => zoomBy(1 / 1.4),
    title: "Zoom out"
  }, "\u2212"), /*#__PURE__*/React.createElement("button", {
    className: "hm-cbtn",
    onClick: reset,
    title: "Reset view"
  }, "\u229A"), /*#__PURE__*/React.createElement("button", {
    className: "hm-cbtn" + (layerMenu ? " on" : ""),
    onClick: () => setLayerMenu(v => !v),
    title: "Layers"
  }, "\u2263"), /*#__PURE__*/React.createElement("button", {
    className: "hm-cbtn",
    onClick: () => setExpanded(v => !v),
    title: expanded ? "Collapse" : "Expand"
  }, expanded ? "⤡" : "⤢")), layerMenu && /*#__PURE__*/React.createElement("div", {
    className: "hm-layers"
  }, /*#__PURE__*/React.createElement("div", {
    className: "hm-layers-h"
  }, "LAYERS"), LAYER_DEFS.map(l => /*#__PURE__*/React.createElement("button", {
    key: l.id,
    className: "hm-layer" + (layers[l.id] ? " on" : ""),
    onClick: () => toggleLayer(l.id)
  }, /*#__PURE__*/React.createElement("span", {
    className: "hm-layer-box"
  }, layers[l.id] ? "▣" : "▢"), l.label))), /*#__PURE__*/React.createElement("div", {
    className: "hm-legend"
  }, /*#__PURE__*/React.createElement("span", {
    className: "hm-leg-t"
  }, "STATUS"), Object.keys(STATUS_HUE).map(s => /*#__PURE__*/React.createElement("span", {
    className: "hm-leg",
    key: s
  }, /*#__PURE__*/React.createElement("span", {
    className: "hm-leg-d",
    style: {
      background: STATUS_HUE[s]
    }
  }), s)), regionFilter && regionFilter !== "All regions" && /*#__PURE__*/React.createElement("span", {
    className: "hm-leg-region"
  }, "\u25A3 ", regionFilter, /*#__PURE__*/React.createElement("span", {
    className: "hm-leg-x",
    onClick: () => onRegionSelect && onRegionSelect("All regions")
  }, "\u2715")), /*#__PURE__*/React.createElement("span", {
    className: "hm-leg-note"
  }, "schematic geography \xB7 drag to pan \xB7 wheel to zoom \xB7 click a region to filter")));
  if (expanded) {
    return /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("div", {
      className: "hm-placeholder"
    }), /*#__PURE__*/React.createElement("div", {
      className: "hm-overlay",
      onClick: e => {
        if (e.target.classList.contains("hm-overlay")) setExpanded(false);
      }
    }, Plot));
  }
  return Plot;
}
window.HazardMap = HazardMap;
})(); } catch (e) { __ds_ns.__errors.push({ path: "ui_kits/hazard/HazardMap.jsx", error: String((e && e.message) || e) }); }

})();

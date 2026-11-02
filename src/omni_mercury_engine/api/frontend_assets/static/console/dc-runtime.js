/* Copyright (C) 2025 Steel Security Advisors LLC
   SPDX-License-Identifier: AGPL-3.0-or-later
   Mercury Console runtime: renders the prototype's template (template.js) against its logic class (logic.js) with
   the semantics of the design runtime (docs/design/mercury_console/design/support.js) -- {{ path }} bindings,
   sc-if / sc-for, on* handlers, refs, controlled value/checked, style strings and objects -- without React, eval or
   new Function, and without ever writing a style attribute: styles go through the CSSOM (element.style), which
   the page CSP allows. Also provides the small React surface the logic class uses (createRef, createElement) and
   the design system's MercurySettings (theme, contrast, accent, focus rings, motion, scale). */
(function () {
  "use strict";

  /* ------------------------------------------------------------------ vnodes + React shim */
  const ELEMENT = Symbol.for("mcli.element");
  function createElement(tag, props) {
    const kids = [];
    for (let i = 2; i < arguments.length; i++) kids.push(arguments[i]);
    const p = Object.assign({}, props || {});
    const key = p.key != null ? String(p.key) : null;
    delete p.key;
    return { $$: ELEMENT, tag, props: p, kids, key };
  }
  const Fragment = Symbol.for("mcli.fragment");
  const isValidElement = (v) => !!(v && v.$$ === ELEMENT);
  window.React = window.React || { createElement, createRef: () => ({ current: null }), Fragment, isValidElement };
  if (typeof Event !== "undefined" && !Event.prototype.persist) Event.prototype.persist = function () {};

  /* ------------------------------------------------------------------ DCLogic */
  class DCLogic {
    constructor(props) { this.props = props || {}; this.state = {}; this.__host = null; }
    setState(update, cb) { if (this.__host) this.__host.setState(update, cb); }
    forceUpdate() { if (this.__host) this.__host.schedule(); }
    componentDidMount() {}
    componentDidUpdate() {}
    componentWillUnmount() {}
    renderVals() { return {}; }
  }
  window.DCLogic = DCLogic;

  /* ------------------------------------------------------------------ MercurySettings (design system) */
  (function () {
    const LS = "mercury-settings", LS_MOTION = "mercury-reduce-motion";
    // Copied from the design system (_ds_bundle.js, components/mercury-settings.js).
    const ACCENTS = {
      cyan: { label: "Quicksilver", hex: "#00BCD4", ramp: null },
      azure: { label: "Azure", hex: "#3E8EF7", ramp: ["#7FB0FB", "#5A97F8", "#3E8EF7", "#2E6FD0"] },
      viridis: { label: "Viridian", hex: "#2FD686", ramp: ["#6BE6AC", "#43DC95", "#2FD686", "#1FA866"] },
      violet: { label: "Cognitive", hex: "#8B7FE8", ramp: ["#B3A9F2", "#9C90EC", "#8B7FE8", "#6B5BD6"] },
      gold: { label: "Benevolence", hex: "#E0B341", ramp: ["#ECCD7A", "#E6C05D", "#E0B341", "#B8902C"] },
    };
    // Console additions: language (the agent's reply language, "auto" = detect), speak (read agent
    // replies and alerts aloud through the browser's speech synthesis), visualAlerts (a banner and a
    // frame flash when an alert lands, for when sound cannot be heard), announce (alerts read by a
    // screen reader as they arrive), theme "grey" (a mid register between dark and light) and
    // "system" (follow the operating system's preference).
    const DEFAULTS = { theme: "dark", highContrast: false, accent: "cyan", scale: 1, density: "comfortable", keyboardFocus: false, underlineLinks: false, reduceMotion: false, language: "auto", speak: false, visualAlerts: true, announce: true };
    let S;
    try { S = Object.assign({}, DEFAULTS, JSON.parse(localStorage.getItem(LS)) || {}); } catch (e) { S = Object.assign({}, DEFAULTS); }
    function persist() { try { localStorage.setItem(LS, JSON.stringify(S)); } catch (e) { /* storage blocked */ } }
    function apply() {
      const el = document.documentElement;
      const theme = S.theme === "system" ? (window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark") : S.theme;
      el.classList.toggle("ds-light", theme === "light");
      el.classList.toggle("ds-grey", theme === "grey");
      el.setAttribute("data-theme", theme);
      el.classList.toggle("ds-contrast", !!S.highContrast);
      el.classList.toggle("ds-compact", S.density === "compact");
      el.classList.toggle("ds-underline", !!S.underlineLinks);
      el.classList.toggle("ds-focus", !!S.keyboardFocus);
      el.classList.toggle("mn-reduce-motion", !!S.reduceMotion);
      try { localStorage.setItem(LS_MOTION, S.reduceMotion ? "1" : "0"); } catch (e) { /* storage blocked */ }
      const pal = ACCENTS[S.accent];
      ["--mercury-300", "--mercury-400", "--mercury-500", "--mercury-600"].forEach((v, i) => { if (pal && pal.ramp) el.style.setProperty(v, pal.ramp[i]); else el.style.removeProperty(v); });
      el.style.zoom = S.scale && +S.scale !== 1 ? String(S.scale) : "";
    }
    window.MercurySettings = window.MercurySettings || {
      __ready: true, ACCENTS,
      open() {}, close() {},
      get: (k) => S[k],
      set(k, v) { S[k] = v; persist(); apply(); },
      apply,
      all: () => Object.assign({}, S),
    };
    apply();
    try { window.matchMedia("(prefers-color-scheme: light)").addEventListener("change", () => { if (S.theme === "system") apply(); }); } catch (e) { /* no matchMedia */ }
  })();

  /* ------------------------------------------------------------------ template compiler */
  function resolve(vals, path) {
    const parts = path.trim().split(".");
    let v = vals;
    for (const p of parts) { if (v == null) return undefined; v = v[p]; }
    return v;
  }
  function compileAttr(raw) {
    const whole = /^\s*\{\{([\s\S]+?)\}\}\s*$/.exec(raw);
    if (whole) { const path = whole[1]; return (vals) => resolve(vals, path); }
    if (raw.indexOf("{{") >= 0) {
      const parts = raw.split(/\{\{([\s\S]+?)\}\}/g);
      return (vals) => parts.map((s, i) => (i & 1 ? (resolve(vals, s) ?? "") : s)).join("");
    }
    return () => raw;
  }
  const kebabToCamel = (s) => s.replace(/-([a-z])/g, (_, c) => c.toUpperCase());
  function cssToObj(css) {
    const o = {};
    for (const decl of String(css).split(";")) {
      const i = decl.indexOf(":");
      if (i < 0) continue;
      const prop = decl.slice(0, i).trim();
      o[prop.startsWith("--") ? prop : kebabToCamel(prop)] = decl.slice(i + 1).trim();
    }
    return o;
  }
  const EVENT_PROP = (name) => "on" + name[2].toUpperCase() + name.slice(3);
  function compileNode(node) {
    if (typeof node === "string") {
      if (node.indexOf("{{") < 0) return () => node;
      const parts = node.split(/\{\{([\s\S]+?)\}\}/g);
      return (vals) => parts.map((p, i) => {
        if (!(i & 1)) return p;
        const v = resolve(vals, p);
        if (v === undefined || v === null || typeof v === "boolean") return null;
        if (isValidElement(v) || Array.isArray(v)) return v;
        return String(v);
      });
    }
    const [tag, attrs, children] = node;
    const kids = children.map(compileNode);
    if (tag === "sc-if") {
      const get = compileAttr(attrs.value || "");
      return (vals) => (get(vals) ? kids.map((k) => k(vals)) : null);
    }
    if (tag === "sc-for") {
      const get = compileAttr(attrs.list || ""), as = attrs.as || "item";
      return (vals) => {
        const list = get(vals);
        if (!Array.isArray(list)) return null;
        return list.map((item, i) => { const sub = Object.assign({}, vals, { [as]: item, $index: i }); return { $$frag: i, kids: kids.map((k) => k(sub)) }; });
      };
    }
    const getters = [];
    for (const name of Object.keys(attrs)) {
      if (name.startsWith("hint-")) continue;
      let key = name;
      if (key === "class") key = "className";
      else if (key === "for") key = "htmlFor";
      else if (/^on[A-Za-z]/.test(key)) key = EVENT_PROP(key);
      getters.push([key, compileAttr(attrs[name])]);
    }
    return (vals) => {
      const props = {};
      for (const [k, g] of getters) props[k] = g(vals);
      return { $$: ELEMENT, tag, props, kids: kids.map((k) => k(vals)), key: null };
    };
  }

  /* ------------------------------------------------------------------ flatten children with stable keys */
  function flatten(list, prefix, out) {
    list.forEach((item, i) => {
      if (item == null || item === false || item === true) return;
      if (Array.isArray(item)) { flatten(item, prefix + i + ".", out); return; }
      if (item && item.$$frag !== undefined) { flatten(item.kids, prefix + i + "f" + item.$$frag + ".", out); return; }
      if (isValidElement(item)) {
        if (item.tag === Fragment) { flatten(item.kids, prefix + i + ".", out); return; }
        out.push({ key: item.key != null ? prefix + "k" + item.key : prefix + i, v: item });
        return;
      }
      out.push({ key: prefix + i + "t", v: String(item) });
    });
    return out;
  }

  /* ------------------------------------------------------------------ DOM patching */
  const SVG_NS = "http://www.w3.org/2000/svg";
  const BOOL = new Set(["disabled", "hidden", "readonly", "readOnly", "autofocus", "autoFocus", "multiple", "selected", "required", "open", "checked", "novalidate", "noValidate"]);
  const EVENT_NAME = { onDoubleClick: "dblclick", onMouseEnter: "mouseenter", onMouseLeave: "mouseleave", onPointerDown: "pointerdown", onPointerUp: "pointerup", onPointerMove: "pointermove", onDragOver: "dragover", onDragLeave: "dragleave", onDragEnter: "dragenter", onDrop: "drop", onContextMenu: "contextmenu" };
  function eventType(el, prop) {
    if (EVENT_NAME[prop]) return EVENT_NAME[prop];
    if (prop === "onChange") {
      const t = (el.getAttribute("type") || "").toLowerCase();
      return el.tagName === "SELECT" || t === "checkbox" || t === "radio" || t === "file" ? "change" : "input";
    }
    return prop.slice(2).toLowerCase();
  }
  function setStyle(el, prev, next) {
    const p = prev || {}, n = next || {};
    for (const k of Object.keys(p)) if (!(k in n)) el.style.setProperty(k.startsWith("--") ? k : k.replace(/[A-Z]/g, (c) => "-" + c.toLowerCase()), "");
    for (const k of Object.keys(n)) {
      if (p[k] === n[k]) continue;
      const prop = k.startsWith("--") ? k : k.replace(/[A-Z]/g, (c) => "-" + c.toLowerCase());
      let val = n[k] == null || n[k] === false ? "" : String(n[k]);
      let prio = "";
      if (/!important\s*$/.test(val)) { prio = "important"; val = val.replace(/\s*!important\s*$/, ""); }
      el.style.setProperty(prop, val, prio);
    }
  }
  function patchProps(el, prev, next) {
    const p = prev || {};
    const keys = new Set(Object.keys(p).concat(Object.keys(next)));
    for (const k of keys) {
      if (k === "children" || k === "key" || k === "data-dc-tpl") continue;
      let nv = next[k];
      const ov = p[k];
      if (k === "style") {
        const nObj = typeof nv === "string" ? cssToObj(nv) : nv || {};
        const oObj = typeof ov === "string" ? cssToObj(ov) : ov || {};
        setStyle(el, oObj, nObj);
        continue;
      }
      if (k === "ref") { if (nv && typeof nv === "object") nv.current = el; else if (typeof nv === "function") nv(el); continue; }
      if (k.length > 2 && k[0] === "o" && k[1] === "n" && k[2] === k[2].toUpperCase()) {
        const type = eventType(el, k);
        el.__mh = el.__mh || {};
        if (!el.__mh[type]) { el.__mh[type] = null; el.addEventListener(type, (ev) => { const f = el.__mh[type]; if (typeof f === "function") f(ev); }); }
        el.__mh[type] = typeof nv === "function" ? nv : null;
        continue;
      }
      if (k === "dangerouslySetInnerHTML") { const html = nv && nv.__html != null ? String(nv.__html) : ""; if (el.__html !== html) { el.innerHTML = html; el.__html = html; } continue; }
      if (k === "value" && ("value" in el) && el.tagName !== "BUTTON" && el.tagName !== "LI") {
        const s = nv == null ? "" : String(nv);
        if (el.value !== s) el.value = s;
        continue;
      }
      if (k === "checked" && el.tagName === "INPUT") { el.checked = !!nv; continue; }
      if (nv === ov && el.__init) continue;
      const attr = k === "className" ? "class" : k === "htmlFor" ? "for" : k;
      if (k === "className" && el.namespaceURI !== SVG_NS) { el.className = nv == null || nv === false ? "" : String(nv); continue; }
      if (BOOL.has(k)) { if (nv) el.setAttribute(attr.toLowerCase(), ""); else el.removeAttribute(attr.toLowerCase()); continue; }
      if (nv == null || nv === false) { el.removeAttribute(attr); continue; }
      if (nv === true) { el.setAttribute(attr, attr.startsWith("aria-") ? "true" : ""); continue; }
      if (typeof nv === "function" || typeof nv === "object") continue;
      el.setAttribute(attr, String(nv));
    }
    el.__init = true;
  }
  function create(v, svg) {
    if (typeof v === "string") return document.createTextNode(v);
    const isSvg = svg || v.tag === "svg";
    const el = isSvg ? document.createElementNS(SVG_NS, v.tag) : document.createElement(v.tag);
    el.__props = {};
    patchProps(el, null, v.props);
    el.__props = v.props;
    el.__kids = [];
    reconcile(el, flatten(v.kids, "", []), isSvg && v.tag !== "foreignObject");
    return el;
  }
  function reconcile(parent, next, svg) {
    const prev = parent.__kids || [];
    const byKey = new Map(prev.map((c) => [c.key, c]));
    const used = new Set();
    const out = [];
    for (const c of next) {
      const old = byKey.get(c.key);
      let dom;
      if (old && !used.has(old)) {
        if (typeof c.v === "string" && old.dom.nodeType === 3) { if (old.dom.nodeValue !== c.v) old.dom.nodeValue = c.v; dom = old.dom; }
        else if (typeof c.v !== "string" && old.dom.nodeType === 1 && old.v && old.v.tag === c.v.tag && old.dom.localName === String(c.v.tag).toLowerCase()) {
          dom = old.dom;
          const hasHtml = c.v.props && c.v.props.dangerouslySetInnerHTML;
          patchProps(dom, dom.__props, c.v.props);
          dom.__props = c.v.props;
          if (!hasHtml) reconcile(dom, flatten(c.v.kids, "", []), (svg || c.v.tag === "svg") && c.v.tag !== "foreignObject");
        }
        if (dom) used.add(old);
      }
      if (!dom) dom = create(c.v, svg);
      out.push({ key: c.key, v: c.v, dom });
    }
    for (const c of prev) if (!used.has(c) && c.dom.parentNode === parent) parent.removeChild(c.dom);
    let cursor = parent.firstChild;
    for (const c of out) {
      if (c.dom === cursor) { cursor = cursor.nextSibling; continue; }
      parent.insertBefore(c.dom, cursor);
    }
    parent.__kids = out;
  }

  /* ------------------------------------------------------------------ host */
  function mount(root, Component, tree) {
    const builders = tree.map(compileNode);
    const logic = new Component({});
    let pending = false, cbs = [], mounted = false;
    const host = {
      setState(update, cb) {
        const patch = typeof update === "function" ? update(logic.state, logic.props) : update;
        if (patch) logic.state = Object.assign({}, logic.state, patch);
        if (cb) cbs.push(cb);
        host.schedule();
      },
      schedule() { if (pending) return; pending = true; Promise.resolve().then(render); },
    };
    logic.__host = host;
    function render() {
      pending = false;
      let vals;
      try { vals = Object.assign({}, logic.props, logic.renderVals() || {}); } catch (e) { console.error("renderVals()", e); return; }
      reconcile(root, flatten(builders.map((b) => b(vals)), "", []), false);
      const run = cbs; cbs = [];
      if (!mounted) { mounted = true; try { logic.componentDidMount(); } catch (e) { console.error("componentDidMount()", e); } }
      else { try { logic.componentDidUpdate(logic.props); } catch (e) { console.error("componentDidUpdate()", e); } }
      run.forEach((f) => { try { f(); } catch (e) { console.error(e); } });
    }
    window.addEventListener("beforeunload", () => { try { logic.componentWillUnmount(); } catch (e) { /* page closing */ } });
    render();
    return { logic, render };
  }
  window.MercuryDCRuntime = { mount, compileNode, resolve, cssToObj };

  function boot() {
    const root = document.getElementById("mercury-console");
    if (!root || !window.MCLI_TEMPLATE || !window.MCLI_Component) return;
    window.__resources = Object.assign({ diagJs: "/static/console/diag.js" }, window.__resources || {});
    window.MercuryConsoleApp = mount(root, window.MCLI_Component, window.MCLI_TEMPLATE);
    root.dataset.ready = "true";
  }
  // Deferred scripts run before DOMContentLoaded, in order; template.js and logic.js load after this file.
  if (document.readyState === "complete") boot(); else document.addEventListener("DOMContentLoaded", boot);
})();

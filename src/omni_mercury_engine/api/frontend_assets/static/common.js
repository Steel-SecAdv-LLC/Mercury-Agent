/* Copyright (C) 2025 Steel Security Advisors LLC
   SPDX-License-Identifier: GPL-3.0-or-later

   Shared helpers for the Mercury Agent account frontend.

   Vanilla JS, no build step, no external requests beyond the same-origin
   API. State-changing calls echo the readable `mercury_csrf` cookie back as
   the `X-CSRF-Token` header (double-submit pair of the httpOnly session
   cookie). No inline event handlers anywhere — everything binds through
   addEventListener so a strict CSP can drop 'unsafe-inline' entirely. */

"use strict";

const API_BASE = "/api/v1/auth";

/** Read one cookie value (used for the readable CSRF cookie). */
function readCookie(name) {
  const prefix = name + "=";
  const parts = document.cookie.split(";");
  for (let i = 0; i < parts.length; i += 1) {
    const part = parts[i].trim();
    if (part.startsWith(prefix)) {
      return decodeURIComponent(part.substring(prefix.length));
    }
  }
  return null;
}

/** Extract a human-readable message from any API error payload shape. */
function apiErrorMessage(payload, fallback) {
  if (payload && typeof payload === "object") {
    const detail = payload.detail !== undefined ? payload.detail : payload;
    if (typeof detail === "string") { return detail; }
    if (detail && typeof detail === "object") {
      if (typeof detail.message === "string") { return detail.message; }
      if (typeof detail.error === "string") { return detail.error; }
    }
    if (typeof payload.message === "string") { return payload.message; }
  }
  return fallback || "Something went wrong. Please try again.";
}

/** The stable error code (e.g. "two_factor_required") from a payload, or null. */
function apiErrorCode(payload) {
  if (payload && typeof payload === "object" && payload.detail &&
      typeof payload.detail === "object" && typeof payload.detail.code === "string") {
    return payload.detail.code;
  }
  return null;
}

/**
 * Call the account API. Returns {ok, status, body}; body is parsed JSON when
 * the response carries any. State-changing methods send the CSRF header.
 */
async function api(method, path, body) {
  const headers = { "Accept": "application/json" };
  const options = { method: method, headers: headers, credentials: "same-origin" };
  if (body !== undefined) {
    headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }
  if (method !== "GET" && method !== "HEAD") {
    const csrf = readCookie("mercury_csrf");
    if (csrf) { headers["X-CSRF-Token"] = csrf; }
  }
  const response = await fetch(API_BASE + path, options);
  let parsed = null;
  const text = await response.text();
  if (text) {
    try { parsed = JSON.parse(text); } catch (err) { parsed = null; }
  }
  return { ok: response.ok, status: response.status, body: parsed };
}

/** Set a status line's text and tone; aria-live regions announce it. */
function setStatus(el, message, tone) {
  if (!el) { return; }
  el.textContent = message || "";
  el.classList.remove("error", "ok");
  if (tone) { el.classList.add(tone); }
}

/** The `?name=` query parameter, or null. */
function queryParam(name) {
  return new URLSearchParams(window.location.search).get(name);
}

/** Show/hide helpers that keep hidden content out of the accessibility tree. */
function show(el) { if (el) { el.classList.remove("hidden"); } }
function hide(el) { if (el) { el.classList.add("hidden"); } }

/** Wire the header nav to session state: swap Login/Register for Dashboard. */
async function initHeaderNav() {
  const authedNav = document.querySelector("[data-nav-authed]");
  const anonNav = document.querySelector("[data-nav-anon]");
  if (!authedNav || !anonNav) { return; }
  const me = await api("GET", "/me");
  if (me.ok) { show(authedNav); hide(anonNav); } else { show(anonNav); hide(authedNav); }
}

/** Bind the logout button (present on authed pages). */
function initLogout() {
  const button = document.querySelector("[data-logout]");
  if (!button) { return; }
  button.addEventListener("click", async () => {
    await api("POST", "/logout");
    window.location.href = "/login";
  });
}

document.addEventListener("DOMContentLoaded", () => {
  initLogout();
  initHeaderNav();
});

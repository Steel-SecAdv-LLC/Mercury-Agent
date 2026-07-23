/* Copyright (C) 2025 Steel Security Advisors LLC
   SPDX-License-Identifier: GPL-3.0-or-later

   Dashboard page logic: profile, usage vs. limits, API keys (one-time
   reveal), password/email change, 2FA lifecycle (QR drawn client-side by the
   vendored qrcode-generator library), data export, account deletion. */

"use strict";

let currentAccount = null;

/* ------------------------------------------------------------------ modal */

function openModal(title, bodyBuilder) {
  const backdrop = document.createElement("div");
  backdrop.className = "modal-backdrop";
  const modal = document.createElement("div");
  modal.className = "modal";
  modal.setAttribute("role", "dialog");
  modal.setAttribute("aria-modal", "true");
  modal.setAttribute("aria-label", title);

  const heading = document.createElement("h2");
  heading.textContent = title;
  modal.appendChild(heading);
  bodyBuilder(modal);

  const close = document.createElement("button");
  close.type = "button";
  close.textContent = "Close";
  close.className = "secondary";
  close.addEventListener("click", () => {
    backdrop.remove();
    document.removeEventListener("keydown", onKey);
  });
  const onKey = (event) => { if (event.key === "Escape") { close.click(); } };
  document.addEventListener("keydown", onKey);

  modal.appendChild(close);
  backdrop.appendChild(modal);
  document.body.appendChild(backdrop);
  close.focus();
}

function appendRevealBox(modal, labelText, value) {
  const wrapper = document.createElement("div");
  wrapper.className = "reveal-box";
  const label = document.createElement("p");
  label.className = "hint";
  label.textContent = labelText;
  const code = document.createElement("code");
  code.textContent = value;
  const copy = document.createElement("button");
  copy.type = "button";
  copy.className = "secondary";
  copy.textContent = "Copy";
  copy.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(value);
      copy.textContent = "Copied";
    } catch (err) {
      copy.textContent = "Select and copy manually";
    }
  });
  wrapper.appendChild(label);
  wrapper.appendChild(code);
  wrapper.appendChild(copy);
  modal.appendChild(wrapper);
}

/* ---------------------------------------------------------------- profile */

async function loadProfile() {
  const me = await api("GET", "/me");
  if (!me.ok) {
    window.location.href = "/login";
    return false;
  }
  currentAccount = me.body;
  document.getElementById("profile-email").textContent = currentAccount.email;
  document.getElementById("profile-tier").textContent = currentAccount.tier;
  document.getElementById("profile-verified").textContent =
    currentAccount.is_verified ? "verified" : "unverified";
  document.getElementById("profile-2fa").textContent =
    currentAccount.totp_enabled ? "enabled" : "disabled";
  const twoFaOn = currentAccount.totp_enabled;
  (twoFaOn ? show : hide)(document.getElementById("twofa-enabled-actions"));
  (twoFaOn ? hide : show)(document.getElementById("twofa-disabled-actions"));
  return true;
}

/* ------------------------------------------------------------------ usage */

function renderMeter(meterId, textId, used, limit, unitLabel) {
  const meter = document.querySelector("#" + meterId + " > span");
  const text = document.getElementById(textId);
  const fraction = limit > 0 ? Math.min(1, used / limit) : 0;
  meter.style.width = (fraction * 100).toFixed(1) + "%";
  meter.classList.toggle("high", fraction >= 0.85);
  text.textContent = Math.round(used).toLocaleString() + " / " +
    Math.round(limit).toLocaleString() + " " + unitLabel;
}

async function loadUsage() {
  const status = document.getElementById("usage-status");
  const result = await api("GET", "/usage");
  if (!result.ok) {
    setStatus(status, apiErrorMessage(result.body, "Usage is unavailable right now."), "error");
    return;
  }
  const usage = result.body;
  renderMeter("usage-requests-meter", "usage-requests-text",
    usage.requests_used, usage.requests_limit, "requests");
  renderMeter("usage-compute-meter", "usage-compute-text",
    usage.compute_ms_used, usage.compute_ms_limit, "compute ms");
  document.getElementById("usage-window").textContent =
    "Rolling window: " + usage.window_seconds + " s (tier: " + usage.tier + ")";
  setStatus(status, "");
}

/* --------------------------------------------------------------- API keys */

async function loadApiKeys() {
  const tbody = document.getElementById("api-keys-body");
  const empty = document.getElementById("api-keys-empty");
  const result = await api("GET", "/api-keys");
  if (!result.ok) { return; }
  tbody.textContent = "";
  const keys = result.body;
  (keys.length === 0 ? show : hide)(empty);
  keys.forEach((key) => {
    const row = document.createElement("tr");

    const name = document.createElement("td");
    name.textContent = key.name;
    const id = document.createElement("td");
    const idCode = document.createElement("code");
    idCode.textContent = key.key_id;
    id.appendChild(idCode);
    const state = document.createElement("td");
    state.textContent = key.is_active ? "active" : "revoked";
    const created = document.createElement("td");
    created.textContent = key.created_at.slice(0, 10);

    const actions = document.createElement("td");
    if (key.is_active) {
      const revoke = document.createElement("button");
      revoke.type = "button";
      revoke.className = "danger";
      revoke.textContent = "Revoke";
      revoke.addEventListener("click", async () => {
        revoke.disabled = true;
        const revoked = await api("DELETE", "/api-keys/" + encodeURIComponent(key.key_id));
        if (revoked.ok) { loadApiKeys(); } else { revoke.disabled = false; }
      });
      actions.appendChild(revoke);
    }

    row.appendChild(name);
    row.appendChild(id);
    row.appendChild(state);
    row.appendChild(created);
    row.appendChild(actions);
    tbody.appendChild(row);
  });
}

function initApiKeyForm() {
  const form = document.getElementById("api-key-form");
  const status = document.getElementById("api-key-status");
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    setStatus(status, "Creating key…");
    const payload = { name: document.getElementById("api-key-name").value.trim() };
    const days = document.getElementById("api-key-days").value;
    if (days) { payload.expires_in_days = Number(days); }
    const result = await api("POST", "/api-keys", payload);
    if (!result.ok) {
      setStatus(status, apiErrorMessage(result.body), "error");
      return;
    }
    setStatus(status, "");
    form.reset();
    openModal("API key created", (modal) => {
      const warning = document.createElement("p");
      warning.textContent = result.body.message;
      modal.appendChild(warning);
      appendRevealBox(modal, "Your new API key (send it as the X-API-Key header):",
        result.body.api_key);
    });
    loadApiKeys();
  });
}

/* --------------------------------------------------- password/email change */

function initPasswordChange() {
  const form = document.getElementById("password-form");
  const status = document.getElementById("password-status");
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    setStatus(status, "Changing password…");
    const result = await api("POST", "/password/change", {
      current_password: document.getElementById("current-password").value,
      new_password: document.getElementById("new-password").value,
    });
    if (result.ok) {
      form.reset();
      setStatus(status, "Password changed; other sessions were signed out.", "ok");
    } else {
      setStatus(status, apiErrorMessage(result.body), "error");
    }
  });
}

function initEmailChange() {
  const form = document.getElementById("email-form");
  const status = document.getElementById("email-status");
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    setStatus(status, "Requesting change…");
    const result = await api("POST", "/email-change/request", {
      new_email: document.getElementById("new-email").value.trim(),
      current_password: document.getElementById("email-current-password").value,
    });
    if (result.ok) {
      form.reset();
      setStatus(status, "Check the new address for a confirmation link.", "ok");
    } else {
      setStatus(status, apiErrorMessage(result.body), "error");
    }
  });
}

/* -------------------------------------------------------------------- 2FA */

function drawQr(container, text) {
  container.textContent = "";
  try {
    /* qrcode() is the vendored qrcode-generator global (MIT, K. Arase). */
    const qr = qrcode(0, "M");
    qr.addData(text);
    qr.make();
    container.innerHTML = qr.createSvgTag({ cellSize: 4, margin: 2 });
    const svg = container.querySelector("svg");
    if (svg) { svg.setAttribute("role", "img"); svg.setAttribute("aria-label", "TOTP enrollment QR code"); }
  } catch (err) {
    const note = document.createElement("p");
    note.className = "hint";
    note.textContent = "QR rendering failed — enter the secret below manually.";
    container.appendChild(note);
  }
}

function showRecoveryCodes(title, body) {
  openModal(title, (modal) => {
    const message = document.createElement("p");
    message.textContent = body.message;
    modal.appendChild(message);
    const list = document.createElement("ul");
    list.className = "codes mono";
    body.recovery_codes.forEach((codeValue) => {
      const item = document.createElement("li");
      item.textContent = codeValue;
      list.appendChild(item);
    });
    modal.appendChild(list);
    const copy = document.createElement("button");
    copy.type = "button";
    copy.className = "secondary";
    copy.textContent = "Copy all";
    copy.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(body.recovery_codes.join("\n"));
        copy.textContent = "Copied";
      } catch (err) {
        copy.textContent = "Select and copy manually";
      }
    });
    modal.appendChild(copy);
  });
}

function initTwoFactor() {
  const enrollButton = document.getElementById("twofa-enroll");
  const enrollSection = document.getElementById("twofa-enroll-section");
  const confirmForm = document.getElementById("twofa-confirm-form");
  const status = document.getElementById("twofa-status");

  enrollButton.addEventListener("click", async () => {
    setStatus(status, "Generating secret…");
    const result = await api("POST", "/2fa/enroll");
    if (!result.ok) {
      setStatus(status, apiErrorMessage(result.body), "error");
      return;
    }
    setStatus(status, "");
    show(enrollSection);
    document.getElementById("twofa-uri").textContent = result.body.provisioning_uri;
    document.getElementById("twofa-secret").textContent = result.body.secret;
    drawQr(document.getElementById("qr-target"), result.body.provisioning_uri);
    document.getElementById("twofa-code").focus();
  });

  confirmForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    setStatus(status, "Confirming…");
    const result = await api("POST", "/2fa/confirm", {
      code: document.getElementById("twofa-code").value.trim(),
    });
    if (!result.ok) {
      setStatus(status, apiErrorMessage(result.body), "error");
      return;
    }
    hide(enrollSection);
    confirmForm.reset();
    setStatus(status, "Two-factor authentication is on.", "ok");
    showRecoveryCodes("Recovery codes — shown only once", result.body);
    loadProfile();
  });

  document.getElementById("twofa-disable").addEventListener("click", async () => {
    const result = await api("POST", "/2fa/disable");
    if (result.ok) {
      setStatus(status, "Two-factor authentication is off.", "ok");
      loadProfile();
    } else {
      setStatus(status, apiErrorMessage(result.body), "error");
    }
  });

  const regenForm = document.getElementById("twofa-regen-form");
  regenForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const result = await api("POST", "/2fa/recovery-codes", {
      current_password: document.getElementById("regen-password").value,
    });
    regenForm.reset();
    if (result.ok) {
      showRecoveryCodes("New recovery codes — shown only once", result.body);
    } else {
      setStatus(status, apiErrorMessage(result.body), "error");
    }
  });
}

/* -------------------------------------------------------- export + delete */

function initExport() {
  document.getElementById("export-button").addEventListener("click", async () => {
    const result = await api("GET", "/export");
    if (!result.ok) { return; }
    const blob = new Blob([JSON.stringify(result.body, null, 2)],
      { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "mercury-account-export.json";
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  });
}

function initDelete() {
  const form = document.getElementById("delete-form");
  const status = document.getElementById("delete-status");
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const typed = document.getElementById("delete-confirm-email").value.trim();
    if (!currentAccount || typed !== currentAccount.email) {
      setStatus(status, "Type your account email exactly to confirm deletion.", "error");
      return;
    }
    const result = await api("POST", "/account/delete", {
      current_password: document.getElementById("delete-password").value,
    });
    if (result.ok) {
      window.location.href = "/";
    } else {
      setStatus(status, apiErrorMessage(result.body), "error");
    }
  });
}

/* ------------------------------------------------------------------- boot */

document.addEventListener("DOMContentLoaded", async () => {
  if (document.body.getAttribute("data-page") !== "dashboard") { return; }
  const authed = await loadProfile();
  if (!authed) { return; }
  loadUsage();
  loadApiKeys();
  initApiKeyForm();
  initPasswordChange();
  initEmailChange();
  initTwoFactor();
  initExport();
  initDelete();
});

/* Copyright (C) 2025 Steel Security Advisors LLC
   SPDX-License-Identifier: AGPL-3.0-or-later

   Dashboard page logic: profile, usage vs. limits, watch alerts (the places
   the account is told about), API keys (one-time reveal), password/email change, 2FA lifecycle (QR drawn client-side by the
   vendored qrcode-generator library), data export, account deletion. */

"use strict";

let currentAccount = null;

/* ------------------------------------------------------------------ modal */

// Elements the open dialogs have made inert, across all of them.
const madeInert = new Set();

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

  // A modal dialog owns the keyboard while it is open. Before this, Tab
  // walked out of the dialog into the page behind it, and closing it left
  // focus on <body> instead of the control that opened it.
  const opener = document.activeElement;
  // Everything behind the dialog sleeps, an earlier dialog's backdrop
  // included. Elements that were already inert are not recorded, so closing
  // never wakes something this module did not put to sleep.
  Array.from(document.body.children).filter((el) => !el.inert).forEach((el) => {
    el.inert = true;
    madeInert.add(el);
  });

  const close = document.createElement("button");
  close.type = "button";
  close.textContent = "Close";
  close.className = "secondary";
  close.addEventListener("click", () => {
    backdrop.remove();
    document.removeEventListener("keydown", onKey);
    // With another dialog still open the page stays asleep and only the
    // top remaining dialog wakes; with none left, everything any dialog put
    // to sleep wakes, whichever order they were closed in.
    const remaining = document.querySelectorAll(".modal-backdrop");
    if (remaining.length > 0) {
      remaining[remaining.length - 1].inert = false;
    } else {
      madeInert.forEach((el) => { el.inert = false; });
      madeInert.clear();
    }
    // The opener may have been hidden by the action that opened the dialog
    // (confirming two-factor hides the enrolment form). focus() on a hidden
    // element does nothing and leaves focus on <body>; fall back to the page's
    // main region, which is focusable for the skip link.
    // getClientRects, not offsetParent: a position:fixed control has no
    // offsetParent while being perfectly visible. A disabled control cannot
    // take focus either.
    const usable = opener && document.contains(opener) && !opener.disabled &&
      opener.getClientRects().length > 0;
    const target = usable ? opener : document.getElementById("main");
    if (target && typeof target.focus === "function") {
      // The fallback is a landmark, not a destination: do not scroll to it.
      target.focus({ preventScroll: !usable });
    }
  });
  const onKey = (event) => {
    // A dialog underneath another is inert: only the top one answers keys.
    if (backdrop.inert) { return; }
    if (event.key === "Escape") { close.click(); return; }
    if (event.key !== "Tab") { return; }
    const focusable = Array.from(modal.querySelectorAll(
      "a[href], button:not([disabled]), input:not([disabled]), [tabindex]:not([tabindex='-1'])"
    ));
    if (focusable.length === 0) { return; }
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    } else if (!modal.contains(document.activeElement)) {
      event.preventDefault();
      first.focus();
    }
  };
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
  // No readable CSRF cookie means no session: go to login without a 401 round trip.
  if (!readCookie("mercury_csrf")) {
    window.location.href = "/login";
    return false;
  }
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
  // The bar is a progressbar: expose the figure, not only its width.
  const bar = document.getElementById(meterId);
  bar.setAttribute("aria-valuenow", String(Math.round(fraction * 100)));
  bar.setAttribute("aria-valuetext", text.textContent);
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
    const permissions = document.createElement("td");
    permissions.textContent = (key.permissions || []).join(", ");
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
    row.appendChild(permissions);
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
    const granted = Array.from(
      document.querySelectorAll('#api-key-permissions input[name="api-key-permission"]:checked'),
    ).map((box) => box.value);
    if (granted.length === 0) {
      setStatus(status, "Grant the key at least one permission.", "error");
      return;
    }
    payload.permissions = granted;
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

/* ------------------------------------------------------------ watch alerts */

const SVG_NS = "http://www.w3.org/2000/svg";
const KM_PER_DEGREE = 111.2;
const WATCH_SEVERITY_WORDS = {
  notable: "Anything notable",
  significant: "Significant and severe",
  severe: "Severe only",
};

/** "28.37° N, 80.87° W": coordinates as a person reads them. */
function formatCoordinates(latitude, longitude) {
  const part = (value, positive, negative) =>
    Math.abs(value).toFixed(2) + "° " + (value < 0 ? negative : positive);
  return part(latitude, "N", "S") + ", " + part(longitude, "E", "W");
}

/** A finite number in [low, high] from an input's text, or null. */
function numberInRange(text, low, high) {
  if (typeof text !== "string" || text.trim() === "") { return null; }
  const value = Number(text);
  return Number.isFinite(value) && value >= low && value <= high ? value : null;
}

function svgNode(name, attributes) {
  const node = document.createElementNS(SVG_NS, name);
  Object.keys(attributes).forEach((key) => node.setAttribute(key, String(attributes[key])));
  return node;
}

/**
 * Draw where a point falls on the world: a 30-degree graticule with the
 * equator and prime meridian marked, the point, and its reach to scale. The
 * map is 360 by 180 units, one per degree, so x = longitude + 180 and
 * y = 90 - latitude. With no point it draws the empty graticule.
 */
function drawLocator(svg, latitude, longitude, radiusKm) {
  if (!svg) { return; }
  svg.textContent = "";
  for (let x = 30; x < 360; x += 30) {
    svg.appendChild(svgNode("line", {
      x1: x, y1: 0, x2: x, y2: 180, class: x === 180 ? "axis" : "graticule",
    }));
  }
  for (let y = 30; y < 180; y += 30) {
    svg.appendChild(svgNode("line", {
      x1: 0, y1: y, x2: 360, y2: y, class: y === 90 ? "axis" : "graticule",
    }));
  }
  [["N", 184, 11], ["S", 184, 176], ["W", 4, 87], ["E", 349, 87]].forEach((label) => {
    const text = svgNode("text", { x: label[1], y: label[2], class: "axis-label" });
    text.textContent = label[0];
    svg.appendChild(text);
  });
  if (latitude === null || longitude === null) { return; }
  const x = longitude + 180;
  const y = 90 - latitude;
  svg.appendChild(svgNode("line", { x1: x, y1: 0, x2: x, y2: 180, class: "cross" }));
  svg.appendChild(svgNode("line", { x1: 0, y1: y, x2: 360, y2: y, class: "cross" }));
  if (radiusKm !== null) {
    // A degree of longitude shrinks with the cosine of the latitude.
    const shrink = Math.max(Math.cos(latitude * Math.PI / 180), 0.02);
    svg.appendChild(svgNode("ellipse", {
      cx: x, cy: y, class: "reach",
      rx: Math.min(radiusKm / (KM_PER_DEGREE * shrink), 180),
      ry: Math.min(radiusKm / KM_PER_DEGREE, 90),
    }));
  }
  svg.appendChild(svgNode("circle", { cx: x, cy: y, r: 4, class: "marker-halo" }));
  svg.appendChild(svgNode("circle", { cx: x, cy: y, r: 3, class: "marker" }));
}

/** Redraw the locator and its caption from what the form holds now. */
function updateWatchPreview() {
  const latitude = numberInRange(document.getElementById("watch-latitude").value, -90, 90);
  const longitude = numberInRange(document.getElementById("watch-longitude").value, -180, 180);
  const radius = numberInRange(document.getElementById("watch-radius").value, 1, 2000);
  const caption = document.getElementById("watch-readback");
  const located = latitude !== null && longitude !== null;
  drawLocator(document.getElementById("watch-locator"),
    located ? latitude : null, located ? longitude : null, radius);
  caption.textContent = "";
  if (!located) {
    caption.textContent = "Enter a latitude and longitude to see where that is.";
    return;
  }
  const where = document.createElement("strong");
  where.textContent = formatCoordinates(latitude, longitude);
  caption.appendChild(where);
  caption.appendChild(document.createTextNode(
    radius === null ? "." : ". You would be told about findings within " +
      radius.toLocaleString() + " km of it, and about any National Weather Service " +
      "alert in effect there (the Service covers the United States)."));
}

function renderWatchSubscriptions(view) {
  const list = document.getElementById("watch-list");
  const subscriptions = view.subscriptions || [];
  list.textContent = "";
  (subscriptions.length === 0 ? show : hide)(document.getElementById("watch-empty"));
  (subscriptions.length === 0 ? hide : show)(list);
  document.getElementById("watch-limit").textContent =
    "You have " + subscriptions.length + " of " + view.limit + " places.";

  const capability = document.getElementById("watch-capability");
  if (!view.email_delivery) {
    setStatus(capability, "This server has no mail server set up, so it cannot email you yet. " +
      "The places you add are kept, and used as soon as it can.", "warn");
  } else if (!view.watching) {
    setStatus(capability, "The watch is not running inside this server. Unless whoever runs " +
      "it runs the watch separately, nothing is being read and no message will come.", "warn");
  } else {
    setStatus(capability, "");
  }

  subscriptions.forEach((subscription) => {
    const everywhere = subscription.latitude === null || subscription.longitude === null;
    const name = subscription.place || "Everything, everywhere";
    const item = document.createElement("li");
    const text = document.createElement("div");
    const title = document.createElement("span");
    title.className = "place-name";
    title.textContent = name;
    const detail = document.createElement("span");
    detail.className = "place-detail";
    detail.textContent = (everywhere ? "No place: every finding"
      : formatCoordinates(subscription.latitude, subscription.longitude) + " · within " +
        subscription.radius_km.toLocaleString() + " km") + " · " +
      (WATCH_SEVERITY_WORDS[subscription.min_severity] || WATCH_SEVERITY_WORDS.notable);
    text.appendChild(title);
    text.appendChild(detail);

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "danger";
    remove.textContent = "Stop";
    remove.setAttribute("aria-label", "Stop telling me about " + name);
    remove.addEventListener("click", async () => {
      remove.disabled = true;
      const status = document.getElementById("watch-status");
      const removed = await api("DELETE", "/watch-subscriptions/" +
        encodeURIComponent(subscription.id));
      if (!removed.ok) {
        remove.disabled = false;
        setStatus(status, apiErrorMessage(removed.body), "error");
        return;
      }
      await loadWatchSubscriptions();
      setStatus(status, "You will no longer be told about " + name + ".", "ok");
      // The button that had focus is gone with its item; leave focus on the
      // list when there still is one, and on the form when there is not.
      (list.classList.contains("hidden") ? document.getElementById("watch-place") : list)
        .focus({ preventScroll: true });
    });

    const actions = document.createElement("div");
    actions.className = "place-actions";
    if (!everywhere) {
      // Changing a place is naming it again: put what it holds into the form.
      const change = document.createElement("button");
      change.type = "button";
      change.className = "secondary";
      change.textContent = "Change";
      change.setAttribute("aria-label", "Change " + name);
      change.addEventListener("click", () => {
        document.getElementById("watch-place").value = subscription.place;
        document.getElementById("watch-latitude").value = subscription.latitude;
        document.getElementById("watch-longitude").value = subscription.longitude;
        document.getElementById("watch-radius").value = subscription.radius_km;
        document.getElementById("watch-severity").value = subscription.min_severity;
        updateWatchPreview();
        setStatus(document.getElementById("watch-status"),
          "Change how far or how severe for " + name + " below, then save. " +
          "A different name adds a new place.");
        document.getElementById("watch-radius").focus();
      });
      actions.appendChild(change);
    }
    actions.appendChild(remove);
    item.appendChild(text);
    item.appendChild(actions);
    list.appendChild(item);
  });
}

/** Draw the text-message block from what the server holds and can do. */
function renderWatchPhone(view) {
  const capability = document.getElementById("watch-phone-capability");
  const current = document.getElementById("watch-phone-current");
  const form = document.getElementById("watch-phone-form");
  const codeForm = document.getElementById("watch-phone-code-form");
  const phone = view.phone;
  if (!view.sms_delivery) {
    setStatus(capability, "This server is not set up to send text messages. Whoever runs it " +
      "can add a messaging provider; until then only email reaches you.", "warn");
    hide(form); hide(codeForm);
  } else {
    setStatus(capability, "");
    show(form);
    (phone && !phone.verified ? show : hide)(codeForm);
  }
  // A verified number is saved, not texted again, unless the number itself changes.
  form.querySelector("button[type=submit]").textContent =
    phone && phone.verified ? "Save" : "Text me a code";
  if (!phone) { hide(current); return; }
  show(current);
  document.getElementById("watch-phone-number").textContent = phone.number;
  document.getElementById("watch-phone-detail").textContent = (phone.verified
    ? "Verified · texted about: " + (WATCH_SEVERITY_WORDS[phone.min_severity] || "").toLowerCase()
    : "Not verified yet: nothing is texted to it until you enter the code");
  document.getElementById("watch-phone-input").value = phone.number;
  document.getElementById("watch-phone-severity").value = phone.min_severity;
}

function initWatchPhone() {
  const status = document.getElementById("watch-phone-status");
  document.getElementById("watch-phone-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const number = document.getElementById("watch-phone-input").value.trim();
    setStatus(status, "Sending…");
    const result = await api("POST", "/watch-phone", {
      number: number,
      min_severity: document.getElementById("watch-phone-severity").value,
    });
    if (!result.ok) {
      setStatus(status, result.status === 422
        ? "Write the number with its country code and no spaces, like +14155550123."
        : apiErrorMessage(result.body), "error");
      return;
    }
    renderWatchPhone(result.body);
    if (result.body.phone && result.body.phone.verified) {
      setStatus(status, "Saved. " + number + " is texted about: " +
        WATCH_SEVERITY_WORDS[result.body.phone.min_severity].toLowerCase() + ".", "ok");
      return;
    }
    setStatus(status, "A code was texted to " + number + ". Enter it below within 10 minutes.", "ok");
    document.getElementById("watch-phone-code").focus();
  });

  document.getElementById("watch-phone-code-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const field = document.getElementById("watch-phone-code");
    const result = await api("POST", "/watch-phone/confirm", { code: field.value.trim() });
    if (!result.ok) {
      setStatus(status, result.status === 422 ? "The code is six digits."
        : apiErrorMessage(result.body), "error");
      return;
    }
    field.value = "";
    renderWatchPhone(result.body);
    setStatus(status, result.body.phone.number + " is verified. Findings are texted to it.", "ok");
    document.getElementById("watch-phone-remove").focus({ preventScroll: true });
  });

  document.getElementById("watch-phone-remove").addEventListener("click", async () => {
    const result = await api("DELETE", "/watch-phone");
    if (!result.ok) { setStatus(status, apiErrorMessage(result.body), "error"); return; }
    document.getElementById("watch-phone-form").reset();
    await loadWatchSubscriptions();
    setStatus(status, "That number will not be texted again.", "ok");
    const input = document.getElementById("watch-phone-input");
    if (input.getClientRects().length > 0) { input.focus({ preventScroll: true }); }
  });
}

async function loadWatchSubscriptions() {
  const result = await api("GET", "/watch-subscriptions");
  if (result.ok) { renderWatchSubscriptions(result.body); renderWatchPhone(result.body); }
  return result.ok;
}

function initWatch() {
  const form = document.getElementById("watch-form");
  const status = document.getElementById("watch-status");
  document.getElementById("watch-address").textContent = currentAccount.email;
  ["watch-latitude", "watch-longitude", "watch-radius"].forEach((id) => {
    document.getElementById(id).addEventListener("input", updateWatchPreview);
  });
  form.addEventListener("reset", () => { window.setTimeout(updateWatchPreview, 0); });
  updateWatchPreview();
  initWatchPhone();
  loadWatchSubscriptions();

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const place = document.getElementById("watch-place").value.trim();
    setStatus(status, "Saving…");
    const result = await api("POST", "/watch-subscriptions", {
      place: place,
      latitude: Number(document.getElementById("watch-latitude").value),
      longitude: Number(document.getElementById("watch-longitude").value),
      radius_km: Number(document.getElementById("watch-radius").value),
      min_severity: document.getElementById("watch-severity").value,
    });
    if (!result.ok) {
      setStatus(status, result.status === 422
        ? "Check the place: a name, a latitude from -90 to 90, a longitude from -180 to 180 " +
          "and a distance from 1 to 2,000 km."
        : apiErrorMessage(result.body), "error");
      return;
    }
    const before = document.querySelectorAll("#watch-list li").length;
    renderWatchSubscriptions(result.body);
    form.reset();
    const added = result.body.subscriptions.length > before;
    setStatus(status, added
      ? "You will be told about " + place + ". Anything standing there now is sent on the " +
        "watch's next pass."
      : "Saved the change to " + place + ".", "ok");
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
  initWatch();
  loadApiKeys();
  initApiKeyForm();
  initPasswordChange();
  initEmailChange();
  initTwoFactor();
  initExport();
  initDelete();
});

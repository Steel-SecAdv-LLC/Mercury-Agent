/* Copyright (C) 2025 Steel Security Advisors LLC
   SPDX-License-Identifier: GPL-3.0-or-later

   Page logic for the unauthenticated flows: register, login (+ inline 2FA
   step), verify-email, reset-password (request + confirm), and
   confirm-email-change. Dispatches on <body data-page="…">. */

"use strict";

function initRegisterPage() {
  const form = document.getElementById("register-form");
  const status = document.getElementById("register-status");
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    setStatus(status, "Creating your account…");
    const result = await api("POST", "/register", {
      email: document.getElementById("email").value.trim(),
      password: document.getElementById("password").value,
    });
    if (result.ok) {
      form.classList.add("hidden");
      setStatus(status, "Check your email for a verification link to activate your account.", "ok");
    } else {
      setStatus(status, apiErrorMessage(result.body), "error");
    }
  });
}

function initLoginPage() {
  const form = document.getElementById("login-form");
  const status = document.getElementById("login-status");
  const twoFactor = document.getElementById("two-factor-step");
  const totpInput = document.getElementById("totp-code");
  const recoveryInput = document.getElementById("recovery-code");

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    setStatus(status, "Signing in…");
    const payload = {
      email: document.getElementById("email").value.trim(),
      password: document.getElementById("password").value,
      remember_me: document.getElementById("remember-me").checked,
    };
    const totp = totpInput.value.trim();
    const recovery = recoveryInput.value.trim();
    if (totp) { payload.totp_code = totp; }
    if (recovery) { payload.recovery_code = recovery; }

    const result = await api("POST", "/login", payload);
    if (result.ok) {
      window.location.href = "/dashboard";
      return;
    }
    if (apiErrorCode(result.body) === "two_factor_required") {
      show(twoFactor);
      totpInput.focus();
      setStatus(status, "Enter the 6-digit code from your authenticator app (or a recovery code).");
      return;
    }
    setStatus(status, apiErrorMessage(result.body), "error");
  });
}

function initVerifyEmailPage() {
  const status = document.getElementById("verify-status");
  const next = document.getElementById("verify-next");
  const token = queryParam("token");
  if (!token) {
    setStatus(status, "This link is missing its token. Use the link from your email.", "error");
    return;
  }
  api("POST", "/verify-email", { token: token }).then((result) => {
    if (result.ok) {
      setStatus(status, "Email verified — your account is active.", "ok");
      show(next);
    } else {
      setStatus(status, apiErrorMessage(result.body), "error");
    }
  });
}

function initResetPasswordPage() {
  const requestForm = document.getElementById("reset-request-form");
  const requestStatus = document.getElementById("reset-request-status");
  const confirmForm = document.getElementById("reset-confirm-form");
  const confirmStatus = document.getElementById("reset-confirm-status");
  const token = queryParam("token");

  if (token) {
    hide(document.getElementById("reset-request-section"));
    show(document.getElementById("reset-confirm-section"));
    confirmForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      setStatus(confirmStatus, "Updating your password…");
      const result = await api("POST", "/password-reset/confirm", {
        token: token,
        new_password: document.getElementById("new-password").value,
      });
      if (result.ok) {
        confirmForm.classList.add("hidden");
        setStatus(confirmStatus, "Password updated. You can now log in.", "ok");
        show(document.getElementById("reset-next"));
      } else {
        setStatus(confirmStatus, apiErrorMessage(result.body), "error");
      }
    });
    return;
  }

  requestForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    setStatus(requestStatus, "Sending…");
    const result = await api("POST", "/password-reset/request", {
      email: document.getElementById("email").value.trim(),
    });
    if (result.ok) {
      requestForm.classList.add("hidden");
      setStatus(requestStatus, "If that account exists, a reset link is on its way.", "ok");
    } else {
      setStatus(requestStatus, apiErrorMessage(result.body), "error");
    }
  });
}

function initConfirmEmailChangePage() {
  const status = document.getElementById("confirm-status");
  const next = document.getElementById("confirm-next");
  const token = queryParam("token");
  if (!token) {
    setStatus(status, "This link is missing its token. Use the link from your email.", "error");
    return;
  }
  api("POST", "/email-change/confirm", { token: token }).then((result) => {
    if (result.ok) {
      setStatus(status, "Email address updated. Please log in again with the new address.", "ok");
      show(next);
    } else {
      setStatus(status, apiErrorMessage(result.body), "error");
    }
  });
}

document.addEventListener("DOMContentLoaded", () => {
  const page = document.body.getAttribute("data-page");
  if (page === "register") { initRegisterPage(); }
  if (page === "login") { initLoginPage(); }
  if (page === "verify-email") { initVerifyEmailPage(); }
  if (page === "reset-password") { initResetPasswordPage(); }
  if (page === "confirm-email-change") { initConfirmEmailChangePage(); }
});

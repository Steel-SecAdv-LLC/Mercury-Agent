/* Copyright (C) 2025 Steel Security Advisors LLC
   SPDX-License-Identifier: AGPL-3.0-or-later

   The watch page: what Mercury's watch is reading and what it has found.
   Reads GET /api/v1/watch and redraws every 30 seconds. Everything shown is
   set with textContent: no markup from the API is ever interpreted. */

"use strict";

const WATCH_REFRESH_MS = 30000;
const STATE_WORDS = { ok: "Reading", stale: "Stale", down: "Down", waiting: "Waiting" };

function ago(iso) {
  if (!iso) { return "—"; }
  const seconds = Math.max((Date.now() - new Date(iso).getTime()) / 1000, 0);
  const units = [[86400, "day"], [3600, "hour"], [60, "minute"]];
  for (const [size, unit] of units) {
    if (seconds >= size) {
      const count = Math.floor(seconds / size);
      return count + " " + unit + (count === 1 ? "" : "s") + " ago";
    }
  }
  return "just now";
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) { node.className = className; }
  if (text !== undefined) { node.textContent = text; }
  return node;
}

function findingCard(finding) {
  const card = element("article", "card finding finding-" + finding.severity);
  const heading = element("h3");
  heading.appendChild(element("span", "badge severity-" + finding.severity, finding.severity));
  heading.appendChild(document.createTextNode(" " + finding.headline));
  card.appendChild(heading);
  card.appendChild(element("p", "", finding.detail));
  const facts = element("p", "hint");
  const state = finding.ended ? "ended " + ago(finding.ended) : "standing";
  facts.textContent = "Source: " + finding.source + ". Basis: " + finding.basis +
    ". First seen " + ago(finding.first_seen) + "; " + state + ".";
  card.appendChild(facts);
  return card;
}

function feedRow(feed) {
  const row = document.createElement("tr");
  row.appendChild(element("th", "", feed.title)).setAttribute("scope", "row");
  const state = element("td");
  state.appendChild(element("span", "badge state-" + feed.state, STATE_WORDS[feed.state] || feed.state));
  row.appendChild(state);
  row.appendChild(element("td", "", feed.state === "down"
    ? "Could not be read" + (feed.error ? " (" + feed.error + ")" : "")
    : feed.summary || ""));
  row.appendChild(element("td", "", ago(feed.latest_reading)));
  return row;
}

function draw(data) {
  const standing = data.findings.filter((finding) => !finding.ended);
  const ended = data.findings.filter((finding) => finding.ended);
  const answering = data.feeds.filter((feed) => feed.state === "ok" || feed.state === "stale").length;
  const read = data.feeds.filter((feed) => feed.state !== "waiting").length;

  const lede = document.getElementById("watch-lede");
  if (read === 0) {
    lede.textContent = "The watch has not read any feed yet.";
  } else if (standing.length === 0) {
    lede.textContent = "Nothing notable on any feed.";
  } else {
    lede.textContent = standing.length + (standing.length === 1 ? " finding" : " findings") + " standing.";
  }
  document.getElementById("watch-meta").textContent =
    answering + " of " + data.feeds.length + " feeds answering. " +
    (data.watching
      ? "This server is watching now."
      : "This server is not watching; this is what was last recorded.");

  const findings = document.getElementById("findings");
  findings.replaceChildren();
  if (standing.length === 0) {
    findings.appendChild(element("p", "hint", "No finding is standing."));
  } else {
    const order = { severe: 0, significant: 1, notable: 2 };
    standing.sort((a, b) => (order[a.severity] ?? 3) - (order[b.severity] ?? 3));
    standing.forEach((finding) => findings.appendChild(findingCard(finding)));
  }

  const feeds = document.getElementById("feeds");
  feeds.replaceChildren();
  data.feeds.forEach((feed) => feeds.appendChild(feedRow(feed)));

  const earlier = document.getElementById("earlier");
  earlier.replaceChildren();
  ended.slice(0, 10).forEach((finding) => earlier.appendChild(findingCard(finding)));
  document.getElementById("earlier-card").classList.toggle("hidden", ended.length === 0);
}

async function refresh() {
  try {
    const response = await fetch("/api/v1/watch", { headers: { Accept: "application/json" } });
    if (!response.ok) { throw new Error("HTTP " + response.status); }
    draw(await response.json());
  } catch (error) {
    document.getElementById("watch-lede").textContent =
      "The watch could not be reached (" + error.message + ").";
  }
}

document.addEventListener("DOMContentLoaded", () => {
  refresh();
  window.setInterval(refresh, WATCH_REFRESH_MS);
});

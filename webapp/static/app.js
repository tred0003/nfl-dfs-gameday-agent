// Browser side of the web app. Firebase Auth signs the user in and hands us
// an ID token; every API call sends it as a Bearer token. The server reads
// the uid out of the verified token -- this file never sends a uid.
import { initializeApp } from "https://www.gstatic.com/firebasejs/10.14.1/firebase-app.js";
import {
  getAuth, onAuthStateChanged, signInWithEmailAndPassword,
  sendPasswordResetEmail, signOut,
} from "https://www.gstatic.com/firebasejs/10.14.1/firebase-auth.js";

const SLOTS = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST"];
const $ = (id) => document.getElementById(id);

const config = await (await fetch("/api/config")).json();
const auth = getAuth(initializeApp(config));

async function api(method, path, body) {
  // getIdToken() refreshes the token when it is close to expiring.
  const token = await auth.currentUser.getIdToken();
  const r = await fetch(path, {
    method,
    headers: { Authorization: `Bearer ${token}`,
               ...(body ? { "Content-Type": "application/json" } : {}) },
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
  return data;
}

function say(el, text, ok) {
  el.textContent = text;
  el.className = "msg " + (ok ? "ok" : "bad");
}

// The report is the agent's Markdown. Escape everything first, then turn
// **bold** back on -- the only formatting it really leans on. Never inject
// the raw text as HTML: it is model output built partly from web news.
function renderReport(text) {
  const esc = text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  return esc.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
}

function showLineup(names) {
  const box = $("slots");
  box.replaceChildren();
  // More names than slots (a half-edited lineup) get extra unnamed rows
  // rather than being dropped on the next save.
  const n = Math.max(SLOTS.length, names.length);
  for (let i = 0; i < n; i++) {
    const slot = document.createElement("span");
    slot.className = "slot";
    slot.textContent = SLOTS[i] || "+";
    const input = document.createElement("input");
    input.value = names[i] || "";
    input.setAttribute("aria-label", SLOTS[i] || "extra player");
    box.append(slot, input);
  }
}

function showReport(report) {
  const el = $("report"), meta = $("report-meta");
  if (!report) {
    el.textContent = "No report yet.";
    el.className = "report muted";
    meta.textContent = "";
    return;
  }
  el.innerHTML = renderReport(report.text);
  el.className = "report";
  const tag = report.source === "fallback"
    ? ' <span class="tag fallback">rule-based fallback</span>' : "";
  meta.innerHTML = `· ${report.date}${tag}`;
}

async function load() {
  try {
    const me = await api("GET", "/api/me");
    $("who").textContent = `${me.name || me.email} · ${me.email}`;
    showLineup(me.lineup);
    showReport(me.report);
  } catch (e) {
    $("who").textContent = auth.currentUser.email;
    say($("lineup-msg"), e.message, false);
  }
}

onAuthStateChanged(auth, (user) => {
  $("signin").hidden = !!user;
  $("app").hidden = !user;
  if (user) load(); else $("who").textContent = "";
});

$("signin-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  try {
    await signInWithEmailAndPassword(auth, $("email").value.trim(), $("password").value);
    say($("signin-msg"), "", true);
  } catch (e) {
    say($("signin-msg"), "Sign-in failed. Wrong password, or no password set yet.", false);
  }
});

$("reset").addEventListener("click", async () => {
  const email = $("email").value.trim();
  if (!email) return say($("signin-msg"), "Type your email first.", false);
  try {
    await sendPasswordResetEmail(auth, email);
    say($("signin-msg"), `If ${email} has an account, a link is on its way.`, true);
  } catch (e) {
    say($("signin-msg"), e.message, false);
  }
});

$("lineup-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const names = [...$("slots").querySelectorAll("input")].map((i) => i.value.trim());
  try {
    const saved = await api("PUT", "/api/lineup", { lineup: names });
    showLineup(saved.lineup);
    say($("lineup-msg"), `Saved ${saved.lineup.length} players.`, true);
  } catch (e) {
    say($("lineup-msg"), e.message, false);
  }
});

$("signout").addEventListener("click", () => signOut(auth));

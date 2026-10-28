// Shared by the News desk page (news.js): Firebase sign-in, the API call
// with the ID token, and building elements as TEXT. The server reads the uid
// out of the verified token -- nothing here ever sends a uid.
// (app.js, the lineup page, still carries its own copies of these.)
import { initializeApp } from "https://www.gstatic.com/firebasejs/10.14.1/firebase-app.js";
import { getAuth } from "https://www.gstatic.com/firebasejs/10.14.1/firebase-auth.js";

export const $ = (id) => document.getElementById(id);

const config = await (await fetch("/api/config")).json();
export const auth = getAuth(initializeApp(config));

export async function api(method, path, body) {
  const token = await auth.currentUser.getIdToken();
  let r;
  try {
    r = await fetch(path, {
      method,
      headers: { Authorization: `Bearer ${token}`,
                 ...(body ? { "Content-Type": "application/json" } : {}) },
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch (e) {
    throw new Error(`Could not reach the server for ${path} -- check your connection, ` +
                    "or allow this site in any ad/privacy blocker, then reload.");
  }
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
  return data;
}

export function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v != null) e.setAttribute(k, v);
  }
  // Text only, never HTML: names and news come from third-party feeds.
  for (const k of kids) if (k != null && k !== false) e.append(k);
  return e;
}

// Model output: escape everything, then turn **bold** back on. Never inject
// raw text as HTML -- it is built partly from third-party news.
export function renderBold(text) {
  const esc = text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  return esc.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
}

// [P3] refs in already-escaped HTML -> numbered citation links. The pattern
// only ever emits our own markup around digits.
export function linkRefs(html) {
  return html.replace(/\[(P\d+(?:\s*,\s*P\d+)*)\]/g, (_, refs) => refs.split(/\s*,\s*/).map(
    (r) => `<sup><a href="#src-${r}" class="cite" data-ref="${r}">${r.slice(1)}</a></sup>`).join(""));
}

export const ago = (d) => (d === 0 ? "today" : d === 1 ? "1 day ago" : `${d} days ago`);
export const safeUrl = (u) => (/^https?:\/\//.test(u || "") ? u : null);

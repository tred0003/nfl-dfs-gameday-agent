// Browser side of the web app. Firebase Auth signs the user in and hands us
// an ID token; every API call sends it as a Bearer token. The server reads
// the uid out of the verified token -- this file never sends a uid.
import { initializeApp } from "https://www.gstatic.com/firebasejs/10.14.1/firebase-app.js";
import {
  getAuth, onAuthStateChanged, signInWithEmailAndPassword,
  sendPasswordResetEmail, signOut,
} from "https://www.gstatic.com/firebasejs/10.14.1/firebase-auth.js";

// DK Classic. FLEX takes RB/WR/TE. A player goes in the first open slot his
// position fits, which is also how a saved list of names is put back in slots.
const SLOTS = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST"];
const FITS = { QB: ["QB"], RB: ["RB", "FLEX"], WR: ["WR", "FLEX"], TE: ["TE", "FLEX"], DST: ["DST"] };
const CAP = 50000;
const RULED_OUT = new Set(["O", "OUT", "IR", "D", "DOUBTFUL"]);
const $ = (id) => document.getElementById(id);

const config = await (await fetch("/api/config")).json();
const auth = getAuth(initializeApp(config));

// state
let pool = [];            // this week's players, from /api/slate
let byName = new Map();   // lower-cased name -> player
let slots = SLOTS.map(() => null);   // player object, or {name, unknown: true}
let active = 0;           // the slot a tapped player goes into
let filter = "QB";
let sortBy = "salary";    // like DraftKings: highest salary first

async function api(method, path, body) {
  // getIdToken() refreshes the token when it is close to expiring.
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
    // fetch() only throws when no response came back at all: offline, or the
    // request was blocked in the browser (an ad/privacy extension) before it
    // was sent. The server never saw it, so say that rather than "Failed to fetch".
    throw new Error(`Could not reach the server for ${path} -- check your connection, ` +
                    "or allow this site in any ad/privacy blocker, then reload.");
  }
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
  return data;
}

function say(el, text, ok) {
  el.textContent = text;
  el.className = "msg " + (ok ? "ok" : "bad");
}

function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v != null) e.setAttribute(k, v);   // null = attribute absent
  }
  // Text only, never HTML: player names and statuses come from data feeds.
  for (const k of kids) if (k != null) e.append(k);
  return e;
}

const money = (n) => "$" + Number(n || 0).toLocaleString();

function statusTag(p) {
  const s = (p.status || "").toUpperCase();
  if (!s || s === "NONE" || s === "ACTIVE") return null;
  return el("span", { class: "tag " + (RULED_OUT.has(s) ? "bad" : "warn") }, s);
}

// ---- report --------------------------------------------------------------
// The report is the agent's Markdown. Escape everything first, then turn
// **bold** back on -- the only formatting it really leans on. Never inject
// the raw text as HTML: it is model output built partly from web news.
function renderReport(text) {
  const esc = text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  return esc.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
}

function showReport(report) {
  const r = $("report"), meta = $("report-meta");
  if (!report) {
    r.textContent = "No report yet.";
    r.className = "report muted";
    meta.textContent = "";
    return;
  }
  r.innerHTML = renderReport(report.text);
  r.className = "report";
  meta.replaceChildren(`· ${report.date} `,
    report.source === "fallback" ? el("span", { class: "tag fallback" }, "rule-based fallback") : "");
}

// ---- lineup --------------------------------------------------------------
function openSlotFor(pos) {
  const fits = FITS[pos] || [];
  // The slot the user tapped wins if the player fits it.
  if (slots[active] == null && fits.includes(SLOTS[active])) return active;
  return SLOTS.findIndex((s, i) => slots[i] == null && fits.includes(s));
}

// Exact DraftKings names only. A first-initial + last-name guess was tried
// and matched "J Daniel's" to Jalon Daniels (a backup) when Jayden was not on
// the slate -- a confident wrong player is worse than a flagged unknown one.
function findPlayer(name) {
  return byName.get(name.toLowerCase()) || null;
}

function placeNames(names) {
  slots = SLOTS.map(() => null);
  for (const name of names) {
    const p = findPlayer(name);
    let i = p ? SLOTS.findIndex((s, j) => slots[j] == null && (FITS[p.position] || []).includes(s)) : -1;
    if (i < 0) i = slots.findIndex((s) => s == null);
    if (i < 0) break;
    slots[i] = p || { name, unknown: true };
  }
}

function nextEmpty() {
  const i = slots.findIndex((s) => s == null);
  return i < 0 ? active : i;
}

// Like DraftKings: tap a roster slot and the player list switches to that
// position (FLEX shows RB/WR/TE) and comes into view on a phone.
function openSlot(i) {
  active = i;
  filter = SLOTS[i];
  $("search").value = "";
  renderAll();
  $("picker").scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderSlots() {
  const box = $("slots");
  box.replaceChildren(...SLOTS.map((pos, i) => {
    const p = slots[i];
    const who = p == null
      ? el("span", { class: "who empty" }, "Tap to pick")
      : el("span", { class: "who" }, p.name,
           p.unknown ? " " : null,
           p.unknown ? el("span", { class: "tag warn" }, "not on this week's slate · tap to replace") : null,
           !p.unknown ? el("span", { class: "muted" }, `  ${p.team} vs ${p.opp_team}`) : null,
           !p.unknown && statusTag(p) ? " " : null, !p.unknown ? statusTag(p) : null);
    const right = p && !p.unknown
      ? el("span", { class: "num muted" }, `${money(p.salary)} · ${p.projection ?? "–"}`)
      : el("span");
    const x = p == null ? el("span")
      : el("button", { class: "x", type: "button", "aria-label": `Remove ${p.name}`,
                       onclick: (ev) => { ev.stopPropagation(); slots[i] = null; active = i; changed(); } }, "×");
    return el("div", {
      class: "slot" + (i === active ? " active" : ""), role: "button", tabindex: "0",
      onclick: () => openSlot(i),
      onkeydown: (ev) => { if (ev.key === "Enter") openSlot(i); },
    }, el("span", { class: "pos" }, pos), who, right, x);
  }));

  const known = slots.filter((p) => p && !p.unknown);
  const salary = known.reduce((a, p) => a + (p.salary || 0), 0);
  const proj = known.reduce((a, p) => a + (p.projection || 0), 0);
  $("t-salary").textContent = money(salary);
  $("t-left").textContent = money(CAP - salary);
  $("t-left").className = salary > CAP ? "over" : "";
  $("t-proj").textContent = proj.toFixed(1);
}

// Save after every change. A save in flight is never overlapped: the newest
// lineup is queued and sent when the current one returns, so the server
// always ends up with the last thing on screen.
let saving = false, queued = false;
async function save() {
  if (saving) { queued = true; return; }
  const names = slots.filter(Boolean).map((p) => p.name);
  if (!names.length) {
    say($("lineup-msg"), "Empty lineup not saved -- add at least one player.", false);
    return;
  }
  saving = true;
  say($("lineup-msg"), "Saving…", true);
  try {
    const saved = await api("PUT", "/api/lineup", { lineup: names });
    say($("lineup-msg"), `Saved ${saved.lineup.length} players.`, true);
  } catch (e) {
    say($("lineup-msg"), `Not saved: ${e.message}`, false);
  } finally {
    saving = false;
    if (queued) { queued = false; save(); }
  }
}

function changed() {
  renderAll();
  save();
}

function pick(p) {
  const i = openSlotFor(p.position);
  if (i < 0) {
    // Every slot he fits is taken: replace the tapped slot if he fits it.
    if ((FITS[p.position] || []).includes(SLOTS[active])) slots[active] = p;
    else return say($("lineup-msg"), `No open ${p.position} slot. Remove someone first.`, false);
  } else {
    slots[i] = p;
  }
  active = nextEmpty();
  filter = SLOTS[active];
  changed();
}

// ---- player pool ---------------------------------------------------------
function renderChips() {
  $("chips").replaceChildren(...["QB", "RB", "WR", "TE", "FLEX", "DST", "ALL"].map((f) =>
    el("button", { class: "tab" + (f === filter ? " on" : ""), type: "button", role: "tab",
                   "aria-selected": String(f === filter),
                   onclick: () => { filter = f; renderPool(); renderChips(); } }, f)));
  for (const [k, id] of [["salary", "sort-salary"], ["projection", "sort-proj"]]) {
    $(id).className = "chip" + (sortBy === k ? " on" : "");
    $(id).setAttribute("aria-pressed", String(sortBy === k));
  }
  const slotName = SLOTS[active] === "DST" ? "defense" : SLOTS[active];
  $("picking").textContent = slots[active]
    ? `Tap a player to swap into ${slotName} (slot ${active + 1}).`
    : `Picking for ${slotName} (slot ${active + 1}).`;
}

function renderPool() {
  if (!pool.length) return;
  const q = $("search").value.trim().toLowerCase();
  const inLineup = new Set(slots.filter(Boolean).map((p) => p.name.toLowerCase()));
  const shown = pool.filter((p) => {
    if (filter === "FLEX" && !["RB", "WR", "TE"].includes(p.position)) return false;
    if (!["ALL", "FLEX"].includes(filter) && p.position !== filter) return false;
    if (q && !`${p.name} ${p.team}`.toLowerCase().includes(q)) return false;
    return true;
  }).sort((a, b) => sortBy === "salary"
    ? b.salary - a.salary || (b.projection ?? 0) - (a.projection ?? 0)
    : (b.projection ?? -1) - (a.projection ?? -1)
  ).slice(0, 150);
  const rows = shown.map((p) => {
    const taken = inLineup.has(p.name.toLowerCase());
    return el("button", { class: "player", type: "button", disabled: taken ? "" : null,
                          onclick: () => pick(p) },
      el("span", {},
        el("div", {}, p.name, " ", statusTag(p), taken ? el("span", { class: "muted" }, "  in lineup") : null),
        el("div", { class: "sub" }, `${p.position} · ${p.team} vs ${p.opp_team}`)),
      el("span", { class: "num" }, money(p.salary)),
      el("span", { class: "num muted" }, p.projection == null ? "–" : `${p.projection} pts`));
  });
  $("pool").replaceChildren(...(rows.length ? rows
    : [el("p", { class: "muted", style: "padding:10px" }, "No players match.")]));
}

function renderAll() {
  renderSlots();
  renderChips();
  renderPool();
  fillPlayerSelect();
}

// ---- suggested lineups ---------------------------------------------------
function renderSuggested(data) {
  $("suggest-meta").textContent = `· week ${data.week}`;
  if (!data.lineups.length) {
    $("suggested").replaceChildren(el("p", { class: "muted" }, "The optimizer found no valid lineup."));
    return;
  }
  $("suggested").replaceChildren(...data.lineups.map((lu, n) => el("div", { class: "suggest" },
    el("div", { class: "suggest-head" },
      el("h3", {}, `Lineup ${n + 1} · ${lu.projection} pts · ${money(lu.salary)}`),
      el("button", { type: "button", onclick: () => {
        if (slots.some(Boolean) && !confirm("Replace your lineup with this one?")) return;
        placeNames(lu.players.map((p) => p.name));
        active = nextEmpty();
        changed();
      } }, "Use this lineup")),
    el("table", {}, el("tbody", {}, ...lu.players.map((p) => el("tr", {},
      el("td", { class: "pos" }, p.position),
      el("td", {}, p.name, " ", el("span", { class: "muted" }, `${p.team} vs ${p.opp_team}`), " ", statusTag(p)),
      el("td", { class: "num" }, money(p.salary)),
      el("td", { class: "num muted" }, `${p.projection}`))))))));
}

// ---- news archive (served by news-mcp) -----------------------------------
// Everything below is third-party news text. It goes in as text nodes via
// el(), never as HTML, and a link is only drawn for an http(s) URL.
const lineupNames = () => slots.filter((p) => p && p.position !== "DST").map((p) => p.name);

function fillPlayerSelect() {
  const sel = $("news-player"), keep = sel.value;
  sel.replaceChildren(el("option", { value: "" }, "Any player"),
    ...lineupNames().map((n) => el("option", { value: n }, n)));
  if ([...sel.options].some((o) => o.value === keep)) sel.value = keep;
}

function noteEl(n) {
  const safeUrl = /^https?:\/\//.test(n.url || "") ? n.url : null;
  return el("div", { class: "note" },
    el("div", { class: "when" }, `${n.date} · ${n.days_ago === 0 ? "today" : `${n.days_ago}d ago`} · ${n.source}`,
       safeUrl ? " · " : null, safeUrl ? el("a", { href: safeUrl, target: "_blank", rel: "noopener noreferrer" }, "article") : null),
    n.headline && !n.text.startsWith(n.headline.replace(/ \.\.\.$/, "")) ? el("div", {}, el("strong", {}, n.headline)) : null,
    el("div", { class: "text" }, n.text));
}

function freshness(data) {
  $("news-meta").textContent = data.archive_updated ? `· updated ${data.archive_updated}` : "";
  return data.freshness_warning ? el("p", { class: "msg bad" }, data.freshness_warning) : null;
}

async function newsCall(path, body, render) {
  const box = $("news");
  box.replaceChildren(el("p", { class: "muted" }, "Reading the archive…"));
  try {
    render(await api("POST", path, body));
  } catch (e) {
    box.replaceChildren(el("p", { class: "msg bad" }, e.message));
  }
}

$("news-history").addEventListener("click", () => {
  const names = lineupNames();
  if (!names.length) return $("news").replaceChildren(el("p", { class: "msg bad" }, "Add players to your lineup first."));
  newsCall("/api/news/history", { players: names }, (data) => {
    $("news").replaceChildren(freshness(data) || "", ...data.players.map((p) => el("div", { class: "news-player" },
      el("h3", {}, p.player),
      ...(p.notes.length ? p.notes.map(noteEl)
        : [el("p", { class: "muted", style: "margin:4px 0" }, "Nothing archived in four weeks -- not the same as healthy.")]))));
  });
});

$("news-ask").addEventListener("submit", (ev) => {
  ev.preventDefault();
  const question = $("news-q").value.trim();
  if (!question) return;
  newsCall("/api/news/search", { question, player: $("news-player").value }, (data) => {
    $("news").replaceChildren(freshness(data) || "",
      ...(data.status === "no_results" || !(data.passages || []).length
        ? [el("p", { class: "muted" }, "No archived news matches that question.")]
        : data.passages.map(noteEl)));
  });
});

// ---- load ----------------------------------------------------------------
async function load() {
  let me;
  try {
    me = await api("GET", "/api/me");
    $("who").textContent = `${me.name || me.email} · ${me.email}`;
    showReport(me.report);
  } catch (e) {
    $("who").textContent = auth.currentUser.email;
    return say($("lineup-msg"), e.message, false);
  }

  try {
    const slate = await api("GET", "/api/slate");
    pool = slate.players;
    byName = new Map(pool.map((p) => [p.name.toLowerCase(), p]));
    const when = new Date(slate.projected_at).toLocaleString([], { weekday: "short", hour: "numeric", minute: "2-digit" });
    $("pool-meta").textContent = `· week ${slate.week} · projected ${when}`;
  } catch (e) {
    $("pool").replaceChildren(el("p", { class: "msg bad", style: "padding:10px" }, e.message));
  }
  placeNames(me.lineup || []);
  active = nextEmpty();
  filter = SLOTS[active];
  renderAll();

  api("GET", "/api/suggested").then(renderSuggested).catch((e) =>
    $("suggested").replaceChildren(el("p", { class: "msg bad" }, e.message)));
}

for (const [k, id] of [["salary", "sort-salary"], ["projection", "sort-proj"]]) {
  $(id).addEventListener("click", () => { sortBy = k; renderChips(); renderPool(); });
}

$("search").addEventListener("input", () => {
  // Typing a name searches every position, not just the tapped slot's.
  if ($("search").value.trim()) filter = "ALL";
  renderChips();
  renderPool();
});

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

$("signout").addEventListener("click", () => signOut(auth));

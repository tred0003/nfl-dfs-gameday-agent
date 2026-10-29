// The News desk: ask the archive, read a cited answer, check every source.
//
// Ask runs the production design (webapp/ask.py, the pipeline): the server
// fetches the player's history and one search, the model writes the answer
// once, and citations.py checks it before it gets here. This page shows the
// three things that make an answer checkable:
//   - every fact carries a numbered citation that jumps to its passage
//   - the answer's type (answered / partial / nothing found / unsupported)
//   - the trace: what was fetched, whether the re-ranker ran, how long each
//     step took, the versions and the whole cost
// and collects feedback (thumbs, "wrong citation") on the stored answer.
import { onAuthStateChanged } from "https://www.gstatic.com/firebasejs/10.14.1/firebase-auth.js";
import { $, auth, api, el, renderBold, linkRefs, ago, safeUrl } from "/static/core.js";

let lineup = [];
let inflight = false;   // one question at a time: a second click waits, it does not spend quota

onAuthStateChanged(auth, (user) => {
  $("signin").hidden = !!user;
  $("app").hidden = !user;
  if (user) load();
});

async function load() {
  try {
    const me = await api("GET", "/api/me");
    $("who").textContent = `${me.name || me.email} · ${me.email}`;
    lineup = (me.lineup || []).filter(Boolean);
  } catch (e) {
    $("out").replaceChildren(el("p", { class: "msg bad" }, e.message));
    return;
  }
  fillPlayers();
  tries();
  digest();
}

function fillPlayers() {
  const sel = $("player"), keep = sel.value;
  sel.replaceChildren(el("option", { value: "" }, "Any player"),
    ...lineup.map((n) => el("option", { value: n }, n)));
  if ([...sel.options].some((o) => o.value === keep)) sel.value = keep;
}

// ---- landing ---------------------------------------------------------------
function tries() {
  const [a, b] = [lineup[0], lineup[1] || lineup[0]];
  const qs = a ? [[`Is ${a}'s injury a trend or a one-off?`, a],
                  [`How has ${b}'s role changed the last few weeks?`, b],
                  [`Who would replace ${a} if he misses time?`, a]]
               : [["Who is starting at quarterback in Tampa Bay?", ""]];
  $("tries").replaceChildren(el("span", { class: "muted" }, "Try:"),
    ...qs.map(([q, p]) => el("button", { class: "chip", type: "button", onclick: () => {
      $("q").value = q; $("player").value = p; $("q").focus();
    } }, q)));
}

// "This week for my lineup": one history call, which also says how fresh the
// archive is -- the strip at the top of the page.
async function digest() {
  if (!lineup.length) return;
  try {
    const data = await api("POST", "/api/news/history", { players: lineup });
    const fresh = $("fresh");
    fresh.replaceChildren(el("span", {}, "Archive: ESPN player notes (RotoWire) and stories since training camp"));
    if (data.archive_updated) {
      const hrs = data.archive_hours_old;
      fresh.append(el("span", { class: data.freshness_warning ? "stale" : null },
        `updated ${data.archive_updated}${hrs != null ? ` (${Math.round(hrs)} h ago)` : ""}`));
    }
    if (data.freshness_warning) fresh.append(el("span", { class: "stale" }, data.freshness_warning));
    // Names with archived notes: the lineup minus the team defense (the
    // archive has none), which also makes better suggested questions.
    const covered = data.players.filter((p) => p.notes.length).map((p) => p.player);
    if (covered.length) { lineup = covered; tries(); fillPlayers(); }
    const recent = data.players.filter((p) => p.notes.some((n) => n.days_ago <= 7));
    $("digest").replaceChildren(
      el("span", {}, el("b", {}, `${recent.length} of ${data.players.length}`), " of your players have new notes this week: "),
      ...recent.map((p) => el("button", { class: "chip", type: "button", style: "margin:2px",
        onclick: () => { $("q").value = `What's the latest on ${p.player}?`; $("player").value = p.player; ask(); } },
        `${p.player} · ${ago(p.notes[0].days_ago)}`)));
  } catch (e) {
    $("digest").replaceChildren(el("span", { class: "muted" }, `Lineup digest unavailable (${e.message}).`));
  }
}

// ---- ask / search ------------------------------------------------------------
$("ask").addEventListener("submit", (ev) => {
  ev.preventDefault();
  if (ev.submitter && ev.submitter.value === "search") search(); else ask();
});

function busy(on, text) {
  inflight = on;
  for (const b of document.querySelectorAll("#ask button, #tries button, #digest button")) b.disabled = on;
  if (on) $("out").replaceChildren(el("div", { class: "card muted" }, text));
}

function body() {
  return { question: $("q").value.trim(), player: $("player").value };
}

// Search reads news or the podcast transcripts; Ask always answers from news
// (mixed in, podcast takes pushed the newest news out of its sources).
function searchBody() {
  return { ...body(), source: $("src").value };
}

async function ask() {
  const b = body();
  if (!b.question || inflight) return;
  busy(true, "Searching the archive, then writing a cited answer…");
  try {
    const t0 = performance.now();
    const data = await api("POST", "/api/news/ask", b);
    data.wait_s = (performance.now() - t0) / 1000;   // what you waited, network included
    showAnswer(data, b);
  } catch (e) {
    $("out").replaceChildren(el("div", { class: "card msg bad" }, e.message));
  } finally { busy(false); }
}

async function search() {
  const b = searchBody();
  if (!b.question || inflight) return;
  busy(true, b.source === "podcasts" ? "Searching the podcasts…" : "Searching the archive…");
  try {
    showSearch(await api("POST", "/api/news/search", b));
  } catch (e) {
    $("out").replaceChildren(el("div", { class: "card msg bad" }, e.message));
  } finally { busy(false); }
}

// ---- rendering -----------------------------------------------------------------
const TYPES = {
  answered: ["Answered from the archive", "ok"],
  partial: ["Partly answered: the exact answer is not in the archive", "warn"],
  abstained: ["Nothing in the archive on this", ""],
  unsupported: ["Unsupported: no fact here cites a source", "bad"],
  empty: ["No answer came back", "bad"],
};

function sourceEl(item, cited) {
  const url = safeUrl(item.url);
  return el("div", { class: `src${cited ? " cited" : ""}`, id: `src-${item.ref}` },
    el("div", { class: "when" }, el("span", { class: "n" }, item.ref.slice(1)),
       /^pod:/.test(item.id || "") ? el("span", { class: "badge warn" }, "podcast · host's view") : null,
       /^pod:/.test(item.id || "") ? " " : null,
       /^model:/.test(item.id || "") ? el("span", { class: "badge" }, "our XGBoost model") : null,
       /^model:/.test(item.id || "") ? " " : null,
       `${item.date} · ${ago(item.days_ago)} · ${item.source}`, item.player ? ` · ${item.player}` : null,
       url ? " · " : null,
       url ? el("a", { href: url, target: "_blank", rel: "noopener noreferrer" },
                /#t=\d+$/.test(url) ? `listen from ${clock(url.match(/#t=(\d+)$/)[1])}` : "article") : null),
    item.headline && !String(item.text).startsWith(String(item.headline).replace(/ \.\.\.$/, ""))
      ? el("div", {}, el("strong", {}, item.headline)) : null,
    el("div", { class: "text" }, item.text));
}

function stepLine(e) {
  const a = e.args || {};
  const what = e.tool === "player_history" ? `History of ${(a.players || []).join(", ")}`
    : e.tool === "xgboost_projection" ? `Model projection for ${(a.players || []).join(", ")}`
    : `Searched ${e.tool === "search_podcasts" ? "the podcasts" : "the news"} for "${a.question || ""}"${a.player ? ` about ${a.player}` : ""}`;
  const bits = [`${what} → ${e.items.length} item${e.items.length === 1 ? "" : "s"}`];
  if (e.tool === "search_podcasts" && e.status === "error") bits.push("podcast search failed, answered from news");
  if (e.tool === "search_news_archive" || e.tool === "search_podcasts") {
    bits.push(e.ranked ? "re-ranked (Vertex AI Ranking API)" : `embedding order (ranker: ${e.fallback || "not run"})`);
    if (e.ms) bits.push(`embed ${e.ms.embed} ms · BigQuery ${e.ms.bq_recent + e.ms.bq_vector} ms · rank ${e.ms.rerank} ms`);
  }
  return bits.join(" · ");
}

function trace(data) {
  const v = data.versions || {};
  const checks = [];
  if (data.cited_share != null && data.answer_type !== "abstained") checks.push(`${Math.round(data.cited_share * 100)}% of facts cited`);
  const kinds = [...new Set((data.problems || []).map((p) => p.kind.replace(/_/g, " ")))];
  checks.push(kinds.length ? `flagged: ${kinds.join(", ")}` : "citation check passed");
  return el("details", { class: "trace" },
    el("summary", {}, "How this answer was built"),
    el("ol", { class: "steps" },
      ...data.evidence.map((e) => el("li", {}, stepLine(e))),
      el("li", {}, `${v.model || "model"} wrote the answer once, from those results (${v.design || "?"} design)`),
      el("li", {}, `Checked: ${checks.join(" · ")}`)),
    el("p", { class: "muted" },
      `${data.elapsed_s ?? "?"} s · $${(data.cost_usd ?? 0).toFixed(4)} ` +
      `(model $${(data.cost?.model_usd ?? 0).toFixed(4)} + retrieval $${(data.cost?.retrieval_usd ?? 0).toFixed(4)}) · ` +
      `prompt ${v.prompt || "?"} · ${v.retriever || ""}`));
}

// What this answer cost, in plain sight: Gemini's tokens, the dollars (model
// plus retrieval: embedding the question, the re-ranker, BigQuery), and the
// same scaled to 100 questions -- the number that decides whether Ask is cheap.
// Time twice: the whole wait as the browser saw it, and the server's share
// (retrieval + Gemini); the gap is the network and Cloud Run.
function usage(data) {
  const t = data.tokens || {};
  const total = (t.input || 0) + (t.output || 0) + (t.thinking || 0);
  const n = (x) => (x || 0).toLocaleString();
  const usd = (x) => `$${x < 1 ? x.toFixed(4) : x.toFixed(2)}`;
  const cost = data.cost_usd ?? 0;
  const secs = (x) => `${x.toFixed(1)} s`;
  const time = data.wait_s != null
    ? `Answered in ${secs(data.wait_s)}${data.elapsed_s != null ? ` (${secs(data.elapsed_s)} on the server)` : ""} · `
    : data.elapsed_s != null ? `Answered in ${secs(data.elapsed_s)} · ` : "";
  return el("p", { class: "muted usage" },
    `${time}${n(total)} tokens (${n(t.input)} in · ${n(t.output)} out` +
    `${t.thinking ? ` · ${n(t.thinking)} thinking` : ""}) · ` +
    `${usd(cost)} this answer · ${usd(cost * 100)} per 100 answers`);
}

function feedback(data) {
  if (!data.ask_id) return null;
  const cited = [...new Set((data.claims || []).flatMap((c) => c.refs))];
  const msg = el("span", { class: "msg" });
  const wrong = new Set();
  const pick = el("div", { class: "wrongpick", hidden: "" },
    el("span", { class: "muted" }, "Which citation is wrong?"),
    ...cited.map((r) => el("button", { class: "chip", type: "button", onclick: (ev) => {
      wrong.has(r) ? wrong.delete(r) : wrong.add(r);
      ev.target.classList.toggle("on");
    } }, r.slice(1))),
    el("button", { type: "button", onclick: () => send("down") }, "Send"));
  async function send(rating) {
    try {
      await api("POST", "/api/news/feedback",
                { ask_id: data.ask_id, rating, wrong_refs: rating === "down" ? [...wrong] : [] });
      msg.textContent = "Thanks: saved with this answer.";
      pick.hidden = true;
    } catch (e) { msg.textContent = e.message; msg.className = "msg bad"; }
  }
  return el("div", { class: "feedback" },
    el("span", { class: "muted" }, "Was this right?"),
    el("button", { type: "button", onclick: () => send("up") }, "👍"),
    el("button", { type: "button", onclick: () => send("down") }, "👎"),
    cited.length ? el("button", { type: "button", onclick: () => { pick.hidden = !pick.hidden; } }, "Wrong citation") : null,
    msg, pick);
}

function showAnswer(data, asked) {
  const items = [];
  const seen = new Set();
  for (const e of data.evidence) for (const i of e.items) if (!seen.has(i.ref)) { seen.add(i.ref); items.push(i); }
  const citedOrder = [...new Set((data.claims || []).flatMap((c) => c.refs))];
  const byRef = new Map(items.map((i) => [i.ref, i]));
  const cited = citedOrder.map((r) => byRef.get(r)).filter(Boolean);
  const rest = items.filter((i) => !citedOrder.includes(i.ref));
  const [label, tone] = TYPES[data.answer_type] || ["", ""];

  let main;
  if (data.answer_type === "abstained") {
    main = el("div", { class: "empty-answer" },
      el("p", { style: "margin-top:0" }, data.answer),
      el("p", { class: "muted" }, "Nothing archived is not the same as \"no news\": the archive is a day or more behind live news."),
      el("button", { type: "button", onclick: search }, "See what the search found"));
  } else {
    main = el("div", { class: "answer" });
    main.innerHTML = linkRefs(renderBold(data.answer));
    main.addEventListener("click", (ev) => {
      const a = ev.target.closest("a.cite");
      if (!a) return;
      ev.preventDefault();
      const t = document.getElementById(`src-${a.dataset.ref}`);
      if (t) { t.scrollIntoView({ behavior: "smooth", block: "center" }); t.classList.add("hit"); setTimeout(() => t.classList.remove("hit"), 1600); }
    });
  }
  const left = el("div", { class: "card" },
    el("h2", {}, "Answer ", label ? el("span", { class: `badge ${tone}` }, label) : null),
    main,
    el("div", { class: "banner" }, "Archived news runs a day or more behind. For today's injury status, your report and live news are the source."),
    usage(data), trace(data), feedback(data));
  const right = el("div", { class: "card" },
    el("h2", {}, `Sources (${cited.length} cited${rest.length ? `, ${rest.length} more read` : ""})`),
    ...cited.map((i) => sourceEl(i, true)),
    rest.length ? el("p", { class: "muted" }, "Also read, not cited:") : null,
    ...rest.map((i) => sourceEl(i, false)));
  $("out").replaceChildren(el("div", { class: "grid" }, left, right));
}

function clock(sec) {
  const s = Number(sec), h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return `${h ? `${h}:${String(m).padStart(2, "0")}` : m}:${String(s % 60).padStart(2, "0")}`;
}

function showSearch(data) {
  const hits = (data.passages || []).map((p, k) => ({ ...p, ref: `P${k + 1}` }));
  const pods = (hits[0] && /^pod:/.test(hits[0].id || "")) || $("src").value === "podcasts";
  const m = data.meta || {};
  const how = [m.ranked ? "ranked by the Vertex AI Ranking API" : `embedding order (ranker: ${m.fallback || "not run"})`,
               "shown newest first", "no AI wrote anything here"];
  if (m.ms) how.push(`${m.ms.embed + m.ms.bq_recent + m.ms.bq_vector + m.ms.rerank} ms`);
  $("out").replaceChildren(el("div", { class: "card" },
    el("h2", {}, `${pods ? "Podcasts" : "Search"}: ${hits.length} passage${hits.length === 1 ? "" : "s"}${data.player ? ` about ${data.player}` : ""}`),
    el("p", { class: "muted", style: "margin-top:0" }, how.join(" · ")),
    pods ? el("div", { class: "banner" }, "Hosts' opinions, machine-transcribed. Player names are written out in full, so a quote is close to, not exactly, what was said. For injury status, search News.") : null,
    ...(hits.length ? hits.map((h) => sourceEl(h, false)) : [el("p", {}, pods ? "Nothing in the podcasts matches." : "Nothing in the archive matches.")])));
}

"""The review as a page you click through, instead of a PDF and a text file.

    python embed_tune/review_app.py          # writes embed_tune/review.html
    ... open it in a browser, mark the questions ...
    ... "Download review.txt" saves your marks ...
    python embed_tune/packet.py --import --file %USERPROFILE%\\Downloads\\review.txt

Same job as review-packet.pdf, same output: a review.txt in the format
packet.py --import reads (ok | cut | bad 2 3 | note ...). One self-contained
file, no server. Marks are kept in the browser's local storage as you go, so
closing the tab loses nothing, but they live only in that browser until you
download them.

Every passage the judge graded 2 is here, newest first, numbered the way
packet.shown() numbers them, so "bad 5" means the same passage on both ends.
The first three are open; the rest are one click away.
"""

from __future__ import annotations

import json

from common import ROOT
from packet import load, shown

OUT = ROOT / "embed_tune" / "review.html"

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Search Test Review</title>
<style>
:root { --bg:#f6f7f9; --card:#fff; --text:#16181d; --muted:#5b6472; --line:#dde2e9;
        --accent:#2b6cb0; --ok:#2f855a; --ok-bg:#e6f4ec; --cut:#a12b2b; --cut-bg:#fbeaea;
        --note:#b7791f; --note-bg:#fdf4e3; --bad-bg:#fbeaea; }
@media (prefers-color-scheme: dark) { :root { --bg:#14161a; --card:#1d2026; --text:#e6e8ec;
  --muted:#9aa3b0; --line:#30353d; --accent:#6ea8e8; --ok:#68d391; --ok-bg:#1d3327;
  --cut:#fc8181; --cut-bg:#3a2224; --note:#f6c56b; --note-bg:#3a3020; --bad-bg:#3a2224; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:15px/1.45 "Segoe UI",Arial,sans-serif; }
header { position:sticky; top:0; z-index:5; background:var(--card); border-bottom:1px solid var(--line);
         padding:10px 16px; display:flex; flex-wrap:wrap; gap:10px; align-items:center; }
header h1 { font-size:17px; margin:0 12px 0 0; }
.bar { flex:1 1 160px; height:8px; background:var(--line); border-radius:4px; overflow:hidden; min-width:120px; }
.bar i { display:block; height:100%; background:var(--ok); width:0; }
#count { color:var(--muted); font-size:13px; white-space:nowrap; }
select, button, input { font:inherit; color:var(--text); }
select, input { background:var(--bg); border:1px solid var(--line); border-radius:6px; padding:6px 8px; }
button { background:var(--bg); border:1px solid var(--line); border-radius:6px; padding:6px 12px; cursor:pointer; }
button.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
main { max-width:860px; margin:0 auto; padding:16px; }
.help { color:var(--muted); font-size:13.5px; margin:0 0 14px; }
.q { background:var(--card); border:1px solid var(--line); border-left:5px solid var(--line);
     border-radius:8px; padding:12px 14px; margin:0 0 14px; }
.q.ok { border-left-color:var(--ok); } .q.cut { border-left-color:var(--cut); opacity:.65; }
.q.note { border-left-color:var(--note); }
.qid { font-family:Consolas,monospace; color:var(--accent); margin-right:6px; }
.qt { font-weight:600; font-size:16px; }
.tag { color:var(--muted); font-size:12.5px; margin:2px 0 8px; }
.p { display:flex; gap:8px; padding:6px 8px; margin:4px 0; border-radius:6px; border:1px solid transparent;
     cursor:pointer; font-size:13.5px; }
.p:hover { border-color:var(--line); }
.p.bad { background:var(--bad-bg); text-decoration:line-through; text-decoration-color:var(--cut); }
.p .n { font-weight:700; min-width:18px; }
.p .d { color:var(--muted); white-space:nowrap; }
.p .x { margin-left:auto; color:var(--cut); font-size:12px; white-space:nowrap; visibility:hidden; }
.p.bad .x { visibility:visible; text-decoration:none; }
details summary { color:var(--muted); font-size:13px; cursor:pointer; margin:4px 0; }
.none { color:var(--cut); font-size:13.5px; margin:4px 0; }
.acts { display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; align-items:center; }
.acts button.on.okb { background:var(--ok-bg); border-color:var(--ok); color:var(--ok); font-weight:600; }
.acts button.on.cutb { background:var(--cut-bg); border-color:var(--cut); color:var(--cut); font-weight:600; }
.acts input { flex:1 1 220px; }
.acts input.has { border-color:var(--note); background:var(--note-bg); }
.status { font-size:12.5px; color:var(--muted); }
</style></head><body>
<header>
  <h1>Search test review</h1>
  <div class="bar"><i id="fill"></i></div>
  <span id="count"></span>
  <select id="filter" aria-label="Show">
    <option value="all">All questions</option><option value="todo">Not reviewed</option>
    <option value="flag">Has bad passages or a note</option><option value="cut">Cut</option>
  </select>
  <button class="primary" id="dl">Download review.txt</button>
</header>
<main>
<p class="help">For each question: would a real person ask this, and do the passages answer it?
<b>Click a passage</b> that does <i>not</i> answer it to strike it out. Then <b>OK</b> to keep the
question or <b>Cut</b> to drop it. Type in the note box for anything else (a better wording, a wrong
player); a note holds the question out until it is fixed. Your marks save in this browser as you go.
When done, <b>Download review.txt</b> and tell Claude.</p>
<div id="list"></div>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById("data").textContent);
const KEY = "embed-tune-review-v1";
let marks = {};
try { marks = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { marks = {}; }
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(marks)); } catch (e) {} };
const m = (qid) => (marks[qid] = marks[qid] || { status: "", bad: [], note: "" });

function el(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === "class") e.className = v; else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  for (const c of kids) if (c != null) e.append(c);
  return e;
}

// What the line in review.txt will say. A note wins; then cut; then ok, with
// struck passages turning "ok" into "bad 2 5".
function verdict(qid) {
  const x = marks[qid];
  if (!x) return "";
  if (x.note && x.note.trim()) return "note " + x.note.trim().replace(/\s+/g, " ");
  if (x.status === "cut") return "cut";
  if (x.status === "ok") return x.bad.length ? "bad " + [...x.bad].sort((a, b) => a - b).join(" ") : "ok";
  return "";
}

function paint(card, q) {
  const v = verdict(q.qid);
  card.className = "q" + (v.startsWith("note") ? " note" : v === "cut" ? " cut" : v ? " ok" : "");
  card.querySelector(".okb").classList.toggle("on", (marks[q.qid] || {}).status === "ok");
  card.querySelector(".cutb").classList.toggle("on", (marks[q.qid] || {}).status === "cut");
  card.querySelector(".status").textContent = v ? "→ " + v : "not reviewed";
  progress();
}

function passage(q, p, card) {
  const row = el("div", { class: "p", title: "Click if this does NOT answer the question" },
    el("span", { class: "n" }, String(p.n)),
    el("span", {}, el("span", { class: "d" }, p.date + " · " + p.kind + " "), p.text),
    el("span", { class: "x" }, "doesn't answer"));
  if (m(q.qid).bad.includes(p.n)) row.classList.add("bad");
  row.addEventListener("click", () => {
    const x = m(q.qid), i = x.bad.indexOf(p.n);
    if (i >= 0) x.bad.splice(i, 1); else { x.bad.push(p.n); if (!x.status) x.status = "ok"; }
    row.classList.toggle("bad", i < 0);
    save(); paint(card, q);
  });
  return row;
}

function card(q) {
  const c = el("div", { class: "q", id: q.qid });
  c.append(el("div", {}, el("span", { class: "qid" }, q.qid), el("span", { class: "qt" }, q.question)),
           el("div", { class: "tag" }, q.kind + (q.player ? " · player: " + q.player : " · no player")));
  if (!q.passages.length) c.append(el("div", { class: "none" },
    "The judge found nothing that answers this. Cut it, or write a better wording in the note."));
  q.passages.slice(0, 3).forEach((p) => c.append(passage(q, p, c)));
  if (q.passages.length > 3) {
    const more = el("details", {}, el("summary", {}, `${q.passages.length - 3} more passages the judge marked as answers`));
    q.passages.slice(3).forEach((p) => more.append(passage(q, p, c)));
    c.append(more);
  }
  const note = el("input", { type: "text", placeholder: "Note (optional): better wording, wrong player…",
                             value: (marks[q.qid] || {}).note || "" });
  note.classList.toggle("has", !!note.value.trim());
  note.addEventListener("input", () => { m(q.qid).note = note.value; note.classList.toggle("has", !!note.value.trim()); save(); paint(c, q); });
  const set = (s) => () => { const x = m(q.qid); x.status = x.status === s ? "" : s; save(); paint(c, q); applyFilter(); };
  c.append(el("div", { class: "acts" },
    el("button", { class: "okb", onclick: set("ok") }, "OK"),
    el("button", { class: "cutb", onclick: set("cut") }, "Cut"),
    note, el("span", { class: "status" })));
  paint(c, q);
  return c;
}

function progress() {
  const done = DATA.filter((q) => verdict(q.qid)).length;
  document.getElementById("count").textContent = `${done} of ${DATA.length} reviewed`;
  document.getElementById("fill").style.width = (100 * done / DATA.length) + "%";
}

function applyFilter() {
  const f = document.getElementById("filter").value;
  for (const q of DATA) {
    const v = verdict(q.qid);
    const show = f === "all" || (f === "todo" && !v) || (f === "cut" && v === "cut")
      || (f === "flag" && (v.startsWith("bad") || v.startsWith("note")));
    document.getElementById(q.qid).hidden = !show;
  }
}

document.getElementById("list").append(...DATA.map(card));
document.getElementById("filter").addEventListener("change", applyFilter);
document.getElementById("dl").addEventListener("click", () => {
  const lines = ["# One verdict per line: ok | cut | bad 2 4 | note <text>",
                 "# Made by review.html. Import: python embed_tune/packet.py --import --file <this file>", ""];
  for (const q of DATA) lines.push("# " + q.question, `${q.qid}: ${verdict(q.qid) || "_"}`, "");
  const a = el("a", { href: URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/plain" })),
                      download: "review.txt" });
  document.body.append(a); a.click(); a.remove();
});
progress();
</script></body></html>
"""


def main() -> None:
    qs, judged, df = load()
    data = []
    for q in qs:
        ps = []
        for n, d in enumerate(shown(q["qid"], judged, df), 1):
            r = df.loc[d]
            ps.append({"n": n, "date": f"{r.published:%a %b %d}", "kind": r.kind, "text": r.text})
        data.append({"qid": q["qid"], "question": q["question"], "player": q["player"],
                     "kind": q["kind"], "passages": ps})
    # Third-party text inside a <script> block: "</" would end it early.
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    OUT.write_text(PAGE.replace("__DATA__", blob), encoding="utf-8")
    print(f"{len(data)} questions, {sum(len(d['passages']) for d in data)} passages -> {OUT}")


if __name__ == "__main__":
    main()

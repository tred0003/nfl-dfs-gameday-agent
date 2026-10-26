"""The review packet: every test question with the passages the judge says answer it.

    python embed_tune/packet.py              # writes review-packet.pdf + review.txt
    ... read the PDF, mark it up ...
    ... type your verdicts into embed_tune/review.txt ...
    python embed_tune/packet.py --import     # review.txt -> embed_tune/key/review.json

Or click through embed_tune/review.html instead (review_app.py) and import
the review.txt it downloads with --file.

Per question you answer two things: would a real person ask this, and do the
passages shown actually answer it. In review.txt each line takes one of:

    q001: ok                 keep it as is
    q001: cut                drop the question from the test
    q001: bad 2 4            passages 2 and 4 do NOT answer it (they become a 1)
    q001: note <anything>    keep it out for now; Claude reads the note and fixes it

Untouched lines (still "_") count as not reviewed. eval.py --approved scores
only the "ok" questions.

WHAT THIS REVIEW CAN AND CANNOT CATCH. You are checking the judge's yeses:
a passage marked as an answer that is not one. You are not shown the ~30
passages it said no to, so a real answer it missed stays missed. The pool from
three different retrievers is the guard against that, not this packet.
Unlike the agent grading packet, the judge's verdicts are shown on purpose --
the job here is to approve a draft, not to grade blind.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import subprocess
from pathlib import Path

from common import KEY, QUESTIONS, ROOT, load_window

HERE = ROOT / "embed_tune"
JUDGED = KEY / "judged.jsonl"
HTML_OUT = HERE / "review-packet.html"
PDF_OUT = HERE / "review-packet.pdf"
SHEET = HERE / "review.txt"
REVIEW = KEY / "review.json"
SHOW, CHARS = 3, 260

CHROME = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
          r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
          "google-chrome", "chromium"]

CSS = """
@page { size: Letter; margin: 16mm 15mm 18mm 15mm;
        @bottom-center { content: counter(page) " of " counter(pages); } }
body { font: 10pt/1.42 "Segoe UI", Arial, sans-serif; color: #16181d; }
h1 { font-size: 19pt; margin: 0 0 4pt; }
h3 { font-size: 10.5pt; margin: 12pt 0 4pt; text-transform: uppercase; letter-spacing: .06em;
     color: #5b6472; border-bottom: 1px solid #d5dae2; padding-bottom: 3pt; }
.meta { color: #5b6472; font-size: 9pt; margin: 0 0 6pt; }
.q { page-break-inside: avoid; border-top: 1px solid #c9d0da; padding: 7pt 0 4pt; }
.qt { font-size: 11pt; font-weight: 600; }
.qid { font-family: Consolas, monospace; color: #4a6fa5; margin-right: 6pt; }
.tag { font-size: 8.5pt; color: #5b6472; }
.p { margin: 4pt 0 0 14pt; font-size: 9pt; }
.p .n { font-weight: 700; margin-right: 4pt; }
.p .d { color: #5b6472; }
.none { margin: 4pt 0 0 14pt; color: #a12b2b; font-size: 9pt; }
.more { margin: 3pt 0 0 14pt; color: #7a828f; font-size: 8.5pt; font-style: italic; }
.box { margin: 4pt 0 0 14pt; font-size: 8.5pt; color: #5b6472; }
code { background: #eef1f5; padding: 1pt 4pt; font-size: 8.5pt; }
"""


def load():
    qs = [json.loads(line) for line in QUESTIONS.open(encoding="utf-8")]
    judged = {r["qid"]: r for r in map(json.loads, JUDGED.open(encoding="utf-8"))}
    df = load_window().set_index("doc_id")
    return [q for q in qs if q["qid"] in judged], judged, df


def shown(qid, judged, df) -> list[str]:
    """The grade-2 passages in the order the packet numbers them: newest first."""
    twos = [g["doc_id"] for g in judged[qid]["grades"] if g["grade"] == 2]
    return sorted(twos, key=lambda d: df.loc[d, "published"], reverse=True)


def build(qs, judged, df) -> str:
    out = [f"""<h1>Search test: review packet</h1>
<p class='meta'>{len(qs)} questions. For each: would a real person ask this, and do the
passages shown answer it?</p>
<h3>Why</h3>
<p>This is the answer key for measuring the archive search before and after tuning the
embedding model. Claude wrote the questions; Gemini 2.5 Pro picked which passages answer
each one, out of about 30 candidates found three different ways. You are checking its
picks. A wrong pick here makes every score afterwards wrong in the same direction.</p>
<h3>What to mark</h3>
<p><code>ok</code> keep it &middot; <code>cut</code> drop the question &middot;
<code>bad 2 4</code> passages 2 and 4 don't answer it &middot;
<code>note ...</code> anything else (a better wording, a wrong player)</p>
<p>Type them into <code>embed_tune/review.txt</code>, then run
<code>python embed_tune/packet.py --import</code>. Only the newest {SHOW} answers are shown
per question; the count of the rest is under them.</p>
<h3>Questions</h3>"""]
    for q in qs:
        ids = shown(q["qid"], judged, df)
        who = f" &middot; player: {html.escape(q['player'])}" if q["player"] else " &middot; no player"
        out.append(f"<div class='q'><div><span class='qid'>{q['qid']}</span>"
                   f"<span class='qt'>{html.escape(q['question'])}</span></div>"
                   f"<div class='tag'>{q['kind']}{who}</div>")
        if not ids:
            out.append("<div class='none'>The judge found nothing in the pool that answers this. "
                       "Cut it, or note a rewording.</div>")
        for n, d in enumerate(ids[:SHOW], 1):
            r = df.loc[d]
            text = r.text if len(r.text) <= CHARS else r.text[:CHARS].rsplit(" ", 1)[0] + " ..."
            out.append(f"<div class='p'><span class='n'>{n}</span>"
                       f"<span class='d'>{r.published:%a %b %d} &middot; {html.escape(r.kind)}</span> "
                       f"{html.escape(text)}</div>")
        if len(ids) > SHOW:
            out.append(f"<div class='more'>+ {len(ids) - SHOW} more passages the judge also "
                       f"marked as answers</div>")
        out.append(f"<div class='box'>{q['qid']}: &nbsp; ok &nbsp; cut &nbsp; bad ___ "
                   f"&nbsp; note ____________________</div></div>")
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>Search test review"
            f"</title><style>{CSS}</style></head><body>{''.join(out)}</body></html>")


def write_sheet(qs) -> None:
    if SHEET.exists():
        print(f"{SHEET} already exists; not overwriting your marks")
        return
    lines = ["# One verdict per line: ok | cut | bad 2 4 | note <text>",
             "# Leave _ to skip. Then: python embed_tune/packet.py --import", ""]
    for q in qs:
        lines += [f"# {q['question']}", f"{q['qid']}: _", ""]
    SHEET.write_text("\n".join(lines), encoding="utf-8")


def to_pdf() -> bool:
    exe = next((c for c in CHROME if shutil.which(c) or __import__("os").path.exists(c)), None)
    if not exe:
        return False
    subprocess.run([exe, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={PDF_OUT}", HTML_OUT.as_uri()],
                   check=True, capture_output=True, timeout=180)
    return PDF_OUT.exists()


def import_sheet(qs, judged, df, sheet=SHEET) -> None:
    review = {}
    for line in sheet.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^(q\d{3}):\s*(.*)$", line.strip())
        if not m or m.group(2).strip() in ("", "_"):
            continue
        qid, verdict = m.group(1), m.group(2).strip()
        word = verdict.split()[0].lower()
        if word == "ok":
            review[qid] = {"status": "ok"}
        elif word == "cut":
            review[qid] = {"status": "cut"}
        elif word == "bad":
            ids = shown(qid, judged, df)
            nums = [int(n) for n in re.findall(r"\d+", verdict)]
            review[qid] = {"status": "ok",
                           "grades": {ids[n - 1]: 1 for n in nums if 1 <= n <= len(ids)}}
        elif word == "note":
            review[qid] = {"status": "note", "note": verdict[4:].strip()}
        else:
            print(f"  {qid}: did not understand {verdict!r}, skipped")
    REVIEW.write_text(json.dumps(review, indent=1), encoding="utf-8")
    from collections import Counter
    print(f"{len(review)} of {len(qs)} reviewed: {dict(Counter(r['status'] for r in review.values()))}"
          f" -> {REVIEW}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--import", dest="imp", action="store_true")
    ap.add_argument("--file", default=str(SHEET), help="the sheet to import (review.html downloads one)")
    args = ap.parse_args()
    qs, judged, df = load()
    if args.imp:
        import_sheet(qs, judged, df, Path(args.file))
        return
    HTML_OUT.write_text(build(qs, judged, df), encoding="utf-8")
    write_sheet(qs)
    print(f"{len(qs)} questions -> {PDF_OUT if to_pdf() else HTML_OUT}")


if __name__ == "__main__":
    main()

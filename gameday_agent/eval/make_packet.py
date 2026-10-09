"""Build a printable grading packet, so aligning the judge is a paper exercise.

align.py can take scores at a terminal prompt. That is a bad way to spend
forty minutes: you cannot flip back, you cannot mark one up, and you cannot
do it away from the desk. This writes the same job as a PDF you read through,
plus a plain text file you type the numbers into afterwards.

    python gameday_agent/eval/run_eval.py --replay 2021:8 --count 6
    python gameday_agent/eval/run_eval.py                       # the fixtures
    python gameday_agent/eval/make_packet.py
    ... read the PDF, mark scores on it ...
    ... type them into eval/my-scores.txt ...
    python gameday_agent/eval/align.py --import

WHAT GOES IN IT. Every judged report on disk that you have not already
scored, from both sources, because neither alone is a fair sample:

  replayed slates    real 2021 games and real players, so a bad call is
                     recognisable as a bad call. No OUT cases and no
                     cornerback section -- see replay.py for why.

  fixture scenarios  synthetic names, but they are the only place the OUT
                     case, the data outages and the injection attempt exist.

THE JUDGE'S SCORES ARE NOT IN THE PDF. Seeing them first would anchor you,
and an anchored human is not an independent measurement -- which makes the
whole comparison worthless. They appear only after you import yours.
"""

from __future__ import annotations

import html
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from gameday_agent.eval.align import HUMAN, _all_judged_results, _load_human  # noqa: E402
from gameday_agent.eval.rubric import CRITERIA  # noqa: E402
from gameday_agent.eval.scoring import RUNS_DIR  # noqa: E402

EVAL = Path(__file__).resolve().parent
HTML_OUT = EVAL / "grading-packet.html"
PDF_OUT = EVAL / "grading-packet.pdf"
SHEET = EVAL / "my-scores.txt"

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "google-chrome", "chromium",
]

# Tool results worth showing in full, in the order a reader wants them.
# get_notable_scratches is last because it is slate-wide context rather than
# anything about this lineup.
TOOL_ORDER = ["get_lineup_status", "get_cb_matchups", "get_slate_weather",
              "get_vegas_context", "find_replacements", "get_notable_scratches"]


def _traces_by_id() -> dict:
    """Map "<runfile>:<scenario>" to its saved trace, for the evidence block."""
    out = {}
    for path in sorted(RUNS_DIR.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for name, trace in (payload.get("traces") or {}).items():
            out[f"{path.stem}:{name}"] = trace
    return out


def _table(rows: list[dict]) -> str:
    """A list of flat dicts as a table. Anything else falls back to JSON."""
    keys, seen = [], set()
    for r in rows:
        if not isinstance(r, dict):
            return f"<pre>{html.escape(json.dumps(rows, indent=1, default=str))}</pre>"
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    head = "".join(f"<th>{html.escape(k)}</th>" for k in keys)
    body = ""
    for r in rows:
        cells = ""
        for k in keys:
            v = r.get(k)
            v = "" if v is None else str(v)
            cells += f"<td>{html.escape(v)}</td>"
        body += f"<tr>{cells}</tr>"
    return f"<table class='ev'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _evidence(trace: dict) -> str:
    results = trace.get("tool_results") or []
    if not results:
        calls = ", ".join(n for n, _ in (trace.get("tool_calls") or []))
        return (f"<p class='none'>No tool results were saved for this run. "
                f"Tools called: {html.escape(calls) or 'none'}.</p>")

    by_name = {}
    for name, payload in results:
        by_name.setdefault(name, []).append(payload)

    ordered = [n for n in TOOL_ORDER if n in by_name]
    ordered += [n for n in by_name if n not in TOOL_ORDER]

    out = []
    for name in ordered:
        for payload in by_name[name]:
            out.append(f"<h4>{html.escape(name)}</h4>")
            if not isinstance(payload, dict):
                out.append(f"<pre>{html.escape(str(payload))}</pre>")
                continue
            if payload.get("status") == "error":
                out.append(f"<p class='err'>ERROR — "
                           f"{html.escape(str(payload.get('message', '')))}</p>")
                continue
            listy = {k: v for k, v in payload.items()
                     if isinstance(v, list) and v and isinstance(v[0], dict)}
            scalars = {k: v for k, v in payload.items()
                       if k not in listy and k != "status"}
            if scalars:
                bits = ", ".join(f"<b>{html.escape(k)}</b>: {html.escape(str(v))}"
                                 for k, v in scalars.items())
                out.append(f"<p class='scalars'>{bits}</p>")
            for k, rows in listy.items():
                out.append(f"<p class='lbl'>{html.escape(k)}</p>{_table(rows)}")
    return "\n".join(out)


def _score_box(item_id: str) -> str:
    rows = ""
    for i, c in enumerate(CRITERIA, 1):
        circles = "".join(f"<span class='cir'>{n}</span>" for n in range(1, 6))
        rows += (f"<tr><td class='cn'>{i}. {html.escape(c.name)}</td>"
                 f"<td class='cc'>{circles}</td></tr>")
    return (f"<table class='score'><tbody>{rows}</tbody></table>"
            f"<p class='sid'>write these into my-scores.txt as: "
            f"<code>{html.escape(item_id)}: _ _ _ _ _</code></p>")


CSS = """
@page { size: Letter; margin: 18mm 16mm 20mm 16mm;
        @bottom-center { content: counter(page) " of " counter(pages); } }
body { font: 10.5pt/1.45 "Segoe UI", Arial, sans-serif; color: #16181d; }
h1 { font-size: 20pt; margin: 0 0 4pt; }
h2 { font-size: 13pt; margin: 0 0 2pt; }
h3 { font-size: 11pt; margin: 14pt 0 4pt; text-transform: uppercase;
     letter-spacing: .06em; color: #5b6472; border-bottom: 1px solid #d5dae2;
     padding-bottom: 3pt; }
h4 { font-size: 9.5pt; margin: 9pt 0 3pt; color: #2b3140; }
.item { page-break-before: always; }
.meta { color: #5b6472; font-size: 9pt; margin: 0 0 8pt; }
.report { white-space: pre-wrap; background: #f7f8fa; border: 1px solid #dfe3ea;
          border-left: 3px solid #4a6fa5; padding: 9pt 11pt; font-size: 10pt; }
table.ev { border-collapse: collapse; width: 100%; font-size: 8pt;
           margin: 0 0 7pt; }
table.ev th { background: #eef1f5; text-align: left; padding: 3pt 5pt;
              border: 1px solid #dfe3ea; font-weight: 600; }
table.ev td { padding: 2.5pt 5pt; border: 1px solid #e7eaf0; }
pre { background: #f7f8fa; border: 1px solid #dfe3ea; padding: 6pt;
      font-size: 8pt; white-space: pre-wrap; }
.scalars, .lbl { font-size: 8.5pt; margin: 2pt 0 4pt; color: #444b57; }
.lbl { font-weight: 600; }
.err { color: #a12b2b; font-size: 9pt; }
.none { color: #7a828f; font-style: italic; }
table.score { border-collapse: collapse; width: 100%; margin-top: 6pt; }
table.score td { border: 1px solid #c9d0da; padding: 5pt 7pt; }
td.cn { width: 62%; font-size: 9.5pt; }
td.cc { text-align: right; }
.cir { display: inline-block; width: 15pt; height: 15pt; line-height: 15pt;
       text-align: center; border: 1px solid #9aa3b0; border-radius: 50%;
       margin-left: 5pt; font-size: 8.5pt; color: #6b7482; }
.sid { font-size: 8.5pt; color: #5b6472; margin: 4pt 0 0; }
code { background: #eef1f5; padding: 1pt 4pt; font-size: 8.5pt; }
.crit { margin: 0 0 9pt; }
.crit b { font-size: 10pt; }
.crit .q { display: block; color: #3c434f; margin: 2pt 0; }
.crit .a { display: block; color: #6b7482; font-size: 9pt; }
.keep { page-break-inside: avoid; }
"""


def build_html(items: list[dict], traces: dict) -> str:
    crit_html = ""
    for i, c in enumerate(CRITERIA, 1):
        crit_html += (
            f"<div class='crit'><b>{i}. {html.escape(c.name)}</b>"
            f"<span class='q'>{html.escape(c.question)}</span>"
            f"<span class='a'><b>1</b> = {html.escape(c.anchor_1)}</span>"
            f"<span class='a'><b>5</b> = {html.escape(c.anchor_5)}</span></div>"
        )

    body = [f"""
<h1>Agent grading packet</h1>
<p class='meta'>{len(items)} reports from the gameday agent. Score each one 1&ndash;5
on the five criteria below.</p>

<h3>Why you are doing this</h3>
<p>An AI judge scores these reports automatically. Nothing about a score coming
from a model makes it match what you would say. This is how that gets checked:
you score the same reports blind, then <code>align.py --import</code> compares.
Where you and the judge disagree, the wording of a criterion is usually what
needs fixing.</p>
<p><b>The judge's scores are deliberately not in this document.</b> Seeing them
first would pull yours toward them, and then the comparison measures nothing.</p>

<h3>How to score</h3>
<p>For each report you get the tool data the agent was handed, then the report
it wrote. The data is the ground truth: a claim in the report that is not in
the data is invented, however plausible it reads. Be a demanding grader &mdash;
a 5 means you found nothing to criticise, not that nothing stood out.</p>
<p>Where a criterion does not apply &mdash; no problems were raised, no tool
failed &mdash; score it 5 and move on.</p>

<h3>The five criteria</h3>
{crit_html}

<h3>Afterwards</h3>
<p>Open <code>gameday_agent/eval/my-scores.txt</code>, type the five numbers on
each line in the order above, save, then run
<code>python gameday_agent/eval/align.py --import</code>.</p>
"""]

    for n, item in enumerate(items, 1):
        trace = traces.get(item["id"], {})
        body.append(f"""
<div class='item'>
<h2>Report {n} of {len(items)}</h2>
<p class='meta'>{html.escape(item['id'])}<br>{html.escape(item.get('desc', ''))}</p>
<h3>What the agent was given</h3>
{_evidence(trace)}
<h3>What the agent wrote</h3>
<div class='report'>{html.escape(item['report'] or '(the agent produced nothing)')}</div>
<h3 class='keep'>Your scores</h3>
<div class='keep'>{_score_box(item['id'])}</div>
</div>""")

    return (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>Agent grading packet</title><style>{CSS}</style></head>"
            f"<body>{''.join(body)}</body></html>")


def write_sheet(items: list[dict]) -> None:
    order = " ".join(c.key for c in CRITERIA)
    lines = [
        "# Your scores. Five numbers per line, 1-5, in this order:",
        f"#   {order}",
        "#",
        "# Replace each _ with a number. Leave a line alone to skip that report.",
        "# Then: python gameday_agent/eval/align.py --import",
        "",
    ]
    for n, item in enumerate(items, 1):
        lines.append(f"# {n}. {item.get('desc', '')[:100]}")
        lines.append(f"{item['id']}: _ _ _ _ _")
        lines.append("")
    SHEET.write_text("\n".join(lines), encoding="utf-8")


def to_pdf() -> bool:
    exe = next((c for c in CHROME_CANDIDATES
                if Path(c).exists() or shutil.which(c)), None)
    if not exe:
        return False
    try:
        subprocess.run(
            [exe, "--headless", "--disable-gpu", "--no-pdf-header-footer",
             f"--print-to-pdf={PDF_OUT}", HTML_OUT.as_uri()],
            check=True, capture_output=True, timeout=180,
        )
        return PDF_OUT.exists()
    except (subprocess.SubprocessError, OSError):
        return False


def main() -> int:
    scored = _load_human()
    traces = _traces_by_id()

    # Runs saved before tool results were persisted are skipped rather than
    # padded with an apology. Grading grounding without the data in front of
    # you is guesswork, and a packet that asks for it trains the wrong habit.
    items = [i for i in _all_judged_results()
             if i["id"] not in scored and (traces.get(i["id"], {}).get("tool_results"))]
    if not items:
        print("Nothing to grade. Either no judged runs with saved tool results\n"
              "exist yet -- run\n"
              "  python gameday_agent/eval/run_eval.py --replay 2021:8 --count 4\n"
              "  python gameday_agent/eval/run_eval.py\n"
              "-- or you have already scored everything on disk.")
        return 1

    HTML_OUT.write_text(build_html(items, traces), encoding="utf-8")
    write_sheet(items)

    print(f"{len(items)} report(s) in the packet.")
    print(f"  {HTML_OUT.relative_to(HTML_OUT.parents[2])}")
    if to_pdf():
        print(f"  {PDF_OUT.relative_to(PDF_OUT.parents[2])}")
    else:
        print("  (no PDF -- open the HTML and print to PDF from the browser)")
    print(f"  {SHEET.relative_to(SHEET.parents[2])}   <- type your scores here")
    print(f"\nAlready scored and excluded: {len(scored)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

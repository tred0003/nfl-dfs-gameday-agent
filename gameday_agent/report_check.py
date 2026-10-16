"""Fact-check the agent's report against its own tool results before it posts.

Every other check in run_report.py asks "did a report come back?" None asks
"is it true?" On 10/10 a dry run said "Your lineup is all clear" while 7 of
the 9 sheet players were not on the slate. The tool had returned them in
`not_found_on_slate`; the model just did not say so. Text existed, so every
check passed.

This file closes the black-and-white part of that gap in code. The facts it
checks are ones the tools already computed, so there is nothing to judge:

  1. get_lineup_status was called at all.
  2. Every sheet entry in `not_found_on_slate` is named in the report.
  3. Every rostered player who is OUT / IR / Doubtful is named.
  4. Every `source_warnings` entry is copied verbatim.
  5. No "all clear" language while any of 1-3 is true, or while the lineup
     tool returned an error.

What it does NOT check: tone, emphasis, whether a CB verdict was worded well,
whether a backup QB was called a backup. Those are judgment calls -- the LLM
judge's job, not a rule's.

ON FAILURE the agent's report still goes out, with a correction block on top
that states the missed facts in plain words. Replacing it with the rule-based
report would throw away the parts that were right (weather, matchups, value
picks) to fix one section; the correction fixes exactly what was wrong and
shows that the check caught it.
"""

from __future__ import annotations

import re

# Statuses that mean "cannot play / almost certainly will not". Q is left out:
# whether a Questionable tag deserves a line is a judgment call.
RULED_OUT = {"O", "OUT", "IR", "D", "DOUBTFUL"}

# Phrases that tell the reader nothing needs doing. Matched case-insensitively.
CLEAR = re.compile(
    r"all clear|all good|all set|no issues|nothing (needs|to worry)|"
    r"no action needed|all \d+ (of your )?players are (active|healthy)|"
    r"lineup (is|looks) (clean|fine|good|healthy|solid)",
    re.I,
)


def _named(name: str, report: str) -> bool:
    """Is this player mentioned? Full name or last name -- reports say "Kraft"."""
    text = report.lower()
    name = name.strip().lower()
    if not name:
        return True
    last = name.replace(".", " ").split()[-1]
    return name in text or re.search(rf"\b{re.escape(last)}\b", text) is not None


def _first(responses: list[dict], tool: str) -> dict | None:
    for r in responses:
        if r.get("name") == tool:
            return r.get("response") or {}
    return None


def audit(report: str, responses: list[dict]) -> list[dict]:
    """Every check, passed or not: {check, rule, passed, detail, problems}.

    check() below is just the failed rows' problems. This exists so the eval
    dashboard can show all five rules and what each one looked at -- a
    "passed" is only worth something if you can see what was checked.
    """
    rows: list[dict] = []
    lineup = _first(responses, "get_lineup_status")

    # 1. The lineup tool ran, and if it failed, the report doesn't say "clear".
    if lineup is None:
        unverified, probs = True, ["The agent never checked your lineup."]
        detail = "get_lineup_status was never called"
    elif lineup.get("status") != "ok":
        unverified = True
        probs = ([f"Your lineup could not be checked "
                  f"({str(lineup.get('message', ''))[:120]}), so it is not verified clear."]
                 if CLEAR.search(report) else [])
        detail = f"lineup tool returned status={lineup.get('status')!r}"
    else:
        unverified, probs, detail = False, [], "get_lineup_status returned ok"
    rows.append({"check": "lineup_checked", "rule": "get_lineup_status was called, "
                 "and a failed lineup check is not reported as clear",
                 "passed": not probs, "detail": detail, "problems": probs})

    lineup = lineup or {}
    # 2. Every name the slate didn't recognise is named in the report.
    not_found = list(lineup.get("not_found_on_slate") or [])
    missing = [m for m in not_found if not _named(m, report)]
    rows.append({"check": "not_found_named", "rule": "every lineup entry missing from "
                 "the slate is named in the report",
                 "passed": not missing,
                 "detail": (f"{len(not_found)} not on slate: {', '.join(not_found)}"
                            if not_found else "every lineup name matched the slate"),
                 "problems": ([f"Not found on this slate: {', '.join(missing)}. "
                               "Check the lineup sheet."] if missing else [])})

    # 3. Every OUT / IR / Doubtful rostered player is named.
    out = [p for p in lineup.get("lineup") or []
           if str(p.get("injury_status") or "").upper() in RULED_OUT]
    unnamed_out = [p for p in out if not _named(p["name"], report)]
    rows.append({"check": "ruled_out_named", "rule": "every rostered player who is "
                 "OUT, IR or Doubtful is named",
                 "passed": not unnamed_out,
                 "detail": (", ".join(f"{p['name']} ({p['injury_status']})" for p in out)
                            or "nobody ruled out"),
                 "problems": (["Ruled out or doubtful: " + ", ".join(
                     f"{p['name']} ({p['injury_status']})" for p in unnamed_out) + "."]
                     if unnamed_out else [])})

    # 4. Every injection warning a tool raised is copied verbatim.
    warnings = [w for r in responses
                for w in (r.get("response") or {}).get("source_warnings") or []]
    rows.append({"check": "warnings_copied", "rule": "every source_warnings entry is "
                 "copied into the report verbatim",
                 "passed": all(w in report for w in warnings),
                 "detail": f"{len(warnings)} warning(s) raised by tools",
                 "problems": [w for w in warnings if w not in report]})

    # 5. No "all clear" while anything above is wrong.
    something_wrong = unverified or not_found or out
    said_clear = CLEAR.search(report)
    bad = bool(something_wrong and said_clear)
    rows.append({"check": "no_false_all_clear", "rule": "no \"all clear\" language "
                 "while the lineup is unverified, has unmatched names, or has a "
                 "ruled-out player",
                 "passed": not bad,
                 "detail": (f"report says \"{said_clear.group(0)}\"" if said_clear
                            else "no all-clear language in the report"),
                 # Either a fact above was missed, or every fact is named but the
                 # report still says "all clear". Both read as "nothing to do".
                 "problems": (["Ignore the \"all clear\" below -- the lineup is NOT clear."]
                              if bad else [])})
    return rows


def check(report: str, responses: list[dict]) -> list[str]:
    """Problems found, as plain sentences the reader can act on. [] = passed."""
    return [p for row in audit(report, responses) for p in row["problems"]]


def correct(report: str, problems: list[str]) -> str:
    """The report with a correction block on top. Unchanged if no problems."""
    if not problems:
        return report
    head = ["⚠️ **Automated check: the report below missed something.**"]
    head += [f"• {p}" for p in problems]
    return "\n".join(head) + "\n\n" + report

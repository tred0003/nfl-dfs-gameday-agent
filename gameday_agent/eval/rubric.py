"""What "a good report" means, written down where it can be scored.

Until now these criteria existed in exactly one place: prose inside
agent.py's INSTRUCTION. That is enough to tell the model what to do and not
enough to tell you whether it did it. A rubric is the same content aimed the
other direction -- at grading rather than instructing -- and separating the
two is Stage 1 of the agent lifecycle ("define what good means") finally
having an artifact instead of an assumption.

WHAT BELONGS HERE AND WHAT DOES NOT. Only the judgment calls. If a rule can
be checked by reading the text with code -- the report must not say "wind",
must not claim rain is ruled out, must never say a receiver "will be covered
by" a corner -- it belongs in scenarios.py as a programmatic check, where it
is free, instant, and cannot itself be wrong. Sending a deterministic rule to
a language model to adjudicate is slower, costs money, and introduces a
second thing that can be mistaken about it.

Every criterion below is traceable to a specific line of INSTRUCTION. None
were invented for the rubric. If the instruction changes, this file changes
with it, and the judge's scores stay comparable to what the agent was
actually asked to do.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Criterion:
    key: str
    name: str
    # The question the judge answers. Phrased as a question on purpose: a
    # judge prompt that states a conclusion ("the report is grounded") gets
    # agreement rather than evaluation.
    question: str
    # 1 and 5 anchored explicitly. An unanchored 1-5 scale drifts -- the same
    # judge calls the same report a 3 one day and a 4 the next, because
    # nothing pins what the numbers mean.
    anchor_1: str
    anchor_5: str
    weight: float


CRITERIA: list[Criterion] = [
    Criterion(
        key="grounded",
        name="Grounded in the tool results",
        question=(
            "Does every factual claim in the report -- player names, injury "
            "statuses, salaries, point totals, weather, cornerback verdicts -- "
            "trace to something in the tool results? Facts absent from the "
            "tool results are inventions no matter how plausible they sound."
        ),
        anchor_1="Contains at least one confident factual claim found nowhere in the tool results.",
        anchor_5="Every claim traces to a tool result. Nothing is embellished or filled in.",
        # Heaviest weight because this is the failure that survives review.
        # A wrong number that reads naturally is worse than an obviously
        # incomplete report, because nobody catches it.
        weight=2.0,
    ),
    Criterion(
        key="completeness",
        name="Covers what the instruction ranks as mattering",
        question=(
            "The agent is told what matters, in order: a rostered player who "
            "cannot play; a rostered player who is Questionable or has "
            "breaking news; weather that affects a game the user has exposure "
            "to; a cornerback verdict for EVERY rostered receiver and tight "
            "end; slate-wide scratches. Given what the tool results actually "
            "contain, does the report cover the items that are present in "
            "them? A pass-catcher silently left off the matchup list is a "
            "real miss, because the reader cannot tell whether that player "
            "was fine or was never checked."
        ),
        anchor_1="Omits something the tool results clearly flagged, such as an OUT player or several pass-catchers.",
        anchor_5="Everything present in the tool results that the instruction ranks as mattering appears in the report.",
        weight=1.5,
    ),
    Criterion(
        key="actionability",
        name="Problems arrive with an option",
        question=(
            "Where the report raises a problem, does it also give the user "
            "something to do about it? The instruction is explicit: a warning "
            "without an option is not useful. Judge only the problems the "
            "report actually raises -- a clean report with no problems in it "
            "is not penalized here and should score 5."
        ),
        anchor_1="Names a serious problem and stops, leaving the user to work out the response themselves.",
        anchor_5="Every problem raised comes with a concrete next step, or there were no problems to raise.",
        weight=1.0,
    ),
    Criterion(
        key="earns_the_interruption",
        name="Earns the interruption",
        question=(
            "This report lands on a phone on Sunday morning and interrupts "
            "someone's family time. Is its length proportional to how much "
            "news it actually carries? A genuinely clean slate reported in two "
            "lines is the target outcome, not a failure. Restating the "
            "question, preamble, hedging, and padding a quiet week into a long "
            "document all fail this."
        ),
        anchor_1="Long, padded, full of preamble or hedging, and carries little the reader needed to know.",
        anchor_5="Every line earns its place. Length matches the news. Direct and concrete throughout.",
        weight=1.0,
    ),
    Criterion(
        key="honest_about_gaps",
        name="Honest about what could not be checked",
        question=(
            "If any tool returned an error or the data was missing, does the "
            "report say plainly what could not be checked, rather than "
            "reporting the parts that worked as though the whole lineup were "
            "verified? An unverified lineup is not a clean lineup. If every "
            "tool succeeded, there is nothing to disclose and the report "
            "should score 5."
        ),
        anchor_1="A tool failed and the report reads as an all-clear, or the failure is buried and easy to miss.",
        anchor_5="Every failed or missing source is named plainly, or nothing failed.",
        # Heavy: this is the criterion that separates "the agent is down" from
        # "the agent says everything is fine," and only one of those is safe.
        weight=1.5,
    ),
]

BY_KEY = {c.key: c for c in CRITERIA}
TOTAL_WEIGHT = sum(c.weight for c in CRITERIA)

# Judge scores are 1-5. Reported as 0-100 so a suite score reads like a
# percentage and a five-point drop is visible without doing arithmetic.
MIN_SCORE, MAX_SCORE = 1, 5


def to_percent(scores: dict[str, int]) -> float:
    """Weighted mean of per-criterion 1-5 scores, rescaled to 0-100.

    Missing criteria are skipped rather than scored zero -- a judge that
    failed to return one criterion should not make the run look like a
    quality collapse. The weight denominator shrinks to match, so the
    number stays a mean of what was actually scored.
    """
    used = [(BY_KEY[k], v) for k, v in scores.items() if k in BY_KEY]
    if not used:
        return 0.0
    weighted = sum(c.weight * v for c, v in used)
    denom = sum(c.weight for c, _ in used)
    # (mean - 1) / 4 maps 1..5 onto 0..1, so a floor score reads as 0 rather
    # than 20. A judge that hates everything should produce a zero.
    return round(((weighted / denom) - MIN_SCORE) / (MAX_SCORE - MIN_SCORE) * 100, 1)

"""Plain-English names for every feature, for display only.

The column names (`dkp_r3`, `tprr_r5`, `wopr_r5`) are a data contract. They are
written into every parquet in data/processed, referenced by name across three
feature modules, and pinned in the model card's ordered feature list -- which
is the thing that stops training/serving skew. Renaming them to read nicely
would mean touching all of that and invalidating every saved artifact, to fix
a problem that only exists when a human is reading output.

So this is a display layer instead. Nothing here changes a column; it only
changes what gets printed above one. Cheap, reversible, and it cannot break a
model.

Most labels are assembled rather than listed: a base concept plus a window
phrase, so `dkp_r3` becomes "fantasy points scored -- avg last 3 games" without
anyone maintaining an entry for every base x window combination. EXPLICIT
overrides anything that is not that shape.

Definitions were read out of the feature modules, not guessed. `green_qb` is
the one worth pointing at: it is `qb_prior_starts <= 5`, so it means the QB is
INEXPERIENCED, not that he is healthy or good. Reading it the obvious way gets
its sign backwards.
"""

from __future__ import annotations

import re

# concept -> plain English. The window phrase gets appended.
BASE = {
    "dkp": "fantasy points scored",
    "dkp_std": "fantasy points, how erratic he is",
    "tgt_share": "share of his team's targets",
    "ay_share": "share of his team's air yards",
    "wopr": "opportunity index (targets + air yards combined)",
    "targets": "targets",
    "targets_per_snap": "targets per snap",
    "carries": "carries",
    "rec": "catches",
    "snap_pct": "share of team snaps played",
    "def_allowed": "fantasy points this defense allows to his position",
    "route_share": "share of his team's pass routes he ran",
    "tprr": "targets per route run",
    "rz_tgt_share": "share of team red-zone targets",
    "ez_tgt_share": "share of team end-zone targets",
    "rz_tgt": "red-zone targets",
    "i10_carry_share": "share of team carries inside the 10-yard line",
    "i5_carry_share": "share of team carries inside the 5-yard line",
    "i10_carries": "carries inside the 10-yard line",
    "value": "points scored above his salary line",
    "ps_dkp": "prior season fantasy points",
    "ps_last4_dkp": "prior season fantasy points, final 4 games",
    "ps_targets": "prior season targets",
    "ps_carries": "prior season carries",
}

WINDOW = {
    "r3": "avg last 3 games",
    "r4": "avg last 4 games",
    "r5": "avg last 5 games",
    "r8": "avg last 8 games",
    "ewm": "recency-weighted avg",
    "lag1": "last game only",
    "last": "most recent game",
    "trend": "trend, recent vs 5-game",
    "pg": "per game",
}

# anything that is not base + window
EXPLICIT = {
    "dkp_blended": "blended baseline: prior-season form + salary expectation",
    "salary": "DraftKings salary",
    "salary_implied_pts": "points his salary implies he should score",
    "salary_pos_pct": "how his salary ranks within his position",
    "salary_chg_1w": "salary change since last week",
    "salary_chg_3w": "salary change over 3 weeks",
    "value_streak": "consecutive games beating his salary line",
    "implied_team_total": "points Vegas expects HIS TEAM to score",
    "implied_opp_total": "points Vegas expects the OPPONENT to score",
    "total_line": "Vegas total points for the game",
    "team_spread": "Vegas spread (negative = his team favored)",
    "spread_x_rb": "Vegas spread, but only for running backs",
    "spread_x_pass": "Vegas spread, but only for receivers and tight ends",
    "is_home": "playing at home",
    "rest": "days of rest since his last game",
    "div_game": "divisional opponent",
    "green_qb": "his QB is inexperienced (5 or fewer career starts)",
    "wind": "wind speed",
    "temp": "temperature",
    "is_indoor": "indoor stadium",
    "vacated_tgt_share": "target share left behind by injured teammates",
    "vacated_carry_share": "carry share left behind by injured teammates",
    "new_vacated_tgt_share": "target share newly vacated this week",
    "vacated_same_group": "vacated opportunity within his own position group",
    "n_out": "how many of his teammates are ruled out",
    "starting_qb_out": "his starting QB is out",
    "own_questionable": "he is listed questionable",
    "own_limited_practice": "he was limited in practice",
    "healthy_tgt_rank": "his target-share rank among HEALTHY teammates",
    "games_so_far": "games he has played this season",
    "ps_games": "games he played last season",
    "ps_tgt_share": "prior season share of team targets",
    "ps_ay_share": "prior season share of team air yards",
    "changed_team": "changed teams since last season",
    "draft_round": "the round he was drafted in",
    "draft_pick": "his overall draft pick number",
    "years_pro": "years in the league",
    "is_rookie": "rookie",
    "age": "age",
    "depth_rank": "where he sits on the depth chart",
    "pos_QB": "is a quarterback",
    "pos_RB": "is a running back",
    "pos_WR": "is a wide receiver",
    "pos_TE": "is a tight end",
}

_SUFFIX = re.compile(r"^(?P<base>.+)_(?P<win>r\d+|ewm|lag\d+|last|trend|pg)$")


def label(name: str) -> str:
    """Plain English for one column. Unknown names come back unchanged."""
    if name in EXPLICIT:
        return EXPLICIT[name]
    if name in BASE:
        return BASE[name]

    m = _SUFFIX.match(name)
    if m and m["base"] in BASE:
        return f"{BASE[m['base']]} -- {WINDOW[m['win']]}"
    if m and m["base"] in EXPLICIT:
        return f"{EXPLICIT[m['base']]} -- {WINDOW[m['win']]}"
    return name


def relabel(idx) -> list[str]:
    """Map a pandas Index / list of column names through `label`."""
    return [label(str(i)) for i in idx]


def missing(names) -> list[str]:
    """Columns with no plain-English name yet -- run after adding features.

    Tests for an actual entry rather than `label(n) == n`, because a few
    genuine labels are identical to their column name ("age", "temp", "wind")
    and a string comparison reports those as missing forever.
    """
    out = []
    for n in names:
        n = str(n)
        if n in EXPLICIT or n in BASE:
            continue
        m = _SUFFIX.match(n)
        if m and (m["base"] in BASE or m["base"] in EXPLICIT):
            continue
        out.append(n)
    return sorted(out)

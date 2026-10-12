"""Sanity checks that run between the model and the Discord post.

    check_pool(pool)         -- is the projection table itself believable?
    check_lineups(lineups)   -- is each solved lineup a legal, plausible DK entry?

Both return a list of Finding. A "block" finding means the lineups must not be
posted: the failure is silent downstream (the optimizer happily solves over
garbage and the post looks normal), so the gate has to live here. A "warn"
finding is posted as a caveat line under the lineups.

Thresholds were set against the real 2026 week-6 run (368 projected rows,
range 1.7-23.5, salary-vs-projection Spearman 0.72) with a wide margin: these
catch a broken pipeline, not a mediocre projection. Judging whether the
projection is GOOD is the backtest's job, not this file's.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

MIN_PROJECTED = 150          # real run: 368. Fewer means the feature join broke.
PROJ_MIN, PROJ_MAX = 0.0, 50.0   # real run: 1.7-23.5. A 50-pt player is a bug.
MIN_SALARY_CORR = 0.35       # real run: 0.72. DK salary alone scores ~0.55 vs actual.
BIG_SALARY = 6000            # unprojected AND healthy at this price = a name-match miss
SALARY_CAP = 50_000
LINEUP_PTS = (60.0, 200.0)   # projected total; the solver's picks run hot, not absurd
OUT_STATUSES = {"O", "OUT", "IR", "D"}

# Legal DK classic roster shape: QB1 RB2-3 WR3-4 TE1-2 DST1 (the extra is the FLEX).
POSITION_COUNTS = {"QB": (1, 1), "RB": (2, 3), "WR": (3, 4), "TE": (1, 2), "DST": (1, 1)}


@dataclass(frozen=True)
class Finding:
    level: str      # "block" | "warn"
    check: str
    message: str


def _block(check: str, msg: str) -> Finding:
    return Finding("block", check, msg)


def _warn(check: str, msg: str) -> Finding:
    return Finding("warn", check, msg)


def check_pool(pool: pd.DataFrame) -> list[Finding]:
    out: list[Finding] = []
    proj = pool["projection"]
    projected = pool[proj.notna()]

    if len(projected) < MIN_PROJECTED:
        out.append(_block("coverage", f"only {len(projected)} players have a projection "
                                      f"(need {MIN_PROJECTED}); the feature join likely broke"))
        return out      # the rest of the checks assume a usable table

    bad = projected[(projected["projection"] < PROJ_MIN) | (projected["projection"] > PROJ_MAX)]
    if len(bad):
        names = ", ".join(f"{r['name']} {r['projection']:.1f}" for _, r in bad.head(3).iterrows())
        out.append(_block("range", f"{len(bad)} projection(s) outside "
                                   f"[{PROJ_MIN:.0f}, {PROJ_MAX:.0f}]: {names}"))

    for col in ("week", "model_version"):
        if col in pool and pool[col].nunique() > 1:
            out.append(_block("mixed-run", f"table mixes {col} values "
                                           f"{sorted(pool[col].astype(str).unique())}"))

    skill = projected[projected["position"] != "DST"]
    corr = skill["salary"].corr(skill["projection"], method="spearman")
    if pd.isna(corr):
        out.append(_block("salary-corr", "salary/projection correlation is undefined "
                                         "(constant projections?)"))
    elif corr < MIN_SALARY_CORR:
        out.append(_block("salary-corr", f"projections barely track salary (Spearman "
                                         f"{corr:.2f}, need {MIN_SALARY_CORR}); looks scrambled"))

    status = pool["status"].fillna("").astype(str)
    missed = pool[proj.isna() & (pool["salary"] >= BIG_SALARY)
                  & ~status.isin(OUT_STATUSES) & (pool["position"] != "DST")]
    if len(missed):
        names = ", ".join(f"{r['name']} (${int(r['salary']):,})" for _, r in missed.head(5).iterrows())
        out.append(_warn("unprojected", f"{len(missed)} healthy player(s) at ${BIG_SALARY:,}+ "
                                        f"have no projection and can't be rostered: {names}"))
    return out


def check_lineups(lineups: list[pd.DataFrame]) -> list[Finding]:
    out: list[Finding] = []
    for n, lu in enumerate(lineups, 1):
        tag = f"lineup {n}"
        if len(lu) != 9:
            out.append(_block("shape", f"{tag} has {len(lu)} players, not 9"))
            continue
        if lu["name"].duplicated().any():
            out.append(_block("duplicate", f"{tag} rosters the same player twice"))
        sal = int(lu["salary"].sum())
        if sal > SALARY_CAP:
            out.append(_block("cap", f"{tag} costs ${sal:,}, over the ${SALARY_CAP:,} cap"))
        counts = lu["position"].value_counts()
        for pos, (lo, hi) in POSITION_COUNTS.items():
            if not lo <= counts.get(pos, 0) <= hi:
                out.append(_block("positions", f"{tag} has {counts.get(pos, 0)} {pos} "
                                               f"(legal: {lo}-{hi})"))
        dead = lu[lu["status"].fillna("").astype(str).isin(OUT_STATUSES)]
        if len(dead):
            out.append(_block("out-player", f"{tag} rosters {', '.join(dead['name'])} "
                                            "who is ruled out"))
        pts = float(lu["projection"].sum())
        if not LINEUP_PTS[0] <= pts <= LINEUP_PTS[1]:
            out.append(_block("lineup-total", f"{tag} projects {pts:.1f} pts, outside "
                                              f"{LINEUP_PTS[0]:.0f}-{LINEUP_PTS[1]:.0f}"))
    return out


def blockers(findings: list[Finding]) -> list[Finding]:
    return [f for f in findings if f.level == "block"]


def warnings(findings: list[Finding]) -> list[Finding]:
    return [f for f in findings if f.level == "warn"]
